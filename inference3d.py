"""
TODO:
Add confidence prediction
"""
from time import time


import os
import sys
import argparse
import yaml
import zarr
import mlflow
import numpy as np
import torch
import torch.nn as nn
from skimage import io
from skimage import measure
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from mitonet import models
from mitonet.inference import engines
from mitonet.inference.matcher import SequentialMatcher
from mitonet.inference.tracker import InstanceTracker
from mitonet.inference import filters
from mitonet.inference.array_utils import *
from mitonet.zarr_utils import *
from mitonet.aggregation.consensus import merge_objects3d
from mitonet.evaluation import *
from config_utils import load_train_config, load_inference_config

archs = sorted(name for name in models.__dict__
    if callable(models.__dict__[name])
)

def factor_pad_tensor(tensor, factor=128):
    h, w = tensor.size()[2:]
    pad_bottom = factor - h % factor if h % factor != 0 else 0
    pad_right = factor - w % factor if w % factor != 0 else 0
    if pad_bottom == 0 and pad_right == 0:
        return tensor
    else:
        return nn.ZeroPad2d((0, pad_right, 0, pad_bottom))(tensor)

def parse_args():
    parser = argparse.ArgumentParser(description='Runs panoptic deeplab training')
    parser.add_argument('model_config', type=str, metavar='model_config', help='Path to a model config yaml file')
    parser.add_argument('infer_config', type=str, metavar='infer_config', help='Path to a inference config yaml file')
    parser.add_argument('volume_path', type=str, metavar='volume_path', help='Path to a Zarr volume')
    return parser.parse_args()

def snakemake_args():
    params = vars(snakemake.params)
    params['model_config'] = snakemake.input[0]
    params['infer_config'] = snakemake.input[1]
    params['volume_path'] = snakemake.input[2]
    del params['_names']
    
    return argparse.Namespace(**params)

if __name__ == "__main__":
    if 'snakemake' in globals():
        args = snakemake_args()
    else:
        args = parse_args()
    
    config_name = args.model_config.split('/')[-1].split('.yaml')[0]
    
    # read the config files
    config = load_train_config(args.model_config)
    infer_config = load_inference_config(args.infer_config)
        
    # merge the config files
    config['INFERENCE'] = infer_config
    
    volume_path = args.volume_path
    weight_path = os.path.join(config['TRAIN']['model_dir'], f'{config_name}_checkpoint.pth.tar')
        
    # validate parameters
    model_arch = config['MODEL']['arch']
    assert model_arch in archs, f"Unrecognized model architecture {model_arch}."
    filter_names = []
    filter_kwargs = []
    if 'filters' in config['INFERENCE']:
        for f in config['INFERENCE']['filters']:
            assert f['name'] in filters.__dict__
            filter_names.append(f['name'])
            del f['name']
            filter_kwargs.append(f)
            
    # setup model and engine class
    #"""
    model = models.__dict__[model_arch](**config['MODEL'])
    
    if 'queue_len' in config['MODEL']['arch']:
        engine_cls = engines.QueueInferenceEngine
    else:
        engine_cls = engines.MedianInferenceEngine
    #"""
    
    """
    model = torch.jit.load('/data/conradrw/mmm_base_model_quantized.pth')#.cuda()
    pr_model = torch.jit.load('/data/conradrw/point_rend_model.pth')#.cuda()
    engine_cls = engines.MultiScaleInferenceEngine
    
    """
    
    # load model state
    state = torch.load(weight_path, map_location='cpu')
    state_dict = state['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
            
    #"""
    msg = model.load_state_dict(state['state_dict'], strict=True)
    model.to('cuda:0') # move model to GPU 0
    #"""
    
    # set the evaluation transforms
    norms = state['norms']
    gray_channels = 1
    eval_tfs = A.Compose([
        A.Normalize(**norms),
        ToTensorV2()
    ])
    
    # load the zarr volume
    data = zarr.open(volume_path, mode='r+')
    volume = data['em']
    shape = volume.shape

    # TODO: ANISOTROPY OPTIONS
    axes = {'xy': 0, 'xz': 1, 'yz': 2}
    axes = {plane: axes[plane] for plane in config['INFERENCE']['axes']}
    
    # create a separate tracker for 
    # each prediction axis and each segmentation class
    trackers = {}
    class_labels = config['INFERENCE']['labels']
    thing_list = config['INFERENCE']['engine_params']['thing_list']
    label_divisor = config['INFERENCE']['engine_params']['label_divisor']
    for axis_name, axis in axes.items():
        trackers[axis_name] = [
            InstanceTracker(class_id, label_divisor, volume.shape, axis_name) 
            for class_id in class_labels
        ]
    
    for axis_name, axis in axes.items():
        print(f'Predicting {axis_name} stack')
        
        # chunk in axis direction only
        chunks = [None, None, None]
        chunks[axis] = 1
        chunks = tuple(chunks)
        stack = data.create_dataset(f'{config_name}_panoptic_{axis_name}', shape=shape, 
                                    dtype=np.uint64, chunks=chunks, 
                                    overwrite=True)
        
        # prime the model for the given image dimension
        #size = tuple([s for i,s in enumerate(shape) if i != axis])
        #print('Priming models for inference...')
        #image = torch.randn((1, 1, *size))
        #for _ in range(3):
        #    out = model(image)
        #    out = pr_model(out['sem_logits'], out['sem_logits'], out['semantic_x'])
        
        # create the inference engine
        inference_engine = engine_cls(model, **config['INFERENCE']['engine_params'])
        #inference_engine = engine_cls(model, pr_model, **config['INFERENCE']['engine_params'])
        
        # create a separate matcher for each thing class
        matchers = [
            SequentialMatcher(thing_class, label_divisor, **config['INFERENCE']['matcher_params'])
            for thing_class in thing_list
        ]
        
        # make axis-specific dataset
        dataset = ZarrData(volume, axis, eval_tfs)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, 
                                pin_memory=True, drop_last=False, num_workers=8)
        
        fill_index = 0
        for batch in tqdm(dataloader, total=len(dataloader)):
            image = batch['image']
            h, w = image.size()[2:]
            image = factor_pad_tensor(image, 128)
            
            pan_seg = inference_engine(image)
            if pan_seg is None:
                # building the queue
                continue
            
            #pan_seg = pan_seg[0]
            pan_seg = pan_seg.squeeze()[:h, :w] # remove padding and unit dimensions
            pan_seg = pan_seg.cpu().numpy()
            
            # update the panoptic segmentations for each
            # thing class by passing it through matchers
            for matcher in matchers:
                if matcher.target_seg is None:
                    pan_seg = matcher.initialize_target(pan_seg)
                else:
                    pan_seg = matcher(pan_seg)
            
            # store the result
            stack = zarr_put3d(stack, fill_index, pan_seg, axis)
            
            # increment the fill_index
            fill_index += 1
            
            #torch.cuda.empty_cache()  # if needed
            
        # if inference engine has a queue,
        # then there will be a few remaining
        # segmentations to fill in
        final_segs = inference_engine.end()
        if final_segs:
            for i, pan_seg in enumerate(final_segs):
                #pan_seg = pan_seg[0]
                pan_seg = pan_seg.squeeze()[:h, :w] # remove padding
                pan_seg = pan_seg.cpu().numpy()
                
                for matcher in matchers:
                    pan_seg = matcher(pan_seg)
                    
                stack = zarr_put3d(stack, fill_index, pan_seg, axis)
                fill_index += 1
                
        print(f'Propagating labels backward through the stack...')
        # set the matchers to not assign new labels
        # and not split disconnected components
        
        for matcher in matchers:
            matcher.assign_new = False
            matcher.force_connected = False
            
        # TODO: multiprocessing the loading with a Queue
        # skip the bottom slice
        rev_indices = np.arange(0, stack.shape[axis] - 1)[::-1]
        for rev_idx in tqdm(rev_indices):
            rev_idx = rev_idx.item()
            pan_seg = zarr_take3d(stack, rev_idx, axis)
                
            
            for matcher in matchers:
                pan_seg = matcher(pan_seg)

            stack = zarr_put3d(stack, rev_idx, pan_seg, axis)
            
            # track each instance for each class
            for tracker in trackers[axis_name]:
                tracker.update(pan_seg, rev_idx)
            
        # finish tracking
        for tracker in trackers[axis_name]:
            tracker.finish()
                
        # apply any filters
        if filter_names:
            for filt,kwargs in zip(filter_names, filter_kwargs):
                for tracker in trackers[axis_name]:
                    filters.__dict__[filt](tracker, **kwargs)
                    
    # create the final instance segmentations
    for class_id, class_name in zip(config['INFERENCE']['labels'], config['DATASET']['class_names']):
        # get the relevant trackers for the class_label
        print(f'Creating consensus segmentation for class {class_name}...')
        
        class_trackers = []
        for axis_name, axis_trackers in trackers.items():
            for tracker in axis_trackers:
                if tracker.class_id == class_id:
                    class_trackers.append(tracker)
        
        # merge instances from orthoplane inference if applicable
        if len(class_trackers) > 1:
            consensus_tracker = InstanceTracker(class_id, label_divisor, volume.shape, 'xy')
            
            consensus_tracker.instances = merge_objects3d(class_trackers)

            # apply filters to final merged segmentation
            if filter_names:
                for filt,kwargs in zip(filter_names, filter_kwargs):
                    filters.__dict__[filt](consensus_tracker, **kwargs)
        else:
            consensus_tracker = class_trackers[0]
            

        # decode and fill the instances
        consensus_vol = data.create_dataset(
            f'{config_name}_{class_name}_pred', shape=shape, dtype=np.uint64,
            overwrite=True, chunks=(1, None, None)
        )
        zarr_fill_instances(consensus_vol, consensus_tracker.instances)
        
        consensus_tracker.write_to_json(os.path.join(volume_path, f'{config_name}_{class_name}_pred.json'))
    
    # run evaluation
    semantic_metrics = {'IoU': iou}
    instance_metrics = {'F1_50': f1_50, 'F1_75': f1_75, 'Precision_50': precision_50, 
                        'Precision_75': precision_75, 'Recall_50': recall_50, 'Recall_75': recall_75}
    panoptic_metrics = {'PQ': panoptic_quality}
    evaluator = Evaluator(semantic_metrics, instance_metrics, panoptic_metrics)
    
    for class_name in config['DATASET']['class_names']:
        gt_json = os.path.join(volume_path, f'{class_name}_gt.json')
        pred_json = os.path.join(volume_path, f'{config_name}_{class_name}_pred.json')
        results = evaluator(gt_json, pred_json)
        results = {f'{class_name}_{k}': v for k,v in results.items()}
        
        for k, v in results.items():
            print(k, v)
        
        run_id = state.get('run_id')
        if run_id is not None:
            volname = os.path.basename(volume_path).split('.zarr')[0][:20]
            with mlflow.start_run(run_id=run_id) as run:
                for k, v in results.items():
                    mlflow.log_metric(f'{volname}_{k}', v, step=0)