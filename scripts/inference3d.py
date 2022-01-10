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

import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from empanada.array_utils import *
from empanada.zarr_utils import *
from empanada.consensus import merge_objects3d
from empanada.config_utils import load_config, load_inference_config
from empanada.inference import engines
from empanada.inference.matcher import SequentialMatcher
from empanada.inference.tracker import InstanceTracker
from empanada.inference import filters

def factor_pad_tensor(tensor, factor=128):
    h, w = tensor.size()[2:]
    pad_bottom = factor - h % factor if h % factor != 0 else 0
    pad_right = factor - w % factor if w % factor != 0 else 0
    if pad_bottom == 0 and pad_right == 0:
        return tensor
    else:
        return nn.ZeroPad2d((0, pad_right, 0, pad_bottom))(tensor)

def parse_args():
    parser = argparse.ArgumentParser(description='Runs empanada model inference.')
    parser.add_argument('model_config', type=str, metavar='model_config', help='Path to a model config yaml file')
    parser.add_argument('infer_config', type=str, metavar='infer_config', help='Path to a inference config yaml file')
    parser.add_argument('volume_path', type=str, metavar='volume_path', help='Path to a Zarr volume')
    parser.add_argument('--use-cpu', action='store_true', metavar='use_cpu', help='Whether to force inference to run on CPU.')
    return parser.parse_args()

def run_forward_matchers(stack, axis, matchers, queue):
    while True:
        fill_index, pan_seg = queue.get()
        if pan_seg is None:
            continue
        elif isinstance(pan_seg, str):
            break
        else:
            for matcher in matchers:
                if matcher.target_seg is None:
                    pan_seg = matcher.initialize_target(pan_seg)
                else:
                    pan_seg = matcher(pan_seg)

            zarr_put3d(stack, fill_index, pan_seg, axis)

if __name__ == "__main__":
    args = parse_args()

    # read the config files
    config = load_config(args.model_config)
    infer_config = load_inference_config(args.infer_config)

    # merge the config files
    config['INFERENCE'] = infer_config

    # validate filter parameters
    filter_names = []
    filter_kwargs = []
    if 'filters' in config['INFERENCE']:
        for f in config['INFERENCE']['filters']:
            assert f['name'] in filters.__dict__
            filter_names.append(f['name'])
            del f['name']
            filter_kwargs.append(f)

    # load the base and render models from file or url
    if os.path.isfile(model_config[f'base_model_{device}']):
        base_model = torch.jit.load(model_config[f'base_model_{device}'])
    else:
        base_model = torch.hub.load_state_dict_from_url(model_config[f'base_model_{device}'])

    if os.path.isfile(model_config[f'render_model_{device}']):
        render_model = torch.jit.load(model_config[f'render_model_{device}'])
    else:
        render_model = torch.hub.load_state_dict_from_url(model_config[f'render_model_{device}'])

    base_model.eval()
    render_model.eval()

    # determine if using cpu or gpu
    device = 'gpu' torch.cuda.is_available() and not args.use_cpu else 'cpu'
    if device == 'gpu':
        base_model = base_model.cuda()
        render_model = render_model.cuda()

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
        size = tuple([s for i,s in enumerate(shape) if i != axis])
        print(f'Priming models for {axis_name} inference...')
        image = torch.randn((1, 1, *size))
        for _ in range(2):
            out = model(image)
            out = pr_model(out['sem_logits'], out['sem_logits'], out['semantic_x'])

        # create the inference engine
        engine = engines.MultiScaleInferenceEngine(
            base_model, render_model, **config['engine_params'], device=device
        )

        # create a separate matcher for each thing class
        matchers = [
            SequentialMatcher(thing_class, label_divisor, **config['INFERENCE']['matcher_params'])
            for thing_class in thing_list
        ]

        queue = mp.Queue()
        matcher_proc = mp.Process(target=run_forward_matchers, args=(stack, axis, matchers, queue))
        matcher_proc.start()

        # make axis-specific dataset
        dataset = ZarrData(volume, axis, eval_tfs)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                pin_memory=True, drop_last=False, num_workers=8)

        fill_index = 0
        for batch in tqdm(dataloader, total=len(dataloader)):
            image = batch['image']
            h, w = image.size()[2:]
            image = factor_pad_tensor(image, config['padding_factor'])

            pan_seg = inference_engine(image)
            if pan_seg is None:
                # building the queue
                queue.put((fill_index, pan_seg))
                continue
            else:
                pan_seg = pan_seg[0]
                pan_seg = pan_seg.squeeze()[:h, :w] # remove padding and unit dimensions
                queue.put((fill_index, pan_seg.cpu().numpy()))
                fill_index += 1

        final_segs = inference_engine.end()
        if final_segs:
            for i, pan_seg in enumerate(final_segs):
                #pan_seg = pan_seg[0]
                pan_seg = pan_seg.squeeze()[:h, :w] # remove padding
                queue.put((fill_index, pan_seg.cpu().numpy()))
                fill_index += 1

        queue.put((fill_index, 'DONE'))
        matcher_proc.join()

        print(f'Propagating labels backward through the stack...')
        # set the matchers to not assign new labels
        # and not split disconnected components

        for matcher in matchers:
            matcher.assign_new = False
            matcher.force_connected = False

        # TODO: multiprocessing the loading with a Queue
        # skip the bottom slice
        rev_indices = np.arange(0, stack.shape[axis])[::-1]
        for rev_idx in tqdm(rev_indices):
            rev_idx = rev_idx.item()
            pan_seg = zarr_take3d(stack, rev_idx, axis)

            for matcher in matchers:
                if matcher.target_seg is None:
                    pan_seg = matcher.initialize_target(pan_seg)
                else:
                    pan_seg = matcher(pan_seg)

            # leave the last slice in the stack alone
            if rev_idx < (stack.shape[axis] - 1):
                zarr_put3d(stack, rev_idx, pan_seg, axis)

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
