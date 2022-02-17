import os
import argparse
import zarr
import mlflow
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from empanada.data import VolumeDataset
from empanada import models
from empanada.inference import engines
from empanada.inference import filters
from empanada.inference.matcher import RLEMatcher
from empanada.inference.tracker import InstanceTracker
from empanada.inference.postprocess import factor_pad
from empanada.array_utils import *
from empanada.zarr_utils import *
from empanada.evaluation import *
from empanada.consensus import merge_objects_from_trackers
from empanada.config_loaders import load_config_with_base
from empanada.inference.rle import pan_seg_to_rle_seg, rle_seg_to_pan_seg

archs = sorted(name for name in models.__dict__
    if callable(models.__dict__[name])
)

def parse_args():
    parser = argparse.ArgumentParser(description='Runs panoptic deeplab training')
    parser.add_argument('model_config', type=str, metavar='model_config', help='Path to a model config yaml file')
    parser.add_argument('infer_config', type=str, metavar='infer_config', help='Path to a inference config yaml file')
    parser.add_argument('volume_path', type=str, metavar='volume_path', help='Path to a Zarr volume')
    parser.add_argument('--save-panoptic', action='store_true', help='Whether to save raw panoptic segmentation for each stack.')
    return parser.parse_args()

def run_forward_matchers(
    matchers,
    queue,
    rle_stack,
    matcher_in,
    end_signal='finish'
):
    """
    Run forward matching of instances between slices in a separate process
    on CPU while model is performing inference on GPU.
    """
    # go until queue gets the kill signal
    while True:
        rle_seg = queue.get()

        if rle_seg is None:
            # building the median filter queue
            continue
        elif rle_seg == end_signal:
            # all images have been matched!
            break
        else:
            # match the rle seg for each class
            for matcher in matchers:
                class_id = matcher.class_id
                if matcher.target_rle is None:
                    matcher.initialize_target(rle_seg[class_id])
                else:
                    rle_seg[class_id] = matcher(rle_seg[class_id])

            rle_stack.append(rle_seg)

    matcher_in.send([rle_stack])
    matcher_in.close()

if __name__ == "__main__":
    args = parse_args()
    config_name = args.model_config.split('/')[-1].split('.yaml')[0]

    # read the config files
    config = load_config_with_base(args.model_config)
    infer_config = load_config_with_base(args.infer_config)

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
    model = models.__dict__[model_arch](**config['MODEL'])
    engine_cls = engines.MedianInferenceEngine

    # load model state
    state = torch.load(weight_path, map_location='cpu')
    state_dict = state['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    msg = model.load_state_dict(state['state_dict'], strict=True)
    model.to('cuda' if torch.cuda.device_count() > 0 else 'cpu') # move model to GPU 0

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
            InstanceTracker(class_id, label_divisor, shape, axis_name)
            for class_id in class_labels
        ]

    for axis_name, axis in axes.items():
        print(f'Predicting {axis_name} stack')

        # create placeholder volume for stack results, if desired
        if args.save_panoptic:
            # chunk in axis direction only
            chunks = [None, None, None]
            chunks[axis] = 1
            chunks = tuple(chunks)
            stack = data.create_dataset(f'{config_name}_panoptic_{axis_name}', shape=shape,
                                        dtype=np.uint64, chunks=chunks,
                                        overwrite=True)

        # create the inference engine
        inference_engine = engine_cls(model, **config['INFERENCE']['engine_params'])

        # create a separate matcher for each thing class
        matchers = [
            RLEMatcher(thing_class, label_divisor, **config['INFERENCE']['matcher_params'])
            for thing_class in thing_list
        ]

        # setup matcher for multiprocessing
        queue = mp.Queue()
        rle_stack = []
        matcher_out, matcher_in = mp.Pipe()
        matcher_proc = mp.Process(target=run_forward_matchers, args=(matchers, queue, rle_stack, matcher_in))
        matcher_proc.start()

        # make axis-specific dataset
        dataset = VolumeDataset(volume, axis, eval_tfs, scale=1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                pin_memory=True, drop_last=False, num_workers=8)

        for batch in tqdm(dataloader, total=len(dataloader)):
            image = batch['image']
            h, w = image.size()[-2:]

            image = factor_pad(image, 128)
            pan_seg = inference_engine(image)

            if pan_seg is None:
                # building the queue
                queue.put(None)
                continue
            else:
                pan_seg = pan_seg.squeeze()[:h, :w].cpu().numpy() # remove padding and unit dimensions

                # convert to a compressed rle segmentation
                rle_seg = pan_seg_to_rle_seg(pan_seg, config['INFERENCE']['labels'], label_divisor, thing_list, force_connected=True)
                queue.put(rle_seg)

        final_segs = inference_engine.end()
        if final_segs:
            for i, pan_seg in enumerate(final_segs):
                pan_seg = pan_seg.squeeze()[:h, :w].cpu().numpy() # remove padding

                # convert to a compressed rle segmentation
                rle_seg = pan_seg_to_rle_seg(pan_seg, config['INFERENCE']['labels'], label_divisor, thing_list, force_connected=True)
                queue.put(rle_seg)

        # finish and close forward matching process
        queue.put('finish')
        rle_stack = matcher_out.recv()[0]
        matcher_proc.join()

        print(f'Propagating labels backward through the stack...')

        # set the matchers to not assign new labels
        for matcher in matchers:
            matcher.target_rle = None
            matcher.assign_new = False

        rev_indices = np.arange(0, shape[axis])[::-1]
        for rev_idx in tqdm(rev_indices):
            rev_idx = rev_idx.item()
            rle_seg = rle_stack[rev_idx]

            for matcher in matchers:
                class_id = matcher.class_id
                if matcher.target_rle is None:
                    matcher.initialize_target(rle_seg[class_id])
                else:
                    rle_seg[class_id] = matcher(rle_seg[class_id])

            # store the panoptic seg if desired
            if args.save_panoptic:
                shape2d = tuple([s for i,s in enumerate(shape) if i != axis])
                pan_seg = rle_seg_to_pan_seg(rle_seg, shape2d)
                put(stack, rev_idx, pan_seg, axis)

            # track each instance for each class
            for tracker in trackers[axis_name]:
                class_id = tracker.class_id
                tracker.update(rle_seg[class_id], rev_idx)

        # finish tracking
        for tracker in trackers[axis_name]:
            tracker.finish()

        # apply any filters
        if filter_names:
            for filt,kwargs in zip(filter_names, filter_kwargs):
                for tracker in trackers[axis_name]:
                    filters.__dict__[filt](tracker, **kwargs)
                    
    import pickle
    with open('salgland_trackers.pkl', mode='wb') as handle:
        pickle.dump(trackers, handle)
                    
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
            # empty tracker
            consensus_tracker = InstanceTracker(class_id, label_divisor, shape, 'xy')

            # fill with the consensus instances
            consensus_tracker.instances = merge_objects_from_trackers(class_trackers, **config['INFERENCE']['consensus_params'])

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

        try:
            run_id = state.get('run_id')
            if run_id is not None:
                volname = os.path.basename(volume_path).split('.zarr')[0][:20]
                with mlflow.start_run(run_id=run_id) as run:
                    for k, v in results.items():
                        mlflow.log_metric(f'{volname}_{k}', v, step=0)
        except:
            print('Results not stored in MLFlow, script was run in the wrong directory.')
