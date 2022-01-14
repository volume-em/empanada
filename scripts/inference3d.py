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
from empanada.config_loaders import load_config, load_config_with_base
from empanada.inference import engines
from empanada.inference.matcher import SequentialMatcher
from empanada.inference.tracker import InstanceTracker
from empanada.inference import filters

def parse_args():
    parser = argparse.ArgumentParser(description='Runs empanada model inference.')
    parser.add_argument('config', type=str, metavar='config', help='Path to a model config yaml file')
    parser.add_argument('volume_path', type=str, metavar='volume_path', help='Path to a Zarr volume')
    parser.add_argument('-mode', type=str, dest='mode', metavar='inference_mode', choices=['orthoplane', 'stack'],
                        default='orthoplane', help='Pick orthoplane (xy, xz, yz) or stack (xy)')
    parser.add_argument('-qlen', type=int, dest='qlen', metavar='qlen', choices=[1, 3, 5, 7, 9, 11],
                        default=3, help='Length of median filtering queue, an odd integer')
    parser.add_argument('-nmax', type=int, dest='label_divisor', metavar='label_divisor', choices=['orthoplane', 'stack'],
                        default=20000, help='Maximum number of objects per instance class allowed in volume.')
    parser.add_argument('-seg_thr', type=float, dest='seg_thr', metavar='seg_thr', default=0.3, 
                        help='Segmentation confidence threshold (0-1)')
    parser.add_argument('-nms_thr', type=float, dest='nms_thr', metavar='nms_thr', default=0.1, 
                        help='Centroid confidence threshold (0-1)')
    parser.add_argument('-nms_kernel', type=int, dest='nms_kernel', metavar='nms_kernel', default=3, 
                        help='Minimum allowed distance, in pixels, between object centers')
    parser.add_argument('-iou_thr', type=float, dest='iou_thr', metavar='iou_thr', default=0.25, 
                        help='Minimum IoU score between objects in adjacent slices for label stiching')
    parser.add_argument('-ioa_thr', type=float, dest='ioa_thr', metavar='ioa_thr', default=0.25, 
                        help='Minimum IoA score between objects in adjacent slices for label merging')
    parser.add_argument('-min_size', type=int, dest='min_size', metavar='min_size', default=500, 
                        help='Minimum object size, in voxels, in the final 3d segmentation')
    parser.add_argument('-min_span', type=int, dest='min_span', metavar='min_span', default=4, 
                        help='Minimum number of consecutive slices that object must appear on in final 3d segmentation')
    parser.add_argument('--fine-boundaries', action='store_true', help='Whether to calculate cells on full resolution image.')
    parser.add_argument('--use-cpu', action='store_true', help='Whether to force inference to run on CPU.')
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
    config = load_config(args.config)

    # load the base and render models from file or url
    device = 'gpu' if torch.cuda.is_available() and not args.use_cpu else 'cpu'
    if os.path.isfile(config[f'base_model_{device}']):
        base_model = torch.jit.load(config[f'base_model_{device}'])
    else:
        base_model = torch.hub.load_state_dict_from_url(config[f'base_model_{device}'])

    if os.path.isfile(config[f'render_model_{device}']):
        render_model = torch.jit.load(config[f'render_model_{device}'])
    else:
        render_model = torch.hub.load_state_dict_from_url(config[f'render_model_{device}'])

    if device == 'gpu':
        base_model = base_model.cuda()
        render_model = render_model.cuda()

    # switch the models to eval mode
    base_model.eval()
    render_model.eval()

    # load the zarr volume
    data = zarr.open(args.volume_path, mode='r+')
    volume = data['em']
    shape = volume.shape

    if args.mode == 'orthoplane':
        axes = {'xy': 0, 'xz': 1, 'yz': 2}
    else:
        axes = {'xy': 0}
    
    eval_tfs = A.Compose([
        A.Normalize(**config['norms']),
        ToTensorV2()
    ])

    # create a separate tracker for
    # each prediction axis and each segmentation class
    trackers = {}
    class_labels = config['labels']
    thing_list = config['thing_list']
    label_divisor = args.label_divisor
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
        stack = data.create_dataset(f'panoptic_{axis_name}', shape=shape,
                                    dtype=np.uint64, chunks=chunks,
                                    overwrite=True)

        # prime the model for the given image dimension
        size = tuple([s for i,s in enumerate(shape) if i != axis])
        print(f'Priming models for {axis_name} inference...')
        image = torch.randn((1, 1, *size), device='cuda' if device=='gpu' else 'cpu')
        for _ in range(3):
            out = base_model(image)
            out = render_model(out['sem_logits'], out['sem_logits'], out['semantic_x'])

        # create the inference engine
        inference_engine = engines.MultiScaleInferenceEngine(
            base_model, render_model, 
            thing_list=thing_list,
            median_kernel_size=args.qlen,
            label_divisor=label_divisor,
            nms_threshold=args.nms_thr,
            nms_kernel=args.nms_kernel,
            confidence_thr=args.seg_thr,
            padding_factor=config['padding_factor'],
            coarse_boundaries=not args.fine_boundaries,
            device=device
        )

        # create a separate matcher for each thing class
        matchers = [
            SequentialMatcher(thing_class, label_divisor, merge_iou_thr=args.iou_thr, merge_ioa_thr=args.ioa_thr)
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
            pan_seg = inference_engine(image)

            if pan_seg is None:
                # building the queue
                queue.put((fill_index, pan_seg))
                continue
            else:
                pan_seg = pan_seg.squeeze() # remove padding and unit dimensions
                queue.put((fill_index, pan_seg.cpu().numpy()))
                fill_index += 1

        final_segs = inference_engine.empty_queue()
        if final_segs:
            for i, pan_seg in enumerate(final_segs):
                #pan_seg = pan_seg[0]
                pan_seg = pan_seg.squeeze() # remove padding
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

            # apply filters
            filters.remove_small_objects(tracker, min_size=args.min_size)
            filters.remove_pancakes(tracker, min_span=args.min_span)


    # create the final instance segmentations
    for class_id, class_name in zip(config['labels'], config['class_names']):
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

            # apply filters
            filters.remove_small_objects(consensus_tracker, min_size=args.min_size)
            filters.remove_pancakes(consensus_tracker, min_span=args.min_span)
        else:
            consensus_tracker = class_trackers[0]


        # decode and fill the instances
        consensus_vol = data.create_dataset(
            f'{class_name}_pred', shape=shape, dtype=np.uint64,
            overwrite=True, chunks=(1, None, None)
        )
        zarr_fill_instances(consensus_vol, consensus_tracker.instances)
        consensus_tracker.write_to_json(os.path.join(args.volume_path, f'{class_name}_pred.json'))
