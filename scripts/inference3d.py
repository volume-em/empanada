import os
import argparse
import zarr
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from empanada.data import VolumeDataset
from empanada.inference import engines
from empanada.inference import filters
from empanada.inference.matcher import RLEMatcher
from empanada.inference.tracker import InstanceTracker
from empanada.array_utils import take, put
from empanada.zarr_utils import *
from empanada.consensus import merge_objects_from_trackers
from empanada.config_loaders import load_config
from empanada.inference.rle import pan_seg_to_rle_seg, rle_seg_to_pan_seg

from empanada.evaluation import *

def parse_args():
    parser = argparse.ArgumentParser(description='Runs empanada model inference.')
    parser.add_argument('config', type=str, metavar='config', help='Path to a model config yaml file')
    parser.add_argument('volume_path', type=str, metavar='volume_path', help='Path to a Zarr volume')
    parser.add_argument('-data-key', type=str, metavar='data-key', default='em',
                        help='Key in zarr volume (if volume_path is a zarr). For multiple keys, separate with a comma.')
    parser.add_argument('-mode', type=str, dest='mode', metavar='inference_mode', choices=['orthoplane', 'stack'],
                        default='orthoplane', help='Pick orthoplane (xy, xz, yz) or stack (xy)')
    parser.add_argument('-qlen', type=int, dest='qlen', metavar='qlen', choices=[1, 3, 5, 7, 9, 11],
                        default=3, help='Length of median filtering queue, an odd integer')
    parser.add_argument('-nmax', type=int, dest='label_divisor', metavar='label_divisor', choices=['orthoplane', 'stack'],
                        default=20000, help='Maximum number of objects per instance class allowed in volume.')
    parser.add_argument('-seg-thr', type=float, dest='seg_thr', metavar='seg_thr', default=0.3,
                        help='Segmentation confidence threshold (0-1)')
    parser.add_argument('-nms-thr', type=float, dest='nms_thr', metavar='nms_thr', default=0.1,
                        help='Centroid confidence threshold (0-1)')
    parser.add_argument('-nms-kernel', type=int, dest='nms_kernel', metavar='nms_kernel', default=3,
                        help='Minimum allowed distance, in pixels, between object centers')
    parser.add_argument('-iou-thr', type=float, dest='iou_thr', metavar='iou_thr', default=0.25,
                        help='Minimum IoU score between objects in adjacent slices for label stiching')
    parser.add_argument('-ioa-thr', type=float, dest='ioa_thr', metavar='ioa_thr', default=0.25,
                        help='Minimum IoA score between objects in adjacent slices for label merging')
    parser.add_argument('-min-size', type=int, dest='min_size', metavar='min_size', default=500,
                        help='Minimum object size, in voxels, in the final 3d segmentation')
    parser.add_argument('-min-span', type=int, dest='min_span', metavar='min_span', default=4,
                        help='Minimum number of consecutive slices that object must appear on in final 3d segmentation')
    parser.add_argument('-downsample-f', type=int, dest='downsample_f', metavar='dowsample_f', default=1,
                        help='Factor by which to downsample images before inference, must be log base 2.')
    parser.add_argument('--fine-boundaries', action='store_true', help='Whether to calculate cells on full resolution image.')
    parser.add_argument('--use-cpu', action='store_true', help='Whether to force inference to run on CPU.')
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

    # read the model config file
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

    # load the volume
    if '.zarr' in args.volume_path:
        zarr_store = zarr.open(args.volume_path, mode='r+')
        keys = args.data_key.split(',')
        volume = zarr_store[keys[0]]
        for key in  keys[1:]:
            volume = volume[key]
    elif '.tif' in args.volume_path:
        zarr_store = None
        volume = io.imread(args.volume_path)
    else:
        raise Exception(f'Unable to read file {args.volume_path}. Volume must be .tif or .zarr')

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
            InstanceTracker(class_id, label_divisor, shape, axis_name)
            for class_id in class_labels
        ]

    for axis_name, axis in axes.items():
        print(f'Predicting {axis_name} stack')

        # create placeholder volume for stack results, if desired
        if args.save_panoptic and zarr_store is not None:
            # chunk in axis direction only
            chunks = [None, None, None]
            chunks[axis] = 1
            stack = zarr_store.create_dataset(
                f'panoptic_{axis_name}', shape=shape,
                dtype=np.uint64, chunks=tuple(chunks), overwrite=True
            )
        elif args.save_panoptic:
            stack = np.zeros(shape, dtype=np.uint64)
        else:
            stack = None

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
            RLEMatcher(thing_class, label_divisor, merge_iou_thr=args.iou_thr, merge_ioa_thr=args.ioa_thr)
            for thing_class in thing_list
        ]

        # setup matcher for multiprocessing
        queue = mp.Queue()
        rle_stack = []
        matcher_out, matcher_in = mp.Pipe()
        matcher_proc = mp.Process(target=run_forward_matchers, args=(matchers, queue, rle_stack, matcher_in))
        matcher_proc.start()

        # make axis-specific dataset
        dataset = VolumeDataset(volume, axis, eval_tfs, scale=args.downsample_f)

        num_workers = 8 if zarr_store is not None else 1
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                pin_memory=(device == 'gpu'), drop_last=False, num_workers=num_workers)

        for batch in tqdm(dataloader, total=len(dataloader)):
            image = batch['image']
            size = batch['size']

            # pads and crops image in the engine
            # upsample output by same factor as downsampled input
            pan_seg = inference_engine(image, size, upsampling=args.downsample_f)

            if pan_seg is None:
                # building the median queue
                queue.put(None)
                continue
            else:
                pan_seg = pan_seg.squeeze().cpu().numpy() # remove padding and unit dimensions

                # convert to a compressed rle segmentation
                rle_seg = pan_seg_to_rle_seg(pan_seg, config['labels'], label_divisor, thing_list, force_connected=True)
                queue.put(rle_seg)

        final_segs = inference_engine.empty_queue()
        if final_segs:
            for i, pan_seg in enumerate(final_segs):
                pan_seg = pan_seg.squeeze().cpu().numpy() # remove padding

                # convert to a compressed rle segmentation
                rle_seg = pan_seg_to_rle_seg(pan_seg, config['labels'], label_divisor, thing_list, force_connected=True)
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
        if args.mode == 'orthoplane':
            # empty tracker
            consensus_tracker = InstanceTracker(class_id, label_divisor, shape, 'xy')

            # fill with the consensus instances
            consensus_tracker.instances = merge_objects_from_trackers(class_trackers)

            # apply filters
            filters.remove_small_objects(consensus_tracker, min_size=args.min_size)
            filters.remove_pancakes(consensus_tracker, min_span=args.min_span)
        else:
            consensus_tracker = class_trackers[0]

        # decode and fill the instances
        if zarr_store is not None:
            consensus_vol = zarr_store.create_dataset(
                f'{class_name}_pred', shape=shape, dtype=np.uint64,
                overwrite=True, chunks=(1, None, None)
            )
            zarr_fill_instances(consensus_vol, consensus_tracker.instances)
        else:
            consensus_vol = np.zeros(shape, dtype=np.uint64).reshape(-1)
            for instance_id, instance_attrs in consensus_tracker.instances.items():
                starts = instance_attrs['starts']
                ends = starts + instance_attrs['runs']

                # fill ranges with instance id
                for s,e in zip(starts, ends):
                    consensus_vol[s:e] = instance_id

            consensus_vol = consensus_vol.reshape(shape)
            volpath = os.path.dirname(args.volume_path)
            volname = os.path.basename(args.volume_path).replace('.tif', f'_{class_name}.tif')
            io.imsave(os.path.join(volpath, volname), consensus_vol)

    """
    # run evaluation
    consensus_tracker.write_to_json(os.path.join(args.volume_path, f'{class_name}_pred.json'))
    semantic_metrics = {'IoU': iou}
    instance_metrics = {'F1_50': f1_50, 'F1_75': f1_75, 'Precision_50': precision_50,
                        'Precision_75': precision_75, 'Recall_50': recall_50, 'Recall_75': recall_75}
    panoptic_metrics = {'PQ': panoptic_quality}
    evaluator = Evaluator(semantic_metrics, instance_metrics, panoptic_metrics)

    for class_name in config['class_names']:
        gt_json = os.path.join(args.volume_path, f'{class_name}_gt.json')
        pred_json = os.path.join(args.volume_path, f'{class_name}_pred.json')
        results = evaluator(gt_json, pred_json)
        results = {f'{class_name}_{k}': v for k,v in results.items()}

        for k, v in results.items():
            print(k, v)
    """
