import numpy as np
import torch
from tqdm import tqdm
from empanada.array_utils import put
from empanada.inference import filters
from empanada.inference.engines import _MedianQueue
from empanada.inference.tracker import InstanceTracker
from empanada.inference.postprocess import merge_semantic_and_instance
from empanada.inference.rle import pan_seg_to_rle_seg, rle_seg_to_pan_seg
from empanada.inference.consensus import merge_objects_from_trackers, merge_semantic_from_trackers


#----------------------------------------------------------
# Utilities for MultiGPU inference
#----------------------------------------------------------

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def all_gather(tensor, group=None):
    # all tensors are same size
    world_size = dist.get_world_size()

    # receiving Tensor from all ranks
    tensor_list = [
        torch.zeros_like(tensor) for _ in range(world_size)
    ]

    dist.all_gather(tensor_list, tensor, group=group)

    return tensor_list

def harden_seg(sem, confidence_thr):
    if sem.size(1) > 1: # multiclass segmentation
        sem = torch.argmax(sem, dim=1, keepdim=True)
    else:
        sem = (sem >= confidence_thr).long() # need integers not bool

    return sem

def get_panoptic_seg(
    sem,
    instance_cells,
    label_divisor,
    thing_list,
    stuff_area=32,
    void_label=0
):
    # keep only label for instance classes
    instance_seg = torch.zeros_like(sem)
    for thing_class in thing_list:
        instance_seg[sem == thing_class] = 1

    # map object ids
    instance_seg = (instance_seg * instance_cells).long()

    pan_seg = merge_semantic_and_instance(
        sem, instance_seg, label_divisor, thing_list,
        stuff_area, void_label
    )

    return pan_seg

def apply_matchers(rle_seg, matchers):
    for matcher in matchers:
        class_id = matcher.class_id
        if matcher.target_rle is None:
            matcher.initialize_target(rle_seg[class_id])
        else:
            rle_seg[class_id] = matcher(rle_seg[class_id])

    return rle_seg

def run_forward_matchers(
    matchers,
    queue,
    rle_stack,
    matcher_in,
    labels,
    label_divisor,
    thing_list,
    end_signal='finish'
):
    r"""
    Run forward matching of instances between slices in a separate process
    on CPU while model is performing inference on GPU.
    """
    # go until queue gets the kill signal
    while True:
        # create the rle_seg
        pan_seg = queue.get()
        pan_seg = pan_seg.squeeze().cpu().numpy()
        rle_seg = pan_seg_to_rle_seg(
            pan_seg, labels, label_divisor, thing_list, force_connected=True
        )

        if rle_seg is None:
            # building the median filter queue
            continue
        elif rle_seg == end_signal:
            # all images have been matched!
            break
        else:
            rle_seg = apply_matchers(rle_seg, matchers)
            rle_stack.append(rle_seg)

    matcher_in.send([rle_stack])
    matcher_in.close()

def run_backward_matchers(
    rle_stack,
    matchers,
    axis_len
):
    # set the matchers to not assign new labels
    for matcher in matchers:
        matcher.target_rle = None
        matcher.assign_new = False

    rev_indices = np.arange(0, axis_len)[::-1]
    for rev_idx in tqdm(rev_indices):
        rev_idx = rev_idx.item()
        rle_seg = rle_stack[rev_idx]
        rle_seg = apply_matchers(rle_seg, matchers)

        yield rev_idx, rle_seg

def update_trackers(
    rle_seg,
    index,
    trackers,
    axis=None,
    stack=None
):
    assert not (axis is None and stack is not None), \
    'Storing segs in stack requires an axis!'

    # store the panoptic seg if needed
    if stack is not None:
        shape2d = tuple([s for i,s in enumerate(shape) if i != axis])
        pan_seg = rle_seg_to_pan_seg(rle_seg, shape2d)
        put(stack, index, pan_seg, axis)

    # track each instance for each class
    for tracker in trackers:
        class_id = tracker.class_id
        tracker.update(rle_seg[class_id], index)

def finish_tracking(trackers):
    for tracker in trackers:
        tracker.finish()

def apply_filters(
    tracker,
    filters_dict
):
    if filters_dict is not None:
        for filt in filters_dict:
            name = filt['name']
            kwargs = {k: v for k,v in filters_dict.items() if k != 'name'}

            # applied in-place
            filters.__dict__[filt](tracker, **kwargs)

def get_axis_trackers_by_class(trackers, class_id):
    class_trackers = []
    for axis_name, axis_trackers in trackers.items():
        for tracker in axis_trackers:
            if tracker.class_id == class_id:
                class_trackers.append(tracker)

    return class_trackers

def create_instance_consensus(
    class_trackers,
    pixel_vote_thr=2,
    cluster_iou_thr=0.75,
    bypass=False
):
    class_id = class_trackers[0].class_id
    label_divisor = class_trackers[0].label_divisor
    shape = class_trackers[0].shape3d

    consensus_tracker = InstanceTracker(class_id, label_divisor, shape, 'xy')
    consensus_tracker.instances = merge_objects_from_trackers(
        class_trackers, pixel_vote_thr, cluster_iou_thr, bypass
    )

    return consensus_tracker

def create_semantic_consensus(
    class_trackers,
    pixel_vote_thr=2
):
    class_id = class_trackers[0].class_id
    label_divisor = class_trackers[0].label_divisor
    shape = class_trackers[0].shape3d

    consensus_tracker = InstanceTracker(class_id, label_divisor, shape, 'xy')
    consensus_tracker.instances = merge_semantic_from_trackers(class_trackers, pixel_vote_thr)

    return consensus_tracker

def run_forward_matchers_mgpu(
    matchers,
    queue,
    rle_stack,
    matcher_in,
    confidence_thr,
    median_kernel_size,
    labels,
    label_divisor,
    thing_list,
    stuff_area=32,
    void_label=0,
    end_signal='finish',
):
    r"""Run forward matching of instances between slices in a separate process
    on CPU while model is performing inference on GPU.
    """
    # create the queue for sem and instance cells
    median_queue = _MedianQueue(median_kernel_size)

    while True:
        sem, cells = queue.get()
        if sem == end_signal:
            # all images have been matched!
            break

        # update the queue
        median_queue.enqueue({'sem': sem, 'cells': cells})
        median_out = median_queue.get_next(keys=['sem'])
        median_sem, cells = median_out['sem'], median_out['cells']

        # get segmentation if not None
        if median_sem is not None:
            median_sem = harden_seg(median_sem, confidence_thr)
            pan_seg = get_panoptic_seg(
                median_sem, cells, label_divisor,
                thing_list, stuff_area, void_label
            )
        else:
            pan_seg = None
            continue

        # convert pan seg to rle
        pan_seg = pan_seg.squeeze().cpu().numpy()
        rle_seg = pan_seg_to_rle_seg(
            pan_seg, labels, label_divisor, thing_list, force_connected=True
        )

        # match the rle seg for each class
        rle_seg = apply_matchers(rle_seg, matchers)
        rle_stack.append(rle_seg)

    # get the final segmentations from the queue
    for qout in median_queue.end():
        sem, cells = qout['sem'], qout['cells']
        sem = harden_seg(sem, confidence_thr)
        pan_seg = get_panoptic_seg(
            sem, cells, label_divisor,
            thing_list, stuff_area, void_label
        )

        pan_seg = pan_seg.squeeze().cpu().numpy()
        rle_seg = pan_seg_to_rle_seg(
            pan_seg, labels, label_divisor, thing_list, force_connected=True
        )

        # match the rle seg for each class
        rle_seg = apply_matchers(rle_seg, matchers)
        rle_stack.append(rle_seg)

    matcher_in.send([rle_stack])
    matcher_in.close()
