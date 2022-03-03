import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import measure
from copy import deepcopy
from empanada.array_utils import *
from empanada.inference.rle import unpack_rle_attrs

__all__ = [
    'fast_matcher',
    'rle_matcher',
    'RLEMatcher'
]

def merge_attrs(rle_attr1, rle_attr2):
    r"""Merges instance bounding boxes and run lengths
    """
    # extract labels, boxes, and rle for a given class id
    rle_attr_out = {}
    rle_attr_out['box'] = merge_boxes(rle_attr1['box'], rle_attr2['box'])

    starts, runs = merge_rles(
        rle_attr1['starts'], rle_attr1['runs'],
        rle_attr2['starts'], rle_attr2['runs']
    )
    rle_attr_out['starts'] = starts
    rle_attr_out['runs'] = runs

    return rle_attr_out

def rle_matcher(
    target_instance_rles,
    match_instance_rles,
    iou_thr=0.5,
    return_ioa=False
):
    r"""Performs Hungarian matching on run length encodings.

    Args:
        target_instance_rles: Dictionary of instances to match against. Keys are
        instance 'labels' and values are a dictionary of ('box', 'starts', 'runs').

        match_instance_rles: Dictionary of instances to match. Keys are
        instance 'labels' and values are a dictionary of ('box', 'starts', 'runs')

        iou_thr: Minimum IoU to consider instances a match.

        return_ioa: Whether to return intersection-over-area scores between
        instances in target and match.

    Returns:
        matched_labels: Tuple of Arrays. First array is matched instances in
        target, second item is matched instances in match.

        all_labels: Tuple of Arrays. First array is all instances in
        target, second item is all instances in match.

        matched_ious: Array. IoU scores between all matched instances in
        matched_labels. Same length as matched_labels.

        ioa_matrix: Array of (n, m) pairwise IoA scores between instances
        in target and match. Only returned in return_ioa is True.
    """
    # screen matches by bounding box iou
    # extract bounding boxes and labels for
    # all objects in each instance segmentation
    target_labels, target_boxes, target_starts, target_runs =\
    unpack_rle_attrs(target_instance_rles)

    match_labels, match_boxes, match_starts, match_runs =\
    unpack_rle_attrs(match_instance_rles)

    if len(target_labels) == 0 or len(match_labels) == 0:
        empty = np.array([])
        if return_ioa:
            # no matches, only labels, no matrices
            return (empty, empty), (target_labels, match_labels), empty, empty
        else:
            return (empty, empty), (target_labels, match_labels), empty

    # compute mask IoUs of all possible matches
    iou_matrix = np.zeros((len(target_boxes), len(match_boxes)), dtype='float')
    if return_ioa:
        ioa_matrix = np.zeros((len(target_boxes), len(match_boxes)), dtype=np.float32)

    # match the boxes
    box_matches = np.array(box_iou(target_boxes, match_boxes).nonzero()).T
    for r1, r2 in box_matches:
        iou_matrix[r1, r2] = rle_iou(
            target_starts[r1], target_runs[r1],
            match_starts[r2], match_runs[r2],
        )

        if return_ioa:
            ioa_matrix[r1, r2] = rle_ioa(
                target_starts[r1], target_runs[r1],
                match_starts[r2], match_runs[r2],
            )

    # returns tuple of indices and ious/ioas of instances
    match_rows, match_cols = linear_sum_assignment(iou_matrix, maximize=True)

    # filter out matches with iou less than thr
    if iou_thr is not None:
        iou_mask = iou_matrix[match_rows, match_cols] >= iou_thr
        match_rows = match_rows[iou_mask]
        match_cols = match_cols[iou_mask]

    # convert from indices in matrix to instance labels
    matched_labels = (target_labels[match_rows], match_labels[match_cols])
    all_labels = [target_labels, match_labels]
    matched_ious = iou_matrix[(match_rows, match_cols)]

    if return_ioa:
        return matched_labels, all_labels, matched_ious, ioa_matrix
    else:
        return matched_labels, all_labels, matched_ious

def fast_matcher(
    target_instance_seg,
    match_instance_seg,
    iou_thr=0.5,
    return_ioa=False
):
    r"""Performs Hungarian matching on segmentation masks.

    Args:
        target_instance_seg: Array. Target instance segmentation against
        which to match.

        match_instance_seg: Array. Match instance segmentation from which
        to match instances.

        iou_thr: Minimum IoU to consider instances a match.

        return_ioa: Whether to return intersection-over-area scores between
        instances in target and match.

    Returns:
        matched_labels: Tuple of Arrays. First array is matched instances in
        target, second item is matched instances in match.

        all_labels: Tuple of Arrays. First array is all instances in
        target, second item is all instances in match.

        matched_ious: Array. IoU scores between all matched instances in
        matched_labels. Same length as matched_labels.

        ioa_matrix: Array of (n, m) pairwise IoA scores between instances
        in target and match. Only returned in return_ioa is True.
    """
    # extract bounding boxes and labels for
    # all objects in each instance segmentation
    rps = measure.regionprops(target_instance_seg)
    boxes1 = np.array([rp.bbox for rp in rps])
    labels1 = np.array([rp.label for rp in rps])

    rps = measure.regionprops(match_instance_seg)
    boxes2 = np.array([rp.bbox for rp in rps])
    labels2 = np.array([rp.label for rp in rps])

    if len(labels1) == 0 or len(labels2) == 0:
        empty = np.array([])
        if return_ioa:
            # no matches, only labels, no matrices
            return (empty, empty), (labels1, labels2), empty, empty
        else:
            return (empty, empty), (labels1, labels2), empty

    # create a placeholder for the instance iou matrix
    iou_matrix = np.zeros((len(labels1), len(labels2)), dtype=np.float32)
    if return_ioa:
        ioa_matrix = np.zeros((len(labels1), len(labels2)), dtype=np.float32)

    # only compute mask iou for instances
    # with overlapping bounding boxes
    # (bounding box screening is only for speed)
    box_matches = np.array(box_iou(boxes1, boxes2).nonzero()).T

    for r1, r2 in box_matches:
        l1 = labels1[r1]
        box1 = boxes1[r1]

        l2 = labels2[r2]
        box2 = boxes2[r2]

        box = merge_boxes(box1, box2)
        m1 = crop_and_binarize(target_instance_seg, box, l1)
        m2 = crop_and_binarize(match_instance_seg, box, l2)

        iou = mask_iou(m1, m2)
        iou_matrix[r1, r2] = iou

        if return_ioa:
            ioa_matrix[r1, r2] = mask_ioa(m1, m2)

    # returns tuple of indices and ious/ioas of instances
    match_rows, match_cols = linear_sum_assignment(iou_matrix, maximize=True)

    # filter out matches with iou less than thr
    if iou_thr is not None:
        iou_mask = iou_matrix[match_rows, match_cols] >= iou_thr
        match_rows = match_rows[iou_mask]
        match_cols = match_cols[iou_mask]

    matched_labels = (labels1[match_rows], labels2[match_cols])
    all_labels = [labels1, labels2]
    matched_ious = iou_matrix[(match_rows, match_cols)]

    if return_ioa:
        return matched_labels, all_labels, matched_ious, ioa_matrix
    else:
        return matched_labels, all_labels, matched_ious

class MaskMatcher:
    r"""Tracks and matches instances across consecutive segmentation masks
    in a stack.

    Args:
        class_id: Integer. The class_id in the panoptic segmentation
        that is covered by this matcher.

        label_divisor: Integer. The label divisor used to postprocess
        the panoptic segmentation.

        merge_iou_thr: Minimum IoU to consider instances a match.

        merge_ioa_thr: Minimum IoA to consider an instance as a false split.

        assign_new: Whether to assign unmatched objects a new label in the
        stack.

        force_connected: Whether to enforce that instances within a mask
        be connected components.

    """
    def __init__(
        self,
        class_id,
        label_divisor,
        merge_iou_thr=0.25,
        merge_ioa_thr=0.25,
        assign_new=True,
        force_connected=True,
    ):
        self.class_id = class_id
        self.label_divisor = label_divisor
        self.merge_iou_thr = merge_iou_thr
        self.merge_ioa_thr = merge_ioa_thr
        self.assign_new = assign_new
        self.next_label = (class_id * label_divisor) + 1
        self.target_seg = None
        self.force_connected = force_connected

    @staticmethod
    def _split_components(instance_seg, label_offset=None):
        instance_seg = measure.label(instance_seg).astype(instance_seg.dtype)
        if label_offset is not None:
            instance_seg[instance_seg > 0] += label_offset

        return instance_seg

    def pan_to_instance_seg(self, pan_seg):
        min_id = self.class_id * self.label_divisor
        max_id = min_id + self.label_divisor

         # zero all objects/semantic segs outside of instance_id range
        outside_mask = np.logical_or(pan_seg < min_id, pan_seg >= max_id)
        pan_seg[outside_mask] = 0

        return pan_seg

    def initialize_target(self, target_pan_seg):
        target_pan_seg = np.copy(target_pan_seg) # copy for safety
        min_id = self.class_id * self.label_divisor
        target_instance_seg = self.pan_to_instance_seg(target_pan_seg)

        if self.force_connected:
            target_instance_seg = self._split_components(target_instance_seg, min_id)

            # update panoptic seg with new instance seg labels
            target_rps = measure.regionprops(target_instance_seg)
            for i,rp in enumerate(target_rps):
                target_pan_seg[tuple(rp.coords.T)] = rp.label

        self.target_seg = target_instance_seg
        # only update the next label if target seg had an object
        obj_max = self.target_seg.max()
        if obj_max > 0:
            self.next_label = obj_max + 1

        return target_pan_seg

    def update_target(self, instance_seg):
        self.target_seg = instance_seg

    def __call__(self, match_pan_seg, update_target=True):
        """Matches the given instance segmentation to target"""
        assert self.target_seg is not None, "Initialize target image before running!"

        min_id = self.class_id * self.label_divisor
        match_instance_seg = self.pan_to_instance_seg(np.copy(match_pan_seg)) # copy for safety

        if self.force_connected:
            match_instance_seg = self._split_components(match_instance_seg, min_id)

        # match instances and get iou and ioa matrices
        matched_labels, all_labels, matched_ious, ioa_matrix = fast_matcher(
            self.target_seg, match_instance_seg, self.merge_iou_thr, return_ioa=True
        )

        # matched instance labels with target
        target_labels = all_labels[0]
        match_labels = all_labels[1]
        label_matches = {ml: tl for tl,ml in zip(matched_labels[0], matched_labels[1])}

        match_rps = measure.regionprops(match_instance_seg)
        for i,rp in enumerate(match_rps):
            if rp.label in label_matches:
                # use label from target
                new_label = label_matches[rp.label]
            else:
                # lookup the IoA (we can use the index of the rp
                # because it was also used by the fast_matcher function)
                assert rp.label == match_labels[i]
                ioa_max = ioa_matrix[:, i].max() if len(ioa_matrix) > 0 else 0

                if ioa_max >= self.merge_ioa_thr:
                    new_label = target_labels[ioa_matrix[:, i].argmax()]
                elif self.assign_new:
                    # assign instance to next available label and increment
                    new_label = self.next_label

                    self.next_label += 1
                else:
                    # keep the existing label
                    #print('kept label', rp.label)
                    new_label = rp.label

            # update all pixels instance and panoptic
            # segmentation to new label
            match_instance_seg[tuple(rp.coords.T)] = new_label
            match_pan_seg[tuple(rp.coords.T)] = new_label

        # make matched instance segmentation the next target
        if update_target:
            self.update_target(match_instance_seg)

        # return the matched panoptic segmentation
        return match_pan_seg

class RLEMatcher:
    r"""Tracks and matches instances across consecutive run length encodings
    in a stack.

    Args:
        class_id: Integer. The class_id in the panoptic segmentation
        that is covered by this matcher.

        label_divisor: Integer. The label divisor used to postprocess
        the panoptic segmentation.

        merge_iou_thr: Minimum IoU to consider instances a match.

        merge_ioa_thr: Minimum IoA to consider an instance as a false split.

        assign_new: Whether to assign unmatched objects a new label in the
        stack.

    """
    def __init__(
        self,
        class_id,
        label_divisor,
        merge_iou_thr=0.25,
        merge_ioa_thr=0.25,
        assign_new=True,
        **kwargs
    ):
        self.class_id = class_id
        self.label_divisor = label_divisor
        self.merge_iou_thr = merge_iou_thr
        self.merge_ioa_thr = merge_ioa_thr
        self.assign_new = assign_new
        self.next_label = (class_id * label_divisor) + 1
        self.target_rle = None

    def initialize_target(self, target_instance_rles):
        self.target_rle = target_instance_rles
        # only update the next label if target seg has an object
        objs = list(target_instance_rles.keys())
        if len(objs) > 0:
            self.next_label = max(objs) + 1

    def update_target(self, instance_rles):
        self.target_rle = instance_rles

    def __call__(self, match_instance_rle, update_target=True):
        """Matches the given instance segmentation to target"""
        assert self.target_rle is not None, "Initialize target rle before running!"

        # match instances and get iou and ioa matrices
        matched_labels, all_labels, matched_ious, ioa_matrix = rle_matcher(
            self.target_rle, match_instance_rle, self.merge_iou_thr, return_ioa=True
        )

        # matched instance labels with target
        target_labels = all_labels[0]
        match_labels = all_labels[1]
        label_matches = {ml: tl for tl,ml in zip(matched_labels[0], matched_labels[1])}

        matched_rles = {}
        for i,(ml, mattrs) in enumerate(match_instance_rle.items()):
            if ml in label_matches:
                # use label from target
                new_label = label_matches[ml]
            else:
                # lookup the IoA (we can use the index of the rp
                # because it was also used by the fast_matcher function)
                assert ml == match_labels[i]
                ioa_max = ioa_matrix[:, i].max() if len(ioa_matrix) > 0 else 0

                if ioa_max >= self.merge_ioa_thr:
                    new_label = target_labels[ioa_matrix[:, i].argmax()]
                elif self.assign_new:
                    # assign instance to next available label and increment
                    new_label = self.next_label
                    self.next_label += 1
                else:
                    # keep the existing label
                    new_label = ml

            # add instance with new label
            if new_label not in matched_rles:
                matched_rles[new_label] = mattrs
            else:
                matched_rles[new_label] = merge_attrs(matched_rles[new_label], mattrs)

        # make matched instance rles the next target
        if update_target:
            self.update_target(matched_rles)

        # return the matched panoptic segmentation
        return matched_rles
