import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import measure
from empanada.array_utils import *

__all__ = [
    'fast_matcher',
    'SequentialMatcher'
]

# TODO create 1 matcher that will work for all classes to save computation

def fast_matcher(
    target_instance_seg, 
    match_instance_seg,
    iou_thr=0.5,
    return_ioa=False
):
    """
    Does this support multiclass instance segmentation? No. Should perform this per class.
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

class SequentialMatcher:
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
        self.next_label = label_divisor + 1
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
