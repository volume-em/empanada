import json
import numpy as np
from empanada.array_utils import *

__all__ = ['Evaluator']

def rle_matcher(
    gt_labels,
    gt_boxes, 
    gt_encodings,
    pred_labels,
    pred_boxes, 
    pred_encodings,
    return_iou_matrix=True
):
    # screen matches by bounding box iou
    box_matches = np.array(box_iou3d(gt_boxes, pred_boxes).nonzero()).T
    
    # compute mask IoUs of all possible matches
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)), dtype='float')
    gt_matched = []
    pred_matched = []
    for r1, r2 in zip(*tuple(box_matches.T)):
        # TODO: convert from string to rle before entering this function
        r1_starts, r1_runs = string_to_rle(gt_encodings[r1])
        r2_starts, r2_runs = string_to_rle(pred_encodings[r2])
        pair_iou = rle_iou(r1_starts, r1_runs, r2_starts, r2_runs)
        
        iou_matrix[r1, r2] = pair_iou
        if pair_iou > 0.5:
            gt_matched.append(r1)
            pred_matched.append(r2)
            
    matched_ious = iou_matrix[gt_matched, pred_matched]
    gt_matched = gt_labels[gt_matched]
    pred_matched = pred_labels[pred_matched]
            
    if return_iou_matrix:
        return gt_matched, pred_matched, matched_ious, iou_matrix
    else:
        return gt_matched, pred_matched, matched_ious

class Evaluator:
    def __init__(
        self,
        semantic_metrics=None, 
        instance_metrics=None, 
        panoptic_metrics=None
    ):
        self.semantic_metrics = semantic_metrics
        self.instance_metrics = instance_metrics
        self.panoptic_metrics = panoptic_metrics
        
    @staticmethod
    def _unpack_instance_dict(instance_dict):
        labels = []
        boxes = []
        encodings = []
        for k in instance_dict.keys():
            labels.append(int(k))
            boxes.append(instance_dict[k]['box'])
            encodings.append(instance_dict[k]['rle'])
            
        return np.array(labels), np.array(boxes), encodings
        
    def __call__(self, gt_json_fpath, pred_json_fpath, return_instances=False):
        # load the json files for each
        with open(gt_json_fpath, mode='r') as f:
            gt_json = json.load(f)
            
        with open(pred_json_fpath, mode='r') as f:
            pred_json = json.load(f)
            
        assert (gt_json['class_id'] == pred_json['class_id']), \
        "Prediction and ground truth classes must match!"
        
        gt_labels, gt_boxes, gt_encodings = self._unpack_instance_dict(gt_json['instances'])
        pred_labels, pred_boxes, pred_encodings = self._unpack_instance_dict(pred_json['instances'])
        
        semantic_results = {}
        instance_results = {}
        panoptic_results = {}
        
        if self.semantic_metrics is not None:
            # decode and concatenate all gt and pred encodings
            # N.B. This will break badly for dense semantic classes!
            gt_indices = np.concatenate([np.stack(string_to_rle(enc), axis=1) for enc in gt_encodings])
            pred_indices = np.concatenate([np.stack(string_to_rle(enc), axis=1) for enc in pred_encodings])
            
            # calculate semantic metrics
            semantic_results = {name: func(gt_indices, pred_indices) for name,func in self.semantic_metrics.items()}
        
        if self.instance_metrics is not None or self.panoptic_metrics is not None:
            # match instances
            gt_matched, pred_matched, matched_ious, iou_matrix = \
            rle_matcher(gt_labels, gt_boxes, gt_encodings, 
                        pred_labels, pred_boxes, pred_encodings, 
                        return_iou_matrix=True)
            
            # determine unmatched instance ids
            gt_unmatched = np.setdiff1d(gt_labels, gt_matched)
            pred_unmatched = np.setdiff1d(pred_labels, pred_matched)
            
            kwargs = {
                'gt_matched': gt_matched, 
                'pred_matched': pred_matched,
                'gt_unmatched': gt_unmatched,
                'pred_unmatched': pred_unmatched,
                'matched_ious': matched_ious
            }
            
            if self.instance_metrics is not None:
                instance_results = {name: func(**kwargs) for name,func in self.instance_metrics.items()}
            
            if self.panoptic_metrics is not None:
                panoptic_results = {name: func(**kwargs) for name,func in self.panoptic_metrics.items()}
            
        # unpack all into 1 dictionary
        results_dict = {**semantic_results, **instance_results, **panoptic_results}
        if return_instances:
            instances_dict = {
                'gt_matched': gt_matched, 'pred_matched': pred_matched,
                'gt_unmatched': gt_unmatched, 'pred_unmatched': pred_unmatched,
                'iou_matrix': iou_matrix
            }
            return results_dict, instances_dict
        else:
            return results_dict
