import numpy as np

def f1(
    gt_matched, 
    gt_unmatched,
    pred_matched,
    pred_unmatched,
    matched_ious,
    iou_thr=0.5
):
    # all unmatched gt are fn
    fn = len(gt_unmatched)
    # all unmatched pred are fp
    fp = len(pred_unmatched)
    
    # matches can be tp, fp or fn depending
    # on the iou score
    tp = np.count_nonzero(matched_ious >= iou_thr)
    
    # add 1 fp and 1 fn for every match that fails
    failed_matches = np.count_nonzero(matched_ious < iou_thr)
    fp += failed_matches
    fn += failed_matches

    if tp + fp + fn == 0:
        # by convention, AP is 1 for empty masks
        return 1
    
    return tp / (tp + 0.5 * fp + 0.5 * fn)

def precision(
    gt_matched, 
    gt_unmatched,
    pred_matched,
    pred_unmatched,
    matched_ious,
    iou_thr=0.5
):
    # all unmatched pred are fp
    fp = len(pred_unmatched)
    
    # matches can be tp, fp or fn depending
    # on the iou score
    tp = np.count_nonzero(matched_ious >= iou_thr)
    
    # add 1 fp for every match that failed
    fp += np.count_nonzero(matched_ious < iou_thr)

    if tp + fp == 0:
        # by convention, precision is 1 for empty masks
        return 1
    
    return tp / (tp + fp)

def recall(
    gt_matched, 
    gt_unmatched,
    pred_matched,
    pred_unmatched,
    matched_ious,
    iou_thr=0.5
):
    # all unmatched gt are fn
    fn = len(gt_unmatched)
    
    # matches can be tp, fp or fn depending
    # on the iou score
    tp = np.count_nonzero(matched_ious >= iou_thr)
    
    # add 1 fn for every match that failed
    fn += np.count_nonzero(matched_ious < iou_thr)

    if tp + fn == 0:
        # by convention, recall is 1 for empty masks
        return 1
    
    return tp / (tp + fn)

def f1_50(**kwargs):
    return f1(**kwargs, iou_thr=0.5)

def f1_75(**kwargs):
    return f1(**kwargs, iou_thr=0.75)

def precision_50(**kwargs):
    return precision(**kwargs, iou_thr=0.5)

def precision_75(**kwargs):
    return precision(**kwargs, iou_thr=0.75)

def recall_50(**kwargs):
    return recall(**kwargs, iou_thr=0.5)

def recall_75(**kwargs):
    return recall(**kwargs, iou_thr=0.75)