import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from empanada.inference.matcher import fast_matcher

class EMAMeter:
    """Computes and stores an exponential moving average and current value"""
    def __init__(self, momentum=0.98):
        self.mom = momentum
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum = (self.sum * self.mom) + (val * (1 - self.mom))
        self.count += 1
        self.avg = self.sum / (1 - self.mom ** (self.count))
        
class AverageMeter:
    """Computes average values"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum = self.sum + self.val
        self.count += 1
        self.avg = self.sum / self.count

class _BaseMetric:
    def __init__(self, meter):
        self.meter = meter
        
    def update(self, value):
        self.meter.update(value)
        
    def reset(self):
        self.meter.reset()
        
    def average(self):
        return self.meter.avg

class Accuracy(_BaseMetric):
    def __init__(self, meter, topk=1, **kwargs):
        super().__init__(meter)
        self.topk = topk
    
    def calculate(self, output, target):
        # apply softmax to output confidences
        output = F.softmax(output['conf'], dim=1) # (N, C)
        target = target['conf'] # (N, 1)
        batch_size = target.size(0)

        _, pred = output.topk(self.topk, 1, True, True) # (N, k)
        pred = pred.t() # (k, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # (k, N)
        correct_k = correct.reshape(-1).float().sum(0, keepdim=True) # (N * k,) -> (1,)
        correct_k.mul_(100.0 / batch_size)
            
        return correct_k
        
class IoU(_BaseMetric):
    def __init__(self, meter, output_key='sem_logits', target_key='sem', **kwargs):
        super().__init__(meter)
        self.output_key = output_key
        self.target_key = target_key
        
    def calculate(self, output, target):
        # only require the semantic segmentation
        output = output[self.output_key]
        target = target[self.target_key]
        
        # make target the same shape as output by unsqueezing
        # the channel dimension, if needed
        if target.ndim == output.ndim - 1:
            target = target.unsqueeze(1)
        
        # get the number of classes from the output channels
        n_classes = output.size(1)
        
        # get reshape size based on number of dimensions
        # can exclude first 2 dims, which are always batch and channel
        empty_dims = (1,) * (target.ndim - 2)
        
        if n_classes > 1:
            # one-hot encode the target (B, 1, H, W) --> (B, N, H, W)
            k = torch.arange(0, n_classes).view(1, n_classes, *empty_dims).to(target.device)
            target = (target == k)
            
            # softmax the output
            output = nn.Softmax(dim=1)(output)
        else:
            # sigmoid the output
            output = (torch.sigmoid(output) > 0.5).long()
            
        # cast target to the correct type for operations
        target = target.type(output.dtype)
        
        # multiply the tensors, everything that is still as 1 is part of the intersection
        # (N,)
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersect = torch.sum(output * target, dims)
        
        # compute the union, (N,)
        union = torch.sum(output + target, dims) - intersect
        
        # avoid division errors by adding a small epsilon
        iou = (intersect + 1e-7) / (union + 1e-7)
        
        return iou
    
class PQ(_BaseMetric):
    def __init__(
        self, 
        meter, 
        labels,
        label_divisor,
        **kwargs
    ):
        super().__init__(meter)
        self.labels = labels
        self.label_divisor = label_divisor
        
    def _to_class_seg(self, pan_seg, label):
        instance_seg = np.copy(pan_seg) # copy for safety
        min_id = label * self.label_divisor
        max_id = min_id + self.label_divisor
        
        # zero all objects/semantic segs outside of instance_id range
        outside_mask = np.logical_or(instance_seg < min_id, instance_seg >= max_id)
        instance_seg[outside_mask] = 0
        return instance_seg
        
    def calculate(self, output, target):
        # convert tensors to numpy
        output = output['pan_seg'].squeeze().long().cpu().numpy()
        target = target['pan_seg'].squeeze().long().cpu().numpy()
        
        # compute the panoptic quality, per class
        per_class_results = []
        for label in self.labels:
            pred_class_seg = self._to_class_seg(output, label)
            tgt_class_seg = self._to_class_seg(target, label)
            
            # match the segmentations
            matched_labels, all_labels, matched_ious = \
            fast_matcher(tgt_class_seg, pred_class_seg, iou_thr=0.5)
            
            tp = len(matched_labels[0])
            fn = len(np.setdiff1d(all_labels[0], matched_labels[0]))
            fp = len(np.setdiff1d(all_labels[1], matched_labels[1]))
            
            if tp + fp + fn == 0:
                # by convention, PQ is 1 for empty masks
                per_class_results.append(1.)
                continue
            
            sq = matched_ious.sum() / (tp + 1e-5)
            rq = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results.append(sq * rq)

        return np.array(per_class_results)
        
class F1(_BaseMetric):
    def __init__(
        self, 
        meter, 
        labels,
        label_divisor,
        iou_thr=0.5,
        **kwargs
    ):
        super().__init__(meter)
        self.labels = labels
        self.label_divisor = label_divisor
        self.iou_thr = iou_thr
        
    def _to_class_seg(self, pan_seg, label):
        instance_seg = np.copy(pan_seg) # copy for safety
        min_id = label * self.label_divisor
        max_id = min_id + self.label_divisor
        
        # zero all objects/semantic segs outside of instance_id range
        outside_mask = np.logical_or(instance_seg < min_id, instance_seg >= max_id)
        instance_seg[outside_mask] = 0
        return instance_seg
        
    def calculate(self, output, target):
        # convert tensors to numpy
        output = output['pan_seg'].squeeze().long().cpu().numpy()
        target = target['pan_seg'].squeeze().long().cpu().numpy()
        
        # compute the panoptic quality, per class
        per_class_results = []
        for label in self.labels:
            pred_class_seg = self._to_class_seg(output, label)
            tgt_class_seg = self._to_class_seg(target, label)
            
            # match the segmentations
            matched_labels, all_labels, matched_ious = \
            fast_matcher(tgt_class_seg, pred_class_seg, iou_thr=self.iou_thr)
            
            tp = len(matched_labels[0])
            fn = len(np.setdiff1d(all_labels[0], matched_labels[0]))
            fp = len(np.setdiff1d(all_labels[1], matched_labels[1]))
            
            if tp + fp + fn == 0:
                # by convention, F1 is 1 for empty masks
                per_class_results.append(1.)
            else:
                f1 = tp / (tp + 0.5 * fn + 0.5 * fp)
                per_class_results.append(f1)
            
        return np.array(per_class_results)

class ComposeMetrics:
    def __init__(
        self, 
        metrics_dict, 
        class_names, 
        reset_on_print=True
    ):
        self.metrics_dict = metrics_dict
        self.class_names = class_names
        self.reset_on_print = reset_on_print
        self.history = {}
        
    def evaluate(self, output, target):
        # calculate all the metrics in the dict
        for metric in self.metrics_dict.values():
            value = metric.calculate(output, target)
            metric.update(value)
            
    def display(self):
        print_names = []
        print_values = []
        for metric_name, metric in self.metrics_dict.items():
            avg_values = metric.average()
            if hasattr(avg_values, 'cpu'):
                avg_values = avg_values.cpu()
            
            # avg_values can be tensor of size (n_classes,) or (1,)
            for i, value in enumerate(avg_values):
                print_values.append(value.item())
                if len(avg_values) == len(self.class_names):
                    value_name = self.class_names[i]
                    print_names.append(f'{value_name}_{metric_name}')
                else:
                    print_names.append(f'{metric_name}')
                
            if self.reset_on_print:
                metric.reset()
                
        for name, value in zip(print_names, print_values):
            if name not in self.history:
                self.history[name] = [value]
            else:
                self.history[name].append(value)
                
            print(name, value)
