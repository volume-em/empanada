"""
Confidence weighting of loss that's efficient.
"""

import torch
import torch.nn as nn
from empanada.models.point_rend import point_sample

"""
class BootstrapCE(nn.Module):
    def __init__(
        self, 
        top_k_percent_pixels=1.0, 
        ignore_label=-1, 
        weight=None,
        binary=False
    ):
        super(BootstrapCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.binary = binary
        if binary:
            assert ignore_label == -1
            self.criterion = nn.BCEWithLogitsLoss(weight=weight, reduction='none')
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=weight, ignore_index=ignore_label, reduction='none'
            )

    def forward(self, logits, labels, weights=None):
        if self.binary:
            labels = labels.unsqueeze(1) # (N, H, W) -> (N, 1, H, W)
            
        pixel_losses = self.criterion(logits, labels)
        if weights is not None:
            # batch sizes should be the same
            assert weights.size(0) == pixel_losses.size(0)
            pixel_losses = pixel_losses * weights
        
        pixel_losses = pixel_losses.contiguous().view(-1)
        
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.sum() / weights.sum()
        
        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        return pixel_losses.mean()
    
class PanopticLoss(nn.Module):
    def __init__(
        self, 
        ce_weight=1, 
        mse_weight=10, 
        l1_weight=0.01,
        ce_top_k_percent=0.15,
        ce_ignore_label=-1,
        conf_loss=False,
        conf_bin_weights=None,
        binary=False
    ):
        super(PanopticLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        
        self.ce_loss = BootstrapCE(ce_top_k_percent, ignore_label=ce_ignore_label, binary=binary)
        self.ce_weight = ce_weight
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        
        if conf_loss:
            self.conf_ce = nn.CrossEntropyLoss()
        else:
            self.conf_ce = None
            
        if conf_bin_weights is not None:
            self.register_buffer('conf_bin_weights', torch.tensor(conf_bin_weights))
        else:
            self.conf_bin_weights = conf_bin_weights
        
    def forward(self, output, target):
        # compute confidence classification loss
        # first and determine example weights
        if self.conf_ce is not None:
            assert 'conf' in output and 'conf' in target
            conf_ce = self.conf_ce(output['conf'], target['conf'])
        else:
            conf_ce = None
        
        # either weights from tensor or ones
        if self.conf_ce is not None and self.conf_bin_weights is not None:
            conf_weights = self.conf_bin_weights[target['conf']] # (N, 1)
        else:
            conf_weights = torch.ones_like(target['conf']).float()
            
        # expand weights to correct dimension
        conf_weights = conf_weights[..., None, None]
            
        ce = self.ce_loss(output['sem'], target['sem'], conf_weights)
        mse = self.mse_loss(output['ctr_hmp'], target['ctr_hmp']) * conf_weights
        mse = mse.mean(dim=(-1, -2)).sum() / conf_weights.sum()
        
        # (N, 1, H, W), weight is 1 inside mask and 0 outside
        # and then scaled by conf_weights
        offset_loss_weights = conf_weights * (target['sem'] > 0)
        offset_weight_sum = offset_loss_weights.sum()
        l1 = self.l1_loss(output['offsets'], target['offsets']) * offset_loss_weights.unsqueeze(1)
        l1 = l1.sum() / offset_weight_sum if offset_weight_sum > 0 else l1.sum() * 0
        
        if conf_ce is None:
            conf_ce = torch.zeros_like(l1) # (1,)
            
        # store loss items before weighting
        aux_loss = {'ce': ce.item(), 'l1': l1.item(), 'mse': mse.item(), 'conf': conf_ce.item()}
        
        # weight and sum the loss
        total_loss = self.ce_weight * ce + self.mse_weight * mse + self.l1_weight * l1 + conf_ce
        return total_loss, aux_loss
"""

class BootstrapCE(nn.Module):
    def __init__(self, top_k_percent_pixels=1.0):
        super(BootstrapCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        if logits.size(1) == 1:
            # add channel dim for BCE
            # (N, H, W) -> (N, 1, H, W)
            labels = labels.unsqueeze(1) 
            pixel_losses = self.bce(logits, labels)
        else:
            pixel_losses = self.ce(logits, labels)
            
        pixel_losses = pixel_losses.contiguous().view(-1)
        
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()
        
        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        
        return pixel_losses.mean()

class HeatmapMSE(nn.Module):
    def __init__(self):
        super(HeatmapMSE, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, output, target):
        return self.mse(output, target)
    
class OffsetL1(nn.Module):
    def __init__(self):
        super(OffsetL1, self).__init__()
        self.l1 = nn.L1Loss(reduction='none')
        
    def forward(self, output, target, offset_weights):
        l1 = self.l1(output, target) * offset_weights
        
        weight_sum = offset_weights.sum()
        if weight_sum == 0:
            return l1.sum() * 0
        else:
            return l1.sum() / weight_sum
        
class PointRendLoss(nn.Module):
    def __init__(self):
        super(PointRendLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, point_logits, point_coords, labels):
        # sample the labels at the given coordinates
        point_labels = point_sample(
            labels.unsqueeze(1).float(), point_coords,
            mode="nearest", align_corners=False
        )
        
        if point_logits.size(1) == 1:
            point_losses = self.bce(point_logits, point_labels)
        else:
            point_labels = point_labels.squeeze(1).long()
            point_losses = self.ce(point_logits, point_labels)
        
        return point_losses
    
class PanopticLoss(nn.Module):
    def __init__(
        self, 
        ce_weight=1, 
        mse_weight=200, 
        l1_weight=0.01,
        top_k_percent=0.15,
        confidence_loss=False,
        cl_weight=0.1,
        pr_weight=1
    ):
        super(PanopticLoss, self).__init__()
        self.mse_loss = HeatmapMSE()
        self.l1_loss = OffsetL1()
        self.ce_loss = BootstrapCE(top_k_percent)
        self.pr_loss = PointRendLoss()
        
        self.ce_weight = ce_weight
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.cl_weight = cl_weight
        self.pr_weight = pr_weight
        
        if confidence_loss:
            self.conf_loss = nn.CrossEntropyLoss()
        else:
            self.conf_loss = None
        
    def forward(self, output, target):
        mse = self.mse_loss(output['ctr_hmp'], target['ctr_hmp'])
        ce = self.ce_loss(output['sem_logits'], target['sem'])
        
        # only evaluate loss inside of ground truth segmentation
        offset_weights = (target['sem'] > 0).unsqueeze(1)
        l1 = self.l1_loss(output['offsets'], target['offsets'], offset_weights)
        
        aux_loss = {'ce': ce.item(), 'l1': l1.item(), 'mse': mse.item()}
        total_loss = self.ce_weight * ce + self.mse_weight * mse + self.l1_weight * l1
        
        if self.conf_loss is not None and 'conf' in output:
            conf_ce = self.conf_loss(output['conf'], target['conf'])
            aux_loss['conf_ce'] = conf_ce.item()
            total_loss += self.cl_weight * conf_ce
            
        if 'sem_points' in output:
            pr_ce = self.pr_loss(output['sem_points'], output['point_coords'], target['sem'])
            aux_loss['pointrend_ce'] = pr_ce.item()
            total_loss += self.pr_weight * pr_ce
        
        return total_loss, aux_loss
