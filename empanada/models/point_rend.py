"""
Copied with modification from:
https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend/point_rend

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from empanada.models.blocks import *

@torch.jit.script
def calculate_uncertainty(logits):
    """
    For each location of the prediction `sem_seg_logits` we estimate uncertainty as the
        difference between top first and top second predicted logits.
    Args:
        logits (Tensor): A tensor of shape (N, C, ...), where N is the minibatch size and
            C is the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (N, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    
    if logits.size(1) == 1:
        # binary segmentation
        uncertainty = -(torch.abs(logits))
    else:
        # multiclass segmentation
        top2_scores = torch.topk(logits, k=2, dim=1)[0]
        uncertainty = (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)
        
    return uncertainty

@torch.jit.script
def point_sample(features, point_coords, mode: str='bilinear', align_corners: bool=False):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        features (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    dim = 2 if features.ndim == 4 else 3
    
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        if dim == 3:
            point_coords = point_coords[:, :, None, None]
        else:
            point_coords = point_coords[:, :, None]
        
    output = F.grid_sample(features, 2.0 * point_coords - 1.0, mode=mode, align_corners=align_corners)
    
    if add_dim:
        output = output.flatten(-dim, -1)
            
    return output

@torch.jit.script
def get_uncertain_point_coords_with_randomness(
    coarse_logits, 
    num_points: int, 
    oversample_ratio: int, 
    importance_sample_ratio: float
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importance sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    
    dim = 2 if coarse_logits.ndim == 4 else 3
    
    point_coords = torch.rand(num_boxes, num_sampled, dim, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    
    point_uncertainties = calculate_uncertainty(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    
    point_coords = point_coords.view(-1, dim)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, dim
    )
    
    if num_random_points > 0:
        random_point_coords = torch.rand(num_boxes, num_random_points, dim, device=coarse_logits.device)
        point_coords = torch.cat([
            point_coords,
            random_point_coords
        ], dim=1)
        
    return point_coords

@torch.jit.script
def get_uncertain_point_coords_on_grid(uncertainty_map, num_points: int):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.
    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.
    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    dim = 2 if uncertainty_map.ndim == 4 else 3
    
    R = uncertainty_map.shape[0]
    spatial_dims = uncertainty_map.shape[-dim:]
    dsize = 1
    for s in spatial_dims:
        dsize *= s

    num_points = min(dsize, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, dsize), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, dim, dtype=torch.float, device=uncertainty_map.device)
    
    divisor = 1
    for i, size in enumerate(spatial_dims[::-1]):
        step = 1.0 / float(size)
        if i == 0:
            point_coords[:, :, i] = step * (point_indices % size).float()
        else:
            point_coords[:, :, i] = step * torch.div(point_indices, divisor, rounding_mode='floor').float()
            
        divisor *= size
    
    return point_indices, point_coords


class StandardPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    
    """
    def __init__(
        self,
        nin,
        num_classes,
        fc_dim,
        num_fc,
        coarse_pred_each_layer=True,
    ):
        super(StandardPointHead, self).__init__()
        
        fc_dim_in = nin + num_classes
        self.fc_layers = nn.ModuleList([])
        for k in range(num_fc):
            fc = nn.Sequential(
                nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1),
                nn.ReLU(inplace=True)
            )
            self.fc_layers.append(fc)
            
            fc_dim_in = fc_dim
            fc_dim_in += num_classes if coarse_pred_each_layer else 0

        self.predictor = nn.Conv1d(fc_dim_in, num_classes, kernel_size=1)

        # initialize the convs in fc layers
        for layer in self.fc_layers:
            nn.init.kaiming_normal_(layer[0].weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(layer[0].bias, 0)
                
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.constant_(self.predictor.bias, 0)
            
        self.coarse_pred_each_layer = coarse_pred_each_layer

    def forward(self, fine_grained_features, coarse_features):
        x = torch.cat([fine_grained_features, coarse_features], dim=1)
        for layer in self.fc_layers:
            x = layer(x)
            if self.coarse_pred_each_layer:
                x = torch.cat([x, coarse_features], dim=1)
                
        return self.predictor(x)
    

class PointRendSemSegHead(nn.Module):
    """
    A semantic segmentation head that combines a head set in `POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME`
    and a point head set in `MODEL.POINT_HEAD.NAME`.
    """

    def __init__(
        self, 
        nin,
        num_classes,
        num_fc=3,
        train_num_points=1024,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        subdivision_steps=2,
        subdivision_num_points=8192,
        **kwargs
    ):
        super(PointRendSemSegHead, self).__init__()
        self.train_num_points = train_num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.subdivision_steps = subdivision_steps
        self.subdivision_num_points = subdivision_num_points
        
        self.point_head = StandardPointHead(nin, num_classes, nin, num_fc, coarse_pred_each_layer=True)
        self.interpolate = Interpolate2d(2, mode='bilinear', align_corners=False)

    def forward(self, coarse_sem_seg_logits, features):
        dim = 2 if features.ndim == 4 else 3
        # for panoptic deeplab, coarse_sem_seg_logits is at 1/4th resolution
        pr_out = {}
        if self.training:
            # pick the points to apply point rend
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    coarse_sem_seg_logits,
                    self.train_num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                
            # sample points at coarse and fine resolutions
            coarse_sem_seg_points = point_sample(coarse_sem_seg_logits, point_coords, align_corners=False)
            fine_point_features = point_sample(features, point_coords, align_corners=False)
            point_logits = self.point_head(fine_point_features, coarse_sem_seg_points)
            
            # point coords are needed to generate targets later
            pr_out['sem_seg_logits'] = coarse_sem_seg_logits
            pr_out['point_logits'] = point_logits
            pr_out['point_coords'] = point_coords
        else:
            sem_seg_logits = coarse_sem_seg_logits.clone()
            
            for _ in range(self.subdivision_steps):
                # upsample by 2
                sem_seg_logits = self.interpolate(sem_seg_logits)
                
                # find the most uncertain point coordinates
                uncertainty_map = calculate_uncertainty(sem_seg_logits)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map, self.subdivision_num_points
                )
                
                # sample the coarse and fine points
                coarse_sem_seg_points = point_sample(coarse_sem_seg_logits, point_coords, align_corners=False)
                fine_point_features = point_sample(features, point_coords, align_corners=False)
                point_logits = self.point_head(fine_point_features, coarse_sem_seg_points)

                # put sem seg point predictions to the right places on the upsampled grid.
                N, C = sem_seg_logits.size()[:2]
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                
                # clunky if statements to appease TorchScript...
                if dim == 3:
                    D, H, W = sem_seg_logits.size()[-3:]
                    dsize = D * H * W
                    
                    sem_seg_logits = (
                        sem_seg_logits.reshape(N, C, dsize)
                        .scatter_(2, point_indices, point_logits)
                        .view(N, C, D, H, W)
                    )
                else:
                    H, W = sem_seg_logits.size()[-2:]
                    dsize = H * W
                
                    sem_seg_logits = (
                        sem_seg_logits.reshape(N, C, dsize)
                        .scatter_(2, point_indices, point_logits)
                        .view(N, C, H, W)
                    )
                
            pr_out['sem_seg_logits'] = sem_seg_logits
        
        return pr_out
