import torch
import torch.nn as nn
from torch.quantization import fuse_modules, QuantStub, DeQuantStub
from empanada.models.point_rend import (
    PointRendSemSegHead, calculate_uncertainty,
    get_uncertain_point_coords_on_grid, point_sample,
    get_uncertain_point_coords_with_randomness
)

__all__ = [
    'QuantizablePointRendSemSegHead'
]

class QuantizablePointRendSemSegHead(PointRendSemSegHead):
    def __init__(self, *args, **kwargs):
        super(QuantizablePointRendSemSegHead, self).__init__(*args, **kwargs)
        
        if kwargs['quantize']:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()
        
    def forward(self, coarse_sem_seg_logits, features):
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
            coarse_sem_seg_logits = self.dequant(coarse_sem_seg_logits)
            features = self.dequant(features)
            
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
                
                fine_point_features = self.quant(fine_point_features)
                coarse_sem_seg_points = self.quant(coarse_sem_seg_points)
                
                point_logits = self.point_head(fine_point_features, coarse_sem_seg_points)
                
                sem_seg_logits = self.dequant(sem_seg_logits)
                point_logits = self.dequant(point_logits)

                # put sem seg point predictions to the right places on the upsampled grid.
                N, C, H, W = sem_seg_logits.size()
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                
                sem_seg_logits = (
                    sem_seg_logits.reshape(N, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(N, C, H, W)
                )
            
                
            pr_out['sem_seg_logits'] = sem_seg_logits
        
        return pr_out

    def fuse_model(self):
        # fuse the PointHead fc layers
        layers = range(len(self.point_head.fc_layers))
        for layer in layers:
            fuse_modules(self.point_head.fc_layers[layer], ['0', '1'], inplace=True)
