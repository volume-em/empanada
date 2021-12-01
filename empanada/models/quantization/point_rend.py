import torch
import torch.nn as nn
from torch.quantization import fuse_modules
from empanada.models.point_rend import (
    PointRendSemSegHead, calculate_uncertainty,
    get_uncertain_point_coords_on_grid, point_sample
)

__all__ = [
    'QuantizablePointRendSemSegHead'
]

class QuantizablePointRendSemSegHead(PointRendSemSegHead):
    def __init__(self, *args, **kwargs):
        super(QuantizablePointRendSemSegHead, self).__init__(*args, **kwargs)
        
    def forward(self, sem_seg_logits, coarse_sem_seg_logits, features):
        
        #sem_seg_logits = coarse_sem_seg_logits.clone()
        
        # upsample by 2
        #for _ in range(steps:)
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
        N, C, H, W = sem_seg_logits.size()
        point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)

        sem_seg_logits = (
            sem_seg_logits.reshape(N, C, H * W)
            .scatter_(2, point_indices, point_logits)
            .view(N, C, H, W)
        )

        return sem_seg_logits

    def fuse_model(self):
        # fuse the PointHead fc layers
        layers = range(len(self.point_head.fc_layers))
        for layer in layers:
            fuse_modules(self.point_head.fc_layers[layer], ['0', '1'], inplace=True)
