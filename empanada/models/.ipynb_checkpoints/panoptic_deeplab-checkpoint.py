import torch
import torch.nn as nn
from empanada.models.decoders import PanopticDeepLabDecoder
from empanada.models.heads import PanopticDeepLabHead
from empanada.models.point_rend import PointRendSemSegHead
from empanada.models.blocks import *
from empanada.models import encoders

backbones = sorted(name for name in encoders.__dict__
    if not name.startswith("__")
    and callable(encoders.__dict__[name])
)

__all__ = [
    'PanopticDeepLab',
    'PanopticDeepLabPR'
]
    
class PanopticDeepLab(nn.Module):
    def __init__(
        self,
        encoder='resnet50',
        num_classes=1,
        stage4_stride=16,
        decoder_channels=256,
        low_level_stages=[3, 2, 1],
        low_level_channels_project=[128, 64, 32],
        atrous_rates=[2, 4, 6],
        aspp_channels=None,
        ins_decoder=False,
        ins_ratio=0.5,
        confidence_head=False,
        confidence_bins=5,
        **kwargs
    ):
        super(PanopticDeepLab, self).__init__()
        
        assert (encoder in backbones), \
        f'Invalid encoder name {encoder}, choices are {backbones}'
        assert stage4_stride in [16, 32]
        assert min(low_level_stages) > 0
        
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes
        self.encoder = encoders.__dict__[encoder](output_stride=stage4_stride)
        
        self.semantic_decoder = PanopticDeepLabDecoder(
            int(self.encoder.cfg.widths[-1]),
            decoder_channels,
            low_level_stages,
            [int(self.encoder.cfg.widths[i - 1]) for i in low_level_stages], 
            low_level_channels_project,
            atrous_rates, 
            aspp_channels
        )
        
        if ins_decoder:
            self.instance_decoder = PanopticDeepLabDecoder(
                int(self.encoder.cfg.widths[-1]),
                decoder_channels,
                low_level_stages,
                [int(self.encoder.cfg.widths[i - 1]) for i in low_level_stages], 
                [int(s * ins_ratio) for s in low_level_channels_project],
                atrous_rates, 
                aspp_channels
            )
        else:
            self.instance_decoder = None
        
        self.semantic_head = PanopticDeepLabHead(decoder_channels, num_classes)
        self.ins_center = PanopticDeepLabHead(decoder_channels, 1)
        self.ins_xy = PanopticDeepLabHead(decoder_channels, 2)
        
        self.interpolate = Interpolate2d(4, mode='bilinear', align_corners=True)
        
        # add classification head, if needed
        if confidence_head:
            assert confidence_bins is not None
            assert confidence_bins >= 3
            self.confidence_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1, -1),
                nn.Linear(self.encoder.cfg.widths[-1], confidence_bins)
            )
        else:
            self.confidence_head = None
            
    def _encode_decode(self, x):
        pyramid_features = self.encoder(x)
        
        semantic_x = self.semantic_decoder(pyramid_features)
        sem = self.semantic_head(semantic_x)
        
        if self.instance_decoder is not None:
            instance_x = self.instance_decoder(pyramid_features)
        else:
            # this shouldn't make a copy!
            instance_x = semantic_x
            
        return pyramid_features, semantic_x, instance_x
    
    def _apply_heads(self, semantic_x, instance_x):
        heads_out = {}
        sem = self.semantic_head(semantic_x)
        ctr_hmp = self.ins_center(instance_x)
        offsets = self.ins_xy(instance_x)
        
        # return at original image resolution (4x)
        heads_out['sem_logits'] = self.interpolate(sem)
        heads_out['ctr_hmp'] = self.interpolate(ctr_hmp)
        heads_out['offsets'] = self.interpolate(offsets)
        
        return heads_out
            
    def forward(self, x):
        pyramid_features, semantic_x, instance_x = self._encode_decode(x)
        output = self._apply_heads(semantic_x, instance_x)
        
        # classify the image annotation confidence
        if self.confidence_head is not None:
            output['conf'] = self.confidence_head(pyramid_features[-1])
        
        return output
    
class PanopticDeepLabPR(PanopticDeepLab):
    def __init__(
        self,
        num_fc=3,
        train_num_points=1024,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        subdivision_steps=2,
        subdivision_num_points=8192,
        **kwargs
    ):
        super(PanopticDeepLabPR, self).__init__(**kwargs)
        
        # change semantic head from regular PDL head to 
        # PDL head + PointRend
        self.semantic_pr = PointRendSemSegHead(
            self.decoder_channels, self.num_classes, num_fc,
            train_num_points, oversample_ratio, 
            importance_sample_ratio, subdivision_steps,
            subdivision_num_points
        )
        
    def _apply_heads(self, semantic_x, instance_x):
        heads_out = {}
        
        sem = self.semantic_head(semantic_x)
        ctr_hmp = self.ins_center(instance_x)
        offsets = self.ins_xy(instance_x)
        pr_out = self.semantic_pr(sem, semantic_x)
        
        if self.training:
            # interpolate to original resolution (4x)
            heads_out['sem_logits'] = self.interpolate(pr_out['sem_seg_logits'])
            heads_out['sem_points'] = pr_out['point_logits']
            heads_out['point_coords'] = pr_out['point_coords']
        else:
            # in eval mode interpolation is handled by point rend
            heads_out['sem_logits'] = pr_out['sem_seg_logits']
            
        # resize to original image resolution (4x)
        heads_out['ctr_hmp'] = self.interpolate(ctr_hmp)
        heads_out['offsets'] = self.interpolate(offsets)
        
        return heads_out
        