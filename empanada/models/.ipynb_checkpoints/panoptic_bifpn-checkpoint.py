import torch
import torch.nn as nn
from empanada.models import encoders
from empanada.models.decoders import BiFPN, BiFPNDecoder
from empanada.models.heads import PanopticDeepLabHead
from empanada.models.blocks import *
from empanada.models import encoders
from einops import rearrange
from typing import List
from copy import deepcopy

backbones = sorted(name for name in encoders.__dict__
    if not name.startswith("__")
    and callable(encoders.__dict__[name])
)

__all__ = [
    'PanopticBiFPN'
]

class _BaseModel(nn.Module):
    def __init__(
        self,
        encoder='regnety_6p4gf',
        num_classes=1,
        fpn_dim=160,
        fpn_layers=3,
        ins_decoder=False,
        confidence_head=False,
        confidence_bins=5,
        **kwargs
    ):
        super(_BaseModel, self).__init__()
        
        assert (encoder in backbones), \
        f'Invalid encoder name {encoder}, choices are {backbones}'
        
        self.encoder = encoders.__dict__[encoder]()
        self.p2_resample = Resample2d(int(self.encoder.cfg.widths[0]), fpn_dim)
        
        # pass input channels from stages 2-4 only (1/8->1/32 resolutions)
        # N.B. EfficientDet BiFPN uses compound scaling rules that we ignore
        if 'resnet' in encoder:
            widths = self.encoder.cfg.widths[1:].tolist()
        else:
            widths = self.encoder.cfg.widths[1:]
            
        self.semantic_fpn = BiFPN(deepcopy(widths), fpn_dim, fpn_layers)
        self.semantic_decoder = BiFPNDecoder(fpn_dim)
        
        # separate BiFPN for instance-level predictions
        # following PanopticDeepLab
        if ins_decoder:
            self.instance_fpn = BiFPN(deepcopy(widths), fpn_dim, fpn_layers)
            self.instance_decoder = BiFPNDecoder(fpn_dim)
        else:
            self.instance_fpn = None
            
        self.semantic_head = PanopticDeepLabHead(fpn_dim, num_classes)
        self.ins_center = PanopticDeepLabHead(fpn_dim, 1)
        self.ins_xy = PanopticDeepLabHead(fpn_dim, 2)
        
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
            
    def _forward_encoder(self, x):
        return self.encoder(x)
            
    def _forward_fpn(self, x: List[torch.Tensor], kind: str='semantic'):
        if kind == 'semantic':
            return self.semantic_fpn(x)
        elif kind == 'instance':
            return self.instance_fpn(x)
        else:
            raise Exception(f'FPN must be semantic or instance, got {kind}.')
            
    def _forward_decoder(self, x: List[torch.Tensor], kind: str='semantic'):
        if kind == 'semantic':
            return self.semantic_decoder(x)
        elif kind == 'instance':
            return self.instance_decoder(x)
        else:
            raise Exception(f'Decoder must be semantic or instance, got {kind}.')
        
    def forward(self, x):
        pyramid_features: List[torch.Tensor] = self._forward_encoder(x)
        p2_features = self.p2_resample(pyramid_features[1])
        
        # only passes features from
        # 1/8 -> 1/32 resolutions (i.e. P3-P5)
        semantic_fpn_features: List[torch.Tensor] = self._forward_fpn(pyramid_features[2:], kind='semantic')
        
        # resample and prepend 1/4 resolution (P2) 
        # features for segmentation (following EfficientDet)
        semantic_fpn_features = [p2_features] + semantic_fpn_features
        
        # decode features upwards through pyramid
        semantic_x = self._forward_decoder(semantic_fpn_features[::-1], kind='semantic')
        sem = self.semantic_head(semantic_x)
        
        if self.instance_fpn is not None:
            instance_fpn_features = self._forward_fpn(pyramid_features[2:], kind='instance')
            instance_fpn_features = [p2_features] + instance_fpn_features
            instance_x = self._forward_decoder(instance_fpn_features[::-1], kind='instance')
            
            ctr_hmp = self.ins_center(instance_x)
            offsets = self.ins_xy(instance_x)
        else:
            ctr_hmp = self.ins_center(semantic_x)
            offsets = self.ins_xy(semantic_x)
        
        # return at original image resolution (4x)
        output = {}
        output['sem_logits'] = self.interpolate(sem)
        output['ctr_hmp'] = self.interpolate(ctr_hmp)
        output['offsets'] = self.interpolate(offsets)
        
        # classify the image annotation confidence
        if self.confidence_head is not None:
            output['conf'] = self.confidence_head(pyramid_features[-1])
        
        return output
    
class PanopticBiFPN(_BaseModel):
    def __init__(
        self,
        encoder='regnety_6p4gf',
        num_classes=1,
        fpn_dim=160,
        fpn_layers=3,
        ins_decoder=False,
        confidence_head=False,
        confidence_bins=5,
        **kwargs
    ):
        super(PanopticBiFPN, self).__init__(
            encoder,
            num_classes,
            fpn_dim,
            fpn_layers,
            ins_decoder,
            confidence_head,
            confidence_bins,
            **kwargs
        )