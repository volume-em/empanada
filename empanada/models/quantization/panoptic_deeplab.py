import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import fuse_modules, QuantStub, DeQuantStub
from empanada.models.quantization import encoders
from empanada.models.quantization.point_rend import QuantizablePointRendSemSegHead
from empanada.models.quantization.decoders import QuantizablePanopticDeepLabDecoder
from empanada.models.heads import PanopticDeepLabHead
from empanada.models.blocks import *
from typing import List

backbones = sorted(name for name in encoders.__dict__
    if not name.startswith("__")
    and callable(encoders.__dict__[name])
)

__all__ = [
    'QuantizablePanopticDeepLab',
    'QuantizablePanopticDeepLabPR'
]

@torch.jit.script
def factor_pad(tensor, factor: int=16):
    h, w = tensor.size()[2:]
    pad_bottom = factor - h % factor if h % factor != 0 else 0
    pad_right = factor - w % factor if w % factor != 0 else 0
    if pad_bottom == 0 and pad_right == 0:
        return tensor
    else:
        return F.pad(tensor, (0, pad_right, 0, pad_bottom))
    
@torch.jit.script
def normalize_tensor(tensor, mean: float, std: float):
    return (tensor - mean) / std

def _replace_relu(module):
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6 or type(mod) == nn.SiLU:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value
    
class QuantizablePanopticDeepLab(nn.Module):
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
        quantize=False,
        **kwargs
    ):
        super(QuantizablePanopticDeepLab, self).__init__()
        
        assert (encoder in backbones), \
        f'Invalid encoder name {encoder}, choices are {backbones}'
        assert stage4_stride in [16, 32]
        assert min(low_level_stages) > 0
        
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes
        self.encoder = encoders.__dict__[encoder](output_stride=stage4_stride)
        
        self.semantic_decoder = QuantizablePanopticDeepLabDecoder(
            int(self.encoder.cfg.widths[-1]),
            decoder_channels,
            low_level_stages,
            [int(self.encoder.cfg.widths[i - 1]) for i in low_level_stages], 
            low_level_channels_project,
            atrous_rates, 
            aspp_channels
        )
        
        if ins_decoder:
            self.instance_decoder = QuantizablePanopticDeepLabDecoder(
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
            
        _replace_relu(self)
        
        if quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()
    
    def set_qconfig(self, observer='fbgemm'):
        self.qconfig = torch.quantization.get_default_qconfig(observer)

    def _encode_decode(self, x):
        pyramid_features: List[torch.Tensor] = self.encoder(x)
        
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
        
        # return at quarter resolution
        heads_out['sem_logits'] = sem
        heads_out['ctr_hmp'] = ctr_hmp
        heads_out['offsets'] = offsets
        heads_out['semantic_x'] = semantic_x
        
        return heads_out
            
    def forward(self, x):
        x = self.quant(x)
            
        pyramid_features, semantic_x, instance_x = self._encode_decode(x)
        output: Dict[str, torch.Tensor] = self._apply_heads(semantic_x, instance_x)
        
        # classify the image annotation confidence
        if self.confidence_head is not None:
            conf = self.confidence_head(pyramid_features[-1])
            output['conf'] = conf
            
        # dequant all outputs
        output = {k: self.dequant(v) for k,v in output.items()}
        
        return output
    
    def fuse_model(self):
        self.encoder.fuse_model()
        self.semantic_decoder.fuse_model()
        if self.instance_decoder is not None:
            self.instance_decoder.fuse_model()

class QuantizablePanopticDeepLabPR(QuantizablePanopticDeepLab):
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
        super(QuantizablePanopticDeepLabPR, self).__init__(**kwargs)
        
        # change semantic head from regular PDL head to 
        # PDL head + PointRend
        self.semantic_pr = QuantizablePointRendSemSegHead(
            self.decoder_channels, self.num_classes, num_fc,
            train_num_points, oversample_ratio, 
            importance_sample_ratio, subdivision_steps,
            subdivision_num_points
        )
        
    def _apply_heads(self, semantic_x, instance_x):
        heads_out = {}
        # only runs in eval mode
        coarse_sem_seg_logits = self.semantic_head(semantic_x)
        ctr_hmp = self.ins_center(instance_x)
        offsets = self.ins_xy(instance_x)
        
        sem_seg_logits = coarse_sem_seg_logits.clone()
        for _ in range(2):
            sem_seg_logits = self.semantic_pr(sem_seg_logits, coarse_sem_seg_logits, semantic_x)
            
        # resize to original image resolution (4x)
        heads_out['sem_logits'] = sem_seg_logits
        heads_out['ctr_hmp'] = self.interpolate(ctr_hmp)
        heads_out['offsets'] = self.interpolate(offsets)
        
        return heads_out
    
    def fuse_model(self):
        self.encoder.fuse_model()
        self.semantic_decoder.fuse_model()
        self.semantic_head.fuse_model()
        if self.instance_decoder is not None:
            self.instance_decoder.fuse_model()
