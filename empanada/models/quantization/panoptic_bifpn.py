import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from empanada.models.quantization import encoders
from empanada.models import PanopticBiFPN
from typing import List

__all__ = [
    'QuantizablePanopticBiFPN'
]

def _replace_relu(module):
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value

class QuantizablePanopticBiFPN(PanopticBiFPN):
    def __init__(
        self,
        quantize=False,
        interpolate=False,
        **kwargs,
    ):
        super(QuantizablePanopticBiFPN, self).__init__(**kwargs)
        
        encoder = kwargs['encoder']
        # only the encoder is quantizable
        self.encoder = encoders.__dict__[encoder]()
        
        _replace_relu(self)
        
        # optionally, turn off interpolation
        # output will be 1/4th size of input
        if not interpolate:
            self.interpolate = nn.Identity()
        
        if quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()
    
    def set_qconfig(self, observer='fbgemm'):
        # only the encoder gets quantized
        self.encoder.qconfig = torch.quantization.get_default_qconfig(observer)
        self.quant.qconfig = torch.quantization.get_default_qconfig(observer)
        self.dequant.qconfig = torch.quantization.get_default_qconfig(observer)
    
    def _forward_encoder(self, x: torch.Tensor):
        x = self.quant(x)
        features: List[torch.Tensor] = self.encoder(x)
        return [self.dequant(t) for t in features]
    
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
    
    def fuse_model(self):
        self.encoder.fuse_model()
            
