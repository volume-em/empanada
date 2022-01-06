import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from empanada.models.quantization import encoders
from empanada.models import PanopticBiFPN
from typing import List

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
    
    def _forward_encoder(self, x: torch.Tensor):
        x = self.quant(x)
        features: List[torch.Tensor] = self.encoder(x)
        return [self.dequant(t) for t in features]
    
    def fuse_model(self):
        self.encoder.fuse_model()
            
