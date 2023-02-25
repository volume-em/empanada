import torch.nn as nn
from empanada.models.blocks import Interpolate

__all__ = ['_make3d']

def _make3d(module):
    r"""Helper function to convert 2D models to 3D"""
    reassign = {}
    for name, mod in module.named_children():
        _make3d(mod)
        
        if type(mod) == nn.Conv2d:
            ks = tuple(3 * [mod.kernel_size[0]])
            st = tuple(3 * [mod.stride[0]])
            pad = tuple(3 * [mod.padding[0]])
            dilation = tuple(3 * [mod.dilation[0]])
            groups = mod.groups
            bias = mod.bias is not None
            
            reassign[name] = nn.Conv3d(
                mod.in_channels, mod.out_channels, ks, stride=st, 
                padding=pad, dilation=dilation, groups=groups, bias=bias
            )
            
        elif type(mod) == nn.ConvTranspose2d:
            ks = tuple(3 * [mod.kernel_size[0]])
            st = tuple(3 * [mod.stride[0]])
            pad = tuple(3 * [mod.padding[0]])
            dilation = tuple(3 * [mod.dilation[0]])
            groups = mod.groups
            bias = mod.bias is not None
            
            reassign[name] = nn.ConvTranspose3d(
                mod.in_channels, mod.out_channels, ks, stride=st, 
                padding=pad, dilation=dilation, groups=groups, bias=bias
            )
            
        elif type(mod) == nn.BatchNorm2d:
            reassign[name] = nn.BatchNorm3d(mod.num_features)
            
        elif type(mod) == nn.MaxPool2d:
            reassign[name] = nn.MaxPool3d(mod.kernel_size, stride=mod.stride, padding=mod.padding)

        elif type(mod) == nn.AvgPool2d:
            ks = tuple(3 * [mod.kernel_size[0]])
            st = tuple(3 * [mod.stride[0] if isinstance(mod.stride, tuple) else mod.stride])
            pad = tuple(3 * [mod.padding[0] if isinstance(mod.padding, tuple) else mod.padding])
            reassign[name] = nn.AvgPool3d(ks, stride=st, padding=pad)
            
        elif type(mod) == nn.AdaptiveAvgPool2d:
            reassign[name] = nn.AdaptiveAvgPool3d(1)
            
        elif type(mod) == Interpolate:
            reassign[name] = Interpolate(mod.scale_factor, 'trilinear', mod.align_corners)
        
    for key, value in reassign.items():
        module._modules[key] = value