"""
TODO: 3D Conv and RNN versions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from empanada.models.blocks import Resample2d

__all__ = [
    'MultiScaleQT'
]

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        nin,
        heads=8,
        attn_pdrop=0.5,
        resid_pdrop=0.5
    ):
        super(MultiHeadAttention, self).__init__()
        assert nin % heads == 0
        self.heads = heads
        self.head_nin = nin // self.heads
        
        # query, key value and output projections
        self.query = nn.Linear(nin, nin)
        self.key = nn.Linear(nin, nin)
        self.value = nn.Linear(nin, nin)
        self.project = nn.Linear(nin, nin)
        
        # dropout for attention and output projection
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(resid_pdrop)

    def forward(self, x):
        """
        """
        N, H, W, L, C = x.size()
        
        # project and reshape with heads
        q = rearrange(self.query(x), 'n h w l (a b) -> n h w a l b', a=self.heads, b=self.head_nin)
        k = rearrange(self.key(x), 'n h w l (a b) -> n h w a l b', a=self.heads, b=self.head_nin)
        v = rearrange(self.value(x), 'n h w l (a b) -> n h w a l b', a=self.heads, b=self.head_nin)
        
        # multiply q and k to get the (L, L) attention weights
        # and scale by the sqrt of queue length
        attn = q @ k.transpose(-2, -1)
        attn = attn / (math.sqrt(k.size(-1)))
        attn = F.softmax(attn, dim=-1)
        
        # apply dropout to attn
        attn = self.attn_drop(attn)
        
        # apply attention to value
        out = attn @ v
        out = rearrange(out, 'n h w a l b -> n h w l (a b)')
        out = self.proj_drop(self.project(out))
        
        return out

class TransformerBlock(nn.Module):
    def __init__(
        self,
        nin,
        heads=8,
        attn_pdrop=0.5,
        resid_pdrop=0.5
    ):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(nin, heads, attn_pdrop, resid_pdrop)

        self.mlp = nn.Sequential(
            nn.Linear(nin, nin * 4),
            nn.GELU(),
            nn.Linear(nin * 4, nin),
            nn.Dropout(resid_pdrop)
        )

        self.ln1 = nn.LayerNorm(nin)
        self.ln2 = nn.LayerNorm(nin)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class QueueTransformer(nn.Module):
    def __init__(
        self,
        nin,
        layers=1,
        heads=8,
        queue_len=5,
        attn_pdrop=0.5,
        resid_pdrop=0.5
    ):
        super(QueueTransformer, self).__init__()

        # create transformer layers
        params = nin, heads, attn_pdrop, resid_pdrop
        self.blocks = nn.Sequential(*[TransformerBlock(*params) for _ in range(layers)])

        # parameter to store position embeddings
        # ones for batch and head dim where we broadcast
        emb_shape = torch.randn(size=(1, queue_len, nin))
        self.pos_emb = nn.Parameter(emb_shape, requires_grad=True)

        # output projection head
        self.ln_out = nn.LayerNorm(nin)
        
    def forward(self, queue):
        N, C, L, H, W = queue.shape
        
        # swap the channel dimension to last position
        # for nn.Linear layers
        queue = rearrange(queue, 'n c l h w -> n h w l c')
        
        # add the learnable position embeddings
        queue = queue + self.pos_emb
        
        # run the queue through the transformer layers
        queue = self.ln_out(self.blocks(queue))
        
        # concatenate queue and channel dimension
        queue = rearrange(queue, 'n h w l c -> n c l h w')
        
        return queue
    
class MultiscaleQT(nn.Module):
    def __init__(
        self,
        nin,
        scales=5,
        layers=1,
        heads=8,
        queue_len=5,
        attn_pdrop=0.5,
        resid_pdrop=0.5
    ):
        super(MultiscaleQT, self).__init__()
        
        # create QueueTransformer for every feature map scale
        params = nin, layers, heads, queue_len, attn_pdrop, resid_pdrop
        self.transformers = nn.ModuleList(
            [QueueTransformer(*params) for _ in range(scales)]
        )
        self.apply(init_weights)
    
    def forward(self, fpn_features):
        assert len(fpn_features) == len(self.transformers)
        
        qt_features = []
        for xformer, features in zip(self.transformers, fpn_features):
            qt_features.append(xformer(features))
            
        return qt_features
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.LayerNorm):
        m.bias.data.zero_()
        m.weight.data.fill_(1.0)
