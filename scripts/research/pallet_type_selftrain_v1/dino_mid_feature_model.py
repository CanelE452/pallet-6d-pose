"""Same image-only head, with a zero-start mid-transformer feature path."""
import torch
from torch import nn
from torch.nn import functional as F
from . import dino_wide_visual_model as V

loss=V.loss
decode=V.decode


class Head(V.Head):
    def __init__(self):
        super().__init__()
        self.mid=nn.Sequential(nn.Conv2d(384,64,1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,64,1))
        nn.init.zeros_(self.mid[-1].weight);nn.init.zeros_(self.mid[-1].bias)

    def forward(self,features,points,valid):
        late,mid=features
        assert late.shape[1:]==mid.shape[1:]==(384,56,42)
        hints=late.new_zeros(len(late),9,56,42)
        y,x=torch.meshgrid(torch.linspace(-1,1,56,device=late.device),
            torch.linspace(-1,1,42,device=late.device),indexing='ij')
        xy=torch.stack([x,y])[None].expand(len(late),-1,-1,-1)
        q=self.mix(torch.cat([self.project(late)+self.mid(mid),hints,xy],1))
        q=self.up(F.interpolate(q,size=(112,84),mode='bilinear',align_corners=False))
        return F.interpolate(self.out(q),size=(192,144),mode='bilinear',align_corners=False)


def block_features(backbone,tensor,both=False):
    """Sixth block (zero-based5); identical official LayerNorm and token removal.

    The cache path stops at6. The inference/audit path also returns block12.
    Nothing in this function consumes points, labels, masks or dimensions.
    """
    assert not backbone.chunked_blocks and len(backbone.blocks)==12
    x=backbone.prepare_tokens_with_masks(tensor);out=[]
    for i,block in enumerate(backbone.blocks):
        x=block(x)
        if i in [5,11]:out.append(backbone.norm(x)[:,1+backbone.num_register_tokens:])
        if i==5 and not both:break
    return out
