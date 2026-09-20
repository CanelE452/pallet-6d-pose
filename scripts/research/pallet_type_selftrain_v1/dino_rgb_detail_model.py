"""Image-only DINO semantics plus an aligned stride-four RGB detail path."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import dino_wide_visual_model as V

loss=V.loss
decode=V.decode


def detail_rgb(rgb,mean):
    """Index (y,x) is crop (4y,4x), not resize's half-pixel-shifted center."""
    image=np.asarray(rgb,np.float32).transpose(1,2,0)+np.asarray(mean,np.float32)
    if image.shape!=(768,576,3) or not np.isfinite(image).all():
        raise ValueError('Expected finite wide RGB crop')
    smoothed=cv2.GaussianBlur(image,(5,5),1.0,borderType=cv2.BORDER_REFLECT_101)
    return np.ascontiguousarray((smoothed[::4,::4]/255.-.5).transpose(2,0,1),dtype=np.float16)


class Head(V.Head):
    def __init__(self):
        super().__init__()
        self.detail=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GroupNorm(4,16),nn.GELU(),
            nn.Conv2d(16,32,3,padding=1),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,32,1))
        nn.init.zeros_(self.detail[-1].weight);nn.init.zeros_(self.detail[-1].bias)

    def forward(self,features,points,valid):
        dino,rgb=features
        assert dino.shape[1:]==(384,56,42) and rgb.shape[1:]==(3,192,144)
        hints=dino.new_zeros(len(dino),9,56,42)
        y,x=torch.meshgrid(torch.linspace(-1,1,56,device=dino.device),
            torch.linspace(-1,1,42,device=dino.device),indexing='ij')
        xy=torch.stack([x,y])[None].expand(len(dino),-1,-1,-1)
        q=self.mix(torch.cat([self.project(dino),hints,xy],1))
        q=self.up(F.interpolate(q,size=(112,84),mode='bilinear',align_corners=False))
        # Preserve parent's output path exactly; extra branch has zero output at init.
        base=F.interpolate(self.out(q),size=(192,144),mode='bilinear',align_corners=False)
        # Subtract the shared bias: it belongs to the base path only.
        return base+F.conv2d(self.detail(rgb),self.out.weight,bias=None)
