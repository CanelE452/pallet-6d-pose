"""Image-only localization control: same wide head, no coordinate hint or prior."""
import torch
from torch.nn import functional as F
from . import dino_wide_model as W

loss=W.loss
decode=W.decode


class Head(W.Head):
    def forward(self,features,points,valid):
        assert features.shape[1:]==(384,56,42)
        # Keep parameter shapes/initialization constant; disable BOTH point pathways.
        hints=features.new_zeros(len(features),9,56,42)
        y,x=torch.meshgrid(torch.linspace(-1,1,56,device=features.device),
                           torch.linspace(-1,1,42,device=features.device),indexing='ij')
        xy=torch.stack([x,y])[None].expand(len(features),-1,-1,-1)
        q=self.mix(torch.cat([self.project(features),hints,xy],dim=1))
        q=self.up(F.interpolate(q,size=(112,84),mode='bilinear',align_corners=False))
        return F.interpolate(self.out(q),size=(W.GRID_H,W.GRID_W),mode='bilinear',align_corners=False)
