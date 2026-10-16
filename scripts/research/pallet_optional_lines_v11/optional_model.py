"""Preserve local correction; lines can only add a gated residual."""
import torch
from torch import nn
from common import Refiner, local, dht, point_maps


class OptionalRefiner(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.refine=Refiner(seed)
        self.gate=nn.Linear(64,1)
        nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-2.)

    def forward(self,O,base_points,lines,mass,arm,off=False):
        obs=dict(O,points=base_points)
        mode='local' if arm=='local_extra' else 'directional'
        out=self.refine(obs,lines,mass,mode)
        gate=torch.sigmoid(self.gate(out['token']))
        delta=gate*out['delta']
        if off:delta=torch.zeros_like(delta)
        return dict(points=base_points+delta,delta=delta,gate=gate,
            proposal=out['proposal'],A=out['A'],weights=out['weights'],base_points=base_points)


def predict_lines(model,O):
    p=model(O,point_maps(O['points'],O['box']),decode=True)
    return p['lines'].flatten(1,2),p['mass'].flatten(1,2)


def donor_lines(lines,mass,boxes,indices):
    src=boxes[indices];wh=(boxes[:,2:]-boxes[:,:2]).clamp_min(1.)
    scale=(src[:,2:]-src[:,:2]).clamp_min(1.)/wh
    l=lines[indices];n=l[...,:2]*scale[:,None]
    c=l[...,2]+(l[...,:2]*(src[:,None,:2]-scale[:,None]*boxes[:,None,:2])).sum(-1)
    return torch.cat((n,c[...,None]),-1),mass[indices]
