"""Learn point-conditioned spatial feature gates before Deep Hough voting.

No point pairs, inferred connecting segments, GT points, or coordinate updates.
The 3-channel point-neighborhood field is invariant to point ordering.
"""
import sys
from pathlib import Path
import torch
from torch import nn
KIT=Path(__file__).resolve().parent
sys.path.insert(0,str(KIT.parent/'pallet_point_line_decisive_v6'))
from diagnostics import load_model
from plpose_local.heads import decode_modes,roi_lines_to_raw
from plpose_local.losses import line_targets
from plpose_local.geometry import symmetry_permutations


def point_maps(points,box,size=24):
    axis=torch.linspace(0,1,size,device=points.device,dtype=points.dtype)
    y,x=torch.meshgrid(axis,axis,indexing='ij');grid=torch.stack((x,y),-1)
    wh=box[:,2:]-box[:,:2]
    raw=box[:,None,None,:2]+grid[None]*wh[:,None,None]
    distance=(raw[:,:,:,None]-points[:,None,None]).square().sum(-1).amin(-1)
    sigma=torch.linalg.vector_norm(wh,dim=-1)[:,None]*points.new_tensor([[.01,.03,.10]])
    return torch.exp(-distance[:,None]/(2*sigma[:,:,None,None].square().clamp_min(1e-6)))


class GuidedDHT(nn.Module):
    def __init__(self,checkpoint,seed):
        super().__init__()
        base,_=load_model(checkpoint,'cpu')
        self.trunk=base.trunk;self.line=base.line
        self.register_buffer('lattice',base.lattice.clone())
        self.thetas,self.rhos,self.modes=base.thetas,base.rhos,base.modes
        torch.manual_seed(19000+seed)
        self.gate=nn.Sequential(nn.Conv2d(51,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,48,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)

    def forward(self,O,maps,decode=False):
        x=self.trunk(O['features'],O['content'],O['K'],O['dims'],O['box'],O['image_hw'])
        # Positive bounded gate preserves at least half of each global feature.
        gate=1.+.5*torch.tanh(self.gate(torch.cat((x,maps),1)))
        logits=self.line(x*gate)
        out=dict(logits=logits,gate=gate)
        if decode:
            lines,mass=decode_modes(logits,self.lattice,self.thetas,self.rhos,self.modes)
            out.update(lines=roi_lines_to_raw(lines,O['box']),mass=mass)
        return out


def targets(S,O):
    original,valid=line_targets(S['points'],O['box'],36,65)
    perm=symmetry_permutations(2)[1]
    flipped,fvalid=line_targets(S['points'][:,perm],O['box'],36,65)
    return torch.stack((original,flipped),1),torch.stack((valid,fvalid),1)


def objective(logits,target,valid,orders):
    logp=logits.log_softmax(-1)[:,None].expand(-1,2,-1,-1)
    nll=-(logp.gather(-1,target[...,None])[...,0]*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    allowed=torch.stack((torch.ones_like(orders,dtype=torch.bool),orders==2),-1)
    return nll.masked_fill(~allowed,float('inf')).min(-1).values.mean()
