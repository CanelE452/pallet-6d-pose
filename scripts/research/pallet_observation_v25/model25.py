"""Shared point/line encoding with a direction-capable zero-initialized output."""
from context25 import *
from model20 import CauseGate,objective
from model19 import observed_samples
from torch import nn
import torch.nn.functional as F

class Correction25(CauseGate):
    def __init__(self,seed):
        super().__init__(seed)
        self.gate.register_forward_pre_hook(self.capture)
        self.edge_encode=nn.Sequential(nn.Linear(171,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.vector_head=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2))
        nn.init.zeros_(self.vector_head[-1].weight);nn.init.zeros_(self.vector_head[-1].bias)

    def capture(self,module,args):self.encoded=args[0]

    def forward(self,O,rep,arm='vector_line',evidence_off=False,evidence_features=None):
        local='local' in arm;mode='unit_weighted_local' if local else 'unit_weighted'
        out=super().forward(O,rep,mode,evidence_off=evidence_off,evidence_features=evidence_features)
        if arm.startswith('gate_'):return out
        extra=self.encoded.new_zeros((*self.encoded.shape[:-1],32))
        if arm.endswith('_edge'):
            b=rep['base'];box=O['box'];wh=(box[:,2:]-box[:,:2]).clamp_min(1);diag=wh.norm(dim=-1)
            y,x=torch.meshgrid(b.new_tensor([-1,0,1]),b.new_tensor([-1,0,1]),indexing='ij');offset=torch.stack((x.flatten(),y.flatten()),-1)
            pos=b[:,:,None]+offset[None,None]*.03*diag[:,None,None,None];grid=2*(pos-box[:,None,None,:2])/wh[:,None,None]-1
            hw=O['image_hw'].flip(-1)[:,None,None];inside=((pos>=0)&(pos<hw)).all(-1)&((grid>=-1)&(grid<=1)).all(-1)
            localpix=F.grid_sample(O['edge'],grid,align_corners=True,padding_mode='zeros').permute(0,2,3,1)*inside[...,None]
            sample=observed_samples(O,rep,'line');pos=sample['pos'];grid=2*(pos-box[:,None,None,:2])/wh[:,None,None]-1
            lp=F.grid_sample(O['edge'],grid,align_corners=True,padding_mode='zeros').permute(0,2,3,1)
            n=sample['normal'];t=torch.stack((-n[...,1],n[...,0]),-1);normal=(lp[...,1:]*n).sum(-1);tangent=(lp[...,1:]*t).sum(-1)
            grouped=normal.reshape(len(b),8,3,12);variation=F.pad(grouped[...,1:]-grouped[...,:-1],(1,0)).flatten(-2)
            valid=sample['valid'];ld=torch.stack((normal,tangent,variation,valid.float()),-1)*valid[...,None]
            if local or evidence_off:ld=torch.zeros_like(ld)
            extra=self.edge_encode(torch.cat((localpix.flatten(-2),ld.flatten(-2)),-1))
        raw=self.vector_head(torch.cat((self.encoded,extra),-1));radius=.15*(O['box'][:,2:]-O['box'][:,:2]).clamp_min(1).norm(dim=-1)
        delta=radius[:,None,None]*raw/torch.sqrt(1+raw.square().sum(-1,keepdim=True))
        out.update(points=rep['base']+delta,delta=delta,raw_vector=raw)
        return out

def loss25(out,gt,valid,arm):
    if arm.startswith('gate_'):return objective(out,gt,valid,'unit_weighted')[:2]
    error=(out['points']-gt).norm(dim=-1)[valid].mean()
    return error,error

