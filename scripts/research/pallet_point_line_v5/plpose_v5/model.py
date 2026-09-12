"""One differentiable per-instance graph: features+dims+K -> points/lines -> SE(3).
Detection, choosing a deployable dimension record, undistortion and source-frame
mapping are outside this core and must be verified by the repository adapter.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math
import torch
from torch import Tensor,nn
from .contracts import Observation
from .geometry import roi_affine,transform_points,rotation_6d,rotation_y
from .hough import Lattice,DHTLineHead,DirectLineHead,decode_modes
from .solver import Measurements,PointLineRefiner,SolverConfig

@dataclass
class ModelConfig:
    arm:str='hough'             # point / direct / hough
    in_channels:int=32
    channels:int=32
    grid:int=24
    theta_bins:int=36
    rho_bins:int=65
    line_modes:int=3
    starts:int=4
    condition_dims:bool=True
    solver_iterations:int=8
    line_weight:float=.25
    point_sigma_fraction:float=.005
    line_sigma_fraction:float=.005

class PointLinePoseModel(nn.Module):
    def __init__(self,config:ModelConfig|None=None):
        super().__init__();self.config=config or ModelConfig();c=self.config
        if c.arm not in ('point','direct','hough') or c.channels%4 or c.starts<1:raise ValueError('Invalid model config')
        self.visual=nn.Sequential(nn.Conv2d(c.in_channels+2,c.channels,3,padding=1),nn.GroupNorm(4,c.channels),nn.SiLU(),
                                  nn.Conv2d(c.channels,c.channels,3,padding=1),nn.GroupNorm(4,c.channels),nn.SiLU())
        self.condition=nn.Sequential(nn.Linear(7,c.channels),nn.SiLU(),nn.Linear(c.channels,2*c.channels))
        nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)
        self.point_map=nn.Conv2d(c.channels,9,1)
        self.point_offset=nn.Linear(c.channels,18)
        nn.init.zeros_(self.point_offset.weight);nn.init.zeros_(self.point_offset.bias)
        # All arms use IMAGE-derived multi-start poses; not a point-PnP gate.
        self.seed_head=nn.Sequential(nn.Linear(c.channels+7,c.channels),nn.SiLU(),nn.Linear(c.channels,c.starts*9))
        nn.init.zeros_(self.seed_head[-1].weight)
        bias=torch.zeros(c.starts,9)
        for i in range(c.starts):
            # Object +y up to camera y down; front face points toward camera.
            R=rotation_y(torch.tensor(2*math.pi*i/c.starts))@torch.diag(torch.tensor([1.,-1.,-1.]))
            bias[i,:6]=R[:,:2].T.flatten();bias[i,8]=0.
        with torch.no_grad():self.seed_head[-1].bias.copy_(bias.flatten())
        self.lattice=Lattice(c.grid,c.grid,c.theta_bins,c.rho_bins)
        self.line_head=(DHTLineHead(c.channels,self.lattice) if c.arm=='hough' else
                        DirectLineHead(c.channels,self.lattice) if c.arm=='direct' else None)
        self.refiner=PointLineRefiner(SolverConfig(iterations=c.solver_iterations,
                                                 line_weight=c.line_weight if c.arm!='point' else 0.))

    def forward(self,obs:Observation):
        obs.validate();c=self.config;b=obs.features.shape[0]
        if obs.features.shape[1:]!=(c.in_channels,c.grid,c.grid):raise ValueError('ROI feature shape must match model config')
        A=roi_affine(obs.box)
        xy=self.lattice.xy.T.reshape(1,2,c.grid,c.grid).expand(b,-1,-1,-1)
        visual=self.visual(torch.cat((obs.features*obs.content,xy),1))
        wh=obs.image_hw[:,[1,0]]
        dimlog=obs.dims.log() if c.condition_dims else torch.zeros_like(obs.dims)
        camera=torch.stack((obs.K[:,0,0]/wh[:,0],obs.K[:,1,1]/wh[:,1],
                            obs.K[:,0,2]/wh[:,0],obs.K[:,1,2]/wh[:,1]),-1)
        condition=torch.cat((dimlog,camera),-1)
        gamma,beta=self.condition(condition).chunk(2,-1)
        visual=(visual*(1+gamma[:,:,None,None])+beta[:,:,None,None])*obs.content
        pooled=visual.sum((-2,-1))/obs.content.sum((-2,-1)).clamp_min(1)
        heat=self.point_map(visual).flatten(2).masked_fill(~obs.content.flatten(2).bool(),-1e4)
        p=torch.softmax(heat,-1)@self.lattice.xy
        # Unbounded learned offsets allow amodal points OUTSIDE the ROI/image.
        # This head is new and is NOT numerically identical to the stock YOLO head.
        p=p+self.point_offset(pooled).reshape(b,9,2)
        points=transform_points(p,torch.linalg.inv(A))
        diag=torch.linalg.vector_norm(wh,dim=-1)
        ps=(c.point_sigma_fraction*diag).clamp_min(.5)[:,None].expand(-1,9)
        point_valid=torch.ones(b,9,dtype=torch.bool,device=points.device)
        if self.line_head is not None:
            logits,bin_valid=self.line_head(visual,obs.content)
            lines,lp,lv=decode_modes(logits,bin_valid,self.lattice,A,c.line_modes)
        else:
            logits=points.new_zeros(b,12,c.theta_bins*c.rho_bins)
            bin_valid=torch.zeros_like(logits,dtype=torch.bool)
            lines=points.new_zeros(b,12,c.line_modes,3);lp=points.new_zeros(b,12,c.line_modes)
            lv=torch.zeros_like(lp,dtype=torch.bool)
        ls=(c.line_sigma_fraction*diag).clamp_min(.5)[:,None,None].expand(-1,12,c.line_modes)
        measurement=Measurements(points,point_valid,ps,lines,lv,lp,ls)
        seed=self.seed_head(torch.cat((pooled,condition),-1)).reshape(b,c.starts,9)
        R0=rotation_6d(seed[...,:6])
        # Image-scale depth proposal, with object-radius minimum so all starts
        # have positive depth. No GT crop, pose or point availability required.
        boxwh=obs.box[:,2:]-obs.box[:,:2]
        size=obs.dims[:,:2].amax(-1)
        focal=(obs.K[:,0,0]+obs.K[:,1,1])*.5
        rough_depth=focal*size/boxwh.amax(-1).clamp_min(1)
        radius=torch.linalg.vector_norm(obs.dims,dim=-1)*.5
        depth=radius[:,None]+.05+rough_depth[:,None]*torch.nn.functional.softplus(seed[...,8])
        center=(obs.box[:,:2]+obs.box[:,2:])*.5
        raw_center=center[:,None]+seed[...,6:8]*boxwh[:,None]
        homogeneous=torch.cat((raw_center,torch.ones_like(raw_center[...,:1])),-1)
        ray=torch.einsum('bij,bmj->bmi',torch.linalg.inv(obs.K),homogeneous)
        t0=ray*depth[...,None]
        result=self.refiner(measurement,obs.dims,obs.K,R0,t0)
        result.update(measurements=measurement,line_logits=logits,line_bin_valid=bin_valid,
                      raw_to_grid=A,seed_R=R0,seed_t=t0,point_heatmap=heat,
                      scope='per-predicted-instance pose; detection coverage must be evaluated separately')
        return result


def initialize_matched(models:dict[str,PointLinePoseModel],seed:int=1):
    """Copy common tensors exactly. Private head tensors are deterministic, not equal-capacity."""
    reference=next(iter(models.values())).state_dict()
    common=[]
    for name in reference:
        if name.startswith('line_head.'):continue
        if all(name in m.state_dict() and m.state_dict()[name].shape==reference[name].shape for m in models.values()):
            common.append(name)
    for model in models.values():
        state=model.state_dict()
        for name in common:state[name]=reference[name].clone()
        model.load_state_dict(state,strict=True)
    return common


class ToyRGBBackbone(nn.Module):
    """For generated wiring tests ONLY. Not a YOLO/pallet benchmark baseline."""
    def __init__(self,out_channels=32,grid=24):
        super().__init__();self.grid=grid
        self.body=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,out_channels,3,stride=2,padding=1),nn.SiLU())
    def forward(self,rgb):
        return torch.nn.functional.adaptive_avg_pool2d(self.body(rgb),(self.grid,self.grid))
