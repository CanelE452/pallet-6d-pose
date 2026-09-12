"""Unrolled, piecewise differentiable point-and-line multi-start pose refinement.

Not EPro-PnP, not an exact probabilistic solver. Robust point residuals and a
shared-mode mixture for each edge's TWO endpoints. Correspondences fixed per
hypothesis; no free per-corner assignment. No GT, symmetry oracle or GT starts.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from .geometry import EDGES,cuboid,project_jacobian,update_pose,project
from .contracts import validate_rotation

@dataclass
class Measurements:
    points: Tensor       # B,9,2
    point_valid: Tensor  # B,9, predicted availability
    point_sigma: Tensor  # B,9, positive pixels; not a calibrated probability
    lines: Tensor        # B,12,Q,3; normalized homogeneous lines in raw pixels
    line_valid: Tensor   # B,12,Q
    line_logprob: Tensor # B,12,Q, normalised within valid modes
    line_sigma: Tensor   # B,12,Q, positive raw pixels

    def repeat(self,n):
        return Measurements(**{k:v.repeat_interleave(n,0) for k,v in vars(self).items()})

    def validate(self):
        b=self.points.shape[0]; q=self.lines.shape[2]
        shapes={'points':(b,9,2),'point_valid':(b,9),'point_sigma':(b,9),
                'lines':(b,12,q,3),'line_valid':(b,12,q),'line_logprob':(b,12,q),'line_sigma':(b,12,q)}
        for key,shape in shapes.items():
            if getattr(self,key).shape!=shape: raise ValueError(f'Invalid measurement shape: {key}')
        if not torch.isfinite(self.points[self.point_valid]).all(): raise ValueError('Nonfinite valid points')
        if not torch.isfinite(self.lines[self.line_valid]).all(): raise ValueError('Nonfinite valid lines')
        if not (self.point_sigma>0).all() or not (self.line_sigma>0).all(): raise ValueError('Positive sigmas required')
        if not torch.isfinite(self.point_sigma).all() or not torch.isfinite(self.line_sigma).all(): raise ValueError('Nonfinite sigma')
        if not torch.isfinite(self.line_logprob[self.line_valid]).all(): raise ValueError('Nonfinite valid logprob')
        norm=torch.linalg.vector_norm(self.lines[...,:2],dim=-1)
        if self.line_valid.any() and not torch.allclose(norm[self.line_valid],torch.ones_like(norm[self.line_valid]),atol=1e-4):
            raise ValueError('Normalize line normals; sigma units depend on it')
        return self

@dataclass
class SolverConfig:
    iterations: int=8
    damping: float=.01
    line_weight: float=.25
    huber_delta: float=3.
    rotation_step: float=.35
    translation_step: float=.35
    min_depth: float=.02

    def __post_init__(self):
        if self.iterations<0 or min(self.damping,self.huber_delta,self.rotation_step,self.translation_step,self.min_depth)<=0 or self.line_weight<0:
            raise ValueError('Invalid solver configuration')


def huber(x:Tensor,delta:float):
    a=x.abs()
    return torch.where(a<=delta,.5*x.square(),delta*(a-.5*delta))


def irls(x:Tensor,delta:float):
    return (delta/x.abs().clamp_min(delta))


def linearize(R:Tensor,t:Tensor,X:Tensor,K:Tensor,m:Measurements,cfg:SolverConfig):
    """Return residual/J/weights and exact objective used for line-search.
    Both endpoints share one latent line mode. The mixture weights are an
    IRLS/EM local surrogate; line-search checks the actual robust mixture.
    """
    uv,z,J=project_jacobian(X,R,t,K)
    b=uv.shape[0]
    valid=m.point_valid[:,:8]
    observed=torch.where(valid[...,None],m.points[:,:8],torch.zeros_like(m.points[:,:8]))
    sigma=m.point_sigma[:,:8].clamp_min(1e-6)
    rp=(uv[:,:8]-observed)/sigma[...,None]
    Jp=J[:,:8]/sigma[...,None,None]
    denom=(valid.sum(-1)*2).clamp_min(1).to(uv.dtype)
    wp=irls(rp,cfg.huber_delta)*valid[...,None]/denom[:,None,None]
    point_energy=(huber(rp,cfg.huber_delta)*valid[...,None]).sum((-2,-1))/denom
    # GT center is supervised in the head but not counted as an independent
    # extra geometric correspondence in the solver.
    e=torch.tensor(EDGES,device=uv.device)
    endpoints=uv[:,e];Je=J[:,e]                     # B,E,2,2 and B,E,2,2,6
    lv=m.line_valid
    line=torch.where(lv[...,None],m.lines,torch.zeros_like(m.lines))
    sl=m.line_sigma.clamp_min(1e-6)
    rl=(torch.einsum('bead,beqd->beqa',endpoints,line[...,:2])+line[...,2,None])/sl[...,None]
    Jl=torch.einsum('beadk,beqd->beqak',Je,line[...,:2])/sl[...,None,None]
    mode_cost=huber(rl,cfg.huber_delta).sum(-1)/2
    logprior=torch.where(lv,m.line_logprob,torch.full_like(m.line_logprob,-1e4))
    logprior=logprior-torch.logsumexp(logprior,-1,keepdim=True)
    logmix=logprior-mode_cost
    responsibilities=torch.softmax(logmix,-1)*lv
    edge_valid=lv.any(-1)
    edge_den=edge_valid.sum(-1).clamp_min(1).to(uv.dtype)
    line_energy=(-torch.logsumexp(logmix,-1)*edge_valid).sum(-1)/edge_den
    wl=(responsibilities[...,None]*irls(rl,cfg.huber_delta)*cfg.line_weight
        / (2*edge_den[:,None,None,None]))
    residual=torch.cat((rp.reshape(b,-1),rl.reshape(b,-1)),-1)
    jacobian=torch.cat((Jp.reshape(b,-1,6),Jl.reshape(b,-1,6)),-2)
    weights=torch.cat((wp.reshape(b,-1),wl.reshape(b,-1)),-1)
    energy=point_energy+cfg.line_weight*line_energy
    constraints=valid.sum(-1)*2+(edge_valid.sum(-1)*2 if cfg.line_weight>0 else 0)
    feasible=(z>cfg.min_depth).all(-1)&torch.isfinite(energy)&(constraints>=6)
    return residual,jacobian,weights,energy,feasible


class PointLineRefiner(nn.Module):
    def __init__(self,config:SolverConfig|None=None):
        super().__init__();self.config=config or SolverConfig()

    def forward(self,measurements:Measurements,dims:Tensor,K:Tensor,R0:Tensor,t0:Tensor):
        """R0[B,M,3,3],t0[B,M,3]. Starts must be predicted, not supplied by GT."""
        measurements.validate();cfg=self.config
        b,n=R0.shape[:2]
        if R0.shape!=(b,n,3,3) or t0.shape!=(b,n,3) or not torch.isfinite(t0).all():
            raise ValueError('Invalid pose starts')
        validate_rotation(R0)
        X=cuboid(dims).repeat_interleave(n,0);kk=K.repeat_interleave(n,0)
        m=measurements.repeat(n)
        R=R0.reshape(-1,3,3);t=t0.reshape(-1,3)
        trace=[];accepted=[]
        I=torch.eye(6,dtype=R.dtype,device=R.device)
        for step in range(cfg.iterations):
            residual,J,w,E,valid=linearize(R,t,X,kk,m,cfg)
            trace.append(E.reshape(b,n))
            JT=J.transpose(-1,-2)
            H=JT@(w[...,None]*J)
            g=(JT@(w*residual)[...,None]).squeeze(-1)
            H=H+cfg.damping*torch.diag_embed(H.diagonal(dim1=-2,dim2=-1).clamp_min(1e-3))+1e-8*I
            delta=torch.linalg.solve(H,-g)
            dr,dt=delta[:,:3],delta[:,3:]
            dr=dr*(cfg.rotation_step/torch.linalg.vector_norm(dr,dim=-1,keepdim=True).clamp_min(cfg.rotation_step))
            dt=dt*(cfg.translation_step/torch.linalg.vector_norm(dt,dim=-1,keepdim=True).clamp_min(cfg.translation_step))
            delta=torch.cat((dr,dt),-1)
            bestR,bestt,bestE=R,t,E
            accepted_mask=torch.zeros_like(valid)
            # Keep masks/correspondences fixed; never hide inconvenient lines
            # by changing their visibility based on a tentative pose.
            for scale in (1.,.5,.25):
                rnew,tnew=update_pose(R,t,scale*delta)
                _,_,_,enew,ok=linearize(rnew,tnew,X,kk,m,cfg)
                take=ok & torch.isfinite(delta).all(-1) & (enew<bestE)
                bestR=torch.where(take[:,None,None],rnew,bestR)
                bestt=torch.where(take[:,None],tnew,bestt)
                bestE=torch.where(take,enew,bestE)
                accepted_mask|=take
            R,t=bestR,bestt
            accepted.append(accepted_mask.reshape(b,n))
        _,J,w,E,valid=linearize(R,t,X,kk,m,cfg)
        trace.append(E.reshape(b,n))
        information=J.transpose(-1,-2)@(w[...,None]*J)
        # Numerical rank is a diagnostic, not a posterior covariance or safety certificate.
        eig=torch.linalg.eigvalsh(information.detach())
        rank=(eig>eig[...,-1:].clamp_min(1e-12)*1e-6).sum(-1)
        ranks=rank.reshape(b,n)
        E=E.reshape(b,n); valid=valid.reshape(b,n)&(ranks==6)
        allR=R.reshape(b,n,3,3);allt=t.reshape(b,n,3)
        selection_cost=torch.where(valid,E,torch.full_like(E,1e9))
        index=selection_cost.argmin(-1)
        rows=torch.arange(b,device=R.device)
        chosenR,chosent=allR[rows,index],allt[rows,index]
        uv,depth=project(cuboid(dims),chosenR,chosent,K)
        return dict(R=chosenR,t=chosent,points=uv,depth=depth,
                    pose_valid=valid.any(-1),selected=index,energy=E,all_R=allR,all_t=allt,
                    candidates_valid=valid,information_rank=ranks,
                    objective_trace=torch.stack(trace,-1),
                    accepted=torch.stack(accepted,-1) if accepted else valid[...,None][:,:,:0],
                    differentiability='Unrolled LM/IRLS with piecewise line-search and hard final selection; no global-optimum guarantee.')
