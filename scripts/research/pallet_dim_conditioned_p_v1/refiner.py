"""Unchanged visual P plus zero-effect canonical-metadata candidate residual."""
import numpy as np
import torch
from torch import nn
import dcp_env as E
from generic_point_refiner import GenericPointRefiner,decode,loss,targets

def dimension_features(dimensions):
    d=np.asarray(dimensions,np.float64)
    if d.shape[-1:]!=(3,) or not np.isfinite(d).all() or not (d>0).all():raise ValueError('Required canonical [W,D,H] must be finite positive metres')
    a=np.log(d);return np.stack([a[...,0],a[...,1],a[...,2],a[...,0]-a[...,1],a[...,2]-.5*(a[...,0]+a[...,1])],-1)

def context(dimensions,orders,normalization,meta):
    z=(dimension_features(dimensions)-np.array(normalization['mean']))/np.array(normalization['scale'])
    g=np.asarray(orders)
    if g.shape!=z.shape[:-1] or not np.isin(g,[1,2,4]).all():raise ValueError('Explicit approved C1/C2/C4 required; never inferred from dimensions')
    return np.concatenate([z,g[...,None]==[1,2,4]],-1).astype(np.float32) if meta else z.astype(np.float32)

class DimensionConditionedPointRefiner(GenericPointRefiner):
    def __init__(self,context_dim,**config):
        super().__init__(**config);assert context_dim in (5,8);self.context_dim=context_dim
        self.metadata_encoder=nn.Sequential(nn.Linear(context_dim,16),nn.SiLU(),nn.Linear(16,16),nn.SiLU())
        self.metadata_scorer=nn.Sequential(nn.Linear(27,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.metadata_scorer[-1].weight);nn.init.zeros_(self.metadata_scorer[-1].bias)
    def forward(self,p3,p4,points,boxes,point_valid,input_shape,*,context,lam=1.,temperature=1.,cap=None):
        if context is None or context.shape!=(len(points),self.context_dim) or not torch.isfinite(context).all():
            raise ValueError('Missing/nonfinite geometry context: explicit failure, no silent fill')
        o=super().forward(p3,p4,points,boxes,point_valid,input_shape,lam=0)
        b=len(points);embedding=self.metadata_encoder(context.to(self.role_embedding.weight))
        role=self.role_embedding.weight[None,:,None].expand(b,-1,222,-1)
        displacement=(self.displacements/.08)[None,None].expand(b,8,-1,-1)
        null=torch.zeros((b,8,222,1),device=points.device,dtype=embedding.dtype);null[:,:,-1]=1
        features=torch.cat([embedding[:,None,None].expand(-1,8,222,-1),role,displacement,null],-1)
        residual=self.metadata_scorer(features).squeeze(-1)
        o['base_logits']=o['logits'];o['metadata_residual']=residual;o['logits']=o['logits']+residual
        o['points']=decode(o,temperature,lam,cap);return o

def specification(arm):
    dim=8 if arm in ['N4_META_SYM','S2_META_SYM','M1_META_SYM'] else 5 if arm in ['N2_DIM_ONLY','N3_DIM_SYM','M0_DIM_SYM'] else 0
    sym=arm not in ['N0_BASE_REPLAY','N2_DIM_ONLY','S0_FIXED','OLD_P']
    return dim,sym

def model(arm,config):
    dim,_=specification(arm)
    return DimensionConditionedPointRefiner(dim,**config) if dim else GenericPointRefiner(**config)

def forward(head,batch):
    args=[batch[k] for k in ('p3','p4','points','boxes','point_valid','input_shape')]
    kw=dict(context=batch['context']) if isinstance(head,DimensionConditionedPointRefiner) else {}
    return head(*args,lam=0,**kw)

def local_phase(points,pred_valid,gt,valid,permutations,group_valid,diagonal):
    """One whole-object branch; first argmin gives identity-first deterministic tie.

    TRAIN/calibration supervision only. This function is never called by inference.
    Missing prediction receives a fixed normalized penalty of 1. No pointwise matches.
    """
    b=len(points);ii=torch.arange(b,device=gt.device)[:,None,None]
    g=gt[ii,permutations];v=valid[ii,permutations]&torch.isfinite(g).all(-1)&~(g==-1).all(-1)
    pvalid=pred_valid&torch.isfinite(points).all(-1)&~(points==-1).all(-1)
    dist=torch.linalg.vector_norm(points[:,None]-g,dim=-1)/diagonal[:,None,None].clamp_min(1)
    dist=torch.where(pvalid[:,None],dist,torch.ones_like(dist));dist=torch.where(v,dist,torch.zeros_like(dist))
    count=v[:,:,:8].sum(-1);cost=dist[:,:,:8].sum(-1)/count.clamp_min(1)
    cost=torch.where(group_valid,cost,torch.full_like(cost,float('inf')))
    branch=cost.argmin(-1);row=torch.arange(b,device=gt.device)
    return g[row,branch],v[row,branch],branch,cost

def train_loss(output,batch,sym):
    gt,valid=batch['gt_points'],batch['gt_valid']
    if sym:gt,valid,_,_=local_phase(output['points_raw'],output['point_valid'],gt,valid,batch['permutations'],batch['group_valid'],output['box_diagonal'])
    return loss(output,gt,valid)
