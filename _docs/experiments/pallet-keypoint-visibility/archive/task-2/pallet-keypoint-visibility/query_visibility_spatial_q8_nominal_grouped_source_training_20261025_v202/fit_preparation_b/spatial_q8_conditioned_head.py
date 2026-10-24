"""Original RAW direct head plus zero-initialized native/mask/q8 residual.
Only random invented CPU instantiation is permitted before root actual fit entry.
"""
import torch
from torch import nn
from torch.nn import functional as F

def require(v,message):
    if not bool(v): raise AssertionError(message)

def validate_context(native,other_mask,other_xy,n,device):
    require(isinstance(native,torch.Tensor) and native.dtype==torch.int64 and native.shape==(n,) and native.device==device,'Original native int64 same-take')
    require(((native>=0)&(native<8)).all(),'Original native0..7 without remap')
    require(isinstance(other_mask,torch.Tensor) and other_mask.dtype==torch.float32 and other_mask.shape==(n,8) and other_mask.device==device,'Same-take FP32 other-mask8')
    require(isinstance(other_xy,torch.Tensor) and other_xy.dtype==torch.float32 and other_xy.shape==(n,16) and other_xy.device==device,'Same-take FP32 other-XY16')
    require(torch.isfinite(other_mask).all() and ((other_mask==0)|(other_mask==1)).all(),'Binary finite context mask')
    require(torch.isfinite(other_xy).all() and (other_xy.abs()<=8).all(),'Finite prospectively clipped context')
    mask=other_mask.bool();xy=other_xy.reshape(n,8,2)
    require((~mask.gather(1,native[:,None])).all(),'Self context mask exactzero')
    require((xy[~mask]==0).all(),'Self/unavailable context XY exactzero')

def make_spatial_q8_head_class(existing_head_module,candidate_module):
    original=candidate_module.make_candidate_head_class(existing_head_module)
    class SpatialQ8DirectEndHead(original):
        def __init__(self,*,spatial_q8):
            require(type(spatial_q8)is bool,'Explicit controlled arm flag')
            super().__init__();self.spatial_q8=spatial_q8
            self.spatial_residual=nn.Sequential(nn.Linear(96,32),nn.SiLU(),nn.Linear(32,1))
            nn.init.zeros_(self.spatial_residual[-1].weight);nn.init.zeros_(self.spatial_residual[-1].bias)
        def residual_from_features(self,features,native,other_mask,other_xy):
            require(features.ndim==2 and features.shape[1]==64 and features.dtype==torch.float32 and torch.isfinite(features).all(),'Unchanged shared FP32 features64')
            validate_context(native,other_mask,other_xy,len(features),features.device)
            category=F.one_hot(native,num_classes=8).to(dtype=features.dtype)
            xy=other_xy if self.spatial_q8 else torch.zeros_like(other_xy)
            return self.spatial_residual(torch.cat((features,category,other_mask,xy),dim=1))[:,0]
        def forward(self,x,native,other_mask,other_xy):
            require(x.ndim==4 and x.shape[1:]==(8,96,96) and x.dtype==torch.float32 and torch.isfinite(x).all(),'Unchanged RAW8 local input')
            features=self.shared(self.pool(self.encoder(x)))
            return dict(support_logits=self.support_head(features),owner_logits=self.owner_head(features),direct_end_external_logit=self.direct_end_head(features)[:,0]+self.residual_from_features(features,native,other_mask,other_xy))
    return SpatialQ8DirectEndHead
