"""Equal-capacity OLD_P replay with native RGB candidate evidence.

The detector/features remain frozen. All original corrector weights train;
all five arms contain identical RGB/scorer parameter slots. No GT reaches tiles.
"""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('M0','M1','M2','M3','M4')

def source_modules(root):
    root=Path(root).resolve()
    for p in (root/'scripts/research/pallet_final_ml_contribution_test_v1',
              root/'scripts/research/pallet_dim_conditioned_p_v1',root):
        sys.path.insert(0,str(p))
    import dcp_env, refiner
    from data import PaperData
    from generic_point_refiner import GenericPointRefiner, targets, decode, loss
    assert dcp_env.ROOT==root
    return dcp_env,refiner,PaperData,GenericPointRefiner,targets,decode,loss

class NativeEvidence(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
        self.rgb=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.SiLU(),
                               nn.Conv2d(16,16,3,padding=1),nn.SiLU())
        self.residual=nn.Sequential(nn.Linear(81,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)

    def forward(self,batch,use_rgb=False,tiles=None):
        capture={}
        hooks=[self.base.scorer[0].register_forward_pre_hook(lambda m,i:capture.update(regular=i[0])),
               self.base.null_scorer[0].register_forward_pre_hook(lambda m,i:capture.update(null=i[0]))]
        try:
            out=self.base(*(batch[k] for k in ('p3','p4','points','boxes','point_valid','input_shape')),lam=0)
        finally:
            for h in hooks:h.remove()
        regular=capture['regular'];null=capture['null']
        regular=torch.cat([regular,regular.new_zeros((*regular.shape[:-1],1))],-1)
        null=torch.cat([null,null.new_zeros((*null.shape[:-1],3)),null.new_ones((*null.shape[:-1],1))],-1)
        features=torch.cat([regular,null[:,:,None]],-2)
        b,k,c,_=features.shape
        if use_rgb:
            assert tiles is not None
            rgb,grid=tiles
            encoded=self.rgb(rgb)
            sampled=F.grid_sample(encoded,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
            rgb_features=sampled[:,:,0].transpose(1,2).reshape(b,k,c,16)
            rgb_features=rgb_features*torch.cat([features.new_ones(221),features.new_zeros(1)])[None,None,:,None]
        else:
            rgb_features=features.new_zeros((b,k,c,16))
        out['logits']=out['logits']+self.residual(torch.cat([features,rgb_features],-1)).squeeze(-1)
        return out

def native_tiles(data,rows,batch,displacements,device='cuda'):
    """Integer crops at original resolution, zero padding, pixel-center grid.

    Source prepared RGB is the existing reflected-border image; remove its
    border before cropping. Tile bounds use only predictions and candidate radius.
    No interpolation/resizing before the CNN. Mean-zero RGB outside the image.
    """
    import cv2
    from functools import lru_cache
    if not hasattr(data,'_rgb_reader'):
        @lru_cache(maxsize=64)
        def read_rgb(index):
            record=data.source['records'][int(index)]
            image=cv2.imread(record['image'],cv2.IMREAD_COLOR)
            if image is None:raise FileNotFoundError(record['id'])
            pad=record['reflect_pad_px'];h,w=record['raw_shape_hw']
            assert list(image.shape[:2])==record['prepared_shape_hw']
            return image[pad:pad+h,pad:pad+w,::-1].copy()
        data._rgb_reader=read_rgb
    points=batch['points'].detach().cpu().numpy()
    point_valid=batch['point_valid'].detach().cpu().numpy()
    diagonals=np.linalg.norm(batch['boxes'].detach().cpu().numpy()[:,2:]-batch['boxes'].detach().cpu().numpy()[:,:2],axis=-1)
    displacement=displacements.detach().cpu().numpy()
    tensors=[];positions=[];size_max=0
    for i,row in enumerate(rows):
        record=data.source['records'][int(data.indices[row])]
        image=data._rgb_reader(int(data.indices[row]));h,w=image.shape[:2]
        gain=float(data.arrays['gain'][row]);ph,pw=record['prepared_shape_hw']
        ih,iw=data.arrays['input_shape'][row]
        left=round((int(iw)-round(pw*gain))/2-0.1);top=round((int(ih)-round(ph*gain))/2-0.1)
        native=(points[i,:8]-np.array([left,top]))/gain-record['reflect_pad_px']
        candidates=diagonals[i]*displacement/gain
        radius=int(np.ceil(np.max(np.linalg.norm(candidates,axis=-1))))+4
        side=2*radius+1;size_max=max(size_max,side)
        for corner in range(8):
            point=native[corner]
            if not np.isfinite(point).all() or not point_valid[i,corner]:point=np.zeros(2)
            origin=np.floor(point).astype(int)-radius
            tile=np.zeros((side,side,3),np.float32)
            x0,y0=origin;x1,y1=x0+side,y0+side
            xa,ya=max(x0,0),max(y0,0);xb,yb=min(x1,w),min(y1,h)
            if xb>xa and yb>ya:
                tile[ya-y0:yb-y0,xa-x0:xb-x0]=(image[ya:yb,xa:xb].astype(np.float32)/255-.5)
            tensors.append(tile);positions.append(point[None]+candidates-origin)
    # Padding to the minibatch maximum preserves native pixel pitch exactly.
    if size_max>1024:raise RuntimeError('BLOCKED_RESOURCE native tile exceeds fixed1024 guard')
    rgb=np.zeros((len(tensors),3,size_max,size_max),np.float32)
    for i,tile in enumerate(tensors):rgb[i,:,:tile.shape[0],:tile.shape[1]]=tile.transpose(2,0,1)
    # Native points use pixel indices; align_corners=False needs +0.5 centers.
    grid=2*(np.stack(positions).astype(np.float32)+.5)/size_max-1
    return torch.from_numpy(rgb).to(device),torch.from_numpy(grid[:,None]).to(device)

def aux_mask(out,gt,valid,gain,raw_hw,targets):
    target=targets(out,gt,valid)
    residual=target['residual']/gain[:,None,None]
    radius=out['box_diagonal']*.08/gain
    # Convex hull of the 13 outermost spokes, each half-plane inclusive.
    theta=(torch.arange(13,device=residual.device,dtype=residual.dtype)+.5)*2*np.pi/13
    normals=torch.stack([theta.cos(),theta.sin()],-1)
    inside=(torch.einsum('bkd,ad->bka',residual,normals)<=radius[:,None,None]*np.cos(np.pi/13)+1e-6).all(-1)
    cap=raw_hw.to(residual).norm(dim=-1)*.01
    eligible=target['support']&inside&(residual.norm(dim=-1)<=cap[:,None]+1e-6)
    return eligible,target

def coordinate_loss(out,gt,eligible,gain,decode):
    expected=decode(out,1.,1.,None)[:,:8]
    error=(expected-gt[:,:8])/gain[:,None,None]
    error=torch.where(eligible[...,None],error,torch.zeros_like(error))
    values=F.smooth_l1_loss(error,torch.zeros_like(error),beta=1.,reduction='none').sum(-1)
    count=eligible.sum(-1)
    per=(values*eligible).sum(-1)/count.clamp_min(1)
    return per.sum()/(count>0).sum().clamp_min(1),error
