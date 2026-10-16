"""One small 1D correspondence network, three input ablations; no GT interface."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .source_audit import EDGES
from .solver import HypothesisBank,visibility

ARMS=('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE')
CONFIG=dict(channels=28,width=32,convolution_blocks=2,kernel_size=3,
    queries_per_edge=7,edges=12,candidates=65,no_match_index=65,
    search_original_pixels=32,image_channels=list(range(19)),role_channels=[25,26,27],
    detector_channels='fixed mean groups: 8 P3 + 8 P4; frozen FP16 neck',
    image_stencil='brightness and tangent/normal Sobel derivatives at each of 65 centers',
    feature_stencil='bilinear at each candidate; zero padding; align_corners=False',
    decoding='argmax 65+none, integer1px bin; no cross-basin averaging',
    geometry='normal offset/32, normalized center x/y, tangent x/y, edge length/diagonal',
    role='predicted projected convex-hull boundary / internal / unavailable')

class CorrespondenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv1d(28,32,3,padding=1),nn.ReLU(),nn.Conv1d(32,32,3,padding=1),nn.ReLU())
        self.score=nn.Conv1d(32,1,1);self.none=nn.Linear(32,1)
    def forward(self,x,arm):
        assert arm in ARMS and x.shape[-2:]==(28,65)
        x=x.clone()
        if arm=='GEOMETRY_ONLY':x[...,:19,:]=0
        if arm=='IMAGE_NO_ROLE':x[...,25:28,:]=0
        shape=x.shape[:-2];h=self.body(x.reshape(-1,28,65))
        scores=self.score(h)[:,0];none=self.none(h.mean(-1))
        return torch.cat([scores,none],-1).reshape(*shape,66)

def query_geometry(points):
    points=np.asarray(points,float);centers=[];normals=[];tangents=[];ids=[];valid=[]
    for edge,(a,b) in enumerate(EDGES):
        vector=points[b]-points[a];length=float(np.linalg.norm(vector));tangent=vector/max(length,1e-6)
        normal=np.array([-tangent[1],tangent[0]])
        for u in np.arange(1,8)/8:
            centers.append((1-u)*points[a]+u*points[b]);normals.append(normal);tangents.append(tangent)
            ids.append([edge,a,b,float(u),length]);valid.append(bool(np.isfinite(vector).all() and length>1e-6 and not np.any(np.all(points[[a,b]]==-1,axis=1))))
    center=np.nan_to_num(np.array(centers),nan=0.,posinf=0.,neginf=0.);normal=np.nan_to_num(np.array(normals));tangent=np.nan_to_num(np.array(tangents))
    candidate=center[:,None,:]+np.arange(-32,33)[None,:,None]*normal[:,None,:]
    return dict(center=center,normal=normal,tangent=tangent,candidate=candidate,identity=ids,valid=np.array(valid))

def initial_geometry(points,K,xyz,hw):
    bank=HypothesisBank(points,K,xyz,image_size=(hw[1],hw[0]));pose=bank.solve(robust=False)
    role=np.zeros((12,3),np.float32);hidden=[]
    if pose['available']:
        projected=np.array(pose['projected']);hull=cv2.convexHull(projected.astype(np.float32),returnPoints=False).ravel().tolist()
        pairs={frozenset([hull[i],hull[(i+1)%len(hull)]]) for i in range(len(hull))}
        for e,(a,b) in enumerate(EDGES):role[e,0 if frozenset([a,b]) in pairs else 1]=1
        model=bank.models[pose['dimension_index']] if 'dimension_index' in pose else __import__(__package__+'.solver',fromlist=['cuboid']).cuboid(*pose['cf_extents'])
        h,_,_=visibility(model,np.array(pose['R_cf']),np.array(pose['centroid']));hidden=np.flatnonzero(h).tolist()
    else:role[:,2]=1
    return pose,role,hidden

def inputs(image,captured,points,K,xyz):
    """GT-free image/features/initial predictions -> candidate tensor and metadata."""
    from . import common as C
    C.source_modules()
    from scripts.research.pallet_line_pose_v1.features import canvas_affine
    q=query_geometry(points);h,w=image.shape[:2];device=captured['p3'].device
    pose,roles,hidden=initial_geometry(points,K,xyz,[h,w]);cc=q['candidate'];x=cc[:,:,0].astype(np.float32);y=cc[:,:,1].astype(np.float32)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/4;gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/4
    def sample(a):return cv2.remap(a,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    sx,sy=sample(gx),sample(gy);normal=q['normal'];tangent=q['tangent']
    rgb=np.stack([sample(gray)*2-1,sx*normal[:,0,None]+sy*normal[:,1,None],sx*tangent[:,0,None]+sy*tangent[:,1,None]],1)
    gain,offset=canvas_affine(captured['canvas_shape'],captured['input_shape']);pad=captured['added_border']
    uv=(cc+pad)*gain+offset;grid=torch.as_tensor(uv/640*2-1,dtype=torch.float32,device=device)[None]
    neck=[]
    for name in ['p3','p4']:
        f=captured[name].float()
        if f.ndim==3:f=f[None]
        assert f.shape[1]%8==0
        f=f.reshape(1,8,f.shape[1]//8,*f.shape[-2:]).mean(2)
        sampled=F.grid_sample(f,grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0].permute(1,0,2)
        neck.append(sampled)
    geom=np.zeros((84,6,65),np.float32);geom[:,0,:]=np.arange(-32,33)/32
    geom[:,1,:]=q['center'][:,0,None]/w;geom[:,2,:]=q['center'][:,1,None]/h
    geom[:,3:5,:]=q['tangent'][:,:,None];geom[:,5,:]=np.array(q['identity'])[:,4,None]/np.hypot(w,h)
    role=np.repeat(roles,7,axis=0)[:,:,None]*np.ones((1,1,65),np.float32)
    x=torch.cat([torch.as_tensor(rgb,device=device),*neck,torch.as_tensor(geom,device=device),torch.as_tensor(role,device=device)],1)
    # The shared source cache stores the entire candidate tensor as FP16. Apply
    # the identical rounding during live inference, then compute the head FP32.
    x=x.to(torch.float16).float()
    q.update(initial_pose=pose,hidden_initial=hidden,role=roles.tolist(),feature_shape=list(x.shape))
    return x,q

def loss(logits,lo,hi,weight,valid):
    log=logits.log_softmax(-1);lo=lo.clamp(min=0);hi=hi.clamp(min=0)
    ce=-(1-weight)*log.gather(-1,lo[...,None])[...,0]-weight*log.gather(-1,hi[...,None])[...,0]
    count=valid.sum(-1);per=(ce*valid).sum(-1)/count.clamp(min=1)
    return per[count>0].mean() if (count>0).any() else logits.sum()*0
