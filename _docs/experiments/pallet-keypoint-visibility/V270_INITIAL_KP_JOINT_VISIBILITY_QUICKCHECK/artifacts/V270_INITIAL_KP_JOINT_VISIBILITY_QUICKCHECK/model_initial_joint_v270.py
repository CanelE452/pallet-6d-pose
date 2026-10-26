"""Actual R0 full-image one2one pose stem with a separate semantic E readout."""
import os, sys, json, hashlib, types
from pathlib import Path
import numpy as np
W = Path(__file__).parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(W/'ultralytics_runtime'))
import torch
from torch import nn
import torch.nn.functional as F
import cv2
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.loss import KeypointLoss, PoseLoss26
ARMS=('XY_ONLY','VIS_ONLY','JOINT')
R0=Path('/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt')
R0_SHA='970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
def safe(p):
    p=Path(p)
    assert p.is_absolute() and not any(x in str(p).lower() for x in ('v112','cache','.pyc','..')),str(p)
    for a in [p,*p.parents]: assert not a.is_symlink(),str(a)
    return p
def binding(p):
    p=safe(p); h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)
def verify(b):
    a=binding(b['path']); assert a==b,(a,b); return Path(b['path'])
def dump(p,d): safe(p).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def load_code(p,name,sha=None):
    p=safe(p); b=p.read_bytes()
    if sha: assert hashlib.sha256(b).hexdigest()==sha
    m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m
    exec(compile(b,str(p),'exec'),m.__dict__);return m
def state_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode()); a=v.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def prepare_rgb(rgb):
    """Whole RGB reflect101 pad100 then square640 letterbox. Return tensor and affine."""
    h,w=rgb.shape[:2];p=cv2.copyMakeBorder(rgb,100,100,100,100,cv2.BORDER_REFLECT_101)
    r=min(640/p.shape[0],640/p.shape[1]);nw,nh=round(p.shape[1]*r),round(p.shape[0]*r)
    left,top=round((640-nw)/2-.1),round((640-nh)/2-.1)
    out=np.full((640,640,3),114,np.uint8);out[top:top+nh,left:left+nw]=cv2.resize(p,(nw,nh),interpolation=cv2.INTER_LINEAR)
    affine=np.array([[r,0,100*r+left],[0,r,100*r+top]],np.float32)
    return torch.from_numpy(out.transpose(2,0,1).copy()).float()/255,affine
class InitialKPJoint(nn.Module):
    def __init__(self,arm='JOINT',device='cpu'):
        super().__init__();assert arm in ARMS;assert binding(R0)['sha256']==R0_SHA
        ck=torch.load(safe(R0),map_location='cpu',weights_only=False)
        self.base=(ck.get('ema') or ck['model']).float().eval()
        self.base.args=types.SimpleNamespace(**dict(self.base.args)) if isinstance(self.base.args,dict) else self.base.args
        for p in self.base.parameters():p.requires_grad_(False)
        self.head=self.base.model[-1];assert self.head.kpt_shape==[9,3] and self.head.end2end
        torch.manual_seed(17)
        self.e_heads=nn.ModuleList(nn.Conv2d(45,9,1) for _ in range(3))
        for head in self.e_heads: nn.init.normal_(head.weight,0,.01);nn.init.zeros_(head.bias)
        self.arm=arm
        for p in self.head.one2one_cv4.parameters():p.requires_grad_(True)
        for p in self.e_heads.parameters():p.requires_grad_(arm!='XY_ONLY')
        self.to(device);self.train(False)
    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'base'):self.base.eval()
        return self
    def extract_pyramid(self,x):
        ys=[]
        for layer in self.base.model[:-1]:
            if layer.f!=-1:x=ys[layer.f] if isinstance(layer.f,int) else [x if j==-1 else ys[j] for j in layer.f]
            x=layer(x);ys.append(x if layer.i in self.base.save else None)
        return [ys[j] for j in self.head.f]
    def forward_pyramid(self,features,with_det=True):
        lat=[self.head.one2one_cv4[i](x) for i,x in enumerate(features)]
        k=torch.cat([self.head.one2one_cv4_kpts[i](x).flatten(2) for i,x in enumerate(lat)],2)
        e=torch.cat([self.e_heads[i](x).flatten(2) for i,x in enumerate(lat)],2)
        anchors,strides=make_anchors(features,self.head.stride,.5)
        raw=k.permute(0,2,1).reshape(k.shape[0],-1,9,3)
        xy=(raw[...,:2]+anchors[None,:,None,:])*strides[None,:,None,:]
        packet=dict(raw_kpts=k,raw9=raw,E_logits=e.permute(0,2,1),xy_px=xy,anchors=anchors,strides=strides)
        if with_det:
            with torch.no_grad():packet['det']=Detect.forward_head(self.head,features,self.head.one2one_cv2,self.head.one2one_cv3)
        return packet
    def forward(self,images):return self.forward_pyramid(self.extract_pyramid(images))
    def head_state(self):return {'cv4':self.head.one2one_cv4.state_dict(),'E_heads':self.e_heads.state_dict()}
    def load_head_state(self,s):
        self.head.one2one_cv4.load_state_dict(s['cv4'],strict=True);self.e_heads.load_state_dict(s['E_heads'],strict=True)
    def frozen_sha(self):
        return state_sha({k:v for k,v in self.base.state_dict().items() if not k.startswith('model.'+str(len(self.base.model)-1)+'.one2one_cv4.')})
def selected(packet,anchor):
    ix=torch.arange(packet['raw9'].shape[0],device=anchor.device)
    return dict(xy_px=packet['xy_px'][ix,anchor],canvas_logit=packet['raw9'][ix,anchor,:,2],E_logit=packet['E_logits'][ix,anchor])
def losses(packet,targets,assigned,bbox,weights,arm):
    keep=assigned>=0;zero=packet['raw_kpts'].sum()*0+packet['E_logits'].sum()*0
    if not keep.any():return dict(total=zero,canvas=zero,xy=zero,E=zero)
    ix=torch.where(keep)[0];a=assigned[keep];raw=packet['raw9'][ix,a];s=packet['strides'][a]
    canvas=F.binary_cross_entropy_with_logits(raw[...,2],targets['canvas_obj9'][keep].float())
    predxy=raw[...,:2]+packet['anchors'][a,None,:]
    gt=targets['xy_px'][keep]/s[:,None,:]
    area=((bbox[keep,2:]-bbox[keep,:2])/s).prod(-1,keepdim=True)
    xy=KeypointLoss(torch.ones(9,device=raw.device)/9)(predxy,gt,targets['loc_valid9'][keep],area)
    el=packet['E_logits'][ix,a];ey=targets['E9'][keep].float();mask=targets['Emask9'][keep]
    eloss=F.binary_cross_entropy_with_logits(el,ey.clamp_min(0),reduction='none')
    ew=torch.where(ey>0,torch.as_tensor(weights[1],device=raw.device),torch.as_tensor(weights[0],device=raw.device))
    E=(eloss*ew*mask).sum()/mask.sum().clamp_min(1)
    total=canvas+ (12*xy if arm!='VIS_ONLY' else zero)+(E if arm!='XY_ONLY' else zero)
    return dict(total=total,canvas=canvas,xy=xy,E=E)
