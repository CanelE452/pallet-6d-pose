"""Frozen fused S1, external independent per-scale residuals on final raw projection."""
import numpy as np
import torch
from torch import nn
from ultralytics import YOLO
from . import common as C

class Residual(nn.Module):
    def __init__(self,c,k):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(c,32,1),nn.SiLU(),nn.Conv2d(32,k,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        mask=torch.ones(1,k,1,1);mask[:,2::3]=0;self.register_buffer('xy_mask',mask)
    def forward(self,x):return self.net(x)*self.xy_mask

class Model:
    def __init__(self,load=False):
        C.setup();self.bindings=C.read(C.RAW/'INFERENCE_BINDINGS.json');C.verify(self.bindings['checkpoint']);self.net=YOLO(str(C.ROOT/self.bindings['checkpoint']['path']),task='pose')
        self.net.predict(np.zeros((680,840,3),np.uint8),conf=.001,imgsz=640,rect=True,half=False,device='cuda',verbose=False)
        self.net.model.eval()
        for p in self.net.model.parameters():p.requires_grad_(False)
        self.base_hash=C.state_hash(self.net.model);head=self.net.model.model[-1];assert type(head).__name__=='Pose26' and head.end2end and head.nk==27
        self.adapter=nn.ModuleList([Residual(layer.in_channels,layer.out_channels) for layer in head.one2one_cv4_kpts]).cuda();self.enabled=True;self.hooks=[]
        for layer,residual in zip(head.one2one_cv4_kpts,self.adapter):
            def add(module,args,out,residual=residual):return out+residual(args[0]) if self.enabled else out
            self.hooks.append(layer.register_forward_hook(add))
        if load:
            fit=C.read(C.DOC/'FIT.json');C.verify(fit['checkpoint']);self.adapter.load_state_dict(torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)['adapter']);self.adapter.eval()
    def integrity(self):
        assert C.state_hash(self.net.model)==self.base_hash
        assert all(p.grad is None and not p.requires_grad for p in self.net.model.parameters());C.verify(self.bindings['checkpoint']);C.verify(self.bindings['scorer'])
    def tensor(self,x,enabled=True):
        self.enabled=enabled;torch.backends.cudnn.allow_tf32=True
        # Stock eval decoder uses in-place sigmoid/slice writes: safe for inference,
        # but invalid autograd versioning when adapter coordinates require gradients.
        # Same Pose26 formula and arithmetic, implemented functionally during fitting.
        import types
        head=self.net.model.model[-1];original=head.kpts_decode
        def differentiable_decode(h,kpts):
            p=kpts.reshape(kpts.shape[0],9,3,-1)
            xx=(p[:,:,0]+h.anchors[0])*h.strides
            yy=(p[:,:,1]+h.anchors[1])*h.strides
            vv=p[:,:,2].sigmoid()
            return torch.stack((xx,yy,vv),dim=2).reshape_as(kpts)
        head.kpts_decode=types.MethodType(differentiable_decode,head)
        try:out=self.net.model(x)[0]
        finally:head.kpts_decode=original
        idx=out[:,:,4].argmax(1);selected=out[torch.arange(len(out),device=out.device),idx]
        return selected,selected[:,6:].reshape(-1,9,3)
    @torch.no_grad()
    def predict(self,image,padding=100,enabled=True):
        import cv2
        self.enabled=enabled;torch.backends.cudnn.allow_tf32=True
        im=cv2.copyMakeBorder(image,padding,padding,padding,padding,cv2.BORDER_REFLECT_101) if padding else image
        p=self.net.predict(im,conf=.001,imgsz=640,rect=True,augment=False,half=False,device='cuda',verbose=False,save=False)[0];rows=[]
        if p.boxes is not None:
            for i in range(len(p.boxes)):rows.append(dict(candidate_index=i,score=float(p.boxes.conf[i]),box_xyxy=(p.boxes.xyxy[i].cpu().numpy()-padding).tolist(),keypoints_xy=(p.keypoints.xy[i].cpu().numpy()-padding).tolist(),keypoints_conf=p.keypoints.conf[i].cpu().tolist()))
        return dict(candidates=rows,selected_index=int(np.argmax([r['score'] for r in rows])) if rows else None)
