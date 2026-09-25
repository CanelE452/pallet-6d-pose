"""Frozen expert inference + head-input capture, no real reference loading."""
import argparse
import time
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from . import common as C
from . import features as F

class Expert:
    def __init__(self,arm):
        self.arm=arm;self.binding=C.read(C.RAW/'BASE_INPUTS.json')['checkpoints'][arm];C.verify(self.binding)
        self.net=YOLO(str(C.ROOT/self.binding['path']),task='pose');self.context=None;self.hook=None;self.available=False
        self.net.predict(np.zeros((680,840,3),np.uint8),conf=.001,imgsz=640,rect=True,half=False,device='cuda',verbose=False)
        self.net.model.eval()
        for p in self.net.model.parameters():p.requires_grad_(False)
        self.initial=C.state_hash(self.net.model)
        head=self.net.model.model[-1]
        self.head_type=type(head).__name__
        if 'Pose' in self.head_type:
            def capture(module,args):
                x=args[0]
                if isinstance(x,(list,tuple)) and all(isinstance(t,torch.Tensor) and t.ndim==4 for t in x):
                    self.context=torch.cat([v.detach().mean((2,3)) for v in x],1).cpu().numpy()
                else:self.context=None
            self.callback=capture;self.hook=head.register_forward_pre_hook(capture);self.available=True

    @torch.no_grad()
    def run(self,images):
        torch.backends.cudnn.allow_tf32=True
        self.context=None
        results=self.net.predict(images,conf=.001,imgsz=640,rect=True,augment=False,half=False,device='cuda',verbose=False,save=False)
        preds=[]
        for p in results:
            rows=[]
            if p.boxes is not None:
                for i in range(len(p.boxes)):
                    rows.append(dict(candidate_index=i,score=float(p.boxes.conf[i]),box_xyxy=p.boxes.xyxy[i].cpu().tolist(),keypoints_xy=p.keypoints.xy[i].cpu().tolist(),keypoints_conf=p.keypoints.conf[i].cpu().tolist()))
            preds.append(dict(candidates=rows,selected_index=int(np.argmax([r['score'] for r in rows])) if rows else None))
        context=None if self.context is None else self.context.copy()
        return preds,context

    def integrity(self):
        assert C.state_hash(self.net.model)==self.initial
        assert all(p.grad is None and not p.requires_grad for p in self.net.model.parameters())
        C.verify(self.binding)

    def capture_test(self,image):
        on,feat=self.run([image]);again,feat2=self.run([image]);assert on==again
        if self.hook:
            assert feat is not None and np.array_equal(feat,feat2)
            self.hook.remove();off,_=self.run([image]);assert on==off
            self.hook=self.net.model.model[-1].register_forward_pre_hook(self.callback)
        self.integrity()
        return dict(arm=self.arm,head=self.head_type,available=feat is not None,feature_dim=0 if feat is None else feat.shape[1],
            hook_on_off_prediction_exact=True,repeated_feature_exact=True,base_state_unchanged=True,base_grad_zero=True,base_optimizer_steps=0)

def main(mode='synth',batch=16):
    C.setup();start=time.perf_counter();gpu=C.gpu()
    if mode=='synth':rows=C.read(C.sraw(2)/'SYNTH_INPUTS.json');output=C.sraw(2)/'FEATURES_CLEAN.npz';predpath=C.sraw(2)/'PREDICTIONS_CLEAN.json'
    else:
        # Metadata is inference-only K/dimensions; split provides image paths, not labels.
        rr=C.read(C.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json')['heldout']
        meta={r['id']:r for r in C.read(C.ROOT/'data/pallet/results/pallet_visible_refine_hidden_pnp_v1/INFERENCE_METADATA.json')}
        rows=[dict(id=r['id'],image=r['image'],K=meta[r['id']]['K'],dims=meta[r['id']]['xyz']) for r in rr]
        output=C.RAW/'REAL_FEATURES.npz';predpath=C.RAW/'REAL_FEATURE_DIAGNOSTICS.json';batch=1
    assert not output.exists()
    arrays={};allpred={};audits=[];timings={}
    existing=C.read(C.PREV_RAW/'PREDICTIONS.json') if mode=='real' else None
    for arm in C.ARMS:
        model=Expert(arm);first=cv2.imread(str(C.ROOT/rows[0]['image']['path']))
        if mode=='real':first=cv2.copyMakeBorder(first,100,100,100,100,cv2.BORDER_REFLECT_101)
        audits.append(model.capture_test(first));geo=[];contexts=[];valid=[];selections=[];allpred[arm]={};begin=time.perf_counter()
        for off in range(0,len(rows),batch):
            if off%256==0:C.gpu();print('EXTRACT',mode,arm,off,'/',len(rows),flush=True)
            items=rows[off:off+batch];ims=[cv2.imread(str(C.ROOT/r['image']['path'])) for r in items]
            nativehws=[im.shape[:2] for im in ims]
            if mode=='real':ims=[cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_REFLECT_101) for im in ims]
            try:pp,cc=model.run(ims)
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError('OOM: rerun with smaller inference batch only; preserve protocol')
            for j,(r,p) in enumerate(zip(items,pp)):
                if mode=='real':
                    for c in p['candidates']:
                        c['keypoints_xy']=(np.array(c['keypoints_xy'])-100).tolist();c['box_xyxy']=(np.array(c['box_xyxy'])-100).tolist()
                    # Keep original cached raw predictions; hook is context-only.
                    old=existing[arm][r['id']];oc,nc=C.selected(old),C.selected(p)
                    assert (oc is None)==(nc is None)
                    if oc is not None:assert np.allclose(oc['keypoints_xy'],nc['keypoints_xy'],atol=1e-4,rtol=1e-6),'Frozen prediction context parity'
                    p=old
                f=F.extract(p,r['K'],r['dims'],nativehws[j]);geo.append(f['features'] if f['valid'] else np.zeros((2,len(F.names()))));valid.append(f['valid']);contexts.append(cc[j] if cc is not None else np.zeros(0));selections.append(C.HYP.index(f['selection']) if f['selection'] in C.HYP else -1)
                allpred[arm][r['id']]=dict(prediction=p,geometry=f,hw=nativehws[j])
        model.integrity();timings[arm]=time.perf_counter()-begin
        arrays[arm+'_geo']=np.array(geo,np.float32);arrays[arm+'_ctx']=np.array(contexts,np.float32);arrays[arm+'_valid']=np.array(valid);arrays[arm+'_current']=np.array(selections)
        del model;torch.cuda.empty_cache()
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,ids=np.array([r['id'] for r in rows]),**arrays)
    C.freeze(predpath,allpred)
    lock=dict(created_at=C.now(),features=C.bind(output),predictions=C.bind(predpath),GT_input=False,feature_contract=C.bind(C.DOC/'SELECTOR_FEATURE_CONTRACT.json'),
        captures=audits,feature_mode='GEO_IMG' if all(a['available'] for a in audits) else 'FROZEN_FEATURE_UNAVAILABLE',
        frames=len(rows),batch=batch,seconds=time.perf_counter()-start,seconds_per_expert=timings,gpu_start=gpu,gpu_end=C.gpu(),base_weights_unchanged=True)
    C.freeze(C.sdoc(2)/'SYNTH_PREDICTION_LOCK.json' if mode=='synth' else C.DOC/'REAL_FEATURE_LOCK.json',lock)
    print('EXTRACTION_COMPLETE',mode,lock['feature_mode'],lock['seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['synth','real']);p.add_argument('--batch',type=int,default=16);a=p.parse_args();main(a.mode,a.batch)
