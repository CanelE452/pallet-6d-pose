"""Frozen selected heads; exact baseline preservation in original pixel space."""
import argparse
import math
import cv2
import numpy as np
import torch
from common import *
from dope_adapter import FrozenDopeAdapter
from selection import load_heads,refined_delta
from train import KEYS

def nullable(a):
    a=np.asarray(a)
    if a.ndim:return [nullable(x) for x in a]
    return float(a) if np.isfinite(a) else None

class Predictor:
    def __init__(self,device='cuda',heads=None):
        verify_lock();self.selection=read(DOC/'SELECTION.json')
        assert self.selection['complete'] and self.selection['real_selection'] is False
        assert self.selection['protocol']['sha256']==sha(DOC/'PROTOCOL.json')
        assert self.selection['validation']['sha256']==sha(DOC/'VALIDATION_OUTPUTS.json')
        self.adapter=FrozenDopeAdapter(device=device);self.device=self.adapter.device
        allheads=load_heads(device)
        self.heads=allheads if heads is None else {k:allheads[k] for k in heads}
        del allheads
        from refiner import GenericPointRefiner
        self.bank=GenericPointRefiner(**CONFIG).displacements.detach().numpy()

    def delta(self,value,support,box,scale,raw_hw,arm):
        rule=self.selection['rules'][arm[0]];T=self.selection['temperatures'][arm]
        boxes=np.asarray(box,np.float32).reshape(1,4)
        valid=np.isfinite(boxes).all(-1)&(boxes[:,2:]>boxes[:,:2]).all(-1)
        safe=np.where(valid[:,None],boxes,np.array([0,0,1,1]))
        diag=np.maximum(np.linalg.norm(safe[:,2:]-safe[:,:2],axis=-1),1.)
        if arm[0]=='P':
            p=(torch.from_numpy(value)/T).softmax(-1).numpy()
            delta=(p[...,None]*self.bank[None,None]).sum(-2)*diag[:,None,None]
        else:delta=value*diag[:,None,None]
        delta=delta*float(rule['lam'])/np.asarray(scale)[None,None]
        frac=rule['max_move_image_diagonal_fraction']
        if frac is not None:
            cap=math.hypot(*raw_hw)*frac
            delta*=np.minimum(1.,cap/np.maximum(np.linalg.norm(delta,axis=-1),1e-12))[...,None]
        return np.where(support[...,None],delta,0.)[0]

    @torch.no_grad()
    def predict(self,image,arms=None,replay=False):
        names=list(self.heads) if arms is None else list(arms)
        base=self.adapter.infer(image,return_features=bool(names))
        results={};max_replay=0.
        if names:
            box=np.full(4,np.nan,np.float32) if base['bbox_net'] is None else np.asarray(base['bbox_net'],np.float32)
            batch=dict(p3=base['features'][0][None].to(torch.float16),p4=base['features'][1][None].to(torch.float16),
                points=torch.from_numpy(base['points_net'][None]).to(self.device),
                boxes=torch.from_numpy(box[None]).to(self.device),point_valid=torch.from_numpy(base['valid'][None]).to(self.device),
                input_shape=torch.tensor([base['input_shape'][-2:]],device=self.device))
            scale=np.diag(base['affine_input_to_net'])[:2]
            for name in names:
                out=self.heads[name](*(batch[k] for k in KEYS),lam=0)
                value=out['logits' if name[0]=='P' else 'delta_normalized'].cpu().numpy()
                support=out['point_support'].cpu().numpy();assert np.isfinite(value).all()
                delta=self.delta(value,support,box,scale,image.shape[:2],name)
                if replay:
                    rule=self.selection['rules'][name[0]]
                    expected=refined_delta(dict(value=value,support=support),name,self.selection['temperatures'][name],
                         dict(boxes=box[None],scale_xy=scale[None],raw_diagonal=np.array([math.hypot(*image.shape[:2])])),
                         rule['lam'],rule['max_move_image_diagonal_fraction'])[0]
                    diff=float(np.max(np.abs(delta-expected)));max_replay=max(max_replay,diff)
                    np.testing.assert_allclose(delta,expected,rtol=0,atol=1e-12)
                points=base['points_original'].copy()
                if base['bbox_original'] is not None:
                    usable=base['valid'][:8]&support[0]
                    points[:8][usable]+=delta[usable]
                assert np.array_equal(points[8],base['points_original'][8],equal_nan=True)
                assert np.array_equal(np.isfinite(points).all(-1),base['valid'])
                if self.selection['rules'][name[0]]['lam']==0:
                    assert np.array_equal(points,base['points_original'],equal_nan=True)
                results[name]=points
            base.pop('features')
        return base,results,max_replay

def dev():
    heldout=read(DOC/'SYNTHETIC_HELDOUT.json')
    assert heldout['complete'] and heldout['selection']==bound(DOC/'SELECTION.json')
    destination=RAW/'DEV_PREDICTIONS.json';assert not destination.exists()
    predictor=Predictor();items=read(DEV)['items'];rows=[];max_replay=0.
    for j,item in enumerate(items):
        path=ROOT/item['image_path'];data=path.read_bytes();h=hashlib.sha256(data).hexdigest()
        image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR);assert image is not None
        base,refined,replay=predictor.predict(image,replay=True);max_replay=max(max_replay,replay)
        key=str(path.resolve().relative_to(ROOT.resolve()))
        rows.append(dict(image_key=key,frame_id=item['frame_id'],session_id=item['session_id'],
            original_hw=list(image.shape[:2]),image_sha256=h,base_points=nullable(base['points_original']),
            point_valid=base['valid'].tolist(),box_original=nullable(base['bbox_original']) if base['bbox_original'] is not None else None,
            score=base['score'],confidence=nullable(base['confidence']),affine_input_to_net=base['affine_input_to_net'].tolist(),
            refined={k:nullable(v) for k,v in refined.items()}))
        if j%25==0:print('DOPE_DEV_INFER',j+1,len(items),flush=True)
    assert len(rows)==319
    payload=dict(complete=True,schema='dope_refiner_dev_predictions_v1',created_at=now(),
        coordinate_system='original_unpadded_pixels',protocol=bound(DOC/'PROTOCOL.json'),
        cache=bound(DOC/'SOURCE_CACHE_COMPLETE.json'),selection=bound(DOC/'SELECTION.json'),
        checkpoints={f'{a}{s}':bound(RAW/'runs'/f'seed{s}'/'paired_last.pt') for s in SEEDS for a in ('P','D')},
        code=[bound(HERE/name) for name in ('inference.py','evaluation.py')],frames=rows,
        selection_inference_replay_max_abs_px=max_replay,center_box_score_and_missing_preserved=True,
        real_accuracy_not_read=True,all_image_bytes_bound=True)
    write(destination,payload)
    write(DOC/'DEV_INFERENCE_COMPLETE.json',dict(complete=True,predictions=bound(destination),frames=319,
        selection_inference_replay_max_abs_px=max_replay,GT_inputs=False,backbone_retraining=False))

if __name__=='__main__':
    torch.set_num_threads(1);cv2.setNumThreads(1);print(gpu(),flush=True);dev()
