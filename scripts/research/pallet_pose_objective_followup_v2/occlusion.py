"""Shared REF-conditioned random placement; never change stored targets.

The canonical REF coordinates are transformed with the identical RNG stream.
They determine only mask feasibility for BOTH arms, not RAW target values.
"""
import copy
import hashlib
from pathlib import Path
import random
import numpy as np
import torch
from scripts.research.pallet_clean19_structured_easyhard_v1.augmentation import cover, overlap, fill

def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()

def restore(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])

def state_seed(state, name):
    return int.from_bytes(hashlib.sha256((repr(state[0])+repr(state[1])+name).encode()).digest()[:8], 'little')

def random_plan(points, mask, box, shape, seed):
    rng=np.random.default_rng(seed); p=np.asarray(points); mask=np.asarray(mask,bool).copy(); mask[8]=False
    area=float(np.prod(np.maximum(0,np.asarray(box)[2:]-np.asarray(box)[:2])))
    ratio=float(rng.choice([.5,1.,2.])); fraction=float(rng.choice([.1,.2,.3]))
    w=max(1,round(np.sqrt(area*fraction*ratio))); h=max(1,round(np.sqrt(area*fraction/ratio)))
    result=dict(seed=int(seed),scheduled=bool(rng.random()<.5),area_fraction=fraction,aspect=ratio,
        size=[w,h],fill_seed=int(rng.integers(0,2**31-1)),applied=False,reason='not_scheduled',
        rectangle=None,covered=[],remaining=int(mask.sum()),supervised=int(mask.sum()),candidates_checked=0)
    if not result['scheduled']: return result
    if mask.sum()<3: result['reason']='cannot_cover1_leave2'; return result
    if w>shape[1] or h>shape[0]: result['reason']='shape_exceeds_input'; return result
    for _ in range(32):
        l=int(rng.integers(0,shape[1]-w+1)); t=int(rng.integers(0,shape[0]-h+1)); rect=[l,t,w,h]
        covered=cover(p,rect)&mask; result['candidates_checked']+=1
        if covered.sum()>=1 and mask.sum()-covered.sum()>=2 and overlap(rect,box)>0:
            result.update(applied=True,reason='random_valid',rectangle=rect,
                covered=np.flatnonzero(covered).tolist(),remaining=int(mask.sum()-covered.sum()),
                bbox_fraction=overlap(rect,box)/max(area,1e-12))
            return result
    result['reason']='no_valid_random_position_32'; return result

class SharedOcclusion:
    def __init__(self, transform, reference_labels):
        self.transform=transform; self.reference_labels=reference_labels

    def __call__(self, labels):
        name=Path(labels['im_file']).name
        if name.startswith('syn__'):
            out=self.transform(labels)
            out['occlusion_info']=dict(role='SOURCE',applied=False,reason='source_unchanged')
            return out
        assert labels['instances'].normalized
        canonical=copy.deepcopy(labels)
        canonical['instances'].keypoints=self.reference_labels[name].copy()
        initial=rng_state()
        out=self.transform(labels); after=rng_state()
        try:
            restore(initial); ref=self.transform(canonical)
            assert torch.equal(out['img'],ref['img']), 'Canonical reference changed image transform'
            assert torch.equal(out['bboxes'],ref['bboxes']), 'Canonical reference changed boxes'
        finally: restore(after)
        info=dict(role='REAL',applied=False,reason='no_transformed_instance')
        if len(ref['bboxes']):
            assert len(ref['bboxes'])==1, 'This locked pool has one pseudo instance per image'
            hh,ww=out['img'].shape[-2:]; center,size=ref['bboxes'][0,:2].numpy(),ref['bboxes'][0,2:].numpy()
            box=np.r_[center-size/2,center+size/2]*[ww,hh,ww,hh]
            points=ref['keypoints'][0,:,:2].numpy()*[ww,hh]
            mask=ref['keypoints'][0,:,2].numpy()==2
            plan=random_plan(points,mask,box,(hh,ww),state_seed(initial,name))
            info.update(plan)
            if plan['applied']:
                l,t,w,h=plan['rectangle']; out['img'][:,t:t+h,l:l+w]=torch.from_numpy(fill(plan))
            info['actual_supervised']=int((out['keypoints'][0,:8,2]==2).sum())
            info['actual_covered']=int((cover(out['keypoints'][0,:8,:2].numpy()*[ww,hh],plan['rectangle']) &
                (out['keypoints'][0,:8,2].numpy()==2)).sum()) if plan['applied'] else 0
        out['occlusion_info']=info
        return out

def reference_labels(protocol, root):
    train_list=root/protocol['datasets']['REF']['train_list']['path']
    values={}
    for row in train_list.read_text().splitlines():
        image=Path(row)
        if image.name.startswith('syn__'): continue
        label=image.parent.parent/'labels'/image.with_suffix('.txt').name
        a=np.array([line.split() for line in label.read_text().splitlines()],dtype=np.float32)
        assert a.shape==(1,32)
        values[image.name]=a[:,5:].reshape(1,9,3)
    return values
