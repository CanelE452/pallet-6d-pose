"""Reuse source/C0 tensors; paired new real tensors from the actual installed loader."""
import copy
import json
import random
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.augment import RandomPerspective
from scripts.research.pallet_clean19_structured_easyhard_v1 import augmentation as A
from . import common as E

def build():
    E.immutable();targets=E.read(E.RAW/'TARGETS.json');pairs=E.read(E.DOC/'TARGET_AND_PAIR_AUDIT.json')['pairs'];by={(r['role'],r.get('pair')):r for r in targets if r['role']!='C0'}
    args=E.read(E.C.DOC/'PREFLIGHT.json')['args'];datasets={}
    for role in ('E','H'):
        folder=E.RAW/'dataset'/role;paths=[]
        for i in range(len(pairs)):
            r=by[role,i];im=cv2.imread(str(E.ROOT/r['image']['path']));h,w=im.shape[:2];dest=folder/'images'/f'pair_{i:02d}.png';dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();assert cv2.imwrite(str(dest),cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_REFLECT_101))
            box=np.array(r['bbox']);box[[0,2]]=np.clip(box[[0,2]],0,w);box[[1,3]]=np.clip(box[[1,3]],0,h);assert (box[2:]-box[:2]>1).all()
            xy=(np.array(r['target'])+100)/[w+200,h+200];mask=np.array(r['mask']);xy[~mask]=.5
            values=[0,*((box[:2]+box[2:])/2+100)/[w+200,h+200],*(box[2:]-box[:2])/[w+200,h+200],*np.c_[xy,np.where(mask,2,1)].ravel()]
            E.save(folder/'labels'/f'pair_{i:02d}.txt',' '.join(f'{v:.9f}' for v in values)+'\n');paths.append(str(dest))
        E.save(folder/'train.txt','\n'.join(paths)+'\n');E.save(folder/'data.yaml',f'path: {folder}\ntrain: {folder/"train.txt"}\nval: {E.C.H.RAW/"dataset/val.txt"}\nnc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n')
        data=check_det_dataset(str(folder/'data.yaml'));ds=build_yolo_dataset(get_cfg(overrides=dict(args,task='pose')),data['train'],16,data,mode='train',rect=False,stride=32);assert len(ds)==len(pairs);datasets[role]=ds
    plans=[json.loads(x) for x in (E.C.DOC/'AUGMENTATION_PLAN.jsonl').read_text().splitlines() if json.loads(x)['material']=='PLASTIC'];out=[];counts=Counter();captured={};original=RandomPerspective.get_params
    def capture(self,labels):
        p=original(self,labels);captured.update(p);return p
    RandomPerspective.get_params=capture
    try:
        realrank=0
        for occ,p in enumerate(plans):
            if not p['real'] or realrank%2==0:
                role='C0' if p['real'] else 'SYNTH';counts[role]+=1
                out.append(dict(occ=occ,epoch=p['epoch'],slot=p['slot'],role=role,old=p,cache=p['cache'],arms_same=True))
                if p['real']:realrank+=1
                continue
            pair=(realrank//2)%len(pairs);realrank+=1;versions={};matrices={}
            for role in ('E','H'):
                ds=datasets[role];r=by[role,pair];random.seed(p['base_seed']);np.random.seed(p['base_seed']);captured.clear();s=ds[pair]
                assert len(s['keypoints'])==1 and s['img'].shape==(3,640,640)
                rh,rw=captured['orig_shape'];h,w=r['hw'];pad=np.eye(3);pad[:2,2]=100;M=np.array(captured['M'])@np.diag([rw/(w+200),rh/(h+200),1])@pad;matrices[role]=M
                arrays={k:s[k].numpy().copy() for k in ('img','keypoints','bboxes','cls','batch_idx')};q=np.c_[r['target'],np.ones(9)]@M.T
                mask=np.array(r['mask']);assert np.abs(np.clip(q[mask,:2],0,640)-arrays['keypoints'][0,mask,:2]*640).max()<.001
                arrays['keypoints'][0,~mask,2]=1;versions[role]=arrays
            manual=np.c_[by['H',pair]['manual'],np.ones(9)]@matrices['H'].T
            support=np.array(pairs[pair]['mask'])&(versions['E']['keypoints'][0,:,2]==2)&(versions['H']['keypoints'][0,:,2]==2)&(manual[:,:2]>=0).all(1)&(manual[:,:2]<640).all(1);support[8]=False
            assert support.any(),'Empty transformed supervision: stop input contract'
            for role,a in versions.items():a['keypoints'][0,:,2]=np.where(support,2,1)
            versions['M']={k:v.copy() for k,v in versions['H'].items()};versions['M']['keypoints'][0,support,:2]=manual[support,:2]/640
            # Reuse S1's predeclared normalized rectangle, size fraction, fill and application flag.
            # Positions are mapped relative to each new predicted box; no mask-count/error search.
            pp={};baseplan=p['plan']
            for role,a in versions.items():
                plan=copy.deepcopy(baseplan)
                if plan['applied']:
                    oldbox=np.array(baseplan['bbox']);rect=np.array(baseplan['S1'],float);wh=oldbox[2:]-oldbox[:2]
                    normcenter=(rect[:2]+rect[2:]/2-oldbox[:2])/wh
                    b=a['bboxes'][0]*640;newbox=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2];area=float(np.prod(b[2:]));nw=max(1,round(np.sqrt(area*plan['area_fraction']*plan['aspect'])));nh=max(1,round(np.sqrt(area*plan['area_fraction']/plan['aspect'])))
                    nw=min(nw,640);nh=min(nh,640);center=newbox[:2]+normcenter*b[2:];left,top=np.clip(np.round(center-[nw/2,nh/2]),[0,0],[640-nw,640-nh]).astype(int)
                    plan['S1']=[int(left),int(top),nw,nh];plan['size']=[nw,nh];plan['normalized_box_center']=normcenter.tolist()
                a['img']=A.apply(a['img'],plan,'S1');pp[role]=plan
            assert np.array_equal(versions['H']['img'],versions['M']['img']) and np.array_equal(versions['H']['bboxes'],versions['M']['bboxes'])
            assert all(np.array_equal(a['keypoints'][0,:,2],versions['H']['keypoints'][0,:,2]) for a in versions.values())
            paths={}
            for role,a in versions.items():
                path=E.RAW/'cache'/role/f'{occ:04d}.npz';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez(path,**a);paths[role]=E.bind(path)
            out.append(dict(occ=occ,epoch=p['epoch'],slot=p['slot'],role='REPLACEMENT',pair=pair,cache=paths,plans=pp,support=support.tolist(),native_to_model={k:v.tolist() for k,v in matrices.items()},supervised_count=int(support.sum()),RGB_H_M_sha=E.C.digest(versions['H']['img'])))
            counts['REPLACEMENT']+=1;counts['supervised_coordinate_exposures']+=int(support.sum());counts['replacement_occluded_occurrences']+=int(baseplan['applied'])
            if occ%1024==1023:print('CACHE_EPOCH',p['epoch']+1,flush=True)
    finally:RandomPerspective.get_params=original
    assert counts['C0']==counts['REPLACEMENT']==1280 and counts['SYNTH']==2560
    E.save(E.RAW/'OCCURRENCES.json',out);E.save(E.DOC/'INPUT_EXPOSURE_AUDIT.json',dict(counts=dict(counts),T1_T2_RGB_bbox_class_order_equal=True,all_three_replacement_masks_equal=True,C0_source_exact=True,existing_S1_applied_flags_fill_fraction_aspect=True,new_policy=False,normalized_mapping='Existing S1 rectangle center relative to old predicted box, mapped to new box; size recomputed from same area fraction/aspect; image-clipped. No placement search.',args=args,occurrences=E.bind(E.RAW/'OCCURRENCES.json'),datasets={k:str(E.RAW/'dataset'/k/'data.yaml') for k in ('E','H')}))
    print('CACHE_READY',counts,flush=True)

if __name__=='__main__':build()
