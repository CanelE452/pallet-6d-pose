"""Frozen, evaluation-coordinate-free matched252 data preparation; no fitting."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_occlusion_refiner_transfer_v2 import run as E
from scripts.research.pallet_occlusion_refiner_transfer_v2 import pilot as P
from scripts.research.pallet_posefix_limited_adaptation_pilot_v1 import common as B

NAME='pallet_pose_stable_improvement_20261001_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
EXCLUDED_SHA='862d175bd2ce92045aa62edbe34c76f20e5d28c8f9729ee481a9e4d0930ef1cd'
read=lambda p:json.loads(Path(p).read_text())
bind=E.bind

def verify(b):
    got=bind(ROOT/b['path'])
    assert got['sha256']==b['sha256'],b['path']
    if 'bytes' in b:assert got['bytes']==b['bytes'],b['path']

def save(p,x):
    p=Path(p);assert p.is_relative_to(DOC) or p.is_relative_to(RAW)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def tensor_save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:torch.save(x,f)

def key(r):return hashlib.sha256(('20261001|'+r['id']).encode()).hexdigest()

def balanced(rows,n):
    pools=defaultdict(list)
    for r in rows:pools[r['recording_id']].append(r)
    pools={k:sorted(v,key=key) for k,v in sorted(pools.items())}
    chosen=[];counts=Counter()
    while len(chosen)<n:
        eligible=[r for r in pools if counts[r]<len(pools[r])]
        assert eligible,'Insufficient eligible unique inputs; no count reduction'
        rec=min(eligible,key=lambda r:(counts[r],r))
        chosen.append(pools[rec][counts[rec]]);counts[rec]+=1
    assert len(counts)>=3
    return sorted(chosen,key=key),dict(counts)

def iou(a,b):
    a=np.asarray(a);b=np.asarray(b);overlap=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(overlap/max(1e-12,np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-overlap))

def cpu():
    started=time.monotonic();cv2.setNumThreads(1);torch.set_num_threads(4)
    assert not (DOC/'DATA_PREPARATION_PROTOCOL.json').exists(),'Preserve existing preparation'
    old=B.read(B.DOC/'INPUT_LOCK.json');pool_path=ROOT/'_docs/experiments/pallet_type_selftrain_v1/POOL.json'
    cache_protocol=ROOT/'_docs/experiments/pallet_type_selftrain_v1/PSEUDO_PROTOCOL.json'
    target_protocol=E.DOC/'PSEUDO_PROTOCOL.json';audit=read(DOC/'DATA_AUDIT.json')
    assert audit['adaptation']['paper319_image_overlap']==0 and audit['adaptation']['paper319_recording_overlap']==[]
    source_cache=read(cache_protocol);target_cache=read(target_protocol)
    cache_bindings=source_cache['sources']+[source_cache['code']]
    for b in cache_bindings+target_cache['bindings']:verify(b)
    r0=next(b for b in cache_bindings if b['path'].endswith('G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'))
    replay=next(b for b in cache_bindings if b['path'].endswith('pallet_posefix_replay_v1/last300.pt'))
    assert any(b['sha256']==r0['sha256'] for b in target_cache['bindings'])
    assert any(b['sha256']==replay['sha256'] for b in target_cache['bindings'])
    members=ROOT/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json'
    # Membership only: never open gt_v2_path or any evaluation coordinates.
    eval_hashes={bind(ROOT/r['image_path'])['sha256'] for r in read(members)['items']}
    assert len(eval_hashes)==319
    excluded_recordings=set(audit['populations']['FULL128']['recordings'])
    protocol=dict(arms=['SINGLE252','DIVERSE252'],unique_each=252,seed=20261001,
        intervention='Only real recording composition; existing frozen target and image-occlusion recipe.',
        target_recipe='Cached R0 .85/min6kp.5 -> cached raw flip/LOO .05 -> frozen cached Replay -> existing hidden-onlyPnP -> all8LOO .05',
        selection='After target eligibility, max-min recording counts, ties recording ID; per-recording SHA256(20261001|id) order. Select252 once before masked-R0 outcomes.',
        pair_failure='Preserve selected IDs and all failure receipts. No replacement/resampling; no final252 inputs lock if any selected pair fails.',
        same_object_guard='clean top vs occluded top bbox IoU>=.5, existing matching criterion; no GT box',
        augmentation='Existing E2/CAD8 recipe, probability.75, seed902106 in selected nonDAY ID-hash order; no force mask/no hard mining',
        crop='Existing native CLEAN/OCC predicted bbox crops and common target-support intersection; maskcenter8False',
        excluded_eval_sha=sorted(eval_hashes),known_original_day_overlap=EXCLUDED_SHA,
        excluded_full128_recordings=sorted(excluded_recordings),
        paper319_recording_caveat='SINGLE REC001 and DIVERSE DAY subset share recording with diagnostic paper66/green; those are exposed diagnostics, never confirmation.',
        source_orders=old['source_orders'],real_order='rng6401 integers0..251 shape300x8 sharedarms; later seeds declared by root',
        cache_protocol=bind(cache_protocol),target_protocol=bind(target_protocol),
        cache_bindings=cache_bindings,target_recipe_bindings=target_cache['bindings'],data_audit=bind(DOC/'DATA_AUDIT.json'),
        membership=bind(members),code=bind(Path(__file__)),GT_input=False,original_manual_teacher_lineage=audit['supervision_lineage']['FULL125'])
    save(DOC/'DATA_PREPARATION_PROTOCOL.json',protocol)
    daymeta={r['id']:r for r in read(E.DOC/'E2_INPUT_LOCK.json')['records']};single=[]
    for b in old['paired_files']:
        verify(b);fid=Path(b['path']).stem;r=daymeta[fid];verify(r['image'])
        if r['image']['sha256'] in eval_hashes:
            assert fid=='DAY264__005516' and r['image']['sha256']==EXCLUDED_SHA
            continue
        single.append(dict(id=fid,image=r['image'],recording_id='REC_001',pair=b,source='original_day_pair',sample_class='frozen_pseudo',
            target_class='Replay+selfocclusionPnP',teacher_direct_recording_exposure=True))
    single=sorted(single,key=key);assert len(single)==252
    pool=[r for r in read(pool_path)['records'] if r['object_type']=='plastic_standard_110x130x11'];assert len(pool)==1000
    decisions=[];eligible=[];bindings=[]
    for r in pool:
        assert r['image']['sha256'] not in eval_hashes and r['recording_id'] not in excluded_recordings
        f=ROOT/'data/pallet/results/pallet_type_selftrain_v1/pseudo_frames'/f"{r['id']}.json"
        x=read(f);assert x['protocol_sha256']==bind(cache_protocol)['sha256'] and x['image']==r['image']
        if x['refined'] is None:
            decisions.append(dict(id=r['id'],recording_id=r['recording_id'],accepted=False,reason='cached_'+x['reason']));continue
        verify(r['image']);assert E.P.passed(x['stage1'],['s_remove','s_flip'])
        assert x['raw_hw']==[480,640]
        c=E.P.top(x['raw']);assert c['score']>=.85 and E.P.valid_points(c)[:8].sum()>=6
        refined=copy.deepcopy(x['refined']);cc=E.P.top(refined);K=np.array(r['K'])
        initial=E.pose.infer(cc['keypoints_xy'],K,np.array([1.1,.11,1.3]))
        target,pnp=E.V.S.correct(cc,initial,K);cc['keypoints_xy']=target.tolist();E.assert_preserved(x['raw'],refined)
        scores=E.scores(cc,K);accepted=bool(E.P.valid_points(cc,False)[:8].all() and E.P.passed(scores,['s_remove']))
        mask=E.P.valid_points(c)&np.isfinite(target).all(1)&(target>=0).all(1)&(target[:,0]<640)&(target[:,1]<480);mask[8]=False
        prepared=dict(**r,source='cached_standard_pool',sample_class='frozen_pseudo',teacher_direct_recording_exposure=False,
            raw=x['raw'],refined=refined,target_original=target.tolist(),mask=mask.tolist(),stage1=x['stage1'],stage2=scores,
            pnp=pnp,cache=bind(f),target_class='Replay+selfocclusionPnP',accepted=accepted)
        path=RAW/'target_candidates'/f"{r['id']}.json";save(path,prepared);bindings.append(bind(path))
        decisions.append(dict(id=r['id'],recording_id=r['recording_id'],accepted=accepted,reason='accepted' if accepted else 'postPnP_all8LOO',pnp_applied=pnp['applied']))
        if accepted:eligible.append(dict(id=r['id'],image=r['image'],recording_id=r['recording_id'],candidate=bind(path),source='cached_standard_pool',
            sample_class='frozen_pseudo',target_class='Replay+selfocclusionPnP',teacher_direct_recording_exposure=False))
    assert len(bindings)==259
    allrows=single+eligible;assert len({r['image']['sha256'] for r in allrows})==len(allrows)
    diverse,counts=balanced(allrows,252)
    order=torch.from_numpy(np.random.default_rng(6401).integers(0,252,size=(300,8)))
    tensor_save(RAW/'REAL_ORDER_SEED1.pt',order)
    save(RAW/'DATA_SELECTION.json',dict(SINGLE252=single,DIVERSE252=diverse,selection_recordings=counts,
        target_eligible_recordings=dict(Counter(r['recording_id'] for r in allrows)),target_candidate_bindings=bindings,decisions=decisions,
        source_orders=old['source_orders'],real_order=bind(RAW/'REAL_ORDER_SEED1.pt'),initialization=B.read(B.DOC/'EXPERIMENT_PROTOCOL.json')['base']))
    save(DOC/'DATA_PREPARATION_CPU.json',dict(complete=True,cached_pool1000=len(pool),cached_refined259=len(bindings),new_target_eligible=len(eligible),
        eligible_recordings=dict(Counter(r['recording_id'] for r in allrows)),selected_recordings=counts,
        selected_new_masked_R0=sum(r['source']=='cached_standard_pool' for r in diverse),
        single252=252,selected252=252,evaluation_coordinate_files_opened=0,image_forward_count=0,
        protocol=bind(DOC/'DATA_PREPARATION_PROTOCOL.json'),selection=bind(RAW/'DATA_SELECTION.json'),seconds=time.monotonic()-started))
    print('CPU_PREPARATION_COMPLETE',counts,'newR0',sum(r['source']=='cached_standard_pool' for r in diverse),flush=True)

def parity(old,new):
    assert old['selected_index']==new['selected_index'] and len(old['candidates'])==len(new['candidates'])
    maxima={}
    for a,b in zip(old['candidates'],new['candidates']):
        for k in ('keypoints_xy','box_xyxy','keypoints_conf','score'):
            if k in a:
                delta=float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k]))));maxima[k]=max(delta,maxima.get(k,0.))
                np.testing.assert_allclose(np.asarray(a[k]),np.asarray(b[k]),atol=1e-4,rtol=0,err_msg=k)
    return maxima

def gpu():
    started=time.monotonic();B.L.setup('cuda');E.N.setup();cv2.setNumThreads(1)
    p=read(DOC/'DATA_PREPARATION_PROTOCOL.json');verify(p['code']);sel=read(RAW/'DATA_SELECTION.json')
    assert not (DOC/'INPUTS.json').exists() and not (DOC/'DATA_PREPARATION_GPU.json').exists(),'Preserve prior preparation'
    for b in p['cache_bindings']+p['target_recipe_bindings']+[sel['source_orders'],sel['real_order']]:verify(b)
    extractor=E.N.E.old('features').FrozenYoloFeatures(E.N.E.R0)
    rng=np.random.default_rng(P.A.SEED);ready=[];receipts=[];forward_count=0
    try:
        first=next(r for r in sel['DIVERSE252'] if r['source']=='cached_standard_pool');verify(first['candidate']);x=read(ROOT/first['candidate']['path'])
        image=cv2.imread(str(ROOT/x['image']['path']));actual=E.raw_infer(extractor,image);forward_count+=1
        smoke=dict(id=x['id'],max_absolute_difference=parity(x['raw'],actual),atol=1e-4,rtol=0,cudnn_TF32=True,matmul_TF32=False)
        save(DOC/'DATA_PREPARATION_GPU_SMOKE.json',smoke)
        for i,r in enumerate(sel['DIVERSE252']):
            E.gpu()
            if r['source']=='original_day_pair':
                verify(r['pair']);entry=torch.load(ROOT/r['pair']['path'],map_location='cpu',weights_only=False)
                meta=entry['metadata'];clean=E.P.top(meta['raw_prediction']);occ=E.P.top(meta['occluded_R0_prediction']);match=iou(clean['box_xyxy'],occ['box_xyxy'])
                assert match>=.5
                ready.append(r);receipts.append(dict(id=r['id'],recording_id=r['recording_id'],success=True,reused=True,pair=r['pair'],bbox_iou=match,
                    supervised_corners=int(entry['pair']['OCC']['target_valid'].sum()),patches=bool(meta['plans'])))
                continue
            verify(r['candidate']);x=read(ROOT/r['candidate']['path']);verify(x['image'])
            image=cv2.imread(str(ROOT/x['image']['path']));assert image.shape==(480,640,3)
            raw=x['raw'];target=np.array(x['target_original']);mask=np.array(x['mask'],bool);c=E.P.top(raw)
            occ,plans=P.augmented(image,c,target,mask,rng,x['id']);occraw=E.raw_infer(extractor,occ);forward_count+=1
            co=E.P.top(occraw);match=None if co is None else iou(c['box_xyxy'],co['box_xyxy'])
            pair=None if co is None or match<.5 else P.paired_items(image,occ,raw,occraw,target,mask)
            reason='accepted' if pair is not None else ('no_occ_detection' if co is None else ('bbox_iou_below_existing_0.5' if match<.5 else 'no_common_target_support'))
            receipt=dict(id=r['id'],recording_id=r['recording_id'],success=pair is not None,reused=False,bbox_iou=match,reason=reason,plans=plans,
                original_RGB_sha=P.array_sha(image),occluded_RGB_sha=P.array_sha(occ),occluded_R0_prediction=occraw)
            if pair is not None:
                for v in pair.values():v['id']=r['id']
                dest=RAW/'paired_inputs'/f"{r['id']}.pt"
                metadata=dict(id=r['id'],image=r['image'],recording_id=r['recording_id'],plans=plans,raw_prediction=raw,occluded_R0_prediction=occraw,
                    R0_actually_rerun=True,target_original=target.tolist(),mask=pair['OCC']['target_valid'].tolist(),target_sha=P.array_sha(target),
                    mask_sha=P.array_sha(pair['OCC']['target_valid']),clean_RGB_sha=P.array_sha(image),occluded_RGB_sha=P.array_sha(occ),candidate=r['candidate'],GT_input=False)
                tensor_save(dest,dict(pair=pair,metadata=metadata));b=bind(dest);receipt.update(pair=b,supervised_corners=int(pair['OCC']['target_valid'].sum()),patches=bool(plans))
                ready.append(dict(**r,pair=b))
            save(RAW/'pair_receipts'/f"{r['id']}.json",receipt);receipts.append(receipt)
            if (i+1)%25==0:print('PAIR_PREPARATION',i+1,'/',len(sel['DIVERSE252']),'forwards',forward_count,flush=True)
    finally:extractor.close()
    failures=[r for r in receipts if not r['success']]
    report=dict(complete=not failures,selected=252,ready=len(ready),failures=failures,no_replacements=True,actual_masked_R0_forwards=forward_count-1,
        smoke_forwards=1,model_forward_total=forward_count,recordings=dict(Counter(r['recording_id'] for r in ready)),
        supervised_corners=sum(r.get('supervised_corners',0) for r in receipts),patched=sum(r.get('patches',False) for r in receipts),
        parity=smoke,receipts=receipts,seconds=time.monotonic()-started,optimizer_updates=0)
    save(DOC/'DATA_PREPARATION_GPU.json',report)
    if failures:print('PREPARATION_BLOCKED_NO_REPLACEMENTS',len(failures),flush=True);return
    assert len(ready)==252 and len({r['image']['sha256'] for r in ready})==252
    final=dict(arms=dict(SINGLE252=sel['SINGLE252'],DIVERSE252=ready),paired_count=252,selection=dict(algorithm=p['selection'],seed=20261001,
        candidate_counts=sel['target_eligible_recordings'],excluded_eval_sha=p['excluded_eval_sha'],preselected_ids=[r['id'] for r in sel['DIVERSE252']],pair_failures=[]),
        source_orders=sel['source_orders'],real_order=sel['real_order'],initialization=sel['initialization'],target_recipe_bindings=p['target_recipe_bindings'],
        cache_bindings=p['cache_bindings'],data_audit=p['data_audit'],data_protocol=bind(DOC/'DATA_PREPARATION_PROTOCOL.json'),
        data_selection=bind(RAW/'DATA_SELECTION.json'),actual_masked_R0_forwards=forward_count-1,parity=smoke,GT_input=False)
    save(DOC/'INPUTS.json',final);print('INPUTS_READY',len(ready),report['recordings'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['cpu','gpu']);args=ap.parse_args();globals()[args.stage]()
