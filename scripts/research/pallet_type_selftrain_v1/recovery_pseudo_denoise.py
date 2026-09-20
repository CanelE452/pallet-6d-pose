"""Fixed pseudo-target PoseFix adaptation; no new real labels or tag inputs.

Unlike the historical capped N2 pseudo adaptation, this trains the uncapped
RGB PoseFix model to recover deliberately displaced input corners. Artificial
recovery to a pseudo target is only a training diagnostic, never real accuracy.
"""
import argparse
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from . import recovery_common as R
from . import recovery_pose as P
from .pseudo import top
from scripts.research.pallet_posefix_large_error_v1 import train as O
from scripts.research.pallet_posefix_replay_v1 import core as N
from scripts.research.pallet_posefix_replay_v1 import train as NT
from scripts.research.pallet_posefix_replay_v1.source import SourceData
from scripts.research.pallet_posefix_replay_v1.evaluate import assert_preserved
from scripts.research.pallet_sensors_submission_v1.prior_model import TFAdam
from scripts.research.pallet_posefix_large_error_v1.evaluate import recovery_damage
from scripts.research.pallet_dim_conditioned_p_v1.eval_math import summary

C=R.C
PHASE='pseudo_denoise'
DOC=R.DOC/PHASE
RAW=R.RAW/PHASE
ARMS=['BASE','CLEAN','DENOISE']
STEPS=300


def perturb(item,rng,force=False):
    """50% raw,25% single-corner,25% all trusted corners; target immutable."""
    x=dict(item,points=item['points'].copy(),valid=item['valid'].copy())
    mode=rng.random()
    if not force and mode<.5:return x
    eligible=np.flatnonzero(item['target_valid'][:8])
    if not len(eligible):return x
    chosen=eligible if mode>=.75 else rng.choice(eligible,1,replace=False)
    angle=rng.uniform(0,2*np.pi,len(chosen))
    radius=rng.uniform(.05,.15,len(chosen))*item['bbox_diagonal']
    delta=np.stack([np.cos(angle),np.sin(angle)],-1)*radius[:,None]
    x['points'][chosen]=item['target'][chosen]+delta@item['matrix'][:2,:2].T
    x['valid'][chosen]=True
    return x


def make_item(row):
    C.verify(row['image']);im=cv2.imread(str(C.ROOT/row['image']['path']))
    assert im is not None and list(im.shape[:2])==row['raw_hw']
    x=N.C.prepare_input(im,row['raw']);assert x is not None
    labels,_=P.paired_labels(row)
    valid=np.asarray(labels['REF'].split(),float)[5:].reshape(9,3)[:,2]==2
    target=N.C.transform_points(np.asarray(top(row['refined'])['keypoints_xy']),x['matrix']).astype(np.float32)
    valid&=np.isfinite(target).all(-1)&(target>=0).all(-1)&(target[:,0]<288)&(target[:,1]<384)
    valid[8]=False
    x.update(id=row['id'],target=target,target_valid=valid,partition='pseudo')
    return x


def prepare():
    parent=C.read(R.DOC/'pose_only/PROTOCOL.json')
    slots=(C.ROOT/parent['datasets']['REF']['train_list']['path']).read_text().splitlines()[512:]
    real_order=[Path(p).stem for p in slots]
    pool=[r for r in C.read(R.BASE_RAW/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC']
    selected=set(real_order);assert len(pool)==249 and len(selected)==217
    evaluation=C.read(R.BASE_DOC/'EVAL_PROTOCOL.json')['records']
    forbidden={r['image']['sha256'] for r in evaluation}
    assert not forbidden & {r['image']['sha256'] for r in pool}
    source=SourceData();old=np.load(N.RAW/'ORDERS.npz')
    assert set(old['source_rows'].ravel())<=set(source.train_rows)
    assert set(old['held_rows'])<=set(source.held_rows)
    held=old['held_rows'][:64].tolist()
    source_rows=old['source_rows'].tolist();assert np.shape(source_rows)==(300,8)
    rng=np.random.default_rng(20260922)
    samples=[[real_order[int(i)] for i in rng.integers(0,512,8)] for _ in range(STEPS)]
    real=[]
    for r in pool:
        x=make_item(r)
        real.append(dict(id=r['id'],image=r['image'],train=r['id'] in selected,
            target_valid=x['target_valid'].tolist(),matrix=x['matrix'].tolist()))
    init=C.read(N.DOC/'FIT.json')['checkpoint'];C.verify(init)
    paths=[__file__,Path(__file__).with_name('test_recovery_pseudo_denoise.py'),P.__file__,O.__file__,N.C.__file__,N.__file__,NT.__file__,
        Path(NT.__file__).with_name('source.py'),R.__file__,R.BASE_RAW/'PSEUDO_ACCEPTED.json',
        R.DOC/'pose_only/PROTOCOL.json',N.DOC/'INPUT_LOCK.json',N.RAW/'ORDERS.npz',
        R.BASE_DOC/'EVAL_PROTOCOL.json',R.BASE_RAW/'EVAL_PREDICTIONS_R0.json']
    protocol=dict(steps=STEPS,arms=ARMS,initialization=init,real_records=real,real_samples=samples,
        source_samples=source_rows,source_held_rows=held,
        source_cache_signature=N.source_cache_signature(source.data,np.unique(np.r_[old['source_rows'].ravel(),held])),
        sources=[C.bound(p) for p in paths],
        objective='Learn large-displacement correction on existing real RGB with fixed Replay pseudo targets, not just mask small corrections.',
        controls='BASE frozen existing Replay; CLEAN adapts with raw R0 inputs; DENOISE same exposures/targets/optimizer plus50% artificial input corruption. No new manual/AprilTag label.',
        targets='Original REF common trusted mask, not the failed AGREE mask. Coordinates never recomputed from GT. Center excluded. No new whole-image filter.',
        real_images='Same217 prior student images;32 remaining accepted images are a non-independent pseudo-consistency probe only, not true GT validation.',
        corruption='Real50% raw;25% one trusted corner;25% all trusted corners shifted from unchanged pseudo target by uniform angle and .05-.15 original predicted-box diagonal. Source same historical50% all-corner corruption in BOTH arms.',
        training='Warm-start frozen Replay checkpoint copy,300 TFAdam1e-4 updates each;real8+source8,micro2;all model parameters trainable,BN buffers frozen;mean real loss+mean source loss+one L2;seed1. Fixed last300, no sweep.',
        evaluation='Original R0 box/input crop;uncapped eight-corner RGB correction;original boxes/confidence/center/nonselected predictions unchanged;no new geometry or selective gate. All194 outputs frozen before GT scoring;also frozen159 subset.',
        gate='At least5 matched >20 to <=10 recoveries across3frames,<=1percent good<5 to >10damage,PCK20>=R0;compare all arms. Source clean PCK10 drop<=1pp and P90 ratio<=1.1. Screen only,not independent confirmation or automatic promotion.',
        limitations='Pseudo targets can be wrong. Same-pool correlated teacher. Artificial denoising is not real error recovery. Prior Replay has9manual training images,3 in full194 DEV. All real evaluation reused.',
        new_annotations=0,new_tags=0,auto_promote=False)
    C.freeze(DOC/'PROTOCOL.json',protocol)
    R.evaluation_protocol(PHASE,ARMS,protocol['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('PSEUDO_DENOISE_PREPARED',len(selected),'train',len(pool)-len(selected),'probe',flush=True)


def verify():
    p=C.read(DOC/'PROTOCOL.json')
    for b in p['sources']+[p['initialization']]:C.verify(b)
    return p


def load_inputs(p):
    pool={r['id']:r for r in C.read(R.BASE_RAW/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC'}
    real={r['id']:make_item(pool[r['id']]) for r in p['real_records']}
    for r in p['real_records']:
        assert real[r['id']]['target_valid'].tolist()==r['target_valid']
        np.testing.assert_array_equal(real[r['id']]['matrix'],r['matrix'])
    source=SourceData()
    rows=np.unique(np.r_[np.asarray(p['source_samples']).ravel(),p['source_held_rows']])
    assert N.source_cache_signature(source.data,rows)==p['source_cache_signature']
    held=[source.item(i) for i in p['source_held_rows']]
    return real,source,held


def save(path,obj):
    assert path.resolve().is_relative_to(RAW) and not path.exists()
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(obj,path)


def train(arm):
    assert arm in ['CLEAN','DENOISE'];p=verify()
    fit=DOC/f'FIT_{arm}.json'
    if fit.exists():C.verify(C.read(fit)['checkpoint']);print('ALREADY_COMPLETE',arm);return
    N.setup();N.E.gpu();assert torch.cuda.is_available()
    real,source,held=load_inputs(p)
    torch.manual_seed(1);torch.cuda.manual_seed_all(1)
    model=N.load_model().requires_grad_(True)
    optimizer=TFAdam(model.parameters(),lr=1e-4)
    before=NT.source_probe(model,held)
    model.train()
    for m in model.modules():
        if isinstance(m,torch.nn.BatchNorm2d):m.eval()
    buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    history=[];start_step=0;prior_seconds=0.
    # Resume checkpoints are append-only at distinct steps, never overwrite.
    resumes=sorted((RAW/arm).glob('resume_*.pt')) if (RAW/arm).exists() else []
    if resumes:
        ck=torch.load(resumes[-1],map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json') and ck['arm']==arm
        model.load_state_dict(ck['model_state_dict']);optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_step=ck['step'];history=ck['history'];prior_seconds=ck['seconds'];del ck
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as workers:
        pending=[workers.submit(source.item,i) for i in p['source_samples'][start_step]] if start_step<STEPS else []
        for idx in range(start_step,STEPS):
            syn=[f.result() for f in pending]
            pending=[workers.submit(source.item,i) for i in p['source_samples'][idx+1]] if idx+1<STEPS else []
            rr=[real[i] for i in p['real_samples'][idx]]
            if arm=='DENOISE':
                rng=np.random.default_rng(np.random.SeedSequence([20260922,idx,0]));rr=[perturb(x,rng) for x in rr]
            srng=np.random.default_rng(np.random.SeedSequence([20260922,idx,1]))
            branches=dict(real=rr,source=[O.corrupted(x,srng) for x in syn])
            optimizer.zero_grad(set_to_none=True);details={}
            for branch,items in branches.items():
                parts=[]
                for j in range(0,8,2):
                    b=O.batch(items[j:j+2]);q=model(b['rgb'],b['points'],b['valid'])
                    _,detail=model.losses(q,b['target'],b['target_valid'])
                    loss=NT.combined_micro_loss(detail,branch);assert torch.isfinite(loss)
                    loss.backward();parts.append({k:float(v.detach()) for k,v in detail.items()})
                    del b,q,detail,loss
                details[branch]={k:float(np.mean([v[k] for v in parts])) for k in parts[0]}
            update=float(optimizer.step());assert np.isfinite(update) and update>0
            history.append(dict(step=idx+1,loss=details,update_norm=update))
            if (idx+1)%100==0:
                save(RAW/arm/f'resume_{idx+1:03d}.pt',dict(arm=arm,step=idx+1,history=history,
                    protocol_sha256=C.sha(DOC/'PROTOCOL.json'),model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),seconds=prior_seconds+time.monotonic()-started))
            if idx==start_step or (idx+1)%50==0:
                print('PSEUDO_DENOISE_TRAIN',arm,idx+1,'/300',round(prior_seconds+time.monotonic()-started,1),N.E.gpu(),flush=True)
    assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
    after=NT.source_probe(model,held)
    pseudo_held=[real[r['id']] for r in p['real_records'] if not r['train']]
    pseudo_probe=NT.source_probe(model,pseudo_held)
    path=RAW/arm/'last300.pt'
    save(path,dict(arm=arm,step=STEPS,model_state_dict=model.state_dict(),protocol_sha256=C.sha(DOC/'PROTOCOL.json')))
    C.freeze(fit,dict(complete=True,arm=arm,step=STEPS,checkpoint=C.bound(path),
        protocol=C.bound(DOC/'PROTOCOL.json'),source_before=before,source_after=after,
        held32_pseudo_consistency_NOT_accuracy=pseudo_probe,BN_buffers_bit_exact=True,
        seconds=prior_seconds+time.monotonic()-started,history=history))
    print('PSEUDO_DENOISE_FIT_COMPLETE',arm,flush=True)


@torch.no_grad()
def infer(arm):
    p=verify();N.setup();N.E.gpu();assert torch.cuda.is_available()
    path=RAW/f'EVAL_PREDICTIONS_{arm}.json'
    if path.exists():C.verify(C.read(path)['checkpoint']);print('INFERENCE_ALREADY_COMPLETE',arm);return
    if arm=='BASE':model=N.load_model();checkpoint=p['initialization']
    else:
        fit=C.read(DOC/f'FIT_{arm}.json');checkpoint=fit['checkpoint'];C.verify(checkpoint)
        ck=torch.load(C.ROOT/checkpoint['path'],map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json') and ck['step']==STEPS
        model=N.C.PoseFixPallet9();model.load_state_dict(ck['model_state_dict']);del ck
        model=model.cuda().eval().requires_grad_(False)
    originals={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    rows=[]
    for i,r in enumerate(C.read(DOC/'EVAL_PROTOCOL.json')['records']):
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
        old=originals[r['id']];out=N.C.predict(model,im,old['prediction'])
        assert_preserved(old['prediction'],out)
        rows.append(dict(id=r['id'],kind='PLASTIC',raw_hw=old['raw_hw'],prediction=out))
        if (i+1)%50==0:print('PSEUDO_DENOISE_INFER',arm,i+1,'/194',flush=True)
    C.freeze(path,dict(complete=True,arm=arm,checkpoint=checkpoint,records=rows,GT_free_predictions=True))


def report():
    verify();bindings=[C.bound(RAW/f'EVAL_PREDICTIONS_{a}.json') for a in ARMS]
    C.freeze(DOC/'OUTPUTS_LOCK.json',dict(artifacts=bindings,all_predictions_before_scoring=True))
    metrics={a:R.score(PHASE,a,False)['metrics'] for a in ARMS}
    metrics['R0']=[r for r in C.read(R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC']
    order=[r['id'] for r in metrics['R0']]
    metrics={a:[{r['id']:r for r in rows}[i] for i in order] for a,rows in metrics.items()}
    subset=C.read(R.BASE_DOC/'large_corner_recovery_v1/extreme_low_subset_v1/SUBSET_MANIFEST.json')
    keep={r['id'] for r in subset['records'] if r['kind']=='PLASTIC'}
    results={};checks={}
    for population,ids in [('full194',set(order)),('retained159',keep)]:
        base=[r for r in metrics['R0'] if r['id'] in ids];table={}
        for a,rr in metrics.items():
            rows=[r for r in rr if r['id'] in ids]
            recovered_frames=sum(b['matched'] and any(x is not None and x>20 and y<=10 for x,y in zip(b['canonical_errors'],n['canonical_errors'])) for b,n in zip(base,rows))
            table[a]=dict(summary=summary(rows),recovery=recovery_damage(base,rows,True),recovered_frames=recovered_frames)
        results[population]=table
    for a in ['CLEAN','DENOISE']:
        x=results['full194'][a];fit=C.read(DOC/f'FIT_{a}.json');b=fit['source_before']['clean'];n=fit['source_after']['clean']
        checks[a]=dict(recovery5=x['recovery']['recovered']>=5,frames3=x['recovered_frames']>=3,
            damage1percent=x['recovery']['damage_rate']<=.01,
            pck20_preserved=x['summary']['PCK']['20']>=results['full194']['R0']['summary']['PCK']['20'],
            source_PCK10_preserved=n['PCK10']>=b['PCK10']-.01,source_P90_preserved=n['P90_px']<=b['P90_px']*1.1)
    result=dict(results=results,checks=checks,passed={a:all(v.values()) for a,v in checks.items()},
        goal_complete=False,new_annotations=0,new_tags=0,independent_confirmation=False,auto_promoted=False)
    C.freeze(DOC/'RESULTS.json',result)
    lines=['# 기존 수도레이블 기반 큰 입력 오차 복구 학습','',
        '추가 수동 레이블·태그0. 기존 Replay를 복제하여 CLEAN과 DENOISE 각각300step. 같은217장과 합성 replay를 사용하며, DENOISE만 실사 입력 코너를 크게 교란한다. 목표 좌표는 기존 수도레이블 그대로이며 평가 GT는 학습·선택에 쓰지 않는다.','',
        '| 평가 | 모델 | 중앙 px | P90 px | PCK20 % | >20→≤10 복구 | <5→>10 손상 | 복구 이미지 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for pop,table in results.items():
        for a in ['R0',*ARMS]:
            x=table[a];s=x['summary'];d=x['recovery']
            lines.append(f'| {pop} | {a} | {s["matched_pooled_corner8_median_px"]:.3f} | {s["matched_pooled_corner8_P90_px"]:.3f} | {100*s["PCK"]["20"]:.2f} | {d["recovered"]}/{d["hard"]} | {d["damaged"]}/{d["good"]} | {x["recovered_frames"]} |')
    lines+=['',f'사전 screen: {result["passed"]}. 기준별 상태: {checks}', '',
        '32장 probe는 수도레이블과의 일치이지 실제 정확도가 아니다. 실제 자연 오류 복구는 위 평가로만 판단한다. 기존 교사 학습과 겹친3장을 포함한 반복 DEV이며 독립 검증이 아니다. 최종 모델은 교체하지 않았다.']
    C.write_text(RAW/'REPORT_KO.md','\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','infer','report']);p.add_argument('arm',nargs='?',choices=ARMS);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='train':train(a.arm)
    elif a.action=='infer':infer(a.arm)
    else:report()
