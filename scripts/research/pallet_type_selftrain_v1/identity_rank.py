"""Absolute RGB-feature identity ranking, not R0-relative utility regression.

Both candidates contain the same physical points. Only role assignments differ.
C2 remains the evaluation group; a quarter-turn is an actual prediction edit.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as TF
from . import recovery_common as R
from . import recovery_pose as P
from . import identity_rank_features as F
from .pseudo import top
from scripts.research.pallet_posefix_replay_v1.source import SourceData
from scripts.research.pallet_posefix_replay_v1 import core as N
from scripts.research.pallet_posefix_replay_v1.evaluate import assert_preserved
from scripts.research.pallet_posefix_large_error_v1.evaluate import recovery_damage
from scripts.research.pallet_dim_conditioned_p_v1.eval_math import summary

C=R.C
PHASE='identity_rank'
DOC=R.DOC/PHASE
RAW=R.RAW/PHASE
ARMS=['SYN','MIX']
SPLIT=C.ROOT/'_docs/experiments/pallet_posefix_utility_selector_v1/SPLIT.json'
SIDECAR=C.ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz'


def pair_errors(q,gt,valid):
    """Targets only: best approved whole C2 correspondence in each class."""
    q=np.asarray(q);gt=np.asarray(gt);valid=np.asarray(valid,bool)[:8]
    assert valid.any() and np.isfinite(gt[:8][valid]).all()
    out=[]
    for perms in [F.PERMS[:2],F.PERMS[2:]]:
        errors=[np.linalg.norm(q[p][:8]-gt[:8],axis=1) for p in perms]
        e=min(errors,key=lambda x:x[valid].mean()).copy();e[~valid]=np.nan;out.append(e)
    return np.asarray(out,np.float32)


def prepare():
    split=C.read(SPLIT);side=np.load(SIDECAR);rng=np.random.default_rng(20260923);records=[]
    data=SourceData().data
    for part,n in [('train',512),('val',128)]:
        eligible=[r for r in split['records'] if r['split']==part and side['order'][r['row']]==2
            and np.asarray(data.arrays['gt_valid'][r['row']])[:8].sum()>=6]
        assert len(eligible)>=n,(part,len(eligible))
        for i in sorted(rng.choice(len(eligible),n,replace=False)):
            r=copy.deepcopy(eligible[i]);meta=data.source['records'][int(data.indices[r['row']])]
            assert meta['source_kind']=='synthetic' and meta['reflect_pad_px']==100
            r['label']=C.bound(meta['label']);records.append(r)
    assert not {r['scenario'] for r in records if r['split']=='train'} & {r['scenario'] for r in records if r['split']=='val'}
    parent=C.read(R.DOC/'pose_only/PROTOCOL.json')
    chosen={Path(v).stem for v in (C.ROOT/parent['datasets']['REF']['train_list']['path']).read_text().splitlines()[512:]}
    pool=[r for r in C.read(R.BASE_RAW/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC'];assert len(chosen)==217
    forbidden={r['image']['sha256'] for r in C.read(R.BASE_DOC/'EVAL_PROTOCOL.json')['records']}
    assert not forbidden & {r['image']['sha256'] for r in pool+records}
    feature_module=C.N.E.old('features')
    sources=[C.bound(x) for x in [__file__,F.__file__,Path(__file__).with_name('test_identity_rank.py'),
        R.__file__,P.__file__,feature_module.__file__,SPLIT,SIDECAR,R.BASE_RAW/'PSEUDO_ACCEPTED.json',
        R.DOC/'pose_only/PROTOCOL.json',data.directory/'CACHE_MANIFEST.json',data.directory/'CACHE_COMPLETE.json',
        data.run_dir/'SOURCE_MANIFEST.json',C.N.E.R0,R.BASE_RAW/'EVAL_PREDICTIONS_R0.json']]
    protocol=dict(arms=ARMS,source_records=records,real_train=sorted(chosen),real_probe=sorted(r['id'] for r in pool if r['id'] not in chosen),
        sources=sources,
        purpose='Classify absolute plausibility of native versus quarter-reassigned roles from RGB neck features. Not the previous unfitted relative-utility forest; its failed coverage gate is preserved.',
        inputs='Frozen R0 FP16-rounded P3/P4 at8corners+4interior samples of4side faces,normalized candidate coordinates and spatial-valid flags. NO view/reindex flag, NO subtraction/comparison with R0 identity, NO GT/camera/dimensions/depth/CAD input.',
        symmetry='Exact C2 group average of scores;90-degree channel reassignment is NOT accepted as GT symmetry. Only source order2 examples; official plastic scoring remains C2.',
        labels='Source: existing syntheticGT chooses better complete C2 group, ambiguous margin<=.01boxdiag has no pair loss. Real: same rule using existing Replay pseudo and original common trusted mask; no new annotation/tag.',
        learner='Shared scalar MLP4648->128ReLU->1;average2 C2 scores per class;pair softmax cross-entropy. 800 AdamW steps,lr.0003,weight_decay.001,seed9023,batch32. SOURCE-only normalization(mean/std,minstd.001,clip8) shared by both arms.',
        controls='SYN:32source/step;MIX:16source+16real/step. Same update count and initialization;source exposure is not matched. No backbone/refiner update. Same217 pseudo images,32 remainder only pseudo-consistency,not accuracy.',
        calibration='Source128 only, before real eval. Evaluate native and artificially quarter-permuted inputs. Thresholds .5,.75,.9,.95,.99,.999;require source pair accuracy>=.8,>=5 artificial+natural hard recoveries,<=1percent good5->bad10 both combined and native-only. Maximize recoveries then fewer damaged then higher threshold;none -> abstain everywhere.',
        inference='Same fixed R0 input and detection. Emit all8 native coordinates or their fixed quarter permutation. Boxes/scores/keypoint confidence/center/other detections bit exact. No frame rejection or new geometry filter. This does not repair coordinate localization itself.',
        real_evaluation='Freeze both checkpoints/thresholds before any real194 descriptors/decisions;freeze all outputs before GT scoring. Full194 and already frozen159. No threshold sweep using realGT.',
        real_gate='>=5 hard>20to<=10 recoveries across>=3frames,<=1percent good5to>10 damage,PCK20>=R0. Report >40,>80,>100 tails and exact user example. Screen not goal completion.',
        caveats='Constructed wrong assignments are not natural validation errors. Source128 plus real32 are not independent final evaluation;oldR0/source exposure and correlated pseudo teachers. Existing Replay had9manual training images,3infull194. No automatic promotion.',
        new_annotations=0,new_tags=0,auto_promote=False)
    C.freeze(DOC/'PROTOCOL.json',protocol);R.evaluation_protocol(PHASE,ARMS,sources+[C.bound(DOC/'PROTOCOL.json')])
    print('IDENTITY_PROTOCOL_LOCKED',len(records),'source',len(chosen),'real',flush=True)


def verify():
    p=C.read(DOC/'PROTOCOL.json')
    for b in p['sources']:C.verify(b)
    return p


def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=np.load(path);assert set(old.files)==set(arrays)
        for k,v in arrays.items():np.testing.assert_array_equal(old[k],v)
    else:np.savez(path,**arrays)


def source_features():
    p=verify();N.setup();data=SourceData().data;xx=[];ee=[];diagonal=[];hashes=[]
    for i,r in enumerate(p['source_records']):
        C.verify(r['image']);C.verify(r['label']);row=r['row'];a=data.arrays
        q=np.array(a['points'][row]);b=np.array(a['boxes'][row])
        xx.append(F.describe(a['p3'][row],a['p4'][row],q,b,a['input_shape'][row]))
        # First form descriptors without labels; only then form supervision.
        meta=data.source['records'][int(data.indices[row])]
        gain,_=C.N.E.old('features').canvas_affine(meta['prepared_shape_hw'],a['input_shape'][row])
        ee.append(pair_errors(q,a['gt_points'][row],a['gt_valid'][row])/gain)
        diagonal.append(float(np.linalg.norm(b[2:]-b[:2]))/gain)
        hashes.append({k:N.array_sha(a[k][row]) for k in ['p3','p4','points','boxes','input_shape','gt_points','gt_valid']})
        if (i+1)%128==0:print('IDENTITY_SOURCE_FEATURES',i+1,'/640',flush=True)
    x=np.asarray(xx,np.float32);e=np.asarray(ee,np.float32);mean=np.nanmean(e,axis=-1)
    eligible=np.abs(mean[:,0]-mean[:,1])>.01*np.asarray(diagonal)
    save_npz(RAW/'source.npz',x=x,errors=e,y=mean.argmin(-1),eligible=eligible,
        train=np.array([r['split']=='train' for r in p['source_records']]),diagonal=np.asarray(diagonal))
    C.freeze(DOC/'SOURCE_COMPLETE.json',dict(features=C.bound(RAW/'source.npz'),row_hashes=hashes,
        eligible_train=int((eligible&np.array([r['split']=='train' for r in p['source_records']])).sum()),
        eligible_val=int((eligible&np.array([r['split']=='val' for r in p['source_records']])).sum())))


def capture_descriptor(extractor,image,original):
    captured=extractor.predict(image,cpu_features=True);features=C.N.E.old('features')
    fresh=captured['candidates'][captured['selected_index']];old=top(original)
    np.testing.assert_allclose(fresh['box_xyxy'],old['box_xyxy'],atol=.002,rtol=0)
    np.testing.assert_allclose(fresh['keypoints_xy'],old['keypoints_xy'],atol=.002,rtol=0)
    gain,offset=features.canvas_affine(captured['canvas_shape'],captured['input_shape'])
    q=(np.asarray(old['keypoints_xy'])+100)*gain+offset
    box=((np.asarray(old['box_xyxy']).reshape(2,2)+100)*gain+offset).reshape(4)
    x=F.describe(captured['p3'].numpy(),captured['p4'].numpy(),q,box,captured['input_shape'])
    return x,q,box,gain,offset


@torch.no_grad()
def real_features():
    p=verify();N.setup();N.E.gpu();assert torch.cuda.is_available();torch.backends.cudnn.allow_tf32=True
    rows=[r for r in C.read(R.BASE_RAW/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC']
    extractor=C.N.E.old('features').FrozenYoloFeatures(C.N.E.R0);xx=[];ee=[];diagonal=[]
    try:
        for i,r in enumerate(rows):
            C.verify(r['image']);image=cv2.imread(str(C.ROOT/r['image']['path']));assert image is not None
            x,q,box,gain,offset=capture_descriptor(extractor,image,r['raw']);xx.append(x)
            target=(np.asarray(top(r['refined'])['keypoints_xy'])+100)*gain+offset
            labels,_=P.paired_labels(r);valid=np.asarray(labels['REF'].split(),float)[5:].reshape(9,3)[:,2]==2
            ee.append(pair_errors(q,target,valid)/gain);diagonal.append(float(np.linalg.norm(box[2:]-box[:2]))/gain)
            if (i+1)%80==0:print('IDENTITY_REAL_FEATURES',i+1,'/249',N.E.gpu(),flush=True)
    finally:extractor.close()
    e=np.asarray(ee,np.float32);mean=np.nanmean(e,axis=-1)
    eligible=np.abs(mean[:,0]-mean[:,1])>.01*np.asarray(diagonal)
    save_npz(RAW/'real.npz',x=np.asarray(xx,np.float32),errors=e,y=mean.argmin(-1),eligible=eligible,
        train=np.array([r['id'] in p['real_train'] for r in rows]))
    C.freeze(DOC/'REAL_COMPLETE.json',dict(features=C.bound(RAW/'real.npz'),ids=[r['id'] for r in rows],
        eligible=int(eligible.sum()),new_annotations=0,no_evaluation_features=True))


def change_counts(initial,after):
    valid=np.isfinite(initial);a=initial[valid];b=after[valid];assert np.isfinite(b).all()
    hard=a>20;good=a<5
    return dict(hard=int(hard.sum()),recovered=int((hard&(b<=10)).sum()),good=int(good.sum()),
        damaged=int((good&(b>10)).sum()),damage_rate=float((good&(b>10)).sum()/max(good.sum(),1)))


def calibrate(prob,errors,y,eligible):
    accuracy=float((prob.argmax(-1)[eligible]==y[eligible]).mean());rows=[]
    for threshold in [.5,.75,.9,.95,.99,.999]:
        c0=np.array([F.choose(v,threshold) for v in prob]);c1=1-np.array([F.choose(v[::-1],threshold) for v in prob])
        idx=np.arange(len(errors));clean=change_counts(errors[:,0],errors[idx,c0])
        combined=change_counts(np.concatenate([errors[:,0],errors[:,1]]),np.concatenate([errors[idx,c0],errors[idx,c1]]))
        passed=accuracy>=.8 and clean['damage_rate']<=.01 and combined['damage_rate']<=.01 and combined['recovered']>=5
        rows.append(dict(threshold=threshold,clean=clean,combined=combined,passed=passed))
    passed=[r for r in rows if r['passed']]
    chosen=max(passed,key=lambda r:(r['combined']['recovered'],-r['combined']['damaged'],r['threshold'])) if passed else None
    return dict(pair_accuracy=accuracy,grid=rows,selected=chosen,threshold=chosen['threshold'] if chosen else None)


def fit(arm):
    verify();N.setup();N.E.gpu();assert torch.cuda.is_available()
    for name in ['SOURCE','REAL']:C.verify(C.read(DOC/f'{name}_COMPLETE.json')['features'])
    s=np.load(RAW/'source.npz');r=np.load(RAW/'real.npz')
    source=s['train']&s['eligible'];real=r['train']&r['eligible'];assert source.sum()>=400 and real.sum()>=150
    if (DOC/f'FIT_{arm}.json').exists():C.verify(C.read(DOC/f'FIT_{arm}.json')['checkpoint']);print('ALREADY_FIT',arm);return
    mean=s['x'][source].mean((0,1,2));std=np.maximum(s['x'][source].std((0,1,2)),.001)
    norm=lambda x:np.clip((x-mean)/std,-8,8).astype(np.float32)
    sx=torch.as_tensor(norm(s['x'][source]),device='cuda');sy=torch.as_tensor(s['y'][source],device='cuda')
    rx=torch.as_tensor(norm(r['x'][real]),device='cuda');ry=torch.as_tensor(r['y'][real],device='cuda')
    torch.manual_seed(9023);torch.cuda.manual_seed_all(9023);model=F.Ranker().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    rng=np.random.default_rng(9023);history=[];started=time.monotonic()
    for step in range(800):
        ii=rng.integers(0,len(sx),32);jj=rng.integers(0,len(rx),16)
        x=sx[ii] if arm=='SYN' else torch.cat([sx[ii[:16]],rx[jj]])
        y=sy[ii] if arm=='SYN' else torch.cat([sy[ii[:16]],ry[jj]])
        optimizer.zero_grad(set_to_none=True);logits=model(x);loss=TF.cross_entropy(logits,y)
        assert torch.isfinite(loss);loss.backward();optimizer.step();history.append(float(loss.detach()))
        if (step+1)%200==0:print('IDENTITY_FIT',arm,step+1,'/800',history[-1],N.E.gpu(),flush=True)
    model.eval()
    with torch.no_grad():
        val=~s['train'];vp=model(torch.as_tensor(norm(s['x'][val]),device='cuda')).softmax(-1).cpu().numpy()
        held=~r['train'];rp=model(torch.as_tensor(norm(r['x'][held]),device='cuda')).softmax(-1).cpu().numpy()
    calibration=calibrate(vp,s['errors'][val],s['y'][val],s['eligible'][val])
    path=RAW/f'{arm}.pt';assert not path.exists()
    torch.save(dict(model_state_dict=model.state_dict(),mean=mean,std=std,arm=arm,step=800,
        protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
    C.freeze(DOC/f'FIT_{arm}.json',dict(complete=True,step=800,checkpoint=C.bound(path),
        source=C.bound(RAW/'source.npz'),real=C.bound(RAW/'real.npz'),protocol=C.bound(DOC/'PROTOCOL.json'),
        calibration=calibration,pseudo_probe_accuracy_NOT_truth=float((rp.argmax(-1)==r['y'][held]).mean()),
        source_train=int(source.sum()),real_train=0 if arm=='SYN' else int(real.sum()),seconds=time.monotonic()-started,
        history=history,real_evaluation_seen=False,auto_promoted=False))
    print('IDENTITY_FIT_COMPLETE',arm,'source_accuracy',calibration['pair_accuracy'],'threshold',calibration['threshold'],flush=True)


@torch.no_grad()
def infer():
    p=verify();N.setup();N.E.gpu();torch.backends.cudnn.allow_tf32=True
    fits={a:C.read(DOC/f'FIT_{a}.json') for a in ARMS}
    C.freeze(DOC/'DECISION_LOCK.json',dict(fits={a:C.bound(DOC/f'FIT_{a}.json') for a in ARMS},before_any_real_eval_features=True))
    models={};normalizers={}
    for a,f in fits.items():
        C.verify(f['checkpoint']);ck=torch.load(C.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json')
        model=F.Ranker().cuda();model.load_state_dict(ck['model_state_dict']);models[a]=model.eval()
        normalizers[a]=(ck['mean'],ck['std'])
    originals={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    rows={a:[] for a in ARMS};decisions=[];extractor=C.N.E.old('features').FrozenYoloFeatures(C.N.E.R0)
    try:
        for i,r in enumerate(C.read(DOC/'EVAL_PROTOCOL.json')['records']):
            C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
            old=originals[r['id']];x,_,_,_,_=capture_descriptor(extractor,im,old['prediction']);choices={}
            for a,model in models.items():
                mean,std=normalizers[a];xx=torch.as_tensor(np.clip((x-mean)/std,-8,8)[None],device='cuda')
                probability=model(xx).softmax(-1)[0].cpu().numpy();threshold=fits[a]['calibration']['threshold']
                choice=F.choose(probability,threshold);out=copy.deepcopy(old['prediction'])
                if choice:top(out)['keypoints_xy'][:8]=np.asarray(top(old['prediction'])['keypoints_xy'])[F.QUARTER][:8].tolist()
                assert_preserved(old['prediction'],out)
                rows[a].append(dict(id=r['id'],kind='PLASTIC',prediction=out,raw_hw=old['raw_hw']))
                choices[a]=dict(probabilities=probability.tolist(),threshold=threshold,choice=choice)
            decisions.append(dict(id=r['id'],choices=choices))
            if (i+1)%50==0:print('IDENTITY_EVAL_INFERENCE',i+1,'/194',flush=True)
    finally:extractor.close()
    for a in ARMS:C.freeze(RAW/f'EVAL_PREDICTIONS_{a}.json',dict(complete=True,arm=a,checkpoint=fits[a]['checkpoint'],records=rows[a],GT_free=True))
    C.freeze(RAW/'DECISIONS.json',decisions)
    C.freeze(DOC/'OUTPUTS_LOCK.json',dict(artifacts=[C.bound(RAW/f'EVAL_PREDICTIONS_{a}.json') for a in ARMS]+[C.bound(RAW/'DECISIONS.json')],before_GT_scoring=True))


def report():
    p=verify()
    for b in C.read(DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    scores={a:R.score(PHASE,a,False)['metrics'] for a in ARMS}
    scores['R0']=[r for r in C.read(R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC']
    ids=[r['id'] for r in scores['R0']];scores={a:[{r['id']:r for r in rr}[i] for i in ids] for a,rr in scores.items()}
    subset=C.read(R.BASE_DOC/'large_corner_recovery_v1/extreme_low_subset_v1/SUBSET_MANIFEST.json')
    keep={r['id'] for r in subset['records'] if r['kind']=='PLASTIC'};results={}
    for label,keys in [('full194',set(ids)),('retained159',keep)]:
        b=[r for r in scores['R0'] if r['id'] in keys];table={}
        for a,rr in scores.items():
            n=[r for r in rr if r['id'] in keys];d=recovery_damage(b,n,True)
            frames=sum(x['matched'] and any(v is not None and v>20 and w<=10 for v,w in zip(x['canonical_errors'],y['canonical_errors'])) for x,y in zip(b,n))
            tails={str(t):dict(hard=sum(v>t for x in b if x['matched'] for v in x['canonical_errors'] if v is not None),
                recovered=sum(v is not None and v>t and w<=10 for x,y in zip(b,n) if x['matched'] for v,w in zip(x['canonical_errors'],y['canonical_errors']))) for t in [40,80,100]}
            table[a]=dict(summary=summary(n),recovery=d,recovered_frames=frames,tails=tails)
        results[label]=table
    decisions=C.read(RAW/'DECISIONS.json');counts={a:sum(r['choices'][a]['choice'] for r in decisions) for a in ARMS}
    checks={a:dict(recovery5=results['full194'][a]['recovery']['recovered']>=5,frames3=results['full194'][a]['recovered_frames']>=3,
        damage1percent=results['full194'][a]['recovery']['damage_rate']<=.01,
        pck20_preserved=results['full194'][a]['summary']['PCK']['20']>=results['full194']['R0']['summary']['PCK']['20']) for a in ARMS}
    example='eval_pallet07:1778652166837872128'
    ex={a:next(r for r in rr if r['id']==example) for a,rr in scores.items()}
    result=dict(results=results,checks=checks,passed={a:all(v.values()) for a,v in checks.items()},changed_frames=counts,
        example=ex,goal_complete=False,new_annotations=0,new_tags=0,auto_promoted=False,independent_confirmation=False)
    C.freeze(DOC/'RESULTS.json',result)
    lines=['# RGB 특징으로 코너 대응 후보 비교','',
        '기존 합성 정답으로 SYN, 합성+기존 실사 수도레이블로 MIX를 학습했다. 새 레이블·태그0. 두 후보의 실제 점 위치 집합은 같고 역할 번호만 다르다. 공식 평가 대칭은 C2 그대로다. 검증된 source threshold를 실사에서 바꾸지 않았다.','',
        '| 평가 | 모델 | PCK20 % | 중앙 px | P90 px | >20→≤10 복구 | 정상 점 손상 | >100→≤10 복구 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for label,table in results.items():
        for a in ['R0',*ARMS]:
            x=table[a];s=x['summary'];d=x['recovery']
            lines.append(f'| {label} | {a} | {100*s["PCK"]["20"]:.2f} | {s["matched_pooled_corner8_median_px"]:.3f} | {s["matched_pooled_corner8_P90_px"]:.3f} | {d["recovered"]}/{d["hard"]} | {d["damaged"]}/{d["good"]} | {x["tails"]["100"]["recovered"]}/{x["tails"]["100"]["hard"]} |')
    lines+=['',f'실제 번호 변경 이미지: {counts}. 사전 screen: {result["passed"]}.',
        '의자·콘 예시 평균오차: '+str({a:r['frame_mean_px'] for a,r in ex.items()}),'',
        '합성 검증의 인위적 번호 교란 복구와 실사 자연 오류 복구는 다르다. 이전 source-selector의 coverage 실패를 고치거나 기준을 낮추지 않은 별도 절대 후보 분류 실험이다. 반복 DEV 결과이며 독립 확인이나 최종 모델 승격이 아니다.']
    C.write_text(RAW/'REPORT_KO.md','\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','source','real','fit','infer','report']);p.add_argument('arm',nargs='?',choices=ARMS);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='source':source_features()
    elif a.action=='real':real_features()
    elif a.action=='fit':fit(a.arm)
    elif a.action=='infer':infer()
    else:report()
