"""Reproducible E0/E1/E2/E4/E7 adapters for the 2026-09-30 diagnosis.

Only this experiment's new directories are writable. Predictions retain the
existing selection, confidence, invalid-point and centroid contracts. GEO is
the previously selected Linear94 model, not the D9 heuristic.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from collections import Counter
import numpy as np
import torch
import cv2
from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M
from scripts.research.pallet_clean_to_pose_transfer_v1 import eval_student as E
from scripts.research.pallet_clean_to_pose_transfer_v1 import common as IO
from scripts.research.pallet_selector_recovery_v1 import features as F, models as G, common as U
from scripts.research.pallet_posefix_limited_adaptation_pilot_v1 import common as L
from scripts.research.pallet_clean_pose_minimal_v1.final_audit import bootstrap

ROOT=IO.ROOT
NAME='pallet_pose_diagnosis_20260930_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
HERE=Path(__file__).resolve().parent
read=IO.read
bind=IO.bind
verify=IO.verify
TOL=1e-7

def save(path, value):
    path=Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(RAW)
    path.parent.mkdir(parents=True,exist_ok=True)
    text=value if isinstance(value,str) else json.dumps(M.clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with path.open('x') as f:f.write(text)

def ledger(stage,start,cpu,**extra):
    import resource
    save(RAW/f'COST_{stage}.json',dict(stage=stage,wall_seconds=time.monotonic()-start,
        CPU_seconds=time.process_time()-cpu,peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        fits=0,optimizer_updates=0,GPU_seconds=0,image_forwards=0,**extra))

def metadata():return read(RAW/'METADATA.json')

def geo_checkpoint():
    lock=read(ROOT/'_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer/SCORER_SELECTION_LOCK.json')
    assert lock['winner']=='GEO_LINEAR';verify(lock['checkpoint'])
    return torch.load(ROOT/lock['checkpoint']['path'],map_location='cpu',weights_only=False)

def candidates(pred,row,ck):
    record=O.candidate_record(pred,row)
    feat=F.extract(pred,row['K'],row['xyz'],row['hw'])
    assert feat['selection']==record['selected_name']
    name=record['selected_name'];scores=None
    if feat['valid']:
        scores=G.scores(ck,np.asarray(feat['features'],np.float32)[None])[0]
        name=U.HYP[int(G.selection(scores[None],U.HYP)[0])]
    selected=next((h['pose'] for h in record['hypotheses'] if h['name']==name),record['current'])
    return dict(record,GEO_name=name,GEO_pose=selected,GEO_scores=scores,
                GEO_fallback=not feat['valid'])

def metric(fid,p,g):return M.extend_metric(O.D.metric(fid,p,g),p,g)

def summarize(rows):return M.summarize(rows)

def paired(before,after,rows,with_ci=True):
    ids=[r['id'] for r in rows];out=M.paired(before,after,ids)
    # Existing M.paired uses strict signs; explicitly supply the locked tolerance
    # for diagnostic direction counts without changing continuous statistics.
    sign=lambda x:'IMPROVE' if x < -TOL else 'WORSEN' if x > TOL else 'TIE'
    count=Counter('T_'+sign(after[i]['translation_cm']-before[i]['translation_cm'])+'__R_'+sign(after[i]['rotation_deg']-before[i]['rotation_deg']) for i in ids if before[i]['available'] and after[i]['available'])
    out['direction_counts_tolerance_1e_7']=dict(count)
    if with_ci:
        out['recording_bootstrap']=dict(bootstrap(before,after,rows),repeats=2000,seed=20260929,
            caveat='Percentile intervals are conditional on finite draws; undefined draws explicitly counted. Reused DEV.')
        out['leave_one_recording_out']={rec:M.paired(before,after,[r['id'] for r in rows if r['recording']!=rec])['difference_of_conditional_medians'] for rec in sorted({r['recording'] for r in rows})}
    return out

def prepare():
    start,cpu=time.monotonic(),time.process_time()
    assert not (DOC/'RUN_MANIFEST.json').exists()
    DOC.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    save(DOC/'REQUEST_PLAN.txt',Path('/home/minjae/Downloads/pallet_pose_goal_plan_20260930.txt').read_text())
    save(RAW/'INITIAL_GIT_STATUS.txt',subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True))
    rows=read(E.paths(42)['metadata']);E.validate_membership(rows)
    split=read(ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json')
    assert [r['id'] for r in rows]==[r['id'] for r in split['heldout']]
    bindings=[]
    for r in split['heldout']:
        for key in ('image','annotation'):verify(r[key]);bindings.append(r[key])
    p=read(L.DOC/'EXPERIMENT_PROTOCOL.json');lock=read(L.DOC/'PREDICTIONS_LOCK.json')
    for b in [lock['baseline'],lock['predictions']['FULL'],lock['checkpoints']['FULL'],p['base']]:verify(b);bindings.append(b)
    frozen=read(ROOT/lock['baseline']['path'])['predictions']
    preds={'identity':frozen['R0'],'PRIOR1':frozen['POSEFIX_SYNTH'],
        'FULL125':read(ROOT/lock['predictions']['FULL']['path'])['predictions'],
        'OLD_REF217':read(E.paths(42)['predictions'])['OLD_REF']}
    ids=[r['id'] for r in rows]
    for a in preds:
        assert set(ids)<=set(preds[a]);preds[a]={i:preds[a][i] for i in ids}
    # Original FULL125 input parity, not merely same image ID.
    oldr0=read(E.paths(42)['predictions'])['R0']
    for i in ids:
        O.D.close(preds['identity'][i],oldr0[i])
        for a in ('PRIOR1','FULL125'):
            L.E.assert_preserved(preds['identity'][i],preds[a][i])
            orig=L.E.P.top(preds['identity'][i]);out=L.E.P.top(preds[a][i])
            if orig is not None:
                q=np.array(orig['keypoints_xy']);z=np.array(out['keypoints_xy'])
                valid=np.isfinite(q).all(1)&~(q==-1).all(1)
                np.testing.assert_array_equal(q[~valid],z[~valid]);np.testing.assert_array_equal(q[8],z[8])
    old=set(p['evaluation_populations']['PRIMARY_OCC96']);new={r['id'] for r in rows if r['severity']!='CLEAN'}
    assert (len(old),len(new),len(old&new))==(93,99,92)
    sourcepaths=[E.paths(42)['metadata'],E.paths(42)['predictions'],E.paths(42)['lock'],L.DOC/'EXPERIMENT_PROTOCOL.json',L.DOC/'PREDICTIONS_LOCK.json',
        ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json',ROOT/'_docs/experiments/pallet_pose_objective_followup_v2/METRIC_AND_SELECTION_LOCK.json',
        ROOT/'_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer/SCORER_SELECTION_LOCK.json']
    geo=read(sourcepaths[-1])['checkpoint'];verify(geo);bindings.append(geo)
    save(RAW/'METADATA.json',rows);save(RAW/'INPUT_PREDICTIONS.json',preds)
    save(RAW/'POPULATIONS.json',dict(groups=E.group_ids(rows),old_natural93=sorted(old),common92=sorted(old&new),old_only=sorted(old-new),current_only=sorted(new-old)))
    plans={
      'E0':('입력·평가 계약','기존 잠금·동일 HEAD','ID/좌표/모델 혼동','S1/S2/예측 lock','일치 시 평가; 불일치 시 해당 수치 차단','필수 계약 불일치'),
      'E1':('FULL125 2D→T/R 연결','old93 2D만 있음','보정·후보 선택 효과','full128 좌표·GEO·참조','동일 W/D와 GEO 재선택 비교','동일 입력 불가'),
      'E2':('후보·검출 여지','D9 oracle은 GEO와 다름','오선택 대 pool 밖 실패','모든 저장 박스/후보','단일 pose T-best/R-best; 검출 꼬리 구분','후보 없음'),
      'E3':('동일 clean29 RGB×좌표','TRAIN probe는 새 이미지 아님','가림 RGB 대 좌표/crop','R0/PRIOR1/FULL125·RGB','CO/OO·controlled/native 비교','입력/모델 계약 실패'),
      'E4':('사람 가시성 부분 개입','old93 검토169','visible/occluded 코너','사람 검토+canonical 매핑','고정 W/D/GEO 참조교체','identity 또는 매칭 미확정'),
      'E5':('조건부 전이·오류구조','TRAIN50 중49 복구','노출량/전이/구조','E1/E3·기존TRAIN','결정에 필요한 단일 추가 비교만','추가 비교가 결정 불변'),
      'E6':('실사 지도 기준','REALFT_A 2D 결과','현재 계약에서 개선 가능성','정확한 캐시/체크포인트','T/R·검출·중복 별도 보고','정확 모델 없음'),
      'E7':('참조·recording 신뢰성','재사용 DEV','집단/참조 불확실성','모든 결과·원클릭/재클릭','paired bootstrap/LORO; noise floor 구분','실측 잡음 자료 없음')}
    manifest=dict(objective='단안 RGB 팔레트 6D pose의 clean→가림 전이에서 약한 연결을 검증하고 자연 가림 T/R 공동 개선을 위한 다음 개입 하나를 선택한다.',
        started_at=IO.now(),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),root=str(ROOT),
        request_document=bind(DOC/'REQUEST_PLAN.txt'),scope=dict(new_fits=0,new_collection=0,GT_edits=0,fixed_model_inference_allowed=True),
        environment=dict(python=os.sys.executable,torch=torch.__version__,opencv=cv2.__version__,numpy=np.__version__,cuda_available=torch.cuda.is_available(),device='CPU',threads=4),
        inputs=bindings+[bind(x) for x in sourcepaths],codes=[bind(Path(m.__file__)) for m in (O,O.D,O.D.Pose,F,G,M,E,L)],
        population=bind(RAW/'POPULATIONS.json'),model_aliases=dict(identity='R0 cached selected detection',PRIOR1='POSEFIX_SYNTH baseline cache',FULL125='historical FULL last300; crop expansion 1.25',OLD_REF217='separate REF_LR5 self-trained student; not refiner causal contrast'),
        contract=dict(T='100*norm(centroid_pred-reference) cm',R='physical-frame C2 full rotation degrees; no W/D90 equivalence',
            reference='stored geometry-derived annotation/K/dimensions, not independent physical GT',
            failure='Full denominator retained; invalid=positive infinity with explicit status. Conditional and common-valid separate.',
            no_pose_IoU_gate=True,corner_order='camera-facing0123; top0145; bottom2367; center8 retained',
            points='original image pixel xy; invalid restored; confidence/box unchanged by refiner',
            crop='axis_aligned_crop_matrix default1.25; RGB288x384; original-image inverse affine',
            selector='frozen GEO_LINEAR shared Linear94, lower score then name tie; invalid pair uses existing D9',
            branch_hold='same W/D hypothesis, new continuous R/t solve; not identical internal PnP root',tie_tolerance=TOL),
        experiments={k:dict(zip(['purpose','existing_answer','competing_explanation','inputs','decision','stop'],v)) for k,v in plans.items()},
        budgets=dict(E1=384,E3=725,E5B=174,E5stress=116,E6=128,E1source=1024,E2ROI=99),
        field_mapping=dict(BASE='PRIOR1',FULL='FULL125',R0='identity',REF_LR5='OLD_REF217',GEO_name='frozen learned selector W/D'))
    save(DOC/'RUN_MANIFEST.json',manifest)
    ledger('E0',start,cpu,reused_prediction_frames=512)
    print('E0_COMPLETE',Counter(r['severity'] for r in rows),flush=True)

def e1():
    start,cpu=time.monotonic(),time.process_time();rows=metadata();preds=read(RAW/'INPUT_PREDICTIONS.json');ck=geo_checkpoint()
    frozen={}
    for arm,pp in preds.items():
        frozen[arm]={r['id']:candidates(pp[r['id']],r,ck) for r in rows}
        print('CANDIDATES',arm,len(frozen[arm]),flush=True)
    save(RAW/'E1_CANDIDATES.json',frozen)
    # All operational selection finished before opening current evaluation references.
    _,gt=O.D.Pose.metadata('REAL_DEV')
    results={};cm={}
    for arm,rr in frozen.items():
        cm[arm]={i:[dict(name=h['name'],metric=metric(i,h['pose'],gt[i])) for h in rec['hypotheses']] for i,rec in rr.items()}
        results[arm]={i:metric(i,rec['GEO_pose'],gt[i]) for i,rec in rr.items()}
        if arm in ('PRIOR1','FULL125'):
            results[arm+'_held_identity']={}
            for i,rec in rr.items():
                name=frozen['identity'][i]['GEO_name'];p=next((h['pose'] for h in rec['hypotheses'] if h['name']==name),dict(available=False))
                results[arm+'_held_identity'][i]=metric(i,p,gt[i])
    save(RAW/'E1_POSE_METRICS.json',results);save(RAW/'E1_CANDIDATE_METRICS.json',cm)
    two=read(L.RAW/'FRAME_METRICS.json')
    oldtwo=read(O.P.RAW/'FRAME_METRICS.json')
    twomap=dict(identity=two['R0'],PRIOR1=two['BASE'],FULL125=two['FULL'],OLD_REF217=oldtwo['REF_LR5'])
    save(RAW/'E1_2D_METRICS.json',{a:{r['id']:v[r['id']] for r in rows} for a,v in twomap.items()})
    summaries={}
    for group,ids in E.group_ids(rows).items():
        rr=[r for r in rows if r['id'] in ids]
        sums={a:summarize(v[i] for i in ids) for a,v in results.items()}
        pairs={a+'-minus-identity':paired(results['identity'],results[a],rr,group in ('NATURAL99','CLEAN29')) for a in ('PRIOR1','FULL125','PRIOR1_held_identity','FULL125_held_identity')}
        pairs['FULL125-minus-PRIOR1']=paired(results['PRIOR1'],results['FULL125'],rr,group in ('NATURAL99','CLEAN29'))
        switches={a:sum(frozen[a][i]['GEO_name']!=frozen['identity'][i]['GEO_name'] for i in ids) for a in ('PRIOR1','FULL125')}
        stats2={}
        for a,v in twomap.items():
            vals=[e for i in ids if v[i]['matched'] for e in v[i]['errors'] if e is not None]
            stats2[a]=dict(matched=sum(v[i]['matched'] for i in ids),detected=sum(v[i]['detected'] for i in ids),N=len(ids),cornerN=len(vals),median=float(np.median(vals)),P90=float(np.quantile(vals,.9)),PCK10=float(np.mean(np.array(vals)<=10)))
        summaries[group]=dict(models=sums,paired=pairs,branch_switch=switches,twoD_matched=stats2)
    save(DOC/'E1_SUMMARY.json',summaries)
    e2(rows,preds,frozen,results,cm,gt,ck)
    ledger('E1_E2',start,cpu,reused_prediction_frames=512)
    print('E1_E2_COMPLETE',flush=True)

def e2(rows,preds,frozen,results,cm,gt,ck):
    oracle={};details={}
    for a in preds:
        alternatives={kind:{} for kind in ('T_best','R_best')};detail={}
        for r in rows:
            i=r['id'];base=results[a][i];pool=[h for h in cm[a][i] if h['metric']['available']]
            detail[i]={}
            for kind,key in [('T_best','translation_cm'),('R_best','rotation_deg')]:
                h=min(pool,key=lambda h:(h['metric'][key],h['name'])) if pool else None
                alternatives[kind][i]=h['metric'] if h else dict(id=i,available=False)
                detail[i][kind]=h['name'] if h else None
            detail[i]['any_single_candidate_both_improves']=bool(base['available'] and any(h['metric']['translation_cm']<base['translation_cm']-TOL and h['metric']['rotation_deg']<base['rotation_deg']-TOL for h in pool))
        oracle[a]={group:{kind:dict(summary=summarize(v[i] for i in ids),paired=paired(results[a],v,[r for r in rows if r['id'] in ids],group in ('NATURAL99','CLEAN29'))) for kind,v in alternatives.items()} for group,ids in E.group_ids(rows).items()}
        details[a]=detail
    # Detection audit includes every cached box, no new ROI inference.
    truth2=read(L.E.V.RAW/'E1_FRAME_METRICS.json')['truth_for_display_only']
    old2=read(L.RAW/'FRAME_METRICS.json')['R0']
    split={r['id']:r for r in read(ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json')['heldout']}
    detection={}
    from scripts.research.pallet_dim_conditioned_p_v1 import eval_math
    for r in rows:
        i=r['id'];ann=read(ROOT/split[i]['annotation']['path'])
        # Store all boxes and GT annotation binding. Annotation bbox schema is
        # resolved explicitly below; no bbox inferred from arbitrary points.
        detection[i]=dict(id=i,recording=r['recording'],severity=r['severity'],selected_index=preds['identity'][i]['selected_index'],
            matched=old2[i]['matched'],detected=old2[i]['detected'],translation_cm=results['identity'][i].get('translation_cm'),
            annotation=split[i]['annotation'],boxes=preds['identity'][i]['candidates'])
    save(RAW/'E2_ORACLE_CHOICES.json',details);save(RAW/'E2_DETECTION_INPUTS.json',detection)
    save(DOC/'E2_ORACLE_SUMMARY.json',dict(models=oracle,semantics='Each oracle selects one whole pose. Different T/R minima never combined. Pool-limited diagnostic.',
        both_gain_counts={a:{g:sum(details[a][i]['any_single_candidate_both_improves'] for i in ids) for g,ids in E.group_ids(rows).items()} for a in details}))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','e1']);a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    {'prepare':prepare,'e1':e1}[a.stage]()

if __name__=='__main__':main()
