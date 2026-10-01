"""Describe frozen sign-versus-Newton source choices; no new selector or raw reference read."""
from . import common as C
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

SCALE=np.array([2.4636887551191258,1.113474019956766],np.float64)
MODELS=C.MODEL_NAMES
ORACLE=C.ROOT/'_docs/experiments/pallet_pose_selector_objective_audit_20261001_v1'
READS=[]


def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=os.fsdecode(args[0])
    forbidden=('GEOMETRY_SIDETABLE','GEOMETRY_RESOLVED_POSE_GT','AXIS_REVIEW_MANIFEST',
        '/data/evaluation/','/SOURCE_MANIFEST.json','/DIMENSION_SIDECAR.json',
        '/SYNTH_RECORDS.json','/SOURCE_TRAIN_LABELS.npz','/SOURCE_FEATURES.npz',
        '/fits/','SOURCE_POSES.json')
    assert not any(token in p for token in forbidden),('FROZEN_RESULTS_ONLY',p)
    assert Path(p).suffix.lower() not in ('.jpg','.jpeg','.png','.webp','.pt','.pth'),('NO_IMAGE_OR_CHECKPOINT',p)
    if str(C.ROOT) in p:READS.append(p)


def distribution(values):
    values=np.asarray(values,np.float64)
    assert np.isfinite(values).all()
    if not len(values):return dict(n=0,median=None,P90=None,maximum=None,mean=None)
    return dict(n=len(values),median=float(np.median(values)),P90=float(np.quantile(values,.9)),
        maximum=float(np.max(values)),mean=float(np.mean(values)))


def load_run(doc,raw):
    gate=C.read(doc/'SOURCE_VAL_GATE.json');lock=C.read(doc/'SOURCE_VAL_ROUTING_LOCK.json')
    assert gate['complete'] and lock['complete']
    assert gate['routing_lock']==C.bind(doc/'SOURCE_VAL_ROUTING_LOCK.json')
    assert lock['choices']==C.bind(raw/'SOURCE_VAL_CHOICES.json')
    assert gate['metrics']==C.bind(raw/'SOURCE_VAL_METRICS.npz')
    assert gate['protocol']==lock['protocol']==C.bind(doc/'TRAIN_PROTOCOL.json')
    choice=C.read(raw/'SOURCE_VAL_CHOICES.json')
    assert choice['models']==list(MODELS) and len(choice['ids'])==len(set(choice['ids']))==1024
    with np.load(raw/'SOURCE_VAL_METRICS.npz') as stored:
        assert stored['ids'].tolist()==choice['ids']
        errors={m:stored[m].copy() for m in stored['models']}
    assert all(v.shape==(1024,2) and np.isfinite(v).all() for v in errors.values())
    for m,v in errors.items():
        expected=gate['summaries'][m]['full_population']
        for a,key in enumerate(('translation_cm','rotation_deg')):
            assert np.median(v[:,a])==expected[key]['median']
            assert abs(np.quantile(v[:,a],.9)-expected[key]['P90'])<=1e-12
    bindings=[C.bind(doc/name) for name in ('SOURCE_VAL_GATE.json','SOURCE_VAL_ROUTING_LOCK.json','TRAIN_PROTOCOL.json')]
    bindings+=[C.bind(raw/name) for name in ('SOURCE_VAL_CHOICES.json','SOURCE_VAL_METRICS.npz')]
    return gate,lock,choice,errors,bindings


def actual_classes(choice,values,anchor):
    index=np.array([r['candidate_index'] for r in choice],int)
    anchor_index=np.array([r['anchor_index'] for r in choice],int)
    assert (index>=0).all() and all(r['pose_available'] and r['status']=='SELECTED' for r in choice)
    same=index==anchor_index
    np.testing.assert_array_equal(values[same],anchor[same])
    delta=values-anchor
    safe=(delta<=0).all(1)
    improving=safe & (delta<0).any(1) & ~same
    equal=safe & ~(delta<0).any(1) & ~same
    unsafe=(delta>0).any(1)
    labels=np.where(same,'anchor',np.where(improving,'safe_improvement',np.where(equal,'safe_equal','unsafe')))
    assert np.sum(same)+np.sum(improving)+np.sum(equal)+np.sum(unsafe)==len(choice)
    count={name:int(np.sum(labels==name)) for name in ('anchor','safe_improvement','safe_equal','unsafe')}
    severity=np.maximum(np.max(delta/SCALE,axis=1),0.)
    violations={name:int(mask.sum()) for name,mask in (
        ('T',delta[:,0]>0),('R',delta[:,1]>0),('T_only',(delta[:,0]>0)&(delta[:,1]<=0)),
        ('R_only',(delta[:,1]>0)&(delta[:,0]<=0)),('both',(delta>0).all(1)))}
    direction={}
    for a,Tsign in enumerate((-1,0,1)):
        for b,Rsign in enumerate((-1,0,1)):
            direction[f'T{Tsign:+d}_R{Rsign:+d}']=int(np.sum((np.sign(delta[:,0])==Tsign)&(np.sign(delta[:,1])==Rsign)&~same))
    return dict(frames=len(choice),failed=0,classes=count,
        class_fraction_full_population={k:v/len(choice) for k,v in count.items()},
        nonanchor_count=int((~same).sum()),violations=violations,
        nonanchor_direction_counts=direction,
        unsafe_normalized_excess=distribution(severity[unsafe])),labels,delta


def axis_diagnostic(prediction,target,mask):
    result=dict(candidate_pairs=int(mask.sum()),axes={})
    for axis,name in enumerate(('T','R')):
        p=prediction[:,:,axis][mask];y=target[:,:,axis][mask];conf=np.zeros((3,3),int)
        for true,pred in zip(np.sign(y).astype(int)+1,np.sign(p).astype(int)+1):conf[true,pred]+=1
        result['axes'][name]=dict(MAE=float(np.mean(abs(p-y))) if len(p) else None,
            RMSE=float(np.sqrt(np.mean((p-y)**2))) if len(p) else None,
            sign_accuracy=float(np.mean(np.sign(p)==np.sign(y))) if len(p) else None,
            true_negative_count=int(np.sum(y<0)),true_positive_count=int(np.sum(y>0)),
            actual_worse_predicted_nonworse=int(np.sum((y>0)&(p<=0))),
            actual_improved_predicted_nonimproving=int(np.sum((y<0)&(p>=0))),
            confusion_true_rows_predicted_columns=conf.tolist(),sign_order=['negative','zero','positive'])
    return result


def main():
    started=time.monotonic();sys.addaudithook(guard)
    verification=C.read(C.DOC/'SOURCE_VAL_VERIFICATION.json')
    assert verification['complete'] and verification['PASS']
    assert verification['source_gate']==C.bind(C.DOC/'SOURCE_VAL_GATE.json')
    assert verification['choices']==C.bind(C.RAW/'SOURCE_VAL_CHOICES.json')
    gate,lock,choices,errors,bindings=load_run(C.DOC,C.RAW)
    old_gate,old_lock,old_choices,old_errors,old_bindings=load_run(C.NEWTON_DOC,C.NEWTON_RAW)
    historical_gate,historical_lock,historical_choices,historical_errors,historical_bindings=load_run(C.RBF_DOC,C.RBF_RAW)
    for key in ('poses','metadata','feature_lock','source_predictions_lock','source_contract'):
        assert lock[key]==old_lock[key]==historical_lock[key],('SOURCE_POOL_IDENTITY',key)
    assert choices['ids']==historical_choices['ids']
    assert choices['ids']==old_choices['ids']
    assert gate['checks_total']==45 and gate['checks_passed']==43 and not gate['PASS']
    assert gate['failed_checks']==['UNION_s3/R0_ONLY/translation_cm_median_strict','UNION_s3/R0_GEO/translation_cm_median_strict']
    assert not gate['real_routing_authorized'] and not old_gate['PASS']
    assert old_gate['checks_passed']==43 and old_gate['failed_checks']==gate['failed_checks']
    assert gate['source_reference_bindings']==old_gate['source_reference_bindings']
    for m in ('R0_GEO','DIVERSE251_s1_GEO','DIVERSE251_s2_GEO','DIVERSE251_s3_GEO'):
        np.testing.assert_array_equal(errors[m],old_errors[m])
        np.testing.assert_array_equal(errors[m],historical_errors[m])
        assert errors[m].tobytes()==old_errors[m].tobytes()==historical_errors[m].tobytes()
    # Equal reported medians do not imply per-frame equality of R0_ONLY/GEO.
    train=C.read(C.DOC/'TRAIN_CONVERGENCE.json')
    assert train['complete'] and train['PASS'] and train['source_TRAIN_only']
    assert train['protocol']==gate['protocol']
    verification=C.read(C.DOC/'SOURCE_VAL_VERIFICATION.json')
    assert verification['complete'] and verification['PASS']
    assert verification['source_gate']==C.bind(C.DOC/'SOURCE_VAL_GATE.json')
    assert verification['choices']==C.bind(C.RAW/'SOURCE_VAL_CHOICES.json')
    assert not verification['source_gate_PASS'] and verification['checks_passed']==43
    oracle=C.read(ORACLE/'VAL_ORACLE.json')
    C.verify(oracle['protocol'])
    oracle_protocol=C.read(C.ROOT/oracle['protocol']['path'])
    assert oracle_protocol['feature_lock']==lock['feature_lock']
    assert oracle['complete'] and oracle['frames']==1024
    np.testing.assert_array_equal(oracle['scale'],SCALE)
    C.verify(oracle['rows'])
    with (C.ROOT/oracle['rows']['path']).open() as f:oracle_rows=list(csv.DictReader(f))
    assert len(oracle_rows)==3072
    oracle_by={(r['id'],int(r['seed'])):r for r in oracle_rows}
    assert len(oracle_by)==3072
    ids=choices['ids'];anchor=errors['R0_GEO'];models={};max_cache_difference=0.
    for model in MODELS:
        rows=[choices['records'][model][fid] for fid in ids]
        before=[old_choices['records'][model][fid] for fid in ids]
        older=[historical_choices['records'][model][fid] for fid in ids]
        now_stats,labels,delta=actual_classes(rows,errors[model],anchor)
        old_stats,old_labels,_=actual_classes(before,old_errors[model],anchor)
        selected_index=np.array([r['candidate_index'] for r in rows],int)
        anchor_index=np.array([r['anchor_index'] for r in rows],int)
        names=['R0:long-face-front','R0:short-face-front']
        if model!='R0_ONLY':names += [f'DIVERSE251_s{model[-1]}:long-face-front',f'DIVERSE251_s{model[-1]}:short-face-front']
        prediction=np.array([r['predicted_signed_axes'] for r in rows],np.float64)
        previous_prediction=np.array([r['predicted_signed_axes'] for r in before],np.float64)
        assert previous_prediction.shape==prediction.shape
        assert np.isfinite(previous_prediction).all()
        assert prediction.shape==(1024,len(names),2) and np.isfinite(prediction).all()
        assert all(all(r['candidate_valid']) for r in rows+before)
        np.testing.assert_array_equal(prediction[np.arange(1024),anchor_index],0.)
        assert np.all(np.max(prediction[np.arange(1024),selected_index],axis=1)<=0.)
        # Only previously scored identities enter this partial candidate cache.
        cache=[{} for _ in ids]
        def add(i,name,value):
            nonlocal max_cache_difference
            assert name in names
            value=np.array(value,np.float64);assert np.isfinite(value).all()
            if name in cache[i]:
                d=float(np.abs(cache[i][name]-value).max());max_cache_difference=max(max_cache_difference,d)
                np.testing.assert_allclose(cache[i][name],value,rtol=0,atol=1e-10)
            else:cache[i][name]=value
        for i,(fid,row,previous) in enumerate(zip(ids,rows,before)):
            assert row['anchor_name']==previous['anchor_name']==older[i]['anchor_name']==names[anchor_index[i]]
            assert previous['anchor_index']==older[i]['anchor_index']==anchor_index[i]
            for identity in (row,previous,older[i]):
                assert identity['candidate_name']==names[identity['candidate_index']]
            add(i,row['anchor_name'],anchor[i]);add(i,row['candidate_name'],errors[model][i])
            add(i,previous['candidate_name'],old_errors[model][i])
            add(i,older[i]['candidate_name'],historical_errors[model][i])
            if model!='R0_ONLY':
                rr=oracle_by[(fid,int(model[-1]))]
                for prefix in ('learned','oracle'):
                    add(i,rr[prefix],[float(rr[prefix+'_T_cm']),float(rr[prefix+'_R_deg'])])
        truth=np.zeros_like(prediction);known=np.zeros((1024,len(names)),bool)
        for i,record in enumerate(cache):
            for name,value in record.items():
                j=names.index(name);known[i,j]=True
                excess=(value-anchor[i])/SCALE
                truth[i,j]=np.sign(excess)*np.log1p(abs(excess))
        assert known[np.arange(1024),selected_index].all()
        selected=np.zeros_like(known);selected[np.arange(1024),selected_index]=True
        nonanchor=known.copy();nonanchor[np.arange(1024),anchor_index]=False
        selected_nonanchor=selected & nonanchor
        previous_selected=np.zeros_like(known)
        previous_selected[np.arange(1024),[r['candidate_index'] for r in before]]=True
        assert known[previous_selected].all()
        previous_selected_nonanchor=previous_selected & nonanchor
        actual_safe=(truth<=0).all(2) & (truth<0).any(2) & nonanchor
        predicted_safe=(prediction<=0).all(2) & (prediction<0).any(2) & nonanchor
        opportunity=actual_safe.any(1)
        captured=opportunity & (labels=='safe_improvement')
        missed=opportunity & ~(labels=='safe_improvement')
        previous_captured=opportunity & (old_labels=='safe_improvement')
        previous_missed=opportunity & ~(old_labels=='safe_improvement')
        assert int(captured.sum())==now_stats['classes']['safe_improvement']
        assert int(previous_captured.sum())==old_stats['classes']['safe_improvement']
        same_hyp=np.array([r['hypothesis']==r['anchor_name'].split(':')[1] for r in rows])
        parent_D=np.array([r['parent']!='R0' for r in rows])
        switched=np.array([r['candidate_name']!=b['candidate_name'] for r,b in zip(rows,before)])
        transitions={a+' -> '+b:int(np.sum((old_labels==a)&(labels==b)))
            for a in ('anchor','safe_improvement','safe_equal','unsafe')
            for b in ('anchor','safe_improvement','safe_equal','unsafe')}
        tr=train['models'][model]
        models[model]=dict(actual_source_selection=now_stats,previous_Newton_actual_selection=old_stats,
            changed_identity_from_previous_Newton=int(switched.sum()),class_transitions_from_Newton=transitions,
            actual_expert_WD=dict(DIVERSE_selected=int(parent_D.sum()),
                selected_hypothesis_differs_from_R0_anchor=int((~same_hyp).sum()),
                DIVERSE_same_hypothesis=int((parent_D&same_hyp).sum()),
                DIVERSE_opposite_hypothesis=int((parent_D&~same_hyp).sum())),
            known_safe_opportunities=dict(frames_with_demonstrated_opportunity=int(opportunity.sum()),
                actual_safe_capture=int(captured.sum()),miss_lower_bound=int(missed.sum()),
                miss_when_anchor=int(np.sum(missed&(labels=='anchor'))),
                miss_when_unsafe=int(np.sum(missed&(labels=='unsafe'))),
                previous_Newton_capture_on_same_known_pool=int(previous_captured.sum()),
                previous_Newton_miss_lower_bound_on_same_known_pool=int(previous_missed.sum()),
                frames_with_complete_error_pool=int(known.all(1).sum()),
                known_nonanchor_pairs=int(nonanchor.sum()),possible_nonanchor_pairs=1024*(len(names)-1),
                known_safe_pairs=int(actual_safe.sum()),
                known_candidate_safe_false_negative=int((actual_safe&~predicted_safe).sum()),
                known_predicted_safe_pairs=int(predicted_safe.sum()),
                known_candidate_safe_false_positive=int((predicted_safe&~actual_safe).sum()),
                scope='Shared partial error cache from actual current sign, previous Newton, historical RBF choices and already published old oracle rows. Current/prior misses use the same known pool. Counts are lower bounds, not full-pool recall; no new candidate scoring or oracle selection.'),
            selected_axis_regression=axis_diagnostic(prediction,truth,selected),
            selected_nonanchor_axis_regression=axis_diagnostic(prediction,truth,selected_nonanchor),
            known_nonanchor_axis_regression=axis_diagnostic(prediction,truth,nonanchor),
            previous_Newton_selected_axis_regression=axis_diagnostic(previous_prediction,truth,previous_selected),
            previous_Newton_selected_nonanchor_axis_regression=axis_diagnostic(previous_prediction,truth,previous_selected_nonanchor),
            previous_Newton_known_nonanchor_axis_regression=axis_diagnostic(previous_prediction,truth,nonanchor),
            source_T_delta=distribution(delta[:,0]),source_R_delta=distribution(delta[:,1]),
            source_metric_summary=gate['summaries'][model],previous_Newton_metric_summary=old_gate['summaries'][model],
            fixed_TRAIN=dict(frames=train['frames'],available_rows=train['available_anchor_rows'],failed_rows=train['failed_rows'],
                risk_statistics=tr['risk_statistics'],regression_statistics=tr['regression_statistics'],
                statistics=tr['statistics'],numerical_certificate=tr['independent_recompute'],
                previous_Newton_risk_statistics=tr['previous_fixed_newton']['risk_statistics'],
                previous_Newton_regression_statistics=tr['previous_fixed_newton']['regression_statistics'],
                same_new_objective_decrease=tr['same_new_objective_decrease'],
                same_new_objective_at_previous_weights=tr['previous_fixed_newton']['same_new_objective']))
    out=dict(complete=True,PASS=True,status='FIXED_SOURCE_DIAGNOSTIC_COMPLETE_METHOD_GATE_STILL_FAIL',created_at=C.now(),
        models=models,source_gate=dict(PASS=False,passed=43,total=45,failed_checks=gate['failed_checks']),
        current_protocol=gate['protocol'],source_VAL_reused_development_set=True,
        source_scale=SCALE.tolist(),cache_duplicate_error_max_difference=max_cache_difference,
        known_candidate_errors_are_partial=True,all_source_rows_retained=1024,
        no_new_selector_policy_evaluated=True,new_argmin_computations=0,new_fits=0,new_image_forwards=0,
        new_PnP_calls=0,raw_GT_reads=0,real_reference_reads=0,new_real_routes=0,threshold_sweeps=0,
        source_labels_recomputed_from_raw=False,new_candidate_physical_error_computations=0,
        predicted_axes_source='Already frozen SOURCE_VAL_CHOICES.json; no checkpoint/model forward in this diagnostic.',
        interpretation='Convergence certifies the fixed objective, not joint medians. Actual selected safe/unsafe and sign errors are observational; partial known opportunities give a missed-opportunity lower bound only.',
        comparison_control='Previous fixed signed_axes_newton models; no seed/checkpoint selection.',
        historical_RBF_cache_role='Additional already-scored candidate errors only; not the primary comparison control.',
        new_intervention_proposed=False,method_success=False,goal_complete=False,
        bindings=bindings+old_bindings+historical_bindings+[C.bind(C.DOC/'TRAIN_CONVERGENCE.json'),
            C.bind(C.DOC/'SOURCE_VAL_VERIFICATION.json'),C.bind(ORACLE/'VAL_ORACLE.json'),oracle['rows'],oracle['protocol'],
            C.bind(ORACLE/'PRIOR_OBJECTIVE_AUDIT_KO.md'),C.bind(__file__)],
        read_paths=sorted(set(READS)),wall_seconds=time.monotonic()-started)
    C.save(C.DOC/'SOURCE_TRANSFER_DIAGNOSTIC_KO.md',report(out))
    C.save(C.DOC/'SOURCE_TRANSFER_DIAGNOSTIC.json',out)
    print('FIXED_SOURCE_DIAGNOSTIC_COMPLETE',json.dumps({m:{'source':r['actual_source_selection']['classes'],
        'known_miss_lower_bound':r['known_safe_opportunities']['miss_lower_bound']} for m,r in models.items()}),flush=True)


def report(x):
    models=x['models'];s3=models['UNION_s3']['source_metric_summary']['full_population']['translation_cm']['median']
    baseline=C.read(C.DOC/'SOURCE_VAL_GATE.json')['summaries']['R0_GEO']['full_population']['translation_cm']['median']
    lines=['# 부호 항 추가 후의 고정 source 선택 진단','',
        '**네 모델의 수렴 검산은 PASS이나 source 조건은43/45이며 전체 FAIL이다.** 직전 Newton을 비교 control로 고정했다. 이미 잠긴 선택·두 축 예측·기존 물리 오류만 요약했다. 새 fit/forward/argmin/threshold/GT 재채점/실사 routing은0이다. 진단 PASS는 방법의 성공이 아니다.','',
        f'UNION_s3 T 중앙값은 {s3:.12f}cm이며 R0_GEO 및 R0_ONLY의 {baseline:.12f}cm보다 크다. 실패는 두 T strict-median 조건이고 이전 Newton과 같은 범주다. 작은 차이에 tolerance를 새로 넣거나 seed1·2만 채택하지 않는다. 이번 모델의 실사 성능은 평가하지 않았다.','',
        '## 고정 Newton 대비 실제 선택','',
        '각 모델1,024행 전체이며 실패0행이다. anchor는 운영 R0 GEO와 후보 정체성이 같음, safe 개선은 다른 후보에서 두 실제 오차가 비증가하고 적어도 하나가 엄격 감소함이다. 정확한 부등호를 쓰며 근사 tolerance를 적용하지 않는다.','',
        '| 모델 | anchor 이전→현재 | safe 개선 이전→현재 | unsafe 이전→현재 | 바뀐 후보 |','|---|---:|---:|---:|---:|']
    for m,r in models.items():
        a=r['previous_Newton_actual_selection']['classes'];b=r['actual_source_selection']['classes']
        lines.append(f"| {m} | {a['anchor']} → {b['anchor']} | {a['safe_improvement']} → {b['safe_improvement']} | {a['unsafe']} → {b['unsafe']} | {r['changed_identity_from_previous_Newton']} |")
    lines+=['','## 실제 선택의 위험과 부호','',
        '선택 score는 두 예측의 max이고 anchor 예측은0이다. 선택한 두 예측이0 이하라는 사실은 실제 두 축 비악화를 보장하지 않는다. 아래는 실제 선택에서만 집계한 위반이다.','',
        '| 모델 | 선택 nonanchor | T 악화 | R 악화 | 양축 악화 | unsafe 위험 P90 |','|---|---:|---:|---:|---:|---:|']
    for m,r in models.items():
        z=r['actual_source_selection'];v=z['violations'];p=z['unsafe_normalized_excess']['P90']
        lines.append(f"| {m} | {z['nonanchor_count']} | {v['T']} | {v['R']} | {v['both']} | {'해당 없음' if p is None else f'{p:.6f}'} |")
    lines+=['','위험은 `max((T−T_anchor)/sT,(R−R_anchor)/sR,0)`이다. JSON의 MAE/RMSE는 signed-log1p 단위이며 cm/degree가 아니다. anchor0을 포함한 정확도를 학습된 분류 능력으로 해석하지 않도록 selected/nonanchor/known-pool 통계를 나눴다. 이전과 현재의 selected 집단은 서로 달라 직접 인과 비교 대상이 아니며, known nonanchor는 같은 부분후보 집합으로 비교한다.','',
        '| 모델 | 같은 known nonanchor 쌍 | T sign 이전→현재 | R sign 이전→현재 |','|---|---:|---:|---:|']
    for m,r in models.items():
        a=r['previous_Newton_known_nonanchor_axis_regression'];b=r['known_nonanchor_axis_regression']
        assert a['candidate_pairs']==b['candidate_pairs']
        lines.append(f"| {m} | {b['candidate_pairs']} | {100*a['axes']['T']['sign_accuracy']:.2f}% → {100*b['axes']['T']['sign_accuracy']:.2f}% | {100*a['axes']['R']['sign_accuracy']:.2f}% → {100*b['axes']['R']['sign_accuracy']:.2f}% |")
    lines+=['','## 이미 확인된 개선 기회와 미포착 하한','',
        '현재 sign·직전 Newton·더 오래된 RBF의 실제 선택, 기존 oracle CSV에 남은 learned/oracle 후보의 이미 채점된 오류만 합쳤다. RBF는 부분 캐시 보충 자료이며 이번 비교 control은 Newton이다. source poses/metadata/features/prediction lock 및 기존 oracle protocol의 feature-lock SHA를 대조했다. 전체 후보의 참조 오류를 새로 계산하거나 새로운 oracle/argmin을 만들지 않았다.','',
        '아래 기회와 miss는 이 부분 캐시에서 확인된 하한이다. 전체 후보 recall/false-negative가 아니다. 이전·현재 모두 **같은 합쳐진 known pool**로 계산하므로, 앞선 진단의 더 작은 캐시에서 발표한 하한과 직접 비교하지 않는다. 다른 safe 후보를 선택해도 capture다.','',
        '| 모델 | 확인된 기회 | safe capture 이전→현재 | miss 하한 이전→현재 | 현재 miss 중 anchor | 알려진 nonanchor/가능 수 |','|---|---:|---:|---:|---:|---:|']
    for m,r in models.items():
        z=r['known_safe_opportunities']
        lines.append(f"| {m} | {z['frames_with_demonstrated_opportunity']} | {z['previous_Newton_capture_on_same_known_pool']} → {z['actual_safe_capture']} | {z['previous_Newton_miss_lower_bound_on_same_known_pool']} → {z['miss_lower_bound']} | {z['miss_when_anchor']} | {z['known_nonanchor_pairs']}/{z['possible_nonanchor_pairs']} |")
    lines+=['','## TRAIN과 source를 구분한 관측','',
        'TRAIN은 기존 독립 검산의2,598행(유효2,597·실패1) 값을 인용한다. 새 목적식은 이전 Newton의 고정 weight에서도 채점된 동일 목적식과 비교해 감소했으나, Huber-only와 Huber+sign의 서로 다른 원시 loss값을 직접 비교하지 않는다. 수렴과 목적식 감소가 source45 통과를 보장하지 않는다.','',
        '| 모델 | TRAIN safe 이전→현재 | TRAIN unsafe 이전→현재 | source safe 이전→현재 | source unsafe 이전→현재 |','|---|---:|---:|---:|---:|']
    for m,r in models.items():
        a=r['fixed_TRAIN']['previous_Newton_risk_statistics']['classes'];b=r['fixed_TRAIN']['risk_statistics']['classes']
        c=r['previous_Newton_actual_selection']['classes'];d=r['actual_source_selection']['classes']
        lines.append(f"| {m} | {a['safe_improvement']['count']} → {b['safe_improvement']['count']} | {a['unsafe']['count']} → {b['unsafe']['count']} | {c['safe_improvement']} → {d['safe_improvement']} | {c['unsafe']} → {d['unsafe']} |")
    lines+=['','이번 고정 계수1의 부호 항 추가는 최종 source gate 실패를 해소하지 못했다. 세부 sign/선택 변화는 관측 결과이며, 표현력·후보 정보·감독 목표·합성 지원범위 중 무엇이 유일 원인인지는 확정하지 않는다. 이 자료는 반복 사용된 source VAL 개발 집합이며 새로운 독립 검증이나 실사 개선의 증거가 아니다.','',
        '이 진단은 다음 loss·weight·threshold·seed를 제안하거나 시험하지 않는다. source45의 어느 조건이라도 실패했으므로 이번 모델의 실사 routing은 금지된 상태다.','',
        '[진단 JSON](SOURCE_TRANSFER_DIAGNOSTIC.json) · [source 판정](SOURCE_VAL_GATE.json) · [독립 source 검산](SOURCE_VAL_VERIFICATION_KO.md) · [TRAIN 검산](TRAIN_CONVERGENCE_KO.md) · [직전 Newton 진단](../pallet_pose_signed_axes_newton_20261001_v1/SOURCE_TRANSFER_DIAGNOSTIC_KO.md)','']
    return '\n'.join(lines)


if __name__=='__main__':main()
