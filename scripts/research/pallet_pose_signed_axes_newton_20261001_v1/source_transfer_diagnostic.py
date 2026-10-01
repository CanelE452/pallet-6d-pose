"""Describe already frozen source choices; no new selector or raw reference read."""
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
    gate,lock,choices,errors,bindings=load_run(C.DOC,C.RAW)
    old_gate,old_lock,old_choices,old_errors,old_bindings=load_run(C.RBF_DOC,C.RBF_RAW)
    assert choices['ids']==old_choices['ids']
    assert gate['checks_total']==45 and gate['checks_passed']==43 and not gate['PASS']
    assert gate['failed_checks']==['UNION_s3/R0_ONLY/translation_cm_median_strict','UNION_s3/R0_GEO/translation_cm_median_strict']
    assert not gate['real_routing_authorized'] and old_gate['PASS']
    assert gate['source_reference_bindings']==old_gate['source_reference_bindings']
    for m in ('R0_GEO','DIVERSE251_s1_GEO','DIVERSE251_s2_GEO','DIVERSE251_s3_GEO'):
        np.testing.assert_array_equal(errors[m],old_errors[m])
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
        now_stats,labels,delta=actual_classes(rows,errors[model],anchor)
        old_stats,old_labels,_=actual_classes(before,old_errors[model],anchor)
        selected_index=np.array([r['candidate_index'] for r in rows],int)
        anchor_index=np.array([r['anchor_index'] for r in rows],int)
        names=['R0:long-face-front','R0:short-face-front']
        if model!='R0_ONLY':names += [f'DIVERSE251_s{model[-1]}:long-face-front',f'DIVERSE251_s{model[-1]}:short-face-front']
        prediction=np.array([r['predicted_signed_axes'] for r in rows],np.float64)
        assert prediction.shape==(1024,len(names),2) and np.isfinite(prediction).all()
        assert all(all(r['candidate_valid']) for r in rows)
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
            assert row['anchor_name']==previous['anchor_name']==names[anchor_index[i]]
            add(i,row['anchor_name'],anchor[i]);add(i,row['candidate_name'],errors[model][i])
            add(i,previous['candidate_name'],old_errors[model][i])
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
        actual_safe=(truth<=0).all(2) & (truth<0).any(2) & nonanchor
        predicted_safe=(prediction<=0).all(2) & (prediction<0).any(2) & nonanchor
        opportunity=actual_safe.any(1)
        captured=opportunity & (labels=='safe_improvement')
        missed=opportunity & ~(labels=='safe_improvement')
        assert int(captured.sum())==now_stats['classes']['safe_improvement']
        same_hyp=np.array([r['hypothesis']==r['anchor_name'].split(':')[1] for r in rows])
        parent_D=np.array([r['parent']!='R0' for r in rows])
        switched=np.array([r['candidate_name']!=b['candidate_name'] for r,b in zip(rows,before)])
        transitions={a+' -> '+b:int(np.sum((old_labels==a)&(labels==b)))
            for a in ('anchor','safe_improvement','safe_equal','unsafe')
            for b in ('anchor','safe_improvement','safe_equal','unsafe')}
        tr=train['models'][model]
        models[model]=dict(actual_source_selection=now_stats,previous_RBF_actual_selection=old_stats,
            changed_identity_from_previous_RBF=int(switched.sum()),class_transitions_from_RBF=transitions,
            actual_expert_WD=dict(DIVERSE_selected=int(parent_D.sum()),
                selected_hypothesis_differs_from_R0_anchor=int((~same_hyp).sum()),
                DIVERSE_same_hypothesis=int((parent_D&same_hyp).sum()),
                DIVERSE_opposite_hypothesis=int((parent_D&~same_hyp).sum())),
            known_safe_opportunities=dict(frames_with_demonstrated_opportunity=int(opportunity.sum()),
                actual_safe_capture=int(captured.sum()),miss_lower_bound=int(missed.sum()),
                miss_when_anchor=int(np.sum(missed&(labels=='anchor'))),
                miss_when_unsafe=int(np.sum(missed&(labels=='unsafe'))),
                frames_with_complete_error_pool=int(known.all(1).sum()),
                known_nonanchor_pairs=int(nonanchor.sum()),possible_nonanchor_pairs=1024*(len(names)-1),
                known_safe_pairs=int(actual_safe.sum()),
                known_candidate_safe_false_negative=int((actual_safe&~predicted_safe).sum()),
                known_predicted_safe_pairs=int(predicted_safe.sum()),
                known_candidate_safe_false_positive=int((predicted_safe&~actual_safe).sum()),
                scope='Partial error cache from actual current/RBF choices and already published old oracle rows. Miss count is a lower bound, not full-pool recall; no new candidate scoring or oracle selection.'),
            selected_axis_regression=axis_diagnostic(prediction,truth,selected),
            selected_nonanchor_axis_regression=axis_diagnostic(prediction,truth,selected_nonanchor),
            known_nonanchor_axis_regression=axis_diagnostic(prediction,truth,nonanchor),
            source_T_delta=distribution(delta[:,0]),source_R_delta=distribution(delta[:,1]),
            source_metric_summary=gate['summaries'][model],previous_RBF_metric_summary=old_gate['summaries'][model],
            fixed_TRAIN=dict(frames=train['frames'],available_rows=train['available_anchor_rows'],failed_rows=train['failed_rows'],
                risk_statistics=tr['risk_statistics'],regression_statistics=tr['regression_statistics'],
                statistics=tr['statistics'],numerical_certificate=tr['independent_recompute'],
                previous_RBF_risk_statistics=tr['previous_fixed_rbf']['risk_statistics']))
    proposal=dict(status='PROPOSAL_ONLY_NOT_EXECUTED',single_change='Add an axis-sign logistic term to the unchanged signed-axis Huber objective.',
        motivation='The existing max-axis runtime depends on the zero-sign boundary, while smooth Huber fits magnitudes. Frozen TRAIN and selected VAL sign confusions expose errors at that boundary; this does not prove representation or supervision is the unique cause.',
        input='Same fixed253 candidate-minus-R0-anchor features, candidates, normalization, four models, and TRAIN2598 rows.',
        labels='Use sign of frozen TRAIN physical axis excess only. Exclude identically-zero anchor/zero-target scalar terms from the new sign term; keep their zero Huber contribution and full frame denominator.',
        objective='J_new=mean_all2598[mean_original_valid_candidates_and_2axes(Huber(p-y,1)+1{y!=0}*softplus(-sign(y)*p))]+(1e-4/2)||W||². Logistic coefficient fixed1, no reweighting or sweep; no division by only nonzero labels.',
        mathematics='Logistic term is convex in each linear axis prediction; ridge preserves strong convexity. Its gradient remains nonzero near a wrongly predicted sign even when the physical gain magnitude is small.',
        runtime='Unchanged maximum of the two predicted axes, original exact ties, no real/source reference or extra threshold.',
        execution='Separate predeclared namespace, four zero-initialized fits, same Newton/Armijo numerical budgets and certifications; unchanged45 source checks/all3 seeds and both original/matched real5 categories. Any source failure keeps real unrouted.',
        distinction='Old four-way/pairwise CE learns a winner/order; old risk-margin CE protects a hard candidate target. This proposed auxiliary term labels each physical T/R sign while retaining continuous two-axis Huber. It is not a claim that logistic classification or mixed regression/classification is novel.',
        counterevidence='Structured DHT already combines classification with2D regression and failed transfer; earlier pairwise/risk/continuous-gain methods also failed. More sign emphasis can accept small true gains yet harm tails, or preserve anchor too often. No gate success follows from the proposed loss.',
        selection_basis='Pre-existing TRAIN sign confusions and the semantic sign boundary; no choice of weights, thresholds, epochs or seed from the two failed VAL checks.')
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
        next_single_intervention=proposal,method_success=False,goal_complete=False,
        bindings=bindings+old_bindings+[C.bind(C.DOC/'TRAIN_CONVERGENCE.json'),
            C.bind(C.DOC/'SOURCE_VAL_VERIFICATION.json'),C.bind(ORACLE/'VAL_ORACLE.json'),oracle['rows'],
            C.bind(ORACLE/'PRIOR_OBJECTIVE_AUDIT_KO.md'),C.bind(__file__)],
        read_paths=sorted(set(READS)),wall_seconds=time.monotonic()-started)
    C.save(C.DOC/'SOURCE_TRANSFER_DIAGNOSTIC_KO.md',report(out))
    C.save(C.DOC/'SOURCE_TRANSFER_DIAGNOSTIC.json',out)
    print('FIXED_SOURCE_DIAGNOSTIC_COMPLETE',json.dumps({m:{'source':r['actual_source_selection']['classes'],
        'known_miss_lower_bound':r['known_safe_opportunities']['miss_lower_bound']} for m,r in models.items()}),flush=True)


def report(x):
    lines=['# 수렴한 두 축 회귀의 source 선택 진단','',
        '**수렴 검산은 네 모델 모두 PASS이나 source 조건은43/45이며 전체 FAIL이다.** 이 진단은 이미 잠긴 선택·예측·오차만 다시 요약한다. 새 fit, checkpoint forward, 후보 argmin, threshold sweep, 물리 참조 재채점, 실사 조회·routing은0이다. 진단 PASS는 방법의 gate PASS가 아니다.','',
        'UNION_s3의 T 중앙값1.682086665cm가 R0_GEO 및 R0_ONLY의1.674594149cm보다 커 두 strict-median 조건을 실패했다. 세 모델 중 좋은 것만 선택하거나 이 차이를 tolerance로 지우지 않는다. 나머지43조건이 통과했다는 사실도 전체 실패를 대체하지 않는다.','',
        '## 실제 선택과 이전 RBF의 비교','',
        '다음은 각1,024행 전체의 실제 선택이다. anchor는 후보 정체성이 운영 R0 GEO와 같은 경우이고, safe 개선은 두 실제 오차가 비증가하면서 적어도 한 축이 엄격 개선된 다른 후보다. 실패행은 현재0이며, 근사 허용값으로 safe/unsafe를 나누지 않았다.','',
        '| 모델 | anchor 이전→현재 | safe 개선 이전→현재 | unsafe 이전→현재 | 바뀐 후보 |','|---|---:|---:|---:|---:|']
    for m,r in x['models'].items():
        a=r['previous_RBF_actual_selection']['classes'];b=r['actual_source_selection']['classes']
        lines.append(f"| {m} | {a['anchor']} → {b['anchor']} | {a['safe_improvement']} → {b['safe_improvement']} | {a['unsafe']} → {b['unsafe']} | {r['changed_identity_from_previous_RBF']} |")
    lines+=['','## 두 축 예측 부호와 실제 선택의 위험','',
        '선택된 score는 max(predicted T,predicted R)이고 anchor 예측은0이므로 선택된 예측 두 축은 모두0 이하이다. 이것은 실제 물리 변화의 부호를 보장하지 않는다. 아래는 anchor를 제외한 **실제 선택**에서 실제 T 또는 R이 증가한 횟수다. 부호 오차는 고정 경계0으로만 계산하며 별도 calibration이나 threshold를 만들지 않았다.','',
        '| 모델 | 선택 nonanchor | T 악화 | R 악화 | 양축 악화 | unsafe 위험 P90 |','|---|---:|---:|---:|---:|---:|']
    for m,r in x['models'].items():
        z=r['actual_source_selection'];v=z['violations'];p=z['unsafe_normalized_excess']['P90']
        lines.append(f"| {m} | {z['nonanchor_count']} | {v['T']} | {v['R']} | {v['both']} | {'해당 없음' if p is None else f'{p:.6f}'} |")
    lines+=['','위험은 `max((T−T_anchor)/sT,(R−R_anchor)/sR,0)`이다. 회귀 MAE/RMSE는 signed-log1p 단위이며 cm/degree가 아니다. JSON에는 anchor를 포함한 선택과 제외한 선택을 나누어 두 축 confusion·MAE·RMSE를 기록했다. anchor의 예측·정답0을 학습된 sign 정확도로 해석하지 않는다.','',
        '## 이미 증명된 개선 기회와 놓친 기회의 하한','',
        '현재 결과의 오차 NPZ에는 실제 선택한 자세만 있다. 과거 source oracle CSV도 일부 선택만 남겼다. 따라서 기존 RBF의 실제 선택, 기존 oracle/learned CSV 및 현재 실제 선택의 **이미 채점된 후보**를 모아 확인했다. 전체 네 후보를 새로 채점하거나 새 oracle를 만들지 않았다. 기회 수와 miss는 이 부분 캐시가 증명하는 하한이며, 전체 pool recall/false-negative 수가 아니다. 다른 safe 후보를 선택해도 capture로 센다.','',
        '| 모델 | 증명된 기회 | 실제 safe capture | miss 하한 | miss 중 anchor | 알려진 nonanchor/가능 수 |','|---|---:|---:|---:|---:|---:|']
    for m,r in x['models'].items():
        z=r['known_safe_opportunities'];lines.append(f"| {m} | {z['frames_with_demonstrated_opportunity']} | {z['actual_safe_capture']} | {z['miss_lower_bound']} | {z['miss_when_anchor']} | {z['known_nonanchor_pairs']}/{z['possible_nonanchor_pairs']} |")
    lines+=['','## TRAIN과 source에서 관측한 것','',
        'TRAIN은 전체2,598행 중 유효2,597행·실패1행을 유지한 기존 독립 검산 값을 그대로 인용했다. source는1,024행 모두 유효하다. 다음 값은 각각 다른 모집단의 실제 고정 선택이며, 차이가 곧 도메인 이동의 인과 추정은 아니다.','',
        '| 모델 | TRAIN safe/unsafe (유효2597) | source safe/unsafe (1024) | TRAIN nonanchor T/R sign 일치율 |','|---|---:|---:|---:|']
    for m,r in x['models'].items():
        t=r['fixed_TRAIN'];a=t['risk_statistics']['classes'];b=r['actual_source_selection']['classes'];g=t['regression_statistics']['nonanchor_valid_candidates']['axes']
        lines.append(f"| {m} | {a['safe_improvement']['count']}/{a['unsafe']['count']} | {b['safe_improvement']}/{b['unsafe']} | {100*g['T']['sign_accuracy']:.2f}% / {100*g['R']['sign_accuracy']:.2f}% |")
    lines+=['','모든 네 모델은 같은 Huber+ridge 목적식의 강볼록 수렴 인증을 통과했다. 그 목적식은 후보의 연속 변화량 적합을 최적화하며, source population의 두 중앙값이나 위험한 선택의 부호 정확도를 직접 보장하지 않는다. 실제 unsafe 선택은 예측 부호 오류가 남았다는 관측이다. 표현력·합성 지원범위·감독 손실 가운데 무엇이 유일 원인인지는 이 진단으로 확정할 수 없다. 반복 사용된 source VAL이므로 새로운 독립 일반화 시험으로 부르지 않는다.','',
        '## 다음 개입 한 가지 제안 — 미실행','',
        '**기존 두 축 Huber에 각 물리 축의 sign logistic 항 하나를 추가하는 TRAIN-only 목적함수 비교**를 제안한다. 입력253·RBF·후보·두 축 출력·runtime max·기존 tie·λ·zero 초기화를 유지한다. `y!=0`인 각 scalar에 `softplus(−sign(y)·prediction)`을 계수1로 더하고, Huber와 동일한 원래 유효 후보×2축 평균 및 전체2,598행 분모를 유지한다. target0인 anchor/정확 동률은 logistic 항0이다. 계수·sign 경계·margin·seed를 VAL에서 탐색하지 않는다.','',
        '근거는 기존 TRAIN에서도 실제 개선을 양수로 예측하거나 실제 악화를 음수로 예측하는 오류가 남고, 운영 규칙이 두 예측의0 경계에 직접 의존한다는 점이다. Huber는 작은 물리 변화의 잘못된 부호에 작은 기울기를 줄 수 있지만 logistic 항은 이 경계에서 분류 신호를 유지한다. 기존 연속 Huber는 남기므로 변화 크기 감독을 버리지 않는다. 추가 항은 선형 예측에 대해 볼록하고 ridge의 강볼록성은 유지된다. 이는 기대 효과의 보장이 아니다.','',
        '이전4way CE는 한 후보 winner, pairwise는 후보 간 순서, risk-margin CE는 hard target 보호를 학습했다. 이번 제안은 각 T/R 축의 물리 변화 sign을 보조 감독한다는 차이가 있다. 그러나 분류+회귀를 섞는 일반 아이디어는 새롭지 않고, Structured DHT의2D 혼합 손실·signed2D gain·기존 RGB MLP 전이 실패를 반대 증거로 유지한다. 작은 개선의 수락이 tail을 해치거나 더 보수적인 anchor 유지로 돌아갈 수 있다. 표현력 부족이 해결된다고 주장하지 않는다.','',
        '별도 namespace에서4개 최종 fit만 허용하고 기존 Newton 예산·인증·source45/all3seed·실사 original/matched5 조건을 전부 유지해야 한다. 어느 source 조건이라도 실패하면 실사 routing은 계속 금지한다. 이번 진단에서 이 손실을 구현·학습하거나 새 선택 규칙을 평가하지 않았다.','',
        '[진단 JSON](SOURCE_TRANSFER_DIAGNOSTIC.json) · [현재 source 판정](SOURCE_VAL_GATE.json) · [독립 TRAIN 검산](TRAIN_CONVERGENCE_KO.md) · [선행 목적함수 감사](../pallet_pose_selector_objective_audit_20261001_v1/PRIOR_OBJECTIVE_AUDIT_KO.md)','']
    return '\n'.join(lines)


if __name__=='__main__':main()
