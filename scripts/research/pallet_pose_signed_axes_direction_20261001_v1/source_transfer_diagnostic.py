"""Post-result description of fixed P/N source choices; no selector execution."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from . import common as C
from scripts.research.pallet_pose_signed_axes_sign_20261001_v1 import source_transfer_diagnostic as N

PROTOCOL=C.DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC_PROTOCOL.json'
READS=set()
CLASSES=('anchor','safe_improvement','safe_equal','unsafe')


def input_paths():
    paths={}
    for tag,doc,raw in [('current',C.DOC,C.RAW),('previous_sign',C.SIGN_DOC,C.SIGN_RAW)]:
        for name in ('SOURCE_VAL_GATE.json','SOURCE_VAL_ROUTING_LOCK.json','SOURCE_VAL_VERIFICATION.json','TRAIN_PROTOCOL.json','TRAIN_CONVERGENCE.json'):
            paths[tag+'_'+name.removesuffix('.json').lower()]=doc/name
        for name in ('SOURCE_VAL_CHOICES.json','SOURCE_VAL_METRICS.npz'):
            paths[tag+'_'+name.split('.')[0].lower()]=raw/name
    paths['prior_objective_audit']=C.ROOT/'_docs/experiments/pallet_pose_selector_objective_audit_20261001_v1/PRIOR_OBJECTIVE_AUDIT_KO.md'
    paths['direction_design']=C.DOC/'DESIGN_KO.md'
    paths['direction_representation']=C.DIRECTION_DOC/'REPRESENTATION_AUDIT.json'
    return paths


def seal():
    assert not PROTOCOL.exists()
    value=dict(complete=True,created_at=C.now(),schema='fixed_direction_source_transfer_diagnostic_v1',
        inputs={k:C.bind(v) for k,v in input_paths().items()},
        codes=[C.bind(__file__),C.bind(C.__file__),C.bind(N.__file__)],
        supersedes_pre_analysis_name_only=C.bind(C.DOC/'SOURCE_TRANSFER_DIAGNOSTIC_PROTOCOL.json'),
        population='Same frozen1024 sourceVAL rows for all4 current and previous sign selectors; no frame exclusion.',
        scope=['Classify each actually selected fixed pose relative to unchanged R0 GEO anchor.',
               'Compare class/identity/expert/hypothesis transitions and paired T/R changes.',
               'Report difference of population medians separately from median paired per-frame changes.',
               'Identify the two existing central order-statistic rows for each1024-frame median.',
               'Inspect cached chosen-axis predictions against the already scored chosen errors only.',
               'Read fixed TRAIN summaries; do not load TRAIN labels or weights.',
               'Any opportunity counts use only current/prior selected poses plus anchor, not the full candidate pool.'],
        no_new_argmin=True,no_new_fit=True,no_image_or_PnP=True,no_raw_reference=True,
        thresholds_changed=False,performance_gate=False,method_success=False,goal_complete=False)
    C.save(PROTOCOL,value);C.save(C.DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC_PROTOCOL_SHA.json',C.bind(PROTOCOL))
    print('SOURCE_FIXED_CHOICE_DIAGNOSTIC_INPUTS_SEALED',C.bind(PROTOCOL),flush=True)


def guard(allowed):
    outputs={C.DOC/name for name in ('SOURCE_FIXED_CHOICE_DIAGNOSTIC.json','SOURCE_FIXED_CHOICE_DIAGNOSTIC_KO.md')}
    allowed=set(allowed)|{PROTOCOL,C.DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC_PROTOCOL_SHA.json'}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();name=str(path);mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing and path.is_relative_to(C.ROOT):assert path in outputs,('DIAGNOSTIC_OUTPUT_ONLY',name);return
        assert not any(t in name for t in ('GEOMETRY_SIDETABLE','GEOMETRY_RESOLVED_POSE_GT','AXIS_REVIEW_MANIFEST',
            '/data/evaluation/','/SOURCE_MANIFEST.json','/DIMENSION_SIDECAR.json','/SYNTH_RECORDS.json',
            'SOURCE_TRAIN_LABELS.npz','SOURCE_FEATURES.npz','SOURCE_POSES.json','/fits/')),('FROZEN_RESULTS_ONLY',name)
        assert path.suffix.lower() not in ('.png','.jpg','.jpeg','.webp','.pt','.pth'),name
        if path.is_relative_to(C.ROOT):
            assert path in allowed or path.suffix in ('.py','.pyc'),('SEALED_INPUT_ONLY',name)
            READS.add(str(path.relative_to(C.ROOT)))
    sys.dont_write_bytecode=True;sys.addaudithook(hook)


def summarize_change(before,after,mask):
    result=dict(n=int(mask.sum()),axes={})
    for a,name in enumerate(('T_cm','R_deg')):
        delta=(after-before)[mask,a]
        result['axes'][name]=dict(distribution=N.distribution(delta),
            improved=int((delta<0).sum()),equal=int((delta==0).sum()),worsened=int((delta>0).sum()),
            previous_population_median=float(np.median(before[mask,a])) if mask.any() else None,
            current_population_median=float(np.median(after[mask,a])) if mask.any() else None,
            difference_of_population_medians=float(np.median(after[mask,a])-np.median(before[mask,a])) if mask.any() else None)
    return result


def median_rows(ids,before,after,oldrows,rows,oldlabels,labels,axis):
    result={}
    for phase,values in [('previous_sign',before),('current_direction',after)]:
        # Deterministic identity ordering for equal values; no selector change.
        order=sorted(range(len(ids)),key=lambda i:(float(values[i,axis]),ids[i]))
        selected=order[(len(ids)//2)-1:(len(ids)//2)+1]
        result[phase]=[dict(id=ids[i],rank_1based=(len(ids)//2)+j,
            old_error=before[i].tolist(),new_error=after[i].tolist(),
            old_choice=oldrows[i]['candidate_name'],new_choice=rows[i]['candidate_name'],
            identity_changed=oldrows[i]['candidate_name']!=rows[i]['candidate_name'],
            old_class=str(oldlabels[i]),new_class=str(labels[i])) for j,i in enumerate(selected)]
        assert np.mean([values[i,axis] for i in selected])==np.median(values[:,axis])
    return result


def selected_prediction(rows,values,anchor):
    index=np.array([r['candidate_index'] for r in rows],int)
    ai=np.array([r['anchor_index'] for r in rows],int)
    selected=np.array([r['predicted_signed_axes'][j] for r,j in zip(rows,index)],np.float64)
    assert selected.shape==(1024,2) and np.isfinite(selected).all()
    assert (selected.max(1)<=0.).all()
    excess=(values-anchor)/N.SCALE;target=np.sign(excess)*np.log1p(abs(excess))
    nonanchor=index!=ai;out={}
    for label,mask in [('all_actual_choices',np.ones(len(rows),bool)),('actual_nonanchor_choices',nonanchor)]:
        axes={}
        for a,name in enumerate(('T','R')):
            y=target[mask,a];p=selected[mask,a]
            axes[name]=dict(n=len(y),MAE=float(np.mean(abs(p-y))) if len(y) else None,
                RMSE=float(np.sqrt(np.mean((p-y)**2))) if len(y) else None,
                exact_sign_accuracy=float(np.mean(np.sign(p)==np.sign(y))) if len(y) else None,
                actual_worse_predicted_nonworse=int(((y>0)&(p<=0)).sum()),
                predicted_negative=int((p<0).sum()),predicted_exact_zero=int((p==0).sum()))
        out[label]=dict(candidate_pairs=int(mask.sum()),axes=axes)
    out['scope']='Only frozen selected pairs; no claims about full-pool calibration or all-candidate accuracy. Predicted axes are learned scores, not cm/deg or calibrated probabilities.'
    return out


def analyze():
    started=time.monotonic();guard(input_paths().values())
    C.verify(C.read(C.DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC_PROTOCOL_SHA.json'))
    p=C.read(PROTOCOL)
    assert p['inputs']=={k:C.bind(v) for k,v in input_paths().items()}
    for b in [*p['inputs'].values(),*p['codes']]:C.verify(b)
    gate,lock,choices,errors,_=N.load_run(C.DOC,C.RAW)
    oldgate,oldlock,oldchoices,olderrors,_=N.load_run(C.SIGN_DOC,C.SIGN_RAW)
    for doc,currentgate,currentchoices in ((C.DOC,gate,choices),(C.SIGN_DOC,oldgate,oldchoices)):
        verified=C.read(doc/'SOURCE_VAL_VERIFICATION.json')
        assert verified['complete'] and verified['PASS'] and verified['source_gate']==C.bind(doc/'SOURCE_VAL_GATE.json')
        assert verified['checks_passed']==currentgate['checks_passed']
    assert choices['ids']==oldchoices['ids'] and len(choices['ids'])==1024
    for key in ('poses','metadata','feature_lock','source_predictions_lock','source_contract'):
        assert lock[key]==oldlock[key],('CANDIDATE_POOL_CHANGED',key)
    assert gate['source_reference_bindings']==oldgate['source_reference_bindings']
    fixed=('R0_GEO','DIVERSE251_s1_GEO','DIVERSE251_s2_GEO','DIVERSE251_s3_GEO')
    for name in fixed:
        assert errors[name].dtype==olderrors[name].dtype and errors[name].tobytes()==olderrors[name].tobytes()
    train=C.read(C.DOC/'TRAIN_CONVERGENCE.json');oldtrain=C.read(C.SIGN_DOC/'TRAIN_CONVERGENCE.json')
    assert train['complete'] and train['PASS'] and oldtrain['complete'] and oldtrain['PASS']
    assert train['protocol']==gate['protocol'] and oldtrain['protocol']==oldgate['protocol']
    ids=choices['ids'];anchor=errors['R0_GEO'];models={}
    for model in C.MODEL_NAMES:
        rows=[choices['records'][model][fid] for fid in ids]
        oldrows=[oldchoices['records'][model][fid] for fid in ids]
        assert all(a['anchor_name']==b['anchor_name'] and a['anchor_index']==b['anchor_index'] for a,b in zip(rows,oldrows))
        stats,labels,delta=N.actual_classes(rows,errors[model],anchor)
        oldstats,oldlabels,olddelta=N.actual_classes(oldrows,olderrors[model],anchor)
        changed=np.array([a['candidate_name']!=b['candidate_name'] for a,b in zip(rows,oldrows)])
        assert errors[model][~changed].tobytes()==olderrors[model][~changed].tobytes()
        oldsafe=oldlabels=='safe_improvement';safe=labels=='safe_improvement'
        transition={a+' -> '+b:int(np.sum((oldlabels==a)&(labels==b))) for a in CLASSES for b in CLASSES}
        bytransition={key:summarize_change(olderrors[model],errors[model],(oldlabels==key.split(' -> ')[0])&(labels==key.split(' -> ')[1])) for key,n in transition.items() if n}
        expert_changed=np.array([a['parent']!=b['parent'] for a,b in zip(rows,oldrows)])
        hyp_changed=np.array([a['hypothesis']!=b['hypothesis'] for a,b in zip(rows,oldrows)])
        source=gate['summaries'][model]['full_population'];oldsource=oldgate['summaries'][model]['full_population']
        summary_difference={axis:{stat:source[axis][stat]-oldsource[axis][stat] for stat in ('median','P90')} for axis in ('translation_cm','rotation_deg')}
        tr=train['models'][model];previous=oldtrain['models'][model]
        assert tr['previous_fixed_sign']['statistics']==previous['statistics']
        models[model]=dict(actual_source_selection=stats,previous_sign_actual_selection=oldstats,
            changed_identity_from_previous_sign=int(changed.sum()),
            class_transitions_from_sign=transition,transition_paired_change=bytransition,
            expert_hypothesis_transitions=dict(expert_changed=int(expert_changed.sum()),hypothesis_changed=int(hyp_changed.sum()),
                expert_and_hypothesis_changed=int((expert_changed&hyp_changed).sum()),
                previous_DIVERSE=int(sum(r['parent']!='R0' for r in oldrows)),current_DIVERSE=int(sum(r['parent']!='R0' for r in rows))),
            paired_change=dict(all1024=summarize_change(olderrors[model],errors[model],np.ones(1024,bool)),
                actually_changed_identity=summarize_change(olderrors[model],errors[model],changed)),
            population_summary_difference=summary_difference,
            median_boundary_rows={name:median_rows(ids,olderrors[model],errors[model],oldrows,rows,oldlabels,labels,a) for a,name in enumerate(('T_cm','R_deg'))},
            known_two_choice_opportunities=dict(frames_with_demonstrated_safe_choice=int((safe|oldsafe).sum()),
                current_miss_lower_bound=int((oldsafe&~safe).sum()),previous_miss_lower_bound=int((safe&~oldsafe).sum()),
                scope='Only actual current and previous chosen poses plus anchor. Selection-conditioned lower bound, not full candidate oracle/recall or a newly evaluated selection policy.'),
            selected_axis_regression=selected_prediction(rows,errors[model],anchor),
            previous_sign_selected_axis_regression=selected_prediction(oldrows,olderrors[model],anchor),
            source_metric_summary=gate['summaries'][model],previous_sign_metric_summary=oldgate['summaries'][model],
            fixed_TRAIN=dict(frames=train['frames'],available_rows=train['available_anchor_rows'],failed_rows=train['failed_rows'],
                actual=tr['risk_statistics'],previous_sign=previous['risk_statistics'],
                actual_statistics=tr['statistics'],previous_sign_statistics=previous['statistics'],
                objective=tr['independent_recompute'],previous_same_objective=tr['previous_fixed_sign']['same_new_objective'],
                same_objective_decrease=tr['same_new_objective_decrease']))
    result=dict(complete=True,PASS=True,created_at=C.now(),protocol=C.bind(PROTOCOL),code=C.bind(__file__),
        inputs=p['inputs'],models=models,frames_per_model=1024,model_count=4,actual_choices_described=4096,
        checks=dict(current=dict(PASS=gate['PASS'],passed=gate['checks_passed'],total=gate['checks_total'],failed=gate['failed_checks']),
            previous_sign=dict(PASS=oldgate['PASS'],passed=oldgate['checks_passed'],total=oldgate['checks_total'],failed=oldgate['failed_checks'])),
        all_fixed_baseline_arrays_byte_identical=True,same_candidate_pool=True,unchanged_selected_identity_error_byte_parity=True,
        interpretation=dict(observed='Signed direction input changes actual choices and lowers TRAIN objective, but its source result is judged by the unchanged45 checks. Population medians and paired per-frame deltas are distinct summaries.',
            not_established=['No proof that feature capacity is sufficient or insufficient.','No proof that residual direction has no useful information.','No causal attribution of each changed decision to18 coefficients because all271 weights were refit.','No real performance result: real routing was not authorized.']),
        recommendation=dict(status='PLANNING_ONLY_NOT_EXECUTED',
            intervention='Replace symmetric residual Huber with fixed asymmetric smooth upper-error supervision, retaining the sign-logistic term and all input/solver/runtime contracts.',
            specification='For e=prediction-signed_target, use Huber(e,1)+Huber(min(e,0),1) on the original valid candidate/two-axis/full2598 denominator, plus the existing nonzero-target sign logistic and ridge1e-4. This gives fixed2:1 underprediction versus overprediction Huber cost; no coefficient sweep or VAL-selected threshold. All other271 inputs, pools, targets, four fresh fits, zero initialization, max-axis runtime,1000/2000 solver caps, gap/gradient certificates and45-source/original-AND-matched-real gates stay fixed.',
            discrimination='Test whether sourceTRAIN harmful underestimation of candidate excess can be reduced without eliminating the safe choices needed for strict joint T/R improvement; record both safe losses and unsafe reductions before interpreting the unchanged gates.',
            evidence='Current selected candidates necessarily have both predicted axes<=0 because the anchor scores0, yet positive actual excess occurs among those selections. A symmetric continuous Huber term assigns equal magnitude penalty to over- and underprediction. The previous sign logistic distinguishes sign but does not impose this directional magnitude asymmetry.',
            counterevidence='Prior risk-margin CE and unweighted/pairwise/regression variants already show that conservative or better-fitting objectives need not achieve joint T/R. The feature-only direction addition does not establish objective asymmetry as the cause. Asymmetry may merely retreat to anchor and lose strict improvement; no expected success claim.',
            novelty_limit='An ordinary asymmetric convex loss is not claimed as a novel method. This precise whole-pose signed-T/R underprediction term differs from prior pointwise utility SmoothL1, signed-line gain MSE, classification margin and local pairwise supervision. A separate protocol and source-only target derivation must precede any execution.'),
        actual_new_fits=0,new_fits=0,new_candidate_selections=0,new_argmin_computations=0,
        no_new_selector_policy_evaluated=True,new_PnP_calls=0,image_forwards=0,raw_reference_reads=0,real_reference_reads=0,new_real_routes=0,
        TRAIN_label_or_weight_reads=0,new_real_routing=0,threshold_sweeps=0,method_success=False,goal_complete=False,
        elapsed_seconds=time.monotonic()-started,read_paths=sorted(READS))
    C.save(C.DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC_KO.md',report(result))
    C.save(C.DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC.json',result)
    print('FIXED_DIRECTION_SOURCE_FIXED_CHOICE_DIAGNOSTIC_PASS',C.bind(C.DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC.json'),flush=True)


def report(x):
    checks=x['checks'];lines=['# 방향18 추가 전후: 고정 source 선택 진단','',
        f"**현재 source {checks['current']['passed']}/{checks['current']['total']}, 직전 sign {checks['previous_sign']['passed']}/{checks['previous_sign']['total']}이며 두 실행 모두 전체 조건을 통과하지 못했다.** 이 문서는 동결된1024행×4모델의 실제 선택만 비교한다. 새 학습·argmin·임계값 탐색·PnP·원참조 재평가·실사 routing은0이다.",'',
        '원래 후보 pool·R0 GEO anchor·참조 binding과4개 고정 GEO baseline 배열이 동일함을 검사했다. 같은 identity를 고른 행의 T/R도 byte 단위로 일치한다. 분석 전에 입력 해시와 비교 범위를 별도 봉인했다.','',
        '## 실제 선택 분류와 전환','',
        '| 모델 | 선택 변경 | 이전 anchor/safe/unsafe | 현재 anchor/safe/unsafe | 이전 safe→현재 unsafe | 이전 unsafe→현재 safe |',
        '|---|---:|---:|---:|---:|---:|']
    for m,r in x['models'].items():
        old=r['previous_sign_actual_selection']['classes'];now=r['actual_source_selection']['classes'];t=r['class_transitions_from_sign']
        fmt=lambda d:'/'.join(str(d[k]) for k in ('anchor','safe_improvement','unsafe'))
        lines.append(f"| {m} | {r['changed_identity_from_previous_sign']} | {fmt(old)} | {fmt(now)} | {t['safe_improvement -> unsafe']} | {t['unsafe -> safe_improvement']} |")
    lines+=['','safe는 R0 anchor보다 두 오차 모두 작거나 같고 적어도 하나가 엄격히 작아진 실제 선택이다. unsafe는 한 축이라도 커진 선택이다. anchor·비anchor 완전동률·실패는 별도 분류하며1024행을 모두 유지했다.','',
        '## 중앙값과 행별 변화는 다르다','',
        '| 모델 | 이전 T 중앙값(cm) | 현재 T 중앙값 | 이전 R 중앙값(°) | 현재 R 중앙값 | T P90 변화(cm) | R P90 변화(°) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,r in x['models'].items():
        a=r['previous_sign_metric_summary']['full_population'];b=r['source_metric_summary']['full_population'];d=r['population_summary_difference']
        lines.append(f"| {m} | {a['translation_cm']['median']:.9f} | {b['translation_cm']['median']:.9f} | {a['rotation_deg']['median']:.9f} | {b['rotation_deg']['median']:.9f} | {d['translation_cm']['P90']:+.9f} | {d['rotation_deg']['P90']:+.9f} |")
    lines+=['','JSON의 `paired_change`는 같은 행의 현재−이전 오차 분포이고 위 표는 모집단 분위수의 차이다. 전자는0이어도 후자는 달라질 수 있다. `median_boundary_rows`에 각 실행의 중앙 두 순위(512·513번째) ID·전후 identity/오차를 보존했다. 그 두 행만으로 전체 손실 변화의 원인이라고 단정하지 않는다.','',
        '## 모델별 관측','']
    for m,r in x['models'].items():
        c=r['paired_change']['actually_changed_identity'];a=c['axes'];e=r['expert_hypothesis_transitions'];q=r['known_two_choice_opportunities']
        lines.append(f"- **{m}**: 변경{c['n']}행 중 T 개선/악화 {a['T_cm']['improved']}/{a['T_cm']['worsened']}, R 개선/악화 {a['R_deg']['improved']}/{a['R_deg']['worsened']}. expert 변경{e['expert_changed']}·W/D 변경{e['hypothesis_changed']}. 직전 실제 safe 선택을 현재 포착하지 못한 하한{q['current_miss_lower_bound']}행, 반대 방향 하한{q['previous_miss_lower_bound']}행.")
    lines+=['','위 기회 하한은 두 실행에서 실제로 골랐던 pose와 anchor만 모은 결과조건부 부분집합이다. 전체 후보의 oracle·recall·보정 정확도로 확대할 수 없다. 선택된 두 예측축은 anchor의0점수 때문에 모두0이하이지만 실제 양의 오차 초과가 남는다. 예측 부호/MAE 통계는 선택된 후보에 대해서만 계산했으며 출력값은 cm/degree나 보정된 확률이 아니다.','',
        '## TRAIN 관측과 해석 한계','']
    for m,r in x['models'].items():
        tr=r['fixed_TRAIN'];old=tr['previous_sign']['classes'];now=tr['actual']['classes']
        lines.append(f"- {m}: 같은 목적식 J 감소{tr['same_objective_decrease']:.9f}; TRAIN safe {old['safe_improvement']['count']}→{now['safe_improvement']['count']}, unsafe {old['unsafe']['count']}→{now['unsafe']['count']}. 전체2,598행·유효2,597행·실패1행 기준을 유지했다.")
    lines+=['','입력의 비중복성 감사와 TRAIN J 감소가 source45개 조건의 공동 통과를 보장하지 않았다. 단, 추가 방향 정보가 무용하거나271표현이 부족하다는 증명도 아니다. 모든 계수를 함께 다시 학습했으므로 바뀐 선택을 추가18계수만의 인과효과로 분리할 수 없다. 실사는 이번 실행에서 미평가다.','',
        '## 다음 단일 개입 제안 — 미실행','',
        '후보의 나쁜 오차 초과를 낮게 예측하는 방향에만 **고정된 추가 Huber 항**을 주는 목적식 비교를 제안한다. `e=p−y`일 때 기존 `Huber(e,1)`에 `Huber(min(e,0),1)`을 한 번 더한다. 따라서 과소예측 대 과대예측 비용은 고정2:1이며 기존 sign logistic·λ·271입력·target·전체 분모·Newton·4개 영초기화 fit·runtime max-axis·source45 및 원래+matched 실사5기준은 유지한다. 계수·threshold sweep이나 VAL을 이용한 값 선택을 하지 않는 별도 사전 계약이 필요하다. 이 손실은 연속미분 가능한 볼록함수이며 ridge가 강볼록성을 유지한다.','',
        '근거는 실제로 선택된 unsafe 후보에서 두 예측축은0이하이나 정답 초과는 양수라는 관측이다. 대칭 Huber는 동일 크기의 과소·과대예측을 동일하게 벌하고 sign logistic은 부호를 보지만 방향별 크기 비용을 위처럼 비대칭으로 만들지는 않는다. **이는 원인 확정이나 성공 예측이 아니다.** 이전 risk-margin CE·pairwise·utility 회귀의 실패는 반대근거이며, 보수화가 anchor 복귀만 늘려 엄격한 개선을 없앨 수 있다. 다음 결과는 unsafe 감소와 safe 소실을 함께 기록하고 원래 gate로 판정해야 한다.','',
        '기존 point utility SmoothL1·signed-line gain MSE·분류 margin과 현재 전체 pose signed-T/R 과소예측 항의 차이는 구분하되, 비대칭 손실 자체를 새로운 학술적 기여로 주장하지 않는다.','',
        '[분석 입력 봉인](SOURCE_FIXED_CHOICE_DIAGNOSTIC_PROTOCOL.json) · [전체 진단 JSON](SOURCE_FIXED_CHOICE_DIAGNOSTIC.json) · [학습 독립 검산](TRAIN_CONVERGENCE_KO.md) · [이전 목적식 음성 선행 감사](../pallet_pose_selector_objective_audit_20261001_v1/PRIOR_OBJECTIVE_AUDIT_KO.md)','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('seal','analyze'))
    (seal if parser.parse_args().stage=='seal' else analyze)()
