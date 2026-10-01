"""Frozen-route transitions and actual score terms; never select a new policy."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import common as C
from . import convex_train as T
from scripts.research.pallet_pose_anchor_risk_20261001_v1.selector_transfer_diagnostic import (
    POPS, category, transition, summarize, numbers)


def load_result(doc):
    result = C.read(doc/'REAL_RESULTS.json')
    assert result['complete']
    for binding in [result['protocol'],result['routing_lock'],*result['artifacts']]:
        C.verify(binding)
    lock = C.read(C.ROOT/result['routing_lock']['path'])
    assert lock['complete'] and lock['frames']==173 and lock['models']==list(C.MODEL_NAMES)
    assert lock['protocol']==result['protocol'] and not lock['real_reference_values_read']
    assert lock['runtime_uses_margin'] is False
    C.verify(lock['choices'])
    choices=C.read(C.ROOT/lock['choices']['path'])
    metric_binding=next(b for b in result['artifacts'] if b['path'].endswith('/POSE_METRICS.json'))
    metrics=C.read(C.ROOT/metric_binding['path'])
    return result,lock,choices,metrics,[C.bind(doc/'REAL_RESULTS.json'),result['protocol'],result['routing_lock'],lock['choices'],metric_binding]


def kernel_summary(x, valid, anchor, names, weight, subset):
    """Describe activations/contrasts on fixed candidate inputs, with no argmin."""
    subset=np.asarray(subset,bool)
    assert subset.shape==(len(x),)
    rows=np.flatnonzero(subset & valid.any(1))
    active=valid & subset[:,None]
    k=x[:,:,189:]
    candidates=k[active]
    candidate_stats={name:numbers(values) for name,values in (
        ('activation_mean_per_candidate',candidates.mean(1)),
        ('activation_max_per_candidate',candidates.max(1)),
        ('activation_L2_per_candidate',np.linalg.norm(candidates,axis=1)))}
    stats=[]
    for i in rows:
        a=int(anchor[i]); assert a in (0,1) and valid[i,a]
        for j in np.flatnonzero(valid[i]):
            if j==a: continue
            dk=k[i,j]-k[i,a]
            prefix=float(np.dot(x[i,j,:189]-x[i,a,:189],weight[:189]))
            added=float(np.dot(dk,weight[189:]))
            stats.append(dict(same_WD=names[j].split(':')[1]==names[a].split(':')[1],
                kernel_delta_L2=float(np.linalg.norm(dk)),kernel_delta_Linf=float(np.max(np.abs(dk))),
                prefix_score_delta=prefix, RBF_score_delta=added,
                absolute_prefix_score_delta=abs(prefix), absolute_RBF_score_delta=abs(added),
                zero_kernel_delta=bool((dk==0).all())))
    contrast={}
    keys=('kernel_delta_L2','kernel_delta_Linf','prefix_score_delta','RBF_score_delta',
          'absolute_prefix_score_delta','absolute_RBF_score_delta')
    for label,items in (('all_nonanchor_candidates',stats),('same_WD',[s for s in stats if s['same_WD']]),
                        ('different_WD',[s for s in stats if not s['same_WD']])):
        contrast[label]={key:numbers([s[key] for s in items]) for key in keys}
        contrast[label]['exact_zero_kernel_delta_count']=sum(s['zero_kernel_delta'] for s in items)
    return dict(frames=int(subset.sum()),valid_frames=len(rows),failed_frames=int(subset.sum()-len(rows)),
        valid_candidates=int(active.sum()),activation=candidate_stats,
        exact_all_zero_kernel_candidates=int((candidates==0).all(1).sum()),
        exact_zero_kernel_elements=int((candidates==0).sum()),total_kernel_elements=int(candidates.size),
        candidate_anchor_contrasts=contrast,no_new_candidate_selection=True)


def score_pair(x,weight,choice):
    j,a=int(choice['candidate_index']),int(choice['anchor_index'])
    assert j>=0 and a in (0,1) and choice['valid'][j] and choice['valid'][a]
    prefix=float(np.dot(x[j,:189]-x[a,:189],weight[:189]))
    rbf=float(np.dot(x[j,189:]-x[a,189:],weight[189:]))
    saved=float(choice['scores'][j]-choice['scores'][a])
    assert abs(prefix+rbf-saved)<=1e-10
    assert saved<=1e-10
    return dict(selected_index=j,anchor_index=a,prefix189_delta=prefix,RBF64_delta=rbf,
        sum_delta=prefix+rbf,frozen_total_score_delta=saved,
        absolute_additivity_error=abs(prefix+rbf-saved),
        RBF_opposes_selected_candidate=rbf>0,RBF_favors_selected_candidate=rbf<0,
        prefix_and_RBF_opposite_sign=prefix*rbf<0)


def summarize_extended(rows):
    out=summarize(rows)
    changed=[r for r in rows if r['previous_risk']['candidate_changed']]
    moved=[r for r in rows if r['class']!='anchor']
    out['previous_risk']=dict(candidate_changed=len(changed),candidate_unchanged=len(rows)-len(changed),
        class_transition_matrix=dict(Counter(r['previous_risk']['class']+' -> '+r['class'] for r in rows)),
        changed_class_transition_matrix=dict(Counter(r['previous_risk']['class']+' -> '+r['class'] for r in changed)),
        same_WD_count=sum(r['previous_risk']['hypothesis']==r['hypothesis'] for r in rows),
        delta_T_cm=numbers([r['T_cm']-r['previous_risk']['T_cm'] for r in rows]),
        delta_R_deg=numbers([r['R_deg']-r['previous_risk']['R_deg'] for r in rows]))
    out['actual_chosen_minus_anchor']=dict(all_rows=len(rows),nonanchor_rows=len(moved),
        prefix189_delta=numbers([r['score_decomposition']['prefix189_delta'] for r in moved]),
        RBF64_delta=numbers([r['score_decomposition']['RBF64_delta'] for r in moved]),
        frozen_total_delta=numbers([r['score_decomposition']['frozen_total_score_delta'] for r in moved]),
        RBF_favors_nonanchor=sum(r['score_decomposition']['RBF_favors_selected_candidate'] for r in moved),
        RBF_opposes_nonanchor=sum(r['score_decomposition']['RBF_opposes_selected_candidate'] for r in moved),
        opposite_sign_count=sum(r['score_decomposition']['prefix_and_RBF_opposite_sign'] for r in moved),
        maximum_additivity_error=max(r['score_decomposition']['absolute_additivity_error'] for r in rows))
    return out


def run():
    result,lock,choices,metrics,bindings=load_result(C.DOC)
    previous_result,previous_lock,previous_choices,previous_metrics,previous_bindings=load_result(C.RISK_DOC)
    assert lock['inputs']==previous_lock['inputs'] and choices['ids']==previous_choices['ids']
    assert len(choices['ids'])==len(set(choices['ids']))==173
    for model in set(metrics)-set(C.MODEL_NAMES):
        assert metrics[model]==previous_metrics[model]
    previous_path=C.RISK_DOC/'SELECTOR_TRANSFER_DIAGNOSTIC.json'
    previous=C.read(previous_path)
    assert previous['complete'] and previous['PASS'] and previous['actual_choice_rows']==692
    for b in previous['bindings']: C.verify(b)
    old_rows={(r['model'],r['id']):r for r in previous['rows']}
    assert len(old_rows)==692
    for key in ('features','metadata','groups'):
        C.verify(lock['inputs'][key])
    metadata=C.read(C.ROOT/lock['inputs']['metadata']['path'])
    groups=C.read(C.ROOT/lock['inputs']['groups']['path'])
    assert [r['id'] for r in metadata]==choices['ids']
    basis_binding=C.bind(C.DOC/'RBF_BASIS.json')
    assert lock['basis_SHA_bind']==basis_binding
    basis=C.read(C.DOC/'RBF_BASIS.json')
    assert basis['complete'] and basis['PASS']
    train_protocol=C.read(C.DOC/'TRAIN_PROTOCOL.json')
    C.verify(C.read(C.DOC/'TRAIN_PROTOCOL_SHA.json'))
    for binding in train_protocol['codes']: C.verify(binding)
    assert train_protocol['inputs']['rbf_basis']==basis_binding
    for key in ('features','source_contract','prefit_review'):
        C.verify(train_protocol['inputs'][key])
    contract=C.read(C.ROOT/train_protocol['inputs']['source_contract']['path'])
    eligible=set(contract['fit_eligibility']['eligible_ids']['TRAIN'])
    prefit=C.read(C.ROOT/train_protocol['inputs']['prefit_review']['path'])
    assert prefit['complete'] and prefit['PASS']
    train=C.read(C.DOC/'TRAIN_CONVERGENCE.json')
    assert train['complete'] and train['PASS'] and train['source_TRAIN_only']
    with np.load(C.ROOT/train_protocol['inputs']['features']['path'],allow_pickle=False) as z:
        idx=np.asarray([i for i,fid in enumerate(z['ids']) if fid in eligible],np.int64)
        assert len(idx)==2598 and (z['split'][idx]=='TRAIN').all()
        train_anchor=z['R0_GEO_index'][idx]
        train_features={m:z[m+'_geo'][idx] for m in ('R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3')}
        train_valid={m:z[m+'_valid'][idx] for m in train_features}
        assert T.OLD.array_sha(z['ids'][idx])==basis['source_ids_sha']
        assert T.OLD.array_sha(idx)==basis['source_index_sha']
    with np.load(C.ROOT/lock['inputs']['features']['path'],allow_pickle=False) as z:
        assert z['ids'].tolist()==choices['ids']
        real_features={m:z[m+'_geo'] for m in train_features}
        real_valid={m:np.repeat(z[m+'_valid'][:,None],2,axis=1) for m in train_features}
    real_anchor=np.asarray([choices['records']['R0_ONLY'][fid]['anchor_index'] for fid in choices['ids']],np.int64)
    kernel_stats={}; rows=[]
    for model in C.MODEL_NAMES:
        fit_binding=lock['fits'][model]['receipt']
        C.verify(fit_binding); fit=C.read(C.ROOT/fit_binding['path']); C.verify(fit['checkpoint'])
        assert fit['checkpoint']==lock['fits'][model]['checkpoint']
        ck=C.read(C.ROOT/fit['checkpoint']['path'])
        assert ck['basis_SHA_bind']==ck['rbf_basis_binding']==basis_binding and ck['rbf_basis']==basis['basis']
        assert not ck['runtime_uses_margin'] and ck['schema']==T.CHECKPOINT_SCHEMA
        weight=np.asarray(ck['weight'],np.float64)
        parents=['R0']+([] if model=='R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
        traw=np.concatenate([train_features[m] for m in parents],axis=1)
        tv=np.concatenate([train_valid[m] for m in parents],axis=1)
        tx=T.rbf_inputs(traw,tv,train_anchor,ck['mean'],ck['std'],ck['rbf_basis'])
        assert T.OLD.array_sha(tx)==prefit['models'][model]['context_sha']
        raw=np.concatenate([real_features[m] for m in parents],axis=1)
        valid=np.concatenate([real_valid[m] for m in parents],axis=1)
        x=T.rbf_inputs(raw,valid,real_anchor,ck['mean'],ck['std'],ck['rbf_basis'])
        kernel_stats[model]={'TRAIN2598':kernel_summary(tx,tv,train_anchor,ck['names'],weight,np.ones(2598,bool))}
        for pop in (*POPS,'ALL173'):
            subset=np.ones(173,bool) if pop=='ALL173' else np.isin(choices['ids'],groups[pop])
            kernel_stats[model][pop]=kernel_summary(x,valid,real_anchor,ck['names'],weight,subset)
        score=np.einsum('nkd,d->nk',x,weight)
        for i,meta in enumerate(metadata):
            fid=meta['id']; choice=choices['records'][model][fid]; old_choice=previous_choices['records'][model][fid]
            old=old_rows[model,fid]; metric=metrics[model][fid]; base=metrics['R0'][fid]
            assert metric['available'] and base['available'] and choice['pose_available']
            assert choice['valid']==valid[i].tolist() and choice['anchor_index']==int(real_anchor[i])
            np.testing.assert_allclose(score[i,valid[i]],np.asarray(choice['scores'])[valid[i]],rtol=0,atol=1e-10)
            anchor=np.asarray([base['translation_cm'],base['rotation_deg']],np.float64)
            error=np.asarray([metric['translation_cm'],metric['rotation_deg']],np.float64)
            np.testing.assert_array_equal(anchor,[old['anchor_T_cm'],old['anchor_R_deg']])
            assert old['candidate_name']==old_choice['candidate_name'] and old['anchor_name']==choice['anchor_name']
            np.testing.assert_array_equal([old['T_cm'],old['R_deg']],
                [previous_metrics[model][fid]['translation_cm'],previous_metrics[model][fid]['rotation_deg']])
            cls=category(choice['candidate_name'],choice['anchor_name'],error,anchor)
            valid_scores=[s for s,v in zip(choice['scores'],choice['valid']) if v]
            updated=dict(old, candidate_name=choice['candidate_name'], parent=choice['parent'],hypothesis=choice['hypothesis'],
                **{'class':cls}, safe_improvement=cls=='safe_improvement',transition=transition(choice['candidate_name'],choice['anchor_name']),
                T_cm=float(error[0]),R_deg=float(error[1]),delta_T_cm=float(error[0]-anchor[0]),delta_R_deg=float(error[1]-anchor[1]),
                normalized_excess=float(max(0.,*((error-anchor)/np.asarray(previous['scale_sT_cm_sR_deg'])))),
                best_score_tie=sum(s==min(valid_scores) for s in valid_scores)>1,
                score_decomposition=score_pair(x[i],weight,choice),
                previous_risk=dict(candidate_name=old['candidate_name'],hypothesis=old['hypothesis'],**{'class':old['class']},
                    T_cm=old['T_cm'],R_deg=old['R_deg'],candidate_changed=choice['candidate_name']!=old['candidate_name']))
            if old['oracle'] is not None:
                o=dict(old['oracle']); o['exact_identity_hit']=o['candidate_name']==choice['candidate_name']
                o['expert_mismatch']=o['candidate_name'].split(':')[0]!=choice['parent']
                o['WD_mismatch']=o['candidate_name'].split(':')[1]!=choice['hypothesis']
                o['opportunity_missed']=o['strict_safe_opportunity'] and cls!='safe_improvement'
                if o['exact_identity_hit']: np.testing.assert_array_equal(error,[o['T_cm'],o['R_deg']])
                updated['oracle']=o
            rows.append(updated)
        bindings += [fit_binding,fit['checkpoint']]
    assert len(rows)==692 and len({(r['model'],r['id']) for r in rows})==692
    populations={}
    for pop in (*POPS,'ALL173'):
        subset=[r for r in rows if pop=='ALL173' or r['population']==pop]
        populations[pop]=dict(models={m:summarize_extended([r for r in subset if r['model']==m]) for m in C.MODEL_NAMES},
            by_recording={rec:{m:summarize_extended([r for r in subset if r['recording']==rec and r['model']==m])
                for m in C.MODEL_NAMES} for rec in sorted({r['recording'] for r in subset})})
    sources=[C.ROOT/'scripts/research/pallet_selector_recovery_v1/router_features.py',
        C.ROOT/'scripts/research/pallet_selector_recovery_v1/train_router.py',
        C.ROOT/'_docs/experiments/pallet_selector_recovery_v1/stage4_clean_preservation/STAGE4_REPORT_KO.md']
    output=dict(complete=True,PASS=True,diagnostic_only=True,method_success=False,goal_complete=False,created_at=C.now(),
        populations=populations,rows=rows,kernel_statistics=kernel_stats,
        TRAIN_comparison={m:train['models'][m] for m in C.MODEL_NAMES},
        real_gate_PASS=result['stability']['PASS'],real_gate_categories={k:v['PASS'] for k,v in result['stability']['gates'].items()},
        actual_choice_rows=692,paired_UNION_oracle_rows=519,basis_SHA_bind=basis_binding,
        decomposition='Actual chosen minus operational R0 anchor only: dot(delta_context189,w189)+dot(delta_RBF64,w64). All other contrasts summarize fixed input candidates, without ranking or policy evaluation.',
        interpretation_limits='Signed feature contributions are arithmetic decomposition of the final model, not causal attribution versus the old model: the first189 weights were refitted too. Different input support/representation and supervision are competing explanations.',
        kernel_population='TRAIN eligible2598 including1 failed row; kernel distributions use only valid candidates. Real all173 available. SOURCE_FEATURES container includes VAL inputs but only TRAIN indices used.',
        oracle_scope=previous['oracle_scope'],miss_definition=previous['miss_definition'],
        raw_GT_reads=0,new_metric_calls=0,new_PnP=0,image_forwards=0,new_fits=0,
        new_selector_routes=0,counterfactual_argmin_policies=0,threshold_sweeps=0,source_VAL_metric_arrays_read=False,
        source_TRAIN_target_or_error_arrays_read=False,cached_reference_errors_used=True,
        bindings=bindings+previous_bindings+train_protocol['codes']+[C.bind(previous_path),basis_binding,C.bind(C.DOC/'TRAIN_CONVERGENCE.json'),
            C.bind(C.DOC/'TRAIN_PROTOCOL.json'),train_protocol['inputs']['features'],train_protocol['inputs']['source_contract'],
            train_protocol['inputs']['prefit_review'],lock['inputs']['features'],lock['inputs']['metadata'],lock['inputs']['groups'],
            C.bind(Path(__file__)),C.bind(Path(T.__file__)),*[C.bind(p) for p in sources]])
    text=report(output)
    C.save(C.DOC/'SELECTOR_TRANSFER_DIAGNOSTIC_KO.md',text)
    C.save(C.DOC/'SELECTOR_TRANSFER_DIAGNOSTIC.json',output)
    print('RBF_TRANSFER_DIAGNOSTIC_PASS_NO_NEW_POLICY',flush=True)


def report(x):
    lines=['# RBF 추가 후 고정 실사 선택·점수·입력 반응 진단','',
        '고정된 실제692개 선택만 직전 risk189 및 기존 anchored oracle와 비교했다. 새로운 argmin·부분특징 selector·문턱·width·센터 수 정책은 시험하지 않았다. raw GT·이미지·PnP·fit은0회이며 저장된 오차만 재사용했다.','',
        '| 모집단 | 모델 | 이전과 다른 후보 | anchor | safe 개선 | unsafe | oracle 기회 | 놓친 기회 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for pop in POPS:
        for model,v in x['populations'][pop]['models'].items():
            c=v['classes'];o=v['oracle_comparison']
            lines.append(f"| {pop} | {model} | {v['previous_risk']['candidate_changed']} | {c['anchor']} | {c['safe_improvement']} | {sum(n for k,n in c.items() if k.startswith('unsafe'))} | {o['opportunities'] if o else 'NA'} | {o['opportunities_missed'] if o else 'NA'} |")
    lines += ['', 'R0_ONLY의2후보 pool에는4후보 oracle 회수율을 적용하지 않는다. 다른 oracle identity라도 실제 pointwise-safe 개선이면 기회를 회수했다고 센다. 모든 recording·seed·692행 및 정확 동률은 JSON에 보존했다.','',
        '## 실제 non-anchor 선택의 점수 기여','',
        '아래는 실제로 선택한 후보에서 R0 anchor를 뺀 점수다. 음수는 선택 후보에 유리하다. anchor를 그대로 고른 행은 차이가0이라 별도로 분모를 표시한다. 기존189만으로 다시 argmin하거나 RBF를 제거한 정책을 평가하지 않았다.','',
        '| 모델 | 실사 non-anchor/173 | prefix189 차이 중앙값 | RBF64 차이 중앙값 | RBF가 선택에 유리/불리 | 가산 검산 최대오차 |',
        '|---|---:|---:|---:|---:|---:|']
    def fmt(v): return 'NA' if v is None else f'{v:.9g}'
    for m,v in x['populations']['ALL173']['models'].items():
        d=v['actual_chosen_minus_anchor']
        lines.append(f"| {m} | {d['nonanchor_rows']}/173 | {fmt(d['prefix189_delta']['median'])} | {fmt(d['RBF64_delta']['median'])} | {d['RBF_favors_nonanchor']}/{d['RBF_opposes_nonanchor']} | {d['maximum_additivity_error']:.3g} |")
    lines += ['', '## TRAIN과 실사의 kernel 반응','',
        '각 후보의64개 RBF 중 최대 활성값과, 같은 W/D의 non-anchor 후보−anchor kernel L2 차이 및 가중 점수차이의 절댓값을 요약했다. 임계값 기반 OOD 분류나 선택 filter는 만들지 않았다. 활성은 거리의 함수이며 정답 확률이 아니다.','',
        '| 모델 | 모집단 | 유효 후보 | 최대활성 중앙값 | 동일 W/D Δkernel L2 중앙값 | 동일 W/D abs(RBF 점수차) 중앙값 | 동일 W/D abs(prefix 점수차) 중앙값 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for m,pops in x['kernel_statistics'].items():
        for pop in ('TRAIN2598','NATURAL99','CLEAN29','WOOD45'):
            v=pops[pop];d=v['candidate_anchor_contrasts']['same_WD']
            lines.append(f"| {m} | {pop} | {v['valid_candidates']} | {fmt(v['activation']['activation_max_per_candidate']['median'])} | {fmt(d['kernel_delta_L2']['median'])} | {fmt(d['absolute_RBF_score_delta']['median'])} | {fmt(d['absolute_prefix_score_delta']['median'])} |")
    lines += ['', '**관측:** natural에서 이전 risk 대비 후보가 바뀐 행은 seed별1·2·4개뿐이었다. 그러나 kernel 자체가 꺼졌다는 설명은 이 고정 결과와 맞지 않는다. 최대활성 중앙값은 TRAIN 약0.913, natural 약0.825–0.829였고, 동일 W/D 후보 사이 kernel L2 대조는 오히려 natural에서 더 컸다. 실제 non-anchor 선택의 RBF 점수항은 UNION1의9/9, UNION2의11/13, UNION3의11/11에서 양수여서 그 선택을 anchor보다 비싸게 만들었다. 이 항을 제외하거나 약하게 한 selector를 새로 평가했다는 뜻은 아니다. 일부 선택이 바뀌어도 공동 중앙값·불확실성·recording 조건을 통과하지 못했고, 단순 입력활성 소실보다 최종 점수의 보수성과 놓친 개선 기회가 직접 관측된다.','',
        'TRAIN2,598행 중 실패1행은 유지하며 kernel 수치는 유효 후보에서 계산했다. 실사173행은 모두 유효하다. 앞189 가중치도 새로 학습됐으므로 RBF 항의 부호나 크기를 이전 모델 대비 변화의 인과 효과라고 해석할 수 없다. 합성 지원범위, 학습 감독, 기하 표현 및 RGB 단서 부족은 분리되지 않은 경쟁 설명이다. 활성 분포 차이만으로 어떤 설명이 원인인지 확정하지 않는다.','',
        '## 기존 RGB router의 반대 증거','',
        '`pallet_selector_recovery_v1/router_features.py:7–43`은 두 expert의 confidence·잔차·box/점/pose 차이와 frozen S1 GAP448을 MLP64/32에 넣었다. `train_router.py`와 Stage4 보고서에서 source clean/occluded 쌍 TRAIN8192·VAL2048·TEST2048, exact C2 ADDnorm의 더 나은 expert target, VAL 정확도로 epoch5를 선택했음을 확인했다. TEST 정확도0.5415, 전체 mean ADDnorm routed0.097688은 S0의0.096217보다 나빴고, 실사 판정은 CLEAN_RECOVERY_BUT_HARD_LOSS였다. 현재 R0→PoseFix 네 후보·physical T/R 감독과 같은 실험은 아니지만, 단순히 global RGB feature와 MLP를 추가하면 해결된다는 주장을 반박한다.','',
        '## 다음 학습 문제 하나의 제안 — 두 축의 anchor 대비 변화 회귀','',
        '같은 RBF의 폭·센터 수·margin 배율을 다시 고르지 않는다. 다음 한 가지 후보는 현재 고정253 입력과 후보를 유지한 채,4-way best-label CE 대신 각 후보의 signed physical T/R anchor-excess를 두 출력으로 직접 학습하는 것이다. 현재의 one-hot target와 max-risk margin은 어떤 축이 얼마나 좋아지거나 나빠지는지 전체 signed 두 축을 연속 target으로 전달하지 않는다. 이 정보는 기존 TRAIN 참조에 이미 있으며 runtime에서 참조를 추가하지 않는다. 다만 이번 진단이 감독 문제의 원인을 증명한 것은 아니므로 성공을 예측하지 않는다.','',
        '구체적인 다음 계약 후보는 e_T=(T_c−T_anchor)/sT, e_R=(R_c−R_anchor)/sR의 각 축에 sign(e)*log1p(abs(e))를 적용한2-output target이다. 원래 TRAIN 고정 sT/sR과2598행을 쓴다. 특징은 기존253의 후보−anchor 차이로 만들고 공유253×2 가중치, bias0을 사용해 anchor의 예측을 정확히(0,0)으로 둔다. 일반적인 shared linear scalar에서 anchor 차분이 argmin에 상쇄되는 사실은 그대로다. 여기서는 두 개의 signed 회귀 target과 max-of-two 출력 판정으로 학습 문제를 바꾸며, 차분 자체를 새 표현력이라고 주장하지 않는다.','',
        'TRAIN valid 후보의 두 축 Huber(delta=1) 평균을 frame마다 평균하고 전체2598행으로 나눈 뒤 λ/2||W||²(λ=1e−4)를 더하는 단일 고정 목적식을 제안한다. invalid/all-invalid는 차분을 계산하지 않고0loss로 원래 분모에 남긴다. λ-strong convexity의 gradient-gap 인증과 기존 단일 solver 예산을 유지하고 네 모델 각1fit만 허용한다. 이 λ와 손실값은 이전 CE 목적과 수치상 동등하다고 주장하지 않는다. 새 Huber폭·변환·가중치 후보를 비교하는 sweep은 하지 않는다.','',
        '추론은 예측된 두 변화의 max를 score로 하여 anchor를 포함한 기존 whole-pose 중 선택하고 정확 동률은R0를 우선한다. reference-safe mask나 실제risk를 넣지 않으며0은 학습한 변화의 부호 기준이지 실사에서 맞춘 threshold가 아니다. 이미 고정된 source45와 원래/개입 실사5개 AND·세seed·실패행을 유지하고 하나라도 실패하면 중단한다. TRAIN에서 두 축 예측이 맞아도 실사 전이·보수성·중앙값 개선은 보장되지 않는다.','',
        '이 제안은 새 RGB 정보를 추가하는 방법이 아니며 이미 실패한 일반 RGB MLP를 반복하지 않는다. 기존 utility/TrackD의 signed2D gain 회귀 및 DHT의2D cost 회귀 음성 결과도 유지한다. 차이는 현재 네 physical whole-pose에 대한 R0-anchor-relative T/R 두 축을 직접 감독한다는 범위이며, 회귀 일반의 새 발명이나 해결책으로 주장하지 않는다. 별도 prefit와 사전 protocol 없이 실행하지 않으며 이번 문서에는 새 fit·새 정책 성능이 없다.','',
        '[전체 고정 진단](SELECTOR_TRANSFER_DIAGNOSTIC.json) · [TRAIN 검산](TRAIN_CONVERGENCE_KO.md) · [실사 결과](REAL_RESULTS.json) · [직전 risk 진단](../pallet_pose_anchor_risk_20261001_v1/SELECTOR_TRANSFER_DIAGNOSTIC_KO.md)','']
    return '\n'.join(lines)


def selfcheck():
    x=np.zeros((4,253));x[2,:189]=1.;x[2,189:]=2.
    w=np.r_[np.ones(189),np.full(64,-2.)]
    choice=dict(candidate_index=2,anchor_index=0,valid=[True]*4,scores=[0.,0.,-67.,0.])
    d=score_pair(x,w,choice)
    assert d['prefix189_delta']==189. and d['RBF64_delta']==-256. and d['sum_delta']==-67.
    assert d['prefix_and_RBF_opposite_sign'] and d['RBF_favors_selected_candidate']
    names=['R0:long-face-front','R0:short-face-front','D:long-face-front','D:short-face-front']
    s=kernel_summary(x[None],np.ones((1,4),bool),np.array([0]),names,w,np.array([True]))
    assert s['candidate_anchor_contrasts']['same_WD']['kernel_delta_L2']['n']==1
    assert category('R0:a','R0:a',[1.,2.],[1.,2.])=='anchor'
    print('RBF_TRANSFER_SELFCHECK_PASS_INVENTED_ONLY')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('selfcheck','run'))
    with threadpool_limits(limits=1):
        (selfcheck if parser.parse_args().stage=='selfcheck' else run)()
