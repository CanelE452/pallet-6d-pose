"""Independent TRAIN-only margin mathematics and unchanged-input audit.

No fits, VAL quality, model policy probes, raw source references or real GT.
All numerical objective tests use invented fixtures. Actual TRAIN labels are
used only to reconstruct the fixed target, risk and margin and bind their SHA.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT/'_docs/experiments'/HERE.name
CONTEXT_DOC = ROOT/'_docs/experiments/pallet_pose_anchor_context_20261001_v1'
ANCHOR_DOC = ROOT/'_docs/experiments/pallet_pose_pareto_anchor_20261001_v1'
MODELS = ('R0_ONLY','UNION_s1','UNION_s2','UNION_s3')
LOSS_RULE = 'TRAIN_LOG1P_ANCHOR_EXCESS_MARGIN_CE'
READS = set()


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def verify(binding):
    assert bind(ROOT/binding['path']) == binding,binding['path']


def array_sha(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(str(value.dtype).encode());h.update(json.dumps(list(value.shape)).encode());h.update(value.tobytes())
    return h.hexdigest()


def install_write_guard():
    outputs={DOC/'PREFIT_REVIEW.json',DOC/'PREFIT_REVIEW_KO.md'}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        p=Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to(ROOT):
            return
        mode=args[1]
        if isinstance(mode,str) and any(c in mode for c in 'wax+'):
            assert p in outputs,('PREFIT_WRITE_SCOPE',str(p))
        else:
            READS.add(str(p.relative_to(ROOT)))
    sys.addaudithook(hook)


def scalar_targets(errors,valid,anchor,index,scale,names):
    target=np.full(len(valid),-1,np.int64);safe=np.zeros_like(valid)
    for i in range(len(valid)):
        if not valid[i].any():
            assert index[i]==-1 and np.isposinf(anchor[i]).all()
            continue
        assert int(index[i]) in (0,1) and valid[i,index[i]]
        np.testing.assert_array_equal(errors[i,index[i]],anchor[i])
        allowed=[]
        for j in range(valid.shape[1]):
            if valid[i,j] and all(float(errors[i,j,a])<=float(anchor[i,a]) for a in (0,1)):
                allowed.append(j);safe[i,j]=True
        cost={j:max(float(errors[i,j,a])/float(scale[a]) for a in (0,1)) for j in allowed}
        tied=[j for j in allowed if cost[j]==min(cost.values())]
        frontier=[j for j in tied if not any(all(float(errors[i,k,a])<=float(errors[i,j,a]) for a in (0,1))
                    and any(float(errors[i,k,a])<float(errors[i,j,a]) for a in (0,1)) for k in tied)]
        def key(j):
            expert,hyp=names[j].split(':');return expert!='R0',hyp,expert
        target[i]=min(frontier,key=key)
    return target,safe


def scalar_context(raw,valid,index,mean,std):
    mean,std=np.asarray(mean,np.float32),np.asarray(std,np.float32)
    out=np.zeros((*valid.shape,189),np.float64)
    for i in range(len(valid)):
        if not valid[i].any():
            assert index[i]==-1
            continue
        assert int(index[i]) in (0,1) and valid[i,index[i]]
        base=((raw[i,index[i]]-mean)/std).astype(np.float64)
        for j in np.flatnonzero(valid[i]):
            z=((raw[i,j]-mean)/std).astype(np.float64)
            out[i,j,:94]=z
            out[i,j,94:188]=[abs(float(z[k])-float(base[k])) for k in range(94)]
            out[i,j,188]=float(j==index[i])
    return out


def scalar_margin(errors,valid,anchor,index,scale,target):
    risk=np.zeros(valid.shape,np.float64);margin=np.zeros(valid.shape,np.float64)
    for i in range(len(valid)):
        if not valid[i].any():
            assert target[i]==index[i]==-1 and np.isposinf(anchor[i]).all()
            continue
        assert int(index[i]) in (0,1) and valid[i,index[i]]
        assert np.isfinite(anchor[i]).all() and np.isfinite(errors[i,valid[i]]).all()
        np.testing.assert_array_equal(errors[i,index[i]],anchor[i])
        for j in np.flatnonzero(valid[i]):
            risk[i,j]=max(0.,(float(errors[i,j,0])-float(anchor[i,0]))/float(scale[0]),
                              (float(errors[i,j,1])-float(anchor[i,1]))/float(scale[1]))
            margin[i,j]=np.log1p(risk[i,j])
        assert valid[i,target[i]] and margin[i,target[i]]==risk[i,target[i]]==0.
    assert np.isfinite(risk).all() and np.isfinite(margin).all()
    return risk,margin


def scalar_objective(weight,x,valid,target,margin,ridge=1e-4):
    ce=0.;gradient=np.zeros(len(weight));prob=np.zeros(valid.shape)
    for i in range(len(x)):
        keep=np.flatnonzero(valid[i]).tolist()
        if not keep:
            assert target[i]==-1
            continue
        assert margin[i,target[i]]==0.
        logits=np.array([-sum(float(a)*float(b) for a,b in zip(weight,x[i,j]))+float(margin[i,j]) for j in keep])
        largest=max(logits);exponential=np.exp(logits-largest);p=exponential/sum(exponential)
        ce+=largest+math.log(sum(exponential))-logits[keep.index(int(target[i]))]
        gradient+=x[i,target[i]]
        for j,pj in zip(keep,p):
            prob[i,j]=pj;gradient-=pj*x[i,j]
    penalty=ridge*float(np.dot(weight,weight))/2
    return dict(value=ce/len(x)+penalty,CE=ce/len(x),penalty=penalty,
                gradient=gradient/len(x)+ridge*weight,probability=prob)


def scalar_hessian(x,valid,probability,ridge=1e-4):
    out=np.zeros((x.shape[-1],x.shape[-1]),np.float64)
    for i in range(len(x)):
        keep=np.flatnonzero(valid[i]).tolist()
        mean=sum((probability[i,j]*x[i,j] for j in keep),np.zeros(x.shape[-1]))
        for j in keep:
            delta=x[i,j]-mean;out+=probability[i,j]*np.outer(delta,delta)
    return out/len(x)+ridge*np.eye(x.shape[-1])


def expect_rejected(callback):
    try:
        callback()
    except (AssertionError,ValueError):
        return
    raise AssertionError('Invalid margin contract was accepted')


def synthetic_review(T,previous):
    rng=np.random.default_rng(2026100111)
    valid=np.array([[1,1,1,1],[0,1,1,0],[1,0,0,0],[0,0,0,0],[1,1,0,1]],bool)
    index=np.array([0,1,0,-1,1],np.int64)
    errors=np.array([[[2,3],[6,0],[1,2],[4,12]],[[np.inf,np.inf],[4,3],[8,0],[np.inf,np.inf]],
                     [[2,2],[np.inf,np.inf],[np.inf,np.inf],[np.inf,np.inf]],[[np.inf,np.inf]]*4,
                     [[5,4],[6,6],[np.inf,np.inf],[1,8]]],np.float64)
    anchor=np.full((5,2),np.inf);present=np.flatnonzero(valid.any(1));anchor[present]=errors[present,index[present]]
    scale=np.array([2.,3.]);names=T.candidate_names('UNION_s1')
    target,safe=scalar_targets(errors,valid,anchor,index,scale,names)
    yt,st=T.anchored_targets(errors,valid,anchor,index,scale,names)
    np.testing.assert_array_equal(target,yt);np.testing.assert_array_equal(safe,st)
    with np.errstate(all='raise'):
        risk,margin=T.margin_arrays(errors,valid,anchor,index,scale,target)
        r2,m2=scalar_margin(errors,valid,anchor,index,scale,target)
    np.testing.assert_array_equal(risk,r2);np.testing.assert_array_equal(margin,m2)
    np.testing.assert_array_equal(risk[0],[0.,2.,0.,3.])
    assert np.count_nonzero(risk[~valid])==np.count_nonzero(margin[~valid])==0
    assert not margin[present,target[present]].any()
    bad_target=target.copy();bad_target[0]=1
    expect_rejected(lambda:T.margin_arrays(errors,valid,anchor,index,scale,bad_target))
    bad_index=index.copy();bad_index[0]=-1
    expect_rejected(lambda:T.margin_arrays(errors,valid,anchor,bad_index,scale,target))
    raw=rng.normal(size=(5,4,94)).astype(np.float32);raw[~valid]=np.nan
    mean=rng.normal(size=94).astype(np.float32);std=(rng.random(94)+.5).astype(np.float32)
    x=scalar_context(raw,valid,index,mean,std)
    np.testing.assert_array_equal(x,T.context_inputs(raw,valid,index,mean,std))
    weight=rng.normal(size=189)*.03
    result=T.objective(weight,x,valid,target,margin)
    other=scalar_objective(weight,x,valid,target,margin)
    for k in ('value','CE','penalty'):
        assert abs(result[k]-other[k])<1e-13
    np.testing.assert_allclose(result['gradient'],other['gradient'],rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(result['probability'],other['probability'],rtol=1e-12,atol=1e-13)
    legacy=previous.objective(weight,x,valid,target)
    zero=T.objective(weight,x,valid,target,np.zeros_like(margin))
    for k in ('value','CE','penalty','gradient','probability'):
        np.testing.assert_array_equal(zero[k],legacy[k])
    # exp(log1p(r)) multiplies a candidate's unnormalized probability by1+r.
    weighted=legacy['probability']*(1+risk)
    weighted[present]/=weighted[present].sum(1)[:,None]
    np.testing.assert_allclose(result['probability'],weighted,rtol=1e-13,atol=1e-14)
    assert result['probability'][0,1]>0 and result['probability'][0,3]>0
    actual_H=T.hessian(x,result['probability'])
    independent_H=scalar_hessian(x,valid,other['probability'])
    np.testing.assert_allclose(actual_H,independent_H,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(actual_H,actual_H.T,rtol=0,atol=1e-12)
    eigen_min=float(np.linalg.eigvalsh(independent_H)[0]);assert eigen_min>=1e-4-1e-10
    gradient_difference=[];hessian_difference=[]
    for j in range(189):
        delta=np.zeros(189);delta[j]=1e-5
        plus=scalar_objective(weight+delta,x,valid,target,margin)
        minus=scalar_objective(weight-delta,x,valid,target,margin)
        gradient_difference.append(abs((plus['value']-minus['value'])/2e-5-result['gradient'][j]))
        hessian_difference.append(float(np.abs((plus['gradient']-minus['gradient'])/2e-5-actual_H[:,j]).max()))
    assert max(gradient_difference)<1e-8 and max(hessian_difference)<1e-8
    empty_valid=np.zeros((2,4),bool);empty_x=np.zeros((2,4,189));empty_y=np.full(2,-1,np.int64)
    empty=T.objective(weight,empty_x,empty_valid,empty_y,np.zeros((2,4)))
    assert empty['CE']==0.;np.testing.assert_array_equal(empty['gradient'],1e-4*weight)
    np.testing.assert_array_equal(T.hessian(empty_x,empty['probability']),1e-4*np.eye(189))
    bad_margin=margin.copy();bad_margin[0,target[0]]=1.
    expect_rejected(lambda:T.objective(weight,x,valid,target,bad_margin))
    assert 'margin' not in inspect.signature(T.score_candidates).parameters
    for fn in ('context_inputs','normalized_inputs','anchored_targets','hessian'):
        assert ast.dump(ast.parse(inspect.getsource(getattr(T,fn))))==ast.dump(ast.parse(inspect.getsource(getattr(previous,fn)))),fn
    final=dict(gradient=np.full(189,1e-7),value=.11,CE=.1,penalty=.01,unadjusted_CE=.09)
    cert=T.certificate(SimpleNamespace(success=True),final,2,1)
    assert cert['PASS'] and cert['lambda_l2']==1e-4 and cert['max_gap_upper_bound']==1e-6
    assert cert['gradient_l2_squared_over_2lambda']==float(np.linalg.norm(final['gradient']))**2/(2e-4)
    assert cert['CE']==.1 and cert['unadjusted_CE']==.09 and cert['loss_rule']==LOSS_RULE
    ck=dict(schema=T.CHECKPOINT_SCHEMA,loss_rule=LOSS_RULE,runtime_uses_margin=False,
        feature_map=T.FEATURE_MAP,feature_dim=189,raw_feature_dim=94,normalization='old_float32_then_float64',
        names=names,bias=0.,lambda_l2=1e-4,mean=mean.tolist(),std=std.tolist(),weight=weight.tolist())
    new_score=T.score_candidates(ck,raw,valid,index)
    old_score=previous.score_candidates({**ck,'schema':previous.CHECKPOINT_SCHEMA},raw,valid,index)
    np.testing.assert_array_equal(new_score,old_score)
    expected_score=np.where(valid,np.sum(x*weight,axis=2),np.inf)
    np.testing.assert_allclose(new_score,expected_score,rtol=1e-13,atol=1e-13)
    expect_rejected(lambda:T.score_candidates({**ck,'runtime_uses_margin':True},raw,valid,index))
    assert T.LAMBDA==1e-4 and T.MAX_ITER==1000 and T.MAX_CALLS==2000 and T.GAP_MAX==1e-6
    return dict(PASS=True,margin_scalar_bit_exact=True,context_scalar_bit_exact=True,target_scalar_bit_exact=True,
        zero_margin_reduces_exactly_to_previous_CE=True,target_margin_exactly_zero=True,
        inf_minus_inf_never_evaluated_under_raise=True,all_invalid_row_kept_zero_CE_and_ridge=True,
        invalid_candidate_risk_margin_zero=True,unsafe_candidates_remain_CE_competitors=True,
        probability_equals_original_times_one_plus_risk=True,
        independent_gradient_max_difference=float(np.abs(result['gradient']-other['gradient']).max()),
        gradient_finite_difference_max=max(gradient_difference),hessian_finite_difference_max=max(hessian_difference),
        independent_hessian_max_difference=float(np.abs(actual_H-independent_H).max()),hessian_min_eigenvalue=eigen_min,
        runtime_score_API_has_no_margin=True,runtime_score_matches_previous_exactly=True,
        certificate_checks_margin_objective_and_unchanged_bound=True,
        unchanged_helper_ASTs=['context_inputs','normalized_inputs','anchored_targets','hessian'],
        objective_scope='Invented fixtures only; no actual-data objective or policy trial.',new_fits=0)


def actual_review(T):
    OLD=T.OLD
    OLD.install_training_guard()
    parent=OLD.load_training_inputs()
    verify(read(CONTEXT_DOC/'TRAIN_PROTOCOL_SHA.json'))
    previous_protocol=read(CONTEXT_DOC/'TRAIN_PROTOCOL.json')
    for key,b in parent['protocol']['inputs'].items():
        assert previous_protocol['inputs'][key]==b
    feasibility_path=ROOT/previous_protocol['inputs']['source_feasibility']['path']
    verify(previous_protocol['inputs']['source_feasibility'])
    feasibility=read(feasibility_path)
    assert feasibility['complete'] and feasibility['PASS'] and feasibility['frames']==2598
    verify(previous_protocol['inputs']['prefit_review'])
    previous_prefit=read(ROOT/previous_protocol['inputs']['prefit_review']['path'])
    assert previous_prefit['complete'] and previous_prefit['PASS']
    old_inputs=previous_prefit['actual_TRAIN_inputs']
    assert old_inputs['ids_sha']==array_sha(parent['ids'])==feasibility['source_ids_sha']
    assert old_inputs['source_index_sha']==array_sha(parent['source_index'])==feasibility['source_index_sha']
    assert old_inputs['normalization_sha']==parent['normalization_sha']
    for key in ('features','train_labels'):
        verify(parent['protocol']['inputs'][key])
    with np.load(ROOT/parent['protocol']['inputs']['features']['path'],allow_pickle=False) as z:
        index=z['R0_GEO_index'][parent['source_index']]
        np.testing.assert_array_equal(z['ids'][parent['source_index']],parent['ids'])
    with np.load(ROOT/parent['protocol']['inputs']['train_labels']['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],parent['ids'])
        anchor=np.stack([z['R0_GEO_T_cm'],z['R0_GEO_R_deg']],axis=-1)
    assert index.shape==(2598,) and anchor.shape==(2598,2)
    finite=np.isfinite(anchor).all(1)
    assert np.array_equal(finite,index>=0) and finite.sum()==2597
    assert parent['ids'][~finite].tolist()==['TEX__shard_04_f0110']
    assert array_sha(anchor)==feasibility['anchor']['errors_sha']
    assert array_sha(index)==feasibility['anchor']['index_sha']==old_inputs['anchor_index_sha']
    models={}
    for model in MODELS:
        parents=['R0']+([] if model=='R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
        names=T.candidate_names(model)
        raw=np.concatenate([parent['features'][m] for m in parents],axis=1)
        valid=np.concatenate([parent['valid'][m] for m in parents],axis=1)
        errors=np.concatenate([parent['errors'][m] for m in parents],axis=1)
        raw_sha,valid_sha,err_sha=array_sha(raw),array_sha(valid),array_sha(errors)
        target,safe=scalar_targets(errors,valid,anchor,index,parent['scale'],names)
        t2,s2=T.anchored_targets(errors,valid,anchor,index,parent['scale'],names)
        np.testing.assert_array_equal(target,t2);np.testing.assert_array_equal(safe,s2)
        with np.errstate(all='raise'):
            risk,margin=scalar_margin(errors,valid,anchor,index,parent['scale'],target)
            r2,m2=T.margin_arrays(errors,valid,anchor,index,parent['scale'],target)
        np.testing.assert_array_equal(risk,r2);np.testing.assert_array_equal(margin,m2)
        context=scalar_context(raw,valid,index,parent['mean'],parent['std'])
        np.testing.assert_array_equal(context,T.context_inputs(raw,valid,index,parent['mean'],parent['std']))
        previous=feasibility['models'][model]
        assert array_sha(target)==previous['target_sha'] and array_sha(safe)==previous['safe_mask_sha']
        assert valid_sha==previous['original_valid_sha'] and err_sha==previous['unscaled_errors_sha']
        assert array_sha(context)==old_inputs['models'][model]['context_sha']
        assert raw_sha==old_inputs['models'][model]['raw_features_sha']
        assert not risk[~valid].any() and not margin[~valid].any()
        assert not margin[finite,target[finite]].any()
        np.testing.assert_array_equal(risk>0,valid&~safe)
        assert np.array_equal(valid.any(1),finite)
        assert array_sha(raw)==raw_sha and array_sha(valid)==valid_sha and array_sha(errors)==err_sha
        models[model]=dict(risk_sha=OLD.array_sha(risk),margin_sha=OLD.array_sha(margin),target_sha=OLD.array_sha(target),
            original_valid_sha=OLD.array_sha(valid),context_sha=OLD.array_sha(context),safe_mask_sha=OLD.array_sha(safe),
            unscaled_errors_sha=err_sha,raw_features_sha=raw_sha,anchor_errors_sha=array_sha(anchor),anchor_index_sha=array_sha(index),
            risk_dtype=str(risk.dtype),margin_dtype=str(margin.dtype),shape=list(risk.shape),frames=2598,
            available_anchor_rows=2597,failed_rows=1,original_valid_candidates=int(valid.sum()),positive_margin_candidates=int((margin>0).sum()),
            invalid_candidates_zero=True,target_margin_zero=True,previous_target_context_valid_error_hashes_exact=True,
            scalar_target_context_risk_margin_exact=True,input_arrays_unchanged=True)
    return models,dict(parent_train_protocol=parent['binding'],context_train_protocol=bind(CONTEXT_DOC/'TRAIN_PROTOCOL.json'),
        previous_prefit=previous_protocol['inputs']['prefit_review'],source_feasibility=previous_protocol['inputs']['source_feasibility'],
        features=parent['protocol']['inputs']['features'],train_labels=parent['protocol']['inputs']['train_labels'],
        source_contract=parent['protocol']['inputs']['source_contract'],source_ids_sha=array_sha(parent['ids']),
        source_index_sha=array_sha(parent['source_index']),normalization_sha=parent['normalization_sha'],
        scale=parent['scale'].tolist(),scale_sha=array_sha(parent['scale']),anchor_index_sha=array_sha(index),
        anchor_errors_sha=array_sha(anchor),failed_ids=parent['ids'][~finite].tolist())


def run(write):
    sys.dont_write_bytecode=True
    install_write_guard()
    from . import convex_train as T
    from scripts.research.pallet_pose_anchor_context_20261001_v1 import convex_train as previous
    assert T.OLD.READS is None
    code=bind(T.__file__)
    with threadpool_limits(limits=1):
        synthetic=synthetic_review(T,previous)
    if not write:
        print(json.dumps(synthetic,indent=2));return
    models,inputs=actual_review(T)
    assert bind(T.__file__)==code,'Trainer changed during review; retry after freeze.'
    result=dict(complete=True,PASS=True,created_at=datetime.now(timezone.utc).isoformat(),
        source_TRAIN_only=True,frames=2598,available_anchor_rows=2597,failed_rows_retained=1,
        loss_rule=LOSS_RULE,runtime_uses_margin=False,feature_map='normalized94_abs_anchor_delta94_identity1',
        raw_feature_dim=94,feature_dim=189,models=models,inputs=inputs,synthetic=synthetic,
        trainer=code,reviewer=bind(__file__),source_TRAIN_cached_label_values_read=True,
        raw_source_reference_reads=0,VAL_quality_read=False,real_targets_read=False,new_fits=0,
        actual_data_objective_trials=0,alternative_policy_probes=0,new_reference_metric_calculations=0,
        image_forwards=0,new_PnP_calls=0,optimizer_steps=0,
        scope='Prefit fixed supervision/input integrity and invented mathematical checks; no model performance computation.',
        mathematical_contract=dict(logit='-score_c+log1p(risk_c)',risk='max((T-Ta)/sT,(R-Ra)/sR,0)',
            target_margin=0,invalid_margin=0,normalization='all2598 rows; invalid rows contribute zeroCE, not zero ridge',
            hessian='mean sum_c p_c (x_c-E[x])(x_c-E[x])^T + lambda I; labels/margins fixed during fit',
            strong_convexity_lambda=1e-4,certificate_scope='Bound for this margin-adjusted objective only; no T/R safety guarantee'),
        separate_real_scope_restoration='Root restores original stable SINGLE251 three-seed comparisons alongside matched R0_ONLY/pairedD real gates before any new learned real route. This is evaluation coverage restoration, not the TRAIN loss change; no real execution or numeric verification in this prefit audit.',
        read_paths=sorted(READS),training_guard_read_paths=sorted(set(T.OLD.READS)),method_success=False,goal_complete=False)
    md=f'''# TRAIN 위험 가산항 손실의 학습 전 독립 검산

**계약·수학 검산 PASS. 새 fit과 성능 평가는 실행하지 않았다.** 이번 변경은 후보별 `risk=max((T−Ta)/sT,(R−Ra)/sR,0)`의 `log1p(risk)`를 TRAIN logit에 더하는 하나의 손실 변경이다. margin은 참조를 쓰는 TRAIN 상수이며 runtime 입력이나 선택 mask가 아니다.

고정된 target y의 margin은 정확히0이다. 새 손실은 `CE(-score+margin,y)+(λ/2)||w||²`, λ=1e−4다. 후보별 비정규화 softmax 질량을 기존 값의 `1+risk`배로 만드는 것과 같고, 원래 유효한 unsafe 후보도 경쟁자로 남는다. `risk=0`이면 이전 CE+ridge의 값·gradient·probability를 정확히 재현한다.

독립 scalar 구현으로 합성 risk·margin·target·189 map을 검산했다. 189개 중앙차분의 최대 gradient 오차는 {synthetic['gradient_finite_difference_max']:.3g}, Hessian 오차는 {synthetic['hessian_finite_difference_max']:.3g}였다. 별도 covariance Hessian과도 일치했으며 최소 고유값은 {synthetic['hessian_min_eigenvalue']:.9g}였다. 고정 margin에서 Hessian은 확률가중 특징 covariance+λI이므로 가중치에 대한 λ-strong convexity를 유지한다. 이는 수정한 목적함수의 수렴 인증 근거이며 T/R 성능 인증이 아니다.

전체 실패행은 `inf−inf`를 계산하지 않는다. 먼저0 배열을 만들고 valid 후보만 차분하며, NumPy 부동소수 예외를 raise로 둔 합성/실제 검산을 통과했다. invalid와 all-invalid risk/margin은0이지만 그 행을 안전한 예측으로 세지 않는다. target=-1인 행은 CE0으로 전체 분모에 남으며, ridge는 그대로다. 잘못된 anchor와 nonzero target margin은 거부한다.

기존 TRAIN guard 아래 허용된 cached TRAIN label NPZ만 사용해 **2,598행**을 재구성했다. 유효 anchor는2,597개, 기존 실패 `TEX__shard_04_f0110` 한 행은 유지했다. 네 모델의 risk/margin은 독립 scalar 계산과 bit 단위로 같고, 모델별 SHA를 JSON에 고정했다. 기존 SOURCE_FEASIBILITY의 target/safe/원본 error/valid SHA, 직전189 prefit의 context/raw feature SHA, 정규화·ID 순서·source index·고정 scale 연결이 모두 같다. 새로운 target 선택 규칙이나 후보/분모를 도입하지 않았다.

runtime score 함수의 API에는 margin이 없으며 context/normalization/target/Hessian helper AST가 직전 구현과 같다. runtime 점수는 같은 합성 입력·가중치의 이전 함수와 정확히 일치한다. 새 loss 식별자와 runtime margin 금지 검사를 추가했고, 수렴 인증은 변경한 margin 목적함수의 gradient를 같은 λ bound로 계산함을 별도 확인했다. 학습 가산항을 VAL이나 실사에서 참조값으로 다시 계산할 경로를 추가하지 않았다.

별도로 root가 원래 실사 계약의 SINGLE251 세 seed 비교를 matched R0_ONLY/pairedD 기준과 함께 복구하고 있다. 이는 실제 실사 실행 전 평가 범위를 원래 요구와 맞추는 작업이며 이번 TRAIN 손실 변경과 구분한다. 본 검산에서는 해당 실사 기준을 채점하거나 실사 참조를 읽지 않았다.

실제 TRAIN 배열에서는 고정 supervision SHA만 계산했다. 새 모델의 objective/정책 시험·오차 요약·oracle 비교는 하지 않았다. VAL 품질·실사 GT·원본 source 참조를 읽지 않았고 fit/optimizer/image forward/PnP는0회다. 이후 고정 protocol에 따른 학습과 기존 source45개 조건을 통과해야 별도의 실사 단계 자격을 평가할 수 있다.

[검산 JSON 및 모델별 SHA](PREFIT_REVIEW.json)
'''
    DOC.mkdir(parents=True,exist_ok=True)
    with (DOC/'PREFIT_REVIEW_KO.md').open('x') as f:f.write(md)
    result['note']=bind(DOC/'PREFIT_REVIEW_KO.md')
    with (DOC/'PREFIT_REVIEW.json').open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('ANCHOR_RISK_PREFIT_REVIEW_PASS',bind(DOC/'PREFIT_REVIEW.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('selfcheck','review'))
    run(parser.parse_args().stage=='review')
