"""Independent271x2 TRAIN verification; no fitting or VAL/real reference reads."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from . import common as C
from . import prefit_review as P
from scripts.research.pallet_pose_signed_axes_sign_20261001_v1 import verify_train as N
from scripts.research.pallet_pose_signed_axes_sign_20261001_v1 import convex_train as PREV
from scripts.research.pallet_pose_anchor_rbf_20261001_v1 import verify_train as V
from scripts.research.pallet_pose_union_selection_20261001_v1 import train as OLD
from scripts.research.pallet_pose_anchor_context_20261001_v1.train_risk_diagnostic import classify

OLD_HASH_KEYS=P.OLD_HASH_KEYS
HASH_KEYS=OLD_HASH_KEYS+P.EXTRA_HASH_KEYS
METADATA=dict(N.METADATA,feature_dim=271,base_feature_dim=253,direction_dim=18,
    feature_map=N.METADATA['feature_map']+'_signed_direction18',
    input_rule='RBF253_DIFFERENCE_PLUS_NORMALIZED_DIRECTION18_DIFFERENCE',
    direction_rule=P.DIRECTION_RULE,direction_normalization=P.DIRECTION_NORMALIZATION)
READS=set()


def guard():
    sys.dont_write_bytecode=True
    arrays={C.PARENT_RAW/'SOURCE_FEATURES.npz',C.PARENT_RAW/'SOURCE_TRAIN_LABELS.npz',C.DIRECTION_RAW/'TRAIN_DIRECTIONS.npz'}
    outputs={C.DOC/'TRAIN_CONVERGENCE.json',C.DOC/'TRAIN_CONVERGENCE_KO.md'}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();name=str(path);mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing and path.is_relative_to(C.ROOT):
            assert path in outputs,('TRAIN_VERIFICATION_OUTPUT_ONLY',name);return
        assert not any(t in name for t in ('/data/evaluation/','/real_gt_v2/','/annotations/',
            'GEOMETRY_RESOLVED_POSE_GT','GEOMETRY_SIDETABLE','SYNTH_RECORDS','SYNTH_LABELS',
            'SOURCE_VAL_','REAL_RESULTS','REAL_CHOICES','POSE_METRICS','TRUTH_FOR_DISPLAY')),('TRAIN_ONLY',name)
        assert path.suffix.lower() not in ('.png','.jpg','.jpeg','.pt','.pth','.onnx'),name
        if path.suffix=='.npz':assert path in arrays,('TRAIN_ARRAYS_ONLY',name)
        if '/fits/' in name:assert path.is_relative_to(C.RAW/'fits') or path.is_relative_to(C.SIGN_RAW/'fits'),name
        if path.is_relative_to(C.ROOT):READS.add(str(path.relative_to(C.ROOT)))
    sys.addaudithook(hook)


def reconstruct(data,model):
    experts=['R0']+([] if model=='R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
    raw,valid,errors=[np.concatenate([data[key][m] for m in experts],axis=1) for key in ('features','valid','errors')]
    direction=np.concatenate([data['direction'][m] for m in experts],axis=1)
    phi189=V.independent_context(raw,valid,data['index'],data['mean'],data['std'])
    phi=V.independent_rbf(phi189,valid,data['basis'])
    x253=N.independent_difference(phi,valid,data['index'])
    extra=P.direction_difference(direction,valid,data['index'],data['mean18'],data['std18'])
    x=np.concatenate([x253,extra],axis=2)
    with np.errstate(all='raise'):
        scaled,target=N.independent_targets(errors,valid,data['anchor'],data['index'],data['scale'])
    hashes=dict(signed_target_sha=P.array_sha(target),input_difference_sha=P.array_sha(x253),
        base_context_sha=P.array_sha(phi),errors_sha=P.array_sha(errors),scaled_excess_sha=P.array_sha(scaled),
        original_valid_sha=P.array_sha(valid),direction_raw_sha=P.array_sha(direction),
        direction_difference_sha=P.array_sha(extra),extended_input_sha=P.array_sha(x))
    for key in OLD_HASH_KEYS:assert hashes[key]==data['previous']['models'][model][key]
    assert hashes['direction_difference_sha']==data['representation']['models'][model]['extra18_sha']
    assert hashes['extended_input_sha']==data['representation']['models'][model]['extended271_sha']
    return raw,valid,errors,direction,x253,x,scaled,target,hashes


def metadata_check(obj):
    for key,value in METADATA.items():assert obj[key]==value,(key,obj.get(key))


def main():
    torch.set_num_threads(1);guard()
    from . import convex_train as T
    protocol=C.protocol('TRAIN_PROTOCOL');pb=C.bind(C.DOC/'TRAIN_PROTOCOL.json')
    assert T.metadata()==METADATA;metadata_check(protocol)
    assert protocol['solver']==T.solver_config()==N.SOLVER_CONFIG
    assert protocol['certificate']==dict(optimizer_success=True,gap_upper_bound_max=1e-6,gradient_linf_max=1e-8)
    assert protocol['lambda_l2']==1e-4 and protocol['bias']==0. and protocol['train_rows']==2598
    complete=C.read(C.DOC/'TRAINING_COMPLETE.json');metadata_check(complete)
    assert complete['complete'] and complete['all_certified'] and complete['fit_count']==4
    assert complete['models']==list(C.MODEL_NAMES) and complete['protocol']==pb
    assert complete['max_objective_calls_per_fit']==2000
    assert complete['source_TRAIN_only'] and not complete['VAL_quality_read'] and not complete['real_targets_read']
    assert {p.name for p in (C.RAW/'fits').iterdir()}==set(C.MODEL_NAMES)
    for m in C.MODEL_NAMES:
        assert not (C.RAW/'fits'/m/'FAILED.json').exists() and not (C.DOC/f'REJECTED_{m}.json').exists()
    for key in (*HASH_KEYS,'final_accepted_call'):assert set(complete[key+'_by_model'])==set(C.MODEL_NAMES)
    prefit=C.read(C.DOC/'PREFIT_REVIEW.json');metadata_check(prefit)
    assert prefit['complete'] and prefit['PASS'] and protocol['inputs']['prefit_review']==C.bind(C.DOC/'PREFIT_REVIEW.json')
    data=P.load_cached_inputs();index,anchor=data['index'],data['anchor'];basis=data['basis'];basis_binding=data['basis_binding']
    norm=data['inputs']['normalization_sha'];dnorm=data['inputs']['direction_normalization_sha'];db=data['inputs']['direction_receipt']
    assert len(data['ids'])==2598 and data['ids'][index<0].tolist()==['TEX__shard_04_f0110']
    for obj in (protocol,complete,prefit):
        assert obj['direction_normalization_sha']==dnorm and obj['direction_receipt_binding']==db
        actual_basis=obj['basis_SHA_bind'] if 'basis_SHA_bind' in obj else obj['inputs']['rbf_basis']
        assert actual_basis==basis_binding
    assert complete['anchor_index_sha']==prefit['anchor_index_sha']==P.array_sha(index)
    for key in ('direction_features','direction_receipt','direction_representation','direction_verification',
                'direction_protocol','previous_sign_prefit','previous_sign_protocol','source_metadata'):
        assert protocol['inputs'][key]==data['inputs'][key]
    previous_binding=C.bind(C.SIGN_DOC/'TRAIN_CONVERGENCE.json')
    assert protocol['evidence']['previous_train_verification']==previous_binding
    previous=C.read(C.SIGN_DOC/'TRAIN_CONVERGENCE.json')
    assert previous['complete'] and previous['PASS'] and previous['source_TRAIN_only']
    # The authenticated prior receipt also lists still-older Newton weights.
    # Do not reopen those unrelated fits; verify the selected sign checkpoint
    # and its own START/trace/receipt chain below.
    previous_bindings={b['path']:b for b in previous['bindings']}
    bindings=[pb,C.bind(C.DOC/'TRAINING_COMPLETE.json'),C.bind(C.DOC/'PREFIT_REVIEW.json'),
        C.bind(C.SIGN_DOC/'TRAIN_CONVERGENCE.json'),basis_binding,db,
        C.bind(C.DIRECTION_DOC/'VERIFICATION.json'),C.bind(C.DIRECTION_DOC/'REPRESENTATION_AUDIT.json')]
    for module in (T,C,P,N,V,PREV,OLD):bindings.append(C.bind(module.__file__))
    bindings.extend([C.bind(__file__),C.bind(Path(classify.__code__.co_filename))])
    codes={b['path']:b for b in protocol['codes']}
    for module in (T,C,OLD,T.B):assert codes[C.bind(module.__file__)['path']]==C.bind(module.__file__)
    records={}
    for fb in complete['fits']:
        C.verify(fb);fit=C.read(C.ROOT/fb['path']);m=fit['model']
        assert m in C.MODEL_NAMES and m not in records and fb==C.bind(C.DOC/f'FIT_{m}.json')
        assert fit['complete'] and fit['protocol']==pb and fit['fits_executed']==1
        assert fit['source_TRAIN_only'] and not fit['VAL_quality_read'] and not fit['real_targets_read']
        for key in ('START','trace','checkpoint'):C.verify(fit[key])
        ck=C.read(C.ROOT/fit['checkpoint']['path']);start=C.read(C.ROOT/fit['START']['path'])
        assert fit['checkpoint']==C.bind(C.RAW/'fits'/m/'final.json')
        assert ck['schema']=='pallet_pose_signed_axes_direction_linear271x2_v1'
        assert ck['model']==start['model']==m and ck['protocol']==start['protocol']==pb
        for obj in (ck,start,fit):
            metadata_check(obj)
            assert obj['basis_SHA_bind']==basis_binding and obj['anchor_index_sha']==P.array_sha(index)
            assert obj['direction_normalization_sha']==dnorm and obj['direction_receipt_binding']==db
        assert ck['rbf_basis']==basis and ck['rbf_basis_binding']==basis_binding
        assert ck['bias']==0. and ck['lambda_l2']==1e-4 and ck['loss_uses_original_valid_mask'] is True
        assert ck['normalization']=='old_float32_then_float64'
        assert ck['normalization_sha']==fit['normalization_sha']==norm
        for key,expected in (('mean',data['mean']),('std',data['std']),('direction_mean',data['mean18']),('direction_std',data['std18'])):
            P.exact(np.asarray(ck[key],np.float32),expected)
        names=T.candidate_names(m);assert ck['names']==start['names']==names
        raw,valid,errors,direction,x253,x,scaled,target,hashes=reconstruct(data,m)
        for key in HASH_KEYS:assert hashes[key]==ck[key]==fit[key]==start[key]==complete[key+'_by_model'][m]==prefit['models'][m][key]
        P.exact(x,T.difference_inputs(raw,valid,index,ck['mean'],ck['std'],basis,direction,ck['direction_mean'],ck['direction_std']))
        s2,y2=T.signed_targets(errors,valid,anchor,index,data['scale']);P.exact(scaled,s2);P.exact(target,y2)
        assert (~valid.any(1)).sum()==start['no_valid_rows']==1 and start['train_rows']==2598
        assert start['source_ids_sha']==P.array_sha(data['ids']) and start['anchor_errors_sha']==P.array_sha(anchor)
        assert ck['axis_order']==start['axis_order']==['translation_cm','rotation_deg']
        P.exact(np.asarray(ck['signed_target_scales']),data['scale']);P.exact(np.asarray(start['signed_target_scales']),data['scale'])
        w=np.asarray(ck['weight'],np.float64);assert w.shape==(271,2) and np.isfinite(w).all()
        assert P.array_sha(w)==fit['final_weight_sha'] and start['initial_weight_sha']==P.array_sha(np.zeros((271,2),np.float64))
        assert start['solver']==protocol['solver'] and start['certificate_rule']==protocol['certificate']
        actual=N.independent_objective(w,x,valid,target,with_hessian=True);native=T.objective(w,x,valid,target)
        np.testing.assert_allclose(actual['gradient'],native['gradient'],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(actual['prediction'],native['prediction'],rtol=1e-12,atol=1e-12)
        cert=ck['certificate'];assert cert==fit['certificate'] and cert['PASS'] and cert['optimizer_success']
        assert cert['lambda_l2']==1e-4 and cert['max_gap_upper_bound']==1e-6 and cert['gradient_linf_max']==1e-8
        assert cert['solver_rule']==METADATA['solver_rule'] and cert['loss_rule']==METADATA['loss_rule']
        assert cert['huber_delta']==1. and cert['output_dim']==2 and cert['sign_rule']==METADATA['sign_rule'] and cert['sign_coefficient']==1.
        for key,ckey in (('objective','objective_value'),('Huber','Huber'),('Sign_logistic','Sign_logistic'),('L2_penalty','L2_penalty'),('gradient_l2','gradient_l2'),('gradient_linf','gradient_linf')):
            assert abs(actual[key]-cert[ckey])<=1e-10,(m,key)
        assert actual['gradient_linf']<=1e-8 and actual['certified_gap_upper_bound']<=1e-6
        assert abs(actual['certified_gap_upper_bound']-cert['gradient_l2_squared_over_2lambda'])<=1e-12
        H=T.hessian(x,valid,native['residual'],target)
        np.testing.assert_allclose(actual['Hessian'],H,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(actual['Hessian'],actual['Hessian'].T,rtol=1e-12,atol=1e-12)
        eigen=np.linalg.eigvalsh(actual['Hessian']);assert eigen[0]>=1e-4-1e-10
        np.testing.assert_allclose([eigen[0],eigen[-1]],[ck['solver']['final_hessian_min'],ck['solver']['final_hessian_max']],rtol=1e-8,atol=1e-10)
        runtime_axes=T.predict_axes(ck,raw,valid,index,direction);runtime=T.score_candidates(ck,raw,valid,index,direction)
        np.testing.assert_allclose(runtime_axes,actual['prediction'],rtol=1e-12,atol=1e-12)
        score=np.where(valid,actual['prediction'].max(2),np.inf)
        np.testing.assert_allclose(runtime,score,rtol=1e-12,atol=1e-12)
        chosen=V.select(runtime,valid,names)
        P.exact(chosen,V.select(score,valid,names));P.exact(chosen,T.select_candidates(runtime,valid,names))
        rows=np.flatnonzero(valid.any(1));assert np.all(runtime[rows,chosen[rows]]<=0.)
        legacy,safe=V.independent_targets(errors,valid,anchor,index,data['scale'],names)
        stats,picked=V.score_statistics(runtime,valid,legacy,safe,errors,anchor,names);P.exact(chosen,picked)
        risk=classify(chosen,valid,errors,anchor,index,data['scale'])
        trace=[json.loads(line) for line in (C.ROOT/fit['trace']['path']).read_text().splitlines()]
        calls=[r for r in trace if r['event']=='objective'];iterations=[r for r in trace if r['event']=='iteration']
        assert len(calls)==fit['objective_calls']==cert['objective_calls'] and len(iterations)==fit['iterations']==cert['iterations']
        assert ck['fits_executed']==1 and ck['optimizer_steps']==fit['optimizer_steps']==len(iterations)
        assert ck['final_accepted_call']==complete['final_accepted_call_by_model'][m]
        tracecheck=N.verify_newton_trace(trace,start,fit,ck)
        zero=N.independent_objective(np.zeros((271,2)),x,valid,target)
        assert abs(calls[0]['objective']-zero['objective'])<=1e-12
        for row in trace:
            metadata_check(row)
            for key in HASH_KEYS:assert row[key]==hashes[key]
            assert row['basis_SHA_bind']==basis_binding and row['anchor_index_sha']==P.array_sha(index)
            assert row['direction_normalization_sha']==dnorm and row['direction_receipt_binding']==db
        oldfb=C.bind(C.SIGN_DOC/f'FIT_{m}.json');assert previous_bindings[oldfb['path']]==oldfb
        oldfit=C.read(C.ROOT/oldfb['path']);C.verify(oldfit['checkpoint'])
        for key in ('START','trace'):
            assert previous_bindings[oldfit[key]['path']]==oldfit[key]
            C.verify(oldfit[key])
        assert previous_bindings[oldfit['checkpoint']['path']]==oldfit['checkpoint']
        oldck=C.read(C.ROOT/oldfit['checkpoint']['path'])
        assert oldck['schema']=='pallet_pose_signed_axes_sign_linear253x2_v1'
        assert oldck['rbf_basis']==basis and oldck['basis_SHA_bind']==basis_binding and oldck['names']==names and oldck['certificate']['PASS']
        for key in ('mean','std'):P.exact(np.asarray(oldck[key],np.float32),data[key])
        for key in OLD_HASH_KEYS:assert hashes[key]==oldck[key]==previous['models'][m]['hashes'][key]
        ow=np.asarray(oldck['weight'],np.float64);assert ow.shape==(253,2)
        embedded=np.concatenate([ow,np.zeros((18,2),np.float64)],axis=0)
        oldnative=N.independent_objective(ow,x253,valid,target)
        oldnew=N.independent_objective(embedded,x,valid,target)
        nativeold=PREV.objective(ow,x253,valid,target)
        np.testing.assert_allclose(oldnative['gradient'],nativeold['gradient'],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(oldnew['gradient'][:253],oldnative['gradient'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(oldnew['prediction'],oldnative['prediction'],rtol=1e-12,atol=1e-12)
        for key,ckey in (('objective','objective_value'),('Huber','Huber'),('Sign_logistic','Sign_logistic'),('L2_penalty','L2_penalty'),('gradient_l2','gradient_l2'),('gradient_linf','gradient_linf')):
            assert abs(oldnative[key]-oldck['certificate'][ckey])<=1e-10
        for key in ('objective','Huber','Sign_logistic','L2_penalty'):assert abs(oldnew[key]-oldnative[key])<=1e-10
        # The embedded old weight is feasible. Strong convexity bounds the
        # current objective above that feasible value only by the current
        # certified gap; do not turn a numerical comparison into a new gate.
        comparison_roundoff=1e-10
        assert actual['objective']<=oldnew['objective']+actual['certified_gap_upper_bound']+comparison_roundoff
        oldaxes=np.einsum('nkd,da->nka',x253,ow);oldscore=np.where(valid,oldaxes.max(2),np.inf)
        oldruntime=PREV.score_candidates(oldck,raw,valid,index);P.exact(oldruntime,oldscore)
        oldstats,oldchosen=V.score_statistics(oldruntime,valid,legacy,safe,errors,anchor,names)
        oldrisk=classify(oldchosen,valid,errors,anchor,index,data['scale'])
        assert oldstats==previous['models'][m]['statistics'] and oldrisk==previous['models'][m]['risk_statistics']
        records[m]=dict(independent_recompute=N.public_objective(actual),original_certificate=cert,
            objective_abs_difference=abs(actual['objective']-cert['objective_value']),
            gradient_vector_max_abs_difference=float(np.abs(actual['gradient']-native['gradient']).max()),
            Hessian_max_abs_difference=float(np.abs(actual['Hessian']-H).max()),
            independent_Hessian_min_eigenvalue=float(eigen[0]),independent_Hessian_max_eigenvalue=float(eigen[-1]),
            Hessian_scope='542x542 C-order block matrix; Huber kink contributes0, nonzero-target logistic curvature and ridge remain.',
            objective_calls=len(calls),iterations=len(iterations),newton_trace=tracecheck,
            statistics=stats,risk_statistics=risk,regression_statistics=N.regression_statistics(runtime_axes,target,valid,index,chosen),
            legacy_discrete_target_accuracy_scope='Secondary anchored-target diagnostic, not regression supervision.',
            previous_fixed_sign=dict(checkpoint=oldfit['checkpoint'],receipt=oldfb,statistics=oldstats,risk_statistics=oldrisk,
                original_choice_replay_PASS=True,same_new_objective=N.public_objective(oldnew),
                original_native253_objective=N.public_objective(oldnative),original_native253_certificate=oldck['certificate'],
                original_objective_and_native253_gradient_parity=True,zero18_embedding_prediction_and_objective_parity=True,
                embedding_atol=1e-10,extra18_gradient_l2=float(np.linalg.norm(oldnew['gradient'][253:])),
                extra18_gradient_not_compared_to_original_certificate=True,
                regression_statistics=N.regression_statistics(oldaxes,target,valid,index,oldchosen),
                comparison_scope='Same Huber+sign+ridge at previous certified253x2 weights padded with18 zero rows. Original gradient certification checked only in native253 space.'),
            same_new_objective_decrease=oldnew['objective']-actual['objective'],
            feasible_old_embedding_consistency=dict(PASS=True,current_gap_allowance=actual['certified_gap_upper_bound'],roundoff_allowance=comparison_roundoff,performance_gate=False),
            changed_choice_count=int((chosen!=oldchosen).sum()),
            basis_SHA_bind=basis_binding,direction_receipt_binding=db,direction_normalization_sha=dnorm,
            hashes=hashes,runtime_axes_sha=P.array_sha(runtime_axes),runtime_axis_and_selection_parity=True,all_invalid_frame_retained=True)
        bindings += [fb,*(fit[k] for k in ('START','trace','checkpoint')),oldfb,oldfit['checkpoint']]
        print('SIGNED_AXES_DIRECTION_TRAIN_VERIFIED',m,'gap',actual['certified_gap_upper_bound'],flush=True)
    assert set(records)==set(C.MODEL_NAMES)
    assert sum(r['objective_calls'] for r in records.values())==complete['total_objective_calls']
    assert sum(r['iterations'] for r in records.values())==complete['total_iterations']
    out=dict(complete=True,PASS=True,independent_check_PASS=True,created_at=C.now(),protocol=pb,**METADATA,
        models=records,bindings=bindings,basis_SHA_bind=basis_binding,direction_receipt_binding=db,direction_normalization_sha=dnorm,
        source_TRAIN_only=True,frames=2598,available_anchor_rows=2597,failed_rows=1,
        VAL_quality_read=False,real_targets_read=False,raw_source_reference_reads=0,new_fits=0,optimizer_steps=0,
        alternative_policy_probes=0,method_success=False,goal_complete=False,previous_weight_warmstart=False,
        comparison='Same Huber+sign+ridge evaluated on current271x2 and prior certified253x2+zero18 weights; native253 gradient certificate is separate from padded271 gradient.',
        objective_definition='Per-frame original valid candidate/two-axis Huber+nonzero-target sign logistic mean, then full2598 mean, ridge1e-4 over542 weights.',
        validation='Independent scalar targets/differences and old253/fixed18 construction, Torch64 autograd, separate weighted-design542 Hessian.',
        read_paths=sorted(READS))
    C.save(C.DOC/'TRAIN_CONVERGENCE_KO.md',report(out));C.save(C.DOC/'TRAIN_CONVERGENCE.json',out)
    print('SIGNED_AXES_DIRECTION_INDEPENDENT_TRAIN_VERIFICATION_PASS_ALL4',flush=True)


def report(x):
    lines=['# 고정 잔차 방향18 추가: 독립 TRAIN 검산','',
        '**독립 수치 검산 PASS는 고정 목적식의 수렴·기록 확인이며 source/실사 T/R 개선 판정이 아니다.** TRAIN2,598행(유효2,597·실패1)을 유지했다. 기존253 입력에 O에서 고정한 signed residual18을 추가해271×2=542개 계수를 학습했다.','',
        '같은 Huber+nonzero-target sign logistic+ridge 목적식·타깃·scale·유효 후보를 유지한다. 각 frame의 원래 유효후보×2축 평균 후 전체2,598행 평균이며 실패1행은 손실0으로 분모에 남는다. 94/18 정규화는 고정 R0 TRAIN FP32 값이고, FP64 변환 뒤 candidate−anchor 차이를 계산한다.','',
        'Torch64 autograd gradient와 별도542×542 가중 design Hessian을 trainer·인증과 대조했다. 초기/최종 값을 독립 계산하고 모든 호출·accepted iteration·Armijo 반감·최종 accepted-state 기록을 검사했다. 중간 weight는 저장되지 않아 중간 Newton 방향을 모두 재계산했다고 주장하지 않는다. 새 fit·VAL·실사 참조는0이다.','',
        '| 모델 | Huber | sign logistic | J | gradient Linf | gap 상한 | 호출 |','|---|---:|---:|---:|---:|---:|---:|']
    for m,r in x['models'].items():
        q=r['independent_recompute'];lines.append(f"| {m} | {q['Huber']:.9f} | {q['Sign_logistic']:.9f} | {q['objective']:.9f} | {q['gradient_linf']:.3g} | {q['certified_gap_upper_bound']:.3g} | {r['objective_calls']} |")
    lines+=['','## 동일 목적식의 이전 모델 비교','',
        '직전 sign253×2 가중치에18개 영행을 붙여 새271입력에서 계산했다. 원래 예측·목적식과253축 gradient를 재현했다. 추가18축 gradient는 일반적으로0이 아니므로 이전 인증과 비교하지 않는다. 두 J는 같은 Huber+sign+ridge이며 이전 weight를 초기값으로 사용하지 않았다.','',
        '| 모델 | 이전253+zero18 J | 새271 J | 감소 | 실제 선택 변경 |','|---|---:|---:|---:|---:|']
    for m,r in x['models'].items():
        lines.append(f"| {m} | {r['previous_fixed_sign']['same_new_objective']['objective']:.9f} | {r['independent_recompute']['objective']:.9f} | {r['same_new_objective_decrease']:.9f} | {r['changed_choice_count']} |")
    lines+=['','JSON에는 이전·현재의 고정 TRAIN 선택 T/R, anchor 유지·safe/unsafe 수, 두 축 회귀 MAE/RMSE/sign confusion을 보존했다. 회귀 출력은 혼합 손실로 학습한 signed-axis 점수이며 cm/degree나 보정된 확률이 아니다. Anchor 입력·예측이0이므로 nonanchor 지표를 별도로 제공한다. 이전 discrete anchored-target 일치율은 부가 진단이며 현재 회귀 목적함수가 아니다.','',
        '[전체 검산 JSON](TRAIN_CONVERGENCE.json) · [사전 입력·수학 검산](PREFIT_REVIEW_KO.md)','']
    return '\n'.join(lines)


def selfcheck():
    from . import convex_train as T
    assert T.metadata()==METADATA
    rng=np.random.default_rng(20261001271542)
    valid=np.array([[1,1,1,1],[0,1,0,0],[0,0,0,0]],bool);index=np.array([0,1,-1])
    x=rng.normal(size=(3,4,271));x[~valid]=0.;x[0,0]=0.;x[1,1]=0.
    target=rng.normal(size=(3,4,2));target[~valid]=0.;target[0,0]=0.;target[1,1]=0.;target[0,2,0]=0.
    w=rng.normal(size=(271,2))*.003
    actual=N.independent_objective(w,x,valid,target,True);native=T.objective(w,x,valid,target)
    for a,b in (('objective','value'),('Huber','Huber'),('Sign_logistic','Sign_logistic'),('L2_penalty','penalty')):
        assert abs(actual[a]-native[b])<1e-12
    np.testing.assert_allclose(actual['gradient'],native['gradient'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(actual['Hessian'],T.hessian(x,valid,native['residual'],target),rtol=1e-11,atol=1e-12)
    for j in range(542):
        step=np.zeros((271,2));step.flat[j]=1e-5
        diff=(N.independent_objective(w+step,x,valid,target)['objective']-N.independent_objective(w-step,x,valid,target)['objective'])/2e-5
        assert abs(diff-actual['gradient'].flat[j])<1e-7
    empty=N.independent_objective(w,np.zeros((1,4,271)),np.zeros((1,4),bool),np.zeros((1,4,2)),True)
    assert empty['Huber']==empty['Sign_logistic']==0.
    P.exact(empty['gradient'],1e-4*w);P.exact(empty['Hessian'],1e-4*np.eye(542))
    N.selfcheck()  # Pure old trace fault fixtures; no artifact loading or fit.
    assert not READS and OLD.READS is None
    print('SIGNED_AXES_DIRECTION_TRAIN_VERIFIER_SELFCHECK_PASS_NO_ARTIFACT_READS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('selfcheck','verify'))
    with threadpool_limits(limits=1):
        (selfcheck if parser.parse_args().stage=='selfcheck' else main)()
