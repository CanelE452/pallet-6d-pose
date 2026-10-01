"""Independent visual656x2 TRAIN verification; no fitting or VAL/real reference reads."""
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
from scripts.research.pallet_pose_signed_axes_asymmetric_20261001_v1 import convex_train as PREV
from scripts.research.pallet_pose_signed_axes_asymmetric_20261001_v1 import verify_train as QV
from scripts.research.pallet_pose_anchor_rbf_20261001_v1 import verify_train as V
from scripts.research.pallet_pose_union_selection_20261001_v1 import train as OLD
from scripts.research.pallet_pose_anchor_context_20261001_v1.train_risk_diagnostic import classify

OLD_HASH_KEYS=P.OLD_HASH_KEYS
HASH_KEYS=P.HASH_KEYS
METADATA=dict(QV.METADATA,feature_dim=656,previous_feature_dim=271,visual_dim=385,
    feature_map=QV.METADATA['feature_map']+'_native_dino385',
    input_rule='DIRECTION271_DIFFERENCE_PLUS_NORMALIZED_NATIVE_DINO385_DIFFERENCE',
    visual_rule=P.VISUAL_RULE,visual_normalization=P.VISUAL_NORMALIZATION)
READS=set()


def guard():
    sys.dont_write_bytecode=True
    arrays={C.PARENT_RAW/'SOURCE_FEATURES.npz',C.PARENT_RAW/'SOURCE_TRAIN_LABELS.npz',C.DIRECTION_RAW/'TRAIN_DIRECTIONS.npz',C.VISUAL_RAW/'TRAIN_APPEARANCE.npz'}
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
        if '/fits/' in name:assert path.is_relative_to(C.RAW/'fits') or path.is_relative_to(C.Q_RAW/'fits'),name
        if path.is_relative_to(C.ROOT):READS.add(str(path.relative_to(C.ROOT)))
    sys.addaudithook(hook)


# Independent Torch/autograd and weighted-design Hessian are dimension-generic;
# this frozen reference never invokes the production objective or its loader.
independent_objective=QV.independent_objective


def reconstruct(data,model):
    raw,valid,errors,direction,x253,x271,scaled,target,hashes=QV.reconstruct(data,model)
    experts=['R0']+([] if model=='R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
    visual=np.concatenate([data['visual'][m] for m in experts],axis=1)
    delta=P.appearance_difference(visual,valid,data['index'],data['visual_mean'],data['visual_std'])
    x=np.concatenate([x271,delta],axis=2)
    P.exact(x[:,:,:271],x271)
    for key in P.PREVIOUS_HASH_KEYS:assert hashes[key]==data['previous_asymmetric']['models'][model][key]
    hashes.update(visual_raw_sha=P.array_sha(visual),visual_difference_sha=P.array_sha(delta),visual_extended_input_sha=P.array_sha(x))
    return raw,valid,errors,direction,visual,x271,x,scaled,target,hashes


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
    visual_fields={key:data[key] for key in P.VISUAL_BINDING_KEYS}
    assert len(data['ids'])==2598 and data['ids'][index<0].tolist()==['TEX__shard_04_f0110']
    for obj in (protocol,complete,prefit):
        assert obj['direction_normalization_sha']==dnorm and obj['direction_receipt_binding']==db
        for key,value in visual_fields.items():assert obj[key]==value,(key,'top_level')
        actual_basis=obj['basis_SHA_bind'] if 'basis_SHA_bind' in obj else obj['inputs']['rbf_basis']
        assert actual_basis==basis_binding
    assert complete['anchor_index_sha']==prefit['anchor_index_sha']==P.array_sha(index)
    for key in ('visual_features','visual_receipt','visual_protocol','visual_verification','previous_asymmetric_prefit','previous_asymmetric_protocol',
                'direction_features','direction_receipt','direction_representation','direction_verification',
                'direction_protocol','previous_sign_prefit','previous_sign_protocol','previous_direction_prefit','previous_direction_protocol','source_metadata'):
        assert protocol['inputs'][key]==data['inputs'][key]
    previous_binding=C.bind(C.Q_DOC/'TRAIN_CONVERGENCE.json')
    assert protocol['evidence']['previous_asymmetric_training']==previous_binding
    previous=C.read(C.Q_DOC/'TRAIN_CONVERGENCE.json')
    assert previous['complete'] and previous['PASS'] and previous['source_TRAIN_only']
    # The authenticated prior receipt also lists still-older Newton weights.
    # Do not reopen those unrelated fits; verify the selected sign checkpoint
    # and its own START/trace/receipt chain below.
    previous_bindings={b['path']:b for b in previous['bindings']}
    bindings=[pb,C.bind(C.DOC/'TRAINING_COMPLETE.json'),C.bind(C.DOC/'PREFIT_REVIEW.json'),
        C.bind(C.Q_DOC/'TRAIN_CONVERGENCE.json'),basis_binding,db,
        C.bind(C.DIRECTION_DOC/'VERIFICATION.json'),C.bind(C.DIRECTION_DOC/'REPRESENTATION_AUDIT.json')]
    bindings.extend(data['inputs'][key] for key in ('visual_features','visual_receipt','visual_protocol','visual_verification'))
    for module in (T,C,P,N,V,PREV,QV,OLD):bindings.append(C.bind(module.__file__))
    bindings.extend([C.bind(__file__),C.bind(Path(classify.__code__.co_filename))])
    codes={b['path']:b for b in protocol['codes']}
    for module in (T,C,P,OLD,T.B,T.A,T.D):assert codes[C.bind(module.__file__)['path']]==C.bind(module.__file__)
    records={}
    for fb in complete['fits']:
        C.verify(fb);fit=C.read(C.ROOT/fb['path']);m=fit['model']
        assert m in C.MODEL_NAMES and m not in records and fb==C.bind(C.DOC/f'FIT_{m}.json')
        assert fit['complete'] and fit['protocol']==pb and fit['fits_executed']==1
        assert fit['source_TRAIN_only'] and not fit['VAL_quality_read'] and not fit['real_targets_read']
        for key in ('START','trace','checkpoint'):C.verify(fit[key])
        ck=C.read(C.ROOT/fit['checkpoint']['path']);start=C.read(C.ROOT/fit['START']['path'])
        assert fit['checkpoint']==C.bind(C.RAW/'fits'/m/'final.json')
        assert ck['schema']=='pallet_pose_signed_axes_visual_linear656x2_v1'
        assert ck['model']==start['model']==m and ck['protocol']==start['protocol']==pb
        for obj in (ck,start,fit):
            metadata_check(obj)
            assert obj['basis_SHA_bind']==basis_binding and obj['anchor_index_sha']==P.array_sha(index)
            assert obj['direction_normalization_sha']==dnorm and obj['direction_receipt_binding']==db
            for key,value in visual_fields.items():assert obj[key]==value,(m,key)
        assert ck['rbf_basis']==basis and ck['rbf_basis_binding']==basis_binding
        assert ck['bias']==0. and ck['lambda_l2']==1e-4 and ck['loss_uses_original_valid_mask'] is True
        assert ck['normalization']=='old_float32_then_float64'
        assert ck['normalization_sha']==fit['normalization_sha']==norm
        for key,expected in (('mean',data['mean']),('std',data['std']),('direction_mean',data['mean18']),('direction_std',data['std18']),
                             ('visual_mean',data['visual_mean']),('visual_std',data['visual_std'])):
            P.exact(np.asarray(ck[key],np.float32),expected)
        names=T.candidate_names(m);assert ck['names']==start['names']==names
        raw,valid,errors,direction,visual,x271,x,scaled,target,hashes=reconstruct(data,m)
        for key in HASH_KEYS:assert hashes[key]==ck[key]==fit[key]==start[key]==complete[key+'_by_model'][m]==prefit['models'][m][key]
        P.exact(x,T.difference_inputs(raw,valid,index,ck['mean'],ck['std'],basis,direction,ck['direction_mean'],ck['direction_std'],visual,ck['visual_mean'],ck['visual_std']))
        s2,y2=T.signed_targets(errors,valid,anchor,index,data['scale']);P.exact(scaled,s2);P.exact(target,y2)
        assert (~valid.any(1)).sum()==start['no_valid_rows']==1 and start['train_rows']==2598
        assert start['source_ids_sha']==P.array_sha(data['ids']) and start['anchor_errors_sha']==P.array_sha(anchor)
        assert ck['axis_order']==start['axis_order']==['translation_cm','rotation_deg']
        P.exact(np.asarray(ck['signed_target_scales']),data['scale']);P.exact(np.asarray(start['signed_target_scales']),data['scale'])
        w=np.asarray(ck['weight'],np.float64);assert w.shape==(656,2) and np.isfinite(w).all()
        assert P.array_sha(w)==fit['final_weight_sha'] and start['initial_weight_sha']==P.array_sha(np.zeros((656,2),np.float64))
        assert start['solver']==protocol['solver'] and start['certificate_rule']==protocol['certificate']
        actual=independent_objective(w,x,valid,target,with_hessian=True);native=T.objective(w,x,valid,target)
        np.testing.assert_allclose(actual['gradient'],native['gradient'],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(actual['prediction'],native['prediction'],rtol=1e-12,atol=1e-12)
        cert=ck['certificate'];assert cert==fit['certificate'] and cert['PASS'] and cert['optimizer_success']
        assert cert['lambda_l2']==1e-4 and cert['max_gap_upper_bound']==1e-6 and cert['gradient_linf_max']==1e-8
        assert cert['solver_rule']==METADATA['solver_rule'] and cert['loss_rule']==METADATA['loss_rule']
        assert cert['huber_delta']==1. and cert['output_dim']==2 and cert['sign_rule']==METADATA['sign_rule'] and cert['sign_coefficient']==1.
        for key,value in P.EXTRA_METADATA.items():assert cert[key]==value
        for key,ckey in (('objective','objective_value'),('Huber','Huber'),('Huber_symmetric','Huber_symmetric'),('Huber_underprediction','Huber_underprediction'),('Sign_logistic','Sign_logistic'),('L2_penalty','L2_penalty'),('gradient_l2','gradient_l2'),('gradient_linf','gradient_linf')):
            assert abs(actual[key]-cert[ckey])<=1e-10,(m,key)
        assert actual['gradient_linf']<=1e-8 and actual['certified_gap_upper_bound']<=1e-6
        assert abs(actual['certified_gap_upper_bound']-cert['gradient_l2_squared_over_2lambda'])<=1e-12
        H=T.hessian(x,valid,native['residual'],target)
        np.testing.assert_allclose(actual['Hessian'],H,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(actual['Hessian'],actual['Hessian'].T,rtol=1e-12,atol=1e-12)
        eigen=np.linalg.eigvalsh(actual['Hessian']);assert eigen[0]>=1e-4-1e-10
        np.testing.assert_allclose([eigen[0],eigen[-1]],[ck['solver']['final_hessian_min'],ck['solver']['final_hessian_max']],rtol=1e-8,atol=1e-10)
        runtime_axes=T.predict_axes(ck,raw,valid,index,direction,visual);runtime=T.score_candidates(ck,raw,valid,index,direction,visual)
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
        for row in trace:
            assert abs(row['Huber']-row['Huber_symmetric']-row['Huber_underprediction'])<=1e-12
            assert row['Huber_symmetric']>=0. and row['Huber_underprediction']>=0.
        for key in ('Huber_symmetric','Huber_underprediction'):
            assert calls[-1][key]==cert[key]
        by_call={r['call']:r for r in calls}
        for row in iterations:
            for key in ('Huber_symmetric','Huber_underprediction'):
                assert row[key]==by_call[row['accepted_call']][key]
        assert complete['objective_components']==['Huber_symmetric','Huber_underprediction','Sign_logistic','L2_penalty']
        assert complete['Huber_definition']=='Huber_symmetric + Huber_underprediction'
        for key,value in complete['final_objective_components_by_model'][m].items():assert cert[key]==value
        zero=independent_objective(np.zeros((656,2)),x,valid,target)
        assert abs(calls[0]['objective']-zero['objective'])<=1e-12
        for key in ('Huber','Huber_symmetric','Huber_underprediction','Sign_logistic','L2_penalty'):
            assert abs(calls[0][key]-zero[key])<=1e-12
        for row in trace:
            metadata_check(row)
            for key in HASH_KEYS:assert row[key]==hashes[key]
            assert row['basis_SHA_bind']==basis_binding and row['anchor_index_sha']==P.array_sha(index)
            assert row['direction_normalization_sha']==dnorm and row['direction_receipt_binding']==db
            for key,value in visual_fields.items():assert row[key]==value,(m,key)
        oldfb=C.bind(C.Q_DOC/f'FIT_{m}.json');assert previous_bindings[oldfb['path']]==oldfb
        oldfit=C.read(C.ROOT/oldfb['path']);C.verify(oldfit['checkpoint'])
        for key in ('START','trace'):
            assert previous_bindings[oldfit[key]['path']]==oldfit[key]
            C.verify(oldfit[key])
        assert previous_bindings[oldfit['checkpoint']['path']]==oldfit['checkpoint']
        oldck=C.read(C.ROOT/oldfit['checkpoint']['path'])
        assert oldck['schema']=='pallet_pose_signed_axes_asymmetric_linear271x2_v1'
        assert oldck['rbf_basis']==basis and oldck['basis_SHA_bind']==basis_binding and oldck['names']==names and oldck['certificate']['PASS']
        for key in ('mean','std'):P.exact(np.asarray(oldck[key],np.float32),data[key])
        for key in P.PREVIOUS_HASH_KEYS:assert hashes[key]==oldck[key]==previous['models'][m]['hashes'][key]
        ow=np.asarray(oldck['weight'],np.float64);assert ow.shape==(271,2)
        for key in ('direction_mean','direction_std'):
            P.exact(np.asarray(oldck[key],np.float32),np.asarray(ck[key],np.float32))
        assert oldck['direction_receipt_binding']==db and oldck['direction_normalization_sha']==dnorm
        padded=np.vstack([ow,np.zeros((385,2),np.float64)])
        oldnative=independent_objective(ow,x271,valid,target)
        oldnew=independent_objective(padded,x,valid,target)
        nativeold=PREV.objective(ow,x271,valid,target)
        np.testing.assert_allclose(oldnative['gradient'],nativeold['gradient'],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(oldnew['prediction'],oldnative['prediction'],rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(oldnew['gradient'][:271],oldnative['gradient'],rtol=1e-9,atol=1e-10)
        for key,ckey in (('objective','objective_value'),('Huber','Huber'),('Huber_symmetric','Huber_symmetric'),
                        ('Huber_underprediction','Huber_underprediction'),('Sign_logistic','Sign_logistic'),
                        ('L2_penalty','L2_penalty'),('gradient_l2','gradient_l2'),('gradient_linf','gradient_linf')):
            assert abs(oldnative[key]-oldck['certificate'][ckey])<=1e-10
        for key in ('objective','Huber','Huber_symmetric','Huber_underprediction','Sign_logistic','L2_penalty'):
            assert abs(oldnew[key]-oldnative[key])<=1e-10
        comparison_roundoff=1e-10
        assert actual['objective']<=oldnew['objective']+actual['certified_gap_upper_bound']+comparison_roundoff
        oldaxes=np.einsum('nkd,da->nka',x271,ow);oldscore=np.where(valid,oldaxes.max(2),np.inf)
        oldruntime=PREV.score_candidates(oldck,raw,valid,index,direction);P.exact(oldruntime,oldscore)
        oldstats,oldchosen=V.score_statistics(oldruntime,valid,legacy,safe,errors,anchor,names)
        oldrisk=classify(oldchosen,valid,errors,anchor,index,data['scale'])
        assert oldstats==previous['models'][m]['statistics'] and oldrisk==previous['models'][m]['risk_statistics']
        records[m]=dict(independent_recompute=N.public_objective(actual),original_certificate=cert,
            objective_abs_difference=abs(actual['objective']-cert['objective_value']),
            gradient_vector_max_abs_difference=float(np.abs(actual['gradient']-native['gradient']).max()),
            Hessian_max_abs_difference=float(np.abs(actual['Hessian']-H).max()),
            independent_Hessian_min_eigenvalue=float(eigen[0]),independent_Hessian_max_eigenvalue=float(eigen[-1]),
            Hessian_scope='1312x1312 C-order blocks; symmetric Huber curvature at abs(e)<1 plus additional curvature at -1<e<0, zero extra curvature at e=0/-1, nonzero-target logistic curvature and ridge remain.',
            objective_calls=len(calls),iterations=len(iterations),newton_trace=tracecheck,
            statistics=stats,risk_statistics=risk,regression_statistics=N.regression_statistics(runtime_axes,target,valid,index,chosen),
            legacy_discrete_target_accuracy_scope='Secondary anchored-target diagnostic, not regression supervision.',
            previous_fixed_asymmetric=dict(checkpoint=oldfit['checkpoint'],receipt=oldfb,statistics=oldstats,risk_statistics=oldrisk,
                original_choice_replay_PASS=True,same_new_objective=N.public_objective(oldnew),
                original_native_objective=N.public_objective(oldnative),original_native_certificate=oldck['certificate'],
                original_objective_and_native271_gradient_parity=True,old271_prefix_byte_exact=True,
                embedded_prediction_max_abs_difference=float(np.abs(oldnew['prediction']-oldnative['prediction']).max()),
                embedding_prediction_tolerance=dict(atol=1e-11,rtol=1e-11),
                expanded656_gradient_not_compared_to_original271_certificate=True,
                additional385_gradient_l2=float(np.linalg.norm(oldnew['gradient'][271:])),
                regression_statistics=N.regression_statistics(oldaxes,target,valid,index,oldchosen),
                comparison_scope='Same unchanged asymmetric Huber+sign+ridge at previous certified Q271x2 weights embedded with zero385x2. Native271 certificate checked only against native271 gradient; expanded656 gradient is separate. No warm start.'),
            same_new_objective_decrease=oldnew['objective']-actual['objective'],
            feasible_old_weight_consistency=dict(PASS=True,current_gap_allowance=actual['certified_gap_upper_bound'],roundoff_allowance=comparison_roundoff,performance_gate=False),
            changed_choice_count=int((chosen!=oldchosen).sum()),
            basis_SHA_bind=basis_binding,direction_receipt_binding=db,direction_normalization_sha=dnorm,
            **visual_fields,hashes=hashes,runtime_axes_sha=P.array_sha(runtime_axes),runtime_axis_and_selection_parity=True,all_invalid_frame_retained=True)
        bindings += [fb,*(fit[k] for k in ('START','trace','checkpoint')),oldfb,oldfit['checkpoint']]
        print('SIGNED_AXES_VISUAL_TRAIN_VERIFIED',m,'gap',actual['certified_gap_upper_bound'],flush=True)
    assert set(records)==set(C.MODEL_NAMES)
    assert sum(r['objective_calls'] for r in records.values())==complete['total_objective_calls']
    assert sum(r['iterations'] for r in records.values())==complete['total_iterations']
    out=dict(complete=True,PASS=True,independent_check_PASS=True,created_at=C.now(),protocol=pb,**METADATA,
        models=records,bindings=bindings,basis_SHA_bind=basis_binding,direction_receipt_binding=db,direction_normalization_sha=dnorm,
        **visual_fields,
        source_TRAIN_only=True,frames=2598,available_anchor_rows=2597,failed_rows=1,
        VAL_quality_read=False,real_targets_read=False,raw_source_reference_reads=0,new_fits=0,optimizer_steps=0,
        alternative_policy_probes=0,method_success=False,goal_complete=False,previous_weight_warmstart=False,
        comparison='Same fixed Q asymmetric Huber+sign+ridge at current656 and prior certified271 weights padded with385 zero rows; native271 certificate and expanded656 gradient kept separate.',
        objective_definition='Per-frame original valid candidate/two-axis Huber(e)+Huber(min(e,0))+nonzero-target sign logistic mean, then full2598 mean, ridge1e-4 over1312 weights.',
        validation='Independent targets and271prefix, row-wise FP32 native385 normalization then FP64 R0-anchor subtraction, Torch64 autograd of all1312 weights and independent weighted-design1312 Hessian.',
        read_paths=sorted(READS))
    C.save(C.DOC/'TRAIN_CONVERGENCE_KO.md',report(out));C.save(C.DOC/'TRAIN_CONVERGENCE.json',out)
    print('SIGNED_AXES_VISUAL_INDEPENDENT_TRAIN_VERIFICATION_PASS_ALL4',flush=True)


def report(x):
    lines=['# Native385 추가의 독립 TRAIN 검산','',
        '**독립 수치 검산 PASS는 고정 목적식의 수렴·기록 확인이며 T/R 일반화 성공 판정이 아니다.** 이전 Q의271 prefix·9해시·target·TRAIN2,598행(유효2,597·실패1)을 유지하고 고정 native385를 추가했다.','',
        'Huber=Huber_symmetric+Huber_underprediction이며 J=Huber+Sign_logistic+L2이다. Q와 같은 목적식·전체2,598행 분모·원래 valid·λ1e−4·Newton/Armijo 상한이다. 모든1,312개 계수에 ridge를 적용한다.','',
        '독립 Torch64 autograd와1,312×1,312 가중 design Hessian을 비교했다. 초기/최종 값과 모든 Armijo trial 기록을 확인했다. 중간 weight는 저장되지 않아 중간 Newton 방향 전수 재계산을 주장하지 않는다.','',
        '| 모델 | 대칭 Huber | 추가 Huber | sign logistic | J | gap 상한 | 호출 |','|---|---:|---:|---:|---:|---:|---:|']
    for m,r in x['models'].items():
        q=r['independent_recompute'];lines.append(f"| {m} | {q['Huber_symmetric']:.9f} | {q['Huber_underprediction']:.9f} | {q['Sign_logistic']:.9f} | {q['objective']:.9f} | {q['certified_gap_upper_bound']:.3g} | {r['objective_calls']} |")
    lines+=['','## 같은 목적식에서 이전 Q와 비교','',
        '이전271×2 가중치 뒤에0의385행을 붙여 같은656입력·같은 목적식으로 평가했다. 이전 native271 인증 gradient는271공간에서만 재확인하며 추가385 gradient는 별도다. padding weight는 비교용이고 새 학습 초기값은 모두0이다.','',
        '| 모델 | 이전 Q의 동일 J | 현재656 J | 감소 | 실제 선택 변경 |','|---|---:|---:|---:|---:|']
    for m,r in x['models'].items():
        lines.append(f"| {m} | {r['previous_fixed_asymmetric']['same_new_objective']['objective']:.9f} | {r['independent_recompute']['objective']:.9f} | {r['same_new_objective_decrease']:.9f} | {r['changed_choice_count']} |")
    lines+=['','JSON에 고정 TRAIN 선택 T/R·anchor·safe/unsafe·실패와 회귀 MAE/RMSE/sign confusion을 기록했다. 출력은 학습된 signed-axis 점수이며 cm/degree 또는 보정된 확률이 아니다. target 일치율은 과거 discrete target의 부가 진단이며 현재 supervision은 signed 두 축이다.','',
        '새 fit·VAL/실사 참조·정책 탐색0. T/R 성공은 별도 봉인된 source45와 원래+matched 실사5기준을 모두 확인해야 한다.','',
        '[전체 검산 JSON](TRAIN_CONVERGENCE.json) · [사전 검산](PREFIT_REVIEW_KO.md)','']
    return '\n'.join(lines)


def selfcheck():
    from . import convex_train as T
    assert T.metadata()==METADATA
    rng=np.random.default_rng(202610016561312)
    valid=np.array([[1,1,1,1],[0,1,0,0],[0,0,0,0]],bool)
    x=rng.normal(size=(3,4,656));x[~valid]=0.;x[0,0]=0.;x[1,1]=0.
    target=rng.normal(size=(3,4,2));target[~valid]=0.;target[0,0]=0.;target[1,1]=0.;target[0,2,0]=0.
    w=rng.normal(size=(656,2))*.001
    actual=independent_objective(w,x,valid,target,True);native=T.objective(w,x,valid,target)
    for a,b in (('objective','value'),('Huber','Huber'),('Huber_symmetric','Huber_symmetric'),('Huber_underprediction','Huber_underprediction'),('Sign_logistic','Sign_logistic'),('L2_penalty','penalty')):
        assert abs(actual[a]-native[b])<1e-12
    np.testing.assert_allclose(actual['gradient'],native['gradient'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(actual['Hessian'],T.hessian(x,valid,native['residual'],target),rtol=1e-11,atol=1e-12)
    for j in np.r_[np.linspace(0,541,8,dtype=int),np.linspace(542,1311,24,dtype=int)]:
        step=np.zeros((656,2));step.flat[j]=1e-5
        diff=(independent_objective(w+step,x,valid,target)['objective']-independent_objective(w-step,x,valid,target)['objective'])/2e-5
        assert abs(diff-actual['gradient'].flat[j])<1e-7
    empty=independent_objective(w,np.zeros((1,4,656)),np.zeros((1,4),bool),np.zeros((1,4,2)),True)
    assert empty['Huber']==empty['Sign_logistic']==0.
    P.exact(empty['gradient'],1e-4*w);P.exact(empty['Hessian'],1e-4*np.eye(1312))
    assert not READS and OLD.READS is None
    print('SIGNED_AXES_VISUAL_TRAIN_VERIFIER_SELFCHECK_PASS_NO_ARTIFACT_READS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('selfcheck','verify'))
    with threadpool_limits(limits=1):
        (selfcheck if parser.parse_args().stage=='selfcheck' else main)()
