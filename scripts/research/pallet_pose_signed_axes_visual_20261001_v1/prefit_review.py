"""Independent native385 extension PREFIT; no fitted weights or quality probes."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from . import common as C
from scripts.research.pallet_pose_signed_axes_asymmetric_20261001_v1 import prefit_review as Q
from scripts.research.pallet_pose_signed_axes_direction_20261001_v1 import prefit_review as P
from scripts.research.pallet_pose_signed_axes_sign_20261001_v1 import prefit_review as N

OLD_HASH_KEYS=P.OLD_HASH_KEYS
EXTRA_HASH_KEYS=P.EXTRA_HASH_KEYS
PREVIOUS_HASH_KEYS=OLD_HASH_KEYS+EXTRA_HASH_KEYS
VISUAL_HASH_KEYS=('visual_raw_sha','visual_difference_sha','visual_extended_input_sha')
HASH_KEYS=PREVIOUS_HASH_KEYS+VISUAL_HASH_KEYS
VISUAL_RULE='NATIVE_UNPADDED_SUPPORTED_PROJECTED_CORNERS8_DINO384_MEAN_PLUS_SUPPORT_FRACTION'
VISUAL_NORMALIZATION='R0_TRAIN_valid_float32_mean_std_floor_1e-6_then_float64_candidate_minus_R0_anchor'
VISUAL_BINDING_KEYS=('visual_normalization_sha','visual_receipt_binding','visual_protocol_binding','visual_verification_binding')
LOSS_RULE=Q.LOSS_RULE
EXTRA_METADATA=Q.EXTRA_METADATA
DIRECTION_RULE=Q.DIRECTION_RULE
DIRECTION_NORMALIZATION=Q.DIRECTION_NORMALIZATION
array_sha,exact,direction_difference=Q.array_sha,Q.exact,Q.direction_difference
independent_objective,independent_hessian,torch_loss=Q.independent_objective,Q.independent_hessian,Q.torch_loss
READS=set()


def guard():
    sys.dont_write_bytecode=True
    arrays={C.PARENT_RAW/'SOURCE_FEATURES.npz',C.PARENT_RAW/'SOURCE_TRAIN_LABELS.npz',
            C.DIRECTION_RAW/'TRAIN_DIRECTIONS.npz',C.VISUAL_RAW/'TRAIN_APPEARANCE.npz'}
    outputs={C.DOC/'PREFIT_REVIEW.json',C.DOC/'PREFIT_REVIEW_KO.md'}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();name=str(path);mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(
            isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing and path.is_relative_to(C.ROOT):assert path in outputs,('PREFIT_OUTPUT_ONLY',name);return
        assert not any(t in name for t in ('/data/evaluation/','/real_gt_v2/','/annotations/',
            'GEOMETRY_RESOLVED_POSE_GT','GEOMETRY_SIDETABLE','SYNTH_RECORDS','SYNTH_LABELS',
            'SOURCE_VAL_','REAL_RESULTS','REAL_CHOICES','POSE_METRICS','TRUTH_FOR_DISPLAY',
            '/fits/','/model_parameters/','TRAIN_CONVERGENCE','TRAIN_TOKENS')),('NO_WEIGHTS_OR_QUALITY',name)
        assert path.suffix.lower() not in ('.png','.jpg','.jpeg','.pt','.pth','.onnx'),name
        if path.suffix=='.npz':assert path in arrays,('TRAIN_ARRAYS_ONLY',name)
        if path.is_relative_to(C.ROOT):READS.add(str(path.relative_to(C.ROOT)))
    sys.addaudithook(hook)


def appearance_difference(raw,valid,index,mean,std):
    """Independent row/candidate path: float32 normalize, then float64 subtract."""
    raw,valid,index=np.asarray(raw),np.asarray(valid),np.asarray(index)
    mean,std=np.asarray(mean,np.float32),np.asarray(std,np.float32)
    assert raw.dtype==np.float32 and raw.shape==(*valid.shape,385) and valid.dtype==bool
    assert valid.ndim==2 and valid.shape[1] in (2,4) and index.shape==(len(valid),)
    assert np.issubdtype(index.dtype,np.integer) and np.isin(index,[-1,0,1]).all()
    assert np.array_equal(index>=0,valid.any(1))
    assert mean.shape==std.shape==(385,) and np.isfinite(mean).all() and np.isfinite(std).all()
    assert (std>=np.float32(1e-6)).all() and np.isfinite(raw[valid]).all()
    result=np.zeros(raw.shape,np.float64)
    for i in range(len(valid)):
        if not valid[i].any():continue
        a=int(index[i]);assert a in (0,1) and valid[i,a]
        reference=np.asarray((raw[i,a]-mean)/std,np.float32).astype(np.float64)
        for j in np.flatnonzero(valid[i]):
            z=np.asarray((raw[i,j]-mean)/std,np.float32).astype(np.float64)
            result[i,j]=z-reference
        assert not result[i,a].any()
    assert np.isfinite(result).all() and not result[~valid].any()
    return result


def code_parity(T,operators=True):
    old=ast.parse((C.Q_HERE/'convex_train.py').read_text());new=ast.parse(Path(T.__file__).read_text())
    names=('objective','hessian','newton_solve','certificate','solver_config','signed_targets',
           'validate_anchor','difference_from_context')
    for name in names:
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),('UNCHANGED_MATH_AST',name)
    gates={}
    if operators:
        for filename,funcs in [('evaluate_source.py',('comparisons_for','chosen_pose','verify_source_reference_chain')),
                               ('evaluate_real.py',('gates_for','combined_gates'))]:
            a=ast.parse((C.Q_HERE/filename).read_text());b=ast.parse((C.HERE/filename).read_text())
            for name in funcs:
                fa=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name)
                fb=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==name)
                assert ast.dump(fa)==ast.dump(fb),(filename,name)
            gates[filename]=dict(functions=list(funcs),previous=C.bind(C.Q_HERE/filename),current=C.bind(C.HERE/filename))
    assert T.LOSS_RULE==LOSS_RULE and T.FEATURE_DIM==656 and T.ALL_HASH_KEYS==HASH_KEYS
    assert T.PREVIOUS_FEATURE_DIM==271 and T.VISUAL_DIM==385
    assert T.VISUAL_RULE==VISUAL_RULE and T.VISUAL_NORMALIZATION==VISUAL_NORMALIZATION
    assert T.LAMBDA==1e-4 and T.MAX_ITER==1000 and T.MAX_CALLS==2000 and T.GAP_MAX==1e-6
    for key,value in EXTRA_METADATA.items():assert T.metadata()[key]==value
    return dict(PASS=True,objective_target_solver_certificate_AST_exact=list(names),evaluation_AST_parity=gates,
        previous_code=C.bind(C.Q_HERE/'convex_train.py'),current_code=C.bind(T.__file__),
        changed_factor='Only append fixed normalized native DINO385 candidate-minus-R0-anchor input to the unchanged271 prefix; target/loss/Newton/certificate/gates preserved.')


def synthetic(T):
    from scripts.research.pallet_pose_signed_axes_asymmetric_20261001_v1 import convex_train as OLD
    rng=np.random.default_rng(202610016561312)
    valid=np.array([[1,1,1,1],[0,1,1,0],[1,0,0,0],[0,0,0,0]],bool)
    index=np.array([0,1,0,-1]);rows=np.flatnonzero(valid.any(1))
    raw=rng.normal(size=(4,4,94)).astype(np.float32);raw[~valid]=np.nan
    direction=rng.normal(size=(4,4,18)).astype(np.float32);direction[~valid]=0.
    visual=rng.normal(size=(4,4,385)).astype(np.float32);visual[~valid]=0.
    mean=np.zeros(94,np.float32);std=np.ones(94,np.float32)
    mean18=np.zeros(18,np.float32);std18=np.ones(18,np.float32)
    mv=rng.normal(size=385).astype(np.float32);sv=np.exp(rng.normal(size=385)).astype(np.float32)
    centers=rng.normal(size=(64,189));width=float(np.median([np.sum((centers[i]-centers[j])**2) for i in range(64) for j in range(i+1,64)]))
    basis=dict(schema='pallet_pose_anchor_rbf_runtime_v1',context_dim=189,rbf_dim=64,centers=centers.tolist(),bandwidth_squared=width,normalization_sha=array_sha(np.stack([mean,std])))
    _,_,_,x271=P.extended_inputs(raw,valid,index,mean,std,basis,direction,mean18,std18)
    extra=appearance_difference(visual,valid,index,mv,sv);x=np.concatenate([x271,extra],axis=2)
    exact(x,T.difference_inputs(raw,valid,index,mean,std,basis,direction,mean18,std18,visual,mv,sv))
    exact(x[:,:,:271],OLD.difference_inputs(raw,valid,index,mean,std,basis,direction,mean18,std18))
    exact(extra,T.appearance_difference(visual,valid,index,mv,sv))
    assert not x[~valid].any() and not x[rows,index[rows]].any()
    # The last expert references the first R0 slots, never its own W/D index.
    np.testing.assert_array_equal(extra[0,2],((visual[0,2]-mv)/sv).astype(np.float64)-((visual[0,0]-mv)/sv).astype(np.float64))
    invalid_visual=visual.copy();invalid_visual[~valid]=np.nan
    exact(extra,appearance_difference(invalid_visual,valid,index,mv,sv))
    target=rng.normal(size=(*valid.shape,2));target[~valid]=0.;target[rows,index[rows]]=0.;target[0,2,0]=0.
    W=rng.normal(size=(656,2))*.001
    actual=T.objective(W,x,valid,target);reference=independent_objective(W,x,valid,target)
    for key in ('value','Huber','Huber_symmetric','Huber_underprediction','Sign_logistic','penalty','gradient','prediction','residual'):
        np.testing.assert_allclose(actual[key],reference[key],rtol=1e-10,atol=1e-12)
    H=independent_hessian(x,valid,reference['residual'],target)
    np.testing.assert_allclose(H,T.hessian(x,valid,actual['residual'],target),rtol=1e-10,atol=1e-12)
    xx=torch.tensor(x,dtype=torch.float64);vv=torch.tensor(valid);yy=torch.tensor(target,dtype=torch.float64)
    w=torch.tensor(W.ravel(),dtype=torch.float64,requires_grad=True)
    loss=lambda flat:torch_loss(flat.reshape(656,2),xx,vv,yy)
    value=loss(w);gradient,=torch.autograd.grad(value,w)
    hessian=torch.autograd.functional.hessian(loss,w).detach().numpy()
    np.testing.assert_allclose(float(value.detach()),actual['value'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(gradient.detach().numpy().reshape(656,2),actual['gradient'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(hessian,H,rtol=1e-10,atol=1e-12)
    coordinates=np.r_[np.linspace(0,541,8,dtype=int),np.linspace(542,1311,24,dtype=int)]
    assert len(set(coordinates))==32 and set(coordinates%2)=={0,1}
    gaps=[];hgaps=[];eps=1e-6
    for j in coordinates:
        step=np.zeros_like(W);step.flat[j]=eps
        plus=independent_objective(W+step,x,valid,target);minus=independent_objective(W-step,x,valid,target)
        gaps.append(abs((plus['value']-minus['value'])/(2*eps)-actual['gradient'].flat[j]))
        hgaps.append(float(np.max(abs((plus['gradient']-minus['gradient']).ravel()/(2*eps)-H[:,j]))))
    assert max(gaps)<1e-8 and max(hgaps)<1e-8
    oldW=rng.normal(size=(271,2))*.002;padded=np.vstack([oldW,np.zeros((385,2))])
    old=OLD.objective(oldW,x271,valid,target);embedded=T.objective(padded,x,valid,target)
    for key in ('value','Huber','Huber_symmetric','Huber_underprediction','Sign_logistic','penalty','prediction'):
        np.testing.assert_allclose(old[key],embedded[key],rtol=1e-11,atol=1e-11)
    common=dict(normalization='old_float32_then_float64',names=T.candidate_names('UNION_s1'),bias=0.,lambda_l2=1e-4,
        mean=mean.tolist(),std=std.tolist(),rbf_basis=basis,rbf_basis_binding={'invented':True},basis_SHA_bind={'invented':True},
        direction_mean=mean18.tolist(),direction_std=std18.tolist(),direction_normalization_sha=array_sha(np.stack([mean18,std18])),direction_receipt_binding={'invented':True})
    oldck=dict(common,schema=OLD.CHECKPOINT_SCHEMA,**OLD.metadata(),weight=oldW.tolist())
    ck=dict(common,schema=T.CHECKPOINT_SCHEMA,**T.metadata(),weight=padded.tolist(),visual_mean=mv.tolist(),visual_std=sv.tolist(),
        visual_normalization_sha=array_sha(np.stack([mv,sv])),visual_receipt_binding={'invented':True},
        visual_protocol_binding={'invented':True},visual_verification_binding={'invented':True})
    original=T.signed_targets
    def denied(*args,**kwargs):raise AssertionError('NO_RUNTIME_TARGET')
    try:
        T.signed_targets=denied
        oldscore=OLD.score_candidates(oldck,raw,valid,index,direction)
        newscore=T.score_candidates(ck,raw,valid,index,direction,visual)
        np.testing.assert_allclose(oldscore,newscore,rtol=1e-11,atol=1e-11)
    finally:T.signed_targets=original
    empty=T.objective(W,np.zeros((1,4,656)),np.zeros((1,4),bool),np.zeros((1,4,2)))
    assert empty['Huber']==empty['Sign_logistic']==0.;exact(empty['gradient'],1e-4*W)
    return dict(PASS=True,invented_only=True,old271_prefix_byte_exact=True,FP32_normalize_before_FP64_difference=True,
        reference_anchor_is_R0=True,all_invalid_and_anchor_zero=True,old_weight_zero385_embedding_atol=1e-11,
        old_weight_zero385_embedding_rtol=1e-11,embedding_prediction_max_difference=float(np.max(abs(old['prediction']-embedded['prediction']))),
        independent_scalar_and_Torch_PASS=True,gradient_parameters_checked=1312,Hessian_shape=[1312,1312],
        finite_difference_coordinates=coordinates.tolist(),gradient_finite_difference_max=max(gaps),Hessian_finite_difference_max=max(hgaps),
        Torch_Hessian_max_difference=float(np.max(abs(hessian-H))),runtime_target_disabled_PASS=True,actual_inputs_or_weights_read=0)


def load_cached_inputs():
    data=Q.load_cached_inputs()
    C.verify(C.read(C.Q_DOC/'TRAIN_PROTOCOL_SHA.json'))
    qp=C.read(C.Q_DOC/'TRAIN_PROTOCOL.json');C.verify(qp['inputs']['prefit_review'])
    previous=C.read(C.ROOT/qp['inputs']['prefit_review']['path'])
    assert previous['complete'] and previous['PASS'] and previous['trainer']==C.bind(C.Q_HERE/'convex_train.py')
    for key,value in data['inputs'].items():assert previous['inputs'][key]==value,('Q_INPUTS_CHANGED',key)
    inputs=dict(data['inputs'],previous_asymmetric_prefit=qp['inputs']['prefit_review'],previous_asymmetric_protocol=C.bind(C.Q_DOC/'TRAIN_PROTOCOL.json'))
    for key,path in dict(visual_features=C.VISUAL_RAW/'TRAIN_APPEARANCE.npz',visual_receipt=C.VISUAL_DOC/'TRAIN_APPEARANCE_INPUTS.json',
                         visual_protocol=C.VISUAL_DOC/'INPUT_PROTOCOL.json',visual_verification=C.VISUAL_DOC/'INPUT_VERIFICATION.json').items():
        inputs[key]=C.bind(path)
    receipt=C.read(C.ROOT/inputs['visual_receipt']['path']);protocol=C.read(C.ROOT/inputs['visual_protocol']['path'])
    checked=C.read(C.ROOT/inputs['visual_verification']['path'])
    assert receipt['complete'] and receipt['input_construction_pass'] and receipt['source_TRAIN_only']
    assert checked['complete'] and checked['PASS'] and checked['verification_PASS'] and checked['source_TRAIN_only']
    assert receipt['protocol']==checked['protocol']==inputs['visual_protocol']
    assert receipt['descriptors']==checked['descriptors']==inputs['visual_features']
    assert checked['public_input_receipt']==inputs['visual_receipt']
    assert checked['input_receipt']['sha256']==inputs['visual_receipt']['sha256']
    assert receipt['descriptor_rule']==protocol['descriptor_rule']==VISUAL_RULE
    assert protocol['train_only'] and protocol['no_fit'] and protocol['no_label_values'] and protocol['output_dim']==385
    assert receipt['frames']==checked['frames']==2598 and checked['all_invalid_rows']==1
    assert receipt['original_candidate_validity_preserved'] and checked['original_candidate_validity_unchanged']
    assert not receipt['source_label_values_read'] and not receipt['source_VAL_features_extracted'] and not receipt['real_features_extracted']
    assert receipt['fits_executed']==receipt['new_image_forwards']==receipt['new_PnP_solves']==0
    assert checked['normalization_exact'] and checked['original_R0_anchor_identity_unchanged']
    with np.load(C.ROOT/inputs['visual_features']['path'],allow_pickle=False) as z:
        keys={'ids','source_index','anchor_index','crop_matrices','mean385','std385'}|{m+s for m in P.EXPERTS for s in ('_appearance385','_valid','_support8')}
        assert set(z.files)==keys
        for key,value in [('ids',data['ids']),('source_index',data['source_index']),('anchor_index',data['index'])]:exact(z[key],value)
        for key in z.files:assert array_sha(z[key])==checked['main_array_hashes'][key]
        visual={m:z[m+'_appearance385'].copy() for m in P.EXPERTS}
        for m in P.EXPERTS:
            exact(z[m+'_valid'],data['valid'][m])
            assert z[m+'_support8'].dtype==bool and z[m+'_support8'].shape==(2598,2,8)
            exact(visual[m][:,:,-1],z[m+'_support8'].sum(2).astype(np.float32)/8)
        stored_mean,stored_std=z['mean385'].copy(),z['std385'].copy()
    for m in P.EXPERTS:
        a=visual[m];valid=data['valid'][m]
        assert a.shape==(2598,2,385) and a.dtype==np.float32 and np.isfinite(a).all() and not a[~valid].any()
        assert array_sha(a)==receipt['models'][m]['descriptor_sha'] and array_sha(valid)==receipt['models'][m]['valid_sha']
    values=visual['R0'][data['valid']['R0']];assert values.shape==(5194,385)
    mean,std=values.mean(0),np.maximum(values.std(0),np.float32(1e-6));norm=array_sha(np.stack([mean,std]))
    exact(stored_mean,mean);exact(stored_std,std)
    n=receipt['normalization'];assert n['source']=='R0_original_valid_TRAIN_candidates' and n['count']==5194 and n['valid_zero_support_candidates_included']
    assert n['dtype']=='float32' and n['std_floor']==1e-6 and n['array_sha']==norm==checked['normalization']['array_sha']
    exact(np.asarray(n['mean385'],np.float32),mean);exact(np.asarray(n['std385'],np.float32),std)
    data.update(previous_asymmetric=previous,visual=visual,visual_mean=mean,visual_std=std,
        visual_normalization_sha=norm,visual_receipt_binding=inputs['visual_receipt'],
        visual_protocol_binding=inputs['visual_protocol'],visual_verification_binding=inputs['visual_verification'],inputs=inputs)
    return data


def actual_review(T):
    assert not (C.RAW/'fits').exists(),'PREFIT must precede all four new fits'
    data=load_cached_inputs();records={}
    for model in C.MODEL_NAMES:
        experts=['R0']+([] if model=='R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
        raw,valid,errors=[np.concatenate([data[key][m] for m in experts],axis=1) for key in ('features','valid','errors')]
        direction=np.concatenate([data['direction'][m] for m in experts],axis=1)
        visual=np.concatenate([data['visual'][m] for m in experts],axis=1)
        phi,base,extra,x271=P.extended_inputs(raw,valid,data['index'],data['mean'],data['std'],data['basis'],direction,data['mean18'],data['std18'])
        visual_delta=appearance_difference(visual,valid,data['index'],data['visual_mean'],data['visual_std'])
        x=np.concatenate([x271,visual_delta],axis=2);exact(x[:,:,:271],x271)
        exact(x,T.difference_inputs(raw,valid,data['index'],data['mean'],data['std'],data['basis'],direction,data['mean18'],data['std18'],visual,data['visual_mean'],data['visual_std']))
        with np.errstate(all='raise'):scaled,target=N.signed_targets(errors,valid,data['anchor'],data['index'],data['scale'])
        s2,y2=T.signed_targets(errors,valid,data['anchor'],data['index'],data['scale']);exact(scaled,s2);exact(target,y2)
        hashes=dict(signed_target_sha=array_sha(target),input_difference_sha=array_sha(base),base_context_sha=array_sha(phi),errors_sha=array_sha(errors),
            scaled_excess_sha=array_sha(scaled),original_valid_sha=array_sha(valid),direction_raw_sha=array_sha(direction),direction_difference_sha=array_sha(extra),extended_input_sha=array_sha(x271),
            visual_raw_sha=array_sha(visual),visual_difference_sha=array_sha(visual_delta),visual_extended_input_sha=array_sha(x))
        for key in PREVIOUS_HASH_KEYS:assert hashes[key]==data['previous_asymmetric']['models'][model][key]==data['previous_direction']['models'][model][key]
        for key in OLD_HASH_KEYS:assert hashes[key]==data['previous']['models'][model][key]
        records[model]=dict(**hashes,frames=2598,failed_rows=1,valid_candidates=int(valid.sum()),input_shape=list(x.shape),target_shape=list(target.shape),
            previous_asymmetric_nine_hashes_exact=True,previous_direction_nine_hashes_exact=True,previous_sign_six_hashes_exact=True,
            raw_features_sha=array_sha(raw),anchor_index_sha=array_sha(data['index']),anchor_errors_sha=array_sha(data['anchor']),
            normalization_sha=data['inputs']['normalization_sha'],direction_normalization_sha=data['inputs']['direction_normalization_sha'],
            direction_receipt_binding=data['inputs']['direction_receipt'],basis_SHA_bind=data['basis_binding'],
            **{key:data[key] for key in VISUAL_BINDING_KEYS},all_invalid_row_retained=True,actual_data_objectives_or_policy_trials=0)
    return records,data


def run(write):
    from . import convex_train as T
    with threadpool_limits(limits=1):
        guard();code=C.bind(T.__file__)
        toy=synthetic(T);parity=code_parity(T,operators=write)
        if not write:print(json.dumps(dict(PASS=True,synthetic=toy,unchanged_functions=parity),indent=2));return
        records,data=actual_review(T)
    assert C.bind(T.__file__)==code
    out=dict(complete=True,PASS=True,created_at=C.now(),**T.metadata(),source_TRAIN_only=True,
        frames=2598,available_anchor_rows=2597,failed_rows_retained=1,models=records,inputs=data['inputs'],
        basis_SHA_bind=data['basis_binding'],direction_receipt_binding=data['inputs']['direction_receipt'],
        direction_normalization_sha=data['inputs']['direction_normalization_sha'],normalization_sha=data['inputs']['normalization_sha'],anchor_index_sha=data['inputs']['anchor_index_sha'],
        **{key:data[key] for key in VISUAL_BINDING_KEYS},synthetic=toy,unchanged_function_audit=parity,trainer=code,reviewer=C.bind(__file__),
        independent_objective_helper=C.bind(Q.__file__),independent_input_helper=C.bind(P.__file__),independent_sign_helper=C.bind(N.__file__),
        source_TRAIN_cached_label_values_read=True,raw_source_reference_reads=0,VAL_quality_read=False,real_targets_read=False,
        new_fits=0,optimizer_steps=0,actual_data_objective_trials=0,actual_data_policy_probes=0,prior_weights_used=False,
        basis_reselected=False,bandwidth_changed=False,direction_normalization_reselected=False,visual_normalization_reselected=False,
        new_reference_metric_calculations=0,image_forwards=0,new_PnP_calls=0,read_paths=sorted(READS),method_success=False,goal_complete=False)
    md=f'''# 고정 native DINO385 추가: 학습 전 독립 검산

**PREFIT PASS는 입력·수학 계약의 검산이며 T/R 개선 결과가 아니다.** 직전 Q의271차원 입력·target·원래valid·TRAIN2,598행(실패1 포함) 및9개 해시는 그대로다. 독립 검산을 통과한 native385만 추가하여656×2 계수를 학습하도록 준비했다.

시각 입력은 원래 R0 TRAIN 유효 후보5,194개의 고정 FP32 mean/std를 사용한다. FP32 정규화 후 FP64로 바꾸어 같은 frame의 R0 operational anchor를 뺀다. 모든 expert가 R0를 기준으로 삼으며, all-invalid1행과 invalid 후보는0이다. 원영상 지원점이0인 원래 유효 후보도 제외하지 않는다. 반사 padding 토큰 문맥의 영향까지 제거했다는 뜻은 아니다.

목적식·Hessian·Newton/Armijo·certificate·target 함수8개와 source45/원래+matched 실사5기준 함수는 Q와 AST가 같다. 대칭 Huber+추가 과소예측 Huber+nonzero sign logistic+λ1e−4 ridge, 행별 유효후보×2축 평균 후 전체2,598행 평균을 유지한다. 초기값0·1000iteration·2000call 상한을 바꾸지 않았다.

순수 fixture에서 독립 scalar/Torch64 gradient1,312개 및 전체1,312×1,312 Hessian을 비교했다. 사전 선택32좌표(새385차원 포함)의 차분 최대 gradient {toy['gradient_finite_difference_max']:.3g}, Hessian {toy['Hessian_finite_difference_max']:.3g}이다. 이전271 weight에0의385행을 붙였을 때 같은 목적값·예측을 atol=rtol1e−11로 확인했다. FP64 내적 reduction 차이를 byte동일로 잘못 요구하지 않으며,271 입력 prefix 자체는 byte동일이다.

실제 TRAIN에서는 입력·target 해시만 확인했다. 기존/새 weight·목적값·후보 선택을 실행하지 않았고 이미지·token·backbone·PnP·VAL/실사 GT를 읽거나 계산하지 않았다. 새 fit0이며 실제 수렴과 일반화는 후속 봉인된 실행에서 별도로 판정한다.

[전체 검산 JSON](PREFIT_REVIEW.json) · [native 입력 독립 검산](../pallet_pose_dino_native_inputs_20261001_v1/INPUT_VERIFICATION_KO.md)
'''
    C.save(C.DOC/'PREFIT_REVIEW.json',out);C.save(C.DOC/'PREFIT_REVIEW_KO.md',md)
    print('VISUAL656_PREFIT_PASS',C.bind(C.DOC/'PREFIT_REVIEW.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true')
    run(parser.parse_args().write)
