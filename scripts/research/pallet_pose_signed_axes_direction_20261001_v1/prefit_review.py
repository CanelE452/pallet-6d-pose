"""Independent fixed-direction input PREFIT; no actual-data weight probes.

Only ``write`` reads cached eligible TRAIN inputs/targets after explicit root
authorization. ``selfcheck`` uses invented arrays. Prior artifacts are immutable.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from . import common as C
from scripts.research.pallet_pose_signed_axes_sign_20261001_v1 import prefit_review as N

STATE={'reads':set(),'labels_allowed':False}
OLD_HASH_KEYS=('signed_target_sha','input_difference_sha','base_context_sha','errors_sha','scaled_excess_sha','original_valid_sha')
EXTRA_HASH_KEYS=('direction_raw_sha','direction_difference_sha','extended_input_sha')
EXPERTS=('R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3')
HYP=('long-face-front','short-face-front')
DIRECTION_RULE='BBOX_DIAGONAL_NORMALIZED_PROJECTED_MINUS_OBSERVED_XY9'
DIRECTION_NORMALIZATION='R0_TRAIN_valid_float32_mean_std_floor_1e-6_then_float64_candidate_minus_anchor'


def array_sha(value):
    value=np.ascontiguousarray(value)
    digest=hashlib.sha256(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def guard():
    sys.dont_write_bytecode=True
    arrays={C.PARENT_RAW/'SOURCE_FEATURES.npz',C.PARENT_RAW/'SOURCE_TRAIN_LABELS.npz',
            C.DIRECTION_RAW/'TRAIN_DIRECTIONS.npz'}
    writable={C.DOC/'PREFIT_REVIEW.json',C.DOC/'PREFIT_REVIEW_KO.md'}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();name=str(p);mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing and p.is_relative_to(C.ROOT):
            assert p in writable,('PREFIT_OUTPUT_ONLY',name);return
        assert not any(t in name for t in ('/data/evaluation/','/real_gt_v2/','/annotations/',
            'GEOMETRY_RESOLVED_POSE_GT','GEOMETRY_SIDETABLE','SYNTH_RECORDS','SYNTH_LABELS',
            'SOURCE_VAL_','REAL_RESULTS','REAL_CHOICES','POSE_METRICS','TRUTH_FOR_DISPLAY',
            '/fits/','/model_parameters/','TRAIN_CONVERGENCE')),('PREFIT_NO_QUALITY_OR_WEIGHT',name)
        assert p.suffix.lower() not in ('.png','.jpg','.jpeg','.pt','.pth','.onnx'),name
        if p.suffix=='.npz':assert p in arrays,('PREFIT_FROZEN_ARRAYS_ONLY',name)
        if not STATE['labels_allowed']:assert 'SOURCE_TRAIN_LABELS' not in name,('PREFIT_LABEL_STAGE_DENIED',name)
        if p.is_relative_to(C.ROOT):STATE['reads'].add(str(p.relative_to(C.ROOT)))
    sys.addaudithook(hook)


def exact(left,right):
    np.testing.assert_array_equal(left,right)
    assert array_sha(left)==array_sha(right)


def direction_difference(raw,valid,index,mean,std):
    raw=np.asarray(raw);mean,std=np.asarray(mean,np.float32),np.asarray(std,np.float32)
    assert raw.dtype==np.float32 and raw.shape==(*valid.shape,18)
    assert mean.shape==std.shape==(18,) and (std>=np.float32(1e-6)).all()
    result=np.zeros(raw.shape,np.float64)
    for row in range(len(valid)):
        active=np.flatnonzero(valid[row]);anchor=int(index[row])
        if not len(active):assert anchor==-1;continue
        assert anchor in (0,1) and valid[row,anchor]
        z={int(j):np.divide(np.subtract(raw[row,j],mean),std).astype(np.float64) for j in active}
        for j in active:result[row,j]=z[int(j)]-z[anchor]
    assert np.isfinite(result).all() and not result[~valid].any()
    return result


def extended_inputs(raw,valid,index,mean,std,basis,direction,mean18,std18):
    phi=N.fixed_map(raw,valid,index,mean,std,basis)
    base=N.differences(phi,valid,index)
    extra=direction_difference(direction,valid,index,mean18,std18)
    return phi,base,extra,np.concatenate([base,extra],axis=2)


def code_parity(T,include_evaluators):
    old=ast.parse((C.SIGN_HERE/'convex_train.py').read_text())
    new=ast.parse(Path(T.__file__).read_text())
    functions=('validate_anchor','signed_targets','objective','hessian','solver_config','newton_solve','certificate')
    for name in functions:
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),('SEALED_MATH_CHANGED',name)
    evaluations={}
    if include_evaluators:
        for filename,names in (('evaluate_source.py',('comparisons_for','chosen_pose','verify_source_reference_chain')),
                               ('evaluate_real.py',('gates_for','combined_gates'))):
            previous=ast.parse((C.SIGN_HERE/filename).read_text());current=ast.parse((C.HERE/filename).read_text())
            for name in names:
                a=next(n for n in previous.body if isinstance(n,ast.FunctionDef) and n.name==name)
                b=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
                assert ast.dump(a)==ast.dump(b),('GATE_OR_PHYSICAL_SCORING_CHANGED',filename,name)
            evaluations[filename]=dict(functions=list(names),previous=C.bind(C.SIGN_HERE/filename),current=C.bind(C.HERE/filename))
    assert T.FEATURE_DIM==271 and T.BASE_FEATURE_DIM==253 and T.DIRECTION_DIM==18
    assert T.HASH_KEYS==OLD_HASH_KEYS and T.EXTRA_HASH_KEYS==EXTRA_HASH_KEYS
    assert T.LAMBDA==1e-4 and T.MAX_ITER==1000 and T.MAX_CALLS==2000 and T.GAP_MAX==1e-6
    return dict(PASS=True,previous_code=C.bind(C.SIGN_HERE/'convex_train.py'),current_code=C.bind(T.__file__),
        exact_AST_functions=list(functions),evaluation_AST_parity=evaluations,
        allowed_change='Append the fixed normalized signed residual18 difference to the unchanged253 input; extend shared weight matrix to271x2 and provenance. Loss, targets, solver and all gates stay fixed.')


def synthetic(T):
    from scripts.research.pallet_pose_signed_axes_sign_20261001_v1 import convex_train as PREV
    from scripts.research.pallet_pose_residual_direction_audit_20261001_v1.verify_audit import scalar_projection
    from . import direction_features as D
    rng=np.random.default_rng(20261001271)
    valid=np.array([[1,1,1,1],[0,1,1,0],[1,0,0,0],[0,0,0,0],[1,1,0,1]],bool)
    index=np.array([0,1,0,-1,1],np.int64);rows=np.flatnonzero(valid.any(1))
    raw=rng.normal(size=(5,4,94)).astype(np.float32);raw[~valid]=np.nan
    direction=rng.normal(size=(5,4,18)).astype(np.float32);direction[~valid]=0.
    mean=rng.normal(size=94).astype(np.float32);std=(rng.random(94)+.5).astype(np.float32)
    mean18=rng.normal(size=18).astype(np.float32);std18=(rng.random(18)+.5).astype(np.float32)
    centers=rng.normal(size=(64,189)).astype(np.float64)
    width=float(np.median([np.sum((centers[i]-centers[j])**2) for i in range(64) for j in range(i+1,64)]))
    basis=dict(schema='pallet_pose_anchor_rbf_runtime_v1',context_dim=189,rbf_dim=64,
        centers=centers.tolist(),bandwidth_squared=width,normalization_sha=array_sha(np.stack([mean,std])))
    phi,x253,extra,x=extended_inputs(raw,valid,index,mean,std,basis,direction,mean18,std18)
    exact(T.rbf_inputs(raw,valid,index,mean,std,basis),phi)
    exact(D.direction_difference(direction,valid,index,mean18,std18),extra)
    exact(T.difference_inputs(raw,valid,index,mean,std,basis,direction,mean18,std18),x)
    exact(PREV.difference_inputs(raw,valid,index,mean,std,basis),x253)
    assert not x[rows,index[rows]].any() and not x[~valid].any()
    errors=np.array([[[2,3],[6,0],[1,2],[4,12]],[[np.inf,np.inf],[4,3],[8,0],[np.inf,np.inf]],
        [[2,2],[np.inf,np.inf],[np.inf,np.inf],[np.inf,np.inf]],[[np.inf,np.inf]]*4,
        [[5,4],[6,6],[np.inf,np.inf],[1,8]]],np.float64)
    anchor=np.full((5,2),np.inf);anchor[rows]=errors[rows,index[rows]]
    scale=np.array([2.,3.]);scaled,target=N.signed_targets(errors,valid,anchor,index,scale)
    exact(T.signed_targets(errors,valid,anchor,index,scale),(scaled,target))
    old_w=rng.normal(size=(253,2))*.005;embedded=np.concatenate([old_w,np.zeros((18,2))],axis=0)
    old=PREV.objective(old_w,x253,valid,target);same=T.objective(embedded,x,valid,target)
    for key in ('value','Huber','Sign_logistic','penalty','prediction','residual'):
        np.testing.assert_allclose(same[key],old[key],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(same['gradient'][:253],old['gradient'],rtol=1e-11,atol=1e-12)
    W=rng.normal(size=(271,2))*.005
    actual=T.objective(W,x,valid,target);independent=N.independent_objective(W,x,valid,target)
    for key in ('value','Huber','Sign_logistic','penalty','gradient','prediction','residual'):
        np.testing.assert_allclose(actual[key],independent[key],rtol=1e-11,atol=1e-12)
    H=N.independent_hessian(x,valid,independent['residual'],target)
    np.testing.assert_allclose(T.hessian(x,valid,actual['residual'],target),H,rtol=1e-11,atol=1e-12)
    xt=torch.tensor(x,dtype=torch.float64);yt=torch.tensor(target,dtype=torch.float64);vt=torch.tensor(valid)
    count=vt.sum(1).clamp(min=1).to(torch.float64)
    def torch_loss(flat):
        weights=flat.reshape(271,2);p=torch.einsum('nkd,da->nka',xt,weights)
        sign=torch.sign(yt);logit=-sign*p
        terms=torch.nn.functional.huber_loss(p,yt,reduction='none',delta=1.)
        terms+=torch.where(yt!=0,torch.logaddexp(torch.zeros_like(logit),logit),0.)
        return ((terms*vt[:,:,None]).sum((1,2))/(2*count)).mean()+.5e-4*weights.square().sum()
    wt=torch.tensor(W.ravel(),dtype=torch.float64,requires_grad=True)
    value=torch_loss(wt);gradient,=torch.autograd.grad(value,wt)
    th=torch.autograd.functional.hessian(torch_loss,wt).detach().numpy()
    np.testing.assert_allclose(float(value.detach()),actual['value'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(gradient.detach().numpy().reshape(271,2),actual['gradient'],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(th,H,rtol=1e-11,atol=1e-12)
    eps=1e-6;gradient_fd=[];hessian_fd=[]
    assert np.min(np.abs(np.abs(actual['residual'][valid])-1.))>1e-3
    for coordinate in range(542):
        delta=np.zeros((271,2));delta.ravel()[coordinate]=eps
        a=N.independent_objective(W+delta,x,valid,target);b=N.independent_objective(W-delta,x,valid,target)
        gradient_fd.append(abs((a['value']-b['value'])/(2*eps)-actual['gradient'].ravel()[coordinate]))
        hessian_fd.append(float(np.max(np.abs((a['gradient']-b['gradient']).ravel()/(2*eps)-H[:,coordinate]))))
    assert max(gradient_fd)<1e-8 and max(hessian_fd)<1e-8
    assert np.linalg.eigvalsh(H)[0]>=1e-4-1e-10
    assert not H[::2,1::2].any() and not H[1::2,::2].any()
    zero=T.objective(W,np.zeros((2,4,271)),np.zeros((2,4),bool),np.zeros((2,4,2)))
    assert zero['Huber']==zero['Sign_logistic']==0.
    exact(zero['gradient'],1e-4*W)
    exact(T.hessian(np.zeros((2,4,271)),np.zeros((2,4),bool),zero['residual'],np.zeros((2,4,2))),np.eye(542)*1e-4)
    # The projection is performed from a frozen pose; no solve or target exists.
    pose=dict(available=True,R_cf=np.eye(3).tolist(),centroid=[.15,-.2,4.],cf_extents=[1.3,.11,1.1])
    camera=np.array([[600.,2.,320.],[0.,610.,240.],[0.,0.,1.]])
    points=rng.normal(size=(9,2))+[330,250];box=np.array([10.,20.,110.,220.])
    projected=D.project_direction(pose,points,box,camera)
    scalar=scalar_projection(pose,camera,points,box)
    np.testing.assert_allclose(projected['direction18'],scalar['direction18'],rtol=1e-6,atol=1e-7)
    # All coordinate origins move together: no hidden extra padding is allowed.
    shift=np.array([100.,100.]);shifted=camera.copy();shifted[:2,2]+=shift
    shifted_projection=D.project_direction(pose,points+shift,box+np.tile(shift,2),shifted)
    np.testing.assert_allclose(projected['direction18'],shifted_projection['direction18'],rtol=1e-6,atol=1e-7)
    ck=dict(schema=T.CHECKPOINT_SCHEMA,**T.metadata(),normalization='old_float32_then_float64',
        names=T.candidate_names('UNION_s1'),bias=0.,lambda_l2=1e-4,mean=mean.tolist(),std=std.tolist(),
        weight=W.tolist(),rbf_basis=basis,rbf_basis_binding={'invented':True},basis_SHA_bind={'invented':True},
        direction_mean=mean18.tolist(),direction_std=std18.tolist(),
        direction_normalization_sha=array_sha(np.stack([mean18,std18])),direction_receipt_binding={'invented':True})
    original=T.signed_targets
    def denied(*args,**kwargs):raise AssertionError('RUNTIME_TARGET_LEAK')
    try:
        T.signed_targets=denied
        score=T.score_candidates(ck,raw,valid,index,direction)
    finally:T.signed_targets=original
    np.testing.assert_allclose(score,np.where(valid,actual['prediction'].max(2),np.inf),rtol=1e-12,atol=1e-12)
    exact(score[rows,index[rows]],np.zeros(len(rows)))
    N.reject(lambda:T.difference_inputs(raw,valid,np.array([-1,1,0,-1,1]),mean,std,basis,direction,mean18,std18))
    return dict(PASS=True,invented_arrays_only=True,old253_prefix_exact=True,
        old_weight_zero18_embedding_objective_and_score_equivalent=True,embedding_atol=1e-12,
        embedded_gradient_old253_equivalent=True,extra18_gradient_not_claimed_zero=True,
        scalar_projection_and_padding_translation_parity=True,FP32_direction_difference_exact=True,
        independent_objective_abs_difference=abs(actual['value']-independent['value']),
        independent_gradient_max_difference=float(np.max(np.abs(actual['gradient']-independent['gradient']))),
        torch_Hessian_max_difference=float(np.max(np.abs(th-H))),
        gradient_finite_difference_max=max(gradient_fd),Hessian_finite_difference_max=max(hessian_fd),
        full542_parameters_checked=True,anchor_and_invalid_zero=True,
        original_full_frame_valid_candidate_two_axis_denominator=True,runtime_target_function_disabled_PASS=True,
        artifact_reads=0,actual_data_weights_or_objectives_or_policies=0)


def read_bound(path):
    b=C.bind(path);return C.read(path),b


def load_cached_inputs():
    """No OLD fixed-array guard, fit API, weights or evaluation import."""
    C.verify(C.read(C.SIGN_DOC/'TRAIN_PROTOCOL_SHA.json'))
    sign_protocol,sign_protocol_binding=read_bound(C.SIGN_DOC/'TRAIN_PROTOCOL.json')
    C.verify(sign_protocol['inputs']['prefit_review'])
    previous=C.read(C.ROOT/sign_protocol['inputs']['prefit_review']['path'])
    assert previous['complete'] and previous['PASS'] and previous['frames']==2598
    assert previous['trainer']==C.bind(C.SIGN_HERE/'convex_train.py')
    C.verify(C.read(C.DIRECTION_DOC/'REPRESENTATION_PROTOCOL_SHA.json'))
    op,opbind=read_bound(C.DIRECTION_DOC/'REPRESENTATION_PROTOCOL.json')
    audited,abind=read_bound(C.DIRECTION_DOC/'REPRESENTATION_AUDIT.json')
    verified,vbind=read_bound(C.DIRECTION_DOC/'VERIFICATION.json')
    receipt,rbind=read_bound(C.DIRECTION_DOC/'FEATURE_AUDIT.json')
    for value in (audited,verified,receipt):assert value['complete'] and value['PASS'] and value['source_TRAIN_only']
    assert audited['protocol']==verified['protocol']==opbind
    assert verified['audited_representation']==abind and verified['audited_feature_receipt']==rbind
    assert receipt['directions']==op['inputs']['direction_features']
    C.verify(receipt['protocol'])
    assert receipt['protocol']==C.bind(C.DIRECTION_DOC/'AUDIT_PROTOCOL.json')
    assert audited['inputs']==op['inputs']
    assert op['inputs']['sign_prefit']==sign_protocol['inputs']['prefit_review']
    assert op['inputs']['direction_receipt']==rbind
    # Hash verification itself opens the cached TRAIN label container. Permit
    # this declared stage only after authenticating both previous reviews.
    STATE['labels_allowed']=True
    for binding in [*op['inputs'].values(),*op['codes']]:C.verify(binding)
    parent,pbind=read_bound(C.PARENT_DOC/'TRAIN_PROTOCOL.json')
    for key in ('features','train_labels','source_contract','feature_lock'):
        assert parent['inputs'][key]==sign_protocol['inputs'][key]
    assert op['inputs']['source_features']==parent['inputs']['features']
    assert op['inputs']['source_train_labels']==parent['inputs']['train_labels']
    basis_binding=op['inputs']['rbf_basis'];C.verify(basis_binding)
    basis_artifact=C.read(C.ROOT/basis_binding['path'])
    assert basis_artifact['complete'] and basis_artifact['PASS'] and previous['basis_SHA_bind']==basis_binding
    contract=C.read(C.ROOT/parent['inputs']['source_contract']['path'])
    assert contract['complete'] and contract['status']=='PASS'
    eligible=set(contract['fit_eligibility']['eligible_ids']['TRAIN']);assert len(eligible)==2598
    with np.load(C.ROOT/parent['inputs']['features']['path'],allow_pickle=False) as z:
        source_index=np.array([i for i,fid in enumerate(z['ids']) if str(fid) in eligible],np.int64)
        ids=z['ids'][source_index].copy();index=z['R0_GEO_index'][source_index].copy()
        assert len(ids)==2598 and (z['split'][source_index]=='TRAIN').all()
        assert z['hypothesis_names'].tolist()==list(HYP)
        features={m:z[m+'_geo'][source_index].copy() for m in EXPERTS}
        valid={m:z[m+'_valid'][source_index].copy() for m in EXPERTS}
    STATE['labels_allowed']=True
    with np.load(C.ROOT/parent['inputs']['train_labels']['path'],allow_pickle=False) as z:
        exact(z['ids'],ids);exact(z['source_index'],source_index)
        exact(np.flatnonzero(z['eligible_train_mask']),source_index)
        assert z['hypothesis_names'].tolist()==list(HYP)
        errors={m:np.stack([z[m+'_T_cm'],z[m+'_R_deg']],axis=-1) for m in EXPERTS}
        anchor=np.stack([z['R0_GEO_T_cm'],z['R0_GEO_R_deg']],axis=-1)
        scale=np.asarray([z['sT_cm'].item(),z['sR_deg'].item()],np.float64)
    with np.load(C.ROOT/receipt['directions']['path'],allow_pickle=False) as z:
        exact(z['ids'],ids);exact(z['source_index'],source_index);exact(z['anchor_index'],index)
        direction={m:z[m+'_direction18'].copy() for m in EXPERTS}
        for m in EXPERTS:exact(z[m+'_valid'],valid[m])
    values=features['R0'][valid['R0']];assert len(values)==5194
    mean,std=values.mean(0),np.maximum(values.std(0),np.float32(1e-6))
    values18=direction['R0'][valid['R0']]
    mean18,std18=values18.mean(0),np.maximum(values18.std(0),np.float32(1e-6))
    exact(np.asarray(receipt['normalization']['mean18'],np.float32),mean18)
    exact(np.asarray(receipt['normalization']['std18'],np.float32),std18)
    norm18=array_sha(np.stack([mean18,std18]));assert norm18==receipt['normalization']['array_sha']==audited['normalization18']['sha']
    norm94=array_sha(np.stack([mean,std]));assert norm94==previous['inputs']['normalization_sha']==basis_artifact['normalization_sha']
    for key,value in (('source_ids_sha',ids),('source_index_sha',source_index),('anchor_index_sha',index),
                      ('anchor_errors_sha',anchor),('scale_sha',scale)):
        assert previous['inputs'][key]==array_sha(value)
    assert ids[index<0].tolist()==['TEX__shard_04_f0110']
    for m in EXPERTS:
        assert features[m].shape==(2598,2,94) and direction[m].shape==(2598,2,18)
        assert features[m].dtype==direction[m].dtype==np.float32 and valid[m].dtype==bool
        assert array_sha(direction[m])==receipt['models'][m]['direction_sha']
        assert np.isfinite(direction[m]).all() and not direction[m][~valid[m]].any()
        assert np.isfinite(errors[m][valid[m]]).all() and np.isposinf(errors[m][~valid[m]]).all()
    source_metadata=receipt['bindings']['inputs']['metadata']
    inputs=dict(previous_sign_prefit=sign_protocol['inputs']['prefit_review'],previous_sign_protocol=sign_protocol_binding,
        previous_RBF_prefit=previous['inputs']['previous_RBF_prefit'],
        parent_train_protocol=pbind,basis=basis_binding,features=parent['inputs']['features'],train_labels=parent['inputs']['train_labels'],
        source_contract=parent['inputs']['source_contract'],source_ids_sha=array_sha(ids),source_index_sha=array_sha(source_index),
        normalization_sha=norm94,scale=scale.tolist(),scale_sha=array_sha(scale),anchor_index_sha=array_sha(index),
        anchor_errors_sha=array_sha(anchor),failed_ids=ids[index<0].tolist(),
        direction_features=receipt['directions'],direction_receipt=rbind,direction_representation=abind,
        direction_verification=vbind,direction_protocol=receipt['protocol'],direction_representation_protocol=opbind,source_metadata=source_metadata,
        direction_normalization_sha=norm18)
    return dict(ids=ids,source_index=source_index,index=index,features=features,valid=valid,errors=errors,anchor=anchor,
        scale=scale,mean=mean,std=std,mean18=mean18,std18=std18,direction=direction,basis=basis_artifact['basis'],
        basis_binding=basis_binding,previous=previous,representation=audited,inputs=inputs)


def actual_review(T):
    assert not (C.RAW/'fits').exists(),'PREFIT must precede every new fit.'
    data=load_cached_inputs();records={}
    for model in C.MODEL_NAMES:
        experts=['R0']+([] if model=='R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
        raw,valid,errors=[np.concatenate([data[key][m] for m in experts],axis=1) for key in ('features','valid','errors')]
        direction=np.concatenate([data['direction'][m] for m in experts],axis=1)
        phi,base,extra,x=extended_inputs(raw,valid,data['index'],data['mean'],data['std'],data['basis'],direction,data['mean18'],data['std18'])
        exact(T.difference_inputs(raw,valid,data['index'],data['mean'],data['std'],data['basis'],direction,data['mean18'],data['std18']),x)
        with np.errstate(all='raise'):
            scaled,target=N.signed_targets(errors,valid,data['anchor'],data['index'],data['scale'])
            actual_scaled,actual_target=T.signed_targets(errors,valid,data['anchor'],data['index'],data['scale'])
        exact(actual_scaled,scaled);exact(actual_target,target)
        hashes=dict(signed_target_sha=array_sha(target),input_difference_sha=array_sha(base),base_context_sha=array_sha(phi),
            errors_sha=array_sha(errors),scaled_excess_sha=array_sha(scaled),original_valid_sha=array_sha(valid),
            direction_raw_sha=array_sha(direction),direction_difference_sha=array_sha(extra),extended_input_sha=array_sha(x))
        for key in OLD_HASH_KEYS:
            assert hashes[key]==data['previous']['models'][model][key]==data['representation']['models'][model]['prior_six_hashes'][key]
        assert hashes['direction_difference_sha']==data['representation']['models'][model]['extra18_sha']
        assert hashes['extended_input_sha']==data['representation']['models'][model]['extended271_sha']
        records[model]=dict(**hashes,frames=2598,valid_candidates=int(valid.sum()),failed_rows=1,
            input_shape=list(x.shape),target_shape=list(target.shape),prior_six_hashes_exact=True,
            direction_O_hashes_exact=True,raw_features_sha=array_sha(raw),
            anchor_index_sha=data['inputs']['anchor_index_sha'],anchor_errors_sha=data['inputs']['anchor_errors_sha'],
            normalization_sha=data['inputs']['normalization_sha'],direction_normalization_sha=data['inputs']['direction_normalization_sha'],
            direction_receipt_binding=data['inputs']['direction_receipt'],basis_SHA_bind=data['basis_binding'],
            old253_prefix_exact=True,all_invalid_row_retained=True,anchor_input_target_zero=True,
            actual_data_objectives_or_policy_trials=0)
    return records,data


def run(write):
    from . import convex_train as T
    # Prewarm library controls before installing strict I/O guards.
    with threadpool_limits(limits=1):
        guard()
        code=C.bind(T.__file__)
        toy=synthetic(T);parity=code_parity(T,include_evaluators=write)
        solver=N.synthetic_newton(T);extremes=N.synthetic_sign_extremes(T)
        if not write:
            print(json.dumps(dict(PASS=True,synthetic=toy,unchanged_functions=parity,solver=solver,sign_extremes=extremes),indent=2));return
        records,data=actual_review(T)
    assert C.bind(T.__file__)==code
    out=dict(complete=True,PASS=True,created_at=C.now(),**T.metadata(),source_TRAIN_only=True,
        frames=2598,available_anchor_rows=2597,failed_rows_retained=1,models=records,inputs=data['inputs'],
        basis_SHA_bind=data['basis_binding'],direction_receipt_binding=data['inputs']['direction_receipt'],
        direction_normalization_sha=data['inputs']['direction_normalization_sha'],
        normalization_sha=data['inputs']['normalization_sha'],anchor_index_sha=data['inputs']['anchor_index_sha'],
        synthetic=toy,unchanged_function_audit=parity,solver_synthetic=solver,sign_extremes=extremes,
        trainer=code,reviewer=C.bind(__file__),independent_math_helper=C.bind(N.__file__),
        source_TRAIN_cached_label_values_read=True,raw_source_reference_reads=0,VAL_quality_read=False,real_targets_read=False,
        new_fits=0,optimizer_steps=0,actual_data_objective_trials=0,actual_data_policy_probes=0,prior_weights_used=False,
        basis_reselected=False,bandwidth_changed=False,direction_normalization_reselected=False,
        new_reference_metric_calculations=0,image_forwards=0,new_PnP_calls=0,read_paths=sorted(STATE['reads']),
        method_success=False,goal_complete=False)
    text=f'''# 고정 잔차 방향18 추가의 학습 전 독립 검산

**PREFIT PASS는 입력·수학 계약 검산이며 학습이나 T/R 개선 결과가 아니다.** 기존253 특징과 O에서 고정한 bbox정규화 signed residual18을 이어 붙인271입력만 변경한다. 기존94 정규화·RBF basis·signed target·scale·유효 후보·TRAIN2,598행(실패1행 포함)을 유지했고, 원래6해시와 O의추가18/271해시를 독립 재구성해 대조했다.

새18의 mean/std는 O의 R0 TRAIN 유효5,194후보에서 고정한 FP32 값이다. FP32 정규화 뒤 FP64로 올려 candidate−R0 GEO anchor 차이를 계산한다. anchor와invalid 입력은0이다. 실제 TRAIN에서는 특징과타깃 해시만 검사했고 이전 weight·목적식·선택 정책을 적용하지 않았다.

합성 예제에서 이전253 가중치에zero18을 붙이면 같은 Huber+sign-logistic+ridge 목적식과 점수를 재현했다(수치허용1e−12). 추가18축 gradient가0이라는 주장은 하지 않는다. 독립 scalar 목적식, Torch64 logaddexp/autograd,542개 좌표 중앙차분과542×542 Hessian을 대조했다. gradient 차분 최대{toy['gradient_finite_difference_max']:.3g}, Hessian 차분 최대{toy['Hessian_finite_difference_max']:.3g}이다. 모든542계수의 ridgeλ1e−4가 강볼록성을 유지한다.

기존 목적식·Hessian·signed target·Newton/Armijo·인증 함수 AST와 source45/원래+matched 실사5기준 AST를 확인했다. solver는 영초기화·최대1,000accepted iteration/2,000objective calls·gradient Linf≤1e−8 및 gap≤1e−6를 그대로 사용한다. 독립 작은 합성문제로 accepted/rejected state와 예산을 검산했다.

고정 pose와9점 관측/K/bbox의 방향 추출을 별도 성분 투영식으로 대조했다. corner0..7과 P8 중심을 유지하고, K·관측·bbox를함께100px 옮겨도 잔차가 같음을 확인했다. runtime의 target 생성함수를 강제로 막아도 점수를 계산했다. 실제 pose 생성·image forward·VAL/실사 참조 열람은0이다.

[검산 JSON](PREFIT_REVIEW.json) · [이전 동일 목적식 검산](../pallet_pose_signed_axes_sign_20261001_v1/PREFIT_REVIEW.json) · [방향 입력 독립 감사](../pallet_pose_residual_direction_audit_20261001_v1/VERIFICATION_KO.md)
'''
    C.save(C.DOC/'PREFIT_REVIEW_KO.md',text);out['note']=C.bind(C.DOC/'PREFIT_REVIEW_KO.md')
    C.save(C.DOC/'PREFIT_REVIEW.json',out)
    print('SIGNED_AXES_DIRECTION_PREFIT_PASS',C.bind(C.DOC/'PREFIT_REVIEW.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('selfcheck','write'))
    run(parser.parse_args().stage=='write')
