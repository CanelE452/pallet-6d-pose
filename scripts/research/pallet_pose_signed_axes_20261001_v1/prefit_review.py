"""Independent signed T/R target, centered RBF input and Huber math audit.

selfcheck uses invented arrays only. Actual TRAIN inspection requires root's
explicit run authorization. The old fixed basis is authenticated and reused;
no centers/width are reselected, no actual-data weights or policies are tested.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from . import common as C

LOSS_RULE='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_HUBER'
TARGET_RULE='SIGNED_LOG1P_NORMALIZED_TR_ANCHOR_EXCESS'
INPUT_RULE='RBF253_CANDIDATE_MINUS_R0_ANCHOR'
PREDICTION_RULE='MAX_TWO_SIGNED_LOG1P_AXES'
STATE={'TRAIN_labels_allowed':False,'reads':set()}


def array_sha(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(str(value.dtype).encode());h.update(json.dumps(list(value.shape)).encode());h.update(value.tobytes())
    return h.hexdigest()


def guard():
    writable={C.DOC/'PREFIT_REVIEW.json',C.DOC/'PREFIT_REVIEW_KO.md'}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to(C.ROOT):return
        mode=args[1];flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_TRUNC))
        if writing:
            assert p in writable,('PREFIT_WRITE_SCOPE',str(p));return
        name=str(p)
        assert not any(s in name for s in ('/data/evaluation/','/real_gt_v2/','/annotations/',
            'GEOMETRY_RESOLVED_POSE_GT','AXIS_REVIEW_MANIFEST','GEOMETRY_SIDETABLE','SYNTH_RECORDS',
            '/SOURCE_VAL_','/REAL_FEATURE','/REAL_RESULTS','/REAL_CHOICES','/REAL_FRAME_RESULTS',
            '/POSE_METRICS.json','/TRUTH_FOR_DISPLAY','CANDIDATE_BOUNDS')),('PREFIT_QUALITY_OR_RAW_REFERENCE_DENIED',name)
        if not STATE['TRAIN_labels_allowed']:
            assert 'SOURCE_TRAIN_LABELS' not in name,('TRAIN_LABEL_ACCESS_NOT_ENABLED',name)
        assert p.suffix.lower() not in ('.png','.jpg','.jpeg','.pt','.pth','.onnx'),name
        STATE['reads'].add(str(p.relative_to(C.ROOT)))
    sys.addaudithook(hook)


def reject(callback):
    try:callback()
    except (AssertionError,ValueError):return
    raise AssertionError('INVALID_CONTRACT_ACCEPTED')


def exact(actual,expected):
    np.testing.assert_array_equal(actual,expected)
    assert array_sha(actual)==array_sha(expected),'ARRAY_BYTES_OR_DTYPE_DIFFER'


def signed_targets(errors,valid,anchor_errors,index,scale):
    errors=np.asarray(errors);valid=np.asarray(valid);index=np.asarray(index);scale=np.asarray(scale)
    assert errors.dtype==np.float64 and valid.dtype==bool and errors.shape==(*valid.shape,2)
    assert anchor_errors.shape==(len(valid),2) and index.shape==(len(valid),)
    assert scale.shape==(2,) and np.isfinite(scale).all() and (scale>0).all()
    assert np.isfinite(errors[valid]).all() and np.isposinf(errors[~valid]).all()
    excess=np.zeros_like(errors);target=np.zeros_like(errors)
    for i in range(len(valid)):
        active=np.flatnonzero(valid[i]).tolist();a=int(index[i])
        if not active:
            assert a==-1 and np.isposinf(anchor_errors[i]).all();continue
        assert a in (0,1) and a in active and np.isfinite(anchor_errors[i]).all()
        np.testing.assert_array_equal(anchor_errors[i],errors[i,a])
        for j in active:
            for axis in range(2):
                value=(errors[i,j,axis]-anchor_errors[i,axis])/scale[axis]
                excess[i,j,axis]=value
                target[i,j,axis]=np.sign(value)*np.log1p(abs(value))
        assert not excess[i,a].any() and not target[i,a].any()
    assert np.isfinite(excess).all() and np.isfinite(target).all()
    assert not excess[~valid].any() and not target[~valid].any()
    return excess,target


def base_context(raw,valid,index,mean,std):
    assert raw.dtype==np.float32 and raw.shape==(*valid.shape,94)
    mean,std=np.asarray(mean,np.float32),np.asarray(std,np.float32)
    assert mean.shape==std.shape==(94,) and (std>=np.float32(1e-6)).all()
    result=np.zeros((*valid.shape,189),np.float64)
    for i in range(len(valid)):
        active=np.flatnonzero(valid[i]).tolist();a=int(index[i])
        if not active:
            assert a==-1;continue
        assert a in (0,1) and a in active
        anchor=((raw[i,a]-mean)/std).astype(np.float64)
        for j in active:
            z=((raw[i,j]-mean)/std).astype(np.float64)
            result[i,j,:94]=z
            result[i,j,94:188]=np.array([abs(float(z[k])-float(anchor[k])) for k in range(94)])
            result[i,j,188]=float(j==a)
    assert np.isfinite(result).all() and not result[~valid].any()
    return result


def fixed_basis(basis,mean,std):
    assert basis['schema']=='pallet_pose_anchor_rbf_runtime_v1' and basis['context_dim']==189 and basis['rbf_dim']==64
    centers=np.asarray(basis['centers'],np.float64);width=float(basis['bandwidth_squared'])
    assert centers.shape==(64,189) and np.isfinite(centers).all() and np.isfinite(width) and width>0
    assert basis['normalization_sha']==array_sha(np.stack([np.asarray(mean,np.float32),np.asarray(std,np.float32)]))
    # The basis is reused exactly. Selection/width recomputation is deliberately
    # left to the previously frozen independent RBF PREFIT receipt.
    return centers,width


def fixed_map(raw,valid,index,mean,std,basis):
    context=base_context(raw,valid,index,mean,std);centers,width=fixed_basis(basis,mean,std)
    result=np.zeros((*valid.shape,253),np.float64);result[:,:,:189]=context
    for i,j in zip(*np.where(valid)):
        distance=np.sum((context[i,j][None,:]-centers)**2,axis=1)
        result[i,j,189:]=np.exp(-distance/(2*width))
    assert np.isfinite(result).all() and not result[~valid].any()
    return result


def differences(phi,valid,index):
    assert phi.dtype==np.float64 and phi.shape==(*valid.shape,253)
    result=np.zeros_like(phi)
    for i in range(len(valid)):
        active=np.flatnonzero(valid[i]).tolist();a=int(index[i])
        if not active:
            assert a==-1;continue
        assert a in (0,1) and a in active
        for j in active:result[i,j]=phi[i,j]-phi[i,a]
        assert not result[i,a].any()
    assert np.isfinite(result).all() and not result[~valid].any()
    return result


def independent_objective(W,x,valid,target,ridge=1e-4):
    assert W.shape==(253,2) and x.shape==(*valid.shape,253) and target.shape==(*valid.shape,2)
    assert np.isfinite(x).all() and np.isfinite(target).all() and not x[~valid].any() and not target[~valid].any()
    n=len(valid);prediction=np.zeros_like(target);residual=np.zeros_like(target)
    gradient=np.zeros_like(W);data_loss=0.
    for i in range(n):
        active=np.flatnonzero(valid[i]).tolist()
        if not active:continue
        coefficient=1/(n*len(active)*2)
        for j in active:
            for axis in range(2):
                predicted=float(np.dot(x[i,j],W[:,axis]))
                r=predicted-target[i,j,axis]
                prediction[i,j,axis]=predicted;residual[i,j,axis]=r
                absolute=abs(r)
                value=.5*r*r if absolute<=1 else absolute-.5
                slope=r if absolute<=1 else np.sign(r)
                data_loss+=coefficient*value
                gradient[:,axis]+=coefficient*slope*x[i,j]
    penalty=ridge/2*float(np.sum(W*W))
    return dict(value=data_loss+penalty,Huber=data_loss,penalty=penalty,gradient=gradient+ridge*W,
                prediction=prediction,residual=residual)


def independent_hessian(x,valid,residual,ridge=1e-4):
    n=len(valid);H=np.eye(506)*ridge
    for i in range(n):
        active=np.flatnonzero(valid[i]).tolist()
        if not active:continue
        coefficient=1/(n*len(active)*2)
        for j in active:
            for axis in range(2):
                # At |residual|=1 the classical Hessian is undefined. This
                # routine's branch convention is tested only away from kinks.
                if abs(residual[i,j,axis])<1:
                    H[axis::2,axis::2]+=coefficient*np.outer(x[i,j],x[i,j])
    return H


def synthetic(T):
    import torch
    rng=np.random.default_rng(20261001253)
    valid=np.array([[1,1,1,1],[0,1,1,0],[1,0,0,0],[0,0,0,0],[1,1,0,1]],bool)
    index=np.array([0,1,0,-1,1],np.int64)
    raw=rng.normal(size=(5,4,94)).astype(np.float32);raw[~valid]=np.nan
    mean=rng.normal(size=94).astype(np.float32);std=(rng.random(94)+.5).astype(np.float32)
    centers=rng.normal(size=(64,189)).astype(np.float64)
    width=float(np.median([np.sum((centers[i]-centers[j])**2) for i in range(64) for j in range(i+1,64)]))
    basis=dict(schema='pallet_pose_anchor_rbf_runtime_v1',context_dim=189,rbf_dim=64,centers=centers.tolist(),
        bandwidth_squared=width,normalization_sha=array_sha(np.stack([mean,std])))
    errors=np.array([[[2,3],[6,0],[1,2],[4,12]],[[np.inf,np.inf],[4,3],[8,0],[np.inf,np.inf]],
        [[2,2],[np.inf,np.inf],[np.inf,np.inf],[np.inf,np.inf]],[[np.inf,np.inf]]*4,
        [[5,4],[6,6],[np.inf,np.inf],[1,8]]],np.float64)
    anchor=np.full((5,2),np.inf);present=np.flatnonzero(valid.any(1));anchor[present]=errors[present,index[present]]
    scale=np.array([2.,3.])
    with np.errstate(all='raise'):
        excess,target=signed_targets(errors,valid,anchor,index,scale)
        actual_excess,actual_target=T.signed_targets(errors,valid,anchor,index,scale)
        exact(actual_excess,excess);exact(actual_target,target)
    assert (target[valid]<0).any() and (target[valid]>0).any()
    phi=fixed_map(raw,valid,index,mean,std,basis)
    x=differences(phi,valid,index)
    exact(T.rbf_inputs(raw,valid,index,mean,std,basis),phi)
    exact(T.difference_inputs(raw,valid,index,mean,std,basis),x)
    assert not x[present,index[present]].any() and not target[present,index[present]].any()
    W=rng.normal(size=(253,2))*.005
    independent=independent_objective(W,x,valid,target)
    actual=T.objective(W,x,valid,target)
    for key in ('value','Huber','penalty','gradient','prediction','residual'):
        np.testing.assert_allclose(actual[key],independent[key],rtol=1e-11,atol=1e-12)
    kink_distance=float(np.min(np.abs(np.abs(independent['residual'][valid])-1)))
    assert kink_distance>1e-3
    H=independent_hessian(x,valid,independent['residual'])
    np.testing.assert_allclose(T.hessian(x,valid,actual['residual']),H,rtol=1e-11,atol=1e-12)
    xt=torch.tensor(x,dtype=torch.float64);yt=torch.tensor(target,dtype=torch.float64)
    vt=torch.tensor(valid);count=vt.sum(1).clamp(min=1).to(torch.float64)
    def torch_loss(flat):
        weights=flat.reshape(253,2)
        prediction=torch.einsum('nkd,da->nka',xt,weights)
        losses=torch.nn.functional.huber_loss(prediction,yt,reduction='none',delta=1.)
        per_frame=(losses*vt[:,:,None]).sum((1,2))/(2*count)
        return per_frame.mean()+.0001/2*(weights*weights).sum()
    tw=torch.tensor(W.reshape(-1),dtype=torch.float64,requires_grad=True)
    tl=torch_loss(tw);tg=torch.autograd.grad(tl,tw,create_graph=True)[0]
    tH=torch.autograd.functional.hessian(torch_loss,tw).detach().numpy()
    np.testing.assert_allclose(float(tl.detach()),independent['value'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(tg.detach().numpy().reshape(253,2),independent['gradient'],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(tH,H,rtol=1e-11,atol=1e-12)
    gradient_fd=[];hessian_fd=[];epsilon=1e-6
    for coordinate in range(506):
        delta=np.zeros((253,2));delta.reshape(-1)[coordinate]=epsilon
        plus=independent_objective(W+delta,x,valid,target);minus=independent_objective(W-delta,x,valid,target)
        gradient_fd.append(abs((plus['value']-minus['value'])/(2*epsilon)-independent['gradient'].reshape(-1)[coordinate]))
        hessian_fd.append(float(np.max(np.abs((plus['gradient']-minus['gradient']).reshape(-1)/(2*epsilon)-H[:,coordinate]))))
    assert max(gradient_fd)<1e-8 and max(hessian_fd)<1e-8
    eigenvalue=float(np.linalg.eigvalsh(H)[0]);assert eigenvalue>=1e-4-1e-10
    empty=T.objective(W,np.zeros((2,4,253)),np.zeros((2,4),bool),np.zeros((2,4,2)))
    assert empty['Huber']==0
    np.testing.assert_array_equal(empty['gradient'],1e-4*W)
    np.testing.assert_array_equal(T.hessian(np.zeros((2,4,253)),np.zeros((2,4),bool),empty['residual']),np.eye(506)*1e-4)
    fixture_valid=np.array([[1,1,0,0],[0,0,0,0]],bool)
    fixture_target=np.zeros((2,4,2));fixture_target[0,0,1]=2.
    assert T.objective(np.zeros((253,2)),np.zeros((2,4,253)),fixture_valid,fixture_target)['Huber']==.1875
    # Kink values/slopes are continuous; no Hessian differentiability claim.
    kink_x=np.zeros((1,2,253));kink_x[0,1,0]=1
    for target_value in (-1.,1.):
        kink_target=np.zeros((1,2,2));kink_target[0,1,0]=target_value
        kink=T.objective(np.zeros((253,2)),kink_x,np.ones((1,2),bool),kink_target)
        assert kink['Huber']==.125
        assert kink['gradient'][0,0]==-target_value/4
    assert not H[::2,1::2].any() and not H[1::2,::2].any()
    assert T.LAMBDA==1e-4 and T.MAX_ITER==1000 and T.MAX_CALLS==2000 and T.GAP_MAX==1e-6
    names=T.candidate_names('UNION_s1')
    ck=dict(schema=T.CHECKPOINT_SCHEMA,feature_map=T.FEATURE_MAP,feature_dim=253,raw_feature_dim=94,
        output_dim=2,loss_rule=LOSS_RULE,target_rule=TARGET_RULE,input_rule=INPUT_RULE,
        prediction_rule=PREDICTION_RULE,huber_delta=1.,runtime_uses_margin=False,
        normalization='old_float32_then_float64',names=names,bias=0.,lambda_l2=1e-4,
        mean=mean.tolist(),std=std.tolist(),weight=W.tolist(),rbf_basis=basis,
        rbf_basis_binding={'invented':True},basis_SHA_bind={'invented':True})
    ck.update(T.metadata())
    expected_score=np.where(valid,np.max(independent['prediction'],axis=2),np.inf)
    original=T.signed_targets
    def forbidden(*args,**kwargs):raise AssertionError('RUNTIME_MUST_NOT_BUILD_PHYSICAL_TARGETS')
    try:
        T.signed_targets=forbidden
        actual_score=T.score_candidates(ck,raw,valid,index)
    finally:T.signed_targets=original
    np.testing.assert_allclose(actual_score,expected_score,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(actual_score[present,index[present]],0.)
    selected=T.select_candidates(actual_score,valid,names)
    assert (actual_score[present,selected[present]]<=0).all() and selected[3]==-1
    assert T.select_candidates(np.zeros((1,4)),np.ones((1,4),bool),names).tolist()==[0]
    reject(lambda:T.signed_targets(errors,valid,anchor,np.array([-1,1,0,-1,1]),scale))
    reject(lambda:T.difference_inputs(raw,valid,np.array([-1,1,0,-1,1]),mean,std,basis))
    return dict(PASS=True,signed_target_scalar_exact=True,negative_positive_and_zero_targets_preserved=True,
        valid_only_arithmetic_avoids_inf_minus_inf=True,original253_and_anchor_difference_bit_exact=True,
        independent_objective_absolute_difference=abs(actual['value']-independent['value']),
        independent_gradient_max_difference=float(np.abs(actual['gradient']-independent['gradient']).max()),
        torch_gradient_max_difference=float(np.abs(tg.detach().numpy().reshape(253,2)-independent['gradient']).max()),
        torch_Hessian_max_difference=float(np.abs(tH-H).max()),
        gradient_finite_difference_max=max(gradient_fd),Hessian_finite_difference_max=max(hessian_fd),
        Hessian_min_eigenvalue=eigenvalue,minimum_distance_from_Huber_kinks=kink_distance,
        Huber_kink_scope='Value and gradient continuous; classical Hessian/finite differences verified away from residual ±1.',
        full_frame_valid_candidate_two_axis_denominator_exact=True,all_invalid_data_loss_zero_with_full_denominator=True,
        anchor_input_target_and_prediction_zero=True,original_tie_policy_preserved=True,
        anchor_zero_bound_is_prediction_only_not_true_T_R_guarantee=True,
        runtime_uses_reference_targets_or_margins=False,artifact_reads=0)


def actual_review(T):
    assert not (C.RAW/'fits').exists(),'PREFIT must precede every new fit.'
    C.verify(C.read(C.RBF_DOC/'TRAIN_PROTOCOL_SHA.json'))
    old_protocol=C.read(C.RBF_DOC/'TRAIN_PROTOCOL.json')
    old_prefit_binding=old_protocol['inputs']['prefit_review'];C.verify(old_prefit_binding)
    prior=C.read(C.ROOT/old_prefit_binding['path']);assert prior['complete'] and prior['PASS']
    basis_binding=old_protocol['inputs']['rbf_basis'];C.verify(basis_binding)
    artifact=C.read(C.ROOT/basis_binding['path']);assert artifact['complete'] and artifact['PASS']
    assert prior['basis_SHA_bind']==basis_binding and artifact['source_TRAIN_only'] and not artifact['labels_read']
    assert not artifact['VAL_quality_read'] and not artifact['real_targets_read']
    basis=artifact['basis']
    STATE['TRAIN_labels_allowed']=True
    T.OLD.install_training_guard();parent=T.OLD.load_training_inputs()
    for key,binding in parent['protocol']['inputs'].items():assert old_protocol['inputs'][key]==binding
    old_inputs=prior['inputs']
    assert old_inputs['source_ids_sha']==array_sha(parent['ids'])
    assert old_inputs['source_index_sha']==array_sha(parent['source_index'])
    assert old_inputs['normalization_sha']==parent['normalization_sha']==basis['normalization_sha']
    np.testing.assert_array_equal(np.asarray(artifact['mean'],np.float32),parent['mean'])
    np.testing.assert_array_equal(np.asarray(artifact['std'],np.float32),parent['std'])
    assert old_inputs['scale_sha']==array_sha(parent['scale'])
    with np.load(C.ROOT/parent['protocol']['inputs']['features']['path'],allow_pickle=False) as z:
        index=z['R0_GEO_index'][parent['source_index']]
    with np.load(C.ROOT/parent['protocol']['inputs']['train_labels']['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],parent['ids'])
        anchor_errors=np.stack([z['R0_GEO_T_cm'],z['R0_GEO_R_deg']],axis=-1)
    assert old_inputs['anchor_index_sha']==array_sha(index)
    assert old_inputs['anchor_errors_sha']==array_sha(anchor_errors)
    assert len(parent['ids'])==2598 and (index>=0).sum()==2597
    assert parent['ids'][index<0].tolist()==['TEX__shard_04_f0110']
    results={}
    for model in C.MODEL_NAMES:
        parts=['R0']+([] if model=='R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
        raw=np.concatenate([parent['features'][p] for p in parts],axis=1)
        valid=np.concatenate([parent['valid'][p] for p in parts],axis=1)
        errors=np.concatenate([parent['errors'][p] for p in parts],axis=1)
        before=prior['models'][model]
        assert before['original_valid_sha']==array_sha(valid)
        assert before['unscaled_errors_sha']==array_sha(errors)
        assert before['raw_features_sha']==array_sha(raw)
        assert before['anchor_errors_sha']==array_sha(anchor_errors) and before['anchor_index_sha']==array_sha(index)
        phi=fixed_map(raw,valid,index,parent['mean'],parent['std'],basis)
        assert before['context_sha']==array_sha(phi)
        exact(T.rbf_inputs(raw,valid,index,parent['mean'],parent['std'],basis),phi)
        x=differences(phi,valid,index)
        exact(T.difference_inputs(raw,valid,index,parent['mean'],parent['std'],basis),x)
        with np.errstate(all='raise'):
            excess,target=signed_targets(errors,valid,anchor_errors,index,parent['scale'])
            actual_excess,actual_target=T.signed_targets(errors,valid,anchor_errors,index,parent['scale'])
        exact(actual_excess,excess);exact(actual_target,target)
        present=np.flatnonzero(valid.any(1))
        assert np.array_equal(valid.any(1),index>=0)
        assert not x[present,index[present]].any() and not target[present,index[present]].any()
        hashes=dict(signed_target_sha=array_sha(target),input_difference_sha=array_sha(x),
            base_context_sha=array_sha(phi),errors_sha=array_sha(errors),scaled_excess_sha=array_sha(excess),
            original_valid_sha=array_sha(valid))
        results[model]=dict(**hashes,frames=2598,valid_candidates=int(valid.sum()),failed_rows=1,
            input_shape=list(x.shape),target_shape=list(target.shape),basis_SHA_bind=basis_binding,
            raw_features_sha=array_sha(raw),normalization_sha=parent['normalization_sha'],
            anchor_index_sha=array_sha(index),anchor_errors_sha=array_sha(anchor_errors),scale_sha=array_sha(parent['scale']),
            previous253_features_errors_and_validity_exact=True,independent_signed_targets_and_inputs_exact=True,
            all_invalid_row_retained=True,invalid_input_target_zero=True,anchor_input_target_zero=True,
            actual_data_objectives_or_policy_trials=0)
    inputs=dict(parent_train_protocol=parent['binding'],previous_RBF_train_protocol=C.bind(C.RBF_DOC/'TRAIN_PROTOCOL.json'),
        previous_RBF_prefit=old_prefit_binding,basis=basis_binding,
        features=parent['protocol']['inputs']['features'],train_labels=parent['protocol']['inputs']['train_labels'],
        source_contract=parent['protocol']['inputs']['source_contract'],
        source_ids_sha=array_sha(parent['ids']),source_index_sha=array_sha(parent['source_index']),
        normalization_sha=parent['normalization_sha'],scale=parent['scale'].tolist(),scale_sha=array_sha(parent['scale']),
        anchor_index_sha=array_sha(index),anchor_errors_sha=array_sha(anchor_errors),
        failed_ids=parent['ids'][index<0].tolist())
    return results,inputs,basis_binding


def run(write):
    sys.dont_write_bytecode=True;guard()
    from . import convex_train as T
    assert T.OLD.READS is None
    assert T.LOSS_RULE==LOSS_RULE and T.TARGET_RULE==TARGET_RULE and T.INPUT_RULE==INPUT_RULE
    assert T.PREDICTION_RULE==PREDICTION_RULE and T.OUTPUT_DIM==2 and T.HUBER_DELTA==1.
    code=C.bind(T.__file__)
    with threadpool_limits(limits=1):
        toy=synthetic(T)
        if not write:
            print(json.dumps(toy,indent=2));return
        models,inputs,basis_binding=actual_review(T)
    assert C.bind(T.__file__)==code,'TRAINER_CHANGED_DURING_REVIEW'
    result=dict(complete=True,PASS=True,created_at=C.now(),source_TRAIN_only=True,frames=2598,
        available_anchor_rows=2597,failed_rows_retained=1,feature_dim=253,raw_feature_dim=94,output_dim=2,
        feature_map=T.FEATURE_MAP,loss_rule=LOSS_RULE,target_rule=TARGET_RULE,input_rule=INPUT_RULE,
        prediction_rule=PREDICTION_RULE,huber_delta=1.,runtime_uses_margin=False,
        basis_SHA_bind=basis_binding,models=models,inputs=inputs,synthetic=toy,
        trainer=code,reviewer=C.bind(__file__),source_TRAIN_cached_label_values_read=True,
        raw_source_reference_reads=0,VAL_quality_read=False,real_targets_read=False,
        new_fits=0,optimizer_steps=0,actual_data_objective_trials=0,actual_data_policy_probes=0,
        prior_weights_used=False,basis_reselected=False,bandwidth_changed=False,
        new_reference_metric_calculations=0,image_forwards=0,new_PnP_calls=0,
        read_paths=sorted(STATE['reads']),training_guard_read_paths=sorted(T.OLD.READS),
        method_success=False,goal_complete=False)
    markdown=f'''# Signed T/R 회귀 학습 전 독립 검산

**입력·목적식 검산 PASS. 새 fit·VAL·실사 성능 평가는 실행하지 않았다.** 기존 고정 RBF253을 재사용하고 후보 특징에서 같은 행 R0 GEO anchor 특징을 뺀253 입력을 두 축 선형 회귀에 사용한다. center·폭·정규화·후보·TRAIN2,598행·scale은 이전 바인딩과 같다.

정답은 각 물리 T/R 오차의 anchor 차이를 기존 TRAIN scale로 나눈 e에 `sign(e) log1p(abs(e))`를 적용한다. 유효 후보만 뺄셈하여 inf−inf를 피하고 anchor 입력·정답은 정확히0이다. invalid 입력·정답은0이며 all-invalid1행도 전체2,598 분모에 남는다. cached TRAIN 오류 외 원본 GT는 읽지 않았다.

목적식은 Huber δ=1을 행별 유효 후보와 두 축에 대해 평균한 뒤 전체 행 평균을 취하고 λ/2‖W‖²를 더한다. 독립 행·후보·축 반복 계산, Torch autograd와506방향 중앙차분을 대조했다. gradient 차분 최대 {toy['gradient_finite_difference_max']:.3g}, Hessian 차분 최대 {toy['Hessian_finite_difference_max']:.3g}이며 Hessian 최소 고유값은 {toy['Hessian_min_eigenvalue']:.9g}이다. |잔차|=1에서 값·gradient는 연속이지만 고전적 Hessian은 정의되지 않으므로 Hessian 검산은 경계에서 떨어진 합성 예제에 한정한다.

추론은 두 예측 축의 max를 원래 유효 후보 사이에서 최소화하며 기존 R0·hypothesis 동률 규칙을 유지한다. anchor 예측은0이므로 선택 점수는 예측상0 이하이지만 **실제 T/R 비악화 보장은 아니다.** 추론 함수에서 GT target 생성기를 금지해도 동작함을 확인했다. margin·safe GT mask·새 참조 오류를 넣지 않는다.

실제 TRAIN에서는 이전 RBF PREFIT의 phi253·raw feature·valid·unscaled 오류·정규화·anchor·scale 해시를 확인하고 signed target과 anchor 차분 입력을 독립 재구축했다. 실제 행에 임의 weight를 적용하거나 objective·선택률·T/R 개선을 시험하지 않았다. 모델별6개 해시는 학습 시작 전에 그대로 대조할 수 있도록 JSON에 기록했다.

본 결과는 학습 구현의 일관성 검증이며 개선 성과를 뜻하지 않는다. 고정 source45와 원래 SINGLE251 및 matched 두 묶음의5개 AND 조건은 별도 실행·판정 대상이다.

[검산 JSON](PREFIT_REVIEW.json) · [재사용한 고정 basis](../pallet_pose_anchor_rbf_20261001_v1/RBF_BASIS.json)
'''
    C.save(C.DOC/'PREFIT_REVIEW_KO.md',markdown);result['note']=C.bind(C.DOC/'PREFIT_REVIEW_KO.md')
    C.save(C.DOC/'PREFIT_REVIEW.json',result)
    print('SIGNED_AXES_PREFIT_PASS',C.bind(C.DOC/'PREFIT_REVIEW.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('selfcheck','write'))
    args=parser.parse_args();run(args.stage=='write')
