"""Independent solver-only Newton, fixed signed targets and Huber math audit.

selfcheck uses invented arrays only. Actual TRAIN inspection requires root's
explicit run authorization. The old fixed basis is authenticated and reused;
no centers/width are reselected, no actual-data weights or policies are tested.
"""
import argparse
import ast
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
SOLVER_RULE='BLOCK_GENERALIZED_NEWTON_ARMIJO'
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


def code_parity(T):
    previous=C.SIGNED_HERE/'convex_train.py'
    old=ast.parse(previous.read_text());new=ast.parse(Path(T.__file__).read_text())
    unchanged=('validate_anchor','difference_from_context','difference_inputs','signed_targets',
               'predict_axes','score_candidates','objective','hessian')
    for name in unchanged:
        left=next(node for node in old.body if isinstance(node,ast.FunctionDef) and node.name==name)
        right=next(node for node in new.body if isinstance(node,ast.FunctionDef) and node.name==name)
        assert ast.dump(left)==ast.dump(right),('OBJECTIVE_OR_RUNTIME_FUNCTION_CHANGED',name)
    aliases=('context_inputs','rbf_inputs','normalized_inputs','load_anchor','select_candidates')
    def assignment(tree,name):
        return next(node for node in tree.body if isinstance(node,ast.Assign) and
            any(isinstance(target,ast.Name) and target.id==name for target in node.targets))
    for name in aliases:
        assert ast.dump(assignment(old,name))==ast.dump(assignment(new,name)),name
    gate_functions={}
    for filename,names in (
        ('evaluate_source.py',('comparisons_for','chosen_pose','verify_source_reference_chain')),
        ('evaluate_real.py',('gates_for','combined_gates'))):
        left=ast.parse((C.SIGNED_HERE/filename).read_text())
        right=ast.parse((C.HERE/filename).read_text())
        for name in names:
            a=next(node for node in left.body if isinstance(node,ast.FunctionDef) and node.name==name)
            b=next(node for node in right.body if isinstance(node,ast.FunctionDef) and node.name==name)
            assert ast.dump(a)==ast.dump(b),('EVALUATION_RULE_CHANGED',filename,name)
        gate_functions[filename]=dict(functions=list(names),previous=C.bind(C.SIGNED_HERE/filename),current=C.bind(C.HERE/filename))
    assert (T.FEATURE_DIM,T.OUTPUT_DIM,T.HUBER_DELTA,T.LAMBDA)==(253,2,1.,1e-4)
    return dict(PASS=True,previous_code=C.bind(previous),current_code=C.bind(T.__file__),
        exact_AST_functions=list(unchanged),exact_AST_aliases=list(aliases),
        objective_features_targets_runtime_unchanged=True,evaluation_function_AST_parity=gate_functions,
        allowed_change='Solver, schema, solver metadata/certification/trace; no new data/objective/representation/selection criterion.')


def scalar_solver_objective(weight,x,valid,target):
    n,k,d=x.shape;assert weight.shape==(d,2)
    gradient=np.zeros_like(weight);pred=np.zeros_like(target);residual=np.zeros_like(target);loss=0.
    for i in range(n):
        active=np.flatnonzero(valid[i]).tolist()
        for j in active:
            coefficient=1/(n*len(active)*2)
            for axis in range(2):
                prediction=float(np.dot(x[i,j],weight[:,axis]))
                r=prediction-float(target[i,j,axis]);a=abs(r)
                pred[i,j,axis]=prediction;residual[i,j,axis]=r
                loss+=coefficient*(.5*r*r if a<=1 else a-.5)
                gradient[:,axis]+=coefficient*(r if a<=1 else np.sign(r))*x[i,j]
    penalty=.0001/2*float(np.sum(weight*weight))
    return dict(value=loss+penalty,Huber=loss,penalty=penalty,gradient=gradient+.0001*weight,
                prediction=pred,residual=residual)


def scalar_solver_blocks(x,valid,residual):
    n,k,d=x.shape;blocks=np.stack([np.eye(d)*1e-4,np.eye(d)*1e-4])
    for i in range(n):
        active=np.flatnonzero(valid[i]).tolist()
        for j in active:
            coefficient=1/(n*len(active)*2)
            for axis in range(2):
                if abs(residual[i,j,axis])<1:
                    blocks[axis]+=coefficient*np.outer(x[i,j],x[i,j])
    return blocks


def scalar_newton(x,valid,target,max_calls,max_iter):
    """Independent tiny-problem reference; never called on actual TRAIN data."""
    weight=np.zeros((x.shape[2],2),np.float64)
    current=scalar_solver_objective(weight,x,valid,target)
    history=[dict(weight=weight.copy(),value=current['value'],alpha=0.,accepted=True,
                  accepted_point_call_before=None,armijo_bound=None)]
    calls=1;iterations=0;accepted_call=1
    while True:
        gradient=current['gradient']
        linf=float(np.max(np.abs(gradient)));gap=float(np.sum(gradient*gradient)/(2e-4))
        if linf<=1e-8 and gap<=1e-6:
            success=True;break
        if iterations>=max_iter or calls>=max_calls:
            success=False;break
        blocks=scalar_solver_blocks(x,valid,current['residual'])
        direction=np.stack([np.linalg.solve(blocks[a],-gradient[:,a]) for a in range(2)],axis=1)
        slope=float(np.sum(gradient*direction));assert slope<0
        alpha=1.
        while True:
            if calls>=max_calls:
                return dict(success=False,weight=weight,final=current,objective_calls=calls,
                    iterations=iterations,final_accepted_call=accepted_call,last_evaluated_call=calls,history=history)
            trial=weight+alpha*direction
            evaluated=scalar_solver_objective(trial,x,valid,target);calls+=1
            bound=current['value']+1e-4*alpha*slope
            accepted=evaluated['value']<=bound
            history.append(dict(weight=trial.copy(),value=evaluated['value'],alpha=alpha,accepted=accepted,
                accepted_point_call_before=accepted_call,armijo_bound=bound))
            if accepted:
                weight=trial;current=evaluated;accepted_call=calls;iterations+=1;break
            alpha*=.5
    return dict(success=success,weight=weight,final=current,objective_calls=calls,
        iterations=iterations,final_accepted_call=accepted_call,last_evaluated_call=calls,history=history)


def synthetic_newton(T):
    expected_config=dict(method='block_generalized_newton_armijo',maxiter=1000,maxfun=2000,
        gradient_linf_tolerance=1e-8,initial_alpha=1.,backtrack_factor=.5,armijo_c1=1e-4,
        initialization='zeros',damping=0.,huber_kink_curvature=0.)
    assert T.solver_config()==expected_config
    valid=np.array([[1,1],[1,1],[0,0]],bool)
    x=np.zeros((3,2,3),np.float64);x[0,1]=[1,.2,0];x[1,1]=[-.5,1,.1]
    target=np.zeros((3,2,2),np.float64);target[0,1]=[3.,-2.];target[1,1]=[-1.5,2.5]
    cases=[]
    for max_calls,max_iter,expected_success in ((200,100,True),(1,100,False),(2,100,False),(200,1,False)):
        reference=scalar_newton(x,valid,target,max_calls,max_iter)
        rows=[];actual=T.newton_solve(x,valid,target,emit=rows.append,max_calls=max_calls,max_iter=max_iter)
        assert actual['success']==reference['success']==expected_success
        for key in ('objective_calls','iterations','final_accepted_call','last_evaluated_call'):
            assert actual[key]==reference[key],(key,actual[key],reference[key])
        np.testing.assert_allclose(actual['weight'],reference['weight'],rtol=1e-10,atol=1e-11)
        for key in ('value','Huber','penalty','gradient','prediction','residual'):
            np.testing.assert_allclose(actual['final'][key],reference['final'][key],rtol=1e-9,atol=1e-11)
        objective=[row for row in rows if row['event']=='objective']
        accepted=[row for row in rows if row['event']=='iteration']
        assert len(objective)==actual['objective_calls'] and len(accepted)==actual['iterations']
        assert [row['call'] for row in objective]==list(range(1,len(objective)+1))
        assert [row['iteration'] for row in accepted]==list(range(1,len(accepted)+1))
        for row,truth in zip(objective,reference['history']):
            np.testing.assert_allclose(row['objective'],truth['value'],rtol=1e-10,atol=1e-11)
            if row['call']>1:
                assert row['alpha']==truth['alpha']
                assert row['armijo_accepted']==truth['accepted']
                assert row['accepted_point_call_before']==truth['accepted_point_call_before']
                np.testing.assert_allclose(row['armijo_bound'],truth['armijo_bound'],rtol=1e-10,atol=1e-11)
        if max_calls==2:
            assert not reference['history'][1]['accepted']
            assert actual['final_accepted_call']==1 and actual['last_evaluated_call']==2
            np.testing.assert_array_equal(actual['weight'],0.)
        if actual['success']:
            assert max(abs(actual['final']['gradient'].ravel()))<=1e-8
            assert np.sum(actual['final']['gradient']**2)/(2e-4)<=1e-6
            assert any(not row['accepted'] for row in reference['history'][1:])
        cases.append(dict(max_calls=max_calls,max_iter=max_iter,success=actual['success'],
            objective_calls=actual['objective_calls'],iterations=actual['iterations'],
            final_accepted_call=actual['final_accepted_call'],last_evaluated_call=actual['last_evaluated_call'],
            rejected_trials=sum(not row['accepted'] for row in reference['history'][1:])))
    empty_valid=np.zeros((2,2),bool);empty_x=np.zeros((2,2,3));empty_y=np.zeros((2,2,2));empty_trace=[]
    empty=T.newton_solve(empty_x,empty_valid,empty_y,emit=empty_trace.append,max_calls=2,max_iter=1)
    assert empty['success'] and empty['iterations']==0 and empty['objective_calls']==1
    np.testing.assert_array_equal(empty['weight'],0.)
    assert empty['final']['Huber']==empty['final']['value']==0.
    assert len(empty_trace)==1
    # Same zero-curvature generalized choice at |residual|=1 as the frozen Hessian.
    kink_target=target.copy();kink_target[0,1]=[1.,-1.]
    atzero=scalar_solver_objective(np.zeros((3,2)),x,valid,kink_target)
    blocks=scalar_solver_blocks(x,valid,atzero['residual'])
    hessian=T.hessian(x,valid,atzero['residual'])
    np.testing.assert_allclose(hessian[::2,::2],blocks[0],rtol=1e-14,atol=1e-15)
    np.testing.assert_allclose(hessian[1::2,1::2],blocks[1],rtol=1e-14,atol=1e-15)
    return dict(PASS=True,solver_config=expected_config,cases=cases,
        independent_block_Newton_and_fixed_Armijo_exact_trace=True,
        rejected_trials_and_initial_evaluation_counted=True,
        final_state_is_last_accepted_not_last_rejected=True,
        success_requires_gradient_linf_and_gap=True,all_invalid_zero_loss_initial_success=True,
        generalized_kink_zero_curvature_preserved=True,synthetic_optimizer_runs_only=len(cases)+1,
        actual_TRAIN_optimizer_or_policy_probes=0,data_artifacts_opened=0)


def actual_review(T):
    assert not (C.RAW/'fits').exists(),'PREFIT must precede every new fit.'
    C.verify(C.read(C.SIGNED_DOC/'TRAIN_PROTOCOL_SHA.json'))
    signed_protocol=C.read(C.SIGNED_DOC/'TRAIN_PROTOCOL.json')
    signed_prefit_binding=signed_protocol['inputs']['prefit_review'];C.verify(signed_prefit_binding)
    signed_prefit=C.read(C.ROOT/signed_prefit_binding['path'])
    assert signed_prefit['complete'] and signed_prefit['PASS'] and signed_prefit['frames']==2598
    assert signed_prefit['available_anchor_rows']==2597 and signed_prefit['failed_rows_retained']==1
    assert not signed_prefit['VAL_quality_read'] and not signed_prefit['real_targets_read']
    assert signed_prefit['trainer']==C.bind(C.SIGNED_HERE/'convex_train.py')
    C.verify(C.read(C.RBF_DOC/'TRAIN_PROTOCOL_SHA.json'))
    old_protocol=C.read(C.RBF_DOC/'TRAIN_PROTOCOL.json')
    old_prefit_binding=old_protocol['inputs']['prefit_review'];C.verify(old_prefit_binding)
    prior=C.read(C.ROOT/old_prefit_binding['path']);assert prior['complete'] and prior['PASS']
    basis_binding=old_protocol['inputs']['rbf_basis'];C.verify(basis_binding)
    artifact=C.read(C.ROOT/basis_binding['path']);assert artifact['complete'] and artifact['PASS']
    assert prior['basis_SHA_bind']==basis_binding and artifact['source_TRAIN_only'] and not artifact['labels_read']
    assert not artifact['VAL_quality_read'] and not artifact['real_targets_read']
    basis=artifact['basis']
    assert signed_prefit['basis_SHA_bind']==basis_binding
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
        for key,value in hashes.items():assert signed_prefit['models'][model][key]==value
        results[model]=dict(**hashes,previous_signed_axes_six_hashes_exact=True,frames=2598,valid_candidates=int(valid.sum()),failed_rows=1,
            input_shape=list(x.shape),target_shape=list(target.shape),basis_SHA_bind=basis_binding,
            raw_features_sha=array_sha(raw),normalization_sha=parent['normalization_sha'],
            anchor_index_sha=array_sha(index),anchor_errors_sha=array_sha(anchor_errors),scale_sha=array_sha(parent['scale']),
            previous253_features_errors_and_validity_exact=True,independent_signed_targets_and_inputs_exact=True,
            all_invalid_row_retained=True,invalid_input_target_zero=True,anchor_input_target_zero=True,
            actual_data_objectives_or_policy_trials=0)
    inputs=dict(previous_signed_prefit=signed_prefit_binding,previous_signed_protocol=C.bind(C.SIGNED_DOC/'TRAIN_PROTOCOL.json'),
        parent_train_protocol=parent['binding'],previous_RBF_train_protocol=C.bind(C.RBF_DOC/'TRAIN_PROTOCOL.json'),
        previous_RBF_prefit=old_prefit_binding,basis=basis_binding,
        features=parent['protocol']['inputs']['features'],train_labels=parent['protocol']['inputs']['train_labels'],
        source_contract=parent['protocol']['inputs']['source_contract'],
        source_ids_sha=array_sha(parent['ids']),source_index_sha=array_sha(parent['source_index']),
        normalization_sha=parent['normalization_sha'],scale=parent['scale'].tolist(),scale_sha=array_sha(parent['scale']),
        anchor_index_sha=array_sha(index),anchor_errors_sha=array_sha(anchor_errors),
        failed_ids=parent['ids'][index<0].tolist())
    for key in ('source_ids_sha','source_index_sha','normalization_sha','scale_sha','anchor_index_sha','anchor_errors_sha'):
        assert inputs[key]==signed_prefit['inputs'][key],('PREVIOUS_SIGNED_INPUT_CHANGED',key)
    assert inputs['features']==signed_prefit['inputs']['features']
    assert inputs['train_labels']==signed_prefit['inputs']['train_labels']
    return results,inputs,basis_binding


def run(write):
    sys.dont_write_bytecode=True;guard()
    from . import convex_train as T
    assert T.OLD.READS is None
    assert T.LOSS_RULE==LOSS_RULE and T.TARGET_RULE==TARGET_RULE and T.INPUT_RULE==INPUT_RULE
    assert T.PREDICTION_RULE==PREDICTION_RULE and T.OUTPUT_DIM==2 and T.HUBER_DELTA==1.
    assert T.SOLVER_RULE==SOLVER_RULE and T.metadata()['solver_rule']==SOLVER_RULE
    code=C.bind(T.__file__)
    with threadpool_limits(limits=1):
        toy=synthetic(T)
        parity=code_parity(T)
        newton=synthetic_newton(T)
        if not write:
            print(json.dumps(dict(mathematics=toy,unchanged_functions=parity,solver=newton),indent=2));return
        models,inputs,basis_binding=actual_review(T)
    assert C.bind(T.__file__)==code,'TRAINER_CHANGED_DURING_REVIEW'
    result=dict(complete=True,PASS=True,created_at=C.now(),source_TRAIN_only=True,frames=2598,
        available_anchor_rows=2597,failed_rows_retained=1,feature_dim=253,raw_feature_dim=94,
        context_dim=189,rbf_dim=64,output_dim=2,
        solver_rule=SOLVER_RULE,solver_config=T.solver_config(),solver_only_change=True,
        feature_map=T.FEATURE_MAP,loss_rule=LOSS_RULE,target_rule=TARGET_RULE,input_rule=INPUT_RULE,
        prediction_rule=PREDICTION_RULE,huber_delta=1.,runtime_uses_margin=False,
        runtime_uses_reference_errors=False,runtime_safe_mask=False,
        basis_SHA_bind=basis_binding,models=models,inputs=inputs,synthetic=toy,
        unchanged_function_audit=parity,solver_synthetic=newton,
        trainer=code,reviewer=C.bind(__file__),source_TRAIN_cached_label_values_read=True,
        raw_source_reference_reads=0,VAL_quality_read=False,real_targets_read=False,
        new_fits=0,optimizer_steps=0,actual_data_objective_trials=0,actual_data_policy_probes=0,
        prior_weights_used=False,basis_reselected=False,bandwidth_changed=False,
        new_reference_metric_calculations=0,image_forwards=0,new_PnP_calls=0,
        read_paths=sorted(STATE['reads']),training_guard_read_paths=sorted(T.OLD.READS),
        method_success=False,goal_complete=False)
    for key,value in T.metadata().items():assert result[key]==value
    markdown=f'''# Solver만 변경한 Newton 학습 전 독립 검산

**PREFIT 검산 PASS. 새 fit·VAL·실사 성능 평가는 실행하지 않았다.** 직전 signed_axes의 고정 phi253에서 R0 GEO anchor를 뺀 입력, 두 축 signed log1p 타깃, 원래 valid mask, TRAIN2,598행, scale, 정규화, basis, Huber δ=1과 λ=1e-4를 유지한다. 독립 재구축한 모델별6개 해시가 이전 signed_axes PREFIT와 모두 같다.

변경은 L-BFGS-B에서 `{SOLVER_RULE}`로의 solver 교체다. 이전 모델의 거부 weight를 초기값으로 쓰지 않고253×2 영행렬에서 시작한다. 축별 generalized Newton block과 고정 Armijo c1=1e-4, 초기 step1, 축소율0.5를 적용한다. 최대1,000 update·2,000 objective 호출은 그대로이며 초기 평가와 거부된 모든 line-search trial도 호출 예산에 포함한다. 성공에는 gradient L∞≤1e-8과 기존 gap 상한≤1e-6을 모두 요구한다.

기존 objective·Hessian·타깃·특징 차분·추론 중요 함수의 AST를 대조했다. 스키마와 solver 메타데이터 외 목적식·선택 규칙의 변경은 없다. independent scalar Huber, Torch autograd와506방향 차분 검산의 gradient 최대 차이 {toy['gradient_finite_difference_max']:.3g}, Hessian 최대 차이 {toy['Hessian_finite_difference_max']:.3g}다. |잔차|=1의 고전적 Hessian은 정의되지 않으므로 미분 검산은 경계 밖에서 수행하며 solver는 기존 zero-curvature generalized convention을 유지한다.

합성 문제에서 독립 Newton 방향·고정 Armijo 순서·모든 평가 호출·성공 조건·예산 실패·all-invalid 분모를 검사했다. 이는 수치 구현 검산이며 실제 데이터의 수렴이나 T/R 개선을 미리 판정하지 않는다. 실제 TRAIN에서는 기존 캐시 입력·오류로 표현과 타깃만 재구축했으며 학습된 weight 적용, objective 시험, 선택률·성능 probe를 수행하지 않았다. 원본 GT·VAL 품질·실사 참조는 읽지 않았다.

추론은 이전과 같은 두 signed 예측의 max를 최소화한다. anchor 예측0은 실제 T/R 비악화 보장이 아니며 기존 source45 및 원래 SINGLE251·matched 두5조건의 AND 판정은 별도 실행 대상이다.

[검산 JSON](PREFIT_REVIEW.json) · [이전 동일 타깃 검산](../pallet_pose_signed_axes_20261001_v1/PREFIT_REVIEW.json) · [고정 basis](../pallet_pose_anchor_rbf_20261001_v1/RBF_BASIS.json)
'''
    C.save(C.DOC/'PREFIT_REVIEW_KO.md',markdown);result['note']=C.bind(C.DOC/'PREFIT_REVIEW_KO.md')
    C.save(C.DOC/'PREFIT_REVIEW.json',result)
    print('SIGNED_AXES_NEWTON_PREFIT_PASS',C.bind(C.DOC/'PREFIT_REVIEW.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('selfcheck','write'))
    args=parser.parse_args();run(args.stage=='write')
