"""Native DINO385 input extension with the fixed Q asymmetric signed-axis loss.

Import, runtime scoring, and selfcheck are pure. Only a separately sealed TRAIN
protocol permits reading frozen TRAIN targets and fitting four final models.
No old CE/risk-margin field is relabelled as the new Huber objective.
"""
from . import common as C
from . import direction_features as D
from . import appearance_features as A
from scripts.research.pallet_pose_signed_axes_asymmetric_20261001_v1 import convex_train as Q
from scripts.research.pallet_pose_signed_axes_sign_20261001_v1 import convex_train as PREV
from scripts.research.pallet_pose_signed_axes_direction_20261001_v1 import convex_train as PREVIOUS
from scripts.research.pallet_pose_union_selection_20261001_v1 import train as OLD
from scripts.research.pallet_pose_anchor_rbf_20261001_v1 import convex_train as RBF
from scripts.research.pallet_pose_anchor_rbf_20261001_v1 import basis as B
import argparse
import json
import os
import sys
from pathlib import Path
import time
import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
from types import SimpleNamespace
from scripts.research.pallet_pose_signed_axes_20261001_v1 import convex_train as SIGNED
import torch
from threadpoolctl import threadpool_limits

LAMBDA=1e-4
MAX_CALLS=2000
MAX_ITER=1000
GAP_MAX=1e-6
GRAD_LINF_MAX=1e-8
SOLVER_RULE='BLOCK_GENERALIZED_NEWTON_ARMIJO'
FEATURE_MAP=Q.FEATURE_MAP+'_native_dino385'
FEATURE_DIM=656
PREVIOUS_FEATURE_DIM=271
VISUAL_DIM=385
VISUAL_RULE='NATIVE_UNPADDED_SUPPORTED_PROJECTED_CORNERS8_DINO384_MEAN_PLUS_SUPPORT_FRACTION'
VISUAL_NORMALIZATION='R0_TRAIN_valid_float32_mean_std_floor_1e-6_then_float64_candidate_minus_R0_anchor'
BASE_FEATURE_DIM=253
DIRECTION_DIM=18
DIRECTION_RULE='BBOX_DIAGONAL_NORMALIZED_PROJECTED_MINUS_OBSERVED_XY9'
DIRECTION_NORMALIZATION='R0_TRAIN_valid_float32_mean_std_floor_1e-6_then_float64_candidate_minus_anchor'
RAW_FEATURE_DIM=94
CONTEXT_DIM=189
RBF_DIM=64
OUTPUT_DIM=2
HUBER_DELTA=1.
CHECKPOINT_SCHEMA='pallet_pose_signed_axes_visual_linear656x2_v1'
INPUT_RULE='DIRECTION271_DIFFERENCE_PLUS_NORMALIZED_NATIVE_DINO385_DIFFERENCE'
TARGET_RULE='SIGNED_LOG1P_NORMALIZED_TR_ANCHOR_EXCESS'
LOSS_RULE='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_ASYMMETRIC_HUBER_PLUS_SIGN_LOGISTIC'
UNDERPREDICTION_COEFFICIENT=1.
UNDERPREDICTION_COST=2.
OVERPREDICTION_COST=1.
UNDERPREDICTION_ZERO_CURVATURE=0.
SIGN_RULE='NONZERO_SIGNED_TARGET_AXES'
SIGN_COEFFICIENT=1.
PREDICTION_RULE='MAX_TWO_SIGNED_LOG1P_AXES'
PROTOCOL=C.DOC/'TRAIN_PROTOCOL.json'
HASH_KEYS=('signed_target_sha','input_difference_sha','base_context_sha','errors_sha','scaled_excess_sha','original_valid_sha')
EXTRA_HASH_KEYS=('direction_raw_sha','direction_difference_sha','extended_input_sha')
PREVIOUS_HASH_KEYS=HASH_KEYS+EXTRA_HASH_KEYS
VISUAL_HASH_KEYS=('visual_raw_sha','visual_difference_sha','visual_extended_input_sha')
ALL_HASH_KEYS=PREVIOUS_HASH_KEYS+VISUAL_HASH_KEYS
VISUAL_BINDING_KEYS=('visual_normalization_sha','visual_receipt_binding','visual_protocol_binding','visual_verification_binding')
EXPERTS=('R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3')
HYP=('long-face-front','short-face-front')
READS=None
_AUTHORIZED=object()

candidate_names=RBF.candidate_names
context_inputs=RBF.context_inputs
rbf_inputs=RBF.rbf_inputs
normalized_inputs=RBF.normalized_inputs
direction_difference=D.direction_difference
appearance_difference=A.appearance_difference
select_candidates=OLD.select_candidates


class ClosureBudgetExceeded(RuntimeError):
    pass


def metadata():
    return dict(feature_map=FEATURE_MAP,feature_dim=FEATURE_DIM,raw_feature_dim=RAW_FEATURE_DIM,
        base_feature_dim=BASE_FEATURE_DIM,direction_dim=DIRECTION_DIM,direction_rule=DIRECTION_RULE,
        direction_normalization=DIRECTION_NORMALIZATION,
        previous_feature_dim=PREVIOUS_FEATURE_DIM,visual_dim=VISUAL_DIM,
        visual_rule=VISUAL_RULE,visual_normalization=VISUAL_NORMALIZATION,
        context_dim=CONTEXT_DIM,rbf_dim=RBF_DIM,output_dim=OUTPUT_DIM,input_rule=INPUT_RULE,
        target_rule=TARGET_RULE,loss_rule=LOSS_RULE,prediction_rule=PREDICTION_RULE,huber_delta=HUBER_DELTA,
        runtime_uses_margin=False,runtime_uses_reference_errors=False,runtime_safe_mask=False,solver_rule=SOLVER_RULE,sign_rule=SIGN_RULE,sign_coefficient=SIGN_COEFFICIENT,
        underprediction_coefficient=UNDERPREDICTION_COEFFICIENT,underprediction_cost=UNDERPREDICTION_COST,
        overprediction_cost=OVERPREDICTION_COST,underprediction_zero_curvature=UNDERPREDICTION_ZERO_CURVATURE)


def solver_config():
    return dict(method='block_generalized_newton_armijo',maxiter=MAX_ITER,maxfun=MAX_CALLS,
        gradient_linf_tolerance=GRAD_LINF_MAX,initial_alpha=1.,backtrack_factor=.5,armijo_c1=1e-4,
        initialization='zeros',damping=0.,huber_kink_curvature=0.)


def validate_anchor(valid,anchor_index):
    valid,anchor_index=np.asarray(valid),np.asarray(anchor_index)
    assert valid.dtype==bool and valid.ndim==2 and valid.shape[1] in (2,4)
    assert anchor_index.shape==(len(valid),) and np.issubdtype(anchor_index.dtype,np.integer)
    assert np.isin(anchor_index,[-1,0,1]).all()
    present=valid.any(1)
    assert np.array_equal(anchor_index>=0,present),'ANCHOR_SUPPORT_CONTRACT'
    rows=np.flatnonzero(present)
    assert valid[rows,anchor_index[rows]].all()
    return rows


def difference_from_context(base_context,valid,anchor_index):
    base_context,valid=np.asarray(base_context),np.asarray(valid)
    assert base_context.dtype==np.float64 and base_context.shape==(*valid.shape,BASE_FEATURE_DIM)
    assert np.isfinite(base_context).all() and not base_context[~valid].any()
    rows=validate_anchor(valid,anchor_index)
    difference=np.zeros_like(base_context)
    frame,candidate=np.nonzero(valid)
    # Only usable candidates participate: absent rows do not even form a
    # reference subtraction, and invalid coordinates remain exactly zero.
    difference[frame,candidate]=base_context[frame,candidate]-base_context[frame,anchor_index[frame]]
    assert np.isfinite(difference).all() and not difference[~valid].any()
    assert not difference[rows,anchor_index[rows]].any()
    return difference


def difference_inputs(raw,valid,anchor_index,mean,std,basis,direction18,direction_mean,direction_std,
                      appearance385,visual_mean,visual_std):
    phi=rbf_inputs(raw,valid,anchor_index,mean,std,basis)
    old=difference_from_context(phi,valid,anchor_index)
    extra=direction_difference(direction18,valid,anchor_index,direction_mean,direction_std)
    visual=appearance_difference(appearance385,valid,anchor_index,visual_mean,visual_std)
    extended=np.concatenate([old,extra,visual],axis=2)
    assert extended.shape==(*np.asarray(valid).shape,FEATURE_DIM) and np.isfinite(extended).all()
    return extended


def signed_targets(errors,valid,anchor_errors,anchor_index,scale):
    errors,anchor_errors=np.asarray(errors,np.float64),np.asarray(anchor_errors,np.float64)
    valid,anchor_index=np.asarray(valid),np.asarray(anchor_index)
    scale=np.asarray(scale,np.float64)
    rows=validate_anchor(valid,anchor_index)
    assert errors.shape==(*valid.shape,2) and anchor_errors.shape==(len(valid),2)
    assert scale.shape==(2,) and np.isfinite(scale).all() and (scale>0).all()
    assert np.isfinite(errors[valid]).all() and (errors[valid]>=0).all()
    assert np.isposinf(errors[~valid]).all()
    assert np.isfinite(anchor_errors[rows]).all() and (anchor_errors[rows]>=0).all()
    assert np.isposinf(anchor_errors[~valid.any(1)]).all()
    np.testing.assert_array_equal(errors[rows,anchor_index[rows]],anchor_errors[rows])
    scaled=np.zeros(errors.shape,np.float64)
    target=np.zeros(errors.shape,np.float64)
    frame,candidate=np.nonzero(valid)
    scaled[frame,candidate]=(errors[frame,candidate]-anchor_errors[frame])/scale
    assert np.isfinite(scaled).all()
    target[valid]=np.sign(scaled[valid])*np.log1p(np.abs(scaled[valid]))
    assert np.isfinite(target).all() and not target[~valid].any()
    assert not scaled[rows,anchor_index[rows]].any() and not target[rows,anchor_index[rows]].any()
    return scaled,target


def predict_axes(checkpoint,raw_features,valid,anchor_index,direction18,appearance385):
    """Transformed signed T/R changes; invalid positions return exact zero."""
    assert checkpoint['schema']==CHECKPOINT_SCHEMA
    for key,value in metadata().items():assert checkpoint[key]==value,(key,checkpoint.get(key))
    assert checkpoint['normalization']=='old_float32_then_float64'
    assert checkpoint['names'][:2]==OLD.candidate_names('R0_ONLY',1)
    assert len(checkpoint['names'])==np.asarray(valid).shape[1] and len(set(checkpoint['names']))==len(checkpoint['names'])
    assert checkpoint['bias']==0. and checkpoint['lambda_l2']==LAMBDA
    assert checkpoint['rbf_basis_binding']==checkpoint['basis_SHA_bind']
    weight=np.asarray(checkpoint['weight'],np.float64)
    assert weight.shape==(FEATURE_DIM,OUTPUT_DIM) and np.isfinite(weight).all()
    mean18,std18=np.asarray(checkpoint['direction_mean'],np.float32),np.asarray(checkpoint['direction_std'],np.float32)
    assert mean18.shape==std18.shape==(DIRECTION_DIM,) and np.isfinite(mean18).all() and np.isfinite(std18).all()
    assert (std18>=np.float32(1e-6)).all()
    assert OLD.array_sha(np.stack([mean18,std18]))==checkpoint['direction_normalization_sha']
    assert isinstance(checkpoint['direction_receipt_binding'],dict)
    mean385,std385=np.asarray(checkpoint['visual_mean'],np.float32),np.asarray(checkpoint['visual_std'],np.float32)
    assert mean385.shape==std385.shape==(VISUAL_DIM,) and np.isfinite(mean385).all() and np.isfinite(std385).all()
    assert (std385>=np.float32(1e-6)).all()
    assert OLD.array_sha(np.stack([mean385,std385]))==checkpoint['visual_normalization_sha']
    for key in ('visual_receipt_binding','visual_protocol_binding','visual_verification_binding'):
        assert isinstance(checkpoint[key],dict)
    x=difference_inputs(raw_features,valid,anchor_index,checkpoint['mean'],checkpoint['std'],checkpoint['rbf_basis'],
        direction18,mean18,std18,appearance385,mean385,std385)
    prediction=np.einsum('nkd,da->nka',x,weight)
    assert np.isfinite(prediction).all() and not prediction[~valid].any()
    rows=validate_anchor(valid,anchor_index)
    assert not prediction[rows,anchor_index[rows]].any()
    return prediction


def score_candidates(checkpoint,raw_features,valid,anchor_index,direction18,appearance385):
    prediction=predict_axes(checkpoint,raw_features,valid,anchor_index,direction18,appearance385)
    return np.where(valid,prediction.max(axis=2),np.inf)


def objective(weight,x,valid,axis_target,ridge=LAMBDA):
    """Mean asymmetric Huber plus sign logistic over candidates/axes/all rows.

    All-invalid rows have zero data loss while remaining in the full N divisor.
    The shared matrix has fixed zero bias; the ridge covers every coefficient.
    Exactly zero targets omit the entire logistic term, including log(2).
    Huber combines Huber(e) and Huber(min(e,0)); each component is also returned.
    """
    weight,x=np.asarray(weight,np.float64),np.asarray(x,np.float64)
    valid,target=np.asarray(valid),np.asarray(axis_target,np.float64)
    assert weight.ndim==2 and weight.shape[1]==OUTPUT_DIM
    assert x.shape==(*valid.shape,len(weight)) and valid.dtype==bool and valid.ndim==2
    assert len(x)>0 and target.shape==(*valid.shape,OUTPUT_DIM)
    assert np.isfinite(weight).all() and np.isfinite(x).all() and np.isfinite(target).all()
    assert not x[~valid].any() and not target[~valid].any() and ridge>0
    prediction=np.einsum('nkd,da->nka',x,weight)
    residual=prediction-target
    absolute=np.abs(residual)
    quadratic=np.minimum(absolute,HUBER_DELTA)
    per_axis=.5*quadratic**2+HUBER_DELTA*(absolute-quadratic)
    coefficient=valid.astype(np.float64)/(len(x)*OUTPUT_DIM*np.maximum(valid.sum(1),1)[:,None])
    huber_symmetric=float(np.sum(per_axis*coefficient[:,:,None]))
    huber_underprediction=float(UNDERPREDICTION_COEFFICIENT*np.sum(np.where(residual<0.,per_axis,0.)*coefficient[:,:,None]))
    huber=huber_symmetric+huber_underprediction
    sign=np.sign(target)
    nonzero=(target!=0.) & valid[:,:,None]
    sign_logistic=float(SIGN_COEFFICIENT*np.sum(np.where(nonzero,np.logaddexp(0.,-sign*prediction),0.)*coefficient[:,:,None]))
    sign_gradient=np.where(nonzero,-sign*expit(-sign*prediction),0.)*SIGN_COEFFICIENT
    penalty=float(.5*ridge*np.sum(weight*weight))
    huber_gradient=np.clip(residual,-HUBER_DELTA,HUBER_DELTA)
    huber_gradient=huber_gradient+UNDERPREDICTION_COEFFICIENT*np.minimum(huber_gradient,0.)
    gradient=np.einsum('nk,nka,nkd->da',coefficient,huber_gradient+sign_gradient,x)+ridge*weight
    assert np.isfinite(prediction).all() and np.isfinite(gradient).all() and np.isfinite(huber+sign_logistic+penalty)
    return dict(value=huber+sign_logistic+penalty,Huber=huber,Huber_symmetric=huber_symmetric,
                Huber_underprediction=huber_underprediction,Sign_logistic=sign_logistic,penalty=penalty,gradient=gradient,
                prediction=prediction,residual=residual)


def hessian(x,valid,residual,axis_target,ridge=LAMBDA):
    """C-order curvature: base 1(|r|<1), extra 1(-1<r<0), plus logistic.

    At a kink this matrix is one generalized curvature choice, not a classical
    Hessian. At r=0 the additional one-sided term uses curvature zero.
    The differentiable strongly convex objective's gradient certificate
    does not require a second derivative there.
    """
    x,valid,residual=np.asarray(x,np.float64),np.asarray(valid),np.asarray(residual,np.float64)
    target=np.asarray(axis_target,np.float64)
    assert valid.dtype==bool and x.shape[:2]==valid.shape and residual.shape==(*valid.shape,2)
    assert target.shape==residual.shape and np.isfinite(target).all() and not target[~valid].any()
    assert np.isfinite(x).all() and np.isfinite(residual).all() and len(x)>0 and ridge>0
    prediction=residual+target;sign=np.sign(target)
    assert np.isfinite(prediction).all()
    logistic_curvature=np.where((target!=0.) & valid[:,:,None],expit(-sign*prediction)*expit(sign*prediction),0.)*SIGN_COEFFICIENT
    d=x.shape[2]
    coefficient=valid.astype(np.float64)/(len(x)*OUTPUT_DIM*np.maximum(valid.sum(1),1)[:,None])
    out=ridge*np.eye(d*OUTPUT_DIM,dtype=np.float64)
    for axis in range(OUTPUT_DIM):
        ra=residual[:,:,axis]
        curvature=(np.abs(ra)<HUBER_DELTA).astype(np.float64)
        curvature+=UNDERPREDICTION_COEFFICIENT*((ra>-HUBER_DELTA)&(ra<0.))
        active=coefficient*(curvature+logistic_curvature[:,:,axis])
        out[axis::OUTPUT_DIM,axis::OUTPUT_DIM]+=np.einsum('nk,nkd,nke->de',active,x,x,optimize=True)
    return out


def certificate(result,final,objective_calls,iterations):
    norm=float(np.linalg.norm(final['gradient']))
    linf=float(np.max(np.abs(final['gradient'])))
    gap=norm**2/(2*LAMBDA)
    passed=bool(result.success and np.isfinite(gap) and gap<=GAP_MAX and linf<=GRAD_LINF_MAX and objective_calls<=MAX_CALLS and iterations<=MAX_ITER)
    return dict(PASS=passed,optimizer_success=bool(result.success),gradient_l2=norm,
        gradient_linf=linf,gradient_linf_max=GRAD_LINF_MAX,solver_rule=SOLVER_RULE,
        gradient_l2_squared_over_2lambda=gap,max_gap_upper_bound=GAP_MAX,lambda_l2=LAMBDA,
        objective_value=final['value'],Huber=final['Huber'],L2_penalty=final['penalty'],
        Huber_symmetric=final['Huber_symmetric'],Huber_underprediction=final['Huber_underprediction'],
        underprediction_coefficient=UNDERPREDICTION_COEFFICIENT,underprediction_cost=UNDERPREDICTION_COST,
        overprediction_cost=OVERPREDICTION_COST,underprediction_zero_curvature=UNDERPREDICTION_ZERO_CURVATURE,
        Sign_logistic=final['Sign_logistic'],sign_rule=SIGN_RULE,sign_coefficient=SIGN_COEFFICIENT,
        loss_rule=LOSS_RULE,huber_delta=HUBER_DELTA,output_dim=OUTPUT_DIM,
        objective_calls=objective_calls,iterations=iterations,
        data_loss_definition='For each frame mean [Huber(e,1)+Huber(min(e,0),1)+1{target!=0}softplus(-sign(target)*prediction)], e=prediction-target, over original valid candidates and2 axes, then mean over all2598 rows. Fixed2:1 under/overprediction cost. Huber field is the sum of both Huber components. Sign coefficient1; zero-target axes omit logistic, not Huber; no-valid row contributes0.',
        theorem='For lambda-strongly-convex differentiable J, J(w)-min J <= ||grad J(w)||²/(2lambda). Applies also at C1 Huber kink.',
        scope='Numerical convergence of this fixed signed two-axis asymmetric Huber+sign-logistic+ridge objective, not a T/R performance certificate.')


def newton_solve(x,valid,axis_target,*,emit=None,max_calls=MAX_CALLS,max_iter=MAX_ITER):
    """Pure deterministic solve; all data supplied, no I/O, no warm start.

    emit receives every counted objective attempt and each accepted iteration.
    The returned final/weight always refer to the last accepted point, even if
    the last counted trial was rejected. Initial evaluation counts toward cap.
    Small caps are supported only for invented tests; fit_one uses fixed caps.
    """
    assert isinstance(max_calls,int) and 1<=max_calls<=MAX_CALLS
    assert isinstance(max_iter,int) and 0<=max_iter<=MAX_ITER
    x,valid,axis_target=np.asarray(x,np.float64),np.asarray(valid),np.asarray(axis_target,np.float64)
    assert x.ndim==3 and valid.shape==x.shape[:2] and valid.dtype==bool
    assert axis_target.shape==(*valid.shape,2) and len(x)>0
    settings=solver_config();weight=np.zeros((x.shape[2],2),np.float64)
    calls,iterations,accepted_call=0,0,0
    current=None
    def send(row):
        if emit is not None:emit(row)
    def stop(success,status,message):
        return dict(success=success,status=status,message=message,weight=weight.copy(),final=current,
            objective_calls=calls,iterations=iterations,final_accepted_call=accepted_call,last_evaluated_call=calls,
            solver_settings=settings)
    def evaluate(trial,phase,alpha,parent_call,parent_weight_sha,parent_objective,directional_derivative,armijo_bound):
        nonlocal calls
        assert calls<max_calls
        calls+=1
        base=dict(event='objective',call=calls,phase=phase,iteration=iterations+(phase=='trial'),alpha=alpha,
            accepted_point_call_before=parent_call,parent_weight_sha=parent_weight_sha,parent_objective=parent_objective,
            directional_derivative=directional_derivative,armijo_bound=armijo_bound,weight_sha=OLD.array_sha(trial))
        try:
            value=objective(trial,x,valid,axis_target)
            finite=(np.isfinite(value['value']) and np.isfinite(value['Huber']) and np.isfinite(value['penalty'])
                    and np.isfinite(value['gradient']).all() and np.isfinite(value['residual']).all())
            if not finite:raise FloatingPointError('NONFINITE_OBJECTIVE_OR_DERIVATIVE')
        except (AssertionError,FloatingPointError,OverflowError,ValueError) as exc:
            send(dict(base,evaluated=False,armijo_accepted=False,objective=None,Huber=None,Huber_symmetric=None,Huber_underprediction=None,Sign_logistic=None,L2_penalty=None,
                gradient_l2=None,gradient_linf=None,gradient_gap_upper_bound=None,error_type=type(exc).__name__,error=str(exc)))
            return None,False
        norm=float(np.linalg.norm(value['gradient']));linf=float(np.max(np.abs(value['gradient'])))
        accepted=bool(phase=='initial' or value['value']<=armijo_bound)
        send(dict(base,evaluated=True,armijo_accepted=accepted,objective=value['value'],Huber=value['Huber'],Sign_logistic=value['Sign_logistic'],
            Huber_symmetric=value['Huber_symmetric'],Huber_underprediction=value['Huber_underprediction'],
            L2_penalty=value['penalty'],gradient_l2=norm,gradient_linf=linf,gradient_gap_upper_bound=norm**2/(2*LAMBDA)))
        return value,accepted
    current,okay=evaluate(weight,'initial',0.,None,None,None,None,None)
    if current is None:return stop(False,'NONFINITE_INITIAL_OBJECTIVE','Initial evaluation failed; no accepted point exists.')
    assert okay;accepted_call=1
    while True:
        grad=current['gradient'];norm=float(np.linalg.norm(grad));linf=float(np.max(np.abs(grad)))
        if linf<=GRAD_LINF_MAX and norm**2/(2*LAMBDA)<=GAP_MAX:
            return stop(True,'CONVERGED','Gradient Linf and strong-convex gap bounds both satisfied at accepted point.')
        if iterations>=max_iter:return stop(False,'ITERATION_BUDGET_EXHAUSTED','Accepted-iteration budget reached; no extension.')
        if calls>=max_calls:return stop(False,'OBJECTIVE_BUDGET_EXHAUSTED','Objective-call budget reached; no extension.')
        try:
            full_h=hessian(x,valid,current['residual'],axis_target)
            if not np.isfinite(full_h).all():return stop(False,'NONFINITE_HESSIAN','Nonfinite generalized Hessian; no fallback.')
            direction=np.zeros_like(weight)
            for axis in range(2):
                block=full_h[axis::2,axis::2]
                factor=cho_factor(block,lower=True,check_finite=True)
                direction[:,axis]=cho_solve(factor,-grad[:,axis],check_finite=True)
        except (np.linalg.LinAlgError,ValueError,FloatingPointError) as exc:
            return stop(False,'SPD_SOLVE_FAILED',str(exc))
        if not np.isfinite(direction).all():return stop(False,'NONFINITE_DIRECTION','Nonfinite block solution; no fallback.')
        derivative=float(np.sum(grad*direction))
        if not np.isfinite(derivative) or derivative>=0:return stop(False,'NON_DESCENT_DIRECTION','Newton direction is not strictly descending; no fallback.')
        alpha=settings['initial_alpha'];parent_call=accepted_call;parent_sha=OLD.array_sha(weight);parent_objective=current['value']
        while True:
            if calls>=max_calls:return stop(False,'OBJECTIVE_BUDGET_EXHAUSTED','Objective-call cap reached during Armijo search; retain prior accepted point.')
            trial=weight+alpha*direction
            if not np.isfinite(trial).all():return stop(False,'NONFINITE_TRIAL_PARAMETERS','Nonfinite trial parameters; no fallback.')
            if np.array_equal(trial,weight):return stop(False,'ARMIJO_STEP_UNDERFLOW','No representable parameter update; no tolerance relaxation.')
            bound=parent_objective+settings['armijo_c1']*alpha*derivative
            if not np.isfinite(bound):return stop(False,'NONFINITE_ARMIJO_BOUND','Nonfinite Armijo bound; no fallback.')
            value,accepted=evaluate(trial,'trial',alpha,parent_call,parent_sha,parent_objective,derivative,bound)
            if value is None:return stop(False,'NONFINITE_TRIAL_OBJECTIVE','Trial evaluation failed; retain prior accepted point.')
            if accepted:
                weight=trial;current=value;accepted_call=calls;iterations+=1
                send(dict(event='iteration',iteration=iterations,objective_calls=calls,accepted_call=accepted_call,
                    alpha=alpha,armijo_bound=bound,armijo_accepted=True,parent_weight_sha=parent_sha,parent_objective=parent_objective,
                    directional_derivative=derivative,weight_sha=OLD.array_sha(weight),objective=current['value'],Huber=current['Huber'],Sign_logistic=current['Sign_logistic'],
                    Huber_symmetric=current['Huber_symmetric'],Huber_underprediction=current['Huber_underprediction'],
                    L2_penalty=current['penalty'],gradient_l2=float(np.linalg.norm(current['gradient'])),
                    gradient_linf=float(np.max(np.abs(current['gradient']))),
                    gradient_gap_upper_bound=float(np.linalg.norm(current['gradient']))**2/(2*LAMBDA)))
                break
            alpha*=settings['backtrack_factor']
            if alpha==0 or not np.isfinite(alpha):return stop(False,'ARMIJO_ALPHA_UNDERFLOW','Backtracking alpha underflow; no fallback.')


def install_training_guard():
    """TRAIN cache allowlist with no legacy permanent NPZ guard installed."""
    global READS
    if READS is not None:return
    READS=set();sys.dont_write_bytecode=True
    arrays={C.PARENT_RAW/'SOURCE_FEATURES.npz',C.PARENT_RAW/'SOURCE_TRAIN_LABELS.npz',
            C.DIRECTION_RAW/'TRAIN_DIRECTIONS.npz',C.VISUAL_RAW/'TRAIN_APPEARANCE.npz'}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();name=str(path);mode,flags=args[1:3]
        writing=((isinstance(mode,str) and any(c in mode for c in 'wax+')) or
            (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))))
        if writing:
            assert path.is_relative_to(C.DOC) or path.is_relative_to(C.RAW) or path==Path('/dev/null'),('TRAIN_WRITE_SCOPE',name)
            return
        assert not any(t in name for t in ('/data/evaluation/','/real_gt_v2/','/annotations/',
            'SOURCE_VAL','VAL_LABEL','VAL_QUALITY','POSE_METRICS','REAL_CHOICES','REAL_FEATURES',
            'REAL_RESULTS','GEOMETRY_RESOLVED_POSE_GT','GEOMETRY_SIDETABLE','SOURCE_MANIFEST',
            'SYNTH_LABELS','SYNTH_RECORDS','/model_parameters/')),('TRAIN_REFERENCE_OR_QUALITY_DENIED',name)
        assert path.suffix.lower() not in ('.png','.jpg','.jpeg','.pt','.pth','.onnx'),('TRAIN_IMAGE_WEIGHT_DENIED',name)
        if '/fits/' in name:assert path.is_relative_to(C.RAW/'fits'),('PREVIOUS_WEIGHT_DENIED',name)
        if path.suffix=='.npz':assert path in arrays,('TRAIN_UNAPPROVED_ARRAY',name)
        assert path.suffix!='.npy',('TRAIN_RAW_TOKEN_READ_DENIED',name)
        if path.is_relative_to(C.ROOT):READS.add(str(path.relative_to(C.ROOT)))
    sys.addaudithook(hook)


def bound_read(binding):
    C.verify(binding);return C.read(C.ROOT/binding['path'])


def load_parent():
    install_training_guard()
    seal = C.U.read(C.U.DOC / 'TRAIN_PROTOCOL_SHA.json')
    protocol = OLD.checked_binding(seal, (C.PARENT_DOC/'TRAIN_PROTOCOL.json'))
    expected = dict(schema='pallet_pose_union_train_v1', arms=list(OLD.ARMS), seeds=list(OLD.SEEDS),
                    train_rows=OLD.ROWS, feature_dim=OLD.DIM, epochs=OLD.EPOCHS, batch_size=OLD.BATCH,
                    updates_per_fit=OLD.UPDATES, device='cpu', max_fits=6)
    for key, value in expected.items():
        assert protocol[key] == value, (key, protocol.get(key), value)
    assert protocol['optimizer'] == dict(name='AdamW', lr=.001, weight_decay=.0001,
                                        betas=[.9, .999], eps=1e-8)
    assert protocol['normalization'] == dict(source='R0_eligible_train_valid_candidates', std_floor=1e-6)
    code = {b['path']: b for b in protocol['codes']}
    for path in (Path(OLD.__file__), Path(C.U.__file__)):
        assert str(path.resolve().relative_to(C.U.ROOT)) in code
    for binding in code.values():
        C.U.verify(binding)
    inputs = protocol['inputs']
    for binding in inputs.values():
        C.U.verify(binding)
    features = C.U.RAW / 'SOURCE_FEATURES.npz'
    labels = C.U.RAW / 'SOURCE_TRAIN_LABELS.npz'
    OLD.checked_binding(inputs['features'], features)
    OLD.checked_binding(inputs['train_labels'], labels)
    gate = OLD.checked_binding(inputs['train_gate'], C.U.DOC / 'SOURCE_TRAIN_GATE.json')
    feasible = OLD.checked_binding(inputs['feasibility_protocol'], C.U.DOC / 'TRAIN_FEASIBILITY_PROTOCOL.json')
    OLD.checked_binding(C.U.read(C.U.DOC / 'TRAIN_FEASIBILITY_PROTOCOL_SHA.json'), C.U.DOC / 'TRAIN_FEASIBILITY_PROTOCOL.json')
    predlock = OLD.checked_binding(inputs['source_predictions_lock'], C.U.DOC / 'SOURCE_PREDICTIONS_LOCK.json')
    assert gate['complete'] and gate['PASS'] and gate['scope'] == 'C2_TRAIN_ONLY'
    assert not gate['VAL_quality_scored'] and not gate['C1_quality_scored']
    assert gate['frames'] == OLD.ROWS and gate['fits'] == 0
    assert gate['labels'] == inputs['train_labels'] and gate['protocol'] == inputs['feasibility_protocol']
    assert feasible['scope'] == 'C2_TRAIN_ONLY' and feasible['eligible_counts'] == dict(TRAIN=OLD.ROWS, VAL=1024)
    assert feasible['scalar_cost'] == 'max(T/sT,R/sR)'
    assert feasible['tie_rule'] == 'exact_cost_then_pareto_then_R0_then_hypothesis'
    assert feasible['scale'] == 'median_R0_GEO_C2_TRAIN'
    assert feasible['median_strict'] and feasible['failure_no_increase'] and feasible['p90_ratio_max'] == 1.05
    featurelock = OLD.checked_binding(gate['feature_lock'], C.U.DOC / 'SOURCE_FEATURE_LOCK.json')
    assert gate['feature_lock'] == inputs['feature_lock']
    assert featurelock['complete'] and not featurelock['source_targets_read']
    assert not featurelock['real_targets_read'] and not featurelock['VAL_quality_scored']
    assert featurelock['features'] == inputs['features'] and featurelock['predictions'] == inputs['source_predictions_lock']
    assert featurelock['models'] == list(C.U.MODELS) and featurelock['hypothesis_names'] == list(HYP)
    assert len(featurelock['feature_names']) == OLD.DIM
    assert predlock['complete'] and predlock['models'] == list(C.U.MODELS) and predlock['frames'] == 5120
    assert predlock['protocol'] == featurelock['protocol'] == feasible['source_protocol']
    OLD.checked_binding(feasible['source_protocol'], C.U.DOC / 'SOURCE_PROTOCOL.json')
    amendment = OLD.checked_binding(inputs['runtime_amendment'], C.U.DOC / 'SOURCE_RUNTIME_AMENDMENT_01.json')
    OLD.checked_binding(C.U.read(C.U.DOC / 'SOURCE_RUNTIME_AMENDMENT_01_SHA.json'), C.U.DOC / 'SOURCE_RUNTIME_AMENDMENT_01.json')
    assert amendment['complete'] and amendment['original_protocol'] == predlock['protocol']
    assert predlock['runtime_amendment'] == inputs['runtime_amendment']
    contract = OLD.checked_binding(gate['source_contract'], C.U.DOC / 'SOURCE_CONTRACT.json')
    assert gate['source_contract'] == feasible['source_contract'] == inputs['source_contract']
    assert contract['complete'] and contract['status'] == 'PASS'
    eligible = contract['fit_eligibility']
    assert eligible['eligible_counts'] == dict(TRAIN=OLD.ROWS, VAL=1024)
    # Do not recursively verify the gate/contract: their raw source-reference
    # bindings contain broader-label NPZs forbidden to this training process.
    fkeys = {'ids', 'split', 'hypothesis_names'} | {m + s for m in C.U.MODELS for s in ('_geo', '_valid', '_GEO_index')}
    lkeys = {'ids', 'source_index', 'eligible_train_mask', 'hypothesis_names', 'sT_cm', 'sR_deg'} | {
        m + s for m in C.U.MODELS for s in ('_T_cm', '_R_deg', '_GEO_T_cm', '_GEO_R_deg')}
    with np.load(features, allow_pickle=False) as data:
        assert set(data.files) == fkeys
        f = {k: data[k] for k in data.files}
    with np.load(labels, allow_pickle=False) as data:
        assert set(data.files) == lkeys
        label = {k: data[k] for k in data.files}
    assert len(f['ids']) == len(set(f['ids'].tolist())) == 5120
    assert f['ids'].ndim == f['split'].ndim == 1 and len(f['split']) == 5120
    assert int(np.sum(f['split'] == 'TRAIN')) == 4096 and int(np.sum(f['split'] == 'VAL')) == 1024
    assert f['hypothesis_names'].tolist() == label['hypothesis_names'].tolist() == list(HYP)
    idx, mask = label['source_index'], label['eligible_train_mask']
    assert idx.shape == (OLD.ROWS,) and np.issubdtype(idx.dtype, np.integer)
    assert mask.shape == (5120,) and mask.dtype == bool
    np.testing.assert_array_equal(idx, np.flatnonzero(mask))
    np.testing.assert_array_equal(f['ids'][idx], label['ids'])
    assert np.all(f['split'][idx] == 'TRAIN') and len(set(label['ids'].tolist())) == OLD.ROWS
    assert set(label['ids'].tolist()) == set(eligible['eligible_ids']['TRAIN'])
    assert set(label['ids'].tolist()).isdisjoint(eligible['eligible_ids']['VAL'])
    scale = np.array([label['sT_cm'].item(), label['sR_deg'].item()], np.float64)
    assert np.isfinite(scale).all() and (scale > 0).all()
    errors, feature, valid = {}, {}, {}
    for model in C.U.MODELS:
        x, v = f[model + '_geo'], f[model + '_valid']
        assert x.shape == (5120, 2, OLD.DIM) and x.dtype == np.float32
        assert v.shape == (5120, 2) and v.dtype == bool and np.isfinite(x[v]).all()
        feature[model], valid[model] = x[idx], v[idx]
        err = np.stack([label[model + '_T_cm'], label[model + '_R_deg']], axis=-1)
        assert err.shape == (OLD.ROWS, 2, 2)
        assert np.isfinite(err[v[idx]]).all() and (err[v[idx]] >= 0).all()
        assert np.isposinf(err[~v[idx]]).all()
        errors[model] = err
        op = np.stack([label[model + '_GEO_T_cm'], label[model + '_GEO_R_deg']], axis=-1)
        assert op.shape == (OLD.ROWS, 2) and not np.isnan(op).any() and not np.isneginf(op).any()
        assert (op >= 0).all() and np.all(np.isfinite(op).all(1) == np.isfinite(op).any(1))
        if model == 'R0':
            np.testing.assert_array_equal(scale, [OLD.source_median(op[:, j]) for j in range(2)])
    vals = feature['R0'][valid['R0']]
    assert vals.shape[1] == OLD.DIM and len(vals) > 1
    mean, std = vals.mean(0), np.maximum(vals.std(0), np.float32(1e-6))
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    normalization_sha = OLD.array_sha(np.stack([mean, std]))
    return dict(_authorized=None, protocol=protocol, binding=C.U.bind((C.PARENT_DOC/'TRAIN_PROTOCOL.json')), ids=label['ids'], source_index=idx,
                features=feature, valid=valid, errors=errors, scale=scale, mean=mean, std=std,
                normalization_sha=normalization_sha)

def load_anchor(parent):
    """Read only the two frozen source caches permitted by this phase's guard."""
    bindings={key:parent['protocol']['inputs'][key] for key in ('features','train_labels')}
    for binding in bindings.values():C.verify(binding)
    with np.load(C.ROOT/bindings['features']['path'],allow_pickle=False) as z:
        index=z['R0_GEO_index'][parent['source_index']]
        np.testing.assert_array_equal(z['ids'][parent['source_index']],parent['ids'])
        np.testing.assert_array_equal(z['R0_valid'][parent['source_index']],parent['valid']['R0'])
    with np.load(C.ROOT/bindings['train_labels']['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],parent['ids'])
        errors=np.stack([z['R0_GEO_T_cm'],z['R0_GEO_R_deg']],axis=-1)
    assert index.dtype==np.int64 and index.shape==(2598,) and errors.shape==(2598,2) and errors.dtype==np.float64
    finite=np.isfinite(errors).all(1)
    assert np.array_equal(finite,np.isfinite(errors).any(1)) and np.isposinf(errors[~finite]).all() and (errors>=0).all()
    rows=validate_anchor(parent['valid']['R0'],index)
    assert np.array_equal(index>=0,finite) and int((~finite).sum())==1
    np.testing.assert_array_equal(parent['errors']['R0'][rows,index[rows]],errors[rows])
    assert np.array_equal(np.logical_or.reduce([parent['valid'][m].any(1) for m in EXPERTS]),finite)
    return dict(index=index,errors=errors,finite=finite,index_sha=OLD.array_sha(index),errors_sha=OLD.array_sha(errors),
                finite_sha=OLD.array_sha(finite),bindings=bindings)


def load_inputs():
    # Resolve native BLAS/threadpool metadata before installing the read guard.
    # The immediately restored limit only primes library discovery, not the data.
    with threadpool_limits(limits=1):pass
    install_training_guard()
    protocol=C.protocol('TRAIN_PROTOCOL')
    expected=dict(**metadata(),train_rows=2598,lambda_l2=LAMBDA,bias=0.,normalization='old_float32_then_float64',max_fits=4)
    for key,value in expected.items():assert protocol[key]==value,(key,protocol.get(key))
    assert protocol['solver']==solver_config()
    assert protocol['certificate']==dict(optimizer_success=True,gap_upper_bound_max=GAP_MAX,gradient_linf_max=GRAD_LINF_MAX)
    assert protocol['models']==list(C.MODEL_NAMES)
    prefit_path=C.DOC/'PREFIT_REVIEW.json'
    assert protocol['inputs']['prefit_review']==C.bind(prefit_path)
    prefit=C.read(prefit_path)
    assert prefit['complete'] and prefit['PASS']
    for key,value in metadata().items():assert prefit[key]==value,(key,prefit.get(key))
    assert set(prefit['models'])==set(C.MODEL_NAMES)
    parent=load_parent()
    bound={b['path']:b for b in protocol['inputs'].values()}
    for binding in [parent['binding'],*parent['protocol']['inputs'].values()]:assert bound.get(binding['path'])==binding
    code={b['path']:b for b in protocol['codes']}
    for path in (Path(__file__),Path(C.__file__),Path(OLD.__file__),Path(RBF.__file__),Path(B.__file__),Path(SIGNED.__file__),Path(PREV.__file__),Path(PREVIOUS.__file__),Path(Q.__file__),Path(D.__file__),Path(A.__file__)):
        assert str(path.resolve().relative_to(C.ROOT)) in code
    basis_path=C.RBF_DOC/'RBF_BASIS.json';basis_binding=C.bind(basis_path)
    assert protocol['inputs']['rbf_basis']==basis_binding and prefit['basis_SHA_bind']==basis_binding
    basis_artifact=C.read(basis_path)
    assert basis_artifact['complete'] and basis_artifact['PASS'] and basis_artifact['schema']==B.BASIS_SCHEMA
    assert basis_artifact['source_TRAIN_only'] and not basis_artifact['labels_read']
    assert not basis_artifact['VAL_quality_read'] and not basis_artifact['real_targets_read']
    assert basis_artifact['protocol']==protocol['inputs']['basis_protocol']
    assert basis_artifact['normalization_sha']==parent['normalization_sha']
    assert basis_artifact['source_ids_sha']==OLD.array_sha(parent['ids'])
    assert basis_artifact['source_index_sha']==OLD.array_sha(parent['source_index'])
    np.testing.assert_array_equal(np.asarray(basis_artifact['mean'],np.float32),parent['mean'])
    np.testing.assert_array_equal(np.asarray(basis_artifact['std'],np.float32),parent['std'])
    B.validate_basis(basis_artifact['basis'],parent['mean'],parent['std'])
    anchor=load_anchor(parent)
    assert basis_artifact['anchor_index_sha']==anchor['index_sha']
    inputs=protocol['inputs']
    expected_paths=dict(direction_features=C.DIRECTION_RAW/'TRAIN_DIRECTIONS.npz',
        direction_receipt=C.DIRECTION_DOC/'FEATURE_AUDIT.json',direction_protocol=C.DIRECTION_DOC/'AUDIT_PROTOCOL.json',
        direction_representation=C.DIRECTION_DOC/'REPRESENTATION_AUDIT.json',
        direction_verification=C.DIRECTION_DOC/'VERIFICATION.json',
        previous_sign_prefit=C.SIGN_DOC/'PREFIT_REVIEW.json',previous_sign_protocol=C.SIGN_DOC/'TRAIN_PROTOCOL.json',
        previous_direction_prefit=C.PREVIOUS_DOC/'PREFIT_REVIEW.json',previous_direction_protocol=C.PREVIOUS_DOC/'TRAIN_PROTOCOL.json',
        previous_asymmetric_prefit=C.Q_DOC/'PREFIT_REVIEW.json',previous_asymmetric_protocol=C.Q_DOC/'TRAIN_PROTOCOL.json',
        visual_features=C.VISUAL_RAW/'TRAIN_APPEARANCE.npz',visual_receipt=C.VISUAL_DOC/'TRAIN_APPEARANCE_INPUTS.json',
        visual_protocol=C.VISUAL_DOC/'INPUT_PROTOCOL.json',visual_verification=C.VISUAL_DOC/'INPUT_VERIFICATION.json')
    for key,path in expected_paths.items():assert inputs[key]==C.bind(path),(key,path)
    receipt=bound_read(inputs['direction_receipt']);previous=bound_read(inputs['previous_sign_prefit'])
    previous_protocol=bound_read(inputs['previous_sign_protocol'])
    C.verify(C.read(C.SIGN_DOC/'TRAIN_PROTOCOL_SHA.json'))
    assert previous_protocol['inputs']['prefit_review']==inputs['previous_sign_prefit']
    for key,value in PREV.metadata().items():assert previous_protocol[key]==value
    for key,value in previous_protocol['inputs'].items():
        if key!='prefit_review':assert inputs[key]==value,('PREVIOUS_INPUT_BINDING',key)
    previous_direction=bound_read(inputs['previous_direction_prefit'])
    previous_direction_protocol=bound_read(inputs['previous_direction_protocol'])
    C.verify(C.read(C.PREVIOUS_DOC/'TRAIN_PROTOCOL_SHA.json'))
    assert previous_direction['complete'] and previous_direction['PASS']
    assert previous_direction_protocol['inputs']['prefit_review']==inputs['previous_direction_prefit']
    for key,value in PREVIOUS.metadata().items():
        assert previous_direction_protocol[key]==value and previous_direction[key]==value
    for key,value in previous_direction_protocol['inputs'].items():
        if key!='prefit_review':assert inputs[key]==value,('PREVIOUS_DIRECTION_INPUT_BINDING',key)
    previous_asymmetric=bound_read(inputs['previous_asymmetric_prefit'])
    previous_asymmetric_protocol=bound_read(inputs['previous_asymmetric_protocol'])
    C.verify(C.read(C.Q_DOC/'TRAIN_PROTOCOL_SHA.json'))
    assert previous_asymmetric['complete'] and previous_asymmetric['PASS']
    assert previous_asymmetric_protocol['inputs']['prefit_review']==inputs['previous_asymmetric_prefit']
    for key,value in Q.metadata().items():
        assert previous_asymmetric_protocol[key]==value and previous_asymmetric[key]==value
    for key,value in previous_asymmetric_protocol['inputs'].items():
        if key!='prefit_review':assert inputs[key]==value,('PREVIOUS_ASYMMETRIC_INPUT_BINDING',key)
    assert receipt['complete'] and receipt['PASS'] and previous['complete'] and previous['PASS']
    assert receipt['directions']==inputs['direction_features'] and receipt['protocol']==inputs['direction_protocol']
    assert receipt['source_TRAIN_only'] and not receipt['source_labels_read'] and not receipt['VAL_quality_read'] and not receipt['real_targets_read']
    audit=bound_read(inputs['direction_representation']);verification=bound_read(inputs['direction_verification'])
    assert audit['complete'] and audit['PASS'] and verification['complete'] and verification['PASS']
    assert verification['audited_representation']==inputs['direction_representation'] and verification['audited_feature_receipt']==inputs['direction_receipt']
    assert audit['inputs']['sign_prefit']==inputs['previous_sign_prefit']
    assert audit['inputs']['direction_features']==inputs['direction_features'] and audit['inputs']['direction_receipt']==inputs['direction_receipt']
    with np.load(C.ROOT/inputs['direction_features']['path'],allow_pickle=False) as z:
        expected_keys={'ids','source_index','anchor_index'}|{m+s for m in EXPERTS for s in ('_direction18','_valid')}
        assert set(z.files)==expected_keys
        np.testing.assert_array_equal(z['ids'],parent['ids']);np.testing.assert_array_equal(z['source_index'],parent['source_index'])
        np.testing.assert_array_equal(z['anchor_index'],anchor['index'])
        direction={m:z[m+'_direction18'].copy() for m in EXPERTS}
        for m in EXPERTS:np.testing.assert_array_equal(z[m+'_valid'],parent['valid'][m])
    for m in EXPERTS:
        raw18=direction[m];valid=parent['valid'][m]
        assert raw18.dtype==np.float32 and raw18.shape==(2598,2,18) and np.isfinite(raw18).all() and not raw18[~valid].any()
        assert receipt['models'][m]['direction_sha']==OLD.array_sha(raw18)
        assert receipt['models'][m]['valid_sha']==OLD.array_sha(valid)
    values=direction['R0'][parent['valid']['R0']];assert values.shape==(5194,18)
    mean18,std18=values.mean(0),np.maximum(values.std(0),np.float32(1e-6))
    norm=receipt['normalization'];norm_sha=OLD.array_sha(np.stack([mean18,std18]))
    assert norm['source']=='R0_eligible_train_valid_candidates' and norm['valid_candidate_count']==5194 and norm['std_floor']==1e-6
    np.testing.assert_array_equal(mean18,np.asarray(norm['mean18'],np.float32))
    np.testing.assert_array_equal(std18,np.asarray(norm['std18'],np.float32))
    assert norm['array_sha']==norm_sha and norm['mean_sha']==OLD.array_sha(mean18) and norm['std_sha']==OLD.array_sha(std18)
    assert prefit['direction_receipt_binding']==inputs['direction_receipt'] and prefit['direction_normalization_sha']==norm_sha
    assert protocol['direction_receipt_binding']==inputs['direction_receipt'] and protocol['direction_normalization_sha']==norm_sha
    np.testing.assert_array_equal(np.asarray(protocol['direction_mean'],np.float32),mean18)
    np.testing.assert_array_equal(np.asarray(protocol['direction_std'],np.float32),std18)
    assert prefit['normalization_sha']==parent['normalization_sha'] and prefit['anchor_index_sha']==anchor['index_sha']
    assert audit['normalization18']['sha']==norm_sha and audit['normalization94_sha']==parent['normalization_sha']
    for name,value in [('source_ids_sha',parent['ids']),('source_index_sha',parent['source_index']),
                       ('anchor_index_sha',anchor['index']),('anchor_errors_sha',anchor['errors']),('scale_sha',parent['scale'])]:
        assert previous['inputs'][name]==OLD.array_sha(value),name
    visual_receipt=bound_read(inputs['visual_receipt'])
    visual_protocol=bound_read(inputs['visual_protocol'])
    visual_verification=bound_read(inputs['visual_verification'])
    assert visual_receipt['complete'] and visual_receipt['input_construction_pass'] and visual_receipt['source_TRAIN_only']
    assert visual_receipt['protocol']==inputs['visual_protocol'] and visual_receipt['descriptors']==inputs['visual_features']
    assert visual_protocol['descriptor_rule']==VISUAL_RULE and visual_receipt['descriptor_rule']==VISUAL_RULE
    assert visual_protocol['frames']==visual_receipt['frames']==2598 and visual_protocol['output_dim']==VISUAL_DIM
    assert visual_receipt['original_candidate_validity_preserved'] and visual_receipt['original_allinvalid_rows']==1
    assert visual_receipt['fits_executed']==visual_receipt['new_image_forwards']==visual_receipt['new_PnP_solves']==0
    assert not visual_receipt['source_label_values_read'] and not visual_receipt['source_VAL_features_extracted'] and not visual_receipt['real_features_extracted']
    assert visual_verification['complete'] and visual_verification['PASS'] and visual_verification['source_TRAIN_only']
    assert visual_verification['protocol']==inputs['visual_protocol'] and visual_verification['descriptors']==inputs['visual_features']
    assert visual_verification['public_input_receipt']==inputs['visual_receipt']
    assert visual_verification['input_receipt']['sha256']==inputs['visual_receipt']['sha256']
    assert visual_verification['input_receipt']['bytes']==inputs['visual_receipt']['bytes']
    assert visual_verification['normalization_exact'] and visual_verification['verification_PASS']
    with np.load(C.ROOT/inputs['visual_features']['path'],allow_pickle=False) as z:
        expected_visual={'ids','source_index','anchor_index','crop_matrices','mean385','std385'}|{
            m+s for m in EXPERTS for s in ('_appearance385','_valid','_support8')}
        assert set(z.files)==expected_visual
        np.testing.assert_array_equal(z['ids'],parent['ids']);np.testing.assert_array_equal(z['source_index'],parent['source_index'])
        np.testing.assert_array_equal(z['anchor_index'],anchor['index'])
        visual={m:z[m+'_appearance385'].copy() for m in EXPERTS}
        for m in EXPERTS:
            np.testing.assert_array_equal(z[m+'_valid'],parent['valid'][m])
            np.testing.assert_array_equal(z[m+'_appearance385'][:,:,-1],z[m+'_support8'].sum(2).astype(np.float32)/8)
            assert visual_receipt['models'][m]['support_sha']==OLD.array_sha(z[m+'_support8'])
        stored_mean,stored_std=z['mean385'].copy(),z['std385'].copy()
    for m in EXPERTS:
        assert visual[m].dtype==np.float32 and visual[m].shape==(2598,2,VISUAL_DIM)
        assert np.isfinite(visual[m]).all() and not visual[m][~parent['valid'][m]].any()
        assert visual_receipt['models'][m]['descriptor_sha']==OLD.array_sha(visual[m])
        assert visual_receipt['models'][m]['valid_sha']==OLD.array_sha(parent['valid'][m])
    vv=visual['R0'][parent['valid']['R0']];assert vv.shape==(5194,VISUAL_DIM)
    visual_mean,visual_std=vv.mean(0),np.maximum(vv.std(0),np.float32(1e-6))
    assert visual_mean.dtype==visual_std.dtype==np.float32
    visual_norm_sha=OLD.array_sha(np.stack([visual_mean,visual_std]))
    visual_norm=visual_receipt['normalization']
    assert visual_norm['source']=='R0_original_valid_TRAIN_candidates' and visual_norm['count']==5194
    assert visual_norm['dtype']=='float32' and visual_norm['std_floor']==1e-6 and visual_norm['valid_zero_support_candidates_included']
    assert visual_norm['array_sha']==visual_norm_sha
    assert visual_verification['normalization']['array_sha']==visual_norm_sha
    for actual,stored,key in ((visual_mean,stored_mean,'mean385'),(visual_std,stored_std,'std385')):
        np.testing.assert_array_equal(actual,stored)
        np.testing.assert_array_equal(actual,np.asarray(visual_norm[key],np.float32))
    visual_fields=dict(visual_normalization_sha=visual_norm_sha,visual_receipt_binding=inputs['visual_receipt'],
        visual_protocol_binding=inputs['visual_protocol'],visual_verification_binding=inputs['visual_verification'])
    for artifact in (prefit,protocol):
        for key,value in visual_fields.items():assert artifact[key]==value,('VISUAL_INPUT_BINDING',key)
    np.testing.assert_array_equal(np.asarray(protocol['visual_mean'],np.float32),visual_mean)
    np.testing.assert_array_equal(np.asarray(protocol['visual_std'],np.float32),visual_std)
    return dict(_authorized=_AUTHORIZED,parent=parent,anchor=anchor,prefit=prefit,protocol=protocol,
        binding=C.bind(PROTOCOL),basis=basis_artifact['basis'],basis_binding=basis_binding,
        direction=direction,direction_mean=mean18,direction_std=std18,direction_normalization_sha=norm_sha,
        direction_receipt_binding=inputs['direction_receipt'],previous_prefit=previous,
        previous_direction_prefit=previous_direction,direction_representation=audit,
        previous_asymmetric_prefit=previous_asymmetric,visual=visual,visual_mean=visual_mean,visual_std=visual_std,
        **visual_fields)


def prepared(model,data):
    parent=data['parent'];names=candidate_names(model)
    models=['R0']+([] if model=='R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
    raw=np.concatenate([parent['features'][m] for m in models],axis=1)
    valid=np.concatenate([parent['valid'][m] for m in models],axis=1)
    errors=np.concatenate([parent['errors'][m] for m in models],axis=1)
    raw18=np.concatenate([data['direction'][m] for m in models],axis=1)
    raw385=np.concatenate([data['visual'][m] for m in models],axis=1)
    base=rbf_inputs(raw,valid,data['anchor']['index'],parent['mean'],parent['std'],data['basis'])
    x253=difference_from_context(base,valid,data['anchor']['index'])
    extra=direction_difference(raw18,valid,data['anchor']['index'],data['direction_mean'],data['direction_std'])
    x271=np.concatenate([x253,extra],axis=2)
    visual=appearance_difference(raw385,valid,data['anchor']['index'],data['visual_mean'],data['visual_std'])
    x=np.concatenate([x271,visual],axis=2)
    scaled,target=signed_targets(errors,valid,data['anchor']['errors'],data['anchor']['index'],parent['scale'])
    assert x.shape==(2598,len(names),FEATURE_DIM)
    hashes=dict(signed_target_sha=OLD.array_sha(target),input_difference_sha=OLD.array_sha(x253),
        base_context_sha=OLD.array_sha(base),errors_sha=OLD.array_sha(errors),
        scaled_excess_sha=OLD.array_sha(scaled),original_valid_sha=OLD.array_sha(valid),
        direction_raw_sha=OLD.array_sha(raw18),direction_difference_sha=OLD.array_sha(extra),extended_input_sha=OLD.array_sha(x271),
        visual_raw_sha=OLD.array_sha(raw385),visual_difference_sha=OLD.array_sha(visual),visual_extended_input_sha=OLD.array_sha(x))
    for key in ALL_HASH_KEYS:assert data['prefit']['models'][model][key]==hashes[key],('PREFIT_MISMATCH',model,key)
    for key in PREVIOUS_HASH_KEYS:
        assert data['previous_direction_prefit']['models'][model][key]==hashes[key],('PREVIOUS_DIRECTION_NINE_HASH_MISMATCH',model,key)
        assert data['previous_asymmetric_prefit']['models'][model][key]==hashes[key],('PREVIOUS_ASYMMETRIC_NINE_HASH_MISMATCH',model,key)
    for key in HASH_KEYS:
        assert data['previous_prefit']['models'][model][key]==hashes[key],('PREVIOUS_SIX_HASH_MISMATCH',model,key)
        assert data['direction_representation']['models'][model]['prior_six_hashes'][key]==hashes[key]
    audited=data['direction_representation']['models'][model]
    assert audited['extra18_sha']==hashes['direction_difference_sha'] and audited['extended271_sha']==hashes['extended_input_sha']
    return dict(names=names,x=x,x253=x253,x271=x271,valid=valid,axis_target=target,scaled_excess=scaled,base_context=base,
        errors=errors,direction18=raw18,direction_difference=extra,appearance385=raw385,visual_difference=visual,hashes=hashes)


def paths(model):
    assert model in C.MODEL_NAMES
    return C.RAW/'fits'/model,C.DOC/f'FIT_{model}.json'


def verify_completed(model,data):
    folder,receipt_path=paths(model);receipt=C.read(receipt_path)
    assert receipt['complete'] and receipt['model']==model and receipt['protocol']==data['binding']
    assert receipt['source_TRAIN_only'] and not receipt['VAL_quality_read'] and not receipt['real_targets_read']
    for key,path in [('START',folder/'START.json'),('trace',folder/'TRACE.jsonl'),('checkpoint',folder/'final.json')]:
        assert receipt[key]['path']==str(path.relative_to(C.ROOT));C.verify(receipt[key])
    ck=C.read(folder/'final.json');start=C.read(folder/'START.json')
    assert ck['schema']==CHECKPOINT_SCHEMA and ck['model']==start['model']==model
    assert ck['names']==start['names']==candidate_names(model)
    assert ck['protocol']==start['protocol']==data['binding']
    for artifact in (ck,start,receipt):
        for key,value in metadata().items():assert artifact[key]==value,(model,key)
        assert artifact['basis_SHA_bind']==data['basis_binding']
        assert artifact['anchor_index_sha']==data['anchor']['index_sha']
        assert artifact['direction_normalization_sha']==data['direction_normalization_sha']
        assert artifact['direction_receipt_binding']==data['direction_receipt_binding']
        for key in VISUAL_BINDING_KEYS:assert artifact[key]==data[key],(model,key)
        for key in ALL_HASH_KEYS:assert artifact[key]==data['prefit']['models'][model][key],(model,key)
    assert ck['rbf_basis_binding']==data['basis_binding'] and ck['rbf_basis']==data['basis']
    B.validate_basis(ck['rbf_basis'],ck['mean'],ck['std'])
    np.testing.assert_array_equal(np.asarray(ck['direction_mean'],np.float32),data['direction_mean'])
    np.testing.assert_array_equal(np.asarray(ck['direction_std'],np.float32),data['direction_std'])
    assert OLD.array_sha(np.stack([np.asarray(ck['direction_mean'],np.float32),np.asarray(ck['direction_std'],np.float32)]))==data['direction_normalization_sha']
    np.testing.assert_array_equal(np.asarray(ck['visual_mean'],np.float32),data['visual_mean'])
    np.testing.assert_array_equal(np.asarray(ck['visual_std'],np.float32),data['visual_std'])
    assert OLD.array_sha(np.stack([np.asarray(ck['visual_mean'],np.float32),np.asarray(ck['visual_std'],np.float32)]))==data['visual_normalization_sha']
    assert ck['bias']==0. and ck['lambda_l2']==LAMBDA
    assert ck['loss_uses_original_valid_mask'] is True
    assert ck['normalization']=='old_float32_then_float64'
    assert start['anchor_errors_sha']==data['anchor']['errors_sha']
    assert ck['certificate']==receipt['certificate'] and ck['certificate']['PASS']
    cert=ck['certificate']
    assert cert['optimizer_success'] and cert['gradient_l2_squared_over_2lambda']<=GAP_MAX
    assert cert['gradient_linf']<=cert['gradient_linf_max']==GRAD_LINF_MAX
    assert 1<=cert['objective_calls']<=MAX_CALLS and 0<=cert['iterations']<=MAX_ITER
    assert cert['loss_rule']==LOSS_RULE and cert['huber_delta']==HUBER_DELTA
    assert receipt['normalization_sha']==ck['normalization_sha']==data['parent']['normalization_sha']
    np.testing.assert_array_equal(np.asarray(ck['mean'],np.float32),data['parent']['mean'])
    np.testing.assert_array_equal(np.asarray(ck['std'],np.float32),data['parent']['std'])
    weight=np.asarray(ck['weight'],np.float64)
    assert weight.shape==(FEATURE_DIM,OUTPUT_DIM) and np.isfinite(weight).all()
    assert OLD.array_sha(weight)==receipt['final_weight_sha']
    trace=[json.loads(line) for line in (folder/'TRACE.jsonl').read_text().splitlines()]
    calls=[r for r in trace if r['event']=='objective']
    assert [r['call'] for r in calls]==list(range(1,receipt['objective_calls']+1))
    assert calls[-1]['weight_sha']==receipt['final_weight_sha']
    assert calls[-1]['objective']==cert['objective_value'] and calls[-1]['Huber']==cert['Huber']
    assert calls[-1]['Sign_logistic']==cert['Sign_logistic']
    for key in ('Huber_symmetric','Huber_underprediction'):
        assert calls[-1][key]==cert[key] and cert[key]>=0.
    np.testing.assert_allclose(cert['Huber'],cert['Huber_symmetric']+cert['Huber_underprediction'],rtol=1e-14,atol=1e-14)
    for key in ('underprediction_coefficient','underprediction_cost','overprediction_cost','underprediction_zero_curvature'):
        assert cert[key]==metadata()[key]
    assert cert['sign_rule']==SIGN_RULE and cert['sign_coefficient']==SIGN_COEFFICIENT
    np.testing.assert_allclose(cert['objective_value'],cert['Huber']+cert['Sign_logistic']+cert['L2_penalty'],rtol=1e-14,atol=1e-14)
    assert calls[-1]['armijo_accepted'] and calls[-1]['evaluated']
    assert ck['solver']['configuration']==solver_config()==start['solver']
    assert calls[-1]['call']==ck['final_accepted_call']==receipt['final_accepted_call']==cert['final_accepted_call']==ck['solver']['final_accepted_call']
    for row in trace:
        for key,value in metadata().items():assert row[key]==value
        assert row['basis_SHA_bind']==data['basis_binding'] and row['anchor_index_sha']==data['anchor']['index_sha']
        for key in ALL_HASH_KEYS:assert row[key]==receipt[key]
        assert row['direction_normalization_sha']==receipt['direction_normalization_sha'] and row['direction_receipt_binding']==receipt['direction_receipt_binding']
        for key in VISUAL_BINDING_KEYS:assert row[key]==receipt[key]
    assert len([r for r in trace if r['event']=='iteration'])==receipt['iterations']==cert['iterations']
    assert receipt['objective_calls']==cert['objective_calls']
    assert receipt['fits_executed']==1 and receipt['optimizer_steps']==receipt['iterations']
    accepted=[r for r in calls if r['phase']=='trial' and r['armijo_accepted']]
    iteration_rows=[r for r in trace if r['event']=='iteration']
    assert len(accepted)==len(iteration_rows)
    for actual,row in zip(accepted,iteration_rows):
        assert actual['call']==row['accepted_call'] and actual['weight_sha']==row['weight_sha']
        assert actual['objective']==row['objective'] and actual['objective']<=actual['armijo_bound']
    return receipt


def fit_one(model,data):
    install_training_guard()
    assert data.get('_authorized') is _AUTHORIZED and model in C.MODEL_NAMES
    C.verify(data['binding'])
    for binding in data['protocol']['codes']+list(data['protocol']['inputs'].values()):C.verify(binding)
    folder,receipt_path=paths(model)
    if receipt_path.exists():return verify_completed(model,data)
    if folder.exists():assert not any(folder.iterdir()),('PARTIAL_FIT_STOP_NO_RETRY',str(folder))
    fit_root=C.RAW/'fits'
    if fit_root.exists():assert all(p.is_dir() and p.name in C.MODEL_NAMES for p in fit_root.iterdir())
    batch=prepared(model,data)
    x,valid,target=batch['x'],batch['valid'],batch['axis_target']
    parent=data['parent']
    fields=dict(**metadata(),**batch['hashes'],basis_SHA_bind=data['basis_binding'],anchor_index_sha=data['anchor']['index_sha'],
        direction_normalization_sha=data['direction_normalization_sha'],direction_receipt_binding=data['direction_receipt_binding'],
        **{key:data[key] for key in VISUAL_BINDING_KEYS})
    initial=np.zeros((FEATURE_DIM,OUTPUT_DIM),np.float64)
    folder.mkdir(parents=True,exist_ok=True)
    C.save(folder/'START.json',dict(**fields,created_at=C.now(),model=model,protocol=data['binding'],
        initialization='all_zero_float64',initial_weight_sha=OLD.array_sha(initial),bias=0.,
        source_TRAIN_only=True,VAL_quality_read=False,real_targets_read=False,
        names=batch['names'],train_rows=len(x),source_ids_sha=OLD.array_sha(parent['ids']),
        anchor_errors_sha=data['anchor']['errors_sha'],normalization_sha=parent['normalization_sha'],
        lambda_l2=LAMBDA,objective='full_frame_mean_valid_candidate_two_axis_Huber_e_plus_Huber_min_e_zero_plus_nonzero_target_sign_logistic_coefficient1_plus_half_lambda_weight_norm_squared',
        objective_components=['Huber_symmetric','Huber_underprediction','Sign_logistic','L2_penalty'],
        Huber_definition='Huber_symmetric + Huber_underprediction',
        axis_order=['translation_cm','rotation_deg'],signed_target_scales=parent['scale'],
        loss_uses_original_valid_mask=True,solver=data['protocol']['solver'],certificate_rule=data['protocol']['certificate'],
        cpu_threads=1,torch_version=str(torch.__version__),numpy_version=np.__version__,scipy_version=scipy.__version__,
        no_valid_rows=int((~valid.any(1)).sum()),valid_candidates=int(valid.sum()),
        distinction='Only append385 native-image-supported frozen DINO features, normalized in float32 on fixed R0 TRAIN candidates then differenced against the operational R0 anchor in float64, to unchanged271 inputs. Preserve all9 original hashes, asymmetric2:1 Huber+sign loss, targets, valid mask, scales, ridge, zero initialization, fixed Newton and max-axis runtime. No old weights or warm starts.'))
    started=time.monotonic()
    calls,iterations=0,0
    try:
        with (folder/'TRACE.jsonl').open('x') as trace:
            def emit(row):
                nonlocal calls,iterations
                if row['event']=='objective':calls=row['call']
                if row['event']=='iteration':iterations=row['iteration']
                trace.write(json.dumps(dict(**fields,**row),allow_nan=False)+'\n');trace.flush()
                if row['event']=='iteration' and iterations%25==0:
                    print('SIGNED_AXES_NEWTON',model,iterations,calls,row['objective'],flush=True)
            result=newton_solve(x,valid,target,emit=emit)
        assert calls==result['objective_calls'] and iterations==result['iterations']
        weight,last=result['weight'],result['final']
        if last is None:
            raise RuntimeError(result['status']+': '+result['message'])
        accepted_call=result['final_accepted_call']
        cert=certificate(SimpleNamespace(success=result['success']),last,calls,iterations)
        cert['final_accepted_call']=accepted_call
        curvature=hessian(x,valid,last['residual'],target);eig=np.linalg.eigvalsh(curvature)
        assert eig[0]>=LAMBDA-1e-10 and np.isfinite(eig).all()
        solver=dict(success=result['success'],status=result['status'],message=result['message'],
            configuration=solver_config(),iterations=iterations,objective_calls=calls,final_accepted_call=accepted_call,
            last_evaluated_call=result['last_evaluated_call'],final_hessian_min=float(eig[0]),final_hessian_max=float(eig[-1]),
            exact_huber_kink_axes=int((np.abs(last['residual'][valid])==HUBER_DELTA).sum()),
            exact_asymmetric_zero_axes=int((last['residual'][valid]==0.).sum()),
            hessian_boundary_convention='base Huber curvature1 for abs(e)<1; additional curvature1 for -1<e<0 and0 at e=0; both Huber curvatures0 at abs(e)==1; retain smooth nonzero-target logistic curvature and lambda I; generalized choice at boundaries',
            weight_flattening='C_order_feature_then_axis',linear_solve='Two symmetric-positive-definite656 blocks, Cholesky solves; no damping or fallback.')
        if not cert['PASS']:
            C.save(C.DOC/f'REJECTED_{model}.json',dict(**fields,complete=True,accepted=False,model=model,
                protocol=data['binding'],certificate=cert,solver=solver,START=C.bind(folder/'START.json'),
                trace=C.bind(folder/'TRACE.jsonl'),final_weight=weight,final_weight_sha=OLD.array_sha(weight),
                final_accepted_call=accepted_call,last_evaluated_call=calls,automatic_retry=False,
                state_scope='Last accepted point, never an unaccepted line-search trial.'))
            raise RuntimeError('SOLVER_OR_CERTIFICATE_REJECTED: '+result['status']+'; no extension or new fit authorized.')
        assert accepted_call==calls
        ck=dict(**fields,schema=CHECKPOINT_SCHEMA,rbf_basis_binding=data['basis_binding'],rbf_basis=data['basis'],
            direction_mean=data['direction_mean'],direction_std=data['direction_std'],
            visual_mean=data['visual_mean'],visual_std=data['visual_std'],
            model=model,names=batch['names'],weight=weight,bias=0.,mean=parent['mean'],std=parent['std'],protocol=data['binding'],
            normalization_sha=parent['normalization_sha'],lambda_l2=LAMBDA,certificate=cert,solver=solver,
            normalization='old_float32_then_float64',axis_order=['translation_cm','rotation_deg'],
            signed_target_scales=parent['scale'],loss_uses_original_valid_mask=True,
            fits_executed=1,optimizer_steps=iterations,final_accepted_call=accepted_call,
            prediction_definition='Signed log1p normalized T and R differences relative to operational R0 GEO anchor; predictions need not be true nonregressions.',
            selection_definition='Minimize max(predicted signed axes) over all original valid whole poses with OLD tie policy; anchor is not given a new tie priority.')
        C.save(folder/'final.json',ck)
        C.save(receipt_path,dict(**fields,complete=True,model=model,protocol=data['binding'],
            START=C.bind(folder/'START.json'),trace=C.bind(folder/'TRACE.jsonl'),checkpoint=C.bind(folder/'final.json'),
            certificate=cert,normalization_sha=parent['normalization_sha'],final_weight_sha=OLD.array_sha(weight),
            objective_calls=calls,iterations=iterations,optimizer_steps=iterations,fits_executed=1,final_accepted_call=accepted_call,
            source_TRAIN_only=True,VAL_quality_read=False,real_targets_read=False,
            created_at=C.now(),wall_seconds=time.monotonic()-started,read_paths=sorted(READS)))
    except BaseException as exc:
        if not (folder/'FAILED.json').exists():
            C.save(folder/'FAILED.json',dict(**fields,created_at=C.now(),model=model,protocol=data['binding'],
                objective_calls=calls,iterations=iterations,error_type=type(exc).__name__,error=str(exc),
                automatic_retry=False,further_iterations_authorized=False))
        raise

    return verify_completed(model,data)


def complete_all(data):
    receipts=[verify_completed(model,data) for model in C.MODEL_NAMES]
    assert len({r['normalization_sha'] for r in receipts})==1
    result=dict(**metadata(),complete=True,basis_SHA_bind=data['basis_binding'],anchor_index_sha=data['anchor']['index_sha'],
        direction_normalization_sha=data['direction_normalization_sha'],direction_receipt_binding=data['direction_receipt_binding'],
        **{key:data[key] for key in VISUAL_BINDING_KEYS},
        models=list(C.MODEL_NAMES),protocol=data['binding'],fit_count=4,fits=[C.bind(paths(model)[1]) for model in C.MODEL_NAMES],
        all_certified=True,max_objective_calls_per_fit=MAX_CALLS,total_objective_calls=sum(r['objective_calls'] for r in receipts),
        total_iterations=sum(r['iterations'] for r in receipts),source_TRAIN_only=True,VAL_quality_read=False,real_targets_read=False)
    for key in ALL_HASH_KEYS:result[key+'_by_model']={r['model']:r[key] for r in receipts}
    result['final_accepted_call_by_model']={r['model']:r['final_accepted_call'] for r in receipts}
    result['objective_components']=['Huber_symmetric','Huber_underprediction','Sign_logistic','L2_penalty']
    result['Huber_definition']='Huber_symmetric + Huber_underprediction'
    result['final_objective_components_by_model']={r['model']:{key:r['certificate'][key] for key in
        ('objective_value','Huber','Huber_symmetric','Huber_underprediction','Sign_logistic','L2_penalty')} for r in receipts}
    assert result['total_objective_calls']<=4*MAX_CALLS
    path=C.DOC/'TRAINING_COMPLETE.json'
    if path.exists():assert C.read(path)==result
    else:C.save(path,result)
    return result


def solver_selfcheck():
    """Small invented optimization problems; never load TRAIN artifacts."""
    global hessian,cho_solve,objective
    x=np.zeros((3,2,2),np.float64);x[0,1]=[1.,0.];x[1,1]=[0.,1.]
    valid=np.array([[1,1],[1,1],[0,0]],bool)
    target=np.zeros((3,2,2),np.float64);target[0,1]=[.2,-.3];target[1,1]=[-.25,.15]
    rows=[];quadratic=newton_solve(x,valid,target,emit=rows.append)
    def scalar_root(y,ridge_scale):
        # Independent bisection of the one-coordinate derivative in invented
        # orthogonal fixtures; no source targets or fitted model are accessed.
        lo,hi=-4.,4.;s=float(np.sign(y))
        for _ in range(100):
            mid=(lo+hi)/2
            clipped=float(np.clip(mid-y,-1.,1.))
            g=clipped+min(clipped,0.)-s/(1+np.exp(s*mid))+ridge_scale*mid
            if g>0:hi=mid
            else:lo=mid
        return (lo+hi)/2
    expected=np.array([[scalar_root(y,12*LAMBDA) for y in row] for row in [[.2,-.3],[-.25,.15]]])
    assert quadratic['success'] and quadratic['iterations']>0
    # The fixed solver stops at gradient Linf1e-8, not at machine precision.
    # Both roots lie in the base quadratic interval; derivative slope >=1/12+λ
    # bounds parameter error from the actual accepted-point gradient.
    labels=np.array([[.2,-.3],[-.25,.15]])
    assert (np.abs(expected-labels)<1.).all() and (np.abs(quadratic['weight']-labels)<1.).all()
    scalar_error_bound=float(np.max(np.abs(quadratic['final']['gradient'])))/(1./12+LAMBDA)
    np.testing.assert_allclose(quadratic['weight'],expected,rtol=0,atol=scalar_error_bound+1e-14)
    assert rows[0]['phase']=='initial' and rows[0]['armijo_accepted'] and rows[0]['call']==1
    assert rows[-1]['event']=='iteration' and rows[-1]['accepted_call']==quadratic['final_accepted_call']==quadratic['objective_calls']
    # The initial residual is exactly at both Huber kinks. Generalized curvature
    #0 while logistic curvature remains; the full proposal still requires
    #genuine backtracking in this invented problem.
    kx=np.array([[[0.],[1.]]]);kv=np.ones((1,2),bool);kt=np.array([[[0.,0.],[1.,-1.]]])
    origin=objective(np.zeros((1,2)),kx,kv,kt)
    np.testing.assert_array_equal(hessian(kx,kv,origin['residual'],kt),(LAMBDA+.0625)*np.eye(2))
    line=[];kink=newton_solve(kx,kv,kt,emit=line.append)
    assert kink['success'] and kink['final_accepted_call']==kink['objective_calls']
    rejected=[r for r in line if r['event']=='objective' and not r['armijo_accepted']]
    assert rejected and all(r['phase']=='trial' and r['objective']>r['armijo_bound'] for r in rejected)
    previous=1;nextalpha=1.;lastobjective=line[0]['objective'];lastsha=line[0]['weight_sha']
    for row in [r for r in line if r['event']=='objective' and r['phase']=='trial']:
        assert row['accepted_point_call_before']==previous and row['alpha']==nextalpha
        assert row['parent_objective']==lastobjective and row['parent_weight_sha']==lastsha
        assert row['directional_derivative']<0
        assert row['armijo_bound']==lastobjective+1e-4*row['alpha']*row['directional_derivative']
        if row['armijo_accepted']:
            assert row['objective']<=row['armijo_bound'];previous=row['call'];nextalpha=1.
            lastobjective=row['objective'];lastsha=row['weight_sha']
        else:nextalpha*=.5
    kink_expected=np.array([[scalar_root(1.,4*LAMBDA),scalar_root(-1.,4*LAMBDA)]])
    assert (np.abs(kink_expected-np.array([[1.,-1.]]))<1.).all()
    assert (np.abs(kink['weight']-np.array([[1.,-1.]]))<1.).all()
    kink_error_bound=float(np.max(np.abs(kink['final']['gradient'])))/(.25+LAMBDA)
    np.testing.assert_allclose(kink['weight'],kink_expected,rtol=0,atol=kink_error_bound+1e-14)
    # Call2 is an unaccepted trial. Never expose it as the returned final state.
    limited=[];budget=newton_solve(kx,kv,kt,max_calls=2,emit=limited.append)
    assert not budget['success'] and budget['status']=='OBJECTIVE_BUDGET_EXHAUSTED'
    assert budget['objective_calls']==2 and budget['iterations']==0 and budget['final_accepted_call']==1
    assert not limited[-1]['armijo_accepted'] and limited[-1]['weight_sha']!=OLD.array_sha(budget['weight'])
    np.testing.assert_array_equal(budget['weight'],np.zeros((1,2)))
    assert budget['final']['value']==origin['value']
    one=newton_solve(kx,kv,kt,max_calls=1)
    assert not one['success'] and one['objective_calls']==1 and one['final_accepted_call']==1
    zero=newton_solve(kx,kv,kt,max_iter=0)
    assert not zero['success'] and zero['iterations']==0 and zero['objective_calls']==1
    empty=newton_solve(np.zeros((2,2,1)),np.zeros((2,2),bool),np.zeros((2,2,2)))
    assert empty['success'] and empty['objective_calls']==1 and empty['iterations']==0
    original_h=hessian;original_solve=cho_solve;original_objective=objective
    try:
        hessian=lambda *args,**kwargs:-np.eye(2)
        failed=newton_solve(kx,kv,kt)
        assert not failed['success'] and failed['status']=='SPD_SOLVE_FAILED' and failed['objective_calls']==1
        hessian=original_h
        cho_solve=lambda factor,b,**kwargs:-b
        failed=newton_solve(kx,kv,kt)
        assert not failed['success'] and failed['status']=='NON_DESCENT_DIRECTION' and failed['objective_calls']==1
        cho_solve=original_solve
        counter=0
        def nonfinite_second(*args,**kwargs):
            nonlocal counter
            counter+=1
            out=original_objective(*args,**kwargs)
            if counter==2:out['value']=np.inf
            return out
        objective=nonfinite_second
        failed=newton_solve(kx,kv,kt)
        assert not failed['success'] and failed['status']=='NONFINITE_TRIAL_OBJECTIVE'
        assert failed['objective_calls']==2 and failed['final_accepted_call']==1 and not failed['weight'].any()
    finally:hessian=original_h;cho_solve=original_solve;objective=original_objective
    return dict(PASS=True,invented_arrays_only=True,orthogonal_problem_matches_independent_scalar_roots=True,kink_backtracking_rejected_trials=len(rejected),
        kink_objective_calls=kink['objective_calls'],kink_accepted_iterations=kink['iterations'],
        hard_caps_and_last_accepted_retention=True,SPD_failure_stop=True,non_descent_stop=True,nonfinite_stop=True,
        warmstart=False,damping=False,actual_fits=0)


def selfcheck():
    """Invented arrays only: no artifact loader, optimizer or training call."""
    global signed_targets
    import ast
    import inspect
    for name in ('objective','hessian','signed_targets','validate_anchor','difference_from_context','newton_solve','certificate','solver_config'):
        assert ast.dump(ast.parse(inspect.getsource(globals()[name])),include_attributes=False)==ast.dump(ast.parse(inspect.getsource(getattr(Q,name))),include_attributes=False),name
    from types import SimpleNamespace
    rng=np.random.default_rng(202610012532)
    mean=rng.normal(size=RAW_FEATURE_DIM).astype(np.float32)
    std=(rng.random(RAW_FEATURE_DIM)+.4).astype(np.float32)
    centers=rng.normal(size=(64,189)).astype(np.float64)
    basis=dict(schema=B.RUNTIME_SCHEMA,context_dim=189,rbf_dim=64,centers=centers.tolist(),
        bandwidth_squared=B.bandwidth_squared(centers),normalization_sha=OLD.array_sha(np.stack([mean,std])))
    raw=rng.normal(size=(5,4,94)).astype(np.float32)
    valid=np.array([[1,1,1,1],[0,1,1,0],[1,0,0,0],[0,0,0,0],[1,1,0,1]],bool)
    anchor_index=np.array([0,1,0,-1,1],np.int64)
    base=rbf_inputs(raw,valid,anchor_index,mean,std,basis)
    direction18=rng.normal(size=(*valid.shape,18)).astype(np.float32);direction18[~valid]=0.
    direction_mean=rng.normal(size=18).astype(np.float32)
    direction_std=(rng.random(18)+.25).astype(np.float32)
    appearance385=rng.normal(size=(*valid.shape,385)).astype(np.float32);appearance385[~valid]=0.
    visual_mean=rng.normal(size=385).astype(np.float32)
    visual_std=(rng.random(385)+.25).astype(np.float32)
    x=difference_inputs(raw,valid,anchor_index,mean,std,basis,direction18,direction_mean,direction_std,appearance385,visual_mean,visual_std)
    z385=((appearance385-visual_mean)/visual_std).astype(np.float64)
    old271=Q.difference_inputs(raw,valid,anchor_index,mean,std,basis,direction18,direction_mean,direction_std)
    np.testing.assert_array_equal(x[:,:,:271],old271)
    zerovisual=difference_inputs(raw,valid,anchor_index,mean,std,basis,direction18,direction_mean,direction_std,np.zeros_like(appearance385),visual_mean,visual_std)
    np.testing.assert_array_equal(zerovisual[:,:,:271],old271);assert not zerovisual[:,:,271:].any()
    z18=((direction18-direction_mean)/direction_std).astype(np.float64)
    expected=np.zeros_like(x)
    for i in range(len(x)):
        for j in np.flatnonzero(valid[i]):
            expected[i,j,:253]=base[i,j]-base[i,anchor_index[i]]
            expected[i,j,253:271]=z18[i,j]-z18[i,anchor_index[i]]
            expected[i,j,271:]=z385[i,j]-z385[i,anchor_index[i]]
    np.testing.assert_array_equal(x,expected)
    errors=np.full((*valid.shape,2),np.inf,np.float64)
    errors[valid]=rng.uniform(.1,5,size=(valid.sum(),2))
    rows=np.flatnonzero(valid.any(1));anchor_errors=np.full((len(x),2),np.inf,np.float64)
    anchor_errors[rows]=errors[rows,anchor_index[rows]]
    # A concrete tradeoff establishes that the two signed axes survive separately.
    errors[0,0]=anchor_errors[0]=[2.,4.];errors[0,1]=[1.,8.]
    scale=np.array([2.,4.])
    with np.errstate(invalid='raise',divide='raise',over='raise'):
        excess,target=signed_targets(errors,valid,anchor_errors,anchor_index,scale)
    np.testing.assert_allclose(target[0,1],[-np.log1p(.5),np.log1p(1.)],rtol=0,atol=0)
    assert not target[~valid].any() and not excess[~valid].any()
    assert not target[rows,anchor_index[rows]].any() and not x[rows,anchor_index[rows]].any()
    missing=anchor_index.copy();missing[0]=-1
    try:difference_from_context(base,valid,missing)
    except AssertionError:pass
    else:raise AssertionError('Missing operational anchor accepted despite valid candidates')
    # Independent full1312 scalar/torch differentiation, deliberately away from |r|=1.
    weight=rng.normal(scale=.001,size=(FEATURE_DIM,2))
    toy_target=np.zeros_like(target)
    toy_target[valid]=rng.choice([-2.,-.35,.25,2.],size=(valid.sum(),2))
    toy_target[rows,anchor_index[rows]]=0.
    checked=objective(weight,x,valid,toy_target)
    assert np.min(np.abs(np.abs(checked['residual'][valid])-1.))>.1
    scalar=0.
    for i in range(len(x)):
        local=0.
        for j in np.flatnonzero(valid[i]):
            for axis in range(2):
                prediction=float(np.dot(x[i,j],weight[:,axis]))
                residual=prediction-toy_target[i,j,axis]
                local+=(.5*residual**2 if abs(residual)<1 else abs(residual)-.5)
                if residual<0.:
                    local+=(.5*residual**2 if abs(residual)<1 else abs(residual)-.5)
                if toy_target[i,j,axis]!=0:
                    logit=-float(np.sign(toy_target[i,j,axis]))*prediction
                    local+=max(logit,0.)+np.log1p(np.exp(-abs(logit)))
        scalar+=local/(2*max(int(valid[i].sum()),1))/len(x)
    scalar+=.5*LAMBDA*float(np.sum(weight**2))
    np.testing.assert_allclose(checked['value'],scalar,rtol=1e-14,atol=1e-14)
    xt=torch.tensor(x,dtype=torch.float64);yt=torch.tensor(toy_target,dtype=torch.float64)
    coefficient=torch.tensor(valid/(len(x)*2*np.maximum(valid.sum(1),1)[:,None]),dtype=torch.float64)
    def torch_loss(flat):
        matrix=flat.reshape(FEATURE_DIM,2)
        predictions=torch.einsum('nkd,da->nka',xt,matrix)
        terms=torch.nn.functional.huber_loss(predictions,yt,reduction='none',delta=1.)
        below=torch.minimum(predictions-yt,torch.zeros_like(yt))
        terms=terms+torch.nn.functional.huber_loss(below,torch.zeros_like(below),reduction='none',delta=1.)
        terms=terms+torch.where(yt!=0,torch.nn.functional.softplus(-torch.sign(yt)*predictions),0.)
        return (terms*coefficient[:,:,None]).sum()+.5*LAMBDA*(matrix*matrix).sum()
    wt=torch.tensor(weight.ravel(),dtype=torch.float64,requires_grad=True)
    tl=torch_loss(wt);tg=torch.autograd.grad(tl,wt)[0].detach().numpy().reshape(FEATURE_DIM,2)
    th=torch.autograd.functional.hessian(torch_loss,wt).detach().numpy()
    nh=hessian(x,valid,checked['residual'],toy_target)
    np.testing.assert_allclose(float(tl),checked['value'],rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(tg,checked['gradient'],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(th,nh,rtol=1e-11,atol=1e-12)
    fd=np.zeros_like(weight);epsilon=1e-5
    for index in np.ndindex(weight.shape):
        plus,minus=weight.copy(),weight.copy();plus[index]+=epsilon;minus[index]-=epsilon
        fd[index]=(objective(plus,x,valid,toy_target)['value']-objective(minus,x,valid,toy_target)['value'])/(2*epsilon)
    np.testing.assert_allclose(fd,checked['gradient'],rtol=2e-6,atol=2e-9)
    direction=rng.normal(size=weight.shape);direction/=np.linalg.norm(direction)
    hfd=(objective(weight+epsilon*direction,x,valid,toy_target)['gradient']-objective(weight-epsilon*direction,x,valid,toy_target)['gradient'])/(2*epsilon)
    np.testing.assert_allclose(hfd.ravel(),nh@direction.ravel(),rtol=1e-7,atol=1e-9)
    eigen=np.linalg.eigvalsh(nh);assert eigen[0]>=LAMBDA-1e-11
    assert not nh[0::2,1::2].any()
    # Target zero omits logistic but keeps both asymmetric Huber components.
    zero_target=np.zeros_like(toy_target)
    zero_loss=objective(weight,x,valid,zero_target)
    old_zero=SIGNED.objective(weight,x,valid,zero_target)
    assert zero_loss['Sign_logistic']==0.
    assert zero_loss['Huber_underprediction']>0.
    np.testing.assert_array_equal(zero_loss['Huber_symmetric'],old_zero['Huber'])
    np.testing.assert_allclose(zero_loss['value'],old_zero['value']+zero_loss['Huber_underprediction'],rtol=1e-14,atol=1e-14)
    zx=np.array([[[0.],[1.]]],np.float64);zv=np.ones((1,2),bool);zy=np.zeros((1,2,2),np.float64)
    pair=objective(np.array([[-.4,.4]]),zx,zv,zy)
    np.testing.assert_allclose(pair['Huber_symmetric'],.04,rtol=0,atol=1e-16)
    np.testing.assert_allclose(pair['Huber_underprediction'],.02,rtol=0,atol=1e-16)
    np.testing.assert_allclose(hessian(zx,zv,pair['residual'],zy),np.diag([.5+LAMBDA,.25+LAMBDA]),rtol=0,atol=1e-16)
    boundary=objective(np.zeros((1,2)),zx,zv,zy)
    np.testing.assert_array_equal(hessian(zx,zv,boundary['residual'],zy),np.eye(2)*(.25+LAMBDA))
    assert boundary['Huber']==boundary['Huber_symmetric']==boundary['Huber_underprediction']==0.
    # Stable softplus/sigmoid at very large positive and negative predictions.
    extreme_x=np.array([[[0.],[1.],[-1.],[.5]]],np.float64)
    extreme_valid=np.ones((1,4),bool)
    extreme_y=np.array([[[0.,0.],[1.,-1.],[1.,-1.],[0.,0.]]],np.float64)
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        extreme=objective(np.array([[1e4,-1e4]]),extreme_x,extreme_valid,extreme_y)
        extreme_h=hessian(extreme_x,extreme_valid,extreme['residual'],extreme_y)
    assert np.isfinite(extreme['value']) and np.isfinite(extreme['gradient']).all() and np.isfinite(extreme_h).all()
    np.testing.assert_allclose(extreme['Sign_logistic'],2500.,rtol=0,atol=0)
    # Anchor is exactly zero, but an equal-score alternative uses the OLD tie rule.
    ck=dict(**metadata(),schema=CHECKPOINT_SCHEMA,normalization='old_float32_then_float64',names=candidate_names('UNION_s1'),
        bias=0.,lambda_l2=LAMBDA,weight=weight.tolist(),mean=mean.tolist(),std=std.tolist(),rbf_basis=basis,
        rbf_basis_binding={'invented':True},basis_SHA_bind={'invented':True},
        direction_mean=direction_mean.tolist(),direction_std=direction_std.tolist(),
        direction_normalization_sha=OLD.array_sha(np.stack([direction_mean,direction_std])),
        direction_receipt_binding={'invented':True},visual_mean=visual_mean.tolist(),visual_std=visual_std.tolist(),
        visual_normalization_sha=OLD.array_sha(np.stack([visual_mean,visual_std])),
        visual_receipt_binding={'invented':True},visual_protocol_binding={'invented':True},visual_verification_binding={'invented':True})
    prediction=predict_axes(ck,raw,valid,anchor_index,direction18,appearance385)
    np.testing.assert_array_equal(prediction,checked['prediction'])
    score=score_candidates(ck,raw,valid,anchor_index,direction18,appearance385)
    assert np.isposinf(score[~valid]).all()
    np.testing.assert_array_equal(score[valid],prediction.max(2)[valid])
    choice=select_candidates(score,valid,ck['names'])
    assert choice[3]==-1 and (score[rows,choice[rows]]<=0).all()
    assert not prediction[rows,anchor_index[rows]].any()
    zero_ck=dict(ck,weight=np.zeros_like(weight).tolist())
    zero_choice=select_candidates(score_candidates(zero_ck,raw,valid,anchor_index,direction18,appearance385),valid,ck['names'])
    assert zero_choice[4]==0 and anchor_index[4]==1,'No new anchor-first tie priority permitted'
    shuffled=np.array([4,0,3,1,2])
    np.testing.assert_array_equal(predict_axes(ck,raw[shuffled],valid[shuffled],anchor_index[shuffled],direction18[shuffled],appearance385[shuffled]),prediction[shuffled])
    original=signed_targets
    def forbidden(*args,**kwargs):raise AssertionError('Runtime attempted target computation')
    try:
        signed_targets=forbidden
        np.testing.assert_array_equal(score_candidates(ck,raw,valid,anchor_index,direction18,appearance385),score)
    finally:signed_targets=original
    for flag in ('runtime_uses_margin','runtime_uses_reference_errors','runtime_safe_mask'):
        bad=dict(ck);bad[flag]=True
        try:score_candidates(bad,raw,valid,anchor_index,direction18,appearance385)
        except AssertionError:pass
        else:raise AssertionError('Forbidden runtime flag accepted: '+flag)
    # Zero385 embedding preserves the exact mathematical old model; floating
    # reduction widths may differ, so record tight numeric deviations honestly.
    old_weight=rng.normal(scale=.02,size=(271,2))
    embedded=np.concatenate([old_weight,np.zeros((385,2),np.float64)],axis=0)
    old_ck=dict(ck,**{k:v for k,v in Q.metadata().items() if k not in ck})
    old_ck.update(Q.metadata());old_ck.update(schema=Q.CHECKPOINT_SCHEMA,weight=old_weight.tolist())
    embedded_ck=dict(ck,weight=embedded.tolist())
    old_axes=Q.predict_axes(old_ck,raw,valid,anchor_index,direction18)
    embedded_axes=predict_axes(embedded_ck,raw,valid,anchor_index,direction18,appearance385)
    np.testing.assert_allclose(embedded_axes,old_axes,rtol=1e-13,atol=1e-13)
    # Identical Q asymmetric loss; only the input dimension is extended.
    old_j=objective(old_weight,x[:,:,:271],valid,toy_target)
    embedded_j=objective(embedded,x,valid,toy_target)
    for key in ('value','Huber','Huber_symmetric','Huber_underprediction','Sign_logistic','penalty'):
        np.testing.assert_allclose(embedded_j[key],old_j[key],rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(embedded_j['gradient'][:271],old_j['gradient'],rtol=1e-12,atol=1e-13)
    # Extra dimensions may be informative: these are two toy candidates with
    # identical old inputs but distinct native visual coordinates.
    flip_raw=np.zeros((1,2,94),np.float32);flip_valid=np.ones((1,2),bool);flip_anchor=np.array([0],np.int64)
    flip_direction=np.zeros((1,2,18),np.float32)
    flip_visual=np.zeros((1,2,385),np.float32);flip_visual[0,1,0]=1.
    flip_weight=np.zeros((656,2),np.float64);flip_weight[271]=[-1.,-2.]
    flip_ck=dict(ck,weight=flip_weight.tolist(),names=candidate_names('R0_ONLY'),
        direction_mean=np.zeros(18,np.float32).tolist(),direction_std=np.ones(18,np.float32).tolist(),
        direction_normalization_sha=OLD.array_sha(np.stack([np.zeros(18,np.float32),np.ones(18,np.float32)])),
        visual_mean=np.zeros(385,np.float32).tolist(),visual_std=np.ones(385,np.float32).tolist(),
        visual_normalization_sha=OLD.array_sha(np.stack([np.zeros(385,np.float32),np.ones(385,np.float32)])))
    scores=score_candidates(flip_ck,flip_raw,flip_valid,flip_anchor,flip_direction,flip_visual)
    assert scores.tolist()==[[0.,-1.]]
    assert select_candidates(scores,flip_valid,flip_ck['names']).tolist()==[1]
    bad_norm=dict(ck,visual_normalization_sha='wrong')
    try:predict_axes(bad_norm,raw,valid,anchor_index,direction18,appearance385)
    except AssertionError:pass
    else:raise AssertionError('Changed visual normalization accepted')
    # Exact strong-convex gap example, using a zero-design invented objective.
    tiny=np.full((FEATURE_DIM,2),1e-4)
    ridge_only=objective(tiny,np.zeros((1,2,FEATURE_DIM)),np.ones((1,2),bool),np.zeros((1,2,2)))
    cert=certificate(SimpleNamespace(success=True),ridge_only,1,0)
    np.testing.assert_allclose(cert['gradient_l2_squared_over_2lambda'],ridge_only['value'],rtol=1e-14)
    assert cert['PASS'] and not certificate(SimpleNamespace(success=False),ridge_only,1,0)['PASS']
    result=dict(PASS=True,invented_arrays_only=True,actual_training_rows_read=0,fits_executed=0,
        zero385_embedding_score_max_abs=float(np.max(np.abs(embedded_axes-old_axes))),
        zero385_embedding_same_asymmetric_objective_max_abs=max(abs(embedded_j[k]-old_j[k]) for k in ('value','Huber','Sign_logistic','penalty')),
        prior_Q_objective_and_newton_AST_exact=True,constant_visual_difference_exact_zero=True,
        native_visual_can_change_invented_ranking=True,visual_normalization_binding_checked=True,
        gradient_fd_max_abs=float(np.max(np.abs(fd-checked['gradient']))),
        gradient_torch_max_abs=float(np.max(np.abs(tg-checked['gradient']))),
        hessian_torch_max_abs=float(np.max(np.abs(th-nh))),hessian_min_eigenvalue=float(eigen[0]),
        zero_target_sign_term_exact_zero=True,zero_target_asymmetric_Huber_retained=True,
        fixed_two_to_one_cost_and_zero_extra_curvature_checked=True,extreme_predictions_stable=True,
        runtime_anchor_axes_exact_zero=True,all_invalid_retained=True,old_tie_policy_preserved=True,
        classical_hessian_claim_excludes_abs_residual_equal_one_and_zero=True)
    result['newton_solver']=solver_selfcheck()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['selfcheck','fit','all'])
    parser.add_argument('--model',choices=C.MODEL_NAMES);args=parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.command=='selfcheck':result=selfcheck()
        else:
            data=load_inputs()
            if args.command=='fit':
                assert args.model is not None;result=fit_one(args.model,data)
            else:
                assert args.model is None
                for model in C.MODEL_NAMES:fit_one(model,data)
                result=complete_all(data)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,default=str),flush=True)


if __name__=='__main__':main()
