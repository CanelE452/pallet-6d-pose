"""Independent final signed-axis TRAIN audit; never fit or read VAL/real GT."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from . import common as C
from scripts.research.pallet_pose_union_selection_20261001_v1 import train as OLD
from scripts.research.pallet_pose_anchor_rbf_20261001_v1 import verify_train as V
from scripts.research.pallet_pose_anchor_context_20261001_v1.train_risk_diagnostic import classify

HASH_KEYS=('signed_target_sha','input_difference_sha','base_context_sha','errors_sha','scaled_excess_sha','original_valid_sha')
METADATA=dict(feature_map='normalized94_abs_anchor_delta94_identity1_fixed_rbf64',feature_dim=253,
    raw_feature_dim=94,context_dim=189,rbf_dim=64,output_dim=2,
    input_rule='RBF253_CANDIDATE_MINUS_R0_ANCHOR',target_rule='SIGNED_LOG1P_NORMALIZED_TR_ANCHOR_EXCESS',
    loss_rule='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_HUBER',prediction_rule='MAX_TWO_SIGNED_LOG1P_AXES',
    huber_delta=1.,runtime_uses_margin=False,runtime_uses_reference_errors=False,runtime_safe_mask=False,
    solver_rule='BLOCK_GENERALIZED_NEWTON_ARMIJO')
SOLVER_CONFIG=dict(method='block_generalized_newton_armijo',maxiter=1000,maxfun=2000,
    gradient_linf_tolerance=1e-8,initial_alpha=1.,backtrack_factor=.5,armijo_c1=1e-4,
    initialization='zeros',damping=0.,huber_kink_curvature=0.)


def independent_targets(errors,valid,anchor,index,scale):
    scaled=np.zeros(errors.shape,np.float64);target=np.zeros(errors.shape,np.float64)
    for i in range(len(valid)):
        if not valid[i].any():
            assert index[i]==-1 and np.isposinf(anchor[i]).all()
            continue
        assert int(index[i]) in (0,1) and valid[i,index[i]]
        np.testing.assert_array_equal(errors[i,index[i]],anchor[i])
        assert np.isfinite(anchor[i]).all() and np.isfinite(errors[i,valid[i]]).all()
        for j in np.flatnonzero(valid[i]):
            for a in range(2):
                scaled[i,j,a]=(float(errors[i,j,a])-float(anchor[i,a]))/float(scale[a])
                target[i,j,a]=np.sign(scaled[i,j,a])*np.log1p(abs(scaled[i,j,a]))
        assert not target[i,index[i]].any()
    assert np.isfinite(scaled).all() and np.isfinite(target).all()
    assert not target[~valid].any()
    return scaled,target


def independent_difference(phi,valid,index):
    out=np.zeros_like(phi)
    for i in range(len(valid)):
        if not valid[i].any():
            assert index[i]==-1
            continue
        assert int(index[i]) in (0,1) and valid[i,index[i]]
        for j in np.flatnonzero(valid[i]):
            out[i,j]=phi[i,j]-phi[i,index[i]]
        assert not out[i,index[i]].any()
    assert np.isfinite(out).all() and not out[~valid].any()
    return out


def independent_objective(weight,x,valid,target,with_hessian=False):
    """Torch Huber/autograd with an independently assembled block Hessian."""
    w=torch.tensor(weight,dtype=torch.float64,requires_grad=True)
    xx=torch.tensor(x,dtype=torch.float64)
    yy=torch.tensor(target,dtype=torch.float64)
    vv=torch.tensor(valid,dtype=torch.bool)
    assert w.ndim==2 and w.shape[1]==2 and xx.shape==(*vv.shape,len(w))
    assert yy.shape==(*vv.shape,2) and not xx[~vv].any() and not yy[~vv].any()
    prediction=torch.einsum('nkd,da->nka',xx,w)
    per_axis=F.huber_loss(prediction,yy,reduction='none',delta=1.)
    count=vv.sum(1).clamp(min=1).to(torch.float64)
    frame_loss=(per_axis*vv[:,:,None]).sum((1,2))/(2*count)
    data_loss=frame_loss.mean()
    penalty=.5e-4*w.square().sum()
    objective=data_loss+penalty
    gradient,=torch.autograd.grad(objective,w)
    g=gradient.detach().numpy();pred=prediction.detach().numpy()
    residual=pred-target
    H=None
    if with_hessian:
        d=len(w);H=1e-4*np.eye(d*2)
        flat=x.reshape(-1,d)
        for axis in range(2):
            # Independent BLAS weighted-design product. At |residual|=1 use
            # zero generalized curvature, as declared before fitting.
            coef=np.array([[float(valid[i,j] and abs(residual[i,j,axis])<1.) /
                (len(x)*2*max(int(valid[i].sum()),1)) for j in range(valid.shape[1])]
                for i in range(len(x))],np.float64).reshape(-1)
            H[axis::2,axis::2]+=flat.T@(flat*coef[:,None])
    norm=float(np.linalg.norm(g))
    return dict(objective=float(objective.detach()),Huber=float(data_loss.detach()),
        L2_penalty=float(penalty.detach()),gradient=g,gradient_l2=norm,
        gradient_linf=float(np.abs(g).max()),certified_gap_upper_bound=norm**2/(2e-4),
        prediction=pred,residual=residual,Hessian=H,
        exact_Huber_kink_scalar_count=int(np.sum((np.abs(residual)==1.) & valid[:,:,None])))


def public_objective(value):
    return {k:v for k,v in value.items() if k not in ('gradient','prediction','residual','Hessian')}


def verify_newton_trace(trace,start,fit,checkpoint):
    """Independent log audit; never replay the optimizer or choose a new step."""
    objectives=[r for r in trace if r['event']=='objective']
    accepted_rows=[r for r in trace if r['event']=='iteration']
    assert len(trace)==len(objectives)+len(accepted_rows)
    assert 1<=len(objectives)<=2000 and len(accepted_rows)<=1000
    assert [r['call'] for r in objectives]==list(range(1,len(objectives)+1))
    assert [r['iteration'] for r in accepted_rows]==list(range(1,len(accepted_rows)+1))
    assert start['initial_weight_sha']==objectives[0]['weight_sha']
    assert start['initialization']=='all_zero_float64'
    assert start['solver']==checkpoint['solver']['configuration']==SOLVER_CONFIG
    assert checkpoint['solver']['status']=='CONVERGED' and checkpoint['solver']['success']
    assert len(objectives)==fit['objective_calls']==checkpoint['solver']['last_evaluated_call']
    assert len(accepted_rows)==fit['iterations']
    initial=objectives[0]
    assert initial['phase']=='initial' and initial['iteration']==0 and initial['alpha']==0.
    assert initial['armijo_accepted'] is True and initial['evaluated'] is True
    for key in ('accepted_point_call_before','parent_weight_sha','parent_objective','directional_derivative','armijo_bound'):
        assert initial[key] is None
    accepted=initial; alpha=1.; accepted_trials=[]; rejected_trials=0; rounded_bounds=0
    for row in objectives:
        assert row['evaluated'] is True
        for key in ('objective','Huber','L2_penalty','gradient_l2','gradient_linf','gradient_gap_upper_bound'):
            assert np.isfinite(row[key]) and row[key]>=0.
        assert abs(row['objective']-row['Huber']-row['L2_penalty'])<=1e-12
        np.testing.assert_allclose(row['gradient_gap_upper_bound'],row['gradient_l2']**2/(2e-4),rtol=1e-12,atol=1e-18)
        if row is initial:continue
        assert row['phase']=='trial' and row['iteration']==len(accepted_trials)+1
        assert row['accepted_point_call_before']==accepted['call']
        assert row['parent_weight_sha']==accepted['weight_sha'] and row['parent_objective']==accepted['objective']
        assert row['alpha']==alpha and 0.<alpha<=1.
        derivative=row['directional_derivative']
        assert np.isfinite(derivative) and derivative<0.
        bound=row['parent_objective']+1e-4*alpha*derivative
        assert row['armijo_bound']==bound and bound<=row['parent_objective']
        rounded_bounds+=int(bound==row['parent_objective'])
        assert row['armijo_accepted']==(row['objective']<=bound)
        if row['armijo_accepted']:
            accepted_trials.append(row);accepted=row;alpha=1.
        else:
            rejected_trials+=1;alpha*=.5
            assert alpha>0.
    assert len(accepted_rows)==len(accepted_trials) and accepted is objectives[-1]
    expected_event_sequence=[]
    for row in objectives:
        expected_event_sequence.append(('objective',row['call']))
        if row is not initial and row['armijo_accepted']:
            expected_event_sequence.append(('iteration',row['iteration']))
    assert [(r['event'],r['call'] if r['event']=='objective' else r['iteration']) for r in trace]==expected_event_sequence
    for row,trial in zip(accepted_rows,accepted_trials):
        assert row['accepted_call']==row['objective_calls']==trial['call']
        assert row['armijo_accepted'] is True
        for key in ('alpha','armijo_bound','parent_weight_sha','parent_objective',
                    'directional_derivative','weight_sha','objective','Huber','L2_penalty',
                    'gradient_l2','gradient_linf','gradient_gap_upper_bound'):
            assert row[key]==trial[key]
    cert=checkpoint['certificate']
    for artifact in (fit,checkpoint,cert,checkpoint['solver']):
        assert artifact['final_accepted_call']==accepted['call']
    assert accepted['weight_sha']==fit['final_weight_sha']
    for key,recorded in (('objective','objective_value'),('Huber','Huber'),
                         ('L2_penalty','L2_penalty'),('gradient_l2','gradient_l2'),('gradient_linf','gradient_linf')):
        assert accepted[key]==cert[recorded]
    assert cert['gradient_linf_max']==1e-8 and accepted['gradient_linf']<=1e-8
    assert accepted['gradient_gap_upper_bound']<=1e-6
    return dict(PASS=True,initial_evaluation_counted=True,zero_initialization=True,
        objective_calls=len(objectives),accepted_iterations=len(accepted_rows),
        rejected_Armijo_trials=rejected_trials,rounded_Armijo_bound_equals_parent=rounded_bounds,
        final_accepted_call=accepted['call'],final_accepted_weight_sha=accepted['weight_sha'],
        exact_half_steps_and_fixed_c1=True,all_accepted_rows_match_trial=True,
        final_accepted_state_matches_certificate=True,optimizer_replays=0,
        intermediate_numerical_scope='Every trace row checked for provenance, order, count, alpha, Armijo arithmetic and accepted-state linkage. Initial/final objectives independently recomputed; intermediate matrices are not stored, so no claim of independent recomputation of every intermediate Newton direction.')


def regression_statistics(prediction,target,valid,index,chosen):
    rows=np.flatnonzero(valid.any(1))
    nonanchor=valid.copy();nonanchor[rows,index[rows]]=False
    selected=np.zeros_like(valid);selected[rows,chosen[rows]]=True
    result={}
    for label,mask in (('all_valid_candidates',valid),('nonanchor_valid_candidates',nonanchor),
                       ('actually_selected_candidates',selected)):
        result[label]={'candidate_pairs':int(mask.sum()),'scalar_observations':int(mask.sum()*2),'axes':{}}
        for axis,name in enumerate(('T','R')):
            truth=target[:,:,axis][mask];pred=prediction[:,:,axis][mask];res=pred-truth
            confusion=np.zeros((3,3),np.int64)
            for a,b in zip(np.sign(truth).astype(int)+1,np.sign(pred).astype(int)+1):confusion[a,b]+=1
            result[label]['axes'][name]=dict(MAE=float(np.mean(np.abs(res))) if len(res) else None,
                RMSE=float(np.sqrt(np.mean(res**2))) if len(res) else None,
                sign_accuracy=float(np.mean(np.sign(pred)==np.sign(truth))) if len(res) else None,
                exact_sign_confusion_true_rows_predicted_columns=confusion.tolist(),
                sign_order=['negative','zero','positive'])
    result['space']='Signed log1p normalized anchor-excess units; MAE/RMSE are not cm/deg. No clipping or sign threshold.'
    result['anchors_separated']='Anchor target/prediction are identically zero; inspect nonanchor statistics to avoid counting this identity as learned accuracy.'
    return result


def main():
    torch.set_num_threads(1)
    OLD.install_training_guard()
    from . import convex_train as T
    from scripts.research.pallet_pose_anchor_rbf_20261001_v1 import convex_train as PREV
    protocol=C.protocol('TRAIN_PROTOCOL');protocol_binding=C.bind(C.DOC/'TRAIN_PROTOCOL.json')
    assert T.metadata()==METADATA
    for key,value in METADATA.items():assert protocol[key]==value
    assert protocol['solver']==T.solver_config()==SOLVER_CONFIG
    assert protocol['certificate']==dict(optimizer_success=True,gap_upper_bound_max=1e-6,gradient_linf_max=1e-8)
    assert protocol['lambda_l2']==1e-4 and protocol['bias']==0. and protocol['train_rows']==2598
    complete=C.read(C.DOC/'TRAINING_COMPLETE.json')
    assert complete['complete'] and complete['all_certified'] and complete['fit_count']==4
    assert complete['models']==list(C.MODEL_NAMES) and complete['protocol']==protocol_binding
    assert complete['max_objective_calls_per_fit']==2000
    assert {p.name for p in (C.RAW/'fits').iterdir()}==set(C.MODEL_NAMES)
    for model in C.MODEL_NAMES:
        assert not (C.RAW/'fits'/model/'FAILED.json').exists()
        assert not (C.DOC/f'REJECTED_{model}.json').exists()
    assert complete['source_TRAIN_only'] and not complete['VAL_quality_read'] and not complete['real_targets_read']
    for key,value in METADATA.items():assert complete[key]==value
    for key in HASH_KEYS:assert set(complete[key+'_by_model'])==set(C.MODEL_NAMES)
    assert set(complete['final_accepted_call_by_model'])==set(C.MODEL_NAMES)
    basis_binding=C.bind(C.RBF_DOC/'RBF_BASIS.json')
    assert complete['basis_SHA_bind']==protocol['inputs']['rbf_basis']==basis_binding
    basis_artifact=C.read(C.RBF_DOC/'RBF_BASIS.json');basis=basis_artifact['basis']
    assert basis_artifact['complete'] and basis_artifact['PASS'] and not basis_artifact['labels_read']
    assert basis_artifact['protocol']==protocol['inputs']['basis_protocol']
    prefit=C.read(C.DOC/'PREFIT_REVIEW.json')
    assert prefit['complete'] and prefit['PASS'] and prefit['basis_SHA_bind']==basis_binding
    assert protocol['inputs']['prefit_review']==C.bind(C.DOC/'PREFIT_REVIEW.json')
    parent=OLD.load_training_inputs();anchor,index=V.load_anchor(parent)
    assert OLD.array_sha(index)==complete['anchor_index_sha']==basis_artifact['anchor_index_sha']
    assert len(parent['ids'])==2598 and parent['ids'][index<0].tolist()==['TEX__shard_04_f0110']
    assert basis_artifact['source_ids_sha']==OLD.array_sha(parent['ids'])
    assert basis_artifact['source_index_sha']==OLD.array_sha(parent['source_index'])
    assert basis_artifact['normalization_sha']==parent['normalization_sha']
    previous=C.read(C.RBF_DOC/'TRAIN_CONVERGENCE.json')
    assert previous['complete'] and previous['PASS'] and previous['source_TRAIN_only']
    for b in previous['bindings']:C.verify(b)
    locked_previous={b['path']:b for b in previous['bindings']}
    bindings=[protocol_binding,C.bind(C.DOC/'TRAINING_COMPLETE.json'),basis_binding,
        C.bind(C.DOC/'PREFIT_REVIEW.json'),C.bind(C.RBF_DOC/'TRAIN_CONVERGENCE.json'),
        C.bind(__file__),C.bind(T.__file__),C.bind(V.__file__),C.bind(PREV.__file__),
        C.bind(Path(classify.__code__.co_filename))]
    codebindings={b['path']:b for b in protocol['codes']}
    for module in (T,C,OLD,T.B):
        actual=C.bind(module.__file__);assert codebindings[actual['path']]==actual
    records={}
    for fb in complete['fits']:
        C.verify(fb);fit=C.read(C.ROOT/fb['path']);model=fit['model']
        assert model in C.MODEL_NAMES and model not in records
        assert fb==C.bind(C.DOC/f'FIT_{model}.json')
        assert fit['complete'] and fit['protocol']==protocol_binding and fit['fits_executed']==1
        assert fit['source_TRAIN_only'] and not fit['VAL_quality_read'] and not fit['real_targets_read']
        for key in ('START','trace','checkpoint'):C.verify(fit[key])
        ck=C.read(C.ROOT/fit['checkpoint']['path']);start=C.read(C.ROOT/fit['START']['path'])
        assert fit['checkpoint']==C.bind(C.RAW/'fits'/model/'final.json')
        assert ck['schema']=='pallet_pose_signed_axes_newton_linear253x2_v1'
        assert ck['model']==start['model']==model and ck['protocol']==start['protocol']==protocol_binding
        for obj in (ck,start,fit):
            for key,value in METADATA.items():assert obj[key]==value,(model,key)
            assert obj['basis_SHA_bind']==basis_binding and obj['anchor_index_sha']==OLD.array_sha(index)
        assert ck['rbf_basis_binding']==basis_binding and ck['rbf_basis']==basis
        assert ck['bias']==0. and ck['lambda_l2']==1e-4 and ck['loss_uses_original_valid_mask'] is True
        assert ck['normalization']=='old_float32_then_float64'
        assert ck['normalization_sha']==fit['normalization_sha']==parent['normalization_sha']
        np.testing.assert_array_equal(np.asarray(ck['mean'],np.float32),parent['mean'])
        np.testing.assert_array_equal(np.asarray(ck['std'],np.float32),parent['std'])
        names=T.candidate_names(model)
        assert ck['names']==start['names']==names
        parents=['R0']+([] if model=='R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
        raw,valid,errors=[np.concatenate([parent[key][m] for m in parents],axis=1) for key in ('features','valid','errors')]
        phi189=V.independent_context(raw,valid,index,parent['mean'],parent['std'])
        phi=V.independent_rbf(phi189,valid,basis)
        x=independent_difference(phi,valid,index)
        with np.errstate(all='raise'):scaled,target=independent_targets(errors,valid,anchor,index,parent['scale'])
        hashes=dict(signed_target_sha=OLD.array_sha(target),input_difference_sha=OLD.array_sha(x),
            base_context_sha=OLD.array_sha(phi),errors_sha=OLD.array_sha(errors),
            scaled_excess_sha=OLD.array_sha(scaled),original_valid_sha=OLD.array_sha(valid))
        for key in HASH_KEYS:
            assert hashes[key]==ck[key]==fit[key]==start[key]==complete[key+'_by_model'][model]==prefit['models'][model][key]
        np.testing.assert_array_equal(x,T.difference_inputs(raw,valid,index,ck['mean'],ck['std'],basis))
        s2,t2=T.signed_targets(errors,valid,anchor,index,parent['scale'])
        np.testing.assert_array_equal(scaled,s2);np.testing.assert_array_equal(target,t2)
        assert (~valid.any(1)).sum()==1 and start['no_valid_rows']==1
        assert start['train_rows']==2598 and start['source_ids_sha']==OLD.array_sha(parent['ids'])
        assert start['anchor_errors_sha']==OLD.array_sha(anchor)
        assert ck['axis_order']==start['axis_order']==['translation_cm','rotation_deg']
        np.testing.assert_array_equal(ck['signed_target_scales'],parent['scale'])
        np.testing.assert_array_equal(start['signed_target_scales'],parent['scale'])
        weight=np.asarray(ck['weight'],np.float64)
        assert weight.shape==(253,2) and np.isfinite(weight).all()
        assert OLD.array_sha(weight)==fit['final_weight_sha']
        assert start['initial_weight_sha']==OLD.array_sha(np.zeros((253,2),np.float64))
        assert start['solver']==protocol['solver'] and start['certificate_rule']==protocol['certificate']
        actual=independent_objective(weight,x,valid,target,with_hessian=True)
        native=T.objective(weight,x,valid,target)
        np.testing.assert_allclose(actual['gradient'],native['gradient'],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(actual['prediction'],native['prediction'],rtol=1e-12,atol=1e-12)
        cert=ck['certificate'];assert cert==fit['certificate'] and cert['PASS'] and cert['optimizer_success']
        assert cert['lambda_l2']==1e-4 and cert['max_gap_upper_bound']==1e-6
        assert cert['solver_rule']==METADATA['solver_rule']
        assert cert['gradient_linf_max']==1e-8 and actual['gradient_linf']<=1e-8
        assert abs(actual['gradient_linf']-cert['gradient_linf'])<=1e-10
        assert ck['solver']['success'] and ck['solver']['status']=='CONVERGED'
        assert ck['solver']['configuration']==SOLVER_CONFIG
        assert ck['final_accepted_call']==complete['final_accepted_call_by_model'][model]
        assert cert['loss_rule']==METADATA['loss_rule'] and cert['huber_delta']==1. and cert['output_dim']==2
        for key,ckey in (('objective','objective_value'),('Huber','Huber'),('L2_penalty','L2_penalty'),('gradient_l2','gradient_l2'),('gradient_linf','gradient_linf')):
            assert abs(actual[key]-cert[ckey])<=1e-10,(model,key)
        assert actual['certified_gap_upper_bound']<=1e-6
        assert abs(actual['certified_gap_upper_bound']-cert['gradient_l2_squared_over_2lambda'])<=1e-12
        H=T.hessian(x,valid,native['residual'])
        np.testing.assert_allclose(actual['Hessian'],H,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(actual['Hessian'],actual['Hessian'].T,rtol=1e-12,atol=1e-12)
        eigen=np.linalg.eigvalsh(actual['Hessian']);assert eigen[0]>=1e-4-1e-10
        np.testing.assert_allclose([eigen[0],eigen[-1]],
            [ck['solver']['final_hessian_min'],ck['solver']['final_hessian_max']],rtol=1e-8,atol=1e-10)
        runtime_axes=T.predict_axes(ck,raw,valid,index)
        np.testing.assert_allclose(runtime_axes,actual['prediction'],rtol=1e-12,atol=1e-12)
        runtime=T.score_candidates(ck,raw,valid,index)
        independent_score=np.where(valid,actual['prediction'].max(2),np.inf)
        np.testing.assert_allclose(runtime,independent_score,rtol=1e-12,atol=1e-12)
        chosen=V.select(runtime,valid,names)
        np.testing.assert_array_equal(chosen,V.select(independent_score,valid,names))
        np.testing.assert_array_equal(chosen,T.select_candidates(runtime,valid,names))
        available=np.flatnonzero(valid.any(1))
        assert np.all(runtime[available,chosen[available]]<=0.)
        # Retain the old discrete target only as a clearly secondary diagnostic,
        # never as supervision of the current two-axis regressor.
        legacy_target,safe=V.independent_targets(errors,valid,anchor,index,parent['scale'],names)
        stats,picked=V.score_statistics(runtime,valid,legacy_target,safe,errors,anchor,names)
        np.testing.assert_array_equal(chosen,picked)
        risk_stats=classify(chosen,valid,errors,anchor,index,parent['scale'])
        trace=[json.loads(line) for line in (C.ROOT/fit['trace']['path']).read_text().splitlines()]
        calls=[r for r in trace if r['event']=='objective'];iterations=[r for r in trace if r['event']=='iteration']
        assert len(calls)==fit['objective_calls']==cert['objective_calls'] and 0<len(calls)<=2000
        assert len(iterations)==fit['iterations']==cert['iterations']<=1000
        assert ck['fits_executed']==1 and ck['optimizer_steps']==fit['optimizer_steps']==len(iterations)
        assert [r['call'] for r in calls]==list(range(1,len(calls)+1))
        assert [r['iteration'] for r in iterations]==list(range(1,len(iterations)+1))
        assert calls[0]['weight_sha']==start['initial_weight_sha']
        assert calls[-1]['weight_sha']==fit['final_weight_sha'] and calls[-1]['objective']==cert['objective_value']
        assert calls[-1]['Huber']==cert['Huber']
        newton_trace=verify_newton_trace(trace,start,fit,ck)
        zero=independent_objective(np.zeros((253,2)),x,valid,target)
        assert abs(calls[0]['objective']-zero['objective'])<=1e-12
        for row in trace:
            for key,value in METADATA.items():assert row[key]==value
            for key in HASH_KEYS:assert row[key]==hashes[key]
            assert row['basis_SHA_bind']==basis_binding and row['anchor_index_sha']==OLD.array_sha(index)
        old_fit_binding=C.bind(C.RBF_DOC/f'FIT_{model}.json')
        assert locked_previous[old_fit_binding['path']]==old_fit_binding
        old_fit=C.read(C.ROOT/old_fit_binding['path']);C.verify(old_fit['checkpoint'])
        assert locked_previous[old_fit['checkpoint']['path']]==old_fit['checkpoint']
        old_ck=C.read(C.ROOT/old_fit['checkpoint']['path'])
        assert old_ck['rbf_basis']==basis and old_ck['basis_SHA_bind']==basis_binding
        assert old_ck['names']==names and old_ck['certificate']['PASS']
        old_w=np.asarray(old_ck['weight'],np.float64);assert old_w.shape==(253,)
        assert OLD.array_sha(phi)==previous['models'][model]['expanded_context_sha']
        old_score=np.where(valid,np.einsum('nkd,d->nk',phi,old_w),np.inf)
        old_runtime=PREV.score_candidates(old_ck,raw,valid,index)
        np.testing.assert_array_equal(old_runtime,old_score)
        old_stats,old_chosen=V.score_statistics(old_runtime,valid,legacy_target,safe,errors,anchor,names)
        old_risk=classify(old_chosen,valid,errors,anchor,index,parent['scale'])
        assert old_stats==previous['models'][model]['statistics']
        assert old_risk==previous['models'][model]['risk_statistics']
        records[model]=dict(independent_recompute=public_objective(actual),original_certificate=cert,
            objective_abs_difference=abs(actual['objective']-cert['objective_value']),
            gradient_vector_max_abs_difference=float(np.abs(actual['gradient']-native['gradient']).max()),
            Hessian_max_abs_difference=float(np.abs(actual['Hessian']-H).max()),
            independent_Hessian_min_eigenvalue=float(eigen[0]),independent_Hessian_max_eigenvalue=float(eigen[-1]),
            Hessian_scope='506x506 block matrix in C-order; zero generalized curvature at exact Huber kink; gap proof uses C1 strong convexity.',
            objective_calls=len(calls),iterations=len(iterations),newton_trace=newton_trace,
            statistics=stats,risk_statistics=risk_stats,
            regression_statistics=regression_statistics(runtime_axes,target,valid,index,chosen),
            legacy_discrete_target_accuracy_scope='Secondary comparison to old anchored whole-pose target; not the new regression target or optimization objective.',
            previous_fixed_rbf=dict(checkpoint=old_fit['checkpoint'],receipt=old_fit_binding,
                statistics=old_stats,risk_statistics=old_risk,original_choice_replay_PASS=True,
                objective_comparison_to_Huber_performed=False),
            changed_choice_count=int((chosen!=old_chosen).sum()),basis_SHA_bind=basis_binding,
            hashes=hashes,runtime_axes_sha=OLD.array_sha(runtime_axes),
            runtime_axis_and_selection_parity=True,all_invalid_frame_retained=True)
        bindings += [fb,*(fit[k] for k in ('START','trace','checkpoint')),old_fit_binding,old_fit['checkpoint']]
        print('SIGNED_AXES_NEWTON_TRAIN_VERIFIED',model,'gap',actual['certified_gap_upper_bound'],flush=True)
    assert set(records)==set(C.MODEL_NAMES)
    assert sum(r['objective_calls'] for r in records.values())==complete['total_objective_calls']
    assert sum(r['iterations'] for r in records.values())==complete['total_iterations']
    out=dict(complete=True,PASS=True,independent_check_PASS=True,created_at=C.now(),protocol=protocol_binding,
        **METADATA,models=records,bindings=bindings,basis_SHA_bind=basis_binding,
        source_TRAIN_only=True,frames=2598,available_anchor_rows=2597,failed_rows=1,
        VAL_quality_read=False,real_targets_read=False,raw_source_reference_reads=0,new_fits=0,optimizer_steps=0,
        alternative_policy_probes=0,method_success=False,goal_complete=False,
        comparison='Actual fixed signed-axis final models versus prior fixed RBF scalar models on identical TRAIN candidate errors. Huber and CE objective values are never directly compared.',
        objective_definition='Per-frame mean Huber(delta1) over original valid candidates and both axes; mean over all2598 frames; ridge1e-4 over506 weights.',
        validation='Independent scalar signed targets/differences, existing independent253 feature construction, Torch64 Huber/autograd, separate weighted-design block506 curvature.',
        read_paths=sorted(set(OLD.READS)))
    md=report(out)
    C.save(C.DOC/'TRAIN_CONVERGENCE_KO.md',md)
    C.save(C.DOC/'TRAIN_CONVERGENCE.json',out)
    print('SIGNED_AXES_NEWTON_INDEPENDENT_TRAIN_VERIFICATION_PASS_ALL4',flush=True)


def report(x):
    lines=['# Newton solver를 적용한 Signed T/R 회귀의 독립 TRAIN 검산','',
        '**독립 수치 검산 PASS.** 이 판정은 실제 T/R 개선이나 실사 일반화 판정이 아니다. 고정253 특징·anchor 차분·signedlog1p target을 독립 복원하고 TRAIN2,598행(유효2,597·실패1)을 모두 유지했다. runtime은 참조나 안전 mask 없이 max(predicted T,predicted R)를 사용한다.','',
        'zero 초기화에서 고정 generalized-Newton/Armijo로 얻은 네 accepted 상태를 검산했다. 초기·모든 trial 호출과 반감 alpha, Armijo 부등식, 최종 accepted-call 연결을 확인하며 재학습하지 않았다. 각 frame의 유효 후보×두 축 Huber(delta1) 평균을 전체2,598행으로 평균하고 모든506 가중치에λ=1e−4 ridge를 적용했다. Torch64 autograd와 독립 가중 design-matrix의506×506 block curvature를 대조했다. residual 절댓값1에서는 고전적 Hessian이 없으므로 사전 선언한0 곡률을 사용하며, gradient-gap 인증은 C1 강볼록성에 근거한다. START/CK/FIT/COMPLETE/PREFIT의 basis와6개SHA 및 전체trace를 검산했다.','',
        '| 모델 | Huber | objective | gradient L2 | gap 상한 | 호출 수 |','|---|---:|---:|---:|---:|---:|']
    for m,r in x['models'].items():
        q=r['independent_recompute'];lines.append(f"| {m} | {q['Huber']:.9f} | {q['objective']:.9f} | {q['gradient_l2']:.3g} | {q['certified_gap_upper_bound']:.3g} | {r['objective_calls']} |")
    lines += ['', '## 동일 TRAIN의 실제 선택 비교','',
        '이전 RBF 모델의 고정 score/선택/물리오차 통계를 정확히 재현한 뒤 현재 고정 모델과 비교했다. 이전 CE와 새 Huber는 다른 목적식이므로 손실값을 직접 비교하지 않는다. 다음 수는 전체2,598행 중 선택 수이며 실패1행은 별도 유지한다. 조건부2,597행의 비율과 각축 위반·T/R 중앙값/P90은 JSON에 보존했다.','',
        '| 모델 | anchor 이전→현재 | safe 개선 이전→현재 | unsafe 이전→현재 | 바뀐 선택 |','|---|---:|---:|---:|---:|']
    for m,r in x['models'].items():
        a=r['previous_fixed_rbf']['risk_statistics']['classes'];b=r['risk_statistics']['classes']
        values=[f"{a[k]['count']} → {b[k]['count']}" for k in ('anchor','safe_improvement','unsafe')]
        lines.append(f"| {m} | "+' | '.join(values)+f" | {r['changed_choice_count']} |")
    lines += ['', '## 회귀 target 오차','',
        '다음은 anchor를 제외한 유효 후보에 대한 signedlog1p 단위 MAE/RMSE와 정확 부호 일치율이다. cm/degree 오차가 아니며, anchor(정답·예측 모두0)를 포함해 회귀 성공률을 부풀리지 않는다. exact0도 독립 class로 남기고 별도 threshold를 도입하지 않았다.','',
        '| 모델 | 축 | MAE | RMSE | 부호 일치율 |','|---|---|---:|---:|---:|']
    for m,r in x['models'].items():
        for axis,v in r['regression_statistics']['nonanchor_valid_candidates']['axes'].items():
            lines.append(f"| {m} | {axis} | {v['MAE']:.6f} | {v['RMSE']:.6f} | {100*v['sign_accuracy']:.3f}% |")
    lines += ['', 'anchor 예측0과 선택된 max예측≤0은 모델 내부 성질이며 실제 두 물리오차 비악화를 보장하지 않는다. 기존 anchored discrete target 일치도는 JSON의 부가 진단이고 새 회귀의 학습 target이 아니다. 새 fit·optimizer step·VAL/실사 품질 열람·추가 선택 정책은 실행하지 않았다. 사전 source45와 원래/개입 실사5개 AND 판정은 별도로 유지한다.','',
        '[검산 JSON](TRAIN_CONVERGENCE.json) · [학습 계약](TRAIN_PROTOCOL.json) · [학습 전 검산](PREFIT_REVIEW_KO.md)','']
    return '\n'.join(lines)


def selfcheck():
    torch.set_num_threads(1)
    from . import convex_train as T
    assert T.metadata()==METADATA
    rng=np.random.default_rng(20261001506)
    valid=np.array([[1,1,1,1],[0,1,1,0],[0,0,0,0]],bool)
    index=np.array([0,1,-1],np.int64)
    errors=rng.uniform(.2,4,(3,4,2));errors[~valid]=np.inf
    anchor=np.full((3,2),np.inf);anchor[0]=errors[0,0];anchor[1]=errors[1,1]
    with np.errstate(all='raise'):
        scaled,y=independent_targets(errors,valid,anchor,index,[2.,3.])
        s2,y2=T.signed_targets(errors,valid,anchor,index,[2.,3.])
    np.testing.assert_array_equal(scaled,s2);np.testing.assert_array_equal(y,y2)
    phi=rng.normal(size=(3,4,253));phi[~valid]=0.
    xx=independent_difference(phi,valid,index)
    np.testing.assert_array_equal(xx,T.difference_from_context(phi,valid,index))
    weight=rng.normal(size=(253,2))*.003
    v=independent_objective(weight,xx,valid,y,True);native=T.objective(weight,xx,valid,y)
    for key,nkey in (('objective','value'),('Huber','Huber'),('L2_penalty','penalty')):
        assert abs(v[key]-native[nkey])<1e-12
    np.testing.assert_allclose(v['gradient'],native['gradient'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(v['Hessian'],T.hessian(xx,valid,native['residual']),rtol=1e-11,atol=1e-12)
    assert np.linalg.eigvalsh(v['Hessian'])[0]>=1e-4-1e-10
    for j in range(506):
        step=np.zeros((253,2));step.flat[j]=1e-5
        plus=independent_objective(weight+step,xx,valid,y)
        minus=independent_objective(weight-step,xx,valid,y)
        assert abs((plus['objective']-minus['objective'])/2e-5-v['gradient'].flat[j])<1e-7
    empty=independent_objective(weight,np.zeros((1,4,253)),np.zeros((1,4),bool),np.zeros((1,4,2)),True)
    assert empty['Huber']==0.
    np.testing.assert_array_equal(empty['gradient'],1e-4*weight)
    np.testing.assert_array_equal(empty['Hessian'],1e-4*np.eye(506))
    # Pure invented trace exercises rejection accounting without any optimizer.
    first=dict(event='objective',call=1,phase='initial',iteration=0,alpha=0.,evaluated=True,
        accepted_point_call_before=None,parent_weight_sha=None,parent_objective=None,
        directional_derivative=None,armijo_bound=None,armijo_accepted=True,weight_sha='zero',
        objective=1.,Huber=1.,L2_penalty=0.,gradient_l2=1.,gradient_linf=.5,gradient_gap_upper_bound=5000.)
    rejected=dict(first,call=2,phase='trial',iteration=1,alpha=1.,accepted_point_call_before=1,
        parent_weight_sha='zero',parent_objective=1.,directional_derivative=-.5,
        armijo_bound=1.-1e-4*.5,armijo_accepted=False,weight_sha='reject',objective=1.1,Huber=1.1)
    final=dict(rejected,call=3,alpha=.5,armijo_bound=1.-1e-4*.5*.5,armijo_accepted=True,
        weight_sha='final',objective=.8,Huber=.8,gradient_l2=1e-9,gradient_linf=1e-9,gradient_gap_upper_bound=5e-15)
    iteration={k:final[k] for k in ('alpha','armijo_bound','armijo_accepted','parent_weight_sha',
        'parent_objective','directional_derivative','weight_sha','objective','Huber','L2_penalty',
        'gradient_l2','gradient_linf','gradient_gap_upper_bound')}
    iteration.update(event='iteration',iteration=1,objective_calls=3,accepted_call=3)
    trace=[first,rejected,final,iteration]
    st=dict(initial_weight_sha='zero',initialization='all_zero_float64',solver=SOLVER_CONFIG)
    fit=dict(objective_calls=3,iterations=1,final_accepted_call=3,final_weight_sha='final')
    ck=dict(final_accepted_call=3,solver=dict(configuration=SOLVER_CONFIG,status='CONVERGED',
        success=True,last_evaluated_call=3,final_accepted_call=3),certificate=dict(final_accepted_call=3,
        objective_value=.8,Huber=.8,L2_penalty=0.,gradient_l2=1e-9,gradient_linf=1e-9,gradient_linf_max=1e-8))
    assert verify_newton_trace(trace,st,fit,ck)['rejected_Armijo_trials']==1
    import copy
    for row,key,value in ((2,'accepted_point_call_before',2),(2,'alpha',.25),
                          (1,'armijo_accepted',True),(3,'accepted_call',2)):
        wrong=copy.deepcopy(trace);wrong[row][key]=value
        try:verify_newton_trace(wrong,st,fit,ck)
        except AssertionError:pass
        else:raise AssertionError(('INVALID_NEWTON_TRACE_ACCEPTED',row,key))
    assert OLD.READS is None
    print('SIGNED_AXES_NEWTON_TRAIN_VERIFIER_SELFCHECK_PASS_NO_ARTIFACT_READS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('selfcheck','verify'))
    with threadpool_limits(limits=1):
        (selfcheck if parser.parse_args().stage=='selfcheck' else main)()
