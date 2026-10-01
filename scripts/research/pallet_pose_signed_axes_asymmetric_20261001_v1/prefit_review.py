"""Independent asymmetric-loss PREFIT; actual TRAIN reads need root authorization."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from . import common as C
from scripts.research.pallet_pose_signed_axes_direction_20261001_v1 import prefit_review as P
from scripts.research.pallet_pose_signed_axes_sign_20261001_v1 import prefit_review as N

OLD_HASH_KEYS=P.OLD_HASH_KEYS
EXTRA_HASH_KEYS=P.EXTRA_HASH_KEYS
HASH_KEYS=OLD_HASH_KEYS+EXTRA_HASH_KEYS
READS=set()
LOSS_RULE='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_ASYMMETRIC_HUBER_PLUS_SIGN_LOGISTIC'
EXTRA_METADATA=dict(underprediction_coefficient=1.,underprediction_cost=2.,overprediction_cost=1.,underprediction_zero_curvature=0.)
array_sha=P.array_sha
exact=P.exact
direction_difference=P.direction_difference
DIRECTION_RULE=P.DIRECTION_RULE
DIRECTION_NORMALIZATION=P.DIRECTION_NORMALIZATION


def guard():
    sys.dont_write_bytecode=True
    arrays={C.PARENT_RAW/'SOURCE_FEATURES.npz',C.PARENT_RAW/'SOURCE_TRAIN_LABELS.npz',C.DIRECTION_RAW/'TRAIN_DIRECTIONS.npz'}
    outputs={C.DOC/'PREFIT_REVIEW.json',C.DOC/'PREFIT_REVIEW_KO.md'}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();name=str(p);mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing and p.is_relative_to(C.ROOT):assert p in outputs,('PREFIT_OUTPUT_ONLY',name);return
        assert not any(t in name for t in ('/data/evaluation/','/real_gt_v2/','/annotations/','GEOMETRY_RESOLVED_POSE_GT',
            'GEOMETRY_SIDETABLE','SYNTH_RECORDS','SYNTH_LABELS','SOURCE_VAL_','REAL_RESULTS','REAL_CHOICES','POSE_METRICS',
            'TRUTH_FOR_DISPLAY','/fits/','/model_parameters/','TRAIN_CONVERGENCE')),('NO_WEIGHTS_OR_QUALITY',name)
        assert p.suffix.lower() not in ('.png','.jpg','.jpeg','.pt','.pth','.onnx'),name
        if p.suffix=='.npz':assert p in arrays,name
        if p.is_relative_to(C.ROOT):READS.add(str(p.relative_to(C.ROOT)))
    sys.addaudithook(hook)


def independent_objective(W,x,valid,target,ridge=1e-4):
    n,k,d=x.shape
    assert W.shape==(d,2) and valid.shape==(n,k) and target.shape==(n,k,2)
    assert np.isfinite(x).all() and np.isfinite(target).all() and not x[~valid].any() and not target[~valid].any()
    prediction=np.zeros_like(target);residual=np.zeros_like(target);gradient=np.zeros_like(W)
    symmetric=0.;under=0.;logistic=0.
    for i in range(n):
        active=np.flatnonzero(valid[i])
        if not len(active):continue
        coefficient=1/(n*len(active)*2)
        for j in active:
            for axis in range(2):
                p=float(np.dot(x[i,j],W[:,axis]));e=p-float(target[i,j,axis]);a=abs(e)
                prediction[i,j,axis]=p;residual[i,j,axis]=e
                base=.5*e*e if a<=1 else a-.5
                slope=e if a<=1 else np.sign(e)
                extra=base if e<0 else 0.;extra_slope=slope if e<0 else 0.
                sign,sign_grad,_=N.sign_term(p,float(target[i,j,axis]))
                symmetric+=coefficient*base;under+=coefficient*extra;logistic+=coefficient*sign
                gradient[:,axis]+=coefficient*(slope+extra_slope+sign_grad)*x[i,j]
    penalty=ridge/2*float(np.sum(W*W));huber=symmetric+under
    return dict(value=huber+logistic+penalty,Huber=huber,Huber_symmetric=symmetric,Huber_underprediction=under,
        Sign_logistic=logistic,penalty=penalty,gradient=gradient+ridge*W,prediction=prediction,residual=residual)


def independent_hessian(x,valid,residual,target,ridge=1e-4):
    n,k,d=x.shape;result=np.eye(2*d)*ridge
    for i in range(n):
        active=np.flatnonzero(valid[i])
        if not len(active):continue
        coefficient=1/(n*len(active)*2)
        for j in active:
            for axis in range(2):
                e=float(residual[i,j,axis]);p=e+float(target[i,j,axis])
                curvature=float(abs(e)<1.)+float(-1.<e<0.)+N.sign_term(p,float(target[i,j,axis]))[2]
                result[axis::2,axis::2]+=coefficient*curvature*np.outer(x[i,j],x[i,j])
    return result


def torch_loss(W,x,valid,target):
    p=torch.einsum('nkd,da->nka',x,W);e=p-target
    symmetric=torch.nn.functional.huber_loss(p,target,reduction='none',delta=1.)
    # Strict branch gives the declared additional curvature0 at e=0.
    under=torch.where(e<0,symmetric,torch.zeros_like(symmetric))
    sign=torch.sign(target);logit=-sign*p
    logistic=torch.where(target!=0,torch.logaddexp(torch.zeros_like(logit),logit),torch.zeros_like(logit))
    count=valid.sum(1).clamp(min=1).to(torch.float64)
    data=(((symmetric+under+logistic)*valid[:,:,None]).sum((1,2))/(2*count)).mean()
    return data+.5e-4*W.square().sum()


def code_parity(T,operators):
    previous=ast.parse((C.PREVIOUS_HERE/'convex_train.py').read_text());current=ast.parse(Path(T.__file__).read_text())
    names=('validate_anchor','signed_targets','difference_from_context','difference_inputs','predict_axes','score_candidates','solver_config')
    for name in names:
        a=next(n for n in previous.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),('INPUT_TARGET_RUNTIME_CHANGED',name)
    gates={}
    if operators:
        for filename,funcs in [('evaluate_source.py',('comparisons_for','chosen_pose','verify_source_reference_chain')),
                               ('evaluate_real.py',('gates_for','combined_gates'))]:
            a=ast.parse((C.PREVIOUS_HERE/filename).read_text());b=ast.parse((C.HERE/filename).read_text())
            for name in funcs:
                fa=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name)
                fb=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==name)
                assert ast.dump(fa)==ast.dump(fb),(filename,name)
            gates[filename]=dict(functions=list(funcs),previous=C.bind(C.PREVIOUS_HERE/filename),current=C.bind(C.HERE/filename))
    assert T.LOSS_RULE==LOSS_RULE and T.FEATURE_DIM==271 and T.ALL_HASH_KEYS==HASH_KEYS
    assert T.LAMBDA==1e-4 and T.MAX_ITER==1000 and T.MAX_CALLS==2000 and T.GAP_MAX==1e-6
    for key,value in EXTRA_METADATA.items():assert T.metadata()[key]==value
    return dict(PASS=True,input_target_runtime_solver_config_AST_exact=list(names),evaluation_AST_parity=gates,
        previous_code=C.bind(C.PREVIOUS_HERE/'convex_train.py'),current_code=C.bind(T.__file__),
        changed_factor='Only fixed coefficient1 Huber(min(prediction-target,0),1) added to existing same271-input Huber+sign+ridge loss; metadata/logging reflect decomposition.')


def synthetic(T):
    from scripts.research.pallet_pose_signed_axes_direction_20261001_v1 import convex_train as PREV
    rng=np.random.default_rng(202610012712)
    valid=np.array([[1,1,1,1],[0,1,1,0],[1,0,0,0],[0,0,0,0],[1,1,0,1]],bool)
    index=np.array([0,1,0,-1,1]);rows=np.flatnonzero(valid.any(1))
    raw=rng.normal(size=(5,4,94)).astype(np.float32);raw[~valid]=np.nan
    direction=rng.normal(size=(5,4,18)).astype(np.float32);direction[~valid]=0.
    mean=np.zeros(94,np.float32);std=np.ones(94,np.float32);mean18=np.zeros(18,np.float32);std18=np.ones(18,np.float32)
    centers=rng.normal(size=(64,189));width=float(np.median([np.sum((centers[i]-centers[j])**2) for i in range(64) for j in range(i+1,64)]))
    basis=dict(schema='pallet_pose_anchor_rbf_runtime_v1',context_dim=189,rbf_dim=64,centers=centers.tolist(),bandwidth_squared=width,normalization_sha=array_sha(np.stack([mean,std])))
    _,_,_,x=P.extended_inputs(raw,valid,index,mean,std,basis,direction,mean18,std18)
    exact(x,T.difference_inputs(raw,valid,index,mean,std,basis,direction,mean18,std18))
    exact(x,PREV.difference_inputs(raw,valid,index,mean,std,basis,direction,mean18,std18))
    target=rng.normal(size=(*valid.shape,2));target[~valid]=0.;target[rows,index[rows]]=0.
    target[0,2,0]=0.;W=rng.normal(size=(271,2))*.003
    actual=T.objective(W,x,valid,target);reference=independent_objective(W,x,valid,target)
    for key in ('value','Huber','Huber_symmetric','Huber_underprediction','Sign_logistic','penalty','gradient','prediction','residual'):
        np.testing.assert_allclose(actual[key],reference[key],rtol=1e-11,atol=1e-12)
    H=independent_hessian(x,valid,reference['residual'],target)
    np.testing.assert_allclose(H,T.hessian(x,valid,actual['residual'],target),rtol=1e-11,atol=1e-12)
    xx=torch.tensor(x,dtype=torch.float64);vv=torch.tensor(valid);yy=torch.tensor(target,dtype=torch.float64)
    w=torch.tensor(W.ravel(),dtype=torch.float64,requires_grad=True)
    loss=lambda flat:torch_loss(flat.reshape(271,2),xx,vv,yy)
    value=loss(w);gradient,=torch.autograd.grad(value,w)
    hessian=torch.autograd.functional.hessian(loss,w).detach().numpy()
    np.testing.assert_allclose(float(value.detach()),actual['value'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(gradient.detach().numpy().reshape(271,2),actual['gradient'],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(hessian,H,rtol=1e-11,atol=1e-12)
    eps=1e-6;gaps=[];hgap=[]
    # Anchors have identically zero x. All other fixture residuals avoid kinks.
    finite=valid.copy();finite[rows,index[rows]]=False
    assert min(np.min(abs(actual['residual'][finite])),np.min(abs(abs(actual['residual'][finite])-1.)))>1e-3
    for j in range(542):
        shift=np.zeros_like(W);shift.ravel()[j]=eps
        plus=independent_objective(W+shift,x,valid,target);minus=independent_objective(W-shift,x,valid,target)
        gaps.append(abs((plus['value']-minus['value'])/(2*eps)-actual['gradient'].ravel()[j]))
        hgap.append(float(np.max(abs((plus['gradient']-minus['gradient']).ravel()/(2*eps)-H[:,j]))))
    assert max(gaps)<1e-8 and max(hgap)<1e-8 and np.linalg.eigvalsh(H)[0]>=1e-4-1e-10
    # Exact boundaries: extra curvature0 at e=0 and |e|=1; target0 still
    # receives asymmetric Huber while sign logistic remains0.
    boundaries=[]
    for residual in (-1000.,-2.,-1.,-.3,0.,.3,1.,2.,1000.):
        xs=np.ones((2,1,1));vs=np.array([[True],[False]]);xs[~vs]=0.;ys=np.zeros((2,1,2))
        ws=np.array([[residual,residual]],np.float64)
        a=T.objective(ws,xs,vs,ys);b=independent_objective(ws,xs,vs,ys)
        for key in ('value','Huber','Huber_symmetric','Huber_underprediction','Sign_logistic','penalty','gradient'):
            np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(T.hessian(xs,vs,a['residual'],ys),independent_hessian(xs,vs,b['residual'],ys),rtol=1e-12,atol=1e-12)
        assert a['Sign_logistic']==0. and (a['Huber_underprediction']==a['Huber_symmetric'] if residual<0 else a['Huber_underprediction']==0.)
        boundaries.append(dict(residual=residual,Huber_symmetric=a['Huber_symmetric'],Huber_underprediction=a['Huber_underprediction'],extra_curvature=float(-1.<residual<0.)))
    empty=T.objective(W,np.zeros((1,4,271)),np.zeros((1,4),bool),np.zeros((1,4,2)))
    assert empty['Huber']==empty['Huber_underprediction']==empty['Sign_logistic']==0.;exact(empty['gradient'],1e-4*W)
    # Same score at the same weight; changed training metadata is not a runtime input.
    common=dict(normalization='old_float32_then_float64',names=T.candidate_names('UNION_s1'),bias=0.,lambda_l2=1e-4,
        mean=mean.tolist(),std=std.tolist(),weight=W.tolist(),rbf_basis=basis,rbf_basis_binding={'invented':True},basis_SHA_bind={'invented':True},
        direction_mean=mean18.tolist(),direction_std=std18.tolist(),direction_normalization_sha=array_sha(np.stack([mean18,std18])),direction_receipt_binding={'invented':True})
    ck=dict(common,schema=T.CHECKPOINT_SCHEMA,**T.metadata());before=dict(common,schema=PREV.CHECKPOINT_SCHEMA,**PREV.metadata())
    expected=PREV.score_candidates(before,raw,valid,index,direction)
    original=T.signed_targets
    def denied(*args,**kwargs):raise AssertionError('NO_RUNTIME_TARGET')
    try:
        T.signed_targets=denied
        exact(T.score_candidates(ck,raw,valid,index,direction),expected)
    finally:T.signed_targets=original
    return dict(PASS=True,invented_only=True,same271_input_and_same_weight_runtime_byte_exact=True,
        independent_scalar_and_Torch_PASS=True,parameters_checked=542,Hessian_shape=[542,542],
        gradient_finite_difference_max=max(gaps),Hessian_finite_difference_max=max(hgap),
        Torch_Hessian_max_difference=float(np.max(abs(hessian-H))),boundary_cases=boundaries,
        underprediction_cost_fixed2_overprediction1=True,target_zero_Huber_not_masked=True,
        sign_target_zero_still_masked=True,full_frame_denominator_and_invalid_zero=True,
        runtime_target_disabled_PASS=True,actual_inputs_or_weights_read=0)


def load_cached_inputs():
    data=P.load_cached_inputs()
    C.verify(C.read(C.PREVIOUS_DOC/'TRAIN_PROTOCOL_SHA.json'))
    protocol=C.read(C.PREVIOUS_DOC/'TRAIN_PROTOCOL.json');C.verify(protocol['inputs']['prefit_review'])
    previous=C.read(C.ROOT/protocol['inputs']['prefit_review']['path'])
    assert previous['complete'] and previous['PASS'] and previous['trainer']==C.bind(C.PREVIOUS_HERE/'convex_train.py')
    assert previous['frames']==2598 and previous['available_anchor_rows']==2597 and previous['failed_rows_retained']==1
    for key,value in data['inputs'].items():assert previous['inputs'][key]==value,('PREVIOUS_CACHED_INPUT_CHANGED',key)
    data['previous_direction']=previous
    data['inputs']=dict(data['inputs'],previous_direction_prefit=protocol['inputs']['prefit_review'],previous_direction_protocol=C.bind(C.PREVIOUS_DOC/'TRAIN_PROTOCOL.json'))
    return data


def actual_review(T):
    assert not (C.RAW/'fits').exists(),'PREFIT must precede every new fit.'
    data=load_cached_inputs();records={}
    for model in C.MODEL_NAMES:
        experts=['R0']+([] if model=='R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
        raw,valid,errors=[np.concatenate([data[key][m] for m in experts],axis=1) for key in ('features','valid','errors')]
        direction=np.concatenate([data['direction'][m] for m in experts],axis=1)
        phi,base,extra,x=P.extended_inputs(raw,valid,data['index'],data['mean'],data['std'],data['basis'],direction,data['mean18'],data['std18'])
        exact(x,T.difference_inputs(raw,valid,data['index'],data['mean'],data['std'],data['basis'],direction,data['mean18'],data['std18']))
        with np.errstate(all='raise'):scaled,target=N.signed_targets(errors,valid,data['anchor'],data['index'],data['scale'])
        s2,y2=T.signed_targets(errors,valid,data['anchor'],data['index'],data['scale']);exact(scaled,s2);exact(target,y2)
        hashes=dict(signed_target_sha=array_sha(target),input_difference_sha=array_sha(base),base_context_sha=array_sha(phi),errors_sha=array_sha(errors),
            scaled_excess_sha=array_sha(scaled),original_valid_sha=array_sha(valid),direction_raw_sha=array_sha(direction),direction_difference_sha=array_sha(extra),extended_input_sha=array_sha(x))
        for key in HASH_KEYS:assert hashes[key]==data['previous_direction']['models'][model][key]
        for key in P.OLD_HASH_KEYS:assert hashes[key]==data['previous']['models'][model][key]
        records[model]=dict(**hashes,frames=2598,failed_rows=1,valid_candidates=int(valid.sum()),input_shape=list(x.shape),target_shape=list(target.shape),
            previous_direction_nine_hashes_exact=True,previous_sign_six_hashes_exact=True,
            raw_features_sha=array_sha(raw),anchor_index_sha=array_sha(data['index']),anchor_errors_sha=array_sha(data['anchor']),
            normalization_sha=data['inputs']['normalization_sha'],direction_normalization_sha=data['inputs']['direction_normalization_sha'],
            direction_receipt_binding=data['inputs']['direction_receipt'],basis_SHA_bind=data['basis_binding'],
            all_invalid_row_retained=True,actual_data_objectives_or_policy_trials=0)
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
        synthetic=toy,unchanged_function_audit=parity,trainer=code,reviewer=C.bind(__file__),
        independent_input_helper=C.bind(P.__file__),independent_sign_helper=C.bind(N.__file__),
        source_TRAIN_cached_label_values_read=True,raw_source_reference_reads=0,VAL_quality_read=False,real_targets_read=False,
        new_fits=0,optimizer_steps=0,actual_data_objective_trials=0,actual_data_policy_probes=0,prior_weights_used=False,
        basis_reselected=False,bandwidth_changed=False,direction_normalization_reselected=False,
        new_reference_metric_calculations=0,image_forwards=0,new_PnP_calls=0,read_paths=sorted(READS),method_success=False,goal_complete=False)
    md=f'''# 과소예측 추가 Huber: 학습 전 독립 검산

**PREFIT PASS는 입력·목적식 계약의 검산이며 학습이나 T/R 개선 결과가 아니다.** 직전 P의271입력·타깃·원래 valid mask·TRAIN2,598행(실패1 포함)과 모든9해시가 같다. N의 원래6해시도 유지한다. R0 TRAIN 정규화94/18과 RBF basis는 다시 선택하지 않았다.

단일 변경은 e=p−y일 때 기존 Huber(e,1)에 Huber(min(e,0),1)을 고정계수1로 추가하는 것이다. 따라서 대칭 Huber와 추가 과소예측 Huber를 별도 기록하고 둘의 합을 `Huber`로 정의한다. J=Huber+Sign_logistic+L2이다. target0에도 Huber는 적용하며, 기존 sign logistic만 nonzero target에 적용한다. 원래 행별 유효후보×2축 평균 후 전체2,598행 평균과λ1e−4 ridge를 유지한다.

독립 scalar 식·Torch64 autograd·542개 좌표 중앙차분·542×542 Hessian을 대조했다. gradient 차분 최대{toy['gradient_finite_difference_max']:.3g}, Hessian 차분 최대{toy['Hessian_finite_difference_max']:.3g}이다. 추가 Huber 곡률은−1<e<0에서1, e=0과e=−1에서는 사전 지정0이다. 원래 Huber kink·logistic 곡률·ridge는 유지한다. 추가항은 볼록이고 연속미분 가능하며 모든542개 계수의 ridge가 강볼록성을 유지한다.

같은 가중치에서 직전 runtime 점수와 byte 단위로 같고, target 생성함수를 막아도 점수를 계산한다. 실제 TRAIN에서는 입력·타깃 해시만 재구성했으며 이전 weight·목적식·선택 정책을 실행하지 않았다. 새 fit·PnP·이미지 forward·VAL/실사 참조0이다.

대칭 Huber의 방향별 비용을 바꾸는 가설 검산이다. 이전 risk-margin CE·utility/부호 회귀의 음성 결과는 반대근거이며, 보수화가 anchor 복귀만 늘려 엄격한 공동 개선을 없앨 수 있다. 비대칭 손실 자체의 학술적 새로움이나 성공을 주장하지 않는다. source45 및 원래+matched 실사5기준은 유지한다.

[검산 JSON](PREFIT_REVIEW.json) · [직전 동일 입력 검산](../pallet_pose_signed_axes_direction_20261001_v1/PREFIT_REVIEW_KO.md)
'''
    C.save(C.DOC/'PREFIT_REVIEW_KO.md',md);out['note']=C.bind(C.DOC/'PREFIT_REVIEW_KO.md')
    C.save(C.DOC/'PREFIT_REVIEW.json',out)
    print('ASYMMETRIC_PREFIT_PASS',C.bind(C.DOC/'PREFIT_REVIEW.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('selfcheck','write'))
    run(parser.parse_args().stage=='write')
