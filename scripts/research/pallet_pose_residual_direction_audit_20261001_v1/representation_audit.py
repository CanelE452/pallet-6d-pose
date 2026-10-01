"""Sealed TRAIN-only input redundancy and exact representation-collision audit.

No model weights, objective probes, selection policy, optimizer, PnP or raw GT.
Signed cached TRAIN labels are used only for exact-collision descriptions.
The SVD helper takes input matrices and has no label argument.
"""
from . import common as C
from scripts.research.pallet_pose_signed_axes_sign_20261001_v1 import convex_train as T
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

EXPERTS=('R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3')
HYP=('long-face-front','short-face-front')
ZERO_BYTES_RULE='canonicalize_both_signed_zeros_to_positive_zero_no_rounding'
SVD_TOL_RULE='max(shape)*float64_eps*smax'
NOVELTY_FROBENIUS_MAX=1e-4
READS=None


def config():
    return dict(base_dimension=253,extra_dimension=18,extended_dimension=271,
        zero_byte_rule=ZERO_BYTES_RULE,svd_population='all_valid_nonanchor_candidates',
        svd_tolerance=SVD_TOL_RULE,novelty_frobenius_max=NOVELTY_FROBENIUS_MAX,
        normalization='R0_TRAIN_valid_float32_mean_std_floor_1e-6_then_float64_candidate_minus_anchor',
        target_use='exact_collision_diagnostics_only_not_projection_or_fitting')


def array_sha(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode());h.update(value.tobytes())
    return h.hexdigest()


def bound_read(binding):
    C.verify(binding)
    return C.read(C.ROOT/binding['path'])


def install_guard():
    global READS
    if READS is not None:return
    READS=set();sys.dont_write_bytecode=True
    json_allow={C.DOC/'REPRESENTATION_PROTOCOL.json',C.DOC/'REPRESENTATION_PROTOCOL_SHA.json',
        C.DOC/'FEATURE_AUDIT.json',C.DOC/'AUDIT_PROTOCOL.json',C.DOC/'AUDIT_PROTOCOL_SHA.json',
        C.PARENT_DOC/'TRAIN_PROTOCOL.json',C.SIGN_DOC/'PREFIT_REVIEW.json',
        C.PARENT_DOC/'SOURCE_CONTRACT.json',C.PARENT_DOC/'SOURCE_FEATURE_LOCK.json',
        C.RBF_DOC/'RBF_BASIS.json'}
    arrays={C.PARENT_RAW/'SOURCE_FEATURES.npz',C.PARENT_RAW/'SOURCE_TRAIN_LABELS.npz',C.RAW/'TRAIN_DIRECTIONS.npz'}
    outputs={C.DOC/'REPRESENTATION_AUDIT.json',C.DOC/'REPRESENTATION_AUDIT_KO.md',C.RAW/'REPRESENTATION_ARRAYS.npz'}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();name=str(p);mode,flags=args[1:3]
        writing=((isinstance(mode,str) and any(c in mode for c in 'wax+')) or
            (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))))
        if writing:
            assert p in outputs,('REPRESENTATION_WRITE_DENIED',name);return
        assert not any(token in name for token in ('/data/evaluation/','/real_gt_v2/','/annotations/',
            'SOURCE_VAL','REAL_CHOICES','REAL_FEATURE','POSE_METRICS','GEOMETRY_RESOLVED_POSE_GT',
            'GEOMETRY_SIDETABLE','SOURCE_MANIFEST','SYNTH_LABELS','SYNTH_RECORDS',
            '/model_parameters/','/fits/','REJECTED_')),('REPRESENTATION_QUALITY_OR_WEIGHT_DENIED',name)
        assert p.suffix.lower() not in ('.png','.jpg','.jpeg','.pt','.pth','.onnx'),('REPRESENTATION_IMAGE_WEIGHT_DENIED',name)
        if p.suffix=='.npz':assert p in arrays,('REPRESENTATION_ARRAY_DENIED',name)
        if p.is_relative_to(C.ROOT):
            if p.suffix=='.json':assert p in json_allow,('REPRESENTATION_JSON_DENIED',name)
            READS.add(str(p.relative_to(C.ROOT)))
    sys.addaudithook(hook)


def exact_groups(matrix):
    """Actual byte equality; normalize signed zeros, with no numeric rounding."""
    matrix=np.asarray(matrix)
    assert matrix.dtype==np.float64 and matrix.ndim==2 and np.isfinite(matrix).all()
    canonical=np.ascontiguousarray(matrix.copy())
    canonical[canonical==0.]=0.
    index={};groups=[];keys=[];assignment=np.empty(len(matrix),np.int64)
    for i,row in enumerate(canonical):
        key=row.tobytes()
        if key not in index:
            index[key]=len(groups);groups.append([]);keys.append(hashlib.sha256(key).hexdigest())
        k=index[key];assignment[i]=k;groups[k].append(i)
    return assignment,groups,keys


def target_description(target):
    target=np.asarray(target,np.float64)
    assert target.ndim==2 and target.shape[1]==2 and len(target) and np.isfinite(target).all()
    result={}
    for j,axis in enumerate(('T','R')):
        values=target[:,j];lo=float(values.min());hi=float(values.max())
        signs=sorted(set(np.sign(values).astype(int).tolist()))
        result[axis]=dict(minimum=lo,maximum=hi,range=hi-lo,distinct_values=len(set(values.tolist())),
            signs=signs,target_conflict=lo!=hi,sign_conflict=len(signs)>1,strict_opposite=lo<0.<hi)
    return result


def group_summary(groups,target,anchor):
    repeated=[members for members in groups if len(members)>1]
    out=dict(groups=len(groups),repeated_groups=len(repeated),repeated_candidate_rows=sum(map(len,repeated)),
        maximum_group_size=max(map(len,groups),default=0),
        anchor_only_groups=0,nonanchor_only_groups=0,anchor_nonanchor_groups=0,axes={})
    for members in groups:
        n=int(anchor[members].sum())
        out['anchor_only_groups' if n==len(members) else 'nonanchor_only_groups' if n==0 else 'anchor_nonanchor_groups']+=1
    for axis in ('T','R'):
        records=[(members,target_description(target[members])[axis]) for members in groups]
        out['axes'][axis]={kind:dict(groups=sum(r[kind] for _,r in records),
            candidate_rows=sum(len(members) for members,r in records if r[kind]))
            for kind in ('target_conflict','sign_conflict','strict_opposite')}
    return out


def collision_audit(x,extra,target,valid,anchor_index,ids,names):
    """Compare old/new equality classes without fitting or choosing poses."""
    x,extra,target=np.asarray(x),np.asarray(extra),np.asarray(target)
    valid,anchor_index=np.asarray(valid),np.asarray(anchor_index)
    assert x.dtype==extra.dtype==target.dtype==np.float64 and valid.dtype==bool
    assert x.shape[:2]==extra.shape[:2]==target.shape[:2]==valid.shape
    assert target.shape[2]==2 and len(ids)==len(valid) and len(names)==valid.shape[1]
    assert np.isfinite(x).all() and np.isfinite(extra).all() and np.isfinite(target).all()
    assert not x[~valid].any() and not extra[~valid].any() and not target[~valid].any()
    present=valid.any(1);rows=np.flatnonzero(present)
    assert np.array_equal(present,anchor_index>=0) and valid[rows,anchor_index[rows]].all()
    anchors=np.zeros(valid.shape,bool);anchors[rows,anchor_index[rows]]=True
    assert not x[anchors].any() and not extra[anchors].any() and not target[anchors].any()
    positions=np.argwhere(valid);flat_anchor=anchors[valid];y=target[valid]
    old_ids,old_groups,old_hash=exact_groups(x[valid])
    new_ids,new_groups,new_hash=exact_groups(np.concatenate([x[valid],extra[valid]],axis=1))
    repeated=[]
    for group_id,members in enumerate(old_groups):
        if len(members)==1:continue
        descriptions=target_description(y[members]);subgroup_ids=sorted(set(new_ids[members].tolist()))
        a=int(flat_anchor[members].sum());subgroups=[]
        for sub in subgroup_ids:
            submembers=[v for v in members if new_ids[v]==sub]
            subgroups.append(dict(extended_group_id=sub,vector_sha256=new_hash[sub],candidate_rows=len(submembers),
                anchor_rows=int(flat_anchor[submembers].sum()),target=target_description(y[submembers])))
        # Full memberships are in NPZ group assignments and original ids/order.
        examples=[dict(id=str(ids[positions[v,0]]),candidate=names[positions[v,1]],
            frame_index=int(positions[v,0]),candidate_index=int(positions[v,1])) for v in members[:5]]
        repeated.append(dict(old_group_id=group_id,vector_sha256=old_hash[group_id],candidate_rows=len(members),
            anchor_rows=a,nonanchor_rows=len(members)-a,mixed_anchor_nonanchor=0<a<len(members),
            forced_anchor_only=a==len(members),target=descriptions,
            extended_subgroups=len(subgroups),split=len(subgroups)>1,subgroups=subgroups,
            first_five_identity_examples=examples))
    by_axis={}
    for axis in ('T','R'):
        by_axis[axis]={}
        for kind in ('target_conflict','sign_conflict','strict_opposite'):
            affected=[g for g in repeated if g['target'][axis][kind]]
            unresolved=[g for g in affected if any(sub['target'][axis][kind] for sub in g['subgroups'])]
            by_axis[axis][kind]=dict(old_groups=len(affected),split_groups=sum(g['split'] for g in affected),
                fully_resolved_groups=len(affected)-len(unresolved),unresolved_groups=len(unresolved))
    old_full=np.full(valid.shape,-1,np.int64);old_full[valid]=old_ids
    new_full=np.full(valid.shape,-1,np.int64);new_full[valid]=new_ids
    nonanchor=~flat_anchor
    _,nonanchor_old,_=exact_groups(x[valid][nonanchor])
    _,nonanchor_new,_=exact_groups(np.concatenate([x[valid],extra[valid]],axis=1)[nonanchor])
    result=dict(zero_bytes_rule=ZERO_BYTES_RULE,rounding=False,valid_candidate_rows=int(valid.sum()),
        forced_anchor_rows=int(anchors.sum()),nonanchor_candidate_rows=int(nonanchor.sum()),
        invalid_candidate_rows=int((~valid).sum()),all_invalid_frame_rows=int((~present).sum()),
        old_all_valid=group_summary(old_groups,y,flat_anchor),extended_all_valid=group_summary(new_groups,y,flat_anchor),
        old_nonanchor_only=group_summary(nonanchor_old,y[nonanchor],np.zeros(nonanchor.sum(),bool)),
        extended_nonanchor_only=group_summary(nonanchor_new,y[nonanchor],np.zeros(nonanchor.sum(),bool)),
        old_repeated_groups_split=sum(g['split'] for g in repeated),conflict_resolution=by_axis,
        repeated_old_groups=repeated,old_group_assignment_sha=array_sha(old_full),extended_group_assignment_sha=array_sha(new_full),
        claim_limit='Exact opposite-sign collisions constrain deterministic predictions on these identical inputs; absence does not prove sufficient information, generalization, or runtime gate feasibility.')
    return result,old_full,new_full


def singular_info(matrix,with_vectors=False):
    matrix=np.asarray(matrix,np.float64)
    assert matrix.ndim==2 and min(matrix.shape)>0 and np.isfinite(matrix).all()
    if with_vectors:u,s,_=np.linalg.svd(matrix,full_matrices=False)
    else:s=np.linalg.svd(matrix,full_matrices=False,compute_uv=False);u=None
    tolerance=float(max(matrix.shape)*np.finfo(np.float64).eps*s[0])
    rank=int(np.count_nonzero(s>tolerance))
    return dict(shape=list(matrix.shape),rank=rank,tolerance=tolerance,singular_values=s.tolist()),u


def input_span_audit(X,E):
    """Unweighted input-column-space calculation; labels cannot enter API."""
    X,E=np.asarray(X,np.float64),np.asarray(E,np.float64)
    assert X.ndim==E.ndim==2 and len(X)==len(E)>0 and np.isfinite(X).all() and np.isfinite(E).all()
    old,U=singular_info(X,True);rank=old['rank']
    residual=E-U[:,:rank]@(U[:,:rank].T@E)
    norm=float(np.linalg.norm(E));residual_norm=float(np.linalg.norm(residual))
    ratios=[];column_norm=np.linalg.norm(E,axis=0);residual_column_norm=np.linalg.norm(residual,axis=0)
    for a,b in zip(residual_column_norm,column_norm):ratios.append(float(a/b) if b>0 else 0.)
    new,_=singular_info(np.concatenate([X,E],axis=1));extra,_=singular_info(E)
    ratio=residual_norm/norm if norm>0 else 0.
    assert np.isfinite(ratio) and np.isfinite(ratios).all()
    return dict(rows=len(X),old=old,extended=new,extra=extra,rank_gain=new['rank']-rank,
        projection_definition='E_minus_U_rankX_times_U_rankX_transpose_times_E',labels_used=False,
        matrix_centering=False,intercept_added=False,frame_or_target_weights=False,
        extra_frobenius_norm=norm,residual_frobenius_norm=residual_norm,frobenius_residual_ratio=ratio,
        extra_column_norm=column_norm.tolist(),residual_column_norm=residual_column_norm.tolist(),
        per_column_residual_ratio=ratios,zero_extra_column_rule='ratio0 exactly when denominator0',
        extra_information_below_fixed_ratio=ratio<=NOVELTY_FROBENIUS_MAX,
        threshold=NOVELTY_FROBENIUS_MAX,threshold_scope='Input redundancy only; not a performance/selection gate.')


def extra_inputs(raw,valid,anchor_index,mean,std):
    raw,valid=np.asarray(raw),np.asarray(valid)
    mean,std=np.asarray(mean,np.float32),np.asarray(std,np.float32)
    assert raw.dtype==np.float32 and raw.shape==(*valid.shape,18) and valid.dtype==bool
    assert mean.shape==std.shape==(18,) and (std>=np.float32(1e-6)).all()
    z=np.zeros(raw.shape,np.float64);z[valid]=((raw[valid]-mean)/std).astype(np.float64)
    result=np.zeros_like(z);i,j=np.nonzero(valid)
    assert np.isfinite(z).all()
    assert valid[i,anchor_index[i]].all()
    result[i,j]=z[i,j]-z[i,anchor_index[i]]
    assert not result[~valid].any()
    active=np.flatnonzero(valid.any(1));assert not result[active,anchor_index[active]].any()
    return result


def load_inputs():
    install_guard()
    protocol=C.protocol('REPRESENTATION_PROTOCOL');assert protocol['representation']==config()
    expected={'source_contract','source_features','source_train_labels','source_feature_lock',
        'parent_train_protocol','sign_prefit','rbf_basis','direction_features','direction_receipt'}
    assert set(protocol['inputs'])==expected
    bindings=protocol['inputs'];parent=bound_read(bindings['parent_train_protocol'])
    assert bindings['parent_train_protocol']==C.bind(C.PARENT_DOC/'TRAIN_PROTOCOL.json')
    for new,old in [('source_contract','source_contract'),('source_features','features'),
                    ('source_train_labels','train_labels'),('source_feature_lock','feature_lock')]:
        assert bindings[new]==parent['inputs'][old],(new,old)
    prefit=bound_read(bindings['sign_prefit']);contract=bound_read(bindings['source_contract'])
    assert prefit['complete'] and prefit['PASS'] and contract['complete'] and contract['status']=='PASS'
    assert bindings['sign_prefit']==C.bind(C.SIGN_DOC/'PREFIT_REVIEW.json')
    for key in ('source_features','source_train_labels'):
        assert prefit['inputs']['features' if key=='source_features' else 'train_labels']==bindings[key]
    lock=bound_read(bindings['source_feature_lock']);assert lock['features']==bindings['source_features']
    assert lock['complete'] and not lock['source_targets_read'] and not lock['real_targets_read']
    basis=bound_read(bindings['rbf_basis']);assert basis['complete'] and basis['PASS']
    assert prefit['basis_SHA_bind']==bindings['rbf_basis']
    receipt=bound_read(bindings['direction_receipt']);assert receipt['complete'] and receipt['PASS']
    assert receipt['directions']==bindings['direction_features']
    assert receipt['protocol']==protocol['feature_protocol']==C.bind(C.DOC/'AUDIT_PROTOCOL.json')
    C.verify(C.read(C.DOC/'AUDIT_PROTOCOL_SHA.json'))
    assert receipt['source_TRAIN_only'] and not receipt['source_labels_read'] and not receipt['VAL_quality_read'] and not receipt['real_targets_read']
    with np.load(C.ROOT/bindings['source_features']['path'],allow_pickle=False) as z:
        all_ids,split=z['ids'],z['split'];eligible=set(contract['fit_eligibility']['eligible_ids']['TRAIN'])
        idx=np.asarray([i for i,fid in enumerate(all_ids) if fid in eligible],np.int64)
        ids=all_ids[idx];index=z['R0_GEO_index'][idx]
        assert len(ids)==len(set(ids.tolist()))==2598 and (split[idx]=='TRAIN').all()
        assert tuple(z['hypothesis_names'].tolist())==HYP
        feature={m:z[m+'_geo'][idx] for m in EXPERTS};valid={m:z[m+'_valid'][idx] for m in EXPERTS}
    with np.load(C.ROOT/bindings['source_train_labels']['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['source_index'],idx)
        np.testing.assert_array_equal(np.flatnonzero(z['eligible_train_mask']),idx)
        assert tuple(z['hypothesis_names'].tolist())==HYP
        errors={m:np.stack([z[m+'_T_cm'],z[m+'_R_deg']],axis=-1) for m in EXPERTS}
        anchor_errors=np.stack([z['R0_GEO_T_cm'],z['R0_GEO_R_deg']],axis=-1)
        scale=np.asarray([z['sT_cm'].item(),z['sR_deg'].item()],np.float64)
    with np.load(C.ROOT/bindings['direction_features']['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['source_index'],idx)
        np.testing.assert_array_equal(z['anchor_index'],index)
        direction={m:z[m+'_direction18'] for m in EXPERTS}
        for m in EXPERTS:np.testing.assert_array_equal(z[m+'_valid'],valid[m])
    for m in EXPERTS:
        assert feature[m].dtype==np.float32 and feature[m].shape==(2598,2,94)
        assert direction[m].dtype==np.float32 and direction[m].shape==(2598,2,18)
        assert valid[m].dtype==bool and np.isfinite(direction[m]).all() and not direction[m][~valid[m]].any()
    values=feature['R0'][valid['R0']];assert len(values)==5194
    mean=values.mean(0);std=np.maximum(values.std(0),np.float32(1e-6))
    values18=direction['R0'][valid['R0']]
    mean18=values18.mean(0);std18=np.maximum(values18.std(0),np.float32(1e-6))
    norm=receipt['normalization']
    np.testing.assert_array_equal(np.asarray(norm['mean18'],np.float32),mean18)
    np.testing.assert_array_equal(np.asarray(norm['std18'],np.float32),std18)
    assert norm['source']=='R0_eligible_train_valid_candidates' and norm['valid_candidate_count']==5194 and norm['std_floor']==1e-6
    assert norm['array_sha']==array_sha(np.stack([mean18,std18]))
    assert norm['mean_sha']==array_sha(mean18) and norm['std_sha']==array_sha(std18)
    assert basis['normalization_sha']==array_sha(np.stack([mean,std]))==prefit['inputs']['normalization_sha']
    for name,value in [('source_ids_sha',ids),('source_index_sha',idx),('anchor_index_sha',index),
                       ('anchor_errors_sha',anchor_errors),('scale_sha',scale)]:
        assert prefit['inputs'][name]==array_sha(value),name
    assert index.dtype==np.int64 and np.count_nonzero(index<0)==1
    T.B.validate_basis(basis['basis'],mean,std)
    return dict(protocol=protocol,bindings=bindings,prefit=prefit,ids=ids,source_index=idx,index=index,
        feature=feature,valid=valid,errors=errors,anchor_errors=anchor_errors,scale=scale,
        mean=mean,std=std,mean18=mean18,std18=std18,direction=direction,basis=basis['basis'])


def prepared(model,data):
    experts=['R0']+([] if model=='R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])
    raw=np.concatenate([data['feature'][m] for m in experts],axis=1)
    valid=np.concatenate([data['valid'][m] for m in experts],axis=1)
    errors=np.concatenate([data['errors'][m] for m in experts],axis=1)
    raw18=np.concatenate([data['direction'][m] for m in experts],axis=1)
    phi=T.rbf_inputs(raw,valid,data['index'],data['mean'],data['std'],data['basis'])
    x=T.difference_from_context(phi,valid,data['index'])
    scaled,target=T.signed_targets(errors,valid,data['anchor_errors'],data['index'],data['scale'])
    hashes=dict(signed_target_sha=array_sha(target),input_difference_sha=array_sha(x),base_context_sha=array_sha(phi),
        errors_sha=array_sha(errors),scaled_excess_sha=array_sha(scaled),original_valid_sha=array_sha(valid))
    for key in T.HASH_KEYS:assert hashes[key]==data['prefit']['models'][model][key],(model,key)
    extra=extra_inputs(raw18,valid,data['index'],data['mean18'],data['std18'])
    names=[f'{m}:{h}' for m in experts for h in HYP]
    return dict(x=x,extra=extra,target=target,valid=valid,names=names,hashes=hashes)


def report_text(out):
    lines=['# TRAIN 입력 표현·잔차 방향 감사','',
        '**이 결과는 학습 성능이 아닌 입력 충돌·중복 진단이다.** 기존 TRAIN2598행·실패1행과253입력/타깃6해시를 유지했다. 모델 weight·objective·새argmin·VAL/실사 오류는 읽거나 평가하지 않았다.','',
        '코너별 signed residual18은 기존 R0/PoseFix 예측과 고정 pose/K의 투영에서 얻었다. R0의 유효TRAIN5194후보만으로 float32 mean/std를 만들고 같은 연산으로 정규화한 뒤 float64에서 후보−anchor 차이를 계산했다. 이미지 forward·PnP·새 참조 오류 계산은 없다.','',
        '## 입력 열공간 중복','',
        'SVD에는 유효 nonanchor 입력만 전달했다. 타깃은 사용하지 않았으며 절편·추가 중심화·타깃 회귀를 하지 않았다. rank tolerance는 각 행렬에 `max(shape)*eps64*smax`로 고정했다.','',
        '| 모델 | nonanchor 행 | 기존 rank | 확장 rank | 잔차 Frobenius 비율 | 비율≤1e−4 |','|---|---:|---:|---:|---:|---|']
    for model,r in out['models'].items():
        s=r['input_span'];lines.append(f"| {model} | {s['rows']} | {s['old']['rank']} | {s['extended']['rank']} | {s['frobenius_residual_ratio']:.9g} | {s['extra_information_below_fixed_ratio']} |")
    lines+=['',out['novelty']['interpretation'],'',
        '1e−4는 새 입력이 기존 열공간에서 벗어나는 정도의 사전 기준이며 pose 선택·T/R 성능 기준이 아니다. 수치 rank는 선택된 정밀도/허용오차의 판정이다. 비중복은 정답 정보·개선·전이 가능성을 증명하지 않는다.','',
        '## 반올림 없는 입력 충돌','',
        '+0/−0만 같은 +0으로 통일하고 float64 실제 byte로 묶었다. 그 외 반올림·근접 임계값은 없다. sign-conflict는−/0/+ 중 두 부호 이상, strict-opposite는 min<0<max다. anchor의 입력/타깃0은 강제 구성이라 별도로 보존했다.','',
        '| 모델 | 기존 반복 그룹 | anchor/nonanchor 혼합 그룹 | T strict 충돌 기존→확장 | R strict 충돌 기존→확장 |','|---|---:|---:|---:|---:|']
    for model,r in out['models'].items():
        c=r['collisions'];old,new=c['old_all_valid'],c['extended_all_valid']
        lines.append(f"| {model} | {old['repeated_groups']} | {old['anchor_nonanchor_groups']} | {old['axes']['T']['strict_opposite']['groups']} → {new['axes']['T']['strict_opposite']['groups']} | {old['axes']['R']['strict_opposite']['groups']} → {new['axes']['R']['strict_opposite']['groups']} |")
    lines+=['','동일 입력의 상반된 타깃은 그 행들을 완벽하게 예측할 수 없다는 제한이다. 충돌이 없다는 사실은 표현 충분성이나 공동 T/R 목표 달성 가능성의 증명이 아니다. 추가18개가 그룹을 나누더라도 새 모델을 학습하거나 선택기를 평가한 결과가 아니다.','',
        'NPZ는 IDs·source index·anchor·정규화·새18개·원타깃·valid·기존/확장 그룹 번호를 보존한다. 전체 singular spectrum과 열별 잔차 비율, 반복 그룹의 타깃 범위·분할/미해결 여부는 JSON에 있다.','',
        'source feature NPZ 컨테이너는 VAL 입력도 포함하지만 적격TRAIN행만 선택했다. 라벨은 이미 고정된 TRAIN 전용 NPZ만 사용했다. 이는 TRAIN 내부의 진단이며 일반화/신뢰도 보정·새 gate·실험 성공을 주장하지 않는다.','',
        '[감사 JSON](REPRESENTATION_AUDIT.json) · [방향 특징 감사](FEATURE_AUDIT.json) · [입력 진단 프로토콜](REPRESENTATION_PROTOCOL.json) · [이전 입력·타깃 해시](../pallet_pose_signed_axes_sign_20261001_v1/PREFIT_REVIEW.json)','']
    return '\n'.join(lines)


def run():
    started=time.monotonic();data=load_inputs()
    if (C.DOC/'REPRESENTATION_AUDIT.json').exists():raise AssertionError('ALREADY_FROZEN_NO_REPEAT')
    models={};arrays=dict(ids=data['ids'],source_index=data['source_index'],anchor_index=data['index'],mean18=data['mean18'],std18=data['std18'])
    with threadpool_limits(limits=1):
        for model in C.MODEL_NAMES:
            p=prepared(model,data);valid=p['valid'];nonanchor=valid.copy()
            rows=np.flatnonzero(valid.any(1));nonanchor[rows,data['index'][rows]]=False
            collisions,oldgroup,newgroup=collision_audit(p['x'],p['extra'],p['target'],valid,data['index'],data['ids'],p['names'])
            span=input_span_audit(p['x'][nonanchor],p['extra'][nonanchor])
            models[model]=dict(names=p['names'],prior_six_hashes=p['hashes'],
                extra18_sha=array_sha(p['extra']),extended271_sha=array_sha(np.concatenate([p['x'],p['extra']],axis=2)),
                collisions=collisions,input_span=span,forced_anchor_zero=True,all_invalid_rows_retained=1)
            for key,value in [('extra18',p['extra']),('axis_target',p['target']),('valid',valid),
                              ('old_group_ids',oldgroup),('extended_group_ids',newgroup)]:arrays[model+'_'+key]=value
            print('REPRESENTATION_INPUT_AUDITED',model,flush=True)
    insufficient=all(r['input_span']['frobenius_residual_ratio']<=NOVELTY_FROBENIUS_MAX for r in models.values())
    artifact=C.RAW/'REPRESENTATION_ARRAYS.npz';artifact.parent.mkdir(parents=True,exist_ok=True)
    with artifact.open('xb') as f:np.savez_compressed(f,**arrays)
    out=dict(complete=True,PASS=True,diagnostic_only=True,created_at=C.now(),code=C.bind(__file__),
        protocol=C.bind(C.DOC/'REPRESENTATION_PROTOCOL.json'),inputs=data['bindings'],representation=config(),
        source_TRAIN_only=True,frames=2598,available_anchor_rows=2597,failed_rows_retained=1,
        source_ids_sha=array_sha(data['ids']),source_index_sha=array_sha(data['source_index']),
        anchor_index_sha=array_sha(data['index']),normalization94_sha=array_sha(np.stack([data['mean'],data['std']])),
        normalization18=dict(mean=data['mean18'],std=data['std18'],sha=array_sha(np.stack([data['mean18'],data['std18']]))),
        models=models,arrays=C.bind(artifact),novelty=dict(all_models_ratio_le1e_4=insufficient,
            status='INSUFFICIENT_ADDITIONAL_INPUT_EVIDENCE' if insufficient else 'INPUT_NONREDUNDANCY_ONLY_NOT_PERFORMANCE_EVIDENCE',
            interpretation='모든 모델의 새 입력 잔차 비율이1e−4 이하라 추가 특징의 근거가 부족하다.' if insufficient else '일부 모델에서 입력 비중복이 관측되지만 학습·성능·전이의 증거는 아니며 fit을 승인하지 않는다.'),
        float64_signed_zero_canonicalization=True,approximate_collision_rounding=False,
        source_feature_container_includes_VAL_inputs=True,VAL_quality_read=False,real_targets_read=False,
        raw_source_reference_reads=0,new_reference_metric_calculations=0,prior_weights_read=False,
        optimizer_steps=0,new_fits=0,actual_data_weight_probes=0,actual_data_objective_probes=0,new_policy_argmin=0,
        image_forwards=0,new_PnP_calls=0,method_success=False,goal_complete=False,read_paths=sorted(READS),
        wall_seconds=time.monotonic()-started)
    C.save(C.DOC/'REPRESENTATION_AUDIT_KO.md',report_text(out))
    C.save(C.DOC/'REPRESENTATION_AUDIT.json',out)
    print('REPRESENTATION_AUDIT_PASS',out['novelty']['status'],flush=True)


def selfcheck():
    # Duplicate signs and conflicting labels are invented, including a forced
    # anchor/nonanchor collision and an all-invalid retained row.
    valid=np.array([[1,1,1],[1,1,0],[0,0,0]],bool);index=np.array([0,0,-1],np.int64)
    x=np.zeros((3,3,2),np.float64);x[0,1]=[1.,-0.];x[0,2]=[1.,0.];x[1,1]=[-0.,0.]
    extra=np.zeros((3,3,1),np.float64);extra[0,1]=1.;extra[0,2]=-1.;extra[1,1]=2.
    y=np.zeros((3,3,2),np.float64);y[0,1]=[1.,-1.];y[0,2]=[-1.,1.];y[1,1]=[0.,1.]
    result,old,new=collision_audit(x,extra,y,valid,index,np.array(['a','b','c']),['A','B','C'])
    assert result['old_all_valid']['groups']==2 and result['extended_all_valid']['groups']==4
    assert result['old_all_valid']['anchor_nonanchor_groups']==1 and result['all_invalid_frame_rows']==1
    assert result['old_all_valid']['axes']['T']['strict_opposite']['groups']==1
    assert result['extended_all_valid']['axes']['T']['strict_opposite']['groups']==0
    assert result['conflict_resolution']['T']['strict_opposite']['fully_resolved_groups']==1
    assert (old[~valid]==-1).all() and (new[~valid]==-1).all()
    # Distinguish opposite signs from zero-versus-positive sign conflicts.
    d=target_description(np.array([[0.,-1.],[1.,-2.]],np.float64))
    assert d['T']['sign_conflict'] and not d['T']['strict_opposite'] and not d['R']['sign_conflict']
    ids,groups,_=exact_groups(np.array([[0.,-0.],[0.,0.],[np.nextafter(0.,1.),0.]],np.float64))
    assert len(groups)==2 and ids[0]==ids[1] and ids[2]!=ids[1]
    # Rank-deficient, zero and novel columns are known analytically.
    X=np.array([[1.,2.],[0.,0.],[0.,0.],[0.,0.]])
    E=np.array([[3.,0.,0.],[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]])
    s=input_span_audit(X,E)
    assert s['old']['rank']==1 and s['extended']['rank']==2 and s['rank_gain']==1
    np.testing.assert_allclose(s['frobenius_residual_ratio'],1/np.sqrt(10.),rtol=0,atol=1e-15)
    np.testing.assert_allclose(s['per_column_residual_ratio'],[0.,1.,0.],rtol=0,atol=1e-15)
    zero=input_span_audit(np.zeros((4,2)),np.zeros((4,3)))
    assert zero['old']['rank']==zero['extended']['rank']==0 and zero['frobenius_residual_ratio']==0.
    duplicate=input_span_audit(X,np.column_stack([X[:,0],X[:,0]*-3]))
    assert duplicate['extra_information_below_fixed_ratio']
    # Hypothetical zero extension only. Split-dot avoids changing original
    # reduction width and exactly embeds the already specified 253 scorer.
    rng=np.random.default_rng(20261001271)
    old_input=rng.normal(size=(5,4,253));new_input=rng.normal(size=(5,4,18));weight=rng.normal(size=(253,2))
    direct=np.einsum('nkd,da->nka',old_input,weight)
    embedded=np.einsum('nkd,da->nka',old_input,weight)+np.einsum('nkd,da->nka',new_input,np.zeros((18,2)))
    np.testing.assert_array_equal(direct,embedded)
    v=np.array([[1,1],[0,0]],bool);idx=np.array([1,-1]);raw=rng.normal(size=(2,2,18)).astype(np.float32);raw[~v]=0.
    e=extra_inputs(raw,v,idx,np.zeros(18,np.float32),np.ones(18,np.float32))
    assert not e[0,1].any() and not e[1].any()
    print(json.dumps(dict(PASS=True,invented_arrays_only=True,actual_data_reads=0,model_weight_reads=0,
        collision_sign_zero_split_checks=True,SVD_rank_and_zero_checks=True,split_dot_zero_embedding_bitexact=True,
        anchor_and_invalid_preserved=True,new_fits=0,new_policy_argmin=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('selfcheck','run'));args=parser.parse_args()
    if args.action=='selfcheck':selfcheck()
    else:run()
