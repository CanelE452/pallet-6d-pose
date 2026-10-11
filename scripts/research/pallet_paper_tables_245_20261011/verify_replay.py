"""Read-only replay audit with separately written corner/pose arithmetic.

This invokes no pose solver, image correction, network or training operation.
It checks all old replay methods against their original full-precision records.
"""
from collections import Counter
from pathlib import Path
import math
import numpy as np
from . import common as C

class Check:
    def __init__(self):self.count=0;self.max_delta=0.;self.by=Counter()
    def equal(self,a,b,label,tolerance=1e-7):
        if isinstance(b,dict):
            for k,v in b.items():
                assert k in a,(label,k)
                self.equal(a[k],v,label+'/'+k,tolerance)
        elif isinstance(b,(list,tuple)):
            assert len(a)==len(b),(label,len(a),len(b))
            for i,(x,y) in enumerate(zip(a,b)):self.equal(x,y,label+'/'+str(i),tolerance)
        elif isinstance(b,(int,float)) and not isinstance(b,bool):
            assert a is not None,(label,a,b)
            delta=abs(float(a)-float(b));assert delta<=tolerance,(label,a,b,delta)
            self.max_delta=max(self.max_delta,delta);self.count+=1
        else:assert a==b,(label,a,b)

def corner_numpy(r):
    p=np.asarray(r['qFinal'],float);g=np.asarray(r['evaluation_reference_points'],float)
    valid=np.asarray(r['evaluation_reference_valid'],bool)&np.isfinite(g).all(1)&~(g==-1).all(1)
    permutations=np.asarray(r['evaluation_permutations'],int)
    pv=np.isfinite(p).all(1)&~(p==-1).all(1)&bool(r['corner']['matched'])
    diagon=math.hypot(*r['raw_hw']);branches=[]
    for perm in permutations:
        errors=[]
        for i in range(9):
            if not valid[perm[i]]:errors.append(0.)
            elif not pv[i]:errors.append(diagon)
            else:errors.append(float(np.linalg.norm(p[i]-g[perm[i]])))
        mask=valid[perm]
        branches.append((sum(errors[:8])/int(mask[:8].sum()),errors,mask,perm))
    branch=min(range(len(branches)),key=lambda j:branches[j][0]);mean,errors,mask,perm=branches[branch]
    canon=[None]*9
    for i in range(9):
        if mask[i]:canon[int(perm[i])]=errors[i]
    all_errors=[errors[i] for i in range(8) if mask[i]]
    observation=[errors[i] for i in range(8) if mask[i] and pv[i]]
    visible=np.isfinite(g[:8]).all(1)&~(g[:8]==-1).all(1)
    boxdiag=float(np.linalg.norm(np.ptp(g[:8][visible],axis=0))) if visible.sum()>=2 else None
    return dict(E_sym=mean/diagon,E_fixed=branches[0][0]/diagon,frame_mean_px=mean,branch=branch,
        errors=all_errors,observed_errors=observation,canonical_errors=canon[:8],canonical_valid=valid[:8].tolist(),
        corners=int(valid[:8].sum()),gt_count=int(valid[:8].sum()),pck10_denominator=int(valid[:8].sum()),
        pck10_hits=sum(x<=10 for x in all_errors),pallet_diagonal_px=boxdiag)

def model(xyz):
    w,h,d=np.asarray(xyz,float)/2
    return np.asarray([[-w,-h,-d],[w,-h,-d],[w,h,-d],[-w,h,-d],[-w,-h,d],[w,-h,d],[w,h,d],[-w,h,d]])

def pose_numpy(actual,truth):
    R=np.asarray(actual['R_physical'],float);G=np.asarray(truth['R'],float)
    t=np.asarray(actual['centroid'],float);gt=np.asarray(truth['t'],float);xyz=np.asarray(truth['xyz'],float);x=model(xyz)
    angles=[];yaw=[];adds=[]
    for j in range(truth['order']):
        angle=2*np.pi*j/truth['order'];c,s=np.cos(angle),np.sin(angle)
        Q=np.asarray([[c,0,s],[0,1,0],[-s,0,c]]);T=G@Q;relative=T.T@R
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))))
        yaw.append(abs(float((np.degrees(np.arctan2(relative[0,2],relative[2,2]))+180)%360-180)))
        adds.append(float(np.linalg.norm(x@R.T+t-(x@T.T+gt),axis=1).mean()))
    add=min(adds)
    return dict(translation_cm=float(np.linalg.norm(t-gt)*100),rotation_deg=min(angles),yaw_deg=min(yaw),
                ADDsym_m=add,ADDsym_normalized=add/float(np.linalg.norm(xyz)))

def old_metrics(check,current,previous,label):
    assert current['available']==previous['available'],(label,'available')
    if current['available']:
        check.equal(current,{k:previous[k] for k in ('translation_cm','rotation_deg','yaw_deg','IoU3D','ADDsym_m','ADDsym_normalized')},label)

def run():
    assert C.read(C.DOC/'EVALUATION_EXECUTION.json')['status']=='COMPLETE'
    assert not (C.DOC/'REPLAY_VERIFICATION.json').exists()
    C.verify_inputs();check=Check();bindings=[C.binding(Path(__file__))]
    rectangle=list(C.rows(C.DOC/'RECT_ROWS.jsonl.gz'));square=list(C.rows(C.DOC/'SQUARE_ROWS.jsonl.gz'))
    assert len(rectangle)==21054 and len(square)==4248
    target={fid:r for fid,r in C.read(C.SOURCE/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json').items() if r['split']=='REAL_DEV'}
    square_reference={r['id']:r for r in C.read(C.ROOT/'_docs/experiments/pallet_square6d_manualpnp_20261010/SQUARE_REFERENCE_POSES.json')['frames'] if r['eligible']}
    # Independently compute all 2D branches/PCK/diagonals and five 6D metrics.
    for population,rows in [('RECT',rectangle),('SQUARE',square)]:
        for r in rows:
            assert r.get('inference_reference_inputs',r.get('inference_reference_access',False)) is False
            check.equal(r['corner'],corner_numpy(r),population+'/'+r['id']+'/corner_numpy')
            if r['pose']['available']:
                g=target[r['id']]['truth'] if population=='RECT' else dict(R=square_reference[r['id']]['R'],t=square_reference[r['id']]['t'],xyz=[1.1,.15,1.1],order=4)
                check.equal(r['pose'],pose_numpy(r['actual_pose'],g),population+'/'+r['id']+'/pose_numpy')
            check.by[population+'_independent_corner_rows']+=1
            check.by[population+'_independent_pose_rows']+=bool(r['pose']['available'])
    existing={(r['method'],r['seed'],r['id']):r for r in C.rows(C.JOINT)}
    original=C.read(C.OLD_RAW);mapping={'N0':'N0_BASE_REPLAY','N1':'N1_SYM_ONLY','N2':'N2_DIM_ONLY','P':'OLD_P'}
    original_dcp_path=C.SOURCE/'data/pallet/results/pallet_dim_conditioned_p_v1/PAPER_POSE_METRICS.json'
    original_dcp=C.read(original_dcp_path)['REAL_DEV'];bindings.append(C.binding(original_dcp_path))
    old={}
    for family,arm in mapping.items():
        for seed in C.SEEDS:
            value=original['core'][f'{arm}_seed{seed}']
            scores=value['pose_scores'] if 'pose_scores' in value else original_dcp[f'{arm}_seed{seed}']
            assert len(scores)==319
            old[family,seed]={r['id']:r for r in scores}
    for family in ('D','L','POSEFIX'):
        original_name='PoseFix' if family=='POSEFIX' else family
        for seed in C.SEEDS:
            value=original['comparators'][f'H_{original_name}_seed{seed}']
            old[family,seed]={r['id']:r for r in value['pose_scores']}
    aux={(r['backbone'],r['method'],r['seed'],r['id']):r for r in C.rows(C.ROOT/'_docs/experiments/pallet_wd_hypothesis_diag_20261010/STAGE1_ROWS_AUX.jsonl.gz')}
    for r in rectangle:
        if r['backbone']=='YOLO' and (r['method'],r['seed'],r['id']) in existing:
            p=existing[r['method'],r['seed'],r['id']]
            assert np.array_equal(np.asarray(r['qFinal'],float),np.asarray(p['qFinal'],float),equal_nan=True)
            check.equal(r['corner'],p['corner'],'existing_YOLO/'+r['id']+'/corner')
            old_metrics(check,r['pose'],p['pose'],'existing_YOLO/'+r['id']+'/pose')
            check.by['YOLO_all_six_existing_methods_parity_rows']+=1
        elif r['backbone']=='YOLO' and (r['method'],r['seed']) in old:
            old_metrics(check,r['pose'],old[r['method'],r['seed']][r['id']],'existing_ablation_external/'+r['id'])
            check.by['YOLO_existing_N0_N1_N2_P_D_L_POSEFIX_pose_parity_rows']+=1
        elif r['backbone']!='YOLO' and (r['backbone'],r['method'],r['seed'],r['id']) in aux:
            p=aux[r['backbone'],r['method'],r['seed'],r['id']]
            old_metrics(check,r['pose'],p['S0'],'existing_AUX/'+r['id'])
            check.by['native_Base_N3_F_parity_rows']+=1
    # Match-aware Square PCK/canonical errors are checked against old119 tables, restrictedto118.
    old_square={}
    y=C.ROOT/'_docs/experiments/pallet_n3_completion_v3/SQUARE_YOLO_RESULTS.json';payload=C.read(y)
    b=payload['metrics'];path=Path(b['path']);path=path if path.is_absolute() else C.SOURCE/path
    assert C.sha(path)==b['sha256'];bindings.append(C.binding(path))
    raw=C.read(path)['modes']['manual_in_frame']['rows']
    for seed in C.SEEDS:
        old_square['YOLO','BASE',seed]={r['id']:r for r in raw['R0']}
        old_square['YOLO','N3_DIM_SYM',seed]={r['id']:r for r in raw[f'N3_DIM_SYM_seed{seed}']}
    for name,backbone in [('DOPE','DOPE'),('RESNET18','ResNet-18')]:
        path=C.ROOT/f'_docs/experiments/pallet_n3_completion_v3/SQUARE_{name}_RESULTS.json';bindings.append(C.binding(path))
        methods=C.read(path)['modes']['manual_in_frame']['methods']
        for seed in C.SEEDS:
            for method,key in [('BASE','base'),('N3_DIM_SYM',f'n3_seed{seed}')]:
                old_square[backbone,method,seed]={r['id']:r for r in methods[key]['result']['corner_rows']}
    for r in square:
        if (r['backbone'],r['method'],r['seed']) in old_square:
            previous=old_square[r['backbone'],r['method'],r['seed']][r['id']]
            fields=('E_sym','E_fixed','frame_mean_px','branch','errors','observed_errors','canonical_errors','canonical_valid','matched','detected','corners')
            check.equal(r['corner'],{k:previous[k] for k in fields},'existing_SQUARE/'+r['id'])
            check.by['Square_existing_Base_N3_2D_parity_rows']+=1
    for b in bindings:C.verify_binding(b)
    assert check.by['YOLO_all_six_existing_methods_parity_rows']==5742
    assert check.by['YOLO_existing_N0_N1_N2_P_D_L_POSEFIX_pose_parity_rows']==6699
    assert check.by['native_Base_N3_F_parity_rows']==3828
    assert check.by['Square_existing_Base_N3_2D_parity_rows']==2124
    C.write('REPLAY_VERIFICATION.json',dict(status='PASS',source_sha256=C.sha(Path(__file__)),bindings=bindings,
        numeric_comparisons=check.count,max_absolute_numeric_delta=check.max_delta,tolerance=1e-7,counts=dict(check.by),
        rectangle_rows_sha256=C.sha(C.DOC/'RECT_ROWS.jsonl.gz'),square_rows_sha256=C.sha(C.DOC/'SQUARE_ROWS.jsonl.gz'),
        all_fixed_inputs_unchanged=True,independent_arithmetic=['whole-object2Dbranch','canonical-cornererrors','PCK10<=10fullmanualdenom','ALL8referencebboxdiagonal','centroidT','proper-groupminimumR/yaw/ADD'],
        IoU3D='Historical exact-value replay verification for existing paths; geometry is not independently implemented.',
        additional_F_calls=0,additional_PnP_calls=0,additional_image_corrections=0,additional_model_forwards=0))
    print('INDEPENDENT_REPLAY_VERIFICATION_PASS',check.count,check.max_delta,dict(check.by),flush=True)

if __name__=='__main__':run()
