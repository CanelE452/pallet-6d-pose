"""A: immutable official scores; pre-existing QA and cached PnP sensitivity only."""
import csv
import math
from collections import Counter
import numpy as np
from common import *

CAT='eval_pallet09:1778653804674198784'
NAMES=['R0',*[f'{m}_seed{s}' for m in ('random','diversity','geometry_weighted_diversity','proposed') for s in (1,2,3)],'full174_seed1']

def cvar(a):
    a=np.asarray(a,float);assert len(a) and np.isfinite(a).all()
    return float(np.mean(np.sort(a)[-math.ceil(.1*len(a)):]))

def qa():
    split=read(TASK_DOC/'SPLIT_BINDING.json')['evaluation']
    files=[ROOT/'challenge/real_gt_v2/audit/LEGACY_GT_PER_FRAME.csv',
        ROOT/'challenge/real_gt_v2/audit/CODE_CONSUMER_AUDIT.md',
        ROOT/'_docs/archive/paper_support_20260830/real_gt_v2/GT_QA_STATUS.md',
        ROOT/'_docs/audits/accuracy_root_cause_v1/GT_TRUST_AUDIT.md']
    review_names=['1778652126319943168','1778652144496057088','1778652156557165568',
        '1778652170735118080','1778652172717607680','1778653804674198784']
    historical=read(TASK_DOC/'SOURCE_BINDING.json')['paths'];rows=[]
    before_task='d653dce26c43db3fd60c387ed1936bea5751aa24'
    for r in split:
        path=ROOT/r['label_path'];assert sha(path)==historical[r['label_path']]
        blob=subprocess.check_output(['git','rev-parse',before_task+':'+r['label_path']],text=True).strip()
        current=subprocess.check_output(['git','hash-object',str(path)],text=True).strip();assert current==blob
        content=read(path);obj=content['objects'][0]
        reasons=list(obj.get('manual_review_reasons') or [])
        if obj.get('migration_status')=='MANUAL_REVIEW_REQUIRED':reasons.append('MIGRATION_STATUS_MANUAL_REVIEW_REQUIRED')
        if Path(r['image_path']).stem in review_names:reasons.append('ARCHIVED_QA_REVIEW_ONLY_LIST_20260827')
        clean=obj.get('migration_status') in ('VERIFIED','CLEAN','CONFIRMED','MANUALLY_VERIFIED') and not reasons
        category='QA_PREEXISTING_FLAGGED' if reasons else ('QA_CLEAN' if clean else 'QA_UNKNOWN')
        manual=obj.get('manual_kps');projected=obj.get('projected_cuboid')
        rows.append(dict(frame_id=r['frame_id'],category=category,preexisting_flagged=bool(reasons),
            reasons=reasons,source_document=r['label_path'],source_sha256=sha(path),pre_task_git_commit=before_task,
            pre_task_git_blob=blob,flag_existed_before_task_risk=True,
            source_last_commit=subprocess.check_output(['git','log','-1','--format=%H %aI',before_task,'--',r['label_path']],text=True).strip(),
            canonical_pose_status='NULL' if obj.get('canonical_pose') is None else 'PRESENT',
            manual_kps_complete=bool(manual is not None and len(manual)==9 and all(p is not None for p in manual)),
            manual_kps_null_indices=[i for i,p in enumerate(manual or []) if p is None],
            projected_cuboid_complete=bool(projected is not None and len(projected)==8 and all(p is not None for p in projected)),
            visibility_provenance=[{k:p.get(k) for k in ('visibility','source','reason','visibility_source')} for p in obj.get('keypoint_annotations',[])]))
    write(A/'PREEXISTING_QA_FLAGS.json',dict(rule=read(DOC/'MASTER_PROTOCOL_LOCK.json')['A_QA_rule'],
        frozen_before_A_score_read=True,rows=rows,counts=dict(Counter(r['category'] for r in rows)),
        archival_review_only_frame_stems=review_names,not_outcome_selected=True,
        sources={str(p.relative_to(ROOT)):dict(sha256=sha(p),last_commit=subprocess.check_output(['git','log','-1','--format=%H %aI','--',str(p.relative_to(ROOT))],text=True).strip()) for p in files}))
    bindings={str(p.relative_to(ROOT)):sha(p) for p in [*files,TASK/'PER_FRAME_POSE.json',TASK_DOC/'PER_SEED_RESULTS.json',TASK_DOC/'VERDICT.json',TASK_DOC/'CVAR_RESULTS.json',TASK_DOC/'SPLIT_BINDING.json']}
    write(A/'TASK_RISK_SOURCE_BINDING.json',dict(paths=bindings,original_verdict='TASK_RISK_AL_NO_SIGNAL',new_training=0))

def scores():
    flags=read(A/'PREEXISTING_QA_FLAGS.json');qa_by={r['frame_id']:r for r in flags['rows']}
    poses=read(TASK/'PER_FRAME_POSE.json');official=read(TASK_DOC/'PER_SEED_RESULTS.json')['per_run']
    split=read(TASK_DOC/'SPLIT_BINDING.json')['evaluation'];ids=[r['frame_id'] for r in split]
    clean={r['frame_id'] for r in flags['rows'] if r['category']=='QA_CLEAN'}
    influence={};sessions={};sens={}
    for name in NAMES:
        rows=poses[name];assert [r['frame_id'] for r in rows]==ids
        per={};ses={}
        for key,official_key in [('translation_cm','translation_CVaR90_cm'),('yaw_deg','yaw_CVaR90_deg')]:
            values=np.array([r[key] for r in rows]);base=cvar(values);assert abs(base-official[name][official_key])<1e-10
            tail=set(np.argsort(-values,kind='stable')[:15]);den=sum(values[i] for i in tail)
            per[key]=dict(official=base,worst15_flagged_count=sum(qa_by[ids[i]]['preexisting_flagged'] for i in tail),
                frames=[dict(frame_id=ids[i],error=float(v),in_worst15=i in tail,
                    worst15_fraction=float(v/den) if i in tail else 0.,
                    leave_one_out_cvar90=cvar(np.delete(values,i)),
                    leave_one_out_minus_official=cvar(np.delete(values,i))-base) for i,v in enumerate(values)])
            ses[key]=[]
            for session in sorted({r['session'] for r in rows}):
                selected=np.array([r['session']!=session for r in rows]);value=cvar(values[selected])
                ses[key].append(dict(left_out_session=session,remaining=int(selected.sum()),cvar90=value,delta=value-base))
        influence[name]=per;sessions[name]=ses
        kp=read((TASK if name.startswith(('proposed','full174')) else AL)/'evaluation'/name/'PER_FRAME.json')
        subsets={}
        for label,membership in [('S0_OFFICIAL',set(ids)),('S1_PREEXISTING_QA_CLEAN_ONLY',clean)]:
            rr=[r for r in rows if r['frame_id'] in membership]
            subsets[label]=dict(n=len(rr),status='ESTIMATED' if rr else 'NOT_ESTIMABLE_EMPTY_PREEXISTING_QA_CLEAN_SET',
                pose_coverage=sum(r['pose_valid'] for r in rr)/len(rr) if rr else None,
                translation_median_cm=float(np.median([r['translation_cm'] for r in rr])) if rr else None,
                translation_CVaR90_cm=cvar([r['translation_cm'] for r in rr]) if rr else None,
                yaw_median_deg=float(np.median([r['yaw_deg'] for r in rr])) if rr else None,
                yaw_CVaR90_deg=cvar([r['yaw_deg'] for r in rr]) if rr else None,
                kp_frame_mean_CVaR90_px=cvar([np.mean(kp[r['frame_id']]['errors_px']) for r in rr]) if rr and all(kp[r['frame_id']]['errors_px'] for r in rr) else None)
        sens[name]=subsets
    write(A/'CVAR_INFLUENCE_PER_FRAME.json',dict(diagnostic_only=True,fixed_official_N=145,worst_count=15,runs=influence))
    write(A/'CVAR_LEAVE_ONE_SESSION_OUT.json',dict(diagnostic_only=True,runs=sessions))
    write(A/'QA_SENSITIVITY_RESULTS.json',dict(same_membership_for_all_methods=True,S1_ids=sorted(clean),
        QA_counts=flags['counts'],runs=sens,original_verdict_unchanged=True,
        caution='No confirmed-clean frame exists under the preexisting metadata rule. Do not reinterpret a universal migration flag as evidence that the proposed-specific catastrophe is a GT error.'))
    assert not clean, 'If population differs, apply master predeclared nonempty rule; do not invent a new rule'
    write(A/'TASK_RISK_ROBUSTNESS_VERDICT.json',dict(verdict='TASK_RISK_RESULT_QA_SENSITIVE_REQUIRES_CAUTION',
        official_verdict='TASK_RISK_AL_NO_SIGNAL',official_verdict_unchanged=True,
        QA_clean=0,QA_flagged=145,QA_unknown=0,QA_filtered_performance='NOT_ESTIMABLE',
        interpretation='QA sensitivity cannot be resolved by the requested clean-only filter; all145 share preexisting review-required status. LOO is diagnostic, not a new primary.',
        new_training_updates=0,new_inference=0,new_PASS=False))

def catastrophe():
    sys.path.insert(0,str(ROOT/'scripts/paper/pose_metric_closure_v1'));sys.path.insert(0,str(ROOT))
    import run_pose_evaluation as canonical
    import symmetry_aware_pose_metrics as metric
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
    split=read(TASK_DOC/'SPLIT_BINDING.json')['evaluation'];record=next(r for r in split if r['frame_id']==CAT)
    label=read(ROOT/record['label_path']);obj=label['objects'][0];raw=label['camera_data']['intrinsics']
    K=np.array([[raw['fx'],0,raw['cx']],[0,raw['fy'],raw['cy']],[0,0,1]],float)
    manifest=read(POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list'];meta=next(r for r in manifest if r['image']==record['image_path'])
    truth=read(POSE/'GEOMETRY_RESOLVED_POSE_GT.json')['frames'][meta['frame_id']]
    gtR,gtt=np.array(truth['R_gt_representative']),np.array(truth['t_gt'])
    xy=np.array([p['xy'] for p in obj['keypoint_annotations']],float);manual=obj.get('manual_kps')
    registry=read(ROOT/'challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json')['objects']
    dims=next(r['physical_dimensions_m'] for r in registry if r['object_type']==record['object_type'])
    long,short,height=max(dims['x'],dims['z']),min(dims['x'],dims['z']),dims['y']
    runs={}
    for name in [n for n in NAMES if not n.startswith('random')]:
        path=(TASK if name.startswith(('proposed','full174')) else AL)/'evaluation'/name/'PREDICTIONS.json'
        predictions=read(path)['frames'][record['image_path']];idx=max(range(len(predictions)),key=lambda j:predictions[j]['score'])
        selected=predictions[idx];points=np.array(selected['keypoints_xy'],float)
        hypotheses=select_pnp_hypotheses(points,K,dict(x=long,y=height,z=short),None)
        chosen=next(h for h in hypotheses.hypotheses if h.name==hypotheses.selected_hypothesis)
        axis='CF_WIDTH' if abs(chosen.camera_facing_dimensions.as_dict()['width']-long)<1e-6 else 'CF_DEPTH'
        fits={}
        for k,a,b in [('CF_WIDTH',long,short),('CF_DEPTH',short,long)]:
            fit=canonical.solve(canonical.cuboid(a,height,b),points[:8],K,np.isfinite(points[:8]).all(-1))
            R,t,reproj=fit;parts=metric.translation_components_m(t,gtt)
            fits[k]=dict(success=True,dimensions=dict(across=a,height=height,along=b),reprojection_px=reproj,
                rotation=R.tolist(),translation_m=t.tolist(),translation_cm=parts['total_m']*100,
                lateral_cm=parts['lateral_m']*100,depth_cm=parts['depth_m']*100,yaw_deg=metric.yaw_error_degrees(R,gtR))
        runs[name]=dict(**selected,cache_candidate_index=idx,dense_grid_index=None,
            dense_index_reason='Not saved in historical evaluation prediction cache; not reconstructed by guessing',
            point_order=list(range(9)),canonical_fit_point_order=list(range(8)),selected_axis=axis,
            selected_hypothesis=hypotheses.selected_hypothesis,selector_hypotheses=[h.to_dict() for h in hypotheses.hypotheses],
            fits=fits,reference_keypoint_error_px=np.linalg.norm(points-xy,axis=1).tolist(),
            available_manual_error_px=[float(np.linalg.norm(points[i]-np.array(p))) if p is not None else None for i,p in enumerate(manual)],
            source_prediction_sha256=sha(path),bbox_diagonal_px=float(np.linalg.norm(np.array(selected['box_xyxy'])[2:]-np.array(selected['box_xyxy'])[:2])),
            point_spread_singular_values=np.linalg.svd(points[:8]-points[:8].mean(0),compute_uv=False).tolist())
    counter=[]
    for seed in (1,2,3):
        proposed=f'proposed_seed{seed}'
        for method in ('diversity','geometry_weighted_diversity'):
            control=f'{method}_seed{seed}'
            for p,h in [(proposed,control),(control,proposed)]:
                axis=runs[h]['selected_axis']
                counter.append(dict(points_from=p,hypothesis_from=h,axis=axis,**runs[p]['fits'][axis]))
    write(A/'CAT_FRAME_DECOMPOSITION.json',dict(frame_id=CAT,label_path=record['label_path'],label_sha256=sha(ROOT/record['label_path']),
        fixed_category_dimensions=dims,intrinsics=K.tolist(),reference_pose=truth,
        reference_keypoints=xy.tolist(),manual_keypoints=manual,runs=runs,counterfactuals=counter,
        GT_used_for_hypothesis_selection=False,deterministic_existing_alternatives_only=True,
        explanation='Canonical evaluation re-solves the selected dimensional parity with first8 SQPnP+LM; selector internal fitted poses are separately retained, never conflated with final canonical output.'))
    print('A_COMPLETE; official unchanged; clean0/flagged145; no training',flush=True)

if __name__=='__main__':
    {'qa':qa,'scores':scores,'catastrophe':catastrophe}[sys.argv[1]]()
