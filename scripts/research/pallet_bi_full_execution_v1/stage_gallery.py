"""Postseal PRIMARY-only derived galleries for explicit completed stage arms.

Consumes saved observations, actual poses, metrics and coverage. No producer,
optimizer, candidate cost evaluator, metric scorer or trainer is called. Public
case JSON uses a whitelist and contains no physical pose matrices/translations
or reference values. Optional private references are used only for dashed
display projections after all prediction/evaluation/coverage bindings pass.
"""
from __future__ import annotations
import argparse, gzip, html, math, time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import s1_union as U

SEEDS=(20261018,20261019,20261020)
PRIMARY=('TEST','REMAINDER')


def display_sources():
    folder=Path(__file__).resolve().parent
    paths=[Path(__file__),folder/'s1_analysis.py',Path(U.__file__),folder.parent/'pallet_boundary_action_repair_v1/display_privacy.py']
    return {path.name:U.sha(path)for path in paths}


def display_roi(frame,result):
    """Prediction-only native-coordinate crop with outside-support disclosure.

    No scored error, selected winner, reference projection or GT enters this
    rule. A full-context derived image is generated even for a valid crop.
    """
    height,width=frame['raw_hw'];full=[0.,0.,float(width),float(height)]
    source=dict(rule='N0 native-cuboid projection plus first8 initial-query envelope;15percent padding,min12px;clipped native canvas',
        GT_or_score_or_selected_pose_used=False,display_only=True,inference_coordinates_changed=False,
        context_always_retained=True)
    try:
        family=result['families']['ORIGINAL_WD'];xyz=np.asarray(family['native_xyz'],float)
        R=np.asarray(frame['initial_pose']['R_physical'],float)@np.asarray(family['physical_registration_Q'],float).T
        camera=(xyz-xyz.mean(0))@R.T+np.asarray(frame['initial_pose']['centroid'],float)
        pixel=camera@np.asarray(frame['K'],float).T
        if not np.isfinite(pixel).all()or np.min(camera[:,2])<=1e-5:
            return dict(source,mode='FULL_CONTEXT',xyxy_raw=full,reason='INVALID_OR_BEHIND_CAMERA_N0_PROJECTION')
        projected=pixel[:,:2]/pixel[:,2:]
        queries=np.asarray(frame['initial_points'],float)[:8,:2]
        queries=queries[np.isfinite(queries).all(1)&(queries>=0).all(1)]
        points=np.vstack([projected,queries]);lo=points.min(0);hi=points.max(0)
        pad=np.maximum(12.,.15*(hi-lo));lo=np.maximum(0,lo-pad);hi=np.minimum([width,height],hi+pad)
        if np.any(hi-lo<16):return dict(source,mode='FULL_CONTEXT',xyxy_raw=full,reason='DEGENERATE_OR_OUTSIDE_INITIAL_ENVELOPE')
        outline=result['packet']['solver_input'].get('outline_observation',{})
        observed=np.asarray(outline.get('observed_boundary_samples_xy',[]),float)
        known=np.zeros((0,2))
        if observed.ndim==2 and observed.shape[1:]==(2,):
            known=observed[np.isfinite(observed).all(1)&(observed>=0).all(1)&(observed<=[width,height]).all(1)]
        outside=(known<lo).any(1)|(known>hi).any(1)if len(known)else np.zeros(0,bool)
        return dict(source,mode='PREDICTION_ROI',xyxy_raw=[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])],
            reason='INITIAL_PREDICTION_ENVELOPE',automatic_outline_guard=False,outline_is_GT=False,
            saved_automatic_outline_known_sample_count=len(known),
            saved_automatic_outline_outside_ROI_count=int(outside.sum()),
            saved_automatic_outline_outside_ROI=bool(outside.any()),
            outside_support_policy='Keep valid prediction ROI; disclose outside samples; full redacted context is always linked')
    except (KeyError,ValueError,TypeError,IndexError,FloatingPointError):
        return dict(source,mode='FULL_CONTEXT',xyxy_raw=full,reason='NA_INVALID_SAVED_INITIAL_GEOMETRY')


def object_read(path):
    path=Path(path)
    with (gzip.open if path.suffix=='.gz' else open)(path,'rt')as stream:return __import__('json').load(stream)


def bound_artifact(path,seal):
    path=Path(path);value=seal.get('artifacts',{}).get(path.name)
    if isinstance(value,dict):value=value.get('sha256')
    choices=[value]+[seal.get(k)for k in ('head_features_sha256','features_sha256','predictions_sha256','prediction_sha256')]
    if U.sha(path)not in choices:raise ValueError('Unbound gallery input: '+str(path))


def predictions(folder):
    folder=Path(folder);path=folder/'PREDICTIONS.jsonl.gz';seal=U.read(folder/'PREDICTION_SEAL.json')
    if not seal.get('complete'):raise ValueError('Complete prediction seal required')
    bound_artifact(path,seal)
    for key in ('reference_reads','reference_inputs','real_reference_inputs','GT_inputs'):
        if seal.get(key)not in (None,0,False):raise ValueError('Production prediction reference firewall differs')
    data=list(U.rows(path));index={row['id']:row for row in data}
    if len(data)!=238 or len(index)!=238:raise ValueError('All238 sealed predictions required')
    return index,dict(path=str(path),sha256=U.sha(path),seal_sha256=U.sha(folder/'PREDICTION_SEAL.json'))


def transition(before,after):
    return 'BOTH_SUCCESS'if before and after else'RECOVERY'if after else'DAMAGE'if before else'BOTH_FAILURE'


def comparison_coverage(coverage,scores,before_prefix):
    result=deepcopy(coverage)
    for row in result:
        for seed in SEEDS:
            value=row['selected'][str(seed)];before=scores[row['id']]['metrics'][before_prefix+'_s'+str(seed)]
            value['baseline_metric']=deepcopy(before)
            value['transition']=transition(before['success'],value['metric']['success'])
    return result


def choose_cases(coverage,fixed_examples=()):
    """All transitions, then fixed ID/cluster/failure and worst-tail rules."""
    selected={};transition_ids=set();rows=[row for row in coverage if row['partition']in PRIMARY]
    def add(row,category,seed=None):
        key=(row['arm'],row['id']);item=dict(category=category)
        if seed is not None:item['seed']=seed
        selected.setdefault(key,[])
        if item not in selected[key]:selected[key].append(item)
    for row in rows:
        for seed in SEEDS:
            value=row['selected'][str(seed)]
            if value['transition']in ('RECOVERY','DAMAGE'):
                add(row,value['transition'],seed);transition_ids.add(row['id'])
    lookup={(row['arm'],row['id']):row for row in rows}
    for example in fixed_examples:
        key=(example['arm'],example['id'])
        if key not in lookup:raise ValueError('Fixed gallery example must belong to a completed PRIMARY arm')
        add(lookup[key],'FIXED_POSTHOC_DIAGNOSTIC/'+example['reason'])
    for arm in sorted({row['arm']for row in rows}):
        armrows=sorted((row for row in rows if row['arm']==arm),key=lambda r:r['id'])
        for cluster in sorted({row['cluster_id']for row in armrows}):
            normal=[r for r in armrows if r['cluster_id']==cluster and all(r['selected'][str(s)]['transition']=='BOTH_SUCCESS'for s in SEEDS)]
            if normal:add(normal[0],'NORMAL_ALL3SEEDS/'+cluster)
        for seed in SEEDS:
            for category in ('GENERATED_ABSENT','ADMISSION_LOSS','SELECTION_LOSS'):
                failures=[r for r in armrows if r['selected'][str(seed)]['failure_stage']==category]
                if failures:add(failures[0],category,seed)
            for side in ('metric','baseline_metric'):
                for axis in ('translation_cm','rotation_deg'):
                    available=[r for r in armrows if r['selected'][str(seed)][side].get(axis)is not None]
                    if available:
                        worst=min(available,key=lambda r:(-r['selected'][str(seed)][side][axis],r['id']))
                        add(worst,'LARGEST_'+side+'/'+axis,seed)
            fallback=[r for r in armrows if r['selected'][str(seed)].get('fallback')]
            if fallback:add(fallback[0],'FALLBACK',seed)
    return selected,sorted(transition_ids)


def saved_action_rank(coverage,prediction,feature):
    """Saved within-frame logits plus previously sealed good-action labels.

    Returns one private saved pose for a diagnostic overlay. No forward,
    reference decode, success recomputation or numerical pose match is used.
    """
    good=set(coverage['good_eligible_ids']);actions=prediction.get('actions',[])
    if any(not math.isfinite(a['score'])for a in actions):raise ValueError('Finite saved action logits required')
    ordered=sorted(enumerate(actions),key=lambda pair:(-pair[1]['score'],pair[0]))
    labelled=[(index,action,bool(good&({action['candidate_id']}|set(action.get('source_aliases',[])))))for index,action in ordered]
    best=next(((rank,action)for rank,(_,action,success)in enumerate(labelled,1)if success),None)
    pose=None
    if best is not None:
        cid=best[1]['candidate_id'];candidate=next((c for c in feature['candidates']if c['candidate_id']==cid),None)
        if candidate is not None:pose=candidate['actual_pose']
        elif cid=='N0':pose=feature.get('fallback')
    selected=prediction.get('selected_score')
    summary=dict(action_count=len(actions),posthoc_good_actions=sum(success for _,_,success in labelled),
        best_good_action_id=best[1]['candidate_id']if best else None,best_good_action_rank=best[0]if best else None,
        best_good_logit_minus_selected=best[1]['score']-selected if best and selected is not None else None,
        top_actions=[dict(candidate_id=a['candidate_id'],source_aliases=a.get('source_aliases',[]),score=a['score'],
            posthoc_actual_success=success)for _,a,success in labelled[:5]],
        logit_is_calibrated_confidence=False,diagnostic_only=True,
        label_scope='previously sealed eligible-bank success labels;no metric recomputation',
        rank_scope='same image/arm/seed;ties retain saved action order',
        best_good_saved_pose_overlay_available=bool(pose and pose.get('available')))
    return summary,pose


def public_safe(value):
    """Reject any accidental full pose/reference payload in derived case JSON."""
    forbidden={'R','t','R_fixed','t_gt','R_physical','R_cf','centroid','reference_pose','references','reference','actual_pose'}
    if isinstance(value,dict):
        if forbidden&set(value):raise ValueError('Pose/reference payload forbidden in public gallery JSON')
        for item in value.values():public_safe(item)
    elif isinstance(value,(list,tuple)):
        for item in value:public_safe(item)
    return U.finite(value)


def observation_records(result):
    """Saved q/cov only; revoked archives and S slots are not 2D measures."""
    spec=result['packet']['solver_input'];partials=spec.get('boundary_observations',[])
    revoked={r['native_corner']for r in partials};full=[];archives=[];normals=[]
    for row in spec['visible']:
        k=row['native_corner'];state=row.get('state',spec['states'][k])
        if state not in ('B','I'):continue
        entry=dict(native_corner=k,state=state,dimension=0 if k in revoked else 2,
            source=row.get('source','UNKNOWN'),physical_source_ids=row.get('physical_source_ids',[]),
            physical_owner='UNKNOWN_NOT_INDEPENDENTLY_CERTIFIED',reliability_weight=row.get('fit_reliability_weight'),
            conditional_covariance_px2=row.get('covariance_px2'),saved_fit_covariance_px2=row.get('fit_covariance_px2'),
            complete_2D_revoked=k in revoked)
        if k in revoked:
            # Keep only the archive's hash; no display point implies a new2D
            # measurement. Actual normal support is recorded independently.
            entry['revoked_archive_payload_sha256']=U.payload_sha(row);archives.append(entry)
        else:entry['xy']=row['xy'];full.append(entry)
    for group in partials:
        for index,line in enumerate(group['lines']):
            record=line.get('provenance',{}).get('line_record',{})
            points=record.get('observed_points',[]);inliers=record.get('inlier_mask')
            if inliers is not None and len(inliers)==len(points):points=[p for p,good in zip(points,inliers)if good]
            normals.append(dict(native_corner=group['native_corner'],state=group.get('state'),dimension=1,
                normal_unit_xy=line['normal_unit_xy'],offset_px=line['offset_px'],normal_variance_px2=line['normal_variance_px2'],
                raw_image_support_xy=points,physical_source_ids=line.get('physical_source_ids',group.get('physical_source_ids',[])),
                physical_source_id=line.get('physical_source_id'),line_source=record.get('measurement_source',line.get('source','UNKNOWN')),
                support_interval_archive_only=line.get('support_tangent_interval_px'),complete_2D_revoked=True,
                tangent_is_measured=False,physical_owner='UNKNOWN_NOT_INDEPENDENTLY_CERTIFIED',line_index=index))
    return dict(states=spec['states'],full_points=full,normal_factors=normals,revoked_archives=archives,
        S_point_observations=0,source='saved actual solver_input; covariance kind explicit, no effective floor reconstructed')


def candidate_details(feature,prediction):
    cid=prediction.get('selected_candidate_id');candidate=next((c for c in feature['candidates']if c['candidate_id']==cid),None)
    result={key:prediction.get(key)for key in ('selected_candidate_id','selected_source_aliases','selected_score','fallback','prediction_status')}
    if candidate is None:return dict(result,candidate_record_available=False,costs='NA_NO_MATCHING_SAVED_FEATURE',both_family_costs='NA_NOT_STORED;no new costs evaluated')
    result.update(candidate_record_available=True,source_aliases=candidate.get('source_aliases',[]),
        source_baseline_aliases=candidate.get('source_baseline_aliases',[]),eligible=candidate['eligible'],
        admission_rule=candidate.get('admission_rule','original candidate eligibility'),feature_context_family=candidate.get('feature_context_family'))
    x=candidate['features']
    result['saved_feature_costs']={mode:dict(value=U.decode(x[j])if x[31+j]else None,available=bool(x[31+j]),
        relative_gap=U.decode(x[26+j])if x[57+j]else None,gap_available=bool(x[57+j]))for j,mode in enumerate(U.MODES)}
    result['saved_min_constraint_slack']=U.decode(x[23])if x[54]else None
    result['both_family_costs']='NA_NOT_STORED_FOR_THIS_POSE;feature cost is selected fixed-score context only;no new cost evaluation'
    return result


def family_readouts(result):
    """Actual saved selected G13 poses; costs are stage costs, not H1 scores."""
    readouts={}
    for family in ('ORIGINAL_WD','SWAPPED_WD'):
        candidate=next((c for c in result['candidates']if c['candidate_id']==f'G13/{family}/selected'),None)
        if candidate is None:
            readouts[family]=dict(pose=None,public=dict(available=False,reason='NA_NO_ACTUAL_SAVED_G13_FAMILY_READOUT'));continue
        pose=dict(available=True,R_physical=candidate['R_physical'],centroid=candidate['centroid'])
        record=None
        for value in result.get('stage_results',{}).get('G13',{}).get('families',[]):
            if value['name']==family:record=value.get('solution')
        if record is None:record=candidate.get('execution_record')
        if record is None:
            for value in result['candidates']:
                if not value['candidate_id'].startswith('G13/'+family+'/endpoint/'):continue
                if np.array_equal(value['R_physical'],pose['R_physical']) and np.array_equal(value['centroid'],pose['centroid']):
                    record=value.get('execution_record');break
        keys=('objective','objective_components','minimum_joint_slack','minimum_constraint_slack','joint_feasible','accepted','status','optimizer_success','optimizer_status')
        saved={key:record[key]for key in keys if record and key in record}
        # Objective audits can contain projected points/pose parameters; export
        # only scalar additive parts, never an entire raw optimization audit.
        if 'objective_components'in saved:
            saved['objective_components']={key:value for key,value in saved['objective_components'].items()if isinstance(value,(int,float,bool))or value is None}
        readouts[family]=dict(pose=pose,public=dict(available=True,candidate_id=candidate['candidate_id'],
            source_native_family=family,stored_stage_costs_and_constraints=saved or 'NA_NOT_STORED',
            cost_scope='saved G13 stage objective,not complete five-mode H1 cost vector',
            family_chosen_by_G10=result.get('selected_family')==family))
    return readouts


def admission_summary(feature):
    result=dict(unique_candidates=len(feature['candidates']),eligible_candidates=sum(c['eligible']for c in feature['candidates']),
        original_projection_rejections=feature.get('projection_rejections',[]))
    for name in ('kernel_extra','extra_union','s1_union'):
        if name not in feature:continue
        info=feature[name]
        for key in ('complete_action_rejections','new_complete_action_rejections','PF_rejections','complete_action_duplicates'):
            if key in info:result[name+'/'+key]=info[key]
    return result


def validate_inputs(manifest,d1,evaluation_dir,coverage_dir,independent_dir=None):
    plan=U.read(manifest);d1=Path(d1);ed=Path(evaluation_dir);cd=Path(coverage_dir)
    es=U.read(ed/'EVALUATION_SEAL.json');cs=U.read(cd/'ANALYSIS_SEAL.json')
    if not es.get('complete')or not cs.get('complete'):raise ValueError('Completed evaluation and postbank analysis required')
    score_path=ed/'SCORED_ROWS.jsonl.gz';coverage_path=cd/'BANK_COVERAGE_ROWS.jsonl.gz'
    bound_artifact(score_path,es);bound_artifact(coverage_path,cs)
    scores={r['id']:r for r in U.rows(score_path)};coverage=list(U.rows(coverage_path))
    baseline,bb=predictions(d1);bindings=dict(baseline=bb,arms={})
    if set(scores)!=set(baseline):raise ValueError('Exact full238 scored/prediction population required')
    if Counter(r['partition']for r in scores.values())!=Counter(TEST=49,REMAINDER=143,CAL=46):raise ValueError('Original reporting denominator differs')
    allowed={row['path']:row['sha256']for row in es['prediction_bindings']}
    if allowed.get(bb['path'])!=bb['sha256']:raise ValueError('Baseline predictions differ from scored binding')
    source={};coverage_index={(r['arm'],r['id']):r for r in coverage}
    if len(coverage_index)!=len(coverage):raise ValueError('Duplicate coverage row')
    if 'INPUT_BINDINGS.json'not in cs.get('artifacts',{}):raise ValueError('Post-bank V2 authenticated INPUT_BINDINGS required for gallery packaging')
    bound_artifact(cd/'INPUT_BINDINGS.json',cs)
    input_binding=U.read(cd/'INPUT_BINDINGS.json')
    for name,spec in plan['arms'].items():
        pred,binding=predictions(spec['prediction_dir']);fs=U.read(spec['feature_seal'])
        if not fs.get('complete'):raise ValueError('Complete own feature seal required')
        bound_artifact(spec['features'],fs);features=list(U.rows(spec['features']));byid={r['id']:r for r in features}
        ps=U.read(Path(spec['prediction_dir'])/'PREDICTION_SEAL.json')
        if ps.get('schema')=='NEW_CAMPAIGN_H1_PREDICTIONS_V1'and(ps['features_sha256']!=U.sha(spec['features'])or ps['feature_seal_sha256']!=U.sha(spec['feature_seal'])):
            raise ValueError('Own H1 prediction feature binding differs from supplied display bank')
        if len(features)!=238 or set(byid)!=set(scores)or set(pred)!=set(scores):raise ValueError('Full arm feature/prediction denominator differs')
        if allowed.get(binding['path'])!=binding['sha256']:raise ValueError('Arm prediction differs from completed scoring')
        prior=input_binding[name]
        if prior['features']['sha256']!=U.sha(spec['features'])or prior['feature_seal']['sha256']!=U.sha(spec['feature_seal'])or prior['prediction']['sha256']!=binding['sha256']:
            raise ValueError('Coverage belongs to another feature/prediction bank')
        for fid,row in pred.items():
            cov=coverage_index.get((name,fid))
            if cov is None:raise ValueError('All238 coverage rows required per arm')
            for key in ('partition','session','cluster_id'):
                if any(value[key]!=scores[fid][key]for value in (row,byid[fid],cov)):
                    raise ValueError('Saved gallery population/cluster identity differs')
            for seed in SEEDS:
                arm=name+'_s'+str(seed);before='D1_H1_MLP_s'+str(seed)
                if row['predictions'][arm]['actual_pose']!=scores[fid]['poses'][arm]or baseline[fid]['predictions'][before]['actual_pose']!=scores[fid]['poses'][before]:
                    raise ValueError('Scored pose differs from sealed actual prediction')
                value=cov['selected'][str(seed)]
                for label,key in (('metric',arm),('baseline_metric',before)):
                    actual=value[label];canonical=scores[fid]['metrics'][key]
                    if actual['success']!=canonical['success']or actual['pose_available']!=canonical['pose_available']:
                        raise ValueError('Coverage stored success/availability differs from completed score')
                    for axis in ('translation_cm','rotation_deg'):
                        if (actual[axis]is None)!=(canonical[axis]is None)or(actual[axis]is not None and abs(actual[axis]-canonical[axis])>1e-6):
                            raise ValueError('Coverage stored numerical metric differs from completed score')
                if value['transition']!=transition(scores[fid]['metrics'][before]['success'],scores[fid]['metrics'][arm]['success']):raise ValueError('Coverage transition differs from completed score')
                if value['selected_candidate_id']!=row['predictions'][arm].get('selected_candidate_id'):raise ValueError('Selected candidate identity differs')
                category='SUCCESS'if scores[fid]['metrics'][arm]['success']else'GENERATED_ABSENT'if not cov['generated_good']else'ADMISSION_LOSS'if not cov['eligible_good']else'SELECTION_LOSS'
                if value['failure_stage']!=category:raise ValueError('Coverage failure-stage attribution differs')
        source[name]=dict(spec=spec,predictions=pred,features=byid,ordered_ids=[r['id']for r in features])
        bindings['arms'][name]=dict(prediction=binding,feature_sha256=U.sha(spec['features']),feature_seal_sha256=U.sha(spec['feature_seal']))
    if independent_dir is not None:
        root=Path(independent_dir);check=U.read(root/'INDEPENDENT_VERIFICATION.json')
        if not check.get('PASS')or check['canonical_evaluation_seal_sha256']!=U.sha(ed/'EVALUATION_SEAL.json'):raise ValueError('Completed independent check belongs to different scoring')
        independently={r['id']:r for r in U.rows(root/'INDEPENDENT_ROWS.jsonl.gz')}
        if set(independently)!=set(scores):raise ValueError('Independent full238 denominator differs')
        for fid,row in independently.items():
            for arm,metric in scores[fid]['metrics'].items():
                actual=row['metrics'][arm]
                if row['poses'][arm]!=scores[fid]['poses'][arm]or actual['success']!=metric['success']:raise ValueError('Independent pose/classification differs')
                for axis in ('translation_cm','rotation_deg'):
                    if (actual[axis]is None)!=(metric[axis]is None)or(actual[axis]is not None and abs(actual[axis]-metric[axis])>1e-6):raise ValueError('Independent numeric score differs')
        bindings['independent']=dict(verification_sha256=U.sha(root/'INDEPENDENT_VERIFICATION.json'),rows_sha256=U.sha(root/'INDEPENDENT_ROWS.jsonl.gz'))
    comparison=plan.get('comparison_baseline')
    if comparison is not None:
        before_prefix=comparison['arm'];alternative,ab=predictions(comparison['prediction_dir'])
        if set(alternative)!=set(scores)or allowed.get(ab['path'])!=ab['sha256']:
            raise ValueError('Explicit matched comparison baseline is not a completed scored prediction')
        fs=U.read(comparison['feature_seal']);bound_artifact(comparison['features'],fs)
        ps=U.read(Path(comparison['prediction_dir'])/'PREDICTION_SEAL.json')
        if not fs.get('complete')or ps.get('features_sha256')!=U.sha(comparison['features'])or ps.get('feature_seal_sha256')!=U.sha(comparison['feature_seal']):
            raise ValueError('Matched comparison baseline own feature binding differs')
        for fid,row in alternative.items():
            for seed in SEEDS:
                arm=before_prefix+'_s'+str(seed)
                if row['predictions'][arm]['actual_pose']!=scores[fid]['poses'][arm]:raise ValueError('Matched baseline actual scored pose differs')
        coverage=comparison_coverage(coverage,scores,before_prefix);baseline=alternative
        bindings['comparison_baseline']=dict(arm=before_prefix,prediction=ab,features_sha256=U.sha(comparison['features']),feature_seal_sha256=U.sha(comparison['feature_seal']))
    bindings.update(manifest_sha256=U.sha(manifest),evaluation_seal_sha256=U.sha(ed/'EVALUATION_SEAL.json'),
        coverage_seal_sha256=U.sha(cd/'ANALYSIS_SEAL.json'),score_rows_sha256=U.sha(score_path),coverage_rows_sha256=U.sha(coverage_path),
        same_manifest_as_coverage=cs['manifest_sha256']==U.sha(manifest),
        display_manifest_extension_scope='Feature/prediction bindings checked even when extra display-only fields differ from bank manifest')
    return plan,source,baseline,scores,[r for r in coverage if r['arm']in source],bindings,es


def core_result(name,source,fid,ordinal,d1):
    spec=source['spec']
    if spec['backend']=='fullnormal':
        folder=Path(spec['result_root'])/f'{ordinal:04d}';sp=folder/'FRAME_SEAL.json';seal=U.read(sp)
        fs=U.read(spec.get('generated_feature_seal',spec['feature_seal']))
        if not fs.get('complete')or fs.get('rows')!=238:raise ValueError('Complete full238 core generated feature seal required')
        if U.sha(sp)!=fs['frame_seal_sha256'][f'{ordinal:04d}']:raise ValueError('Fullnormal display packet row ancestry differs')
        path=folder/'RESULT.json.gz'
        if U.sha(path)!=seal['artifacts']['RESULT.json.gz']:raise ValueError('Fullnormal actual result changed')
        result=object_read(path)
    elif spec['backend']=='original_union':
        folder=Path(d1)/'d1_real_candidates_v1/frames'/f'{ordinal:04d}';sp=folder/'FRAME_SEAL.json';seal=U.read(sp)
        fs=U.read(Path(d1)/'d1_real_candidates_v1/FEATURE_SEAL.json')
        if U.sha(sp)!=fs['per_frame_execution_seal_sha256'][f'{ordinal:04d}']:raise ValueError('Original packet row ancestry differs')
        path=folder/'E_RUN.jsonl.gz'
        if U.sha(path)!=seal['e_run_sha256']:raise ValueError('Original actual result changed')
        result=next(U.rows(path))
    else:raise ValueError('Explicit original_union/fullnormal backend required')
    if result['id']!=fid:raise ValueError('Display packet identity differs')
    return result,dict(result_sha256=U.sha(path),frame_seal_sha256=U.sha(sp),backend=spec['backend'])


def support_result(spec,fid,ordinal):
    if not spec.get('support_root'):return None,dict(available=False,reason='NA_NO_STORED_GENERIC_SUPPORT_ARCHIVE_PROVIDED')
    root=Path(spec['support_root']);archive_seal=Path(spec['support_archive_seal']);seal=U.read(archive_seal)
    if not seal.get('complete')or seal.get('rows')!=1241:raise ValueError('Complete material all-row archive seal required')
    bp=archive_seal.parent/'ALL_ROW_BINDINGS.jsonl.gz'
    if U.sha(bp)!=seal['all_row_seal_and_frame_bindings_sha256']:raise ValueError('Generic support row bindings differ')
    row=next((r for r in U.rows(bp)if r['id']==fid),None)
    if row is None:raise ValueError('Missing generic support identity binding')
    path=root/f'{ordinal:04d}'/'FRAME.jsonl.gz';sp=path.parent/'SEAL.json'
    if U.sha(path)!=row['frame_sha256']or U.sha(sp)!=row['row_seal_sha256']:raise ValueError('Generic actual support archive differs')
    result=next(U.rows(path))
    if result['id']!=fid:raise ValueError('Generic support identity differs')
    return result,dict(available=True,frame_sha256=U.sha(path),row_seal_sha256=U.sha(sp),archive_seal_sha256=U.sha(archive_seal))


def line_segment(normal,offset,width,height):
    n=np.asarray(normal,float);points=[]
    for x in (0.,float(width)):
        if abs(n[1])>1e-12:
            y=(-offset-n[0]*x)/n[1]
            if 0<=y<=height:points.append([x,y])
    for y in (0.,float(height)):
        if abs(n[0])>1e-12:
            x=(-offset-n[1]*y)/n[0]
            if 0<=x<=width:points.append([x,y])
    unique=[]
    for point in points:
        if point not in unique:unique.append(point)
    return np.asarray(unique[:2])


def draw_observations(ax,records,hw):
    from matplotlib.patches import Ellipse
    for row in records['full_points']:
        xy=np.asarray(row['xy']);color='#08d9d6'if row['state']=='B'else'#ca8cff'
        ax.scatter(*xy,s=18,facecolors='none',edgecolors=color)
        ax.annotate(str(row['native_corner'])+':'+row['state'],xy,fontsize=5,color=color)
        cov=row.get('saved_fit_covariance_px2')or row.get('conditional_covariance_px2')
        if cov is not None:
            matrix=np.asarray(cov,float)
            if matrix.shape==(2,2)and np.isfinite(matrix).all():
                values,vectors=np.linalg.eigh(matrix)
                if values.min()>0:
                    angle=math.degrees(math.atan2(vectors[1,1],vectors[0,1]))
                    ax.add_patch(Ellipse(xy,2*math.sqrt(values[1]),2*math.sqrt(values[0]),angle=angle,fill=False,ec=color,lw=.7))
    for row in records['normal_factors']:
        points=np.asarray(row['raw_image_support_xy'],float)
        if points.ndim==2 and points.shape[1:]==(2,)and len(points):ax.plot(points[:,0],points[:,1],'.',c='#ff4f9a',ms=1)
        else:
            segment=line_segment(row['normal_unit_xy'],row['offset_px'],hw[1],hw[0])
            if len(segment)==2:ax.plot(segment[:,0],segment[:,1],ls=':',c='#ff4f9a',lw=.6)
    ax.set_title('Saved q / Sigma / normal constraints\nB cyan,I purple;revoked2D not plotted;owner UNKNOWN',fontsize=7)


def generic_records(name,spec,result):
    if result is None:return dict(available=False,reason='NA_NO_ACTUAL_GENERIC_SUPPORT_ARCHIVE')
    stage=spec.get('stage',name)
    if stage.startswith('S3_'):
        policy=stage.split('_',1)[1];observation=result.get('material_observations')
        if observation is None:return dict(available=False,reason='NO_VALID_INITIALIZATION_NO_SUPPORT_PACKET')
        packet=observation['policy_packets'][policy]
        factors=[]
        for row in packet['factors']:
            factors.append({key:row[key]for key in ('support_id','role','observation_dimension','xy','object_xyz','normal_unit_xy','offset_px',
                'effective_covariance_px2','effective_variance_px2','physical_source_ids','raw_image_support_xy','cue')if key in row})
        return dict(available=True,stage=stage,actual_factor_count=len(factors),factors=factors,
            no_direct_support=packet['no_direct_support'],physical_owner='UNKNOWN_REGISTERED_PROXY_ONLY',
            normal_variance_or_covariance='saved actual generic packet',observation_dimension_is_role_forced=False)
    if stage.startswith('S4_'):
        branch=stage.split('_',1)[1];iterations=result['s4'][branch]['iterations'];records=[]
        for row in iterations:
            record={key:row[key]for key in ('start_id','update','status','accepted_correspondences','observed_representation','proposal_accepted')if key in row}
            if 'matched_observed_xy'in row:
                xy=row['matched_observed_xy'];indices=np.linspace(0,len(xy)-1,min(80,len(xy)),dtype=int)if xy else[]
                record['display_observed_xy_sample']=[xy[i]for i in indices]
                record['display_sample_only']=True
            if 'selected_correspondences'in row:
                records2=row['selected_correspondences'];indices=np.linspace(0,len(records2)-1,min(80,len(records2)),dtype=int)if records2 else[]
                record['display_observed_xy_sample']=[records2[i]['xy']for i in indices]
                record['display_sample_only']=True
            records.append(record)
        return dict(available=True,stage=stage,iteration_correspondence_summary=records,
            physical_owner='UNKNOWN_REGISTERED_PROXY_ONLY',full_correspondence_archive_retained_privately=True,
            original_native_packet_is_downstream_feature_context=True)
    return dict(available=False,reason='NA_NO_GENERIC_STAGE_DECLARED')


def draw_generic(ax,generic):
    if not generic.get('available'):return
    for factor in generic.get('factors',[]):
        xy=factor.get('xy');color='#ffc233'
        if xy is not None and np.isfinite(xy).all():
            if factor['observation_dimension']==2:ax.plot(*xy,marker='o',mfc='none',mec=color,ms=4)
            elif 'normal_unit_xy'in factor:
                normal=np.asarray(factor['normal_unit_xy']);ax.plot([xy[0]-6*normal[0],xy[0]+6*normal[0]],
                    [xy[1]-6*normal[1],xy[1]+6*normal[1]],c=color,lw=.8)
    for iteration in generic.get('iteration_correspondence_summary',[]):
        points=np.asarray(iteration.get('display_observed_xy_sample',[]),float)
        if points.ndim==2 and len(points):ax.plot(points[:,0],points[:,1],'.',c='#ffc233',ms=1,alpha=.5)
    ax.set_title('Saved stage support in yellow;native feature q shown\nphysical owner UNKNOWN;S4 samples are display-only',fontsize=7)


def render(manifest,d1,evaluation_dir,coverage_dir,output,*,reference_dir=None,independent_dir=None,root_GO=False):
    if root_GO is not True:raise PermissionError('Postseal gallery rendering requires explicit root GO')
    started=time.monotonic();output=Path(output);d1=Path(d1)
    plan,source,baseline,scores,coverage,bindings,es=validate_inputs(manifest,d1,evaluation_dir,coverage_dir,independent_dir)
    selected,all_transition_ids=choose_cases(coverage,plan.get('fixed_examples',()))
    coverage_index={(row['arm'],row['id']):row for row in coverage}
    baseline_prefix=plan.get('comparison_baseline',{}).get('arm','D1_H1_MLP')
    rights=U.read(d1/'RIGHTS_SCOPE_BINDING.json');privacy=U.read(d1/'DISPLAY_PRIVACY_PLAN.json')
    if U.sha(d1/'DISPLAY_PRIVACY_PLAN.json')!=rights['display_privacy_plan_sha256']or U.sha(d1/'RIGHTS_RECEIPT.json')!=rights['rights_receipt_sha256']:
        raise ValueError('Existing display rights/privacy binding differs')
    primary={fid for fid,row in scores.items()if row['partition']in PRIMARY}
    if set(rights['approved_ids'])!=primary or len(primary)!=192:raise ValueError('Exact PRIMARY192 display scope required')
    if any(fid not in primary for _,fid in selected):raise ValueError('CAL RGB display is prohibited')
    refpath=None
    if reference_dir is not None:
        refpath=Path(reference_dir)/'FIXED_REFERENCES.json';rs=U.read(Path(reference_dir)/'SCORE_SEAL.json')
        if U.sha(refpath)!=es['reference_sha256']or U.sha(refpath)!=rs['artifacts'][refpath.name]:raise ValueError('Optional private display reference differs from completed score')
    completed=output/'GALLERY_MANIFEST.json'
    if completed.exists():
        receipt=U.read(completed)
        if not receipt['complete']or receipt['input_bindings']!=bindings or receipt['display_source_closure']!=display_sources():raise ValueError('Completed gallery input/source binding differs')
        if receipt['privacy_plan_sha256']!=U.sha(d1/'DISPLAY_PRIVACY_PLAN.json')or receipt['reference_projection_requested']!=(reference_dir is not None):raise ValueError('Completed gallery display policy differs')
        for name,digest in receipt['artifact_sha256'].items():
            if U.sha(output/name)!=digest:raise ValueError('Completed gallery artifact differs')
        return receipt
    refs=None
    if refpath is not None:
        refs=U.read(refpath)['references']
    # Image libraries and image reads occur only after all complete input seals
    # and PRIMARY-specific display rights have passed.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.research.pallet_boundary_action_repair_v1.display_privacy import display_image
    from .s1_analysis import wire
    output.mkdir(parents=True,exist_ok=True);upstream=list(U.rows(U.paths(d1,'real')[0]));indices={f['id']:i for i,f in enumerate(upstream)}
    if {r['id']for r in upstream}!=set(scores):raise ValueError('Exact original upstream IDs required')
    cases=[]
    def metric_title(metric):
        return 'NO POSE'if metric['translation_cm']is None else f"{metric['translation_cm']:.2f}cm / {metric['rotation_deg']:.2f}deg / "+('SUCCESS'if metric['success']else'FAIL')
    for order,((name,fid),categories)in enumerate(sorted(selected.items())):
        ordinal=indices[fid];frame=upstream[ordinal];entry=source[name]
        original,original_binding=core_result('BASELINE',dict(spec=plan.get('comparison_baseline',dict(backend='original_union'))),fid,ordinal,d1)
        result,current_binding=core_result(name,entry,fid,ordinal,d1)
        support,support_binding=support_result(entry['spec'],fid,ordinal)
        before_obs=observation_records(original);current_obs=observation_records(result);generic=generic_records(name,entry['spec'],support)
        baseline_generic=dict(available=False,reason='Original D1 baseline has no new generic support intervention')
        baseline_support_binding=dict(available=False)
        if plan.get('comparison_baseline',{}).get('support_root'):
            baseline_support,baseline_support_binding=support_result(plan['comparison_baseline'],fid,ordinal)
            baseline_generic=generic_records(baseline_prefix,plan['comparison_baseline'],baseline_support)
        image=display_image(frame,privacy);reference_pose=None
        roi=display_roi(frame,original);x0,y0,x1,y1=roi['xyxy_raw']
        if refs is not None:
            rr=refs[frame['partition']][fid];reference_pose=dict(available=True,R_physical=rr['R'],centroid=rr['t'])
        fig,axes=plt.subplots(4,3,figsize=(12,12));fig.suptitle(name+' | '+fid+' | '+','.join(sorted({c['category']for c in categories})),fontsize=8)
        for ax in axes.flat:
            ax.imshow(image);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_xticks([]);ax.set_yticks([])
        wire(axes[0,0],frame['initial_pose'],frame,original,'#30a8ff');axes[0,0].set_title('Saved N0 estimate;fixed automatic states\n'+','.join(f'{i}:{s}'for i,s in enumerate(current_obs['states'])),fontsize=7)
        draw_observations(axes[0,1],before_obs,frame['raw_hw']);draw_observations(axes[0,2],current_obs,frame['raw_hw'])
        draw_generic(axes[0,1],baseline_generic);draw_generic(axes[0,2],generic)
        axes[0,1].set_title(baseline_prefix+' saved q/Sigma/support\nB cyan,I purple,1D pink,generic yellow;owner UNKNOWN',fontsize=7)
        axes[0,2].set_title(name+' saved q/Sigma/support\nRevoked2D not plotted;S4 yellow samples are display-only',fontsize=7)
        actions={};before_actions={};ranks={}
        for row,seed in enumerate(SEEDS,1):
            before=baseline_prefix+'_s'+str(seed);after=name+'_s'+str(seed)
            p0=baseline[fid]['predictions'][before];p1=entry['predictions'][fid]['predictions'][after]
            for col,pred,key,color,context in ((0,p0,before,'#30a8ff',original),(1,p1,after,'#ff5e62',result)):
                if reference_pose is not None:wire(axes[row,col],reference_pose,frame,context,'#24ee7c','--',.85)
                wire(axes[row,col],pred['actual_pose'],frame,context,color);axes[row,col].set_title(key+'\n'+metric_title(scores[fid]['metrics'][key]),fontsize=7)
            detail=candidate_details(entry['features'][fid],p1);actions[str(seed)]=detail
            rank,good_pose=saved_action_rank(coverage_index[(name,fid)],p1,entry['features'][fid]);ranks[str(seed)]=rank
            if good_pose and good_pose.get('available'):
                wire(axes[row,1],good_pose,frame,result,'#ffc233','-.',.85)
            before_actions[str(seed)]={key:p0.get(key)for key in ('selected_candidate_id','selected_source_aliases','selected_score','fallback','prediction_status')}
            # This panel contains text rather than native image coordinates.
            # Clear the inherited imshow and anchor labels in axes coordinates;
            # a cropped image extent otherwise puts (0,.95) outside the panel.
            import textwrap
            label_lines=['seed '+str(seed),'baseline chosen '+str(p0.get('selected_candidate_id')),
                'new chosen '+str(detail.get('selected_candidate_id')),
                'aliases '+str(detail.get('source_aliases',detail.get('selected_source_aliases'))),
                'eligible '+str(detail.get('eligible','NA')),
                'context '+str(detail.get('feature_context_family','NA')),
                'best posthoc good action '+str(rank['best_good_action_id']),
                'rank '+str(rank['best_good_action_rank'])+'; logit minus selected '+str(rank['best_good_logit_minus_selected']),
                'Within-frame logits; uncalibrated','Both-family pose costs: NA unless saved',
                'No new forward / cost / solve / fit']
            label='\n'.join(textwrap.fill(line,width=69,break_long_words=True,break_on_hyphens=False)for line in label_lines)
            axes[row,2].clear();axes[row,2].axis('off')
            axes[row,2].text(0,.98,label,transform=axes[row,2].transAxes,va='top',fontsize=6,clip_on=True)
        fig.text(.5,.015,'Green dashed: private reference projection when requested. Solid: saved selected estimate. Amber dash-dot: saved best posthoc good action,not selected.\n'+roi['mode']+': '+roi['reason']+'; saved outline samples outside ROI: '+str(roi.get('saved_automatic_outline_outside_ROI_count','NA'))+'. Full context linked separately.\nRevoked2D/S are not point observations. Display-only privacy;PRIMARY development data.',ha='center',fontsize=6)
        fig.subplots_adjust(left=.02,right=.98,top=.95,bottom=.06,hspace=.3,wspace=.08)
        filename=f'case_{order:04d}.jpg';fig.savefig(output/filename,dpi=100,pil_kwargs=dict(quality=82));plt.close(fig)
        contextfig,contextax=plt.subplots(figsize=(6,4.8));contextax.imshow(image)
        wire(contextax,frame['initial_pose'],frame,original,'#30a8ff')
        wire(contextax,entry['predictions'][fid]['predictions'][name+'_s'+str(SEEDS[0])]['actual_pose'],frame,result,'#ff5e62')
        from matplotlib.patches import Rectangle
        contextax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='#ffc233',lw=1.2))
        contextax.set_xlim(0,frame['raw_hw'][1]);contextax.set_ylim(frame['raw_hw'][0],0);contextax.axis('off')
        contextax.set_title('Full redacted context | blue N0,red representative saved estimate\nYellow prediction-only display ROI;'+roi['reason'],fontsize=6)
        contextfig.tight_layout();contextfile=f'case_{order:04d}_context.jpg'
        contextfig.savefig(output/contextfile,dpi=100,pil_kwargs=dict(quality=82));plt.close(contextfig)
        families={side:family_readouts(value)for side,value in (('baseline',original),('new',result))}
        ff,aa=plt.subplots(2,2,figsize=(9,7))
        for row,side in enumerate(('baseline','new')):
            context=original if side=='baseline'else result
            for col,family in enumerate(('ORIGINAL_WD','SWAPPED_WD')):
                ax=aa[row,col];ax.imshow(image);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.axis('off')
                value=families[side][family]
                if reference_pose is not None:wire(ax,reference_pose,frame,context,'#24ee7c','--')
                if value['pose']is not None:wire(ax,value['pose'],frame,context,'#ffc233')
                detail=value['public'].get('stored_stage_costs_and_constraints');cost='NA';slack='NA'
                if isinstance(detail,dict):cost=detail.get('objective','NA');slack=detail.get('minimum_joint_slack',detail.get('minimum_constraint_slack','NA'))
                ax.set_title(side+' '+family+'\n'+('saved actual family readout'if value['pose']else'NA: no saved actual family readout')+'\nG13 stage objective '+str(cost)+' / min slack '+str(slack),fontsize=7)
        ff.tight_layout();familyfile=f'case_{order:04d}_families.jpg';ff.savefig(output/familyfile,dpi=100,pil_kwargs=dict(quality=82));plt.close(ff)
        data=public_safe(dict(id=fid,arm=name,categories=categories,partition=frame['partition'],panel=filename,family_panel=familyfile,
            full_context_panel=contextfile,display_roi=roi,
            original_image_sha256=frame['image_sha256'],privacy_rectangle_count=len(privacy['frames'][fid]['redactions']),
            comparison_baseline=baseline_prefix,
            metrics={key:scores[fid]['metrics'][key]for seed in SEEDS for key in (baseline_prefix+'_s'+str(seed),name+'_s'+str(seed))},
            selected_actions=actions,baseline_selected_actions=before_actions,baseline_native_observations=before_obs,current_native_observations=current_obs,
            saved_action_rank_diagnostic=ranks,
            generic_support=generic,baseline_generic_support=baseline_generic,
            candidate_admission=admission_summary(entry['features'][fid]),
            actual_family_readouts={side:{key:value['public']for key,value in table.items()}for side,table in families.items()},
            display_input_bindings=dict(baseline=original_binding,current=current_binding,generic=support_binding,baseline_generic=baseline_support_binding),
            physical_ownership='UNKNOWN;automaticnative/proxyassociationonly',reference_projection_present=reference_pose is not None,
            reference_pose_or_GT_published=False,raw_RGB_copied=False,new_cost_evaluations=0,new_pose_calls=0,new_fits=0))
        U.write(output/f'case_{order:04d}.json',data);cases.append(dict(id=fid,arm=name,categories=categories,panel=filename,family_panel=familyfile,context_panel=contextfile,json=f'case_{order:04d}.json'))
    cards=['<article><h2>'+html.escape(c['arm']+' '+c['id'])+'</h2><p>'+html.escape(', '.join(r['category']for r in c['categories']))+'</p><img src="'+c['panel']+'"><details><summary>Full redacted context / prediction-only ROI</summary><img src="'+c['context_panel']+'"></details><details><summary>Actual family readouts and provenance</summary><img src="'+c['family_panel']+'"><a href="'+c['json']+'">Derived metadata (no reference pose)</a></details></article>'for c in cases]
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Completed stage recovery/damage gallery</title><style>body{font:14px system-ui;margin:20px;background:#edf1f5}article{background:white;margin:18px 0;padding:12px}img{max-width:1200px;width:100%}</style><h1>Completed PRIMARY paired gallery</h1><p>Every recovery/damage ID across all three seeds,plus fixed normal/failure/tail rules. Physical ownership unknown. Derived redacted panels only;pixel review pending.</p>'+''.join(cards))
    receipt=dict(complete=True,component='derived gallery generation,not pixel approval or final scientific stage',case_count=len(cases),cases=cases,
        all_PRIMARY_transition_ids=all_transition_ids,input_bindings=bindings,rights_scope_sha256=U.sha(d1/'RIGHTS_SCOPE_BINDING.json'),
        privacy_plan_sha256=U.sha(d1/'DISPLAY_PRIVACY_PLAN.json'),source_sha256=U.sha(__file__),
        display_source_closure=display_sources(),
        reference_display_scope='private canonical projection after completed prediction/score/coverage seals'if refs is not None else'NA_NO_REFERENCE_PROJECTION_REQUESTED',
        reference_projection_requested=reference_dir is not None,
        raw_RGB_copied=False,CAL_RGB_opened=0,public_reference_pose_or_GT=False,new_cost_evaluations=0,new_pose_calls=0,new_fits=0,
        pixel_review='PENDING_ACTUAL_DERIVED_PIXEL_REVIEW',wall_seconds=time.monotonic()-started,
        artifact_sha256={p.name:U.sha(p)for p in output.iterdir()if p.is_file()and p.name!='GALLERY_MANIFEST.json'})
    U.write(output/'GALLERY_MANIFEST.json',receipt);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('manifest','d1','evaluation-dir','coverage-dir','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--reference-dir');parser.add_argument('--independent-dir');parser.add_argument('--root-go',action='store_true');args=parser.parse_args()
    result=render(args.manifest,args.d1,args.evaluation_dir,args.coverage_dir,args.output,
        reference_dir=args.reference_dir,independent_dir=args.independent_dir,root_GO=args.root_go)
    print(U.canonical({k:result[k]for k in ('complete','case_count','new_cost_evaluations','new_pose_calls','new_fits')}),flush=True)
