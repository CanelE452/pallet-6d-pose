"""Postseal full-campaign evaluation. Never imported by the prediction solver.

Reuses the already verified physical/C2 metric and descriptive bootstrap
policy. Candidate success is computed HERE only, after all output seals, for
diagnosis; it is never written back to the prediction/candidate inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import resource
import time

import numpy as np

from scripts.research.pallet_accuracy_closeout_20261017_v1.baseline_audit import metric
from scripts.research.pallet_accuracy_closeout_20261017_v1.evaluation import aggregate, paired
from .common import read, rows, save_rows, sha, write

BASELINES = ('R0','N0','E0','POSEFIX_RAW','POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR')
COMPARATORS = ('E0','N0','POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR')
POPULATIONS = {'PRIMARY192':('TEST','REMAINDER'),'TEST49':('TEST',),
               'REMAINDER143':('REMAINDER',),'CAL46':('CAL',)}
EXPECTED_N = {'PRIMARY192':192,'TEST49':49,'REMAINDER143':143,'CAL46':46}
ARM_MODE = {'CTRL_CARRY':'fixed','CTRL_BUDGET':'fixed','CTRL_FIXED':'fixed',
            'A_FIXED_BANK':'A','B_FIXED_BANK':'B','A_FULL':'A','B_FULL':'B',
            'AB_FULL':'AB','CTRL_EDGE_BOUNDARY':'edge'}


def _csv(path, values):
    if not values:
        raise ValueError('Nonempty CSV expected')
    keys = list(dict.fromkeys(key for row in values for key in row))
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(values)


def _safe_metric(pose, reference):
    try:
        return {**metric(pose,reference),'metric_error':None}
    except (ValueError,KeyError,TypeError,ArithmeticError) as exc:
        # A real output error remains a full-denominator failure, with its
        # original pose and explicit scoring error preserved in private rows.
        return dict(pose_available=False,translation_cm=None,rotation_deg=None,
                    success=False,metric_error=repr(exc))


def _status(entry):
    source = entry.get('prediction_status')
    result = dict(source) if isinstance(source,dict) else {}
    for key in ('accepted','fallback','technical_failure','solver_rejected','timeout','status','reason'):
        result.setdefault(key,entry.get(key))
    if isinstance(source,str):
        result['status']=source
    for key in ('accepted','fallback','technical_failure','solver_rejected','timeout'):
        if result[key] is not None and type(result[key]) is not bool:
            raise ValueError('Explicit boolean or unknown required: '+key)
    return result


def verify_new_seal(run_dir, *, expected_plan_sha256=None,expected_prediction_seal_sha256=None):
    run_dir=Path(run_dir)
    plan,sp=run_dir/'PLAN_LOCK.json',run_dir/'PREDICTION_SEAL.json'
    if expected_plan_sha256 is not None and sha(plan)!=expected_plan_sha256:
        raise ValueError('Campaign plan changed')
    if expected_prediction_seal_sha256 is not None and sha(sp)!=expected_prediction_seal_sha256:
        raise ValueError('Prediction seal changed')
    seal=read(sp)
    if (seal.get('plan_lock_sha256')!=sha(plan) or seal.get('before_scoring') is not True
            or seal.get('reference_inputs') is not False or seal.get('complete') is not True):
        raise ValueError('Complete reference-free predictions must bind the frozen plan before scoring')
    prediction_path=run_dir/seal.get('prediction_filename','PREDICTIONS.jsonl.gz')
    if sha(prediction_path)!=seal['predictions_sha256']:
        raise ValueError('Prediction journal bytes changed')
    ledger=list(rows(prediction_path))
    ids=[r['id'] for r in ledger]
    if (len(ids)!=238 or len(set(ids))!=238 or ids!=seal['ordered_ids']
            or Counter(r['partition'] for r in ledger)!=Counter(CAL=46,TEST=49,REMAINDER=143)):
        raise ValueError('Exact unique ordered CAL46/TEST49/REMAINDER143 predictions required')
    inventory=set(ledger[0]['predictions'])
    if any(set(r['predictions'])!=inventory for r in ledger):
        raise ValueError('A method cannot silently disappear on failing frames')
    if set(seal.get('arms',inventory))!=inventory:
        raise ValueError('Prediction arm inventory differs from its seal')
    return ledger,seal,prediction_path


def _load_old_after_seal(baseline_folder,baseline_comparison_folder,ids):
    baseline_folder,comparison=Path(baseline_folder),Path(baseline_comparison_folder)
    oldseal=read(comparison/'EVALUATION_SEAL.json')
    pose_path=comparison/'FAIR_SCORED_ROWS.json'
    if sha(pose_path)!=oldseal['artifacts']['FAIR_SCORED_ROWS.json']:
        raise ValueError('Old six-arm poses changed')
    refpath=baseline_folder/'FIXED_REFERENCES.json'
    source_seal=read(baseline_folder/'SCORE_SEAL.json')
    reference_sha=sha(refpath)
    if (reference_sha!=source_seal['artifacts']['FIXED_REFERENCES.json']
            or reference_sha!=oldseal['input_bindings']['original']['FIXED_REFERENCES.json']):
        raise ValueError('Frozen physical/C2 reference changed')
    # No reference document is decoded until verify_new_seal has succeeded.
    old=read(pose_path)['rows'];references=read(refpath)['references']
    if len(old)!=238 or {r['id'] for r in old}!=set(ids):
        raise ValueError('Old and new exact population IDs differ')
    return {r['id']:r for r in old},references,dict(old_evaluation_seal=sha(comparison/'EVALUATION_SEAL.json'),
        old_six_pose_ledger=sha(pose_path),fixed_references=reference_sha,old_reference_seal=sha(baseline_folder/'SCORE_SEAL.json'))


def _candidate_pose(candidate):
    if isinstance(candidate.get('actual_pose'),dict):
        return candidate['actual_pose']
    if 'R_physical' in candidate and 'centroid' in candidate:
        return dict(available=True,R_physical=candidate['R_physical'],centroid=candidate['centroid'])
    pose=(candidate.get('solve') or {}).get('actual_pose')
    return pose if isinstance(pose,dict) else dict(available=False)


def _bank_success(bank,reference):
    successful=[];errors=[]
    for candidate in bank:
        m=_safe_metric(_candidate_pose(candidate),reference)
        if m['success']:successful.append(candidate.get('candidate_id','UNRECORDED'))
        if m['metric_error']:errors.append(dict(candidate_id=candidate.get('candidate_id'),error=m['metric_error']))
    return dict(candidate_count=len(bank),has_success=bool(successful),success_ids=successful,
                invalid_or_unavailable_candidate_count=sum(not _safe_metric(_candidate_pose(c),reference)['pose_available'] for c in bank),
                metric_errors=errors)


def _eligible_family_winners(bank,scores,mode):
    eligible=set();winners={}
    for i,(candidate,scored) in enumerate(zip(bank,scores,strict=True)):
        for family,value in scored.items():
            if value.get('feasible',{}).get(mode) and np.isfinite(value.get('scores',{}).get(mode,float('nan'))):
                eligible.add(i)
                key=(value['scores'][mode],i)
                if family not in winners or key<winners[family][0]:winners[family]=(key,i)
    return [bank[i] for i in sorted(eligible)],{family:bank[index] for family,(_,index) in winners.items()}


def stage_diagnosis(row,reference,arms,baseline_poses=None):
    carried=row.get('carried',[])
    carry=_bank_success(carried,reference)
    result={'carried':carry,'arms':{},'new_attempt_count':len(row.get('attempts',[])),
            'original_candidate_archive_counts':row.get('original_candidate_archive_counts',row.get('candidate_archive_counts')),
            'carried_unique_count':len(carried)}
    for arm in arms:
        final=_safe_metric(row['predictions'][arm]['actual_pose'],reference)
        if arm=='CTRL_RANSAC':
            result['arms'][arm]=dict(status='ARCHIVE_GAP',reason='Only selected RANSAC pose and attempt metadata archived; all R/t roots unavailable',
                candidate_oracle_available=False,family_intermediate_status='ARCHIVE_GAP',
                ransac_attempt_metadata=row['predictions'][arm].get('ransac_attempts',[]),
                final_success=final['success'],final_candidate_id=row['predictions'][arm].get('selected_candidate_id'))
            continue
        selection=(row.get('selection_banks') or {}).get(arm)
        if selection is not None:
            pool=selection['candidates']
            eligible_ids=set(selection.get('eligible_ids',[c['candidate_id'] for c in pool]))
            if eligible_ids-{c['candidate_id'] for c in pool}:
                raise ValueError('Selected learned eligibility refers to an absent candidate')
            eligible=[c for c in pool if c['candidate_id'] in eligible_ids]
            generated=_bank_success(pool,reference);admissible=_bank_success(eligible,reference)
            loss=('FINAL_SUCCESS' if final['success'] else 'NO_SUCCESS_IN_LEARNED_POOL' if not generated['has_success']
                  else 'LEARNED_ELIGIBILITY_FILTER_LOSS' if not admissible['has_success']
                  else 'LEARNED_SELECTION_OR_ABSTENTION_LOSS')
            result['arms'][arm]=dict(status='DIAGNOSED_POSTSEAL_ONLY',candidate_oracle_available=True,
                mode='LEARNED_FINITE_CANDIDATE_SELECTOR',generated_before_feasibility=generated,
                after_constraints=admissible,retained_bank=generated,
                eligibility_is_not_new_geometric_constraint=True,
                family_intermediate_status='NOT_APPLICABLE_DIRECT_SOFTMAX_SELECTION',
                final_success=final['success'],final_candidate_id=row['predictions'][arm].get('selected_candidate_id'),
                stage_loss_class=loss)
            continue
        if arm=='CTRL_FIXED':
            attempts=[c for c in row.get('attempts',[]) if (c.get('solve') or {}).get('start_id')=='N0_LOCAL_START']
            bank=[c for c in attempts if (c.get('solve') or {}).get('feasible') is True]
            n0=(baseline_poses or {}).get('N0') or (row.get('features') or {}).get('fallback')
            if n0 is None:
                n0=next((_candidate_pose(c) for c in carried if c.get('candidate_id')=='N0'),None)
            initial=[] if n0 is None else [dict(candidate_id='N0_INITIAL_OR_FALLBACK',actual_pose=n0)]
            generated=_bank_success(initial+attempts,reference);admissible=_bank_success(bank,reference)
            loss=('FINAL_SUCCESS' if final['success'] else 'NO_SUCCESS_IN_N0_AND_TWO_FIXED_START_ATTEMPTS'
                  if not generated['has_success'] else 'SUCCESS_LOST_BEFORE_ADMISSIBLE_FIXED_TWO_START_BANK'
                  if not admissible['has_success'] else 'FIXED_SELECTION_LOSS_FAMILY_ARCHIVE_GAP')
            result['arms'][arm]=dict(status='DIAGNOSED_POSTSEAL_ONLY',mode='fixed',candidate_oracle_available=True,
                exact_scope='two N0_LOCAL_START family solves, plus original N0 initialization/fallback for pre-admission oracle; excludes CTRL_BUDGET yaw bank',
                generated_before_feasibility=generated,after_constraints=admissible,retained_bank=admissible,
                family_intermediate_status='ARCHIVE_GAP_CROSS_FAMILY_FIXED_SCORES_NOT_SAVED',
                original_N0_available_for_scope=n0 is not None,new_attempt_count=len(attempts),
                final_success=final['success'],final_candidate_id=row['predictions'][arm].get('selected_candidate_id'),
                stage_loss_class=loss)
            continue
        mode=ARM_MODE.get(arm)
        if mode is None:
            result['arms'][arm]=dict(status='NO_METHOD_BANK_MAP',reason='learned selector/bank association supplied separately',
                                     final_success=_safe_metric(row['predictions'][arm]['actual_pose'],reference)['success'])
            continue
        fixed=arm in ('CTRL_CARRY','A_FIXED_BANK','B_FIXED_BANK')
        bank=carried if fixed else row.get('banks',{}).get(mode)
        scores=row.get('carried_scores',[]) if fixed else row.get('bank_scores',{}).get(mode)
        if bank is None or scores is None:
            result['arms'][arm]=dict(status='BANK_NOT_PRODUCED',mode=mode)
            continue
        if len(bank)!=len(scores):raise ValueError('Candidate bank/score alignment changed')
        attempts=[] if fixed else [c for c in row.get('attempts',[]) if (c.get('solve') or {}).get('mode')==mode
                                  or c.get('source')=='NEW_'+mode]
        raw_generated=carried+attempts
        generated=_bank_success(raw_generated,reference)
        eligible,winners=_eligible_family_winners(bank,scores,mode)
        admissible=_bank_success(eligible,reference)
        winner_success={family:_safe_metric(_candidate_pose(c),reference)['success'] for family,c in winners.items()}
        if final['success']:failure='FINAL_SUCCESS'
        elif not generated['has_success']:failure='NO_SUCCESS_IN_CARRIED_OR_FINITE_NEW_ATTEMPTS'
        elif not admissible['has_success']:failure='SUCCESS_LOST_BEFORE_ADMISSIBLE_BANK_OR_BY_CONSTRAINTS'
        elif not any(winner_success.values()):failure='WITHIN_FAMILY_SELECTION_LOSS'
        else:failure='FAMILY_OR_FINAL_SELECTION_LOSS'
        result['arms'][arm]=dict(status='DIAGNOSED_POSTSEAL_ONLY',mode=mode,fixed_bank_only=fixed,candidate_oracle_available=True,
            generated_before_feasibility=generated,retained_bank=_bank_success(bank,reference),
            after_constraints=admissible,family_winner_ids={f:c.get('candidate_id') for f,c in winners.items()},
            family_winner_success=winner_success,any_family_winner_success=any(winner_success.values()),
            final_success=final['success'],final_candidate_id=row['predictions'][arm].get('selected_candidate_id'),
            stage_loss_class=failure,optimizer_reported_failed_attempts=sum((c.get('solve') or {}).get('optimizer_success') is False for c in attempts),
            exception_attempts=sum('exception' in (c.get('solve') or {}) for c in attempts),
            infeasible_attempts=sum((c.get('solve') or {}).get('feasible') is False for c in attempts),
            success_created_beyond_carried=generated['has_success'] and not carry['has_success'])
    return result


def _both_failure_deltas(rows,before,after):
    values={}
    selected=[r for r in rows if not r['metrics'][before]['success'] and not r['metrics'][after]['success']]
    for field in ('translation_cm','rotation_deg'):
        details=[]
        for row in selected:
            a,b=row['metrics'][before][field],row['metrics'][after][field]
            if a is not None and b is not None:details.append(dict(id=row['id'],before=a,after=b,delta=b-a))
        values[field]=dict(common_available_n=len(details),both_failure_full_n=len(selected),
            improved_n=sum(v['delta']<0 for v in details),worsened_n=sum(v['delta']>0 for v in details),
            tied_n=sum(v['delta']==0 for v in details),improvement_sum=sum(v['delta'] for v in details if v['delta']<0),
            worsening_sum=sum(v['delta'] for v in details if v['delta']>0),
            largest_worsenings=sorted([v for v in details if v['delta']>0],key=lambda v:(-v['delta'],v['id'])),
            all_common_available_deltas=details)
    return values


def _unexecuted_arms(registry,executed,execution_status=None):
    if not registry:return {}
    arms=registry.get('arms',registry)
    if isinstance(arms,list):arms={r.get('arm',r.get('name')):r for r in arms}
    result={}
    for arm,value in arms.items():
        if arm in executed or arm in BASELINES or not isinstance(value,dict):continue
        actual=(execution_status or {}).get(arm,{})
        if isinstance(actual,str):actual={'status':actual}
        if actual.get('status') in ('EXECUTED','DONE'):
            raise ValueError('Executed arm is absent from sealed predictions: '+arm)
        result[arm]={**value,**actual}
        if result[arm].get('status') in (None,'REGISTERED'):
            result[arm]['status']='NOT_EXECUTED'
        result[arm].setdefault('reason','No sealed numerical output produced; full-denominator result N/A')
    return result


def _same_pose(a,b):
    if not a.get('available') or not b.get('available'):return False
    return bool(np.max(np.abs(np.asarray(a['centroid'])-b['centroid']))<=1e-7
                and np.max(np.abs(np.asarray(a['R_physical'])-b['R_physical']))<=1e-7)


def _selected_treatment(row,arm):
    entry=row['predictions'][arm];mode=ARM_MODE.get(arm)
    applied=entry.get('boundary_applied')
    result=dict(B_treatment_active=applied if type(applied) is bool else None,A_owner_entropy=None,A_none_mean=None)
    if arm in ('CTRL_FIXED','CTRL_RANSAC'):
        result['B_treatment_active']=False
        return result
    if mode is None:return result
    fixed=arm in ('CTRL_CARRY','A_FIXED_BANK','B_FIXED_BANK')
    bank=row.get('carried',[]) if fixed else row.get('banks',{}).get(mode,[])
    scores=row.get('carried_scores',[]) if fixed else row.get('bank_scores',{}).get(mode,[])
    index=next((i for i,c in enumerate(bank) if c.get('candidate_id')==entry.get('selected_candidate_id')),None)
    family=entry.get('selected_family')
    if index is None or family is None or index>=len(scores):return result
    detail=scores[index].get(family,{}).get('audits',{}).get(mode,{})
    if type(applied) is not bool:
        result['B_treatment_active']=detail.get('B_supported') if mode in ('B','AB','edge') else False
    owner=detail.get('owner',{})
    weights=np.asarray(owner.get('global_owner_responsibilities',[]),float)
    if len(weights):
        result['A_owner_entropy']=float(-sum(v*math.log(v) for v in weights if v>0))
        none=np.asarray(owner.get('per_h_none_responsibilities',[]),float)
        if none.ndim==2 and none.shape[0]==len(weights) and none.shape[1]>0:
            result['A_none_mean']=float(weights@none.mean(axis=1))
    return result


def run(run_dir,*,baseline_folder,baseline_comparison_folder,output=None,scoring_GO=False,
        expected_plan_sha256=None,expected_prediction_seal_sha256=None):
    if scoring_GO is not True:
        raise PermissionError('Root explicit final-seal scoring GO is required')
    start,cpu=time.monotonic(),time.process_time();run_dir=Path(run_dir)
    ledger,seal,prediction_path=verify_new_seal(run_dir,expected_plan_sha256=expected_plan_sha256,
                                            expected_prediction_seal_sha256=expected_prediction_seal_sha256)
    old,references,bindings=_load_old_after_seal(baseline_folder,baseline_comparison_folder,[r['id'] for r in ledger])
    new_arms=tuple(ledger[0]['predictions']);arms=BASELINES+new_arms
    if set(BASELINES)&set(new_arms):raise ValueError('New methods cannot overwrite old arms')
    scored=[];diagnoses=[]
    old_maximum=0.
    for new in ledger:
        source=old[new['id']]
        if any(new[k]!=source[k] for k in ('partition','session','cluster_id')):
            raise ValueError('Original partition/session/cluster changed')
        reference=references[new['partition']][new['id']]
        value={k:new[k] for k in ('id','partition','session','cluster_id')}
        value.update(metrics={},status={},poses={})
        for arm in BASELINES:
            pose=source['poses'][arm]
            m=_safe_metric(pose,reference)
            if m['success']!=source['metrics'][arm]['success']:raise ValueError('Original success policy changed')
            for key in ('translation_cm','rotation_deg'):
                old_maximum=max(old_maximum,abs(m[key]-source['metrics'][arm][key]))
            value['metrics'][arm]=m;value['status'][arm]=source['status'][arm];value['poses'][arm]=pose
        for arm,entry in new['predictions'].items():
            value['metrics'][arm]=_safe_metric(entry['actual_pose'],reference)
            value['status'][arm]=_status(entry);value['poses'][arm]=entry['actual_pose']
        value['runtime_diagnostics']={arm:{**_selected_treatment(new,arm),
            'pose_unchanged_from_N0':_same_pose(value['poses'][arm],value['poses']['N0']),
            'pose_unchanged_from_E0':_same_pose(value['poses'][arm],value['poses']['E0'])} for arm in new_arms}
        scored.append(value)
        diagnoses.append({**{k:new[k] for k in ('id','partition','session','cluster_id')},
                           **stage_diagnosis(new,reference,new_arms,baseline_poses=source['poses'])})
    subsets={name:[r for r in scored if r['partition'] in parts] for name,parts in POPULATIONS.items()}
    if {k:len(v) for k,v in subsets.items()}!=EXPECTED_N:raise ValueError('Exact population denominators changed')
    summaries={name:aggregate(value,arms) for name,value in subsets.items()}
    for name,subset in subsets.items():
        for arm in arms:
            summaries[name][arm]['metric_error_ids']=[r['id'] for r in subset if r['metrics'][arm]['metric_error']]
            summaries[name][arm]['metric_error_n']=len(summaries[name][arm]['metric_error_ids'])
            if arm in new_arms:
                summaries[name][arm]['application_diagnostics']=dict(
                    B_active_n=sum(r['runtime_diagnostics'][arm]['B_treatment_active'] is True for r in subset),
                    B_unsupported_n=sum(r['runtime_diagnostics'][arm]['B_treatment_active'] is False for r in subset),
                    B_application_unknown_n=sum(r['runtime_diagnostics'][arm]['B_treatment_active'] is None for r in subset),
                    unchanged_from_N0_n=sum(r['runtime_diagnostics'][arm]['pose_unchanged_from_N0'] for r in subset),
                    unchanged_from_E0_n=sum(r['runtime_diagnostics'][arm]['pose_unchanged_from_E0'] for r in subset),
                    success_with_B_unapplied_n=sum(r['metrics'][arm]['success'] and
                        r['runtime_diagnostics'][arm]['B_treatment_active'] is False for r in subset))
    contrasts=[(before,after) for after in new_arms for before in COMPARATORS]
    comparisons={name:{after+'_minus_'+before:{**paired(value,before,after),
        'both_failure_continuous':_both_failure_deltas(value,before,after)} for before,after in contrasts}
        for name,value in subsets.items()}
    registry=read(run_dir/'ARM_REGISTRY.json') if (run_dir/'ARM_REGISTRY.json').exists() else {}
    execution_status=read(run_dir/'ARM_EXECUTION_STATUS.json') if (run_dir/'ARM_EXECUTION_STATUS.json').exists() else {}
    unavailable=_unexecuted_arms(registry,new_arms,execution_status)
    output=Path(output or run_dir/'evaluation_v1');output.mkdir(parents=True,exist_ok=False)
    comparison_rows=[];metric_rows=[];transition_rows=[]
    for name,methods in summaries.items():
        for arm,s in methods.items():
            row=dict(population=name,arm=arm,execution_status='EXECUTED',denominator=s['denominator'],
                successes=s['successes'],success_rate=s['success_rate'],no_pose=s['no_pose'],
                metric_error_n=s['metric_error_n'],
                T_gt50cm=s['T_gt50cm'],T_gt100cm=s['T_gt100cm'],R_gt45deg=s['R_gt45deg'])
            for metric_name in ('translation_cm','rotation_deg'):
                row.update({metric_name+'_'+key:s[metric_name][key] for key in
                    ('n','mean','sample_std','sample_variance','median','P90','P95','max')})
            for key in ('accepted','fallback','technical_failure','solver_rejected','timeout'):
                row.update({key+'_'+k:v for k,v in s['status_counts'][key].items()})
            comparison_rows.append(row)
        for arm,reason in unavailable.items():
            comparison_rows.append(dict(population=name,arm=arm,execution_status=reason.get('status','NOT_EXECUTED'),
                                        denominator=EXPECTED_N[name],successes=None,success_rate=None,
                                        not_executed_reason=reason.get('reason',str(reason))))
    for row in scored:
        identity={k:row[k] for k in ('id','partition','session','cluster_id')}
        for arm in arms:metric_rows.append({**identity,'arm':arm,**row['metrics'][arm],**row['status'][arm],
                                          **row['runtime_diagnostics'].get(arm,{})})
        for before,after in contrasts:
            a,b=row['metrics'][before],row['metrics'][after]
            transition=('BOTH_SUCCESS' if a['success'] and b['success'] else 'RECOVERY' if b['success']
                        else 'DAMAGE' if a['success'] else 'BOTH_FAILURE')
            transition_rows.append(dict(**identity,before=before,after=after,transition=transition,
                before_T_cm=a['translation_cm'],after_T_cm=b['translation_cm'],
                before_R_deg=a['rotation_deg'],after_R_deg=b['rotation_deg']))
    _csv(output/'COMPARISON_ALL_ARMS.csv',comparison_rows)
    _csv(output/'ROW_METRICS.csv',metric_rows)
    _csv(output/'PAIRED_TRANSITIONS.csv',transition_rows)
    stage_summary={}
    for name,parts in POPULATIONS.items():
        ds=[r for r in diagnoses if r['partition'] in parts]
        archive_counts=[r['original_candidate_archive_counts'] for r in ds if isinstance(r['original_candidate_archive_counts'],dict)]
        count_keys=sorted({k for item in archive_counts for k,v in item.items() if type(v) in (int,float)})
        stage_summary[name]=dict(denominator=len(ds),carried_oracle_successes=sum(r['carried']['has_success'] for r in ds),
            original_archive_count_known_frames=len(archive_counts),
            original_archive_vs_positive_depth_admission_totals={k:sum(v.get(k,0) for v in archive_counts) for k in count_keys},arms={})
        for arm in new_arms:
            values=[r['arms'][arm] for r in ds]
            generated_known=[v for v in values if 'generated_before_feasibility' in v]
            eligible_known=[v for v in values if 'after_constraints' in v]
            family_known=[v for v in values if 'any_family_winner_success' in v]
            stage_summary[name]['arms'][arm]=dict(status_counts=dict(Counter(v['status'] for v in values)),
                stage_loss_class_counts=dict(Counter(v.get('stage_loss_class','NO_BANK_MAP') for v in values)),
                generated_oracle_known_frames=len(generated_known),after_constraints_oracle_known_frames=len(eligible_known),
                family_intermediate_known_frames=len(family_known),
                family_intermediate_status_counts=dict(Counter(v.get('family_intermediate_status','APPLICABLE') for v in values)),
                generated_before_constraints_oracle_successes=sum(v['generated_before_feasibility']['has_success'] for v in generated_known) if generated_known else None,
                after_constraints_oracle_successes=sum(v['after_constraints']['has_success'] for v in eligible_known) if eligible_known else None,
                any_family_winner_successes=sum(v['any_family_winner_success'] for v in family_known) if family_known else None,
                final_successes=sum(v.get('final_success',False) for v in values))
    write(output/'CANDIDATE_STAGE_DIAGNOSIS.json',dict(schema='POSTSEAL_CANDIDATE_STAGE_DIAGNOSIS_V1',
        diagnostic_labels_never_in_prediction_inputs=True,populations=stage_summary,rows=diagnoses))
    save_rows(output/'SCORED_ROWS.jsonl.gz',scored)
    result=dict(schema='STATEWISE_RECOVERY_EXPLORATORY_EVALUATION_V1',arms=arms,new_arms=new_arms,
        populations=summaries,paired=comparisons,unexecuted_arms=unavailable,
        session_summaries={name:{session:aggregate([r for r in value if r['session']==session],arms)
            for session in sorted({r['session'] for r in value})} for name,value in subsets.items()},
        baseline_metric_max_absolute_difference=old_maximum,
        fixed_reference_policy='same derived physical/C2, never rectangular90; not independent physical GT',
        exploratory_development_evaluation=True,all_frames_and_failures_kept=True,
        bootstrap_policy=dict(seed=20260917,draws=10000,primary_unit='original recording cluster_id',
                              sensitivity_unit='named session',not_independent_or_equivalence=True),
        input_bindings={**bindings,'new_predictions':sha(prediction_path),'new_prediction_seal':sha(run_dir/'PREDICTION_SEAL.json'),
                        'plan_lock':sha(run_dir/'PLAN_LOCK.json')},
        elapsed_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(output/'RESULTS.json',result)
    write(output/'EVALUATION_SEAL.json',dict(schema='STATEWISE_RECOVERY_EVALUATION_SEAL_V1',
        prediction_rows=238,metric_rows=238*len(arms),explicit_root_scoring_GO=True,
        input_bindings=result['input_bindings'],artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        elapsed_seconds=result['elapsed_seconds']))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--baseline-folder',required=True)
    p.add_argument('--baseline-comparison-folder',required=True);p.add_argument('--output')
    p.add_argument('--scoring-GO',action='store_true');p.add_argument('--expected-plan-sha256')
    p.add_argument('--expected-prediction-seal-sha256');a=p.parse_args()
    r=run(a.run_dir,baseline_folder=a.baseline_folder,baseline_comparison_folder=a.baseline_comparison_folder,
        output=a.output,scoring_GO=a.scoring_GO,expected_plan_sha256=a.expected_plan_sha256,
        expected_prediction_seal_sha256=a.expected_prediction_seal_sha256)
    print({arm:v['successes'] for arm,v in r['populations']['PRIMARY192'].items()})


if __name__=='__main__':main()
