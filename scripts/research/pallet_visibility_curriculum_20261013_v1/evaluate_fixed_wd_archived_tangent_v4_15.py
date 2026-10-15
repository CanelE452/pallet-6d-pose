"""Frozen correspondence plus uncertain archived tangent controls, whole238.

Compare exact e baseline, typed-only variance4 and typed+legacy variance4.
One already inferred W/D family is used throughout; no new family/P1 search,
RGB/model forward, calibration or GT/ID parameter routing occurs. Existing
WD0 samples/P1 seed and G10/D winners initialize the same-family refinement.
All238 predictions seal before each new control's posthoc reference metrics.
Both oldTEST49 and remaining143 are reused exploratory data, not holdouts.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from copy import deepcopy
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import argparse
import shutil
import time
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_face_partial_BI_outline_v4_13d as D
from . import evaluate_remainder_frozen_wd_v4_14 as R
from . import select_frozen_wd_partial_refinement_v4_13e as SELECTOR
from . import statewise_archived_BI_tangent_solver_v4_15 as T
from . import statewise_archived_BI_tangent_outline_solver_v4_15 as G
from . import evaluate_global_outline_pose_v4_10 as G10
from . import evaluate_two_wd_families_v4_9 as W
from .statewise_uncertainty_calibration_v4 import load_uncertainty_model
from .scoring import pose_metric,distribution
from . import score_remainder_derived_reference_v4_14 as REF

OUTPUT=G10.OUTPUT.parent/'fixed_wd_archived_tangent_eval_v4_15'
ARMS=('SAME_FROZEN_E','T_TYPED_VARIANCE4','T_ALL_ARCHIVED_VARIANCE4')
MODEL=E.FIX/'calibration/final_adapter_CAL46/MODEL.json'


def _checked(folder,count):
    protocol=C.read(folder/'PROTOCOL.json');seal=C.read(folder/'PREDICTION_SEAL.json');path=folder/'PREDICTIONS.jsonl.gz'
    if (seal['rows']!=count or C.sha(path)!=seal['predictions_sha256']
            or C.sha(folder/'PROTOCOL.json')!=seal['protocol_sha256']):raise ValueError('Complete immutable parent prediction seal required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    rows=list(C.rows(path))
    if len(rows)!=count or [r['id']for r in rows]!=protocol['ids']:raise ValueError('Exact parent population/order required')
    return rows,protocol


def _task(row,old,partial,baseline,wd,prior,config,mask_root):
    name=old['selected_family'];family=deepcopy(next(f for f in partial['arms'][D.ARMS[1]]['families']if f['name']==name))
    if baseline['WD_family']!=name:raise ValueError('Fixed old/e family binding required')
    merged=partial['merged_partial_observation'];finish=deepcopy(row)
    finish['partial_boundary_observation']=deepcopy(merged)
    spec=E._physics_spec(row['_original_source'],row['frozen_BI_fit'],config)
    spec.update(xyz=family['native_xyz'],coarse_pose=family['coarse_pose'])
    spec['config']=deepcopy(config);spec['config']['prior_weight']=0.
    records=deepcopy(merged['boundary_observations'])
    all_records=T.archive_legacy_B_tangent_predictions(records,spec['visible'],spec['states'],spec['config'])
    for rr in (records,all_records):
        normalized=T._boundary_inputs(rr,spec['states'],spec['image_hw'],spec['xyz'])
        T.validate_archived_predictions(normalized,spec['visible'],spec['config'])
    record=row['_MASK'];mask_path=mask_root/record['mask_npz']
    if C.sha(mask_path)!=record['mask_file_sha256'] or record['image_sha256']!=row['image_sha256']:
        raise ValueError('Exact frozen current mask/image binding required')
    with np.load(mask_path)as arrays:mask=arrays['pallet_mask_clean'].copy()
    outline=G.observe_global_outline(mask,spec['image_hw'],image_binding=dict(image_sha256=row['image_sha256'],
        mask_file_sha256=record['mask_file_sha256'],mask_npz=record['mask_npz']))
    if {k:v for k,v in outline.items()if k!='observed_boundary_samples_xy'}!=old['mask_observation']:
        raise ValueError('Same frozen global outline construction required')
    # A read-only bank also permits a separately versioned later family
    # experiment; this driver still refines only its already fixed family.
    bank={}
    for ff in partial['arms'][D.ARMS[1]]['families']:
        wd_family=next(f for f in wd['families']if f['name']==ff['name'])
        old_family=next(f for f in old['families']if f['name']==ff['name'])
        for f in (wd_family,old_family):
            for key in ('native_xyz','cf_extents','physical_registration_Q'):
                if not np.allclose(f[key],ff[key],rtol=0,atol=1e-12):raise ValueError('Same family model/Q required')
        initials=[deepcopy(ff['solution']),deepcopy(wd_family['solution']),deepcopy(old_family['solution'])]
        for index in (0,2):initials[index]['feasible_pose_samples']=[];initials[index]['pose_family_candidates']=[]
        # Immutable G10/G13 included P1's complete native sample set only in
        # ORIGINAL_WD. Retain that set and its own acceptance/fallback flags.
        if ff['name']=='ORIGINAL_WD':initials.append(deepcopy(prior['solution']))
        Q=np.asarray(ff['physical_registration_Q']);pR=np.asarray(prior['actual_pose']['R_physical'])
        starts=[dict(R=(pR@Q.T).tolist(),t=prior['actual_pose']['centroid'],
            name='SAME_EXISTING_P1_ACCEPTED_OR_OWN_FALLBACK_PHYSICAL_SEED_ONLY')]
        bank[ff['name']]=dict(family={k:deepcopy(ff[k])for k in
            ('name','native_xyz','cf_extents','physical_registration_Q','coarse_pose')},
            initial_solutions=initials,initializations=starts)
    finish={k:finish[k]for k in ('_original_source','frozen_BI_fit','partial_boundary_observation','u_region_observation')}
    meta={k:deepcopy(family[k])for k in ('name','native_xyz','cf_extents','physical_registration_Q','coarse_pose')}
    return C.finite(dict(id=row['id'],partition=row['partition'],session=row['session'],image=row['image'],
        image_sha256=row['image_sha256'],fixed_family=name,baseline=baseline,spec=spec,finish_input=finish,
        family=meta,boundary_typed=records,boundary_all=all_records,u_regions=row['u_region_observation']['u_regions'],
        outline=outline,initial_solutions=bank[name]['initial_solutions'],initializations=bank[name]['initializations'],
        boundary_original_legacy=deepcopy(old['partial_boundary_observation']['boundary_observations']),
        stage1_family_seed_bank=bank,
        mask_binding=E._binding(mask_path)))


def _load():
    rows,model,config,bindings,parent,old_by,locality=D._load()
    old_e,ep=_checked(SELECTOR.OUTPUT,95);old_d,dp=_checked(D.OUTPUT,95)
    fresh,rp=_checked(R.OUTPUT,143)
    from .audit_remainder_pipeline_v4_14 import verify_source_closure
    closure=verify_source_closure();sources={**ep['source_sha256'],**dp['source_sha256'],**rp['source_sha256'],
        **closure['conservatively_bound_source_union_sha256']};inputs={**bindings,**ep['input_sha256'],**dp['input_sha256'],**rp['input_sha256']}
    protected=parent['protected_user_sha256']
    if any(p['protected_user_sha256']!=protected for p in (ep,dp,rp)):raise ValueError('Protected user6 binding mismatch')
    d_by={v['id']:v for v in old_d};e_by={v['id']:v for v in old_e};tasks=[]
    for row in rows:
        tasks.append(_task(row,old_by[row['id']],d_by[row['id']],e_by[row['id']]['arms'][SELECTOR.ARMS[2]],
            row['_WD'],row['_cached_V48']['arms'][W.PREVIOUS.ARMS[0]],config,G10.MASKS))
    observed={v['id']:v for v in C.rows(R.INPUTS/'OBSERVATIONS.jsonl.gz')}
    masks={v['id']:v for v in C.read(R.INPUTS/'masks/RECORDS.json')}
    fresh_config=C.read(R.INPUTS/'PROTOCOL.json')['solver_configuration']
    for p in fresh:
        row=observed[p['id']];row['_MASK']=masks[p['id']]
        prior_arm=W.V47._finish(row['_original_source'],row['frozen_BI_fit'],model,p['full_prior_seed'],
            row['u_region_observation']['u_regions'],'EXISTING_P1_ACCEPTED_OR_OWN_FALLBACK_SEED',row['partial_boundary_observation'])
        tasks.append(_task(row,p['old_global_outline'],p['partial_global_outline'],p['arms'][SELECTOR.ARMS[2]],
            p['full_WD_search'],prior_arm,fresh_config,R.INPUTS/'masks'))
    if len(tasks)!=238 or len({t['id']for t in tasks})!=238 or Counter(t['partition']for t in tasks)!=Counter(CAL=46,TEST=49,REMAINDER=143):
        raise ValueError('Keep every original95 and remaining143; severe0')
    for folder in (SELECTOR.OUTPUT,D.OUTPUT,R.OUTPUT):
        for name in ('PROTOCOL.json','PREDICTION_SEAL.json','PREDICTIONS.jsonl.gz'):
            path=folder/name;inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    for path in (MODEL,R.INPUTS/'OBSERVATION_SEAL.json',R.INPUTS/'OBSERVATIONS.jsonl.gz',
                 R.INPUTS.parent/'independent_pipeline_audit_v4_14/ADDITIONAL_EXECUTED_SOURCE_BINDINGS_V1.json',
                 E.COHORT/'OOF_pose_v1/ISOLATED_CAL_SCORING.json',E.COHORT/'test_reused_v1/ISOLATED_TEST_SCORING.json',
                 REF.OUTPUT/'PLAN.json',REF.OUTPUT/'PREPARATION_SEAL.json'):
        inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    # Pin the already-fixed reference policy and annotation bytes, without
    # decoding numeric targets in prediction/input preparation.
    inputs.update(C.read(REF.OUTPUT/'PLAN.json')['input_sha256'])
    for task in tasks:inputs[task['mask_binding']['path']]=task['mask_binding']['sha256']
    for module in (__file__,T.__file__,G.__file__,T.T.__file__,T.P.__file__,G.OLD.__file__,G.OLD.OLD.__file__,REF.__file__):
        path=Path(module);sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    return tasks,model,sources,inputs,protected


def _preserve_frozen_family_uncertainty(arm,baseline):
    """A same-family local fit does not resolve an unsearched W/D branch."""
    hierarchy=deepcopy(baseline['hierarchical_correspondence_audit'])
    initial=bool(hierarchy['initial_family_ambiguity'].get('detected'))
    hierarchy['refined_within_family_ambiguity']=deepcopy(arm['ambiguity_audit'])
    hierarchy.update(initial_family_or_refined_pose_globally_unique_proved=False,
        archived_prediction_tangent_refined_only_the_already_fixed_family=True,
        historical_D_cross_family_audit_is_not_a_new_control_search=True,
        old_pose_used_as_output=False,refined_d_same_family_numeric_pose_exact=False,
        new_tangent_same_family_numeric_pose=True)
    arm['hierarchical_correspondence_audit']=hierarchy
    if initial or arm['ambiguity_audit'].get('material_ambiguity'):
        arm['ambiguity_audit'].update(material_ambiguity=True,frozen_family_choice_does_not_prove_uniqueness=True)
        arm['solution'].setdefault('ambiguity',{}).update(detected=True,initial_family_ambiguity_preserved=initial)
        arm['solution'].update(OT_solved=False,OT_unique_solution_claimed=False)
        readout=arm['keypoint_readout'];readout.update(OT_solved=False,OT_unique_solution_claimed=False)
        for k,state in enumerate(arm['states']):
            if state in ('O','T') and readout['reported_points'][k] is not None:
                readout['point_sources'][k]='AMBIGUOUS_FROZEN_FAMILY_JOINT_PROJECTION'
    return arm


def _run(task,model):
    started=time.monotonic();before=E._digest(task);arms={ARMS[0]:deepcopy(task['baseline'])};audit={}
    for name,key in ((ARMS[1],'boundary_typed'),(ARMS[2],'boundary_all')):
        records=task[key];spec=task['spec'];normalized=T._boundary_inputs(records,spec['states'],spec['image_hw'],spec['xyz'])
        active=T._tangent_records(normalized,4.)
        if not active:
            arms[name]=deepcopy(task['baseline']);audit[name]=dict(exact_baseline_alias=True,active_tangent_ids=[]);continue
        solution=G.refine_global_outline_joint_pose(spec['xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],
            spec['image_hw'],spec['coarse_pose'],config=spec['config'],outline_observation=task['outline'],
            initial_solutions=task['initial_solutions'],initializations=task['initializations'],
            boundary_observations=records,u_regions=task['u_regions'],prediction_tangent_weight=1.,tangent_variance_inflation=4.)
        finish=deepcopy(task['finish_input']);finish['partial_boundary_observation']['boundary_observations']=deepcopy(records)
        arm=W._family_finish(finish,model,task['family'],name,solution)
        readout=arm['keypoint_readout'];readout['partial_BI_observations']=deepcopy(records)
        readout['partial_B_observations']=[deepcopy(b)for b in records if b['state']=='B']
        active_ids={r['native_corner']for r in active}
        for b in records:
            k=b['native_corner'];readout['reported_points'][k]=None
            readout['point_sources'][k]=('PARTIAL_'+b['state']+'_RGB_NORMAL_AND_UNCERTAIN_PREDICTION_TANGENT'
                if k in active_ids else 'PARTIAL_'+b['state']+'_NORMAL_AND_TANGENT_INTERVAL')
        readout['uncertain_archived_tangent_factor']=deepcopy(solution['tangent_prediction_factor_audit'])
        _preserve_frozen_family_uncertainty(arm,task['baseline'])
        arm['solution']=E._compact_solution(arm['solution']);arms[name]=arm
        audit[name]=dict(exact_baseline_alias=False,active_tangent_ids=[r['native_corner']for r in active],
            legacy_tangent_ids=[b['native_corner']for b in normalized if T.ARCHIVE_KEY in b],variance_inflation=4.,std_inflation=2.,outside_weight=1.)
    if E._digest(task)!=before:raise ValueError('Frozen physics/archive input changed')
    return C.finite(dict(id=task['id'],partition=task['partition'],session=task['session'],image=task['image'],
        image_sha256=task['image_sha256'],fixed_family=task['fixed_family'],arms=arms,tangent_control_audit=audit,
        input_sha256=before,elapsed_seconds=time.monotonic()-started,
        contract=dict(no_GT_or_ID_parameter_routing=True,S_U_point_or_line_cost_added=False,
            states_all_OT_U_normal_variance_BI_covariance_reliability_Q_preserved=True,
            same_fixed_family_no_new_WD_or_P1_search=True,no_new_RGB_model_forward_or_uncertainty_fit=True,
            tangent_is_archived_prediction_not_precise_RGB_KP=True,normal_tangent_independence_claimed=False)))


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every previous versioned control')
    tasks,model,sources,inputs,protected=_load()
    for hashes in (sources,inputs,protected):E._verify_sources(hashes)
    output.mkdir(parents=True)
    protocol=dict(schema='FROZEN_WD_ARCHIVED_TANGENT_PAIRED238_V4_15',ids=[r['id']for r in tasks],arms=list(ARMS),
        population=dict(CAL=46,TEST=49,REMAINDER=143,severe=0),source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=protected,model_binding=E._binding(MODEL),fixed_variance_inflation=4.,fixed_std_inflation=2.,
        outside_factor_weight=1.,CAL_reliability_applied_once=True,normal_covariances_and_all_constraints_unchanged=True,
        fixed_geometry_family_from_saved_OLD_G10_not_reselected=True,legacy_prediction_archive_is_not_fake_typed_revocation=True,
        seed_rule='all saved same-family WD0 samples; original-native P1 sample set where originally supplied; existing P1 accepted/own-fallback seed plus G10/D winners only',
        all238_complete_before_new_control_scores=True,no_gt_in_prediction=True,
        hypothesis_motivated_posthoc=True,TEST49_and_REMAINDER143_reused_exploratory=True)
    C.write(output/'PROTOCOL.json',protocol);C.save_rows(output/'INPUTS.jsonl.gz',tasks)
    for relative,digest in sources.items():
        path=output/'code_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,path)
        if C.sha(path)!=digest:raise ValueError('Frozen source snapshot mismatch')
    C.write(output/'INPUT_SEAL.json',dict(rows=238,input_sha256=C.sha(output/'INPUTS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        before_new_pose_fits_and_control_metrics=True,ordered_ids=protocol['ids']))
    return tasks,model


def predict(output=OUTPUT,workers=3):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'INPUT_SEAL.json')
    if (seal['rows']!=238 or C.sha(output/'INPUTS.jsonl.gz')!=seal['input_sha256']
        or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']):raise ValueError('Complete238 input seal required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    if (output/'PREDICTION_SEAL.json').exists():raise FileExistsError('Never overwrite sealed control predictions')
    tasks=list(C.rows(output/'INPUTS.jsonl.gz'));model=load_uncertainty_model(MODEL)
    if [t['id']for t in tasks]!=protocol['ids']:raise ValueError('Exact238 order')
    started=time.monotonic();done={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'))as pool:
        futures={pool.submit(_run,t,model):t['id']for t in tasks}
        for future in as_completed(futures):
            row=future.result();done[row['id']]=row;C.write(output/'frames'/(row['id'].replace(':','__')+'.json'),row)
            C.write(output/'LIVE_PROGRESS.json',dict(completed=len(done),total=238,latest_id=row['id'],elapsed_seconds=time.monotonic()-started))
            print(f'{len(done)}/238 {row["id"]}',flush=True)
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    C.save_rows(output/'PREDICTIONS.jsonl.gz',[done[t['id']]for t in tasks])
    C.write(output/'PREDICTION_SEAL.json',dict(rows=238,CAL=46,TEST=49,REMAINDER=143,severe=0,
        predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_seal_sha256=C.sha(output/'INPUT_SEAL.json'),all3_arms_complete=True,before_new_control_metric_read=True,
        ordered_ids=protocol['ids'],elapsed_seconds=time.monotonic()-started))


def score(output=OUTPUT):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREDICTION_SEAL.json')
    if (seal['rows']!=238 or not seal['all3_arms_complete'] or not seal['before_new_control_metric_read']
            or C.sha(output/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']
            or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']
            or C.sha(output/'INPUT_SEAL.json')!=seal['input_seal_sha256']):raise ValueError('Complete238 allarm seal before metrics')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    rows=list(C.rows(output/'PREDICTIONS.jsonl.gz'))
    if [r['id']for r in rows]!=protocol['ids']:raise ValueError('Every original238 row/order retained')
    reports={};bindings={}
    for partition,count in (('CAL',46),('TEST',49),('REMAINDER',143)):
        population=[r for r in rows if r['partition']==partition]
        if len(population)!=count:raise ValueError('No native/fallback/ambiguity filtering')
        if partition=='REMAINDER':
            plan=C.read(REF.OUTPUT/'PLAN.json');annotations={a['id']:a for a in plan['annotations']}
            references={r['id']:REF.reference_from_document(C.read(C.ROOT/annotations[r['id']]['annotation']))for r in population}
            metric=REF.metric_for_reference;bindings[partition]=E._binding(REF.OUTPUT/'PLAN.json')
        else:
            path=E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json'if partition=='CAL'else'test_reused_v1/ISOLATED_TEST_SCORING.json')
            labels=C.read(path)['rows'];references={r['id']:labels[r['id']]['target']['truth']for r in population}
            metric=pose_metric;bindings[partition]=E._binding(path)
        values=[dict(id=r['id'],fixed_family=r['fixed_family'],metrics={a:metric(r['arms'][a]['actual_pose'],references[r['id']])for a in ARMS},
            accepted={a:r['arms'][a]['accepted']for a in ARMS},tangent_control_audit=r['tangent_control_audit'])for r in population]
        summary={}
        for arm in ARMS:
            translations=[r['metrics'][arm]['translation_cm']for r in values]
            rotations=[r['metrics'][arm]['rotation_deg']for r in values if r['metrics'][arm]['rotation_deg']is not None]
            summary[arm]=dict(translation_cm=distribution(translations),rotation_deg=distribution(rotations),
                translation_le5=sum(v<=5 for v in translations),successes_5cm_5deg=sum(bool(r['metrics'][arm]['success_5cm_5deg'])for r in values),
                rotation_unresolved=count-len(rotations),fallback_count=sum(not r['accepted'][arm]for r in values),
                max_translation_id=values[int(np.argmax(translations))]['id'])
        reports[partition]=dict(rows=values,summary=summary,count=count,reference_binding=bindings[partition],
            all_rows_kept=True,new_parameter_or_source_selection_by_metrics=False,
            TEST49_and_REMAINDER143_reused_exploratory=True,remaining_reference_is_derived_not_independent_metrology=True)
        C.write(output/f'{partition}_SCORE.json',reports[partition]);print(partition,summary,flush=True)
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    C.write(output/'SCORE_SEAL.json',dict(prediction_seal=E._binding(output/'PREDICTION_SEAL.json'),
        score_sha256={p:C.sha(output/f'{p}_SCORE.json')for p in reports},reference_bindings=bindings,
        all_numeric_references_decoded_only_after_new_control238_seal=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=3);a=parser.parse_args()
    if a.command=='prepare':prepare(a.output)
    elif a.command=='predict':predict(a.output,a.workers)
    else:score(a.output)
