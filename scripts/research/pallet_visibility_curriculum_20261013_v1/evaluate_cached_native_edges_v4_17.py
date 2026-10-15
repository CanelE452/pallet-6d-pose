"""Uniform cached-proposal native-edge controls, all original95 + remaining143.

No optimizer, RGB query, network forward or target-dependent proposal filter.
Every archived pose is freshly checked in both registered W/D hypotheses.
Native Pareto/minimum-sum ranking precedes global-mask numerical tie breaking.
The exact e baseline and all own-coarse fallbacks remain in the population.
All238 predictions seal before this control's reused-data reference scores.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import argparse
import hashlib
import shutil
import time
import types
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_fixed_wd_archived_tangent_v4_15 as ARCHIVE
from . import evaluate_two_stage_top_normal_v4_16c as NORMAL
from . import evaluate_global_outline_pose_v4_10 as OLD10
from . import evaluate_face_partial_BI_outline_v4_13d as D
from . import evaluate_remainder_frozen_wd_v4_14 as REMAINDER
from . import hierarchical_native_candidate_score_v4_17 as AUTH
from . import calibrate_native_edge_angle_v4_17 as ANGLE
from . import native_edge_problem_v4_17 as NATIVE
from . import statewise_partial_BI_outline_solver_v4_13 as OUTLINE

OUTPUT=ARCHIVE.OUTPUT.parent/'cached_native_edge_candidates_v4_17'
OLD_BASELINE=D.BASELINE
ARMS=('SAME_FROZEN_E','NATIVE_ORIGINAL_BI_LEGACY','NATIVE_D_MERGED_ALL')
KIND_MAP={'COMPLETE_BI':'BI_COMPLETE','UNREPLACED_PARTIAL_NORMAL':'PARTIAL_NORMAL',
    'PROJECTED_NATIVE_EDGE':'RGB_EDGE','UNCERTAIN_ARCHIVED_TANGENT':'ARCHIVED_TANGENT'}


def _checked(folder,count):
    protocol=C.read(folder/'PROTOCOL.json');seal=C.read(folder/'PREDICTION_SEAL.json')
    if (seal['rows']!=count or C.sha(folder/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']
            or C.sha(folder/'PROTOCOL.json')!=seal['protocol_sha256']):
        raise ValueError('Complete immutable proposal source seal required')
    return protocol,seal


def _frame_digests(folder):
    return {r['id']:E._digest(r)for r in C.rows(folder/'PREDICTIONS.jsonl.gz')}


def _sealed_frame(folder,ident,digests):
    row=C.read(folder/'frames'/(ident.replace(':','__')+'.json'))
    if row['id']!=ident or E._digest(row)!=digests[ident]:
        raise ValueError('Per-frame cached archive differs from sealed whole prediction journal')
    return row


def _valid_pose(R,t):
    try:R=np.asarray(R,float);t=np.asarray(t,float)
    except (TypeError,ValueError):return None
    if R.shape!=(3,3)or t.shape!=(3,)or not NATIVE.V._valid_rotation(R)or not np.isfinite(t).all():return None
    return R,t


def _add_pose(bank,R,t,Q,origin):
    parsed=_valid_pose(R,t)
    if parsed is None:return False
    R,t=parsed;Q=np.asarray(Q,float)
    if Q.shape!=(3,3)or not NATIVE.V._valid_rotation(Q):raise ValueError('Known proper proposal registration Q required')
    physical=R@Q
    key=hashlib.sha256(np.round(np.r_[physical.ravel(),t],12).tobytes()).hexdigest()
    if key in bank:
        bank[key]['origins'].append(origin)
    else:
        bank[key]=dict(R_physical=physical.tolist(),t=t.tolist(),origins=[origin],pose_digest=key)
    return True


def collect_solution_poses(bank,solution,Q,origin):
    """Published pose and all saved endpoint/sample records, no old cost gate.

    Fallback/seed dictionaries nested inside a solution are not recursively
    mined. An old rejected published pose is only a proposal; its status is
    discarded and current hard guards decide admission independently.
    """
    if not isinstance(solution,dict):return dict(saved_records=0,invalid_records=0)
    count=0;invalid=0
    entries=[('published',solution)]
    for key in ('feasible_pose_samples','pose_family_candidates','candidates'):
        values=solution.get(key,[])
        if isinstance(values,list):entries.extend((key+':'+str(i),v)for i,v in enumerate(values))
    for kind,record in entries:
        if not isinstance(record,dict)or 'R'not in record or 't'not in record:continue
        count+=1
        if not _add_pose(bank,record['R'],record['t'],Q,origin+'/'+kind):invalid+=1
    return dict(saved_records=count,invalid_records=invalid)


def collect_proposal_bank(task,old,d,normal,tangent):
    """Use every declared old source equally for every frame, both WD bases."""
    bank={};inventory=[]
    for name,entry in task['stage1_family_seed_bank'].items():
        Q=entry['family']['physical_registration_Q']
        for index,solution in enumerate(entry['initial_solutions']):
            inventory.append(collect_solution_poses(bank,solution,Q,'saved_seed_bank/'+name+'/'+str(index)))
    for stage,outer in (('G10',old),('D',d['arms'][D.ARMS[1]])):
        for family in outer['families']:
            inventory.append(collect_solution_poses(bank,family['solution'],family['physical_registration_Q'],stage+'/'+family['name']))
    for name,audit in normal['normal_two_stage_audit'].items():
        for family in audit['stage1_all_families']:
            inventory.append(collect_solution_poses(bank,family['solution'],family['physical_registration_Q'],'normal_stage1/'+name+'/'+family['name']))
    for stage,outer in (('normal_stage2',normal),('tangent',tangent)):
        for name,arm in outer['arms'].items():
            Q=arm['actual_pose']['physical_registration_Q']
            inventory.append(collect_solution_poses(bank,arm['solution'],Q,stage+'/'+name))
    return list(bank.values()),dict(distinct_physical_poses=len(bank),
        saved_pose_records=sum(v['saved_records']for v in inventory),
        invalid_pose_records=sum(v['invalid_records']for v in inventory),
        no_old_accepted_objective_metric_or_GT_filter=True,
        dedup_rule='physical R/t rounded to12decimal arithmetic precision; provenance union retained',
        old_seed_or_fallback_dictionary_not_mined=True)


def _verified_model():
    path=ANGLE.OUTPUT/'MODEL.json';seal=C.read(ANGLE.OUTPUT/'MODEL_SEAL.json')
    if seal.get('fit_partition')!='CAL'or C.sha(path)!=seal['model_sha256']:
        raise ValueError('Completed fixed CAL-only edge-angle model required')
    for key in ('source_sha256','input_sha256'):E._verify_sources(seal[key])
    model=C.read(path);ANGLE.authentication_angular_parameters(model)
    return model


def _load():
    previous=C.read(ARCHIVE.OUTPUT/'PROTOCOL.json');input_seal=C.read(ARCHIVE.OUTPUT/'INPUT_SEAL.json')
    if (input_seal['rows']!=238 or C.sha(ARCHIVE.OUTPUT/'INPUTS.jsonl.gz')!=input_seal['input_sha256']
            or C.sha(ARCHIVE.OUTPUT/'PROTOCOL.json')!=input_seal['protocol_sha256']):
        raise ValueError('Exact pre-existing 238 physics/seed archive required')
    normal_protocol,_=_checked(NORMAL.OUTPUT,238);_checked(ARCHIVE.OUTPUT,238)
    original_protocol,_=_checked(D.OUTPUT,95);old_protocol,_=_checked(OLD_BASELINE,95)
    fresh_protocol,_=_checked(REMAINDER.OUTPUT,143)
    digest_maps={folder:_frame_digests(folder)for folder in(ARCHIVE.OUTPUT,NORMAL.OUTPUT,D.OUTPUT,OLD_BASELINE,REMAINDER.OUTPUT)}
    model=_verified_model();tasks=C.rows(ARCHIVE.OUTPUT/'INPUTS.jsonl.gz')
    result=[]
    for task in tasks:
        ident=task['id']
        if task['partition']=='REMAINDER':
            original=_sealed_frame(REMAINDER.OUTPUT,ident,digest_maps[REMAINDER.OUTPUT]);old=original['old_global_outline'];partial=original['partial_global_outline']
        else:
            old=_sealed_frame(OLD_BASELINE,ident,digest_maps[OLD_BASELINE])
            partial=_sealed_frame(D.OUTPUT,ident,digest_maps[D.OUTPUT])
        normal=_sealed_frame(NORMAL.OUTPUT,ident,digest_maps[NORMAL.OUTPUT]);tangent=_sealed_frame(ARCHIVE.OUTPUT,ident,digest_maps[ARCHIVE.OUTPUT])
        proposals,inventory=collect_proposal_bank(task,old,partial,normal,tangent)
        initial=task['finish_input']['_original_source']['initial_geometry']
        hidden=initial['initial_self_hidden']
        if hidden!=[s=='S'for s in task['spec']['states']]:raise ValueError('Original fixed S gate must remain authoritative')
        result.append(C.finite(dict(id=ident,partition=task['partition'],session=task['session'],image=task['image'],
            image_sha256=task['image_sha256'],states=task['spec']['states'],initial_self_hidden=hidden,
            baseline=task['baseline'],spec=task['spec'],families={name:entry['family']for name,entry in task['stage1_family_seed_bank'].items()},
            boundary_original=task['boundary_original_legacy'],boundary_merged=task['boundary_all'],
            u_regions=task['u_regions'],outline=task['outline'],proposals=proposals,proposal_inventory=inventory)))
    if (len(result)!=238 or [r['id']for r in result]!=previous['ids']
            or Counter(r['partition']for r in result)!=Counter(CAL=46,TEST=49,REMAINDER=143)):
        raise ValueError('Keep ordered original95 and remaining143, severe0')
    sources={};inputs={};protected=previous['protected_user_sha256']
    for protocol in (previous,normal_protocol,original_protocol,old_protocol,fresh_protocol):
        sources.update(protocol['source_sha256']);inputs.update(protocol['input_sha256'])
        if protocol['protected_user_sha256']!=protected:raise ValueError('Protected user6 binding changed')
    for folder in (ARCHIVE.OUTPUT,NORMAL.OUTPUT,D.OUTPUT,OLD_BASELINE,REMAINDER.OUTPUT):
        for name in ('PROTOCOL.json','PREDICTION_SEAL.json','PREDICTIONS.jsonl.gz'):
            path=folder/name;inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    for name in ('MODEL.json','MODEL_SEAL.json','PROTOCOL.json','PREPARATION_SEAL.json'):
        path=ANGLE.OUTPUT/name;inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    for module in (__file__,AUTH.__file__,ANGLE.__file__,NATIVE.__file__,NATIVE.EDGE.__file__,OUTLINE.__file__):
        path=Path(module);sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    return result,model,sources,inputs,protected


def _bridge_costs(records):
    return [{**r,'kind':KIND_MAP[r['kind']]}for r in records]


def _compact_selection(selection):
    """Keep every candidate vector/frontier, compact quadratic audit lists."""
    result={k:v for k,v in selection.items()if k!='pareto_domination'}
    result['pareto_domination']=[dict(index=r['index'],dominated_by_count=len(r['dominated_by']),
        dominated_by_indices_sha256=E._digest(r['dominated_by']))for r in selection['pareto_domination']]
    result['full_domination_sets_recomputed_before_compact_audit']=True
    return result


def _actual_pose(family,R,t,accepted,source):
    R=np.asarray(R,float);t=np.asarray(t,float);Q=np.asarray(family['physical_registration_Q'],float)
    return dict(available=True,R_cf=R.tolist(),centroid=t.tolist(),R_physical=(R@Q).tolist(),
        physical_registration_Q=Q.tolist(),cf_extents=deepcopy(family['cf_extents']),
        source=source,joint_feasible=bool(accepted),fallback=not bool(accepted),
        fallback_pose_does_not_solve_OT=not bool(accepted),fallback_pose_does_not_solve_U_regions=not bool(accepted))


def _readout(task,records,selected,pose):
    q,_=NATIVE.V.project_native(np.asarray(selected['family']['native_xyz']),np.asarray(pose['R_cf']),
        np.asarray(pose['centroid']),np.asarray(task['spec']['K']))
    visible={NATIVE.V._id(m):m for m in task['spec']['visible']};partial={r['native_corner']for r in records}
    points=[];sources=[]
    for k,state in enumerate(task['states']):
        if state=='S':points.append(q[k].tolist());sources.append('POSE_ONLY_MODEL_PROJECTION')
        elif state=='U':points.append(None);sources.append('UNRESOLVED_REGION_ONLY')
        elif state in('O','T'):
            points.append(q[k].tolist()if selected['accepted']else None)
            sources.append('FINITE_FEASIBLE_CANDIDATE_PROJECTION_NOT_UNIQUE'if selected['accepted']else'FALLBACK_DOES_NOT_SOLVE_OT')
        elif k in partial:points.append(None);sources.append('PARTIAL_RGB_NATIVE_EDGE_OR_OLD_NORMAL_AND_UNCERTAIN_TANGENT')
        else:points.append(deepcopy(visible[k]['xy']));sources.append(visible[k]['source'])
    return dict(reported_points=points,point_sources=sources,model_projection8=q.tolist(),
        S_projection_only=True,U_point_or_line_measurements=0,new_full_2D_corner_measurements=0,
        OT_solved=False,OT_unique_solution_claimed=False,finite_candidate_hypothesis_not_continuous_or_unique_solution=True)


def _run(task,model):
    started=time.monotonic();before=E._digest(task);arms={ARMS[0]:deepcopy(task['baseline'])};audits={}
    angle=ANGLE.authentication_angular_parameters(model)
    for arm,key in ((ARMS[1],'boundary_original'),(ARMS[2],'boundary_merged')):
        records=task[key];evaluations=[];full={};bundles={};mask_cfg=OUTLINE._outline_config(task['outline']['configuration'])
        for name,family in task['families'].items():
            spec=deepcopy(task['spec']);spec.update(xyz=family['native_xyz'],coarse_pose=family['coarse_pose'])
            spec['config']['prior_weight']=0.
            if not np.allclose(np.asarray(spec['xyz']).mean(0),0,rtol=0,atol=1e-12):raise ValueError('Centred registered model required')
            binding=dict(sha256=task['image_sha256'],image_hw=spec['image_hw'])
            bundle=AUTH.authenticate_supported_native_edges(spec['xyz'],spec['states'],spec['visible'],records,spec['image_hw'],
                initial_self_hidden=task['initial_self_hidden'],image_binding=binding,config=spec['config'],**angle)
            problem=NATIVE.prepare_native_edge_problem(spec['xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],
                spec['image_hw'],spec['coarse_pose'],config=spec['config'],boundary_observations=records,
                u_regions=task['u_regions'],edge_bundle=bundle)
            if not np.allclose(problem.Q,family['physical_registration_Q'],rtol=0,atol=1e-12):raise ValueError('Exact family physical Q binding required')
            bundles[name]=bundle
            Q=np.asarray(family['physical_registration_Q'])
            for index,proposal in enumerate(task['proposals']):
                R=np.asarray(proposal['R_physical'])@Q.T;t=np.asarray(proposal['t'])
                checked=problem.evaluate(R,t);token=name+'/'+str(index)
                mask_cost=None;mask_audit=None
                if checked['accepted']and task['outline'].get('supported'):
                    p=NATIVE.V._pose_vector(R,t,problem.scale)
                    mask_cost,_,mask_audit=OUTLINE.outline_value_and_gradient(p,problem.xyz,problem.K,problem.scale,task['outline'],mask_cfg)
                    if not mask_audit.get('valid_projection')or not np.isfinite(mask_cost):mask_cost=None
                entry=dict(name=token,family=name,accepted=checked['accepted'],all_joint_constraints_satisfied=checked['accepted'],
                    native_costs=_bridge_costs(checked.get('native_costs',[])),global_mask_cost=mask_cost,
                    physical_pose_digest=proposal['pose_digest'],minimum_joint_slack=checked.get('minimum_joint_slack'),
                    rejected_reason=checked.get('reason'))
                evaluations.append(entry)
                if checked['accepted']:full[token]=dict(evaluation=checked,family=family,mask_audit=mask_audit,proposal_index=index)
        selection=_compact_selection(AUTH.select_hierarchical_native_candidates(task['states'],evaluations))
        token=selection['selected_name']
        if token is None:
            family=task['families']['ORIGINAL_WD'];coarse=family['coarse_pose']
            selected=dict(accepted=False,family=family,R=coarse['R'],t=coarse['t'],
                reason='no_saved_proposal_satisfies_all_current_joint_guards',native_costs=[],objective=None)
        else:
            chosen=full[token];selected={**chosen['evaluation'],'family':chosen['family'],
                'proposal_index':chosen['proposal_index'],'global_mask_audit':chosen['mask_audit']}
        pose=_actual_pose(selected['family'],selected['R'],selected['t'],selected['accepted'],arm)
        arms[arm]=dict(actual_pose=pose,accepted=selected['accepted'],states=deepcopy(task['states']),
            keypoint_readout=_readout(task,records,selected,pose),solution=selected,
            ambiguity_audit=deepcopy(selection['ambiguity']),candidate_selection=selection,
            partial_boundary_observation=dict(boundary_observations=records,edge_authentication_by_family=bundles),
            fallback_audit=dict(own_original_initial_coarse_only=not selected['accepted'],old_best_pose_fallback=False),
            method=arm,WD_family=selected['family']['name'])
        audits[arm]=dict(selection_sha256=E._digest(selection),selected_name=selection['selected_name'],
            pareto_frontier_count=len(selection['pareto_frontier_indices']),proposal_inventory=task['proposal_inventory'],
            evaluated_candidates=len(evaluations),accepted_candidates=sum(r['accepted']for r in evaluations),
            invalid_candidate_does_not_drop_frame=True,old_accepted_flags_not_used=True,
            native_objective_only=True,global_mask_numeric_ties_only=True,new_optimizations=0,
            no_old_best_pose_fallback=True,all_OT_U_partial_guards_recomputed=True)
    if E._digest(task)!=before:raise ValueError('Frozen cached input changed')
    tangent_audit={arm:dict(variance_inflation=4.,outside_precision_multiplier=1.,
        active_native_ids=sorted({r['native_corner']for r in arms[arm]['solution']['native_costs']
            if r['kind']=='UNCERTAIN_ARCHIVED_TANGENT'}),
        archived_prediction_tangent_not_precise_RGB_point=True,native_candidate_selection_control=True)
        for arm in ARMS[1:]}
    return C.finite(dict(id=task['id'],partition=task['partition'],session=task['session'],image=task['image'],
        image_sha256=task['image_sha256'],states=task['states'],arms=arms,native_candidate_audit=audits,
        fixed_family={arm:arms[arm].get('WD_family',task['baseline'].get('WD_family'))for arm in ARMS},
        tangent_control_audit=tangent_audit,
        input_sha256=before,elapsed_seconds=time.monotonic()-started,
        contract=dict(whole238_uniform=True,GT_or_ID_or_partition_cost_routing=False,new_RGB_queries=0,new_NN_forwards=0,
            new_pose_optimizations=0,S_local_queries=0,U_local_queries=0,U_only_regions=True,
            original_BI_xy_Sigma_weights_OT_U_Q_preserved=True,old_vertex_normal_not_double_counted=True,
            cached_direction_guide_censoring_not_certified_absent=True,new_observer_inputs_used=False)))


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior versioned predictions')
    tasks,model,sources,inputs,protected=_load()
    for mapping in(sources,inputs,protected):E._verify_sources(mapping)
    output.mkdir(parents=True)
    protocol=dict(schema='CACHED_NATIVE_EDGE_CANDIDATE_ALL238_V4_17',ids=[t['id']for t in tasks],arms=list(ARMS),
        population=dict(CAL=46,TEST=49,REMAINDER=143,severe=0),source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=protected,angular_model_binding=E._binding(ANGLE.OUTPUT/'MODEL.json'),
        proposal_rule='all saved published/sample/optimizer-endpoint physical poses from P1/WD0/G10/D/T15/c, both family hypotheses; fresh guards only',
        native_selection='Pareto audit; minimum summed native energy; globalmask only machine-numeric ties',
        controls_uniform=True,all238_before_new_metrics=True,new_optimizations=0,new_NN_or_RGB_queries=0,
        native_identity_or_global_optimum_proved=False,reused_TEST_and_remaining_exploratory=True,
        new_NMS_observer_not_used=True,positive_CAL_plus_pixel_span_angular_variance=True,
        no_default_adoption=True)
    C.write(output/'PROTOCOL.json',protocol);C.save_rows(output/'INPUTS.jsonl.gz',tasks)
    for rel,digest in sources.items():
        target=output/'code_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/rel,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot mismatch')
    C.write(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=protocol['ids'],input_sha256=C.sha(output/'INPUTS.jsonl.gz'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),before_new_candidate_predictions_and_metrics=True))
    print('PREPARED',len(tasks),'physical proposals',sum(len(t['proposals'])for t in tasks),flush=True)


def predict(output=OUTPUT,workers=3):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'INPUT_SEAL.json')
    if seal['rows']!=238 or C.sha(output/'INPUTS.jsonl.gz')!=seal['input_sha256']or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']:
        raise ValueError('Complete immutable 238 input archive required')
    if(output/'PREDICTION_SEAL.json').exists():raise FileExistsError('Never overwrite a complete prediction seal')
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    tasks=list(C.rows(output/'INPUTS.jsonl.gz'));model=_verified_model();done={};started=time.monotonic()
    if[t['id']for t in tasks]!=protocol['ids']:raise ValueError('Exact ordered population required')
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'))
    futures={pool.submit(_run,t,model):t['id']for t in tasks}
    try:
        for future in as_completed(futures):
            row=future.result();done[row['id']]=row
            C.write(output/'frames'/(row['id'].replace(':','__')+'.json'),row)
            C.write(output/'LIVE_PROGRESS.json',dict(completed=len(done),total=238,latest_id=row['id'],elapsed_seconds=time.monotonic()-started))
            print(f'{len(done)}/238 {row["id"]}',flush=True)
    except BaseException:
        for future in futures:future.cancel()
        for process in list(getattr(pool,'_processes',{}).values()):process.terminate()
        pool.shutdown(wait=False,cancel_futures=True)
        raise
    else:pool.shutdown(wait=True)
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    C.save_rows(output/'PREDICTIONS.jsonl.gz',[done[t['id']]for t in tasks])
    C.write(output/'PREDICTION_SEAL.json',dict(rows=238,CAL=46,TEST=49,REMAINDER=143,severe=0,
        predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_seal_sha256=C.sha(output/'INPUT_SEAL.json'),all3_arms_complete=True,before_new_control_metric_read=True,
        ordered_ids=protocol['ids'],elapsed_seconds=time.monotonic()-started))


def score(output=OUTPUT):
    scorer=types.FunctionType(ARCHIVE.score.__code__,{**ARCHIVE.score.__globals__,'ARMS':ARMS},ARCHIVE.score.__name__)
    return scorer(Path(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=8)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.output)
    elif a.command=='predict':predict(a.output,a.workers)
    else:score(a.output)
