"""Uniform fresh geometry-guided actual-edge PnP controls on all238.

Two declared source policies, each native-only and one-sided containment.
Frozen bounded O/S regularizers remain; full/TP/normal pose priors are absent.
The existing v19 native-ranked six starts plus own coarse and cached contender
bank are reused. No numeric e pose is inserted as a prior or fallback. All
five arms and all frames are sealed before independent audit/reference score.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from itertools import zip_longest

from . import evaluate_prepartial_RGB_preservation_v4_26 as PARENT
from . import geometry_guided_edge_problem_v4_29 as G
from . import geometry_guided_edge_authentication_v4_29 as AUTH
from . import prepare_geometry_guided_native_evidence_v4_29 as PREP
from . import calibrate_geometry_guided_edge_noise_v4_29 as CAL
from . import bounded_geometric_priors_problem_v4_22 as BOUNDED
from . import coarse_prior_containment_problem_v4_25 as CONTAIN

C,E,BASE,OLD,REFINE=PARENT.C,PARENT.E,PARENT.BASE,PARENT.OLD,PARENT.REFINE
OUTPUT=PREP.B/'geometry_guided_edge_eval_v4_29'
PLAN=PREP.DECLARATION
ARMS=('SAME_FROZEN_E','GEOMETRY_EDGE_ALL_BI_NATIVE','GEOMETRY_EDGE_ALL_BI_CONTAINMENT',
      'GEOMETRY_EDGE_SOURCE_NATIVE','GEOMETRY_EDGE_SOURCE_CONTAINMENT')
POLICY_ARMS={'ALL_BI_REPLACE':ARMS[1:3],'RGB_NONFALLBACK_PRESERVE':ARMS[3:5]}
KIND_MAP={'COMPLETE_BI':'BI_COMPLETE','UNREPLACED_PARTIAL_NORMAL':'PARTIAL_NORMAL',
    'PROJECTED_NATIVE_EDGE':'RGB_EDGE','UNCERTAIN_ARCHIVED_TANGENT':'ARCHIVED_TANGENT',
    'ARCHIVED_Q_SELECTED_MEASURED_LINE_TANGENT':'ARCHIVED_TANGENT'}


def _merge_frozen(first,second):
    merged=dict(first)
    for relative,digest in second.items():
        if relative in merged and merged[relative]!=digest:
            raise ValueError('Frozen source/input ancestry conflict: '+relative)
        merged[relative]=digest
    return merged


def _models():
    # These frozen models supply O/S and the unchanged v19 start selector.
    # Their old line/tangent models never parameterize new v29 likelihoods.
    models=PARENT.V._verified_models();folder=CAL.OUTPUT
    seal=C.read(folder/'MODEL_SEAL.json')
    if seal['fit_partition']!='CAL'or seal['CAL_inventory']!=46 or seal['compatible_inventory']!=44:
        raise ValueError('New separately sealed CAL46/fixed44 v29 models required')
    for key in('source_sha256','input_sha256'):E._verify_sources(seal[key])
    if C.sha(folder/'PREPARATION_SEAL.json')!=seal['preparation_seal_sha256']:raise ValueError('New CAL preparation changed')
    for rel,digest in seal['assets'].items():
        if C.sha(folder/rel)!=digest:raise ValueError('New v29 CAL model/evidence changed')
    models['V29']={}
    for policy in G.SOURCE_POLICIES:
        line=C.read(folder/policy/'MODEL.json');tangent=C.read(folder/policy/'TANGENT_MODEL.json')
        unavailable=line.get('model')is None and line.get('status','').startswith('MODEL_UNAVAILABLE')
        unavailable=unavailable or(tangent.get('model')is None and tangent.get('status','').startswith('MODEL_UNAVAILABLE'))
        if unavailable:
            models['V29'][policy]=dict(available=False,old_model_substituted=False)
        else:
            noise=CAL.M.mixture_noise_parameters(line,source_policy=policy)
            tparams=CAL.M.tangent_variance_parameters(tangent,source_policy=policy)
            models['V29'][policy]=dict(available=True,noise=noise,tangent=tparams,
                line_model_sha256=C.sha(folder/policy/'MODEL.json'),tangent_model_sha256=C.sha(folder/policy/'TANGENT_MODEL.json'))
    return models


def _problem(task,family,models,policy):
    base=PREP.native_baseline(task,family);parameters=models['V29'][policy]
    if parameters['available']:
        noise=parameters['noise']
        if noise['noise_provenance']['source_policy']!=policy:raise ValueError('Exact policy CAL model required')
        bundle=AUTH.authenticate_geometry_guided_edges(base,task['v29_observation'],
            image_binding=dict(sha256=task['image_sha256'],image_hw=task['spec']['image_hw']),
            initial_geometry=task['v29_initial_geometry'],expected_source_policy=policy,**noise)
        gamma=parameters['tangent']['gamma']
    else:
        bundle=dict(observation_family=G.OBSERVATION_FAMILY,population=G.POPULATION,groups=[],ordered_corner_pairs=[],
            noise_unavailable_abstention=True,old_model_substituted=False)
        gamma=0.
    problem=G.GeometryGuidedEdgeProblem(base,bundle,source_policy=policy,tangent_gamma=gamma)
    problem.reference_start_problem=OLD.V.PARENT._problem(task,family,models['NMS']['OLD'])[0]
    problem.plan['CAL_noise_binding']=deepcopy(parameters)
    return problem,bundle


def _bridge(records):
    return [{**r,'kind':KIND_MAP[r['kind']]}for r in records]


def select_candidates(states,candidates,*,containment_weight):
    """Authenticate every fresh feasible vector; stable scalar minimum, no mask tie."""
    candidates=list(candidates);weight=float(containment_weight)
    if not np.isfinite(weight)or weight<0:raise ValueError('Fixed nonnegative containment weight required')
    authenticated=BOUNDED.AUTH.select_hierarchical_native_candidates(states,candidates)
    rows=[]
    for r in authenticated['candidates']:
        source=candidates[r['index']];regularizers=source['regularizer_costs']
        if [a['kind']for a in regularizers]!=list(BOUNDED.REGULARIZER_KINDS):
            raise ValueError('Only frozen bounded O/S regularizers; no hidden rotation/TP prior')
        prior=sum(float(a['weighted_cost'])for a in regularizers);shape=float(source['global_mask_cost'])
        if not np.isfinite([prior,shape]).all()or min(prior,shape)<0:raise ValueError('Finite nonnegative separate energies required')
        rows.append(dict(index=r['index'],name=r['name'],family=r['family'],native_energy=r['native_energy'],
            regularizer_energy=prior,global_containment_cost=shape,weighted_containment_objective=weight*shape,
            total_energy=r['native_energy']+prior+weight*shape))
    selected=None
    if rows:
        minimum=min(r['total_energy']for r in rows)
        ties=[r for r in rows if abs(r['total_energy']-minimum)<=32*np.finfo(float).eps*max(1.,abs(minimum),abs(r['total_energy']))]
        selected=min(ties,key=lambda r:r['index'])
    ambiguity=deepcopy(authenticated['ambiguity']);ambiguity['global_uniqueness_proved']=False
    return dict(selected_name=None if selected is None else selected['name'],
        selected_family=None if selected is None else selected['family'],candidates=rows,
        native_authentication_audit=BOUNDED._compact_authentication(authenticated),ambiguity=ambiguity,
        native_Pareto_audit_not_prefilter=True,old_full_mask_cost_or_numeric_tie_break_used=False,
        e_numeric_pose_prior_or_fallback=False,containment_weight=weight)


def _readout(task,selected,pose):
    plan=selected.get('v29_cost_plan',{})
    restored=set(PARENT._restored_ids(task));owners=set(plan.get('new_owner_ids',[]))
    records=[b for b in task['boundary_merged']if b['native_corner']not in restored|owners]
    readout=BASE._readout(task,records,selected,pose)
    for k in owners:
        if task['initial_self_hidden'][k]or task['states'][k]not in('B','I'):raise ValueError('No S/U point readout revocation')
        readout['reported_points'][k]=None;readout['point_sources'][k]='ACTUAL_NATIVE_EDGE_CONSTRAINTS_NO_PRECISE_2D_KP'
    readout.update(continuous_SE3_search_performed=True,new_full_2D_corner_measurements=0,
        S_image_or_2D_terms=0,U_range_only=True,OT_solved=False,OT_unique_solution_claimed=False,
        v29_new_owner_ids=sorted(owners),archived_nonfallback_q_restored_without_new_owner=True)
    return readout


def _run(task,models):
    before=E._digest(task);started=time.monotonic();arms={ARMS[0]:deepcopy(task['baseline'])};audits={}
    regions,oparams,original_S=OLD._prior_inputs(task,models)
    cfg=BASE.OUTLINE._outline_config(task['outline']['configuration'])
    shape_weight=cfg['outline_factor_weight']if task['outline'].get('supported')else 0.
    start_names={}
    for policy,policy_arms in POLICY_ARMS.items():
        native_endpoints=[]
        for containment,arm in zip((False,True),policy_arms):
            candidates=[];details={};searches={};plans={}
            for name,family in task['families'].items():
                core,bundle=_problem(task,family,models,policy);plans[name]=core.plan
                rebound=BOUNDED.S.rebind_bounded_visibility_trial_geometry(original_S,family['native_xyz'])
                bounded=BOUNDED.BoundedGeometricPriorsProblem(core,regions,oparams,rebound)
                problem=CONTAIN.ContinuousContainmentProblem(bounded,task['outline'],cfg)if containment else bounded
                Q=np.asarray(family['physical_registration_Q'])
                proposals=[dict(name='CACHED_'+str(i),R=(np.asarray(p['R_physical'])@Q.T).tolist(),t=p['t'])
                           for i,p in enumerate(task['proposals'])]
                def add(R,t,token,origin):
                    checked=problem.evaluate(R,t)
                    shape=float(checked.get('global_containment_cost',0.))if containment else 0.
                    candidates.append(dict(name=token,family=name,accepted=checked['accepted'],
                        all_joint_constraints_satisfied=checked['accepted'],native_costs=_bridge(checked.get('native_costs',[])),
                        regularizer_costs=checked.get('regularizer_costs',[]),global_mask_cost=shape,origin=origin))
                    if checked['accepted']:
                        expected=checked['native_observation_objective']+checked['regularizer_objective']+(shape_weight*shape if containment else 0.)
                        if not np.isclose(expected,checked['objective'],atol=1e-9,rtol=1e-10):raise ValueError('Optimizer/selector total mismatch')
                        details[token]=dict(evaluation=checked,family=family,origin=origin)
                for i,p in enumerate(proposals):add(p['R'],p['t'],name+'/CACHED_'+str(i),'CACHED')
                if containment:
                    for i,p in enumerate(native_endpoints):
                        if p['family']==name:add(p['R'],p['t'],name+'/NATIVE_ENDPOINT_'+str(i),'SAME_POLICY_NATIVE_ENDPOINT')
                refined=PARENT.V.V._REFINER(problem,proposals,family['coarse_pose'],configuration=REFINE.DEFAULT_CONFIG)
                for i,result in enumerate(refined['solutions']):
                    add(result['R'],result['t'],name+'/REFINED_'+str(i),'CONTINUOUS_CONTAINMENT_ENDPOINT'if containment else'CONTINUOUS_NATIVE_ENDPOINT')
                    if not containment:native_endpoints.append(dict(family=name,R=result['R'],t=result['t']))
                names=refined['start_selection']['selected_start_names']
                if name in start_names and start_names[name]!=names:raise ValueError('Frozen v19 native six-start policy changed between arms')
                start_names[name]=names
                searches[name]=dict(search=refined['search'],start_selection=refined['start_selection'],candidates=refined['candidates'],
                    feasible_endpoints=[{k:r.get(k)for k in('R','t','R_physical','objective','initialization','branch')}for r in refined['solutions']])
            selection=select_candidates(task['states'],candidates,containment_weight=shape_weight if containment else 0.)
            token=selection['selected_name']
            if token is None:
                family=task['families']['ORIGINAL_WD'];coarse=family['coarse_pose'];origin='OWN_INITIAL_COARSE_FALLBACK'
                selected=dict(accepted=False,family=family,R=coarse['R'],t=coarse['t'],objective=None,native_costs=[],
                    regularizer_costs=[],v29_cost_plan=plans['ORIGINAL_WD'],reason='no_current_feasible_candidate')
            else:
                detail=details[token];origin=detail['origin'];selected={**detail['evaluation'],'family':detail['family']}
                chosen=next(r for r in selection['candidates']if r['name']==token)
                if not np.isclose(chosen['total_energy'],selected['objective'],atol=1e-9,rtol=1e-10):raise ValueError('Selected current scalar cost differs')
            pose=BASE._actual_pose(selected['family'],selected['R'],selected['t'],selected['accepted'],arm)
            arms[arm]=dict(actual_pose=pose,accepted=selected['accepted'],states=deepcopy(task['states']),solution=selected,
                keypoint_readout=_readout(task,selected,pose),WD_family=selected['family']['name'],source_policy=policy,
                method=arm,candidate_selection=selection,ambiguity_audit=selection['ambiguity'],
                new_geometry_edge_cost_plans=plans,selected_endpoint_origin=origin,
                fallback_audit=dict(own_original_initial_coarse_only=not selected['accepted'],old_best_pose_fallback=False,e_numeric_pose_used=False))
            audits[arm]=dict(source_policy=policy,containment_enabled=containment,family_refinements=searches,
                native_start_names=deepcopy(start_names),all_cached_contenders_retained=True,
                same_policy_native_endpoints_retained=containment,old_full_mask_or_numeric_tie_break=False,
                full_pose_TP_topnormal_prior_weights=0.,O_S_frozen_bounded_regularizers_only=True,
                cost_improvement_is_not_GT_accuracy_guarantee=True)
    if E._digest(task)!=before:raise ValueError('Frozen old observations/candidate bank mutated')
    return C.finite(dict(schema='UNIFORM_GEOMETRY_GUIDED_EDGE_PREDICTION_V29',id=task['id'],partition=task['partition'],
        session=task['session'],image=task['image'],image_sha256=task['image_sha256'],states=task['states'],arms=arms,
        fixed_family={a:arms[a].get('WD_family')for a in ARMS},bounded_prior_refinement_audit=audits,
        tangent_control_audit=dict(new_policy_specific_CAL_actual_line_tangents=True,old_nonowner_fallback_T4_preserved=True,
            S_U_point_costs=0),input_sha256=before,elapsed_seconds=time.monotonic()-started))


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve all preparations/failures')
    if C.sha(PLAN)!=PREP.DECLARATION_SHA256 or C.read(PLAN)['arms']!=list(ARMS):raise ValueError('Exact five-arm declaration required')
    models=_models();parent=C.read(PREP.TASKS/'PROTOCOL.json');pseal=C.read(PREP.TASKS/'INPUT_SEAL.json')
    obsprotocol=C.read(PREP.OBS/'PROTOCOL.json');obsseal=C.read(PREP.OBS/'OBSERVATION_SEAL.json')
    if (pseal['rows']!=238 or C.sha(PREP.TASKS/'INPUTS.jsonl.gz')!=pseal['input_sha256']
            or obsseal['rows']!=238 or obsseal['ordered_ids']!=parent['ids']
            or C.sha(PREP.OBS/'OBSERVATIONS.jsonl.gz')!=obsseal['observations_sha256']
            or C.sha(PREP.OBS/'PROTOCOL.json')!=obsseal['protocol_sha256']):
        raise ValueError('Exact complete all238 physics and fresh observation seals required')
    if parent['protected_user_sha256']!=obsprotocol['protected_user_sha256']:
        raise ValueError('Exact same protected user artifact contract required')
    for source_protocol in(parent,obsprotocol):
        for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(source_protocol[key])
    sources=_merge_frozen(parent['source_sha256'],obsprotocol['source_sha256'])
    inputs=_merge_frozen(parent['input_sha256'],obsprotocol['input_sha256'])
    modelseal=C.read(CAL.OUTPUT/'MODEL_SEAL.json')
    for key,target in(('source_sha256',sources),('input_sha256',inputs)):
        for relative,digest in modelseal[key].items():
            if relative in target and target[relative]!=digest:raise ValueError('Frozen dependency ancestry conflict')
            target[relative]=digest
    for module in(sys.modules[__name__],G,AUTH,PREP,CAL,CAL.M,PARENT,BOUNDED,CONTAIN,CONTAIN.FACTOR):
        p=Path(module.__file__).resolve();sources[str(p.relative_to(C.ROOT))]=C.sha(p)
    for p in(PLAN,PREP.TASKS/'PROTOCOL.json',PREP.TASKS/'INPUT_SEAL.json',PREP.TASKS/'INPUTS.jsonl.gz',
             PREP.OBS/'PROTOCOL.json',PREP.OBS/'PREPARATION_SEAL.json',PREP.OBS/'OBSERVATION_SEAL.json',PREP.OBS/'OBSERVATIONS.jsonl.gz',
             CAL.OUTPUT/'MODEL_SEAL.json',CAL.OUTPUT/'PREPARATION_SEAL.json'):
        inputs[str(p.relative_to(C.ROOT))]=C.sha(p)
    for rel,digest in modelseal['assets'].items():inputs[str((CAL.OUTPUT/rel).relative_to(C.ROOT))]=digest
    for mapping in(sources,inputs,parent['protected_user_sha256']):E._verify_sources(mapping)
    output.mkdir(parents=True);ids=[];counts=Counter()
    def joined_rows():
        for task,row in zip_longest(C.rows(PREP.TASKS/'INPUTS.jsonl.gz'),C.rows(PREP.OBS/'OBSERVATIONS.jsonl.gz')):
            if task is None or row is None:raise ValueError('Exact complete physics/observation row counts required')
            PREP.validate_runtime_join(task,row);joined=task
            joined['v29_observation']=row['observation'];joined['v29_initial_geometry']=row['initial_geometry']
            counts[joined['partition']]+=1;ids.append(joined['id']);yield joined
    C.save_rows(output/'INPUTS.jsonl.gz',joined_rows())
    if ids!=parent['ids']or counts!=Counter(CAL=46,TEST=49,REMAINDER=143):raise ValueError('No drops or primary-family filtering')
    protocol={**parent,'schema':'UNIFORM_GEOMETRY_GUIDED_EDGE_ALL238_PROTOCOL_V29','arms':list(ARMS),
        'source_sha256':sources,'input_sha256':inputs,'declaration':E._binding(PLAN),
        'observation_generation':AUTH.PRODUCER_GENERATION,'noise_observation_family':G.OBSERVATION_FAMILY,
        'new_noise_policies':list(G.SOURCE_POLICIES),'runtime_objective':'native + frozen bounded O/S + optional one-sided C1',
        'full_pose_TP_topnormal_prior_weights':0.,'old_e_numeric_pose_added_as_prior_or_fallback':False,
        'exact_old_v19_bank_native_ranked_six_plus_coarse_rule':True,'all5arms_all238_seal_before_reference_reads':True,
        'source_binding_scope':'conservative sealed ancestry union; not every union file executes in each worker',
        'source_family_native_channel_correspondence_hypotheses_unchanged':True,'no_default_adoption':True}
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        p=output/'code_snapshot'/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,p)
        if C.sha(p)!=digest:raise ValueError('Source snapshot mismatch')
    for mapping in(sources,inputs,parent['protected_user_sha256']):E._verify_sources(mapping)
    C.write(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=parent['ids'],protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_sha256=C.sha(output/'INPUTS.jsonl.gz'),before_new_predictions_or_primary_metrics=True))
    print(dict(prepared=238,arms=5),flush=True)


def predict(output=OUTPUT,workers=4):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'INPUT_SEAL.json')
    if workers not in range(1,9):raise ValueError('Use1..8 explicitly budgeted workers')
    if (seal['rows']!=238 or seal['input_sha256']!=C.sha(output/'INPUTS.jsonl.gz')or
            seal['protocol_sha256']!=C.sha(output/'PROTOCOL.json')or protocol['arms']!=list(ARMS)):
        raise ValueError('Exact new all238 five-arm preparation required')
    if (output/'PREDICTION_SEAL.json').exists():raise FileExistsError('Preserve completed predictions')
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    models=_models();started=time.monotonic();done=set();tasks=iter(C.rows(output/'INPUTS.jsonl.gz'))
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'));pending={}
    def submit_one():
        task=next(tasks,None)
        if task is None:return False
        pending[pool.submit(_run,task,models)]=task['id'];return True
    try:
        for _ in range(workers):
            if not submit_one():break
        while pending:
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                expected=pending.pop(future);row=future.result()
                if row['id']!=expected or expected in done or set(row['arms'])!=set(ARMS):raise ValueError('Every declared arm/frame required')
                C.write(output/'frames'/(expected.replace(':','__')+'.json'),row);done.add(expected)
                C.write(output/'LIVE_PROGRESS.json',dict(completed=len(done),total=238,id=expected,elapsed_seconds=time.monotonic()-started))
                print(str(len(done))+'/238 '+expected,flush=True);submit_one()
    except BaseException as error:
        C.write(output/'FAILED_BEFORE_SEAL.json',dict(completed=len(done),error=repr(error),no_reference_reads=True))
        for future in pending:future.cancel()
        for process in list(getattr(pool,'_processes',{}).values()):process.terminate()
        pool.shutdown(wait=False,cancel_futures=True);raise
    else:pool.shutdown(wait=True)
    if done!=set(protocol['ids']):raise ValueError('Keep all238 including fallback frames')
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    C.save_rows(output/'PREDICTIONS.jsonl.gz',(C.read(output/'frames'/(i.replace(':','__')+'.json'))for i in protocol['ids']))
    C.write(output/'PREDICTION_SEAL.json',dict(rows=238,CAL=46,TEST=49,REMAINDER=143,severe=0,
        predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_seal_sha256=C.sha(output/'INPUT_SEAL.json'),ordered_ids=protocol['ids'],arms=list(ARMS),arm_count=5,
        all_declared_arms_complete=True,before_new_control_metric_read=True,elapsed_seconds=time.monotonic()-started))


def score(output=OUTPUT):
    """Same frozen metrics/references; all238 audit and full5 seal first."""
    output=Path(output).resolve();seal=C.read(output/'PREDICTION_SEAL.json');protocol=C.read(output/'PROTOCOL.json')
    if (seal['rows']!=238 or seal['arms']!=list(ARMS)or seal['arm_count']!=5 or not seal['all_declared_arms_complete']
            or seal['predictions_sha256']!=C.sha(output/'PREDICTIONS.jsonl.gz')
            or seal['protocol_sha256']!=C.sha(output/'PROTOCOL.json')
            or seal['input_seal_sha256']!=C.sha(output/'INPUT_SEAL.json')or seal['ordered_ids']!=protocol['ids']):
        raise ValueError('Full declared five-arm all238 seal required before references')
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    auditpath=output/'independent_root_audit_v1/AUDIT.json';audit=C.read(auditpath);aseal=C.read(auditpath.with_name('AUDIT_SEAL.json'))
    if audit['rows']!=238 or audit['failure_count']or audit['failures']or C.sha(auditpath)!=aseal['receipt_sha256']or audit['prediction_seal']['sha256']!=C.sha(output/'PREDICTION_SEAL.json'):
        raise ValueError('Current independent zero-failure all238 audit required')
    A=BASE.ARCHIVE;small=[]
    for row in C.rows(output/'PREDICTIONS.jsonl.gz'):
        if set(row['arms'])!=set(ARMS):raise ValueError('No arm dropout before reference decode')
        small.append(dict(id=row['id'],partition=row['partition'],fixed_family=row['fixed_family'],
            actual_poses={a:row['arms'][a]['actual_pose']for a in ARMS},accepted={a:row['arms'][a]['accepted']for a in ARMS},
            tangent_control_audit=row['tangent_control_audit']))
    if [r['id']for r in small]!=protocol['ids']:raise ValueError('No population drop before reference access')
    reports={};bindings={}
    for partition,count in(('CAL',46),('TEST',49),('REMAINDER',143)):
        requested=[r['id']for r in small if r['partition']==partition]
        if partition=='REMAINDER':
            plan=C.read(A.REF.OUTPUT/'PLAN.json');annotations={a['id']:a for a in plan['annotations']}
            references={i:A.REF.reference_from_document(C.read(C.ROOT/annotations[i]['annotation']))for i in requested}
            bindings[partition]=E._binding(A.REF.OUTPUT/'PLAN.json');metric=A.REF.metric_for_reference
        else:
            path=A.E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json'if partition=='CAL'else'test_reused_v1/ISOLATED_TEST_SCORING.json')
            labels=C.read(path)['rows'];references={i:labels[i]['target']['truth']for i in requested}
            bindings[partition]=E._binding(path);metric=A.pose_metric
        rows=[dict(id=r['id'],fixed_family=r['fixed_family'],accepted=r['accepted'],tangent_control_audit=r['tangent_control_audit'],
            metrics={a:metric(r['actual_poses'][a],references[r['id']])for a in ARMS})for r in small if r['partition']==partition]
        if len(rows)!=count:raise ValueError('Keep every native-flagged/fallback frame')
        summary={}
        for arm in ARMS:
            ts=[r['metrics'][arm]['translation_cm']for r in rows];rs=[r['metrics'][arm]['rotation_deg']for r in rows if r['metrics'][arm]['rotation_deg']is not None]
            summary[arm]=dict(translation_cm=A.distribution(ts),rotation_deg=A.distribution(rs),translation_le5=sum(t<=5 for t in ts),
                successes_5cm_5deg=sum(bool(r['metrics'][arm]['success_5cm_5deg'])for r in rows),rotation_unresolved=count-len(rs),
                fallback_count=sum(not r['accepted'][arm]for r in rows),max_translation_id=rows[int(np.argmax(ts))]['id'])
        reports[partition]=dict(rows=rows,summary=summary,count=count,reference_binding=bindings[partition],all_rows_kept=True,
            TEST49_and_REMAINDER143_reused_exploratory=True,remaining_reference_is_derived_not_independent_metrology=True,
            new_parameter_or_source_selection_by_metrics=False)
        C.write(output/(partition+'_SCORE.json'),reports[partition]);print(partition,{a:(s['translation_le5'],s['translation_cm']['mean'],s['translation_cm']['max'])for a,s in summary.items()},flush=True)
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    C.write(output/'SCORE_SEAL.json',dict(prediction_seal=E._binding(output/'PREDICTION_SEAL.json'),score_sha256={p:C.sha(output/(p+'_SCORE.json'))for p in reports},
        reference_bindings=bindings,all_numeric_references_decoded_only_after_new_control238_seal=True,
        same_unchanged_numeric_metric_reference_and_summary_functions=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    dict(prepare=prepare,predict=lambda o:predict(o,args.workers),score=score)[args.stage](args.output)
