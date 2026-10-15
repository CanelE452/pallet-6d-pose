"""Uniform saved-start native-edge refinement, exact cached238 inputs.

All fresh feasible cached proposals remain contenders. Six diverse native
starts plus own coarse are continuously optimized per family and profile;
the global mask is evaluated only at machine-numeric native-energy ties.
Every original row/fallback remains. Predictions seal before metric reads.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import time
import types

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_cached_native_edges_v4_17 as BASE
from . import refine_native_edge_pose_v4_17 as REFINE

OUTPUT = BASE.OUTPUT.parent / 'refined_native_edge_candidates_v4_17'
ARMS = ('SAME_FROZEN_E', 'REFINED_NATIVE_ORIGINAL_BI_LEGACY', 'REFINED_NATIVE_D_MERGED_ALL')


def _inputs(folder):
    folder = Path(folder).resolve()
    protocol = C.read(folder/'PROTOCOL.json'); seal = C.read(folder/'INPUT_SEAL.json')
    if (seal['rows'] != 238 or C.sha(folder/'INPUTS.jsonl.gz') != seal['input_sha256']
            or C.sha(folder/'PROTOCOL.json') != seal['protocol_sha256']
            or protocol['population'] != dict(CAL=46, TEST=49, REMAINDER=143, severe=0)):
        raise ValueError('Exact immutable cached238 prediction inputs required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    E._verify_binding(protocol['angular_model_binding'])
    return protocol, seal


def _select_with_mask_ties(states, candidates, geometry, task):
    """Apply the same selector; compute mask only on its native-minimum ties."""
    selection = BASE.AUTH.select_hierarchical_native_candidates(states, candidates)
    mask_audits = {}
    if task['outline'].get('supported'):
        cfg = BASE.OUTLINE._outline_config(task['outline']['configuration'])
        for index in selection['mask_tie_break_indices']:
            item = candidates[index]; detail = geometry[item['name']]
            checked, problem = detail['evaluation'], detail['problem']
            p = BASE.NATIVE.V._pose_vector(np.asarray(checked['R']), np.asarray(checked['t']), problem.scale)
            try:
                value, _, audit = BASE.OUTLINE.outline_value_and_gradient(
                    p, problem.xyz, problem.K, problem.scale, task['outline'], cfg)
                if audit.get('valid_projection') and np.isfinite(value):
                    item['global_mask_cost'] = float(value)
                mask_audits[item['name']] = audit
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                mask_audits[item['name']] = dict(valid_projection=False, reason=str(error))
    return BASE.AUTH.select_hierarchical_native_candidates(states, candidates), mask_audits


def _sample(solution):
    return {k:deepcopy(solution.get(k)) for k in ('R', 't', 'R_physical', 'accepted', 'objective',
        'native_costs', 'minimum_joint_slack', 'initialization', 'branch')}


def _run(task, model):
    started = time.monotonic(); before = E._digest(task)
    arms = {ARMS[0]:deepcopy(task['baseline'])}; audits = {}; tangent_audits = {}
    angle = BASE.ANGLE.authentication_angular_parameters(model)
    for arm, key in ((ARMS[1], 'boundary_original'), (ARMS[2], 'boundary_merged')):
        records = task[key]; candidates = []; details = {}; bundles = {}; searches = {}
        cached_energies = []
        for name, family in task['families'].items():
            spec = deepcopy(task['spec']); spec.update(xyz=family['native_xyz'], coarse_pose=family['coarse_pose'])
            spec['config']['prior_weight'] = 0.
            if not np.allclose(np.asarray(spec['xyz']).mean(0), 0., atol=1e-12, rtol=0):
                raise ValueError('Registered centered cuboid required')
            bundle = BASE.AUTH.authenticate_supported_native_edges(spec['xyz'], spec['states'], spec['visible'],
                records, spec['image_hw'], initial_self_hidden=task['initial_self_hidden'],
                image_binding=dict(sha256=task['image_sha256'], image_hw=spec['image_hw']),
                config=spec['config'], **angle)
            problem = BASE.NATIVE.prepare_native_edge_problem(spec['xyz'], spec['K'], spec['states'], spec['visible'],
                spec['domains'], spec['image_hw'], spec['coarse_pose'], config=spec['config'],
                boundary_observations=records, u_regions=task['u_regions'], edge_bundle=bundle)
            Q = np.asarray(family['physical_registration_Q'])
            if not np.allclose(problem.Q, Q, atol=1e-12, rtol=0):
                raise ValueError('Current family physical Q changed')
            bundles[name] = bundle
            proposals = [dict(name='CACHED_'+str(i), R=(np.asarray(p['R_physical'])@Q.T).tolist(),
                t=p['t']) for i,p in enumerate(task['proposals'])]

            def add(R, t, token, origin):
                checked = problem.evaluate(np.asarray(R), np.asarray(t))
                mapped = BASE._bridge_costs(checked.get('native_costs', []))
                candidates.append(dict(name=token, family=name, accepted=checked['accepted'],
                    all_joint_constraints_satisfied=checked['accepted'], native_costs=mapped,
                    global_mask_cost=None, origin=origin, minimum_joint_slack=checked.get('minimum_joint_slack'),
                    rejected_reason=checked.get('reason')))
                if checked['accepted']:
                    details[token] = dict(evaluation=checked, family=family, problem=problem, origin=origin)
                    if origin == 'CACHED': cached_energies.append(sum(r['weighted_robust_cost'] for r in mapped))

            # Keep every current-feasible cached contender, beyond the start cap.
            for index, proposal in enumerate(proposals):
                add(proposal['R'], proposal['t'], name+'/CACHED_'+str(index), 'CACHED')
            result = REFINE.refine_native_edge_pose(problem, proposals, family['coarse_pose'],
                configuration=REFINE.DEFAULT_CONFIG)
            for index, solution in enumerate(result['solutions']):
                add(solution['R'], solution['t'], name+'/REFINED_'+str(index), 'CONTINUOUS_SEARCH_ENDPOINT')
            searches[name] = dict(search=result['search'], start_selection=result['start_selection'],
                candidates=result['candidates'], feasible_samples=[_sample(s) for s in result['solutions']],
                all_cached_contenders_retained=True)
        selection, mask_audits = _select_with_mask_ties(task['states'], candidates, details, task)
        token = selection['selected_name']
        if token is None:
            family = task['families']['ORIGINAL_WD']; coarse = family['coarse_pose']
            selected = dict(accepted=False, family=family, R=coarse['R'], t=coarse['t'],
                objective=None, native_costs=[], reason='no_current_feasible_cached_or_refined_native_candidate')
            origin = 'OWN_INITIAL_COARSE_FALLBACK'
        else:
            detail = details[token]; origin = detail['origin']
            selected = {**detail['evaluation'], 'family':detail['family'],
                'selected_endpoint_origin':origin, 'global_mask_audit':mask_audits.get(token)}
            if cached_energies:
                baseline = min(cached_energies); now = selected['objective']
                tolerance = 128*np.finfo(float).eps*max(1., abs(baseline), abs(now))
                if now > baseline + tolerance:
                    raise RuntimeError('Retaining cached contenders must not worsen minimum native energy')
        pose = BASE._actual_pose(selected['family'], selected['R'], selected['t'], selected['accepted'], arm)
        readout = BASE._readout(task, records, selected, pose)
        readout.pop('finite_candidate_hypothesis_not_continuous_or_unique_solution', None)
        readout.update(joint_feasible_representative_not_unique_solution=True,
            continuous_SE3_search_performed=True, selected_endpoint_origin=origin,
            selected_pose_is_cached_contender=origin=='CACHED')
        arms[arm] = dict(actual_pose=pose, accepted=selected['accepted'], states=deepcopy(task['states']),
            keypoint_readout=readout, solution=selected,
            ambiguity_audit=dict(native_selection=deepcopy(selection['ambiguity']),
                historical_baseline=deepcopy(task['baseline'].get('ambiguity_audit', {})),
                global_uniqueness_proved=False),
            candidate_selection=BASE._compact_selection(selection),
            partial_boundary_observation=dict(boundary_observations=records, edge_authentication_by_family=bundles),
            fallback_audit=dict(own_original_initial_coarse_only=not selected['accepted'], old_best_pose_fallback=False),
            method=arm, WD_family=selected['family']['name'])
        tangent_audits[arm] = dict(tangent_variance_inflation=4.,
            active_native_ids=sorted({r['native_corner'] for r in selected['native_costs']
                if r['kind']=='UNCERTAIN_ARCHIVED_TANGENT'}),
            archived_prediction_tangent_not_precise_RGB_point=True)
        audits[arm] = dict(selection=BASE._compact_selection(selection),
            proposal_inventory=task['proposal_inventory'], family_refinements=searches,
            evaluated_candidates=len(candidates), fresh_feasible_candidates=sum(c['accepted'] for c in candidates),
            all_cached_feasible_contenders_retained=True,
            cached_minimum_native_energy=None if not cached_energies else min(cached_energies),
            selected_native_energy=selected['objective'], native_energy_nonworsening_not_GT_accuracy_guarantee=True,
            mask_numeric_tie_audits=mask_audits, new_P3P_or_global_search=0,
            no_normal_or_full_pose_or_mask_prior=True, all_OT_U_partial_guards_recomputed=True)
    if E._digest(task) != before: raise ValueError('Frozen prediction input mutated')
    return C.finite(dict(id=task['id'], partition=task['partition'], session=task['session'],
        image=task['image'], image_sha256=task['image_sha256'], states=task['states'], arms=arms,
        fixed_family={a:arms[a].get('WD_family',task['baseline'].get('WD_family')) for a in ARMS},
        tangent_control_audit=tangent_audits, native_refinement_audit=audits,
        input_sha256=before, elapsed_seconds=time.monotonic()-started,
        contract=dict(all238_uniform=True, GT_ID_partition_cost_routing=False, new_RGB_or_NN_queries=0,
            S_local_queries=0, U_local_queries=0, U_only_regions=True,
            original_BI_q_covariance_reliability_domains_Q_preserved=True,
            old_vertex_normal_not_double_counted=True, new_NMS_observer_used=False,
            optional_finite_native_edge_extent_guard_used=False)))


def prepare(output=OUTPUT, cached_folder=BASE.OUTPUT):
    output = Path(output).resolve(); cached_folder = Path(cached_folder).resolve()
    if output.exists(): raise FileExistsError('Preserve prior immutable outputs')
    parent, parent_seal = _inputs(cached_folder); model = BASE._verified_model()
    sources = dict(parent['source_sha256'])
    for filename in (__file__, BASE.__file__, REFINE.__file__, BASE.NATIVE.__file__):
        path=Path(filename); sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    inputs = dict(parent['input_sha256'])
    for filename in ('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz'):
        path=cached_folder/filename; inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    for mapping in (sources, inputs, parent['protected_user_sha256']): E._verify_sources(mapping)
    output.mkdir(parents=True)
    protocol = dict(schema='REFINED_NATIVE_EDGE_ALL238_V17', ids=parent['ids'], arms=list(ARMS),
        population=parent['population'], source_sha256=sources, input_sha256=inputs,
        protected_user_sha256=parent['protected_user_sha256'], angular_model_binding=parent['angular_model_binding'],
        refinement_configuration=deepcopy(REFINE.DEFAULT_CONFIG),
        parent_cached_input_binding=E._binding(cached_folder/'INPUT_SEAL.json'),
        prepared_inputs_byte_identical_to_cached=True, cached_inputs_sha256=parent_seal['input_sha256'],
        candidate_policy='all cached proposals plus every freshly feasible continuous endpoint; no old acceptance/cost filter',
        selection='same native Pareto/min-sum rule; global mask only machine-numeric ties',
        original_joint_iterations_and_all_T_union_branches=True, no_new_P3P_or_global_search=True,
        no_RGB_or_NN_or_parameter_fitting=True, no_full_pose_top_normal_or_mask_objective=True,
        all238_before_new_metric_reads=True, optional_finite_edge_extent_guard_used=False,
        reused_TEST_and_remaining_exploratory=True, no_default_adoption=True)
    C.write(output/'PROTOCOL.json',protocol)
    shutil.copyfile(cached_folder/'INPUTS.jsonl.gz',output/'INPUTS.jsonl.gz')
    if C.sha(output/'INPUTS.jsonl.gz') != parent_seal['input_sha256']:
        raise ValueError('Byte-identical prepared physics/proposal inputs required')
    for rel,digest in sources.items():
        target=output/'code_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/rel,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot mismatch')
    C.write(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=protocol['ids'],
        input_sha256=C.sha(output/'INPUTS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        before_new_refined_predictions_and_metrics=True))
    print(dict(prepared=238,model_digest=model['model_digest_sha256'],identical_cached_input=True),flush=True)


def predict(output=OUTPUT, workers=3):
    output=Path(output).resolve();protocol,seal=_inputs(output)
    if (output/'PREDICTION_SEAL.json').exists():raise FileExistsError('Never overwrite completed predictions')
    if (output/'PREDICTIONS.jsonl.gz').exists():raise FileExistsError('Preserve existing journal without a seal')
    tasks=list(C.rows(output/'INPUTS.jsonl.gz'));model=BASE._verified_model()
    if [t['id'] for t in tasks]!=protocol['ids']:raise ValueError('Exact ordered238 population required')
    done={};started=time.monotonic();protocol_sha=C.sha(output/'PROTOCOL.json')
    for task in tasks:
        path=output/'frames'/(task['id'].replace(':','__')+'.json')
        if path.exists():
            row=C.read(path)
            if row.get('protocol_sha256')!=protocol_sha or row['input_sha256']!=E._digest(task) or set(row['arms'])!=set(ARMS):
                raise ValueError('Checkpoint source/input/arm binding mismatch')
            done[task['id']]=row
    pending=[t for t in tasks if t['id'] not in done]
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'))
    futures={pool.submit(_run,t,model):t['id'] for t in pending}
    try:
        for future in as_completed(futures):
            row=future.result();row['protocol_sha256']=protocol_sha;done[row['id']]=row
            C.write(output/'frames'/(row['id'].replace(':','__')+'.json'),row)
            C.write(output/'LIVE_PROGRESS.json',dict(completed=len(done),total=238,latest_id=row['id'],elapsed_seconds=time.monotonic()-started))
            print(f'{len(done)}/238 {row["id"]}',flush=True)
    except BaseException as error:
        C.write(output/'FAILED_BEFORE_SEAL.json',dict(completed=len(done),total=238,
            error_type=type(error).__name__,reason=str(error),prediction_seal_written=False,new_reference_reads=0))
        for future in futures:future.cancel()
        for process in list(getattr(pool,'_processes',{}).values()):process.terminate()
        pool.shutdown(wait=False,cancel_futures=True);raise
    else:pool.shutdown(wait=True)
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    C.save_rows(output/'PREDICTIONS.jsonl.gz',[done[t['id']] for t in tasks])
    C.write(output/'PREDICTION_SEAL.json',dict(rows=238,CAL=46,TEST=49,REMAINDER=143,severe=0,
        predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),protocol_sha256=protocol_sha,
        input_seal_sha256=C.sha(output/'INPUT_SEAL.json'),all3_arms_complete=True,
        before_new_control_metric_read=True,ordered_ids=protocol['ids'],elapsed_seconds=time.monotonic()-started))


def score(output=OUTPUT):
    scorer=types.FunctionType(BASE.ARCHIVE.score.__code__,{**BASE.ARCHIVE.score.__globals__,'ARMS':ARMS},BASE.ARCHIVE.score.__name__)
    return scorer(Path(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--cached-folder',type=Path,default=BASE.OUTPUT)
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    if args.command=='prepare':prepare(args.output,args.cached_folder)
    elif args.command=='predict':predict(args.output,args.workers)
    else:score(args.output)
