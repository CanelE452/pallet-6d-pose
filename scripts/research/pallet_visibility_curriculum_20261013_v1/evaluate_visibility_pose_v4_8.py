"""Whole-population V47/no-prior paired pose evaluation from frozen observations.

Only prior_weight changes to zero. V47 coordinates, original CAL effective
covariances/reliability, partial B lines/tangent intervals, every OT/U region,
states, K, native XYZ/Q, starts and all search settings remain identical.
No image/mask/state/correction forward, calibration or label access occurs in
predict. All CAL46+TEST49 predictions are sealed before separate scoring.
This is an iterative reused TEST experiment, not a new unseen validation set.
"""
from __future__ import annotations

import os
for _setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_setting] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_u_regions_v4_5 as P
from . import evaluate_shared_edge_partial_b_v4_7 as V47
from .boundary_partial_observations_v4_4 import fixed_fit_measurements
from .statewise_partial_boundary_u_solver_v4_6 import solve_boundary_u_region_joint_ot_pnp
from .statewise_uncertainty_calibration_v4 import _reference_alignment
from .scoring import pose_metric, distribution


ARMS = ('SAME_V47', 'NO_PRIOR_FROZEN_WEIGHTS')
OUTPUT = P.BOUNDARY / 'no_prior_pose_eval_v4_8'


def _load():
    originals, _, model, _, _, old_bindings, _ = P._inputs()
    original = {r['id']: r for r in originals}
    directory = V47.OUTPUT
    protocol_path = directory / 'PROTOCOL.json'
    protocol = C.read(protocol_path)
    E._verify_sources(protocol['source_sha256'])
    E._verify_sources(protocol['input_sha256'])
    E._verify_sources(protocol['protected_user_sha256'])
    complete_path = directory / 'PREDICTION_COMPLETE.json'
    complete = C.read(complete_path)
    if complete.get('rows') != 95 or complete.get('severe') != 0:
        raise ValueError('A complete sealed V47 nonsevere95 run is required')
    config = deepcopy(protocol['solver_configuration'])
    if config['prior_weight'] != 1. or config['bias_strength'] != 1.:
        raise ValueError('Paired V47 prior/mean binding changed')
    bindings = dict(old_bindings)
    bindings.update(protocol['input_sha256'])
    for path in (protocol_path, complete_path):
        bindings[str(path.relative_to(C.ROOT))] = C.sha(path)
    rows = []
    for partition, count in (('CAL', 46), ('TEST', 49)):
        journal = directory / f'{partition}_PREDICTIONS.jsonl.gz'
        seal_path = directory / f'{partition}_PREDICTION_SEAL.json'
        seal = C.read(seal_path)
        if (seal.get('rows') != count or seal.get('before_new_annotation_score') is not True
                or C.sha(journal) != seal['predictions_sha256']
                or C.sha(protocol_path) != seal['protocol_sha256']):
            raise ValueError('V47 population/prediction seal changed')
        E._verify_sources(seal['source_sha256']); E._verify_sources(seal['input_sha256'])
        values = list(C.rows(journal))
        expected = [r['id'] for r in originals if r['partition'] == partition]
        if [r['id'] for r in values] != expected or len(set(expected)) != count:
            raise ValueError('Exact ordered whole-population V47 records required')
        for path in (journal, seal_path):
            bindings[str(path.relative_to(C.ROOT))] = C.sha(path)
        for row in values:
            source = original[row['id']]['runtime']
            fit = fixed_fit_measurements(source['calibrated_visible_measurements'], row['correction'])
            if (row['partition'] != partition or row['cohort'] != 'PRIMARY_NONSEVERE'
                    or row['states'] != source['states']
                    or row['runtime']['initial_geometry'] != source['initial_geometry']
                    or row['image_sha256'] != original[row['id']]['image_sha256']
                    or row['frozen_BI_fit'] != fit):
                raise ValueError('V47 frozen observation/geometry binding changed: ' + row['id'])
            hidden = {k for k, flag in enumerate(source['initial_geometry']['initial_self_hidden']) if flag}
            partial = row['partial_boundary_observation']['boundary_observations']
            regions = row['u_region_observation']['u_regions']
            if (any(m['native_corner'] in hidden for m in fit)
                    or any(source['states'][r['native_corner']] != 'B' or r['native_corner'] in hidden for r in partial)
                    or any(source['states'][r['native_corner']] != 'U' or r['native_corner'] in hidden for r in regions)):
                raise ValueError('Frozen S/B/U factor contract changed')
            baseline = row['arms'][V47.ARMS[2]]
            if baseline['states'] != row['states'] or baseline['method'] != V47.ARMS[2]:
                raise ValueError('The paired cached V47 arm changed')
            row['_original_source'] = source
            rows.append(row)
    if len(rows) != 95 or set(original) != {r['id'] for r in rows}:
        raise ValueError('All95 observations including all fallbacks required')
    return rows, model, config, bindings, protocol


def _spec(row, config):
    spec = E._physics_spec(row['_original_source'], row['frozen_BI_fit'], config)
    return {**spec,
            'boundary_observations': deepcopy(row['partial_boundary_observation']['boundary_observations']),
            'u_regions': deepcopy(row['u_region_observation']['u_regions']),
            'boundary_factor_weight': 1.}


def _only_prior_changed(baseline, no_prior):
    restored = deepcopy(no_prior)
    restored['config']['prior_weight'] = baseline['config']['prior_weight']
    return restored == baseline and no_prior['config']['prior_weight'] == 0.


def _run(task):
    row, model, config = task
    started = time.monotonic()
    baseline_spec = _spec(row, config)
    no_prior_spec = deepcopy(baseline_spec)
    no_prior_spec['config']['prior_weight'] = 0.
    if not _only_prior_changed(baseline_spec, no_prior_spec):
        raise ValueError('Only prior_weight may change')
    immutable = E._digest(baseline_spec)
    solution = solve_boundary_u_region_joint_ot_pnp(**no_prior_spec)
    if E._digest(baseline_spec) != immutable:
        raise ValueError('The solver modified immutable paired observations')
    if solution['states'] != row['states']:
        raise ValueError('No-prior solver changed states')
    if solution.get('prior_audit', {}).get('enabled') or solution.get('prior_audit', {}).get('bias_active_in_loss'):
        raise ValueError('Prior/bias remains physically active in zero-weight arm')
    regions = row['u_region_observation']['u_regions']
    partial = row['partial_boundary_observation']
    selected = V47._finish(row['_original_source'], row['frozen_BI_fit'], model,
        solution, regions, ARMS[1], partial)
    # SAME_V47 is the exact sealed cached arm, including honest fallbacks and
    # ambiguity. Its original method tag stays intact for bytewise parity.
    baseline = deepcopy(row['arms'][V47.ARMS[2]])
    return C.finite(dict(id=row['id'], partition=row['partition'], session=row['session'],
        cohort=row['cohort'], image=row['image'], image_sha256=row['image_sha256'], states=row['states'],
        runtime=row['runtime'], frozen_BI_fit=row['frozen_BI_fit'],
        partial_boundary_observation=partial, u_region_observation=row['u_region_observation'],
        arms={ARMS[0]: baseline, ARMS[1]: selected},
        baseline_mathematical_input_sha256=immutable,
        no_prior_mathematical_input_sha256=E._digest(no_prior_spec),
        contract=dict(only_prior_weight_changed=True, prior_weight_before=1., prior_weight_after=0.,
            CAL_fit_covariance_and_reliability_fixed=True, partial_B_lines_and_tangent_ranges_fixed=True,
            all_OT_U_native_K_Q_states_raw_inputs_fixed=True, no_conditional_covariance_as_total_noise=True,
            source_baseline_arm='SHARED_EDGE_PARTIAL_B', S_image_queries=0, U_image_queries=0,
            backbone_mask_state_or_line_forwards=0, GT_inputs_to_prediction=False),
        elapsed_seconds=time.monotonic()-started, pose_solver_calls=1))


def predict(output=OUTPUT, workers=3):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve previous evaluation: ' + str(output))
    if not isinstance(workers, int) or not 1 <= workers <= 3:
        raise ValueError('One to three CPU1 workers required')
    rows, model, config, bindings, parent = _load()
    sources = deepcopy(parent['source_sha256'])
    sources[str(Path(__file__).relative_to(C.ROOT))] = C.sha(__file__)
    protected = parent['protected_user_sha256']
    output.mkdir(parents=True)
    protocol = dict(schema='WHOLE_NONSEVERE_NO_PRIOR_PAIRED_POSE_V4_8', arms=list(ARMS),
        primary_population={'CAL':46, 'TEST':49, 'severe':0}, ids=[r['id'] for r in rows],
        source_sha256=sources, input_sha256=bindings, protected_user_sha256=protected,
        baseline_solver_configuration=config,
        no_prior_solver_configuration={**config, 'prior_weight':0.},
        only_solver_change='prior_weight=0; prior mean/bias is physically inactive',
        all95_fallbacks_and_incompatible_frames_kept=True, no_frame_ID_or_GT_oracle_routing=True,
        paired_baseline_cached_from=str(V47.OUTPUT.relative_to(C.ROOT)),
        baseline_arm='SHARED_EDGE_PARTIAL_B', fixed_original_CAL_total_error_weights=True,
        partial_B_normal_tangent_OT_U_constraints_identical=True,
        S_image_and_region_terms=0, U_image_or_point_cost_terms=0,
        observations_reused=True, new_model_mask_RGB_or_line_inference=0,
        labels_read_only_after_complete95_prediction_seal=True,
        TEST_status='REUSED_ITERATIVE_EXPLORATORY_NOT_UNSEEN',
        hypothesis_motivated_by_previously_scored_case=True,
        new_noise_model_or_CAL_fit=False, workers=workers, CPU_threads_per_worker=1)
    C.write(output/'PROTOCOL.json', protocol)
    for relative, digest in sources.items():
        dest = output/'code_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT/relative, dest)
        if C.sha(dest) != digest:
            raise ValueError('Source snapshot mismatch')
    started = time.monotonic(); completed = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        futures = {pool.submit(_run, (r, model, config)):r['id'] for r in rows}
        for future in as_completed(futures):
            value = future.result(); completed[value['id']] = value
            C.write(output/f"{value['partition']}_frames"/(value['id'].replace(':','__')+'.json'),value)
            C.write(output/'LIVE_PROGRESS.json',dict(completed=len(completed),total=95,
                elapsed_seconds=time.monotonic()-started,latest_id=value['id']))
            print(f"{len(completed)}/95 {value['id']} accepted={value['arms'][ARMS[1]]['accepted']}",flush=True)
    E._verify_sources(sources);E._verify_sources(bindings);E._verify_sources(protected)
    seals = {}
    for partition, count in (('CAL',46),('TEST',49)):
        population=[r for r in rows if r['partition']==partition]
        journal=output/f'{partition}_PREDICTIONS.jsonl.gz'
        C.save_rows(journal,[completed[r['id']] for r in population])
        if len(population)!=count:
            raise ValueError('Fixed population count changed')
        seal=dict(schema='WHOLE_NO_PRIOR_PREDICTION_SEAL_V4_8',rows=count,
            protocol_sha256=C.sha(output/'PROTOCOL.json'),predictions_sha256=C.sha(journal),
            source_sha256=sources,input_sha256=bindings,before_new_annotation_score=True,
            labels_read_in_prediction=False,all95_finished_before_partition_seal=True)
        C.write(output/f'{partition}_PREDICTION_SEAL.json',seal)
        seals[partition]=C.sha(output/f'{partition}_PREDICTION_SEAL.json')
    C.write(output/'PREDICTION_COMPLETE.json',dict(rows=95,CAL=46,TEST=49,severe=0,
        ordered_partition_prediction_seal_sha256=seals,elapsed_seconds=time.monotonic()-started,
        before_any_new_pose_scoring=True,pose_solver_calls=95,baseline_poses_reused=95))


def _summary(values,arm):
    result=V47._summary(values,arm)
    result['translation_le_5cm_count']=sum(v['metrics'][arm]['translation_cm']<=5. for v in values)
    result['all_translation_le_5cm']=all(v['metrics'][arm]['translation_cm']<=5. for v in values)
    result['translation_over_5cm_rows']=[dict(id=v['id'],translation_cm=v['metrics'][arm]['translation_cm'],
        rotation_deg=v['metrics'][arm]['rotation_deg'],accepted=v['accepted'][arm],ambiguous=v['ambiguous'][arm])
        for v in sorted(values,key=lambda v:v['metrics'][arm]['translation_cm'],reverse=True)
        if v['metrics'][arm]['translation_cm']>5.]
    return result


def score(output=OUTPUT):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json')
    complete=C.read(output/'PREDICTION_COMPLETE.json')
    if complete['rows']!=95 or not complete['before_any_new_pose_scoring']:
        raise ValueError('Seal every CAL46/TEST49 paired prediction before scoring')
    E._verify_sources(protocol['source_sha256']);E._verify_sources(protocol['input_sha256'])
    E._verify_sources(protocol['protected_user_sha256'])
    predictions={}
    # Verify BOTH complete partition seals and parent pose parity before the
    # first coordinate/pose label is opened, not one partition at a time.
    for partition,count in (('CAL',46),('TEST',49)):
        seal_path=output/f'{partition}_PREDICTION_SEAL.json';seal=C.read(seal_path)
        journal=output/f'{partition}_PREDICTIONS.jsonl.gz'
        if (seal['rows']!=count or not seal['before_new_annotation_score']
                or C.sha(journal)!=seal['predictions_sha256']
                or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']
                or C.sha(seal_path)!=complete['ordered_partition_prediction_seal_sha256'][partition]):
            raise ValueError('Complete prediction seal changed')
        values=list(C.rows(journal));parents={r['id']:r for r in C.rows(V47.OUTPUT/f'{partition}_PREDICTIONS.jsonl.gz')}
        expected=[i for i in protocol['ids'] if i in parents]
        if [r['id'] for r in values]!=expected or len(values)!=count:
            raise ValueError('Complete ordered all-frame score population changed')
        if any(r['arms'][ARMS[0]]!=parents[r['id']]['arms'][V47.ARMS[2]] for r in values):
            raise ValueError('Cached SAME_V47 arm differs from sealed parent')
        predictions[partition]=values
    reports={}
    for partition in ('CAL','TEST'):
        target_path=E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if partition=='CAL' else 'test_reused_v1/ISOLATED_TEST_SCORING.json')
        labels=C.read(target_path)['rows'];labels=labels if isinstance(labels,dict) else {r['id']:r for r in labels}
        values=[]
        for row in predictions[partition]:
            label=labels[row['id']];alignment=_reference_alignment(row,label)
            metrics={arm:pose_metric(row['arms'][arm]['actual_pose'],label['target']['truth']) for arm in ARMS}
            baseline,new=[row['arms'][arm]['actual_pose'] for arm in ARMS]
            values.append(dict(id=row['id'],image=row['image'],states=row['states'],metrics=metrics,
                accepted={arm:row['arms'][arm]['accepted'] for arm in ARMS},
                ambiguous={arm:row['arms'][arm]['ambiguity_audit']['material_ambiguity'] for arm in ARMS},
                physical_native_compatible=bool(alignment['compatible']),geometry_gap_m=float(alignment['geometry_gap_m']),
                partial_B_ids=[b['native_corner'] for b in row['partial_boundary_observation']['boundary_observations']],
                no_prior_changes_selected_pose=baseline['R_cf']!=new['R_cf'] or baseline['centroid']!=new['centroid']))
        report=dict(schema='WHOLE_NO_PRIOR_POSTSEAL_POSE_SCORE_V4_8',partition=partition,severe=0,
            all_frames_include_fallbacks=True,summary_all={a:_summary(values,a) for a in ARMS},
            summary_reference_native_compatible={a:_summary([v for v in values if v['physical_native_compatible']],a) for a in ARMS},
            rows=values,target_binding=E._binding(target_path),source_prediction_seal=E._binding(output/f'{partition}_PREDICTION_SEAL.json'),
            TEST_is_reused_iterative_exploratory=True,scorer_uses_same_derived_physical_C2_reference=True)
        C.write(output/f'{partition}_POSE_SCORE.json',report);reports[partition]=report
    E._verify_sources(protocol['source_sha256']);E._verify_sources(protocol['input_sha256'])
    E._verify_sources(protocol['protected_user_sha256'])
    C.write(output/'SCORE_SEAL.json',dict(schema='WHOLE_NO_PRIOR_SCORE_SEAL_V4_8',
        CAL_pose_score_sha256=C.sha(output/'CAL_POSE_SCORE.json'),TEST_pose_score_sha256=C.sha(output/'TEST_POSE_SCORE.json'),
        prediction_complete_sha256=C.sha(output/'PREDICTION_COMPLETE.json'),
        TEST49_translation_le_5cm_goal_pass=reports['TEST']['summary_all'][ARMS[1]]['all_translation_le_5cm'],
        no_GT_or_per_frame_oracle_routing=True,all_source_input_protected_checks_pass=True))
    print(C.finite({p:r['summary_all'] for p,r in reports.items()}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--stage',choices=('predict','score','all','check'),default='all')
    args=parser.parse_args()
    if args.stage=='check':
        rows,model,config,bindings,parent=_load()
        for row in rows:
            old=_spec(row,config);new=deepcopy(old);new['config']['prior_weight']=0.
            if not _only_prior_changed(old,new):raise ValueError('Prior-only preflight failed')
        print('PREFLIGHT_PASS rows=95 CAL=46 TEST=49 only_prior_weight_changes source=' + C.sha(__file__),flush=True)
    elif args.stage=='predict':predict(args.output,args.workers)
    elif args.stage=='score':score(args.output)
    else:predict(args.output,args.workers);score(args.output)


if __name__=='__main__':main()
