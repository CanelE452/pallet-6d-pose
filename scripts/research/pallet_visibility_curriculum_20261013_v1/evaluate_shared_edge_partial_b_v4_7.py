"""Frozen-upstream corner-reach and shared-edge B paired evaluation.

Predictions are sealed before annotation scoring. V4.6 successful B/I,
states, geometry, O/T domains, U regions and CAL fit scales remain fixed.
This is an exploratory rerun of the existing nonsevere population.
"""
from __future__ import annotations

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

import cv2
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_u_regions_v4_5 as PARENT
from . import statewise_uncertain_solver_v4_bias as BASE
from . import visible_bi_corner_reach_v4_7 as RECOVER
from .boundary_partial_observations_v4_4 import fixed_fit_measurements
from .boundary_shared_edge_observations_v4_7 import observe_shared_supported_b_lines
from .statewise_partial_boundary_u_solver_v4_6 import solve_boundary_u_region_joint_ot_pnp
from .statewise_u_region_solver_v4_5 import (
    solve_uncertain_u_region_joint_ot_pnp, reuse_parent_solution_if_u_feasible)
from .uncertain_u_regions_v4_5 import build_u_regions
from .statewise_uncertainty_calibration_v4 import _reference_alignment
from .scoring import pose_metric, distribution


ARMS = ('CURRENT_V4_5', 'CORNER_REACH_COMPLETE_B', 'SHARED_EDGE_PARTIAL_B')
OUTPUT = PARENT.BOUNDARY / 'shared_edge_partial_b_eval_v4_7'
PREVIOUS = PARENT.BOUNDARY / 'full_mask_partial_b_eval_v4_6'
TARGET = 'eval_pallet07:1778652166837872128'


def _finish(source, fit, model, solution, regions, arm, partial=None):
    result = PARENT._finish(source, fit, model, solution, regions, PARENT.ARMS[1])
    result['solution'] = E._compact_solution(solution)
    result['method'] = arm
    if partial:
        result['partial_boundary_observation'] = deepcopy(partial)
        readout = result['keypoint_readout']
        readout['partial_B_observations'] = deepcopy(partial['boundary_observations'])
        for observation in partial['boundary_observations']:
            k = observation['native_corner']
            # A line and an interval are not an observed complete 2D corner.
            readout['reported_points'][k] = None
            readout['point_sources'][k] = 'PARTIAL_B_NORMAL_AND_TANGENT_INTERVAL'
        readout['BI_observations_preserved'] = not bool(partial['boundary_observations'])
        readout['original_BI_reference_measurements_preserved'] = True
    return result


def _run(task):
    row, old_correction, parent, meta, mask_dir, model, config, region_model = task
    started = time.monotonic()
    source = row['runtime']; geometry = source['initial_geometry']
    rgb_path = C.ROOT / row['image']
    if C.sha(rgb_path) != row['image_sha256'] or meta['image_sha256'] != row['image_sha256']:
        raise ValueError('Actual RGB binding changed')
    mask_path = Path(mask_dir) / meta['mask_npz']
    if C.sha(mask_path) != meta['mask_file_sha256']:
        raise ValueError('New SAM mask binding changed')
    image = cv2.imread(str(rgb_path))
    with np.load(mask_path) as masks:
        mask = masks['pallet_mask_clean'].copy()
    base = deepcopy(old_correction)
    raw = deepcopy(base['raw_points'])
    for k, hidden in enumerate(geometry['initial_self_hidden']):
        if hidden and raw[k] != [-1., -1.]:
            raise ValueError('S raw must already be absent before correction')
    initial = dict(R_cf=geometry['R_cf'], centroid=geometry['centroid'],
                   cf_extents=geometry['cf_extents'])
    correction = RECOVER.recover_visible_boundary(
        image, raw, geometry['K'], initial, source['states'], pallet_mask=mask,
        base_correction=base, native_xyz=geometry['native_xyz'])
    before = {m['native_corner']: m for m in base['visible_measurements']}
    after = {m['native_corner']: m for m in correction['visible_measurements']}
    records = {r['native_corner']: r for r in base['records']}
    eligible = {k for k in before if source['states'][k] == 'B' and not records[k]['corrected']}
    changed = {k for k in before if before[k] != after[k]}
    if not changed <= eligible or set(before) != set(after):
        raise ValueError('Existing successful B/I changed')
    if any(source['states'][k] != 'B' for k in correction['recovery_queried_native_ids']):
        raise ValueError('Only B recovery image queries allowed')
    fit = fixed_fit_measurements(source['calibrated_visible_measurements'], correction)
    spec = E._physics_spec(source, fit, config)
    old_fit = fixed_fit_measurements(source['calibrated_visible_measurements'], base)
    old_spec = E._physics_spec(source, old_fit, config)
    regions = build_u_regions(source, region_model)
    partial = observe_shared_supported_b_lines(
        correction, geometry, source['states'], fit, localization_model=region_model,
        nonself_scale_px=source['ot_domains']['nonself_prediction_scale_px'],
        image_binding=dict(sha256=row['image_sha256'], image_hw=geometry['image_hw']))
    same_math = E._digest(PARENT.mathematical_spec(spec)) == E._digest(PARENT.mathematical_spec(old_spec))
    if same_math:
        complete = deepcopy(parent['previous_full_mask_v4_6']['arms']['FULL_MASK_COMPLETE_B']['solution'])
    else:
        no_u = BASE.solve_uncertain_joint_ot_pnp(**spec)
        complete = reuse_parent_solution_if_u_feasible(
            spec['xyz'], spec['K'], spec['states'], no_u, regions['u_regions'], config=config)
        if complete is None:
            complete = solve_uncertain_u_region_joint_ot_pnp(**spec, u_regions=regions['u_regions'])
    complete.pop('parent_baseline_solution', None)
    if partial['boundary_observations']:
        selected = solve_boundary_u_region_joint_ot_pnp(
            **spec, boundary_observations=partial['boundary_observations'],
            u_regions=regions['u_regions'], boundary_factor_weight=1.)
        selected.pop('parent_baseline_solution', None)
    else:
        selected = deepcopy(complete)
    arms = {ARMS[0]: deepcopy(parent['arms'][PARENT.ARMS[1]])}
    arms[ARMS[0]]['method'] = ARMS[0]
    arms[ARMS[1]] = _finish(source, fit, model, complete, regions['u_regions'], ARMS[1])
    arms[ARMS[2]] = _finish(source, fit, model, selected, regions['u_regions'], ARMS[2], partial)
    return C.finite(dict(
        id=row['id'], partition=row['partition'], session=row['session'], cohort=row['cohort'],
        image=row['image'], image_sha256=row['image_sha256'], states=source['states'],
        runtime={'initial_geometry': geometry}, arms=arms,
        old_correction=base, correction=correction, frozen_BI_fit=fit,
        partial_boundary_observation=partial, u_region_observation=regions,
        cache_audit=dict(complete_solver_math_equal_to_parent=same_math),
        contract=dict(existing_successful_B_and_all_I_exact=True,
            changed_complete_B_ids=sorted(changed), eligible_fallback_B_ids=sorted(eligible),
            S_image_queries=0, U_image_queries=0, U_exact_point_costs=0,
            state_gate_geometry_OT_Q_scale_fixed=True, CAL_weights_refitted=False,
            GT_inputs_to_inference=False), elapsed_seconds=time.monotonic()-started))


def _load(mask_dir):
    rows, corrections, model, config, region_model, bindings, old_protocol = PARENT._inputs()
    parents = {}
    for partition in ('CAL', 'TEST'):
        seal_path = PARENT.OUTPUT / f'{partition}_PREDICTION_SEAL.json'
        seal = C.read(seal_path)
        journal = PARENT.OUTPUT / f'{partition}_PREDICTIONS.jsonl.gz'
        protocol = PARENT.OUTPUT / 'PROTOCOL.json'
        if C.sha(journal) != seal['predictions_sha256'] or C.sha(protocol) != seal['protocol_sha256']:
            raise ValueError('Previous latest-U evaluation binding changed')
        E._verify_sources(seal['source_sha256']); E._verify_sources(seal['input_sha256'])
        values = list(C.rows(journal))
        if {r['id'] for r in values} != {r['id'] for r in rows if r['partition'] == partition}:
            raise ValueError('Paired parent frame population changed')
        parents.update({r['id']: r for r in values})
        for path in (seal_path, journal, protocol):
            bindings[str(path.relative_to(C.ROOT))] = C.sha(path)
    mask_dir = Path(mask_dir).resolve()
    seal = C.read(mask_dir / 'SEAL.json')
    if seal['records'] != 95 or seal['GT_or_pose_labels_read']:
        raise ValueError('Full population annotation-free new mask seal required')
    E._verify_sources(seal['source_sha256']); E._verify_sources(seal['input_sha256'])
    for name, key in (('RECORDS.json', 'records_sha256'), ('PROTOCOL.json', 'protocol_sha256')):
        path = mask_dir / name
        if C.sha(path) != seal[key]: raise ValueError('Mask cache seal changed')
        bindings[str(path.relative_to(C.ROOT))] = C.sha(path)
    meta = {r['id']: r for r in C.read(mask_dir / 'RECORDS.json')}
    if set(meta) != {r['id'] for r in rows}: raise ValueError('SAM population changed')
    bindings[str((mask_dir / 'SEAL.json').relative_to(C.ROOT))] = C.sha(mask_dir / 'SEAL.json')
    for rec in meta.values():
        path = mask_dir / rec['mask_npz']
        if C.sha(path) != rec['mask_file_sha256']: raise ValueError('SAM evidence changed')
        bindings[str(path.relative_to(C.ROOT))] = rec['mask_file_sha256']
    for row in rows:
        source = row['runtime']; parent = parents[row['id']]
        expected_fit = fixed_fit_measurements(source['calibrated_visible_measurements'], corrections[row['id']])
        if (parent['states'] != source['states']
                or parent['runtime']['initial_geometry'] != source['initial_geometry']
                or parent['frozen_BI_fit'] != expected_fit
                or parent['u_region_observation'] != build_u_regions(source, region_model)):
            raise ValueError('Paired cached parent mathematical inputs differ: ' + row['id'])
    for partition in ('CAL', 'TEST'):
        seal_path = PREVIOUS / f'{partition}_PREDICTION_SEAL.json'
        seal = C.read(seal_path)
        journal = PREVIOUS / f'{partition}_PREDICTIONS.jsonl.gz'
        protocol_path = PREVIOUS / 'PROTOCOL.json'
        if C.sha(journal) != seal['predictions_sha256'] or C.sha(protocol_path) != seal['protocol_sha256']:
            raise ValueError('Previous full-mask evaluation seal changed')
        E._verify_sources(seal['source_sha256']); E._verify_sources(seal['input_sha256'])
        for path in (seal_path, journal, protocol_path):
            bindings[str(path.relative_to(C.ROOT))] = C.sha(path)
        values = list(C.rows(journal))
        expected = {r['id'] for r in rows if r['partition'] == partition}
        if {r['id'] for r in values} != expected:
            raise ValueError('Previous full-mask population changed')
        for value in values:
            source = next(r['runtime'] for r in rows if r['id'] == value['id'])
            if (value['states'] != source['states']
                    or value['runtime']['initial_geometry'] != source['initial_geometry']
                    or value['frozen_BI_fit'] != fixed_fit_measurements(source['calibrated_visible_measurements'], value['correction'])
                    or value['u_region_observation'] != build_u_regions(source, region_model)):
                raise ValueError('Previous full-mask mathematical binding changed')
            parents[value['id']]['previous_full_mask_v4_6'] = value
            corrections[value['id']] = value['correction']
    return rows, corrections, parents, meta, model, config, region_model, bindings, old_protocol


def predict(output, mask_dir, workers):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve previous result: ' + str(output))
    rows, corrections, parents, meta, model, config, region_model, bindings, old_protocol = _load(mask_dir)
    # Bind the sealed parent dependency set plus the new inference modules.
    # Independent report rendering/tests may continue without changing fits.
    sources = deepcopy(C.read(PARENT.OUTPUT / 'PROTOCOL.json')['source_sha256'])
    for name in (Path(__file__).name, 'boundary_partial_observations_v4_6.py',
                 'statewise_partial_boundary_u_solver_v4_6.py',
                 'capture_boundary_recovery_masks_v4_6.py',
                 'visible_bi_corner_reach_v4_7.py',
                 'boundary_shared_edge_observations_v4_7.py'):
        path = Path(__file__).parent / name
        sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    protected = old_protocol['protected_user_sha256']; E._verify_sources(protected)
    output.mkdir(parents=True)
    protocol = dict(schema='CORNER_REACH_SHARED_B_PAIRED_V4_7', arms=list(ARMS),
        primary_population={'CAL': 46, 'TEST': 49, 'severe': 0}, ids=[r['id'] for r in rows],
        source_sha256=sources, input_sha256=bindings, protected_user_sha256=protected,
        mask_directory=str(Path(mask_dir).resolve().relative_to(C.ROOT)), solver_configuration=config,
        original_successful_BI_unchanged=True, initial_states_fixed=True,
        B_single_line_policy='replace raw B point likelihood with observed normal likelihood; tangent interval only',
        partial_B_error_budget_is_borrowed_CAL_not_new_calibration=True,
        U_policy='frozen v4.5 range only; no image search, seed or exact point term',
        S_policy='initial S gate before any local query; final rigid PnP projection only',
        mask_forward_calls_per_frame=0, mask_cache_reused=True, backbone_state_training_forwards=0,
        corner_search_floor='old along_end36 + old boundary_context7; remaining guards unchanged',
        candidate_pair_policy='finite observed support guard before best sine selection',
        shared_edge_policy='validated B-to-B RGB line constrains both eligible endpoints; no new KP invented',
        shared_line_correlation_budget_is_engineering_inflation_not_calibrated=True,
        TEST_parameter_selection=False, TEST_is_reused_exploratory=True,
        annotations_read_only_after_all_predictions_sealed=True, workers=workers)
    C.write(output / 'PROTOCOL.json', protocol)
    for relative, digest in sources.items():
        destination = output / 'code_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT / relative, destination)
        if C.sha(destination) != digest: raise ValueError('Source snapshot mismatch')
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        for partition in ('CAL', 'TEST'):
            population = [r for r in rows if r['partition'] == partition]
            futures = {pool.submit(_run, (r, corrections[r['id']], parents[r['id']], meta[r['id']],
                       str(mask_dir), model, config, region_model)): r['id'] for r in population}
            completed = {}
            for future in as_completed(futures):
                value = future.result(); completed[value['id']] = value
                C.write(output / f'{partition}_frames' / (value['id'].replace(':', '__') + '.json'), value)
                C.write(output / 'LIVE_PROGRESS.json', dict(partition=partition, completed=len(completed),
                    total=len(population), elapsed_seconds=time.monotonic()-started))
                print(f"{partition} {len(completed)}/{len(population)} {value['id']} complete={value['contract']['changed_complete_B_ids']} partial={[x['native_corner'] for x in value['partial_boundary_observation']['boundary_observations']]}", flush=True)
            journal = output / f'{partition}_PREDICTIONS.jsonl.gz'
            C.save_rows(journal, [completed[r['id']] for r in population])
            E._verify_sources(sources); E._verify_sources(bindings); E._verify_sources(protected)
            C.write(output / f'{partition}_PREDICTION_SEAL.json', dict(rows=len(population),
                protocol_sha256=C.sha(output / 'PROTOCOL.json'), predictions_sha256=C.sha(journal),
                source_sha256=sources, input_sha256=bindings, before_new_annotation_score=True,
                GT_inputs_to_inference=False))
    C.write(output / 'PREDICTION_COMPLETE.json', dict(rows=95, CAL=46, TEST=49, severe=0,
        elapsed_seconds=time.monotonic()-started))


def _summary(values, arm):
    metrics = [r['metrics'][arm] for r in values]
    result = {key: distribution([m[key] for m in metrics])
              for key in ('translation_cm', 'rotation_deg', 'ADDsym_cm')}
    result.update(frames=len(values), successes_5cm_5deg=sum(m['success_5cm_5deg'] for m in metrics),
        fallback_count=sum(not r['accepted'][arm] for r in values),
        ambiguous_count=sum(r['ambiguous'][arm] for r in values))
    result['maximum_examples'] = {key: {'id': max(values, key=lambda r: r['metrics'][arm][key])['id']}
        for key in ('translation_cm', 'rotation_deg', 'ADDsym_cm')} if values else {}
    return result


def score(output):
    output = Path(output).resolve(); protocol = C.read(output / 'PROTOCOL.json')
    if not (output / 'PREDICTION_COMPLETE.json').exists(): raise ValueError('Seal all predictions before scoring')
    E._verify_sources(protocol['source_sha256']); E._verify_sources(protocol['input_sha256'])
    E._verify_sources(protocol['protected_user_sha256'])
    reports = {}
    for partition in ('CAL', 'TEST'):
        journal = output / f'{partition}_PREDICTIONS.jsonl.gz'
        seal = C.read(output / f'{partition}_PREDICTION_SEAL.json')
        if C.sha(journal) != seal['predictions_sha256'] or C.sha(output / 'PROTOCOL.json') != seal['protocol_sha256']:
            raise ValueError('Prediction binding changed')
        path = E.COHORT / ('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if partition == 'CAL'
                          else 'test_reused_v1/ISOLATED_TEST_SCORING.json')
        labels = C.read(path)['rows']
        if isinstance(labels, list): labels = {r['id']: r for r in labels}
        values = []
        for row in C.rows(journal):
            label = labels[row['id']]; alignment = _reference_alignment(row, label)
            metrics = {arm: pose_metric(row['arms'][arm]['actual_pose'], label['target']['truth']) for arm in ARMS}
            a, b, c = [row['arms'][arm]['actual_pose'] for arm in ARMS]
            values.append(dict(id=row['id'], image=row['image'], states=row['states'], metrics=metrics,
                accepted={arm: row['arms'][arm]['accepted'] for arm in ARMS},
                ambiguous={arm: row['arms'][arm]['ambiguity_audit']['material_ambiguity'] for arm in ARMS},
                physical_native_compatible=bool(alignment['compatible']),
                geometry_gap_m=float(alignment['geometry_gap_m']),
                changed_complete_B_ids=row['contract']['changed_complete_B_ids'],
                partial_B_ids=[x['native_corner'] for x in row['partial_boundary_observation']['boundary_observations']],
                mask_changes_pose=a['R_cf'] != b['R_cf'] or a['centroid'] != b['centroid'],
                partial_changes_pose=b['R_cf'] != c['R_cf'] or b['centroid'] != c['centroid']))
        report = dict(partition=partition, severe=0, all_frames_include_fallbacks=True,
            summary_all={arm: _summary(values, arm) for arm in ARMS},
            summary_reference_native_compatible={arm: _summary([v for v in values if v['physical_native_compatible']], arm) for arm in ARMS},
            complete_B_changes=sum(len(v['changed_complete_B_ids']) for v in values),
            partial_B_count=sum(len(v['partial_B_ids']) for v in values),
            mask_changed_pose_count=sum(v['mask_changes_pose'] for v in values),
            partial_changed_pose_count=sum(v['partial_changes_pose'] for v in values),
            target=next((v for v in values if v['id'] == TARGET), None),
            TEST_is_reused_exploratory=True, target_binding=E._binding(path), rows=values)
        C.write(output / f'{partition}_POSE_SCORE.json', report); reports[partition] = report
    C.write(output / 'SCORE_SEAL.json', dict(
        CAL_pose_score_sha256=C.sha(output / 'CAL_POSE_SCORE.json'),
        TEST_pose_score_sha256=C.sha(output / 'TEST_POSE_SCORE.json'),
        annotations_used_for_inference_or_selection=False))
    print(C.finite({p: {'summary_all': r['summary_all'], 'complete_B_changes': r['complete_B_changes'],
        'partial_B_count': r['partial_B_count'], 'target': r['target']} for p, r in reports.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    if not args.score_only: predict(args.output, args.masks, args.workers)
    score(args.output)


if __name__ == '__main__': main()

