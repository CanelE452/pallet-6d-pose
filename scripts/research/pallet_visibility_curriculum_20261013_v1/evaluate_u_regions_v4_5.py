"""Paired frozen-upstream U-region pose evaluation, with post-seal scoring.

U is a broad positional region, never a local image query or point objective.
The conservative region radius is borrowed from the existing all-nonS CAL max
model. This is reused TEST analysis, not an independent generalization test.
"""
from __future__ import annotations

import os
for _thread_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_thread_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import numpy as np

from . import common as C
from . import evaluation_interval_v4_2b as OLD_RUN
from . import statewise_evaluation_v4 as E
from . import statewise_uncertain_solver_v4_bias as BASE
from .boundary_partial_observations_v4_4 import fixed_fit_measurements
from .scoring import distribution, pose_metric
from .statewise_joint_api_v4 import finish_observations
from .statewise_uncertainty_calibration_v4 import _reference_alignment
from .truncation_interval_v4_2 import load_truncation_interval_model
from .uncertain_u_regions_v4_5 import build_u_regions
from .statewise_u_region_solver_v4_5 import (
    solve_uncertain_u_region_joint_ot_pnp, reuse_parent_solution_if_u_feasible)


BOUNDARY = E.FIX / 'boundary_recovery_v1'
RECOVERY = BOUNDARY / 'evaluation_v3'
OUTPUT = BOUNDARY / 'u_region_pose_eval_v1'
REGION_MODEL = E.FIX / 'truncation_interval_model_CAL_v4_2/MODEL.json'
ARMS = ('LATEST_BI_NO_U_REGION', 'LATEST_BI_WITH_U_REGION')


def mathematical_spec(spec):
    """Hash all solver-read visible fields; exclude reporting-only metadata.

The other inputs are the identical complete original spec. Observed covariance
and protection are included even when not active in the current objective.
"""
    result = deepcopy(spec)
    keys = ('native_corner', 'xy', 'covariance_px2', 'protection_px',
            'fit_covariance_px2', 'fit_reliability_weight')
    result['visible'] = [
        {key: deepcopy(m[key]) for key in keys if key in m}
        for m in sorted(spec['visible'], key=lambda x: x['native_corner'])]
    return result


def _finish(source, fit, model, solution, regions, arm):
    observation = deepcopy(source)
    observation['frozen_visible_measurements'] = deepcopy([
        {k: deepcopy(v) for k, v in m.items()
         if k not in ('fit_covariance_px2', 'fit_reliability_weight', 'fixed_original_fit_scale')}
        for m in fit])
    finished = finish_observations(observation, solution, fit, model)
    # U remains unresolved; a projected model vertex is not a measured U point.
    for k, state in enumerate(finished['states']):
        if state == 'U':
            assert finished['keypoint_readout']['reported_points'][k] is None
            assert finished['keypoint_readout']['point_sources'][k] == 'UNRESOLVED'
        elif state == 'S':
            assert finished['keypoint_readout']['point_sources'][k] == 'FINAL_PNP_S_PROJECTION'
    if arm == ARMS[1]:
        finished['actual_pose']['fallback_pose_does_not_solve_U_regions'] = not finished['accepted']
        if regions and finished['accepted']:
            finished['actual_pose']['source'] = 'SOFT_BI_PRIOR_ALL_OT_U_REGION_JOINT_SE3'
    return {key: deepcopy(finished[key]) for key in (
        'actual_pose', 'accepted', 'states', 'keypoint_readout', 'ambiguity_audit', 'fallback_audit')}


def _run(task):
    row, correction, model, config, region_model = task
    source = row['runtime']
    fit = fixed_fit_measurements(source['calibrated_visible_measurements'], correction)
    old_spec = E._physics_spec(source, source['calibrated_visible_measurements'], config)
    if E._digest(old_spec) != row['all_OT_mathematical_input_sha256']:
        raise ValueError('Old cached solver mathematical binding changed: ' + row['id'])
    spec = E._physics_spec(source, fit, config)
    old_digest, new_digest = [E._digest(mathematical_spec(x)) for x in (old_spec, spec)]
    exact = old_digest == new_digest
    regions = build_u_regions(source, region_model)
    started = time.monotonic()
    if exact:
        baseline = deepcopy(source['joint_pose'])
        baseline_calls = 0
    else:
        baseline = BASE.solve_uncertain_joint_ot_pnp(**spec)
        baseline_calls = 1
    reused = reuse_parent_solution_if_u_feasible(
        spec['xyz'], spec['K'], spec['states'], baseline, regions['u_regions'], config=spec['config'])
    if reused is not None:
        selected = reused
        u_calls = 0
    else:
        selected = solve_uncertain_u_region_joint_ot_pnp(**spec, u_regions=regions['u_regions'])
        u_calls = 1
    # No duplication of the paired parent result inside the U arm journal.
    selected.pop('parent_baseline_solution', None)
    arms = {}
    for arm, solution in zip(ARMS, (baseline, selected)):
        arms[arm] = _finish(source, fit, model, solution, regions['u_regions'], arm)
        arms[arm]['solution'] = E._compact_solution(solution)
    return C.finite(dict(
        id=row['id'], partition=row['partition'], session=row['session'],
        cohort=row['cohort'], image=row['image'], image_sha256=row['image_sha256'],
        states=source['states'], runtime={'initial_geometry': source['initial_geometry']},
        frozen_BI_fit=fit, u_region_observation=regions, arms=arms,
        cache_audit=dict(parent_mathematical_input_equal=exact,
            old_math_sha256=old_digest, latest_math_sha256=new_digest,
            selected_U_reuses_feasible_parent=reused is not None,
            numerical_parent_solver_calls=baseline_calls, numerical_U_solver_calls=u_calls,
            comparison_holds_BI_xy_covariance_reliability_OT_prior_geometry_fixed=True),
        elapsed_seconds=time.monotonic()-started,
        S_image_queries=0, U_image_queries=0, U_exact_point_objective_terms=0,
        initial_state_gate_unchanged=True, annotation_targets_accessed=False,
        upstream_observations_reused=True, new_backbone_or_mask_forwards=False))


def _inputs():
    rows, old_protocol, old_seal = E._verify_prediction_seal(OLD_RUN.OUTPUT)
    rows = [r for r in rows if r['cohort'] == 'PRIMARY_NONSEVERE']
    if {p: sum(r['partition'] == p for r in rows) for p in ('CAL', 'TEST')} != {'CAL': 46, 'TEST': 49}:
        raise ValueError('Fixed primary nonsevere population changed')
    model = C.read(E._verify_binding(old_protocol['calibration_model']))
    config = old_protocol['solver_configuration']
    if config['prior_weight'] != 1. or config['bias_strength'] != 1.:
        raise ValueError('Frozen CAL lambda/beta changed')
    region_model = load_truncation_interval_model(REGION_MODEL)
    bindings = {str(p.relative_to(C.ROOT)): C.sha(p) for p in (
        OLD_RUN.OUTPUT / 'PREDICTIONS.jsonl.gz', OLD_RUN.OUTPUT / 'PREDICTION_SEAL.json',
        OLD_RUN.OUTPUT / 'PROTOCOL.json', REGION_MODEL, REGION_MODEL.parent / 'MODEL_SEAL.json',
        C.ROOT / old_protocol['calibration_model']['path'])}
    corrections = {}
    for partition in ('CAL', 'TEST'):
        seal_path = RECOVERY / f'{partition}_RECOVERY_SEAL.json'
        seal = C.read(seal_path)
        E._verify_sources(seal['source_sha256'])
        journal = RECOVERY / f'{partition}_RECOVERY_PREDICTIONS.jsonl.gz'
        protocol = RECOVERY / f'{partition}_RECOVERY_PROTOCOL.json'
        if C.sha(journal) != seal['predictions_sha256'] or C.sha(protocol) != seal['protocol_sha256']:
            raise ValueError('Latest BI observation seal changed')
        journal_rows = list(C.rows(journal))
        if len(journal_rows) != seal['rows']:
            raise ValueError('Latest BI population changed')
        journal_by_id = {r['id']: r for r in journal_rows}
        if set(journal_by_id) != {r['id'] for r in rows if r['partition'] == partition}:
            raise ValueError('Latest BI identities changed')
        for path in (seal_path, journal, protocol):
            bindings[str(path.relative_to(C.ROOT))] = C.sha(path)
        for row in rows:
            if row['partition'] != partition:
                continue
            path = RECOVERY / f'{partition}_predictions' / (row['id'].replace(':', '__') + '.json')
            corrections[row['id']] = C.read(path)
            if corrections[row['id']] != journal_by_id[row['id']]['results']['FULL']:
                raise ValueError('Latest BI individual artifact differs from sealed journal')
            if journal_by_id[row['id']]['states'] != row['runtime']['states']:
                raise ValueError('Latest BI state gate changed')
            if corrections[row['id']]['image_pixel_sha256'] != row['runtime']['visible_correction']['image_pixel_sha256']:
                raise ValueError('BI correction actual RGB changed')
            bindings[str(path.relative_to(C.ROOT))] = C.sha(path)
    return rows, corrections, model, config, region_model, bindings, old_protocol


def predict(output=OUTPUT, workers=3):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing pose evaluation: ' + str(output))
    rows, corrections, model, config, region_model, bindings, old_protocol = _inputs()
    # Bind complete current research source set, including new pure solver.
    sources = {str(p.relative_to(C.ROOT)): C.sha(p) for p in Path(__file__).parent.glob('*.py')}
    protected = old_protocol['protected_user_sha256']
    E._verify_sources(protected)
    output.mkdir(parents=True)
    protocol = dict(schema='FIXED_UPSTREAM_PAIRED_U_REGION_POSE_V4_5', arms=list(ARMS),
        primary_population={'CAL': 46, 'TEST': 49, 'severe': 0}, ids=[r['id'] for r in rows],
        U_policy='No image/corner search; initial nonS raw center plus borrowed CAL max circle; U state retained',
        U_radius_ratio=region_model['radius_ratio'], U_radius_model_point_count=264,
        U_radius_model_is_U_specific=False, future_coverage_guarantee=False,
        region_clipped_to_canvas=False, U_added_to_point_cost_or_seed_or_scale=False,
        same_BI_fit_covariance_and_reliability_in_both_arms=True,
        BI_observations='Frozen latest SAM v4_3b recovery; original CAL fit scales retained',
        solver_configuration=config, exact_sealed_parent_math_cache_allowed=True,
        feasible_parent_reuse='Unchanged objective; parent winner satisfying all U retained, audited searched family filtered; no global optimality claim',
        new_backbone_or_mask_forwards=False, annotation_targets_accessed=False,
        source_sha256=sources, input_sha256=bindings, protected_user_sha256=protected,
        TEST_status='REUSED_EXPLORATORY_TEST_NO_INDEPENDENT_GENERALIZATION', workers=workers)
    C.write(output / 'PROTOCOL.json', protocol)
    for relative, digest in sources.items():
        path = output / 'code_snapshot' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT / relative, path)
        assert C.sha(path) == digest
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        for partition in ('CAL', 'TEST'):
            population = [r for r in rows if r['partition'] == partition]
            futures = {pool.submit(_run, (r, corrections[r['id']], model, config, region_model)): r['id'] for r in population}
            completed = {}
            for future in as_completed(futures):
                result = future.result()
                if result['id'] != futures[future] or result['id'] in completed:
                    raise ValueError('Incorrect pose result identity')
                completed[result['id']] = result
                # Durable per-frame review artifact, even if a later worker fails.
                C.write(output / f'{partition}_frames' / (result['id'].replace(':', '__') + '.json'), result)
                C.write(output / 'LIVE_PROGRESS.json', dict(partition=partition, completed=len(completed), total=len(population), elapsed_seconds=time.monotonic()-started))
                print(f"{partition} {len(completed)}/{len(population)} {result['id']} U={len(result['u_region_observation']['u_regions'])} cache={result['cache_audit']['parent_mathematical_input_equal']}", flush=True)
            ordered = [completed[r['id']] for r in population]
            journal = output / f'{partition}_PREDICTIONS.jsonl.gz'
            C.save_rows(journal, ordered)
            E._verify_sources(sources)
            E._verify_sources(bindings)
            E._verify_sources(protected)
            C.write(output / f'{partition}_PREDICTION_SEAL.json', dict(
                schema='U_REGION_POSE_PARTITION_SEAL_V4_5', partition=partition, rows=len(ordered),
                before_new_partition_pose_scoring=True, annotation_targets_accessed=False,
                predictions_sha256=C.sha(journal), protocol_sha256=C.sha(output / 'PROTOCOL.json'),
                source_sha256=sources, input_sha256=bindings,
                S_image_queries=0, U_image_queries=0, U_exact_point_objective_terms=0))
    C.write(output / 'PREDICTION_COMPLETE.json', dict(rows=95, CAL=46, TEST=49, severe=0, elapsed_seconds=time.monotonic()-started))
    return output


def _summary(rows, arm):
    metrics = [r['metrics'][arm] for r in rows]
    result = {key: distribution([m[key] for m in metrics]) for key in ('translation_cm', 'rotation_deg', 'ADDsym_cm')}
    result.update(frames=len(rows), successes_5cm_5deg=sum(m['success_5cm_5deg'] for m in metrics),
                  fallback_count=sum(not r['accepted'][arm] for r in rows),
                  ambiguous_count=sum(r['ambiguous'][arm] for r in rows))
    result['maximum_examples'] = {
        key: {name: deepcopy(worst[name]) for name in ('id', 'image', 'metrics', 'physical_native_compatible', 'u_native_ids', 'accepted')}
        for key in ('translation_cm', 'rotation_deg', 'ADDsym_cm')
        for worst in [max(rows, key=lambda r: r['metrics'][arm][key])]} if rows else {}
    return result


def score(output=OUTPUT):
    output = Path(output).resolve()
    protocol = C.read(output / 'PROTOCOL.json')
    reports = {}
    # Only after all95 predictions are complete may new target scoring occur.
    if not (output / 'PREDICTION_COMPLETE.json').is_file():
        raise ValueError('All95 pose predictions must finish before target reads')
    for partition in ('CAL', 'TEST'):
        journal = output / f'{partition}_PREDICTIONS.jsonl.gz'
        seal = C.read(output / f'{partition}_PREDICTION_SEAL.json')
        if not seal['before_new_partition_pose_scoring'] or C.sha(journal) != seal['predictions_sha256'] or C.sha(output / 'PROTOCOL.json') != seal['protocol_sha256']:
            raise ValueError('New partition pose seal changed')
        E._verify_sources(seal['source_sha256'])
        E._verify_sources(seal['input_sha256'])
        prediction = list(C.rows(journal))
        target_path = E.COHORT / ('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if partition == 'CAL' else 'test_reused_v1/ISOLATED_TEST_SCORING.json')
        labels = C.read(target_path)['rows']
        if isinstance(labels, list):
            labels = {r['id']: r for r in labels}
        values = []
        for row in prediction:
            label = labels[row['id']]
            alignment = _reference_alignment(row, label)
            metrics = {arm: pose_metric(row['arms'][arm]['actual_pose'], label['target']['truth']) for arm in ARMS}
            a, b = [row['arms'][arm]['actual_pose'] for arm in ARMS]
            identical = a['R_cf'] == b['R_cf'] and a['centroid'] == b['centroid']
            values.append(dict(id=row['id'], image=row['image'], states=row['states'],
                u_native_ids=[r['native_corner'] for r in row['u_region_observation']['u_regions']],
                u_radius_px=row['u_region_observation']['radius_px'],
                physical_native_compatible=bool(alignment['compatible']),
                geometry_gap_m=float(alignment['geometry_gap_m']), metrics=metrics,
                accepted={arm: row['arms'][arm]['accepted'] for arm in ARMS},
                ambiguous={arm: row['arms'][arm]['ambiguity_audit']['material_ambiguity'] for arm in ARMS},
                U_changes_selected_pose=not identical, cache_audit=row['cache_audit'],
                U_constraint_audit=row['arms'][ARMS[1]]['solution'].get('U_region_constraint_audit', [])))
        report = dict(schema='POSTSEAL_U_REGION_PAIRED_POSE_SCORE_V4_5', partition=partition,
            severe_count=0, all_frames_include_fallbacks=True,
            summary_all={arm: _summary(values, arm) for arm in ARMS},
            summary_physical_native_compatible={arm: _summary([r for r in values if r['physical_native_compatible']], arm) for arm in ARMS},
            incompatible_count=sum(not r['physical_native_compatible'] for r in values),
            U_point_count=sum(len(r['u_native_ids']) for r in values),
            U_frame_count=sum(bool(r['u_native_ids']) for r in values),
            U_changed_pose_count=sum(r['U_changes_selected_pose'] for r in values),
            target_binding=E._binding(target_path), prediction_seal=E._binding(output / f'{partition}_PREDICTION_SEAL.json'),
            no_U_specific_coverage_claim=True, TEST_is_reused_exploratory=True, rows=values)
        C.write(output / f'{partition}_POSE_SCORE.json', report)
        reports[partition] = report
    test = reports['TEST']
    summary = test['summary_all'][ARMS[1]]
    target = summary['maximum_examples']['translation_cm']
    rotation = summary['maximum_examples']['rotation_deg']
    text = (
        '# U를 위치 영역으로만 사용하는 PnP 실험\n\n'
        'U의 교점·경계를 다시 찾지 않고 초기 비S 예측 위치 주변의 원형 영역만 제약으로 사용했다. '
        '반경은 기존 CAL 264점 최대 오차 모델에서 빌린 값이다. U 전용 신뢰구간은 아니며, 프레임 밖 영역도 허용한다. '
        'B/I는 최신 SAM 경계 보정 좌표와 기존 CAL 유효 공분산·가중치를 두 비교군에 똑같이 사용했다. S는 PnP 투영만 한다.\n\n'
        f"- TEST: clean/moderate 49장, severe 0장. U {test['U_point_count']}개/{test['U_frame_count']}장.\n"
        f"- 최대 위치 오차: **{summary['translation_cm']['max']:.4f} cm**, `{target['id']}`.\n"
        f"- 최대 회전 오차: **{summary['rotation_deg']['max']:.4f}°**, `{rotation['id']}`.\n"
        f"- 평균 위치/회전 오차: {summary['translation_cm']['mean']:.4f} cm / {summary['rotation_deg']['mean']:.4f}°.\n"
        f"- 5cm·5° 성공: {summary['successes_5cm_5deg']}/49. fallback {summary['fallback_count']}장.\n"
        f"- U 영역 때문에 선택 자세가 바뀐 TEST: {test['U_changed_pose_count']}장.\n\n"
        f"전체 49장을 먼저 집계했다. 기존 물리 모델/native 대응 불일치 {test['incompatible_count']}장은 제외하지 않았으며 별도 호환 42장 지표도 JSON에 보존했다. "
        '기존 TEST를 재사용한 분석이며 독립 일반화 성능으로 주장하지 않는다. 초기 상태 판정 및 초기 S 오류는 이번 변경에서 고치지 않았다.\n')
    (output / 'README_KO.md').write_text(text)
    C.write(output / 'SCORE_SEAL.json', dict(before_scoring_partition_prediction_seals_verified=True,
        CAL_pose_score_sha256=C.sha(output / 'CAL_POSE_SCORE.json'),
        TEST_pose_score_sha256=C.sha(output / 'TEST_POSE_SCORE.json'),
        README_sha256=C.sha(output / 'README_KO.md')))
    return {partition: report['summary_all'][ARMS[1]] for partition, report in reports.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    if not args.score_only:
        predict(args.output, args.workers)
    print(C.finite(score(args.output)), flush=True)


if __name__ == '__main__':
    main()
