"""Freeze GT-only view strata, then describe completed DEV results by stratum.

Preparation reads no model predictions. Analysis is secondary explanation,
never a training, threshold, checkpoint, or primary-verdict selection step.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
POSE_DIR = ROOT / 'data/pallet/results/paper_pose_metric_closure_v1'
POSITIVE = ROOT / 'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json'
GEOMETRY_GT = POSE_DIR / 'GEOMETRY_RESOLVED_POSE_GT.json'
AXIS_MANIFEST = POSE_DIR / 'AXIS_REVIEW_MANIFEST.json'
GT_BUILDER = ROOT / 'scripts/paper/pose_metric_closure_v1/build_geometry_resolved_pose_gt.py'
STATISTICS = ROOT / 'scripts/research/pallet_line_pose_v1/aggregate_results.py'
ARMS = ('point_only', 'hough_features', 'hough_joint')
SEEDS = (1, 2, 3)
ELEVATION_BINS = ('elev_lt5', 'elev_5_15', 'elev_15_30', 'elev_ge30')
FRONTNESS_BINS = ('side_le0p1', 'side_0p1_0p3', 'side_gt0p3', 'unknown')
EXPECTED_ELEVATION = dict(zip(ELEVATION_BINS, (55, 97, 82, 85)))
EXPECTED_FRONTNESS = dict(zip(FRONTNESS_BINS, (144, 51, 96, 28)))
FACES = {'front': (0, 3, 2, 1), 'left': (0, 4, 7, 3), 'right': (1, 2, 6, 5)}
USER_CASE = 'eval_pallet07:1778652166837872128'
POINT_METRICS = ('keypoint_location_median_px', 'keypoint_location_p90_px', 'frame_mean_error_px')
POSE_METRICS = ('rotation_median_deg', 'translation_median_cm', 'iou3d_median', 'add_sym_auc')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_hashes(hashes):
    for path, digest in hashes.items():
        require(sha(path) == digest, f'Bound input changed: {path}')


def load_frozen(root):
    protocol_path = root / 'VIEW_DIAGNOSTIC_PROTOCOL.json'
    strata_path = root / 'VIEW_STRATA.json'
    protocol, strata = read(protocol_path), read(strata_path)
    require(protocol['status'] == 'PREPARED_NOT_EVALUATED' and protocol['selection_permitted'] is False,
            'Not the frozen explanatory protocol')
    require(protocol['script_sha256'] == sha(__file__), 'View diagnostic script changed after preparation')
    verify_hashes(protocol['source_sha256'])
    require(strata['protocol_sha256'] == sha(protocol_path) and strata['model_predictions_read'] == 0
            and strata['frozen_before_new_real_predictions'] is True, 'Strata provenance differs')
    require(strata['counts']['elevation'] == EXPECTED_ELEVATION
            and strata['counts']['frontness_proxy'] == EXPECTED_FRONTNESS, 'Frozen stratum denominator differs')
    require(len(strata['records']) == len({r['frame_id'] for r in strata['records']}) == 319,
            'Strata must retain exactly319 DEV frames')
    return protocol, strata


def prepare(root):
    protocol_path = root / 'VIEW_DIAGNOSTIC_PROTOCOL.json'
    strata_path = root / 'VIEW_STRATA.json'
    if protocol_path.exists() and strata_path.exists():
        load_frozen(root)
        print('Existing metadata-only strata verified; no model predictions read.')
        return
    require(not list((root / 'evaluation').glob('*/PREDICTIONS.json')),
            'Cannot claim prospective view strata after new real predictions exist')
    population, geometry, axis = read(POSITIVE), read(GEOMETRY_GT), read(AXIS_MANIFEST)
    require(population['role'] == 'DEV' and population['expected_count'] == 319
            and len(population['items']) == 319, 'Only canonical DEV319 is permitted')
    require(geometry['model_predictions_used'] is False and geometry['source_annotations_modified'] is False
            and geometry['resolved'] == geometry['total'] == 319, 'Existing model-independent GT reference required')
    axis_by_image = {str((ROOT / r['image']).resolve()): r for r in axis['frames_list']}
    source_paths = [POSITIVE, GEOMETRY_GT, AXIS_MANIFEST, GT_BUILDER, STATISTICS]
    rows = []
    for item in population['items']:
        annotation_path = (ROOT / item['gt_v2_path']).resolve()
        source_paths.append(annotation_path)
        annotation = read(annotation_path)['objects'][0]
        require(annotation['keypoint_frame'] == 'camera_dynamic_0123_v4', 'Camera-facing convention required')
        points = annotation['keypoint_annotations']
        require(len(points) == 9, 'Canonical annotation must have nine entries')
        xy = np.array([p['xy'] if p['xy'] is not None else [np.nan, np.nan] for p in points[:8]], dtype=float)
        known = np.array([p['visibility'] > 0 for p in points[:8]]) & np.isfinite(xy).all(-1)
        side, front_area, unknown_reason = None, None, None
        if known.all():
            areas = {}
            for name, indices in FACES.items():
                q = xy[list(indices)]
                areas[name] = float(.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1)
                                               - q[:, 1] * np.roll(q[:, 0], -1)))
            front_area = abs(areas['front'])
            if front_area >= 1.:
                side = max(-areas['left'], -areas['right'], 0.) / front_area
            else:
                unknown_reason = 'front_face_area_below_1_pixel_squared'
        else:
            unknown_reason = 'not_all_eight_corner_coordinates_supervised'
        front_bin = 'unknown' if side is None else 'side_le0p1' if side <= .1 else 'side_0p1_0p3' if side <= .3 else 'side_gt0p3'
        frame = axis_by_image[str((ROOT / item['image_path']).resolve())]
        truth = geometry['frames'][frame['frame_id']]
        elevation = float(truth['elevation_deg'])
        require(np.isfinite(elevation) and 0 <= elevation <= 90, 'Invalid reference-derived elevation')
        elevation_bin = 'elev_lt5' if elevation < 5 else 'elev_5_15' if elevation < 15 else 'elev_15_30' if elevation < 30 else 'elev_ge30'
        rows.append(dict(frame_id=item['frame_id'], pose_frame_id=frame['frame_id'],
            session_id=item['session_id'], object_type=item['object_type'], domain=item.get('domain'),
            elevation_deg_reconstructed_unsigned=elevation, elevation_bin=elevation_bin,
            frontness_proxy=side, frontness_bin=front_bin, front_area_px2=front_area,
            supervised_corner_count=int(known.sum()), unknown_corner_indices=np.flatnonzero(~known).tolist(),
            frontness_unknown_reason=unknown_reason, reference_chosen_reprojection_px=float(truth['chosen_reproj_px'])))
    counts = dict(elevation=dict(Counter(r['elevation_bin'] for r in rows)),
                  frontness_proxy=dict(Counter(r['frontness_bin'] for r in rows)))
    require(counts['elevation'] == EXPECTED_ELEVATION and counts['frontness_proxy'] == EXPECTED_FRONTNESS,
            'GT-only counts differ from reviewed cohort definitions')
    hashes = {str(p.resolve()): sha(p) for p in dict.fromkeys(source_paths)}
    old = read(protocol_path) if protocol_path.exists() else None
    protocol = dict(schema='pallet_dht_joint_view_diagnostic_protocol_v1', status='PREPARED_NOT_EVALUATED',
        prepared_at_utc=old['prepared_at_utc'] if old else datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(__file__), source_sha256=hashes,
        population=dict(name='PAPER_EVAL_ALL_POS', role='reused DEV', frames=319, sessions=13, independent_final=False),
        purpose='Secondary explanatory view-stratum analysis of the frozen same-budget architecture experiment.',
        selection_permitted=False, changes_primary_verdict=False, new_model_forwards=0,
        primary_comparison=dict(left='hough_joint', right='point_only', seeds=list(SEEDS)),
        definitions=dict(
            elevation='Existing frozen GT-reference elevation_deg: abs(asin(abs(n_top dot normalized(t_gt)))). Degrees, unsigned. Reconstructed from manual cuboid coordinates, calibration and registered dimensions; not a measured world-camera angle. No new PnP fit.',
            elevation_bins={'elev_lt5': '[0,5)', 'elev_5_15': '[5,15)', 'elev_15_30': '[15,30)', 'elev_ge30': '[30,90]'},
            frontness='max(-signed_area(left),-signed_area(right),0)/abs(signed_area(front)). Image-space ratio, not yaw degrees. Both side faces and front use the frozen camera-facing ordering.',
            faces=FACES, frontness_bins={'side_le0p1': '[0,.1]', 'side_0p1_0p3': '(.1,.3]', 'side_gt0p3': '(.3,infinity)', 'unknown': 'unsupported GT coordinates or front area<1px²'},
            coordinate_support='All eight finite corner xy with annotation visibility>0. Occluded supervised coordinates are allowed; physical edge visibility is not inferred. Visibility0 is not filled from a fit.',
            strata_join='Canonical2D IDs and MAIN6D IDs joined through frozen AXIS_REVIEW image paths; neither label convention is rewritten.'),
        expected_counts=counts,
        analysis=dict(arms=list(ARMS), seeds=list(SEEDS),
            points='Pool supervised 0..8 coordinate errors per seed under unchanged top-score IoU>=.5 matching; median/P90. frame_mean_error_px is the mean of per-frame supervised-point means on matched frames.',
            pose='All four unchanged MAIN metrics per subset, including subset median-diameter 1001-threshold ADD AUC.',
            aggregation='Compute each seed statistic first, then mean and sampleSD(ddof1); expose every seed matched/pose count.',
            paired='Also compare joint-control on common observed frames across all six seed-arm stores; point and pose intersections are separate.',
            subgroup_inference='Descriptive only; no new hypothesis test/CI, no error-selected group boundaries, no exclusion based on reference reprojection quality.'),
        limitations=['Elevation uses the existing reconstructed reference, including its calibration/geometry assumptions and unsigned-normal ambiguity.',
                     'Projected face-area ratios also depend on object proportions and perspective, and are not physical visibility labels.',
                     'Within-group frames and three seeds are not independent new captures; session counts are displayed.',
                     'Observed-only metric changes can reflect changed matching coverage; full group denominators and paired subsets are shown.'])
    root.mkdir(parents=True, exist_ok=True)
    if old is not None:
        require(old == protocol, 'Existing view protocol differs; do not silently replace frozen metadata')
    else:
        write(protocol_path, protocol)
    strata = dict(schema='pallet_dht_joint_view_strata_v1', status='PREPARED_NOT_EVALUATED',
        protocol_sha256=sha(protocol_path), script_sha256=sha(__file__), source_sha256=hashes,
        model_predictions_read=0, new_model_forwards=0, frozen_before_new_real_predictions=True,
        n_frames=319, n_sessions=len({r['session_id'] for r in rows}), counts=counts, records=rows,
        user_case=next(r for r in rows if r['frame_id'] == USER_CASE))
    write(strata_path, strata)
    print(json.dumps(dict(status='PREPARED_NOT_EVALUATED', counts=counts,
        user_case_frontness=strata['user_case']['frontness_bin'], model_predictions_read=0,
        protocol_sha256=sha(protocol_path), strata_sha256=sha(strata_path)), ensure_ascii=False))


def statistics_module():
    spec = importlib.util.spec_from_file_location('frozen_view_statistic_primitives', STATISTICS)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def point_summary(rows, statistics):
    out = statistics.keypoint_summary(rows)
    frame_errors = [float(np.mean(r['errors'])) for r in rows if len(r['errors'])]
    out['frame_mean_error_px'] = float(np.mean(frame_errors)) if frame_errors else None
    return out


def seed_average(per_seed, metric_names):
    result = {}
    for metric in metric_names:
        values = [row.get(metric) for row in per_seed]
        available = [v for v in values if v is not None]
        result[metric] = dict(values=values, n_available_seeds=len(available),
            mean=float(np.mean(values)) if len(available) == 3 else None,
            sample_std_ddof1=float(np.std(values, ddof=1)) if len(available) == 3 else None)
    return result


def difference(left, right):
    return None if left is None or right is None else float(left - right)


def number(value):
    return '—' if value is None else f'{value:.3f}'


def markdown(result):
    lines = ['# 구도별 공동학습 진단', '',
        '동일 예산 `hough_joint`와 `point_only`를 비교한 **설명용 사후 진단**입니다. 학습·모델 선택·주 실험 판정은 바꾸지 않습니다.', '',
        '앙각은 기존 GT로 재구성한 상면 기준 시선각의 크기이며 카메라 실측값이 아닙니다. 정면성은 2D 측면/앞면 투영면적 비율이며 yaw 각도가 아닙니다. 좌표가 부족한 28장은 unknown으로 유지했습니다.', '',
        '아래 값은 각 seed의 지표를 계산한 뒤 낸 3개 seed 평균입니다. 점 지표는 centroid를 포함한 0–8번 중 감독된 좌표를 사용합니다. `frameMean`은 관측 프레임별 점오차 평균의 평균입니다.', '',
        '| 구도 | GT N / 세션 | 9점 median: 점→공동(px) | P90: 점→공동(px) | frameMean: 점→공동(px) | 매칭 N: 점 / 공동(각 seed) |',
        '|---|---:|---:|---:|---:|---|']
    for group in result['groups']:
        a, b = group['arms']['point_only'], group['arms']['hough_joint']
        pairs = [f"{number(a['point_mean'][m]['mean'])} → {number(b['point_mean'][m]['mean'])}" for m in POINT_METRICS]
        lines.append('| ' + ' | '.join([group['label'], f"{group['n_frames']} / {group['n_sessions']}", *pairs,
            f"{[r['matched_frames'] for r in a['points_by_seed']]} / {[r['matched_frames'] for r in b['points_by_seed']]}"]) + ' |')
    lines += ['', '양쪽 모델의 세 seed가 모두 관측한 공통 프레임만 사용한 점 비교도 표시합니다. 전체 모집단 성능이나 새로운 배포 gate가 아닙니다.', '',
              '| 구도 | 공통 점 프레임 N | 공동−점 median(px) | 공동−점 P90(px) | 공동−점 frameMean(px) |', '|---|---:|---:|---:|---:|']
    for group in result['groups']:
        paired = group['paired_common']
        lines.append('| ' + ' | '.join([group['label'], str(paired['point_frames']),
            *[number(paired['point_delta'][m]) for m in POINT_METRICS]]) + ' |')
    lines += ['', 'MAIN 6D는 기존 solver·GT 계약의 관측 가능한 pose만 집계합니다. ADD AUC의 기준 길이는 해당 subset의 기존 정의를 따릅니다.', '',
              '| 구도 | pose N: 점 / 공동(각 seed) | 회전 median(°) | 이동 median(cm) | IoU3D median | ADD AUC |',
              '|---|---|---:|---:|---:|---:|']
    for group in result['groups']:
        a, b = group['arms']['point_only'], group['arms']['hough_joint']
        pairs = [f"{number(a['pose_mean'][m]['mean'])} → {number(b['pose_mean'][m]['mean'])}" for m in POSE_METRICS]
        lines.append('| ' + ' | '.join([group['label'], f"{[r['n'] for r in a['poses_by_seed']]} / {[r['n'] for r in b['poses_by_seed']]}", *pairs]) + ' |')
    case = result['user_case']
    lines += ['', f"사용자 사례 `{case['frame_id']}`의 정면성은 `{case['frontness_bin']}`입니다. 부족한 corner index는 `{case['unknown_corner_indices']}`이며 채워 넣지 않았습니다.", '',
              '전체 세 구조의 seed별 값·공통 pose 비교·세션 구성·입력 SHA는 [VIEW_DIAGNOSIS.json](VIEW_DIAGNOSIS.json)에 있습니다. 구도 코호트는 새 모델의 실사 예측 전에 [VIEW_DIAGNOSTIC_PROTOCOL.json](VIEW_DIAGNOSTIC_PROTOCOL.json)과 [VIEW_STRATA.json](VIEW_STRATA.json)으로 고정했습니다.', '',
              '새로운 성능 우월성 판정은 하지 않습니다. 주 실험의 판정은 기존 `VERDICT.json` 그대로입니다.']
    return '\n'.join(lines) + '\n'


def analyze(root):
    protocol, strata = load_frozen(root)
    statistics = statistics_module()
    input_hashes = {str(root / name): sha(root / name) for name in
                    ('VIEW_DIAGNOSTIC_PROTOCOL.json', 'VIEW_STRATA.json', 'SUMMARY.json', 'VERDICT.json')}
    summary = read(root / 'SUMMARY.json'); verdict = read(root / 'VERDICT.json')
    require(summary.get('complete') and summary.get('PASS') and len(summary['runs']) == 9,
            'All nine actual evaluations and SUMMARY must be completed first')
    stores = {arm: {'points': [], 'poses': []} for arm in ARMS}
    all_point_ids = {r['frame_id'] for r in strata['records']}
    all_pose_ids = {r['pose_frame_id'] for r in strata['records']}
    for arm in ARMS:
        for seed in SEEDS:
            label = f'{arm}_seed{seed}'; folder = root / 'evaluation' / label
            done = read(folder / 'COMPLETION.json')
            require(done.get('complete') and done.get('PASS') and done['arm'] == arm and done['seed'] == seed
                    and done['actual_positive_forwards'] == 319 and done['actual_negative_forwards'] == 2689
                    and done['baseline_candidate_copying'] is False, f'Incomplete actual evaluation: {label}')
            for name in ('COMPLETION.json', 'RESULTS.json', 'PAPER_2D_per_frame.csv', 'POSE_PER_FRAME_BY_ARM.json'):
                path = folder / name; input_hashes[str(path)] = sha(path)
                if name != 'COMPLETION.json':
                    require(done['output_sha256'][name] == sha(path), f'Changed completed evaluation: {path}')
            point = statistics.load_keypoint_rows(folder / 'PAPER_2D_per_frame.csv')
            pose_rows = read(folder / 'POSE_PER_FRAME_BY_ARM.json')['per_frame'][label]
            pose = {r['frame_id']: r for r in pose_rows}
            require(set(point) == all_point_ids and set(pose) <= all_pose_ids and len(pose) == len(pose_rows),
                    '2D/6D subset IDs do not match the frozen image-based cohort join')
            stores[arm]['points'].append(point); stores[arm]['poses'].append(pose)
    definitions = [('all', '전체319', None, None)]
    definitions += [(b, 'GT 재구성 앙각 ' + label, 'elevation_bin', b) for b, label in zip(ELEVATION_BINS, ('<5°', '5–15°', '15–30°', '≥30°'))]
    definitions += [(b, '정면성 proxy ' + label, 'frontness_bin', b) for b, label in zip(FRONTNESS_BINS, ('≤0.1', '0.1–0.3', '>0.3', 'unknown'))]
    groups = []
    for group_id, label, field, value in definitions:
        rows = [r for r in strata['records'] if field is None or r[field] == value]
        point_ids = [r['frame_id'] for r in rows]; pose_ids = [r['pose_frame_id'] for r in rows]
        group = dict(id=group_id, label=label, n_frames=len(rows), n_sessions=len({r['session_id'] for r in rows}),
            session_frame_counts=dict(Counter(r['session_id'] for r in rows)),
            material_frame_counts=dict(Counter(r['object_type'] for r in rows)), arms={})
        for arm in ARMS:
            points = [point_summary([s[k] for k in point_ids], statistics) for s in stores[arm]['points']]
            poses = [statistics.pose_summary([s[k] for k in pose_ids if k in s]) for s in stores[arm]['poses']]
            group['arms'][arm] = dict(seeds=list(SEEDS), points_by_seed=points, poses_by_seed=poses,
                point_mean=seed_average(points, POINT_METRICS), pose_mean=seed_average(poses, POSE_METRICS),
                point_coverage_by_seed=[p['matched_frames'] / len(rows) if rows else None for p in points],
                pose_coverage_by_seed=[p['n'] / len(rows) if rows else None for p in poses])
        left, right = group['arms']['hough_joint'], group['arms']['point_only']
        group['observed_only_delta'] = {m: difference(left['point_mean' if m in POINT_METRICS else 'pose_mean'][m]['mean'],
            right['point_mean' if m in POINT_METRICS else 'pose_mean'][m]['mean']) for m in POINT_METRICS + POSE_METRICS}
        paired = {}
        for kind, ids, metrics in [('point', point_ids, POINT_METRICS), ('pose', pose_ids, POSE_METRICS)]:
            six = stores['hough_joint'][kind + 's'] + stores['point_only'][kind + 's']
            common = [k for k in ids if all(k in s and (kind == 'pose' or len(s[k]['errors'])) for s in six)]
            per_seed = [(point_summary([s[k] for k in common], statistics) if kind == 'point'
                         else statistics.pose_summary([s[k] for k in common])) for s in six]
            a, b = seed_average(per_seed[:3], metrics), seed_average(per_seed[3:], metrics)
            paired[kind + '_frames'] = len(common)
            paired[kind + '_delta'] = {m: difference(a[m]['mean'], b[m]['mean']) for m in metrics}
        group['paired_common'] = paired; groups.append(group)
    result = dict(schema='pallet_dht_joint_view_diagnosis_v1', complete=True, PASS=True,
        meaning='Completed secondary explanatory analysis; PASS means input/arithmetic integrity, not a new success criterion.',
        selection_performed=False, new_model_forwards=0, training_changed=False, primary_verdict_changed=False,
        original_overall_accuracy_improved=verdict['overall_accuracy_improved'],
        protocol_sha256=sha(root / 'VIEW_DIAGNOSTIC_PROTOCOL.json'), strata_sha256=sha(root / 'VIEW_STRATA.json'),
        source_sha256={str(Path(__file__).resolve()): sha(__file__), str(STATISTICS): sha(STATISTICS)},
        input_sha256=input_hashes, n_frames=319, n_sessions=13, groups=groups,
        user_case=strata['user_case'], limitations=protocol['limitations'])
    verify_hashes(input_hashes)
    write(root / 'VIEW_DIAGNOSIS.json', result)
    (root / 'VIEW_DIAGNOSIS.md').write_text(markdown(result))
    print(json.dumps(dict(complete=True, diagnostic_only=True, model_selection_changed=False,
        groups=len(groups), output=str(root / 'VIEW_DIAGNOSIS.json')), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    prepare(args.run_dir.resolve()) if args.prepare_only else analyze(args.run_dir.resolve())
