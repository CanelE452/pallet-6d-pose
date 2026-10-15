"""Render sealed, image-only lower-boundary alternatives without new inference.

The report consumes CAL/TEST predictions and their postseal 2D scores. It never
loads annotations, a network, a fitter, or a pose solver. All displayed geometry
comes from each method's saved observations or the shared initial model guide.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from . import common as C


BASE = C.DOC / ('nonsevere_v1/statewise_joint_ot_20261014_v2/'
                'nonsevere_v4_fix/boundary_recovery_v1')
COMPARISON = BASE / 'nonsam_boundary_v2'
OUTPUT = COMPARISON / 'report_v1'
ORIGINAL_PREDICTIONS = BASE.parent / 'evaluation_interval_v4_2b/PREDICTIONS.jsonl.gz'
METHODS = ('RGB_GRADIENT', 'TINYUNET', 'MOBILESAM')
NAMES = {'RGB_GRADIENT': 'RGB gradient · 마스크 없이 경계 검색',
         'TINYUNET': 'TinyUNet · 마스크 + RGB 경계',
         'MOBILESAM': 'MobileSAM · 마스크 + RGB 경계'}
TARGET = 'eval_pallet09:1778653661653195264'


def write_new(path, value):
    if path.exists():
        raise FileExistsError('Preserve completed report: ' + str(path))
    C.write(path, value)


def verify_bindings(bindings):
    for relative, expected in bindings.items():
        path = C.ROOT / relative
        if not path.is_file() or C.sha(path) != expected:
            raise ValueError('Input/source binding changed: ' + str(path))


def slim_branch(branch):
    if not branch:
        return None
    keys = ('normal', 'offset', 'observed_points', 'inlier_mask', 'inlier_count',
            'inlier_fraction', 'inlier_span_fraction', 'inlier_rms_px',
            'angle_change_deg', 'reason', 'supported', 'native_neighbor')
    return {key: deepcopy(branch.get(key)) for key in keys}


def slim_method(value):
    """Preserve every plotted sample; omit unplotted per-split branch payloads."""
    fit = value.get('fit_payload')
    result = {key: deepcopy(value.get(key))
              for key in ('status', 'audit', 'coverage', 'fit_only_elapsed_ms')}
    if not fit:
        result['fit_payload'] = None
        return result
    selected = fit.get('selected_representative')
    representative = None
    if selected:
        representative = {key: deepcopy(selected.get(key)) for key in (
            'intersection_xy', 'intersection_sine', 'normal_matrix_condition_number',
            'robust_Huber_cost', 'split_contour_index', 'split_contour_xy',
            'finite_observed_support_pass', 'observed_support_audit')}
        representative['left'] = slim_branch(selected.get('left'))
        representative['right'] = slim_branch(selected.get('right'))
    qualified = [deepcopy(point['intersection_xy'])
                 for point in fit.get('split_profile', [])
                 if point.get('both_original_line_guards_pass')
                 and point.get('finite_intersection')
                 and point.get('finite_observed_support_pass')
                 and point.get('intersection_xy') is not None]
    result['fit_payload'] = dict(
        contour=deepcopy(fit.get('contour', [])),
        contour_segments=deepcopy(fit.get('contour_segments')),
        refined_points=deepcopy(fit.get('refined_points', [])),
        RGB_eligible=deepcopy(fit.get('RGB_eligible', [])),
        selected_representative=representative,
        qualified_intersection_xy=qualified,
        contour_source_scope='saved method proposal/arc/branch samples; missing RGB support stays missing',
        accepted_native_corner=False)
    return result


def prepare_data(comparison):
    """Require both completed score stages before using any posthoc score row."""
    paths = [comparison / 'PROTOCOL.json', ORIGINAL_PREDICTIONS]
    for partition in ('CAL', 'TEST'):
        paths += [comparison / (partition + suffix) for suffix in (
            '_SEAL.json', '_PREDICTIONS.jsonl.gz', '_SCORE.json')]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError('Both CAL and TEST predictions, seals and scores must be complete')
    before = {str(path.relative_to(C.ROOT)): C.sha(path) for path in paths}
    protocol = C.read(comparison / 'PROTOCOL.json')
    if tuple(protocol['methods']) != METHODS:
        raise ValueError('Unexpected method order')
    if before[str(ORIGINAL_PREDICTIONS.relative_to(C.ROOT))] != protocol['original_prediction_sha256']:
        raise ValueError('Shared initial model/state predictions changed')
    if not protocol['all_methods_same_geometry_only_endpoints'] or protocol['existing_SAM_corrected_endpoints_used']:
        raise ValueError('The report requires geometry-only shared endpoints')
    if protocol['new_full_cohort_pose_fits'] != 0 or protocol['native_keypoint_adopted']:
        raise ValueError('Unexpected native point adoption or pose experiment')
    verify_bindings(protocol['source_sha256'])
    verify_bindings(protocol['protected_user_sha256'])
    originals = {row['id']: row for row in C.rows(ORIGINAL_PREDICTIONS)}
    cases, partitions, assets = [], {}, {}
    for partition, frame_count in (('CAL', 46), ('TEST', 49)):
        seal = C.read(comparison / (partition + '_SEAL.json'))
        prediction_path = comparison / (partition + '_PREDICTIONS.jsonl.gz')
        if (C.sha(prediction_path) != seal['predictions_sha256']
                or C.sha(comparison / 'PROTOCOL.json') != seal['protocol_sha256']
                or not seal['before_new_partition_annotation_score']):
            raise ValueError('Incomplete prediction seal: ' + partition)
        rows = list(C.rows(prediction_path))
        if len(rows) != frame_count or [row['id'] for row in rows] != seal['ids']:
            raise ValueError('Incorrect complete partition/order: ' + partition)
        score = C.read(comparison / (partition + '_SCORE.json'))
        if (score['partition'] != partition or score['frames'] != frame_count
                or score['prediction_sha256'] != seal['predictions_sha256']
                or score['new_pose_fits'] != 0 or score['adopted_native_keypoints'] != 0
                or score['GT_inputs_to_predictions']):
            raise ValueError('Unexpected postseal scoring contract: ' + partition)
        score_points = {(point['id'], point['native_corner']): point for point in score['points']}
        if len(score_points) != sum(len(row['results']) for row in rows):
            raise ValueError('Candidate/score population mismatch')
        timing = {method: [] for method in METHODS}
        for row in rows:
            original = originals[row['id']]
            geometry = original['runtime']['initial_geometry']
            if row['partition'] != partition:
                raise ValueError('Partition mismatch')
            if (row['GT_inputs'] or row['states_changed'] or row['new_pose_fits'] != 0
                    or row['original_BI_corrected_coordinates_read']
                    or row['raw_corner_coordinate_inputs']):
                raise ValueError('Unexpected experimental input')
            image_path = C.ROOT / row['image']
            expected = row['bindings']['image_sha256']
            if expected != original['image_sha256'] or C.sha(image_path) != expected:
                raise ValueError('Actual photograph binding changed')
            if row['results']:
                assets[expected] = dict(source=row['image'], source_sha256=expected,
                                        asset='assets/' + expected + image_path.suffix.lower())
            for candidate in row['results']:
                k, neighbors = candidate['native_corner'], candidate['neighbor_ids']
                if tuple(candidate['methods']) != METHODS:
                    raise ValueError('Every candidate must retain all three methods')
                if any(geometry['initial_self_hidden'][j] for j in [k, *neighbors]):
                    raise ValueError('Initial S cannot be a rendered local candidate or endpoint')
                point = score_points[(row['id'], k)]
                model = {str(j): geometry['initial_projection_xy'][j] for j in [k, *neighbors]}
                if point['model_projection_xy'] != model[str(k)]:
                    raise ValueError('Score/model guide mismatch')
                for method in METHODS:
                    value = candidate['methods'][method]
                    timing[method].append(float(value['fit_only_elapsed_ms']))
                    selected = (value.get('fit_payload') or {}).get('selected_representative')
                    if (point['accepted'][method] != (selected is not None)
                            or point['intersection_xy'][method] != (selected['intersection_xy'] if selected else None)):
                        raise ValueError('Saved score/candidate mismatch')
                cases.append(dict(
                    key=row['id'] + '|KP' + str(k), id=row['id'], partition=partition,
                    native_corner=k, state=candidate['state'], neighbor_ids=neighbors,
                    initial_direction_sine=candidate['direction_sine'],
                    image=assets[expected]['asset'], image_sha256=expected,
                    image_hw=geometry['image_hw'], initial_model_projection=model,
                    score=deepcopy(point),
                    methods={method: slim_method(candidate['methods'][method]) for method in METHODS}))
        partitions[partition] = dict(
            frames=frame_count, candidates=sum(len(row['results']) for row in rows),
            coverage=score['coverage'], strata=score['strata'],
            fit_only_elapsed_ms={method: dict(
                n=len(values), median=float(np.median(values)) if values else None,
                mean=float(np.mean(values)) if values else None)
                for method, values in timing.items()},
            prediction_sha256=seal['predictions_sha256'],
            score_sha256=C.sha(comparison / (partition + '_SCORE.json')),
            annotation_sha256=score['annotation_sha256'],
            TEST_reused_exploratory=score['TEST_reused_exploratory'])
        threshold = float(protocol['method_config']['minimum_intersection_sine'])
        partition_cases = [case for case in cases if case['partition'] == partition]
        groups = {}
        for name, predicate in (
                ('STABLE_INITIAL_GEOMETRY', lambda c: c['initial_direction_sine'] >= threshold),
                ('SHALLOW_INITIAL_GEOMETRY', lambda c: c['initial_direction_sine'] < threshold)):
            population = [case for case in partition_cases if predicate(case)]
            primary = [case for case in population if case['score']['valid_annotation']
                       and case['score']['native_compatible']
                       and case['score']['human_state'] == 'DIRECT_VISIBLE']
            groups[name] = dict(candidates=len(population), DIRECT_COMPATIBLE=len(primary),
                selected_DIRECT_COMPATIBLE={method: sum(case['score']['accepted'][method] for case in primary)
                                            for method in METHODS})
        partitions[partition]['initial_geometry_groups'] = groups
        sines = [case['initial_direction_sine'] for case in partition_cases]
        partitions[partition]['initial_geometry_sine_range'] = [min(sines), max(sines)] if sines else None
        partitions[partition]['geometry_dispatch_sine_threshold'] = threshold
    v1_reference = {}
    old = comparison.parent / 'nonsam_boundary_v1'
    if old != comparison and (old / 'PROTOCOL.json').is_file():
        old_protocol_sha = C.sha(old / 'PROTOCOL.json')
        for partition in ('CAL', 'TEST'):
            score_path, seal_path, prediction_path = [old / (partition + suffix) for suffix in
                ('_SCORE.json', '_SEAL.json', '_PREDICTIONS.jsonl.gz')]
            if not all(path.is_file() for path in (score_path, seal_path, prediction_path)):
                continue
            old_seal = C.read(seal_path)
            old_score = C.read(score_path)
            if (old_seal['protocol_sha256'] != old_protocol_sha
                    or old_seal['predictions_sha256'] != C.sha(prediction_path)
                    or old_score['prediction_sha256'] != old_seal['predictions_sha256']):
                raise ValueError('V1 reference score/prediction seal changed')
            v1_reference[partition] = dict(frames=old_score['frames'], coverage=old_score['coverage'],
                source='nonsam_boundary_v1', scope='global unknown-split template diagnostic only')
            for path in (old / 'PROTOCOL.json', score_path, seal_path, prediction_path):
                before[str(path.relative_to(C.ROOT))] = C.sha(path)
    return C.finite(dict(
        schema='LOWER_BOUNDARY_METHOD_REPORT_DATA_V1', methods=list(METHODS), names=NAMES,
        target_id=TARGET, cases=cases, partitions=partitions,
        protocol=protocol, input_sha256=before, assets=list(assets.values()),
        V1_template_reference=v1_reference,
        renderer_reads_original_annotation_files=False, new_inference_calls=0,
        new_fit_calls=0, new_pose_fits=0, adopted_native_keypoints=0))


def line_segment(branch):
    """The solid line covers actual inlier tangent support, not an infinite ray."""
    if not branch:
        return None
    points = np.asarray(branch.get('observed_points', []), float)
    mask = np.asarray(branch.get('inlier_mask', []), bool)
    normal = np.asarray(branch.get('normal'), float)
    if points.ndim != 2 or points.shape[1:] != (2,) or mask.shape != (len(points),) or not mask.any():
        return None
    tangent = np.array([-normal[1], normal[0]])
    values = points[mask] @ tangent
    return np.array([-float(branch['offset']) * normal + value * tangent
                     for value in (values.min(), values.max())])


def case_limits(case):
    height, width = case['image_hw']
    points = list(case['initial_model_projection'].values())
    for value in case['methods'].values():
        fit = value.get('fit_payload') or {}
        points.extend(fit.get('contour', []))
        selected = fit.get('selected_representative')
        if selected:
            q = selected['intersection_xy']
            if 0 <= q[0] < width and 0 <= q[1] < height:
                points.append(q)
    array = np.asarray(points, float)
    return [max(0., array[:, 0].min() - 24), min(width - 1., array[:, 0].max() + 24),
            max(0., array[:, 1].min() - 28), min(height - 1., array[:, 1].max() + 28)]


def representative_png(output, bundle):
    """One publication-style actual-image comparison; no annotation point drawn."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import matplotlib.image as mpimg

    font = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    if not font.is_file():
        raise FileNotFoundError('Korean figure font is required: ' + str(font))
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    target = next((case for case in bundle['cases'] if case['id'] == TARGET and case['native_corner'] == 3),
                  bundle['cases'][0] if bundle['cases'] else None)
    if target is None:
        return None
    image = mpimg.imread(output / target['image'])
    figure, axes = plt.subplots(1, 3, figsize=(17, 6.7), constrained_layout=False)
    limits = case_limits(target)
    for ax, method in zip(axes, METHODS):
        value = target['methods'][method]
        fit = value.get('fit_payload') or {}
        selected = fit.get('selected_representative')
        ax.imshow(image)
        guide = np.asarray([target['initial_model_projection'][str(j)]
                            for j in [target['neighbor_ids'][0], target['native_corner'], target['neighbor_ids'][1]]])
        ax.plot(guide[:, 0], guide[:, 1], '--', color='#c5cbd3', lw=1.5, label='초기 모델 guide')
        ax.plot(*guide[1], 'o', color='#c5cbd3', ms=5)
        contour = np.asarray(fit.get('contour', []), float)
        segments = fit.get('contour_segments') or ([contour] if contour.size else [])
        for index, segment in enumerate(segments):
            if isinstance(segment, dict):
                segment = segment.get('points', segment.get('contour', []))
            array = np.asarray(segment, float)
            if array.ndim == 2 and array.shape[1:] == (2,) and len(array) > 1:
                ax.plot(array[:, 0], array[:, 1], color='#f4d54c' if method == 'RGB_GRADIENT' else '#20d46d',
                        lw=1.5, ls='--' if method == 'RGB_GRADIENT' else '-',
                        label=('관측 경계 표본' if fit.get('contour_segments') else
                            '검색 trace' if method == 'RGB_GRADIENT' else '마스크 외곽 arc') if index == 0 else None)
        points = np.asarray(fit.get('refined_points', []), float)
        support = np.asarray(fit.get('RGB_eligible', []), bool)
        if points.ndim == 2 and support.shape == (len(points),) and support.any():
            ax.scatter(points[support, 0], points[support, 1], s=5, color='#38d9ed', label='RGB 지원 표본')
        if selected:
            for side, color in (('left', '#ff992e'), ('right', '#b280ff')):
                segment = line_segment(selected[side])
                if segment is not None:
                    ax.plot(segment[:, 0], segment[:, 1], color=color, lw=2.6,
                            label='왼쪽 실제 선' if side == 'left' else '오른쪽 실제 선')
            q = selected['intersection_xy']
            ax.plot(*q, 'x', color='#ff525b', ms=9, mew=2.3, label='교점 후보 · KP 채택 아님')
        warning = value['audit'].get('endpoint_guard_pass') is False
        suffix = '\n19px 끝점 연관 경고 유지' if warning else ''
        status = '두 선 후보 있음' if selected else '두 선 후보 보류'
        ax.set_title(NAMES[method] + '\n' + status + suffix, fontsize=12, color='#b33820' if warning else '#17212c')
        ax.set_xlim(limits[0], limits[1]); ax.set_ylim(limits[3], limits[2])
        ax.set_aspect('equal'); ax.set_xlabel('실제 영상 x (px)'); ax.set_ylabel('실제 영상 y (px)')
        ax.legend(loc='lower left', fontsize=7, framealpha=.80)
    figure.suptitle('동일 초기 기하에서 하단 경계 검색 비교 · ' + target['id'] + ' · KP' + str(target['native_corner']), fontsize=15)
    figure.text(.5, .025,
                '회색은 이미지 관측이 아닌 초기 모델입니다. 붉은 ×는 native 대응이 검증되지 않은 교점 후보입니다.\n'
                '19px 초과는 계속 표시한 연관 경고이며 품질 보증이 아닙니다. GT·사용자 표시점·PnP는 사용하지 않았습니다.',
                ha='center', fontsize=10)
    figure.subplots_adjust(left=.045, right=.988, top=.83, bottom=.17, wspace=.23)
    path = output / 'TARGET_COMPARISON_KO.png'
    figure.savefig(path, dpi=170, facecolor='white')
    plt.close(figure)
    return dict(path=path.name, sha256=C.sha(path), id=target['id'], native_corner=target['native_corner'],
                annotation_drawn=False, geometry_source='initial_model_projection',
                observed_line_extent='actual_inlier_tangent_support', association_conditional=True)


HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>기하별 경계 fitting · RGB / TinyUNet / MobileSAM 실제 비교</title>
<style>
*{box-sizing:border-box}body{font-family:system-ui,"Nanum Gothic",sans-serif;background:#edf2f7;color:#182634;margin:0;line-height:1.6}main{max-width:1650px;padding:22px;margin:auto}h1{font-size:25px;line-height:1.4;margin:0 0 12px}h2{font-size:19px;margin:0 0 12px}h3{font-size:17px;margin:0}.box,.method{background:#fff;border:1px solid #d7e0e9;border-radius:12px;padding:17px;margin-bottom:16px}.subtle{font-size:13px;color:#556779}.note{border-left:4px solid #527da8;padding-left:12px}.warning{color:#a43e19}.good{color:#187049}.controls,.legend{display:flex;flex-wrap:wrap;gap:9px 18px;align-items:center}.legend{font-size:13px;margin:12px 0}.legend span{white-space:nowrap}select,button{font:inherit;padding:7px 10px;border:1px solid #b8c8d6;border-radius:6px;background:#fff;max-width:100%}label{font-size:14px;display:inline-flex;gap:6px;align-items:center}.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.method{min-width:0;padding:12px}.method canvas{width:100%;height:auto;display:block;background:#14202b;border-radius:5px;margin:10px 0}.stat{font-size:13px;margin:5px 0;overflow-wrap:anywhere}.status{display:inline-block;padding:2px 6px;border-radius:4px;background:#eaf0f7;font-size:12px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid #dce4ec;text-align:left;white-space:nowrap}th{background:#f2f6fa}summary{cursor:pointer;font-size:13px;color:#476b8d}pre{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:230px;overflow:auto;background:#f3f6f9;padding:10px}a{color:#1768a1}.figure{display:block;width:100%;height:auto}.pill{display:inline-block;background:#e7eef6;padding:2px 7px;border-radius:5px;margin-right:4px;font-size:13px}.smallGap{margin-top:8px}#caseSelect{min-width:250px;max-width:100%}#caseMeta{overflow-wrap:anywhere}@media(max-width:1050px){.views{grid-template-columns:1fr}.method canvas{max-height:none}main{padding:13px}}@media(max-width:600px){h1{font-size:21px}.box{padding:12px}#caseSelect{min-width:0;width:100%}.controls label{max-width:100%}}
</style></head><body><main>
<h1>마스크 없이 찾은 경계와 TinyUNet·MobileSAM 외곽, 실제로 어떻게 다른가</h1>
<div class="box">
<p class="note">같은 초기 모델 투영과 고정 B/U 상태에서 세 경계 방법을 비교했습니다. 기존 SAM 보정 B 좌표·사용자 표시점·GT 좌표는 경계 검색에 쓰지 않았습니다. <b>붉은 ×는 두 선의 교점 후보이며, 실제 native KP로 채택하거나 PnP에 넣은 점이 아닙니다.</b></p>
<p class="subtle">RGB 방법은 <b>경계 fitting 단계</b>에서 마스크를 사용하지 않습니다. 상위 상태 분류는 세 방법에 공통으로 고정했습니다. 초기 방향 sine가 기존 0.25 이상이면 각각의 incident 경계를 국소 검색하고, 미만이면 얕은 경계의 split을 검색합니다. 이는 <b>초기 모델 기하</b>의 분류이며 사진 경계의 정답 인증이 아닙니다. TinyUNet·MobileSAM 마스크는 둘 다 고정 0.5로 이진화해 점유 여부를 비교했습니다. 얕은 경계의 마스크 arc 방식에서 끝점 거리가 원래 19px 기준을 넘으면 경고를 유지한 채 후보만 계산했습니다. 비교는 이미 본 TEST의 탐색적 2D 평가이며, 전체 자세 정확도·보정된 확률·실제 native 대응의 인증을 제공하지 않습니다.</p>
<div id="population" class="subtle"></div></div>
<div class="box">
<div class="controls"><label>실제 사례 <select id="partition"><option value="ALL">CAL + TEST</option><option value="CAL">CAL</option><option value="TEST">TEST</option></select></label><select id="caseSelect" aria-label="사례 선택"></select><button id="previous">이전</button><button id="next">다음</button></div>
<div class="controls smallGap"><label><input id="zoom" type="checkbox" checked>하단 경계 확대</label><label><input id="model" type="checkbox" checked>초기 모델 guide</label><label><input id="support" type="checkbox" checked>RGB 지원 표본</label><label><input id="cloud" type="checkbox">허용 split 교점들</label><label><input id="annotation" type="checkbox">후평가 주석점 (검색 입력 아님)</label></div>
<div class="legend"><span style="color:#b49914">┄ RGB 관측 표본 / 얕은 경계 검색 trace</span><span style="color:#159f4e">━ 마스크+RGB 관측 표본 / 얕은 경계 외곽 arc</span><span style="color:#0897b1">● 실제 RGB 지원 표본</span><span style="color:#d27810">━ 왼쪽 inlier 선</span><span style="color:#855ad0">━ 오른쪽 inlier 선</span><span style="color:#d43044">× 교점 후보</span><span style="color:#7b8e9f">○ 초기 모델 투영</span></div>
<div id="caseMeta"></div></div>
<div class="views" id="views"></div>
<div class="box"><h2>같은 전체 대상에서의 탐지 범위와 2D 오차</h2><div class="controls"><label>평가 분할 <select id="metricPartition"><option value="TEST">TEST 49 (재사용)</option><option value="CAL">CAL 46</option></select></label></div><div id="metrics"></div><p class="subtle">오차 평균은 후보가 나온 표본에 조건부입니다. 누락 수를 함께 보아야 하며, 공통 합격 표본 표에서 같은 주석점을 비교합니다. U 중 실제 DIRECT 주석 표본은 별도로 표시합니다. native 대응 불일치 사례는 주 지표에서 분리했습니다. 기준 모델 오차도 각 방법과 같은 합격 표본에서 계산했습니다.</p></div>
<div class="box"><h2>대표 실제 사진</h2><img class="figure" src="TARGET_COMPARISON_KO.png" alt="동일 초기 기하에서 RGB, TinyUNet, MobileSAM 경계 비교"><p class="subtle">이 그림의 원선과 교점은 각 방법의 봉인된 결과입니다. 초기 모델 guide는 회색, 실제 지지된 선은 주황·보라, 후보 교점은 붉은 ×로 구분합니다. GT 주석은 그리지 않았습니다.</p></div>
<div class="box"><details><summary>입력 봉인·범위·출처</summary><p><a href="../PROTOCOL.json">고정 비교 프로토콜</a> · <a href="../CAL_SEAL.json">CAL 예측 봉인</a> · <a href="../TEST_SEAL.json">TEST 예측 봉인</a> · <a href="../CAL_SCORE.json">CAL 후평가</a> · <a href="../TEST_SCORE.json">TEST 후평가</a> · <a href="REPORT_RECEIPT.json">보고서 입력·이미지 해시</a> · <a href="CASES.json">표시 데이터</a></p><pre id="provenance"></pre></details></div>
</main><script id="reportData" type="application/json">__BUNDLE__</script><script>
const DATA=JSON.parse(document.getElementById('reportData').textContent), M=DATA.methods;
const el=id=>document.getElementById(id), esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(v,n=2)=>Number.isFinite(v)?v.toFixed(n):'—';
const point=q=>Array.isArray(q)&&q.length===2&&q.every(Number.isFinite);
let active=DATA.cases.find(c=>c.id===DATA.target_id&&c.native_corner===3)||DATA.cases[0], generation=0;
const images=new Map();
function imageFor(path){if(!images.has(path)){const img=new Image();const promise=new Promise((resolve,reject)=>{img.onload=()=>resolve(img);img.onerror=()=>reject(new Error('사진 로드 실패: '+path));});img.src=path;images.set(path,promise);}return images.get(path);}
function fillCases(){const partition=el('partition').value,cases=DATA.cases.filter(c=>partition==='ALL'||c.partition===partition);if(!cases.some(c=>c.key===active?.key))active=cases[0];el('caseSelect').innerHTML=cases.map(c=>`<option value="${esc(c.key)}">${esc(c.partition+' · '+c.id+' · KP'+c.native_corner+' '+c.state)}</option>`).join('');if(active)el('caseSelect').value=active.key;renderCase();}
function branchSegment(branch){if(!branch||!point(branch.normal))return null;const points=(branch.observed_points||[]).filter((p,i)=>branch.inlier_mask?.[i]&&point(p));if(!points.length)return null;const n=branch.normal,t=[-n[1],n[0]],v=points.map(p=>p[0]*t[0]+p[1]*t[1]);return [Math.min(...v),Math.max(...v)].map(s=>[-branch.offset*n[0]+s*t[0],-branch.offset*n[1]+s*t[1]]);}
function viewport(c){const [h,w]=c.image_hw;if(!el('zoom').checked)return [0,w,0,h];const pts=Object.values(c.initial_model_projection).filter(point);for(const m of M){const f=c.methods[m].fit_payload;for(const p of f?.contour||[])if(point(p))pts.push(p);const q=f?.selected_representative?.intersection_xy;if(point(q)&&q[0]>=0&&q[0]<w&&q[1]>=0&&q[1]<h)pts.push(q);}return [Math.max(0,Math.min(...pts.map(p=>p[0]))-24),Math.min(w,Math.max(...pts.map(p=>p[0]))+24),Math.max(0,Math.min(...pts.map(p=>p[1]))-28),Math.min(h,Math.max(...pts.map(p=>p[1]))+28)];}
function draw(canvas,img,c,method){const value=c.methods[method],fit=value.fit_payload||{},selected=fit.selected_representative,bounds=viewport(c),[x0,x1,y0,y1]=bounds;const width=1200,height=Math.max(160,Math.round(width*(y1-y0)/(x1-x0)));canvas.width=width;canvas.height=height;const ctx=canvas.getContext('2d'),sx=width/(x1-x0),sy=height/(y1-y0),xy=p=>[(p[0]-x0)*sx,(p[1]-y0)*sy];ctx.drawImage(img,x0,y0,x1-x0,y1-y0,0,0,width,height);
 function line(points,color,lw=2,dash=[]){const ps=points.filter(point);if(ps.length<2)return;ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.setLineDash(dash);ctx.beginPath();ps.forEach((p,i)=>{const q=xy(p);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.stroke();ctx.setLineDash([]);}
 function dot(p,color,r=3){if(!point(p))return;const q=xy(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(...q,r,0,2*Math.PI);ctx.fill();}
 function cross(p,color,r=8){if(!point(p))return;const q=xy(p);ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(q[0]-r,q[1]-r);ctx.lineTo(q[0]+r,q[1]+r);ctx.moveTo(q[0]-r,q[1]+r);ctx.lineTo(q[0]+r,q[1]-r);ctx.stroke();}
 if(el('model').checked){const ids=[c.neighbor_ids[0],c.native_corner,c.neighbor_ids[1]],guide=ids.map(j=>c.initial_model_projection[j]);line(guide,'#c5cbd3',2,[8,6]);ids.forEach(j=>{const p=c.initial_model_projection[j];dot(p,'#c5cbd3',5);const q=xy(p);ctx.font='18px sans-serif';ctx.fillStyle='#eef1f5';ctx.strokeStyle='#192633';ctx.lineWidth=3;const text='모델 '+j;ctx.strokeText(text,q[0]+8,q[1]-9);ctx.fillText(text,q[0]+8,q[1]-9);});}
 for(const segment of fit.contour_segments||[fit.contour||[]])line(Array.isArray(segment)?segment:(segment.points||segment.contour||[]),method==='RGB_GRADIENT'?'#f4d54c':'#20d46d',2,method==='RGB_GRADIENT'?[6,4]:[]);
 if(el('support').checked)(fit.refined_points||[]).forEach((p,i)=>{if(fit.RGB_eligible?.[i])dot(p,'#38d9ed',2.3);});
 if(el('cloud').checked)(fit.qualified_intersection_xy||[]).forEach(p=>dot(p,'#d3d6df',3));
 if(selected){for(const [side,color]of [['left','#ff992e'],['right','#b280ff']]){const branch=selected[side],segment=branchSegment(branch);if(segment)line(segment,color,4);const pts=(branch?.observed_points||[]).filter((p,i)=>branch.inlier_mask?.[i]);if(el('support').checked)pts.forEach(p=>dot(p,color,1.7));}cross(selected.intersection_xy,'#ff525b',9);}
 if(el('annotation').checked&&c.score.valid_annotation&&point(c.score.annotation_xy)){const q=xy(c.score.annotation_xy);ctx.strokeStyle='#fb65e2';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(q[0],q[1]-10);ctx.lineTo(q[0]+10,q[1]);ctx.lineTo(q[0],q[1]+10);ctx.lineTo(q[0]-10,q[1]);ctx.closePath();ctx.stroke();ctx.font='17px sans-serif';ctx.fillStyle='#fb65e2';ctx.fillText('후평가 주석',q[0]+12,q[1]+20);}}
function methodCard(c,m){const v=c.methods[m],a=v.audit||{},f=v.fit_payload||{},s=f.selected_representative,cov=v.coverage||{},warning=a.endpoint_guard_pass===false,q=s?.intersection_xy;let reason=a.reason||v.status;const bounds=c.image_hw,off=point(q)&&!(q[0]>=0&&q[0]<bounds[1]&&q[1]>=0&&q[1]<bounds[0]);const qualified=s?'두 선 교점 후보 있음':'두 선 교점 후보 보류';return `<section class="method"><h3>${esc(DATA.names[m])}</h3><span class="status">${qualified}</span> <span class="subtle">${esc(v.status)}</span>${warning?`<p class="stat warning"><b>끝점 연관 경고 유지</b> · ${esc((a.nearest_projected_endpoint_distance_px||[]).map(x=>num(x)).join(' / '))}px · 원래 ${num(a.endpoint_distance_guard_px,0)}px 기준 초과<br>경고를 통과 판정으로 바꾸지 않고 선 후보만 계산했습니다.</p>`:''}<canvas id="canvas_${m}" aria-label="${esc(DATA.names[m])} 실제 사진과 검색 결과"></canvas><p class="stat">실제 RGB 지원 ${cov.RGB_supported_samples??0}개 · pair/split 후보 ${cov.searched_splits??0}개 · 양쪽 선 품질 통과 ${cov.two_line_guard_pass??0}개 · 유한 지지 교점 ${cov.finite_supported_pairs??0}개</p>${s?`<p class="stat">교점 후보 [${num(q?.[0])}, ${num(q?.[1])}]px${off?' <b class="warning">영상 밖</b>':''}<br>두 선 sine ${num(s.intersection_sine,3)} · 조건수 ${num(s.normal_matrix_condition_number,2)}${s.intersection_sine<.25?' <b class="warning">얕은 교차 · 위치 불안정</b>':''}</p><p class="stat">왼쪽/오른쪽 RMS ${num(s.left?.inlier_rms_px,3)} / ${num(s.right?.inlier_rms_px,3)}px · inlier ${s.left?.inlier_count??0} / ${s.right?.inlier_count??0}</p>`:`<p class="stat warning">보류 이유: ${esc(reason)}</p>`}<p class="stat">후평가 2D 오차 ${num(c.score.error_px[m])}px${!c.score.native_compatible?' <span class="warning">(native 불일치 진단용)</span>':''} · 후보 검색+fit ${num(v.fit_only_elapsed_ms,1)}ms</p><details><summary>검색·연관·실패 진단 원본</summary><pre>${esc(JSON.stringify(a,null,2))}</pre></details></section>`;}
async function renderCase(){
const c=active,token=++generation;
if(!c){el('views').innerHTML='표시할 후보가 없습니다.';return;}
const threshold=DATA.partitions[c.partition].geometry_dispatch_sine_threshold;
const route=DATA.protocol.topology_dispatch?(c.initial_direction_sine>=threshold?'incident 경계 각각의 국소 검색':'얕은 경계의 전역 split 검색'):'V1 전역 split 검색';
el('caseMeta').innerHTML=`<p><span class="pill">${esc(c.partition)}</span><b>${esc(c.id)} · KP${c.native_corner}</b> · 고정 상태 ${esc(c.state)} · 이웃 ${c.neighbor_ids.join(' / ')}</p><p class="subtle">초기 모델 방향 sine ${num(c.initial_direction_sine,3)} → <b>${route}</b><br>회색 guide는 각 방법에 공통인 <b>초기 모델 투영</b>입니다. 실제 이미지 KP가 아닙니다. 현재 상태·기본 모델·PnP는 바꾸지 않았습니다.<br>후평가 사람 상태: ${esc(c.score.human_state)} · 유효 주석 ${c.score.valid_annotation?'있음':'없음'} · native 대응 ${c.score.native_compatible?'호환':'<b class="warning">불일치: 이미지 2D 사례 진단만 가능</b>'} · 초기 모델의 해당 주석 오차 ${num(c.score.model_projection_error_px)}px</p>`;
el('views').innerHTML=M.map(m=>methodCard(c,m)).join('');
try{const img=await imageFor(c.image);if(token!==generation)return;M.forEach(m=>draw(el('canvas_'+m),img,c,m));}
catch(error){if(token===generation)el('views').insertAdjacentHTML('afterbegin',`<p class="warning">${esc(error.message)}</p>`);}
}
function table(headers,rows){return `<div class="scroll"><table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(x=>`<td>${esc(x)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
function renderMetrics(){
const partition=el('metricPartition').value,p=DATA.partitions[partition],str=p.strata,primary=str.DIRECT_COMPATIBLE,coverage=p.coverage;
let out='<h3 class="smallGap">초기 기하와 주 평가 표본의 범위</h3>';
const groups=p.initial_geometry_groups;
out+=table(['초기 모델 형태','모든 후보','DIRECT·native 호환','RGB 후보','Tiny 후보','SAM 후보'],[['STABLE_INITIAL_GEOMETRY','sine ≥ '+p.geometry_dispatch_sine_threshold],['SHALLOW_INITIAL_GEOMETRY','sine < '+p.geometry_dispatch_sine_threshold]].map(([key,label])=>{const v=groups[key];return [label,v.candidates,v.DIRECT_COMPATIBLE,...M.map(m=>v.selected_DIRECT_COMPATIBLE[m])];}));
out+=`<p class="subtle">초기 모델 방향 sine 범위 ${(p.initial_geometry_sine_range||[]).map(x=>num(x,3)).join('–')}. 사진의 실제 꺾임이나 native 대응을 인증하는 분류가 아닙니다.</p>`;
out+='<h3 class="smallGap">모든 이미지 후보의 단계별 탐지 범위</h3>';
out+=table(['방법','대상','arc/trace','RGB≥6','양쪽 선','교점 후보','19px 경고','경고 포함 후보'],M.map(m=>{const v=coverage[m];return [m,v.eligible,v.contour_found,v.RGB_supported,v.two_line_qualified,v.selected,v.endpoint_association_warnings,v.selected_with_endpoint_warning];}));
out+='<h3 class="smallGap">실제 DIRECT · native 호환 · 유효 주석</h3>';
out+=table(['방법','대상','후보 / 대상','평균 오차(px)','중앙값','P90','같은 표본 초기모델 평균','모델보다 낮은 오차'],M.map(m=>{const v=primary.methods[m];return [m,v.eligible,`${v.accepted} / ${v.eligible}`,num(v.accepted_error.mean_px),num(v.accepted_error.median_px),num(v.accepted_error.P90_px),num(v.initial_model_on_same_accepted.mean_px),v.better_than_initial_model];}));
out+='<h3 class="smallGap">두 방법이 모두 후보를 낸 동일 표본</h3>';
out+=table(['비교','공통 n','대안 평균(px)','SAM 평균(px)','대안 오차가 낮은 수'],Object.entries(primary.pairs).map(([key,v])=>[key,v.n,num(v.alternative_error.mean_px),num(v.SAM_error.mean_px),v.alternative_better]));
out+=`<p class="subtle">세 방법 공통 합격 n=${primary.all_three_matched_n}: ${M.map(m=>m+' '+num(primary.all_three_matched_error[m].mean_px)+'px').join(' · ')}</p>`;
for(const [name,title]of [['U_DIRECT_COMPATIBLE','고정 U 중 실제 DIRECT · native 호환'],['DIRECT_INCOMPATIBLE_DIAGNOSTIC','실제 DIRECT · native 불일치 (주 지표 아님)']]){out+=`<h3 class="smallGap">${title}</h3>`;out+=table(['방법','후보 / 대상','조건부 평균(px)','P90'],M.map(m=>{const v=str[name].methods[m];return [m,`${v.accepted} / ${v.eligible}`,num(v.accepted_error.mean_px),num(v.accepted_error.P90_px)];}));}
out+='<h3 class="smallGap">실패 이유와 실행시간 범위</h3>';
out+=table(['방법','후보 보류 이유 · 수','후보 검색+fit 중앙값(ms)'],M.map(m=>[m,Object.entries(coverage[m].failure_reasons).map(([k,v])=>k+': '+v).join(' · ')||'없음',num(p.fit_only_elapsed_ms[m].median,1)]));
out+='<p class="subtle">시간은 저장된 마스크를 입력으로 받은 경계 검색·선 fitting입니다. 마스크 모델 forward·전체 파이프라인의 속도 비교가 아닙니다. RGB는 새로운 신경망 가중치를 추가하지 않았습니다. 경고 포함 교점을 native KP로 채택한 수와 새 PnP 실행 수는 모두 0입니다.</p>';
const old=DATA.V1_template_reference?.[partition];
if(old){out+='<details><summary>보존된 V1: 전역 split 모형을 모든 모서리에 적용한 진단 참고</summary><p class="subtle">V1은 안정적인 꺾임에도 얕은 경계용 전역 split을 적용했고 Tiny의 원래 soft mask 점유값을 사용했습니다. 최종 V2의 기하별 검색·양쪽 binary mask 비교와 분리해야 합니다. 아래 낮은 검출 수를 마스크 모델의 우열로 해석하지 않습니다.</p>';out+=table(['방법','대상','V1 교점 후보'],M.map(m=>[m,old.coverage[m].eligible,old.coverage[m].selected]));out+='</details>';}
el('metrics').innerHTML=out;
}
el('population').textContent=`CAL ${DATA.partitions.CAL.frames}장 / ${DATA.partitions.CAL.candidates}개 후보 · TEST ${DATA.partitions.TEST.frames}장 / ${DATA.partitions.TEST.candidates}개 후보 · 총 ${DATA.cases.length}개 실제 이미지 사례`;
el('provenance').textContent=JSON.stringify({input_sha256:DATA.input_sha256,scope:{GT_used_for_line_search:false,annotations_only_postseal_scoring:true,renderer_annotation_file_reads:false,new_fit_calls:0,new_model_forwards:0,new_pose_fits:0,native_points_adopted:0,endpoint_warning_threshold_px:19,upstream_state_frozen:true,TEST_reused_exploratory:true},protocol:DATA.protocol},null,2);
el('partition').addEventListener('change',fillCases);el('caseSelect').addEventListener('change',()=>{active=DATA.cases.find(c=>c.key===el('caseSelect').value);renderCase();});for(const id of ['zoom','model','support','cloud','annotation'])el(id).addEventListener('change',renderCase);for(const [id,step]of [['previous',-1],['next',1]])el(id).addEventListener('click',()=>{const select=el('caseSelect');if(!select.options.length)return;select.selectedIndex=(select.selectedIndex+step+select.options.length)%select.options.length;active=DATA.cases.find(c=>c.key===select.value);renderCase();});el('metricPartition').addEventListener('change',renderMetrics);fillCases();renderMetrics();
</script></body></html>'''


def render(comparison=COMPARISON, output=OUTPUT):
    comparison, output = Path(comparison).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve completed report: ' + str(output))
    renderer_sha = C.sha(__file__)
    bundle = prepare_data(comparison)
    if not bundle['cases']:
        raise ValueError('No eligible actual-image cases to display')
    output.mkdir(parents=True)
    (output / 'assets').mkdir()
    asset_bindings = {}
    for asset in bundle['assets']:
        source = C.ROOT / asset['source']
        if C.sha(source) != asset['source_sha256']:
            raise ValueError('Actual photograph changed during render')
        destination = output / asset['asset']
        shutil.copyfile(source, destination)
        if C.sha(destination) != asset['source_sha256']:
            raise ValueError('Original actual image copy changed')
        asset_bindings[asset['asset']] = asset['source_sha256']
    write_new(output / 'CASES.json', bundle)
    summary = {key: bundle[key] for key in ('schema', 'partitions', 'input_sha256')}
    write_new(output / 'COMPARISON_SUMMARY.json', summary)
    encoded = json.dumps(bundle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    # Inline JSON keeps this fully functional on local file:// without fetch/CORS.
    encoded = encoded.replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    (output / 'index.html').write_text(HTML.replace('__BUNDLE__', encoded), encoding='utf-8')
    figure = representative_png(output, bundle)
    verify_bindings(bundle['input_sha256'])
    verify_bindings(bundle['protocol']['source_sha256'])
    verify_bindings(bundle['protocol']['protected_user_sha256'])
    if C.sha(__file__) != renderer_sha:
        raise ValueError('Renderer source changed during execution')
    evidence = {name: C.sha(output / name) for name in
                ('index.html', 'CASES.json', 'COMPARISON_SUMMARY.json')}
    if figure:
        evidence[figure['path']] = figure['sha256']
    receipt = dict(
        schema='LOWER_BOUNDARY_METHOD_REPORT_RECEIPT_V1', passed=True,
        renderer_source=str(Path(__file__).resolve().relative_to(C.ROOT)),
        renderer_source_sha256=renderer_sha, output=str(output.relative_to(C.ROOT)),
        input_sha256=bundle['input_sha256'], output_sha256=evidence,
        original_image_asset_sha256=asset_bindings,
        cases=len(bundle['cases']), actual_photographs=len(asset_bindings),
        frames={'CAL': bundle['partitions']['CAL']['frames'], 'TEST': bundle['partitions']['TEST']['frames']},
        candidates={'CAL': bundle['partitions']['CAL']['candidates'], 'TEST': bundle['partitions']['TEST']['candidates']},
        default_target=TARGET, figure=figure,
        plotted_lines='saved selected method lines over actual inlier tangent support',
        projected_model_point='shared initial model only, never a measured native point',
        original_image_copies_exact=True, annotation_default_visible=False,
        renderer_annotation_file_reads=False, new_fits=0, new_inference_calls=0,
        new_pose_fits=0, native_keypoints_adopted=0,
        endpoint_warning_preserved=True, confidence_or_full_pose_claim=False,
        browser_execution_checked=False)
    write_new(output / 'REPORT_RECEIPT.json', receipt)
    print(json.dumps(C.finite(dict(passed=True, output=str(output), cases=len(bundle['cases']),
                                  photographs=len(asset_bindings), figure=figure)), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir', '--comparison', dest='comparison', type=Path, default=COMPARISON)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    render(args.comparison, args.output or args.comparison / 'report_v1')


if __name__ == '__main__':
    main()
