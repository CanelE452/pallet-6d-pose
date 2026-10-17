"""Post-seal full-scene displays of the frozen visible-support pilot.

Only saved inputs, old mask/censor records, actual selected candidates and
post-seal evaluator scalars are read. No observer, cost, ranking, fit or neural
model is called. Pillow draws directly in original pixel coordinates.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


VARIANTS = ('BASELINE_E', 'HULL_ONEWAY', 'HULL_ONEWAY_MASS', 'VISIBLE_SUPPORT')
LABELS = ('기존 전체 e', '원 hull 단방향', '같은 질량 단방향', '새 가시 경계')
COLORS = ('#0072b2', '#d59100', '#9859b6', '#d84a35')
FAMILIES = ('ORIGINAL_WD', 'SWAPPED_WD')
STAGES = ('G10', 'G13')
PILOT = '_docs/experiments/pallet_non_target_outline_pilot_v1'
FONT_PATH = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def csvrows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(path, digest):
    require(sha(path) == digest, 'Bound input changed: '+Path(path).name)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def font(size):
    require(FONT_PATH.is_file(), 'Previously installed NanumGothic font required')
    return ImageFont.truetype(str(FONT_PATH), size)


def text(draw, point, message, fill='#142c3e', size=16):
    draw.text(point, message, font=font(size), fill=fill, stroke_width=0)


def line(image, xy, fill, width=2):
    # No resize, crop, letterbox, closure or hidden-span interpolation here.
    ImageDraw.Draw(image).line([tuple(map(float, p)) for p in xy], fill=fill, width=width)


def polylines(image, observation, fill='#00ede0', width=2, labels=False):
    draw = ImageDraw.Draw(image)
    for polyline in observation['polylines']:
        line(image, polyline['points_xy'], fill, width)
        if labels:
            x, y = polyline['points_xy'][0]
            draw.text((x+3, y-18), polyline['line_id'], font=font(12), fill=fill,
                      stroke_width=1, stroke_fill='#111111')


def tinted(rgb, mask, color, alpha=.35):
    result = np.asarray(rgb, float).copy()
    require(mask.shape == result.shape[:2], 'Mask display must use original canvas')
    from PIL import ImageColor
    result[mask] = result[mask]*(1-alpha)+np.array(ImageColor.getrgb(color))*alpha
    return Image.fromarray(np.rint(result).astype(np.uint8))


def boxed(image, title, subtitle='', footer=''):
    w, h = image.size
    head, foot = 70, 34
    out = Image.new('RGB', (w, h+head+foot), '#ffffff')
    out.paste(image, (0, head))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, w, head-1), fill='#e6edf2')
    text(draw, (12, 8), title, size=19)
    text(draw, (12, 37), subtitle, size=14)
    text(draw, (12, head+h+7), footer, size=13)
    return out


def composite(images, columns=4):
    width, height = images[0].size
    require(all(image.size == (width, height) for image in images), 'Parallel panel sizes differ')
    nrows = (len(images)+columns-1)//columns
    out = Image.new('RGB', (columns*width, nrows*height), '#ffffff')
    for index, image in enumerate(images):
        out.paste(image, ((index%columns)*width, (index//columns)*height))
    return out


def project(candidate, K):
    xyz, R, t, K = map(lambda v: np.asarray(v, float),
                       (candidate['native_xyz'], candidate['R'], candidate['t'], K))
    require(xyz.shape == (8, 3), 'Own native cuboid XYZ required')
    camera = xyz@R.T+t
    require(np.min(camera[:, 2]) > 0, 'Selected original candidate has nonpositive depth')
    homogeneous = camera@K.T
    xy = homogeneous[:, :2]/homogeneous[:, 2:3]
    # Separate homogeneous-matrix formulation guards drawing convention mistakes.
    transform = np.column_stack((R, t))
    second = (K@transform@np.column_stack((xyz, np.ones(8))).T).T
    independently = second[:, :2]/second[:, 2:3]
    gap = float(np.max(np.abs(xy-independently)))
    require(gap <= 1e-9, 'Independent projection formulations disagree')
    edges = [(i, j) for i in range(8) for j in range(i+1, 8)
             if np.count_nonzero(~np.isclose(xyz[i], xyz[j], atol=1e-12, rtol=0)) == 1]
    require(len(edges) == 12, 'Native cuboid must have twelve edges')
    return xy, edges, gap


def draw_model(image, xy, edges, color):
    draw = ImageDraw.Draw(image)
    for i, j in edges:
        line(image, xy[[i, j]], color, 2)
    for index, point in enumerate(xy):
        x, y = map(float, point)
        if 0 <= x < image.width and 0 <= y < image.height:
            draw.text((x+2, y+2), str(index), font=font(11), fill=color,
                      stroke_width=1, stroke_fill='#ffffff')


def draw_old_hull(image, outline, parents=None):
    points = outline['polygon_xy']
    line(image, points+[points[0]], '#edb400', 2)
    if parents is not None:
        for parent in parents:
            line(image, parent['xy'], '#ed197b' if parent['censor'] else '#17b970', 3)


def roi_for_display(frame, observation, target):
    """Explicit fixed input-only crop, never conditioned on selected results."""
    points = [np.asarray(poly['points_xy']) for poly in observation['polylines']]
    for stage in STAGES:
        for family in FAMILIES:
            outline = frame['outlines'][stage][family]
            if outline.get('supported'):
                points.append(np.asarray(outline['polygon_xy']))
    yy, xx = np.nonzero(target)
    if len(xx):
        points.append(np.array([[xx.min(), yy.min()], [xx.max(), yy.max()]]))
    require(bool(points), 'No original support available for optional display crop')
    p = np.concatenate(points)
    h, w = frame['image_hw']
    return [max(0, int(np.floor(p[:, 0].min()))-28),
            max(0, int(np.floor(p[:, 1].min()))-28),
            min(w, int(np.ceil(p[:, 0].max()))+29),
            min(h, int(np.ceil(p[:, 1].max()))+29)]


def metric_text(metric):
    return f'위치 {float(metric["translation_cm"]):.2f}cm / 회전 {float(metric["rotation_deg"]):.2f}°'


def cell(value):
    return html.escape(str(value))


def _render(private_root, run, evaluation, source_root, prior_root, output, guard):
    started, cpu = time.monotonic(), time.process_time()
    root, run, evaluation, source, prior, out = map(Path,
        (private_root, run, evaluation, source_root, prior_root, output))
    require(not out.exists(), 'Preserve previous galleries; choose a fresh output folder')
    package_path = root/'neutral_v1/NEUTRAL_PACKAGE.json'
    package, seal = read(package_path), read(run/'PREDICTION_SEAL.json')
    lock = read(root/'PROTOCOL_LOCK.json')
    bound(root/'VISIBLE_INPUTS.jsonl', lock['input_sha256'])
    require(lock['input_sha256'] == seal['input_manifest_sha256'], 'New input changed after freeze')
    bound(package_path, seal['neutral_package_sha256'])
    require(seal['frames'] == 6 and seal['candidates'] == 351 and seal['prediction_rows'] == 24,
            'All six/full bank/four arms required')
    require(seal['variants'] == list(VARIANTS) and seal['reference_free_selection'] and
            seal['before_new_reference_evaluation'], 'Prediction/evaluation boundary required')
    for name, digest in seal['artifacts'].items():
        bound(run/name, digest)
    receipt = read(evaluation/'EVALUATION_RECEIPT.json')
    require(receipt['prediction_seal_sha256'] == sha(run/'PREDICTION_SEAL.json') and
            receipt['predictions_sealed_before_reference_open'], 'Wrong post-seal evaluator artifacts')
    for name, digest in receipt['artifacts'].items():
        bound(evaluation/name, digest)
    for role in ('frames', 'candidates', 'RGB'):
        bound(package_path.parent/package[role]['path'], package[role]['sha256'])
    frames = rows(package_path.parent/package['frames']['path'])
    candidates = rows(package_path.parent/package['candidates']['path'])
    require([frame['id'] for frame in frames] == package['ids'], 'Original six/order required')
    by_candidate = {(c['id'], c['candidate_id']): c for c in candidates}
    require(len(candidates) == len(by_candidate) == 351, 'Candidate identity not unique')
    inputs = {p['id']: p for p in rows(root/'VISIBLE_INPUTS.jsonl')}
    predictions = {(p['id'], p['variant']): p for p in rows(run/'PREDICTIONS.jsonl.gz')}
    scores = {(p['id'], p['candidate_id']): p for p in rows(run/'CANDIDATE_SCORES.jsonl.gz')}
    metrics = {(p['id'], p['variant']): p for p in csvrows(evaluation/'METRICS.csv')}
    require(len(predictions) == len(metrics) == 24 and len(scores) == 351, 'Incomplete saved outputs')
    prior_run = prior/'actual_rerank_v2'
    prior_seal = read(prior_run/'PREDICTION_SEAL.json')
    bound(prior_run/'PARENT_SUPPORT.jsonl', prior_seal['artifacts']['PARENT_SUPPORT.jsonl'])
    parents = rows(prior_run/'PARENT_SUPPORT.jsonl')
    prior_input_path = prior/'frozen_inputs_v1/INPUTS_FROZEN.jsonl'
    bound(prior_input_path, prior_seal['input_manifest_sha256'])
    prior_inputs = {p['id']: p for p in rows(prior_input_path)}
    permission_path = source/PILOT/'gallery/PUBLICATION_PERMISSION.json'
    permission = read(permission_path)
    permissions = {p['path']: p for p in permission['original_members']}
    require(len(permissions) == 12 and permission['derived_displays_only'], 'Prior exact publication permission required')
    out.mkdir(parents=True)
    initial_articles, articles, proofs = [], [], []
    for frame in frames:
        ident, ordinal = frame['id'], frame['ordinal']
        folder = out/f'case_{ordinal:02d}'
        folder.mkdir()
        inp = inputs[ident]
        require(inp['image_sha256'] == frame['image_sha256'] and inp['image_hw'] == frame['image_hw'], 'Input RGB binding differs')
        rgb_path = source/frame['image']
        bound(rgb_path, frame['image_sha256'])
        require(permissions[frame['image']]['sha256'] == frame['image_sha256'], 'RGB publication binding differs')
        rgb = Image.open(rgb_path).convert('RGB')
        require([rgb.height, rgb.width] == frame['image_hw'], 'Original canvas changed')
        # Only full-scene recompressed JPEG and derived annotations are published.
        rgb.save(folder/'scene_RGB_unmarked.jpg', quality=94)
        filename = f'{ordinal:02d}_{ident.replace(":", "_")}_comparison.png'
        original_sheet = root/'input_only'/filename
        bound(original_sheet, lock['input_only_sheet_sha256'][filename])
        shutil.copyfile(original_sheet, folder/'frozen_input_only.png')
        require(sha(original_sheet) == sha(folder/'frozen_input_only.png'), 'Frozen input-only preview changed')
        relative_mask = next(p['path'] for p in permission['original_members']
            if p['kind'] == 'original_target_mask' and Path(p['path']).stem == ident.replace(':', '__'))
        mask_path = source/relative_mask
        bound(mask_path, permissions[relative_mask]['sha256'])
        with np.load(mask_path, allow_pickle=False) as archive:
            target = archive['pallet_mask_clean'].copy()
            require(np.array_equal(target, archive['pallet_mask']), 'Original clean mask alias differs')
        require(set(np.unique(target)).issubset({0, 1}), 'Original target mask not binary')
        target = target.astype(bool)
        roi = roi_for_display(frame, inp, target)
        old_input = prior_inputs[ident]
        foreign = None
        if old_input['admissible_for_censoring']:
            bound(old_input['mask_path'], old_input['mask_sha256'])
            foreign = np.load(old_input['mask_path'], allow_pickle=False)
            require(set(np.unique(foreign)).issubset({0, 1}) and foreign.shape == target.shape,
                    'Original old foreign mask canvas differs')
            bound_pixel = pixel_sha(foreign.astype(np.uint8))
            require(bound_pixel == old_input['pixel_bytes_sha256'], 'Old foreign mask changed')
            foreign = foreign.astype(bool)
        baseline = predictions[(ident, 'BASELINE_E')]
        baseline_candidate = by_candidate[(ident, baseline['candidate_id'])]
        stage, family = baseline_candidate['stage'], baseline_candidate['family']
        outline = frame['outlines'][stage][family]
        context = [p for p in parents if (p['id'], p['stage'], p['family']) == (ident, stage, family)]
        require(np.array_equal([p['xy'] for p in context], outline['observed_boundary_segments_xy']), 'Old finite parents changed')
        mask_display = tinted(rgb, target, '#52b2ff', .42)
        hull_display = rgb.copy()
        draw_old_hull(hull_display, outline)
        old_display = rgb.copy() if foreign is None else tinted(rgb, foreign, '#b65bdb', .25)
        draw_old_hull(old_display, outline, context)
        visible_display = rgb.copy()
        polylines(visible_display, inp, labels=True)
        history = [boxed(mask_display, '원 target mask', '기존 배열 그대로 · 새 관측 생성에 사용 안 함', '파랑 채움 = 기존 대상 mask'),
            boxed(hull_display, '원 관측 hull', f'{stage}/{family} · 원 polygon 그대로', '노랑 = 원 닫힌 hull (새 입력 아님)'),
            boxed(old_display, '이전 부모 전체 제외', f'원 부모 {len(context)}개 · 결측 {sum(p["censor"] for p in context)}개',
                  '자홍 = 예전 결측 부모 · 초록 = 예전 유지'),
            boxed(visible_display, '새 RGB 가시 경계', f'{len(inp["polylines"])}개 열린 polyline · 모델 보조 관측',
                  '청록 = 실제 표시 구간 · 끊긴 틈 연결 없음')]
        composite(history).save(folder/'observation_history_full.jpg', quality=94)
        proof_models, full_panels, crop_panels, cards = [], [], [], []
        for variant, label, color in zip(VARIANTS, LABELS, COLORS):
            pred, metric = predictions[(ident, variant)], metrics[(ident, variant)]
            candidate = by_candidate[(ident, pred['candidate_id'])]
            cost = scores[(ident, pred['candidate_id'])]
            require(candidate['family'] == pred['selected_family'] == metric['family'] and
                    metric['candidate_id'] == pred['candidate_id'], 'Selected candidate/family metric binding differs')
            require(np.allclose(pred['pose']['R_cf'], candidate['R'], atol=1e-9, rtol=0) and
                    np.allclose(pred['pose']['centroid'], candidate['t'], atol=1e-9, rtol=0), 'Selected native pose differs')
            require(np.allclose(pred['pose']['R_physical'],
                    np.asarray(candidate['R'])@np.asarray(candidate['physical_registration_Q']), atol=1e-9, rtol=0), 'Physical Q convention differs')
            xy, edges, gap = project(candidate, frame['K'])
            if cost['visible_evidence'] is not None:
                require(np.allclose(xy, cost['visible_evidence']['model_projected_vertices_xy'], atol=1e-9, rtol=0), 'Renderer model differs from scored model')
            displayed = rgb.copy()
            draw_model(displayed, xy, edges, color)
            metric_line = metric_text(metric)
            full = boxed(displayed, label, metric_line,
                         f'{candidate["stage"]}/{candidate["family"]} · 모델 투영선 (영상 관측 아님)')
            full_panels.append(full)
            full.save(folder/f'{variant}_full.jpg', quality=95)
            # Explicit nearest-neighbour enlargement. Full scene is always first.
            crop = displayed.crop(tuple(roi))
            scale = min(640/crop.width, 480/crop.height)
            resized = crop.resize((round(crop.width*scale), round(crop.height*scale)), Image.Resampling.NEAREST)
            crop_canvas = Image.new('RGB', (640, 480), '#e2e4e6')
            dx, dy = (640-resized.width)//2, (480-resized.height)//2
            crop_canvas.paste(resized, (dx, dy))
            scale_x, scale_y = resized.width/crop.width, resized.height/crop.height
            # Pillow resampling aligns pixel centres: the +.5/- .5 terms
            # matter when mapping original integer-centre annotations through
            # this optional enlargement. Full-scene panels stay identity.
            matrix = np.array([[scale_x, 0, dx-scale_x*roi[0]+.5*scale_x-.5],
                               [0, scale_y, dy-scale_y*roi[1]+.5*scale_y-.5], [0, 0, 1]])
            raw_vertices = np.concatenate([np.asarray(p['points_xy'], float) for p in inp['polylines']])
            raw_h = np.column_stack((raw_vertices, np.ones(len(raw_vertices))))
            roundtrip = (np.linalg.inv(matrix)@(matrix@raw_h.T)).T
            roundtrip_gap = float(np.max(np.abs(roundtrip[:, :2]-raw_vertices)))
            require(roundtrip_gap <= 1e-9, 'Explicit crop coordinate inverse differs')
            crop_panels.append(boxed(crop_canvas, label+' · 명시적 확대', metric_line,
                f'원 crop {roi} · 동일 crop/축척 · scale={scale:.6f}'))
            ev = cost['visible_evidence']
            proof_models.append(dict(variant=variant, candidate_id=pred['candidate_id'],
                stage=candidate['stage'], family=candidate['family'], native_xyz=candidate['native_xyz'],
                R=candidate['R'], t=candidate['t'], physical_registration_Q=candidate['physical_registration_Q'],
                projected_native_corners_xy=xy.tolist(), native_cuboid_edges=edges,
                independently_projected_max_abs_px=gap, selected_cost=cost['scores'][variant],
                original_components=cost['original_weighted_components'], C=cost['non_outline_cost'],
                r=cost['r'], Z_H=cost['old_observed_mass_Z_H'], Z_E=cost['visible_mass_Z_E'],
                visible_weighted_cost=cost['replacement_outline_costs']['VISIBLE_SUPPORT'],
                A_E=None if ev is None else ev['robust_mean_A_E'],
                raw_visible_distance_mean_px=None if ev is None else ev['raw_distance_mean_px'],
                raw_visible_distance_max_px=None if ev is None else ev['raw_distance_max_px'],
                evaluator_metric=metric, treatment_active=pred['treatment_active'],
                no_input_identity=pred['no_input_identity'], accepted=pred['accepted'], fallback=pred['fallback'],
                original_constraint_pass=pred['original_constraint_pass'],
                recomputed_constraint_pass=pred['recomputed_constraint_pass'],
                minimum_constraint_slack=pred['minimum_constraint_slack'],
                original_stage_winner_ids=pred['stage_winner_ids'],
                crop_original_xyxy=roi, crop_scale=scale,
                original_to_crop_display=matrix.tolist(), crop_coordinate_convention='integer pixel centres; Pillow half-pixel resize alignment',
                crop_roundtrip_max_abs_original_px=roundtrip_gap,
                crop_integer_resize_xy_scale=[resized.width/crop.width, resized.height/crop.height]))
            components = cost['original_weighted_components']
            cards.append(f'<section><h4>{cell(label)}</h4><p>{cell(metric_line)}<br>'
                f'참조상 성공={cell(metric["success_le5"])} · 실제 처치={pred["treatment_active"]}<br>'
                f'후보 {cell(pred["candidate_id"])}<br>family {cell(pred["selected_family"])}<br>'
                f'기존과 후보 변경={pred["candidate_id"] != baseline["candidate_id"]} · 무입력 유지={pred["no_input_identity"]}<br>'
                f'full={components["full_BI"]:.8g}, partial={components["partial_normal"]:.8g}, prior={components["prior"]:.8g}<br>'
                f'C={cost["non_outline_cost"]:.8g} · 교체 외곽={cost["replacement_outline_costs"][variant]:.8g}<br>'
                f'G13 선택 점수={cost["scores"][variant]:.8g} · r={cost["r"]}<br>'
                f'accepted={pred["accepted"]}, fallback={pred["fallback"]}</p>'
                f'<a href="case_{ordinal:02d}/{variant}_full.jpg">개별 전체 장면</a></section>')
        final_strip = composite(full_panels)
        final_strip.save(folder/'four_final_paths_full.jpg', quality=94)
        composite(crop_panels).save(folder/'four_final_paths_explicit_zoom.jpg', quality=94)
        composite(history+full_panels).save(folder/'process_comparison_full.jpg', quality=94)
        # A separate observed-versus-projected display; the two are never one label.
        visible_pred = predictions[(ident, 'VISIBLE_SUPPORT')]
        visible_candidate = by_candidate[(ident, visible_pred['candidate_id'])]
        q, edges, _ = project(visible_candidate, frame['K'])
        combined = rgb.copy()
        draw_model(combined, q, edges, COLORS[-1])
        polylines(combined, inp, fill='#00ede0', width=2)
        boxed(combined, '가시 관측과 실제 선택 모델을 구분', '청록 열린 선 = 입력 / 빨강 직육면체 = 모델 투영',
              '표시점 끝은 물리 코너 정답이 아님 · 모델선 전체가 보인다는 뜻 아님').save(folder/'visible_observed_vs_model_full.jpg', quality=95)
        length = sum(np.linalg.norm(np.diff(np.asarray(p['points_xy'], float), axis=0), axis=1).sum() for p in inp['polylines'])
        initial_articles.append(f'<article id="input{ordinal}"><h2>{ordinal}. {cell(ident)}</h2>'
            f'<p>ASSISTED_MODEL · {len(inp["polylines"])}개 열린 관측 · 총 길이 {length:.2f}px. '
            '후보·참조·점수·성공 여부를 제공하지 않은 별도 모델 문맥에서 작성했습니다. 기존 선정 사례라는 한계는 유지합니다.</p>'
            f'<img loading="lazy" src="case_{ordinal:02d}/frozen_input_only.png" alt="원 RGB와 동결된 보조 관측만">'
            f'<details><summary>미확정 구간과 원 영상</summary><p>{cell(inp["unresolved_regions"])}</p>'
            f'<img loading="lazy" src="case_{ordinal:02d}/scene_RGB_unmarked.jpg" alt="전체 원 RGB 재압축 표시"></details></article>')
        articles.append(f'<article id="case{ordinal}"><h2>{ordinal}. {cell(ident)} · 네 경로 비교</h2>'
            f'<p><a href="#input{ordinal}">먼저 입력만 보기</a> · 원 full RGB 640×480 · 추가 letterbox 없음. '
            f'확대는 원 {cell(roi)} crop이며 전체 장면을 먼저 표시합니다.</p>'
            f'<h3>1. 관측의 출처를 분리</h3><a href="case_{ordinal:02d}/observation_history_full.jpg">'
            f'<img loading="lazy" src="case_{ordinal:02d}/observation_history_full.jpg" alt="원 mask, 원 hull, 예전 결측, 새 가시 경계"></a>'
            '<p>이전 cone mask와 부모 제외는 동결 후 비교 표시용으로 읽었습니다. 새 입력은 mask 차집합이나 hull이 아닙니다. '
            '자홍 부모가 모두 비대상 물체라는 뜻은 아닙니다. 실제 팔레트 지지도 함께 잃었을 수 있습니다.</p>'
            f'<h3>2. 봉인된 네 최종 선택 · 모두 같은 전체 장면</h3><a href="case_{ordinal:02d}/four_final_paths_full.jpg">'
            f'<img loading="lazy" src="case_{ordinal:02d}/four_final_paths_full.jpg" alt="네 경로 실제 선택 자세 전체 장면"></a><div class="cards">'
            + ''.join(cards) + '</div><p>모델선은 각 선택 후보의 native XYZ, R/t, K로 투영했습니다. Q를 다시 곱하지 않습니다. '
            '오차는 사후 기존 기하 참조 기준이며 독립 물리 정답이 아닙니다. G10과 G13 총점은 직접 비교하지 않습니다.</p>'
            f'<h3>3. 동일한 입력 기준 crop 확대</h3><a href="case_{ordinal:02d}/four_final_paths_explicit_zoom.jpg">'
            f'<img loading="lazy" src="case_{ordinal:02d}/four_final_paths_explicit_zoom.jpg" alt="동일한 원좌표 crop의 네 자세 확대"></a>'
            f'<h3>4. 입력과 선택 모델을 별도로 표시</h3><img loading="lazy" src="case_{ordinal:02d}/visible_observed_vs_model_full.jpg" '
            'alt="청록 가시 관측과 빨강 선택 모델을 분리"><p>청록 관측을 물리 직육면체 외곽으로 인증한 것이 아닙니다. '
            '표본은 선분의 수치적 길이 가중이며 독립 증거 수가 아닙니다.</p>'
            f'<details><summary>출처→선택 과정을 한 장으로 병렬 표시</summary><a href="case_{ordinal:02d}/process_comparison_full.jpg">'
            f'<img loading="lazy" src="case_{ordinal:02d}/process_comparison_full.jpg"></a></details></article>')
        contexts = []
        for context_stage in STAGES:
            for context_family in FAMILIES:
                original_outline = frame['outlines'][context_stage][context_family]
                old_parents = [p for p in parents if (p['id'], p['stage'], p['family']) ==
                               (ident, context_stage, context_family)]
                require(np.array_equal([p['xy'] for p in old_parents], original_outline['observed_boundary_segments_xy']),
                        'One original stage/family parent context differs')
                contexts.append(dict(stage=context_stage, family=context_family,
                    old_polygon_xy=original_outline['polygon_xy'], old_finite_parents=old_parents,
                    original_outline_configuration=original_outline['configuration']))
        proofs.append(dict(id=ident, ordinal=ordinal, image_relative_path=frame['image'],
            image_sha256=frame['image_sha256'], original_canvas_hw=frame['image_hw'], K=frame['K'],
            full_scene_display_crop_or_resize=False, full_scene_is_recompressed_JPEG=True,
            input_only_sheet_sha256=sha(original_sheet), frozen_input_status=inp['status'],
            input_author_kind=inp['author_kind'], input_mode=inp['input_mode'],
            raw_open_polylines=inp['polylines'], unresolved_regions=inp['unresolved_regions'],
            original_xy_to_full_overlay=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            new_observed_hull_created=False, hidden_spans_joined=False, new_segment_fit=False,
            target_mask_relative_path=relative_mask, target_mask_sha256=permissions[relative_mask]['sha256'],
            target_binary_pixel_sha256=pixel_sha(target.astype(np.uint8)),
            old_censor_display_context=dict(stage=stage, family=family),
            old_foreign_mask_used_for_display_only=foreign is not None,
            old_foreign_mask_pixel_sha256=None if foreign is None else pixel_sha(foreign.astype(np.uint8)),
            all_original_contexts=contexts, selected_models=proof_models))
    style = '<style>body{font-family:sans-serif;max-width:1900px;margin:20px auto;padding:16px;background:#eef2f5;line-height:1.6}header,article{background:white;padding:20px;margin:18px 0;border-radius:8px}img{max-width:100%;height:auto}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.cards section{padding:12px;background:#edf1f5;overflow-wrap:anywhere;font-size:13px}details{margin:12px 0}a{color:#07558a}@media(max-width:900px){.cards{display:block}}</style>'
    common = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'+style
    introduction = ('<header><h1>보이는 팔레트 경계 · 입력 보조 6장 기능 시험</h1><p>먼저 여섯 장의 원 RGB와 동결 입력만 보여줍니다. '
        '그 아래에서 원 mask/hull, 예전 부모 제외, 새 관측, 봉인된 네 실제 선택을 나누어 표시합니다. '
        '자동 추출 성능·독립 blind 평가·물리 GT가 아닙니다.</p><p>사진을 클릭하면 원 크기로 볼 수 있습니다. '
        '전체 장면을 생략하지 않으며, 확대 crop은 별도로 표시합니다.</p>'
        '<p><a href="initial_index.html">입력 전용 페이지</a> · <a href="../REPORT_KO.md">결과 보고서</a> · '
        '<a href="GALLERY_PROOF.json">출처·좌표·native 모델 투영 장부</a></p></header>')
    input_html = ''.join(initial_articles)
    (out/'initial_index.html').write_text(common+'<title>가시 경계 동결 입력만 보기</title>'+introduction+
        input_html+'</html>')
    (out/'index.html').write_text(common+'<title>가시 경계 네 경로 비교</title>'+introduction+input_html+
        '<header><h1>동결 뒤 결과: 같은 351개 후보의 네 선택</h1><p>이 아래부터 참조 오차와 실제 선택을 표시합니다. '
        '새 점수는 원 outline을 대체했으며 full B/I, partial normal, prior와 원 제약은 보존했습니다.</p></header>'+
        ''.join(articles)+'</html>')
    proof = dict(schema='VISIBLE_PALLET_BOUNDARY_POSTSEAL_GALLERY_V1', cases=proofs,
        prediction_seal_sha256=sha(run/'PREDICTION_SEAL.json'),
        evaluation_receipt_sha256=sha(evaluation/'EVALUATION_RECEIPT.json'),
        neutral_package_sha256=sha(package_path), frozen_input_manifest_sha256=sha(root/'VISIBLE_INPUTS.jsonl'),
        protocol_lock_sha256=sha(root/'PROTOCOL_LOCK.json'),
        previous_censor_seal_sha256=sha(prior_run/'PREDICTION_SEAL.json'),
        previous_censor_parents_sha256=sha(prior_run/'PARENT_SUPPORT.jsonl'),
        prior_publication_permission_sha256=sha(permission_path), prior_publication_permission=permission,
        input_only_before_any_pose_or_evaluation_in_index=True,
        no_reference_files_opened_by_renderer=True,
        new_NN_PnP_observation_cost_or_rank_calls=0)
    write(out/'GALLERY_PROOF.json', proof)
    decoded = 0
    for path in out.rglob('*'):
        if path.suffix in ('.jpg', '.png'):
            with Image.open(path) as image:
                image.load()
            decoded += 1
    result = dict(schema='VISIBLE_PALLET_BOUNDARY_GALLERY_ACTUAL_COST_V1', passed=True,
        cases=len(proofs), final_selected_panels=24, original_candidate_bank_count=351,
        frozen_input_only_previews_preserved=6, full_scene_RGB_and_24_projection_panels=True,
        full_scene_overlays_identity_original_pixel_transform=True,
        explicit_input_based_crop_only=True, decoded_images=decoded,
        actual_treated_visible_controls=sum(p['treatment_active'] for c in proofs if c['ordinal'] in (5, 6)
            for p in c['selected_models'] if p['variant'] == 'VISIBLE_SUPPORT'),
        no_new_model_or_fit_or_cost_or_selection=True, runtime_guard=guard.receipt(),
        independent_projection_max_abs_px=max(p['independently_projected_max_abs_px'] for c in proofs for p in c['selected_models']),
        elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
        maximum_resident_set_size_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=sha(__file__),
        assets={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    write(out/'GALLERY_RECEIPT.json', result)
    return result


def render(private_root, run, evaluation, source_root, prior_root, output):
    from scripts.research.pallet_outline_physical_support_audit_v1.audit_guard import SupportAuditOnly
    with SupportAuditOnly() as guard:
        return _render(private_root, run, evaluation, source_root, prior_root, output, guard)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('private-root', 'run', 'evaluation', 'source-root', 'prior-root', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    result = render(args.private_root, args.run, args.evaluation, args.source_root, args.prior_root, args.output)
    print(json.dumps({k: result[k] for k in ('passed', 'cases', 'final_selected_panels', 'decoded_images',
        'elapsed_seconds', 'process_cpu_seconds', 'maximum_resident_set_size_kib')}, ensure_ascii=False))
