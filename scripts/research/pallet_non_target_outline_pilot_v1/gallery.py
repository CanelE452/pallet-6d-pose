"""Post-seal six-case display of frozen inputs, parent censoring and outputs.

Reads the three published selections and evaluator scalars. It never calls a
supplier, cost evaluator, nearest-segment search, selector or pose solver.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image


VARIANTS = ('BASELINE_E', 'SUPPORT_AWARE_RERANK', 'MASS_MATCHED_CONTROL')
LABELS = ('기존 전체 e', '비대상 부모 결측 처리', '같은 유지 mass의 전체 약화')
COLORS = ('#0072b2', '#d55e00', '#9b59b6')
STAGES = ('G10', 'G13')
FAMILIES = ('ORIGINAL_WD', 'SWAPPED_WD')
AUDIT = '_docs/experiments/pallet_outline_physical_support_audit_v1'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def require(condition, description):
    if not condition:
        raise ValueError(description)


def csvrows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def bound(path, digest):
    require(sha(path) == digest, 'Original bound file changed: ' + Path(path).name)


def permission(source_root, export):
    prior = read(source_root / AUDIT / 'INPUT_BINDINGS.json')['publication_permission']
    inventory = source_root / prior['path']
    bound(inventory, prior['sha256'])
    allowed = {r['relpath']: r for r in prior['selected_members']}
    result = []
    for asset in export['original_assets']:
        path = Path(asset['path'])
        relative = str(path.relative_to(source_root))
        entry = allowed.get(relative)
        require(entry is not None and entry['prior_release_exact'] is True,
                'Original RGB/mask publication permission not established')
        require(entry['sha256'] == asset['sha256'], 'Permission asset SHA differs')
        bound(path, asset['sha256'])
        result.append(dict(path=relative, sha256=asset['sha256'], kind=asset['kind'],
                           prior_release_exact=True))
    require(len(result) == 12, 'All six RGB and target-mask permissions required')
    return dict(original_inventory_path=prior['path'], original_inventory_sha256=prior['sha256'],
                original_members=result, raw_full_RGB_and_NPZ_not_copied=True,
                derived_displays_only=True)


def plot_library():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # The existing font may not be in an old Matplotlib cache. Register the
    # already installed file explicitly; no package/download is performed.
    local_font = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    if local_font.is_file():
        font_manager.fontManager.addfont(str(local_font))
    for name in ('NanumGothic', 'Noto Sans CJK KR', 'DejaVu Sans'):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams['font.family'] = name
            break
        except ValueError:
            pass
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def canvas(ax, rgb, roi, title):
    ax.imshow(rgb, interpolation='nearest')
    ax.set_xlim(roi[0] - .5, roi[2] - .5)
    ax.set_ylim(roi[3] - .5, roi[1] - .5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('원영상 x (px)', fontsize=8)
    ax.set_ylabel('원영상 y (px)', fontsize=8)
    ax.tick_params(labelsize=7)


def overlay_mask(ax, mask, color, alpha=.35):
    from matplotlib.colors import to_rgba
    layer = np.zeros((*mask.shape, 4), dtype=float)
    layer[mask] = to_rgba(color, alpha)
    ax.imshow(layer, interpolation='nearest')


def mask_canvas(ax, mask, roi, title):
    ax.imshow(mask, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xlim(roi[0] - .5, roi[2] - .5)
    ax.set_ylim(roi[3] - .5, roi[1] - .5)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)


def save_figure(plt, fig, path):
    fig.savefig(path, dpi=115, facecolor='white', pil_kwargs={'quality': 91})
    plt.close(fig)


def display_roi(frame, prompt, target_mask):
    """Display-only union; never replaces the original observation polygon."""
    points = [np.asarray(o['polygon_xy'], float) for stage in frame['outlines'].values()
              for o in stage.values() if o.get('supported')]
    # Include the actual original target array, rather than a model/GT crop.
    yy, xx = np.nonzero(target_mask)
    if len(xx):
        points.append(np.array([[xx.min(), yy.min()], [xx.max(), yy.max()]], float))
    if prompt.get('box_xyxy') is not None:
        x0, y0, x1, y1 = prompt['box_xyxy']
        points.append(np.array([[x0, y0], [x1, y1]], float))
    require(bool(points), 'No original outline/mask support for display ROI')
    all_points = np.concatenate(points)
    h, w = frame['image_hw']
    return [max(0, int(np.floor(all_points[:, 0].min())) - 24),
            max(0, int(np.floor(all_points[:, 1].min())) - 24),
            min(w, int(np.ceil(all_points[:, 0].max())) + 25),
            min(h, int(np.ceil(all_points[:, 1].max())) + 25)]


def project(candidate, K):
    xyz, R, t, K = map(lambda v: np.asarray(v, float),
                       (candidate['native_xyz'], candidate['R'], candidate['t'], K))
    camera = xyz @ R.T + t
    projected_h = camera @ K.T
    require(np.all(camera[:, 2] > 0), 'Selected cuboid has nonpositive depth')
    q = projected_h[:, :2] / projected_h[:, 2:3]
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = R, t
    other = (K @ transform[:3] @ np.c_[xyz, np.ones(len(xyz))].T).T
    independent = other[:, :2] / other[:, 2:3]
    gap = float(np.max(np.abs(q - independent)))
    require(gap <= 1e-9, 'Fixed projection arithmetic mismatch')
    edges = [(i, j) for i in range(len(xyz)) for j in range(i + 1, len(xyz))
             if np.count_nonzero(~np.isclose(xyz[i], xyz[j], rtol=0, atol=1e-12)) == 1]
    require(len(xyz) == 8 and len(edges) == 12, 'Expected native registered cuboid')
    return q, edges, gap


def draw_parents(ax, outline, parents, labels=False, samples=False):
    poly = np.asarray(outline['polygon_xy'], float)
    closed = np.vstack([poly, poly[0]])
    ax.plot(closed[:, 0], closed[:, 1], ':', color='#555555', lw=.85, alpha=.8)
    used_edges = set()
    for parent in parents:
        used_edges.add(parent['original_hull_edge_index'])
        full = np.asarray(parent['full_parent_xy'], float)
        finite = np.asarray(parent['xy'], float)
        color = '#d81b60' if parent['censor'] else '#00a86b'
        ax.plot(finite[:, 0], finite[:, 1], color=color, lw=2.0,
                linestyle='--' if parent['censor'] else '-')
        ax.scatter(full[:, 0], full[:, 1], s=12, facecolors='none', edgecolors=color, zorder=5)
        for source, clipped in zip(full, finite):
            if not np.array_equal(source, clipped):
                ax.plot([source[0], clipped[0]], [source[1], clipped[1]], color='#ff9500', lw=2)
        if samples:
            xy = np.asarray(outline['observed_boundary_samples_xy'])[parent['sample_indices']]
            ax.scatter(xy[:, 0], xy[:, 1], s=2.5, color=color, zorder=4)
        if labels:
            mid = finite.mean(axis=0)
            ax.annotate(f'E{parent["segment_id"]}/H{parent["original_hull_edge_index"]}', mid,
                        fontsize=6, color=color, clip_on=True,
                        bbox=dict(facecolor='white', alpha=.7, edgecolor='none', pad=.4))
    for i, (a, b) in enumerate(zip(poly, np.roll(poly, -1, axis=0))):
        if i not in used_edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], color='#ff9500', lw=2)


def metric_text(metric):
    return f'위치 {float(metric["translation_cm"]):.2f}cm / 회전 {float(metric["rotation_deg"]):.2f}°'


def cell(value):
    return html.escape(str(value))


def _render(private_root, run, evaluation, source_root, output, guard):
    started, cpu = time.monotonic(), time.process_time()
    root, run, evaluation, source, out = map(Path, (private_root, run, evaluation, source_root, output))
    require(not out.exists(), 'Preserve previous gallery; choose a new output folder')
    package_path = root / 'neutral_v1/NEUTRAL_PACKAGE.json'
    package = read(package_path)
    seal = read(run / 'PREDICTION_SEAL.json')
    bound(package_path, seal['neutral_package_sha256'])
    for name, digest in seal['artifacts'].items():
        bound(run / name, digest)
    require(seal['frames'] == 6 and seal['candidates'] == 351 and seal['prediction_rows'] == 18,
            'Original six/full bank/three selections required')
    require(seal['reference_free_selection'] and seal['before_new_reference_evaluation'],
            'Prediction/evaluation boundary must be preserved')
    eval_receipt = read(evaluation / 'EVALUATION_RECEIPT.json')
    require(eval_receipt['prediction_seal_sha256'] == sha(run / 'PREDICTION_SEAL.json') and
            eval_receipt['predictions_sealed_before_reference_open'], 'Wrong post-seal evaluation')
    for name, digest in eval_receipt['artifacts'].items():
        bound(evaluation / name, digest)
    for record in (package['frames'], package['candidates'], package['RGB']):
        bound(package_path.parent / record['path'], record['sha256'])
    frames = rows(package_path.parent / package['frames']['path'])
    candidates = rows(package_path.parent / package['candidates']['path'])
    by_candidate = {(c['id'], c['candidate_id']): c for c in candidates}
    require(len(candidates) == len(by_candidate) == 351, 'Ambiguous original candidate identity')
    prompts = {p['image_id']: p for p in rows(root / 'PROMPT_MANIFEST.jsonl')}
    supplier = {p['image_id']: p for p in rows(root / 'supplier_v3/NON_TARGET_INPUT_MANIFEST.jsonl')}
    frozen = {p['id']: p for p in rows(root / 'frozen_inputs_v1/INPUTS_FROZEN.jsonl')}
    frozen_seal = read(root / 'frozen_inputs_v1/INPUT_SEAL.json')
    bound(root / 'frozen_inputs_v1/INPUTS_FROZEN.jsonl', seal['input_manifest_sha256'])
    predictions = {(p['id'], p['variant']): p for p in rows(run / 'PREDICTIONS.jsonl.gz')}
    metrics = {(p['id'], p['variant']): p for p in csvrows(evaluation / 'METRICS.csv')}
    costs = {(c['id'], c['candidate_id']): c for c in rows(run / 'CANDIDATE_COSTS.jsonl.gz')}
    parents = rows(run / 'PARENT_SUPPORT.jsonl')
    require(len(predictions) == len(metrics) == 18, 'All eighteen predictions/metrics required')
    export = read(root / 'neutral_v1/EXPORT_RECEIPT.json')
    permissions = permission(source, export)
    mask_assets = {Path(a['path']).stem: a for a in export['original_assets'] if a['kind'] == 'original_target_mask'}
    require([f['id'] for f in frames] == package['ids'], 'Fixed six ID/order changed')
    plt = plot_library()
    out.mkdir(parents=True, exist_ok=False)
    articles, initial_articles, proofs = [], [], []
    for frame in frames:
        ident, ordinal = frame['id'], frame['ordinal']
        folder = out / f'case_{ordinal:02d}'
        folder.mkdir()
        prompt, supplied, inp = prompts[ident], supplier[ident], frozen[ident]
        rgb_path = source / frame['image']
        bound(rgb_path, frame['image_sha256'])
        rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
        require(list(rgb.shape[:2]) == frame['image_hw'], 'Original RGB canvas differs')
        target_asset = mask_assets[ident.replace(':', '__')]
        bound(target_asset['path'], target_asset['sha256'])
        with np.load(target_asset['path'], allow_pickle=False) as archive:
            target = archive['pallet_mask_clean'].copy()
            require(np.array_equal(target, archive['pallet_mask']), 'Actual clean-mask alias differs')
        require(set(np.unique(target)).issubset({0, 1}), 'Original target clean mask must remain binary')
        target = target.astype(bool)
        roi = display_roi(frame, prompt, target)
        x0, y0, x1, y1 = roi
        Image.fromarray(rgb).save(folder / 'scene_RGB_unmarked.jpg', quality=94)
        Image.fromarray(rgb[y0:y1, x0:x1]).save(folder / 'crop_RGB_unmarked.png')
        require(np.array_equal(np.asarray(Image.open(folder / 'crop_RGB_unmarked.png')), rgb[y0:y1, x0:x1]),
                'Displayed PNG crop differs from original source pixels')
        preview = root / 'input_preview' / f'case_{ordinal:02d}.jpg'
        shutil.copyfile(preview, folder / 'input_preview_before_selection.jpg')
        alternatives, quality, selected = None, None, None
        if supplied.get('all_masks_path'):
            bound(supplied['all_masks_path'], supplied['all_masks_file_sha256'])
            with np.load(supplied['all_masks_path'], allow_pickle=False) as archive:
                alternatives, quality = archive['masks'].copy(), archive['quality'].copy()
                selected = int(archive['selected_index'])
            require(alternatives.dtype == bool and alternatives.shape == (3, *rgb.shape[:2]), 'Saved alternative mask canvas/dtype differs')
            require(selected == supplied['selected_index'] == int(np.argmax(quality)), 'Frozen maximum-quality/first-tie selection differs')
            require(np.array_equal(quality, np.asarray(supplied['quality'], quality.dtype)), 'Raw model quality differs')
            require([pixel_sha(a.astype(np.uint8)) for a in alternatives] == supplied['alternative_pixel_sha256'],
                    'Original three alternative mask bytes differ')
        foreign = None
        if inp['admissible_for_censoring']:
            bound(inp['mask_path'], inp['mask_sha256'])
            foreign = np.load(inp['mask_path'], allow_pickle=False)
            require(foreign.dtype in (np.dtype(bool), np.dtype(np.uint8)) and
                    set(np.unique(foreign)).issubset({0, 1}) and foreign.shape == rgb.shape[:2] and
                    pixel_sha(foreign.astype(np.uint8)) == inp['pixel_bytes_sha256'], 'Frozen foreign input differs')
            # freeze_inputs stores the approved binary bytes as uint8 0/1;
            # the bool view below is display indexing only, not a new mask.
            foreign = foreign.astype(bool)
            require(alternatives is not None and np.array_equal(foreign, alternatives[selected]), 'Input changed after supplier review')
        else:
            require(inp['status'] == 'NO_ADMISSIBLE_NON_TARGET_EVIDENCE', 'Unknown input must remain explicitly unknown')
        fig, axes = plt.subplots(2, 3, figsize=(17, 8), layout='constrained')
        fig.suptitle(f'{ordinal}. {ident} · 입력 보조/모델 검토 · 자동·독립·GT 아님', fontsize=12)
        canvas(axes[0, 0], rgb, roi, '원 RGB + 고정 ASSISTED box/양성점')
        if prompt.get('box_xyxy') is not None:
            from matplotlib.patches import Rectangle
            bx0, by0, bx1, by1 = prompt['box_xyxy']
            axes[0, 0].add_patch(Rectangle((bx0, by0), bx1-bx0, by1-by0, fill=False, color='#ff8c00', lw=1.5))
            point = prompt['positive_point_xy']
            axes[0, 0].scatter(*point, color='#ffff00', marker='*', s=90, edgecolors='black')
        mask_canvas(axes[0, 1], target, roi, '원 target clean mask · 변경 없음')
        canvas(axes[0, 2], rgb, roi, '새 비대상 선택 mask · GT/깊이순서 인증 아님' if foreign is not None else '입력 미확정 · 검증된 배경/가림 없음 아님')
        if foreign is not None:
            overlay_mask(axes[0, 2], foreign, '#ff00c8', .5)
        for k in range(3):
            if alternatives is None:
                canvas(axes[1, k], rgb, roi, f'대안{k}: 미실행 · 명확한 cone 입력 없음')
            else:
                canvas(axes[1, k], rgb, roi, f'원 대안{k} · quality={float(quality[k]):.6f}' + (' · 동결 선택' if k == selected else ''))
                overlay_mask(axes[1, k], alternatives[k], '#ff00c8', .5)
        save_figure(plt, fig, folder / 'prompt_target_foreign_alternatives.jpg')
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout='constrained')
        context_proof, parent_tables = [], []
        for ax, (stage, family) in zip(axes.flat, ((s, f) for s in STAGES for f in FAMILIES)):
            outline = frame['outlines'][stage][family]
            own = [p for p in parents if (p['id'], p['stage'], p['family']) == (ident, stage, family)]
            require([p['segment_id'] for p in own] == list(range(len(own))), 'Parent source order differs')
            require(np.array_equal([p['xy'] for p in own], outline['observed_boundary_segments_xy']), 'Original finite parent segments differ')
            count = sum(p['censor'] for p in own)
            canvas(ax, rgb, roi, f'{stage} / {family}\n유한 부모 {len(own)}개 · 결측 {count}개')
            if foreign is not None:
                overlay_mask(ax, foreign, '#c56aff', .2)
            draw_parents(ax, outline, own, labels=True, samples=True)
            context_proof.append(dict(stage=stage, family=family, original_polygon_xy=outline['polygon_xy'],
                original_finite_segments_xy=outline['observed_boundary_segments_xy'],
                original_observed_sample_xy=outline['observed_boundary_samples_xy'], parents=own,
                censored_parent_count=count, censored_observed_samples=sum(len(p['sample_indices']) for p in own if p['censor']),
                new_hull_or_segments_created=False))
            for p in own:
                note = '부모 전체 관측 제외; 실제 팔레트 지지가 함께 사라질 수 있음' if p['censor'] else '원 관측 유지; target-only 물리 인증 아님'
                parent_tables.append('<tr>' + ''.join(f'<td>{cell(v)}</td>' for v in (stage, family,
                    f'E{p["segment_id"]}/H{p["original_hull_edge_index"]}', p['full_parent_xy'], p['xy'],
                    p['endpoint_pixels_xy'], p['endpoint_mask_membership'], len(p['sample_indices']), p['support_status'], note)) + '</tr>')
        save_figure(plt, fig, folder / 'original_preclip_and_finite_parents.jpg')
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.7), layout='constrained')
        model_proofs, method_cards = [], []
        for ax, variant, label, color in zip(axes, VARIANTS, LABELS, COLORS):
            pred, metric = predictions[(ident, variant)], metrics[(ident, variant)]
            candidate = by_candidate[(ident, pred['candidate_id'])]
            require(candidate['family'] == pred['selected_family'] == metric['family'] and metric['candidate_id'] == pred['candidate_id'],
                    'Selected family/candidate/metric binding differs')
            q, edges, gap = project(candidate, frame['K'])
            require(np.allclose(pred['pose']['R_cf'], candidate['R'], rtol=0, atol=1e-9) and
                    np.allclose(pred['pose']['centroid'], candidate['t'], rtol=0, atol=1e-9), 'Selected pose differs from native candidate')
            require(np.allclose(pred['pose']['R_physical'], np.asarray(candidate['R']) @ np.asarray(candidate['physical_registration_Q']), rtol=0, atol=1e-9),
                    'Physical basis contract differs')
            own = [p for p in parents if (p['id'], p['stage'], p['family']) == (ident, candidate['stage'], candidate['family'])]
            canvas(ax, rgb, roi, label + '\n' + metric_text(metric))
            draw_parents(ax, frame['outlines'][candidate['stage']][candidate['family']], own)
            for i, j in edges:
                ax.plot(q[[i, j], 0], q[[i, j], 1], color=color, lw=1.8)
            for i, point in enumerate(q):
                ax.annotate(str(i), point, color=color, fontsize=7, clip_on=True)
            current_cost = costs[(ident, pred['candidate_id'])]
            directions = []
            for direction in current_cost['directions']:
                original = next(o for o in candidate['dense']['per_observation'] if o['direction'] == direction['direction'])
                require(len(direction['retained_sample_flags']) == len(original['sample_xy']) == direction['original_count'], 'Own directional sample count differs')
                directions.append(dict(direction, original_sample_xy=original['sample_xy'],
                    original_normalizer=original['normalizer'], original_normalizer_type=original['normalizer_type'],
                    original_weighted_sample_contributions=original['weighted_sample_contributions'],
                    model_indices_not_paired_with_other_candidate=True))
            model_proofs.append(dict(variant=variant, candidate_id=pred['candidate_id'], stage=candidate['stage'], family=candidate['family'],
                native_xyz=candidate['native_xyz'], R=candidate['R'], t=candidate['t'],
                physical_registration_Q=candidate['physical_registration_Q'], projected_native_corners_xy=q.tolist(),
                cuboid_edges=edges, independent_homogeneous_projection_max_abs_px=gap, evaluator_metric=metric,
                accepted=pred['accepted'], fallback=pred['fallback'], original_constraint_pass=pred['original_constraint_pass'],
                recomputed_constraint_pass=pred['recomputed_constraint_pass'], minimum_constraint_slack=pred['minimum_constraint_slack'],
                selected_G13_score=current_cost['scores'][variant], selected_candidate_scores=current_cost['scores'],
                original_components=current_cost['original_weighted_components'], own_directional_samples=directions))
            base = predictions[(ident, 'BASELINE_E')]
            mass_text = '; '.join(d['direction'] + ': ' + str(d['retained_mass_ratio']) for d in current_cost['directions'])
            method_cards.append(f'<section><h4>{cell(label)}</h4><p>{cell(metric_text(metric))} · 참조상 성공={cell(metric["success_le5"])}<br>'
                f'후보 {cell(pred["candidate_id"])}<br>family {cell(pred["selected_family"])} · 기존과 후보 변경={pred["candidate_id"] != base["candidate_id"]}<br>'
                f'accepted={pred["accepted"]}, fallback={pred["fallback"]}<br>원/보존 제약={pred["original_constraint_pass"]}/{pred["recomputed_constraint_pass"]}, '
                f'min slack={cell(pred["minimum_constraint_slack"])}<br>이 후보의 {candidate["stage"]} 점수={cell(current_cost["scores"][variant])}<br>'
                f'이 후보에 신규 규칙을 적용한 유지 mass 비율: {cell(mass_text)}</p></section>')
        fig.suptitle(f'{ordinal}. {ident} · 동일 ROI/축척 · 모델 투영점은 영상 관측점이 아님', fontsize=12)
        save_figure(plt, fig, folder / 'three_final_paths.jpg')
        # Saved model-direction samples remain separate per selected candidate.
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), layout='constrained')
        for ax, proof, label in zip(axes, model_proofs, LABELS):
            canvas(ax, rgb, roi, label + '\nmodel→observed: 후보별 원 표본/원 분모')
            model = next(d for d in proof['own_directional_samples'] if d['direction'] == 'model_to_observed')
            xy = np.asarray(model['original_sample_xy'])
            keep = np.asarray(model['retained_sample_flags'], bool)
            if proof['variant'] == 'BASELINE_E':
                ax.scatter(xy[:, 0], xy[:, 1], s=12, color=COLORS[0], label='원 표본 전부 사용')
            elif proof['variant'] == 'MASS_MATCHED_CONTROL':
                ax.scatter(xy[:, 0], xy[:, 1], s=12, color=COLORS[2], label='원 표본 전부 사용/일괄 약화')
            else:
                ax.scatter(xy[keep, 0], xy[keep, 1], s=12, color='#00a86b', label='유지')
                ax.scatter(xy[~keep, 0], xy[~keep, 1], s=17, color='#d81b60', marker='x', label='결측')
            ax.legend(fontsize=8)
        save_figure(plt, fig, folder / 'own_model_direction_samples.jpg')
        preview_digest = sha(preview)
        require(preview_digest == sha(folder / 'input_preview_before_selection.jpg'), 'Original input-only preview changed')
        initial_articles.append(f'<article id="case{ordinal}"><h2>{ordinal}. {cell(ident)}</h2><p>{cell(inp["status"])} · '
            'ASSISTED/모델 검토, GT 아님. 입력 없음은 검증된 배경/가림 없음이 아닙니다.</p>'
            f'<img src="case_{ordinal:02d}/scene_RGB_unmarked.jpg"><img src="case_{ordinal:02d}/crop_RGB_unmarked.png">'
            f'<h3>원 실행에서 후보/비용 전 보존한 입력 검토 화면</h3><img src="case_{ordinal:02d}/input_preview_before_selection.jpg">'
            f'<h3>동결 prompt·원target·새foreign·원3대안</h3><img src="case_{ordinal:02d}/prompt_target_foreign_alternatives.jpg"></article>')
        articles.append(f'<article id="case{ordinal}"><h2>{ordinal}. {cell(ident)} · {cell(inp["status"])}</h2>'
            f'<p><a href="initial_index.html#case{ordinal}">입력 전용 보기</a> · 원 RGB SHA256 {cell(frame["image_sha256"])} · 표시 ROI {cell(roi)}. '
            '원좌표, 추가 offset/resize/letterbox 없음.</p>'
            f'<details><summary>표시선 없는 전체 RGB와 정확한 원 crop</summary><img src="case_{ordinal:02d}/scene_RGB_unmarked.jpg">'
            f'<img src="case_{ordinal:02d}/crop_RGB_unmarked.png"></details>'
            f'<h3>1. 원입력과 동결 mask</h3><img src="case_{ordinal:02d}/prompt_target_foreign_alternatives.jpg">'
            '<p>quality는 모델의 원 예측값이며 정답·대상 정체·깊이 순서의 인증이 아닙니다. 선택 대안은 결과에 따라 바꾸지 않았습니다.</p>'
            f'<h3>2. 원 부모와 결측 처리</h3><img src="case_{ordinal:02d}/original_preclip_and_finite_parents.jpg">'
            '<p>회색 점선=원 preclip hull. 초록=유지한 finite 부모/표본. 자홍=비대상 끝점 지지로 결측 처리한 원 finite 부모/표본. '
            '주황=원 canvas margin에서 제외된 부분이며 비대상 증거가 아닙니다. 이 관측 유효성 표시는 신규 규칙의 근거이며 baseline은 원 표본을 모두 사용합니다. 새 polygon/연결선은 없습니다. '
            '결측 부모의 모든 픽셀이 cone이라는 뜻이 아니며 실제 팔레트 지지가 함께 사라질 수 있습니다.</p>'
            f'<details><summary>모든 stage/family의 원 부모 장부 {len(parent_tables)}행 · 정보 손실 포함</summary><table><tr>'
            '<th>stage</th><th>family</th><th>부모</th><th>preclip 끝점</th><th>finite 부분</th><th>조회 pixel</th><th>mask membership</th><th>원 observed 수</th><th>상태</th><th>한계</th></tr>'
            + ''.join(parent_tables) + '</table></details>'
            f'<h3>3. 봉인된 세 최종 선택</h3><img src="case_{ordinal:02d}/three_final_paths.jpg"><div class="cards">'
            + ''.join(method_cards) + '</div><p>각 family의 자체 native XYZ/Q를 사용합니다. G10과 G13의 목적함수 값을 서로 비교하지 않습니다. '
            '여기 오차는 사후 평가기의 기존 기하 참조 기준이며 독립 물리 정답이 아닙니다.</p>'
            f'<h3>4. 후보마다 다른 model 방향의 실제 유지/결측</h3><img src="case_{ordinal:02d}/own_model_direction_samples.jpg">'
            '<p>각 후보의 원 sample·호 길이·분모·비용을 개별 보존했습니다. baseline은 모든 표본, support는 유지 표본, mass control은 모든 표본의 잔차를 방향별 mass 비율로 약화합니다. '
            '서로 다른 후보의 model 표본 index끼리 차를 구하지 않습니다.</p></article>')
        proofs.append(dict(id=ident, ordinal=ordinal, image_relative_path=frame['image'], image_sha256=frame['image_sha256'],
            original_canvas_hw=frame['image_hw'], K=frame['K'], target_mask_sha256=target_asset['sha256'], actual_target_array='pallet_mask_clean',
            target_binary_pixel_sha256=pixel_sha(target.astype(np.uint8)), display_crop_original_xyxy=roi,
            crop_transform_original_to_display=[[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], display_resize=False,
            crop_PNG_source_pixels_exact=True, unmarked_full_scene_is_recompressed_JPEG=True,
            original_preselection_preview_sha256=preview_digest,
            input_status=inp['status'], admissible_for_censoring=inp['admissible_for_censoring'],
            no_input_is_verified_background=False, frozen_input_pixel_sha256=inp.get('pixel_bytes_sha256'),
            prompt={k: prompt.get(k) for k in ('category', 'box_xyxy', 'positive_point_xy', 'author_kind', 'prompt_mode', 'frozen_at_utc', 'explanation')},
            supplier=dict(selected_index=selected, original_quality=None if quality is None else quality.tolist(),
                alternatives_pixel_sha256=supplied.get('alternative_pixel_sha256'), quality_is_GT=False,
                original_manifest_status=supplied['status'], selected_after_new_metrics=False,
                depth_order_asserted=False), contexts=context_proof, selected_models=model_proofs))
    style = '<style>body{font-family:sans-serif;max-width:1900px;margin:20px auto;padding:16px;background:#eef2f5;line-height:1.6}header,article{background:white;padding:20px;margin:18px 0;border-radius:8px}img{max-width:100%;height:auto}table{font-size:11px;border-collapse:collapse;width:100%}td,th{padding:5px;border-bottom:1px solid #ddd;overflow-wrap:anywhere}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.cards section{padding:12px;background:#edf1f5;overflow-wrap:anywhere}details{margin:12px 0}@media(max-width:900px){.cards{display:block}}</style>'
    common = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">' + style
    (out / 'initial_index.html').write_text(common + '<title>동결 입력 전용 보기</title><header><h1>입력 전용: 여섯 사례</h1><p>후보/참조/오차/비용 미표시. '
        '원 실행에서 실제 선택 전에 보존한 입력 화면과 동일 동결 입력의 재표시입니다. 기존에 알던 사례이며 독립 blind가 아닙니다. '
        '이 HTML의 생성시각을 최초 입력 검토시각으로 주장하지 않습니다.</p></header>' + ''.join(initial_articles) + '</html>')
    (out / 'index.html').write_text(common + '<title>비대상 외곽선 기능 파일럿</title><header><h1>비대상 외곽선 처리: ASSISTED6 기능 시험</h1>'
        '<p>기존 생성기의 고정 후보를 유지한 세 선택 경로입니다. 기존 사례를 알고 만든 모델 보조 입력이며 자동 전체 성능·독립 검증·oracle mask가 아닙니다. '
        '신규 GT/모델 forward/PnP/cost/선택은 이 렌더러에서 실행하지 않습니다.</p>'
        '<p>성공수의 손익과 연속 오차 변화는 별개입니다. 5cm/5° 성공이 바뀌지 않아도 위치·회전은 나빠질 수 있습니다. '
        '정보를 많이 잃은 사례도 모두 표시하며 입력 없는 성공 반례는 무입력 항등성 확인입니다. 실제 censoring의 무손상 안전성을 입증하지 않습니다.</p>'
        '<p><a href="initial_index.html">후보/비용 없는 입력 전용 화면</a> · <a href="../REPORT_KO.md">실제 결과와 한계</a> · '
        '<a href="GALLERY_PROOF.json">원좌표/표본/선택/출처 장부</a></p></header>' + ''.join(articles) + '</html>')
    proof = dict(schema='NON_TARGET_OUTLINE_PILOT_POSTSEAL_GALLERY_V1', cases=proofs,
        prediction_seal_sha256=sha(run / 'PREDICTION_SEAL.json'), evaluation_receipt_sha256=sha(evaluation / 'EVALUATION_RECEIPT.json'),
        neutral_package_sha256=sha(package_path), frozen_input_manifest_sha256=sha(root / 'frozen_inputs_v1/INPUTS_FROZEN.jsonl'),
        frozen_input_seal_sha256=sha(root / 'frozen_inputs_v1/INPUT_SEAL.json'),
        prompt_manifest_sha256=sha(root / 'PROMPT_MANIFEST.jsonl'), supplier_manifest_sha256=sha(root / 'supplier_v3/NON_TARGET_INPUT_MANIFEST.jsonl'),
        previous_mask_review_sha256=sha(root / 'MASK_REVIEW.json'), public_permission=permissions,
        no_reference_files_opened_by_renderer=True, new_NN_PnP_cost_rank_calls=0)
    write(out / 'GALLERY_PROOF.json', proof)
    write(out / 'PUBLICATION_PERMISSION.json', permissions)
    decoded = 0
    for path in out.rglob('*'):
        if path.suffix in ('.jpg', '.png'):
            with Image.open(path) as image:
                image.load()
            decoded += 1
    receipt = dict(schema='NON_TARGET_OUTLINE_PILOT_GALLERY_ACTUAL_COST_V1', passed=True, cases=len(proofs),
        final_selected_panels=sum(len(c['selected_models']) for c in proofs), original_candidate_bank_count=351,
        all_parent_context_rows=sum(len(ctx['parents']) for c in proofs for ctx in c['contexts']),
        original_input_only_previews_preserved=6, original_RGB_exact_PNG_crop_checks=6,
        original_full_scene_JPEG_displays=6, decoded_images=decoded,
        admitted_inputs=sum(c['admissible_for_censoring'] for c in proofs), unknown_inputs=sum(not c['admissible_for_censoring'] for c in proofs),
        new_inference_calls=0, new_pose_fit_calls=0, new_cost_evaluations=0, new_selections=0,
        parent_index_or_model_sample_reassignment_calls=0, runtime_guard=guard.receipt(),
        independent_projection_max_abs_px=max(p['independent_homogeneous_projection_max_abs_px'] for c in proofs for p in c['selected_models']),
        elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
        assets={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    write(out / 'GALLERY_RECEIPT.json', receipt)
    return receipt


def render(private_root, run, evaluation, source_root, output):
    from scripts.research.pallet_outline_physical_support_audit_v1.audit_guard import SupportAuditOnly
    with SupportAuditOnly() as guard:
        return _render(private_root, run, evaluation, source_root, output, guard)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('private-root', 'run', 'evaluation', 'source-root', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    result = render(args.private_root, args.run, args.evaluation, args.source_root, args.output)
    print(json.dumps({key: result[key] for key in ('passed', 'cases', 'final_selected_panels',
        'all_parent_context_rows', 'elapsed_seconds', 'process_cpu_seconds')}, ensure_ascii=False))
