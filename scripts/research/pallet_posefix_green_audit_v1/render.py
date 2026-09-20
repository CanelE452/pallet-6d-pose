"""CPU-only GREEN150 diagnostic overlays from immutable saved predictions.

RGB pixels are unchanged; all graphics are conventional coordinate overlays.
Only verified manual-click GT is displayed. Correspondences use each model's
single, approved whole-object C4 branch, never per-corner nearest-neighbour GT.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/pallet_posefix_green_audit_v1'
REPLAY = ROOT / 'data/pallet/results/pallet_posefix_replay_v1'
REAL = ROOT / 'data/pallet/results/pallet_posefix_large_error_v1'
SNAPSHOT = ROOT / '_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json'
CONTRACT = ROOT / '_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json'
ARMS = ('R0', 'A_N2', 'REAL_ONLY', 'REPLAY')
LABELS = dict(R0='R0', A_N2='Current DIM-only', REAL_ONLY='Real-only PoseFix', REPLAY='Replay PoseFix')
CASES = (
    ('worst', 'capture_20260902__009323', 1,
     'Largest matched GREEN150 frame-mean worsening: replay minus current DIM-only'),
    ('truncated', 'capture_20260902_kimjihoon__003249', 2,
     'Additional visibly truncated failure example, chosen diagnostically (not a random representative)'),
    ('improved', 'capture_20260902_kimjihoon__007309', 4,
     'Largest matched GREEN150 frame-mean improvement: replay minus current DIM-only'),
    ('nontruncated', 'capture_20260902_kimjihoon__006349', 0,
     'Non-truncated failure example, chosen diagnostically to avoid attributing all damage to truncation'),
)
BG = '#101923'
GT_COLOR = '#ff65ee'
PRED_COLOR = '#ffe45b'
INPUT_COLOR = '#56ddff'


def read(path):
    return json.loads(Path(path).read_text())


def bound(path):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT)
    data = path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def verify(binding):
    actual = bound(ROOT / binding['path'])
    assert actual['sha256'] == binding['sha256'], binding['path']
    if 'bytes' in binding:
        assert actual['bytes'] == binding['bytes'], binding['path']


def top(prediction):
    index = prediction['selected_index']
    assert index is not None
    return prediction['candidates'][index]


def manual_gt(annotation):
    entries = annotation['objects'][0]['keypoint_annotations']
    assert len(entries) == 9
    gt = np.array([e['xy'] if e.get('xy') is not None else [np.nan, np.nan] for e in entries])
    mask = np.array([e.get('visibility', 0) != 0 and e.get('source') == 'manual_click'
                     and e.get('xy') is not None for e in entries])
    mask &= np.isfinite(gt).all(1) & ~(gt == -1).all(1)
    mask[8] = False
    return gt, mask


def verify_metric(points, gt, manual, permutations, score):
    """Independent manual-only whole-C4 recomputation, same stored metric."""
    assert score['matched'] and score['detected'] and score['evaluable']
    means = []
    for perm in permutations:
        mask = manual[perm][:8]
        errors = np.linalg.norm(points[:8] - gt[perm][:8], axis=1)
        means.append(float(errors[mask].mean()))
    branch = int(np.argmin(means))
    assert branch == score['branch'], (branch, score['branch'])
    assert abs(means[branch] - score['frame_mean_px']) < 1e-6
    perm = permutations[branch]
    for canonical in np.flatnonzero(manual):
        native = int(np.flatnonzero(perm == canonical)[0])
        error = float(np.linalg.norm(points[native] - gt[canonical]))
        assert abs(error - score['canonical_errors'][canonical]) < 1e-6


def clipped_limits(points, hw, margin=25, minimum=100):
    h, w = hw
    points = np.asarray(points, float)
    points = points[np.isfinite(points).all(1)]
    low, high = points.min(0) - margin, points.max(0) + margin
    middle = (low + high) / 2
    half = np.maximum((high - low) / 2, minimum / 2)
    low, high = middle - half, middle + half
    low = np.maximum(low, [0, 0])
    high = np.minimum(high, [w - 1, h - 1])
    for axis, maximum in enumerate((w - 1, h - 1)):
        if high[axis] - low[axis] < minimum:
            if low[axis] == 0:
                high[axis] = min(minimum, maximum)
            else:
                low[axis] = max(0, high[axis] - minimum)
    return [float(low[0]), float(high[0]), float(low[1]), float(high[1])]


def inside(point, limits):
    x0, x1, y0, y1 = limits
    return bool(x0 <= point[0] <= x1 and y0 <= point[1] <= y1)


def draw(ax, image, points, initial, gt, manual, edges, limits, focus=None):
    ax.imshow(image, interpolation='nearest')
    for a, b in edges:
        if np.isfinite(points[[a, b]]).all():
            ax.plot(points[[a, b], 0], points[[a, b], 1], color=PRED_COLOR,
                    linewidth=1.25, alpha=.92, zorder=3)
    for native in range(8):
        if not np.isfinite(points[native]).all():
            continue
        ax.scatter(*points[native], s=27, c=PRED_COLOR, edgecolors='#202020',
                   linewidths=.5, zorder=6)
        if np.linalg.norm(points[native] - initial[native]) > .15:
            arrow = ax.annotate('', xy=points[native], xytext=initial[native],
                                arrowprops=dict(arrowstyle='->', color='white', lw=1.2),
                                annotation_clip=True, zorder=5)
            arrow.arrow_patch.set_clip_on(True)
            arrow.arrow_patch.set_clip_path(ax.patch)
    for canonical in np.flatnonzero(manual):
        if inside(gt[canonical], limits):
            ax.scatter(*gt[canonical], marker='x', s=80, c=GT_COLOR,
                       linewidths=2.2, zorder=8)
            ax.annotate(f'G{canonical}', xy=gt[canonical], xytext=(6, -13),
                        textcoords='offset points', color=GT_COLOR, fontsize=9,
                        bbox=dict(facecolor=BG, alpha=.8, edgecolor='none', pad=1), zorder=9)
    if focus is not None:
        native, canonical = focus
        if inside(initial[native], limits):
            ax.scatter(*initial[native], s=90, facecolors='none', edgecolors=INPUT_COLOR,
                       linewidths=1.5, zorder=7)
        if inside(points[native], limits):
            ax.scatter(*points[native], s=110, facecolors='none', edgecolors=PRED_COLOR,
                       linewidths=1.5, zorder=7)
            ax.annotate(f'P{native}', xy=points[native], xytext=(6, 8),
                        textcoords='offset points', color=PRED_COLOR, fontsize=9,
                        bbox=dict(facecolor=BG, alpha=.8, edgecolor='none', pad=1), zorder=9)
        else:
            ax.text(.5, .05, f'P{native} outside shared zoom\n'
                    f'({points[native,0]:.1f}, {points[native,1]:.1f}) px',
                    transform=ax.transAxes, color='#ffb2a6', ha='center', va='bottom',
                    fontsize=10, bbox=dict(facecolor=BG, alpha=.9, edgecolor='none'))
    x0, x1, y0, y1 = limits
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis('off')


def main():
    pred_path = REPLAY / 'PREDICTIONS.json'
    metrics_path = REPLAY / 'PER_FRAME_METRICS.json'
    old_pred_path = REAL / 'FINETUNED_PREDICTIONS.json'
    old_metrics_path = REAL / 'FINETUNED_PER_FRAME_METRICS.json'
    result_paths = [ROOT / '_docs/experiments/pallet_posefix_replay_v1/REAL_RESULTS.json',
                    ROOT / '_docs/experiments/pallet_posefix_large_error_v1/FINETUNED_RESULTS.json']
    for path in result_paths:
        result = read(path)
        assert result['complete'] and result['evaluated_images'] == 222
        for artifact in result['artifacts']:
            verify(artifact)
    predictions = {r['id']: r for r in read(pred_path)['GREEN150']}
    previous = {r['id']: r for r in read(old_pred_path)['GREEN150']}
    metrics = {a: {r['id']: r for r in rows}
               for a, rows in read(metrics_path)['GREEN150_MANUAL'].items()}
    old_metrics = {r['id']: r for r in read(old_metrics_path)['GREEN150_MANUAL']['POSEFIX_RAW']}
    records = {r['id']: r for r in read(SNAPSHOT)['records']}
    contract = read(CONTRACT)
    edges = contract['edges']
    square = next(o for o in contract['objects'] if o['object_type'] == 'plastic_standard_110x110x15')
    permutations = np.array(square['permutations'])
    eligible = [(metrics['POSEFIX_RAW'][key]['frame_mean_px'] - metrics['A_N2'][key]['frame_mean_px'], key)
                for key in predictions if metrics['POSEFIX_RAW'][key].get('matched')
                and metrics['POSEFIX_RAW'][key].get('evaluable')]
    assert max(eligible)[1] == CASES[0][1]
    assert min(eligible)[1] == CASES[2][1]
    OUT.mkdir(parents=True, exist_ok=True)
    figures = []
    for tag, key, canonical, selection in CASES:
        row, old, record = predictions[key], previous[key], records[key]
        assert row['image'] == old['image'] == record['image']
        assert row['annotation'] == old['annotation'] == record['annotation']
        verify(row['image'])
        verify(row['annotation'])
        image = np.asarray(Image.open(ROOT / row['image']['path']).convert('RGB'))
        assert list(image.shape[:2]) == row['raw_hw']
        gt, manual = manual_gt(read(ROOT / row['annotation']['path']))
        assert manual[canonical]
        selected = dict(R0=top(row['predictions']['R0']), A_N2=top(row['predictions']['A_N2']),
                        REAL_ONLY=top(old['predictions']['POSEFIX_RAW']),
                        REPLAY=top(row['predictions']['POSEFIX_RAW']))
        points = {arm: np.asarray(candidate['keypoints_xy'], float) for arm, candidate in selected.items()}
        scores = dict(R0=metrics['R0'][key], A_N2=metrics['A_N2'][key],
                      REAL_ONLY=old_metrics[key], REPLAY=metrics['POSEFIX_RAW'][key])
        model_info = {}
        for arm in ARMS:
            verify_metric(points[arm], gt, manual, permutations, scores[arm])
            branch = scores[arm]['branch']
            perm = permutations[branch]
            native = int(np.flatnonzero(perm == canonical)[0])
            delta = points[arm][native] - points['R0'][native]
            model_info[arm] = dict(label=LABELS[arm], native_corner=native, canonical_GT_corner=canonical,
                C4_branch=branch, approved_whole_permutation=perm.tolist(),
                predicted_xy=points[arm][native].tolist(), R0_same_native_xy=points['R0'][native].tolist(),
                delta_from_same_native_R0_xy=delta.tolist(), movement_from_same_native_R0_px=float(np.linalg.norm(delta)),
                canonical_corner_error_px=scores[arm]['canonical_errors'][canonical],
                manual_frame_mean_px=scores[arm]['frame_mean_px'],
                all_native_corner_predictions=points[arm][:8].tolist())
        zoom_points = [gt[canonical]] + [points[a][model_info[a]['native_corner']] for a in ('R0', 'A_N2', 'REPLAY')]
        zoom_limits = clipped_limits(zoom_points, row['raw_hw'])
        crop_points = np.concatenate([gt[manual], points['R0'][:8], points['A_N2'][:8], points['REPLAY'][:8]])
        crop_limits = clipped_limits(crop_points, row['raw_hw'], margin=30)
        generated = []
        for mode in ('full', 'crop', 'corner'):
            is_corner = mode == 'corner'
            if is_corner:
                fig, axes = plt.subplots(1, 4, figsize=(18, 7.2), facecolor=BG)
                limits = zoom_limits
            else:
                fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor=BG)
                h, w = row['raw_hw']
                limits = [0, w-1, 0, h-1] if mode == 'full' else crop_limits
            for ax, arm in zip(axes.flat, ARMS):
                info = model_info[arm]
                focus = (info['native_corner'], canonical) if is_corner else None
                draw(ax, image, points[arm], points['R0'], gt, manual, edges, limits, focus)
                if is_corner:
                    dx, dy = info['delta_from_same_native_R0_xy']
                    title = (f'{LABELS[arm]}\nG{canonical} error {info["canonical_corner_error_px"]:.2f} px | '
                             f'b{info["C4_branch"]}: P{info["native_corner"]}\n'
                             f'R0 movement dx {dx:+.1f}, dy {dy:+.1f} px')
                else:
                    title = (f'{LABELS[arm]}\nmanual-GT mean error {info["manual_frame_mean_px"]:.2f} px '
                             f'| C4 whole-object branch b{info["C4_branch"]}')
                ax.set_title(title, fontsize=11, color='white', pad=10)
            title = dict(worst='Largest worsening', truncated='Truncated example', improved='Largest improvement',
                         nontruncated='Non-truncated failure')[tag]
            suffix = f' | same GT corner G{canonical}, same crop' if is_corner else f' | {mode} view'
            fig.suptitle(f'GREEN150: {title}{suffix}\n{key}', fontsize=15, color='white', y=.98)
            footer = ('Magenta x = manual GT only; yellow = prediction; white arrow = same native corner from frozen R0.\n'
                      'No PnP/unknown GT drawn. C4 branch is one whole-object permutation per model, not a per-corner match.\n'
                      'Diagnostic saved-coordinate overlays; identical RGB and crop in each row; reused development data.')
            if is_corner:
                footer += '\nCyan ring = same native R0 input. Other corners/edges remain visible for context; focus is labeled P/G.'
            fig.text(.5, .02, footer, ha='center', va='bottom', color='white', fontsize=9)
            fig.subplots_adjust(left=.02, right=.98, top=.78 if is_corner else .87,
                                bottom=.18 if is_corner else .12,
                                wspace=.045, hspace=.24)
            output = OUT / f'{tag}_{mode}.png'
            fig.savefig(output, dpi=130, facecolor=BG)
            plt.close(fig)
            generated.append(bound(output))
        figures.append(dict(tag=tag, id=key, selection=selection, image=row['image'], annotation=row['annotation'],
            manual_GT_indices=np.flatnonzero(manual).tolist(), manual_GT_xy=gt[manual].tolist(),
            focused_canonical_GT_corner=canonical, focused_GT_xy=gt[canonical].tolist(),
            crop_limits_xy=crop_limits, corner_zoom_limits_xy=zoom_limits,
            corner_zoom_source='GT and R0/current/replay whole-C4 canonical counterpart union +25px, min100px, clipped to original image',
            models=model_info, real_only_focus_inside_zoom=inside(points['REAL_ONLY'][model_info['REAL_ONLY']['native_corner']], zoom_limits),
            branch_changed_vs_R0={a: scores[a]['branch'] != scores['R0']['branch'] for a in ARMS},
            delta_replay_vs_DIM_frame_mean_px=scores['REPLAY']['frame_mean_px']-scores['A_N2']['frame_mean_px'],
            files=generated))
    sources = [pred_path, metrics_path, old_pred_path, old_metrics_path, SNAPSHOT, CONTRACT, *result_paths]
    payload = dict(figures=figures, eligible_matched_GREEN150_frames=len(eligible),
        source_bindings=[bound(p) for p in sources], source_code=bound(Path(__file__)),
        image_generation=False, image_retouch=False, inference_rerun=False, GPU_used=False,
        manual_GT_only=True, no_GT_per_corner_reassignment=True,
        uncertainty='Visualization explains individual stored outcomes, not causal proof of a training-factor effect',
        score_verification='All shown frame means, whole-C4 branches and manual canonical errors independently reproduced to 1e-6 px')
    destination = OUT / 'FIGURES.json'
    data = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if destination.exists():
        old = read(destination)
        assert old['source_bindings'] == payload['source_bindings'], 'Source predictions/GT changed'
        assert [f['id'] for f in old['figures']] == [f['id'] for f in figures], 'Case selection changed'
    # The diagnostic render/manifest can be regenerated; bound scientific inputs cannot.
    destination.write_text(data)
    print(json.dumps(dict(manifest=bound(destination), cases=[dict(id=f['id'], delta=f['delta_replay_vs_DIM_frame_mean_px']) for f in figures]), indent=2))


if __name__ == '__main__':
    main()
