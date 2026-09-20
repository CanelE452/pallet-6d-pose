"""Saved-coordinate RGB overlays; no image generation or inference rerun."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_posefix_replay_v1 import core as N
from scripts.research.pallet_posefix_replay_v1.evaluate import ARMS, EXAMPLE, top, verify_result

LABELS = dict(R0='Frozen R0 input', A_N2='Current frozen DIM-only refiner',
              POSEFIX_RAW='PoseFix real + synthetic replay / uncapped',
              POSEFIX_CAP8='Same replay output / 1% image-diagonal cap')


def main():
    N.verify()
    pred_path, metric_path = N.RAW / 'PREDICTIONS.json', N.RAW / 'PER_FRAME_METRICS.json'
    result_path = N.DOC / 'REAL_RESULTS.json'
    result = N.E.read(result_path)
    assert result['model'] == 'replay'
    verify_result(result)
    predictions, metrics = N.E.read(pred_path), N.E.read(metric_path)
    edges = N.E.read(N.E.SYM_DOC / 'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['edges']
    candidates = []
    for dataset in ('DEV72', 'GREEN150'):
        score_key = dataset if dataset == 'DEV72' else 'GREEN150_MANUAL'
        scores = {arm: {row['id']: row for row in metrics[score_key][arm]} for arm in ARMS}
        for row in predictions[dataset]:
            initial, raw = scores['R0'][row['id']], scores['POSEFIX_RAW'][row['id']]
            if not initial['evaluable'] or not initial['matched'] or not raw['matched']:
                continue
            assert all(top(row['predictions'][arm]) is not None for arm in ARMS)
            candidates.append(dict(dataset=dataset, row=row,
                scores={arm: values[row['id']] for arm, values in scores.items()},
                delta_raw_vs_R0_px=raw['frame_mean_px'] - initial['frame_mean_px'],
                delta_raw_vs_N2_px=raw['frame_mean_px'] - scores['A_N2'][row['id']]['frame_mean_px']))
    assert candidates
    fixed = [row for row in candidates if row['row']['id'] == EXAMPLE]
    assert len(fixed) == 1
    chosen = [('user_example', fixed[0]),
              ('most_favorable_raw_change', min(candidates, key=lambda r: (r['delta_raw_vs_R0_px'], r['row']['id']))),
              ('most_unfavorable_raw_change', max(candidates, key=lambda r: (r['delta_raw_vs_R0_px'], r['row']['id'])))]
    N.OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for tag, entry in chosen:
        row = entry['row']
        N.F.verify(row['image'])
        N.F.verify(row['annotation'])
        im = np.array(Image.open(ROOT / row['image']['path']).convert('RGB'))
        h, w = im.shape[:2]
        assert [h, w] == row['raw_hw']
        annotation = N.E.read(ROOT / row['annotation']['path'])
        gt, valid = N.C.R.annotation_arrays(annotation)
        _, manual = N.C.R.annotation_arrays(annotation, True)
        valid &= np.isfinite(gt).all(-1) & ~(gt == -1).all(-1)
        points = {arm: np.asarray(top(row['predictions'][arm])['keypoints_xy'], float) for arm in ARMS}
        initial = points['R0']
        displacements = {arm: np.linalg.norm(q[:8] - initial[:8], axis=-1).tolist()
                         for arm, q in points.items()}
        generated = []
        for zoom in ((False, True) if tag == 'user_example' else (False,)):
            fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor='#141b25')
            for ax, arm in zip(axes.flat, ARMS):
                q = points[arm]
                ax.imshow(im)
                for a, b in edges:
                    if valid[a] and valid[b]:
                        ax.plot(gt[[a, b], 0], gt[[a, b], 1], color='#77ff65',
                                lw=1.6, alpha=.9, zorder=3)
                    if arm != 'R0':
                        ax.plot(initial[[a, b], 0], initial[[a, b], 1],
                                color='#00d9ff', lw=1, ls='--', alpha=.65, zorder=2)
                    if np.isfinite(q[[a, b]]).all():
                        ax.plot(q[[a, b], 0], q[[a, b], 1],
                                color='#00d9ff' if arm == 'R0' else '#ffac39', lw=1.6, zorder=4)
                for corner in range(8):
                    if valid[corner]:
                        ax.scatter(*gt[corner], s=16,
                                   facecolors='#77ff65' if manual[corner] else 'none',
                                   edgecolors='#77ff65', linewidths=.9, zorder=5)
                    if np.isfinite(q[corner]).all():
                        ax.scatter(*q[corner], s=22, c='#ffe05a', zorder=6)
                if zoom:
                    assert (h, w) == (480, 640)
                    ax.set_xlim(475, 640)
                    ax.set_ylim(425, 245)
                    if arm != 'R0':
                        for before, after in zip(initial[:8], q[:8]):
                            if np.isfinite([before, after]).all():
                                ax.annotate('', xy=after, xytext=before,
                                    arrowprops=dict(arrowstyle='->', color='white', lw=1.1), zorder=7)
                else:
                    ax.set_xlim(0, w)
                    ax.set_ylim(h, 0)
                move = np.asarray(displacements[arm])
                score = entry['scores'][arm]
                ax.set_title(f'{LABELS[arm]}\n'
                    f'movement mean/max {move.mean():.1f}/{move.max():.1f} px | '
                    f'whole-object-sym mean error {score["frame_mean_px"]:.1f} px',
                    fontsize=10, color='white')
                ax.axis('off')
            case_title = 'User-specified right edge' if zoom else tag.replace('_', ' ')
            fig.suptitle(f'Real + synthetic replay last300 | {case_title}\n{row["id"]}',
                         color='white', fontsize=13, y=.98)
            fig.text(.5, .035,
                'Green = stored annotation; cyan = frozen R0; orange = replay model output; white arrows = movement.\n'
                'Solid green dots = manual clicks; hollow = other/unknown provenance. No per-corner GT reassignment.\n'
                'Uncapped/capped outputs both shown, without selecting the better one. Reused development data, not independent validation.',
                ha='center', color='white', fontsize=9)
            fig.subplots_adjust(top=.88, bottom=.13, left=.02, right=.98, wspace=.04, hspace=.20)
            destination = N.OUT / f'{tag}{"_right_zoom" if zoom else ""}.png'
            fig.savefig(destination, dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            generated.append(N.E.bound(destination))
        manifest.append(dict(tag=tag, id=row['id'], dataset=entry['dataset'],
            image=row['image'], annotation=row['annotation'], files=generated,
            scores=entry['scores'], movement_px=displacements,
            delta_raw_vs_R0_frame_mean_px=entry['delta_raw_vs_R0_px'],
            delta_raw_vs_N2_frame_mean_px=entry['delta_raw_vs_N2_px'],
            selection='fixed user example' if tag == 'user_example' else
                'minimum/maximum uncapped replay PoseFix minus R0 symmetry-aware frame mean error on all matched DEV72+GREEN150 manual cases; both extremes included'))
    payload = dict(model='replay', figures=manifest, eligible_matched_frames=len(candidates),
        source_predictions=N.E.bound(pred_path), source_metrics=N.E.bound(metric_path),
        results=N.E.bound(result_path), source_code=N.E.bound(Path(__file__)),
        generated_imagery=False, inference_rerun=False,
        GT_per_corner_reassignment=False, checkpoint_selection=False,
        note='Diagnostic visualizations only; case selection does not affect training or full-population metrics')
    N.freeze(N.OUT / 'FIGURES.json', payload)
    print(N.OUT, flush=True)
    for entry in manifest:
        print(entry['tag'], entry['id'], entry['delta_raw_vs_R0_frame_mean_px'], flush=True)
    return payload


if __name__ == '__main__':
    main()
