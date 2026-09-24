"""Reproduce the read-only H36/T2 corner-identity audit and report figures.

Writes documentation artifacts only. No model/annotation changes or training.
Run from repository root with the pallet-yolo26 environment.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'data/pallet/results/pallet_existing_data_transfer_v1'
DOC = ROOT / '_docs/experiments/pallet_existing_data_transfer_v1'
FIG = DOC / 'figures'
CASE = 'plastic_day_01:011067'
ARM = 'T2_HARD_MANUAL'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def selected(row):
    pred = row['prediction']
    return np.asarray(pred['candidates'][pred['selected_index']]['keypoints_xy'])


def main():
    targets = read(RAW / 'TARGETS.json')
    target = next(r for r in targets if r['id'] == CASE)
    diagnostics = read(RAW / 'DIAGNOSTICS.json')
    gt = np.asarray(target['manual'])
    teacher = np.asarray(target['target'])
    models = {arm: selected(next(r for r in diagnostics[arm]['train'] if r['id'] == CASE))
              for arm in ('T1_HARD_PSEUDO', ARM)}
    pred = models[ARM]
    mask = np.asarray(target['mask'], dtype=bool)
    assert np.flatnonzero(mask).tolist() == [0, 3, 4]
    rows = []
    for j in np.flatnonzero(mask):
        distance = np.linalg.norm(pred[:8] - gt[j], axis=1)
        other = int(np.argmin(distance))
        rows.append(dict(corner=int(j), gt_xy=gt[j].tolist(),
                         teacher_xy=teacher[j].tolist(), T2_xy=pred[j].tolist(),
                         teacher_error=float(np.linalg.norm(teacher[j]-gt[j])),
                         T1_error=float(np.linalg.norm(models['T1_HARD_PSEUDO'][j]-gt[j])),
                         T2_error=float(distance[j]), nearest_T2_channel=other,
                         nearest_T2_error=float(distance[other])))
    hard_errors = []
    for t in targets:
        if t['role'] != 'H':
            continue
        p = selected(next(r for r in diagnostics[ARM]['train'] if r['id'] == t['id']))
        for j in np.flatnonzero(t['mask']):
            hard_errors.append(dict(id=t['id'], corner=int(j),
                                    error=float(np.linalg.norm(p[j]-np.asarray(t['manual'])[j]))))
    failures = [r for r in hard_errors if r['error'] > 20]
    assert len(hard_errors) == 36 and len(failures) == 3
    assert {r['id'] for r in failures} == {CASE}

    occurrences = [r for r in read(RAW / 'OCCURRENCES.json')
                   if r['role'] == 'REPLACEMENT' and r['pair'] == target['pair']]
    trace = {r['occ']: r for r in read(RAW / 'TRACE_T2_HARD_MANUAL.json')}
    exposures = np.zeros(9, dtype=int)
    covered = np.zeros(9, dtype=int)
    max_forward = max_inverse = 0.
    for r in occurrences:
        cache = ROOT / r['cache']['M']['path']
        assert sha(cache) == r['cache']['M']['sha256']
        with np.load(cache) as z:
            k = z['keypoints']
            assert digest(k) == trace[r['occ']]['target']
            assert digest(k[:, :, 2]) == trace[r['occ']]['mask']
            assert trace[r['occ']]['supervised'] == 3
            assert np.array_equal(k[0, :, 2] == 2, mask)
            exposures += (k[0, :, 2] == 2)
            affine = np.asarray(r['native_to_model']['H'])
            expected = np.c_[gt, np.ones(9)] @ affine.T
            actual = k[0, :, :2].astype(float) * 640
            max_forward = max(max_forward, float(np.abs(expected[mask, :2]-actual[mask]).max()))
            inverse = np.c_[actual, np.ones(9)] @ np.linalg.inv(affine).T
            max_inverse = max(max_inverse, float(np.abs(inverse[mask, :2]-gt[mask]).max()))
            if r['plans']['M']['applied']:
                x, y, w, h = r['plans']['M']['S1']
                covered += ((actual[:, 0] >= x) & (actual[:, 0] < x+w)
                            & (actual[:, 1] >= y) & (actual[:, 1] < y+h) & mask)
    for field in ('image', 'annotation'):
        assert sha(ROOT / target[field]['path']) == target[field]['sha256']
    result = dict(case=CASE, scope='TRAIN H36 diagnostic; not heldout performance',
                  interpretation='Corner identity mismatch pattern; optimization cause not established',
                  rows=rows, all_H36=hard_errors, failures=failures,
                  remaining_33_max_error=max(r['error'] for r in hard_errors if r['error'] <= 20),
                  cache_audit=dict(occurrences=len(occurrences), exposures=exposures.tolist(),
                                   epoch_counts=dict(Counter(r['epoch'] for r in occurrences)),
                                   cache_sha_and_actual_trace_match=True,
                                   max_forward_abs_pixel_error=max_forward,
                                   max_inverse_abs_pixel_error=max_inverse,
                                   S1_patch_covered_counts=covered.tolist()),
                  inputs={name:dict(path=str((RAW/name).relative_to(ROOT)), sha256=sha(RAW/name))
                          for name in ('TARGETS.json','DIAGNOSTICS.json','OCCURRENCES.json',
                                       'TRACE_T2_HARD_MANUAL.json')},
                  image=target['image'], annotation=target['annotation'])
    (DOC / 'THREE_CORNER_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')

    im = np.asarray(Image.open(ROOT / target['image']['path']).convert('RGB'))
    def canvas(ax):
        ax.imshow(im)
        ax.set_xlim(325, 640)
        ax.set_ylim(375, 175)
        ax.axis('off')
    def points(ax, pts, ids, color, prefix, offset=(5, -11)):
        ids = list(ids)
        ax.scatter(pts[ids, 0], pts[ids, 1], s=55, facecolors='none',
                   edgecolors=color, linewidths=1.7)
        for j in ids:
            ax.annotate(f'{prefix}{j}', pts[j], xytext=offset, textcoords='offset points',
                        color=color, fontsize=10, weight='bold',
                        bbox=dict(facecolor='black', alpha=.6, pad=1, edgecolor='none'))
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, title, pts, ids, col, prefix in zip(
        axes, ['Manual targets used in T2 (3 points)', 'Frozen teacher (not original R0)',
               'T1: hard pseudo supervision', 'T2: hard manual supervision'],
        [gt, teacher, models['T1_HARD_PSEUDO'], pred],
        [[0, 3, 4], range(8), range(8), range(8)],
        ['#36ff57', '#ffc83b', '#00ddff', '#00ddff'], ['GT P', 'P', 'P', 'P']):
        canvas(ax)
        points(ax, pts, ids, col, prefix)
        ax.set_title(title, fontsize=11)
    fig.suptitle('TRAIN plastic_day_01:011067 | native keypoints, NO new PnP or relabeling', fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / 'three_corner_01_comparison.jpg', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, row in zip(axes, rows):
        j, near = row['corner'], row['nearest_T2_channel']
        canvas(ax)
        ax.plot([gt[j, 0], pred[j, 0]], [gt[j, 1], pred[j, 1]], '--', color='#ff6666', lw=2)
        points(ax, gt, [j], '#36ff57', 'GT P', offset=(5, 12))
        points(ax, pred, [j], '#ff6666', 'T2 P')
        points(ax, pred, [near], '#00ddff', 'T2 P')
        ax.set_title(f'GT P{j}: same P{j} = {row["T2_error"]:.2f}px\n'
                     f'Nearest other P{near} = {row["nearest_T2_error"]:.2f}px', fontsize=12)
    fig.suptitle('Identity mismatch diagnosis only: nearest-channel distance is NOT a corrected metric', fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / 'three_corner_02_identity.jpg', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    errors = [r['error'] for r in hard_errors]
    ax.bar(np.arange(36), errors, color=['#d84b4b' if e > 20 else '#348ab5' for e in errors])
    ax.axhline(20, color='#d84b4b', ls='--', label='20px')
    ax.axhline(5, color='#348ab5', ls=':', label='5px')
    ax.set(xlabel='Supervised H corner (fixed target order)', ylabel='Native image error (px)',
           title='T2 TRAIN fit: 33/36 within 5px; all 3 failures belong to ONE frame')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / 'three_corner_03_H36_fit.png', dpi=160)
    plt.close(fig)
    print(json.dumps({k: result[k] for k in ('rows','cache_audit','remaining_33_max_error')}, indent=2))


if __name__ == '__main__':
    main()
