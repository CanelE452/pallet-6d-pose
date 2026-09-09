"""Decisive-frame overlay for pallet_point_line_v4 (analysis mode).

For two synthetic-val frames, draws on the raw pixel grid: the GT 8 corners
(+ centroid), the original prediction (candidate index 0) and the coordinates
H_seed1 actually selected.  Left column = full frame for context, right column
= zoom on the pallet with corner indices.

Illustration of a mechanism only.  The quantitative verdict lives in
VERDICT.json / COMPARISON_synth_val.json and is not derived from this figure.
"""
import json, sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

ASSETS = Path.home() / '.claude/agents/viz-expert/assets'
plt.style.use(ASSETS / 'analysis.mplstyle')
sys.path.insert(0, str(ASSETS))
from palette import colors_for  # noqa: E402

ROOT = Path('/home/minjae/Documents/github/pallet-pose')
EXP = ROOT / 'data/pallet/results/pallet_point_line_v4'
EXPORT = EXP / 'export'
EVAL = EXP / 'evaluations/H_seed1_synth.json'
PROBE = ROOT / 'data/pallet/results/pallet_dht_decoder_probe_v1/MANIFEST.json'
OUT = EXP / 'figures/decisive_frames.png'
PAD = 100  # reflect-101 border baked into prepared_image files

TARGETS = ['G38__G__f7377', 'G38__G__f5154']
TAGS = ['Frame repaired by the selection', 'Frame destroyed by the selection']

# camera-facing 0123: front face 0-1-2-3, rear face 4-5-6-7, verticals 0-4 ...
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
# vertical pillars (top, bottom) in ring order around the pallet
PILLARS = [(0, 3), (1, 2), (5, 6), (4, 7)]

C_GT, C_BASE, C_SEL = colors_for(['gt', 'baseline', 'selected'])

man = json.load(open(EXPORT / 'synth_val.json'))['records']
ev = json.load(open(EVAL))
rows, sel_all = ev['frame_rows'], ev['selected_coordinates']
assert all(a['frame_id'] == b['frame_id'] for a, b in zip(man, rows)), 'manifest/eval order mismatch'
probe = {r['id']: r for r in json.load(open(PROBE))['records']}


def yaw_rotation_perms():
    """the 4 index permutations that are pure k*90 deg yaw relabellings"""
    out = {}
    for k in range(4):
        p = np.arange(8)
        for j, (t, b) in enumerate(PILLARS):
            tt, bb = PILLARS[(j + k) % 4]
            p[t], p[b] = tt, bb
        out[k] = p
    return out


YAW = yaw_rotation_perms()


def describe(P, gt):
    """optimal index assignment of P onto GT corners + its residual"""
    d = np.linalg.norm(P[:8, None, :] - gt[None, :8, :], axis=2)
    _, perm = linear_sum_assignment(d)          # perm[i] = GT index P[i] lands on
    resid = d[np.arange(8), perm]
    n_aligned = int((perm == np.arange(8)).sum())
    kind = 'identity' if n_aligned == 8 else 'other perm'
    for k, q in YAW.items():
        if k and np.array_equal(perm, q):
            kind = f'{90 * k} deg yaw'
    return perm, resid, n_aligned, kind


def load(fid):
    i = next(k for k, r in enumerate(rows) if r['frame_id'] == fid)
    rec, row = man[i], rows[i]
    obs = torch.load(EXPORT / rec['observation'], weights_only=True)
    sup = torch.load(EXPORT / rec['supervision'], weights_only=True)
    base = obs['baseline'][0].numpy().astype(np.float64)
    gt = sup['points'][0].numpy().astype(np.float64)
    svd = sup['supervised'][0].numpy().astype(bool)
    assert bool(sup['matched'][0]), f'{fid} not matched'
    sel = np.asarray(sel_all[i], dtype=np.float64)
    if row['chosen'] > 0:
        assert np.allclose(sel, obs['layouts'][0].numpy()[row['chosen']]), 'selected != layouts[chosen]'

    p = probe[fid]
    img = cv2.imread(p['image'])
    assert img is not None, p['image']
    if p.get('prepared_image'):
        img = img[PAD:-PAD, PAD:-PAD]
    assert img.shape[:2] == (p['height'], p['width']), f"{fid}: {img.shape[:2]} vs {(p['height'], p['width'])}"

    for key, P in (('base_mean_px', base), ('new_mean_px', sel)):
        got = np.linalg.norm((P[:8] - gt[:8])[svd[:8]], axis=1).mean()
        assert abs(got - row[key]) < 1e-3, f'{fid} {key}: {got} vs {row[key]}'
    return row, img, gt, base, sel, svd


def show_image(ax, img):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray', vmin=0, vmax=255,
              aspect='equal', zorder=0)
    ax.set_aspect('equal')
    ax.grid(False)


def draw_points(ax, gt, base, sel, m, labels, s=(70, 55, 60), lw=1.2):
    for a, b in EDGES:
        if m[a] and m[b]:
            ax.plot([gt[a, 0], gt[b, 0]], [gt[a, 1], gt[b, 1]],
                    color=C_GT, lw=0.8, alpha=0.5, zorder=1)
    for P, c in ((base, C_BASE), (sel, C_SEL)):
        for i in np.where(m)[0]:
            ax.plot([gt[i, 0], P[i, 0]], [gt[i, 1], P[i, 1]],
                    color=c, lw=1.0, alpha=0.8, zorder=2)
    ax.scatter(gt[m, 0], gt[m, 1], s=s[0], facecolors='none', edgecolors=C_GT,
               linewidths=lw, marker='o', zorder=4, label='GT (supervised)')
    ax.scatter(base[m, 0], base[m, 1], s=s[1], c=C_BASE, marker='x',
               linewidths=lw + 0.2, zorder=5, label='Original prediction (candidate 0)')
    ax.scatter(sel[m, 0], sel[m, 1], s=s[2], c=C_SEL, marker='+',
               linewidths=lw + 0.2, zorder=5, label='H_seed1 selected candidate')
    if labels:
        for i in np.where(m)[0]:
            ax.annotate(str(i), gt[i], textcoords='offset points', xytext=(7, 6),
                        fontsize=8, color=C_GT, zorder=6)
            ax.annotate(str(i), base[i], textcoords='offset points', xytext=(-14, 4),
                        fontsize=8, color=C_BASE, zorder=6)
            ax.annotate(str(i), sel[i], textcoords='offset points', xytext=(3, -13),
                        fontsize=8, color=C_SEL, zorder=6)


fig = plt.figure(figsize=(15.0, 11.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.45])
report = []

for r, (fid, tag) in enumerate(zip(TARGETS, TAGS)):
    row, img, gt, base, sel, svd = load(fid)
    h, w = img.shape[:2]
    m = svd.copy()
    pts = np.vstack([gt[m], base[m], sel[m]])
    cx0, cx1 = pts[:, 0].min(), pts[:, 0].max()
    cy0, cy1 = pts[:, 1].min(), pts[:, 1].max()
    mx = max(0.22 * (cx1 - cx0), 26.0)
    my = max(0.22 * (cy1 - cy0), 26.0)
    zx = (cx0 - mx, cx1 + mx)
    zy = (cy0 - my, cy1 + my)

    axc = fig.add_subplot(gs[r, 0])
    show_image(axc, img)
    draw_points(axc, gt, base, sel, m, labels=False, s=(22, 18, 20), lw=0.8)
    axc.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0],
                            fill=False, ec='0.25', lw=1.0, ls='--', zorder=8))
    axc.set_xlim(0, w); axc.set_ylim(h, 0)
    axc.set_xlabel('x (raw px)'); axc.set_ylabel('y (raw px)')
    axc.set_title(f'{fid}  full frame {w}x{h}  (dashed = zoom)', fontsize=10)

    axz = fig.add_subplot(gs[r, 1])
    show_image(axz, img)
    draw_points(axz, gt, base, sel, m, labels=True)
    band = 0.36 * (zy[1] - zy[0])
    axz.set_xlim(*zx); axz.set_ylim(zy[1], zy[0] - band)
    # no tick labels inside the reserved white band - they are outside the frame
    axz.set_yticks([t for t in axz.get_yticks() if zy[0] - 1 <= t <= zy[1] + 1])
    axz.set_xlabel('x (raw px)'); axz.set_ylabel('y (raw px)')

    pb, rb, nb, kb = describe(base, gt)
    ps, rs, ns, ks = describe(sel, gt)
    box = '\n'.join([
        f"candidate      {row['chosen']} / 64",
        f"base        {row['base_mean_px']:9.3f} px",
        f"selected    {row['new_mean_px']:9.3f} px",
        f"delta       {row['new_mean_px'] - row['base_mean_px']:+9.3f} px",
        "-- after optimal index assignment --",
        f"base index  {nb}/8 kept ({kb})",
        f"sel  index  {ns}/8 kept ({ks})",
        f"base        {rb.mean():9.3f} px",
        f"selected    {rs.mean():9.3f} px",
    ])
    axz.text(0.015, 0.985, box, transform=axz.transAxes, ha='left', va='top',
             family='monospace', fontsize=8.5,
             bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.95), zorder=7)
    axz.set_title(f"{tag}\n{fid}   base {row['base_mean_px']:.3f}px "
                  f"-> selected {row['new_mean_px']:.3f}px  (mean over 8 corners)", fontsize=11)
    report.append((fid, row, nb, kb, rb.mean(), ns, ks, rs.mean(), pb.tolist(), ps.tolist()))

fig.suptitle('H_seed1 candidate selection on synthetic val (512 frames) - the two decisive frames\n'
             'Error = mean L2 over the 8 supervised corners (index 8 = centroid, excluded from the metric).\n'
             'Both flips are the same 270 deg yaw re-indexing of the corner labels (one 90 deg step): '
             'after an optimal index assignment '
             'every estimate sits within ~5 px of the GT corners.\n'
             'Shown to expose the mechanism; not evidence of an accuracy change.', fontsize=11)
handles, labels = fig.axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=True, framealpha=0.9,
           columnspacing=1.2, handletextpad=0.4, borderpad=0.4)
fig.text(0.005, 0.004, f'src: {EVAL.relative_to(ROOT)}', ha='left', fontsize=8, color='gray')
fig.text(0.995, 0.004, datetime.now().strftime('%Y-%m-%d %H:%M'), ha='right', fontsize=8, color='gray')
fig.savefig(OUT)
print('saved', OUT)

for fid, row, nb, kb, rbm, ns, ks, rsm, pb, ps in report:
    print(f'{fid}: base {row["base_mean_px"]:.3f} -> sel {row["new_mean_px"]:.3f} px, cand {row["chosen"]}')
    print(f'   base perm {pb} kept {nb}/8 [{kb}] index-matched err {rbm:.3f}px')
    print(f'   sel  perm {ps} kept {ns}/8 [{ks}] index-matched err {rsm:.3f}px')
