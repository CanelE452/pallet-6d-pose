"""Plot recorded raw/refined coordinates; never generate or alter predictions."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_real_refiner_twostage_v1'
OUT = ROOT / 'outputs/pallet_real_refiner_twostage_v1'
RAW_COLOR = '#00d9ff'
REF_COLOR = '#ff983f'


def main():
    data = json.loads((DOC / 'STAGE2.json').read_text())
    rows = sorted(data['accepted'], key=lambda r: (-float(np.mean(r['correction_px'])), r['image_sha256']))
    row = rows[0]
    path = ROOT / row['image_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['image_sha256']
    raw = np.array(row['raw_points'], float)
    refined = np.array(row['refined_points'], float)
    distance = np.linalg.norm(raw[:8] - refined[:8], axis=1)
    np.testing.assert_allclose(distance, row['correction_px'], atol=1e-10, rtol=0)
    picture = np.asarray(Image.open(path).convert('RGB'))
    h, w = picture.shape[:2]
    edges = json.loads((ROOT / '_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json').read_text())['edges']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig = plt.figure(figsize=(14, 10), facecolor='#131923')
    grid = fig.add_gridspec(2, 2, height_ratios=[1, .65], hspace=.18, wspace=.05)
    axs = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    bottom = max(h, np.max(np.r_[raw[:8, 1], refined[:8, 1]]) + 20)
    top = max(0, min(raw[:8, 1].min(), refined[:8, 1].min()) - 25)
    for ax in axs:
        ax.set_facecolor('#344051')
        ax.imshow(picture, extent=(-.5, w-.5, h-.5, -.5), interpolation='nearest')
        ax.set_xlim(-.5, w-.5)
        ax.set_ylim(bottom, -.5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values(): spine.set_visible(False)
        if bottom > h:
            ax.axhline(h-.5, color='#b4bbc8', ls='--', lw=.8)
            ax.text(8, bottom-8, 'Gray area: outside the recorded image', color='#e4e8ef', fontsize=8)

    def draw(ax, points, color, labels=True, alpha=1., dashed=False):
        for a, b in edges:
            ax.plot(points[[a,b], 0], points[[a,b], 1], color=color, lw=1.3,
                    alpha=alpha, ls='--' if dashed else '-')
        ax.scatter(points[:8, 0], points[:8, 1], s=22, c=color, edgecolors='black', linewidths=.6, zorder=4)
        if labels:
            for j, (x, y) in enumerate(points[:8]):
                ax.annotate(str(j), (x, y), xytext=(5, -10), textcoords='offset points',
                    color='white', fontsize=10, weight='bold', zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])

    draw(axs[0], raw, RAW_COLOR)
    draw(axs[1], refined, REF_COLOR)
    axs[0].set_title('BEFORE  |  R0 raw keypoints', color=RAW_COLOR, fontsize=14, pad=12)
    axs[1].set_title('AFTER  |  frozen N2 refined keypoints', color=REF_COLOR, fontsize=14, pad=12)
    axs[2].set_ylim(bottom, top)
    draw(axs[2], raw, RAW_COLOR, labels=False, alpha=.6, dashed=True)
    draw(axs[2], refined, REF_COLOR, labels=False, alpha=.85)
    for j, (a, b, d) in enumerate(zip(raw[:8], refined[:8], distance)):
        axs[2].annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', color='white', lw=1.15), zorder=6)
        axs[2].annotate(f'{j}: {d:.2f}px', b, xytext=(7, -14 if j % 2 else 10),
            textcoords='offset points', color='white', fontsize=9, weight='bold', zorder=7,
            path_effects=[pe.withStroke(linewidth=2.8, foreground='black')])
    axs[2].set_title('DETAIL  |  cyan = raw, orange = refined; arrows show actual movement (not exaggerated)',
                     color='white', fontsize=12, pad=10)
    fig.suptitle(f'Largest mean correction among {len(rows)} accepted real pseudo-label images\n'
                 f'Mean corner movement: {distance.mean():.2f} px   |   Maximum: {distance.max():.2f} px',
                 color='white', fontsize=16, y=.975)
    fig.text(.5, .028, f"{row['capture_session']} / {path.name}  |  8 corners, centroid excluded\n"
             'No ground-truth accuracy claim: largest movement is not necessarily the best correction.',
             ha='center', color='#d0d7e1', fontsize=10)
    fig.subplots_adjust(left=.025, right=.975, top=.87, bottom=.10)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / 'largest_pseudo_correction.png'
    fig.savefig(destination, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    metadata = dict(selection='maximum arithmetic mean movement of eight corners among STAGE2 accepted images',
        population=len(rows), rank=1, image=row['image_path'], image_sha256=row['image_sha256'],
        mean_movement_px=float(distance.mean()), max_movement_px=float(distance.max()),
        movement_px=distance.tolist(), raw_points=raw.tolist(), refined_points=refined.tolist(),
        displayed_refiner='frozen N2_DIM_ONLY seed1 teacher used for pseudo labels, NOT the newly adapted student',
        source_stage2_sha256=hashlib.sha256((DOC/'STAGE2.json').read_bytes()).hexdigest(),
        selected_by_accuracy=False, gt_used=False,
        figure=str(destination.relative_to(ROOT)))
    (OUT / 'largest_pseudo_correction.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(destination)
    print(json.dumps({k:metadata[k] for k in ['population','mean_movement_px','max_movement_px','image']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
