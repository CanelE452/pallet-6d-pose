"""Plot the saved SOURCE reference-alignment census; no image/model access."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V226_ROOT_ALIGNMENT')


def main():
    rows = [json.loads(s) for s in (R / 'REFERENCE_ALIGNMENT368.jsonl').read_text().splitlines()]
    assert len(rows) == 368
    by_label = {k: [r for r in rows if r['existing_weak_role'] == k] for k in ('D', 'E')}
    colors = {'D': '#2876b5', 'E': '#d34b32'}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.3), gridspec_kw={'width_ratios': [1, 1, .85]})
    outside_zoom = {}
    for k, data in by_label.items():
        outside_zoom[k] = sum(abs(r['dx_over_L']) > .65 or abs(r['dy_over_L']) > .65 for r in data)
        for ax in axes[:2]:
            ax.scatter([r['dx_over_L'] for r in data], [r['dy_over_L'] for r in data],
                       s=16 if k == 'D' else 29, alpha=.5 if k == 'D' else .85,
                       marker='o' if k == 'D' else '^', color=colors[k], label=f'{k} weak ({len(data)})', zorder=3)
    for ax in axes[:2]:
        ax.add_patch(Rectangle((-.5, -.5), 1, 1, fill=False, edgecolor='#333333', lw=1.5, zorder=4))
        ax.scatter([0], [0], color='black', marker='x', s=55, zorder=5, label='Initial query')
        ax.axhline(0, color='#b9bec5', lw=.6); ax.axvline(0, color='#b9bec5', lw=.6)
        ax.set_xlabel('(reference x - query x) / L')
        ax.set_ylabel('(reference y - query y) / L')
        ax.grid(alpha=.15); ax.set_aspect('equal', adjustable='box')
    axes[0].invert_yaxis(); axes[0].set_title('All 368 points (full extent)', fontsize=11)
    axes[0].legend(fontsize=8, loc='lower right')
    axes[1].set_xlim(-.65, .65); axes[1].set_ylim(.65, -.65)
    axes[1].set_title('Crop neighborhood (zoom)', fontsize=11)
    axes[1].text(.02, .02, f'Outside this zoom: D={outside_zoom["D"]}, E={outside_zoom["E"]}\nAll points remain in the left panel.',
                 transform=axes[1].transAxes, fontsize=7.5, va='bottom', bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    ax = axes[2]
    for j, k in enumerate(('D', 'E')):
        data = by_label[k]; n = len(data); inside = sum(r['reference_in_continuous_crop_square'] for r in data)
        pct = inside / n * 100
        ax.barh(j, pct, color='#4e997c', height=.55)
        ax.barh(j, 100-pct, left=pct, color='#c9ced5', height=.55)
        ax.text(3, j, f'{inside}/{n} inside\n{pct:.1f}%', va='center', fontsize=10, color='white')
        ax.text(99, j, f'{n-inside}\noutside', va='center', ha='right', fontsize=8)
    ax.set_yticks([0,1], ['D weak', 'E weak']); ax.invert_yaxis(); ax.set_xlim(0,100)
    ax.set_xlabel('Rows (%)'); ax.set_title('Reference inside original crop', fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Public TRAIN: nominal reference alignment at unchanged initial queries', fontsize=14, y=.98)
    fig.text(.02, .045, 'Square = q +/- L/2 (continuous geometry). This is not a replay of RAW96 sampling/validity.', fontsize=9)
    fig.text(.02, .015, 'D/E = existing assistant weak role labels. CAD/pose references are not physical keypoint gold. No training or current223 evaluation.', fontsize=9)
    fig.subplots_adjust(left=.06, right=.99, top=.86, bottom=.17, wspace=.36)
    dest = R / 'SOURCE_REFERENCE_ALIGNMENT.png'
    assert not dest.exists()
    fig.savefig(dest, dpi=170, facecolor='white'); plt.close(fig)
    print(json.dumps({'plot': str(dest), 'outside_zoom': outside_zoom, 'all_rows': len(rows)}))


if __name__ == '__main__':
    main()
