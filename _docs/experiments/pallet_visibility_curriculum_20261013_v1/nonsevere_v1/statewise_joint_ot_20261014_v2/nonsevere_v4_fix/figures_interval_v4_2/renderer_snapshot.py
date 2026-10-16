"""Render sealed actual-image comparisons; no fitting or parameter selection."""
import argparse
from copy import deepcopy
import html
from pathlib import Path
import shutil

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from . import common as C
from .joint_ot_pnp import project_native
from .statewise_joint_visualize_v2 import EDGES

COLORS = {'S': '#c689f4', 'B': '#4eb6ff', 'I': '#5cffab',
          'O': '#ffad42', 'T': '#ff6969', 'U': '#bbbbbb'}
FIXED_CASES = ['eval_pallet07:1778652170735118080', 'wood_183705:000706',
               'eval_night08:1779449488068910592',
               'eval_pallet09:1778653661653195264']


def wire(ax, points, color, label, alpha=1., style='-'):
    points = np.asarray(points)
    for n, (a, b) in enumerate(EDGES):
        ax.plot(points[[a, b], 0], points[[a, b], 1], color=color,
                alpha=alpha, linestyle=style, lw=1.7,
                label=label if n == 0 else None)


def render(row, scored, output, selected_arm):
    image = cv2.imread(str(C.ROOT / row['image']))
    if image is None or C.sha(C.ROOT / row['image']) != row['image_sha256']:
        raise ValueError('Actual image binding changed')
    rt = row['runtime']; geom = rt['initial_geometry']; h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    for ax in axes:
        ax.imshow(rgb); ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title('Initial PnP + non-S backbone points')
    axes[1].set_title('Frozen B/I observations + all O/T regions')
    axes[2].set_title('Selected rigid pose; S = PnP projection')
    initial = np.asarray(geom['initial_projection_xy'])
    final = np.asarray(rt['keypoint_readout']['rigid_model_corners'])
    wire(axes[0], initial, '#f4e45b', 'Initial PnP')
    wire(axes[2], initial, '#f4e45b', 'Initial PnP', .65, '--')
    wire(axes[2], final, '#59edec', 'Selected rigid pose')
    for k, state in enumerate(rt['states']):
        if state != 'S':
            q = np.asarray(rt['original_raw_points'][k])
            axes[0].scatter(*q, s=40, color=COLORS[state], marker='x')
            axes[0].annotate(f'{k}:{state}', q, xytext=(4, 4),
                             textcoords='offset points', color=COLORS[state], fontsize=8)
        if state == 'S':
            axes[2].scatter(*final[k], color=COLORS[state], marker='x', s=60)
            axes[2].annotate(f'{k}:S', final[k], xytext=(4, 4),
                             textcoords='offset points', color=COLORS[state], fontsize=8)
    for measured in rt['frozen_visible_measurements']:
        q = measured['xy']; k = measured['native_corner']; state = measured['state']
        axes[1].scatter(*q, s=50, color=COLORS[state], edgecolors='black', lw=.5)
        axes[1].annotate(f'{k}:{state}', q, xytext=(4, 4), textcoords='offset points',
                         color=COLORS[state], fontsize=9)
    for d in rt['ot_domains']['domains']:
        k = d['native_corner']
        if d['shape'] == 'ellipse':
            axes[1].add_patch(Ellipse(d['center_xy'], *[2*x for x in d['half_axes_px']],
                angle=d.get('angle_deg', 0.), edgecolor=COLORS['O'], facecolor='none', lw=1.5))
            axes[1].annotate(f'{k}:O region', d['center_xy'], color=COLORS['O'], fontsize=8)
        elif d['state'] == 'T':
            axes[1].text(.02, .04+.045*k, f'{k}:T outside {d.get("outside_canvas_side", "unknown")}',
                         transform=axes[1].transAxes, color=COLORS['T'], fontsize=8)
        else:
            raise ValueError('This scientific renderer requires ellipse O regions')
    for ax in (axes[0], axes[2]):
        ax.legend(loc='upper right', fontsize=8)
    metrics = scored['metrics']; old = metrics['ORIGINAL_N3_PNP']; new = metrics[selected_arm]
    previous = metrics.get('FROZEN_STATEWISE_V3')
    status = 'ambiguous pose family' if scored['material_ambiguity'] else 'one selected hypothesis; global uniqueness unproved'
    if scored['actual_pose_fallback']:
        status = 'explicit fallback; O/T unresolved'
    caption = (f'Initial: T {old["translation_cm"]:.2f} cm, R {old["rotation_deg"]:.2f} deg | '
               f'Selected: T {new["translation_cm"]:.2f} cm, R {new["rotation_deg"]:.2f} deg\n'
               f'BI={scored["BI_count"]}, OT={scored["OT_count"]}; {status}; '
               f'OT changes pose vs NO_OT={scored["OT_changes_pose_vs_no_OT"]}\n'
               'O ellipses are CAL-expanded backbone regions, not measured occluder outlines. '
               'T is conditional unless independently certified.\n'
               'Scores use historical annotation-derived C2 reference; raw RGB is an actual evaluation image.')
    if previous is not None:
        caption += f' Previous v3: T {previous["translation_cm"]:.2f} cm, R {previous["rotation_deg"]:.2f} deg.'
    fig.suptitle(row['id'], fontsize=12)
    fig.text(.015, .025, caption, fontsize=8, linespacing=1.5)
    fig.subplots_adjust(left=.01, right=.995, top=.86, bottom=.22, wspace=.03)
    fig.savefig(output.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(output.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)
    return caption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--selected-arm', required=True)
    args = parser.parse_args()
    run, out = args.evaluation.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError('Preserve completed figures')
    seal = C.read(run / 'PREDICTION_SEAL.json')
    if not seal['before_scoring'] or C.sha(run / 'PREDICTIONS.jsonl.gz') != seal['predictions_sha256']:
        raise ValueError('A complete prediction seal is required')
    protocol = C.read(run / 'PROTOCOL.json')
    if C.sha(run / 'PROTOCOL.json') != seal['protocol_sha256']:
        raise ValueError('Frozen evaluation protocol changed')
    if args.selected_arm != protocol['arms'][0]:
        raise ValueError('The plotted runtime must match the primary scored arm')
    score = C.read(run / 'POSTSEAL_SCORING.json')
    if C.sha(run / 'PREDICTION_SEAL.json') != score['prediction_seal']['sha256']:
        raise ValueError('Score/prediction seal mismatch')
    rows = {r['id']: r for r in C.rows(run / 'PREDICTIONS.jsonl.gz')}
    scored = {r['id']: r for r in score['rows']}
    primary = [r for r in score['rows'] if r['cohort'] == 'PRIMARY_NONSEVERE']
    regressions = [r['id'] for r in primary if r['metrics']['ORIGINAL_N3_PNP']['success_5cm_5deg']
                   and not r['metrics'][args.selected_arm]['success_5cm_5deg']]
    changed_OT = [r['id'] for r in primary if r['OT_changes_pose_vs_no_OT']]
    selected = list(dict.fromkeys(FIXED_CASES + regressions + changed_OT))
    out.mkdir(parents=True)
    records = []
    for fid in selected:
        row = rows[fid]; filename = fid.replace(':', '__')
        caption = render(row, scored[fid], out / filename, args.selected_arm)
        records.append({'id': fid, 'png': filename+'.png', 'svg': filename+'.svg',
                        'caption': caption, 'metrics': deepcopy(scored[fid]['metrics']),
                        'image_file_sha256': row['image_sha256'],
                        'png_sha256': C.sha(out / (filename+'.png')),
                        'svg_sha256': C.sha(out / (filename+'.svg'))})
    index = '<meta charset="utf-8"><title>Actual-image statewise fix review</title>'
    index += '<h1>Actual-image comparison</h1><p>Four fixed historical cases, every original-success to selected-failure case, and every primary OT pose-change case. No fitting or pose selection occurs here.</p>'
    for r in records:
        index += '<h2>'+html.escape(r['id'])+'</h2><img style="width:100%" src="'+html.escape(r['png'])+'"><pre>'+html.escape(r['caption'])+'</pre>'
    (out / 'index.html').write_text(index, encoding='utf-8')
    shutil.copyfile(__file__, out / 'renderer_snapshot.py')
    C.write(out / 'FIGURE_RECEIPT.json', {'schema': 'POSTSEAL_ACTUAL_RGB_FIX_FIGURES_V1',
        'prediction_seal_sha256': C.sha(run / 'PREDICTION_SEAL.json'),
        'score_sha256': C.sha(run / 'POSTSEAL_SCORING.json'), 'renderer_sha256': C.sha(__file__),
        'selection': 'four fixed historical cases plus all original-success to selected-failure and all primary OT pose changes',
        'no_training_or_tuning': True, 'annotation_files_opened': False, 'records': records})


if __name__ == '__main__':
    main()
