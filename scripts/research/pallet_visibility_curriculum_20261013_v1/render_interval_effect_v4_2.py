"""Actual images of every primary pose changed by the previous O/T constraints."""
import argparse
import html
from pathlib import Path
import shutil

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import common as C
from .render_statewise_fix_v4_1 import wire


def load(run):
    seal = C.read(run / 'PREDICTION_SEAL.json')
    score = C.read(run / 'POSTSEAL_SCORING.json')
    protocol = C.read(run / 'PROTOCOL.json')
    assert seal['before_scoring']
    assert C.sha(run / 'PREDICTIONS.jsonl.gz') == seal['predictions_sha256']
    assert C.sha(run / 'PROTOCOL.json') == seal['protocol_sha256']
    assert C.sha(run / 'PREDICTION_SEAL.json') == score['prediction_seal']['sha256']
    return ({r['id']: r for r in C.rows(run / 'PREDICTIONS.jsonl.gz')},
            {r['id']: r for r in score['rows']}, protocol['arms'][0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--after', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve previous figures')
    before, sb, arm_b = load(a.before)
    after, sa, arm_a = load(a.after)
    ids = [fid for fid, r in sb.items() if r['cohort'] == 'PRIMARY_NONSEVERE'
           and r['OT_changes_pose_vs_no_OT']]
    a.output.mkdir(parents=True)
    records = []
    for fid in ids:
        old, new = before[fid], after[fid]
        assert old['image'] == new['image'] and old['image_sha256'] == new['image_sha256']
        assert C.sha(C.ROOT / new['image']) == new['image_sha256']
        image = cv2.imread(str(C.ROOT / new['image']))
        if image is None:
            raise ValueError('Actual RGB image unreadable')
        changed = [k for k, (x, y) in enumerate(zip(old['runtime']['states'], new['runtime']['states']))
                   if x != y]
        assert all(old['runtime']['states'][k] == 'T' and new['runtime']['states'][k] == 'U'
                   for k in changed)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        before_title = 'Before: conditional T forced outside' if changed else 'Before: joint O/T pose'
        after_title = 'After: uncertain T withheld; O retained' if changed else 'After: O constraints retained'
        for ax, row, score, arm, title, color in (
            (axes[0], old, sb[fid], arm_b, before_title, '#ff6666'),
            (axes[1], new, sa[fid], arm_a, after_title, '#43e3e4')):
            ax.imshow(rgb); ax.set_xlim(0, w); ax.set_ylim(h, 0)
            ax.set_xticks([]); ax.set_yticks([])
            xy = np.asarray(row['runtime']['keypoint_readout']['rigid_model_corners'])
            wire(ax, xy, color, 'Selected pose')
            for k in changed:
                if 0 <= xy[k, 0] <= w and 0 <= xy[k, 1] <= h:
                    ax.scatter(*xy[k], s=55, color=color, edgecolors='black')
                    ax.annotate(f'{k}:{row["runtime"]["states"][k]}', xy[k],
                                xytext=(5, -6), textcoords='offset points', color=color)
            metric = score['metrics'][arm]
            ax.set_title(title + f'\nT={metric["translation_cm"]:.3f} cm, R={metric["rotation_deg"]:.3f} deg')
        old_m, new_m = sb[fid]['metrics'][arm_b], sa[fid]['metrics'][arm_a]
        caption = (f'{fid} | changed T IDs: {changed} | BI={sa[fid]["BI_count"]}, '
                   f'retained OT={sa[fid]["OT_count"]}\n'
                   'The interval is a CAL maximum localization-error disk, not an occluder segmentation. '
                   'U means abstention, not observed visibility.\n'
                   'S remains pose projection only. Reused exploratory images; annotation-derived C2 reference.')
        fig.text(.02, .025, caption, fontsize=8)
        fig.subplots_adjust(left=.01, right=.995, bottom=.18, top=.86, wspace=.03)
        stem = fid.replace(':', '__')
        for ext in ('png', 'svg'):
            fig.savefig(a.output / f'{stem}.{ext}', dpi=170, bbox_inches='tight')
        plt.close(fig)
        records.append({'id': fid, 'changed_T_ids': changed, 'before_metrics': old_m,
                        'after_metrics': new_m, 'caption': caption,
                        'image_sha256': new['image_sha256'], 'png': stem+'.png',
                        'png_sha256': C.sha(a.output / (stem+'.png')),
                        'svg_sha256': C.sha(a.output / (stem+'.svg'))})
    index = '<meta charset="utf-8"><title>Actual T constraint repair</title><h1>Actual T constraint repair</h1>'
    index += '<p>Every primary frame whose previous pose changed with O/T constraints. Includes CAL and TEST, gains and losses.</p>'
    for r in records:
        index += '<h2>'+html.escape(r['id'])+'</h2><img style="width:100%" src="'+r['png']+'"><pre>'+html.escape(r['caption'])+'</pre>'
    (a.output / 'index.html').write_text(index, encoding='utf-8')
    shutil.copyfile(__file__, a.output / 'renderer_snapshot.py')
    C.write(a.output / 'FIGURE_RECEIPT.json', {
        'schema': 'POSTSEAL_ACTUAL_INTERVAL_EFFECT_FIGURES_V4_2',
        'before_prediction_seal_sha256': C.sha(a.before / 'PREDICTION_SEAL.json'),
        'after_prediction_seal_sha256': C.sha(a.after / 'PREDICTION_SEAL.json'),
        'before_score_sha256': C.sha(a.before / 'POSTSEAL_SCORING.json'),
        'after_score_sha256': C.sha(a.after / 'POSTSEAL_SCORING.json'),
        'selection': 'all primary frames where previous OT constraints changed the pose; no outcome filtering',
        'no_fitting_or_tuning': True, 'renderer_sha256': C.sha(__file__), 'records': records})


if __name__ == '__main__':
    main()
