"""Plot admitted saved SOURCE scalar outputs only; no models or original pixels."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).absolute().parent.parent
OUT = Path(__file__).absolute().parent
ROW = R / 'V244_OWN_BRANCH_DIAGNOSTIC_A/actual_diagnosis_v1/ALL7617_OWN_BRANCH_ROWS.csv'
PAIR = OUT / 'v235_pair_replay_v1/SUMMARY.json'


def main():
    # The row binding is taken only from the exact admitted COMPLETE, never a path walk.
    c = R / 'V244_OWN_BRANCH_DIAGNOSTIC_A/actual_diagnosis_v1/COMPLETE.json'
    cb = c.read_bytes()
    assert hashlib.sha256(cb).hexdigest() == '3df1dbbe5aa6a199b7e4e35c77b1b88c28b3eb92a99b2f635925078c41222293'
    complete = json.loads(cb)
    rb = ROW.read_bytes()
    assert hashlib.sha256(rb).hexdigest() == complete['rows_CSV']['sha256']
    pb = PAIR.read_bytes()
    assert hashlib.sha256(pb).hexdigest() == '256b73ea650d5d23de02ae5f35c3464a4381177362384a8b6768552e3818a7db'
    rows = list(csv.DictReader(rb.decode().splitlines()))
    selected = [r for r in rows if r['family'] == '1' and r['target'] == '1']
    pairs = json.loads(pb)['family_metrics']
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.1), dpi=160)
    for role, color, marker in [('FIT', '#277da8', 'o'), ('DEV', '#ce4c40', 'x')]:
        rr = [r for r in selected if r['role'] == role]
        ax[0].scatter([float(r['selected_own_base_logit']) for r in rr],
                      [float(r['selected_own_residual_logit']) for r in rr],
                      s=22, color=color, marker=marker, alpha=.65,
                      label=f'SOURCE {role} occluded ({len(rr)})')
    lo, hi = -15, 15
    ax[0].plot([lo, hi], [hi, lo], color='#777777', linestyle='--', linewidth=1,
               label='Total = 0 decision boundary')
    ax[0].axhline(0, color='#aaaaaa', linewidth=.6)
    ax[0].axvline(0, color='#aaaaaa', linewidth=.6)
    ax[0].set(xlim=(lo, hi), ylim=(-12, 15), xlabel='Own base logit',
              ylabel='Own context residual logit', title='A. V240: context pushes 56/57 DEV E toward visible')
    ax[0].legend(loc='upper left', fontsize=7.5)
    ax[0].text(.03, .04, 'Target E: positive total is correct\nSame frozen TREAT checkpoint; no CONTROL substitution',
               transform=ax[0].transAxes, fontsize=7.5)
    labels = ['COVER: E', 'SHAM: D', 'Both correct']
    x = np.arange(3)
    for offset, key, label, color in [(-.18, 'SYN_MAIN', 'Synthetic FIT pairs (4537)', '#6e9d76'),
                                     (.18, 'YCB_ASSOCIATED_R_AUX', 'Public YCB FIT pairs (830)', '#aa854c')]:
        item = pairs[key]
        vals = [100 * item[k] for k in ('COVER_E_rate', 'SHAM_D_rate', 'paired_crossing_rate')]
        bars = ax[1].bar(x + offset, vals, .34, color=color, label=label)
        for bar, value in zip(bars, vals):
            ax[1].text(bar.get_x() + bar.get_width()/2, value + 1.0, f'{value:.1f}%',
                       ha='center', va='bottom', fontsize=8)
    ax[1].set(xticks=x, xticklabels=labels, ylim=(0, 105), ylabel='Correct pairs / all successful pairs (%)',
              title='B. V235: old constructed training pairs are mostly learned')
    ax[1].legend(loc='lower left', fontsize=8)
    ax[1].grid(axis='y', alpha=.2)
    fig.suptitle('Frozen-head diagnosis: learned constructed cues do not establish natural-scene transfer', fontsize=12)
    fig.text(.5, .012, 'SOURCE only; FIT pairs are not held-out. DEV is exposed/reused. No new training, private evaluation or PnP.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .95))
    for suffix in ('png', 'svg'):
        p = OUT / ('FROZEN_BEHAVIOR_DIAGNOSIS.' + suffix)
        assert not p.exists()
        fig.savefig(p)
    plt.close(fig)
    inputs = {'own_branch_rows': complete['rows_CSV'], 'pair_summary_sha256': hashlib.sha256(pb).hexdigest(),
              'scatter_population': 'all YCB targetE: SOURCE FIT283 and exposed DEV57',
              'no_score_based_subset_selection': True, 'scatter_points': selected,
              'pair_metrics': pairs, 'new_model_training_or_original_pixels': 0}
    with (OUT / 'PLOT_INPUTS.json').open('x') as f:
        json.dump(inputs, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'completed': True, 'PASS': True, 'png': str(OUT / 'FROZEN_BEHAVIOR_DIAGNOSIS.png')}))


if __name__ == '__main__':
    main()
