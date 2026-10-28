#!/usr/bin/env python3
"""Plot frozen aggregate results; no training, inference, or raw annotations."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path('/home/minjae/Documents/pallet-research/dope-r2-count-proportional-per-type-20261028-v2')
OUT = Path(__file__).resolve().parent
INPUTS = {
    'aggregate': ROOT / 'evaluation/final/SEED_AGGREGATE_SUMMARY.csv',
    'controls': ROOT / 'evaluation/final/CONTEXT_CONTROL_SUMMARY.csv',
    'paired_ci': ROOT / 'evaluation/final/PAIRED_CI_SEED_MEAN.csv',
    'fixed_plan': ROOT / 'FIXED_PLAN.json',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target_png = OUT / 'Count_proportional_R2_three_metrics.png'
    target_pdf = OUT / 'Count_proportional_R2_three_metrics.pdf'
    qa_path = OUT / 'PLOT_QA.json'
    if any(p.exists() for p in [target_png, target_pdf, qa_path]):
        raise SystemExit('STOP: output exists; no overwrite allowed')
    before = {key: digest(path) for key, path in INPUTS.items()}
    with INPUTS['aggregate'].open() as h:
        rows = list(csv.DictReader(h))
    with INPUTS['controls'].open() as h:
        control_rows = list(csv.DictReader(h))
    with INPUTS['paired_ci'].open() as h:
        ci_rows = list(csv.DictReader(h))
    plan = json.loads(INPUTS['fixed_plan'].read_text())
    kinds = ['PLASTIC', 'WOOD', 'GREEN']
    conditions = ['A', 'B', 'C', 'D', 'E']
    assert len(rows) == 15 and len(control_rows) == 3 and len(ci_rows) == 135
    assert plan['training_seeds'] == [42, 43, 44]
    assert plan['evaluation']['primary_total'] == 372
    data = {(r['kind'], r['condition']): r for r in rows}
    controls = {r['kind']: r for r in control_rows}
    assert set(data) == {(k, c) for k in kinds for c in conditions}
    assert set(controls) == set(kinds)
    for row in rows:
        assert row['n_training_seeds'] == '3'
        assert row['training_seeds'] == '42;43;44'
        assert row['training_seed_SD_ddof'] == '1'
    x = np.asarray([plan['synthetic_percent'][c] for c in conditions], dtype=float)
    assert list(x) == [20., 50., 100., 200., 400.]
    for kind in kinds:
        expected_ratios = [100 * plan['synthetic_unique_counts'][kind][c] /
                           plan['real_unique_counts'][kind] for c in conditions]
        assert np.allclose(x, expected_ratios, atol=1e-12)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.titlesize': 13, 'axes.labelsize': 10.5,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.right': False, 'axes.spines.top': False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 6.3))
    fig.subplots_adjust(left=.058, right=.985, bottom=.315, top=.755, wspace=.26)
    colors = {'PLASTIC': '#0072B2', 'WOOD': '#D55E00', 'GREEN': '#009E73'}
    labels = {'PLASTIC': 'Rectangular plastic', 'WOOD': 'Small wooden', 'GREEN': 'Square plastic'}
    markers = {'PLASTIC': 'o', 'WOOD': 's', 'GREEN': '^'}
    metrics = [
        ('PCK10', 'PCK@10 px', (30, 76)),
        ('coverage', 'Keypoint coverage', (76, 102)),
        ('conditional_precision10', 'Conditional precision@10 px', (32, 78)),
    ]
    evidence = []
    for ax, (metric, title, ylim) in zip(axes, metrics):
        for kind in kinds:
            # CSV stores fractions; values and sample SD are converted to percent.
            mean = np.asarray([float(data[kind, c][metric + '_seed_mean']) for c in conditions])
            sd = np.asarray([float(data[kind, c][metric + '_training_seed_sample_SD']) for c in conditions])
            control = float(controls[kind][metric])
            assert np.isfinite(mean).all() and np.isfinite(sd).all()
            assert (mean >= 0).all() and (mean <= 1).all() and (sd >= 0).all()
            assert 0 <= control <= 1
            plotted = mean * 100
            errors = sd * 100
            assert np.allclose(plotted / 100, mean, rtol=0, atol=1e-15)
            assert np.allclose(errors / 100, sd, rtol=0, atol=1e-15)
            ax.errorbar(x, plotted, yerr=errors, color=colors[kind],
                        marker=markers[kind], markersize=5.5, linewidth=1.9,
                        elinewidth=1.2, capsize=3, capthick=1.2, zorder=3)
            ax.axhline(control * 100, color=colors[kind], linestyle=(0, (4, 3)),
                       linewidth=1.4, alpha=.75, zorder=2)
            evidence.append({
                'kind': kind, 'metric': metric, 'conditions': conditions,
                'x_synthetic_percent': x.tolist(), 'plotted_mean_percent': plotted.tolist(),
                'plotted_sample_SD_percentage_points': errors.tolist(),
                'context_control_percent': control * 100,
            })
        ax.set_xscale('log', base=2)
        ax.set_xticks(x, [f'{v:g}%' for v in x])
        ax.minorticks_off()
        ax.set_xlim(16, 500)
        ax.set_ylim(*ylim)
        ax.grid(axis='y', linewidth=.65, color='#DDE2E8', zorder=0)
        ax.set_title(title, pad=10, fontweight='semibold')
        ax.set_ylabel('Percent (%)')
        ax.set_xlabel('Synthetic / real unique-count ratio', labelpad=10)
    fig.suptitle('Count-proportional R2 sampling: all five conditions',
                 x=.058, y=.965, ha='left', fontsize=17, fontweight='semibold')
    fig.text(.058, .905, 'Same frozen R1 teacher and pseudo-labels; 828 R2 updates per run; batch size 16',
             fontsize=11.2, color='#3D4651')
    legend = [Line2D([0], [0], color=colors[k], marker=markers[k], linewidth=1.9,
                     markersize=5.5, label=labels[k]) for k in kinds]
    legend += [Line2D([0], [0], color='#555555', linestyle=(0, (4, 3)), linewidth=1.5,
                      label='Historical synthetic-only R2 control (context)')]
    fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.052, .865), ncol=4,
               frameon=False, fontsize=10.2, handlelength=2.4, columnspacing=1.7)
    footer = [
        'Points and error bars: mean +/- sample SD (ddof=1), R2 seeds 42/43/44 conditional on the same R1 and fixed pseudo-label sets.',
        'Primary evaluation: 372 frames (plastic 128, wood 125, green 119). Native-pixel PCK; missing predictions are failures.',
        'Dashed controls are historical context: the synthetic-only control has a different R1 training history; no control seed SD is estimated.',
        'Wood combines 1280x720 and 640x480 frames and shares capture recordings with training. Paired-frame 95% CIs are in separate tables.',
    ]
    for i, line in enumerate(footer):
        fig.text(.058, .205 - i * .034, line, fontsize=9.2, color='#3D4651', ha='left')
    fig.savefig(target_png, dpi=300, facecolor='white')
    fig.savefig(target_pdf, facecolor='white', metadata={
        'Title': 'Count-proportional R2 sampling: all five conditions',
        'Author': 'Pallet DOPE experiment analysis',
        'Subject': 'Fixed three-seed R2 means and sample SD; historical control context',
    })
    plt.close(fig)
    after = {key: digest(path) for key, path in INPUTS.items()}
    assert before == after
    qa = {
        'schema': 'final_proportional_plot_qa_v1',
        'utc': datetime.now(timezone.utc).isoformat(),
        'input_files': {k: {'path': str(v), 'sha256': before[k]} for k, v in INPUTS.items()},
        'input_hashes_unchanged': before == after,
        'series_count': len(evidence), 'conditions_per_series': 5,
        'training_seed_count': 3, 'training_seeds': [42, 43, 44], 'SD_ddof': 1,
        'correct_percent_conversion_checked': True,
        'x_ratio_equals_unique_count_ratio_checked': True,
        'paired_CI_rows_read_only': len(ci_rows),
        'paired_CI_not_plotted_as_seed_SD': True,
        'all_conditions_plotted': True,
        'prediction_or_training_run': False,
        'raw_reference_files_opened': 0,
        'primary_repo_Git_commands': 0,
        'plotted_series': evidence,
        'outputs': {
            target.name: {'path': str(target), 'sha256': digest(target), 'bytes': target.stat().st_size}
            for target in [target_png, target_pdf]
        },
        'script_sha256': digest(Path(__file__)),
        'PNG_visual_review': 'pending', 'PDF_render_review': 'pending',
    }
    assert len(evidence) == 9
    qa_path.write_text(json.dumps(qa, indent=2) + '\n')
    print(json.dumps({'png': str(target_png), 'pdf': str(target_pdf),
                      'qa': str(qa_path), 'series_count': len(evidence)}))


if __name__ == '__main__':
    main()
