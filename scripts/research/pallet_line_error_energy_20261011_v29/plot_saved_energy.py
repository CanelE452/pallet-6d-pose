"""All saved certified-POS lines, with matched weighted error definitions."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = ('GEOMETRY_ONLY', 'IMAGE_NO_ROLE', 'IMAGE_ROLE')
COLORS = ('#3266ad', '#df7c20', '#288548')


def binding(p):
    b = p.read_bytes()
    return dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def save(p, r):
    with p.open('x') as f:
        json.dump(r, f, indent=2)
        f.write('\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    a.output.mkdir()
    line_path, summary_path = a.root/'LINE_ROWS.jsonl.gz', a.root/'SUMMARY.json'
    frozen = {str(x): binding(x) for x in (line_path, summary_path, Path(__file__))}
    save(a.output/'PROTOCOL.json', dict(inputs=frozen, expected_lines=1194,
        selected_source_CAL_POS_normal_component_only=True, matched_weighted_definitions=True,
        x_axis='symlog with linear interval below 1e-12 px; zero values preserved',
        new_line_fit=0, new_PnP=0, new_projection=0))
    with gzip.open(line_path, 'rt') as f:
        rows = [json.loads(s) for s in f if s.strip()]
    rows = [r for r in rows if r['source_certified_all_POS']]
    summary = json.loads(summary_path.read_text())
    assert len(rows) == 1194
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(top=.86, bottom=.15, left=.07, right=.98, hspace=.46, wspace=.29)
    plotted = {}
    for j, (arm, color) in enumerate(zip(ARMS, COLORS)):
        rr = [r for r in rows if r['arm'] == arm]
        g = summary['groups'][arm]
        x = [r['TLS_self_RMS_weighted_px'] for r in rr]
        y = [r['weighted_actual_normal_RMS_px'] for r in rr]
        ax = axes[0, j]
        ax.scatter(x, y, s=15, alpha=.65, color=color, edgecolors='none')
        ax.plot([1e-5, 10], [1e-5, 10], color='#777777', linestyle='--', linewidth=1)
        ax.set_xscale('symlog', linthresh=1e-12)
        ax.set_yscale('log')
        ax.set_xlim(-1e-13, 10)
        ax.set_ylim(1e-5, 10)
        ax.set_xticks([0, 1e-10, 1e-6, .01, 1])
        ax.set_xticklabels(['0', '1e-10', '1e-6', '.01', '1'])
        ax.set_xlabel('Weighted TLS self RMS (px)')
        ax.set_ylabel('Weighted true normal-error RMS (px)')
        ax.set_title(arm+' | '+str(len(rr))+' selected POS lines', fontsize=11)
        ax.grid(alpha=.18)
        z = sorted(r['mean_component_fraction'] for r in rr if r['mean_component_fraction'] is not None)
        ax = axes[1, j]
        ax.step(z, [(i+1)/len(z) for i in range(len(z))], where='post', color=color)
        median = g['mean_component_fraction']['median']
        aggregate = g['aggregate_equal_line_mean_energy_fraction']
        ax.axvline(median, color=color, linestyle='--', linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel('Mean component / total error energy per line')
        ax.set_ylabel('Fraction of selected POS lines')
        ax.set_title('Median '+format(median, '.3f')+' | aggregate energy '+format(aggregate, '.1%'), fontsize=11)
        ax.grid(alpha=.18)
        plotted[arm] = dict(lines=len(rr), zero_self_RMS=sum(v == 0 for v in x), all_rows_plotted=True)
    fig.suptitle('Source CAL: straight fitted supports can still be at the wrong location', fontsize=16, y=.955)
    fig.text(.07, .9, 'Fixed saved normal; 1,194 certified-POS lines from 128 source scenes × 3 arms. No line refit or pose experiment.', fontsize=11)
    fig.text(.07, .055, 'Top: all points, with a dashed equal-error guide. Bottom: line-level CDF; dashed vertical line is its median.\nAggregate energy is sum(mean²) / sum(MSE), with equal line weight; it is not the median or query-error correlation.\nThe same CAL selected these supports and calibrated uncertainty. This is not independent coverage or real-image pose evidence.', fontsize=10)
    files = {}
    for suffix in ('png', 'svg'):
        dest = a.output/('ALL1194_LINE_ERROR_ENERGY.'+suffix)
        fig.savefig(dest, dpi=170)
        files[dest.name] = binding(dest)
    plt.close(fig)
    assert frozen == {str(x): binding(x) for x in (line_path, summary_path, Path(__file__))}
    save(a.output/'CHECKS.json', dict(passed=True, all1194_rows_plotted=True, groups=plotted,
        weighted_self_and_actual_RMS_matched=True, files=files, inputs_unchanged=True,
        no_error_fraction_undefined_lines_plotted_as_zero=True))
    print(json.dumps(dict(passed=True, plotted_POS_lines=1194, files=files)))


if __name__ == '__main__': main()
