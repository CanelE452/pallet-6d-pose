from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE / 'PLOT_NUMBERS.json').read_text())['models']
fig, axes = plt.subplots(1, 2, figsize=(13.7, 5.9))
x, width = np.arange(len(rows)), .34
for delta, key, color, label in [
    (-width/2, 'D_recall', '#187c77', 'Visible retention (D recall)'),
    (width/2, 'E_response_precision', '#cd7737', 'Hidden response precision'),
]:
    bars = axes[0].bar(x+delta, [r[key]*100 for r in rows], width=width,
                       color=color, label=label)
    axes[0].bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
axes[0].set_title('Same query / native visibility reference\nD=1,147, external E=90; all abstentions retained', fontsize=11)
axes[0].set_ylabel('Percent')
axes[0].legend(loc='upper right', fontsize=8)
bars = axes[1].bar(x, [r['near376_recall']*100 for r in rows],
                   color=['#8a99a6', '#5876a1', '#baa788', '#187c77', '#6a8a67'])
axes[1].bar_label(bars, labels=[f"{r['near376_visible']}/376\n{r['near376_recall']*100:.1f}%" for r in rows],
                  padding=5, fontsize=9)
axes[1].set_title('Manual reference error <= 3 px\n376 visible points; existing reference proxy', fontsize=11)
axes[1].set_ylabel('Visible retention (%)')
for ax in axes:
    ax.set_xticks(x, [r['model'] for r in rows])
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
fig.suptitle('Current223: direct-owner models still reject many visible keypoints', y=.97, fontsize=14)
fig.subplots_adjust(left=.065, right=.985, bottom=.20, top=.79, wspace=.24)
fig.text(.5, .07, 'F/BG -> hidden is a diagnostic response, not physical external-occlusion proof.', ha='center', fontsize=9)
fig.text(.5, .03, 'No model selected. Private data used only for evaluation. Legacy primary abstention: 14.3%.', ha='center', fontsize=9)
for suffix in ('png', 'pdf'):
    fig.savefig(HERE / f'CURRENT223_DIRECT_OWNER_DIAGNOSTIC_V2.{suffix}', dpi=160)
plt.close(fig)
