"""Plot the user's named Stage1 family without replacing the original figure."""
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from . import common as C


def run():
    start = time.monotonic()
    source = C.DOC/'REQUESTED_PRIMARY_PAIRED.json'
    packet = C.read(source)
    rows = packet['results']; assert len(rows) == 8
    target = C.DOC/'FIGURES/09_requested_stage1_paired_effects.png'
    assert not target.exists()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), sharey=True)
    labels = ['GV - G', 'GVD - GV', 'D JOINT - CONTROL (Stage1)', 'GVD - OFFICIAL']
    for ax, metric, title, color in zip(axes, ['wd_confusion','pose_success'],
        ['W/D confusion (lower is better)', '5 cm / 5 deg success (higher is better)'],
        ['#b85450', '#326991']):
        selected = [r for r in rows if r['metric'] == metric]
        x = np.array([r['effect_pp'] for r in selected])
        low = np.array([r['CI95'][0] for r in selected])
        high = np.array([r['CI95'][1] for r in selected])
        ax.errorbar(x, np.arange(4), xerr=np.stack([x-low, high-x]), fmt='o',
                    color=color, ecolor=color, capsize=3)
        ax.axvline(0, color='#444444', linewidth=.8, linestyle='--')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('After - before (percentage points)')
        ax.set_yticks(np.arange(4), labels)
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle('Requested contrasts: 9-session paired bootstrap, 95% CI', fontsize=12)
    fig.text(.5, .02, 'Post-execution protocol alignment; original SubPix family retained. Exploratory development data.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .93))
    fig.savefig(target, dpi=300); plt.close(fig)
    C.write(C.DOC/'REQUESTED_PRIMARY_FIGURE_RECEIPT.json', dict(
        input=C.binding(source), output=C.binding(target), code=C.binding(__file__),
        DPI=300, language='English', new_fits=0, new_PnP_calls=0,
        wall_seconds=time.monotonic()-start, original_figures_unchanged=True))
    print('REQUESTED_PRIMARY_FIGURE', target.name)


if __name__ == '__main__': run()
