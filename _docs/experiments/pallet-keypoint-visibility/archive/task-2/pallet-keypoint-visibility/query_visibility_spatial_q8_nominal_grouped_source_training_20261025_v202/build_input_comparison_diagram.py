"""Scientific schematic of the prespecified input comparison; no source data."""
from pathlib import Path
import os
import json
import hashlib

HERE = Path(__file__).absolute().parent
OUT = HERE / 'input_comparison_diagram_v2'


def main():
    scope = json.loads((HERE/'ROOT_PROSPECTIVE_SPATIAL_Q8_NOMINAL_SOURCE_SCOPE.json').read_text())
    assert scope['input_contract']['parameters_each'] == 118359
    assert scope['training']['updates_each'] == 4000
    OUT.mkdir(exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(OUT/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set(xlim=(0, 14), ylim=(0, 5.5))
    ax.axis('off')

    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04',
                                   facecolor=color, edgecolor='#506070', linewidth=1.2))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=10.5)

    def arrow(a, b, label=None):
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', color='#405264', lw=1.5))
        if label:
            ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.12, label, ha='center', fontsize=9)

    ax.text(7, 5.15, 'V202: identical native + mask inputs; only other-point XY differs',
            ha='center', fontsize=16, weight='bold')
    box(.2, 3.4, 2.3, 1.05, 'Original RGB96 + valid\nconstant crop XY\nSAM channels = zero', '#eaf0f5')
    box(3.2, 3.4, 2.7, 1.05, 'Same CNN architecture\n3 x 3 pool -> features64\nseparate trained copies', '#eaf0f5')
    box(.2, 1.65, 2.3, 1.1, 'Original initializer q8\noriginal availability\noriginal native ID', '#e7f2fc')
    box(3.2, 1.65, 2.7, 1.1, 'Other7: (q_j - q) / L\nF64 -> clip[-8,8] -> FP32\nself XY + mask = zero', '#e7f2fc')
    box(7.0, 3.35, 3.5, 1.12, 'C60 control\nimage64 + native8 + mask8\nother XY16 = ZERO', '#fff0de')
    box(7.0, 1.6, 3.5, 1.12, 'C61 spatial context\nimage64 + native8 + mask8\nother XY16 = ACTUAL', '#e4f3ed')
    box(11.2, 3.35, 2.5, 1.12, 'Direct-end logit\nD / E at fixed zero\noriginal input U kept', '#f0edf7')
    box(11.2, 1.6, 2.5, 1.12, 'Direct-end logit\nD / E at fixed zero\noriginal input U kept', '#f0edf7')
    arrow((2.5, 3.92), (3.2, 3.92))
    arrow((2.5, 2.2), (3.2, 2.2))
    arrow((5.9, 3.92), (7.0, 3.92))
    arrow((5.9, 3.55), (7.0, 2.6))
    arrow((5.9, 2.5), (7.0, 3.4))
    arrow((5.9, 2.2), (7.0, 2.2))
    arrow((10.5, 3.92), (11.2, 3.92))
    arrow((10.5, 2.2), (11.2, 2.2))
    ax.text(8.8, 1.15, 'Each: old direct head + residual MLP 96 -> 32 -> 1 (final layer zero initially)',
            fontsize=10, ha='center')
    ax.text(7, .65, 'Both: fresh identical initial states, same TRAIN draws, fixed 4,000 updates.\n'
            'Fit excludes active-only jitter/scale states; original TRAIN/HOLDOUT roles remain unchanged.',
            fontsize=10.5, ha='center')
    ax.text(7, .15, 'Weak finite-end supervision; human native D/E is a separate later evaluation. '
            'No GT end box / pose input, query recentering, new initializer, or PnP.',
            fontsize=9.5, ha='center')
    fig.tight_layout(pad=.5)
    outputs = []
    for suffix in ['png', 'svg']:
        f = OUT/('PRESPECIFIED_INPUT_COMPARISON.'+suffix)
        fig.savefig(f, dpi=150, facecolor='white')
        outputs.append(dict(path=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size))
    plt.close(fig)
    with (OUT/'COMPLETE.json').open('x') as f:
        json.dump(dict(completed=True, PASS=True, outputs=outputs, source_pixel_array_model_or_score_reads=0,
                       architecture_schematic_not_performance_result=True, actual_image_tool_view=False), f, indent=2)
        f.write('\n')
    print(json.dumps(dict(completed=True, outputs=outputs)))


if __name__ == '__main__':
    main()
