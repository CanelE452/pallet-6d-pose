"""Metadata-only layout correction: raise panel headroom so legends hide no counts.
Original V1 plot/source/receipt preserved. Reuse independently audited counts.
"""
from pathlib import Path
import hashlib,json
H=Path(__file__).absolute().parent
def bind(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def write(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
def plot_v2(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig = plt.figure(figsize=(15, 8.4), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.12])
    ax = fig.add_subplot(gs[0, 0])
    r = result['native8']
    x = np.arange(8)
    known = [a['synthetic_supported'] for a in r]
    u = [a['synthetic_UNKNOWN'] for a in r]
    ax.bar(x, known, color='#286aa6', label='Protocol supported')
    ax.bar(x, u, bottom=known, color='#d5d8dc', label='UNKNOWN/failure retained')
    for i, n in enumerate(known):
        ax.text(i, n + 3, str(n), ha='center', fontsize=10)
    ax.set(xticks=x, xlabel='Requested native slot', ylabel='Attempts', ylim=(0, 165), title='Synthetic TRAIN: 97 / 1024 supported triplets (128 attempts per slot)')
    ax.legend(loc='upper right', fontsize=9)
    ax = fig.add_subplot(gs[0, 1])
    known = [a['real_supported'] for a in r]
    u = [a['real_UNKNOWN_or_failure'] for a in r]
    ax.bar(x, known, color='#c15b1b', label='Weak controlled pairs')
    ax.bar(x, u, bottom=known, color='#d5d8dc', label='UNKNOWN/failure retained')
    for i, a in enumerate(r):
        ax.text(i, a['real_attempts'] + 0.35, str(known[i]) + '/' + str(a['real_attempts']), ha='center', fontsize=10)
    ax.set(xticks=x, xlabel='Requested native slot', ylabel='Attempts', ylim=(0, 22), title='Public real TRAIN: 5 / 66 supported (native0, two frames)')
    ax.legend(loc='upper right', fontsize=9)
    ax = fig.add_subplot(gs[1, :])
    cells = result['factor_cells']
    x = np.arange(8)
    known = [a['supported'] for a in cells]
    u = [a['UNKNOWN'] for a in cells]
    labels = [('Polymer' if a['material'] == 'DARK_POLYMER' else 'Wood') + ' / ' + ('Rect' if a['aspect'] == 'RECTANGULAR' else 'Square') + '\n' + ('Board-block' if a['geometry_family'] == 'BOARD_BLOCK_RUNNER' else 'Rib-leg') for a in cells]
    ax.bar(x, known, color='#286aa6')
    ax.bar(x, u, bottom=known, color='#d5d8dc')
    for i, n in enumerate(known):
        ax.text(i, n + 3, str(n) + '/128', ha='center', fontsize=10)
    ax.set(xticks=x, xticklabels=labels, ylabel='Attempts', ylim=(0, 145), title='All eight material × aspect × geometry cells retained; geometry style is separate from material/aspect')
    for a in fig.axes:
        a.spines[['top', 'right']].set_visible(False)
        a.grid(axis='y', alpha=0.16)
        a.set_axisbelow(True)
    fig.suptitle('V188 source-control coverage — data proof counts, not model performance', fontsize=17, y=0.985)
    fig.text(0.5, 0.015, 'Fixed q/L and native roles. UNKNOWN and input-U retained. 11 real cases fail conservative raw-tap coverage inside the crop extent; Q0057 is separately nonlocal.\nNo training admission, new fit/forward/evaluation, model adoption or full-goal qualification.', ha='center', va='bottom', fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.955))
    path = H / 'SOURCE_CONTROL_COVERAGE_V2.png'
    fig.savefig(path)
    plt.close(fig)
    write(H / 'PLOT_COMPLETE_V2.json', dict(completed=True, PASS=True, plot=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()), source_review=bind(H / 'REVIEW.json'), generated_from_table_metadata_only=True, actual_tool_visual_QA='PENDING_VIEW_IMAGE', no_source_pixels_read=True))

if __name__=='__main__':
    plot_v2(json.loads((H/'REVIEW.json').read_text()))
