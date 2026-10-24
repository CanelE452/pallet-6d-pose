"""Source-TRAIN metadata figure; no RGB, array, model, or score loading."""
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import os
import stat

HERE = Path(__file__).absolute().parent
OUT = HERE / 'scientific_readout_v1'
SOURCE = HERE / 'target_correspondence_b/ALL109_TRAIN_FINITE_END_QUERY_CORRESPONDENCE.csv'
EXPECTED_SHA = 'f7f8fd3c854168f47f6a3440c56f828d9124a8fcd35c2ec21238077b208f8812'


def read_ordinary(path):
    path = Path(path)
    assert path.is_absolute() and '..' not in path.parts
    assert all('v112' not in part.casefold() for part in path.parts)
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        final = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    assert stat.S_ISREG(os.fstat(final).st_mode)
    with os.fdopen(final, 'rb') as stream:
        return stream.read()


def main():
    raw = read_ordinary(SOURCE)
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    assert len(rows) == 109 and len({r['attempt_id'] for r in rows}) == 109
    domains = ['SYNTH_FINITE_END', 'REAL_SEMANTIC_END_WEAK']
    stats = {}
    for domain, expected in zip(domains, (82, 27)):
        group = [r for r in rows if r['domain'] == domain]
        assert len(group) == expected
        distances, centers = [], []
        inside = 0
        for row in group:
            q = (float(row['original_q_x']), float(row['original_q_y']))
            scale = 96.0 / float(row['original_L'])
            boxes = json.loads(row['candidate_boxes_fullRGB_json'])
            projected = [[47.5 + (v - q[i % 2]) * scale for i, v in enumerate(b)] for b in boxes]
            saved = json.loads(row['candidate_boxes_crop96_json'])
            assert len(saved) == len(projected) > 0
            assert all(abs(a - b) < 1e-10 for x, y in zip(saved, projected) for a, b in zip(x, y))
            d = min(math.hypot(max(b[0] - 47.5, 0.0, 47.5 - b[2]),
                               max(b[1] - 47.5, 0.0, 47.5 - b[3])) for b in projected)
            c = min(math.hypot((b[0] + b[2]) / 2 - 47.5,
                               (b[1] + b[3]) / 2 - 47.5) for b in projected)
            assert abs(d - float(row['min_query_distance_to_candidate_region_crop_px'])) < 1e-10
            assert abs(c - float(row['min_candidate_center_distance_crop_px'])) < 1e-10
            assert (d == 0) == (row['query_inside_any_candidate'] == 'True')
            assert all(0 <= b[0] <= b[2] <= 95 and 0 <= b[1] <= b[3] <= 95 for b in projected)
            inside += int(d == 0)
            distances.append(d)
            centers.append(c)
        ordered = sorted(distances)
        n = len(ordered)
        median = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
        stats[domain] = dict(triplets=n, query_inside_weak_enclosure=inside,
                             unique_source_groups=len({r['source_group'] for r in group}),
                             region_distance_median=median, region_distance_max=max(distances))
    assert stats[domains[0]]['query_inside_weak_enclosure'] == 3
    assert stats[domains[1]]['query_inside_weak_enclosure'] == 5
    OUT.mkdir(exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(OUT / 'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={'width_ratios': [1, 1, 1.1]})
    colors = ['#1467a6', '#bc4f13']
    names = ['Synthetic finite end', 'Public-real weak end']
    for ax, domain, color, title in zip(axes[:2], domains, colors, names):
        group = [r for r in rows if r['domain'] == domain]
        for row in group:
            for b in json.loads(row['candidate_boxes_crop96_json']):
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=False, edgecolor=color, alpha=.35, linewidth=.8))
                ax.plot((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, '.', color=color, alpha=.55, markersize=3)
        s = stats[domain]
        ax.plot(47.5, 47.5, '+', color='#111111', markersize=15, markeredgewidth=2.3, label='Original query')
        ax.set(xlim=(-.5, 95.5), ylim=(95.5, -.5), aspect='equal', xlabel='Original crop x', ylabel='Original crop y')
        ax.set_title(f'{title}\n{s["triplets"]} triplets / {s["unique_source_groups"]} source groups\nquery inside enclosure: {s["query_inside_weak_enclosure"]}/{s["triplets"]}', fontsize=11)
        ax.grid(alpha=.15)
    for domain, color, name in zip(domains, colors, names):
        values = sorted(float(r['min_query_distance_to_candidate_region_crop_px']) for r in rows if r['domain'] == domain)
        axes[2].step([0] + values, [0] + [(i+1)/len(values) for i in range(len(values))], where='post', color=color,
                     label=f'{name}\nmedian {stats[domain]["region_distance_median"]:.2f} px')
    axes[2].set(xlabel='Query to nearest weak enclosure (crop px)', ylabel='Fraction of TRAIN triplets',
                xlim=(0, 36), ylim=(0, 1.02), title='Distance to region, not to its center')
    axes[2].grid(alpha=.2)
    axes[2].legend(fontsize=9, loc='lower right')
    fig.suptitle('The supervised weak end can be away from the original query', fontsize=15)
    fig.text(.5, .025, 'TRAIN metadata only. Boxes have different source definitions; sizes are not comparable physical areas.\nDistances are not GT coordinate errors. No model predictions, private evaluation, or RGB pixels were read.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .12, 1, .91))
    outputs = []
    for suffix in ('png', 'svg'):
        path = OUT / ('TRAIN_QUERY_AND_WEAK_END.' + suffix)
        fig.savefig(path, dpi=150, facecolor='white')
        data = read_ordinary(path)
        outputs.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    plt.close(fig)
    receipt = dict(completed=True, PASS=True, input=dict(path=str(SOURCE), sha256=EXPECTED_SHA),
                   rows_recomputed=109, raw_to_crop_box_projection_and_region_distance_parity=True,
                   stats=stats, outputs=outputs, RGB_NPZ_model_score_reads=0, actual_tool_image_view=False,
                   distance_is_physical_point_error=False, model_selected=None, goal_solved=False)
    (OUT/'COMPLETE.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
