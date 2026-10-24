"""Reduce frozen V222 SOURCE outputs and render every SOURCE-DEV triplet."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
S = P / 'query_visibility_expected_part_counterfactual_source_bridge_20261025_v222'
FIT = S / 'paired_source_fit_v1'
OUT = S / 'saved_source_comparison_v2'
ARMS = ('C66CONTINUATION_CONTROL', 'C67PLUS_EXPECTED_PART_CF_WEAK')


def safe(f):
    f = Path(f)
    assert f.is_absolute() and '..' not in f.parts and 'v112' not in str(f).lower()
    assert f.is_relative_to(S)
    for item in (*reversed(f.parents), f):
        assert not item.is_symlink()
    return f


def binding(f):
    data = safe(f).read_bytes()
    return dict(path=str(f), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def read_bound(b):
    assert binding(b['path'])['sha256'] == b['sha256']
    raw = safe(b['path']).read_text()
    return [json.loads(line) for line in raw.splitlines() if line.strip()] if str(b['path']).endswith('.jsonl') else json.loads(raw)


def write(f, data):
    with safe(f).open('x') as out:
        out.write(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    return binding(f)


def metrics(rows, predkey):
    cm = [[0, 0, 0], [0, 0, 0]]
    for r in rows:
        if r['target'] not in (0, 1):
            continue
        pred = r[predkey]
        cm[r['target']][pred if pred in (0, 1) else 2] += 1
    classes = []
    for k in (0, 1):
        n = sum(cm[k]); positive = cm[0][k] + cm[1][k]
        classes.append(dict(class_D0_E1=k, truth=n, correct=cm[k][k],
                            precision=cm[k][k] / positive if positive else None,
                            recall=cm[k][k] / n if n else None))
    known = sum(map(sum, cm))
    return dict(rows=len(rows), known_D_E=known, truth_UNKNOWN=len(rows) - known,
                CM_D_E_rows_D_E_U_columns=cm, classes=classes,
                correct=cm[0][0] + cm[1][1],
                predicted_U_on_known=cm[0][2] + cm[1][2],
                U_rate_on_known=(cm[0][2] + cm[1][2]) / known if known else None)


def triplets(rows):
    groups = defaultdict(dict)
    for r in rows:
        groups[r['parent_row_id']][r['variant']] = r
    summary = dict(parents=len(groups), all_three_correct=0, BEFORE_D=0, COVER_E=0,
                   SHAM_D=0, cover_logit_greater_before=0, cover_logit_greater_sham=0)
    for rr in groups.values():
        b, c, s = [rr[x] for x in ('BEFORE', 'COVER', 'SHAM')]
        bb, cc, ss = [r['prediction_D0_E1_Uminus1'] for r in (b, c, s)]
        summary['BEFORE_D'] += bb == 0; summary['COVER_E'] += cc == 1
        summary['SHAM_D'] += ss == 0; summary['all_three_correct'] += (bb, cc, ss) == (0, 1, 0)
        if all(r['raw_logit'] is not None for r in (b, c, s)):
            summary['cover_logit_greater_before'] += c['raw_logit'] > b['raw_logit']
            summary['cover_logit_greater_sham'] += c['raw_logit'] > s['raw_logit']
    return summary


def paired(a, b, predkey):
    aa = {r['row_id']: r for r in a}; bb = {r['row_id']: r for r in b}
    assert aa.keys() == bb.keys()
    out = dict(rows=len(aa), recovered=0, damaged=0, both_correct=0, both_wrong=0, classes={})
    for k in (0, 1):
        selected = [rid for rid in aa if aa[rid]['target'] == k]
        recover = damage = bc = bw = 0
        for rid in selected:
            assert aa[rid]['target'] == bb[rid]['target']
            ca = aa[rid][predkey] == k; cb = bb[rid][predkey] == k
            recover += not ca and cb; damage += ca and not cb; bc += ca and cb; bw += not ca and not cb
        out['classes'][str(k)] = dict(truth=len(selected), recovered=recover, damaged=damage,
                                     both_correct=bc, both_wrong=bw)
        for key, value in [('recovered', recover), ('damaged', damage), ('both_correct', bc), ('both_wrong', bw)]:
            out[key] += value
    out['net_correct_change'] = out['recovered'] - out['damaged']
    return out


def main():
    complete_binding = binding(FIT / 'COMPLETE.json')
    complete = read_bound(complete_binding)
    assert complete['completed'] and complete['PASS'] and complete['steps_each'] == 1100
    freeze = read_bound(complete['all2_freeze'])
    assert freeze['completed'] and freeze['PASS'] and freeze['selected_arm'] is None
    family_fields = [('CF', 'CF_all144_outputs', 'rows144', 'prediction_D0_E1_Uminus1'),
                     ('NATURAL_WEAK', 'weak57_outputs', 'rows57', 'prediction_D0_E1'),
                     ('STRONG_POINT_TRAIN', 'strong_outputs', 'rows', 'prediction_D0_E1')]
    result = dict(completed=True, kind='FROZEN_SOURCE_ONLY_WEAK_RELATION_COMPARISON',
                  training_complete=complete_binding, all2_freeze=complete['all2_freeze'],
                  families={}, protected_evaluation=False, new_native_point_gold=False,
                  selected_arm=None, adopted=False, task_solved=False)
    rows_by_family = {}
    flat = []
    for family, field, rowkey, predkey in family_fields:
        rows_by_family[family] = {}
        models = {}
        for arm in ARMS:
            rows = read_bound(complete['arms'][arm][field])[rowkey]
            rows_by_family[family][arm] = rows
            models[arm] = {}
            for role in ['ALL', *sorted({r['source_role'] for r in rows})]:
                rr = rows if role == 'ALL' else [r for r in rows if r['source_role'] == role]
                m = metrics(rr, predkey)
                if family == 'CF':
                    m['triplets'] = triplets(rr)
                    m['components'] = sorted({r['component_id'] for r in rr})
                    m['by_native'] = {str(k): metrics([r for r in rr if r['native_slot'] == k], predkey)
                                      for k in sorted({r['native_slot'] for r in rr})}
                models[arm][role] = m
                cm = m['CM_D_E_rows_D_E_U_columns']; d, e = m['classes']
                flat.append(dict(family=family, arm=arm, role=role, rows=m['rows'], D=d['truth'], E=e['truth'],
                                 D_D=cm[0][0], D_E=cm[0][1], D_U=cm[0][2], E_D=cm[1][0], E_E=cm[1][1], E_U=cm[1][2],
                                 D_recall=d['recall'], E_recall=e['recall'], E_precision=e['precision'], U_rate=m['U_rate_on_known']))
        role_pairs = {}
        for role in ['ALL', *sorted({r['source_role'] for r in rows_by_family[family][ARMS[0]]})]:
            aa, bb = [[r for r in rows_by_family[family][arm] if role == 'ALL' or r['source_role'] == role] for arm in ARMS]
            role_pairs[role] = paired(aa, bb, predkey)
        result['families'][family] = dict(models=models, paired=role_pairs)
    attempts_path = S / 'expected_part_counterfactual_pairs_v1/ALL48_ATTEMPTS.json'
    attempts = read_bound(binding(attempts_path))['attempts48']
    geometry = {}
    for a in attempts:
        def center_count(rect):
            x0, y0, x1, y1 = rect
            return sum(x0 <= x <= x1 and y0 <= y <= y1 for x in (47, 48) for y in (47, 48))
        geometry[a['parent_row_id']] = dict(COVER=center_count(a['geometry']['COVER_rectangle']),
                                           SHAM=center_count(a['geometry']['SHAM_rectangle']))
    result['fixed_central_sample_subsets'] = {}
    for arm in ARMS:
        result['fixed_central_sample_subsets'][arm] = {}
        for role in ('SOURCE_FIT', 'SOURCE_DEV'):
            rr = [r for r in rows_by_family['CF'][arm] if r['source_role'] == role]
            cover = [r for r in rr if r['variant'] == 'COVER' and geometry[r['parent_row_id']]['COVER'] == 0]
            sham = [r for r in rr if r['variant'] == 'SHAM' and geometry[r['parent_row_id']]['SHAM'] > 0]
            result['fixed_central_sample_subsets'][arm][role] = {
                'COVER_E_with_central_four_clear': metrics(cover, 'prediction_D0_E1_Uminus1'),
                'SHAM_D_with_central_four_any_blocked': metrics(sham, 'prediction_D0_E1_Uminus1')}
    # All computations above precede output creation; a premature call cannot leave partial output.
    safe(OUT).mkdir(exist_ok=False)
    with (OUT / 'SOURCE_FAMILIES_SEPARATE_CM_PRECISION_RECALL.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    summary = write(OUT / 'SOURCE_COMPARISON.json', result)
    meta = {r['row_id']: r for r in read_bound(binding(S / 'expected_part_counterfactual_pairs_v1/ALL144_VARIANT_ROWS.jsonl'))}
    predictions = {arm: {r['row_id']: r for r in rows_by_family['CF'][arm]} for arm in ARMS}
    devparents = sorted({r['parent_row_id'] for r in rows_by_family['CF'][ARMS[0]] if r['source_role'] == 'SOURCE_DEV'})
    assert len(devparents) == 9
    gallery = ['# All9 SOURCE DEV triplets with frozen C66/C67 outputs', '',
               'GT-derived parent query; finite-part weak target, not native-point gold. Center + and cyan ROI are display annotations only. All training RGBs remain unmarked. Two frame/components; correlated nominal/jitter examples.', '']
    for parent in devparents:
        fig, axes = plt.subplots(1, 3, figsize=(10, 5.2))
        for ax, variant in zip(axes, ('BEFORE', 'COVER', 'SHAM')):
            rid = parent + '__' + variant; row = meta[rid]
            data = safe(row['RGB96']['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == row['RGB96']['sha256']
            with Image.open(safe(row['RGB96']['path'])) as image:
                rgb = np.asarray(image).copy()
            mask = np.zeros(9216, bool)
            for start, length in row['exact_target_union_RLE']:
                mask[start:start + length] = True
            ax.imshow(rgb); ax.contour(mask.reshape(96, 96), levels=[.5], colors=['cyan'], linewidths=.9)
            ax.plot(47.5, 47.5, '+', color='red', markersize=10, markeredgewidth=1.5)
            ax.set_title(variant + (' / weak E' if variant == 'COVER' else ' / weak D'))
            labels = []
            for arm, short in zip(ARMS, ('C66', 'C67')):
                r = predictions[arm][rid]
                labels.append(f"{short}: {'E' if r['prediction_D0_E1_Uminus1'] == 1 else 'D'}  z={r['raw_logit']:+.3f}")
            ax.set_xlabel('\n'.join(labels), fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(parent + ' / SOURCE_DEV / GT-derived training query', fontsize=12)
        fig.text(.5, .015, 'Same q/L/native/q8/valid; cyan = observed finite-part weak ROI, red + = crop center. No primary native-point gold.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .18, 1, .93))
        name = 'SOURCE_DEV_' + parent + '.png'; fig.savefig(OUT / name, dpi=150); plt.close(fig)
        gallery += ['## ' + parent, '', '![](' + name + ')', '']
    (OUT / 'PORTABLE_GALLERY_KO.md').write_text('\n'.join(gallery))
    write(OUT / 'COMPLETE.json', dict(completed=True, PASS=True, summary=summary,
                                     SOURCE_DEV_triplet_figures=9, new_model_forwards=0,
                                     saved_source_only=True, protected_evaluation=False))
    print(json.dumps({'completed': True, 'SOURCE_DEV_CF': {arm: result['families']['CF']['models'][arm]['SOURCE_DEV'] for arm in ARMS},
                      'SOURCE_DEV_natural': {arm: result['families']['NATURAL_WEAK']['models'][arm]['SOURCE_DEV'] for arm in ARMS},
                      'geometry_subsets': {arm: result['fixed_central_sample_subsets'][arm]['SOURCE_DEV'] for arm in ARMS}}))


if __name__ == '__main__':
    main()
