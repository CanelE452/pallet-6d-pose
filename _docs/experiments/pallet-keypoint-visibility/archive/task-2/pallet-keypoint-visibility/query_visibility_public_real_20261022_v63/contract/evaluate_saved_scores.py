"""Evaluate source-frozen saved query scores; imports no model/training/PnP code.

Example: python evaluate_saved_scores.py --scores ../runs/real.csv \
  --freeze ../runs/MODEL_FREEZE.json --score-column p_visible --model v63_spatial

The negative prediction is ABSENT_NONVISIBLE, never asserted external occlusion.
Current human D/E/S truth and initial-reference error are evaluation-only inputs.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D, E, S = 'DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED', 'SELF_OCCLUDED'
TASKS = {'D_vs_E': [D, E], 'D_vs_S': [D, S], 'D_vs_ES': [D, E, S]}
GROUPS = ['WOOD_PRIMARY105', 'SQUARE118']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readcsv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def writecsv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def yes(r, key):
    return r[key] == 'True'


def ratio(a, b):
    return a / b if b else None


def call(row, model, task, probabilities, threshold, rules=None):
    if model == 'v61':
        return 'VISIBLE' if yes(row, 'present_at_fixed_0_5') else 'ABSENT_NONVISIBLE'
    if model == 'legacy':
        s = row['legacy_stored_state']
        if s in ('B', 'I'):
            return 'VISIBLE'
        if s == 'O':
            return 'ABSENT_NONVISIBLE'
        if s == 'S':
            return 'OUT_OF_TASK_GEOMETRY_S' if task == 'D_vs_E' else 'ABSENT_NONVISIBLE'
        return 'ABSTAIN' if s == 'U' else 'OTHER_UNRESOLVED'
    rules = rules or {}
    key = (row['id'], row['native_corner'])
    if model == rules.get('geometry_model') and rules.get('geometry_flags', {}).get(key) is True:
        return 'ABSENT_NONVISIBLE'
    p = probabilities.get(key)
    if p is None:
        return 'OTHER_UNRESOLVED'
    visible = p > threshold if rules.get('threshold_comparison', '>=') == '>' else p >= threshold
    return 'VISIBLE' if visible else 'ABSENT_NONVISIBLE'


def calculate(rows, model, task, probabilities, threshold, rules=None):
    cm = Counter((r['human_state'] == D, call(r, model, task, probabilities, threshold, rules)) for r in rows)
    pos = sum(r['human_state'] == D for r in rows)
    neg = len(rows) - pos
    tp, fn = cm[True, 'VISIBLE'], cm[True, 'ABSENT_NONVISIBLE']
    fp, tn = cm[False, 'VISIBLE'], cm[False, 'ABSENT_NONVISIBLE']
    up, un = cm[True, 'ABSTAIN'], cm[False, 'ABSTAIN']
    sp, sn = cm[True, 'OUT_OF_TASK_GEOMETRY_S'], cm[False, 'OUT_OF_TASK_GEOMETRY_S']
    op, on = cm[True, 'OTHER_UNRESOLVED'], cm[False, 'OTHER_UNRESOLVED']
    assert tp + fn + up + sp + op == pos and fp + tn + un + sn + on == neg
    return dict(N=len(rows), frames=len({r['id'] for r in rows}), visible_support=pos, occluded_support=neg,
                TP=tp, FN_decided=fn, FP=fp, TN=tn, U_visible=up, U_occluded=un,
                out_of_task_S_visible=sp, out_of_task_S_occluded=sn,
                other_unresolved_visible=op, other_unresolved_occluded=on,
                visible_precision=ratio(tp, tp + fp), visible_recall_all_GT=ratio(tp, pos),
                occluded_precision=ratio(tn, tn + fn), occluded_recall_all_GT=ratio(tn, neg),
                abstention_rate=ratio(up + un, len(rows)), out_of_task_S_rate=ratio(sp + sn, len(rows)),
                unresolved_rate=ratio(up + un + sp + sn + op + on, len(rows)),
                decision_coverage=ratio(tp + fn + fp + tn, len(rows)),
                correct_fraction_all_GT=ratio(tp + tn, len(rows)),
                selective_accuracy=ratio(tp + tn, tp + fn + fp + tn))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scores', required=True)
    p.add_argument('--freeze', required=True)
    p.add_argument('--score-column', default='p_visible')
    p.add_argument('--model', default='v63_spatial')
    p.add_argument('--output-prefix', default='V63_SAVED')
    p.add_argument('--include-geometry', action='store_true')
    p.add_argument('--geometry-column', default='geometry_self_hidden')
    p.add_argument('--allow-unresolved-scores', action='store_true')
    args = p.parse_args()
    assert args.model not in ['v61', 'legacy']
    prefix = HERE / args.output_prefix
    assert prefix.resolve().is_relative_to(HERE), 'Output must stay in contract directory'
    freeze = json.loads(Path(args.freeze).read_text())
    assert 'threshold' in freeze, 'Freeze must contain source-selected threshold'
    threshold = float(freeze['threshold'])
    assert 0 <= threshold <= 1
    comparison = freeze.get('threshold_comparison', '>=')
    assert comparison in ['>=', '>'], 'Only frozen >= or > comparison is supported'
    rules = {'threshold_comparison': comparison, 'geometry_flags': {}, 'geometry_model': args.model + '+GEOMETRY' if args.include_geometry else None}
    # Reject explicit leaks, while preserving honest unknowns in the report.
    for key in ['current_eval_used_for_selection', 'private_eval_used_for_selection', 'real_eval_used_for_selection']:
        assert freeze.get(key) is not True, 'Current evaluation used for selection'
    if 'new_PnP_calls' in freeze:
        assert freeze['new_PnP_calls'] == 0
    manifest = ROOT / 'evaluation_preparation_20261022' / 'PREPARED_QUERY_MANIFEST.csv'
    rows = readcsv(manifest)
    assert len(rows) == 1784
    bykey = {(r['id'], r['native_corner']): r for r in rows}
    assert len(bykey) == len(rows)
    scores = readcsv(args.scores)
    probabilities = {}
    valid_rows = 0
    for r in scores:
        key = (r['id'], r['native_corner'])
        assert key in bykey, ('Score row not registered', key)
        assert key not in probabilities, ('Duplicate score', key)
        if 'image_sha256' in r:
            assert r['image_sha256'] == bykey[key]['image_sha256'], ('Image mismatch', key)
        for c in ['query_x', 'query_y']:
            if c in r and r[c] and bykey[key][c]:
                assert abs(float(r[c]) - float(bykey[key][c])) <= 1e-5, ('Query changed', key, c)
        if args.include_geometry:
            assert args.geometry_column in r, 'Geometry route requires separately saved pure proxy flag'
            flag = str(r[args.geometry_column]).lower()
            assert flag in ['true', 'false', '', 'none'], ('Unknown geometry flag', key, flag)
            rules['geometry_flags'][key] = True if flag == 'true' else False if flag == 'false' else None
        raw = r.get(args.score_column, '')
        if raw and math.isfinite(float(raw)):
            prob = float(raw)
            assert 0 <= prob <= 1, ('Probability out of range', key)
            probabilities[key] = prob
            valid_rows += 1
        else:
            probabilities[key] = None
    newmodels = [args.model] + ([rules['geometry_model']] if args.include_geometry else [])
    models = ['v61', 'legacy', *newmodels]
    metrics, confusion, changes, strata, recordings, geometry_tradeoffs = [], [], [], [], [], []
    missing_eligible = []
    for kind in ['paired_same_query_RGB', 'same_slot_pipeline']:
        for group in GROUPS:
            for task, states in TASKS.items():
                flag = kind + '_' + ('D_vs_ES' if task == 'D_vs_S' else task)
                rr = [r for r in rows if r['group'] == group and yes(r, flag) and r['human_state'] in states]
                if not any(r['human_state'] != D for r in rr):
                    continue
                missing_eligible += [(r['id'], r['native_corner']) for r in rr if probabilities.get((r['id'], r['native_corner'])) is None]
                for model in models:
                    metrics.append(dict(cohort=kind, group=group, task=task, model=model,
                                        **calculate(rr, model, task, probabilities, threshold, rules)))
                    for state in states:
                        zz = [r for r in rr if r['human_state'] == state]
                        counts = Counter(call(r, model, task, probabilities, threshold, rules) for r in zz)
                        confusion.append(dict(cohort=kind, group=group, task=task, model=model, human_state=state,
                                              N=len(zz), pred_visible=counts['VISIBLE'], pred_absent_nonvisible=counts['ABSENT_NONVISIBLE'],
                                              pred_U=counts['ABSTAIN'], pred_out_of_task_geometry_S=counts['OUT_OF_TASK_GEOMETRY_S'],
                                              pred_other_unresolved=counts['OTHER_UNRESOLVED']))
                    for rec in sorted({r['recording'] for r in rr}):
                        zz = [r for r in rr if r['recording'] == rec]
                        recordings.append(dict(cohort=kind, group=group, task=task, model=model, recording=rec,
                                               **calculate(zz, model, task, probabilities, threshold, rules)))
                for newmodel in newmodels:
                    baselines = ['v61', 'legacy'] + ([args.model] if newmodel != args.model else [])
                    for base in baselines:
                        counts = Counter()
                        for r in rr:
                            a, b = call(r, base, task, probabilities, threshold, rules), call(r, newmodel, task, probabilities, threshold, rules)
                            if b not in ['VISIBLE', 'ABSENT_NONVISIBLE']:
                                counts['new_unresolved'] += 1
                                continue
                            gt = 'VISIBLE' if r['human_state'] == D else 'ABSENT_NONVISIBLE'
                            if a not in ['VISIBLE', 'ABSENT_NONVISIBLE']:
                                counts['baseline_unresolved_new_correct' if b == gt else 'baseline_unresolved_new_wrong'] += 1
                            else:
                                counts['both_correct' if a == gt and b == gt else 'baseline_correct_new_wrong' if a == gt else 'baseline_wrong_new_correct' if b == gt else 'both_wrong'] += 1
                        keys = ['both_correct', 'baseline_correct_new_wrong', 'baseline_wrong_new_correct', 'both_wrong', 'baseline_unresolved_new_correct', 'baseline_unresolved_new_wrong', 'new_unresolved']
                        changes.append(dict(cohort=kind, group=group, task=task, baseline=base, new_model=newmodel, N=len(rr), **{k: counts[k] for k in keys}))
                if args.include_geometry:
                    for state in states:
                        zz = [r for r in rr if r['human_state'] == state]
                        raw_calls = Counter(call(r, args.model, task, probabilities, threshold, rules) for r in zz)
                        hybrid_calls = Counter(call(r, rules['geometry_model'], task, probabilities, threshold, rules) for r in zz)
                        correct = 'VISIBLE' if state == D else 'ABSENT_NONVISIBLE'
                        geometry_tradeoffs.append(dict(cohort=kind, group=group, task=task, human_state=state, N=len(zz), raw_model=args.model, hybrid_model=rules['geometry_model'], geometry_true=sum(rules['geometry_flags'].get((r['id'], r['native_corner'])) is True for r in zz), geometry_unknown=sum(rules['geometry_flags'].get((r['id'], r['native_corner'])) is None for r in zz), raw_pred_visible=raw_calls['VISIBLE'], hybrid_pred_visible=hybrid_calls['VISIBLE'], suppressed_raw_visible=raw_calls['VISIBLE']-hybrid_calls['VISIBLE'], raw_class_recall=ratio(raw_calls[correct], len(zz)), hybrid_class_recall=ratio(hybrid_calls[correct], len(zz))))
            rr = [r for r in rows if r['group'] == group and yes(r, kind + '_D_vs_ES')]
            for state in [D, E, S]:
                ss = [r for r in rr if r['human_state'] == state]
                bins = [('px_le3', lambda r: float(r['initial_reference_error_px']) <= 3),
                        ('px_3_to10', lambda r: 3 < float(r['initial_reference_error_px']) <= 10),
                        ('px_10_to30', lambda r: 10 < float(r['initial_reference_error_px']) <= 30),
                        ('px_gt30', lambda r: float(r['initial_reference_error_px']) > 30),
                        ('reference_inside_crop', lambda r: yes(r, 'human_reference_inside_nominal_crop')),
                        ('reference_outside_crop', lambda r: not yes(r, 'human_reference_inside_nominal_crop'))]
                for label, keep in bins:
                    zz = [r for r in ss if keep(r)]
                    for model in models:
                        counts = Counter(call(r, model, 'D_vs_ES', probabilities, threshold, rules) for r in zz)
                        strata.append(dict(cohort=kind, group=group, human_state=state, error_bin=label, model=model, N=len(zz),
                                           pred_visible=counts['VISIBLE'], pred_absent_nonvisible=counts['ABSENT_NONVISIBLE'],
                                           pred_U=counts['ABSTAIN'], pred_other_unresolved=counts['OTHER_UNRESOLVED'],
                                           class_recall_all_GT=ratio(counts['VISIBLE' if state == D else 'ABSENT_NONVISIBLE'], len(zz))))
    if not args.allow_unresolved_scores:
        assert not missing_eligible, ('Missing frozen scores on eligible rows', sorted(set(missing_eligible))[:10])
    # Independently reproduce every old metric before reporting new model.
    old = readcsv(ROOT / 'saved_output_evaluation_20261022' / 'CLASSIFICATION_METRICS.csv')
    oldby = {(r['cohort'], r['group'], r['task'], r['model']): r for r in old}
    for r in metrics:
        if r['model'] not in ['v61', 'legacy']:
            continue
        before = oldby[r['cohort'], r['group'], r['task'], r['model']]
        for key in ['N', 'TP', 'FP', 'FN_decided', 'TN', 'visible_precision', 'visible_recall_all_GT', 'occluded_precision', 'occluded_recall_all_GT', 'abstention_rate', 'decision_coverage']:
            assert r[key] is None and before[key] == '' or r[key] is not None and abs(float(before[key]) - r[key]) < 1e-12, ('Baseline metric mismatch', key, r)
    suffixes = {'CLASSIFICATION_METRICS.csv': metrics, 'CONFUSION_BY_HUMAN_STATE.csv': confusion,
                'PAIRED_CHANGE.csv': changes, 'ERROR_STRATA.csv': strata, 'PER_RECORDING_METRICS.csv': recordings, 'GEOMETRY_TRADEOFF.csv': geometry_tradeoffs}
    for suffix, rr in suffixes.items():
        writecsv(Path(str(prefix) + '_' + suffix), rr)
    summary = dict(schema='SOURCE_FROZEN_SAVED_QUERY_SCORE_EVALUATION_V1', model=args.model, threshold=threshold, threshold_comparison=comparison, geometry_hybrid_model=rules['geometry_model'], geometry_rule='saved conditional cuboid self flag True -> ABSENT_NONVISIBLE; all other flags use unchanged raw score; not learned self supervision' if args.include_geometry else None,
                   score_rows=len(scores), finite_score_rows=valid_rows, missing_eligible_scores=len(set(missing_eligible)),
                   new_training_or_inference_or_PnP_by_evaluator=0,
                   previous_v61_and_legacy_metrics_independently_reproduced=True,
                   scope='semantic transfer; negative=absent/nonvisible, not external-mechanism assertion; no current-real exact-pixel GT',
                   location_error='descriptive initial-to-existing-reference coordinate error, not verified query-miss truth',
                   self='separate human self strata; mask support model cannot be claimed to learn semantic self',
                   current223='evaluation-only in this run, historically repeated development exposure; no new independent holdout claim',
                   freeze=freeze, source_bindings=[dict(path=str(Path(z).resolve()), sha256=sha(z)) for z in [manifest, args.scores, args.freeze, Path(__file__)]])
    Path(str(prefix) + '_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'model': args.model, 'metrics': len(metrics), 'baseline_reproduction': True, 'output_prefix': str(prefix)}))


if __name__ == '__main__':
    main()
