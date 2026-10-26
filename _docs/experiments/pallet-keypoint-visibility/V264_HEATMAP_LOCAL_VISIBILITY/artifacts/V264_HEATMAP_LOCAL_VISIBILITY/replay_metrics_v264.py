#!/usr/bin/env python3
"""Portable stdlib-only replay of V264 saved classification metrics.

This deliberately imports no original evaluation/model code and reads no images,
features, or checkpoints. Artifact binding paths are treated as provenance strings;
only explicitly mapped files beside this script are opened.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat


MODELS = ('GLOBAL_MATCHED', 'HEATMAP_LOCAL', 'v61', 'CAL_LEGACY', 'V263_JOINT', 'V263_VIS', 'V261_CONTROL', 'V261_TREAT')
GROUPS = ('ALL', 'WOOD_PRIMARY105', 'SQUARE118')
COHORTS = ('PRIMARY', 'ALL_REGISTERED_D_E', 'MANUAL_CLICK_D_ALL', 'MANUAL_CLICK_D_LE3_VALID_INPUT')
INTEGER_FIELDS = ('N', 'truth_D', 'truth_E', 'D_D', 'D_E', 'D_U', 'E_D', 'E_E', 'E_U', 'U')
FLOAT_FIELDS = ('D_precision', 'D_recall', 'D_F1', 'E_precision', 'E_recall', 'E_F1', 'U_rate', 'coverage', 'accuracy_including_U_as_unresolved', 'balanced_recall', 'accepted_accuracy')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe(value, fresh=False):
    p = Path(value)
    require(p.is_absolute() and '..' not in p.parts, 'Absolute nontraversal path required')
    require(not any(term in str(p).casefold() for term in ('v112', 'cache', '.pyc')), 'Forbidden lexical artifact path')
    for ancestor in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(ancestor).st_mode
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(mode), 'Artifact symlink/ancestor forbidden')
    if fresh:
        require(not p.exists(), 'Fresh output required: ' + str(p))
    return p


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def binding(path):
    path = safe(path)
    raw = path.read_bytes()
    return dict(path=str(path), sha256=digest(raw), size_bytes=len(raw))


def read_mapped(root, relative, provenance):
    # Never resolve or open a serialized absolute artifact path.
    require(Path(relative).is_absolute() is False and '..' not in Path(relative).parts, 'Explicit relative mapping only')
    require(all(k in provenance for k in ('path', 'sha256', 'size_bytes')), 'Complete SHA/size binding required')
    require(str(provenance['path']).replace('\\', '/').split('/')[-1] == Path(relative).name, 'Mapped filename/provenance mismatch')
    local = safe(root / relative)
    raw = local.read_bytes()
    require(len(raw) == provenance['size_bytes'] and digest(raw) == provenance['sha256'], 'Mapped SHA/size mismatch: ' + relative)
    return raw, dict(relative=relative, sha256=digest(raw), size_bytes=len(raw), absolute_provenance_opened=False)


def nested_bindings(obj):
    if isinstance(obj, dict):
        if all(k in obj for k in ('path', 'sha256', 'size_bytes')):
            yield {k: obj[k] for k in ('path', 'sha256', 'size_bytes')}
        for value in obj.values():
            yield from nested_bindings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from nested_bindings(value)


def optional_number(value):
    if value is None or value == '':
        return None
    result = float(value)
    require(math.isfinite(result), 'Finite metric/score required')
    return result


def divide(a, b):
    return a / b if b else None


def cohort_member(row, name):
    if name == 'PRIMARY':
        return row['PRIMARY'] == 'True'
    if name == 'ALL_REGISTERED_D_E':
        return row['human_state'] in ('DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED')
    if name == 'MANUAL_CLICK_D_ALL':
        return row['human_state'] == 'DIRECT_VISIBLE' and row['annotation_coordinate_source'] == 'manual_click'
    if name == 'MANUAL_CLICK_D_LE3_VALID_INPUT':
        return row['MANUAL_D_Q0_LE3'] == 'True'
    raise ValueError(name)


def recompute(rows, model):
    matrix = [[0, 0, 0], [0, 0, 0]]
    for row in rows:
        require(row['human_state'] in ('DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED'), 'Only known D/E may enter binary metrics')
        prediction = row[model + '_prediction']
        require(prediction in ('D', 'E', 'U'), 'Saved prediction outside D/E/U')
        matrix[0 if row['human_state'] == 'DIRECT_VISIBLE' else 1][('D', 'E', 'U').index(prediction)] += 1
    dd, de, du = matrix[0]
    ed, ee, eu = matrix[1]
    n_d, n_e = sum(matrix[0]), sum(matrix[1])
    n = n_d + n_e
    correct, accepted = dd + ee, n - du - eu
    result = dict(N=n, truth_D=n_d, truth_E=n_e, D_D=dd, D_E=de, D_U=du, E_D=ed, E_E=ee, E_U=eu, U=du + eu)
    for label, tp, fp, missed, true_total in (('D', dd, ed, de + du, n_d), ('E', ee, de, ed + eu, n_e)):
        result[label + '_precision'] = divide(tp, tp + fp)
        result[label + '_recall'] = divide(tp, true_total)
        result[label + '_F1'] = divide(2 * tp, 2 * tp + fp + missed)
    result.update(U_rate=divide(du + eu, n), coverage=divide(accepted, n), accuracy_including_U_as_unresolved=divide(correct, n),
                  accepted_accuracy=divide(correct, accepted), balanced_recall=((dd/n_d + ee/n_e)/2 if n_d and n_e else None))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).absolute().parent))
    parser.add_argument('--output', default='VERIFY_METRICS_REPLAY.json')
    args = parser.parse_args()
    root = safe(args.root)
    require(Path(args.output).name == args.output and '..' not in args.output, 'Single fresh output filename only')
    output = safe(root / args.output, fresh=True)

    complete_path = safe(root / 'evaluation_v1/COMPLETE.json')
    complete_raw = complete_path.read_bytes()
    complete = json.loads(complete_raw)
    require(complete.get('completed') is True and complete.get('PASS') is True, 'Evaluation completion required')
    joined_raw, joined_identity = read_mapped(root, 'evaluation_v1/ALL1784_COMPARISON.csv', complete['outputs']['joined'])
    metric_raw, metric_identity = read_mapped(root, 'evaluation_v1/CLASSIFICATION.csv', complete['outputs']['classification'])
    rows = list(csv.DictReader(io.StringIO(joined_raw.decode('utf-8-sig'))))
    metrics = list(csv.DictReader(io.StringIO(metric_raw.decode('utf-8-sig'))))
    require(len(rows) == 1784 and len(metrics) == 96, 'Exactly1784 saved rows and96 metric rows')
    for i, row in enumerate(rows):
        require(int(row['global_index']) == i and int(row['native_corner']) == i % 8, 'Exact row/native order')
        require(row['row_id'] == row['id'] + ':native' + str(i % 8), 'Exact row identity')
        require(row['PRIMARY'] in ('True', 'False') and row['MANUAL_D_Q0_LE3'] in ('True', 'False'), 'Boolean cohort flags')
    require(sum(cohort_member(r, 'PRIMARY') for r in rows) == 1237, 'Fixed primary1237')
    require(sum(cohort_member(r, 'MANUAL_CLICK_D_LE3_VALID_INPUT') for r in rows) == 376, 'Fixed nearD376')
    require(sum(r['human_state'] == 'SELF_OCCLUDED' for r in rows) == 136, 'SELF136 retained outside binary metrics')

    observed_keys, maximum_difference = set(), 0.0
    recomputed = []
    for metric in metrics:
        group, cohort, model = (metric[k] for k in ('group', 'cohort', 'model'))
        require(group in GROUPS and cohort in COHORTS and model in MODELS, 'Declared metric key only')
        key = (group, cohort, model)
        require(key not in observed_keys, 'Duplicate metric row')
        observed_keys.add(key)
        selected = [r for r in rows if (group == 'ALL' or r['group'] == group) and cohort_member(r, cohort)]
        actual = recompute(selected, model)
        for field in INTEGER_FIELDS:
            require(int(metric[field]) == actual[field], 'Count mismatch: ' + '/'.join(key) + '/' + field)
        for field in FLOAT_FIELDS:
            expected, value = optional_number(metric[field]), actual[field]
            require((expected is None) == (value is None), 'Undefined metric mismatch: ' + '/'.join(key) + '/' + field)
            if expected is not None:
                difference = abs(expected - value)
                maximum_difference = max(maximum_difference, difference)
                require(difference <= 1e-12, 'Rate mismatch: ' + '/'.join(key) + '/' + field)
        recomputed.append(dict(group=group, cohort=cohort, model=model, **actual))
    require(observed_keys == {(g, c, m) for g in GROUPS for c in COHORTS for m in MODELS}, 'All96 declared combinations required')

    freeze_path = safe(root / 'CURRENT_PREDICTIONS_FROZEN.json')
    freeze_raw = freeze_path.read_bytes()
    freeze = json.loads(freeze_raw)
    require(freeze.get('completed') is True or freeze.get('PASS') is True, 'New current freeze must be complete')
    prediction_bindings = [b for b in nested_bindings(freeze) if str(b['path']).replace('\\', '/').split('/')[-1] == 'CURRENT_ALL1784_PREDICTIONS.jsonl']
    require(prediction_bindings and all(b == prediction_bindings[0] for b in prediction_bindings), 'Exact prediction binding required')
    prediction_raw, prediction_identity = read_mapped(root, 'CURRENT_ALL1784_PREDICTIONS.jsonl', prediction_bindings[0])
    predictions = [json.loads(line) for line in prediction_raw.splitlines() if line]
    require(len(predictions) == 1784, 'New prediction rows1784')
    for i, (prediction, row) in enumerate(zip(predictions, rows)):
        require(prediction['global_index'] == i and prediction['native_slot'] == i % 8 and prediction['frame_id'] == row['id'] and prediction['row_id'] == row['row_id'], 'Current prediction/CSV identity mismatch')
        require(prediction['available'] == (row['original_query_valid'] == 'True'), 'Original support policy changed')
        for model in ('GLOBAL_MATCHED', 'HEATMAP_LOCAL'):
            require(prediction[model]['pred'] == row[model + '_prediction'], 'Current prediction/CSV response mismatch')
            for field in ('E_logit', 'p_E'):
                saved, copied = optional_number(prediction[model][field]), optional_number(row[model + '_' + field])
                require((saved is None) == (copied is None) and (saved is None or abs(saved - copied) <= 1e-12), 'Current prediction/CSV score mismatch')

    result = dict(completed=True, PASS=True, stdlib_only=True, original_evaluator_model_encoder_RGB_reads=0,
        saved_rows_verified=1784, metric_rows_recomputed=96, integer_fields_per_metric=len(INTEGER_FIELDS),
        rate_fields_per_metric=len(FLOAT_FIELDS), rate_tolerance=1e-12, maximum_absolute_rate_difference=maximum_difference,
        new_prediction_response_pairs_verified=3568, no_training_threshold_selection_or_solver=True,
        portable_explicit_relative_mapping=True, absolute_artifact_provenance_paths_opened=0,
        inputs=[dict(relative='evaluation_v1/COMPLETE.json', sha256=digest(complete_raw), size_bytes=len(complete_raw)), joined_identity, metric_identity,
                dict(relative='CURRENT_PREDICTIONS_FROZEN.json', sha256=digest(freeze_raw), size_bytes=len(freeze_raw)), prediction_identity],
        replay_code=binding(Path(__file__).absolute()), recomputed_classification=recomputed,
        created_UTC=dt.datetime.now(dt.timezone.utc).isoformat())
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(completed=True, PASS=True, output=binding(output), metric_rows=96, maximum_absolute_rate_difference=maximum_difference)))


if __name__ == '__main__':
    main()
