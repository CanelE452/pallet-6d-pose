"""Scalar independent saved-output audit; no evaluator or model imports."""
from pathlib import Path
from collections import Counter
import csv
import datetime
import hashlib
import json
import math
import sys
import time

import numpy as np

V = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MODELS = ('C45', 'C46', 'v61')
OWNER6 = ('INCIDENT_TARGET_DECLARED_CF', 'OTHER_TARGET', 'FOREIGN', 'BG', 'AMBIGUOUS', 'INVALID_SUPPORT')
OWNER3 = ('TARGET_VISIBLE_SUPPORTED', 'FOREIGN_VISIBLE_SUPPORTED', 'BACKGROUND_SUPPORTED')
FIELDS = ('original_global_index', 'original_occurrence_id', 'scene_index', 'base_index', 'native_slot',
          'query_xy', 'L', 'split', 'variant', 'material', 'aspect', 'geometry_family', 'foreign_family',
          'signed_quarter', 'original_semantic_state', 'query_xy_missing', 'L_missing',
          'selected_index_missing', 'query_available', 'factual_query_available', 'feature_constructed',
          'crop_constructed', 'context_available', 'Q_inside', 'RGB_central4_available',
          'native_continuous_valid', 'native_nearest_valid')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    return {'path': str(path), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def check(binding):
    assert sha(binding['path']) == binding['sha256'], binding['path']


def equal_scalar(a, b):
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        return (math.isnan(float(a)) and math.isnan(float(b))) or a == b
    return a == b


def scalar_row_equal(a, b):
    aa, bb = np.asarray(a), np.asarray(b)
    return aa.shape == bb.shape and all(equal_scalar(x.item(), y.item()) for x, y in zip(aa.ravel(), bb.ravel()))


def selected_indices(truth, cohort, group):
    parts = group.split('::')
    indices = []
    for i, r in enumerate(truth):
        if cohort == 'ALL3_MATCHED_AVAILABLE' and not r['paired']:
            continue
        if parts[0] != 'ALL' and r['split'] != parts[0]:
            continue
        if len(parts) > 1 and str(r[parts[1]]) != parts[2]:
            continue
        indices.append(i)
    return indices


def independent_metric(truth, model, ids, owner_task):
    size = 3 if owner_task else 2
    cells = [[[] for _ in range(size+1)] for _ in range(size)]
    unknown_cells = [[] for _ in range(size+1)]
    predictions = [0]*(size+1)
    for i in ids:
        r = truth[i]
        pred = r['models'][model]['prediction']
        if model != 'v61' and not owner_task and pred >= 0:
            pred = int(pred != 0)
        column = size if pred < 0 else pred
        predictions[column] += 1
        masses = r['known'] if owner_task else [r['known'][0], r['known'][1]+r['known'][2]]
        for row, mass in enumerate(masses):
            cells[row][column].append(mass)
        unknown_cells[column].append(r['ambiguous_mass']+r['invalid_mass']+r['missing_inventory_mass'])
    cm = [[math.fsum(values) for values in row] for row in cells]
    support = [math.fsum(row) for row in cm]
    predicted_mass = [math.fsum(cm[row][col] for row in range(size)) for col in range(size)]
    precision = [cm[k][k]/predicted_mass[k] if predicted_mass[k] else None for k in range(size)]
    recall = [cm[k][k]/support[k] if support[k] else None for k in range(size)]
    return dict(rows=len(ids), known_mass=math.fsum(support), expected_soft_CM=cm,
                truth_known_mass=support, expected_precision_on_known_truth_mass=precision,
                expected_recall_including_predictionU=recall, prediction_counts=predictions,
                predictionU_rows=predictions[-1],
                ambiguous_mass=math.fsum(truth[i]['ambiguous_mass'] for i in ids),
                invalid_mass=math.fsum(truth[i]['invalid_mass'] for i in ids),
                missing_q_inventory_unknown_rows=sum(int(truth[i]['missing_inventory_mass']) for i in ids),
                unknown_mass_by_prediction=[math.fsum(values) for values in unknown_cells])


DIFFS = {'max_mass_abs_difference': 0., 'max_ratio_abs_difference': 0.}


def compare_metric(expected, reported):
    assert set(expected) == set(reported)
    def compare(a, b, path):
        if isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b), path
            for j, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path+f'[{j}]')
        elif a is None:
            assert b is None, path
        elif isinstance(a, int):
            assert a == b, (path, a, b)
        else:
            delta = abs(a-b)
            ratio = 'precision' in path or 'recall' in path
            key = 'max_ratio_abs_difference' if ratio else 'max_mass_abs_difference'
            DIFFS[key] = max(DIFFS[key], delta)
            assert math.isclose(a, b, abs_tol=2e-13 if ratio else 5e-10, rel_tol=2e-12), (path, a, b)
    for key in expected:
        compare(expected[key], reported[key], key)


def main():
    started = time.monotonic()
    protocol_path = V/'saved_evaluation_preparation/PROTOCOL_BEFORE_SAVED_SCORE_OR_SOFT_TRUTH_BODIES.json'
    complete_path = V/'saved_owner_source_benchmark_v1/COMPLETE.json'
    assert sha(complete_path) == 'b0ef94f49e074b3273ff9590be783cd72ff16d151352b9fc9858b53237d98a11'
    protocol = read(protocol_path)
    complete = read(complete_path)
    assert complete['protocol'] == bind(protocol_path)
    source_bindings = {f'truth_{k}': b for k, b in protocol['sidecars'].items()}
    source_bindings.update({f'scores_{k}': b for k, b in protocol['saved_packets'].items()})
    for b in source_bindings.values():
        check(b)
    # Read original bound soft-truth packets and derive all per-row masses first.
    truth = [None]*6144
    for split in ('TRAIN', 'VALIDATION'):
        with np.load(protocol['sidecars'][split]['path'], allow_pickle=False) as z:
            assert z['owner_class_order'].tolist() == list(OWNER6)
            assert z['training_admission'].item() is False
            assert z['initializer_physical_native_identity_certified'].item() is False
            arrays = {key: z[key].copy() for key in FIELDS}
            owner = z['scene_owner_point'].copy()
        assert owner.dtype == np.float64 and owner.shape == (4096 if split == 'TRAIN' else 2048, 6)
        for local, original_index in enumerate(arrays['original_global_index']):
            index = int(original_index)
            assert truth[index] is None
            row = {key: value[local].tolist() for key, value in arrays.items()}
            assert row['split'] == split
            if row['query_xy_missing']:
                assert np.isnan(owner[local]).all()
                row.update(known=[0., 0., 0.], ambiguous_mass=0., invalid_mass=0., missing_inventory_mass=1.)
            else:
                values = [float(x) for x in owner[local]]
                assert all(math.isfinite(x) and x >= 0 for x in values)
                assert abs(math.fsum(values)-1.) <= 1e-12
                row.update(known=[values[0]+values[1], values[2], values[3]],
                           ambiguous_mass=values[4], invalid_mass=values[5], missing_inventory_mass=0.)
            row['models'] = {}
            truth[index] = row
    assert all(row is not None for row in truth)
    alignments = []
    for model in MODELS:
        with np.load(protocol['saved_packets'][model]['path'], allow_pickle=False) as z:
            identity = {key: z[key].copy() for key in ('global_index', 'occurrence_id', 'scene_index', 'base_index', 'native_slot', 'query_xy', 'L', 'no_detection')}
            available = z['query_available'].copy()
            if model == 'v61':
                raw = z['raw_visible_logit'].copy()
                saved_decoder = z['prediction_D0_E1_Uminus1'].copy()
                assert raw.shape == (6144,)
            else:
                assert z['owner_class_order'].tolist() == list(OWNER3)
                raw = z['raw_owner_three_logits'].copy()
                saved_decoder = None
                assert raw.shape == (6144, 3)
        assert raw.dtype == np.float32 and available.shape == (6144,)
        mismatch = nonfinite = 0
        for i, row in enumerate(truth):
            fields = []
            for a, b in [('global_index', 'original_global_index'), ('occurrence_id', 'original_occurrence_id'),
                         ('scene_index', 'scene_index'), ('base_index', 'base_index'), ('native_slot', 'native_slot'),
                         ('query_xy', 'query_xy'), ('L', 'L'), ('no_detection', 'selected_index_missing')]:
                if not scalar_row_equal(identity[a][i], row[b]):
                    fields.append(a)
            matched = not fields
            finite = all(math.isfinite(float(x)) for x in np.asarray(raw[i]).ravel())
            pred = -1
            if matched and bool(available[i]) and finite:
                if model == 'v61':
                    pred = 0 if float(raw[i]) >= 0. else 1
                else:
                    # Built-in max returns the first tie; no np.argmax/evaluator helper.
                    pred = max(range(3), key=lambda k: float(raw[i, k]))
            if saved_decoder is not None and matched:
                assert pred == int(saved_decoder[i]), (i, 'v61 original decoder')
            row['models'][model] = dict(prediction=pred, original_available=bool(available[i]),
                                        matched=matched, finite=finite, mismatch_fields=fields)
            mismatch += int(not matched)
            nonfinite += int(matched and bool(available[i]) and not finite)
        alignments.append(dict(model=model, matched_rows=6144-mismatch, mismatch_rows=mismatch,
                               original_inputU=sum(not bool(x) for x in available), available_nonfinite=nonfinite,
                               saved_query_availability_equals_original_census=all(bool(a) == r['query_available'] for a, r in zip(available, truth))))
    for row in truth:
        row['paired'] = all(x['matched'] and x['original_available'] and x['finite'] for x in row['models'].values())
    assert sum(row['query_xy_missing'] for row in truth) == 56
    assert sum(not row['query_available'] for row in truth) == 88
    assert all(x['matched_rows'] == 6144 and x['original_inputU'] == 88 and x['available_nonfinite'] == 0 for x in alignments)
    # Seal independent source derivation before reading published CSV/report bodies.
    independent_path = OUT/'INDEPENDENT6144_PREDICTIONS_AND_TRUTH_MASS.jsonl'
    with independent_path.open('x') as stream:
        for row in truth:
            clean = {k: row[k] for k in ('original_global_index', 'original_occurrence_id', 'split', 'native_slot', 'known', 'ambiguous_mass', 'invalid_mass', 'missing_inventory_mass', 'models', 'paired')}
            stream.write(json.dumps(clean, allow_nan=False, separators=(',', ':'))+'\n')
    expected_groups = set()
    for split in ('ALL', 'TRAIN', 'VALIDATION'):
        expected_groups.add(split)
        for field in ('native_slot', 'material', 'aspect', 'variant', 'original_semantic_state'):
            expected_groups.update(split+'::'+field+'::'+str(value) for value in {r[field] for r in truth})
    computed = {}
    for cohort in ('FULL_INVENTORY', 'ALL3_MATCHED_AVAILABLE'):
        computed[cohort] = {}
        for group in sorted(expected_groups):
            ids = selected_indices(truth, cohort, group)
            computed[cohort][group] = {}
            for model in MODELS:
                record = {'binary_target_presence': independent_metric(truth, model, ids, False)}
                if model != 'v61':
                    record['owner3'] = independent_metric(truth, model, ids, True)
                computed[cohort][group][model] = record
    metrics_path = OUT/'INDEPENDENT_EXPECTED_SOFT_CONFUSIONS_AND_STRATA.json'
    metrics_path.write_text(json.dumps(dict(reports=computed), separators=(',', ':'), allow_nan=False)+'\n')

    check(complete['ledger'])
    with Path(complete['ledger']['path']).open(newline='') as stream:
        ledger = list(csv.DictReader(stream))
    assert len(ledger) == 6144
    for i, (row, published) in enumerate(zip(truth, ledger)):
        for key in ('original_global_index', 'original_occurrence_id', 'scene_index', 'base_index', 'native_slot', 'split', 'material', 'aspect', 'variant', 'original_semantic_state', 'query_available', 'query_xy_missing', 'L_missing'):
            assert published[key] == str(row[key]), (i, key)
        for key, expected in [('w_TARGET', row['known'][0]), ('w_FOREIGN', row['known'][1]), ('w_BG', row['known'][2]),
                              ('ambiguous_mass', row['ambiguous_mass']), ('invalid_mass', row['invalid_mass']), ('missing_q_inventory_unknown', row['missing_inventory_mass'])]:
            assert float(published[key]) == expected, (i, key)
        for model in MODELS:
            m = row['models'][model]
            assert int(published[model+'_prediction']) == m['prediction']
            assert published[model+'_original_available'] == str(m['original_available'])
            assert published[model+'_matched'] == str(m['matched'])
            assert published[model+'_mismatch_fields'] == '|'.join(m['mismatch_fields'])
    check(complete['reports'])
    published_report = read(complete['reports']['path'])
    assert published_report['primary'] == 'FULL_INVENTORY/VALIDATION'
    assert published_report['TRAIN_apparent_only'] is True
    count = 0
    for cohort, groups in computed.items():
        assert set(published_report['reports'][cohort]) == expected_groups
        for group, models in groups.items():
            for model, tasks in models.items():
                assert set(tasks) == set(published_report['reports'][cohort][group][model])
                for task, metric in tasks.items():
                    compare_metric(metric, published_report['reports'][cohort][group][model][task])
                    count += 1
    assert alignments == complete['alignment']
    assert sum(r['paired'] for r in truth) == complete['paired_matched_available_rows'] == 6056
    for b in source_bindings.values():
        check(b)
    primary = computed['FULL_INVENTORY']['VALIDATION']
    report = dict(schema='V174_INDEPENDENT_ACTUAL_SAVED_OWNER_EVALUATION_REVIEW_A_v1', completed=True, PASS=True,
                  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), code=bind(Path(__file__).resolve()),
                  complete=bind(complete_path), protocol=bind(protocol_path), source_bindings=source_bindings,
                  independent_rows=bind(independent_path), independent_metrics=bind(metrics_path),
                  published_ledger=complete['ledger'], published_reports=complete['reports'],
                  method='Original saved score3 packets and source soft-truth2 packets first; scalar row equality/NaN checks, built-in first-tie max, original v61>=0 rule, inputU precedence. Scalar soft-truth merge and math.fsum CM/P/R computed and sealed before published report/CSV reads. No evaluator/producer/helper imports.',
                  coverage=dict(original_rows=6144, TRAIN=4096, VALIDATION=2048, missing_q=56, original_inputU=88,
                                raw_score_packets=3, GT_soft_sidecars=2, source_packet_hashes_before_after_match=True,
                                saved_row_alignments_checked=3*6144, predictions_checked=3*6144,
                                all_ledger_rows_and_mass_columns_exact=True, paired_rows=6056,
                                groups_per_cohort=len(expected_groups), cohorts=2, model_task_metric_records=count,
                                all_reported_strata_checked=True, **DIFFS),
                  alignment=alignments, independent_primary_validation=primary,
                  limitations=['Expected known-mass CM/P/R for literal original-q raster owner, not original native E, external/self cause, coordinate gold, physical native identity or full task success.',
                               'The original synthetic TRAIN split is apparent; VALIDATION is the fixed historically reused public source split, not a new blind/external dataset.',
                               'This audit independently evaluates saved outputs and previously audited source point masses. It does not reread source RGB/masks or resample central4/32; no new model/encoder/forward/fit or threshold selection.'],
                  new_model_encoder_feature_weight_forward_fit_private_current_V169_V112=0,
                  source_labels_qL_and_qualification_unchanged=True, training_admission=False,
                  selected_model=None, adopted=False, fully_solved=False, seconds=time.monotonic()-started)
    path = OUT/'REVIEW.json'
    assert not path.exists()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(PASS=True, review=bind(path), coverage=report['coverage'])), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        p = OUT/'AUDIT_FAILURE_PRESERVED.json'
        if not p.exists():
            p.write_text(json.dumps(dict(completed=False, traceback=traceback.format_exc()), indent=2)+'\n')
        raise
