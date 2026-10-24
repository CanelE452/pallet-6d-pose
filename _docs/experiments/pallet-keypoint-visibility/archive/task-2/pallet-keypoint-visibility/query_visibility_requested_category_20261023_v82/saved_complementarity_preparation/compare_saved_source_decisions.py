"""Source saved decisions only; no model, RGB, native masks, or current data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import csv
import numpy as np

HERE = Path(__file__).resolve().parent


def bind(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check(record):
    assert bind(record['path']) == record, record['path']


def summarize(y, old, mask):
    """old C/V, mask C/V/U. U is a failure, never a negative or removed row."""
    y, old, mask = [np.asarray(a, dtype=np.int8) for a in (y, old, mask)]
    assert y.ndim == old.ndim == mask.ndim == 1
    assert y.shape == old.shape == mask.shape
    assert np.isin(y, [0, 1]).all() and np.isin(old, [0, 1]).all()
    assert np.isin(mask, [0, 1, 2]).all()
    joint = np.zeros((2, 2, 3), np.int64)
    np.add.at(joint, (y, old, mask), 1)
    oc = old == y
    mc = mask == y
    unknown = mask == 2
    accepted_wrong = ~mc & ~unknown
    partition = {
        'both_correct': int((oc & mc).sum()),
        'V76_only_correct_V82_accepted_wrong': int((oc & accepted_wrong).sum()),
        'V76_only_correct_V82_U': int((oc & unknown).sum()),
        'V82_only_correct': int((~oc & mc).sum()),
        'both_wrong_V82_accepted_wrong': int((~oc & accepted_wrong).sum()),
        'both_fail_V82_U': int((~oc & unknown).sum())}
    assert sum(partition.values()) == len(y) == int(joint.sum())
    old_cm, mask_cm = joint.sum(axis=2), joint.sum(axis=1)
    return {'N': len(y), 'GT_CV_support': np.bincount(y, minlength=2).tolist(),
        'joint_axes': ['GT_CV', 'V76_CV', 'V82_CVU'], 'joint_counts': joint.tolist(),
        'V76_CM_rows_GT_CV_cols_pred_CV': old_cm.tolist(),
        'V82_CM_rows_GT_CV_cols_pred_CVU': mask_cm.tolist(),
        'mutually_exclusive_partition': partition,
        'V76_only_correct_including_V82_U': int((oc & ~mc).sum()),
        'both_fail_including_V82_U': int((~oc & ~mc).sum()),
        'V76_error_N': int((~oc).sum()), 'V82_accepted_error_N': int(accepted_wrong.sum()),
        'V82_U_N': int(unknown.sum()), 'V82_failure_U_is_miss_N': int((~mc).sum()),
        'partition_is_not_an_ensemble_accuracy_or_upper_bound': True}


def generated():
    # Every GT x old decision x mask decision cell occurs once, including U.
    rows = np.array([(g, a, b) for g in [0, 1] for a in [0, 1] for b in [0, 1, 2]])
    result = summarize(*rows.T)
    assert np.array_equal(result['joint_counts'], np.ones((2, 2, 3), np.int64))
    assert result['mutually_exclusive_partition'] == {
        'both_correct': 2, 'V76_only_correct_V82_accepted_wrong': 2,
        'V76_only_correct_V82_U': 2, 'V82_only_correct': 2,
        'both_wrong_V82_accepted_wrong': 2, 'both_fail_V82_U': 2}
    assert result['V82_failure_U_is_miss_N'] == 8
    assert result['V76_only_correct_including_V82_U'] == result['both_fail_including_V82_U'] == 4
    empty = summarize([], [], [])
    assert empty['N'] == 0 and np.asarray(empty['joint_counts']).sum() == 0
    # Frozen probability .5 is represented by finite raw-logit >=0, tie positive.
    assert (np.array([-1e-12, 0., 1e-12]) >= 0).astype(int).tolist() == [0, 1, 1]
    write(HERE / 'GENERATED_COMPLEMENTARITY_TESTS.json', {
        'PASS': True, 'generated_only': True, 'code': bind(__file__),
        'all12_cells_partition_and_U_failure_exact': True, 'empty_denominator_safe': True,
        'logit_zero_tie_visible': True, 'actual_score_GT_RGB_masks_models_current_reads': 0})


def run(authorization, destination):
    entry_path = HERE / 'PREPARATION_ENTRYLOCK.json'
    entry = read(entry_path)
    auth = read(authorization)
    assert auth['approved'] is True and auth['prepared_entrylock'] == bind(entry_path)
    assert auth['source_only_saved'] is True and auth['current223'] is False
    assert auth['new_forward'] == 0 and auth['new_fit'] is False and auth['PnP'] == 0
    assert entry['code'] == bind(__file__)
    for record in entry['bindings'].values():
        check(record)
    policy = read(entry['bindings']['policy']['path'])
    assert policy['execution_requires_new_root_authorization']
    destination = Path(destination).resolve()
    assert not destination.exists()
    destination.mkdir(parents=True)
    # Failure leaves the started receipt and all partial artifacts intact.
    write(destination / 'EXECUTION_STARTED.json', {
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'authorization': bind(authorization),
        'preparation_entrylock': bind(entry_path), 'model_forward': 0, 'current223': False})
    freeze = read(entry['bindings']['V76_model_freeze']['path'])
    assert freeze['completed'] and freeze['best_update'] == 200
    assert freeze['visible_probability_threshold'] == .5
    assert freeze['current_used_for_fit_selection'] is False and freeze['PnP'] == 0
    with np.load(entry['bindings']['V76_selected_saved_scores']['path'], allow_pickle=False) as z:
        old = {k: z[k] for k in ['id', 'source', 'split', 'support_logit', 'primary_loss_mask']}
    with np.load(entry['bindings']['V82_saved_evaluation']['path'], allow_pickle=False) as z:
        new = {k: z[k] for k in ['input_id', 'original_id', 'source', 'prediction_CVU',
            'original_primary8', 'reference_local_label8', 'strict535_mask8', 'q_pixel_state8', 'q_pixel_valid8']}
    with np.load(entry['bindings']['unchanged_V81_GT_pack']['path'], allow_pickle=False) as z:
        for key in ['input_id', 'original_id', 'source', 'original_primary8', 'reference_local_label8', 'strict535_mask8']:
            assert np.array_equal(z[key], new[key]), key
        assert np.array_equal(z['q_pixel_state8_CVO_invalid'], new['q_pixel_state8'])
        assert np.array_equal(z['q_pixel_valid8'], new['q_pixel_valid8'])
    assert len(old['id']) == 5203 and len(set(old['id'].tolist())) == 5203
    assert len(new['original_id']) == 1035 and len(set(new['original_id'].tolist())) == 1035
    oi = {str(ident): i for i, ident in enumerate(old['id'])}
    rows = np.array([oi[str(ident)] for ident in new['original_id']], np.int64)
    assert np.array_equal(old['id'][rows], new['original_id'])
    with np.load(entry['bindings']['V76_unchanged_GT_corpus']['path'], allow_pickle=False) as z:
        assert np.array_equal(z['id'], old['id'])
        assert np.array_equal(z['label8'][rows], new['reference_local_label8'])
    assert (old['split'][rows] == 'val').all()
    assert np.array_equal(old['primary_loss_mask'][rows], new['original_primary8'])
    ycb = new['source'] == 'YCB_PUBLIC_REAL'
    assert ycb.sum() == 800
    assert (old['source'][rows[ycb]] == 'YCB_PUBLIC_REAL').all()
    primary = ycb[:, None] & new['original_primary8']
    strict = ycb[:, None] & new['strict535_mask8']
    assert primary.sum() == 2018 and strict.sum() == 535 and not (strict & ~primary).any()
    y = new['reference_local_label8']
    assert np.bincount(y[primary], minlength=2).tolist() == [287, 1731]
    assert np.bincount(y[strict], minlength=2).tolist() == [76, 459]
    assert np.array_equal(y[strict], new['q_pixel_state8'][strict])
    logits = old['support_logit'][rows]
    assert logits.shape == y.shape == new['prediction_CVU'].shape == (1035, 8)
    assert np.isfinite(logits[primary]).all()
    op = (logits >= 0).astype(np.int8)
    mp = new['prediction_CVU']
    records = []
    outputs = {}
    for name, cohort in [('NATURAL_YCB_PRIMARY2018', primary), ('FIXED_STRICT535', strict)]:
        outputs[name] = summarize(y[cohort], op[cohort], mp[cohort])
        outputs[name]['by_GT_class'] = {}
        for cls in [0, 1]:
            take = cohort & (y == cls)
            outputs[name]['by_GT_class'][str(cls)] = summarize(y[take], op[take], mp[take])
        outputs[name]['reference_GT_CV_vs_query_GT_invalid_C_V_O'] = [
            [int((cohort & (y == cls) & (new['q_pixel_state8'] == qstate)).sum())
             for qstate in [-1, 0, 1, 2]] for cls in [0, 1]]
        for i, k in zip(*np.nonzero(cohort)):
            records.append({'cohort': name, 'original_id': str(new['original_id'][i]), 'native_k': int(k),
                'GT_reference_support_CV': int(y[i,k]), 'V76_decision_CV': int(op[i,k]),
                'V82_query_mask_decision_CVU': int(mp[i,k]), 'V76_correct': bool(op[i,k] == y[i,k]),
                'V82_correct_U_is_miss': bool(mp[i,k] == y[i,k]), 'V82_U': bool(mp[i,k] == 2)})
    with (destination / 'JOINED_DECISIONS_GT_ONLY.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    result = {'completed': True, 'PASS': True, 'policy': entry['bindings']['policy'],
        'prepared_entrylock': bind(entry_path), 'authorization': bind(authorization),
        'cohorts': outputs, 'all1035_ID_primary_GT_joins_exact': True,
        'interpretation_KO': policy['interpretation_KO'], 'new_forward_fit_current_PnP': 0,
        'source_RGB_native_mask_pixels_features_weights_read': False,
        'only_saved_source_logits_decisions_labels_and_metadata_read': True}
    write(destination / 'SAVED_COMPLEMENTARITY_DIAGNOSIS_KO.json', result)
    np.savez_compressed(destination / 'JOINED_DECISIONS_GT_ONLY.npz', original_id=new['original_id'],
        source=new['source'], reference_label8=y, V76_decision_CV=op, V82_decision_CVU=mp,
        primary2018=primary, strict535=strict)
    write(destination / 'SEAL.json', {'completed': True, 'PASS': True,
        'files': [bind(path) for path in sorted(destination.iterdir()) if path.is_file()],
        'code': bind(__file__), 'inputs': entry['bindings'], 'new_forward_fit_current_PnP': 0})
    print(json.dumps({'PASS': True, 'seal': bind(destination / 'SEAL.json')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['generated', 'run'])
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'generated':
        generated()
    else:
        assert args.authorization and args.output
        run(args.authorization, args.output)
