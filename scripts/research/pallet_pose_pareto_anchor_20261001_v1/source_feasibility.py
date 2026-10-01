"""TRAIN-only feasibility of reference-anchored whole-pose targets.

``selfcheck`` uses invented arrays only. ``score`` requires SOURCE_PROTOCOL to
be sealed first and reads only the original guarded TRAIN error cache. The
reference-based safe mask defines targets; it is never an inference mask.
"""
import argparse
from collections import Counter
import math
from pathlib import Path
import time

import numpy as np

KEYS = ('translation_cm', 'rotation_deg')
MODELS = ('R0_ONLY', 'UNION_s1', 'UNION_s2', 'UNION_s3')
SCALE = (2.4636887551191258, 1.113474019956766)


def independent_targets(errors, valid, anchor_errors, anchor_index, scale, names):
    """Independent scalar-loop eligibility, cost, Pareto, and deterministic tie."""
    errors, valid = np.asarray(errors), np.asarray(valid)
    safe = np.zeros(valid.shape, bool)
    target = np.full(len(errors), -1, np.int64)
    assert valid.dtype == bool and errors.shape == (*valid.shape, 2)
    assert np.isfinite(scale).all() and np.all(np.asarray(scale) > 0)
    for row in range(len(errors)):
        if not valid[row].any():
            assert anchor_index[row] == -1 and np.isposinf(anchor_errors[row]).all()
            continue
        index = int(anchor_index[row])
        assert index in (0, 1) and names[index].startswith('R0:') and valid[row, index]
        np.testing.assert_array_equal(errors[row, index], anchor_errors[row])
        assert np.isfinite(anchor_errors[row]).all()
        allowed = []
        for candidate in range(valid.shape[1]):
            if valid[row, candidate] and all(float(errors[row, candidate, axis]) <=
                    float(anchor_errors[row, axis]) for axis in (0, 1)):
                safe[row, candidate] = True
                allowed.append(candidate)
        assert index in allowed
        costs = {j: max(float(errors[row, j, 0]) / float(scale[0]),
                        float(errors[row, j, 1]) / float(scale[1])) for j in allowed}
        minimum = min(costs.values())
        tied = [j for j in allowed if costs[j] == minimum]
        frontier = [j for j in tied if not any(
            all(float(errors[row, k, axis]) <= float(errors[row, j, axis]) for axis in (0, 1))
            and any(float(errors[row, k, axis]) < float(errors[row, j, axis]) for axis in (0, 1))
            for k in tied)]
        def key(j):
            expert, hypothesis = names[j].split(':')
            return expert != 'R0', hypothesis, expert
        target[row] = min(frontier, key=key)
    return target, safe


def quantile(values, fraction):
    values = np.sort(np.asarray(values, np.float64))
    assert not np.isnan(values).any() and not np.isneginf(values).any()
    if not len(values):
        return None
    rank = (len(values) - 1) * fraction
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return float(values[lower])
    if np.isposinf(values[upper]):
        return float('inf')
    return float(values[lower] + (rank - lower) * (values[upper] - values[lower]))


def summary(errors):
    errors = np.asarray(errors, np.float64)
    assert errors.ndim == 2 and errors.shape[1] == 2
    assert not np.isnan(errors).any() and not np.isneginf(errors).any() and (errors >= 0).all()
    available = np.isfinite(errors).all(1)
    assert np.array_equal(np.isfinite(errors).any(1), available)
    out = dict(frames=len(errors), valid_pose=int(available.sum()), failed_pose=int((~available).sum()))
    for mode, mask in [('full_population', np.ones(len(errors), bool)), ('conditional', available)]:
        out[mode] = {}
        for axis, key in enumerate(KEYS):
            out[mode][key] = {}
            for label, fraction in [('median', .5), ('P90', .9)]:
                value = quantile(errors[mask, axis], fraction)
                out[mode][key][label] = value if value is not None and np.isfinite(value) else None
                out[mode][key][label + '_status'] = ('NA_EMPTY' if value is None else
                    'FINITE' if np.isfinite(value) else 'POSITIVE_INFINITY')
    return out


def gate(before, after, ratio=1.05):
    old, new = summary(before), summary(after)
    checks = dict(failure_no_increase=new['failed_pose'] <= old['failed_pose'])
    for axis, key in enumerate(KEYS):
        oldmedian, newmedian = quantile(before[:, axis], .5), quantile(after[:, axis], .5)
        oldtail, newtail = quantile(before[:, axis], .9), quantile(after[:, axis], .9)
        checks[key + '_median_strict'] = bool(oldmedian is not None and newmedian is not None and newmedian < oldmedian)
        checks[key + '_P90_guard'] = bool(oldtail is not None and newtail is not None and newtail <= ratio * oldtail)
    return dict(PASS=all(checks.values()), checks=checks, before=old, after=new)


def pool(parent, model, old):
    if model == 'R0_ONLY':
        return parent['errors']['R0'], parent['valid']['R0'], old.candidate_names('R0_ONLY', 1)
    seed = int(model[-1])
    expert = f'DIVERSE251_s{seed}'
    return (np.concatenate([parent['errors']['R0'], parent['errors'][expert]], axis=1),
            np.concatenate([parent['valid']['R0'], parent['valid'][expert]], axis=1),
            old.candidate_names('UNION', seed))


def operational(parent, old):
    """Read cached TRAIN operational errors, never raw source reference poses."""
    inputs = parent['protocol']['inputs']
    old.C.verify(inputs['features'])
    old.C.verify(inputs['train_labels'])
    with np.load(old.C.ROOT / inputs['features']['path'], allow_pickle=False) as feature, \
            np.load(old.C.ROOT / inputs['train_labels']['path'], allow_pickle=False) as label:
        np.testing.assert_array_equal(label['ids'], parent['ids'])
        index = parent['source_index']
        np.testing.assert_array_equal(feature['ids'][index], parent['ids'])
        out = {}
        for model in parent['errors']:
            value = np.stack([label[model + '_GEO_T_cm'], label[model + '_GEO_R_deg']], axis=-1)
            selected = feature[model + '_GEO_index'][index]
            assert value.shape == (len(index), 2) and selected.shape == (len(index),)
            assert np.isin(selected, [-1, 0, 1]).all()
            same = np.full_like(value, np.inf)
            mask = selected >= 0
            same[mask] = parent['errors'][model][np.flatnonzero(mask), selected[mask]]
            np.testing.assert_array_equal(value, same)
            out[model] = value
    return out


def score():
    from . import common as C
    from . import convex_train as T
    started = time.monotonic()
    T.OLD.install_training_guard()
    protocol = C.protocol('SOURCE_PROTOCOL')
    assert protocol['schema'] == 'pallet_pose_pareto_anchor_source_feasibility_v1'
    assert protocol['source_TRAIN_only'] and protocol['scope'] == 'C2_TRAIN_ONLY'
    assert protocol['frames'] == 2598 and protocol['models'] == list(MODELS)
    assert protocol['scale'] == list(SCALE) and protocol['p90_ratio_max'] == 1.05
    assert protocol['median_strict'] and protocol['failure_no_increase']
    assert protocol['target_rule'] == 'R0_GEO_PARETO_NONREGRESSION_FIXED_TRAIN_MINMAX'
    assert protocol['required_checks'] == 45 and protocol['all_seeds_required']
    authorization = C.read(C.ROOT / protocol['inputs']['real_authorization']['path'])
    assert authorization['complete'] and authorization['all_original_real_diagnostic_gates_pass']
    assert authorization['independent_verification_pass']
    assert not authorization['real_per_frame_values_in_this_receipt']
    assert not authorization['real_quality_values_in_this_receipt']
    assert not (C.DOC / 'SOURCE_FEASIBILITY.json').exists()
    for path in (Path(__file__).resolve(), Path(T.__file__).resolve(), Path(T.OLD.__file__).resolve()):
        assert C.bind(path) in protocol['codes'], ('UNBOUND_SOURCE_OPERATOR', str(path))
    parent = T.OLD.load_training_inputs()
    assert protocol['inputs']['parent_train_protocol'] == parent['binding']
    for key in ('features', 'train_labels', 'source_contract', 'train_gate'):
        assert protocol['inputs'][key] == parent['protocol']['inputs'][key]
    anchor = T.load_anchor(parent)
    np.testing.assert_array_equal(parent['scale'], SCALE)
    assert len(parent['ids']) == 2598 and len(set(parent['ids'].tolist())) == 2598
    assert int(anchor['finite'].sum()) == 2597 and int((~anchor['finite']).sum()) == 1
    baseline = operational(parent, T.OLD)
    np.testing.assert_array_equal(baseline['R0'], anchor['errors'])
    outputs, selected_errors, choices = {}, {}, {}
    for model in MODELS:
        errors, valid, names = pool(parent, model, T.OLD)
        error_sha, valid_sha = T.OLD.array_sha(errors), T.OLD.array_sha(valid)
        expected_target, expected_safe = independent_targets(errors, valid, anchor['errors'],
            anchor['index'], parent['scale'], names)
        target, safe = T.anchored_targets(errors, valid, anchor['errors'], anchor['index'], parent['scale'], names)
        np.testing.assert_array_equal(target, expected_target)
        np.testing.assert_array_equal(safe, expected_safe)
        assert target.dtype == np.int64 and safe.dtype == bool
        assert T.OLD.array_sha(errors) == error_sha and T.OLD.array_sha(valid) == valid_sha
        assert np.array_equal(target >= 0, valid.any(1))
        picked = np.full((len(errors), 2), np.inf, np.float64)
        available = target >= 0
        picked[available] = errors[np.flatnonzero(available), target[available]]
        assert np.array_equal(available, anchor['finite'])
        assert np.all(picked[available] <= anchor['errors'][available])
        assert np.isposinf(picked[~available]).all()
        assert int((~available).sum()) == 1
        selected_errors[model] = picked
        choices[model] = dict(Counter(names[j] if j >= 0 else 'FAILED' for j in target))
        outputs[model] = dict(candidate_names=names, target_sha=T.OLD.array_sha(target),
            safe_mask_sha=T.OLD.array_sha(safe), anchor_errors_sha=anchor['errors_sha'],
            anchor_index_sha=anchor['index_sha'], original_valid_sha=valid_sha,
            unscaled_errors_sha=error_sha, target_dtype='int64', safe_mask_dtype='bool',
            source_rows=len(errors), candidate_count=errors.shape[1],
            original_valid_candidates=int(valid.sum()), safe_candidates=int(safe.sum()),
            safe_count_histogram={str(k): int(v) for k, v in Counter(safe.sum(1).tolist()).items()},
            full_population_summary=summary(picked), selected_errors_sha=T.OLD.array_sha(picked),
            choices=choices[model], independent_target_safe_parity=True,
            input_error_and_valid_hashes_unchanged=True, selected_T_R_each_nonincrease_vs_R0_anchor=True,
            geometric_valid_mask_used_for_later_CE_and_inference=True)
    comparisons, failed = {}, []
    for seed in (1, 2, 3):
        model, diverse = f'UNION_s{seed}', f'DIVERSE251_s{seed}'
        controls = {'R0_ONLY': selected_errors['R0_ONLY'], 'R0_GEO': baseline['R0'],
                    diverse + '_GEO': baseline[diverse]}
        comparisons[model] = {}
        for control, before in controls.items():
            result = gate(before, selected_errors[model])
            comparisons[model][control] = result
            failed.extend(f'{model}/{control}/{criterion}' for criterion, passed in result['checks'].items() if not passed)
    passed = not failed
    out = dict(complete=True, PASS=passed, status='PASS_SOURCE_ANCHORED_FEASIBILITY' if passed else
        'STOP_SOURCE_ANCHORED_FEASIBILITY', created_at=C.now(), source_TRAIN_only=True,
        scope='C2_TRAIN_ONLY', frames=2598, models=outputs,
        protocol=C.bind(C.DOC / 'SOURCE_PROTOCOL.json'), parent_train_protocol=parent['binding'],
        input_bindings=protocol['inputs'], source_ids_sha=T.OLD.array_sha(parent['ids']),
        source_index_sha=T.OLD.array_sha(parent['source_index']), scale=parent['scale'].tolist(),
        anchor=dict(errors_sha=anchor['errors_sha'], index_sha=anchor['index_sha'],
                    finite_sha=anchor['finite_sha'], valid_rows=2597, invalid_rows=1,
                    failed_ids=parent['ids'][~anchor['finite']].tolist(), bindings=anchor['bindings']),
        operational={model: summary(value) for model, value in baseline.items()},
        comparisons=comparisons, checks_total=45, checks_passed=45-len(failed), failed_checks=failed,
        gate_rule='Every UNION seed: strict full-population T/R medians; both P90<=1.05; no more failures, versus newly anchored R0_ONLY oracle, old R0_GEO and paired DIVERSE_GEO.',
        target_rule='Source-only exact T<=R0 operational T AND R<=R0 operational R, then fixed TRAIN minmax/Pareto/R0/hypothesis/expert tie. Original geometric-valid CE competitors retained.',
        failure_policy='All2598 rows retained. Original all-invalid row stays failed/+infinity in both axes; no resampling or zero-error substitution.',
        diagnostic_only=True, inference_use=False, labels_rewritten=False,
        VAL_quality_read=False, real_targets_read=False, real_reference_reads=0, raw_source_reference_reads=0,
        fits=0, image_forwards=0, new_PnP_calls=0, optimizer_steps=0,
        code=C.bind(Path(__file__)), read_paths=sorted(set(T.OLD.READS)),
        wall_seconds=time.monotonic()-started, method_success=False, goal_complete=False)
    C.save(C.DOC / 'SOURCE_FEASIBILITY.json', out)
    print('SOURCE_ANCHOR_FEASIBILITY', dict(PASS=passed, checks_passed=out['checks_passed'],
        checks_total=45, finite_anchor=2597, invalid_anchor=1), flush=True)


def selfcheck():
    from . import convex_train as T
    names = ['R0:long-face-front', 'R0:short-face-front',
             'DIVERSE251_s1:long-face-front', 'DIVERSE251_s1:short-face-front']
    errors = np.array([[[2, 2], [1, 3], [1, 1], [3, 1]],
        [[5, 5], [2, 6], [4, 4], [5, 1]],
        [[2, 2], [np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf]],
        [[np.inf, np.inf]] * 4,
        [[5, 2], [5, 1], [5, 1], [5, 1]]], np.float64)
    valid = np.isfinite(errors).all(2)
    anchor = errors[:, 0].copy()
    index = np.array([0, 0, 0, -1, 0], np.int64)
    original_errors, original_valid = errors.copy(), valid.copy()
    target, safe = T.anchored_targets(errors, valid, anchor, index, np.ones(2), names)
    independent, independent_safe = independent_targets(errors, valid, anchor, index, np.ones(2), names)
    np.testing.assert_array_equal(target, independent)
    np.testing.assert_array_equal(safe, independent_safe)
    np.testing.assert_array_equal(target, [2, 2, 0, -1, 1])
    np.testing.assert_array_equal(errors, original_errors)
    np.testing.assert_array_equal(valid, original_valid)
    assert not safe[0, 1] and valid[0, 1]
    # Unsafe candidates remain runtime competitors: source target safety is not a learned guarantee.
    scores = np.zeros(valid.shape)
    scores[0, 1] = -1
    assert T.OLD.select_candidates(scores, valid, names)[0] == 1
    test = np.array([[1., 2.], [3., 4.], [np.inf, np.inf]])
    assert quantile(test[:, 0], .5) == 3 and np.isposinf(quantile(test[:, 0], .9))
    assert summary(test)['failed_pose'] == 1
    assert not gate(np.array([[1., 1.]]), np.array([[1., 1.]]))['PASS']
    assert not gate(np.array([[1., 1.]]), np.array([[np.inf, np.inf]]))['PASS']
    assert T.OLD.READS is None, 'Selfcheck must not open artifacts or install the data guard.'
    print('SOURCE_FEASIBILITY_SELFCHECK_PASS: invented exact safety, Pareto ties, one/all invalid, original competitors, full failure denominator; no source/VAL/real artifacts read.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('selfcheck', 'score'))
    arguments = parser.parse_args()
    (selfcheck if arguments.stage == 'selfcheck' else score)()
