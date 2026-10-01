"""Diagnostic feasibility of frozen real candidate unions; never a method.

Only cached error tables and metadata are read. No image, learned checkpoint,
raw reference, inference, fitting, or PnP execution is permitted. The CLI
selfcheck imports neither experiment common nor historical evaluation modules.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

DIAGNOSTICS = ('AXIS_LOWER_BOUND', 'FIXED_TRAIN_COST_ORACLE')
SEEDS = (1, 2, 3)
HYPOTHESES = ('long-face-front', 'short-face-front')
SCALE = (2.4636887551191258, 1.113474019956766)
METRICS = ('translation_cm', 'rotation_deg')
MODELS = ('R0', 'PRIOR1', 'FULL125', 'SINGLE251_s1', 'SINGLE251_s2', 'SINGLE251_s3',
          'DIVERSE251_s1', 'DIVERSE251_s2', 'DIVERSE251_s3')


def tie_key(candidate):
    return candidate['model'] != 'R0', candidate['hypothesis'], candidate['model']


def nondominated(pool):
    """Remove strict Pareto dominance only; retain equal-error identities."""
    return [p for p in pool if not any(
        q['T_cm'] <= p['T_cm'] and q['R_deg'] <= p['R_deg']
        and (q['T_cm'] < p['T_cm'] or q['R_deg'] < p['R_deg']) for q in pool)]


def deduplicate(pool):
    unique = {}
    for p in pool:
        key = p['model'], p['hypothesis']
        assert p['hypothesis'] in HYPOTHESES
        assert np.isfinite([p['T_cm'], p['R_deg']]).all()
        assert p['T_cm'] >= 0 and p['R_deg'] >= 0
        if key in unique:
            assert (p['T_cm'], p['R_deg']) == (unique[key]['T_cm'], unique[key]['R_deg']), ('DUPLICATE_POSE_ERROR_MISMATCH', key)
        else:
            unique[key] = dict(p)
    return sorted(unique.values(), key=tie_key)


def choose(pool, diagnostic, scale=SCALE):
    assert diagnostic in DIAGNOSTICS
    scale = np.asarray(scale, np.float64)
    assert scale.shape == (2,) and np.isfinite(scale).all() and (scale > 0).all()
    pool = nondominated(deduplicate(pool))
    if not pool:
        return dict(available=False, T_cm=None, R_deg=None, chosen=None,
                    T_source=None, R_source=None, cost=None, candidate_count=0)
    t = min(pool, key=lambda p: (p['T_cm'], tie_key(p)))
    r = min(pool, key=lambda p: (p['R_deg'], tie_key(p)))
    if diagnostic == 'AXIS_LOWER_BOUND':
        return dict(available=True, T_cm=t['T_cm'], R_deg=r['R_deg'], chosen=None,
                    T_source=t, R_source=r, cost=None, candidate_count=len(pool))
    costs = np.array([max(p['T_cm'] / scale[0], p['R_deg'] / scale[1]) for p in pool])
    assert np.isfinite(costs).all()
    ties = [p for p, c in zip(pool, costs) if c == costs.min()]
    picked = min(nondominated(ties), key=tie_key)
    return dict(available=True, T_cm=picked['T_cm'], R_deg=picked['R_deg'], chosen=picked,
                T_source=picked, R_source=picked, cost=float(costs.min()), candidate_count=len(pool))


def metric(fid, selected):
    if not selected['available']:
        return dict(id=fid, available=False, translation_cm=None, rotation_deg=None)
    return dict(id=fid, available=True, translation_cm=selected['T_cm'], rotation_deg=selected['R_deg'])


def finite_metric(value):
    assert isinstance(value['available'], bool)
    if not value['available']:
        return False
    pair = [value[k] for k in METRICS]
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in pair)
    assert np.isfinite(pair).all() and min(pair) >= 0
    return True


def compare_tree(actual, expected, atol=1e-12, path='root'):
    """Exact schema/boolean parity; bounded roundoff for historical floats."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        return sum(compare_tree(actual[k], v, atol, path + '.' + k) for k, v in expected.items())
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        return sum(compare_tree(a, b, atol, path + f'[{i}]') for i, (a, b) in enumerate(zip(actual, expected)))
    if isinstance(expected, float):
        assert isinstance(actual, (int, float)) and np.isfinite([actual, expected]).all(), path
        assert abs(actual - expected) <= atol, (path, actual, expected)
        return 1
    assert actual == expected and type(actual) is type(expected), (path, actual, expected)
    return 1


def guard_inputs(C):
    """Deny repository data outside explicitly sealed cached input paths."""
    protocol_path = C.DOC / 'PROTOCOL.json'
    seal_path = C.DOC / 'PROTOCOL_SHA.json'
    seal = C.read(seal_path)
    C.verify(seal)
    assert seal == C.bind(protocol_path)
    preliminary = C.read(protocol_path)
    allowed = {str((C.ROOT / b['path']).resolve()) for b in preliminary['inputs'].values()}
    allowed.update((str(protocol_path.resolve()), str(seal_path.resolve())))
    reads = []
    forbidden = ('/annotations/', 'GEOMETRY_RESOLVED_POSE_GT', 'TRUTH_FOR_DISPLAY',
        'AXIS_REVIEW_MANIFEST', '/real_gt_v2/', 'GEOMETRY_SIDETABLE', '/SOURCE_MANIFEST.json')

    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        s = str(p)
        assert not any(token in s for token in forbidden), ('RAW_REFERENCE_FORBIDDEN', s)
        assert p.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp', '.pt', '.pth', '.onnx'), ('IMAGE_OR_CHECKPOINT_FORBIDDEN', s)
        # External software/font caches are not experiment data. Dataset reads
        # remain globally denied above; repository files use a strict allowlist.
        if not p.is_relative_to(C.ROOT):
            return
        mode = args[1]
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = ((isinstance(mode, str) and any(k in mode for k in ('w', 'a', 'x', '+')))
                   or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        if writing:
            assert p.is_relative_to(C.DOC) or p.is_relative_to(C.RAW), ('DIAGNOSTIC_WRITE_SCOPE', s)
            return
        if p.suffix in ('.py', '.pyc'):
            return
        assert s in allowed or p.is_relative_to(C.DOC) or p.is_relative_to(C.RAW), ('UNBOUND_REPOSITORY_DATA_READ', s)
        reads.append(str(p.relative_to(C.ROOT)))

    sys.addaudithook(hook)
    sys.dont_write_bytecode = True
    p = C.protocol()
    assert p == preliminary
    return p, reads


def load_cached(C, protocol):
    required = {'candidate_rows', 'candidate_bounds', 'operational_metrics', 'eval_metadata', 'eval_groups',
        'stable_protocol', 'stable_results', 'source_train_gate', 'parent_publication',
        'stable_pose_candidates', 'historical_metadata'}
    assert required.issubset(protocol['inputs'])
    assert protocol['scale'] == list(SCALE)
    assert protocol['diagnostics'] == list(DIAGNOSTICS) and protocol['seeds'] == list(SEEDS)
    assert not protocol['method_success'] and not protocol['goal_complete']
    inp = protocol['inputs']
    for b in inp.values():
        C.verify(b)
    values = {key: C.read(C.ROOT / b['path']) for key, b in inp.items() if key != 'candidate_rows'}
    bound = values['candidate_bounds']
    assert inp['candidate_rows'] in bound['artifacts']
    assert bound['original_primary_result'] == inp['stable_results']
    assert values['input_audit']['complete'] and values['input_audit']['PASS']
    assert values['stable_protocol_sha'] == inp['stable_protocol']
    assert bound['pose_lock'] == inp['stable_pose_lock']
    assert inp['stable_pose_candidates'] in values['stable_pose_lock']['files']
    assert inp['eval_groups'] in values['stable_pose_lock']['files']
    assert values['stable_pose_lock']['prediction_lock'] == inp['stable_prediction_lock']
    assert values['stable_prediction_lock']['metadata'] == inp['eval_metadata']
    assert inp['operational_metrics'] in values['stable_results']['artifacts']
    publication = {b['path']: b for b in values['parent_publication']['files']}
    for key in ('candidate_rows', 'candidate_bounds'):
        assert publication.get(inp[key]['path']) == inp[key]
    source_scale = values['source_train_gate']['source_scale']
    assert source_scale['valid']
    assert [source_scale['sT_cm'], source_scale['sR_deg']] == list(SCALE)
    rows = values['eval_metadata']
    assert isinstance(rows, list) and len(rows) == 173
    ids = [row['id'] for row in rows]
    assert len(set(ids)) == 173
    meta = {r['id']: r for r in rows}
    operations = values['operational_metrics']
    poses = values['stable_pose_candidates']
    assert set(operations) == set(poses) == set(MODELS)
    original_available = {}
    for model in MODELS:
        assert set(operations[model]) == set(poses[model]) == set(ids)
        original_available[model] = sum(finite_metric(operations[model][fid]) for fid in ids)
        for fid in ids:
            hypotheses = poses[model][fid]['hypotheses']
            assert len(hypotheses) == 2 and {h['name'] for h in hypotheses} == set(HYPOTHESES)
            assert all(isinstance(h['pose']['available'], bool) for h in hypotheses)
    with (C.ROOT / inp['candidate_rows']['path']).open(newline='') as f:
        records = list(csv.DictReader(f))
    assert len(records) == 9 * 173 * 2
    indexed = {}
    for row in records:
        model, fid, axis = row['model'], row['id'], row['oracle']
        assert model in MODELS and fid in meta and axis in ('T_best', 'R_best')
        assert row['recording'] == meta[fid]['recording'] and row['diagnostic_only'] == 'True'
        key = model, fid, axis
        assert key not in indexed
        available = bool(row['selected_whole_pose'])
        if available:
            hyp = row['selected_whole_pose']
            assert hyp in HYPOTHESES
            assert next(h['pose']['available'] for h in poses[model][fid]['hypotheses'] if h['name'] == hyp)
            candidate = dict(model=model, hypothesis=hyp, T_cm=float(row['T_cm']), R_deg=float(row['R_deg']))
            deduplicate([candidate])  # finite, nonnegative metric validation
        else:
            assert row['T_cm'] == row['R_deg'] == ''
            assert not any(h['pose']['available'] for h in poses[model][fid]['hypotheses'])
            candidate = None
        indexed[key] = candidate
    per_model = {}
    for model in MODELS:
        per_model[model] = {}
        for fid in ids:
            a, b = indexed[model, fid, 'T_best'], indexed[model, fid, 'R_best']
            assert (a is None) == (b is None)
            if a is not None:
                assert a['T_cm'] <= b['T_cm'] and b['R_deg'] <= a['R_deg']
            pool = deduplicate([p for p in (a, b) if p is not None])
            assert len(pool) <= 2
            per_model[model][fid] = pool
    return values, rows, per_model, original_available


def trace_row(diagnostic, seed, row, selected, original_count):
    def value(source, key):
        return selected[source][key] if selected[source] else None
    return dict(diagnostic=diagnostic, seed=seed, id=row['id'], recording=row['recording'],
        available=selected['available'], T_cm=selected['T_cm'], R_deg=selected['R_deg'],
        selected_model=value('chosen', 'model'), selected_hypothesis=value('chosen', 'hypothesis'),
        T_source_model=value('T_source', 'model'), T_source_hypothesis=value('T_source', 'hypothesis'),
        R_source_model=value('R_source', 'model'), R_source_hypothesis=value('R_source', 'hypothesis'),
        physical_complete_pose=diagnostic == 'FIXED_TRAIN_COST_ORACLE' and selected['available'],
        GT_derived_diagnostic_only=True, fixed_TRAIN_cost=selected['cost'],
        deduplicated_union_candidates=original_count, nondominated_union_candidates=selected['candidate_count'])


def summarize(E, metrics, rows, populations):
    groups = dict(populations, ALL173=[r['id'] for r in rows])
    summary, seeds = {}, {}
    arm_names = {'SINGLE251': [f'SINGLE251_s{s}' for s in SEEDS],
                 'DIVERSE251': [f'DIVERSE251_s{s}' for s in SEEDS]}
    arm_names.update({name: [name] * 3 for name in ('R0', 'PRIOR1', 'FULL125')})
    for population, ids in groups.items():
        summary[population] = {name: E.mean_seed_summary(E.error_tensor(metrics, models, ids))
                               for name, models in arm_names.items()}
        seeds[population] = {name: E.mean_seed_summary(E.error_tensor(metrics, [name], ids)) for name in MODELS}
    metadata = {r['id']: r for r in rows}
    recordings = {}
    for population in ('NATURAL99', 'CLEAN29', 'WOOD45'):
        recordings[population] = {}
        for rec in sorted({metadata[fid]['recording'] for fid in groups[population]}):
            ids = [fid for fid in groups[population] if metadata[fid]['recording'] == rec]
            recordings[population][rec] = dict(frames=len(ids), summaries={name:
                E.mean_seed_summary(E.error_tensor(metrics, models, ids)) for name, models in arm_names.items()})
    return summary, seeds, recordings


def run():
    # Keep imports inside execution: selfcheck cannot transitively open real files.
    from . import common as C
    started = time.monotonic()
    protocol, reads = guard_inputs(C)
    values, rows, pools, original_available = load_cached(C, protocol)
    from scripts.research.pallet_pose_stable_improvement_20261001_v1 import evaluate as E
    assert E.C.ARMS == ('SINGLE251', 'DIVERSE251') and E.C.SEEDS == SEEDS
    assert E.BOOT_REPEATS == 2000 and E.BOOT_SEED == 20261001 and E.GUARD_RATIO == 1.05
    for path in (Path(__file__), Path(C.__file__), Path(E.__file__), Path(E.C.__file__), Path(E.C.D.__file__), Path(E.C.D.M.__file__)):
        assert C.bind(path) in protocol['codes'], ('UNBOUND_DIAGNOSTIC_OPERATOR', str(path))
    assert set(E.model_names()) == set(MODELS)
    populations = E.groups(rows)
    assert populations == values['eval_groups']
    assert [r['id'] for r in values['historical_metadata']] == populations['FULL128']
    original_metrics = values['operational_metrics']
    historical, historical_hierarchy = E.gate_results(original_metrics, rows, populations, values['stable_protocol'])
    stable = values['stable_results']
    assert stable['complete'] and stable['models'] == E.model_names()
    parity_leaves = compare_tree(historical, stable['stability'])
    parity_leaves += compare_tree(historical_hierarchy, stable['hierarchy'])
    all_original_finite = all(n == 173 for n in original_available.values())
    outputs, traces, selected_error_cache = {}, [], {}
    for diagnostic in DIAGNOSTICS:
        metrics = dict(original_metrics)
        counts = Counter()
        selected_error_cache[diagnostic] = {}
        for seed in SEEDS:
            model = f'DIVERSE251_s{seed}'
            chosen = {}
            selected_error_cache[diagnostic][model] = {}
            for row in rows:
                fid = row['id']
                pool = deduplicate(pools['R0'][fid] + pools[model][fid])
                selected = choose(pool, diagnostic, protocol['scale'])
                chosen[fid] = metric(fid, selected)
                selected_error_cache[diagnostic][model][fid] = selected
                counts[f"deduplicated_{len(pool)}"] += 1
                counts[f"nondominated_{selected['candidate_count']}"] += 1
                traces.append(trace_row(diagnostic, seed, row, selected, len(pool)))
            metrics[model] = chosen
        # Baselines are immutable references to the original metrics dictionary.
        assert all(metrics[m] is original_metrics[m] for m in MODELS if not m.startswith('DIVERSE251_'))
        gate, hierarchy = E.gate_results(metrics, rows, populations, values['stable_protocol'])
        gate['original_gate_verdict'] = gate['verdict']
        gate['verdict'] = ('DIAGNOSTIC_GATES_PASS_NOT_METHOD_SUCCESS' if gate['PASS']
                           else 'DIAGNOSTIC_GATES_FAIL_NOT_METHOD_VERDICT')
        gate.update(diagnostic_only=True, method_success=False, goal_complete=False,
                    choice_uses_cached_GT_derived_errors=True)
        summaries, per_seed, by_recording = summarize(E, metrics, rows, populations)
        finite = all(finite_metric(metrics[f'DIVERSE251_s{s}'][r['id']]) for s in SEEDS for r in rows)
        outputs[diagnostic] = dict(summaries=summaries, per_seed_summaries=per_seed,
            by_recording=by_recording, stability=gate, hierarchy=hierarchy,
            union_rows=3*173, all_union_errors_finite=finite, candidate_counts=dict(counts),
            diagnostic_slot_aliases={'DIVERSE251': diagnostic,
                **{f'DIVERSE251_s{s}': f'{diagnostic}_s{s}' for s in SEEDS}},
            slot_warning='DIVERSE251 keys are unchanged evaluator slots replaced by this GT-derived diagnostic. They are not new measurements of the learned DIVERSE251 model.',
            physical_complete_pose=diagnostic == 'FIXED_TRAIN_COST_ORACLE',
            diagnostic_only=True, method_success=False, goal_complete=False)
    # Failure-safe pointwise lower bound check, without mixing pose sources.
    for seed in SEEDS:
        model = f'DIVERSE251_s{seed}'
        for row in rows:
            fid = row['id']
            bound = selected_error_cache['AXIS_LOWER_BOUND'][model][fid]
            oracle = selected_error_cache['FIXED_TRAIN_COST_ORACLE'][model][fid]
            assert bound['available'] == oracle['available']
            if bound['available']:
                assert bound['T_cm'] <= oracle['T_cm'] and bound['R_deg'] <= oracle['R_deg']
    lower, oracle = [outputs[d] for d in DIAGNOSTICS]
    finite_contract = all_original_finite and lower['all_union_errors_finite'] and oracle['all_union_errors_finite']
    if finite_contract:
        for key, value in oracle['stability']['gates'].items():
            assert not value['PASS'] or lower['stability']['gates'][key]['PASS'], ('MONOTONE_GATE_VIOLATION', key)
    lower['interpretation'] = dict(
        necessary_exclusion_logic_valid=finite_contract,
        fixed_union_goal_excluded=bool(finite_contract and not lower['stability']['PASS']),
        PASS_is_joint_feasibility_proof=False,
        reason='Independent T and R minima may come from different complete poses. With all reference and union rows finite, the original zero-failure guards prevent censoring; each original gate is monotone under componentwise lower errors and identical bootstrap draws. Only then a failed lower-bound gate excludes every selector restricted to this fixed union.',
        conditional_failure_caveat='If finite support fails, failure-inclusive tensors remain intact but the aggregate FAIL is not asserted to be a necessary impossibility result.')
    oracle['interpretation'] = dict(
        gate_passing_whole_pose_witness=bool(oracle['stability']['PASS']),
        FAIL_excludes_all_whole_pose_selections=False,
        deployable=False,
        reason='One whole pose is selected per frame using cached GT-derived errors and the pre-existing TRAIN scales. PASS witnesses feasibility of these frozen candidates under the old DEV gates, not a learnable or deployable rule. FAIL of this single fixed cost does not exclude all other choices.')
    assert len(traces) == 2*3*173 == 1038
    frame_path = C.DOC / 'FRAME_RESULTS.csv'
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fieldnames=list(traces[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(traces)
    C.save(frame_path, handle.getvalue())
    result = dict(complete=True, created_at=datetime.now(timezone.utc).isoformat(), diagnostic_only=True,
        method_success=False, goal_complete=False, protocol=C.bind(C.DOC / 'PROTOCOL.json'),
        inputs=protocol['inputs'], codes=protocol['codes'], scale=list(SCALE),
        candidate_contract='Exactly two frozen hypotheses per expert. Cached T-best/R-best rows retain every potentially useful nondominated candidate for positive monotone cost and axis minima; duplicate identities require exact T/R agreement. Distinct experts remain distinct candidates.',
        historical_gate_parity=dict(PASS=True, scalar_leaves_checked=parity_leaves, float_atol=1e-12,
            original_stability=historical, full_hierarchy_checked=True),
        finite_contract=dict(all_original_operational_rows_finite=all_original_finite,
            original_available_by_model=original_available, all_union_rows_finite=lower['all_union_errors_finite'],
            necessary_exclusion_logic_valid=finite_contract),
        populations={k: len(v) for k, v in dict(populations, ALL173=rows).items()},
        diagnostics=outputs, full_frame_rows=len(traces), artifacts=[C.bind(frame_path)],
        read_paths=sorted(set(reads)), image_forwards=0, new_fits=0, optimizer_steps=0,
        new_PnP_calls=0, raw_GT_reads=0, learned_checkpoints_read=0,
        GT_derived_cached_errors_read=True,
        row_failure_policy='Every frame and seed retained. Missing whole-pose candidate has both errors unavailable; unchanged stable error_tensor maps both to+inf and retains failure counts.',
        summary_contract='Means of per-seed quantiles; fixed baselines repeated across three paired seeds. ALL173 is descriptive; original gates use unchanged NATURAL99/CLEAN29 and report WOOD45.',
        unchanged_primary_artifacts=True, wall_seconds=time.monotonic()-started)
    C.save(C.DOC / 'RESULTS.json', result)
    print('REAL_UNION_DIAGNOSTICS_COMPLETE_NOT_METHOD_SUCCESS',
        {d: outputs[d]['stability']['PASS'] for d in DIAGNOSTICS}, flush=True)


def selfcheck():
    def p(model, hyp, t, r):
        return dict(model=model, hypothesis=hyp, T_cm=float(t), R_deg=float(r))
    a = p('R0', HYPOTHESES[0], 1, 9)
    b = p('DIVERSE251_s1', HYPOTHESES[1], 9, 1)
    bound = choose([a, b], 'AXIS_LOWER_BOUND', (1., 1.))
    whole = choose([a, b], 'FIXED_TRAIN_COST_ORACLE', (1., 1.))
    assert (bound['T_cm'], bound['R_deg']) == (1., 1.) and bound['chosen'] is None
    assert bound['T_source'] == a and bound['R_source'] == b
    assert whole['chosen'] == a and (whole['T_cm'], whole['R_deg']) == (1., 9.)
    # Exact scalar-cost tie with Pareto dominance must overrule R0 priority.
    dominated = p('R0', HYPOTHESES[0], 5, 2)
    dominating = p('DIVERSE251_s1', HYPOTHESES[1], 5, 1)
    assert choose([dominated, dominating], 'FIXED_TRAIN_COST_ORACLE', (1., 1.))['chosen'] == dominating
    same_hyp_other = p('DIVERSE251_s1', HYPOTHESES[0], 1, 9)
    assert len(deduplicate([a, a, same_hyp_other])) == 2
    try:
        deduplicate([a, dict(a, T_cm=2.)])
    except AssertionError:
        pass
    else:
        raise AssertionError('Mismatched duplicate must fail')
    for diagnostic in DIAGNOSTICS:
        failed = choose([], diagnostic)
        assert not failed['available'] and not metric('invented', failed)['available']
        assert failed['T_cm'] is failed['R_deg'] is None
    # Every binary candidate subset reproduces full-pool minima and fixed-cost
    # selection after reduction to its separate axis-optimal whole poses.
    rng = np.random.default_rng(61001)
    lows, selected = [], []
    for _ in range(100):
        union, reduced = [], []
        for model in ('R0', 'DIVERSE251_s1'):
            pool = [p(model, h, *rng.uniform(.1, 20, size=2)) for h in HYPOTHESES]
            union.extend(pool)
            reduced.extend([min(pool, key=lambda q: (q['T_cm'], q['hypothesis'])),
                            min(pool, key=lambda q: (q['R_deg'], q['hypothesis']))])
        for diagnostic in DIAGNOSTICS:
            full, short = choose(union, diagnostic), choose(reduced, diagnostic)
            for key in ('T_cm', 'R_deg', 'chosen', 'T_source', 'R_source'):
                assert full[key] == short[key]
        lb, oracle = choose(union, 'AXIS_LOWER_BOUND'), choose(union, 'FIXED_TRAIN_COST_ORACLE')
        lows.append([lb['T_cm'], lb['R_deg']])
        selected.append([oracle['T_cm'], oracle['R_deg']])
    lows, selected = np.asarray(lows), np.asarray(selected)
    assert (lows <= selected).all()
    for q in (.5, .9):
        assert (np.quantile(lows, q, axis=0) <= np.quantile(selected, q, axis=0)).all()
    assert 'scripts.research.pallet_pose_stable_improvement_20261001_v1.evaluate' not in sys.modules
    assert 'scripts.research.pallet_pose_real_union_feasibility_20261001_v1.common' not in sys.modules
    print('REAL_UNION_FEASIBILITY_SELFCHECK_PASS: invented pose mixture, exact tie/Pareto, identity dedup, failures, two-hypothesis reduction and quantile monotonicity. No real files read.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['selfcheck', 'score'])
    args = parser.parse_args()
    (selfcheck if args.stage == 'selfcheck' else run)()
