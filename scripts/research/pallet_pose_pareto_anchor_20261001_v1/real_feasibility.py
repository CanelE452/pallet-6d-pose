"""GT-derived Pareto-anchor feasibility diagnostic, never a runtime selector.

Only the admissible set changes: retain complete poses with both cached errors
no larger than the operational R0 GEO anchor. The anchor is always included.
"""
import argparse
from collections import Counter
import csv
import io
from pathlib import Path
import sys
import time

import numpy as np

from scripts.research.pallet_pose_real_union_feasibility_20261001_v1 import evaluate as P

DIAGNOSTIC = 'PARETO_ANCHOR_ORACLE'


def anchored_choice(compressed_pool, anchor, scale=P.SCALE):
    """Pure function: restore the anchor, enforce both inequalities, choose."""
    assert anchor['model'] == 'R0'
    original = P.deduplicate(compressed_pool)
    anchor_id = anchor['model'], anchor['hypothesis']
    restored = not any((p['model'], p['hypothesis']) == anchor_id for p in original)
    # This also enforces exact error parity when the anchor identity was cached.
    full = P.deduplicate([*original, anchor])
    assert 1 <= len(full) <= 4
    admissible = [p for p in full if p['T_cm'] <= anchor['T_cm'] and p['R_deg'] <= anchor['R_deg']]
    assert anchor in admissible
    selected = P.choose(admissible, 'FIXED_TRAIN_COST_ORACLE', scale)
    assert selected['available'] and selected['chosen'] is not None
    assert selected['T_cm'] <= anchor['T_cm'] and selected['R_deg'] <= anchor['R_deg']
    returned_anchor = (selected['chosen']['model'], selected['chosen']['hypothesis']) == anchor_id
    counts = dict(compressed_candidates=len(original), anchor_augmented_candidates=len(full),
                  eligible_candidates=len(admissible), rejected_candidates=len(full)-len(admissible),
                  anchor_restored=restored, anchor_returned=returned_anchor,
                  anchor_only_eligible=len(admissible) == 1,
                  strictly_dominating_eligible=sum(p['T_cm'] < anchor['T_cm'] or p['R_deg'] < anchor['R_deg'] for p in admissible),
                  T_strict_improvement=selected['T_cm'] < anchor['T_cm'],
                  R_strict_improvement=selected['R_deg'] < anchor['R_deg'],
                  both_strict_improvement=selected['T_cm'] < anchor['T_cm'] and selected['R_deg'] < anchor['R_deg'])
    return selected, counts


def load_inputs(C, protocol):
    assert protocol['diagnostics'] == [DIAGNOSTIC]
    assert protocol['seeds'] == list(P.SEEDS) and protocol['scale'] == list(P.SCALE)
    assert not protocol['method_success'] and not protocol['goal_complete']
    for name in ('previous_feasibility', 'previous_protocol'):
        C.verify(protocol['inputs'][name])
    previous_protocol = C.read(C.ROOT / protocol['inputs']['previous_protocol']['path'])
    previous = C.read(C.ROOT / protocol['inputs']['previous_feasibility']['path'])
    assert previous['complete'] and previous['diagnostic_only']
    assert not previous['method_success'] and not previous['goal_complete']
    assert previous['protocol'] == protocol['inputs']['previous_protocol']
    assert previous['inputs'] == previous_protocol['inputs']
    for name, binding in previous_protocol['inputs'].items():
        assert protocol['inputs'][name] == binding, ('CHANGED_FROZEN_INPUT', name)
    # Authenticate the unmodified previous loading contract. It computes no
    # diagnostic outcome and reads the same explicitly bound caches only.
    values, rows, pools, availability = P.load_cached(C, previous_protocol)
    assert all(n == 173 for n in availability.values()), 'Anchor proof requires full finite support'
    old = previous['diagnostics']['FIXED_TRAIN_COST_ORACLE']
    assert not old['stability']['PASS'], 'This protocol follows the frozen cost-oracle failure'
    return values, rows, pools, previous


def run():
    from . import common as C
    started = time.monotonic()
    protocol, reads = P.guard_inputs(C)
    values, rows, pools, previous = load_inputs(C, protocol)
    from scripts.research.pallet_pose_stable_improvement_20261001_v1 import evaluate as E
    assert E.C.ARMS == ('SINGLE251', 'DIVERSE251') and E.C.SEEDS == P.SEEDS
    assert E.BOOT_REPEATS == 2000 and E.BOOT_SEED == 20261001 and E.GUARD_RATIO == 1.05
    for module in (sys.modules[__name__], C, P, E, E.C, E.C.D, E.C.D.M):
        assert C.bind(Path(module.__file__)) in protocol['codes'], ('UNBOUND_OPERATOR', module.__file__)
    populations = E.groups(rows)
    assert populations == values['eval_groups']
    assert [r['id'] for r in values['historical_metadata']] == populations['FULL128']
    operations = values['operational_metrics']
    historical, historical_hierarchy = E.gate_results(operations, rows, populations, values['stable_protocol'])
    parity = P.compare_tree(historical, values['stable_results']['stability'])
    parity += P.compare_tree(historical_hierarchy, values['stable_results']['hierarchy'])
    metrics = dict(operations)
    traces, by_seed, overall = [], {}, Counter()
    for seed in P.SEEDS:
        model = f'DIVERSE251_s{seed}'
        chosen, tally = {}, Counter()
        for row in rows:
            fid = row['id']
            r0 = values['stable_pose_candidates']['R0'][fid]
            hypothesis = r0['GEO_name']
            assert hypothesis in P.HYPOTHESES
            matching = [h['pose'] for h in r0['hypotheses'] if h['name'] == hypothesis]
            assert len(matching) == 1 and matching[0]['available']
            assert r0['GEO_pose'] == matching[0], 'Operational anchor must be the named frozen whole pose'
            op = operations['R0'][fid]
            assert P.finite_metric(op)
            anchor = dict(model='R0', hypothesis=hypothesis, T_cm=op['translation_cm'], R_deg=op['rotation_deg'])
            pool = P.deduplicate(pools['R0'][fid] + pools[model][fid])
            selected, counts = anchored_choice(pool, anchor, protocol['scale'])
            chosen[fid] = P.metric(fid, selected)
            for key in ('anchor_restored', 'anchor_returned', 'anchor_only_eligible',
                        'T_strict_improvement', 'R_strict_improvement', 'both_strict_improvement'):
                tally[key] += int(counts[key])
            for key in ('eligible_candidates', 'rejected_candidates', 'anchor_augmented_candidates', 'strictly_dominating_eligible'):
                tally[f'{key}:{counts[key]}'] += 1
            tally['frames'] += 1
            tally['T_nonincrease'] += int(selected['T_cm'] <= anchor['T_cm'])
            tally['R_nonincrease'] += int(selected['R_deg'] <= anchor['R_deg'])
            trace = P.trace_row(DIAGNOSTIC, seed, row, selected, len(pool))
            trace.update(anchor_model='R0', anchor_hypothesis=hypothesis,
                         anchor_T_cm=anchor['T_cm'], anchor_R_deg=anchor['R_deg'],
                         T_nonincrease=True, R_nonincrease=True, **counts)
            # The old generic trace recognizes only its own whole-pose name.
            trace['physical_complete_pose'] = True
            traces.append(trace)
        assert tally['frames'] == tally['T_nonincrease'] == tally['R_nonincrease'] == 173
        metrics[model] = chosen
        by_seed[f's{seed}'] = dict(tally)
        overall.update(tally)
    assert len(traces) == 519 and all(t['physical_complete_pose'] for t in traces)
    assert all(metrics[m] is operations[m] for m in P.MODELS if not m.startswith('DIVERSE251_'))
    gate, hierarchy = E.gate_results(metrics, rows, populations, values['stable_protocol'])
    gate['original_gate_verdict'] = gate['verdict']
    gate['verdict'] = 'DIAGNOSTIC_GATES_PASS_NOT_METHOD_SUCCESS' if gate['PASS'] else 'DIAGNOSTIC_GATES_FAIL_NOT_METHOD_VERDICT'
    gate.update(diagnostic_only=True, method_success=False, goal_complete=False,
                choice_uses_cached_GT_derived_errors=True)
    # These R0 guard consequences follow from pointwise nonincrease; the
    # remaining original gates (including SINGLE comparisons) are not assumed.
    assert gate['gates']['natural_tails']['comparisons']['R0']['PASS']
    assert gate['gates']['clean_preservation']['PASS']
    summaries, per_seed, by_recording = P.summarize(E, metrics, rows, populations)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(traces[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(traces)
    csv_path = C.DOC / 'REAL_FEASIBILITY_ROWS.csv'
    C.save(csv_path, buffer.getvalue())
    for binding in [*protocol['inputs'].values(), *protocol['codes']]:
        C.verify(binding)
    result = dict(complete=True, created_at=C.now(), diagnostic=DIAGNOSTIC, PASS=gate['PASS'],
        diagnostic_only=True, method_success=False, goal_complete=False, deployable=False,
        protocol=C.bind(C.DOC / 'PROTOCOL.json'), inputs=protocol['inputs'], codes=protocol['codes'],
        scale=list(P.SCALE), seeds=list(P.SEEDS), frames=173, frame_seed_rows=519,
        populations={k: len(v) for k, v in populations.items()},
        stability=gate, hierarchy=hierarchy, summaries=summaries,
        per_seed_summaries=per_seed, by_recording=by_recording,
        historical_gate_parity=dict(PASS=True, scalar_leaves_checked=parity, float_atol=1e-12),
        pointwise_anchor=dict(PASS=True, all_rows_retained=True, all_selected_poses_finite=True,
            T_nonincrease=overall['T_nonincrease'], R_nonincrease=overall['R_nonincrease'],
            strict_inequality_tolerance=0., counts=dict(overall), by_seed=by_seed),
        fallback_counts=dict(anchor_returned=overall['anchor_returned'],
            anchor_only_eligible=overall['anchor_only_eligible'],
            definition='anchor_returned is exact whole-pose identity equality, not a failed pose; anchor_only_eligible counts frames with no other admissible identity.'),
        previous_fixed_cost_oracle=dict(binding=protocol['inputs']['previous_feasibility'],
            PASS=previous['diagnostics']['FIXED_TRAIN_COST_ORACLE']['stability']['PASS'],
            stability=previous['diagnostics']['FIXED_TRAIN_COST_ORACLE']['stability'],
            summaries=previous['diagnostics']['FIXED_TRAIN_COST_ORACLE']['summaries']),
        diagnostic_slot_aliases={'DIVERSE251': DIAGNOSTIC,
            **{f'DIVERSE251_s{s}': f'{DIAGNOSTIC}_s{s}' for s in P.SEEDS}},
        slot_warning='DIVERSE251 keys are evaluator slots containing this GT-derived oracle, not new learned-model performance.',
        interpretation='PASS is existence of a GT-derived complete-pose selection satisfying the original reused-DEV gates. FAIL excludes this fixed anchored cost rule only. Learned R0_ONLY real comparison is absent; expanded UNION method success, deployability, learnability and generalization are not established.',
        anchor_contract='R0 operational GEO whole pose explicitly added even if omitted from the compressed frontier. Identical cached candidate identities require exact T/R equality. Both errors must be <= anchor without epsilon, then unchanged TRAIN-scale minmax and exact-cost/Pareto/R0/hypothesis/expert tie handling.',
        compressed_pool_reason='An omitted candidate is weakly dominated by a retained candidate. If the omitted candidate obeys the anchor inequalities, its dominator also obeys them. Thus omitted candidates cannot improve the admissible minmax/Pareto choice; the anchor identity is restored explicitly.',
        artifacts=[C.bind(csv_path)], read_paths=sorted(set(reads)), image_forwards=0,
        new_fits=0, optimizer_updates=0, new_PnP=0, new_reference_metric_calls=0,
        raw_GT_reads=0, learned_checkpoints_read=0, new_learned_real_routing=0,
        unchanged_primary_artifacts=True, wall_seconds=time.monotonic()-started)
    C.save(C.DOC / 'REAL_FEASIBILITY.json', result)
    print('PARETO_ANCHOR_REAL_FEASIBILITY_DIAGNOSTIC', result['PASS'], flush=True)


def selfcheck():
    def p(model, hyp, t, r):
        return dict(model=model, hypothesis=P.HYPOTHESES[hyp], T_cm=float(t), R_deg=float(r))
    anchor = p('R0', 0, 10, 2)
    unsafe = p('DIVERSE251_s1', 1, 9, 3)
    assert P.choose([anchor, unsafe], 'FIXED_TRAIN_COST_ORACLE', (1, 1))['chosen'] == unsafe
    selected, counts = anchored_choice([unsafe], anchor, (1, 1))
    assert selected['chosen'] == anchor and counts['anchor_restored'] and counts['anchor_only_eligible']
    dominating = p('DIVERSE251_s1', 1, 10, 1)
    selected, counts = anchored_choice([anchor, dominating], anchor, (1, 1))
    assert selected['chosen'] == dominating and counts['R_strict_improvement'] and not counts['T_strict_improvement']
    try:
        anchored_choice([dict(anchor, T_cm=11)], anchor)
    except AssertionError:
        pass
    else:
        raise AssertionError('Anchor identity mismatch must stop')
    rng = np.random.default_rng(20261002)
    for _ in range(100):
        full, compressed = [], []
        for model in ('R0', 'DIVERSE251_s1'):
            pool = [p(model, h, *rng.uniform(.1, 20, 2)) for h in (0, 1)]
            full.extend(pool)
            compressed.extend([min(pool, key=lambda v: (v['T_cm'], v['hypothesis'])),
                               min(pool, key=lambda v: (v['R_deg'], v['hypothesis']))])
        anchor = full[int(rng.integers(0, 2))]
        a, _ = anchored_choice(full, anchor)
        b, _ = anchored_choice(compressed, anchor)
        assert a['chosen'] == b['chosen'] and a['cost'] == b['cost']
        assert a['T_cm'] <= anchor['T_cm'] and a['R_deg'] <= anchor['R_deg']
    assert 'scripts.research.pallet_pose_stable_improvement_20261001_v1.evaluate' not in sys.modules
    assert 'scripts.research.pallet_pose_pareto_anchor_20261001_v1.common' not in sys.modules
    print('PARETO_ANCHOR_SELFCHECK_PASS: invented tradeoff, omitted-anchor restoration, exact identity, Pareto tie, compressed/full pool parity. No real files read.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['selfcheck', 'score'])
    args = parser.parse_args()
    (selfcheck if args.stage == 'selfcheck' else run)()
