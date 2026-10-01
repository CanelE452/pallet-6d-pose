"""Freeze two diagnostic rules before computing new union results."""
from . import common as C


def main():
    stable = C.ROOT / '_docs/experiments/pallet_pose_stable_improvement_20261001_v1'
    raw = C.ROOT / 'data/pallet/results/pallet_pose_stable_improvement_20261001_v1'
    joint = C.ROOT / '_docs/experiments/pallet_pose_joint_recovery_20261001_v1'
    source = C.ROOT / '_docs/experiments/pallet_pose_union_selection_20261001_v1'
    inputs = dict(candidate_rows=joint / 'CANDIDATE_BOUND_ROWS.csv',
        candidate_bounds=joint / 'CANDIDATE_BOUNDS.json',
        operational_metrics=raw / 'POSE_METRICS.json',
        eval_metadata=raw / 'EVAL_METADATA.json', eval_groups=raw / 'EVAL_GROUPS.json',
        stable_pose_candidates=raw / 'POSE_CANDIDATES.json',
        historical_metadata=C.ROOT / 'data/pallet/results/pallet_pose_diagnosis_20260930_v1/METADATA.json',
        stable_protocol=stable / 'PROTOCOL_EFFECTIVE.json', stable_results=stable / 'RESULTS.json',
        stable_protocol_sha=stable / 'EFFECTIVE_PROTOCOL_SHA.json',
        stable_pose_lock=stable / 'POSE_PREDICTIONS_LOCK.json',
        stable_prediction_lock=stable / 'PREDICTIONS_LOCK.json',
        source_train_gate=source / 'SOURCE_TRAIN_GATE.json',
        parent_publication=joint / 'PUBLICATION_MANIFEST.json',
        input_audit=C.DOC / 'INPUT_AUDIT.json')
    audit = C.read(inputs['input_audit'])
    assert audit['PASS'] and audit['complete']
    source_scale = C.read(inputs['source_train_gate'])['source_scale']
    scale = [source_scale['sT_cm'], source_scale['sR_deg']]
    assert scale == [2.4636887551191258, 1.113474019956766] and source_scale['valid']
    code_paths = [C.HERE / n for n in ('common.py', 'seal_protocol.py', 'evaluate.py')]
    code_paths += [C.ROOT / 'scripts/research/pallet_pose_stable_improvement_20261001_v1' / n
                   for n in ('evaluate.py', 'common.py')]
    code_paths += [C.ROOT / 'scripts/research/pallet_pose_diagnosis_20260930_v1/run.py']
    code_paths += [C.ROOT / 'scripts/research/pallet_pose_joint_recovery_20261001_v1/candidate_bounds.py']
    # The statistical helper's quantile implementation is also sealed.
    from scripts.research.pallet_pose_stable_improvement_20261001_v1 import evaluate as E
    from pathlib import Path
    code_paths.append(Path(E.C.D.M.__file__).resolve())
    protocol = dict(schema='pallet_pose_real_union_feasibility_v1', created_at=C.now(),
        objective='Check whether fixed existing real candidate poses can satisfy the original stable T/R criteria before another learned objective attempt.',
        input_contract='One RGB image + pallet dimensions + existing calibrated K. No temporal information or new runtime input.',
        scope='Post-result diagnostic on reused DEV. Uses existing reference-derived errors only; no new reference scoring, pose solving, learned routing, fit or image forward.',
        diagnostics=['AXIS_LOWER_BOUND', 'FIXED_TRAIN_COST_ORACLE'], seeds=[1, 2, 3], scale=scale,
        candidates='Per seed, frozen R0 + paired DIVERSE251_sN; exactly two W/D whole poses per model. Never mix coordinates, rotations or translations across candidates.',
        compressed_pool_proof='With exactly two candidates per expert, the union of its T_best/R_best includes every nondominated error pair. If both choose the same pose, it weakly dominates the omitted pose. Positive monotone minmax plus Pareto-first tie handling preserves the selected minimum.',
        axis_lower_bound='For each frame/seed, min T and min R independently; may combine different poses. This is a potentially unattainable optimistic error-pair bound, never exported as a physical pose prediction.',
        whole_pose_oracle='Choose one complete candidate minimizing max(T/sT,R/sR), using only scales fixed by source TRAIN. No real re-scaling or threshold search.',
        tie_rule='Exact floating cost equality; remove Pareto-dominated candidates within that tie; then R0 first, hypothesis lexicographic, expert lexicographic. No epsilon/isclose.',
        population={'all': 173, 'NATURAL99': 99, 'CLEAN29': 29, 'WOOD45': 45, 'natural_recordings': 6},
        controls=['paired SINGLE251_sN', 'R0', 'PRIOR1', 'FULL125'],
        stable_gates='Reuse stable.evaluate.gate_results with only DIVERSE251 error arrays replaced by diagnostic arrays. Preserve all five original gates, seeds, populations, bootstrap draws, strict inequalities and 1.05 guards.',
        interpretation={'axis_FAIL':'If every protected frame has a finite union option and baseline failures are zero, the original failure guards require any admissible selector to retain all frames. A failed mandatory monotone gate then rules out every selector confined to this union and these original gates.',
            'axis_PASS':'Necessary conditions only. Mixed-axis minima need not be jointly attainable.',
            'oracle_PASS':'Existence of a whole-pose selection satisfying original gates on reused DEV; does not establish deployable selection or generalization.',
            'oracle_FAIL':'This fixed cost rule failed; other whole-pose selections are not excluded unless the optimistic bound also failed.'},
        unmeasured_comparator='Learned R0_ONLY real routing was never run after source failures. No claim that this diagnostic satisfies the later expanded UNION method contract.',
        reference_limitations='Repeated DEV with geometry-derived reference, not independent measured 6D ground truth. Original teacher lineage includes 38 manual corners/9 images; no new real labels are used for training.',
        budgets={'fits': 0, 'optimizer_updates': 0, 'image_forwards': 0, 'new_PnP': 0,
                 'new_reference_metric_calls': 0, 'new_learned_real_routing': 0},
        fixed_attempts=2, sweep=False, threshold_changes=False, method_success=False, goal_complete=False,
        inputs={k: C.bind(p) for k, p in inputs.items()}, codes=[C.bind(p) for p in sorted(set(code_paths))])
    assert not (C.DOC / 'RESULTS.json').exists()
    C.save(C.DOC / 'PROTOCOL.json', protocol)
    C.save(C.DOC / 'PROTOCOL_SHA.json', C.bind(C.DOC / 'PROTOCOL.json'))
    print('SEALED', C.bind(C.DOC / 'PROTOCOL.json'))


if __name__ == '__main__':
    main()
