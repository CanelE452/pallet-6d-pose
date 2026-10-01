"""Freeze supervision, reused baseline, original gates and bounded new fits."""
from . import common as C


def main():
    assert not (C.DOC / 'PROTOCOL.json').exists()
    old = C.read(C.PARENT_DOC / 'TRAIN_PROTOCOL.json')
    previous = C.read(C.CONV_DOC / 'PROTOCOL.json')
    audit_path = C.DOC / 'TARGET_CONTRACT_AUDIT.json'
    audit = C.read(audit_path)
    assert audit['complete'] and audit['PASS']
    baseline = C.read(C.CONV_DOC / 'FIT_R0_ONLY.json')
    assert baseline['certificate']['PASS']
    inputs = dict(old['inputs'])
    inputs.update(parent_train_protocol=C.bind(C.PARENT_DOC / 'TRAIN_PROTOCOL.json'),
                  source_poses=C.bind(C.PARENT_RAW / 'SOURCE_POSES.json'),
                  convergence_protocol=C.bind(C.CONV_DOC / 'PROTOCOL.json'),
                  R0_origin_receipt=C.bind(C.CONV_DOC / 'FIT_R0_ONLY.json'),
                  R0_origin_checkpoint=baseline['checkpoint'], R0_origin_START=baseline['START'],
                  R0_origin_trace=baseline['trace'], target_audit=C.bind(audit_path))
    codes = [C.bind(C.HERE / n) for n in ('common.py', 'train.py', 'evaluate_source.py', 'seal.py')]
    codes += [C.bind(C.U.HERE / n) for n in ('common.py', 'source_features.py', 'train.py')]
    conv_code = C.ROOT / 'scripts/research/pallet_pose_selector_convergence_20261001_v1'
    codes += [C.bind(conv_code / n) for n in ('common.py', 'convex_train.py')]
    objective_doc = C.ROOT / '_docs/experiments/pallet_pose_selector_objective_audit_20261001_v1'
    evidence = dict(previous_source_gate=C.bind(C.CONV_DOC / 'SOURCE_VAL_GATE.json'),
                    previous_public_review=C.bind(C.CONV_DOC / 'PUBLIC_REVIEW.json'),
                    previous_training_convergence=C.bind(C.CONV_DOC / 'TRAIN_CONVERGENCE.json'),
                    TRAIN_diagnostic=C.bind(objective_doc / 'TRAIN_DIAGNOSTIC.json'),
                    prior_objectives=C.bind(objective_doc / 'PRIOR_OBJECTIVE_AUDIT.json'))
    for b in [*inputs.values(), *codes, *evidence.values()]:
        C.verify(b)
    p = dict(schema='pallet_pose_pairwise_supervision_v1', complete=True, created_at=C.now(),
        models=list(C.MODEL_NAMES), new_models=list(C.NEW_MODELS), reused_models=['R0_ONLY'],
        model_count=4, max_fits=3, train_rows=2598, feature_dim=94,
        lambda_l2=1e-4, bias=0., normalization='old_float32_then_float64',
        device='cpu', cpu_threads=1, initialization='All94 weights zero for each new UNION fit.',
        solver=previous['solver'], certificate=previous['certificate'],
        objective='equal_group_pairwise_logistic',
        edge_groups={'WD':[[0,1],[2,3]], 'expert':[[0,2],[1,3]]},
        group_weights={'WD':.5,'expert':.5}, fixed_edge_denominators=2,
        edge_loss='softplus(score_winner - score_loser); lower score wins.',
        reduction='Each UNION group averages its two fixed edges; group means weighted0.5 each. '
            'Average over all2598 frames, then add0.5*1e-4*||weight94||^2. '
            'Unavailable edges contribute zero without renormalizing groups or frames.',
        target_order='Per frame: ascending original physical max(T/sT,R/sR) cost; '
            'within each exact-cost tie group, successive nondominated Pareto fronts; '
            'within each front, original R0 then hypothesis-name then expert tie key. '
            'All edge winners use this single total order.',
        tie_correction='Pair-local Pareto/tie comparisons can form a theoretical cycle. '
            'Global ordering is fixed before learning, preserves original whole-best target, '
            'and may change ordering of lower-ranked exact ties. Actual differences are audited.',
        R0_reuse='R0_ONLY has one long/short edge, weight1. Its target and logistic objective '
            'equal the converged two-class CE+ridge exactly as mathematical functions. '
            'Reuse original numeric parameters; validate independent objective/gradient equivalence, '
            'write an explicit metadata wrapper, perform zero new baseline optimizer steps.',
        inference='All four UNION candidates remain eligible; one shared linear94 argmin chooses '
            'an entire R,t. No sequential WD filter, new image feature, cost-gap weight or tuned threshold.',
        supervision_limit='Four edges omit diagonal comparisons. Acyclic labels do not guarantee '
            'that perfect edge predictions identify the unique global cost-best candidate. '
            'TRAIN audit counts multiple zero-indegree candidates; keep the fixed graph and '
            'assess the original whole-pose source and real gates, not edge accuracy alone.',
        checkpoint='One certified final result per UNION. No best seed, iteration or hyperparameter search.',
        runtime_inputs=previous['runtime_inputs'], seed_meaning=previous['seed_meaning'],
        source_val=previous['source_val'], real_evaluation=previous['real_evaluation'],
        real_operator_seal=previous['real_operator_seal'],
        before_after_TRAIN_diagnostics='Describe frozen old/new model edge accuracy, WD/expert errors '
            'and operational pose costs without choosing thresholds/checkpoints or adding gates.',
        source_VAL_limitation='Repeated development VAL, not new independent TEST. No criteria are '
            'changed based on already observed VAL differences.',
        inputs=inputs, codes=codes, evidence=evidence,
        evidence_access='Evidence is verified by sealer only; fitting protocol() does not open it.',
        stop='No more than3 new fits. Each max1000iterations/2000closures. Failed certificate or '
            'any original source VAL comparison stops this method before real routing. No extension.',
        previous_goal_turn_classification='PROGRESS:4 certified fits, new source evidence, '
            '58 public files remotely verified at92d1c2f7; full stable-real-T/R goal remains unmet.',
        goal_complete=False)
    C.save(C.DOC / 'PROTOCOL.json', p)
    C.save(C.DOC / 'PROTOCOL_SHA.json', C.bind(C.DOC / 'PROTOCOL.json'))
    print('PAIRWISE_PROTOCOL_FROZEN', C.sha(C.DOC / 'PROTOCOL.json'), flush=True)


if __name__ == '__main__':
    main()
