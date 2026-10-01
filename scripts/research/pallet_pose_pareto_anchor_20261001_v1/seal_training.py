"""One bounded target-only intervention, with all previous source/real gates."""
from . import common as C


def main():
    gate = C.read(C.DOC / 'SOURCE_FEASIBILITY.json')
    assert gate['complete'] and gate['PASS']
    assert not (C.RAW / 'fits').exists()
    prior = C.read(C.CONV_DOC / 'PROTOCOL.json')
    inputs = dict(prior['inputs'])
    inputs.update(source_feasibility=C.bind(C.DOC / 'SOURCE_FEASIBILITY.json'),
        source_protocol=C.bind(C.DOC / 'SOURCE_PROTOCOL.json'),
        real_authorization=C.bind(C.DOC / 'REAL_FEASIBILITY_AUTHORIZATION.json'))
    codes = [C.bind(C.HERE / n) for n in ('common.py', 'convex_train.py', 'evaluate_source.py', 'seal_training.py')]
    codes += [C.bind(C.ROOT / 'scripts/research/pallet_pose_selector_convergence_20261001_v1' / n)
              for n in ('convex_train.py', 'common.py')]
    codes += [C.bind(C.ROOT / 'scripts/research/pallet_pose_union_selection_20261001_v1' / n)
              for n in ('common.py', 'source_features.py', 'train.py')]
    codes += [C.bind(C.HERE / 'source_feasibility.py')]
    p = dict(schema='pallet_pose_pareto_anchor_training_v1', complete=True, created_at=C.now(),
        models=list(C.MODEL_NAMES), feature_dim=94, train_rows=2598, max_fits=4,
        device='cpu', cpu_threads=1, lambda_l2=1e-4, bias=0., normalization='old_float32_then_float64',
        initialization=prior['initialization'], solver=prior['solver'], certificate=prior['certificate'],
        certificate_definition=prior['certificate_definition'], objective=prior['objective'],
        target_rule='R0_GEO_PARETO_NONREGRESSION_FIXED_TRAIN_MINMAX',
        changed_factor='Only source target admissibility: errors in both T and R must be no greater than the frozen operational R0 GEO errors. Same converged CE+ridge, features, normalization, candidates, source rows, scale and solver.',
        target='Use original fixed TRAIN-scale minmax and exact ties within the R0-anchored admissible set. The original geometric-valid mask remains unchanged in CE and inference.',
        anchor_rule='Frozen R0 GEO candidate on the same frame; candidate index/errors parity required. One original all-invalid row remains failure and zero CE over full2598 denominator.',
        runtime_safety='A constrained oracle target gives no guarantee that approximate GT-free predictions satisfy nonregression. Only the original measured source and real gates can establish performance.',
        candidate_pool=prior['candidate_pool'], inference_tie=prior['inference_tie'], no_valid_fallback=prior['no_valid_fallback'],
        architecture=prior['architecture'], seed_meaning=prior['seed_meaning'], runtime_inputs=prior['runtime_inputs'],
        real_operator_seal=prior['real_operator_seal'], checkpoint=prior['checkpoint'],
        source_val=prior['source_val'], real_evaluation=prior['real_evaluation'],
        input_labels='Source TRAIN only; no new real reference labels or source VAL qualities in optimization.',
        source_VAL_reuse='Same held-out split reused for several method diagnoses, not a fresh generalization claim. Frozen refiner-training source overlaps remain disclosed.',
        inputs=inputs, codes=codes,
        evidence={'prior_converged_protocol': C.bind(C.CONV_DOC / 'PROTOCOL.json'),
                  'prior_methods': C.bind(C.DOC / 'PRIOR_METHOD_AUDIT.json'),
                  'real_feasibility': C.bind(C.DOC / 'REAL_FEASIBILITY.json')},
        evidence_access='Sealer-only rationale; trainer opens its input bindings and codes, never these real-quality evidence contents.',
        stop='Max4 deterministic fits; max1000 iterations/2000 objective calls each; failed solver/certificate or any of45 source VAL checks stops this method before new learned real routing. No sweep/seed extension/threshold changes.',
        method_success=False, goal_complete=False)
    for b in [*inputs.values(), *codes, *p['evidence'].values()]:
        C.verify(b)
    C.save(C.DOC / 'TRAIN_PROTOCOL.json', p)
    C.save(C.DOC / 'TRAIN_PROTOCOL_SHA.json', C.bind(C.DOC / 'TRAIN_PROTOCOL.json'))
    print('PARETO_ANCHOR_TRAIN_PROTOCOL_SEALED', C.bind(C.DOC / 'TRAIN_PROTOCOL.json'))


if __name__ == '__main__':
    main()
