"""Freeze one bounded convergence intervention; no fitting or reference scoring."""
from . import common as C


def main():
    old = C.read(C.PARENT_DOC / 'TRAIN_PROTOCOL.json')
    assert not (C.DOC / 'PROTOCOL.json').exists()
    audit = C.ROOT / '_docs/experiments/pallet_pose_selector_objective_audit_20261001_v1'
    evidence = {name: C.bind(audit / (name + '.json')) for name in
                ('TRAIN_DIAGNOSTIC', 'VAL_ORACLE', 'PRIOR_OBJECTIVE_AUDIT',
                 'DETECTION_SOURCE_FEASIBILITY')}
    evidence['previous_source_gate'] = C.bind(C.PARENT_DOC / 'SOURCE_VAL_GATE.json')
    inputs = dict(old['inputs'])
    inputs['parent_train_protocol'] = C.bind(C.PARENT_DOC / 'TRAIN_PROTOCOL.json')
    inputs['source_poses'] = C.bind(C.PARENT_RAW / 'SOURCE_POSES.json')
    codes = [C.bind(C.HERE / n) for n in
             ('common.py', 'convex_train.py', 'evaluate_source.py', 'seal.py')]
    codes += [C.bind(C.U.HERE / n) for n in ('common.py', 'source_features.py', 'train.py')]
    for binding in [*inputs.values(), *evidence.values(), *codes]:
        C.verify(binding)
    source = dict(old['source_val'])
    source['comparators'] = ['R0_ONLY', 'R0_GEO', 'paired_DIVERSE_GEO']
    real = dict(old['real_evaluation'])
    for k, v in real.items():
        if isinstance(v, str):
            real[k] = v.replace('paired R0_ONLY', 'shared R0_ONLY')
    p = dict(schema='pallet_pose_convex_convergence_v1', created_at=C.now(),
             complete=True, models=list(C.MODEL_NAMES), feature_dim=94,
             train_rows=2598, max_fits=4, device='cpu', cpu_threads=1,
             lambda_l2=1e-4, bias=0., normalization='old_float32_then_float64',
             initialization='All94 weights zero; no optimizer initialization seed search.',
             solver=dict(method='L-BFGS-B', maxiter=1000, maxfun=2000,
                         ftol=1e-15, gtol=1e-8, bounds=None),
             certificate=dict(optimizer_success=True, gap_upper_bound_max=1e-6),
             certificate_definition='For full-frame mean CE + lambda/2 ||w94||^2, '
                 'strong convexity gives objective gap <= ||gradient||^2/(2*lambda). '
                 'A numerical certificate of this objective, not a T/R or generalization guarantee.',
             objective='Same old masked one-hot CE(-score), divided by all2598 rows; '
                 'zero/one-valid rows contribute zero CE. Add explicit lambda/2 ||w94||^2.',
             intervention_caveat='Explicit ridge, float64 arithmetic and converged full-batch solver '
                 'jointly differ from old float32 minibatch AdamW decoupled decay. '
                 'This is not an optimizer-only causal experiment or an identical objective.',
             candidate_pool=old['candidate_pool'], target=old['target'],
             inference_tie=old['inference_tie'], no_valid_fallback=old['no_valid_fallback'],
             architecture='Shared linear94; no expert ID. Frozen R0 and three frozen DIVERSE251 refiners.',
             seed_meaning='s1/s2/s3 are existing frozen refiner seeds. Deterministic shared R0_ONLY '
                 'is fitted once and repeated only as a statistical control, not three independent fits.',
             runtime_inputs='One RGB image and pallet physical dimensions, with existing calibrated '
                 'camera K/geometry contract. No temporal inputs or new real-GT training.',
             real_operator_seal='Source gate PASS is required before any new real routing. '
                 'Real evaluator code and cached real input bindings will be frozen in a separate '
                 'REAL_PROTOCOL before routing, using exactly the real_evaluation criteria fixed here.',
             checkpoint='One final solver result per model; no best iteration or best seed selection.',
             source_val=source, real_evaluation=real, inputs=inputs, codes=codes,
             evidence=evidence,
             evidence_access='Evidence is bound by this sealer for rationale only. Training protocol() '
                 'does not open evidence; no VAL or real quality may enter fitting.',
             stop='Hard cap1000 iterations/2000 objective closures and max4 fits. Failed certificate '
                 'or source VAL gate stops this method before real routing. No extension or retuning.',
             goal_complete=False)
    C.save(C.DOC / 'PROTOCOL.json', p)
    C.save(C.DOC / 'PROTOCOL_SHA.json', C.bind(C.DOC / 'PROTOCOL.json'))
    print('PROTOCOL_FROZEN', C.sha(C.DOC / 'PROTOCOL.json'), flush=True)


if __name__ == '__main__':
    main()
