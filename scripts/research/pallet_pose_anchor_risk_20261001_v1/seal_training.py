"""Seal one fixed TRAIN-only risk-margin objective before four new fits."""
from . import common as C


def main():
    review = C.read(C.DOC / 'PREFIT_REVIEW.json')
    assert review['complete'] and review['PASS']
    assert not (C.RAW / 'fits').exists()
    prior = C.read(C.CONTEXT_DOC / 'TRAIN_PROTOCOL.json')
    C.verify(C.read(C.CONTEXT_DOC / 'TRAIN_PROTOCOL_SHA.json'))
    source = C.read(C.ANCHOR_DOC / 'SOURCE_FEASIBILITY.json')
    assert source['complete'] and source['PASS'] and source['source_TRAIN_only']
    original_goal_path = C.U.PREV / 'PROTOCOL_EFFECTIVE.json'
    original_goal = C.read(original_goal_path)
    inputs = dict(prior['inputs'])
    inputs['prefit_review'] = C.bind(C.DOC / 'PREFIT_REVIEW.json')
    assert inputs['source_feasibility'] == C.bind(C.ANCHOR_DOC / 'SOURCE_FEASIBILITY.json')
    codes = [C.bind(C.HERE / n) for n in
             ('common.py', 'convex_train.py', 'evaluate_source.py', 'evaluate_real.py',
              'seal_training.py', 'prefit_review.py')]
    codes += prior['codes']
    codes = list({b['path']: b for b in codes}.values())
    p = dict(prior)
    p.update(schema='pallet_pose_anchor_risk_training_v1', created_at=C.now(),
        loss_rule='TRAIN_LOG1P_ANCHOR_EXCESS_MARGIN_CE', runtime_uses_margin=False,
        objective='Full2598 mean of logsumexp(-score_c + margin_c) + score_target, over original valid candidates, plus lambda/2 ||w189||^2. Target margin exactly zero. No-valid rows contribute zero data loss but remain in the denominator.',
        risk_definition='r_c=max((T_c-T_R0_GEO)/sT,(R_c-R_R0_GEO)/sR,0), using only existing TRAIN labels and fixed TRAIN scales sT=2.4636887551191258cm, sR=1.113474019956766deg.',
        margin_definition='margin_c=log1p(r_c), one fixed unscaled and unclipped formula. Initialize risk/margin to zero, compute only on valid candidates with finite anchors; do not evaluate inf-inf. Invalid margins and all-invalid row margins stay zero. Every anchored target and R0 anchor has margin exactly zero.',
        margin_scope='TRAIN supervision only. No margin, reference error or new runtime mask is provided to source/real inference. Runtime keeps the previous shared189 score and all original valid whole-pose candidates.',
        certificate_definition='With fixed TRAIN margins the log-sum-exp of affine scores is convex. L2 on all189 weights yields lambda-strong convexity and objective gap <= ||gradient||^2/(2*lambda). This is a numerical objective certificate, not T/R or generalization certification.',
        changed_factor='Only the loss adds fixed TRAIN log1p anchor-excess margins to competing logits. Preserve context189, target/safe hashes, original validity, full2598 denominator, scales, lambda, zero initialization, solver caps, and source/real gates.',
        embedding_control='Exactly zero margins reduce value, gradient and Hessian to the previous context189 CE+ridge. Existing context189 weights can be evaluated under both fixed objectives but are not new fits or checkpoint candidates.',
        generalization_limit='Cost/no-harm losses and relative context have prior failures. The current model already favors R0; increasing unsafe-candidate penalties may sacrifice strict gains. No success is inferred from fewer unsafe TRAIN choices.',
        original_goal_criteria=original_goal['stability_criteria'],
        real_stability_contract='matched_intervention_AND_original_SINGLE251_stability',
        original_goal_evaluation='Retain the matched R0_ONLY/paired-DIVERSE intervention comparisons and also run the original frozen SINGLE251/R0/PRIOR1/FULL125 five stability gates on the new UNION poses. Both sets must pass; no old gate, seed, frame, threshold or uncertainty check is removed.',
        original_goal_contract_correction='Previous unused learned-real evaluators omitted explicit SINGLE251 comparisons. No learned real results were produced by those failed-source methods. Restore the original goal checks before sealing this experiment, with no change to source45 gates or learning.',
        inputs=inputs, codes=codes,
        evidence={'previous_context_protocol': C.bind(C.CONTEXT_DOC / 'TRAIN_PROTOCOL.json'),
                  'previous_context_report': C.bind(C.CONTEXT_DOC / 'REPORT_KO.md'),
                  'train_risk_diagnostic': C.bind(C.CONTEXT_DOC / 'TRAIN_RISK_DIAGNOSTIC.json'),
                  'original_goal_protocol': C.bind(original_goal_path)},
        evidence_access='Sealer-only rationale. Optimization opens input/code bindings, never source VAL or real-quality evidence.',
        stop='Max4 deterministic final fits; original1000 iteration/2000 objective-call limits and numerical certificate. Any failed certificate or any failed original45 source checks stops this method before learned real routing. No sweep, winner promotion, altered thresholds or dropped seeds.',
        method_success=False, goal_complete=False)
    assert p['feature_dim'] == 189 and p['raw_feature_dim'] == 94
    assert p['feature_map'] == 'normalized94_abs_anchor_delta94_identity1'
    assert p['target_rule'] == 'R0_GEO_PARETO_NONREGRESSION_FIXED_TRAIN_MINMAX'
    assert p['source_val'] == prior['source_val'] and p['real_evaluation'] == prior['real_evaluation']
    for b in [*inputs.values(), *codes, *p['evidence'].values()]:
        C.verify(b)
    C.save(C.DOC / 'TRAIN_PROTOCOL.json', p)
    C.save(C.DOC / 'TRAIN_PROTOCOL_SHA.json', C.bind(C.DOC / 'TRAIN_PROTOCOL.json'))
    print('ANCHOR_RISK_TRAIN_PROTOCOL_SEALED', C.bind(C.DOC / 'TRAIN_PROTOCOL.json'))


if __name__ == '__main__':
    main()
