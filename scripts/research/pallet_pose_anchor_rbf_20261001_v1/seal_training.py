"""Seal a fixed RBF expansion; preserve risk loss and every performance gate."""
from . import common as C


def main():
    review = C.read(C.DOC / 'PREFIT_REVIEW.json')
    assert review['complete'] and review['PASS']
    assert not (C.RAW / 'fits').exists()
    basis = C.read(C.DOC / 'RBF_BASIS.json')
    assert basis['complete'] and basis['PASS']
    prior = C.read(C.RISK_DOC / 'TRAIN_PROTOCOL.json')
    C.verify(C.read(C.RISK_DOC / 'TRAIN_PROTOCOL_SHA.json'))
    inputs = dict(prior['inputs'])
    inputs['prefit_review'] = C.bind(C.DOC / 'PREFIT_REVIEW.json')
    inputs['rbf_basis'] = C.bind(C.DOC / 'RBF_BASIS.json')
    inputs['basis_protocol'] = C.bind(C.DOC / 'BASIS_PROTOCOL.json')
    codes = [C.bind(C.HERE / n) for n in ('common.py', 'basis.py', 'convex_train.py',
        'evaluate_source.py', 'evaluate_real.py', 'seal_training.py', 'seal_basis.py', 'prefit_review.py')]
    codes += prior['codes']
    codes = list({b['path']: b for b in codes}.values())
    p = dict(prior)
    p.update(schema='pallet_pose_anchor_rbf_training_v1', created_at=C.now(),
        feature_dim=253, raw_feature_dim=94, context_dim=189, rbf_dim=64,
        feature_map='normalized94_abs_anchor_delta94_identity1_fixed_rbf64',
        basis_SHA_bind=inputs['rbf_basis'],
        objective='Unchanged full2598 mean of logsumexp(-score_c + margin_c) + score_target over original valid candidates, plus lambda/2 ||w253||^2. Fixed TRAIN target margin zero. No-valid rows contribute zero data loss and remain in denominator.',
        margin_scope='TRAIN supervision only, unchanged risk formula, scale and labels. Runtime uses fixed253 score and all original valid whole-pose candidates, never margins or a reference-derived filter.',
        certificate_definition='The frozen253 feature map gives affine scores in trainable weights. The unchanged fixed-margin log-sum-exp is convex; L2 on all253 weights yields lambda-strong convexity and gap <= ||gradient||^2/(2*lambda). No T/R or generalization guarantee.',
        changed_factor='Only concatenate64 frozen TRAIN-input-only RBF features to the previous189 anchor context. Preserve raw94 normalization, anchored target/safe hashes, original validity, risk/margin hashes, full2598 denominator, scales, lambda, zero initialization, solver caps, and source/real gates.',
        rbf_definition='Append exp(-||context189-center_j||^2/(2*bandwidth_squared)) for64 fixed centers. No extra normalization; all253 coordinates zero for invalid candidates. Basis construction is separately sealed and independently reconstructed before fitting.',
        embedding_control='Previous189 weight followed by64 zeros preserves the previous scores and margin objective mathematically. Actual TRAIN score parity and invented-fixture objective/gradient/Hessian parity checked before fitting. No prior weight is a new fitted competitor or candidate for selection.',
        generalization_limit='Nonlinear geometry and RGB-context selectors have previous failures. This fixed RBF feature map isolates one expressiveness hypothesis while retaining a certified optimizer; no new RGB information is introduced. Better TRAIN fit or source45 alone cannot establish real joint improvement.',
        original_goal_contract_correction='The preceding risk experiment restored the original SINGLE251 criteria before its first learned-real routing. Retain exactly that restored original-plus-matched contract in this experiment.',
        inputs=inputs, codes=codes,
        evidence=dict(previous_risk_protocol=C.bind(C.RISK_DOC / 'TRAIN_PROTOCOL.json'),
            previous_risk_report=C.bind(C.RISK_DOC / 'REPORT_KO.md'),
            transfer_diagnostic=C.bind(C.RISK_DOC / 'SELECTOR_TRANSFER_DIAGNOSTIC.json'),
            original_goal_protocol=prior['evidence']['original_goal_protocol']),
        evidence_access='Sealer-only rationale; optimizer opens input/code bindings, never source VAL or real-quality evidence.',
        stop='Max4 deterministic final fits with original1000 iterations/2000 calls and numerical certificate. Any failure stops the method before learned real routing. No width/count/scale/threshold sweep, better-seed promotion, altered guards, dropped rows or budget extension.',
        method_success=False, goal_complete=False)
    for key in ('source_val', 'real_evaluation', 'original_goal_criteria', 'real_stability_contract',
                'target_rule', 'loss_rule', 'risk_definition', 'margin_definition', 'solver', 'certificate'):
        assert p[key] == prior[key], key
    assert p['lambda_l2'] == 1e-4 and p['normalization'] == 'old_float32_then_float64'
    assert p['models'] == list(C.MODEL_NAMES) and p['max_fits'] == 4
    for b in [*inputs.values(), *codes, *p['evidence'].values()]:
        C.verify(b)
    C.save(C.DOC / 'TRAIN_PROTOCOL.json', p)
    C.save(C.DOC / 'TRAIN_PROTOCOL_SHA.json', C.bind(C.DOC / 'TRAIN_PROTOCOL.json'))
    print('ANCHOR_RBF_TRAIN_PROTOCOL_SEALED', C.bind(C.DOC / 'TRAIN_PROTOCOL.json'))


if __name__ == '__main__':
    main()
