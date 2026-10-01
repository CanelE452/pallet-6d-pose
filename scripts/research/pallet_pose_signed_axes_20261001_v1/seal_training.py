"""Seal one signed physical T/R regression intervention before any fit."""
from . import common as C
from . import convex_train as T


def main():
    review = C.read(C.DOC / 'PREFIT_REVIEW.json')
    assert review['complete'] and review['PASS']
    assert not (C.RAW / 'fits').exists()
    prior = C.read(C.RBF_DOC / 'TRAIN_PROTOCOL.json')
    C.verify(C.read(C.RBF_DOC / 'TRAIN_PROTOCOL_SHA.json'))
    basis_binding = C.bind(C.RBF_DOC / 'RBF_BASIS.json')
    basis = C.read(C.ROOT / basis_binding['path'])
    assert basis['complete'] and basis['PASS'] and review['basis_SHA_bind'] == basis_binding
    inputs = dict(prior['inputs'])
    inputs.update(prefit_review=C.bind(C.DOC / 'PREFIT_REVIEW.json'),
                  rbf_basis=basis_binding, basis_protocol=C.bind(C.RBF_DOC / 'BASIS_PROTOCOL.json'))
    codes = [C.bind(C.HERE / name) for name in ('common.py', 'convex_train.py',
        'evaluate_source.py', 'evaluate_real.py', 'seal_training.py', 'prefit_review.py')]
    codes += prior['codes']
    codes = list({b['path']: b for b in codes}.values())
    p = dict(prior)
    for key in ('risk_definition', 'margin_definition', 'margin_scope', 'embedding_control'):
        p.pop(key, None)
    p.update(schema='pallet_pose_signed_axes_training_v1', created_at=C.now(),
        feature_dim=253, raw_feature_dim=94, context_dim=189, rbf_dim=64, output_dim=2,
        feature_map=T.FEATURE_MAP, input_rule=T.INPUT_RULE, target_rule=T.TARGET_RULE,
        loss_rule=T.LOSS_RULE, prediction_rule=T.PREDICTION_RULE, huber_delta=1.,
        runtime_uses_margin=False, runtime_uses_reference_errors=False, runtime_safe_mask=False,
        basis_SHA_bind=basis_binding,
        initialization='All253x2 weights zero; fixed bias0; one deterministic final fit per model.',
        architecture='Shared253x2 linear regression on frozen candidate-minus-operational-anchor features; max of the two outputs supplies the whole-pose selection score.',
        feature_transform='Preserve prior normalized94+abs-anchor94+identity1+fixedRBF64. Subtract the same input-only R0 GEO anchor253 from each valid candidate. No extra normalization. Set invalid differences to zero without subtracting invalid placeholders.',
        objective='For each frame, average Huber(delta=1) residual loss over its original valid candidates and both axes. Mean over all2598 frames, with the all-invalid frame contributing zero. Add lambda/2 times Frobenius norm squared of all253x2 weights, lambda=1e-4.',
        target='On valid candidates with finite operational R0 anchors, e=(candidate physical T/R error minus anchor physical T/R error) divided by the original fixed TRAIN scales; y=sign(e)*log1p(abs(e)) per axis. Never subtract inf-inf. Anchor targets are exactly zero; invalid placeholders remain zero.',
        scale_definition='Reuse TRAIN sT=2.4636887551191258cm and sR=1.113474019956766deg; no refitting, clipping, sign reweighting or axis weight search.',
        anchor_rule='Frozen input-only R0 operational GEO identity; never learned R0_ONLY or a reference-selected candidate. Index0/1 on every valid row; all-invalid row index-1 contributes zero loss but stays in the full2598 denominator. Missing valid anchor is a contract error.',
        runtime_safety='Predicted changes need not equal actual changes. max-of-two with anchor score0 cannot certify actual nonregression. Preserve exact R0-then-hypothesis ties; a different R0 hypothesis tied at0 can be selected. Original source and both families of real stability criteria alone determine success.',
        certificate_definition='Huber(delta=1) of affine outputs is convex and continuously differentiable. L2 on all506 parameters makes J lambda-strongly convex, so J(W)-min J <= ||gradient J(W)||_F^2/(2lambda). Huber Hessian is piecewise defined away from residual magnitude1; no T/R or generalization guarantee.',
        changed_factor='Replace one-hot best-candidate margin CE with two continuous signed physical T/R changes and their fixed max-of-two decision rule. Retain original candidates, TRAIN2598, physical labels/scales, raw normalization, frozen RBF basis, ridge coefficient, zero initialization, solver budget, and every source/real gate. Candidate-minus-anchor is needed to fix anchor predictions0, not claimed as new raw information.',
        generalization_limit='Previous signed2D utility regression and global RGB MLP transfer failures remain negative prior evidence. This bounded comparison concerns physical T/R on current whole-pose candidates. Lower TRAIN Huber or source PASS is not stable real improvement. No new RGB information or new real-GT training is introduced.',
        inputs=inputs, codes=codes,
        evidence=dict(previous_RBF_protocol=C.bind(C.RBF_DOC / 'TRAIN_PROTOCOL.json'),
            previous_RBF_report=C.bind(C.RBF_DOC / 'REPORT_KO.md'),
            transfer_diagnostic=C.bind(C.RBF_DOC / 'SELECTOR_TRANSFER_DIAGNOSTIC.json'),
            original_goal_protocol=prior['evidence']['original_goal_protocol']),
        evidence_access='Rationale is read by this sealer only; fitting opens fixed input/code bindings, never source VAL quality or real-quality evidence.',
        stop='At most4 deterministic final fits, each1000 iterations/2000 objective calls and gradient-gap certificate. Failed fit or source45 gate prevents real routing. No restarts, best-iteration/seed selection, Huber/transform/weight/threshold sweep, frame exclusions or budget extensions.',
        method_success=False, goal_complete=False)
    for key in ('source_val', 'real_evaluation', 'original_goal_criteria', 'real_stability_contract',
                'solver', 'certificate', 'candidate_pool', 'inference_tie', 'no_valid_fallback',
                'runtime_inputs', 'models', 'train_rows', 'max_fits', 'lambda_l2', 'normalization'):
        assert p[key] == prior[key], key
    assert p['models'] == list(C.MODEL_NAMES) and p['max_fits'] == 4
    assert p['loss_rule'] == review['loss_rule'] and p['target_rule'] == review['target_rule']
    for b in [*inputs.values(), *codes, *p['evidence'].values()]:
        C.verify(b)
    C.save(C.DOC / 'TRAIN_PROTOCOL.json', p)
    C.save(C.DOC / 'TRAIN_PROTOCOL_SHA.json', C.bind(C.DOC / 'TRAIN_PROTOCOL.json'))
    print('SIGNED_AXES_TRAIN_PROTOCOL_SEALED', C.bind(C.DOC / 'TRAIN_PROTOCOL.json'))


if __name__ == '__main__':
    main()
