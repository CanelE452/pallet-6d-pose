"""Seal the one fixed sign-logistic addition before any new fit."""
from . import common as C
from . import convex_train as T


def main():
    review=C.read(C.DOC/'PREFIT_REVIEW.json')
    assert review['complete'] and review['PASS'] and not (C.RAW/'fits').exists()
    prior=C.read(C.NEWTON_DOC/'TRAIN_PROTOCOL.json')
    C.verify(C.read(C.NEWTON_DOC/'TRAIN_PROTOCOL_SHA.json'))
    training=C.read(C.NEWTON_DOC/'TRAIN_CONVERGENCE.json')
    gate=C.read(C.NEWTON_DOC/'SOURCE_VAL_GATE.json')
    diagnostic=C.read(C.NEWTON_DOC/'SOURCE_TRANSFER_DIAGNOSTIC.json')
    assert training['complete'] and training['PASS']
    assert gate['complete'] and not gate['PASS'] and gate['checks_passed']==43
    assert diagnostic['complete'] and diagnostic['PASS'] and not diagnostic['method_success']
    assert diagnostic['next_single_intervention']['status']=='PROPOSAL_ONLY_NOT_EXECUTED'
    inputs=dict(prior['inputs'],prefit_review=C.bind(C.DOC/'PREFIT_REVIEW.json'))
    codes=[C.bind(C.HERE/name) for name in ('common.py','convex_train.py','evaluate_source.py',
        'evaluate_real.py','seal_training.py','prefit_review.py')]+prior['codes']
    codes=list({b['path']:b for b in codes}.values())
    p=dict(prior)
    p.update(schema='pallet_pose_signed_axes_sign_training_v1',created_at=C.now(),
        loss_rule=T.LOSS_RULE,sign_rule=T.SIGN_RULE,sign_coefficient=T.SIGN_COEFFICIENT,
        objective='For each frame, average Huber(p-y,delta1)+1{y!=0}*softplus(-sign(y)*p) over ORIGINAL valid candidates and both axes. Mean over all2598 frames including the all-invalid zero-loss row, then add lambda/2 times Frobenius norm squared of all253x2 weights. lambda=1e-4; sign coefficient exactly1.',
        sign_supervision='Derive sign(y) and y!=0 from the unchanged frozen TRAIN signed-log1p physical T/R targets. Exact zero targets, including anchors, contribute zero logistic loss, gradient and curvature, not log(2). No epsilon/sign threshold, class reweighting, nonzero-only normalization, new labels or learned calibration.',
        changed_factor='Add only the fixed physical-axis sign logistic term to the previous certified two-axis Huber objective. Preserve all input/target arrays, original normalization, fixed RBF basis, four candidate pools, frame denominator, scales, zero initialization, ridge, Newton/Armijo settings, runtime max-of-two with old ties, and every source/real criterion.',
        certificate_definition='Huber of affine outputs plus masked logistic of affine outputs is convex and continuously differentiable. Ridge on all506 parameters makes the objective lambda-strongly convex, so the unchanged gradient-gap certificate applies. Generalized curvature uses zero Huber data curvature at |p-y|=1 plus smooth logistic curvature and lambda I. Float64 verification is not interval arithmetic and gives no T/R/generalization guarantee.',
        solver_accounting='The initial objective and every trial count toward the unchanged2000-call cap; only accepted points count toward the1000-iteration cap. Use the last accepted point and its already computed gradient. Each direction solves two253-dimensional SPD generalized Hessian blocks. At |p-y|=1 only the Huber curvature is zero; preserve masked smooth logistic curvature and lambda I. No damping, warm start, restart, tolerance search or budget extension.',
        prediction_interpretation='The two outputs are signed-axis scores trained jointly for continuous changes and their sign. Do not claim calibrated physical errors or probabilities. Runtime remains max of the two outputs over every original valid whole-pose candidate with original tie policy.',
        generalization_limit='The previous converged Huber fits still selected unsafe source candidates. Sign emphasis is a hypothesis, not a proven cause or improvement. Previous pairwise/CE/risk, signed2D utility, structured DHT mixed losses and RGB MLP transfer failures remain counterevidence. Reused VAL/real DEV is not a new independent test.',
        inputs=inputs,codes=codes,
        evidence=dict(previous_protocol=C.bind(C.NEWTON_DOC/'TRAIN_PROTOCOL.json'),
            previous_train_verification=C.bind(C.NEWTON_DOC/'TRAIN_CONVERGENCE.json'),
            previous_source_gate=C.bind(C.NEWTON_DOC/'SOURCE_VAL_GATE.json'),
            previous_fixed_choice_diagnostic=C.bind(C.NEWTON_DOC/'SOURCE_TRANSFER_DIAGNOSTIC.json'),
            original_goal_protocol=prior['evidence']['original_goal_protocol']),
        evidence_access='This sealer alone reads prior performance/rationale. Fitting opens sealed input/code bindings and begins at zero; no prior weights, VAL quality or real-quality evidence is read for fitting.',
        method_success=False,goal_complete=False)
    unchanged=('source_val','real_evaluation','original_goal_criteria','real_stability_contract',
        'candidate_pool','inference_tie','no_valid_fallback','runtime_inputs','models','train_rows','max_fits',
        'lambda_l2','normalization','initialization','architecture','feature_transform','target','scale_definition',
        'anchor_rule','runtime_safety','feature_dim','raw_feature_dim','context_dim','rbf_dim','output_dim',
        'feature_map','input_rule','target_rule','prediction_rule','huber_delta','basis_SHA_bind',
        'solver_rule','solver','certificate','solver_success','stop')
    for key in unchanged:assert p[key]==prior[key],key
    for key,value in T.metadata().items():assert p[key]==review[key]==value,key
    assert p['solver']==T.solver_config() and p['sign_coefficient']==1.
    assert p['basis_SHA_bind']==review['basis_SHA_bind']==C.bind(C.RBF_DOC/'RBF_BASIS.json')
    for b in [*inputs.values(),*codes,*p['evidence'].values()]:C.verify(b)
    C.save(C.DOC/'TRAIN_PROTOCOL.json',p)
    C.save(C.DOC/'TRAIN_PROTOCOL_SHA.json',C.bind(C.DOC/'TRAIN_PROTOCOL.json'))
    print('SIGNED_AXES_SIGN_PROTOCOL_SEALED',C.bind(C.DOC/'TRAIN_PROTOCOL.json'))


if __name__=='__main__':main()
