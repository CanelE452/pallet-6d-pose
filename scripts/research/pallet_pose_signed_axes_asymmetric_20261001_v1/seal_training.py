"""Seal the fixed asymmetric residual loss before four fresh TRAIN fits."""
from . import common as C
from . import convex_train as T
from scripts.research.pallet_pose_signed_axes_direction_20261001_v1 import convex_train as PREVIOUS


def main():
    review=C.read(C.DOC/'PREFIT_REVIEW.json')
    assert review['complete'] and review['PASS'] and not (C.RAW/'fits').exists()
    for key in ('trainer','reviewer','independent_input_helper','independent_sign_helper'):
        C.verify(review[key])
    assert review['trainer']==C.bind(T.__file__)
    for comparison in review['unchanged_function_audit']['evaluation_AST_parity'].values():
        C.verify(comparison['previous']);C.verify(comparison['current'])
    C.verify(C.read(C.PREVIOUS_DOC/'TRAIN_PROTOCOL_SHA.json'))
    prior=C.read(C.PREVIOUS_DOC/'TRAIN_PROTOCOL.json')
    training=C.read(C.PREVIOUS_DOC/'TRAIN_CONVERGENCE.json')
    gate=C.read(C.PREVIOUS_DOC/'SOURCE_VAL_GATE.json')
    diagnostic=C.read(C.PREVIOUS_DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC.json')
    incident=C.read(C.PREVIOUS_DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC_EXECUTION.json')
    assert training['complete'] and training['PASS']
    assert gate['complete'] and not gate['PASS'] and gate['checks_passed']==43 and gate['checks_total']==45
    assert diagnostic['complete'] and diagnostic['PASS'] and diagnostic['no_new_selector_policy_evaluated']
    assert incident['complete'] and incident['PASS'] and incident['diagnostic']==C.bind(C.PREVIOUS_DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC.json')
    inputs=dict(prior['inputs'],prefit_review=C.bind(C.DOC/'PREFIT_REVIEW.json'),
        previous_direction_prefit=C.bind(C.PREVIOUS_DOC/'PREFIT_REVIEW.json'),
        previous_direction_protocol=C.bind(C.PREVIOUS_DOC/'TRAIN_PROTOCOL.json'))
    codes=[C.bind(C.HERE/name) for name in ('common.py','convex_train.py','direction_features.py',
        'evaluate_source.py','evaluate_real.py','seal_training.py','prefit_review.py')]+prior['codes']
    codes=list({b['path']:b for b in codes}.values())
    p=dict(prior)
    p.update(T.metadata())
    p.update(schema='pallet_pose_signed_axes_asymmetric_training_v1',created_at=C.now(),
        objective='For each frame average Huber(p-y,1)+Huber(min(p-y,0),1)+1{y!=0}softplus(-sign(y)*p) over ORIGINAL valid candidates and both axes. Mean over all2598 frames including the all-invalid zero-loss row, then add lambda/2 times squared Frobenius norm of all271x2 weights. lambda=1e-4; additional underprediction coefficient1; sign coefficient1.',
        changed_factor='Only add unit Huber(min(prediction-target,0),1) to the existing symmetric Huber term: fixed2:1 underprediction/overprediction cost. Preserve all271 inputs and nine hashes, targets, original valid masks, all rows/scales/normalizations/basis, four candidate pools, zero initialization, ridge, Newton/Armijo settings, runtime max-of-two and old ties, all source/real criteria.',
        certificate_definition='Symmetric Huber plus Huber(min(affine residual,0),1) and masked sign logistic are convex and continuously differentiable. Ridge on all542 parameters makes J lambda-strongly convex, so the unchanged gradient-gap certificate applies. Float64 is not interval arithmetic and gives no T/R/generalization guarantee.',
        solver_accounting='Same1000 accepted iterations/2000 objective calls; initial and every trial count. Last accepted point only, two271-dimensional SPD generalized-Hessian blocks, fixed Armijo. Extra underprediction curvature0 at e=0; Huber curvature0 at abs(e)=1. Retain logistic and lambdaI. No damping, warm start, restart, tolerance search or budget extension.',
        generalization_limit='Prior271 model converged but failed43/45. Selected candidates with nonpositive predicted axes still had actual positive excess; this supports a directional-cost comparison but does not identify a unique cause. Prior risk-margin/pairwise/utility failures are counterevidence; conservative retreat to anchor can remove safe improvements. Reused VAL/realDEV is not independent TEST.',
        checkpoint='Four final271x2 zero-initialized fits; same input hashes, no validation checkpoint selection. Huber field is base plus extra and both components are separately logged.',
        inputs=inputs,codes=codes,
        evidence=dict(previous_protocol=C.bind(C.PREVIOUS_DOC/'TRAIN_PROTOCOL.json'),
            previous_train_verification=C.bind(C.PREVIOUS_DOC/'TRAIN_CONVERGENCE.json'),
            previous_source_gate=C.bind(C.PREVIOUS_DOC/'SOURCE_VAL_GATE.json'),
            previous_fixed_choice_diagnostic=C.bind(C.PREVIOUS_DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC.json'),
            previous_diagnostic_execution=C.bind(C.PREVIOUS_DOC/'SOURCE_FIXED_CHOICE_DIAGNOSTIC_EXECUTION.json'),
            input_representation=prior['evidence']['input_representation'],
            input_verification=prior['evidence']['input_verification'],
            design=C.bind(C.DOC/'DESIGN_KO.md'),original_goal_protocol=prior['evidence']['original_goal_protocol']),
        evidence_access='Sealer reads previous failure evidence; fitting reads frozen TRAIN inputs and PREFIT only, never prior weights or VAL/real quality. Independent post-fit comparison may read priorP final weights under the same new objective.',
        method_success=False,goal_complete=False)
    unchanged=('source_val','real_evaluation','original_goal_criteria','real_stability_contract',
        'original_goal_evaluation','original_goal_contract_correction','candidate_pool','inference_tie',
        'no_valid_fallback','runtime_inputs','models','train_rows','max_fits','lambda_l2','normalization',
        'initialization','target','scale_definition','anchor_rule','runtime_safety','bias','basis_SHA_bind',
        'solver','certificate','solver_success','stop','sign_supervision','direction_receipt_binding',
        'direction_normalization_sha','direction_mean','direction_std','direction_projection_contract')
    for key in unchanged:assert p[key]==prior[key],key
    for key,value in PREVIOUS.metadata().items():
        if key!='loss_rule':assert p[key]==prior[key]==value,key
    for key,value in T.metadata().items():assert p[key]==review[key]==value,key
    assert p['solver']==T.solver_config() and p['sign_coefficient']==1.
    assert p['underprediction_coefficient']==1. and p['underprediction_cost']==2. and p['overprediction_cost']==1.
    assert p['underprediction_zero_curvature']==0.
    previous_prefit=C.read(C.PREVIOUS_DOC/'PREFIT_REVIEW.json')
    for model in C.MODEL_NAMES:
        for key in T.ALL_HASH_KEYS:assert review['models'][model][key]==previous_prefit['models'][model][key],(model,key)
    assert review['direction_receipt_binding']==p['direction_receipt_binding']
    assert review['direction_normalization_sha']==p['direction_normalization_sha']
    for b in [*inputs.values(),*codes,*p['evidence'].values()]:C.verify(b)
    C.save(C.DOC/'TRAIN_PROTOCOL.json',p)
    C.save(C.DOC/'TRAIN_PROTOCOL_SHA.json',C.bind(C.DOC/'TRAIN_PROTOCOL.json'))
    print('SIGNED_AXES_ASYMMETRIC_PROTOCOL_SEALED',C.bind(C.DOC/'TRAIN_PROTOCOL.json'))


if __name__=='__main__':main()
