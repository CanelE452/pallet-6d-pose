"""Seal the fixed 253+18 input intervention before four fresh TRAIN fits."""
from . import common as C
from . import convex_train as T


def main():
    review=C.read(C.DOC/'PREFIT_REVIEW.json')
    assert review['complete'] and review['PASS'] and not (C.RAW/'fits').exists()
    C.verify(C.read(C.SIGN_DOC/'TRAIN_PROTOCOL_SHA.json'))
    prior=C.read(C.SIGN_DOC/'TRAIN_PROTOCOL.json')
    training=C.read(C.SIGN_DOC/'TRAIN_CONVERGENCE.json')
    gate=C.read(C.SIGN_DOC/'SOURCE_VAL_GATE.json')
    feature=C.read(C.DIRECTION_DOC/'FEATURE_AUDIT.json')
    representation=C.read(C.DIRECTION_DOC/'REPRESENTATION_AUDIT.json')
    verification=C.read(C.DIRECTION_DOC/'VERIFICATION.json')
    assert training['complete'] and training['PASS']
    assert gate['complete'] and not gate['PASS'] and gate['checks_passed']==43
    for value in (feature,representation,verification):assert value['complete'] and value['PASS']
    assert representation['new_fits']==0 and not representation['method_success']
    assert not representation['novelty']['all_models_ratio_le1e_4']
    assert verification['audited_representation']==C.bind(C.DIRECTION_DOC/'REPRESENTATION_AUDIT.json')
    assert verification['audited_feature_receipt']==C.bind(C.DIRECTION_DOC/'FEATURE_AUDIT.json')
    inputs=dict(prior['inputs'],prefit_review=C.bind(C.DOC/'PREFIT_REVIEW.json'),
        previous_sign_prefit=C.bind(C.SIGN_DOC/'PREFIT_REVIEW.json'),
        previous_sign_protocol=C.bind(C.SIGN_DOC/'TRAIN_PROTOCOL.json'),
        direction_features=C.bind(C.DIRECTION_RAW/'TRAIN_DIRECTIONS.npz'),
        direction_receipt=C.bind(C.DIRECTION_DOC/'FEATURE_AUDIT.json'),
        direction_representation=C.bind(C.DIRECTION_DOC/'REPRESENTATION_AUDIT.json'),
        direction_verification=C.bind(C.DIRECTION_DOC/'VERIFICATION.json'),
        direction_protocol=C.bind(C.DIRECTION_DOC/'AUDIT_PROTOCOL.json'),
        source_metadata=C.bind(C.PARENT_RAW/'SOURCE_INPUTS.json'))
    assert feature['directions']==inputs['direction_features']
    codes=[C.bind(C.HERE/name) for name in ('common.py','convex_train.py','direction_features.py',
        'evaluate_source.py','evaluate_real.py','seal_training.py','prefit_review.py')]+prior['codes']
    codes += C.read(C.DIRECTION_DOC/'REPRESENTATION_PROTOCOL.json')['codes']
    codes=list({b['path']:b for b in codes}.values())
    p=dict(prior)
    p.update(T.metadata())
    p.update(schema='pallet_pose_signed_axes_direction_training_v1',created_at=C.now(),
        objective='For each frame average Huber(p-y,delta1)+1{y!=0}*softplus(-sign(y)*p) over ORIGINAL valid candidates and both axes. Mean over all2598 frames including the all-invalid zero-loss row, then add lambda/2 times Frobenius norm squared of all271x2 weights. lambda=1e-4; sign coefficient exactly1.',
        changed_factor='Append only the fixed18 signed reprojection residual features to the previous253 feature differences. Preserve old253 inputs, signed targets, original valid masks, frame denominator, scales, old94 normalization, fixed RBF basis, four candidate pools, zero initialization, ridge, Newton/Armijo settings, runtime max-of-two with old ties, and every source/real criterion.',
        architecture='Shared271x2 linear regression: old253 candidate-minus-operational-anchor features concatenated with18 fixed normalized signed residual differences; max of the two outputs supplies the whole-pose score.',
        feature_transform='Preserve prior normalized94+abs-anchor94+identity1+fixedRBF64 and its candidate-minus-R0-anchor subtraction. Append projected-minus-observed9xy divided by bbox diagonal, float32 normalized by frozen O R0 TRAIN5194 mean/std, then float64 candidate-minus-same-anchor. No refitting old normalization/basis. Invalid differences and anchor differences are exactly zero.',
        direction_receipt_binding=inputs['direction_receipt'],
        direction_normalization_sha=feature['normalization']['array_sha'],
        direction_mean=feature['normalization']['mean18'],direction_std=feature['normalization']['std18'],
        direction_source='Frozen TRAIN-only O audit; same2598 source rows including1 all-invalid row; no extraction from VAL or real for normalization.',
        direction_projection_contract=dict(points=9,dimensions=18,sign='projected_minus_observed',
            coordinates='Preserve already padded source q/K and raw real q/K; no extra padding or C2 transformation.',
            pose='Existing candidate R_cf, centroid, cf_extents; eight corners plus cuboid origin at index8.',
            denominator='sqrt(sum(max(bbox_wh,1e-6)^2))',dtype='float32',parity_atol_px=1e-4,parity_rtol=1e-6),
        certificate_definition='Unchanged Huber plus masked sign logistic of affine outputs is convex and continuously differentiable. Ridge on all542 parameters makes it lambda-strongly convex, so the unchanged gradient-gap certificate applies. Float64 verification is not interval arithmetic and gives no T/R/generalization guarantee.',
        solver_accounting='Initial objective and every trial count toward the unchanged2000-call cap; only accepted points count toward the1000-iteration cap. Use the last accepted point and its already computed gradient. Two271-dimensional SPD generalized Hessian blocks, fixed Armijo; retain logistic curvature and lambda I at Huber kinks. No damping, warm start, restart, tolerance search or budget extension.',
        generalization_limit='The O audit shows only input linear nonredundancy; zero exact opposite-sign collisions does not prove sufficient representation. Previous converged sign fits still fail source43/45, and RGB/DHT transfer failures are counterevidence. New18 derives from existing predictions and poses, not independent RGB evidence. Reused VAL/real DEV is not new independent TEST.',
        checkpoint='Four final271x2 zero-initialized fits, immutable input and normalization hashes, all coefficients and certificates; no validation checkpoint selection.',
        inputs=inputs,codes=codes,
        evidence=dict(previous_protocol=C.bind(C.SIGN_DOC/'TRAIN_PROTOCOL.json'),
            previous_train_verification=C.bind(C.SIGN_DOC/'TRAIN_CONVERGENCE.json'),
            previous_source_gate=C.bind(C.SIGN_DOC/'SOURCE_VAL_GATE.json'),
            input_representation=C.bind(C.DIRECTION_DOC/'REPRESENTATION_AUDIT.json'),
            input_verification=C.bind(C.DIRECTION_DOC/'VERIFICATION.json'),
            design=C.bind(C.DOC/'DESIGN_KO.md'),
            original_goal_protocol=prior['evidence']['original_goal_protocol']),
        evidence_access='The sealer reads previous failure evidence. Fitting may read frozen TRAIN input diagnostics but never prior weights or source VAL/real-quality evidence; all four fits start from zero.',
        method_success=False,goal_complete=False)
    unchanged=('source_val','real_evaluation','original_goal_criteria','real_stability_contract',
        'original_goal_evaluation','original_goal_contract_correction',
        'candidate_pool','inference_tie','no_valid_fallback','runtime_inputs','models','train_rows','max_fits',
        'lambda_l2','normalization','initialization','target','scale_definition','anchor_rule','runtime_safety',
        'raw_feature_dim','context_dim','rbf_dim','output_dim','target_rule','prediction_rule','huber_delta',
        'basis_SHA_bind','solver_rule','solver','certificate','solver_success','stop',
        'loss_rule','sign_rule','sign_coefficient','sign_supervision','bias')
    for key in unchanged:assert p[key]==prior[key],key
    for key,value in T.metadata().items():assert p[key]==review[key]==value,key
    assert p['solver']==T.solver_config() and p['sign_coefficient']==1.
    assert p['feature_dim']==271 and p['base_feature_dim']==253 and p['direction_dim']==18
    assert p['basis_SHA_bind']==review['basis_SHA_bind']==C.bind(C.RBF_DOC/'RBF_BASIS.json')
    assert review['direction_receipt_binding']==inputs['direction_receipt']
    assert review['direction_normalization_sha']==p['direction_normalization_sha']
    for b in [*inputs.values(),*codes,*p['evidence'].values()]:C.verify(b)
    C.save(C.DOC/'TRAIN_PROTOCOL.json',p)
    C.save(C.DOC/'TRAIN_PROTOCOL_SHA.json',C.bind(C.DOC/'TRAIN_PROTOCOL.json'))
    print('SIGNED_AXES_DIRECTION_PROTOCOL_SEALED',C.bind(C.DOC/'TRAIN_PROTOCOL.json'))


if __name__=='__main__':main()
