"""Seal one input addition before four fresh, unchanged-objective fits."""
from . import common as C
from . import convex_train as T


def main():
    assert not (C.RAW/'fits').exists()
    review=C.read(C.DOC/'PREFIT_REVIEW.json')
    assert review['complete'] and review['PASS'] and review['new_fits']==0
    assert review['trainer']==C.bind(T.__file__)
    C.verify(review['trainer']);C.verify(review['reviewer'])
    C.verify(C.read(C.Q_DOC/'TRAIN_PROTOCOL_SHA.json'))
    previous=C.read(C.Q_DOC/'TRAIN_PROTOCOL.json')
    previous_review=C.read(C.Q_DOC/'PREFIT_REVIEW.json')
    previous_training=C.read(C.Q_DOC/'TRAIN_CONVERGENCE.json')
    previous_gate=C.read(C.Q_DOC/'SOURCE_VAL_GATE.json')
    assert previous_training['complete'] and previous_training['PASS']
    assert previous_gate['complete'] and not previous_gate['PASS']
    assert previous_gate['checks_passed']==43 and previous_gate['checks_total']==45
    native=C.read(C.VISUAL_DOC/'TRAIN_APPEARANCE_INPUTS.json')
    native_verification=C.read(C.VISUAL_DOC/'INPUT_VERIFICATION.json')
    assert native['complete'] and native['input_construction_pass']
    assert native_verification['complete'] and native_verification['PASS']
    assert native['frames']==2598 and native['fits_executed']==0 and native['image_forwards']==0
    inputs=dict(previous['inputs'])
    inputs.update({key:value for key,value in review['inputs'].items()
                   if isinstance(value,dict) and 'path' in value and 'sha256' in value})
    inputs['prefit_review']=C.bind(C.DOC/'PREFIT_REVIEW.json')
    expected=dict(visual_features=C.bind(C.VISUAL_RAW/'TRAIN_APPEARANCE.npz'),
        visual_receipt=C.bind(C.VISUAL_DOC/'TRAIN_APPEARANCE_INPUTS.json'),
        visual_protocol=C.bind(C.VISUAL_DOC/'INPUT_PROTOCOL.json'),
        visual_verification=C.bind(C.VISUAL_DOC/'INPUT_VERIFICATION.json'),
        previous_asymmetric_prefit=C.bind(C.Q_DOC/'PREFIT_REVIEW.json'),
        previous_asymmetric_protocol=C.bind(C.Q_DOC/'TRAIN_PROTOCOL.json'))
    for key,value in expected.items():
        assert inputs[key]==value,key
    names=('common.py','convex_train.py','direction_features.py','appearance_features.py',
           'evaluate_source.py','evaluate_real.py','seal_real.py','seal_training.py','prefit_review.py')
    codes=[C.bind(C.HERE/name) for name in names]+previous['codes']
    codes += [C.bind(C.NATIVE_HERE/'visual_features.py'),
              C.bind(C.ROOT/'scripts/research/pallet_pose_dino_input_audit_20261001_v1/visual_features.py'),
              C.bind(C.ROOT/'scripts/research/pallet_pose_dino_input_audit_20261001_v1/freeze_train.py'),
              C.bind(C.ROOT/'scripts/research/pallet_sensors_submission_v1/posefix_contract_math.py')]
    codes=list({b['path']:b for b in codes}.values())
    p=dict(previous)
    p.update(T.metadata())
    p.update(schema='pallet_pose_signed_axes_visual_training_v1',created_at=C.now(),
        objective='Unchanged full-frame original-valid-candidate two-axis asymmetric Huber plus masked sign logistic and lambda/2 Frobenius ridge; now all656x2 weights. lambda1e-4, underprediction cost2:1, sign coefficient1, bias0.',
        changed_factor='Append385 frozen candidate-local native-image DINO appearance differences to the unchanged271 inputs. No loss, solver, target, candidate pool, tie, gate, old normalization or RBF changes. Feature count and model capacity increase together.',
        certificate_definition='The same convex differentiable asymmetric Huber/sign loss and ridge apply to an enlarged fixed linear input. All1312 parameters carry lambda1e-4; the same gradient-gap bound applies. Floating-point certification does not certify T/R generalization.',
        solver_accounting='Unchanged Q full generalized-Hessian construction, two656-dimensional SPD blocks and Newton/Armijo policy. Maximum1000 accepted iterations/2000 objective calls; zero initialization; no damping, warm start, restart or tolerance/budget changes.',
        generalization_limit='PriorQ converged but source43/45 failed. Raw frozen image evidence is a single followup, not a claim that DINO or whole-candidate RGB selection is new or generally successful. Mean pooling loses corner topology; contextual padding remains. Source VAL and real DEV have prior exposure.',
        checkpoint='Exactly four final656x2 zero-initialized fits, no validation checkpoint selection.',
        inputs=inputs,codes=codes,
        visual_receipt_binding=expected['visual_receipt'],
        visual_protocol_binding=expected['visual_protocol'],
        visual_verification_binding=expected['visual_verification'],
        visual_normalization_sha=native['normalization']['array_sha'],
        visual_mean=native['normalization']['mean385'],visual_std=native['normalization']['std385'],
        evidence=dict(previous['evidence'],
            previous_asymmetric_protocol=C.bind(C.Q_DOC/'TRAIN_PROTOCOL.json'),
            previous_asymmetric_training=C.bind(C.Q_DOC/'TRAIN_CONVERGENCE.json'),
            previous_asymmetric_source_gate=C.bind(C.Q_DOC/'SOURCE_VAL_GATE.json'),
            visual_input_protocol=expected['visual_protocol'],
            visual_input_verification=expected['visual_verification'],
            prepared_padding_audit=C.bind(C.APPEARANCE_DOC/'PADDING_SUPPORT_AUDIT.json'),
            design=C.bind(C.DOC/'DESIGN_KO.md')),
        evidence_access='Sealer reads prior outcome evidence. Fits consume fixed TRAIN inputs and cached TRAIN targets only, never prior weights or VAL/real quality. A separately verified embedded-Q weight comparison is diagnostic, never initialization.',
        method_success=False,goal_complete=False)
    unchanged=('source_val','real_evaluation','original_goal_criteria','real_stability_contract',
        'original_goal_evaluation','original_goal_contract_correction','candidate_pool','inference_tie',
        'no_valid_fallback','runtime_inputs','models','train_rows','max_fits','lambda_l2','normalization',
        'initialization','target','scale_definition','anchor_rule','runtime_safety','bias','basis_SHA_bind',
        'solver','certificate','solver_success','stop','sign_supervision','direction_receipt_binding',
        'direction_normalization_sha','direction_mean','direction_std','direction_projection_contract',
        'loss_rule','huber_delta','sign_coefficient','underprediction_coefficient','underprediction_cost',
        'overprediction_cost','underprediction_zero_curvature')
    for key in unchanged:
        assert p[key]==previous[key],key
    for key,value in T.metadata().items():
        assert p[key]==review[key]==value,key
    assert p['solver']==T.solver_config() and p['feature_dim']==656
    for model in C.MODEL_NAMES:
        for key in previous_review['models'][model]:
            if key in ('signed_target_sha','input_difference_sha','base_context_sha','errors_sha',
                       'scaled_excess_sha','original_valid_sha','direction_raw_sha',
                       'direction_difference_sha','extended_input_sha'):
                assert review['models'][model][key]==previous_review['models'][model][key],(model,key)
    for b in [*inputs.values(),*codes,*p['evidence'].values()]:
        C.verify(b)
    C.save(C.DOC/'TRAIN_PROTOCOL.json',p)
    C.save(C.DOC/'TRAIN_PROTOCOL_SHA.json',C.bind(C.DOC/'TRAIN_PROTOCOL.json'))
    print('VISUAL656_TRAIN_PROTOCOL_SEALED',C.bind(C.DOC/'TRAIN_PROTOCOL.json'),flush=True)


if __name__=='__main__':main()
