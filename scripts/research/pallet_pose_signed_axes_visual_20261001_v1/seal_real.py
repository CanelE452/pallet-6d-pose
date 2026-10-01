"""Seal the input-only real contract only after all source gates pass."""
from . import common as C
from . import evaluate_real as E


def main():
    assert not E.REAL_PROTOCOL.exists() and not E.ROUTE.exists()
    assert not E.ROUTE_LOCK.exists() and not E.RESULTS.exists()
    verification = C.read(C.DOC / 'SOURCE_VAL_VERIFICATION.json')
    assert verification['complete'] and verification['PASS']
    assert verification['checks_passed'] == verification['checks_total'] == 45
    p = E.real_protocol_spec()
    assert p['frames'] == 173 and p['source_PASS_required']
    assert len(p['models']) == 4 and len(p['baselines']) == 9
    assert p['real_stability_contract'] == 'matched_intervention_AND_original_SINGLE251_stability'
    assert p['runtime_uses_fixed_rbf'] and p['basis_SHA_bind'] == C.bind(C.RBF_DOC / 'RBF_BASIS.json')
    assert p['direction_dim'] == 18 and p['base_feature_dim'] == 253 and p['feature_dim'] == 656
    assert p['visual_dim']==385 and p['previous_feature_dim']==271
    assert p['visual_receipt_binding']==C.bind(C.NATIVE_DOC/'TRAIN_APPEARANCE_INPUTS.json')
    assert p['visual_protocol_binding']==C.bind(C.NATIVE_DOC/'INPUT_PROTOCOL.json')
    assert p['visual_verification_binding']==C.bind(C.NATIVE_DOC/'INPUT_VERIFICATION.json')
    assert p['direction_receipt_binding'] == C.bind(C.DIRECTION_DOC/'FEATURE_AUDIT.json')
    assert p['runtime_uses_margin'] is False and not E._REFERENCES['allowed']
    C.save(E.REAL_PROTOCOL, p)
    C.save(C.DOC / 'REAL_PROTOCOL_SHA.json', C.bind(E.REAL_PROTOCOL))
    print('SIGNED_AXES_VISUAL_REAL_PROTOCOL_SEALED_NO_REFERENCE_VALUES', C.bind(E.REAL_PROTOCOL))


if __name__ == '__main__':
    main()
