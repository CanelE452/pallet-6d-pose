"""Fix the TRAIN-input-only RBF center rule before any basis construction."""
from . import common as C


def main():
    assert not (C.DOC / 'RBF_BASIS.json').exists()
    assert not (C.RAW / 'fits').exists()
    prior = C.read(C.RISK_DOC / 'TRAIN_PROTOCOL.json')
    C.verify(C.read(C.RISK_DOC / 'TRAIN_PROTOCOL_SHA.json'))
    inputs = dict(source_contract=prior['inputs']['source_contract'],
        source_features=prior['inputs']['features'], source_feature_lock=prior['inputs']['feature_lock'],
        parent_train_protocol=C.bind(C.RISK_DOC / 'TRAIN_PROTOCOL.json'))
    codes = [C.bind(C.HERE / n) for n in ('common.py', 'basis.py', 'seal_basis.py')]
    codes += prior['codes']
    codes = list({b['path']: b for b in codes}.values())
    p = dict(schema='pallet_pose_anchor_rbf_basis_protocol_v1', created_at=C.now(),
        train_rows=2598, context_dim=189, rbf_dim=64, feature_dim=253,
        feature_map='normalized94_abs_anchor_delta94_identity1_fixed_rbf64',
        center_rule='sha256_identity_first64_byte_distinct_float64_context',
        identity_encoding='UTF-8 JSON [frame_id, expert, hypothesis], ensure_ascii=False, separators=(comma,colon). Sort by SHA256 hex then encoded identity; choose first64 byte-distinct contiguous float64 context vectors.',
        candidate_pool=['R0', 'DIVERSE251_s1', 'DIVERSE251_s2', 'DIVERSE251_s3'],
        duplicate_rule='The R0 identity occurs only once per frame and hypothesis. Distinct identities with exactly equal float64 context bytes contribute only the first sorted vector.',
        eligibility_rule='Use only pre-existing SOURCE_CONTRACT.fit_eligibility.eligible_ids.TRAIN; preserve all2598 eligible rows. Validity and operational R0_GEO_index come from features. No labels or target quality participate in center selection.',
        bandwidth_rule='median_positive_pairwise_squared_center_distance',
        pair_rule='All unordered i<j center pairs; squared Euclidean float64 distances; retain finite strictly positive values. No distance-based center selection.',
        basis_formula='exp(-sum((context-center)**2)/(2*bandwidth_squared))',
        normalization='old_float32_then_float64',
        normalization_source='R0 eligible TRAIN valid raw94 candidates only; float32 mean/std and unchanged1e-6 std floor; then original189 anchor context.',
        extra_normalization=False, invalid_feature_rule='All253 coordinates zero on invalid candidates, including64 RBF columns.',
        inputs=inputs, codes=codes,
        evidence=dict(previous_fixed_result=C.bind(C.RISK_DOC / 'REAL_RESULTS.json'),
            frozen_choice_diagnosis=C.bind(C.RISK_DOC / 'SELECTOR_TRANSFER_DIAGNOSTIC.json')),
        evidence_scope='Sealer-only rationale; builder reads input/code bindings only, not previous performance, labels, real data or VAL quality.',
        source_container_disclosure='SOURCE_FEATURES.npz also contains frozen VAL inputs; only eligible TRAIN indices supply normalization, candidate contexts, centers and bandwidth.',
        calibration='One predeclared center count64 and one median bandwidth formula; no sweeps or data-driven adjustment.',
        stop='Fewer than64 byte-distinct valid context vectors, missing operational R0 anchor with valid candidates, or nonpositive/nonfinite bandwidth stops construction. No alternative count, seed, center rule, width or retry.',
        new_fits=0, real_targets_read=False, source_TRAIN_labels_read=False, VAL_quality_read=False,
        method_success=False, goal_complete=False)
    for b in [*inputs.values(), *codes, *p['evidence'].values()]:
        C.verify(b)
    C.save(C.DOC / 'BASIS_PROTOCOL.json', p)
    C.save(C.DOC / 'BASIS_PROTOCOL_SHA.json', C.bind(C.DOC / 'BASIS_PROTOCOL.json'))
    print('RBF_BASIS_PROTOCOL_SEALED', C.bind(C.DOC / 'BASIS_PROTOCOL.json'))


if __name__ == '__main__':
    main()
