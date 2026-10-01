"""Seal a TRAIN-only representation audit, with no selector training authority."""
import argparse
from . import common as C


def settings():
    return dict(frames=2598,models=['R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3'],
        scorer_models=list(C.MODEL_NAMES),source_TRAIN_only=True,direction_dim=18,
        direction_rule='BBOX_DIAGONAL_NORMALIZED_PROJECTED_MINUS_OBSERVED_XY9',
        parity=dict(atol_px=1e-4,rtol=1e-6),
        normalization=dict(source='R0_eligible_train_valid_candidates',std_floor=1e-6),
        representation=dict(base_dimension=253,extra_dimension=18,extended_dimension=271,
            zero_byte_rule='canonicalize_both_signed_zeros_to_positive_zero_no_rounding',
            svd_population='all_valid_nonanchor_candidates',svd_tolerance='max(shape)*float64_eps*smax',
            novelty_frobenius_max=1e-4,
            normalization='R0_TRAIN_valid_float32_mean_std_floor_1e-6_then_float64_candidate_minus_anchor',
            target_use='exact_collision_diagnostics_only_not_projection_or_fitting'),
        representation_scope=dict(
            decision='If all four residual Frobenius ratios are <=1e-4, reject linear nonredundancy as motivation for this addition. Otherwise report input nonredundancy only; this does not authorize a fit or predict performance.',
            collision_scope='Exact numerical vectors after only signed-zero canonicalization. Preserve exact targets; report distinct sign and strict opposite-sign conflicts separately. Separate forced anchors, nonanchors, cross-role collisions, split and unresolved groups.',
            projection_scope='SVD/orthogonal decomposition of input columns only. Labels are never passed to the span function; no target regression, fitted selector, candidate scoring or argmin.'),
        budgets=dict(new_selector_fits=0,new_optimizer_steps=0,new_image_forwards=0,
            new_PnP_solves=0,new_pose_error_calculations=0,new_policy_argmin_calls=0,
            VAL_quality_reads=0,real_reference_reads=0,real_routes=0),
        coordinates='Use the frozen camera-facing R_cf/centroid/cf_extents and existing padded K/observed q9. Cuboid corners0..7 and origin P8. No added padding, physical/C2 basis remapping, PnP or pose generation.',
        feature_container_disclosure='Historical source feature/pose/metadata containers include5120 TRAIN/VAL rows. Select only the2598 pre-existing eligible TRAIN IDs for projection and diagnostics; no VAL prediction files or quality are opened.',
        invalid_policy='Preserve original valid mask, one all-invalid TRAIN row and finite zero placeholders. Stop on parity/nonfinite mismatch; no candidate pruning or posthoc tolerance adjustment.',
        interpretation='Norm alone discards direction, but this is not a proof of collision/insufficiency of the full94 PnP-derived feature manifold. Nonredundancy of18 columns is not identification of physical improvement or domain transfer.',
        stage_transition='Feature parity and independent representation audit are diagnostic prerequisites only. Any future271-feature training needs a separate predeclared protocol, unchanged four model pools/source45 and original plus matched real5 criteria.',
        method_success=False,goal_complete=False)


def features():
    assert not (C.DOC/'AUDIT_PROTOCOL.json').exists()
    assert not (C.RAW/'TRAIN_DIRECTIONS.npz').exists()
    gate=C.read(C.SIGN_DOC/'SOURCE_VAL_GATE.json')
    assert gate['complete'] and not gate['PASS'] and gate['checks_passed']==43
    assert C.read(C.SIGN_DOC/'REAL_EVALUATION_NOT_RUN.json')['learned_real_routes']==0
    lock=C.read(C.PARENT_DOC/'SOURCE_FEATURE_LOCK.json')
    inputs=dict(source_contract=C.bind(C.PARENT_DOC/'SOURCE_CONTRACT.json'),
        feature_lock=C.bind(C.PARENT_DOC/'SOURCE_FEATURE_LOCK.json'),
        features=lock['features'],poses=lock['poses'],metadata=lock['metadata'],
        source_predictions_lock=lock['predictions'])
    for binding in inputs.values():C.verify(binding)
    eligible=C.read(C.ROOT/inputs['source_contract']['path'])['fit_eligibility']['eligible_ids']['TRAIN']
    gallery_ids=[]
    for family in ('G38','P0','TEX'):
        family_ids=sorted(fid for fid in eligible if fid.startswith(family+'__'))
        assert len(family_ids)>=2
        gallery_ids.extend(family_ids[:2])
    assert len(gallery_ids)==len(set(gallery_ids))==6
    codes=[C.bind(C.HERE/name) for name in ('common.py','direction_features.py',
        'representation_audit.py','seal.py')]
    # Authenticate the full frozen helper lineage used to reconstruct253/targets.
    previous=C.read(C.SIGN_DOC/'TRAIN_PROTOCOL.json')
    C.verify(C.read(C.SIGN_DOC/'TRAIN_PROTOCOL_SHA.json'))
    codes+=previous['codes']
    codes=list({binding['path']:binding for binding in codes}.values())
    for binding in codes:C.verify(binding)
    p=dict(schema='pallet_pose_residual_direction_input_audit_v1',complete=True,
        created_at=C.now(),inputs=inputs,codes=codes,**settings(),
        gallery=dict(ids=gallery_ids,selection='First two lexicographically sorted eligible TRAIN IDs for each existing G38/P0/TEX family, fixed before direction extraction; no error/target/learned outcome selection.',
            scope='TRAIN input illustrations only, physical dimensions from the original source metadata, frozen operational R0 observations/projection; no reference-pose outline or learned improvement claim.',
            display_arrow_scale=20.,new_image_forwards=0,real_images_or_reference_reads=0),
        evidence=dict(previous_source_gate=C.bind(C.SIGN_DOC/'SOURCE_VAL_GATE.json'),
            previous_diagnostic=C.bind(C.SIGN_DOC/'SOURCE_TRANSFER_DIAGNOSTIC.json'),
            previous_report=C.bind(C.SIGN_DOC/'REPORT_KO.md'),
            prior_objective_audit=C.bind(C.ROOT/'_docs/experiments/pallet_pose_selector_objective_audit_20261001_v1/PRIOR_OBJECTIVE_AUDIT_KO.md')),
        evidence_access='Only the sealer reads old performance/rationale. The extractor reads only declared input bindings and code, never the evidence objects or cached TRAIN label file.')
    C.save(C.DOC/'AUDIT_PROTOCOL.json',p)
    C.save(C.DOC/'AUDIT_PROTOCOL_SHA.json',C.bind(C.DOC/'AUDIT_PROTOCOL.json'))
    print('DIRECTION_INPUT_PROTOCOL_SEALED',C.bind(C.DOC/'AUDIT_PROTOCOL.json'))


def representation():
    assert not (C.DOC/'REPRESENTATION_PROTOCOL.json').exists()
    before=C.protocol('AUDIT_PROTOCOL')
    for key,value in settings().items():assert before[key]==value,key
    review=C.read(C.DOC/'FEATURE_AUDIT.json')
    assert review['complete'] and review['PASS']
    C.verify(review['directions'])
    assert review['directions']==C.bind(C.RAW/'TRAIN_DIRECTIONS.npz')
    p=dict(before)
    p.update(schema='pallet_pose_residual_direction_representation_audit_v1',created_at=C.now(),
        feature_protocol=C.bind(C.DOC/'AUDIT_PROTOCOL.json'),
        inputs=dict(source_contract=before['inputs']['source_contract'],
            source_features=before['inputs']['features'],
            source_train_labels=C.bind(C.PARENT_RAW/'SOURCE_TRAIN_LABELS.npz'),
            source_feature_lock=before['inputs']['feature_lock'],
            parent_train_protocol=C.bind(C.PARENT_DOC/'TRAIN_PROTOCOL.json'),
            sign_prefit=C.bind(C.SIGN_DOC/'PREFIT_REVIEW.json'),
            rbf_basis=C.bind(C.RBF_DOC/'RBF_BASIS.json'),
            direction_features=review['directions'],direction_receipt=C.bind(C.DOC/'FEATURE_AUDIT.json')),
        label_access='Only the already frozen eligible TRAIN2598 candidate errors/targets may be read. They are used for exact-collision descriptions, never for input span decomposition, feature choice, fitting, candidate argmin or new pose error calculations.')
    for binding in p['inputs'].values():C.verify(binding)
    C.save(C.DOC/'REPRESENTATION_PROTOCOL.json',p)
    C.save(C.DOC/'REPRESENTATION_PROTOCOL_SHA.json',C.bind(C.DOC/'REPRESENTATION_PROTOCOL.json'))
    print('DIRECTION_REPRESENTATION_PROTOCOL_SEALED',C.bind(C.DOC/'REPRESENTATION_PROTOCOL.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('features','representation'))
    globals()[parser.parse_args().stage]()
