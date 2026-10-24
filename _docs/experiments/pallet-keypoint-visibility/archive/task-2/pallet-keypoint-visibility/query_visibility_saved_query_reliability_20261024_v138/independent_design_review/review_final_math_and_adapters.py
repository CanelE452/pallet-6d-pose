"""Pre-GT independent code review receipt and invented-array math checks."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent


def bind(p):
    return {'path': str(Path(p).resolve()), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()}


def module(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    geom_path = V / 'input_preparation/reliability_math.py'
    auc_path = V / 'diagnosis_preparation/observability_math.py'
    target_path = V / 'diagnosis_preparation/target_contract.py'
    g = module(geom_path, 'independent_pure_input_math')
    a = module(auc_path, 'independent_pure_auc_math')
    t = module(target_path, 'independent_pure_target_math')
    sign = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float)
    xyz = sign * [.7, .15, .6]
    projection = np.array([[530.,20.,30.,1050.],[17.,460.,-12.,930.],[.18,.15,.9,5.]])
    hom = np.c_[xyz, np.ones(8)] @ projection.T; q = hom[:, :2] / hom[:, 2, None]
    assert all(g.edge_signals(q, k)['global_rms'] < 1e-12 and g.edge_signals(q, k)['heldout_rms'] < 1e-12 for k in range(8))
    bad = q.copy(); bad[0] += [9., -4.]
    measured = g.edge_signals(bad, 0)
    assert measured['global_rms'] > 1e-5 and measured['heldout_rms'] > 1e-3
    for equivalent in (bad + [2000., -1700.], bad * 1e-4):
        actual = g.edge_signals(equivalent, 0)
        for key in ('global_group_residual', 'heldout_group_residual'):
            assert np.allclose(actual[key], measured[key], rtol=1e-10, atol=1e-12)
    affine = xyz @ np.array([[130.,17.],[31.,96.],[58.,-23.]]) + [270.,220.]
    assert all(g.edge_signals(affine, k)['heldout_rms'] < 1e-12 for k in range(8))
    collapsed = q.copy(); collapsed[0] = collapsed[1]
    other = [g.line_through(collapsed[i], collapsed[j]) for i, j in g.EDGE_GROUPS[0][1:]]
    assert g.null_of_lines(other)[0] is not None
    out = g.edge_signals(collapsed, 0)
    assert not out['heldout_group_valid'][0] and np.isnan(out['heldout_rms'])
    for degenerate in (np.tile([20.,30.], (8,1)), np.c_[np.arange(8), 2*np.arange(8)+5], None):
        out = g.edge_signals(degenerate, 0)
        assert np.isnan(out['global_rms']) and np.isnan(out['heldout_rms'])
    score = np.array([0., 1., 1., np.nan, 2., -1.]); target = np.array([0, 1, 0, 1, 1, -1])
    result = a.fixed_auc(score, target, np.ones(6, bool))
    positives = score[(target == 1) & np.isfinite(score)]; negatives = score[(target == 0) & np.isfinite(score)]
    pairwise_auc = float(np.mean([float(p > n) + .5 * float(p == n) for p in positives for n in negatives]))
    assert result['AUC_fixed_higher_score_positive'] == pairwise_auc == .875
    assert result['unranked_positive'] == 1 and result['target_unknown'] == 1
    bins, proof = a.train_input_quintiles([np.nan,np.nan,3.], [True,True,False])
    assert bins.tolist() == [-1,-1,-2] and proof['no_finite_TRAIN_reference']
    counts = a.target_bin_counts(bins, [-1,1,0], np.ones(3,bool))
    assert sum(r['original_rows'] for r in counts.values()) == 3 and counts['NO_TRAIN_REFERENCE']['original_rows'] == 1
    assert a.generated_checks()['PASS'] and t.generated_checks()['PASS']
    files = ['ROOT_SAVED_INPUT_RELIABILITY_POLICY_BEFORE_FEATURES.json', 'ROOT_PROSPECTIVE_EXTERNAL_SIGNAL_TARGET_ADDENDUM_BEFORE_GT.json',
        'input_preparation/build_input_reliability.py', 'input_preparation/reliability_math.py', 'input_preparation/FEATURE_SCHEMA.json',
        'input_preparation/PROTOCOL_BEFORE_INPUT_BUILD.json', 'diagnosis_preparation/diagnose_input_observability.py',
        'diagnosis_preparation/PROTOCOL_BEFORE_GT_JOIN.json', 'diagnosis_preparation/observability_math.py',
        'diagnosis_preparation/target_contract.py', 'diagnosis_preparation/PROSPECTIVE_DIAGNOSTIC_DESIGN.json']
    receipt = dict(completed=True, PASS=True, stage='Independent code/design and invented-array review before GT join; not an actual saved-result audit.',
        created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), code=bind(__file__), reviewed_bindings={f: bind(V/f) for f in files},
        static_cuboid_certificate=bind(H/'STATIC_REGISTERED_CUBOID_CF_EDGE_CERTIFICATE.json'), material_blockers=[],
        resolved_before_input_build=['Own collapsed heldout edge stays undefined even when other3 lines have rank2 support.',
            'Exact maxspan normalization preserves subpixel uniform scale invariance.',
            'Finite score with no TRAIN cut reference uses separate NO_TRAIN_REFERENCE bin.',
            'G42/sentinel zero placeholders are masked before ranks/bins.'],
        independent_invented_checks=dict(projected_cuboid_incidence_zero=True, single_corner_perturbation_positive=True,
            global_and_heldout_translation_and_subpixel_scale_invariant=True, infinite_vanishing_point_valid=True,
            own_zero_edge_unknown_with_other_rank2_support=True, rank1_and_collapsed_and_missing_unknown=True,
            tied_AUC_matches_independent_pairwise_definition=True, undefined_inputs_unknown_targets_and_no_TRAIN_reference_counts_preserved=True,
            pure_AUC_and_target_generated_checks_PASS=True),
        producer_review=['Original q8/L and all26624 IDs/slots retained; no GT/factor/modelscore inputs.',
            'Exact V74G42, noquery/sentinel descriptor masks and original factual/context availability.',
            'Only valid96/global_index/crop_constructed members loaded; row valid96 hashes checked; RGB body neither loaded nor whole-file-hashed.',
            'SVD is a numerical 2D line-incidence descriptor; no fitted camera/pose or corrected query is emitted.'],
        diagnosis_review=['Durable input-only TRAIN-bin seal precedes all GT/C29 score body reads/hashes.',
            'Fixed10 risks plus58 raw positive-value signed descriptive signals; no sign/feature/model/threshold selection.',
            'All original known/unknown/undefined/U/SU counts retained with availability-only supplements.',
            'GTerror, available D/E mistake and all-known D/E failure including predictionU remain separate.',
            'Teacher D/E-conditioned confidence analyses and prospective E1/D0 target diagnose occlusion versus position confounding.'],
        interpretation_limits=['Low concurrence cannot certify absolute coordinates or native identity; common projective transforms can preserve it.',
            'Heldout edge lines omit the incident edge, but common normalization uses all8 queries; no strict q-independent predictor claim.',
            'Availability flags partly define inputU failures; association is partly definitional.',
            'Univariate source association proves neither causality nor real transfer nor multivariate/image unobservability.'],
        actual_input_table_GT_C29score_RGB_visualfeatures_checkpoint_model_fit_private_V112_PnP_reads_or_calls=0,
        other_agent_files_modified=False, policy_gates_labels_coordinates_changed=False)
    path = H / 'PRE_GT_FINAL_DESIGN_AND_MATH_REVIEW.json'
    with path.open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(completed=True, PASS=True, receipt=bind(path))), flush=True)


if __name__ == '__main__':
    main()
