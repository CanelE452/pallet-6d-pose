"""Fixed-geometry contract certificate and model-control readiness review.

Reads only fixed registered mesh anchors, source code and saved JSON metadata.
No per-image GT, input features, RGB, checkpoint, model, fit or PnP is used.
"""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
POLICY = V / 'ROOT_SAVED_INPUT_RELIABILITY_POLICY_BEFORE_FEATURES.json'
POLICY_SHA = 'cd94b9d260d0d2111dc9ad0ced34dd437577a0c48c799f8654db5ba6973cccfd'
GEOMETRY = P / 'query_visibility_spatial_stencil_control_20261023_v127/registered_native_teacher_feasibility/RESULT.json'
MATH = P / 'query_visibility_registered_native_diversity_20261023_v129/render_preparation/corpus_math.py'
G42_CODE = P / 'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py'
G42_SEAL = P / 'query_visibility_native_geometry_control_20261024_v133/input_geometry_v1/INPUT_GEOMETRY_SEAL.json'
HEAD_CODE = P / 'query_visibility_native_geometry_control_20261024_v133/source_fit_preparation/train_matched_geometry.py'
SIGN = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
FAMILIES = {'width': [[0,1],[3,2],[4,5],[7,6]], 'vertical': [[0,3],[1,2],[4,7],[5,6]], 'depth': [[0,4],[1,5],[2,6],[3,7]]}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bind(p):
    return {'path': str(Path(p).resolve()), 'sha256': sha(p)}


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    assert sha(POLICY) == POLICY_SHA
    tree = ast.parse(MATH.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'quarter_mapping')
    ns = {'np': np, 'SIGNS': SIGN}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(MATH) + '::fixed_contract', 'exec'), ns)
    fixed = read(GEOMETRY); assert fixed['completed']
    proofs = []
    combinations = (SIGN + 1) / 2
    for record in fixed['results']:
        mesh = record['mesh']; assert sha(mesh['path']) == mesh['sha256']
        with np.load(mesh['path'], allow_pickle=False) as f:
            anchors = np.array(f['native_anchor3D'], np.float64, copy=True)
        assert anchors.shape == (8, 3) and np.isfinite(anchors).all()
        center = (anchors.max(0) + anchors.min(0)) / 2; half = np.ptp(anchors, axis=0) / 2
        assert (half > 0).all() and np.array_equal(anchors, center + SIGN * half)
        quarter_proofs = []
        for quarter in range(4):
            _, perm = ns['quarter_mapping'](anchors, quarter)
            points = anchors[perm]
            basis = np.stack([points[1]-points[0], points[3]-points[0], points[4]-points[0]])
            # CF index signs correspond to width/down/depth binary choices.
            reconstructed = points[0] + combinations @ basis
            reconstruction_error = float(np.max(np.abs(reconstructed - points)))
            assert reconstruction_error < 1e-14
            normalized = basis / np.linalg.norm(basis, axis=1)[:, None]
            assert np.max(np.abs(normalized @ normalized.T - np.eye(3))) < 1e-14
            residuals = {}
            for name, edges in FAMILIES.items():
                vector = np.array([points[b] - points[a] for a, b in edges])
                lengths = np.linalg.norm(vector, axis=1); assert (lengths > 0).all()
                normalized_cross = np.linalg.norm(np.cross(vector[0], vector), axis=1) / (lengths[0] * lengths)
                assert normalized_cross.max() < 1e-14
                residuals[name] = dict(nonzero_edges=4, max_normalized_cross=float(normalized_cross.max()), lengths=lengths.tolist())
            quarter_proofs.append(dict(quarter=90 * quarter, CF_to_canonical=perm, max_cuboid_reconstruction_error=reconstruction_error,
                                       parallel_edge_families=residuals))
        proofs.append(dict(mesh=mesh, only_npz_array_read='native_anchor3D', anchor_array_sha256=hashlib.sha256(anchors.tobytes()).hexdigest(),
                           center=center.tolist(), half_extents=half.tolist(), full_SIGNS_cartesian_cuboid_exact=True,
                           fixed_native_anchors_are_not_inset_different_footprints=True, quarters=quarter_proofs))
    assert len(proofs) == 2
    report = dict(completed=True, PASS=True, prepared_design_review_only=True, code=bind(__file__), root_policy=bind(POLICY),
        contract_exception='Only the two fixed registered mesh anchor arrays and fixed source geometry/math metadata are used; no per-image GT input.',
        fixed_geometry_result=bind(GEOMETRY), fixed_source_math=bind(MATH), CF_edge_families=FAMILIES, fixed_mesh_certificates=proofs,
        cuboid_extension='V129 scaled_data asserts scaled native_anchor3D == SIGNS*dimensions/2. Positive diagonal scaling and orthogonal signed-quarter rotation preserve the parallel families.',
        line_math_requirements=[
            'Use original CF native channels; rear pairing is not independently imageX-sorted.',
            'Four disjoint edges in each of three parallel families; three incident edges at one vertex are trivially concurrent and are not a useful test.',
            'Homogeneous vanishing points at infinity are valid; no division by vanishing-point homogeneous Z is required.',
            'Coincident/zero-length/rank1 or poorly conditioned families must be undefined/flagged, not assigned perfect-zero risk.',
            'A native held-out edge residual can implicate the query or its partner; it does not identify the uniquely wrong endpoint.',
            'Concurrence is preserved by common image projective transformations, common translation and some valid cuboid permutations; low residual does not certify absolute keypoint identity or <=3px error.'
        ],
        G42_code=bind(G42_CODE), original_G42_input_seal=bind(G42_SEAL), head_class_code=bind(HEAD_CODE),
        expanded_geometry_control_readiness=dict(
            status='Assessable as a prospective matched source experiment; not implemented, trained, admitted or chosen here.',
            existing_old_native='Reuse sealed V133 actual6144 G42 and its identity/q/L lineage; do not fabricate native geometry from old AUX boxes.',
            new_native='A new GT-free20480 G42 packet/seal from actual original q8/native k/q8 minmax context/HW is still required before such a fit; use exact V74 operator and preserve noquery/OOB/degenerate flags.',
            old_AUX51004='Geometry unknown: retain exact all-zero42 input, separate from known native geometry. Native onehot8 sum1 distinguishes known native rows from unknown zero placeholders.',
            initial_and_compute_controls='Same C16 update4500 initialization, zero geometry weights, C29 old192+oldnative64+newnative64 draws/loss means, fixed8000 optimizer/update budget and all64 gates/rank49; update0 old/native/new raw parity including missing/OOB remains mandatory.',
            G42_semantics='Same unchanged query8, k, minmax box and HW; scale is max predicted q8 span, not L. No GT dimensions/pose/reference/order/error/factor feature and no fitted normalization.',
            source_indicator_limit='Known native versus unknown AUX geometry can act as a source/context-availability indicator. Synthetic native G42 conditioning alone cannot prove transfer to real native geometry.'),
        gradient_analysis=dict(
            formula='h=Wv*visual+b+Wg*g; z=wo*ReLU(h)+bo. dL/dWg=delta_h*g^T.',
            AUX_g_zero='For g=0, geometry contribution and its data gradient are exactly zero. This does not remove AUX gradients from shared visual layers/output weights.',
            compatibility_claim='Public AUX and registered-native visibility contracts differ in supervision/identity strength. Shared-gradient tradeoff is possible; saved confusion matrices do not prove incompatible gradients or label contradiction.',
            evidence='V133 actual G42 C25 had D747/828,E45/60 versus matched G0 C24 D758/828,E45/60; both ineligible. V137 C29 improved native coverage while some public LOCO/TOMIE gates failed. Neither observation certifies geometry conditioning as a solution.'),
        freeze_interpretations=dict(
            freeze_geometry_at_zero='No learned geometry effect; with otherwise exact C29 protocol this reduces to the G0 path.',
            freeze_shared_visual_train_geometry='Can preserve AUX g0 raw outputs algebraically while changing native predictions. It changes trainable capacity/optimization hypothesis; it is a separate controlled experiment, not an explanation already established by saved metrics.',
            acceptance='Freezing a branch need not change acceptance. All existing64 gates, threshold0, full denominators and source-only selection can and should remain; freeze alone never justifies waiving public AUX or native gates.',
            no_private_candidate='Failed C29 is not admitted as a private candidate. No failed model is promoted, calibrated or used to relax gates here.'),
        diagnostic_interpretation_guards=[
            'Confidence associations with coordinate error must be reported within teacher D and within teacher E; E-associated confidence loss is not itself bad-position proof.',
            'inputUnavailable partly defines known failure including predictionU; flag the definitional association and retain available-only supplements.',
            'Mask G42 zero placeholders by constructed/defined status for scalar ranks; zero placeholders are not measured geometry.',
            'All26624 original rows, noquery/OOB, teacher D/E/S/U and undefined scores remain accounted for; diagnostic scores do not become filters, labels or new U rules.'
        ],
        per_image_GT_RGB_features_checkpoint_model_fit_private_V112_PnP_reads_or_calls=0,
        fixed_registered_geometry_anchor_reads=2, new_input_feature_or_G42_packets_built=0, policy_or_gate_or_source_label_changes=False)
    write(H / 'FIXED_CONTRACT_AND_MODEL_CONTROL_READINESS.json', report)
    print(json.dumps(dict(completed=True, PASS=True, report=bind(H / 'FIXED_CONTRACT_AND_MODEL_CONTROL_READINESS.json'), fixed_meshes=2)), flush=True)


if __name__ == '__main__':
    main()
