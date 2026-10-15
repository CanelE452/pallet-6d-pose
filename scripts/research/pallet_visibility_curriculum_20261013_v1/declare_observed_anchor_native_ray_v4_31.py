"""Declare one uniform q-anchored signed-native-ray population before CAL46.

Only cached actual V29 query records are admitted. A model query is not moved
to an observed q and passed off as a new measurement. This declaration does
not run a selector, read labels or change the frozen V30 experiment.
"""
from datetime import datetime, timezone
from pathlib import Path

from . import common as C
from . import geometry_guided_edge_problem_v4_31 as G
from . import geometry_guided_edge_authentication_v4_31 as AUTH
from . import geometry_guided_edge_noise_calibration_v4_31 as NOISE
from . import observe_geometry_guided_edges_v4_29b as OBS
from . import observed_anchor_native_ray_association_v4_31 as ADMISSION
from . import prepare_geometry_guided_native_evidence_v4_31 as METADATA

B = OBS.OUTPUT.parent
OUTPUT = B / 'observed_anchor_native_ray_method_plan_v4_31'
PLAN = OUTPUT / 'PREDECLARATION.json'
ARMS = ('SAME_FROZEN_E', 'ANCHOR_RAY_EDGE_ALL_BI_NATIVE',
    'ANCHOR_RAY_EDGE_ALL_BI_CONTAINMENT', 'ANCHOR_RAY_EDGE_SOURCE_NATIVE',
    'ANCHOR_RAY_EDGE_SOURCE_CONTAINMENT')
ADMISSION_SHA256 = 'ccdb311a4468825919a6731620e6836a0b52bffc775cc594d90c6bea8d2babad'
CAL3_SEAL_SHA256 = '8949845af1b85f6a6e7e0e5a59478574e6c69d40869b13ddef73fa7efb036468'
METADATA_SMOKE_SEAL_SHA256 = '0987d6c2ead6b591f029f954eea72007dec21ea921d6b21396ef1ac9146b4c73'


def declare():
    if OUTPUT.exists():
        raise FileExistsError('Preserve every q-anchor method declaration')
    if C.sha(ADMISSION.__file__) != ADMISSION_SHA256:
        raise ValueError('Exact source-reviewed candidate policy required')
    parent = C.read(OBS.OUTPUT / 'PROTOCOL.json')
    rawseal = C.read(OBS.OUTPUT / 'OBSERVATION_SEAL.json')
    if (rawseal['rows'] != 238 or rawseal['ordered_ids'] != parent['ids']
            or len(set(parent['ids'])) != 238):
        raise ValueError('Complete unchanged raw238 inventory required')
    if (G.OBSERVATION_FAMILY != NOISE.OBSERVATION_FAMILY
            or G.POPULATION != NOISE.POPULATION
            or G.SELECTION_PROFILE != ADMISSION.SELECTION_PROFILE
            or NOISE.SELECTION_PROFILE != ADMISSION.SELECTION_PROFILE):
        raise ValueError('One separately typed selection and fitted-noise population required')
    cal3 = B / 'observed_anchor_native_ray_CAL_cases_v4_31/RERANK_SEAL.json'
    if C.sha(cal3) != CAL3_SEAL_SHA256:
        raise ValueError('Exact prior fixed-three-CAL cached diagnosis required')
    smoke_seal = B / 'geometry_guided_native_metadata_smoke_v4_31/SMOKE_SEAL.json'
    if C.sha(smoke_seal) != METADATA_SMOKE_SEAL_SHA256:
        raise ValueError('Exact prior fixed-three-CAL metadata API smoke required')
    modules = (C, G, G.PARENT, G.V30, G.BASELINE, G.EDGE, G.N, G.J,
        AUTH, AUTH.PARENT, AUTH.RELATION, AUTH.OBS,
        NOISE, NOISE.PARENT, NOISE.LINE, NOISE.TANGENT,
        ADMISSION, ADMISSION.F, ADMISSION.P, ADMISSION.G, ADMISSION.B,
        ADMISSION.M, OBS, METADATA, METADATA.PARENT, METADATA.PREFIX)
    sources = {str(Path(m.__file__).resolve().relative_to(C.ROOT)): C.sha(m.__file__)
        for m in modules}
    sources[str(Path(__file__).resolve().relative_to(C.ROOT))] = C.sha(__file__)
    test_names = ('test_observed_anchor_native_ray_association_v4_31.py',
        'test_geometry_guided_edge_problem_v4_31.py',
        'test_geometry_guided_edge_authentication_v4_31.py',
        'test_geometry_guided_edge_noise_calibration_v4_31.py',
        'test_prepare_geometry_guided_native_evidence_v4_31.py',
        'test_smoke_geometry_guided_native_metadata_v4_31.py')
    for name in test_names:
        path = Path(__file__).with_name(name)
        if not path.is_file():
            raise FileNotFoundError('Required method contract tests missing: ' + name)
        sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    smoke_source = Path(__file__).with_name('smoke_geometry_guided_native_metadata_v4_31.py')
    sources[str(smoke_source.relative_to(C.ROOT))] = C.sha(smoke_source)
    paths = [OBS.OUTPUT / n for n in ('PROTOCOL.json', 'PREPARATION_SEAL.json',
        'CAL_OBSERVATION_SEAL.json', 'REST_OBSERVATION_SEAL.json', 'OBSERVATION_SEAL.json')]
    paths += [cal3,
        B / 'observed_anchor_native_ray_CAL_plan_v4_31/PREDECLARATION.json',
        B / 'observed_anchor_native_ray_association_peer_review_v4_31/SOURCE_REVIEW_V1.json',
        B / 'geometry_guided_edges_peer_review_v4_31/PROBLEM_AUTH_SOURCE_REVIEW_V1.json',
        B / 'geometry_guided_edge_noise_v4_31_checks/SOURCE_AND_SYNTHETIC_RECEIPT_V1.json',
        B / 'geometry_guided_native_metadata_peer_review_v4_31/SOURCE_REVIEW_V1.json',
        B / 'geometry_guided_native_metadata_smoke_plan_v4_31/PREDECLARATION.json',
        smoke_seal,
        B / 'frozen_actual_support_guard_source_audit_v4_30_v1/AUDIT_SEAL.json']
    inputs = {str(p.relative_to(C.ROOT)): C.sha(p) for p in paths}
    for mapping in (sources, inputs, parent['protected_user_sha256']):
        for relative, digest in mapping.items():
            if C.sha(C.ROOT / relative) != digest:
                raise ValueError('Source/input/protected binding changed: ' + relative)
    plan = dict(schema='UNIFORM_OBSERVED_Q_ANCHOR_SIGNED_NATIVE_RAY_METHOD_PLAN_V31',
        UTC=datetime.now(timezone.utc).isoformat(), arms=list(ARMS),
        partitions=dict(CAL=46, TEST=49, REMAINDER=143, PRIMARY=192, severe=0),
        observation_family=G.OBSERVATION_FAMILY, population=G.POPULATION,
        selection_profile=G.SELECTION_PROFILE,
        raw_observation_generation=G.RAW_OBSERVATION_GENERATION,
        admission_recipe=ADMISSION.RECIPE,
        change='Keep V30 normal43 association; admit only an actually recorded original BI-q query anchor and a query direction in the immutable native reference hemisphere.',
        raw_chains_pixels_TLS_q_covariances_states_masks_guides_and_indices_unchanged=True,
        model_anchor_queries_relocated_or_relabelled_as_q_queries=False,
        signed_reference='both measured BI endpoints if length>=existing10px; otherwise O/T/U mate uses ORIGINAL initial model native ray; missing BI/short/orthogonal abstains',
        S_image_or_2D_or_region_queries=0, U_precise_point_or_line_queries=0,
        S_only_frozen_bounded_model_prior_and_PnP_projection=True,
        U_only_approximate_regions=True,
        finite_physical_edge_identity_certified_by_admission=False,
        target_mask_absence_is_occlusion=False,
        independent_occluder_mask_unavailable_is_unknown=True,
        mask_usage='Original actual NMS boundary/internal context and prequery gates retained; expected-segment visibility classes remain diagnostics, not a newly calibrated pose factor.',
        original_physical_guide_transport_retained=True,
        source_policies=list(G.SOURCE_POLICIES),
        numerical_factor_gradient_hard_support_guard_and_O_S_C_math='UNCHANGED_V29_V30',
        objective='Same native point/actual edge/archived selected-line tangent + frozen bounded O/S; optional unchanged one-sided C1.',
        new_noise='Separate CAL46 selected population before labels; full+four LOGO fits for both source policies; unchanged likelihood, starts, floors and weighting. Preserve fixed inventory46/compatible44.',
        noise_recipe=NOISE.RECIPE, tangent_recipe=NOISE.TANGENT_RECIPE,
        prior_V29_or_V30_noise_model_transferred_or_relabelled=False,
        runtime_GT_ID_error_or_partition_routing=False,
        numeric_e_pose_prior_start_candidate_or_fallback_added=False,
        full_pose_TP_topnormal_prior_weights=0.,
        starts='Same native-ranked six diverse + own coarse + both WD + all cached feasible contenders; C also retains same-policy native endpoints.',
        full_evaluation='All238 x5 sealed, then independent selected-pose audit and fixed-reference score; all192 primary and fallback/zero-support retained; actual fixed-case image review.',
        default='SAME_FROZEN_E until all active-goal conditions independently hold.',
        motivation='Postscore V29 actual support/guide audits and three prior CAL cached reranks; reused exploratory data, not a fresh held-out benchmark.',
        current_V30_sources_models_or_results_changed=False,
        labels_primary_metrics_new_actual_CAL46_selection_or_model_fit_in_declare=0,
        source_sha256=sources, input_sha256=inputs,
        protected_user_sha256=parent['protected_user_sha256'],
        downstream_operation_source_input_and_state_closures_required=True)
    OUTPUT.mkdir(parents=True)
    C.write(PLAN, plan)
    for relative, digest in sources.items():
        target = OUTPUT / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((C.ROOT / relative).read_bytes())
        if C.sha(target) != digest:
            raise ValueError('Method source snapshot differs')
    C.write(OUTPUT / 'PLAN_SEAL.json', dict(plan_sha256=C.sha(PLAN),
        source_sha256=sources, input_sha256=inputs, before_new_CAL46_selection_or_fit=True))
    print(dict(plan_sha256=C.sha(PLAN), arms=ARMS, new_CAL46_or_labels=0), flush=True)


if __name__ == '__main__':
    declare()
