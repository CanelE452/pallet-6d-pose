"""Freeze the uniform V30 method before CAL46 selection and new noise fitting."""
from datetime import datetime, timezone
from pathlib import Path

from . import common as C
from . import geometry_guided_edge_problem_v4_30 as G
from . import geometry_guided_edge_authentication_v4_30 as AUTH
from . import observe_geometry_guided_edges_v4_29b as OBS

B = OBS.OUTPUT.parent
OUTPUT = B / 'observed_bi_associated_edges_plan_v4_30'
PLAN = OUTPUT / 'PREDECLARATION.json'
ARMS = ('SAME_FROZEN_E', 'ASSOCIATED_EDGE_ALL_BI_NATIVE',
    'ASSOCIATED_EDGE_ALL_BI_CONTAINMENT', 'ASSOCIATED_EDGE_SOURCE_NATIVE',
    'ASSOCIATED_EDGE_SOURCE_CONTAINMENT')


def declare():
    if OUTPUT.exists():
        raise FileExistsError('Preserve every V30 central declaration')
    parent = C.read(OBS.OUTPUT / 'PROTOCOL.json')
    modules = (C, G, G.PARENT, G.ASSOCIATION, G.ASSOCIATION.P,
        G.ASSOCIATION.G, G.ASSOCIATION.B, G.ASSOCIATION.M,
        AUTH, AUTH.PARENT, OBS)
    sources = {str(Path(m.__file__).resolve().relative_to(C.ROOT)): C.sha(m.__file__)
        for m in modules}
    sources[str(Path(__file__).resolve().relative_to(C.ROOT))] = C.sha(__file__)
    test = Path(__file__).with_name('test_geometry_guided_population_v4_30.py')
    sources[str(test.relative_to(C.ROOT))] = C.sha(test)
    input_paths = [OBS.OUTPUT / 'PROTOCOL.json', OBS.OUTPUT / 'PREPARATION_SEAL.json',
        OBS.OUTPUT / 'CAL_OBSERVATION_SEAL.json', OBS.OUTPUT / 'REST_OBSERVATION_SEAL.json',
        B / 'geometry_guided_edges_plan_v4_29/PREDECLARATION.json',
        B / 'observed_bi_line_association_plan_v4_30_v2/PREDECLARATION.json',
        B / 'observed_bi_line_association_CAL_cases_v4_30/RERANK_SEAL.json',
        B / 'observed_bi_line_association_CAL_process_v4_30_v2/RENDER_SEAL.json']
    inputs = {str(p.relative_to(C.ROOT)): C.sha(p) for p in input_paths}
    for relative, digest in parent['protected_user_sha256'].items():
        if C.sha(C.ROOT / relative) != digest:
            raise ValueError('Protected original edit changed: ' + relative)
    plan = dict(schema='UNIFORM_OBSERVED_BI_ASSOCIATED_EDGE_METHOD_PREDECLARATION_V30',
        UTC=datetime.now(timezone.utc).isoformat(), arms=list(ARMS),
        partitions=dict(CAL=46, TEST=49, REMAINDER=143, PRIMARY=192, severe=0),
        raw_observation_generation=G.RAW_OBSERVATION_GENERATION,
        observation_family=G.OBSERVATION_FAMILY, population=G.POPULATION,
        selection_profile=G.SELECTION_PROFILE, association_recipe=G.ASSOCIATION.RECIPE,
        source_policies=list(G.SOURCE_POLICIES),
        change='Admit each actual TLS candidate only if every eligible original named B/I endpoint is within the original configured normal search reach of its infinite line; then run the same ordered graph.',
        configured_normal_search_radius_px=parent['configuration']['line_configuration']['normal_search_radius_px'],
        geometric_distance='absolute n dot q + c; not Euclidean endpoint or along-line distance',
        original_raw_chains_indices_guides_masks_q_covariance_states_U_regions_unchanged=True,
        S_image_or_2D_or_region_queries=0, U_precise_point_or_line_queries=0,
        S_only_frozen_bounded_model_prior_and_PnP_projection=True,
        U_approximate_region_only=True, independent_occluder_mask_not_available_is_unknown=True,
        guide_physical_transport_unchanged_from_V29=True,
        objective='Same V29 native physical-edge and archived selected-line tangent costs + frozen bounded O/S; optional one-sided foreground containment at unchanged weight.',
        numerical_factor_gradient_and_hardguard_math_unchanged=True,
        new_noise='Recompute all46 CAL selected populations before references, then fit separate line normal/sine components and archived tangent gamma for each source policy; inventory46/fixed compatible44 + unchanged4LOGO retained.',
        old_V29_noise_or_model_relabelled=False,
        likelihood='Original uncentered physical-line mixture likelihood and scalar archived-tangent likelihood; same engineering floors once and same optimizer starts/recipe.',
        provenance='V29 raw generation remains V29; new family/population/selection_profile follow each prepared row, fitted model, export, authenticated bundle and runtime plan.',
        source_policy_or_reference_dependent_selection=False,
        runtime_GT_ID_error_or_partition_routing=False,
        new_numeric_e_pose_prior_or_fallback=False, full_pose_TP_topnormal_prior_weights=0.,
        start_policy='Same native-ranked six diverse starts plus own coarse, both WD families, all cached feasible contenders; containment also keeps same-policy native endpoints.',
        no_new_RGB_NN_NMS_TLS_candidates_or_pose_guide_fits=True,
        selected_support_metadata_reconstruction_from_cache_allowed=True,
        zero_candidate_zero_selected_or_failed_rows_preserved=True,
        full_evaluation='All238 x5 arms sealed before fixed-reference score; independent selected-pose audit and actual-image review; CAL separate; all192 primary kept.',
        baseline_and_default='SAME_FROZEN_E until all original49 and additional143 each <=5cm and other active-goal requirements independently verified.',
        reused_data='Exploratory reuse of existing CAL and test material, not a new held-out benchmark. Three CAL diagnostics motivated this repair; no references participate in candidate selection.',
        source_sha256=sources, input_sha256=inputs,
        protected_user_sha256=parent['protected_user_sha256'],
        downstream_operation_source_and_input_closures_required=True,
        full238_raw_parent_final_seal_required_before_uniform_replay=True,
        labels_primary_scores_or_new_actual_reranks_in_declare=0,
        new_CAL46_rerank_and_noise_fit_require_root_execution_GO=True,
        current_V29_sources_models_and_observations_changed=False)
    if float(plan['configured_normal_search_radius_px']) != 43.:
        raise ValueError('Existing unchanged normal reach must be43px')
    OUTPUT.mkdir(parents=True)
    C.write(PLAN, plan)
    for relative, digest in sources.items():
        path = OUTPUT / 'code_snapshot' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((C.ROOT / relative).read_bytes())
        if C.sha(path) != digest:
            raise ValueError('Declaration source snapshot mismatch')
    C.write(OUTPUT / 'PLAN_SEAL.json', dict(plan_sha256=C.sha(PLAN),
        source_sha256=sources, input_sha256=inputs))
    print(dict(plan_sha256=C.sha(PLAN), arms=ARMS, actual_new_CAL46_or_labels=0))


if __name__ == '__main__':
    declare()
