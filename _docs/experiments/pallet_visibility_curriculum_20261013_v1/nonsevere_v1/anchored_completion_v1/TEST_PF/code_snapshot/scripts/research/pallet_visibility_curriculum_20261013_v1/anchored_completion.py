"""Rigid completion from image anchors and conservative visible measurements.

The observations are immutable. A rigid pose generally cannot pass through
every noisy anchor exactly: protection constrains each anchor's model projection
and robust residual relative to one frozen accepted reference instead. Hidden,
truncated and unowned corners are never filled before fitting. Completion is a
separate projection of the final supplied cuboid, including the measured slots.

This research solver is opt-in and does not modify previous solvers or defaults.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from . import boundary_rules as B
from . import curriculum as M


@dataclass(frozen=True)
class AnchoredCompletionConfig:
    coordinate_source: str = "FUSED"
    anchors: bool = True
    supplement: bool = True
    supplemental_pool: str = "interior_only"
    anchor_protection: bool = True
    final_guard_policy: str = "anchor_priority"
    use_supplemental_observations: bool = True
    allow_detector_supplements: bool = True
    image_pixel_sha256: object = None
    supplemental_visibility: float = .65
    supplemental_accuracy: float = .25
    anchor_max_projection_shift_px: float = 2.
    anchor_max_huber_increase_px2: float = 1.
    image_hw: object = None
    pose: dict = field(default_factory=dict)


def _huber(errors, radius):
    errors = np.asarray(errors, float)
    return np.where(errors <= radius, errors**2, 2*radius*errors-radius**2)


def _anchor_guard(xyz, q, K, R, t, ids, reference, R0, t0, cfg, pose_cfg):
    """Unweighted per-point loss; no compensation between different anchors."""
    ids = np.asarray(ids, int)
    if not cfg.anchor_protection or not len(ids):
        return dict(usable=True, reason="disabled" if not cfg.anchor_protection else "no_boundary_anchors",
                    reference_source=None, records=[])
    baseline_R, baseline_t = (R0, t0) if reference is None else (reference["R"], reference["t"])
    baseline = M.project(xyz, baseline_R, baseline_t, K)[ids]
    candidate = M.project(xyz, R, t, K)[ids]
    baseline_error = np.linalg.norm(baseline-q[ids], axis=1)
    error = np.linalg.norm(candidate-q[ids], axis=1)
    motion = np.linalg.norm(candidate-baseline, axis=1)
    increase = _huber(error, pose_cfg.robust_px)-_huber(baseline_error, pose_cfg.robust_px)
    # Before a corrected reference exists, protect observed anchor residuals,
    # while allowing the initial pose to move toward those actual observations.
    shift_ok = np.ones(len(ids), bool) if reference is None else motion <= cfg.anchor_max_projection_shift_px+1e-8
    loss_ok = increase <= cfg.anchor_max_huber_increase_px2+1e-8
    records = [dict(corner=int(k), reference_error_px=float(a), candidate_error_px=float(b),
                    projection_shift_px=float(d), huber_increase_px2=float(e),
                    projection_shift_pass=bool(s), per_point_loss_pass=bool(l))
               for k, a, b, d, e, s, l in zip(ids, baseline_error, error, motion, increase, shift_ok, loss_ok)]
    usable = bool(np.all(shift_ok & loss_ok))
    return dict(usable=usable, reason="usable" if usable else "protected_anchor_budget_exceeded",
                reference_source="initial_residual_only_before_bootstrap" if reference is None else reference["source"],
                shift_budget_applied=reference is not None, records=records)


def _uncertainty(xyz, K, R, t, initial_t, ids, weights, estimated_dof, cfg):
    """Local information only; neither semantic confidence nor calibrated CI."""
    ids = np.asarray(ids, int)
    scale = max(np.linalg.norm(initial_t), np.linalg.norm(np.ptp(xyz, axis=0)), 1e-6)
    projection_J = M._projection_jacobian(np.zeros(6), xyz @ R.T, K, t, scale).reshape(8, 2, 6)
    J = projection_J[ids].reshape(-1, 6).copy()
    J *= np.repeat(np.sqrt(weights[ids]), 2)[:, None]
    _, singular, vt = np.linalg.svd(J, full_matrices=True)
    rank = int(np.sum(singular > (singular[0]*1e-9 if len(singular) else 0.)))
    result = dict(full_rigid_data_rank=rank, full_rigid_nullspace=vt[rank:].tolist(),
        data_singular_values=singular.tolist(), full_rigid_parameter_count=6,
        estimated_parameter_count=int(estimated_dof), estimated_subspace=None,
        local_covariance=None, assumed_pixel_noise_std=1.,
        parameterization="local_rotation_radians_then_translation_over_fixed_initial_depth_scale",
        depth_scale=float(scale), calibrated_confidence=False, global_pose_uniqueness_proved=False,
        excludes_corner_semantic_bias=True, shared_line_correlation_modelled=False,
        robust_influence_modelled=False, covariance_model="frozen_weight_L2_information_surrogate",
        prior_curvature_model="diagonal_local_approximation_to_fixed_initial_prior",
        projected_covariance_px2=None, conditions_on_exact_initial_rotation=estimated_dof == 3,
        uncertainty_sources_not_modelled=["initial_rotation_error_in_translation_only_branch",
            "model_dimension_error", "camera_calibration_error", "native_correspondence_error",
            "corner_semantic_bias", "shared_image_line_correlation"])
    if not estimated_dof:
        return result
    selected = np.arange(3, 6) if estimated_dof == 3 else np.arange(6)
    data = J[:, selected].T @ J[:, selected]
    prior_diag = cfg.prior_strength_px**2 * np.r_[
        np.full(3, 1/np.deg2rad(cfg.rotation_prior_deg)**2),
        np.full(3, 1/cfg.translation_prior_fraction**2)][selected]
    prior = np.diag(prior_diag)
    covariance = np.linalg.inv(data+prior)
    propagated = np.einsum("pik,kl,pjl->pij", projection_J[:, :, selected], covariance, projection_J[:, :, selected])
    eigenvalues, eigenvectors = np.linalg.eigh(data+prior)
    fractions = np.array([float(v @ prior @ v / e) for e, v in zip(eigenvalues, eigenvectors.T)])
    result.update(estimated_subspace="translation_with_exact_fixed_rotation" if estimated_dof == 3 else "full_rigid_pose",
        local_covariance=covariance.tolist(), information_eigenvalues=eigenvalues.tolist(),
        projected_covariance_px2=propagated.tolist(),
        directional_prior_information_fraction=fractions.tolist(),
        translation_local_std_metres=(np.sqrt(np.diag(covariance)[-3:])*scale).tolist())
    if estimated_dof == 6:
        result["rotation_local_std_degrees"] = np.rad2deg(np.sqrt(np.diag(covariance)[:3])).tolist()
    return result


def solve_anchored_completion(observation_result, xyz, K, initial_R, initial_t, config=None):
    """Seed, guarded measured-point progression and separate rigid completion.

    ``interior_only`` means original projected non-outline and original RGB
    visible support. A local RGB boundary can be a hole or crease within the
    object, so patch and geometric roles remain separate. ``all_visible`` allows weak
    outline measurements, labelled DETECTOR_VISIBLE rather than image anchors.
    Accuracy scores are heuristic admission evidence, not verified precision.

    Three active measurements estimate translation with exact initial rotation.
    Four estimate full pose only with the existing seven finite ambiguity probes;
    larger sets retain the fixed-initial local prior, rank and trust safeguards.
    Final immutable-pool validation selects the latest passing checkpoint.
    ``anchor_priority`` retains an independently accepted boundary seed when
    weak supplemental measurements veto all expanded poses; it reports that
    full-pool exception explicitly. ``strict_eligible_pool`` can instead veto
    the seed and restore initial. A bootstrap without an independent boundary
    seed receives no exception under either policy.
    """
    cfg = config if isinstance(config, AnchoredCompletionConfig) else AnchoredCompletionConfig(**deepcopy(config or {}))
    source = cfg.coordinate_source.upper()
    if (source not in ("RAW", "FUSED") or cfg.supplemental_pool not in ("interior_only", "all_visible")
            or cfg.final_guard_policy not in ("anchor_priority", "strict_eligible_pool")):
        raise ValueError("Expected RAW/FUSED and interior_only/all_visible")
    for flag in (cfg.anchors, cfg.supplement, cfg.anchor_protection,
                 cfg.use_supplemental_observations, cfg.allow_detector_supplements):
        if not isinstance(flag, bool):
            raise ValueError("Solver switches must be booleans")
    values = np.array([cfg.supplemental_visibility, cfg.supplemental_accuracy,
                       cfg.anchor_max_projection_shift_px, cfg.anchor_max_huber_increase_px2], float)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or np.any(values[:2] > 1)):
        raise ValueError("Finite nonnegative budgets and scores in [0,1] required")
    pose_cfg = M.CurriculumConfig(**deepcopy(cfg.pose))
    # Reuse the existing complete model/canvas/line/fusion bindings. Its pose is
    # only applied when anchors=True; the visibility ablation starts at initial.
    canvas = observation_result.get("image_hw") if cfg.image_hw is None else cfg.image_hw
    seed = B.solve_boundary_rules(observation_result, xyz, K, initial_R, initial_t,
        dict(coordinate_source=source, image_hw=canvas, pose=cfg.pose))
    xyz, K, R0, t0 = map(lambda a: np.asarray(a, float), (xyz, K, initial_R, initial_t))
    raw = np.asarray(observation_result["raw_points"], float)
    source_q = np.asarray(observation_result["raw_points" if source == "RAW" else "fused_points"], float)
    q = source_q.copy()
    evidence = seed["evidence"]
    app = observation_result["visibility_appearance"]
    visibility = M._array_field(app, "visibility_score", 0.)
    accuracy = np.asarray(app["local_accuracy_score"], float)
    if accuracy.shape != (8,) or not np.isfinite(accuracy).all() or np.any((accuracy < 0) | (accuracy > 1)):
        raise ValueError("Finite local_accuracy_score[8] in [0,1] required")
    RGB_states = np.asarray(app["states"])
    weights = np.maximum(visibility, .05)*np.maximum(accuracy, .05)
    weights[RGB_states == "VISIBLE_INTERIOR"] *= pose_cfg.interior_weight
    anchors = list(evidence["strong_ids"]) if cfg.anchors else []
    verified_RGB = {}
    if cfg.use_supplemental_observations and observation_result.get("interior_observations") is not None:
        from . import common as C
        from .interior_observation import validate_interior_observations
        dimensions = np.ptp(xyz, axis=0)
        if not np.array_equal(xyz, C.cuboid(*dimensions)):
            raise ValueError("RGB junction validator requires the unchanged centered native cuboid")
        initial = dict(available=True, R_cf=R0.tolist(), centroid=t0.tolist(), cf_extents=dimensions.tolist())
        actual_canvas = canvas if canvas is not None else observation_result["boundary_visibility"].get("configuration", {}).get("actual_image_hw")
        pixel_binding = cfg.image_pixel_sha256 if cfg.image_pixel_sha256 is not None else observation_result.get("image_pixel_sha256")
        verified_RGB = validate_interior_observations(observation_result["interior_observations"],
            raw, K, initial, observation_result, image_hw=actual_canvas, image_pixel_sha256=pixel_binding)
        verified_RGB = {int(k): deepcopy(v) for k, v in verified_RGB.items()}
    supplements, roles, exclusion = [], {}, {}
    for row in evidence["records"]:
        k = row["corner"]
        if k in anchors:
            roles[k] = "IMAGE_BOUNDARY_ANCHOR"
            continue
        if row["self_occluded_conditional"]:
            reason = "SELF_OCCLUDED"
        elif row["truncated_conditional"]:
            reason = "TRUNCATED"
        elif row["raw_prediction_missing"] or row["raw_prediction_out_of_frame"]:
            reason = "MISSING_OR_OUTSIDE"
        elif row["occluder_evidence"]["independent_region_supported"]:
            reason = "OTHER_OBJECT_EVIDENCE"
        elif not cfg.supplement:
            reason = "SUPPLEMENT_DISABLED"
        elif RGB_states[k] not in ("VISIBLE_BOUNDARY", "VISIBLE_INTERIOR") or visibility[k] < cfg.supplemental_visibility:
            reason = "INSUFFICIENT_VISIBLE_EVIDENCE"
        elif k not in verified_RGB and not cfg.allow_detector_supplements:
            reason = "DETECTOR_SUPPLEMENTS_DISABLED"
        elif k not in verified_RGB and accuracy[k] < cfg.supplemental_accuracy:
            reason = "INSUFFICIENT_COORDINATE_EVIDENCE"
        elif cfg.supplemental_pool == "interior_only" and row["expected_outline"]:
            reason = "NOT_TRUE_MODEL_INTERIOR"
        else:
            supplements.append(k)
            if k in verified_RGB:
                q[k] = verified_RGB[k]["measured_xy"]
                roles[k] = "RGB_FACE_JUNCTION_INTERIOR"
            else:
                roles[k] = "DETECTOR_VISIBLE_MODEL_INTERIOR" if not row["expected_outline"] else "DETECTOR_VISIBLE_WEAK_BOUNDARY"
            continue
        exclusion[k] = reason
        roles[k] = "OUTPUT_PROJECTION_ONLY"
    supplements.sort(key=lambda k: (-float(visibility[k]*accuracy[k]), k))
    pool = sorted(anchors+supplements)
    R, t = R0.copy(), t0.copy()
    used = list(anchors)
    reference = None
    checkpoints = []
    stages = []
    seed_applied = bool(cfg.anchors and seed["accepted"])
    if seed_applied:
        R, t = np.asarray(seed["R"]), np.asarray(seed["t"])
        reference = dict(R=R.copy(), t=t.copy(), source="boundary_rule_seed")
        checkpoints.append(dict(R=R.copy(), t=t.copy(), ids=used.copy(), dof=seed["estimated_dof"], stage=0))
    stages.append(dict(name="boundary_seed", accepted=seed_applied, used_ids=anchors if seed_applied else [],
        estimated_dof=seed["estimated_dof"] if seed_applied else 0,
        reason="accepted_boundary_seed" if seed_applied else "seed_not_applied" if not cfg.anchors else seed["fallback_reason"],
        boundary_rule_branch=seed["branch"], boundary_seed_rotation_exact_only_if_3DOF=True))
    pending, rejected = [], []
    for new_id in supplements:
        pending.append(new_id)
        attempted = used+pending
        dof = 3 if len(attempted) == 3 else 6
        geometry = (B._translation_geometry(xyz, q[:8], K, R, t, attempted, pose_cfg)
                    if len(attempted) == 3 else M._geometry(xyz, q[:8], attempted, R, t, K, pose_cfg))
        stage = dict(name="add_visible_"+str(new_id), attempted_ids=attempted.copy(),
            pending_ids=pending.copy(), used_ids=[], accepted=False, estimated_dof=0, geometry=geometry,
            new_measurement_role=roles[new_id], original_prior_reference="supplied_initial_pose")
        if not geometry["usable"]:
            stage["reason"] = "deferred_"+geometry["reason"]
            stages.append(stage)
            continue
        before = M._aggregate_objective(xyz, q[:8], K, R, t, R0, t0, attempted, weights, pose_cfg)
        starts = [("current", R)]
        if len(attempted) == 4:
            for axis in range(3):
                for sign in (1, -1):
                    delta = np.zeros(3)
                    delta[axis] = sign*np.deg2rad(pose_cfg.rotation_prior_deg)
                    starts.append(("initial_axis"+str(axis)+("+" if sign > 0 else "-"), cv2.Rodrigues(delta)[0] @ R0))
        candidates, probes = [], []
        for index, (name, start_R) in enumerate(starts):
            probe = dict(name=name, probe_index=index, accepted=False)
            try:
                if dof == 3:
                    candidate_R, candidate_t, optimizer = B._fit_translation(xyz, q[:8], K, R0, t0, attempted, weights[attempted], pose_cfg)
                else:
                    candidate_R, candidate_t, optimizer = M._fit(xyz, q[:8], K, R0, t0, start_R, t, attempted, weights[attempted], pose_cfg)
                gate = B._pose_gate(xyz, q[:8], K, candidate_R, candidate_t, R0, t0, attempted, pose_cfg, dof == 3)
                protection = _anchor_guard(xyz, q[:8], K, candidate_R, candidate_t, anchors, reference, R0, t0, cfg, pose_cfg)
                after = M._aggregate_objective(xyz, q[:8], K, candidate_R, candidate_t, R0, t0, attempted, weights, pose_cfg)
                gain = before["objective"]-after["objective"]
                valid = gate["usable"] and protection["usable"] and gain > pose_cfg.minimum_gain_px2
                reason = (gate["reason"] if not gate["usable"] else protection["reason"] if not protection["usable"]
                          else "usable" if valid else "no_active_aggregate_gain")
                probe.update(reason=reason, gate=gate, anchor_guard=protection, optimizer=optimizer,
                    candidate_R=candidate_R.tolist(), candidate_t=candidate_t.tolist(),
                    aggregate_before=before, aggregate_after=after, objective_gain=gain)
                if valid:
                    candidates.append(dict(probe_index=index, R=candidate_R, t=candidate_t, objective=after))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, cv2.error) as error:
                probe.update(reason="optimizer_failure", error=str(error))
            probes.append(probe)
        conflicts = B._ambiguity_records(candidates, xyz, t0, pose_cfg) if len(attempted) == 4 else []
        stage.update(probes=probes, ambiguity_probe_conflicts=conflicts)
        if conflicts or not candidates:
            stage["reason"] = "four_point_near_equal_distinct_pose_candidates" if conflicts else "no_guarded_candidate"
            rejected.extend(pending)
        else:
            best = min(candidates, key=lambda c: (c["objective"]["objective"], c["probe_index"]))
            R, t = best["R"].copy(), best["t"].copy()
            used = attempted.copy()
            probes[best["probe_index"]]["accepted"] = True
            stage.update(accepted=True, used_ids=used.copy(), estimated_dof=dof, reason="accepted_guarded_visible_addition")
            if reference is None:
                reference = dict(R=R.copy(), t=t.copy(), source="first_accepted_measured_bootstrap")
            checkpoints.append(dict(R=R.copy(), t=t.copy(), ids=used.copy(), dof=dof, stage=len(stages)))
        pending = []
        stages.append(stage)
    initial_objective = M._aggregate_objective(xyz, q[:8], K, R0, t0, R0, t0, pool, weights, pose_cfg)
    final_checks, chosen = [], None
    for checkpoint in checkpoints:
        objective = M._aggregate_objective(xyz, q[:8], K, checkpoint["R"], checkpoint["t"], R0, t0, pool, weights, pose_cfg)
        passing = initial_objective["objective"]-objective["objective"] > pose_cfg.minimum_gain_px2
        final_checks.append(dict(stage=checkpoint["stage"], used_ids=checkpoint["ids"],
            passes_immutable_pool_initial_guard=passing, aggregate=objective))
        if passing:
            chosen = checkpoint
    seed_exception = False
    if chosen is None and cfg.final_guard_policy == "anchor_priority" and seed_applied:
        chosen = checkpoints[0]
        seed_exception = True
    accepted = chosen is not None
    if accepted:
        R, t, used, dof = chosen["R"], chosen["t"], chosen["ids"], chosen["dof"]
        retained_stage = chosen["stage"]
        fallback = "weak_pool_veto_trusted_boundary_seed_retained" if seed_exception else None
    else:
        R, t, used, dof = R0.copy(), t0.copy(), [], 0
        retained_stage = None
        fallback = "final_immutable_pool_guard_rejected_all_checkpoints" if checkpoints else "insufficient_or_rejected_independent_support"
    projected = M.project(xyz, R, t, K)
    completed = q.copy()
    completed[:8] = projected
    records = deepcopy(evidence["records"])
    for row in records:
        k = row["corner"]
        row.update(measurement_role=roles[k], included_in_immutable_pool=k in pool,
            used_for_final_pose=k in used, exclusion_reason=exclusion.get(k),
            output_corner_source="rigid_model_projection", requested_3D_corner_certified=False)
    protection = _anchor_guard(xyz, q[:8], K, R, t, anchors, reference, R0, t0, cfg, pose_cfg) if accepted else None
    return dict(schema="anchored_rigid_completion_v1", available=True, accepted=accepted,
        R=R.tolist(), t=t.tolist(), R_cf=R.tolist(), centroid=t.tolist(), used_ids=used,
        estimated_dof=dof, final_rotation_exactly_initial=bool(np.array_equal(R, R0)),
        reason="trusted_boundary_seed_retained" if seed_exception else "guarded_rigid_refinement" if accepted else "initial_pose_preserved",
        fallback_reason=fallback,
        boundary_seed=seed, boundary_seed_applied=seed_applied, anchor_ids=anchors,
        anchor_measurements=M._finite_list(q[:8][anchors]), anchor_weights=weights[anchors].tolist(),
        anchor_reference=None if reference is None else dict(R=reference["R"].tolist(), t=reference["t"].tolist(), source=reference["source"]),
        final_anchor_guard=protection, supplemental_pool=cfg.supplemental_pool,
        eligible_supplemental_ids=supplements, supplemental_order=supplements,
        final_supplemental_ids=[k for k in used if k not in anchors], rejected_addition_ids=rejected,
        deferred_ids=pending, immutable_pool_ids=pool, omitted_pool_ids=[k for k in pool if k not in used],
        frozen_pool_weights=weights[pool].tolist(), states=[r["state"] for r in records], keypoint_status=records,
        self_hidden=deepcopy(observation_result["self_hidden"]), stages=stages,
        final_guard=dict(initial_aggregate=initial_objective, checkpoints=final_checks,
            selected_stage=retained_stage, selection="latest_passing_checkpoint_or_independent_boundary_seed" if cfg.final_guard_policy == "anchor_priority" else "latest_passing_checkpoint",
            policy=cfg.final_guard_policy, independent_boundary_seed_exception=seed_exception,
            selected_checkpoint_passes_full_pool_guard=accepted and not seed_exception,
            independent_boundary_seed_objective_gain=seed["aggregate_before"]["objective"]-seed["aggregate_after"]["objective"] if seed_applied else None,
            boundary_seed_can_be_vetoed=cfg.final_guard_policy == "strict_eligible_pool", batch_refit_performed=False,
            discarded_checkpoint_count=len(checkpoints)-int(accepted)),
        uncertainty=_uncertainty(xyz, K, R, t, t0, used, weights, dof, pose_cfg),
        raw_points=M._finite_list(raw), measured_points=M._finite_list(q),
        source_measured_points=M._finite_list(source_q),
        RGB_junction_measurement_ids=[k for k in supplements if k in verified_RGB],
        observed_measurements_by_id={str(k): dict(native_corner=k, xy=q[k].tolist(),
            measurement_role=roles[k], used_for_final_pose=k in used,
            local_precision_calibrated=False) for k in pool},
        projected_points=projected.tolist(), completed_points=M._finite_list(completed),
        coordinate_source=source, configuration=asdict(cfg),
        provenance=dict(GT_inputs=False, source_observation_coordinates_modified=False,
            RGB_junction_new_measurements_from_verified_current_pixels=True,
            original_model_native_order_and_center_preserved=True, original_self_mask_preserved=True,
            projected_points_used_as_measurements=False, excluded_points_used_for_fit=False,
            completed_points_used_for_fit=False, hard_output_clamp=False,
            visibility_or_accuracy_updated_from_pose=False, initial_prior_frozen=True,
            optimizer_enforces_anchor_constraints=False, anchor_constraints_checked_before_admission=True,
            independent_means_image_observed_not_statistically_independent=True,
            anchor_correctness_certified=False, default_methods_changed=False),
        limits=["Initial cuboid self-occlusion, truncation and native correspondence remain conditional",
            "Local RGB visibility and coordinate accuracy are heuristic admission evidence, not physical-corner proof",
            "Strong SAM outline intersections can belong to rounded/virtual/different surfaces",
            "Protection is a fixed-reference 2D tolerance, not an exact noisy-pixel clamp",
            "Three-point translation bootstrap keeps R fixed; later full-rank additions may change R",
            "Anchor-priority can retain an independently accepted boundary seed despite the weak full-pool veto; this is an explicitly logged exception",
            "Strict eligible-pool validation can reject that seed and return initial",
            "Local covariance is a frozen-weight L2 information surrogate; it omits Huber influence, semantic bias and shared-line correlation",
            "Finite four-point probes do not prove global pose uniqueness"])
