"""Finite pose proposals and a separate mixed 2D measurement/model readout.

The V1 measured coordinates, eligible pool, weights, native model and reference
are immutable. External refiners supply poses, never replacement measurements.
The selector keeps V1 retained support/DOF and requires both current-pose and
initial-pose improvement under the same original full-pool objective.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

from . import anchored_completion as A
from . import boundary_rules as B
from . import curriculum as M

MINIMUM_PROPOSAL_GAIN_PX2 = 1e-7


@dataclass(frozen=True)
class ProtectedProposalConfig:
    enabled: bool = True


def _pose(R, t):
    R, t = np.asarray(R, float), np.asarray(t, float)
    if (R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all()
            or not np.isfinite(t).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-6)):
        raise ValueError("Finite proper rotation and translation required")
    return R.copy(), t.copy()


def _ids(value, name):
    ids = list(value)
    if (len(set(ids)) != len(ids) or any(not isinstance(k, (int, np.integer))
                                      or isinstance(k, bool) or not 0 <= k < 8 for k in ids)):
        raise ValueError(name+" must contain distinct native corner IDs")
    return [int(k) for k in ids]


def _excluded(record):
    return bool(record.get("self_occluded_conditional") or record.get("truncated_conditional")
        or record.get("raw_prediction_missing") or record.get("raw_prediction_out_of_frame")
        or record.get("occluder_evidence", {}).get("independent_region_supported")
        or record.get("state") in ("SELF_OCCLUDED", "TRUNCATED", "UNKNOWN", "UNKNOWN_OR_OCCLUSION",
                                  "OTHER_OBJECT_OCCLUSION", "OTHER_OBJECT_SUSPECTED"))


def _context(base, xyz, K, initial_R, initial_t):
    if base.get("schema") != "anchored_rigid_completion_v1":
        raise ValueError("Unmodified V1 anchored solution required; mixed readouts are not solver inputs")
    xyz, K = np.asarray(xyz, float), np.asarray(K, float)
    if (xyz.shape != (8, 3) or K.shape != (3, 3) or not np.isfinite(xyz).all()
            or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0):
        raise ValueError("Original ordered model and positive-focal camera required")
    R0, t0 = _pose(initial_R, initial_t)
    stored_initial = base["boundary_seed"]["stages"][0]
    if (not np.array_equal(R0, np.asarray(stored_initial["R"], float))
            or not np.array_equal(t0, np.asarray(stored_initial["t"], float))):
        raise ValueError("Supplied initial pose differs from the frozen V1 initial prior")
    R, t = _pose(base["R"], base["t"])
    if np.any((xyz @ R0.T+t0)[:, 2] <= 1e-6) or np.any((xyz @ R.T+t)[:, 2] <= 1e-6):
        raise ValueError("Initial and returned models must be in front of camera")
    q = np.asarray(base["measured_points"], float)
    raw = np.asarray(base["raw_points"], float)
    if q.shape != raw.shape or q.shape not in ((8, 2), (9, 2)) or not np.array_equal(q[8:], raw[8:], equal_nan=True):
        raise ValueError("V1 native slots and centre must remain unchanged")
    pool, used, anchors = (_ids(base[key], key) for key in ("immutable_pool_ids", "used_ids", "anchor_ids"))
    if not set(used) <= set(pool) or not set(anchors) <= set(pool) or not np.isfinite(q[pool]).all():
        raise ValueError("Retained support and anchors must belong to the finite immutable pool")
    records = base["keypoint_status"]
    if len(records) != 8 or [r["corner"] for r in records] != list(range(8)):
        raise ValueError("Exactly eight native point-status records required")
    if any(_excluded(records[k]) for k in pool):
        raise ValueError("Excluded states cannot belong to proposal observation pool")
    role_anchors = [r["corner"] for r in records if r.get("measurement_role") == "IMAGE_BOUNDARY_ANCHOR"]
    if sorted(role_anchors) != sorted(anchors):
        raise ValueError("Anchor IDs must bind authoritative V1 measurement roles")
    weights = np.zeros(8)
    values = np.asarray(base["frozen_pool_weights"], float)
    if values.shape != (len(pool),) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Positive frozen V1 pool weights required")
    weights[pool] = values
    cfg = A.AnchoredCompletionConfig(**deepcopy(base["configuration"]))
    pose_cfg = M.CurriculumConfig(**deepcopy(cfg.pose))
    if cfg.anchor_max_projection_shift_px != 2. or cfg.anchor_max_huber_increase_px2 != 1.:
        raise ValueError("The declared unchanged two-pixel/one-pixel-squared anchor budgets are required")
    for k in pool:
        record = records[k]
        expected = max(record["p_visible"], .05)*max(record["coordinate_accuracy_score"], .05)
        if record["RGB_state_at_raw_query"] == "VISIBLE_INTERIOR":
            expected *= pose_cfg.interior_weight
        if not np.isclose(weights[k], expected, atol=1e-12, rtol=0):
            raise ValueError("V1 frozen weights disagree with original point evidence")
    if (not np.allclose(np.asarray(base["projected_points"]), M.project(xyz, R, t, K), atol=1e-10, rtol=0)
            or not np.array_equal(np.asarray(base["anchor_measurements"], float).reshape(-1, 2), q[anchors])):
        raise ValueError("V1 returned projection or immutable anchor coordinate binding differs")
    dof = int(base["estimated_dof"])
    if base["accepted"]:
        if (dof == 3 and len(used) != 3) or (dof == 6 and len(used) < 4) or dof not in (3, 6):
            raise ValueError("Retained V1 support must bind its estimated DOF")
        if dof == 3 and not np.array_equal(R, R0):
            raise ValueError("V1 translation-only result must retain exact initial rotation")
    elif dof != 0 or used or not np.array_equal(R, R0) or not np.array_equal(t, t0):
        raise ValueError("V1 initial fallback must remain exact with no retained support")
    reference = base["anchor_reference"]
    if base["accepted"] and anchors and reference is None:
        raise ValueError("Accepted anchored V1 result requires its frozen anchor reference")
    if reference is not None:
        ref_R, ref_t = _pose(reference["R"], reference["t"])
        reference = dict(R=ref_R, t=ref_t, source=reference["source"])
    if base["provenance"].get("projected_points_used_as_measurements") is not False:
        raise ValueError("Pose-created observations are forbidden")
    return xyz, K, q, R0, t0, R, t, pool, used, anchors, weights, cfg, pose_cfg, reference, dof


def select_protected_pose(base_solution, xyz, K, initial_R, initial_t, proposals, config=None):
    """Choose a protected finite pose; do not refit or alter V1 support/DOF.

    Each proposal requires name/R/t/cf_extents, ORIGINAL_NATIVE_CF coordinates
    and native_order_preserved=True. Physical-basis conversion belongs to the
    caller. A retained three-DOF result uses proposal translation only, with
    exact initial R before all evaluation. Every newly selected proposal must
    improve both the current returned and supplied initial full-pool objective.
    """
    config = config if isinstance(config, ProtectedProposalConfig) else ProtectedProposalConfig(**deepcopy(config or {}))
    if not isinstance(config.enabled, bool):
        raise ValueError("enabled must be a boolean")
    (xyz, K, q, R0, t0, current_R, current_t, pool, used, anchors,
     weights, cfg, pose_cfg, reference, dof) = _context(base_solution, xyz, K, initial_R, initial_t)
    guard_cfg = replace(cfg, anchor_protection=True)
    current = M._aggregate_objective(xyz, q[:8], K, current_R, current_t, R0, t0, pool, weights, pose_cfg)
    initial = M._aggregate_objective(xyz, q[:8], K, R0, t0, R0, t0, pool, weights, pose_cfg)
    output = deepcopy(base_solution)
    audit, candidates, names = [], [], set()
    for index, proposal in enumerate(proposals):
        record = dict(index=index, name=None, admissible=False, selected=False,
                      extra_coordinates_used_as_measurements=False)
        try:
            if not isinstance(proposal, Mapping):
                raise ValueError("Pose proposals must be mappings")
            name = proposal["name"]
            record["name"] = name
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("Distinct named finite proposals required")
            names.add(name)
            if (proposal.get("coordinate_frame") != "ORIGINAL_NATIVE_CF"
                    or proposal.get("native_order_preserved") is not True):
                raise ValueError("Proposal must bind unchanged original native CF")
            dimensions = np.asarray(proposal["cf_extents"], float)
            if dimensions.shape != (3,) or not np.allclose(dimensions, np.ptp(xyz, axis=0), atol=1e-12, rtol=0):
                raise ValueError("Proposal changed the original registered model dimensions")
            proposed_R, proposed_t = _pose(proposal["R"], proposal["t"])
            candidate_R = R0.copy() if dof == 3 else proposed_R
            record.update(proposed_R=proposed_R.tolist(), evaluated_R=candidate_R.tolist(),
                evaluated_t=proposed_t.tolist(), proposal_rotation_discarded=dof == 3,
                retained_used_ids=used, retained_estimated_dof=dof,
                coordinate_frame="ORIGINAL_NATIVE_CF", physical_basis_conversion_verified_by_caller=True)
            if not config.enabled:
                record["reason"] = "selector_disabled"
            elif not base_solution["accepted"] or len(used) < 3:
                record["reason"] = "insufficient_retained_independent_support"
            else:
                gate = B._pose_gate(xyz, q[:8], K, candidate_R, proposed_t, R0, t0, used, pose_cfg, dof == 3)
                protection = A._anchor_guard(xyz, q[:8], K, candidate_R, proposed_t, anchors, reference,
                                             R0, t0, guard_cfg, pose_cfg)
                objective = M._aggregate_objective(xyz, q[:8], K, candidate_R, proposed_t, R0, t0, pool, weights, pose_cfg)
                current_gain = current["objective"]-objective["objective"]
                initial_gain = initial["objective"]-objective["objective"]
                finite = bool(np.isfinite([objective["objective"], current_gain, initial_gain]).all())
                valid = (finite and gate["usable"] and protection["usable"]
                    and current_gain > MINIMUM_PROPOSAL_GAIN_PX2 and initial_gain > MINIMUM_PROPOSAL_GAIN_PX2)
                reason = ("nonfinite_objective" if not finite else gate["reason"] if not gate["usable"]
                    else protection["reason"] if not protection["usable"]
                    else "no_current_pool_objective_gain" if current_gain <= MINIMUM_PROPOSAL_GAIN_PX2
                    else "no_initial_pool_objective_gain" if initial_gain <= MINIMUM_PROPOSAL_GAIN_PX2 else "admissible_protected_pose")
                record.update(reason=reason, admissible=valid, gate=gate, anchor_guard=protection,
                    objective=objective, current_objective_gain=current_gain, initial_objective_gain=initial_gain)
                if valid:
                    candidates.append(dict(index=index, R=candidate_R, t=proposed_t, objective=objective))
        except (KeyError, TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError, cv2.error) as error:
            record.update(reason="invalid_pose_proposal", error=str(error))
        audit.append(record)
    chosen = min(candidates, key=lambda c: (c["objective"]["objective"], c["index"])) if candidates else None
    if chosen is not None:
        R, t = chosen["R"], chosen["t"]
        audit[chosen["index"]]["selected"] = True
        output.update(R=R.tolist(), t=t.tolist(), R_cf=R.tolist(), centroid=t.tolist(),
            reason="protected_pose_proposal_selected", fallback_reason=None,
            final_rotation_exactly_initial=bool(np.array_equal(R, R0)),
            uncertainty=A._uncertainty(xyz, K, R, t, t0, used, weights, dof, pose_cfg))
        output["base_final_guard"] = deepcopy(base_solution["final_guard"])
        output["final_guard"] = dict(policy="dual_current_and_initial_proposal_guard",
            initial_aggregate=initial, current_aggregate=current, selected_aggregate=chosen["objective"],
            selected_checkpoint_passes_full_pool_guard=True, independent_boundary_seed_exception=False,
            batch_refit_performed=False, retained_used_ids=used, retained_estimated_dof=dof)
    else:
        R, t = current_R, current_t
    projected = M.project(xyz, R, t, K)
    completed = q.copy()
    completed[:8] = projected
    output.update(schema="protected_pose_proposals_v1", base_schema=base_solution["schema"],
        selected_pose_source=audit[chosen["index"]]["name"] if chosen else "V1_BASE",
        proposal_selected=chosen is not None, proposal_audit=audit,
        proposal_configuration=asdict(config), projected_points=projected.tolist(),
        rigid_model_corners=projected.tolist(), completed_points=M._finite_list(completed),
        proposal_selection=dict(current_objective=current, initial_objective=initial,
            full_pool_ids=pool, retained_used_ids=used, retained_estimated_dof=dof,
            selection="minimum_objective_among_valid_finite_proposals",
            all_rejected_preserves_V1_pose_and_seed_exception=True,
            frozen_anchor_reference_preserved=True, mandatory_anchor_protection=True,
            new_proposals_require_both_current_and_initial_gain=True,
            independent_seed_exception_inherited_by_new_proposal=False))
    selected_guard = A._anchor_guard(xyz, q[:8], K, R, t, anchors, reference,
                                     R0, t0, guard_cfg, pose_cfg) if output["accepted"] else None
    output["proposal_selection"].update(minimum_objective_gain_px2=MINIMUM_PROPOSAL_GAIN_PX2,
                                         selected_pose_anchor_guard=selected_guard)
    if chosen is not None:
        output["final_anchor_guard"] = selected_guard
    return output


def state_keypoint_readout(solution):
    """Expose measured image coordinates and model-only completion separately.

    Outline-anchor measurements survive even a low-support pose fallback.
    Other image coordinates survive only for actually retained RGB/detector
    observations. The mixed readout is never a new all-point PnP input.
    """
    q = np.asarray(solution["measured_points"], float)
    raw = solution["raw_points"]
    raw_array = np.asarray(raw, float)
    projected = np.asarray(solution.get("rigid_model_corners", solution["projected_points"]), float)
    if (q.shape not in ((8, 2), (9, 2)) or raw_array.shape != q.shape
            or not np.array_equal(q[8:], raw_array[8:], equal_nan=True)
            or projected.shape != (8, 2) or not np.isfinite(projected).all()):
        raise ValueError("Native measurement carrier and finite rigid eight-corner projection required")
    anchors, used = _ids(solution["anchor_ids"], "anchor_ids"), _ids(solution["used_ids"], "used_ids")
    records = solution["keypoint_status"]
    if len(records) != 8 or [r["corner"] for r in records] != list(range(8)):
        raise ValueError("Eight ordered native status records required")
    role_anchors = [r["corner"] for r in records if r.get("measurement_role") == "IMAGE_BOUNDARY_ANCHOR"]
    if sorted(role_anchors) != sorted(anchors) or any(_excluded(records[k]) for k in anchors):
        raise ValueError("Image anchors must bind non-excluded authoritative V1 roles")
    reported = projected.tolist()+deepcopy(raw[8:])
    sources = ["RIGID_MODEL_COMPLETION"]*8+(["ORIGINAL_CENTER_METADATA"] if len(q) == 9 else [])
    preserved = []
    for k, record in enumerate(records):
        if _excluded(record):
            continue
        role = record.get("measurement_role")
        if k in anchors and role == "IMAGE_BOUNDARY_ANCHOR":
            source = "IMAGE_BOUNDARY_ANCHOR_MEASUREMENT"
        elif k in used and role == "RGB_FACE_JUNCTION_INTERIOR":
            source = "USED_RGB_JUNCTION_MEASUREMENT"
        elif k in used and role in ("DETECTOR_VISIBLE_MODEL_INTERIOR", "DETECTOR_VISIBLE_WEAK_BOUNDARY"):
            source = "USED_DETECTOR_VISIBLE_MEASUREMENT"
        else:
            continue
        if not np.isfinite(q[k]).all():
            raise ValueError("Preserved image measurements must be finite")
        reported[k] = q[k].tolist()
        sources[k] = source
        preserved.append(k)
    return dict(schema="state_keypoint_mixed_readout_v1", reported_points=reported,
        rigid_model_corners=projected.tolist(), point_sources=sources,
        preserved_measurement_ids=preserved, mixed_points_are_one_rigid_pose=False,
        provenance=dict(projected_points_are_output_only=True, reported_points_used_for_pose=False,
            mixed_readout_must_not_be_fed_to_PnP=True, native_order_and_center_preserved=True,
            original_prediction_carriers_are_not_all_admitted_observations=True,
            image_measurement_semantics_remain_conditional=True, GT_inputs=False))
