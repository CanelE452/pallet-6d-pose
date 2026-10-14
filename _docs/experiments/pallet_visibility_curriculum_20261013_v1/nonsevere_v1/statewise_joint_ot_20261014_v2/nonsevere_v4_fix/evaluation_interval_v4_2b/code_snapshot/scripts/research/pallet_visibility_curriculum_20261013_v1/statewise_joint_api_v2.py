"""Current-RGB S/B/I/O/T runtime, independent of historical observation APIs.

S is fixed from the registered coarse cuboid before mask prompts or RGB
queries. Actual DIRECT/EXTERNAL inference yields B/I/O; corrected B/I image
measurements are frozen before every O/T domain enters joint SE(3) fitting.
This opt-in API does not read labels, write experiments, or change models.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from . import common as C
from .joint_ot_pnp import project_native, solve_joint_ot_pnp
from .ot_domains_v2 import (_array_digest, build_ot_domains, image_pixel_sha256,
                            validate_domain_bindings)
from .state_visibility_v2 import (ROOT, StateVisibilityModel, geometry_state_gates,
                                 infer_state_visibility, load_state_model, sha256)
from .visible_bi_correction_v2 import correct_visible_bi


SCHEMA = "STATEWISE_JOINT_RGB_V2"
_STATES = {"S", "B", "I", "O", "T", "U"}


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _array(value: Any, shape: tuple, name: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return array


def _rotation(value: Any, name: str) -> np.ndarray:
    R = _array(value, (3, 3), name)
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0) or not np.isclose(np.linalg.det(R), 1, atol=1e-7):
        raise ValueError(f"{name} must be a proper rotation")
    return R


def registered_geometry(image_hw, K, native_xyz, coarse_pose):
    """Resolve geometric S and silhouette B/I without raw detector queries.

    Registered ordering and dimensions are checked against the original native
    cuboid. Physical-axis registration Q is carried unchanged to the final pose.
    Convex-hull outward planes provide the fixed coarse self-occlusion gate;
    projected silhouette membership determines only the initial B/I role.
    """
    K = _array(K, (3, 3), "K")
    xyz = _array(native_xyz, (8, 3), "native_xyz")
    extents = _array(coarse_pose["cf_extents"], (3,), "coarse cf_extents")
    if np.any(extents <= 0) or not np.allclose(xyz, C.cuboid(*extents), atol=1e-10, rtol=0):
        raise ValueError("native_xyz must preserve the registered original cuboid basis, dimensions and order")
    R = _rotation(coarse_pose.get("R_cf", coarse_pose.get("R")), "coarse R_cf")
    t = _array(coarse_pose.get("centroid", coarse_pose.get("t")), (3,), "coarse centroid")
    physical = _rotation(coarse_pose.get("R_physical", R), "coarse R_physical")
    Q = R.T @ physical
    projected, depth = project_native(xyz, R, t, K)
    if np.any(depth <= 0) or not np.isfinite(projected).all():
        raise ValueError("Coarse registered cuboid must project with positive camera depth")
    camera_in_object = -R.T @ t
    planes = ConvexHull(xyz).equations
    scale = max(float(np.linalg.norm(extents)), 1e-9)
    hidden = []
    for point in xyz:
        incident = np.abs(planes[:, :3] @ point + planes[:, 3]) <= 1e-8 * scale
        ray = camera_in_object - point
        cosines = planes[incident, :3] @ ray / np.linalg.norm(ray)
        hidden.append(bool(np.max(cosines) < -np.sin(np.deg2rad(2))))
    hull = cv2.convexHull(projected.astype(np.float32)).reshape((-1, 2))
    contour = hull.astype(np.float32).reshape((-1, 1, 2))
    roles = ["B" if abs(cv2.pointPolygonTest(contour, tuple(map(float, point)), True)) < 1e-3
             else "I" for point in projected]
    return dict(native_xyz=xyz, K=K, R_cf=R, centroid=t, R_physical=physical,
                physical_registration_Q=Q, cf_extents=extents,
                initial_projection_xy=projected, initial_self_hidden=np.array(hidden, dtype=bool),
                initial_roles=roles, image_hw=list(image_hw),
                S_source="fixed_registered_coarse_cuboid_outward_face_visibility",
                S_is_conditional_on_coarse_pose=True,
                B_I_role_is_geometry_not_actual_visibility=True,
                raw_points_used_for_geometry=False)


def _mask(array: Any, image_hw: tuple, name: str) -> np.ndarray:
    mask = np.asarray(array, dtype=float)
    if mask.shape != image_hw or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError(f"{name} must be finite canvas probabilities in [0,1]")
    return mask.copy()


def _prompt(raw: np.ndarray, hidden: np.ndarray, image_hw: tuple) -> tuple:
    height, width = image_hw
    ids = [i for i in range(8) if not hidden[i] and np.isfinite(raw[i]).all()
           and not np.array_equal(raw[i], [-1., -1.])
           and 0 <= raw[i, 0] < width and 0 <= raw[i, 1] < height]
    points = raw[ids].copy()
    box = np.r_[points.min(axis=0), points.max(axis=0)].tolist() if ids else None
    return ids, points, box


class StatewiseJointV2:
    """Opt-in runtime with a production visibility model and optional masks.

    ``mask_provider(image_bgr, *, prompt_points, prompt_native_ids,
    prompt_box_xyxy, initial_projection_xy, K)`` returns a probability mask or
    ``{'state_pallet_mask': tiny_mask, 'pallet_mask': sam_mask, 'provenance': ...}``.
    State Tiny features and B boundary masks remain separate when provided.
    ``state_producer`` injection has infer_state_visibility's signature and is
    intended for contract tests; production defaults to the new classifier.
    """

    def __init__(self, state_model=None, ot_model=None, mask_provider: Callable | None = None,
                 config: Mapping | None = None, state_producer: Callable | None = None):
        self.state_model = state_model
        self.ot_model = ot_model
        self.mask_provider = mask_provider
        self.state_producer = state_producer or infer_state_visibility
        self.injected_state_producer = state_producer is not None
        self.config = deepcopy(dict(config or {}))
        unknown = set(self.config) - {"bi", "solver", "prediction_source", "record_id", "device"}
        if unknown:
            raise ValueError("Unknown statewise runtime config fields: " + ", ".join(sorted(unknown)))

    def _masks(self, image, raw, geometry, eligible_ids, pallet_mask, state_pallet_mask):
        ids, points, box = _prompt(raw, geometry["initial_self_hidden"], image.shape[:2])
        audit = dict(prompt_native_ids=ids, prompt_points=points.tolist(), prompt_box_xyxy=box,
                     S_excluded_before_prompt=True, ninth_center_used=False,
                     initial_projection_used_as_prompt=False, provider_invoked=False)
        model = self.state_model
        if len(eligible_ids) and not self.injected_state_producer and not isinstance(model, StateVisibilityModel):
            if model is None:
                raise ValueError("Production actual visibility requires a new state_model descriptor")
            model = load_state_model(str(model), self.config.get("device", "cpu"))
        if pallet_mask is not None:
            correction_mask = _mask(pallet_mask, image.shape[:2], "pallet_mask")
            state_mask = _mask(state_pallet_mask if state_pallet_mask is not None else pallet_mask,
                               image.shape[:2], "state_pallet_mask")
            audit.update(source="explicit_current_canvas_masks", shared_mask=state_pallet_mask is None)
        elif self.mask_provider is not None and len(eligible_ids):
            bundle = self.mask_provider(image, prompt_points=points.copy(), prompt_native_ids=ids.copy(),
                prompt_box_xyxy=deepcopy(box), initial_projection_xy=geometry["initial_projection_xy"].copy(),
                K=geometry["K"].copy())
            if isinstance(bundle, Mapping):
                correction_mask = _mask(bundle["pallet_mask"], image.shape[:2], "provider pallet_mask")
                state_mask = _mask(bundle.get("state_pallet_mask", bundle["pallet_mask"]),
                                   image.shape[:2], "provider state_pallet_mask")
                audit["provider_provenance"] = deepcopy(bundle.get("provenance", {}))
                audit["shared_mask"] = "state_pallet_mask" not in bundle
            else:
                correction_mask = state_mask = _mask(bundle, image.shape[:2], "provider mask")
                audit["shared_mask"] = True
            audit.update(source="current_RGB_mask_provider", provider_invoked=True)
        elif len(eligible_ids):
            if self.injected_state_producer:
                raise ValueError("An injected state producer with RGB queries needs an explicit mask or provider")
            binding = model.descriptor.get("pallet_segmenter")
            if not binding:
                raise ValueError("This state descriptor has no bound segmenter; supply a current-RGB mask_provider or explicit pallet_mask")
            weights = ROOT / binding["weights_repo_relative"]
            if sha256(weights) != binding["weights_sha256"]:
                raise ValueError("Frozen new-state Tiny segmenter binding changed")
            from .segmentation import segment
            segment_device = getattr(model.frozen_backbone, "device", self.config.get("device", "cpu"))
            state_mask = _mask(segment(image, weights, segment_device), image.shape[:2], "Tiny mask")
            correction_mask = state_mask.copy()
            audit.update(source="new_state_descriptor_bound_global_Tiny", shared_mask=True,
                         global_image_forward=True, native_point_prompt_count=0)
        else:
            # S/T/U-only gates need neither a visibility forward nor local BI
            # correction. Empty masks satisfy the pure corrector input contract.
            correction_mask = state_mask = np.zeros(image.shape[:2], dtype=float)
            audit.update(source="no_gated_RGB_queries_no_mask_forward", shared_mask=True)
        return correction_mask, state_mask, audit, model

    def refine(self, image_bgr, raw_points, K, native_xyz, coarse_pose, *, pallet_mask=None,
               state_pallet_mask=None, independent_truncation=None, crop_transform=None):
        image = np.asarray(image_bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
            raise ValueError("image_bgr must be actual nonempty uint8 BGR[H,W,3]")
        raw = np.array(raw_points, dtype=float, copy=True)
        if raw.shape not in ((8, 2), (9, 2)):
            raise ValueError("raw_points must preserve native eight corners, optionally ninth center metadata")
        original_raw = raw.copy()
        original_pose = deepcopy(coarse_pose)
        geometry = registered_geometry(image.shape[:2], K, native_xyz, coarse_pose)
        if crop_transform is not None:
            if (crop_transform.get("schema") != "REAL_RGB_TRANSLATION_CROP_V2"
                    or tuple(crop_transform["canvas_hw"]) != image.shape[:2]
                    or crop_transform["crop_image_sha256"] != image_pixel_sha256(image)
                    or crop_transform["crop_K_sha256"] != _array_digest(geometry["K"])):
                raise ValueError("Real crop must bind the actual current RGB canvas and K before state inference")
        hidden = geometry["initial_self_hidden"]
        # Excluded S carriers cannot accidentally leak into a new provider's
        # scale, bbox, local feature extraction or correction implementation.
        gated_raw = raw.copy()
        gated_raw[:8][hidden] = [-1., -1.]
        gate_records, eligible_ids = geometry_state_gates(image, gated_raw, hidden,
            geometry["initial_roles"], geometry["initial_projection_xy"],
            independent_truncation=independent_truncation)
        correction_mask, state_mask, mask_audit, model = self._masks(
            image, gated_raw, geometry, eligible_ids, pallet_mask, state_pallet_mask)
        observation = self.state_producer(image, gated_raw.copy(), hidden.copy(),
            list(geometry["initial_roles"]), geometry["initial_projection_xy"].copy(), model,
            pallet_mask=state_mask, independent_truncation=deepcopy(independent_truncation))
        states = list(observation["states"])
        if len(states) != 8 or any(state not in _STATES for state in states):
            raise ValueError("State producer must return eight native S/B/I/O/T/U states")
        records = observation.get("records", [])
        if len(records) != 8 or any(int(record["native_id"]) != i or record["state"] != states[i]
                                    for i, record in enumerate(records)):
            raise ValueError("State records must preserve original native ID and state ordering")
        queried = observation.get("rgb_queried_ids", [])
        if any(i not in eligible_ids for i in queried) or any(records[i].get("rgb_evaluated") for i in np.flatnonzero(hidden)):
            raise ValueError("State producer queried an excluded S/T/U geometric gate")
        if any(states[i] != "S" for i in np.flatnonzero(hidden)) or any(states[i] == "S" for i in np.flatnonzero(~hidden)):
            raise ValueError("Initial registered geometric S must remain fixed before RGB inference")
        if any(states[i] in {"B", "I"} and states[i] != geometry["initial_roles"][i] for i in range(8)):
            raise ValueError("Actual visible B/I must retain its initial geometric boundary/interior role")
        normalized_pose = dict(original_pose, R_cf=geometry["R_cf"].tolist(),
                               centroid=geometry["centroid"].tolist(), cf_extents=geometry["cf_extents"].tolist())
        correction = correct_visible_bi(image, gated_raw.copy(), geometry["K"], normalized_pose,
            states, pallet_mask=correction_mask, config=self.config.get("bi"), native_xyz=geometry["native_xyz"])
        measurements = deepcopy(correction["visible_measurements"])
        # New solver copies these observations once more; keeping this explicit
        # frozen snapshot makes all downstream readouts auditable.
        frozen_measurements = deepcopy(measurements)
        domain_states = ["V" if state in {"B", "I"} else state for state in states]
        # A coarse conditional T fixes neither a physical side nor a finite
        # location. Only independently supplied metadata narrows the side union.
        truncation_sides = {}
        for record in records:
            if record["state"] == "T" and record.get("physical_truncation_certified"):
                sides = record.get("outside_canvas_side", [])
                if isinstance(sides, str):
                    sides = [sides]
                if len(sides) == 1:
                    truncation_sides[record["native_id"]] = sides[0]
        record_id = self.config.get("record_id", "current")
        prediction_source = self.config.get("prediction_source", "raw_backbone")
        domain_bundle = build_ot_domains(image, gated_raw, domain_states, geometry["K"], geometry["native_xyz"],
            self.ot_model, prediction_source=prediction_source, record_id=record_id,
            self_hidden=hidden, truncation_sides=truncation_sides, crop_transform=crop_transform)
        validate_domain_bindings(domain_bundle, image, gated_raw, domain_states, geometry["K"], geometry["native_xyz"],
            prediction_source=prediction_source, record_id=record_id, self_hidden=hidden)
        joint = solve_joint_ot_pnp(geometry["native_xyz"], geometry["K"], states, measurements,
            domain_bundle["domains"], image.shape[:2],
            initializations=[{"name": "REGISTERED_COARSE_INITIALIZATION_ONLY", "R": geometry["R_cf"],
                              "t": geometry["centroid"]}], config=self.config.get("solver"))
        if measurements != frozen_measurements:
            raise RuntimeError("Joint solver changed frozen corrected B/I measurements")
        accepted = bool(joint["accepted"])
        BI_seed = joint.get("corrected_BI_seed")
        BI_seed_budgets_pass = bool(BI_seed and BI_seed.get("visible_measurement_budgets_pass"))
        BI_seed_support_pass = bool(BI_seed and BI_seed.get("visible_support_guards_pass"))
        BI_seed_pose_family = bool(BI_seed and BI_seed.get("contains_underconstrained_pose_family"))
        material_ambiguity = bool((joint.get("ambiguity") or {}).get("detected", False))
        OT_hypothesis_available = accepted and bool(domain_bundle["domains"])
        if accepted:
            R, t = np.array(joint["R"]), np.array(joint["t"])
            pose_source = "all_joint_OT_feasible_corrected_BI_SE3"
        elif BI_seed_support_pass and not BI_seed_pose_family:
            R, t = np.array(BI_seed["R"]), np.array(BI_seed["t"])
            pose_source = "explicit_corrected_BI_only_seed_fallback_S_output_only"
        else:
            R, t = geometry["R_cf"], geometry["centroid"]
            pose_source = "explicit_registered_coarse_pose_fallback_S_output_only"
        projected, _ = project_native(geometry["native_xyz"], R, t, geometry["K"])
        reported, sources = [None] * 8, ["UNRESOLVED"] * 8
        for measurement in frozen_measurements:
            native_id = measurement["native_corner"]
            reported[native_id] = deepcopy(measurement["xy"])
            sources[native_id] = "FROZEN_CORRECTED_CURRENT_RGB_BI_MEASUREMENT"
        for i, state in enumerate(states):
            if state == "S":
                reported[i] = projected[i].tolist()
                sources[i] = "FINAL_JOINT_RIGID_S_COMPLETION" if accepted else "EXPLICIT_FALLBACK_RIGID_S_PROJECTION"
            elif state in {"O", "T"} and accepted:
                reported[i] = projected[i].tolist()
                sources[i] = "AMBIGUOUS_JOINT_POSE_FAMILY_PROJECTION" if material_ambiguity else "ALL_OT_JOINT_FEASIBLE_RIGID_COMPLETION"
        center_metadata = original_raw[8].tolist() if len(original_raw) == 9 else None
        if len(original_raw) == 9:
            reported.append(center_metadata)
            sources.append("ORIGINAL_NINTH_CENTER_METADATA")
        actual_pose = dict(available=True, R_cf=R.tolist(), centroid=t.tolist(),
            R_physical=(R @ geometry["physical_registration_Q"]).tolist(),
            physical_registration_Q=geometry["physical_registration_Q"].tolist(),
            cf_extents=geometry["cf_extents"].tolist(), selected_hypothesis=original_pose.get("selected_hypothesis"),
            source=pose_source, joint_feasible=accepted, fallback=not accepted,
            fallback_pose_does_not_solve_OT=not accepted)
        result = dict(schema=SCHEMA, method="STATEWISE_JOINT_OT_V2", accepted=accepted,
            states=states, state_observation=observation, initial_geometry=geometry,
            mask_audit=mask_audit, visible_correction=correction,
            frozen_visible_measurements=frozen_measurements, ot_domains=domain_bundle,
            joint_pose=joint, solution=joint, actual_pose=actual_pose,
            keypoint_readout=dict(reported_points=reported, point_sources=sources,
                rigid_model_corners=projected.tolist(), native_order_preserved=True,
                original_ninth_center_metadata=center_metadata,
                ninth_center_metadata_unchanged=True, BI_observations_preserved=True,
                mixed_points_are_one_rigid_pose=False, OT_solved=OT_hypothesis_available and not material_ambiguity,
                OT_constraints_satisfied=bool(joint.get("OT_constraints_satisfied", accepted)),
                OT_feasible_hypothesis_available=OT_hypothesis_available, OT_unique_solution_claimed=False,
                pose_family_candidates=joint.get("pose_family_candidates", []),
                U_unresolved_ids=[i for i, state in enumerate(states) if state == "U"]),
            fallback_audit=dict(applied=not accepted, pose_source=pose_source,
                reason=joint.get("fallback_reason"), O_T_reported_as_solved=OT_hypothesis_available and not material_ambiguity,
                S_projection_is_image_measurement=False,
                corrected_BI_seed_used_as_OT_solution=False,
                corrected_BI_seed_available=BI_seed is not None,
                corrected_BI_seed_measurement_budgets_pass=BI_seed_budgets_pass if BI_seed is not None else None,
                corrected_BI_seed_visible_support_guards_pass=BI_seed_support_pass if BI_seed is not None else None,
                corrected_BI_seed_visible_geometry_pass=BI_seed.get("visible_geometry_pass") if BI_seed else None,
                corrected_BI_seed_visible_canvas_pass=BI_seed.get("visible_canvas_pass") if BI_seed else None,
                corrected_BI_seed_is_underconstrained_family=BI_seed_pose_family,
                corrected_BI_seed_rejected_as_underconstrained_family=bool(not accepted and BI_seed_pose_family),
                corrected_BI_seed_rejected_for_measurement_budget=bool(not accepted and BI_seed is not None and not BI_seed_budgets_pass),
                solver_fallback_diagnostics=joint.get("fallback_diagnostics")),
            ambiguity_audit=dict(material_ambiguity=material_ambiguity,
                accepted_pose_is_one_joint_feasible_branch=accepted,
                globally_unique_pose_proved=False, solver_ambiguity=joint.get("ambiguity")),
            original_raw_points=original_raw.tolist(), original_coarse_pose=original_pose,
            provenance=dict(GT_inputs=False, old_boundary_observation_used=False,
                old_visibility_model_used=False, S_image_queries=0, S_mask_prompt_queries=0,
                S_coordinates_removed_before_image_pipeline=True,
                current_RGB_state_inference=True, B_I_measured_before_joint_OT=True,
                all_OT_simultaneous=True, O_T_regions_used_as_precise_points=False,
                initial_pose_is_initialization_only=True, original_native_model_and_registration_preserved=True,
                state_producer_injected_for_tests=self.injected_state_producer))
        return _json(result)


def refine_statewise_joint_v2(image_bgr, raw_points, K, native_xyz, coarse_pose, *,
                             state_model=None, ot_model=None, mask_provider=None, config=None,
                             state_producer=None, pallet_mask=None, state_pallet_mask=None,
                             independent_truncation=None, crop_transform=None):
    """Functional adapter to StatewiseJointV2; no hidden model/data defaults."""
    runtime = StatewiseJointV2(state_model, ot_model, mask_provider, config, state_producer)
    return runtime.refine(image_bgr, raw_points, K, native_xyz, coarse_pose,
        pallet_mask=pallet_mask, state_pallet_mask=state_pallet_mask,
        independent_truncation=independent_truncation, crop_transform=crop_transform)
