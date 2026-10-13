"""First-stage RGB visibility at initially geometric silhouette corners only.

No pose fitting, joint spatial backbone, coordinate correction or annotation
access occurs here. Geometric truncation remains an initial-pose hypothesis.
RGB evidence is queried at unchanged detector coordinates, never projections.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from . import common as C
from .calibration import predict
from .geometric_boundary_audit import initial_geometric_boundary


def _infer_evidence(image, points, hidden, weights, appearance_config, device):
    from .segmentation import segment
    from .segmented_appearance import infer_segmented_appearance
    mask = segment(image, weights, device)
    return infer_segmented_appearance(image, points, hidden, mask, appearance_config)


class BoundaryVisibility:
    """Assess a restricted first-stage subset using the frozen source/RGB heads."""

    def __init__(self, visibility_model, segmentation_weights, device=None,
                 appearance_config=None, expected_model_sha256=None):
        model_path = Path(visibility_model).resolve() if isinstance(visibility_model, (str, Path)) else None
        weights = Path(segmentation_weights).resolve()
        self.model = C.read(model_path) if model_path else deepcopy(visibility_model)
        self.weights = str(weights)
        self.device = device
        self.appearance_config = dict(appearance_config or {})
        self.model_bindings = dict(visibility_model=C.sha(model_path) if model_path else None,
                                   segmentation_weights=C.sha(weights))
        for name, digest in (expected_model_sha256 or {}).items():
            if name not in self.model_bindings or self.model_bindings[name] != digest:
                raise ValueError("Frozen boundary-visibility model binding differs: " + name)
        self.config_binding = None

    @classmethod
    def from_config(cls, path, device=None):
        """Resolve portable paths and verify a companion freeze when provided."""
        path = Path(path).resolve()
        config = C.read(path)
        expected = dict(config.get("model_sha256", {}))
        freeze_path = path.parent / "CANDIDATE_FREEZE.json"
        if freeze_path.is_file():
            freeze = C.read(freeze_path)
            if path.name in freeze.get("config_sha256", {}):
                if C.sha(path) != freeze["config_sha256"][path.name]:
                    raise ValueError("Frozen boundary configuration changed")
                expected.update(visibility_model=freeze["model_sha256"]["legacy_visibility_model"],
                                segmentation_weights=freeze["model_sha256"]["segmentation_weights"])
        result = cls((path.parent / config["visibility_model"]).resolve(),
                     (path.parent / config["segmentation_weights"]).resolve(), device,
                     config.get("appearance"), expected)
        result.config_binding = dict(path=str(path), sha256=C.sha(path))
        return result

    def assess(self, image_bgr, points, K, initial_pose):
        """Preserve q/image/K/centre; report RGB only on the geometric subset."""
        started = time.perf_counter()
        image = np.asarray(image_bgr)
        q = np.asarray(points, float)
        camera = np.asarray(K, float)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
            raise ValueError("Expected nonempty uint8 BGR image")
        if q.shape not in ((8, 2), (9, 2)):
            raise ValueError("Expected raw native corners8x2 or corners plus unchanged centre")
        if camera.shape != (3, 3) or not np.isfinite(camera).all() or min(camera[0, 0], camera[1, 1]) <= 0:
            raise ValueError("Expected finite raw-image K with positive focal lengths")
        if not initial_pose.get("available", False):
            raise ValueError("Available original initial pose required")
        R, t = np.asarray(initial_pose["R_cf"], float), np.asarray(initial_pose["centroid"], float)
        dimensions = np.asarray(initial_pose["cf_extents"], float)
        if (R.shape != (3, 3) or t.shape != (3,) or dimensions.shape != (3,)
                or not np.isfinite(R).all() or not np.isfinite(t).all()
                or not np.isfinite(dimensions).all() or np.any(dimensions <= 0)
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(R), 1., atol=1e-6)):
            raise ValueError("Initial pose must define a proper finite metre cuboid")
        xyz = C.cuboid(*dimensions)
        if np.any((xyz @ R.T + t)[:, 2] <= 1e-6):
            raise ValueError("Initial model must lie in front of camera")
        H = C.self_hidden(xyz, R, t)
        # This exact same geometry-only selector produced the published audit.
        geometry = initial_geometric_boundary(q, camera, initial_pose, H, image.shape[:2])
        eligible = np.zeros(8, bool)
        eligible[geometry["selected_ids"]] = True
        appearance = predict(_infer_evidence(image, q[:8], H, self.weights,
                                             self.appearance_config, self.device), self.model)
        threshold = float(self.model["visible_threshold"])
        records = []
        for k in range(8):
            reason = "NON_HULL_MODEL_INTERIOR" if not geometry["hull"][k] else "FIXED_SELF_HIDDEN" if H[k] else "RAW_PREDICTION_MISSING" if not geometry["raw_valid"][k] else "RAW_PREDICTION_OUTSIDE" if not geometry["raw_inside"][k] else "INITIAL_PROJECTION_OUTSIDE" if not geometry["projected_inside"][k] else "GEOMETRIC_BOUNDARY_RGB_QUERY"
            state = "NOT_ASSESSED"
            probability, RGB_state, boundary_score = None, "NOT_ASSESSED", None
            if eligible[k]:
                probability = float(appearance["visibility_score"][k])
                RGB_state = appearance["states"][k]
                boundary_score = float(appearance["boundary_score"][k])
                state = "VISIBLE" if probability >= threshold else "EXTERNAL_OCCLUSION_SUSPECTED"
            records.append(dict(corner=k, expected_outline=bool(geometry["hull"][k]),
                geometric_self_occluded=bool(H[k]), prediction_missing=bool(not geometry["raw_valid"][k]),
                prediction_out_of_frame=bool(not geometry["raw_inside"][k]) if geometry["raw_valid"][k] else None,
                conditional_truncation=bool(not geometry["projected_inside"][k]),
                geometric_boundary_candidate=bool(eligible[k]), assessment_reason=reason,
                visibility_state=state, p_visible=probability, RGB_state=RGB_state,
                RGB_boundary_score=boundary_score,
                RGB_boundary_role="BOUNDARY" if RGB_state == "VISIBLE_BOUNDARY" else "INTERIOR" if RGB_state == "VISIBLE_INTERIOR" else "UNKNOWN",
                RGB_query_xy=q[k].tolist() if eligible[k] else None,
                external_occlusion_confirmed=False, initial_projected_xy=geometry["projected"][k].tolist()))
        return C.finite(dict(schema="boundary_visibility_first_stage_v1", observed_points=q.tolist(),
            self_hidden=H.tolist(), conditional_truncation=[r["conditional_truncation"] for r in records],
            expected_outline=geometry["hull"].tolist(), geometric_boundary_candidates=eligible.tolist(),
            selected_corner_ids=geometry["selected_ids"], states=[r["visibility_state"] for r in records],
            p_visible=[r["p_visible"] for r in records], records=records,
            configuration=dict(visibility_threshold=threshold, actual_image_hw=list(image.shape[:2]),
                               model_sha256=self.model_bindings, config_binding=self.config_binding),
            provenance=dict(ground_truth_inputs=False, observation_coordinates_changed=False,
                projection_used_for="initial geometric outline/truncation flags only", projected_coordinates_sampled=False,
                pose_fits=0, joint_spatial_backbone_forwards=0, coordinate_regressions=0,
                RGB_visibility_source="frozen_clean_source_Tiny_and_fresh_CAL46_RGB_logistic",
                interior_visibility_assessed=False, self_hidden_source="fixed_original_initial_pose",
                external_occlusion_is_hypothesis=True, centre_preserved=True),
            timing=dict(total_first_stage_seconds=time.perf_counter() - started)))
