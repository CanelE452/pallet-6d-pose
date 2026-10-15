"""Optional image-observed outline intersections and conservative bounded blend.

Visibility uses the original clean-source Tiny/RGB head. Coordinates use a
class-agnostic BOX_ONLY MobileSAM mask, independently sampled outline lines,
and the previously declared sigma2 blend. Neither projected model points nor
annotation coordinates are measurements. Line-bootstrap covariance is not
calibrated semantic or keypoint accuracy confidence.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np

from . import common as C
from .boundary_visibility import BoundaryVisibility, _infer_evidence
from .calibration import predict
from .boundary_refine import infer_rgb_boundary_points
from .geometric_boundary_audit import initial_geometric_boundary
from scripts.research.pallet_foundation_mask_20261013_v1.foundation import choose_mask
from scripts.research.pallet_foundation_mask_20261013_v1.uncertainty import intersection_uncertainty, fuse_observations


class BoundaryObservation:
    def __init__(self, visibility, *, boundary_profile, uncertainty, fusion,
                 mobile_sam_vendor=None, mobile_sam_checkpoint=None,
                 mobile_sam_checkpoint_sha256=None, mask_provider=None):
        self.visibility = visibility
        self.boundary_profile = deepcopy(boundary_profile)
        self.uncertainty = dict(uncertainty)
        self.fusion = dict(fusion)
        self.vendor = Path(mobile_sam_vendor).resolve() if mobile_sam_vendor else None
        self.checkpoint = Path(mobile_sam_checkpoint).resolve() if mobile_sam_checkpoint else None
        self.checkpoint_sha256 = mobile_sam_checkpoint_sha256
        self.mask_provider = mask_provider
        self.predictor = None
        self.config_binding = None
        if mask_provider is None and (self.vendor is None or self.checkpoint is None or not self.checkpoint_sha256):
            raise ValueError("Frozen MobileSAM vendor/checkpoint/SHA required")

    @classmethod
    def from_config(cls, path, device=None):
        path = Path(path).resolve()
        config = C.read(path)
        checkpoint = (path.parent / config["mobile_sam_checkpoint"]).resolve()
        expected = config["model_sha256"]["mobile_sam_checkpoint"]
        if C.sha(checkpoint) != expected:
            raise ValueError("Frozen MobileSAM checkpoint binding differs")
        result = cls(BoundaryVisibility.from_config(path.parent / config["visibility_config"], device),
            boundary_profile=config["boundary_profile"], uncertainty=config["uncertainty"], fusion=config["fusion"],
            mobile_sam_vendor=path.parent / config["mobile_sam_vendor"], mobile_sam_checkpoint=checkpoint,
            mobile_sam_checkpoint_sha256=expected)
        result.config_binding = dict(path=str(path), sha256=C.sha(path))
        return result

    def _mask_candidates(self, image, box):
        if self.mask_provider is not None:
            return self.mask_provider(image, box.copy())
        import cv2
        import torch
        if self.predictor is None:
            if C.sha(self.checkpoint) != self.checkpoint_sha256:
                raise ValueError("MobileSAM checkpoint changed before lazy load")
            sys.path.insert(0, str(self.vendor))
            from mobile_sam import SamPredictor, sam_model_registry
            import mobile_sam
            if not Path(mobile_sam.__file__).resolve().is_relative_to(self.vendor):
                raise ValueError("MobileSAM imported from outside configured vendor")
            model = sam_model_registry["vit_t"]()
            model.load_state_dict(torch.load(self.checkpoint, map_location="cpu", weights_only=True))
            device = self.visibility.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.predictor = SamPredictor(model.to(device).eval().requires_grad_(False))
        with torch.inference_mode():
            self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            masks, quality, _ = self.predictor.predict(box=box, multimask_output=True)
        return dict(masks=masks, quality=quality)

    def assess(self, image_bgr, points, K, initial_pose):
        started = time.perf_counter()
        image = np.asarray(image_bgr)
        q = np.asarray(points, float)
        K = np.asarray(K, float)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
            raise ValueError("Expected nonempty uint8 BGR image")
        if q.shape not in ((8, 2), (9, 2)) or K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise ValueError("Expected nativeq8/9 and finite positive-focal K")
        if not initial_pose.get("available", False):
            raise ValueError("Available original initial pose required")
        R = np.asarray(initial_pose["R_cf"], float)
        t = np.asarray(initial_pose["centroid"], float)
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
        geometry = initial_geometric_boundary(q, K, initial_pose, H, image.shape[:2])
        # Exactly one Tiny/appearance pass; SAM never changes these feature meanings.
        appearance = predict(_infer_evidence(image, q[:8], H, self.visibility.weights,
                            self.visibility.appearance_config, self.visibility.device), self.visibility.model)
        threshold = float(self.visibility.model["visible_threshold"])
        eligible = np.zeros(8, bool)
        eligible[geometry["selected_ids"]] = True
        restricted = dict(self_hidden=H.tolist(), expected_outline=geometry["hull"].tolist(),
            conditional_truncation=(~geometry["projected_inside"]).tolist(),
            geometric_boundary_candidates=eligible.tolist(), selected_corner_ids=geometry["selected_ids"],
            states=[("VISIBLE" if appearance["visibility_score"][k] >= threshold else "EXTERNAL_OCCLUSION_SUSPECTED")
                    if eligible[k] else "NOT_ASSESSED" for k in range(8)],
            p_visible=[float(appearance["visibility_score"][k]) if eligible[k] else None for k in range(8)],
            RGB_states=[appearance["states"][k] if eligible[k] else "NOT_ASSESSED" for k in range(8)],
            visibility_threshold=threshold, external_occlusion_confirmed=False)
        valid = np.isfinite(q[:8]).all(1) & ~np.all(q[:8] == -1, axis=1)
        h, w = image.shape[:2]
        box = None
        if valid.any():
            lo = np.maximum(q[:8][valid].min(0), [0., 0.])
            hi = np.minimum(q[:8][valid].max(0), [w - 1., h - 1.])
            if np.all(hi - lo >= 2):
                box = np.r_[lo, hi]
        if box is None:
            mask = np.zeros((h, w), bool)
            mask_info = dict(policy="BOX_ONLY", box=None, reason="missing_or_degenerate_raw_query_box", mask_forward_calls=0)
        else:
            candidates = self._mask_candidates(image, box)
            mask, index = choose_mask(candidates["masks"], candidates["quality"])
            if mask.shape != (h, w):
                raise ValueError("MobileSAM mask must retain original canvas dimensions")
            mask_info = dict(policy="BOX_ONLY", box=box.tolist(), points=[], labels=[],
                             quality=np.asarray(candidates["quality"]).tolist(), selected_index=index,
                             quality_is_calibrated_visibility=False, mask_forward_calls=1)
        correction = infer_rgb_boundary_points(image, q, H, appearance, self.boundary_profile,
                material_mask=mask, mask_source="FRESH_CAL46_FROZEN_OFFICIAL_MOBILESAM_FULLFRAME")
        covariance, lines = intersection_uncertainty(correction, **self.uncertainty)
        fused, blends = fuse_observations(q, correction, covariance, **self.fusion)
        observed = np.asarray(correction["points"], float)
        np.testing.assert_array_equal(observed[:8][H], q[:8][H])
        np.testing.assert_array_equal(fused[:8][H], q[:8][H])
        np.testing.assert_array_equal(observed[8:], q[8:])
        np.testing.assert_array_equal(fused[8:], q[8:])
        return C.finite(dict(schema="boundary_observation_v1", raw_points=q.tolist(),
            boundary_visibility=restricted, visibility_appearance=appearance,
            initial_geometric_outline=geometry["hull"].tolist(), fixed_H=H.tolist(), self_hidden=H.tolist(),
            observed_points=correction["points"], fused_points=fused.tolist(),
            correction=correction, line_support=lines, intersection_covariance=covariance, fusion=blends,
            mask=mask_info, config_binding=self.config_binding,
            model_sha256=dict(self.visibility.model_bindings, mobile_sam_checkpoint=self.checkpoint_sha256),
            provenance=dict(GT_inputs=False, pose_fits=0, joint_spatial_backbone_forwards=0,
                raw_points_and_center_preserved=True, geometric_flags_fixed_before_correction=True,
                coordinate_hull_source="original_raw_q8", curriculum_outline_source="original_initial_projected_cuboid8",
                projected_coordinates_used_as_measurements=False, projected_coordinates_used_for_mask_prompt=False,
                SAM_changed_RGB_head_features=False, covariance_is_calibrated_keypoint_confidence=False,
                fusion_is_calibrated_Bayes_estimator=False, detector_sigma_is_assumed=True),
            limits=["SAM mask is class agnostic; foreground support does not prove requested3Dcorner ownership",
                    "Box and local line search depend on rawq; RGB coordinates are not statistically independent of detector coordinates",
                    "Line covariance measures conditional resampling variability, omitting semantic/virtual/roundedcorner bias; priorCAL95ellipse coverage31.8percent",
                    "Adjacent corrected corners share fitted/resampled edges and remain correlated",
                    "No threshold or uncertainty-based visibility/solver weight change is made"],
            timing=dict(total_observation_seconds=time.perf_counter() - started)))
