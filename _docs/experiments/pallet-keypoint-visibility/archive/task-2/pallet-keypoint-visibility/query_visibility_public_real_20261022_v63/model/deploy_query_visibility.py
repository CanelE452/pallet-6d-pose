"""Frozen query-pixel visibility API for a NEW RGB image and eight N0 points.

No annotations, CAL state, pose fitting, or PnP are imported. Assets are bound
by the deployment bundle. Heavy YOLO/SAM dependencies load only in fresh-RGB
mode; predict_precomputed() needs the frozen small head and NumPy/Torch only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(binding):
    path = Path(binding["path"]).resolve()
    if sha256(path) != binding["sha256"]:
        raise ValueError(f"frozen deployment asset changed: {path}")
    return path


def module(binding, role):
    path = verify(binding)
    name = "_frozen_query_" + role + "_" + binding["sha256"][:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
    return sys.modules[name]


def points8(points):
    q = np.asarray(points, dtype=np.float64)
    if q.shape != (8, 2):
        raise ValueError("exactly eight native-order initial query points are required")
    return q


def query_metadata(points, raw_hw):
    q = points8(points)
    h, w = map(int, raw_hw)
    if min(h, w) <= 0:
        raise ValueError("positive raw image dimensions required")
    finite = np.isfinite(q).all(1) & ~(q == -1).all(1)
    safe = np.where(np.isfinite(q), q, -1e8)
    pixel = np.floor(safe + .5).astype(np.int64)
    valid = finite & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    valid &= (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
    box = None
    if finite.sum() >= 2:
        lo = np.maximum(q[finite].min(0), [0., 0.])
        hi = np.minimum(q[finite].max(0), [w - 1., h - 1.])
        if (hi - lo >= 2).all():
            box = np.r_[lo, hi]
    return q, pixel, valid, box


def sam_grid(logit, points, side):
    q = points8(points)
    h, w = logit.shape
    offset = (np.arange(9) - 4) * float(side) / 9
    dx, dy = np.meshgrid(offset, offset)
    xy = q[:, None, None] + np.stack((dx, dy), -1)[None]
    finite = np.isfinite(xy).all(-1)
    pixel = np.floor(np.where(np.isfinite(xy), xy, -1e8) + .5).astype(np.int64)
    valid = finite & (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    valid &= (pixel[..., 0] >= 0) & (pixel[..., 0] < w) & (pixel[..., 1] >= 0) & (pixel[..., 1] < h)
    sampled = logit[np.clip(pixel[..., 1], 0, h - 1), np.clip(pixel[..., 0], 0, w - 1)]
    return np.where(valid, np.clip(sampled / 8, -4, 4), 0).astype(np.float16)[:, None]


class QueryVisibility:
    def __init__(self, bundle=None, device="cpu"):
        self.bundle_path = Path(bundle or Path(__file__).with_name("DEPLOYMENT_BUNDLE.json")).resolve()
        self.bundle = json.loads(self.bundle_path.read_text())
        if "entrypoint" in self.bundle:
            verify(self.bundle["entrypoint"])
        self.device = device
        asset = self.bundle["assets"]
        self.head_code = module(asset["head_code"], "head")
        self.model, self.payload = self.head_code.load_model(verify(asset["checkpoint"]), device)
        freeze = json.loads(verify(asset["model_freeze"]).read_text())
        if freeze["checkpoint_sha256"] != asset["checkpoint"]["sha256"] or freeze["threshold"] != .5:
            raise ValueError("source model freeze and fixed threshold do not match bundle")
        if freeze["eval_data_used"] is not False or freeze["PnP_used"] is not False:
            raise ValueError("only source-trained, PnP-free frozen checkpoint is supported")
        self.model.eval().requires_grad_(False)
        self.threshold = .5
        self.backbone = self.sam_predictor = self.sam_model = None
        self._sam_image_key = None
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False

    def _geometry(self, pose):
        if not pose or not pose.get("available", True):
            return [None] * 8, "UNKNOWN_NO_SAVED_INITIAL_POSE"
        try:
            geometry = module(self.bundle["assets"]["geometry"], "geometry")
            flags = geometry.conditional_self_hidden(pose["cf_extents"], pose["R_cf"], pose["centroid"])
            return flags.astype(bool).tolist(), "SAVED_INITIAL_CUBOID_PROXY_ONLY"
        except (KeyError, ValueError, TypeError) as error:
            return [None] * 8, "UNKNOWN_INVALID_SAVED_GEOMETRY: " + str(error)

    def _empty(self, points, raw_hw, reason, pose=None):
        q, pixel, valid, box = query_metadata(points, raw_hw)
        flags, geometry_status = self._geometry(pose)
        rows = []
        for k in range(8):
            rows.append({"native_corner": k, "query_xy": [float(v) if np.isfinite(v) else None for v in q[k]],
                         "pixel_xy": pixel[k].tolist() if valid[k] else None, "status": "UNKNOWN",
                         "reason": reason if valid[k] else "INVALID_QUERY", "p_visible_exact": None,
                         "pred_visible_exact": None, "learned_logit_exact": None,
                         "raw_SAM_center_logit": None, "raw_SAM_center_source": "UNAVAILABLE",
                         "nearby_r2_support": {"status": "UNKNOWN", "raw_logit": None, "present": None},
                         "self_hidden_geometry_proxy": flags[k], "geometry_status": geometry_status})
        return {"schema": "FROZEN_QUERY_VISIBILITY_V1", "candidate": self.bundle["candidate"],
                "threshold": self.threshold, "raw_hw": list(map(int, raw_hw)),
                "prompt_box_xyxy": box.tolist() if box is not None else None,
                "queries": rows, "PnP_calls": 0,
                "self_geometry_changes_learned_probability": False}

    def predict_precomputed(self, *, points, raw_hw, x, rgb, sam, context=None,
                            sam_full_logits=None, saved_initial_pose=None):
        """Read one target's frozen 8-query caches; no RGB file is opened.

        x=(8,65,9,9), rgb=uint8(8,3,65,65), sam=(8,1,9,9) normalized
        selected SAM logits/8 clipped[-4,4]. Context defaults to the same
        valid-cell average used in source training. Full SAM logits are
        optional; without them raw center is reconstructed from the clipped
        cache and radius-2 support is explicitly UNKNOWN.
        """
        result = self._empty(points, raw_hw, "MISSING_OR_DEGENERATE_QUERY_BOX", saved_initial_pose)
        if result["prompt_box_xyxy"] is None:
            return result
        q, pixel, valid, _ = query_metadata(points, raw_hw)
        x, rgb, sam = np.asarray(x), np.asarray(rgb), np.asarray(sam)
        if x.shape != (8, 65, 9, 9) or rgb.shape != (8, 3, 65, 65) or sam.shape != (8, 1, 9, 9):
            raise ValueError("precomputed x/rgb/sam dimensions do not match the frozen recipe")
        if rgb.dtype != np.uint8 or not np.isfinite(x).all() or not np.isfinite(sam).all() or np.abs(sam).max() > 4.000001:
            raise ValueError("finite source-compatible features, uint8 RGB and clipped SAM grid are required")
        roi_valid = x[:, 64].astype(bool).any(axis=(1, 2))
        for k in np.flatnonzero(valid & ~roi_valid):
            result["queries"][k]["reason"] = "INVALID_OR_EMPTY_ROI"
        valid &= roi_valid
        prototype = self.head_code.target_context_from_rois(x) if context is None else np.asarray(context)
        if prototype.shape == (64,):
            prototype = np.repeat(prototype[None], 8, axis=0)
        if prototype.shape != (8, 64) or not np.isfinite(prototype).all():
            raise ValueError("context must be finite (64,) or (8,64)")
        ix = np.flatnonzero(valid)
        if not len(ix):
            return self._empty(points, raw_hw, "NO_VALID_QUERY_ROI", saved_initial_pose)
        with torch.inference_mode():
            tensor = lambda a: torch.as_tensor(np.asarray(a), device=self.device).float()
            z = self.model(tensor(x[ix]), tensor(rgb[ix]) / 255, context=tensor(prototype[ix]), sam=tensor(sam[ix])).cpu().numpy()
        if sam_full_logits is not None:
            sam_full_logits = np.asarray(sam_full_logits)
            if sam_full_logits.shape != tuple(raw_hw) or not np.isfinite(sam_full_logits).all():
                raise ValueError("finite original-resolution SAM logits required for exact neighborhood readout")
        for k, logit in zip(ix, z):
            row = result["queries"][k]
            if not np.isfinite(logit):
                row["reason"] = "NONFINITE_HEAD_OUTPUT"
                continue
            p = float(1 / (1 + np.exp(-np.clip(float(logit), -80, 80))))
            row.update(status="OK", reason=None, learned_logit_exact=float(logit), p_visible_exact=p,
                       pred_visible_exact=bool(p >= self.threshold), raw_SAM_center_logit=float(8 * sam[k, 0, 4, 4]),
                       raw_SAM_center_source="CLIPPED_NORMALIZED_CACHE_RECONSTRUCTION")
            if sam_full_logits is not None:
                px, py = pixel[k]
                h, w = raw_hw
                center = float(sam_full_logits[py, px])
                maximum = float(sam_full_logits[max(0, py - 2):min(h, py + 3), max(0, px - 2):min(w, px + 3)].max())
                row.update(raw_SAM_center_logit=center, raw_SAM_center_source="FULL_RESOLUTION_SELECTED_LOGIT")
                row["nearby_r2_support"] = {"status": "OK", "raw_logit": maximum, "present": bool(maximum > 0),
                                            "definition": "any frozen-SAM support in clipped 5x5 pixel square"}
        return result

    def _load_backbone(self):
        if self.backbone is None:
            os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).parent / "deployment_runtime/ultralytics"))
            assets = self.bundle["assets"]
            self.backbone_module = module(assets["backbone_code"], "backbone")
            self.roi_module = module(assets["roi_code"], "roi")
            self.backbone = self.backbone_module.FrozenYoloFeatures(verify(assets["backbone_checkpoint"]), device=self.device)

    def _load_sam(self):
        if self.sam_predictor is None:
            asset = self.bundle["sam"]
            for binding in asset["runtime_files"]:
                verify(binding)
            weights = verify(asset["checkpoint"])
            vendor = Path(asset["vendor"]).resolve()
            sys.path.insert(0, str(vendor))
            import mobile_sam
            if not Path(mobile_sam.__file__).resolve().is_relative_to(vendor):
                # Reuse an already imported byte-identical pinned package.
                # This permits integration with an existing generic SAM user,
                # while a different package/version still fails its hashes.
                loaded_vendor = Path(mobile_sam.__file__).resolve().parent.parent
                for binding in asset["runtime_files"]:
                    relative = Path(binding["path"]).resolve().relative_to(vendor)
                    verify({"path": str(loaded_vendor / relative), "sha256": binding["sha256"]})
            self.sam_model = mobile_sam.sam_model_registry["vit_t"]()
            self.sam_model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
            self.sam_model.to(self.device).eval().requires_grad_(False)
            self.sam_predictor = mobile_sam.SamPredictor(self.sam_model)

    def predict_rgb(self, image_rgb, points, saved_initial_pose=None):
        """NEW uint8 RGB array + unmodified native-order initial queries."""
        image_rgb = np.asarray(image_rgb)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("uint8 H,W,3 RGB required")
        hw = image_rgb.shape[:2]
        q, _, _, box = query_metadata(points, hw)
        if box is None:
            return self._empty(points, hw, "MISSING_OR_DEGENERATE_QUERY_BOX", saved_initial_pose)
        self._load_backbone()
        self._load_sam()
        safe_q = q.copy()
        safe_q[~np.isfinite(safe_q).all(1)] = [-1, -1]
        captured = self.backbone.predict(np.ascontiguousarray(image_rgb[..., ::-1]), cpu_features=False)
        gain, offset = self.backbone_module.canvas_affine(captured["canvas_shape"], captured["input_shape"])
        roi, _, side = self.roi_module.patches(captured, safe_q, hw, gain, offset)
        image_key = hashlib.sha256(str(image_rgb.shape).encode() + image_rgb.tobytes()).hexdigest()
        if image_key != self._sam_image_key:
            self.sam_predictor.set_image(image_rgb)
            self._sam_image_key = image_key
        with torch.inference_mode():
            logits, quality, _ = self.sam_predictor.predict(box=box.astype(np.float32), multimask_output=True, return_logits=True)
        quality = np.asarray(quality)
        if quality.shape != (3,) or not np.isfinite(quality).all():
            return self._empty(points, hw, "INVALID_SAM_MASK_QUALITY", saved_initial_pose)
        choice = int(np.argmax(quality))
        selected = np.asarray(logits[choice])
        recipe = module(self.bundle["assets"]["rgb_recipe"], "rgb")
        crops = np.zeros((8, 3, 65, 65), np.uint8)
        for k in range(8):
            if np.isfinite(q[k]).all() and not (q[k] == -1).all():
                crops[k] = recipe.query_rgb_crop(image_rgb, *q[k], side).transpose(2, 0, 1)
        result = self.predict_precomputed(points=q, raw_hw=hw, x=roi.numpy(), rgb=crops,
                                          sam=sam_grid(selected, q, side), sam_full_logits=selected,
                                          saved_initial_pose=saved_initial_pose)
        result.update(roi_side=float(side), sam_selected_index=choice, sam_uncalibrated_mask_quality=quality.tolist(),
                      image_RGB_sha256=image_key, image_encoder_recipe="frozen source YOLO26 + frozen MobileSAM box-only")
        return result

    def close(self):
        if self.backbone is not None:
            self.backbone.close()
        self.backbone = self.sam_predictor = self.sam_model = None
        self._sam_image_key = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, help="image, initial_points(8x2), optional initial_pose/request_id")
    parser.add_argument("--bundle")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("refusing to overwrite inference JSON")
    request = json.loads(Path(args.input_json).read_text())
    if set(request) - {"image", "initial_points", "initial_pose", "request_id"}:
        raise ValueError("request supports only RGB path, N0 points and optional saved initial geometry/request ID")
    from PIL import Image
    with Image.open(request["image"]) as image:
        rgb = np.array(image.convert("RGB"))
    api = QueryVisibility(args.bundle, args.device)
    try:
        result = api.predict_rgb(rgb, request["initial_points"], request.get("initial_pose"))
        result.update(image_file_sha256=sha256(request["image"]), deployment_bundle_sha256=sha256(api.bundle_path),
                      request_id=request.get("request_id"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(str(out.resolve()))
    finally:
        api.close()


if __name__ == "__main__":
    main()
