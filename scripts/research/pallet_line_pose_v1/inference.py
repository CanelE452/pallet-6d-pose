"""Original BGR -> one frozen YOLO forward -> selected-instance line refinement.

Only completed, hash-bound checkpoints and synthetic-frozen selection rules are
accepted. The wrapper has no GT argument. Detection boxes, scores, confidence
values, nonselected instances and centroid8 are preserved. Lambda zero skips
the neural branch and returns exact baseline copies.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from features import (BASELINE_SHA, FrozenYoloFeatures, branch_inputs,
                      canvas_affine, restore_refinement, sha)
from model import PalletLinePoseHead
from readout import readout

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARMS = ("image_joint", "geometry_joint", "image_line_only")
MODEL_CONFIG = {"c3": 64, "c4": 128, "hidden": 16, "along_samples": 32}


def _resolve(path: str | Path, relative_to: Path) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else relative_to / candidate).resolve()


def _bound_file(binding: dict, selection_dir: Path, expected_sha: str, name: str) -> Path:
    if binding.get("sha256") != expected_sha:
        raise ValueError(f"{name} hash differs between checkpoint and selection")
    path = _resolve(binding["path"], selection_dir)
    if not path.is_file() or sha(path) != expected_sha:
        raise ValueError(f"{name} file binding failed: {path}")
    return path


def load_selection(path: str | Path, checkpoint: dict, checkpoint_sha: str) -> dict:
    """Validate one arm/seed against a completed synthetic selection artifact."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if (data.get("schema") != "pallet_line_pose_synthetic_selection_v1"
            or data.get("complete") is not True or data.get("no_real_selection") is not True):
        raise ValueError("a complete, explicitly synthetic-only selection is required")
    _bound_file(data["source_manifest"], path.parent,
                checkpoint["source_manifest_sha256"], "source manifest")
    protocol_path = _bound_file(data["training_protocol"], path.parent,
                                checkpoint["train_protocol_sha256"], "training protocol")
    protocol = json.loads(protocol_path.read_text())
    if (protocol.get("schema") != "pallet_line_pose_training_protocol_v1"
            or protocol.get("complete") is not True or protocol.get("steps") != 6000):
        raise ValueError("the fixed main training protocol is required")
    for filename in ("model.py", "features.py", "source_data.py"):
        source = HERE / filename
        if protocol["source_sha256"].get(str(source)) != sha(source):
            raise ValueError(f"training protocol bound to different {filename}")
    for key, filename in (("model_sha256", "model.py"), ("readout_sha256", "readout.py")):
        if data.get(key) != sha(HERE / filename):
            raise ValueError(f"selection bound to different {filename}")
    arm, seed = checkpoint["arm"], int(checkpoint["seed"])
    runs = [r for r in data["runs"] if r.get("arm") == arm and int(r.get("seed", -1)) == seed]
    if len(runs) != 1 or runs[0].get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("selection must bind exactly this arm/seed checkpoint")
    selected_checkpoint = _resolve(runs[0]["checkpoint"], path.parent)
    if not selected_checkpoint.is_file() or sha(selected_checkpoint) != checkpoint_sha:
        raise ValueError("selected checkpoint file hash failed")
    temperature = float(data["temperatures"][arm][str(seed)]["temperature"])
    rule = data["selected_rules"][arm]
    lam = float(rule["lam"])
    fraction = rule["max_move_image_diagonal_fraction"]
    if (not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(lam) or lam < 0):
        raise ValueError("invalid temperature or fusion lambda")
    if fraction is not None:
        fraction = float(fraction)
        if not math.isfinite(fraction) or fraction < 0:
            raise ValueError("movement cap fraction must be finite and nonnegative")
    if (temperature not in protocol["calibration"]["temperature_grid"]
            or lam not in protocol["selection"]["lambda_grid"]
            or fraction not in protocol["selection"]["max_move_image_diagonal_fractions"]):
        raise ValueError("selection rule lies outside the frozen synthetic search grid")
    return {"path": str(path), "sha256": sha(path), "temperature": temperature,
            "lam": lam, "max_move_image_diagonal_fraction": fraction}


class PalletLinePoseInference:
    """Reusable image inference; initialized once for a checkpoint/selected rule.

    Call predict(original_bgr) for real/original images. For the already-padded
    source corpus only, call predict(prepared_bgr, already_padded=True). Returned
    candidate coordinates use the same frame as the supplied image.
    """
    def __init__(self, checkpoint: str | Path, selection: str | Path, device: str = "cuda"):
        self.checkpoint_path = Path(checkpoint).resolve()
        self.checkpoint_sha256 = sha(self.checkpoint_path)
        # Local training artifacts are trusted user-owned torch dictionaries.
        data = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if (data.get("schema") != "pallet_line_pose_checkpoint_v1"
                or data.get("complete") is not True or data.get("step") != 6000):
            raise ValueError("only completed step-6000 line-head checkpoints are accepted")
        self.arm, self.seed = data["arm"], int(data["seed"])
        if self.arm not in ARMS or self.seed not in (1, 2, 3):
            raise ValueError("unknown preregistered arm/seed")
        if data.get("model_config") != MODEL_CONFIG:
            raise ValueError("checkpoint architecture does not match the frozen model/readout")
        if data.get("baseline_checkpoint_sha256") != BASELINE_SHA:
            raise ValueError("checkpoint does not bind the frozen paper R0")
        self.selection = load_selection(selection, data, self.checkpoint_sha256)
        self.device = torch.device(device)
        self.head = PalletLinePoseHead(**data["model_config"])
        self.head.load_state_dict(data["model_state_dict"], strict=True)
        self.head.to(self.device).requires_grad_(False).eval()
        baseline_value = Path(data["baseline_checkpoint"]).expanduser()
        baseline = _resolve(baseline_value, self.checkpoint_path.parent)
        if not baseline_value.is_absolute() and not baseline.is_file():
            baseline = _resolve(baseline_value, REPO)
        self.extractor = FrozenYoloFeatures(baseline, device=device)
        self.closed = False

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def predict(self, bgr: np.ndarray, *, already_padded: bool = False,
                measure_time: bool = False, include_logits: bool = False) -> dict:
        if self.closed:
            raise RuntimeError("inference wrapper is closed")
        if (not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3
                or bgr.dtype != np.uint8 or min(bgr.shape[:2]) <= 0):
            raise ValueError("expected a nonempty uint8 BGR image")
        if any(m.training for m in self.head.modules()) or any(p.requires_grad for p in self.head.parameters()):
            raise RuntimeError("line head must remain frozen in evaluation mode")
        if already_padded and min(bgr.shape[:2]) <= 200:
            raise ValueError("already-padded source needs a positive unpadded shape")
        if measure_time:
            self._synchronize()
        started = time.perf_counter()
        captured = self.extractor.predict(bgr, already_padded=already_padded)
        candidates = copy.deepcopy(captured["candidates"])
        selected = captured["selected_index"]
        if selected is not None and not 0 <= selected < len(candidates):
            raise ValueError("extractor returned an invalid selected index")
        temperature, lam = self.selection["temperature"], self.selection["lam"]
        raw_shape = (bgr.shape[0] - 200, bgr.shape[1] - 200) if already_padded else bgr.shape[:2]
        result = {"schema": "pallet_line_pose_prediction_v1", "arm": self.arm, "seed": self.seed,
                  "checkpoint_sha256": self.checkpoint_sha256,
                  "selection_sha256": self.selection["sha256"],
                  "coordinate_frame": "supplied_image_pixels", "image_shape_hw": list(bgr.shape[:2]),
                  "raw_shape_hw": list(raw_shape), "already_padded": bool(already_padded),
                  "selected_index": selected, "candidates": candidates,
                  "baseline_selected": None if selected is None else copy.deepcopy(candidates[selected]),
                  "temperature": temperature, "lam": lam,
                  "head_used": False, "refinement_applied": False, "diagnostics": None}
        if selected is None:
            result["status"] = "no_detection_baseline_preserved"
        elif selected != int(np.argmax([r["score"] for r in candidates])):
            raise ValueError("extractor did not select the highest predicted score")
        elif lam == 0:
            # Do not round-trip baseline coordinates through float32 tensors.
            result["status"] = "lambda_zero_baseline_preserved"
        else:
            inputs = branch_inputs(captured)
            valid = np.asarray(inputs["point_valid"], bool)
            if not valid[:8].any():
                result["status"] = "no_valid_corner_baseline_preserved"
            else:
                def tensor(value, dtype=torch.float32):
                    return torch.as_tensor(value, device=self.device, dtype=dtype)[None]
                points = tensor(inputs["points"])
                boxes = tensor(inputs["boxes"])
                point_valid = tensor(valid, torch.bool)
                shape = tensor(inputs["input_shape"])
                p3 = captured["p3"].to(self.device)
                p4 = captured["p4"].to(self.device)
                if p3.ndim == 3:
                    p3, p4 = p3[None], p4[None]
                output = self.head(p3, p4, points, boxes, point_valid, shape,
                                   lam=0., geometry_only=self.arm == "geometry_joint")
                result["head_used"] = True
                fraction = self.selection["max_move_image_diagonal_fraction"]
                cap = None if fraction is None else fraction * math.hypot(*raw_shape) * inputs["gain"]
                decoded = readout(output["logits"], points, boxes, point_valid, shape,
                                  temperature=temperature, lam=lam, cap=cap)
                refined = decoded["points"][0].cpu().numpy()
                if not np.isfinite(refined[valid]).all():
                    raise RuntimeError("nonfinite refined valid point; refusing a silent fallback")
                if not bool(decoded["line_valid"].any()):
                    result["status"] = "no_valid_line_baseline_preserved"
                else:
                    restored = restore_refinement(candidates[selected], inputs["points"], refined,
                                                  inputs["gain"], lam=lam)
                    # Preserve nonmovable bits and centroid even if a zero delta
                    # would otherwise change the sign of a floating-point zero.
                    delta_input = refined - inputs["points"]
                    changed = valid & np.any(delta_input != 0, axis=-1)
                    changed[8] = False
                    original = candidates[selected]["keypoints_xy"]
                    final = original.copy()
                    final[changed] = restored[changed]
                    candidates[selected]["keypoints_xy"] = final
                    result["refinement_applied"] = bool(changed.any())
                    result["status"] = "refined" if changed.any() else "zero_movement_baseline_preserved"
                gain, offset = canvas_affine(captured["canvas_shape"], captured["input_shape"])
                mean_line = decoded["h_mean"][0].cpu().numpy()
                supplied_line = mean_line.astype(np.float64).copy()
                supplied_line[:, 2] = (mean_line[:, 2] + mean_line[:, :2] @
                                      (offset + captured["added_border"] * gain)) / gain
                probabilities = decoded["conditional_probability"][0]
                coverage = decoded["sample_coverage"][0]
                peak_index = probabilities.argmax(-1)
                role_index = torch.arange(8, device=peak_index.device)
                peak_line = decoded["candidate_h"][0, role_index, peak_index].cpu().numpy()
                peak_supplied = peak_line.astype(np.float64).copy()
                peak_supplied[:, 2] = (peak_line[:, 2] + peak_line[:, :2] @
                                      (offset + captured["added_border"] * gain)) / gain
                movement = np.linalg.norm(candidates[selected]["keypoints_xy"] -
                                          result["baseline_selected"]["keypoints_xy"], axis=-1)
                result["diagnostics"] = {
                    "input_shape_hw": list(inputs["input_shape"]), "gain": float(gain),
                    "cap_input_px": cap, "move_px": movement,
                    "line_h_input": mean_line, "line_h_supplied_image": supplied_line,
                    "peak_candidate_index": peak_index.cpu().numpy(),
                    "peak_line_h_supplied_image": peak_supplied,
                    "peak_probability_conditional": probabilities.amax(-1).cpu().numpy(),
                    "expected_sample_coverage": (probabilities * coverage).sum(-1).cpu().numpy(),
                    "peak_sample_coverage": coverage[role_index, peak_index].cpu().numpy(),
                    "linepool_along_samples": 32,
                    "line_valid": decoded["line_valid"][0].cpu().numpy(),
                    "null_probability": decoded["null_probability"][0].cpu().numpy(),
                    "entropy_normalized": decoded["entropy_normalized"][0].cpu().numpy(),
                    "line_variance_input_px2": decoded["line_variance"][0].cpu().numpy(),
                    "uncertainty_semantics": "tempered model-distribution residual spread; not a calibrated physical-error guarantee"}
                if include_logits:
                    result["diagnostics"]["logits"] = output["logits"][0].cpu().numpy()
        if measure_time:
            self._synchronize()
            result["inference_ms"] = (time.perf_counter() - started) * 1000
        return result

    def close(self) -> None:
        if not self.closed:
            self.extractor.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--already-padded", action="store_true")
    parser.add_argument("--include-logits", action="store_true")
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise ValueError(f"image cannot be decoded: {args.image}")
    with PalletLinePoseInference(args.checkpoint, args.selection, args.device) as predictor:
        output = predictor.predict(bgr, already_padded=args.already_padded,
                                   measure_time=True, include_logits=args.include_logits)
    output.update(image=str(args.image.resolve()), image_sha256=sha(args.image))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(jsonable(output), handle, allow_nan=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "status": output["status"],
                      "n_candidates": len(output["candidates"]), "lam": output["lam"]}))


if __name__ == "__main__":
    main()
