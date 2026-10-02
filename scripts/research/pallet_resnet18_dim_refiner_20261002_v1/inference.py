"""GT-free DEV319 inference for FULL and twelve frozen correction heads."""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from . import common as C
from .refiner import forward_arm
from .selection import load_heads, refined_delta, validate_selection_payload


SCHEMA = "resnet18_full_dim_refiner_dev_predictions_v1"
BASELINE = "FULL"
METHODS = (BASELINE,) + tuple(C.arm_key(arm, seed)
                             for arm in C.HEAD_ARMS for seed in C.SEEDS)


def nullable(value):
    array = np.asarray(value)
    if array.ndim:
        return [nullable(item) for item in array]
    return float(array) if np.isfinite(array) else None


def canonical_dimensions(object_type: str, registry_path: Path | str = C.GEOMETRY_REGISTRY):
    payload = C.read_json(registry_path)
    matches = [row for row in payload["objects"] if row["object_type"] == object_type]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate deployment object type: {object_type}")
    raw = matches[0]["physical_dimensions_m"]
    dimensions = np.asarray([raw["x"], raw["z"], raw["y"]], np.float64)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or not (dimensions > 0).all():
        raise ValueError(f"Invalid registry dimensions: {object_type}")
    return dimensions


def validate_heldout_receipt(payload: dict) -> dict:
    expected = {BASELINE, *METHODS[1:]}
    if (payload.get("complete") is not True
            or payload.get("scope") !=
            "Frozen source heldout partition; not an independent backbone-development test."
            or payload.get("selection") != C.binding(C.DOC / "SELECTION.json")
            or set(payload.get("results", {})) != expected):
        raise ValueError("Synthetic heldout receipt drift")
    for method, result in payload["results"].items():
        if (result.get("frames") != 1985 or result.get("gt9", -1) < 1
                or result.get("missing9", -1) < 0
                or not np.isfinite(result.get("score", np.nan))):
            raise ValueError(f"Invalid synthetic heldout result: {method}")
    return payload


def validate_prediction_payload(payload: dict, *, protocol=None) -> dict:
    """Validate a complete DEV payload so an orphaned immutable output is reusable."""
    protocol = C.verify_protocol() if protocol is None else protocol
    if (payload.get("complete") is not True or payload.get("schema") != SCHEMA
            or payload.get("coordinate_system") != "original_unpadded_pixels"
            or payload.get("methods") != list(METHODS)
            or payload.get("real_accuracy_read") is not False
            or payload.get("GT_inputs") is not False
            or payload.get("camera_inputs") is not False
            or payload.get("all_image_bytes_bound") is not True
            or payload.get("center_box_score_instance_and_missing_preserved") is not True):
        raise ValueError("Incomplete or wrong-schema DEV predictions")
    replay = payload.get("selection_inference_replay_max_abs_px", np.nan)
    if not np.isfinite(replay) or replay < 0:
        raise ValueError("Invalid selection/inference replay check")
    if payload.get("protocol") != C.binding(C.PROTOCOL):
        raise ValueError("Prediction protocol binding drift")
    for key in ("baseline", "baseline_protocol", "baseline_training_complete"):
        if payload.get(key) != protocol[key]:
            raise ValueError(f"Prediction {key}/protocol mismatch")
        C.verify_binding(payload[key])
    expected_artifacts = {
        "cache": C.DOC / "SOURCE_CACHE_COMPLETE.json",
        "selection": C.DOC / "SELECTION.json",
        "training": C.DOC / "TRAINING_COMPLETE.json",
        "synthetic_heldout": C.DOC / "SYNTHETIC_HELDOUT.json",
        "population": C.DEV_POS,
        "registry": C.GEOMETRY_REGISTRY,
    }
    for key, expected in expected_artifacts.items():
        if payload.get(key) != C.binding(expected):
            raise ValueError(f"Prediction {key} binding drift")
    from .source_cache import _verify_completion
    _verify_completion(C.read_json(C.DOC / "SOURCE_CACHE_COMPLETE.json"))
    selection = validate_selection_payload(C.read_json(C.DOC / "SELECTION.json"), protocol=protocol)
    del selection
    validate_heldout_receipt(C.read_json(C.DOC / "SYNTHETIC_HELDOUT.json"))
    training = C.read_json(C.DOC / "TRAINING_COMPLETE.json")
    if (training.get("complete") is not True or training.get("baseline") != protocol["baseline"]
            or training.get("protocol") != C.binding(C.PROTOCOL)
            or training.get("fits") != len(C.HEAD_ARMS) * len(C.SEEDS)
            or training.get("steps_per_fit") != C.STEPS
            or training.get("real_training_images") != 0):
        raise ValueError("Prediction training provenance drift")
    seed_receipts = [C.read_json(C.DOC / f"TRAIN_SEED{seed}.json") for seed in C.SEEDS]
    if training.get("seeds") != seed_receipts:
        raise ValueError("Training/seed completion receipt drift")
    expected_checkpoints = {C.arm_key(arm, seed): C.binding(
        C.RAW / "runs" / f"seed{seed}" / "paired_last.pt")
        for arm in C.HEAD_ARMS for seed in C.SEEDS}
    if payload.get("checkpoints") != expected_checkpoints:
        raise ValueError("Prediction checkpoint binding drift")
    for seed, receipt in zip(C.SEEDS, seed_receipts):
        if (receipt.get("complete") is not True or receipt.get("seed") != seed
                or receipt.get("updates_per_arm") != C.STEPS
                or receipt.get("arms") != list(C.HEAD_ARMS)
                or receipt.get("real_training_images") != 0
                or receipt.get("checkpoint") != expected_checkpoints[C.arm_key("D0", seed)]):
            raise ValueError(f"Invalid training completion receipt: seed {seed}")
    expected_code = [C.binding(C.HERE / "inference.py"), C.binding(C.HERE / "evaluation.py")]
    if payload.get("code") != expected_code:
        raise ValueError("Prediction code binding drift")
    records = dev_records()
    if len(payload.get("frames", [])) != 319:
        raise ValueError("DEV frame count drift")
    from .evaluation import validate_rows
    validate_rows(payload["frames"], [record["image_key"] for record in records])
    for row, record in zip(payload["frames"], records):
        if (row.get("frame_id") != record["frame_id"]
                or row.get("session_id") != record["session_id"]
                or row.get("object_type") != record["object_type"]
                or not np.array_equal(np.asarray(row.get("canonical_WDH_m"), np.float64),
                                      record["dimensions"])):
            raise ValueError("DEV prediction identity/dimension drift")
        raw = record["image"].read_bytes()
        if row.get("image_sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError(f"DEV image-byte binding drift: {record['frame_id']}")
        affine = np.asarray(row.get("affine_input_to_net"), np.float64)
        if affine.shape != (3, 3) or not np.isfinite(affine).all():
            raise ValueError(f"Invalid DEV affine: {record['frame_id']}")
    return payload


def inference_receipt(payload: dict, destination: Path | str) -> dict:
    return dict(
        complete=True,
        predictions=C.binding(destination),
        frames=319,
        methods=len(METHODS),
        real_accuracy_read=False,
        selection_inference_replay_max_abs_px=payload[
            "selection_inference_replay_max_abs_px"],
    )


def dev_records() -> list[dict]:
    payload = C.read_json(C.DEV_POS)
    items = payload["items"]
    if len(items) != 319 or len({row["frame_id"] for row in items}) != 319:
        raise ValueError("DEV319 membership drift")
    rows = []
    for item in items:
        image = C.ROOT / item["image_path"]
        rows.append(dict(
            frame_id=item["frame_id"],
            session_id=item["session_id"],
            object_type=item["object_type"],
            image=image,
            image_key=str(image.resolve().relative_to(C.ROOT.resolve())),
            dimensions=canonical_dimensions(item["object_type"]),
        ))
    if len({row["session_id"] for row in rows}) != 13:
        raise ValueError("DEV session membership drift")
    return rows


class Predictor:
    def __init__(self, device="cuda", *, adapter=None, heads=None, selection=None):
        protocol = C.verify_protocol()
        if selection is None:
            selection = C.read_json(C.DOC / "SELECTION.json")
        selection = validate_selection_payload(selection, protocol=protocol)
        if adapter is None:
            from .full_adapter import FrozenFullAdapter
            adapter = FrozenFullAdapter(device)
        self.adapter = adapter
        self.device = adapter.device
        if adapter.checkpoint["sha256"] != protocol["baseline"]["sha256"]:
            raise ValueError("Inference baseline differs from protocol")
        self.heads = load_heads(self.device) if heads is None else heads
        expected = set(METHODS[1:])
        if set(self.heads) != expected:
            raise ValueError("Inference requires all twelve final seeded heads")
        self.selection = selection

    def _delta(self, value, support, base, arm, key, raw_hw):
        arrays = dict(
            boxes=np.asarray(base["bbox_net"], np.float32)[None],
            scale_xy=np.diag(np.asarray(base["affine_input_to_net"], np.float64))[None, :2],
            raw_diagonal=np.asarray([math.hypot(*raw_hw)], np.float64),
        )
        rule = self.selection["rules"][arm]
        delta = refined_delta(
            dict(value=np.asarray(value), support=np.asarray(support)), arm,
            self.selection["temperatures"][key], arrays,
            rule["lam"], rule["max_move_image_diagonal_fraction"])
        return delta[0]

    @torch.no_grad()
    def predict(self, image, dimensions):
        base = self.adapter.infer(image, dimensions, return_features=True)
        box = np.asarray(base["bbox_net"], np.float32)
        batch = dict(
            p3=base["features"][0][None].to(torch.float16),
            p4=base["features"][1][None].to(torch.float16),
            points=torch.from_numpy(np.asarray(base["points_net"], np.float32)[None]).to(self.device),
            boxes=torch.from_numpy(box[None]).to(self.device),
            point_valid=torch.from_numpy(np.asarray(base["valid"], bool)[None]).to(self.device),
            input_shape=torch.tensor([base["input_shape"][-2:]], device=self.device),
            dimension_context=torch.from_numpy(
                np.asarray(base["dimension_context"], np.float32)[None]).to(self.device),
        )
        refined = {}
        max_replay = 0.
        for arm in C.HEAD_ARMS:
            for seed in C.SEEDS:
                key = C.arm_key(arm, seed)
                output = forward_arm(self.heads[key], batch, arm, lam=0.)
                field = "delta_normalized" if arm == "D0" else "logits"
                value = output[field].detach().cpu().numpy()
                support = output["point_support"].detach().cpu().numpy()
                if not np.isfinite(value).all():
                    raise FloatingPointError(f"Nonfinite correction output: {key}")
                delta = self._delta(value, support, base, arm, key, image.shape[:2])
                points = np.asarray(base["points_original"], np.float64).copy()
                usable = np.asarray(base["valid"][:8], bool) & support[0]
                points[:8][usable] += delta[usable]
                if not np.array_equal(points[8], base["points_original"][8], equal_nan=True):
                    raise AssertionError("Center point changed")
                if not np.array_equal(np.isfinite(points).all(-1), base["valid"]):
                    raise AssertionError("Missing mask changed")
                rule = self.selection["rules"][arm]
                if rule["lam"] == 0. and not np.array_equal(
                        points, base["points_original"], equal_nan=True):
                    raise AssertionError("Zero-lambda rule failed exact preservation")
                # Recompute through the same pure path to bind train/selection inference math.
                replay = self._delta(value, support, base, arm, key, image.shape[:2])
                max_replay = max(max_replay, float(np.max(np.abs(replay - delta))))
                refined[key] = points
        base.pop("features")
        return base, refined, max_replay


def dev(device="cuda") -> dict:
    protocol = C.verify_protocol()
    destination = C.RAW / "DEV_PREDICTIONS.json"
    if destination.exists():
        payload = validate_prediction_payload(C.read_json(destination), protocol=protocol)
        C.write_frozen_json(C.DOC / "DEV_INFERENCE_COMPLETE.json",
                            inference_receipt(payload, destination))
        return payload
    heldout = C.read_json(C.DOC / "SYNTHETIC_HELDOUT.json")
    validate_heldout_receipt(heldout)
    predictor = Predictor(device)
    rows = []
    max_replay = 0.
    for index, record in enumerate(dev_records()):
        raw = record["image"].read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(record["image"])
        base, refined, replay = predictor.predict(image, record["dimensions"])
        max_replay = max(max_replay, replay)
        rows.append(dict(
            image_key=record["image_key"],
            frame_id=record["frame_id"],
            session_id=record["session_id"],
            object_type=record["object_type"],
            canonical_WDH_m=record["dimensions"].tolist(),
            original_hw=list(image.shape[:2]),
            image_sha256=hashlib.sha256(raw).hexdigest(),
            base_points=nullable(base["points_original"]),
            point_valid=np.asarray(base["valid"], bool).tolist(),
            box_original=nullable(base["bbox_original"]),
            score=float(base["score"]),
            confidence=nullable(base["confidence"]),
            affine_input_to_net=np.asarray(base["affine_input_to_net"]).tolist(),
            refined={key: nullable(value) for key, value in refined.items()},
        ))
        if index == 0 or (index + 1) % 25 == 0:
            print("FULL_DIM_REFINER_DEV", index + 1, 319, flush=True)
    payload = dict(
        complete=True,
        schema=SCHEMA,
        created_at=C.now(),
        coordinate_system="original_unpadded_pixels",
        methods=list(METHODS),
        protocol=C.binding(C.PROTOCOL),
        baseline=protocol["baseline"],
        baseline_protocol=protocol["baseline_protocol"],
        baseline_training_complete=protocol["baseline_training_complete"],
        cache=C.binding(C.DOC / "SOURCE_CACHE_COMPLETE.json"),
        selection=C.binding(C.DOC / "SELECTION.json"),
        training=C.binding(C.DOC / "TRAINING_COMPLETE.json"),
        synthetic_heldout=C.binding(C.DOC / "SYNTHETIC_HELDOUT.json"),
        checkpoints={C.arm_key(arm, seed): C.binding(
            C.RAW / "runs" / f"seed{seed}" / "paired_last.pt")
            for arm in C.HEAD_ARMS for seed in C.SEEDS},
        population=C.binding(C.DEV_POS),
        registry=C.binding(C.GEOMETRY_REGISTRY),
        code=[C.binding(C.HERE / "inference.py"), C.binding(C.HERE / "evaluation.py")],
        frames=rows,
        selection_inference_replay_max_abs_px=max_replay,
        center_box_score_instance_and_missing_preserved=True,
        real_accuracy_read=False,
        GT_inputs=False,
        camera_inputs=False,
        all_image_bytes_bound=True,
    )
    C.write_frozen_json(destination, payload)
    payload = validate_prediction_payload(payload, protocol=protocol)
    C.write_frozen_json(C.DOC / "DEV_INFERENCE_COMPLETE.json",
                        inference_receipt(payload, destination))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dev(args.device)


if __name__ == "__main__":
    main()
