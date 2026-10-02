"""GT-free N3 inference for DEV319 and the separate GREEN0918_119 audit."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from . import common as C
from .adapters import build_adapter
from .model import build_n3, normalize_dimensions
from .selection import load_heads
from .train import KEYS, load_protocol


SCHEMA = "pallet_n3_completion_v3_prediction_v1"
REGISTRY = C.ROOT / "challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json"
SQUARE = C.ROOT / "_docs/experiments/pallet_green0918_dimension_audit_v1/DATASET_SNAPSHOT.json"


def nullable(value):
    array = np.asarray(value)
    if array.ndim:
        return [nullable(item) for item in array]
    if np.issubdtype(array.dtype, np.bool_):
        return bool(array)
    if np.issubdtype(array.dtype, np.integer):
        return int(array)
    return float(array) if np.isfinite(array) else None


def canonical_dimensions(object_type: str) -> np.ndarray:
    registry = C.read(REGISTRY)
    matches = [row for row in registry["objects"] if row["object_type"] == object_type]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate object type: {object_type}")
    raw = matches[0]["physical_dimensions_m"]
    result = np.asarray([raw["x"], raw["z"], raw["y"]], np.float64)
    if result.shape != (3,) or not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError(f"Invalid registered dimensions: {object_type}")
    return result


def dataset_records(dataset: str) -> tuple[list[dict], dict]:
    if dataset == "DEV319":
        manifest = C.read(C.DEV)
        items = manifest["items"]
        if len(items) != 319:
            raise RuntimeError("DEV319 membership drift")
        records = []
        for item in items:
            image = C.ROOT / item["image_path"]
            records.append({
                "id": item["frame_id"], "frame_id": item["frame_id"],
                "session_id": item["session_id"], "object_type": item["object_type"],
                "image": image, "image_key": str(image.relative_to(C.ROOT)),
                "dimensions": canonical_dimensions(item["object_type"]),
            })
        if len({row["id"] for row in records}) != 319:
            raise RuntimeError("DEV319 frame identities are not unique")
        return records, C.binding(C.DEV)
    if dataset == "GREEN0918_119":
        snapshot = C.read(SQUARE)
        if len(snapshot.get("records", [])) != 119:
            raise RuntimeError("GREEN0918_119 membership drift")
        records = []
        for item in snapshot["records"]:
            image = C.ROOT / item["image"]["path"]
            dimensions = np.asarray(item["canonical_WDH_m"], np.float64)
            if not np.array_equal(dimensions, canonical_dimensions(
                    "plastic_standard_110x110x15")):
                raise RuntimeError("GREEN0918 dimension registry drift")
            records.append({
                "id": item["id"], "frame_id": item["id"],
                "session_id": item["session"],
                "object_type": "plastic_standard_110x110x15",
                "image": image, "image_key": str(image.relative_to(C.ROOT)),
                "dimensions": dimensions,
                "expected_image_sha256": item["image"]["sha256"],
            })
        return records, C.binding(SQUARE)
    raise ValueError(dataset)


def prediction_block(base: dict, points, *, method: str, seed=None,
                     temperature=None, n3_checkpoint_sha256=None) -> dict:
    valid = np.asarray(base["valid"], bool)
    box = base.get("bbox_original")
    block = {
        "method": method, "seed": seed, "temperature": temperature,
        "detected": bool(valid[:8].any() and box is not None),
        "status": str(base.get("status", "OK" if box is not None else "NO_BOX")),
        "points": nullable(points), "valid": valid.tolist(),
        "bbox": None if box is None else nullable(box),
        "score": float(base.get("score", 0.)),
        "confidence": nullable(base.get("confidence", np.full(9, np.nan))),
        "base_checkpoint_sha256": str(base["checkpoint_sha256"]),
        "n3_checkpoint_sha256": n3_checkpoint_sha256,
    }
    return block


class Predictor:
    """Frozen image-only base plus three fixed final N3 heads."""
    def __init__(self, backbone: str, device="cuda"):
        self.protocol = load_protocol()
        self.backbone = backbone
        self.adapter = build_adapter(backbone, device=device)
        self.device = self.adapter.device
        self.heads = load_heads(backbone, self.device)
        selection_path = C.DOC / f"SELECTION_{backbone.upper()}.json"
        self.selection = C.read(selection_path)
        if (not self.selection.get("complete")
                or set(self.selection.get("temperatures", {})) != {"1", "2", "3"}
                or self.selection.get("real_selection") is not False):
            raise RuntimeError(f"Invalid fixed selection receipt: {selection_path}")
        self.normalization = C.read(C.NORMALIZATION)
        self.bank = build_n3(backbone).displacements.detach().cpu().numpy()
        signature = inspect.signature(self.adapter.network.forward).parameters
        if backbone == "resnet18" and (
                "dimension_context" in signature or "dimensions" in signature):
            raise RuntimeError("ResNet base unexpectedly accepts dimensions")

    def _batch(self, base: dict, dimensions: np.ndarray) -> dict:
        box = (np.full(4, np.nan, np.float32) if base.get("bbox_net") is None
               else np.asarray(base["bbox_net"], np.float32))
        context = normalize_dimensions(
            torch.from_numpy(dimensions[None]), self.normalization).float()
        return {
            "p3": base["features"][0][None].to(self.device, torch.float16),
            "p4": base["features"][1][None].to(self.device, torch.float16),
            "points": torch.from_numpy(np.asarray(base["points_net"], np.float32)[None]).to(self.device),
            "boxes": torch.from_numpy(box[None]).to(self.device),
            "point_valid": torch.from_numpy(np.asarray(base["valid"], bool)[None]).to(self.device),
            "input_shape": torch.tensor([base["input_shape"][-2:]], device=self.device),
            "dimension_context": context.to(self.device),
        }

    def _original_delta(self, output: dict, base: dict, raw_hw, temperature: float):
        logits = output["logits"].detach().cpu()
        probability = (logits / float(temperature)).softmax(-1).numpy()
        box = np.asarray(base.get("bbox_net"), np.float64)
        if box.shape != (4,) or not np.isfinite(box).all() or not (box[2:] > box[:2]).all():
            diagonal = 1.
        else:
            diagonal = max(float(np.linalg.norm(box[2:] - box[:2])), 1.)
        network = (probability[..., None] * self.bank[None, None]).sum(-2)
        network *= diagonal * C.LAMBDA
        affine = np.asarray(base["affine_input_to_net"], np.float64)
        linear = affine[:2, :2]
        if (affine.shape != (3, 3) or not np.isfinite(affine).all()
                or not np.allclose(linear, np.diag(np.diag(linear)), rtol=0, atol=0)
                or (np.diag(linear) <= 0).any()):
            raise RuntimeError("Only the bound anisotropic axis-aligned affine is supported")
        original = np.linalg.solve(linear, network.reshape(-1, 2).T).T.reshape(network.shape)
        cap = C.CAP_FRACTION * math.hypot(*raw_hw)
        norm = np.linalg.norm(original, axis=-1)
        original *= np.minimum(1., cap / np.maximum(norm, 1e-12))[..., None]
        support = output["point_support"].detach().cpu().numpy()
        return np.where(support[..., None], original, 0.)[0], support[0]

    @torch.no_grad()
    def predict(self, image: np.ndarray, dimensions: np.ndarray) -> dict:
        # The base API receives RGB only. Registered dimensions first enter here,
        # in the N3 batch constructed after frozen base inference.
        base = self.adapter.infer(image, source_pre_padded=False, return_features=True)
        batch = self._batch(base, dimensions)
        methods = {"base": prediction_block(
            base, np.asarray(base["points_original"], np.float64), method="base")}
        original = np.asarray(base["points_original"], np.float64)
        valid = np.asarray(base["valid"], bool)
        for seed, head in self.heads.items():
            output = head(*(batch[key] for key in KEYS),
                          dimension_context=batch["dimension_context"], lam=0.)
            if not np.isfinite(output["logits"].detach().cpu().numpy()).all():
                raise FloatingPointError((self.backbone, seed))
            temperature = float(self.selection["temperatures"][str(seed)])
            delta, support = self._original_delta(output, base, image.shape[:2], temperature)
            points = original.copy()
            usable = valid[:8] & support
            points[:8][usable] += delta[usable]
            if not np.array_equal(points[8], original[8], equal_nan=True):
                raise AssertionError("N3 changed center8")
            if not np.array_equal(np.isfinite(points).all(-1), valid):
                raise AssertionError("N3 changed the missing-point mask")
            receipt = C.read(C.DOC / f"TRAIN_{self.backbone.upper()}_SEED{seed}.json")
            methods[f"N3_seed{seed}"] = prediction_block(
                base, points, method="N3", seed=seed, temperature=temperature,
                n3_checkpoint_sha256=receipt["checkpoint"]["sha256"])
        base.pop("features", None)
        return {
            "methods": methods,
            "affine_input_to_net": nullable(base["affine_input_to_net"]),
            "input_shape": list(base["input_shape"]),
        }


def validate_payload(payload: dict, backbone: str, dataset: str) -> dict:
    records, population = dataset_records(dataset)
    if (payload.get("schema") != SCHEMA or payload.get("complete") is not True
            or payload.get("backbone") != backbone or payload.get("dataset") != dataset
            or payload.get("population") != population
            or payload.get("methods") != ["base", "N3_seed1", "N3_seed2", "N3_seed3"]
            or payload.get("GT_inputs") is not False
            or payload.get("base_receives_dimensions") is not False
            or payload.get("center_box_score_order_and_missing_preserved") is not True
            or len(payload.get("frames", [])) != len(records)):
        raise RuntimeError("N3 prediction payload contract mismatch")
    for entry in (payload["protocol"], payload["selection"], payload["training"],
                  payload["registry"], payload["code"]):
        C.verify(entry)
    for expected, frame in zip(records, payload["frames"]):
        if (frame.get("id") != expected["id"]
                or frame.get("image_key") != expected["image_key"]
                or frame.get("object_type") != expected["object_type"]
                or set(frame.get("predictions", {})) != set(payload["methods"])):
            raise RuntimeError("Prediction frame identity/order drift")
        raw = expected["image"].read_bytes()
        if frame.get("image_sha256") != hashlib.sha256(raw).hexdigest():
            raise RuntimeError("Prediction image-byte binding drift")
        base = frame["predictions"]["base"]
        for method in payload["methods"][1:]:
            value = frame["predictions"][method]
            for field in ("detected", "status", "valid", "bbox", "score",
                          "base_checkpoint_sha256"):
                if value[field] != base[field]:
                    raise RuntimeError(f"N3 failed preservation contract: {field}")
            if value["points"][8] != base["points"][8]:
                raise RuntimeError("N3 failed center8 preservation")
    return payload


def run(backbone: str, dataset: str, device="cuda") -> dict:
    load_protocol()
    records, population = dataset_records(dataset)
    destination = C.RAW / "predictions" / f"{backbone}_{dataset}.json"
    if destination.exists():
        return validate_payload(C.read(destination), backbone, dataset)
    predictor = Predictor(backbone, device)
    frames = []
    for index, record in enumerate(records):
        raw = record["image"].read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if record.get("expected_image_sha256") not in (None, digest):
            raise RuntimeError(f"Image hash drift: {record['id']}")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable image: {record['image']}")
        prediction = predictor.predict(image, record["dimensions"])
        frames.append({
            "id": record["id"], "frame_id": record["frame_id"],
            "session_id": record["session_id"], "image_key": record["image_key"],
            "image_sha256": digest, "raw_shape_hw": list(image.shape[:2]),
            "object_type": record["object_type"],
            "dimensions_wdh_m": record["dimensions"].tolist(),
            "predictions": prediction["methods"],
            "affine_input_to_net": prediction["affine_input_to_net"],
            "input_shape": prediction["input_shape"],
        })
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(records):
            print("N3_INFERENCE", backbone, dataset, index + 1, len(records), flush=True)
    payload = {
        "schema": SCHEMA, "complete": True, "backbone": backbone,
        "dataset": dataset, "coordinate_system": "original_unpadded_pixels",
        "methods": ["base", "N3_seed1", "N3_seed2", "N3_seed3"],
        "frames": frames, "population": population,
        "protocol": C.binding(C.DOC / "PROTOCOL.json"),
        "selection": C.binding(C.DOC / f"SELECTION_{backbone.upper()}.json"),
        "training": C.binding(C.DOC / f"TRAINING_{backbone.upper()}_COMPLETE.json"),
        "registry": C.binding(REGISTRY), "code": C.binding(Path(__file__)),
        "GT_inputs": False, "camera_inputs": False,
        "base_receives_dimensions": False, "n3_receives_dimensions": True,
        "center_box_score_order_and_missing_preserved": True,
        "real_accuracy_read": False, "all_image_bytes_bound": True,
    }
    C.write(destination, payload, freeze=True)
    payload = validate_payload(payload, backbone, dataset)
    C.write(C.DOC / f"INFERENCE_{backbone.upper()}_{dataset}_COMPLETE.json", {
        "complete": True, "backbone": backbone, "dataset": dataset,
        "frames": len(frames), "predictions": C.binding(destination),
        "GT_inputs": False, "base_receives_dimensions": False,
    }, freeze=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", choices=tuple(C.CONFIGS))
    parser.add_argument("dataset", choices=("DEV319", "GREEN0918_119"))
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    output = run(arguments.backbone, arguments.dataset, arguments.device)
    print(json.dumps({
        "complete": output["complete"], "backbone": output["backbone"],
        "dataset": output["dataset"], "frames": len(output["frames"]),
    }, indent=2))


if __name__ == "__main__":
    main()
