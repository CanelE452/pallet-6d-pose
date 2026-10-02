"""Selected deployment wrappers for YOLO, DOPE and dimension-aware ResNet-18.

Only fixed, completed checkpoints are loaded.  The ResNet wrapper loads seed 1
for the four declared rows rather than constructing the twelve-head evaluation
bank.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from . import common as C


YOLO_CODE = C.ROOT / "scripts/research/pallet_final_ml_contribution_test_v1"
YOLO_R0 = (C.ROOT / "challenge/yolo_pose_one_model/spatial_concat_scratch/runs"
           / "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")
DOPE_CODE = C.ROOT / "scripts/research/pallet_dope_refiner_20261001_v1"


@contextmanager
def isolated_legacy_import(directory: Path, names: tuple[str, ...]):
    """Temporarily expose old absolute imports without polluting other stacks."""
    saved = {name: sys.modules.get(name) for name in names}
    old_path = list(sys.path)
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        sys.path[:] = old_path
        for name in names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


def _nullable_points(value):
    if value is None:
        return None
    points = np.asarray(value, np.float64)
    if points.shape != (9, 2) or np.isinf(points).any():
        raise ValueError("Expected nine finite-or-NaN xy points")
    return points


class YoloBaseline:
    """Canonical R0 wrapper accepting a decoded native uint8 BGR frame."""
    def __init__(self, device="cuda"):
        from ultralytics import YOLO
        self.device = torch.device(device)
        self.model = YOLO(str(YOLO_R0), task="pose")
        self.model.model.requires_grad_(False).eval()

    @torch.no_grad()
    def predict(self, image):
        padded = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
        result = self.model.predict(padded, conf=.001, imgsz=640, device=str(self.device),
                                    verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        scores = result.boxes.conf.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy() - 100
        points = None if result.keypoints is None else result.keypoints.xy.detach().cpu().numpy() - 100
        return [dict(score=float(scores[index]), box_xyxy=boxes[index].astype(np.float64),
                     keypoints_xy=None if points is None else points[index].astype(np.float64))
                for index in range(len(scores))]


class YoloSelected:
    def __init__(self, device="cuda"):
        names = ("common", "generic_point_refiner", "point_inference")
        with isolated_legacy_import(YOLO_CODE, names):
            module = importlib.import_module("point_inference")
            self.predictor = module.PointInference(1, device=device)
        self.device = torch.device(device)

    def predict(self, image):
        return self.predictor.predict(image)

    def close(self):
        self.predictor.close()


class DopeSelected:
    """Frozen DOPE with only the source-selected P1 head resident."""
    def __init__(self, device="cuda"):
        names = ("common", "data", "refiner", "train", "selection",
                 "dope_adapter", "inference")
        with isolated_legacy_import(DOPE_CODE, names):
            common = importlib.import_module("common")
            refiner = importlib.import_module("refiner")
            train = importlib.import_module("train")
            adapter_module = importlib.import_module("dope_adapter")
            common.verify_lock()
            selection = common.read(common.DOC / "SELECTION.json")
            if selection.get("complete") is not True or selection.get("real_selection") is not False:
                raise ValueError("DOPE synthetic-only selection is incomplete")
            receipt = common.read(common.DOC / "TRAIN_SEED1.json")
            checkpoint_path = common.ROOT / receipt["checkpoint"]["path"]
            if common.sha(checkpoint_path) != receipt["checkpoint"]["sha256"]:
                raise ValueError("DOPE P1 checkpoint binding changed")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (checkpoint.get("complete") is not True or checkpoint.get("step") != 6000
                    or checkpoint.get("seed") != 1 or "P" not in checkpoint.get("models", {})):
                raise ValueError("DOPE seed-1 checkpoint is incomplete")
            head = train.CLASSES["P"](**common.CONFIG).to(device).eval()
            head.load_state_dict(checkpoint["models"]["P"], strict=True)
            head.requires_grad_(False)
            self.adapter = adapter_module.FrozenDopeAdapter(device=device)
            self.keys = tuple(train.KEYS)
            self.config = dict(common.CONFIG)
            self.protocol_sha256 = common.sha(common.DOC / "PROTOCOL.json")
            self.checkpoint_sha256 = receipt["checkpoint"]["sha256"]
            self.refiner_source = str(Path(refiner.__file__).resolve())
        self.device = torch.device(device)
        self.head = head
        self.selection = selection
        self.bank = head.displacements.detach().cpu().numpy()

    def _delta(self, value, support, base, raw_hw):
        rule = self.selection["rules"]["P"]
        temperature = float(self.selection["temperatures"]["P1"])
        box = (np.full(4, np.nan, np.float32) if base["bbox_net"] is None
               else np.asarray(base["bbox_net"], np.float32))
        boxes = box.reshape(1, 4)
        valid = np.isfinite(boxes).all(-1) & (boxes[:, 2:] > boxes[:, :2]).all(-1)
        safe = np.where(valid[:, None], boxes, np.array([0., 0., 1., 1.]))
        diagonal = np.maximum(np.linalg.norm(safe[:, 2:] - safe[:, :2], axis=-1), 1.)
        probability = (torch.from_numpy(value) / temperature).softmax(-1).numpy()
        delta = ((probability[..., None] * self.bank[None, None]).sum(-2)
                 * diagonal[:, None, None])
        scale = np.diag(np.asarray(base["affine_input_to_net"], np.float64))[:2]
        delta = delta * float(rule["lam"]) / scale[None, None]
        fraction = rule["max_move_image_diagonal_fraction"]
        if fraction is not None:
            cap = math.hypot(*raw_hw) * float(fraction)
            norm = np.maximum(np.linalg.norm(delta, axis=-1), 1e-12)
            delta *= np.minimum(1., cap / norm)[..., None]
        return np.where(np.asarray(support, bool)[..., None], delta, 0.)[0]

    @torch.no_grad()
    def predict(self, image, refined: bool):
        base = self.adapter.infer(image, return_features=refined)
        results = {}
        if refined:
            box = (np.full(4, np.nan, np.float32) if base["bbox_net"] is None
                   else np.asarray(base["bbox_net"], np.float32))
            batch = dict(
                p3=base["features"][0][None].to(torch.float16),
                p4=base["features"][1][None].to(torch.float16),
                points=torch.from_numpy(np.asarray(base["points_net"], np.float32)[None]).to(self.device),
                boxes=torch.from_numpy(box[None]).to(self.device),
                point_valid=torch.from_numpy(np.asarray(base["valid"], bool)[None]).to(self.device),
                input_shape=torch.tensor([base["input_shape"][-2:]], device=self.device),
            )
            output = self.head(*(batch[key] for key in self.keys), lam=0)
            value = output["logits"].detach().cpu().numpy()
            support = output["point_support"].detach().cpu().numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError("Nonfinite DOPE P1 logits")
            delta = self._delta(value, support, base, image.shape[:2])
            points = np.asarray(base["points_original"], np.float64).copy()
            if base["bbox_original"] is not None:
                usable = np.asarray(base["valid"][:8], bool) & support[0]
                points[:8][usable] += delta[usable]
            if not np.array_equal(points[8], base["points_original"][8], equal_nan=True):
                raise AssertionError("DOPE center point changed")
            results["P1"] = points
            base.pop("features")
        return base, results


class ResnetSelected:
    """FULL plus four fixed seed-1 heads, never the twelve-head bank."""
    HEADS = ("P0", "D0", "P5_CONSTANT", "P5")

    def __init__(self, device="cuda"):
        from scripts.research.pallet_resnet18_dim_refiner_20261002_v1 import common as R
        from scripts.research.pallet_resnet18_dim_refiner_20261002_v1.full_adapter import FrozenFullAdapter
        from scripts.research.pallet_resnet18_dim_refiner_20261002_v1.refiner import model_for_arm, forward_arm
        from scripts.research.pallet_resnet18_dim_refiner_20261002_v1.selection import refined_delta, validate_selection_payload

        protocol = R.verify_protocol()
        receipt = R.read_json(R.DOC / "TRAIN_SEED1.json")
        checkpoint_path = R.verify_binding(receipt["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (checkpoint.get("complete") is not True or checkpoint.get("step") != R.STEPS
                or checkpoint.get("seed") != 1
                or set(checkpoint.get("models", {})) != set(R.HEAD_ARMS)):
            raise ValueError("ResNet seed-1 paired checkpoint is incomplete")
        selection = validate_selection_payload(R.read_json(R.DOC / "SELECTION.json"),
                                               protocol=protocol)
        self.device = torch.device(device)
        self.adapter = FrozenFullAdapter(device=device)
        self.models = {}
        for arm in self.HEADS:
            model = model_for_arm(arm).to(self.device).eval()
            model.load_state_dict(checkpoint["models"][arm], strict=True)
            model.requires_grad_(False)
            self.models[arm] = model
        self.selection = selection
        self.forward_arm = forward_arm
        self.refined_delta = refined_delta
        self.checkpoint_sha256 = receipt["checkpoint"]["sha256"]
        self.baseline_sha256 = protocol["baseline"]["sha256"]

    @torch.no_grad()
    def predict(self, image, dimensions, arm: str | None):
        if arm is not None and arm not in self.models:
            raise ValueError(f"Unloaded ResNet head requested: {arm}")
        base = self.adapter.infer(image, dimensions, return_features=arm is not None)
        results = {}
        if arm is not None:
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
            output = self.forward_arm(self.models[arm], batch, arm, lam=0.)
            field = "delta_normalized" if arm == "D0" else "logits"
            value = output[field].detach().cpu().numpy()
            support = output["point_support"].detach().cpu().numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite ResNet correction output: {arm}")
            key = f"{arm}_S1"
            rule = self.selection["rules"][arm]
            arrays = dict(
                boxes=box[None],
                scale_xy=np.diag(np.asarray(base["affine_input_to_net"], np.float64))[None, :2],
                raw_diagonal=np.asarray([math.hypot(*image.shape[:2])], np.float64),
            )
            delta = self.refined_delta(dict(value=value, support=support), arm,
                self.selection["temperatures"][key], arrays, rule["lam"],
                rule["max_move_image_diagonal_fraction"])[0]
            points = np.asarray(base["points_original"], np.float64).copy()
            usable = np.asarray(base["valid"][:8], bool) & support[0]
            points[:8][usable] += delta[usable]
            if not np.array_equal(points[8], base["points_original"][8], equal_nan=True):
                raise AssertionError("ResNet center point changed")
            if not np.array_equal(np.isfinite(points).all(-1), base["valid"]):
                raise AssertionError("ResNet missing mask changed")
            results[key] = points
            base.pop("features")
        return base, results


class UnifiedModels:
    BACKEND_BY_FAMILY = {
        "YOLO": dict(matmul_allow_tf32=False, cudnn_allow_tf32=True,
                     cudnn_benchmark=False, cudnn_deterministic=False),
        "DOPE": dict(matmul_allow_tf32=False, cudnn_allow_tf32=False,
                     cudnn_benchmark=False, cudnn_deterministic=False),
        "RESNET": dict(matmul_allow_tf32=False, cudnn_allow_tf32=False,
                       cudnn_benchmark=False, cudnn_deterministic=False),
    }

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Production benchmark requires the declared CUDA desktop")
        self.yolo_base = YoloBaseline(device)
        self.yolo_p1 = YoloSelected(device)
        self.dope = DopeSelected(device)
        self.resnet = ResnetSelected(device)

    @classmethod
    def backend_for(cls, arm: str) -> dict:
        family = arm.split("_", 1)[0]
        if family not in cls.BACKEND_BY_FAMILY:
            raise ValueError(f"Unknown benchmark arm: {arm}")
        return dict(cls.BACKEND_BY_FAMILY[family])

    def configure_backend(self, arm: str) -> dict:
        """Restore the source experiment's frozen CUDA numerical settings."""
        value = self.backend_for(arm)
        torch.backends.cuda.matmul.allow_tf32 = value["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = value["cudnn_allow_tf32"]
        torch.backends.cudnn.benchmark = value["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = value["cudnn_deterministic"]
        return value

    def _check_backend(self, arm: str) -> None:
        expected = self.backend_for(arm)
        actual = dict(
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
        )
        if actual != expected:
            raise RuntimeError(f"CUDA backend contract drift for {arm}: {actual} != {expected}")

    @torch.no_grad()
    def predict(self, arm: str, image: np.ndarray, dimensions: np.ndarray) -> dict:
        self._check_backend(arm)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Decoded uint8 BGR input is required")
        if arm == "YOLO_R0":
            candidates = self.yolo_base.predict(image)
            selected = int(np.argmax([row["score"] for row in candidates])) if candidates else None
            points = None if selected is None else _nullable_points(candidates[selected]["keypoints_xy"])
            return dict(points=points, has_detection=selected is not None,
                        native=dict(candidates=candidates, selected_index=selected))
        if arm == "YOLO_P1":
            prediction = self.yolo_p1.predict(image)
            selected = prediction["selected_index"]
            points = None if selected is None else _nullable_points(
                prediction["candidates"][selected]["keypoints_xy"])
            return dict(points=points, has_detection=selected is not None, native=prediction)
        if arm in ("DOPE_BASE", "DOPE_P1"):
            base, refined = self.dope.predict(image, refined=arm == "DOPE_P1")
            points = (np.asarray(base["points_original"], np.float64) if arm == "DOPE_BASE"
                      else np.asarray(refined["P1"], np.float64))
            return dict(points=points, has_detection=base["bbox_original"] is not None,
                        native=dict(base=base, refined=refined))
        resnet_arm = {
            "RESNET_FULL": None,
            "RESNET_P0_S1": "P0",
            "RESNET_D0_S1": "D0",
            "RESNET_P5_CONSTANT_S1": "P5_CONSTANT",
            "RESNET_P5_S1": "P5",
        }.get(arm, "UNKNOWN")
        if resnet_arm == "UNKNOWN":
            raise ValueError(f"Unknown benchmark arm: {arm}")
        base, refined = self.resnet.predict(image, dimensions, resnet_arm)
        key = None if resnet_arm is None else f"{resnet_arm}_S1"
        points = (np.asarray(base["points_original"], np.float64) if key is None
                  else np.asarray(refined[key], np.float64))
        return dict(points=points, has_detection=base["bbox_original"] is not None,
                    native=dict(base=base, refined=refined))

    def identity(self) -> dict:
        return dict(
            resnet_loaded_heads=[f"{arm}_S1" for arm in self.resnet.HEADS],
            resnet_loaded_head_count=len(self.resnet.models),
            resnet_evaluation_bank_head_count=12,
            resnet_checkpoint_sha256=self.resnet.checkpoint_sha256,
            resnet_baseline_sha256=self.resnet.baseline_sha256,
            dope_loaded_heads=["P1"],
            dope_checkpoint_sha256=self.dope.checkpoint_sha256,
            yolo_weights=C.binding(YOLO_R0),
            cuda_backend_by_family=self.BACKEND_BY_FAMILY,
        )

    def close(self):
        self.yolo_p1.close()
        del self.yolo_base, self.yolo_p1, self.dope, self.resnet
