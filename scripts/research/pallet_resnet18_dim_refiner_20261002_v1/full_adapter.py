"""Strict frozen adapter for the completed dimension-conditioned FULL baseline.

The adapter reproduces the bound DSNT spatial-softmax decoder.  It never uses
camera intrinsics, ground-truth keypoints, or pose-selected dimension swaps.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional

from . import common as C
from .dimensions import load_normalization, normalized_context
from .protocol import validate_full_completion


FEATURE_CHANNELS = (128, 256)
FEATURE_STRIDES = (8, 16)
FEATURE_SHAPES = ((128, 48, 64), (256, 24, 32))
HEATMAP_HW = (96, 128)
STRIDE = 4


def _module(key: str, path: Path):
    name = f"pallet_resnet18_dim_refiner_{key}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def input_module():
    return _module("bound_input", C.DSNT_HERE / "input_data.py")


def model_module():
    return _module("bound_model", C.DSNT_HERE / "model.py")


def spatial_coordinates(logits: torch.Tensor):
    """Byte-for-byte mathematical equivalent of the bound DSNT decoder."""
    if logits.ndim != 4 or tuple(logits.shape[1:]) != (9, *HEATMAP_HW):
        raise ValueError("Expected Bx9x96x128 logits")
    batch, channels, height, width = logits.shape
    flat = logits.reshape(batch, channels, -1)
    log_probability = functional.log_softmax(flat, dim=-1)
    probability = log_probability.exp()
    xs = torch.linspace(-1., 1., width, device=logits.device, dtype=logits.dtype)
    ys = torch.linspace(-1., 1., height, device=logits.device, dtype=logits.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    return probability @ grid, log_probability, probability


def decode_logits(logits: torch.Tensor, prepared: list[dict]) -> list[dict]:
    if len(prepared) != len(logits) or not torch.isfinite(logits).all():
        raise ValueError("Finite logits and one preparation record per image are required")
    coordinates, _, probability = spatial_coordinates(logits)
    coordinates = coordinates.detach().cpu().numpy()
    peaks = probability.detach().amax(-1).cpu().numpy()
    grid = np.empty_like(coordinates)
    grid[..., 0] = (coordinates[..., 0] + 1.) * .5 * (HEATMAP_HW[1] - 1)
    grid[..., 1] = (coordinates[..., 1] + 1.) * .5 * (HEATMAP_HW[0] - 1)
    I = input_module()
    outputs = []
    for index, meta in enumerate(prepared):
        points_net = (grid[index] * STRIDE).astype(np.float32)
        points_original = I.transform_points(points_net, meta["affine_net_to_input"])
        valid = np.ones(9, dtype=bool)
        box_net = I.predicted_box(points_net, valid)
        box_original = I.predicted_box(points_original, valid)
        if box_net is None or box_original is None:
            raise ValueError("Finite DSNT corner expectations must define a hull")
        outputs.append(dict(
            points_net=points_net,
            points_original=points_original,
            valid=valid,
            confidence=peaks[index].astype(np.float32),
            score=float(peaks[index].mean()),
            bbox_net=np.asarray(box_net, np.float32),
            bbox_original=np.asarray(box_original, np.float64),
            grid_points=grid[index].astype(np.float32),
            affine_input_to_net=np.asarray(meta["affine_input_to_net"], np.float64),
            affine_net_to_input=np.asarray(meta["affine_net_to_input"], np.float64),
            input_shape=list(meta["input_shape"]),
            original_hw=list(meta["input_hw"]),
        ))
    return outputs


def recipe() -> dict:
    return dict(
        schema="frozen_resnet18_full_dsnt_adapter_v1",
        baseline_arm="FULL",
        decoder="spatial softmax expectation on linspace[-1,1], then stride4",
        input=input_module().contract(),
        feature_taps=["layer2", "layer3"],
        feature_shapes=[list(value) for value in FEATURE_SHAPES],
        feature_dtype="float32; FP16 roundtrip only at correction-head boundary",
        canonical_dimension_input="W,D,H -> bound TRAIN55980 normalized 5D",
        checkpoint_fallback=False,
        GT_inputs=False,
        camera_inputs=False,
        frozen=True,
        TF32=False,
        AMP=False,
    )


class FrozenFullAdapter:
    def __init__(self, device="cpu", checkpoint: Path | str | None = None):
        baseline = validate_full_completion()
        checkpoint_path = C.verify_binding(baseline["checkpoint"])
        if checkpoint is not None and Path(checkpoint).resolve() != checkpoint_path:
            raise ValueError("Only the completed FULL final checkpoint is accepted")
        full_protocol = C.read_json(C.verify_binding(baseline["protocol"]))
        bound = {Path(value["path"]).name: value for value in full_protocol["bindings"]}
        for filename in ("model.py", "input_data.py"):
            expected = bound.get(filename)
            if expected is None or C.verify_binding(expected) != (C.DSNT_HERE / filename).resolve():
                raise ValueError(f"FULL protocol does not bind current {filename}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (payload.get("arm"), payload.get("epoch"), payload.get("step")) != ("FULL", 10, 34990):
            raise ValueError("Unexpected FULL checkpoint identity")
        if C.verify_binding(payload["protocol"]) != C.FULL_PROTOCOL.resolve():
            raise ValueError("Checkpoint protocol binding mismatch")
        network = model_module().SimpleBaselineResNet18(num_keypoints=9, pretrained_state=None)
        expected = network.state_dict()
        state = payload.get("model_state_dict", {})
        if set(state) != set(expected):
            raise ValueError("FULL state dictionary is incomplete")
        for key, value in state.items():
            if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                    or value.dtype != expected[key].dtype or value.is_meta
                    or (value.is_floating_point() and not torch.isfinite(value).all())):
                raise ValueError(f"Invalid FULL state tensor: {key}")
        network.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.network = network.requires_grad_(False).eval().to(self.device)
        self.normalization = load_normalization()
        self.checkpoint = baseline["checkpoint"]
        self.protocol = baseline["protocol"]
        self.completion = baseline["completion"]
        self.trained_full_loaded = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def prepare(image, source_pre_padded=False):
        return input_module().prepare(image, source_pre_padded=source_pre_padded)

    def _check_runtime(self, tensor):
        if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (3, 384, 512) or tensor.dtype != torch.float32:
            raise ValueError("Expected normalized float32 Bx3x384x512")
        if any(module.training for module in self.network.modules()):
            raise RuntimeError("Frozen FULL network entered training mode")
        if any(parameter.requires_grad for parameter in self.network.parameters()):
            raise RuntimeError("Frozen FULL network has trainable parameters")

    @staticmethod
    def _check_features(features, batch):
        if len(features) != 2:
            raise ValueError("Expected two frozen feature taps")
        for value, shape in zip(features, FEATURE_SHAPES):
            if tuple(value.shape) != (batch, *shape) or value.dtype != torch.float32:
                raise ValueError("Frozen FULL feature contract mismatch")
            if value.requires_grad or torch.is_inference(value):
                raise ValueError("Head training requires ordinary detached tensors")

    @torch.no_grad()
    def features_only(self, batch_tensor):
        self._check_runtime(batch_tensor)
        with torch.inference_mode(False), torch.no_grad():
            value = batch_tensor.to(self.device)
            if torch.is_inference(value):
                value = value.clone()
            features = tuple(item.detach() for item in self.network.features_only(value))
        self._check_features(features, len(batch_tensor))
        return features

    @torch.no_grad()
    def infer_batch(self, images, dimensions, *, source_pre_padded=False, return_features=True):
        if not images:
            raise ValueError("At least one image is required")
        if len(images) != len(dimensions):
            raise ValueError("Each image requires canonical W,D,H")
        prepared = [self.prepare(image, source_pre_padded) for image in images]
        context = normalized_context(dimensions, self.normalization)
        with torch.inference_mode(False), torch.no_grad():
            tensor = torch.stack([item["tensor"] for item in prepared]).to(self.device)
            self._check_runtime(tensor)
            z = torch.from_numpy(context).to(self.device)
            result = self.network(tensor, z, return_features=return_features)
            if return_features:
                if set(result) != {"heatmaps", "features"}:
                    raise ValueError("FULL return_features schema drift")
                logits = result["heatmaps"]
                features = tuple(value.detach() for value in result["features"])
                self._check_features(features, len(images))
            else:
                logits, features = result, ()
            decoded = decode_logits(logits, prepared)
        for index, output in enumerate(decoded):
            output.update({key: value for key, value in prepared[index].items() if key != "tensor"})
            output.update(
                dimension_context=context[index].copy(),
                checkpoint_sha256=self.checkpoint["sha256"],
                baseline_protocol_sha256=self.protocol["sha256"],
                training_receipt_sha256=self.completion["sha256"],
                belief_shape=[9, *HEATMAP_HW],
                status="OK",
            )
            if return_features:
                output["features"] = tuple(value[index] for value in features)
        return decoded

    def infer(self, image, dimensions, *, source_pre_padded=False, return_features=True):
        return self.infer_batch([image], [dimensions], source_pre_padded=source_pre_padded,
                                return_features=return_features)[0]
