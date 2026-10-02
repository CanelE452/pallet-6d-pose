"""Frozen RGB adapter derived exactly from the ResNet-18 CONSTANT arm.

The completed ten-epoch CONSTANT checkpoint contains a decoder FiLM branch,
but its training context is fixed to ``z=zeros(B, 5)``. This module folds that
constant affine operation into the final 1x1 heatmap convolution, removes every
``dimension_film`` tensor, and strict-loads the result into the bound RGB-only
SimpleBaseline implementation. The resulting base forward accepts images only;
physical dimensions are unavailable to it and remain inputs to N3.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from . import common as C


DSNT_HERE = C.ROOT / "scripts/research/pallet_resnet18_dimension_20261002_v1"
DSNT_DOC = C.ROOT / "_docs/experiments/pallet_resnet18_dimension_20261002_v1"
MODEL_SOURCE = DSNT_HERE / "model.py"
INPUT_SOURCE = DSNT_HERE / "input_data.py"
RGB_MODEL_SOURCE = C.ROOT / "scripts/research/pallet_resnet18_refiner_20261001_v1/model.py"
PROTOCOL = DSNT_DOC / "DSNT_FULL_TRAIN_PROTOCOL.json"
COMPLETION = DSNT_DOC / "DSNT_FULL_CONSTANT_COMPLETE.json"
CHECKPOINT = C.RESNET_CONSTANT

CHECKPOINT_SHA256 = "a94e55f028b704057c77fd045d96dd2dbb2ae6be2a49e945b23c821a30d8e1c4"
PROTOCOL_SHA256 = "f24f98082bd8f226968dd01bcf7a7deade339d0bb060601ae546d41c48b8038a"
COMPLETION_SHA256 = "3ee47b7e284c40792df92ccab9b2535dc0de387f3bd2b9ef4400701e9fabbc9b"
MODEL_SHA256 = "d96ecfea682e4a85880fab86c585f598ee1da5c07580e886ce57bbf838addfe1"
INPUT_SHA256 = "d7e8a5220482d8ef054af63585902299f10d9c99c0d6eb927391da73172a967f"
RGB_MODEL_SHA256 = "8c25f00b02af7546e199129484ce6a195716020f91ae2e00745becc47c50fcd0"
FEATURE_CHANNELS = (128, 256)
FEATURE_STRIDES = (8, 16)
FEATURE_SHAPES = ((128, 48, 64), (256, 24, 32))
INPUT_CHW = (3, 384, 512)
HEATMAP_SHAPE = (9, 96, 128)
FILM_KEYS = {
    "dimension_film.0.weight", "dimension_film.0.bias",
    "dimension_film.2.weight", "dimension_film.2.bias",
}


def input_module():
    return C.local_module("resnet_constant_input", INPUT_SOURCE)


def conditioned_model_module():
    return C.local_module("resnet_constant_conditioned_model", MODEL_SOURCE)


def rgb_model_module():
    return C.local_module("resnet_constant_rgb_model", RGB_MODEL_SOURCE)


def decoder_module():
    # Import as a package module because the audited decoder has relative
    # imports of its own contract files.
    return importlib.import_module(
        "scripts.research.pallet_resnet18_dim_refiner_20261002_v1.full_adapter")


def _same_binding(left: dict, right: dict) -> bool:
    return (left.get("path") == right.get("path")
            and left.get("sha256") == right.get("sha256")
            and left.get("bytes") == right.get("bytes"))


def load_constant_contract(*, load_checkpoint: bool = True):
    """Verify the receipt, every protocol binding, and checkpoint identity."""
    if C.sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("CONSTANT protocol hash changed")
    if C.sha256(COMPLETION) != COMPLETION_SHA256:
        raise RuntimeError("CONSTANT completion receipt hash changed")
    if C.sha256(MODEL_SOURCE) != MODEL_SHA256 or C.sha256(INPUT_SOURCE) != INPUT_SHA256:
        raise RuntimeError("Bound CONSTANT model/input source changed")
    if C.sha256(RGB_MODEL_SOURCE) != RGB_MODEL_SHA256:
        raise RuntimeError("Bound pure-RGB model source changed")

    receipt = C.read(COMPLETION)
    if (receipt.get("complete") is not True
            or receipt.get("arm") != "CONSTANT"
            or receipt.get("epochs") != 10
            or receipt.get("updates") != 34990
            or receipt.get("real_training_images") != 0
            or receipt.get("final_epoch_fixed") is not True):
        raise RuntimeError("CONSTANT completion identity is invalid")
    if not _same_binding(receipt.get("final_checkpoint", {}), {
            "path": str(CHECKPOINT.relative_to(C.ROOT)),
            "sha256": CHECKPOINT_SHA256,
            "bytes": 61663050}):
        raise RuntimeError("CONSTANT checkpoint receipt changed")
    if not _same_binding(receipt.get("protocol", {}), {
            "path": str(PROTOCOL.relative_to(C.ROOT)),
            "sha256": PROTOCOL_SHA256,
            "bytes": 4364}):
        raise RuntimeError("CONSTANT protocol receipt changed")

    checkpoint_path = C.verify(receipt["final_checkpoint"])
    protocol_path = C.verify(receipt["protocol"])
    protocol = C.read(protocol_path)
    if (protocol.get("schema") != "resnet18_dimension_dsnt_full_source_train_v1"
            or protocol.get("epochs") != 10
            or protocol.get("batch") != 16
            or protocol.get("total_updates_per_arm") != 34990
            or protocol.get("arm_context_masks", {}).get("CONSTANT") != [0, 0, 0, 0, 0]):
        raise RuntimeError("CONSTANT protocol no longer fixes a zero context")
    for binding in protocol.get("bindings", []):
        C.verify(binding)
    by_name = {Path(binding["path"]).name: binding for binding in protocol["bindings"]}
    if (by_name.get("model.py", {}).get("sha256") != MODEL_SHA256
            or by_name.get("input_data.py", {}).get("sha256") != INPUT_SHA256):
        raise RuntimeError("Protocol does not bind the expected model/input sources")

    payload = None
    identity = None
    if load_checkpoint:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        identity = {key: payload.get(key) for key in ("arm", "epoch", "step")}
        if identity != {"arm": "CONSTANT", "epoch": 10, "step": 34990}:
            raise RuntimeError("CONSTANT checkpoint header changed")
        if not _same_binding(payload.get("protocol", {}), receipt["protocol"]):
            raise RuntimeError("CONSTANT checkpoint protocol binding changed")
        state = payload.get("model_state_dict")
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError("CONSTANT checkpoint has no model state")
    audit = {
        "checkpoint": receipt["final_checkpoint"],
        "protocol": receipt["protocol"],
        "completion": C.binding(COMPLETION),
        "checkpoint_identity": identity,
        "constant_context_mask": [0, 0, 0, 0, 0],
        "runtime_dimension_context": False,
        "source_training_epochs": 10,
        "source_training_images": "synthetic only",
    }
    return audit, payload


def _finite_tensor(state: Mapping[str, torch.Tensor], key: str, shape: tuple[int, ...]):
    value = state.get(key)
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
            or not value.is_floating_point() or value.is_meta
            or not torch.isfinite(value).all()):
        raise ValueError(f"Invalid CONSTANT state tensor: {key}")
    return value


def zero_context_film(state: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the checkpoint FiLM MLP at the exact constant z=zeros(5)."""
    first_weight = _finite_tensor(state, "dimension_film.0.weight", (32, 5))
    first_bias = _finite_tensor(state, "dimension_film.0.bias", (32,))
    final_weight = _finite_tensor(state, "dimension_film.2.weight", (512, 32))
    final_bias = _finite_tensor(state, "dimension_film.2.bias", (512,))
    if len({value.dtype for value in (first_weight, first_bias, final_weight, final_bias)}) != 1:
        raise ValueError("FiLM tensors must use one dtype")
    zero = torch.zeros((1, 5), dtype=first_weight.dtype, device=first_weight.device)
    hidden = functional.silu(functional.linear(zero, first_weight, first_bias))
    gamma, beta = functional.linear(hidden, final_weight, final_bias).squeeze(0).chunk(2)
    return gamma, beta


def fold_constant_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return an RGB state mathematically equal to CONSTANT at ``z=0``.

    The input mapping is never mutated. FiLM is folded as
    ``W'=W*(1+gamma)`` and ``b'=b+sum_c(W_c*beta_c)`` before all FiLM keys are
    removed. A later strict load into the RGB model proves that no conditioned
    parameters survive.
    """
    if not isinstance(state, Mapping) or any(not isinstance(key, str) for key in state):
        raise TypeError("Expected a model state mapping")
    actual_film = {key for key in state if key.startswith("dimension_film.")}
    if actual_film != FILM_KEYS:
        raise ValueError(f"Unexpected FiLM key set: {sorted(actual_film)}")
    gamma, beta = zero_context_film(state)
    weight = _finite_tensor(state, "final_layer.weight", (9, 256, 1, 1))
    bias = _finite_tensor(state, "final_layer.bias", (9,))
    if gamma.dtype != weight.dtype or beta.dtype != weight.dtype or bias.dtype != weight.dtype:
        raise ValueError("FiLM and final convolution dtypes differ")

    folded = {
        key: value.detach().clone()
        for key, value in state.items()
        if key not in FILM_KEYS
    }
    folded["final_layer.weight"] = (
        weight * (1. + gamma).reshape(1, 256, 1, 1)).detach().clone()
    folded["final_layer.bias"] = (
        bias + (weight[:, :, 0, 0] * beta.reshape(1, 256)).sum(dim=1)
    ).detach().clone()
    if any(key.startswith("dimension_film.") for key in folded):
        raise AssertionError("FiLM keys survived folding")
    return folded


def folded_rgb_network(state: Mapping[str, torch.Tensor]):
    """Fold CONSTANT state and strict-load the bound pure-RGB architecture."""
    if C.sha256(RGB_MODEL_SOURCE) != RGB_MODEL_SHA256:
        raise RuntimeError("Bound pure-RGB model source changed")
    network = rgb_model_module().SimpleBaselineResNet18(
        num_keypoints=9, pretrained_state=None)
    expected = network.state_dict()
    folded = fold_constant_state_dict(state)
    if set(folded) != set(expected):
        missing = sorted(set(expected) - set(folded))
        extra = sorted(set(folded) - set(expected))
        raise RuntimeError(f"Folded RGB state mismatch: missing={missing}, extra={extra}")
    for key, value in folded.items():
        reference = expected[key]
        if (value.shape != reference.shape or value.dtype != reference.dtype
                or value.is_meta
                or (value.is_floating_point() and not torch.isfinite(value).all())):
            raise RuntimeError(f"Invalid folded RGB state tensor: {key}")
    network.load_state_dict(folded, strict=True)
    return network


def recipe() -> dict:
    return {
        "schema": "frozen_resnet18_constant_folded_rgb_adapter_v1",
        "baseline_arm": "CONSTANT",
        "source_training": (
            "10-epoch synthetic CONSTANT architecture-control arm; "
            "this is not the 60-epoch RGB baseline"),
        "external_inputs": ["RGB"],
        "runtime_dimension_context": False,
        "fold": "z=zeros(5); W'=W*(1+gamma); b'=b+sum_c(W_c*beta_c)",
        "decoder": "bound DSNT spatial-softmax expectation then stride4",
        "input": input_module().contract(),
        "feature_taps": ["layer2", "layer3"],
        "feature_channels": list(FEATURE_CHANNELS),
        "feature_strides": list(FEATURE_STRIDES),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "rgb_model_source_sha256": RGB_MODEL_SHA256,
        "strict_rgb_load": True,
        "frozen": True,
        "GT_inputs": False,
        "camera_inputs": False,
        "TF32": False,
        "AMP": False,
    }


def network_delta_to_original(
    delta_net,
    affine_input_to_net,
    original_hw,
    *,
    cap_fraction: float = C.CAP_FRACTION,
) -> np.ndarray:
    """Invert affine scale, then apply a circular cap in original pixels."""
    delta = np.asarray(delta_net, np.float64)
    affine = np.asarray(affine_input_to_net, np.float64)
    hw = np.asarray(original_hw, np.float64)
    if delta.shape[-1:] != (2,) or affine.shape != (3, 3) or hw.shape != (2,):
        raise ValueError("Expected delta[...,2], affine[3,3], original_hw[2]")
    if (not np.isfinite(delta).all() or not np.isfinite(affine).all()
            or not np.isfinite(hw).all() or (hw <= 0).any()):
        raise ValueError("Affine conversion inputs must be finite and positive")
    linear = affine[:2, :2]
    if not np.allclose(linear, np.diag(np.diag(linear)), rtol=0, atol=0):
        raise ValueError("The bound preprocessing contract requires an axis-aligned affine")
    if ((np.diag(linear) <= 0).any() or not math.isfinite(float(cap_fraction))
            or cap_fraction < 0):
        raise ValueError("Affine scale and cap fraction must be nonnegative/positive")
    # Translation is intentionally absent for a displacement.
    original = np.linalg.solve(linear, delta.reshape(-1, 2).T).T.reshape(delta.shape)
    cap = float(cap_fraction) * math.hypot(float(hw[0]), float(hw[1]))
    norm = np.linalg.norm(original, axis=-1)
    factor = np.minimum(1., cap / np.maximum(norm, 1e-12))
    return original * factor[..., None]


class FrozenConstantResnetAdapter:
    """Strict CONSTANT checkpoint folded into an image-only RGB network."""

    def __init__(self, device="cpu", checkpoint: Path | str | None = None):
        audit, payload = load_constant_contract(load_checkpoint=True)
        expected_checkpoint = C.verify(audit["checkpoint"])
        if checkpoint is not None and Path(checkpoint).resolve() != expected_checkpoint.resolve():
            raise ValueError("Only the completed CONSTANT final checkpoint is accepted")
        self.device = torch.device(device)
        self.network = folded_rgb_network(payload["model_state_dict"])
        self.network = self.network.requires_grad_(False).eval().to(self.device)
        self.checkpoint = audit["checkpoint"]
        self.protocol = audit["protocol"]
        self.completion = audit["completion"]
        self.trained_checkpoint_loaded = True
        self.dimension_film_folded_at_zero = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    @classmethod
    def _cpu_fixture(cls, network):
        """Explicit unit-test construction; never claims a trained checkpoint."""
        value = cls.__new__(cls)
        value.device = torch.device("cpu")
        value.network = network.requires_grad_(False).eval()
        value.checkpoint = {"path": "CPU_FIXTURE", "sha256": "UNTRAINED_FIXTURE"}
        value.protocol = {"path": "CPU_FIXTURE", "sha256": "UNTRAINED_FIXTURE"}
        value.completion = {"path": "CPU_FIXTURE", "sha256": "UNTRAINED_FIXTURE"}
        value.trained_checkpoint_loaded = False
        value.dimension_film_folded_at_zero = True
        return value

    @staticmethod
    def prepare(image, source_pre_padded=False):
        return input_module().prepare(image, source_pre_padded=source_pre_padded)

    def _check_runtime(self, tensor):
        if (tensor.ndim != 4 or tuple(tensor.shape[1:]) != INPUT_CHW
                or tensor.dtype != torch.float32):
            raise ValueError("Expected normalized float32 Bx3x384x512")
        if any(module.training for module in self.network.modules()):
            raise RuntimeError("Frozen CONSTANT network entered training mode")
        if any(parameter.requires_grad for parameter in self.network.parameters()):
            raise RuntimeError("Frozen CONSTANT network has trainable parameters")

    @staticmethod
    def _check_features(features, batch):
        if len(features) != 2:
            raise ValueError("Expected layer2/layer3 features")
        for value, shape in zip(features, FEATURE_SHAPES):
            if tuple(value.shape) != (batch, *shape) or value.dtype != torch.float32:
                raise ValueError("CONSTANT feature tap contract changed")
            if value.requires_grad or torch.is_inference(value):
                raise ValueError("N3 training requires normal detached feature tensors")

    @torch.no_grad()
    def features_only(self, batch_tensor):
        self._check_runtime(batch_tensor)
        with torch.inference_mode(False), torch.no_grad():
            tensor = batch_tensor.to(self.device)
            if torch.is_inference(tensor):
                tensor = tensor.clone()
            features = tuple(value.detach() for value in self.network.features_only(tensor))
        self._check_features(features, len(batch_tensor))
        return features

    @torch.no_grad()
    def infer_batch(self, images, *, source_pre_padded=False, return_features=True):
        if not images:
            raise ValueError("At least one RGB image is required")
        prepared = [self.prepare(image, source_pre_padded=source_pre_padded) for image in images]
        with torch.inference_mode(False), torch.no_grad():
            tensor = torch.stack([item["tensor"] for item in prepared]).to(self.device)
            self._check_runtime(tensor)
            result = self.network(tensor, return_features=return_features)
            if return_features:
                if set(result) != {"heatmaps", "features"}:
                    raise RuntimeError("CONSTANT return_features schema changed")
                logits = result["heatmaps"]
                features = tuple(value.detach() for value in result["features"])
                self._check_features(features, len(images))
            else:
                logits, features = result, ()
            if tuple(logits.shape[1:]) != HEATMAP_SHAPE or not torch.isfinite(logits).all():
                raise RuntimeError("CONSTANT heatmap contract changed")
            decoded = decoder_module().decode_logits(logits, prepared)
        for index, output in enumerate(decoded):
            output.update({key: value for key, value in prepared[index].items() if key != "tensor"})
            output.update(
                base_rgb_only=True,
                dimension_film_folded_at_zero=True,
                runtime_dimension_context=False,
                checkpoint_sha256=self.checkpoint["sha256"],
                baseline_protocol_sha256=self.protocol["sha256"],
                training_receipt_sha256=self.completion["sha256"],
                belief_shape=list(HEATMAP_SHAPE),
                status="OK",
            )
            if return_features:
                output["features"] = tuple(value[index] for value in features)
        return decoded

    def infer(self, image, *, source_pre_padded=False, return_features=True):
        return self.infer_batch(
            [image], source_pre_padded=source_pre_padded,
            return_features=return_features)[0]


def actual_checkpoint_smoke(device="cuda") -> dict:
    """Fold the real checkpoint and execute one synthetic image-only forward."""
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    adapter = FrozenConstantResnetAdapter(device)
    image = np.zeros((480, 640, 3), np.uint8)
    output = adapter.infer(image, return_features=True)
    forward_parameters = inspect.signature(adapter.network.forward).parameters
    result = {
        "PASS": True,
        "device": str(adapter.device),
        "trained_checkpoint_loaded": adapter.trained_checkpoint_loaded,
        "checkpoint_sha256": adapter.checkpoint["sha256"],
        "checkpoint_identity": {"arm": "CONSTANT", "epoch": 10, "step": 34990},
        "dimension_film_folded_at_zero": adapter.dimension_film_folded_at_zero,
        "runtime_dimension_context": "dimension_context" in forward_parameters,
        "point_shape": list(np.asarray(output["points_net"]).shape),
        "feature_shapes": [list(value.shape) for value in output["features"]],
        "normal_detached_features": all(
            not value.requires_grad and not torch.is_inference(value)
            for value in output["features"]),
    }
    if result["runtime_dimension_context"]:
        raise AssertionError("Folded RGB base unexpectedly accepts a dimension context")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("contract", "smoke"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "contract":
        audit, payload = load_constant_contract(load_checkpoint=True)
        folded = fold_constant_state_dict(payload["model_state_dict"])
        result = {
            "PASS": True,
            **audit,
            "folded_state_tensors": len(folded),
            "film_keys_removed": sorted(FILM_KEYS),
            "recipe": recipe(),
        }
    else:
        result = actual_checkpoint_smoke(args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
