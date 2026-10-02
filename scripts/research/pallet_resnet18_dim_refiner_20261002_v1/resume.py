"""Exact-state checkpoint helpers; importing this module never touches CUDA."""
from __future__ import annotations

import os
from pathlib import Path
import random

import numpy as np
import torch

from .common import HEAD_ARMS, STEPS


def capture_rng(*, include_cuda: bool) -> dict:
    if include_cuda and not torch.cuda.is_initialized():
        raise RuntimeError("CUDA RNG capture requires an already initialized training device")
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if include_cuda else None,
    )


def restore_rng(state: dict, *, include_cuda: bool) -> None:
    if set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Incomplete RNG state")
    if include_cuda != (state["cuda"] is not None):
        raise ValueError("CUDA RNG resume mode differs from checkpoint")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if include_cuda:
        if not torch.cuda.is_initialized():
            raise RuntimeError("CUDA must already be initialized before restoring its RNG")
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_payload(*, step: int, protocol_sha256: str, baseline_sha256: str,
                       order_sha256: str, models: dict, optimizers: dict,
                       history: list, elapsed_seconds: float, include_cuda: bool) -> dict:
    if set(models) != set(HEAD_ARMS) or set(optimizers) != set(HEAD_ARMS):
        raise ValueError("Every paired head arm must be checkpointed together")
    if not 0 <= step <= STEPS:
        raise ValueError("Checkpoint step outside fixed budget")
    if any(not isinstance(value, str) or len(value) != 64
           for value in (protocol_sha256, baseline_sha256, order_sha256)):
        raise ValueError("Checkpoint bindings must be SHA-256 strings")
    return dict(
        schema="resnet18_dim_refiner_exact_resume_v1",
        complete=step == STEPS,
        step=step,
        arms=list(HEAD_ARMS),
        protocol_sha256=protocol_sha256,
        baseline_sha256=baseline_sha256,
        order_sha256=order_sha256,
        models={arm: models[arm].state_dict() for arm in HEAD_ARMS},
        optimizers={arm: optimizers[arm].state_dict() for arm in HEAD_ARMS},
        rng=capture_rng(include_cuda=include_cuda),
        history=list(history),
        elapsed_seconds=float(elapsed_seconds),
    )


def validate_checkpoint(payload: dict, *, protocol_sha256: str,
                        baseline_sha256: str, order_sha256: str) -> None:
    if payload.get("schema") != "resnet18_dim_refiner_exact_resume_v1":
        raise ValueError("Resume schema mismatch")
    if payload.get("arms") != list(HEAD_ARMS):
        raise ValueError("Resume arm set/order mismatch")
    if set(payload.get("models", {})) != set(HEAD_ARMS) or set(payload.get("optimizers", {})) != set(HEAD_ARMS):
        raise ValueError("Resume is missing a paired head")
    if payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Resume protocol mismatch")
    if payload.get("baseline_sha256") != baseline_sha256:
        raise ValueError("Resume frozen baseline mismatch")
    if payload.get("order_sha256") != order_sha256:
        raise ValueError("Resume batch order mismatch")
    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or not 0 <= step <= STEPS:
        raise ValueError("Resume step outside fixed budget")
    if payload.get("complete") is not (step == STEPS):
        raise ValueError("Resume completion flag disagrees with step")


def restore_training_state(payload: dict, *, models: dict, optimizers: dict,
                           protocol_sha256: str, baseline_sha256: str,
                           order_sha256: str, include_cuda: bool) -> None:
    validate_checkpoint(payload, protocol_sha256=protocol_sha256,
                        baseline_sha256=baseline_sha256, order_sha256=order_sha256)
    if set(models) != set(HEAD_ARMS) or set(optimizers) != set(HEAD_ARMS):
        raise ValueError("Runtime paired arms differ from checkpoint")
    for arm in HEAD_ARMS:
        models[arm].load_state_dict(payload["models"][arm], strict=True)
        optimizers[arm].load_state_dict(payload["optimizers"][arm])
    restore_rng(payload["rng"], include_cuda=include_cuda)


def atomic_torch_save(path: Path | str, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".pending")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(destination)

