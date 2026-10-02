"""Paired fixed-budget training for all four frozen-FULL correction heads."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import random
import time

import numpy as np
import torch

from . import common as C
from .data import order_digest, paired_order
from .refiner import forward_arm, loss_for_arm, model_for_arm
from .resume import (atomic_torch_save, checkpoint_payload,
                     restore_training_state)
from .source_cache import FeatureDataset


def learning_rate(step: int, optimizer_contract: dict) -> float:
    if not 1 <= step <= C.STEPS:
        raise ValueError("Learning-rate step outside fixed budget")
    warmup = int(optimizer_contract["warmup_steps"])
    base = float(optimizer_contract["lr"])
    if step <= warmup:
        return base * step / warmup
    fraction = (step - warmup) / (C.STEPS - warmup)
    final = float(optimizer_contract["cosine_final_lr_fraction"])
    return base * (final + (1. - final) * .5 * (1. + math.cos(math.pi * fraction)))


def build_heads(seed: int, device, optimizer_contract: dict):
    models, optimizers = {}, {}
    for arm in C.HEAD_ARMS:
        # Reseeding makes P0's inherited visual state identical to P5 and makes
        # P5/P5_CONSTANT identical over every parameter at update zero.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.device(device).type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = model_for_arm(arm).to(device).train()
        models[arm] = model
        optimizers[arm] = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_contract["lr"]),
            betas=tuple(optimizer_contract["betas"]),
            weight_decay=float(optimizer_contract["weight_decay"]),
        )
    p0 = models["P0"].state_dict()
    for arm in ("P5", "P5_CONSTANT"):
        state = models[arm].state_dict()
        for key in p0:
            if key in state and not torch.equal(p0[key], state[key]):
                raise ValueError(f"Paired visual initialization drift: {arm}/{key}")
    for key, value in models["P5"].state_dict().items():
        if not torch.equal(value, models["P5_CONSTANT"].state_dict()[key]):
            raise ValueError(f"P5/P5_CONSTANT initialization drift: {key}")
    return models, optimizers


def materialize_orders(data: FeatureDataset) -> dict:
    bindings = {}
    for seed in C.SEEDS:
        order = paired_order(data.train_rows, seed)
        path = C.RAW / f"order_seed{seed}.npy"
        if path.exists():
            if not np.array_equal(np.load(path), order):
                raise ValueError(f"Existing batch order differs for seed {seed}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                np.save(handle, order)
        bindings[str(seed)] = C.binding(path)
        if order_digest(order) != hashlib.sha256(order.astype("<i8").tobytes()).hexdigest():
            raise AssertionError("Order hash implementation drift")
    C.write_frozen_json(C.DOC / "BATCH_ORDER.json", dict(
        complete=True,
        protocol=C.binding(C.PROTOCOL),
        seeds=bindings,
        train_rows=len(data.train_rows),
        train_rows_sha256=hashlib.sha256(data.train_rows.astype("<i8").tobytes()).hexdigest(),
        all_four_arms_same_order=True,
        per_seed_physical_backbone_exposures=C.STEPS * C.BATCH,
        per_seed_nominal_head_exposures=len(C.HEAD_ARMS) * C.STEPS * C.BATCH,
    ))
    return bindings


def train(adapter, *, checkpoint_interval=1) -> dict:
    protocol = C.verify_protocol()
    if torch.device(adapter.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Production head training requires an explicitly supplied CUDA FULL adapter")
    if (not getattr(adapter, "trained_full_loaded", False)
            or adapter.checkpoint["sha256"] != protocol["baseline"]["sha256"]):
        raise ValueError("Training adapter differs from sealed FULL baseline")
    if checkpoint_interval != 1:
        raise ValueError("Exact resume requires an atomic checkpoint after every completed update")
    cache = C.read_json(C.DOC / "SOURCE_CACHE_COMPLETE.json")
    if not cache.get("complete") or cache.get("protocol_sha256") != C.sha256(C.PROTOCOL):
        raise ValueError("Verified source cache is required")
    data = FeatureDataset()
    orders = materialize_orders(data)
    optimizer_contract = protocol["training"]["optimizer"]
    baseline_sha = protocol["baseline"]["sha256"]
    protocol_sha = C.sha256(C.PROTOCOL)
    completions = []
    try:
        for seed in C.SEEDS:
            directory = C.RAW / "runs" / f"seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            completion_path = C.DOC / f"TRAIN_SEED{seed}.json"
            checkpoint_path = directory / "paired_last.pt"
            if completion_path.exists():
                existing = C.read_json(completion_path)
                if (existing.get("complete") is not True
                        or C.verify_binding(existing["checkpoint"]) != checkpoint_path.resolve()):
                    raise ValueError(f"Invalid existing seed completion: {seed}")
                completions.append(existing)
                continue
            models, optimizers = build_heads(seed, adapter.device, optimizer_contract)
            order = np.load(C.verify_binding(orders[str(seed)]))
            order_sha = orders[str(seed)]["sha256"]
            start = 0
            history = []
            elapsed_before = 0.
            resumes = []
            include_cuda = True
            if checkpoint_path.exists():
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if payload.get("seed") != seed:
                    raise ValueError("Resume seed/path mismatch")
                restore_training_state(
                    payload, models=models, optimizers=optimizers,
                    protocol_sha256=protocol_sha, baseline_sha256=baseline_sha,
                    order_sha256=order_sha, include_cuda=include_cuda,
                )
                start = payload["step"]
                history = list(payload["history"])
                elapsed_before = float(payload["elapsed_seconds"])
                resumes = list(payload.get("resumes", [])) + [dict(time=C.now(), from_step=start)]
                if payload["complete"]:
                    start = C.STEPS
            began = time.perf_counter()
            with ThreadPoolExecutor(max_workers=1) as prefetch:
                future = prefetch.submit(data.prepare, order[start], adapter) if start < C.STEPS else None
                for step in range(start + 1, C.STEPS + 1):
                    prepared = future.result()
                    if step < C.STEPS:
                        future = prefetch.submit(data.prepare, order[step], adapter)
                    batch = data.batch_features(order[step - 1], adapter, prepared)
                    lr = learning_rate(step, optimizer_contract)
                    update = dict(step=step, lr=lr, arms={})
                    for arm in C.HEAD_ARMS:
                        model = models[arm]
                        optimizer = optimizers[arm]
                        optimizer.zero_grad(set_to_none=True)
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                        output = forward_arm(model, batch, arm, lam=0.)
                        value = loss_for_arm(output, batch, arm)
                        if not torch.isfinite(value) or not output["point_support"].any():
                            raise FloatingPointError(f"Invalid {arm} update at seed{seed}/step{step}")
                        value.backward()
                        norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(optimizer_contract["gradient_clip_norm"]),
                            error_if_nonfinite=True)
                        before = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
                        optimizer.step()
                        after = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
                        change = float((after - before).norm())
                        if not torch.isfinite(after).all() or change <= 0:
                            raise FloatingPointError(f"No finite {arm} update at seed{seed}/step{step}")
                        mask = output["point_support"] & batch["gt_valid"][:, :8]
                        update["arms"][arm] = dict(
                            loss=float(value.detach()),
                            preclip_grad_norm=float(norm),
                            parameter_update_norm=change,
                            supported_frames=int(mask.any(-1).sum()),
                            supported_corners=int(mask.sum()),
                        )
                    if any(parameter.grad is not None or parameter.requires_grad
                           for parameter in adapter.network.parameters()):
                        raise RuntimeError("Frozen FULL backbone received a gradient")
                    history.append(update)
                    elapsed = elapsed_before + time.perf_counter() - began
                    if step == 1 or step % checkpoint_interval == 0 or step == C.STEPS:
                        state = checkpoint_payload(
                            step=step, protocol_sha256=protocol_sha,
                            baseline_sha256=baseline_sha, order_sha256=order_sha,
                            models=models, optimizers=optimizers, history=history,
                            elapsed_seconds=elapsed, include_cuda=include_cuda)
                        state.update(seed=seed, resumes=resumes)
                        atomic_torch_save(checkpoint_path, state)
                    if step == 1 or step % 100 == 0 or step == C.STEPS:
                        print("FULL_DIM_REFINER_TRAIN", seed, step,
                              " ".join(f"{arm}={update['arms'][arm]['loss']:.6f}"
                                       for arm in C.HEAD_ARMS), flush=True)
                    del batch, prepared
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if state.get("complete") is not True or state.get("step") != C.STEPS:
                raise RuntimeError("Final paired checkpoint is incomplete")
            metrics_path = directory / "STEP_METRICS.json"
            C.write_frozen_json(metrics_path, dict(seed=seed, updates=history))
            completion = dict(
                complete=True,
                seed=seed,
                updates_per_arm=C.STEPS,
                exposures_per_arm=C.STEPS * C.BATCH,
                arms=list(C.HEAD_ARMS),
                usable_training_rows=len(data.train_rows),
                checkpoint=C.binding(checkpoint_path),
                order=orders[str(seed)],
                metrics=C.binding(metrics_path),
                parameters={arm: sum(parameter.numel() for parameter in models[arm].parameters())
                            for arm in C.HEAD_ARMS},
                elapsed_seconds=float(state["elapsed_seconds"]),
                resumes=resumes,
                real_training_images=0,
                backbone_retrained=False,
                final_checkpoint_only=True,
                paired_shared_frozen_features=True,
            )
            C.write_frozen_json(completion_path, completion)
            completions.append(completion)
            del models, optimizers, state
            torch.cuda.empty_cache()
    finally:
        data.close()
    if C.sha256(C.verify_binding(protocol["baseline"])) != baseline_sha:
        raise ValueError("FULL checkpoint changed during head training")
    result = dict(
        complete=True,
        fits=len(C.HEAD_ARMS) * len(C.SEEDS),
        steps_per_fit=C.STEPS,
        total_head_updates=len(C.HEAD_ARMS) * len(C.SEEDS) * C.STEPS,
        nominal_head_exposures=len(C.HEAD_ARMS) * len(C.SEEDS) * C.STEPS * C.BATCH,
        physical_shared_backbone_exposures=len(C.SEEDS) * C.STEPS * C.BATCH,
        seeds=completions,
        baseline=protocol["baseline"],
        protocol=C.binding(C.PROTOCOL),
        real_training_images=0,
    )
    C.write_frozen_json(C.DOC / "TRAINING_COMPLETE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from .full_adapter import FrozenFullAdapter
    train(FrozenFullAdapter(args.device))


if __name__ == "__main__":
    main()
