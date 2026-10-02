"""Exactly six fixed-budget N3 fits with durable exact-state resume."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import torch

from . import common as C
from .adapters import build_adapter
from .data import N3Dataset, cache_resnet18
from .model import build_n3, n3_loss


KEYS = ("p3", "p4", "points", "boxes", "point_valid", "input_shape")


def forward(head, batch, *, lam=0., temperature=1.):
    return head(*(batch[key] for key in KEYS),
                dimension_context=batch["dimension_context"],
                lam=lam, temperature=temperature)


def state_hash(state) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_protocol():
    path = C.DOC / "PROTOCOL.json"
    protocol = C.read(path)
    if protocol["schema"] != "pallet_n3_completion_v3_protocol_v1":
        raise RuntimeError("N3 protocol schema drift")
    for entry in protocol["bindings"]:
        C.verify(entry)
    return protocol


def checkpoint_path(backbone: str, seed: int, smoke=False) -> Path:
    group = "smoke" if smoke else "runs"
    return C.RAW / group / backbone / f"seed{seed}" / "last.pt"


def receipt_path(backbone: str, seed: int, smoke=False) -> Path:
    prefix = "SMOKE" if smoke else "TRAIN"
    return C.DOC / f"{prefix}_{backbone.upper()}_SEED{seed}.json"


def _save(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name("last.pending.pt")
    torch.save(state, pending)
    pending.replace(path)


def run_fit(backbone: str, adapter, data: N3Dataset, seed: int,
            *, steps=C.STEPS, smoke=False) -> dict:
    if backbone not in C.CONFIGS or seed not in C.SEEDS:
        raise ValueError((backbone, seed))
    if smoke and not 1 <= steps <= 100:
        raise ValueError("Smoke is capped at 100 steps")
    if not smoke and steps != C.STEPS:
        raise ValueError("Main N3 fits must use exactly 6000 steps")
    protocol = load_protocol()
    path = checkpoint_path(backbone, seed, smoke)
    receipt_file = receipt_path(backbone, seed, smoke)
    if receipt_file.exists():
        receipt = C.read(receipt_file)
        if (not receipt["complete"] or receipt["steps"] != steps
                or C.sha256(path) != receipt["checkpoint"]["sha256"]):
            raise RuntimeError(f"Invalid existing fit receipt: {receipt_file}")
        return receipt
    C.gpu_snapshot()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = build_n3(backbone).to(adapter.device).train()
    initial_hash = state_hash(head.state_dict())
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=C.OPTIMIZER["lr"],
        betas=C.OPTIMIZER["betas"], weight_decay=C.OPTIMIZER["weight_decay"])
    full_order, order_binding = data.order(seed)
    order = full_order[:steps]
    order_hash = hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()
    start_step = 0
    elapsed_before = 0.
    history = []
    resumes = []
    first_step = None
    second_step = None
    protocol_sha = C.sha256(C.DOC / "PROTOCOL.json")
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (state["protocol_sha256"] != protocol_sha
                or state["order_sha256"] != order_hash
                or state["initial_state_sha256"] != initial_hash
                or state["steps"] != steps):
            raise RuntimeError("Resume contract mismatch")
        head.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        start_step = state["step"]
        elapsed_before = state["elapsed_seconds"]
        history = state["history"]
        first_step = state.get("first_step")
        second_step = state.get("second_step")
        resumes = state.get("resumes", []) + [dict(time=C.now(), from_step=start_step)]
    started = time.perf_counter()
    # Only image decoding/preprocessing is prefetched.  Frozen-backbone CUDA
    # work remains on this thread, so checkpoint/resume ordering is exact and
    # there is never a second model stream competing with the optimizer step.
    with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
        prepared_future = (prefetch_pool.submit(
            data.prepare, order[start_step], adapter) if start_step < steps else None)
        for step in range(start_step + 1, steps + 1):
            prepared = prepared_future.result()
            prepared_future = (prefetch_pool.submit(
                data.prepare, order[step], adapter) if step < steps else None)
            batch = data.batch_features(order[step - 1], adapter, prepared=prepared)
            optimizer.zero_grad(set_to_none=True)
            lr = C.learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            output = forward(head, batch, lam=0.)
            if step == 1 and not torch.equal(output["logits"], output["base_logits"]):
                raise RuntimeError("Zero-initialized metadata changed step-zero logits")
            value = n3_loss(output, batch)
            if not torch.isfinite(value):
                raise FloatingPointError((backbone, seed, step, value))
            if not output["point_support"].any():
                raise RuntimeError("All candidate stencils unsupported")
            value.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), C.OPTIMIZER["gradient_clip_norm"],
                error_if_nonfinite=True)
            gradients = {name: (None if parameter.grad is None else float(parameter.grad.norm()))
                         for name, parameter in head.named_parameters()}
            if step == 1:
                if not (gradients["adapt3.0.weight"] > 0
                        and gradients["adapt4.0.weight"] > 0
                        and gradients["metadata_scorer.2.weight"] > 0
                        and gradients["metadata_encoder.0.weight"] == 0):
                    raise RuntimeError("First-step visual/metadata gradient contract failed")
                first_step = dict(gradients=gradients, zero_effect_logits=True,
                                  metadata_encoder_zero_expected=True)
            optimizer.step()
            if step == 2:
                if not gradients["metadata_encoder.0.weight"] > 0:
                    raise RuntimeError("Metadata encoder was not connected on step two")
                second_step = dict(metadata_encoder_connected=True,
                                   metadata_encoder_gradient=gradients["metadata_encoder.0.weight"])
            elapsed = elapsed_before + time.perf_counter() - started
            if step == 1 or step % 100 == 0 or step == steps:
                row = dict(step=step, loss=float(value.detach()), lr=lr,
                           preclip_gradient_norm=float(gradient_norm),
                           supported_frames=int((output["point_support"]
                               & batch["gt_valid"][:, :8]).any(-1).sum()),
                           supported_corners=int((output["point_support"]
                               & batch["gt_valid"][:, :8]).sum()),
                           elapsed_seconds=elapsed)
                history.append(row)
                C.write(C.RAW / "TRAIN_PROGRESS.json",
                        dict(backbone=backbone, seed=seed, smoke=smoke,
                             **row, gpu=C.gpu_snapshot()))
                print("N3_TRAIN", backbone, seed, step, steps,
                      round(row["loss"], 6), round(elapsed, 1), flush=True)
            if step % 500 == 0 or step == steps:
                state = dict(
                    complete=step == steps, backbone=backbone, seed=seed,
                    step=step, steps=steps, smoke=smoke,
                    model_state_dict=head.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state_all(),
                    protocol_sha256=protocol_sha, order_sha256=order_hash,
                    initial_state_sha256=initial_hash, elapsed_seconds=elapsed,
                    history=history, first_step=first_step,
                    second_step=second_step, resumes=resumes,
                    base_checkpoint_sha256=(adapter.checkpoint_sha256),
                )
                _save(path, state)
            del batch, output, value, prepared
    if C.sha256(C.DOPE_WEIGHTS if backbone == "dope" else C.RESNET_CONSTANT) != adapter.checkpoint_sha256:
        raise RuntimeError("Frozen base checkpoint changed during head training")
    receipt = dict(
        schema="pallet_n3_completion_v3_fit_v1", complete=True,
        backbone=backbone, seed=seed, smoke=smoke, steps=steps,
        exposures=steps * C.BATCH, checkpoint=C.binding(path),
        order=order_binding, order_prefix_sha256=order_hash,
        initial_state_sha256=initial_hash,
        final_state_sha256=state_hash(head.state_dict()),
        base_checkpoint_sha256=adapter.checkpoint_sha256,
        trainable_parameters=sum(parameter.numel() for parameter in head.parameters()),
        elapsed_seconds=state["elapsed_seconds"], history=history,
        first_step=first_step, second_step=second_step, resumes=resumes,
        optimizer=dict(C.OPTIMIZER), FP32=True, AMP=False,
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
        TF32_cudnn=torch.backends.cudnn.allow_tf32,
        real_training_images=0, base_retrained=False,
        symmetry_supervision=True, dimension_input_to_N3=True,
        base_receives_dimensions=False, final_step_only=True,
    )
    receipt["optimizer"]["betas"] = list(receipt["optimizer"]["betas"])
    C.write(receipt_file, receipt, freeze=True)
    del head, optimizer, state
    torch.cuda.empty_cache()
    return receipt


def smoke(backbone: str) -> dict:
    adapter = build_adapter(backbone)
    if backbone == "resnet18":
        cache_resnet18(adapter)
    data = N3Dataset(backbone)
    try:
        receipt = run_fit(backbone, adapter, data, 1, steps=2, smoke=True)
    finally:
        data.close()
    output = dict(complete=True, backbone=backbone, updates=2,
                  performance_verdict=False, discarded=True, fit=receipt)
    C.write(C.DOC / f"SMOKE_{backbone.upper()}_COMPLETE.json", output, freeze=True)
    return output


def train_backbone(backbone: str) -> dict:
    smoke_receipt = C.read(C.DOC / f"SMOKE_{backbone.upper()}_COMPLETE.json")
    if not smoke_receipt["complete"] or not smoke_receipt["discarded"]:
        raise RuntimeError("Discarded smoke must complete first")
    adapter = build_adapter(backbone)
    if backbone == "resnet18":
        cache_resnet18(adapter)
    data = N3Dataset(backbone)
    try:
        fits = [run_fit(backbone, adapter, data, seed) for seed in C.SEEDS]
    finally:
        data.close()
    output = dict(
        schema="pallet_n3_completion_v3_training_v1", complete=True,
        backbone=backbone, fits=3, seeds=list(C.SEEDS),
        steps_per_fit=C.STEPS, total_updates=3 * C.STEPS,
        total_exposures=3 * C.STEPS * C.BATCH, runs=fits,
        no_outcome_driven_extension=True, base_retrained=False,
    )
    C.write(C.DOC / f"TRAINING_{backbone.upper()}_COMPLETE.json", output, freeze=True)
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("smoke", "train"))
    parser.add_argument("backbone", choices=tuple(C.CONFIGS))
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    result = smoke(arguments.backbone) if arguments.stage == "smoke" else train_backbone(arguments.backbone)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
