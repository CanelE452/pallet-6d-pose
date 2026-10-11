"""Matched small residual scorers around an immutable image-only P checkpoint.

Only the new 2,305-parameter scorers train. Candidate feature hooks are read-only.
The source tree, source caches and checkpoints are never written by this script.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn


ARMS = ("D_CONTROL", "D_JOINT_DIM")


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_modules(root):
    root = Path(root).resolve()
    locations = [root / "scripts/research/pallet_dim_conditioned_p_v1",
                 root / "scripts/research/pallet_final_ml_contribution_test_v1", root]
    for path in reversed(locations):
        sys.path.insert(0, str(path))
    import dcp_env
    import refiner
    from data import PaperData
    from generic_point_refiner import GenericPointRefiner
    assert dcp_env.ROOT == root
    return dcp_env, refiner, PaperData, GenericPointRefiner


class ResidualScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(70, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, features, metadata):
        assert features.shape[-2:] == (222, 65)
        assert metadata.shape == (len(features), 5)
        condition = metadata[:, None, None].expand(-1, 8, 222, -1)
        return self.scorer(torch.cat([features, condition], -1)).squeeze(-1)


class FrozenCandidates:
    """Reads original scorer inputs without changing any original output."""
    def __init__(self, base):
        self.base = base.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, batch):
        captures = {}

        def capture_regular(module, inputs):
            captures["regular"] = inputs[0].detach()

        def capture_null(module, inputs):
            captures["null"] = inputs[0].detach()

        hooks = [self.base.scorer[0].register_forward_pre_hook(capture_regular),
                 self.base.null_scorer[0].register_forward_pre_hook(capture_null)]
        try:
            output = self.base(*(batch[key] for key in
                                     ("p3", "p4", "points", "boxes", "point_valid", "input_shape")), lam=0)
        finally:
            for hook in hooks:
                hook.remove()
        regular, null = captures["regular"], captures["null"]
        assert regular.shape == (len(batch["points"]), 8, 221, 64)
        assert null.shape == (len(batch["points"]), 8, 61)
        regular = torch.cat([regular, regular.new_zeros((*regular.shape[:-1], 1))], -1)
        null = torch.cat([null, null.new_zeros((*null.shape[:-1], 3)),
                          null.new_ones((*null.shape[:-1], 1))], -1)
        features = torch.cat([regular, null[:, :, None]], -2)
        assert features.shape == (len(batch["points"]), 8, 222, 65)
        return output, features


def altered_output(base_output, features, head, metadata):
    output = dict(base_output)
    output["base_logits"] = base_output["logits"]
    output["residual"] = head(features, metadata)
    output["logits"] = output["base_logits"] + output["residual"]
    return output


def setup(args, device="cpu"):
    env, refiner, data_type, base_type = source_modules(args.source_root)
    protocol = read(env.DOC / "TRAIN_PROTOCOL_LOCK.json")
    checkpoint_path = env.C.BRAW / "runs/seed1/last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["complete"] and checkpoint["step"] == 6000
    assert checkpoint["baseline_checkpoint_sha256"] == env.R0_SHA
    base = base_type(**checkpoint["config"])
    base.load_state_dict(checkpoint["model_state_dict"])
    base = base.to(device).eval().requires_grad_(False)
    torch.manual_seed(1)
    if device == "cuda":
        torch.cuda.manual_seed_all(1)
    first = ResidualScorer().to(device)
    second = copy.deepcopy(first)
    assert state_sha(first.state_dict()) == state_sha(second.state_dict())
    assert sum(parameter.numel() for parameter in first.parameters()) == 2305
    return env, refiner, data_type, protocol, checkpoint_path, base, dict(zip(ARMS, [first, second]))


def contract(args):
    env, refiner, data_type, protocol, path, base, heads = setup(args)
    data = data_type()
    order_path = env.C.BRAW / "order_seed1.npy"
    order = np.load(order_path, mmap_mode="r")
    assert order.shape == (6000, 16)
    assert np.isin(order[:1000], data.train_rows).all()
    bindings = [path, order_path, env.DOC / "TRAIN_PROTOCOL_LOCK.json",
                env.DOC / "DIM_NORMALIZATION_LOCK.json", env.RAW / "DIMENSION_SIDECAR.npz",
                env.LINE / "SOURCE_MANIFEST.json", env.LINE / "cache/CACHE_MANIFEST.json",
                env.OLD_CODE / "generic_point_refiner.py", env.HERE / "refiner.py"]
    value = dict(status="CONTRACT_VERIFIED_NOT_FITTED", arms=list(ARMS), seed=1,
                 frozen_checkpoint=dict(path=str(path.relative_to(env.ROOT)), sha256=file_sha(path)),
                 frozen_parameters=sum(p.numel() for p in base.parameters()),
                 trainable_parameters={name: sum(p.numel() for p in head.parameters()) for name, head in heads.items()},
                 shared_initial_sha256=state_sha(heads[ARMS[0]].state_dict()),
                 original_candidates=222, regular_input=64, null_input=61, null_zero_padding=3,
                 explicit_null_flag=True, metadata_dimension=5, residual_input=70, hidden_width=32,
                 activation="SiLU", final_layer_initialization="weight=0,bias=0",
                 frozen_features="original scorer[0] and null_scorer[0] forward-pre-hook inputs",
                 metadata="canonical TRAIN-normalized logW/logD/logH/log(W/D)/log(H/sqrt(WD)); control all zeros",
                 supervision="unchanged whole-object local_phase and original Gaussian candidate CE",
                 calibration="fixed-index original CE on all synthetic calibration frames; T in [0.5,1,2,4]; nearest |logT| then smaller T",
                 lambda_cap=protocol["selected_rule"], planned_updates_per_arm=1000, batch_size=16,
                 batch_order_sha256=hashlib.sha256(np.asarray(order[:1000], dtype="<i8").tobytes()).hexdigest(),
                 unique_exposed_train_frames=int(len(np.unique(order[:1000]))), train_usable_frames=len(data.train_rows),
                 optimizer=protocol["optimizer"],
                 replay_schedule="100-step linear warmup, cosine over remaining 900 steps, final LR fraction 0.1",
                 checkpoint="last common step only; never best-real/best-val",
                 source_bindings=[dict(path=str(p.relative_to(env.ROOT)), sha256=file_sha(p)) for p in bindings],
                 code_sha256=file_sha(__file__), source_arrays_read_only=all(not v.flags.writeable for v in data.arrays.values()),
                 real_access=False, new_raw_RGB_feature_extraction=0)
    write_new(Path(args.public_dir) / "D_CONTRACT.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def throughput(args):
    env, refiner, data_type, protocol, path, base, heads = setup(args, "cuda")
    data = data_type()
    order = np.load(env.C.BRAW / "order_seed1.npy", mmap_mode="r")
    frozen = FrozenCandidates(base)
    before = state_sha(base.state_dict())
    torch.cuda.reset_peak_memory_stats()
    times = []
    checks = {}
    for i in range(6):
        torch.cuda.synchronize()
        begin = time.monotonic()
        batch = data.batch(order[i], "N3_DIM_SYM")
        output, features = frozen.forward(batch)
        for arm, head in heads.items():
            context = batch["context"] if arm == "D_JOINT_DIM" else torch.zeros_like(batch["context"])
            modified = altered_output(output, features, head, context)
            assert torch.equal(modified["logits"], output["logits"])
            assert torch.equal(refiner.decode(modified, 1, 1), refiner.decode(output, 1, 1))
            value = refiner.train_loss(modified, batch, True)
            value.backward()
            assert all(p.grad is None for p in base.parameters())
            assert float(head.scorer[-1].weight.grad.norm()) > 0
            head.zero_grad(set_to_none=True)
        if i == 0:
            with torch.no_grad():
                bare = base(*(batch[k] for k in ("p3", "p4", "points", "boxes", "point_valid", "input_shape")), lam=0)
            assert torch.equal(output["logits"], bare["logits"])
            checks = dict(hook_output_exact=True, zero_residual_logits_exact=True,
                          zero_residual_decoded_coordinates_exact=True, frozen_gradients_absent=True,
                          frozen_parameters_sha256=before)
        torch.cuda.synchronize()
        times.append(time.monotonic() - begin)
    assert state_sha(base.state_dict()) == before
    value = dict(status="DRY_RUN_ONLY_ZERO_OPTIMIZER_UPDATES", batch_size=16, timed_batches=6,
                 pair_seconds=times, steady_pair_seconds=float(np.median(times[1:])),
                 estimated_1000_pair_seconds=float(np.median(times[1:]) * 1000),
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(), checks=checks,
                 actual_device_seconds=sum(times), actual_updates=0, new_PnP_calls=0,
                 torch_version=str(torch.__version__), TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
                 TF32_cudnn=torch.backends.cudnn.allow_tf32)
    write_new(Path(args.public_dir) / "D_THROUGHPUT.json", value)
    print(json.dumps(value), flush=True)


def learning_rate(step, optimizer, steps=1000):
    warmup = optimizer["warmup_steps"]
    if step <= warmup:
        return optimizer["lr"] * step / warmup
    progress = (step - warmup) / (steps - warmup)
    last = optimizer["cosine_final_lr_fraction"]
    return optimizer["lr"] * (last + (1-last) * 0.5 * (1+math.cos(math.pi*progress)))


def train_pair(args):
    assert args.plan_lock is not None and Path(args.plan_lock).is_file(), "Root PLAN_LOCK must precede fitting"
    contract_value = read(Path(args.public_dir) / "D_CONTRACT.json")
    assert contract_value["code_sha256"] == file_sha(__file__), "Code changed after contract lock"
    throughput_value = read(Path(args.public_dir) / "D_THROUGHPUT.json")
    assert throughput_value["estimated_1000_pair_seconds"] < args.device_seconds_limit
    env, refiner, data_type, protocol, checkpoint_path, base, heads = setup(args, "cuda")
    for binding in contract_value["source_bindings"]:
        assert file_sha(env.ROOT / binding["path"]) == binding["sha256"]
    data = data_type()
    order = np.asarray(np.load(env.C.BRAW / "order_seed1.npy", mmap_mode="r")[:1000])
    frozen = FrozenCandidates(base)
    initial_base = state_sha(base.state_dict())
    settings = protocol["optimizer"]
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=settings["lr"],
                                        betas=tuple(settings["betas"]), weight_decay=settings["weight_decay"])
                  for name, head in heads.items()}
    private = Path(args.private_dir) / "D"
    private.mkdir(parents=True, exist_ok=True)
    assert all(not (private / (arm + ".pt")).exists() for arm in ARMS), "No overwrite or implicit refit"
    history = []
    first_audit = None
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    step = 0
    for step, rows in enumerate(order, 1):
        batch = data.batch(rows, "N3_DIM_SYM")
        output, features = frozen.forward(batch)
        row = dict(step=step, losses={}, gradient_norms={})
        for arm in ARMS:
            head, optimizer = heads[arm], optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step, settings)
            for group in optimizer.param_groups:
                group["lr"] = lr
            metadata = batch["context"] if arm == "D_JOINT_DIM" else torch.zeros_like(batch["context"])
            altered = altered_output(output, features, head, metadata)
            if step == 1:
                assert torch.equal(altered["logits"], output["logits"])
            value = refiner.train_loss(altered, batch, True)
            assert torch.isfinite(value)
            value.backward()
            assert all(parameter.grad is None for parameter in base.parameters())
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), settings["gradient_clip_norm"], error_if_nonfinite=True)
            row["losses"][arm] = float(value.detach())
            row["gradient_norms"][arm] = float(norm)
            optimizer.step()
        row["lr"] = lr
        if step == 1 or step % 100 == 0:
            torch.cuda.synchronize()
            row["elapsed_seconds"] = time.monotonic() - begin
            history.append(row)
            print("D_PAIR_TRAIN", json.dumps(row), flush=True)
        if step == 1:
            first_audit = dict(frozen_gradients_absent=True, initial_logits_exact=True,
                               zero_first_layer_gradient_expected=True)
        if time.monotonic() - begin >= args.device_seconds_limit:
            break
    torch.cuda.synchronize()
    elapsed = time.monotonic() - begin
    assert state_sha(base.state_dict()) == initial_base
    for arm in ARMS:
        assert all(torch.isfinite(parameter).all() for parameter in heads[arm].parameters())
        torch.save(dict(arm=arm, seed=1, step=step, expected_step=1000, state_dict=heads[arm].state_dict(),
                        frozen_checkpoint_sha256=file_sha(checkpoint_path),
                        batch_order_sha256=hashlib.sha256(order[:step].astype("<i8").tobytes()).hexdigest(),
                        contract_sha256=file_sha(Path(args.public_dir) / "D_CONTRACT.json"),
                        plan_lock_sha256=file_sha(args.plan_lock)), private / (arm + ".pt"))
    receipt = dict(status="DONE" if step == 1000 else "BUDGET_LIMITED", fits=2,
                   updates_per_arm={arm: step for arm in ARMS}, total_updates=2*step,
                   batch_size=16, seed=1, common_last_step_only=True,
                   unique_exposed_frames=int(len(np.unique(order[:step]))),
                   original_checkpoint_unchanged=file_sha(checkpoint_path)==contract_value["frozen_checkpoint"]["sha256"],
                   frozen_state_unchanged=True, frozen_gradients_absent=True,
                   initial_scorer_sha256=contract_value["shared_initial_sha256"],
                   actual_device_seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), history=history, first_step=first_audit,
                   new_PnP_calls=0, raw_RGB_feature_extractions=0,
                   checkpoints={arm: dict(private_namespace="D/"+arm+".pt", sha256=file_sha(private/(arm+".pt"))) for arm in ARMS})
    write_new(Path(args.public_dir) / "D_TRAINING_RECEIPT.json", receipt)
    print("D_PAIR_COMPLETE", json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["contract", "throughput", "train"])
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--private-dir", required=True)
    parser.add_argument("--public-dir", required=True)
    parser.add_argument("--plan-lock")
    parser.add_argument("--device-seconds-limit", type=float, default=600)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.stage == "contract":
        contract(args)
    elif args.stage == "throughput":
        throughput(args)
    else:
        train_pair(args)


if __name__ == "__main__":
    main()
