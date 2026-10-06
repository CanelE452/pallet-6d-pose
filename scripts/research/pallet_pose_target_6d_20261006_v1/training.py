"""Matched hard ADDsym supervision; all legacy artifacts are read-only.

The network, banks, feature tensors, seed-specific order and optimizer arithmetic
are inherited verbatim. Only the soft 2D cross entropy target is replaced.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch
sys.dont_write_bytecode = True
from .baseline import BASELINE_ROOT
from scripts.research.pallet_joint_action_handoff_20261006_v1.a_common import (
    dcp_env, finite, local_phase, hash_value, read, sha, write)
from scripts.research.pallet_joint_action_handoff_20261006_v1.a_data import Data, network_bank
from scripts.research.pallet_joint_action_handoff_20261006_v1.scorer import (
    JointActionScorer, action_scores)

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "_docs/experiments/pallet_pose_target_6d_20261006_v1"
OLD_DOC = BASELINE_ROOT / "_docs/experiments/pallet_joint_action_handoff_20261006_v1"
OLD_CODE = BASELINE_ROOT / "scripts/research/pallet_joint_action_handoff_20261006_v1"
STEPS = 6000
BATCH = 16


def locked_paths(args):
    protocol = read(DOC / "PROTOCOL.json")
    for key, actual in (("source_root", args.source_root),
                        ("candidate_bank_cache", args.bank_cache),
                        ("pose_cost_cache", args.cost_cache)):
        assert Path(protocol[key]).resolve() == Path(actual).resolve(), f"Locked input path differs: {key}"


def frozen_input_contract(data):
    inputs_path = DOC / "INPUT_BINDINGS.json"
    preflight = read(DOC / "PREFLIGHT.json")
    assert preflight["status"] == "PASS"
    assert preflight["input_bindings_sha256"] == sha(inputs_path)
    inputs = {Path(v["path"]).resolve(): v for v in read(inputs_path)["external_inputs"]}
    paths = [Path(data.arrays[k].filename) for k in ("p3", "p4")]
    paths += [data.dim / "DIMENSION_SIDECAR.npz",
        data.root / "_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json"]
    result = []
    for path in paths:
        entry = inputs[path.resolve()]
        stat = path.stat()
        assert entry["sha256"] == entry["expected_sha256"]
        assert stat.st_size == entry["bytes"] and stat.st_mtime_ns == entry["mtime_ns"], "Frozen input changed after full-byte preflight"
        result.append(dict(path=str(path), sha256=entry["sha256"], bytes=entry["bytes"],
            expected_sha256=entry["expected_sha256"],
            verification="current unchanged size/mtime against this run's completed full-byte SHA audit"))
    return result


def numeric_contract():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


def learning_rate(step):
    progress = (step - 100) / 5900
    return .001 * step / 100 if step <= 100 else .001 * (
        .1 + .9 * .5 * (1 + np.cos(np.pi * progress)))


def forward_bank(head, batch, candidates, valid):
    assert not any(k in batch for k in ("truth", "gt_pose", "oracle_action_index"))
    return head.forward_bank(
        *(batch[k] for k in ("p3", "p4", "points", "boxes", "point_valid", "input_shape")),
        context=batch["context"], candidate_points=candidates, action_valid=valid)


def hard_pose_loss(output, target):
    """First argmin is provided by the cost cache; -1 means all F failed.

    No target is reassigned and no score support is changed. A nonzero oracle
    with zero inference support cannot be represented by the inherited decoder.
    This fails closed rather than inventing an alternative supervision rule.
    """
    target = target.to(device=output["logits"].device, dtype=torch.long)
    eligible = target >= 0
    scores = action_scores(output)
    if eligible.any():
        chosen = target[eligible]
        assert (chosen < scores.shape[1]).all()
        assert output["action_valid"][eligible, chosen].all()
        incompatible = (output["point_support"].sum(-1)[eligible] == 0) & (chosen != 0)
        assert not incompatible.any(), "Pose target inaccessible under inherited support0 NoOp decoder"
        value = torch.nn.functional.cross_entropy(scores[eligible], chosen)
    else:
        # Retain a differentiable zero without evaluating padded -inf logits.
        value = torch.where(torch.isfinite(output["logits"]), output["logits"],
                            torch.zeros_like(output["logits"])).sum() * 0
    return value, dict(excluded_frames=int((~eligible).sum()), eligible_frames=int(eligible.sum()))


ACTION_FIT_KEYS = (
    "observed_exposures", "eligible_target_exposures", "excluded_target_exposures",
    "matching_eligible_exposures", "moving_target_exposures", "matching_moving_target_exposures",
    "NoOp_target_exposures", "matching_NoOp_target_exposures")


def empty_action_fit():
    return dict({key: 0 for key in ACTION_FIT_KEYS}, eligible_CE_sum=0.)


@torch.no_grad()
def observed_action_fit(output, target, value):
    """Read the existing pre-update forward; never run another model pass.

    These are repeated ordered training exposures under changing checkpoints,
    not final-checkpoint accuracy over unique TRAIN frames.
    """
    target = target.detach().to(device=output["logits"].device, dtype=torch.long)
    chosen = action_scores(output).detach().argmax(-1)
    eligible, moving, noop = target >= 0, target > 0, target == 0
    matching = chosen == target
    return dict(observed_exposures=int(target.numel()),
        eligible_target_exposures=int(eligible.sum()), excluded_target_exposures=int((~eligible).sum()),
        matching_eligible_exposures=int((matching & eligible).sum()),
        moving_target_exposures=int(moving.sum()), matching_moving_target_exposures=int((matching & moving).sum()),
        NoOp_target_exposures=int(noop.sum()), matching_NoOp_target_exposures=int((matching & noop).sum()),
        eligible_CE_sum=float(value.detach()) * int(eligible.sum()))


def accumulate_action_fit(total, observation):
    for key in (*ACTION_FIT_KEYS, "eligible_CE_sum"):
        total[key] += observation[key]


def action_fit_summary(counts):
    result = dict(counts)
    for name, numerator, denominator in (
        ("eligible_action_accuracy", "matching_eligible_exposures", "eligible_target_exposures"),
        ("moving_target_action_accuracy", "matching_moving_target_exposures", "moving_target_exposures"),
        ("NoOp_target_action_accuracy", "matching_NoOp_target_exposures", "NoOp_target_exposures"),
        ("eligible_mean_CE", "eligible_CE_sum", "eligible_target_exposures")):
        result[name] = counts[numerator] / counts[denominator] if counts[denominator] else None
    result["scope"] = "existing pre-update training forwards; ordered repeated exposures under changing checkpoints; not final whole-TRAIN accuracy or independent underfitting evidence"
    result["additional_model_forwards"] = 0
    result["additional_optimizer_updates"] = 0
    return result


class ReadOnlyBanks:
    def __init__(self, data, directory):
        self.data = data
        self.directory = Path(directory)
        self.points = np.load(self.directory / "source_banks.npy", mmap_mode="r")
        self.counts = np.load(self.directory / "source_counts.npy", mmap_mode="r")
        self.hypotheses = np.load(self.directory / "source_hypothesis.npy", mmap_mode="r")
        self.names = read(self.directory / "BANK_NAMES.json")
        self.binding = read(self.directory / "BANK_BINDING.json")["binding"]
        expected = read(OLD_DOC / "A_manifest.json")
        assert self.binding == expected["bank_binding"]
        saved = {v["name"]: v for v in expected["cache_files"]}
        for name in ("source_banks.npy", "source_counts.npy", "source_hypothesis.npy"):
            path = self.directory / name
            assert path.stat().st_size == saved[name]["bytes"]
            assert sha(path) == saved[name]["sha256"], f"Old bank bytes changed: {name}"
        self.bank_sha256 = saved["source_banks.npy"]["sha256"]
        guard = read(self.directory / "GENERATION_CODE_BINDINGS.json")
        for rel, digest in guard["code_bindings"].items():
            assert sha(BASELINE_ROOT / rel) == digest, f"Old bank generation code changed: {rel}"

    def get(self, row, frame=None):
        count = int(self.counts[row])
        assert 1 <= count <= 201
        bank = dict(points=np.array(self.points[row, :count], copy=True),
                    hypotheses=[self.names[int(x)] for x in self.hypotheses[row, :count]])
        frame = self.data.source_frame(row) if frame is None else frame
        assert np.array_equal(bank["points"][0], frame["q"], equal_nan=True), "Bank/source frozen NoOp differs"
        assert np.array_equal(bank["points"][:, 8], np.broadcast_to(frame["q"][8], (count, 2)), equal_nan=True), "Bank centre differs from frozen RAW"
        return bank

    def tensor_batch(self, rows, batch, device="cuda"):
        maximum = max(2, max(int(self.counts[r]) for r in rows))
        points, valid = [], []
        for i, row in enumerate(rows):
            frame = self.data.source_frame(row)
            bank = self.get(row, frame)
            q = network_bank(bank, frame, batch["points"][i].detach().cpu().numpy())
            padded = np.repeat(q[:1], maximum, axis=0)
            padded[:len(q)] = q
            points.append(padded)
            valid.append(np.arange(maximum) < len(q))
        return torch.from_numpy(np.stack(points)).to(device), torch.from_numpy(np.stack(valid)).to(device)


class ReadOnlyTargets:
    def __init__(self, data, directory):
        self.directory = Path(directory)
        self.header = read(self.directory / "cache_header.json")
        manifest = read(DOC / "POSE_COST_CACHE_MANIFEST.json")
        assert manifest["status"] in ("PASS", "DONE", "COMPLETE"), "Whole TRAIN cost cache not complete"
        assert manifest["binding"] == self.header["binding"]
        assert self.header["protocol_sha256"] == sha(DOC / "PROTOCOL.json")
        assert manifest["header_sha256"] == sha(self.directory / "cache_header.json")
        assert all("sha256" in v for v in manifest["files"]), "Whole cache byte bindings absent"
        for artifact in manifest["files"]:
            path = self.directory / artifact["file"]
            assert path.stat().st_size == artifact["bytes"] and sha(path) == artifact["sha256"]
        self.cost = np.load(self.directory / "cost_ADDsym_m.npy", mmap_mode="r")
        self.index = np.load(self.directory / "oracle_index.npy", mmap_mode="r")
        self.done = np.load(self.directory / "done.npy", mmap_mode="r")
        self.usable = np.load(self.directory / "usable_train.npy", mmap_mode="r")
        row_state = np.load(self.directory / "row_state.npy", mmap_mode="r")
        assert self.cost.shape == (len(data.indices), 201)
        assert np.array_equal(np.flatnonzero(self.usable), data.train_rows)
        assert len(data.train_rows) == 55915 and self.done[data.train_rows].all()
        assert (row_state[data.train_rows] == 2).all(), "Incomplete TRAIN rows"
        assert np.all(self.index[data.train_rows] >= -1)
        # First index on ties, failures remain +inf, all-F-invalid remain -1.
        for start in range(0, len(data.train_rows), 2048):
            rows = data.train_rows[start:start + 2048]
            costs = np.array(self.cost[rows])
            exists = np.isfinite(costs).any(-1)
            expected = np.where(exists, costs.argmin(-1), -1)
            assert np.array_equal(self.index[rows], expected), "Cost/target candidate index mismatch"
        self.manifest_sha256 = sha(DOC / "POSE_COST_CACHE_MANIFEST.json")
        self.header_sha256 = sha(self.directory / "cache_header.json")
        self.content_sha256 = hash_value(dict(binding=self.header["binding"], files=manifest["files"]))


def initialize(seed, config, device="cpu"):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return JointActionScorer(5, **config).to(device)


def matched_contract(data, banks, seed, config, head):
    old = read(OLD_DOC / f"A_fits/GEO_seed{seed}.json")
    order_path = data.root / f"data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/order_seed{seed}.npy"
    order = np.load(order_path, mmap_mode="r")
    digest = dcp_env.state_sha(head.state_dict())
    assert digest == old["first_step"]["initial_state_sha256"], "Seed initialization digest mismatch"
    assert order.shape == (6000, 16) and np.isin(order, data.train_rows).all()
    assert sha(order_path) == old["order_sha256"]
    params = sum(p.numel() for p in head.parameters())
    assert params == old["params"] == 20259
    contract = dict(
        candidate_bank_sha256=banks.bank_sha256, bank_binding=banks.binding,
        architecture_sha256=sha(OLD_CODE / "scorer.py"), parameter_count=params,
        feature_source="unchanged original frozen p3/p4 source cache",
        dimension_input="unchanged original five-dimensional normalized context",
        feature_and_dimension_bindings=frozen_input_contract(data),
        optimizer=dict(name="AdamW", lr=.001, weight_decay=.0001, betas=[.9, .999]),
        schedule=dict(warmup=100, cosine_steps=5900, final_lr_fraction=.1),
        gradient_clipping=5, batch=16, updates=6000, config=config,
        order_path=str(order_path), order_sha256=old["order_sha256"],
        initial_state_sha256=digest, inference_readout="unchanged hard joint J/action_scores",
        final_F_code_sha256=sha(BASELINE_ROOT / "scripts/research/pallet_dim_conditioned_p_v1/pose.py"),
        numeric=dict(dtype="FP32", TF32_matmul=False, TF32_cudnn=True, cudnn_benchmark=False),
        unchanged_unused_parameters="All original 20259 parameters retained, including inherited radial API/heads; no architecture pruning",
        only_changed_component="training supervision: soft2D target CE -> hard final ADDsym argmin target CE",
        old_loss_regularizers="None: original joint_loss is exclusively soft target cross entropy")
    shared = {key: value for key, value in contract.items() if key not in (
        "only_changed_component", "old_loss_regularizers", "unchanged_unused_parameters")}
    contract["OLD"] = dict(shared, training_target="soft2D_target_CE")
    contract["NEW"] = dict(shared, training_target="hard_final_ADDsym_argmin_CE")
    contract["changed_components"] = ["training_target"]
    contract["initial_state_sha_equal"] = True
    contract["order_sha_equal"] = True
    return contract, order


def inference_support_without_features(head, batch, candidates, valid):
    """Literal original support arithmetic, no RGB/feature/scorer forward."""
    points, boxes = batch["points"], batch["boxes"]
    input_shape = batch["input_shape"]
    point_valid = finite(points, batch["point_valid"])
    box_valid = torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > boxes[:, :2]).all(-1)
    safe = torch.where(point_valid[..., None], points, torch.zeros_like(points))
    bb = torch.where(box_valid[:, None], boxes, boxes.new_tensor([0, 0, 1, 1]))
    diag = (bb[:, 2:] - bb[:, :2]).norm(dim=-1).clamp_min(1)
    safe_q = torch.where(point_valid[:, None, :, None], candidates, safe[:, None])
    locations = safe_q[:, 1:, :8].transpose(1, 2)
    positions = locations[:, :, :, None] + diag[:, None, None, None, None] * head.stencil_fraction * head.stencil[None, None, None]
    inside = point_valid[:, :8, None, None] & (positions[..., 0] >= 0) & (positions[..., 1] >= 0) & (positions[..., 0] < input_shape[:, None, None, None, 1]) & (positions[..., 1] < input_shape[:, None, None, None, 0])
    support = point_valid[:, :8] & box_valid[:, None] & (inside.any(-1) & valid[:, None, 1:]).any(-1)
    return support, diag, point_valid


def teacher_2d(output, batch):
    gt, gt_valid, branch, _ = local_phase(output["points_raw"], output["point_valid"],
        batch["gt_points"], batch["gt_valid"], batch["permutations"],
        batch["group_valid"], output["box_diagonal"])
    mask = finite(gt, gt_valid)[:, :8] & output["point_support"]
    count = mask.sum(-1)
    error = (output["candidate_points"][:, :, :8] - gt[:, None, :8]).square().sum(-1)
    error = torch.where(mask[:, None], error, torch.zeros_like(error)).sum(-1) / count.clamp_min(1)[:, None]
    sigma = output["box_diagonal"] * .08 / 17
    probability = (-error / (2 * sigma[:, None].square())).masked_fill(~output["action_valid"], float("-inf")).softmax(-1)
    index = probability.argmax(-1)
    return torch.where(count > 0, index, index.new_full(index.shape, -1)), count, branch


@torch.no_grad()
def auxiliary(data, banks, targets, args, config):
    dest = Path(args.cost_cache)
    out = DOC / "TRAIN_2D_6D_TARGET_COMPARISON.json"
    binding = hash_value(dict(protocol=sha(DOC / "PROTOCOL.json"), code=sha(Path(__file__)),
        target_header=targets.header_sha256, bank=banks.bank_sha256))
    auxiliary_guard = dest / "teacher_2d_binding.json"
    if auxiliary_guard.exists():
        assert read(auxiliary_guard)["binding"] == binding, "Partial auxiliary target code/input binding changed"
    else:
        write(auxiliary_guard, dict(binding=binding, scope="pure FP32 CUDA target arithmetic, no feature/CNN forward"))
    if out.exists():
        prior = read(out)
        assert prior["binding"] == binding
        for artifact in prior["array_artifacts"]:
            assert sha(Path(artifact["path"])) == artifact["sha256"]
        return prior
    n = len(data.indices)
    arrays = {}
    for key, dtype, fill in [("teacher_2d_index", "int16", -1), ("teacher_2d_count", "uint8", 0),
                              ("teacher_2d_support_count", "uint8", 0), ("teacher_2d_done", "bool", False)]:
        path = dest / f"{key}.npy"
        existed = path.exists()
        arrays[key] = np.lib.format.open_memmap(path, mode="r+" if existed else "w+", dtype=dtype, shape=(n,))
        if not existed: arrays[key][:] = fill
        assert arrays[key].shape == (n,) and arrays[key].dtype == np.dtype(dtype)
    head = initialize(1, config, "cuda")
    started = time.monotonic()
    calculated = 0
    # p3/p4 never loaded by this target-only pass.
    rows_to_do = data.train_rows[~arrays["teacher_2d_done"][data.train_rows]]
    for start in range(0, len(rows_to_do), 16):
        rows = rows_to_do[start:start + 16]
        batch = {key: torch.from_numpy(np.array(data.arrays[key][rows], copy=True)).cuda()
                 for key in ("points", "boxes", "point_valid", "input_shape", "gt_points", "gt_valid")}
        for key in ("permutations", "group_valid"):
            batch[key] = torch.from_numpy(data.side[key][rows].copy()).cuda()
        q, valid = banks.tensor_batch(rows, batch)
        support, diag, pv = inference_support_without_features(head, batch, q, valid)
        output = dict(points_raw=batch["points"], point_valid=pv, point_support=support,
                      box_diagonal=diag, candidate_points=q, action_valid=valid)
        index, count, _ = teacher_2d(output, batch)
        arrays["teacher_2d_index"][rows] = index.cpu().numpy()
        arrays["teacher_2d_count"][rows] = count.cpu().numpy()
        arrays["teacher_2d_support_count"][rows] = support.sum(-1).cpu().numpy()
        # Commit values before done flags, so an interrupted auxiliary pass
        # cannot reuse a flag whose target/mask bytes were never durable.
        for key in ("teacher_2d_index", "teacher_2d_count", "teacher_2d_support_count"):
            arrays[key].flush()
        arrays["teacher_2d_done"][rows] = True
        arrays["teacher_2d_done"].flush()
        calculated += len(rows)
        if start % 2048 == 0:
            for array in arrays.values(): array.flush()
            print("TARGET_AUX", calculated, len(rows_to_do), flush=True)
    for array in arrays.values(): array.flush()
    rows = data.train_rows
    assert arrays["teacher_2d_done"][rows].all()
    two = np.asarray(arrays["teacher_2d_index"][rows])
    six = np.asarray(targets.index[rows])
    common = (two >= 0) & (six >= 0)
    incompatible = (arrays["teacher_2d_support_count"][rows] == 0) & (six > 0)
    result = dict(schema="pose_target_2d_6d_TRAIN_aux_v1", status="PASS" if not incompatible.any() else "BLOCKED_INTEGRITY",
        binding=binding, full_usable_TRAIN=len(rows), common_defined=int(common.sum()),
        undefined_2d=int((two < 0).sum()), all_F_invalid_6d=int((six < 0).sum()),
        exact_same_index=int(((two == six) & common).sum()),
        both_NoOp=int(((two == 0) & (six == 0) & common).sum()),
        two_d_NoOp_six_d_move=int(((two == 0) & (six > 0)).sum()),
        two_d_move_six_d_NoOp=int(((two > 0) & (six == 0)).sum()),
        both_move_different_action=int(((two > 0) & (six > 0) & (two != six)).sum()),
        unsupported_nonzero_pose_target_rows=rows[incompatible].tolist(),
        new_refiner_forward_calls=0, new_backbone_calls=0, new_F_calls=0, optimizer_updates=0,
        target_arithmetic_frames_this_invocation=calculated, seconds=time.monotonic() - started,
        target_scope="original literal FP32 phase/mask/error/softmax first argmax; GPU coordinate/support math only, no CNN or logits",
        array_artifacts=[dict(path=str(dest / f"{key}.npy"), sha256=sha(dest / f"{key}.npy"),
                             bytes=(dest / f"{key}.npy").stat().st_size) for key in arrays])
    result["exact_same_index_fraction"] = result["exact_same_index"] / result["common_defined"] if result["common_defined"] else None
    write(out, result)
    return result


def parity(data, banks, config):
    result = []
    for seed in (1, 2, 3):
        head = initialize(seed, config)
        contract, order = matched_contract(data, banks, seed, config, head)
        result.append(dict(seed=seed, status="PASS", contract=contract,
                           first_batch_source_rows=order[0].tolist()))
    out = dict(schema="pose_target_initialization_order_parity_v1", status="PASS",
               seeds=result, new_model_forward_calls=0, optimizer_updates=0)
    write(DOC / "TRAIN_INITIALIZATION_PARITY.json", out)
    return out


def fit(data, banks, targets, args, config):
    seed = args.seed
    if seed != 1:
        decision = read(DOC / "SEED1_SCREENING.json")
        assert decision["decision"] in ("CONTINUE", "MIXED_CONTINUE"), "seed2/3 not authorized by locked synthetic screen"
        assert decision["continue_seeds"] is True
    aux = read(DOC / "TRAIN_2D_6D_TARGET_COMPARISON.json")
    assert aux["status"] == "PASS" and aux["full_usable_TRAIN"] == 55915
    dest = Path(args.cost_cache) / "fits" / f"POSE_TARGET_GEO_seed{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    path = DOC / "fits" / f"POSE_TARGET_GEO_seed{seed}.json"
    old = read(OLD_DOC / "A_protocol.json")
    binding = hash_value(dict(protocol=sha(DOC / "PROTOCOL.json"), code=sha(Path(__file__)),
        old_training_code={f: sha(OLD_CODE / f) for f in ("scorer.py", "a_data.py", "a_common.py")},
        bank=banks.bank_sha256, cost_content=targets.content_sha256, cost_header=targets.header_sha256,
        auxiliary=sha(DOC / "TRAIN_2D_6D_TARGET_COMPARISON.json")))
    ckpath = dest / "last.pt"
    if path.exists():
        prior = read(path)
        assert prior["complete"] and prior["binding"] == binding
        assert sha(ckpath) == prior["checkpoint_sha256"]
        return prior
    head = initialize(seed, config, "cuda").train()
    contract, order = matched_contract(data, banks, seed, config, head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001, betas=(.9, .999))
    start, prior_seconds, history, excluded = 0, 0., [], 0
    action_fit_total, action_fit_window, action_fit_window_start = empty_action_fit(), empty_action_fit(), 1
    final_action_fit_window = None
    if ckpath.exists():
        ck = torch.load(ckpath, map_location="cpu", weights_only=False)
        assert ck["binding"] == binding and ck["order_sha256"] == contract["order_sha256"]
        head.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
        start, prior_seconds = ck["step"], ck["seconds"]
        history, excluded = ck["history"], ck["excluded_exposures"]
        action_fit_total = ck["training_action_fit_totals"]
        action_fit_window = ck["training_action_fit_window"]
        action_fit_window_start = ck["training_action_fit_window_start_step"]
        final_action_fit_window = ck["final_training_action_fit_window"]
        assert action_fit_total["observed_exposures"] == start * 16
    attempts_path = dest / "ATTEMPT_STATE.json"
    if attempts_path.exists():
        state = read(attempts_path)
        assert state["binding"] == binding and state["completed_updates"] == start
        assert state["status"] == "CHECKPOINTED", "Interrupted update lacks durable checkpoint; do not silently replay"
    began = time.monotonic()
    teacher_index = np.load(Path(args.cost_cache) / "teacher_2d_index.npy", mmap_mode="r")
    teacher_support_count = np.load(Path(args.cost_cache) / "teacher_2d_support_count.npy", mmap_mode="r")
    for step in range(start + 1, 6001):
        rows = order[step - 1]
        write(attempts_path, dict(binding=binding, status="FORWARD_ATTEMPT", attempted_step=step,
                                 completed_updates=step - 1, completed_exposures=(step - 1) * 16))
        batch = data.batch(rows)
        candidates, valid = banks.tensor_batch(rows, batch)
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step)
        for group in optimizer.param_groups: group["lr"] = lr
        output = forward_bank(head, batch, candidates, valid)
        if step == 1:
            two, _, _ = teacher_2d(output, batch)
            assert np.array_equal(two.detach().cpu().numpy(), teacher_index[rows])
            assert np.array_equal(output["point_support"].sum(-1).cpu().numpy(), teacher_support_count[rows])
            assert np.array_equal(output["point_support"].cpu().numpy(),
                                  inference_support_without_features(head, batch, candidates, valid)[0].cpu().numpy())
        target = torch.from_numpy(np.array(targets.index[rows], copy=True)).cuda().long()
        value, audit = hard_pose_loss(output, target)
        assert torch.isfinite(value), "Non-finite hard pose supervision"
        observation = observed_action_fit(output, target, value)
        assert observation["eligible_target_exposures"] == audit["eligible_frames"]
        assert observation["excluded_target_exposures"] == audit["excluded_frames"]
        accumulate_action_fit(action_fit_total, observation)
        accumulate_action_fit(action_fit_window, observation)
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
        if step == 1:
            gradients = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in head.named_parameters()}
            assert gradients["adapt3.0.weight"] > 0 and gradients["adapt4.0.weight"] > 0
            assert dcp_env.state_sha(head.state_dict()) == contract["initial_state_sha256"]
            write(dest / "FIRST_STEP.json", dict(gradients=gradients, params=20259,
                actual_source_rows=rows.tolist(), real_access=False, initial_state_sha256=contract["initial_state_sha256"],
                hard_pose_target_indices=target.cpu().tolist(), auxiliary_mask_and_2d_teacher_bit_parity=True,
                observed_preupdate_action_fit=action_fit_summary(observation)))
        write(attempts_path, dict(binding=binding, status="OPTIMIZER_ATTEMPT", attempted_step=step,
                                 completed_updates=step - 1, completed_exposures=(step - 1) * 16))
        optimizer.step()
        excluded += audit["excluded_frames"]
        elapsed = prior_seconds + time.monotonic() - began
        if step == 1 or step % 100 == 0:
            record = dict(step=step, loss=float(value.detach()), lr=float(lr), gradient_norm=float(norm), seconds=elapsed,
                cumulative_observed_preupdate_action_fit=action_fit_summary(action_fit_total),
                window_observed_preupdate_action_fit=dict(first_step=action_fit_window_start, last_step=step,
                    **action_fit_summary(action_fit_window)))
            history.append(record)
            write(DOC / f"TRAIN_PROGRESS_seed{seed}.json", dict(seed=seed, **record))
            print("POSE_TARGET_FIT", seed, step, 6000, round(record["loss"], 6), round(elapsed, 1), flush=True)
        if step % 100 == 0:
            final_action_fit_window = dict(first_step=action_fit_window_start, last_step=step,
                **action_fit_summary(action_fit_window))
            action_fit_window, action_fit_window_start = empty_action_fit(), step + 1
        # Durable state after each update prevents hidden repeated updates on
        # resume. This I/O cadence affects elapsed time, not optimizer arithmetic.
        ck = dict(step=step, complete=step == 6000, binding=binding,
            order_sha256=contract["order_sha256"], seed=seed, config=config,
            model_state_dict=head.state_dict(), optimizer=optimizer.state_dict(),
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            seconds=elapsed, history=history, excluded_exposures=excluded,
            training_action_fit_totals=action_fit_total,
            training_action_fit_window=action_fit_window,
            training_action_fit_window_start_step=action_fit_window_start,
            final_training_action_fit_window=final_action_fit_window)
        pending = dest / "last.pending.pt"
        torch.save(ck, pending)
        pending.replace(ckpath)
        write(attempts_path, dict(binding=binding, status="CHECKPOINTED", attempted_step=step,
                                 completed_updates=step, completed_exposures=step * 16))
    result = dict(schema="pose_target_matched_formal_fit_v1", complete=True, status="DONE",
        seed=seed, arm="GEO", method="POSE_TARGET_GEO_J", smoke=False,
        updates=6000, exposures=96000, excluded_target_exposures=excluded,
        checkpoint_path=str(ckpath), checkpoint_sha256=sha(ckpath), binding=binding,
        order_sha256=contract["order_sha256"], seconds=prior_seconds + time.monotonic() - began,
        history=history, final_checkpoint_only=True, real_training=0,
        first_step=read(dest / "FIRST_STEP.json"), params=20259, matched_contract=contract,
        torch=torch.__version__, TF32_matmul=False, TF32_cudnn=True, auxiliary_target_CNN_forwards=0,
        operational_difference="checkpoint every update to preserve exact actual-update accounting; OLD wrote every500; model/optimizer arithmetic unchanged",
        standalone_smoke_optimizer_updates=0, code_sha256=sha(Path(__file__)))
    assert action_fit_total["observed_exposures"] == 96000
    assert action_fit_total["excluded_target_exposures"] == excluded
    assert action_fit_total["eligible_target_exposures"] + excluded == 96000
    result["observed_preupdate_training_action_fit"] = action_fit_summary(action_fit_total)
    result["final_observed_preupdate_training_window"] = final_action_fit_window
    result["cost_manifest_execution_sha256"] = targets.manifest_sha256
    result["cost_content_sha256"] = targets.content_sha256
    write(path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bank-cache", type=Path, required=True)
    parser.add_argument("--cost-cache", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--stage", choices=("parity", "aux", "fit"), required=True)
    args = parser.parse_args(argv)
    assert (DOC / "PROTOCOL.json").exists(), "Root protocol must be locked first"
    locked_paths(args)
    numeric_contract()
    data = Data(args.source_root)
    banks = ReadOnlyBanks(data, args.bank_cache)
    config = read(OLD_DOC / "A_protocol.json")["config"]
    if args.stage == "parity": result = parity(data, banks, config)
    else:
        targets = ReadOnlyTargets(data, args.cost_cache)
        result = auxiliary(data, banks, targets, args, config) if args.stage == "aux" else fit(data, banks, targets, args, config)
    print(json.dumps(dict(stage=args.stage, status=result["status"]), ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
