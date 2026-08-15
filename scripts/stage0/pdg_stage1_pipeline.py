#!/usr/bin/env python
"""Stage-1 autonomous runner.

Runs the whole chain outside any interactive session: wrapper parity, gradient
calibration, smoke, two fixed 3-epoch trainings, development evaluation and the
foundation gate.  State is a single atomic JSON so the same command can be
re-issued and picks up at the first incomplete phase.

A holdout guard wraps every image reader in the process.  If any path under the
sealed evaluation sets is opened, the run stops as HARD_BLOCKED rather than
producing a number nobody may look at.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import sys
import time
import traceback

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT / "scripts/stage0", ROOT / "Deep_Object_Pose/common",
           ROOT / "Deep_Object_Pose/train", ROOT / "challenge/scripts",
           ROOT / "scripts/data_prep/eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/pdg_unified_program")
RUN = OUT / "stage1_runner"
STATE = RUN / "state.json"
LOCK = RUN / "pipeline.lock"
PIDFILE = RUN / "pipeline.pid"
WEIGHTS = ROOT / "weights/paper_s2/paper_s2_pdg"
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"
SEED = 1
EPOCHS = 3
SMOKE_STEPS = 100
CALIB_BATCHES = 8
CALIB_SLICE = 4
WARMUP_STEPS = 50
TARGET_RATIO = {"palletness": 0.10, "visibility": 0.15, "truncation": 0.05}
LAMBDA_CLAMP = (1e-5, 10.0)
PHASES = ["PREPARE", "PARITY", "CALIBRATE", "SMOKE_A1", "SMOKE_A2",
          "TRAIN_A1", "TRAIN_A2", "EVALUATE", "DECIDE", "REPORT"]

SEALED_TOKENS = ("_outside_eval_manual_gt", "capture0403noapril_manual_gt",
                 "capturepalletcad_manual_gt", "wood_pallet_20260618",
                 "capturenight08", "capturenight09", "capturepallet07",
                 "capturepallet09", "testset_full8_manifest", "handannot17")
HOLDOUT_HITS = {"e44": 0, "w45": 0, "final_test": 0, "paths": []}


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


# ---------------------------------------------------------------- guard
def install_holdout_guard(eval_paths: set[str]) -> None:
    """Every image read passes through here.  Reading membership JSON is fine;
    opening a sealed image is not."""
    import cv2
    from PIL import Image

    def check(path) -> None:
        text = str(path)
        if not text.lower().endswith((".png", ".jpg", ".jpeg")):
            return
        resolved = os.path.abspath(text)
        if resolved in eval_paths:
            HOLDOUT_HITS["e44" if "wood_pallet" not in resolved else "w45"] += 1
            HOLDOUT_HITS["paths"].append(resolved)
            raise SystemExit(f"HARD_BLOCKED: sealed holdout image opened {resolved}")
        for token in SEALED_TOKENS[4:]:
            if token in resolved:
                HOLDOUT_HITS["final_test"] += 1
                HOLDOUT_HITS["paths"].append(resolved)
                raise SystemExit(f"HARD_BLOCKED: final-test image opened {resolved}")

    original_imread, original_open = cv2.imread, Image.open

    def guarded_imread(path, *a, **k):
        check(path)
        return original_imread(path, *a, **k)

    def guarded_open(path, *a, **k):
        check(path)
        return original_open(path, *a, **k)

    cv2.imread = guarded_imread
    Image.open = guarded_open


def sealed_image_paths() -> set[str]:
    paths = set()
    for name in ("eval56_manifest.json", "wood_manifest.json"):
        manifest = OUT.parent.parent.parent / name
        if manifest.is_file():
            payload = json.loads(manifest.read_text("utf-8"))
            for frame in payload["frames"]:
                paths.add(os.path.abspath(frame["image_path"]))
    return paths


# ---------------------------------------------------------------- state
def atomic_write(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
    os.replace(temporary, path)


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text("utf-8"))
    return {"phases": {name: {"status": "PENDING"} for name in PHASES},
            "expected_head": "199e1af7b5a3ceabffe66227f152ce420891c133",
            "ep57_sha": EP57_SHA, "holdout": {"e44_open": 0, "w45_open": 0,
                                              "final_test_open": 0},
            "optimizer_steps": 0, "final_gate": None, "last_error": None}


def set_phase(state: dict, name: str, status: str, **extra) -> None:
    entry = state["phases"].setdefault(name, {})
    entry["status"] = status
    entry.update(extra)
    entry["stamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["holdout"] = {"e44_open": HOLDOUT_HITS["e44"],
                        "w45_open": HOLDOUT_HITS["w45"],
                        "final_test_open": HOLDOUT_HITS["final_test"]}
    atomic_write(STATE, state)


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- losses
def belief_affinity_loss(result, batch, device):
    belief_target = batch["beliefs"].to(device).float()
    belief_mask = batch["belief_channel_mask"].to(device).float()[:, :, None, None]
    affinity_target = batch["affinities"].to(device).float()
    affinity_mask = batch["affinity_channel_mask"].to(device).float()[:, :, None, None]
    total = torch.zeros((), device=device)
    for stage in result["beliefs"]:
        total = total + (((stage[:, :9] - belief_target) ** 2) * belief_mask).mean()
    for stage in result["affinities"]:
        total = total + (((stage[:, :16] - affinity_target) ** 2) * affinity_mask).mean()
    return total


def focal_bce(logits, target, gamma=2.0, alpha=0.5):
    probability = torch.sigmoid(logits)
    weight = alpha * target + (1 - alpha) * (1 - target)
    modulator = (target * (1 - probability) + (1 - target) * probability) ** gamma
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none")
    return (weight * modulator * bce).mean()


def head_losses(result, batch, device, class_weight, alpha):
    palletness = batch["pdg_palletness"].to(device).float()
    visibility = batch["pdg_visibility"].to(device).long()
    visibility_mask = batch["pdg_visibility_mask"].to(device).float()
    truncated = batch["pdg_truncated"].to(device).float()
    l_pallet = focal_bce(result["palletness"], palletness, 2.0, alpha)
    logits = result["visibility"].reshape(-1, 3)
    labels = visibility.reshape(-1)
    per_element = torch.nn.functional.cross_entropy(
        logits, labels, weight=class_weight.to(device), reduction="none")
    denominator = visibility_mask.reshape(-1).sum().clamp_min(1.0)
    l_visibility = (per_element * visibility_mask.reshape(-1)).sum() / denominator
    l_trunc = torch.nn.functional.binary_cross_entropy_with_logits(
        result["truncation"], truncated)
    return {"palletness": l_pallet, "visibility": l_visibility,
            "truncation": l_trunc}


def make_optimizer(model):
    local, heads = model.trainable_groups()
    groups = [{"params": local, "lr": 5e-5}]
    if heads:
        groups.append({"params": heads, "lr": 2e-4})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def build_arm_loader(arm: str):
    screen = load_module("screen",
                         "scripts/stage0/paper_s2/paper_s2_corner_replacement_screen.py")
    import pdg_stage1_dataset as DS
    options = screen.canonical_options()
    dataset, loader, _, _ = DS.build(arm, options, taca_seed=SEED)
    return dataset, loader


# ---------------------------------------------------------------- phases
def phase_prepare(state):
    RUN.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        (ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth").read_bytes()).hexdigest()
    if digest != EP57_SHA:
        raise SystemExit("HARD_BLOCKED: ep57 SHA mismatch")
    state["current_head"] = os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()
    set_phase(state, "PREPARE", "DONE", ep57_sha=digest)


def phase_parity(state):
    import pdg_stage1_dataset as DS
    import pdg_stage1_model as MODEL
    screen = load_module("screen",
                         "scripts/stage0/paper_s2/paper_s2_corner_replacement_screen.py")
    options = screen.canonical_options()
    parity_ds, _, _, _ = DS.build("PARITY", options)
    a1_ds, _, _, _ = DS.build("A1", options)
    canonical_ds, _, _, _ = screen.build_loader(options)
    worst = {"img": 0.0, "kp": 0.0, "belief": 0.0, "affinity": 0.0, "mask": 0.0}
    corner_delta = 0.0
    indices = [int(i * len(parity_ds) / 32) for i in range(32)]
    for index in indices:
        seed_all(0)
        parity = parity_ds[index]
        seed_all(0)
        canonical = canonical_ds[index]
        seed_all(0)
        a1 = a1_ds[index]
        worst["img"] = max(worst["img"], float((parity["img"] - canonical["img"]).abs().max()))
        worst["kp"] = max(worst["kp"], float((parity["refine_keypoints"] - canonical["refine_keypoints"]).abs().max()))
        worst["belief"] = max(worst["belief"], float((parity["beliefs"] - canonical["beliefs"]).abs().max()))
        worst["affinity"] = max(worst["affinity"], float((parity["affinities"] - canonical["affinities"]).abs().max()))
        worst["mask"] = max(worst["mask"], float((parity["belief_channel_mask"] - canonical["belief_channel_mask"]).abs().max()))
        corner_delta = max(corner_delta, float((a1["beliefs"][:8] - parity["beliefs"][:8]).abs().max()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models import DopeNetwork
    base = DopeNetwork(numSeg=1)
    base.load_state_dict(torch.load(str(MODEL.EP57), map_location="cpu",
                                    weights_only=True), strict=True)
    base = base.to(device).eval()
    sample = torch.randn(2, 3, 400, 400, device=device)
    with torch.no_grad():
        reference = base(sample)
    step0 = {}
    for arm in ("A1", "A2"):
        model = MODEL.PDGStage1Model(arm).to(device).eval()
        with torch.no_grad():
            out = model(sample)
        step0[arm] = {"h6": float((out["h6"] - reference[0][-1]).abs().max()),
                      "a6": float((out["a6"] - reference[1][-1]).abs().max())}
        del model
        torch.cuda.empty_cache()
    payload = {"wrapper": worst, "a1_corner_delta": corner_delta, "step0": step0}
    atomic_write(OUT / "wrapper_parity.json", payload)
    ok = (worst["img"] <= 1e-6 and worst["kp"] <= 1e-6 and worst["belief"] <= 1e-6
          and worst["affinity"] <= 1e-6 and corner_delta <= 1e-6
          and all(v["h6"] <= 1e-8 and v["a6"] <= 1e-8 for v in step0.values()))
    if not ok:
        set_phase(state, "PARITY", "HARD_BLOCKED", detail=payload)
        raise SystemExit(f"HARD_BLOCKED: wrapper parity failed {payload}")
    set_phase(state, "PARITY", "DONE", detail=payload)


def _class_statistics(loader, batches=20):
    """Focal alpha and visibility class weights, from the training split only."""
    positives = total = 0.0
    counts = np.zeros(3, dtype=np.float64)
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        hull = batch["pdg_palletness"].numpy()
        positives += float(hull.sum())
        total += float(hull.size)
        labels = batch["pdg_visibility"].numpy().reshape(-1)
        mask = batch["pdg_visibility_mask"].numpy().reshape(-1) > 0
        for state_index in range(3):
            counts[state_index] += float(((labels == state_index) & mask).sum())
    alpha = float(np.clip(1.0 - positives / max(total, 1.0), 0.05, 0.95))
    frequency = counts / max(counts.sum(), 1.0)
    weight = 1.0 / np.sqrt(np.clip(frequency, 1e-6, None))
    weight = weight / weight.mean()
    return alpha, torch.tensor(weight, dtype=torch.float32), counts.tolist()


def _grad_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True,
                                allow_unused=True)
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().pow(2).sum())
    return total ** 0.5


def phase_calibrate(state):
    import pdg_stage1_model as MODEL
    device = torch.device("cuda")
    seed_all()
    dataset, loader = build_arm_loader("A2")
    alpha, class_weight, counts = _class_statistics(loader)
    model = MODEL.PDGStage1Model("A2", seed=SEED).to(device)
    for parameter in model.net.parameters():
        parameter.requires_grad_(False)
    heads = [p for module in (model.prh, model.kvh, model.trunc)
             for p in module.parameters()]
    warm = torch.optim.AdamW(heads, lr=2e-4, weight_decay=1e-4)
    model.train()
    steps = 0
    for batch in loader:
        if steps >= WARMUP_STEPS:
            break
        result = model(batch["img"].to(device))
        losses = head_losses(result, batch, device, class_weight, alpha)
        loss = sum(losses.values())
        warm.zero_grad()
        loss.backward()
        warm.step()
        steps += 1
    # The heads branch off the frozen VGG feature and the visibility head reads a
    # detached belief, so no head loss can reach the stage-6 belief conv: its
    # gradient there is exactly zero, and calibrating against it clamped every
    # lambda to the ceiling on the first attempt.  The input image is the one
    # node every loss passes through, and the frozen VGG still propagates to it,
    # so that is the common reference.
    for parameter in model.net.parameters():
        parameter.requires_grad_(True)
    for name, parameter in model.net.named_parameters():
        if name.startswith("vgg") or any(f"m{s}_" in name for s in (1, 2, 3)):
            parameter.requires_grad_(False)
    norms = {"local": [], "palletness": [], "visibility": [], "truncation": []}
    for index, batch in enumerate(loader):
        if index >= CALIB_BATCHES:
            break
        # four gradient-norm probes keep the graph alive, so calibration runs on
        # a 4-sample slice; the quantity wanted is a ratio, not a batch mean
        sliced = {k: (v[:CALIB_SLICE] if hasattr(v, "shape") else v)
                  for k, v in batch.items()}
        images = sliced["img"].to(device).requires_grad_(True)
        result = model(images)
        local = belief_affinity_loss(result, sliced, device)
        losses = head_losses(result, sliced, device, class_weight, alpha)
        norms["local"].append(_grad_norm(local, [images]))
        for key, value in losses.items():
            norms[key].append(_grad_norm(value, [images]))
        del result, local, losses
        torch.cuda.empty_cache()
    medians = {key: float(np.median(value)) if value else 0.0
               for key, value in norms.items()}
    lambdas, clamped = {}, {}
    for key, ratio in TARGET_RATIO.items():
        raw = ratio * medians["local"] / max(medians[key], 1e-12)
        value = float(np.clip(raw, *LAMBDA_CLAMP))
        lambdas[key] = value
        clamped[key] = bool(abs(value - raw) > 1e-12)
    payload = {"grad_norm_median": medians, "lambda": lambdas,
               "clamped": clamped, "target_ratio": TARGET_RATIO,
               "warmup_steps": steps, "focal_alpha": alpha,
               "visibility_class_counts": counts,
               "visibility_class_weight": class_weight.tolist(),
               "achieved_ratio": {k: (lambdas[k] * medians[k] / max(medians["local"], 1e-12))
                                  for k in TARGET_RATIO}}
    atomic_write(OUT / "grad_calibration.json", payload)
    del model
    torch.cuda.empty_cache()
    set_phase(state, "CALIBRATE", "DONE", detail=payload)


def _train_arm(state, arm: str, steps_limit: int | None, tag: str):
    import pdg_stage1_model as MODEL
    device = torch.device("cuda")
    seed_all()
    dataset, loader = build_arm_loader(arm)
    calibration = json.loads((OUT / "grad_calibration.json").read_text("utf-8"))
    alpha = calibration["focal_alpha"]
    class_weight = torch.tensor(calibration["visibility_class_weight"],
                                dtype=torch.float32)
    lambdas = calibration["lambda"]
    model = MODEL.PDGStage1Model(arm, seed=SEED).to(device)
    frozen_before = model.frozen_checksum()
    optimizer = make_optimizer(model)
    model.train()
    history, step = [], 0
    epochs = 1 if steps_limit else EPOCHS
    directory = WEIGHTS / arm if steps_limit is None else WEIGHTS / f"_smoke_{arm}"
    directory.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        running = {"total": 0.0, "local": 0.0, "palletness": 0.0,
                   "visibility": 0.0, "truncation": 0.0}
        seen = 0
        for batch in loader:
            result = model(batch["img"].to(device))
            local = belief_affinity_loss(result, batch, device)
            loss = local
            parts = {"local": float(local.item())}
            if arm == "A2":
                losses = head_losses(result, batch, device, class_weight, alpha)
                for key, value in losses.items():
                    loss = loss + lambdas[key] * value
                    parts[key] = float(value.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]], 1.0)
            optimizer.step()
            step += 1
            seen += 1
            running["total"] += float(loss.item())
            for key, value in parts.items():
                running[key] += value
            if not np.isfinite(float(loss.item())):
                raise SystemExit(f"HARD_BLOCKED: non-finite loss in {arm}")
            if steps_limit and step >= steps_limit:
                break
            if step % 200 == 0:
                log(f"  {tag} epoch {epoch} step {step} loss {loss.item():.5f}")
        summary = {key: value / max(seen, 1) for key, value in running.items()}
        summary.update({"epoch": epoch, "steps": step, "arm": arm})
        history.append(summary)
        log(f"  {tag} epoch {epoch} mean loss {summary['total']:.5f}")
        if steps_limit is None:
            path = directory / f"epoch_{epoch:03d}.pth"
            torch.save(model.state_dict(), path)
            atomic_write(directory / "run_state.json",
                         {"arm": arm, "epoch": epoch, "epochs": EPOCHS,
                          "completed": epoch == EPOCHS, "steps": step,
                          "history": history,
                          "frozen_checksum": model.frozen_checksum(),
                          "checkpoint_sha256": hashlib.sha256(
                              path.read_bytes()).hexdigest()})
        if steps_limit and step >= steps_limit:
            break
    frozen_after = model.frozen_checksum()
    if frozen_before != frozen_after:
        raise SystemExit(f"HARD_BLOCKED: frozen parameters moved in {arm}")
    if arm == "A2" and steps_limit is None:
        records = getattr(dataset, "taca_records", [])
        atomic_write(OUT / "taca_distribution.json",
                     {"records": len(records),
                      "classes": {name: sum(1 for r in records if r["class"] == name)
                                  for name in ("legacy", "frame_edge_truncation",
                                               "constant_margin_scale")},
                      "fallback": sum(1 for r in records if r.get("fallback"))})
    state["optimizer_steps"] = state.get("optimizer_steps", 0) + step
    del model
    torch.cuda.empty_cache()
    return history, frozen_before


def phase_smoke(state, arm: str):
    history, checksum = _train_arm(state, arm, SMOKE_STEPS, f"smoke {arm}")
    set_phase(state, f"SMOKE_{arm}", "DONE",
              detail={"history": history, "frozen_checksum": checksum})


def phase_train(state, arm: str):
    started = time.perf_counter()
    history, checksum = _train_arm(state, arm, None, f"train {arm}")
    set_phase(state, f"TRAIN_{arm}", "DONE",
              detail={"history": history, "frozen_checksum": checksum,
                      "runtime_s": time.perf_counter() - started})


# ---------------------------------------------------------------- evaluation
def _dev_frames():
    runner = load_module("eval56", "scripts/stage0/paper_s2/paper_s2_eval56.py")
    members = json.loads((OUT.parent / "no_response_frames"
                          / "nrf_membership.json").read_text("utf-8"))
    meta = {f["frame_id"]: f for f in runner.cal_n87_frames()}
    d13 = [meta[u] for u in members["R0"] + members["R1"]]
    c13 = [meta[u] for u in members["C0"]]
    return runner, d13, c13


@torch.no_grad()
def _evaluate_arm(runner, model, frames, device, use_vapa: bool):
    import pdg_heads as HEADS
    rows = []
    for spec in frames:
        frame = runner.EvalFrame(spec)
        import cv2
        image = cv2.imread(spec["image_path"])
        tensor = runner.FZ.preprocess_squash(image).to(device)
        result = model(tensor) if model is not None else None
        if result is None:
            continue
        belief = result["h6"][0, :9].float().cpu().numpy()
        width, height = spec["image_width"], spec["image_height"]
        points, _, peaks = runner.pad_decode(belief, "A0_original", width, height)
        off_probability = [0.0] * 9
        if use_vapa and "visibility" in result:
            off_probability = HEADS.off_screen_probability(
                result["visibility"])[0].float().cpu().numpy().tolist()
            points, _ = HEADS.visibility_aware_assembly(points, off_probability)
        pose = frame.solve(points)
        metrics = frame.metrics(pose)
        detected = [p is not None for p in points]
        errors = []
        for corner in range(8):
            gt = frame.gt_points[corner]
            point = points[corner]
            errors.append(np.nan if (gt is None or point is None)
                          else float(np.hypot(point[0] - gt[0], point[1] - gt[1])))
        finite = np.asarray([e for e in errors if np.isfinite(e)])
        rows.append({"frame_id": spec["frame_id"], "domain": spec["domain"],
                     "centroid_raw": float(belief[8].max()),
                     "centroid_detected": bool(detected[8]),
                     "corners_detected": int(sum(detected[:8])),
                     "R4": bool(belief[8].max() > 0.30 and sum(detected[:8]) >= 4),
                     "R6": bool(belief[8].max() > 0.30 and sum(detected[:8]) >= 6),
                     "pose_success": pose is not None,
                     "reproj": metrics["reproj_fixed_gt_px"],
                     "yaw": metrics["yaw_err_deg"],
                     "corner_median": float(np.median(finite)) if len(finite) else np.nan,
                     "t50": int((finite > 50).sum()), "t100": int((finite > 100).sum()),
                     "nan_corner": int(8 - len(finite)),
                     "off_probability": off_probability})
    return rows


def phase_evaluate(state):
    import pandas as pd
    import pdg_stage1_model as MODEL
    runner, d13, c13 = _dev_frames()
    device = torch.device("cuda")
    everything = []
    for arm in ("A0", "A1", "A2"):
        if arm == "A0":
            from models import DopeNetwork
            net = DopeNetwork(numSeg=1)
            net.load_state_dict(torch.load(str(MODEL.EP57), map_location="cpu",
                                           weights_only=True), strict=True)
            net = net.to(device).eval()

            class _Wrap:
                def __call__(self, x):
                    out = net(x)
                    return {"h6": out[0][-1], "a6": out[1][-1]}
            model = _Wrap()
        else:
            checkpoint = WEIGHTS / arm / f"epoch_{EPOCHS:03d}.pth"
            if not checkpoint.is_file():
                raise SystemExit(f"HARD_BLOCKED: missing checkpoint {checkpoint}")
            model = MODEL.PDGStage1Model(arm, seed=SEED)
            model.load_state_dict(torch.load(str(checkpoint), map_location="cpu",
                                             weights_only=True))
            model = model.to(device).eval()
        for label, frames in (("D13", d13), ("C13", c13)):
            for path in (("D0",) if arm != "A2" else ("D0", "D0V")):
                rows = _evaluate_arm(runner, model, frames, device,
                                     use_vapa=(path == "D0V"))
                for row in rows:
                    everything.append({**row, "arm": arm, "path": path,
                                       "set": label})
        del model
        torch.cuda.empty_cache()
    table = pd.DataFrame(everything)
    table.to_csv(OUT / "development_metrics.csv", index=False)
    set_phase(state, "EVALUATE", "DONE", detail={"rows": len(table)})


def phase_decide(state):
    import pandas as pd
    table = pd.read_csv(OUT / "development_metrics.csv")

    def block(arm, label, path):
        return table[(table.arm == arm) & (table.set == label) & (table.path == path)]

    summary = {}
    for arm in ("A0", "A1", "A2"):
        for label in ("D13", "C13"):
            for path in (("D0",) if arm != "A2" else ("D0", "D0V")):
                sub = block(arm, label, path)
                if not len(sub):
                    continue
                summary[f"{arm}|{label}|{path}"] = {
                    "frames": int(len(sub)),
                    "centroid": int(sub.centroid_detected.sum()),
                    "R4": int(sub.R4.sum()), "R6": int(sub.R6.sum()),
                    "pnp": int(sub.pose_success.sum()),
                    "reproj_median": float(pd.to_numeric(sub.reproj, errors="coerce").median()),
                    "t100": int(sub.t100.sum())}
    a2_d13 = summary.get("A2|D13|D0V", summary.get("A2|D13|D0", {}))
    a1_d13 = summary.get("A1|D13|D0", {})
    a1_c13 = summary.get("A1|C13|D0", {})
    a2_c13 = summary.get("A2|C13|D0V", summary.get("A2|C13|D0", {}))
    full = (a2_d13.get("R4", 0) >= 8 and a2_d13.get("centroid", 0) >= 10
            and a2_d13.get("pnp", 0) >= 6 and a2_d13.get("t100", 1) == 0
            and a2_c13.get("pnp", 0) >= a1_c13.get("pnp", 0))
    partial = (not full
               and (a2_d13.get("R4", 0) - a1_d13.get("R4", 0) >= 4
                    or a2_d13.get("pnp", 0) - a1_d13.get("pnp", 0) >= 4)
               and a2_c13.get("pnp", 0) >= a1_c13.get("pnp", 0)
               and a2_d13.get("t100", 1) == 0)
    decision = ("FOUNDATION_FULL_PASS" if full
                else "FOUNDATION_PARTIAL_GO" if partial else "FOUNDATION_STOP")
    payload = {"decision": decision, "summary": summary,
               "holdout": {"e44_open": HOLDOUT_HITS["e44"],
                           "w45_open": HOLDOUT_HITS["w45"],
                           "final_test_open": HOLDOUT_HITS["final_test"]}}
    atomic_write(OUT / "stage1_decision.json", payload)
    state["final_gate"] = decision
    set_phase(state, "DECIDE", "DONE", detail={"decision": decision})


def phase_report(state):
    decision = json.loads((OUT / "stage1_decision.json").read_text("utf-8"))
    parity = json.loads((OUT / "wrapper_parity.json").read_text("utf-8"))
    calibration = json.loads((OUT / "grad_calibration.json").read_text("utf-8"))
    lines = ["# PDG-Net Stage 1 — final report", "",
             f"## Decision: {decision['decision']}", "",
             "```", json.dumps(decision["summary"], indent=1), "```", "",
             "## Wrapper parity", "", "```",
             json.dumps(parity, indent=1), "```", "",
             "## Gradient calibration", "", "```",
             json.dumps({k: calibration[k] for k in
                         ("lambda", "clamped", "achieved_ratio",
                          "grad_norm_median")}, indent=1), "```", "",
             "## Training", "", "```"]
    for arm in ("A1", "A2"):
        entry = state["phases"].get(f"TRAIN_{arm}", {}).get("detail", {})
        lines.append(f"{arm}: {json.dumps(entry.get('history', []))}")
        lines.append(f"{arm} runtime_s: {entry.get('runtime_s')}")
    lines += ["```", "", "## Holdout", "", "```",
              json.dumps(decision["holdout"], indent=1), "```", ""]
    (OUT / "PDG_STAGE1_FINAL_REPORT.md").write_text("\n".join(lines), "utf-8")
    (RUN / "COMPLETE").write_text(decision["decision"], "utf-8")
    set_phase(state, "REPORT", "DONE")


RUNNERS = {
    "PREPARE": phase_prepare,
    "PARITY": phase_parity,
    "CALIBRATE": phase_calibrate,
    "SMOKE_A1": lambda s: phase_smoke(s, "A1"),
    "SMOKE_A2": lambda s: phase_smoke(s, "A2"),
    "TRAIN_A1": lambda s: phase_train(s, "A1"),
    "TRAIN_A2": lambda s: phase_train(s, "A2"),
    "EVALUATE": phase_evaluate,
    "DECIDE": phase_decide,
    "REPORT": phase_report,
}


def run_all(state, only=None):
    install_holdout_guard(sealed_image_paths())
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()), "utf-8")
    for name in PHASES:
        if only and name != only:
            continue
        entry = state["phases"].get(name, {})
        if entry.get("status") == "DONE":
            log(f"{name}: already DONE, skipping")
            continue
        log(f"{name}: RUNNING")
        set_phase(state, name, "RUNNING")
        try:
            RUNNERS[name](state)
            log(f"{name}: DONE")
        except SystemExit as error:
            state["last_error"] = str(error)
            set_phase(state, name, "HARD_BLOCKED", error=str(error))
            log(f"{name}: HARD_BLOCKED {error}")
            return 2
        except Exception:
            trace = traceback.format_exc()
            state["last_error"] = trace
            set_phase(state, name, "FAILED", error=trace[-4000:])
            log(f"{name}: FAILED\n{trace}")
            return 1
    log(f"pipeline COMPLETE  gate={state.get('final_gate')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "parity", "calibrate",
                                            "smoke", "train", "evaluate",
                                            "decide", "report", "all",
                                            "resume", "status"])
    arguments = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if arguments.command == "status":
        for name in PHASES:
            entry = state["phases"].get(name, {})
            print(f"{name:12s} {entry.get('status', 'PENDING'):12s} "
                  f"{entry.get('stamp', '')}")
        print(f"gate: {state.get('final_gate')}")
        print(f"holdout: {state.get('holdout')}")
        return 0
    single = {"prepare": "PREPARE", "parity": "PARITY", "calibrate": "CALIBRATE",
              "evaluate": "EVALUATE", "decide": "DECIDE", "report": "REPORT"}
    only = single.get(arguments.command)
    return run_all(state, only)


if __name__ == "__main__":
    raise SystemExit(main())
