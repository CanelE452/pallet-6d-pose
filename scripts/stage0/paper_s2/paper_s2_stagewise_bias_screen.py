"""Stage-wise bias loss — 5-epoch screen on the canonical Stage-B dataset.

The architecture does not change: VGG19, the six DOPE stages, the nine belief
maps, the existing decoder and the canonical OpenCV PnP with the centroid
included.  Only belief stages 4-6 train, and only four extra losses are added on
top of the legacy Gaussian MSE.

Weights are calibrated by gradient norm, not by raw loss magnitude: the previous
screen's raw-magnitude rule produced a lambda so small that the term it weighted
could never compete with a fixed penalty.

    python scripts/stage0/paper_s2/paper_s2_stagewise_bias_screen.py --all
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "data/pallet/results/paper_s2_stagewise_bias_screen"
WEIGHTS = ROOT / "weights/paper_s2_stagewise_bias_screen"
STAGE0 = ROOT / "scripts/stage0"
DOPE = ROOT / "Deep_Object_Pose"
for extra in (STAGE0, DOPE / "common", DOPE / "train"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EP57 = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"
EPOCHS, SEED, LR = 5, 1, 5e-5
CALIBRATION_BATCHES = 8
TARGET_RATIO = {"mass": 0.20, "rank": 0.15, "distance": 0.10, "progress": 0.05}
LAMBDA_CLAMP = (1e-6, 10.0)
TRAINABLE_PREFIX = ("m4_2.", "m5_2.", "m6_2.")


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MD = _load("MD", STAGE0 / "paper_s2_mechanism_diagnostic.py")
SCREEN = _load("SCREEN", STAGE0 / "paper_s2" / "paper_s2_corner_replacement_screen.py")
FZ = MD.FZ
import stagewise_corner_loss as SCL  # noqa: E402
from heatmap_refinement import channel_masked_mse  # noqa: E402

NEAR, FAR = MD.NEAR_KP, MD.FAR_KP


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ============================================================================
# model
# ============================================================================
def build_model(device: torch.device):
    from models import DopeNetwork

    actual = hashlib.sha256(EP57.read_bytes()).hexdigest()
    if actual != EP57_SHA:
        raise SystemExit(f"BLOCKED: checkpoint SHA mismatch {actual}")
    net = DopeNetwork(numSeg=1)
    state = torch.load(str(EP57), map_location="cpu", weights_only=True)
    net.load_state_dict({k.removeprefix("module."): v for k, v in state.items()},
                        strict=True)
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, parameter in net.named_parameters():
        if name.startswith(TRAINABLE_PREFIX):
            parameter.requires_grad_(True)
            trainable.append(name)
    net.to(device)
    return net, trainable


def reference_parameter(net) -> tuple[str, torch.nn.Parameter]:
    """Belief stage 6's final output convolution weight."""
    candidates = [(name, parameter) for name, parameter in net.named_parameters()
                  if name.startswith("m6_2.") and name.endswith(".weight")]
    return candidates[-1]


def legacy_loss(beliefs, affinities, batch, device):
    target_belief = batch["beliefs"].to(device)
    target_aff = batch["affinities"].to(device)
    belief_mask = batch["belief_channel_mask"].to(device)
    aff_mask = batch["affinity_channel_mask"].to(device)
    total = torch.zeros((), device=device)
    for stage in range(len(beliefs)):
        total = total + channel_masked_mse(beliefs[stage], target_belief, belief_mask)
        total = total + channel_masked_mse(affinities[stage], target_aff, aff_mask)
    return total


def batch_targets(batch, device):
    centres = batch["refine_keypoints"].to(device)
    valid = batch["refine_keypoints_valid"].to(device)
    return centres, valid


# ============================================================================
# Phase C — gradient-norm calibration
# ============================================================================
def calibrate(net, loader, device) -> dict[str, Any]:
    name, parameter = reference_parameter(net)
    criterion = SCL.StagewiseCornerLoss()
    norms = {"legacy": [], **{key: [] for key in TARGET_RATIO}}
    raw = {"legacy": [], **{key: [] for key in TARGET_RATIO}}
    files = []
    net.train()
    for index, batch in enumerate(loader):
        if index >= CALIBRATION_BATCHES:
            break
        files.append(str(batch["file_name"][0]))
        images = batch["img"].to(device)
        outputs = net(images)
        beliefs, affinities = outputs[0], outputs[1]
        centres, valid = batch_targets(batch, device)
        pieces = {"legacy": legacy_loss(beliefs, affinities, batch, device)}
        pieces.update({key: value for key, value in
                       criterion(beliefs, centres, valid).items()
                       if key in TARGET_RATIO})
        for key, value in pieces.items():
            grad = torch.autograd.grad(value, parameter, retain_graph=True,
                                       allow_unused=True)[0]
            norms[key].append(0.0 if grad is None else float(grad.norm()))
            raw[key].append(float(value.item()))
        net.zero_grad(set_to_none=True)
    median_norm = {key: float(np.median(value)) for key, value in norms.items()}
    median_raw = {key: float(np.median(value)) for key, value in raw.items()}
    lambdas, clamped = {}, {}
    for key, ratio in TARGET_RATIO.items():
        value = ratio * median_norm["legacy"] / (median_norm[key] + 1e-12)
        lambdas[key] = float(np.clip(value, *LAMBDA_CLAMP))
        clamped[key] = bool(value != lambdas[key])
    return {"parameter": name, "batches": CALIBRATION_BATCHES,
            "files": files, "raw_loss_median": median_raw,
            "grad_norm_median": median_norm, "lambda": lambdas,
            "clamped": clamped, "target_ratio": TARGET_RATIO,
            "weighted_grad_norm": {k: lambdas[k] * median_norm[k]
                                   for k in TARGET_RATIO}}


# ============================================================================
# Phase G — training
# ============================================================================
def train(net, loader, device, lambdas, options, steps: Optional[int] = None):
    parameters = [p for p in net.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(parameters, lr=LR)
    criterion = SCL.StagewiseCornerLoss()
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, (1 if steps else EPOCHS) + 1):
        net.train()
        started = time.time()
        totals: dict[str, list[float]] = {"total": [], "legacy": [],
                                          **{k: [] for k in TARGET_RATIO}}
        for index, batch in enumerate(loader):
            if steps and index >= steps:
                break
            images = batch["img"].to(device, non_blocking=True)
            outputs = net(images)
            beliefs, affinities = outputs[0], outputs[1]
            centres, valid = batch_targets(batch, device)
            stagewise = criterion(beliefs, centres, valid)
            legacy = legacy_loss(beliefs, affinities, batch, device)
            total = legacy + sum(lambdas[key] * stagewise[key] for key in TARGET_RATIO)
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            optimiser.step()
            totals["total"].append(float(total.item()))
            totals["legacy"].append(float(legacy.item()))
            for key in TARGET_RATIO:
                totals[key].append(float(stagewise[key].item()))
            if index % 200 == 0:
                log(f"  epoch {epoch} step {index}/{len(loader)} "
                    f"total {np.mean(totals['total'][-200:]):.5f} "
                    f"rank {np.mean(totals['rank'][-200:]):.4f} "
                    f"mass {np.mean(totals['mass'][-200:]):.4f}")
        elapsed = time.time() - started
        row = {"epoch": epoch, "seconds": elapsed,
               "peak_gpu_mb": torch.cuda.max_memory_allocated() // 2 ** 20,
               **{k: float(np.mean(v)) for k, v in totals.items()}}
        history.append(row)
        if steps:
            return history
        log(f"  epoch {epoch} done in {elapsed/60:.1f} min  "
            f"total {row['total']:.5f}  legacy {row['legacy']:.5f}")
        torch.save(net.state_dict(), WEIGHTS / f"epoch_{epoch:03d}.pth")
        torch.save(net.state_dict(), WEIGHTS / "last.pth")
        torch.save(optimiser.state_dict(), WEIGHTS / "optimizer_last.pth")
        (WEIGHTS / "run_state.json").write_text(json.dumps(
            {"epoch": epoch, "epochs": EPOCHS, "completed": epoch == EPOCHS,
             "history": history}, indent=1))
    return history


# ============================================================================
# Phase H/I — canonical evaluation, centroid included
# ============================================================================
@torch.no_grad()
def evaluate(net, device, tag: str) -> pd.DataFrame:
    net.eval()
    audit = FZ.InputAudit()
    criterion_mask = SCL.gt_window_mask
    rows = []
    for spec in SCREEN.mechanism_frames():
        uid = spec["frame_id"]
        geometry = MD.FrameGeometry(spec, audit)
        image = audit.read_image(spec["image_path"])
        tensor = FZ.preprocess_squash(image).to(device)
        beliefs = net(tensor)[0]
        scale_x = spec["image_width"] / 50.0
        scale_y = spec["image_height"] / 50.0
        stage_maps = {4: beliefs[3][0].float().cpu().numpy(),
                      5: beliefs[4][0].float().cpu().numpy(),
                      6: beliefs[5][0].float().cpu().numpy()}
        decoded = {stage: MD.decode_all(array, scale_x, scale_y, geometry.gt_points)
                   for stage, array in stage_maps.items()}
        # canonical PnP: the full nine-point correspondence set, centroid included
        points = decoded[6]["D0"]
        pose = geometry.solve(points)
        metrics = geometry.metrics(pose)
        entry = {"frame_id": uid, "session_id": spec["session_id"],
                 "domain": spec["domain"], "arm": tag,
                 "pose_success": bool(pose is not None),
                 "yaw_err_deg": metrics["yaw_err_deg"],
                 "reproj_px": metrics["reproj_fixed_gt_px"],
                 "rotation_err_deg": metrics["rotation_err_deg"],
                 "translation_err_m": metrics["translation_err_m"]}
        for stage in (4, 5, 6):
            array = stage_maps[stage]
            series = decoded[stage]["D0"]
            centres = torch.tensor(
                [[gt[0] / scale_x, gt[1] / scale_y] if gt is not None else [-1.0, -1.0]
                 for gt in geometry.gt_points[:8]], dtype=torch.float32)[None]
            heat = torch.tensor(array[:8], dtype=torch.float32)[None]
            mass = SCL.gt_mass(heat, centres)[0].numpy()
            window = criterion_mask(heat, centres)[0].numpy()
            for corner in range(8):
                gt = geometry.gt_points[corner]
                point = series[corner]
                entry[f"s{stage}_peak_{corner}"] = float(array[corner].max())
                entry[f"s{stage}_mass_{corner}"] = float(mass[corner])
                outside = np.where(window[corner], -1e4, array[corner])
                entry[f"s{stage}_wrong_{corner}"] = float(outside.max())
                if gt is None or point is None:
                    entry[f"s{stage}_err_{corner}"] = np.nan
                    entry[f"s{stage}_dx_{corner}"] = np.nan
                    entry[f"s{stage}_dy_{corner}"] = np.nan
                else:
                    entry[f"s{stage}_err_{corner}"] = float(
                        np.hypot(point[0] - gt[0], point[1] - gt[1]))
                    entry[f"s{stage}_dx_{corner}"] = float(point[0] - gt[0])
                    entry[f"s{stage}_dy_{corner}"] = float(point[1] - gt[1])
        rows.append(entry)
    if audit.prohibited_attempts:
        raise RuntimeError(f"final-test access: {audit.prohibited_attempts}")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--evaluate", type=str, default="")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda")

    log("[A] identity + baseline reproduction")
    gate = SCREEN.baseline_reproduction()
    log(f"    strict {gate['strict_n']} gt2d {gate['gt2d_pose_success']} "
        f"pred {gate['pred_pose_success']} yaw {gate['yaw_median_deg']:.6f} "
        f"reproj {gate['fixed_gt_reproj_median_px']:.6f} passed={gate['passed']}")
    if not gate["passed"]:
        raise SystemExit(f"BLOCKED: {gate['problems']}")

    net, trainable = build_model(device)
    count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log(f"[E] trainable tensors {len(trainable)}  params {count:,}  "
        f"(belief stages 4-6 only)")
    frozen_audit = {
        "vgg_trainable": sum(1 for n, p in net.named_parameters()
                             if n.startswith("vgg.") and p.requires_grad),
        "belief123_trainable": sum(1 for n, p in net.named_parameters()
                                   if n.startswith(("m1_2.", "m2_2.", "m3_2."))
                                   and p.requires_grad),
        "affinity_trainable": sum(1 for n, p in net.named_parameters()
                                  if "_1." in n and p.requires_grad),
    }
    if any(frozen_audit.values()):
        raise SystemExit(f"BLOCKED: frozen boundary violated {frozen_audit}")
    log(f"    frozen audit OK {frozen_audit}")

    if args.evaluate:
        if args.evaluate != "ep57":
            net.load_state_dict(torch.load(args.evaluate, map_location="cpu",
                                           weights_only=True), strict=True)
        tag = "C0" if args.evaluate == "ep57" else "C1"
        table = evaluate(net, device, tag)
        table.to_parquet(OUT / f"mechanism_{tag}.parquet")
        log(f"[eval] wrote mechanism_{tag}.parquet rows={len(table)}")
        return 0

    options = SCREEN.canonical_options()
    options.batchsize = args.batch
    dataset, loader, sampler, _ = SCREEN.build_loader(options)
    log(f"[A] dataset {len(dataset)} frames  {len(loader)} batches  "
        f"batch {options.batchsize}  roots "
        f"{[pathlib.Path(r).name for r in options.data]}")

    log("[C] gradient-norm calibration (8 batches, no update, train only)")
    calibration = calibrate(net, loader, device)
    (WEIGHTS / "grad_calibration.json").write_text(
        json.dumps(calibration, indent=1), encoding="utf-8")
    (OUT / "stagewise_loss_grad_calibration.json").write_text(
        json.dumps(calibration, indent=1), encoding="utf-8")
    log(f"    reference {calibration['parameter']}")
    log(f"    |g| legacy {calibration['grad_norm_median']['legacy']:.4e}  "
        + "  ".join(f"{k} {calibration['grad_norm_median'][k]:.3e}"
                    for k in TARGET_RATIO))
    log(f"    lambda " + "  ".join(f"{k} {calibration['lambda'][k]:.4g}"
                                   for k in TARGET_RATIO)
        + f"   clamped={[k for k, v in calibration['clamped'].items() if v]}")

    if args.smoke:
        log("[F] 20-step smoke")
        before = {n: p.detach().clone() for n, p in net.named_parameters()}
        history = train(net, loader, device, calibration["lambda"], options, steps=20)
        moved = sum(1 for n, p in net.named_parameters()
                    if not torch.equal(p.detach(), before[n]))
        frozen_moved = sum(1 for n, p in net.named_parameters()
                           if not n.startswith(TRAINABLE_PREFIX)
                           and not torch.equal(p.detach(), before[n]))
        log(f"    params moved {moved}  frozen moved {frozen_moved}  "
            f"total {history[0]['total']:.5f}  peak GPU {history[0]['peak_gpu_mb']} MB")
        if frozen_moved:
            raise SystemExit("BLOCKED: a frozen parameter changed")
        return 0

    provenance = {
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip(),
        "checkpoint_sha256": EP57_SHA, "roots": list(options.data),
        "balance_groups": options.balance_groups, "dataset_frames": len(dataset),
        "batches": len(loader), "batch_size": options.batchsize,
        "epochs": EPOCHS, "seed": SEED, "lr": LR,
        "trainable_tensors": len(trainable), "trainable_params": count,
        "frozen_audit": frozen_audit, "baseline_gate": MD.jsonable(gate),
        "calibration": calibration,
    }
    (OUT / "stagewise_run_provenance.json").write_text(
        json.dumps(MD.jsonable(provenance), indent=1), encoding="utf-8")

    log("[H] epoch 0 evaluation (C0 = canonical ep57)")
    evaluate(net, device, "C0").to_parquet(OUT / "mechanism_C0.parquet")

    log(f"[G] training {EPOCHS} epochs")
    started = time.time()
    history = train(net, loader, device, calibration["lambda"], options)
    minutes = (time.time() - started) / 60.0
    pd.DataFrame(history).to_csv(OUT / "stagewise_epoch_metrics.csv", index=False)
    log(f"[G] total {minutes:.1f} min")

    log("[H] epoch 5 evaluation (C1)")
    evaluate(net, device, "C1").to_parquet(OUT / "mechanism_C1.parquet")
    (OUT / "stagewise_runtime.json").write_text(
        json.dumps({"total_minutes": minutes, "history": MD.jsonable(history)},
                   indent=1), encoding="utf-8")
    log(f"[done] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
