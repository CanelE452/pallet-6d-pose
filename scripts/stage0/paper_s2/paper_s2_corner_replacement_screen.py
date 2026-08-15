"""Corner proposal replacement — 5-epoch architecture screen.

ep57 is the initialisation, not a frozen teacher: the last VGG block, DOPE belief
stages 4-6 and a new corner proposal branch train together on the full canonical
Stage-B dataset so that corner identity can form in the feature itself.  Exactly
five epochs, no checkpoint selection, epoch 5 is the evaluated model.

    python scripts/stage0/paper_s2/paper_s2_corner_replacement_screen.py --all
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "data/pallet/results/paper_s2_corner_replacement_screen"
WEIGHTS = ROOT / "weights/paper_s2/paper_s2_corner_replacement_screen"
STAGE0 = ROOT / "scripts/stage0"
DOPE = ROOT / "Deep_Object_Pose"
for extra in (STAGE0, DOPE / "common", DOPE / "train"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EP57 = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"
EPOCHS = 5
SEED = 1
LAMBDA_GATE = 0.01
TARGET_SHARE = 0.20     # weighted proposal / refined vs the DOPE stage 4-6 loss
CALIBRATION_BATCHES = 20


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MD = _load("MD", STAGE0 / "paper_s2_mechanism_diagnostic.py")
FZ = MD.FZ
import corner_proposal_replacement as CPR  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ============================================================================
# Phase A — identity and canonical configuration
# ============================================================================
def canonical_options() -> Any:
    """Stage-B's own Namespace, with only the screen's knobs changed."""
    header = (ROOT / "weights/paper_s2_stageB/header.txt").read_text("utf-8")
    line = header.splitlines()[0]
    from argparse import Namespace  # noqa: F401  (eval target)

    options = eval(line, {"Namespace": Namespace, "__builtins__": {}})
    options.epochs = EPOCHS
    options.net_path = str(EP57)
    options.outf = str(WEIGHTS)
    return options


def identity() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(args, cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()

    return {
        "head": run("git", "rev-parse", "HEAD"),
        "origin_main": run("git", "rev-parse", "origin/main"),
        "dirty_files": len([l for l in run("git", "status", "--porcelain").splitlines() if l]),
        "checkpoint_sha256": hashlib.sha256(EP57.read_bytes()).hexdigest(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_free_mb": (torch.cuda.mem_get_info()[0] // 2 ** 20)
        if torch.cuda.is_available() else None,
    }


def baseline_reproduction() -> dict[str, Any]:
    manifest = json.loads(MD.MANIFEST_PATH.read_text("utf-8"))
    primary = [f for f in manifest["frames"] if f["population"] == "primary"]
    manifest = dict(manifest, frames=primary)
    audit = FZ.InputAudit()
    tensors = MD.load_cached_tensors()
    geometries, decoded = {}, {}
    for spec in primary:
        uid = spec["frame_id"]
        geometry = MD.FrameGeometry(spec, audit)
        stack = tensors[f"{uid}|belief_stages"]
        geometries[uid] = geometry
        decoded[uid] = MD.decode_all(
            stack[5], spec["image_width"] / 50.0, spec["image_height"] / 50.0,
            geometry.gt_points)
    gate = MD.baseline_gate(manifest, geometries, decoded)
    if audit.prohibited_attempts:
        raise RuntimeError(f"final-test access: {audit.prohibited_attempts}")
    return gate


def mechanism_frames() -> list[dict[str, Any]]:
    manifest = json.loads(MD.MANIFEST_PATH.read_text("utf-8"))
    return [f for f in manifest["frames"] if f["population"] == "primary"]


# ============================================================================
# Phase B — model assembly
# ============================================================================
class ScreenModel(torch.nn.Module):
    """ep57 DOPE plus the proposal branch, with the trainable boundary set."""

    def __init__(self) -> None:
        super().__init__()
        from models import DopeNetwork

        actual = hashlib.sha256(EP57.read_bytes()).hexdigest()
        if actual != EP57_SHA:
            raise RuntimeError(f"checkpoint SHA mismatch: {actual}")
        self.net = DopeNetwork(numSeg=1)
        state = torch.load(str(EP57), map_location="cpu", weights_only=True)
        self.net.load_state_dict(
            {k.removeprefix("module."): v for k, v in state.items()}, strict=True)
        self.high: Optional[torch.Tensor] = None
        self.low: Optional[torch.Tensor] = None
        self.branch: Optional[CPR.CornerProposalReplacement] = None
        self._hooked = False

    def discover(self, sample: torch.Tensor) -> dict[str, int]:
        index_high, channels_high = CPR.find_feature_layer(
            self.net.vgg, sample, CPR.HIGH_GRID)
        index_low, channels_low = CPR.find_feature_layer(
            self.net.vgg, sample, CPR.BELIEF_GRID)
        self.net.vgg[index_high].register_forward_hook(
            lambda m, i, o: setattr(self, "high", o))
        self.net.vgg[index_low].register_forward_hook(
            lambda m, i, o: setattr(self, "low", o))
        self._hooked = True
        torch.manual_seed(SEED)
        self.branch = CPR.CornerProposalReplacement(channels_high, channels_low)
        self.index_high, self.index_low = index_high, index_low
        return {"index_high": index_high, "channels_high": channels_high,
                "index_low": index_low, "channels_low": channels_low}

    def set_trainable(self) -> dict[str, Any]:
        """Freeze early VGG, stages 1-3, affinity and segmentation."""
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        names: list[str] = []
        for name, parameter in self.net.named_parameters():
            index = None
            if name.startswith("vgg."):
                index = int(name.split(".")[1])
            # "last VGG block" = the layers after the final downsample, i.e.
            # everything operating at 50x50.  index_high is the last 100x100
            # layer, so > index_high is exactly that block (index_low sits
            # inside it and must itself stay trainable).
            trainable = (
                (index is not None and index > self.index_high)
                or name.startswith(("m4_2.", "m5_2.", "m6_2."))         # belief 4-6
            )
            if trainable:
                parameter.requires_grad_(True)
                names.append(name)
        for parameter in self.branch.parameters():
            parameter.requires_grad_(True)
        groups = {
            "vgg_last": sum(p.numel() for n, p in self.net.named_parameters()
                            if p.requires_grad and n.startswith("vgg.")),
            "belief_stage4_6": sum(p.numel() for n, p in self.net.named_parameters()
                                   if p.requires_grad and not n.startswith("vgg.")),
            "proposal_branch": sum(p.numel() for p in self.branch.parameters()),
        }
        groups["total"] = sum(groups.values())
        return {"trainable_names": names, "param_groups": groups}

    def forward_full(self, images: torch.Tensor, canonical: torch.Tensor,
                     dimensions: torch.Tensor) -> dict[str, Any]:
        outputs = self.net(images)
        beliefs = outputs[0]
        assert self.high is not None and self.low is not None
        result = self.branch(self.high, self.low, beliefs[3][:, :8],
                             beliefs[4][:, :8], beliefs[5], canonical, dimensions)
        result["beliefs"] = beliefs
        result["affinities"] = outputs[1]
        return result


def unit_canonical() -> torch.Tensor:
    """Corner signs of a unit cuboid: an ID-determined constant, dims separate."""
    from pallet_graph_geometry import make_corners

    return torch.tensor(np.asarray(make_corners(1.0, 1.0, 1.0)[:8], np.float32))


# ============================================================================
# Phase C — losses
# ============================================================================
def dope_stage_loss(beliefs, target, mask) -> torch.Tensor:
    from heatmap_refinement import channel_masked_mse

    total = torch.zeros((), device=target.device)
    for stage in (3, 4, 5):
        total = total + channel_masked_mse(beliefs[stage], target, mask)
    return total


def screen_losses(result: dict[str, Any], batch: dict[str, Any],
                  device: torch.device) -> dict[str, torch.Tensor]:
    target = batch["beliefs"].to(device)
    mask = batch["belief_channel_mask"].to(device)
    centres = batch["refine_keypoints"].to(device)[:, :8]
    valid = batch["refine_keypoints_valid"].to(device)[:, :8]
    # Validity follows the transformed GT centre, not a raster that happens to
    # be empty; a corner outside the belief grid is dropped from the coordinate
    # objectives because no full-map proposal can place a peak there.
    inside = ((centres[..., 0] >= 0) & (centres[..., 0] <= CPR.BELIEF_GRID - 1)
              & (centres[..., 1] >= 0) & (centres[..., 1] <= CPR.BELIEF_GRID - 1))
    valid = valid * inside.float()
    lower = centres.amin(dim=1)
    upper = centres.amax(dim=1)
    diagonal = torch.linalg.norm(upper - lower, dim=-1).clamp_min(1.0)
    return {
        "dope": dope_stage_loss(result["beliefs"], target, mask),
        "proposal": CPR.proposal_objective(result["proposal"], centres, valid,
                                           diagonal),
        "refined": CPR.proposal_objective(result["refined"], centres, valid,
                                          diagonal),
        "gate": result["gate"].mean(),
    }


# ============================================================================
# Phase D — training
# ============================================================================
def build_loader(options: Any):
    import train as TRAIN

    diffpnp_index = {}
    import glob

    for path in glob.glob(os.path.join(options.diffpnp_index_dir, "*.json")):
        for rel, entry in json.load(open(path)).items():
            diffpnp_index[os.path.abspath(
                os.path.join(options.diffpnp_root, rel))] = entry
    dataset, loader, sampler = TRAIN.build_training_loader(
        options, 50, bool(options.mask_aux), False, True, diffpnp_index, SEED)
    return dataset, loader, sampler, diffpnp_index


def make_batch_inputs(batch, device, canonical):
    images = batch["img"].to(device, non_blocking=True)
    dims = batch["dims_m"].to(device)
    dims_valid = batch["dims_valid"].to(device)
    # Roots without a physical extent contribute a zero vector rather than an
    # invented size; the corner-ID embedding still identifies the query.
    dims = dims * dims_valid
    scale = dims.amax(dim=-1, keepdim=True).clamp_min(1e-3)
    normalised = dims / scale
    canonical_batch = canonical[None].expand(images.shape[0], 8, 3).to(device)
    return images, canonical_batch, normalised


def calibrate(model, loader, device, canonical) -> dict[str, float]:
    """20 batches, no update, on train data only."""
    values = {"dope": [], "proposal": [], "refined": []}
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= CALIBRATION_BATCHES:
                break
            images, canonical_batch, dims = make_batch_inputs(batch, device, canonical)
            result = model.forward_full(images, canonical_batch, dims)
            losses = screen_losses(result, batch, device)
            for key in values:
                values[key].append(float(losses[key].item()))
    medians = {key: float(np.median(value)) for key, value in values.items()}
    lambda_prop = TARGET_SHARE * medians["dope"] / max(medians["proposal"], 1e-9)
    lambda_ref = TARGET_SHARE * medians["dope"] / max(medians["refined"], 1e-9)
    return {"median": medians, "lambda_proposal": lambda_prop,
            "lambda_refined": lambda_ref, "target_share": TARGET_SHARE,
            "batches": CALIBRATION_BATCHES}


def train(model, loader, device, canonical, calibration, batch_size) -> list[dict]:
    parameters_branch = list(model.branch.parameters())
    parameters_stage = [p for n, p in model.net.named_parameters()
                        if p.requires_grad and not n.startswith("vgg.")]
    parameters_vgg = [p for n, p in model.net.named_parameters()
                      if p.requires_grad and n.startswith("vgg.")]
    optimiser = torch.optim.AdamW([
        {"params": parameters_branch, "lr": 3e-4},
        {"params": parameters_stage, "lr": 5e-5},
        {"params": parameters_vgg, "lr": 1e-5},
    ], weight_decay=1e-4)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        started = time.time()
        totals = {"loss": [], "dope": [], "proposal": [], "refined": [], "gate": []}
        for index, batch in enumerate(loader):
            images, canonical_batch, dims = make_batch_inputs(batch, device, canonical)
            result = model.forward_full(images, canonical_batch, dims)
            losses = screen_losses(result, batch, device)
            total = (losses["dope"]
                     + calibration["lambda_proposal"] * losses["proposal"]
                     + calibration["lambda_refined"] * losses["refined"]
                     + LAMBDA_GATE * losses["gate"])
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            optimiser.step()
            totals["loss"].append(float(total.item()))
            for key in ("dope", "proposal", "refined", "gate"):
                totals[key].append(float(losses[key].item()))
            if index % 200 == 0:
                log(f"  epoch {epoch} step {index}/{len(loader)} "
                    f"loss {np.mean(totals['loss'][-200:]):.4f} "
                    f"gate {np.mean(totals['gate'][-200:]):.4f}")
        elapsed = time.time() - started
        row = {"epoch": epoch, "seconds": elapsed,
               "samples_per_sec": len(loader) * batch_size / max(elapsed, 1e-9),
               "peak_gpu_mb": torch.cuda.max_memory_allocated() // 2 ** 20,
               **{key: float(np.mean(value)) for key, value in totals.items()}}
        history.append(row)
        log(f"  epoch {epoch} done in {elapsed/60:.1f} min  "
            f"loss {row['loss']:.4f}  gate {row['gate']:.4f}")
        torch.save({"net": model.net.state_dict(), "branch": model.branch.state_dict()},
                   WEIGHTS / f"epoch_{epoch:03d}.pth")
        torch.save({"net": model.net.state_dict(), "branch": model.branch.state_dict()},
                   WEIGHTS / "last.pth")
        torch.save(optimiser.state_dict(), WEIGHTS / "optimizer_last.pth")
        (WEIGHTS / "run_state.json").write_text(
            json.dumps({"epoch": epoch, "epochs": EPOCHS,
                        "completed": epoch == EPOCHS, "history": history}, indent=1))
    return history


# ============================================================================
# Phase E — mechanism evaluation
# ============================================================================
@torch.no_grad()
def evaluate_mechanism(model, device, canonical, tag: str) -> pd.DataFrame:
    """One forward per N87 frame; decode base / proposal / refined."""
    frames = mechanism_frames()
    audit = FZ.InputAudit()
    model.eval()
    rows = []
    for spec in frames:
        uid = spec["frame_id"]
        geometry = MD.FrameGeometry(spec, audit)
        image = audit.read_image(spec["image_path"])
        tensor = FZ.preprocess_squash(image).to(device)
        dims = torch.tensor(np.asarray(geometry.dims, np.float32))[None].to(device)
        dims = dims / dims.amax(dim=-1, keepdim=True).clamp_min(1e-3)
        result = model.forward_full(tensor, canonical[None].to(device), dims)
        scale_x = spec["image_width"] / 50.0
        scale_y = spec["image_height"] / 50.0
        centroid = result["beliefs"][5][:, 8:9]
        maps = {
            "base": torch.cat([result["base"], centroid], dim=1),
            "proposal": torch.cat([result["proposal_transformed"], centroid], dim=1),
            "refined": torch.cat([result["refined"], centroid], dim=1),
        }
        entry = {"frame_id": uid, "session_id": spec["session_id"],
                 "domain": spec["domain"], "arm": tag,
                 "gate_mean": float(result["gate"].mean().item())}
        for index in range(8):
            entry[f"gate_{index}"] = float(result["gate"][0, index].item())
        for name, belief in maps.items():
            array = belief[0].float().cpu().numpy()
            decoded = MD.decode_all(array, scale_x, scale_y, geometry.gt_points)
            points = decoded["D0"]
            pose = geometry.solve(points)
            metrics = geometry.metrics(pose)
            matched = geometry.matched_2d_error(points)
            entry[f"{name}_pose_success"] = bool(pose is not None)
            entry[f"{name}_yaw_err"] = metrics["yaw_err_deg"]
            entry[f"{name}_reproj"] = metrics["reproj_fixed_gt_px"]
            entry[f"{name}_far_px"] = matched["far_matched_median_px"]
            entry[f"{name}_near_px"] = matched["near_matched_median_px"]
            entry[f"{name}_med_px"] = matched["matched_median_px"]
            for corner in range(8):
                gt = geometry.gt_points[corner]
                point = points[corner]
                heat = array[corner]
                entry[f"{name}_peak_{corner}"] = float(heat.max())
                if gt is not None and point is not None:
                    entry[f"{name}_err_{corner}"] = float(
                        np.hypot(point[0] - gt[0], point[1] - gt[1]))
                    # signed components: the bias direction is the metric, the
                    # magnitude alone cannot show a systematic shift
                    entry[f"{name}_dx_{corner}"] = float(point[0] - gt[0])
                    entry[f"{name}_dy_{corner}"] = float(point[1] - gt[1])
                else:
                    entry[f"{name}_err_{corner}"] = np.nan
                    entry[f"{name}_dx_{corner}"] = np.nan
                    entry[f"{name}_dy_{corner}"] = np.nan
        rows.append(entry)
    if audit.prohibited_attempts:
        raise RuntimeError(f"final-test access: {audit.prohibited_attempts}")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--reevaluate", type=str, default="",
                        help="checkpoint to evaluate ('ep57' for the C0 baseline)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log("[A] identity and baseline reproduction")
    info = identity()
    log(f"    HEAD {info['head'][:8]}  ckpt {info['checkpoint_sha256'][:16]}  "
        f"{info['gpu']}  free {info['gpu_free_mb']} MB")
    gate = baseline_reproduction()
    log(f"    strict {gate['strict_n']}  gt2d {gate['gt2d_pose_success']}  "
        f"pred {gate['pred_pose_success']}  yaw {gate['yaw_median_deg']:.6f}  "
        f"reproj {gate['fixed_gt_reproj_median_px']:.6f}  passed={gate['passed']}")
    if not gate["passed"]:
        raise SystemExit(f"BLOCKED: baseline reproduction failed {gate['problems']}")

    options = canonical_options()
    options.batchsize = args.batch
    device = torch.device("cuda")
    log(f"[A] canonical roots: {[pathlib.Path(d).name for d in options.data]}")
    log(f"    balance_groups: {options.balance_groups}")
    dataset, loader, sampler, diffpnp_index = build_loader(options)
    log(f"    dataset {len(dataset)} frames  loader {len(loader)} batches "
        f"batch={options.batchsize}  sampler={'ratio' if sampler else 'shuffle'}")

    model = ScreenModel()
    sample = torch.zeros(1, 3, options.imagesize, options.imagesize)
    features = model.discover(sample)
    log(f"[B] F100 = vgg[{features['index_high']}] {features['channels_high']}ch  "
        f"F50 = vgg[{features['index_low']}] {features['channels_low']}ch")
    boundary = model.set_trainable()
    log(f"[B] trainable params {boundary['param_groups']}")
    model.to(device)
    canonical = unit_canonical()

    provenance = {"identity": info, "baseline_gate": MD.jsonable(gate),
                  "roots": list(options.data), "balance_groups": options.balance_groups,
                  "dataset_frames": len(dataset), "batches": len(loader),
                  "batch_size": options.batchsize, "epochs": EPOCHS, "seed": SEED,
                  "features": features, "param_groups": boundary["param_groups"],
                  "sigma": options.sigma, "imagesize": options.imagesize,
                  "mask_aux": bool(options.mask_aux), "diffpnp_index": len(diffpnp_index)}
    (OUT / "corner_replacement_provenance.json").write_text(
        json.dumps(MD.jsonable(provenance), indent=1), encoding="utf-8")
    (WEIGHTS / "config_resolved.json").write_text(
        json.dumps(MD.jsonable(vars(options)), indent=1), encoding="utf-8")

    if args.reevaluate:
        # The mechanism set may only be read at epoch 0 and epoch 5.  Offline
        # re-evaluation must obey the same rule, so any other checkpoint is
        # refused rather than silently allowed.
        allowed = args.reevaluate == "ep57" or \
            pathlib.Path(args.reevaluate).name in ("epoch_005.pth", "last.pth")
        if not allowed:
            raise SystemExit(
                f"BLOCKED: N87 may only be evaluated at epoch 0 or 5, "
                f"got {args.reevaluate}")
        tag = "C0" if args.reevaluate == "ep57" else "C1"
        if args.reevaluate != "ep57":
            payload = torch.load(args.reevaluate, map_location="cpu",
                                 weights_only=True)
            model.net.load_state_dict(payload["net"], strict=True)
            model.branch.load_state_dict(payload["branch"], strict=True)
        model.to(device)
        log(f"[E] re-evaluating {args.reevaluate} as {tag}")
        table = evaluate_mechanism(model, device, canonical, tag)
        suffix = "epoch0" if tag == "C0" else "epoch5"
        table.to_parquet(OUT / f"mechanism_{suffix}.parquet")
        log(f"    wrote mechanism_{suffix}.parquet  rows={len(table)}")
        return 0

    if args.smoke:
        log("[smoke] 20 steps")
        model.train()
        optimiser = torch.optim.AdamW(model.branch.parameters(), lr=3e-4)
        for index, batch in enumerate(loader):
            if index >= 20:
                break
            images, canonical_batch, dims = make_batch_inputs(batch, device, canonical)
            result = model.forward_full(images, canonical_batch, dims)
            losses = screen_losses(result, batch, device)
            (losses["dope"] + losses["proposal"] + losses["refined"]).backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            if index == 0:
                log(f"    identity check: max|refined-base| = "
                    f"{float((result['refined'] - result['base']).abs().max()):.5f}"
                    f"  gate mean {float(result['gate'].mean()):.5f}")
        log(f"[smoke] OK  peak GPU {torch.cuda.max_memory_allocated() // 2**20} MB")
        return 0

    log("[E] epoch 0 baseline evaluation (C0)")
    epoch0 = evaluate_mechanism(model, device, canonical, "C0")
    epoch0.to_parquet(OUT / "mechanism_epoch0.parquet")

    log("[C5] loss calibration (20 batches, no update, train only)")
    calibration = calibrate(model, loader, device, canonical)
    (WEIGHTS / "loss_calibration.json").write_text(
        json.dumps(calibration, indent=1), encoding="utf-8")
    log(f"    medians {calibration['median']}  "
        f"lambda_prop {calibration['lambda_proposal']:.4f}  "
        f"lambda_ref {calibration['lambda_refined']:.4f}")

    log(f"[D] training {EPOCHS} epochs")
    started = time.time()
    history = train(model, loader, device, canonical, calibration, options.batchsize)
    total_minutes = (time.time() - started) / 60.0
    pd.DataFrame(history).to_csv(OUT / "corner_replacement_epoch_metrics.csv",
                                 index=False)
    log(f"[D] total training {total_minutes:.1f} min")

    log("[E] epoch 5 evaluation (C1-base / C1-proposal / C1-refined)")
    epoch5 = evaluate_mechanism(model, device, canonical, "C1")
    epoch5.to_parquet(OUT / "mechanism_epoch5.parquet")
    (OUT / "corner_replacement_runtime.json").write_text(
        json.dumps({"total_training_minutes": total_minutes,
                    "history": MD.jsonable(history)}, indent=1), encoding="utf-8")
    log(f"[done] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
