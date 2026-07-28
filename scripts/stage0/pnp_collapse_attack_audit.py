#!/usr/bin/env python3
"""Read-only ep57 audit for low-frequency, targeted footprint-collapse attacks.

This is a feasibility gate for adversarial fine-tuning, not a trainer.  It uses
only the six locked PAPER_S2 training directories.  A small RGB perturbation is
optimized so the current eight predicted corners move toward a uniformly
contracted copy of themselves.  The original image label is never changed.

The audit answers three questions before spending GPU time on a fine-tune:

* can a bounded, low-frequency RGB perturbation reproduce footprint shrinkage;
* does that shrinkage reach the PnP depth tail seen in real UC failures; and
* does it do so without merely deleting heatmap detections?
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "Deep_Object_Pose" / "common"
TRAIN = ROOT / "Deep_Object_Pose" / "train"
EVAL = ROOT / "scripts" / "data_prep" / "eval"
for path in (COMMON, TRAIN, EVAL):
    sys.path.insert(0, str(path))

from diffpnp3d_loss import DiffPnP3DLoss, LocalSoftArgmax2D  # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
from heatmap_refinement import unpack_dope_output  # noqa: E402
from models import DopeNetwork  # noqa: E402
from utils_dataset import CleanVisiiDopeLoader  # noqa: E402


DATA_NAMES = (
    "mixed_v8_train",
    "v4_split_base",
    "aug_squash_v2",
    "aug_trunc_v2",
    "aug_scale_v2",
    "paper_4pallet_mask_v1",
)
BASE = ROOT / "weights" / "paper_s2_stageB" / "net_epoch_0057.pth"
INDEX_DIR = (ROOT / "data" / "pallet" / "results" /
             "paper_s2_scratch_diffpnp" / "pnp_valid_3d_index")
DATA_ROOT = ROOT / "data" / "pallet" / "training_data"
DEFAULT_OUT = (ROOT / "data" / "pallet" / "eval_results" /
               "paper_s2_rgb1_pnp_adv_v4")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_index() -> dict[str, dict]:
    index: dict[str, dict] = {}
    for path in glob.glob(str(INDEX_DIR / "*.json")):
        if Path(path).name == "val.json":
            continue
        for relative, entry in json.load(open(path)).items():
            absolute = os.path.abspath(DATA_ROOT / relative)
            index[absolute] = entry
    return index


def load_model(device: torch.device) -> torch.nn.Module:
    state = torch.load(BASE, map_location=device)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value
                 for key, value in state.items()}
    model = DopeNetwork(numVec=0, numSeg=1).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}")
    return model.eval()


def balanced_loader(batch_size: int, batches: int, seed: int):
    paths = [str(DATA_ROOT / name) for name in DATA_NAMES]
    dataset = CleanVisiiDopeLoader(
        paths, objects=["pallet"], sigma=2.0, output_size=50,
        mask_aux=True, aspect_resize=True, diffpnp_index=load_index())
    sample_paths = [item[0] for item in dataset.imgs]
    base_substrings = DATA_NAMES[:-1]
    base = np.asarray([
        any(name in path for name in base_substrings) for path in sample_paths])
    mask = np.asarray([
        DATA_NAMES[-1] in path for path in sample_paths])
    weights = np.zeros(len(sample_paths), dtype=np.float64)
    weights[base] = 0.60 / max(1, int(base.sum()))
    weights[mask] = 0.40 / max(1, int(mask.sum()))
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise RuntimeError("invalid 60:40 sampler weights")
    generator = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=batch_size * batches, replacement=True,
        generator=generator)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=0,
        drop_last=True, generator=generator)


def model_belief(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    beliefs, _affinities, _vec, _seg, _refinement = unpack_dope_output(
        model(images))
    return beliefs[-1][:, :8]


def targeted_collapse_attack(
        model: torch.nn.Module,
        images: torch.Tensor,
        valid: torch.Tensor,
        soft_argmax: LocalSoftArgmax2D,
        epsilon_255: float,
        steps: int,
        target_scale: float,
        low_resolution: int,
        seed: int) -> torch.Tensor:
    """Return a detached batch; only DiffPnP-valid frames are perturbed."""
    # torch.flatnonzero is unavailable in the project's pinned PyTorch.
    selected = torch.nonzero(valid.bool(), as_tuple=False).flatten()
    if selected.numel() == 0 or epsilon_255 <= 0:
        return images.detach().clone()
    source = images.index_select(0, selected).detach()
    with torch.no_grad():
        clean_xy, _ = soft_argmax(model_belief(model, source))
        center = clean_xy.mean(dim=1, keepdim=True)
        target_xy = (center + target_scale * (clean_xy - center)).detach()

    mean = source.new_tensor(MEAN).view(1, 3, 1, 1)
    std = source.new_tensor(STD).view(1, 3, 1, 1)
    source_pixel = (source * std + mean).clamp(0.0, 1.0)
    epsilon = float(epsilon_255) / 255.0
    generator = torch.Generator(device=source.device).manual_seed(seed)
    delta = torch.empty(
        source.shape[0], 3, low_resolution, low_resolution,
        device=source.device, dtype=source.dtype)
    delta.uniform_(-epsilon, epsilon, generator=generator)
    # RS-FGSM uses one full boundary-reaching step.  Two-step PGD uses a
    # slightly smaller step so the second update can change direction.
    alpha = epsilon * (1.25 if steps == 1 else 0.75)
    coordinate_scale = source.new_tensor([640.0, 480.0]).view(1, 1, 2)

    for _ in range(steps):
        delta.requires_grad_(True)
        full_delta = F.interpolate(
            delta, size=source.shape[-2:], mode="bilinear",
            align_corners=False)
        adversarial_pixel = (source_pixel + full_delta).clamp(0.0, 1.0)
        adversarial = (adversarial_pixel - mean) / std
        predicted_xy, _ = soft_argmax(model_belief(model, adversarial))
        targeted_loss = (
            ((predicted_xy - target_xy) / coordinate_scale).square()
        ).mean()
        gradient, = torch.autograd.grad(
            targeted_loss, delta, only_inputs=True)
        # Targeted attack: descend toward the contracted coordinate target.
        delta = (delta.detach() - alpha * gradient.sign()).clamp(
            -epsilon, epsilon)

    full_delta = F.interpolate(
        delta, size=source.shape[-2:], mode="bilinear", align_corners=False)
    adversarial_pixel = (source_pixel + full_delta).clamp(0.0, 1.0)
    adversarial = ((adversarial_pixel - mean) / std).detach()
    output = images.detach().clone()
    output[selected] = adversarial
    return output


def footprint_ratios(clean_xy: torch.Tensor,
                     attacked_xy: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """PCA-axis minimum-span and area ratios, axes fixed by clean prediction."""
    clean_centered = clean_xy - clean_xy.mean(dim=1, keepdim=True)
    attacked_centered = attacked_xy - attacked_xy.mean(dim=1, keepdim=True)
    covariance = torch.bmm(clean_centered.transpose(1, 2), clean_centered) / 8.0
    _values, axes = torch.linalg.eigh(covariance)
    clean_axis = torch.bmm(clean_centered, axes)
    attacked_axis = torch.bmm(attacked_centered, axes)
    clean_span = clean_axis.amax(dim=1) - clean_axis.amin(dim=1)
    attacked_span = attacked_axis.amax(dim=1) - attacked_axis.amin(dim=1)
    ratio = attacked_span.clamp_min(1.0e-6) / clean_span.clamp_min(1.0e-6)
    return (ratio.amin(dim=1).detach().cpu().numpy(),
            ratio.prod(dim=1).detach().cpu().numpy())


def detection_counts(belief: torch.Tensor) -> np.ndarray:
    counts = []
    for maps in belief.detach().cpu().numpy():
        decoded = extract_keypoints_from_belief(maps, threshold=0.3)
        counts.append(sum(point[0] >= 0 for point in decoded[:8]))
    return np.asarray(counts, dtype=np.int64)


def evaluate_pair(model, clean, attacked, targets, soft_argmax, pnp_loss):
    valid = targets["diffpnp_valid"].to(clean.device).bool()
    with torch.no_grad():
        clean_belief = model_belief(model, clean)
        attacked_belief = model_belief(model, attacked)
        clean_xy, clean_conf = soft_argmax(clean_belief)
        attacked_xy, attacked_conf = soft_argmax(attacked_belief)
        selected = torch.nonzero(valid, as_tuple=False).flatten()
        if selected.numel() == 0:
            return None
        min_span, area = footprint_ratios(
            clean_xy[selected], attacked_xy[selected])
        _loss, pnp_info = pnp_loss(
            attacked_xy,
            targets["diffpnp_X"].to(clean.device),
            targets["diffpnp_K"].to(clean.device),
            targets["diffpnp_R"].to(clean.device),
            targets["diffpnp_t"].to(clean.device),
            targets["diffpnp_diag"].to(clean.device), valid)
        clean_detect = detection_counts(clean_belief[selected])
        attacked_detect = detection_counts(attacked_belief[selected])
        clean_peak = clean_conf["peak"][selected].amin(dim=1).cpu().numpy()
        attacked_peak = attacked_conf["peak"][selected].amin(dim=1).cpu().numpy()
    return {
        "n_selected": int(selected.numel()),
        "n_pnp": int(pnp_info["n_valid"]),
        "min_span": min_span,
        "area": area,
        "clean_detect": clean_detect,
        "attacked_detect": attacked_detect,
        "clean_peak": clean_peak,
        "attacked_peak": attacked_peak,
        "pnp_hard_n": float(pnp_info["hard_fraction"] * pnp_info["n_valid"]),
        "pnp_tz_sum": float(pnp_info["mean_tz_ratio"] * pnp_info["n_valid"]),
        "pnp_min_span_sum": float(
            pnp_info["mean_min_span_ratio"] * pnp_info["n_valid"]),
    }


def summarize(accumulator: dict) -> dict:
    n_selected = int(accumulator["n_selected"])
    n_pnp = int(accumulator["n_pnp"])
    min_span = np.concatenate(accumulator["min_span"])
    area = np.concatenate(accumulator["area"])
    clean_detect = np.concatenate(accumulator["clean_detect"])
    attacked_detect = np.concatenate(accumulator["attacked_detect"])
    clean_peak = np.concatenate(accumulator["clean_peak"])
    attacked_peak = np.concatenate(accumulator["attacked_peak"])
    originally_all8 = clean_detect == 8
    retention = ((attacked_detect[originally_all8] == 8).mean()
                 if originally_all8.any() else 0.0)
    result = {
        "n_diffpnp_selected": n_selected,
        "n_pnp_guard_valid": n_pnp,
        "raw_min_span_ratio_median": float(np.median(min_span)),
        "raw_area_ratio_median": float(np.median(area)),
        "pnp_tz_ratio_mean": accumulator["pnp_tz_sum"] / max(1, n_pnp),
        "pnp_tz_gt_1p15_fraction": accumulator["pnp_hard_n"] / max(1, n_pnp),
        "pnp_min_span_ratio_mean": (
            accumulator["pnp_min_span_sum"] / max(1, n_pnp)),
        "clean_all8_n": int(originally_all8.sum()),
        "attacked_all8_retention": float(retention),
        "clean_detection_mean": float(clean_detect.mean()),
        "attacked_detection_mean": float(attacked_detect.mean()),
        "min_peak_ratio_median": float(np.median(
            attacked_peak / np.maximum(clean_peak, 1.0e-6))),
    }
    result["passes_attack_gate"] = bool(
        result["raw_min_span_ratio_median"] <= 0.90
        and result["pnp_tz_gt_1p15_fraction"] >= 0.20
        and result["attacked_all8_retention"] >= 0.95)
    return result


def render_report(args, results: dict[str, dict]) -> str:
    lines = [
        "# PAPER_S2 targeted PnP-collapse attack audit",
        "",
        f"Base: `{BASE.relative_to(ROOT)}`; batches={args.batches}; "
        f"batch={args.batch_size}; target scale={args.target_scale}; "
        f"low-frequency grid={args.low_resolution}x{args.low_resolution}.",
        "",
        "This audit does not train or alter a checkpoint. It uses the locked six "
        "training directories and keeps the original labels unchanged.",
        "",
        "| attack | raw min span | PnP tz>1.15 | all-8 retention | gate |",
        "|---|---:|---:|---:|---|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['raw_min_span_ratio_median']:.3f} | "
            f"{100*result['pnp_tz_gt_1p15_fraction']:.1f}% | "
            f"{100*result['attacked_all8_retention']:.1f}% | "
            f"{'PASS' if result['passes_attack_gate'] else 'STOP'} |")
    lines += [
        "",
        "Pre-registered feasibility gate: median raw minimum-span ratio <=0.90, "
        "PnP-valid tz ratio >1.15 in >=20%, and all-eight detection retention "
        ">=95%. A fine-tune is allowed only for an attack arm satisfying all "
        "three conditions.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--target-scale", type=float, default=0.80)
    parser.add_argument("--low-resolution", type=int, default=25)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not (0.0 < args.target_scale < 1.0):
        parser.error("--target-scale must be in (0,1)")
    if args.batches <= 0 or args.batch_size <= 0:
        parser.error("--batches and --batch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = load_model(device)
    soft_argmax = LocalSoftArgmax2D(
        window=7, temperature=0.1, orig_size=(640, 480),
        belief_size=(50, 50)).to(device)
    # hard_fraction is depth-only here because span_under_weight=0.
    pnp_loss = DiffPnP3DLoss(
        n_gn=4, geometry_weight=0.0, undercoverage_weight=0.0,
        span_under_weight=0.0, depth_under_weight=1.0,
        hard_depth_threshold=1.15).to(device)
    cached_batches = list(balanced_loader(
        args.batch_size, args.batches, args.seed))

    configurations = [(float(eps), int(steps))
                      for eps in args.eps for steps in args.steps]
    results: dict[str, dict] = {}
    for config_index, (epsilon, steps) in enumerate(configurations):
        name = f"eps{epsilon:g}_steps{steps}"
        print(f"[{name}]", flush=True)
        accumulator = defaultdict(list)
        for batch_index, targets in enumerate(cached_batches):
            clean = targets["img"].to(device)
            valid = targets["diffpnp_valid"].to(device).bool()
            attacked = targeted_collapse_attack(
                model, clean, valid, soft_argmax, epsilon, steps,
                args.target_scale, args.low_resolution,
                args.seed + 1000 * config_index + batch_index)
            measured = evaluate_pair(
                model, clean, attacked, targets, soft_argmax, pnp_loss)
            if measured is None:
                continue
            for key, value in measured.items():
                if key in {"n_selected", "n_pnp", "pnp_hard_n",
                           "pnp_tz_sum", "pnp_min_span_sum"}:
                    accumulator[key] = accumulator.get(key, 0) + value
                else:
                    accumulator[key].append(value)
        if not accumulator or int(accumulator["n_selected"]) == 0:
            raise RuntimeError(f"{name}: no DiffPnP-valid samples")
        results[name] = summarize(accumulator)
        print(json.dumps(results[name], indent=2), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_checkpoint": str(BASE),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "target_scale": args.target_scale,
        "low_resolution": args.low_resolution,
        "data": list(DATA_NAMES),
        "results": results,
    }
    (args.out_dir / "ATTACK_AUDIT.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    report = render_report(args, results)
    (args.out_dir / "ATTACK_AUDIT.md").write_text(report)
    print(report)
    print(f"saved: {args.out_dir / 'ATTACK_AUDIT.md'}")


if __name__ == "__main__":
    main()
