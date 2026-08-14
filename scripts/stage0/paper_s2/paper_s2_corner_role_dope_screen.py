"""PCR-DOPE screen — Gate A capacity, Gate B group-disjoint + real transfer.

The belief maps, the decoder and the PnP are untouched.  What changes is the
feature stages 4-6 read: a role encoder learns corner identity, and a
zero-initialised FiLM modulates the shared feature with it.  ep57 is the
initialisation and the frozen teacher.

    python scripts/stage0/paper_s2/paper_s2_corner_role_dope_screen.py --gate-a
    python scripts/stage0/paper_s2/paper_s2_corner_role_dope_screen.py --gate-b
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
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "data/pallet/results/paper_s2_corner_role_dope"
WEIGHTS = ROOT / "weights/paper_s2_corner_role_dope"
STAGE0 = ROOT / "scripts/stage0"
DOPE = ROOT / "Deep_Object_Pose"
for extra in (STAGE0, DOPE / "common", DOPE / "train"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EP57 = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"
SEED = 1
GATE_A_STEPS, GATE_A_BATCH = 600, 8
GATE_B_TRAIN, GATE_B_HELD, GATE_B_EPOCHS = 3000, 1000, 3
TARGET_RATIO = {"proto": 0.08, "cross": 0.08, "wrong": 0.06,
                "teacher_wrong": 0.06, "local": 0.02, "anchor": 0.10}
LAMBDA_CLAMP = (1e-6, 10.0)
# mixed_v8_train carries no scene identity at all, so it cannot take part in a
# group-disjoint split; it stays in the canonical full run but not in Gate B.
GROUPABLE_ROOTS = ("v4_split_base", "aug_squash_v2", "aug_trunc_v2",
                   "aug_scale_v2", "paper_4pallet_mask_v1")


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MD = _load("MD", STAGE0 / "paper_s2_mechanism_diagnostic.py")
SCREEN = _load("SCREEN", STAGE0 / "paper_s2" / "paper_s2_corner_replacement_screen.py")
FZ = MD.FZ
import corner_role_adapter as CRA  # noqa: E402
import corner_role_loss as CRL  # noqa: E402
from heatmap_refinement import channel_masked_mse  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ============================================================================
# model
# ============================================================================
class RoleModel(torch.nn.Module):
    """ep57 plus the role encoder and, optionally, the FiLM adapters."""

    def __init__(self, use_film: bool, use_role: bool) -> None:
        super().__init__()
        from models import DopeNetwork

        if hashlib.sha256(EP57.read_bytes()).hexdigest() != EP57_SHA:
            raise SystemExit("BLOCKED: checkpoint SHA mismatch")
        state = torch.load(str(EP57), map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.net = DopeNetwork(numSeg=1)
        self.net.load_state_dict(state, strict=True)
        self.teacher = DopeNetwork(numSeg=1)
        self.teacher.load_state_dict(state, strict=True)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.use_film, self.use_role = use_film, use_role
        self.high = self.low = None
        self.encoder: Optional[CRA.CornerRoleEncoder] = None
        self.film = None

    def discover(self, sample: torch.Tensor) -> dict[str, int]:
        index_high, channels_high = CRA.find_feature_layer(
            self.net.vgg, sample, CRA.HIGH_GRID)
        index_low, channels_low = CRA.find_feature_layer(
            self.net.vgg, sample, CRA.BELIEF_GRID)
        self.net.vgg[index_high].register_forward_hook(
            lambda m, i, o: setattr(self, "high", o))
        self.index_high, self.index_low = index_high, index_low
        torch.manual_seed(SEED)
        if self.use_role:
            self.encoder = CRA.CornerRoleEncoder(channels_high, channels_low)
        if self.use_film:
            self.film = torch.nn.ModuleList(
                [CRA.RoleConditionedFiLM(channels_low) for _ in range(3)])
        return {"index_high": index_high, "channels_high": channels_high,
                "index_low": index_low, "channels_low": channels_low}

    def set_trainable(self) -> dict[str, Any]:
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        names = []
        for name, parameter in self.net.named_parameters():
            index = int(name.split(".")[1]) if name.startswith("vgg.") else None
            if (index is not None and index > self.index_high) or \
                    name.startswith(("m4_2.", "m5_2.", "m6_2.")):
                parameter.requires_grad_(True)
                names.append(name)
        groups = {
            "vgg_last": sum(p.numel() for n, p in self.net.named_parameters()
                            if p.requires_grad and n.startswith("vgg.")),
            "belief_stage4_6": sum(p.numel() for n, p in self.net.named_parameters()
                                   if p.requires_grad and not n.startswith("vgg.")),
            "role_encoder": sum(p.numel() for p in self.encoder.parameters())
            if self.encoder is not None else 0,
            "film": sum(p.numel() for p in self.film.parameters())
            if self.film is not None else 0,
        }
        groups["total"] = sum(groups.values())
        return {"trainable_names": names, "param_groups": groups}

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        """Legacy path when neither role nor FiLM is active."""
        network = self.net
        shared = network.vgg(images)
        role = None
        if self.encoder is not None:
            role = self.encoder(self.high, shared)

        def stage(module, inputs):
            return module(inputs)

        out1_2, out1_1 = network.m1_2(shared), network.m1_1(shared)
        out2 = torch.cat([out1_2, out1_1, shared], 1)
        out2_2, out2_1 = network.m2_2(out2), network.m2_1(out2)
        out3 = torch.cat([out2_2, out2_1, shared], 1)
        out3_2, out3_1 = network.m3_2(out3), network.m3_1(out3)

        feeds = []
        for index in range(3):
            if self.film is not None and role is not None:
                feeds.append(self.film[index](role["embedding"], role["score"],
                                              shared))
            else:
                feeds.append(shared)
        out4 = torch.cat([out3_2, out3_1, feeds[0]], 1)
        out4_2, out4_1 = network.m4_2(out4), network.m4_1(out4)
        out5 = torch.cat([out4_2, out4_1, feeds[1]], 1)
        out5_2, out5_1 = network.m5_2(out5), network.m5_1(out5)
        out6 = torch.cat([out5_2, out5_1, feeds[2]], 1)
        out6_2, out6_1 = network.m6_2(out6), network.m6_1(out6)
        return {
            "beliefs": [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2],
            "affinities": [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1],
            "role": role, "shared": shared,
        }

    @torch.no_grad()
    def teacher_forward(self, images: torch.Tensor):
        return self.teacher(images)[0]


# ============================================================================
# losses
# ============================================================================
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


def role_terms(result, batch, device, teacher_beliefs):
    points = batch["refine_keypoints"].to(device)[:, :8]
    flags = batch["refine_keypoints_valid"].to(device)[:, :8]
    valid = CRA.valid_corner_mask(points, flags)
    objective = CRL.CornerRoleObjective()
    losses = objective(result["role"]["score"], points, valid,
                       result["beliefs"][5], teacher_belief=teacher_beliefs[5])
    teacher_peaks = CRL.peak_coordinates(teacher_beliefs[5][:, :8])
    hard_tail = (torch.linalg.norm(teacher_peaks - points, dim=-1) > 20.0 / 12.8)
    losses["anchor"] = CRL.teacher_anchor_loss(
        [result["beliefs"][i] for i in (3, 4, 5)],
        [teacher_beliefs[i] for i in (3, 4, 5)],
        batch["belief_channel_mask"].to(device), hard_tail)
    losses["_valid"] = valid
    return losses


def calibrate(model, loader, device, batches: int = 8) -> dict[str, Any]:
    name, parameter = next(
        (n, p) for n, p in reversed(list(model.net.named_parameters()))
        if n.startswith("vgg.") and p.requires_grad and n.endswith("weight"))
    norms = {k: [] for k in list(TARGET_RATIO) + ["legacy"]}
    model.train()
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        images = batch["img"].to(device)
        result = model(images)
        teacher = model.teacher_forward(images)
        pieces = {"legacy": legacy_loss(result["beliefs"], result["affinities"],
                                        batch, device)}
        if model.encoder is not None:
            terms = role_terms(result, batch, device, teacher)
            pieces.update({k: terms[k] for k in TARGET_RATIO})
        for key, value in pieces.items():
            grad = torch.autograd.grad(value, parameter, retain_graph=True,
                                       allow_unused=True)[0]
            norms[key].append(0.0 if grad is None else float(grad.norm()))
        model.zero_grad(set_to_none=True)
    median = {k: (float(np.median(v)) if v else 0.0) for k, v in norms.items()}
    lambdas, clamped = {}, {}
    for key, ratio in TARGET_RATIO.items():
        value = ratio * median["legacy"] / (median[key] + 1e-12)
        lambdas[key] = float(np.clip(value, *LAMBDA_CLAMP))
        clamped[key] = bool(value != lambdas[key])
    return {"parameter": name, "batches": batches, "grad_norm_median": median,
            "lambda": lambdas, "clamped": clamped, "target_ratio": TARGET_RATIO,
            "weighted": {k: lambdas[k] * median[k] for k in TARGET_RATIO}}


# ============================================================================
# Gate A — 32-frame capacity
# ============================================================================
def select_capacity_frames(dataset, count: int = 32) -> list[int]:
    """Deterministic pick implementing the written criteria.

    All eight corners valid comes first -- a capacity test cannot ask whether the
    module can move a wrong corner if the sampled frames have none -- then a
    spread over source roots and over object scale.  Chosen before any result is
    seen and never re-picked afterwards.
    """
    paths = [t[0] for t in dataset.imgs]
    candidates = []
    stride = max(1, len(paths) // 4000)
    for index in range(0, len(paths), stride):
        try:
            sample = dataset[index]
        except Exception:
            continue
        flags = sample.get("refine_keypoints_valid")
        points = sample.get("refine_keypoints")
        if flags is None or points is None:
            continue
        corners = points[:8]
        inside = ((corners[:, 0] >= 0) & (corners[:, 0] < CRA.BELIEF_GRID)
                  & (corners[:, 1] >= 0) & (corners[:, 1] < CRA.BELIEF_GRID))
        if not bool((flags[:8] > 0).all() and inside.all()):
            continue
        span = corners.max(dim=0).values - corners.min(dim=0).values
        scale = float(torch.linalg.norm(span))
        root = next((r for r in GROUPABLE_ROOTS + ("mixed_v8_train",)
                     if r in paths[index]), "other")
        candidates.append((index, root, scale))
        if len(candidates) >= 1200:
            break
    if len(candidates) < count:
        raise SystemExit(f"BLOCKED: only {len(candidates)} all-8-valid frames found")
    by_root: dict[str, list] = {}
    for entry in candidates:
        by_root.setdefault(entry[1], []).append(entry)
    chosen: list[int] = []
    roots = sorted(by_root)
    # inside each root, walk the scale-sorted list so small and large objects
    # both appear rather than whichever the file order happens to give
    for root in roots:
        by_root[root].sort(key=lambda e: e[2])
    position = 0
    while len(chosen) < count and position < max(len(v) for v in by_root.values()):
        for root in roots:
            bucket = by_root[root]
            if not bucket or len(chosen) >= count:
                continue
            pick = bucket[(position * 7) % len(bucket)]
            if pick[0] not in chosen:
                chosen.append(pick[0])
        position += 1
    return sorted(chosen[:count])


def role_diagnostics(result, batch, device, teacher_beliefs) -> dict[str, float]:
    points = batch["refine_keypoints"].to(device)[:, :8]
    flags = batch["refine_keypoints_valid"].to(device)[:, :8]
    valid = CRA.valid_corner_mask(points, flags)
    scores = result["role"]["score"]
    labels, _ = CRL.choose_assignment(scores, points, valid)
    sampled = CRA.bilinear_sample(scores, points)
    predicted = sampled.argmax(dim=-1)
    weight = valid.float()
    accuracy = float(((predicted == labels).float() * weight).sum()
                     / weight.sum().clamp_min(1.0))
    own = CRL._gather_assigned(sampled, labels)

    def beat_rate(peaks):
        other = CRL._gather_assigned(CRA.bilinear_sample(scores, peaks), labels)
        distance = torch.linalg.norm(peaks - points, dim=-1)
        usable = valid & (distance > CRL.WRONG_MIN_CELLS)
        if not bool(usable.any()):
            return float("nan")
        return float(((own > other).float() * usable.float()).sum()
                     / usable.float().sum())

    # structural: own score at GT vs at another GT corner
    batch_size, corners = valid.shape
    distance = torch.cdist(points, points)
    pair = (valid[:, :, None] & valid[:, None, :]
            & (distance > CRL.CLOSE_PAIR_CELLS)
            & ~torch.eye(corners, dtype=torch.bool, device=device)[None])
    index = labels[:, None, :].expand(batch_size, corners, corners)
    at_other = sampled.gather(2, index).transpose(1, 2)
    structural = float(((own[:, :, None] > at_other).float() * pair.float()).sum()
                       / pair.float().sum().clamp_min(1.0))
    return {
        "proto_accuracy": accuracy, "structural_gt_beats_other": structural,
        "gt_beats_student_wrong": beat_rate(
            CRL.peak_coordinates(result["beliefs"][5][:, :8])),
        "gt_beats_teacher_wrong": beat_rate(
            CRL.peak_coordinates(teacher_beliefs[5][:, :8])),
    }


def corner_error_px(result, batch, device) -> tuple[float, float]:
    """Median corner error and centroid error in belief cells -> pixels."""
    points = batch["refine_keypoints"].to(device)
    flags = batch["refine_keypoints_valid"].to(device)
    belief = result["beliefs"][5]
    peaks = CRL.peak_coordinates(belief)
    valid = CRA.valid_corner_mask(points[:, :8], flags[:, :8])
    error = torch.linalg.norm(peaks[:, :8] - points[:, :8], dim=-1)
    corner = float(error[valid].median()) if bool(valid.any()) else float("nan")
    centroid_valid = CRA.valid_corner_mask(points[:, 8:9], flags[:, 8:9])
    centroid_err = torch.linalg.norm(peaks[:, 8:9] - points[:, 8:9], dim=-1)
    centroid = (float(centroid_err[centroid_valid].median())
                if bool(centroid_valid.any()) else float("nan"))
    return corner * 12.8, centroid * 12.8


def run_gate_a(arm: str, frames, model, device, calibration) -> dict[str, Any]:
    parameters = [
        {"params": [p for n, p in model.net.named_parameters()
                    if p.requires_grad and n.startswith("vgg.")], "lr": 1e-5},
        {"params": [p for n, p in model.net.named_parameters()
                    if p.requires_grad and not n.startswith("vgg.")], "lr": 5e-5},
    ]
    extra = []
    if model.encoder is not None:
        extra += list(model.encoder.parameters())
    if model.film is not None:
        extra += list(model.film.parameters())
    if extra:
        parameters.append({"params": extra, "lr": 3e-4})
    optimiser = torch.optim.AdamW(parameters, weight_decay=1e-4)
    history = []
    model.train()
    for step in range(GATE_A_STEPS):
        batch = frames[step % len(frames)]
        images = batch["img"].to(device)
        result = model(images)
        teacher = model.teacher_forward(images)
        losses = role_terms(result, batch, device, teacher)
        legacy = legacy_loss(result["beliefs"], result["affinities"], batch, device)
        total = legacy + sum(calibration["lambda"][k] * losses[k]
                             for k in TARGET_RATIO)
        optimiser.zero_grad(set_to_none=True)
        total.backward()
        optimiser.step()
        if step % 100 == 0 or step == GATE_A_STEPS - 1:
            with torch.no_grad():
                diagnostics = role_diagnostics(result, batch, device, teacher)
                corner, centroid = corner_error_px(result, batch, device)
            history.append({"arm": arm, "step": step, "total": float(total),
                            "legacy": float(legacy), "corner_px": corner,
                            "centroid_px": centroid, **diagnostics})
            log(f"    {arm} step {step:4d} total {float(total):.5f} "
                f"acc {diagnostics['proto_accuracy']:.3f} "
                f"struct {diagnostics['structural_gt_beats_other']:.3f} "
                f"gt>stu {diagnostics['gt_beats_student_wrong']:.3f} "
                f"gt>tea {diagnostics['gt_beats_teacher_wrong']:.3f} "
                f"corner {corner:.1f}px")
    return {"history": history}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-a", action="store_true")
    parser.add_argument("--batch", type=int, default=GATE_A_BATCH)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda")

    log("[A] identity and baseline")
    gate = SCREEN.baseline_reproduction()
    log(f"    strict {gate['strict_n']} gt2d {gate['gt2d_pose_success']} "
        f"pred {gate['pred_pose_success']} yaw {gate['yaw_median_deg']:.6f} "
        f"reproj {gate['fixed_gt_reproj_median_px']:.6f} passed={gate['passed']}")
    if not gate["passed"]:
        raise SystemExit(f"BLOCKED: {gate['problems']}")

    options = SCREEN.canonical_options()
    options.batchsize = args.batch
    dataset, loader, _, _ = SCREEN.build_loader(options)
    log(f"[A] canonical loader {len(dataset)} samples, batch {options.batchsize}")

    sample = torch.zeros(1, 3, options.imagesize, options.imagesize)
    parity = RoleModel(use_film=False, use_role=False)
    features = parity.discover(sample)
    parity.to(device).eval()
    with torch.no_grad():
        probe = torch.randn(2, 3, 400, 400, device=device)
        mine = parity(probe)["beliefs"][5]
        legacy = parity.net(probe)[0][5]
    delta = float((mine - legacy).abs().max())
    log(f"[B] flag-off legacy parity: max|delta| = {delta:.3e}")
    if delta > 1e-6:
        raise SystemExit("BLOCKED: flag-off forward is not the legacy forward")
    del parity
    torch.cuda.empty_cache()

    log(f"[B] F100 vgg[{features['index_high']}] {features['channels_high']}ch  "
        f"F50 vgg[{features['index_low']}] {features['channels_low']}ch")

    if not args.gate_a:
        log("nothing to do; pass --gate-a")
        return 0

    indices = select_capacity_frames(dataset, 32)
    picked = [dataset[i] for i in indices]
    frames = []
    for start in range(0, len(picked), args.batch):
        chunk = picked[start:start + args.batch]
        frames.append({key: torch.stack([c[key] for c in chunk])
                       for key in ("img", "beliefs", "affinities",
                                   "belief_channel_mask", "affinity_channel_mask",
                                   "refine_keypoints", "refine_keypoints_valid")})
    log(f"[H] Gate A on {len(indices)} deterministic frames "
        f"({len(frames)} batches)")
    (OUT / "pcr_capacity_frames.json").write_text(
        json.dumps({"indices": indices,
                    "paths": [dataset.imgs[i][0] for i in indices]}, indent=1))

    results = {}
    for arm, use_film in (("C2", False), ("C3", True)):
        torch.manual_seed(SEED)
        model = RoleModel(use_film=use_film, use_role=True)
        model.discover(sample)
        boundary = model.set_trainable()
        model.to(device)
        log(f"  {arm}: trainable {boundary['param_groups']}")
        if use_film:
            with torch.no_grad():
                out = model(frames[0]["img"].to(device))
                base = model.net(frames[0]["img"].to(device))[0][5]
            identity = float((out["beliefs"][5] - base).abs().max())
            log(f"  {arm}: zero-init identity max|delta| = {identity:.3e}")
            if identity > 1e-6:
                raise SystemExit("BLOCKED: FiLM zero-init is not identity")
        calibration = calibrate(model, [frames[0], frames[1]] * 4, device)
        log(f"  {arm}: lambda " + " ".join(
            f"{k} {calibration['lambda'][k]:.3g}" for k in TARGET_RATIO))
        outcome = run_gate_a(arm, frames, model, device, calibration)
        outcome["calibration"] = calibration
        outcome["param_groups"] = boundary["param_groups"]
        results[arm] = outcome
        del model
        torch.cuda.empty_cache()

    rows = [r for arm in results for r in results[arm]["history"]]
    pd.DataFrame(rows).to_csv(OUT / "pcr_capacity_history.csv", index=False)
    verdict = {}
    for arm, outcome in results.items():
        first, last = outcome["history"][0], outcome["history"][-1]
        checks = [
            ("1 proto accuracy >=0.95", last["proto_accuracy"] >= 0.95,
             last["proto_accuracy"]),
            ("2 structural GT>other >=0.95", last["structural_gt_beats_other"] >= 0.95,
             last["structural_gt_beats_other"]),
            ("3 GT>student wrong >=0.90", last["gt_beats_student_wrong"] >= 0.90,
             last["gt_beats_student_wrong"]),
            ("4 GT>teacher wrong >=0.90", last["gt_beats_teacher_wrong"] >= 0.90,
             last["gt_beats_teacher_wrong"]),
            ("5 corner error -50%",
             last["corner_px"] <= first["corner_px"] * 0.5,
             1.0 - last["corner_px"] / max(first["corner_px"], 1e-9)),
            ("6 centroid <= +10%",
             last["centroid_px"] <= first["centroid_px"] * 1.10,
             last["centroid_px"] / max(first["centroid_px"], 1e-9) - 1.0),
            ("7 no NaN", all(np.isfinite(v) for v in
                             (last["total"], last["corner_px"])), 0.0),
        ]
        verdict[arm] = {"checks": [{"name": n, "passed": bool(p), "value": float(v)}
                                   for n, p, v in checks],
                        "passed": all(p for _, p, _ in checks),
                        "calibration": outcome["calibration"],
                        "param_groups": outcome["param_groups"]}
    (OUT / "pcr_gate_a.json").write_text(json.dumps(MD.jsonable(verdict), indent=1),
                                         encoding="utf-8")
    log("\n[H] Gate A verdict")
    for arm, value in verdict.items():
        for check in value["checks"]:
            log(f"  {arm}  {'PASS' if check['passed'] else 'FAIL'}  "
                f"{check['name']:<30} {check['value']:>9.4f}")
        log(f"  {arm}  -> {'PASS' if value['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
