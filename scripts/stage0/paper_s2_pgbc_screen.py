"""PGBC feasibility gates — three learning-free screens run before any training.

The question is whether a bounded residual on top of the frozen ep57 belief can
move far-face corners at all, whether the frozen 50x50 feature even knows where
the GT corner is, and whether the seven remaining corners carry enough
information to place the eighth.  All three are answered from the existing
mechanism cache plus one extra forward for the shared feature; nothing is
trained here.

    python scripts/stage0/paper_s2_pgbc_screen.py --all
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data/pallet/results/paper_s2_pgbc_screen"
STAGE0 = ROOT / "scripts/stage0"
if str(STAGE0) not in sys.path:
    sys.path.insert(0, str(STAGE0))


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MD = _load("MD", STAGE0 / "paper_s2_mechanism_diagnostic.py")
FZ = MD.FZ

BELIEF = MD.BELIEF                      # 50
N_KP = MD.N_KP                          # 9 (8 corners + centroid)
NEAR_KP, FAR_KP = MD.NEAR_KP, MD.FAR_KP  # camera-facing 0123 grouping, reused
STAGE6 = 5                              # belief_stages is (6, 9, 50, 50)
AMPLITUDE = 0.25                        # fixed; never tuned on results
SEED = 1


def log(message: str) -> None:
    print(message, flush=True)


# ============================================================================
# Phase A — identity, baseline reproduction, shared context
# ============================================================================
def checkpoint_sha() -> str:
    path = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_context() -> dict[str, Any]:
    """Manifest + geometries + cached stage-6 belief + baseline decode."""
    manifest = json.loads(MD.MANIFEST_PATH.read_text(encoding="utf-8"))
    audit = FZ.InputAudit()
    tensors = MD.load_cached_tensors()

    geometries: dict[str, Any] = {}
    belief6: dict[str, np.ndarray] = {}
    stages: dict[str, np.ndarray] = {}
    decoded: dict[str, Any] = {}
    scales: dict[str, tuple[float, float]] = {}
    # the manifest also carries the exploratory manual population; the cache and
    # every baseline number are the strict primary set only.
    primary = [f for f in manifest["frames"] if f["population"] == "primary"]
    manifest = dict(manifest, frames=primary)
    for spec in primary:
        uid = spec["frame_id"]
        geometry = MD.FrameGeometry(spec, audit)
        stack = tensors[f"{uid}|belief_stages"]
        assert stack.shape == (6, N_KP, BELIEF, BELIEF), stack.shape
        scale_x = float(spec["image_width"]) / BELIEF
        scale_y = float(spec["image_height"]) / BELIEF
        geometries[uid] = geometry
        stages[uid] = stack
        belief6[uid] = stack[STAGE6]
        scales[uid] = (scale_x, scale_y)
        decoded[uid] = MD.decode_all(stack[STAGE6], scale_x, scale_y, geometry.gt_points)

    gate = MD.baseline_gate(manifest, geometries, decoded)
    if audit.prohibited_attempts:
        raise RuntimeError(f"final-test access attempted: {audit.prohibited_attempts}")
    return {"manifest": manifest, "geometries": geometries, "belief6": belief6,
            "stages": stages, "decoded": decoded, "scales": scales,
            "baseline_gate": gate, "audit": audit}


def failure_classes() -> pd.DataFrame:
    path = MD.OUT_DIR / "failure_class_frames.csv"
    if not path.is_file():
        raise FileNotFoundError(f"run the mechanism diagnostic first: {path}")
    return pd.read_csv(path)


# ============================================================================
# shared small helpers
# ============================================================================
def grid_of(point: Optional[list[float]], scale: tuple[float, float]
            ) -> Optional[np.ndarray]:
    if point is None:
        return None
    return np.array([point[0] / scale[0], point[1] / scale[1]], float)


def global_coordinate(heat: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Full-map spatial softmax expectation, in belief-grid units."""
    flat = heat.reshape(-1).astype(np.float64) / temperature
    weights = np.exp(flat - flat.max())
    weights /= weights.sum()
    ys, xs = np.mgrid[0:heat.shape[0], 0:heat.shape[1]]
    return np.array([float((weights * xs.reshape(-1)).sum()),
                     float((weights * ys.reshape(-1)).sum())])


def top1_coordinate(heat: np.ndarray) -> np.ndarray:
    y, x = np.unravel_index(int(np.argmax(heat)), heat.shape)
    return np.array([float(x), float(y)])


def oracle_residual(shape: tuple[int, int], centre: np.ndarray) -> np.ndarray:
    """+A inside the 3x3 around the GT cell, -A everywhere else."""
    delta = np.full(shape, -AMPLITUDE, np.float32)
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    y0, y1 = max(0, cy - 1), min(shape[0], cy + 2)
    x0, x1 = max(0, cx - 1), min(shape[1], cx + 2)
    delta[y0:y1, x0:x1] = AMPLITUDE
    return delta


def px_error(grid_point: np.ndarray, gt_grid: np.ndarray,
             scale: tuple[float, float]) -> float:
    diff = (grid_point - gt_grid) * np.array(scale)
    return float(np.hypot(diff[0], diff[1]))


# ============================================================================
# G0 — residual capacity of a fixed +-0.25 additive residual
# ============================================================================
def run_g0(ctx: dict[str, Any], classes: pd.DataFrame) -> pd.DataFrame:
    f2 = set(classes.loc[classes.failure_class == "F2_CONFIDENT_WRONG", "frame_id"])
    rows = []
    for spec in ctx["manifest"]["frames"]:
        uid = spec["frame_id"]
        if uid not in f2:
            continue
        geometry, scale = ctx["geometries"][uid], ctx["scales"][uid]
        belief = ctx["belief6"][uid]
        for k in range(8):
            gt = grid_of(geometry.gt_points[k], scale)
            if gt is None:
                continue
            heat = belief[k].astype(np.float32)
            # a truncated GT corner can fall outside the 50x50 map; the residual
            # physically cannot place a peak there, so it is kept and flagged
            # rather than dropped (dropping would inflate the pass rate).
            in_grid = bool(0 <= gt[0] <= BELIEF - 1 and 0 <= gt[1] <= BELIEF - 1)
            cell = (int(np.clip(round(gt[1]), 0, BELIEF - 1)),
                    int(np.clip(round(gt[0]), 0, BELIEF - 1)))
            refined = heat + oracle_residual(heat.shape, gt)
            base_top1 = top1_coordinate(heat)
            base_glob = global_coordinate(heat)
            ref_top1 = top1_coordinate(refined)
            ref_glob = global_coordinate(refined)
            rows.append({
                "frame_id": uid, "session_id": spec["session_id"],
                "domain": spec["domain"], "corner": k,
                "group": "far" if k in FAR_KP else "near",
                "peak_base": float(heat.max()),
                "gt_in_grid": in_grid,
                "peak_at_gt": float(heat[cell]),
                "peak_refined_at_gt": float(refined[cell]),
                "peak_refined_max": float(refined.max()),
                "err_top1_base": px_error(base_top1, gt, scale),
                "err_top1_ref": px_error(ref_top1, gt, scale),
                "err_glob_base": px_error(base_glob, gt, scale),
                "err_glob_ref": px_error(ref_glob, gt, scale),
                "argmax_moved_to_gt": bool(
                    np.abs(ref_top1 - np.round(gt)).max() <= 1.0),
            })
    table = pd.DataFrame(rows)
    for mode in ("top1", "glob"):
        base, ref = table[f"err_{mode}_base"], table[f"err_{mode}_ref"]
        table[f"reduction_{mode}"] = np.where(base > 1e-9, 1.0 - ref / base, 0.0)
    return table


def g0_gate(table: pd.DataFrame) -> dict[str, Any]:
    far = table[table.group == "far"]
    result: dict[str, Any] = {"n_f2_frames": int(table.frame_id.nunique()),
                              "n_corners_far": int(len(far)),
                              "n_corners_all": int(len(table))}
    for mode in ("top1", "glob"):
        share = float((far[f"reduction_{mode}"] >= 0.50).mean()) if len(far) else 0.0
        result[f"far_share_50pct_{mode}"] = share
        result[f"far_median_base_{mode}"] = float(far[f"err_{mode}_base"].median())
        result[f"far_median_ref_{mode}"] = float(far[f"err_{mode}_ref"].median())
    result["argmax_moved_to_gt_rate"] = float(far["argmax_moved_to_gt"].mean()) \
        if len(far) else 0.0
    # stratified by how wrong the corner started: a corner that was already
    # right cannot show a 50% reduction, and the confidently-wrong regime is the
    # one PGBC targets.  Reported as breakdown; the verdict stays the pre-fixed rule.
    strata = []
    for low, high, name in ((0, 5, "<5px"), (5, 20, "5-20px"),
                            (20, 50, "20-50px"), (50, 1e9, ">50px")):
        block = far[(far.err_top1_base >= low) & (far.err_top1_base < high)]
        if not len(block):
            continue
        strata.append({
            "bin": name, "n": int(len(block)),
            "share_50pct_top1": float((block.reduction_top1 >= 0.50).mean()),
            "argmax_moved_to_gt": float(block.argmax_moved_to_gt.mean()),
            "peak_base_median": float(block.peak_base.median()),
            "peak_at_gt_median": float(block.peak_at_gt.median()),
            "gt_wins_after_residual": float(
                (block.peak_at_gt + AMPLITUDE > block.peak_base - AMPLITUDE).mean()),
            "err_median_base": float(block.err_top1_base.median()),
            "err_median_refined": float(block.err_top1_ref.median()),
        })
    result["strata"] = strata
    result["gt_outside_grid"] = int((~far.gt_in_grid).sum())
    result["belief_cell_px"] = "1 cell = 12.8 x 9.6 px, so a top1 read-out cannot beat ~1 cell"
    result["threshold"] = {"share_of_far_corners_with_50pct_reduction": 0.80}
    result["passed"] = bool(result["far_share_50pct_top1"] >= 0.80)
    result["note"] = ("PASS uses the top1 read-out because the fixed additive "
                      "residual has to win the argmax to change the decoded point; "
                      "the global read-out is reported alongside.")
    return result


# ============================================================================
# G1 — is the GT corner observable in the frozen 50x50 feature?
# ============================================================================
class SharedFeature:
    """One forward per frame, hooked on the frozen VGG trunk."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = FZ.load_model(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.captured: Optional[torch.Tensor] = None
        trunk = self.model.vgg
        trunk.register_forward_hook(
            lambda m, i, o: setattr(self, "captured", o.detach()))

    @torch.inference_mode()
    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        tensor = FZ.preprocess_squash(image_bgr).to(self.device)
        self.model(tensor)
        feature = self.captured
        assert feature is not None and feature.shape[-2:] == (BELIEF, BELIEF), \
            f"shared feature is not {BELIEF}x{BELIEF}: {None if feature is None else feature.shape}"
        return feature[0].float().cpu().numpy()


def sample_feature(feature: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Bilinear read of a C x 50 x 50 map at a grid coordinate."""
    x = float(np.clip(point[0], 0, BELIEF - 1))
    y = float(np.clip(point[1], 0, BELIEF - 1))
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, BELIEF - 1), min(y0 + 1, BELIEF - 1)
    ax, ay = x - x0, y - y0
    return (feature[:, y0, x0] * (1 - ax) * (1 - ay)
            + feature[:, y0, x1] * ax * (1 - ay)
            + feature[:, y1, x0] * (1 - ax) * ay
            + feature[:, y1, x1] * ax * ay)


def run_g1(ctx: dict[str, Any], classes: pd.DataFrame,
           min_wrong_px: float = 20.0) -> pd.DataFrame:
    """One positive (GT location) and one negative (wrong peak) per bad corner."""
    f2 = set(classes.loc[classes.failure_class == "F2_CONFIDENT_WRONG", "frame_id"])
    extractor = SharedFeature()
    audit = ctx["audit"]
    rows = []
    for spec in ctx["manifest"]["frames"]:
        uid = spec["frame_id"]
        if uid not in f2:
            continue
        geometry, scale = ctx["geometries"][uid], ctx["scales"][uid]
        image = audit.read_image(spec["image_path"])
        feature = extractor(image)
        belief = ctx["belief6"][uid]
        dims = np.asarray(geometry.dims, float)
        for k in range(8):
            gt = grid_of(geometry.gt_points[k], scale)
            if gt is None:
                continue
            peak = top1_coordinate(belief[k].astype(np.float32))
            if px_error(peak, gt, scale) < min_wrong_px:
                continue  # this corner is not wrong; nothing to discriminate
            rows.append({
                "frame_id": uid, "session_id": spec["session_id"],
                "domain": spec["domain"], "corner": k,
                "group": "far" if k in FAR_KP else "near",
                "wrong_px": px_error(peak, gt, scale),
                "f_gt": sample_feature(feature, gt).astype(np.float32),
                "f_wrong": sample_feature(feature, peak).astype(np.float32),
                "dims": dims.astype(np.float32),
            })
    return pd.DataFrame(rows)


def session_folds(sessions: list[str], n_folds: int = 3) -> list[list[str]]:
    """Greedy balanced grouping of whole capture sessions."""
    counts = pd.Series(sessions).value_counts()
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for name, count in counts.items():
        index = int(np.argmin(sizes))
        folds[index].append(str(name))
        sizes[index] += int(count)
    return folds


def logistic_probe(x_train, y_train, x_test, steps: int = 400):
    """Tiny linear probe, standardised, no hidden layer."""
    torch.manual_seed(SEED)
    mean, std = x_train.mean(0, keepdim=True), x_train.std(0, keepdim=True) + 1e-6
    xt, xe = (x_train - mean) / std, (x_test - mean) / std
    model = torch.nn.Linear(xt.shape[1], 1)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        optimiser.zero_grad()
        loss = loss_fn(model(xt).squeeze(1), y_train)
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        return model(xe).squeeze(1).numpy()


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive, negative = labels == 1, labels == 0
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def g1_evaluate(table: pd.DataFrame, use_feature: bool = True) -> dict[str, Any]:
    """Session-grouped 3-fold; pooled AUC plus the paired GT>wrong rate."""
    folds = session_folds(table.session_id.tolist())
    identity = np.eye(8, dtype=np.float32)
    per_fold = []
    for held_out in folds:
        train = table[~table.session_id.isin(held_out)]
        test = table[table.session_id.isin(held_out)]
        if not len(train) or not len(test):
            continue

        def design(frame: pd.DataFrame):
            blocks, labels, pair = [], [], []
            for index, (_, row) in enumerate(frame.iterrows()):
                context = np.concatenate([identity[row["corner"]], row["dims"]])
                for key, label in (("f_gt", 1.0), ("f_wrong", 0.0)):
                    part = np.asarray(row[key]) if use_feature else np.zeros(0, np.float32)
                    blocks.append(np.concatenate([part, context]))
                    labels.append(label)
                    pair.append(index)
            return (torch.tensor(np.stack(blocks), dtype=torch.float32),
                    torch.tensor(labels, dtype=torch.float32), np.array(pair))

        x_train, y_train, _ = design(train)
        x_test, y_test, pair = design(test)
        scores = logistic_probe(x_train, y_train, x_test)
        labels = y_test.numpy()
        wins = [scores[(pair == p) & (labels == 1)][0] > scores[(pair == p) & (labels == 0)][0]
                for p in np.unique(pair)]
        per_fold.append({
            "held_out_sessions": held_out, "n_train_pairs": len(train),
            "n_test_pairs": len(test), "auc": auc(scores, labels),
            "accuracy": float(((scores > 0) == (labels == 1)).mean()),
            "gt_beats_wrong_rate": float(np.mean(wins)),
        })
    return {"folds": per_fold, "use_feature": use_feature}


def g1_gate(main: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    aucs = [f["auc"] for f in main["folds"]]
    accs = [f["accuracy"] for f in main["folds"]]
    wins = [f["gt_beats_wrong_rate"] for f in main["folds"]]
    passed = bool(len(aucs) == 3
                  and all(max(a, b) >= 0.75 for a, b in zip(aucs, accs))
                  and min(wins) >= 0.70)
    return {
        "fold_auc": aucs, "fold_accuracy": accs, "fold_gt_beats_wrong": wins,
        "control_no_feature_auc": [f["auc"] for f in control["folds"]],
        "control_no_feature_gt_beats_wrong": [
            f["gt_beats_wrong_rate"] for f in control["folds"]],
        "threshold": {"every_fold_auc_or_accuracy": 0.75, "gt_beats_wrong": 0.70},
        "passed": passed,
        "note": ("the control drops the 50x50 feature and keeps only corner ID and "
                 "dimensions, which are identical within a pair, so it must sit at "
                 "chance; any discrimination therefore comes from the feature."),
    }


# ============================================================================
# G2 — do the other seven corners predict the eighth?
# ============================================================================
def run_g2(ctx: dict[str, Any], classes: pd.DataFrame) -> pd.DataFrame:
    f2 = set(classes.loc[classes.failure_class == "F2_CONFIDENT_WRONG", "frame_id"])
    rows = []
    for spec in ctx["manifest"]["frames"]:
        uid = spec["frame_id"]
        if uid not in f2:
            continue
        geometry, scale = ctx["geometries"][uid], ctx["scales"][uid]
        predicted = ctx["decoded"][uid]["D0"]
        for k in range(8):
            gt = geometry.gt_points[k]
            if gt is None or predicted[k] is None:
                continue
            held = [None if index == k else predicted[index]
                    for index in range(len(predicted))]
            held[8] = None  # centroid is never a PnP correspondence
            n_used = sum(1 for index in range(8) if held[index] is not None)
            pose = geometry.solve(held) if n_used >= 4 else None
            if pose is None:
                rows.append({"frame_id": uid, "session_id": spec["session_id"],
                             "corner": k, "group": "far" if k in FAR_KP else "near",
                             "n_used": n_used, "solved": False,
                             "err_base": px_error(np.asarray(predicted[k]) / np.asarray(scale),
                                                  np.asarray(gt) / np.asarray(scale), scale),
                             "err_graph": np.nan, "dx_base": np.nan, "dy_base": np.nan,
                             "dx_graph": np.nan, "dy_graph": np.nan})
                continue
            object_points = MD.APNP.make_pallet_keypoints_3d(
                *tuple(pose.get("dims", geometry.dims)))
            projected = MD.APNP.project_3d(
                object_points, np.asarray(pose["R"], float),
                np.asarray(pose["t"], float), geometry.K)
            point = projected[k]
            if point is None or not np.all(np.isfinite(np.asarray(point, float))):
                continue
            rows.append({
                "frame_id": uid, "session_id": spec["session_id"], "corner": k,
                "group": "far" if k in FAR_KP else "near",
                "n_used": n_used, "solved": True,
                "err_base": float(np.hypot(predicted[k][0] - gt[0], predicted[k][1] - gt[1])),
                "err_graph": float(np.hypot(point[0] - gt[0], point[1] - gt[1])),
                "dx_base": float(predicted[k][0] - gt[0]),
                "dy_base": float(predicted[k][1] - gt[1]),
                "dx_graph": float(point[0] - gt[0]),
                "dy_graph": float(point[1] - gt[1]),
            })
    return pd.DataFrame(rows)


def g2_gate(table: pd.DataFrame) -> dict[str, Any]:
    far = table[(table.group == "far") & table.solved].dropna(
        subset=["err_graph", "err_base"])
    if not len(far):
        return {"n_far": 0, "passed": False, "note": "no solvable far corner"}
    bias_base = np.hypot(far.dx_base.mean(), far.dy_base.mean())
    bias_graph = np.hypot(far.dx_graph.mean(), far.dy_graph.mean())
    median_base = float(far.err_base.median())
    median_graph = float(far.err_graph.median())
    error_drop = 1.0 - median_graph / median_base if median_base > 0 else 0.0
    bias_drop = 1.0 - bias_graph / bias_base if bias_base > 0 else 0.0
    return {
        "n_far": int(len(far)), "n_far_unsolved": int(
            ((table.group == "far") & ~table.solved).sum()),
        "median_err_base_px": median_base, "median_err_graph_px": median_graph,
        "error_reduction": float(error_drop),
        "signed_bias_base_px": float(bias_base),
        "signed_bias_graph_px": float(bias_graph),
        "bias_reduction": float(bias_drop),
        "paired_improved": int((far.err_graph < far.err_base).sum()),
        "paired_worsened": int((far.err_graph > far.err_base).sum()),
        "threshold": {"error_reduction": 0.20, "bias_reduction": 0.20},
        "passed": bool(error_drop >= 0.20 and bias_drop >= 0.20),
    }


# ============================================================================
# main
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--g0", action="store_true")
    parser.add_argument("--g1", action="store_true")
    parser.add_argument("--g2", action="store_true")
    args = parser.parse_args()
    if not (args.g0 or args.g1 or args.g2):
        args.all = True
    OUT.mkdir(parents=True, exist_ok=True)

    log("[A] identity + baseline reproduction")
    context = build_context()
    gate = context["baseline_gate"]
    log(f"    strict {gate['strict_n']}  gt2d {gate['gt2d_pose_success']}  "
        f"pred {gate['pred_pose_success']}  yaw {gate['yaw_median_deg']:.6f}  "
        f"reproj {gate['fixed_gt_reproj_median_px']:.6f}  passed={gate['passed']}")
    if not gate["passed"]:
        raise SystemExit(f"BLOCKED: baseline reproduction failed {gate['problems']}")
    classes = failure_classes()
    log(f"    F2 {int((classes.failure_class == 'F2_CONFIDENT_WRONG').sum())}  "
        f"F1 {int((classes.failure_class == 'F1_NO_RESPONSE').sum())}")

    results: dict[str, Any] = {
        "head": MD.git_head(), "checkpoint_sha256": checkpoint_sha(),
        "baseline_gate": MD.jsonable(gate),
        "amplitude": AMPLITUDE, "seed": SEED,
    }

    if args.all or args.g0:
        log("[G0] residual capacity of a fixed +-0.25 additive residual")
        table = run_g0(context, classes)
        table.to_csv(OUT / "pgbc_g0_residual_capacity.csv", index=False)
        verdict = g0_gate(table)
        results["G0"] = verdict
        log(f"    far corners {verdict['n_corners_far']}  "
            f"share>=50% (top1) {verdict['far_share_50pct_top1']:.3f}  "
            f"(global) {verdict['far_share_50pct_glob']:.3f}  "
            f"argmax moved {verdict['argmax_moved_to_gt_rate']:.3f}  "
            f"-> {'PASS' if verdict['passed'] else 'FAIL'}")

    if args.all or args.g1:
        log("[G1] feature observability probe on the frozen 50x50 feature")
        table = run_g1(context, classes)
        slim = table.drop(columns=["f_gt", "f_wrong", "dims"])
        slim.to_csv(OUT / "pgbc_g1_probe_samples.csv", index=False)
        main_run = g1_evaluate(table, use_feature=True)
        control = g1_evaluate(table, use_feature=False)
        verdict = g1_gate(main_run, control)
        results["G1"] = verdict
        log(f"    pairs {len(table)}  fold AUC "
            f"{[round(a, 3) for a in verdict['fold_auc']]}  "
            f"GT>wrong {[round(w, 3) for w in verdict['fold_gt_beats_wrong']]}  "
            f"control AUC {[round(a, 3) for a in verdict['control_no_feature_auc']]}  "
            f"-> {'PASS' if verdict['passed'] else 'FAIL'}")

    if args.all or args.g2:
        log("[G2] leave-one-corner-out graph information")
        table = run_g2(context, classes)
        table.to_csv(OUT / "pgbc_g2_leave_one_out.csv", index=False)
        verdict = g2_gate(table)
        results["G2"] = verdict
        log(f"    far {verdict['n_far']}  median {verdict['median_err_base_px']:.2f} "
            f"-> {verdict['median_err_graph_px']:.2f} px "
            f"({verdict['error_reduction']*100:.1f}%)  bias "
            f"{verdict['signed_bias_base_px']:.2f} -> "
            f"{verdict['signed_bias_graph_px']:.2f} px "
            f"({verdict['bias_reduction']*100:.1f}%)  "
            f"-> {'PASS' if verdict['passed'] else 'FAIL'}")

    (OUT / "pgbc_gate.json").write_text(json.dumps(MD.jsonable(results), indent=1),
                                        encoding="utf-8")
    log(f"[done] {OUT / 'pgbc_gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
