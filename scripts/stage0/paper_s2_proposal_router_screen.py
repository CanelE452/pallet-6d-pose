"""Re-evaluate the frozen epoch-5 proposal with raw-Q decoders, then measure how
complementary base and proposal coordinates actually are.

Nothing is retrained here: the epoch-5 checkpoint is loaded read-only and the
learned router, if the oracle gate allows it, is a small MLP over per-corner
statistics.  The anchor is ep57 (C0), not the fine-tuned base, because the
fine-tuned base already regressed on real.

    python scripts/stage0/paper_s2_proposal_router_screen.py --all
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data/pallet/results/paper_s2_corner_replacement_screen"
STAGE0 = ROOT / "scripts/stage0"
DOPE = ROOT / "Deep_Object_Pose"
for extra in (STAGE0, DOPE / "common", DOPE / "train"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EP57 = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
EP5 = ROOT / "weights/paper_s2_corner_replacement_screen/epoch_005.pth"
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"
EP5_SHA = "aad97f6bead6067d58ae178e99e404c738f63708e38de622ca3a6f07087da4e5"
SEED = 1
MARGIN = 3.0

FOLDS = {
    "A": ["capturepallet08", "capturenight05"],
    "B": ["capturepallet03", "capturepallet02", "capturenight07"],
    "C": ["capturepallet04", "capturepallet05", "capturenight06"],
}


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MD = _load("MD", STAGE0 / "paper_s2_mechanism_diagnostic.py")
SCREEN = _load("SCREEN", STAGE0 / "paper_s2_corner_replacement_screen.py")
FZ = MD.FZ
import corner_branch_router as CBR  # noqa: E402

NEAR, FAR = MD.NEAR_KP, MD.FAR_KP


def log(message: str) -> None:
    print(message, flush=True)


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ============================================================================
# Phase A/C — one forward per frame, every coordinate path
# ============================================================================
@torch.no_grad()
def collect() -> pd.DataFrame:
    """Per (frame, corner): C0, C1-base and the three raw-Q proposal readouts."""
    if not EP5.is_file():
        raise SystemExit(f"BLOCKED: epoch-5 checkpoint missing at {EP5}")
    for path, expected in ((EP57, EP57_SHA), (EP5, EP5_SHA)):
        if sha(path) != expected:
            raise SystemExit(f"BLOCKED: {path.name} SHA mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    canonical = SCREEN.unit_canonical()

    baseline = SCREEN.ScreenModel()
    sample = torch.zeros(1, 3, 400, 400)
    baseline.discover(sample)
    baseline.set_trainable()
    baseline.to(device).eval()

    tuned = SCREEN.ScreenModel()
    tuned.discover(sample)
    tuned.set_trainable()
    payload = torch.load(str(EP5), map_location="cpu", weights_only=True)
    tuned.net.load_state_dict(payload["net"], strict=True)
    tuned.branch.load_state_dict(payload["branch"], strict=True)
    tuned.to(device).eval()
    for model in (baseline, tuned):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    audit = FZ.InputAudit()
    rows = []
    for spec in SCREEN.mechanism_frames():
        uid = spec["frame_id"]
        geometry = MD.FrameGeometry(spec, audit)
        image = audit.read_image(spec["image_path"])
        tensor = FZ.preprocess_squash(image).to(device)
        dims = torch.tensor(np.asarray(geometry.dims, np.float32))[None].to(device)
        dims = dims / dims.amax(dim=-1, keepdim=True).clamp_min(1e-3)
        scale = (spec["image_width"] / 50.0, spec["image_height"] / 50.0)

        c0 = baseline.forward_full(tensor, canonical[None].to(device), dims)
        c1 = tuned.forward_full(tensor, canonical[None].to(device), dims)
        belief0 = c0["beliefs"][5][0].float().cpu().numpy()
        belief1 = c1["beliefs"][5][0].float().cpu().numpy()
        logits = c1["proposal"][0].float().cpu().numpy()
        h4 = c1["beliefs"][3][0, :8].float().cpu().numpy()
        h5 = c1["beliefs"][4][0, :8].float().cpu().numpy()

        points = {
            "C0": MD.decode_all(belief0, *scale, geometry.gt_points)["D0"],
            "C1base": MD.decode_all(belief1, *scale, geometry.gt_points)["D0"],
        }
        for name, decode in CBR.DECODERS.items():
            points[f"P{name}"] = decode(logits, scale) + [None]

        for corner in range(8):
            gt = geometry.gt_points[corner]
            entry = {
                "frame_id": uid, "session_id": spec["session_id"],
                "domain": spec["domain"], "corner": corner,
                "group": "far" if corner in FAR else "near",
                "gt_x": None if gt is None else float(gt[0]),
                "gt_y": None if gt is None else float(gt[1]),
                # base-branch statistics (features, never GT)
                "h4_peak": float(h4[corner].max()),
                "h5_peak": float(h5[corner].max()),
                "h6_peak": float(belief0[corner].max()),
                "h6_gap": CBR.top_two_gap(belief0[corner]),
                "h6_entropy": CBR.spatial_entropy(belief0[corner]),
                "h6_sharpness": CBR.local_sharpness(belief0[corner]),
                # proposal-branch statistics
                "q_gap": CBR.top_two_gap(logits[corner]),
                "q_entropy": CBR.spatial_entropy(logits[corner]),
                "q_sharpness": CBR.local_sharpness(logits[corner]),
            }
            for name, series in points.items():
                point = series[corner]
                entry[f"{name}_x"] = None if point is None else float(point[0])
                entry[f"{name}_y"] = None if point is None else float(point[1])
                if gt is None or point is None:
                    entry[f"{name}_err"] = np.nan
                    entry[f"{name}_dx"] = np.nan
                    entry[f"{name}_dy"] = np.nan
                else:
                    entry[f"{name}_err"] = float(np.hypot(point[0] - gt[0],
                                                          point[1] - gt[1]))
                    entry[f"{name}_dx"] = float(point[0] - gt[0])
                    entry[f"{name}_dy"] = float(point[1] - gt[1])
            rows.append(entry)
    if audit.prohibited_attempts:
        raise RuntimeError(f"final-test access: {audit.prohibited_attempts}")
    return pd.DataFrame(rows)


# ============================================================================
# pose evaluation for an arbitrary per-corner coordinate set
# ============================================================================
def pose_metrics(table: pd.DataFrame, column: str) -> dict[str, Any]:
    audit = FZ.InputAudit()
    success = 0
    yaws, reprojs = [], []
    for spec in SCREEN.mechanism_frames():
        uid = spec["frame_id"]
        geometry = MD.FrameGeometry(spec, audit)
        subset = table[table.frame_id == uid].sort_values("corner")
        points: list[Optional[list[float]]] = []
        for _, row in subset.iterrows():
            x, y = row[f"{column}_x"], row[f"{column}_y"]
            points.append(None if (pd.isna(x) or pd.isna(y)) else [float(x), float(y)])
        points.append(None)  # centroid is never a correspondence
        pose = geometry.solve(points)
        success += int(pose is not None)
        if pose is not None:
            metrics = geometry.metrics(pose)
            if metrics["yaw_err_deg"] is not None:
                yaws.append(metrics["yaw_err_deg"])
            if metrics["reproj_fixed_gt_px"] is not None:
                reprojs.append(metrics["reproj_fixed_gt_px"])
    return {"pose_success": success,
            "yaw_median_deg": float(np.median(yaws)) if yaws else np.nan,
            "reproj_median_px": float(np.median(reprojs)) if reprojs else np.nan}


def corner_summary(table: pd.DataFrame, column: str, f2: set[str]) -> dict[str, Any]:
    errors = table[f"{column}_err"]
    far = table[table.group == "far"]
    subset = table[table.frame_id.isin(f2)]
    far_f2 = subset[subset.group == "far"]
    clean = far_f2.dropna(subset=[f"{column}_dx", f"{column}_dy"])
    return {
        "arm": column,
        "median_px": float(errors.median()),
        "near_median_px": float(table[table.group == "near"][f"{column}_err"].median()),
        "far_median_px": float(far[f"{column}_err"].median()),
        "f2_far_median_px": float(far_f2[f"{column}_err"].median()),
        "f2_far_signed_bias_px": float(np.hypot(clean[f"{column}_dx"].mean(),
                                                clean[f"{column}_dy"].mean()))
        if len(clean) else np.nan,
        "p90_px": float(errors.quantile(0.90)),
        "tail_gt20": int((errors > 20).sum()),
        "tail_gt50": int((errors > 50).sum()),
        "tail_gt100": int((errors > 100).sum()),
        "nan_err": int(errors.isna().sum()),
    }


# ============================================================================
# main
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log("[A] identity / checkpoint gate")
    gate = SCREEN.baseline_reproduction()
    log(f"    baseline strict {gate['strict_n']} gt2d {gate['gt2d_pose_success']} "
        f"pred {gate['pred_pose_success']} yaw {gate['yaw_median_deg']:.6f} "
        f"reproj {gate['fixed_gt_reproj_median_px']:.6f} passed={gate['passed']}")
    if not gate["passed"]:
        raise SystemExit(f"BLOCKED: {gate['problems']}")

    log("[C] one forward per frame, raw-Q decoders")
    table = collect()
    table.to_parquet(OUT / "proposal_router_corners.parquet")
    classes = pd.read_csv(MD.OUT_DIR / "failure_class_frames.csv")
    f2 = set(classes.loc[classes.failure_class == "F2_CONFIDENT_WRONG", "frame_id"])

    arms = ["C0", "C1base", "Pargmax", "Plocal", "Pdsnt"]
    rows = []
    for arm in arms:
        summary = corner_summary(table, arm, f2)
        summary.update(pose_metrics(table, arm))
        rows.append(summary)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "proposal_decoder_metrics.csv", index=False)
    log("\n" + metrics.to_string(index=False))

    # complementarity, always computed even if the decoders look bad
    base_err = table["C0_err"].to_numpy()
    prop_err = table["Plocal_err"].to_numpy()
    valid = np.isfinite(base_err) & np.isfinite(prop_err)
    better = np.zeros(len(table), bool)
    better[valid] = prop_err[valid] < base_err[valid]
    table["proposal_better"] = better
    table["proposal_gain_px"] = np.where(valid, base_err - prop_err, np.nan)
    weak = table.h6_peak < 0.1
    confident_wrong = (table.h6_peak >= 0.5) & (table.C0_err > 20)
    complementarity = []
    for name, mask in (("all", np.ones(len(table), bool)),
                       ("far", (table.group == "far").to_numpy()),
                       ("near", (table.group == "near").to_numpy()),
                       ("F2", table.frame_id.isin(f2).to_numpy()),
                       ("weak_corner", weak.to_numpy()),
                       ("confident_wrong", confident_wrong.to_numpy())):
        block = table[mask & valid]
        if not len(block):
            continue
        complementarity.append({
            "group": name, "n": int(len(block)),
            "proposal_better": float(block.proposal_better.mean()),
            "better_by_3px": float((block.proposal_gain_px > 3).mean()),
            "better_by_10px": float((block.proposal_gain_px > 10).mean()),
            "median_base_px": float(block.C0_err.median()),
            "median_proposal_px": float(block.Plocal_err.median()),
        })
    complement = pd.DataFrame(complementarity)
    complement.to_csv(OUT / "proposal_corner_complementarity.csv", index=False)
    log("\n" + complement.to_string(index=False))

    # oracle routing
    for tag, margin in (("oracle_exact", 0.0), ("oracle_margin", MARGIN)):
        take = np.zeros(len(table), bool)
        take[valid] = CBR.route_oracle(base_err[valid], prop_err[valid], margin)
        table[f"{tag}_take"] = take
        for axis in ("x", "y"):
            table[f"{tag}_{axis}"] = np.where(take, table[f"Plocal_{axis}"],
                                              table[f"C0_{axis}"])
        table[f"{tag}_err"] = np.where(take, table["Plocal_err"], table["C0_err"])
        for axis in ("dx", "dy"):
            table[f"{tag}_{axis}"] = np.where(take, table[f"Plocal_{axis}"],
                                              table[f"C0_{axis}"])

    oracle_rows = []
    for tag in ("oracle_exact", "oracle_margin"):
        summary = corner_summary(table, tag, f2)
        summary.update(pose_metrics(table, tag))
        summary["proposal_share"] = float(table[f"{tag}_take"].mean())
        oracle_rows.append(summary)
    oracle = pd.DataFrame(oracle_rows)
    oracle.to_csv(OUT / "proposal_oracle_router.csv", index=False)
    log("\n" + oracle.to_string(index=False))
    table.to_parquet(OUT / "proposal_router_corners.parquet")

    c0 = rows[0]
    plocal = [r for r in rows if r["arm"] == "Plocal"][0]

    def drop(before, after):
        return float(1.0 - after / before) if before else 0.0

    interface = {
        "primary_decoder": "Plocal",
        "conditions": {
            "A f2_far_median -10%": drop(c0["f2_far_median_px"],
                                         plocal["f2_far_median_px"]) >= 0.10,
            "B tail_gt50 -10%": drop(c0["tail_gt50"], plocal["tail_gt50"]) >= 0.10,
            "D confident_wrong better_by_10px >= 20%": bool(
                complement.loc[complement.group == "confident_wrong",
                               "better_by_10px"].iloc[0] >= 0.20)
            if (complement.group == "confident_wrong").any() else False,
        },
        "values": {"f2_far_drop": drop(c0["f2_far_median_px"],
                                       plocal["f2_far_median_px"]),
                   "tail50_drop": drop(c0["tail_gt50"], plocal["tail_gt50"]),
                   "c0_pnp": c0["pose_success"], "plocal_pnp": plocal["pose_success"]},
    }
    interface["passed"] = any(interface["conditions"].values())
    (OUT / "proposal_interface_gate.json").write_text(
        json.dumps(MD.jsonable(interface), indent=1), encoding="utf-8")

    margin_row = [r for r in oracle_rows if r["arm"] == "oracle_margin"][0]
    oracle_gate = {
        "conditions": {
            "1 f2_far -20%": drop(c0["f2_far_median_px"],
                                  margin_row["f2_far_median_px"]) >= 0.20,
            "2 f2 signed bias -20%": drop(c0["f2_far_signed_bias_px"],
                                          margin_row["f2_far_signed_bias_px"]) >= 0.20,
            "3 tail_gt50 -20%": drop(c0["tail_gt50"], margin_row["tail_gt50"]) >= 0.20,
            "4 PnP >= 74": margin_row["pose_success"] >= 74,
            "5 reproj -10%": drop(c0["reproj_median_px"],
                                  margin_row["reproj_median_px"]) >= 0.10,
            "6 near <= +2%": margin_row["near_median_px"] <= c0["near_median_px"] * 1.02,
            "8 no new NaN": margin_row["nan_err"] <= c0["nan_err"],
        },
        "values": {k: margin_row[k] for k in
                   ("f2_far_median_px", "f2_far_signed_bias_px", "tail_gt50",
                    "pose_success", "reproj_median_px", "near_median_px",
                    "nan_err", "proposal_share")},
        "c0": {k: c0[k] for k in
               ("f2_far_median_px", "f2_far_signed_bias_px", "tail_gt50",
                "pose_success", "reproj_median_px", "near_median_px", "nan_err")},
    }
    oracle_gate["passed"] = all(oracle_gate["conditions"].values())
    (OUT / "proposal_oracle_gate.json").write_text(
        json.dumps(MD.jsonable(oracle_gate), indent=1), encoding="utf-8")

    log("\n[C3] proposal decoder interface gate")
    for name, value in interface["conditions"].items():
        log(f"    {'PASS' if value else 'FAIL'}  {name}")
    log(f"    -> {'GO' if interface['passed'] else 'FAIL'}")
    log("\n[D2] oracle router gate (margin 3px)")
    for name, value in oracle_gate["conditions"].items():
        log(f"    {'PASS' if value else 'FAIL'}  {name}")
    log(f"    -> {'PASS' if oracle_gate['passed'] else 'FAIL'}"
        f"  (learned router {'runs' if oracle_gate['passed'] else 'NOT RUN'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
