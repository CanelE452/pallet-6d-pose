"""Gate 0 — is the corner bias repeatable given viewpoint and corner role?

No training happens here.  Every decoded corner on the mechanism set is paired
with the viewpoint derived from its GT pose, a bias model is fitted on seven
capture sessions and applied to the eighth, and the corrected corners go back
through the canonical PnP with the centroid included.  If a leave-one-session-out
correction cannot move the numbers, a view-conditioned architecture has no basis
and the screen stops here.

    python scripts/stage0/paper_s2/paper_s2_vcr_bias_atlas.py --all
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "data/pallet/results/paper_s2_vcr_dope_screen"
STAGE0 = ROOT / "scripts/stage0"
DOPE = ROOT / "Deep_Object_Pose"
for extra in (STAGE0, DOPE / "common", DOPE / "train"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EP57 = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MD = _load("MD", STAGE0 / "paper_s2_mechanism_diagnostic.py")
FZ = MD.FZ
APNP = MD.APNP
import view_corner_bias as VCB  # noqa: E402

NEAR, FAR = MD.NEAR_KP, MD.FAR_KP
TOP, BOTTOM = MD.TOP_KP, MD.BOTTOM_KP
LEFT, RIGHT = MD.DEPTH_LEFT_KP, MD.DEPTH_RIGHT_KP
ROLES = VCB.corner_roles(NEAR, FAR, TOP, BOTTOM, LEFT, RIGHT)
ARMS = ("B0", "B1", "B2", "B3")


def log(message: str) -> None:
    print(message, flush=True)


# ============================================================================
# rows
# ============================================================================
def build_rows() -> tuple[pd.DataFrame, dict, dict, dict]:
    manifest = json.loads(MD.MANIFEST_PATH.read_text("utf-8"))
    primary = [f for f in manifest["frames"] if f["population"] == "primary"]
    audit = FZ.InputAudit()
    tensors = MD.load_cached_tensors()
    classes = pd.read_csv(MD.OUT_DIR / "failure_class_frames.csv").set_index("frame_id")

    geometries, decoded, stages = {}, {}, {}
    rows = []
    for spec in primary:
        uid = spec["frame_id"]
        geometry = MD.FrameGeometry(spec, audit)
        stack = tensors[f"{uid}|belief_stages"]
        scale_xy = (spec["image_width"] / 50.0, spec["image_height"] / 50.0)
        geometries[uid] = geometry
        stages[uid] = stack
        decoded[uid] = {s: MD.decode_all(stack[s], *scale_xy, geometry.gt_points)
                        for s in (3, 5)}

        pose = geometry.gt_pose
        psi, epsilon = VCB.view_angles(pose["R"], np.asarray(pose["t"]).reshape(3))
        projected = np.asarray(
            [p for p in geometry.gt_points[:8] if p is not None], float)
        scale = VCB.object_scale(projected,
                                 (spec["image_width"], spec["image_height"]))
        phi_full = VCB.view_feature(psi, epsilon, scale, full=True)
        phi_base = VCB.view_feature(psi, epsilon, scale, full=False)

        points6 = decoded[uid][5]["D0"]
        points4 = decoded[uid][3]["D0"]
        for corner in range(8):
            gt = geometry.gt_points[corner]
            point = points6[corner]
            if gt is None or point is None:
                continue
            early = points4[corner]
            entry = {
                "frame_id": uid, "session_id": spec["session_id"],
                "domain": spec["domain"], "corner": corner,
                "role_depth": ROLES[corner]["depth"],
                "role_height": ROLES[corner]["height"],
                "role_side": ROLES[corner]["side"],
                "failure_class": classes.loc[uid, "failure_class"],
                "pred_x": float(point[0]), "pred_y": float(point[1]),
                "gt_x": float(gt[0]), "gt_y": float(gt[1]),
                "dx": float(point[0] - gt[0]), "dy": float(point[1] - gt[1]),
                "err": float(np.hypot(point[0] - gt[0], point[1] - gt[1])),
                "psi": psi, "epsilon": epsilon, "scale": scale,
                "stage6_peak": float(stack[5, corner].max()),
                "stage4_peak": float(stack[3, corner].max()),
                "stage4_err": (float(np.hypot(early[0] - gt[0], early[1] - gt[1]))
                               if early is not None else np.nan),
            }
            for name, value in zip(VCB.FEATURE_NAMES_B3, phi_full):
                entry[f"phi_{name}"] = float(value)
            for name, value in zip(VCB.FEATURE_NAMES_B2, phi_base):
                entry[f"phib_{name}"] = float(value)
            rows.append(entry)
    if audit.prohibited_attempts:
        raise RuntimeError(f"final-test access: {audit.prohibited_attempts}")
    return pd.DataFrame(rows), geometries, decoded, {
        "manifest": manifest, "primary": primary}


# ============================================================================
# leave-one-session-out correction
# ============================================================================
def run_loso(rows: pd.DataFrame) -> pd.DataFrame:
    full = [f"phi_{n}" for n in VCB.FEATURE_NAMES_B3]
    base = [f"phib_{n}" for n in VCB.FEATURE_NAMES_B2]
    table = rows.copy()
    for arm in ARMS:
        table[f"{arm}_dx"] = np.nan
        table[f"{arm}_dy"] = np.nan
    for session in sorted(rows.session_id.unique()):
        train = rows[rows.session_id != session]
        test = rows[rows.session_id == session]
        index = test.index
        models = {
            "B0": VCB.BiasModel("none"),
            "B1": VCB.BiasModel("constant").fit(
                train.corner.to_numpy(), train[full].to_numpy(),
                train[["dx", "dy"]].to_numpy()),
            "B2": VCB.BiasModel("linear").fit(
                train.corner.to_numpy(), train[base].to_numpy(),
                train[["dx", "dy"]].to_numpy()),
            "B3": VCB.BiasModel("linear").fit(
                train.corner.to_numpy(), train[full].to_numpy(),
                train[["dx", "dy"]].to_numpy()),
        }
        for arm, model in models.items():
            columns = base if arm == "B2" else full
            predicted = model.predict(test.corner.to_numpy(),
                                      test[columns].to_numpy())
            table.loc[index, f"{arm}_dx"] = predicted[:, 0]
            table.loc[index, f"{arm}_dy"] = predicted[:, 1]
    for arm in ARMS:
        table[f"{arm}_x"] = table.pred_x - table[f"{arm}_dx"]
        table[f"{arm}_y"] = table.pred_y - table[f"{arm}_dy"]
        table[f"{arm}_err"] = np.hypot(table[f"{arm}_x"] - table.gt_x,
                                       table[f"{arm}_y"] - table.gt_y)
        table[f"{arm}_rx"] = table[f"{arm}_x"] - table.gt_x
        table[f"{arm}_ry"] = table[f"{arm}_y"] - table.gt_y
    return table


def pose_for_arm(table: pd.DataFrame, arm: str, geometries, decoded,
                 primary) -> dict[str, Any]:
    """Corrected 8 corners plus the original predicted centroid."""
    success, reprojs, yaws = 0, [], []
    per_frame = {}
    for spec in primary:
        uid = spec["frame_id"]
        geometry = geometries[uid]
        points: list[Optional[list[float]]] = list(decoded[uid][5]["D0"])
        block = table[table.frame_id == uid]
        for _, row in block.iterrows():
            points[int(row["corner"])] = [float(row[f"{arm}_x"]),
                                          float(row[f"{arm}_y"])]
        pose = geometry.solve(points)          # centroid kept as predicted
        success += int(pose is not None)
        per_frame[uid] = pose is not None
        if pose is not None:
            metrics = geometry.metrics(pose)
            if metrics["reproj_fixed_gt_px"] is not None:
                reprojs.append(metrics["reproj_fixed_gt_px"])
            if metrics["yaw_err_deg"] is not None:
                yaws.append(metrics["yaw_err_deg"])
    return {"pose_success": success,
            "reproj_median_px": float(np.median(reprojs)) if reprojs else np.nan,
            "yaw_median_deg": float(np.median(yaws)) if yaws else np.nan,
            "per_frame": per_frame}


def summarise(table: pd.DataFrame, arm: str) -> dict[str, Any]:
    f2 = table[table.failure_class == "F2_CONFIDENT_WRONG"]
    f2_far = f2[f2.role_depth == "far"]
    far = table[table.role_depth == "far"]
    near = table[table.role_depth == "near"]
    errors = table[f"{arm}_err"]
    return {
        "arm": arm,
        "median_px": float(errors.median()),
        "near_median_px": float(near[f"{arm}_err"].median()),
        "far_median_px": float(far[f"{arm}_err"].median()),
        "f2_far_median_px": float(f2_far[f"{arm}_err"].median()),
        "f2_far_signed_bias_px": float(np.hypot(f2_far[f"{arm}_rx"].mean(),
                                                f2_far[f"{arm}_ry"].mean())),
        "tail_gt20": int((errors > 20).sum()),
        "tail_gt50": int((errors > 50).sum()),
        "tail_gt100": int((errors > 100).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    log("[A] identity and baseline")
    if hashlib.sha256(EP57.read_bytes()).hexdigest() != EP57_SHA:
        raise SystemExit("BLOCKED: checkpoint SHA mismatch")
    rows, geometries, decoded, context = build_rows()
    primary = context["primary"]
    gate = MD.baseline_gate(dict(context["manifest"], frames=primary),
                            geometries,
                            {uid: decoded[uid][5] for uid in decoded})
    log(f"    strict {gate['strict_n']} gt2d {gate['gt2d_pose_success']} "
        f"pred {gate['pred_pose_success']} yaw {gate['yaw_median_deg']:.6f} "
        f"reproj {gate['fixed_gt_reproj_median_px']:.6f} passed={gate['passed']}")
    if not gate["passed"]:
        raise SystemExit(f"BLOCKED: {gate['problems']}")
    far_rows = rows[rows.role_depth == "far"]
    log(f"    stage trajectory far: stage4 {far_rows.stage4_err.median():.2f}px "
        f"-> stage6 {far_rows.err.median():.2f}px"
        f"   peak {far_rows.stage4_peak.median():.3f} -> "
        f"{far_rows.stage6_peak.median():.3f}")
    log(f"    corner rows {len(rows)}  sessions {rows.session_id.nunique()}")

    log("[C] leave-one-session-out bias correction")
    table = run_loso(rows)
    table.to_csv(OUT / "vcr_bias_rows.csv", index=False)

    summaries, poses = [], {}
    for arm in ARMS:
        summary = summarise(table, arm)
        pose = pose_for_arm(table, arm, geometries, decoded, primary)
        summary.update({k: v for k, v in pose.items() if k != "per_frame"})
        poses[arm] = pose
        summaries.append(summary)
    metrics = pd.DataFrame(summaries)
    metrics.to_csv(OUT / "vcr_loso_metrics.csv", index=False)
    log("\n" + metrics.to_string(index=False))

    b0 = summaries[0]

    def drop(before, after):
        return float(1.0 - after / before) if before else 0.0

    def paired(arm):
        good = int((table[f"{arm}_err"] < table.B0_err).sum())
        bad = int((table[f"{arm}_err"] > table.B0_err).sum())
        return good, bad

    conditions, best = {}, None
    for arm in ("B2", "B3"):
        s = [x for x in summaries if x["arm"] == arm][0]
        good, bad = paired(arm)
        checks = [
            ("1 F2 signed bias -25%",
             drop(b0["f2_far_signed_bias_px"], s["f2_far_signed_bias_px"]) >= 0.25,
             drop(b0["f2_far_signed_bias_px"], s["f2_far_signed_bias_px"])),
            ("2 F2 far median -15%",
             drop(b0["f2_far_median_px"], s["f2_far_median_px"]) >= 0.15,
             drop(b0["f2_far_median_px"], s["f2_far_median_px"])),
            ("3 >50px tail -15%", drop(b0["tail_gt50"], s["tail_gt50"]) >= 0.15,
             drop(b0["tail_gt50"], s["tail_gt50"])),
            ("4 paired improved > worsened", good > bad, good - bad),
            ("5 near <= +5%", s["near_median_px"] <= b0["near_median_px"] * 1.05,
             s["near_median_px"] / b0["near_median_px"] - 1.0),
            ("6 PnP >= 72 or reproj -8%",
             s["pose_success"] >= 72 or (
                 s["pose_success"] >= b0["pose_success"]
                 and drop(b0["reproj_median_px"], s["reproj_median_px"]) >= 0.08),
             s["pose_success"]),
            ("7 no new >100px", s["tail_gt100"] <= b0["tail_gt100"],
             s["tail_gt100"] - b0["tail_gt100"]),
        ]
        conditions[arm] = {"checks": [{"name": n, "passed": bool(p),
                                       "value": float(v)} for n, p, v in checks],
                           "passed": all(p for _, p, _ in checks),
                           "paired_improved": good, "paired_worsened": bad}
        if conditions[arm]["passed"]:
            best = arm

    b1 = [x for x in summaries if x["arm"] == "B1"][0]
    necessity = {}
    for arm in ("B2", "B3"):
        s = [x for x in summaries if x["arm"] == arm][0]
        rescue = sum(1 for uid, ok in poses[arm]["per_frame"].items()
                     if ok and not poses["B1"]["per_frame"][uid])
        checks = {
            "signed bias -10% vs B1":
                drop(b1["f2_far_signed_bias_px"], s["f2_far_signed_bias_px"]) >= 0.10,
            "F2 far -7.5% vs B1":
                drop(b1["f2_far_median_px"], s["f2_far_median_px"]) >= 0.075,
            ">50px tail -5% vs B1":
                drop(b1["tail_gt50"], s["tail_gt50"]) >= 0.05,
            "PnP rescue >= 2 vs B1": rescue >= 2,
        }
        necessity[arm] = {"checks": checks, "passed": any(checks.values()),
                          "rescue_vs_B1": rescue}

    decision = {"summaries": summaries, "conditions": conditions,
                "view_necessity": necessity,
                "passed": bool(best is not None
                               and necessity[best]["passed"]),
                "passing_arm": best}
    (OUT / "vcr_gate0.json").write_text(json.dumps(MD.jsonable(decision), indent=1),
                                        encoding="utf-8")

    log("\n[C6] Gate 0")
    for arm in ("B2", "B3"):
        log(f"  {arm}:")
        for check in conditions[arm]["checks"]:
            log(f"    {'PASS' if check['passed'] else 'FAIL'}  "
                f"{check['name']:<30} {check['value']:>9.4f}")
        log(f"    view necessity vs B1: "
            f"{ {k: bool(v) for k, v in necessity[arm]['checks'].items()} }")
    log(f"\n  -> Gate 0 {'PASS' if decision['passed'] else 'FAIL'}"
        f"  (passing arm {best})")

    provenance = {
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip(),
        "checkpoint_sha256": EP57_SHA, "training_steps": 0,
        "baseline_gate": MD.jsonable(gate),
        "n_corner_rows": int(len(rows)), "n_sessions": int(rows.session_id.nunique()),
        "ridge_lambda": VCB.RIDGE_LAMBDA,
        "feature_basis_B2": list(VCB.FEATURE_NAMES_B2),
        "feature_basis_B3": list(VCB.FEATURE_NAMES_B3),
        "protocol": "leave-one-session-out, 8 folds, centroid kept predicted",
    }
    (OUT / "vcr_gate0_provenance.json").write_text(
        json.dumps(MD.jsonable(provenance), indent=1), encoding="utf-8")
    log(f"[done] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
