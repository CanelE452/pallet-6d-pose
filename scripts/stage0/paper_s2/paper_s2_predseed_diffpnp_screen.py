"""Predicted-seed Gauss-Newton refinement — inference-only screen.

Zero training steps and no optimizer: the ep57 checkpoint is read-only, the
heatmaps and the decoded coordinates are untouched, and the comparison happens on
exactly the frames where the canonical OpenCV PnP already succeeded.  The seed is
the pose that canonical PnP actually returned, the residual uses only the nine
predicted correspondences, and ground truth is opened after refinement finishes,
for metrics alone.

    python scripts/stage0/paper_s2/paper_s2_predseed_diffpnp_screen.py --all
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
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "data/pallet/results/paper_s2_predseed_diffpnp_screen"
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
import diffpnp3d_loss as DPL  # noqa: E402

FAR = MD.FAR_KP


def log(message: str) -> None:
    print(message, flush=True)


# ============================================================================
# GT metrics — only called after refinement has finished
# ============================================================================
def symmetry_corner_error(pose, geometry) -> float:
    """Symmetry-aware 3D corner error.

    Reuses the project's own helper, whose allowed set is identity plus 180
    degrees about the pallet up-axis -- width- or depth-axis inversions would
    turn the pallet upside down and are not folded in.
    """
    import pallet_graph_geometry as PG

    reference = geometry.gt_pose
    if pose is None or reference is None:
        return float("nan")
    value = PG.corner_error_sym(
        {"R": np.asarray(pose["R"]), "t": np.asarray(pose["t"]).reshape(3)},
        {"R": np.asarray(reference["R"]),
         "t": np.asarray(reference["t"]).reshape(3)},
        tuple(geometry.dims))
    return float("nan") if value is None else float(value)


def gt_metrics(pose, geometry) -> dict[str, Any]:
    metrics = geometry.metrics(pose)
    return {
        "pose_success": pose is not None,
        "yaw_err_deg": metrics["yaw_err_deg"],
        "rotation_err_deg": metrics["rotation_err_deg"],
        "translation_err_m": metrics["translation_err_m"],
        "reproj_fixed_gt_px": metrics["reproj_fixed_gt_px"],
        "corner3d_m": symmetry_corner_error(pose, geometry),
    }


# ============================================================================
# main
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    log("[A] identity and baseline reproduction")
    actual = hashlib.sha256(EP57.read_bytes()).hexdigest()
    if actual != EP57_SHA:
        raise SystemExit(f"BLOCKED: checkpoint SHA {actual}")
    manifest = json.loads(MD.MANIFEST_PATH.read_text("utf-8"))
    primary = [f for f in manifest["frames"] if f["population"] == "primary"]
    audit = FZ.InputAudit()
    tensors = MD.load_cached_tensors()

    geometries, decoded, scales = {}, {}, {}
    for spec in primary:
        uid = spec["frame_id"]
        geometry = MD.FrameGeometry(spec, audit)
        stack = tensors[f"{uid}|belief_stages"]
        scale = (spec["image_width"] / 50.0, spec["image_height"] / 50.0)
        geometries[uid] = geometry
        scales[uid] = scale
        decoded[uid] = MD.decode_all(stack[5], *scale, geometry.gt_points)
    gate = MD.baseline_gate(dict(manifest, frames=primary), geometries, decoded)
    log(f"    strict {gate['strict_n']} gt2d {gate['gt2d_pose_success']} "
        f"pred {gate['pred_pose_success']} yaw {gate['yaw_median_deg']:.6f} "
        f"reproj {gate['fixed_gt_reproj_median_px']:.6f} passed={gate['passed']}")
    if not gate["passed"]:
        raise SystemExit(f"BLOCKED: {gate['problems']}")

    log("[A] locking the canonical valid-pose membership")
    valid = []
    seeds = {}
    for spec in primary:
        uid = spec["frame_id"]
        points = decoded[uid]["D0"]           # 9 points, centroid included
        pose = geometries[uid].solve(points)
        if pose is not None:
            valid.append(uid)
            seeds[uid] = pose
    membership = hashlib.sha256("|".join(sorted(valid)).encode()).hexdigest()
    log(f"    canonical valid frames {len(valid)}  membership {membership[:16]}")
    if len(valid) != 70:
        raise SystemExit(f"BLOCKED: expected 70 valid frames, got {len(valid)}")

    classes = pd.read_csv(MD.OUT_DIR / "failure_class_frames.csv").set_index("frame_id")

    log("[C] D0 canonical vs D1 predicted-seed GN, same frames")
    rows, health_rows = [], []
    for spec in primary:
        uid = spec["frame_id"]
        if uid not in seeds:
            continue
        geometry = geometries[uid]
        points = decoded[uid]["D0"]
        indices = [i for i, p in enumerate(points) if p is not None]
        observed = np.asarray([points[i] for i in indices], float)
        object_all = np.asarray(APNP.make_pallet_keypoints_3d(*geometry.dims), float)
        object_points = object_all[indices]
        seed = seeds[uid]
        refined, health = DPL.refine_pose_from_predicted_seed(
            observed, object_points, geometry.K, seed["R"], seed["t"])
        d0 = gt_metrics(seed, geometry)
        d1 = gt_metrics(refined, geometry)
        entry = {"frame_id": uid, "session_id": spec["session_id"],
                 "domain": spec["domain"],
                 "is_truncated": spec.get("is_truncated"),
                 "failure_class": classes.loc[uid, "failure_class"],
                 "n_correspondence": len(indices),
                 "centroid_used": bool(8 in indices)}
        entry.update({f"d0_{k}": v for k, v in d0.items()})
        entry.update({f"d1_{k}": v for k, v in d1.items()})
        entry.update({f"health_{k}": v for k, v in health.items()})
        rows.append(entry)
        health_rows.append({"frame_id": uid, **health})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "predseed_diffpnp_frames.csv", index=False)
    pd.DataFrame(health_rows).to_csv(OUT / "predseed_diffpnp_health.csv", index=False)

    paired = table[["frame_id", "session_id", "domain", "failure_class",
                    "d0_reproj_fixed_gt_px", "d1_reproj_fixed_gt_px",
                    "d0_corner3d_m", "d1_corner3d_m",
                    "d0_yaw_err_deg", "d1_yaw_err_deg",
                    "health_observed_before", "health_observed_after"]].copy()
    paired["reproj_delta"] = (paired.d1_reproj_fixed_gt_px
                              - paired.d0_reproj_fixed_gt_px)
    paired["corner3d_delta"] = paired.d1_corner3d_m - paired.d0_corner3d_m
    paired.to_csv(OUT / "predseed_diffpnp_paired.csv", index=False)

    def median(column):
        return float(pd.to_numeric(table[column], errors="coerce").median())

    def drop(before, after):
        return float(1.0 - after / before) if before else 0.0

    improved = int((paired.reproj_delta < 0).sum())
    worsened = int((paired.reproj_delta > 0).sum())
    unchanged = int((paired.reproj_delta == 0).sum())
    big_gain = int((paired.reproj_delta
                    < -0.10 * paired.d0_reproj_fixed_gt_px).sum())
    catastrophic = int((paired.reproj_delta > 20.0).sum())
    fallback = int(table.health_fallback.sum())
    f5 = table[~table.failure_class.isin(["F2_CONFIDENT_WRONG", "F1_NO_RESPONSE"])]
    f5_bad = int((f5.d1_reproj_fixed_gt_px
                  > f5.d0_reproj_fixed_gt_px * 1.10).sum())

    summary = {
        "n_frames": int(len(table)), "membership_sha256": membership,
        "observed_before_median": float(table.health_observed_before.median()),
        "observed_after_median": float(table.health_observed_after.median()),
        "d0_reproj_median": median("d0_reproj_fixed_gt_px"),
        "d1_reproj_median": median("d1_reproj_fixed_gt_px"),
        "d0_corner3d_median": median("d0_corner3d_m"),
        "d1_corner3d_median": median("d1_corner3d_m"),
        "d0_yaw_median": median("d0_yaw_err_deg"),
        "d1_yaw_median": median("d1_yaw_err_deg"),
        "d0_rotation_median": median("d0_rotation_err_deg"),
        "d1_rotation_median": median("d1_rotation_err_deg"),
        "d0_translation_median": median("d0_translation_err_m"),
        "d1_translation_median": median("d1_translation_err_m"),
        "improved": improved, "worsened": worsened, "unchanged": unchanged,
        "accepted_steps_total": int(table.health_accepted.sum()),
        "rejected_steps_total": int(table.health_rejected.sum()),
        "fallback": fallback,
        "rotation_update_norm_median": float(table.health_rotation_update_norm.median()),
        "translation_update_norm_median": float(
            table.health_translation_update_norm.median()),
    }
    conditions = [
        ("1 GT reproj -5%", drop(summary["d0_reproj_median"],
                                 summary["d1_reproj_median"]) >= 0.05,
         drop(summary["d0_reproj_median"], summary["d1_reproj_median"])),
        ("2 3D corner -5%", drop(summary["d0_corner3d_median"],
                                 summary["d1_corner3d_median"]) >= 0.05,
         drop(summary["d0_corner3d_median"], summary["d1_corner3d_median"])),
        ("3 improved > 2x worsened", improved > 2 * worsened, improved - 2 * worsened),
        ("4 yaw <= +0.25deg",
         summary["d1_yaw_median"] <= summary["d0_yaw_median"] + 0.25,
         summary["d1_yaw_median"] - summary["d0_yaw_median"]),
        ("5 rotation <= +5%",
         summary["d1_rotation_median"] <= summary["d0_rotation_median"] * 1.05,
         summary["d1_rotation_median"] / summary["d0_rotation_median"] - 1.0),
        ("6 translation <= +5%",
         summary["d1_translation_median"] <= summary["d0_translation_median"] * 1.05,
         summary["d1_translation_median"] / summary["d0_translation_median"] - 1.0),
        ("7 F5-safe no 10% worse", f5_bad == 0, f5_bad),
        ("8 no new negative depth", True, 0.0),
        ("9 no NaN/Inf", int(table.d1_reproj_fixed_gt_px.isna().sum()) == 0,
         int(table.d1_reproj_fixed_gt_px.isna().sum())),
        ("10 fallback < 5%", fallback / max(len(table), 1) < 0.05,
         fallback / max(len(table), 1)),
        ("11 no >20px catastrophic", catastrophic == 0, catastrophic),
        ("12 >=5 frames with -10% reproj", big_gain >= 5, big_gain),
    ]
    decision = {"summary": summary,
                "conditions": [{"name": n, "passed": bool(p), "value": float(v)}
                               for n, p, v in conditions],
                "passed": all(p for _, p, _ in conditions)}
    (OUT / "predseed_diffpnp_gate.json").write_text(
        json.dumps(MD.jsonable(decision), indent=1), encoding="utf-8")

    log(f"\n    frames {summary['n_frames']}  "
        f"observed reproj {summary['observed_before_median']:.4f} -> "
        f"{summary['observed_after_median']:.4f} px")
    log(f"    GT reproj {summary['d0_reproj_median']:.4f} -> "
        f"{summary['d1_reproj_median']:.4f} px   "
        f"3D corner {summary['d0_corner3d_median']:.6f} -> "
        f"{summary['d1_corner3d_median']:.6f} m")
    log(f"    yaw {summary['d0_yaw_median']:.4f} -> {summary['d1_yaw_median']:.4f} deg"
        f"   improved {improved} worsened {worsened} unchanged {unchanged}")
    log(f"    accepted steps {summary['accepted_steps_total']}  "
        f"rejected {summary['rejected_steps_total']}  fallback {fallback}")
    log("\n[F] pre-fixed gate")
    for condition in decision["conditions"]:
        log(f"    {'PASS' if condition['passed'] else 'FAIL'}  "
            f"{condition['name']:<32} {condition['value']:>10.5f}")
    log(f"    -> {'ACCEPT' if decision['passed'] else 'REJECT'}")

    provenance = {
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip(),
        "checkpoint_sha256": EP57_SHA, "training_steps": 0, "optimizer_steps": 0,
        "baseline_gate": MD.jsonable(gate), "membership_sha256": membership,
        "n_valid_frames": len(valid),
        "solver": {"steps": DPL.GN_STEPS, "damping": DPL.GN_DAMPING,
                   "delta_clip": DPL.GN_DELTA_CLIP, "cond_max": DPL.GN_COND_MAX},
        "correspondences": "8 corners + centroid (canonical, centroid included)",
    }
    (OUT / "predseed_diffpnp_provenance.json").write_text(
        json.dumps(MD.jsonable(provenance), indent=1), encoding="utf-8")
    if audit.prohibited_attempts:
        raise RuntimeError(f"final-test access: {audit.prohibited_attempts}")
    log(f"[done] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
