"""Stage 0C/0D — does the local translation-sensitivity surrogate track the real
PnP depth error on PAPER_EVAL 319?

Read-only.  No inference, no training.  Inputs are the frozen R0 prediction cache
and the geometry-resolved GT; both are hashed into the output.

The decisive question is NOT "does L_TR correlate with depth error" on its own --
a plain residual magnitude would too, because a frame with larger 2D error has
larger everything.  The question is whether the *sensitivity weighting* adds
information over plain residual magnitude, so every correlation here is reported
next to a plain-RMS control and a rank-partial correlation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pnp_jacobian import pnp_jacobian, project

REPO = Path(__file__).resolve().parents[3]
CLOSURE = REPO / "data/pallet/results/paper_pose_metric_closure_v1"
GT_PATH = CLOSURE / "GEOMETRY_RESOLVED_POSE_GT.json"
MANIFEST = CLOSURE / "AXIS_REVIEW_MANIFEST.json"
CONTRACT = CLOSURE / "POSE_EVAL_OBJECT_CONTRACT.json"
PRED = CLOSURE / "predictions/R0.json"
OUT_JSON = REPO / "data/pallet/results/pallet_translation_loss_v1/TRANSLATION_SURROGATE_AUDIT.json"
OUT_CSV = REPO / "data/pallet/results/pallet_translation_loss_v1/TRANSLATION_SURROGATE_PER_FRAME.csv"

RCOND = 1e-6            # [추정][미검증] pinv singular cutoff; swept below
CF_WIDTH, CF_DEPTH = "CF_WIDTH", "CF_DEPTH"


def cuboid(across, height, along):
    ha, hh, hb = across / 2.0, height / 2.0, along / 2.0
    return np.array([
        [-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
        [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb],
    ], dtype=np.float64)


def extents(long_axis, long_m, short_m, height_m):
    if long_axis == CF_WIDTH:
        return long_m, height_m, short_m
    if long_axis == CF_DEPTH:
        return short_m, height_m, long_m
    raise ValueError(long_axis)


def solve_pnp(model, points, camera, usable):
    ok, rvec, tvec = cv2.solvePnP(model[usable], points[usable], camera, None,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(model[usable], points[usable], camera, None,
                                      rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(-1)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partial_spearman(a, b, control):
    """Spearman of a and b after linearly removing `control` from both, in rank space."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    rc = np.argsort(np.argsort(control)).astype(float)
    def resid(r):
        A = np.stack([rc, np.ones_like(rc)], axis=1)
        coef, *_ = np.linalg.lstsq(A, r, rcond=None)
        return r - A @ coef
    ea, eb = resid(ra), resid(rb)
    return float(pearsonr(ea, eb)[0])


def main() -> int:
    gt = json.loads(GT_PATH.read_text())["frames"]
    manifest = {f["frame_id"]: f for f in json.loads(MANIFEST.read_text())["frames_list"]}
    contract = json.loads(CONTRACT.read_text())
    preds = json.loads(PRED.read_text())["frames"]

    rows = []
    jac_cache = []
    for frame_id, truth in gt.items():
        pred = preds.get(frame_id)
        if not pred or pred.get("status") != "OK":
            continue
        frame = manifest[frame_id]
        payload = json.loads((REPO / frame["annotation"]).read_text())
        raw = payload["camera_data"]["intrinsics"]
        K = np.array([[raw["fx"], 0.0, raw["cx"]],
                      [0.0, raw["fy"], raw["cy"]], [0.0, 0.0, 1.0]], np.float64)
        spec = contract[truth["object_type"]]["physical_dimensions_m"]
        across, height, along = extents(truth["physical_long_axis"],
                                        spec["long"], spec["short"], spec["height"])
        X = cuboid(across, height, along)
        R_gt = np.asarray(truth["R_gt_representative"], np.float64)
        t_gt = np.asarray(truth["t_gt"], np.float64)

        pts = np.asarray(pred["keypoints_xy"], np.float64)[:8]
        usable = np.isfinite(pts).all(axis=1)
        if usable.sum() < 6:
            continue

        u_gt = project(K, R_gt, t_gt, X)
        r = (pts - u_gt).reshape(-1)                    # 16-vector, GT-anchored residual
        J = pnp_jacobian(K, R_gt, t_gt, X)              # (16, 6)
        A = np.linalg.pinv(J, rcond=RCOND)              # (6, 16)
        A_t = A[3:6, :]                                 # translation rows
        s = np.sum(A_t ** 2, axis=0)                    # (16,) per-coordinate sensitivity
        s_depth = A_t[2, :] ** 2                        # depth-only sensitivity

        L_TR = float(np.sqrt(np.sum(s * r ** 2)))
        L_TR_depth = float(np.sqrt(np.sum(s_depth * r ** 2)))
        dxi = A @ r
        dt_lin = dxi[3:6]

        # actual: ORACLE axis supplied, so the axis selector cannot confound this
        sol = solve_pnp(X, pts, K, usable)
        if sol is None:
            continue
        R_pred, t_pred = sol
        dt_act = t_pred - t_gt

        jac_cache.append({"J": J, "r": r})
        rms = float(np.sqrt(np.mean(r ** 2)))
        corner_px = np.linalg.norm(pts - u_gt, axis=1)
        rows.append({
            "frame_id": frame_id,
            "object_type": truth["object_type"],
            "session_id": truth["session_id"],
            "elevation_deg": truth.get("elevation_deg"),
            "gt_depth_m": float(t_gt[2]),
            "corner_median_px": float(np.median(corner_px)),
            "corner_max_px": float(corner_px.max()),
            "residual_rms_px": rms,
            "L_TR": L_TR,
            "L_TR_depth_only": L_TR_depth,
            "lin_depth_m": float(abs(dt_lin[2])),
            "lin_total_m": float(np.linalg.norm(dt_lin)),
            "act_depth_m": float(abs(dt_act[2])),
            "act_total_m": float(np.linalg.norm(dt_act)),
            "act_lateral_m": float(np.linalg.norm(dt_act[[0, 1]])),
        })

    n = len(rows)
    def col(k):
        return np.array([r[k] for r in rows], float)

    def block(idx, label):
        if idx.sum() < 12:
            return {"label": label, "n": int(idx.sum()), "note": "n<12, not reported"}
        LTR, LTRD = col("L_TR")[idx], col("L_TR_depth_only")[idx]
        RMS = col("residual_rms_px")[idx]
        LIN = col("lin_depth_m")[idx]
        GTZ = col("gt_depth_m")[idx]
        AD, AT = col("act_depth_m")[idx], col("act_total_m")[idx]
        return {
            "label": label,
            "n": int(idx.sum()),
            "spearman_LTR_vs_actual_depth": float(spearmanr(LTR, AD).statistic),
            "spearman_LTR_vs_actual_total_t": float(spearmanr(LTR, AT).statistic),
            "spearman_LTRdepth_vs_actual_depth": float(spearmanr(LTRD, AD).statistic),
            "spearman_linearised_depth_vs_actual_depth": float(spearmanr(LIN, AD).statistic),
            "CONTROL_spearman_plainRMS_vs_actual_depth": float(spearmanr(RMS, AD).statistic),
            "CONTROL_spearman_plainRMS_vs_actual_total_t": float(spearmanr(RMS, AT).statistic),
            "PARTIAL_LTR_vs_depth_given_plainRMS": partial_spearman(LTR, AD, RMS),
            "PARTIAL_linearised_depth_vs_depth_given_plainRMS": partial_spearman(LIN, AD, RMS),
            "CONTROL_spearman_gtRange_vs_actual_depth": float(spearmanr(GTZ, AD).statistic),
            "CONTROL_spearman_LTR_vs_gtRange": float(spearmanr(LTR, GTZ).statistic),
            "PARTIAL_LTR_vs_depth_given_gtRange": partial_spearman(LTR, AD, GTZ),
            "PARTIAL_linearised_depth_vs_depth_given_gtRange": partial_spearman(LIN, AD, GTZ),
            "pearson_LTR_vs_actual_depth": float(pearsonr(LTR, AD)[0]),
            "median_actual_depth_cm": float(np.median(AD) * 100),
            "median_linearised_depth_cm": float(np.median(LIN) * 100),
        }

    cm = col("corner_median_px")
    blocks = [
        block(np.ones(n, bool), "ALL"),
        block(cm < 5, "corner_median_lt_5px"),
        block((cm >= 5) & (cm < 10), "corner_median_5_to_10px"),
        block(cm >= 10, "corner_median_ge_10px"),
        block(np.array([r["object_type"].startswith("plastic") for r in rows]), "plastic"),
        block(np.array([r["object_type"].startswith("wood") for r in rows]), "wood"),
    ]

    # rcond sensitivity — the cutoff is an unvalidated constant, so show it matters or not
    rcond_sweep = {}
    for rc in (1e-4, 1e-6, 1e-8):
        vals = []
        for row, frame_id in zip(rows, [r["frame_id"] for r in rows]):
            vals.append(row["L_TR"])
        rcond_sweep[f"{rc:g}"] = "recomputed below"

    RCOND_SWEEP = {}
    for rc in (1e-3, 1e-4, 1e-6, 1e-8):
        ltr = []
        for cache in jac_cache:
            A = np.linalg.pinv(cache["J"], rcond=rc)
            s_ = np.sum(A[3:6, :] ** 2, axis=0)
            ltr.append(float(np.sqrt(np.sum(s_ * cache["r"] ** 2))))
        ltr = np.array(ltr)
        RCOND_SWEEP[f"{rc:g}"] = {
            "spearman_LTR_vs_actual_depth": float(spearmanr(ltr, col("act_depth_m")).statistic),
            "L_TR_median": float(np.median(ltr)),
        }

    result = {
        "schema_version": "pallet_translation_loss_v1_surrogate_audit_v1",
        "read_only": True,
        "new_inference": 0,
        "new_training": 0,
        "population": "PAPER_EVAL 319, geometry-resolved GT, ORACLE axis supplied",
        "frames_used": n,
        "inputs_sha256": {
            "GEOMETRY_RESOLVED_POSE_GT.json": sha(GT_PATH),
            "predictions/R0.json": sha(PRED),
            "POSE_EVAL_OBJECT_CONTRACT.json": sha(CONTRACT),
        },
        "pinv_rcond": RCOND,
        "residual_definition": "pred_2d - project(K, R_gt, t_gt, cuboid); 8 corners, centroid excluded",
        "actual_definition": "SQPnP + RefineLM on the same 8 predicted corners with the ORACLE model, minus t_gt",
        "blocks": blocks,
        "rcond_sensitivity": RCOND_SWEEP,
        "median_gt_depth_m": float(np.median(col("gt_depth_m"))),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    keys = list(rows[0].keys())
    with OUT_CSV.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
