"""REAL_DEV F0 vs F3 -- the final inference path, measured on real frames.

The earlier real number used the corner decoder and Point PnP only, which is F0.
It threw the line branch away, so it was not the final pose route.  This runs the
canonical F3 from `mh_fusion.solve_arms` -- no new solver is written here.

Two decisions that could quietly change what is measured, made explicit:

1. corner input.  `solve_arms` needs all eight corners, so the solver reads
   `MS._decode_peaks` (argmax, no threshold), exactly as the synthetic F3
   validation did.  The thresholded detector (`extract_keypoints_from_belief`
   at 0.3) is reported separately as the detection population, because that is
   what a deployed system would gate on.  Mixing the two would let detection
   failures masquerade as pose error.

2. line support.  In the synthetic evaluation `support` came from GT corners.
   On real frames that would be an oracle, so the main arm derives support from
   the PREDICTED corners through the same `visible_segments`.  The GT-derived
   variant is still computed and reported as a parity reference, never as the
   deployable number.

SEALED: the four FINAL_TEST sessions are not opened.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0", "scripts/stage0/paper_s2", "scripts/stage0/multihead",
            "scripts/stage0/line", "scripts/stage0/real_eval", "challenge",
            "scripts/annotate"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import cv2                                        # noqa: E402
import paper_s2_real_eval as PRE                  # noqa: E402
import data_paths as DP                           # noqa: E402
import mh_data as MD                              # noqa: E402
import mh_screen as MS                            # noqa: E402
import mh_splitlate as SL                         # noqa: E402
import mh_fusion as FU                            # noqa: E402
import mh_cigm as CG                              # noqa: E402
import line_feature_capacity_v2 as V2             # noqa: E402
import re_metrics as RM                           # noqa: E402
import annotate_pnp as APNP                       # noqa: E402
from mh_arms import DH                            # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/paper_s2_multihead/final_train")
CKPT = os.path.join(ROOT, "weights/paper_s2/paper_s2_multihead")
STEP, THRESH, N_DET_MIN = 25000, 0.3, 6
OPEN_SETS = [k for k in DP.EVAL_CANONICAL if k not in DP.FINAL_TEST]
BOOTSTRAP, BOOT_SEED = 10000, 20260907


def load(seed):
    path = os.path.join(CKPT, f"screen_A1_CORNER_LINE_FINAL40K_seed{seed}",
                        f"step_{STEP:05d}.pth")
    state = torch.load(path, map_location=MD.DEV, weights_only=False)
    model = SL.SplitLate(state["arm"])
    model.load_state_dict(state["model"])
    model.to(MD.DEV).eval()
    return model, os.path.relpath(path, ROOT)


def frames(set_key):
    folder = os.path.join(ROOT, DP.EVAL_CANONICAL[set_key])
    out = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        jp = os.path.join(folder, name)
        payload = json.load(open(jp))
        objects = payload.get("objects") or []
        if not objects or objects[0].get("split") != "eval":
            continue
        for ext in (".png", ".jpg", ".jpeg"):
            ip = jp[:-5] + ext
            if os.path.exists(ip):
                out.append((jp, ip, payload))
                break
    return out


def support_from_grid(grid9, width, height):
    """`visible_segments(...)['hit']` on the same 12 structural edges."""
    grid = np.asarray(grid9, float)[None, :, :]
    _, _, p0, p1, length = V2.gt_lines(grid, CG.EDGES)
    return V2.visible_segments(p0, p1, length)["hit"][0]


def pixels_to_grid(pixels, width, height, grid=50):
    return np.stack([np.asarray(pixels)[:, 0] * grid / width,
                     np.asarray(pixels)[:, 1] * grid / height], 1)


def run_frame(model, features, jp, ip, label, weight):
    image = cv2.imread(ip)
    if image is None:
        return None
    height, width = image.shape[:2]
    obj = label["objects"][0]
    dims = obj["dimensions_m"]
    extents = (dims["width"], dims["height"], dims["depth"])
    model_pts = APNP.make_pallet_keypoints_3d_diagram(
        width=dims["width"], depth=dims["depth"], height=dims["height"])[:8]
    K = CG.intrinsics(label)
    R_gt, t_gt = CG.gt_pose(label)
    gt8 = np.asarray(obj["projected_cuboid"], float)[:8]

    with torch.no_grad():
        out = model(PRE.preprocess_squash(image).to(MD.DEV), features)
        beliefs = out["beliefs"]
        theta_hat, rho_hat = DH.decode(out["line_scores"], *DH.lattice())
        theta_can, rho_can = DH.canonical_from_centred(theta_hat, rho_hat)
    belief = beliefs[-1][0].detach().cpu().numpy()

    peaks = MS._decode_peaks(beliefs[-1][:, :9])[0]          # (9,2) grid
    corner_grid = peaks[:8]
    peak_values = np.sort(np.max(belief[:8].reshape(8, -1), axis=1))[::-1]
    score_4kp = float(peak_values[3])

    thresholded = extract_keypoints_from_belief(belief, THRESH)
    n_det = int(sum(1 for k in thresholded[:8] if k[0] >= 0))

    gt_grid9 = pixels_to_grid(
        np.vstack([gt8, np.asarray(obj["projected_cuboid_centroid"], float)]),
        width, height)
    data = {
        "resolution": np.array([[width, height]]),
        "model": np.array([model_pts]),
        "K": np.array([K]),
        "pred_corner": np.array([peaks]),
        "pred_theta": theta_can.detach().cpu().numpy(),
        "pred_rho": rho_can.detach().cpu().numpy(),
    }
    row = {"fid": os.path.splitext(os.path.basename(jp))[0],
           "n_det": n_det, "det": int(n_det >= N_DET_MIN),
           "score_4kp": score_4kp}

    for tag, support in (("PRED", support_from_grid(peaks, width, height)),
                         ("GTSUP", support_from_grid(gt_grid9, width, height))):
        data["support"] = np.array([support])
        arms, corner_px, _, _ = FU.solve_arms(data, 0, weight)
        for arm in ("F0", "F3"):
            pose = arms.get(arm)
            key = arm if tag == "PRED" else f"{arm}_{tag}"
            if pose is None:
                row.update({f"{key}_ok": 0, f"{key}_R": np.nan,
                            f"{key}_t": np.nan, f"{key}_add": np.nan,
                            f"{key}_adds": np.nan, f"{key}_iou": np.nan,
                            f"{key}_5cm5": 0})
                continue
            R_p, t_p = pose
            degrees, metres = RM.pose_error(R_p, t_p, R_gt, t_gt)
            row.update({
                f"{key}_ok": 1,
                f"{key}_R": degrees, f"{key}_t": metres,
                f"{key}_add": RM.add(model_pts, R_p, t_p, R_gt, t_gt),
                f"{key}_adds": RM.add_s(model_pts, R_p, t_p, R_gt, t_gt),
                f"{key}_iou": RM.iou_3d(R_p, t_p, extents, R_gt, t_gt, extents),
                f"{key}_5cm5": int(RM.success_5cm5deg(R_p, t_p, R_gt, t_gt))})
        if tag == "PRED":
            row["n_support_pred"] = int(np.sum(support))
        else:
            row["n_support_gt"] = int(np.sum(support))
    return row


def stats(values):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size == 0:
        return {"n": 0, "median": None, "mean": None, "p90": None}
    return {"n": int(v.size), "median": round(float(np.median(v)), 4),
            "mean": round(float(v.mean()), 4),
            "p90": round(float(np.percentile(v, 90)), 4)}


def summarise(rows, arm, total):
    ok = [r for r in rows if r.get(f"{arm}_ok")]
    return {
        "n_frames": len(rows),
        "solved": len(ok),
        "solve_rate": round(len(ok) / max(total, 1), 4),
        "R_deg": stats([r[f"{arm}_R"] for r in rows]),
        "t_m": stats([r[f"{arm}_t"] for r in rows]),
        "ADD": stats([r[f"{arm}_add"] for r in rows]),
        "ADD_S": stats([r[f"{arm}_adds"] for r in rows]),
        "IoU3D": stats([r[f"{arm}_iou"] for r in rows]),
        "success_5cm5deg_unconditional":
            round(sum(r.get(f"{arm}_5cm5", 0) for r in rows) / max(total, 1), 4),
    }


def paired_delta(rows, metric, rng):
    pairs = [(r[f"F0_{metric}"], r[f"F3_{metric}"]) for r in rows
             if np.isfinite(r.get(f"F0_{metric}", np.nan))
             and np.isfinite(r.get(f"F3_{metric}", np.nan))]
    if len(pairs) < 3:
        return {"n_pairs": len(pairs), "delta_median": None, "CI95": None}
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    diff = b - a
    idx = rng.integers(0, len(diff), (BOOTSTRAP, len(diff)))
    boot = np.median(diff[idx], axis=1)
    lo, hi = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
    return {"n_pairs": len(pairs),
            "F0_median": round(float(np.median(a)), 4),
            "F3_median": round(float(np.median(b)), 4),
            "delta_median": round(float(np.median(diff)), 4),
            "CI95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(hi < 0 or lo > 0)}


def main():
    MS.deterministic()
    _, _, _, features = MS.lattice()
    weight = json.loads(
        open(os.path.join(ROOT, "data/pallet/results/paper_s2_multihead",
                          "theta_posealigned_d0.json")).read()
    )["seeds"]["seed1"]["selected_lambda_theta"]
    rng = np.random.default_rng(BOOT_SEED)

    report = {
        "name": "REAL_DEV_F0_F3_EVAL",
        "population": "REAL_DEV_POS_V1 (non-sealed canonical eval, positives only)",
        "sealed_not_accessed": list(DP.FINAL_TEST),
        "corner_input": "MS._decode_peaks (argmax, no threshold) -- parity with "
                        "the synthetic F3 validation",
        "detection_gate": f"extract_keypoints_from_belief @{THRESH}, "
                          f"n_det >= {N_DET_MIN} (reported, not used to filter)",
        "line_support": {
            "PRED": "derived from predicted corners -- deployable, MAIN",
            "GTSUP": "derived from GT corners -- parity reference only, oracle"},
        "lambda_theta": weight,
        "negative_dev": "MISSING -- no real negatives yet, so no precision / AP / "
                        "FPR is computed and no threshold is chosen",
        "seeds": {}}

    all_rows = {}
    for seed in (1, 2):
        model, path = load(seed)
        rows_by_set, rows = {}, []
        for key in OPEN_SETS:
            block = []
            for jp, ip, label in frames(key):
                row = run_frame(model, features, jp, ip, label, weight)
                if row is not None:
                    row["set"] = key
                    block.append(row)
            rows_by_set[key] = block
            rows.extend(block)
        all_rows[seed] = rows

        total = len(rows)
        det_rows = [r for r in rows if r["det"]]
        pnp_rows = [r for r in rows if r.get("F0_ok")]
        block = {"checkpoint": path, "n_frames": total,
                 "populations": {
                     "A_all_positive": {"n": total},
                     "B_corner_detected": {"n": len(det_rows),
                                           "rate": round(len(det_rows) / total, 4)},
                     "C_pnp_success": {"n": len(pnp_rows),
                                       "rate": round(len(pnp_rows) / total, 4)}},
                 "MAIN_unconditional_A": {
                     arm: summarise(rows, arm, total) for arm in ("F0", "F3")},
                 "diagnostic_B_detected": {
                     arm: summarise(det_rows, arm, len(det_rows) or 1)
                     for arm in ("F0", "F3")},
                 "parity_GTSUP": {
                     arm: summarise(rows, arm, total)
                     for arm in ("F0_GTSUP", "F3_GTSUP")},
                 "per_set": {
                     key: {arm: summarise(block, arm, len(block) or 1)
                           for arm in ("F0", "F3")}
                     for key, block in rows_by_set.items()},
                 "F3_minus_F0_paired": {
                     m: paired_delta(rows, m, rng)
                     for m in ("R", "t", "adds", "iou")},
                 "score_4kp": {
                     "positive_only": True,
                     "min": round(float(min(r["score_4kp"] for r in rows)), 4),
                     "median": round(float(np.median(
                         [r["score_4kp"] for r in rows])), 4),
                     "max": round(float(max(r["score_4kp"] for r in rows)), 4),
                     "recall_at": {str(t): round(float(np.mean(
                         [r["score_4kp"] >= t for r in rows])), 4)
                         for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)},
                     "note": "positive recall only. precision / AP / FPR need "
                             "real negatives, which do not exist yet."}}
        report["seeds"][f"seed{seed}"] = block
        main_f0 = block["MAIN_unconditional_A"]["F0"]
        main_f3 = block["MAIN_unconditional_A"]["F3"]
        print(f"  seed{seed}  A n={total}  det {len(det_rows)}  pnp {len(pnp_rows)}",
              flush=True)
        for name, s in (("F0", main_f0), ("F3", main_f3)):
            print(f"    {name}  R med {s['R_deg']['median']}  t med {s['t_m']['median']}"
                  f"  ADD-S {s['ADD_S']['median']}  IoU {s['IoU3D']['median']}"
                  f"  5cm5 {s['success_5cm5deg_unconditional']}", flush=True)
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(OUT, "REAL_DEV_F0_F3_EVAL.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    fields = sorted({k for rows in all_rows.values() for r in rows for k in r})
    with open(os.path.join(OUT, "REAL_DEV_PER_FRAME.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed"] + fields)
        writer.writeheader()
        for seed, rows in all_rows.items():
            for r in rows:
                writer.writerow({"seed": seed, **r})
    print("-> REAL_DEV_F0_F3_EVAL.json / REAL_DEV_PER_FRAME.csv")


if __name__ == "__main__":
    main()
