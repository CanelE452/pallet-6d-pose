"""utility_selftrain_v1 section 3 — GT-free feature table for pool and labeled frames.

Two sources share one feature definition:

  --source pool      reads the frozen teacher cache (1000 pool frames).  No inference.
  --source labeled   runs the teacher on POLICY_DEV / POLICY_VAL frames under the same
                     recipe as the cache, and attaches GT error columns for ANALYSIS ONLY.

STUDENT_EVAL is never read here.

Every feature below is computable at deployment time.  GT keypoint error, GT bbox IoU
and GT yaw error appear only in the gt_* columns of the labeled table and are forbidden
as selection inputs (instruction sections 3, 16, 17).

Pose features come from the PnP hypothesis with the smallest s_reproj.  The registry
leaves the width/depth assignment ambiguous, so a single pose must be named before any
pose feature is defined; this choice is recorded in METHOD_LOCK and never revisited.
Elevation reuses the canonical formula from
scripts/research/accuracy_root_cause_v1/elevation_check.py, with the predicted pose
substituted for the GT pose.
"""
from __future__ import annotations

import argparse, json, math, sys, csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/self_training_yolo"))
sys.path.insert(0, str(ROOT / "scripts/annotate"))
from pseudo_label_filters import (  # noqa: E402
    geometry_scores, registry_hypotheses, _solve, _project, projected_diagonal,
    N_CORNERS,
)

OUT = ROOT / "data/pallet/results/utility_selftrain_v1"
DOCS = ROOT / "_docs/experiments/utility_selftrain_v1"
CACHE = ROOT / "data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json"
LOCK = ROOT / "data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json"
REGISTRY = ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
WS = ROOT / "data/evaluation/pallet_eval_v1"

PAD, IMGSZ, CONF_FLOOR = 100, 640, 0.001          # cache recipe
FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6, 8]
POOL_OBJECT = "plastic_standard_110x130x11"


# ── geometry helpers ────────────────────────────────────────────────────────

def registry_dims(object_type: str) -> dict:
    reg = json.loads(REGISTRY.read_text())
    for obj in reg["objects"]:
        if obj["object_type"] == object_type or object_type in obj.get("aliases", []):
            return obj["physical_dimensions_m"]
    raise SystemExit(f"OBJECT_TYPE_NOT_IN_REGISTRY: {object_type}")


def best_pose(kp_xy, valid, K, dims):
    """PnP pose of the hypothesis with the smallest reprojection score."""
    kp = np.asarray(kp_xy, float)
    v = np.asarray(valid, bool)
    idx = np.flatnonzero(v[:N_CORNERS])
    if len(idx) < 4:
        return None
    best = None
    for name, kp3d in registry_hypotheses(dims):
        solved = _solve(kp3d[idx], kp[idx], K)
        if solved is None:
            continue
        rvec, tvec = solved
        reproj = _project(kp3d[:N_CORNERS], rvec, tvec, K)
        diag = projected_diagonal(reproj)
        if not np.isfinite(diag) or diag < 1e-6:
            continue
        resid = float(np.median(np.linalg.norm(reproj[idx] - kp[idx], axis=1)))
        score = resid / diag
        if best is None or score < best["score"]:
            best = {"name": name, "score": score, "rvec": rvec, "tvec": tvec,
                    "reproj_px": resid, "diag_px": diag}
    return best


def pose_features(pose):
    """Predicted elevation / depth / yaw / lateral offset.  All GT-free."""
    if pose is None:
        return {k: None for k in ("pred_elev_deg", "pred_depth_m", "pred_yaw_c2_deg",
                                  "pred_lateral_m", "pnp_reproj_px",
                                  "proj_cuboid_diag_px", "pose_hypothesis")}
    import cv2
    R, _ = cv2.Rodrigues(pose["rvec"])
    t = np.asarray(pose["tvec"], float).reshape(3)
    n = R @ np.array([0.0, -1.0, 0.0])              # pallet top normal, camera frame
    u = -t / (np.linalg.norm(t) + 1e-12)            # object -> camera
    elev = math.degrees(math.asin(float(np.clip(abs(float(n @ u)), 0, 1))))
    # yaw about the camera-frame vertical, folded into the C2 equivalence {0,180}
    fwd = R @ np.array([0.0, 0.0, 1.0])
    yaw = math.degrees(math.atan2(float(fwd[0]), float(fwd[2])))
    yaw_c2 = abs((yaw + 90.0) % 180.0 - 90.0)
    return {
        "pred_elev_deg": elev,
        "pred_depth_m": float(t[2]),
        "pred_yaw_c2_deg": yaw_c2,
        "pred_lateral_m": float(t[0]),
        "pnp_reproj_px": pose["reproj_px"],
        "proj_cuboid_diag_px": pose["diag_px"],
        "pose_hypothesis": pose["name"],
    }


def box_iou(a, b):
    if a is None or b is None:
        return None
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ub = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = ua + ub - inter
    return float(inter / den) if den > 1e-9 else None


def unflip_box(box, width):
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return [width - 1 - x1, y0, width - 1 - x0, y1]


def unflip_keypoints(kp, conf, width):
    kp = np.asarray(kp, float).copy()
    kp[:, 0] = width - 1 - kp[:, 0]
    return kp[FLIP_IDX], np.asarray(conf, float)[FLIP_IDX]


# ── feature row ─────────────────────────────────────────────────────────────

def build_row(*, frame_id, session, condition, W, H, K, dims,
              box, box_conf, kp_xy, kp_conf, flip_box, flip_kp, flip_conf,
              kp_conf_threshold, min_corners):
    kp_xy = np.asarray(kp_xy, float)
    kp_conf = np.asarray(kp_conf, float)
    valid = kp_conf >= kp_conf_threshold
    n_valid = int(valid[:N_CORNERS].sum())

    # flip_* arrive already mapped back into the original image frame.  The frozen
    # teacher cache stores them that way (verified: flip_top1 box matches top1 without
    # any transform), and pseudo_label_filters compares them to top1 directly.
    if flip_kp is not None:
        f_kp = np.asarray(flip_kp, float)
        f_valid = np.asarray(flip_conf, float) >= kp_conf_threshold
    else:
        f_kp = f_valid = None

    # build_pseudo_manifests.py:137 gates the same way.  With fewer than six valid
    # corners the leave-one-out solve drops to four points, which cv2 rejects.
    candidate = n_valid >= min_corners
    if candidate:
        scores = geometry_scores(kp_xy, valid, K, dims, f_kp, f_valid)
        s_r, s_m, s_f = scores["s_reproj"], scores["s_remove"], scores["s_flip"]
        pose = best_pose(kp_xy, valid, K, dims)
    else:
        s_r = s_m = s_f = None
        pose = None
    finite = [v for v in (s_r, s_m, s_f) if v is not None and math.isfinite(v)]
    f4pair = [v for v in (s_m, s_f) if v is not None and math.isfinite(v)]
    pf = pose_features(pose)

    img_diag = math.hypot(W, H)
    bw = bh = bdiag = barea = None
    if box is not None:
        bw, bh = box[2] - box[0], box[3] - box[1]
        bdiag = math.hypot(bw, bh) / img_diag
        barea = (bw * bh) / (W * H)

    row = {
        "frame_id": frame_id, "session": session, "condition": condition,
        # A. confidence
        "box_conf": box_conf,
        "kp_conf_min": float(kp_conf[:N_CORNERS].min()),
        "kp_conf_mean": float(kp_conf[:N_CORNERS].mean()),
        "kp_conf_std": float(kp_conf[:N_CORNERS].std()),
        "valid_corner_count": n_valid,
        "candidate": candidate,
        # B. geometry consistency
        "s_reproj": s_r, "s_remove": s_m, "s_flip": s_f,
        "max_geom": max(finite) if finite else None,
        "mean_geom": (sum(finite) / len(finite)) if finite else None,
        "max_geom_f4": max(f4pair) if len(f4pair) == 2 else None,
        "ratio_reproj_remove": (s_r / s_m) if (s_r and s_m and math.isfinite(s_r)
                                               and math.isfinite(s_m) and s_m > 1e-9) else None,
        "ratio_remove_flip": (s_m / s_f) if (s_m and s_f and math.isfinite(s_m)
                                             and math.isfinite(s_f) and s_f > 1e-9) else None,
        # C. scale / projected size
        "bbox_w_px": bw, "bbox_h_px": bh,
        "bbox_area_frac": barea, "bbox_diag_frac": bdiag,
        "proj_diag_frac": (pf["proj_cuboid_diag_px"] / img_diag)
        if pf["proj_cuboid_diag_px"] else None,
        # E. GT-free transform consistency
        "flip_box_iou": box_iou(box, flip_box),
    }
    row.update(pf)
    return row


# ── sources ─────────────────────────────────────────────────────────────────

def rows_from_pool(kp_conf_threshold, min_corners):
    cache = json.loads(CACHE.read_text())
    dims = registry_dims(POOL_OBJECT)
    out = []
    for e in cache["entries"]:
        top = e.get("top1")
        if not top:
            continue
        flip = e.get("flip_top1")
        out.append(build_row(
            frame_id=e["image_sha256"], session=e["capture_session"],
            condition=e["paper_condition"], W=e["image_width"], H=e["image_height"],
            K=np.asarray(e["camera_matrix"], float), dims=dims,
            box=top.get("box_xyxy"), box_conf=top["box_conf"],
            kp_xy=top["keypoints_xy"], kp_conf=top["keypoints_conf"],
            flip_box=flip.get("box_xyxy") if flip else None,
            flip_kp=flip.get("keypoints_xy") if flip else None,
            flip_conf=flip.get("keypoints_conf") if flip else None,
            kp_conf_threshold=kp_conf_threshold, min_corners=min_corners))
    return out


def rows_from_labeled(roles, kp_conf_threshold, min_corners, device):
    """Run the teacher under the cache recipe and attach GT error columns."""
    import cv2
    from ultralytics import YOLO

    split = json.loads((DOCS / "SPLIT_CONTRACT.json").read_text())
    lock = json.loads(LOCK.read_text())
    weights = ROOT / lock["teacher_checkpoint"]
    model = YOLO(str(weights))

    sys.path.insert(0, str(ROOT / "scripts/evaluation"))
    from eval_workspace import load_frames, evaluation_population_views
    frames = load_frames(WS)
    pos = evaluation_population_views(frames)["PAPER_EVAL_POSITIVE"]
    want = {s for r in roles for s in split["roles"][r]["sessions"]}
    sel = [f for f in pos if f["session_id"] in want]

    def predict(img):
        padded = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        res = model.predict(padded, imgsz=IMGSZ, conf=CONF_FLOOR, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None, None, None, None
        conf = res.boxes.conf.cpu().numpy()
        i = int(np.argmax(conf))
        box = res.boxes.xyxy.cpu().numpy()[i] - PAD
        kp = res.keypoints.xy.cpu().numpy()[i] - PAD
        kc = res.keypoints.conf.cpu().numpy()[i]
        return box.tolist(), float(conf[i]), kp, kc

    out = []
    for n, f in enumerate(sel, 1):
        img = cv2.imread(str(WS / f["image_path"]))
        if img is None:
            continue
        H, W = img.shape[:2]
        ann = json.loads((WS / f["annotation_path"]).read_text())
        obj = ann["objects"][0]
        ins = ann["camera_data"]["intrinsics"]
        K = np.asarray([[ins["fx"], 0.0, ins["cx"]],
                        [0.0, ins["fy"], ins["cy"]],
                        [0.0, 0.0, 1.0]], float)
        dims = registry_dims(f["object_type"])

        box, bconf, kp, kc = predict(img)
        if box is None:
            continue
        fbox, _, fkp, fkc = predict(cv2.flip(img, 1))
        if fbox is not None:
            fbox = unflip_box(fbox, W)
            fkp, fkc = unflip_keypoints(fkp, fkc, W)

        row = build_row(
            frame_id=f["frame_id"], session=f["session_id"],
            condition=f["lighting"] or "unknown", W=W, H=H, K=K, dims=dims,
            box=box, box_conf=bconf,
            kp_xy=kp, kp_conf=kc, flip_box=fbox, flip_kp=fkp, flip_conf=fkc,
            kp_conf_threshold=kp_conf_threshold, min_corners=min_corners)
        row["role"] = next(r for r in roles if f["session_id"] in split["roles"][r]["sessions"])
        row["object_type"] = f["object_type"]
        row["occlusion"] = f["occlusion"]
        row["distance_bin"] = f["distance_bin"]

        # GT columns — ANALYSIS ONLY
        gt = obj.get("keypoint_annotations")
        if gt:
            # same supervision mask as filter_quality_m4.py:104-107
            gxy, sup = [], []
            for a in gt:
                xy = a.get("xy")
                gxy.append(xy if xy else [float("nan")] * 2)
                sup.append(bool(a.get("visibility", 0)) and a.get("in_frame", True)
                           and xy is not None)
            g = np.asarray(gxy, float)
            m = np.asarray(sup, bool)
            if m.any():
                err = np.linalg.norm(kp[:len(g)][m] - g[m], axis=1)
                row["gt_corner_median_px"] = float(np.median(err))
                row["gt_corner_max_px"] = float(err.max())
                row["gt_gross_gt20"] = int((err > 20).sum())
                row["gt_catastrophic_gt40"] = int((err > 40).sum())
                row["gt_n_supervised"] = int(m.sum())
        out.append(row)
        if n % 25 == 0:
            print(f"  labeled {n}/{len(sel)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pool", "labeled"], required=True)
    ap.add_argument("--roles", nargs="*", default=["POLICY_DEV", "POLICY_VAL"])
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    if "STUDENT_EVAL" in args.roles:
        print("REFUSED: STUDENT_EVAL is frozen until section 12.")
        return 2

    lock = json.loads(LOCK.read_text())
    thr = lock["keypoint_validity"]["kp_conf_threshold"]
    minc = lock["keypoint_validity"]["min_valid_corners"]

    OUT.mkdir(parents=True, exist_ok=True)
    if args.source == "pool":
        rows = rows_from_pool(thr, minc)
        path = OUT / "FEATURES_POOL.csv"
    else:
        rows = rows_from_labeled(args.roles, thr, minc, args.device)
        path = OUT / "FEATURES_LABELED.csv"

    if not rows:
        print("no rows")
        return 1
    cols = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)}  rows={len(rows)}  cols={len(cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
