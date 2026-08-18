"""Task-level evaluation of a checkpoint on the SYNTHETIC validation set.

This is not the real-domain evaluator the deployment ultimately needs. Real data is not
part of this round, so no real detection rate, no real PnP rate and no real yaw error is
produced here. What this measures is whether the predicted keypoints are good enough to
recover a pose at all, on data drawn from the same synthetic distribution as training.

Per frame it records:
  detected, bbox_conf, keypoint pixel errors (near / far / centroid / all),
  bbox-diagonal-normalised error, PnP success, median reprojection error,
  predicted vs GT yaw (synthetic GT pose is exact), and the yaw error.

PnP follows the deployment contract: object_points from
depth_cam/calib/pose6d_adapter.py, EPnP + solvePnPRefineLM for >=6 visible points.
Coordinates have 100 px of padding removed before PnP, and the ORIGINAL K is used.

Usage:
  python .../eval_task_pose.py --weights runs/.../best.pt --split val --domain T
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "challenge/yolo_pose_one_model"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(OUT / "scripts"))
from verify_kp_contract import model_points  # noqa: E402

PAD = 100


def yaw_from_R(R):
    """Rotation about the camera's vertical axis, in degrees.

    Uses the pallet's local +X (its width axis) projected on the camera's XZ plane.
    Reported for the synthetic domain only; the deployment's psi convention
    (pose6d_adapter.pose6d_to_align_vars) is a different quantity and is not claimed here.
    """
    x_axis = R[:, 0]
    return math.degrees(math.atan2(x_axis[0], x_axis[2]))


def gt_from_json(ann_path):
    d = json.load(open(ann_path, encoding="utf-8"))
    o = d["objects"][0]
    cd = d["camera_data"]
    it = cd["intrinsics"]
    K = np.array([[it["fx"], 0, it["cx"]], [0, it["fy"], it["cy"]], [0, 0, 1]], float)
    if isinstance(o.get("dimensions_m"), dict):
        dm = o["dimensions_m"]
        dims = (dm["width"], dm["height"], dm["depth"])
    else:
        dep, wid, hei = o["cuboid_dimensions_m"]      # [depth, width, height]
        dims = (wid, hei, dep)
    kp = np.array([*o["projected_cuboid"][:8],
                   o.get("projected_cuboid_centroid") or [-1, -1]], float)
    R = None
    pt = o.get("pose_transform")
    if pt:
        R = np.array(pt, float)[:3, :3]
    return kp, K, dims, R


def solve_pnp(kps, K, dims):
    obj = model_points(*dims)
    vis = (kps[:, 0] >= 0) & (kps[:, 1] >= 0) & ~np.isnan(kps).any(axis=1)
    n = int(vis.sum())
    if n < 6:
        return None
    o = obj[vis].reshape(-1, 1, 3)
    i = kps[vis].reshape(-1, 1, 2)
    D = np.zeros((5, 1))
    try:
        ok, rv, tv = cv2.solvePnP(o, i, K, D, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None
        rv, tv = cv2.solvePnPRefineLM(o, i, K, D, rv, tv)
    except cv2.error:
        return None
    proj, _ = cv2.projectPoints(o, rv, tv, K, D)
    med = float(np.median(np.linalg.norm(proj.reshape(-1, 2) - i.reshape(-1, 2), axis=1)))
    R, _ = cv2.Rodrigues(rv)
    return {"n_used": n, "median_reproj": med, "R": R, "t": tv.ravel()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", default="datasets/stage_a")
    ap.add_argument("--split", default="val")
    ap.add_argument("--domain", default="T", choices=["T", "G", "all"])
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--kp-conf", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from ultralytics import YOLO

    reg = {r["sample_id"].replace("/", "__"): r for r in
           csv.DictReader(open(OUT / "manifests/all_samples.csv", encoding="utf-8"))}
    img_dir = OUT / args.dataset / "images" / args.split
    prefix = {"T": "T__", "G": "G__", "all": ""}[args.domain]
    imgs = sorted(p for p in img_dir.glob(f"{prefix}*.png") if "__rep" not in p.stem)
    if args.limit:
        imgs = imgs[:args.limit]
    print(f"{len(imgs)} frames  weights={args.weights}")

    model = YOLO(args.weights)
    rows = []
    for i in range(0, len(imgs), 32):
        batch = imgs[i:i + 32]
        res = model.predict([str(p) for p in batch], conf=args.det_conf, verbose=False,
                            device=0, imgsz=640)
        for p, r in zip(batch, res):
            rec = {"sample_id": p.stem, "detected": 0, "bbox_conf": "", "pnp_ok": 0}
            meta = reg.get(p.stem)
            if meta is None:
                continue
            gt_kp, K, dims, R_gt = gt_from_json(REPO / meta["annotation_path"])
            gt_pad = gt_kp + PAD
            if len(r.boxes) == 0:
                rows.append(rec)
                continue
            # associate with the detection whose keypoints sit closest to the GT centroid
            kxy = r.keypoints.xy.cpu().numpy()          # (n,9,2) padded pixels
            kcf = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
                   else np.ones(kxy.shape[:2]))
            cen = gt_pad[8]
            j = int(np.argmin([np.linalg.norm(k[8] - cen) for k in kxy])) if np.isfinite(
                cen).all() else int(np.argmax(r.boxes.conf.cpu().numpy()))
            pred = kxy[j].copy()
            conf = kcf[j]
            rec["detected"] = 1
            rec["bbox_conf"] = float(r.boxes.conf.cpu().numpy()[j])

            valid = (gt_pad[:, 0] >= 0) & (gt_pad[:, 1] >= 0)
            err = np.full(9, np.nan)
            err[valid] = np.linalg.norm(pred[valid] - gt_pad[valid], axis=1)
            b = r.boxes.xyxy.cpu().numpy()[j]
            diag = math.hypot(b[2] - b[0], b[3] - b[1]) or 1.0
            rec.update({
                "err_near": float(np.nanmean(err[:4])),
                "err_far": float(np.nanmean(err[4:8])),
                "err_centroid": float(err[8]),
                "err_all": float(np.nanmean(err)),
                "err_norm": float(np.nanmean(err) / diag),
                "n_kp_conf_ok": int((conf >= args.kp_conf).sum()),
            })
            # PnP on ORIGINAL coordinates with the ORIGINAL K
            un = pred - PAD
            un[conf < args.kp_conf] = -1.0
            s = solve_pnp(un, K, dims)
            if s:
                rec.update({"pnp_ok": 1, "pnp_n_used": s["n_used"],
                            "median_reproj": s["median_reproj"],
                            "yaw_pred": yaw_from_R(s["R"])})
                if R_gt is not None:
                    yg = yaw_from_R(R_gt)
                    rec["yaw_gt"] = yg
                    d = abs(((rec["yaw_pred"] - yg + 180) % 360) - 180)
                    rec["yaw_err"] = d
                    rec["yaw_err_mod180"] = min(d, abs(180 - d))
            rows.append(rec)

    cols = ["sample_id", "detected", "bbox_conf", "err_near", "err_far", "err_centroid",
            "err_all", "err_norm", "n_kp_conf_ok", "pnp_ok", "pnp_n_used",
            "median_reproj", "yaw_pred", "yaw_gt", "yaw_err", "yaw_err_mod180"]
    dst = Path(args.out) if args.out else (OUT / "reports" /
                                           f"eval_{args.domain}_{Path(args.weights).stem}.csv")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    n = len(rows)
    det = sum(r["detected"] for r in rows)
    pnp = sum(r["pnp_ok"] for r in rows)

    def med(key, only_pnp=False):
        v = [r[key] for r in rows if r.get(key) not in ("", None)
             and (not only_pnp or r["pnp_ok"])]
        v = [x for x in v if isinstance(x, float) and not math.isnan(x)]
        return float(np.median(v)) if v else float("nan")

    ye = [r["yaw_err_mod180"] for r in rows if r.get("yaw_err_mod180") not in ("", None)]
    print(f"\n domain={args.domain} n={n}")
    print(f"  detection rate   {det}/{n} = {100*det/max(n,1):.1f}%")
    print(f"  PnP success      {pnp}/{n} = {100*pnp/max(n,1):.1f}%  (of detected: "
          f"{100*pnp/max(det,1):.1f}%)")
    print(f"  kp err median    near {med('err_near'):.2f}  far {med('err_far'):.2f}  "
          f"centroid {med('err_centroid'):.2f}  all {med('err_all'):.2f} px")
    print(f"  normalised err   {med('err_norm'):.4f}")
    print(f"  median reproj    {med('median_reproj', True):.2f} px")
    if ye:
        a = np.array(ye)
        print(f"  yaw |err| (mod180, SYNTHETIC only)  median {np.median(a):.2f}  "
              f"p95 {np.percentile(a,95):.2f}  >15deg {100*np.mean(a>15):.1f}%")
    print(f"\nwrote {dst.relative_to(REPO)}")


if __name__ == "__main__":
    main()
