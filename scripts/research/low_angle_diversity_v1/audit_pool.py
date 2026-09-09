"""§3 — 미사용 oblique pool 전수 재검증.  기존 문서를 복사하지 않고 label 에서 재생성한다."""
from __future__ import annotations
import json, glob, collections
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
POOL = REPO / "data/pallet/training_data/paper_release/oblique/extracted"
OUT = REPO / "data/pallet/results/low_angle_diversity_v1"
PKGS = ["corner_la_oblique_v1_y15_30", "corner_la_oblique_v1_y30_plus"]


def elevation_deg(R, t):
    """물체 up 축(fixed y)과 시선의 각.  0 = edge-on, 90 = 바로 위에서."""
    up = R[:, 1]
    d = t / np.linalg.norm(t)
    return float(np.degrees(np.arcsin(np.clip(abs(float(up @ d)), 0, 1))))


def main():
    rows, bad = [], collections.Counter()
    for pkg in PKGS:
        for f in sorted(glob.glob(str(POOL / pkg / "labels" / "*_label.json"))):
            d = json.load(open(f))
            cam = d.get("camera_data") or {}
            intr = cam.get("intrinsics") or {}
            objs = d.get("objects") or []
            if len(objs) != 1:
                bad["not_one_object"] += 1
                continue
            o = objs[0]
            M = np.asarray(o["pose_transform"], float)
            R, t = M[:3, :3], M[:3, 3]
            dm = o["dimensions_m"]
            pc = np.asarray(o["projected_cuboid"], float)[:8]
            ok_K = all(isinstance(intr.get(k), (int, float)) and intr[k] > 0 for k in ("fx", "fy"))
            ok_pose = bool(np.isfinite(M).all() and t[2] > 0)
            ok_kp = bool(np.isfinite(pc).all() and pc.shape == (8, 2))
            if not (ok_K and ok_pose and ok_kp):
                bad["invalid_geometry"] += 1
            w, h, dp = dm["width"], dm["height"], dm["depth"]
            diag = float(np.linalg.norm(pc.max(0) - pc.min(0)))
            rows.append({
                "pkg": pkg, "stem": Path(f).name.replace("_label.json", ""),
                "source_asset": o.get("source_asset"),
                "keypoint_convention": o.get("keypoint_convention"),
                "elevation_deg": elevation_deg(R, t),
                "distance_m": float(np.linalg.norm(t)),
                "diag_px": diag,
                "diag_ratio": diag / float(np.hypot(cam.get("width", 1), cam.get("height", 1))),
                "w": w, "h": h, "d": dp,
                "aspect": max(w, dp) / min(w, dp),
                "scene_preset": cam.get("scene_preset"),
                "background": cam.get("background_asset"),
                "perm_v4_ok": isinstance(o.get("perm_v4"), list) and len(o["perm_v4"]) == 8,
                "K_ok": ok_K, "pose_ok": ok_pose, "kp_ok": ok_kp,
            })
    n = len(rows)
    elev = np.array([r["elevation_deg"] for r in rows])
    dist = np.array([r["distance_m"] for r in rows])
    diag = np.array([r["diag_ratio"] for r in rows])
    asp = np.array([r["aspect"] for r in rows])
    assets = collections.Counter(r["source_asset"] for r in rows)
    p = np.array(list(assets.values()), float) / n
    eff = float(np.exp(-(p * np.log(p)).sum()))
    conv = collections.Counter(r["keypoint_convention"] for r in rows)

    def pct(a, ps=(5, 50, 95)):
        return {f"p{q}": float(np.percentile(a, q)) for q in ps}

    rep = {
        "schema_version": "low_angle_diversity_v1_pool_audit_v1",
        "N_total": n,
        "invalid": dict(bad),
        "N_elev_lt8": int((elev < 8).sum()),
        "frac_elev_lt8": float((elev < 8).mean()),
        "elevation_deg": {**pct(elev), "min": float(elev.min()), "max": float(elev.max())},
        "distance_m": pct(dist), "diag_ratio": pct(diag), "aspect_ratio": pct(asp),
        "source_asset_counts": dict(assets),
        "effective_asset_count": eff,
        "keypoint_convention": dict(conv),
        "scene_preset": dict(collections.Counter(r["scene_preset"] for r in rows)),
        "background": dict(collections.Counter(r["background"] for r in rows)),
        "K_valid": int(sum(r["K_ok"] for r in rows)),
        "pose_valid": int(sum(r["pose_ok"] for r in rows)),
        "keypoints_valid": int(sum(r["kp_ok"] for r in rows)),
        "perm_v4_valid": int(sum(r["perm_v4_ok"] for r in rows)),
        "by_pkg": {pkg: {"n": sum(1 for r in rows if r["pkg"] == pkg),
                         "elev_p50": float(np.median([r["elevation_deg"] for r in rows if r["pkg"] == pkg])),
                         "assets": dict(collections.Counter(r["source_asset"] for r in rows if r["pkg"] == pkg))}
                   for pkg in PKGS},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "POOL_AUDIT.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    keys = list(rows[0].keys())
    with (OUT / "POOL_FRAMES.csv").open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    print(json.dumps(rep, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
