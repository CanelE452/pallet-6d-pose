#!/usr/bin/env python
"""pnp_valid_3d_audit.py — PAPER_S2 DiffPnP3D go/no-go audit.

각 학습 프레임에 대해 `pnp_valid_3d` 를 계산한다:
  dims(=dimensions_m 또는 3D cuboid edge 유도) 기반 object-frame 3D cuboid 를
  GT pose(pose_transform) + K 로 2D 투영 → JSON projected_cuboid 와의 max reproj err
  가 reproj_tol(기본 1px) 이하이고 모든 corner depth>0 이면 valid.

convention = camera-facing 0123 (프레임마다 corner index→octant 부호가 카메라
방위각에 따라 회전). 따라서 고정 object-frame 부호패턴을 강제하지 않고, box 의
48 대칭(signed permutation) 중 최소 reproj 를 취한다(order-free / 48-sym).
distinct W,D,H 박스에서는 올바른 camera-facing 배정만 ~0px, 나머지는 크게 벗어나
valid 판정이 안정적이다. (검증: clean 데이터 median 0.000px / 2D-aug ~134px)

산출:
  <out>/pnp_valid_3d_audit.json   : 데이터셋별·aug-kind별 통계 + 히스토그램 + invalid 예시
  <out>/pnp_valid_3d_index/<ds>.json : 프레임경로 -> {pnp_valid_3d,V,V8,reproj_min,dims,aug_kind,best_sym}
  <out>/audit_overlays/*.png       : valid/invalid 시각 대조 (--overlays)
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np


# ---- 48 box symmetries (signed permutation, det = +-1) ----
def _box_syms():
    syms = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3), dtype=np.float64)
            for i, pj in enumerate(perm):
                M[i, pj] = signs[i]
            syms.append(M)
    return np.stack(syms, axis=0)  # (48,3,3)


SYMS = _box_syms()


def canonical_box(W, D, H):
    """camera-facing 0123 reference box: X=width, Y=height, Z=depth.
    {0,1,4,5}=top(+H/2), {2,3,6,7}=bottom(-H/2), 0-3=front(-D/2), 4-7=back(+D/2).
    실제 프레임별 배정은 48-sym 탐색으로 정합(이 reference 는 탐색 시작점)."""
    hw, hh, hd = W / 2.0, H / 2.0, D / 2.0
    return np.array([
        [ hw,  hh, -hd], [-hw,  hh, -hd], [-hw, -hh, -hd], [ hw, -hh, -hd],
        [ hw,  hh,  hd], [-hw,  hh,  hd], [-hw, -hh,  hd], [ hw, -hh,  hd],
    ], dtype=np.float64)


def dims_from_cuboid(cub):
    """3D cuboid(8x3, camera-facing order) edge 에서 W/D/H 유도 (frame-invariant).
    edge 0-1=width, 0-4=depth, 0-3=height."""
    W = np.linalg.norm(cub[0] - cub[1])
    D = np.linalg.norm(cub[0] - cub[4])
    H = np.linalg.norm(cub[0] - cub[3])
    return float(W), float(D), float(H)


def load_frame(path):
    d = json.load(open(path))
    cam = d["camera_data"]
    ci = cam["intrinsics"]
    K = np.array([[ci["fx"], 0, ci["cx"]],
                  [0, ci["fy"], ci["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    W_img = cam.get("width", 640)
    H_img = cam.get("height", 480)
    o = d["objects"][0]
    P = np.array(o["pose_transform"], dtype=np.float64)
    R, t = P[:3, :3], P[:3, 3]
    cub = np.array(o["cuboid"], dtype=np.float64)
    pc = np.array(o["projected_cuboid"], dtype=np.float64)[:8]
    dm = o.get("dimensions_m")
    a = o.get("aug")
    if a and a.get("kind"):
        kind = a["kind"]
    elif o.get("ratio_randomized"):
        kind = "render_ratio"
    else:
        kind = "plain"
    return K, R, t, cub, pc, dm, kind, W_img, H_img


def eval_frame(K, R, t, cub, pc, dm):
    """48-sym min reproj. returns reproj_min, best_sym_idx, all_depth_pos, X_best(8x3)."""
    if dm is not None:
        W, D, H = dm["width"], dm["depth"], dm["height"]
    else:
        W, D, H = dims_from_cuboid(cub)
    X0 = canonical_box(W, D, H)                     # (8,3)
    Xs = np.einsum("sij,nj->sni", SYMS, X0)         # (48,8,3)
    Pc = Xs @ R.T + t                               # (48,8,3) camera frame
    z = Pc[:, :, 2:3]
    uvw = Pc @ K.T
    uv = uvw[:, :, :2] / np.clip(uvw[:, :, 2:3], 1e-6, None)
    err = np.linalg.norm(uv - pc[None], axis=2).max(axis=1)  # (48,)
    b = int(np.argmin(err))
    depth_pos = bool((z[b] > 0).all())
    dims = {"W": float(W), "D": float(D), "H": float(H),
            "diag": float(np.sqrt(W * W + D * D + H * H))}
    return float(err[b]), b, depth_pos, dims, Xs[b]


def compute_V(pc, W_img, H_img):
    inside = (pc[:, 0] >= 0) & (pc[:, 0] < W_img) & (pc[:, 1] >= 0) & (pc[:, 1] < H_img)
    return int(inside.sum())


def hist(vals, bins):
    h, edges = np.histogram(np.clip(vals, 0, bins[-1]), bins=bins)
    return {f"[{edges[i]:.2f},{edges[i+1]:.2f})": int(h[i]) for i in range(len(h))}


def blank_stats():
    return {"n": 0, "valid": 0, "V8": 0, "valid_and_V8": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=[
        "mixed_v8_train", "v4_split_base", "aug_squash_v2", "aug_trunc_v2",
        "aug_scale_v2", "paper_4pallet_mask_v1", "val"])
    ap.add_argument("--root", default="data/pallet/training_data")
    ap.add_argument("--out", default="data/pallet/results/paper_s2_scratch_diffpnp")
    ap.add_argument("--max_frames", type=int, default=0, help="0=all")
    ap.add_argument("--reproj_tol", type=float, default=1.0)
    ap.add_argument("--n_invalid_examples", type=int, default=20)
    ap.add_argument("--overlays", action="store_true")
    ap.add_argument("--n_overlays", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    idx_dir = os.path.join(args.out, "pnp_valid_3d_index")
    os.makedirs(idx_dir, exist_ok=True)

    hist_bins = [0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1e9]
    KINDS = ["plain", "render_ratio", "squash", "scale", "trunc"]

    audit = {"config": {"reproj_tol": args.reproj_tol, "root": args.root,
                        "hist_bins": hist_bins, "note": "48-sym min reproj (camera-facing 0123)"},
             "datasets": {}, "totals": {}, "invalid_examples": []}
    overlay_candidates = {"valid": [], "invalid": []}
    grand_valid_v8 = 0

    for ds in args.datasets:
        files = sorted(glob.glob(os.path.join(args.root, ds, "*.json")))
        if args.max_frames:
            files = files[:args.max_frames]
        ds_stat = blank_stats()
        kind_stat = {k: blank_stats() for k in KINDS}
        reproj_all = []
        index = {}
        n_err = 0
        for p in files:
            try:
                K, R, t, cub, pc, dm, kind, Wi, Hi = load_frame(p)
                reproj_min, best_sym, depth_pos, dims, Xb = eval_frame(K, R, t, cub, pc, dm)
            except Exception as e:  # 손상 프레임 스킵(정직 집계)
                n_err += 1
                continue
            V = compute_V(pc, Wi, Hi)
            V8 = (V == 8)
            valid = (reproj_min <= args.reproj_tol) and depth_pos
            reproj_all.append(reproj_min)

            ds_stat["n"] += 1
            ds_stat["valid"] += int(valid)
            ds_stat["V8"] += int(V8)
            ds_stat["valid_and_V8"] += int(valid and V8)
            ks = kind_stat.setdefault(kind, blank_stats())
            ks["n"] += 1
            ks["valid"] += int(valid)
            ks["V8"] += int(V8)
            ks["valid_and_V8"] += int(valid and V8)

            rel = os.path.relpath(p, args.root)
            index[rel] = {"pnp_valid_3d": valid, "V": V, "V8": V8,
                          "reproj_min_px": round(reproj_min, 4),
                          "best_sym": best_sym, "aug_kind": kind, "dims": dims}
            if (not valid) and len(audit["invalid_examples"]) < args.n_invalid_examples:
                audit["invalid_examples"].append(
                    {"path": rel, "reproj_min_px": round(reproj_min, 3),
                     "aug_kind": kind, "depth_pos": depth_pos, "V": V})
            if args.overlays:
                png = p[:-5] + ".png"
                if os.path.exists(png):
                    rec = (png, pc, Xb, K, R, t, reproj_min, kind, ds)
                    if valid and V8 and len(overlay_candidates["valid"]) < args.n_overlays:
                        overlay_candidates["valid"].append(rec)
                    elif (not valid) and len(overlay_candidates["invalid"]) < args.n_overlays:
                        overlay_candidates["invalid"].append(rec)

        json.dump(index, open(os.path.join(idx_dir, f"{ds}.json"), "w"))
        reproj_all = np.array(reproj_all) if reproj_all else np.array([0.0])
        audit["datasets"][ds] = {
            **ds_stat,
            "n_load_error": n_err,
            "reproj_median": round(float(np.median(reproj_all)), 4),
            "reproj_p90": round(float(np.percentile(reproj_all, 90)), 3),
            "reproj_hist": hist(reproj_all, hist_bins),
            "by_kind": {k: v for k, v in kind_stat.items() if v["n"] > 0},
        }
        grand_valid_v8 += ds_stat["valid_and_V8"]
        print(f"{ds:24s} n={ds_stat['n']:6d} valid={ds_stat['valid']:6d} "
              f"V8={ds_stat['V8']:6d} valid&V8={ds_stat['valid_and_V8']:6d} "
              f"reproj_med={audit['datasets'][ds]['reproj_median']:.3f}")

    audit["totals"]["valid_and_V8_all"] = grand_valid_v8
    audit["totals"]["by_dataset_valid_and_V8"] = {
        ds: audit["datasets"][ds]["valid_and_V8"] for ds in args.datasets}
    json.dump(audit, open(os.path.join(args.out, "pnp_valid_3d_audit.json"), "w"), indent=2)
    print(f"\nTOTAL valid&V8 (DiffPnP3D pool) = {grand_valid_v8}")
    print(f"audit -> {os.path.join(args.out, 'pnp_valid_3d_audit.json')}")

    if args.overlays:
        _render_overlays(overlay_candidates, os.path.join(args.out, "audit_overlays"))


def _render_overlays(cands, out_dir):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    n = 0
    for tag in ("valid", "invalid"):
        for (png, pc, Xb, K, R, t, reproj, kind, ds) in cands[tag]:
            img = cv2.imread(png)
            if img is None:
                continue
            Pc = Xb @ R.T + t
            uvw = Pc @ K.T
            reuv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
            for (a, b) in EDGES:
                cv2.line(img, tuple(pc[a].astype(int)), tuple(pc[b].astype(int)), (0, 255, 0), 1)   # GT green
                cv2.line(img, tuple(reuv[a].astype(int)), tuple(reuv[b].astype(int)), (0, 0, 255), 1)  # reproj red
            cv2.putText(img, f"{tag} {ds} {kind} reproj={reproj:.2f}px", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            out = os.path.join(out_dir, f"{tag}_{ds}_{os.path.basename(png)}")
            cv2.imwrite(out, img)
            n += 1
    print(f"overlays ({n}) -> {out_dir}  [green=GT projected_cuboid, red=dims-box reproj via GT pose]")


if __name__ == "__main__":
    main()
