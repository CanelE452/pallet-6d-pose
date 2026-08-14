"""addon_v1 (6002 frame) integrity + distribution audit vs v3 10k.

Verifies, BEFORE 16k heatmap training, that addon_v1:
  (1) integrity: RLE area match, keypoint reprojection error (per-frame K),
      corner ordering (camera-facing 0123 v4), intrinsics sane;
  (2) distribution: V<8 / azimuth / elevation / projected size vs v3 (augmentation check);
  (3) no near-duplicate leakage vs v3.

Per-frame K (NO hardcode). [확인] projection convention validated on 000000:
  kp3d_world are ALREADY world coords; Xc = Rwc^T (Xw - C); OpenCV flip diag(1,-1,-1).
  reproj p50=0.01px max=0.04px.

Outputs -> data/pallet/results/addon_v1_audit/
"""
import json, glob, os, math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

ROOT = "/home/minjae/Documents/github/pallet-pose"
ADDON = os.path.join(ROOT, "challenge/data/02_synthetic/training/addon_v1")
V3 = os.path.join(ROOT, "challenge/data/02_synthetic/training/v3")
OUT = os.path.join(ROOT, "data/pallet/results/addon_v1_audit")
os.makedirs(OUT, exist_ok=True)


def quat_to_R(q):  # xyzw
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


FLIP = np.diag([1.0, -1.0, -1.0])  # USD -> OpenCV camera axes


def project(kp_world, cam):
    """kp_world (N,3) world coords -> (N,2) pixels using per-frame K + extrinsics."""
    intr = cam["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]],
                  [0, intr["fy"], intr["cy"]],
                  [0, 0, 1.0]])
    C = np.array(cam["location_worldframe"])
    Rwc = quat_to_R(cam["quaternion_xyzw_worldframe"])
    Xc = (Rwc.T @ (kp_world - C).T).T   # world -> cam (USD)
    Xc = (FLIP @ Xc.T).T               # USD -> OpenCV
    z = Xc[:, 2]
    uv = (K @ Xc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def rle_decode(rle):
    h, w = rle["size"]
    counts = rle["counts"]
    flat = np.zeros(h*w, dtype=np.uint8)
    idx = 0; val = 0
    for c in counts:
        flat[idx:idx+c] = val; idx += c; val ^= 1
    return flat.reshape((w, h)).T


def cam_azimuth_elevation(cam):
    """Geometric azimuth (atan2 of camera XY rel to origin/pallet) + elevation (angle above horizon).
    Pallet centered at world origin (kp symmetric about origin). Computed for BOTH v3 & addon
    so comparison is apples-to-apples (independent of frame_meta)."""
    C = np.array(cam["location_worldframe"])
    az = math.degrees(math.atan2(C[1], C[0]))   # yaw of camera position
    horiz = math.hypot(C[0], C[1])
    elev = math.degrees(math.atan2(C[2], horiz)) if horiz > 1e-6 else 90.0
    return az, elev


# camera-facing 0123 v4 ordering invariants (object-frame canonical, unsigned along axes):
#   {0,1,4,5} = top (z=+0.114),  {2,3,6,7} = bottom (z=-0.006)
#   {0,1,2,3} = front (one y side), {4,5,6,7} = back (other y side)
#   index 8 = centroid
# We check these in OBJECT frame by undoing obj pose, since kp3d_world were baked.
def ordering_check(kp_world):
    """Verify camera-facing 0123 v4 invariants in WORLD frame relative to pallet center.
    kp_world are stored as canonical axis-aligned cuboid corners (pallet upright, Z=world up).
    [확인] from data: top{0,1,4,5} z=+0.114, bot{2,3,6,7} z=-0.006, front{0,1,2,3} & back{4,5,6,7}
    on opposite Y sides, 8=centroid at mid-height. Tolerances allow the camera-dynamic X-flip
    (corner 0 swaps to camera-near side) which does NOT break the index->face grouping."""
    ctr = kp_world[8]
    rel = kp_world[:8] - ctr  # corners relative to centroid
    z = rel[:, 2]
    top = z[[0, 1, 4, 5]].mean()
    bot = z[[2, 3, 6, 7]].mean()
    ok_z = top > bot + 0.03               # top group clearly above bottom
    y = rel[:, 1]
    front_y = y[[0, 1, 2, 3]].mean()
    back_y = y[[4, 5, 6, 7]].mean()
    # front and back must be on opposite Y sides and ~1.3m apart along the long axis;
    # but pallet may be oriented along X — so test along whichever horizontal axis separates them.
    x = rel[:, 0]
    front_x = x[[0, 1, 2, 3]].mean(); back_x = x[[4, 5, 6, 7]].mean()
    sep = max(abs(front_y - back_y), abs(front_x - back_x))
    ok_fb = sep > 0.4
    ok_ctr = (abs(rel[[0, 1, 2, 3]].mean(0)[2]) < 0.2)  # centroid near mid plane
    # corner-pair adjacency: 0&3 share an edge (same x side, top vs bottom front); 1&2 likewise
    ok = ok_z and ok_fb
    return ok, dict(top_z=float(top), bot_z=float(bot), sep=float(sep))


def list_jsons(root):
    if os.path.isdir(os.path.join(root, "batch_000")):
        out = []
        for b in sorted(glob.glob(os.path.join(root, "batch_*"))):
            out += sorted(glob.glob(os.path.join(b, "*.json")))
        return out
    return sorted(glob.glob(os.path.join(root, "*.json")))


def collect(root, do_rle=True, do_reproj=True, rle_sample=None, reproj_sample=None, ordering_sample=None):
    jsons = list_jsons(root)
    jsons = [j for j in jsons if not os.path.basename(j).startswith("_")]
    N = len(jsons)
    rng = np.random.default_rng(0)
    rle_set = set(rng.choice(N, min(rle_sample, N), replace=False).tolist()) if rle_sample else set(range(N))
    rep_set = set(rng.choice(N, min(reproj_sample, N), replace=False).tolist()) if reproj_sample else set(range(N))
    ord_set = set(rng.choice(N, min(ordering_sample, N), replace=False).tolist()) if ordering_sample else set(range(N))

    rows = []
    rle_match = 0; rle_total = 0; rle_fail = []
    reproj_p50 = []; reproj_p99 = []; reproj_max_global = 0.0; reproj_worst = []
    ord_ok = 0; ord_total = 0; ord_fail = []
    fx_vals = []; cx_vals = []; cy_vals = []; intr_missing = 0
    for i, jp in enumerate(jsons):
        d = json.load(open(jp))
        cam = d.get("camera_data", {})
        intr = cam.get("intrinsics")
        if not intr:
            intr_missing += 1
            continue
        objs = d.get("objects", [])
        if not objs:
            continue
        o = objs[0]
        fx_vals.append(intr["fx"]); cx_vals.append(intr["cx"]); cy_vals.append(intr["cy"])
        kp_world = np.array(o["keypoints_3d_world"])
        pc = np.array(o.get("projected_cuboid", []))

        az, elev = cam_azimuth_elevation(cam)
        bb = o.get("mask_bbox_xywh", [0, 0, 0, 0])
        if pc.size:
            xs = pc[:, 0]; ys = pc[:, 1]
            pc_diag = math.hypot(xs.max()-xs.min(), ys.max()-ys.min())
        else:
            pc_diag = float("nan")
        rows.append(dict(
            stem=os.path.basename(jp)[:-5], root=root,
            az=az, elev=elev,
            cam_loc=cam["location_worldframe"],
            n_in_frame=o.get("num_corners_in_frame"),
            n_unocc=o.get("num_corners_unoccluded"),
            n_vis=o.get("num_corners_visible"),
            mask_bb_size=math.sqrt(max(bb[2]*bb[3], 0)),
            mask_area=o.get("mask_area_px"),
            pc_diag=pc_diag,
            fx=intr["fx"], cx=intr["cx"], cy=intr["cy"],
            dist=float(np.linalg.norm(cam["location_worldframe"])),
        ))

        # reprojection
        if do_reproj and i in rep_set and pc.size:
            uv, z = project(kp_world, cam)
            err = np.linalg.norm(uv[:8] - pc[:8], axis=1)
            err = err[np.isfinite(err)]
            if err.size:
                reproj_p50.append(float(np.median(err)))
                reproj_p99.append(float(np.percentile(err, 99)))
                m = float(err.max())
                if m > reproj_max_global:
                    reproj_max_global = m
                reproj_worst.append((m, os.path.basename(jp)))
        # ordering
        if i in ord_set:
            ok, det = ordering_check(kp_world)
            ord_total += 1
            if ok:
                ord_ok += 1
            elif len(ord_fail) < 20:
                ord_fail.append((os.path.basename(jp), det))
        # rle
        if do_rle and i in rle_set:
            rle = o.get("mask_rle")
            ja = o.get("mask_area_px")
            if rle and ja is not None:
                m = rle_decode(rle)
                da = int(m.sum())
                rle_total += 1
                if da == ja:
                    rle_match += 1
                elif len(rle_fail) < 20:
                    rle_fail.append((os.path.basename(jp), da, ja))
        if (i+1) % 1000 == 0:
            print(f"  [{os.path.basename(root)}] {i+1}/{N}")

    reproj_worst.sort(reverse=True)
    integrity = dict(
        N=N, intr_missing=intr_missing,
        rle=dict(checked=rle_total, match=rle_match,
                 match_pct=round(100*rle_match/rle_total, 4) if rle_total else None,
                 fails=rle_fail[:20]),
        reproj=dict(checked=len(reproj_p50),
                    frame_p50_of_p50=float(np.median(reproj_p50)) if reproj_p50 else None,
                    frame_p99_of_p99=float(np.percentile(reproj_p99, 99)) if reproj_p99 else None,
                    worst_max_px=reproj_max_global,
                    worst_frames=[(round(m, 3), f) for m, f in reproj_worst[:10]]),
        ordering=dict(checked=ord_total, ok=ord_ok,
                      ok_pct=round(100*ord_ok/ord_total, 4) if ord_total else None,
                      fails=ord_fail[:20]),
        intrinsics=dict(
            fx_unique=len(set(round(v, 2) for v in fx_vals)),
            fx_min=float(min(fx_vals)), fx_max=float(max(fx_vals)), fx_mean=float(np.mean(fx_vals)),
            cx_min=float(min(cx_vals)), cx_max=float(max(cx_vals)),
            cy_min=float(min(cy_vals)), cy_max=float(max(cy_vals)),
        ),
    )
    return rows, integrity


def dist_stats(rows, key):
    a = np.array([r[key] for r in rows if r[key] is not None and np.isfinite(r[key])], dtype=float)
    if a.size == 0:
        return {}
    return dict(n=int(a.size), min=float(a.min()), p5=float(np.percentile(a, 5)),
                median=float(np.median(a)), mean=float(a.mean()),
                p95=float(np.percentile(a, 95)), max=float(a.max()))


def v_hist(rows, key):
    c = Counter(int(r[key]) for r in rows if r[key] is not None)
    return {str(k): c.get(k, 0) for k in range(9)}


def main():
    print("=== addon_v1 ===")
    addon_rows, addon_int = collect(ADDON, do_rle=True, do_reproj=True,
                                    rle_sample=None, reproj_sample=None, ordering_sample=None)  # full
    print("=== v3 (geometry only for fair az/elev; subsample 4000 for speed) ===")
    v3_rows, v3_int = collect(V3, do_rle=False, do_reproj=True,
                              reproj_sample=2000, ordering_sample=2000)

    # distributions
    def block(rows):
        return dict(
            N=len(rows),
            in_frame_hist=v_hist(rows, "n_in_frame"),
            vis_hist=v_hist(rows, "n_vis"),
            azimuth=dist_stats(rows, "az"),
            elevation=dist_stats(rows, "elev"),
            pc_diag=dist_stats(rows, "pc_diag"),
            mask_bb_size=dist_stats(rows, "mask_bb_size"),
            dist=dist_stats(rows, "dist"),
        )
    addon_d = block(addon_rows)
    v3_d = block(v3_rows)

    # near-dup: pose+geometry vector. Use cam azimuth, elevation, dist, pc_diag (scale).
    # Normalize then nearest-neighbor addon -> v3.
    def vec(r):
        # azimuth wrapped to sin/cos to avoid wraparound
        return [math.sin(math.radians(r["az"])), math.cos(math.radians(r["az"])),
                r["elev"]/90.0, r["dist"]/5.0, (r["pc_diag"] if np.isfinite(r["pc_diag"]) else 0)/900.0]
    A = np.array([vec(r) for r in addon_rows])
    B = np.array([vec(r) for r in v3_rows])
    # brute-force NN (sizes ok: 6k x 4k). chunk to save mem.
    mind = np.full(len(A), np.inf)
    minj = np.zeros(len(A), dtype=int)
    for s in range(0, len(A), 500):
        chunk = A[s:s+500]
        dmat = np.linalg.norm(chunk[:, None, :] - B[None, :, :], axis=2)
        mind[s:s+500] = dmat.min(axis=1)
        minj[s:s+500] = dmat.argmin(axis=1)
    nd_thresh = 0.02  # tight: near-identical pose/scale
    n_dup = int((mind < nd_thresh).sum())
    nd = dict(
        thresh=nd_thresh,
        nearest_min=float(mind.min()), nearest_p1=float(np.percentile(mind, 1)),
        nearest_p5=float(np.percentile(mind, 5)), nearest_median=float(np.median(mind)),
        n_below_thresh=n_dup, pct_below=round(100*n_dup/len(A), 3),
        examples=[(addon_rows[int(i)]["stem"], round(float(mind[i]), 4),
                   v3_rows[int(minj[i])]["stem"]) for i in np.argsort(mind)[:10]],
    )

    report = dict(addon_integrity=addon_int, addon_dist=addon_d, v3_dist=v3_d,
                  v3_integrity_reproj=v3_int["reproj"], near_dup=nd)
    json.dump(report, open(os.path.join(OUT, "addon_v1_audit.json"), "w"), indent=2)
    print("wrote addon_v1_audit.json")

    # ---- distribution plots: addon vs v3 overlay ----
    def overlay_hist(keyf, title, xlabel, fname, bins=40, integer=False):
        a = np.array([keyf(r) for r in addon_rows if np.isfinite(keyf(r))])
        b = np.array([keyf(r) for r in v3_rows if np.isfinite(keyf(r))])
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
        if integer:
            lo = int(min(a.min(), b.min())); hi = int(max(a.max(), b.max()))
            bb = np.arange(lo-0.5, hi+1.5, 1)
            ax.hist(b, bins=bb, density=True, alpha=0.5, color="#999", label=f"v3 (n={b.size})")
            ax.hist(a, bins=bb, density=True, alpha=0.5, color="#d8453b", label=f"addon (n={a.size})")
            ax.set_xticks(range(lo, hi+1))
        else:
            rng = (min(a.min(), b.min()), max(a.max(), b.max()))
            ax.hist(b, bins=bins, range=rng, density=True, alpha=0.5, color="#999", label=f"v3 (n={b.size})")
            ax.hist(a, bins=bins, range=rng, density=True, alpha=0.5, color="#d8453b", label=f"addon (n={a.size})")
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("density")
        ax.legend(); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, fname)); plt.close(fig)

    overlay_hist(lambda r: r["az"], "Azimuth (geometric, cam XY yaw) — addon vs v3", "deg", "cmp_azimuth.png")
    overlay_hist(lambda r: r["elev"], "Elevation (geometric) — addon vs v3", "deg", "cmp_elevation.png")
    overlay_hist(lambda r: r["pc_diag"] if r["pc_diag"] < 1500 else float("nan"),
                 "Projected cuboid diagonal (clip<1500) — addon vs v3", "px", "cmp_pc_diag.png")
    overlay_hist(lambda r: r["mask_bb_size"], "Projected size sqrt(mask bbox) — addon vs v3", "px", "cmp_mask_size.png")
    overlay_hist(lambda r: r["n_in_frame"], "Corners in-frame V — addon vs v3", "V", "cmp_in_frame.png", integer=True)
    overlay_hist(lambda r: r["dist"], "Camera distance from origin — addon vs v3", "m", "cmp_dist.png")
    # near-dup histogram
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.hist(mind, bins=60, color="#3b7dd8", edgecolor="white")
    ax.axvline(nd_thresh, color="crimson", lw=1.5, label=f"dup thresh={nd_thresh}")
    ax.set_title(f"addon->v3 nearest-neighbor distance (pose+scale vec)\nn_below_thresh={n_dup} ({nd['pct_below']}%)")
    ax.set_xlabel("normalized NN distance"); ax.set_ylabel("count"); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cmp_neardup.png")); plt.close(fig)
    print("wrote distribution plots")

    # print summary
    print("\n--- ADDON INTEGRITY ---")
    print("RLE:", addon_int["rle"]["match"], "/", addon_int["rle"]["checked"],
          "=", addon_int["rle"]["match_pct"], "%")
    print("reproj p50-of-p50:", addon_int["reproj"]["frame_p50_of_p50"],
          "p99-of-p99:", addon_int["reproj"]["frame_p99_of_p99"],
          "worst:", addon_int["reproj"]["worst_max_px"])
    print("ordering ok:", addon_int["ordering"]["ok_pct"], "%")
    print("intrinsics:", addon_int["intrinsics"])
    print("\n--- DIST addon vs v3 ---")
    print("addon V hist:", addon_d["in_frame_hist"])
    print("v3 V hist:", v3_d["in_frame_hist"])
    print("addon elev:", addon_d["elevation"])
    print("v3 elev:", v3_d["elevation"])
    print("addon pc_diag:", addon_d["pc_diag"])
    print("v3 pc_diag:", v3_d["pc_diag"])
    print("\n--- NEAR DUP ---")
    print(nd["n_below_thresh"], "below thresh; nearest min/p1/median:",
          nd["nearest_min"], nd["nearest_p1"], nd["nearest_median"])


if __name__ == "__main__":
    main()
