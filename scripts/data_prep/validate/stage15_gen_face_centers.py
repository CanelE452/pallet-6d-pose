"""STAGE15 Gate 0 + Gate 1: side-face-center anchor target generation + audit.

NVIDIA-pallet-style anchor = visible SIDE FACE CENTER (vertical 4 faces only:
object-frame ±X / ±Y; top/bottom EXCLUDED). NOT the global centroid (STEP8).

Data: challenge/data/02_synthetic/training/v3/batch_000..009 + challenge/data/02_synthetic/training/addon_v1.
Convention: camera_dynamic_0123_v4. JSON has keypoints_3d_world (8 corners + centroid,
world), camera_data.intrinsics (per-frame K), projected_cuboid (8, 2D).

Face / corner-index grouping reused from existing convention
(pnp_solver.make_pallet_keypoints_3d_isaac + geo_loss_struct CUBOID_EDGES + the
audit_addon_v1 ordering_check which verifies index->face stability under the
camera-dynamic X-flip):

  front {0,1,2,3}  back {4,5,6,7}     (separated along object Y / +Y vs -Y side)
  left  {0,3,4,7}  right{1,2,5,6}     (separated along object X / +X vs -X side)
  top   {0,1,4,5}  bottom{2,3,6,7}    (object Z = world-up; HORIZONTAL -> NOT side)

  -> SIDE FACES = {front, back, left, right}  (4 vertical faces)

[확인] verified on v3 000000: rel-corner positions show front/back spread along
X(=1.1m)+Z(=0.12m height) on ±Y; left/right spread along Y(=1.3m)+Z on ±X;
top/bottom spread along X+Y with zero Z -> horizontal. So index groups == faces.

Each side-face center = mean of its 4 corners (3D world). Normal = computed from
the 4 corner 3D positions (cross of two edge vectors), oriented outward (away from
centroid). camera_facing: dot(normal_w, C_cam_world - F_w) > 0.
edge-on optional drop: |normalized dot(normal, view_dir)| < 0.15.

Projection convention REUSED from audit_addon_v1.py (addon EDA reproj p99 0.056px):
  Xc = Rwc^T (Xw - C); FLIP = diag(1,-1,-1) USD->OpenCV; uv = K Xc / z.

Gate 1-A: keypoints_3d_world -> project -> compare vs stored projected_cuboid.
  ordered & set-matched err: median/p95/p99/max + >1px frame count. PASS if both p99<1px.
Gate 1-B: target audit json + 30 overlays.

NO TRAINING. Outputs ->
  data/pallet/eval_results/stage15_face_center_offset/
    target_audit_face_centers.json
    convention_check.json
    face_center_overlay_samples/*.png
  + per-frame target written back as JSON sidecar:
    <dataset>/<stem>.face_centers.json   (NON-destructive sidecar, original JSON untouched)
"""
import json, glob, os, math, sys
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/minjae/Documents/github/pallet-pose"
V3 = os.path.join(ROOT, "challenge/data/02_synthetic/training/v3")
ADDON = os.path.join(ROOT, "challenge/data/02_synthetic/training/addon_v1")
OUT = os.path.join(ROOT, "data/pallet/eval_results/stage15_face_center_offset")
OVL = os.path.join(OUT, "face_center_overlay_samples")
os.makedirs(OVL, exist_ok=True)

# ── corner index -> face grouping (existing convention) ──
SIDE_FACES = {
    "front": [0, 1, 2, 3],   # +Y object side (vertical)
    "back":  [4, 5, 6, 7],   # -Y object side (vertical)
    "left":  [0, 3, 4, 7],   # +X object side (vertical)
    "right": [1, 2, 5, 6],   # -X object side (vertical)
}
FACE_ORDER = ["front", "back", "left", "right"]  # stable face_id 0..3
TOP_BOTTOM = {"top": [0, 1, 4, 5], "bottom": [2, 3, 6, 7]}  # EXCLUDED (horizontal)

EDGE_ON_THRESH = 0.15  # |cos angle(normal, view_dir)| below this = edge-on -> drop
DEPTH_EPS = 1e-3

FLIP = np.diag([1.0, -1.0, -1.0])  # USD -> OpenCV camera axes


def quat_to_R(q):  # xyzw
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def cam_extrinsics(cam):
    C = np.array(cam["location_worldframe"], dtype=float)
    Rwc = quat_to_R(cam["quaternion_xyzw_worldframe"])
    return C, Rwc


def K_of(cam):
    intr = cam["intrinsics"]
    return np.array([[intr["fx"], 0, intr["cx"]],
                     [0, intr["fy"], intr["cy"]],
                     [0, 0, 1.0]]), intr["resolution"]


def project_world(pts_world, K, C, Rwc):
    """pts_world (N,3) -> (N,2) px, depth (N,). REUSED audit convention."""
    pts = np.atleast_2d(np.asarray(pts_world, dtype=float))
    Xc = (Rwc.T @ (pts - C).T).T   # world -> USD cam
    Xc = (FLIP @ Xc.T).T           # USD -> OpenCV cam
    z = Xc[:, 2]
    uv = (K @ Xc.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def face_center_and_normal(kp_world, idx):
    """3D face center (4-corner mean) + outward normal (world), using actual 3D
    corner positions (robust to camera-dynamic index swaps within the group)."""
    ctr = kp_world[8]
    c4 = kp_world[idx]
    F = c4.mean(axis=0)
    # normal from two non-parallel in-face edges (corners are a planar quad)
    e1 = c4[1] - c4[0]
    e2 = c4[3] - c4[0] if not np.allclose(np.cross(c4[1]-c4[0], c4[2]-c4[0]), 0) else c4[2]-c4[0]
    n = np.cross(e1, e2)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        # degenerate; fall back to direction from centroid to face center
        n = F - ctr
        nn = np.linalg.norm(n)
    n = n / max(nn, 1e-12)
    # orient outward (away from cuboid centroid)
    if np.dot(n, F - ctr) < 0:
        n = -n
    return F, n


def list_jsons(root):
    if os.path.isdir(os.path.join(root, "batch_000")):
        out = []
        for b in sorted(glob.glob(os.path.join(root, "batch_*"))):
            out += sorted(glob.glob(os.path.join(b, "*.json")))
        return out
    out = sorted(glob.glob(os.path.join(root, "*.json")))
    return [j for j in out if not os.path.basename(j).startswith("_")
            and not j.endswith(".face_centers.json")]


def process(jp, write_sidecar=True):
    d = json.load(open(jp))
    cam = d.get("camera_data", {})
    intr = cam.get("intrinsics")
    objs = d.get("objects", [])
    if not intr or not objs:
        return None
    o = objs[0]
    kp_world = np.array(o["keypoints_3d_world"], dtype=float)
    if kp_world.shape[0] < 9:
        return None
    pc = np.array(o.get("projected_cuboid", []), dtype=float)
    K, res = K_of(cam)
    W, H = res[0], res[1]
    C, Rwc = cam_extrinsics(cam)

    # ---- Gate 1-A: keypoint reprojection (ordered + set-matched) ----
    reproj = None
    if pc.size and pc.shape[0] >= 8:
        uv, z = project_world(kp_world[:8], K, C, Rwc)
        ok = np.isfinite(uv).all(axis=1) & np.isfinite(pc[:8]).all(axis=1)
        if ok.sum() >= 2:
            ordered = np.linalg.norm(uv[ok] - pc[:8][ok], axis=1)
            # set-matched: greedy nearest assignment (8x8)
            D = np.linalg.norm(uv[ok][:, None, :] - pc[:8][ok][None, :, :], axis=2)
            used = set(); matched = []
            for i in np.argsort(D.min(axis=1)):
                row = D[i].copy()
                for u in used:
                    row[u] = np.inf
                j = int(np.argmin(row))
                used.add(j); matched.append(row[j])
            matched = np.array(matched)
            reproj = dict(ordered=ordered.tolist(), matched=matched.tolist())

    # ---- Gate 0: side face centers ----
    cam_F, _ = project_world(np.array([C]), K, C, Rwc)  # not used; keep C
    face_status = []
    F3d_list = []
    F2d_list = []
    for fid, name in enumerate(FACE_ORDER):
        idx = SIDE_FACES[name]
        F, n = face_center_and_normal(kp_world, idx)
        view = C - F                      # camera direction from face (world)
        view_n = view / max(np.linalg.norm(view), 1e-12)
        dot = float(np.dot(n, view_n))    # >0 => facing camera
        uvF, zF = project_world(np.array([F]), K, C, Rwc)
        u, v = float(uvF[0, 0]), float(uvF[0, 1])
        depth = float(zF[0])
        inside = bool(0 <= u < W and 0 <= v < H)
        facing = bool(dot > 0)
        edge_on = bool(abs(dot) < EDGE_ON_THRESH)
        valid = bool(facing and inside and depth > DEPTH_EPS and not edge_on)
        F3d_list.append([float(x) for x in F])
        F2d_list.append([u, v])
        face_status.append(dict(
            face_id=fid, face_name=name, xy=[u, v], depth_camera=depth,
            camera_facing=facing, inside_image=inside, edge_on=edge_on,
            heatmap_valid=valid, normal_dot_camera=dot))

    sidecar = dict(
        source=os.path.relpath(jp, ROOT),
        anchor_type="side_face_center",
        face_order=FACE_ORDER,
        face_centers_3d_world=F3d_list,
        face_centers_2d=F2d_list,
        face_center_status=face_status,
    )
    if write_sidecar:
        scp = jp[:-5] + ".face_centers.json"
        json.dump(sidecar, open(scp, "w"))

    n_valid = sum(1 for s in face_status if s["heatmap_valid"])
    V = o.get("num_corners_in_frame")
    return dict(jp=jp, face_status=face_status, n_valid=n_valid, reproj=reproj,
                V=V, F2d=F2d_list, kp_world=kp_world, cam=cam, pc=pc)


def pctl(a, p):
    return float(np.percentile(a, p)) if len(a) else None


def main():
    datasets = [("v3", V3), ("addon_v1", ADDON)]
    all_rows = []
    ordered_all = []; matched_all = []
    ordered_frame_max = []; matched_frame_max = []
    n_gt1px_ordered = 0; n_gt1px_matched = 0; n_reproj_frames = 0
    for dsname, root in datasets:
        jsons = list_jsons(root)
        print(f"[{dsname}] {len(jsons)} frames")
        for i, jp in enumerate(jsons):
            r = process(jp, write_sidecar=True)
            if r is None:
                continue
            r["ds"] = dsname
            all_rows.append(r)
            if r["reproj"]:
                o_e = np.array(r["reproj"]["ordered"])
                m_e = np.array(r["reproj"]["matched"])
                ordered_all.extend(o_e.tolist())
                matched_all.extend(m_e.tolist())
                ordered_frame_max.append(float(o_e.max()))
                matched_frame_max.append(float(m_e.max()))
                n_reproj_frames += 1
                if o_e.max() > 1.0:
                    n_gt1px_ordered += 1
                if m_e.max() > 1.0:
                    n_gt1px_matched += 1
            if (i + 1) % 2000 == 0:
                print(f"  {dsname} {i+1}/{len(jsons)}")

    # ── Gate 1-A: convention report ──
    oa = np.array(ordered_all); ma = np.array(matched_all)
    conv = dict(
        n_frames_checked=n_reproj_frames,
        ordered=dict(median=pctl(oa, 50), p95=pctl(oa, 95), p99=pctl(oa, 99),
                     max=float(oa.max()) if oa.size else None,
                     frames_gt1px=n_gt1px_ordered),
        set_matched=dict(median=pctl(ma, 50), p95=pctl(ma, 95), p99=pctl(ma, 99),
                         max=float(ma.max()) if ma.size else None,
                         frames_gt1px=n_gt1px_matched),
        pass_ordered_p99_lt1=bool(pctl(oa, 99) is not None and pctl(oa, 99) < 1.0),
        pass_matched_p99_lt1=bool(pctl(ma, 99) is not None and pctl(ma, 99) < 1.0),
    )
    conv["PASS"] = bool(conv["pass_ordered_p99_lt1"] and conv["pass_matched_p99_lt1"])
    json.dump(conv, open(os.path.join(OUT, "convention_check.json"), "w"), indent=2)

    # ── Gate 1-B: target audit ──
    valid_counts = Counter(r["n_valid"] for r in all_rows)
    nframes = len(all_rows)
    n_valid0 = sum(1 for r in all_rows if r["n_valid"] == 0)
    n_valid_ge3 = sum(1 for r in all_rows if r["n_valid"] >= 3)
    face_id_pass = Counter()
    inside_cnt = 0; facing_cnt = 0; total_faces = 0
    normal_dots = []
    for r in all_rows:
        for s in r["face_status"]:
            total_faces += 1
            if s["inside_image"]:
                inside_cnt += 1
            if s["camera_facing"]:
                facing_cnt += 1
            normal_dots.append(s["normal_dot_camera"])
            if s["heatmap_valid"]:
                face_id_pass[s["face_id"]] += 1
    # V=8 vs V<8 split valid-face-count
    v8 = [r["n_valid"] for r in all_rows if r["V"] == 8]
    vlt8 = [r["n_valid"] for r in all_rows if r["V"] is not None and r["V"] < 8]
    nd = np.array(normal_dots)
    audit = dict(
        n_frames=nframes,
        datasets={ds: sum(1 for r in all_rows if r["ds"] == ds) for ds, _ in datasets},
        valid_face_count_hist={str(k): valid_counts.get(k, 0) for k in range(5)},
        valid0_ratio=round(n_valid0 / nframes, 4),
        valid_ge3_ratio=round(n_valid_ge3 / nframes, 4),
        mean_valid_per_frame=round(float(np.mean([r["n_valid"] for r in all_rows])), 3),
        face_id_pass_distribution={FACE_ORDER[k]: face_id_pass.get(k, 0) for k in range(4)},
        inside_ratio=round(inside_cnt / total_faces, 4),
        camera_facing_ratio=round(facing_cnt / total_faces, 4),
        normal_dot=dict(min=float(nd.min()), p5=pctl(nd, 5), median=pctl(nd, 50),
                        p95=pctl(nd, 95), max=float(nd.max())),
        V8=dict(n=len(v8), mean_valid=round(float(np.mean(v8)), 3) if v8 else None,
                hist={str(k): int(np.sum(np.array(v8) == k)) for k in range(5)} if v8 else {}),
        Vlt8=dict(n=len(vlt8), mean_valid=round(float(np.mean(vlt8)), 3) if vlt8 else None,
                  hist={str(k): int(np.sum(np.array(vlt8) == k)) for k in range(5)} if vlt8 else {}),
    )
    json.dump(audit, open(os.path.join(OUT, "target_audit_face_centers.json"), "w"), indent=2)

    # ── 30 overlays ──
    rng = np.random.default_rng(0)
    # prefer frames with >=1 valid face, mix v3 + addon
    cand = [r for r in all_rows if r["n_valid"] >= 1]
    if len(cand) < 30:
        cand = all_rows
    sel = rng.choice(len(cand), min(30, len(cand)), replace=False)
    FACE_COL = {0: "#ff3b3b", 1: "#ffb000", 2: "#00c2ff", 3: "#34d058"}
    for k, si in enumerate(sel):
        r = cand[int(si)]
        png = r["jp"][:-5] + ".png"
        if not os.path.exists(png):
            continue
        img = plt.imread(png)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
        ax.imshow(img)
        pc = r["pc"]
        # projected cuboid corners + edges
        if pc.size and pc.shape[0] >= 8:
            from itertools import product
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for a, b in edges:
                ax.plot([pc[a,0],pc[b,0]],[pc[a,1],pc[b,1]], color="w", lw=0.8, alpha=0.7)
            ax.scatter(pc[:8,0], pc[:8,1], s=14, c="white", edgecolors="k", zorder=5)
            for ci in range(8):
                ax.text(pc[ci,0]+3, pc[ci,1]+3, str(ci), color="white", fontsize=7)
        # side face centers: filled=valid, hollow=invalid
        for s in r["face_status"]:
            u, v = s["xy"]
            col = FACE_COL[s["face_id"]]
            if s["heatmap_valid"]:
                ax.scatter([u],[v], s=180, marker="o", facecolors=col,
                           edgecolors="k", linewidths=1.5, zorder=10)
            else:
                ax.scatter([u],[v], s=120, marker="x", c=col, linewidths=2, zorder=10)
            tag = s["face_name"][0].upper()
            ax.text(u+5, v-6, f"{tag} d{s['normal_dot_camera']:+.2f}",
                    color=col, fontsize=8, weight="bold")
        ax.set_title(f"{r['ds']}/{os.path.basename(r['jp'])[:-5]}  "
                     f"V={r['V']} valid_faces={r['n_valid']}  (o=valid x=invalid)",
                     fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(OVL, f"overlay_{k:02d}_{r['ds']}_{os.path.basename(r['jp'])[:-5]}.png"))
        plt.close(fig)

    # ── console summary ──
    print("\n=== Gate 1-A: convention ===")
    print("ordered  p50/p95/p99/max:",
          round(conv["ordered"]["median"],4), round(conv["ordered"]["p95"],4),
          round(conv["ordered"]["p99"],4), round(conv["ordered"]["max"],4),
          " frames>1px:", conv["ordered"]["frames_gt1px"])
    print("matched  p50/p95/p99/max:",
          round(conv["set_matched"]["median"],4), round(conv["set_matched"]["p95"],4),
          round(conv["set_matched"]["p99"],4), round(conv["set_matched"]["max"],4),
          " frames>1px:", conv["set_matched"]["frames_gt1px"])
    print("PASS (both p99<1px):", conv["PASS"])
    print("\n=== Gate 1-B: audit ===")
    print("n_frames:", audit["n_frames"], audit["datasets"])
    print("valid_face_count_hist:", audit["valid_face_count_hist"])
    print("valid0_ratio:", audit["valid0_ratio"], " valid_ge3_ratio:", audit["valid_ge3_ratio"],
          " mean_valid/frame:", audit["mean_valid_per_frame"])
    print("face_id_pass:", audit["face_id_pass_distribution"])
    print("inside_ratio:", audit["inside_ratio"], " camera_facing_ratio:", audit["camera_facing_ratio"])
    print("normal_dot:", {k: (round(v,3) if v is not None else None) for k,v in audit["normal_dot"].items()})
    print("V8:", audit["V8"])
    print("Vlt8:", audit["Vlt8"])
    print("\noverlays ->", OVL)
    print("audit json ->", os.path.join(OUT, "target_audit_face_centers.json"))
    print("convention json ->", os.path.join(OUT, "convention_check.json"))


if __name__ == "__main__":
    main()
