"""paper_dataset_validator — Paper-safe (RGB + mask_rle) dataset integrity + geometry audit.

Runs on a generated dataset root BEFORE it is used for Paper-S1 heatmap+mask-aux
training. Verifies that the (Windows/FoundationPose-generated) frames are geometrically
consistent with the camera_dynamic_0123_v4 convention and that mask_rle is intact.

★ Reuse (do NOT reinvent):
  - projection convention from scripts/data_prep/validate/audit_addon_v1.py
      FLIP = diag(1,-1,-1) (USD->OpenCV);  Xc = Rwc^T (Xw - C);  reproj p50~0.01px.
  - V_geom / elevation / frsep / mask-area logic from scripts/stage0/stage16_trunc_addon_audit.py
  - keypoint overlay (front 0-3 / rear 4-7 color, sentinel [-1,-1] + nan skip, no phantom
      diagonals) from scripts/stage0/stage22_myannot_individual_overlays.py

Convention (camera_dynamic_0123_v4):
  0-3 = front face (near), 4-7 = rear face (far), 8 = centroid.
  top    = {0,1,4,5},  bottom = {2,3,6,7}
  front  = {0,1,2,3},  rear   = {4,5,6,7}
  depth (near->far) edges = (0,4)(1,5)(2,6)(3,7).

Checks
  1. presence      : RGB + JSON exist; mask_rle field present.
  2. mask_rle      : COCO-style uncompressed RLE decodes; decoded area == json mask_area_px.
  3. convention    : projected_cuboid round-trip reproj (per-frame K) p99 < 1px;
                     object-frame face grouping (top/bot, front/rear) invariants hold.
  4. clamp         : off-screen corners kept ALIVE (real coords beyond frame) and NOT
                     clamped to the image border — truncation training needs live coords.
  5. border-suppress: near-border in-frame corners are not silently set to sentinel.
  6. distributions : V_geom (num_corners_in_frame) + heatmap_valid (in-frame count) hist.
  7. overlays      : 20 sample overlays (front cyan / rear orange, GT cuboid) for eye-check.

Outputs -> {root}/audit_report.json, {root}/bad_frames.txt, {root}/overlays/*.jpg

Run (validator itself; execution needs a real dataset):
  python scripts/data_prep/validate/paper_dataset_validator.py --root <dataset_root>
  # smoke on addon_v1 (same JSON schema; addon is NOT paper-safe, format-check only):
  python scripts/data_prep/validate/paper_dataset_validator.py \
      --root challenge/data/02_synthetic/training/addon_v1 --max_frames 300 --n_overlays 20
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os

import numpy as np

try:
    import cv2
except Exception:  # overlays skipped if cv2 missing
    cv2 = None

# ----------------------------------------------------------------------------
# convention constants
FRONT = [0, 1, 2, 3]
REAR = [4, 5, 6, 7]
TOP = [0, 1, 4, 5]
BOT = [2, 3, 6, 7]
DEPTH_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),      # front loop
         (4, 5), (5, 6), (6, 7), (7, 4),      # rear loop
         (0, 4), (1, 5), (2, 6), (3, 7)]      # near->far connectors
FRONT_TXT = (255, 255, 0)   # cyan
REAR_TXT = (0, 200, 255)    # orange
GT_COL = (0, 255, 0)        # green cuboid

FLIP = np.diag([1.0, -1.0, -1.0])  # USD -> OpenCV camera axes (audit_addon_v1)


# ----------------------------------------------------------------------------
# reused math (audit_addon_v1.py / stage16_trunc_addon_audit.py)
def quat_to_R(q):  # xyzw
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])


def K_of(cam):
    intr = cam["intrinsics"]
    return np.array([[intr["fx"], 0, intr["cx"]],
                     [0, intr["fy"], intr["cy"]],
                     [0, 0, 1.0]])


def project(kp_world, cam):
    """(N,3) world coords -> (N,2) pixels; per-frame K + extrinsics. audit_addon_v1 convention."""
    K = K_of(cam)
    C = np.array(cam["location_worldframe"])
    Rwc = quat_to_R(cam["quaternion_xyzw_worldframe"])
    Xc = (Rwc.T @ (kp_world - C).T).T   # world -> cam (USD)
    Xc = (FLIP @ Xc.T).T                # USD -> OpenCV
    uv = (K @ Xc.T).T
    z = Xc[:, 2]
    uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def rle_decode(rle):
    """COCO uncompressed RLE (counts toggling 0/1, column-major). audit_addon_v1."""
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        return None  # compressed RLE (needs pycocotools) -> flagged upstream
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        flat[idx:idx + c] = val
        idx += c
        val ^= 1
    return flat.reshape((w, h)).T


def elev_from_world(d):
    """arctan2(dz, horiz(cam-obj)); edge-on~0deg, top-down~90deg. stage16."""
    cam = np.array(d["camera_data"]["location_worldframe"], float)
    obj = np.array(d["objects"][0]["location"], float)
    v = cam - obj
    horiz = np.linalg.norm(v[:2])
    return float(np.degrees(np.arctan2(v[2], horiz)))


def frsep(pc):
    """median 2D length of the 4 depth (near->far) edges from projected_cuboid. stage16."""
    g = np.array(pc, float)
    try:
        ds = [np.linalg.norm(g[a] - g[b]) for a, b in DEPTH_EDGES]
        ds = [x for x in ds if np.isfinite(x)]
        return float(np.median(ds)) if ds else None
    except Exception:
        return None


# ----------------------------------------------------------------------------
def valid_mask(pts):
    """valid corner = not nan and not sentinel (-1,-1). stage22 overlay logic."""
    x, y = pts[:, 0], pts[:, 1]
    sentinel = (x == -1.0) & (y == -1.0)
    return ~(np.isnan(x) | sentinel)


def is_sentinel(p):
    return float(p[0]) == -1.0 and float(p[1]) == -1.0


def ordering_check(kp_world):
    """camera_dynamic_0123_v4 object-frame face-grouping invariants (audit_addon_v1)."""
    ctr = kp_world[8]
    rel = kp_world[:8] - ctr
    z = rel[:, 2]
    top = z[[0, 1, 4, 5]].mean()
    bot = z[[2, 3, 6, 7]].mean()
    ok_z = top > bot + 0.03
    y = rel[:, 1]
    x = rel[:, 0]
    sep = max(abs(y[FRONT].mean() - y[REAR].mean()),
              abs(x[FRONT].mean() - x[REAR].mean()))
    ok_fb = sep > 0.4
    return bool(ok_z and ok_fb), dict(top_z=float(top), bot_z=float(bot), sep=float(sep))


def _is_frame_json(p):
    """frame json = '{stem}.json' (single dot). Skip sidecars like '000000.face_centers.json'
    and '_'-prefixed meta (e.g. _summary.json)."""
    b = os.path.basename(p)
    return b.count(".") == 1 and not b.startswith("_")


def list_jsons(root):
    if os.path.isdir(os.path.join(root, "batch_000")):
        out = []
        for b in sorted(glob.glob(os.path.join(root, "batch_*"))):
            out += sorted(glob.glob(os.path.join(b, "*.json")))
    else:
        out = sorted(glob.glob(os.path.join(root, "*.json")))
    return [p for p in out if _is_frame_json(p)]


def find_image(jp):
    stem = jp[:-5]
    for ext in (".png", ".jpg", ".jpeg"):
        if os.path.exists(stem + ext):
            return stem + ext
    return None


def draw_overlay(img, gt8, gtc):
    """front cyan / rear orange, skip sentinel/nan edges (stage22)."""
    pts = np.asarray(gt8, float)
    ok = valid_mask(pts)
    for a, b in EDGES:
        if ok[a] and ok[b]:
            cv2.line(img, (int(pts[a, 0]), int(pts[a, 1])),
                     (int(pts[b, 0]), int(pts[b, 1])), GT_COL, 1, cv2.LINE_AA)
    for i in range(8):
        if not ok[i]:
            continue
        p = (int(pts[i, 0]), int(pts[i, 1]))
        cv2.circle(img, p, 4, GT_COL, 2, cv2.LINE_AA)
        tc = FRONT_TXT if i < 4 else REAR_TXT
        cv2.putText(img, str(i), (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, tc, 1, cv2.LINE_AA)
    if gtc is not None and not is_sentinel(gtc) and np.isfinite(gtc[0]):
        c = (int(gtc[0]), int(gtc[1]))
        cv2.drawMarker(img, c, GT_COL, cv2.MARKER_CROSS, 10, 2)


# ----------------------------------------------------------------------------
def validate(root, max_frames, n_overlays, border_px, reproj_p99_tol):
    jsons = list_jsons(root)
    jsons = [j for j in jsons if not os.path.basename(j).startswith("_")]
    if max_frames:
        jsons = jsons[:max_frames]
    N = len(jsons)
    if N == 0:
        raise SystemExit(f"[FATAL] no JSON under {root}")

    out_overlays = os.path.join(root, "overlays")
    os.makedirs(out_overlays, exist_ok=True)

    rep = {
        "root": root, "n_frames": N,
        "convention": "camera_dynamic_0123_v4",
        "reproj_p99_tol_px": reproj_p99_tol, "border_px": border_px,
    }
    bad_frames = []          # (stem, reason)
    n_no_img = n_no_mask = n_mask_str = 0
    mask_checked = mask_match = 0
    mask_fail = []
    reproj_p50 = []
    reproj_p99 = []
    reproj_worst = []        # (max_px, stem)
    ord_ok = 0
    ord_fail = []
    vgeom_hist = {k: 0 for k in range(10)}
    hmvalid_hist = {k: 0 for k in range(10)}
    clamp_suspect = 0        # off-screen corner stored ON border (clamped) instead of alive
    offscreen_alive = 0      # off-screen corner stored with real off-screen coords (good)
    border_suppress = 0      # in-frame near-border corner stored as sentinel (bad)
    n_overlaid = 0

    for jp in jsons:
        d = json.load(open(jp))
        stem = os.path.basename(jp)[:-5]
        cam = d.get("camera_data", {})
        objs = d.get("objects", [])
        if not cam.get("intrinsics") or not objs:
            bad_frames.append((stem, "no intrinsics/objects"))
            continue
        o = objs[0]

        # (1) presence
        ip = find_image(jp)
        if ip is None:
            n_no_img += 1
            bad_frames.append((stem, "missing RGB"))
        img_wh = (cam.get("width"), cam.get("height"))

        # (2) mask_rle
        rle = o.get("mask_rle")
        ja = o.get("mask_area_px")
        if rle is None:
            n_no_mask += 1
        elif isinstance(rle.get("counts"), str):
            n_mask_str += 1
            bad_frames.append((stem, "mask_rle counts is str (compressed; needs pycocotools)"))
        elif ja is not None:
            m = rle_decode(rle)
            if m is not None:
                mask_checked += 1
                da = int(m.sum())
                if da == ja:
                    mask_match += 1
                else:
                    if len(mask_fail) < 30:
                        mask_fail.append([stem, da, ja])
                    bad_frames.append((stem, f"mask area {da}!=json {ja}"))

        # geometry
        kp_world = np.array(o.get("keypoints_3d_world", o.get("cuboid", [])))
        pc = np.array(o.get("projected_cuboid", []), float)
        W = img_wh[0] or (cam["intrinsics"]["cx"] * 2)
        H = img_wh[1] or (cam["intrinsics"]["cy"] * 2)

        # (3) convention: reproj round-trip + face grouping
        if kp_world.shape[0] >= 8 and pc.size:
            uv, z = project(kp_world, cam)
            vm = valid_mask(pc[:8])
            err = np.linalg.norm(uv[:8] - pc[:8], axis=1)
            err = err[vm & np.isfinite(err)]
            if err.size:
                reproj_p50.append(float(np.median(err)))
                p99 = float(np.percentile(err, 99))
                reproj_p99.append(p99)
                mx = float(err.max())
                reproj_worst.append((mx, stem))
                if p99 >= reproj_p99_tol:
                    bad_frames.append((stem, f"reproj p99 {p99:.2f}px >= {reproj_p99_tol}"))
            ok, det = ordering_check(kp_world)
            if ok:
                ord_ok += 1
            else:
                if len(ord_fail) < 30:
                    ord_fail.append([stem, det])
                bad_frames.append((stem, f"convention grouping fail {det}"))

            # (4/5) clamp + border-suppress: compare recomputed projection vs stored pc
            for i in range(8):
                u, v = uv[i]
                inb = (0 <= u < W) and (0 <= v < H)
                stored = pc[i]
                stored_sent = is_sentinel(stored)
                near_border = inb and (u < border_px or u > W - border_px or
                                       v < border_px or v > H - border_px)
                if not inb:
                    # corner projects off-screen: should be stored ALIVE (real off coords)
                    if stored_sent:
                        pass  # null-annotated: acceptable policy, not clamp
                    elif (0 <= stored[0] < W) and (0 <= stored[1] < H):
                        clamp_suspect += 1  # off-screen but stored inside -> clamped
                    else:
                        offscreen_alive += 1
                elif near_border and stored_sent:
                    border_suppress += 1

        # distributions
        v_geom = o.get("num_corners_in_frame")
        if v_geom is None and pc.size:
            v_geom = int(sum(1 for i in range(8)
                             if valid_mask(pc[:8])[i]))
        if v_geom is not None and 0 <= v_geom <= 9:
            vgeom_hist[v_geom] += 1
        # heatmap_valid = # keypoints (0..8) that land in-frame (would seed a belief peak)
        if pc.size:
            uvp = pc
            hv = 0
            for i in range(min(9, uvp.shape[0])):
                p = uvp[i]
                if not is_sentinel(p) and np.isfinite(p[0]) and 0 <= p[0] < W and 0 <= p[1] < H:
                    hv += 1
            hmvalid_hist[hv] += 1

        # (7) overlays
        if cv2 is not None and ip and n_overlaid < n_overlays and pc.size:
            im = cv2.imread(ip)
            if im is not None:
                gt8 = pc[:8]
                gtc = pc[8] if pc.shape[0] > 8 else None
                draw_overlay(im, gt8, gtc)
                e = elev_from_world(d) if kp_world.shape[0] >= 8 else float("nan")
                hdr = f"{stem} V={v_geom} elev={e:.1f} frsep={frsep(pc)}"
                cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
                cv2.putText(im, hdr, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(out_overlays, f"{n_overlaid:02d}_{stem}.jpg"), im)
                n_overlaid += 1

    reproj_worst.sort(reverse=True)
    rep["presence"] = dict(n=N, missing_rgb=n_no_img,
                           missing_mask_rle=n_no_mask, compressed_rle=n_mask_str)
    rep["mask_rle"] = dict(
        checked=mask_checked, match=mask_match,
        match_pct=round(100 * mask_match / mask_checked, 3) if mask_checked else None,
        fails=mask_fail)
    rep["convention"] = dict(
        reproj_checked=len(reproj_p50),
        reproj_p50_of_p50=float(np.median(reproj_p50)) if reproj_p50 else None,
        reproj_p99_of_p99=float(np.percentile(reproj_p99, 99)) if reproj_p99 else None,
        reproj_worst=[[round(m, 3), s] for m, s in reproj_worst[:10]],
        ordering_ok=ord_ok,
        ordering_ok_pct=round(100 * ord_ok / len(reproj_p50), 3) if reproj_p50 else None,
        ordering_fails=ord_fail)
    rep["clamp"] = dict(
        offscreen_corners_alive=offscreen_alive,
        clamp_suspect=clamp_suspect,
        border_suppress=border_suppress,
        note="clamp_suspect>0 => off-screen corners stored inside frame (clamped: BAD for "
             "truncation). offscreen_alive good. border_suppress => near-border in-frame "
             "corner set to sentinel (silent suppression: BAD).")
    rep["distributions"] = dict(
        V_geom_hist={str(k): v for k, v in vgeom_hist.items() if v},
        heatmap_valid_hist={str(k): v for k, v in hmvalid_hist.items() if v})
    rep["n_overlays"] = n_overlaid
    rep["n_bad_frames"] = len(bad_frames)

    # verdict
    problems = []
    if n_no_img:
        problems.append(f"{n_no_img} missing RGB")
    if n_mask_str:
        problems.append(f"{n_mask_str} compressed RLE (no pycocotools decode)")
    if mask_checked and mask_match != mask_checked:
        problems.append(f"mask area mismatch {mask_checked - mask_match}/{mask_checked}")
    if reproj_p99 and np.percentile(reproj_p99, 99) >= reproj_p99_tol:
        problems.append(f"reproj p99-of-p99 >= {reproj_p99_tol}px")
    if clamp_suspect:
        problems.append(f"{clamp_suspect} clamped off-screen corners")
    if border_suppress:
        problems.append(f"{border_suppress} border-suppressed corners")
    rep["verdict"] = "PASS" if not problems else "FAIL"
    rep["problems"] = problems

    json.dump(rep, open(os.path.join(root, "audit_report.json"), "w"), indent=2)
    with open(os.path.join(root, "bad_frames.txt"), "w") as f:
        for stem, reason in bad_frames:
            f.write(f"{stem}\t{reason}\n")

    print(f"[validator] {root}")
    print(f"  frames={N}  verdict={rep['verdict']}")
    print(f"  RGB missing={n_no_img}  mask_rle missing={n_no_mask}  compressed={n_mask_str}")
    print(f"  mask area match={rep['mask_rle']['match_pct']}%  ({mask_match}/{mask_checked})")
    print(f"  reproj p50-of-p50={rep['convention']['reproj_p50_of_p50']}  "
          f"p99-of-p99={rep['convention']['reproj_p99_of_p99']}")
    print(f"  ordering ok={rep['convention']['ordering_ok_pct']}%")
    print(f"  clamp_suspect={clamp_suspect} offscreen_alive={offscreen_alive} "
          f"border_suppress={border_suppress}")
    print(f"  V_geom={rep['distributions']['V_geom_hist']}")
    print(f"  heatmap_valid={rep['distributions']['heatmap_valid_hist']}")
    print(f"  overlays={n_overlaid} -> {out_overlays}")
    print(f"  bad_frames={len(bad_frames)} -> {os.path.join(root, 'bad_frames.txt')}")
    if problems:
        print(f"  PROBLEMS: {problems}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="generated dataset root (flat or batch_*)")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all")
    ap.add_argument("--n_overlays", type=int, default=20)
    ap.add_argument("--border_px", type=int, default=8,
                    help="in-frame corner closer than this to an edge = near-border")
    ap.add_argument("--reproj_p99_tol", type=float, default=1.0,
                    help="convention round-trip reproj p99 must be < this (px)")
    args = ap.parse_args()
    validate(args.root, args.max_frames, args.n_overlays, args.border_px, args.reproj_p99_tol)


if __name__ == "__main__":
    main()
