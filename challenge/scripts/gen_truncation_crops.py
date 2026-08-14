#!/usr/bin/env python3
"""Synthesize truncation (frame-edge clipping) samples by crop+resize.

For each NDDS (json+png) source frame we generate N variants. Each variant
picks a side/corner and crops the image so the pallet is clipped on that side
by a random fraction f (deep: U(0.35,0.55), shallow: U(0.10,0.30)). The crop
window keeps a 4:3 aspect ratio (no shear) and uses only real source pixels
(no synthetic fill). It is then uniform-resized to 640x480 and keypoints are
mapped into the new frame. Off-image keypoints after the transform are kept
as-is (truncation, negative/over-bound OK).

This script does crop+resize ONLY. Padding for training is handled later by
convert_to_yolo_pose.py --pad 100 and is intentionally not touched here.
"""
import argparse
import glob
import json
import os
import random

import cv2
import numpy as np
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
from challenge.data_paths import get as _dp  # 경로 단일 출처

W_OUT, H_OUT = 640, 480
ASPECT = W_OUT / H_OUT  # 4:3
SENTINEL = -1.0

# Cut-side sampling weights (forklift pans left/right -> mostly side clipping).
# Top clipping is unrealistic and almost excluded. Normalized at sample time.
CUT_WEIGHTS = {
    "L": 0.275, "R": 0.275,            # pure sides (main)
    "B": 0.15, "BL": 0.075, "BR": 0.075,  # bottom family (~0.30)
    "T": 0.02, "TL": 0.015, "TR": 0.015,  # top family (~0.05, almost excluded)
}

# Degenerate-rejection thresholds (applied after crop+resize).
MIN_IN_IMAGE = 5
MIN_VIS_AREA = W_OUT * H_OUT * 0.10   # 30720 px^2
MIN_VIS_DIM = 50.0                    # px (reject thin strips / lines)
EDGE_PAIRS = [  # cuboid wireframe (near face 0-3, far face 4-7)
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

DEFAULT_SRC_GLOBS = [
    # 2026-08-14 재편: capture*_manual_gt 가 두 구획으로 갈렸다.
    # noapril 은 정본 평가셋이라 eval_canonical/, 나머지는 manual_gt/.
    # 한 glob 으로 두 곳을 못 잡으므로 둘 다 적는다.
    "challenge/data/01_real/manual_gt/capture*_manual_gt",
    "challenge/data/01_real/eval_canonical/capture*_manual_gt",
    "data/outside/forklift_raw_20260528_163408/gt_manual",
]


def collect_sources(src_globs):
    """Return list of (json_path, png_path, src_tag, stem)."""
    out = []
    for base in src_globs:
        for jp in sorted(glob.glob(os.path.join(base, "*.json"))):
            pp = jp[:-5] + ".png"
            if not os.path.exists(pp):
                continue
            d = os.path.basename(os.path.dirname(jp))
            tag = d.replace("_manual_gt", "").replace("gt_manual", "forklift")
            if tag == "gt_manual":
                tag = "forklift"
            stem = os.path.splitext(os.path.basename(jp))[0]
            out.append((jp, pp, tag, stem))
    return out


def get_keypoints(obj):
    """9 kps = projected_cuboid[8] + centroid. Returns (9,2) float array."""
    kps = [list(map(float, p)) for p in obj["projected_cuboid"]]
    kps.append(list(map(float, obj["projected_cuboid_centroid"])))
    return np.array(kps, dtype=np.float64)


def is_sentinel(p):
    return p[0] == SENTINEL or p[1] == SENTINEL


def real_extent(kps):
    """Bounding box of non-sentinel kps (includes off-image coords)."""
    pts = np.array([p for p in kps if not is_sentinel(p)], dtype=np.float64)
    if len(pts) == 0:
        return None
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    return x0, y0, x1, y1


def make_crop_window(extent, img_w, img_h, side, f, rng):
    """Build a 4:3 crop window (within real image) that clips the pallet on
    `side` by fraction f. Returns (cx0, cy0, cw, ch) or None if infeasible.

    Coordinate convention: crop edges live in source-image pixels and are
    clamped to [0,img_w]x[0,img_h] (real pixels only, no fill).
    """
    px0, py0, px1, py1 = extent
    pw = max(px1 - px0, 1.0)
    ph = max(py1 - py0, 1.0)
    margin_x = pw * rng.uniform(0.05, 0.20)
    margin_y = ph * rng.uniform(0.05, 0.20)

    # Default loose window = pallet bbox + margin, clamped to image.
    L = px0 - margin_x
    R = px1 + margin_x
    T = py0 - margin_y
    B = py1 + margin_y

    # Apply the cut on chosen side(s): move that edge INTO the pallet by f.
    if "L" in side:
        L = px0 + f * pw
    if "R" in side:
        R = px1 - f * pw
    if "T" in side:
        T = py0 + f * ph
    if "B" in side:
        B = py1 - f * ph

    # Clamp to real image bounds.
    L = max(0.0, L)
    T = max(0.0, T)
    R = min(float(img_w), R)
    B = min(float(img_h), B)
    cw = R - L
    ch = B - T
    if cw < 20 or ch < 20:
        return None

    # Enforce 4:3 aspect by GROWING the shorter dimension (keeps the cut edge
    # intact; never shrinks below the requested clip so truncation is real).
    if cw / ch > ASPECT:
        # too wide -> grow height
        need_h = cw / ASPECT
        grow = need_h - ch
        # distribute growth away from cut edges where possible
        gt = grow * (0.0 if "T" in side else (1.0 if "B" in side else 0.5))
        gb = grow - gt
        T -= gt
        B += gb
    else:
        need_w = ch * ASPECT
        grow = need_w - cw
        gl = grow * (0.0 if "L" in side else (1.0 if "R" in side else 0.5))
        gr = grow - gl
        L -= gl
        R += gr

    # Re-clamp; if growth hit a border, shift the window to stay 4:3 inside img.
    if L < 0:
        R += -L
        L = 0.0
    if T < 0:
        B += -T
        T = 0.0
    if R > img_w:
        L -= (R - img_w)
        R = float(img_w)
    if B > img_h:
        T -= (B - img_h)
        B = float(img_h)
    L = max(0.0, L)
    T = max(0.0, T)
    R = min(float(img_w), R)
    B = min(float(img_h), B)

    cw = R - L
    ch = B - T
    if cw < 20 or ch < 20:
        return None
    return L, T, cw, ch


def transform_kps(kps, win):
    """Map kps into resized 640x480 frame. Sentinels preserved as -1."""
    cx0, cy0, cw, ch = win
    sx = W_OUT / cw
    sy = H_OUT / ch
    out = []
    for p in kps:
        if is_sentinel(p):
            out.append([SENTINEL, SENTINEL])
        else:
            out.append([(p[0] - cx0) * sx, (p[1] - cy0) * sy])
    return np.array(out, dtype=np.float64)


def count_in_image(kps):
    n = 0
    for p in kps:
        if is_sentinel(p):
            continue
        if 0 <= p[0] < W_OUT and 0 <= p[1] < H_OUT:
            n += 1
    return n


SIDES = list(CUT_WEIGHTS.keys())
SIDE_WEIGHTS = list(CUT_WEIGHTS.values())


def visible_bbox(kps):
    """Axis-aligned bbox of in-image (non-sentinel) kps. Returns (w,h,area)."""
    pts = [p for p in kps if not is_sentinel(p)
           and 0 <= p[0] < W_OUT and 0 <= p[1] < H_OUT]
    if not pts:
        return 0.0, 0.0, 0.0
    pts = np.asarray(pts, dtype=np.float64)
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()
    return w, h, w * h


def gen_variant(img, kps, deep, rng, max_tries):
    """Return (out_img, out_kps, side, f, in_count) or None.

    A variant is accepted only if, after crop+resize, it passes ALL
    degenerate-rejection checks (in-image>=MIN_IN_IMAGE, visible-bbox area
    >=MIN_VIS_AREA, min(width,height)>=MIN_VIS_DIM). Otherwise retry.
    """
    img_h, img_w = img.shape[:2]
    extent = real_extent(kps)
    if extent is None:
        return None
    for _ in range(max_tries):
        side = rng.choices(SIDES, weights=SIDE_WEIGHTS, k=1)[0]
        f = rng.uniform(0.35, 0.55) if deep else rng.uniform(0.10, 0.30)
        win = make_crop_window(extent, img_w, img_h, side, f, rng)
        if win is None:
            continue
        cx0, cy0, cw, ch = win
        crop = img[int(round(cy0)):int(round(cy0 + ch)),
                   int(round(cx0)):int(round(cx0 + cw))]
        if crop.size == 0:
            continue
        out_img = cv2.resize(crop, (W_OUT, H_OUT), interpolation=cv2.INTER_LINEAR)
        out_kps = transform_kps(kps, win)
        in_cnt = count_in_image(out_kps)
        if in_cnt < MIN_IN_IMAGE:
            continue
        vw, vh, varea = visible_bbox(out_kps)
        if varea < MIN_VIS_AREA:
            continue
        if min(vw, vh) < MIN_VIS_DIM:
            continue
        return out_img, out_kps, side, f, in_cnt
    return None


def write_output(out_dir, stem, img, kps, src_obj, src_cam):
    cv2.imwrite(os.path.join(out_dir, stem + ".png"), img)
    obj = {
        "class": "pallet",
        "name": src_obj.get("name", "real_pallet"),
        "visibility": src_obj.get("visibility", 1),
        "projected_cuboid": [[float(x), float(y)] for x, y in kps[:8]],
        "projected_cuboid_centroid": [float(kps[8][0]), float(kps[8][1])],
        "dimensions_m": src_obj.get("dimensions_m", {}),
    }
    if "pose_transform" in src_obj:
        obj["pose_transform"] = src_obj["pose_transform"]
    out = {
        "camera_data": {
            "width": W_OUT,
            "height": H_OUT,
            "intrinsics": src_cam.get("intrinsics", {}),
        },
        "objects": [obj],
    }
    with open(os.path.join(out_dir, stem + ".json"), "w") as fp:
        json.dump(out, fp, indent=2)


def verify(out_dir, records, rng, n=6):
    vdir = os.path.join(out_dir, "_verify")
    os.makedirs(vdir, exist_ok=True)
    pick = rng.sample(records, min(n, len(records)))
    print("\n=== VERIFY (random {}) ===".format(len(pick)))
    for rec in pick:
        stem, side, f, deep = rec["stem"], rec["side"], rec["f"], rec["deep"]
        img = cv2.imread(os.path.join(out_dir, stem + ".png"))
        d = json.load(open(os.path.join(out_dir, stem + ".json")))
        o = d["objects"][0]
        kps = [list(map(float, p)) for p in o["projected_cuboid"]]
        kps.append(list(map(float, o["projected_cuboid_centroid"])))
        kps = np.array(kps)

        # Canvas with margin so off-image points are visible.
        m = 120
        canvas = np.full((H_OUT + 2 * m, W_OUT + 2 * m, 3), 40, np.uint8)
        canvas[m:m + H_OUT, m:m + W_OUT] = img
        cv2.rectangle(canvas, (m, m), (m + W_OUT, m + H_OUT), (0, 255, 255), 2)

        def cv_pt(p):
            return int(round(p[0] + m)), int(round(p[1] + m))

        for a, b in EDGE_PAIRS:
            if is_sentinel(kps[a]) or is_sentinel(kps[b]):
                continue
            cv2.line(canvas, cv_pt(kps[a]), cv_pt(kps[b]), (0, 200, 0), 1)
        in_c = off_c = 0
        for i, p in enumerate(kps):
            if is_sentinel(p):
                continue
            inside = 0 <= p[0] < W_OUT and 0 <= p[1] < H_OUT
            in_c += inside
            off_c += (not inside)
            col = (0, 0, 255) if i == 8 else ((0, 255, 0) if inside else (255, 0, 255))
            cv2.circle(canvas, cv_pt(p), 4, col, -1)
            cv2.putText(canvas, str(i), (cv_pt(p)[0] + 4, cv_pt(p)[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        _, _, varea = visible_bbox(kps)
        area_pct = 100.0 * varea / (W_OUT * H_OUT)
        label = "{} cut={} f={:.2f} {} in={} off={} area={:.1f}%".format(
            stem, side, f, "DEEP" if deep else "shallow", in_c, off_c, area_pct)
        cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(vdir, stem + "_overlay.png"), canvas)
        print("  {}  cut={:2s} f={:.2f} {:7s} in={} off={} area={:.1f}%".format(
            stem, side, f, "deep" if deep else "shallow", in_c, off_c, area_pct))
    print("overlays -> {}".format(vdir))


def clear_out_dir(out_dir):
    """Remove previously generated crops + _verify, recreate empty dir."""
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            p = os.path.join(out_dir, name)
            if name == "_verify" and os.path.isdir(p):
                for f in os.listdir(p):
                    os.remove(os.path.join(p, f))
                os.rmdir(p)
            elif name.endswith((".png", ".json")):
                os.remove(p)
    os.makedirs(out_dir, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=_dp("derived.truncation_crops"))
    ap.add_argument("--src-glob", action="append", default=None,
                    help="source dir glob (repeatable). default=real capture set")
    ap.add_argument("--sample", type=int, default=0,
                    help="randomly sample N source frames (0=use all)")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--deep-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tries", type=int, default=40)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    clear_out_dir(args.out_dir)
    src_globs = args.src_glob if args.src_glob else DEFAULT_SRC_GLOBS
    sources = collect_sources(src_globs)
    print("source globs: {}".format(src_globs))
    print("sources: {} (json+png pairs)".format(len(sources)))
    if args.sample and args.sample < len(sources):
        sources = rng.sample(sources, args.sample)
        print("sampled: {} frames (seed={})".format(len(sources), args.seed))

    records = []
    n_deep = n_shallow = n_fail = 0
    side_hist = {}
    for jp, pp, tag, stem in sources:
        d = json.load(open(jp))
        cam = d.get("camera_data", {})
        objs = d.get("objects", [])
        if not objs:
            continue
        obj = objs[0]
        kps = get_keypoints(obj)
        img = cv2.imread(pp)
        if img is None:
            continue
        for vi in range(args.variants):
            deep = rng.random() < args.deep_ratio
            res = gen_variant(img, kps, deep, rng, args.max_tries)
            if res is None:
                n_fail += 1
                continue
            out_img, out_kps, side, f, in_cnt = res
            _, _, varea = visible_bbox(out_kps)
            uniq = "{}_{}_t{}".format(tag, stem, vi)
            write_output(args.out_dir, uniq, out_img, out_kps, obj, cam)
            records.append({"stem": uniq, "side": side, "f": f, "deep": deep,
                            "in": in_cnt, "area_pct": 100.0 * varea / (W_OUT * H_OUT)})
            side_hist[side] = side_hist.get(side, 0) + 1
            if deep:
                n_deep += 1
            else:
                n_shallow += 1

    print("\n=== SUMMARY ===")
    print("source frames        : {}".format(len(sources)))
    print("variants/frame        : {}".format(args.variants))
    print("generated             : {}".format(len(records)))
    print("  shallow / deep      : {} / {}  (deep_ratio={})".format(
        n_shallow, n_deep, args.deep_ratio))
    print("  rejected (degenerate/retry exh.) : {}".format(n_fail))
    print("cut-side distribution : {}".format(
        dict(sorted(side_hist.items()))))
    fam = {"side(L/R)": 0, "bottom(B/BL/BR)": 0, "top(T/TL/TR)": 0}
    for s, c in side_hist.items():
        if s in ("L", "R"):
            fam["side(L/R)"] += c
        elif s in ("B", "BL", "BR"):
            fam["bottom(B/BL/BR)"] += c
        else:
            fam["top(T/TL/TR)"] += c
    tot = max(sum(fam.values()), 1)
    print("cut-family            : " + ", ".join(
        "{} {} ({:.1f}%)".format(k, v, 100.0 * v / tot) for k, v in fam.items()))
    if records:
        ins = [r["in"] for r in records]
        areas = [r["area_pct"] for r in records]
        print("in-image kp/frame     : min={} median={} max={} (of 9)".format(
            min(ins), int(np.median(ins)), max(ins)))
        print("visible bbox area%%    : min=%.1f median=%.1f max=%.1f" % (
            min(areas), float(np.median(areas)), max(areas)))
    verify(args.out_dir, records, rng, n=6)


if __name__ == "__main__":
    main()
