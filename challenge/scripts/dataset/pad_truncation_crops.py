#!/usr/bin/env python3
"""Add reflect-padding to truncation-crop NDDS samples so off-image keypoints
move back INSIDE the frame, then resize the padded canvas back to 640x480.

Why DOPE needs this (analysis):
  The DOPE loader (CleanVisiiDopeLoader) builds belief maps with
  CreateBeliefMap(), which paints a keypoint's Gaussian ONLY if the point sits
  fully inside the map with a 2*sigma margin. Off-image keypoints (negative or
  >= size) are silently skipped -> a truncated corner produces an all-zero
  belief channel and gets NO supervision. YOLO-pose worked around this with
  convert_to_yolo_pose.py --pad 100 (expand canvas + shift keypoints). DOPE
  needs the same idea, but a FIXED pad=100 is insufficient: measured off-image
  extents on mixed_v8 crops have median 132px and p90 311px, so a fixed 100px
  pad would leave ~64% of truncated corners still outside -> dropped.

Approach (offline, mirrors YOLO --pad but DYNAMIC + resize-back):
  For each crop we compute the largest off-image distance over all
  non-sentinel keypoints and pad symmetrically by that amount + margin on all
  four sides (uniform on each axis -> no shear, geometry preserved). All
  truncated corners now sit inside the padded canvas. We then resize the padded
  canvas back to 640x480 so the loader receives its usual resolution and its
  RandomCrop(400,400)+Resize(50) pipeline behaves exactly as for normal frames.
  Keypoints are transformed exactly (shift then scale); pose_transform / cuboid
  (3D, camera-frame) are passed through untouched.

This is step 2 of the DOPE truncation pipeline:
  1. gen_truncation_crops.py  -> crop+resize 640x480 (off-image kps kept)
  2. pad_truncation_crops.py  -> pad off-image kps inside + resize back 640x480
"""
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import argparse
import glob
import json
import os

import cv2
import numpy as np

W_OUT, H_OUT = 640, 480
SENTINEL = -1.0
# Keep padded corners at least this fraction of each dim away from the border.
# CreateBeliefMap only paints a Gaussian when a kp sits >= 2*sigma inside the
# belief map. At output_size=50, sigma=4 -> 2*sigma/50 = 0.16, so corners must
# be >=16% inside each edge of the 640x480 frame to survive the resize-to-50.
# We pad a bit beyond that (0.20) so RandomCrop also has room to keep them.
MARGIN_FRAC = 0.20
EDGE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


import keypoint_annotations_transform as kat  # noqa: E402


def is_sentinel(p):
    return p[0] == SENTINEL or p[1] == SENTINEL


def get_kps(obj):
    kps = [list(map(float, p)) for p in obj["projected_cuboid"]]
    kps.append(list(map(float, obj["projected_cuboid_centroid"])))
    return np.array(kps, dtype=np.float64)


def required_pad(kps):
    """Smallest symmetric pad P (px, source space) such that after pad+resize
    back to 640x480 every non-sentinel kp lands within the margin band
    [MARGIN_FRAC, 1-MARGIN_FRAC] on both axes.

    Final coord on x: fx = (x + P) * W / (W + 2P). Require mx <= fx <= W - mx
    with mx = MARGIN_FRAC * W. Per axis this gives two lower bounds on P:
      from left  edge (smallest x):  P >= (m*W' ... )  -> derived below
    We solve numerically with a small ascending search (cheap, exact enough).
    """
    pts = np.array([p for p in kps if not is_sentinel(p)], dtype=np.float64)
    if len(pts) == 0:
        return 0
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    mx, my = MARGIN_FRAC * W_OUT, MARGIN_FRAC * H_OUT

    def fits(P):
        dw, dh = W_OUT + 2 * P, H_OUT + 2 * P
        sx, sy = W_OUT / dw, H_OUT / dh
        fxmin = (xmin + P) * sx
        fxmax = (xmax + P) * sx
        fymin = (ymin + P) * sy
        fymax = (ymax + P) * sy
        return (fxmin >= mx and fxmax <= W_OUT - mx
                and fymin >= my and fymax <= H_OUT - my)

    if fits(0):
        return 0
    P = 1
    while not fits(P) and P < 5000:
        P += max(1, P // 8)  # geometric-ish ascent for speed
    # refine down to the minimum within the last step
    lo = max(0, P - max(1, P // 8))
    for q in range(lo, P + 1):
        if fits(q):
            return q
    return P


def pad_and_resize(img, kps, pad_mode):
    """Pad symmetrically so all kps fit the margin band, resize back to 640x480.

    Returns (out_img, out_kps[(9,2)], pad_px).
    """
    pad = required_pad(kps)
    if pad <= 0:
        return img.copy(), kps.copy(), 0

    border = {
        "reflect": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
        "black": cv2.BORDER_CONSTANT,
    }.get(pad_mode, cv2.BORDER_REFLECT_101)
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, border, value=(0, 0, 0))
    ph, pw = padded.shape[:2]
    out_img = cv2.resize(padded, (W_OUT, H_OUT), interpolation=cv2.INTER_LINEAR)

    sx = W_OUT / pw
    sy = H_OUT / ph
    out = []
    for p in kps:
        if is_sentinel(p):
            out.append([SENTINEL, SENTINEL])
        else:
            out.append([(p[0] + pad) * sx, (p[1] + pad) * sy])
    return out_img, np.array(out, dtype=np.float64), pad


def write_output(out_dir, stem, img, kps, src_obj, src_cam, pad=None,
                 src_size=None, parent_frame=None):
    cv2.imwrite(os.path.join(out_dir, stem + ".png"), img)
    obj = {
        "class": "pallet",
        "name": src_obj.get("name", "real_pallet"),
        "visibility": src_obj.get("visibility", 1),
        "projected_cuboid": [[float(x), float(y)] for x, y in kps[:8]],
        "projected_cuboid_centroid": [float(kps[8][0]), float(kps[8][1])],
        "dimensions_m": src_obj.get("dimensions_m", {}),
    }
    # Pass through 3D / pose fields untouched (camera-frame, padding-invariant).
    for k in ("pose_transform", "cuboid", "location", "quaternion_xyzw"):
        if k in src_obj:
            obj[k] = src_obj[k]
    # 정본 keypoint 필드도 같은 pad+resize affine 으로 옮긴다.
    if pad is not None and src_size is not None:
        sw, sh = src_size
        sx, sy = W_OUT / (sw + 2 * pad), H_OUT / (sh + 2 * pad)
        kat.attach(obj, src_obj,
                   [[sx, 0.0, pad * sx], [0.0, sy, pad * sy]], W_OUT, H_OUT,
                   parent_frame=parent_frame,
                   transformation={"kind": "pad_resize", "pad_px": pad,
                                   "scale": [sx, sy], "out_size": [W_OUT, H_OUT]})
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


def clear_out_dir(out_dir):
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


def verify(out_dir, stems, n=8):
    vdir = os.path.join(out_dir, "_verify")
    os.makedirs(vdir, exist_ok=True)
    rng = np.random.default_rng(0)
    pick = list(stems) if len(stems) <= n else [stems[i] for i in
                                                rng.choice(len(stems), n, replace=False)]
    print("\n=== VERIFY (padded, {}) ===".format(len(pick)))
    for stem in pick:
        img = cv2.imread(os.path.join(out_dir, stem + ".png"))
        d = json.load(open(os.path.join(out_dir, stem + ".json")))
        o = d["objects"][0]
        kps = get_kps(o)
        m = 60
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
        cv2.putText(canvas, "{} in={} off={}".format(stem, in_c, off_c),
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(vdir, stem + "_padded.png"), canvas)
        print("  {}  in={} off={}".format(stem, in_c, off_c))
    print("overlays -> {}".format(vdir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="crop dir from gen_truncation_crops.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pad-mode", default="reflect",
                    choices=["reflect", "replicate", "black"])
    ap.add_argument("--verify", type=int, default=8)
    args = ap.parse_args()

    clear_out_dir(args.out_dir)
    jsons = sorted(glob.glob(os.path.join(args.in_dir, "*.json")))
    stems = []
    pads = []
    still_off = 0
    for jp in jsons:
        pp = jp[:-5] + ".png"
        if not os.path.exists(pp):
            continue
        d = json.load(open(jp))
        cam = d.get("camera_data", {})
        objs = d.get("objects", [])
        if not objs:
            continue
        obj = objs[0]
        kps = get_kps(obj)
        img = cv2.imread(pp)
        if img is None:
            continue
        out_img, out_kps, pad = pad_and_resize(img, kps, args.pad_mode)
        for p in out_kps:
            if not is_sentinel(p) and not (0 <= p[0] < W_OUT and 0 <= p[1] < H_OUT):
                still_off += 1
        stem = os.path.splitext(os.path.basename(jp))[0]
        write_output(args.out_dir, stem, out_img, out_kps, obj, cam,
                     pad=pad, src_size=(img.shape[1], img.shape[0]),
                     parent_frame=jp)
        stems.append(stem)
        pads.append(pad)

    print("=== PAD SUMMARY ===")
    print("input crops      : {}".format(len(jsons)))
    print("written          : {}".format(len(stems)))
    if pads:
        pa = np.array(pads)
        print("pad px           : min={} median={} mean={:.0f} max={}".format(
            pa.min(), int(np.median(pa)), pa.mean(), pa.max()))
    print("kps still off-img: {}  (should be 0)".format(still_off))
    print("out_dir          : {}".format(args.out_dir))
    if args.verify > 0 and stems:
        verify(args.out_dir, stems, n=args.verify)


if __name__ == "__main__":
    main()
