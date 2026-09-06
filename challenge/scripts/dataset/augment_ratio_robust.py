#!/usr/bin/env python3
"""Ratio-robust + truncation augmentation for paper_base DOPE training.

Source: mixed_v8_train (9000 NDDS frames, camera-facing 0123 convention; the
`.json` keypoints are already v4-converted). For each picked source frame we
emit augmented NDDS (png+json) samples of three kinds:

  squash : independent horizontal/vertical scale (aspect-ratio warp). Covers
           unseen pallets with different W:D:H aspect ratios -- the paper's
           generalization claim. A 2x2 affine A=diag(sx,sy) about the object
           centroid is applied to BOTH the image (warpAffine) and the 9
           keypoints, so image<->keypoint stay locked.
  scale  : isotropic zoom (sx==sy). Object-size variation.
  trunc  : L/R-biased frame-edge clipping (memory truncation-side-cut-bias:
           top clipping is unrealistic + degenerate, so it is nearly excluded).
           Off-image corners are kept and then reflect-padded back inside so
           DOPE's CreateBeliefMap can still supervise them (memory
           dope_truncation_pad_pipeline: belief is only painted when a kp sits
           >=2*sigma inside the 50-map -> MARGIN_FRAC=0.20).

Convention guarantees:
  * camera-facing 0123 order is preserved. squash/scale/crop are
    orientation-preserving (positive scale, no reflection, no transpose), so
    they never permute the keypoint indices. (A horizontal FLIP would swap
    L<->R and is intentionally NOT used here.)
  * 3D fields (cuboid / pose_transform / location / quaternion_xyzw) are
    PASSED THROUGH unchanged. The DOPE loader uses them only for (a) off-image
    binary visibility, which is driven by the 2D projected_cuboid we DO update,
    and (b) a rotation-only face-normal sign. Neither squash nor scale changes
    which face points at the camera, so the pass-through stays consistent. For
    truncation we likewise pass them through (camera-frame, 2D-padding-invariant).
    NOTE: pose_transform is therefore NOT a metric-correct pose for squashed
    samples (anisotropic warp has no rigid 3D pre-image); it is retained only to
    keep the loader's visibility weighting well-defined. Belief/affinity targets
    -- what DOPE actually trains on -- come purely from the warped 2D keypoints.

Run examples (env: pallet-pose, needs cv2):
  python challenge/scripts/dataset/augment_ratio_robust.py --kind squash \
      --out_dir data/pallet/training_data/aug_squash --sample 3000 --variants 1
  python challenge/scripts/dataset/augment_ratio_robust.py --kind scale \
      --out_dir data/pallet/training_data/aug_scale  --sample 1500 --variants 1
  python challenge/scripts/dataset/augment_ratio_robust.py --kind trunc \
      --out_dir data/pallet/training_data/aug_trunc  --sample 3000 --variants 1
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
import random

import cv2
import numpy as np

W_IMG, H_IMG = 640, 480
ASPECT = W_IMG / H_IMG
SENTINEL = -1.0
MARGIN_FRAC = 0.20  # belief-border safe margin (2*sigma/50=0.16, +slack)

EDGE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# Truncation cut-side weights (L/R biased, top almost excluded).
CUT_WEIGHTS = {
    "L": 0.30, "R": 0.30,
    "B": 0.15, "BL": 0.075, "BR": 0.075,
    "T": 0.01, "TL": 0.005, "TR": 0.005,
}
CUT_SIDES = list(CUT_WEIGHTS.keys())
CUT_W = list(CUT_WEIGHTS.values())

MIN_IN_IMAGE = 5
MIN_VIS_AREA = W_IMG * H_IMG * 0.10
MIN_VIS_DIM = 50.0


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
import keypoint_annotations_transform as kat  # noqa: E402


def _assert_matches(M, src_kps, dst_kps, tol=1e-6):
    """합성한 affine 이 실제로 계산된 좌표를 재현하는지 확인한다.

    합성 공식을 손으로 유도했으므로, 틀렸을 때 조용히 어긋난 라벨을 내보내지 않고
    여기서 죽게 한다.
    """
    for a, b in zip(src_kps, dst_kps):
        if is_sentinel(a) or is_sentinel(b):
            continue
        x = M[0, 0] * a[0] + M[0, 1] * a[1] + M[0, 2]
        y = M[1, 0] * a[0] + M[1, 1] * a[1] + M[1, 2]
        assert abs(x - b[0]) < tol and abs(y - b[1]) < tol, (
            f"affine 합성이 좌표를 재현하지 못한다: {(x, y)} != {tuple(b)}")


def is_sentinel(p):
    return p[0] == SENTINEL or p[1] == SENTINEL


def get_kps(obj):
    """9 kps = projected_cuboid[8] + centroid -> (9,2) float64."""
    kps = [list(map(float, p)) for p in obj["projected_cuboid"][:8]]
    kps.append(list(map(float, obj["projected_cuboid_centroid"])))
    return np.array(kps, dtype=np.float64)


def visible_pts(kps):
    return np.array([p for p in kps if not is_sentinel(p)], dtype=np.float64)


def extent(kps):
    pts = visible_pts(kps)
    if len(pts) == 0:
        return None
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def count_in_image(kps):
    n = 0
    for p in kps:
        if is_sentinel(p):
            continue
        if 0 <= p[0] < W_IMG and 0 <= p[1] < H_IMG:
            n += 1
    return n


def visible_bbox(kps):
    pts = [p for p in kps if not is_sentinel(p)
           and 0 <= p[0] < W_IMG and 0 <= p[1] < H_IMG]
    if not pts:
        return 0.0, 0.0, 0.0
    pts = np.asarray(pts, dtype=np.float64)
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()
    return w, h, w * h


def write_output(out_dir, stem, img, kps, src_obj, src_cam, meta, M=None,
                 parent_frame=None):
    cv2.imwrite(os.path.join(out_dir, stem + ".png"), img)
    obj = {
        "class": "pallet",
        "name": src_obj.get("name", "pallet"),
        "visibility": src_obj.get("visibility", 1),
        "projected_cuboid": [[float(x), float(y)] for x, y in kps[:8]],
        "projected_cuboid_centroid": [float(kps[8][0]), float(kps[8][1])],
        "aug": meta,
    }
    # Pass-through 3D / pose fields (see module docstring for why this is OK).
    for k in ("pose_transform", "cuboid", "location", "quaternion_xyzw",
              "euler_angles", "dimensions_m"):
        if k in src_obj:
            obj[k] = src_obj[k]
    # 정본 keypoint 필드도 같은 affine 으로 옮긴다.  이 스크립트는 flip 을 하지
    # 않으므로 인덱스 순열은 없다(docstring 참조) — perm 을 주지 않는다.
    if M is not None:
        kat.attach(obj, src_obj, M, W_IMG, H_IMG,
                   parent_frame=parent_frame,
                   transformation={"kind": "ratio_affine", "matrix":
                                   [[float(v) for v in row] for row in M],
                                   "meta": meta})
    out = {
        "camera_data": {
            "width": W_IMG,
            "height": H_IMG,
            "intrinsics": src_cam.get("intrinsics", {}),
        },
        "objects": [obj],
    }
    with open(os.path.join(out_dir, stem + ".json"), "w") as fp:
        json.dump(out, fp, indent=2)


# --------------------------------------------------------------------------- #
# squash / scale (affine about centroid, keep 640x480 canvas)
# --------------------------------------------------------------------------- #
def apply_affine(img, kps, sx, sy):
    """Scale by (sx,sy) about the visible centroid, then translate so the
    object stays roughly centered. Image and keypoints get the SAME affine, so
    they remain locked. Returns (out_img, out_kps)."""
    pts = visible_pts(kps)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    # 2x3 affine: x' = sx*(x-cx)+cx + tx ; we recentре to image center too.
    tx = W_IMG / 2.0 - cx
    ty = H_IMG / 2.0 - cy
    M = np.array([
        [sx, 0.0, cx - sx * cx + tx],
        [0.0, sy, cy - sy * cy + ty],
    ], dtype=np.float64)
    out_img = cv2.warpAffine(
        img, M, (W_IMG, H_IMG),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    out = []
    for p in kps:
        if is_sentinel(p):
            out.append([SENTINEL, SENTINEL])
        else:
            x = M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2]
            y = M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]
            out.append([x, y])
    return out_img, np.array(out, dtype=np.float64), M


def gen_affine_variant(img, kps, kind, rng, args):
    """kind in {squash, scale}. Returns (img,kps,meta) or None."""
    if kind == "scale":
        s = rng.uniform(args.scale_min, args.scale_max)
        sx = sy = s
    else:  # squash: independent axes
        sx = rng.uniform(args.squash_min, args.squash_max)
        sy = rng.uniform(args.squash_min, args.squash_max)
        # ensure it is actually anisotropic enough to matter
        if abs(sx - sy) < 0.12:
            sy = sx * rng.choice([0.65, 0.8, 1.25, 1.5])
            sy = float(np.clip(sy, args.squash_min, args.squash_max))
    out_img, out_kps, M = apply_affine(img, kps, sx, sy)
    # keep enough of the pallet on screen
    if count_in_image(out_kps) < 6:
        return None
    _, _, varea = visible_bbox(out_kps)
    if varea < W_IMG * H_IMG * 0.03:
        return None
    meta = {"kind": kind, "sx": round(sx, 4), "sy": round(sy, 4)}
    return out_img, out_kps, meta, M


# --------------------------------------------------------------------------- #
# truncation (crop + reflect-pad back, reuses proven pipeline)
# --------------------------------------------------------------------------- #
def make_crop_window(ext, img_w, img_h, side, f, rng):
    px0, py0, px1, py1 = ext
    pw = max(px1 - px0, 1.0)
    ph = max(py1 - py0, 1.0)
    mx = pw * rng.uniform(0.05, 0.20)
    my = ph * rng.uniform(0.05, 0.20)
    L, R, T, B = px0 - mx, px1 + mx, py0 - my, py1 + my
    if "L" in side:
        L = px0 + f * pw
    if "R" in side:
        R = px1 - f * pw
    if "T" in side:
        T = py0 + f * ph
    if "B" in side:
        B = py1 - f * ph
    L, T = max(0.0, L), max(0.0, T)
    R, B = min(float(img_w), R), min(float(img_h), B)
    if R - L < 20 or B - T < 20:
        return None
    cw, ch = R - L, B - T
    if cw / ch > ASPECT:
        grow = cw / ASPECT - ch
        gt = grow * (0.0 if "T" in side else (1.0 if "B" in side else 0.5))
        T -= gt
        B += grow - gt
    else:
        grow = ch * ASPECT - cw
        gl = grow * (0.0 if "L" in side else (1.0 if "R" in side else 0.5))
        L -= gl
        R += grow - gl
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
    L, T = max(0.0, L), max(0.0, T)
    R, B = min(float(img_w), R), min(float(img_h), B)
    cw, ch = R - L, B - T
    if cw < 20 or ch < 20:
        return None
    return L, T, cw, ch


def crop_kps(kps, win):
    cx0, cy0, cw, ch = win
    sx, sy = W_IMG / cw, H_IMG / ch
    out = []
    for p in kps:
        if is_sentinel(p):
            out.append([SENTINEL, SENTINEL])
        else:
            out.append([(p[0] - cx0) * sx, (p[1] - cy0) * sy])
    return np.array(out, dtype=np.float64)


def required_pad(kps):
    pts = visible_pts(kps)
    if len(pts) == 0:
        return 0
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    mx, my = MARGIN_FRAC * W_IMG, MARGIN_FRAC * H_IMG

    def fits(P):
        dw, dh = W_IMG + 2 * P, H_IMG + 2 * P
        sx, sy = W_IMG / dw, H_IMG / dh
        return ((xmin + P) * sx >= mx and (xmax + P) * sx <= W_IMG - mx
                and (ymin + P) * sy >= my and (ymax + P) * sy <= H_IMG - my)

    if fits(0):
        return 0
    P = 1
    while not fits(P) and P < 5000:
        P += max(1, P // 8)
    lo = max(0, P - max(1, P // 8))
    for q in range(lo, P + 1):
        if fits(q):
            return q
    return P


def pad_back(img, kps):
    pad = required_pad(kps)
    if pad <= 0:
        return img, kps, 0
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape[:2]
    out_img = cv2.resize(padded, (W_IMG, H_IMG), interpolation=cv2.INTER_LINEAR)
    sx, sy = W_IMG / pw, H_IMG / ph
    out = []
    for p in kps:
        if is_sentinel(p):
            out.append([SENTINEL, SENTINEL])
        else:
            out.append([(p[0] + pad) * sx, (p[1] + pad) * sy])
    return out_img, np.array(out, dtype=np.float64), pad


def gen_trunc_variant(img, kps, rng, args):
    img_h, img_w = img.shape[:2]
    ext = extent(kps)
    if ext is None:
        return None
    for _ in range(args.max_tries):
        side = rng.choices(CUT_SIDES, weights=CUT_W, k=1)[0]
        deep = rng.random() < args.deep_ratio
        f = rng.uniform(0.35, 0.55) if deep else rng.uniform(0.10, 0.30)
        win = make_crop_window(ext, img_w, img_h, side, f, rng)
        if win is None:
            continue
        cx0, cy0, cw, ch = win
        crop = img[int(round(cy0)):int(round(cy0 + ch)),
                   int(round(cx0)):int(round(cx0 + cw))]
        if crop.size == 0:
            continue
        cimg = cv2.resize(crop, (W_IMG, H_IMG), interpolation=cv2.INTER_LINEAR)
        ckps = crop_kps(kps, win)
        if count_in_image(ckps) < MIN_IN_IMAGE:
            continue
        vw, vh, varea = visible_bbox(ckps)
        if varea < MIN_VIS_AREA or min(vw, vh) < MIN_VIS_DIM:
            continue
        pimg, pkps, pad = pad_back(cimg, ckps)
        # after pad everything should be inside
        still_off = sum(
            1 for p in pkps if not is_sentinel(p)
            and not (0 <= p[0] < W_IMG and 0 <= p[1] < H_IMG))
        meta = {"kind": "trunc", "side": side, "f": round(f, 3),
                "deep": deep, "pad": int(pad), "off_after_pad": still_off}
        # crop 과 pad_back 을 합성한 affine.  정본 keypoint 필드를 같은 변환으로
        # 옮기기 위해 필요하다.  두 단계 모두 축정렬 scale+translate 라 합성이 닫힌다.
        csx, csy = W_IMG / cw, H_IMG / ch
        psx = W_IMG / (W_IMG + 2 * pad) if pad > 0 else 1.0
        psy = H_IMG / (H_IMG + 2 * pad) if pad > 0 else 1.0
        M = np.array([[csx * psx, 0.0, (-cx0 * csx + pad) * psx],
                      [0.0, csy * psy, (-cy0 * csy + pad) * psy]])
        _assert_matches(M, kps, pkps)
        return pimg, pkps, meta, M
    return None


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def collect_sources(src_dir):
    out = []
    for jp in sorted(glob.glob(os.path.join(src_dir, "*.json"))):
        if jp.endswith(".json.orig"):
            continue
        pp = jp[:-5] + ".png"
        if os.path.exists(pp):
            out.append((jp, pp, os.path.splitext(os.path.basename(jp))[0]))
    return out


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


def verify(out_dir, records, rng, n=8):
    vdir = os.path.join(out_dir, "_verify")
    os.makedirs(vdir, exist_ok=True)
    pick = rng.sample(records, min(n, len(records)))
    print("\n=== VERIFY (random {}) ===".format(len(pick)))
    for rec in pick:
        stem = rec["stem"]
        img = cv2.imread(os.path.join(out_dir, stem + ".png"))
        d = json.load(open(os.path.join(out_dir, stem + ".json")))
        o = d["objects"][0]
        kps = get_kps(o)
        m = 80
        canvas = np.full((H_IMG + 2 * m, W_IMG + 2 * m, 3), 40, np.uint8)
        canvas[m:m + H_IMG, m:m + W_IMG] = img
        cv2.rectangle(canvas, (m, m), (m + W_IMG, m + H_IMG), (0, 255, 255), 2)

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
            inside = 0 <= p[0] < W_IMG and 0 <= p[1] < H_IMG
            in_c += inside
            off_c += (not inside)
            col = (0, 0, 255) if i == 8 else ((0, 255, 0) if inside else (255, 0, 255))
            cv2.circle(canvas, cv_pt(p), 5, col, -1)
            cv2.putText(canvas, str(i), (cv_pt(p)[0] + 5, cv_pt(p)[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        meta = o.get("aug", {})
        label = "{} {}".format(stem, json.dumps(meta))
        cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.putText(canvas, "in={} off={}".format(in_c, off_c), (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(vdir, stem + "_overlay.png"), canvas)
        print("  {}  {}  in={} off={}".format(stem, json.dumps(meta), in_c, off_c))
    print("overlays -> {}".format(vdir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["squash", "scale", "trunc"])
    ap.add_argument("--src_dir",
                    default="data/pallet/training_data/mixed_v8_train")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample", type=int, default=0,
                    help="randomly sample N source frames (0=all)")
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tries", type=int, default=40)
    # squash/scale ranges
    ap.add_argument("--squash_min", type=float, default=0.6)
    ap.add_argument("--squash_max", type=float, default=1.5)
    ap.add_argument("--scale_min", type=float, default=0.6)
    ap.add_argument("--scale_max", type=float, default=1.6)
    # trunc
    ap.add_argument("--deep-ratio", type=float, default=0.3)
    ap.add_argument("--verify", type=int, default=8)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    clear_out_dir(args.out_dir)
    sources = collect_sources(args.src_dir)
    print("src_dir : {}".format(args.src_dir))
    print("sources : {} (json+png pairs)".format(len(sources)))
    if args.sample and args.sample < len(sources):
        sources = rng.sample(sources, args.sample)
        print("sampled : {} (seed={})".format(len(sources), args.seed))

    records = []
    n_fail = 0
    pads = []
    for jp, pp, stem in sources:
        d = json.load(open(jp))
        cam = d.get("camera_data", {})
        objs = d.get("objects", [])
        if not objs:
            continue
        obj = objs[0]
        if obj.get("visibility", 1) <= 0:
            continue
        kps = get_kps(obj)
        img = cv2.imread(pp)
        if img is None:
            continue
        for vi in range(args.variants):
            if args.kind in ("squash", "scale"):
                res = gen_affine_variant(img, kps, args.kind, rng, args)
            else:
                res = gen_trunc_variant(img, kps, rng, args)
            if res is None:
                n_fail += 1
                continue
            out_img, out_kps, meta, M = res
            uniq = "{}_{}_v{}".format(args.kind, stem, vi)
            write_output(args.out_dir, uniq, out_img, out_kps, obj, cam, meta,
                         M=M, parent_frame=jp)
            records.append({"stem": uniq, "meta": meta})
            if "pad" in meta:
                pads.append(meta["pad"])

    print("\n=== SUMMARY ({}) ===".format(args.kind))
    print("source frames : {}".format(len(sources)))
    print("variants/frame: {}".format(args.variants))
    print("generated     : {}".format(len(records)))
    print("failed        : {}".format(n_fail))
    if args.kind == "trunc" and pads:
        pa = np.array(pads)
        print("pad px        : min={} median={} mean={:.0f} max={}".format(
            pa.min(), int(np.median(pa)), pa.mean(), pa.max()))
        off = sum(r["meta"].get("off_after_pad", 0) for r in records)
        print("kps still off : {}  (should be 0)".format(off))
    if args.kind in ("squash", "scale"):
        sxs = [r["meta"]["sx"] for r in records]
        sys_ = [r["meta"]["sy"] for r in records]
        print("sx range      : {:.2f}..{:.2f}".format(min(sxs), max(sxs)))
        print("sy range      : {:.2f}..{:.2f}".format(min(sys_), max(sys_)))
        ar = [r["meta"]["sx"] / r["meta"]["sy"] for r in records]
        print("aspect(sx/sy) : {:.2f}..{:.2f} median={:.2f}".format(
            min(ar), max(ar), float(np.median(ar))))
    if records and args.verify > 0:
        verify(args.out_dir, records, rng, n=args.verify)
    print("\nout_dir       : {}".format(args.out_dir))


if __name__ == "__main__":
    main()
