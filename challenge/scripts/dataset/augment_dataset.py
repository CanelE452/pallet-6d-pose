"""challenge/scripts/dataset/augment_dataset.py

Manual GT 데이터를 augmentation으로 늘린다.
augment 종류 (rotate 제외):
  - horizontal flip (좌우 mirror, ID swap 없음 → object-fixed convention 유지)
  - random crops (5 positions × 0.85 scale)
  - shear (±5% in x/y)

출력은 NDDS 호환 JSON + PNG.  잘린 keypoint 는 [-1,-1] sentinel.
pose_transform 은 augment 후 부정확하므로 저장 안 함 (학습은 belief map 만 사용).

사용:
  python challenge/scripts/dataset/augment_dataset.py \
      --src challenge/data/01_real/manual_gt/capturepallet07_manual_gt \
      --out challenge/data/01_real/augmented/capturepallet07_augmented \
      --include_original
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
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3]))
from challenge.data_paths import get as _dp  # 경로 단일 출처


def _transform_kps(kps, M):
    """affine M (2x3) 적용. None 은 그대로."""
    out = []
    for kp in kps:
        if kp is None:
            out.append(None); continue
        p = np.array([kp[0], kp[1], 1.0])
        u, v = (M @ p)[:2]
        out.append([float(u), float(v)])
    return out


def hflip(img, kps, K):
    h, w = img.shape[:2]
    M = np.array([[-1, 0, w - 1], [0, 1, 0]], dtype=np.float64)
    img2 = cv2.flip(img, 1)
    kps2 = _transform_kps(kps, M)
    K2 = K.copy()
    K2[0, 2] = w - 1 - K[0, 2]
    return img2, kps2, K2


def crop(img, kps, K, x0, y0, cw, ch):
    img2 = img[y0:y0 + ch, x0:x0 + cw].copy()
    M = np.array([[1, 0, -x0], [0, 1, -y0]], dtype=np.float64)
    kps2 = _transform_kps(kps, M)
    K2 = K.copy()
    K2[0, 2] -= x0
    K2[1, 2] -= y0
    return img2, kps2, K2


def shear(img, kps, K, sx, sy):
    h, w = img.shape[:2]
    M = np.array([[1, sx, 0], [sy, 1, 0]], dtype=np.float64)
    img2 = cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REPLICATE)
    kps2 = _transform_kps(kps, M)
    return img2, kps2, K.copy()


def write_gt(stem_path, img, kps_2d, K, dims, gt_source):
    h, w = img.shape[:2]
    cuboid = []
    for kp in kps_2d[:8]:
        if kp is None or kp[0] < 0 or kp[1] < 0 or kp[0] >= w or kp[1] >= h:
            cuboid.append([-1.0, -1.0])
        else:
            cuboid.append([float(kp[0]), float(kp[1])])
    if len(kps_2d) > 8 and kps_2d[8] is not None and \
       0 <= kps_2d[8][0] < w and 0 <= kps_2d[8][1] < h:
        centroid = [float(kps_2d[8][0]), float(kps_2d[8][1])]
    else:
        centroid = [-1.0, -1.0]
    manual = []
    for kp in kps_2d:
        if kp is None or kp[0] < 0 or kp[1] < 0 or kp[0] >= w or kp[1] >= h:
            manual.append(None)
        else:
            manual.append([float(kp[0]), float(kp[1])])
    ann = {
        "camera_data": {
            "width": w, "height": h,
            "intrinsics": {
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            },
        },
        "objects": [{
            "class": "pallet",
            "name": "real_pallet",
            "visibility": 1,
            "projected_cuboid": cuboid,
            "projected_cuboid_centroid": centroid,
            "dimensions_m": dims,
            "gt_source": gt_source,
            "manual_kps": manual,
        }],
    }
    with open(stem_path + ".json", "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2)
    cv2.imwrite(stem_path + ".png", img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=_dp("manual.pallet07"))
    ap.add_argument("--out", default=_dp("real.pallet07_augmented"))
    ap.add_argument("--include_original", action="store_true",
                    help="원본도 결과에 포함")
    ap.add_argument("--crop_scale", type=float, default=0.85)
    ap.add_argument("--shear_amount", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    json_paths = sorted(glob.glob(os.path.join(args.src, "*.json")))
    print(f"[Augment] src={args.src}  n={len(json_paths)}")
    print(f"          out={args.out}")

    out_idx = 0
    for jp in json_paths:
        png_p = jp[:-5] + ".png"
        if not os.path.isfile(png_p):
            continue
        with open(jp, "r", encoding="utf-8") as f:
            d = json.load(f)
        img = cv2.imread(png_p)
        h, w = img.shape[:2]
        o = d["objects"][0]
        ci = d["camera_data"]["intrinsics"]
        K = np.array([[ci["fx"], 0, ci["cx"]],
                      [0, ci["fy"], ci["cy"]],
                      [0, 0, 1]], dtype=np.float64)
        kps_raw = o.get("manual_kps") or []
        kps = [list(k) if k is not None else None for k in kps_raw]
        if len(kps) < 9:
            kps += [None] * (9 - len(kps))
        dims = o.get("dimensions_m", {"width": 1.1, "height": 0.11, "depth": 1.3})
        gtsrc = "manual_aug"

        # 1) 원본 (옵션)
        if args.include_original:
            write_gt(os.path.join(args.out, f"{out_idx:06d}"),
                     img, kps, K, dims, "manual"); out_idx += 1

        # 2) Horizontal flip
        img_f, kps_f, K_f = hflip(img, kps, K)
        write_gt(os.path.join(args.out, f"{out_idx:06d}"),
                 img_f, kps_f, K_f, dims, gtsrc); out_idx += 1

        # 3) Crops — 5 positions, scale 0.85
        cw, ch = int(w * args.crop_scale), int(h * args.crop_scale)
        positions = [
            (0, 0),
            (w - cw, 0),
            (0, h - ch),
            (w - cw, h - ch),
            ((w - cw) // 2, (h - ch) // 2),
        ]
        for x0, y0 in positions:
            img_c, kps_c, K_c = crop(img, kps, K, x0, y0, cw, ch)
            write_gt(os.path.join(args.out, f"{out_idx:06d}"),
                     img_c, kps_c, K_c, dims, gtsrc); out_idx += 1

        # 4) Shear — 4 directions
        s = args.shear_amount
        for sx, sy in [(s, 0), (-s, 0), (0, s), (0, -s)]:
            img_s, kps_s, K_s = shear(img, kps, K, sx, sy)
            write_gt(os.path.join(args.out, f"{out_idx:06d}"),
                     img_s, kps_s, K_s, dims, gtsrc); out_idx += 1

    print(f"[Done] {out_idx} files → {args.out}")


if __name__ == "__main__":
    main()
