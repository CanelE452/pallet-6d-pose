"""verify_pvnet_gt.py — STEP E 검증: PVNet dense 벡터 GT 시각/수치 sanity.

3 샘플(mixed_v8 1 + palletobj v1 정상 1 + v1 truncation 1) 에 대해 belief 격자(50px)
에서 마스크/키포인트/벡터 quiver overlay 를 원본 이미지에 그려 저장하고, offset
정의 sanity assert(픽셀 + v_k == kp_k) 를 수행한다.

실행: conda env pallet-pose 에서
  cd Deep_Object_Pose/common && python verify_pvnet_gt.py
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils_pvnet import make_cuboid_mask, make_vector_field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "stage1_pvnet_gt")
GRID = 50  # belief 격자

SAMPLES = [
    ("mixed_v8",
     os.path.join(ROOT, "data/pallet/training_data/mixed_v8_train/000000.png"),
     os.path.join(ROOT, "data/pallet/training_data/mixed_v8_train/000000.json")),
    ("palletobj_v1_normal",
     os.path.join(ROOT, "challenge/data/training/v1/part_000/train_palletobj_v1/000873.png"),
     os.path.join(ROOT, "challenge/data/training/v1/part_000/train_palletobj_v1/000873.json")),
    ("palletobj_v1_truncation",
     os.path.join(ROOT, "challenge/data/training/v1/part_000/train_palletobj_v1/000000.png"),
     os.path.join(ROOT, "challenge/data/training/v1/part_000/train_palletobj_v1/000000.json")),
]


def load_kps9(json_path):
    o = json.load(open(json_path))["objects"][0]
    kps = list(o["projected_cuboid"])
    kps.append(o["projected_cuboid_centroid"])
    return np.array(kps, dtype=np.float64)  # (9,2) original image px


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[OUT] {OUT_DIR}")
    all_ok = True

    for tag, img_path, json_path in SAMPLES:
        if not (os.path.exists(img_path) and os.path.exists(json_path)):
            print(f"[SKIP] {tag}: missing file")
            continue
        img = cv2.imread(img_path)  # BGR
        H, W = img.shape[:2]
        kps_img = load_kps9(json_path)

        # 원본 px → 50px 격자 좌표 (로더의 output_size Resize 와 동일한 anisotropic scale)
        sx, sy = GRID / W, GRID / H
        kps50 = kps_img.copy()
        kps50[:, 0] *= sx
        kps50[:, 1] *= sy

        # GT 생성 (offset 모드 = 기본)
        vec, mask = make_vector_field(kps50, GRID, unit=False)

        # ---- sanity assert: 마스크 내 임의 픽셀 + v_k == kp_k ----
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            print(f"[WARN] {tag}: empty mask")
            all_ok = False
        else:
            idx = np.random.RandomState(0).choice(len(xs), size=min(50, len(xs)),
                                                   replace=False)
            max_err = 0.0
            for i in idx:
                px, py = xs[i], ys[i]
                for k in range(9):
                    rec = np.array([px + vec[2 * k, py, px],
                                    py + vec[2 * k + 1, py, px]])
                    err = np.linalg.norm(rec - kps50[k])
                    max_err = max(max_err, err)
            ok = max_err < 1e-3
            all_ok = all_ok and ok
            print(f"[SANITY] {tag}: offset 복원 max_err={max_err:.2e} "
                  f"({'OK' if ok else 'FAIL'})  mask_px={len(xs)}")

        # ---- overlay (원본 해상도로 업샘플해서 그림) ----
        vis = img.copy()
        # (a) 마스크 반투명 (50px → 원본 nearest 업샘플)
        mask_full = cv2.resize(mask * 255, (W, H), interpolation=cv2.INTER_NEAREST)
        green = np.zeros_like(vis); green[:, :, 1] = 255
        m3 = (mask_full > 0)[:, :, None]
        vis = np.where(m3, (0.6 * vis + 0.4 * green).astype(np.uint8), vis)

        # (c) quiver: 마스크 내 subsample 픽셀 → centroid(kp8) 방향 (원본 px 로 환산)
        step = 3
        for py in range(0, GRID, step):
            for px in range(0, GRID, step):
                if mask[py, px] == 0:
                    continue
                # centroid 로 향하는 offset 벡터 (50px) → 원본 px
                dx = vec[16, py, px] / sx
                dy = vec[17, py, px] / sy
                ox, oy = px / sx, py / sy
                cv2.arrowedLine(vis, (int(ox), int(oy)),
                                (int(ox + dx), int(oy + dy)),
                                (255, 200, 0), 1, tipLength=0.15)

        # (b) 9 키포인트 (원본 px). 0-3 front=red, 4-7 rear=blue, 8 centroid=yellow
        for k in range(9):
            x, y = int(kps_img[k, 0]), int(kps_img[k, 1])
            if k < 4:
                c = (0, 0, 255)
            elif k < 8:
                c = (255, 0, 0)
            else:
                c = (0, 255, 255)
            # off-frame 코너도 화살표 끝이 보이게 경계로 clamp 한 위치에 표시
            xc, yc = np.clip(x, 0, W - 1), np.clip(y, 0, H - 1)
            cv2.circle(vis, (xc, yc), 5, c, -1)
            cv2.putText(vis, str(k), (xc + 4, yc - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        out_path = os.path.join(OUT_DIR, f"{tag}_overlay.jpg")
        cv2.imwrite(out_path, vis)
        print(f"[SAVE] {out_path}")

        # 추가: unit 모드도 한 장 (방향만 확인용) — mixed_v8 만
        if tag == "mixed_v8":
            vec_u, _ = make_vector_field(kps50, GRID, mask=mask, unit=True)
            visu = img.copy()
            for py in range(0, GRID, step):
                for px in range(0, GRID, step):
                    if mask[py, px] == 0:
                        continue
                    dx = vec_u[16, py, px] * 12  # 단위벡터 → 보이게 12px 스케일
                    dy = vec_u[17, py, px] * 12
                    ox, oy = px / sx, py / sy
                    cv2.arrowedLine(visu, (int(ox), int(oy)),
                                    (int(ox + dx), int(oy + dy)),
                                    (0, 200, 255), 1, tipLength=0.2)
            cx, cy = int(kps_img[8, 0]), int(kps_img[8, 1])
            cv2.circle(visu, (cx, cy), 5, (0, 255, 255), -1)
            up = os.path.join(OUT_DIR, f"{tag}_unit_quiver.jpg")
            cv2.imwrite(up, visu)
            print(f"[SAVE] {up}")

    print("\n[RESULT]", "ALL SANITY OK" if all_ok else "SOME SANITY FAILED")


if __name__ == "__main__":
    main()
