#!/usr/bin/env python3
"""padding 학습본과 무패딩 학습본을 **원본 좌표계**에서 같은 잣대로 비교한다.

두 모델은 학습 canvas 가 달라(840x680 vs 640x480) 각자의 val 로 재면 비교가 안 된다.
그래서 GT 를 원본 640x480 좌표로 되돌리고, 각 모델에게 자기 계약대로 입력을 준다.

    padding 본   원본 -> +100 reflect -> 추론 -> 좌표 -100
    무패딩 본    원본 -> 그대로 추론

`live_gt_trunc_val` 은 디스크에 840x680 패딩본으로만 있으므로 중앙을 잘라 원본을
복원한다(reflect padding 이라 중앙 crop = 원본).

잘림 프레임에서는 "화면 밖 코너를 예측하느냐" 가 논점이다(memory
`yolo-padding-truncation-wins`).  그래서 두 가지를 나눠 본다 —
화면 안 코너의 위치 오차와, PnP 6점 요건을 채우는 프레임 비율.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
REPO = HERE.parents[3]
ONE = REPO / "challenge/yolo_pose_one_model"
PAD = 100

# 학습 산출물은 `pallet_yolo_loss.model.ChallengeC4PoseModel` 로 pickle 돼 있어 이
# 경로 없이는 로드조차 안 된다(배포본은 strip_custom_model_class.py 로 떼어낸다).
sys.path.insert(0, str(REPO))

PERMS = json.loads((TRACK / "C4_PERMUTATIONS.json").read_text(encoding="utf-8"))
P = {int(k): list(v) for k, v in PERMS["permutations"].items()}
DEGREES = [0, 90, 180, 270]


def load_set(ds: str, padded_on_disk: bool):
    """(원본 BGR, 원본 좌표 GT 9x2, in-frame mask) 목록."""
    out = []
    for ip in sorted(glob.glob(str(ONE / f"datasets/{ds}/images/val/*.png"))):
        lp = ip.replace("/images/", "/labels/").replace(".png", ".txt")
        im = cv2.imread(ip)
        h, w = im.shape[:2]
        kp = np.array(open(lp).read().split()[5:], float).reshape(-1, 3)
        gt = kp[:, :2] * [w, h]
        sup = kp[:, 2] > 0                      # 라벨이 감독한 코너
        if padded_on_disk:
            im = im[PAD:h - PAD, PAD:w - PAD]   # 중앙 crop = 원본 복원
            gt = gt - PAD
        oh, ow = im.shape[:2]
        inframe = sup & (gt[:, 0] >= 0) & (gt[:, 0] < ow) & (gt[:, 1] >= 0) & (gt[:, 1] < oh)
        out.append((Path(ip).stem, im, gt, sup, inframe))
    return out


def predict(model, im, use_pad: bool):
    src = cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101) if use_pad else im
    r = model.predict(src, imgsz=640, conf=0.25, verbose=False)[0]
    if not len(r.boxes):
        return None
    kp = r.keypoints.data[0][:, :2].cpu().numpy()
    return kp - PAD if use_pad else kp


def per_frame(model, data, use_pad: bool) -> dict:
    """프레임별 결과.  집계는 하지 않는다 — 검출 집합이 arm 마다 다르므로
    median 을 각자 집합에서 내면 비교가 성립하지 않는다(2026-06-22 에 겪은 함정)."""
    rows = {}
    for stem, im, gt, sup, inframe in data:
        pr = predict(model, im, use_pad)
        if pr is None:
            rows[stem] = None
            continue
        oh, ow = im.shape[:2]
        inside = ((pr[:8, 0] >= 0) & (pr[:8, 0] < ow) &
                  (pr[:8, 1] >= 0) & (pr[:8, 1] < oh)).sum()
        rows[stem] = {
            "inframe": (float(np.linalg.norm(gt[inframe] - pr[inframe], axis=1).mean())
                        if inframe.any() else float("nan")),
            "c4": min(float(np.linalg.norm(gt[P[d]][sup] - pr[sup], axis=1).mean())
                      for d in DEGREES),
            "pnp6": bool(inside >= 6),
        }
    return rows


def summarize(rows: dict, both: set) -> dict:
    """`both` = 두 arm 이 **모두** 검출한 프레임.  거기서만 오차를 비교한다."""
    det = [s for s, r in rows.items() if r]
    paired = [rows[s] for s in sorted(both)]
    inf = np.array([r["inframe"] for r in paired if not np.isnan(r["inframe"])])
    c4 = np.array([r["c4"] for r in paired])
    n = len(rows)
    return {
        "n": n, "detected": len(det), "det_rate": len(det) / n,
        "paired_n": len(paired),
        "paired_inframe_median_px": float(np.median(inf)) if len(inf) else float("nan"),
        "paired_c4_median_px": float(np.median(c4)) if len(c4) else float("nan"),
        # 집합이 달라도 비교되는 절대수 — 검출한 것 중 좋은/망가진 프레임
        "good_lt10px": int(sum(1 for r in rows.values() if r and r["c4"] < 10)),
        "gross_gt20px": int(sum(1 for r in rows.values() if r and r["c4"] > 20)),
        "pnp6_frames": int(sum(1 for r in rows.values() if r and r["pnp6"])),
        "pnp6_rate": sum(1 for r in rows.values() if r and r["pnp6"]) / n,
    }


def bench(model, im, use_pad: bool, n: int = 60) -> float:
    import torch
    for _ in range(15):
        predict(model, im, use_pad)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        predict(model, im, use_pad)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TRACK / "PADDING_VARIANT_COMPARISON.json"))
    args = ap.parse_args(argv)
    from ultralytics import YOLO

    arms = {
        "pad100": (TRACK / "clean_label/stage2_c4/weights/best.pt", True),
        "nopad":  (TRACK / "nopad_label/stage2_c4/weights/best.pt", False),
    }
    sets = {
        "clean_155": load_set("live_gt_v7_nopad", padded_on_disk=False),
        "truncated_210": load_set("live_gt_trunc_val", padded_on_disk=True),
    }
    for k, v in sets.items():
        print(f"{k}: {len(v)} 프레임, 원본 {v[0][1].shape[1]}x{v[0][1].shape[0]}")

    raw, res = {}, {}
    for arm, (w, use_pad) in arms.items():
        if not w.is_file():
            print(f"[STOP] {arm} 가중치 없음: {w}")
            return 1
        m = YOLO(str(w))
        raw[arm] = {sk: per_frame(m, data, use_pad) for sk, data in sets.items()}
        res[arm] = {"weights": str(w.relative_to(REPO)), "inference_padding": use_pad,
                    "latency_ms": bench(m, sets["clean_155"][0][1], use_pad)}

    for sk in sets:
        both = {s for s in raw["pad100"][sk]
                if raw["pad100"][sk][s] and raw["nopad"][sk][s]}
        print(f"\n--- {sk}  (두 arm 이 모두 검출한 {len(both)} 프레임에서 오차 비교) ---")
        for arm in arms:
            s = summarize(raw[arm][sk], both)
            res[arm][sk] = s
            print(f"  {arm:7s} 검출 {s['detected']:3d}/{s['n']} ({s['det_rate']:5.1%})  "
                  f"짝 화면안 {s['paired_inframe_median_px']:7.2f}px  "
                  f"짝 C4 {s['paired_c4_median_px']:7.2f}px  "
                  f"good<10px {s['good_lt10px']:3d}  gross>20px {s['gross_gt20px']:3d}  "
                  f"PnP6 {s['pnp6_rate']:5.1%}")
    for arm in arms:
        print(f"{arm:7s} 지연 {res[arm]['latency_ms']:.2f} ms/frame")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
