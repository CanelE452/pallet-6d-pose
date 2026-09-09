#!/usr/bin/env python3
"""연속 촬영에서 예측 face phase 가 프레임마다 튀는지 진단한다 (GT 미사용).

`newauto` 의 계약은 ``POSE_FACE_KPTS = (0,1,2,3)`` 이고, ``_box_object_points`` 의
원점이 **전면 중심**이다.  즉 0~3 이 어느 면이냐에 따라 ``tvec`` 이 통째로 옮겨간다
(정사각 1.10 m 기준 인접 면 중심까지 0.55·√2 ≈ 0.78 m).  yaw 도 90도씩 튄다.

그러므로 C4 로 학습해 symmetry metric 이 좋아져도, 예측의 face phase 가 프레임마다
바뀌면 downstream 정렬은 깨진다.  그것을 여기서 잰다.

**GT 로 permutation 을 고르지 않는다.**  배포 시에는 GT 가 없기 때문이다.  이웃한
두 프레임의 예측만 비교해서, 다음 프레임이 이전 프레임 대비 몇 도 돌아 보이는지를
잰다.  카메라와 물체가 연속적으로 움직이면 정답은 항상 0도여야 한다.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

TRACK = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
CAP = REPO / "challenge/data/01_real/_live_captures"
ONE = REPO / "challenge/yolo_pose_one_model"

PERMS = json.loads((TRACK / "C4_PERMUTATIONS.json").read_text(encoding="utf-8"))
P = {int(k): list(v) for k, v in PERMS["permutations"].items()}
DEGREES = [0, 90, 180, 270]
PAD = 100


def sequence(session: str, n: int, stride: int) -> list[str]:
    hits = sorted(CAP.glob(f"*/sessions/{session}/rgb"))
    if not hits:
        raise SystemExit(f"세션 없음: {session}")
    imgs = sorted(glob.glob(str(hits[0] / "*.png")))
    return imgs[::stride][:n]


def run(weights: Path, imgs: list[str]) -> dict:
    model = YOLO(str(weights))
    prev = None
    jumps = {d: 0 for d in DEGREES}
    face_shift, miss, n_pair = [], 0, 0
    for ip in imgs:
        im = cv2.imread(ip)
        padded = cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        r = model.predict(padded, imgsz=640, conf=0.25, verbose=False)[0]
        if not len(r.boxes):
            miss += 1
            prev = None
            continue
        kp = r.keypoints.data[0][:, :2].cpu().numpy() - PAD
        if prev is not None:
            # 이웃 프레임 대비 몇 도 돌아 보이는가 — 연속 촬영이면 0도가 정답이다.
            d = {g: float(np.linalg.norm(kp[P[g]] - prev, axis=1).mean()) for g in DEGREES}
            best = min(d, key=d.get)
            jumps[best] += 1
            n_pair += 1
            # 전면 중심(0~3 평균)의 이동 — newauto tvec 이 이 점을 따라간다.
            face_shift.append(float(np.linalg.norm(kp[:4].mean(0) - prev[:4].mean(0))))
        prev = kp
    fs = np.array(face_shift) if face_shift else np.array([np.nan])
    stable = jumps[0] / n_pair if n_pair else float("nan")
    return {
        "n_images": len(imgs), "n_missed": miss, "n_pairs": n_pair,
        "phase_hist": {str(d): jumps[d] for d in DEGREES},
        "phase_stable_rate": stable,
        "face_center_shift_px": {
            "median": float(np.median(fs)), "p90": float(np.percentile(fs, 90)),
            "max": float(fs.max()),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="forklift_v4_20260904_142318")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args(argv)

    imgs = sequence(args.session, args.n, args.stride)
    print(f"세션 {args.session}   프레임 {len(imgs)}장 (stride {args.stride})\n")

    out = {"session": args.session, "n_images": len(imgs), "stride": args.stride,
           "note": "GT 미사용 — 이웃 프레임 예측끼리만 비교", "arms": {}}
    arms = {"F0": TRACK / "F0/weights/best.pt", "F1": TRACK / "F1/weights/best.pt",
            "F2": TRACK / "F2/weights/best.pt",
            "v4": ONE / "runs_live_gt/ft_live_gt_v4/weights/best.pt"}
    for arm, w in arms.items():
        if not w.is_file():
            print(f"[skip] {arm} 가중치 없음")
            continue
        res = run(w, imgs)
        out["arms"][arm] = res
        h = res["phase_hist"]
        print(f"{arm}  pairs {res['n_pairs']}  미검출 {res['n_missed']}")
        print(f"   phase  0도 {h['0']}  90도 {h['90']}  180도 {h['180']}  270도 {h['270']}"
              f"   안정률 {res['phase_stable_rate']*100:.1f}%")
        fc = res["face_center_shift_px"]
        print(f"   전면중심 이동  median {fc['median']:.1f}px  p90 {fc['p90']:.1f}  "
              f"max {fc['max']:.1f}\n")

    (TRACK / "DOWNSTREAM_PHASE.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"기록: {TRACK/'DOWNSTREAM_PHASE.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
