"""인터넷 팔레트 10장에 DOPE(paper_s2_stageB) 추론 -> json.

`internet_pallet_yolo_ab.py` 가 좌측 패널을 이 json 에서 읽는다.  DOPE 는 `pallet-pose`,
YOLO26 은 `pallet-yolo26` env 라 한 프로세스에서 둘 다 못 돌리기 때문이다.

추론 배선은 `scripts/stage0/internet_pallet_infer.py` 의 것을 그대로 쓴다 (reflect-pad
belief 추론 -> belief_to_pred).  여기서 새 전처리를 만들면 비교가 오염된다.

Usage: <pallet-pose python> challenge/yolo_pose_one_model/scripts/internet_pallet_dope_dump.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "annotate"))
# internet_pallet_infer 는 `paper_s2_testset17_9filters` 를 stage0 바로 아래에서 찾는데,
# 그 파일은 stage0/paper_s2/ 로 옮겨졌다.  원본을 고치면 그 파일이 참조하는 다른 산출물의
# 재현성이 흔들리므로, 여기서 경로만 더해 준다.
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0", "paper_s2"))

import cv2                                                     # noqa: E402
import internet_pallet_infer as IPI                            # noqa: E402

SRC = os.path.join(ROOT, "data/pallet/raw_data/internet_pallet_data")
OUT = os.path.join(ROOT, "data/pallet/eval_results/internet_pallet_dope")


def main():
    os.makedirs(OUT, exist_ok=True)
    model = IPI.E.load_model(IPI.T.WEIGHTS, IPI.DEV)
    files = sorted(f for f in glob.glob(os.path.join(SRC, "*"))
                   if os.path.splitext(f)[1].lower() in
                   (".jpg", ".jpeg", ".png", ".webp"))
    out = {}
    for fp in files:
        name = os.path.basename(fp)
        img = cv2.imread(fp)
        if img is None:
            continue
        belief, geom, wh = IPI.M.infer_belief(model, img, IPI.DEV, IPI.PAD)
        pred8, pred_c, peaks, _ = IPI.M.belief_to_pred(
            belief, geom, wh, IPI.PAD, IPI.THRESH)
        n = int((~np.isnan(pred8[:, 0])).sum())
        det = [i for i in range(8) if not np.isnan(pred8[i, 0])]
        out[name] = {
            "pred8": [None if np.isnan(pred8[i, 0])
                      else [float(pred8[i, 0]), float(pred8[i, 1])]
                      for i in range(8)],
            "pred_c": (None if pred_c is None or np.isnan(pred_c[0])
                       else [float(pred_c[0]), float(pred_c[1])]),
            "n_det": n,
            # YOLO 의 box conf 자리에 놓을 값 — belief 최댓값이다.  정의가 다르므로
            # 두 모델의 이 숫자를 직접 비교하면 안 된다.
            "conf": float(max(peaks[i] for i in det)) if det else 0.0,
        }
        print(f"  {name[:44]:<46} det {n}/8  peak {out[name]['conf']:.2f}",
              flush=True)

    dst = os.path.join(OUT, "pred_dope.json")
    json.dump({"weights": os.path.relpath(IPI.T.WEIGHTS, ROOT), "pred": out},
              open(dst, "w"), indent=1, ensure_ascii=False)
    print(f"-> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
