"""AP 용 점수 덤프 — positive 정본 161 + real negative 2,689.

AP 는 pose 표와 **다른 덤프가 필요하다**.  `mc_dump_yolo.py` 는 배포 계약대로
conf 0.4 에서 자르므로 그 아래 프레임의 점수가 사라진다.  AP 는 곡선 전체를 쓰는
threshold-free 지표라, 여기서는 conf 0.001 로 훑고 프레임당 **최고 box conf 하나**만
남긴다.  검출이 아예 없으면 0 이다.

negative 는 `data/pallet/raw_data/negative_real_20260823/` 2,689 장.  detection AP 는
intrinsics 를 쓰지 않으므로, 이 셋의 camera_info K 가 640x480 전용이라는 문제
(memory `real-negative-set-20260823-for-ap`)는 여기 영향이 없다.

DOPE 계열은 담지 않는다.  `score_4kp` 는 box confidence 와 의미가 달라 같은 곡선에
올릴 수 없다.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_frames as MF  # noqa: E402
import mc_dump_yolo as MDY  # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
NEG_DIR = os.path.join(ROOT, "data/pallet/raw_data/negative_real_20260823")
PAD, IMGSZ, CONF = 100, 640, 0.001

EXTRA = {
    "yolo26n_paper_generic_v1":
        "challenge/yolo_pose_one_model/runs_paper/"
        "yolo26n_paper_generic_v1_seed42/weights/best.pt",
    "yolo26n_broad40k_5ep":
        "challenge/yolo_pose_one_model/runs_broad40k/"
        "b_yolo26n_broad40k_5ep/weights/best.pt",
}


def negative_images():
    out = []
    for base, _dirs, names in os.walk(NEG_DIR):
        if os.path.basename(base) == "depth":
            continue
        for name in sorted(names):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                out.append(os.path.join(base, name))
    return sorted(out)


def top_score(model, image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    padded = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD,
                                cv2.BORDER_REFLECT_101)
    result = model.predict(padded, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    if result.boxes is None or not len(result.boxes):
        return 0.0
    return float(result.boxes.conf.cpu().numpy().max())


def main():
    from ultralytics import YOLO
    models = dict(MDY.MODELS)
    models.update(EXTRA)

    pos = [ip for _k, _s, _jp, ip, _lab in MF.frames()]
    neg = negative_images()
    print(f"positive {len(pos)}장  negative {len(neg)}장", flush=True)

    payload = {"recipe": {"pad": PAD, "border": "REFLECT_101", "imgsz": IMGSZ,
                          "conf": CONF, "score": "max box confidence per frame",
                          "note": "threshold-free — AP 는 곡선 전체를 쓴다"},
               "n_positive": len(pos), "n_negative": len(neg), "models": {}}

    for name, rel in models.items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"  {name}: 가중치 없음 -> 건너뜀", flush=True)
            continue
        net = YOLO(path, task="pose")
        p = [top_score(net, ip) for ip in pos]
        n = [top_score(net, ip) for ip in neg]
        p = [x for x in p if x is not None]
        n = [x for x in n if x is not None]
        payload["models"][name] = {"weights": rel, "pos": p, "neg": n}
        print(f"  {name:26} pos median {np.median(p):.3f}  "
              f"neg p99 {np.percentile(n, 99):.3f}", flush=True)

    target = os.path.join(OUT, "AP_SCORES.json")
    json.dump(payload, open(target, "w"))
    print(f"-> {target}", flush=True)


if __name__ == "__main__":
    main()
