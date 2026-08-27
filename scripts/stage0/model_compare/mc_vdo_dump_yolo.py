"""논문 트랙 YOLO 로 배포영상 4 프레임의 키포인트만 덤프한다 (pallet-yolo26 env).

여기서 PnP 도 그림도 하지 않는다.  yolo26 은 ultralytics 8.4.60, DOPE 계열은
8.0.120 이라 한 프로세스에 못 올린다 -- `mc_dump_yolo.py` 가 이미 쓰는 분업을
그대로 따른다.  좌표만 넘기고 pose 와 그림은 `mc_vdo_draw.py` 가 **FINAL40K 와
같은 경로**로 처리해야 비교가 성립한다.

논문 트랙이란: 학습 데이터가 `datasets/broad40k/data.yaml` 뿐이라는 뜻이다.
FINAL40K(SPLIT_LATE)와 같은 BROAD 합성 40,000 이고 real 은 0 장이다.  challenge
트랙의 `*_ft` 가중치(real 파인튜닝)는 여기 넣지 않는다 -- 섞으면 "같은 데이터에서
두 아키텍처" 라는 비교 자체가 깨진다.

추론 레시피는 release README 의 배포 계약 그대로: PAD=100 reflect_101, imgsz 640,
conf 0.4, 최고신뢰 인스턴스, 좌표 -PAD.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
SRC = os.path.join(ROOT, "data/pallet/raw_data/vdoframes")
OUT = os.path.join(ROOT, "data/pallet/results/model_compare/vdo_infer")
FRAMES = ["000037", "000076", "000483", "001678"]
PAD, IMGSZ, CONF = 100, 640, 0.4

MODELS = {
    "yolo26n_paper_generic_v1":
        "challenge/yolo_pose_one_model/runs_paper/"
        "yolo26n_paper_generic_v1_seed42/weights/last.pt",
    "yolo26n_broad40k_5ep":
        "challenge/yolo_pose_one_model/runs_broad40k/"
        "b_yolo26n_broad40k_5ep/weights/last.pt",
}


def main():
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)

    for name, relative in MODELS.items():
        path = os.path.join(ROOT, relative)
        if not os.path.exists(path):
            print(f"  {name}: 가중치 없음 -> 건너뜀 ({relative})", flush=True)
            continue
        model = YOLO(path, task="pose")
        frames = []
        for stem in FRAMES:
            image_path = os.path.join(SRC, f"{stem}.png")
            image = cv2.imread(image_path)
            padded = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD,
                                        cv2.BORDER_REFLECT_101)
            result = model.predict(padded, imgsz=IMGSZ, conf=CONF,
                                   verbose=False)[0]
            entry = {"frame": stem, "shape": list(image.shape[:2]),
                     "kps": None, "kp_conf": None, "box_conf": None,
                     "n_instances": 0}
            if result.boxes is not None and len(result.boxes):
                confidences = result.boxes.conf.cpu().numpy()
                best = int(np.argmax(confidences))
                entry["kps"] = (result.keypoints.xy.cpu().numpy()[best]
                                - PAD).tolist()
                entry["box_conf"] = float(confidences[best])
                entry["n_instances"] = int(len(confidences))
                if result.keypoints.conf is not None:
                    entry["kp_conf"] = \
                        result.keypoints.conf.cpu().numpy()[best].tolist()
            frames.append(entry)
            print(f"  {name:26} {stem}  box_conf "
                  f"{entry['box_conf'] if entry['box_conf'] is None else round(entry['box_conf'], 3)}"
                  f"  instances {entry['n_instances']}", flush=True)

        target = os.path.join(OUT, f"vdo_kps_{name}.json")
        json.dump({"model": name, "weights": relative,
                   "train_data": "datasets/broad40k/data.yaml (BROAD 합성 40,000, real 0장)",
                   "recipe": {"pad": PAD, "border": "REFLECT_101", "imgsz": IMGSZ,
                              "conf": CONF, "instance": "highest box confidence",
                              "source": "release README deployment contract"},
                   "frames": frames},
                  open(target, "w"), indent=1, ensure_ascii=False)
        print(f"  -> {os.path.relpath(target, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
