"""1단계 (pallet-yolo26 env) — YOLO 3종의 키포인트를 덤프한다.

채점은 하지 않는다.  네 모델을 공정하게 비교하려면 downstream 이 한 벌이어야
하는데 yolo26 은 ultralytics 8.4.60, DOPE 계열은 8.0.120 환경이라 한 프로세스에
못 올린다.  그래서 각자 환경에서 **원본 픽셀 좌표의 9 키포인트**만 뽑고, 점수는
`mc_score.py` 한 곳에서 낸다.

추론 레시피는 release README 의 배포 계약 그대로다 — 지어내지 않았다:
    PAD=100 reflect_101, imgsz 640, conf 0.4, 최고신뢰 인스턴스, 좌표 -PAD
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

OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
PAD, IMGSZ, CONF = 100, 640, 0.4

MODELS = {
    "yolo26n_synth": "challenge/yolo_pose_one_model/runs/"
                     "stage_a_synth_640_b32_seed42/weights/best.pt",
    "yolo26n_ft": "challenge/yolo_pose_one_model/release/"
                  "pallet-pose-yolo26n-ft/pallet_yolo26n_pose_ft.pt",
    "yolo26m_ft": "challenge/yolo_pose_one_model/release/"
                  "pallet-pose-yolo26m-ft/pallet_yolo26m_pose_ft.pt",
    # 2026-08-26 반입 (Windows 학습, G38 negative 대조쌍).  같은 레시피로 덤프해야
    # 기존 4종과 한 표에 놓을 수 있다 — 카드의 자체 수치는 다른 셋·다른 자다.
    "Y0E": "challenge/yolo_pose_one_model/runs_neg_g38/Y0E/weights/best.pt",
    "YN": "challenge/yolo_pose_one_model/runs_neg_g38/YN/weights/best.pt",
    # 2026-08-30 3자 대조 — 전부 60ep / batch32 / seed42 / 같은 pretrained init.
    # 데이터 구성만 다르므로 차이를 데이터 탓으로 읽을 수 있다.
    "JOINT_G38_LEGACY_TEX": "challenge/yolo_pose_one_model/spatial_concat_scratch/"
                            "runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/"
                            "weights/best.pt",
    "G38_ONLY_60EP": "challenge/yolo_pose_one_model/runs_camera_facing_loss/"
                     "OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/best.pt",
}


def main():
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    rows = MF.frames()
    print(f"프레임 {len(rows)}장", flush=True)

    for name, rel in MODELS.items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"  {name}: 가중치 없음 -> 건너뜀 ({rel})", flush=True)
            continue
        model = YOLO(path, task="pose")
        dump, n_det = [], 0
        for key, sealed, jp, ip, _label in rows:
            image = cv2.imread(ip)
            padded = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD,
                                        cv2.BORDER_REFLECT_101)
            result = model.predict(padded, imgsz=IMGSZ, conf=CONF,
                                   verbose=False)[0]
            entry = {"set": key, "sealed": sealed,
                     "fid": os.path.splitext(os.path.basename(jp))[0],
                     "image": os.path.relpath(ip, ROOT),
                     "kps": None, "kp_conf": None, "box_conf": None}
            if result.boxes is not None and len(result.boxes):
                i = int(np.argmax(result.boxes.conf.cpu().numpy()))
                kps = result.keypoints.xy.cpu().numpy()[i] - PAD
                entry["kps"] = kps.tolist()
                entry["box_conf"] = float(result.boxes.conf.cpu().numpy()[i])
                if result.keypoints.conf is not None:
                    entry["kp_conf"] = \
                        result.keypoints.conf.cpu().numpy()[i].tolist()
                n_det += 1
            dump.append(entry)
        payload = {"model": name, "weights": rel,
                   "recipe": {"pad": PAD, "border": "REFLECT_101",
                              "imgsz": IMGSZ, "conf": CONF,
                              "instance": "highest box confidence",
                              "source": "release README deployment contract"},
                   "n_frames": len(dump), "n_detected": n_det, "frames": dump}
        target = os.path.join(OUT, f"kps_{name}.json")
        json.dump(payload, open(target, "w"), indent=1)
        print(f"  {name:16} 검출 {n_det}/{len(dump)} -> {os.path.basename(target)}",
              flush=True)


if __name__ == "__main__":
    main()
