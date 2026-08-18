"""internet_pallet_kps_dump.py — YOLO26 keypoint 를 JSON 으로 덤프(진단용).

`internet_pallet_yolo_ab.py` 의 predict() 와 완전히 같은 전처리(reflect pad 100,
conf>=0.5)를 쓴다. PnP 진단은 ultralytics 가 필요 없으므로, 검출을 한 번만 돌려
저장해 두고 이후 분석은 pallet-pose env 에서 numpy/cv2 만으로 반복한다.

사용: <pallet-yolo26 python> .../internet_pallet_kps_dump.py [--pattern 1016]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "challenge", "yolo_pose_one_model", "scripts"))
import internet_pallet_yolo_ab as AB  # noqa: E402

OUT_JSON = os.path.join(ROOT, "data/pallet/eval_results/internet_pallet_pnp_diag/kps.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="1016")
    ap.add_argument("--weights", default=AB.MODELS[0][1])
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(os.path.join(ROOT, args.weights), task="pose")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    records = {}
    for fp in sorted(glob.glob(os.path.join(AB.SRC, "*"))):
        name = os.path.basename(fp)
        if args.pattern not in name:
            continue
        img = cv2.imread(fp)
        if img is None:
            continue
        pred8, pred_c, conf = AB.predict(model, img)
        dims, hdef = AB.parse_dims_m(name)
        records[name] = {
            "path": fp,
            "shape": [int(img.shape[0]), int(img.shape[1])],
            "dims_from_filename_m": list(dims),
            "height_is_default": bool(hdef),
            "box_conf": conf,
            "kps8": [[None, None] if not np.isfinite(pred8[i, 0])
                     else [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)],
            "centroid": pred_c,
            "weights": args.weights,
            "pad": AB.PAD,
        }
        print(f"{name}  det={int(np.isfinite(pred8[:,0]).sum())}/8  conf={conf:.3f}")

    with open(OUT_JSON, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[out] {OUT_JSON}")


if __name__ == "__main__":
    main()
