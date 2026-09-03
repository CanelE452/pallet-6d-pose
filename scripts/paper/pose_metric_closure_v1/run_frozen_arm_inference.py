"""동결된 arm 하나를 정본 recipe 로 319장에 추론하고 raw 2D 예측을 저장한다.

    python3 scripts/paper/pose_metric_closure_v1/run_frozen_arm_inference.py --arm R0
    python3 scripts/paper/pose_metric_closure_v1/run_frozen_arm_inference.py --arm R5_PROPOSED

conda env 는 `pallet-yolo26` 이어야 한다 — `pallet-pose` 의 ultralytics 8.0.120 은
C3k2 가 없어 yolo26 가중치를 못 읽는다.

recipe 는 `INFERENCE_REPLAY_LOCK.json`, checkpoint 는 `POSE_ARM_CHECKPOINT_LOCK.json`
에서만 읽는다.  이 스크립트는 checkpoint 를 고르지 않고 epoch 을 다시 뽑지 않는다.

**pose 를 저장하지 않는다.**  box · box_conf · 9 keypoint xy · 9 keypoint conf 만
남긴다.  2D 예측이 source of truth 로 남아야 나중에 모델을 다시 돌리지 않고도
evaluator 를 재검증할 수 있다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
D = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
MANIFEST = D / "AXIS_REVIEW_MANIFEST.json"
RECIPE_LOCK = D / "INFERENCE_REPLAY_LOCK.json"
CKPT_LOCK = D / "POSE_ARM_CHECKPOINT_LOCK.json"
OUT_DIR = D / "predictions"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--limit", type=int, default=0, help="smoke 용, 앞 N 장만")
    args = parser.parse_args()

    recipe = json.loads(RECIPE_LOCK.read_text())
    if recipe.get("status") != "FROZEN":
        raise SystemExit("INFERENCE_REPLAY_LOCK is not FROZEN")
    spec = recipe["recipe"]
    checkpoints = json.loads(CKPT_LOCK.read_text())["arms"]
    if args.arm not in checkpoints:
        raise SystemExit(f"{args.arm} is not in POSE_ARM_CHECKPOINT_LOCK")
    entry = checkpoints[args.arm]
    weights = REPO_ROOT / entry["checkpoint"]
    actual = sha256_file(weights)
    if actual != entry["sha256"]:
        raise SystemExit(f"checkpoint sha mismatch for {args.arm}: "
                         f"{actual} != {entry['sha256']}")

    frames = json.loads(MANIFEST.read_text())["frames_list"]
    if args.limit:
        frames = frames[:args.limit]

    from ultralytics import YOLO

    model = YOLO(str(weights), task="pose")
    pad = int(spec["pad_px"])
    imgsz = int(spec["input_size"])
    conf = float(spec["confidence_floor"])

    predictions: dict[str, dict] = {}
    no_detection = 0
    for frame in frames:
        image = cv2.imread(str(REPO_ROOT / frame["image"]))
        if image is None:
            predictions[frame["frame_id"]] = {"status": "IMAGE_MISSING"}
            continue
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        result = model.predict(padded, conf=conf, imgsz=imgsz, augment=False,
                               half=False, device=spec["device"], verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            predictions[frame["frame_id"]] = {"status": "NO_DETECTION"}
            no_detection += 1
            continue
        scores = result.boxes.conf.detach().cpu().numpy()
        best = int(np.argmax(scores))                     # top-1 by box confidence
        box = result.boxes.xyxy.detach().cpu().numpy()[best] - pad
        keypoints = keypoint_conf = None
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.detach().cpu().numpy()[best] - pad
            if result.keypoints.conf is not None:
                keypoint_conf = result.keypoints.conf.detach().cpu().numpy()[best]
        predictions[frame["frame_id"]] = {
            "status": "OK",
            "box_xyxy": box.tolist(),
            "box_conf": float(scores[best]),
            "keypoints_xy": (keypoints.tolist() if keypoints is not None else None),
            "keypoints_conf": (keypoint_conf.tolist()
                               if keypoint_conf is not None else None),
            "detections": int(len(scores)),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.arm}.json"
    out.write_text(json.dumps({
        "schema_version": "frozen_arm_prediction_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "checkpoint": entry["checkpoint"],
        "checkpoint_sha256": actual,
        "recipe": spec,
        "recipe_lock_sha256": sha256_file(RECIPE_LOCK),
        "population_frame_order_sha256": recipe["population"]["frame_order_sha256"],
        "n_frames": len(frames),
        "no_detection": no_detection,
        "new_training": 0,
        "checkpoint_reselection": 0,
        "frames": predictions,
    }, indent=2) + "\n")
    print(f"{args.arm}  frames {len(frames)}  no_detection {no_detection}  -> "
          f"{out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
