"""동결 teacher 를 DEV_EVAL 319 장에 추론한다 — 학습 0, checkpoint 선택 0.

    python run_teacher_inference.py --teacher T1_YOLOV8N_G38
    python run_teacher_inference.py --all

recipe 와 checkpoint 는 TEACHER_REGISTRY.json 에서만 읽고, sha 가 다르면 죽는다.
출력은 `frozen_arm_prediction_v1` 스키마 그대로라 기존 pose evaluator 가 바로 먹는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M
import mtcd_teachers as T


def run_one(tid: str, spec: dict, frames: list[dict], limit: int = 0) -> Path:
    weights = M.REPO_ROOT / spec["checkpoint"]
    actual = M.sha256_file(weights)
    if actual != spec["sha256"]:
        raise SystemExit(f"{tid}: checkpoint sha mismatch {actual} != {spec['sha256']}")
    kind = spec["kind"]
    model = T.load_dope(weights) if kind == "dope" else T.load_yolo(weights)
    todo = frames[:limit] if limit else frames

    predictions, no_detection, failures = {}, 0, []
    started = time.time()
    for frame in todo:
        image = cv2.imread(str(M.REPO_ROOT / frame["image"]))
        if image is None:
            predictions[frame["frame_id"]] = {"status": "IMAGE_MISSING"}
            failures.append({"frame_id": frame["frame_id"], "stage": "read",
                             "type": "ImageMissing", "message": frame["image"]})
            continue
        try:
            out = (T.infer_dope(model, image) if kind == "dope"
                   else T.infer_yolo(model, image))
        except Exception as exc:                      # 조용히 baseline 으로 떨어지지 않는다
            failures.append({"frame_id": frame["frame_id"], "stage": "infer",
                             "type": type(exc).__name__, "message": str(exc)[:300]})
            predictions[frame["frame_id"]] = {"status": "INFERENCE_ERROR"}
            continue
        if out.get("status") != "OK":
            no_detection += 1
        predictions[frame["frame_id"]] = out
    elapsed = time.time() - started

    M.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    out_path = M.PREDICTIONS / f"{tid}.json"
    out_path.write_text(json.dumps({
        "schema_version": "frozen_arm_prediction_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arm": tid,
        "teacher_kind": kind,
        "architecture": spec["architecture"],
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": actual,
        "recipe": {"pad_px": T.PAD_PX, "border": "BORDER_REFLECT_101",
                   "imgsz": T.DOPE_IMGSZ if kind == "dope" else T.YOLO_IMGSZ,
                   "confidence_floor": (T.DOPE_THRESHOLD if kind == "dope"
                                        else T.YOLO_CONF)},
        "population_frame_order_sha256":
            json.loads(M.RECIPE_LOCK_PATH.read_text())["population"]["frame_order_sha256"],
        "n_frames": len(todo),
        "no_detection": no_detection,
        "failures": failures,
        "seconds": round(elapsed, 1),
        "new_training": 0,
        "checkpoint_reselection": 0,
        "frames": predictions,
    }, indent=2) + "\n")
    print(f"{tid:26} frames {len(todo):4d}  no_detection {no_detection:3d}  "
          f"failures {len(failures):3d}  {elapsed:6.1f}s  -> {out_path.name}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    registry = json.loads((M.TRACK / "TEACHER_REGISTRY.json").read_text())["teachers"]
    frames = M.dev_eval_frames()
    targets = list(registry) if args.all else [args.teacher]
    for tid in targets:
        if tid not in registry:
            raise SystemExit(f"{tid} is not in TEACHER_REGISTRY")
        run_one(tid, registry[tid], frames, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
