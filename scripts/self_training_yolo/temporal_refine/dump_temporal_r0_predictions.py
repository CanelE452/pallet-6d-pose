"""TEMPORAL PILOT — tracklet 전 프레임에 R0 예측을 채운다.  parity gate 포함.

    python3 scripts/self_training_yolo/temporal_refine/dump_temporal_r0_predictions.py \
        --output-dir data/pallet/results/paper_temporal_selftrain_v1/pilot

출력  R0_TEMPORAL_TEACHER_CACHE.json · R0_TEMPORAL_PARITY.json

recipe·checkpoint 는 기존 `R0_TEACHER_CACHE.json` 에서 읽는다.  값을 다시 적지
않는다.  이미 캐시에 있는 프레임은 재사용하고 없는 것만 추론한다.

parity: 기존 캐시가 가진 프레임을 새 경로로 다시 추론해 box·conf·keypoint 가
직렬화 오차 안에서 같은지 확인한다.  어긋나면 중단한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json"
SITE_CACHE = REPO_ROOT / "data/pallet/results/paper_selftrain_site_v1/preflight/SITE_A_TEACHER_CACHE.json"
FULL_ROI = REPO_ROOT / "data/pallet/results/paper_depth_selftrain_v1/gate0b/R0_FULL_ADAPT_ROI_CACHE.json"
PARITY_SAMPLES = 30
TOLERANCE_PX = 1e-4
TOLERANCE_CONF = 1e-6


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_instance(result, pad: int) -> dict | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    scores = result.boxes.conf.cpu().numpy()
    index = int(np.argmax(scores))
    box = result.boxes.xyxy.cpu().numpy()[index] - pad
    keypoints = result.keypoints.xy.cpu().numpy()[index] - pad
    confidences = (result.keypoints.conf.cpu().numpy()[index]
                   if result.keypoints.conf is not None
                   else np.full(keypoints.shape[0], np.nan))
    return {
        "box_xyxy": [float(v) for v in box],
        "box_conf": float(scores[index]),
        "keypoints_xy": [[float(x), float(y)] for x, y in keypoints],
        "keypoints_conf": [float(v) for v in confidences],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()

    reference = json.loads(REFERENCE.read_text())
    recipe = reference["recipe"]
    checkpoint = REPO_ROOT / reference["teacher_checkpoint"]
    actual = sha256_file(checkpoint)
    if actual != reference["teacher_sha256"]:
        raise SystemExit(f"checkpoint sha mismatch: {actual}")
    pad, imgsz = int(recipe["pad"]), int(recipe["imgsz"])
    conf_floor = float(recipe["confidence_floor"])
    print(f"recipe: pad {pad}  imgsz {imgsz}  conf {conf_floor}  sha {actual[:12]}")

    existing: dict[str, dict] = {}
    for path in (REFERENCE, SITE_CACHE):
        for entry in json.loads(path.read_text())["entries"]:
            if entry.get("top1"):
                existing.setdefault(entry["image_path"], entry["top1"])
    if FULL_ROI.exists():
        for key, entry in json.loads(FULL_ROI.read_text())["frames"].items():
            if entry.get("top1"):
                existing.setdefault(key, entry["top1"])
    print(f"existing predictions available for {len(existing)} frames")

    wanted: set[str] = set()
    for row in csv.DictReader((out_dir / "TEMPORAL_PILOT_POPULATION.csv").open()):
        if row["eligible"] != "True":
            continue
        wanted.add(row["center_rgb"])
        wanted.update(p for p in row["neighbor_rgb_paths"].split("|") if p)
    wanted = sorted(wanted)
    print(f"tracklet frames needed {len(wanted)}  "
          f"already cached {sum(1 for p in wanted if p in existing)}")

    from ultralytics import YOLO
    model = YOLO(str(checkpoint), task="pose")

    def infer(relative_path: str) -> dict | None:
        image = cv2.imread(str(REPO_ROOT / relative_path))
        if image is None:
            return None
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        result = model.predict(padded, imgsz=imgsz, conf=conf_floor,
                               device=args.device, verbose=False)[0]
        return top_instance(result, pad)

    shared = [p for p in wanted if p in existing]
    positions = np.linspace(0, len(shared) - 1,
                            min(PARITY_SAMPLES, max(len(shared), 1))).round().astype(int)
    box_d, conf_d, kp_d = [], [], []
    for index in positions:
        if not shared:
            break
        relative_path = shared[int(index)]
        fresh = infer(relative_path)
        old = existing[relative_path]
        if fresh is None:
            continue
        box_d.append(float(np.abs(np.array(fresh["box_xyxy"]) - np.array(old["box_xyxy"])).max()))
        conf_d.append(abs(fresh["box_conf"] - old["box_conf"]))
        kp_d.append(float(np.abs(np.array(fresh["keypoints_xy"])
                                 - np.array(old["keypoints_xy"])).max()))
    parity = {
        "schema_version": "r0_temporal_parity_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": actual,
        "recipe_source": str(REFERENCE.relative_to(REPO_ROOT)),
        "samples": len(box_d),
        "box_max_abs_delta_px": max(box_d) if box_d else None,
        "confidence_max_abs_delta": max(conf_d) if conf_d else None,
        "keypoint_max_abs_delta_px": max(kp_d) if kp_d else None,
        "tolerance_px": TOLERANCE_PX, "tolerance_conf": TOLERANCE_CONF,
    }
    parity["passed"] = bool(box_d and max(box_d) <= TOLERANCE_PX
                            and max(kp_d) <= TOLERANCE_PX
                            and max(conf_d) <= TOLERANCE_CONF)
    (out_dir / "R0_TEMPORAL_PARITY.json").write_text(json.dumps(parity, indent=2) + "\n")
    print(f"parity  box {parity['box_max_abs_delta_px']:.2e}  "
          f"kp {parity['keypoint_max_abs_delta_px']:.2e}  "
          f"conf {parity['confidence_max_abs_delta']:.2e}  "
          f"-> {'PASS' if parity['passed'] else 'FAIL'}")
    if not parity["passed"]:
        return 1

    frames, reused, new, detected = {}, 0, 0, 0
    for order, relative_path in enumerate(wanted, start=1):
        top = existing.get(relative_path)
        if top is not None:
            reused += 1
            source = "existing_cache"
        else:
            top = infer(relative_path)
            new += 1
            source = "new_inference"
        if top is not None:
            detected += 1
        frames[relative_path] = {"top1": top, "source": source}
        if order % 200 == 0:
            print(f"  {order}/{len(wanted)}", flush=True)

    (out_dir / "R0_TEMPORAL_TEACHER_CACHE.json").write_text(json.dumps({
        "schema_version": "r0_temporal_teacher_cache_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_checkpoint": reference["teacher_checkpoint"],
        "teacher_sha256": actual,
        "recipe": recipe,
        "recipe_source": str(REFERENCE.relative_to(REPO_ROOT)),
        "frames": len(wanted), "reused": reused, "new_inference": new,
        "with_detection": detected,
        "this_is_not_a_pseudo_label_manifest": True,
        "predictions": frames,
    }, indent=2) + "\n")
    print(f"\nframes {len(wanted)}  reused {reused}  new {new}  detected {detected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
