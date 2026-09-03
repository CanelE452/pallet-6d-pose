"""GATE 0B — adaptation 8,031장 전체에 R0 ROI 제안을 채운다.

    python3 scripts/self_training_yolo/depth_corrected/build_full_roi_cache.py \
        --output-dir data/pallet/results/paper_depth_selftrain_v1/gate0b

출력  R0_FULL_ADAPT_ROI_CACHE.json · R0_RECIPE_PARITY.json

기존 cache 가 덮는 프레임은 **무조건 재사용**하고, 없는 프레임만 같은 frozen
checkpoint 와 같은 recipe 로 추론한다.  recipe 는 기존
`R0_TEACHER_CACHE.json` 에서 읽는다 — 값을 여기서 다시 적지 않는다.

이건 pseudo-label manifest 가 아니다.  ROI 감사에 필요한 예측만 담는다.
teacher 의 정확도는 평가하지 않는다.

parity gate: 기존 cache 에 이미 있는 프레임을 새 경로로 다시 추론해 box ·
confidence · keypoint 차이를 잰다.  직렬화 오차로 설명되지 않으면 중단한다.
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
RAW = REPO_ROOT / "data/pallet/raw_data"
REFERENCE_CACHE = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/teacher_cache/R0_TEACHER_CACHE.json"
SITE_CACHE = REPO_ROOT / "data/pallet/results/paper_selftrain_site_v1/preflight/SITE_A_TEACHER_CACHE.json"
RECORDINGS = [
    ("day", "outside/capturepallet01"), ("day", "outside/capturepallet10"),
    ("day", "outside/capturepallet11"),
    ("night", "night/capturenight01"), ("night", "night/capturenight02"),
    ("night", "night/capturenight03"), ("night", "night/capturenight04"),
    ("night", "night/capturenight10"),
]
PARITY_SAMPLES = 40
# 직렬화 오차 한계.  좌표는 float 로 저장되므로 사실상 0 이어야 한다.
PARITY_TOLERANCE_PX = 1e-4
PARITY_TOLERANCE_CONF = 1e-6


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_instance(result, pad: int) -> dict | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    scores = result.boxes.conf.cpu().numpy()
    index = int(np.argmax(scores))
    box = result.boxes.xyxy.cpu().numpy()[index] - pad
    keypoints = result.keypoints.xy.cpu().numpy()[index] - pad
    if result.keypoints.conf is not None:
        confidences = result.keypoints.conf.cpu().numpy()[index]
    else:
        confidences = np.full(keypoints.shape[0], np.nan)
    corner_conf = np.asarray(confidences[:8], float)
    return {
        "box_xyxy": [float(v) for v in box],
        "box_conf": float(scores[index]),
        "keypoints_xy": [[float(x), float(y)] for x, y in keypoints],
        "keypoints_conf": [float(v) for v in confidences],
        "kp_conf_median8": float(np.nanmedian(corner_conf)),
        "n_instances": int(len(scores)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = json.loads(REFERENCE_CACHE.read_text())
    recipe = reference["recipe"]
    checkpoint = REPO_ROOT / reference["teacher_checkpoint"]
    declared_sha = reference["teacher_sha256"]
    actual_sha = sha256_file(checkpoint)
    if actual_sha != declared_sha:
        raise SystemExit(f"checkpoint sha mismatch: {actual_sha} != {declared_sha}")
    pad, imgsz = int(recipe["pad"]), int(recipe["imgsz"])
    conf_floor = float(recipe["confidence_floor"])
    print(f"recipe from {REFERENCE_CACHE.name}: pad {pad} imgsz {imgsz} conf {conf_floor}")

    existing: dict[str, dict] = {}
    for path in (REFERENCE_CACHE, SITE_CACHE):
        for entry in json.loads(path.read_text())["entries"]:
            existing.setdefault(entry["image_path"], entry)

    wanted: list[tuple[str, str]] = []
    for lighting, relative in RECORDINGS:
        for image in sorted((RAW / relative / "rgb").iterdir()):
            wanted.append((lighting, str(image.relative_to(REPO_ROOT))))
    print(f"population {len(wanted)}   existing cache covers "
          f"{sum(1 for _, p in wanted if p in existing)}")

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

    # ── parity gate: 기존 cache 프레임을 새 경로로 다시 추론
    shared = [p for _, p in wanted if p in existing and existing[p].get("top1")]
    positions = np.linspace(0, len(shared) - 1,
                            min(PARITY_SAMPLES, len(shared))).round().astype(int)
    box_deltas, conf_deltas, kp_deltas = [], [], []
    for index in positions:
        relative_path = shared[index]
        fresh = infer(relative_path)
        old = existing[relative_path]["top1"]
        if fresh is None:
            continue
        box_deltas.append(float(np.abs(np.array(fresh["box_xyxy"])
                                       - np.array(old["box_xyxy"])).max()))
        conf_deltas.append(abs(fresh["box_conf"] - old["box_conf"]))
        kp_deltas.append(float(np.abs(np.array(fresh["keypoints_xy"])
                                      - np.array(old["keypoints_xy"])).max()))
    parity = {
        "schema_version": "r0_recipe_parity_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "samples": len(box_deltas),
        "box_max_abs_delta_px": max(box_deltas) if box_deltas else None,
        "confidence_max_abs_delta": max(conf_deltas) if conf_deltas else None,
        "keypoint_max_abs_delta_px": max(kp_deltas) if kp_deltas else None,
        "tolerance_px": PARITY_TOLERANCE_PX,
        "tolerance_conf": PARITY_TOLERANCE_CONF,
        "checkpoint_sha256": actual_sha,
        "recipe_source": str(REFERENCE_CACHE.relative_to(REPO_ROOT)),
    }
    parity["passed"] = bool(
        box_deltas
        and max(box_deltas) <= PARITY_TOLERANCE_PX
        and max(kp_deltas) <= PARITY_TOLERANCE_PX
        and max(conf_deltas) <= PARITY_TOLERANCE_CONF)
    (out_dir / "R0_RECIPE_PARITY.json").write_text(json.dumps(parity, indent=2) + "\n")
    print(f"parity  box {parity['box_max_abs_delta_px']:.2e}  "
          f"kp {parity['keypoint_max_abs_delta_px']:.2e}  "
          f"conf {parity['confidence_max_abs_delta']:.2e}  -> "
          f"{'PASS' if parity['passed'] else 'FAIL'}")
    if not parity["passed"]:
        print("recipe parity 실패 — 중단")
        return 1

    # ── 전수 ROI
    frames, reused, inferred, detected = {}, 0, 0, 0
    for order, (lighting, relative_path) in enumerate(wanted, start=1):
        cached = existing.get(relative_path)
        if cached is not None:
            top = cached.get("top1")
            source = "existing_cache"
            reused += 1
        else:
            top = infer(relative_path)
            source = "new_inference"
            inferred += 1
        if top is not None:
            detected += 1
        frames[relative_path] = {
            "capture_session": Path(relative_path).parent.parent.name,
            "lighting": lighting,
            "roi_source": source,
            "top1": top,
        }
        if order % 1000 == 0:
            print(f"  {order}/{len(wanted)}  reused {reused}  new {inferred}", flush=True)

    by_recording: dict[str, dict] = {}
    for entry in frames.values():
        block = by_recording.setdefault(entry["capture_session"],
                                        {"frames": 0, "detected": 0,
                                         "reused": 0, "new": 0})
        block["frames"] += 1
        block["detected"] += entry["top1"] is not None
        block["reused" if entry["roi_source"] == "existing_cache" else "new"] += 1

    (out_dir / "R0_FULL_ADAPT_ROI_CACHE.json").write_text(json.dumps({
        "schema_version": "r0_full_adapt_roi_cache_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "this_is_not_a_pseudo_label_manifest": True,
        "purpose": "ROI proposals for the depth audit; teacher correctness is not evaluated",
        "teacher_checkpoint": reference["teacher_checkpoint"],
        "teacher_sha256": actual_sha,
        "recipe": recipe,
        "recipe_source": str(REFERENCE_CACHE.relative_to(REPO_ROOT)),
        "population": len(wanted),
        "reused_from_existing_cache": reused,
        "new_inference": inferred,
        "frames_with_detection": detected,
        "by_recording": by_recording,
        "frames": frames,
    }, indent=2) + "\n")

    print(f"\npopulation {len(wanted)}  reused {reused}  new {inferred}  "
          f"detected {detected}")
    print(f"{'recording':20}{'frames':>8}{'detected':>10}{'reused':>9}{'new':>8}")
    print("-" * 55)
    for name, block in sorted(by_recording.items()):
        print(f"{name:20}{block['frames']:8d}{block['detected']:10d}"
              f"{block['reused']:9d}{block['new']:8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
