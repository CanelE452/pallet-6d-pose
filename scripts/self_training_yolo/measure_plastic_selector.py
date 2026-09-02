"""현재 checkpoint 로 plastic prediction-only W/D selector 를 다시 잰다.

왜 하는가
    pose metric 이 BLOCKED 인 이유 중 하나가 이 selector 다.  그런데 저장된 진단은
    **옛 checkpoint**(OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42)로 잰 것이다.
    self-training 이 keypoint 를 개선했다면 selector 도 나아졌을 수 있고, 그러면
    사람이 프레임을 열지 않고도 pose 가 열린다.  그래서 사람 작업을 시키기 전에
    먼저 이걸 잰다.

GT 누수 방지
    selector 는 예측 keypoint · camera intrinsics · registry 치수 · 동결된 config
    네 가지만 받는다.  GT axis assignment 는 **선택이 전부 끝난 뒤** parity 확인에만
    쓴다.  `select_pnp_hypotheses` 자체가 GT 를 인자로 받지 않는다.

population 은 고를 수 없다
    `SELECTOR_DIAGNOSTIC_POPULATION_BY_OBJECT` 가 plastic -> DEV_POS140 으로
    사전등록돼 있다.  결과가 좋아 보이는 population 으로 바꾸지 않는다.

실행:  conda activate pallet-yolo26
    python scripts/self_training_yolo/measure_plastic_selector.py --model R5_PROPOSED
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "challenge"))
sys.path.insert(0, str(REPO_ROOT))

from evaluation_v2 import pnp_selector as PS  # noqa: E402
from evaluation_v2.real_dataset_contract import (  # noqa: E402
    PopulationId,
    load_population_manifest,
    manifest_path,
)

OUT_DIR = REPO_ROOT / "challenge/evaluation_v2/selector_diagnostic"
RUNS = REPO_ROOT / "challenge/yolo_pose_one_model/paper_selftrain_v1"

MODELS = {
    "R0": REPO_ROOT / (
        "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
        "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt"
    ),
    "R5_PROPOSED": RUNS / "R5_PROPOSED__FULL/weights/last.pt",
}

PAD, IMGSZ, CONF_FLOOR = 100, 640, 0.001
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry_dimensions(object_type: str) -> dict:
    for entry in json.loads(REGISTRY.read_text())["objects"]:
        if entry["object_type"] == object_type:
            return entry["physical_dimensions_m"]
    raise SystemExit(f"OBJECT_TYPE_NOT_IN_REGISTRY: {object_type}")


def expected_hypothesis(payload: dict, dimensions: dict) -> str | None:
    """GT 의 정답 parity.  **선택이 끝난 뒤에만** 읽는다.

    legacy GT 는 `selected_hypothesis` 문자열을 갖고 있지 않고 대신
    `camera_facing_pnp.dimensions_m.width` 로 어느 면이 카메라를 향하는지 담는다.
    이름은 그 width 가 물체의 짧은 변인지 긴 변인지로 정해진다 (pallet_geometry 의
    camera_facing_hypothesis_name 과 같은 규칙).
    """

    facing = payload["objects"][0].get("camera_facing_pnp") or {}
    width = (facing.get("dimensions_m") or {}).get("width")
    if width is None:
        return None
    short = min(float(dimensions["x"]), float(dimensions["z"]))
    long_side = max(float(dimensions["x"]), float(dimensions["z"]))
    if abs(float(width) - short) <= abs(float(width) - long_side):
        return "short-face-front"
    return "long-face-front"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="R5_PROPOSED", choices=sorted(MODELS))
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    weights = MODELS[args.model]
    if not weights.exists():
        raise SystemExit(f"CHECKPOINT_MISSING: {weights}")

    population_id = PopulationId.DEV_POS140
    manifest = load_population_manifest(manifest_path(population_id))
    dimensions = registry_dimensions(PS.PLASTIC_OBJECT_TYPE)
    config = PS.SelectorConfig()
    model = YOLO(str(weights), task="pose")
    print(f"{args.model}  population {population_id.value} n={manifest.count}", flush=True)

    records: list[dict] = []
    for index, item in enumerate(manifest.items):
        image = cv2.imread(str(REPO_ROOT / item.image))
        if image is None:
            raise SystemExit(f"IMAGE_DECODE_FAILED: {item.image}")
        payload = json.loads((REPO_ROOT / item.label).read_text())
        intrinsics = payload["camera_data"]["intrinsics"]
        camera = np.array([[intrinsics["fx"], 0.0, intrinsics["cx"]],
                           [0.0, intrinsics["fy"], intrinsics["cy"]],
                           [0.0, 0.0, 1.0]], dtype=float)

        padded = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        result = model.predict(padded, imgsz=IMGSZ, conf=CONF_FLOOR,
                               device=args.device, verbose=False)[0]

        record = {
            "frame_id": item.frame_id,
            "domain": item.domain or "DAY",
            "session": item.session_id or item.source_set,
            "detection_count": 0 if result.boxes is None else int(len(result.boxes)),
            "top_score": None,
            "selector_status": "NO_DETECTION",
            "selected_hypothesis": None,
            "expected_hypothesis": expected_hypothesis(payload, dimensions),
            "correct": False,
            "prediction_failure": "NO_DETECTION",
        }

        if result.boxes is not None and len(result.boxes):
            best = int(np.argmax(result.boxes.conf.cpu().numpy()))
            record["top_score"] = float(result.boxes.conf.cpu().numpy()[best])
            keypoints = result.keypoints.xy.cpu().numpy()[best] - PAD
            # selector 는 예측 keypoint · intrinsics · registry 치수 · 동결 config 만 받는다.
            selection = PS.select_pnp_hypotheses(keypoints, camera, dimensions, config)
            record["selector_status"] = getattr(
                selection.status, "value", str(selection.status)
            )
            name = selection.selected_hypothesis
            record["selected_hypothesis"] = name
            record["ambiguity"] = selection.ambiguity
            record["prediction_failure"] = None
            # ↓ 여기서부터가 parity 확인.  선택은 이미 끝났다.
            record["correct"] = bool(
                name is not None
                and record["expected_hypothesis"] is not None
                and name == record["expected_hypothesis"]
            )
        records.append(record)
        if (index + 1) % 40 == 0:
            print(f"  {index + 1}/{manifest.count}", flush=True)

    report = PS.assess_selector_diagnostics(
        records, object_type=PS.PLASTIC_OBJECT_TYPE,
        population_id=population_id.value,
    )
    gate = report.to_dict() if hasattr(report, "to_dict") else dict(report.__dict__)

    payload = {
        "schema_version": "pallet_pose_selector_diagnostic_v1",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             cwd=REPO_ROOT).decode().strip(),
        "role": "DEV_DIAGNOSTIC_NOT_FINAL",
        "population": {
            "population_id": population_id.value,
            "count": manifest.count,
            "membership_sha256": manifest.membership_sha256,
            "manifest": str(manifest_path(population_id).relative_to(REPO_ROOT)),
        },
        "checkpoint": {
            "model": args.model,
            "path": str(weights.relative_to(REPO_ROOT)),
            "sha256": sha256_file(weights),
        },
        "inference_recipe": {
            "top_candidate_rule": "highest box confidence per frame",
            "pad": PAD, "border": "BORDER_REFLECT_101", "imgsz": IMGSZ,
            "confidence_floor": CONF_FLOOR, "device": args.device,
        },
        "gt_leakage_contract": {
            "selector_inputs": ["predicted_9_keypoints", "camera_intrinsics",
                                "fixed_physical_dimensions", "frozen_selector_config"],
            "forbidden": ["GT dimensions_m", "GT pose", "GT axis assignment",
                          "GT keypoint error", "session prior"],
            "comparison_phase": "GT parity read only after all selection decisions complete",
        },
        "gate": gate,
        "records": records,
    }
    target = OUT_DIR / f"PLASTIC_SELECTOR_DIAGNOSTIC__{args.model}.json"
    target.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str) + "\n")

    correct = sum(r["correct"] for r in records)
    night = [r for r in records if r["domain"] == "NIGHT"]
    print(f"\n  overall {correct}/{len(records)} = {correct / len(records):.4f}"
          f"   (gate >= {PS.OVERALL_AXIS_ACCURACY_MIN})")
    if night:
        n_ok = sum(r["correct"] for r in night)
        print(f"  night   {n_ok}/{len(night)} = {n_ok / len(night):.4f}"
              f"   (gate >= {PS.NIGHT_AXIS_ACCURACY_MIN})")
    print(f"  gate status: {gate.get('status')}   reason: {gate.get('blocked_reason')}")
    print(f"  wrote {target.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
