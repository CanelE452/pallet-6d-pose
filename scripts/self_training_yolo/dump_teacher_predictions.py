"""R0 teacher 를 U_MAIN 에 **한 번만** 돌려 예측 cache 를 만든다.

filter arm 마다 YOLO 를 다시 돌리지 않는다.  같은 teacher, 같은 이미지, 한 번의
inference 로 만든 cache 에서 모든 gate(F0~F4)를 뽑는다.  그래야 arm 간 차이가
"pseudo-label selection rule" 하나로만 남는다.

horizontal-flip consistency 도 여기서 같이 뜬다.  flip 은 별도 filter 가 아니라
같은 teacher 의 두 번째 forward 이므로, 나중에 다시 돌리면 cache 계약이 깨진다.

추론 레시피는 배포 계약 그대로다 — 지어내지 않았다.

    PAD 100  BORDER_REFLECT_101   imgsz 640   최고 box confidence 인스턴스   좌표 -PAD

단 confidence floor 만 배포값 0.4 가 아니라 evaluator 계약값 0.001 을 쓴다.
quality gate 를 `model.predict(conf=...)` 에서 미리 걸면 후단 threshold 선택이
분포를 못 보게 된다.  gate 는 전부 cache 이후에 적용한다.

    conf floor 0.001  <<  TAU_BOX 후보 0.70~0.85     (이중 threshold 아님)

flip 규약: 원본을 좌우반전 → 추론 → x 를 (W-1-x) 로 되돌림 → semantic pair swap.
pair 는 dataset 의 flip_idx 에서 읽어 온다 (여기서 새로 적지 않는다).

실행:  conda activate pallet-yolo26
    python scripts/self_training_yolo/dump_teacher_predictions.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

TEACHER = (
    REPO_ROOT / "challenge/yolo_pose_one_model/spatial_concat_scratch/runs"
    / "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt"
)
TEACHER_DATA_YAML = (
    REPO_ROOT / "challenge/yolo_pose_one_model/datasets"
    / "g38_legacy_v1v2_p0_tex20k/data.yaml"
)
POOL_CSV = (
    REPO_ROOT / "data/evaluation/pallet_eval_v1/adaptation/MAIN_UNLABELED_BALANCED.csv"
)
OUT_DIR = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/teacher_cache"

PAD = 100
IMGSZ = 640
CONF_FLOOR = 0.001
N_CORNERS = 8  # 0..7 = cuboid corners, 8 = centroid


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flip_permutation() -> list[int]:
    """dataset 의 flip_idx 를 그대로 쓴다.  좌우반전 시 semantic 짝을 바꾼다."""

    data = yaml.safe_load(TEACHER_DATA_YAML.read_text())
    flip_idx = data.get("flip_idx")
    if not isinstance(flip_idx, list) or len(flip_idx) != 9:
        raise SystemExit(f"BAD_FLIP_IDX: {flip_idx!r}")
    if sorted(flip_idx) != list(range(9)):
        raise SystemExit(f"FLIP_IDX_NOT_A_PERMUTATION: {flip_idx!r}")
    return [int(value) for value in flip_idx]


def _camera_matrix(image_path: Path) -> list[list[float]] | None:
    """capture 세션의 cam_K.txt.  rgb/ 의 부모가 세션 폴더다."""

    cam_k = image_path.parent.parent / "cam_K.txt"
    if not cam_k.exists():
        return None
    values = [float(token) for token in cam_k.read_text().split()]
    if len(values) != 9:
        raise SystemExit(f"BAD_CAM_K: {cam_k}")
    return [values[0:3], values[3:6], values[6:9]]


def _top_instance(result) -> tuple[int, int] | tuple[None, int]:
    if result.boxes is None or len(result.boxes) == 0:
        return None, 0
    confidences = result.boxes.conf.cpu().numpy()
    return int(np.argmax(confidences)), int(len(confidences))


def _extract(result, index: int) -> dict:
    box = result.boxes.xyxy.cpu().numpy()[index] - PAD
    keypoints = result.keypoints.xy.cpu().numpy()[index] - PAD
    if result.keypoints.conf is not None:
        kp_conf = result.keypoints.conf.cpu().numpy()[index]
    else:
        kp_conf = np.full(keypoints.shape[0], np.nan)
    return {
        "box_xyxy": [float(value) for value in box],
        "box_conf": float(result.boxes.conf.cpu().numpy()[index]),
        "keypoints_xy": [[float(x), float(y)] for x, y in keypoints],
        "keypoints_conf": [float(value) for value in kp_conf],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="smoke 용 상한 (0=전체)")
    parser.add_argument("--out-name", default="R0_TEACHER_CACHE.json")
    parser.add_argument("--device", default="0")
    # 아래 둘은 기본값이 기존 경로라 기존 동작은 그대로다.  다른 pool 에 같은
    # teacher·같은 recipe 를 적용할 때만 넘긴다.
    parser.add_argument("--pool-csv", default=str(POOL_CSV))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    from ultralytics import YOLO
    import torch
    import ultralytics

    flip_perm = _flip_permutation()
    pool_csv = Path(args.pool_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    rows = list(csv.DictReader(pool_csv.open(encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    print(f"pool rows {len(rows)}   flip_idx {flip_perm}", flush=True)

    teacher_sha = sha256_file(TEACHER)
    model = YOLO(str(TEACHER), task="pose")

    entries: list[dict] = []
    n_detected = 0
    n_flip_detected = 0
    started = time.time()
    for position, row in enumerate(rows):
        image_path = REPO_ROOT / row["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"UNREADABLE_IMAGE: {image_path}")
        height, width = image.shape[:2]

        padded = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        result = model.predict(padded, imgsz=IMGSZ, conf=CONF_FLOOR,
                               device=args.device, verbose=False)[0]
        index, n_instances = _top_instance(result)

        # 두 번째 forward: 좌우반전본.  같은 teacher, 같은 레시피.
        flipped = cv2.flip(image, 1)
        flipped_padded = cv2.copyMakeBorder(flipped, PAD, PAD, PAD, PAD,
                                            cv2.BORDER_REFLECT_101)
        flip_result = model.predict(flipped_padded, imgsz=IMGSZ, conf=CONF_FLOOR,
                                    device=args.device, verbose=False)[0]
        flip_index, n_flip_instances = _top_instance(flip_result)

        entry: dict = {
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "capture_session": row["capture_session"],
            "paper_condition": row["paper_condition"],
            "image_width": int(width),
            "image_height": int(height),
            "camera_matrix": _camera_matrix(image_path),
            "n_instances": n_instances,
            "n_flip_instances": n_flip_instances,
            "top1": None,
            "flip_top1": None,
        }

        if index is not None:
            top = _extract(result, index)
            corner_conf = np.asarray(top["keypoints_conf"][:N_CORNERS], dtype=float)
            top["kp_conf_min8"] = float(np.nanmin(corner_conf))
            top["kp_conf_median8"] = float(np.nanmedian(corner_conf))
            top["centroid_conf"] = float(top["keypoints_conf"][N_CORNERS])
            entry["top1"] = top
            n_detected += 1

        if flip_index is not None:
            flip_top = _extract(flip_result, flip_index)
            # x 되돌리기 -> semantic pair swap.  둘 중 하나만 하면 조용히 틀린다.
            unflipped = [
                [float(width - 1 - x), float(y)] for x, y in flip_top["keypoints_xy"]
            ]
            flip_conf = flip_top["keypoints_conf"]
            flip_top["keypoints_xy"] = [unflipped[flip_perm[i]] for i in range(9)]
            flip_top["keypoints_conf"] = [flip_conf[flip_perm[i]] for i in range(9)]
            x1, y1, x2, y2 = flip_top["box_xyxy"]
            flip_top["box_xyxy"] = [
                float(width - 1 - x2), float(y1), float(width - 1 - x1), float(y2)
            ]
            entry["flip_top1"] = flip_top
            n_flip_detected += 1

        entries.append(entry)
        if (position + 1) % 100 == 0:
            rate = (position + 1) / (time.time() - started)
            print(f"  {position + 1}/{len(rows)}  det {n_detected}  "
                  f"flip_det {n_flip_detected}  {rate:.1f} img/s", flush=True)

    payload = {
        "schema_version": "paper_teacher_prediction_cache_v1",
        "teacher_checkpoint": str(TEACHER.relative_to(REPO_ROOT)),
        "teacher_sha256": teacher_sha,
        "pool_manifest": str(pool_csv.resolve().relative_to(REPO_ROOT)),
        "pool_manifest_sha256": sha256_file(pool_csv),
        "recipe": {
            "pad": PAD,
            "border": "BORDER_REFLECT_101",
            "imgsz": IMGSZ,
            "confidence_floor": CONF_FLOOR,
            "instance": "highest box confidence",
            "source": (
                "release README deployment contract; confidence floor lowered to the "
                "evaluator contract value so pseudo-label gates see the full "
                "distribution instead of a pre-applied 0.4 cut"
            ),
        },
        "flip_contract": {
            "flip_idx": flip_perm,
            "source": str(TEACHER_DATA_YAML.relative_to(REPO_ROOT)),
            "unflip": "x -> width - 1 - x, then semantic pair swap by flip_idx",
        },
        "environment": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
        },
        "n_images": len(entries),
        "n_detected": n_detected,
        "n_flip_detected": n_flip_detected,
        "entries": entries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / args.out_name
    target.write_text(json.dumps(payload) + "\n")
    print(f"\nwrote {target.relative_to(REPO_ROOT)}  "
          f"{target.stat().st_size / 1e6:.1f} MB")
    print(f"detected {n_detected}/{len(entries)}   flip {n_flip_detected}/{len(entries)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
