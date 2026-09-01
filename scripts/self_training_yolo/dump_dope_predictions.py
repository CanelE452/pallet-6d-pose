"""DOPE 예측을 PAPER_EVAL 전체(positive+negative)에 대해 덤프한다.

목적은 DOPE 를 **같은 evaluator·같은 population·같은 metric** 으로 M1 에 넣는 것이다.
별도 채점기를 만들면 M1 의 행들이 서로 다른 자로 잰 값이 되어 비교가 성립하지 않는다.
그래서 여기서는 예측만 뽑고, 채점은 `paper_real_eval.py --predictions` 가 한다.

## 추론 규약 — 지어내지 않았다

DOPE 는 이 저장소의 정본 함수를 그대로 쓴다.
`scripts/stage0/selftrain/s1_cad_9filters.py` 의 `infer_belief` / `belief_to_pred`,
그리고 `paper_s2_real_eval` 의 `load_model` / `belief_to_orig_pad` 다.

    PAD 100 reflect  →  학습 전처리  →  belief(9,H,W)  →  원본 픽셀 좌표

reflect-padding 은 선택이 아니다.  plain squash 로 추론하면 truncation·근접에서
체계적으로 과소검출되어 모델을 부당하게 나쁘게 만든다 (이 저장소에서 2회 교정된 함정).

## YOLO 와 비대칭인 지점 — 숨기지 않는다

DOPE 에는 box head 가 없다.  AP 와 IoU@0.5 매칭에는 box 가 필요하므로
**검출된 cuboid 코너의 bounding box** 를 쓴다.  YOLO 의 box 는 학습된 예측이고
DOPE 의 box 는 keypoint 에서 유도한 것이라 같은 양이 아니다.  report 의
`recipe.box_source` 에 그대로 적고, 표 각주로도 남긴다.

score 도 마찬가지다.  YOLO 는 box confidence, DOPE 는 belief peak 최대값이다.
둘 다 "이 프레임에 팔레트가 있다" 의 모델 고유 점수지만 같은 척도가 아니다.

실행:  conda activate pallet-pose
    python scripts/self_training_yolo/dump_dope_predictions.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for sub in ("", "scripts/stage0", "scripts/stage0/paper_s2",
            "scripts/stage0/selftrain", "scripts/annotate", "challenge"):
    sys.path.insert(0, str(REPO_ROOT / sub) if sub else str(REPO_ROOT))

MANIFESTS = REPO_ROOT / "challenge/real_gt_v2/manifests"
WEIGHTS = REPO_ROOT / "weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth"
OUT = REPO_ROOT / "data/pallet/results/paper_eval_v1/baselines/DOPE_R0_PREDICTIONS.json"

PAD = 100
# belief peak threshold.  s1_cad_9filters / mc_dump_dope 가 쓰는 값과 같다.
THRESH = 0.3
MIN_CORNERS_FOR_BOX = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_images(name: str) -> list[str]:
    payload = json.loads((MANIFESTS / f"{name}.json").read_text())
    # legacy manifest 는 `image`, object-aware manifest 는 `image_path` 를 쓴다.
    return [item.get("image_path") or item["image"] for item in payload["items"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import cv2
    import torch
    import s1_cad_9filters as S
    from paper_s2_real_eval import E

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E.load_model(str(WEIGHTS), device)
    print(f"DOPE loaded on {device}  sha {sha256_file(WEIGHTS)[:16]}", flush=True)

    images = manifest_images("PAPER_EVAL_ALL_POS") + manifest_images("DEV_NEG2689")
    # negative manifest 는 중복 image 를 담을 수 있다 — 추론은 한 번만 한다.
    unique = list(dict.fromkeys(images))
    if args.limit:
        unique = unique[: args.limit]
    print(f"frames {len(unique)} (unique of {len(images)})", flush=True)

    frames: dict[str, list[dict]] = {}
    detected = 0
    started = time.time()
    for position, relative in enumerate(unique):
        image = cv2.imread(str(REPO_ROOT / relative))
        if image is None:
            raise SystemExit(f"IMAGE_DECODE_FAILED: {relative}")

        belief, geom, wh = S.infer_belief(model, image, device, PAD)
        pred8, pred_c, peaks, _ = S.belief_to_pred(belief, geom, wh, PAD, THRESH)

        valid = ~np.isnan(pred8[:, 0])
        if int(valid.sum()) < MIN_CORNERS_FOR_BOX:
            frames[relative] = []
            continue

        points = pred8[valid]
        box = [float(points[:, 0].min()), float(points[:, 1].min()),
               float(points[:, 0].max()), float(points[:, 1].max())]
        if box[2] - box[0] <= 1.0 or box[3] - box[1] <= 1.0:
            frames[relative] = []
            continue

        # keypoints 는 9x2 여야 evaluator 가 keypoint 통계를 낸다.  검출 안 된 코너는
        # 유한값으로 지어내지 않는다 — NaN 이면 evaluator 가 그 프레임을 keypoint
        # 통계에서 제외한다(shape/finite 검사).  centroid 는 belief 8번 채널이다.
        keypoints = np.full((9, 2), np.nan)
        keypoints[:8] = pred8
        if pred_c is not None:
            keypoints[8] = pred_c

        frames[relative] = [{
            "score": float(np.max(peaks[:8])),
            "box_xyxy": box,
            "keypoints_xy": [
                [None if np.isnan(x) else float(x), None if np.isnan(y) else float(y)]
                for x, y in keypoints
            ],
            "n_detected_corners": int(valid.sum()),
        }]
        detected += 1
        if (position + 1) % 200 == 0:
            rate = (position + 1) / (time.time() - started)
            print(f"  {position + 1}/{len(unique)}  det {detected}  "
                  f"{rate:.1f} img/s", flush=True)

    payload = {
        "schema_version": "paper_cached_predictions_v1",
        "model": "DOPE backbone_dope_final_v1 epoch60",
        "weights": str(WEIGHTS.relative_to(REPO_ROOT)),
        "weights_sha256": sha256_file(WEIGHTS),
        "recipe": {
            "pad": PAD,
            "border": "BORDER_REFLECT_101 (E.pad_frame)",
            "preprocess": "the model's training preprocess (448, sigma 4.0 contract)",
            "belief_threshold": THRESH,
            "source": (
                "scripts/stage0/selftrain/s1_cad_9filters.py infer_belief/belief_to_pred; "
                "paper_s2_real_eval.E.load_model/belief_to_orig_pad"
            ),
            "box_source": (
                "BOUNDING_BOX_OF_DETECTED_CUBOID_CORNERS — DOPE has no box head. "
                "YOLO's box is a learned prediction; this one is derived from "
                "keypoints.  They are not the same quantity."
            ),
            "score_source": (
                "max belief peak over the 8 cuboid corners.  YOLO uses box "
                "confidence.  Both are the model's own presence score but they "
                "are not on the same scale."
            ),
            "reflect_padding_note": (
                "plain squash inference systematically under-detects truncated and "
                "close-range pallets and would make DOPE look worse than it is"
            ),
        },
        "n_frames": len(frames),
        "n_detected": detected,
        "frames": frames,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}  "
          f"{OUT.stat().st_size / 1e6:.1f} MB   detected {detected}/{len(frames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
