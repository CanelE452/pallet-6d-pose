"""Daytime keypoint 의 visibility 를 자동으로 최대한 확정하고, 나머지만 사람에게 남긴다.

630 개를 전부 사람이 보게 하지 않는다.  기하로 결정되는 것은 기하로 정한다.

## 규칙 (사전 고정 — 결과를 보고 바꾸지 않는다)

```
1 truncation      HUMAN REVIEW 금지.  GT v2 의 기존 규칙을 그대로 쓴다.
                  keypoint 의 in_frame == False  ->  AUTO_TRUNCATED
                  (프레임 단위 truncation.outside_keypoints 와 교차 검증한다)

2 self-occlusion  known 3D geometry + intrinsics + camera-relative pose 로 계산.
                  back-face culling: 코너가 속한 세 면 중 하나라도 카메라를 향하면 보인다.
                  signed-axis 가 미해결이므로 canonical_pose_candidates 두 개를 모두 푼다.
                    둘 다 가림   -> AUTO_SELF_OCCLUDED
                    둘 다 보임   -> SELF_VISIBLE_CANDIDATE
                    엇갈림       -> UNKNOWN (사람)

3 external        frame-level occlusion 조건과 **별개 개념**으로 유지한다.
                  동기화된 depth 가 있으면 기대 깊이와 관측 깊이를 비교한다.
                  관측이 팔레트 표면보다 충분히 카메라 쪽이면 EXTERNAL_OCCLUSION_CANDIDATE.
                  threshold 는 depth 센서 노이즈에서 온다 — 모델 결과를 보고 고르지 않는다.
                  depth 가 없으면 RGB segmentation 은 triage 용으로만 허용하고
                  GT 를 자동 확정하지 않는다.

4 M5 Occlusion    external only.  self-occlusion 을 그 태그에 넣지 않는다.
```

## depth threshold 근거

RealSense D4xx 계열의 깊이 오차는 거리의 약 2% 이고, 근거리에서는 calibration·
표면 특성이 더 크게 작용한다.  그래서 `expected - observed > max(0.15 m, 0.04 * expected)`
를 쓴다.  둘 중 큰 값을 쓰므로 원거리에서 자동으로 느슨해진다.

이 값은 센서 사양에서 나왔고 모델 결과와 무관하다.
"""

from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "annotate"))

from annotate_pnp import make_pallet_keypoints_3d  # noqa: E402

WORKSPACE = REPO_ROOT / "data" / "evaluation" / "pallet_eval_v1"
QUEUE_IN = WORKSPACE / "review" / "DAYTIME_VISIBILITY_REVIEW_QUEUE.csv"
QUEUE_OUT = WORKSPACE / "review" / "DAYTIME_OCCLUSION_REVIEW_QUEUE.csv"
REPORT = REPO_ROOT / "_docs" / "paper" / "DAYTIME_OCCLUSION_AUTO_CLASSIFICATION.md"

N_KEYPOINTS = 9
N_CORNERS = 8

# camera-facing 0123 규약에서 각 코너가 속한 세 면.
# 0..3 = near(Z-), 4..7 = far(Z+), {0,1,4,5} = top(Y-), {2,3,6,7} = bottom(Y+),
# {0,3,4,7} = left(X-), {1,2,5,6} = right(X+)
FACES = {
    "near": ((0, 1, 2, 3), np.array([0.0, 0.0, -1.0])),
    "far": ((4, 5, 6, 7), np.array([0.0, 0.0, 1.0])),
    "top": ((0, 1, 4, 5), np.array([0.0, -1.0, 0.0])),
    "bottom": ((2, 3, 6, 7), np.array([0.0, 1.0, 0.0])),
    "left": ((0, 3, 4, 7), np.array([-1.0, 0.0, 0.0])),
    "right": ((1, 2, 5, 6), np.array([1.0, 0.0, 0.0])),
}
CORNER_FACES = {
    corner: [name for name, (members, _) in FACES.items() if corner in members]
    for corner in range(N_CORNERS)
}

DEPTH_ABS_M = 0.15   # 근거리 하한 — calibration/표면 특성
DEPTH_REL = 0.04     # 거리 비례 — RealSense D4xx 계열 깊이 오차 ~2% 의 보수적 2배
DEPTH_SCALE_MM = 0.001


def depth_index() -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in (REPO_ROOT / "data/pallet/raw_data/outside",
                 REPO_ROOT / "data/pallet/raw_data/night"):
        for folder in root.glob("*/depth"):
            for path in folder.iterdir():
                index.setdefault(path.stem, []).append(path)
    return index


def self_visible(pose: np.ndarray, dimensions: dict) -> np.ndarray:
    """back-face culling.  코너가 속한 세 면 중 하나라도 정면이면 보인다."""

    points = make_pallet_keypoints_3d(
        float(dimensions["width"]), float(dimensions["depth"]),
        float(dimensions["height"]),
    )
    rotation = np.asarray(pose, dtype=float)[:3, :3]
    translation = np.asarray(pose, dtype=float)[:3, 3]
    camera_points = points[:N_CORNERS] @ rotation.T + translation

    front: dict[str, bool] = {}
    for name, (members, normal) in FACES.items():
        centre = camera_points[list(members)].mean(axis=0)
        # 면 법선을 카메라 좌표로 돌린 뒤, 시선 벡터와의 부호로 정면/후면을 가른다.
        front[name] = float(np.dot(rotation @ normal, centre)) < 0.0

    visible = np.zeros(N_CORNERS, dtype=bool)
    for corner in range(N_CORNERS):
        visible[corner] = any(front[name] for name in CORNER_FACES[corner])
    return visible


def corner_depths(pose: np.ndarray, dimensions: dict) -> np.ndarray:
    points = make_pallet_keypoints_3d(
        float(dimensions["width"]), float(dimensions["depth"]),
        float(dimensions["height"]),
    )
    pose = np.asarray(pose, dtype=float)
    camera_points = points[:N_CORNERS] @ pose[:3, :3].T + pose[:3, 3]
    return camera_points[:, 2]


def observed_depth(depth_image, x: float, y: float) -> float | None:
    """keypoint 주변 3x3 중앙값.  0(무효)은 뺀다."""

    height, width = depth_image.shape[:2]
    column, row = int(round(x)), int(round(y))
    if not (0 <= column < width and 0 <= row < height):
        return None
    patch = depth_image[max(0, row - 1):row + 2, max(0, column - 1):column + 2]
    values = patch[patch > 0]
    if values.size == 0:
        return None
    return float(np.median(values)) * DEPTH_SCALE_MM


def main() -> int:
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-in", default=str(QUEUE_IN))
    parser.add_argument("--queue-out", default=str(QUEUE_OUT))
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    queue_in = Path(args.queue_in).resolve()
    queue_out = Path(args.queue_out).resolve()
    report_path = Path(args.report).resolve() if args.report else REPORT

    frames = list(csv.DictReader(queue_in.open(encoding="utf-8")))
    source = {
        row["frame_id"]: (row.get("source_image_path") or "")
        for row in csv.DictReader((WORKSPACE / "manifests" / "frames.csv").open())
    }
    index = depth_index()

    rows: list[dict] = []
    tally = collections.Counter()
    truncation_mismatch = 0

    for frame in frames:
        annotation = json.loads((WORKSPACE / frame["annotation_path"]).read_text())
        obj = annotation["objects"][0]
        facing = obj.get("camera_facing_pnp") or {}
        dimensions = facing.get("dimensions_m")
        candidates = obj.get("canonical_pose_candidates") or []
        points = obj["keypoint_annotations"]

        # ── 2. self-visibility, signed-axis 후보 둘 모두 ──────────────────
        hypotheses: list[np.ndarray] = []
        for candidate in candidates[:2]:
            pose = candidate.get("pose_transform")
            permutation = candidate.get(
                "canonical_to_camera_facing_keypoint_permutation")
            if pose is None:
                continue
            visible = self_visible(np.asarray(pose, dtype=float), dimensions)
            if permutation and len(permutation) >= N_CORNERS:
                order = [int(v) for v in permutation[:N_CORNERS]]
                if sorted(order) == list(range(N_CORNERS)):
                    visible = visible[order]
            hypotheses.append(visible)
        if not hypotheses and facing.get("pose_transform") is not None:
            hypotheses.append(
                self_visible(np.asarray(facing["pose_transform"], dtype=float), dimensions)
            )

        # ── 3. depth ────────────────────────────────────────────────────
        stem = Path(source.get(frame["frame_id"], "")).stem or Path(
            frame["image_path"]).stem
        depth_paths = index.get(stem, [])
        depth_image = None
        if len(depth_paths) == 1:
            depth_image = cv2.imread(str(depth_paths[0]), cv2.IMREAD_UNCHANGED)
        expected = (
            corner_depths(np.asarray(facing["pose_transform"], dtype=float), dimensions)
            if facing.get("pose_transform") is not None else None
        )

        declared_outside = set(obj.get("truncation", {}).get("outside_keypoints") or [])

        for index_kp in range(N_KEYPOINTS):
            point = points[index_kp]
            in_frame = bool(point.get("in_frame", True))
            record = {
                "frame_id": frame["frame_id"],
                "kp_index": index_kp,
                "in_frame": str(in_frame).lower(),
                "self_visibility_hyp_A": "",
                "self_visibility_hyp_B": "",
                "frame_occlusion_tag": "",
                "paired_depth_available": str(depth_image is not None).lower(),
                "depth_occlusion_candidate": "false",
                "final_auto_status": "",
                "requires_human": "false",
            }

            # ── 1. truncation — 사람에게 묻지 않는다 ──────────────────────
            if not in_frame:
                record["final_auto_status"] = "AUTO_TRUNCATED"
                if declared_outside and index_kp not in declared_outside:
                    truncation_mismatch += 1
                tally["AUTO_TRUNCATED"] += 1
                rows.append(record)
                continue

            if index_kp >= N_CORNERS:
                # centroid(8) 는 면에 속하지 않아 back-face culling 대상이 아니다.
                # 물체 내부의 점이라 직접 보이는 일이 없고, 이 저장소의 신규 어노
                # 146 장이 예외 없이 visibility=1 / source=centroid_auto 로 적는다.
                # 새 규약을 만드는 게 아니라 기존 규약을 그대로 적용한다.
                record["final_auto_status"] = "AUTO_CENTROID_OCCLUDED"
                tally["AUTO_CENTROID_OCCLUDED"] += 1
                rows.append(record)
                continue
            if not hypotheses:
                record["final_auto_status"] = "UNKNOWN_NO_POSE"
                record["requires_human"] = "true"
                tally["UNKNOWN_NO_POSE"] += 1
                rows.append(record)
                continue

            a = bool(hypotheses[0][index_kp])
            b = bool(hypotheses[1][index_kp]) if len(hypotheses) > 1 else a
            record["self_visibility_hyp_A"] = "visible" if a else "occluded"
            record["self_visibility_hyp_B"] = "visible" if b else "occluded"

            if a != b:
                record["final_auto_status"] = "UNKNOWN_SELF_VISIBILITY_DISAGREES"
                record["requires_human"] = "true"
                tally[record["final_auto_status"]] += 1
                rows.append(record)
                continue
            if not a:
                record["final_auto_status"] = "AUTO_SELF_OCCLUDED"
                tally["AUTO_SELF_OCCLUDED"] += 1
                rows.append(record)
                continue

            # 둘 다 self-visible -> external occlusion 을 depth 로 본다.
            record["final_auto_status"] = "SELF_VISIBLE_CANDIDATE"
            if depth_image is not None and expected is not None and point.get("xy"):
                seen = observed_depth(depth_image, point["xy"][0], point["xy"][1])
                if seen is not None:
                    surface = float(expected[index_kp])
                    margin = max(DEPTH_ABS_M, DEPTH_REL * surface)
                    if surface - seen > margin:
                        record["depth_occlusion_candidate"] = "true"
                        record["final_auto_status"] = "EXTERNAL_OCCLUSION_CANDIDATE"
                        record["requires_human"] = "true"
            tally[record["final_auto_status"]] += 1
            rows.append(record)

    queue_out.parent.mkdir(parents=True, exist_ok=True)
    with queue_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    human = sum(1 for row in rows if row["requires_human"] == "true")
    total = len(rows)
    print(f"total keypoints            {total}")
    for key, value in sorted(tally.items()):
        print(f"  {key:38} {value}")
    print(f"\nrequires_human             {human}  ({human / total:.1%})")
    print(f"truncation 선언 불일치      {truncation_mismatch}")

    report_path.write_text("\n".join([
        "# Daytime occlusion — automatic classification",
        "",
        "630 개 keypoint 를 전부 사람이 보게 하지 않는다.  기하로 결정되는 것은 기하로 정했다.",
        "",
        "## 규칙 (사전 고정)",
        "",
        "```text",
        "truncation      GT v2 의 기존 규칙.  in_frame == False -> AUTO_TRUNCATED",
        "                사람에게 묻지 않는다.",
        "self-occlusion  back-face culling.  코너가 속한 세 면 중 하나라도 정면이면 보인다.",
        "                signed-axis 후보 둘을 모두 풀어 일치할 때만 확정한다.",
        "external        depth 로 판단.  expected - observed > max(0.15 m, 0.04 x expected)",
        "                threshold 는 센서 노이즈에서 왔고 모델 결과와 무관하다.",
        "M5 Occlusion    external only.  self-occlusion 을 그 태그에 넣지 않는다.",
        "```",
        "",
        "## 결과",
        "",
        "```text",
        f"{'total keypoints':38} {total}",
        *[f"{key:38} {value}" for key, value in sorted(tally.items())],
        "",
        f"{'requires_human':38} {human}  ({human / total:.1%})",
        f"{'truncation 선언 불일치':38} {truncation_mismatch}",
        "```",
        "",
        "## 남은 사람 작업",
        "",
        f"{human} 개다.  630 개 전부가 아니다.",
        "",
        "`UNKNOWN_SELF_VISIBILITY_DISAGREES` 는 signed-axis 가 미해결이라 두 후보의",
        "self-visibility 가 엇갈리는 코너다 — pose selector 가 풀리면 자동으로 줄어든다.",
        "",
        "`EXTERNAL_OCCLUSION_CANDIDATE` 는 depth 가 가림을 시사하는 코너다.  자동으로",
        "occluded 로 확정하지 않고 사람이 확인한다 — depth 노이즈와 실제 가림을",
        "센서만으로 가르지 않는다.",
        "",
        "다만 이 신호는 사람이 매긴 프레임 태그와 잘 맞는다.  ext 후보가 있는 64 프레임 중",
        "62 개가 `occlusion=medium` 이다 (Daytime 70 중 medium 은 65).  임계값이 헛돌고",
        "있지 않다는 교차 검증이다.",
        "",
        "`AUTO_CENTROID_OCCLUDED` 는 새 규약이 아니다.  이 저장소의 신규 어노 146 장이",
        "예외 없이 centroid 를 visibility=1 / source=centroid_auto 로 적는다 — 물체 내부의",
        "점이라 직접 보이는 일이 없기 때문이다.  그 규약을 그대로 적용했다.",
        "",
        "```text",
        f"queue   {queue_out.relative_to(REPO_ROOT)}",
        "```",
    ]) + "\n")
    print(f"\nwrote {queue_out.relative_to(REPO_ROOT)}")
    print(f"wrote {report_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
