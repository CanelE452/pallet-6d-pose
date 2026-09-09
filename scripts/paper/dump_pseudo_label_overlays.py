"""self-training 이 실제로 학습한 pseudo-label 을 프레임당 그림 한 장으로 덤프한다.

    python3 scripts/paper/dump_pseudo_label_overlays.py --arm R5_PROPOSED
    python3 scripts/paper/dump_pseudo_label_overlays.py --arm R1_NAIVE

출력: data/pallet/results/paper_selftrain_v1/pl_review/<ARM>/{session}__{frame}.jpg
      같은 폴더에 INDEX.csv (키포인트 신뢰도 오름차순 — 약한 것부터 보기)

라벨은 manifest 를 다시 계산하지 않고 **학습이 읽은 그 파일**
(`datasets/paper_selftrain_v1/<ARM>/labels/train/pl__*.txt`) 에서 가져온다.
학습에 들어간 좌표와 그림이 어긋날 여지를 없애기 위해서다.

이 pool 프레임에는 manual GT 가 없다.  그려진 점은 전부 모델의 예측이며
정답이 아니다 — 배너에 그렇게 적는다.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/paper_selftrain_v1"
MANIFESTS = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/pseudo_manifests"
OUT_ROOT = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/pl_review"

# arm 이 어느 필터 manifest 로 만들어졌는지
ARM_MANIFEST = {
    "R1_NAIVE": "F0_NAIVE",
    "R2_CONF": "F1_CONF",
    "R3_CONF_REPROJ": "F2_CONF_REPROJ",
    "R4_CONF_REMOVE": "F3_CONF_REMOVE",
    "R5_PROPOSED": "F4_PROPOSED",
    "R6_CONF_FLIP": "F5_CONF_FLIP",
}

# camera-facing 0123 — 0~3 앞면, 4~7 뒷면, {0,1,4,5} 위 / {2,3,6,7} 아래, 8 centroid
NEAR = [(0, 1), (1, 2), (2, 3), (3, 0)]
FAR = [(4, 5), (5, 6), (6, 7), (7, 4)]
DEPTH = [(0, 4), (1, 5), (2, 6), (3, 7)]
NEAR_COLOUR = (80, 220, 255)      # 앞면 — 노랑
FAR_COLOUR = (255, 170, 80)       # 뒷면 — 파랑
DEPTH_COLOUR = (170, 170, 170)    # 깊이 변 — 회색
CENTROID_COLOUR = (120, 120, 255)
BOX_COLOUR = (90, 255, 120)


def number(row: dict, key: str, spec: str = ".3f") -> str:
    """manifest 에 빈 칸이 있을 수 있다 — naive 필터는 기하 점수를 계산하지 않는다."""

    value = (row or {}).get(key, "")
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "n/a"


def read_label(path: Path, width: int, height: int):
    """YOLO-pose 라벨 한 줄을 픽셀 좌표로."""

    values = [float(v) for v in path.read_text().split()]
    cx, cy, bw, bh = values[1:5]
    box = np.array([(cx - bw / 2) * width, (cy - bh / 2) * height,
                    (cx + bw / 2) * width, (cy + bh / 2) * height])
    flat = np.array(values[5:]).reshape(-1, 3)
    points = np.stack([flat[:, 0] * width, flat[:, 1] * height], axis=1)
    visibility = flat[:, 2]
    return box, points, visibility


def draw(image, box, points, visibility):
    canvas = image.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOUR, 1, cv2.LINE_AA)

    def visible(index):
        return (index < len(points) and visibility[index] > 0
                and np.isfinite(points[index]).all())

    for edges, colour, thickness in ((DEPTH, DEPTH_COLOUR, 1),
                                     (FAR, FAR_COLOUR, 2),
                                     (NEAR, NEAR_COLOUR, 2)):
        for a, b in edges:
            if visible(a) and visible(b):
                cv2.line(canvas, tuple(np.int32(points[a])),
                         tuple(np.int32(points[b])), colour, thickness, cv2.LINE_AA)

    for index in range(min(9, len(points))):
        if not visible(index):
            continue
        centre = tuple(np.int32(points[index]))
        colour = (CENTROID_COLOUR if index == 8
                  else NEAR_COLOUR if index < 4 else FAR_COLOUR)
        cv2.circle(canvas, centre, 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, centre, 3, colour, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (centre[0] + 6, centre[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (centre[0] + 6, centre[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return canvas


def banner(canvas, lines):
    pad = 6
    height = 18 * len(lines) + pad
    strip = canvas[:height].copy()
    cv2.rectangle(strip, (0, 0), (canvas.shape[1], height), (0, 0, 0), -1)
    canvas[:height] = cv2.addWeighted(strip, 0.65, canvas[:height], 0.35, 0)
    for index, line in enumerate(lines):
        y = 16 + index * 18
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="R5_PROPOSED",
                        help=f"one of {', '.join(sorted(ARM_MANIFEST))}")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()

    dataset = DATASETS / args.arm
    if not dataset.exists():
        raise SystemExit(f"no such training dataset: {dataset}")
    images = sorted((dataset / "images/train").glob("pl__*"))
    if not images:
        raise SystemExit(f"{args.arm} has no pl__ images")

    scores: dict[str, dict] = {}
    manifest_name = ARM_MANIFEST.get(args.arm)
    if manifest_name and (MANIFESTS / f"{manifest_name}.csv").exists():
        for row in csv.DictReader((MANIFESTS / f"{manifest_name}.csv").open()):
            scores[Path(row["image_path"]).stem] = row

    out_dir = OUT_ROOT / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()

    index_rows = []
    missing_label = 0
    for image_path in images:
        label_path = dataset / "labels/train" / f"{image_path.stem}.txt"
        if not label_path.exists() or not label_path.read_text().strip():
            missing_label += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            missing_label += 1
            continue
        height, width = image.shape[:2]
        box, points, visibility = read_label(label_path, width, height)

        _, session, frame_id = image_path.stem.split("__", 2)
        row = scores.get(frame_id, {})
        canvas = draw(image, box, points, visibility)
        canvas = banner(canvas, [
            f"{args.arm}   {session}   {frame_id}",
            (f"box_conf {number(row, 'box_conf')}   "
             f"kp_conf_med8 {number(row, 'kp_conf_median8')}   "
             f"corners {row.get('valid_corners') or 'n/a'}   "
             f"{row.get('paper_condition', '')}"
             if row else "manifest row not found"),
            (f"s_reproj {number(row, 's_reproj', '.4f')}   "
             f"s_remove {number(row, 's_remove', '.4f')}   "
             f"s_flip {number(row, 's_flip', '.4f')}" if row else ""),
            "PSEUDO-LABEL, NOT GROUND TRUTH - this pool frame has no manual GT",
        ])
        out_path = out_dir / f"{session}__{frame_id}.jpg"
        cv2.imwrite(str(out_path), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        index_rows.append({
            "file": out_path.name,
            "session": session,
            "frame_id": frame_id,
            "condition": row.get("paper_condition", ""),
            "box_conf": row.get("box_conf", ""),
            "kp_conf_median8": row.get("kp_conf_median8", ""),
            "valid_corners": row.get("valid_corners", ""),
            "s_reproj": row.get("s_reproj", ""),
            "s_remove": row.get("s_remove", ""),
            "s_flip": row.get("s_flip", ""),
            "source_image": str(image_path.resolve().relative_to(REPO_ROOT)),
        })

    def sort_key(row):
        try:
            return float(row["kp_conf_median8"])
        except (TypeError, ValueError):
            return 1.0

    index_rows.sort(key=sort_key)
    with (out_dir / "INDEX.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    (out_dir / "README.txt").write_text(
        f"arm            {args.arm}\n"
        f"filter         {manifest_name}\n"
        f"frames         {len(index_rows)}\n"
        f"generated      {datetime.now(timezone.utc).isoformat()}\n"
        f"\n"
        f"이 그림들은 self-training 이 학습한 pseudo-label 이다.  좌표는\n"
        f"{dataset.relative_to(REPO_ROOT)}/labels/train 의 라벨 파일 그대로이며\n"
        f"다시 계산하지 않았다.\n"
        f"\n"
        f"출처는 adaptation pool (data/pallet/raw_data/{{outside,night}}) 이고\n"
        f"PAPER_EVAL 319 와 겹치지 않는다.  이 세션들에는 manual GT 가 없으므로\n"
        f"그려진 점은 전부 모델 예측이다 - 정답이 아니다.\n"
        f"\n"
        f"색   앞면 0-3 노랑 / 뒷면 4-7 파랑 / 깊이 변 회색 / 8 centroid 보라\n"
        f"     초록 사각형은 box 라벨\n"
        f"\n"
        f"INDEX.csv 는 kp_conf_median8 오름차순 - 약한 것부터 보면 된다.\n")

    print(f"arm {args.arm}   filter {manifest_name}")
    print(f"wrote {len(index_rows)} overlays  (label missing/unreadable {missing_label})")
    print(f"  {out_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
