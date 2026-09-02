"""Proposed 필터를 통과한 pseudo-label 중 GT 에 가장 가까운 것과 가장 먼 것을 그린다.

주의 — 실제 pseudo-label 이 만들어지는 adaptation pool 273 장에는 GT 가 없다.
그래서 여기서는 **같은 선정 규칙을 GT 가 있는 PAPER_EVAL 에 적용했을 때** 통과한
프레임을 쓴다 (M4 가 재는 것과 같은 것).  reader 에게 이 구분을 숨기지 않는다.

축 순열이 원인인지도 함께 표시한다.  큰 오차의 상당수는 코너가 딴 데 간 것이 아니라
라벨이 90 도 돌아간 것이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RECORDS = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/M4_FRAME_RECORDS.json"
OUT = REPO_ROOT / "data/pallet/results/paper_selftrain_v4/repair_visual"

GREEN, BLUE, GREY = (0, 220, 0), (255, 170, 0), (150, 150, 150)
CELL = (640, 500)
YAW90 = (1, 5, 6, 2, 0, 4, 7, 3, 8)
FLIP_IDX = (1, 0, 3, 2, 5, 4, 7, 6, 8)


def compose(outer, inner):
    return tuple(inner[index] for index in outer)


PERMUTATIONS = {
    "yaw90": YAW90,
    "yaw180": compose(YAW90, YAW90),
    "yaw270": compose(compose(YAW90, YAW90), YAW90),
    "mirror": FLIP_IDX,
}


def explains(keypoints, gt, supervised) -> str:
    """어떤 라벨 순열이 이 오차를 설명하는가.  판정은 **최대** 오차로 한다."""

    identity = float(np.max(np.linalg.norm(keypoints - gt, axis=1)[supervised]))
    best, name = float("inf"), None
    for label, perm in PERMUTATIONS.items():
        value = float(np.max(
            np.linalg.norm(keypoints[list(perm)] - gt, axis=1)[supervised]))
        if value < best:
            best, name = value, label
    if best < 25.0 and best < 0.5 * identity:
        return f"{name} explains it ({identity:.0f}px -> {best:.0f}px)"
    return "not a label permutation"


def banner(canvas, lines, colour=(255, 255, 255)):
    y = 17
    for line in lines:
        cv2.putText(canvas, line, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1, cv2.LINE_AA)
        y += 16


def cell(record: dict, title: str) -> np.ndarray:
    image = cv2.imread(str(REPO_ROOT / record["image_path"]))
    if image is None:
        raise SystemExit(f"UNREADABLE_IMAGE: {record['image_path']}")
    canvas = image.copy()
    gt = np.asarray(record["gt_xy"], dtype=float)
    keypoints = np.asarray(record["keypoints_xy"], dtype=float)
    supervised = np.asarray(record["gt_supervised"], dtype=bool)

    for index in range(9):
        if not supervised[index]:
            continue
        a = tuple(np.round(gt[index]).astype(int))
        b = tuple(np.round(keypoints[index]).astype(int))
        cv2.line(canvas, a, b, GREY, 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 5, GREEN, -1, cv2.LINE_AA)
        cv2.circle(canvas, b, 4, BLUE, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (b[0] + 5, b[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE, 1, cv2.LINE_AA)

    errors = np.asarray(record["errors_px"], dtype=float)
    height, width = canvas.shape[:2]
    scale = min(CELL[0] / width, CELL[1] / height)
    canvas = cv2.resize(canvas, (int(width * scale), int(height * scale)))
    tile = np.zeros((CELL[1], CELL[0], 3), dtype=np.uint8)
    tile[: canvas.shape[0], : canvas.shape[1]] = canvas
    banner(tile, [
        f"{title}   {record['frame_id']}   {record['domain']}",
        f"median {np.median(errors):.2f} px    max {errors.max():.2f} px"
        f"    gross(>20px) {record.get('gross_keypoints')}",
        f"box_conf {record['box_conf']:.3f}   s_reproj {record['s_reproj']:.4f}"
        f"   s_remove {record['s_remove']:.4f}   s_flip {record['s_flip']:.4f}",
        explains(keypoints, gt, supervised),
        "GT green | pseudo-label blue",
    ])
    cv2.rectangle(tile, (0, 0), (CELL[0] - 1, CELL[1] - 1), GREY, 1)
    return tile


def main() -> int:
    records = json.loads(RECORDS.read_text())["frames"]
    accepted = [r for r in records
                if r["verdict"].get("F4_PROPOSED") and r.get("errors_px")]
    accepted.sort(key=lambda r: float(np.median(r["errors_px"])))

    picks = [(accepted[0], "BEST"), (accepted[1], "BEST #2"),
             (accepted[-1], "WORST"), (accepted[-2], "WORST #2")]
    tiles = [cell(record, title) for record, title in picks]

    sheet = np.zeros((2 * CELL[1] + 30, 2 * CELL[0], 3), dtype=np.uint8)
    cv2.putText(sheet,
                f"Proposed-accepted pseudo-labels scored against GT "
                f"({len(accepted)} frames) - best and worst",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for index, tile in enumerate(tiles):
        r, c = divmod(index, 2)
        sheet[30 + r * CELL[1]: 30 + (r + 1) * CELL[1],
              c * CELL[0]: (c + 1) * CELL[0]] = tile
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "PSEUDO_BEST_WORST.jpg"
    cv2.imwrite(str(path), sheet)

    print(f"accepted {len(accepted)} frames")
    for record, title in picks:
        errors = np.asarray(record["errors_px"], dtype=float)
        print(f"  {title:8} {record['frame_id']:42} median {np.median(errors):7.2f} px"
              f"  max {errors.max():7.2f} px  gross {record.get('gross_keypoints')}")
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
