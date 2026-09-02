"""Daytime keypoint 의 visibility 만 사람이 확인하는 도구.  좌표는 건드릴 수 없다.

## 왜 별도 스크립트인가

`annotate.py` 는 12,000 줄이 넘고 WASD·QE·JKIL 이 pose 조작에 묶여 있다.  거기에
`--visibility-only` 플래그를 다는 방식은 **플래그 하나가 잘못되면 좌표가 망가진다.**
이 저장소에는 이미 그런 사고가 있었다 — `u<0` 오판으로 72 프레임 153 코너가
invisible 로 저장됐다.

그래서 좌표·pose·bbox 를 바꾸는 코드가 **아예 없는** 별도 프로그램으로 만든다.
금지 편집은 플래그가 아니라 구조로 불가능하다.

## 화면에 절대 나오지 않는 것

모델 예측 · overlay · corner error · filter score · confidence · 모델 이름 ·
pass/fail.  이 프로그램은 그 값을 읽지도 않는다.

## 무엇을 저장하나

GT 파일을 직접 고치지 않는다.  사람의 판단만 **amendment layer** 에 쌓는다.

    data/evaluation/pallet_eval_v1/review/DAYTIME_VISIBILITY_AMENDMENTS.json

GT 반영은 `apply_visibility_amendments.py` 가 따로 하고, 거기서 허용 필드 외에
한 글자라도 바뀌면 롤백한다.  review 중 크래시가 나도 GT 는 그대로다.

## 사용

    python scripts/annotate/review_visibility_only.py
    python scripts/annotate/review_visibility_only.py --all-keypoints   # 119 개 외도 보기

## 키

    v  visible      o  occluded     t  truncated    u  unknown(보류)
    n  다음 후보     p  이전 후보     [ ]  프레임 이동
    -  =  확대 축소 / 확대
    c  coordinate_review_flag 토글 (좌표가 의심스러울 때 — 고치지 않고 표시만)
    s  저장          q  저장 후 종료
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT / "data" / "evaluation" / "pallet_eval_v1"
QUEUE = WORKSPACE / "review" / "DAYTIME_OCCLUSION_REVIEW_QUEUE.csv"
AMENDMENTS = WORKSPACE / "review" / "DAYTIME_VISIBILITY_AMENDMENTS.json"

WINDOW = "visibility review (prediction-blinded)"
STATES = {"v": 2, "o": 1, "t": 0, "u": None}
STATE_NAME = {2: "visible", 1: "occluded", 0: "truncated", None: "unknown"}
STATE_COLOUR = {2: (80, 220, 80), 1: (60, 160, 240), 0: (150, 150, 150),
                None: (60, 60, 230)}
PANEL_WIDTH = 330
# 팔레트가 화면에서 작게 잡히는 프레임이 많다.  확대 없이는 "저 코너가 고깔 뒤인가"
# 를 사람이 판단할 수 없다 — 실제 프레임을 렌더해 보고 넣었다.
ZOOM_BOX = 140      # 원본에서 잘라낼 정사각 크기(px).  - / = 로 조절한다.
ZOOM_BOX_MIN, ZOOM_BOX_MAX = 50, 320
ZOOM_VIEW = 300     # 확대 패널 한 변(px)
# cuboid 구조를 알아야 어느 면인지 판단할 수 있다.  camera-facing 0123 규약.
CUBOID_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7))


def load_amendments() -> dict:
    if AMENDMENTS.exists():
        return json.loads(AMENDMENTS.read_text())
    return {"schema_version": "paper_daytime_visibility_amendments_v1",
            "protocol": "prediction-blinded; visibility only; coordinates untouched",
            "frames": {}}


def save_amendments(data: dict) -> None:
    data["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    AMENDMENTS.parent.mkdir(parents=True, exist_ok=True)
    AMENDMENTS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def magnifier(image, points, cursor, targets, amendment, box=ZOOM_BOX):
    """cursor keypoint 주변을 확대한다.  이게 없으면 가림 판단 자체가 불가능하다."""

    view = np.zeros((ZOOM_VIEW, ZOOM_VIEW, 3), dtype=np.uint8)
    centre = points[cursor].get("xy")
    if centre is None:
        return view
    height, width = image.shape[:2]
    half = box // 2
    x = int(round(centre[0]))
    y = int(round(centre[1]))
    x0, y0 = max(0, x - half), max(0, y - half)
    x1, y1 = min(width, x + half), min(height, y + half)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return view
    scale = ZOOM_VIEW / max(crop.shape[0], crop.shape[1])
    resized = cv2.resize(crop, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST)
    view[: resized.shape[0], : resized.shape[1]] = resized

    def to_view(px, py):
        return int(round((px - x0) * scale)), int(round((py - y0) * scale))

    for a, b in CUBOID_EDGES:
        pa, pb = points[a].get("xy"), points[b].get("xy")
        if pa is None or pb is None:
            continue
        cv2.line(view, to_view(*pa), to_view(*pb), (70, 70, 70), 1, cv2.LINE_AA)
    for index, point in enumerate(points):
        if point.get("xy") is None:
            continue
        vx, vy = to_view(*point["xy"])
        if not (0 <= vx < ZOOM_VIEW and 0 <= vy < ZOOM_VIEW):
            continue
        state = amendment.get(str(index))
        colour = STATE_COLOUR[STATES[state]] if state in STATES else (110, 110, 110)
        if index == cursor:
            cv2.drawMarker(view, (vx, vy), (255, 255, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.circle(view, (vx, vy), 13, (255, 255, 255), 1)
        else:
            cv2.circle(view, (vx, vy), 5, colour, 1)
        cv2.putText(view, str(index), (vx + 9, vy - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    cv2.rectangle(view, (0, 0), (ZOOM_VIEW - 1, ZOOM_VIEW - 1), (90, 90, 90), 1)
    cv2.putText(view, f"KP{cursor}  x{scale:.1f}", (8, ZOOM_VIEW - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return view


ZOOM_TOP = 24 + 23 * 19   # 텍스트 패널 아래.  높이 계산과 배치가 같은 값을 쓴다.


def draw(image, points, frame_row, targets, cursor, amendment, frame_index,
         frame_total, box=ZOOM_BOX):
    panel_height = max(image.shape[0], ZOOM_TOP + ZOOM_VIEW + 14)
    canvas = np.zeros((panel_height, image.shape[1] + PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[: image.shape[0], : image.shape[1]] = image

    # cuboid 모서리를 흐리게 그려 어느 면인지 알아볼 수 있게 한다.
    for a, b in CUBOID_EDGES:
        pa, pb = points[a].get("xy"), points[b].get("xy")
        if pa is None or pb is None:
            continue
        cv2.line(canvas, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 (70, 70, 70), 1, cv2.LINE_AA)

    for index, point in enumerate(points):
        if point.get("xy") is None:
            continue
        x, y = int(round(point["xy"][0])), int(round(point["xy"][1]))
        state = amendment.get(str(index), "__none__")
        value = STATES.get(state) if state in STATES else None
        is_target = index in targets
        colour = STATE_COLOUR[value] if state in STATES else (110, 110, 110)
        radius = 9 if index == cursor else (7 if is_target else 4)
        cv2.circle(canvas, (x, y), radius, colour, 2 if is_target else 1)
        if index == cursor:
            cv2.circle(canvas, (x, y), radius + 5, (255, 255, 255), 1)
        cv2.putText(canvas, str(index), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    left = image.shape[1] + 12
    def line(row, text, colour=(230, 230, 230), scale=0.44):
        cv2.putText(canvas, text, (left, 24 + row * 19),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)

    done = sum(1 for index in targets if str(index) in amendment)
    line(0, "VISIBILITY REVIEW", (255, 255, 255), 0.55)
    line(1, f"Frame {frame_index + 1} / {frame_total}")
    line(2, frame_row["frame_id"][:30], (170, 170, 170))
    line(3, f"candidates {done} / {len(targets)}",
         (80, 220, 80) if done == len(targets) else (60, 170, 240))
    line(4, "READY" if done == len(targets) else "INCOMPLETE",
         (80, 220, 80) if done == len(targets) else (60, 60, 230))
    line(6, "keypoints", (255, 255, 255), 0.5)
    for offset, index in enumerate(range(len(points))):
        state = amendment.get(str(index))
        label = STATE_NAME[STATES[state]] if state in STATES else "-"
        mark = ">" if index == cursor else (" " if index not in targets else "*")
        colour = (255, 255, 255) if index == cursor else (
            (200, 200, 200) if index in targets else (110, 110, 110))
        line(7 + offset, f"{mark} KP{index}  {label}", colour)
    line(17, "v visible   o occluded", (170, 170, 170))
    line(18, "t truncated u unknown", (170, 170, 170))
    line(19, "n/p next-prev  [ ] frame", (170, 170, 170))
    line(20, "c coord-flag  s save  q quit", (170, 170, 170))
    line(21, "- / =  zoom out / in", (170, 170, 170))
    flag = amendment.get("coordinate_review_flag")
    if flag:
        line(22, "COORDINATE REVIEW FLAGGED", (60, 200, 255))

    zoom = magnifier(image, points, cursor, targets, amendment, box)
    if (ZOOM_TOP + ZOOM_VIEW <= canvas.shape[0]
            and left + ZOOM_VIEW <= canvas.shape[1]):
        canvas[ZOOM_TOP:ZOOM_TOP + ZOOM_VIEW, left:left + ZOOM_VIEW] = zoom
    else:  # 배치가 어긋나면 조용히 사라지지 않게 알린다
        cv2.putText(canvas, "ZOOM PANEL DID NOT FIT", (left, ZOOM_TOP + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 230), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-keypoints", action="store_true",
                        help="자동 확정된 점까지 전부 편집 대상으로 연다")
    args = parser.parse_args()

    rows = list(csv.DictReader(QUEUE.open(encoding="utf-8")))
    queue_by_frame: dict[str, list[dict]] = {}
    for row in rows:
        queue_by_frame.setdefault(row["frame_id"], []).append(row)

    review_frames = [
        frame_id for frame_id, items in queue_by_frame.items()
        if args.all_keypoints or any(i["requires_human"] == "true" for i in items)
    ]
    if not review_frames:
        print("사람이 볼 프레임이 없다.")
        return 0

    index_rows = {
        row["frame_id"]: row
        for row in csv.DictReader(
            (WORKSPACE / "review" / "DAYTIME_VISIBILITY_REVIEW_QUEUE.csv").open())
    }
    amendments = load_amendments()
    total_targets = sum(
        1 for row in rows
        if args.all_keypoints or row["requires_human"] == "true"
    )
    print(f"frames {len(review_frames)}   keypoints to confirm {total_targets}")
    print("모델 예측은 표시하지 않는다.  좌표는 편집할 수 없다.")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    frame_index = 0
    box = ZOOM_BOX
    while True:
        frame_id = review_frames[frame_index]
        meta = index_rows[frame_id]
        annotation = json.loads((WORKSPACE / meta["annotation_path"]).read_text())
        points = annotation["objects"][0]["keypoint_annotations"]
        image = cv2.imread(str(WORKSPACE / meta["image_path"]))
        if image is None:
            print(f"이미지를 못 읽었다: {meta['image_path']}")
            return 2

        targets = [
            int(row["kp_index"]) for row in queue_by_frame[frame_id]
            if args.all_keypoints or row["requires_human"] == "true"
        ]
        amendment = amendments["frames"].setdefault(frame_id, {})
        cursor = targets[0]

        while True:
            canvas = draw(image, points, meta, targets, cursor, amendment,
                          frame_index, len(review_frames), box)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            char = chr(key) if 32 <= key < 127 else ""

            if char in STATES:
                amendment[str(cursor)] = char
                position = targets.index(cursor) if cursor in targets else -1
                if 0 <= position < len(targets) - 1:
                    cursor = targets[position + 1]
            elif char == "n":
                position = targets.index(cursor) if cursor in targets else -1
                cursor = targets[(position + 1) % len(targets)]
            elif char == "p":
                position = targets.index(cursor) if cursor in targets else 0
                cursor = targets[(position - 1) % len(targets)]
            elif char == "-":
                box = min(ZOOM_BOX_MAX, box + 30)   # 잘라내는 상자가 커지면 배율은 준다
            elif char in "=+":
                box = max(ZOOM_BOX_MIN, box - 30)
            elif char == "c":
                amendment["coordinate_review_flag"] = not amendment.get(
                    "coordinate_review_flag", False)
            elif char == "s":
                save_amendments(amendments)
                print(f"saved -> {AMENDMENTS.relative_to(REPO_ROOT)}")
            elif char == "]":
                frame_index = (frame_index + 1) % len(review_frames)
                break
            elif char == "[":
                frame_index = (frame_index - 1) % len(review_frames)
                break
            elif char == "q":
                save_amendments(amendments)
                cv2.destroyAllWindows()
                confirmed = sum(
                    1 for frame in amendments["frames"].values()
                    for key_name in frame if key_name.isdigit()
                )
                print(f"\nsaved -> {AMENDMENTS.relative_to(REPO_ROOT)}")
                print(f"confirmed keypoints {confirmed} / {total_targets}")
                print("GT 는 아직 바뀌지 않았다.  반영은 "
                      "scripts/annotate/apply_visibility_amendments.py 가 한다.")
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
