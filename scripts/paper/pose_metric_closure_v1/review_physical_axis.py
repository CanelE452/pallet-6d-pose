"""사람이 물리적 긴 축이 A 인지 B 인지만 고르는 검수 도구.

    python3 scripts/paper/pose_metric_closure_v1/review_physical_axis.py

물어보는 것은 하나뿐이다.

    Plastic  130 cm 긴 변이 Axis A 인가 Axis B 인가
    Wood      80 cm 긴 변이 Axis A 인가 Axis B 인가

키포인트를 다시 찍지 않고, yaw 를 숫자로 넣지 않고, 회전행렬을 입력하지 않는다.

**모델 예측은 이 화면에 절대 나오지 않는다.**  R0/R5 예측, selector 점수, ADD, yaw
오차 어느 것도 표시하지 않는다.  사람 판정이 모델 결과에 오염되면 안 되기 때문이다.

PnP 적합·잔차도 이 경로에 존재하지 않는다.  "어느 가설이 더 잘 맞는가" 를 보여주면
사람이 독립적으로 판단하지 않고 적합도를 따라가게 된다.  `V` 오버레이는 각 가설이
**어느 방향을 긴 축이라고 부르는지** 만 굵기로 보여준다 (`test_review_gui_leakage.py` 가
AST 로 강제).

원본 annotation 은 수정하지 않는다.  판정은 sidecar 에만 쌓인다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
MANIFEST = OUT_DIR / "AXIS_REVIEW_MANIFEST.json"
LABELS = OUT_DIR / "AXIS_REVIEW_LABELS.json"
SMOKE_LABELS = OUT_DIR / "AXIS_REVIEW_LABELS_SMOKE.json"
PROGRESS = OUT_DIR / "AXIS_REVIEW_PROGRESS.json"
SMOKE_PROGRESS = OUT_DIR / "AXIS_REVIEW_PROGRESS_SMOKE.json"
RECHECK_LIST = OUT_DIR / "AXIS_RECHECK_LIST.json"
RECHECK_LABELS = OUT_DIR / "AXIS_REVIEW_LABELS_RECHECK.json"
RECHECK_PROGRESS = OUT_DIR / "AXIS_REVIEW_PROGRESS_RECHECK.json"

WINDOW = "physical axis review"
MAX_W, MAX_H = 1500, 820
HEADER_H, FOOTER_H = 112, 132
MIN_CANVAS_W = 1180

COL_A = (80, 200, 255)      # amber, BGR
COL_B = (255, 170, 90)      # blue
COL_KP = (255, 255, 255)
COL_OK = (110, 240, 130)
COL_WARN = (110, 170, 255)
COL_BG = (28, 28, 30)

AXIS_A_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7)]
AXIS_B_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]
CUBOID_EDGES = AXIS_A_EDGES + AXIS_B_EDGES + [(0, 3), (1, 2), (4, 7), (5, 6)]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_labels(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "schema_version": "physical_axis_review_v1",
        "review_definition":
            "physical long-axis identity only; 180-degree sign is not annotated",
        "source_annotations_modified": False,
        "model_predictions_shown_to_reviewer": False,
        "frames": {},
    }


def save(labels: dict, index: int, total: int, *,
         labels_path: Path = LABELS, progress_path: Path = PROGRESS) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels["updated_utc"] = now()
    labels_path.write_text(json.dumps(labels, indent=2) + "\n")
    entries = labels["frames"].values()
    progress_path.write_text(json.dumps({
        "schema_version": "physical_axis_review_progress_v1",
        "total": total,
        "reviewed": len(labels["frames"]),
        "confirmed": sum(1 for e in entries if e.get("status") == "CONFIRMED"),
        "unclear": sum(1 for e in entries if e.get("status") == "UNCLEAR"),
        "last_index": index,
        "updated_utc": now(),
    }, indent=2) + "\n")


def text(canvas, message, origin, colour=(235, 235, 235), scale=0.6, weight=1):
    """ASCII only — OpenCV renders non-ASCII as '???'."""

    cv2.putText(canvas, message, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, weight, cv2.LINE_AA)


def draw_frame(frame: dict, entry: dict | None, index: int, total: int,
               progress: tuple[int, int], overlay_mode: int) -> np.ndarray:
    image = cv2.imread(str(REPO_ROOT / frame["image"]))
    if image is None:
        image = np.full((480, 640, 3), 40, np.uint8)
        text(image, "IMAGE NOT FOUND", (30, 240), (90, 90, 255), 0.9, 2)

    points = np.array([p if p else [np.nan, np.nan]
                       for p in frame["keypoints_xy"]], dtype=np.float64)
    height, width = image.shape[:2]
    scale = min(MAX_W / width, (MAX_H - HEADER_H - FOOTER_H) / height, 1.0)
    view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pts = points * scale

    def point(i):
        return (int(round(pts[i][0])), int(round(pts[i][1])))

    finite = np.isfinite(pts).all(axis=1)

    if overlay_mode:
        # 설명용 오버레이다.  어느 가설이 더 잘 맞는지는 보여주지 않는다 —
        # 각 가설이 "어느 방향을 긴 축이라고 부르는지" 만 굵게 강조한다.
        # PnP 적합도, 잔차, 점수는 이 경로에 존재하지 않는다.
        long_edges, short_edges, long_colour, short_colour, tag = (
            (AXIS_A_EDGES, AXIS_B_EDGES, COL_A, COL_B, "A")
            if overlay_mode == 1 else
            (AXIS_B_EDGES, AXIS_A_EDGES, COL_B, COL_A, "B"))
        for a, b in short_edges:
            if finite[a] and finite[b]:
                cv2.line(view, point(a), point(b), (120, 120, 120), 1, cv2.LINE_AA)
        for a, b in long_edges:
            if finite[a] and finite[b]:
                cv2.line(view, point(a), point(b), long_colour, 9, cv2.LINE_AA)
        banner = (f"if Axis {tag} is the LONG side:  "
                  f"thick = {frame['physical_long_cm']} cm, "
                  f"thin = {frame['physical_short_cm']} cm")
        text(view, banner, (14, 28), (0, 0, 0), 0.62, 4)
        text(view, banner, (14, 28), long_colour, 0.62, 2)

    for a, b in AXIS_A_EDGES:
        if finite[a] and finite[b]:
            cv2.line(view, point(a), point(b), COL_A, 3, cv2.LINE_AA)
    for a, b in AXIS_B_EDGES:
        if finite[a] and finite[b]:
            cv2.line(view, point(a), point(b), COL_B, 3, cv2.LINE_AA)

    # 화살표는 그리지 않는다.  중심에서 뻗은 화살표가 팔레트 윗면을 가려
    # 정작 판단해야 할 물체가 안 보인다는 지적을 받았다.  축 구분은 선 색으로 한다
    # (상단 범례: A = 주황, B = 파랑).

    for i in range(9):
        if not finite[i]:
            continue
        cv2.circle(view, point(i), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(view, point(i), 4, COL_KP, -1, cv2.LINE_AA)
        text(view, str(i), (point(i)[0] + 7, point(i)[1] - 7), COL_KP, 0.52, 2)

    canvas = np.full((view.shape[0] + HEADER_H + FOOTER_H,
                      max(view.shape[1], MIN_CANVAS_W), 3), COL_BG, np.uint8)
    canvas[HEADER_H:HEADER_H + view.shape[0], :view.shape[1]] = view

    reviewed, unclear = progress
    text(canvas, f"Frame {index + 1} / {total}", (18, 34), (255, 255, 255), 0.80, 2)
    text(canvas, f"reviewed {reviewed}/{total}   unclear {unclear}",
         (18, 62), (170, 170, 170), 0.56, 1)
    text(canvas, f"{frame['session_id']}", (18, 88), (140, 140, 140), 0.5, 1)

    text(canvas, frame["display_name"], (300, 34), (225, 225, 225), 0.68, 2)
    text(canvas, f"LONG {frame['physical_long_cm']} cm"
                 f"    SHORT {frame['physical_short_cm']} cm",
         (300, 62), (215, 215, 215), 0.62, 1)
    text(canvas, str(frame["frame_id"])[:34], (300, 88), (140, 140, 140), 0.5, 1)

    text(canvas, "A = camera-facing WIDTH", (700, 34), COL_A, 0.56, 2)
    text(canvas, "edges 0-1  2-3  4-5  6-7", (700, 58), COL_A, 0.48, 1)
    text(canvas, "B = camera-facing DEPTH", (700, 82), COL_B, 0.56, 2)
    text(canvas, "edges 0-4  1-5  2-6  3-7", (700, 104), COL_B, 0.48, 1)

    base = HEADER_H + view.shape[0]
    question = (f"Which axis is the physical LONG side "
                f"({frame['physical_long_cm']} cm)?")
    text(canvas, question, (18, base + 32), (255, 255, 255), 0.74, 2)

    if entry is None:
        text(canvas, "not reviewed", (18, base + 64), COL_WARN, 0.62, 1)
    elif entry.get("status") == "UNCLEAR":
        text(canvas, "UNCLEAR", (18, base + 64), COL_WARN, 0.68, 2)
    else:
        axis = "A" if entry.get("long_axis") == "CF_WIDTH" else "B"
        note = "  (propagated)" if entry.get("propagated_by_session") else ""
        text(canvas, f"LONG = Axis {axis}{note}", (18, base + 64), COL_OK, 0.68, 2)

    text(canvas, "[A/1] long = A    [B/2] long = B    [U] unclear    "
                 "[V] which axis is long    [Backspace] clear",
         (18, base + 96), (185, 185, 185), 0.53, 1)
    text(canvas, "[N/Space/->] next   [P/<-] prev   [G] go to   [S] save   [Q/Esc] save and quit",
         (18, base + 120), (185, 185, 185), 0.53, 1)
    return canvas


def ask_index(total: int) -> int | None:
    print(f"go to frame (1-{total}): ", end="", flush=True)
    try:
        raw = sys.stdin.readline().strip()
    except Exception:
        return None
    if not raw.isdigit():
        return None
    value = int(raw)
    return value - 1 if 1 <= value <= total else None


def main() -> int:
    parser = argparse.ArgumentParser(description="physical long-axis review")
    parser.add_argument("--smoke", type=int, metavar="N", default=0,
                        help="첫 N 장만 연습한다.  결과는 본 검수와 분리된 "
                             "AXIS_REVIEW_LABELS_SMOKE.json 에 저장된다.")
    parser.add_argument("--recheck", action="store_true",
                        help="AXIS_RECHECK_LIST.json 의 프레임만 다시 본다.  "
                             "이전 답과 기하 판정은 화면에 표시하지 않는다.")
    args = parser.parse_args()

    if not MANIFEST.exists():
        print("manifest missing — run build_axis_review_manifest.py first")
        return 1
    manifest = json.loads(MANIFEST.read_text())
    frames = manifest["frames_list"]
    if args.smoke:
        frames = frames[:max(1, args.smoke)]
    if args.recheck:
        if not RECHECK_LIST.exists():
            print("recheck list missing — run validate_axis_review.py first")
            return 1
        wanted = set(json.loads(RECHECK_LIST.read_text())["frames"])
        frames = [f for f in frames if f["frame_id"] in wanted]
        if not frames:
            print("recheck list is empty")
            return 1
    labels_path = (RECHECK_LABELS if args.recheck
                   else SMOKE_LABELS if args.smoke else LABELS)
    progress_path = (RECHECK_PROGRESS if args.recheck
                     else SMOKE_PROGRESS if args.smoke else PROGRESS)
    total = len(frames)

    # 검수 화면은 annotation 을 다시 열지 않는다.  manifest 의 키포인트·치수면 충분하고,
    # 열지 않으면 저장된(미확인) parity 나 pose 가 이 경로로 새어들 수 없다.

    labels = load_labels(labels_path)
    if args.recheck:
        labels["recheck"] = True
        labels["blind"] = ("the previous answer and the geometric verdict are not shown; "
                           "this is a second independent look")
    if args.smoke:
        labels["smoke"] = True
        labels["not_for_evaluation"] = ("practice run; never merged into "
                                        "AXIS_REVIEW_LABELS.json")
    index = next((i for i, f in enumerate(frames)
                  if f["frame_id"] not in labels["frames"]), 0)
    overlay_mode = 0

    window = WINDOW + (" [RECHECK]" if args.recheck
                       else " [SMOKE]" if args.smoke else "")
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    if args.recheck:
        print(f"RECHECK — {total} frames flagged by an independent check.")
        print("Your previous answer is hidden on purpose. Judge each one fresh.")
    if args.smoke:
        print(f"SMOKE MODE — first {total} frames, saved separately to "
              f"{labels_path.name}")
    print(f"{total} frames.  starting at {index + 1}.  "
          f"already reviewed: {len(labels['frames'])}")

    while True:
        frame = frames[index]
        entry = labels["frames"].get(frame["frame_id"])
        # 재검수는 blind 다.  이번 세션에서 방금 누른 것만 보여주고, 1차 판정은
        # 절대 표시하지 않는다 — 보면 그대로 다시 누르게 된다.
        entries = labels["frames"].values()
        progress = (len(labels["frames"]),
                    sum(1 for e in entries if e.get("status") == "UNCLEAR"))
        cv2.imshow(window, draw_frame(frame, entry, index, total, progress, overlay_mode))
        key = cv2.waitKey(0) & 0xFF

        def record(long_axis: str | None, status: str) -> None:
            labels["frames"][frame["frame_id"]] = {
                "object_type": frame["object_type"],
                "session_id": frame["session_id"],
                "long_axis": long_axis,
                "short_axis": (None if long_axis is None
                               else ("CF_DEPTH" if long_axis == "CF_WIDTH" else "CF_WIDTH")),
                "status": status,
                "reviewer": "human",
                "review_timestamp": now(),
                "source": "manual_visual_review",
                "propagated_by_session": False,
            }
            save(labels, index, total, labels_path=labels_path,
                 progress_path=progress_path)

        if key in (ord("a"), ord("A"), ord("1")):
            record("CF_WIDTH", "CONFIRMED")
            index = min(index + 1, total - 1)
        elif key in (ord("b"), ord("B"), ord("2")):
            record("CF_DEPTH", "CONFIRMED")
            index = min(index + 1, total - 1)
        elif key in (ord("u"), ord("U")):
            record(None, "UNCLEAR")
            index = min(index + 1, total - 1)
        elif key in (8, 127):  # backspace / delete
            labels["frames"].pop(frame["frame_id"], None)
            save(labels, index, total, labels_path=labels_path,
                 progress_path=progress_path)
        elif key in (ord("v"), ord("V")):
            overlay_mode = (overlay_mode + 1) % 3
        elif key in (ord("n"), ord("N"), 32, 83, 84):
            index = min(index + 1, total - 1)
        elif key in (ord("p"), ord("P"), 81, 82):
            index = max(index - 1, 0)
        elif key in (ord("g"), ord("G")):
            target = ask_index(total)
            if target is not None:
                index = target
        elif key in (ord("s"), ord("S")):
            save(labels, index, total, labels_path=labels_path,
                 progress_path=progress_path)
            print(f"saved — {len(labels['frames'])}/{total}")
        elif key in (ord("q"), ord("Q"), 27):
            save(labels, index, total, labels_path=labels_path,
                 progress_path=progress_path)
            break

    cv2.destroyAllWindows()
    entries = labels["frames"].values()
    print(f"reviewed  {len(labels['frames'])}/{total}")
    print(f"confirmed {sum(1 for e in entries if e.get('status') == 'CONFIRMED')}")
    print(f"unclear   {sum(1 for e in entries if e.get('status') == 'UNCLEAR')}")
    print(f"labels    {labels_path.relative_to(REPO_ROOT)}")
    if args.smoke:
        print("SMOKE — these labels are separate and are never used for evaluation.")
        print("main review:  python scripts/paper/pose_metric_closure_v1/review_physical_axis.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
