"""사람이 물리적 긴 축이 A 인지 B 인지만 고르는 검수 도구.

    python3 scripts/paper/pose_metric_closure_v1/review_physical_axis.py

물어보는 것은 하나뿐이다.

    Plastic  130 cm 긴 변이 Axis A 인가 Axis B 인가
    Wood      80 cm 긴 변이 Axis A 인가 Axis B 인가

키포인트를 다시 찍지 않고, yaw 를 숫자로 넣지 않고, 회전행렬을 입력하지 않는다.

**모델 예측은 이 화면에 절대 나오지 않는다.**  R0/R5 예측, selector 점수, ADD, yaw
오차 어느 것도 표시하지 않는다.  사람 판정이 모델 결과에 오염되면 안 되기 때문이다.

원본 annotation 은 수정하지 않는다.  판정은 sidecar 에만 쌓인다.
"""

from __future__ import annotations

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
PROGRESS = OUT_DIR / "AXIS_REVIEW_PROGRESS.json"

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


def load_labels() -> dict:
    if LABELS.exists():
        return json.loads(LABELS.read_text())
    return {
        "schema_version": "physical_axis_review_v1",
        "review_definition":
            "physical long-axis identity only; 180-degree sign is not annotated",
        "source_annotations_modified": False,
        "model_predictions_shown_to_reviewer": False,
        "frames": {},
    }


def save(labels: dict, index: int, total: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels["updated_utc"] = now()
    LABELS.write_text(json.dumps(labels, indent=2) + "\n")
    entries = labels["frames"].values()
    PROGRESS.write_text(json.dumps({
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


def solve_hypothesis(points: np.ndarray, camera: np.ndarray,
                     across: float, along: float, height: float):
    """Fit the cuboid under one W/D assignment and return reprojected corners.

    Uses only the annotated keypoints, the intrinsics and the physical dimensions.
    No ground-truth pose is read.
    """

    half_a, half_b, half_h = across / 2.0, along / 2.0, height / 2.0
    # corner order must match camera-facing 0123:
    # 0,1 near top   2,3 near bottom   4,5 far top   6,7 far bottom
    model = np.array([
        [-half_a, -half_h, -half_b], [+half_a, -half_h, -half_b],
        [+half_a, +half_h, -half_b], [-half_a, +half_h, -half_b],
        [-half_a, -half_h, +half_b], [+half_a, -half_h, +half_b],
        [+half_a, +half_h, +half_b], [-half_a, +half_h, +half_b],
    ], dtype=np.float64)
    usable = np.isfinite(points[:8]).all(axis=1)
    if usable.sum() < 6:
        return None
    try:
        ok, rvec, tvec = cv2.solvePnP(
            model[usable], points[:8][usable].astype(np.float64), camera, None,
            flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(
        model[usable], points[:8][usable].astype(np.float64), camera, None, rvec, tvec)
    projected, _ = cv2.projectPoints(model, rvec, tvec, camera, None)
    projected = projected.reshape(-1, 2)
    residual = float(np.linalg.norm(
        projected[usable] - points[:8][usable], axis=1).mean())
    return projected, residual


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

    if overlay_mode:  # V key: show the cuboid implied by each hypothesis
        raw = frame.get("_intrinsics")
        if raw:
            camera = np.array([[raw["fx"], 0, raw["cx"]],
                               [0, raw["fy"], raw["cy"]], [0, 0, 1]], np.float64)
            long_m, short_m = frame["physical_long_m"], frame["physical_short_m"]
            hgt = frame.get("_height_m", 0.11)
            across, along = ((long_m, short_m) if overlay_mode == 1
                             else (short_m, long_m))
            solved = solve_hypothesis(points, camera, across, along, hgt)
            if solved is not None:
                projected, residual = solved
                shown = projected * scale
                colour = COL_A if overlay_mode == 1 else COL_B
                for a, b in CUBOID_EDGES:
                    pa = (int(round(shown[a][0])), int(round(shown[a][1])))
                    pb = (int(round(shown[b][0])), int(round(shown[b][1])))
                    cv2.line(view, pa, pb, colour, 1, cv2.LINE_AA)
                tag = "A" if overlay_mode == 1 else "B"
                text(view, f"hypothesis {tag} is LONG   fit residual {residual:5.1f} px",
                     (14, 26), colour, 0.62, 2)

    for a, b in AXIS_A_EDGES:
        if finite[a] and finite[b]:
            cv2.line(view, point(a), point(b), COL_A, 3, cv2.LINE_AA)
    for a, b in AXIS_B_EDGES:
        if finite[a] and finite[b]:
            cv2.line(view, point(a), point(b), COL_B, 3, cv2.LINE_AA)

    if finite[8]:
        centre = np.array(point(8), float)
        for edges, colour, tag in ((AXIS_A_EDGES, COL_A, "A"),
                                   (AXIS_B_EDGES, COL_B, "B")):
            vectors = [pts[b] - pts[a] for a, b in edges if finite[a] and finite[b]]
            if not vectors:
                continue
            direction = np.mean(vectors, axis=0)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            direction = direction / norm * float(np.clip(norm * 0.75, 90.0, 190.0))
            tip = centre + direction
            cv2.arrowedLine(view, tuple(centre.astype(int)), tuple(tip.astype(int)),
                            colour, 4, cv2.LINE_AA, tipLength=0.18)
            cv2.arrowedLine(view, tuple(centre.astype(int)),
                            tuple((centre - direction).astype(int)),
                            colour, 4, cv2.LINE_AA, tipLength=0.18)
            label = tuple((tip + direction * 0.22).astype(int))
            cv2.circle(view, tuple(tip.astype(int)), 3, colour, -1, cv2.LINE_AA)
            text(view, tag, label, (0, 0, 0), 1.15, 6)
            text(view, tag, label, colour, 1.15, 3)

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
                 "[V] cuboid overlay    [Backspace] clear",
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
    if not MANIFEST.exists():
        print("manifest missing — run build_axis_review_manifest.py first")
        return 1
    manifest = json.loads(MANIFEST.read_text())
    frames = manifest["frames_list"]
    total = len(frames)

    # intrinsics 는 overlay 에만 쓰이므로 여기서 붙인다 (manifest 를 키우지 않는다)
    for frame in frames:
        payload = json.loads((REPO_ROOT / frame["annotation"]).read_text())
        camera = payload.get("camera_data", {})
        frame["_intrinsics"] = camera.get("intrinsics")
        dims = payload["objects"][0].get("physical_dimensions_m") or {}
        frame["_height_m"] = float(dims.get("y", 0.11))

    labels = load_labels()
    index = next((i for i, f in enumerate(frames)
                  if f["frame_id"] not in labels["frames"]), 0)
    overlay_mode = 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print(f"{total} frames.  starting at {index + 1}.  "
          f"already reviewed: {len(labels['frames'])}")

    while True:
        frame = frames[index]
        entry = labels["frames"].get(frame["frame_id"])
        entries = labels["frames"].values()
        progress = (len(labels["frames"]),
                    sum(1 for e in entries if e.get("status") == "UNCLEAR"))
        cv2.imshow(WINDOW, draw_frame(frame, entry, index, total, progress, overlay_mode))
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
            save(labels, index, total)

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
            save(labels, index, total)
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
            save(labels, index, total)
            print(f"saved — {len(labels['frames'])}/{total}")
        elif key in (ord("q"), ord("Q"), 27):
            save(labels, index, total)
            break

    cv2.destroyAllWindows()
    entries = labels["frames"].values()
    print(f"reviewed  {len(labels['frames'])}/{total}")
    print(f"confirmed {sum(1 for e in entries if e.get('status') == 'CONFIRMED')}")
    print(f"unclear   {sum(1 for e in entries if e.get('status') == 'UNCLEAR')}")
    print(f"labels    {LABELS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
