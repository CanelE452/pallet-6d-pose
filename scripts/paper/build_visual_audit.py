"""corner 회귀를 눈으로 확인할 contact sheet 를 만든다.  진단 전용이다.

숫자만 보고 원인을 정하지 않는다.  어떤 프레임에서 무엇이 어긋나는지 본다.

PAPER_EVAL (GT 있음)
    A  BOTH_DETECTED 에서 R5 가 R0 보다 나빠진 상위 20
    B  좋아진 상위 20
    C  Night 에서 R5 만 검출한 프레임 전부
    D  Proposed PASS 인데 gross > 20 px
    E  Proposed REJECT 이고 gross > 20 px

U_MAIN (GT 없음 — QUALITATIVE_ONLY)
    F/G  Proposed accepted  Day 20 / Night 20
    H/I  Confidence 는 통과했는데 Proposed 가 버린 것  Day 20 / Night 20

오버레이: GT(초록) · R0(파랑) · R5(빨강) · per-kp error · visibility ·
box_conf · s_reproj / s_remove / s_flip.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo"))
from pseudo_label_filters import geometry_scores  # noqa: E402

RESULTS = REPO_ROOT / "data" / "pallet" / "results"
DIAGNOSIS = RESULTS / "paper_eval_v1" / "REGRESSION_DIAGNOSIS.json"
ARMS_DIR = RESULTS / "paper_eval_v1" / "arms"
M4_RECORDS = RESULTS / "paper_selftrain_v1" / "M4_FRAME_RECORDS.json"
TEACHER = RESULTS / "paper_selftrain_v1" / "teacher_cache" / "R0_TEACHER_CACHE.json"
SCORED = RESULTS / "paper_selftrain_v1" / "pseudo_manifests" / "ALL_SCORED.csv"
LOCK = REPO_ROOT / "data" / "pallet" / "results" / "paper_selftrain_v1" / "SELFTRAIN_EXPOSURE_LOCK.json"
FILTER_LOCK = REPO_ROOT / "data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json"
OUT = RESULTS / "paper_eval_v1" / "visual_audit"

PAD, IMGSZ, CONF_FLOOR = 100, 640, 0.001
FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6, 8]
GROSS_PX = 20.0
TOP_N = 20
CELL = (480, 360)
COLUMNS = 4

GREEN, BLUE, RED, GREY = (0, 220, 0), (255, 160, 0), (0, 0, 255), (170, 170, 170)


def canonical(frame_id: str) -> str:
    return frame_id.replace("__", ":")


def parse_errors(value: str) -> list[float]:
    return [float(part) for part in value.split(";") if part] if value else []


def load_per_frame(name: str) -> dict[str, dict]:
    path = ARMS_DIR / f"{name}_per_frame.csv"
    return {canonical(row["frame_id"]): row
            for row in csv.DictReader(path.open(encoding="utf-8"))
            if row["kind"] == "POSITIVE"}


# ── 오버레이 ────────────────────────────────────────────────────────────

def draw_points(canvas, points, valid, colour, radius=4, label=None):
    for index, point in enumerate(points):
        if point is None or not np.isfinite(point).all():
            continue
        if valid is not None and not valid[index]:
            continue
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(canvas, (x, y), radius, colour, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    if label:
        pass


def banner(canvas, lines, colour=(255, 255, 255)):
    y = 16
    for line in lines:
        cv2.putText(canvas, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    colour, 1, cv2.LINE_AA)
        y += 15


def render_cell(image, overlays: list[tuple], caption: list[str]) -> np.ndarray:
    canvas = image.copy()
    for points, valid, colour in overlays:
        if points is not None:
            draw_points(canvas, np.asarray(points, dtype=float), valid, colour)
    height, width = canvas.shape[:2]
    scale = min(CELL[0] / width, CELL[1] / height)
    canvas = cv2.resize(canvas, (int(width * scale), int(height * scale)))
    cell = np.zeros((CELL[1], CELL[0], 3), dtype=np.uint8)
    cell[: canvas.shape[0], : canvas.shape[1]] = canvas
    banner(cell, caption)
    cv2.rectangle(cell, (0, 0), (CELL[0] - 1, CELL[1] - 1), GREY, 1)
    return cell


def contact_sheet(cells: list[np.ndarray], path: Path, title: str) -> None:
    if not cells:
        path.write_bytes(b"")
        return
    rows = (len(cells) + COLUMNS - 1) // COLUMNS
    sheet = np.zeros((rows * CELL[1] + 30, COLUMNS * CELL[0], 3), dtype=np.uint8)
    cv2.putText(sheet, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    for index, cell in enumerate(cells):
        r, c = divmod(index, COLUMNS)
        sheet[30 + r * CELL[1]: 30 + (r + 1) * CELL[1],
              c * CELL[0]: (c + 1) * CELL[0]] = cell
    cv2.imwrite(str(path), sheet)


# ── 예측 ────────────────────────────────────────────────────────────────

def predict(model, image, kp_threshold: float):
    padded = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    result = model.predict(padded, imgsz=IMGSZ, conf=CONF_FLOOR, verbose=False)[0]
    if result.boxes is None or not len(result.boxes):
        return None
    best = int(np.argmax(result.boxes.conf.cpu().numpy()))
    keypoints = result.keypoints.xy.cpu().numpy()[best] - PAD
    confidences = (result.keypoints.conf.cpu().numpy()[best]
                   if result.keypoints.conf is not None else np.zeros(9))
    return {
        "keypoints": keypoints,
        "valid": np.nan_to_num(confidences, nan=0.0) >= kp_threshold,
        "box_conf": float(result.boxes.conf.cpu().numpy()[best]),
    }


def main() -> int:
    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    diagnosis = json.loads(DIAGNOSIS.read_text())
    m4 = json.loads(M4_RECORDS.read_text())
    lock = json.loads(FILTER_LOCK.read_text())
    kp_threshold = float(lock["keypoint_validity"]["kp_conf_threshold"])
    thresholds = lock["geometry_thresholds"]
    tau_box = float(lock["TAU_BOX"])

    base_rows = load_per_frame("R0")
    prop_rows = load_per_frame("R5_PROPOSED")
    m4_by_frame = {canonical(f["frame_id"]): f for f in m4["frames"]}

    # ── 선정 ────────────────────────────────────────────────────────
    deltas: list[tuple[float, str]] = []
    for domain, block in diagnosis["domains"].items():
        for frame in block["frames"]["BOTH_DETECTED"]:
            a = parse_errors(base_rows[frame]["top_keypoint_supervised_errors_px"])
            b = parse_errors(prop_rows[frame]["top_keypoint_supervised_errors_px"])
            if len(a) == len(b) and a:
                deltas.append((float(np.median(b)) - float(np.median(a)), frame))
    deltas.sort(reverse=True)
    worse = [f for _, f in deltas[:TOP_N]]
    better = [f for _, f in deltas[-TOP_N:]][::-1]
    night_only = diagnosis["domains"]["nighttime"]["frames"]["R5_ONLY"]

    def m4_gross(frame_id: str) -> int:
        record = m4_by_frame.get(frame_id)
        return int(record.get("gross_keypoints") or 0) if record else 0

    pass_gross = [f for f, r in m4_by_frame.items()
                  if r["verdict"].get("F4_PROPOSED") and m4_gross(f) > 0]
    reject_gross = [f for f, r in m4_by_frame.items()
                    if not r["verdict"].get("F4_PROPOSED") and m4_gross(f) > 0]

    sets = [
        ("A_WORSE_TOP20", worse, "BOTH_DETECTED - top 20 where R5 is worse than R0"),
        ("B_BETTER_TOP20", better, "BOTH_DETECTED - top 20 where R5 is better"),
        ("C_NIGHT_R5_ONLY", night_only, "Night - every frame only R5 detected"),
        ("D_PROPOSED_PASS_GROSS", pass_gross[:40],
         f"Proposed PASS but gross > {GROSS_PX:.0f} px"),
        ("E_PROPOSED_REJECT_GROSS", reject_gross[:40],
         f"Proposed REJECT and gross > {GROSS_PX:.0f} px"),
    ]

    needed = sorted({f for _, frames, _ in sets for f in frames})
    print(f"eval 프레임 {len(needed)} 장에 대해 R0·R5 추론", flush=True)

    cache_path = OUT / "AUDIT_PREDICTIONS.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    models: dict = {}
    if all(frame in cache for frame in needed):
        print("  캐시 적중 — 추론을 건너뛴다", flush=True)
    else:
        for name, path in (("R0", REPO_ROOT / (
                "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
                "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")),
                ("R5", REPO_ROOT / "challenge/yolo_pose_one_model/paper_selftrain_v1"
                 / "R5_PROPOSED__FULL" / "weights" / "last.pt")):
            if not path.exists():
                raise SystemExit(f"CHECKPOINT_MISSING: {path}")
            models[name] = YOLO(str(path), task="pose")
    predictions: dict[str, dict] = {}
    for index, frame in enumerate(needed):
        if frame in cache:
            entry = {"image_path": cache[frame]["image_path"]}
            for name in ("R0", "R5"):
                blob = cache[frame].get(name)
                entry[name] = None if blob is None else {
                    "keypoints": np.asarray(blob["keypoints"], dtype=float),
                    "valid": np.asarray(blob["valid"], dtype=bool),
                    "box_conf": blob["box_conf"],
                }
            predictions[frame] = entry
            continue
        record = m4_by_frame.get(frame)
        row = base_rows.get(frame)
        image_path = REPO_ROOT / (record["image_path"] if record
                                  else row["image"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"UNREADABLE_IMAGE: {image_path}")
        entry = {"image_path": str(image_path)}
        for name, model in models.items():
            entry[name] = predict(model, image, kp_threshold)
        predictions[frame] = entry
        cache[frame] = {
            "image_path": entry["image_path"],
            **{name: (None if entry[name] is None else {
                "keypoints": entry[name]["keypoints"].tolist(),
                "valid": entry[name]["valid"].tolist(),
                "box_conf": entry[name]["box_conf"]})
               for name in ("R0", "R5")},
        }
        if (index + 1) % 20 == 0:
            print(f"  {index + 1}/{len(needed)}", flush=True)
    cache_path.write_text(json.dumps(cache) + "\n")

    # ── 렌더 ────────────────────────────────────────────────────────
    index_rows = []
    for name, frames, title in sets:
        cells = []
        for frame in frames:
            entry = predictions.get(frame)
            if entry is None:
                continue
            image = cv2.imread(entry["image_path"])
            record = m4_by_frame.get(frame)
            gt = np.asarray(record["gt_xy"], dtype=float) if record else None
            supervised = (np.asarray(record["gt_supervised"], dtype=bool)
                          if record else None)
            r0, r5 = entry.get("R0"), entry.get("R5")
            a = parse_errors(base_rows[frame]["top_keypoint_supervised_errors_px"]) \
                if frame in base_rows else []
            b = parse_errors(prop_rows[frame]["top_keypoint_supervised_errors_px"]) \
                if frame in prop_rows else []
            caption = [
                frame,
                f"R0 med {np.median(a):.1f}px  R5 med {np.median(b):.1f}px"
                if a and b else
                (f"R5 med {np.median(b):.1f}px (R0 top-1 IoU<0.5)" if b else "no error"),
            ]
            if record:
                caption.append(
                    f"box_conf {record['box_conf']:.3f}  gross {record.get('gross_keypoints')}"
                    if record.get("box_conf") is not None else "not detected")
                caption.append(
                    "s_reproj {} s_remove {} s_flip {}".format(
                        *[("—" if record.get(k) is None else f"{record[k]:.4f}")
                          for k in ("s_reproj", "s_remove", "s_flip")]))
                caption.append(
                    f"visible {int(np.count_nonzero(supervised))}/9"
                    if supervised is not None else "")
            overlays = [(gt, supervised, GREEN)]
            if r0 is not None:
                overlays.append((r0["keypoints"], r0["valid"], BLUE))
            if r5 is not None:
                overlays.append((r5["keypoints"], r5["valid"], RED))
            cells.append(render_cell(image, overlays, caption))
        path = OUT / f"{name}.jpg"
        contact_sheet(cells, path, f"{title}   [GT green | R0 blue | R5 red]")
        index_rows.append((name, len(cells), title, "GT_SCORED"))
        print(f"  wrote {path.name}  ({len(cells)} frames)", flush=True)

    # ── U_MAIN (GT 없음) ────────────────────────────────────────────
    teacher = {e["image_path"]: e for e in json.loads(TEACHER.read_text())["entries"]}
    scored = list(csv.DictReader(SCORED.open(encoding="utf-8")))

    def accepted(row: dict, arm: str) -> bool:
        if row["detected"] != "True" or int(row["valid_corners"]) < 6:
            return False
        if float(row["box_conf"]) < tau_box:
            return False
        if arm == "CONF":
            return True
        try:
            return (float(row["s_remove"]) <= float(thresholds["tau_remove"])
                    and float(row["s_flip"]) <= float(thresholds["tau_flip"]))
        except (TypeError, ValueError):
            return False

    pool_sets = []
    for condition in ("daytime", "nighttime"):
        rows = [r for r in scored if r["paper_condition"] == condition]
        prop = [r for r in rows if accepted(r, "PROPOSED")][:TOP_N]
        gap = [r for r in rows
               if accepted(r, "CONF") and not accepted(r, "PROPOSED")][:TOP_N]
        pool_sets += [
            (f"F_POOL_PROPOSED_ACCEPTED_{condition.upper()}", prop,
             f"U_MAIN {condition} - Proposed accepted"),
            (f"H_POOL_CONF_ONLY_{condition.upper()}", gap,
             f"U_MAIN {condition} - Confidence accepted, Proposed rejected"),
        ]

    for name, rows, title in pool_sets:
        cells = []
        for row in rows:
            image = cv2.imread(str(REPO_ROOT / row["image_path"]))
            if image is None:
                continue
            entry = teacher.get(row["image_path"])
            points = (np.asarray(entry["top1"]["keypoints_xy"], dtype=float)
                      if entry and entry.get("top1") else None)
            caption = [
                Path(row["image_path"]).stem,
                f"box_conf {float(row['box_conf']):.3f}  valid {row['valid_corners']}/8",
                "s_reproj {} s_remove {} s_flip {}".format(
                    *[("—" if not row[k] else f"{float(row[k]):.4f}")
                      for k in ("s_reproj", "s_remove", "s_flip")]),
                "QUALITATIVE_ONLY - no GT",
            ]
            cells.append(render_cell(image, [(points, None, RED)], caption))
        path = OUT / f"{name}.jpg"
        contact_sheet(cells, path, f"{title}   [teacher R0 pseudo-label in red]")
        index_rows.append((name, len(cells), title, "QUALITATIVE_ONLY"))
        print(f"  wrote {path.name}  ({len(cells)} frames)", flush=True)

    lines = ["# Visual audit — contact sheets", "",
             "진단 전용이다.  이 문서를 근거로 threshold·pool·model 을 바꾸지 않는다.", "",
             "오버레이: GT 초록 · R0 파랑 · R5 빨강.  숫자는 keypoint index 다.", "",
             "```text",
             f"{'sheet':38} {'frames':>7}  {'scoring':16} title", "─" * 96]
    for name, count, title, scoring in index_rows:
        lines.append(f"{name:38} {count:7d}  {scoring:16} {title}")
    lines += ["```", "",
              "`QUALITATIVE_ONLY` 는 GT 가 없어 정오를 판정할 수 없다는 뜻이다 — ",
              "이 시트로 정량 주장을 하지 않는다.", "",
              f"파일: `{OUT.relative_to(REPO_ROOT)}/`", ""]
    (REPO_ROOT / "_docs" / "paper" / "generated" / "VISUAL_AUDIT.md").write_text(
        "\n".join(lines) + "\n")
    print("wrote _docs/paper/generated/VISUAL_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
