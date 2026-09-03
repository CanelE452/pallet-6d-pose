"""검수 결과를 검증하고, 사람이 실수했을 만한 곳을 찾아준다.

    python3 scripts/paper/pose_metric_closure_v1/validate_axis_review.py
    python3 scripts/paper/pose_metric_closure_v1/validate_axis_review.py --contact-sheet

출력: AXIS_REVIEW_VALIDATION.json
      AXIS_REVIEW_RECHECK.md
      AXIS_REVIEW_CONTACT_SHEET/page_XXX.png   (--contact-sheet 일 때만)

**라벨을 자동으로 고치지 않는다.**  의심스러운 프레임 목록만 만들고, 판단은 사람이
다시 한다.  unclear 프레임을 조용히 버리지도 않는다 — coverage 를 항상 함께 낸다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
MANIFEST = OUT_DIR / "AXIS_REVIEW_MANIFEST.json"
LABELS = OUT_DIR / "AXIS_REVIEW_LABELS.json"
VALIDATION = OUT_DIR / "AXIS_REVIEW_VALIDATION.json"
RECHECK = OUT_DIR / "AXIS_REVIEW_RECHECK.md"
SHEET_DIR = OUT_DIR / "AXIS_REVIEW_CONTACT_SHEET"

EXPECTED_TOTAL = 319
EXPECTED_BY_OBJECT = {"plastic_standard_110x130x11": 194, "wood_small_80x59x14": 125}


def isolated_flips(sequence: list[tuple[str, str]]) -> list[str]:
    """A A A B A A 처럼 이웃 둘 사이에 하나만 튀는 프레임을 찾는다."""

    suspect = []
    for i in range(1, len(sequence) - 1):
        before, current, after = sequence[i - 1][1], sequence[i][1], sequence[i + 1][1]
        if current and before and after and before == after and current != before:
            suspect.append(sequence[i][0])
    return suspect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-sheet", action="store_true",
                        help="검수 확인용 썸네일 시트를 만든다")
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    if not MANIFEST.exists():
        print("manifest missing — run build_axis_review_manifest.py first")
        return 1
    if not LABELS.exists():
        print("no review yet — run review_physical_axis.py first")
        return 1

    manifest = json.loads(MANIFEST.read_text())
    frames = manifest["frames_list"]
    labels = json.loads(LABELS.read_text())["frames"]

    total = len(frames)
    by_object = Counter(f["object_type"] for f in frames)
    status = Counter()
    per_object_status: dict[str, Counter] = defaultdict(Counter)
    per_object_axis: dict[str, Counter] = defaultdict(Counter)
    per_session_axis: dict[str, Counter] = defaultdict(Counter)
    missing: list[str] = []
    unclear: list[str] = []
    propagated = 0

    for frame in frames:
        entry = labels.get(frame["frame_id"])
        if entry is None:
            missing.append(frame["frame_id"])
            status["MISSING"] += 1
            continue
        state = entry.get("status")
        status[state] += 1
        per_object_status[frame["object_type"]][state] += 1
        if entry.get("propagated_by_session"):
            propagated += 1
        if state == "UNCLEAR":
            unclear.append(frame["frame_id"])
            continue
        axis = entry.get("long_axis")
        per_object_axis[frame["object_type"]][axis] += 1
        per_session_axis[str(frame["session_id"])][axis] += 1

    # 세션 안에서 프레임 순서대로 정렬해 고립된 뒤집힘을 찾는다
    by_session: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for frame in sorted(frames, key=lambda f: (str(f["session_id"]), str(f["frame_id"]))):
        entry = labels.get(frame["frame_id"]) or {}
        by_session[str(frame["session_id"])].append(
            (frame["frame_id"], entry.get("long_axis") or ""))
    recheck: list[str] = []
    for session, sequence in by_session.items():
        recheck.extend(isolated_flips(sequence))

    # 저장돼 있던(미확인) parity 와 사람 판정이 얼마나 다른가 — 감사용
    agree = disagree = comparable = 0
    disagreements: list[str] = []
    for frame in frames:
        stored = frame.get("_hidden_stored_long_axis")
        entry = labels.get(frame["frame_id"])
        if not stored or not entry or entry.get("status") != "CONFIRMED":
            continue
        comparable += 1
        if entry.get("long_axis") == stored:
            agree += 1
        else:
            disagree += 1
            disagreements.append(frame["frame_id"])

    confirmed = status.get("CONFIRMED", 0)
    counts_ok = (total == EXPECTED_TOTAL
                 and all(by_object.get(k) == v for k, v in EXPECTED_BY_OBJECT.items()))

    report = {
        "schema_version": "physical_axis_review_validation_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "expected_total": EXPECTED_TOTAL,
        "counts_match_expected": counts_ok,
        "by_object": dict(by_object),
        "expected_by_object": EXPECTED_BY_OBJECT,
        "status": dict(status),
        "confirmed": confirmed,
        "unclear": len(unclear),
        "missing": len(missing),
        "axis_gt_coverage": confirmed / total if total else 0.0,
        "coverage_note":
            "unclear and missing frames are NOT silently dropped. Any pose table built "
            "from this review must print axis-GT coverage alongside its numbers.",
        "per_object_status": {k: dict(v) for k, v in per_object_status.items()},
        "per_object_axis": {k: dict(v) for k, v in per_object_axis.items()},
        "per_session_axis": {k: dict(v) for k, v in per_session_axis.items()},
        "propagated_by_session": propagated,
        "isolated_flips": recheck,
        "isolated_flip_note":
            "a frame whose label differs from both neighbours in the same session. "
            "Under the camera_dynamic convention this can be legitimate, so these are "
            "candidates for a second look, not errors.",
        "stored_parity_comparison": {
            "comparable": comparable,
            "agree": agree,
            "disagree": disagree,
            "disagreement_rate": (disagree / comparable) if comparable else None,
            "frames": disagreements[:80],
            "meaning":
                "the annotations already carried an unverified camera-facing parity "
                "(axis_assignment_confirmed was False on all 319). This measures how "
                "often the human disagreed with it. It was never shown during review.",
        },
        "missing_frames": missing[:80],
        "unclear_frames": unclear[:80],
        "gate": {
            "all_frames_reviewed": len(missing) == 0,
            "counts_match_expected": counts_ok,
            "ready_for_gt_build": len(missing) == 0 and counts_ok and confirmed > 0,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION.write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Frames to look at again", "",
             "자동으로 고치지 않았다.  아래만 다시 보면 된다.", ""]
    if recheck:
        lines += ["## Isolated flips", "",
                  "같은 세션에서 앞뒤와 다른 한 장. camera_dynamic 규약에서는 정당할 수도 있다.",
                  "", "```text"] + [f"  {f}" for f in recheck] + ["```", ""]
    if unclear:
        lines += ["## Marked unclear", "", "```text"] + \
                 [f"  {f}" for f in unclear] + ["```", ""]
    if missing:
        lines += ["## Not reviewed", "", "```text"] + \
                 [f"  {f}" for f in missing[:200]] + ["```", ""]
    if not (recheck or unclear or missing):
        lines += ["아무것도 없다.  검수가 깨끗하다.", ""]
    lines += [f"go-to 는 GUI 에서 `G` 키.  총 {len(set(recheck) | set(unclear) | set(missing))} 장."]
    RECHECK.write_text("\n".join(lines) + "\n")

    if args.contact_sheet:
        build_contact_sheet(frames, labels, args.columns, args.rows)

    print(f"total       {total} / {EXPECTED_TOTAL}   counts_ok={counts_ok}")
    for name, count in sorted(by_object.items()):
        print(f"  {name:32} {count:4d}  expected {EXPECTED_BY_OBJECT.get(name)}")
    print(f"confirmed   {confirmed}")
    print(f"unclear     {len(unclear)}")
    print(f"missing     {len(missing)}")
    print(f"coverage    {report['axis_gt_coverage']:.4f}")
    print(f"recheck     {len(recheck)} isolated flips")
    if comparable:
        print(f"stored parity disagreement  {disagree}/{comparable} "
              f"({disagree / comparable:.1%})")
    print(f"wrote {VALIDATION.relative_to(REPO_ROOT)}")
    print(f"wrote {RECHECK.relative_to(REPO_ROOT)}")
    return 0


def build_contact_sheet(frames, labels, columns: int, rows: int) -> None:
    import cv2
    import numpy as np

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    for existing in SHEET_DIR.glob("page_*.png"):
        existing.unlink()

    cell_w, cell_h, caption = 300, 230, 34
    per_page = columns * rows
    pages = 0
    for start in range(0, len(frames), per_page):
        chunk = frames[start:start + per_page]
        page = np.full((rows * (cell_h + caption), columns * cell_w, 3), 24, np.uint8)
        for slot, frame in enumerate(chunk):
            row, col = divmod(slot, columns)
            image = cv2.imread(str(REPO_ROOT / frame["image"]))
            if image is None:
                continue
            scale = min(cell_w / image.shape[1], cell_h / image.shape[0])
            thumb = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
            entry = labels.get(frame["frame_id"]) or {}
            axis = entry.get("long_axis")
            points = np.array([p if p else [np.nan, np.nan]
                               for p in frame["keypoints_xy"]], float) * scale
            edges = ([(0, 1), (2, 3), (4, 5), (6, 7)] if axis == "CF_WIDTH"
                     else [(0, 4), (1, 5), (2, 6), (3, 7)] if axis == "CF_DEPTH" else [])
            for a, b in edges:
                if np.isfinite(points[a]).all() and np.isfinite(points[b]).all():
                    cv2.line(thumb, tuple(points[a].astype(int)),
                             tuple(points[b].astype(int)), (110, 240, 130), 2, cv2.LINE_AA)
            y0 = row * (cell_h + caption)
            x0 = col * cell_w
            page[y0:y0 + thumb.shape[0], x0:x0 + thumb.shape[1]] = thumb
            tag = ("A" if axis == "CF_WIDTH" else "B" if axis == "CF_DEPTH"
                   else entry.get("status", "none"))
            label = f"{start + slot + 1} {frame['object_type'][:7]} long={tag}"
            cv2.putText(page, label, (x0 + 6, y0 + cell_h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (225, 225, 225), 1, cv2.LINE_AA)
        pages += 1
        cv2.imwrite(str(SHEET_DIR / f"page_{pages:03d}.png"), page)
    print(f"contact sheet  {pages} pages -> {SHEET_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
