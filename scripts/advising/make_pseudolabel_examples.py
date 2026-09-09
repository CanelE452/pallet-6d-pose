"""상담용 pseudo-label good/bad 예시를 만든다.  새 추론 0 — 저장된 기록만 쓴다.

    python3 scripts/advising/make_pseudolabel_examples.py

왜 pool 이 아니라 PAPER_EVAL 인가.
    self-training pool 프레임에는 manual GT 가 없어서 "잘 됐다/안 됐다" 를 눈대중으로만
    말할 수 있다.  `M4_FRAME_RECORDS.json` 은 같은 teacher(R0, sha 970a0913...)와 같은
    frozen 필터를 **GT 가 있는** PAPER_EVAL plastic positive 194 장에 적용해 두었다.
    그래서 good/bad 를 GT 대비 실제 오차로 고를 수 있다.

네 범주 (전부 기록된 값으로 정의한다 — 눈대중 아님)
    GOOD_ACCEPTED      필터가 통과시켰고 GT 대비 최대 코너 오차가 작은 것
    BAD_ACCEPTED       필터가 통과시켰는데 GT 대비 최대 코너 오차가 큰 것   ★핵심
    REJECTED_CORRECT   필터가 기각했고 실제로 오차가 컸던 것
    REJECTED_COSTLY    필터가 기각했는데 실제로는 정확했던 것            ★비용
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "data/pallet/results/paper_selftrain_v1/M4_FRAME_RECORDS.json"
QUALITY = ROOT / "data/pallet/results/paper_selftrain_v1/M4_FILTER_QUALITY.json"
OUT = ROOT / "_docs/advising/2026-09-professor-consult"
FILTER = "F4_PROPOSED"          # 논문의 proposed filter
PER_CATEGORY = 6

# camera-facing 0123 — 0~3 앞면, 4~7 뒷면, {0,1,4,5} 위 / {2,3,6,7} 아래, 8 centroid
NEAR = [(0, 1), (1, 2), (2, 3), (3, 0)]
FAR = [(4, 5), (5, 6), (6, 7), (7, 4)]
DEPTH = [(0, 4), (1, 5), (2, 6), (3, 7)]
PRED_NEAR, PRED_FAR, PRED_DEPTH = (80, 220, 255), (255, 170, 80), (170, 170, 170)
GT_COLOUR = (120, 255, 120)


def draw_cuboid(canvas, points, near, far, depth, thickness):
    for group, colour in ((NEAR, near), (FAR, far), (DEPTH, depth)):
        for a, b in group:
            pa, pb = points[a], points[b]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            cv2.line(canvas, tuple(np.int32(pa)), tuple(np.int32(pb)), colour,
                     thickness, cv2.LINE_AA)


def banner(canvas, lines):
    height = 20 * len(lines) + 10
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (8, 21 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    payload = json.loads(RECORDS.read_text())
    quality = json.loads(QUALITY.read_text())
    frames = [f for f in payload["frames"] if f.get("detected")]
    gross = float(payload["gross_px"])

    accepted = [f for f in frames if f["verdict"].get(FILTER)]
    rejected = [f for f in frames if not f["verdict"].get(FILTER)]
    by_max = lambda rows: sorted(rows, key=lambda r: r["corner_max_px"])

    categories = {
        "good_accepted": by_max(accepted)[:PER_CATEGORY],
        "bad_accepted": by_max(accepted)[-PER_CATEGORY:][::-1],
        "rejected_correct": by_max(rejected)[-PER_CATEGORY:][::-1],
        "rejected_costly": by_max(rejected)[:PER_CATEGORY],
    }
    titles = {
        "good_accepted": "ACCEPTED by the proposed filter, and accurate",
        "bad_accepted": "ACCEPTED by the proposed filter, but WRONG",
        "rejected_correct": "REJECTED, and it was right to reject",
        "rejected_costly": "REJECTED, but the label was actually accurate",
    }

    index = []
    for key, rows in categories.items():
        for rank, row in enumerate(rows, 1):
            image = cv2.imread(str(ROOT / row["image_path"]))
            if image is None:
                print(f"  이미지를 못 읽음: {row['image_path']}")
                continue
            canvas = image.copy()
            gt = np.asarray(row["gt_xy"], float)
            pred = np.asarray(row["keypoints_xy"], float)
            draw_cuboid(canvas, gt, GT_COLOUR, GT_COLOUR, GT_COLOUR, 1)
            draw_cuboid(canvas, pred, PRED_NEAR, PRED_FAR, PRED_DEPTH, 2)
            for i in range(min(8, len(pred))):
                if np.isfinite(pred[i]).all():
                    cv2.circle(canvas, tuple(np.int32(pred[i])), 4, PRED_NEAR, -1, cv2.LINE_AA)
            worst = int(np.argmax([np.linalg.norm(pred[i] - gt[i])
                                   if row["gt_supervised"][i] else -1 for i in range(8)]))
            cv2.line(canvas, tuple(np.int32(gt[worst])), tuple(np.int32(pred[worst])),
                     (60, 60, 255), 2, cv2.LINE_AA)

            banner(canvas, [
                f"{titles[key]}  (#{rank})",
                f"{row['frame_id']}  [{row['domain']}]",
                f"err vs GT: med {row['corner_median_px']:.1f}px"
                f"  max {row['corner_max_px']:.1f}px"
                f"  n>{gross:.0f}px: {row['gross_keypoints']}",
                f"conf {row['box_conf']:.3f}  reproj {row['s_reproj']:.4f}"
                f"  remove {row['s_remove']:.4f}  flip {row['s_flip']:.4f}",
                "green=GT   yellow/blue=prediction   red=worst corner",
            ])
            name = f"pseudolabel_{key}_{rank:02d}.png"
            cv2.imwrite(str(figures / name), canvas)
            index.append({
                "file": f"figures/{name}", "category": key, "rank": rank,
                "frame_id": row["frame_id"], "domain": row["domain"],
                "filter": FILTER,
                "accepted_by_filter": bool(row["verdict"].get(FILTER)),
                "corner_median_px": round(row["corner_median_px"], 3),
                "corner_max_px": round(row["corner_max_px"], 3),
                "gross_keypoints_over_20px": row["gross_keypoints"],
                "box_conf": round(row["box_conf"], 4),
                "s_reproj": round(row["s_reproj"], 5),
                "s_remove": round(row["s_remove"], 5),
                "s_flip": round(row["s_flip"], 5),
                "image_path": row["image_path"],
            })

    with (OUT / "manifests" / "PSEUDOLABEL_EXAMPLES.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    (OUT / "manifests" / "PSEUDOLABEL_EXAMPLES.json").write_text(json.dumps({
        "schema_version": "advising_pseudolabel_examples_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "new_inference": 0, "new_training": 0,
        "source_records": "data/pallet/results/paper_selftrain_v1/M4_FRAME_RECORDS.json",
        "source_summary": "data/pallet/results/paper_selftrain_v1/M4_FILTER_QUALITY.json",
        "population": payload["population"],
        "teacher_sha256": payload["teacher_sha256"],
        "filter_lock_sha256": payload["filter_lock_sha256"],
        "filter_shown": FILTER,
        "gross_px": gross,
        "counts": {"frames_detected": len(frames),
                   "accepted": len(accepted), "rejected": len(rejected),
                   "retention": round(len(accepted) / max(len(frames), 1), 4)},
        "filter_summary_from_artifact": quality["filters"].get(FILTER),
        "selection_rule": {
            "good_accepted": "accepted, smallest GT-vs-prediction max corner error",
            "bad_accepted": "accepted, largest GT-vs-prediction max corner error",
            "rejected_correct": "rejected, largest max corner error",
            "rejected_costly": "rejected, smallest max corner error",
            "why_max_not_median": "the failure mode is a 90-degree axis permutation, "
                                  "which leaves the median small and blows up one corner",
        },
        "examples": index,
    }, indent=2, ensure_ascii=False) + "\n")

    print(f"detected {len(frames)}  accepted {len(accepted)}  rejected {len(rejected)}")
    for key, rows in categories.items():
        span = f"{rows[0]['corner_max_px']:.1f}..{rows[-1]['corner_max_px']:.1f}"
        print(f"  {key:18} {len(rows)} 장   max corner error {span} px")
    print(f"-> {len(index)} figures in {figures.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
