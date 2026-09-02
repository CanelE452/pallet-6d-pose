"""teacher 여럿의 예측을 좌표별 median 으로 합치고, 그 품질을 GT 로 채점한다.

§13~§14 의 규칙을 그대로 구현한다.

    valid teacher 예측이 3 개 미만        -> TRUE IGNORE
    consensus = coordinate-wise median
    d_i = median_t || p_i^t - p_i* || / D   (D = consensus projected cuboid diagonal)
    d_i > 0.05                            -> TRUE IGNORE
    q >= 0.75 (near-square)               -> semantic corner 0..7 TRUE IGNORE

0.05 는 새 임계가 아니라 기존 geometry normalised tolerance 재사용이다.
새 학습 가중치가 없다 — median 과 고정 tolerance 뿐이다.

pseudo 6D pose · 승리 hypothesis · GT 축은 **저장하지 않는다** (§16).

`--teachers` 로 몇 개 view 를 합칠지 고른다.  T1 이 아직 없으면 T0 의 두 view 만으로도
메커니즘의 방향을 볼 수 있다 (그 경우 valid 최소 3 개를 만족할 수 없으므로
`--min-teachers` 를 함께 낮춰 **탐색용**으로만 쓴다 — 그 사실을 출력에 박는다).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
for relative in ("scripts/self_training_yolo/v2", "scripts/self_training_yolo",
                 "scripts/evaluation"):
    sys.path.insert(0, str(REPO_ROOT / relative))

from eval_workspace import load_frames, evaluation_population_views  # noqa: E402
from keypoint_scores import ambiguity_q  # noqa: E402
from pseudo_label_filters import projected_diagonal  # noqa: E402

WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
OUT_DIR = REPO_ROOT / "data/pallet/results/paper_strong_teacher_v1"
T0_CACHE = REPO_ROOT / "data/pallet/results/paper_selftrain_v4/V4_PROXY_TEACHER_CACHE.json"

GROSS_PX, CATASTROPHIC_PX = 20.0, 40.0
DISAGREEMENT_TOLERANCE = 0.05
AMBIGUITY_Q = 0.75
KP_CONF_FLOOR = 0.5
BOX_CONF = 0.85
N_CORNERS = 8


def canonical(frame_id: str) -> str:
    return frame_id.replace("__", ":")


def load_views(names: list[str], cache_paths: dict[str, Path]) -> dict[str, dict]:
    """view 이름 -> {frame: {keypoints, conf, box_conf}}"""

    views: dict[str, dict] = {}
    for path_name, path in cache_paths.items():
        cache = json.loads(path.read_text())
        for suffix, key in (("orig", "top1"), ("flip", "flip_top1")):
            name = f"{path_name}_{suffix}"
            if name not in names:
                continue
            views[name] = {}
            for frame, entry in cache.items():
                blob = entry.get(key)
                if not blob:
                    continue
                views[name][frame] = {
                    "keypoints": np.asarray(blob["keypoints_xy"], dtype=float),
                    "conf": np.nan_to_num(
                        np.asarray(blob["keypoints_conf"], dtype=float), nan=0.0),
                    "box_conf": float(blob["box_conf"]),
                }
    missing = [n for n in names if n not in views]
    if missing:
        raise SystemExit(f"VIEW_NOT_AVAILABLE: {missing}")
    return views


def consensus(views: dict[str, dict], frame: str, min_teachers: int) -> dict | None:
    """좌표별 median 과 teacher 간 불일치.  GT 를 쓰지 않는다."""

    stacks, confidences = [], []
    for view in views.values():
        blob = view.get(frame)
        if blob is None or blob["box_conf"] < BOX_CONF:
            continue
        stacks.append(blob["keypoints"])
        confidences.append(blob["conf"])
    if len(stacks) < min_teachers:
        return None
    points = np.stack(stacks)               # (T, 9, 2)
    confidence = np.stack(confidences)      # (T, 9)
    valid = confidence >= KP_CONF_FLOOR

    merged = np.full((9, 2), np.nan)
    disagreement = np.full(9, np.inf)
    counts = valid.sum(axis=0)
    for index in range(9):
        usable = points[valid[:, index], index, :]
        if len(usable) < min_teachers:
            continue
        merged[index] = np.median(usable, axis=0)

    diagonal = projected_diagonal(merged[:N_CORNERS]) if np.isfinite(
        merged[:N_CORNERS]).all() else float("nan")
    if np.isfinite(diagonal) and diagonal > 1e-6:
        for index in range(9):
            usable = points[valid[:, index], index, :]
            if len(usable) < min_teachers or not np.isfinite(merged[index]).all():
                continue
            disagreement[index] = float(np.median(
                np.linalg.norm(usable - merged[index], axis=1))) / diagonal

    q = ambiguity_q(merged)
    ambiguous = bool(np.isfinite(q) and q >= AMBIGUITY_Q)
    accepted = np.zeros(9, dtype=bool)
    for index in range(9):
        if not np.isfinite(merged[index]).all():
            continue
        if disagreement[index] > DISAGREEMENT_TOLERANCE:
            continue
        if ambiguous and index < N_CORNERS:
            continue
        accepted[index] = True

    return {
        "keypoints": merged,
        "disagreement": disagreement,
        "valid_teacher_count": counts.tolist(),
        "accepted": accepted,
        "q": float(q) if np.isfinite(q) else None,
        "ambiguous_view": ambiguous,
        "n_teachers": len(stacks),
        "projected_diagonal_px": float(diagonal) if np.isfinite(diagonal) else None,
    }


def evaluate(points_by_frame: dict, accepted_by_frame: dict | None,
             context: dict) -> dict:
    """GT 대비 품질.  accepted 가 주어지면 그 keypoint 만 센다."""

    errors, normalised, per_frame = [], [], []
    for frame, item in context.items():
        points = points_by_frame.get(frame)
        if points is None:
            continue
        mask = item["supervised"][:N_CORNERS].copy()
        if accepted_by_frame is not None:
            accept = accepted_by_frame.get(frame)
            if accept is None:
                continue
            mask &= accept[:N_CORNERS]
        if not mask.any():
            continue
        distance = np.linalg.norm(points[:N_CORNERS] - item["gt"][:N_CORNERS], axis=1)
        selected = distance[mask]
        selected = selected[np.isfinite(selected)]
        if not selected.size:
            continue
        errors += selected.tolist()
        normalised += (selected / item["diagonal"]).tolist()
        per_frame.append(float(np.median(selected)))

    if not errors:
        return {"n_keypoints": 0, "n_frames": 0}
    array, norm = np.asarray(errors), np.asarray(normalised)
    return {
        "n_keypoints": int(array.size),
        "n_frames": len(per_frame),
        "nme_median": float(np.median(norm)),
        "nme_p90": float(np.percentile(norm, 90)),
        "px_median": float(np.median(array)),
        "px_p90": float(np.percentile(array, 90)),
        "gross20_rate": float(np.mean(array > GROSS_PX)),
        "catastrophic40_rate": float(np.mean(array > CATASTROPHIC_PX)),
    }


def build_context() -> dict:
    context = {}
    for row in evaluation_population_views(load_frames(WORKSPACE))["PAPER_EVAL_POSITIVE"]:
        frame = canonical(row["frame_id"])
        payload = json.loads((WORKSPACE / row["annotation_path"]).read_text())
        points = payload["objects"][0]["keypoint_annotations"]
        gt = np.array([p["xy"] if p.get("xy") else [np.nan, np.nan] for p in points],
                      dtype=float)
        supervised = np.array(
            [bool(p.get("visibility", 0)) and p.get("xy") is not None for p in points])
        if not np.isfinite(gt[:N_CORNERS]).all():
            continue
        context[frame] = {
            "gt": gt, "supervised": supervised,
            "diagonal": projected_diagonal(gt[:N_CORNERS]),
            "domain": row.get("paper_domain"),
        }
    return context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teachers", nargs="+", default=["T0_orig", "T0_flip"])
    parser.add_argument("--min-teachers", type=int, default=3)
    parser.add_argument("--medium-cache", default=None,
                        help="T1 예측 캐시 (있으면 T1_orig / T1_flip 을 쓸 수 있다)")
    parser.add_argument("--tag", default="T0_TWO_VIEW")
    args = parser.parse_args()

    caches = {"T0": T0_CACHE}
    if args.medium_cache:
        caches["T1"] = Path(args.medium_cache)
    views = load_views(args.teachers, caches)
    context = build_context()

    baseline_points, baseline_accept = {}, {}
    for frame, blob in views["T0_orig"].items():
        if blob["box_conf"] < BOX_CONF:
            continue
        baseline_points[frame] = blob["keypoints"]
        baseline_accept[frame] = blob["conf"] >= KP_CONF_FLOOR

    merged_points, merged_accept, stats = {}, {}, {
        "no_consensus": 0, "disagreement_ignored": 0, "ambiguity_ignored": 0,
        "accepted": 0}
    for frame in context:
        result = consensus(views, frame, args.min_teachers)
        if result is None:
            stats["no_consensus"] += 1
            continue
        merged_points[frame] = result["keypoints"]
        merged_accept[frame] = result["accepted"]
        for index in range(N_CORNERS):
            if result["accepted"][index]:
                stats["accepted"] += 1
            elif result["ambiguous_view"]:
                stats["ambiguity_ignored"] += 1
            elif result["disagreement"][index] > DISAGREEMENT_TOLERANCE:
                stats["disagreement_ignored"] += 1

    def by_domain(points, accept, domain=None):
        subset = {f: c for f, c in context.items()
                  if domain is None or c["domain"] == domain}
        return evaluate(points, accept, subset)

    report = {
        "schema_version": "teacher_consensus_v1",
        "tag": args.tag,
        "teachers": args.teachers,
        "min_teachers": args.min_teachers,
        "exploratory": args.min_teachers < 3,
        "exploratory_note": ("min_teachers < 3 은 §13 의 계약이 아니다.  T1 이 없을 때 "
                             "방향만 보려는 탐색 설정이며, 이 수치로 gate 를 판정하지 "
                             "않는다."),
        "counts": stats,
        "baseline_R0": {
            "ALL": by_domain(baseline_points, baseline_accept),
            "daytime": by_domain(baseline_points, baseline_accept, "daytime"),
            "nighttime": by_domain(baseline_points, baseline_accept, "nighttime"),
        },
        "consensus": {
            "ALL": by_domain(merged_points, merged_accept),
            "daytime": by_domain(merged_points, merged_accept, "daytime"),
            "nighttime": by_domain(merged_points, merged_accept, "nighttime"),
        },
        "consensus_on_same_keypoints_as_baseline": {
            "note": ("합의가 받아들인 keypoint 만 세면 모집단이 달라 비교가 기운다.  "
                     "여기서는 baseline 과 **같은 keypoint 집합** 에서 좌표만 바꿔 잰다."),
            "ALL": by_domain(merged_points, baseline_accept),
            "daytime": by_domain(merged_points, baseline_accept, "daytime"),
            "nighttime": by_domain(merged_points, baseline_accept, "nighttime"),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"TEACHER_CONSENSUS_{args.tag}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"teachers {args.teachers}  min_teachers {args.min_teachers}"
          + ("   [EXPLORATORY]" if report["exploratory"] else ""))
    print(f"counts {stats}")
    print(f"\n{'set':34} {'n_kp':>6} {'NME med':>9} {'NME p90':>9} "
          f"{'px med':>8} {'px p90':>8} {'gross20':>8} {'cat40':>7}")
    print("-" * 92)
    for label, block in (("R0 baseline", report["baseline_R0"]),
                         ("consensus (accepted only)", report["consensus"]),
                         ("consensus (same keypoints)",
                          report["consensus_on_same_keypoints_as_baseline"])):
        for domain in ("ALL", "daytime", "nighttime"):
            item = block[domain] if domain in block else {}
            if not item.get("n_keypoints"):
                continue
            print(f"{label + ' / ' + domain:34} {item['n_keypoints']:6d} "
                  f"{item['nme_median']:9.4f} {item['nme_p90']:9.4f} "
                  f"{item['px_median']:8.2f} {item['px_p90']:8.2f} "
                  f"{item['gross20_rate']:8.3f} {item['catastrophic40_rate']:7.3f}")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
