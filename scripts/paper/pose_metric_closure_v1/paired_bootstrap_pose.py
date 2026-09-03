"""arm 쌍의 6D pose 차이에 신뢰구간을 붙인다.  새 학습·새 추론 없음.

    python3 scripts/paper/pose_metric_closure_v1/paired_bootstrap_pose.py

출력  POSE_PAIRED_BOOTSTRAP.json
      _docs/paper/pose_metric_closure_v1/POSE_PAIRED_BOOTSTRAP.md

두 가지 재표본을 함께 낸다.

    frame-level      프레임을 복원추출.  프레임이 독립이라는 가정.
    session-cluster  세션 전체를 복원추출.  같은 세션 프레임이 서로 닮았다는
                     사실을 반영하므로 이쪽이 더 정직하고 구간이 넓다.

**paired** 다 — 같은 프레임에서 두 arm 을 비교하고 그 차이를 재표본한다.
arm 마다 따로 재표본해 빼면 짝지음을 버려 구간이 과대해진다.

세션이 13 개뿐이라 cluster bootstrap 자체가 저표본이다.  구간이 0 을 포함해도
"차이가 없다" 는 뜻이 아니라 "이 데이터로는 못 가른다" 는 뜻이다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
DOC_DIR = REPO_ROOT / "_docs/paper/pose_metric_closure_v1"
PER_FRAME = OUT_DIR / "POSE_PER_FRAME_BY_ARM.json"

N_RESAMPLES = 10000
SEED = 20260903
CI = 95.0

PAIRS = [("R5_PROPOSED", "R0"), ("R1_NAIVE", "R0"), ("R2_CONF", "R0"),
         ("R3_CONF_REPROJ", "R0"), ("R4_CONF_REMOVE", "R0"), ("R0_CONT", "R0")]
# (키, 표시명, 통계, 좋은 방향)
METRICS = [("iou3d", "IoU3D median", "median", "higher"),
           ("add_sym_m", "ADDsym AUC", "auc", "higher"),
           ("yaw_error_deg", "yaw median [deg]", "median", "lower"),
           ("translation_error_cm", "translation median [cm]", "median", "lower")]


def statistic(values, diameters, kind):
    if kind == "median":
        return float(np.median(values))
    # ADD AUC: [0, 0.1 x diameter] 위 정확도 곡선의 면적, 본 평가와 같은 정의
    diameter = float(np.median(diameters))
    thresholds = np.linspace(0.0, 0.1 * diameter, 1001)
    accuracy = (values[None, :] <= thresholds[:, None]).mean(axis=1)
    return float(np.trapz(accuracy, thresholds) / (0.1 * diameter))


def main() -> int:
    if not PER_FRAME.exists():
        print(f"{PER_FRAME.name} 이 없다 — evaluate_pose_by_session.py 를 먼저 돌려라")
        return 1
    payload = json.loads(PER_FRAME.read_text())
    per_arm = payload["per_frame"]

    rng = np.random.default_rng(SEED)
    results = []
    for arm_a, arm_b in PAIRS:
        rows_a = {r["frame_id"]: r for r in per_arm[arm_a]}
        rows_b = {r["frame_id"]: r for r in per_arm[arm_b]}
        shared = sorted(set(rows_a) & set(rows_b))
        sessions = np.array([rows_a[f]["session_id"] for f in shared])
        unique_sessions = sorted(set(sessions))
        session_index = {s: np.where(sessions == s)[0] for s in unique_sessions}

        entry = {"arm": arm_a, "reference": arm_b, "paired_frames": len(shared),
                 "frames_dropped_for_pairing": {
                     arm_a: len(rows_a) - len(shared),
                     arm_b: len(rows_b) - len(shared)},
                 "sessions": len(unique_sessions), "metrics": {}}

        for key, title, kind, better in METRICS:
            va = np.array([rows_a[f][key] for f in shared], float)
            vb = np.array([rows_b[f][key] for f in shared], float)
            da = np.array([rows_a[f]["diameter_m"] for f in shared], float)
            db = np.array([rows_b[f]["diameter_m"] for f in shared], float)
            observed = statistic(va, da, kind) - statistic(vb, db, kind)

            frame_draws = np.empty(N_RESAMPLES)
            cluster_draws = np.empty(N_RESAMPLES)
            n = len(shared)
            for i in range(N_RESAMPLES):
                pick = rng.integers(0, n, n)
                frame_draws[i] = (statistic(va[pick], da[pick], kind)
                                  - statistic(vb[pick], db[pick], kind))
                chosen = rng.integers(0, len(unique_sessions), len(unique_sessions))
                idx = np.concatenate([session_index[unique_sessions[c]] for c in chosen])
                cluster_draws[i] = (statistic(va[idx], da[idx], kind)
                                    - statistic(vb[idx], db[idx], kind))

            def interval(draws):
                low, high = np.percentile(draws, [(100 - CI) / 2, 100 - (100 - CI) / 2])
                return {"low": float(low), "high": float(high),
                        "excludes_zero": bool(low > 0 or high < 0)}

            entry["metrics"][key] = {
                "title": title, "better": better,
                "value_arm": statistic(va, da, kind),
                "value_reference": statistic(vb, db, kind),
                "observed_difference": observed,
                "frame_level": interval(frame_draws),
                "session_cluster": interval(cluster_draws),
            }
        results.append(entry)
        print(f"  {arm_a} vs {arm_b}  paired {len(shared)}  sessions {len(unique_sessions)}")

    report = {
        "schema_version": "pose_paired_bootstrap_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "resamples": N_RESAMPLES, "seed": SEED, "confidence": CI,
        "paired": True,
        "new_training": 0, "new_inference": 0,
        "note": ("session_cluster is the honest interval; frame_level assumes frames "
                 "are independent, which frames from one recording are not"),
        "low_cluster_count_caveat": ("13 sessions is a small number of clusters, so a "
                                     "cluster interval that contains zero means this "
                                     "data cannot separate the arms, not that they are "
                                     "equal"),
        "comparisons": results,
    }
    (OUT_DIR / "POSE_PAIRED_BOOTSTRAP.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Paired bootstrap — 6D pose, each arm against R0", "",
             "Every interval is **paired**: the two arms are compared on the same frame",
             "and it is the per-frame difference that gets resampled.", "",
             "Two resampling schemes are reported. `frame` treats frames as independent.",
             "`cluster` resamples whole sessions, which respects the fact that frames from",
             "one recording resemble each other — that is the interval to quote.", "",
             f"{N_RESAMPLES} resamples, seed {SEED}, {CI:.0f}% interval. No model ran again.",
             "", "**13 sessions is a small number of clusters.** An interval containing zero",
             "means this data cannot separate the arms — not that they perform equally.", ""]
    for key, title, _, better in METRICS:
        lines += [f"## {title}  ({better} is better)", "", "```text",
                  f"{'arm vs R0':20}{'diff':>9}  {'frame 95% CI':>20}"
                  f"  {'cluster 95% CI':>20}   cluster",
                  "─" * 87]
        for entry in results:
            block = entry["metrics"][key]
            frame, cluster = block["frame_level"], block["session_cluster"]
            frame_ci = "[{:+.3f}, {:+.3f}]".format(frame["low"], frame["high"])
            cluster_ci = "[{:+.3f}, {:+.3f}]".format(cluster["low"], cluster["high"])
            verdict = "excludes 0" if cluster["excludes_zero"] else "contains 0"
            lines.append(
                f"{entry['arm']:20}{block['observed_difference']:+9.3f}"
                f"  {frame_ci:>20}  {cluster_ci:>20}   {verdict}")
        lines += ["```", ""]
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "POSE_PAIRED_BOOTSTRAP.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {(OUT_DIR / 'POSE_PAIRED_BOOTSTRAP.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
