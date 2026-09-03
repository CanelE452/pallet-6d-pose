"""V1B paired bootstrap — frame-level 과 session-cluster 를 함께 낸다.

    python3 scripts/paper/fast6d_screen_v1b/bootstrap_v1b.py \
        --output-dir data/pallet/results/paper_fast6d_screen_v1b

정의는 `paper_pose_metric_closure_v1/paired_bootstrap_pose.py` 와 동일하다 —
median 은 median, ADDsym 은 [0, 0.1 x diameter] 위 정확도 곡선의 면적.
새 통계를 만들지 않는다.

세션이 13 개뿐이라 cluster bootstrap 은 저표본이다.  구간이 0 을 포함하면
"차이가 없다" 가 아니라 "이 데이터로는 못 가른다" 는 뜻이다.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

METRICS = [("iou3d", "IoU3D median", "median", "higher"),
           ("add_sym_m", "ADDsym AUC", "auc", "higher"),
           ("rotation_deg", "rotation median [deg]", "median", "lower"),
           ("translation_cm", "translation median [cm]", "median", "lower")]


def statistic(values, diameters, kind):
    if kind == "median":
        return float(np.median(values))
    diameter = float(np.median(diameters))
    thresholds = np.linspace(0.0, 0.1 * diameter, 1001)
    accuracy = (values[None, :] <= thresholds[:, None]).mean(axis=1)
    return float(np.trapz(accuracy, thresholds) / (0.1 * diameter))


def contrast(rows, arm, reference, rng, n_resamples):
    usable = [r for r in rows if arm in r and reference in r]
    sessions = np.array([r["session_id"] for r in usable])
    unique = sorted(set(sessions))
    index_of = {s: np.where(sessions == s)[0] for s in unique}
    entry = {"arm": arm, "reference": reference, "paired_frames": len(usable),
             "sessions": len(unique), "metrics": {}}
    n = len(usable)
    for key, title, kind, better in METRICS:
        va = np.array([r[arm][key] for r in usable], float)
        vb = np.array([r[reference][key] for r in usable], float)
        da = np.array([r[arm]["diameter_m"] for r in usable], float)
        observed = statistic(va, da, kind) - statistic(vb, da, kind)
        frame_draws = np.empty(n_resamples)
        cluster_draws = np.empty(n_resamples)
        for i in range(n_resamples):
            pick = rng.integers(0, n, n)
            frame_draws[i] = (statistic(va[pick], da[pick], kind)
                              - statistic(vb[pick], da[pick], kind))
            drawn = rng.integers(0, len(unique), len(unique))
            pick = np.concatenate([index_of[unique[j]] for j in drawn])
            cluster_draws[i] = (statistic(va[pick], da[pick], kind)
                                - statistic(vb[pick], da[pick], kind))
        frame_ci = [float(np.percentile(frame_draws, 2.5)),
                    float(np.percentile(frame_draws, 97.5))]
        cluster_ci = [float(np.percentile(cluster_draws, 2.5)),
                      float(np.percentile(cluster_draws, 97.5))]
        entry["metrics"][key] = {
            "title": title, "better": better,
            "reference": statistic(vb, da, kind), "arm": statistic(va, da, kind),
            "delta": observed,
            "frame_CI95": frame_ci, "frame_excludes_zero": bool(frame_ci[1] < 0 or frame_ci[0] > 0),
            "session_CI95": cluster_ci,
            "session_excludes_zero": bool(cluster_ci[1] < 0 or cluster_ci[0] > 0),
        }
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    screen = out_dir / "screen"
    lock = json.loads((out_dir / "FAST_6D_SCREEN_V1B_LOCK.json").read_text())
    n_resamples = int(lock["uncertainty"]["paired_frame_bootstrap"])
    seed = int(lock["uncertainty"]["seed"])

    result = {"schema_version": "fast6d_v1b_bootstrap_v1",
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "n_resamples": n_resamples, "seed": seed,
              "grouping": lock["uncertainty"]["grouping"],
              "contrasts": {}}

    rng = np.random.default_rng(seed)
    bbox_rows = json.loads((screen / "C_PER_FRAME.json").read_text())["frames"]
    result["contrasts"]["C1-C0"] = contrast(bbox_rows, "C1", "C0", rng, n_resamples)

    for s in (1, 2):
        rows = json.loads((screen / f"LINE_PER_FRAME_seed{s}.json").read_text())["frames"]
        for arm in ("L2", "L3", "L4"):
            rng = np.random.default_rng(seed)          # 대조마다 같은 재표본
            result["contrasts"][f"{arm}-L0 seed{s}"] = contrast(rows, arm, "L0", rng, n_resamples)

    (screen / "PAIRED_BOOTSTRAP.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"{'contrast':18}{'metric':10}{'delta':>11}{'frame CI95':>26}{'session CI95':>26}")
    print("-" * 91)
    for name, entry in result["contrasts"].items():
        for key in ("iou3d", "add_sym_m"):
            m = entry["metrics"][key]
            fc, sc = m["frame_CI95"], m["session_CI95"]
            star = " *" if m["session_excludes_zero"] else "  "
            print(f"{name:18}{key:10}{m['delta']:11.4f}"
                  f"{'[' + f'{fc[0]:+.4f}, {fc[1]:+.4f}' + ']':>26}"
                  f"{'[' + f'{sc[0]:+.4f}, {sc[1]:+.4f}' + ']' + star:>26}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
