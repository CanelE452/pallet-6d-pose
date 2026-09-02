"""§18-C 짝지은 불확실성 · §18-D 꼬리 분석.

## 왜 paired 인가

R0 와 Proposed 는 **같은 프레임**을 본다.  그래서 두 집단을 따로 요약해 빼는 것보다,
프레임마다 짝지어 차이를 보는 쪽이 분산이 훨씬 작고 정직하다.  프레임 난이도가
공통 요인으로 빠지기 때문이다.

## 왜 session-cluster 도 보는가

같은 촬영 세션의 프레임은 독립이 아니다.  프레임 단위 bootstrap 은 유효 표본 수를
부풀린다.  그래서 세션을 통째로 재표집한 구간도 함께 낸다 — 넓은 쪽이 정직한 구간이다.

## 꼬리 분석

평균이 좋아져도 최악 구간이 나빠지면 배포에서는 그게 문제다.  조건별로 상위 오차
프레임을 자동 목록화해 둔다.  **모델 overlay 를 만들지 않는다** — 이 산출물은
annotation review UI 와 섞이면 안 된다 (review 는 prediction-blinded 다).
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ARMS = REPO_ROOT / "data/pallet/results/paper_eval_v1/arms"
WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
OUT_JSON = REPO_ROOT / "data/pallet/results/paper_eval_v1/PAIRED_UNCERTAINTY.json"
TAIL_CSV = REPO_ROOT / "data/pallet/results/paper_eval_v1/TAIL_HIGH_ERROR_FRAMES.csv"
REPORT = REPO_ROOT / "_docs/paper/generated/APPENDIX_UNCERTAINTY.md"

BOOTSTRAP = 10000
RNG = np.random.default_rng(20260902)
GROSS_PX = 20.0
TAIL_FRACTION = 0.10

PAIRS = [
    ("Synthetic-only", "R0", "Proposed", "R5_PROPOSED"),
    ("Confidence", "R2_CONF", "Proposed", "R5_PROPOSED"),
    ("Reprojection", "R3_CONF_REPROJ", "Proposed", "R5_PROPOSED"),
]
CONDITIONS = ("Daytime", "Nighttime", "Occlusion", "Truncation")


def per_frame(model: str) -> dict[str, dict]:
    path = ARMS / f"{model}_per_frame.csv"
    out: dict[str, dict] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row["kind"] != "POSITIVE":
            continue
        strict = row.get("top_keypoint_supervised_errors_px") or ""
        annotated = row.get("top_keypoint_all_annotated_errors_px") or ""
        out[row["image"]] = {
            "session": row.get("session_id") or row.get("source_set") or "",
            "matched": row["top_iou50_match"] == "True",
            "strict": [float(v) for v in strict.split(";")] if strict else [],
            "annotated": [float(v) for v in annotated.split(";")] if annotated else [],
        }
    return out


def workspace_conditions() -> dict[str, dict]:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from evaluation.eval_workspace import (evaluation_population_views, load_frames)
    rows = evaluation_population_views(load_frames(WORKSPACE))["PAPER_EVAL_POSITIVE"]
    return {f"data/evaluation/pallet_eval_v1/{row['image_path']}": row for row in rows}


def pooled_median_bootstrap(left: dict, right: dict, keys: list[str],
                            clusters: np.ndarray | None = None) -> dict:
    """사전등록 헤드라인 지표(전 프레임 keypoint 를 풀링한 median)의 차이를 재표집한다.

    프레임별 median 을 다시 평균/중앙값 내는 것과는 **다른 통계량**이다.
    evaluator 가 쓰는 정의가 pooled median 이므로, 불확실성도 그 정의 위에서 재야
    한다.  프레임을 재표집할 때마다 양쪽 모델의 pooled median 을 다시 계산한다.
    """

    def errors_of(store: dict, key: str) -> list[float]:
        # **strict 만 쓴다.**  strict 가 빈 프레임에서 all-annotated 로 대체하면
        # 스케일이 다른 두 모집단(4 px 대 10~12 px)을 한 median 에 섞게 되고,
        # 그러면 evaluator 의 헤드라인 정의와 다른 값이 나온다.  실제로 섞었더니
        # 헤드라인(4.420 -> 4.180 개선)과 부호가 반대인 결과가 나왔다.
        return store[key]["strict"]

    left_lists = [errors_of(left, key) for key in keys]
    right_lists = [errors_of(right, key) for key in keys]
    usable = [i for i in range(len(keys)) if left_lists[i] and right_lists[i]]
    if not usable:
        return {"n": 0, "observed": None, "ci_low": None, "ci_high": None,
                "p_better": None}

    observed = (
        float(np.median(np.concatenate([right_lists[i] for i in usable])))
        - float(np.median(np.concatenate([left_lists[i] for i in usable])))
    )
    if clusters is not None:
        names = np.unique(clusters[usable])
        buckets = {name: [i for i in usable if clusters[i] == name] for name in names}

    draws = np.empty(BOOTSTRAP)
    for step in range(BOOTSTRAP):
        if clusters is None:
            picked = RNG.choice(usable, size=len(usable), replace=True)
        else:
            chosen = RNG.choice(list(buckets), size=len(buckets), replace=True)
            picked = [i for name in chosen for i in buckets[name]]
        draws[step] = (
            float(np.median(np.concatenate([right_lists[i] for i in picked])))
            - float(np.median(np.concatenate([left_lists[i] for i in picked])))
        )
    return {
        "n": len(usable),
        "observed": observed,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_better": float((draws < 0).mean()),
    }


def paired_bootstrap(deltas: np.ndarray, clusters: np.ndarray | None = None) -> dict:
    """짝지은 차이의 평균에 대한 백분위 bootstrap 구간."""

    if deltas.size == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None,
                "p_better": None}
    if clusters is None:
        draws = RNG.integers(0, deltas.size, size=(BOOTSTRAP, deltas.size))
        means = deltas[draws].mean(axis=1)
    else:
        names = np.unique(clusters)
        buckets = [deltas[clusters == name] for name in names]
        means = np.empty(BOOTSTRAP)
        for index in range(BOOTSTRAP):
            picked = RNG.integers(0, len(buckets), size=len(buckets))
            means[index] = np.concatenate([buckets[p] for p in picked]).mean()
    return {
        "n": int(deltas.size),
        "mean": float(deltas.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        # delta = proposed - baseline.  corner 는 낮을수록 좋으므로 음수가 개선.
        "p_better": float((means < 0).mean()),
    }


def main() -> int:
    conditions = workspace_conditions()
    cache = {name: per_frame(name) for _, name, _, name2 in
             [(a, b, c, d) for a, b, c, d in PAIRS]
             for name in (name, name2)} if False else {}
    for _, base, _, other in PAIRS:
        for name in (base, other):
            if name not in cache:
                cache[name] = per_frame(name)

    report: dict[str, dict] = {}
    print(f"{'comparison':34} {'metric':10} {'n':>5} {'mean Δ':>9} "
          f"{'95% CI (frame)':>22} {'95% CI (session)':>24} {'P(better)':>10}")
    print("─" * 122)
    for base_label, base, other_label, other in PAIRS:
        left, right = cache[base], cache[other]
        shared = [key for key in left if key in right]

        # corner: 프레임별 median.  strict 가 비면 all-annotated 진단을 쓴다.
        deltas, clusters, det_deltas, det_clusters = [], [], [], []
        for key in shared:
            a, b = left[key], right[key]
            for field in ("strict", "annotated"):
                if a[field] and b[field]:
                    deltas.append(float(np.median(b[field])) - float(np.median(a[field])))
                    clusters.append(a["session"])
                    break
            det_deltas.append(float(b["matched"]) - float(a["matched"]))
            det_clusters.append(a["session"])

        entry = {}
        # 사전등록 헤드라인(pooled keypoint median)의 불확실성 — 이게 primary 다.
        cluster_array = np.asarray([left[key]["session"] for key in shared])
        entry["pooled_corner_median"] = {
            "frame": pooled_median_bootstrap(left, right, shared),
            "session_cluster": pooled_median_bootstrap(left, right, shared, cluster_array),
        }
        for metric, values, group in (
            ("corner", np.asarray(deltas), np.asarray(clusters)),
            ("detection", np.asarray(det_deltas), np.asarray(det_clusters)),
        ):
            frame_ci = paired_bootstrap(values)
            session_ci = paired_bootstrap(values, group)
            entry[metric] = {"frame": frame_ci, "session_cluster": session_ci}
            better = frame_ci["p_better"]
            if metric == "detection" and better is not None:
                better = 1.0 - better  # detection 은 높을수록 좋다
            print(f"{base_label + ' -> ' + other_label:34} {metric:10} "
                  f"{frame_ci['n']:>5} {frame_ci['mean']:>9.4f} "
                  f"[{frame_ci['ci_low']:>8.4f},{frame_ci['ci_high']:>8.4f}] "
                  f"[{session_ci['ci_low']:>10.4f},{session_ci['ci_high']:>10.4f}] "
                  f"{better:>10.3f}")
        pooled = entry["pooled_corner_median"]
        print(f"{base_label + ' -> ' + other_label:34} {'pooled~':10} "
              f"{pooled['frame']['n']:>5} {pooled['frame']['observed']:>9.4f} "
              f"[{pooled['frame']['ci_low']:>8.4f},{pooled['frame']['ci_high']:>8.4f}] "
              f"[{pooled['session_cluster']['ci_low']:>10.4f},"
              f"{pooled['session_cluster']['ci_high']:>10.4f}] "
              f"{pooled['frame']['p_better']:>10.3f}")
        report[f"{base}__vs__{other}"] = entry

    # ── §18-D 꼬리 ──────────────────────────────────────────────────────
    proposed = cache["R5_PROPOSED"]
    baseline = cache["R0"]
    rows: list[dict] = []
    for key, value in proposed.items():
        meta = conditions.get(key)
        if meta is None:
            continue
        source = "strict" if value["strict"] else "annotated"
        errors = value[source]
        if not errors:
            continue
        base_errors = baseline.get(key, {}).get(source) or []
        rows.append({
            "image": key,
            "session_id": value["session"],
            "object_type": meta["object_type"],
            "paper_domain": meta.get("paper_domain", ""),
            "occlusion": meta.get("occlusion", ""),
            "truncation": meta.get("truncation", ""),
            "distance_bin": meta.get("distance_bin", ""),
            "error_source": source,
            "proposed_corner_median_px": round(float(np.median(errors)), 3),
            "proposed_corner_max_px": round(float(np.max(errors)), 3),
            "baseline_corner_median_px": (
                round(float(np.median(base_errors)), 3) if base_errors else ""),
            "gross_keypoints": int(np.count_nonzero(np.asarray(errors) > GROSS_PX)),
        })
    rows.sort(key=lambda row: -row["proposed_corner_median_px"])
    cut = max(1, int(len(rows) * TAIL_FRACTION))
    tail = rows[:cut]
    for row in rows:
        row["in_worst_decile"] = "true" if row in tail else "false"
    with TAIL_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n꼬리(상위 {TAIL_FRACTION:.0%}, n={cut}) 조건 분포")
    tail_lines = []
    for field in ("paper_domain", "occlusion", "truncation", "object_type"):
        overall = collections.Counter(row[field] or "-" for row in rows)
        worst = collections.Counter(row[field] or "-" for row in tail)
        for key in sorted(overall):
            share = worst[key] / cut if cut else 0.0
            base_share = overall[key] / len(rows)
            line = (f"  {field:14} {key:12} 꼬리 {worst[key]:>3}/{cut} = {share:.2%}"
                    f"   전체비중 {base_share:.2%}"
                    f"   {'과대' if share > base_share * 1.3 else ''}")
            print(line)
            tail_lines.append(line.strip())

    OUT_JSON.write_text(json.dumps({
        "schema_version": "paper_paired_uncertainty_v1",
        "bootstrap_draws": BOOTSTRAP,
        "note": ("delta = proposed - baseline.  corner 는 낮을수록 좋으므로 음수가 개선. "
                 "session_cluster 는 같은 세션 프레임이 독립이 아님을 반영한 넓은 구간."),
        "comparisons": report,
        "tail": {"fraction": TAIL_FRACTION, "n": cut,
                 "csv": str(TAIL_CSV.relative_to(REPO_ROOT))},
    }, indent=2, ensure_ascii=False) + "\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Appendix — paired uncertainty and tail analysis", "",
             "## 짝지은 불확실성", "",
             "R0 와 Proposed 는 같은 프레임을 본다.  그래서 프레임마다 짝지어 차이를 본다 —",
             "프레임 난이도가 공통 요인으로 빠져 분산이 작고 정직하다.", "",
             "같은 세션의 프레임은 독립이 아니므로 세션을 통째로 재표집한 구간도 함께 낸다.",
             "**넓은 쪽(session)이 정직한 구간이다.**", "",
             "```text",
             f"{'comparison':34} {'metric':20} {'n':>5} {'Δ':>9} "
             f"{'95% CI frame':>22} {'95% CI session':>24}",
             "─" * 118]
    for base_label, base, other_label, other in PAIRS:
        entry = report[f"{base}__vs__{other}"]
        for metric in ("pooled_corner_median", "corner", "detection"):
            f_ci = entry[metric]["frame"]
            s_ci = entry[metric]["session_cluster"]
            centre = f_ci.get("observed", f_ci.get("mean"))
            lines.append(
                f"{base_label + ' -> ' + other_label:34} {metric:20} {f_ci['n']:>5} "
                f"{centre:>9.4f} "
                f"[{f_ci['ci_low']:>8.4f},{f_ci['ci_high']:>8.4f}] "
                f"[{s_ci['ci_low']:>10.4f},{s_ci['ci_high']:>10.4f}]")
    lines += ["```", "",
              "`pooled_corner_median` 이 primary 다 — evaluator 의 헤드라인 정의(감독",
              "keypoint 를 전 프레임 풀링한 median)와 같은 통계량이고, 프레임을 재표집하며",
              "그 값을 다시 계산한다.  **strict 만 쓴다** — all-annotated 로 대체하면",
              "스케일이 다른 두 모집단을 한 median 에 섞게 된다.",
              "",
              "`corner` 는 프레임별 median 차이의 **평균**이라 다른 통계량이고 소수의 파국",
              "프레임에 끌린다 — 참고용으로만 둔다.", "",
              "`delta = proposed - baseline`.  corner 는 음수가 개선, detection 은 양수가 개선.",
              "구간이 0 을 포함하면 그 비교에서는 개선을 주장하지 않는다.", "",
              "## 꼬리 분석", "",
              f"Proposed 의 프레임별 corner median 상위 {TAIL_FRACTION:.0%}(n={cut})가",
              "어떤 조건에 몰려 있는지 본다.  평균이 좋아져도 최악 구간이 나빠지면",
              "배포에서는 그게 문제다.", "", "```text", *tail_lines, "```", "",
              f"프레임 목록: `{TAIL_CSV.relative_to(REPO_ROOT)}`", "",
              "이 산출물은 **annotation review UI 와 섞지 않는다** — review 는",
              "prediction-blinded 이고 여기에는 모델 오차가 들어 있다."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {TAIL_CSV.relative_to(REPO_ROOT)}")
    print(f"wrote {REPORT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
