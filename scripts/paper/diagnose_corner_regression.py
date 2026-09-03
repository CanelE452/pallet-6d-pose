"""self-training 이 corner 를 악화시킨 원인을 네 갈래로 분해한다.  진단만 한다.

성능을 고치지 않는다.  threshold·tau·pseudo pool·model 을 건드리지 않는다.

raw corner 는 모델마다 **다른 프레임 집합** 위에서 계산된다 — 검출된 프레임에서만
오차가 모이기 때문이다.  그래서 R0 대 R5 의 raw 차이는 다음 셋이 섞여 있다.

    A  raw px scale         프레임마다 물체 크기·해상도가 달라 px 가 비교 불가
    B  detection selection  R5 가 새로 회수한 어려운 프레임이 평균을 끌어내림
    C  paired drift         같은 프레임·같은 keypoint 에서 실제로 나빠짐
    D  pseudo-label noise   통과한 PL 자체의 품질

여기서는 A(§6 NME) · B(§3 교차표, §5 회수분) · C(§4 paired) 를 분리해 낸다.
D 는 M4 와 contact sheet 가 맡는다.

출력:
    data/pallet/results/paper_eval_v1/REGRESSION_DIAGNOSIS.json
    _docs/archive/paper_pre_final_20260903/legacy_paper_outputs/generated/REGRESSION_DIAGNOSIS.md
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo"))

from eval_workspace import load_frames, evaluation_population_views  # noqa: E402
from pseudo_label_filters import projected_diagonal  # noqa: E402

WORKSPACE = REPO_ROOT / "data" / "evaluation" / "pallet_eval_v1"
ARMS = REPO_ROOT / "data" / "pallet" / "results" / "paper_eval_v1" / "arms"
OUT_JSON = ARMS.parent / "REGRESSION_DIAGNOSIS.json"
OUT_MD = REPO_ROOT / "_docs" / "paper" / "generated" / "REGRESSION_DIAGNOSIS.md"

BASE, PROPOSED = "R0", "R5_PROPOSED"
DOMAINS = ("daytime", "nighttime")
BOOTSTRAP = 10_000
SEED = 20260902


def canonical_frame_id(frame_id: str) -> str:
    """두 소스의 frame_id 표기를 한 형태로 맞춘다."""

    return frame_id.replace("__", ":")


def parse_errors(value: str) -> list[float]:
    if not value:
        return []
    return [float(part) for part in value.split(";") if part]


def load_per_frame(name: str) -> dict[str, dict]:
    path = ARMS / f"{name}_per_frame.csv"
    if not path.exists():
        raise SystemExit(f"PER_FRAME_MISSING: {path}")
    return {
        canonical_frame_id(row["frame_id"]): row
        for row in csv.DictReader(path.open(encoding="utf-8"))
        if row["kind"] == "POSITIVE"
    }


def detected(row: dict) -> bool:
    return row.get("top_iou50_match") == "True"


def gt_scales() -> dict[str, dict]:
    """프레임별 정규화 분모.  GT 에서만 온다 — 예측을 쓰지 않는다."""

    scales: dict[str, dict] = {}
    for row in evaluation_population_views(load_frames(WORKSPACE))["PAPER_EVAL_POSITIVE"]:
        payload = json.loads((WORKSPACE / row["annotation_path"]).read_text())
        obj = payload["objects"][0]
        cuboid = obj.get("projected_cuboid")
        points = obj["keypoint_annotations"]
        xy = np.array([p["xy"] for p in points[:8] if p.get("xy")], dtype=np.float64)
        cuboid_diag = None
        if isinstance(cuboid, list) and len(cuboid) >= 8:
            array = np.asarray(cuboid, dtype=np.float64)[:8]
            if np.isfinite(array).all():
                cuboid_diag = projected_diagonal(array)
        if cuboid_diag is None and len(xy) == 8:
            cuboid_diag = projected_diagonal(xy)
        bbox_diag = None
        if len(xy) >= 2:
            span = xy.max(axis=0) - xy.min(axis=0)
            bbox_diag = float(np.hypot(*span))
        # population view 는 `session__stamp`, evaluator per-frame CSV 는
        # `session:stamp` 를 쓴다.  둘을 그냥 맞대면 교집합이 0 이 된다.
        scales[canonical_frame_id(row["frame_id"])] = {
            "paper_domain": row.get("paper_domain"),
            "object_type": row.get("object_type"),
            "session_id": row.get("session_id"),
            "cuboid_diagonal_px": cuboid_diag,
            "gt_bbox_diagonal_px": bbox_diag,
        }
    return scales


def quantiles(values) -> dict:
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                       dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "median": None, "p90": None, "mean": None}
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "mean": float(array.mean()),
    }


def paired_bootstrap(deltas: np.ndarray) -> dict:
    """프레임 단위 재표집.  중앙값 차이의 CI."""

    if deltas.size == 0:
        return {"n": 0, "median_delta": None, "ci95": None, "p_two_sided": None}
    rng = np.random.default_rng(SEED)
    index = rng.integers(0, deltas.size, size=(BOOTSTRAP, deltas.size))
    samples = np.median(deltas[index], axis=1)
    observed = float(np.median(deltas))
    centred = samples - observed
    p = float(np.mean(np.abs(centred) >= abs(observed)))
    return {
        "n": int(deltas.size),
        "median_delta": observed,
        "ci95": [float(np.percentile(samples, 2.5)),
                 float(np.percentile(samples, 97.5))],
        "p_two_sided": p,
    }


def main() -> int:
    base_rows = load_per_frame(BASE)
    prop_rows = load_per_frame(PROPOSED)
    scales = gt_scales()
    overlap = len(set(scales) & set(base_rows) & set(prop_rows))
    if overlap != len(scales):
        raise SystemExit(
            f"FRAME_ID_JOIN_INCOMPLETE: {overlap} / {len(scales)} 만 맞물린다.  "
            "표기 정규화를 확인하라 — 조용히 0 을 보고하지 않는다.")

    report: dict = {
        "schema_version": "selftraining_regression_diagnosis_v1",
        "base": BASE,
        "proposed": PROPOSED,
        "corner_metric": ("original-image Euclidean pixel error; predictions are "
                          "un-padded by INFERENCE_PAD before comparison"),
        "domains": {},
        "scale": {},
    }

    for domain in DOMAINS:
        frames = [f for f, s in scales.items() if s["paper_domain"] == domain]
        frames = [f for f in frames if f in base_rows and f in prop_rows]

        # ── §3 검출 교차표 ────────────────────────────────────────────
        buckets: dict[str, list[str]] = {
            "BOTH_DETECTED": [], "R0_ONLY": [], "R5_ONLY": [], "BOTH_MISSED": []}
        for frame in frames:
            b, p = detected(base_rows[frame]), detected(prop_rows[frame])
            key = ("BOTH_DETECTED" if b and p else
                   "R0_ONLY" if b else "R5_ONLY" if p else "BOTH_MISSED")
            buckets[key].append(frame)

        # ── §4 BOTH_DETECTED 에서만 paired ──────────────────────────
        per_frame_delta: list[float] = []
        base_pool: list[float] = []
        prop_pool: list[float] = []
        length_mismatch = 0
        for frame in buckets["BOTH_DETECTED"]:
            a = parse_errors(base_rows[frame]["top_keypoint_supervised_errors_px"])
            b = parse_errors(prop_rows[frame]["top_keypoint_supervised_errors_px"])
            if len(a) != len(b) or not a:
                length_mismatch += 1
                continue
            base_pool.extend(a)
            prop_pool.extend(b)
            per_frame_delta.append(float(np.median(b)) - float(np.median(a)))

        # 악화가 고르게 퍼졌는지, 몇 장에 몰렸는지.  frame median CI 가 0 을 물면
        # "고른 열화" 와 "소수의 파국" 이 같은 수치를 만들 수 있다 — 갈라 둔다.
        delta_array = np.asarray(per_frame_delta)
        concentration = {}
        if delta_array.size:
            order = np.argsort(-delta_array)
            for drop in (0, 1, 3, 5):
                keep = order[drop:]
                concentration[f"pooled_after_dropping_worst_{drop}"] = None
            concentration = {
                "n_worse": int(np.count_nonzero(delta_array > 0)),
                "n_better": int(np.count_nonzero(delta_array < 0)),
                "n_equal": int(np.count_nonzero(delta_array == 0)),
                "delta_p10": float(np.percentile(delta_array, 10)),
                "delta_p50": float(np.percentile(delta_array, 50)),
                "delta_p90": float(np.percentile(delta_array, 90)),
                "delta_max": float(delta_array.max()),
                "sum_positive": float(delta_array[delta_array > 0].sum()),
                "sum_top5_positive": float(np.sort(delta_array)[-5:].sum()),
            }
            total_positive = concentration["sum_positive"]
            concentration["top5_share_of_positive"] = (
                None if total_positive <= 0
                else concentration["sum_top5_positive"] / total_positive)

        paired = {
            "n_frames": len(per_frame_delta),
            "delta_concentration": concentration,
            "supervision_mask_mismatch_frames": length_mismatch,
            "base_pooled": quantiles(base_pool),
            "proposed_pooled": quantiles(prop_pool),
            "frame_median_delta": paired_bootstrap(np.asarray(per_frame_delta)),
        }

        # ── §5 R5 가 새로 회수한 프레임 ────────────────────────────
        # 통계를 raw 표와 **같게** 둔다 — keypoint 를 통째로 모아 median 을 낸다.
        # 프레임 median 의 median 을 쓰면 M2 수치와 안 맞아 분해가 성립하지 않는다.
        def pooled(rows, frames) -> list[float]:
            out: list[float] = []
            for frame in frames:
                out.extend(parse_errors(
                    rows[frame]["top_keypoint_supervised_errors_px"]))
            return out

        recovered = pooled(prop_rows, buckets["R5_ONLY"])
        prop_both = pooled(prop_rows, buckets["BOTH_DETECTED"])
        base_both = pooled(base_rows, buckets["BOTH_DETECTED"])
        base_only = pooled(base_rows, buckets["R0_ONLY"])

        raw_base = quantiles(base_both + base_only)
        raw_prop = quantiles(prop_both + recovered)
        prop_without_recovered = quantiles(prop_both)
        base_on_both = quantiles(base_both)

        def diff(a, b):
            return None if a is None or b is None else a - b

        selection_effect = diff(raw_prop["median"], prop_without_recovered["median"])
        localisation_effect = diff(prop_without_recovered["median"],
                                   base_on_both["median"])
        base_population_effect = diff(base_on_both["median"], raw_base["median"])
        total = diff(raw_prop["median"], raw_base["median"])

        report["domains"][domain] = {
            "n_frames": len(frames),
            "detection_crosstab": {k: len(v) for k, v in buckets.items()},
            "frames": {k: sorted(v) for k, v in buckets.items()},
            "paired_both_detected": paired,
            "recovered_by_proposed": {
                "n_frames": len(buckets["R5_ONLY"]),
                **{f"pooled_{k}": v for k, v in quantiles(recovered).items()},
            },
            "pooled_keypoint_median": {
                "base_all_detected": raw_base,
                "base_on_both_detected": base_on_both,
                "proposed_all_detected": raw_prop,
                "proposed_excluding_recovered": prop_without_recovered,
            },
            "decomposition_px": {
                "total_raw_delta": total,
                "base_population_effect": base_population_effect,
                "localisation_drift_on_both_detected": localisation_effect,
                "selection_effect_from_recovered": selection_effect,
            },
        }

    # ── §6 scale-normalised keypoint error ──────────────────────────
    for denominator in ("cuboid_diagonal_px", "gt_bbox_diagonal_px"):
        block: dict = {}
        for domain in DOMAINS:
            frames = [f for f, s in scales.items() if s["paper_domain"] == domain]
            both = set(report["domains"][domain]["frames"]["BOTH_DETECTED"])
            for model, rows in ((BASE, base_rows), (PROPOSED, prop_rows)):
                for scope, keep in (("all_detected", frames),
                                    ("both_detected", [f for f in frames if f in both])):
                    values: list[float] = []
                    for frame in keep:
                        row = rows.get(frame)
                        scale = scales[frame][denominator]
                        if row is None or not scale or not np.isfinite(scale):
                            continue
                        values.extend(e / scale for e in parse_errors(
                            row["top_keypoint_supervised_errors_px"]))
                    block[f"{domain}/{model}/{scope}"] = quantiles(values)
        report["scale"][denominator] = block

    # 분모 자체의 분포 — px 가 비교 가능한 단위인지 본다
    report["scale"]["denominator_spread"] = {}
    for domain in DOMAINS:
        values = [scales[f]["cuboid_diagonal_px"] for f, s in scales.items()
                  if s["paper_domain"] == domain and s["cuboid_diagonal_px"]]
        report["scale"]["denominator_spread"][domain] = quantiles(values)

    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    render(report)
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


def number(value, spec=".3f") -> str:
    return "—" if value is None else format(value, spec)


def render(report: dict) -> None:
    lines = [
        "# Self-training corner regression — 원인 분해",
        "",
        "진단 전용이다.  이 문서를 근거로 threshold·tau·pseudo pool·model 을 바꾸지 않는다.",
        "",
        f"corner 정의: {report['corner_metric']}.",
        "",
        "## §3 검출 교차표 (R0 대 R5)",
        "",
        "```text",
        f"{'domain':11} {'N':>4} {'BOTH_DET':>9} {'R0_ONLY':>8} {'R5_ONLY':>8} {'BOTH_MISS':>10}",
        "─" * 56,
    ]
    for domain, block in report["domains"].items():
        table = block["detection_crosstab"]
        lines.append(f"{domain:11} {block['n_frames']:4d} {table['BOTH_DETECTED']:9d} "
                     f"{table['R0_ONLY']:8d} {table['R5_ONLY']:8d} {table['BOTH_MISSED']:10d}")
    lines += ["```", "",
              "## §4 BOTH_DETECTED 만 — 같은 프레임·같은 supervised keypoint",
              "",
              "raw 표와 **별개**다.  여기서는 모집단이 두 모델에 대해 동일하다.",
              "",
              "```text",
              f"{'domain':11} {'frames':>7} {'R0 med[px]':>11} {'R5 med[px]':>11} "
              f"{'Δframe med':>11} {'CI95':>22} {'p':>7}",
              "─" * 76]
    for domain, block in report["domains"].items():
        paired = block["paired_both_detected"]
        boot = paired["frame_median_delta"]
        ci = ("—" if boot["ci95"] is None
              else f"[{boot['ci95'][0]:+.3f}, {boot['ci95'][1]:+.3f}]")
        lines.append(
            f"{domain:11} {paired['n_frames']:7d} "
            f"{number(paired['base_pooled']['median']):>11} "
            f"{number(paired['proposed_pooled']['median']):>11} "
            f"{number(boot['median_delta'], '+.3f'):>11} {ci:>22} "
            f"{number(boot['p_two_sided'], '.3f'):>7}")
    lines += ["```", "",
              "`Δframe med` 는 프레임별 median 오차의 차이이고 CI 는 프레임 재표집이다.",
              "",
              "### 악화가 고르게 퍼졌나, 몇 장에 몰렸나",
              "",
              "```text",
              f"{'domain':11} {'worse':>6} {'better':>7} {'Δp50':>8} {'Δp90':>8} "
              f"{'Δmax':>9} {'상위5 몫':>9}",
              "─" * 62]
    for domain, block in report["domains"].items():
        c = block["paired_both_detected"]["delta_concentration"]
        if not c:
            continue
        lines.append(
            f"{domain:11} {c['n_worse']:6d} {c['n_better']:7d} "
            f"{number(c['delta_p50'], '+.2f'):>8} {number(c['delta_p90'], '+.2f'):>8} "
            f"{number(c['delta_max'], '+.1f'):>9} "
            f"{number(c['top5_share_of_positive'], '.1%'):>9}")
    lines += ["```", "",
              "`상위5 몫` = 악화량 합계에서 가장 나빠진 5 장이 차지하는 비율.",
              "",
              "## §5 R5 가 새로 회수한 프레임 (R5_ONLY) 과 raw 차이의 분해",
              "",
              "통계는 raw 표와 같다 — supervised keypoint 를 통째로 모은 median 이다.",
              "",
              "```text",
              f"{'domain':11} {'회수 frame':>10} {'회수 kp med[px]':>16} "
              f"{'R0 raw':>8} {'R5 raw':>8} {'R5(회수제외)':>13}",
              "─" * 70]
    for domain, block in report["domains"].items():
        rec = block["recovered_by_proposed"]
        pool = block["pooled_keypoint_median"]
        lines.append(
            f"{domain:11} {rec['n_frames']:10d} {number(rec['pooled_median']):>16} "
            f"{number(pool['base_all_detected']['median']):>8} "
            f"{number(pool['proposed_all_detected']['median']):>8} "
            f"{number(pool['proposed_excluding_recovered']['median']):>13}")
    lines += ["```", "",
              "### raw 차이가 어디서 왔나 (px, 합이 정확히 total 이 된다)",
              "",
              "```text",
              f"{'domain':11} {'total':>8} {'base 모집단':>12} {'localisation':>13} "
              f"{'selection':>10}",
              "─" * 58]
    for domain, block in report["domains"].items():
        d = block["decomposition_px"]
        lines.append(
            f"{domain:11} {number(d['total_raw_delta'], '+.3f'):>8} "
            f"{number(d['base_population_effect'], '+.3f'):>12} "
            f"{number(d['localisation_drift_on_both_detected'], '+.3f'):>13} "
            f"{number(d['selection_effect_from_recovered'], '+.3f'):>10}")
    lines += ["```", "",
              "`base 모집단` = R0 가 BOTH 에서 낸 값 − R0 가 자기 전체에서 낸 값 "
              "(R0_ONLY 프레임의 영향).",
              "`localisation` = 같은 프레임에서 R5 가 R0 보다 나빠진 몫.",
              "`selection` = R5 가 새로 회수한 어려운 프레임이 끌어올린 몫.",
              "",
              "## §6 scale-normalised keypoint error (NME)",
              "",
              "분모는 GT 에서만 온다.  raw px 는 지우지 않는다 — 둘 다 보고한다.",
              ""]
    for denominator in ("cuboid_diagonal_px", "gt_bbox_diagonal_px"):
        lines += [f"### 분모 = {denominator}", "", "```text",
                  f"{'domain/model/scope':40} {'n_kp':>6} {'NME med':>10} {'NME p90':>10}",
                  "─" * 68]
        for key, stats in report["scale"][denominator].items():
            lines.append(f"{key:40} {stats['n']:6d} {number(stats['median'], '.4f'):>10} "
                         f"{number(stats['p90'], '.4f'):>10}")
        lines += ["```", ""]
    lines += ["### 분모 자체의 산포 — px 가 비교 가능한 단위인가", "", "```text",
              f"{'domain':11} {'n':>5} {'med[px]':>9} {'p90[px]':>9}", "─" * 36]
    for domain, stats in report["scale"]["denominator_spread"].items():
        lines.append(f"{domain:11} {stats['n']:5d} {number(stats['median'], '.1f'):>9} "
                     f"{number(stats['p90'], '.1f'):>9}")
    lines += ["```", ""]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
