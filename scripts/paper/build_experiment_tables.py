"""결과 artifact 에서 논문 표를 생성한다.  숫자를 손으로 복붙하지 않는다.

읽는 것
    data/pallet/results/paper_eval_v1/arms/ARM_RESULTS.json        모델 결과
    data/pallet/results/paper_eval_v1/baselines/*.json             DOPE / YOLO baseline
    data/pallet/results/paper_selftrain_v1/M4_FILTER_QUALITY.json  filter 채점
    data/pallet/results/paper_selftrain_v1/pseudo_manifests/...    funnel
    challenge/real_gt_v2/manifests/PAPER_EVAL_*.json               population N

쓰는 것
    _docs/paper/generated/TABLE_M1.md ... TABLE_M5.md
    _docs/paper/generated/APPENDIX_TABLES.md
    _docs/paper/ABSTRACT_RESULT_SLOTS.md

불변식 (§33) — 어기면 표를 만들지 않고 실패한다
    M2 Daytime N   == PAPER_DOMAIN_COVERAGE Daytime N
    M2 Nighttime N == PAPER_DOMAIN_COVERAGE Nighttime N
    M5 Plastic N   == PAPER_EVAL Plastic N
    M5 Wood N      == PAPER_EVAL Wood N

pose 가 BLOCKED 면 pose 열을 `—` 로 남긴다.  2D 개선을 pose 개선으로 바꿔 쓰지 않는다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARMS = REPO_ROOT / "data/pallet/results/paper_eval_v1/arms/ARM_RESULTS.json"
BASELINES = REPO_ROOT / "data/pallet/results/paper_eval_v1/baselines"
M4 = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/M4_FILTER_QUALITY.json"
FUNNEL = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/pseudo_manifests/PSEUDO_MANIFEST_SUMMARY.json"
EXPOSURE = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/SELFTRAIN_EXPOSURE_LOCK.json"
DATASET_REPORT = REPO_ROOT / "data/pallet/results/paper_selftrain_v1/PSEUDO_DATASET_REPORT.json"
COVERAGE = REPO_ROOT / "data/evaluation/pallet_eval_v1/reports/PAPER_DOMAIN_COVERAGE.md"
MANIFESTS = REPO_ROOT / "challenge/real_gt_v2/manifests"
OUT = REPO_ROOT / "_docs/paper/generated"

# reader-facing row 이름.  내부 arm id 를 논문 표에 그대로 쓰지 않는다.
M2_ROWS = [
    ("R0", "Synthetic-only"),
    ("R0_CONT", "Source-only continuation"),
    ("R1_NAIVE", "Naive self-training"),
    ("R2_CONF", "Confidence-based self-training"),
    ("R3_CONF_REPROJ", "Reprojection-based self-training"),
    ("R5_PROPOSED", "Proposed"),
]
M3_ROWS = [
    ("R0", "Base"),
    ("R0_CONT", "Source-only continuation"),
    ("R1_NAIVE", "+ self-training (no filter)"),
    ("R2_CONF", "+ Confidence filtering"),
    ("R4_CONF_REMOVE", "+ Keypoint-removal reprojection consistency"),
    ("R5_PROPOSED", "+ Horizontal-flip keypoint consistency"),
]
M5_GROUPS = ["Plastic", "Wood", "Daytime", "Nighttime",
             "Clean", "Occlusion", "Truncation", "Far"]
APPENDIX_GROUPS = ["Low", "Mid", "High", "Lighting_day", "Lighting_night"]


def number(value, spec: str = ".3f") -> str:
    return "—" if value is None else format(value, spec)


def coverage_counts() -> dict[str, int]:
    text = COVERAGE.read_text()
    counts = {}
    for condition in ("Daytime", "Nighttime"):
        match = re.search(rf"^{condition}\s+\S+\s+(\d+)", text, re.MULTILINE)
        if not match:
            raise SystemExit(f"COVERAGE_COUNT_NOT_FOUND: {condition}")
        counts[condition] = int(match.group(1))
    return counts


def manifest_count(name: str) -> int:
    return int(json.loads((MANIFESTS / f"{name}.json").read_text())["expected_count"])


def check_invariants(results: dict) -> dict:
    """표를 만들기 전에 N 이 원천과 맞는지 본다.  틀리면 만들지 않는다."""

    reference = MODEL_FOR_N = next(iter(results.values()))["subgroups"]
    expected = coverage_counts()
    expected["Plastic"] = manifest_count("PAPER_EVAL_PLASTIC_POS")
    expected["Wood"] = manifest_count("PAPER_EVAL_WOOD_POS")
    expected["ALL"] = manifest_count("PAPER_EVAL_ALL_POS")

    failures = []
    for group, value in expected.items():
        actual = reference.get(group, {}).get("N")
        if actual != value:
            failures.append(f"{group}: table={actual} source={value}")
    # 모든 모델이 같은 population 을 봤는지도 본다.
    for name, model in results.items():
        for group in expected:
            if model["subgroups"].get(group, {}).get("N") != expected[group]:
                failures.append(f"{name}/{group} population mismatch")
    if failures:
        raise SystemExit("TABLE_GENERATION_FAILED_INVARIANTS:\n  " + "\n  ".join(failures))
    return expected


def metric_block(model: dict, group: str) -> dict:
    return model["subgroups"].get(group, {})


def build_m1(results: dict, expected: dict) -> str:
    rows = [
        ("DOPE (same-data backbone control)", "DOPE"),
        ("YOLO26n-Pose (synthetic-only)", "R0"),
        ("Proposed", "R5_PROPOSED"),
    ]
    lines = [
        "# Table M1 — Main method comparison",
        "",
        "population `PAPER_EVAL`.  N 은 manifest 에서 읽는다.  세 행 모두 **같은",
        "evaluator·같은 319/2689·같은 metric 정의**로 채점했다 — 별도 채점기를",
        "만들면 행끼리 비교가 성립하지 않는다.",
        "",
        "```text",
        f"{'Method':36} {'N_pos':>6} {'N_neg':>6} {'corner↓':>8} {'det↑':>6} "
        f"{'AP50-95↑':>9} {'AUROC↑':>8} {'FPR95↓':>8} {'R med↓':>8} {'yaw↓':>7}",
        "─" * 112,
    ]
    for label, key in rows:
        if key not in results:
            lines.append(f"{label:36} {expected['ALL']:>6} {2689:>6} "
                         + " ".join(f"{'—':>8}" for _ in range(7)))
            continue
        model = results[key]
        allg = metric_block(model, "ALL")
        lines.append(
            f"{label:36} {expected['ALL']:>6} {2689:>6} "
            f"{number(allg.get('corner_median_px'), '.3f'):>8} "
            f"{number(allg.get('detection_rate_iou50'), '.3f'):>6} "
            f"{number(model.get('box_ap50_95'), '.4f'):>9} "
            f"{number(allg.get('auroc'), '.4f'):>8} "
            f"{number(allg.get('fpr95'), '.4f'):>8} "
            f"{'—':>8} {'—':>7}"
        )
    lines += [
        "```",
        "",
        "`R med` 와 `yaw` 는 `POSE_METRICS_STATUS = BLOCKED` 이라 비워 둔다.",
        "2D 개선을 6D pose 개선이라고 쓰지 않는다.",
        "",
        "## DOPE 행의 비대칭 — 각주로 반드시 남긴다",
        "",
        "DOPE 에는 box head 가 없다.  AP 와 IoU@0.5 매칭에 필요한 box 는 **검출된",
        "cuboid 코너의 bounding box** 로 유도했다.  YOLO 의 box 는 학습된 예측이므로",
        "`AP50-95` 열의 두 값은 같은 양이 아니다.  score 도 DOPE 는 belief peak,",
        "YOLO 는 box confidence라 `AUROC`/`FPR95` 의 척도가 서로 다르다.",
        "",
        "직접 비교가 성립하는 열은 **corner 와 det** 이다 — 둘 다 GT 의 2D keypoint 와",
        "IoU 만 쓰고 모델 고유 출력 형식에 의존하지 않는다.",
        "",
        "DOPE 추론은 reflect-padding 을 썼다.  plain squash 로 돌리면 truncation·근접에서",
        "체계적으로 과소검출되어 DOPE 를 부당하게 나쁘게 만든다.",
        "",
        "## 아직 없는 행",
        "",
        "```text",
        "SingleShotPose   INCOMPATIBLE   저장소에 구현이 없다",
        "PVNet            NEEDS_TRAIN    구현 자산은 있으나 과거 negative 결과가 있다",
        "Real-FT          NEEDS_AUDIT    PAPER_EVAL 과의 학습 중복을 먼저 감사해야 한다",
        "```",
        "",
        "근거는 `_docs/paper/EXTERNAL_BASELINE_AUDIT.md`.  억지 wrapper 로 숫자를",
        "만들지 않는다.",
    ]
    return "\n".join(lines) + "\n"


def build_m2(results: dict, expected: dict) -> str:
    day_reference = metric_block(next(iter(results.values())), "Daytime")
    strict_available = day_reference.get("n_keypoints", 0) > 0

    lines = [
        "# Table M2 — Target-domain adaptation under daytime and nighttime conditions",
        "",
        f"Daytime N={expected['Daytime']}, Nighttime N={expected['Nighttime']} "
        "(plastic only — morphology 를 lighting 효과와 섞지 않는다).",
        "",
        "## primary — detection and ranking",
        "",
        "MAIN Daytime 의 70 프레임은 전부 legacy 세션이고 keypoint visibility 가",
        "unknown 이라 **supervision mask 가 비어 있다**. strict keypoint 오차를",
        "그 조건에서 낼 수 없으므로, 두 조건 모두에서 계산 가능한 detection 과",
        "ranking 을 primary 로 둔다.",
        "",
        "```text",
        f"{'Method':34} {'Day det↑':>9} {'Night det↑':>11} {'Mean↑':>8} {'Worst↑':>8} "
        f"{'AUROC↑':>8} {'FPR95↓':>8}",
        "─" * 92,
    ]
    for key, label in M2_ROWS:
        if key not in results:
            continue
        day = metric_block(results[key], "Daytime")
        night = metric_block(results[key], "Nighttime")
        allg = metric_block(results[key], "ALL")
        values = [day.get("detection_rate_iou50"), night.get("detection_rate_iou50")]
        mean = sum(values) / 2 if all(v is not None for v in values) else None
        worst = min(values) if all(v is not None for v in values) else None
        lines.append(
            f"{label:34} {number(values[0]):>9} {number(values[1]):>11} "
            f"{number(mean):>8} {number(worst):>8} "
            f"{number(allg.get('auroc'), '.4f'):>8} "
            f"{number(allg.get('fpr95'), '.4f'):>8}"
        )
    lines += [
        "```",
        "",
        "detection 은 ↑ 가 좋으므로 `Worst` 는 두 조건 중 **낮은** 쪽이다.",
        "AUROC / FPR95 는 전체 population 대 negative 2,689 로 계산한 frame-level 값이다.",
        "",
        "## secondary — keypoint localisation",
        "",
        "`strict` 는 evaluator 의 supervision mask 를 쓴 값이고, `diagnostic` 은",
        "visibility 가 unknown 인 legacy 점까지 포함한다. **diagnostic 은",
        "visible/occluded 주장이 아니다** — 두 열은 서로 다른 모집단이라 직접 비교하지 않는다.",
        "",
        "```text",
        f"{'Method':34} {'Day strict↓':>12} {'Night strict↓':>14} "
        f"{'Day diag↓':>10} {'Night diag↓':>12} {'ALL strict↓':>12}",
        "─" * 98,
    ]
    for key, label in M2_ROWS:
        if key not in results:
            continue
        day = metric_block(results[key], "Daytime")
        night = metric_block(results[key], "Nighttime")
        allg = metric_block(results[key], "ALL")
        lines.append(
            f"{label:34} {number(day.get('corner_median_px'), '.3f'):>12} "
            f"{number(night.get('corner_median_px'), '.3f'):>14} "
            f"{number(day.get('corner_median_px_all_annotated'), '.3f'):>10} "
            f"{number(night.get('corner_median_px_all_annotated'), '.3f'):>12} "
            f"{number(allg.get('corner_median_px'), '.3f'):>12}"
        )
    lines += [
        "```",
        "",
        f"Daytime strict keypoint 수 = {day_reference.get('n_keypoints', 0)} "
        f"({'사용 가능' if strict_available else 'UNAVAILABLE_METADATA'}).",
        "",
        "POSE_METRIC_BLOCKED: primary 로 쓸 6D pose metric 이 아직 없다.",
        "",
        "모든 ST arm 은 EXPOSURE-MATCHED 다 — 같은 init, 같은 optimizer update 수,",
        "같은 pseudo/synthetic 노출 수. 다른 것은 pseudo-label selection rule 뿐이다.",
    ]
    return "\n".join(lines) + "\n"


def build_m3(results: dict) -> str:
    lines = [
        "# Table M3 — Core self-training component ablation",
        "",
        "```text",
        f"{'Configuration':46} {'corner↓':>8} {'det↑':>7} {'AUROC↑':>8} "
        f"{'FPR95↓':>8} {'R med↓':>8} {'yaw↓':>7}",
        "─" * 96,
    ]
    for key, label in M3_ROWS:
        if key not in results:
            continue
        allg = metric_block(results[key], "ALL")
        lines.append(
            f"{label:46} {number(allg.get('corner_median_px')):>8} "
            f"{number(allg.get('detection_rate_iou50')):>7} "
            f"{number(allg.get('auroc'), '.4f'):>8} "
            f"{number(allg.get('fpr95'), '.4f'):>8} {'—':>8} {'—':>7}"
        )
    lines += ["```", "", "## 단계별 차이", "", "```text"]
    steps = [("R0", "R0_CONT", "추가 최적화 자체 (real pseudo-label 없음)"),
             ("R0_CONT", "R1_NAIVE", "real pseudo-label 로 학습한다는 것 자체"),
             ("R1_NAIVE", "R2_CONF", "confidence filtering"),
             ("R2_CONF", "R4_CONF_REMOVE", "keypoint-removal reprojection consistency"),
             ("R4_CONF_REMOVE", "R5_PROPOSED", "horizontal-flip keypoint consistency")]
    for source, target, meaning in steps:
        if source not in results or target not in results:
            continue
        before = metric_block(results[source], "ALL").get("corner_median_px")
        after = metric_block(results[target], "ALL").get("corner_median_px")
        delta = after - before if before is not None and after is not None else None
        lines.append(
            f"{source:16} -> {target:16} {number(before):>8} -> {number(after):<8} "
            f"Δ {number(delta, '+.3f'):>8}   {meaning}"
        )
    lines += [
        "```",
        "",
        "`R0 -> R1` 을 곧바로 self-training 효과라고 부르지 않는다.",
        "그 차이에는 추가 최적화 자체의 몫이 섞여 있고, 그 몫이 `R0 -> R0-CONT` 다.",
    ]
    return "\n".join(lines) + "\n"


def build_m4() -> str:
    if not M4.exists():
        return "# Table M4 — filter quality\n\n결과 artifact 가 아직 없다.\n"
    report = json.loads(M4.read_text())
    funnel = json.loads(FUNNEL.read_text())["funnel"] if FUNNEL.exists() else {}
    lines = [
        "# Table M4 — Pseudo-label filter quality",
        "",
        f"population `{report['population']}`  N={report['n_frames']}  "
        f"detected={report['n_detected']}  CORRECT_2D={report['n_correct_2d']}",
        "",
        f"CORRECT_2D = {report['criterion']['CORRECT_2D']}  "
        f"(gross {report['criterion']['gross_px']} px, "
        f"{report['criterion']['source']})",
        "",
        "```text",
        f"{'Filter':44} {'Pass':>5} {'Ret.':>6} {'Pass~px':>8} {'Rej~px':>8} "
        f"{'Sep↑':>7} {'Gross↓':>7} {'Prec↑':>6} {'Rec↑':>6} {'F1↑':>6}",
        "─" * 118,
    ]
    for key, value in report["filters"].items():
        lines.append(
            f"{value['reader_facing_name']:44} {value['accepted']:>5} "
            f"{number(value['retention']):>6} "
            f"{number(value['pass']['median_px'], '.2f'):>8} "
            f"{number(value['reject']['median_px'], '.2f'):>8} "
            f"{number(value['separation_px'], '.2f'):>7} "
            f"{number(value['pass']['gross_rate']):>7} "
            f"{number(value['precision']):>6} {number(value['recall']):>6} "
            f"{number(value['f1']):>6}"
        )
    lines += ["```", "", "## Confidence bin 진단", "",
              "\"0.7~0.8 을 넘으면 실제로 더 맞는가\" 에 답한다.", "", "```text",
              f"{'box_conf bin':16} {'N':>5} {'corner~px':>10} {'p90':>8} {'gross':>8}",
              "─" * 52]
    for name, stats in report["confidence_bins"].items():
        lines.append(f"{name:16} {stats['n_frames']:>5} "
                     f"{number(stats['median_px'], '.2f'):>10} "
                     f"{number(stats['p90_px'], '.2f'):>8} "
                     f"{number(stats['gross_rate']):>8}")
    lines += ["```"]
    if funnel:
        lines += ["", "## Pseudo-label funnel (unlabeled pool)", "", "```text"]
        for key, value in funnel.items():
            lines.append(f"{key:32} {value}")
        lines += ["```"]
    lines += ["", "threshold 는 이 결과를 보기 전에 동결됐고, 보고 나서 바꾸지 않았다."]
    return "\n".join(lines) + "\n"


def build_m5(results: dict, expected: dict) -> str:
    lines = [
        "# Table M5 — Robustness and pallet morphology generalization",
        "",
        "Synthetic-only(R0) 대 Proposed(R5).  새 학습 없이 subgroup 평가만 한다.",
        "",
        "```text",
        f"{'Condition':14} {'N':>5} {'R0 corner↓':>11} {'R5 corner↓':>11} {'Δ':>8} "
        f"{'R0 det↑':>8} {'R5 det↑':>8}",
        "─" * 74,
    ]
    for group in M5_GROUPS:
        base = metric_block(results.get("R0", {"subgroups": {}}), group)
        proposed = metric_block(results.get("R5_PROPOSED", {"subgroups": {}}), group)
        before, after = base.get("corner_median_px"), proposed.get("corner_median_px")
        delta = after - before if before is not None and after is not None else None
        lines.append(
            f"{group:14} {base.get('N', 0):>5} {number(before):>11} {number(after):>11} "
            f"{number(delta, '+.3f'):>8} "
            f"{number(base.get('detection_rate_iou50')):>8} "
            f"{number(proposed.get('detection_rate_iou50')):>8}"
        )
    lines += ["```", "",
              "조건은 서로 중복될 수 있다. N 은 PAPER_EVAL manifest 와 workspace tag 에서 온다.",
              "Low/Mid/High 와 넓은 lighting 분할은 Appendix 로 뺀다."]
    return "\n".join(lines) + "\n"


def build_appendix(results: dict) -> str:
    lines = ["# Appendix tables", "", "## A7 — elevation and broad lighting subgroups", "",
             "```text",
             f"{'Condition':16} {'N':>5} {'R0 corner↓':>11} {'R5 corner↓':>11} {'Δ':>8}",
             "─" * 56]
    for group in APPENDIX_GROUPS:
        base = metric_block(results.get("R0", {"subgroups": {}}), group)
        proposed = metric_block(results.get("R5_PROPOSED", {"subgroups": {}}), group)
        before, after = base.get("corner_median_px"), proposed.get("corner_median_px")
        delta = after - before if before is not None and after is not None else None
        lines.append(f"{group:16} {base.get('N', 0):>5} {number(before):>11} "
                     f"{number(after):>11} {number(delta, '+.3f'):>8}")
    lines += ["```", ""]

    if DATASET_REPORT.exists() and EXPOSURE.exists():
        dataset = json.loads(DATASET_REPORT.read_text())
        exposure = json.loads(EXPOSURE.read_text())
        lines += ["## A1 — pseudo-label counts and exposure contract", "", "```text",
                  f"{'Arm':16} {'filter':18} {'unique PL':>10} {'pseudo/unique':>14} "
                  f"{'pseudo exp':>11} {'synth exp':>10}",
                  "─" * 84]
        for arm, stats in dataset["arms"].items():
            lines.append(
                f"{arm:16} {str(stats['filter']):18} {stats['unique_pseudo_labels']:>10} "
                f"{str(stats['pseudo_exposures_per_unique']):>14} "
                f"{stats['pseudo_exposures_per_epoch'] * exposure['epochs']:>11} "
                f"{stats['synthetic_exposures_per_epoch'] * exposure['epochs']:>10}"
            )
        lines += ["```", "",
                  f"모든 arm 이 같은 {exposure['total_optimizer_updates']} optimizer update 를 쓴다.",
                  "MAIN 은 EXPOSURE-MATCHED 이고, unique PL 개수를 맞추는 실험은 A2 다.", ""]

    # ── A3 repeatability ────────────────────────────────────────────────
    replicates = {
        "Naive self-training": ["R1_NAIVE", "R1_NAIVE_P43", "R1_NAIVE_P44"],
        "Proposed": ["R5_PROPOSED", "R5_PROPOSED_P43", "R5_PROPOSED_P44"],
    }
    if all(k in results for keys in replicates.values() for k in keys):
        import statistics
        lines += [
            "## A3 — repeatability across pseudo-sampling replicates",
            "",
            "Ultralytics 의 `seed` 는 dataloader 에 도달하지 않는다.  seed 42/43/44 로",
            "학습한 가중치는 **비트 동일**했다 (max|Δw| = 0, 텐서 비교로 확인) — 따라서",
            "seed override 는 독립 반복이 아니다.  여기서 쓰는 replicate 는 우리가",
            "통제하는 **pseudo 샘플링**을 바꾼 것이고, 노출 총량은 그대로다.",
            "",
            "```text",
            f"{'Method':24} {'metric':10} {'rep1':>8} {'rep2':>8} {'rep3':>8} "
            f"{'mean':>8} {'std':>8}",
            "─" * 82,
        ]
        for label, keys in replicates.items():
            for metric, spec, path in (("corner↓", ".3f", "corner_median_px"),
                                       ("det↑", ".3f", "detection_rate_iou50"),
                                       ("AUROC↑", ".4f", "auroc"),
                                       ("FPR95↓", ".4f", "fpr95")):
                values = [metric_block(results[k], "ALL").get(path) for k in keys]
                if any(v is None for v in values):
                    continue
                lines.append(
                    f"{label:24} {metric:10} "
                    + " ".join(f"{number(v, spec):>8}" for v in values)
                    + f" {number(statistics.mean(values), spec):>8}"
                    + f" {number(statistics.pstdev(values), spec):>8}"
                )
        lines += ["```", ""]
        gaps = []
        for metric, path, lower_is_better in (("corner", "corner_median_px", True),
                                              ("AUROC", "auroc", False),
                                              ("FPR95", "fpr95", True)):
            naive = [metric_block(results[k], "ALL")[path] for k in replicates["Naive self-training"]]
            proposed = [metric_block(results[k], "ALL")[path] for k in replicates["Proposed"]]
            separated = (max(proposed) < min(naive)) if lower_is_better else (min(proposed) > max(naive))
            gaps.append(f"{metric}: {'구간 분리' if separated else '구간 겹침'}")
        lines += [
            "세 replicate 에서 Proposed 와 Naive 의 구간이 겹치는지: " + " · ".join(gaps),
            "",
            "구간이 겹치면 그 지표에서는 효과가 산포 안이라는 뜻이므로 주장하지 않는다.",
            "",
        ]

    # ── A2 unique-quantity-matched control ──────────────────────────────
    a2 = ["A2_MATCHED_S1", "A2_MATCHED_S2", "A2_MATCHED_S3"]
    proposed = ["R5_PROPOSED", "R5_PROPOSED_P43", "R5_PROPOSED_P44"]
    if all(k in results for k in a2 + proposed):
        import statistics
        lines += [
            "## A2 — unique-quantity-matched control",
            "",
            "성능 향상이 **선별 품질** 때문인지 **pseudo-label 개수** 때문인지 가른다.",
            "Naive pool 에서 Proposed 와 같은 unique 개수(259)를 무작위로 뽑아,",
            "나머지는 전부 같게 두고 3 회 반복했다.  MAIN 의 EXPOSURE-MATCHED 와는",
            "다른 실험이다 — 여기서 맞추는 것은 노출량이 아니라 unique 개수다.",
            "",
            "```text",
            f"{'Arm':28} {'metric':10} {'rep1':>8} {'rep2':>8} {'rep3':>8} "
            f"{'mean':>8} {'std':>8}",
            "─" * 86,
        ]
        for label, keys in (("Naive, quantity-matched", a2), ("Proposed", proposed)):
            for metric, spec, path in (("corner↓", ".3f", "corner_median_px"),
                                       ("AUROC↑", ".4f", "auroc"),
                                       ("FPR95↓", ".4f", "fpr95")):
                values = [metric_block(results[k], "ALL").get(path) for k in keys]
                if any(v is None for v in values):
                    continue
                lines.append(
                    f"{label:28} {metric:10} "
                    + " ".join(f"{number(v, spec):>8}" for v in values)
                    + f" {number(statistics.mean(values), spec):>8}"
                    + f" {number(statistics.pstdev(values), spec):>8}"
                )
        lines += ["```", ""]
        verdicts = []
        for metric, path, lower in (("corner", "corner_median_px", True),
                                    ("AUROC", "auroc", False),
                                    ("FPR95", "fpr95", True)):
            control = [metric_block(results[k], "ALL")[path] for k in a2]
            ours = [metric_block(results[k], "ALL")[path] for k in proposed]
            sep = (max(ours) < min(control)) if lower else (min(ours) > max(control))
            verdicts.append(f"{metric}: {'구간 분리' if sep else '구간 겹침'}")
        lines += [
            "판정: " + " · ".join(verdicts),
            "",
            "구간이 겹치는 지표에서는 **개수를 맞춘 무작위 선별로도 비슷한 값에",
            "도달한다**는 뜻이다.  그 지표에 대해서는 geometry filter 의 기여를",
            "주장하지 않는다.  개수 효과와 선별 품질 효과를 합쳐 말하지 않는다.",
            "",
        ]

    lines += ["## External keypoint baselines", "", "```text",
              "SingleShotPose   NOT_EVALUATED   repository audit 미실시",
              "PVNet            NOT_EVALUATED   repository audit 미실시",
              "```", "",
              "억지 wrapper 로 숫자를 만들지 않는다. 감사 결과가 나오면 여기 채운다."]
    return "\n".join(lines) + "\n"


def build_abstract_slots(results: dict, expected: dict) -> str:
    base = metric_block(results.get("R0", {"subgroups": {}}), "ALL")
    proposed = metric_block(results.get("R5_PROPOSED", {"subgroups": {}}), "ALL")
    before, after = base.get("corner_median_px"), proposed.get("corner_median_px")
    improvement = (
        (before - after) / before * 100 if before and after is not None else None
    )
    worst_before = worst_after = None
    if results.get("R0") and results.get("R5_PROPOSED"):
        pairs = [
            (metric_block(results["R0"], group).get("corner_median_px"),
             metric_block(results["R5_PROPOSED"], group).get("corner_median_px"))
            for group in ("Daytime", "Nighttime")
        ]
        if all(a is not None and b is not None for a, b in pairs):
            worst_before = max(a for a, _ in pairs)
            worst_after = max(b for _, b in pairs)

    return "\n".join([
        "# Abstract result slots",
        "",
        "초록에 넣을 수 있는 값과, 아직 넣을 수 없는 값을 구분한다.",
        "",
        "```text",
        f"Strongest baseline        Synthetic-only YOLO26n-Pose (R0)",
        f"Primary metric            supervised keypoint location median px (PAPER_EVAL)",
        f"Baseline value            {number(before, '.3f')} px",
        f"Proposed value            {number(after, '.3f')} px",
        f"Improvement X             {number(improvement, '.1f')} %",
        f"Worst-condition before    {number(worst_before, '.3f')} px",
        f"Worst-condition after     {number(worst_after, '.3f')} px",
        f"YAW_RESULT_SLOT           BLOCKED",
        "```",
        "",
        "`YAW_RESULT_SLOT = BLOCKED` 이므로 초록의",
        '"reduces median yaw error by [Y]" 문장은 **사용할 수 없다**.',
        "다른 2D metric 을 yaw 라고 바꿔 쓰지 않는다.",
        "",
        "pose evaluator 가 READY 가 되면 R0~R5 를 같은 evaluator 로 다시 배치 평가하고",
        "이 파일과 M1/M2/M3/M5 의 pose 열을 함께 갱신한다.",
    ]) + "\n"


def main() -> int:
    if not ARMS.exists():
        raise SystemExit(f"ARM_RESULTS_MISSING: {ARMS}")
    results = json.loads(ARMS.read_text())["models"]
    expected = check_invariants(results)
    OUT.mkdir(parents=True, exist_ok=True)

    written = {
        "TABLE_M1.md": build_m1(results, expected),
        "TABLE_M2.md": build_m2(results, expected),
        "TABLE_M3.md": build_m3(results),
        "TABLE_M4.md": build_m4(),
        "TABLE_M5.md": build_m5(results, expected),
        "APPENDIX_TABLES.md": build_appendix(results),
    }
    for name, text in written.items():
        (OUT / name).write_text(text)
        print(f"wrote _docs/paper/generated/{name}")
    (REPO_ROOT / "_docs/paper/ABSTRACT_RESULT_SLOTS.md").write_text(
        build_abstract_slots(results, expected)
    )
    print("wrote _docs/paper/ABSTRACT_RESULT_SLOTS.md")
    print(f"\ninvariants OK: {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
