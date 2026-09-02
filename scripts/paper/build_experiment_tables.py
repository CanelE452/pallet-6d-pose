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
        "2026-09-02 의 visibility 확정 전에는 MAIN Daytime 70 장의 supervision mask 가",
        "비어 strict keypoint 오차를 낼 수 없었다. 지금은 319 장 전부 strict 를 내지만,",
        "arm 을 가르는 축은 여전히 detection 과 ranking 이다. 그래서 detection 과",
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
        "visibility 와 무관하게 좌표가 있는 점을 전부 센다. **diagnostic 은",
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

    # ── 두 제안 필터의 2x2 단독/조합 ablation ────────────────────────────
    cells = [
        ("neither (Confidence only)", ["R2_CONF", "R2_CONF_P43", "R2_CONF_P44"]),
        ("+ Reprojection only",
         ["R3_CONF_REPROJ", "R3_CONF_REPROJ_P43", "R3_CONF_REPROJ_P44"]),
        ("+ Keypoint-removal only",
         ["R4_CONF_REMOVE", "R4_CONF_REMOVE_P43", "R4_CONF_REMOVE_P44"]),
        ("+ Horizontal-flip only",
         ["R6_CONF_FLIP", "R6_CONF_FLIP_P43", "R6_CONF_FLIP_P44"]),
        ("+ both (Proposed)", ["R5_PROPOSED", "R5_PROPOSED_P43", "R5_PROPOSED_P44"]),
    ]
    if all(key in results for _, keys in cells for key in keys):
        import statistics
        lines += [
            "",
            "## 각 기하 필터의 단독 기여",
            "",
            "위 누적 표만으로는 flip 의 **단독** 기여를 못 뽑는다.  `R4 -> R5` 는",
            "keypoint-removal 이 이미 걸린 상태에서 flip 을 더한 값이기 때문이다.",
            "그래서 각 필터를 단독으로 학습했다.  모두 같은 confidence 전처리 위에서,",
            "같은 exposure·update·init 으로 돌았고 replicate 3 회씩이다.",
            "",
            "```text",
            f"{'Configuration':28} {'unique PL':>10} {'corner↓ mean':>13} {'std':>7} "
            f"{'AUROC↑ mean':>12} {'FPR95↓ mean':>12}",
            "─" * 88,
        ]
        stats_by_cell = {}
        for label, keys in cells:
            corner = [metric_block(results[k], "ALL")["corner_median_px"] for k in keys]
            auroc = [metric_block(results[k], "ALL")["auroc"] for k in keys]
            fpr = [metric_block(results[k], "ALL")["fpr95"] for k in keys]
            # unique PL 은 manifest summary 에서 읽는다.  PSEUDO_DATASET_REPORT 는
            # 마지막 빌드의 arm 만 담고 있어 여기 쓰면 대부분 비어 보인다.
            unique = "—"
            if FUNNEL.exists():
                accepted = {
                    v["reader_facing_name"]: v["accepted"]
                    for v in json.loads(FUNNEL.read_text())["arms"].values()
                }
                for name, value in accepted.items():
                    if label.startswith("neither") and name == "Confidence":
                        unique = str(value)
                    elif "Reprojection only" in label and name.endswith("Reprojection"):
                        unique = str(value)
                    elif "Keypoint-removal only" in label and "Keypoint-removal" in name:
                        unique = str(value)
                    elif "flip only" in label and "Horizontal-flip" in name:
                        unique = str(value)
                    elif "both" in label and name == "Proposed":
                        unique = str(value)
            stats_by_cell[label] = (corner, auroc, fpr)
            lines.append(
                f"{label:28} {unique:>10} "
                f"{statistics.mean(corner):>13.3f} {statistics.pstdev(corner):>7.3f} "
                f"{statistics.mean(auroc):>12.4f} {statistics.mean(fpr):>12.4f}"
            )
        lines += ["```", "", "### 단독 기여와 상호작용", "", "```text"]
        base_corner = statistics.mean(stats_by_cell["neither (Confidence only)"][0])
        removal_only = statistics.mean(stats_by_cell["+ Keypoint-removal only"][0])
        flip_only = statistics.mean(stats_by_cell["+ Horizontal-flip only"][0])
        both = statistics.mean(stats_by_cell["+ both (Proposed)"][0])
        lines += [
            f"keypoint-removal 단독   {base_corner:.3f} -> {removal_only:.3f}   "
            f"Δ {removal_only - base_corner:+.3f}",
            f"horizontal-flip 단독    {base_corner:.3f} -> {flip_only:.3f}   "
            f"Δ {flip_only - base_corner:+.3f}",
            f"둘 다 (Proposed)        {base_corner:.3f} -> {both:.3f}   "
            f"Δ {both - base_corner:+.3f}",
            "",
            f"단독 합                 Δ {(removal_only - base_corner) + (flip_only - base_corner):+.3f}",
            f"실제 조합               Δ {both - base_corner:+.3f}",
            f"상호작용                Δ {(both - base_corner) - ((removal_only - base_corner) + (flip_only - base_corner)):+.3f}",
            "```",
            "",
            "상호작용 항이 음수면 두 필터가 서로를 보완하고, 양수면 겹치는 일을 한다.",
            "replicate 3 회의 std 를 함께 보고 산포보다 큰 차이만 주장한다.",
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
              "\"0.7~0.8 을 넘으면 실제로 더 맞는가\" 에 답한다.",
              "",
              "`src` 는 keypoint 통계 출처다.  `strict` 는 evaluator 의 supervision mask,",
              "`diag` 는 all-annotated (visibility 무시, 좌표가 있는 점 전부).",
              "저신뢰 bin 은 전부 legacy 프레임이라 strict 가 비어 diag 로 채웠다.",
              "**두 출처의 절대값을 직접 비교하지 않는다.**",
              "",
              "```text",
              f"{'box_conf bin':16} {'N':>5} {'src':>7} {'n_kp':>6} {'corner~px':>10} "
              f"{'p90':>9} {'gross':>8}",
              "─" * 66]
    for name, stats in report["confidence_bins"].items():
        lines.append(f"{name:16} {stats['n_frames']:>5} "
                     f"{stats.get('source', 'strict'):>7} {stats['n_keypoints']:>6} "
                     f"{number(stats['median_px'], '.2f'):>10} "
                     f"{number(stats['p90_px'], '.2f'):>9} "
                     f"{number(stats['gross_rate']):>8}")
    lines += ["```", "",
              "confidence 가 TAU_BOX 아래인 검출은 눈에 띄게 나쁘다 — corner 가 한 자릿수에서",
              "두 자릿수로 뛰고 gross rate 도 몇 배가 된다.  confidence pre-filter 가 하는",
              "일이 여기서 보인다.  다만 그 위 구간(0.70~1.00) 안에서는 단조 개선이 아니다."]
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
        "`src` 열이 그 행의 keypoint 통계 출처다.",
        "",
        "```text",
        "strict   evaluator 의 supervision mask (reviewed visibility)",
        "diag     all-annotated — visibility 를 무시하고 좌표가 있는 점을 전부 센다.",
        "         visible/occluded 주장이 아니다.  strict 가 0 개인 조건에서만 쓴다.",
        "```",
        "",
        "```text",
        f"{'Condition':14} {'N':>5} {'src':>6} {'n_kp':>6} {'R0 corner↓':>11} "
        f"{'R5 corner↓':>11} {'Δ':>8} {'R0 det↑':>8} {'R5 det↑':>8}",
        "─" * 92,
    ]
    for group in M5_GROUPS:
        base = metric_block(results.get("R0", {"subgroups": {}}), group)
        proposed = metric_block(results.get("R5_PROPOSED", {"subgroups": {}}), group)
        # strict 가 비면 all-annotated 진단으로 채우고 출처를 밝힌다.  숫자가 있는데
        # 비워 두지 않되, 두 출처를 같은 값처럼 보이게 하지도 않는다.
        if base.get("n_keypoints", 0) > 0:
            source, count = "strict", base.get("n_keypoints", 0)
            before = base.get("corner_median_px")
            after = proposed.get("corner_median_px")
        else:
            source, count = "diag", base.get("n_keypoints_annotated", 0)
            before = base.get("corner_median_px_all_annotated")
            after = proposed.get("corner_median_px_all_annotated")
        delta = after - before if before is not None and after is not None else None
        lines.append(
            f"{group:14} {base.get('N', 0):>5} {source:>6} {count:>6} "
            f"{number(before):>11} {number(after):>11} {number(delta, '+.3f'):>8} "
            f"{number(base.get('detection_rate_iou50')):>8} "
            f"{number(proposed.get('detection_rate_iou50')):>8}"
        )
    lines += [
        "```",
        "",
        "조건은 서로 중복될 수 있다. N 은 PAPER_EVAL manifest 와 workspace tag 에서 온다.",
        "",
        "`diag` 행과 `strict` 행의 절대값을 서로 비교하지 않는다 — 다른 모집단이다.",
        "행 안에서의 R0 대 R5 비교는 같은 모집단이므로 유효하다.",
        "",
        "Low/Mid/High 와 넓은 lighting 분할은 Appendix A7 로 뺀다.",
    ]
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

    # ── A12 self-training strength sensitivity ──────────────────────────
    sensitivity = [("0.25", "A12_PSEUDO25"), ("0.50 (MAIN)", "R5_PROPOSED"),
                   ("0.75", "A12_PSEUDO75")]
    if all(key in results for _, key in sensitivity):
        lines += [
            "## A12 — self-training strength sensitivity",
            "",
            "Proposed(F4) pseudo-label manifest 를 그대로 쓰고 pseudo:synthetic 비율만",
            "바꿨다.  총 optimizer update · LR · init · augmentation · seed 는 모두 같다.",
            "",
            "묻는 것은 \"Proposed 의 효과가 특정 mixing ratio 에만 의존하는가\" 다.",
            "**hyperparameter search 가 아니다** — MAIN row 는 결과와 무관하게 0.50 이다.",
            "",
            "```text",
            f"{'Pseudo fraction':16} {'corner↓':>8} {'det↑':>7} {'AUROC↑':>8} "
            f"{'FPR95↓':>8} {'Day↓*':>8} {'Night↓*':>9}",
            "─" * 70,
        ]
        for label, key in sensitivity:
            allg = metric_block(results[key], "ALL")
            day = metric_block(results[key], "Daytime")
            night = metric_block(results[key], "Nighttime")
            lines.append(
                f"{label:16} {number(allg.get('corner_median_px')):>8} "
                f"{number(allg.get('detection_rate_iou50')):>7} "
                f"{number(allg.get('auroc'), '.4f'):>8} "
                f"{number(allg.get('fpr95'), '.4f'):>8} "
                f"{number(day.get('corner_median_px_all_annotated')):>8} "
                f"{number(night.get('corner_median_px_all_annotated')):>9}"
            )
        values = [metric_block(results[k], "ALL")["corner_median_px"]
                  for _, k in sensitivity]
        spread = max(values) - min(values)
        lines += [
            "```",
            "",
            f"corner spread across ratios: {spread:.3f} px "
            f"(min {min(values):.3f} / max {max(values):.3f})",
            "",
            "* Day/Night 는 all-annotated 진단값이다.",
            "",
            "0.25 나 0.75 가 더 좋아도 MAIN row 를 교체하지 않는다.",
            "",
        ]

    # ── Real-FT supervised upper bound ──────────────────────────────────
    realft = [("ft_a (real157+neg259+synth12k)", "REALFT_A"),
              ("ft_b (patience0 ep40)", "REALFT_B"),
              ("legacy v1v2 FT", "REALFT_LV1V2")]
    if any(key in results for _, key in realft):
        lines += [
            "## Supervised upper bound (Real-FT)",
            "",
            "**controlled comparison 이 아니다.**  이 checkpoint 들은 real GT 로 직접",
            "학습했고, base 도 R0 가 아니라 다른 synthetic run 이다.  M1 의 controlled",
            "row 로 읽으면 안 된다 — 도달 가능한 상한을 가늠하는 용도다.",
            "",
            "leakage 감사: PAPER_EVAL 319 와의 중복이 **이미지 SHA 0 건, 파일명 stem",
            "0 건** 이다.  따라서 `LEAKED_SUPERVISED_UPPER_BOUND` 가 아니라 그냥",
            "`SUPERVISED UPPER BOUND` 로 표기한다.",
            "",
            "```text",
            f"{'Checkpoint':34} {'corner↓':>8} {'det↑':>7} {'AP50-95↑':>9} "
            f"{'AUROC↑':>8} {'FPR95↓':>8}",
            "─" * 80,
        ]
        for label, key in realft + [("Proposed (label-free)", "R5_PROPOSED")]:
            if key not in results:
                continue
            allg = metric_block(results[key], "ALL")
            lines.append(
                f"{label:34} {number(allg.get('corner_median_px')):>8} "
                f"{number(allg.get('detection_rate_iou50')):>7} "
                f"{number(results[key].get('box_ap50_95'), '.4f'):>9} "
                f"{number(allg.get('auroc'), '.4f'):>8} "
                f"{number(allg.get('fpr95'), '.4f'):>8}"
            )
        lines += ["```", "",
                  "Proposed 는 real label 을 한 장도 쓰지 않았다.  같은 표에 두는 이유는",
                  "상한과의 거리를 보이기 위해서지 같은 조건의 비교라서가 아니다.", ""]

    # ── A2b self-training baseline comparison (aggregate) ───────────────
    a2b = [("Synthetic only", "R0"), ("Naive ST", "R1_NAIVE"),
           ("Reproj-only ST", "R3_CONF_REPROJ"), ("Ours", "R5_PROPOSED")]
    if all(k in results for _, k in a2b):
        lines += [
            "## A2b — self-training baseline comparison (aggregate)",
            "",
            "제안 방법의 효과가 단순 self-training 이나 reprojection filtering 만으로",
            "얻어지는지 확인한다.  전체 population 집계다 (도메인별은 M2).",
            "",
            "```text",
            f"{'Method':18} {'corner↓':>8} {'det↑':>7} {'@5px↑':>7} {'@10px↑':>7} "
            f"{'@20px↑':>7} {'AP↑':>7} {'AUROC↑':>8} {'FPR95↓':>8} {'R med↓':>8} {'yaw↓':>7}",
            "─" * 104,
        ]
        for label, key in a2b:
            a = metric_block(results[key], "ALL")
            lines.append(
                f"{label:18} {number(a.get('corner_median_px')):>8} "
                f"{number(a.get('detection_rate_iou50')):>7} "
                f"{number(a.get('proj_at_5px')):>7} {number(a.get('proj_at_10px')):>7} "
                f"{number(a.get('proj_at_20px')):>7} "
                f"{number(results[key].get('box_ap50_95'), '.4f'):>7} "
                f"{number(a.get('auroc'), '.4f'):>8} {number(a.get('fpr95'), '.4f'):>8} "
                f"{'—':>8} {'—':>7}"
            )
        lines += ["```", "", "반드시 답해야 하는 질문:", "", "```text"]
        base = metric_block(results["R0"], "ALL")["corner_median_px"]
        naive = metric_block(results["R1_NAIVE"], "ALL")["corner_median_px"]
        reproj = metric_block(results["R3_CONF_REPROJ"], "ALL")["corner_median_px"]
        ours = metric_block(results["R5_PROPOSED"], "ALL")["corner_median_px"]
        lines += [
            f"1. synthetic only 보다 self-training 이 좋은가   {base:.3f} -> {naive:.3f}"
            f"   {'예' if naive < base else '아니오'}",
            f"2. naive 보다 filtering 이 좋은가                {naive:.3f} -> {ours:.3f}"
            f"   {'예' if ours < naive else '아니오'}",
            f"3. reproj-only 보다 제안 filter 가 좋은가        {reproj:.3f} -> {ours:.3f}"
            f"   {'예' if ours < reproj else '아니오'}",
            "```",
            "",
            "1·3 은 replicate 산포와 함께 읽어야 한다 — A3 · 2x2 표 참조.",
            "",
        ]

    # ── A7 full robustness metric battery ───────────────────────────────
    battery = ["Plastic", "Wood", "Daytime", "Nighttime", "Lighting_day",
               "Lighting_night", "Clean", "Occlusion", "Truncation", "Far",
               "Low", "Mid", "High"]
    if "R5_PROPOSED" in results:
        lines += [
            "## A7 — full robustness metric battery (Proposed)",
            "",
            "M5 는 지면 때문에 지표를 줄여 싣는다.  여기서는 같은 subgroup 을 전체 열로 본다.",
            "subgroup 은 서로 중복될 수 있어 합계가 전체 N 이 되지 않는다.",
            "",
            "```text",
            f"{'Subgroup':16} {'N':>5} {'src':>7} {'corner↓':>8} {'p90↓':>8} "
            f"{'@5px↑':>7} {'@10px↑':>7} {'@20px↑':>7} {'gross↓':>7} "
            f"{'det↑':>7} {'AUROC↑':>8} {'FPR95↓':>8} {'R med↓':>7} {'yaw↓':>6}",
            "─" * 122,
        ]
        for group in battery:
            g = metric_block(results["R5_PROPOSED"], group)
            if not g.get("N"):
                continue
            strict = g.get("n_keypoints", 0) > 0
            suffix = "" if strict else "_all_annotated"
            lines.append(
                f"{group:16} {g['N']:>5} {('strict' if strict else 'diag'):>7} "
                f"{number(g.get('corner_median_px' + suffix)):>8} "
                f"{number(g.get('corner_p90_px' + suffix), '.2f'):>8} "
                f"{number(g.get('proj_at_5px' + suffix)):>7} "
                f"{number(g.get('proj_at_10px' + suffix)):>7} "
                f"{number(g.get('proj_at_20px' + suffix)):>7} "
                f"{number(g.get('gross_rate' + suffix)):>7} "
                f"{number(g.get('detection_rate_iou50')):>7} "
                f"{number(g.get('auroc'), '.4f'):>8} {number(g.get('fpr95'), '.4f'):>8} "
                f"{'—':>7} {'—':>6}"
            )
        lines += ["```", "",
                  "`src` 가 diag 인 행은 all-annotated 진단값이다 — strict 행과 절대값을",
                  "직접 비교하지 않는다.  pose 열은 BLOCKED.", ""]

    # ── A9 same-data backbone control ───────────────────────────────────
    if "DOPE" in results and "R0" in results:
        lines += [
            "## A9 — same-data backbone control",
            "",
            "\"왜 YOLO26 인가\" 에 답한다.  두 모델은 같은 55,980 synthetic frame 으로",
            "60 epoch 학습했고 real 감독은 0 이다.  다른 것은 백본뿐이다.",
            "",
            "```text",
            f"{'Model':16} {'Train frames':>13} {'Epochs':>7} {'det↑':>7} "
            f"{'corner med↓':>12} {'corner p90↓':>12} {'@5px↑':>7} {'@10px↑':>7} {'@20px↑':>7}",
            "─" * 94,
        ]
        for label, key in (("DOPE", "DOPE"), ("YOLO26n-Pose", "R0")):
            a = metric_block(results[key], "ALL")
            lines.append(
                f"{label:16} {'55,980':>13} {'60':>7} "
                f"{number(a.get('detection_rate_iou50')):>7} "
                f"{number(a.get('corner_median_px')):>12} "
                f"{number(a.get('corner_p90_px'), '.2f'):>12} "
                f"{number(a.get('proj_at_5px')):>7} {number(a.get('proj_at_10px')):>7} "
                f"{number(a.get('proj_at_20px')):>7}"
            )
        lines += ["```", "",
                  "`det 8/8` 와 `det>=6` 은 넣지 않았다.  DOPE 의 코너 검출은 belief",
                  "threshold, YOLO 는 keypoint confidence 라 같은 양이 아니다 — 한 열에",
                  "놓으면 비교처럼 보이지만 비교가 아니다.  대신 GT 와의 거리로만 정의되는",
                  "corner 와 Proj@N 을 쓴다.", ""]

    # ── A1 도메인별 pass / retention ────────────────────────────────────
    if FUNNEL.exists():
        summary = json.loads(FUNNEL.read_text())
        funnel = summary["funnel"]
        pool = {"Daytime": 500, "Nighttime": 500}
        order = ["F0_NAIVE", "F1_CONF", "F2_CONF_REPROJ", "F3_CONF_REMOVE",
                 "F5_CONF_FLIP", "F4_PROPOSED"]
        present = [a for a in order if a in summary["arms"]]
        lines += [
            "## A1 — pseudo-label pass / retention by domain",
            "",
            "필터가 unlabeled pool 에서 pseudo-label 을 얼마나 남기는지 본다.",
            "**모델 정확도 표가 아니다** — 통과 수가 적다는 사실만으로 품질을 주장하지 않는다.",
            "",
            "```text",
            f"{'Filter':44} {'Day pass':>9} {'Day ret':>8} "
            f"{'Night pass':>11} {'Night ret':>10} {'Total':>7}",
            "─" * 94,
        ]
        for arm in present:
            v = summary["arms"][arm]
            day, night = v["daytime_accepted"], v["nighttime_accepted"]
            lines.append(
                f"{v['reader_facing_name']:44} {day:>9} "
                f"{day / pool['Daytime']:>8.3f} {night:>11} "
                f"{night / pool['Nighttime']:>10.3f} {v['accepted']:>7}"
            )
        lines += ["```", "", "funnel (unlabeled 1000 장):", "", "```text"]
        for key, value in funnel.items():
            lines.append(f"{key:34} {value}")
        lines += ["```", "",
                  "retention = 통과 수 / 그 도메인 pool 500.", ""]

    # ── A6 evaluation dataset composition ───────────────────────────────
    lines += [
        "## A6 — evaluation dataset composition",
        "",
        "논문이 쓴 평가셋이 무엇인지 재현 가능하게 남긴다.  수치는 manifest 에서 읽는다.",
        "",
        "```text",
        f"{'Population':26} {'N':>6}   근거",
        "─" * 78,
        f"{'PAPER_EVAL_ALL_POS':26} {manifest_count('PAPER_EVAL_ALL_POS'):>6}   "
        "plastic + wood, SHA-dedup union(DEV_EVAL, NEW_EVAL)",
        f"{'PAPER_EVAL_PLASTIC_POS':26} {manifest_count('PAPER_EVAL_PLASTIC_POS'):>6}   "
        "DEV role",
        f"{'PAPER_EVAL_WOOD_POS':26} {manifest_count('PAPER_EVAL_WOOD_POS'):>6}   "
        "CROSS_SHAPE_DEV role",
        f"{'DEV_NEG2689':26} {manifest_count('DEV_NEG2689'):>6}   "
        "negative, 2,688 unique image",
        "```",
        "",
        "```text",
        f"{'Condition':16} {'N':>6}",
        "─" * 24,
    ]
    if "R5_PROPOSED" in results:
        for group in ("Daytime", "Nighttime", "Clean", "Occlusion",
                      "Truncation", "Far", "Low", "Mid", "High"):
            block = metric_block(results["R5_PROPOSED"], group)
            if block.get("N"):
                lines.append(f"{group:16} {block['N']:>6}")
    lines += ["```", "",
              "조건은 서로 중복될 수 있고 합계가 전체 N 이 되지 않는다.",
              "held_out_final 은 false 다 — PAPER_EVAL 은 DEV role 이다.",
              "",
              "adaptation pool 은 평가셋과 분리돼 있다.",
              "",
              "```text",
              "adapt session ∩ eval session   0",
              "adapt image SHA ∩ eval SHA     0",
              "U_MAIN                         1000  (Daytime 500 + Nighttime 500)",
              "```",
              ""]

    # ── A8 cross-domain transfer matrix ─────────────────────────────────
    a8 = [("None", "R0"), ("Daytime", "A8_DAY_ONLY"),
          ("Nighttime", "A8_NIGHT_ONLY"), ("Day + Night", "R5_PROPOSED")]
    if all(k in results for _, k in a8):
        lines += [
            "## A8 — cross-domain transfer matrix",
            "",
            "M2 는 \"그 도메인 데이터로 적응하면 그 도메인이 좋아지는가\" 를 묻는다.",
            "여기서는 **다른 도메인으로 적응해도 좋아지는가** 를 묻는다.",
            "",
            "행 = 적응에 쓴 unlabeled 도메인, 열 = 평가 도메인.  값은 detection rate(↑)와",
            "괄호 안 corner(↓, all-annotated 진단).  모두 Proposed 필터를 쓴 pseudo-label 이다.",
            "",
            "```text",
            f"{'Adaptation':16} {'unique PL':>10} {'Test Daytime':>22} {'Test Nighttime':>22}",
            "─" * 74,
        ]
        pl_counts = {}
        if FUNNEL.exists():
            arms = json.loads(FUNNEL.read_text())["arms"]
            proposed = arms.get("F4_PROPOSED", {})
            pl_counts = {
                "Daytime": proposed.get("daytime_accepted"),
                "Nighttime": proposed.get("nighttime_accepted"),
                "Day + Night": proposed.get("accepted"),
                "None": 0,
            }
        for label, key in a8:
            day = metric_block(results[key], "Daytime")
            night = metric_block(results[key], "Nighttime")
            lines.append(
                f"{label:16} {str(pl_counts.get(label, '—')):>10} "
                f"{number(day.get('detection_rate_iou50')) + ' (' + number(day.get('corner_median_px_all_annotated'), '.2f') + ')':>22} "
                f"{number(night.get('detection_rate_iou50')) + ' (' + number(night.get('corner_median_px_all_annotated'), '.2f') + ')':>22}"
            )
        lines += ["```", "", "### 해석 (규칙은 결과 보기 전에 고정됐다)", "", "```text",
                  "대각선만 개선     target-specific adaptation — 도메인마다 데이터가 필요",
                  "비대각선도 개선   cross-domain transfer — 한 도메인이 다른 도메인도 돕는다",
                  "Day+Night 최선    도메인을 나눌 필요가 없다",
                  "Day+Night 열위    도메인 혼합이 해롭다 (negative transfer)",
                  "```", "",
                  "행마다 unlabeled pool 크기가 다르다 (Day 120 · Night 139 · 합 259).",
                  "그 차이가 결과를 설명할 수 있으므로 unique PL 을 같이 싣는다.", ""]

    # ── §18-B geometry incremental control ──────────────────────────────
    b_random = ["B_CONF_RANDOM_S1", "B_CONF_RANDOM_S2", "B_CONF_RANDOM_S3"]
    b_others = [("confidence-ranking top-N", "B_CONF_TOPN"),
                ("confidence-decile matched", "B_CONF_DECILE")]
    proposed_keys = ["R5_PROPOSED", "R5_PROPOSED_P43", "R5_PROPOSED_P44"]
    if all(k in results for k in b_random + proposed_keys):
        import statistics
        lines += [
            "## A2c — geometry incremental control (confidence pool)",
            "",
            "A2 는 Naive pool 에서 뽑았다.  이건 **confidence 를 이미 통과한** pool(272)에서",
            "Proposed 와 같은 unique 수(259)를 뽑는다.  즉 geometry 가 추가로 걷어낸 13 장의",
            "고유 기여만 분리한다.  A2 와 다른 실험이므로 섞지 않는다.",
            "",
            "```text",
            f"{'Selection from confidence pool':32} {'unique':>7} {'corner↓':>9} "
            f"{'AUROC↑':>9} {'FPR95↓':>9}",
            "─" * 72,
        ]
        def row(label, keys):
            corner = [metric_block(results[k], "ALL")["corner_median_px"] for k in keys]
            auroc = [metric_block(results[k], "ALL")["auroc"] for k in keys]
            fpr = [metric_block(results[k], "ALL")["fpr95"] for k in keys]
            suffix = f" (n={len(keys)})" if len(keys) > 1 else ""
            lines.append(
                f"{label + suffix:32} {'259':>7} "
                f"{statistics.mean(corner):>9.3f} {statistics.mean(auroc):>9.4f} "
                f"{statistics.mean(fpr):>9.4f}")
            return corner, auroc, fpr
        random_stats = row("random matched", b_random)
        for label, key in b_others:
            if key in results:
                row(label, [key])
        proposed_stats = row("Proposed (geometry)", proposed_keys)
        lines += [f"{'Confidence only, all 272':32} {'272':>7} "
                  f"{metric_block(results['R2_CONF'], 'ALL')['corner_median_px']:>9.3f} "
                  f"{metric_block(results['R2_CONF'], 'ALL')['auroc']:>9.4f} "
                  f"{metric_block(results['R2_CONF'], 'ALL')['fpr95']:>9.4f}"]
        lines += ["```", "", "### 판정", "", "```text"]
        for name, index, lower in (("corner", 0, True), ("AUROC", 1, False),
                                   ("FPR95", 2, True)):
            control = random_stats[index]
            ours = proposed_stats[index]
            separated = (max(ours) < min(control)) if lower else (min(ours) > max(control))
            lines.append(
                f"{name:8} random {statistics.mean(control):.4f}  "
                f"Proposed {statistics.mean(ours):.4f}   "
                f"{'구간 분리' if separated else '구간 겹침'}")
        lines += ["```", "",
                  "**confidence 를 통과한 pool 안에서는 무작위로 같은 수를 뽑아도 geometry",
                  "선별과 구분되지 않는다.**  A2 에서 유일하게 분리됐던 AUROC 도 여기서는",
                  "겹친다 — 그 이득은 geometry 가 아니라 confidence 단계에서 온 것이다.",
                  "",
                  "geometry 필터의 **추가** 기여는 이 데이터로 입증되지 않는다.  M4 가 보여준",
                  "선별 능력(통과분 gross 0.072 대 기각분 0.299)은 실재하지만, 그것이",
                  "downstream 성능 이득으로 전이된다는 증거는 없다.", ""]

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
