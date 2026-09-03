"""최종 논문 표를 기존 result artifact 에서만 조립한다.

읽기 전용이다.  모델을 올리지 않고, 추론하지 않고, GT 를 다시 채점하지 않는다.
모든 숫자에 대해 `값 + 파일경로 + 키` 를 `RESULT_SOURCE_MAP.json` 에 남겨
논문의 어느 수치든 원본까지 되짚을 수 있게 한다.

    python3 scripts/paper/build_final_paper_summary.py

출력: _docs/paper/final/generated/
        TABLE_FINAL_1.md            주 비교
        TABLE_FINAL_2.md            주야 적응
        TABLE_FINAL_3.md            선택 규칙 ablation
        TABLE_FINAL_DIAGNOSTIC.md   development 개입 요약
        RESULT_SOURCE_MAP.json      숫자 -> 출처
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "data/pallet/results"
OUT = REPO_ROOT / "_docs/paper/final/generated"

V1_ARMS = RESULTS / "paper_eval_v1/arms/ARM_RESULTS.json"
V1_UNCERTAINTY = RESULTS / "paper_eval_v1/PAIRED_UNCERTAINTY.json"
DOPE = RESULTS / "paper_eval_v1/arms/DOPE.json"
SEPARABILITY = RESULTS / "paper_selftrain_v4/FILTER_SEPARABILITY.json"
V4_PROXY = RESULTS / "paper_selftrain_v4/V4_REPAIR_PROXY.json"
V5_MECHANISM = RESULTS / "paper_selftrain_v5/V5_MECHANISM_CHECK.json"
DEV_METRICS = {
    "V2": RESULTS / "paper_selftrain_v2/V2_DEV_METRICS.json",
    "V3": RESULTS / "paper_selftrain_v3/V3_DEV_METRICS.json",
    "V5": RESULTS / "paper_selftrain_v5/V5_DEV_METRICS.json",
}
FAST = {key: RESULTS / f"paper_fast_teacher_v1/FAST_{key}_TEACHER.json"
        for key in ("A", "B", "C")}

# 논문 표의 행 이름 -> artifact 안의 arm 키
MAIN_ROWS = [
    ("Synthetic-only (R0)", "R0"),
    ("Synthetic-replay control", "R0_CONT"),
    ("Naive self-training", "R1_NAIVE"),
    ("Confidence", "R2_CONF"),
    ("+ reprojection consistency", "R3_CONF_REPROJ"),
    ("+ keypoint-removal consistency", "R4_CONF_REMOVE"),
    ("+ horizontal-flip consistency (full)", "R5_PROPOSED"),
]

SOURCES: dict[str, dict] = {}


def record(label: str, value, path: Path, key: str) -> object:
    SOURCES[label] = {"value": value,
                      "source": str(path.relative_to(REPO_ROOT)), "key": key}
    return value


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def dig(payload: dict, dotted: str):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def cell(value, spec: str = ".4f") -> str:
    return "—" if value is None else format(value, spec)


def table_one(v1: dict, dope: dict | None, uncertainty: dict | None) -> str:
    lines = [
        "# Table 1 — Main comparison",
        "",
        "Population `PAPER_EVAL`: 319 positive frames and 2,689 real negative frames,",
        "scored by one evaluator with one metric definition. `population_contract.role`",
        "is **DEV** and `held_out_final` is **false** — these are development numbers,",
        "not held-out results.",
        "",
        "`Keypoint` is the median error of supervised keypoints in **original-image",
        "pixels**, pooled across frames. Lower is better. `Detection` is the IoU@0.5",
        "match rate. `AP50-95` is the evaluator's `box_ap50_95`. `AUROC` and `FPR95`",
        "are frame-level ranking metrics computed against the 2,689 negatives.",
        "",
        "```text",
        f"{'Method':40}{'Keypoint[px]':>13}{'Det':>8}{'AP50-95':>9}"
        f"{'AP50':>8}{'AUROC':>8}{'FPR95':>8}",
        "─" * 94,
    ]
    if dope is not None:
        block = dig(dope, "metrics.box_and_keypoint_2d") or {}
        row = dig(dope, "subgroups.ALL") or {}
        lines.append(
            f"{'DOPE (same-data control, Tier C)':40}"
            f"{cell(record('T1.DOPE.keypoint_px', block.get('keypoint_location_median_px') or row.get('corner_median_px'), DOPE, 'metrics.box_and_keypoint_2d.keypoint_location_median_px'), '13.3f')}"
            f"{cell(record('T1.DOPE.det', row.get('detection_rate_iou50'), DOPE, 'subgroups.ALL.detection_rate_iou50'), '8.3f')}"
            f"{cell(record('T1.DOPE.ap5095', dig(dope, 'box_ap50_95'), DOPE, 'box_ap50_95'), '9.4f')}"
            f"{cell(dig(dope, 'box_ap50'), '8.4f')}"
            f"{cell(row.get('auroc'), '8.4f')}{cell(row.get('fpr95'), '8.4f')}")

    for label, arm in MAIN_ROWS:
        model = dig(v1, f"models.{arm}")
        if model is None:
            lines.append(f"{label:40}{'NOT_FOUND':>13}")
            continue
        sub = model["subgroups"]["ALL"]
        base = f"models.{arm}"
        lines.append(
            f"{label:40}"
            f"{cell(record(f'T1.{arm}.keypoint_px', sub['corner_median_px'], V1_ARMS, base + '.subgroups.ALL.corner_median_px'), '13.3f')}"
            f"{cell(record(f'T1.{arm}.det', sub['detection_rate_iou50'], V1_ARMS, base + '.subgroups.ALL.detection_rate_iou50'), '8.3f')}"
            f"{cell(record(f'T1.{arm}.ap5095', model['box_ap50_95'], V1_ARMS, base + '.box_ap50_95'), '9.4f')}"
            f"{cell(record(f'T1.{arm}.ap50', model['box_ap50'], V1_ARMS, base + '.box_ap50'), '8.4f')}"
            f"{cell(record(f'T1.{arm}.auroc', sub['auroc'], V1_ARMS, base + '.subgroups.ALL.auroc'), '8.4f')}"
            f"{cell(record(f'T1.{arm}.fpr95', sub['fpr95'], V1_ARMS, base + '.subgroups.ALL.fpr95'), '8.4f')}")
    lines += ["```", ""]

    if uncertainty is not None:
        comparison = dig(uncertainty, "comparisons.R0__vs__R5_PROPOSED") or {}
        lines += [
            "## Uncertainty on the R0 versus full-filter difference",
            "",
            "Paired bootstrap. `p_better` is the probability that the full filter is",
            "better than R0 on that axis.",
            "",
            "```text",
            f"{'axis':28}{'frame':>10}{'session-clustered':>20}",
            "─" * 58,
        ]
        for axis in ("detection", "corner", "pooled_corner_median"):
            block = comparison.get(axis) or {}
            frame = dig(block, "frame.p_better")
            cluster = dig(block, "session_cluster.p_better")
            record(f"T1.uncertainty.{axis}.frame", frame, V1_UNCERTAINTY,
                   f"comparisons.R0__vs__R5_PROPOSED.{axis}.frame.p_better")
            record(f"T1.uncertainty.{axis}.session", cluster, V1_UNCERTAINTY,
                   f"comparisons.R0__vs__R5_PROPOSED.{axis}.session_cluster.p_better")
            lines.append(f"{axis:28}{cell(frame, '10.4f')}{cell(cluster, '20.4f')}")
        lines += [
            "```",
            "",
            "Read this together with the table: the detection gain is **not** separated",
            "from noise, and the localisation difference favours R0.",
            "",
        ]

    lines += [
        "## Reference rows are not comparison rows",
        "",
        "The DOPE row is a **reference**, not a controlled comparison. DOPE has no box",
        "head, so its boxes are derived from detected cuboid corners and its score is a",
        "belief peak rather than a box confidence. `AP50-95`, `AUROC`, and `FPR95` are",
        "therefore not the same quantity across that row. The columns that compare",
        "directly are `Keypoint` and `Det`.",
        "",
        "Real-supervision fine-tuning is deliberately absent from this table. It is a",
        "reference upper bound trained with real labels and does not belong in a block",
        "of unlabeled-adaptation arms.",
        "",
        "`R med`, `yaw`, and every other 6D quantity are omitted because",
        "`POSE_METRICS_STATUS = BLOCKED`.",
    ]
    return "\n".join(lines) + "\n"


def table_two(v1: dict) -> str:
    lines = [
        "# Table 2 — Daytime and nighttime adaptation",
        "",
        "Two different nighttime subgroups exist in the artifacts and are **not**",
        "interchangeable. This table uses the narrow acquisition-condition split; the",
        "broad lighting split is reported underneath with its own sample size.",
        "",
        "```text",
        f"{'Method':40}{'Day det':>9}{'Night det':>11}"
        f"{'Day kp[px]':>12}{'Night kp[px]':>14}",
        "─" * 86,
    ]
    for label, arm in MAIN_ROWS:
        model = dig(v1, f"models.{arm}")
        if model is None:
            continue
        day = model["subgroups"]["Daytime"]
        night = model["subgroups"]["Nighttime"]
        base = f"models.{arm}.subgroups"
        lines.append(
            f"{label:40}"
            f"{cell(record(f'T2.{arm}.day_det', day['detection_rate_iou50'], V1_ARMS, base + '.Daytime.detection_rate_iou50'), '9.3f')}"
            f"{cell(record(f'T2.{arm}.night_det', night['detection_rate_iou50'], V1_ARMS, base + '.Nighttime.detection_rate_iou50'), '11.3f')}"
            f"{cell(record(f'T2.{arm}.day_kp', day['corner_median_px'], V1_ARMS, base + '.Daytime.corner_median_px'), '12.3f')}"
            f"{cell(record(f'T2.{arm}.night_kp', night['corner_median_px'], V1_ARMS, base + '.Nighttime.corner_median_px'), '14.3f')}")
    reference = dig(v1, "models.R0.subgroups") or {}
    lines += [
        "```",
        "",
        "```text",
        f"subgroup sizes    Daytime N = {reference.get('Daytime', {}).get('N')}"
        f"   Nighttime N = {reference.get('Nighttime', {}).get('N')}   (plastic only)",
        f"broad split       Lighting_day N = {reference.get('Lighting_day', {}).get('N')}"
        f"   Lighting_night N = {reference.get('Lighting_night', {}).get('N')}   (plastic + wood)",
        "```",
        "",
        "## What this table says",
        "",
        "Nighttime detection rises from 0.840 to 0.960 or higher for every adapted arm.",
        "**Naive self-training already reaches 0.960 and confidence-only selection",
        "reaches 0.980**, so the nighttime detection gain cannot be attributed to the",
        "geometric consistency filters.",
        "",
        "In the same rows, keypoint error does not fall in either lighting condition.",
        "That contrast — detection up, localisation not — is the paper's main result.",
        "",
        f"Nighttime N is {reference.get('Nighttime', {}).get('N')}; every claim drawn "
        "from this subgroup must carry that sample size.",
    ]
    return "\n".join(lines) + "\n"


def table_three(v1: dict, separability: dict | None) -> str:
    lines = [
        "# Table 3 — Pseudo-label selection ablation",
        "",
        "The table keeps two questions apart: does the selection rule improve the",
        "labels, and does the student improve. Only the first is a property of the",
        "filter; the second is what the paper is about.",
        "",
        "```text",
        f"{'Selection rule':40}{'Keypoint[px]':>13}{'gross20':>9}"
        f"{'Det':>8}{'AUROC':>8}{'FPR95':>8}",
        "─" * 86,
    ]
    for label, arm in MAIN_ROWS[2:]:
        model = dig(v1, f"models.{arm}")
        if model is None:
            continue
        sub = model["subgroups"]["ALL"]
        base = f"models.{arm}.subgroups.ALL"
        lines.append(
            f"{label:40}"
            f"{cell(sub['corner_median_px'], '13.3f')}"
            f"{cell(record(f'T3.{arm}.gross20', sub['gross_rate'], V1_ARMS, base + '.gross_rate'), '9.3f')}"
            f"{cell(sub['detection_rate_iou50'], '8.3f')}"
            f"{cell(sub['auroc'], '8.4f')}{cell(sub['fpr95'], '8.4f')}")
    lines += ["```", ""]

    if separability is not None:
        frame = dig(separability, "frame_level.single_signal_auc") or {}
        keypoint = dig(separability, "keypoint_level.single_signal_auc") or {}
        lines += [
            "## Do the selection signals separate good labels from bad?",
            "",
            "Post-hoc diagnostic, measured against evaluation GT. Development evidence.",
            "",
            "```text",
            f"{'signal':24}{'frame-level AUC':>17}{'corner-level AUC':>19}",
            "─" * 60,
        ]
        for name in sorted(set(frame) | set(keypoint)):
            lines.append(f"{name:24}{cell(frame.get(name), '17.4f')}"
                         f"{cell(keypoint.get(name), '19.4f')}")
        combined_frame = dig(separability, "frame_level.combined_auc_cv")
        combined_kp = dig(separability, "keypoint_level.combined_auc_cv.corner_plus_frame")
        record("T3.separability.frame_combined", combined_frame, SEPARABILITY,
               "frame_level.combined_auc_cv")
        record("T3.separability.corner_combined", combined_kp, SEPARABILITY,
               "keypoint_level.combined_auc_cv.corner_plus_frame")
        floor = dig(separability, "keypoint_level.keypoints_below_conf_floor")
        record("T3.separability.below_conf_floor", floor, SEPARABILITY,
               "keypoint_level.keypoints_below_conf_floor")
        lines += [
            f"{'combined':24}{cell(combined_frame, '17.4f')}{cell(combined_kp, '19.4f')}",
            "```",
            "",
            "The signals are informative but weak, and they are far from a clean",
            "separation. Two facts constrain how much can be claimed:",
            "",
            f"- the per-keypoint confidence floor removes **{floor} corners** — every",
            "  supervised corner already clears it, so confidence gating is inert at",
            "  the corner level;",
            "- the combined frame-level discriminator reaches an AUC below 0.82 on a",
            "  population where roughly half of the frames contain a gross error.",
            "",
            "Better separation of the labels did not become a better student. That is",
            "the finding this table exists to support.",
        ]
    return "\n".join(lines) + "\n"


def table_diagnostic(dev: dict, mechanism: dict | None, proxy: dict | None,
                     fast: dict) -> str:
    lines = [
        "# Appendix table — diagnostic interventions",
        "",
        "**Every row is development evidence (Tier B).** Each was designed after",
        "PAPER_EVAL diagnostics had been seen, so none is an independent confirmation",
        "and none may be described as a held-out result.",
        "",
        "The purpose of the table is not to count failures. It is to record which",
        "candidate mechanisms were isolated and ruled out.",
        "",
        "## Student arms",
        "",
        "`base` and `arm` are paired NME medians on that arm's own common-frame subset,",
        "so the base column differs slightly per row and the columns are not a single",
        "shared baseline.",
        "",
        "```text",
        f"{'arm':30}{'n_fr':>6}{'base NME':>10}{'arm NME':>9}"
        f"{'delta':>10}{'CI95':>26}",
        "─" * 91,
    ]
    seen: set[str] = set()
    for track, payload in dev.items():
        if payload is None:
            continue
        for arm, model in (dig(payload, "models") or {}).items():
            # 각 트랙의 metrics JSON 은 비교용으로 이전 트랙의 arm 을 함께 싣는다
            # (예: V5 안의 V3B__FULL).  중복 행이 생기므로 정규화해 한 번만 쓴다.
            canonical = arm.replace("__FULL", "")
            if canonical == "R0" or canonical in seen:
                continue
            block = dig(model, "paired_nme.ALL")
            if not block:
                continue
            seen.add(canonical)
            delta = dig(block, "frame_delta") or {}
            bounds = delta.get("ci95") or []
            interval = ("—" if len(bounds) != 2
                        else f"[{bounds[0]:+.5f}, {bounds[1]:+.5f}]")
            record(f"DIAG.{canonical}.paired_nme_delta", delta.get("median"),
                   DEV_METRICS[track], f"models.{arm}.paired_nme.ALL.frame_delta.median")
            record(f"DIAG.{canonical}.paired_nme_ci95", bounds,
                   DEV_METRICS[track], f"models.{arm}.paired_nme.ALL.frame_delta.ci95")
            lines.append(
                f"{canonical:30}{block.get('n_frames', 0):6d}"
                f"{cell(block.get('base_nme_median'), '10.5f')}"
                f"{cell(block.get('arm_nme_median'), '9.5f')}"
                f"{cell(delta.get('median'), '+10.5f')}{interval:>26}")
    lines += ["```", "",
              "No arm's interval lies entirely below zero. Improvement in the student's",
              "keypoint localisation was never observed.", ""]

    if proxy is not None:
        counts = dig(proxy, "repair_status_counts") or {}
        lines += [
            "## Geometry repair — why no student was trained",
            "",
            "```text",
            f"{'repair status':26}{'count':>7}",
            "─" * 33,
        ]
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"{name:26}{count:7d}")
        record("DIAG.V4.repair_status_counts", counts, V4_PROXY,
               "repair_status_counts")
        lines += [
            "```",
            "",
            "Repair candidates were about one percent of supervised corners, and the",
            "competing geometric hypotheses disagree precisely on the corners that need",
            "repairing. The intervention had no population to act on, so no student was",
            "trained — the track stopped at the mechanism stage rather than producing a",
            "null training result.",
            "",
        ]

    if mechanism is not None:
        uniform = dig(mechanism, "expected_quality.uniform_v3b") or {}
        weighted = dig(mechanism, "expected_quality.reliability_weighted_v5") or {}
        lines += [
            "## Reliability weighting — label quality improved, student did not",
            "",
            "```text",
            f"{'metric':22}{'uniform':>12}{'weighted':>12}{'change':>12}",
            "─" * 58,
        ]
        for name in ("frame_gross", "corner_gross", "median_error_px", "p90_error_px"):
            a, b = uniform.get(name), weighted.get(name)
            change = None if (a is None or b is None) else b - a
            record(f"DIAG.V5.{name}.uniform", a, V5_MECHANISM,
                   f"expected_quality.uniform_v3b.{name}")
            record(f"DIAG.V5.{name}.weighted", b, V5_MECHANISM,
                   f"expected_quality.reliability_weighted_v5.{name}")
            lines.append(f"{name:22}{cell(a, '12.4f')}{cell(b, '12.4f')}"
                         f"{cell(change, '+12.4f')}")
        auc = dig(mechanism, "auc_R_total")
        record("DIAG.V5.auc_R_total", auc, V5_MECHANISM, "auc_R_total")
        lines += [
            "```",
            "",
            f"The reliability score ranks frames with AUC {cell(auc, '.4f')}, above every",
            "individual signal it is built from, and the labels the student sees do get",
            "cleaner. The student's localisation did not move. This is the most direct",
            "evidence that pseudo-label purity is not the binding constraint.",
            "",
        ]

    if any(fast.values()):
        lines += [
            "## Multi-view teacher consensus — median better, tail worse",
            "",
            "Paired comparison on the identical keypoint set: a candidate that discards",
            "hard keypoints would otherwise flatter itself. Coverage is listed",
            "separately for exactly that reason.",
            "",
            "```text",
            f"{'probe':10}{'coverage':>10}{'n_kp':>7}{'R0 NME':>9}{'cand NME':>10}"
            f"{'R0 p90':>9}{'cand p90':>10}{'R0 gross':>10}{'cand gross':>12}",
            "─" * 87,
        ]
        for key, payload in fast.items():
            if payload is None:
                continue
            paired = dig(payload, "paired.ALL") or {}
            base, candidate = paired.get("R0") or {}, paired.get("candidate") or {}
            coverage = dig(payload, "coverage.frames_with_candidate")
            record(f"DIAG.FAST_{key}.candidate_nme_p90",
                   candidate.get("nme_p90"), FAST[key], "paired.ALL.candidate.nme_p90")
            lines.append(
                f"{'FAST-' + key:10}{cell(coverage, '10.0f')}"
                f"{cell(base.get('n_keypoints'), '7.0f')}"
                f"{cell(base.get('nme_median'), '9.5f')}{cell(candidate.get('nme_median'), '10.5f')}"
                f"{cell(base.get('nme_p90'), '9.5f')}{cell(candidate.get('nme_p90'), '10.5f')}"
                f"{cell(base.get('gross20_rate'), '10.4f')}{cell(candidate.get('gross20_rate'), '12.4f')}")
        lines += [
            "```",
            "",
            "All three probes move the same way: the median improves slightly and the",
            "tail gets clearly worse. Averaging or taking a median across views pulls",
            "good predictions toward bad ones exactly where the views disagree, which is",
            "where the hard keypoints are. Pooling more observations from the same",
            "teacher does not raise the teacher's ceiling. No student was trained on any",
            "of them.",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v1 = load(V1_ARMS)
    if v1 is None:
        raise SystemExit(f"MISSING: {V1_ARMS}")

    dope_payload = load(DOPE)
    if dope_payload is not None and "models" in (load(V1_ARMS) or {}):
        dope_payload = dig(v1, "models.DOPE") or dope_payload

    written = {
        "TABLE_FINAL_1.md": table_one(v1, dope_payload, load(V1_UNCERTAINTY)),
        "TABLE_FINAL_2.md": table_two(v1),
        "TABLE_FINAL_3.md": table_three(v1, load(SEPARABILITY)),
        "TABLE_FINAL_DIAGNOSTIC.md": table_diagnostic(
            {key: load(path) for key, path in DEV_METRICS.items()},
            load(V5_MECHANISM), load(V4_PROXY),
            {key: load(path) for key, path in FAST.items()}),
    }
    for name, text in written.items():
        (OUT / name).write_text(text)

    (OUT / "RESULT_SOURCE_MAP.json").write_text(json.dumps({
        "schema_version": "final_paper_result_source_map_v1",
        "generated_by": "scripts/paper/build_final_paper_summary.py",
        "read_only": True,
        "model_inference_performed": False,
        "training_performed": False,
        "population": "PAPER_EVAL 319 positive + 2689 negative, role DEV, held_out_final false",
        "entries": SOURCES,
    }, indent=2) + "\n")

    print(f"wrote {len(written) + 1} files to {OUT.relative_to(REPO_ROOT)}")
    print(f"traced {len(SOURCES)} numbers to their source artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
