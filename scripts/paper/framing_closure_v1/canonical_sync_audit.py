"""PAPER CANONICAL SYNC — stale 문자열 감사 · 숫자 출처 추적 · 정적 테스트.

    python3 scripts/paper/framing_closure_v1/canonical_sync_audit.py

새 추론 0 · 새 학습 0 · 새 metric 0.  기존 artifact 를 읽고 문서와 대조만 한다.

출력
    _docs/paper/final/PAPER_CANONICAL_SYNC_AUDIT.json
    _docs/paper/final/PAPER_CANONICAL_SYNC_TESTS.json
    _docs/paper/final/PAPER_CANONICAL_NUMBER_SOURCES.json
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FINAL = ROOT / "_docs/paper/final"
CLOSURE = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
ARMS = ROOT / "data/pallet/results/paper_eval_v1/arms"
FRAMING = ROOT / "data/pallet/results/paper_framing_closure_v1"

# 활성 paper namespace — 여기만 고친다
ACTIVE = [FINAL, FRAMING, ROOT / "_docs/paper/EXPERIMENTS.md",
          ROOT / "_docs/paper/README.md", ROOT / "scripts/paper"]
# 역사 기록 — 검색은 하되 자동 수정하지 않는다
HISTORICAL = [ROOT / "_docs/paper/pose_metric_closure_v1",
              ROOT / "data/pallet/results/OVERNIGHT_6D_DECISION_20260904.md",
              ROOT / "data/pallet/results/paper_fast6d_screen_v1",
              ROOT / "data/pallet/results/paper_fast6d_screen_v1b",
              ROOT / "_docs/history"]

STALE = [
    ("POSE_METRICS_STATUS = BLOCKED", "pose layer is REPORTABLE"),
    ("POSE_METRICS_STATUS\": \"BLOCKED\"", "pose layer is REPORTABLE"),
    ("POSE_METRICS_BLOCKED", "pose layer is REPORTABLE"),
    ("pose columns are absent", "6D lives in its own table"),
    ("pose metrics removed", "6D lives in its own table"),
    ("site-matched approval pending", "the small arm was already evaluated"),
    ("site-matched experiment still open", "the small arm was already evaluated"),
    ("wood pose unresolved", "wood 125 is included"),
    ("wood awaiting decision", "wood 125 is included"),
    ("held-out confirmation", "no confirmation population exists"),
    ("metrology-grade", "geometry-reconstructed 6D reference pose"),
]
# 금지어를 '쓰지 말라' 고 적은 줄은 false positive
PROHIBITION = re.compile(
    r"never|금지|forbidden|not\s|use\s|대신|instead|do not|don't|아니다|말 것|쓰지",
    re.IGNORECASE)


# 이 감사가 스스로 만든 출력 — 자기 보고서를 다시 훑으면 무한히 자기 자신을 찾는다
SELF_OUTPUT = {"PAPER_CANONICAL_SYNC_AUDIT.json",
               "PAPER_CANONICAL_SYNC_TESTS.json",
               "PAPER_CANONICAL_NUMBER_SOURCES.json",
               "PAPER_CANONICAL_SYNC_REPORT_20260904.md"}   # 고친 내용을 인용한다


def walk(targets, skip_self=False):
    for t in targets:
        if t.is_file():
            if skip_self and t.name in SELF_OUTPUT:
                continue
            yield t
        elif t.is_dir():
            for f in sorted(t.rglob("*")):
                if f.is_file() and f.suffix in (".md", ".json", ".py") \
                        and "__pycache__" not in f.parts:
                    if skip_self and f.name in SELF_OUTPUT:
                        continue
                    yield f


# 의도적으로 과거 상태를 적어 둔 자리 — 지우면 오히려 기록이 사라진다
HISTORICAL_MARKER = re.compile(
    r"historical|first[_ ]pass|original_block|lock 시점|옛 판|preserved|amended|과거",
    re.IGNORECASE)


def scan(targets, active: bool):
    hits = []
    for f in walk(targets, skip_self=True):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        lines = text.splitlines()
        for line_no, line in enumerate(lines, 1):
            for needle, why in STALE:
                if needle.lower() in line.lower():
                    context = "\n".join(lines[max(0, line_no - 9):line_no + 2])
                    hits.append({"file": str(f.relative_to(ROOT)), "line": line_no,
                                 "pattern": needle, "text": line.strip()[:160],
                                 "why_stale": why,
                                 "active": active,
                                 "looks_like_a_prohibition": bool(PROHIBITION.search(line)),
                                 "in_a_historical_block": bool(HISTORICAL_MARKER.search(context))})
    return hits


def classify(hit):
    path = Path(hit["file"]).name
    if not hit["active"]:
        return "HISTORICAL_KEEP"
    if hit["looks_like_a_prohibition"]:
        return "FALSE_POSITIVE"
    if path.startswith("test_"):
        return "FALSE_POSITIVE"          # 테스트 픽스처지 주장이 아니다
    if path in ("PAPER_CANONICAL_SYNC_AUDIT.json", "canonical_sync_audit.py",
                "static_missing_stat_audit.py"):
        return "FALSE_POSITIVE"          # 감사 도구 자신의 패턴 목록
    if path == "PAPER_STATIC_STAT_AUDIT.json":
        return "HISTORICAL_KEEP"         # 날짜 박힌 감사 결과물
    if hit["in_a_historical_block"]:
        return "HISTORICAL_KEEP"         # 일부러 남긴 과거 상태
    return "NEEDS_USER"


def main() -> int:
    stamp = datetime.now(timezone.utc).isoformat()

    # ---------------------------------------------------------------- audit
    hits = scan(ACTIVE, True) + scan(HISTORICAL, False)
    # 2026-09-04 sync 에서 실제로 고친 자리.  고친 뒤에는 패턴이 남지 않으므로
    # 스캔으로는 잡히지 않는다 — 무엇을 고쳤는지 기록으로 남긴다.
    ACTIVE_FIXED = [
        {"file": "_docs/paper/final/PAPER_CLAIM_LOCK.json",
         "was": "pose_metrics.POSE_METRICS_STATUS = BLOCKED",
         "now": "REPORTABLE + can_claim_6d_improvement false; first pass preserved "
                "under pose_metrics.historical_first_pass"},
        {"file": "_docs/paper/final/PAPER_CLAIM_LOCK.md",
         "was": "Pose metrics section declared the layer blocked",
         "now": "REPORTABLE, with the historical block and the permitted/forbidden "
                "sentence lists rewritten"},
        {"file": "_docs/paper/final/LIMITATIONS.md",
         "was": "3. Pose metrics are blocked / 8. Ranking differences carry no interval",
         "now": "3. the reference is reconstructed, not sensor GT; 8. frame-level "
                "ranking interval exists, session-clustered still unavailable; "
                "8b and 8c added"},
        {"file": "_docs/paper/final/METRIC_NAMING_LOCK.md",
         "was": "6D pose layer BLOCKED, columns removed",
         "now": "REPORTABLE with fixed reader-facing names and the reference wording"},
        {"file": "_docs/paper/final/DISCUSSION.md",
         "was": "pose metrics blocked",
         "now": "pose reported, no pose improvement claimed"},
        {"file": "_docs/paper/final/ABSTRACT_DRAFT.md",
         "was": "6D listed under deliberate omissions because BLOCKED",
         "now": "6D reportable but not resolved; the abstract states the absence of a gain"},
        {"file": "_docs/paper/final/FINAL_ABSTRACT_RESULT_SLOTS.md",
         "was": "seven BLOCKED pose slots",
         "now": "slots filled with R0/R5 values, and an explicit note that no "
                "improvement slot is fillable"},
        {"file": "_docs/paper/final/INTRODUCTION_STORY.md",
         "was": "the downstream 6D stage is not used for quantitative claims",
         "now": "three measured layers, with the no-improvement rule stated"},
        {"file": "_docs/paper/final/METHOD_OUTLINE.md",
         "was": "6D pose layer BLOCKED; quantitative claims stop at 2D",
         "now": "6D pose layer REPORTABLE; claims span three layers"},
        {"file": "_docs/paper/final/TITLE_CANDIDATES.md",
         "was": "pose metrics are blocked",
         "now": "6D measured but no resolved improvement"},
        {"file": "_docs/paper/final/RESULTS_STORY.md",
         "was": "no 6D pose quantity appears anywhere",
         "now": "6D appears in its own table, never as an improvement claim"},
        {"file": "_docs/paper/final/CONTRIBUTIONS.md",
         "was": "not claimed: a method that improves 6D pose accuracy",
         "now": "same, with the reason spelled out"},
        {"file": "_docs/paper/final/FIGURE_PLAN.md",
         "was": "Figure 2 was a two-axis trade-off with an optional ranking panel",
         "now": "three-panel hierarchy: detection/ranking, 2D localisation, downstream 6D"},
        {"file": "_docs/paper/EXPERIMENTS.md",
         "was": "pose columns permanently empty because BLOCKED",
         "now": "6D lives in the separate pose table; the old diagnosis is explained"},
        {"file": "scripts/paper/build_final_paper_summary.py",
         "was": "Table 1 caption said pose columns are absent because BLOCKED",
         "now": "caption points at the separate pose table; regenerated, 265 numbers unchanged"},
        {"file": "scripts/paper/pose_metric_closure_v1/build_pose_tables.py",
         "was": "header called the reference 'ground truth'",
         "now": "geometry-reconstructed 6D reference pose, plus the 318/319 population "
                "note and the 0/24 statement; regenerated, all numeric rows unchanged"},
        {"file": "scripts/paper/build_experiment_tables.py",
         "was": "caption said R med and yaw are blank because BLOCKED",
         "now": "points at the separate pose table. Its output lives under _docs/archive, "
                "which is not regenerated by this sync"},
        {"file": "data/pallet/results/paper_framing_closure_v1/PAPER_MAIN_CLAIMS.md",
         "was": "line fusion gets worse as lambda grows",
         "now": "corrected — seed and lambda are confounded (seed1 lambda 3.0, "
                "seed2 lambda 1.0); only the per-seed statement is supportable"},
        {"file": "data/pallet/results/paper_framing_closure_v1/PAPER_NO_CLAIMS.md",
         "was": "same lambda claim, plus a short forbidden list",
         "now": "corrected, and the temporal/depth/confirmation prohibitions added"},
        {"file": "data/pallet/results/paper_framing_closure_v1/PAPER_EVIDENCE_TIER.md",
         "was": "site-matched self-training not yet run, awaiting approval; 6D placement undecided",
         "now": "small arm already evaluated; full-site scaling NOT_RUN_AND_NOT_PLANNED; "
                "6D table is MAIN"},
        {"file": "data/pallet/results/paper_framing_closure_v1/PAPER_FRAMING_DECISION.md",
         "was": "five open user decisions D1-D5",
         "now": "all five resolved; only two BLOCKED_MISSING_ARTIFACT items remain, "
                "both properties of the data"},
        {"file": "data/pallet/results/paper_framing_closure_v1/PAPER_REVIEWER_GAP_AUDIT.md",
         "was": "section 1 was an open conflict requiring a user decision",
         "now": "RESOLVED, with the full list of files synchronised"},
        {"file": "data/pallet/results/paper_framing_closure_v1/PAPER_TABLE_PLAN.md",
         "was": "6D table placement undecided",
         "now": "pose table is a main table, kept separate from the 2D/detection table"},
        {"file": "data/pallet/results/paper_framing_closure_v1/PAPER_FIGURE_PLAN.md",
         "was": "6D panel conditional on a decision",
         "now": "confirmed and reflected in _docs/paper/final/FIGURE_PLAN.md"},
    ]
    buckets = {"ACTIVE_FIXED": ACTIVE_FIXED, "HISTORICAL_KEEP": [],
               "FALSE_POSITIVE": [], "NEEDS_USER": []}
    for hit in hits:
        buckets[classify(hit)].append(hit)
    audit = {
        "schema_version": "paper_canonical_sync_audit_v1",
        "generated_utc": stamp,
        "rule": "historical, archived and result artifacts are searched but never "
                "auto-edited. Only active paper-facing documents were changed.",
        "prohibition_lines_are_not_stale": "a line that says never to use a phrase is "
                                           "the fix, not the defect",
        "counts": {k: len(v) for k, v in buckets.items()},
        "buckets": buckets,
    }
    (FINAL / "PAPER_CANONICAL_SYNC_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    # ------------------------------------------------------- number sources
    six = {a: json.loads((CLOSURE / f"POSE_EVALUATION_{a}.json").read_text())
           for a in ("R0", "R0_CONT", "R1_NAIVE", "R2_CONF", "R3_CONF_REPROJ",
                     "R4_CONF_REMOVE", "R5_PROPOSED")}
    boot = json.loads((CLOSURE / "POSE_PAIRED_BOOTSTRAP.json").read_text())
    stat = json.loads((FRAMING / "PAPER_STATIC_STAT_AUDIT.json").read_text())
    site = json.loads((CLOSURE / "SITE_A_ARM_EVALUATION.json").read_text())

    def two_d(arm):
        return json.loads((ARMS / f"{arm}.json").read_text())["metrics"]["box_and_keypoint_2d"]

    sources = {
        "schema_version": "paper_canonical_number_sources_v1",
        "generated_utc": stamp,
        "note": "numbers quoted in paper-facing prose, traced to file and key. "
                "Complements generated/RESULT_SOURCE_MAP.json, which covers the "
                "2D/detection tables.",
        "pose_6d": {
            arm: {"value": {k: six[arm]["paths"]["MAIN"]["ALL"][k]
                            for k in ("n", "axis_accuracy", "rotation_median_deg",
                                      "yaw_median_deg", "translation_median_cm",
                                      "iou3d_median", "add_sym_auc")},
                  "file": f"data/pallet/results/paper_pose_metric_closure_v1/POSE_EVALUATION_{arm}.json",
                  "key": "paths.MAIN.ALL"}
            for arm in six},
        "pose_bootstrap": {
            "file": "data/pallet/results/paper_pose_metric_closure_v1/POSE_PAIRED_BOOTSTRAP.json",
            "comparisons": len(boot["comparisons"]),
            "metric_blocks": sum(len(c["metrics"]) for c in boot["comparisons"]),
            "session_cluster_excluding_zero": sum(
                1 for c in boot["comparisons"] for m in c["metrics"].values()
                if m["session_cluster"]["excludes_zero"]),
            "session_cluster_excluding_zero_in_improvement_direction": sum(
                1 for c in boot["comparisons"] for m in c["metrics"].values()
                if m["session_cluster"]["excludes_zero"]
                and ((m["better"] == "higher" and m["observed_difference"] > 0)
                     or (m["better"] == "lower" and m["observed_difference"] < 0))),
            "frame_level_excluding_zero": sum(
                1 for c in boot["comparisons"] for m in c["metrics"].values()
                if m["frame_level"]["excludes_zero"]),
        },
        "two_d": {arm: {"value": {k: two_d(arm)[k] for k in
                                  ("box_ap50_95", "box_ap50",
                                   "keypoint_location_median_px",
                                   "keypoint_location_p90_px")},
                        "file": f"data/pallet/results/paper_eval_v1/arms/{arm}.json",
                        "key": "metrics.box_and_keypoint_2d"}
                  for arm in six},
        "ranking": {
            "file": "data/pallet/results/paper_framing_closure_v1/PAPER_STATIC_STAT_AUDIT.json",
            "key": "G1_ranking_uncertainty",
            "R0": {k: stat["G1_ranking_uncertainty"]["arms"]["R0"][k]
                   for k in ("auroc", "fpr95", "auroc_frame_CI95", "fpr95_frame_CI95")},
            "R5_PROPOSED": {k: stat["G1_ranking_uncertainty"]["arms"]["R5_PROPOSED"][k]
                            for k in ("auroc", "fpr95", "auroc_frame_CI95", "fpr95_frame_CI95")},
            "paired_R5_minus_R0": stat["G1_ranking_uncertainty"]["paired_R5_minus_R0"],
            "session_cluster_interval": "BLOCKED_MISSING_ARTIFACT — negative rows carry no session_id",
        },
        "site_matched": {
            "file": "data/pallet/results/paper_pose_metric_closure_v1/SITE_A_ARM_EVALUATION.json",
            "n_frames": site["n_frames"],
            "clusters": site["contrasts"][0]["clusters"],
            "summaries": {k: {m: v[m] for m in ("kp_median_px", "iou3d_median",
                                                "add_sym_auc", "axis_accuracy")}
                          for k, v in site["summaries"].items()},
            "A8_vs_R0": next(
                {k: {"delta": m["observed_difference"],
                     "cluster_CI95": [m["cluster_ci_low"], m["cluster_ci_high"]],
                     "excludes_zero": bool(m["cluster_ci_high"] < 0 or m["cluster_ci_low"] > 0)}
                 for k, m in c["metrics"].items()}
                for c in site["contrasts"]
                if c["arm"] == "A8_DAY_ONLY" and c["reference"] == "R0"),
        },
        "population": {
            "file": "data/pallet/results/paper_pose_metric_closure_v1/POSE_EVAL_OBJECT_CONTRACT.json",
            "key": "population",
            "value": json.loads((CLOSURE / "POSE_EVAL_OBJECT_CONTRACT.json").read_text())["population"],
        },
    }
    (FINAL / "PAPER_CANONICAL_NUMBER_SOURCES.json").write_text(
        json.dumps(sources, indent=2, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- tests
    tests = []

    def check(name, ok, detail):
        tests.append({"test": name, "pass": bool(ok), "detail": detail})

    # T1 every modified JSON parses
    bad = []
    for f in walk([FINAL, FRAMING]):
        if f.suffix == ".json":
            try:
                json.loads(f.read_text())
            except Exception as error:
                bad.append(f"{f.name}: {error}")
    check("all_paper_json_parses", not bad, bad or "all parse")

    # T2 claim lock canonical state
    lock = json.loads((FINAL / "PAPER_CLAIM_LOCK.json").read_text())
    pose = lock["pose_metrics"]
    check("claim_lock_pose_reportable", pose["POSE_METRICS_STATUS"] == "REPORTABLE",
          pose["POSE_METRICS_STATUS"])
    check("claim_lock_no_6d_improvement", pose["can_claim_6d_improvement"] is False,
          pose["can_claim_6d_improvement"])
    check("claim_lock_history_preserved",
          pose.get("historical_first_pass", {}).get("status") == "BLOCKED",
          "historical_first_pass.status")

    # T3 population arithmetic
    pop = sources["population"]["value"]
    check("population_319_equals_194_plus_125",
          pop["ALL"] == 319 and pop["plastic_standard_110x130x11"] == 194
          and pop["wood_small_80x59x14"] == 125 and 194 + 125 == pop["ALL"], pop)

    # T4 generated pose table matches the source JSON at printed precision
    table = (FINAL / "generated/TABLE_FINAL_POSE.md").read_text()
    mismatch = []
    for label, arm in (("Synthetic-only (R0)", "R0"),
                       ("Full consistency self-training", "R5_PROPOSED")):
        block = six[arm]["paths"]["MAIN"]["ALL"]
        row = next((l for l in table.splitlines() if l.startswith(label)), None)
        if row is None:
            mismatch.append(f"{label}: row missing"); continue
        for value in (block["iou3d_median"], block["add_sym_auc"],
                      block["rotation_median_deg"]):
            if f"{value:.3f}" not in row and f"{value:.2f}" not in row:
                mismatch.append(f"{label}: {value} not printed in row")
    check("pose_table_matches_source_json", not mismatch, mismatch or "R0 and R5 agree")

    # T5 no session-cluster resolved improvement
    check("zero_session_cluster_resolved_6d_improvements",
          sources["pose_bootstrap"]["session_cluster_excluding_zero_in_improvement_direction"] == 0,
          sources["pose_bootstrap"])

    # T6 ranking values consistent with the static audit
    paired = sources["ranking"]["paired_R5_minus_R0"]
    check("ranking_auroc_delta_positive_and_excludes_zero",
          paired["auroc_delta"] > 0 and paired["auroc_excludes_zero"] is True,
          {"delta": paired["auroc_delta"], "ci": paired["auroc_frame_CI95"]})
    check("ranking_fpr95_delta_contains_zero",
          paired["fpr95_excludes_zero"] is False,
          {"delta": paired["fpr95_delta"], "ci": paired["fpr95_frame_CI95"]})

    # T7 site-matched shows no resolved improvement
    resolved = [k for k, v in sources["site_matched"]["A8_vs_R0"].items() if v["excludes_zero"]]
    check("site_matched_no_resolved_improvement", not resolved,
          resolved or "all four cluster intervals contain zero")

    # T8 referenced paths exist
    missing = [s for s in (
        "data/pallet/results/paper_pose_metric_closure_v1/POSE_EVALUATION_R0.json",
        "data/pallet/results/paper_pose_metric_closure_v1/POSE_PAIRED_BOOTSTRAP.json",
        "data/pallet/results/paper_pose_metric_closure_v1/POSE_EVAL_OBJECT_CONTRACT.json",
        "data/pallet/results/paper_framing_closure_v1/PAPER_STATIC_STAT_AUDIT.json",
        "_docs/paper/final/generated/TABLE_FINAL_POSE.md",
        "_docs/paper/final/PAPER_CANONICAL_SYNC_20260904.md",
        "_docs/paper/pose_metric_closure_v1/SITE_A_ARM_EVALUATION.md",
    ) if not (ROOT / s).exists()]
    check("referenced_paths_exist", not missing, missing or "all present")

    # T9 no unresolved stale phrase in active documents
    check("no_active_stale_phrase", len(buckets["NEEDS_USER"]) == 0,
          buckets["NEEDS_USER"] or "none")

    report = {
        "schema_version": "paper_canonical_sync_tests_v1",
        "generated_utc": stamp,
        "new_inference": 0, "new_training": 0, "new_metric_definition": 0,
        "tests": tests,
        "passed": sum(1 for t in tests if t["pass"]),
        "failed": sum(1 for t in tests if not t["pass"]),
        "ALL_PASS": all(t["pass"] for t in tests),
    }
    (FINAL / "PAPER_CANONICAL_SYNC_TESTS.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("audit counts:", audit["counts"])
    for t in tests:
        print(f"  [{'PASS' if t['pass'] else 'FAIL'}] {t['test']}")
    print(f"\nALL_PASS = {report['ALL_PASS']}  ({report['passed']}/{len(tests)})")
    return 0 if report["ALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
