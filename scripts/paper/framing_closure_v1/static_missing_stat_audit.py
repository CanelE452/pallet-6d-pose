"""§29 — frozen artifact 만으로 계산 가능한 논문용 누락 통계.

    python3 scripts/paper/framing_closure_v1/static_missing_stat_audit.py \
        --output-dir data/pallet/results/paper_framing_closure_v1

새 추론·새 학습·새 threshold·새 arm 선택 없음.  이미 저장된 per-frame CSV 와
result JSON 만 읽는다.  계산 불가능한 것은 BLOCKED_MISSING_ARTIFACT 로 남긴다.

여기서 채우는 것:

    G1  ranking 지표(AUROC · FPR95)의 부트스트랩 구간 — claim lock 이
        "no bootstrap interval exists" 라고 적어둔 공백
    G2  main 표 정합성 — 같은 수치가 여러 artifact 에 다르게 적혀 있지 않은가
    G3  population count 정합성
    G4  pose 표 정합성 — arm JSON · by-session · bootstrap 이 서로 맞는가
    G5  pose metric status 정합성 — claim lock 과 closure status 가 어긋나 있다
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ARMS_DIR = REPO_ROOT / "data/pallet/results/paper_eval_v1/arms"
CLOSURE = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
FINAL = REPO_ROOT / "_docs/paper/final"

RANKING_ARMS = ["R0", "R0_CONT", "R1_NAIVE", "R2_CONF", "R3_CONF_REPROJ",
                "R4_CONF_REMOVE", "R5_PROPOSED"]
N_RESAMPLES = 10000
SEED = 20260904


def ranking(positive_scores: np.ndarray, negative_scores: np.ndarray) -> dict:
    """evaluate_arms.ranking 과 **동일한 정의**.  새 정의를 만들지 않는다."""
    if positive_scores.size == 0 or negative_scores.size == 0:
        return {"auroc": None, "fpr95": None}
    labels = np.concatenate([np.ones(positive_scores.size), np.zeros(negative_scores.size)])
    scores = np.concatenate([positive_scores, negative_scores])
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = tps / positive_scores.size
    fpr = fps / negative_scores.size
    auroc = float(np.trapz(tpr, fpr))
    index = min(int(np.searchsorted(tpr, 0.95, side="left")), fpr.size - 1)
    return {"auroc": auroc, "fpr95": float(fpr[index])}


def load_scores(arm: str):
    path = ARMS_DIR / f"{arm}_per_frame.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    def score(row):
        raw = row.get("top_score", "")
        return float(raw) if raw not in ("", None) else 0.0
    positives = [(row["session_id"], score(row)) for row in rows if row["kind"] == "POSITIVE"]
    negatives = np.array([score(row) for row in rows if row["kind"] == "NEGATIVE"], float)
    return positives, negatives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"schema_version": "paper_static_missing_stat_audit_v1",
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "new_inference": 0, "new_training": 0, "new_threshold": 0,
              "new_arm": 0, "new_model_selection": 0}

    # ---------------- G1 ranking uncertainty
    rng = np.random.default_rng(SEED)
    g1 = {"definition": "evaluate_arms.ranking, reused unchanged",
          "why_it_was_missing": "the claim lock records that ranking metrics carry no "
                                "bootstrap interval in the artifacts",
          "negative_session_labels": "ABSENT — every negative row has an empty "
                                     "session_id, so a fully session-clustered interval "
                                     "is NOT computable and remains BLOCKED_MISSING_ARTIFACT",
          "arms": {}}
    cache = {}
    for arm in RANKING_ARMS:
        positives, negatives = load_scores(arm)
        cache[arm] = (positives, negatives)
        pos = np.array([s for _, s in positives], float)
        sessions = np.array([s for s, _ in positives])
        unique = sorted(set(sessions))
        index_of = {s: np.where(sessions == s)[0] for s in unique}
        point = ranking(pos, negatives)

        frame_auroc = np.empty(N_RESAMPLES)
        frame_fpr = np.empty(N_RESAMPLES)
        cluster_auroc = np.empty(N_RESAMPLES)
        cluster_fpr = np.empty(N_RESAMPLES)
        for i in range(N_RESAMPLES):
            p = rng.integers(0, pos.size, pos.size)
            n = rng.integers(0, negatives.size, negatives.size)
            out = ranking(pos[p], negatives[n])
            frame_auroc[i], frame_fpr[i] = out["auroc"], out["fpr95"]
            drawn = rng.integers(0, len(unique), len(unique))
            pick = np.concatenate([index_of[unique[j]] for j in drawn])
            out = ranking(pos[pick], negatives)
            cluster_auroc[i], cluster_fpr[i] = out["auroc"], out["fpr95"]

        stored = json.loads((ARMS_DIR / f"{arm}.json").read_text())
        stored_all = None
        for candidate in (stored.get("metrics", {}).get("box_and_keypoint_2d", {}),):
            if "auroc" in candidate:
                stored_all = candidate
        g1["arms"][arm] = {
            "auroc": point["auroc"], "fpr95": point["fpr95"],
            "auroc_frame_CI95": [float(np.percentile(frame_auroc, 2.5)),
                                 float(np.percentile(frame_auroc, 97.5))],
            "fpr95_frame_CI95": [float(np.percentile(frame_fpr, 2.5)),
                                 float(np.percentile(frame_fpr, 97.5))],
            "auroc_positive_session_cluster_CI95": [
                float(np.percentile(cluster_auroc, 2.5)),
                float(np.percentile(cluster_auroc, 97.5))],
            "fpr95_positive_session_cluster_CI95": [
                float(np.percentile(cluster_fpr, 2.5)),
                float(np.percentile(cluster_fpr, 97.5))],
            "cluster_interval_semantics":
                "positive sessions resampled, the negative pool held fixed. It does NOT "
                "cover negative-side variability and is not a full cluster interval.",
            "reproduces_stored_value": (
                None if stored_all is None else
                {"stored_auroc": stored_all.get("auroc"),
                 "stored_fpr95": stored_all.get("fpr95"),
                 "auroc_abs_diff": (None if stored_all.get("auroc") is None
                                    else abs(stored_all["auroc"] - point["auroc"]))}),
        }

    # paired R5 - R0 on the ranking metrics
    rng = np.random.default_rng(SEED)
    pos_r0 = np.array([s for _, s in cache["R0"][0]], float)
    pos_r5 = np.array([s for _, s in cache["R5_PROPOSED"][0]], float)
    neg_r0, neg_r5 = cache["R0"][1], cache["R5_PROPOSED"][1]
    ids_r0 = [f for f, _ in cache["R0"][0]]
    ids_r5 = [f for f, _ in cache["R5_PROPOSED"][0]]
    paired_possible = ids_r0 == ids_r5 and neg_r0.size == neg_r5.size
    if paired_possible:
        deltas_auroc = np.empty(N_RESAMPLES)
        deltas_fpr = np.empty(N_RESAMPLES)
        for i in range(N_RESAMPLES):
            p = rng.integers(0, pos_r0.size, pos_r0.size)
            n = rng.integers(0, neg_r0.size, neg_r0.size)
            a = ranking(pos_r5[p], neg_r5[n])
            b = ranking(pos_r0[p], neg_r0[n])
            deltas_auroc[i] = a["auroc"] - b["auroc"]
            deltas_fpr[i] = a["fpr95"] - b["fpr95"]
        observed_a = (ranking(pos_r5, neg_r5)["auroc"] - ranking(pos_r0, neg_r0)["auroc"])
        observed_f = (ranking(pos_r5, neg_r5)["fpr95"] - ranking(pos_r0, neg_r0)["fpr95"])
        g1["paired_R5_minus_R0"] = {
            "paired_by_frame_order": True,
            "auroc_delta": observed_a,
            "auroc_frame_CI95": [float(np.percentile(deltas_auroc, 2.5)),
                                 float(np.percentile(deltas_auroc, 97.5))],
            "auroc_excludes_zero": bool(np.percentile(deltas_auroc, 2.5) > 0
                                        or np.percentile(deltas_auroc, 97.5) < 0),
            "fpr95_delta": observed_f,
            "fpr95_frame_CI95": [float(np.percentile(deltas_fpr, 2.5)),
                                 float(np.percentile(deltas_fpr, 97.5))],
            "fpr95_excludes_zero": bool(np.percentile(deltas_fpr, 2.5) > 0
                                        or np.percentile(deltas_fpr, 97.5) < 0),
        }
    else:
        g1["paired_R5_minus_R0"] = "BLOCKED_MISSING_ARTIFACT: frame order differs"
    report["G1_ranking_uncertainty"] = g1

    # ---------------- G2 main table consistency
    summary = json.loads((ARMS_DIR / "ARM_RESULTS.json").read_text())["models"]
    g2 = {"checked": [], "mismatches": []}
    for arm in RANKING_ARMS + ["DOPE"]:
        per_arm = json.loads((ARMS_DIR / f"{arm}.json").read_text())
        block = per_arm["metrics"]["box_and_keypoint_2d"]
        roll = (summary.get(arm) or {}).get("subgroups", {}).get("ALL")
        if roll is None:
            g2["checked"].append({"arm": arm, "status": "NOT_IN_ARM_RESULTS"})
            continue
        pairs = [("box_ap50_95", "box_ap50_95", summary[arm].get("box_ap50_95")),
                 ("box_ap50", "box_ap50", summary[arm].get("box_ap50")),
                 ("keypoint_location_median_px", "corner_median_px", roll.get("corner_median_px")),
                 ("keypoint_location_p90_px", "corner_p90_px", roll.get("corner_p90_px"))]
        for arm_key, _, rolled in pairs:
            mine = block.get(arm_key)
            if mine is None or rolled is None:
                continue
            if abs(mine - rolled) > 1e-6:
                g2["mismatches"].append({"arm": arm, "field": arm_key,
                                         "arm_json": mine, "ARM_RESULTS": rolled})
        g2["checked"].append({"arm": arm, "status": "COMPARED"})
    g2["consistent"] = not g2["mismatches"]
    report["G2_main_table_consistency"] = g2

    # ---------------- G3 population counts
    positives, negatives = cache["R0"]
    manifest = json.loads((CLOSURE / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]
    contract = json.loads((CLOSURE / "POSE_EVAL_OBJECT_CONTRACT.json").read_text())["population"]
    counted = {"positives_in_per_frame_csv": len(positives),
               "negatives_in_per_frame_csv": int(negatives.size),
               "frames_in_axis_manifest": len(manifest),
               "sessions_in_axis_manifest": len({f["session_id"] for f in manifest}),
               "contract_ALL": contract["ALL"],
               "contract_plastic": contract["plastic_standard_110x130x11"],
               "contract_wood": contract["wood_small_80x59x14"]}
    counted["all_agree_on_319"] = bool(
        counted["positives_in_per_frame_csv"] == counted["frames_in_axis_manifest"]
        == counted["contract_ALL"] == 319)
    counted["material_split_agrees"] = bool(
        counted["contract_plastic"] + counted["contract_wood"] == 319)
    report["G3_population_counts"] = counted

    # ---------------- G4 pose table consistency
    by_session = json.loads((CLOSURE / "POSE_EVALUATION_BY_SESSION.json").read_text())
    boot = json.loads((CLOSURE / "POSE_PAIRED_BOOTSTRAP.json").read_text())
    g4 = {"arms": {}, "mismatches": []}
    for arm in RANKING_ARMS:
        table = json.loads((CLOSURE / f"POSE_EVALUATION_{arm}.json").read_text())
        main = table["paths"]["MAIN"]["ALL"]
        rolled = (by_session.get("by_arm", {}).get(arm) or {}).get("ALL")
        g4["arms"][arm] = {"iou3d_median": main["iou3d_median"],
                           "add_sym_auc": main["add_sym_auc"], "n": main["n"]}
        if rolled:
            for key in ("iou3d_median", "add_sym_auc"):
                if key in rolled and abs(rolled[key] - main[key]) > 1e-9:
                    g4["mismatches"].append({"arm": arm, "field": key,
                                             "arm_json": main[key], "by_session": rolled[key]})
    for comparison in boot["comparisons"]:
        arm, reference = comparison["arm"], comparison["reference"]
        for key, stored in (("iou3d", "iou3d_median"), ("add_sym_m", "add_sym_auc")):
            block = comparison["metrics"][key]
            expected = g4["arms"][arm][stored] - g4["arms"][reference][stored]
            if abs(block["observed_difference"] - expected) > 1e-6:
                g4["mismatches"].append({"arm": arm, "field": f"bootstrap {key}",
                                         "bootstrap": block["observed_difference"],
                                         "from_tables": expected})
    session_ci_excluding_zero = sum(
        1 for c in boot["comparisons"] for m in c["metrics"].values()
        if m["session_cluster"]["excludes_zero"])
    frame_ci_excluding_zero = sum(
        1 for c in boot["comparisons"] for m in c["metrics"].values()
        if m["frame_level"]["excludes_zero"])
    g4["metric_blocks"] = sum(len(c["metrics"]) for c in boot["comparisons"])
    g4["session_cluster_CIs_excluding_zero"] = session_ci_excluding_zero
    g4["frame_level_CIs_excluding_zero"] = frame_ci_excluding_zero
    g4["consistent"] = not g4["mismatches"]
    report["G4_pose_table_consistency"] = g4

    # ---------------- G5 pose metric status
    claim = json.loads((FINAL / "PAPER_CLAIM_LOCK.json").read_text())
    status = json.loads((CLOSURE / "POSE_CLOSURE_STATUS.json").read_text())
    report["G5_pose_metric_status_inconsistency"] = {
        "PAPER_CLAIM_LOCK.pose_metrics.POSE_METRICS_STATUS":
            claim["pose_metrics"]["POSE_METRICS_STATUS"],
        "PAPER_CLAIM_LOCK.declared_at_head": claim["declared_at_head"],
        "POSE_CLOSURE_STATUS.POSE_METRICS_STATUS": status["POSE_METRICS_STATUS"],
        "POSE_CLOSURE_STATUS.metrics_still_blocked_field_is_stale_first_pass":
            status.get("metrics_still_blocked"),
        "INCONSISTENT": bool(claim["pose_metrics"]["POSE_METRICS_STATUS"]
                             != status["POSE_METRICS_STATUS"]),
        "what_actually_changed": status["second_pass"]["what_changed"],
        "which_one_is_current": "POSE_CLOSURE_STATUS second pass — the 6D table exists "
                                "and was produced under a GT rule frozen before any "
                                "result was seen",
        "what_does_NOT_change": "can_claim_6d_improvement stays false: all 24 "
                                "session-cluster intervals contain zero",
        "action": "REQUIRES_USER_DECISION — a claim lock is not edited by an "
                  "autonomous pass; the discrepancy is reported, not resolved",
    }

    (out_dir / "PAPER_STATIC_STAT_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")

    print("G1 ranking uncertainty")
    for arm, block in report["G1_ranking_uncertainty"]["arms"].items():
        a, f = block["auroc_frame_CI95"], block["fpr95_frame_CI95"]
        print(f"  {arm:16} AUROC {block['auroc']:.4f} [{a[0]:.4f}, {a[1]:.4f}]   "
              f"FPR95 {block['fpr95']:.4f} [{f[0]:.4f}, {f[1]:.4f}]")
    paired = report["G1_ranking_uncertainty"]["paired_R5_minus_R0"]
    if isinstance(paired, dict):
        print(f"  paired R5-R0 AUROC {paired['auroc_delta']:+.5f} "
              f"{[round(x, 5) for x in paired['auroc_frame_CI95']]} "
              f"excl0={paired['auroc_excludes_zero']}")
        print(f"  paired R5-R0 FPR95 {paired['fpr95_delta']:+.5f} "
              f"{[round(x, 5) for x in paired['fpr95_frame_CI95']]} "
              f"excl0={paired['fpr95_excludes_zero']}")
    print(f"G2 main table consistent  {report['G2_main_table_consistency']['consistent']}")
    print(f"G3 counts agree on 319    {report['G3_population_counts']['all_agree_on_319']}")
    print(f"G4 pose table consistent  {report['G4_pose_table_consistency']['consistent']}  "
          f"session CIs excluding zero "
          f"{report['G4_pose_table_consistency']['session_cluster_CIs_excluding_zero']}"
          f"/{report['G4_pose_table_consistency']['metric_blocks']}")
    print(f"G5 pose status INCONSISTENT "
          f"{report['G5_pose_metric_status_inconsistency']['INCONSISTENT']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
