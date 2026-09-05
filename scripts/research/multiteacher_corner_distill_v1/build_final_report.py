"""최종 산출물 — 모든 게이트 결과를 하나의 JSON + 표로 모은다.

숫자는 전부 게이트 artifact 에서 읽는다. 산문에서 복사하지 않는다.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M

REF = "T0_R0_YOLO26N_G38LEGACY"


def read(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, w=8, d=3):
    return f"{'-':>{w}}" if v is None else f"{v:{w}.{d}f}"


def main() -> int:
    lock = read(M.METHOD_LOCK_PATH)
    registry = read(M.TRACK / "TEACHER_REGISTRY.json")
    ga = read(M.GATE_A / "GATE_A_RESULT.json")
    gb = read(M.GATE_B / "GATE_B_RESULT.json")
    gbd = read(M.GATE_B / "GATE_B_DENSITY_CONTROL.json")
    gc2 = read(M.GATE_C / "GATE_C_2D.json")
    gcv = read(M.GATE_C / "GATE_C_VERDICT.json")
    gct = read(M.GATE_C / "CONSENSUS_TRANSFER_DIAGNOSTIC.json")
    gd = read(M.GATE_D / "DISTILL_TARGET_QUALITY.json")
    gdu = read(M.GATE_D / "DISTILL_TARGET_QUALITY_UNCERTAINTY.json")
    ge = read(M.GATE_E / "GATE_E_RESULT.json")
    r0_6d = read(M.GATE_A / f"POSE_EVALUATION_{REF}.json")

    headroom = ga["verdict"]["MULTI_TEACHER_HEADROOM"] if ga else "NOT_RUN"
    local = gb["verdict"]["LOCAL_CORNER_HEADROOM"] if gb else "NOT_RUN"
    specialist = gcv["REAL_LOCAL_SPECIALIST"] if gcv else "NOT_RUN"
    target_quality = gd["DISTILL_TARGET_QUALITY"] if gd else "NOT_RUN"
    student = "NOT_RUN"
    domain = ge["TARGET_BIAS_SIGNAL"] if ge else "NOT_RUN"
    adapter = (ge["adapter_admission"] if ge else "NOT_RUN")

    if headroom != "STRONG":
        case = "CASE_A"
    elif specialist == "GO" and target_quality == "PASS":
        case = "CASE_D"
    elif specialist == "GO":
        case = "CASE_C"
    elif target_quality == "FAIL":
        case = "CASE_B"
    else:
        case = "CASE_F"
    case_names = {
        "CASE_A": "NO_TEACHER_COMPLEMENTARITY",
        "CASE_B": "ORACLE_COMPLEMENTARITY_ONLY",
        "CASE_C": "LOCAL_SPECIALIST_SIGNAL_ONLY",
        "CASE_D": "MULTITEACHER_STUDENT_POSITIVE",
        "CASE_E": "TARGET_ADAPTER_ADDITIONAL_GAIN",
        "CASE_F": "NO_PROMOTABLE_SIGNAL"}

    result = {
        "schema_version": "mtcd_final_result_v1",
        "track": "MULTI_TEACHER_CORNER_DISTILL_V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method_lock_sha256": M.sha256_file(M.METHOD_LOCK_PATH),
        "teacher_registry_sha256": M.sha256_file(M.TRACK / "TEACHER_REGISTRY.json"),
        "population": {"DEV_EVAL": 319, "role": "DEV — 반복 사용된 개발 모집단",
                       "INDEPENDENT_CONFIRMATION_AVAILABLE": "NO"},
        "interpretation_status": "DEVELOPMENT_METHOD_SIGNAL",
        "verdicts": {
            "MULTI_TEACHER_HEADROOM": headroom,
            "LOCAL_CORNER_HEADROOM": local,
            "CLASSICAL_LOCAL_SELECTOR_SIGNAL": gb["verdict"]["CLASSICAL_LOCAL_SELECTOR_SIGNAL"] if gb else None,
            "REAL_LOCAL_SPECIALIST": specialist,
            "DISTILL_TARGET_QUALITY": target_quality,
            "BEST_STUDENT": student,
            "TARGET_BIAS_SIGNAL": domain,
            "TARGET_ADAPTER": adapter,
            "FINAL_CASE": f"{case} {case_names[case]}",
        },
        "counts": {
            "teachers": registry["counts"]["n_teachers"],
            "training_runs_total": sum(1 for a in ("C0", "C1")
                                       if (M.GATE_C / f"TRAIN_{a}.json").exists()),
            "gpu_inference_runs": 7 + 7 + (2 if gc2 else 0),
        },
        "gate_a": ga["verdict"] if ga else None,
        "gate_b": gb["verdict"] if gb else None,
        "gate_b_density_control": {k: gbd[k] for k in
                                   ("coverage_classical", "coverage_uniform_random_same_count",
                                    "lift")} if gbd else None,
        "consensus_transfer": gct["pass_rate_at_tau"] if gct else None,
        "gate_c": gcv if gcv else None,
        "gate_d": {"coverage": gd["coverage"], "checks": gd["checks_on_usable_subset"],
                   "uncertainty": gdu} if gd else None,
        "gate_e": ge,
        "baseline_r0_6d": r0_6d["paths"]["MAIN"]["ALL"] if r0_6d else None,
    }
    out = M.FINAL / "MULTITEACHER_FINAL_RESULT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float) + "\n")

    # ---------------------------------------------------------------- 표 ----
    lines = []
    A = lines.append
    A("# MULTI_TEACHER_CORNER_DISTILL_V1 — 최종 보고")
    A("")
    A("모집단 DEV_EVAL 319 (PAPER_EVAL positive). 역할은 DEV 다 — 반복 사용됐고 held-out 이 아니다.")
    A("독립 확증 모집단이 저장소에 없어(`INDEPENDENT_CONFIRMATION_AVAILABLE = NO`)")
    A("아래 어떤 수치도 held-out / confirmed / final / state-of-the-art 가 아니다.")
    A("상태는 **DEVELOPMENT_METHOD_SIGNAL** 이다.")
    A("")
    A("## 판정")
    A("")
    A("```")
    for k, v in result["verdicts"].items():
        A(f"{k:34}{v}")
    A("```")
    A("")

    if ga:
        A("## [TEACHERS]")
        A("")
        A("```")
        A(f"{'teacher':26}{'축':26}{'검출':>6}{'median':>9}{'p90':>9}{'gross20':>9}")
        A("─" * 85)
        for t, b in ga["standalone"].items():
            v = b["visible_corners"]
            A(f"{t:26}{b['axis']:26}{b['coverage_supervised']:6.3f}"
              f"{fmt(v['median_px'],9,2)}{fmt(v['p90_px'],9,2)}{fmt(v['gross20'],9,3)}")
        A("```")
        A("")
        A("## [HEADROOM]  visible 코너")
        A("")
        A("```")
        A(f"{'arm':30}{'median':>9}{'p90':>9}{'gross20':>9}{'gross40':>9}")
        A("─" * 66)
        f = ga["fusion_controls"]
        for key, label in (("F0_r0_only", "best single = R0"),
                           ("F1_component_median", "F1 좌표 성분별 median"),
                           ("F2_geometric_medoid", "F2 geometric medoid")):
            v = f[key]["visible"]
            A(f"{label:30}{fmt(v['median_px'],9,2)}{fmt(v['p90_px'],9,2)}"
              f"{fmt(v['gross20'],9,3)}{fmt(v['gross40'],9,3)}")
        v = ga["oracle_all_teachers"]["visible"]
        A(f"{'ORACLE per-keypoint (배포불가)':30}{fmt(v['median_px'],9,2)}{fmt(v['p90_px'],9,2)}"
          f"{fmt(v['gross20'],9,3)}{fmt(v['gross40'],9,3)}")
        A("")
        A(f"F3 불확실성 가중 = BLOCKED_NOT_COMPARABLE (SIGMA_STATUS = DIAGNOSTIC_ONLY)")
        A(f"R0 gross20 구제율 (어느 교사라도 <=10px) = "
          f"{ga['conditional_rescue']['ANY_OTHER_TEACHER']['rescue_le10px']:.3f}")
        A("```")
        A("")

    if gb:
        A("## [CORNER EVIDENCE]  visible 코너, 반경 12px")
        A("")
        A("```")
        p = gb["PRIMARY_visible"]
        A(f"{'arm':30}{'median':>9}{'p90':>9}{'gross20':>9}")
        A("─" * 57)
        for key, label in (("R0", "R0"), ("ORACLE_CANDIDATE", "oracle 후보"),
                           ("PREDICTION_ONLY_SELECTOR", "prediction-only 선택기")):
            v = p[key]
            A(f"{label:30}{fmt(v['median_px'],9,2)}{fmt(v['p90_px'],9,2)}{fmt(v['gross20'],9,3)}")
        A("")
        c = p["candidate_coverage"]
        A(f"GT 5px 이내 후보 존재율   {c['within_5px']:.3f}")
        if gbd:
            A(f"  같은 개수 균등난수 대비 lift   3px {gbd['lift']['3']:+.3f}  "
              f"5px {gbd['lift']['5']:+.3f}  10px {gbd['lift']['10']:+.3f}")
        A("```")
        A("")

    if gcv:
        A("## [LOCAL SPECIALIST]  visible 코너")
        A("")
        A("```")
        A(f"{'arm':30}{'median':>9}{'p90':>9}{'gross20':>9}{'IoU3D':>9}{'ADDauc':>9}")
        A("─" * 75)
        b = gcv["baseline"]
        A(f"{'R0':30}{fmt(b['2d_visible']['median_px'],9,2)}{fmt(b['2d_visible']['p90_px'],9,2)}"
          f"{fmt(b['2d_visible']['gross20'],9,3)}{fmt(b['6d_ALL']['iou3d_median'],9,3)}"
          f"{fmt(b['6d_ALL']['add_sym_auc'],9,3)}")
        for arm, label in (("C0", "C0 SYN_LOCAL"), ("C1", "C1 SYN_PLUS_REAL_SOFT")):
            a = gcv["arms"].get(arm, {})
            if a.get("status") == "MISSING":
                A(f"{label:30}MISSING")
                continue
            A(f"{label:30}{fmt(a['2d_visible']['median_px'],9,2)}{fmt(a['2d_visible']['p90_px'],9,2)}"
              f"{fmt(a['2d_visible']['gross20'],9,3)}{fmt(a['6d_ALL']['iou3d_median'],9,3)}"
              f"{fmt(a['6d_ALL']['add_sym_auc'],9,3)}")
        A("```")
        A("")

    if gd:
        A("## [DISTILL TARGET]  usable 부분집합")
        A("")
        A("```")
        u = gd["USABLE_only"]
        A(f"{'arm':30}{'median':>9}{'p90':>9}{'gross20':>9}")
        A("─" * 57)
        for a in ("R0", "F1", "F2", "F2S"):
            if a in u and u[a]["n"]:
                A(f"{a:30}{fmt(u[a]['median_px'],9,2)}{fmt(u[a]['p90_px'],9,2)}{fmt(u[a]['gross20'],9,3)}")
        A("")
        A(f"coverage  {gd['coverage']['n_usable']}/{gd['coverage']['n_keypoints']} = "
          f"{gd['coverage']['usable_rate']:.4f}")
        A("```")
        A("")

    if gct:
        A("## [합의 게이트의 전이]")
        A("")
        A("```")
        A(f"{'모집단':36}{'통과율':>10}")
        A("─" * 46)
        for k, v in gct["pass_rate_at_tau"].items():
            A(f"{k:36}{v:10.4f}")
        A("```")
        A("")

    if ge:
        A("## [DOMAIN]")
        A("")
        A("```")
        for lv, b in ge["domain_auroc_by_level"].items():
            A(f"{lv:12} feature dim {b['feature_dim']:5d}   domain AUROC {b['auroc_mean']:.4f}")
        a = ge["association_on_dev_eval"]
        A("")
        A(f"DEV_EVAL n={a['n_frames']}   spearman(score, kp median) {a['spearman_score_vs_kp_median']:+.4f}")
        A(f"gross-frame 분리 AUC  {a['auc_score_separates_gross_frames']}")
        A("```")
        A("")

    A("## [ADAPTER]")
    A("")
    A("```")
    A(f"{adapter}")
    A("```")

    md = M.FINAL / "MULTITEACHER_FINAL_REPORT.md"
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out.relative_to(M.REPO_ROOT)}")
    print(f"-> {md.relative_to(M.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
