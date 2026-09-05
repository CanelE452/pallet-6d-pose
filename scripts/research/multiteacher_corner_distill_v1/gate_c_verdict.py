"""GATE C 판정 — 임계는 METHOD_LOCK 에서 읽는다. 여기서 고르지 않는다.

SPECIALIST_GO 는 다음을 모두 만족할 때만:
    C1 vs R0  visible p90 >= 10% 개선  OR  gross20 >= 15% 상대감소
    median 악화 <= 5%
    R0-good(<=10px) harm rate <= 10%
    PoseCov 하락 <= 1%p
    IoU3D 또는 ADDsym AUC 점추정이 R0 보다 악화하지 않음
    C1 이 C0 보다 좋을 것
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M

REF = "T0_R0_YOLO26N_G38LEGACY"


def rel_improvement(base, new):
    return None if base in (None, 0) else 100.0 * (base - new) / base


def load_6d(arm, folder):
    p = folder / f"POSE_EVALUATION_{arm}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["paths"]["MAIN"]


def main() -> int:
    lock = json.loads(M.METHOD_LOCK_PATH.read_text())
    two_d = json.loads((M.GATE_C / "GATE_C_2D.json").read_text())
    blocks = two_d["two_d"]
    rh = two_d["rescue_and_harm"]

    r0_2d = blocks["R0"]["POOLED_ALL"]["visible"]
    r0_6d = load_6d(REF, M.GATE_A)
    if r0_6d is None:
        raise SystemExit("R0 의 6D 평가가 없다 — eval_pose_arm.py 를 먼저 돌려라")

    out = {"schema_version": "mtcd_gate_c_verdict_v1",
           "method_lock_sha256": M.sha256_file(M.METHOD_LOCK_PATH),
           "criteria_source": "METHOD_LOCK.gate_c.GO_criterion",
           "baseline": {"2d_visible": r0_2d, "6d_ALL": r0_6d["ALL"],
                        "coverage": r0_6d["coverage"]},
           "arms": {}}

    for arm in ("C0", "C1"):
        a2 = blocks.get(arm, {}).get("POOLED_ALL", {}).get("visible")
        a6 = load_6d(arm, M.GATE_C)
        if a2 is None or a6 is None:
            out["arms"][arm] = {"status": "MISSING"}
            continue
        block = {
            "2d_visible": a2,
            "6d_ALL": a6["ALL"], "coverage": a6["coverage"],
            "p90_relative_improvement_pct": rel_improvement(r0_2d["p90_px"], a2["p90_px"]),
            "gross20_relative_reduction_pct": rel_improvement(r0_2d["gross20"], a2["gross20"]),
            "median_change_pct": -rel_improvement(r0_2d["median_px"], a2["median_px"]),
            "harm_rate": rh[arm]["harm_rate"], "rescue_rate": rh[arm]["rescue_rate"],
            "posecov_drop_pp": 100.0 * (r0_6d["coverage"] - a6["coverage"]),
            "iou3d_delta": a6["ALL"]["iou3d_median"] - r0_6d["ALL"]["iou3d_median"],
            "addsym_auc_delta": a6["ALL"]["add_sym_auc"] - r0_6d["ALL"]["add_sym_auc"],
        }
        out["arms"][arm] = block

    c1 = out["arms"].get("C1", {})
    c0 = out["arms"].get("C0", {})
    checks = {}
    if c1.get("status") != "MISSING":
        checks["tail_improved"] = bool((c1["p90_relative_improvement_pct"] or 0) >= 10.0 or
                                       (c1["gross20_relative_reduction_pct"] or 0) >= 15.0)
        checks["median_not_worse_than_5pct"] = bool((c1["median_change_pct"] or 0) <= 5.0)
        checks["harm_rate_le_10pct"] = bool((c1["harm_rate"] or 0) <= 0.10)
        checks["posecov_drop_le_1pp"] = bool(c1["posecov_drop_pp"] <= 1.0)
        checks["holistic_6d_not_worse"] = bool(c1["iou3d_delta"] >= 0 or
                                               c1["addsym_auc_delta"] >= 0)
        checks["c1_better_than_c0"] = bool(
            c0.get("status") != "MISSING" and
            c1["2d_visible"]["p90_px"] <= c0["2d_visible"]["p90_px"] and
            c1["2d_visible"]["gross20"] <= c0["2d_visible"]["gross20"])
    out["checks"] = checks
    out["REAL_LOCAL_SPECIALIST"] = "GO" if checks and all(checks.values()) else "STOP"
    out["consequence"] = (
        "GO — specialist 를 Gate D 의 teacher pool 에 넣는다"
        if out["REAL_LOCAL_SPECIALIST"] == "GO" else
        "STOP — specialist 를 teacher pool 에 넣지 않는다. "
        "Gate A 가 STRONG 이므로 specialist 없이 student distillation 은 진행 가능하다.")

    p = M.GATE_C / "GATE_C_VERDICT.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float) + "\n")

    print(f"{'arm':6}{'med':>8}{'p90':>9}{'g20':>8}{'IoU3D':>8}{'ADDauc':>9}{'harm':>8}{'rescue':>8}")
    b = out["baseline"]
    print(f"{'R0':6}{r0_2d['median_px']:8.2f}{r0_2d['p90_px']:9.2f}{r0_2d['gross20']:8.3f}"
          f"{b['6d_ALL']['iou3d_median']:8.3f}{b['6d_ALL']['add_sym_auc']:9.3f}{'-':>8}{'-':>8}")
    for arm in ("C0", "C1"):
        a = out["arms"][arm]
        if a.get("status") == "MISSING":
            print(f"{arm:6}MISSING")
            continue
        v = a["2d_visible"]
        print(f"{arm:6}{v['median_px']:8.2f}{v['p90_px']:9.2f}{v['gross20']:8.3f}"
              f"{a['6d_ALL']['iou3d_median']:8.3f}{a['6d_ALL']['add_sym_auc']:9.3f}"
              f"{(a['harm_rate'] or 0):8.3f}{(a['rescue_rate'] or 0):8.3f}")
    print()
    for k, v in checks.items():
        print(f"  {k:34}{v}")
    print(f"\nREAL_LOCAL_SPECIALIST = {out['REAL_LOCAL_SPECIALIST']}")
    print(f"-> {p.relative_to(M.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
