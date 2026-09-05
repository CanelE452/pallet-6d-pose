"""GATE D — 융합 pseudo-target 자체의 품질을 학생 학습 **전에** 잰다.

TARGET_UNLABELED 에는 GT 가 없으므로, **완전히 같은 규칙**을 DEV_EVAL 에 적용해
target 이 R0 보다 나은지 본다.  GT 는 채점에만 쓰고 target 생성에는 쓰지 않는다.

    R0    기준 하드 좌표
    F1    좌표 성분별 median
    F2    geometric medoid
    F2S   교사 가우시안 혼합의 기댓값 (soft distribution mean)
    F3S   F2S + 국소 전문가  (Gate C 가 GO 일 때만)

abstention 은 METHOD_LOCK 에 사전등록돼 있다 — 불일치가 tau 를 넘으면 그 keypoint 에
real loss 를 주지 않는다.  그래서 품질은 **usable 부분집합** 에서 재고, coverage 를 함께 싣는다.

DISTILL_TARGET_QUALITY = FAIL 이면 student 학습을 하지 않는다.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M

REF = "T0_R0_YOLO26N_G38LEGACY"
MIXTURE_SIGMA_PX = 3.0


def soft_mean(points, sigma=MIXTURE_SIGMA_PX):
    """등가중 가우시안 혼합의 기댓값 = 좌표 평균. 분산도 함께 돌려준다."""
    P = np.asarray(points, float)
    mu = P.mean(axis=0)
    spread = float(np.mean(np.linalg.norm(P - mu, axis=1)))
    return mu, np.sqrt(spread ** 2 + sigma ** 2)


def build_rows(tids, gts, preds, specialist_pred=None):
    rows = []
    for gt in gts:
        ref = preds[REF].get(gt["frame_id"])
        if not ref or ref.get("status") != "OK":
            continue
        bx = ref["box_xyxy"]
        diag = float(np.hypot(bx[2] - bx[0], bx[3] - bx[1]))
        if diag <= 1:
            continue
        pt = {t: M.prediction_keypoints(preds[t].get(gt["frame_id"])) for t in tids}
        sp = (M.prediction_keypoints(specialist_pred.get(gt["frame_id"]))
              if specialist_pred else None)
        for k in range(8):
            if not gt["supervised"][k]:
                continue
            P = np.array([pt[t][k] for t in tids
                          if pt[t] is not None and np.isfinite(pt[t][k]).all()])
            if len(P) < 3:
                continue
            d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
            disagree = float(d[np.triu_indices(len(P), 1)].max()) / diag
            mu, sd = soft_mean(P)
            medoid = P[int(np.argmin(np.linalg.norm(
                P[:, None, :] - P[None, :, :], axis=2).sum(1)))]
            row = {"frame_id": gt["frame_id"], "session": gt["session_id"], "kp": k,
                   "visible": bool(gt["visible"][k]), "disagreement": disagree,
                   "R0": float(np.linalg.norm(pt[REF][k] - gt["xy"][k])),
                   "F1": float(np.linalg.norm(np.median(P, axis=0) - gt["xy"][k])),
                   "F2": float(np.linalg.norm(medoid - gt["xy"][k])),
                   "F2S": float(np.linalg.norm(mu - gt["xy"][k])),
                   "soft_sigma": sd}
            if sp is not None and np.isfinite(sp[k]).all():
                blend = 0.5 * (mu + sp[k])
                row["F3S"] = float(np.linalg.norm(blend - gt["xy"][k]))
            rows.append(row)
    return rows


def main() -> int:
    tau = float(json.loads((M.GATE_C / "CONSENSUS_GATE.json").read_text())
                ["TAU_CONSENSUS_NORMALISED"])
    registry = json.loads((M.TRACK / "TEACHER_REGISTRY.json").read_text())["teachers"]
    tids = list(registry)
    preds = {t: M.load_prediction_file(M.PREDICTIONS / f"{t}.json") for t in tids}
    gts = [M.load_gt(f) for f in M.dev_eval_frames()]

    specialist_pred, specialist_arm = None, None
    verdict_path = M.GATE_C / "GATE_C_VERDICT.json"
    if verdict_path.exists():
        v = json.loads(verdict_path.read_text())
        if v.get("REAL_LOCAL_SPECIALIST") == "GO":
            specialist_arm = "C1"
            specialist_pred = M.load_prediction_file(M.PREDICTIONS / "C1.json")

    rows = build_rows(tids, gts, preds, specialist_pred)
    arms = ["R0", "F1", "F2", "F2S"] + (["F3S"] if specialist_arm else [])

    def block(subset):
        out = {"n": len(subset)}
        for a in arms:
            out[a] = M.error_stats([r[a] for r in subset if a in r])
        return out

    usable = [r for r in rows if r["disagreement"] <= tau]
    usable_vis = [r for r in usable if r["visible"]]
    report = {
        "schema_version": "mtcd_gate_d_target_quality_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method_lock_sha256": M.sha256_file(M.METHOD_LOCK_PATH),
        "tau_consensus_normalised": tau,
        "specialist_included": specialist_arm,
        "gt_used_for": "scoring only — target generation never sees GT",
        "coverage": {"n_keypoints": len(rows), "n_usable": len(usable),
                     "usable_rate": len(usable) / len(rows) if rows else 0.0},
        "ALL_supervised": block(rows),
        "USABLE_only": block(usable),
        "USABLE_visible_only": block(usable_vis),
    }

    u = report["USABLE_only"]
    best_fusion = min([a for a in arms if a != "R0"],
                      key=lambda a: (u[a]["p90_px"], u[a]["median_px"]))
    report["best_fusion_arm_on_usable"] = best_fusion
    checks = {
        "median_not_worse": u[best_fusion]["median_px"] <= u["R0"]["median_px"],
        "p90_not_worse": u[best_fusion]["p90_px"] <= u["R0"]["p90_px"],
        "gross20_not_worse": u[best_fusion]["gross20"] <= u["R0"]["gross20"],
    }
    report["checks_on_usable_subset"] = checks
    report["DISTILL_TARGET_QUALITY"] = "PASS" if all(checks.values()) else "FAIL"
    report["consequence"] = (
        "PASS — student 학습을 진행한다"
        if report["DISTILL_TARGET_QUALITY"] == "PASS"
        else "FAIL — 융합 target 이 R0 보다 낫지 않다. METHOD_LOCK 규칙에 따라 student 학습을 하지 않는다.")

    out = M.GATE_D / "DISTILL_TARGET_QUALITY.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float) + "\n")

    for name in ("ALL_supervised", "USABLE_only", "USABLE_visible_only"):
        b = report[name]
        print(f"=== {name}  n={b['n']} ===")
        print(f"{'arm':6}{'med':>8}{'p90':>9}{'g20':>8}{'g40':>8}")
        for a in arms:
            s = b[a]
            if s["n"]:
                print(f"{a:6}{s['median_px']:8.2f}{s['p90_px']:9.2f}"
                      f"{s['gross20']:8.3f}{s['gross40']:8.3f}")
        print()
    print(f"coverage usable {report['coverage']['usable_rate']:.4f} "
          f"({report['coverage']['n_usable']}/{report['coverage']['n_keypoints']})")
    print(f"best fusion on usable = {best_fusion}")
    for k, v in checks.items():
        print(f"  {k:22}{v}")
    print(f"\nDISTILL_TARGET_QUALITY = {report['DISTILL_TARGET_QUALITY']}")
    print(f"-> {out.relative_to(M.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
