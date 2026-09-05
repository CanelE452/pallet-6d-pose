"""SOURCE_DEV 에서 정한 합의 임계가 real 로 옮겨가지 않는다 — 그 크기를 잰다.

임계를 바꾸지 않는다.  왜 안 옮겨가는지를 수치로 남긴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M

REF = "T0_R0_YOLO26N_G38LEGACY"


def real_disagreements(frames_pred, gts, tids):
    out = []
    for gt in gts:
        ref = frames_pred[REF].get(gt["frame_id"])
        if not ref or ref.get("status") != "OK":
            continue
        bx = ref["box_xyxy"]
        diag = float(np.hypot(bx[2] - bx[0], bx[3] - bx[1]))
        if diag <= 1:
            continue
        pts = {}
        for tid in tids:
            e = frames_pred[tid].get(gt["frame_id"])
            pts[tid] = (np.asarray(e["keypoints_xy"], float)
                        if e and e.get("status") == "OK" and e.get("keypoints_xy") else None)
        for k in range(8):
            if not gt["visible"][k]:
                continue
            P = np.array([pts[t][k] for t in tids
                          if pts[t] is not None and np.isfinite(pts[t][k]).all()])
            if len(P) < 3:
                continue
            d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
            out.append({
                "disagree": float(d[np.triu_indices(len(P), 1)].max()) / diag,
                "consensus_err": float(np.linalg.norm(np.median(P, axis=0) - gt["xy"][k])) / diag,
                "r0_err": float(np.linalg.norm(pts[REF][k] - gt["xy"][k])) / diag,
                "domain": gt["paper_domain"], "kp": k})
    return out


def main() -> int:
    tau = float(json.loads((M.GATE_C / "CONSENSUS_GATE.json").read_text())["TAU_CONSENSUS_NORMALISED"])
    src = json.loads((M.GATE_C / "CONSENSUS_GATE.json").read_text())
    registry = json.loads((M.TRACK / "TEACHER_REGISTRY.json").read_text())["teachers"]
    tids = list(registry)
    preds = {t: M.load_prediction_file(M.PREDICTIONS / f"{t}.json") for t in tids}
    gts = [M.load_gt(f) for f in M.dev_eval_frames()]
    rows = real_disagreements(preds, gts, tids)

    dis = np.array([r["disagree"] for r in rows])
    err = np.array([r["consensus_err"] for r in rows])
    r0e = np.array([r["r0_err"] for r in rows])
    cache = json.loads((M.GATE_C / "TARGET_TEACHER_CACHE.json").read_text())
    pool_dis = np.array([kp["disagreement_normalised"] for f in cache["frames"].values()
                         for kp in f["keypoints"] if "disagreement_normalised" in kp])

    def q(a):
        return {p: float(np.percentile(a, p)) for p in (10, 25, 50, 75, 90, 95, 99)}

    report = {
        "schema_version": "mtcd_consensus_transfer_v1",
        "tau_frozen_on_source": tau,
        "tau_not_changed": True,
        "disagreement_quantiles_normalised": {
            "SOURCE_DEV_synthetic_val": {"note": "freeze_consensus_gate.py 의 곡선에서 재구성",
                                         "coverage_at_tau": src["coverage_at_tau"]},
            "DEV_EVAL_real_visible": q(dis),
            "TARGET_UNLABELED_real_all_corners": q(pool_dis)},
        "pass_rate_at_tau": {
            "SOURCE_DEV": src["coverage_at_tau"],
            "DEV_EVAL_real_visible": float(np.mean(dis <= tau)),
            "TARGET_UNLABELED": float(np.mean(pool_dis <= tau))},
        "consensus_quality_on_passing_real": {
            "n": int((dis <= tau).sum()),
            "consensus_err_median_frac": float(np.median(err[dis <= tau])) if (dis <= tau).any() else None,
            "consensus_err_p90_frac": float(np.percentile(err[dis <= tau], 90)) if (dis <= tau).any() else None,
            "r0_err_median_frac": float(np.median(r0e[dis <= tau])) if (dis <= tau).any() else None,
            "r0_err_p90_frac": float(np.percentile(r0e[dis <= tau], 90)) if (dis <= tau).any() else None},
        "consensus_quality_on_failing_real": {
            "n": int((dis > tau).sum()),
            "consensus_err_median_frac": float(np.median(err[dis > tau])),
            "r0_err_median_frac": float(np.median(r0e[dis > tau]))},
        "interpretation":
            "SOURCE_DEV 에서 정한 tau 가 real 에서 통과율을 20% 에서 1~4% 로 떨어뜨린다. "
            "교사 불일치 분포 자체가 도메인 사이에서 다르다 — 합의 게이트를 source 에서 "
            "보정하는 방식이 이 문제에는 옮겨가지 않는다는 뜻이다. 임계는 바꾸지 않는다.",
    }
    out = M.GATE_C / "CONSENSUS_TRANSFER_DIAGNOSTIC.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"tau = {tau:.3f}  (SOURCE_DEV 에서 동결, 변경 없음)")
    print(f"{'모집단':34}{'통과율':>10}")
    for k, v in report["pass_rate_at_tau"].items():
        print(f"{k:34}{v:10.4f}")
    print(f"\n불일치 분위수 (box 대각선 대비)")
    print(f"{'모집단':34}" + "".join(f"{p:>9}" for p in (10, 25, 50, 75, 90, 95, 99)))
    for name in ("DEV_EVAL_real_visible", "TARGET_UNLABELED_real_all_corners"):
        qq = report["disagreement_quantiles_normalised"][name]
        print(f"{name:34}" + "".join(f"{qq[p]:9.4f}" for p in (10, 25, 50, 75, 90, 95, 99)))
    p = report["consensus_quality_on_passing_real"]
    print(f"\n통과한 real keypoint {p['n']} 개에서  합의 오차 median {p['consensus_err_median_frac']:.4f} "
          f"p90 {p['consensus_err_p90_frac']:.4f}   (R0 median {p['r0_err_median_frac']:.4f} p90 {p['r0_err_p90_frac']:.4f})")
    print(f"-> {out.relative_to(M.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
