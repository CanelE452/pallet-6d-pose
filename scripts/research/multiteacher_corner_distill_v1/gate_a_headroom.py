"""GATE A — 여러 교사 상보성 감사. 학습 0, 새 추론 0(캐시 재사용).

측정 순서
  1. teacher 단독 성능
  2. teacher 오차 상보성 (좌표 불일치 / GT 잔차 상관 / 조건부 정확도)
  3. oracle-best-teacher 상한  ← 배포 방법이 아니라 상한
  4. prediction-only fusion 통제 F0/F1/F2  (F3 = BLOCKED_NOT_COMPARABLE)
  5. 불일치가 gross error 를 예측하는가 (AUC)
  6. 판정

임계는 METHOD_LOCK 에서 읽는다. 여기서 새로 고르지 않는다.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M

REF = "T0_R0_YOLO26N_G38LEGACY"


def build_table(registry, gts):
    """(frame, keypoint) 롱테이블. supervised 인 것만."""
    preds = {tid: M.load_prediction_file(M.PREDICTIONS / f"{tid}.json") for tid in registry}
    tids = list(registry)
    rows = []
    for gt in gts:
        entries = {tid: preds[tid].get(gt["frame_id"]) for tid in tids}
        xy = {tid: M.prediction_keypoints(entries[tid]) for tid in tids}
        for k in range(9):
            if not gt["supervised"][k]:
                continue
            row = {"frame_id": gt["frame_id"], "session": gt["session_id"],
                   "object_type": gt["object_type"], "domain": gt["paper_domain"],
                   "kp": k, "visible": bool(gt["visible"][k]),
                   "is_corner": k < 8, "gt": gt["xy"][k]}
            for tid in tids:
                p = xy[tid]
                if p is None or not np.isfinite(p[k]).all():
                    row[tid] = None
                    row[tid + "__err"] = np.nan
                    row[tid + "__conf"] = np.nan
                else:
                    row[tid] = p[k]
                    row[tid + "__err"] = float(np.linalg.norm(p[k] - gt["xy"][k]))
                    confs = entries[tid].get("keypoints_conf")
                    row[tid + "__conf"] = float(confs[k]) if confs else np.nan
            rows.append(row)
    return tids, rows


def subset_stats(rows, tid, mask=None):
    errs = [r[tid + "__err"] for r in rows
            if (mask is None or mask(r)) and np.isfinite(r[tid + "__err"])]
    return M.error_stats(errs)


def rel(base, new):
    return None if base in (None, 0) else 100.0 * (base - new) / base


def main() -> int:
    lock = json.loads(M.METHOD_LOCK_PATH.read_text())
    registry = json.loads((M.TRACK / "TEACHER_REGISTRY.json").read_text())["teachers"]
    gts = [M.load_gt(f) for f in M.dev_eval_frames()]
    tids, rows = build_table(registry, gts)
    corners = [r for r in rows if r["is_corner"]]
    visible = [r for r in corners if r["visible"]]

    report = {"schema_version": "mtcd_gate_a_v1",
              "method_lock_sha256": M.sha256_file(M.METHOD_LOCK_PATH),
              "population": "DEV_EVAL 319, supervised keypoints",
              "n_rows_all": len(rows), "n_rows_corner": len(corners),
              "n_rows_visible_corner": len(visible),
              "new_training": 0, "new_inference": 0}

    # ---------------------------------------------------- 1. 단독 성능 -------
    standalone = {}
    for tid in tids:
        block = {
            "architecture": registry[tid]["architecture"],
            "axis": registry[tid]["heterogeneity_axis"],
            "training_data": registry[tid]["training_data"],
            "coverage_supervised": float(np.mean([np.isfinite(r[tid + "__err"]) for r in rows])),
            "ALL_supervised": subset_stats(rows, tid),
            "corners": subset_stats(corners, tid),
            "visible_corners": subset_stats(visible, tid),
            "centroid": subset_stats([r for r in rows if r["kp"] == 8], tid),
            "by_corner": {str(k): subset_stats(corners, tid, lambda r, k=k: r["kp"] == k)
                          for k in range(8)},
            "by_object": {o: subset_stats(corners, tid, lambda r, o=o: r["object_type"] == o)
                          for o in ("plastic_standard_110x130x11", "wood_small_80x59x14")},
            "by_domain": {d: subset_stats(corners, tid, lambda r, d=d: r["domain"] == d)
                          for d in ("none", "daytime", "nighttime")},
        }
        standalone[tid] = block
    report["standalone"] = standalone

    # 최고 단독 teacher — visible corners 의 p90 기준(게이트가 그걸 본다)
    ranked = sorted(tids, key=lambda t: standalone[t]["visible_corners"]["p90_px"])
    report["best_single_teacher_by_visible_p90"] = ranked[0]
    report["teacher_ranking_visible_p90"] = [
        {"teacher": t, "p90": standalone[t]["visible_corners"]["p90_px"],
         "median": standalone[t]["visible_corners"]["median_px"],
         "gross20": standalone[t]["visible_corners"]["gross20"]} for t in ranked]

    # ---------------------------------------------------- 2. 상보성 ---------
    pairwise = {}
    for a, b in combinations(tids, 2):
        both = [r for r in corners
                if np.isfinite(r[a + "__err"]) and np.isfinite(r[b + "__err"])]
        if not both:
            continue
        da = np.array([r[a + "__err"] for r in both])
        db = np.array([r[b + "__err"] for r in both])
        disagree = np.array([float(np.linalg.norm(r[a] - r[b])) for r in both])
        ea = np.array([r[a] - r["gt"] for r in both])
        eb = np.array([r[b] - r["gt"] for r in both])
        from scipy.stats import pearsonr, spearmanr
        pairwise[f"{a}|{b}"] = {
            "n": len(both),
            "disagreement_median_px": float(np.median(disagree)),
            "disagreement_p90_px": float(np.percentile(disagree, 90)),
            "residual_dx_pearson": float(pearsonr(ea[:, 0], eb[:, 0])[0]),
            "residual_dy_pearson": float(pearsonr(ea[:, 1], eb[:, 1])[0]),
            "residual_dx_spearman": float(spearmanr(ea[:, 0], eb[:, 0])[0]),
            "residual_dy_spearman": float(spearmanr(ea[:, 1], eb[:, 1])[0]),
            "error_magnitude_pearson": float(pearsonr(da, db)[0]),
            "error_magnitude_spearman": float(spearmanr(da, db)[0]),
            "gross20_cooccurrence": float(np.mean((da > 20) & (db > 20))),
            "gross20_a": float(np.mean(da > 20)), "gross20_b": float(np.mean(db > 20)),
            "gross20_jaccard": float(np.sum((da > 20) & (db > 20)) /
                                     max(np.sum((da > 20) | (db > 20)), 1)),
        }
    report["pairwise"] = pairwise

    # 조건부 정확도 — 이게 핵심이다
    conditional = {}
    for tid in tids:
        if tid == REF:
            continue
        g20 = [r for r in corners if np.isfinite(r[REF + "__err"]) and r[REF + "__err"] > 20
               and np.isfinite(r[tid + "__err"])]
        g40 = [r for r in corners if np.isfinite(r[REF + "__err"]) and r[REF + "__err"] > 40
               and np.isfinite(r[tid + "__err"])]
        conditional[tid] = {
            "n_ref_gross20": len(g20),
            "rescue_le10px": float(np.mean([r[tid + "__err"] <= 10 for r in g20])) if g20 else None,
            "rescue_le20px": float(np.mean([r[tid + "__err"] <= 20 for r in g20])) if g20 else None,
            "n_ref_gross40": len(g40),
            "rescue_le20px_on_gross40": float(np.mean([r[tid + "__err"] <= 20 for r in g40])) if g40 else None,
        }
    # 어느 teacher라도 구하는 비율 (합집합)
    g20 = [r for r in corners if np.isfinite(r[REF + "__err"]) and r[REF + "__err"] > 20]
    others = [t for t in tids if t != REF]
    conditional["ANY_OTHER_TEACHER"] = {
        "n_ref_gross20": len(g20),
        "rescue_le10px": float(np.mean([
            any(np.isfinite(r[t + "__err"]) and r[t + "__err"] <= 10 for t in others)
            for r in g20])) if g20 else None,
        "rescue_le20px": float(np.mean([
            any(np.isfinite(r[t + "__err"]) and r[t + "__err"] <= 20 for t in others)
            for r in g20])) if g20 else None,
    }
    report["conditional_rescue"] = conditional

    # ---------------------------------------------------- 3. oracle 상한 ----
    def oracle_errs(rows_, pool):
        out = []
        for r in rows_:
            vals = [r[t + "__err"] for t in pool if np.isfinite(r[t + "__err"])]
            if vals:
                out.append(min(vals))
        return out

    nested = []
    pool = [REF]
    for tid in ranked:
        if tid not in pool:
            pool.append(tid)
        nested.append({"n_teachers": len(pool), "pool": list(pool),
                       "visible": M.error_stats(oracle_errs(visible, pool)),
                       "corners": M.error_stats(oracle_errs(corners, pool))})
    report["oracle_nested"] = nested
    oracle_all = {"visible": M.error_stats(oracle_errs(visible, tids)),
                  "corners": M.error_stats(oracle_errs(corners, tids)),
                  "note": "GT 를 써서 keypoint 마다 최선 teacher 를 고른 값. 배포 불가한 상한이다."}
    report["oracle_all_teachers"] = oracle_all

    # ---------------------------------------------------- 4. fusion 통제 ----
    def fuse(rows_, mode, pool):
        out = []
        for r in rows_:
            pts = [r[t] for t in pool if r[t] is not None and np.isfinite(r[t]).all()]
            if not pts:
                continue
            P = np.asarray(pts)
            if mode == "F1":
                q = np.median(P, axis=0)
            elif mode == "F2":
                d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2).sum(1)
                q = P[int(np.argmin(d))]
            else:
                raise ValueError(mode)
            out.append(float(np.linalg.norm(q - r["gt"])))
        return out

    fusion = {
        "F0_r0_only": {"visible": subset_stats(visible, REF), "corners": subset_stats(corners, REF)},
        "F1_component_median": {"visible": M.error_stats(fuse(visible, "F1", tids)),
                                "corners": M.error_stats(fuse(corners, "F1", tids))},
        "F2_geometric_medoid": {"visible": M.error_stats(fuse(visible, "F2", tids)),
                                "corners": M.error_stats(fuse(corners, "F2", tids))},
        "F3_uncertainty_weighted": {"status": "BLOCKED_NOT_COMPARABLE",
                                    "reason": lock["gate_a"]["F3_block_reason"]},
        "no_fitted_weights": True,
    }
    report["fusion_controls"] = fusion

    # ---------------------------------------------------- 5. 불일치 -> AUC --
    def auc(scores, labels):
        s = np.asarray(scores, float); y = np.asarray(labels, bool)
        ok = np.isfinite(s)
        s, y = s[ok], y[ok]
        if y.sum() == 0 or (~y).sum() == 0:
            return None
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(s) + 1)
        # 동점 평균 순위
        _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
        return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))

    usable = [r for r in corners if np.isfinite(r[REF + "__err"])]
    labels = [r[REF + "__err"] > 20 for r in usable]
    signals = {"max_disagreement": [], "median_pair_disagreement": [],
               "teacher_spread_trace": [], "r0_keypoint_conf": []}
    for r in usable:
        pts = [r[t] for t in tids if r[t] is not None and np.isfinite(r[t]).all()]
        P = np.asarray(pts)
        if len(P) >= 2:
            d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
            iu = np.triu_indices(len(P), 1)
            signals["max_disagreement"].append(float(d[iu].max()))
            signals["median_pair_disagreement"].append(float(np.median(d[iu])))
            signals["teacher_spread_trace"].append(float(np.trace(np.cov(P.T))))
        else:
            for k in ("max_disagreement", "median_pair_disagreement", "teacher_spread_trace"):
                signals[k].append(np.nan)
        signals["r0_keypoint_conf"].append(-r[REF + "__conf"] if np.isfinite(r[REF + "__conf"]) else np.nan)
    aucs = {k: auc(v, labels) for k, v in signals.items()}
    ranks = []
    for k in ("max_disagreement", "median_pair_disagreement", "teacher_spread_trace", "r0_keypoint_conf"):
        v = np.asarray(signals[k], float)
        r = np.full(len(v), np.nan)
        ok = np.isfinite(v)
        order = np.argsort(v[ok]); tmp = np.empty(ok.sum()); tmp[order] = np.arange(ok.sum())
        r[ok] = tmp / max(ok.sum() - 1, 1)
        ranks.append(r)
    aucs["simple_rank_average"] = auc(np.nanmean(np.vstack(ranks), axis=0), labels)
    report["gross20_prediction_auc"] = {
        "n": len(usable), "positive_rate": float(np.mean(labels)), "auc": aucs,
        "rule": "각 scalar 단독 + 정규화 없는 단순 rank average 만. 가중치 fit 0."}

    # ---------------------------------------------------- 6. 판정 -----------
    best = ranked[0]
    b_vis = standalone[best]["visible_corners"]
    o_vis = oracle_all["visible"]
    crit = {
        "best_single_teacher": best,
        "best_single_visible_p90": b_vis["p90_px"],
        "best_single_visible_gross20": b_vis["gross20"],
        "oracle_visible_p90": o_vis["p90_px"],
        "oracle_visible_gross20": o_vis["gross20"],
        "p90_relative_improvement_pct": rel(b_vis["p90_px"], o_vis["p90_px"]),
        "gross20_relative_reduction_pct": rel(b_vis["gross20"], o_vis["gross20"]),
        "threshold_p90_pct": 15.0, "threshold_gross20_pct": 20.0,
        "oracle_criterion_met": None,
        "rescue_rate_any_other_le10px": conditional["ANY_OTHER_TEACHER"]["rescue_le10px"],
        "threshold_rescue": 0.20,
        "rescue_criterion_met": None,
    }
    crit["oracle_criterion_met"] = bool(
        (crit["p90_relative_improvement_pct"] or 0) >= 15.0 or
        (crit["gross20_relative_reduction_pct"] or 0) >= 20.0)
    crit["rescue_criterion_met"] = bool((crit["rescue_rate_any_other_le10px"] or 0) >= 0.20)
    crit["MULTI_TEACHER_HEADROOM"] = ("STRONG" if crit["oracle_criterion_met"]
                                      and crit["rescue_criterion_met"] else "INSUFFICIENT")
    report["verdict"] = crit

    out = M.GATE_A / "GATE_A_RESULT.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float) + "\n")
    print(json.dumps(crit, indent=2, ensure_ascii=False, default=float))
    print(f"\n-> {out.relative_to(M.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
