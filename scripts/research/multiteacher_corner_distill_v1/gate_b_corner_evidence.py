"""GATE B — 국소 코너 증거 감사. 학습 0.

질문: R0 예측 주변 실제 RGB 안에 더 정확한 semantic corner 후보가 존재하는가?

primary 는 supervised **visible**(visibility==2) 코너만이다.
occluded / truncated 는 별도 진단으로만 본다.

후보 생성 파라미터·반경은 METHOD_LOCK 에서 읽는다. 여기서 고르지 않는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M
from calibrate_on_source import (candidates_gradient_junction, candidates_harris,
                                 candidates_lsd_intersections, candidates_shi_tomasi)

REF = "T0_R0_YOLO26N_G38LEGACY"
FAMILIES = {"shi_tomasi": candidates_shi_tomasi, "harris": candidates_harris,
            "lsd": candidates_lsd_intersections, "junction": candidates_gradient_junction}
KEY_MAP = {"shi_tomasi": "shi_tomasi", "harris": "harris",
           "lsd": "lsd_intersections", "junction": "gradient_junction"}


def cuboid_edge_directions(pred_xy, corner):
    """그 코너에 붙는 3개 cuboid 변의 예측 투영 방향 (단위벡터)."""
    dirs = []
    for a, b in M.INCIDENT_EDGES[corner]:
        other = b if a == corner else a
        if not (np.isfinite(pred_xy[corner]).all() and np.isfinite(pred_xy[other]).all()):
            continue
        v = pred_xy[other] - pred_xy[corner]
        n = np.linalg.norm(v)
        if n > 1e-6:
            dirs.append(v / n)
    return dirs


def collect(image_gray, centre, radius, cfg):
    """반경 patch 안의 후보 + 각 후보의 부가 신호."""
    h, w = image_gray.shape
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(w, cx + radius + 1), min(h, cy + radius + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return [], {}
    patch = image_gray[y0:y1, x0:x1]
    origin = np.array([x0, y0], float)
    out, meta = [], {}
    for family, func in FAMILIES.items():
        try:
            pts = func(patch, **cfg[KEY_MAP[family]])
        except Exception:
            pts = []
        meta[family] = len(pts)
        for p in pts:
            out.append({"xy": np.asarray(p, float) + origin, "family": family})
    # corner response (Shi-Tomasi 최소고유값) — 선택기의 term 하나
    resp = cv2.cornerMinEigenVal(np.float32(patch), 3, 3)
    for c in out:
        loc = c["xy"] - origin
        ix, iy = int(round(loc[0])), int(round(loc[1]))
        c["response"] = (float(resp[iy, ix]) if 0 <= iy < resp.shape[0]
                         and 0 <= ix < resp.shape[1] else 0.0)
    # LSD 선분 — 교차각·방향 일치도용
    try:
        lines = cv2.createLineSegmentDetector().detect(patch)[0]
    except Exception:
        lines = None
    segs = [] if lines is None else [l[0] for l in lines]
    return out, {"counts": meta, "segments": segs, "origin": origin}


def selector_score(cands, info, coarse_xy, edge_dirs):
    """prediction-only 점수 — 각 term 의 rank 를 동일 비중 평균한다.

    magnitude 튜닝도 가중치 학습도 하지 않는다.
    """
    if not cands:
        return None
    P = np.array([c["xy"] for c in cands])
    dist = np.linalg.norm(P - coarse_xy, axis=1)
    resp = np.array([c["response"] for c in cands])
    segs, origin = info["segments"], info["origin"]

    cross_angle = np.zeros(len(cands))
    edge_align = np.zeros(len(cands))
    if segs:
        S = np.array(segs, float)
        mid = (S[:, :2] + S[:, 2:]) / 2.0 + origin
        d = S[:, 2:] - S[:, :2]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / np.maximum(n, 1e-6)
        for i, p in enumerate(P):
            near = np.argsort(np.linalg.norm(mid - p, axis=1))[:4]
            if len(near) >= 2:
                best = 0.0
                for a in range(len(near)):
                    for b in range(a + 1, len(near)):
                        best = max(best, 1.0 - abs(float(d[near[a]] @ d[near[b]])))
                cross_angle[i] = best
            if edge_dirs:
                al = [max(abs(float(d[j] @ e)) for e in edge_dirs) for j in near]
                edge_align[i] = float(np.mean(al)) if al else 0.0

    def rank(v, higher_is_better):
        order = np.argsort(v if higher_is_better else -v)
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v))
        return r / max(len(v) - 1, 1)

    score = (rank(dist, False) + rank(resp, True) +
             rank(cross_angle, True) + rank(edge_align, True)) / 4.0
    return cands[int(np.argmax(score))], float(score.max())


def main() -> int:
    lock = json.loads(M.METHOD_LOCK_PATH.read_text())
    gb = lock["gate_b"]
    radius = int(gb["search_radius_px"])
    cfg = gb["candidate_generators_frozen"]

    gts = [M.load_gt(f) for f in M.dev_eval_frames()]
    preds = M.load_prediction_file(M.PREDICTIONS / f"{REF}.json")

    rec = {"visible": [], "occluded": []}
    cand_counts = []
    for gt in gts:
        pred = M.prediction_keypoints(preds.get(gt["frame_id"]))
        if pred is None:
            continue
        image = cv2.imread(str(M.REPO_ROOT / gt["image"]))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for k in range(8):
            if not gt["supervised"][k] or not np.isfinite(pred[k]).all():
                continue
            bucket = "visible" if gt["visible"][k] else "occluded"
            cands, info = collect(gray, pred[k], radius, cfg)
            cand_counts.append(len(cands))
            base_err = float(np.linalg.norm(pred[k] - gt["xy"][k]))
            row = {"kp": k, "session": gt["session_id"], "frame_id": gt["frame_id"],
                   "domain": gt["paper_domain"], "object_type": gt["object_type"],
                   "r0_err": base_err, "n_candidates": len(cands)}
            if cands:
                dists = np.array([np.linalg.norm(c["xy"] - gt["xy"][k]) for c in cands])
                row["oracle_err"] = float(dists.min())
                row["within3"] = bool(dists.min() <= 3)
                row["within5"] = bool(dists.min() <= 5)
                row["within10"] = bool(dists.min() <= 10)
                chosen, _ = selector_score(cands, info, pred[k],
                                           cuboid_edge_directions(pred, k))
                row["selector_err"] = float(np.linalg.norm(chosen["xy"] - gt["xy"][k]))
                row["selector_family"] = chosen["family"]
            else:
                row["oracle_err"] = base_err        # 후보 없으면 R0 유지
                row["within3"] = row["within5"] = row["within10"] = False
                row["selector_err"] = base_err
                row["selector_family"] = "abstain_no_candidate"
            rec[bucket].append(row)

    def block(rows):
        if not rows:
            return {"n": 0}
        r0 = M.error_stats([r["r0_err"] for r in rows])
        orc = M.error_stats([r["oracle_err"] for r in rows])
        sel = M.error_stats([r["selector_err"] for r in rows])
        return {
            "n": len(rows),
            "R0": r0, "ORACLE_CANDIDATE": orc, "PREDICTION_ONLY_SELECTOR": sel,
            "candidate_coverage": {
                "within_3px": float(np.mean([r["within3"] for r in rows])),
                "within_5px": float(np.mean([r["within5"] for r in rows])),
                "within_10px": float(np.mean([r["within10"] for r in rows])),
                "no_candidate_rate": float(np.mean([r["n_candidates"] == 0 for r in rows])),
                "candidates_per_patch_median": float(np.median([r["n_candidates"] for r in rows])),
            },
            "oracle_vs_r0": {
                "p90_relative_improvement_pct": (100 * (r0["p90_px"] - orc["p90_px"]) / r0["p90_px"]),
                "gross20_relative_reduction_pct": (100 * (r0["gross20"] - orc["gross20"]) / r0["gross20"]) if r0["gross20"] else None,
                "median_relative_improvement_pct": (100 * (r0["median_px"] - orc["median_px"]) / r0["median_px"]),
            },
            "selector_vs_r0": {
                "p90_relative_improvement_pct": (100 * (r0["p90_px"] - sel["p90_px"]) / r0["p90_px"]),
                "gross20_relative_reduction_pct": (100 * (r0["gross20"] - sel["gross20"]) / r0["gross20"]) if r0["gross20"] else None,
                "median_relative_improvement_pct": (100 * (r0["median_px"] - sel["median_px"]) / r0["median_px"]),
                "rescue_rate_on_r0_gross20": float(np.mean(
                    [r["selector_err"] <= 10 for r in rows if r["r0_err"] > 20])) if any(r["r0_err"] > 20 for r in rows) else None,
                "harm_rate_on_r0_good": float(np.mean(
                    [r["selector_err"] > 10 for r in rows if r["r0_err"] <= 10])) if any(r["r0_err"] <= 10 for r in rows) else None,
            },
            "by_corner": {str(k): {
                "n": sum(1 for r in rows if r["kp"] == k),
                "R0_median": M.error_stats([r["r0_err"] for r in rows if r["kp"] == k])["median_px"],
                "ORACLE_median": M.error_stats([r["oracle_err"] for r in rows if r["kp"] == k])["median_px"],
                "within_5px": float(np.mean([r["within5"] for r in rows if r["kp"] == k])) if any(r["kp"] == k for r in rows) else None,
            } for k in range(8)},
        }

    report = {"schema_version": "mtcd_gate_b_v1",
              "method_lock_sha256": M.sha256_file(M.METHOD_LOCK_PATH),
              "search_radius_px": radius,
              "candidate_generators": cfg,
              "new_training": 0,
              "PRIMARY_visible": block(rec["visible"]),
              "DIAGNOSTIC_occluded": block(rec["occluded"])}

    p = report["PRIMARY_visible"]
    ov = p["oracle_vs_r0"]
    crit = {
        "oracle_p90_improvement_pct": ov["p90_relative_improvement_pct"],
        "oracle_gross20_reduction_pct": ov["gross20_relative_reduction_pct"],
        "candidate_within_5px_coverage": p["candidate_coverage"]["within_5px"],
        "threshold_p90_pct": 15.0, "threshold_gross20_pct": 20.0,
        "threshold_coverage_5px": 0.60,
    }
    crit["headroom_metric_met"] = bool((ov["p90_relative_improvement_pct"] or 0) >= 15.0 or
                                       (ov["gross20_relative_reduction_pct"] or 0) >= 20.0)
    crit["coverage_met"] = bool(p["candidate_coverage"]["within_5px"] >= 0.60)
    crit["LOCAL_CORNER_HEADROOM"] = ("STRONG" if crit["headroom_metric_met"]
                                     and crit["coverage_met"] else "INSUFFICIENT")
    sv = p["selector_vs_r0"]
    crit["CLASSICAL_LOCAL_SELECTOR_SIGNAL"] = (
        "POSITIVE" if ((sv["p90_relative_improvement_pct"] or 0) > 0 and
                       (sv["median_relative_improvement_pct"] or 0) >= 0)
        else "ORACLE_ONLY_HEADROOM")
    report["verdict"] = crit

    out = M.GATE_B / "GATE_B_RESULT.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float) + "\n")
    print(json.dumps(crit, indent=2, ensure_ascii=False, default=float))
    print(f"\n-> {out.relative_to(M.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
