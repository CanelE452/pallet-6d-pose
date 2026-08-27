"""개정 지표(2026-08-26)로 모델을 한 표에 놓는다 — 5cm5deg 는 싣지 않는다.

`mc_geom` 이 프레임별 기하를 푸는 코드는 그대로 쓴다.  바뀐 것은 **집계**뿐이다:
headline 이 `success_5cm5deg` 에서 `ADD / ADD-S AUC` 로 옮겨졌다
(`metric_split_lock.md` §2.3 [개정 2026-08-26]).

AUC 는 프레임마다 자기 GT diameter 로 오차를 정규화한 뒤 [0, 0.1] 구간에서 잰다.
파렛트는 W/D 가 프레임마다 스왑돼 라벨되므로 diameter 를 상수로 두면 그 스왑이
AUC 에 새어든다.

PnP 를 못 푼 프레임은 **버리지 않고 오차 무한대로 센다**(unconditional).  분모를
바꾸면 검출이 나쁜 모델이 유리해진다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0/real_eval",):
    sys.path.insert(0, os.path.join(ROOT, sub))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re_metrics as RM          # noqa: E402
import mc_geom as MG             # noqa: E402
import mc_frames as MF           # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
MAX_FRACTION = 0.1


def per_frame(model, rows_by_fid):
    """[(sealed, row, diameter)] — 프레임 전수. 미검출도 행을 남긴다."""
    out = []
    for key, sealed, jp, _ip, label in MF.frames():
        fid = os.path.splitext(os.path.basename(jp))[0]
        entry = rows_by_fid.get((key, fid))
        if entry is None:
            continue
        truth = MG.gt_of(label)
        points = MG.points_of(entry, model)
        row = MG.metrics(points, truth)
        pose = MG.solve(points, truth) if row["pnp_ok"] else None
        row["yaw"] = (RM.yaw_error(pose[0], truth["R"])
                      if pose is not None else np.nan)
        out.append((key, sealed, row, RM.model_diameter(truth["model"])))
    return out


def stat(values, field="median"):
    v = np.array(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return round(float(np.median(v) if field == "median"
                       else np.percentile(v, 90)), 4)


def aggregate(rows):
    """rows = [(row, diameter)].  분모는 전수다."""
    total = len(rows)
    if total == 0:
        return None
    solved = [r for r, _ in rows if r["pnp_ok"]]
    # AUC — 미해결 프레임은 정규화 오차 무한대로 남긴다
    add_n, adds_n = [], []
    for r, dia in rows:
        if r["pnp_ok"] and np.isfinite(r["add"]):
            add_n.append(r["add"] / dia)
            adds_n.append(r["adds"] / dia)
        else:
            add_n.append(np.inf)
            adds_n.append(np.inf)
    return {
        "n": total,
        "pnp_rate": round(len(solved) / total, 4),
        "corner_px_med": stat([r["corner_med"] for r, _ in rows]),
        "R_deg_med": stat([r["R"] for r, _ in rows]),
        "R_deg_p90": stat([r["R"] for r, _ in rows], "p90"),
        "t_m_med": stat([r["t"] for r, _ in rows]),
        "yaw_deg_med": stat([r["yaw"] for r, _ in rows]),
        "yaw_deg_p90": stat([r["yaw"] for r, _ in rows], "p90"),
        "ADD_med": stat([r["add"] for r, _ in rows]),
        "ADD_S_med": stat([r["adds"] for r, _ in rows]),
        "ADD_AUC": round(RM.pose_auc(add_n, 1.0, MAX_FRACTION), 4),
        "ADD_S_AUC": round(RM.pose_auc(adds_n, 1.0, MAX_FRACTION), 4),
        "IoU3D_med": stat([r["iou"] for r, _ in rows]),
    }


def detection_ap():
    """{model: {AP, AUROC, FPR@TPR95}} — 없으면 빈 dict."""
    path = os.path.join(OUT, "AP_SCORES.json")
    if not os.path.exists(path):
        return {}
    payload = json.load(open(path))
    out = {}
    for name, d in payload["models"].items():
        scores = np.array(d["pos"] + d["neg"], float)
        labels = np.array([1] * len(d["pos"]) + [0] * len(d["neg"]), int)
        pos, neg = np.array(d["pos"], float), np.array(d["neg"], float)
        # AUROC = P(양성 점수 > 음성 점수), 동점은 0.5
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), float)
        ranks[order] = np.arange(1, len(scores) + 1)
        # 동점 평균순위
        for value in np.unique(scores):
            hit = scores == value
            if hit.sum() > 1:
                ranks[hit] = ranks[hit].mean()
        auroc = (ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) \
            / (len(pos) * len(neg))
        tau = float(np.percentile(pos, 5))          # TPR 95% 를 주는 임계
        out[name] = {
            "AP": round(RM.average_precision(scores, labels), 4),
            "AUROC": round(float(auroc), 4),
            "FPR_at_TPR95": round(float((neg >= tau).mean()), 4),
            "n_pos": len(pos), "n_neg": len(neg)}
    return out


def main(models):
    report = {"metric_revision": "2026-08-26 — AUC replaces 5cm5deg",
              "contract": {
                  "points": "8 corners only (centroid excluded)",
                  "min_points": MG.MIN_POINTS,
                  "solver": "SOLVEPNP_SQPNP -> solvePnPRefineLM",
                  "confidence_filter": "NONE",
                  "auc": f"per-frame diameter-normalised, 0 ~ {MAX_FRACTION} d",
                  "unsolved_frames": "counted as infinite error (unconditional)"},
              "models": {}}
    ap = detection_ap()
    report["detection"] = {
        "population": "positive = canonical 161, negative = real 2,689",
        "score": "max box confidence, conf 0.001 (threshold-free)",
        "excluded": "FINAL40K — score_4kp 는 box conf 와 의미가 다르다",
        "per_model": ap}
    for name in models:
        path = os.path.join(OUT, f"kps_{name}.json")
        if not os.path.exists(path):
            print(f"  {name}: 덤프 없음 -> 건너뜀", flush=True)
            continue
        payload = json.load(open(path))
        by_fid = {(e["set"], e["fid"]): e for e in payload["frames"]}
        rows = per_frame(name, by_fid)
        report["models"][name] = {
            "weights": payload.get("weights"),
            "OPEN_56": aggregate([(r, d) for _k, s, r, d in rows if not s]),
            "SEALED_105": aggregate([(r, d) for _k, s, r, d in rows if s]),
            "ALL_161": aggregate([(r, d) for _k, _s, r, d in rows])}
        a = report["models"][name]["ALL_161"]
        print(f"  {name:24} ADD_AUC {a['ADD_AUC']:.4f}  "
              f"ADD_S_AUC {a['ADD_S_AUC']:.4f}  pnp {a['pnp_rate']:.3f}",
              flush=True)
    target = os.path.join(OUT, "MODEL_COMPARE_AUC.json")
    json.dump(report, open(target, "w"), indent=1)
    print(f"-> {target}", flush=True)
    return report


if __name__ == "__main__":
    main(sys.argv[1:] or ["yolo26n_synth", "yolo26n_ft", "yolo26m_ft",
                          "yolo26n_paper_generic_v1", "yolo26n_broad40k_5ep",
                          "Y0E", "YN", "FINAL40K_seed1"])
