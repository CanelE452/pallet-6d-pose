"""point source x line source 매트릭스 — 학습 없이 hybrid 가치를 판정한다.

답하는 질문은 하나다: **robust 한 point estimator 위에서도 Direct-Hough + F3 가
독립적인 가치를 가지는가.**  이미 확인된 것(solver 정상, Hough 미붕괴, corner 가
가장 크게 무너짐, YOLO point 가 같은 BROAD 에서도 더 robust)은 다시 재지 않는다.

주 판정은 **B1 vs P1** — 둘 다 y_BROAD40K 의 점을 쓰고, target pallet real 을 본
적이 없는 조건이라 "강한 점 위에서 우리 Hough 가 뭘 더 주는가" 가 가장 깨끗하게 보인다.

계약 (전 arm 공통)
    8 cuboid corner only (centroid 제외)
    같은 K / 같은 dimensions / 같은 3D 점 순서
    최소 4 대응
    SQPnP -> refineLM   (Point PnP arm)
    F3 는 mh_fusion.solve_arms 정본 그대로
    ★ 신뢰도 threshold 로 점을 제거하지 않는다 — YOLO 의 visibility 와 belief peak
      은 같은 양이 아니다 (median 1.000 vs 0.423)
    ★ support 는 **예측 점**에서 만든다.  GT line/support 는 oracle arm 에만 쓴다.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0", "scripts/stage0/multihead", "scripts/stage0/line",
            "scripts/stage0/real_eval", "scripts/annotate", "challenge",
            "scripts/stage0/stage_screens"):
    sys.path.insert(0, os.path.join(ROOT, sub))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2                       # noqa: E402
import mc_frames as MF           # noqa: E402
import mc_geom as MG             # noqa: E402
import mc_hough as MH            # noqa: E402
import mh_fusion as FU           # noqa: E402
import re_metrics as RM          # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
POINT_SOURCES = {"B0/P0": "FINAL40K_seed1", "B1/P1": "yolo26n_broad40k_5ep",
                 "B2/P2": "yolo26n_synth", "B3/P3": "yolo26n_ft"}
BOOT, BOOT_SEED = 10000, 20260908

# ---- PHASE 6: 결과를 보기 전에 박는다 -------------------------------------
RULE = {
    "STRONG_PASS": {"R_rel_improve_min": 0.10, "t_median_degrade_max": 0.05,
                    "R_win_fraction_min": 0.55,
                    "challenge_CI_upper_below_zero": True},
    "FAIL": {"challenge_R_rel_improve_below": 0.05,
             "t_median_degrade_above": 0.10, "R_win_fraction_below": 0.50},
    "note": "사이면 WEAK_PASS. 결과 보고 threshold 변경 금지.",
}


def load_points():
    out = {}
    for tag, name in POINT_SOURCES.items():
        d = json.load(open(os.path.join(OUT, f"kps_{name}.json")))
        out[tag] = {e["fid"]: MG.points_of(e, name) for e in d["frames"]}
    return out


def main():
    final_line = {e["fid"]: e for e in json.load(
        open(os.path.join(OUT, "kps_FINAL40K_seed1.json")))["frames"]}
    weight = json.loads(open(os.path.join(
        ROOT, "data/pallet/results/paper_s2_multihead",
        "theta_posealigned_d0.json")).read())["seeds"]["seed1"]["selected_lambda_theta"]
    points = load_points()

    rows = []
    for key, sealed, jp, ip, label in MF.frames():
        fid = os.path.splitext(os.path.basename(jp))[0]
        image = cv2.imread(ip)
        height, width = image.shape[:2]
        truth = MG.gt_of(label)
        gt9 = MH.to_grid(np.vstack([truth["gt8"], np.asarray(
            label["objects"][0]["projected_cuboid_centroid"], float)]),
            width, height)
        gt_t, gt_r, gt_sup = MH.gt_lines_from(gt9)
        pred_t = np.asarray(final_line[fid]["line_theta"], float)
        pred_r = np.asarray(final_line[fid]["line_rho"], float)

        row = {"fid": fid, "set": key, "sealed": sealed}

        def record(arm, px, theta, rho, support, use_f3):
            ok = np.isfinite(px).all(1)
            row[f"{arm}_n"] = int(ok.sum())
            if ok.sum() < 4:
                for k in ("R", "t", "add", "adds", "iou", "corner"):
                    row[f"{arm}_{k}"] = np.nan
                row[f"{arm}_5cm5"] = 0
                row[f"{arm}_ok"] = 0
                return
            row[f"{arm}_corner"] = float(np.median(
                np.linalg.norm(px[ok] - truth["gt8"][ok], axis=1)))
            if not use_f3:
                pose = MG.solve(px, truth)
            else:
                if not ok.all():
                    pose = None
                else:
                    grid9 = np.vstack([MH.to_grid(px, width, height), [[0.0, 0.0]]])
                    data = {"resolution": np.array([[width, height]]),
                            "model": np.array([truth["model"]]),
                            "K": np.array([truth["K"]]),
                            "pred_corner": np.array([grid9]),
                            "pred_theta": np.array([theta]),
                            "pred_rho": np.array([rho]),
                            "support": np.array([support])}
                    arms, _, _, _ = FU.solve_arms(data, 0, weight)
                    pose = arms.get("F3")
            if pose is None:
                for k in ("R", "t", "add", "adds", "iou"):
                    row[f"{arm}_{k}"] = np.nan
                row[f"{arm}_5cm5"] = 0
                row[f"{arm}_ok"] = 0
                return
            R_p, t_p = pose
            deg, met = RM.pose_error(R_p, t_p, truth["R"], truth["t"])
            row.update({f"{arm}_ok": 1, f"{arm}_R": deg, f"{arm}_t": met,
                        f"{arm}_add": RM.add(truth["model"], R_p, t_p,
                                             truth["R"], truth["t"]),
                        f"{arm}_adds": RM.add_s(truth["model"], R_p, t_p,
                                                truth["R"], truth["t"]),
                        f"{arm}_iou": RM.iou_3d(R_p, t_p, truth["extents"],
                                                truth["R"], truth["t"],
                                                truth["extents"]),
                        f"{arm}_5cm5": int(RM.success_5cm5deg(
                            R_p, t_p, truth["R"], truth["t"]))})

        for tag in POINT_SOURCES:
            b, p = tag.split("/")
            px = points[tag][fid]
            # support 는 예측 점에서 — deployment-valid 경로
            sup = (MH.gt_lines_from(np.vstack([MH.to_grid(px, width, height),
                                               [[0.0, 0.0]]]))[2]
                   if np.isfinite(px).all() else gt_sup)
            record(b, px, None, None, None, use_f3=False)
            record(p, px, pred_t, pred_r, sup, use_f3=True)
        # oracle — diagnostic only
        record("O1", truth["gt8"], pred_t, pred_r, gt_sup, use_f3=True)
        record("O2", truth["gt8"], gt_t, gt_r, gt_sup, use_f3=True)
        rows.append(row)

    def stats(sub, arm, field):
        v = np.array([r.get(f"{arm}_{field}", np.nan) for r in sub], float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return {"n": int(v.size), "median": round(float(np.median(v)), 4),
                "mean": round(float(v.mean()), 4),
                "p90": round(float(np.percentile(v, 90)), 4)}

    def summarise(sub, arm):
        return {"available": round(float(np.mean(
                    [r.get(f"{arm}_ok", 0) for r in sub])), 4),
                "R_deg": stats(sub, arm, "R"), "t_m": stats(sub, arm, "t"),
                "ADD": stats(sub, arm, "add"), "ADD_S": stats(sub, arm, "adds"),
                "IoU3D": stats(sub, arm, "iou"),
                "corner_px": stats(sub, arm, "corner"),
                "success_5cm5deg": round(float(np.mean(
                    [r.get(f"{arm}_5cm5", 0) for r in sub])), 4)}

    def paired(sub, a, b, field, rng):
        pairs = [(r[f"{a}_{field}"], r[f"{b}_{field}"]) for r in sub
                 if np.isfinite(r.get(f"{a}_{field}", np.nan))
                 and np.isfinite(r.get(f"{b}_{field}", np.nan))]
        if len(pairs) < 5:
            return {"n_pairs": len(pairs)}
        x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
        diff = y - x
        idx = rng.integers(0, len(diff), (BOOT, len(diff)))
        boot = np.median(diff[idx], axis=1)
        lo, hi = (float(v) for v in np.quantile(boot, [0.025, 0.975]))
        return {"n_pairs": len(pairs),
                "base_median": round(float(np.median(x)), 4),
                "hybrid_median": round(float(np.median(y)), 4),
                "delta_median": round(float(np.median(diff)), 4),
                "rel_improve": round(float((np.median(x) - np.median(y))
                                           / max(abs(np.median(x)), 1e-9)), 4),
                "CI95": [round(lo, 4), round(hi, 4)],
                "CI_upper_below_zero": bool(hi < 0),
                "win_fraction": round(float(np.mean(y < x)), 4),
                "no_worse_fraction": round(float(np.mean(y <= x)), 4)}

    rng = np.random.default_rng(BOOT_SEED)
    pops = {"REAL_DEV_OPEN_56": [r for r in rows if not r["sealed"]],
            "REAL_CHALLENGE_DEV_105": [r for r in rows if r["sealed"]]}
    arms = [a for tag in POINT_SOURCES for a in tag.split("/")] + ["O1", "O2"]

    report = {"question": "robust point estimator 위에서도 Direct-Hough + F3 가 "
                          "독립적 가치를 가지는가",
              "primary": "B1 vs P1 (y_BROAD40K points, target-unseen)",
              "contract": {"points": "8 corners, centroid 제외",
                           "confidence_filter": "NONE",
                           "solver_base": "SQPnP -> refineLM",
                           "solver_hybrid": "mh_fusion.solve_arms F3 (정본)",
                           "support": "예측 점에서 생성 (O1/O2 만 GT)",
                           "point_order_audited": "index-wise/order-free 비율 "
                                                  "1.02~1.17 -> 순열 일치 확인"},
              "point_sources": POINT_SOURCES,
              "prelocked_rule": RULE,
              "populations": {}}

    for pop, sub in pops.items():
        block = {"n": len(sub), "arms": {a: summarise(sub, a) for a in arms},
                 "paired": {}}
        for tag in POINT_SOURCES:
            b, p = tag.split("/")
            block["paired"][f"{p} vs {b}"] = {
                f: paired(sub, b, p, f, rng) for f in ("R", "t", "adds", "iou")}
        report["populations"][pop] = block

    # ---- PHASE 6 판정 (B1 vs P1) -----------------------------------------
    o = report["populations"]["REAL_DEV_OPEN_56"]["paired"]["P1 vs B1"]
    c = report["populations"]["REAL_CHALLENGE_DEV_105"]["paired"]["P1 vs B1"]
    def deg(p):  # translation 악화율 (양수면 나빠짐)
        return (p["hybrid_median"] - p["base_median"]) / max(abs(p["base_median"]), 1e-9)
    strong = (o["R"]["rel_improve"] >= 0.10 and c["R"]["rel_improve"] >= 0.10
              and deg(o["t"]) <= 0.05 and deg(c["t"]) <= 0.05
              and o["R"]["win_fraction"] >= 0.55 and c["R"]["win_fraction"] >= 0.55
              and c["R"]["CI_upper_below_zero"])
    fail = (c["R"]["rel_improve"] < 0.05 or deg(c["t"]) > 0.10
            or c["R"]["win_fraction"] < 0.50)
    verdict = ("HYBRID_SUPPORTED" if strong else
               "HOUGH_INCREMENTAL_VALUE_NOT_ESTABLISHED" if fail else
               "HYBRID_PROMISING_NOT_ESTABLISHED")
    report["VERDICT"] = verdict
    report["verdict_inputs"] = {
        "open_R_rel_improve": o["R"]["rel_improve"],
        "challenge_R_rel_improve": c["R"]["rel_improve"],
        "open_t_degrade": round(deg(o["t"]), 4),
        "challenge_t_degrade": round(deg(c["t"]), 4),
        "open_R_win": o["R"]["win_fraction"],
        "challenge_R_win": c["R"]["win_fraction"],
        "challenge_CI": c["R"]["CI95"]}

    json.dump(report, open(os.path.join(OUT, "HYBRID_POINT_LINE_MATRIX.json"), "w"),
              indent=1, default=str)
    fields = sorted({k for r in rows for k in r})
    with open(os.path.join(OUT, "HYBRID_POINT_LINE_PER_FRAME.csv"), "w",
              newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=fields)
        wtr.writeheader()
        wtr.writerows(rows)

    for pop, blk in report["populations"].items():
        print(f"\n=== {pop} (n={blk['n']}) ===")
        print(f"{'arm':6}{'avail':>7}{'R med':>9}{'t med':>9}{'ADD-S':>9}"
              f"{'IoU':>7}{'5cm5':>8}")
        for a in arms:
            s = blk["arms"][a]
            r = s["R_deg"]["median"] if s["R_deg"] else float("nan")
            t = s["t_m"]["median"] if s["t_m"] else float("nan")
            ad = s["ADD_S"]["median"] if s["ADD_S"] else float("nan")
            iu = s["IoU3D"]["median"] if s["IoU3D"] else float("nan")
            print(f"{a:6}{s['available']:>7.3f}{r:>9.2f}{t:>9.4f}{ad:>9.4f}"
                  f"{iu:>7.3f}{s['success_5cm5deg']:>8.3f}")
        p = blk["paired"]["P1 vs B1"]["R"]
        print(f"  P1 vs B1  R: base {p.get('base_median')} -> {p.get('hybrid_median')}"
              f"  delta {p.get('delta_median')}  CI {p.get('CI95')}"
              f"  win {p.get('win_fraction')}")
    print(f"\nVERDICT = {verdict}")
    print("-> HYBRID_POINT_LINE_MATRIX.json / HYBRID_POINT_LINE_PER_FRAME.csv")


if __name__ == "__main__":
    main()
