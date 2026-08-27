"""PHASE 1 — 조건부 Hough 의 **oracle headroom**.

질문: 완벽한 gate 가 존재한다고 가정해도, point 가 불안정한 프레임에서만 Hough/F3
를 켜는 것이 값어치가 있는가.

★ oracle 은 deployment 결과가 아니다.  gate 를 만들 가치가 있는지 **상한**을 재는
것이다.  상한이 작으면 gate 를 만드는 실험 자체를 하지 않는다.

재추론 0 — 저장된 YOLO 60ep 키포인트와 FINAL 의 line theta/rho 만 쓴다.
"""
from __future__ import annotations

import csv, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0", "scripts/stage0/multihead", "scripts/stage0/line",
            "scripts/stage0/model_compare", "scripts/stage0/real_eval",
            "scripts/annotate", "challenge"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import cv2                    # noqa: E402
import mc_geom as MG          # noqa: E402
import mc_hough as MH         # noqa: E402
import mh_fusion as FU        # noqa: E402
import re_metrics as RM       # noqa: E402

OUT = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis_pre_v2")
PIPE = os.path.join(ROOT, "challenge/yolo_pose_one_model/paper_generic_pipeline")
DUMP = os.path.join(ROOT, "data/pallet/results/model_compare")
YOLO = "yolo26n_paper_generic_v1"

# PHASE 1 STOP RULE — 결과 보기 전에 박는다
STOP = {"oracle_R_improve_min": 0.10, "oracle_5cm5_gain_min": 0.05,
        "hough_R_win_fraction_min": 0.25}


def main():
    man = {i["frame_id"]: i for i in
           json.load(open(os.path.join(PIPE, "eval_manifest.json")))["items"]}
    yolo = {e["fid"]: e for e in
            json.load(open(os.path.join(DUMP, f"kps_{YOLO}.json")))["frames"]}
    final = {e["fid"]: e for e in
             json.load(open(os.path.join(DUMP,
                                         "kps_FINAL40K_seed1.json")))["frames"]}
    weight = json.loads(open(os.path.join(
        ROOT, "data/pallet/results/paper_s2_multihead",
        "theta_posealigned_d0.json")).read())["seeds"]["seed1"]["selected_lambda_theta"]

    rows = []
    for fid, m in man.items():
        e = yolo[fid]
        px = MG.points_of(e, YOLO)
        if not np.isfinite(px).all():
            continue                       # PnP 자체가 없으면 F3 도 못 돈다
        image = cv2.imread(os.path.join(ROOT, m["image"]))
        h, w = image.shape[:2]
        truth = {"R": np.asarray(m["R_gt"], float),
                 "t": np.asarray(m["t_gt"], float),
                 "K": np.asarray(m["K"], float),
                 "model": np.asarray(m["object_points"], float),
                 "gt8": np.asarray(m["gt_corners_2d"], float),
                 "extents": (m["dimensions_m"]["width"],
                             m["dimensions_m"]["height"],
                             m["dimensions_m"]["depth"])}
        # Y0 — point only
        p0 = MG.solve(px, truth)
        if p0 is None:
            continue
        r0, t0 = RM.pose_error(p0[0], p0[1], truth["R"], truth["t"])
        # YH — 같은 점 + FINAL 의 예측 line, support 는 예측 점에서
        grid9 = np.vstack([MH.to_grid(px, w, h), [[0.0, 0.0]]])
        sup = MH.gt_lines_from(grid9)[2]
        data = {"resolution": np.array([[w, h]]),
                "model": np.array([truth["model"]]), "K": np.array([truth["K"]]),
                "pred_corner": np.array([grid9]),
                "pred_theta": np.array([np.asarray(final[fid]["line_theta"], float)]),
                "pred_rho": np.array([np.asarray(final[fid]["line_rho"], float)]),
                "support": np.array([sup])}
        arms, _, _, _ = FU.solve_arms(data, 0, weight)
        ph = arms.get("F3")
        rh, th = (RM.pose_error(ph[0], ph[1], truth["R"], truth["t"])
                  if ph is not None else (np.nan, np.nan))
        # 품질 신호 (gate 후보, GT 미사용)
        cam = (p0[0] @ truth["model"].T).T + p0[1]
        z = np.clip(cam[:, 2], 1e-6, None)
        proj = (truth["K"] @ (cam/z[:, None]).T).T[:, :2]
        reproj = float(np.median(np.linalg.norm(proj - px, axis=1)))
        kc = e.get("kp_conf") or [1.0]*9
        conf = np.array([kc[i] if kc[i] is not None else 0.0 for i in range(8)])

        def pack(pose, r, t):
            if pose is None:
                return {"R": np.nan, "t": np.nan, "adds": np.nan,
                        "iou": np.nan, "s5": 0}
            return {"R": r, "t": t,
                    "adds": RM.add_s(truth["model"], pose[0], pose[1],
                                     truth["R"], truth["t"]),
                    "iou": RM.iou_3d(pose[0], pose[1], truth["extents"],
                                     truth["R"], truth["t"], truth["extents"]),
                    "s5": int(RM.success_5cm5deg(pose[0], pose[1],
                                                 truth["R"], truth["t"]))}
        a0, ah = pack(p0, r0, t0), pack(ph, rh, th)
        # ORACLE — GT 를 보고 고른다. deployment 아님
        pick_r = "YH" if (np.isfinite(ah["R"]) and ah["R"] < a0["R"]) else "Y0"
        pick_5 = "YH" if (ah["s5"] > a0["s5"]) else "Y0"
        best = ah if pick_r == "YH" else a0
        best5 = ah if pick_5 == "YH" else a0
        rows.append({"fid": fid, "set": m["set"], "population": m["population"],
                     "reproj": reproj, "box_conf": e.get("box_conf"),
                     "kp_conf_mean": float(conf.mean()),
                     "kp_conf_4th": float(np.sort(conf)[3]),
                     **{f"Y0_{k}": v for k, v in a0.items()},
                     **{f"YH_{k}": v for k, v in ah.items()},
                     "ORACLE_R": best["R"], "ORACLE_t": best["t"],
                     "ORACLE_adds": best["adds"], "ORACLE_iou": best["iou"],
                     "ORACLE_s5": best5["s5"],
                     "hough_wins_R": int(pick_r == "YH"),
                     "hough_wins_5cm5": int(ah["s5"] > a0["s5"])})

    with open(os.path.join(OUT, "CONDITIONAL_HOUGH_PER_FRAME.csv"), "w",
              newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0])); wtr.writeheader()
        wtr.writerows(rows)

    def agg(sub, pre):
        R = np.array([r[f"{pre}_R"] for r in sub], float)
        t = np.array([r[f"{pre}_t"] for r in sub], float)
        g = np.isfinite(R)
        return {"n": len(sub),
                "R_median": round(float(np.median(R[g])), 3) if g.any() else None,
                "R_p90": round(float(np.percentile(R[g], 90)), 3) if g.any() else None,
                "t_median": round(float(np.nanmedian(t)), 4),
                "ADD_S": round(float(np.nanmedian(
                    [r[f"{pre}_adds"] for r in sub])), 4),
                "IoU": round(float(np.nanmedian(
                    [r[f"{pre}_iou"] for r in sub])), 4),
                "success_5cm5": round(float(np.mean(
                    [r[f"{pre}_s5"] for r in sub])), 4)}

    report = {"note": "oracle 은 deployment 결과가 아니다. gate 를 만들 가치가 "
                      "있는지 상한을 재는 것이다.",
              "inference": "재추론 0. YOLO 60ep 키포인트 + FINAL line theta/rho 재사용",
              "yolo": YOLO, "stop_rule": STOP, "populations": {}}
    for pop in ("REAL_DEV_OPEN_56", "REAL_CHALLENGE_DEV_105"):
        sub = [r for r in rows if r["population"] == pop]
        if not sub:
            continue
        blk = {a: agg(sub, a) for a in ("Y0", "YH", "ORACLE")}
        blk["hough_R_win_fraction"] = round(float(np.mean(
            [r["hough_wins_R"] for r in sub])), 4)
        blk["hough_5cm5_win_fraction"] = round(float(np.mean(
            [r["hough_wins_5cm5"] for r in sub])), 4)
        y0R, orR = blk["Y0"]["R_median"], blk["ORACLE"]["R_median"]
        blk["oracle_R_relative_improve"] = round((y0R - orR)/max(y0R, 1e-9), 4)
        blk["oracle_5cm5_gain_pp"] = round(
            blk["ORACLE"]["success_5cm5"] - blk["Y0"]["success_5cm5"], 4)
        report["populations"][pop] = blk

    ch = report["populations"]["REAL_CHALLENGE_DEV_105"]
    fail = (ch["oracle_R_relative_improve"] < STOP["oracle_R_improve_min"]
            and ch["oracle_5cm5_gain_pp"] < STOP["oracle_5cm5_gain_min"]
            and ch["hough_R_win_fraction"] < STOP["hough_R_win_fraction_min"])
    report["PHASE1_VERDICT"] = ("CONDITIONAL_HOUGH_HEADROOM_TOO_SMALL"
                                if fail else "HEADROOM_SUFFICIENT_PROCEED")
    report["HOUGH_TRACK"] = "CLOSED" if fail else "OPEN"

    # Hough 가 이기는 프레임의 성격
    wins = [r for r in rows if r["hough_wins_R"]]
    if wins:
        report["hough_win_frames_profile"] = {
            "n": len(wins),
            "Y0_R_median": round(float(np.nanmedian([r["Y0_R"] for r in wins])), 3),
            "reproj_median": round(float(np.nanmedian(
                [r["reproj"] for r in wins])), 3),
            "reproj_median_all": round(float(np.nanmedian(
                [r["reproj"] for r in rows])), 3),
            "box_conf_median": round(float(np.nanmedian(
                [r["box_conf"] for r in wins if r["box_conf"]])), 3),
            "by_set": {s: sum(1 for r in wins if r["set"] == s)
                       for s in sorted({r["set"] for r in wins})}}
    json.dump(report, open(os.path.join(OUT, "CONDITIONAL_HOUGH_ORACLE.json"),
                           "w"), indent=1, ensure_ascii=False)

    for pop, b in report["populations"].items():
        print(f"\n=== {pop} (n={b['Y0']['n']}) ===")
        print(f"{'arm':8}{'R med':>9}{'R p90':>9}{'t med':>9}{'ADD-S':>9}"
              f"{'IoU':>8}{'5cm5':>8}")
        for a in ("Y0", "YH", "ORACLE"):
            s = b[a]
            print(f"{a:8}{s['R_median']:>9.2f}{s['R_p90']:>9.2f}"
                  f"{s['t_median']:>9.4f}{s['ADD_S']:>9.4f}"
                  f"{s['IoU']:>8.3f}{s['success_5cm5']:>8.3f}")
        print(f"  Hough R 승률 {b['hough_R_win_fraction']:.3f}  "
              f"5cm5 승률 {b['hough_5cm5_win_fraction']:.3f}")
        print(f"  oracle R 개선 {b['oracle_R_relative_improve']:.1%}  "
              f"5cm5 이득 {b['oracle_5cm5_gain_pp']:+.3f}")
    print(f"\nPHASE1_VERDICT = {report['PHASE1_VERDICT']}")
    print(f"HOUGH_TRACK = {report['HOUGH_TRACK']}")
    if "hough_win_frames_profile" in report:
        print(f"  Hough 승리 프레임 성격: {report['hough_win_frames_profile']}")


if __name__ == "__main__":
    main()
