"""D3~D7 — 저장된 예측만으로 오프라인 분해.  새 추론 0.

D3 keypoint semantic permutation   역할 혼동인지
D4 keypoint subset PnP             나쁜 점 하나가 전체를 오염시키는지
D5 solver sanity                   solver 취약성인지 점 오차인지
D6 GT / camera calibration         평가 자체의 노이즈 바닥
D7 confidence calibration          catastrophic pose 를 사전에 버릴 수 있나
"""
from __future__ import annotations

import itertools, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0/model_compare", "scripts/stage0/real_eval",
            "scripts/annotate", "challenge"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import cv2                    # noqa: E402
import mc_geom as MG          # noqa: E402
import re_metrics as RM       # noqa: E402

OUT = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis_pre_v2")
PIPE = os.path.join(ROOT, "challenge/yolo_pose_one_model/paper_generic_pipeline")
DUMP = os.path.join(ROOT, "data/pallet/results/model_compare")
MODEL = "yolo26n_paper_generic_v1"
GROSS_R = 10.0

# camera-facing 0123: 0-3 near(0 TL,1 TR,2 BR,3 BL), 4-7 far 동일 순서
PERMS = {
    "identity": [0, 1, 2, 3, 4, 5, 6, 7],
    "lr_swap": [1, 0, 3, 2, 5, 4, 7, 6],
    "near_far_swap": [4, 5, 6, 7, 0, 1, 2, 3],
    "near_far_lr": [5, 4, 7, 6, 1, 0, 3, 2],
    "top_bottom": [3, 2, 1, 0, 7, 6, 5, 4],
    "rot180_face": [2, 3, 0, 1, 6, 7, 4, 5],
}
SOLVERS = {"SQPNP": (cv2.SOLVEPNP_SQPNP, False),
           "SQPNP_refineLM": (cv2.SOLVEPNP_SQPNP, True),
           "ITERATIVE": (cv2.SOLVEPNP_ITERATIVE, False),
           "EPNP_refineLM": (cv2.SOLVEPNP_EPNP, True)}


def solve_with(points, truth, flag, refine):
    ok = np.isfinite(points).all(1)
    if int(ok.sum()) < 4:
        return None
    op = truth["model"][ok].astype(np.float64)
    ip = points[ok].astype(np.float64)
    try:
        good, rvec, tvec = cv2.solvePnP(op, ip, truth["K"], None, flags=flag)
    except cv2.error:
        return None
    if not good:
        return None
    if refine:
        rvec, tvec = cv2.solvePnPRefineLM(op, ip, truth["K"], None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def err(pose, truth):
    if pose is None:
        return np.nan, np.nan
    return RM.pose_error(pose[0], pose[1], truth["R"], truth["t"])


def main():
    man = {i["frame_id"]: i for i in
           json.load(open(os.path.join(PIPE, "eval_manifest.json")))["items"]}
    dump = json.load(open(os.path.join(DUMP, f"kps_{MODEL}.json")))
    truths, gt2d = {}, {}
    for fid, m in man.items():
        truths[fid] = {"R": np.asarray(m["R_gt"], float),
                       "t": np.asarray(m["t_gt"], float),
                       "K": np.asarray(m["K"], float),
                       "model": np.asarray(m["object_points"], float),
                       "gt8": np.asarray(m["gt_corners_2d"], float),
                       "extents": (m["dimensions_m"]["width"],
                                   m["dimensions_m"]["height"],
                                   m["dimensions_m"]["depth"]),
                       "set": m["set"], "population": m["population"]}

    # ---------------- D6 : GT / K audit (먼저. 평가 노이즈 바닥) -------------
    d6 = {"note": "GT pose 로 8 코너를 재투영해 사람이 클릭한 GT 2D 와 비교. "
                  "이 값이 곧 평가의 노이즈 바닥이다.", "per_set": {}}
    bys = {}
    for fid, t in truths.items():
        cam = (t["R"] @ t["model"].T).T + t["t"]
        z = np.clip(cam[:, 2], 1e-6, None)
        proj = (t["K"] @ (cam/z[:, None]).T).T[:, :2]
        res = np.linalg.norm(proj - t["gt8"], axis=1)
        pose = solve_with(t["gt8"], t, cv2.SOLVEPNP_SQPNP, True)
        dr, dt = err(pose, t)
        bys.setdefault(t["set"], []).append((float(np.median(res)),
                                             float(np.percentile(res, 90)),
                                             dr, dt))
    for s, rows in bys.items():
        a = np.array(rows, float)
        d6["per_set"][s] = {
            "n": len(rows),
            "reproj_median_px": round(float(np.median(a[:, 0])), 3),
            "reproj_p90_px": round(float(np.median(a[:, 1])), 3),
            "gt2d_to_pose_R_disagree_deg": round(float(np.nanmedian(a[:, 2])), 4),
            "gt2d_to_pose_t_disagree_m": round(float(np.nanmedian(a[:, 3])), 5)}
    json.dump(d6, open(os.path.join(OUT, "GT_CAMERA_CALIBRATION_AUDIT.json"), "w"),
              indent=1, ensure_ascii=False)

    # ---------------- D6D : dimension sensitivity --------------------------
    d6d = {"note": "GT 2D 점을 그대로 두고 3D 치수만 흔들어 pose 민감도를 본다. "
                   "translation 오차 바닥을 알기 위한 것.", "perturbations": {}}
    for pct in (0.01, 0.02, 0.05):
        drs, dts = [], []
        for fid, t in list(truths.items())[:80]:
            scaled = dict(t); scaled["model"] = t["model"] * (1 + pct)
            pose = solve_with(t["gt8"], scaled, cv2.SOLVEPNP_SQPNP, True)
            dr, dt = err(pose, t)
            drs.append(dr); dts.append(dt)
        d6d["perturbations"][f"+{int(pct*100)}%"] = {
            "R_median_deg": round(float(np.nanmedian(drs)), 4),
            "t_median_m": round(float(np.nanmedian(dts)), 5)}
    json.dump(d6d, open(os.path.join(OUT, "DIMENSION_SENSITIVITY.json"), "w"),
              indent=1, ensure_ascii=False)

    # ---------------- D3/D4/D5/D7 : 예측 기반 ------------------------------
    perm_rows, subset_rows, solver_rows, calib_rows = [], [], [], []
    for e in dump["frames"]:
        fid = e["fid"]
        t = truths[fid]
        px = MG.points_of(e, MODEL)
        if not np.isfinite(px).all():
            continue
        base = solve_with(px, t, cv2.SOLVEPNP_SQPNP, True)
        bR, bt = err(base, t)

        # D3 permutation (진단 전용)
        pr = {"fid": fid, "set": t["set"], "population": t["population"],
              "identity_R": bR}
        for name, order in PERMS.items():
            p = solve_with(px[order], t, cv2.SOLVEPNP_SQPNP, True)
            r, _ = err(p, t)
            pr[f"{name}_R"] = r
            pr[f"{name}_corner"] = float(np.median(
                np.linalg.norm(px[order] - t["gt8"], axis=1)))
        perm_rows.append(pr)

        # D4 subset — confidence 상위 K
        kc = e.get("kp_conf") or [1.0]*9
        conf = np.array([kc[i] if kc[i] is not None else 0.0 for i in range(8)])
        sr = {"fid": fid, "set": t["set"], "population": t["population"],
              "P0_all8_R": bR, "P0_all8_t": bt}
        for k in (7, 6, 5, 4):
            keep = np.argsort(-conf)[:k]
            sub = np.full((8, 2), np.nan); sub[keep] = px[keep]
            mask = np.zeros(8, bool); mask[keep] = True
            tt = dict(t); tt["model"] = t["model"]
            p = solve_with(np.where(mask[:, None], px, np.nan), tt,
                           cv2.SOLVEPNP_SQPNP, True)
            r, tr = err(p, t)
            sr[f"top{k}_R"] = r; sr[f"top{k}_t"] = tr
        for i in range(8):
            mask = np.ones(8, bool); mask[i] = False
            p = solve_with(np.where(mask[:, None], px, np.nan), t,
                           cv2.SOLVEPNP_SQPNP, True)
            r, _ = err(p, t)
            sr[f"loo{i}_R"] = r
        for name, idx in (("near_only", [0, 1, 2, 3]), ("far_only", [4, 5, 6, 7]),
                          ("top_only", [0, 1, 4, 5]), ("bottom_only", [2, 3, 6, 7])):
            mask = np.zeros(8, bool); mask[idx] = True
            p = solve_with(np.where(mask[:, None], px, np.nan), t,
                           cv2.SOLVEPNP_SQPNP, True)
            r, _ = err(p, t)
            sr[f"{name}_R"] = r
        subset_rows.append(sr)

        # D5 solver
        row = {"fid": fid, "population": t["population"]}
        for name, (flag, ref) in SOLVERS.items():
            r, _ = err(solve_with(px, t, flag, ref), t)
            row[f"pred_{name}_R"] = r
            rg, _ = err(solve_with(t["gt8"], t, flag, ref), t)
            row[f"gt_{name}_R"] = rg
        solver_rows.append(row)

        # D7 calibration
        cam = (base[0] @ t["model"].T).T + base[1] if base else None
        reproj = np.nan
        if cam is not None:
            z = np.clip(cam[:, 2], 1e-6, None)
            pj = (t["K"] @ (cam/z[:, None]).T).T[:, :2]
            reproj = float(np.median(np.linalg.norm(pj - px, axis=1)))
        calib_rows.append({
            "fid": fid, "population": t["population"], "set": t["set"],
            "box_conf": e.get("box_conf"),
            "kp_conf_mean": float(conf.mean()), "kp_conf_min": float(conf.min()),
            "kp_conf_4th_lowest": float(np.sort(conf)[3]),
            "kp_spread": float(np.std(px, axis=0).mean()),
            "pnp_reproj": reproj, "R": bR, "t": bt,
            "success_5cm5": int(np.isfinite(bR) and bR <= 5 and bt <= 0.05),
            "corner_med": float(np.median(np.linalg.norm(px - t["gt8"], axis=1)))})

    # ---- D3 집계
    gross = [r for r in perm_rows if np.isfinite(r["identity_R"])
             and r["identity_R"] > GROSS_R]
    best = {}
    for r in gross:
        cand = {n: r[f"{n}_R"] for n in PERMS if np.isfinite(r[f"{n}_R"])}
        if cand:
            best[min(cand, key=cand.get)] = best.get(min(cand, key=cand.get), 0)+1
    d3 = {"note": "oracle permutation 은 진단 전용. main metric 으로 쓰지 않는다.",
          "gross_threshold_R_deg": GROSS_R,
          "n_solved": len(perm_rows), "n_gross": len(gross),
          "best_permutation_counts": best,
          "identity_best_rate": round(best.get("identity", 0)/max(len(gross), 1), 4),
          "role_confusion_rate": round(
              1 - best.get("identity", 0)/max(len(gross), 1), 4)}
    json.dump({**d3, "rows": perm_rows},
              open(os.path.join(OUT, "KEYPOINT_PERMUTATION_AUDIT.json"), "w"),
              indent=1, default=str)

    # ---- D4 집계
    def med(rows, k):
        v = np.array([r.get(k, np.nan) for r in rows], float)
        v = v[np.isfinite(v)]
        return round(float(np.median(v)), 3) if v.size else None
    d4 = {"note": "저장된 예측 재사용. DEV 에서 최적 subset 을 찾더라도 FINAL rule 은 "
                  "새 REAL_TEST 전에 freeze 해야 한다.", "populations": {}}
    for pop in ("REAL_DEV_OPEN_56", "REAL_CHALLENGE_DEV_105"):
        sub = [r for r in subset_rows if r["population"] == pop]
        d4["populations"][pop] = {
            "n": len(sub),
            "arms": {k: med(sub, k) for k in
                     ("P0_all8_R", "top7_R", "top6_R", "top5_R", "top4_R",
                      "near_only_R", "far_only_R", "top_only_R", "bottom_only_R")},
            "leave_one_out_R": {f"drop_kp{i}": med(sub, f"loo{i}_R")
                                for i in range(8)}}
    json.dump({**d4, "rows": subset_rows},
              open(os.path.join(OUT, "KEYPOINT_SUBSET_PNP.json"), "w"),
              indent=1, default=str)

    # ---- D5 집계
    d5 = {"note": "GT 점에서 solver 간 차이가 크면 solver/evaluator 문제, "
                  "GT 점에서 모두 정확하면 point prediction 문제.",
          "predicted_points": {k: med(solver_rows, f"pred_{k}_R") for k in SOLVERS},
          "gt_points": {k: med(solver_rows, f"gt_{k}_R") for k in SOLVERS}}
    json.dump(d5, open(os.path.join(OUT, "PNP_SOLVER_SANITY.json"), "w"),
              indent=1, ensure_ascii=False)

    # ---- D7 집계
    from scipy.stats import spearmanr
    feats = ["box_conf", "kp_conf_mean", "kp_conf_min", "kp_conf_4th_lowest",
             "pnp_reproj", "kp_spread"]
    ch = [r for r in calib_rows if r["population"] == "REAL_CHALLENGE_DEV_105"]
    d7 = {"note": "real negative 가 없어 threshold 를 확정하지 않는다. "
                  "selective prediction 잠재력만 본다.",
          "spearman_vs_R": {}, "coverage_curve": {}}
    for f in feats:
        x = np.array([r[f] if r[f] is not None else np.nan for r in calib_rows], float)
        y = np.array([r["R"] for r in calib_rows], float)
        g = np.isfinite(x) & np.isfinite(y)
        if g.sum() > 5:
            d7["spearman_vs_R"][f] = round(float(spearmanr(x[g], y[g]).statistic), 4)
    for f in ("pnp_reproj", "kp_conf_4th_lowest", "box_conf"):
        curve = []
        x = np.array([r[f] if r[f] is not None else np.nan for r in ch], float)
        good = np.isfinite(x)
        order = np.argsort(x[good]) if f == "pnp_reproj" else np.argsort(-x[good])
        idx = np.arange(len(ch))[good][order]
        for cov in (1.0, 0.9, 0.8, 0.7, 0.5):
            keep = idx[:max(1, int(len(idx)*cov))]
            R = np.array([ch[i]["R"] for i in keep], float)
            s5 = np.mean([ch[i]["success_5cm5"] for i in keep])
            curve.append({"coverage": cov, "n": len(keep),
                          "R_median": round(float(np.nanmedian(R)), 3),
                          "success_5cm5_within_kept": round(float(s5), 4)})
        d7["coverage_curve"][f] = curve
    json.dump({**d7, "rows": calib_rows},
              open(os.path.join(OUT, "CONFIDENCE_ERROR_CALIBRATION.json"), "w"),
              indent=1, default=str)

    print("=== D6 GT/K audit (평가 노이즈 바닥) ===")
    for s, v in d6["per_set"].items():
        print(f"  {s.replace('eval_',''):12} n={v['n']:>3} reproj med "
              f"{v['reproj_median_px']:>7.2f}px p90 {v['reproj_p90_px']:>8.2f}px  "
              f"GT2D->pose R {v['gt2d_to_pose_R_disagree_deg']:>7.3f}deg")
    print("\n=== D6D 치수 민감도 ===")
    for k, v in d6d["perturbations"].items():
        print(f"  {k:>6}  R {v['R_median_deg']:>7.4f}deg  t {v['t_median_m']:>8.5f}m")
    print(f"\n=== D3 permutation (gross R>{GROSS_R}deg, n={d3['n_gross']}) ===")
    print(f"  best permutation: {d3['best_permutation_counts']}")
    print(f"  identity 가 최선인 비율 {d3['identity_best_rate']:.1%}  "
          f"-> role confusion {d3['role_confusion_rate']:.1%}")
    print("\n=== D4 subset (R median) ===")
    for pop, v in d4["populations"].items():
        print(f"  {pop} n={v['n']}")
        print(f"    {v['arms']}")
    print("\n=== D5 solver ===")
    print(f"  예측 점: {d5['predicted_points']}")
    print(f"  GT  점: {d5['gt_points']}")
    print("\n=== D7 상관 (vs R 오차) ===")
    print(f"  {d7['spearman_vs_R']}")
    for f, c in d7["coverage_curve"].items():
        print(f"  {f}: " + "  ".join(
            f"cov{e['coverage']:.0%} R{e['R_median']}/5cm5 {e['success_5cm5_within_kept']:.2f}"
            for e in c))


if __name__ == "__main__":
    main()
