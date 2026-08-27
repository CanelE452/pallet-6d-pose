"""PHASE 2~7 — conf sweep, threshold rule, metrics, recovered pose, verdict.

★ 아래 정의와 게이트는 **결과를 보기 전에** 고정됐다.  결과를 보고 0.95 등으로
바꾸지 않는다.

  availability(tau)     survivor 를 하나라도 가진 positive 프레임 비율
  recall(tau)           top-1 survivor 의 IoU>=0.5  <- **1차 지표, 배포가 실제로 쓰는 것**
  recall_available(tau) survivor 중 아무거나 IoU>=0.5  <- 진단용 천장
  tau*                  FP/image 최소 s.t. recall >= 0.98
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import glob as _glob
for _p in ("/usr/share/fonts/**/NanumGothic.ttf",
           "/usr/share/fonts/**/NotoSansCJK*.ttc"):
    _h = _glob.glob(_p, recursive=True)
    if _h:
        fm.fontManager.addfont(_h[0])
        plt.rcParams["font.family"] = fm.FontProperties(fname=_h[0]).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0/model_compare", "scripts/stage0/real_eval",
            "scripts/annotate", "challenge"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import cv2                    # noqa: E402
import re_metrics as RM       # noqa: E402

A2 = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis_pre_v2")
PL = os.path.join(A2, "plots")
PIPE = os.path.join(ROOT, "challenge/yolo_pose_one_model/paper_generic_pipeline")
CONFS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
IOU_MATCH, RECALL_TARGET, CURRENT = 0.5, 0.98, 0.40
GROSS_R = 10.0                       # PHASE 5 catastrophic 기준 (D3 와 동일)


def bbox_of(pts, w, h):
    """8 corner -> axis-aligned bbox, 이미지 경계로 clip.
    pred 와 GT 에 **똑같이** 적용한다 — 한쪽만 clip 하면 IoU 가 편향된다."""
    p = np.asarray(pts, float)
    x0, y0 = float(p[:, 0].min()), float(p[:, 1].min())
    x1, y1 = float(p[:, 0].max()), float(p[:, 1].max())
    return [max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)]


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(0.0, a[2]-a[0])*max(0.0, a[3]-a[1]) + \
        max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pose_of(kps, truth):
    px = np.asarray(kps, float)[:8]
    if not np.isfinite(px).all():
        return None
    ok, rvec, tvec = cv2.solvePnP(truth["model"][:8], px.reshape(-1, 1, 2),
                                  truth["K"], None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(truth["model"][:8], px.reshape(-1, 1, 2),
                                      truth["K"], None, rvec, tvec)
    return cv2.Rodrigues(rvec)[0], tvec.reshape(3)


def main():
    raw = json.load(open(os.path.join(A2, "_cc_raw_dump.json")))
    man = {i["frame_id"]: i for i in
           json.load(open(os.path.join(PIPE, "eval_manifest.json")))["items"]}

    # ---- positive: 프레임마다 후보별 IoU / pose 를 한 번만 계산 ----
    pos = []
    for e in raw["positive"]:
        m = man[e["fid"]]
        im = cv2.imread(os.path.join(ROOT, m["image"]))
        h, w = im.shape[:2]
        truth = {"R": np.asarray(m["R_gt"], float),
                 "t": np.asarray(m["t_gt"], float),
                 "K": np.asarray(m["K"], float),
                 "model": np.asarray(m["object_points"], float),
                 "ext": (m["dimensions_m"]["width"], m["dimensions_m"]["height"],
                         m["dimensions_m"]["depth"]),
                 "gt8": np.asarray(m["gt_corners_2d"], float)}
        gtb = bbox_of(truth["gt8"], w, h)
        cands = []
        for b in e["boxes"]:
            pb = [max(0.0, b["xyxy"][0]), max(0.0, b["xyxy"][1]),
                  min(float(w), b["xyxy"][2]), min(float(h), b["xyxy"][3])]
            v = iou(pb, gtb)
            p = pose_of(b["kps"], truth)
            if p is None:
                mets = {"R": np.nan, "t": np.nan, "corner": np.nan, "s5": 0}
            else:
                R, t = RM.pose_error(p[0], p[1], truth["R"], truth["t"])
                cn = float(np.median(np.linalg.norm(
                    np.asarray(b["kps"], float)[:8] - truth["gt8"][:8], axis=1)))
                mets = {"R": R, "t": t, "corner": cn,
                        "s5": int(RM.success_5cm5deg(p[0], p[1],
                                                     truth["R"], truth["t"]))}
            cands.append({"conf": b["conf"], "iou": v, **mets})
        pos.append({"fid": e["fid"], "set": e["set"],
                    "population": e["population"], "cands": cands})

    neg = [{"frame": e["frame"],
            "confs": [b["conf"] for b in e["boxes"]]} for e in raw["negative"]]

    # ---- PHASE 2 sweep ----
    def at(tau):
        avail = corr_top1 = corr_any = 0
        R, T, C, S = [], [], [], []
        for f in pos:
            s = [c for c in f["cands"] if c["conf"] >= tau]
            if not s:
                continue
            avail += 1
            top = max(s, key=lambda c: c["conf"])
            if top["iou"] >= IOU_MATCH:
                corr_top1 += 1
            if any(c["iou"] >= IOU_MATCH for c in s):
                corr_any += 1
            R.append(top["R"]); T.append(top["t"])
            C.append(top["corner"]); S.append(top["s5"])
        n = len(pos)
        fp = [sum(1 for c in e["confs"] if c >= tau) for e in neg]
        mx = [max([c for c in e["confs"] if c >= tau], default=0.0) for e in neg]
        R = np.array(R, float); T = np.array(T, float); C = np.array(C, float)
        g = np.isfinite(R)
        return {"conf": tau,
                "availability": avail / n,
                "recall": corr_top1 / n,
                "recall_available": corr_any / n,
                "corner_median": float(np.nanmedian(C)) if len(C) else None,
                "R_median": float(np.median(R[g])) if g.any() else None,
                "t_median": float(np.nanmedian(T)) if len(T) else None,
                "success_5cm5": float(np.mean(S)) if S else 0.0,
                "neg_fp_per_image": float(np.mean(fp)),
                "neg_frac_with_fp": float(np.mean([x > 0 for x in fp])),
                "neg_max_box_conf": float(np.max(mx)) if mx else 0.0,
                "neg_candidate_count": int(np.sum(fp))}

    sweep = [at(t) for t in CONFS]

    # ---- PHASE 3 threshold rule (사전등록) ----
    feas = [s for s in sweep if s["recall"] >= RECALL_TARGET]
    if feas:
        tau_star = min(feas, key=lambda s: (s["neg_fp_per_image"], -s["conf"]))
        rule = {"status": "SELECTED", "tau": tau_star["conf"],
                "rule": "FP/image 최소 s.t. recall>=0.98"}
    else:
        best = max(sweep, key=lambda s: s["recall"])
        tau_star = best
        rule = {"status": "UNRESOLVED",
                "reason": f"recall>={RECALL_TARGET} 를 만족하는 threshold 가 "
                          f"grid 에 없다. 최대 달성 recall "
                          f"{best['recall']:.4f} @ conf={best['conf']}",
                "max_achievable_recall": best["recall"],
                "at_conf": best["conf"],
                "fp_per_image_there": best["neg_fp_per_image"],
                "note": "게이트를 낮추지 않는다. tradeoff 를 보고한다."}

    # ---- PHASE 4 metrics (frame-level presence classifier) ----
    score_p = np.array([max([c["conf"] for c in f["cands"]], default=0.0)
                        for f in pos])
    score_n = np.array([max(e["confs"], default=0.0) for e in neg])
    y = np.r_[np.ones(len(score_p)), np.zeros(len(score_n))]
    s = np.r_[score_p, score_n]
    order = np.argsort(s)
    lab = y[order]
    ranks = np.arange(1, len(lab) + 1)
    npos, nneg = int(lab.sum()), int(len(lab) - lab.sum())
    auroc = float((ranks[lab == 1].sum() - npos*(npos+1)/2) / (npos*nneg))
    thr = np.unique(s)[::-1]
    prec, rec = [], []
    for t in thr:
        tp = int(((s >= t) & (y == 1)).sum()); fp = int(((s >= t) & (y == 0)).sum())
        prec.append(tp/max(tp+fp, 1)); rec.append(tp/npos)
    prec, rec = np.array(prec), np.array(rec)
    auprc = float(np.sum(np.diff(np.r_[0, rec]) * prec))

    def conf_matrix(tau):
        tp = int(((score_p >= tau)).sum()); fn = len(score_p) - tp
        fp = int(((score_n >= tau)).sum()); tn = len(score_n) - fp
        return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "recall_presence": tp/max(tp+fn, 1),
                "precision_presence": tp/max(tp+fp, 1)}

    tau = tau_star["conf"]
    per_pop = {}
    for pop in ("REAL_DEV_OPEN_56", "REAL_CHALLENGE_DEV_105"):
        sub = [f for f in pos if f["population"] == pop]
        av = sum(1 for f in sub if any(c["conf"] >= tau for c in f["cands"]))
        ok = sum(1 for f in sub
                 if (lambda s2: bool(s2) and
                     max(s2, key=lambda c: c["conf"])["iou"] >= IOU_MATCH)
                 ([c for c in f["cands"] if c["conf"] >= tau]))
        per_pop[pop] = {"n": len(sub), "availability": av/len(sub),
                        "recall": ok/len(sub)}
    metrics = {"AUROC": round(auroc, 4), "AUPRC": round(auprc, 4),
               "n_positive": len(score_p), "n_negative": len(score_n),
               "at_tau": {"tau": tau, **conf_matrix(tau),
                          "fp_per_image": tau_star["neg_fp_per_image"],
                          "by_population": per_pop},
               "at_current_0.40": {"tau": CURRENT, **conf_matrix(CURRENT)}}

    # ---- PHASE 5 recovered low-conf positives ----
    def top_at(f, t):
        s2 = [c for c in f["cands"] if c["conf"] >= t]
        return max(s2, key=lambda c: c["conf"]) if s2 else None

    rec_rows, base_rows = [], []
    for f in pos:
        now, new = top_at(f, CURRENT), top_at(f, tau)
        if new is None:
            continue
        (base_rows if now is not None else rec_rows).append((f, new))

    def block(rows):
        if not rows:
            return {"n": 0}
        R = np.array([c["R"] for _, c in rows], float)
        T = np.array([c["t"] for _, c in rows], float)
        C = np.array([c["corner"] for _, c in rows], float)
        g = np.isfinite(R)
        return {"n": len(rows),
                "correct_iou_frac": float(np.mean(
                    [c["iou"] >= IOU_MATCH for _, c in rows])),
                "corner_median": round(float(np.nanmedian(C)), 3),
                "corner_p90": round(float(np.nanpercentile(C, 90)), 3),
                "R_median": round(float(np.median(R[g])), 3) if g.any() else None,
                "R_p90": round(float(np.percentile(R[g], 90)), 3) if g.any() else None,
                "t_median": round(float(np.nanmedian(T)), 4),
                "t_p90": round(float(np.nanpercentile(T, 90)), 4),
                "success_5cm5": round(float(np.mean(
                    [c["s5"] for _, c in rows])), 4),
                "gross_R_gt10_frac": round(float(np.mean(
                    [(not np.isfinite(c["R"])) or c["R"] > GROSS_R
                     for _, c in rows])), 4)}
    recovered, baseline = block(rec_rows), block(base_rows)

    # ---- PHASE 6 role confusion (D3 재사용, 새 학습 없음) ----
    d3 = json.load(open(os.path.join(A2, "KEYPOINT_PERMUTATION_AUDIT.json")))
    role = {"source": "KEYPOINT_PERMUTATION_AUDIT.json (D3, 재계산 안 함)",
            "gross_threshold_R_deg": d3["gross_threshold_R_deg"],
            "n_gross": d3["n_gross"],
            "best_permutation_counts": d3["best_permutation_counts"],
            "near_far_swap_recovery_fraction": round(
                d3["best_permutation_counts"].get("near_far_swap", 0)
                / max(d3["n_gross"], 1), 4),
            "role_confusion_rate": d3["role_confusion_rate"],
            "V2_requirement": "LOW_ANGLE_ROLE_DISAMBIGUATION_COVERAGE",
            "constraint": "near/far swap oracle 을 main inference 에 사용하지 않는다."}

    # ---- PHASE 7 verdict (게이트는 위에서 고정) ----
    recall_ok = tau_star["recall"] >= RECALL_TARGET
    pose_ok = (recovered["n"] == 0 or
               (recovered.get("gross_R_gt10_frac", 1.0)
                <= baseline.get("gross_R_gt10_frac", 0.0) + 0.20))
    verdict = ("CONF_CALIBRATION_SUPPORTED"
               if (recall_ok and pose_ok and rule["status"] == "SELECTED")
               else "CONF_CALIBRATION_UNRESOLVED")

    out = {"phase0": "REAL_NEG_DEV_AUDIT.json 참조",
           "prelocked": {"conf_grid": CONFS, "iou_match": IOU_MATCH,
                         "recall_target": RECALL_TARGET,
                         "rule": "tau* = FP/image 최소 s.t. recall>=0.98",
                         "recall_definition": "top-1 survivor 의 IoU>=0.5 "
                                              "(배포가 실제로 쓰는 것)",
                         "gross_R_deg": GROSS_R,
                         "pose_gate": "recovered 의 gross R>10 비율이 기존 "
                                      "detected population 대비 +20pp 이내"},
           "sweep": sweep, "threshold_rule": rule, "metrics": metrics,
           "recovered": recovered, "baseline_detected_at_0.40": baseline,
           "role_confusion": role, "VERDICT": verdict,
           "fp_scope": "FP/image 는 REAL_NEG_DEV_V1 259 장 기준이며, 그 셋이 "
                       "max_conf<0.20 로 선별된 편향 표본이므로 **하한**이다. "
                       "natural prevalence 를 모르므로 배포 threshold 는 별도."}
    json.dump(out, open(os.path.join(A2, "YOLO_CONF_SWEEP.json"), "w"),
              indent=1, ensure_ascii=False)

    with open(os.path.join(A2, "YOLO_CONF_SWEEP_PER_FRAME.csv"), "w",
              newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["kind", "id", "set", "population", "n_cand_at_0.001",
                      "top_conf", "top_iou", "top_R", "top_t", "top_corner",
                      "top_s5", "detected_at_0.40", "detected_at_tau",
                      "recovered"])
        for f in pos:
            top = max(f["cands"], key=lambda c: c["conf"]) if f["cands"] else None
            d40 = top is not None and top["conf"] >= CURRENT
            dts = top is not None and top["conf"] >= tau
            wtr.writerow(["positive", f["fid"], f["set"], f["population"],
                          len(f["cands"]),
                          None if not top else round(top["conf"], 5),
                          None if not top else round(top["iou"], 4),
                          None if not top else round(top["R"], 3),
                          None if not top else round(top["t"], 4),
                          None if not top else round(top["corner"], 3),
                          None if not top else top["s5"],
                          int(d40), int(dts), int(dts and not d40)])
        for e in neg:
            wtr.writerow(["negative", e["frame"], "forklift_raw_20260528",
                          "REAL_NEG_DEV_V1", len(e["confs"]),
                          round(max(e["confs"], default=0.0), 5),
                          None, None, None, None, None,
                          int(max(e["confs"], default=0.0) >= CURRENT),
                          int(max(e["confs"], default=0.0) >= tau), None])

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, c="#2c3e50")
    ax.scatter([metrics["at_tau"]["recall_presence"]],
               [metrics["at_tau"]["precision_presence"]], c="#c0392b", zorder=5,
               label=f"tau={tau}")
    ax.scatter([metrics["at_current_0.40"]["recall_presence"]],
               [metrics["at_current_0.40"]["precision_presence"]], c="#2980b9",
               zorder=5, label="현재 conf=0.40")
    ax.set_xlabel("presence recall"); ax.set_ylabel("presence precision")
    ax.set_title(f"PR curve  AUPRC={auprc:.4f}  AUROC={auroc:.4f}")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PL, "PR_CURVE.png"), dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    r = [s["recall"] for s in sweep]
    ra = [s["recall_available"] for s in sweep]
    fpi = [s["neg_fp_per_image"] for s in sweep]
    ax[0].plot(CONFS, r, "o-", c="#2c3e50", label="recall (top-1)")
    ax[0].plot(CONFS, ra, "s--", c="#7f8c8d", label="recall_available (천장)")
    ax[0].axhline(RECALL_TARGET, ls=":", c="#c0392b", label="목표 0.98")
    ax[0].set_xscale("log"); ax[0].set_xlabel("conf"); ax[0].set_ylabel("recall")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_title("positive recall")
    ax[1].plot(r, fpi, "o-", c="#c0392b")
    for s in sweep:
        ax[1].annotate(f"{s['conf']}", (s["recall"], s["neg_fp_per_image"]),
                       fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("positive recall (top-1)")
    ax[1].set_ylabel("FP/image (REAL_NEG_DEV_V1, 하한)")
    ax[1].grid(alpha=.3); ax[1].set_title("FP vs recall tradeoff")
    fig.tight_layout(); fig.savefig(os.path.join(PL, "FP_VS_RECALL.png"), dpi=130)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for i, (k, lb) in enumerate((("corner", "corner err (px)"),
                                 ("R", "R err (deg)"), ("t", "t err (m)"))):
        a = [c[k] for _, c in base_rows if np.isfinite(c[k])]
        b = [c[k] for _, c in rec_rows if np.isfinite(c[k])]
        ax[i].boxplot([a, b] if b else [a], labels=(["기존 검출", "회수분"]
                                                    if b else ["기존 검출"]),
                      showfliers=False)
        ax[i].set_ylabel(lb); ax[i].grid(alpha=.3)
    fig.suptitle(f"회수된 low-conf positive 의 pose 품질 "
                 f"(n={recovered['n']} vs {baseline['n']})")
    fig.tight_layout()
    fig.savefig(os.path.join(PL, "RECOVERED_LOWCONF_POSE.png"), dpi=130)

    # ---- print ----
    print(f"{'conf':>7}{'avail':>8}{'recall':>8}{'r_avail':>9}{'corner':>8}"
          f"{'R med':>8}{'5cm5':>7}{'FP/img':>8}{'frac FP':>9}{'maxconf':>9}")
    print("─" * 81)
    for s in sweep:
        print(f"{s['conf']:>7}{s['availability']:>8.3f}{s['recall']:>8.3f}"
              f"{s['recall_available']:>9.3f}{s['corner_median']:>8.2f}"
              f"{s['R_median']:>8.2f}{s['success_5cm5']:>7.3f}"
              f"{s['neg_fp_per_image']:>8.3f}{s['neg_frac_with_fp']:>9.3f}"
              f"{s['neg_max_box_conf']:>9.3f}")
    print(f"\nTHRESHOLD RULE: {json.dumps(rule, ensure_ascii=False)}")
    print(f"AUROC {auroc:.4f}  AUPRC {auprc:.4f}")
    print(f"recovered {recovered}")
    print(f"baseline  {baseline}")
    print(f"role      near/far swap recovery {role['near_far_swap_recovery_fraction']}")
    print(f"\nVERDICT = {verdict}")


if __name__ == "__main__":
    main()
