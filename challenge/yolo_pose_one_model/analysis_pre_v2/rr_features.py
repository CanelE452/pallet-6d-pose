"""PHASE 3~4 — GT-free 단일 feature 의 reranking 능력 + session LOSO.

learned reranker 금지.  feature 를 **하나씩** 잰다.  결과를 보고 조합하지 않는다.

--- 결과 보기 전에 고정 -------------------------------------------------------
  target_box     candidate 의 IoU>=0.5
  target_usable  IoU>=0.5 AND R<=10deg AND t<=0.10m  (배포가 실제 원하는 것)
  지표           frame 별 ranking AUC, MRR(첫 정답의 역순위)
  LOSO           held-out session 빼고 6 개에서 MRR(usable) 최고 feature 선택
  판정
    RERANKING_SUPPORTED
        CV recall gain >= +5pp AND 5cm5 악화 <= 0pp AND R median 악화 <= 0
    RANKING_TRACK_CLOSED
        CV recall 손실 > 0 OR 5cm5 손실 > 2pp
    RERANKING_HEADROOM_BUT_NOT_PREDICTABLE
        그 외 (상한은 있는데 예측 가능한 규칙으로 못 얻음)
"""
from __future__ import annotations
import csv, json, os
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
A2 = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis_pre_v2")
PL = os.path.join(A2, "plots")
IOU_MATCH, USABLE_R, USABLE_T = 0.5, 10.0, 0.10
GATE = {"support_recall_gain_pp": 5.0, "support_s5_degrade_pp": 0.0,
        "close_recall_loss_pp": 0.0, "close_s5_loss_pp": 2.0}
# 값이 클수록 좋은 방향으로 부호를 맞춰 둔다 (GT 미사용)
FEATURES = {
    "box_conf":        lambda c: c["conf"],
    "kp_conf_mean":    lambda c: c["kp_conf_mean"],
    "kp_conf_min":     lambda c: c["kp_conf_min"],
    "box_area":        lambda c: c["box_area"],
    "box_diag":        lambda c: c["box_diag"],
    "neg_pnp_reproj":  lambda c: -c["reproj"] if np.isfinite(c["reproj"]) else -1e9,
    "depth_valid":     lambda c: float(c["depth_ok"]),
    "cuboid_plausible": lambda c: float(c["cuboid_ok"]),
}


def label_box(c):
    return int(c["iou"] >= IOU_MATCH)


def label_usable(c):
    return int(c["iou"] >= IOU_MATCH and np.isfinite(c["R"])
               and c["R"] <= USABLE_R and c["t"] <= USABLE_T)


def frame_auc(scores, labels):
    pos = int(sum(labels)); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    o = np.argsort(np.asarray(scores, float))
    lab = np.asarray(labels)[o]
    ranks = np.arange(1, len(lab)+1)
    return float((ranks[lab == 1].sum() - pos*(pos+1)/2) / (pos*neg))


def frame_rr(scores, labels):
    """첫 정답의 역순위. 정답이 없으면 None."""
    if sum(labels) == 0:
        return None
    o = np.argsort(-np.asarray(scores, float), kind="stable")
    for i, j in enumerate(o, 1):
        if labels[j]:
            return 1.0/i
    return None


def main():
    frames = json.load(open(os.path.join(A2, "_rr_cands.json")))
    sessions = sorted({f["set"] for f in frames})

    # ---- PHASE 3 ----
    ph3 = {}
    for name, fn in FEATURES.items():
        row = {}
        for tag, lab_fn in (("box", label_box), ("usable", label_usable)):
            aucs, rrs = [], []
            for f in frames:
                cs = f["cands"]
                if len(cs) < 2:
                    continue
                s = [fn(c) for c in cs]
                y = [lab_fn(c) for c in cs]
                a = frame_auc(s, y)
                if a is not None:
                    aucs.append(a)
                r = frame_rr(s, y)
                if r is not None:
                    rrs.append(r)
            row[tag] = {"AUC": round(float(np.mean(aucs)), 4) if aucs else None,
                        "n_auc_frames": len(aucs),
                        "MRR": round(float(np.mean(rrs)), 4) if rrs else None,
                        "n_mrr_frames": len(rrs)}
        ph3[name] = row

    # ---- PHASE 4 LOSO ----
    def evaluate(pick_fn, subset):
        """pick_fn 이 고른 top-1 로 recall / R / t / 5cm5."""
        rec = R = []
        rec, Rs, Ts, S = 0, [], [], []
        for f in subset:
            if not f["cands"]:
                S.append(0); continue
            c = pick_fn(f["cands"])
            rec += int(c["iou"] >= IOU_MATCH)
            if np.isfinite(c["R"]):
                Rs.append(c["R"]); Ts.append(c["t"])
            S.append(c["s5"])
        n = len(subset)
        return {"n": n, "recall": rec/n,
                "R_median": float(np.median(Rs)) if Rs else None,
                "t_median": float(np.median(Ts)) if Ts else None,
                "success_5cm5": float(np.mean(S)) if S else 0.0}

    native = lambda cs: cs[0]                       # noqa: E731

    folds = []
    for held in sessions:
        train = [f for f in frames if f["set"] != held]
        test = [f for f in frames if f["set"] == held]
        best, best_mrr = None, -1.0
        for name, fn in FEATURES.items():
            rrs = [r for f in train
                   if len(f["cands"]) >= 2 and
                   (r := frame_rr([fn(c) for c in f["cands"]],
                                  [label_usable(c) for c in f["cands"]]))
                   is not None]
            m = float(np.mean(rrs)) if rrs else 0.0
            if m > best_mrr:
                best, best_mrr = name, m
        fn = FEATURES[best]
        pick = lambda cs, fn=fn: max(cs, key=fn)    # noqa: E731
        a, b = evaluate(native, test), evaluate(pick, test)
        folds.append({"held_out": held, "feature": best,
                      "train_MRR_usable": round(best_mrr, 4),
                      "native": a, "reranked": b,
                      "recall_gain_pp": round((b["recall"]-a["recall"])*100, 2),
                      "s5_gain_pp": round((b["success_5cm5"]
                                           - a["success_5cm5"])*100, 2)})

    feat_by = {f["held_out"]: FEATURES[f["feature"]] for f in folds}
    cv_pick = lambda f: max(f["cands"], key=feat_by[f["set"]])   # noqa: E731
    nat_all = evaluate(native, frames)
    cv_all = {"n": len(frames)}
    rec, Rs, Ts, S = 0, [], [], []
    for f in frames:
        if not f["cands"]:
            S.append(0); continue
        c = cv_pick(f)
        rec += int(c["iou"] >= IOU_MATCH)
        if np.isfinite(c["R"]):
            Rs.append(c["R"]); Ts.append(c["t"])
        S.append(c["s5"])
    cv_all.update({"recall": rec/len(frames),
                   "R_median": float(np.median(Rs)) if Rs else None,
                   "t_median": float(np.median(Ts)) if Ts else None,
                   "success_5cm5": float(np.mean(S))})

    pops = {}
    for pop in ("REAL_DEV_OPEN_56", "REAL_CHALLENGE_DEV_105"):
        sub = [f for f in frames if f["population"] == pop]
        rec2, S2 = 0, []
        for f in sub:
            c = cv_pick(f) if f["cands"] else None
            rec2 += int(c is not None and c["iou"] >= IOU_MATCH)
            S2.append(0 if c is None else c["s5"])
        pops[pop] = {"n": len(sub), "native": evaluate(native, sub),
                     "reranked_recall": rec2/len(sub),
                     "reranked_5cm5": float(np.mean(S2))}

    rg = (cv_all["recall"] - nat_all["recall"])*100
    sg = (cv_all["success_5cm5"] - nat_all["success_5cm5"])*100
    rdeg = (cv_all["R_median"] - nat_all["R_median"])
    if rg < -GATE["close_recall_loss_pp"] or sg < -GATE["close_s5_loss_pp"]:
        verdict = "RANKING_TRACK_CLOSED"
    elif (rg >= GATE["support_recall_gain_pp"] and sg >= 0.0 and rdeg <= 0):
        verdict = "RERANKING_SUPPORTED"
    else:
        verdict = "RERANKING_HEADROOM_BUT_NOT_PREDICTABLE"

    out = {"prelocked": {"iou_match": IOU_MATCH,
                         "usable": f"IoU>=0.5 AND R<={USABLE_R} AND t<={USABLE_T}",
                         "gates": GATE,
                         "no_learned_reranker": True,
                         "no_feature_combination": "결과 보고 조합하지 않는다"},
           "phase3_single_feature": ph3,
           "phase4_folds": folds,
           "phase4_cv_overall": {"native_top1": nat_all, "reranked_top1": cv_all,
                                 "recall_gain_pp": round(rg, 2),
                                 "s5_gain_pp": round(sg, 2),
                                 "R_median_change": round(rdeg, 3)},
           "phase4_by_population": pops,
           "VERDICT": verdict}
    json.dump(out, open(os.path.join(A2, "RERANK_FEATURES.json"), "w"),
              indent=1, ensure_ascii=False)

    with open(os.path.join(A2, "RERANK_PER_FRAME.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fid", "set", "population", "n_cand", "native_iou",
                    "native_R", "native_t", "native_s5", "cv_feature",
                    "cv_rank_chosen", "cv_iou", "cv_R", "cv_t", "cv_s5"])
        fmap = {f["held_out"]: f["feature"] for f in folds}
        for f in frames:
            if not f["cands"]:
                continue
            a, b = f["cands"][0], cv_pick(f)
            w.writerow([f["fid"], f["set"], f["population"], len(f["cands"]),
                        round(a["iou"], 4), round(a["R"], 3), round(a["t"], 4),
                        a["s5"], fmap[f["set"]], b["rank"],
                        round(b["iou"], 4), round(b["R"], 3), round(b["t"], 4),
                        b["s5"]])

    print("PHASE 3 — 단일 feature (frame 별 평균)")
    print(f"{'feature':20}{'AUC box':>9}{'MRR box':>9}{'AUC use':>9}{'MRR use':>9}")
    print("─"*56)
    for k, v in sorted(ph3.items(), key=lambda x: -(x[1]['usable']['MRR'] or 0)):
        print(f"{k:20}{(v['box']['AUC'] or 0):>9.4f}{(v['box']['MRR'] or 0):>9.4f}"
              f"{(v['usable']['AUC'] or 0):>9.4f}{(v['usable']['MRR'] or 0):>9.4f}")
    print("\nPHASE 4 — session LOSO")
    print(f"{'held out':12}{'feature':18}{'nat rec':>9}{'rr rec':>9}"
          f"{'nat 5cm5':>10}{'rr 5cm5':>9}")
    print("─"*67)
    for f in folds:
        print(f"{f['held_out'].replace('eval_',''):12}{f['feature']:18}"
              f"{f['native']['recall']:>9.3f}{f['reranked']['recall']:>9.3f}"
              f"{f['native']['success_5cm5']:>10.3f}"
              f"{f['reranked']['success_5cm5']:>9.3f}")
    print(f"\n전체 CV  recall {nat_all['recall']:.3f} -> {cv_all['recall']:.3f} "
          f"({rg:+.2f}pp)   5cm5 {nat_all['success_5cm5']:.3f} -> "
          f"{cv_all['success_5cm5']:.3f} ({sg:+.2f}pp)   "
          f"R med {nat_all['R_median']:.2f} -> {cv_all['R_median']:.2f}")
    print(f"VERDICT = {verdict}")


if __name__ == "__main__":
    main()
