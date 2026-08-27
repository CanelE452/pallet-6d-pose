"""PROMOTION GATE — PURPOSE.md / METHOD_SPEC 21절에 결과 보기 전 고정한 기준 그대로.

여기서 threshold 를 만들지 않는다.  전부 상수로 박혀 있고, 결과를 보고 고치면
사전등록의 의미가 없어진다.

    SAFETY  전부 통과해야 함
      S1  ALL detection recall (conf>=0.40)   HC 대비 drop <= 2pp
      S2  ALL top1-cbox                        HC 대비 drop <= 2pp
      S3  NIGHT top1-cbox                      HC 대비 drop <= 5pp
      S4  positive conf p05                    HC 대비 30% 이상 상대 붕괴 금지

    BENEFIT  2개 이상
      B1  FPR@TPR95                 HC 대비 >= 15% 상대 개선
      B2  FP/image @ matched det recall 0.90   >= 20% 상대 개선
      B3  negative AUPRC            >= +0.01 절대
      B4  HF 가 HM 보다 high-recall FP 지표에서 명확히 개선

verdict  SAFETY 실패 -> HF_STOP_POSITIVE_SUPPRESSION
         BENEFIT 0~1 -> HF_NO_USEFUL_SIGNAL
         SAFETY PASS + BENEFIT>=2 -> HF_PROMOTE_30EP
         HM > HC 이고 HF 추가이득 없음 -> HARD_MINING_SUFFICIENT
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
HN = os.path.join(ROOT, "challenge/yolo_pose_one_model/hard_negative_v1")
EVAL = os.path.join(HN, "evaluation")
ARMS = ["HC", "HM", "HF"]


def load(arm):
    pos = json.load(open(os.path.join(EVAL, f"POSITIVE_DEV__{arm}.json")))
    neg = json.load(open(os.path.join(EVAL, f"NEGSCORE__{arm}.json")))
    return pos, neg


def positive_metrics(pos):
    rec = pos["per_frame"]
    night = [r for r in rec if r["domain"] == "NIGHT"]

    def cbox_rate(rs):
        return float(np.mean([bool(r.get("correct_box", False)) for r in rs])) if rs else None
    return {
        "n_total": pos["n_total"],
        "det_recall_deploy": pos["detection_recall_deploy"],
        "det_recall_threshold_free": pos["detection_recall"],
        "top1_cbox_ALL": pos["correct_box_recall"],
        "top1_cbox_NIGHT": cbox_rate(night),
        "n_night": len(night),
        "corner_med_cbox": pos["correct_box"].get("corner_median"),
        "corner_p90_cbox": pos["correct_box"].get("corner_p90"),
    }


def negative_metrics(neg):
    rows = neg["rows"]
    p = np.array([r["max_conf"] for r in rows if r["label"] == 1])
    n = np.array([r["max_conf"] for r in rows if r["label"] == 0])
    n_counts = [r["confs"] for r in rows if r["label"] == 0]
    scores = np.concatenate([p, n])
    labels = np.concatenate([np.ones_like(p), np.zeros_like(n)])

    order = np.argsort(-scores)
    lab = labels[order]
    tp, fp = np.cumsum(lab == 1), np.cumsum(lab == 0)
    recall = tp / max(int((labels == 1).sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    auprc = float(np.sum(np.diff(np.concatenate([[0.0], recall]))
                         * np.concatenate([[1.0], precision])[1:]))
    # AUROC — rank 기반
    ranks = np.empty(len(scores), float)
    ranks[np.argsort(scores, kind="mergesort")] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):
        hit = scores == v
        if hit.sum() > 1:
            ranks[hit] = ranks[hit].mean()
    auroc = float((ranks[labels == 1].sum() - len(p) * (len(p) + 1) / 2)
                  / (len(p) * len(n)))

    def tau_at_tpr(t):
        return float(np.percentile(p, (1 - t) * 100))

    def fpr_at(t):
        return float((n >= tau_at_tpr(t)).mean())

    def fp_per_image_at(t):
        tau = tau_at_tpr(t)
        return float(np.mean([sum(1 for c in cs if c >= tau) for cs in n_counts]))

    return {
        "n_pos": int(len(p)), "n_neg": int(len(n)),
        "pos_conf_p05": float(np.percentile(p, 5)),
        "pos_conf_median": float(np.median(p)),
        "pos_conf_p95": float(np.percentile(p, 95)),
        "AUPRC": auprc, "AUROC": auroc,
        "FPR_at_TPR90": fpr_at(0.90), "FPR_at_TPR95": fpr_at(0.95),
        "FP_per_image_at_det_recall": {f"{t:.2f}": fp_per_image_at(t)
                                       for t in (0.80, 0.90, 0.95)},
        "FP_per_image_at_conf_0.40": float(
            np.mean([sum(1 for c in cs if c >= 0.40) for cs in n_counts])),
    }


def main():
    M = {}
    for arm in ARMS:
        pos, neg = load(arm)
        M[arm] = {**positive_metrics(pos), **negative_metrics(neg)}
    json.dump(M, open(os.path.join(EVAL, "SCREEN_COMPARISON.json"), "w"), indent=1)

    hc, hm, hf = M["HC"], M["HM"], M["HF"]
    pp = lambda a, b: (a - b) * 100.0                      # noqa: E731  pp drop
    rel = lambda a, b: (b - a) / a if a else float("nan")  # noqa: E731

    S = {
        "S1_det_recall_drop_pp": pp(hc["det_recall_deploy"], hf["det_recall_deploy"]),
        "S2_top1_cbox_drop_pp": pp(hc["top1_cbox_ALL"], hf["top1_cbox_ALL"]),
        "S3_night_cbox_drop_pp": pp(hc["top1_cbox_NIGHT"], hf["top1_cbox_NIGHT"]),
        "S4_pos_conf_p05_rel_drop": (hc["pos_conf_p05"] - hf["pos_conf_p05"])
        / hc["pos_conf_p05"] if hc["pos_conf_p05"] else float("nan"),
    }
    S["S1_PASS"] = bool(S["S1_det_recall_drop_pp"] <= 2.0)
    S["S2_PASS"] = bool(S["S2_top1_cbox_drop_pp"] <= 2.0)
    S["S3_PASS"] = bool(S["S3_night_cbox_drop_pp"] <= 5.0)
    S["S4_PASS"] = bool(S["S4_pos_conf_p05_rel_drop"] < 0.30)
    safety = all(S[f"S{i}_PASS"] for i in (1, 2, 3, 4))

    b1 = (hc["FPR_at_TPR95"] - hf["FPR_at_TPR95"]) / hc["FPR_at_TPR95"] \
        if hc["FPR_at_TPR95"] else float("nan")
    k = "0.90"
    b2 = (hc["FP_per_image_at_det_recall"][k] - hf["FP_per_image_at_det_recall"][k]) \
        / hc["FP_per_image_at_det_recall"][k] if hc["FP_per_image_at_det_recall"][k] \
        else float("nan")
    b3 = hf["AUPRC"] - hc["AUPRC"]
    b4 = (hm["FPR_at_TPR95"] - hf["FPR_at_TPR95"]) / hm["FPR_at_TPR95"] \
        if hm["FPR_at_TPR95"] else float("nan")
    B = {"B1_FPR95_rel_improve": b1, "B1_PASS": bool(b1 >= 0.15),
         "B2_FP_at_recall090_rel_improve": b2, "B2_PASS": bool(b2 >= 0.20),
         "B3_AUPRC_abs_gain": b3, "B3_PASS": bool(b3 >= 0.01),
         "B4_HF_over_HM_FPR95_rel": b4, "B4_PASS": bool(b4 >= 0.10)}
    n_benefit = sum(B[f"B{i}_PASS"] for i in (1, 2, 3, 4))

    hm_beats_hc = (hm["AUPRC"] > hc["AUPRC"]) and \
                  (pp(hc["det_recall_deploy"], hm["det_recall_deploy"]) <= 2.0)
    hf_adds = B["B4_PASS"]

    if not safety:
        verdict = "HF_STOP_POSITIVE_SUPPRESSION"
    elif n_benefit >= 2:
        verdict = "HF_PROMOTE_30EP"
    elif hm_beats_hc and not hf_adds:
        verdict = "HARD_MINING_SUFFICIENT"
    else:
        verdict = "HF_NO_USEFUL_SIGNAL"

    out = {"metrics": M, "SAFETY": S, "BENEFIT": B, "n_benefit_pass": n_benefit,
           "safety_pass": safety, "hm_beats_hc": bool(hm_beats_hc),
           "verdict": verdict,
           "gate_source": "PURPOSE.md / METHOD_SPEC 21 — 결과 보기 전 고정"}
    json.dump(out, open(os.path.join(EVAL, "PROMOTION_GATE.json"), "w"), indent=1)

    w = 34
    print("\n[10EP SCREEN]")
    print(f"{'metric':{w}}{'HC':>10}{'HM':>10}{'HF':>10}")
    print("-" * (w + 30))
    for key in ("det_recall_deploy", "top1_cbox_ALL", "top1_cbox_NIGHT",
                "corner_med_cbox", "corner_p90_cbox", "pos_conf_p05",
                "pos_conf_median", "AUPRC", "AUROC", "FPR_at_TPR90",
                "FPR_at_TPR95", "FP_per_image_at_conf_0.40"):
        v = [M[a].get(key) for a in ARMS]
        print(f"{key:{w}}" + "".join(
            f"{x:>10.4f}" if isinstance(x, float) else f"{str(x):>10}" for x in v))
    print(f"{'FP/img @ det recall 0.80':{w}}" + "".join(
        f"{M[a]['FP_per_image_at_det_recall']['0.80']:>10.4f}" for a in ARMS))
    print(f"{'FP/img @ det recall 0.90':{w}}" + "".join(
        f"{M[a]['FP_per_image_at_det_recall']['0.90']:>10.4f}" for a in ARMS))
    print(f"{'FP/img @ det recall 0.95':{w}}" + "".join(
        f"{M[a]['FP_per_image_at_det_recall']['0.95']:>10.4f}" for a in ARMS))

    print("\n[PROMOTION]")
    for i in (1, 2, 3, 4):
        print(f"  S{i}  PASS={S[f'S{i}_PASS']}")
    for i in (1, 2, 3, 4):
        print(f"  B{i}  PASS={B[f'B{i}_PASS']}")
    print(f"\nverdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
