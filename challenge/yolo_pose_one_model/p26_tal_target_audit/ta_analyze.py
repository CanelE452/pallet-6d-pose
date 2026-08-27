from __future__ import annotations
import csv, json, os, sys
import numpy as np
from scipy.stats import spearmanr

NS = "/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/p26_tal_target_audit"
R = list(csv.DictReader(open(f"{NS}/TAL_PER_CANDIDATE.csv")))
def f(x):
    return None if x in ("", "None", None) else float(x)
for r in R:
    for k in ("class_score", "class_logit", "iou_to_gt", "target_score", "align_metric",
              "tal_overlap_ciou", "kp_error_px", "kp8_error_px", "kp_p90_px"):
        r[k] = f(r[k])
    r["fg_mask"] = int(r["fg_mask"])
by = {}
for r in R:
    by.setdefault((r["dataset"], r["image_id"]), {})[r["role"]] = r


def stats(v):
    v = np.asarray([x for x in v if x is not None], float)
    if not v.size: return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}


def paired(pos_role, neg_role, dset, dom=None):
    ps = [(v[pos_role], v[neg_role]) for (d, _), v in by.items()
          if d == dset and pos_role in v and neg_role in v
          and (dom is None or v[pos_role]["domain"] == dom)]
    if not ps: return {"n": 0}
    tm = [a["target_score"] - b["target_score"] for a, b in ps]
    lm = [a["class_logit"] - b["class_logit"] for a, b in ps]
    im = [a["iou_to_gt"] - b["iou_to_gt"] for a, b in ps]
    km = [(-(a["kp_error_px"] or 0)) - (-(b["kp_error_px"] or 0)) for a, b in ps]
    return {"n": len(ps), "target_margin": stats(tm), "logit_margin": stats(lm),
            "iou_margin": stats(im), "kp_quality_margin_negpx": stats(km),
            "frac_target_pos_gt_neg": float(np.mean(np.array(tm) > 0)),
            "frac_logit_pos_gt_neg": float(np.mean(np.array(lm) > 0)),
            "frac_target_tie_zero": float(np.mean(np.array(tm) == 0)),
            "neg_target_score": stats([b["target_score"] for _, b in ps]),
            "pos_target_score": stats([a["target_score"] for a, _ in ps]),
            "pos_fg_rate": float(np.mean([a["fg_mask"] for a, _ in ps])),
            "neg_fg_rate": float(np.mean([b["fg_mask"] for _, b in ps]))}


REAL = {s: paired("Rplus", "RW", "REAL", None if s == "ALL" else s)
        for s in ("ALL", "DAY", "NIGHT")}
rf = [(v["Rplus"], v["RW"]) for (d, _), v in by.items()
      if d == "REAL" and "Rplus" in v and "RW" in v
      and v["RW"]["class_logit"] > v["Rplus"]["class_logit"]]
REAL["RANKFAIL"] = {"n": len(rf),
    "pos_fg_rate": float(np.mean([a["fg_mask"] for a, _ in rf])) if rf else None,
    "target_margin": stats([a["target_score"] - b["target_score"] for a, b in rf]),
    "logit_margin": stats([a["class_logit"] - b["class_logit"] for a, b in rf]),
    "iou_margin": stats([a["iou_to_gt"] - b["iou_to_gt"] for a, b in rf]),
    "neg_target_score": stats([b["target_score"] for _, b in rf]),
    "pos_target_score": stats([a["target_score"] for a, _ in rf]),
    "frac_target_pos_gt_neg": float(np.mean([a["target_score"] > b["target_score"] for a, b in rf])) if rf else None,
    "night_frac": float(np.mean([a["domain"] == "NIGHT" for a, _ in rf])) if rf else None}
json.dump(REAL, open(f"{NS}/REAL_PAIRED_TARGET_ANALYSIS.json", "w"), indent=2, ensure_ascii=False)

SY = {"ALL": paired("Splus", "SW", "SYNTH")}
sf = [(v["Splus"], v["SW"]) for (d, _), v in by.items()
      if d == "SYNTH" and "Splus" in v and "SW" in v
      and v["SW"]["class_logit"] > v["Splus"]["class_logit"]]
SY["SYNTH_RANKFAIL"] = {"n": len(sf),
    "rate": len(sf) / max(SY["ALL"]["n"], 1),
    "target_margin": stats([a["target_score"] - b["target_score"] for a, b in sf]),
    "iou_margin": stats([a["iou_to_gt"] - b["iou_to_gt"] for a, b in sf])}
json.dump(SY, open(f"{NS}/SYNTH_PAIRED_TARGET_ANALYSIS.json", "w"), indent=2, ensure_ascii=False)

# ---------------- target quality correlation (assigned positives)
def corr(rows, a, b):
    x = [r[a] for r in rows if r.get(a) is not None and r.get(b) is not None]
    y = [r[b] for r in rows if r.get(a) is not None and r.get(b) is not None]
    if len(x) < 8: return {"n": len(x), "rho": None}
    s = spearmanr(x, y)
    return {"n": len(x), "rho": float(s.statistic), "p": float(s.pvalue)}


TQ = {}
for dset in ("REAL", "SYNTH"):
    pos = [r for r in R if r["dataset"] == dset and r["role"] in ("Rplus", "Splus")
           and r["fg_mask"] == 1]
    posall = [r for r in R if r["dataset"] == dset and r["role"] in ("Rplus", "Splus")]
    TQ[dset] = {"n_assigned_positive": len(pos), "n_pos_total": len(posall),
                "fg_rate_of_correct_candidate": len(pos) / max(len(posall), 1)}
    for scope, rows in [("ALL", posall)] + [(lv, [x for x in posall if x["level"] == lv])
                                            for lv in ("P3", "P4", "P5")]:
        TQ[dset][scope] = {
            "n": len(rows),
            "target_vs_iou": corr(rows, "target_score", "iou_to_gt"),
            "target_vs_kp_err": corr(rows, "target_score", "kp_error_px"),
            "logit_vs_target": corr(rows, "class_logit", "target_score"),
            "logit_vs_iou": corr(rows, "class_logit", "iou_to_gt"),
            "logit_vs_kp_err": corr(rows, "class_logit", "kp_error_px")}
    good = [r for r in posall if r["kp_error_px"] is not None and r["kp_error_px"] <= 20]
    bad = [r for r in posall if r["kp_error_px"] is not None and r["kp_error_px"] > 20]
    TQ[dset]["GOOD_vs_BAD_KP"] = {
        "n_good": len(good), "n_bad": len(bad),
        "target_score": {"GOOD": stats([r["target_score"] for r in good]),
                         "BAD": stats([r["target_score"] for r in bad])},
        "class_logit": {"GOOD": stats([r["class_logit"] for r in good]),
                        "BAD": stats([r["class_logit"] for r in bad])},
        "iou": {"GOOD": stats([r["iou_to_gt"] for r in good]),
                "BAD": stats([r["iou_to_gt"] for r in bad])},
        "kp_error": {"GOOD": stats([r["kp_error_px"] for r in good]),
                     "BAD": stats([r["kp_error_px"] for r in bad])}}
TQ["note"] = "kp_error_px 는 작을수록 좋음 — rho 부호를 그렇게 읽는다"
json.dump(TQ, open(f"{NS}/TARGET_QUALITY_CORRELATION.json", "w"), indent=2, ensure_ascii=False)

# ---------------- level calibration
LC = {"logit_distribution": {}, "paired": {}}
for dset, pr, nr in (("REAL", "Rplus", "RW"), ("SYNTH", "Splus", "SW")):
    LC["logit_distribution"][dset] = {}
    for lv in ("P3", "P4", "P5"):
        LC["logit_distribution"][dset][lv] = {
            "correct": stats([r["class_logit"] for r in R if r["dataset"] == dset
                              and r["role"] == pr and r["level"] == lv]),
            "wrong": stats([r["class_logit"] for r in R if r["dataset"] == dset
                            and r["role"] == nr and r["level"] == lv])}
ps = [(v["Rplus"], v["RW"]) for (d, _), v in by.items()
      if d == "REAL" and "Rplus" in v and "RW" in v]
same = [(a, b) for a, b in ps if a["level"] == b["level"]]
cross = [(a, b) for a, b in ps if a["level"] != b["level"]]
for lab, g in (("same_level", same), ("cross_level", cross)):
    m = np.array([a["class_logit"] - b["class_logit"] for a, b in g]) if g else np.array([])
    LC["paired"][lab] = {"n": len(g),
                         "frac_pos_gt_neg": float((m > 0).mean()) if m.size else None,
                         "logit_margin": stats(m.tolist())}
# oracle level offset (진단용 상한, 배포 아님)
lv_i = {"P3": 0, "P4": 1, "P5": 2}
best, bestoff = None, None
grid = np.arange(-3.0, 3.01, 0.05)
base_fail = sum(1 for a, b in ps if (a["class_logit"] - b["class_logit"]) <= 0)
for o3 in grid:
    for o4 in grid[::4]:
        fails = 0
        for a, b in ps:
            off = {"P3": o3, "P4": o4, "P5": 0.0}
            if (a["class_logit"] + off[a["level"]]) - (b["class_logit"] + off[b["level"]]) <= 0:
                fails += 1
        if best is None or fails < best:
            best, bestoff = fails, (float(o3), float(o4), 0.0)
LC["oracle_level_offset"] = {
    "baseline_failures": base_fail, "n_pairs": len(ps),
    "best_failures": best, "best_offsets_P3_P4_P5": bestoff,
    "recovered_pairs": base_fail - best,
    "★caveat": "real DEV 에 fit 한 oracle. 배포 성능 아님 — 'level bias 가 원인인가' 만 본다."}
json.dump(LC, open(f"{NS}/LEVEL_CALIBRATION_AUDIT.json", "w"), indent=2, ensure_ascii=False)

RA = {"REAL_RANKFAIL": REAL["RANKFAIL"], "SYNTH_RANKFAIL": SY["SYNTH_RANKFAIL"],
      "real_rankfail_rate": REAL["RANKFAIL"]["n"] / max(REAL["ALL"]["n"], 1),
      "synth_rankfail_rate": SY["SYNTH_RANKFAIL"]["rate"]}
json.dump(RA, open(f"{NS}/RANKFAIL_ANALYSIS.json", "w"), indent=2, ensure_ascii=False)
print("analysis 완료")
