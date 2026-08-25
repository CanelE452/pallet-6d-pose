"""REAL POS/NEG 분리능력 — 새 학습 0, threshold tune 0, inference only.

★ negative 2,689 중 259 는 FT 학습에 쓰였다 → PRIMARY 는 held-out 2,430.
★ ROC 단독 해석 금지 — AP/AUPRC + FP 지표를 같이 본다.
"""
import collections, csv, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
NEG = f"{ROOT}/data/pallet/raw_data/negative_real_20260823/rgb"
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/REAL_GT_QA_20260821T133405Z/"
        "REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
PAD, CONF = 100, 0.001
NIGHT = {"eval_night08", "eval_night09"}
LEAK = set(json.load(open(f"{Q}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
PROV = json.load(open(f"{Q}/NEGATIVE_PROVENANCE.json"))
FT_NEG = set(PROV["NEGATIVE_SESSION_OVERLAP"])
# 전체 겹침 재계산 (provenance 는 앞 5개만 저장)
ftids = {f[:-4].split("__")[-1] for f in os.listdir(f"{Y}/datasets/ft_a/labels/train")
         if f.startswith("neg__")}

import cv2
import torch
from ultralytics import YOLO

MODELS = [("A42", f"{CFR}/CF_A0_V1MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt"),
          ("C43", f"{CFR}/CF_DATA_C_V2_EARLY10K_STD_60EP_SEED43_UBUNTU/weights/last.pt"),
          ("G38", f"{CFR}/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt"),
          ("OLD", f"{Y}/runs/stage_a_synth_640_b32_seed42/weights/best.pt"),
          ("FT", "/home/minjae/Documents/github/25y_automatic_lifter-master/pallet_yolo26n_pose_ft.pt")]

man = json.load(open(MANI))
POS = []
for it in man["items"]:
    if it["frame_id"] in LEAK:
        continue
    jp, ip = os.path.join(ROOT, it["label"]), os.path.join(ROOT, it["image"])
    if os.path.exists(jp) and os.path.exists(ip):
        s = it.get("set", "?")
        POS.append({"frame": it["frame_id"], "img": ip, "json": jp,
                    "domain": "NIGHT" if s in NIGHT else "DAY"})
NEGF = sorted(os.listdir(NEG))
print(f"  positive {len(POS)}  negative {len(NEGF)}  (FT 학습 겹침 {len(ftids & {f[:-4] for f in NEGF})})")

ROWS = []
for tag, w in MODELS:
    if not os.path.exists(w):
        print(f"  {tag} weight 없음")
        continue
    m = YOLO(w, task="pose")
    for r in POS:
        im = cv2.imread(r["img"])
        p = cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        res = m.predict(p, conf=CONF, imgsz=640, device=0, verbose=False)[0]
        cf = res.boxes.conf.cpu().numpy() if (res.boxes is not None and len(res.boxes)) else np.array([])
        bx = (res.boxes.xyxy.cpu().numpy() - PAD) if cf.size else np.zeros((0, 4))
        ROWS.append({"frame": r["frame"], "domain": r["domain"], "label": 1, "model": tag,
                     "max_conf": float(cf.max()) if cf.size else 0.0,
                     "n_candidates": int(cf.size),
                     "confs": cf.tolist(), "boxes": bx.tolist(), "gt": r["json"]})
    for f in NEGF:
        im = cv2.imread(f"{NEG}/{f}")
        if im is None:
            continue
        p = cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        res = m.predict(p, conf=CONF, imgsz=640, device=0, verbose=False)[0]
        cf = res.boxes.conf.cpu().numpy() if (res.boxes is not None and len(res.boxes)) else np.array([])
        ROWS.append({"frame": f[:-4], "domain": "NEG", "label": 0, "model": tag,
                     "max_conf": float(cf.max()) if cf.size else 0.0,
                     "n_candidates": int(cf.size), "confs": cf.tolist(), "boxes": [],
                     "gt": None})
    del m
    torch.cuda.empty_cache()
    print(f"  {tag} 완료")

with open(f"{Q}/REAL_POS_NEG_SCORES.csv", "w", newline="") as fh:
    w_ = csv.DictWriter(fh, fieldnames=["frame", "domain", "label", "model",
                                        "max_conf", "n_candidates"], extrasaction="ignore")
    w_.writeheader()
    w_.writerows(ROWS)


def roc_ap(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    o = np.argsort(-s)
    y = y[o]
    P, N = y.sum(), (1 - y).sum()
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / max(P, 1)
    fpr = fp / max(N, 1)
    auroc = float(np.trapz(tpr, fpr))
    prec = tp / np.maximum(tp + fp, 1)
    rec = tpr
    # AP = sum over rank of P(k) * Δrecall  (sklearn average_precision_score 정의)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    i95 = int(np.searchsorted(rec, 0.95))
    fpr95 = float(fpr[min(i95, len(fpr) - 1)]) if len(fpr) else None
    return auroc, ap, fpr95, (fpr, tpr, rec, prec)


THR = [0.05, 0.10, 0.25, 0.40]
RES = {}
CURVES = {}
for scope, negfilter in (("HELDOUT", lambda f: f not in ftids), ("FULL", lambda f: True)):
    RES[scope] = {}
    for tag, _ in MODELS:
        rs = [r for r in ROWS if r["model"] == tag and
              (r["label"] == 1 or negfilter(r["frame"]))]
        if not rs:
            continue
        y = [r["label"] for r in rs]
        s = [r["max_conf"] for r in rs]
        auroc, ap, fpr95, cur = roc_ap(y, s)
        pos = np.array([r["max_conf"] for r in rs if r["label"] == 1])
        neg = np.array([r["max_conf"] for r in rs if r["label"] == 0])
        ncand = np.array([r["n_candidates"] for r in rs if r["label"] == 0])
        d = {"n_pos": int(len(pos)), "n_neg": int(len(neg)),
             "AUROC": auroc, "AP": ap, "FPR@TPR95": fpr95,
             "pos_p10": float(np.percentile(pos, 10)), "pos_median": float(np.median(pos)),
             "neg_median": float(np.median(neg)), "neg_p90": float(np.percentile(neg, 90)),
             "neg_p95": float(np.percentile(neg, 95)), "neg_p99": float(np.percentile(neg, 99)),
             "separation_margin": float(np.percentile(pos, 10) - np.percentile(neg, 90)),
             "neg_cand_mean": float(ncand.mean()), "neg_cand_median": float(np.median(ncand)),
             "thresholds": {}}
        for t in THR:
            tp = int((pos >= t).sum()); fn = int((pos < t).sum())
            fp = int((neg >= t).sum()); tn = int((neg < t).sum())
            fpi = float(np.mean([sum(1 for c in r["confs"] if c >= t)
                                 for r in rs if r["label"] == 0]))
            prec = tp / max(tp + fp, 1); rec_ = tp / max(tp + fn, 1)
            d["thresholds"][str(t)] = {
                "recall_TPR": rec_, "precision": prec,
                "FPR": fp / max(fp + tn, 1), "specificity": tn / max(fp + tn, 1),
                "F1": (2 * prec * rec_ / max(prec + rec_, 1e-9)),
                "neg_detect_rate": fp / max(fp + tn, 1), "FP_per_image": fpi}
        RES[scope][tag] = d
        if scope == "HELDOUT":
            CURVES[tag] = cur

# ---- day/night positive score ----
DN = {}
for tag, _ in MODELS:
    rs = [r for r in ROWS if r["model"] == tag]
    if not rs:
        continue
    DN[tag] = {}
    for dm in ("DAY", "NIGHT", "NEG"):
        v = np.array([r["max_conf"] for r in rs if r["domain"] == dm])
        if v.size:
            DN[tag][dm] = {"n": int(v.size), "median": float(np.median(v)),
                           "p10": float(np.percentile(v, 10)),
                           "p90": float(np.percentile(v, 90))}

# ---- detection AP (mixed) ----
def iou(b, g):
    xx = max(0.0, min(b[2], g[2]) - max(b[0], g[0]))
    yy = max(0.0, min(b[3], g[3]) - max(b[1], g[1]))
    i = xx * yy
    return i / max((b[2]-b[0])*(b[3]-b[1]) + (g[2]-g[0])*(g[3]-g[1]) - i, 1e-9)


DET = {}
for tag, _ in MODELS:
    rs = [r for r in ROWS if r["model"] == tag and (r["label"] == 1 or r["frame"] not in ftids)]
    if not rs:
        continue
    per_thr = {}
    for T in (0.5, *[round(0.5 + 0.05*k, 2) for k in range(1, 10)]):
        dets = []
        npos = 0
        for r in rs:
            if r["label"] == 1:
                npos += 1
                g = np.array(json.load(open(r["gt"]))["objects"][0]["projected_cuboid"],
                             dtype=float)[:8]
                gb = [g[:, 0].min(), g[:, 1].min(), g[:, 0].max(), g[:, 1].max()]
                used = False
                for c, b in sorted(zip(r["confs"], r["boxes"]), key=lambda x: -x[0]):
                    hit = (not used) and iou(b, gb) >= T
                    if hit:
                        used = True
                    dets.append((c, 1 if hit else 0))
            else:
                for c in r["confs"]:
                    dets.append((c, 0))
        dets.sort(key=lambda x: -x[0])
        tp = np.cumsum([d[1] for d in dets])
        fp = np.cumsum([1 - d[1] for d in dets])
        rec = tp / max(npos, 1)
        prec = tp / np.maximum(tp + fp, 1)
        per_thr[str(T)] = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    DET[tag] = {"AP50": per_thr["0.5"],
                "AP50_95": float(np.mean(list(per_thr.values()))),
                "note": "held-out negative 포함 mixed set. positive-only synthetic mAP 와 다르다."}

OUT = {"provenance": PROV, "ft_negative_overlap_n": len(ftids & {f[:-4] for f in NEGF}),
       "PRIMARY_SCOPE": "HELDOUT (FT 학습 negative 259 제외)",
       "image_level": RES, "day_night_score": DN, "detection_ap_mixed": DET,
       "ap_definition": "sum(Δrecall × precision) — sklearn average_precision_score 와 동일 정의",
       "★fairness": {"NEGATIVE_USED_IN_FT": True,
                     "FT_FULL_SCOPE_IS_IN_SAMPLE": True,
                     "note": "FULL scope 의 FT 수치는 IN-SAMPLE REFERENCE. honest winner 아님."},
       "★caveat": ["negative 촬영지가 사무실·교내라 팔레트 작업장 분포와 다르다",
                   "육안 검증 25장(전수 아님)", "세 블록 해상도 혼재",
                   "ROC 단독 해석 금지 — AP/FP 와 같이 볼 것"]}
json.dump(OUT, open(f"{Q}/ROC_PR_RESULTS.json", "w"), indent=2, ensure_ascii=False)

with open(f"{Q}/NEGATIVE_FP_TABLE.csv", "w", newline="") as fh:
    w_ = csv.writer(fh)
    w_.writerow(["scope", "model", "neg_detect@0.05", "@0.10", "@0.25", "@0.40",
                 "FP_per_image@0.40", "cand_per_img", "neg_conf_p90"])
    for sc in ("HELDOUT", "FULL"):
        for tag, d in RES[sc].items():
            t = d["thresholds"]
            w_.writerow([sc, tag, round(t["0.05"]["neg_detect_rate"], 4),
                         round(t["0.1"]["neg_detect_rate"], 4),
                         round(t["0.25"]["neg_detect_rate"], 4),
                         round(t["0.4"]["neg_detect_rate"], 4),
                         round(t["0.4"]["FP_per_image"], 4),
                         round(d["neg_cand_mean"], 2), round(d["neg_p90"], 4)])

# ---- curves ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for nm, idx, xl, yl in (("roc_curve.png", (0, 1), "FPR", "TPR"),
                            ("pr_curve.png", (2, 3), "Recall", "Precision")):
        plt.figure(figsize=(6, 5))
        for tag, cur in CURVES.items():
            plt.plot(cur[idx[0]], cur[idx[1]], label=tag, lw=1.5)
        plt.xlabel(xl); plt.ylabel(yl); plt.legend(); plt.grid(alpha=.3)
        plt.title(f"{yl} vs {xl} — HELDOUT negatives (FT 259 제외)")
        plt.tight_layout(); plt.savefig(f"{Q}/{nm}", dpi=110); plt.close()
except Exception as e:
    print(f"  curve 저장 실패(무시): {e}")

P = RES["HELDOUT"]
L = ["# REAL POS/NEG SEPARATION — HELDOUT primary", "",
     f"positive {len(POS)} (FT leak 12 제외) · negative held-out "
     f"{P[list(P)[0]]['n_neg']} (FT 학습 259 제외, 전체 {len(NEGF)})", "",
     "## IMAGE LEVEL (held-out)", "```",
     f"{'model':6} {'AUROC':>8} {'AP':>8} {'FPR@TPR95':>10} {'pos p10':>9} {'neg p90':>9} {'margin':>9}",
     "-" * 66]
for tag in ("A42", "C43", "G38", "OLD", "FT"):
    if tag not in P:
        continue
    d = P[tag]
    L.append(f"{tag:6} {d['AUROC']:8.4f} {d['AP']:8.4f} "
             f"{(d['FPR@TPR95'] if d['FPR@TPR95'] is not None else float('nan')):10.4f} "
             f"{d['pos_p10']:9.4f} {d['neg_p90']:9.4f} {d['separation_margin']:+9.4f}")
L += ["```", "", "## NEGATIVE suppression (held-out)", "```",
      f"{'model':6} {'det@.05':>8} {'@.25':>8} {'@.40':>8} {'FP/img@.40':>11} {'cand/img':>9}",
      "-" * 56]
for tag in ("A42", "C43", "G38", "OLD", "FT"):
    if tag not in P:
        continue
    t = P[tag]["thresholds"]
    L.append(f"{tag:6} {t['0.05']['neg_detect_rate']:8.4f} {t['0.25']['neg_detect_rate']:8.4f} "
             f"{t['0.4']['neg_detect_rate']:8.4f} {t['0.4']['FP_per_image']:11.4f} "
             f"{P[tag]['neg_cand_mean']:9.2f}")
L += ["```", "", "## DETECTION AP (mixed, held-out negatives)", "```",
      f"{'model':6} {'AP50':>9} {'AP50-95':>9}", "-" * 28]
for tag in ("A42", "C43", "G38", "OLD", "FT"):
    if tag in DET:
        L.append(f"{tag:6} {DET[tag]['AP50']:9.4f} {DET[tag]['AP50_95']:9.4f}")
L += ["```", "", "## POSITIVE score by domain vs NEGATIVE", "```",
      f"{'model':6} {'DAY med':>9} {'NIGHT med':>10} {'NEG med':>9} {'NEG p90':>9}", "-" * 50]
for tag in ("A42", "C43", "G38", "OLD", "FT"):
    if tag not in DN:
        continue
    g = lambda k, s: (f"{DN[tag][k][s]:9.4f}" if k in DN[tag] else "      n/a")
    L.append(f"{tag:6} {g('DAY','median')} {g('NIGHT','median'):>10} "
             f"{g('NEG','median')} {g('NEG','p90')}")
L += ["```", "",
      "★ negative 2,689 중 259 는 FT 학습에 사용됐다 → FULL scope 의 FT 수치는 IN-SAMPLE.",
      "  위 표는 그 259 를 제외한 held-out 이다.",
      "★ negative 촬영지는 사무실·교내 — 팔레트 작업장 분포가 아니다. 육안 검증 25장(전수 아님).",
      "★ ROC 단독 해석 금지."]
open(f"{Q}/ROC_PR_RESULTS.md", "w").write("\n".join(L))
print("\n".join(L))
