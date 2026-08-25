"""NIGHT RANKING AUDIT — FT 가 무엇을 바꿔 ranking 을 풀었는가.

새 학습 없음. 기존 checkpoint 추론만. threshold tune 금지.
correct-box rule / conf / padding 전부 기존 계약 그대로.
"""
import collections, csv, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/REAL_GT_QA_20260821T133405Z/"
        "REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
FTW = "/home/minjae/Documents/github/25y_automatic_lifter-master/pallet_yolo26n_pose_ft.pt"
PAD, IOU_T, CONF = 100, 0.5, 0.001
NIGHT = {"eval_night08", "eval_night09"}

import cv2
import torch
from ultralytics import YOLO

MODELS = [("A42", f"{CFR}/CF_A0_V1MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt"),
          ("C42", f"{CFR}/CF_DATA_C_V2_EARLY10K_STD_60EP_SEED42_UBUNTU/weights/last.pt"),
          ("C43", f"{CFR}/CF_DATA_C_V2_EARLY10K_STD_60EP_SEED43_UBUNTU/weights/last.pt"),
          ("E42", f"{CFR}/CF_DATA_E_V2_COMPLETE12K5_STD_60EP_SEED42_UBUNTU/weights/last.pt"),
          ("FT", FTW)]

man = json.load(open(MANI))
rows = []
for it in man["items"]:
    jp, ip = os.path.join(ROOT, it["label"]), os.path.join(ROOT, it["image"])
    if os.path.exists(jp) and os.path.exists(ip):
        s = it.get("set", "?")
        rows.append({"fid": it["frame_id"], "session": s,
                     "domain": "NIGHT" if s in NIGHT else "DAY", "jp": jp, "ip": ip})
night = [r for r in rows if r["domain"] == "NIGHT"]
day_all = sorted([r for r in rows if r["domain"] == "DAY"], key=lambda r: r["fid"])
rng = np.random.default_rng(42)
day = [day_all[i] for i in sorted(rng.choice(len(day_all), len(night), replace=False))]
TARGET = night + day


def gt_box(jp):
    g = np.array(json.load(open(jp))["objects"][0]["projected_cuboid"], dtype=float)[:8]
    return [g[:, 0].min(), g[:, 1].min(), g[:, 0].max(), g[:, 1].max()]


def iou(b, g):
    xx = max(0.0, min(b[2], g[2]) - max(b[0], g[0]))
    yy = max(0.0, min(b[3], g[3]) - max(b[1], g[1]))
    it_ = xx * yy
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - it_
    return it_ / max(ua, 1e-9)


CAND = []          # 모든 candidate
FRAME = {}         # (model, fid) -> 요약
for tag, w in MODELS:
    if not os.path.exists(w):
        print(f"  {tag} weight 없음 — 건너뜀")
        continue
    m = YOLO(w, task="pose")
    for r in TARGET:
        im = cv2.imread(r["ip"])
        p = cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        res = m.predict(p, conf=CONF, imgsz=640, device=0, verbose=False)[0]
        gb = gt_box(r["jp"])
        f = {"model": tag, "frame_id": r["fid"], "domain": r["domain"],
             "session": r["session"], "n_cand": 0}
        if res.boxes is not None and len(res.boxes):
            cf = res.boxes.conf.cpu().numpy()
            bx = res.boxes.xyxy.cpu().numpy() - PAD
            kc = (res.keypoints.conf.cpu().numpy()
                  if (res.keypoints is not None and res.keypoints.conf is not None) else None)
            order = np.argsort(-cf)
            cor_c, wrong_c = [], []
            for rk, i in enumerate(order, 1):
                v = float(iou(bx[i], gb))
                ok = v >= IOU_T
                (cor_c if ok else wrong_c).append(float(cf[i]))
                CAND.append({"model": tag, "frame_id": r["fid"], "domain": r["domain"],
                             "rank": rk, "conf": float(cf[i]), "iou": round(v, 4),
                             "correct_box": ok,
                             "bbox": [round(float(x), 1) for x in bx[i]],
                             "bbox_area": round(float((bx[i][2]-bx[i][0])*(bx[i][3]-bx[i][1])), 1),
                             "bbox_aspect": round(float((bx[i][2]-bx[i][0]) /
                                                        max(bx[i][3]-bx[i][1], 1e-6)), 3),
                             "kp_conf_mean": (round(float(kc[i].mean()), 4)
                                              if kc is not None else None)})
            f.update({"n_cand": int(len(cf)), "top1_conf": float(cf[order[0]]),
                      "top1_iou": float(iou(bx[order[0]], gb)),
                      "top1_is_correct": bool(iou(bx[order[0]], gb) >= IOU_T),
                      "any_correct": bool(cor_c),
                      "best_correct_conf": (max(cor_c) if cor_c else None),
                      "best_wrong_conf": (max(wrong_c) if wrong_c else None)})
            if cor_c:
                f["correct_rank"] = int(next(rk for rk, i in enumerate(order, 1)
                                             if iou(bx[i], gb) >= IOU_T))
            if cor_c and wrong_c:
                f["margin"] = max(cor_c) - max(wrong_c)
            elif cor_c:
                f["margin"] = max(cor_c)          # 경쟁 wrong 후보 없음
        FRAME[(tag, r["fid"])] = f
    del m
    torch.cuda.empty_cache()
    print(f"  {tag} 완료  (candidate {sum(1 for c in CAND if c['model']==tag)})")

# ---- PHASE 2 margin -----------------------------------------------------------
def margin_summary(tag, dm):
    v = [f["margin"] for (t, _), f in FRAME.items()
         if t == tag and f["domain"] == dm and f.get("margin") is not None]
    if not v:
        return {"n": 0}
    a = np.array(v)
    return {"n": len(a), "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
            "positive_frac": float((a > 0).mean())}


NMS = {t: margin_summary(t, "NIGHT") for t, _ in MODELS if any(k[0] == t for k in FRAME)}
DMS = {t: margin_summary(t, "DAY") for t, _ in MODELS if any(k[0] == t for k in FRAME)}
json.dump({"night": NMS, "contract": f"margin = best_correct_conf - best_wrong_conf, IoU>={IOU_T}"},
          open(f"{Q}/NIGHT_RANKING_MARGIN_SUMMARY.json", "w"), indent=2)
json.dump({"day_control": DMS, "n_day": len(day), "sample": "seed 42 deterministic"},
          open(f"{Q}/DAY_RANKING_MARGIN_SUMMARY.json", "w"), indent=2)

with open(f"{Q}/NIGHT_RANKING_MARGIN.csv", "w", newline="") as fh:
    cols = ["model", "frame_id", "domain", "session", "n_cand", "top1_conf", "top1_iou",
            "top1_is_correct", "any_correct", "correct_rank", "best_correct_conf",
            "best_wrong_conf", "margin"]
    w_ = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w_.writeheader()
    for (t, fid), f in sorted(FRAME.items()):
        if f["domain"] == "NIGHT":
            w_.writerow(f)

# ---- PHASE 3 transition C43 -> FT ---------------------------------------------
TR = collections.Counter()
T0 = []
for r in night:
    c = FRAME.get(("C43", r["fid"]), {})
    f = FRAME.get(("FT", r["fid"]), {})
    cc, fc = c.get("top1_is_correct"), f.get("top1_is_correct")
    k = ("T1" if (cc and fc) else "T0" if (not cc and fc)
         else "T3" if (cc and not fc) else "T2")
    TR[k] += 1
    if k == "T0":
        T0.append({"frame_id": r["fid"], "session": r["session"],
                   "C43_top1_conf": c.get("top1_conf"), "C43_top1_iou": c.get("top1_iou"),
                   "C43_correct_conf": c.get("best_correct_conf"),
                   "C43_correct_rank": c.get("correct_rank"), "C43_margin": c.get("margin"),
                   "FT_top1_conf": f.get("top1_conf"), "FT_correct_conf": f.get("best_correct_conf"),
                   "FT_best_wrong_conf": f.get("best_wrong_conf"), "FT_margin": f.get("margin")})

# ---- PHASE 5 conf 변화 분해 (같은 프레임 paired) --------------------------------
def paired(dm, key):
    a, b = [], []
    for r in (night if dm == "NIGHT" else day):
        c = FRAME.get(("C43", r["fid"]), {}).get(key)
        f = FRAME.get(("FT", r["fid"]), {}).get(key)
        if c is not None and f is not None:
            a.append(c)
            b.append(f)
    if not a:
        return None
    a, b = np.array(a), np.array(b)
    return {"n": len(a), "C43_median": float(np.median(a)), "FT_median": float(np.median(b)),
            "delta_median": float(np.median(b - a)),
            "FT_higher_frac": float((b > a).mean())}


CHG = {"correct_candidate_conf": paired("NIGHT", "best_correct_conf"),
       "wrong_candidate_conf": paired("NIGHT", "best_wrong_conf"),
       "top1_conf": paired("NIGHT", "top1_conf")}

# ---- PHASE 7 separation --------------------------------------------------------
def sep(tag, dm):
    ok = [c["conf"] for c in CAND if c["model"] == tag and c["domain"] == dm and c["correct_box"]]
    ng = [c["conf"] for c in CAND if c["model"] == tag and c["domain"] == dm and not c["correct_box"]]
    fr = [f for (t, _), f in FRAME.items() if t == tag and f["domain"] == dm
          and f.get("best_correct_conf") is not None and f.get("best_wrong_conf") is not None]
    return {"n_correct_cand": len(ok), "n_wrong_cand": len(ng),
            "correct_conf_median": float(np.median(ok)) if ok else None,
            "wrong_conf_median": float(np.median(ng)) if ng else None,
            "P_correct_gt_wrong_framelevel": (
                float(np.mean([f["best_correct_conf"] > f["best_wrong_conf"] for f in fr]))
                if fr else None), "n_frames_both": len(fr)}


SEP = {t: {"NIGHT": sep(t, "NIGHT"), "DAY": sep(t, "DAY")}
       for t, _ in MODELS if any(k[0] == t for k in FRAME)}

# ---- PHASE 8 routing ------------------------------------------------------------
dc = CHG["correct_candidate_conf"]
dw = CHG["wrong_candidate_conf"]
BIG = 0.05
up_c = dc and dc["delta_median"] >= BIG
dn_w = dw and dw["delta_median"] <= -BIG
if dn_w and not up_c:
    case, route = "R1", "NEGATIVE_SCREEN"
elif up_c and not dn_w:
    case, route = "R2", "PSEUDO_POSITIVE_SCREEN"
elif up_c and dn_w:
    case, route = "R3", "2x2_ADAPTATION_SCREEN"
else:
    case, route = "R4", "MORE_AUDIT"

# ---- PHASE 9 self-training coverage (C43 만, filter 미적용) -----------------------
def st_cov(dm):
    fr = [f for (t, _), f in FRAME.items() if t == "C43" and f["domain"] == dm]
    n = len(fr)
    multi = [f for f in fr if f.get("margin") is not None and abs(f["margin"]) < 0.10]
    return {"n": n,
            "frames_any_candidate": sum(f["n_cand"] > 0 for f in fr),
            "frames_any_correct": sum(bool(f.get("any_correct")) for f in fr),
            "frames_top1_correct": sum(bool(f.get("top1_is_correct")) for f in fr),
            "top1_conf_median": float(np.median([f["top1_conf"] for f in fr if "top1_conf" in f])),
            "correct_but_not_top1": sum(1 for f in fr
                                        if f.get("any_correct") and not f.get("top1_is_correct")),
            "close_competition_margin_lt_0.10": len(multi)}


ST = {"NIGHT": st_cov("NIGHT"), "DAY": st_cov("DAY")}
ST["★top1_only_pseudo_label_risk"] = (
    f"NIGHT 에서 correct 후보가 있는데 top1 이 아닌 프레임 "
    f"{ST['NIGHT']['correct_but_not_top1']}/{ST['NIGHT']['n']} — "
    f"top1 만 pseudo-label 로 쓰면 그만큼 distractor 를 강화한다")

json.dump({"transition_C43_to_FT": dict(TR), "T0_frames": T0,
           "conf_change_paired_night": CHG, "separation": SEP,
           "night_margin": NMS, "day_margin": DMS,
           "ROUTING_CASE": case, "NEXT_ONE_ACTION": route,
           "self_training_coverage_C43": ST,
           "★caveat": ("FT 는 real positive + negative + synthetic 을 동시에 썼다. "
                       "margin 변화만으로 'negative 가 원인' 이라고 인과 확정하지 않는다. "
                       "observational evidence 만 기록한다."),
           "contract": {"iou": IOU_T, "conf": CONF, "pad": PAD,
                        "big_delta_threshold": BIG, "n_night": len(night), "n_day": len(day)}},
          open(f"{Q}/ADAPTATION_ROUTING.json", "w"), indent=2, ensure_ascii=False)

json.dump(CAND[:20000], open(f"{Q}/NIGHT_CANDIDATES_SAMPLE.json", "w"))

L = ["# FT RANKING CHANGE ANALYSIS", "",
     f"NIGHT {len(night)} · DAY control {len(day)} (seed42) · IoU>={IOU_T} · conf={CONF} · pad={PAD}",
     "", "## margin = best_correct_conf − best_wrong_conf", "```",
     f"{'model':6} {'n':>4} {'p10':>8} {'p25':>8} {'median':>8} {'p75':>8} {'pos%':>7}", "-" * 50]
for t, v in NMS.items():
    if v.get("n"):
        L.append(f"{t:6} {v['n']:4d} {v['p10']:8.4f} {v['p25']:8.4f} {v['median']:8.4f} "
                 f"{v['p75']:8.4f} {100*v['positive_frac']:6.1f}%")
L += ["```", "", "## DAY control margin", "```",
      f"{'model':6} {'n':>4} {'median':>8} {'pos%':>7}", "-" * 30]
for t, v in DMS.items():
    if v.get("n"):
        L.append(f"{t:6} {v['n']:4d} {v['median']:8.4f} {100*v['positive_frac']:6.1f}%")
L += ["```", "", f"## C43 → FT transition (night): {dict(TR)}",
      "  T0 = C43 wrong → FT correct (핵심), T1 둘 다 correct, T2 둘 다 wrong, T3 역행", "",
      "## conf 변화 (같은 프레임 paired, C43 → FT)", "```"]
for k, v in CHG.items():
    if v:
        L.append(f"{k:26} n{v['n']:3d}  C43 {v['C43_median']:.4f} → FT {v['FT_median']:.4f}  "
                 f"Δmed {v['delta_median']:+.4f}  FT우세 {100*v['FT_higher_frac']:.0f}%")
L += ["```", "", "## candidate separation (frame-level P(correct conf > wrong conf))", "```",
      f"{'model':6} {'NIGHT':>8} {'DAY':>8}", "-" * 26]
for t, v in SEP.items():
    n_, d_ = v["NIGHT"]["P_correct_gt_wrong_framelevel"], v["DAY"]["P_correct_gt_wrong_framelevel"]
    L.append(f"{t:6} {('  n/a' if n_ is None else f'{n_:8.3f}')} "
             f"{('  n/a' if d_ is None else f'{d_:8.3f}')}")
L += ["```", "", f"**ROUTING_CASE = {case}   NEXT_ONE_ACTION = {route}**", "",
      f"self-training 위험: NIGHT correct-but-not-top1 "
      f"{ST['NIGHT']['correct_but_not_top1']}/{ST['NIGHT']['n']}", "",
      "★ FT 는 real positive + negative + synthetic 동시 사용 — 인과 확정 금지. observational only."]
open(f"{Q}/FT_RANKING_CHANGE_ANALYSIS.md", "w").write("\n".join(L))
print("\n".join(L))
