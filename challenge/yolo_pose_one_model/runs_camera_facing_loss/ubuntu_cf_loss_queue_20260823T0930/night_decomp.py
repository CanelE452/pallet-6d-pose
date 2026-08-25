"""NIGHT FAILURE DECOMPOSITION — night correct-box=0 이 어느 단계에서 깨지는가.

★ 새 학습 없음. 새 split 없음. correct-box rule 을 새로 만들지 않는다
  (기존 cf_real_eval.py 와 동일: IoU >= 0.5, conf=0.001, pad100 REFLECT_101, top-1 by conf).
★ primary metric 은 계속 top-1.  all-candidate 는 진단 전용이다.
"""
import collections, hashlib, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Y = f"{ROOT}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/REAL_GT_QA_20260821T133405Z/"
        "REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
FTW = "/home/minjae/Documents/github/25y_automatic_lifter-master/pallet_yolo26n_pose_ft.pt"
PAD, IOU_T, CONF, GROSS = 100, 0.5, 0.001, 20.0     # 전부 기존 계약
NIGHT = {"eval_night08", "eval_night09"}

import cv2
import torch
from ultralytics import YOLO

MODELS = [("A42", f"{CFR}/CF_A0_V1MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt"),
          ("C42", f"{CFR}/CF_DATA_C_V2_EARLY10K_STD_60EP_SEED42_UBUNTU/weights/last.pt"),
          ("C43", f"{CFR}/CF_DATA_C_V2_EARLY10K_STD_60EP_SEED43_UBUNTU/weights/last.pt"),
          ("E42", f"{CFR}/CF_DATA_E_V2_COMPLETE12K5_STD_60EP_SEED42_UBUNTU/weights/last.pt")]
FT_OK = os.path.exists(FTW)
if FT_OK:
    MODELS.append(("FT", FTW))

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
idx = sorted(rng.choice(len(day_all), size=min(len(night), len(day_all)), replace=False))
day = [day_all[i] for i in idx]                      # deterministic day control
TARGET = night + day
print(f"  night {len(night)}  day-control {len(day)}  (seed 42 deterministic)")


def gt_of(jp):
    o = json.load(open(jp))["objects"][0]
    g = np.array(o["projected_cuboid"], dtype=float)[:8]
    return g, [g[:, 0].min(), g[:, 1].min(), g[:, 0].max(), g[:, 1].max()]


def iou(b, g):
    xx = max(0.0, min(b[2], g[2]) - max(b[0], g[0]))
    yy = max(0.0, min(b[3], g[3]) - max(b[1], g[1]))
    inter = xx * yy
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
    return inter / max(ua, 1e-9)


PER = []
for tag, w in MODELS:
    if not os.path.exists(w):
        print(f"  {tag}: weight 없음 — 건너뜀")
        continue
    m = YOLO(w, task="pose")
    for r in TARGET:
        im = cv2.imread(r["ip"])
        luma = float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean())
        p = cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        res = m.predict(p, conf=CONF, imgsz=640, device=0, verbose=False)[0]
        gt8, gb = gt_of(r["jp"])
        gt_area = (gb[2] - gb[0]) * (gb[3] - gb[1])
        rec = {"frame_id": r["fid"], "model": tag, "domain": r["domain"],
               "session": r["session"], "mean_luma": round(luma, 2),
               "gt_bbox_area": round(gt_area, 1)}
        if res.boxes is None or len(res.boxes) == 0:
            rec.update({"n_detections": 0, "failure_class": "N0",
                        "any_correct_candidate": False})
            PER.append(rec)
            continue
        conf = res.boxes.conf.cpu().numpy()
        order = np.argsort(-conf)                                    # conf 내림차순
        boxes = res.boxes.xyxy.cpu().numpy() - PAD
        kps = res.keypoints.xy.cpu().numpy() - PAD if res.keypoints is not None else None
        ious = np.array([iou(boxes[i], gb) for i in order])
        ok = ious >= IOU_T
        i0 = order[0]
        rec.update({"n_detections": int(len(conf)), "top1_conf": float(conf[i0]),
                    "top1_box": [round(float(x), 1) for x in boxes[i0]],
                    "top1_iou": float(ious[0]), "top1_correct_box": bool(ious[0] >= IOU_T),
                    "pred_bbox_area": round(float((boxes[i0][2] - boxes[i0][0]) *
                                                  (boxes[i0][3] - boxes[i0][1])), 1),
                    "any_correct_candidate": bool(ok.any()),
                    "best_candidate_iou": float(ious.max())})
        if ok.any():
            k = int(np.argmax(ok))
            rec["first_correct_rank"] = k + 1
            rec["first_correct_conf"] = float(conf[order[k]])
        if rec["top1_correct_box"] and kps is not None:
            e = np.linalg.norm(kps[i0][:8] - gt8, axis=1)
            rec.update({"valid_kp_count": 8,          # 현 계약엔 kp-valid gate 가 없다
                        "kp_median_px": float(np.median(e)), "kp_max_px": float(e.max()),
                        "front_error": float(np.median(e[[0, 1, 2, 3]])),
                        "rear_error": float(np.median(e[[4, 5, 6, 7]])),
                        "bottom_error": float(np.median(e[[2, 3, 6, 7]]))})
            # N4 성공 = 기존 gross20 임계(20px)를 재사용. 새 threshold 만들지 않는다.
            rec["failure_class"] = "N4" if rec["kp_median_px"] <= GROSS else "N3"
        elif rec["any_correct_candidate"]:
            rec["failure_class"] = "N1B"             # 정답 후보는 있는데 top1 이 아님
        else:
            rec["failure_class"] = "N1A"             # 검출은 있으나 정답 후보 없음
        PER.append(rec)
    del m
    torch.cuda.empty_cache()
    print(f"  {tag} 완료")

# ---- CSV ---------------------------------------------------------------------
cols = ["frame_id", "model", "domain", "session", "n_detections", "top1_conf", "top1_iou",
        "top1_correct_box", "any_correct_candidate", "best_candidate_iou",
        "first_correct_rank", "first_correct_conf", "failure_class", "valid_kp_count",
        "kp_median_px", "kp_max_px", "front_error", "rear_error", "bottom_error",
        "gt_bbox_area", "pred_bbox_area", "mean_luma", "top1_box"]
import csv
with open(f"{Q}/NIGHT_FAILURE_PER_FRAME.csv", "w", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w_.writeheader()
    w_.writerows(PER)

CLS = ["N0", "N1A", "N1B", "N2", "N3", "N4"]


def table(dm):
    out = {}
    for tag, _ in MODELS:
        rs = [r for r in PER if r["model"] == tag and r["domain"] == dm]
        if not rs:
            continue
        c = collections.Counter(r["failure_class"] for r in rs)
        n = len(rs)
        out[tag] = {"n": n, **{k: c.get(k, 0) for k in CLS},
                    "any_det": float(np.mean([r["n_detections"] > 0 for r in rs])),
                    "any_cbox": float(np.mean([r.get("any_correct_candidate", False) for r in rs])),
                    "top1_cbox": float(np.mean([r.get("top1_correct_box", False) for r in rs])),
                    "best_iou_median": float(np.median([r.get("best_candidate_iou", 0.0) for r in rs])),
                    "reached_kp": int(c.get("N3", 0) + c.get("N4", 0))}
        kp = [r["kp_median_px"] for r in rs if "kp_median_px" in r]
        out[tag]["kp_median"] = float(np.median(kp)) if kp else None
        out[tag]["kp_p90"] = float(np.percentile(kp, 90)) if kp else None
    return out


NT, DT = table("NIGHT"), table("DAY")

# ---- ranking histogram --------------------------------------------------------
hist = {}
for tag, _ in MODELS:
    rs = [r for r in PER if r["model"] == tag and r["domain"] == "NIGHT"]
    if not rs:
        continue
    h = collections.Counter()
    for r in rs:
        k = r.get("first_correct_rank")
        h["not_present" if k is None else ("rank1" if k == 1 else "rank2" if k == 2 else "rank3+")] += 1
    hist[tag] = dict(h)
json.dump(hist, open(f"{Q}/NIGHT_RANKING_HIST.json", "w"), indent=2)

# ---- 교차표 (association only) --------------------------------------------------
def cross(tag, key, bins=None):
    rs = [r for r in PER if r["model"] == tag and r["domain"] == "NIGHT"]
    o = collections.defaultdict(collections.Counter)
    for r in rs:
        v = r.get(key)
        if bins is not None and v is not None:
            v = f"Q{int(np.searchsorted(bins, v))}"
        o[str(v)][r["failure_class"]] += 1
    return {k: dict(v) for k, v in o.items()}


lum = [r["mean_luma"] for r in PER if r["domain"] == "NIGHT" and r["model"] == MODELS[0][0]]
ar = [r["gt_bbox_area"] for r in PER if r["domain"] == "NIGHT" and r["model"] == MODELS[0][0]]
lb = np.percentile(lum, [25, 50, 75]) if lum else []
ab = np.percentile(ar, [25, 50, 75]) if ar else []
CROSS = {t: {"luma_quartile": cross(t, "mean_luma", lb),
             "bbox_size_quartile": cross(t, "gt_bbox_area", ab),
             "session": cross(t, "session")} for t, _ in MODELS
         if any(r["model"] == t for r in PER)}

# ---- verdict --------------------------------------------------------------------
prim = "C42"
p = NT.get(prim, {})
tot = max(p.get("n", 1), 1)
frac = {k: p.get(k, 0) / tot for k in CLS}
dom_cls = max(frac, key=frac.get)
dominant = frac[dom_cls] >= 0.6
VERD = {"N0": "NIGHT_DETECTION_FAILURE", "N1A": "NIGHT_LOCALIZATION_DISTRACTOR_FAILURE",
        "N1B": "NIGHT_RANKING_FAILURE", "N2": "NIGHT_KEYPOINT_FAILURE",
        "N3": "NIGHT_KEYPOINT_FAILURE", "N4": "NIGHT_OK"}
verdict = VERD[dom_cls] if dominant else "MIXED_NIGHT_FAILURE"
seed_robust = (NT.get("C42", {}).get("top1_cbox", 0) == 0
               and NT.get("C43", {}).get("top1_cbox", 0) == 0)
data_helped = NT.get("E42", {}).get("any_cbox", 0) > NT.get("C42", {}).get("any_cbox", 0)
ft_solves = (NT.get("FT", {}).get("top1_cbox", 0) > 0) if FT_OK else None
kp_not_reached = all(v.get("reached_kp", 0) == 0 for v in NT.values())

out = {"night_membership": {"n": len(night),
                            "sessions": dict(collections.Counter(r["session"] for r in night)),
                            "sha16": hashlib.sha256(
                                "\n".join(sorted(r["fid"] for r in night)).encode()).hexdigest()[:16]},
       "contract": {"correct_box_rule": f"IoU >= {IOU_T} (기존 cf_real_eval.py 그대로)",
                    "conf": CONF, "pad": PAD, "border": "BORDER_REFLECT_101",
                    "primary_metric": "top-1 by box conf (변경 없음)",
                    "all_candidate": "진단 전용",
                    "N4_rule": f"top1 correct-box AND frame median kp err <= {GROSS}px "
                               f"(기존 gross20 임계 재사용, 새 threshold 아님)",
                    "N2_note": "현 평가 계약에 kp-valid gate 가 없어 N2 는 도달 불가 — 항상 0"},
       "night_cascade": NT, "day_control_cascade": DT,
       "ranking_hist_night": hist, "cross_tab_night": CROSS,
       "KEYPOINT_FAILURE_NOT_REACHED": bool(kp_not_reached),
       "NIGHT_FAILURE_TYPE": verdict, "dominant_class": dom_cls,
       "dominant_fraction": frac[dom_cls], "dominant_threshold": 0.6,
       "NIGHT_FAILURE_SEED_ROBUST": bool(seed_robust),
       "DATA_SCALE_HELPED_NIGHT": bool(data_helped),
       "FT_REFERENCE": ("AVAILABLE" if FT_OK else "UNAVAILABLE"),
       "FT_SOLVES_NIGHT": ft_solves}
json.dump(out, open(f"{Q}/NIGHT_FAILURE_DECOMPOSITION.json", "w"), indent=2, ensure_ascii=False)

L = ["# NIGHT FAILURE DECOMPOSITION", "",
     f"membership n={len(night)} sha16={out['night_membership']['sha16']}  "
     f"sessions {out['night_membership']['sessions']}",
     f"correct-box rule = IoU >= {IOU_T} (기존 계약 그대로), conf={CONF}, pad={PAD}", "",
     "## CASCADE (NIGHT)", "```",
     f"{'model':6} {'n':>4} {'N0':>5} {'N1A':>5} {'N1B':>5} {'N2':>4} {'N3':>4} {'N4':>4}", "-" * 44]
for t, v in NT.items():
    L.append(f"{t:6} {v['n']:4d} {v['N0']:5d} {v['N1A']:5d} {v['N1B']:5d} "
             f"{v['N2']:4d} {v['N3']:4d} {v['N4']:4d}")
L += ["```", "", "## BOX ROUTING (NIGHT)  ★가장 중요", "```",
      f"{'model':6} {'any-det':>9} {'any-cbox':>10} {'top1-cbox':>10} {'best IoU med':>13}", "-" * 52]
for t, v in NT.items():
    L.append(f"{t:6} {v['any_det']:9.3f} {v['any_cbox']:10.3f} {v['top1_cbox']:10.3f} "
             f"{v['best_iou_median']:13.3f}")
L += ["```", "", "## DAY CONTROL (동일 cascade)", "```",
      f"{'model':6} {'n':>4} {'N0':>5} {'N1A':>5} {'N1B':>5} {'N3':>4} {'N4':>4} "
      f"{'top1-cbox':>10} {'kp med':>8}", "-" * 62]
for t, v in DT.items():
    L.append(f"{t:6} {v['n']:4d} {v['N0']:5d} {v['N1A']:5d} {v['N1B']:5d} {v['N3']:4d} "
             f"{v['N4']:4d} {v['top1_cbox']:10.3f} "
             f"{(v['kp_median'] if v['kp_median'] is not None else float('nan')):8.2f}")
L += ["```", "", "## RANKING (night, 정답 후보의 순위)", "```", str(hist), "```", "",
      f"**NIGHT_FAILURE_TYPE = {verdict}**  (dominant {dom_cls} {100*frac[dom_cls]:.0f}%)", "",
      f"- KEYPOINT_FAILURE_NOT_REACHED = {kp_not_reached}",
      f"- NIGHT_FAILURE_SEED_ROBUST = {seed_robust}",
      f"- DATA_SCALE_HELPED_NIGHT = {data_helped}",
      f"- FT_REFERENCE = {out['FT_REFERENCE']}   FT_SOLVES_NIGHT = {ft_solves}", "",
      "association only — 인과 주장 아님. 새 학습·새 loss 없음."]
open(f"{Q}/NIGHT_FAILURE_DECOMPOSITION.md", "w").write("\n".join(L))
print("\n".join(L))
