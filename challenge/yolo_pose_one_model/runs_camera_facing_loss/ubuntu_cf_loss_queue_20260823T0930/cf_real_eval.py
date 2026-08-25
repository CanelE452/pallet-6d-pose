"""camera-facing REAL 평가 — real GT 가 이미 camera-facing 이므로 순열 몫을 쓰지 않는다.

★ 검증된 recipe: pad=100 + BORDER_REFLECT_101, imgsz 640, top-1 by box conf.
★ 세 모집단을 섞지 않는다: per-model available / A0 공통 intersection / correct-box.
★ day/night 를 합쳐 평균만 내지 않는다.
"""
import argparse, hashlib, json, os, sys, collections
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
Q = os.path.dirname(os.path.abspath(__file__))
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/REAL_GT_QA_20260821T133405Z/"
        "REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
PAD = 100

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--conf_deploy", type=float, default=0.4)
A = ap.parse_args()

import cv2
import torch
from ultralytics import YOLO

man = json.load(open(MANI))
rows = []
for it in man["items"]:
    jp = os.path.join(ROOT, it["label"])
    ip = os.path.join(ROOT, it["image"])
    if os.path.exists(jp) and os.path.exists(ip):
        rows.append((it.get("set", "?"), it["frame_id"], jp, ip))
if not rows:
    raise SystemExit("membership 0")

NIGHT = {"eval_night08", "eval_night09"}


def dom(s):
    return "NIGHT" if s in NIGHT else "DAY"


m = YOLO(A.weights, task="pose")
pred = {}
for sess, fid, jp, ip in rows:
    im = cv2.imread(ip)
    p = cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    r = m.predict(p, conf=0.001, imgsz=640, device=0, verbose=False)[0]   # threshold-free
    if r.keypoints is None or len(r.boxes) == 0:
        pred[fid] = None
        continue
    cf = r.boxes.conf.cpu().numpy()
    i = int(np.argmax(cf))
    pred[fid] = {"conf": float(cf[i]),
                 "kps": (r.keypoints.xy.cpu().numpy()[i] - PAD),
                 "box": (r.boxes.xyxy.cpu().numpy()[i] - PAD)}
del m
torch.cuda.empty_cache()

rec = []
for sess, fid, jp, ip in rows:
    o = json.load(open(jp))["objects"][0]
    gt = np.array(o["projected_cuboid"], dtype=float)[:8]
    gc = o.get("projected_cuboid_centroid")
    gb = [gt[:, 0].min(), gt[:, 1].min(), gt[:, 0].max(), gt[:, 1].max()]
    P = pred.get(fid)
    r = {"frame": fid, "session": sess, "domain": dom(sess),
         "present_top1": P is not None,
         "present_deploy": bool(P and P["conf"] >= A.conf_deploy)}
    if P is not None:
        b = P["box"]
        xx = max(0, min(b[2], gb[2]) - max(b[0], gb[0]))
        yy = max(0, min(b[3], gb[3]) - max(b[1], gb[1]))
        inter = xx * yy
        ua = (b[2] - b[0]) * (b[3] - b[1]) + (gb[2] - gb[0]) * (gb[3] - gb[1]) - inter
        iou = inter / max(ua, 1e-9)
        e = np.linalg.norm(P["kps"][:8] - gt, axis=1)
        r.update({"iou": float(iou), "correct_box": bool(iou >= 0.5),
                  "err": e.tolist(), "conf": P["conf"],
                  "centroid": (float(np.linalg.norm(P["kps"][8] - np.array(gc, dtype=float)))
                               if gc is not None else None)})
    rec.append(r)

det = [r for r in rec if r.get("err")]
cbox = [r for r in det if r["correct_box"]]


def agg(rs):
    if not rs:
        return {}
    e = np.concatenate([r["err"] for r in rs])
    E = np.array([r["err"] for r in rs])
    cen = [r["centroid"] for r in rs if r.get("centroid") is not None]
    d = {"n": len(rs), "corner_median": float(np.median(e)),
         "corner_p90": float(np.percentile(e, 90)),
         "gross20": float((e > 20).mean()), "gross40": float((e > 40).mean()),
         "bottom_p90": float(np.percentile(E[:, [2, 3, 6, 7]], 90)),
         "front_p90": float(np.percentile(E[:, [0, 1, 2, 3]], 90)),
         "rear_p90": float(np.percentile(E[:, [4, 5, 6, 7]], 90)),
         "centroid_p90": float(np.percentile(cen, 90)) if cen else None}
    for dm in ("DAY", "NIGHT"):
        s = [r for r in rs if r["domain"] == dm]
        if s:
            ee = np.concatenate([r["err"] for r in s])
            d[f"{dm.lower()}_p90"] = float(np.percentile(ee, 90))
            d[f"{dm.lower()}_median"] = float(np.median(ee))
            d[f"{dm.lower()}_n"] = len(s)
    return d


# ★ 세 모집단을 섞지 않는다.
#   paired      = A0 공통 detected (threshold-free — 검출실패가 섞인다)
#   cbox_paired = A0 공통 **correct-box** (IoU>=0.5 양쪽) — keypoint 품질 전용, GATE 는 이것
a0f = f"{Q}/REAL_A0.json"
paired = det
cbox_paired = cbox
if A.tag != "A0" and os.path.exists(a0f):
    a0pf = json.load(open(a0f))["per_frame"]
    keep = {r["frame"] for r in a0pf if r.get("err")}
    keepc = {r["frame"] for r in a0pf if r.get("correct_box")}
    paired = [r for r in det if r["frame"] in keep]
    cbox_paired = [r for r in cbox if r["frame"] in keepc]

boot = {}
if A.tag != "A0" and os.path.exists(a0f):
    a0 = {r["frame"]: r for r in json.load(open(a0f))["per_frame"]}
    fr = [r for r in cbox_paired if a0.get(r["frame"], {}).get("err")]   # GATE 모집단과 동일
    sess_of = {r["frame"]: r["session"] for r in fr}
    sessions = sorted(set(sess_of.values()))
    def pf(r, k):
        e = np.array(r["err"])
        return {"median": float(np.median(e)), "gross20": float((e > 20).mean()),
                "bottom": float(np.percentile(e[[2, 3, 6, 7]], 90))}[k]
    rng = np.random.default_rng(0)
    for k in ("median", "gross20", "bottom"):
        d = np.array([pf(a0[r["frame"]], k) - pf(r, k) for r in fr])   # >0 = 후보 우세
        groups = [np.array([i for i, r in enumerate(fr) if r["session"] == s]) for s in sessions]
        bs = []
        for _ in range(10000):
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[j] for j in pick if len(groups[j])])
            bs.append(d[idx].mean() if len(idx) else 0.0)
        bs = np.array(bs)
        boot[k] = {"delta": float(d.mean()),
                   "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

out = {"tag": A.tag, "weights": A.weights, "n_total": len(rec),
       "detection_recall": float(np.mean([r["present_top1"] for r in rec])),
       "detection_recall_deploy": float(np.mean([r["present_deploy"] for r in rec])),
       "correct_box_recall": float(np.mean([r.get("correct_box", False) for r in rec])),
       "available": agg(det), "correct_box": agg(cbox), "paired": agg(paired),
       "cbox_paired": agg(cbox_paired),
       "GATE_POPULATION": "cbox_paired (A0 공통 correct-box) — 검출실패가 keypoint 신호를 덮지 않게",
       "bootstrap": boot, "sessions": sorted({r["session"] for r in rec}),
       "recipe": {"pad": PAD, "border": "BORDER_REFLECT_101", "imgsz": 640,
                  "selection": "top-1 by box conf", "conf_deploy": A.conf_deploy},
       "★note": ("real GT 가 camera-facing 이므로 순열 몫 없이 직접 비교. "
                 "PnP 6D 는 GT-independent W/D 선택이 없어 POSE_EVAL_BLOCKED."),
       "POSE_EVAL": "BLOCKED_NO_GT_INDEPENDENT_WD_SELECTOR",
       "per_frame": rec}
json.dump(out, open(f"{Q}/REAL_{A.tag}.json", "w"), ensure_ascii=False)
p = out["cbox_paired"]
print(f"{A.tag:12} det {out['detection_recall']:.3f} cbox {out['correct_box_recall']:.3f} "
      f"| GATE(cbox_paired) n{p.get('n',0)} med {p.get('corner_median',float('nan')):.2f} "
      f"p90 {p.get('corner_p90',float('nan')):.2f} bottom {p.get('bottom_p90',float('nan')):.2f}")
