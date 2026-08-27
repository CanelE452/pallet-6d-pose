"""M0 parity + REAL/SYNTH paired TAL target audit.  training-0."""
from __future__ import annotations
import csv, hashlib, json, os, subprocess, sys, time

import numpy as np, cv2, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ta_core as TC                                                # noqa: E402

ROOT, NS, Y, W = TC.ROOT, TC.NS, TC.Y, TC.W
QY = f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
DS = f"{Y}/datasets/g38_generic_only"
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/"
        "REAL_GT_QA_20260821T133405Z/REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
NIGHT = {"eval_night08", "eval_night09"}
IOU_T = 0.5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gt_real(jp):
    o = json.load(open(jp))["objects"][0]
    g = np.array(o["projected_cuboid"], float)[:8]
    c = np.array(o["projected_cuboid_centroid"], float)
    return ([g[:, 0].min(), g[:, 1].min(), g[:, 0].max(), g[:, 1].max()], np.vstack([g, c]))


def gt_synth(lp, w, h):
    v = [float(x) for x in open(lp).read().strip().split("\n")[0].split()]
    cx, cy, bw, bh = v[1:5]
    k = np.array(v[5:]).reshape(-1, 3)
    return ([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h],
            np.stack([k[:, 0]*w, k[:, 1]*h], 1))


P = TC.TALProbe()
ROWS = []


def process(image_id, im, gt_box_img, gt_kp_img, domain, dset, pad):
    r, padded = P.predict(im, pad=pad)
    if r.boxes is None or not len(r.boxes):
        return None
    cf = r.boxes.conf.cpu().numpy()
    bx = r.boxes.xyxy.cpu().numpy()                 # padded 이미지 좌표계
    kp = r.keypoints.xy.cpu().numpy()
    flat = P.final_flat(len(cf))
    if flat is None:
        return None
    gb = [gt_box_img[0] + pad, gt_box_img[1] + pad, gt_box_img[2] + pad, gt_box_img[3] + pad]
    gk = gt_kp_img + pad
    ious = np.array([TC.iou_xyxy(b, gb) for b in bx])
    A = P.assign(P.gt_to_input(gb, padded.shape))
    ok = np.where(ious >= IOU_T)[0]
    bad = np.where(ious < IOU_T)[0]
    out = {}
    for tag, sel in (("POS", ok), ("NEG", bad)):
        if not len(sel):
            continue
        i = sel[int(np.argmax(cf[sel]))]
        f = int(flat[i])
        lv, li = P.level_of(f)
        d = np.linalg.norm(kp[i][:len(gk)] - gk[:len(kp[i])], axis=1)
        rec = {"image_id": image_id, "dataset": dset, "domain": domain,
               "role": ("Splus" if (dset == "SYNTH" and tag == "POS") else
                        "SW" if dset == "SYNTH" else
                        "Rplus" if tag == "POS" else "RW"),
               "final_rank": int(i), "class_score": float(cf[i]),
               "class_logit": float(A["logits"][f]),
               "level": lv, "flat": f,
               "iou_to_gt": float(ious[i]),
               "fg_mask": int(A["fg_mask"][f]),
               "target_score": float(A["target_scores"][f]),
               "align_metric": float(A["align_metric"][f]),
               "tal_overlap_ciou": float(A["overlaps"][f]),
               "kp_error_px": float(np.median(d)),
               "kp8_error_px": float(np.median(d[:8])),
               "kp_p90_px": float(np.percentile(d, 90))}
        out[tag] = rec
        ROWS.append(rec)
    # assigned anchor 자체도 기록 (fg 인 anchor 중 target 최대)
    fg = np.where(A["fg_mask"] > 0)[0]
    if len(fg):
        j = int(fg[int(np.argmax(A["target_scores"][fg]))])
        lv, _ = P.level_of(j)
        ROWS.append({"image_id": image_id, "dataset": dset, "domain": domain,
                     "role": "ASSIGNED", "final_rank": -1,
                     "class_score": float(1/(1+np.exp(-A["logits"][j]))),
                     "class_logit": float(A["logits"][j]), "level": lv, "flat": j,
                     "iou_to_gt": None, "fg_mask": 1,
                     "target_score": float(A["target_scores"][j]),
                     "align_metric": float(A["align_metric"][j]),
                     "tal_overlap_ciou": float(A["overlaps"][j]),
                     "kp_error_px": None, "kp8_error_px": None, "kp_p90_px": None,
                     "assigned_is_top1_candidate": int(j == int(flat[0]))})
    out["n_fg"] = int((A["fg_mask"] > 0).sum())
    return out


# ---------------------------------------------------------------- M0 PARITY
LEAK = set(json.load(open(f"{QY}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
items = [it for it in json.load(open(MANI))["items"]
         if it["frame_id"] not in LEAK and os.path.exists(os.path.join(ROOT, it["image"]))]
key = lambda it: hashlib.sha1(f"TA_PARITY_V1|{it['frame_id']}".encode()).hexdigest()
sub = sorted(items, key=key)[:64]
A_, PROV = [], []
for it in sub:
    im = cv2.imread(os.path.join(ROOT, it["image"]))
    r, _ = P.predict(im)
    cf = r.boxes.conf.cpu().numpy() if (r.boxes is not None and len(r.boxes)) else np.array([])
    A_.append({"f": it["frame_id"], "n": int(cf.size), "conf": cf.tolist(),
               "box": (r.boxes.xyxy.cpu().numpy().tolist() if cf.size else []),
               "kps": (r.keypoints.xy.cpu().numpy().tolist() if cf.size else [])})
    if cf.size:
        fl = P.final_flat(len(cf))
        if fl is not None:
            lg = P.cap["preds"]["one2one"]["scores"][0, 0].cpu().numpy()
            PROV.append(float(abs(1/(1+np.exp(-lg[int(fl[0])])) - cf[0])))
json.dump(sub, open("/tmp/_ta_items.json", "w"))
open("/tmp/_ta_stock.py", "w").write(f'''
import json, os, sys
import numpy as np, cv2
from ultralytics import YOLO
items = json.load(open("/tmp/_ta_items.json"))
m = YOLO("{W}", task="pose")
out = []
for it in items:
    im = cv2.imread(os.path.join("{ROOT}", it["image"]))
    p = cv2.copyMakeBorder(im, {TC.PAD}, {TC.PAD}, {TC.PAD}, {TC.PAD}, cv2.BORDER_REFLECT_101)
    r = m.predict(p, conf={TC.CONF}, imgsz={TC.IMGSZ}, iou={TC.IOU_NMS},
                  max_det={TC.MAX_DET}, device=0, verbose=False)[0]
    cf = r.boxes.conf.cpu().numpy() if (r.boxes is not None and len(r.boxes)) else np.array([])
    out.append({{"f": it["frame_id"], "n": int(cf.size), "conf": cf.tolist(),
                 "box": (r.boxes.xyxy.cpu().numpy().tolist() if cf.size else []),
                 "kps": (r.keypoints.xy.cpu().numpy().tolist() if cf.size else [])}})
json.dump(out, open("/tmp/_ta_stock.json", "w"))
''')
rr = subprocess.run([sys.executable, "-u", "/tmp/_ta_stock.py"], capture_output=True, text=True)
if not os.path.exists("/tmp/_ta_stock.json"):
    raise SystemExit(f"stock 실패: {(rr.stderr or rr.stdout)[-1000:]}")
B_ = {x["f"]: x for x in json.load(open("/tmp/_ta_stock.json"))}
dmax = {"n": 0, "conf": 0.0, "box": 0.0, "kps": 0.0}
for a in A_:
    b = B_[a["f"]]
    dmax["n"] = max(dmax["n"], abs(a["n"] - b["n"]))
    if a["n"] and a["n"] == b["n"]:
        for k in ("conf", "box", "kps"):
            dmax[k] = max(dmax[k], float(np.abs(np.array(a[k]) - np.array(b[k])).max()))
PAR = {"n_frames": len(sub), "max_abs_diff": dmax,
       "PASS": bool(dmax["n"] == 0 and dmax["conf"] == 0.0 and dmax["box"] == 0.0
                    and dmax["kps"] == 0.0),
       "provenance_sigmoid_logit_vs_conf_max": (max(PROV) if PROV else None),
       "tolerance": "exact 0 우선, provenance 는 <=1e-7"}
json.dump(PAR, open(f"{NS}/M0_PARITY.json", "w"), indent=2, ensure_ascii=False)
log(f"M0 PARITY {PAR['PASS']}  {dmax}  provenance {PAR['provenance_sigmoid_logit_vs_conf_max']:.2e}")
if not PAR["PASS"]:
    raise SystemExit("M0_PARITY_FAIL")

# ---------------------------------------------------------------- REAL
log("REAL 128")
for it in items:
    im = cv2.imread(os.path.join(ROOT, it["image"]))
    gb, gk = gt_real(os.path.join(ROOT, it["label"]))
    process(it["frame_id"], im, gb, gk,
            "NIGHT" if it.get("set") in NIGHT else "DAY", "REAL", TC.PAD)
log(f"  rows {len(ROWS)}")

# ---------------------------------------------------------------- SYNTH
log("SYNTH G38 val 1,998")
vf = sorted(os.listdir(f"{DS}/images/val"))
for n_, f in enumerate(vf):
    im = cv2.imread(f"{DS}/images/val/{f}")
    lp = f"{DS}/labels/val/{os.path.splitext(f)[0]}.txt"
    if im is None or not os.path.exists(lp):
        continue
    h, w = im.shape[:2]
    gb, gk = gt_synth(lp, w, h)
    process(os.path.splitext(f)[0], im, gb, gk, "SYNTH", "SYNTH", 0)
    if (n_ + 1) % 400 == 0:
        log(f"  synth {n_+1}/{len(vf)}")

keys = ["image_id", "dataset", "domain", "role", "final_rank", "class_score", "class_logit",
        "level", "flat", "iou_to_gt", "fg_mask", "target_score", "align_metric",
        "tal_overlap_ciou", "kp_error_px", "kp8_error_px", "kp_p90_px",
        "assigned_is_top1_candidate"]
with open(f"{NS}/TAL_PER_CANDIDATE.csv", "w", newline="") as fh:
    w_ = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
    w_.writeheader()
    for r in ROWS:
        w_.writerow(r)
log(f"저장 완료 rows={len(ROWS)}")
