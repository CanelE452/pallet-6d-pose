"""S+ / R+ / RW / RW_RANKFAIL / RN 후보 추출 + one2one cls path tap vector 저장."""
from __future__ import annotations
import csv, hashlib, json, os, sys, time

import numpy as np, cv2, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ra_core as RC                                                # noqa: E402

ROOT, NS, Y = RC.ROOT, RC.NS, RC.Y
QY = f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
DS = f"{Y}/datasets/g38_generic_only"
NEG = f"{ROOT}/data/pallet/raw_data/negative_real_20260823/rgb"
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/"
        "REAL_GT_QA_20260821T133405Z/REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
NIGHT_SETS = {"eval_night08", "eval_night09"}
IOU_T = 0.5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gt_real(jp):
    o = json.load(open(jp))["objects"][0]
    g = np.array(o["projected_cuboid"], float)[:8]
    c = np.array(o["projected_cuboid_centroid"], float)
    return ([g[:, 0].min(), g[:, 1].min(), g[:, 0].max(), g[:, 1].max()],
            np.vstack([g, c]))


def gt_synth(lp, w, h):
    v = [float(x) for x in open(lp).read().strip().split("\n")[0].split()]
    cx, cy, bw, bh = v[1:5]
    box = [(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h]
    k = np.array(v[5:]).reshape(-1, 3)
    return box, np.stack([k[:, 0]*w, k[:, 1]*h], 1)


inst = RC.Instrumented(hooks=True)
ROWS, VEC = [], {}


def store(rec, flat):
    v = inst.vectors(int(flat))
    rec.update({"level": v["level"], "src_y": v["y"], "src_x": v["x"], "flat": int(flat)})
    kbase = (rec["group"], v["level"])
    VEC.setdefault(kbase, {t: [] for t in ("neck_in", "cls1", "cls_pen", "logit", "pose_pen")})
    for t in ("neck_in", "cls1", "cls_pen", "logit", "pose_pen"):
        VEC[kbase][t].append(v[t].astype(np.float32))
    VEC[kbase].setdefault("row_id", []).append(len(ROWS))
    ROWS.append(rec)


def candidates(r, pad):
    if r.boxes is None or not len(r.boxes):
        return None
    cf = r.boxes.conf.cpu().numpy()
    bx = r.boxes.xyxy.cpu().numpy() - pad
    kp = r.keypoints.xy.cpu().numpy() - pad
    flat = RC.map_final_to_flat(inst, cf)
    return cf, bx, kp, flat


# ---------------------------------------------------------------- S+
val_files = sorted(os.listdir(f"{DS}/images/val"))
miss_s = 0
for n_, f in enumerate(val_files):
    im = cv2.imread(f"{DS}/images/val/{f}")
    lp = f"{DS}/labels/val/{os.path.splitext(f)[0]}.txt"
    if im is None or not os.path.exists(lp):
        miss_s += 1; continue
    h, w = im.shape[:2]
    gb, gk = gt_synth(lp, w, h)
    r = inst.predict(im, pad=0)                      # synthetic val 은 이미 PAD100 캔버스
    c = candidates(r, 0)
    if c is None:
        miss_s += 1; continue
    cf, bx, kp, flat = c
    ious = np.array([RC.iou_xyxy(b, gb) for b in bx])
    ok = np.where(ious >= IOU_T)[0]
    if not len(ok) or flat is None:
        miss_s += 1; continue
    i = ok[int(np.argmax(cf[ok]))]
    d = np.linalg.norm(kp[i][:len(gk)] - gk[:len(kp[i])], axis=1)
    store({"image_id": os.path.splitext(f)[0], "group": "S+", "domain": "SYNTH",
           "final_rank": int(i), "class_score": float(cf[i]),
           "box": bx[i].tolist(), "iou": float(ious[i]),
           "kp_median": float(np.median(d)), "kp_p90": float(np.percentile(d, 90)),
           "kp8_median": float(np.median(d[:8]))}, flat[i])
    if (n_ + 1) % 500 == 0:
        log(f"  S+ {n_+1}/{len(val_files)}")
log(f"S+ 완료  n={sum(1 for r in ROWS if r['group']=='S+')}  missing={miss_s}")

# ---------------------------------------------------------------- R+/RW
LEAK = set(json.load(open(f"{QY}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
items = [it for it in json.load(open(MANI))["items"] if it["frame_id"] not in LEAK]
miss_r = {"no_cand": 0, "no_correct": 0, "no_wrong": 0}
for it in items:
    ip, jp = os.path.join(ROOT, it["image"]), os.path.join(ROOT, it["label"])
    if not (os.path.exists(ip) and os.path.exists(jp)):
        continue
    im = cv2.imread(ip)
    gb, gk = gt_real(jp)
    dom = "NIGHT" if it.get("set") in NIGHT_SETS else "DAY"
    r = inst.predict(im, pad=RC.PAD)
    c = candidates(r, RC.PAD)
    if c is None:
        miss_r["no_cand"] += 1; continue
    cf, bx, kp, flat = c
    if flat is None:
        continue
    ious = np.array([RC.iou_xyxy(b, gb) for b in bx])
    ok = np.where(ious >= IOU_T)[0]
    bad = np.where(ious < IOU_T)[0]
    base = {"image_id": it["frame_id"], "domain": dom, "set": it.get("set")}
    if len(ok):
        i = ok[int(np.argmax(cf[ok]))]
        d = np.linalg.norm(kp[i][:len(gk)] - gk[:len(kp[i])], axis=1)
        store({**base, "group": "R+", "final_rank": int(i), "class_score": float(cf[i]),
               "box": bx[i].tolist(), "iou": float(ious[i]),
               "kp_median": float(np.median(d)), "kp_p90": float(np.percentile(d, 90)),
               "kp8_median": float(np.median(d[:8]))}, flat[i])
    else:
        miss_r["no_correct"] += 1
    if len(bad):
        j = bad[int(np.argmax(cf[bad]))]
        store({**base, "group": "RW", "final_rank": int(j), "class_score": float(cf[j]),
               "box": bx[j].tolist(), "iou": float(ious[j])}, flat[j])
    else:
        miss_r["no_wrong"] += 1
    # RW_RANKFAIL — correct 는 있는데 top1 이 wrong
    if len(ok) and ious[0] < IOU_T:
        store({**base, "group": "RW_RANKFAIL", "final_rank": 0,
               "class_score": float(cf[0]), "box": bx[0].tolist(),
               "iou": float(ious[0])}, flat[0])
log(f"R+ {sum(1 for r in ROWS if r['group']=='R+')}  RW {sum(1 for r in ROWS if r['group']=='RW')}  "
    f"RANKFAIL {sum(1 for r in ROWS if r['group']=='RW_RANKFAIL')}  miss {miss_r}")

# ---------------------------------------------------------------- RN
negf = sorted(os.listdir(NEG))
no_cand = 0
for n_, f in enumerate(negf):
    im = cv2.imread(f"{NEG}/{f}")
    if im is None:
        continue
    r = inst.predict(im, pad=RC.PAD)
    c = candidates(r, RC.PAD)
    if c is None:
        no_cand += 1; continue
    cf, bx, kp, flat = c
    if flat is None:
        continue
    i = int(np.argmax(cf))
    store({"image_id": os.path.splitext(f)[0], "group": "RN", "domain": "NEG",
           "final_rank": i, "class_score": float(cf[i]), "box": bx[i].tolist(),
           "iou": None}, flat[i])
    if (n_ + 1) % 500 == 0:
        log(f"  RN {n_+1}/{len(negf)}")
log(f"RN {sum(1 for r in ROWS if r['group']=='RN')}  NO_CANDIDATE {no_cand}")
inst.close()

# ---------------------------------------------------------------- 저장
with open(f"{NS}/CANDIDATE_PROVENANCE.csv", "w", newline="") as fh:
    keys = ["image_id", "group", "domain", "set", "final_rank", "class_score", "iou",
            "level", "src_y", "src_x", "flat", "box", "kp_median", "kp_p90", "kp8_median"]
    w_ = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
    w_.writeheader()
    for r in ROWS:
        w_.writerow({**r, "box": json.dumps(r.get("box"))})

flat_npz = {}
for (g, lv), d in VEC.items():
    for t, arr in d.items():
        flat_npz[f"{g}|{lv}|{t}"] = np.asarray(arr)
np.savez_compressed(f"{NS}/FEATURE_VECTORS.npz", **flat_npz)

cnt = {"S+": 0, "R+": 0, "RW": 0, "RW_RANKFAIL": 0, "RN": 0}
lvl = {g: {"P3": 0, "P4": 0, "P5": 0} for g in cnt}
for r in ROWS:
    cnt[r["group"]] += 1
    lvl[r["group"]][r["level"]] += 1
CC = {"counts": cnt, "source_level_distribution": lvl,
      "source_level_fraction": {g: {k: (v / max(sum(d.values()), 1)) for k, v in d.items()}
                                for g, d in lvl.items()},
      "missing": {"S+": miss_s, "real": miss_r, "RN_NO_CANDIDATE": no_cand},
      "dataset_counts": {"synth_val": len(val_files), "real": len(items), "neg": len(negf)},
      "definitions": {
          "S+": "G38 val 프레임에서 IoU>=0.5 후보 중 conf 최고",
          "R+": "real128 프레임에서 IoU>=0.5 후보 중 conf 최고",
          "RW": "같은 real 프레임에서 IoU<0.5 후보 중 conf 최고 (가장 강한 distractor)",
          "RW_RANKFAIL": "correct 가 존재하는데 top1 이 wrong 인 프레임의 그 top1",
          "RN": "real negative 프레임의 conf 최고 후보 (threshold 올리지 않음, conf=0.001)"},
      "conf_threshold": RC.CONF}
json.dump(CC, open(f"{NS}/CANDIDATE_COUNTS.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({"counts": cnt, "level_frac": CC["source_level_fraction"],
                  "missing": CC["missing"]}, indent=2, ensure_ascii=False))
