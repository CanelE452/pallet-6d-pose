"""NIGHT 28 프레임별 진단 — night_cand_one.py 의 정의를 그대로 쓴다 (PAD/IOU_T/CONF 동일).

기존 파일은 건드리지 않고, per-frame 세부만 추가로 남긴다. diagnostic only.
"""
import argparse, json, os, sys
import numpy as np, cv2

ROOT = "/home/minjae/Documents/github/pallet-pose"; sys.path.insert(0, ROOT)
NS = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True)
ap.add_argument("--tag", required=True); A = ap.parse_args()
from ultralytics import YOLO
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/REAL_GT_QA_20260821T133405Z/"
        "REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
PAD, IOU_T, CONF = 100, 0.5, 0.001
NIGHT = {"eval_night08", "eval_night09"}
man = json.load(open(MANI))
night = [(it["frame_id"], os.path.join(ROOT, it["label"]), os.path.join(ROOT, it["image"]))
         for it in man["items"] if it.get("set") in NIGHT]


def gtinfo(jp):
    o = json.load(open(jp))["objects"][0]
    g = np.array(o["projected_cuboid"], dtype=float)[:8]
    c = np.array(o["projected_cuboid_centroid"], dtype=float)
    return [g[:, 0].min(), g[:, 1].min(), g[:, 0].max(), g[:, 1].max()], np.vstack([g, c])


def iou(b, g):
    xx = max(0, min(b[2], g[2]) - max(b[0], g[0])); yy = max(0, min(b[3], g[3]) - max(b[1], g[1]))
    i = xx * yy
    return i / max((b[2]-b[0])*(b[3]-b[1]) + (g[2]-g[0])*(g[3]-g[1]) - i, 1e-9)


m = YOLO(A.weights, task="pose"); rows = []
for fid, jp, ip in night:
    im = cv2.imread(ip)
    p = cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    r = m.predict(p, conf=CONF, imgsz=640, device=0, verbose=False)[0]
    gb, gk9 = gtinfo(jp)
    d = {"frame": fid, "n": 0, "top1": False, "any": False, "top1_conf": None,
         "top1_iou": None, "correct_conf": None, "wrong_conf": None, "kp_median": None}
    if r.boxes is not None and len(r.boxes):
        cf = r.boxes.conf.cpu().numpy(); bx = r.boxes.xyxy.cpu().numpy() - PAD
        kp = r.keypoints.xy.cpu().numpy() - PAD
        o = np.argsort(-cf); iv = np.array([iou(bx[i], gb) for i in o]); ok = iv >= IOU_T
        d.update({"n": int(cf.size), "top1": bool(ok[0]), "any": bool(ok.any()),
                  "top1_conf": float(cf[o][0]), "top1_iou": float(iv[0]),
                  "correct_conf": float(cf[o][ok].max()) if ok.any() else None,
                  "wrong_conf": float(cf[o][~ok].max()) if (~ok).any() else None,
                  "rank": (int(np.argmax(ok)) + 1) if ok.any() else None})
        k = kp[o][0]
        n = min(len(k), len(gk9))
        d["kp_median"] = float(np.median(np.linalg.norm(k[:n] - gk9[:n], axis=1)))
    rows.append(d)
json.dump({"tag": A.tag, "weights": A.weights, "n": len(rows), "rows": rows},
          open(f"{NS}/results/NIGHT_FW_{A.tag}.json", "w"), indent=2)
print(f"{A.tag}: top1 {sum(r['top1'] for r in rows)}/{len(rows)} any {sum(r['any'] for r in rows)}/{len(rows)}")
