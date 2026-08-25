"""단일 checkpoint 의 POS(128) / NEG(2,689) 점수 덤프 — 검증된 real recipe 그대로."""
import argparse, json, os, sys
import numpy as np, cv2, torch
from ultralytics import YOLO

R = "/home/minjae/Documents/github/pallet-pose"
Y = f"{R}/challenge/yolo_pose_one_model"
Q = f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
NEG = f"{R}/data/pallet/raw_data/negative_real_20260823/rgb"
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/REAL_GT_QA_20260821T133405Z/"
        "REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
PAD, CONF = 100, 0.001
NIGHT = {"eval_night08", "eval_night09"}
LEAK = set(json.load(open(f"{Q}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])

ap = argparse.ArgumentParser(); ap.add_argument("--weights"); ap.add_argument("--tag")
A = ap.parse_args()
out = f"{Q}/NEGSCORE_{A.tag}.json"
if os.path.exists(out):
    print("cached", out); sys.exit(0)

POS = []
for it in json.load(open(MANI))["items"]:
    if it["frame_id"] in LEAK: continue
    ip = os.path.join(R, it["image"])
    if os.path.exists(ip):
        POS.append({"frame": it["frame_id"], "img": ip,
                    "domain": "NIGHT" if it.get("set", "?") in NIGHT else "DAY"})
NEGF = sorted(os.listdir(NEG))
m = YOLO(A.weights, task="pose")
rows = []
for lab, items in ((1, POS), (0, [{"frame": f[:-4], "img": f"{NEG}/{f}", "domain": "NEG"}
                                  for f in NEGF])):
    for r in items:
        im = cv2.imread(r["img"])
        if im is None: continue
        p = cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        res = m.predict(p, conf=CONF, imgsz=640, device=0, verbose=False)[0]
        cf = res.boxes.conf.cpu().numpy() if (res.boxes is not None and len(res.boxes)) else np.array([])
        rows.append({"frame": r["frame"], "domain": r["domain"], "label": lab,
                     "max_conf": float(cf.max()) if cf.size else 0.0,
                     "n_candidates": int(cf.size), "confs": [float(x) for x in cf]})
del m; torch.cuda.empty_cache()
json.dump({"weights": A.weights, "tag": A.tag, "n_pos": sum(r["label"] for r in rows),
           "n_neg": sum(1 for r in rows if r["label"] == 0), "rows": rows},
          open(out, "w"))
print(f"{A.tag}  pos {sum(r['label'] for r in rows)}  neg {sum(1 for r in rows if r['label']==0)}")
