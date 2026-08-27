"""M0 PARITY (HARD BLOCK) — stock 경로 vs patched M0 가 같은가.

raw tensor 수준(top1 box/conf/9kp)과 evaluator 수준 둘 다 본다.
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
Y = f"{ROOT}/challenge/yolo_pose_one_model"
NS = f"{Y}/p26_inference_path_audit"
QY = f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/"
        "REAL_GT_QA_20260821T133405Z/REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
W = os.path.join(ROOT, json.load(open(f"{Y}/runs_arch_baseline/RESULT_Y0.json"))["weights"])
PAD, CONF = 100, 0.001
N_PARITY = 64
sys.path.insert(0, NS)


def frames():
    leak = set(json.load(open(f"{QY}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
    items = [it for it in json.load(open(MANI))["items"] if it["frame_id"] not in leak]
    items = [it for it in items if os.path.exists(os.path.join(ROOT, it["image"]))]
    key = lambda it: hashlib.sha1(f"P26_PARITY_V1|{it['frame_id']}".encode()).hexdigest()
    return sorted(items, key=key)[:N_PARITY]      # 결정론적, random 없음


def run(mode):
    """mode=None -> stock (patch 없음)."""
    code = f'''
import json, os, sys
import numpy as np, cv2, torch
sys.path.insert(0, "{NS}")
MODE = {repr(mode)}
if MODE is not None:
    import p26_paths; p26_paths.install(MODE)
from ultralytics import YOLO
items = json.load(open("/tmp/_p26_parity_items.json"))
m = YOLO("{W}", task="pose")
out = []
for it in items:
    im = cv2.imread(os.path.join("{ROOT}", it["image"]))
    p = cv2.copyMakeBorder(im, {PAD}, {PAD}, {PAD}, {PAD}, cv2.BORDER_REFLECT_101)
    r = m.predict(p, conf={CONF}, imgsz=640, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        out.append({{"frame": it["frame_id"], "n": 0}}); continue
    cf = r.boxes.conf.cpu().numpy()
    i = int(np.argmax(cf))
    out.append({{"frame": it["frame_id"], "n": int(cf.size),
                 "conf": float(cf[i]),
                 "box": r.boxes.xyxy.cpu().numpy()[i].tolist(),
                 "kps": r.keypoints.xy.cpu().numpy()[i].tolist()}})
json.dump(out, open("/tmp/_p26_parity_{mode or 'stock'}.json", "w"))
'''
    sc = f"/tmp/_p26_parity_run_{mode or 'stock'}.py"
    open(sc, "w").write(code)
    r = subprocess.run([sys.executable, "-u", sc], capture_output=True, text=True)
    p = f"/tmp/_p26_parity_{mode or 'stock'}.json"
    if not os.path.exists(p):
        raise SystemExit(f"parity run 실패 ({mode}): {(r.stderr or r.stdout)[-1200:]}")
    return json.load(open(p))


items = frames()
json.dump(items, open("/tmp/_p26_parity_items.json", "w"))
print(f"parity frames {len(items)} (deterministic sha1 order)", flush=True)
A = {r["frame"]: r for r in run(None)}
B = {r["frame"]: r for r in run("E2E")}

diffs = {"conf": [], "box": [], "kps": [], "n": []}
missing = []
for f in A:
    a, b = A[f], B.get(f)
    if b is None:
        missing.append(f); continue
    diffs["n"].append(abs(a["n"] - b["n"]))
    if a["n"] == 0 or b["n"] == 0:
        continue
    diffs["conf"].append(abs(a["conf"] - b["conf"]))
    diffs["box"].append(float(np.abs(np.array(a["box"]) - np.array(b["box"])).max()))
    diffs["kps"].append(float(np.abs(np.array(a["kps"]) - np.array(b["kps"])).max()))
mx = {k: (max(v) if v else 0.0) for k, v in diffs.items()}
raw_pass = bool(mx["conf"] <= 1e-6 and mx["box"] <= 1e-6 and mx["kps"] <= 1e-6
                and mx["n"] == 0 and not missing)

out = {"n_frames": len(items), "missing": missing,
       "max_abs_diff": mx, "tolerance": 1e-6, "RAW_PARITY_PASS": raw_pass,
       "note": ("patch 는 fuse 를 no-op 으로 만들어 one2many head 를 남긴다. "
                "one2one 경로는 그 모듈을 읽지 않으므로 수치가 같아야 한다 — 그 주장을 "
                "여기서 검증한다."),
       "selection": "sha1('P26_PARITY_V1|frame') 오름차순 앞 64 개 (random 미사용)"}
json.dump(out, open(f"{NS}/tests/P26_M0_PARITY_RAW.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({"max_abs_diff": mx, "RAW_PARITY_PASS": raw_pass}, indent=2))
