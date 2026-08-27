"""M0 PARITY + FEATURE_TAP_AUDIT + provenance 정합성 검사."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys

import numpy as np, cv2, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ra_core as RC                                                # noqa: E402

ROOT, NS = RC.ROOT, RC.NS
Y = RC.Y
QY = f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/"
        "REAL_GT_QA_20260821T133405Z/REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
LEAK = set(json.load(open(f"{QY}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
items = [it for it in json.load(open(MANI))["items"]
         if it["frame_id"] not in LEAK and os.path.exists(os.path.join(ROOT, it["image"]))]
key = lambda it: hashlib.sha1(f"RA_PARITY_V1|{it['frame_id']}".encode()).hexdigest()
sub = sorted(items, key=key)[:64]

# ---------------- instrumented ON ----------------
on = RC.Instrumented(hooks=True)
A, TAP, PROV = [], None, {"logit_matches_conf": [], "n": 0, "flat_ok": 0}
for it in sub:
    im = cv2.imread(os.path.join(ROOT, it["image"]))
    r = on.predict(im)
    cf = r.boxes.conf.cpu().numpy() if (r.boxes is not None and len(r.boxes)) else np.array([])
    A.append({"f": it["frame_id"], "n": int(cf.size),
              "conf": cf.tolist(),
              "box": (r.boxes.xyxy.cpu().numpy().tolist() if cf.size else []),
              "kps": (r.keypoints.xy.cpu().numpy().tolist() if cf.size else [])})
    if TAP is None:
        sizes = on.level_sizes()
        TAP = {"input": [1, 3, im.shape[0] + 200, im.shape[1] + 200], "imgsz": RC.IMGSZ,
               "levels": {}}
        for i in range(on.nl):
            lv = RC.LEVEL_NAME[i]
            TAP["levels"][lv] = {
                "level_index": i, "stride": float(on.head.stride[i]),
                "spatial": list(sizes[i]),
                "modules": {
                    "neck_in": f"model.model[-1].one2one_cv3[{i}][0]  (입력)",
                    "cls1": f"model.model[-1].one2one_cv3[{i}][0]",
                    "cls_pen": f"model.model[-1].one2one_cv3[{i}][1]",
                    "logit": f"model.model[-1].one2one_cv3[{i}][2]",
                    "pose_pen": f"model.model[-1].one2one_cv4[{i}]"},
                "channels": {k: (None if on.cap.get(f"{lv}_{k}") is None
                                 else int(on.cap[f"{lv}_{k}"].shape[1]))
                             for k in ("cls1", "cls_pen", "logit", "pose_pen")},
                "neck_in_channels": int(on.cap[f"{lv}_neck_in"].shape[1]),
                "branch": "one2one"}
        TAP["flat_index_order"] = "level 0(P3) -> 1(P4) -> 2(P5), 각 H*W 를 이어붙임 (Detect.forward_head)"
        TAP["primary_path"] = "one2one classification path (실제 추론에 쓰이는 경로)"
    # provenance 정합성 — 매핑된 cell 의 logit sigmoid 가 final conf 와 같은가
    if cf.size:
        flat = RC.map_final_to_flat(on, cf)
        if flat is not None and len(flat) == cf.size:
            PROV["flat_ok"] += 1
            v = on.vectors(int(flat[0]))
            PROV["logit_matches_conf"].append(
                float(abs(1 / (1 + np.exp(-v["logit"][0])) - cf[0])))
        PROV["n"] += 1
on.close()
del on
torch.cuda.empty_cache()

# ---------------- instrumented OFF (stock) ----------------
code = f'''
import json, os, sys
import numpy as np, cv2
sys.path.insert(0, "{NS}")
from ultralytics import YOLO
items = json.load(open("/tmp/_ra_items.json"))
m = YOLO("{RC.W}", task="pose")
out = []
for it in items:
    im = cv2.imread(os.path.join("{ROOT}", it["image"]))
    p = cv2.copyMakeBorder(im, {RC.PAD}, {RC.PAD}, {RC.PAD}, {RC.PAD}, cv2.BORDER_REFLECT_101)
    r = m.predict(p, conf={RC.CONF}, imgsz={RC.IMGSZ}, iou={RC.IOU_NMS},
                  max_det={RC.MAX_DET}, device=0, verbose=False)[0]
    cf = r.boxes.conf.cpu().numpy() if (r.boxes is not None and len(r.boxes)) else np.array([])
    out.append({{"f": it["frame_id"], "n": int(cf.size), "conf": cf.tolist(),
                 "box": (r.boxes.xyxy.cpu().numpy().tolist() if cf.size else []),
                 "kps": (r.keypoints.xy.cpu().numpy().tolist() if cf.size else [])}})
json.dump(out, open("/tmp/_ra_stock.json", "w"))
'''
json.dump(sub, open("/tmp/_ra_items.json", "w"))
open("/tmp/_ra_stock.py", "w").write(code)
r = subprocess.run([sys.executable, "-u", "/tmp/_ra_stock.py"], capture_output=True, text=True)
if not os.path.exists("/tmp/_ra_stock.json"):
    raise SystemExit(f"stock run 실패: {(r.stderr or r.stdout)[-1200:]}")
B = {x["f"]: x for x in json.load(open("/tmp/_ra_stock.json"))}

d = {"n": 0, "conf": 0.0, "box": 0.0, "kps": 0.0}
for a in A:
    b = B[a["f"]]
    d["n"] = max(d["n"], abs(a["n"] - b["n"]))
    if a["n"] and b["n"] and a["n"] == b["n"]:
        for k in ("conf", "box", "kps"):
            d[k] = max(d[k], float(np.abs(np.array(a[k]) - np.array(b[k])).max()))

lm = PROV["logit_matches_conf"]
out = {"n_frames": len(sub), "selection": "sha1('RA_PARITY_V1|frame') 앞 64 (random 미사용)",
       "max_abs_diff": d, "tolerance": 0.0,
       "PARITY_PASS": bool(d["n"] == 0 and d["conf"] == 0.0 and d["box"] == 0.0 and d["kps"] == 0.0),
       "provenance_check": {
           "frames_with_candidates": PROV["n"], "flat_mapped": PROV["flat_ok"],
           "max_abs_diff_sigmoid_logit_vs_conf": (max(lm) if lm else None),
           "note": ("매핑된 source cell 의 logit 에 sigmoid 를 씌운 값이 final confidence 와 "
                    "같아야 한다 — provenance 매핑이 맞다는 독립 증거.")},
       "fuse_disabled": True,
       "why_exact_zero_possible": ("hook 은 값을 읽기만 하고 get_topk_index 래퍼는 원본 반환값을 "
                                   "그대로 통과시킨다. 연산 그래프가 바뀌지 않으므로 exact 0 이어야 한다.")}
out["PROVENANCE_PASS"] = bool(out["provenance_check"]["max_abs_diff_sigmoid_logit_vs_conf"] is not None
                              and out["provenance_check"]["max_abs_diff_sigmoid_logit_vs_conf"] <= 1e-5
                              and PROV["flat_ok"] == PROV["n"])
json.dump(out, open(f"{NS}/M0_PARITY.json", "w"), indent=2, ensure_ascii=False)
json.dump(TAP, open(f"{NS}/FEATURE_TAP_AUDIT.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({"max_abs_diff": d, "PARITY_PASS": out["PARITY_PASS"],
                  "provenance": out["provenance_check"],
                  "PROVENANCE_PASS": out["PROVENANCE_PASS"]}, indent=2, ensure_ascii=False))
