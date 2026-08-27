"""D1 + D2 — conf sweep 과 inference resolution screen (pallet-yolo26 env).

학습 0.  같은 checkpoint 로 추론만 다시 한다.

★ threshold 를 고르지 않는다.  positive DEV 뿐이라 낮은 conf 가 recall 을 얼마나
회수하는지만 본다.  real negative 없이 conf 를 확정하면 FP 를 못 보고 정하는 것이다.
"""
from __future__ import annotations

import json, os, sys
import cv2, numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts/stage0/model_compare"))
import mc_frames as MF  # noqa: E402

OUT = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis_pre_v2")
CK = os.path.join(ROOT, "challenge/yolo_pose_one_model/runs_paper/"
                        "yolo26n_paper_generic_v1_seed42/weights/last.pt")
PAD = 100
CONFS = [0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40]
SIZES = [640, 960, 1280]
TOPK = 5


def main():
    from ultralytics import YOLO
    model = YOLO(CK, task="pose")
    frames = MF.frames()
    out = {"note": "학습 0. 같은 checkpoint 로 추론만. threshold 선택 금지 "
                   "(positive DEV 뿐이라 FP 를 못 본다).",
           "pad": PAD, "checkpoint": os.path.relpath(CK, ROOT),
           "confs": CONFS, "sizes": SIZES, "frames": {}}
    for i, (key, sealed, jp, ip, label) in enumerate(frames, 1):
        fid = os.path.splitext(os.path.basename(jp))[0]
        image = cv2.imread(ip)
        padded = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD,
                                    cv2.BORDER_REFLECT_101)
        gt8 = np.asarray(label["objects"][0]["projected_cuboid"], float)[:8]
        gtb = [gt8[:, 0].min(), gt8[:, 1].min(), gt8[:, 0].max(), gt8[:, 1].max()]
        entry = {"set": key, "sealed": sealed, "gt_bbox": gtb, "runs": {}}
        for size in SIZES:
            r = model.predict(padded, imgsz=size, conf=min(CONFS),
                              verbose=False)[0]
            cands = []
            if r.boxes is not None and len(r.boxes):
                bc = r.boxes.conf.cpu().numpy()
                xy = r.boxes.xyxy.cpu().numpy() - PAD
                kps = r.keypoints.xy.cpu().numpy() - PAD
                kpc = (r.keypoints.conf.cpu().numpy()
                       if r.keypoints.conf is not None else None)
                order = np.argsort(-bc)[:TOPK]
                for j in order:
                    bx = xy[j]
                    ix1, iy1 = max(bx[0], gtb[0]), max(bx[1], gtb[1])
                    ix2, iy2 = min(bx[2], gtb[2]), min(bx[3], gtb[3])
                    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
                    ua = ((bx[2]-bx[0])*(bx[3]-bx[1])
                          + (gtb[2]-gtb[0])*(gtb[3]-gtb[1]) - inter)
                    cands.append({"box_conf": float(bc[j]),
                                  "bbox": bx.tolist(),
                                  "iou_with_gt": float(inter/ua) if ua > 0 else 0.0,
                                  "kp_conf_mean": float(np.mean(kpc[j]))
                                  if kpc is not None else None,
                                  "kps": kps[j][:8].tolist()})
            entry["runs"][str(size)] = {
                "n_candidates_at_min_conf": len(cands),
                "candidates": cands,
                "n_above": {str(c): int(sum(1 for x in cands
                                            if x["box_conf"] >= c))
                            for c in CONFS}}
        out["frames"][fid] = entry
        if i % 40 == 0:
            print(f"    {i}/{len(frames)}", flush=True)
    json.dump(out, open(os.path.join(OUT, "_d1d2_raw.json"), "w"), indent=1)
    print(f"  {len(out['frames'])} frames -> _d1d2_raw.json")


if __name__ == "__main__":
    main()
