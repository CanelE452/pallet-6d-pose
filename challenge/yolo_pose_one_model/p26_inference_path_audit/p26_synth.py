"""G38 val 1,998 — mAP(ultralytics val) + 9kp(real 과 동일 정의).  secondary."""
import argparse, json, os, sys
import numpy as np
ROOT = "/home/minjae/Documents/github/pallet-pose"
Y = f"{ROOT}/challenge/yolo_pose_one_model"
NS = f"{Y}/p26_inference_path_audit"
sys.path.insert(0, f"{Y}/runs_arch_baseline")
import ab_common as AC                                              # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True)
ap.add_argument("--tag", required=True); A = ap.parse_args()
out = {"tag": A.tag}
try:
    from ultralytics import YOLO
    m = YOLO(A.weights, task="pose")
    r = m.val(data=f"{Y}/datasets/g38_generic_only/data.yaml", imgsz=640, batch=32,
              device=0, workers=8, plots=False, project=f"{NS}/_val", name=A.tag,
              exist_ok=True, verbose=False)
    out["map"] = {"box_map50": float(r.box.map50), "box_map": float(r.box.map),
                  "pose_map50": float(r.pose.map50), "pose_map": float(r.pose.map)}
    del m
except Exception as e:
    out["map"] = None
    out["map_error"] = f"{type(e).__name__}: {e}"
out["kp9"] = AC.synth_9kp(A.weights, A.tag)
json.dump(out, open(f"{NS}/results/SYNTH_{A.tag}.json", "w"), indent=2, ensure_ascii=False)
print(f"{A.tag}: map {out['map']}  9kp med {out['kp9']['kp_median']}")
