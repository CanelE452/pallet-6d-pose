"""실험 B 체크포인트를 161장에 돌려 같은 형식으로 덤프 (pallet-yolo26 env)."""
from __future__ import annotations
import json, os, sys
import cv2, numpy as np
ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_frames as MF  # noqa: E402
OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
# 한 줄로 둔다.  이전에 두 줄로 쪼개져 있어 sed 치환이 조용히 실패했고,
# 60 epoch 판정이 5 epoch 체크포인트로 나왔다.
CK = os.path.join(ROOT, "challenge/yolo_pose_one_model/runs_paper/yolo26n_paper_generic_v1_seed42/weights/last.pt")
PAD, IMGSZ, CONF = 100, 640, 0.4
NAME = "yolo26n_paper_generic_v1"


def main():
    from ultralytics import YOLO
    model = YOLO(CK, task="pose")
    dump, n_det = [], 0
    for key, sealed, jp, ip, _ in MF.frames():
        image = cv2.imread(ip)
        padded = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD,
                                    cv2.BORDER_REFLECT_101)
        r = model.predict(padded, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        e = {"set": key, "sealed": sealed,
             "fid": os.path.splitext(os.path.basename(jp))[0],
             "image": os.path.relpath(ip, ROOT),
             "kps": None, "kp_conf": None, "box_conf": None}
        if r.boxes is not None and len(r.boxes):
            i = int(np.argmax(r.boxes.conf.cpu().numpy()))
            e["kps"] = (r.keypoints.xy.cpu().numpy()[i] - PAD).tolist()
            e["box_conf"] = float(r.boxes.conf.cpu().numpy()[i])
            if r.keypoints.conf is not None:
                e["kp_conf"] = r.keypoints.conf.cpu().numpy()[i].tolist()
            n_det += 1
        dump.append(e)
    json.dump({"model": NAME, "weights": os.path.relpath(CK, ROOT),
               "recipe": {"pad": PAD, "imgsz": IMGSZ, "conf": CONF,
                          "checkpoint": "last.pt (60 epoch). in-train val 이므로 "
                                        "best 선택 안 함"},
               "n_frames": len(dump), "n_detected": n_det, "frames": dump},
              open(os.path.join(OUT, f"kps_{NAME}.json"), "w"), indent=1)
    print(f"  {NAME} 검출 {n_det}/{len(dump)}")


if __name__ == "__main__":
    main()
