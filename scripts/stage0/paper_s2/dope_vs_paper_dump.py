"""DOPE(paper_s2_stageB) vs 논문용 YOLO — 지정 프레임 추론 덤프.

08-16 에 히어독으로 한 번 돌리고 버린 코드를 repo 에 남긴다.  그때 것은 scratchpad 에
옛 세션 UUID 가 박혀 있어 다시 돌릴 수 없었다.

**env 가 갈려 한 프로세스에서 둘 다 못 한다.**  DOPE 는 `pallet-pose`, YOLO 는
`pallet-yolo26` 이라 `--part` 로 나눠 두 번 돌린다.  그리기는 여기서 나온 json 만 읽는다.

전처리는 `canonical_det_kp_add.py` 의 parity 규약을 그대로 import 해서 쓴다 —
여기서 새로 정의하지 않는다.  두 모델의 비교가 전처리 차이로 오염되면 안 된다.

    DOPE   anisotropic squash 640x480 -> 400x400, belief(50) -> orig x(W/50, H/50)
    YOLO   BORDER_REFLECT_101 pad 100, imgsz 640, kp conf >= 0.5
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
OUT = f"{ROOT}/data/pallet/eval_results/dope_vs_paper"

# 사용자가 지정한 프레임 — 08-16 그림과 같은 4장.
FIDS = ["1779448868035222528",     # capturenight03    형식 A
        "1779448633156790272",     # capturenight01    형식 A
        "1779449266426633216",     # capturenight06    형식 B
        "1778651530557153024"]     # capturepallet02   형식 B

DOPE_W = f"{ROOT}/weights/paper_s2_stageB/net_epoch_0057.pth"
# ★오른쪽 자리를 목표(yolo26n_base) 에서 논문용 JOINT 로 바꾼 것이 이번 변경의 전부다.
YOLO_W = (f"{ROOT}/challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
          "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt")

_spec = importlib.util.spec_from_file_location(
    "cka", f"{ROOT}/scripts/stage0/paper_s2/canonical_det_kp_add.py")
C = importlib.util.module_from_spec(_spec)
sys.modules["cka"] = C
_spec.loader.exec_module(C)


def frame_paths(fid):
    """fid -> (json, png).  png 가 같이 있는 폴더를 고른다."""
    for pat in (f"{ROOT}/challenge/data/01_real/manual_gt/*_manual_gt/{fid}.json",
                f"{ROOT}/challenge/data/01_real/*/*_manual_gt/{fid}.json"):
        for jp in sorted(glob.glob(pat)):
            ip = jp[:-5] + ".png"
            if os.path.exists(ip):
                return jp, ip
    raise FileNotFoundError(f"{fid}: png 가 있는 manual_gt 를 못 찾음")


def row(pred8, pred_c, peak, jp, sess):
    """그리기에 필요한 것 + 헤더에 찍을 수치.  지표 정의는 C 것을 그대로 쓴다."""
    d, gt8, _, _, _ = C.load_gt(jp)
    dists = C.hungarian(pred8, gt8)
    kp_err = (float(np.median(dists)) if dists is not None and len(dists)
              else float("inf"))
    return {"sess": sess,
            "pred8": [None if not np.isfinite(pred8[i, 0])
                      else [float(pred8[i, 0]), float(pred8[i, 1])]
                      for i in range(8)],
            "pred_c": pred_c,
            "gt8": np.asarray(gt8, float).tolist(),
            "peak": float(peak),
            "n_det": int(sum(np.isfinite(pred8[:, 0]))),
            "kp_err": kp_err,
            "dims": list(C.load_gt(jp)[4])}


def run_dope():
    import cv2
    import torch
    E = C._load_E()
    # C 가 이미 sys.path 를 세워 import 해 뒀다 — 여기서 경로를 다시 만들지 않는다.
    extract_keypoints_from_belief = C.extract_keypoints_from_belief
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E.load_model(DOPE_W, device)
    out = {}
    for fid in FIDS:
        jp, ip = frame_paths(fid)
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        with torch.no_grad():
            beliefs, _ = model(C.preprocess_squash(img).to(device))
        bel = beliefs[-1][0].cpu().numpy()
        kps = extract_keypoints_from_belief(bel, C.THRESH)
        sx, sy = W / 50.0, H / 50.0
        pred8 = np.full((8, 2), np.nan)
        for i, k in enumerate(kps[:8]):
            if k[0] >= 0:
                pred8[i] = [k[0] * sx, k[1] * sy]
        pred_c = ([float(kps[8][0] * sx), float(kps[8][1] * sy)]
                  if kps[8][0] >= 0 else None)
        # peak 은 코너 8채널 기준.  centroid(9번째)를 넣으면 08-16 그림과 값이 달라진다
        # (night03 에서 0.10 -> 0.15 로 뜬다).
        out[fid] = row(pred8, pred_c, float(bel[:8].max()), jp,
                       os.path.basename(os.path.dirname(jp)).replace("_manual_gt", ""))
    return out


def run_yolo():
    import cv2
    from ultralytics import YOLO
    model = YOLO(YOLO_W, task="pose")
    out = {}
    for fid in FIDS:
        jp, ip = frame_paths(fid)
        img = cv2.imread(ip)
        inp = cv2.copyMakeBorder(img, C.YOLO_PAD, C.YOLO_PAD, C.YOLO_PAD,
                                 C.YOLO_PAD, cv2.BORDER_REFLECT_101)
        r = model.predict(inp, verbose=False, conf=0.4, imgsz=640)[0]
        pred8, pred_c, peak = np.full((8, 2), np.nan), None, 0.0
        if r.boxes is not None and len(r.boxes):
            b = int(np.argmax(r.boxes.conf.cpu().numpy()))
            peak = float(r.boxes.conf.cpu().numpy()[b])
            kp = r.keypoints.data.cpu().numpy()[b]
            kp[:, 0] -= C.YOLO_PAD
            kp[:, 1] -= C.YOLO_PAD
            for i in range(8):
                if kp[i, 2] >= 0.5:
                    pred8[i] = kp[i, :2]
            if kp[8, 2] >= 0.5:
                pred_c = [float(kp[8, 0]), float(kp[8, 1])]
        out[fid] = row(pred8, pred_c, peak, jp,
                       os.path.basename(os.path.dirname(jp)).replace("_manual_gt", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["dope", "yolo"], required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = run_dope() if a.part == "dope" else run_yolo()

    dst = f"{OUT}/pred_{a.part}.json"
    json.dump({"weights": os.path.relpath(
                   DOPE_W if a.part == "dope" else YOLO_W, ROOT),
               "fids": FIDS, "pred": res}, open(dst, "w"), indent=1)
    for fid in FIDS:
        r = res[fid]
        print(f"  {fid}  {r['sess']:18} det {r['n_det']}/8  "
              f"peak {r['peak']:.3f}  kp_med {r['kp_err']:.1f}px", flush=True)
    print(f"-> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
