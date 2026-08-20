"""1단계 (pallet-pose env) — FINAL40K 의 키포인트를 YOLO 와 같은 형식으로 덤프.

같은 것을 재기 위한 두 가지 대칭:
  * 좌표는 **원본 픽셀** 로 낸다 (YOLO 와 동일).  belief 50-grid 는 (W/50, H/50) 로
    되돌린다 — REAL_DEV 평가에서 쓴 것과 같은 변환.
  * 검출 여부는 threshold 0.3 을 넘긴 코너 개수로 판정한다.  YOLO 는 box conf 0.4
    로 프레임 단위 검출을 판정하므로 의미가 완전히 같지는 않다.  그 비대칭을
    숨기지 않고 `detection_criterion` 에 적어 둔다.

전처리는 이 모델이 학습된 anisotropic squash (400x400) 다.  YOLO 의 PAD=100 은
YOLO 계약이지 이 모델 계약이 아니므로 적용하지 않는다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0", "scripts/stage0/paper_s2", "scripts/stage0/multihead",
            "scripts/stage0/line", "scripts/stage0/final_train", "challenge"):
    sys.path.insert(0, os.path.join(ROOT, sub))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2                                        # noqa: E402
import paper_s2_real_eval as PRE                  # noqa: E402
import mh_data as MD                              # noqa: E402
import mh_screen as MS                            # noqa: E402
import ft_f0f3_eval as EV                         # noqa: E402
import mc_frames as MF                            # noqa: E402
from mh_arms import DH                            # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
THRESH, N_DET_MIN = 0.3, 6


def main(seed=1, step=25000, tag=None):
    MS.deterministic()
    _, _, _, features = MS.lattice()
    import torch as _t, mh_splitlate as _SL
    _path = os.path.join(ROOT, "weights/paper_s2/paper_s2_multihead",
                         f"screen_A1_CORNER_LINE_FINAL40K_seed{seed}",
                         f"step_{step:05d}.pth")
    _state = _t.load(_path, map_location=MD.DEV, weights_only=False)
    model = _SL.SplitLate(_state["arm"]); model.load_state_dict(_state["model"])
    model.to(MD.DEV).eval(); ckpt = os.path.relpath(_path, ROOT)
    os.makedirs(OUT, exist_ok=True)
    rows = MF.frames()

    dump, n_det_frames = [], 0
    for key, sealed, jp, ip, _label in rows:
        image = cv2.imread(ip)
        height, width = image.shape[:2]
        with torch.no_grad():
            out = model(PRE.preprocess_squash(image).to(MD.DEV), features)
            theta_hat, rho_hat = DH.decode(out["line_scores"], *DH.lattice())
            theta_can, rho_can = DH.canonical_from_centred(theta_hat, rho_hat)
        belief = out["beliefs"][-1][0].detach().cpu().numpy()
        peaks = MS._decode_peaks(out["beliefs"][-1][:, :9])[0]
        sx, sy = width / 50.0, height / 50.0

        thresholded = extract_keypoints_from_belief(belief, THRESH)
        kps = []
        for i in range(9):
            k = thresholded[i]
            kps.append([float(k[0] * sx), float(k[1] * sy)]
                       if k[0] >= 0 else [float("nan"), float("nan")])
        n_det = int(sum(1 for k in kps[:8] if np.isfinite(k[0])))
        n_det_frames += int(n_det >= N_DET_MIN)
        peak_values = np.sort(np.max(belief[:8].reshape(8, -1), axis=1))[::-1]

        dump.append({
            "set": key, "sealed": sealed,
            "fid": os.path.splitext(os.path.basename(jp))[0],
            "image": os.path.relpath(ip, ROOT),
            "kps": kps,
            "kps_argmax": (peaks * np.array([sx, sy])).tolist(),
            "kp_conf": [float(np.max(belief[i])) for i in range(9)],
            "box_conf": None,
            "score_4kp": float(peak_values[3]),
            "n_det": n_det,
            "line_theta": theta_can[0].detach().cpu().numpy().tolist(),
            "line_rho": rho_can[0].detach().cpu().numpy().tolist(),
        })

    name = tag or f"FINAL40K_seed{seed}"
    payload = {"model": name, "weights": ckpt,
               "recipe": {"preprocess": "anisotropic squash 400x400 "
                                        "(the model's training preprocess)",
                          "belief_to_pixels": "(W/50, H/50)",
                          "threshold": THRESH,
                          "pad": None,
                          "note": "YOLO's PAD=100 is a YOLO contract, not this "
                                  "model's -- applying it would be off-distribution"},
               "detection_criterion": f"n_det >= {N_DET_MIN} corners above "
                                      f"belief {THRESH} (YOLO uses box conf 0.4 "
                                      f"instead -- not the same quantity)",
               "extra": "kps_argmax = threshold-free peaks, needed by the F3 "
                        "solver which requires all 8",
               "n_frames": len(dump), "n_detected": n_det_frames, "frames": dump}
    target = os.path.join(OUT, f"kps_{name}.json")
    json.dump(payload, open(target, "w"), indent=1)
    print(f"  {name}  검출(n_det>=6) {n_det_frames}/{len(dump)} "
          f"-> {os.path.basename(target)}", flush=True)


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1:
        for _st in (int(x) for x in _s.argv[1].split(",")):
            main(1, _st, f"FINAL40K_seed1_step{_st}")
    else:
        main(1)
