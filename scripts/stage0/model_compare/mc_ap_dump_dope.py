"""DOPE 계열의 AP 용 점수 덤프 — positive 정본 161 + real negative 2,689.

이 파일이 생긴 이유는 내 판단 착오다.  `mc_ap_dump` 는 DOPE 를 "score_4kp 는 box
conf 와 의미가 달라 같은 곡선에 못 올린다" 며 뺐는데, **AP / AUROC / FPR@TPR95 는
전부 순위 기반이라 점수의 단조변환에 불변**이다.  재는 것은 "그 모델이 자기 점수로
positive 와 negative 를 얼마나 갈라내는가" 이고, 그건 점수 정의가 달라도 비교된다.
오히려 threshold-free 지표를 쓰는 이유가 그것이다.

점수 = `score_4kp` (코너 belief peak 중 4번째로 큰 값).  `mc_dump_dope` 가 쓰는 것과
같은 양이고 전처리도 같다 — anisotropic squash 400x400, YOLO 의 PAD=100 은 이 모델
계약이 아니라 적용하지 않는다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0", "scripts/stage0/paper_s2",
            "scripts/stage0/multihead", "scripts/stage0/line",
            "scripts/stage0/final_train", "challenge"):
    sys.path.insert(0, os.path.join(ROOT, sub))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2                                        # noqa: E402
import paper_s2_real_eval as PRE                  # noqa: E402
import mh_data as MD                              # noqa: E402
import mh_screen as MS                            # noqa: E402
import mc_frames as MF                            # noqa: E402
import mc_ap_dump as MAP                          # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/model_compare")
STEPS = {"FINAL40K_seed1": 25000,
         "FINAL40K_seed1_step18000": 18000,
         "FINAL40K_seed1_step12000": 12000,
         "FINAL40K_seed1_step6000": 6000}


def load(step, seed=1):
    import mh_splitlate as SL
    path = os.path.join(ROOT, "weights/paper_s2/paper_s2_multihead",
                        f"screen_A1_CORNER_LINE_FINAL40K_seed{seed}",
                        f"step_{step:05d}.pth")
    state = torch.load(path, map_location=MD.DEV, weights_only=False)
    model = SL.SplitLate(state["arm"])
    model.load_state_dict(state["model"])
    return model.to(MD.DEV).eval(), os.path.relpath(path, ROOT)


def score(model, features, image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    with torch.no_grad():
        out = model(PRE.preprocess_squash(image).to(MD.DEV), features)
    belief = out["beliefs"][-1][0].detach().cpu().numpy()
    peaks = np.sort(np.max(belief[:8].reshape(8, -1), axis=1))[::-1]
    return float(peaks[3])


def main():
    MS.deterministic()
    _, _, _, features = MS.lattice()
    pos = [ip for _k, _s, _jp, ip, _lab in MF.frames()]
    neg = MAP.negative_images()
    print(f"positive {len(pos)}장  negative {len(neg)}장", flush=True)

    target = os.path.join(OUT, "AP_SCORES.json")
    payload = json.load(open(target))
    for name, step in STEPS.items():
        model, ckpt = load(step)
        p = [score(model, features, ip) for ip in pos]
        n = [score(model, features, ip) for ip in neg]
        p = [x for x in p if x is not None]
        n = [x for x in n if x is not None]
        payload["models"][name] = {"weights": ckpt, "pos": p, "neg": n,
                                   "score": "score_4kp (4th corner belief peak)"}
        print(f"  {name:26} pos median {np.median(p):.3f}  "
              f"neg p99 {np.percentile(n, 99):.3f}", flush=True)
    json.dump(payload, open(target, "w"))
    print(f"-> {target}", flush=True)


if __name__ == "__main__":
    main()
