"""FINAL40K 를 배포영상 프레임 4장에 추론 — 처음 보는 환경에서 어디까지 가나.

★ 적용범위 경고를 그림에 박아 넣는다. 이 모델의 학습셋은 BROAD 합성 40,000 뿐이고
  real 은 0장, challenge 트랙의 v1/v2 자기 팔레트도 의도적으로 제외했다. 이 영상은
  그 밖이다. 못 잡아도 모델 결함이 아니라 적용범위 밖이라는 뜻이다.

★ K 가 없다. 1280x720 이고 task.yaml 의 K 는 640x480(4:3) 용이라 단순 스케일이
  성립하지 않는다. 지어내지 않고 repo 의 기존 해법 `internet_pallet_infer.fit_K`
  (HFOV 스윕으로 reproj 최소 focal 채택)를 그대로 쓴다. 고정 HFOV 를 가정하면 K
  오차가 포즈 기울임으로 흡수돼 큐보이드가 깔때기로 왜곡된다.

★ 치수도 모른다. challenge PALLET_DIMS (1.1 x 1.3 x 0.11 m) 를 가정으로 쓰고
  헤더에 assumed 라고 적는다.

추론 경로는 REAL_DEV 평가와 동일하다 — 같은 squash 전처리, 같은 `_decode_peaks`,
같은 `solve_arms`. 그래야 이 그림과 앞선 숫자가 같은 것을 말한다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0", "scripts/stage0/paper_s2", "scripts/stage0/multihead",
            "scripts/stage0/line", "scripts/stage0/real_eval", "challenge",
            "scripts/annotate"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import cv2                                        # noqa: E402
import paper_s2_real_eval as PRE                  # noqa: E402
import mh_data as MD                              # noqa: E402
import mh_screen as MS                            # noqa: E402
import mh_fusion as FU                            # noqa: E402
import annotate_pnp as APNP                       # noqa: E402
from mh_arms import DH                            # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
import ft_f0f3_eval as EV                         # noqa: E402
import internet_pallet_infer as IPI               # noqa: E402

SRC = os.path.join(ROOT, "data/pallet/raw_data/vdoframes")
OUT = os.path.join(ROOT,
                   "data/pallet/results/paper_s2_multihead/final_train/vdo_infer")
FRAMES = ["000037", "000076", "000483", "001678"]
DIMS = APNP.PALLET_DIMS                    # (width, depth, height) — assumed
THRESH = 0.3
# 0..8 별 색 (BGR).  index 를 눈으로 세지 않아도 되게 채도를 갈라 둔다.
KP_COLORS = [(255, 255, 0), (255, 0, 0), (255, 0, 255), (0, 0, 255),
             (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255),
             (255, 255, 255)]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def draw(image, pred8, pred_c, proj, header, sub):
    vis = image.copy()
    if proj is not None:
        for a, b in EDGES:
            if np.isfinite(proj[a]).all() and np.isfinite(proj[b]).all():
                cv2.line(vis, tuple(np.int32(proj[a])), tuple(np.int32(proj[b])),
                         (0, 255, 255), 2, cv2.LINE_AA)
    for i in range(8):
        if not np.isfinite(pred8[i, 0]):
            continue
        p = (int(pred8[i, 0]), int(pred8[i, 1]))
        cv2.circle(vis, p, 6, KP_COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(vis, p, 7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, str(i), (p[0] + 8, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, KP_COLORS[i], 2, cv2.LINE_AA)
    if pred_c is not None:
        cv2.circle(vis, (int(pred_c[0]), int(pred_c[1])), 6, (255, 255, 255),
                   -1, cv2.LINE_AA)
        cv2.putText(vis, "8", (int(pred_c[0]) + 8, int(pred_c[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    for j, (text, colour) in enumerate((header, sub)):
        cv2.putText(vis, text, (10, 28 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, (10, 28 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, colour, 2, cv2.LINE_AA)
    return vis


def main(seed=1):
    MS.deterministic()
    _, _, _, features = MS.lattice()
    weight = json.loads(open(os.path.join(
        ROOT, "data/pallet/results/paper_s2_multihead",
        "theta_posealigned_d0.json")).read())["seeds"]["seed1"]["selected_lambda_theta"]
    model, ckpt = EV.load(seed)
    os.makedirs(OUT, exist_ok=True)

    panels, records = [], []
    for stem in FRAMES:
        path = os.path.join(SRC, f"{stem}.png")
        image = cv2.imread(path)
        height, width = image.shape[:2]

        with torch.no_grad():
            out = model(PRE.preprocess_squash(image).to(MD.DEV), features)
            theta_hat, rho_hat = DH.decode(out["line_scores"], *DH.lattice())
            theta_can, rho_can = DH.canonical_from_centred(theta_hat, rho_hat)
        belief = out["beliefs"][-1][0].detach().cpu().numpy()
        peaks = MS._decode_peaks(out["beliefs"][-1][:, :9])[0]
        sx, sy = width / 50.0, height / 50.0

        kps = extract_keypoints_from_belief(belief, THRESH)
        pred8 = np.full((8, 2), np.nan)
        for i, k in enumerate(kps[:8]):
            if k[0] >= 0:
                pred8[i] = [k[0] * sx, k[1] * sy]
        pred_c = ([kps[8][0] * sx, kps[8][1] * sy] if kps[8][0] >= 0 else None)
        n_det = int(np.isfinite(pred8[:, 0]).sum())
        score_4kp = float(np.sort(np.max(belief[:8].reshape(8, -1), 1))[::-1][3])

        fit = IPI.fit_K(pred8, pred_c, DIMS, image.shape)
        proj, note = None, "PnP skipped (det<6)"
        R_deg = None
        if fit is not None:
            K, hfov, reproj = fit
            data = {"resolution": np.array([[width, height]]),
                    "model": np.array([APNP.make_pallet_keypoints_3d_diagram(
                        width=DIMS[0], depth=DIMS[1], height=DIMS[2])[:8]]),
                    "K": np.array([K]), "pred_corner": np.array([peaks]),
                    "pred_theta": theta_can.detach().cpu().numpy(),
                    "pred_rho": rho_can.detach().cpu().numpy(),
                    "support": np.array([EV.support_from_grid(peaks, width, height)])}
            arms, _, model_pts, _ = FU.solve_arms(data, 0, weight)
            pose = arms.get("F3") or arms.get("F0")
            used = "F3" if arms.get("F3") is not None else (
                "F0" if arms.get("F0") is not None else None)
            if pose is not None:
                camera = (pose[0] @ model_pts.T).T + pose[1]
                depth = np.clip(camera[:, 2], 1e-6, None)
                proj = (K @ (camera / depth[:, None]).T).T[:, :2]
                note = (f"{used}  hfov {hfov:.0f}deg  reproj {reproj:.1f}px  "
                        f"dist {np.linalg.norm(pose[1]):.2f}m")
            else:
                note = "PnP FAIL"
        colour = (0, 255, 0) if n_det >= 6 else (0, 200, 255)
        header = f"{stem}   det {n_det}/8   score_4kp {score_4kp:.2f}"
        panels.append(draw(image, pred8, pred_c, proj, (header, colour),
                           (note + "   | dims ASSUMED 1.1x1.3x0.11m  K from HFOV fit",
                            (200, 200, 200))))
        records.append({"frame": stem, "n_det": n_det,
                        "score_4kp": round(score_4kp, 4), "note": note})
        print(f"  {stem}  det {n_det}/8  score_4kp {score_4kp:.3f}  {note}",
              flush=True)
        cv2.imwrite(os.path.join(OUT, f"{stem}_seed{seed}.jpg"), panels[-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    top = np.hstack(panels[:2]); bottom = np.hstack(panels[2:])
    grid = np.vstack([top, bottom])
    grid_path = os.path.join(OUT, f"vdo_grid_seed{seed}.jpg")
    cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    json.dump({"model": ckpt, "frames": records,
               "scope_warning": "학습셋 = BROAD 합성 40,000 뿐. real 0장, "
                                "challenge v1/v2 제외. 이 영상은 적용범위 밖이다.",
               "K": "unknown -> HFOV sweep (internet_pallet_infer.fit_K)",
               "dims": "ASSUMED PALLET_DIMS 1.1 x 1.3 x 0.11 m"},
              open(os.path.join(OUT, f"vdo_infer_seed{seed}.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"-> {os.path.relpath(grid_path, ROOT)}")


if __name__ == "__main__":
    main(1)
