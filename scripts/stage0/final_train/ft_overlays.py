"""56장 REAL_DEV overlay -- 숫자로는 안 보이는 실패 원인을 눈으로 가른다.

평가와 **같은 경로**로 뽑는다: 같은 squash 전처리, 같은 `_decode_peaks`, 같은
`solve_arms`.  따로 그리는 코드를 쓰면 그림과 표가 다른 것을 말하게 된다.

한 장에 담는 것
    초록 실선   GT 8코너 큐보이드
    빨강 점선   F3 재투영 8코너      (최종 추론 경로)
    파랑 점선   F0 재투영 8코너      (Point PnP 만)
    노랑 x      threshold 0.3 을 넘긴 corner 검출점 (배포 검출기가 보는 것)
    헤더        R/t 오차, IoU, 검출 개수, score_4kp

파일명이 곧 정렬 키다: 회전 오차 내림차순 순번을 앞에 붙여서, 폴더를 열면
가장 나쁜 프레임이 맨 위에 온다.
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
import data_paths as DP                           # noqa: E402
import mh_data as MD                              # noqa: E402
import mh_screen as MS                            # noqa: E402
import mh_splitlate as SL                         # noqa: E402
import mh_fusion as FU                            # noqa: E402
import mh_cigm as CG                              # noqa: E402
import re_metrics as RM                           # noqa: E402
import annotate_pnp as APNP                       # noqa: E402
from mh_arms import DH                            # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
import ft_f0f3_eval as EV                         # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/paper_s2_multihead/final_train")
OVER = os.path.join(OUT, "overlays")
GREEN, RED, BLUE, YELLOW = (60, 200, 60), (60, 60, 235), (235, 160, 60), (40, 220, 240)
# camera-facing 0123: 0-3 near face, 4-7 far face, i <-> i+4 depth edges
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_box(canvas, pts, colour, dashed=False, thickness=2):
    ok = np.isfinite(pts).all(1)
    for a, b in EDGES:
        if not (ok[a] and ok[b]):
            continue
        p, q = tuple(np.int32(pts[a])), tuple(np.int32(pts[b]))
        if not dashed:
            cv2.line(canvas, p, q, colour, thickness, cv2.LINE_AA)
            continue
        n = max(int(np.hypot(q[0] - p[0], q[1] - p[1]) / 9), 1)
        for k in range(n):
            if k % 2:
                continue
            s = (int(p[0] + (q[0] - p[0]) * k / n), int(p[1] + (q[1] - p[1]) * k / n))
            e = (int(p[0] + (q[0] - p[0]) * (k + 1) / n),
                 int(p[1] + (q[1] - p[1]) * (k + 1) / n))
            cv2.line(canvas, s, e, colour, thickness, cv2.LINE_AA)


def project(R, t, model_pts, K):
    camera = (R @ model_pts.T).T + t
    depth = np.clip(camera[:, 2], 1e-6, None)
    return (K @ (camera / depth[:, None]).T).T[:, :2]


def banner(canvas, lines):
    pad = 6 + 20 * len(lines)
    strip = canvas[:pad].copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], pad), (0, 0, 0), -1)
    cv2.addWeighted(strip, 0.35, canvas[:pad], 0.65, 0, canvas[:pad])
    for i, (text, colour) in enumerate(lines):
        cv2.putText(canvas, text, (8, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, colour, 1, cv2.LINE_AA)


def main(seed=1):
    MS.deterministic()
    _, _, _, features = MS.lattice()
    weight = json.loads(open(os.path.join(
        ROOT, "data/pallet/results/paper_s2_multihead",
        "theta_posealigned_d0.json")).read())["seeds"]["seed1"]["selected_lambda_theta"]
    model, ckpt = EV.load(seed)

    records = []
    for key in EV.OPEN_SETS:
        for jp, ip, label in EV.frames(key):
            image = cv2.imread(ip)
            if image is None:
                continue
            height, width = image.shape[:2]
            obj = label["objects"][0]
            dims = obj["dimensions_m"]
            model_pts = APNP.make_pallet_keypoints_3d_diagram(
                width=dims["width"], depth=dims["depth"],
                height=dims["height"])[:8]
            extents = (dims["width"], dims["height"], dims["depth"])
            K = CG.intrinsics(label)
            R_gt, t_gt = CG.gt_pose(label)
            gt8 = np.asarray(obj["projected_cuboid"], float)[:8]

            with torch.no_grad():
                out = model(PRE.preprocess_squash(image).to(MD.DEV), features)
                theta_hat, rho_hat = DH.decode(out["line_scores"], *DH.lattice())
                theta_can, rho_can = DH.canonical_from_centred(theta_hat, rho_hat)
            belief = out["beliefs"][-1][0].detach().cpu().numpy()
            peaks = MS._decode_peaks(out["beliefs"][-1][:, :9])[0]
            thresholded = extract_keypoints_from_belief(belief, EV.THRESH)
            n_det = int(sum(1 for k in thresholded[:8] if k[0] >= 0))
            score_4kp = float(np.sort(
                np.max(belief[:8].reshape(8, -1), axis=1))[::-1][3])

            data = {"resolution": np.array([[width, height]]),
                    "model": np.array([model_pts]), "K": np.array([K]),
                    "pred_corner": np.array([peaks]),
                    "pred_theta": theta_can.detach().cpu().numpy(),
                    "pred_rho": rho_can.detach().cpu().numpy(),
                    "support": np.array([EV.support_from_grid(peaks, width, height)])}
            arms, _, _, _ = FU.solve_arms(data, 0, weight)

            canvas = image.copy()
            draw_box(canvas, gt8, GREEN, dashed=False, thickness=2)
            info = [(f"{os.path.basename(ip)}   {key}   seed{seed}", (255, 255, 255))]
            metrics = {}
            for arm, colour in (("F0", BLUE), ("F3", RED)):
                pose = arms.get(arm)
                if pose is None:
                    info.append((f"{arm}: PnP FAILED", colour))
                    metrics[arm] = None
                    continue
                R_p, t_p = pose
                draw_box(canvas, project(R_p, t_p, model_pts, K), colour,
                         dashed=True, thickness=2)
                deg, met = RM.pose_error(R_p, t_p, R_gt, t_gt)
                iou = RM.iou_3d(R_p, t_p, extents, R_gt, t_gt, extents)
                metrics[arm] = deg
                info.append((f"{arm}: R {deg:6.2f}deg  t {met:6.3f}m  IoU {iou:5.3f}",
                             colour))
            for k in thresholded[:8]:
                if k[0] < 0:
                    continue
                cv2.drawMarker(canvas, (int(k[0] * width / 50), int(k[1] * height / 50)),
                               YELLOW, cv2.MARKER_TILTED_CROSS, 12, 2)
            info.append((f"detected {n_det}/8 @0.3   score_4kp {score_4kp:.3f}"
                         f"   GT=green  F0=blue  F3=red", (200, 200, 200)))
            banner(canvas, info)
            records.append({"key": key, "canvas": canvas,
                            "rank_by": metrics.get("F3", np.inf) if
                            metrics.get("F3") is not None else 1e9,
                            "name": os.path.splitext(os.path.basename(ip))[0],
                            "n_det": n_det})

    folder = os.path.join(OVER, f"seed{seed}")
    os.makedirs(folder, exist_ok=True)
    for key in EV.OPEN_SETS:
        os.makedirs(os.path.join(folder, key), exist_ok=True)
    records.sort(key=lambda r: -r["rank_by"])
    for i, rec in enumerate(records):
        path = os.path.join(folder, rec["key"],
                            f"{i:02d}_R{rec['rank_by']:06.2f}_"
                            f"det{rec['n_det']}_{rec['name']}.jpg")
        cv2.imwrite(path, rec["canvas"], [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  seed{seed}: {len(records)}장 -> {os.path.relpath(folder, ROOT)}",
          flush=True)
    del model
    torch.cuda.empty_cache()
    return len(records)


if __name__ == "__main__":
    total = sum(main(s) for s in (1, 2))
    print(f"총 {total}장")
