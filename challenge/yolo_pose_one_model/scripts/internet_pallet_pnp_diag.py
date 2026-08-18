"""internet_pallet_pnp_diag.py — "점은 맞는데 큐보이드가 뒤틀린다" 원인 진단.

`internet_pallet_yolo_ab.py` 의 A/B 오버레이에서 keypoint 는 팔레트 코너에 붙었는데
빨간 큐보이드만 어긋나 보이는 현상을 분해한다. 결론은 두 층이다.

  (1) 지배 원인 — 그리기 버그.  `solve_pose(auto_swap_dims=True)` 는 as-given 과
      W/D swap 두 가설을 모두 풀고 reproj 가 낮은 쪽을 고른다. 이 이미지에서는
      swapped(1.220 x 1.016) 가 선택된다. 그런데 A/B 의 draw() 는 pose 를 만든
      dims 가 아니라 파일명 dims(1.016 x 1.220) 로 3D 박스를 다시 만들어 같은
      (R,t) 로 투영한다. 즉 pose 와 다른 모양의 상자를 그린다 → 화면에서 39px
      (최대 82px) 어긋남. 헤더에 찍히는 reproj 5.5px 는 *풀린* pose 의 값이라
      숫자와 그림이 따로 논다.

  (2) 2차 원인 — keypoint 의 높이 과대.  코너가 시사하는 높이는 150mm 인데
      파일명 H=120mm 를 강제하므로 reproj 가 2.3 -> 5.5px 로 올라간다. 다만 이건
      z(깊이)와 f 로 흡수되고 회전은 0.23도밖에 안 움직인다 → "뒤틀림"의 원인은
      아니다.

사용: conda activate pallet-pose
      python challenge/yolo_pose_one_model/scripts/internet_pallet_pnp_diag.py
      (keypoint 는 internet_pallet_kps_dump.py 가 만든 kps.json 을 읽는다)
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "annotate"))
import annotate_pnp as APNP  # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/eval_results/internet_pallet_pnp_diag")
KPS_JSON = os.path.join(OUT, "kps.json")
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def K_from_hfov(w, h, hfov):
    fx = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1]], np.float64)


def sweep(kps9, dims, shape, lo=30.0, hi=121.0, step=2.5):
    """A/B 스크립트와 동일한 HFOV 스윕 (reproj 최소 focal 채택)."""
    h, w = shape[:2]
    best = (None, None, np.inf)
    for hfov in np.arange(lo, hi, step):
        try:
            pose = APNP.solve_pose(kps9, K_from_hfov(w, h, hfov), dims=dims,
                                   img_shape=shape)
        except Exception:
            continue
        if pose is not None and pose["reproj_error_px"] < best[2]:
            best = (pose, float(hfov), float(pose["reproj_error_px"]))
    return best


def project(dims, pose, K):
    X = APNP.make_pallet_keypoints_3d(*dims)
    R = np.asarray(pose["R"], float)
    t = np.asarray(pose["t"], float).ravel()
    P = (R @ X.T).T + t
    return np.stack([K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2],
                     K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]], 1)


def draw_panel(img, obs, boxes, title, sub):
    """boxes: [(uv(9,2), color, thickness)] — 여러 큐보이드를 겹쳐 그린다."""
    vis = img.copy()
    for uv, color, th in boxes:
        Q = [(int(round(u)), int(round(v))) for u, v in uv[:8]]
        for a, b in EDGES:
            cv2.line(vis, Q[a], Q[b], color, th, cv2.LINE_AA)
    for i, (x, y) in enumerate(obs[:8]):
        c = (255, 60, 0) if i in (0, 1, 4, 5) else (0, 170, 255)
        cv2.drawMarker(vis, (int(round(x)), int(round(y))), c,
                       cv2.MARKER_CROSS, 15, 2)
    cv2.circle(vis, (int(obs[8, 0]), int(obs[8, 1])), 4, (255, 255, 255), -1)
    bar = np.zeros((62, vis.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, sub, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (170, 220, 170), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def main():
    rec = list(json.load(open(KPS_JSON)).values())[0]
    img = cv2.imread(rec["path"])
    h, w = img.shape[:2]
    shape = img.shape
    P8 = np.array([[float(a), float(b)] for a, b in rec["kps8"]])
    C = np.array(rec["centroid"], float)
    obs = np.vstack([P8, C])
    kps9 = [list(p) for p in P8] + [list(C)]
    Wf, Df, Hf = rec["dims_from_filename_m"]

    def err(uv):
        e = np.linalg.norm(uv - obs, axis=1)
        return e.mean(), e.max()

    # ── 1) 현행 경로: 파일명 dims 로 solve → swapped 채택 → 파일명 dims 로 그림
    pose, hfov, _ = sweep(kps9, (Wf, Df, Hf), shape)
    K = K_from_hfov(w, h, hfov)
    uv_bug = project((Wf, Df, Hf), pose, K)        # draw() 가 실제로 그리는 상자
    uv_fix = project(pose["dims"], pose, K)        # pose 가 실제로 푼 상자
    e_bug, m_bug = err(uv_bug)
    e_fix, m_fix = err(uv_fix)

    # ── 2) 그리기 수정 + H 를 자유변수로 (스윕으로 최적 H 재탐색)
    best_H, best = None, (None, None, np.inf)
    for Hmm in range(100, 201, 5):
        p, hf, e = sweep(kps9, (pose["dims"][0], pose["dims"][1], Hmm / 1000.0), shape)
        if p is not None and e < best[2]:
            best, best_H = (p, hf, e), Hmm
    poseH, hfovH, _ = best
    KH = K_from_hfov(w, h, hfovH)
    uv_H = project(poseH["dims"], poseH, KH)
    e_H, m_H = err(uv_H)

    print(f"[현행 draw]  hfov={hfov:.1f} pose_dims={pose['dims']} "
          f"hyp={pose['_wd_hypothesis']}  그린 상자 reproj mean={e_bug:.2f} max={m_bug:.2f}px")
    print(f"[draw 수정]  같은 pose, pose['dims'] 로 그림      "
          f"reproj mean={e_fix:.2f} max={m_fix:.2f}px  z={pose['t'][2]:.2f}m")
    print(f"[H 자유]     H={best_H}mm hfov={hfovH:.1f}          "
          f"reproj mean={e_H:.2f} max={m_H:.2f}px  z={poseH['t'][2]:.2f}m")

    RED, GRN, YEL = (0, 0, 255), (0, 220, 0), (0, 220, 255)
    p1 = draw_panel(img, obs, [(uv_bug, RED, 2)],
                    "A) CURRENT internet_pallet_yolo_ab.draw()  -- BUG",
                    f"pose solved with dims {pose['dims']} but box built from "
                    f"filename dims ({Wf},{Df},{Hf}) | mean {e_bug:.1f}px max {m_bug:.1f}px")
    p2 = draw_panel(img, obs, [(uv_fix, GRN, 2)],
                    "B) SAME POSE, drawn with pose['dims']  -- draw bug fixed",
                    f"dims {pose['dims']}  hfov {hfov:.0f}  z {pose['t'][2]:.2f}m | "
                    f"mean {e_fix:.1f}px max {m_fix:.1f}px")
    p3 = draw_panel(img, obs, [(uv_H, YEL, 2)],
                    f"C) draw fixed + H free ({best_H}mm instead of 120mm)",
                    f"dims {tuple(round(v,3) for v in poseH['dims'])}  hfov {hfovH:.0f}  "
                    f"z {poseH['t'][2]:.2f}m | mean {e_H:.1f}px max {m_H:.1f}px")
    p4 = draw_panel(img, obs, [(uv_bug, RED, 2), (uv_fix, GRN, 2), (uv_H, YEL, 2)],
                    "D) overlay: RED=current bug  GREEN=fixed(H120)  YELLOW=fixed(H free)",
                    "rotation between GREEN and YELLOW = 0.23 deg -- H error does NOT twist pose")

    os.makedirs(OUT, exist_ok=True)
    montage = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])
    fp = os.path.join(OUT, "pnp_distortion_diagnosis.jpg")
    cv2.imwrite(fp, montage, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"[out] {fp}")


if __name__ == "__main__":
    main()
