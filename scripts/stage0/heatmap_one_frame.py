"""heatmap_one_frame.py — 한 프레임의 9개 belief 히트맵(코너 0~7 + centroid 8)을
s2(Stage B, squash) 추론으로 시각화. 어느 코너 채널이 약한지(미검출 원인) 눈으로 확인.

프레임: noapril/1775201432466607872 (det 6/8, f3(flip)로 fail). --jp/--ip 로 교체 가능.

- belief(9,50,50) squash decode → 각 채널을 원본 프레임 크기로 upsample, JET 컬러로
  회색조 이미지 위 alpha 블렌드. argmax(피크) 위치에 원(검출=THRESH 넘으면 실선, 미만=점선표시)
  + 헤더에 peak 값. 채널별 자기-정규화(위치 잘 보이게) + peak 절대값 병기.
- 상단: 원본 + 검출 키포인트(파랑) 오버레이 1장.

산출: data/pallet/eval_results/paper_s2_scratch_diffpnp/heatmap_<fid>.jpg

Usage: conda activate pallet-pose; python scripts/stage0/heatmap_one_frame.py
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import paper_s2_testset17_9filters as T   # noqa: E402  (infer_squash, THRESH, E, WEIGHTS)
import cv2      # noqa: E402
import torch    # noqa: E402

THRESH = T.THRESH
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "paper_s2_scratch_diffpnp")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NAMES = ["0 front-top-L", "1 front-top-R", "2 front-bot-L", "3 front-bot-R",
         "4 rear-top-L", "5 rear-top-R", "6 rear-bot-L", "7 rear-bot-R", "8 centroid"]
FRONT_TXT, REAR_TXT, CEN_TXT = (255, 255, 0), (0, 200, 255), (0, 255, 0)


def heat_panel(gray_bgr, chan, name, peak, detected):
    """belief 채널(50x50) → 원본크기 JET 오버레이 패널."""
    H, W = gray_bgr.shape[:2]
    c = chan.copy()
    c = np.clip(c, 0, None)
    m = c.max()
    norm = (c / m) if m > 1e-9 else c
    big = cv2.resize((norm * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
    jet = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    over = cv2.addWeighted(gray_bgr, 0.45, jet, 0.55, 0)
    # peak 위치
    py, px = np.unravel_index(int(np.argmax(chan)), chan.shape)
    ox, oy = int(px / 50.0 * W), int(py / 50.0 * H)
    col = (0, 255, 0) if detected else (0, 0, 255)
    cv2.circle(over, (ox, oy), 9, col, 2, cv2.LINE_AA)
    cv2.drawMarker(over, (ox, oy), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
    # 헤더
    tag = "DET" if detected else "miss(<thr)"
    idx = int(name.split()[0])
    tc = CEN_TXT if idx == 8 else (FRONT_TXT if idx < 4 else REAR_TXT)
    cv2.rectangle(over, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(over, f"{name}  peak={peak:.2f}  {tag}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 1, cv2.LINE_AA)
    return over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jp", default=os.path.join(
        ROOT, "challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt/1775201432466607872.json"))
    ap.add_argument("--ip", default=os.path.join(
        ROOT, "challenge/data/01_real/eval_canonical/capture0403noapril_manual_gt/1775201432466607872.png"))
    args = ap.parse_args()
    fid = os.path.splitext(os.path.basename(args.ip))[0]

    img = cv2.imread(args.ip)
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    model = T.E.load_model(T.WEIGHTS, DEV)
    belief, pred8, pred_c, peaks, ratios = T.infer_squash(model, img, DEV)
    n_det = int((~np.isnan(pred8[:, 0])).sum())
    det_flags = [not np.isnan(pred8[i, 0]) for i in range(8)] + [pred_c is not None]
    print(f"[frame] {fid}  det={n_det}/8  peaks=" +
          " ".join(f"{i}:{peaks[i]:.2f}" for i in range(9)))
    miss = [i for i in range(8) if not det_flags[i]]
    print(f"  미검출 코너(<{THRESH}): {miss}")

    # 상단: 원본 + 검출 키포인트
    top = img.copy()
    for i in range(8):
        if det_flags[i]:
            p = (int(pred8[i, 0]), int(pred8[i, 1]))
            cv2.circle(top, p, 5, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(top, str(i), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)
    if pred_c is not None:
        cv2.drawMarker(top, (int(pred_c[0]), int(pred_c[1])), (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
    cv2.rectangle(top, (0, 0), (top.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(top, f"{fid}  det={n_det}/8  detected kp=blue  (miss corners: {miss})",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 9 채널 패널
    panels = [heat_panel(gray, belief[i], NAMES[i], peaks[i], det_flags[i]) for i in range(9)]

    # 레이아웃: 1행 원본(2칸폭) + 3x3 히트맵
    W = img.shape[1]
    def resize_w(x, w):
        return cv2.resize(x, (w, int(x.shape[0] * w / x.shape[1])))
    cell_w = W // 3 * 3 // 3  # = W/3 approx; keep panels at W/3
    pw = W // 3
    grid_rows = []
    for r in range(3):
        row = [resize_w(panels[r * 3 + c], pw) for c in range(3)]
        h = max(x.shape[0] for x in row)
        row = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)) for x in row]
        grid_rows.append(np.hstack(row))
    grid = np.vstack(grid_rows)
    topb = resize_w(top, grid.shape[1])
    out = np.vstack([topb, grid])
    p = os.path.join(OUT_DIR, f"heatmap_{fid}.jpg")
    cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"[save] {p}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
