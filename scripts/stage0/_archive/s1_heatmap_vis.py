"""s1_heatmap_vis.py — Paper-S1 raw belief map(heatmap) 시각화.

목적: DOPE(Paper-S1, net_epoch_0065)가 한 프레임을 "어떻게 보는지"를 belief map
      9채널(corner 0~7 + centroid 8) 그대로 이미지 위에 컬러맵 오버레이해서
      "각 코너를 어디에 얼마나 강하게 보는지" 보여준다. (PnP/pose 없음)

전처리: reflect-pad100 (overlay 관례와 동일; belief_to_orig_pad 역변환으로 정합).
belief 격자(50x50 등)는 padded-canvas(W+2P,H+2P)로 upsample 후 pad crop 하여
원본 좌표에 정합 (검출 kp 의 belief_to_orig_pad 와 동일 매핑).

출력: data/pallet/eval_results/s1_heatmap_vis/
  {fid}_9ch_grid.jpg   9채널 3x3 그리드 (각 채널 raw belief, 고정 0~1 스케일)
  {fid}_combined.jpg   9채널 max 합성 1장
  chN.jpg (--per_channel)  개별 채널 9장
  summary.md           코너별 peak 강도 + 관찰
"""
from __future__ import annotations
import argparse
import importlib.util
import os
import sys

import cv2
import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

from filter_pr_camfacing import extract_keypoints_from_belief
from eval_pvnet_heads import preprocess

# 재사용: load_model / pad_frame / belief_to_orig_pad (paper_s1 파이프라인과 동일)
ECAD_PATH = os.path.join(
    ROOT, "data/pallet/eval_results/stage16_truncation_addon/"
    "capturecad_b2_eval/eval_capturecad_b2.py")
_spec = importlib.util.spec_from_file_location("ecad", ECAD_PATH)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

WEIGHTS = os.path.join(ROOT, "weights/paper_s1_maskaux/net_epoch_0065.pth")
DEF_FRAME = os.path.join(
    ROOT, "data/pallet/raw_data/outside/capturepalletcad/rgb/"
    "1778653008732328960.png")
OUT_DIR = os.path.join(ROOT, "data/pallet/eval_results/s1_heatmap_vis")
PAD = 100
THRESH = 0.3

CH_NAMES = ["corner0", "corner1", "corner2", "corner3",
            "corner4", "corner5", "corner6", "corner7", "centroid8"]
# front 0-3 = green title, rear 4-7 = orange, centroid = magenta
CH_COL = [(0, 255, 0)] * 4 + [(0, 165, 255)] * 4 + [(255, 0, 255)]


def belief_to_orig_map(belief_ch, W, H, pad):
    """belief 격자(bh,bw) -> 원본 이미지 좌표(H,W)로 정합된 heatmap.

    belief_to_orig_pad 와 동일 선형 매핑: belief 격자는 padded-resized canvas(W×H)를
    꽉 채우고, 그 canvas 는 padded 원본(W+2P,H+2P)의 축소본 → pad crop 하면 원본.
    두 선형 resize 를 합쳐 (W+2P,H+2P) 로 한 번에 확대 후 pad 제거.
    """
    if pad > 0:
        full = cv2.resize(belief_ch, (W + 2 * pad, H + 2 * pad),
                          interpolation=cv2.INTER_LINEAR)
        return full[pad:pad + H, pad:pad + W]
    return cv2.resize(belief_ch, (W, H), interpolation=cv2.INTER_LINEAR)


def overlay_heat(img, heat, vmax=1.0, alpha=0.75):
    """heat(0..vmax) 를 JET 컬러맵으로 img 위에 heat-비례 알파 블렌딩."""
    h = np.clip(heat / vmax, 0.0, 1.0)
    color = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    a = (h * alpha)[..., None]           # 강한 곳만 진하게, 약/diffuse 는 흐리게
    return (img * (1 - a) + color * a).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=DEF_FRAME)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--per_channel", action="store_true",
                    help="개별 채널 9장(chN.jpg)도 저장")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E.load_model(args.weights, device)

    img = cv2.imread(args.frame)
    if img is None:
        raise SystemExit(f"프레임 로드 실패: {args.frame}")
    fid = os.path.splitext(os.path.basename(args.frame))[0]
    H, W = img.shape[:2]

    proc = E.pad_frame(img, PAD)
    tensor, nw, nh, sc = preprocess(proc)
    with torch.no_grad():
        beliefs, _ = model(tensor.to(device))
    belief = beliefs[-1][0].cpu().numpy()          # (9, bh, bw)
    n_ch, bh, bw = belief.shape

    # 검출 peak 위치(원본 좌표) + raw peak 강도
    kps = extract_keypoints_from_belief(belief, THRESH)
    peak_raw = [float(belief[i].max()) for i in range(n_ch)]
    peak_xy = []
    for i in range(n_ch):
        if kps[i][0] >= 0:
            px, py = E.belief_to_orig_pad(kps[i][0], kps[i][1], bw, bh,
                                          nw, nh, sc, PAD, W, H)
            peak_xy.append((px, py))
        else:
            peak_xy.append(None)

    # per-channel heatmap(원본 정합) 사전 계산
    heats = [belief_to_orig_map(belief[i], W, H, PAD) for i in range(n_ch)]

    # ── 1) 9채널 그리드 (고정 0~1 스케일: 강/약 직접 비교) ──
    tiles = []
    for i in range(n_ch):
        tile = overlay_heat(img, heats[i], vmax=1.0)
        # 검출 peak 표시
        if peak_xy[i] is not None:
            p = (int(peak_xy[i][0]), int(peak_xy[i][1]))
            cv2.drawMarker(tile, p, (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
        det = "DET" if peak_raw[i] >= THRESH else "weak"
        hdr = f"{CH_NAMES[i]}  peak={peak_raw[i]:.2f} [{det}]"
        cv2.rectangle(tile, (0, 0), (W, 24), (0, 0, 0), -1)
        cv2.putText(tile, hdr, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    CH_COL[i], 2, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (W - 1, H - 1), CH_COL[i], 2)
        tiles.append(tile)
        if args.per_channel:
            cv2.imwrite(os.path.join(OUT_DIR, f"{fid}_ch{i}.jpg"), tile)

    rows = [np.hstack(tiles[r * 3:r * 3 + 3]) for r in range(3)]
    grid = np.vstack(rows)
    grid_path = os.path.join(OUT_DIR, f"{fid}_9ch_grid.jpg")
    cv2.imwrite(grid_path, grid)

    # ── 2) 합성 heatmap (9채널 max) ──
    comb = np.max(np.stack(heats, 0), axis=0)
    comb_img = overlay_heat(img, comb, vmax=1.0)
    for i in range(n_ch):
        if peak_xy[i] is not None:
            p = (int(peak_xy[i][0]), int(peak_xy[i][1]))
            cv2.drawMarker(comb_img, p, (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
            tc = (0, 255, 0) if i < 4 else ((0, 165, 255) if i < 8 else (255, 0, 255))
            cv2.putText(comb_img, str(i), (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 2, cv2.LINE_AA)
    cv2.rectangle(comb_img, (0, 0), (W, 24), (0, 0, 0), -1)
    n_det = sum(1 for i in range(8) if peak_raw[i] >= THRESH)
    cv2.putText(comb_img, f"{fid}  combined  corner_det={n_det}/8",
                (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                cv2.LINE_AA)
    comb_path = os.path.join(OUT_DIR, f"{fid}_combined.jpg")
    cv2.imwrite(comb_path, comb_img)

    # ── summary.md ──
    front = np.mean([peak_raw[i] for i in range(4)])
    rear = np.mean([peak_raw[i] for i in range(4, 8)])
    lines = [
        f"# Paper-S1 belief heatmap — {fid}",
        "",
        f"- weights: `{args.weights}`",
        f"- frame: `{args.frame}`  ({W}x{H})",
        f"- belief grid: {bw}x{bh}  | preprocess: reflect-pad{PAD}",
        f"- corner det (peak>= {THRESH}): {n_det}/8",
        "",
        "## 코너별 belief peak 강도 (raw, 0~1)",
        "```",
        "ch  name       peak    status   loc(x,y)",
        "-------------------------------------------------",
    ]
    for i in range(n_ch):
        loc = (f"({peak_xy[i][0]:.0f},{peak_xy[i][1]:.0f})"
               if peak_xy[i] is not None else "-")
        st = "DET " if peak_raw[i] >= THRESH else "WEAK"
        lines.append(f"{i}   {CH_NAMES[i]:<9} {peak_raw[i]:.3f}   {st}    {loc}")
    lines += [
        "```",
        "",
        f"- front(0-3) mean peak = {front:.3f}",
        f"- rear (4-7) mean peak = {rear:.3f}",
        f"- centroid(8) peak     = {peak_raw[8]:.3f}",
        "",
        f"출력: `{grid_path}`, `{comb_path}`",
    ]
    with open(os.path.join(OUT_DIR, f"{fid}_summary.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"[done] grid={grid_path}")
    print(f"[done] combined={comb_path}")
    print(f"corner_det={n_det}/8  front={front:.3f} rear={rear:.3f} "
          f"centroid={peak_raw[8]:.3f}")
    for i in range(n_ch):
        print(f"  ch{i} {CH_NAMES[i]:<9} peak={peak_raw[i]:.3f}")


if __name__ == "__main__":
    main()
