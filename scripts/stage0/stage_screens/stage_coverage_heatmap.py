"""stage_coverage_heatmap.py — Stage B belief(9ch) + mask-aux head 진단 시각화.

목적: 사용자가 고른 5 프레임("파렛트 일부만 cuboid로 잡는" scale/coverage 실패 의심)에서
      Stage B(net_epoch_0057, mask_aux numSeg=1)의 belief 9채널 히트맵과 mask 헤드(seg)
      출력을 뽑아, "일부만 잡음"을 거를 신호가 (a) belief 2차peak (b) mask vs keypoint
      커버리지 차이로 보이는지 관찰한다. (관찰 전용, 과결론 금지)

재사용: squash-parity 전처리(preprocess_squash, paper_s2_real_eval 와 동일),
        extract_keypoints_from_belief, s1_heatmap_vis 의 overlay_heat.
Stage B forward = numSeg=1 -> (beliefs, affinities, None, [seg1, seg2]).
belief/seg 격자 50x50 -> 원본 (W/50, H/50) 직접 매핑(squash).

출력: data/pallet/eval_results/paper_s2_scratch_diffpnp/coverage_heatmap/
  <domain>_<fid>_belief.jpg    9채널 belief 그리드(peak/2nd-peak/f2) + combined argmax
  <domain>_<fid>_mask.jpg      mask 헤드 sigmoid 히트맵 + 원본 오버레이
  <domain>_<fid>_coverage.jpg  원본 + keypoint cuboid(red) + mask 영역(반투명) 겹침
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os
import sys

import cv2
import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

from models import DopeNetwork  # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402

WEIGHTS = os.path.join(ROOT, "weights/paper_s2_stageB/net_epoch_0057.pth")
OUT_DIR = os.path.join(
    ROOT, "data/pallet/eval_results/paper_s2_scratch_diffpnp/coverage_heatmap")
THRESH = 0.3
MASK_THRESH = 0.5
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# 5 target frames (fid, domain) — 사용자 지정
TARGETS = [
    ("1775201432466607872", "noapril", "FAIL(f3; W/D swap honest8=39.2)"),
    ("1779449194023912448", "night", "pass"),
    ("1779449196392532480", "night", "pass"),
    ("1779449266426633216", "night", "pass"),
    ("1778651651444080384", "outside", "pass"),
]

CH_NAMES = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "ctr8"]
CH_COL = [(0, 255, 0)] * 4 + [(0, 165, 255)] * 4 + [(255, 0, 255)]
# camera-facing 0123: 0-3 front, 4-7 rear; {0,1,4,5}=top {2,3,6,7}=bottom.
# front quad 0-1-3-2, rear quad 4-5-7-6, connect 0-4 1-5 2-6 3-7.
CUBOID_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0),
                (4, 5), (5, 7), (7, 6), (6, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)]


def load_model_seg(wp, device):
    """numSeg=1 로 로드해 seg 헤드까지 살린다(추론시 원설계는 미사용, 진단용 추출)."""
    state = torch.load(wp, map_location=device)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    m = DopeNetwork(numVec=0, numSeg=1)
    missing, unexpected = m.load_state_dict(state, strict=False)
    seg_missing = [k for k in missing if "seg" in k.lower()]
    assert not seg_missing, f"seg 헤드 로드 실패: {seg_missing[:3]}"
    return m.to(device).eval()


def preprocess_squash(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (400, 400), interpolation=cv2.INTER_LINEAR)
    t = (r.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0)


def overlay_heat(img, heat, vmax=1.0, alpha=0.75):
    h = np.clip(heat / vmax, 0.0, 1.0)
    color = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    a = (h * alpha)[..., None]
    return (img * (1 - a) + color * a).astype(np.uint8)


def grid_to_orig(ch, W, H):
    """50x50 belief/seg 격자 -> 원본(H,W) squash 역매핑(등방 확대)."""
    return cv2.resize(ch, (W, H), interpolation=cv2.INTER_LINEAR)


def second_peak_ratio(ch, sup=3):
    """1차peak 억제(±sup 셀) 후 2차peak. f2 = 2nd/1st (다봉/모호도 프록시)."""
    g = ch.copy()
    p1 = float(g.max())
    if p1 <= 0:
        return 0.0, 0.0, 0.0
    yy, xx = np.unravel_index(int(np.argmax(g)), g.shape)
    y0, y1 = max(0, yy - sup), min(g.shape[0], yy + sup + 1)
    x0, x1 = max(0, xx - sup), min(g.shape[1], xx + sup + 1)
    g[y0:y1, x0:x1] = 0.0
    p2 = float(g.max())
    return p1, p2, p2 / p1


def find_fid_path(fid, domain):
    """fid 로 rgb png 를 찾는다(raw_data/<domain>/*/rgb/, _eval_sets, manual_gt)."""
    import glob
    cands = []
    cands += glob.glob(os.path.join(
        ROOT, "data/pallet/raw_data", domain, "*", "rgb", fid + ".png"))
    cands += glob.glob(os.path.join(ROOT, "data/_eval_sets", "*", fid + ".png"))
    cands += glob.glob(os.path.join(
        ROOT, "data/pallet/raw_data", "**", "rgb", fid + ".png"),
        recursive=True)
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_seg(WEIGHTS, device)
    print(f"[model] loaded seg-head, device={device}")

    report = []
    for fid, domain, tag in TARGETS:
        ip = find_fid_path(fid, domain)
        if ip is None:
            print(f"[MISS] {domain} {fid}: rgb not found")
            report.append((domain, fid, tag, None))
            continue
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        sx, sy = W / 50.0, H / 50.0

        with torch.no_grad():
            out = model(preprocess_squash(img).to(device))
        beliefs, _aff, _vec, seg = out[0], out[1], out[2], out[3]
        belief = beliefs[-1][0].cpu().numpy()          # (9, 50, 50)
        seg_logit = seg[-1][0, 0].cpu().numpy()        # (50, 50)
        seg_prob = 1.0 / (1.0 + np.exp(-seg_logit))    # sigmoid

        # keypoints (belief argmax -> orig px)
        kps = extract_keypoints_from_belief(belief, THRESH)
        pred = np.full((9, 2), np.nan)
        for i in range(9):
            if kps[i][0] >= 0:
                pred[i] = [kps[i][0] * sx, kps[i][1] * sy]

        peak, p2r, f2 = [], [], []
        for i in range(9):
            a, b, r = second_peak_ratio(belief[i])
            peak.append(a)
            p2r.append(b)
            f2.append(r)
        peak = np.array(peak)
        f2 = np.array(f2)

        # ---- 1) belief 9ch grid ----
        heats = [grid_to_orig(belief[i], W, H) for i in range(9)]
        tiles = []
        for i in range(9):
            t = overlay_heat(img, heats[i], vmax=1.0)
            if not np.isnan(pred[i, 0]):
                cv2.drawMarker(t, (int(pred[i, 0]), int(pred[i, 1])),
                               (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
            det = "DET" if peak[i] >= THRESH else "weak"
            cv2.rectangle(t, (0, 0), (W, 24), (0, 0, 0), -1)
            cv2.putText(t, f"{CH_NAMES[i]} pk={peak[i]:.2f} f2={f2[i]:.2f}[{det}]",
                        (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CH_COL[i], 2,
                        cv2.LINE_AA)
            cv2.rectangle(t, (0, 0), (W - 1, H - 1), CH_COL[i], 2)
            tiles.append(t)
        grid = np.vstack([np.hstack(tiles[r * 3:r * 3 + 3]) for r in range(3)])
        cv2.imwrite(os.path.join(OUT_DIR, f"{domain}_{fid}_belief.jpg"), grid)

        # ---- 2) mask head ----
        seg_orig = grid_to_orig(seg_prob.astype(np.float32), W, H)
        mimg = overlay_heat(img, seg_orig, vmax=1.0, alpha=0.7)
        n_det = int(np.sum(peak[:8] >= THRESH))
        cv2.rectangle(mimg, (0, 0), (W, 24), (0, 0, 0), -1)
        cv2.putText(mimg, f"{domain} {fid} mask-aux(seg sigmoid) "
                    f"max={seg_prob.max():.2f} det={n_det}/8",
                    (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT_DIR, f"{domain}_{fid}_mask.jpg"), mimg)

        # ---- 3) coverage: keypoint cuboid vs mask area ----
        cov = img.copy()
        mask_bin = (seg_orig >= MASK_THRESH).astype(np.uint8)
        mask_area = int(mask_bin.sum())
        # mask 반투명 파랑
        blue = np.zeros_like(cov)
        blue[..., 0] = 255
        m3 = mask_bin[..., None].astype(bool)
        cov = np.where(m3, (cov * 0.55 + blue * 0.45).astype(np.uint8), cov)
        # keypoint cuboid (red)
        for a, b in CUBOID_EDGES:
            if not (np.isnan(pred[a, 0]) or np.isnan(pred[b, 0])):
                cv2.line(cov, (int(pred[a, 0]), int(pred[a, 1])),
                         (int(pred[b, 0]), int(pred[b, 1])), (0, 0, 255), 2,
                         cv2.LINE_AA)
        for i in range(8):
            if not np.isnan(pred[i, 0]):
                cv2.circle(cov, (int(pred[i, 0]), int(pred[i, 1])), 4,
                           (0, 0, 255), -1)
                cv2.putText(cov, str(i), (int(pred[i, 0]) + 4, int(pred[i, 1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                            cv2.LINE_AA)
        # keypoint bbox
        det_pts = pred[:8][~np.isnan(pred[:8, 0])]
        kp_bbox_area = 0
        if len(det_pts) >= 2:
            x0, y0 = det_pts[:, 0].min(), det_pts[:, 1].min()
            x1, y1 = det_pts[:, 0].max(), det_pts[:, 1].max()
            kp_bbox_area = int(max(0, x1 - x0) * max(0, y1 - y0))
            cv2.rectangle(cov, (int(x0), int(y0)), (int(x1), int(y1)),
                          (0, 255, 0), 1)
        ratio = (mask_area / kp_bbox_area) if kp_bbox_area > 0 else float("nan")
        cv2.rectangle(cov, (0, 0), (W, 24), (0, 0, 0), -1)
        cv2.putText(cov, f"{domain} {fid} mask/kpbbox={ratio:.2f} [{tag}]",
                    (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT_DIR, f"{domain}_{fid}_coverage.jpg"), cov)

        stat = {
            "domain": domain, "fid": fid, "tag": tag,
            "n_det": n_det,
            "peak8_min": float(peak[:8].min()),
            "peak8_mean": float(peak[:8].mean()),
            "rear_peak_mean": float(peak[4:8].mean()),
            "front_peak_mean": float(peak[:4].mean()),
            "f2_max8": float(f2[:8].max()),
            "f2_mean8": float(f2[:8].mean()),
            "seg_max": float(seg_prob.max()),
            "mask_area": mask_area,
            "kp_bbox_area": kp_bbox_area,
            "mask_over_kpbbox": ratio,
        }
        report.append((domain, fid, tag, stat))
        print(f"[done] {domain} {fid} n_det={n_det}/8 "
              f"peak8(min/mean)={stat['peak8_min']:.2f}/{stat['peak8_mean']:.2f} "
              f"rear={stat['rear_peak_mean']:.2f} f2max={stat['f2_max8']:.2f} "
              f"mask/kpbbox={ratio:.2f}")

    # ---- numeric report ----
    lines = ["# Stage B coverage/belief/mask 진단 — 5 frames", "",
             f"weights: {WEIGHTS}", f"out: {OUT_DIR}", "",
             "```",
             "domain   fid                  tag        nd  pk8min pk8avg rearpk frontpk f2max f2avg segmax maskA/kpbbox",
             "---------------------------------------------------------------------------------------------------------"]
    for domain, fid, tag, s in report:
        if s is None:
            lines.append(f"{domain:<8} {fid:<20} {tag:<10} MISSING")
            continue
        lines.append(
            f"{s['domain']:<8} {s['fid']:<20} {tag[:10]:<10} "
            f"{s['n_det']}/8 {s['peak8_min']:.2f}   {s['peak8_mean']:.2f}   "
            f"{s['rear_peak_mean']:.2f}   {s['front_peak_mean']:.2f}    "
            f"{s['f2_max8']:.2f}  {s['f2_mean8']:.2f}  {s['seg_max']:.2f}   "
            f"{s['mask_over_kpbbox']:.2f}")
    lines += ["```", ""]
    with open(os.path.join(OUT_DIR, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
