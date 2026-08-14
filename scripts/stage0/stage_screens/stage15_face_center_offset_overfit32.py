"""STAGE15 Gate 2 — side-face-center anchor + corner-offset head 32장 overfit gate.

NVIDIA식 표현(side face center anchor heatmap + 그 anchor 위치에서 각 면의 4 corner
까지 2D offset 회귀)이 "구현/표현이 학습 가능한가"를 32장 순수 암기로 검증.
전체 학습 아님 — 표현 capacity gate.

base = B2 frozen (weights/stage11_16k_B2_maskaux/final_net_epoch_0084.pth).
strict=False 로드(mask aux head 24 unexpected 무시, belief/affinity/vgg backbone
0 missing 확인). encoder+belief+affinity 전부 freeze, 새 2 head 만 학습.
frozen feat(vgg out1, 128ch 50×50) 캐시. offset_overfit32 / v3_overfit32 패턴 재사용.

── 채널 구성 결정 근거 (Gate1 확정) ──
side face = front / left / right 3채널만. back 드롭.
  근거: convention camera_dynamic_0123_v4 가 front{0,1,2,3}을 항상 카메라-근접
  long side 로 재인덱싱 → back 면은 정의상 valid=0 (Gate1: n_valid≥3=0%, back
  heatmap_valid 8414중 0). v3 10000장 valid 분포(=확인): front 863 / left 828 /
  right 822 / front+left 172 / front+right 165 / none 150, 최대 2면. 그래서:
  face_center_head = 3ch heatmap (front/left/right), 안 보이는 면 채널은 GT 없으면
  loss mask (전부 0 으로 supervise — peak 없음 학습).
  offset_head = 3 face × 4 corner × (dx,dy) = 24ch. 각 면 anchor(face-center) 위치
  에서만 그 면 4 corner offset 을 supervise (보이는 면만).

face / corner-index 그룹 (stage15_gen_face_centers.py 와 동일 convention):
  front{0,1,2,3}  left{0,3,4,7}  right{1,2,5,6}.

좌표 (output stride=8, 50×50 map; overfit32 와 동일하게 400×400 aspect resize):
  anchor target = face_centers_2d(orig px) → input px(×sx,×sy) → output(/8).
  corner target = projected_cuboid[idx](orig px) → 동일 변환.
  offset target T(q) = (corner_out - q) / GRID  (full 2D, GRID=50 정규화).
  ★ anchor 와 corner 의 perspective 불일치(2D mean ≠ 3D mean proj, ~3px) 는
  offset = corner - anchor 로 자기일관 흡수.

face_center_head loss = MSE(pred_heatmap, gaussian target) over all 3 ch
  (보이는 면=peak gaussian, 안 보이는 면=0 map).
offset_head loss = smooth_l1, anchor 주변 support(radius 2) gaussian-weighted,
  보이는 면만 (valid mask).

판정 (Gate2 PASS = 전부 충족):
  D1 (oracle): GT face-center 셀 offset 디코드 corner median < 1 output cell
  D2 (predicted): 예측 face-center peak 디코드 corner median < 2 cells
  face-center peak localization median < 1 cell
  NaN decode = 0
undertraining(loss 미수렴)이면 FAIL 아니라 INCONCLUSIVE.

출력: data/pallet/eval_results/stage15_face_center_offset/overfit32/
★ 대규모/3k 학습 금지 — 32 overfit gate 까지만.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from models import DopeNetwork  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
INPUT = 400
GRID = 50
STRIDE = INPUT / GRID  # 8.0
RADIUS = 2             # offset support radius (output px)
GAUSS_SIGMA = 1.0      # offset support weighting sigma (output px)
HM_SIGMA = 2.0         # face-center heatmap target sigma (output px); wider than
                       # offset support so plain-MSE peak doesn't collapse to 0.
HM_POS_W = 50.0        # positive-region up-weight in heatmap MSE (counter the
                       # ~2500-cell background-zero dominance in a 50x50 map).
N_FACE = 3             # front, left, right
N_CORNER = 4

FACE_ORDER = ["front", "left", "right"]   # 3 supervised channels (back dropped)
SIDE_FACES = {"front": [0, 1, 2, 3], "left": [0, 3, 4, 7], "right": [1, 2, 5, 6]}

DEFAULT_WEIGHTS = os.path.join(
    ROOT, "weights", "stage11_16k_B2_maskaux", "final_net_epoch_0084.pth")
V3 = os.path.join(ROOT, "challenge", "data", "training", "v3")
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results",
                       "stage15_face_center_offset", "overfit32")


# ───────────────────────── heads ─────────────────────────
class FaceCenterHead(nn.Module):
    """out1(128,50,50) -> 3ch face-center heatmap (sigmoid)."""
    def __init__(self, c=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, N_FACE, 1))

    def forward(self, feat):
        return torch.sigmoid(self.net(feat))   # (B,3,H,W)


class OffsetHead(nn.Module):
    """out1(128,50,50) -> 3face*4corner*2 = 24ch linear offset field."""
    def __init__(self, c=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, N_FACE * N_CORNER * 2, 1))

    def forward(self, feat):
        return self.net(feat)   # (B,24,H,W) linear


def load_frozen_base(weights, device):
    model = DopeNetwork()
    state = torch.load(weights, map_location=device)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert len(missing) == 0, f"missing keys in B2 load: {missing[:5]}"
    print(f"[base] strict=False load: 0 missing, {len(unexpected)} unexpected "
          f"(mask aux head, ignored)")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def preprocess_fixed(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]
    r = cv2.resize(rgb, (INPUT, INPUT))
    t = ((r.astype(np.float32) / 255.0 - MEAN) / STD)
    tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, INPUT / w0, INPUT / h0


def load_sample(jp, sc_p, ip):
    """Returns dict with frozen-feature inputs + per-face anchor/corner (output px)
    + visibility mask. None if no usable visible face / off-grid."""
    o = json.load(open(jp))["objects"][0]
    sc = json.load(open(sc_p))
    gt8 = np.array(o["projected_cuboid"], float)[:8]           # orig px
    fc2d = np.array(sc["face_centers_2d"], float)             # (4,2) front,back,left,right
    status = sc["face_center_status"]
    valid_by_name = {s["face_name"]: bool(s["heatmap_valid"]) for s in status}
    fc_by_name = {s["face_name"]: np.array(s["xy"], float) for s in status}

    img = cv2.imread(ip)
    tensor, sx, sy = preprocess_fixed(img)

    def to_out(p):
        return np.array([p[0] * sx, p[1] * sy]) / STRIDE

    anchors = np.zeros((N_FACE, 2), float)          # output px
    corners = np.zeros((N_FACE, N_CORNER, 2), float)
    vis = np.zeros(N_FACE, bool)
    for fi, name in enumerate(FACE_ORDER):
        a = to_out(fc_by_name[name])
        anchors[fi] = a
        idx = SIDE_FACES[name]
        for ci in range(N_CORNER):
            corners[fi, ci] = to_out(gt8[idx[ci]])
        v = valid_by_name[name]
        # require anchor inside grid to supervise
        if v and (0 <= a[0] < GRID and 0 <= a[1] < GRID):
            vis[fi] = True
    if not vis.any():
        return None
    return dict(tensor=tensor, anchors=anchors, corners=corners, vis=vis)


def gaussian_map(cx, cy, sigma=GAUSS_SIGMA, H=GRID, W=GRID):
    ys, xs = np.mgrid[0:H, 0:W]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def build_targets(s):
    """face heatmap target (3,H,W), offset target (24,H,W), offset weight (3,H,W)."""
    H = W = GRID
    hm = np.zeros((N_FACE, H, W), np.float32)            # visible -> gaussian, else 0
    off = np.zeros((N_FACE * N_CORNER * 2, H, W), np.float32)
    ow = np.zeros((N_FACE, H, W), np.float32)            # offset support weight per face
    for fi in range(N_FACE):
        if not s["vis"][fi]:
            continue
        ax, ay = s["anchors"][fi]
        hm[fi] = gaussian_map(ax, ay, sigma=HM_SIGMA)
        ci0, cj0 = int(round(ay)), int(round(ax))
        for di in range(-RADIUS, RADIUS + 1):
            for dj in range(-RADIUS, RADIUS + 1):
                qi, qj = ci0 + di, cj0 + dj
                if not (0 <= qi < H and 0 <= qj < W):
                    continue
                qx, qy = float(qj), float(qi)
                for ci in range(N_CORNER):
                    base = (fi * N_CORNER + ci) * 2
                    off[base, qi, qj] = (s["corners"][fi, ci, 0] - qx) / GRID
                    off[base + 1, qi, qj] = (s["corners"][fi, ci, 1] - qy) / GRID
                d2 = (qx - ax) ** 2 + (qy - ay) ** 2
                ow[fi, qi, qj] = float(np.exp(-d2 / (2 * GAUSS_SIGMA ** 2)))
    return hm, off, ow


def decode_corners(off_pred, fi, ax, ay):
    """offset pred (24,H,W) numpy; decode 4 corners of face fi at cell (round ax,ay)."""
    cj, ci = int(round(ax)), int(round(ay))
    ci = min(max(ci, 0), GRID - 1)
    cj = min(max(cj, 0), GRID - 1)
    out = np.zeros((N_CORNER, 2))
    for c in range(N_CORNER):
        base = (fi * N_CORNER + c) * 2
        dx = off_pred[base, ci, cj] * GRID
        dy = off_pred[base + 1, ci, cj] * GRID
        out[c] = (cj + dx, ci + dy)
    return out


def peak_of(hm_ch):
    """argmax peak (x,y) of a heatmap channel."""
    i = int(np.argmax(hm_ch))
    return float(i % GRID), float(i // GRID)


def stratified_select(n, seed):
    """Pick n frames from v3 with diverse visible-face combos
    (front-only, left-only, right-only, front+left, front+right)."""
    rng = np.random.default_rng(seed)
    buckets = {}
    files = sorted(glob.glob(os.path.join(V3, "batch_*", "*.face_centers.json")))
    for sc_p in files:
        d = json.load(open(sc_p))
        valids = frozenset(s["face_name"] for s in d["face_center_status"]
                           if s["heatmap_valid"] and s["face_name"] in FACE_ORDER)
        if not valids:
            continue
        buckets.setdefault(valids, []).append(sc_p)
    combos = sorted(buckets.keys(), key=lambda k: (len(k), sorted(k)))
    per = max(1, n // max(len(combos), 1))
    chosen = []
    for k in combos:
        lst = buckets[k]
        rng.shuffle(lst)
        chosen.extend(lst[:per])
    rng.shuffle(chosen)
    chosen = chosen[:n]
    # top up if short
    if len(chosen) < n:
        rest = [f for f in files if f not in set(chosen)]
        rng.shuffle(rest)
        chosen.extend(rest[:n - len(chosen)])
    dist = {}
    for sc_p in chosen:
        d = json.load(open(sc_p))
        v = frozenset(s["face_name"] for s in d["face_center_status"]
                      if s["heatmap_valid"] and s["face_name"] in FACE_ORDER)
        dist[tuple(sorted(v))] = dist.get(tuple(sorted(v)), 0) + 1
    return chosen, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_hm", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = load_frozen_base(args.weights, device)
    fc_head = FaceCenterHead(128).to(device)
    off_head = OffsetHead(128).to(device)
    opt = torch.optim.Adam(list(fc_head.parameters()) + list(off_head.parameters()),
                           lr=args.lr)
    sl1 = nn.SmoothL1Loss(reduction="none")

    sel, dist = stratified_select(args.n, args.seed)
    print(f"[data] selected {len(sel)} frames; visible-face combo dist: {dist}")

    samples = []
    for sc_p in sel:
        jp = sc_p[:-len(".face_centers.json")] + ".json"
        ip = sc_p[:-len(".face_centers.json")] + ".png"
        if not (os.path.exists(jp) and os.path.exists(ip)):
            continue
        s = load_sample(jp, sc_p, ip)
        if s is None:
            continue
        hm, off, ow = build_targets(s)
        s["fid"] = os.path.basename(jp)[:-5]
        s["img_path"] = ip
        s["hm_t"] = torch.from_numpy(hm).unsqueeze(0).to(device)
        s["off_t"] = torch.from_numpy(off).unsqueeze(0).to(device)
        s["ow_t"] = torch.from_numpy(ow).to(device)   # (3,H,W)
        s["tensor"] = s["tensor"].to(device)
        samples.append(s)
    print(f"[data] usable {len(samples)} samples")

    # ── sanity: sidecar target -> decode round-trip (before training) ──
    s0 = samples[0]
    rt_errs = []
    for fi in range(N_FACE):
        if not s0["vis"][fi]:
            continue
        ax, ay = s0["anchors"][fi]
        gt_off = s0["off_t"][0].cpu().numpy()
        dec = decode_corners(gt_off, fi, ax, ay)
        e = np.linalg.norm(dec - s0["corners"][fi], axis=1)
        rt_errs.extend(e.tolist())
    rt_med = float(np.median(rt_errs)) if rt_errs else None
    print(f"[sanity] target round-trip corner err median={rt_med:.5f} cells "
          f"(should be ~0; coord/channel bug check)")
    assert rt_med is not None and rt_med < 0.05, "round-trip target bug!"

    # cache frozen features
    with torch.no_grad():
        for s in samples:
            s["feat"] = base.vgg(s["tensor"])

    log = []
    for it in range(1, args.iters + 1):
        fc_head.train(); off_head.train()
        opt.zero_grad()
        tot_fc = tot_off = 0.0
        for s in samples:
            feat = s["feat"]
            hm_p = fc_head(feat)               # (1,3,H,W)
            off_p = off_head(feat)             # (1,24,H,W)
            # positive-weighted MSE: emphasise gaussian region so the sparse peak
            # is not washed out by ~2500 background-zero cells (collapse fix).
            w_hm = 1.0 + HM_POS_W * s["hm_t"]
            l_fc = ((hm_p - s["hm_t"]) ** 2 * w_hm).sum() / w_hm.sum()
            # offset: per-face support-weighted, visible only
            ow = s["ow_t"]                     # (3,H,W)
            w24 = ow.repeat_interleave(N_CORNER * 2, dim=0).unsqueeze(0)  # (1,24,H,W)
            lm = sl1(off_p, s["off_t"])
            num = (lm * w24).sum()
            den = w24.sum() + 1e-6
            l_off = num / den
            loss = args.lambda_hm * l_fc + l_off
            loss.backward()
            tot_fc += float(l_fc); tot_off += float(l_off)
        opt.step()
        if it % 250 == 0 or it == 1:
            lf, lo = tot_fc / len(samples), tot_off / len(samples)
            print(f"[it {it}] face_mse={lf:.6f} offset_sl1={lo:.6f}")
            log.append({"iter": it, "face_mse": lf, "offset_sl1": lo})

    # ── eval ──
    fc_head.eval(); off_head.eval()
    d1_errs, d2_errs, peak_errs = [], [], []   # corner cells / corner cells / anchor cells
    nan_count = 0
    per_frame = []
    overlay_pool = []
    with torch.no_grad():
        for s in samples:
            feat = s["feat"]
            hm_p = fc_head(feat)[0].cpu().numpy()    # (3,H,W)
            off_p = off_head(feat)[0].cpu().numpy()  # (24,H,W)
            fr = {"fid": s["fid"], "faces": []}
            for fi in range(N_FACE):
                if not s["vis"][fi]:
                    continue
                ax, ay = s["anchors"][fi]
                # D1 oracle: decode at GT anchor cell
                dec_o = decode_corners(off_p, fi, ax, ay)
                e_o = np.linalg.norm(dec_o - s["corners"][fi], axis=1)
                # peak localization
                px, py = peak_of(hm_p[fi])
                pe = float(np.hypot(px - ax, py - ay))
                # D2 predicted: decode at predicted peak cell
                dec_p = decode_corners(off_p, fi, px, py)
                e_p = np.linalg.norm(dec_p - s["corners"][fi], axis=1)
                if not (np.isfinite(dec_o).all() and np.isfinite(dec_p).all()):
                    nan_count += 1
                d1_errs.extend(e_o.tolist())
                d2_errs.extend(e_p.tolist())
                peak_errs.append(pe)
                fr["faces"].append(dict(
                    face=FACE_ORDER[fi], d1_med=float(np.median(e_o)),
                    d2_med=float(np.median(e_p)), peak_err=pe))
            per_frame.append(fr)
            overlay_pool.append((s, hm_p, off_p))

    def stat(a):
        a = np.array(a)
        return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                    max=float(a.max()), n=int(a.size)) if a.size else None

    d1, d2, pk = stat(d1_errs), stat(d2_errs), stat(peak_errs)
    # converged = loss curve has flattened (last quarter rel-change small), i.e. more
    # iters won't help. Used only to split FAIL vs INCONCLUSIVE(undertraining).
    tail = log[max(0, len(log) - 4):]
    fc_tail = [t["face_mse"] for t in tail]
    off_tail = [t["offset_sl1"] for t in tail]
    fc_flat = (max(fc_tail) - min(fc_tail)) / (max(fc_tail) + 1e-9) < 0.05
    off_flat = log[-1]["offset_sl1"] < 1e-3
    converged = bool(fc_flat and off_flat)
    pass_d1 = d1["median"] < 1.0
    pass_d2 = d2["median"] < 2.0
    pass_peak = pk["median"] < 1.0
    pass_nan = nan_count == 0
    all_pass = pass_d1 and pass_d2 and pass_peak and pass_nan
    if all_pass:
        verdict = "PASS"
    elif not converged:
        verdict = "INCONCLUSIVE (undertraining: loss not converged)"
    else:
        verdict = "FAIL"

    result = dict(
        n_samples=len(samples), iters=args.iters, lr=args.lr,
        lambda_hm=args.lambda_hm, combo_dist={str(k): v for k, v in dist.items()},
        roundtrip_med_cell=rt_med,
        D1_oracle_corner=d1, D2_predicted_corner=d2, peak_localization=pk,
        nan_decode=nan_count,
        converged=bool(converged),
        loss_final={"face_mse": log[-1]["face_mse"], "offset_sl1": log[-1]["offset_sl1"]},
        pass_D1_lt1=bool(pass_d1), pass_D2_lt2=bool(pass_d2),
        pass_peak_lt1=bool(pass_peak), pass_nan0=bool(pass_nan),
        verdict=verdict, loss_log=log, per_frame=per_frame)
    json.dump(result, open(os.path.join(OUT_DIR, "overfit32.json"), "w"), indent=2)

    # ── sanity overlays (>=3): peak + decoded corners ──
    n_ov = 0
    FACE_COL = {0: (0, 0, 255), 1: (255, 200, 0), 2: (0, 210, 90)}  # BGR front/left/right
    for s, hm_p, off_p in overlay_pool:
        if n_ov >= 5:
            break
        img = cv2.resize(cv2.imread(s["img_path"]), (INPUT, INPUT))
        for fi in range(N_FACE):
            if not s["vis"][fi]:
                continue
            col = FACE_COL[fi]
            ax, ay = s["anchors"][fi]
            px, py = peak_of(hm_p[fi])
            # GT anchor (circle), predicted peak (cross) in input px
            cv2.circle(img, (int(ax * STRIDE), int(ay * STRIDE)), 6, col, 2)
            cv2.drawMarker(img, (int(px * STRIDE), int(py * STRIDE)), col,
                           cv2.MARKER_TILTED_CROSS, 14, 2)
            dec = decode_corners(off_p, fi, px, py)        # predicted-peak decode
            gt = s["corners"][fi]
            for c in range(N_CORNER):
                cv2.circle(img, (int(dec[c, 0] * STRIDE), int(dec[c, 1] * STRIDE)),
                           4, col, -1)                      # decoded corner (filled)
                cv2.circle(img, (int(gt[c, 0] * STRIDE), int(gt[c, 1] * STRIDE)),
                           5, (255, 255, 255), 1)           # GT corner (white ring)
        cv2.imwrite(os.path.join(OUT_DIR, f"sanity_overlay_{n_ov:02d}_{s['fid']}.png"), img)
        n_ov += 1

    # ── console summary ──
    print("\n=== STAGE15 Gate 2 (overfit32) ===")
    print(f"samples={len(samples)} iters={args.iters} lr={args.lr}")
    print(f"loss final: face_mse={log[-1]['face_mse']:.6f} "
          f"offset_sl1={log[-1]['offset_sl1']:.6f}  converged={converged}")
    print(f"D1 oracle  corner cells: med={d1['median']:.3f} p95={d1['p95']:.3f} "
          f"max={d1['max']:.3f}  (<1 ? {pass_d1})")
    print(f"D2 pred    corner cells: med={d2['median']:.3f} p95={d2['p95']:.3f} "
          f"max={d2['max']:.3f}  (<2 ? {pass_d2})")
    print(f"peak loc   anchor cells: med={pk['median']:.3f} p95={pk['p95']:.3f} "
          f"max={pk['max']:.3f}  (<1 ? {pass_peak})")
    print(f"NaN decode: {nan_count}  (=0 ? {pass_nan})")
    print(f"VERDICT: {verdict}")
    print(f"[save] {OUT_DIR}/overfit32.json + {n_ov} overlays")


if __name__ == "__main__":
    main()
