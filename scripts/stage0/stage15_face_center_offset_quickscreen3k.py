"""STAGE15 Gate 3 — face-center anchor + corner-offset head 3k quick screen.

Gate2(32 overfit) PASS = 표현 capacity 확인. Gate3 = 진짜 질문:
  face-center+offset 표현이 현실적 데이터 예산(3k)으로 학습돼 held-out 에
  일반화하나, 특히 heatmap 이 구조적으로 못 하는 V<8(truncation) 코너 복원에서
  이득이 있나.

★핵심 가설: DOPE heatmap(CreateBeliefMap)은 화면밖/가장자리 코너 gaussian 을
  suppress → V<8 에서 코너 자체를 못 낸다. face-center anchor+offset 은 "보이는
  면 중심"에서 offset 회귀로 화면밖 코너까지 복원 가능 → V<8 구조적 우위 기대.
  Gate3 PASS/FAIL 은 주로 held-out V<8 offset-path corner 정확도가 B2 heatmap-
  path 를 의미있게 이기는가로 결정.

설계 (Gate2 의 head/decode/loss/target 재사용):
  base = B2 frozen (stage11_16k_B2_maskaux/final_net_epoch_0084.pth).
    belief/affinity/vgg backbone + heatmap head 전부 freeze (heatmap-path = B2 그대로
    비교 기준). 새 face_center_head(3ch) + offset_head(24ch) 만 학습.
  train = v3(batch_000~008) + addon_v1 에서 stratified 3k (V<8 비중 확보).
  held-out = v3 batch_009 (학습에 안 씀, 누수 없음).
  seen-subset = train 에서 떼낸 200 (anti-undertraining 게이트).
  비교: 동일 프레임에서 offset-path(face-center peak decode) vs heatmap-path
    (B2 belief -> DOPE extract_keypoints). order-free Hungarian, V=8/V<8 분리.

판정:
  (anti-undertraining, 먼저) seen-subset offset-path corner median 이 학습됨
    (~<10px 급). seen 서도 큰 오차면 INCONCLUSIVE (FAIL 아님).
  (주 판정) held-out V<8 offset-path corner median 이 B2 heatmap-path 보다
    의미있게 낮거나, heatmap 이 못 낸 화면밖 코너를 offset 이 정확히 복원하면 이득.
  V=8 offset-path 가 heatmap 대비 catastrophic 하게 나쁘지 않을 것(sane guard).
  PASS = (학습됨) AND (held-out V<8 offset 이득). FAIL/lean-neg = 학습됐는데 이득無.

출력: data/pallet/eval_results/stage15_face_center_offset/quickscreen3k/
★ synth held-out 신호일 뿐 real 보장 아님 — 과결론 금지.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from models import DopeNetwork  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

# ── reuse Gate2 head/decode/target/loss constants verbatim ──
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
INPUT = 400
GRID = 50
STRIDE = INPUT / GRID  # 8.0
RADIUS = 2
GAUSS_SIGMA = 1.0
HM_SIGMA = 2.0
HM_POS_W = 50.0
N_FACE = 3
N_CORNER = 4

FACE_ORDER = ["front", "left", "right"]
SIDE_FACES = {"front": [0, 1, 2, 3], "left": [0, 3, 4, 7], "right": [1, 2, 5, 6]}

DEFAULT_WEIGHTS = os.path.join(
    ROOT, "weights", "stage11_16k_B2_maskaux", "final_net_epoch_0084.pth")
V3 = os.path.join(ROOT, "challenge", "data", "training", "v3")
ADDON = os.path.join(ROOT, "challenge", "data", "training", "addon_v1")
HELDOUT_BATCH = os.path.join(V3, "batch_009")   # never in train
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results",
                       "stage15_face_center_offset", "quickscreen3k")


# ───────────────────────── heads (verbatim from Gate2) ─────────────────────────
class FaceCenterHead(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, N_FACE, 1))

    def forward(self, feat):
        return torch.sigmoid(self.net(feat))


class OffsetHead(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, N_FACE * N_CORNER * 2, 1))

    def forward(self, feat):
        return self.net(feat)


def load_frozen_base(weights, device):
    model = DopeNetwork()
    state = torch.load(weights, map_location=device)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert len(missing) == 0, f"missing keys in B2 load: {missing[:5]}"
    print(f"[base] strict=False load: 0 missing, {len(unexpected)} unexpected")
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
    """Per-face anchor/corner (output px) + visibility + GT8 (output px) + per-corner
    in-frame flags + V (num_corners_in_frame). None if no usable visible face."""
    o = json.load(open(jp))["objects"][0]
    sc = json.load(open(sc_p))
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    status = sc["face_center_status"]
    valid_by_name = {s["face_name"]: bool(s["heatmap_valid"]) for s in status}
    fc_by_name = {s["face_name"]: np.array(s["xy"], float) for s in status}

    img = cv2.imread(ip)
    if img is None:
        return None
    tensor, sx, sy = preprocess_fixed(img)

    def to_out(p):
        return np.array([p[0] * sx, p[1] * sy]) / STRIDE

    gt8_out = np.array([to_out(p) for p in gt8])     # (8,2) output px
    kp_in = o.get("keypoint_in_frame", [True] * 9)[:8]
    V = int(o.get("num_corners_in_frame", sum(bool(x) for x in kp_in)))

    anchors = np.zeros((N_FACE, 2), float)
    corners = np.zeros((N_FACE, N_CORNER, 2), float)
    vis = np.zeros(N_FACE, bool)
    for fi, name in enumerate(FACE_ORDER):
        a = to_out(fc_by_name[name])
        anchors[fi] = a
        idx = SIDE_FACES[name]
        for ci in range(N_CORNER):
            corners[fi, ci] = gt8_out[idx[ci]]
        if valid_by_name[name] and (0 <= a[0] < GRID and 0 <= a[1] < GRID):
            vis[fi] = True
    if not vis.any():
        return None
    return dict(tensor=tensor, anchors=anchors, corners=corners, vis=vis,
                gt8_out=gt8_out, kp_in=[bool(x) for x in kp_in], V=V,
                sx=sx, sy=sy)


def gaussian_map(cx, cy, sigma=GAUSS_SIGMA, H=GRID, W=GRID):
    ys, xs = np.mgrid[0:H, 0:W]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def build_targets(s):
    H = W = GRID
    hm = np.zeros((N_FACE, H, W), np.float32)
    off = np.zeros((N_FACE * N_CORNER * 2, H, W), np.float32)
    ow = np.zeros((N_FACE, H, W), np.float32)
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
    i = int(np.argmax(hm_ch))
    return float(i % GRID), float(i // GRID)


# ───────────────── heatmap-path baseline (B2 belief -> DOPE peaks) ─────────────
def extract_belief_peaks(belief_maps, threshold=0.3):
    """DOPE-style sub-pixel peak per channel. belief_maps (9,H,W) numpy.
    Returns list[9] of (x,y) output px or None (= can't produce that corner)."""
    OFFSET = 0.4395
    RAN = 5
    kps = []
    for i in range(belief_maps.shape[0]):
        bmap = belief_maps[i]
        if bmap.max() < threshold:
            kps.append(None)
            continue
        sm = gaussian_filter(bmap, sigma=2)
        p = 1
        pl = np.zeros_like(sm); pl[p:, :] = sm[:-p, :]
        pr = np.zeros_like(sm); pr[:-p, :] = sm[p:, :]
        pu = np.zeros_like(sm); pu[:, p:] = sm[:, :-p]
        pd = np.zeros_like(sm); pd[:, :-p] = sm[:, p:]
        peaks = ((sm >= pl) & (sm >= pr) & (sm >= pu) & (sm >= pd) & (sm > threshold))
        pys, pxs = np.nonzero(peaks)
        if len(pxs) == 0:
            kps.append(None)
            continue
        vals = [bmap[py, px] for py, px in zip(pys, pxs)]
        bi = int(np.argmax(vals))
        px, py = int(pxs[bi]), int(pys[bi])
        y0, y1 = max(0, py - RAN), min(bmap.shape[0], py + RAN + 1)
        x0, x1 = max(0, px - RAN), min(bmap.shape[1], px + RAN + 1)
        patch = bmap[y0:y1, x0:x1]
        if patch.sum() > 0:
            xg, yg = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            wx = float(np.average(xg, weights=patch) + OFFSET)
            wy = float(np.average(yg, weights=patch) + OFFSET)
        else:
            wx, wy = float(px), float(py)
        kps.append((wx, wy))
    return kps[:8]   # 8 corners (drop centroid)


def hungarian_match(pred, gt):
    """pred (M,2), gt (N,2) numpy. Returns matched err array over min(M,N) pairs,
    using order-free Hungarian on euclidean cost."""
    if len(pred) == 0 or len(gt) == 0:
        return np.array([])
    C = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    ri, cj = linear_sum_assignment(C)
    return C[ri, cj]


# ───────────────────────── data selection ─────────────────────────
def collect_pool(dirs):
    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, "batch_*", "*.face_centers.json"))))
        files.extend(sorted(glob.glob(os.path.join(d, "*.face_centers.json"))))
    return files


def frame_meta(sc_p):
    """visible-face combo + V<8 flag, for stratification. Cheap (json only)."""
    jp = sc_p[:-len(".face_centers.json")] + ".json"
    try:
        d = json.load(open(sc_p))
        valids = frozenset(s["face_name"] for s in d["face_center_status"]
                           if s["heatmap_valid"] and s["face_name"] in FACE_ORDER)
        if not valids:
            return None
        o = json.load(open(jp))["objects"][0]
        V = int(o.get("num_corners_in_frame", 8))
        return valids, V
    except Exception:
        return None


def stratified_train(n, seed):
    """3k from v3 batch_000..008 + addon (NOT batch_009). Stratify by
    (visible-face combo, V<8?) so V<8 well represented."""
    rng = np.random.default_rng(seed)
    v3_files = sorted(glob.glob(os.path.join(V3, "batch_00[0-8]", "*.face_centers.json")))
    addon_files = sorted(glob.glob(os.path.join(ADDON, "*.face_centers.json")))
    pool = v3_files + addon_files
    rng.shuffle(pool)
    buckets = {}
    for sc_p in pool:
        m = frame_meta(sc_p)
        if m is None:
            continue
        valids, V = m
        key = (valids, V < 8)
        buckets.setdefault(key, []).append(sc_p)
        # early stop once buckets big enough to fill 3k comfortably
        if sum(len(v) for v in buckets.values()) > n * 4:
            break
    keys = list(buckets.keys())
    # target: >=40% V<8 (hypothesis core). round-robin across buckets, V<8 first.
    v_lt8 = [k for k in keys if k[1]]
    v_eq8 = [k for k in keys if not k[1]]
    chosen = []
    target_lt8 = int(n * 0.45)
    chosen += _round_robin(buckets, v_lt8, target_lt8, rng)
    chosen += _round_robin(buckets, v_eq8, n - len(chosen), rng)
    rng.shuffle(chosen)
    return chosen[:n]


def _round_robin(buckets, keys, target, rng):
    pools = {k: list(buckets[k]) for k in keys}
    for k in pools:
        rng.shuffle(pools[k])
    out = []
    while len(out) < target and any(pools.values()):
        for k in keys:
            if pools[k]:
                out.append(pools[k].pop())
                if len(out) >= target:
                    break
    return out


def heldout_frames(n, seed):
    rng = np.random.default_rng(seed + 1)
    files = sorted(glob.glob(os.path.join(HELDOUT_BATCH, "*.face_centers.json")))
    rng.shuffle(files)
    return files[:n]


# ───────────────────────── train + eval ─────────────────────────
def materialize(sel, device, want_belief, base, limit=None):
    """Build sample dicts; cache frozen vgg feat (+ optional B2 belief peaks)."""
    out = []
    for sc_p in sel:
        jp = sc_p[:-len(".face_centers.json")] + ".json"
        ip = sc_p[:-len(".face_centers.json")] + ".png"
        if not (os.path.exists(jp) and os.path.exists(ip)):
            continue
        s = load_sample(jp, sc_p, ip)
        if s is None:
            continue
        hm, off, ow = build_targets(s)
        s["fid"] = os.path.relpath(jp, ROOT)
        s["img_path"] = ip
        s["hm_t"] = torch.from_numpy(hm).unsqueeze(0)
        s["off_t"] = torch.from_numpy(off).unsqueeze(0)
        s["ow_t"] = torch.from_numpy(ow)
        with torch.no_grad():
            t = s["tensor"].to(device)
            s["feat"] = base.vgg(t).cpu()
            if want_belief:
                bel, _ = base(t)
                s["belief"] = bel[-1][0].cpu().numpy()   # (9,50,50)
        del s["tensor"]
        out.append(s)
        if limit and len(out) >= limit:
            break
    return out


def run_eval(samples, fc_head, off_head, device, tag):
    """Per frame: offset-path corners (predicted peak decode, per visible face),
    heatmap-path corners (B2 belief peaks). order-free Hungarian vs GT8.
    Split V=8 / V<8. Also track per-corner off-screen recovery."""
    fc_head.eval(); off_head.eval()
    res = {"V8": {"offset": [], "heatmap": []},
           "Vlt8": {"offset": [], "heatmap": []}}
    # off-screen corner recovery: corners with kp_in==False
    offscreen = {"heatmap_na": 0, "offset_err": [], "n_offscreen": 0, "frames": set()}
    per_frame = []
    with torch.no_grad():
        for s in samples:
            feat = s["feat"].to(device)
            hm_p = fc_head(feat)[0].cpu().numpy()
            off_p = off_head(feat)[0].cpu().numpy()
            # offset-path: union of decoded corners over visible faces (predicted peak)
            off_pts = []
            for fi in range(N_FACE):
                if not s["vis"][fi]:
                    continue
                px, py = peak_of(hm_p[fi])
                dec = decode_corners(off_p, fi, px, py)
                off_pts.extend(dec.tolist())
            off_pts = np.array(off_pts) if off_pts else np.zeros((0, 2))
            # heatmap-path: B2 belief peaks (None = not produced)
            hb = extract_belief_peaks(s["belief"])
            hm_pts = np.array([p for p in hb if p is not None]) if any(
                p is not None for p in hb) else np.zeros((0, 2))

            gt8 = s["gt8_out"]
            bucket = "Vlt8" if s["V"] < 8 else "V8"
            e_off = hungarian_match(off_pts, gt8)
            e_hm = hungarian_match(hm_pts, gt8)
            res[bucket]["offset"].extend(e_off.tolist())
            res[bucket]["heatmap"].extend(e_hm.tolist())

            # off-screen corner recovery (per corner with kp_in False)
            fr = {"fid": s["fid"], "V": s["V"],
                  "off_med": float(np.median(e_off)) if len(e_off) else None,
                  "hm_med": float(np.median(e_hm)) if len(e_hm) else None}
            for ci in range(8):
                if s["kp_in"][ci]:
                    continue
                offscreen["n_offscreen"] += 1
                offscreen["frames"].add(s["fid"])
                gt_c = gt8[ci]
                # heatmap NA for this corner?
                if hb[ci] is None:
                    offscreen["heatmap_na"] += 1
                # nearest offset-path point to this GT corner
                if len(off_pts):
                    d = np.linalg.norm(off_pts - gt_c, axis=1).min()
                    offscreen["offset_err"].append(float(d))
            per_frame.append(fr)
    offscreen["frames"] = len(offscreen["frames"])
    return res, offscreen, per_frame


def stat(a):
    a = np.array(a, float)
    if a.size == 0:
        return None
    return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                mean=float(a.mean()), max=float(a.max()), n=int(a.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_heldout", type=int, default=200)
    ap.add_argument("--n_seen_eval", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_hm", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    base = load_frozen_base(args.weights, device)
    fc_head = FaceCenterHead(128).to(device)
    off_head = OffsetHead(128).to(device)
    opt = torch.optim.Adam(list(fc_head.parameters()) + list(off_head.parameters()),
                           lr=args.lr)
    sl1 = nn.SmoothL1Loss(reduction="none")

    # ── select ──
    train_sel = stratified_train(args.n_train, args.seed)
    held_sel = heldout_frames(args.n_heldout, args.seed)
    print(f"[data] train_sel={len(train_sel)} held_sel={len(held_sel)} "
          f"(held-out = v3 batch_009, NOT in train)")

    print("[data] materializing train (cache vgg feat)...")
    train = materialize(train_sel, device, want_belief=False, base=base)
    print(f"[data] usable train={len(train)}")
    # train V dist
    vd = {}
    for s in train:
        vd[s["V"]] = vd.get(s["V"], 0) + 1
    print(f"[data] train V dist: {dict(sorted(vd.items()))} "
          f"V<8 frac={sum(v for k,v in vd.items() if k<8)/max(len(train),1):.3f}")

    print("[data] materializing held-out (cache vgg feat + B2 belief)...")
    held = materialize(held_sel, device, want_belief=True, base=base,
                       limit=args.n_heldout)
    print(f"[data] usable held={len(held)}")
    # seen-eval subset (from train, but also need belief -> recompute belief on these)
    seen_sub = train[:args.n_seen_eval]
    print("[data] computing B2 belief for seen-eval subset...")
    with torch.no_grad():
        for s in seen_sub:
            img = cv2.imread(s["img_path"])
            t, _, _ = preprocess_fixed(img)
            bel, _ = base(t.to(device))
            s["belief"] = bel[-1][0].cpu().numpy()

    # ── sanity round-trip on first train sample ──
    s0 = train[0]
    rt = []
    for fi in range(N_FACE):
        if not s0["vis"][fi]:
            continue
        ax, ay = s0["anchors"][fi]
        dec = decode_corners(s0["off_t"][0].numpy(), fi, ax, ay)
        rt.extend(np.linalg.norm(dec - s0["corners"][fi], axis=1).tolist())
    rt_med = float(np.median(rt)) if rt else None
    print(f"[sanity] target round-trip corner err median={rt_med:.5f} cells")
    assert rt_med is not None and rt_med < 0.05, "round-trip target bug!"

    # ── train (mini-batch) ──
    n = len(train)
    log = []
    idx = np.arange(n)
    for ep in range(1, args.epochs + 1):
        fc_head.train(); off_head.train()
        np.random.shuffle(idx)
        ep_fc = ep_off = 0.0
        nb = 0
        for bi in range(0, n, args.batch):
            chunk = idx[bi:bi + args.batch]
            opt.zero_grad()
            l_fc_acc = l_off_acc = 0.0
            for j in chunk:
                s = train[j]
                feat = s["feat"].to(device)
                hm_p = fc_head(feat)
                off_p = off_head(feat)
                hm_t = s["hm_t"].to(device)
                w_hm = 1.0 + HM_POS_W * hm_t
                l_fc = ((hm_p - hm_t) ** 2 * w_hm).sum() / w_hm.sum()
                ow = s["ow_t"].to(device)
                w24 = ow.repeat_interleave(N_CORNER * 2, dim=0).unsqueeze(0)
                lm = sl1(off_p, s["off_t"].to(device))
                l_off = (lm * w24).sum() / (w24.sum() + 1e-6)
                loss = args.lambda_hm * l_fc + l_off
                loss.backward()
                l_fc_acc += float(l_fc); l_off_acc += float(l_off)
            opt.step()
            ep_fc += l_fc_acc / len(chunk)
            ep_off += l_off_acc / len(chunk)
            nb += 1
        lf, lo = ep_fc / nb, ep_off / nb
        log.append({"epoch": ep, "face_mse": lf, "offset_sl1": lo})
        print(f"[ep {ep}] face_mse={lf:.6f} offset_sl1={lo:.6f}  "
              f"({time.time()-t0:.0f}s)")

    # ── eval ──
    print("[eval] held-out + seen-subset...")
    held_res, held_off, held_pf = run_eval(held, fc_head, off_head, device, "held")
    seen_res, seen_off, seen_pf = run_eval(seen_sub, fc_head, off_head, device, "seen")

    def pack(res):
        return {b: {p: stat(res[b][p]) for p in ("offset", "heatmap")}
                for b in ("V8", "Vlt8")}

    held_pack = pack(held_res)
    seen_pack = pack(seen_res)

    # ── verdict ──
    seen_off_lt8 = seen_pack["Vlt8"]["offset"]
    seen_off_v8 = seen_pack["V8"]["offset"]
    # anti-undertraining: seen offset corner median small (<10 px). px = cell*STRIDE.
    seen_med_px = None
    for cand in (seen_off_lt8, seen_off_v8):
        if cand:
            seen_med_px = cand["median"] * STRIDE
            break
    off_flat = log[-1]["offset_sl1"] < 5e-3
    learned = (seen_med_px is not None and seen_med_px < 10.0)

    # main: held-out V<8 offset vs heatmap
    h_lt8_off = held_pack["Vlt8"]["offset"]
    h_lt8_hm = held_pack["Vlt8"]["heatmap"]
    h_v8_off = held_pack["V8"]["offset"]
    h_v8_hm = held_pack["V8"]["heatmap"]

    gain_lt8 = None
    if h_lt8_off and h_lt8_hm:
        gain_lt8 = (h_lt8_hm["median"] - h_lt8_off["median"]) * STRIDE  # +ve = offset better (px)
    # off-screen recovery: heatmap NA fraction + offset err on those corners
    recov = None
    if held_off["n_offscreen"] > 0:
        recov = dict(
            n_offscreen=held_off["n_offscreen"],
            frames=held_off["frames"],
            heatmap_na=held_off["heatmap_na"],
            heatmap_na_frac=held_off["heatmap_na"] / held_off["n_offscreen"],
            offset_err_px=stat([e * STRIDE for e in held_off["offset_err"]]))

    if not learned:
        verdict = ("INCONCLUSIVE (undertraining: seen-subset offset corner "
                   f"median={seen_med_px:.1f}px >= 10px or no seen offset data)")
    else:
        meaningful = gain_lt8 is not None and gain_lt8 > 2.0
        recov_win = (recov is not None and recov["heatmap_na_frac"] > 0.2 and
                     recov["offset_err_px"] is not None and
                     recov["offset_err_px"]["median"] < 30.0)
        v8_sane = True
        if h_v8_off and h_v8_hm:
            v8_sane = h_v8_off["median"] * STRIDE < (h_v8_hm["median"] * STRIDE + 40.0)
        if (meaningful or recov_win) and v8_sane:
            verdict = "PASS (held-out V<8 offset gain over heatmap)"
        elif (meaningful or recov_win) and not v8_sane:
            verdict = ("MIXED (V<8 gain but V8 offset catastrophic vs heatmap "
                       "— representation not sane on full view)")
        else:
            verdict = ("FAIL/lean-negative (learned but no held-out V<8 gain "
                       "over heatmap)")

    result = dict(
        config=dict(n_train_req=args.n_train, n_train_used=len(train),
                    n_heldout=len(held), n_seen_eval=len(seen_sub),
                    epochs=args.epochs, batch=args.batch, lr=args.lr,
                    lambda_hm=args.lambda_hm, seed=args.seed,
                    input=INPUT, grid=GRID, stride=STRIDE),
        units="corner median reported in OUTPUT CELLS; *_px = cells*8 (input-400 px)",
        roundtrip_med_cell=rt_med,
        train_V_dist={str(k): v for k, v in sorted(vd.items())},
        loss_log=log,
        learned=bool(learned), seen_offset_median_px=seen_med_px,
        offset_loss_flat=bool(off_flat),
        held_out={"V8": held_pack["V8"], "Vlt8": held_pack["Vlt8"]},
        seen={"V8": seen_pack["V8"], "Vlt8": seen_pack["Vlt8"]},
        held_Vlt8_offset_gain_over_heatmap_px=gain_lt8,
        offscreen_recovery=recov,
        verdict=verdict,
        caveat=("synth held-out signal only — NOT a real-domain guarantee. "
                "single config, synthetic. order-free Hungarian corner match."),
    )
    json.dump(result, open(os.path.join(OUT_DIR, "quickscreen3k.json"), "w"),
              indent=2)

    # ── V<8 comparison overlays (offset-path vs GT vs heatmap-path) ──
    n_ov = 0
    for s in held:
        if n_ov >= 5 or s["V"] >= 8:
            continue
        feat = s["feat"].to(device)
        with torch.no_grad():
            hm_p = fc_head(feat)[0].cpu().numpy()
            off_p = off_head(feat)[0].cpu().numpy()
        img = cv2.resize(cv2.imread(s["img_path"]), (INPUT, INPUT))
        # GT8 (white ring)
        for c in range(8):
            g = s["gt8_out"][c]
            col = (255, 255, 255) if s["kp_in"][c] else (180, 180, 180)
            cv2.circle(img, (int(g[0] * STRIDE), int(g[1] * STRIDE)), 6, col, 1)
            if not s["kp_in"][c]:
                cv2.putText(img, "off", (int(g[0]*STRIDE)+4, int(g[1]*STRIDE)),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (180, 180, 180), 1)
        # offset-path (green filled)
        for fi in range(N_FACE):
            if not s["vis"][fi]:
                continue
            px, py = peak_of(hm_p[fi])
            dec = decode_corners(off_p, fi, px, py)
            for c in range(N_CORNER):
                cv2.circle(img, (int(dec[c, 0] * STRIDE), int(dec[c, 1] * STRIDE)),
                           4, (0, 210, 90), -1)
        # heatmap-path (red cross; only produced corners)
        hb = extract_belief_peaks(s["belief"])
        for p in hb:
            if p is None:
                continue
            cv2.drawMarker(img, (int(p[0]*STRIDE), int(p[1]*STRIDE)), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 12, 2)
        legend = "white=GT(grey=offscreen) green=offset-path red=heatmap-path"
        cv2.putText(img, legend, (4, INPUT-8), cv2.FONT_HERSHEY_PLAIN, 0.8,
                    (255, 255, 0), 1)
        cv2.putText(img, f"V={s['V']}", (4, 16), cv2.FONT_HERSHEY_PLAIN, 1.2,
                    (255, 255, 0), 1)
        fid = s["fid"].replace("/", "_").replace(".json", "")
        cv2.imwrite(os.path.join(OUT_DIR, f"vlt8_cmp_{n_ov:02d}_{fid}.png"), img)
        n_ov += 1

    # ── console summary ──
    def row(name, p):
        if p is None:
            return f"  {name:28s}  (no data)"
        return (f"  {name:28s}  med={p['median']*STRIDE:6.1f}px "
                f"p95={p['p95']*STRIDE:6.1f}px n={p['n']}")
    print("\n=== STAGE15 Gate 3 (quickscreen 3k) ===")
    print(f"train={len(train)} held={len(held)} seen_eval={len(seen_sub)} "
          f"epochs={args.epochs}  ({time.time()-t0:.0f}s)")
    print(f"loss final: face_mse={log[-1]['face_mse']:.6f} "
          f"offset_sl1={log[-1]['offset_sl1']:.6f}")
    print("-- SEEN-subset (anti-undertraining) --")
    print(row("V8  offset-path", seen_pack["V8"]["offset"]))
    print(row("V<8 offset-path", seen_pack["Vlt8"]["offset"]))
    print("-- HELD-OUT (main judgment) --")
    print(row("V8  offset-path", held_pack["V8"]["offset"]))
    print(row("V8  heatmap-path", held_pack["V8"]["heatmap"]))
    print(row("V<8 offset-path", held_pack["Vlt8"]["offset"]))
    print(row("V<8 heatmap-path", held_pack["Vlt8"]["heatmap"]))
    if gain_lt8 is not None:
        print(f"  >> held V<8 offset gain over heatmap = {gain_lt8:+.1f}px "
              f"(+ve = offset better)")
    if recov:
        print(f"-- off-screen corner recovery (held, kp_in=False) --")
        print(f"  n_offscreen_corners={recov['n_offscreen']} in "
              f"{recov['frames']} frames")
        print(f"  heatmap NA on {recov['heatmap_na']} "
              f"({recov['heatmap_na_frac']*100:.0f}%) of them")
        if recov["offset_err_px"]:
            print(f"  offset-path err to those GT corners: "
                  f"med={recov['offset_err_px']['median']:.1f}px "
                  f"n={recov['offset_err_px']['n']}")
    print(f"\nVERDICT: {verdict}")
    print(f"[save] {OUT_DIR}/quickscreen3k.json + {n_ov} V<8 overlays")


if __name__ == "__main__":
    main()
