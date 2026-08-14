"""wood_infer_filter.py — PAPER_S2 Stage B best on UNSEEN wood pallet + deployable filters.

목적: Stage B best(net_epoch_0057) 를 나무 파렛트(wood, 학습에 없던 파렛트 =
      치수 0.8x0.59x0.14m·텍스처 다름) 프레임에 추론하고, GT 없이도 쓸 수 있는
      self-supervised 필터(f1~f7 + reproj self-consistency)를 걸어 무엇이 통과하는지 본다.

★ caveat (과장 금지):
  - wood = UNSEEN (치수·텍스처 학습에 없음) → 일반화 탐색.
  - GT 없음 → 정확도(honest8/corner_med) 측정 불가. self-consistency(reproj) + 육안만.
  - 소표본·단일 세션 2개. 필터 임계는 heuristic(관례값).

전처리:
  - PRIMARY = squash-parity (Stage B 학습 정합; 640x480->400x400 aniso, belief(50)->orig
    x(W/50,H/50)). paper_s2_real_eval.eval_frame_squash 방식.
  - pad100 = 검출 대비용 (aspect-preserving reflect-pad; Stage B 엔 OFF-distribution =
    squash-trained. det 대조만, 필터는 squash 로).

필터 (배포가능 = GT 불요):
  f1_peak, f2_peak_ratio, f3_flip, f4_tta_stab, f5_rear_conf, f6_frsep, f7_posdepth
  + reproj self-consistency gate (pred kp -> solve_pose(wood dims) -> 재투영 vs pred kp
    median px. GT 아님 = 자기일관성. <~10px = pose 자기일관).
  ★ f8_size_env / f9_bbox_iou 는 GT envelope/bbox 필요 → wood GT 없음 → 적용 불가(제외).

dims: solve_pose 는 module PALLET_DIMS 를 call-time 에 읽으므로 (0.8,0.59,0.14) 로 override.
      W/D swap 은 solve_pose 가 자동(110/130 정면 둘 다 시도).
K: wood = RealSense D435I. 1280x720 = 1080p 에서 16:9 선형유도 (annotate_wood.K_for_resolution).
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import glob
import importlib.util
import os
import sys

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))

import torch  # noqa: E402


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# reuse s1_cad_9filters pure scorers + E (load_model, pad_frame, belief_to_orig_pad)
s1 = _load("s1f", os.path.join(ROOT, "scripts", "stage0", "selftrain", "s1_cad_9filters.py"))
E = s1.E
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
from eval_pvnet_heads import preprocess  # noqa: E402
import annotate_pnp as APNP  # noqa: E402
from annotate_wood import K_for_resolution  # noqa: E402

# ── wood pallet override (solve_pose reads module PALLET_DIMS at call-time) ──
WOOD_DIMS = (0.8, 0.59, 0.14)
APNP.PALLET_DIMS = WOOD_DIMS

WEIGHTS = os.path.join(ROOT, "weights", "paper_s2_stageB", "net_epoch_0057.pth")
WOOD_ROOT = os.path.join(ROOT, "data", "pallet", "raw_data", "wood", "selected")
SESSIONS = ["pallet_20260618_183705", "pallet_20260618_184309"]
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results",
                       "paper_s2_scratch_diffpnp")
OV_DIR = os.path.join(OUT_DIR, "wood_overlays")

THRESH = 0.3
N_DET_MIN = 6
PAD = 100
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FLIP_PAIRS = s1.FLIP_PAIRS

# heuristic thresholds (관례값 = s1.TAU) + reproj self-consistency gate
TAU = dict(s1.TAU)
REPROJ_TAU = 10.0   # median self-consistency reproj px
FILTERS = ["f1_peak", "f2_peak_ratio", "f3_flip", "f4_tta_stab",
           "f5_rear_conf", "f6_frsep", "f7_posdepth", "reproj_selfcons"]


# ── squash decode (Stage B training parity) ──────────────────────────────
def preprocess_squash(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (400, 400), interpolation=cv2.INTER_LINEAR)
    t = (r.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0)


def decode_squash(model, img_bgr, device):
    """squash belief -> pred8(8,2 nan), pred_c, peaks(9,), ratios(8,)."""
    H, W = img_bgr.shape[:2]
    sx, sy = W / 50.0, H / 50.0
    with torch.no_grad():
        beliefs, _ = model(preprocess_squash(img_bgr).to(device))
    belief = beliefs[-1][0].cpu().numpy()
    kps = extract_keypoints_from_belief(belief, THRESH)
    pred8 = np.full((8, 2), np.nan)
    peaks = np.array([kps[i][2] for i in range(9)])
    for i in range(8):
        if kps[i][0] >= 0:
            pred8[i] = [kps[i][0] * sx, kps[i][1] * sy]
    pred_c = [kps[8][0] * sx, kps[8][1] * sy] if kps[8][0] >= 0 else None
    ratios = np.array([s1.top2_local_peak_ratio(belief[i]) for i in range(8)])
    return pred8, pred_c, peaks, ratios


def flip_infer_squash(model, img_bgr, device):
    H, W = img_bgr.shape[:2]
    pred8, pred_c, _, _ = decode_squash(model, cv2.flip(img_bgr, 1), device)
    # un-flip = exact cv2.flip inverse: cv2.flip maps col i -> (W-1)-i, so
    # x_orig = (W-1) - x_flip (NOT W-x). This is the pixel-flip-axis convention;
    # the residual ~18px that remains is MODEL L-R non-equivariance (no flip aug
    # in training), NOT a coordinate offset — verified via belief-space mirror
    # which yields the same ~20px. See _wood_flip_diag.py.
    kpf = [None] * 9
    for i in range(8):
        if not np.isnan(pred8[i, 0]):
            kpf[i] = ((W - 1) - pred8[i, 0], pred8[i, 1])
    if pred_c is not None:
        kpf[8] = ((W - 1) - pred_c[0], pred_c[1])
    swap = list(range(9))
    for a, b in FLIP_PAIRS:
        swap[a], swap[b] = b, a
    return [kpf[swap[i]] for i in range(9)]


def tta_stab_squash(model, img_bgr, device):
    H, W = img_bgr.shape[:2]
    variants = [img_bgr,
                np.clip(img_bgr.astype(np.float32) * 1.2, 0, 255).astype(np.uint8),
                np.clip(img_bgr.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)]
    small = cv2.resize(img_bgr, (int(W * 0.9), int(H * 0.9)))
    variants.append(cv2.resize(small, (W, H)))
    preds = np.stack([decode_squash(model, v, device)[0] for v in variants], 0)
    stds = []
    for i in range(8):
        col = preds[:, i, :]
        ok = ~np.isnan(col[:, 0])
        if ok.sum() >= 3:
            stds.append(float(np.mean(np.std(col[ok], axis=0))))
    return float(np.mean(stds)) if stds else None


def solve_wood(pred8, pred_c, K, img_shape):
    """solve_pose (wood dims via module override) -> (pose, posdepth_ok, reproj_med)."""
    kps9 = [None if np.isnan(pred8[i, 0]) else
            [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)]
    kps9.append(list(pred_c) if pred_c is not None else None)
    if sum(1 for k in kps9 if k is not None) < N_DET_MIN:
        return None, False, None
    try:
        pose = APNP.solve_pose(kps9, K, dims=WOOD_DIMS, img_shape=img_shape)
    except Exception:
        return None, False, None
    if pose is None:
        return None, False, None
    R, t = np.array(pose["R"], float), np.array(pose["t"], float).ravel()
    kp3d = APNP.make_pallet_keypoints_3d_diagram(*WOOD_DIMS)
    cam = (R @ kp3d.T).T + t
    posdepth = bool(np.all(cam[:, 2] > 0) and t[2] > 0)
    # reproj self-consistency: reprojected cuboid vs pred kp (GT 아님 = 자기일관)
    proj = np.array(pose["projected_all"], float)   # (9,2), sentinel (-1,-1)
    errs = []
    for i in range(9):
        p = kps9[i]
        if p is None:
            continue
        u, v = proj[i]
        if u == -1.0 and v == -1.0:
            continue
        errs.append(float(np.hypot(u - p[0], v - p[1])))
    reproj_med = float(np.median(errs)) if errs else None
    return pose, posdepth, reproj_med


def process(model, ip, K, device):
    img = cv2.imread(ip)
    if img is None:
        return None
    H, W = img.shape[:2]
    pred8, pred_c, peaks, ratios = decode_squash(model, img, device)
    n_det = int((~np.isnan(pred8[:, 0])).sum())

    det8 = [i for i in range(8) if not np.isnan(pred8[i, 0])]
    rear = [i for i in (4, 5, 6, 7) if not np.isnan(pred8[i, 0])]
    f1 = float(min(peaks[i] for i in det8)) if len(det8) >= N_DET_MIN else None
    f2 = float(min(ratios[i] for i in det8)) if len(det8) >= N_DET_MIN else None
    f5 = float(min(peaks[i] for i in rear)) if len(rear) == 4 else None
    f6 = s1.frsep_frac(pred8)

    f3 = f4 = None
    pose = None
    posdepth = False
    reproj = None
    if n_det >= N_DET_MIN:
        kpB = flip_infer_squash(model, img, device)
        f3 = s1.flip_score(pred8, pred_c, kpB)
        f4 = tta_stab_squash(model, img, device)
        pose, posdepth, reproj = solve_wood(pred8, pred_c, K, img.shape)

    proj_all = None
    if pose is not None:
        pa = np.array(pose["projected_all"], float)[:8]
        bad = (pa[:, 0] == -1.0) & (pa[:, 1] == -1.0)
        pa[bad] = np.nan
        proj_all = pa

    scores = {"f1_peak": f1, "f2_peak_ratio": f2, "f3_flip": f3,
              "f4_tta_stab": f4, "f5_rear_conf": f5, "f6_frsep": f6,
              "f7_posdepth": posdepth, "reproj_selfcons": reproj}
    return {"ip": ip, "fid": os.path.splitext(os.path.basename(ip))[0],
            "n_det": n_det, "pred8": pred8, "pred_c": pred_c,
            "proj_all": proj_all, "scores": scores}


def passes(fname, sc, n_det):
    if n_det < N_DET_MIN:
        return False
    v = sc[fname]
    if fname == "f7_posdepth":
        return bool(v)
    if v is None:
        return False
    if fname in ("f1_peak", "f2_peak_ratio", "f5_rear_conf", "f6_frsep"):
        return v >= TAU[fname]
    if fname in ("f3_flip", "f4_tta_stab"):
        return v <= TAU[fname]
    if fname == "reproj_selfcons":
        return v <= REPROJ_TAU
    return False


# ── overlay: pred kp blue + PnP cuboid red ───────────────────────────────
EDGES = s1.EDGES
BLUE = (255, 0, 0)
RED = (0, 0, 255)


def save_overlay(rec, verdict, path):
    img = cv2.imread(rec["ip"])
    if img is None:
        return
    pa = rec["proj_all"]
    if pa is not None:
        for a, b in EDGES:
            if not (np.isnan(pa[a, 0]) or np.isnan(pa[b, 0])):
                cv2.line(img, (int(pa[a, 0]), int(pa[a, 1])),
                         (int(pa[b, 0]), int(pa[b, 1])), RED, 2, cv2.LINE_AA)
    p8 = rec["pred8"]
    for i in range(8):
        if not np.isnan(p8[i, 0]):
            p = (int(p8[i, 0]), int(p8[i, 1]))
            cv2.circle(img, p, 5, BLUE, -1, cv2.LINE_AA)
            cv2.putText(img, str(i), (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1, cv2.LINE_AA)
    if rec["pred_c"] is not None:
        c = rec["pred_c"]
        cv2.drawMarker(img, (int(c[0]), int(c[1])), BLUE, cv2.MARKER_CROSS, 12, 2)
    sc = rec["scores"]
    npass = sum(1 for f in FILTERS if passes(f, sc, rec["n_det"]))
    rp = f"{sc['reproj_selfcons']:.1f}" if sc["reproj_selfcons"] is not None else "n/a"
    pk = f"{sc['f1_peak']:.2f}" if sc["f1_peak"] is not None else "n/a"
    h1 = f"{verdict}  det={rec['n_det']}/8  #pass={npass}/8  reproj={rp}px  peak={pk}"
    h2 = "wood UNSEEN (0.8x0.59x0.14) | NO GT: self-consistency only | pred=blue PnP-cuboid=red"
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, h1, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, h2, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def pad_det_count(model, ip, device):
    """pad100 aspect decode det count (Stage B OFF-distribution; det 대조만)."""
    img = cv2.imread(ip)
    if img is None:
        return None
    belief, geom, wh = s1.infer_belief(model, img, device, PAD)
    pred8, _, _, _ = s1.belief_to_pred(belief, geom, wh, PAD, THRESH)
    return int((~np.isnan(pred8[:, 0])).sum())


def main():
    os.makedirs(OV_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(OV_DIR, "*.jpg")):
        os.remove(old)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = s1.E.load_model(WEIGHTS, device)

    frames = []
    for ses in SESSIONS:
        for jp in sorted(glob.glob(os.path.join(WOOD_ROOT, ses, "*.jpg"))):
            frames.append((ses, jp))
    img0 = cv2.imread(frames[0][1])
    H, W = img0.shape[:2]
    K, ksrc = K_for_resolution(W, H)
    print(f"[wood] N={len(frames)}  res={W}x{H}  K={ksrc}")
    print(f"       dims={WOOD_DIMS}  weights=Stage B ep0057  preprocess=SQUASH(parity)")

    recs = []
    for i, (ses, ip) in enumerate(frames):
        r = process(model, ip, K, device)
        if r is None:
            continue
        r["ses"] = ses
        r["pad_det"] = pad_det_count(model, ip, device)
        recs.append(r)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(frames)}] {r['fid']} det={r['n_det']} "
                  f"pad_det={r['pad_det']}")

    # ── aggregate ────────────────────────────────────────────────────────
    N = len(recs)
    n_det = sum(1 for r in recs if r["n_det"] >= N_DET_MIN)
    n_pad_det = sum(1 for r in recs if r["pad_det"] is not None and r["pad_det"] >= N_DET_MIN)
    per_filter = {f: 0 for f in FILTERS}
    for r in recs:
        for f in FILTERS:
            if passes(f, r["scores"], r["n_det"]):
                per_filter[f] += 1

    def combo(fs, r):
        return all(passes(f, r["scores"], r["n_det"]) for f in fs)

    STRONG = ["f1_peak", "f4_tta_stab", "f5_rear_conf", "reproj_selfcons"]
    ALL7R = ["f1_peak", "f2_peak_ratio", "f3_flip", "f4_tta_stab",
             "f5_rear_conf", "f6_frsep", "f7_posdepth", "reproj_selfcons"]
    n_strong = sum(1 for r in recs if combo(STRONG, r))
    n_all = sum(1 for r in recs if combo(ALL7R, r))
    for r in recs:
        r["accept"] = combo(STRONG, r)   # headline accept = strong combo

    # ── overlays: accepts + rejects ──────────────────────────────────────
    accepts = [r for r in recs if r["accept"]]
    dets = [r for r in recs if r["n_det"] >= N_DET_MIN]
    rejects = [r for r in dets if not r["accept"]]
    # rank rejects worst reproj first (show filter catching collapse)
    rejects.sort(key=lambda r: -(r["scores"]["reproj_selfcons"] or 1e9))
    accepts.sort(key=lambda r: (r["scores"]["reproj_selfcons"] or 1e9))
    no_det = [r for r in recs if r["n_det"] < N_DET_MIN]

    n_ov = 0
    for r in accepts[:6]:
        save_overlay(r, "ACCEPT", os.path.join(OV_DIR, f"accept_{r['ses'][-6:]}_{r['fid']}.jpg"))
        n_ov += 1
    for r in rejects[:6]:
        save_overlay(r, "REJECT", os.path.join(OV_DIR, f"reject_{r['ses'][-6:]}_{r['fid']}.jpg"))
        n_ov += 1

    # ── markdown report ──────────────────────────────────────────────────
    write_md(recs, N, n_det, n_pad_det, per_filter, n_strong, n_all,
             len(accepts), len(rejects), len(no_det), n_ov, K, ksrc, W, H)
    print(f"\n[save] {OUT_DIR}/wood_infer_filter.md")
    print(f"[save] {n_ov} overlays -> {OV_DIR}")


def _cell(v, fmt="{:.2f}"):
    return "  -  " if v is None else fmt.format(v)


def write_md(recs, N, n_det, n_pad_det, per_filter, n_strong, n_all,
             n_acc, n_rej, n_nodet, n_ov, K, ksrc, W, H):
    L = []
    L.append("# Stage B best on UNSEEN wood pallet + deployable filters")
    L.append("")
    L.append("- weights: `weights/paper_s2_stageB/net_epoch_0057.pth` (PAPER_S2 Stage B best)")
    L.append(f"- data: wood RealSense D435I, N={N} ({', '.join(SESSIONS)}), {W}x{H}")
    L.append(f"- dims (unseen, measured): {WOOD_DIMS} m | K: {ksrc}")
    L.append(f"        fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    L.append("- preprocess: **SQUASH** (Stage B training parity). pad100 = det 대조(OFF-dist).")
    L.append("- ★ wood = UNSEEN(치수·텍스처 학습에 없음), **GT 없음**(정확도 미측정 =")
    L.append("  self-consistency reproj + 육안만), 소표본. 필터 임계 = heuristic 관례값.")
    L.append("- f8_size_env / f9_bbox_iou = GT envelope/bbox 필요 → wood GT 없음 → **제외**.")
    L.append("")
    L.append("## 검출률 (n_det>=6)")
    L.append("```")
    L.append(f"squash (parity)      : {n_det}/{N}  ({100*n_det/N:.0f}%)")
    L.append(f"pad100 (OFF-dist 대조): {n_pad_det}/{N}  ({100*n_pad_det/N:.0f}%)")
    L.append("```")
    L.append("")
    L.append("## 필터별 통과 (검출 프레임 대상, threshold=관례값)")
    L.append("```")
    L.append(f"{'filter':<16}{'tau':>8}{'n_pass':>8}{'pass%(/N)':>11}")
    L.append("-" * 43)
    taus = {**TAU, "f7_posdepth": "-", "reproj_selfcons": f"<={REPROJ_TAU}"}
    for f in FILTERS:
        t = taus.get(f)
        ts = f"{t}" if not isinstance(t, float) else f"{t:g}"
        L.append(f"{f:<16}{ts:>8}{per_filter[f]:>8}{100*per_filter[f]/N:>10.0f}%")
    L.append("-" * 43)
    L.append(f"COMBO strong(f1&f4&f5&reproj)  n={n_strong}  ({100*n_strong/N:.0f}%)  [headline ACCEPT]")
    L.append(f"COMBO all7&reproj              n={n_all}  ({100*n_all/N:.0f}%)")
    L.append("```")
    L.append("")
    L.append(f"accept(strong)={n_acc}  reject(det but not accept)={n_rej}  "
             f"no-det(<6)={n_nodet}  overlays={n_ov}")
    L.append("")
    L.append("## per-frame")
    L.append("```")
    hdr = (f"{'frame':<24}{'det':>4}{'peak':>6}{'p_rat':>6}{'flip':>6}"
           f"{'tta':>6}{'rear':>6}{'frsep':>6}{'pdz':>4}{'reproj':>7}"
           f"{'#p':>4}  accept")
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in recs:
        sc = r["scores"]
        npass = sum(1 for f in FILTERS if passes(f, sc, r["n_det"]))
        pdz = "Y" if sc["f7_posdepth"] else "n"
        acc = "ACCEPT" if r["accept"] else ("reject" if r["n_det"] >= N_DET_MIN else "no-det")
        L.append(
            f"{r['ses'][-6:]+'/'+r['fid']:<24}{r['n_det']:>4}"
            f"{_cell(sc['f1_peak']):>6}{_cell(sc['f2_peak_ratio']):>6}"
            f"{_cell(sc['f3_flip'],'{:.1f}'):>6}{_cell(sc['f4_tta_stab'],'{:.1f}'):>6}"
            f"{_cell(sc['f5_rear_conf']):>6}{_cell(sc['f6_frsep']):>6}"
            f"{pdz:>4}{_cell(sc['reproj_selfcons'],'{:.1f}'):>7}{npass:>4}  {acc}")
    L.append("```")
    L.append("")
    L.append("## 요약 / 정성 판정")
    L.append(f"- 검출: unseen wood 에서 squash {n_det}/{N}({100*n_det/N:.0f}%) 검출.")
    L.append(f"- 필터: strong combo(f1&f4&f5&reproj) accept {n_acc}/{N}"
             f"({100*n_acc/N:.0f}%), 전필터 accept {n_all}.")
    L.append("- (오버레이 육안 판정은 보고 본문 참조. GT 없음 = 정확도 아님.)")
    L.append("")
    L.append("★ caveat: wood UNSEEN(치수·텍스처 OOD) + GT 없음(reproj=자기일관성, 정확도 아님)")
    L.append("  + 소표본 2세션. 필터 임계 heuristic. 절대수치 과해석 금지.")
    with open(os.path.join(OUT_DIR, "wood_infer_filter.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
