"""s1_cad_9filters.py — Paper-S1 on cad(22) with 9 pseudo-label ACCEPT filters,
each applied INDIVIDUALLY (not combined).

목적: Paper-S1 를 cad(손라벨 22장)에 추론하고, 9개 PL accept 필터를 각각 하나씩
      적용해 어느 프레임이 통과하는지 본다. cad 는 GT(손라벨 8/8) 가 있어
      "통과 = 실제 정확한가"(precision = 통과 프레임 중 corner_med<10px 비율) 도 측정.

★ 주의 (의심 §5):
  - N=22 극소표본. cad = challenge palletobj = paper-track S1 엔 unseen (도메인갭 큼).
  - 극근접 near-field → reflect-pad100 추론 (near-field 검출 확보용, official eval 아님. 명시).
  - 통과수만 보지 말고 GT precision 병행 (통과했는데 부정확 = 쓸모없는 통과).
  - 과거 "cad clean PL ≈ 0"(6/18) 와 대조. 과결론 금지.

인프라 재사용 (새 기하 없음):
  eval_capturecad_b2 (E) : load_model, pad_frame, belief_to_orig_pad, hungarian,
                           K_from_json, PALLET_DIMS 경유 solve_pose
  eval_pvnet_heads       : preprocess, split_metrics
  filter_pr_camfacing    : extract_keypoints_from_belief
  annotate_pnp (APNP)    : solve_pose, make_pallet_keypoints_3d_diagram
  stage18                : front_rear_sep (depth-edge 2D 분리)

9 필터 (각각 단독):
  f1 peak        : 8코너 belief peak 최소 >= tau         (conf gate 관례 0.5)
  f2 peak_ratio  : 8코너 1등/2등-local-peak 비 최소 >= tau (worst corner, 관례 1.5)
  f3 flip        : L-R flip TTA index-aligned 평균거리 <= tau (deployed 10px)
  f4 tta_stab    : scale/brightness TTA 코너 위치 std 평균 <= tau (관례 5px)
  f5 rear_conf   : rear 코너 4-7 belief peak 최소 >= tau  (관례 0.5)
  f6 frsep       : pred depth-edge(0-4..3-7) 2D 분리 / cuboid diag >= tau (붕괴 아님)
  f7 posdepth    : solve_pose 후 9 kp3d cam-z>0 & t_z>0
  f8 size_env    : pred bbox size-ratio & aspect 가 GT envelope 내부 (a4_score==0)
  f9 bbox_iou    : pred 8코너 bbox vs GT bbox IoU >= tau (관례 0.5)

판정: 통과수/22 + precision(통과 중 corner_med<10px). corner_med = order-free
      Hungarian median (convention 무관). 필터별 통과 프레임 오버레이 이미지 저장.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import importlib.util
import json
import os
import sys

import cv2
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))


sys.path[:0] = [os.path.join(ROOT, "challenge", "scripts", _s)
                for _s in ("annotate", "infer", "live")]
def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _load("ecad", os.path.join(
    ROOT, "data/pallet/eval_results/stage16_truncation_addon/"
    "capturecad_b2_eval/eval_capturecad_b2.py"))

import torch  # noqa: E402
from eval_pvnet_heads import preprocess, split_metrics  # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
from stage18_elevation_threshold import front_rear_sep  # noqa: E402
import annotate_pnp as APNP  # noqa: E402

WEIGHTS = os.path.join(ROOT, "weights", "paper_s1_maskaux", "net_epoch_0065.pth")
CAD_DIR = os.path.join(ROOT, "challenge", "data", "capturepalletcad_manual_gt")
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "s1_cad_filters")
PAD = 100
THRESH = 0.3
N_DET_MIN = 6
GOOD_PX = 10.0
FLIP_PAIRS = [(0, 1), (3, 2), (4, 5), (7, 6)]

# ── canonical thresholds (관례값; 아래 SWEEP 로 민감도 병기) ──────────────
TAU = {
    "f1_peak": 0.5,        # min corner belief peak
    "f2_peak_ratio": 1.5,  # min (1st/2nd local peak) over corners
    "f3_flip": 10.0,       # deployed flip gate (px)
    "f4_tta_stab": 5.0,    # mean per-corner TTA position std (px)
    "f5_rear_conf": 0.5,   # min rear(4-7) belief peak
    "f6_frsep": 0.06,      # pred depth-sep / cuboid-diag (collapse if below)
    "f9_bbox_iou": 0.5,    # pred bbox vs GT bbox IoU
}
SWEEP = {
    "f1_peak": [0.3, 0.4, 0.5, 0.6],
    "f2_peak_ratio": [1.2, 1.5, 2.0],
    "f3_flip": [5.0, 8.0, 10.0, 15.0],
    "f4_tta_stab": [3.0, 5.0, 8.0, 10.0],
    "f5_rear_conf": [0.3, 0.4, 0.5, 0.6],
    "f6_frsep": [0.03, 0.05, 0.08],
    "f9_bbox_iou": [0.3, 0.5, 0.7],
}
FILTER_ORDER = ["f1_peak", "f2_peak_ratio", "f3_flip", "f4_tta_stab",
                "f5_rear_conf", "f6_frsep", "f7_posdepth", "f8_size_env",
                "f9_bbox_iou"]
FILTER_DESC = {
    "f1_peak": "heatmap peak >=tau (min over 8 corners)",
    "f2_peak_ratio": "1st/2nd local peak >=tau (min over 8 corners)",
    "f3_flip": "L-R flip TTA mean dist <=tau px",
    "f4_tta_stab": "scale/brightness TTA pos std <=tau px",
    "f5_rear_conf": "rear(4-7) peak >=tau (min)",
    "f6_frsep": "pred depth-sep/cuboid-diag >=tau (not collapsed)",
    "f7_posdepth": "solve_pose all 9 cam-z>0 & t_z>0",
    "f8_size_env": "pred bbox size&aspect inside GT envelope",
    "f9_bbox_iou": "pred bbox vs GT bbox IoU >=tau",
}

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
GT_COL = (0, 255, 0)      # green
PRED_COL = (0, 0, 255)    # red
FRONT_TXT = (255, 255, 0)
REAR_TXT = (0, 200, 255)


# ── belief helpers ────────────────────────────────────────────────────
def top2_local_peak_ratio(bmap):
    """1등/2등 local-max 비. 2등 = 1등 주변 NMS 후 다음 local max.
    붕괴/모호(2 후보 팽팽)면 비가 1 근처 -> 낮음. 확실하면 크다(2등 없으면 inf)."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    m1 = float(bmap.max())
    if m1 <= 1e-6:
        return 0.0
    sm = gaussian_filter(bmap, sigma=2)
    mx = maximum_filter(sm, size=5)
    peaks = (sm == mx) & (sm > 1e-6)
    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        return float("inf")
    vals = np.sort(np.array([bmap[y, x] for y, x in zip(ys, xs)]))[::-1]
    if len(vals) < 2:
        return float("inf")   # 단봉 = 매우 확실
    return float(vals[0] / max(vals[1], 1e-6))


def infer_belief(model, img_bgr, device, pad):
    """pad+preprocess -> belief(9,H,W); return belief, (nw,nh,sc), (W,H)."""
    H, W = img_bgr.shape[:2]
    proc = E.pad_frame(img_bgr, pad)
    tensor, nw, nh, sc = preprocess(proc)
    with torch.no_grad():
        beliefs, _ = model(tensor.to(device))
    belief = beliefs[-1][0].cpu().numpy()
    return belief, (nw, nh, sc), (W, H)


def belief_to_pred(belief, geom, wh, pad, thr):
    """belief -> pred8 (8,2 nan), pred_c, peaks(9,), ratios(8,)."""
    nw, nh, sc = geom
    W, H = wh
    bh, bw = belief.shape[1], belief.shape[2]
    kps_bel = extract_keypoints_from_belief(belief, thr)
    pred8 = np.full((8, 2), np.nan)
    peaks = np.zeros(9)
    for i in range(9):
        peaks[i] = kps_bel[i][2]
    for i in range(8):
        if kps_bel[i][0] >= 0:
            pred8[i] = E.belief_to_orig_pad(kps_bel[i][0], kps_bel[i][1],
                                            bw, bh, nw, nh, sc, pad, W, H)
    pred_c = None
    if kps_bel[8][0] >= 0:
        pred_c = list(E.belief_to_orig_pad(kps_bel[8][0], kps_bel[8][1],
                                           bw, bh, nw, nh, sc, pad, W, H))
    ratios = np.array([top2_local_peak_ratio(belief[i]) for i in range(8)])
    return pred8, pred_c, peaks, ratios


# ── filter scores ─────────────────────────────────────────────────────
def bbox_of(pts):
    v = pts[~np.isnan(pts[:, 0])]
    if len(v) < 3:
        return None
    return np.array([v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()])


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 1e-6 else 0.0


def frsep_frac(pred8):
    """pred depth-edge 2D 분리 / pred cuboid bbox diag (scale-free)."""
    depth = [(0, 4), (1, 5), (2, 6), (3, 7)]
    ds = []
    for a, b in depth:
        if not (np.isnan(pred8[a, 0]) or np.isnan(pred8[b, 0])):
            ds.append(float(np.linalg.norm(pred8[a] - pred8[b])))
    if not ds:
        return None
    bb = bbox_of(pred8)
    if bb is None:
        return None
    diag = float(np.hypot(bb[2] - bb[0], bb[3] - bb[1]))
    if diag < 1e-6:
        return None
    return float(np.median(ds)) / diag


def size_aspect(pts, img_diag):
    bb = bbox_of(pts)
    if bb is None:
        return None, None
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    if w <= 1e-6 or h <= 1e-6:
        return None, None
    return float(np.hypot(w, h) / img_diag), float(w / h)


def excursion(v, lo, hi):
    if v is None or hi <= lo:
        return 1e9 if v is None else 0.0
    half = (hi - lo) / 2.0
    mid = (lo + hi) / 2.0
    return max(0.0, abs(v - mid) - half) / half


def flip_infer(model, img_bgr, device, pad, thr):
    """L-R flip 추론 -> un-flip + camera-facing swap, 원본 px 9 kp."""
    H, W = img_bgr.shape[:2]
    flip = cv2.flip(img_bgr, 1)
    belief, geom, wh = infer_belief(model, flip, device, pad)
    pred8, pred_c, _, _ = belief_to_pred(belief, geom, wh, pad, thr)
    # flipped-orig px -> un-flip x (W - x)
    kpf = [None] * 9
    for i in range(8):
        if not np.isnan(pred8[i, 0]):
            kpf[i] = (W - pred8[i, 0], pred8[i, 1])
    if pred_c is not None:
        kpf[8] = (W - pred_c[0], pred_c[1])
    swap = list(range(9))
    for a, b in FLIP_PAIRS:
        swap[a], swap[b] = b, a
    return [kpf[swap[i]] for i in range(9)]


def flip_score(pred8, pred_c, kpB):
    kpA = [None if np.isnan(pred8[i, 0]) else tuple(pred8[i]) for i in range(8)]
    kpA.append(tuple(pred_c) if pred_c is not None else None)
    ds = []
    for i in range(9):
        if kpA[i] is not None and kpB[i] is not None:
            ds.append(float(np.linalg.norm(np.array(kpA[i]) - np.array(kpB[i]))))
    if len(ds) < 4:
        return None
    return float(np.mean(ds))


def tta_stability(model, img_bgr, device, pad, thr):
    """scale/brightness TTA (coordinate-frame preserving) 코너 위치 std 평균.
    crop 은 좌표계 이동이라 제외(원본 px 비교 위해). variants: 원본 + bright↑/↓ + scale."""
    variants = [img_bgr]
    variants.append(np.clip(img_bgr.astype(np.float32) * 1.2, 0, 255).astype(np.uint8))
    variants.append(np.clip(img_bgr.astype(np.float32) * 0.8, 0, 255).astype(np.uint8))
    H, W = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (int(W * 0.9), int(H * 0.9)))
    variants.append(cv2.resize(small, (W, H)))   # scale down-up (frame preserved)
    preds = []
    for v in variants:
        belief, geom, wh = infer_belief(model, v, device, pad)
        p8, _, _, _ = belief_to_pred(belief, geom, wh, pad, thr)
        preds.append(p8)
    preds = np.stack(preds, 0)   # (V,8,2)
    stds = []
    for i in range(8):
        col = preds[:, i, :]
        ok = ~np.isnan(col[:, 0])
        if ok.sum() >= 3:
            stds.append(float(np.mean(np.std(col[ok], axis=0))))
    if not stds:
        return None
    return float(np.mean(stds))


def posdepth_ok(pred8, pred_c, K, img_shape):
    """solve_pose 후 9 kp3d cam-frame z>0 & t_z>0."""
    kps9 = [None if np.isnan(pred8[i, 0]) else
            [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)]
    kps9.append(list(pred_c) if pred_c is not None else None)
    if sum(1 for k in kps9 if k is not None) < N_DET_MIN:
        return False, None
    try:
        pose = APNP.solve_pose(kps9, K, dims=APNP.PALLET_DIMS, img_shape=img_shape)
    except Exception:
        return False, None
    if pose is None:
        return False, None
    R, t = np.array(pose["R"], float), np.array(pose["t"], float).ravel()
    kp3d = APNP.make_pallet_keypoints_3d_diagram(*APNP.PALLET_DIMS)
    cam = (R @ kp3d.T).T + t
    ok = bool(np.all(cam[:, 2] > 0) and t[2] > 0)
    return ok, pose


# ── per-frame processing ──────────────────────────────────────────────
def process(model, jp, ip, device):
    img = cv2.imread(ip)
    if img is None:
        return None
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    gtc = np.array(o["projected_cuboid_centroid"], float)
    H, W = img.shape[:2]
    K = E.K_from_json(d)
    inb = ((gt8[:, 0] >= 0) & (gt8[:, 0] < W) & (gt8[:, 1] >= 0) & (gt8[:, 1] < H))
    v_geom = int(inb.sum())
    img_diag = float(np.hypot(W, H))

    belief, geom, wh = infer_belief(model, img, device, PAD)
    pred8, pred_c, peaks, ratios = belief_to_pred(belief, geom, wh, PAD, THRESH)
    n_det = int((~np.isnan(pred8[:, 0])).sum())

    m = split_metrics(pred8, gt8)
    corner_med = m["overall"] if np.isfinite(m["overall"]) else None

    # per-filter continuous score (threshold applied later)
    det8 = [i for i in range(8) if not np.isnan(pred8[i, 0])]
    f1 = float(min(peaks[i] for i in det8)) if len(det8) >= N_DET_MIN else None
    f2 = float(min(ratios[i] for i in det8)) if len(det8) >= N_DET_MIN else None
    kpB = flip_infer(model, img, device, PAD, THRESH)
    f3 = flip_score(pred8, pred_c, kpB)
    f4 = tta_stability(model, img, device, PAD, THRESH)
    rear = [i for i in (4, 5, 6, 7) if not np.isnan(pred8[i, 0])]
    f5 = float(min(peaks[i] for i in rear)) if len(rear) == 4 else None
    f6 = frsep_frac(pred8)
    f7, pose = posdepth_ok(pred8, pred_c, K, img.shape)
    pred_sr, pred_asp = size_aspect(pred8, img_diag)
    gt_sr, gt_asp = size_aspect(gt8, img_diag)
    pred_bb = bbox_of(pred8)
    gt_bb = bbox_of(gt8)
    f9 = bbox_iou(pred_bb, gt_bb)

    proj_all = None
    if pose is not None:
        pa = np.array(pose["projected_all"], float)[:8]
        bad = (pa[:, 0] == -1.0) & (pa[:, 1] == -1.0)
        pa[bad] = np.nan
        proj_all = pa

    return {
        "fid": os.path.splitext(os.path.basename(jp))[0],
        "ip": ip, "v_geom": v_geom, "n_det": n_det,
        "corner_med": corner_med, "good": bool(corner_med is not None and corner_med < GOOD_PX),
        "gt8": gt8.tolist(), "gtc": gtc.tolist(),
        "pred8": pred8.tolist(), "pred_c": pred_c,
        "proj_all": proj_all.tolist() if proj_all is not None else None,
        "scores": {"f1_peak": f1, "f2_peak_ratio": f2, "f3_flip": f3,
                   "f4_tta_stab": f4, "f5_rear_conf": f5, "f6_frsep": f6,
                   "f9_bbox_iou": f9},
        "f7_posdepth": bool(f7),
        "pred_sr": pred_sr, "pred_asp": pred_asp,
        "gt_sr": gt_sr, "gt_asp": gt_asp,
    }


def apply_filter(fname, rec, tau, env):
    # PL 후보 전제: 코너 >=6 검출 (검출 안 된 프레임은 어떤 필터든 PL 불가).
    if rec["n_det"] < N_DET_MIN:
        return False
    s = rec["scores"]
    if fname == "f1_peak":
        return s["f1_peak"] is not None and s["f1_peak"] >= tau
    if fname == "f2_peak_ratio":
        return s["f2_peak_ratio"] is not None and s["f2_peak_ratio"] >= tau
    if fname == "f3_flip":
        return s["f3_flip"] is not None and s["f3_flip"] <= tau
    if fname == "f4_tta_stab":
        return s["f4_tta_stab"] is not None and s["f4_tta_stab"] <= tau
    if fname == "f5_rear_conf":
        return s["f5_rear_conf"] is not None and s["f5_rear_conf"] >= tau
    if fname == "f6_frsep":
        return s["f6_frsep"] is not None and s["f6_frsep"] >= tau
    if fname == "f7_posdepth":
        return rec["f7_posdepth"]
    if fname == "f8_size_env":
        if rec["pred_sr"] is None or rec["pred_asp"] is None:
            return False
        es = excursion(rec["pred_sr"], env["size_lo"], env["size_hi"])
        ea = excursion(rec["pred_asp"], env["asp_lo"], env["asp_hi"])
        return max(es, ea) == 0.0
    if fname == "f9_bbox_iou":
        return s["f9_bbox_iou"] is not None and s["f9_bbox_iou"] >= tau
    return False


# ── overlay ───────────────────────────────────────────────────────────
def draw_cuboid(img, pts8, centroid, col, r, dot_thick, label_off, num=True):
    pts = np.asarray(pts8, float)
    ok = ~np.isnan(pts[:, 0])
    for a, b in EDGES:
        if ok[a] and ok[b]:
            cv2.line(img, (int(pts[a, 0]), int(pts[a, 1])),
                     (int(pts[b, 0]), int(pts[b, 1])), col, 1, cv2.LINE_AA)
    for i in range(8):
        if not ok[i]:
            continue
        p = (int(pts[i, 0]), int(pts[i, 1]))
        cv2.circle(img, p, r, col, dot_thick, cv2.LINE_AA)
        if num:
            tc = FRONT_TXT if i < 4 else REAR_TXT
            cv2.putText(img, str(i), (p[0] + label_off, p[1] - label_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, tc, 1, cv2.LINE_AA)
    if centroid is not None and not (isinstance(centroid, float) and np.isnan(centroid)):
        c = np.asarray(centroid, float)
        if not np.isnan(c[0]):
            cv2.drawMarker(img, (int(c[0]), int(c[1])), col, cv2.MARKER_CROSS, 10, 2)


def save_overlay(rec, fname, tau, out_path):
    img = cv2.imread(rec["ip"])
    if img is None:
        return
    draw_cuboid(img, rec["gt8"], rec["gtc"], GT_COL, 6, 2, 7, num=False)
    draw_cuboid(img, np.array(rec["pred8"], float), rec["pred_c"], PRED_COL, 4, -1, 6)
    s = rec["scores"].get(fname)
    if fname == "f7_posdepth":
        sval = "OK"
    elif fname == "f8_size_env":
        sval = f"sr={rec['pred_sr']:.3f} asp={rec['pred_asp']:.2f}" if rec["pred_sr"] else "n/a"
    else:
        sval = f"{s:.2f}" if s is not None else "n/a"
    cm = f"{rec['corner_med']:.1f}" if rec["corner_med"] is not None else "n/a"
    good = "GOOD" if rec["good"] else "BAD"
    h1 = f"{fname} tau={tau}  val={sval}  PASS"
    h2 = (f"V={rec['v_geom']} det={rec['n_det']}/8  corner_med={cm}px [{good}]"
          f"  GT=green S1=red")
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, h1, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, h2, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = []
    import glob
    for jp in sorted(glob.glob(os.path.join(CAD_DIR, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(CAD_DIR, fid + ".png")
        if os.path.exists(ip):
            frames.append((jp, ip))
    print(f"[set] cad N={len(frames)}  weights=Paper-S1  pad={PAD} (near-field, NOT official)")

    model = E.load_model(WEIGHTS, device)
    recs = []
    for i, (jp, ip) in enumerate(frames):
        r = process(model, jp, ip, device)
        if r is not None:
            recs.append(r)
        print(f"  [{i+1}/{len(frames)}] {r['fid'] if r else 'FAIL'} "
              f"det={r['n_det'] if r else '-'} cm={r['corner_med'] if r else '-'}")

    # f8 envelope from GT bboxes (model-independent, [p2.5,p97.5], max GT pass)
    gsr = np.array([r["gt_sr"] for r in recs if r["gt_sr"] is not None])
    gasp = np.array([r["gt_asp"] for r in recs if r["gt_asp"] is not None])
    env = {"size_lo": float(np.percentile(gsr, 2.5)),
           "size_hi": float(np.percentile(gsr, 97.5)),
           "asp_lo": float(np.percentile(gasp, 2.5)),
           "asp_hi": float(np.percentile(gasp, 97.5))}

    N = len(recs)
    n_good_total = sum(1 for r in recs if r["good"])
    n_det_total = sum(1 for r in recs if r["n_det"] >= N_DET_MIN)

    # ── build pass matrix + per-filter stats at canonical tau ──────────
    matrix = {}       # fname -> {fid: pass}
    filter_stats = {}
    for fname in FILTER_ORDER:
        tau = TAU.get(fname)
        passed = []
        for r in recs:
            p = apply_filter(fname, r, tau, env)
            matrix.setdefault(r["fid"], {})[fname] = bool(p)
            if p:
                passed.append(r)
        n_pass = len(passed)
        n_pass_good = sum(1 for r in passed if r["good"])
        prec = round(n_pass_good / n_pass, 3) if n_pass else None
        filter_stats[fname] = {
            "tau": tau, "n_pass": n_pass, "n_pass_good": n_pass_good,
            "precision": prec,
            "passed_fids": [r["fid"] for r in passed],
        }

    # ── sweep (score-based only) ───────────────────────────────────────
    sweep_stats = {}
    for fname, taus in SWEEP.items():
        sweep_stats[fname] = {}
        for tau in taus:
            passed = [r for r in recs if apply_filter(fname, r, tau, env)]
            ng = sum(1 for r in passed if r["good"])
            sweep_stats[fname][str(tau)] = {
                "n_pass": len(passed),
                "precision": round(ng / len(passed), 3) if passed else None,
                "n_pass_good": ng}

    # ── save overlays per filter ───────────────────────────────────────
    for fname in FILTER_ORDER:
        fdir = os.path.join(OUT_DIR, fname)
        os.makedirs(fdir, exist_ok=True)
        for old in glob.glob(os.path.join(fdir, "*")):
            os.remove(old)   # clear stale overlays from prior runs
        tau = TAU.get(fname)
        passed = [r for r in recs if matrix[r["fid"]][fname]]
        if not passed:
            open(os.path.join(fdir, "EMPTY_no_frame_passed.txt"), "w").write(
                f"{fname} tau={tau}: 0/{N} frames passed.\n")
            continue
        for r in sorted(passed, key=lambda r: (not r["good"], r["fid"])):
            cm = int(round(r["corner_med"])) if r["corner_med"] is not None else -1
            g = "good" if r["good"] else "bad"
            fn = f"{g}_cm{cm}_{r['fid']}.jpg"
            save_overlay(r, fname, tau, os.path.join(fdir, fn))

    # ── filter_results.json ────────────────────────────────────────────
    out = {
        "weights": WEIGHTS, "pad": PAD, "note_pad": "reflect-pad100 (near-field 검출 확보용, NOT official eval)",
        "n_frames": N, "n_detected(>=6)": n_det_total, "n_good_total": n_good_total,
        "good_px": GOOD_PX, "taus": TAU, "size_envelope": env,
        "filter_desc": FILTER_DESC,
        "filter_stats": filter_stats,
        "sweep": sweep_stats,
        "pass_matrix": matrix,
        "per_frame": [{"fid": r["fid"], "v_geom": r["v_geom"], "n_det": r["n_det"],
                       "corner_med": r["corner_med"], "good": r["good"],
                       "scores": r["scores"], "f7_posdepth": r["f7_posdepth"]}
                      for r in recs],
    }
    with open(os.path.join(OUT_DIR, "filter_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)

    # ── summary.md ─────────────────────────────────────────────────────
    L = []
    L.append("# Paper-S1 on cad(22): 9 PL accept filters (each INDIVIDUAL)")
    L.append("")
    L.append(f"- weights: `{os.path.relpath(WEIGHTS, ROOT)}`")
    L.append(f"- data: cad manual GT, N={N} | detected(>=6 corner)={n_det_total} "
             f"| GT-good(corner_med<10px)={n_good_total}/{N}")
    L.append(f"- inference: reflect-pad{PAD} (near-field 검출 확보; NOT official eval)")
    L.append("- precision = 통과 프레임 중 corner_med<10px 비율 (order-free Hungarian median)")
    L.append("- 모든 필터 공통 전제: 코너>=6 검출 (PL 후보). 검출 안 된 프레임은 통과 불가.")
    L.append("- ★ N=22 극소 + cad=paper-track S1 엔 unseen(도메인갭). 과결론 금지.")
    L.append("")
    L.append("```")
    L.append(f"{'filter':<15}{'tau':>7}{'n_pass':>8}{'pass%':>7}"
             f"{'good':>6}{'precision':>11}   desc")
    L.append("-" * 92)
    for fname in FILTER_ORDER:
        st = filter_stats[fname]
        tau = st["tau"]
        taus = f"{tau}" if tau is not None else "-"
        pp = round(100 * st["n_pass"] / N, 1)
        prec = st["precision"]
        precs = f"{prec:.3f}" if prec is not None else "  n/a"
        L.append(f"{fname:<15}{taus:>7}{st['n_pass']:>8}{pp:>6.0f}%"
                 f"{st['n_pass_good']:>6}{precs:>11}   {FILTER_DESC[fname]}")
    L.append("```")
    L.append("")
    L.append("## Sweep (score-based filters)")
    L.append("```")
    for fname in SWEEP:
        L.append(f"{fname}:")
        for tau, s in sweep_stats[fname].items():
            precs = f"{s['precision']:.3f}" if s["precision"] is not None else "n/a"
            L.append(f"    tau={tau:<6} n_pass={s['n_pass']:<3} "
                     f"good={s['n_pass_good']:<3} precision={precs}")
    L.append("```")
    L.append("")
    L.append("## 판정")
    # verdict heuristic
    useful = [(f, filter_stats[f]) for f in FILTER_ORDER
              if filter_stats[f]["n_pass"] > 0 and (filter_stats[f]["precision"] or 0) >= 0.7]
    if useful:
        L.append("- 통과 있고 precision>=0.7 인 필터:")
        for f, st in useful:
            L.append(f"  - {f}: n_pass={st['n_pass']} precision={st['precision']}")
    else:
        L.append("- precision>=0.7 로 통과시키는 단일 필터 없음.")
    if n_good_total == 0:
        L.append(f"- ★ 근본 원인: base(S1)가 cad 22장 중 corner_med<10px 프레임 0개 "
                 f"(검출>=6 도 {n_det_total}/{N}뿐, best corner_med≈11px). "
                 f"good PL 자체가 존재하지 않으므로 어떤 필터도 precision>0 불가 — "
                 f"필터 실패가 아니라 base 천장 문제. (memory: 필터 천장=base 코너 정확도)")
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    # ── index.txt ──────────────────────────────────────────────────────
    idx = ["# s1_cad_9filters — pass matrix (rows=frame, cols=f1..f9)  P=pass .=fail",
           f"# N={N} pad={PAD}(near-field) | good=corner_med<10px",
           "# " + " ".join(f"{f.split('_')[0]}" for f in FILTER_ORDER),
           ""]
    hdr = f"{'fid':<22}{'V':>2}{'det':>4}{'cm':>7}{'good':>5}  " + \
          "".join(f"{f.split('_')[0]:>4}" for f in FILTER_ORDER)
    idx.append(hdr)
    for r in recs:
        cm = f"{r['corner_med']:.1f}" if r["corner_med"] is not None else "n/a"
        row = (f"{r['fid']:<22}{r['v_geom']:>2}{r['n_det']:>4}{cm:>7}"
               f"{('Y' if r['good'] else 'n'):>5}  ")
        row += "".join(f"{('P' if matrix[r['fid']][f] else '.'):>4}"
                       for f in FILTER_ORDER)
        idx.append(row)
    idx.append("")
    for fname in FILTER_ORDER:
        st = filter_stats[fname]
        idx.append(f"# {fname:<15} n_pass={st['n_pass']:<3} precision={st['precision']}")
    with open(os.path.join(OUT_DIR, "index.txt"), "w") as f:
        f.write("\n".join(idx) + "\n")

    print("\n" + "\n".join(L))
    print(f"\n[save] {OUT_DIR}/filter_results.json , summary.md , index.txt")
    print(f"[save] per-filter overlays: {OUT_DIR}/f*/")


if __name__ == "__main__":
    main()
