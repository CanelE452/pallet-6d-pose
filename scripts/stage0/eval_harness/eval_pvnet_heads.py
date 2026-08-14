"""eval_pvnet_heads.py — STEP 4 PVNet 헤드 decode + 평가 하니스.

각 프레임에서 두 종류 키포인트 세트를 decode 하여 GT 와 비교:
  1) heatmap kp : extract_keypoints_from_belief(belief)  (= multitask 효과)
  2) voting  kp : vec 헤드로 voting  (= 벡터 직접 효과)
       offset: 마스크 픽셀 (x,y)+pred_offset_k 의 robust median.
       unit  : PVNet RANSAC voting — 마스크 픽셀 단위벡터로 키포인트별 가설
               (픽셀쌍 직선 교점) 샘플, angle inlier 최다 가설 채택.

★ voting 추론 마스크 = heatmap 에서 유도 (사용자 결정 "둘 다 단계적"의 1단계):
   검출된 belief 피크들의 convex hull 을 dilate. 마스크가 heatmap 의존이라
   "voting 독립성"이 약함 — 예비 1단계 타협(결과에 명시).

좌표계: belief / vec 모두 동일 50px 격자 → kp 도 50px. 평가 시 GT(원본px)와 맞추려면
   padding_compare.infer_one 과 동일 전처리/역매핑(여기선 padding 없음=aspect-only)을
   따른다. GT = objects[0]["projected_cuboid"] 8 코너 (원본 px).

비교군: control-heatmap / offset-heatmap / offset-voting / unit-heatmap / unit-voting.
   (heatmap kp = multitask 효과, voting kp = 벡터 직접효과). 평가셋별 분리.

출력: 공백정렬 표 + json → data/pallet/eval_results/stage4_pvnet/.
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

import cv2  # noqa: E402
import torch  # noqa: E402
from models import DopeNetwork  # noqa: E402
from filter_pr_camfacing import (  # noqa: E402
    extract_keypoints_from_belief, hungarian_mean_dist,
)
from tau_calibrate import collect_val_frames  # noqa: E402

GOOD_PX = 10.0
GROSS_PX = 20.0
N_DET_MIN = 6
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
MANUAL_GT_DIR = os.path.join(
    ROOT, "data", "pallet", "eval_results", "stage0_gt_candidates", "manual_gt")
SYN_VAL_DIR = os.path.join(ROOT, "data", "pallet", "training_data", "val")
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "stage4_pvnet")

FRONT = [0, 1, 2, 3]
BACK = [4, 5, 6, 7]


# ── 모델 로드 (numVec 자동 추론, strict=False) ────────────────────────────
def _head_out_channels(state, prefix):
    """state_dict 에서 prefix(m_vec1/m_seg1) 헤드의 마지막 conv out_channels.
    헤드 없으면 0."""
    ks = [k for k in state if k.startswith(prefix) and k.endswith(".weight")]
    if not ks:
        return 0
    return state[sorted(ks)[-1]].shape[0]


def load_pvnet_model(weights_path, device, numVec=None, numSeg=None):
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    if numVec is None:
        numVec = _head_out_channels(state, "m_vec1")
    if numSeg is None:
        numSeg = _head_out_channels(state, "m_seg1")
    model = DopeNetwork(numVec=numVec, numSeg=numSeg)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), numVec, numSeg


# ── 전처리 (padding_compare.infer_one PAD=0 'none' 경로와 동일) ────────────
def preprocess(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ph, pw = rgb.shape[:2]
    sc = 400.0 / min(ph, pw)
    nw = max(8, int(round(pw * sc)) & ~7)
    nh = max(8, int(round(ph * sc)) & ~7)
    t = ((cv2.resize(rgb, (nw, nh)).astype(np.float32) / 255.0 - MEAN) / STD)
    tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, nw, nh, sc


def belief_to_orig(bx, by, bw, bh, nw, nh, sc):
    """belief 격자 좌표 -> 원본 px (padding 없음). infer_one 매핑과 동일."""
    ux, uy = nw / bw, nh / bh
    ox = (bx * ux) / sc
    oy = (by * uy) / sc
    return ox, oy


# ── heatmap 에서 마스크 유도: 검출 피크 convex hull dilate ─────────────────
def heatmap_mask(kps_bel, size, dilate=3):
    pts = np.array([(k[0], k[1]) for k in kps_bel[:8] if k[0] >= 0],
                   dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)
    if len(pts) < 3:
        return mask
    pts[:, 0] = np.clip(pts[:, 0], 0, size - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, size - 1)
    hull = cv2.convexHull(pts.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    if dilate > 0:
        k = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        mask = cv2.dilate(mask, k)
    return mask


# ── GT real mask(mask_rle) -> belief 격자 마스크 ──────────────────────────
def gt_mask_to_belief(jp, bw, bh, nw, nh, sc):
    """v3 json 의 mask_rle(real per-pixel) 를 preprocess 와 동일 좌표계의 belief
    격자(bh,bw) 마스크로 변환. orig(640x480) -> nw×nh(aspect-preserving resize,
    preprocess 와 동일) -> bh×bw(=nh/8, nw/8) area-preserving downsample.
    mask_rle 없으면 None."""
    from utils_pvnet import decode_mask_rle
    d = json.load(open(jp))
    o = d["objects"][0]
    if "mask_rle" not in o:
        return None
    m = decode_mask_rle(o["mask_rle"]).astype(np.float32)   # (H0,W0)
    m_net = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    m_bel = cv2.resize(m_net, (bw, bh), interpolation=cv2.INTER_AREA)
    return (m_bel > 0.5).astype(np.uint8)


# ── seg 헤드 예측 마스크: sigmoid(seg logit) > thr ────────────────────────
def seg_pred_mask(seg_logit, size, thr=0.5):
    """seg_logit (1,H,W) or (H,W) numpy logit -> (size,size) uint8 mask.
    sigmoid>thr. H==size 가정(belief 와 동일 50px 격자)."""
    sl = seg_logit
    if sl.ndim == 3:
        sl = sl[0]
    prob = 1.0 / (1.0 + np.exp(-sl))
    mask = (prob > thr).astype(np.uint8)
    return mask


# ── offset voting: 마스크 픽셀 (x,y)+vec_k 의 robust median ────────────────
def vote_offset(vec, mask):
    """vec (18,H,W), mask (H,W) bool/uint8. -> (9,2) belief 격자 좌표 (NaN if <px)."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return np.full((9, 2), np.nan)
    kp = np.full((9, 2), np.nan)
    for k in range(9):
        dx = vec[2 * k][ys, xs]
        dy = vec[2 * k + 1][ys, xs]
        px = xs + dx
        py = ys + dy
        kp[k, 0] = np.median(px)
        kp[k, 1] = np.median(py)
    return kp


# ── unit RANSAC voting (표준 PVNet) ───────────────────────────────────────
def vote_unit_ransac(vec, mask, n_hyp=128, inlier_thr_deg=2.0, seed=0):
    """단위벡터장 vec (18,H,W), mask. 키포인트별로 픽셀쌍 2개의 단위벡터 직선
    교점을 가설로 샘플, 모든 마스크 픽셀의 (가설방향 vs 예측 단위벡터) angle
    inlier 최다 가설 채택. -> (9,2) belief 좌표 (NaN if 실패)."""
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], axis=1).astype(np.float64)  # (M,2)
    M = len(P)
    kp = np.full((9, 2), np.nan)
    if M < 8:
        return kp
    cos_thr = np.cos(np.deg2rad(inlier_thr_deg))
    for k in range(9):
        d = np.stack([vec[2 * k][ys, xs], vec[2 * k + 1][ys, xs]], axis=1)
        nrm = np.linalg.norm(d, axis=1)
        good = nrm > 1e-6
        if good.sum() < 4:
            continue
        Pg = P[good]
        dg = d[good] / nrm[good, None]   # unit dirs (Mg,2)
        Mg = len(Pg)
        best_inl = -1
        best_pt = None
        for _ in range(n_hyp):
            i, j = rng.integers(0, Mg, size=2)
            if i == j:
                continue
            # 두 직선 교점: p_i + t*d_i = p_j + s*d_j
            A = np.array([[dg[i, 0], -dg[j, 0]],
                          [dg[i, 1], -dg[j, 1]]])
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if abs(det) < 1e-9:
                continue
            b = Pg[j] - Pg[i]
            t = (A[1, 1] * b[0] - A[0, 1] * b[1]) / det
            hyp = Pg[i] + t * dg[i]      # 교점 후보
            # inlier: 각 픽셀->hyp 방향 vs 예측 단위벡터 angle
            to_hyp = hyp[None, :] - Pg
            tn = np.linalg.norm(to_hyp, axis=1)
            ok = tn > 1e-6
            cos = np.einsum("ij,ij->i", to_hyp[ok] / tn[ok, None], dg[ok])
            inl = int((cos >= cos_thr).sum())
            if inl > best_inl:
                best_inl = inl
                best_pt = hyp
        if best_pt is not None:
            kp[k] = best_pt
    return kp


def kp_to_pred8(kp_bel, bw, bh, nw, nh, sc):
    """(9,2) belief 좌표 -> (8,2) 원본 px (centroid 제외, NaN 보존)."""
    pred8 = np.full((8, 2), np.nan)
    for i in range(8):
        if not np.isfinite(kp_bel[i, 0]):
            continue
        ox, oy = belief_to_orig(kp_bel[i, 0], kp_bel[i, 1], bw, bh, nw, nh, sc)
        pred8[i] = (ox, oy)
    return pred8


def heatmap_pred8(kps_bel, bw, bh, nw, nh, sc):
    pred8 = np.full((8, 2), np.nan)
    for i, k in enumerate(kps_bel[:8]):
        if k[0] < 0:
            continue
        ox, oy = belief_to_orig(k[0], k[1], bw, bh, nw, nh, sc)
        pred8[i] = (ox, oy)
    return pred8


# ── front/back 분해 hungarian median ──────────────────────────────────────
def split_metrics(pred8, gt8):
    """전체 hungarian median + front(0-3)/back(4-7) subset hungarian median.
    NaN<6 -> overall inf. subset 은 매칭된 코너의 GT index 기준으로 분류."""
    valid = ~np.isnan(pred8[:, 0])
    res = {"overall": float("inf"), "front": float("inf"),
           "back": float("inf"), "n_det": int(valid.sum())}
    if valid.sum() < N_DET_MIN:
        return res
    from scipy.optimize import linear_sum_assignment
    P = pred8[valid]
    pidx = np.nonzero(valid)[0]
    cost = np.linalg.norm(P[:, None, :] - gt8[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    d = cost[ri, ci]
    res["overall"] = float(np.median(d))
    fr = [d[m] for m, gi in enumerate(ci) if gi in FRONT]
    bk = [d[m] for m, gi in enumerate(ci) if gi in BACK]
    if fr:
        res["front"] = float(np.median(fr))
    if bk:
        res["back"] = float(np.median(bk))
    return res


def collect_manual(n=0):
    out = []
    for jp in sorted(glob.glob(os.path.join(MANUAL_GT_DIR, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(MANUAL_GT_DIR, fid + ".png")
        if os.path.exists(ip):
            out.append(("manual", fid, jp, ip))
    if n > 0:
        out = out[:n]
    return out


def collect_syn(n=0):
    out = []
    for jp in sorted(glob.glob(os.path.join(SYN_VAL_DIR, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(SYN_VAL_DIR, fid + ".png")
        if os.path.exists(ip):
            out.append(("synthetic", fid, jp, ip))
    if n > 0 and n < len(out):
        idx = np.linspace(0, len(out) - 1, n).round().astype(int)
        out = [out[int(i)] for i in sorted(set(idx))]
    return out


def collect_v3(batch_dir, n=0):
    """v3 합성 batch(mask_rle 보유) 의 (json, png) 프레임. GT-mask voting 평가용."""
    out = []
    for jp in sorted(glob.glob(os.path.join(batch_dir, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(batch_dir, fid + ".png")
        if os.path.exists(ip):
            out.append(("v3", fid, jp, ip))
    if n > 0 and n < len(out):
        idx = np.linspace(0, len(out) - 1, n).round().astype(int)
        out = [out[int(i)] for i in sorted(set(idx))]
    return out


def collect_manifest(manifest_path, n=0):
    """testset manifest(도메인 frame_id json_path img_path, # 주석 무시)의 프레임."""
    out = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            dom, fid, jp, ip = parts[0], parts[1], parts[2], parts[3]
            jp = jp if os.path.isabs(jp) else os.path.join(ROOT, jp)
            ip = ip if os.path.isabs(ip) else os.path.join(ROOT, ip)
            if os.path.exists(jp) and os.path.exists(ip):
                out.append((dom, fid, jp, ip))
    if n > 0:
        out = out[:n]
    return out


def load_gt8(jp):
    d = json.load(open(jp))
    return np.array(d["objects"][0]["projected_cuboid"], float)[:8]


def bucket(metric_list, key, occ_filter=None):
    """occ_filter: None(전체) | "full"(occ==8) | "trunc"(0<=occ<8)."""
    sel = metric_list
    if occ_filter == "full":
        sel = [m for m in metric_list if m.get("occ", -1) == 8]
    elif occ_filter == "trunc":
        sel = [m for m in metric_list if 0 <= m.get("occ", -1) < 8]
    n_frames = len(sel)
    vals = [m[key] for m in sel if np.isfinite(m[key])]
    if not vals:
        return {"N": 0, "n_frames": n_frames, "median": None,
                "good": 0, "gross": 0}
    v = np.array(vals)
    return {"N": int(v.size), "n_frames": n_frames,
            "median": round(float(np.median(v)), 1),
            "good": int((v < GOOD_PX).sum()), "gross": int((v > GROSS_PX).sum())}


def run_evalset(model, numVec, vec_mode, frames, threshold, device,
                ransac_hyp, inlier_deg, numSeg=0, use_seg_mask=False,
                seg_thr=0.5):
    """frames -> dict arm -> list of split_metrics. arm 은 모델 능력에 따라.

    voting 마스크 출처:
      - heatmap hull(임시): "{pre}-voting"
      - seg 헤드 예측(numSeg>0 & use_seg_mask): "{pre}-voting-seg"
    둘 다 가능하면 둘 다 평가(마스크 출처 영향 비교)."""
    arms = {}
    has_vec = numVec > 0
    has_seg = numSeg > 0
    do_seg_vote = has_vec and has_seg and use_seg_mask
    # arm 이름은 vec_mode 로 prefix (off→control)
    pre = "control" if vec_mode == "off" else vec_mode
    arms[f"{pre}-heatmap"] = []
    if has_vec:
        arms[f"{pre}-voting"] = []
        arms[f"{pre}-voting-gtmask"] = []   # GT real mask(mask_rle) voting (synthetic only)
    if do_seg_vote:
        arms[f"{pre}-voting-seg"] = []

    def _vote(vec, mask):
        if vec_mode == "unit":
            return vote_unit_ransac(vec, mask, n_hyp=ransac_hyp,
                                    inlier_thr_deg=inlier_deg)
        return vote_offset(vec, mask)

    for dom, fid, jp, ip in frames:
        img = cv2.imread(ip)
        if img is None:
            continue
        gt8 = load_gt8(jp)
        try:
            _occ = int(json.load(open(jp))["objects"][0]
                       .get("num_corners_unoccluded", -1))
        except Exception:
            _occ = -1
        # real GT(occ 필드 없음)는 in-frame corner 수로 visibility 근사.
        # 8 corner 전부 이미지 안 -> V=8(full), 일부 밖 -> V<8(truncation).
        if _occ < 0:
            h0, w0 = img.shape[:2]
            inb = int(np.sum((gt8[:, 0] >= 0) & (gt8[:, 0] < w0)
                             & (gt8[:, 1] >= 0) & (gt8[:, 1] < h0)))
            _occ = inb
        tensor, nw, nh, sc = preprocess(img)
        tensor = tensor.to(device)
        with torch.no_grad():
            out = model(tensor)
        # forward arity: numSeg>0 -> 4-tuple, numVec>0 -> 3-tuple, else 2-tuple
        seg = None
        if has_seg:
            beliefs, _, vec_stages, seg_stages = out
            vec = vec_stages[-1][0].cpu().numpy() if has_vec else None
            seg = seg_stages[-1][0].cpu().numpy()   # (1,H,W) logit
        elif has_vec:
            beliefs, _, vec_stages = out
            vec = vec_stages[-1][0].cpu().numpy()    # (18,H,W)
        else:
            beliefs, _ = out
            vec = None
        belief = beliefs[-1][0].cpu().numpy()       # (9,H,W)
        bh, bw = belief.shape[1], belief.shape[2]
        kps_bel = extract_keypoints_from_belief(belief, threshold)

        # heatmap arm
        ph8 = heatmap_pred8(kps_bel, bw, bh, nw, nh, sc)
        mh = split_metrics(ph8, gt8); mh["occ"] = _occ
        arms[f"{pre}-heatmap"].append(mh)

        # voting arm (heatmap hull mask)
        if has_vec:
            mask = heatmap_mask(kps_bel, bw, dilate=3)
            kp_bel = _vote(vec, mask)
            pv8 = kp_to_pred8(kp_bel, bw, bh, nw, nh, sc)
            m = split_metrics(pv8, gt8); m["occ"] = _occ; arms[f"{pre}-voting"].append(m)
            # GT real-mask voting (synthetic v3 only: mask_rle 보유)
            gtm = gt_mask_to_belief(jp, bw, bh, nw, nh, sc)
            if gtm is not None and gtm.sum() >= 8:
                kp_belg = _vote(vec, gtm)
                pvg8 = kp_to_pred8(kp_belg, bw, bh, nw, nh, sc)
                mg = split_metrics(pvg8, gt8); mg["occ"] = _occ
                arms[f"{pre}-voting-gtmask"].append(mg)

        # voting-seg arm (seg head predicted mask)
        if do_seg_vote:
            smask = seg_pred_mask(seg, bw, thr=seg_thr)
            kp_bel_s = _vote(vec, smask)
            ps8 = kp_to_pred8(kp_bel_s, bw, bh, nw, nh, sc)
            ms = split_metrics(ps8, gt8); ms["occ"] = _occ
            arms[f"{pre}-voting-seg"].append(ms)
    return arms


def fmt_table(title, arm_metrics, by_visibility=False):
    """by_visibility=True 면 arm 별로 all / V=8 / V<8 3 행 출력.
    det = corner-median 유한 frame / 전체 frame (검출률)."""
    lines = [f"# {title}"]
    hdr = (f"{'arm':<22}{'vis':>6}{'N':>5}{'det':>10}{'ov_med':>9}"
           f"{'fr_med':>9}{'bk_med':>9}{'good':>7}{'gross':>7}")
    lines.append(hdr)
    lines.append("─" * len(hdr))
    filters = [("all", None), ("V=8", "full"), ("V<8", "trunc")] \
        if by_visibility else [("all", None)]
    for arm, mlist in arm_metrics.items():
        for vlab, vf in filters:
            ov = bucket(mlist, "overall", vf)
            fr = bucket(mlist, "front", vf)
            bk = bucket(mlist, "back", vf)
            nf = ov["n_frames"]
            det = f"{ov['N']}/{nf}" if nf else "0/0"
            lines.append(
                f"{arm:<22}{vlab:>6}{ov['N']:>5}{det:>10}"
                f"{('-' if ov['median'] is None else ov['median']):>9}"
                f"{('-' if fr['median'] is None else fr['median']):>9}"
                f"{('-' if bk['median'] is None else bk['median']):>9}"
                f"{ov['good']:>7}{ov['gross']:>7}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True,
                    help="ckpt 경로(들). meta_*.json 있으면 vec_mode 자동.")
    ap.add_argument("--vec_mode", nargs="+", default=None,
                    help="weights 와 1:1 (off/offset/unit). 생략 시 meta/state 추론")
    ap.add_argument("--evalset",
                    choices=["manual", "filter-val", "synthetic", "v3",
                             "manifest"],
                    default="manual")
    ap.add_argument("--batch_dir", default=None,
                    help="evalset=v3 일 때 합성 batch 디렉토리(mask_rle 보유)")
    ap.add_argument("--manifest", default=None,
                    help="evalset=manifest 일 때 testset 매니페스트 파일 경로")
    ap.add_argument("--arm_tags", nargs="+", default=None,
                    help="weights 와 1:1 arm prefix(충돌 방지, 예: M0 M1 M2)")
    ap.add_argument("--by_visibility", action="store_true",
                    help="arm 별 all/V=8/V<8 분리 출력")
    ap.add_argument("--domains", nargs="+", default=None,
                    help="filter-val 도메인 필터 (예: outside night). 기본 outside")
    ap.add_argument("--n_frames", type=int, default=0, help="0=전체")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--ransac_hyp", type=int, default=128)
    ap.add_argument("--inlier_deg", type=float, default=2.0)
    ap.add_argument("--use_seg_mask", action="store_true",
                    help="seg 헤드 보유 시 voting 마스크를 seg 예측(sigmoid>thr)으로")
    ap.add_argument("--seg_thr", type=float, default=0.5,
                    help="seg 마스크 임계 (sigmoid)")
    ap.add_argument("--tag", default="", help="출력 파일 접미사")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.evalset == "manual":
        frames = collect_manual(args.n_frames)
    elif args.evalset == "synthetic":
        frames = collect_syn(args.n_frames)
    elif args.evalset == "v3":
        assert args.batch_dir, "evalset=v3 requires --batch_dir"
        frames = collect_v3(args.batch_dir, args.n_frames)
    elif args.evalset == "manifest":
        assert args.manifest, "evalset=manifest requires --manifest"
        frames = collect_manifest(args.manifest, args.n_frames)
    else:
        frames = collect_val_frames()
        doms = args.domains if args.domains else ["outside"]
        frames = [f for f in frames if f[0] in doms]
        if args.n_frames > 0:
            frames = frames[:args.n_frames]
    print(f"[eval] evalset={args.evalset} frames={len(frames)}")

    all_arms = {}
    for wi, w in enumerate(args.weights):
        vm = None
        if args.vec_mode:
            vm = args.vec_mode[wi]
        else:
            meta = os.path.join(os.path.dirname(w),
                                f"meta_{os.path.basename(w)}.json")
            # meta 파일은 보통 meta_{vec_mode}.json — 디렉토리에서 탐색
            for cand in glob.glob(os.path.join(os.path.dirname(w),
                                               "meta_*.json")):
                m = json.load(open(cand))
                if os.path.basename(m.get("final_ckpt", "")) == \
                        os.path.basename(w):
                    vm = m["vec_mode"]
                    break
        model, numVec, numSeg = load_pvnet_model(w, device)
        if vm is None:
            vm = "off" if numVec == 0 else "offset"  # state 만으로는 unit 구분 불가
        print(f"[model] {w} numVec={numVec} numSeg={numSeg} vec_mode={vm} "
              f"use_seg_mask={args.use_seg_mask}")
        arms = run_evalset(model, numVec, vm, frames, args.threshold, device,
                           args.ransac_hyp, args.inlier_deg,
                           numSeg=numSeg, use_seg_mask=args.use_seg_mask,
                           seg_thr=args.seg_thr)
        # arm_tags 로 prefix (여러 weight 의 동일 arm 이름 충돌 방지)
        if args.arm_tags and wi < len(args.arm_tags):
            arms = {f"{args.arm_tags[wi]}:{k}": v for k, v in arms.items()}
        all_arms.update(arms)

    title = (f"STEP4 PVNet heads — evalset={args.evalset} "
             f"frames={len(frames)} (good<{GOOD_PX} gross>{GROSS_PX})")
    table = fmt_table(title, all_arms, by_visibility=args.by_visibility)
    note = ("# ⚠ voting 마스크 = heatmap convex-hull dilate 유도 → voting 독립성 "
            "약함(예비 1단계). heatmap arm=multitask효과, voting arm=벡터직접효과.")
    print("\n" + table + "\n" + note + "\n")

    suf = f"_{args.tag}" if args.tag else ""
    out_dir = (os.path.join(ROOT, "data", "pallet", "eval_results",
                            "stage10_v3mask")
               if args.evalset == "v3" else OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, f"eval_{args.evalset}{suf}.txt")
    with open(out_txt, "w") as f:
        f.write(table + "\n" + note + "\n")
    out_json = os.path.join(out_dir, f"eval_{args.evalset}{suf}.json")
    json.dump({"evalset": args.evalset, "n_frames": len(frames),
               "good_px": GOOD_PX, "gross_px": GROSS_PX,
               "arms": {arm: {"overall": bucket(m, "overall"),
                              "front": bucket(m, "front"),
                              "back": bucket(m, "back")}
                        for arm, m in all_arms.items()}},
              open(out_json, "w"), indent=2)
    print(f"[save] {out_txt}\n[save] {out_json}")


if __name__ == "__main__":
    main()
