"""eval_ab_crop.py — A(padding) vs B(no-padding) truncation robustness 평가.

YOLO26-pose 팔레트 6D 추정에서 화면 밖 keypoint(truncation) 상황의 강건성을
crop 강도(level 0/1/2)별로 A/B 두 학습 방식 정량 비교.

핵심 convention (반드시 유지):
  - keypoint 순서 = annotate.py camera-facing (0~3 near, 4~7 far, 8 centroid).
    GT projected_cuboid 순서와 동일. PnP 3D 모델 = annotate_pnp.make_pallet_keypoints_3d
    (= 1.1 x 1.3 x 0.11 m). 이 순서/모델로 GT pose 와 pred pose 를 모두 풀어 동일 frame.
  - A 모델 추론: 입력 100px reflect pad → predict → keypoint (-100,-100) shift.
  - B 모델 추론: crop 이미지 그대로 predict.
  - cam_K 고정. GT 6D = JSON pose_transform (manual 진실, reproj ~1px). off-screen
    GT keypoint(-1.0 sentinel) 는 invalid 마스킹.

지표 (crop level x {A,B}):
  reproj px(in-screen / all) · PnP 성공률 · ADD mm · 5cm5%

산출:
  - CSV + 콘솔 표
  - A/B 비교 곡선 (truncation vs metric)
  - 대표 frame overlay (GT 녹색 / A 파랑 / B 빨강)

사용:
  conda activate pallet-yolo26
  python challenge/scripts/evaluate/eval_ab_crop.py
  python challenge/scripts/evaluate/eval_ab_crop.py --conf 0.1 --kp_conf 0.5 --out_dir <dir>
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# 2026-08-15 재편: 이 파일이 challenge/scripts/<계열>/ 로 한 단계 내려갔다.
# dirname 을 하나 더 감싸야 저장소 루트다 (계열 -> scripts -> challenge -> repo).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts", "self_training"))

from annotate_pnp import make_pallet_keypoints_3d, PALLET_DIMS  # noqa: E402

# ── 고정 상수 ────────────────────────────────────────────────────────────────
CAM_K = np.array([[614.18, 0, 329.28],
                  [0, 614.31, 234.53],
                  [0, 0, 1]], dtype=np.float64)
IMG_W, IMG_H = 640, 480
PAD = 100  # A 모델 reflect pad 폭 (학습과 동일)

# cuboid wireframe edges (annotate face convention)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),   # near face
         (4, 5), (5, 6), (6, 7), (7, 4),   # far face
         (0, 4), (1, 5), (2, 6), (3, 7)]   # connectors

# 5cm5° threshold
THR_T_M = 0.05
THR_R_DEG = 5.0


# ── GT 로드 ──────────────────────────────────────────────────────────────────
def load_stems(path):
    return [l.strip() for l in open(path)
            if l.strip() and not l.startswith("#")]


def load_gt(stem):
    """원본 GT: keypoints(9,2 + valid mask), pose_transform(R,t), dims."""
    prefix, fname = stem.split("__", 1)
    jp = os.path.join(_REPO, "challenge", "data", prefix, fname + ".json")
    d = json.load(open(jp))
    o = d["objects"][0]
    pc = np.array(o["projected_cuboid"], dtype=np.float64)          # (8,2)
    cen = np.array(o["projected_cuboid_centroid"], dtype=np.float64)  # (2,)
    kps = np.vstack([pc, cen[None, :]])                              # (9,2)
    # -1.0 sentinel = off-screen / not annotated
    valid = ~((kps[:, 0] < -0.5) & (kps[:, 1] < -0.5))
    T = np.array(o["pose_transform"], dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    dm = o["dimensions_m"]
    # stored pose 가 dim-swap(130-front)으로 풀린 케이스 자동 판별: 두 dims 후보 중
    # projected_cuboid 에 reproj 더 잘맞는 dims 선택 (ADD 3D 모델 일관성).
    dims = _best_dims_for_pose(R, t, kps, valid, dm)
    return {"kps": kps, "valid": valid, "R": R, "t": t, "dims": dims}


def _reproj(kp3d, R, t):
    Pc = (R @ kp3d.T).T + t
    z = Pc[:, 2]
    u = CAM_K[0, 0] * Pc[:, 0] / z + CAM_K[0, 2]
    v = CAM_K[1, 1] * Pc[:, 1] / z + CAM_K[1, 2]
    return np.stack([u, v], 1)


def _best_dims_for_pose(R, t, kps, valid, dm):
    cands = [(dm["width"], dm["depth"], dm["height"]),
             (dm["depth"], dm["width"], dm["height"])]
    best, best_e = cands[0], 1e18
    for dims in cands:
        pr = _reproj(make_pallet_keypoints_3d(*dims), R, t)
        e = np.linalg.norm(pr[valid] - kps[valid], axis=1).mean()
        if e < best_e:
            best_e, best = e, dims
    return best


# ── Crop 생성 ────────────────────────────────────────────────────────────────
def make_crops(img, gt):
    """level0=원본, level1=1~2개 off, level2=3~4개 off. 한쪽에서 잘라낸다.

    객체가 가장 가까운 가장자리 방향으로 crop offset 을 키워 화면 밖으로 나가는
    GT keypoint 수를 맞춘다. offset=(ox,oy) → crop_img = img[oy:, ox:] (왼/위 crop).
    crop_GT = orig_GT - offset (음수/초과 유지).
    반환: list of dict {level, img, offset(ox,oy), gt_kps(보정), valid}
    """
    kps, valid = gt["kps"], gt["valid"]
    vis = kps[valid]
    cx = vis[:, 0].mean()
    cy = vis[:, 1].mean()
    # crop 방향: 객체 중심이 가까운 가장자리. left/top crop (offset 양수) 우선,
    # 객체가 오른/아래에 치우치면 그쪽(right/bottom)을 잘라 음...
    # 단순화: 객체를 왼쪽 또는 위로 밀어내는(=left/top crop) 방식. 어느 축이 더
    # 많은 코너를 빠르게 빼는지 자동 선택.
    out = [{"level": 0, "img": img, "offset": (0, 0),
            "gt_kps": kps.copy(), "valid": valid.copy()}]

    def count_off(kp, v):
        on = (kp[:, 0] >= 0) & (kp[:, 0] < IMG_W) & \
             (kp[:, 1] >= 0) & (kp[:, 1] < IMG_H)
        return int((v & ~on).sum())

    # 후보 crop 방향 4종, 각각 offset 을 키우며 목표 off 개수 도달점 탐색
    targets = {1: (1, 2), 2: (3, 4)}
    # 축 선택: keypoint 분포가 넓은 축(잘라서 빨리 빠지는) — 보통 가로
    for level, (lo, hi) in targets.items():
        best = None
        # 방향: left(ox>0), top(oy>0), right(ox<0 → crop from right), bottom
        for axis, sign in [("x", +1), ("x", -1), ("y", +1), ("y", -1)]:
            for off in range(10, 600, 5):
                ox = oy = 0
                if axis == "x":
                    ox = sign * off
                else:
                    oy = sign * off
                k2 = kps.copy()
                k2[:, 0] -= ox
                k2[:, 1] -= oy
                noff = count_off(k2, valid)
                if lo <= noff <= hi:
                    cimg, real_off = _apply_crop(img, ox, oy)
                    # 보정 GT 는 실제 적용 offset 기준
                    k3 = kps.copy()
                    k3[:, 0] -= real_off[0]
                    k3[:, 1] -= real_off[1]
                    cand = {"level": level, "img": cimg, "offset": real_off,
                            "gt_kps": k3, "valid": valid.copy(),
                            "n_off": count_off(k3, valid)}
                    # 객체가 여전히 충분히 화면에 남아있어야 함 (>=4 visible)
                    on = (k3[:, 0] >= 0) & (k3[:, 0] < cimg.shape[1]) & \
                         (k3[:, 1] >= 0) & (k3[:, 1] < cimg.shape[0])
                    if int((valid & on).sum()) >= 4:
                        best = cand
                        break
            if best is not None:
                break
        if best is None:
            # 목표 도달 실패 시 level-1 결과 복제 (graceful)
            prev = out[-1]
            best = {"level": level, "img": prev["img"], "offset": prev["offset"],
                    "gt_kps": prev["gt_kps"].copy(), "valid": valid.copy(),
                    "n_off": count_off(prev["gt_kps"], valid), "degenerate": True}
        out.append(best)
    return out


def _apply_crop(img, ox, oy):
    """ox,oy>0 = left/top crop (offset 양수). <0 = right/bottom crop.

    right/bottom crop 은 이미지 크기를 줄여 객체가 오른/아래 가장자리에서 잘리게 함.
    이 경우 좌표 offset 은 0 (좌상단 기준 그대로) 이지만 이미지가 작아져 객체가
    경계 밖으로. 통일을 위해: 항상 좌상단을 원점으로 유지하고 crop 으로 가장자리 제거.
    """
    h, w = img.shape[:2]
    if ox >= 0 and oy >= 0:
        c = img[oy:, ox:]
        return c, (ox, oy)
    if ox < 0:  # right crop: 오른쪽 |ox| px 제거 → offset (0,0)
        c = img[:, :max(1, w + ox)]
        return c, (0, 0)
    if oy < 0:  # bottom crop
        c = img[:max(1, h + oy), :]
        return c, (0, 0)
    return img, (0, 0)


# ── 추론 ─────────────────────────────────────────────────────────────────────
def predict(model, img, pad, kp_conf, conf, gt_anchor=None):
    """반환: pred_kps(9,2), pred_conf(9,). 검출 실패 시 None.

    gt_anchor (2,): 주어지면 multi-instance 중 centroid 가 gt_anchor 에 가장 가까운
    detection 선택 (instance association — 동일 frame 에 팔레트 2 개 있는 케이스에서
    엉뚱한 객체 선택 방지). None 이면 box conf 최대.
    """
    if pad:
        inp = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT)
    else:
        inp = img
    r = model.predict(inp, verbose=False, conf=conf)[0]
    if r.keypoints is None or len(r.keypoints) == 0:
        return None, None
    allkp = r.keypoints.data.cpu().numpy().astype(np.float64)  # (N,9,3)
    if pad:
        allkp = allkp.copy()
        allkp[:, :, 0] -= PAD
        allkp[:, :, 1] -= PAD
    if allkp.shape[0] == 1:
        bi = 0
    elif gt_anchor is not None:
        cents = allkp[:, 8, :2]  # predicted centroid per instance
        bi = int(np.argmin(np.linalg.norm(cents - gt_anchor[None, :], axis=1)))
    elif r.boxes is not None:
        bi = int(np.argmax(r.boxes.conf.cpu().numpy()))
    else:
        bi = 0
    kp = allkp[bi]
    return kp[:, :2].copy(), kp[:, 2].copy()


SQPNP_MAX_MED_REPROJ = 12.0  # px. SQPnP 풀이 후 median reproj 가 이보다 크면 실패 처리


def solve_pnp(kps_2d, kp_conf, kp_conf_thr, dims):
    """conf 높은 점만 SQPnP 로 직접 풀이. 반환 (ok, R, t, n_used).

    YOLO 추론 경로 전용. DOPE self-training 의 PalletPnPSolver(EPnP+RANSAC)와
    독립. SQPnP(cv2.SOLVEPNP_SQPNP)는 n>=3 지원·비최소·전역최적이라 RANSAC 불필요.
    RANSAC 이 없으므로 풀이 후 reproj median 임계로 outlier 안전장치를 둔다.
    좌표 convention / 3D 모델 / dims / keypoint 순서는 일절 변경하지 않음.
    """
    kp3d = make_pallet_keypoints_3d(*dims)  # (9,3) 기존과 동일 3D 모델
    dist = np.zeros((5, 1), dtype=np.float64)  # 현행과 동일 (distortion 없음)

    obj_pts, img_pts = [], []
    for i in range(9):
        if kp_conf[i] >= kp_conf_thr:
            obj_pts.append(kp3d[i])
            img_pts.append([float(kps_2d[i, 0]), float(kps_2d[i, 1])])
    n = len(obj_pts)
    if n < 6:  # 현행과 동일한 최소 점수 게이트 유지
        return False, None, None, n

    obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, CAM_K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return False, None, None, n

    # 1-step LM 정제 (간단한 안전장치)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_pts, img_pts, CAM_K, dist, rvec, tvec)
    except cv2.error:
        pass

    # outlier 안전장치: 사용한 점들의 median reproj error 가 너무 크면 실패 처리
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, CAM_K, dist)
    med_reproj = float(np.median(
        np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)))
    if med_reproj > SQPNP_MAX_MED_REPROJ:
        return False, None, None, n

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    if t[2] < 0:  # 카메라 뒤로 풀리면 부호 뒤집기 (기존 solver 와 동일 처리)
        t, R = -t, -R
    return True, R, t, n


# ── 지표 ─────────────────────────────────────────────────────────────────────
def add_metric(R_gt, t_gt, dims_gt, R_pr, t_pr, dims_pr):
    """ADD: 동일 3D 모델 점을 양 pose 로 변환한 평균 거리(mm).

    GT/Pred dims 가 다를 수 있어(110 vs 130 front) 공통 비교를 위해 GT dims 모델
    사용 (물리 팔레트 동일). 8 corner 사용.
    """
    P = make_pallet_keypoints_3d(*dims_gt)[:8]
    Xg = (R_gt @ P.T).T + t_gt
    Xp = (R_pr @ P.T).T + t_pr
    return float(np.linalg.norm(Xg - Xp, axis=1).mean()) * 1000.0


def rot_err_deg(R_gt, R_pr):
    Rd = R_gt.T @ R_pr
    c = (np.trace(Rd) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def reproj_err(pred_kps, gt_kps, valid, img_shape):
    """반환 (mean_in, med_in, mean_all, med_all). in = 화면 안 GT 만."""
    h, w = img_shape[:2]
    d = np.linalg.norm(pred_kps - gt_kps, axis=1)
    m_all = valid.copy()
    on = (gt_kps[:, 0] >= 0) & (gt_kps[:, 0] < w) & \
         (gt_kps[:, 1] >= 0) & (gt_kps[:, 1] < h)
    m_in = valid & on
    di = d[m_in] if m_in.any() else np.array([np.nan])
    da = d[m_all] if m_all.any() else np.array([np.nan])
    return (float(np.mean(di)), float(np.median(di)),
            float(np.mean(da)), float(np.median(da)))


# ── overlay ──────────────────────────────────────────────────────────────────
def draw_cuboid(img, kps, color, valid=None, thick=1):
    for a, b in EDGES:
        if valid is not None and (not valid[a] or not valid[b]):
            continue
        pa, pb = kps[a], kps[b]
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 color, thick, cv2.LINE_AA)


def reproj_from_pose(R, t, dims):
    return _reproj(make_pallet_keypoints_3d(*dims), R, t)


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", default=os.path.join(
        _REPO, "challenge", "data", "holdout_stems.txt"))
    ap.add_argument("--img_dir", default=os.path.join(
        _REPO, "challenge", "data", "yolo_pose_manual", "images", "val"))
    ap.add_argument("--weights_a", default=os.path.join(
        _REPO, "runs/pose/challenge/weights/yolo26n_pose_v1_ft_pad_ho/weights/best.pt"))
    ap.add_argument("--weights_b", default=os.path.join(
        _REPO, "runs/pose/challenge/weights/yolo26n_pose_v1_ft_nopad_ho/weights/best.pt"))
    ap.add_argument("--out_dir", default=os.path.join(
        _REPO, "challenge", "data", "ab_crop_eval"))
    ap.add_argument("--conf", type=float, default=0.1, help="detection conf")
    ap.add_argument("--kp_conf", type=float, default=0.5, help="keypoint vis thr for PnP")
    ap.add_argument("--n_overlay", type=int, default=9)
    ap.add_argument("--pad_b", action="store_true",
                    help="B 슬롯 모델도 100px reflect pad 추론 (B가 padding 학습 모델일 때)")
    args = ap.parse_args()

    from ultralytics import YOLO
    os.makedirs(args.out_dir, exist_ok=True)
    ov_dir = os.path.join(args.out_dir, "overlays")
    os.makedirs(ov_dir, exist_ok=True)

    stems = load_stems(args.stems)
    print(f"[load] {len(stems)} stems")
    models = {"A": YOLO(args.weights_a), "B": YOLO(args.weights_b)}
    pad_map = {"A": True, "B": args.pad_b}

    rows = []          # per (stem, level, model)
    overlay_pool = []  # (n_off, stem, level, data) truncation 우선

    for si, stem in enumerate(stems):
        gt = load_gt(stem)
        img = cv2.imread(os.path.join(args.img_dir, stem + ".png"))
        if img is None:
            print("  no img", stem)
            continue
        crops = make_crops(img, gt)
        for cr in crops:
            cimg = cr["img"]
            gk = cr["gt_kps"]
            cv = cr["valid"]
            n_off = cr.get("n_off", int((cv & ~(
                (gk[:, 0] >= 0) & (gk[:, 0] < cimg.shape[1]) &
                (gk[:, 1] >= 0) & (gk[:, 1] < cimg.shape[0]))).sum()))
            per_model = {}
            anchor = gk[8] if cv[8] else gk[cv].mean(axis=0)
            for mk, model in models.items():
                pk, pc = predict(model, cimg, pad_map[mk], args.kp_conf,
                                 args.conf, gt_anchor=anchor)
                row = {"stem": stem, "level": cr["level"], "model": mk,
                       "n_off": n_off, "detected": pk is not None,
                       "reproj_in": np.nan, "reproj_all": np.nan,
                       "reproj_in_med": np.nan, "reproj_all_med": np.nan,
                       "pnp_ok": False, "add_mm": np.nan, "pass5": False}
                if pk is not None:
                    mi, mdi, ma, mda = reproj_err(pk, gk, cv, cimg.shape)
                    row["reproj_in"], row["reproj_in_med"] = mi, mdi
                    row["reproj_all"], row["reproj_all_med"] = ma, mda
                    ok, R, t, nused = solve_pnp(pk, pc, args.kp_conf, gt["dims"])
                    row["n_pnp_pts"] = nused
                    if ok:
                        row["pnp_ok"] = True
                        add = add_metric(gt["R"], gt["t"], gt["dims"],
                                         R, t, gt["dims"])
                        rerr = rot_err_deg(gt["R"], R)
                        terr = float(np.linalg.norm(gt["t"] - t))
                        row["add_mm"] = add
                        row["rot_deg"] = rerr
                        row["t_err_m"] = terr
                        row["pass5"] = (terr < THR_T_M and rerr < THR_R_DEG)
                        per_model[mk] = {"R": R, "t": t, "pk": pk, "pc": pc}
                rows.append(row)
            overlay_pool.append({"n_off": n_off, "stem": stem, "level": cr["level"],
                                 "img": cimg, "gt": gt, "gk": gk, "cv": cv,
                                 "per_model": per_model})
        if (si + 1) % 10 == 0:
            print(f"  [{si+1}/{len(stems)}] processed")

    _write_csv(rows, args.out_dir)
    summary = _summarize(rows)
    _print_table(summary)
    _write_summary_csv(summary, args.out_dir)
    _plot_curves(summary, args.out_dir)
    _make_overlays(overlay_pool, ov_dir, args.n_overlay, gt_dims_key="dims")

    print("\n[done] outputs in", args.out_dir)


def _write_csv(rows, out_dir):
    p = os.path.join(out_dir, "per_frame_results.csv")
    keys = ["stem", "level", "model", "n_off", "detected", "reproj_in",
            "reproj_in_med", "reproj_all", "reproj_all_med", "pnp_ok",
            "n_pnp_pts", "add_mm", "rot_deg", "t_err_m", "pass5"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[csv]", p)


def _summarize(rows):
    out = {}
    for lvl in (0, 1, 2):
        for mk in ("A", "B"):
            sub = [r for r in rows if r["level"] == lvl and r["model"] == mk]
            if not sub:
                continue
            det = [r for r in sub if r["detected"]]
            ri = [r["reproj_in"] for r in det if not np.isnan(r["reproj_in"])]
            ra = [r["reproj_all"] for r in det if not np.isnan(r["reproj_all"])]
            pnp = [r for r in sub if r["pnp_ok"]]
            add = [r["add_mm"] for r in pnp if not np.isnan(r["add_mm"])]
            out[(lvl, mk)] = {
                "n": len(sub),
                "det_rate": len(det) / len(sub),
                "reproj_in_mean": float(np.mean(ri)) if ri else np.nan,
                "reproj_in_med": float(np.median(ri)) if ri else np.nan,
                "reproj_all_mean": float(np.mean(ra)) if ra else np.nan,
                "reproj_all_med": float(np.median(ra)) if ra else np.nan,
                "pnp_rate": len(pnp) / len(sub),
                "add_mean": float(np.mean(add)) if add else np.nan,
                "add_med": float(np.median(add)) if add else np.nan,
                "pass5_rate": sum(r["pass5"] for r in sub) / len(sub),
            }
    return out


def _print_table(summary):
    print("\n" + "=" * 92)
    print(" A(padding) vs B(no-padding) — Truncation Robustness")
    print("=" * 92)
    hdr = (f"{'lvl':>3} {'M':>2} {'n':>3} {'det%':>6} "
           f"{'reproj_in(mean/med)':>20} {'reproj_all(mn/md)':>18} "
           f"{'PnP%':>6} {'ADD(mn/md)mm':>14} {'5cm5°%':>7}")
    print(hdr)
    print("-" * 92)
    for lvl in (0, 1, 2):
        for mk in ("A", "B"):
            s = summary.get((lvl, mk))
            if not s:
                continue
            print(f"{lvl:>3} {mk:>2} {s['n']:>3} "
                  f"{100*s['det_rate']:>5.1f}% "
                  f"{s['reproj_in_mean']:>9.1f}/{s['reproj_in_med']:<9.1f} "
                  f"{s['reproj_all_mean']:>8.1f}/{s['reproj_all_med']:<8.1f} "
                  f"{100*s['pnp_rate']:>5.1f}% "
                  f"{s['add_mean']:>6.0f}/{s['add_med']:<6.0f} "
                  f"{100*s['pass5_rate']:>6.1f}%")
        print("-" * 92)
    print("(mean 은 multi-pallet/심한 occlusion 소수 frame 에 skew → median 이 더 robust)")


def _write_summary_csv(summary, out_dir):
    p = os.path.join(out_dir, "summary_table.csv")
    keys = ["level", "model", "n", "det_rate", "reproj_in_mean", "reproj_in_med",
            "reproj_all_mean", "reproj_all_med", "pnp_rate", "add_mean",
            "add_med", "pass5_rate"]
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for (lvl, mk), s in sorted(summary.items()):
            w.writerow([lvl, mk] + [round(s[k], 4) if isinstance(s[k], float)
                                    else s[k] for k in keys[2:]])
    print("[csv]", p)


def _plot_curves(summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = [0, 1, 2]
    metrics = [
        ("reproj_in_mean", "Reproj (in-screen) [px] ↓", False),
        ("reproj_all_mean", "Reproj (all incl. off) [px] ↓", False),
        ("pnp_rate", "PnP success rate [%] ↑", True),
        ("add_mean", "ADD [mm] ↓", False),
        ("pass5_rate", "5cm5° accuracy [%] ↑", True),
        ("det_rate", "Detection rate [%] ↑", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colA, colB = "#1f77b4", "#d62728"
    for ax, (key, title, pct) in zip(axes.flat, metrics):
        ya = [summary[(l, "A")][key] for l in levels]
        yb = [summary[(l, "B")][key] for l in levels]
        if pct:
            ya = [v * 100 for v in ya]
            yb = [v * 100 for v in yb]
        ax.plot(levels, ya, "o-", color=colA, lw=2.4, ms=9, label="A (padding)")
        ax.plot(levels, yb, "s--", color=colB, lw=2.4, ms=9, label="B (no-pad)")
        for l, va, vb in zip(levels, ya, yb):
            ax.annotate(f"{va:.1f}", (l, va), textcoords="offset points",
                        xytext=(0, 8), fontsize=8, color=colA, ha="center")
            ax.annotate(f"{vb:.1f}", (l, vb), textcoords="offset points",
                        xytext=(0, -14), fontsize=8, color=colB, ha="center")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(levels)
        ax.set_xticklabels(["L0\n(orig)", "L1\n(1-2 off)", "L2\n(3-4 off)"])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("A (padding) vs B (no-padding) — Truncation Robustness\n"
                 "x-axis = crop level (truncation severity increases →)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out_dir, "ab_comparison_curves.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("[plot]", p)


def _make_overlays(pool, ov_dir, n_overlay, gt_dims_key):
    # level 골고루 + 각 level 내 truncation/both-detected 우선
    per_level = max(1, n_overlay // 3)
    picked = []
    for lvl in (2, 1, 0):  # 심한 truncation 먼저 노출
        cand = [x for x in pool if x["level"] == lvl and len(x["per_model"]) == 2]
        cand.sort(key=lambda x: -x["n_off"])
        picked.extend(cand[:per_level])
    # 부족하면 아무거나 채움
    if len(picked) < n_overlay:
        seen = {(x["stem"], x["level"]) for x in picked}
        for x in sorted(pool, key=lambda x: -x["n_off"]):
            if (x["stem"], x["level"]) not in seen and len(x["per_model"]) >= 1:
                picked.append(x)
                seen.add((x["stem"], x["level"]))
            if len(picked) >= n_overlay:
                break
    picked = picked[:n_overlay]
    for item in picked:
        img = item["img"].copy()
        gk, cv = item["gk"], item["cv"]
        draw_cuboid(img, gk, (0, 255, 0), valid=cv, thick=2)  # GT green
        pm = item["per_model"]
        for mk, col in [("A", (255, 80, 0)), ("B", (0, 0, 255))]:  # A blue, B red
            if mk in pm:
                R, t = pm[mk]["R"], pm[mk]["t"]
                pr = reproj_from_pose(R, t, item["gt"]["dims"])
                draw_cuboid(img, pr, col, thick=1)
        txt = f"{item['stem'][:28]} L{item['level']} off={item['n_off']}"
        cv2.putText(img, txt, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "GT=green A=blue B=red", (5, img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        outp = os.path.join(ov_dir, f"L{item['level']}_off{item['n_off']}_"
                            f"{item['stem'][:30]}.png")
        cv2.imwrite(outp, img)
    print(f"[overlay] {len(picked)} saved ->", ov_dir)


if __name__ == "__main__":
    main()
