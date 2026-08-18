"""정본 real manual GT 에서 DOPE(paper_s1/s2) 와 YOLO26n-pose 를 같은 지표로 비교.

지표 (사용자 지정):
  1. Detection rate  : 유효 corner >= 6 AND PnP 성립 (높을수록 좋음)
  2. 2D kp err <20px : order-free Hungarian corner 오차가 20px 미만인 프레임 비율 (높을수록)
  3. ADD (m)         : mean ||R_gt x + t_gt - (R_pred x + t_pred)|| over 8 corner (낮을수록)

★ 셋 = 정본만. `data/_eval_sets/*combined` 는 구본이라 쓰지 않는다 (CLAUDE.md 금지,
  memory "development set 도 정본에서" — 구본에서 뽑아 판정 4건이 뒤집힌 이력).
  기존 paper_s2_real_eval.py 의 filterval N123 은 그 구본 기반이라 여기서 재구성하지 않는다.

★ EVAL-PARITY (paper_s2_real_eval.py 와 동일, 타협 불가):
  - paper_s2 Stage A/B : anisotropic squash 640x480->400x400, belief(50)->orig x(W/50,H/50)
  - paper_s1 / base_v2 : aspect-preserving no-pad (PAD=0)
  - YOLO               : reflect pad 100 (학습 그대로)
  전처리만 모델별로 다르고, 그 뒤 지표 경로(Hungarian + solve_pose + ADD)는 완전히 동일하다.

★ 누수: YOLO base(stage_a) 와 DOPE paper_* 는 real 을 학습하지 않았다. YOLO ft 는
  pallet11/night01~07/pallet02~05,08/forklift 를 학습했으므로 [FT-SEEN] 으로 표시한다.

★ final-test 4 세션(pallet07/09, night08/09)은 봉인 대상이라 총계와 분리해 낸다.
  참고 측정일 뿐 threshold 튜닝·모델 선택에 쓰지 않는다.

사용:
  python scripts/stage0/paper_s2/canonical_det_kp_add.py
"""
from __future__ import annotations
import os as _os, sys as _sys

_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import glob
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "scripts"))
sys.path[:0] = [os.path.join(ROOT, "scripts", "annotate")]

import cv2  # noqa: E402
import torch  # noqa: E402
import annotate_pnp as APNP  # noqa: E402
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
from eval_pvnet_heads import split_metrics  # noqa: E402

# stage25_paperbase_eval.py 의 ECAD_PATH 는 stage16_truncation_addon/ 을 가리키는데
# 그 폴더는 eval_results/achieve/multi_model_comparison/ 아래로 옮겨졌다(그 스크립트는
# 현재 그대로는 import 실패한다). 실제 위치로 잡는다.
_ECAD = os.path.join(ROOT, "data/pallet/eval_results/achieve/multi_model_comparison/"
                           "stage16_truncation_addon/capturecad_b2_eval/"
                           "eval_capturecad_b2.py")


def _load_E():
    """DOPE 파트 전용 lazy import. YOLO 는 다른 env(pallet-yolo26)에서 도는데
    이 모듈이 DOPE 네트워크를 끌고 오므로 거기서는 import 하지 않는다."""
    spec = importlib.util.spec_from_file_location("ecad", _ECAD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def K_from_json(d):
    ci = d["camera_data"]["intrinsics"]
    return np.array([[ci["fx"], 0, ci["cx"]],
                     [0, ci["fy"], ci["cy"]],
                     [0, 0, 1]], float)


def hungarian(pred, gt):
    """order-free 최소비용 매칭 거리 (E.hungarian 과 같은 역할, env 독립)."""
    from scipy.optimize import linear_sum_assignment
    ok = np.isfinite(pred[:, 0]) & np.isfinite(pred[:, 1])
    if ok.sum() == 0:
        return None
    C = np.linalg.norm(pred[ok][:, None, :] - gt[None, :, :], axis=2)
    r, c = linear_sum_assignment(C)
    return C[r, c]

THRESH, PAD_ASPECT, N_DET_MIN = 0.3, 0, 6
KP_OK_PX = 20.0
YOLO_PAD = 100
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

OUT = os.path.join(ROOT, "data/pallet/eval_results/canonical_det_kp_add")

DOPE_MODELS = [
    ("paper_s2_stageB", "weights/paper_s2_stageB/net_epoch_0057.pth", "squash"),
    ("paper_s2_stageA", "weights/paper_s2/paper_s2_stageA/net_epoch_0042.pth", "squash"),
    ("paper_s1", "weights/paper_s1/paper_s1_maskaux/net_epoch_0065.pth", "aspect"),
    ("paper_base_v2", "weights/paper_base/paper_base_v2/final_net_epoch_0060.pth", "aspect"),
]
YOLO_MODELS = [
    ("yolo26n_base(synth)",
     "challenge/yolo_pose_one_model/final/pallet_yolo26n_pose_640_b32_final.pt"),
    ("yolo26n_ft(real+neg)",
     "challenge/yolo_pose_one_model/runs_ft/ft_a_real157_neg259_synth12k/weights/best.pt"),
]

FINAL_TEST = ("capturepallet07_manual_gt", "capturepallet09_manual_gt",
              "capturenight08_manual_gt", "capturenight09_manual_gt")
SKIP = ("_night_eval_manual_gt",)      # night05/06/07 중복
WOOD = ("wood_pallet",)                # 다른 물체


# ── 프레임 수집 (정본만) ──────────────────────────────────────────────────────
def collect_frames():
    out = []
    for d in sorted(glob.glob(os.path.join(
            ROOT, "challenge/data/01_real/*/*_manual_gt"))):
        name = os.path.basename(d)
        if any(s in name for s in SKIP) or any(s in name for s in WOOD):
            continue
        grp = "final_test" if name in FINAL_TEST else "main"
        for jp in sorted(glob.glob(os.path.join(d, "*.json"))):
            ip = os.path.splitext(jp)[0] + ".png"
            if os.path.exists(ip):
                out.append((name, grp, jp, ip))
    return out


# ── GT ───────────────────────────────────────────────────────────────────────
def load_gt(jp):
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8 = np.array(o["projected_cuboid"], float)[:8]
    T = np.array(o["pose_transform"], float)
    dm = o.get("dimensions_m") or {}
    dims = (float(dm.get("width", 1.1)), float(dm.get("depth", 1.3)),
            float(dm.get("height", 0.11)))
    return d, gt8, T[:3, :3], T[:3, 3], dims


def yaw_deg(R):
    """ψ_pallet (deg). deployment pose6d_adapter 의 psi 와 등가 (2026-08-16 확인).

    이 프로젝트의 object frame(+Y=down, near=-Z)에서는 offset 없이 atan2 가 곧 ψ 다.
    """
    R = np.asarray(R, float).reshape(3, 3)
    return float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))


def wrap180(a):
    return ((a + 180.0) % 360.0) - 180.0


def add_metric(R_gt, t_gt, dims_gt, R_pr, t_pr, dims_pr):
    """ADD (m). GT/pred 각자의 dims 로 8 corner 를 만들어 대응점 거리 평균.

    GT pose_transform 의 object frame 이 make_pallet_keypoints_3d 와 일치함을
    확인했다 (projected_cuboid 재현 0.84px).
    """
    Xg = APNP.make_pallet_keypoints_3d(*dims_gt)[:8]
    Xp = APNP.make_pallet_keypoints_3d(*dims_pr)[:8]
    Pg = (R_gt @ Xg.T).T + t_gt
    Pp = (R_pr @ Xp.T).T + t_pr
    return float(np.mean(np.linalg.norm(Pg - Pp, axis=1)))


# ── 공통 후처리: pred8/pred_c -> row ─────────────────────────────────────────
def finish_row(pred8, pred_c, jp, img_shape):
    d, gt8, R_gt, t_gt, dims_gt = load_gt(jp)
    K = K_from_json(d)
    dists = hungarian(pred8, gt8)
    kp_err = float(np.median(dists)) if dists is not None and len(dists) else np.inf

    kps9 = [None if not np.isfinite(pred8[i, 0]) else
            [float(pred8[i, 0]), float(pred8[i, 1])] for i in range(8)]
    kps9.append(pred_c)
    n_det = sum(1 for k in kps9[:8] if k is not None)

    pnp_ok, add, yaw_e = 0, np.inf, np.inf
    if n_det >= N_DET_MIN:
        try:
            pose = APNP.solve_pose(kps9, K, dims=dims_gt, img_shape=img_shape)
            if pose is not None:
                pnp_ok = 1
                dims_pr = tuple(pose.get("dims", dims_gt))
                R_pr = np.asarray(pose["R"], float)
                add = add_metric(R_gt, t_gt, dims_gt, R_pr,
                                 np.asarray(pose["t"], float).ravel(), dims_pr)
                yaw_e = abs(wrap180(yaw_deg(R_pr) - yaw_deg(R_gt)))
        except Exception:
            pass
    return {"fid": os.path.splitext(os.path.basename(jp))[0],
            "n_det": n_det, "pnp_ok": pnp_ok,
            "det": 1 if (n_det >= N_DET_MIN and pnp_ok) else 0,
            "kp_err": kp_err, "add": add, "yaw_err": yaw_e}


# ── DOPE ─────────────────────────────────────────────────────────────────────
def preprocess_squash(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (400, 400), interpolation=cv2.INTER_LINEAR)
    t = (r.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0)


def dope_pred(E, model, jp, ip, device, mode):
    img = cv2.imread(ip)
    if img is None:
        return None
    H, W = img.shape[:2]
    if mode == "squash":
        with torch.no_grad():
            beliefs, _ = model(preprocess_squash(img).to(device))
        bel = beliefs[-1][0].cpu().numpy()
        kps = extract_keypoints_from_belief(bel, THRESH)
        sx, sy = W / 50.0, H / 50.0
    else:
        r = E.eval_frame(model, jp, ip, device, THRESH, PAD_ASPECT)
        if r is None:
            return None
        pred8 = np.array(r["pred8"], float)
        return finish_row(pred8, r.get("pred_c"), jp, img.shape)

    pred8 = np.full((8, 2), np.nan)
    for i, k in enumerate(kps[:8]):
        if k[0] >= 0:
            pred8[i] = [k[0] * sx, k[1] * sy]
    pred_c = ([float(kps[8][0] * sx), float(kps[8][1] * sy)]
              if kps[8][0] >= 0 else None)
    return finish_row(pred8, pred_c, jp, img.shape)


# ── YOLO ─────────────────────────────────────────────────────────────────────
def yolo_pred(model, jp, ip, conf):
    img = cv2.imread(ip)
    if img is None:
        return None
    inp = cv2.copyMakeBorder(img, YOLO_PAD, YOLO_PAD, YOLO_PAD, YOLO_PAD,
                             cv2.BORDER_REFLECT_101)
    r = model.predict(inp, verbose=False, conf=conf, imgsz=640)[0]
    pred8 = np.full((8, 2), np.nan)
    pred_c = None
    if r.boxes is not None and len(r.boxes):
        b = int(np.argmax(r.boxes.conf.cpu().numpy()))
        kp = r.keypoints.data.cpu().numpy()[b]
        kp[:, 0] -= YOLO_PAD
        kp[:, 1] -= YOLO_PAD
        for i in range(8):
            if kp[i, 2] >= 0.5:
                pred8[i] = kp[i, :2]
        if kp[8, 2] >= 0.5:
            pred_c = [float(kp[8, 0]), float(kp[8, 1])]
    return finish_row(pred8, pred_c, jp, img.shape)


# ── 집계 ─────────────────────────────────────────────────────────────────────
def summarize(rows):
    n = len(rows)
    if not n:
        return None
    det = sum(r["det"] for r in rows)
    dr = [r for r in rows if r["det"]]
    kp = [r["kp_err"] for r in dr if np.isfinite(r["kp_err"])]
    add = [r["add"] for r in dr if np.isfinite(r["add"])]
    yaw = [r["yaw_err"] for r in dr if np.isfinite(r.get("yaw_err", np.inf))]
    return {
        "n": n, "det": det, "det_pct": 100.0 * det / n,
        "kp20_pct": 100.0 * sum(1 for r in dr
                                if np.isfinite(r["kp_err"])
                                and r["kp_err"] < KP_OK_PX) / n,
        "kp_med": float(np.median(kp)) if kp else float("nan"),
        "add_med": float(np.median(add)) if add else float("nan"),
        "add_mean": float(np.mean(add)) if add else float("nan"),
        "yaw_med": float(np.median(yaw)) if yaw else float("nan"),
        "yaw5_pct": 100.0 * sum(1 for v in yaw if v < 5.0) / n if yaw else float("nan"),
        "yaw10_pct": 100.0 * sum(1 for v in yaw if v < 10.0) / n if yaw else float("nan"),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["dope", "yolo", "report"], required=True,
                    help="dope=pallet-pose env / yolo=pallet-yolo26 env / report=합쳐 출력")
    a_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a_.part == "report":
        return report()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = collect_frames()
    main_f = [f for f in frames if f[1] == "main"]
    ft_f = [f for f in frames if f[1] == "final_test"]
    print(f"[set] 정본 manual GT  main={len(main_f)}  final_test(봉인)={len(ft_f)}  "
          f"total={len(frames)}")
    print(f"[dev] {device}   det = corner>={N_DET_MIN} AND PnP 성립   "
          f"kp<{KP_OK_PX:.0f}px = Hungarian median\n")

    results = {}
    E = _load_E() if a_.part == "dope" else None
    for mname, wrel, mode in (DOPE_MODELS if a_.part == "dope" else []):
        wp = os.path.join(ROOT, wrel)
        if not os.path.exists(wp):
            print(f"[skip] {mname}: weight 없음")
            continue
        mdl = E.load_model(wp, device)
        rows = {"main": [], "final_test": []}
        for name, grp, jp, ip in frames:
            r = dope_pred(E, mdl, jp, ip, device, mode)
            if r is not None:
                r["folder"] = name
                rows[grp].append(r)
        results[mname] = rows
        del mdl
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[done] {mname} ({mode})")

    if a_.part == "yolo":
        from ultralytics import YOLO
    for mname, wrel in (YOLO_MODELS if a_.part == "yolo" else []):
        wp = os.path.join(ROOT, wrel)
        if not os.path.exists(wp):
            print(f"[skip] {mname}: weight 없음")
            continue
        mdl = YOLO(wp, task="pose")
        rows = {"main": [], "final_test": []}
        for name, grp, jp, ip in frames:
            r = yolo_pred(mdl, jp, ip, 0.4)
            if r is not None:
                r["folder"] = name
                rows[grp].append(r)
        results[mname] = rows
        print(f"[done] {mname} (yolo pad{YOLO_PAD} conf0.4)")

    with open(os.path.join(OUT, f"part_{a_.part}.json"), "w") as f:
        json.dump({"summary": {m: {g: summarize(results[m][g])
                                   for g in ("main", "final_test")} for m in results},
                   "per_frame": {m: {g: results[m][g] for g in ("main", "final_test")}
                                 for m in results}},
                  f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else float(x))
    print(f"\n[save] {OUT}/part_{a_.part}.json  ({len(results)} models)")


def report():
    """part_dope.json + part_yolo.json 을 합쳐 표로."""
    merged = {}
    for part in ("dope", "yolo"):
        fp = os.path.join(OUT, f"part_{part}.json")
        if not os.path.exists(fp):
            print(f"[warn] {fp} 없음 — 그 파트는 빠진다")
            continue
        merged.update(json.load(open(fp))["summary"])
    for grp, label in (("main", "정본 main (final-test 제외)"),
                       ("final_test", "final-test 4세션 (봉인, 참고용)")):
        print(f"\n=== {label} ===")
        hdr = (f"{'model':<24}{'n':>5}{'det%':>8}{'kp<20px%':>10}"
               f"{'kp med':>9}{'ADD med(m)':>12}{'yaw med':>9}"
               f"{'yaw<5':>8}{'yaw<10':>8}")
        print(hdr)
        print("-" * len(hdr))
        for mname in merged:
            s = merged[mname].get(grp)
            if s is None:
                continue
            print(f"{mname:<24}{s['n']:>5}{s['det_pct']:>7.1f}%"
                  f"{s['kp20_pct']:>9.1f}%{s['kp_med']:>9.2f}"
                  f"{s['add_med']:>12.4f}{s.get('yaw_med', float('nan')):>9.2f}"
                  f"{s.get('yaw5_pct', float('nan')):>7.1f}%"
                  f"{s.get('yaw10_pct', float('nan')):>7.1f}%")

    with open(os.path.join(OUT, "canonical_det_kp_add.json"), "w") as f:
        json.dump({"kp_ok_px": KP_OK_PX, "n_det_min": N_DET_MIN,
                   "summary": merged}, f, indent=2)
    print(f"\n[save] {OUT}/canonical_det_kp_add.json")


if __name__ == "__main__":
    main()
