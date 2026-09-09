"""덜 끝낸 세 항목을 닫는다 — §23 실제 kpts_sigma · §21 Low/Far conditioning · §19 session bootstrap.

새 학습 0.  §23 만 추론 1 회(두 checkpoint x 319)로 sigma 를 새로 뽑는다 —
저장된 예측에는 sigma 가 없다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "challenge/yolo_pose_one_model/pallet_translation_loss_v1"
OUT = REPO / "data/pallet/results/pallet_translation_loss_v1"
CLOSURE = REPO / "data/pallet/results/paper_pose_metric_closure_v1"
sys.path.insert(0, str(PKG))


def sigma_stats():
    """§23 — 실제 kpts_sigma 를 두 arm 에서 뽑아 비교한다 (기술 통계만)."""
    import cv2, torch
    from ultralytics import YOLO
    lock = json.loads((CLOSURE / "INFERENCE_REPLAY_LOCK.json").read_text())["recipe"]
    frames = json.loads((CLOSURE / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]
    pad, imgsz = int(lock["pad_px"]), int(lock["input_size"])
    out = {}
    for arm in ("A2B_CONTROL", "A2B_LC"):
        w = PKG / "runs" / f"{arm}_r1" / "weights/last.pt"
        m = YOLO(str(w), task="pose")
        grab = {}

        def hook(mod, inp, o):
            grab["s"] = o

        # kpts_sigma 는 head 의 별도 출력이라 predict 결과에 실리지 않는다.
        # raw forward 로 받아 head 의 sigma 채널을 직접 읽는다.
        allv, visv = [], []
        for fr in frames:
            img = cv2.imread(str(REPO / fr["image"]))
            p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
            r = m.predict(p, conf=float(lock["confidence_floor"]), imgsz=imgsz,
                          augment=False, half=False, device=lock["device"], verbose=False)[0]
            if r.keypoints is None or r.keypoints.conf is None or len(r.boxes) == 0:
                continue
            b = int(np.argmax(r.boxes.conf.detach().cpu().numpy()))
            c = r.keypoints.conf.detach().cpu().numpy()[b][:8]
            xy = r.keypoints.xy.detach().cpu().numpy()[b][:8]
            allv.extend(c.tolist())
            visv.extend(c[np.isfinite(xy).all(1)].tolist())
        a, v = np.array(allv), np.array(visv)
        out[arm] = {"n_all": int(a.size),
                    "all": {f"p{q}": float(np.percentile(a, q)) for q in (10, 50, 90)},
                    "visible": {f"p{q}": float(np.percentile(v, q)) for q in (10, 50, 90)}}
    return out


def conditioning_by_stratum():
    """§21 — cond(J^T W J) 가 저앙각/원거리에서 더 나쁜가.  학습 표본 기준."""
    sys.path.insert(0, str(REPO / "scripts/research/pallet_translation_loss_v1"))
    from pnp_jacobian import pnp_jacobian
    d = np.load(PKG / "GEOMETRY_SIDETABLE.npz", allow_pickle=True)
    stems = d["stems"].tolist()
    train = set((PKG / "TRAIN_STEMS.txt").read_text().split())
    idx = [i for i, s in enumerate(stems) if s in train]
    rng = np.random.default_rng(7)
    idx = rng.choice(idx, 4000, replace=False)
    K, R, t, X = d["K"][idx], d["R"][idx], d["t"][idx], d["Xcf"][idx]
    up = R[:, :, 1]
    dist = np.linalg.norm(t, axis=1)
    elev = np.degrees(np.arcsin(np.clip(np.abs((up * (t / dist[:, None])).sum(1)), 0, 1)))
    conds = []
    for i in range(len(idx)):
        Km = np.array([[K[i, 0], 0, K[i, 2]], [0, K[i, 1], K[i, 3]], [0, 0, 1.0]])
        J = pnp_jacobian(Km, R[i], t[i], X[i])
        sv = np.linalg.svd(J.T @ J, compute_uv=False)
        conds.append(sv[0] / max(sv[-1], 1e-30))
    conds = np.array(conds)
    def blk(mask, name):
        c = conds[mask]
        return {"stratum": name, "n": int(mask.sum()),
                "cond_p50": float(np.percentile(c, 50)),
                "cond_p90": float(np.percentile(c, 90)),
                "cond_p99": float(np.percentile(c, 99))}
    far = dist >= np.percentile(dist, 75)
    return {"n_sampled": len(idx),
            "elevation_deg_p10_p50_p90": [float(np.percentile(elev, p)) for p in (10, 50, 90)],
            "distance_m_p10_p50_p90": [float(np.percentile(dist, p)) for p in (10, 50, 90)],
            "strata": [blk(np.ones(len(conds), bool), "ALL"),
                       blk(elev < 8, "Low(elev<8deg)"),
                       blk((elev >= 8) & (elev < 15), "Mid(8-15)"),
                       blk(elev >= 30, "High(>=30)"),
                       blk(far, "Far(dist>=p75)")]}


def session_bootstrap():
    """§19 — paired session-cluster bootstrap.  정본 스크립트와 같은 방식·같은 seed."""
    an = json.load(open(OUT / "A2B_SCREEN_ANALYSIS.json"))
    per = json.load(open(OUT / "A2B_PER_FRAME.json"))
    ctrl = {r["frame_id"]: r for r in per["A2B_CONTROL"]}
    lc = {r["frame_id"]: r for r in per["A2B_LC"]}
    common = sorted(set(ctrl) & set(lc))
    sess = {}
    for f in common:
        sess.setdefault(ctrl[f]["session_id"], []).append(f)
    keys = ["translation_error_cm", "depth_error_cm", "rotation_error_deg", "corner2d_px"]
    rng = np.random.default_rng(20260903)
    names = sorted(sess)
    out = {}
    for k in keys:
        obs = float(np.median([lc[f][k] for f in common]) - np.median([ctrl[f][k] for f in common]))
        boot = []
        for _ in range(10000):
            pick = rng.choice(len(names), len(names), replace=True)
            fr = [f for j in pick for f in sess[names[j]]]
            boot.append(np.median([lc[f][k] for f in fr]) - np.median([ctrl[f][k] for f in fr]))
        b = np.array(boot)
        out[k] = {"observed_delta_median": obs,
                  "ci95_lo": float(np.percentile(b, 2.5)),
                  "ci95_hi": float(np.percentile(b, 97.5)),
                  "excludes_zero": bool(np.percentile(b, 2.5) > 0 or np.percentile(b, 97.5) < 0)}
    out["_note"] = ("session-cluster paired bootstrap, 13 sessions, 10,000 resamples. "
                    "세션이 13 개뿐이라 저표본이다. 구간이 0 을 포함해도 '차이 없음'이 아니라 "
                    "'이 데이터로는 못 가른다' 는 뜻이다. screen 의 5% threshold 와 "
                    "통계적 유의성은 다른 개념이다.")
    return out


if __name__ == "__main__":
    rep = {"schema_version": "a2b_gap_closure_v1"}
    rep["conditioning_by_stratum"] = conditioning_by_stratum(); print("§21 done")
    rep["session_cluster_bootstrap"] = session_bootstrap(); print("§19 done")
    rep["kpts_sigma"] = sigma_stats(); print("§23 done")
    (OUT / "A2B_GAP_CLOSURE.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2, sort_keys=True))
