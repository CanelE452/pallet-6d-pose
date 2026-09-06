"""self-training 데이터셋용 DiffPnP 사이드카.

두 종류가 섞여 있다.

    replay__G38__G__fNNNN   합성 — 이미 만든 인덱스의 GT 기반 참조를 그대로 쓴다
    pl__<session>__<ts>     실제 pseudo-label — GT 가 없다.  참조 pose 는
                            **PL keypoint 에 SQPnP+RefineLM 을 건 결과**다.
                            평가가 pose 를 읽는 연산과 같은 연산이라, 사용자 요청
                            ("SQPnP 로 학습한 것을 self-train 할 때 PL 로 DiffPnP")
                            의 기준이 된다.

PL 참조는 교사가 낸 좌표에서 읽은 pose 이므로 **교사가 모르는 것을 알려주지 않는다.**
이 항의 역할은 2D 오차를 pose 영향도로 재가중하는 것이다.

    conda run -n pallet-yolo26 python -u \
        scripts/research/diffpnp_yolo_v1/build_selftrain_index.py --arm R5_PROPOSED
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
YOLO = REPO / "challenge/yolo_pose_one_model"
SYN_INDEX = REPO / "data/pallet/results/diffpnp_yolo_v1"
RAW = REPO / "data/pallet/raw_data"

# metric_split_lock.md §3.2 [LOCKED] — 실측 1.10 x 1.30 x 0.12 m.
# 어느 변이 width 인지는 미결이므로 두 가설을 다 풀고 재투영이 낮은 쪽을 쓴다
# (평가의 select_pnp_hypotheses 와 같은 방식).
PALLET_LONG, PALLET_SHORT, PALLET_H = 1.30, 1.10, 0.12

REPLAY_RE = re.compile(r"^replay__(.+)$")   # G38 / P0 / TEX 모두
PL_RE = re.compile(r"^pl__([A-Za-z0-9]+)__(\d+)$")


def cuboid(across, height, along):
    ha, hh, hb = across / 2.0, height / 2.0, along / 2.0
    return np.array([
        [-ha, -hh, -hb], [+ha, -hh, -hb], [+ha, +hh, -hb], [-ha, +hh, -hb],
        [-ha, -hh, +hb], [+ha, -hh, +hb], [+ha, +hh, +hb], [-ha, +hh, +hb],
    ], dtype=np.float64)


HYPOTHESES = {"LONG_ACROSS": cuboid(PALLET_LONG, PALLET_H, PALLET_SHORT),
              "SHORT_ACROSS": cuboid(PALLET_SHORT, PALLET_H, PALLET_LONG)}


def session_intrinsics(session: str, cache: dict) -> np.ndarray | None:
    if session in cache:
        return cache[session]
    hits = list(RAW.glob(f"*/{session}/cam_K.txt"))
    cache[session] = np.loadtxt(hits[0]) if hits else None
    return cache[session]


def pl_entry(label_path: Path, image_path: Path, session: str, cache: dict):
    K = session_intrinsics(session, cache)
    if K is None:
        return None, "no_intrinsics"
    parts = label_path.read_text().split()
    if not parts:
        return None, "empty_label"
    values = np.asarray(parts[1:], np.float64)
    kp = values[4:].reshape(-1, 3)
    image = cv2.imread(str(image_path))
    if image is None:
        return None, "image_missing"
    h, w = image.shape[:2]
    uv = np.column_stack([kp[:8, 0] * w, kp[:8, 1] * h])
    visible = kp[:8, 2] > 0
    if visible.sum() < 6:
        return None, "too_few_visible"

    best = None
    for name, X in HYPOTHESES.items():
        ok, rvec, tvec = cv2.solvePnP(X[visible], uv[visible], K, None,
                                      flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(X[visible], uv[visible], K, None, rvec, tvec)
        projected, _ = cv2.projectPoints(X, rvec, tvec, K, None)
        residual = float(np.linalg.norm(
            projected.reshape(-1, 2)[visible] - uv[visible], axis=1).mean())
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        if float((X @ R.T + t)[:, 2].min()) <= 0.0:
            continue
        if best is None or residual < best[0]:
            best = (residual, name, X, R, t)
    if best is None:
        return None, "pnp_failed"
    residual, name, X, R, t = best
    # 교사 좌표가 기하적으로 안 풀리는 프레임은 참조로 쓰지 않는다.
    if residual > 12.0:                      # metric_split_lock 의 reject 임계와 같은 값
        return None, "reference_reproj_rejected"
    return {"K": K, "X": X, "uv": uv, "R": R, "t": t, "hypothesis": name,
            "residual": residual}, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="R5_PROPOSED")
    args = ap.parse_args()

    dataset = YOLO / "datasets/paper_selftrain_v1" / args.arm
    out_dir = REPO / "data/pallet/results/diffpnp_yolo_v1" / f"selftrain_{args.arm}"
    out_dir.mkdir(parents=True, exist_ok=True)

    syn_stems = json.loads((SYN_INDEX / "diffpnp_index_stems.json").read_text())
    syn = np.load(SYN_INDEX / "diffpnp_index.npz")

    stems, rows, reasons, residuals = [], [], {}, []
    cache: dict = {}
    images = sorted((dataset / "images/train").glob("*.png"))
    print(f"{args.arm}: {len(images)} images", flush=True)
    for image in images:
        stem = image.stem
        m = REPLAY_RE.match(stem)
        if m:
            key = f"train/{m.group(1)}"
            row = syn_stems.get(key, -1)
            if row < 0:
                reasons["replay_not_in_synthetic_index"] = \
                    reasons.get("replay_not_in_synthetic_index", 0) + 1
                continue
            rows.append({"K": syn["K"][row].astype(np.float64),
                         "X": syn["X"][row].astype(np.float64),
                         "uv": syn["uv"][row].astype(np.float64),
                         "R": syn["R"][row].astype(np.float64),
                         "t": syn["t"][row].astype(np.float64),
                         "kind": 0})
            stems.append(f"train/{stem}")
            reasons["replay_ok"] = reasons.get("replay_ok", 0) + 1
            continue
        m = PL_RE.match(stem)
        if not m:
            reasons["unrecognised_stem"] = reasons.get("unrecognised_stem", 0) + 1
            continue
        label = dataset / "labels/train" / f"{stem}.txt"
        entry, why = pl_entry(label, image, m.group(1), cache)
        reasons[f"pl_{why}"] = reasons.get(f"pl_{why}", 0) + 1
        if entry is None:
            continue
        entry["kind"] = 1
        residuals.append(entry["residual"])
        rows.append(entry)
        stems.append(f"train/{stem}")

    if not rows:
        print("사이드카에 넣을 프레임이 없다")
        return 1

    np.savez_compressed(
        out_dir / "diffpnp_index.npz",
        K=np.stack([r["K"] for r in rows]).astype(np.float32),
        X=np.stack([r["X"] for r in rows]).astype(np.float32),
        uv=np.stack([r["uv"] for r in rows]).astype(np.float32),
        R=np.stack([r["R"] for r in rows]).astype(np.float32),
        t=np.stack([r["t"] for r in rows]).astype(np.float32),
        kind=np.asarray([r["kind"] for r in rows], np.int8))
    (out_dir / "diffpnp_index_stems.json").write_text(
        json.dumps({s: i for i, s in enumerate(stems)}))
    meta = {"schema_version": "diffpnp_selftrain_index_v1", "arm": args.arm,
            "n_indexed": len(stems),
            "n_replay": int(sum(1 for r in rows if r["kind"] == 0)),
            "n_pseudo_label": int(sum(1 for r in rows if r["kind"] == 1)),
            "reasons": reasons,
            "pl_reference": "cv2.SOLVEPNP_SQPNP + RefineLM on the pseudo-label keypoints",
            "pl_reference_caveat": "교사가 낸 좌표에서 읽은 pose 다 — 교사가 모르는 것을 알려주지 않는다",
            "pl_dims_m": {"long": PALLET_LONG, "short": PALLET_SHORT, "height": PALLET_H},
            "pl_reference_reproj_px": {
                "median": float(np.median(residuals)) if residuals else None,
                "p90": float(np.percentile(residuals, 90)) if residuals else None},
            "replay_reference": "GT projected_cuboid 로 푼 PnP (합성 인덱스 재사용)"}
    (out_dir / "SELFTRAIN_INDEX_META.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
