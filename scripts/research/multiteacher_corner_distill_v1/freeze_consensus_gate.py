"""Gate C 의 real 합의 임계를 SOURCE_DEV 에서만 정한다 — real GT 미사용.

규칙 (결과를 보기 전에 문장으로 고정)
  불일치      d_k = 교사 좌표 쌍거리의 최댓값 / R0 box 대각선     (해상도 불변)
  합의 오차   e_k = ||성분별 median 좌표 - GT|| / R0 box 대각선
  임계        tau = "d_k <= tau 인 keypoint 들의 e_k p90 <= 0.02" 를 만족하는 가장 큰 d

0.02 는 box 대각선의 2% 다.  합성 val 은 해상도가 840~1160 px 로 섞여 있어
px 절대값으로 자르면 real 로 옮길 때 깨진다.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M
import mtcd_teachers as T

SYN = M.REPO_ROOT / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
ERROR_FRACTION_CRITERION = 0.02
N_FRAMES = 250


def syn_frame(stem):
    img = cv2.imread(str(SYN / "images/val" / f"{stem}.png"))
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    v = list(map(float, (SYN / "labels/val" / f"{stem}.txt").read_text()
                 .split("\n")[0].split()[5:]))
    return (img, np.array([[v[3 * i] * w, v[3 * i + 1] * h] for i in range(9)]),
            np.array([v[3 * i + 2] for i in range(9)]))


def main() -> int:
    registry = json.loads((M.TRACK / "TEACHER_REGISTRY.json").read_text())["teachers"]
    stems = sorted(p.stem for p in (SYN / "images/val").glob("*.png"))
    random.Random(20260906).shuffle(stems)
    stems = stems[:N_FRAMES]

    cache = {}
    for tid, spec in registry.items():
        weights = M.REPO_ROOT / spec["checkpoint"]
        if M.sha256_file(weights) != spec["sha256"]:
            raise SystemExit(f"{tid}: sha mismatch")
        model = (T.load_dope(weights) if spec["kind"] == "dope" else T.load_yolo(weights))
        per_frame = {}
        for stem in stems:
            img, _, _ = syn_frame(stem)
            if img is None:
                continue
            per_frame[stem] = (T.infer_dope(model, img, already_padded=True)
                               if spec["kind"] == "dope"
                               else T.infer_yolo(model, img, already_padded=True))
        cache[tid] = per_frame
        del model
        print(f"  cached {tid}")

    tids = list(registry)
    rows = []
    for stem in stems:
        img, gt, vis = syn_frame(stem)
        if img is None:
            continue
        ref = cache["T0_R0_YOLO26N_G38LEGACY"].get(stem)
        if not ref or ref.get("status") != "OK":
            continue
        bx = ref["box_xyxy"]
        diag = float(np.hypot(bx[2] - bx[0], bx[3] - bx[1]))
        if diag <= 1:
            continue
        pts = {}
        for tid in tids:
            e = cache[tid].get(stem)
            pts[tid] = (np.asarray(e["keypoints_xy"], float)
                        if e and e.get("status") == "OK" and e.get("keypoints_xy") else None)
        for k in range(8):
            if vis[k] <= 0:
                continue
            P = np.array([pts[t][k] for t in tids
                          if pts[t] is not None and np.isfinite(pts[t][k]).all()])
            if len(P) < 3:
                continue
            d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
            iu = np.triu_indices(len(P), 1)
            rows.append({"kp": k, "n_teachers": len(P),
                         "disagree": float(d[iu].max()) / diag,
                         "consensus_err": float(np.linalg.norm(np.median(P, axis=0) - gt[k])) / diag,
                         "r0_err": float(np.linalg.norm(pts["T0_R0_YOLO26N_G38LEGACY"][k] - gt[k])) / diag})

    dis = np.array([r["disagree"] for r in rows])
    err = np.array([r["consensus_err"] for r in rows])
    r0e = np.array([r["r0_err"] for r in rows])

    curve = []
    for tau in np.concatenate([np.arange(0.005, 0.101, 0.005), np.arange(0.12, 0.51, 0.02)]):
        m = dis <= tau
        if m.sum() < 30:
            continue
        curve.append({"tau": float(tau), "coverage": float(m.mean()), "n": int(m.sum()),
                      "consensus_err_p90": float(np.percentile(err[m], 90)),
                      "consensus_err_median": float(np.median(err[m])),
                      "r0_err_p90": float(np.percentile(r0e[m], 90))})
    passing = [c for c in curve if c["consensus_err_p90"] <= ERROR_FRACTION_CRITERION]
    chosen = max(passing, key=lambda c: c["tau"]) if passing else None

    report = {
        "schema_version": "mtcd_consensus_gate_v1",
        "population": "SOURCE_DEV synthetic val — real GT 미사용",
        "n_frames": len(stems), "n_keypoints": len(rows),
        "disagreement_definition": "max pairwise teacher distance / R0 box diagonal",
        "consensus_definition": "component-wise median across teachers",
        "criterion": f"p90(consensus error / box diagonal) <= {ERROR_FRACTION_CRITERION}",
        "rule": "그 기준을 만족하는 가장 큰 tau 를 고른다. 결과를 보고 기준을 바꾸지 않는다.",
        "curve": curve,
        "TAU_CONSENSUS_NORMALISED": chosen["tau"] if chosen else None,
        "coverage_at_tau": chosen["coverage"] if chosen else None,
        "consensus_err_p90_at_tau": chosen["consensus_err_p90"] if chosen else None,
        "status": "FROZEN" if chosen else "NO_TAU_SATISFIES_CRITERION",
    }
    M.GATE_C.mkdir(parents=True, exist_ok=True)
    out = M.GATE_C / "CONSENSUS_GATE.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nn keypoints {len(rows)}")
    print(f"{'tau':>7}{'coverage':>10}{'cons p90':>10}{'cons med':>10}{'R0 p90':>9}")
    for c in curve[::2]:
        print(f"{c['tau']:7.3f}{c['coverage']:10.3f}{c['consensus_err_p90']:10.4f}"
              f"{c['consensus_err_median']:10.4f}{c['r0_err_p90']:9.4f}")
    print(f"\nTAU = {report['TAU_CONSENSUS_NORMALISED']}  coverage {report['coverage_at_tau']}  "
          f"status {report['status']}")
    print(f"-> {out.relative_to(M.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
