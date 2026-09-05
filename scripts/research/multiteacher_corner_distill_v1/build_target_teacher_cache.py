"""TARGET_UNLABELED 1,000 장에 교사 7종을 추론해 캐시한다. real GT 사용 0.

출력 = gate_c_local_specialist/TARGET_TEACHER_CACHE.json
각 프레임에 교사별 9 keypoint 와, METHOD_LOCK 의 합의 임계로 판정한 usable/abstain.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M
import mtcd_teachers as T

POOL = M.REPO_ROOT / "data/evaluation/pallet_eval_v1/adaptation/MAIN_UNLABELED_BALANCED.csv"
REF = "T0_R0_YOLO26N_G38LEGACY"


def main() -> int:
    gate = json.loads((M.GATE_C / "CONSENSUS_GATE.json").read_text())
    if gate["status"] != "FROZEN":
        raise SystemExit("consensus gate is not FROZEN")
    tau = float(gate["TAU_CONSENSUS_NORMALISED"])
    registry = json.loads((M.TRACK / "TEACHER_REGISTRY.json").read_text())["teachers"]
    rows = list(csv.DictReader(POOL.open()))
    print(f"pool {len(rows)}  tau {tau}")

    per_teacher = {}
    for tid, spec in registry.items():
        weights = M.REPO_ROOT / spec["checkpoint"]
        if M.sha256_file(weights) != spec["sha256"]:
            raise SystemExit(f"{tid}: sha mismatch")
        model = T.load_dope(weights) if spec["kind"] == "dope" else T.load_yolo(weights)
        started = time.time()
        out = {}
        for row in rows:
            image = cv2.imread(str(M.REPO_ROOT / row["image_path"]))
            if image is None:
                out[row["image_sha256"]] = {"status": "IMAGE_MISSING"}
                continue
            out[row["image_sha256"]] = (T.infer_dope(model, image) if spec["kind"] == "dope"
                                        else T.infer_yolo(model, image))
        per_teacher[tid] = out
        print(f"  {tid:26} {time.time()-started:6.1f}s")
        del model

    tids = list(registry)
    frames, n_usable, n_total = {}, 0, 0
    for row in rows:
        sha = row["image_sha256"]
        ref = per_teacher[REF].get(sha)
        if not ref or ref.get("status") != "OK":
            continue
        bx = ref["box_xyxy"]
        diag = float(np.hypot(bx[2] - bx[0], bx[3] - bx[1]))
        if diag <= 1:
            continue
        pts = {}
        for tid in tids:
            e = per_teacher[tid].get(sha)
            pts[tid] = (np.asarray(e["keypoints_xy"], float).tolist()
                        if e and e.get("status") == "OK" and e.get("keypoints_xy") else None)
        kps = []
        for k in range(8):
            P = np.array([pts[t][k] for t in tids
                          if pts[t] is not None and np.isfinite(pts[t][k]).all()])
            n_total += 1
            if len(P) < 3:
                kps.append({"kp": k, "usable": False, "reason": "fewer than three teachers"})
                continue
            d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
            disagree = float(d[np.triu_indices(len(P), 1)].max()) / diag
            usable = disagree <= tau
            n_usable += usable
            kps.append({"kp": k, "usable": bool(usable),
                        "reason": "ok" if usable else "disagreement above tau",
                        "disagreement_normalised": disagree,
                        "teacher_xy": P.tolist(),
                        "consensus_median_xy": np.median(P, axis=0).tolist()})
        frames[sha] = {"image_path": row["image_path"],
                       "capture_session": row["capture_session"],
                       "paper_condition": row["paper_condition"],
                       "r0_box_xyxy": bx, "r0_box_diagonal": diag,
                       "r0_keypoints_xy": pts[REF], "keypoints": kps}

    payload = {"schema_version": "mtcd_target_teacher_cache_v1",
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "pool_manifest": str(POOL.relative_to(M.REPO_ROOT)),
               "pool_manifest_sha256": M.sha256_file(POOL),
               "teachers": tids, "tau_consensus_normalised": tau,
               "real_gt_used": 0,
               "n_pool": len(rows), "n_frames_with_r0": len(frames),
               "n_keypoint_slots": n_total, "n_usable": int(n_usable),
               "usable_rate": n_usable / n_total if n_total else 0.0,
               "frames": frames}
    out = M.GATE_C / "TARGET_TEACHER_CACHE.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nframes with R0 {len(frames)}/{len(rows)}   usable keypoints "
          f"{n_usable}/{n_total} = {payload['usable_rate']:.4f}")
    print(f"-> {out.relative_to(M.REPO_ROOT)}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
