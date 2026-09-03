"""SplitLate line branch 를 PAPER_EVAL 319 프레임에 frozen inference 로 돌린다.

    python3 scripts/paper/fast6d_screen_v1b/dump_line_predictions.py \
        --output-dir data/pallet/results/paper_fast6d_screen_v1b --seed 1

new training = 0.  전처리·디코딩은 `ft_f0f3_eval.py` 의 canonical path 를 그대로
가져온다 — 새 resize 규약을 만들지 않는다.

    preprocess_squash -> SplitLate -> DH.decode -> DH.canonical_from_centred

support 는 **예측된 YOLO 코너**에서 만든다.  GT 코너로 만들면 oracle 이 된다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for sub in ("scripts/stage0", "scripts/stage0/paper_s2", "scripts/stage0/multihead",
            "scripts/stage0/line", "scripts/stage0/real_eval", "challenge",
            "scripts/annotate"):
    sys.path.insert(0, str(ROOT / sub))

import cv2                                        # noqa: E402
import numpy as np                                # noqa: E402
import torch                                      # noqa: E402

import paper_s2_real_eval as PRE                  # noqa: E402
import mh_data as MD                              # noqa: E402
import mh_screen as MS                            # noqa: E402
import mh_splitlate as SL                         # noqa: E402
import mh_cigm as CG                              # noqa: E402
import line_feature_capacity_v2 as V2             # noqa: E402
from mh_arms import DH                            # noqa: E402

CLOSURE = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
STEP = 25000


def support_from_grid(grid9):
    """`ft_f0f3_eval.support_from_grid` 와 같은 함수 — 12 구조 edge 의 가시성."""
    grid = np.asarray(grid9, float)[None, :, :]
    _, _, p0, p1, length = V2.gt_lines(grid, CG.EDGES)
    return V2.visible_segments(p0, p1, length)["hit"][0]


def pixels_to_grid(pixels, width, height, grid=50):
    pixels = np.asarray(pixels, float)
    return np.stack([pixels[:, 0] * grid / width, pixels[:, 1] * grid / height], 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(1, 2))
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    lock = json.loads((out_dir / "FAST_6D_SCREEN_V1B_LOCK.json").read_text())
    entry = lock["line_checkpoints"][f"seed{args.seed}"]
    path = ROOT / entry["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise SystemExit(f"checkpoint sha mismatch: {digest} != {entry['sha256']}")

    MS.deterministic()
    _, _, _, features = MS.lattice()
    state = torch.load(path, map_location=MD.DEV, weights_only=False)
    model = SL.SplitLate(state["arm"])
    model.load_state_dict(state["model"])
    model.to(MD.DEV).eval()
    print(f"seed{args.seed} loaded  step {state.get('step')}  sha {digest[:16]}", flush=True)

    manifest = {f["frame_id"]: f for f in
                json.loads((CLOSURE / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]}
    gt_all = json.loads((CLOSURE / "GEOMETRY_RESOLVED_POSE_GT.json").read_text())["frames"]
    predictions = json.loads((CLOSURE / "predictions/R0.json").read_text())["frames"]

    rows, failures = [], []
    for frame_id in gt_all:
        frame = manifest[frame_id]
        image_path = ROOT / frame["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            failures.append({"frame_id": frame_id, "exception_type": "ImageUnreadable",
                             "message": str(image_path), "fallback_reason": "frame skipped"})
            continue
        height, width = image.shape[:2]
        try:
            with torch.no_grad():
                out = model(PRE.preprocess_squash(image).to(MD.DEV), features)
                theta, rho = DH.decode(out["line_scores"], *DH.lattice())
                theta_can, rho_can = DH.canonical_from_centred(theta, rho)
        except Exception as error:
            failures.append({"frame_id": frame_id, "exception_type": type(error).__name__,
                             "message": str(error)[:200],
                             "fallback_reason": "no line prediction for this frame"})
            continue

        pred = predictions.get(frame_id) or {}
        support = None
        if pred.get("status") == "OK" and pred.get("keypoints_xy"):
            keypoints = np.asarray(pred["keypoints_xy"], float)[:9]
            if np.isfinite(keypoints).all():
                support = support_from_grid(pixels_to_grid(keypoints, width, height))

        rows.append({
            "frame_id": frame_id,
            "session_id": frame["session_id"],
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "image_size": [int(width), int(height)],
            "pred_theta_canonical_rad": [float(v) for v in theta_can[0].cpu().numpy()],
            "pred_rho_canonical_grid": [float(v) for v in rho_can[0].cpu().numpy()],
            "support_from_predicted_corners":
                None if support is None else [bool(v) for v in support],
            "n_support": None if support is None else int(np.sum(support)),
        })
        if len(rows) % 100 == 0:
            print(f"  {len(rows)}", flush=True)

    payload = {
        "schema_version": "fast6d_v1b_line_cache_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "checkpoint": entry["path"],
        "checkpoint_sha256": digest,
        "step": STEP,
        "new_training": 0,
        "preprocessing": "paper_s2_real_eval.preprocess_squash (RGB, 400x400, mean/std)",
        "decode": "DH.decode(line_scores, *DH.lattice()) then DH.canonical_from_centred",
        "support_source": "predicted YOLO R0 corners mapped into the canonical 50 grid; "
                          "GT corners are never used",
        "n_frames": len(rows),
        "n_with_support": sum(1 for r in rows if r["n_support"] is not None),
        "support_median": float(np.median([r["n_support"] for r in rows
                                           if r["n_support"] is not None])),
        "exceptions": {"count": len(failures), "records": failures[:50]},
        "frames": rows,
    }
    target = cache / f"line_predictions_seed{args.seed}.json"
    target.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"-> {target.name}  frames {len(rows)}  support median "
          f"{payload['support_median']}  exceptions {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
