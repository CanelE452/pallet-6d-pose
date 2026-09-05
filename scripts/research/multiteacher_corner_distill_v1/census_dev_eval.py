"""DEV_EVAL(PAPER_EVAL positive 319) 모집단 인구조사 — 학습 0, 추론 0."""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M

frames = M.dev_eval_frames()
per_index = {"supervised": np.zeros(9, int), "visible": np.zeros(9, int),
             "in_frame": np.zeros(9, int), "annotated_xy": np.zeros(9, int)}
sessions, objects, domains = Counter(), Counter(), Counter()
sup_per_frame, vis_per_frame = [], []
img_sizes = Counter()
image_sha = {}
for f in frames:
    g = M.load_gt(f)
    per_index["supervised"] += g["supervised"]
    per_index["visible"] += g["visible"]
    per_index["in_frame"] += g["in_frame"]
    per_index["annotated_xy"] += np.isfinite(g["xy"]).all(axis=1)
    sessions[g["session_id"]] += 1
    objects[g["object_type"]] += 1
    domains[g["paper_domain"]] += 1
    sup_per_frame.append(int(g["supervised"][:8].sum()))
    vis_per_frame.append(int(g["visible"][:8].sum()))
    img_sizes[g["image_size"]] += 1

out = {
    "population": "PAPER_EVAL_POSITIVE (DEV_EVAL)",
    "n_frames": len(frames),
    "manifest": str(M.MANIFEST_PATH.relative_to(M.REPO_ROOT)),
    "manifest_sha256": M.sha256_file(M.MANIFEST_PATH),
    "frame_order_sha256": json.loads(M.RECIPE_LOCK_PATH.read_text())["population"]["frame_order_sha256"],
    "by_session": dict(sorted(sessions.items())),
    "by_object": dict(objects),
    "by_domain": dict(domains),
    "n_sessions": len(sessions),
    "image_sizes": {f"{w}x{h}": n for (w, h), n in img_sizes.items()},
    "keypoint_counts_by_index": {
        k: v.tolist() for k, v in per_index.items()},
    "corner_totals_0_7": {
        "supervised": int(per_index["supervised"][:8].sum()),
        "visible": int(per_index["visible"][:8].sum()),
        "in_frame": int(per_index["in_frame"][:8].sum()),
        "possible": len(frames) * 8},
    "centroid_index8": {
        "supervised": int(per_index["supervised"][8]),
        "visible": int(per_index["visible"][8])},
    "supervised_corners_per_frame": {
        "median": float(np.median(sup_per_frame)),
        "min": int(min(sup_per_frame)), "max": int(max(sup_per_frame)),
        "hist": {str(k): int(v) for k, v in sorted(Counter(sup_per_frame).items())}},
    "visible_corners_per_frame": {
        "median": float(np.median(vis_per_frame)),
        "min": int(min(vis_per_frame)), "max": int(max(vis_per_frame)),
        "hist": {str(k): int(v) for k, v in sorted(Counter(vis_per_frame).items())}},
}
M.AUDIT.mkdir(parents=True, exist_ok=True)
(M.AUDIT / "DEV_EVAL_CENSUS.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
