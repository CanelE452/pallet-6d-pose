"""A1 학습셋 29,308 장 전수 — 팔레트가 얼마나 가려져 있나 / 적재물이 있나.

가림 지표 세 가지를 라벨에서 직접 센다.
  visibility          모든 셋에 있는 객체 단위 스칼라
  raycast_visibility  v4/paper_4pallet 계열
  mask_area / hull    paper_4pallet_mask_v1 만 — 실제 보이는 마스크 대 amodal 투영 면적
"""
from __future__ import annotations
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TD = ROOT / "data/pallet/training_data"
OUT = ROOT / "data/pallet/results/hough_line_visual"
SETS = ["mixed_v8_train", "v4_split_base", "aug_squash_v2",
        "aug_trunc_v2", "aug_scale_v2", "paper_4pallet_mask_v1"]


def hull_area(pc):
    p = np.asarray(pc, float)[:8]
    if not np.isfinite(p).all():
        return None
    # convex hull 면적 (shoelace on hull) — amodal 투영 면적의 대용
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(p)
        return float(h.volume)
    except Exception:
        return None


def main():
    rep = {}
    for name in SETS:
        files = sorted(glob.glob(str(TD / name / "**" / "*.json"), recursive=True))
        files = [f for f in files if not Path(f).name.startswith("_")]
        vis, ray, cover = [], [], []
        n_obj_multi = 0
        for i, f in enumerate(files):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            objs = d.get("objects") or []
            if len(objs) != 1:
                n_obj_multi += 1
            if not objs:
                continue
            o = objs[0]
            v = o.get("visibility")
            if isinstance(v, (int, float)):
                vis.append(float(v))
            r = o.get("raycast_visibility")
            if isinstance(r, (int, float)):
                ray.append(float(r))
            ma = o.get("mask_area_px")
            if isinstance(ma, (int, float)) and ma > 0:
                a = hull_area(o.get("projected_cuboid"))
                if a and a > 1:
                    cover.append(min(float(ma) / a, 2.0))
        def q(a):
            if not a:
                return None
            a = np.asarray(a, float)
            return {"n": int(a.size), **{f"p{p}": round(float(np.percentile(a, p)), 4)
                                         for p in (5, 25, 50, 75, 95)},
                    "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
                    "frac_lt_0.95": round(float((a < 0.95).mean()), 4),
                    "frac_lt_0.80": round(float((a < 0.80).mean()), 4),
                    "frac_lt_0.50": round(float((a < 0.50).mean()), 4)}
        rep[name] = {"files": len(files), "multi_object_frames": n_obj_multi,
                     "visibility": q(vis), "raycast_visibility": q(ray),
                     "visible_mask_over_amodal_hull": q(cover)}
        print(f"{name:24s} n={len(files):6d}  vis={q(vis) and q(vis)['p50']}  "
              f"ray={q(ray) and q(ray)['p50']}  cover={q(cover) and q(cover)['p50']}", flush=True)
    (OUT / "A1_TRAINSET_OCCLUSION.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(f"\nwrote {(OUT/'A1_TRAINSET_OCCLUSION.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
