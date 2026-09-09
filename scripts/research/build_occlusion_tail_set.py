"""가림 하위꼬리 평가셋 — 마스크가 있는 유일한 합성셋에서 실측으로 뽑는다.

visibility 스칼라(전부 1.0)가 아니라 visible_mask / amodal 볼록껍질 비율을 쓴다.
★ 이 비율의 중앙값 0.657 은 가림이 아니라 팔레트가 **격자 구조**라 슬랫 사이로
   배경이 보이기 때문이다.  그래서 절대 임계가 아니라 **그 셋 안의 하위 꼬리**로 정의한다.
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/pallet/training_data/paper_4pallet_mask_v1"
OUT = ROOT / "data/pallet/results/hough_line_visual"


def main():
    files = sorted(glob.glob(str(SRC / "*.json")))
    rows = []
    for i, f in enumerate(files):
        if i % 2000 == 0:
            print(f"  {i}/{len(files)}", flush=True)
        try:
            d = json.load(open(f))
        except Exception:
            continue
        o = (d.get("objects") or [{}])[0]
        ma = o.get("mask_area_px")
        pc = np.asarray(o.get("projected_cuboid", []), float)[:8]
        if not isinstance(ma, (int, float)) or ma <= 0 or pc.shape != (8, 2) \
           or not np.isfinite(pc).all():
            continue
        try:
            hull = ConvexHull(pc).volume
        except Exception:
            continue
        if hull <= 1:
            continue
        rows.append({"stem": Path(f).stem, "cover": float(min(ma / hull, 2.0)),
                     "mask_area_px": float(ma), "hull_px": float(hull),
                     "raycast_visibility": o.get("raycast_visibility")})
    c = np.array([r["cover"] for r in rows])
    q = {f"p{p}": float(np.percentile(c, p)) for p in (1, 5, 10, 25, 50, 75, 95)}
    print(f"\nn={len(rows)}  cover 분위 {json.dumps(q)}")

    tiers = {}
    for name, thr in (("TAIL_lt40", 0.40), ("TAIL_lt50", 0.50), ("TAIL_lt60", 0.60)):
        sel = sorted([r["stem"] for r in rows if r["cover"] < thr])
        tiers[name] = {"threshold": thr, "n": len(sel),
                       "frac": round(len(sel) / len(rows), 4)}
        (OUT / f"OCCL_{name}.txt").write_text("\n".join(sel) + "\n")
        print(f"  {name:12s} < {thr:.2f}  n={len(sel):5d} ({len(sel)/len(rows)*100:5.2f}%)")

    rep = {"schema_version": "occlusion_tail_set_v1", "source": str(SRC.relative_to(ROOT)),
           "n_scanned": len(files), "n_with_mask": len(rows),
           "cover_definition": "visible mask_area_px / convex hull of projected_cuboid",
           "caveat": ("중앙값 0.657 은 가림이 아니라 팔레트 격자 구조 탓이다. "
                      "따라서 이 꼬리는 '가림이 심한' 이 아니라 '이 셋에서 상대적으로 "
                      "덜 보이는' 프레임이며, 상판 위 적재물 유무는 별개로 확인해야 한다."),
           "cover_quantiles": q, "tiers": tiers}
    (OUT / "OCCLUSION_TAIL_SET.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {(OUT/'OCCLUSION_TAIL_SET.json').relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
