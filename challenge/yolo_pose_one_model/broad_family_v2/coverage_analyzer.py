"""PHASE 9 — BROAD 40K 의 asset/appearance coverage 를 DEV 실패와 대조.

★ 렌더링을 시작하지 않는다.  결과 전에 새 데이터 수량을 발명하지 않는다.
WEAK_PASS 이거나 "generic 은 좋은데 target/night 만 실패" 일 때만 spec 초안을 쓴다.
"""
from __future__ import annotations
import collections, json, os, pathlib, sys
import numpy as np
ROOT = pathlib.Path("/home/minjae/Documents/github/pallet-pose")
BROAD = ROOT/"data/pallet/training_data/paper_release/v2_prod40k_clean_merged"
OUT = ROOT/"challenge/yolo_pose_one_model/paper_generic_pipeline/broad_family_v2"


def main():
    per = collections.defaultdict(list)
    meta = collections.defaultdict(collections.Counter)
    for p in (BROAD/"labels").iterdir():
        o = json.load(open(p))["objects"][0]
        v2 = o.get("v2_labels", {})
        t = v2.get("pallet_type")
        d = o["dimensions_m"]
        per[t].append((d["width"], d["height"], d["depth"],
                       v2.get("luma_actual"), v2.get("elevation_deg_actual"),
                       v2.get("camera_distance_actual_m")))
        meta[t]["asset:" + str(o.get("source_asset"))] += 1
        meta[t]["material:" + str(v2.get("material_variant_actual"))] += 1
        meta[t]["scene:" + str(json.load(open(p))["camera_data"].get("scene_preset"))] += 1
    report = {"note": "렌더 금지. 현재 support 를 표로 만들 뿐이다.", "per_type": {}}
    thick_all, aspect_all, luma_all = [], [], []
    for t, rows in per.items():
        a = np.array([r[:3] for r in rows], float)
        fp = np.sort(a[:, [0, 2]], 1); aspect = fp[:, 1]/fp[:, 0]
        srt = np.sort(a, 1)[:, ::-1]; thick = srt[:, 2]/srt[:, 0]
        luma = np.array([r[3] for r in rows if r[3] is not None], float)
        thick_all.append(thick); aspect_all.append(aspect); luma_all.append(luma)
        report["per_type"][t] = {
            "n": len(rows),
            "aspect": [round(float(aspect.min()), 3), round(float(np.median(aspect)), 3),
                       round(float(aspect.max()), 3)],
            "thickness": [round(float(thick.min()), 4), round(float(np.median(thick)), 4),
                          round(float(thick.max()), 4)],
            "luma": [round(float(luma.min()), 1), round(float(np.median(luma)), 1),
                     round(float(luma.max()), 1)] if luma.size else None,
            "meta": dict(meta[t].most_common(8))}
    thick = np.concatenate(thick_all); aspect = np.concatenate(aspect_all)
    luma = np.concatenate([l for l in luma_all if l.size])
    report["all"] = {
        "n": int(len(thick)),
        "aspect": [round(float(aspect.min()), 3), round(float(np.median(aspect)), 3),
                   round(float(aspect.max()), 3)],
        "thickness": [round(float(thick.min()), 4), round(float(np.median(thick)), 4),
                      round(float(thick.max()), 4)],
        "luma": [round(float(luma.min()), 1), round(float(np.median(luma)), 1),
                 round(float(luma.max()), 1)]}
    report["missing_cells"] = {
        "geometry_thin": {"target_thickness": 0.0923,
                          "broad_frames_at_or_below": int((thick <= 0.0923).sum()),
                          "rate": round(float((thick <= 0.0923).mean()), 4)},
        "appearance_bright": {"real_luma_p50": 123,
                              "broad_frames_ge_100": int((luma >= 100).sum()),
                              "rate": round(float((luma >= 100).mean()), 4)},
        "n_unique_assets": len({k for c in meta.values() for k in c
                                if k.startswith("asset:")}),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"BROAD40K_COVERAGE.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["all"], indent=1))
    print(json.dumps(report["missing_cells"], indent=1))
    print(f"-> {OUT/'BROAD40K_COVERAGE.json'}")


if __name__ == "__main__":
    main()
