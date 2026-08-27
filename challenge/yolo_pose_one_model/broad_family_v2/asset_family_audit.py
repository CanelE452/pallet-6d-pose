"""PHASE 2 — BROAD 40K 의 asset family 를 전수 표로.

★ 핵심 구분: 같은 mesh 를 W/D/H 로 스케일한 것은 **새 object instance 로 세지
않는다.**  frame 다양성과 mesh 다양성을 섞으면 "다양하다" 는 착시가 생긴다.
"""
from __future__ import annotations

import collections, csv, json, math, pathlib
import numpy as np

ROOT = pathlib.Path("/home/minjae/Documents/github/pallet-pose")
BROAD = ROOT/"data/pallet/training_data/paper_release/v2_prod40k_clean_merged"
OUT = ROOT/"challenge/yolo_pose_one_model/broad_family_v2"


def main():
    rows = collections.defaultdict(list)
    assets = collections.Counter()
    mats = collections.defaultdict(collections.Counter)
    for p in (BROAD/"labels").iterdir():
        d = json.load(open(p))
        o = d["objects"][0]
        v2 = o.get("v2_labels", {})
        a = o.get("source_asset")
        assets[a] += 1
        dm = o["dimensions_m"]
        rows[a].append((dm["width"], dm["height"], dm["depth"],
                        v2.get("luma_actual"),
                        v2.get("elevation_deg_actual"),
                        v2.get("camera_distance_actual_m")))
        mats[a][str(v2.get("material_variant_actual"))] += 1

    v2sig = json.load(open(OUT/"TARGET_ASSET_EXCLUSION_AUDIT_V2.json"))
    total = sum(assets.values())
    share = {a: n/total for a, n in assets.items()}
    entropy = -sum(s*math.log(s) for s in share.values())
    table = []
    for a, n in sorted(assets.items(), key=lambda kv: -kv[1]):
        arr = np.array([r[:3] for r in rows[a]], float)
        fp = np.sort(arr[:, [0, 2]], 1)
        aspect = fp[:, 1]/fp[:, 0]
        srt = np.sort(arr, 1)[:, ::-1]
        thick = srt[:, 2]/srt[:, 0]
        luma = np.array([r[3] for r in rows[a] if r[3] is not None], float)
        sig = v2sig["resolved"].get(a)
        table.append({
            "source_asset": a,
            "pallet_type": (sig or {}).get("pallet_type",
                            v2sig["unresolved"].get(a, {}).get("pallet_type")),
            "frames": n, "exposure_share": round(share[a], 4),
            "mesh_resolved": bool(sig),
            "mesh_vertices": (sig or {}).get("vertices"),
            "mesh_faces": (sig or {}).get("faces"),
            "mesh_aspect": (sig or {}).get("footprint_aspect"),
            "mesh_thickness": (sig or {}).get("thickness_ratio"),
            "frame_aspect_min": round(float(aspect.min()), 3),
            "frame_aspect_med": round(float(np.median(aspect)), 3),
            "frame_aspect_max": round(float(aspect.max()), 3),
            "frame_thick_min": round(float(thick.min()), 4),
            "frame_thick_med": round(float(np.median(thick)), 4),
            "frame_thick_max": round(float(thick.max()), 4),
            "luma_med": round(float(np.median(luma)), 1) if luma.size else None,
            "material_top": mats[a].most_common(1)[0][0] if mats[a] else None,
        })
    summary = {
        "total_frames": total,
        "unique_source_assets": len(assets),
        "unique_mesh_instances_verified": len(v2sig["resolved"]),
        "unverified_assets": len(v2sig["unresolved"]),
        "effective_asset_count_exp_entropy": round(math.exp(entropy), 3),
        "single_asset_max_share": round(max(share.values()), 4),
        "note": "frame 별 W/D/H 스케일 랜덤화는 mesh 다양성이 아니다. "
                "unique mesh 는 4 이고 그 중 2 개만 실제 mesh 로 검증됐다.",
    }
    with open(OUT/"CURRENT_ASSET_FAMILY_AUDIT.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader()
        w.writerows(table)
    md = ["# CURRENT ASSET FAMILY AUDIT", "",
          "★ 같은 mesh 를 스케일한 frame 은 새 instance 로 세지 않았다.", "",
          "```", json.dumps(summary, indent=1, ensure_ascii=False), "```", "",
          "## asset 별", "", "```",
          f"{'asset':38}{'type':10}{'frames':>7}{'share':>8}{'mesh?':>7}"
          f"{'verts':>9}{'m.thick':>9}{'frame thick min/med/max':>26}"]
    for t in table:
        rng = "{} / {} / {}".format(t["frame_thick_min"], t["frame_thick_med"],
                                    t["frame_thick_max"])
        md.append(f"{str(t['source_asset'])[:36]:38}{str(t['pallet_type']):10}"
                  f"{t['frames']:>7}{t['exposure_share']:>8.3f}"
                  f"{str(t['mesh_resolved']):>7}"
                  f"{(t['mesh_vertices'] or 0):>9,}"
                  f"{(t['mesh_thickness'] or 0):>9.4f}"
                  f"{rng:>26}")
    md += ["```", "",
           f"평가 대상 두께비 **{v2sig['target']['thickness_ratio']}**, "
           f"종횡비 **{v2sig['target']['footprint_aspect']}**", ""]
    (OUT/"CURRENT_ASSET_FAMILY_AUDIT.md").write_text("\n".join(md)+"\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    for t in table:
        print(f"  {str(t['source_asset'])[:34]:36} frames {t['frames']:>6} "
              f"share {t['exposure_share']:.3f}  mesh_thick "
              f"{t['mesh_thickness']}  frame_thick {t['frame_thick_min']}~{t['frame_thick_max']}")


if __name__ == "__main__":
    main()
