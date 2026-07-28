#!/usr/bin/env python3
"""STAGE16 truncation_addon_v1 pre-training audit (전수 6000 frame 통계).

목적: 파일럿 300장 게이트 스킵됨 -> 6000 전수 통계로 대체 검증.
핵심 질문: "이게 rear 저앙각 붕괴를 고칠 데이터인가" (분포 검증).

elevation 은 각 프레임 GT world pose 에서 직접 재계산 (생성기 기록값 _summary/frame_meta 는 참고만).
convention: camera_dynamic_0123_v4. 0-3=front(near), 4-7=rear(far). depth edges 0-4,1-5,2-6,3-7.
elev 정의: STAGE18 elev_from_world 재사용. edge-on~0deg, top-down(부감)~90deg.
"""
import os, sys, json, glob
import numpy as np

DATA = "/home/minjae/Documents/github/pallet-pose/challenge/data/training/truncation_addon_v1"
OUT = "/home/minjae/Documents/github/pallet-pose/data/pallet/eval_results/stage16_audit"

FRONT = [0, 1, 2, 3]
REAR = [4, 5, 6, 7]
DEPTH_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]

# ---- prescription (STAGE18 저앙각 처방) ----
RX_BINS = [(-1e9, 3), (3, 8), (8, 15), (15, 25), (25, 1e9)]
RX_LBL = ["<3", "3-8", "8-15", "15-25", "25+"]
RX_PCT = [10, 30, 30, 15, 15]  # target percent

# V_geom 처방
VG_RX = {8: 10, 7: 20, 6: 30, 5: 25, 4: 15}


def elev_from_world(d):
    cam = np.array(d["camera_data"]["location_worldframe"], float)
    obj = np.array(d["objects"][0]["location"], float)
    v = cam - obj
    horiz = np.linalg.norm(v[:2])
    return float(np.degrees(np.arctan2(v[2], horiz)))


def frsep(pc):
    """depth 변(front->rear) 2D 길이 median from projected_cuboid(8, unclamped)."""
    g = np.array(pc, float)
    ds = [np.linalg.norm(g[a] - g[b]) for a, b in DEPTH_EDGES]
    return float(np.median(ds))


def rx_bin(e):
    for i, (lo, hi) in enumerate(RX_BINS):
        if lo <= e < hi:
            return i
    return len(RX_BINS) - 1


def agg(vals):
    a = np.array([v for v in vals if v is not None], float)
    if a.size == 0:
        return {"n": 0, "median": None, "mean": None, "p10": None, "p90": None}
    return {"n": int(a.size), "median": round(float(np.median(a)), 1),
            "mean": round(float(a.mean()), 1),
            "p10": round(float(np.percentile(a, 10)), 1),
            "p90": round(float(np.percentile(a, 90)), 1)}


def main():
    os.makedirs(OUT, exist_ok=True)
    jps = sorted(glob.glob(os.path.join(DATA, "0*.json")))
    print(f"[load] {len(jps)} json")

    recs = []
    meta_diff = []
    for jp in jps:
        d = json.load(open(jp))
        o = d["objects"][0]
        fm = d.get("frame_meta", {})
        e = elev_from_world(d)
        kif = o["keypoint_in_frame"]
        kfront = o.get("keypoint_in_front_of_camera", [True] * 9)
        r = {
            "fid": os.path.splitext(os.path.basename(jp))[0],
            "elev": e,
            "elev_meta": fm.get("camera_elevation_deg"),
            "V_geom": o.get("V_geom"),
            "num_in_frame": o.get("num_corners_in_frame"),
            "bitmask": o.get("missing_corner_bitmask"),
            "frsep": frsep(o["projected_cuboid"]),
            "rear_in_frame": sum(1 for i in REAR if kif[i]),
            "rear_in_front": sum(1 for i in REAR if kfront[i]),
            "front_in_frame": sum(1 for i in FRONT if kif[i]),
            "dims": tuple(round(x, 3) for x in o["cuboid_dimensions_m"]),
            "bg": fm.get("background_3d"),
            "hdri": fm.get("hdri"),
            "floor_tex": (fm.get("floor") or {}).get("floor_texture"),
            "trunc_mode": fm.get("truncation_mode"),
            "trunc_applied": fm.get("truncation_applied"),
            "cargo": fm.get("cargo_applied"),
            "occ": fm.get("occlusion_applied"),
            "exposure": fm.get("exposure_ev"),
        }
        if r["elev_meta"] is not None:
            meta_diff.append(e - r["elev_meta"])
        recs.append(r)

    n = len(recs)
    elevs = np.array([r["elev"] for r in recs])

    # ===== 감사1: elevation 분포 =====
    elev_hist = {}
    for i, lbl in enumerate(RX_LBL):
        lo, hi = RX_BINS[i]
        cnt = int(((elevs >= lo) & (elevs < hi)).sum())
        elev_hist[lbl] = {"n": cnt, "pct": round(100 * cnt / n, 1),
                          "target_pct": RX_PCT[i]}
    low_3_15 = int(((elevs >= 3) & (elevs < 15)).sum())
    audit1 = {
        "n": n,
        "elev_stats": {"min": round(float(elevs.min()), 1),
                       "p10": round(float(np.percentile(elevs, 10)), 1),
                       "median": round(float(np.median(elevs)), 1),
                       "p90": round(float(np.percentile(elevs, 90)), 1),
                       "max": round(float(elevs.max()), 1)},
        "hist_vs_prescription": elev_hist,
        "low_elev_3_15_pct": round(100 * low_3_15 / n, 1),
        "low_elev_3_15_target": 60,
        "topdown_25plus_pct": elev_hist["25+"]["pct"],
        "meta_recompute_diff": {
            "n": len(meta_diff),
            "mean": round(float(np.mean(meta_diff)), 2) if meta_diff else None,
            "median": round(float(np.median(meta_diff)), 2) if meta_diff else None,
            "p10": round(float(np.percentile(meta_diff, 10)), 2) if meta_diff else None,
            "p90": round(float(np.percentile(meta_diff, 90)), 2) if meta_diff else None,
            "note": "recomputed(centroid) - frame_meta(target). sign consistent.",
        },
    }

    # ===== 감사2: 저앙각 rear 관측 (frsep, rear valid) =====
    def rear_stats(mask_recs):
        if not mask_recs:
            return {"n": 0}
        return {
            "n": len(mask_recs),
            "frsep": agg([r["frsep"] for r in mask_recs]),
            "rear_in_frame_mean": round(np.mean([r["rear_in_frame"] for r in mask_recs]), 2),
            "rear_in_front_mean": round(np.mean([r["rear_in_front"] for r in mask_recs]), 2),
            "frac_all4_rear_in_front": round(
                np.mean([r["rear_in_front"] == 4 for r in mask_recs]), 3),
            "frac_ge2_rear_in_frame": round(
                np.mean([r["rear_in_frame"] >= 2 for r in mask_recs]), 3),
        }
    audit2 = {
        "low_lt8": rear_stats([r for r in recs if r["elev"] < 8]),
        "band_3_8": rear_stats([r for r in recs if 3 <= r["elev"] < 8]),
        "vlow_lt3": rear_stats([r for r in recs if r["elev"] < 3]),
        "band_8_15": rear_stats([r for r in recs if 8 <= r["elev"] < 15]),
        "note": ("STAGE18 기준: 저앙각(3-8)서 frsep 50-78px면 관측가능=학습가치. "
                 "<3deg frsep~10px 겹침=단안한계라 적어야 정상."),
    }
    # frsep by elev fine bin
    fine = [(-1e9, 3), (3, 5), (5, 8), (8, 12), (12, 18), (18, 25), (25, 1e9)]
    fine_lbl = ["<3", "3-5", "5-8", "8-12", "12-18", "18-25", "25+"]
    frsep_by_elev = {}
    for (lo, hi), lbl in zip(fine, fine_lbl):
        g = [r for r in recs if lo <= r["elev"] < hi]
        frsep_by_elev[lbl] = {"n": len(g),
                              "frsep": agg([r["frsep"] for r in g]),
                              "rear_in_front_mean": round(
                                  np.mean([r["rear_in_front"] for r in g]), 2) if g else None}
    audit2["frsep_by_elev_fine"] = frsep_by_elev

    # ===== 감사3: V_geom + bitmask =====
    vg_hist = {}
    from collections import Counter
    vc = Counter(r["num_in_frame"] for r in recs)
    for v in sorted(vc, reverse=True):
        vg_hist[v] = {"n": vc[v], "pct": round(100 * vc[v] / n, 1),
                      "target_pct": VG_RX.get(v)}
    # bitmask: which corner bits set (missing)
    bit_missing = np.zeros(8, int)
    bmc = Counter()
    for r in recs:
        bm = r["bitmask"] or 0
        bmc[bm] += 1
        for i in range(8):
            if bm & (1 << i):
                bit_missing[i] += 1
    # near-face = idx 0-3 (Phase A: near face 위주여야)
    near_missing = int(bit_missing[:4].sum())
    far_missing = int(bit_missing[4:].sum())
    audit3 = {
        "V_geom_hist": vg_hist,
        "corner_missing_count": {str(i): int(bit_missing[i]) for i in range(8)},
        "near_face_0_3_missing_total": near_missing,
        "far_face_4_7_missing_total": far_missing,
        "near_frac_of_all_missing": round(near_missing / max(near_missing + far_missing, 1), 3),
        "top_bitmasks": {str(k): v for k, v in bmc.most_common(10)},
        "trunc_mode_dist": {k: v for k, v in Counter(r["trunc_mode"] for r in recs).items()},
    }

    # ===== 감사5: appearance 다양성 =====
    def dist(key):
        c = Counter(r[key] for r in recs)
        return {str(k): v for k, v in c.most_common(30)}
    audit5 = {
        "background_3d": dist("bg"),
        "hdri_n_unique": len(set(r["hdri"] for r in recs)),
        "hdri_top": {k: v for k, v in Counter(r["hdri"] for r in recs).most_common(12)},
        "floor_tex_n_unique": len(set(r["floor_tex"] for r in recs)),
        "floor_tex_top": {k: v for k, v in Counter(r["floor_tex"] for r in recs).most_common(12)},
        "cargo_applied_frac": round(np.mean([bool(r["cargo"]) for r in recs]), 3),
        "occlusion_applied_frac": round(np.mean([bool(r["occ"]) for r in recs]), 3),
        "dims_unique": {str(k): v for k, v in Counter(r["dims"] for r in recs).items()},
    }

    result = {"data_dir": DATA, "n_frames": n,
              "convention": "camera_dynamic_0123_v4",
              "elev_definition": "elev_from_world: arctan2(dz, horiz(cam-obj)); edge-on~0, topdown~90",
              "audit1_elevation": audit1,
              "audit2_lowelev_rear": audit2,
              "audit3_vgeom_bitmask": audit3,
              "audit5_appearance": audit5}
    json.dump(result, open(os.path.join(OUT, "trunc_addon_audit.json"), "w"), indent=1)
    print(f"[done] -> {os.path.join(OUT, 'trunc_addon_audit.json')}")
    # keep recs for overlay selection
    json.dump([{"fid": r["fid"], "elev": r["elev"], "rear_in_front": r["rear_in_front"],
                "num_in_frame": r["num_in_frame"], "frsep": r["frsep"]} for r in recs],
              open(os.path.join(OUT, "_recs.json"), "w"))
    return result


if __name__ == "__main__":
    main()
