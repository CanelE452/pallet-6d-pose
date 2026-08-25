"""OLD Stage-A vs V2 SUPPORT AUDIT — CPU only, 학습 없음.

OG = OLD_GENERIC (stage_a G_, 38,002)      소스 v2_prod40k_clean_merged
OT = OLD_TARGET  unique (17,957)           소스 challenge/data/02_synthetic/training/v1,v2
V2 = CURRENT V2 EARLY10K (10,000)

없는 필드는 NULL 로 둔다. 지어내지 않는다.
threshold 는 freeze 된 값만 사용 (deep-thin r3d<=0.10).
"""
import csv, json, os, re, sys, collections
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
Y = f"{ROOT}/challenge/yolo_pose_one_model"
Q = f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
OUT = f"{Q}/old_support"
os.makedirs(OUT, exist_ok=True)
DEEP_THIN = 0.10          # freeze 된 값
LOW_ELEV = 8.0            # freeze 된 generator bin

SCHEMA = ["dataset_group", "stage_a_entry_id", "source_dataset", "source_frame_id",
          "source_realpath_or_logical_id", "asset_id", "topology", "W", "D", "H", "r3d",
          "elevation", "azimuth", "distance", "bbox_w", "bbox_h", "bbox_area",
          "bbox_min_side", "r2d_projected_thickness", "visibility_count", "background",
          "hdri_lighting", "material", "occlusion", "image_width", "image_height",
          "padding_state", "keypoint_convention"]


def num(x):
    try:
        return float(x)
    except Exception:
        return None


def from_json(p, group, entry_id, src_ds, src_id, realpath):
    """raw annotation JSON 에서 뽑을 수 있는 것만. 없으면 None."""
    r = {k: None for k in SCHEMA}
    r.update({"dataset_group": group, "stage_a_entry_id": entry_id,
              "source_dataset": src_ds, "source_frame_id": src_id,
              "source_realpath_or_logical_id": realpath})
    try:
        j = json.load(open(p))
    except Exception:
        return r
    cd = j.get("camera_data") or {}
    r["image_width"] = cd.get("width")
    r["image_height"] = cd.get("height")
    if r["image_width"] is None:                 # 구형: intrinsics 의 cx/cy 로 유도
        ic = cd.get("intrinsics") or {}
        res = ic.get("resolution")
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            r["image_width"], r["image_height"] = int(res[0]), int(res[1])
        elif ic.get("cx") and ic.get("cy"):
            r["image_width"], r["image_height"] = int(round(ic["cx"] * 2)), int(round(ic["cy"] * 2))
    o = (j.get("objects") or [{}])[0]
    r["keypoint_convention"] = o.get("keypoint_convention")
    r["asset_id"] = o.get("source_asset")
    # 스키마가 두 가지다: 신규 dimensions_m{width,depth,height} / 구형 cuboid_dimensions_m[3]
    dm = o.get("dimensions_m") or {}
    if dm:
        W, D, H = num(dm.get("width")), num(dm.get("depth")), num(dm.get("height"))
    else:
        cd3 = o.get("cuboid_dimensions_m")
        if isinstance(cd3, (list, tuple)) and len(cd3) >= 3:
            # ★ 실측 확인: cuboid_dimensions_m == keypoints_3d_world 의 ptp(x,y,z)
            #   = [W, D, H].  convention 이름이 y_up 이라고 index1 을 H 로 보면 틀린다
            #   (그렇게 하면 r3d 가 10.8 이 되어 팔레트로 불가능한 값이 나온다).
            W, D, H = num(cd3[0]), num(cd3[1]), num(cd3[2])
        else:
            W = D = H = None
    r["W"], r["D"], r["H"] = W, D, H
    if W and D and H:
        r["r3d"] = H / max(min(W, D), 1e-9)
    pc = o.get("projected_cuboid")
    if pc:
        a = np.array(pc[:8], dtype=float)
        bw, bh = a[:, 0].ptp(), a[:, 1].ptp()
        r["bbox_w"], r["bbox_h"] = float(bw), float(bh)
        r["bbox_area"] = float(bw * bh)
        r["bbox_min_side"] = float(min(bw, bh))
        r["r2d_projected_thickness"] = float(min(bw, bh) / max(max(bw, bh), 1e-9))
        if r["image_width"]:
            iw, ih = r["image_width"], r["image_height"]
            r["visibility_count"] = int(((a[:, 0] >= 0) & (a[:, 0] < iw) &
                                         (a[:, 1] >= 0) & (a[:, 1] < ih)).sum())
    pt = o.get("pose_transform")
    t = None
    if pt:
        T = np.array(pt, dtype=float)
        t = T[:3, 3] if T.shape == (4, 4) else None
    elif o.get("location") is not None:
        loc = np.array(o["location"], dtype=float)
        cam = (j.get("camera_data") or {}).get("location_worldframe")
        t = (loc - np.array(cam, dtype=float)) if cam else loc
    if True:
        if t is not None and np.linalg.norm(t) > 0:
            r["distance"] = float(np.linalg.norm(t))
            # elevation: 카메라 광축(+Z) 기준 물체 중심의 올려본각
            hor = float(np.hypot(t[0], t[2])) if abs(t[2]) > 0 else float(abs(t[0]))
            r["elevation"] = float(np.degrees(np.arctan2(abs(t[1]), max(hor, 1e-9))))
            r["azimuth"] = float(np.degrees(np.arctan2(t[0], max(abs(t[2]), 1e-9))))
    for k, src in (("background", "scene_placement_v2"), ("material", "v2_labels")):
        v = o.get(src)
        if isinstance(v, dict):
            r[k] = json.dumps(v, ensure_ascii=False)[:120]
    r["topology"] = (o.get("v2_labels") or {}).get("topology") if isinstance(o.get("v2_labels"), dict) else None
    return r


# ---------------- OG ----------------
GSRC = f"{ROOT}/data/pallet/training_data/paper_release/v2_prod40k_clean_merged"
og_lines = [l.strip() for l in open(f"{Y}/manifests/generic_train.txt") if l.strip()]
OG = []
for i, l in enumerate(og_lines):
    m = re.search(r"(f\d+)_rgb", l)
    if not m:
        continue
    fid = m.group(1)
    OG.append(from_json(f"{GSRC}/labels/{fid}_label.json", "OG", f"G__{fid}",
                        "v2_prod40k_clean_merged", fid, l))
print(f"  OG {len(OG)}")

# ---------------- OT (unique) ----------------
ot_lines = [l.strip() for l in open(f"{Y}/manifests/target_train.txt") if l.strip()]
OT = []
for l in ot_lines:
    p = os.path.join(ROOT, l)
    jp = os.path.splitext(p)[0] + ".json"
    m = re.search(r"training/(v\d)/(part_\d+)/[^/]+/(\d+)\.png", l)
    sid = f"{m.group(1)}__{m.group(2)}__{m.group(3)}" if m else os.path.basename(l)
    OT.append(from_json(jp, "OT", f"T__{sid}", "challenge_02_synthetic_training", sid, l))
print(f"  OT unique {len(OT)}")

# ---------------- V2 ----------------
VSRC = "/home/minjae/V2_CF_TO_UBUNTU_20260823T0933/extracted"
v2_stems = sorted(f[:-4] for sp in ("train", "val")
                  for f in os.listdir(f"{Y}/datasets/v2_cf_early10k/labels/{sp}"))
V2 = []
for s in v2_stems:
    cand = [f"{VSRC}/raw_annotations/{s}.json", f"{VSRC}/raw_annotations/{s}_label.json"]
    jp = next((c for c in cand if os.path.exists(c)), None)
    if jp is None:
        g = [f"{VSRC}/raw_annotations/{d}/{s}.json" for d in os.listdir(f"{VSRC}/raw_annotations")
             if os.path.isdir(f"{VSRC}/raw_annotations/{d}")] if os.path.isdir(f"{VSRC}/raw_annotations") else []
        jp = next((c for c in g if os.path.exists(c)), None)
    V2.append(from_json(jp or "", "V2", s, "V2_CF_EARLY10K", s, s))
print(f"  V2 {len(V2)}  (raw json 매칭 {sum(1 for r in V2 if r['r3d'] is not None)})")

for nm, rs in (("OLD_GENERIC", OG), ("OLD_TARGET", OT), ("V2_EARLY10K", V2)):
    with open(f"{OUT}/{nm}_METADATA.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA, extrasaction="ignore")
        w.writeheader()
        w.writerows(rs)

# OT exposure 기준 테이블 (rep1 포함)
with open(f"{OUT}/OLD_TARGET_EXPOSURE_METADATA.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=SCHEMA + ["exposure_alias"], extrasaction="ignore")
    w.writeheader()
    for r in OT:
        for a in (0, 1):
            d = dict(r)
            d["exposure_alias"] = a
            w.writerow(d)

# ---------------- U3 support audit ----------------
NUMK = ["r3d", "elevation", "distance", "bbox_area", "bbox_min_side",
        "r2d_projected_thickness", "visibility_count"]


def col(rs, k):
    return np.array([r[k] for r in rs if r[k] is not None], dtype=float)


def desc(a):
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            **{f"p{p}": float(np.percentile(a, p)) for p in (10, 25, 50, 75, 90)}}


def smd(a, b):
    if a.size == 0 or b.size == 0:
        return None
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else None


def ks(a, b):
    if a.size == 0 or b.size == 0:
        return None
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b).statistic)
    except Exception:
        x = np.sort(np.concatenate([a, b]))
        return float(np.abs(np.searchsorted(np.sort(a), x, "right") / a.size
                            - np.searchsorted(np.sort(b), x, "right") / b.size).max())


SUP = {"numeric": {}, "categorical": {}, "frozen_thresholds":
       {"deep_thin_r3d": DEEP_THIN, "low_elevation_deg": LOW_ELEV}}
for k in NUMK:
    g, t, v = col(OG, k), col(OT, k), col(V2, k)
    SUP["numeric"][k] = {"OG": desc(g), "OT": desc(t), "V2": desc(v),
                         "SMD_OT_vs_OG": smd(t, g), "SMD_OT_vs_V2": smd(t, v),
                         "SMD_OG_vs_V2": smd(g, v),
                         "KS_OT_vs_OG": ks(t, g), "KS_OT_vs_V2": ks(t, v),
                         "KS_OG_vs_V2": ks(g, v)}
for k in ("asset_id", "topology", "background", "material"):
    def share(rs):
        c = collections.Counter(str(r[k]) for r in rs if r[k] is not None)
        n = sum(c.values())
        return {kk: v / n for kk, v in c.most_common(12)} if n else {}
    SUP["categorical"][k] = {"OG": share(OG), "OT": share(OT), "V2": share(V2)}

# deep-thin / low-elev 비율
for nm, rs in (("OG", OG), ("OT", OT), ("V2", V2)):
    r3 = col(rs, "r3d")
    el = col(rs, "elevation")
    SUP.setdefault("frozen_bins", {})[nm] = {
        "deep_thin_frac": float((r3 <= DEEP_THIN).mean()) if r3.size else None,
        "low_elev_frac": float((el < LOW_ELEV).mean()) if el.size else None}

# ---------------- U4 joint ----------------
def joint(rs, ka, kb, ba, bb):
    a, b = [], []
    for r in rs:
        if r[ka] is not None and r[kb] is not None:
            a.append(r[ka])
            b.append(r[kb])
    if not a:
        return {}
    H, _, _ = np.histogram2d(a, b, bins=[ba, bb])
    return (H / max(H.sum(), 1)).round(5).tolist()


R3B = [0, .05, .10, .15, .25, .40, 10]
ELB = [0, 4, 8, 15, 25, 40, 90]
BAB = [0, 5e3, 2e4, 5e4, 1e5, 3e5, 1e12]
JOINT = {}
for nm, rs in (("OG", OG), ("OT", OT), ("V2", V2)):
    JOINT[nm] = {"r3d_x_elev": joint(rs, "r3d", "elevation", R3B, ELB),
                 "r3d_x_bbox": joint(rs, "r3d", "bbox_area", R3B, BAB),
                 "elev_x_bbox": joint(rs, "elevation", "bbox_area", ELB, BAB)}
JOINT["bins"] = {"r3d": R3B, "elevation": ELB, "bbox_area": BAB}
# OT-heavy cell
def cells(a, b, ba, bb):
    if not isinstance(a, list) or not isinstance(b, list) or not a or not b:
        return []
    A, B = np.array(a, dtype=float), np.array(b, dtype=float)
    if A.size == 0 or B.size == 0 or A.shape != B.shape:
        return []
    d = A - B
    out = []
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if abs(d[i, j]) >= 0.02:
                out.append({"r3d_bin": [ba[i], ba[i+1]], "elev_bin": [bb[j], bb[j+1]],
                            "OT": float(A[i, j]), "other": float(B[i, j]),
                            "diff": float(d[i, j])})
    return sorted(out, key=lambda x: -abs(x["diff"]))[:10]


JOINT["OT_minus_V2_r3d_x_elev_top"] = cells(JOINT["OT"]["r3d_x_elev"],
                                            JOINT["V2"]["r3d_x_elev"], R3B, ELB)
JOINT["OT_minus_OG_r3d_x_elev_top"] = cells(JOINT["OT"]["r3d_x_elev"],
                                            JOINT["OG"]["r3d_x_elev"], R3B, ELB)
json.dump(SUP, open(f"{OUT}/OLD_VS_V2_SUPPORT_AUDIT_UBUNTU.json", "w"), indent=2, ensure_ascii=False)
json.dump(JOINT, open(f"{OUT}/OLD_VS_V2_JOINT_SUPPORT.json", "w"), indent=2)

# ---------------- U6 feasibility (generic 변수만) ----------------
FEAT = ["r3d", "elevation", "bbox_area", "visibility_count"]


def mat(rs):
    ok = [r for r in rs if all(r[k] is not None for k in FEAT)]
    return np.array([[r[k] for k in FEAT] for r in ok], dtype=float), ok


VT, _ = mat(V2)
OTM, _ = mat(OT)
FEAS = {"features_used": FEAT,
        "forbidden": ["target asset id", "target texture", "RGB similarity"],
        "V2_pool_N": int(VT.shape[0]), "OT_N": int(OTM.shape[0])}
if VT.size and OTM.size:
    mu, sd = VT.mean(0), VT.std(0) + 1e-9
    A = (OTM - mu) / sd
    B = (VT - mu) / sd
    step = max(1, A.shape[0] // 4000)
    As = A[::step]
    d = np.sqrt(((As[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    nn = d.min(1)
    FEAS.update({"sampled_OT": int(As.shape[0]),
                 "nn_distance": {f"p{p}": float(np.percentile(nn, p)) for p in (10, 50, 75, 90, 95)},
                 "coverage_frac_within_0.5sd": float((nn <= 0.5).mean()),
                 "coverage_frac_within_1.0sd": float((nn <= 1.0).mean()),
                 "matched_possible_N": int(min(OTM.shape[0], VT.shape[0])),
                 "note": "표준화 4차원 최근접거리. RGB/asset 미사용."})
json.dump(FEAS, open(f"{OUT}/V2_SUPPORT_MATCH_FEASIBILITY.json", "w"), indent=2, ensure_ascii=False)

# ---------------- lineage ----------------
LIN = {"stage_a_total": 73916, "generic_exposure": 38002, "target_exposure": 35914,
       "target_unique": 17957, "target_duplicate_alias": 17957,
       "T_v1_base": 8982, "T_v2_base": 8975,
       "T_v1_vs_T_v2": "서로 다른 실제 렌더 (RGB 해시 교집합 0/40 표본). alias 는 __rep1 접미사",
       "alias_evidence": "base vs __rep1  라벨 60/60 동일, RGB sha256 60/60 동일",
       "composition_json": "datasets/stage_a/_composition.json (target_repeat 2, target_alias 17957)",
       "OLD_GENERIC_SOURCE": "data/pallet/training_data/paper_release/v2_prod40k_clean_merged",
       "OLD_TARGET_SOURCE": "challenge/data/02_synthetic/training/{v1,v2}/part_*/train_palletobj_*",
       "OLD_GENERIC_RELATION_TO_LEGACY_V1": "SAME_SOURCE",
       "evidence": {"OG_subset_of_broad40k": "38002/38002 (100%)",
                    "V1_CF_MATCHED10K_in_OG": "9496/10000 (95.0%)",
                    "OG_minus_broad40k": 0,
                    "V2_EARLY10K_overlap": 0},
       "padding_state": "pad=100 BORDER_REFLECT_101 (_prepare_train.json)"}
json.dump(LIN, open(f"{OUT}/OLD_STAGE_A_LINEAGE_AUDIT.json", "w"), indent=2, ensure_ascii=False)

L = ["# OLD STAGE-A LINEAGE AUDIT", "", "```",
     f"stage_a train      {LIN['stage_a_total']}",
     f"  generic exposure {LIN['generic_exposure']}   (unique = 동일, repeat 없음)",
     f"  target exposure  {LIN['target_exposure']}   = unique {LIN['target_unique']} × 2",
     f"  target alias     {LIN['target_duplicate_alias']}  (__rep1)",
     f"  T v1 base {LIN['T_v1_base']}   T v2 base {LIN['T_v2_base']}", "```", "",
     f"- T__v1 vs T__v2 : {LIN['T_v1_vs_T_v2']}",
     f"- alias 근거     : {LIN['alias_evidence']}", "",
     "## OLD_GENERIC 계보", "```",
     f"source  {LIN['OLD_GENERIC_SOURCE']}",
     f"관계    {LIN['OLD_GENERIC_RELATION_TO_LEGACY_V1']}",
     f"  OG ⊂ broad40k        {LIN['evidence']['OG_subset_of_broad40k']}",
     f"  V1_CF_MATCHED10K ⊂ OG {LIN['evidence']['V1_CF_MATCHED10K_in_OG']}",
     f"  V2_EARLY10K 와 겹침   {LIN['evidence']['V2_EARLY10K_overlap']}", "```"]
open(f"{OUT}/OLD_STAGE_A_LINEAGE_AUDIT.md", "w").write("\n".join(L))
print("\n".join(L))
print("\n=== SUPPORT (median) ===")
print(f"{'metric':26} {'OG':>12} {'OT':>12} {'V2':>12}  {'SMD OT-V2':>10}")
for k in NUMK:
    s = SUP["numeric"][k]
    g = s["OG"].get("p50")
    t = s["OT"].get("p50")
    v = s["V2"].get("p50")
    sm = s["SMD_OT_vs_V2"]
    fmt = lambda x: "        n/a" if x is None else f"{x:12.4f}"
    print(f"{k:26} {fmt(g)} {fmt(t)} {fmt(v)}  {('n/a' if sm is None else f'{sm:10.3f}')}")
print("\n=== frozen bins ===")
for nm, v in SUP["frozen_bins"].items():
    print(f"  {nm:4} deep-thin(r3d<={DEEP_THIN}) {v['deep_thin_frac']}   low-elev(<{LOW_ELEV}) {v['low_elev_frac']}")
print("\n=== V2 support match feasibility ===")
for k, v in FEAS.items():
    if k not in ("forbidden",):
        print(f"  {k}: {v}")
