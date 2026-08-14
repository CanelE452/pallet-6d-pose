#!/usr/bin/env python3
"""STAGE18: rear-corner(4-7) collapse vs camera elevation.

목적: STAGE17 이 밝힌 "rear 코너가 저앙각/edge-on 에서 depth 붕괴" 를 정량화.
  (1) rear 오차가 accurate->collapse 로 전이하는 elevation 임계(도) 산출
  (2) (가)데이터갭 vs (나)기하 ill-conditioned 판별
  (3) STAGE16 저앙각 데이터 생성 elevation 분포 처방

재사용: 추론 X. STAGE17 corner01_diagnosis.json 의 e_corner/gt8 재사용.
        elevation 은 GT pose_transform(real/syn_val) 또는 world-frame(v3 train) 에서 계산.
Convention: camera-facing 0123 v4. 0-3=front(near), 4-7=rear(far). object +Y=down(OpenCV).
"""
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os, sys, json, glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))

sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
from tau_calibrate import collect_val_frames  # noqa
from eval_pvnet_heads import collect_manual, collect_syn, collect_v3  # noqa

STAGE17 = os.path.join(ROOT, "data", "pallet", "eval_results",
                       "stage17_corner01_diagnosis", "corner01_diagnosis.json")
CAD_DIR = os.path.join(ROOT, "challenge", "data", "capturepalletcad_manual_gt")
V3_HELDOUT = os.path.join(ROOT, "challenge", "data", "training", "v3", "batch_009")
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "stage18_elevation_threshold")

FRONT = [0, 1, 2, 3]
REAR = [4, 5, 6, 7]
SPIKE_PX = 15.0
BINS = [(-90, 5), (5, 10), (10, 15), (15, 25), (25, 90)]
BIN_LBL = ["0-5", "5-10", "10-15", "15-25", "25+"]


# ---------- elevation ----------
def elev_from_pose(T):
    """object->camera 4x4 (OpenCV). elevation = angle of viewing ray above pallet top plane.
    edge-on(카메라 팔레트 높이)=~0, top-down(부감)=~90. [확인: outside 5-7, top-down 90]"""
    T = np.array(T, float)
    R, t = T[:3, :3], T[:3, 3]
    up = -R[:, 1]                       # pallet physical up in cam (object +Y = down)
    n = np.linalg.norm(up)
    if n < 1e-9:
        return None
    up /= n
    tn = np.linalg.norm(t)
    if tn < 1e-9:
        return None
    look = t / tn                       # cam -> pallet centroid ray
    ang = np.degrees(np.arccos(np.clip(np.dot(look, up), -1, 1)))
    return ang - 90.0


def elev_from_world(d):
    """v3/train: world Z up. elevation = elev angle of camera above pallet on ground."""
    cam = np.array(d["camera_data"]["location_worldframe"], float)
    obj = np.array(d["objects"][0]["location"], float)
    v = cam - obj
    horiz = np.linalg.norm(v[:2])
    return float(np.degrees(np.arctan2(v[2], horiz)))


# ---------- source path lookup (mirror stage17 collectors) ----------
def build_path_lut():
    lut = {}
    for dom, fid, jp, ip in collect_val_frames():          # outside/night
        lut[(dom, str(fid))] = jp
    for _, fid, jp, ip in collect_manual():
        lut[("manual", str(fid))] = jp
    for jp in sorted(glob.glob(os.path.join(CAD_DIR, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        lut[("cad", str(fid))] = jp
    for _, fid, jp, ip in collect_syn(300):
        lut[("syn_val", str(fid))] = jp
    for _, fid, jp, ip in collect_v3(V3_HELDOUT, 200):
        lut[("syn_v3h", str(fid))] = jp
    return lut


def gt_pose(jp):
    d = json.load(open(jp))
    o = d["objects"][0]
    return np.array(o["pose_transform"], float) if "pose_transform" in o else None


# ---------- 2D front/rear separation from gt8 ----------
def front_rear_sep(gt8):
    """depth 변(front->rear) 2D 길이 median. 저앙각서 이게 붕괴하면 ill-conditioned.
    depth edges (camera-facing): 0-4, 1-5, 2-6, 3-7. GT8 는 항상 8점 완비."""
    g = np.array(gt8, float)
    edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
    ds = [np.linalg.norm(g[a] - g[b]) for a, b in edges]
    return float(np.median(ds))


def cuboid_diag(gt8):
    g = np.array(gt8, float)
    return float(np.linalg.norm(g.max(0) - g.min(0)))


def bin_of(e):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= e < hi:
            return i
    return len(BINS) - 1


def agg(vals):
    a = np.array([v for v in vals if v is not None], float)
    if a.size == 0:
        return {"n": 0, "median": None, "mean": None, "p90": None}
    return {"n": int(a.size), "median": round(float(np.median(a)), 2),
            "mean": round(float(a.mean()), 2), "p90": round(float(np.percentile(a, 90)), 2)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = json.load(open(STAGE17))["records"]
    lut = build_path_lut()

    # attach elevation to each record
    enriched, miss = [], 0
    for r in recs:
        key = (r["dom"], str(r["fid"]))
        jp = lut.get(key)
        elev = None
        if jp:
            T = gt_pose(jp)
            if T is not None:
                elev = elev_from_pose(T)
        if elev is None:
            miss += 1
            continue
        r2 = dict(r)
        r2["elev"] = round(float(elev), 2)
        # front/rear err (order-canonical e_corner already matched to gt8)
        ec = r["e_corner"]
        ef = [ec[i] for i in FRONT if ec[i] is not None]
        er = [ec[i] for i in REAR if ec[i] is not None]
        if not ef or not er:            # need both faces detected to compare
            continue
        r2["e_front"] = float(np.mean(ef))
        r2["e_rear"] = float(np.mean(er))
        r2["n_front_det"] = len(ef)
        r2["n_rear_det"] = len(er)
        r2["fr_sep"] = front_rear_sep(r["gt8"])
        r2["cub_diag"] = cuboid_diag(r["gt8"])
        r2["fr_sep_frac"] = r2["fr_sep"] / max(r2["cub_diag"], 1e-6)
        enriched.append(r2)

    print(f"[enrich] {len(enriched)}/{len(recs)} with elevation ({miss} no-pose)")

    # ---- bin aggregation: real vs syn, front vs rear ----
    def by_bin(group_recs):
        out = {}
        for i, lbl in enumerate(BIN_LBL):
            g = [r for r in group_recs if bin_of(r["elev"]) == i]
            rear_spike = sum(1 for r in g if r["e_rear"] > SPIKE_PX)
            front_spike = sum(1 for r in g if r["e_front"] > SPIKE_PX)
            out[lbl] = {
                "n": len(g),
                "rear_err": agg([r["e_rear"] for r in g]),
                "front_err": agg([r["e_front"] for r in g]),
                "rear_spike_rate": round(rear_spike / len(g), 3) if g else None,
                "front_spike_rate": round(front_spike / len(g), 3) if g else None,
                "fr_sep_px": agg([r["fr_sep"] for r in g]),
                "fr_sep_frac": agg([r["fr_sep_frac"] for r in g]),
                "elev_span": [round(min(r["elev"] for r in g), 1),
                              round(max(r["elev"] for r in g), 1)] if g else None,
            }
        return out

    real = [r for r in enriched if r["is_real"]]
    syn = [r for r in enriched if not r["is_real"]]
    by_dom = {}
    for dom in ("outside", "night", "manual", "cad", "syn_val", "syn_v3h"):
        dr = [r for r in enriched if r["dom"] == dom]
        if dr:
            by_dom[dom] = by_bin(dr)

    # ---- finer REAL bins (real is the reliable/clean signal) ----
    fine = [(-90, 2), (2, 4), (4, 6), (6, 8), (8, 15), (15, 25), (25, 90)]
    fine_lbl = ["<2", "2-4", "4-6", "6-8", "8-15", "15-25", "25+"]
    real_fine = {}
    for (lo, hi), lbl in zip(fine, fine_lbl):
        g = [r for r in real if lo <= r["elev"] < hi]
        rspk = sum(1 for r in g if r["e_rear"] > SPIKE_PX)
        real_fine[lbl] = {
            "n": len(g),
            "rear_err": agg([r["e_rear"] for r in g]),
            "front_err": agg([r["e_front"] for r in g]),
            "rear_spike_rate": round(rspk / len(g), 3) if g else None,
            "fr_sep_px": agg([r["fr_sep"] for r in g]),
            "fr_sep_frac": agg([r["fr_sep_frac"] for r in g]),
        }

    # ---- (나) discrimination: does rear_err track geometry(frsep) or not? ----
    import numpy as _np
    ra = _np.array([[r["elev"], r["fr_sep"], r["e_rear"]] for r in real], float)
    def _sp(x, y):
        try:
            from scipy.stats import spearmanr
            return round(float(spearmanr(x, y)[0]), 3)
        except Exception:
            return None
    low = [r for r in real if r["elev"] < 8]
    discrim = {
        "spearman_elev_frsep": _sp(ra[:, 0], ra[:, 1]),
        "spearman_frsep_rearerr": _sp(ra[:, 1], ra[:, 2]),
        "spearman_elev_rearerr": _sp(ra[:, 0], ra[:, 2]),
        "low_elev(<8)_frsep_med": round(float(_np.median([r["fr_sep"] for r in low])), 1),
        "low_elev(<8)_rearerr_med": round(float(_np.median([r["e_rear"] for r in low])), 1),
        "note": ("frsep>>rear_err & no frsep~rear_err corr => geometry observable, "
                 "model-limited (가 data). frsep~rear_err at <2deg => degenerate tail (나)."),
    }

    result = {
        "n_enriched": len(enriched),
        "spike_px": SPIKE_PX,
        "bins_deg": BIN_LBL,
        "elev_definition": "angle of cam->pallet ray above pallet top plane; edge-on~0deg, top-down~90deg. [verified: outside ground frames 5-7deg, constructed top-down 90deg]",
        "syn_caveat": "syn_val/syn_v3h pose_transform is mixed_v8 convention (height axis NOT col1); elevation & rear numbers there confounded by domain-gap (front also 105px). Excluded from transition analysis.",
        "real_bins": by_bin(real),
        "real_fine_bins": real_fine,
        "discrimination_ga_vs_na": discrim,
        "by_domain": by_dom,
        "real_elev_range": [round(min(r["elev"] for r in real), 1),
                            round(max(r["elev"] for r in real), 1)] if real else None,
    }

    # ---- (가) training elevation coverage: v3 train batches + addon + syn_val ----
    cov = {}
    train_specs = [
        ("v3_train_b000-008", [os.path.join(ROOT, "challenge/data/02_synthetic/training/v3", f"batch_{i:03d}")
                               for i in range(9)]),   # b009 = held-out (eval), exclude
        ("addon_v1_train", [os.path.join(ROOT, "challenge/data/02_synthetic/training/addon_v1_train")]),
        ("syn_val", [os.path.join(ROOT, "data/pallet/training_data/val")]),
    ]
    # syn_val 은 obj 'location' 이 camera-frame 이라 world elevation 불가 + mixed_v8 pose 신뢰불가 -> 제외
    train_specs = [s for s in train_specs if s[0] != "syn_val"]
    cov_hist = {}
    for name, dirs in train_specs:
        els = []
        for dd in dirs:
            for jp in sorted(glob.glob(os.path.join(dd, "*.json"))):
                if jp.endswith(".face_centers.json"):
                    continue
                try:
                    d = json.load(open(jp))
                except Exception:
                    continue
                # v3/addon: object 'location' is WORLD frame, camera has location_worldframe
                if ("camera_data" in d and "location_worldframe" in d.get("camera_data", {})
                        and d.get("objects")):
                    e = elev_from_world(d)
                else:
                    e = None
                if e is not None:
                    els.append(e)
        if not els:
            cov[name] = {"n": 0}
            continue
        a = np.array(els)
        hist = {lbl: int(((a >= BINS[i][0]) & (a < BINS[i][1])).sum()) for i, lbl in enumerate(BIN_LBL)}
        cov[name] = {"n": len(els), "median": round(float(np.median(a)), 1),
                     "p10": round(float(np.percentile(a, 10)), 1),
                     "p90": round(float(np.percentile(a, 90)), 1),
                     "frac_below_10deg": round(float((a < 10).mean()), 3),
                     "frac_below_15deg": round(float((a < 15).mean()), 3),
                     "hist": hist}
        cov_hist[name] = a
    result["train_elev_coverage"] = cov

    json.dump(result, open(os.path.join(OUT_DIR, "elevation_threshold.json"), "w"), indent=1)

    # ---- txt table ----
    write_table(result)
    write_summary(result, real, syn)
    print(f"[done] -> {OUT_DIR}")
    return result


def _fmt(a):
    return f"n={a['n']:>4} med={str(a['median']):>6} p90={str(a['p90']):>7}"


def write_table(res):
    L = []
    L.append("STAGE18 rear-corner collapse vs elevation (REAL only; syn=domain-gap confounded, excluded)")
    L.append("elevation: edge-on(카메라=팔레트높이)~0deg, top-down(부감)~90deg\n")
    L.append("=== REAL fine bins ===")
    L.append(f"{'elev':>7} {'n':>4} | {'rear_med':>8} {'front_med':>9} | "
             f"{'rearspk':>7} | {'frsep_px':>8} {'frsep_frac':>10}")
    L.append("-" * 70)
    for lbl in ["<2", "2-4", "4-6", "6-8", "8-15", "15-25", "25+"]:
        b = res["real_fine_bins"][lbl]
        L.append(f"{lbl:>7} {b['n']:>4} | {str(b['rear_err']['median']):>8} "
                 f"{str(b['front_err']['median']):>9} | "
                 f"{str(b['rear_spike_rate']):>7} | "
                 f"{str(b['fr_sep_px']['median']):>8} {str(b['fr_sep_frac']['median']):>10}")
    L.append("")
    d = res["discrimination_ga_vs_na"]
    L.append("=== (가)/(나) discrimination ===")
    L.append(f"  spearman(elev, frsep)      = {d['spearman_elev_frsep']}  (frsep grows with elev)")
    L.append(f"  spearman(frsep, rear_err)  = {d['spearman_frsep_rearerr']}  (~0 => rear_err NOT geometry-limited)")
    L.append(f"  spearman(elev, rear_err)   = {d['spearman_elev_rearerr']}")
    L.append(f"  low-elev(<8) frsep_med={d['low_elev(<8)_frsep_med']}px  rear_err_med={d['low_elev(<8)_rearerr_med']}px")
    L.append("")
    L.append("=== per-domain rear_med (elev bin) ===")
    L.append(f"{'dom':>9} | " + " ".join(f"{l:>8}" for l in BIN_LBL))
    for dom, bb in res["by_domain"].items():
        row = []
        for l in BIN_LBL:
            b = bb[l]
            row.append(f"{str(b['rear_err']['median']):>4}/{b['n']:<3}" if b["n"] else f"{'-':>8}")
        L.append(f"{dom:>9} | " + " ".join(f"{c:>8}" for c in row))
    L.append("  (cell = rear_med / n)\n")
    L.append("=== TRAIN elevation coverage (가 검증) ===")
    L.append(f"{'set':>18} {'n':>6} {'med':>6} {'p10':>6} {'<10deg':>7} {'<15deg':>7} | hist bins")
    for name, c in res["train_elev_coverage"].items():
        if c.get("n", 0) == 0:
            L.append(f"{name:>18} n=0")
            continue
        h = c["hist"]
        L.append(f"{name:>18} {c['n']:>6} {c['median']:>6} {c['p10']:>6} "
                 f"{c['frac_below_10deg']:>7} {c['frac_below_15deg']:>7} | "
                 + " ".join(f"{l}:{h[l]}" for l in BIN_LBL))
    open(os.path.join(OUT_DIR, "tables.txt"), "w").write("\n".join(L))
    print("\n".join(L))


def write_summary(res, real, syn):
    # nothing heavy here; summary.md is written by hand-analysis in the agent message.
    pass


if __name__ == "__main__":
    main()
