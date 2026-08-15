"""Durable PEQ Round-1 pipeline: data gates, then a nested architecture search.

Runs unattended.  Every phase writes its state atomically, so a kill or a crash
resumes from the last completed phase rather than from the beginning.  The
geometry gates run before any module is instantiated: a learned head on top of a
broken projection would train happily and mean nothing.

Locked upstream and never re-derived here: the appearance-combination split, the
projection convention, and the loader coordinate contract (image 400x400,
refine_keypoints and beliefs on a 50 grid, factor 8).
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse, csv, hashlib, importlib.util, json, os, pathlib, random
import subprocess, sys, time, traceback
from typing import Any, Optional

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
STAGE0 = ROOT / "scripts/stage0"
for _e in (STAGE0, ROOT / "Deep_Object_Pose/common", ROOT / "Deep_Object_Pose/train",
           ROOT / "challenge/scripts", ROOT / "scripts/data_prep/blender"):
    if str(_e) not in sys.path:
        sys.path.insert(0, str(_e))

OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
DATA = ROOT / "data/pallet/training_data/pallet6d_v2_10k"
ALLDIR = DATA / "all"
WEIGHTS = ROOT / "weights/paper_s2/paper_s2_edge_fast"
A1_CKPT = ROOT / "weights/paper_s2/paper_s2_pdg/A1/epoch_003.pth"
A1_SHA = "00a0dcd8730e21d14b8a86e2f2a398650b78026006e4e358eabc438148fb9657"
SPLIT_SHA = "9a755438dcb55e0ff60415d5b2f861a29e60b23d921a2e0985a23eb2e214415f"

GRID, IMAGE, FACTOR = 50, 400, 8
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEALED = ("capturenight08", "capturenight09", "capturepallet07", "capturepallet09",
          "testset_full8_manifest", "handannot17")
PHASES = ("eligibility", "mask-audit", "edge-role", "target-parity", "cigm-oracle",
          "subsets", "smoke", "round1", "decide")
SMOKE_STEPS, BATCH = 200, 12
QUOTA_2K = {"hard": 900, "medium": 700, "easy": 400}
QUOTA_6K = {"hard": 2700, "medium": 2100, "easy": 1200}
ROUND1 = {"orientation_deg": 15.0, "offset_cells": 5.0, "cigm_valid": 0.50,
          "edge_only_le20": 0.20, "id12_gain": 0.03, "r4_gain": 1,
          "far_gt50_increase": 0.05, "shuffle_drop": 0.10}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
def atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = path.with_suffix(path.suffix + ".tmp"); t.write_text(payload, "utf-8")
    os.replace(t, path)
def guard(p):
    s = str(p)
    for tok in SEALED:
        if tok in s: raise RuntimeError(f"BLOCKED: sealed token {tok} in {s}")
    return p


class State:
    def __init__(self):
        self.path = OUT / "state.json"
        self.d = json.loads(self.path.read_text()) if self.path.is_file() else {"phases": {}}
        self.d.setdefault("phases", {})
    def get(self, p): return self.d["phases"].get(p, {}).get("status", "PENDING")
    def set(self, p, s, **x):
        e = self.d["phases"].setdefault(p, {})
        e.update({"status": s, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        e.update(x); atomic(self.path, json.dumps(self.d, indent=1))
    def beat(self, phase, note=""):
        atomic(OUT / "heartbeat.json", json.dumps(
            {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "pid": os.getpid(), "phase": phase, "note": note}, indent=1))


_M = {}
def mods():
    if _M: return _M
    spec = importlib.util.spec_from_file_location("screen", STAGE0 / "paper_s2" / "paper_s2_corner_replacement_screen.py")
    screen = importlib.util.module_from_spec(spec); spec.loader.exec_module(screen)
    import instance_edge_topology as IET, blender_math as BM
    _M.update({"screen": screen, "IET": IET, "BM": BM})
    return _M


def split_rows(which=None):
    rows = list(csv.DictReader(open(OUT / "paper_group_split.csv")))
    return [r for r in rows if which is None or r["split"] == which]


def index_rows():
    return {r["index"]: r for r in csv.DictReader(open(DATA / "index.csv"))}


# ---------------------------------------------------------------- phases
def phase_eligibility(st):
    idx = index_rows(); rows = []; excl = {}
    for i in range(20000):
        stem = f"{i:06d}"; m = idx[stem]
        try:
            d = json.loads(guard(ALLDIR / f"{stem}.json").read_text("utf-8"))
            cd, o = d["camera_data"], d["objects"][0]
            K = [cd["intrinsics"][k] for k in ("fx", "fy", "cx", "cy")]
            cub = np.asarray(o["cuboid"], float); pc = np.asarray(o["projected_cuboid"], float)
            cen = np.asarray(o["projected_cuboid_centroid"], float)
            dim = [o["dimensions_m"][k] for k in ("width", "depth", "height")]
            vkp = int(m["visible_kp_count"] or 0)
            ok = (np.isfinite(K).all() and cub.shape == (8, 3) and np.isfinite(cub).all()
                  and pc.shape == (8, 2) and np.isfinite(pc).all() and np.isfinite(cen).all()
                  and np.isfinite(dim).all() and vkp >= 4)
            reason = "" if ok else "schema_or_visible_kp"
        except Exception as e:
            ok, reason, vkp = False, f"read:{type(e).__name__}", 0
        if not ok: excl[reason] = excl.get(reason, 0) + 1
        rows.append({"index": stem, "frame_uid": f'{m["run"]}|{m["shard"]}|{m["frame_id"]}',
                     "eligible": ok, "reason": reason, "visible_kp": vkp,
                     "bbox_min_side": m["bbox_vis_min_side_px"], "tiny": m["tiny_warning"],
                     "mode": m["diagnostic_mode"], "pallet": m["pallet_type"]})
        if i % 4000 == 0: st.beat("eligibility", f"{i}/20000")
    n = len(rows); e = sum(r["eligible"] for r in rows); loss = (n - e) / n
    with open(OUT / "eligibility_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    atomic(OUT / "eligibility_summary.json", json.dumps(
        {"total": n, "eligible": e, "loss": loss, "exclusions": excl,
         "not_filters": ["tiny_warning", "pnp_size_eligible_*", "PnP solver success"],
         "visible_kp_source": "source metadata; sample['visibility'] never used"}, indent=1))
    log(f"[eligibility] {e}/{n} eligible  loss {100*loss:.3f}%  {excl}")
    if loss > 0.01: raise RuntimeError("HARD_BLOCKED_DATASET_ELIGIBILITY")


def phase_mask_audit(st):
    import cv2
    idx = index_rows(); bad = {"missing": 0, "unreadable": 0, "res": 0, "empty_amodal": 0}
    viol = []; rows = []
    for i in range(0, 20000, 2):
        m = idx[f"{i:06d}"]
        d = json.loads((ALLDIR / f"{i:06d}.json").read_text("utf-8"))["camera_data"]
        mv, ma = DATA / m["mask_visible"], DATA / m["mask_amodal"]
        if not (mv.is_file() and ma.is_file()): bad["missing"] += 1; continue
        a = cv2.imread(str(ma), cv2.IMREAD_GRAYSCALE); v = cv2.imread(str(mv), cv2.IMREAD_GRAYSCALE)
        if a is None or v is None: bad["unreadable"] += 1; continue
        if a.shape != (d["height"], d["width"]): bad["res"] += 1; continue
        ab, vb = a > 127, v > 127
        if not ab.any(): bad["empty_amodal"] += 1
        out = float((vb & ~ab).sum()) / max(vb.size, 1)
        viol.append(out)
        rows.append({"index": f"{i:06d}", "amodal_px": int(ab.sum()), "visible_px": int(vb.sum()),
                     "visible_outside_amodal_frac": out, "amodal_ge_visible": bool(ab.sum() >= vb.sum())})
        if i % 4000 == 0: st.beat("mask-audit", f"{i}/20000")
    va = np.array(viol) if viol else np.zeros(1)
    ge = np.mean([r["amodal_ge_visible"] for r in rows]) if rows else 0.0
    ok = (bad["missing"] == 0 and bad["unreadable"] == 0 and bad["res"] == 0
          and bad["empty_amodal"] == 0 and float(np.percentile(va, 99)) <= 0.001 and ge >= 0.995)
    with open(OUT / "mask_audit.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    atomic(OUT / "mask_audit_summary.json", json.dumps(
        {"checked": len(rows), "problems": bad,
         "visible_outside_amodal_p99": float(np.percentile(va, 99)),
         "amodal_ge_visible_rate": float(ge), "E3_mask_ok": bool(ok),
         "status": "OK" if ok else "E3_MASK_SEMANTICS_BLOCKED"}, indent=1))
    log(f"[mask] problems {bad}  outside p99 {np.percentile(va,99):.2e}  "
        f"amodal>=visible {100*ge:.1f}%  E3 {'OK' if ok else 'BLOCKED'}")


def phase_edge_role(st):
    IET = mods()["IET"]; topo = IET.build_topology()
    inc = {int(k): v for k, v in topo["corner_edge_incidence"].items()}
    roles = [{"query_id": k, "corner_pair": topo["edges"][k], "axis_class": topo["edge_classes"][k],
              "incident_corners": topo["edges"][k]} for k in range(12)]
    assert len(roles) == 12 and all(len(v) == 3 for v in inc.values())
    atomic(OUT / "edge_role_manifest.json", json.dumps(
        {"naming": "12 fixed cuboid-edge roles under camera_dynamic_0123_v4",
         "not": "not the same physical wooden edge across frames; corner indices are "
                "assigned per frame from the camera viewpoint",
         "topology_sha256": topo["topology_sha256"], "roles": roles,
         "corner_edge_incidence": topo["corner_edge_incidence"],
         "hungarian": False}, indent=1))
    log(f"[edge-role] 12 roles, incidence 3/corner, topology sha {topo['topology_sha256'][:12]}")


def _squash_to_grid(pts, w, h):
    return np.stack([pts[:, 0] * GRID / w, pts[:, 1] * GRID / h], 1)


def _edge_targets(corners_grid, edges):
    out = []
    for (i, j) in edges:
        a, b = corners_grid[i], corners_grid[j]
        c = 0.5 * (a + b); d = b - a; L = float(np.hypot(*d))
        u = d / max(L, 1e-9)
        out.append({"centre": c, "direction": u, "half_length": L / 2.0})
    return out


def phase_target_parity(st):
    IET, BM = mods()["IET"], mods()["BM"]
    topo = IET.build_topology(); edges = [tuple(e) for e in topo["edges"]]
    rows = []; worst = {"endpoint": 0.0, "centre": 0.0, "angle": 0.0, "half": 0.0}
    for i in range(0, 20000, 20):
        d = json.loads((ALLDIR / f"{i:06d}.json").read_text("utf-8"))
        cd, o = d["camera_data"], d["objects"][0]
        K = np.array([[cd["intrinsics"]["fx"], 0, cd["intrinsics"]["cx"]],
                      [0, cd["intrinsics"]["fy"], cd["intrinsics"]["cy"]], [0, 0, 1.]])
        R, t = BM.build_view_matrix(cd["location_worldframe"], cd["look_worldframe"])
        Xc = (R @ np.asarray(o["cuboid"], float).T).T + t
        uv = (K @ Xc.T).T; A = uv[:, :2] / uv[:, 2:3]
        B = np.asarray(o["projected_cuboid"], float)
        ga = _squash_to_grid(A, cd["width"], cd["height"])
        gb = _squash_to_grid(B, cd["width"], cd["height"])
        ta, tb = _edge_targets(ga, edges), _edge_targets(gb, edges)
        for x, y in zip(ta, tb):
            worst["centre"] = max(worst["centre"], float(np.abs(x["centre"] - y["centre"]).max()))
            worst["half"] = max(worst["half"], abs(x["half_length"] - y["half_length"]))
            cos = abs(float(np.dot(x["direction"], y["direction"]))); cos = min(cos, 1.0)
            worst["angle"] = max(worst["angle"], float(np.degrees(np.arccos(cos))))
        worst["endpoint"] = max(worst["endpoint"], float(np.abs(ga - gb).max()))
        rows.append({"index": f"{i:06d}", "endpoint": float(np.abs(ga - gb).max())})
        if i % 4000 == 0: st.beat("target-parity", f"{i}/20000")
    gates = {"endpoint<1e-5": worst["endpoint"] < 1e-5, "centre<1e-5": worst["centre"] < 1e-5,
             "angle<1e-5deg": worst["angle"] < 1e-5, "half<1e-5": worst["half"] < 1e-5}
    atomic(OUT / "edge_target_parity.json", json.dumps(
        {"frames": len(rows), "worst": worst, "gates": {k: bool(v) for k, v in gates.items()},
         "all_pass": all(bool(v) for v in gates.values())}, indent=1))
    log(f"[target-parity] worst {worst}  pass {all(gates.values())}")
    if not all(gates.values()): raise RuntimeError("HARD_BLOCKED_EDGE_TARGET_PATH_DRIFT")


def phase_cigm_oracle(st):
    import corner_incident_geometry as CIGM
    IET, BM = mods()["IET"], mods()["BM"]
    topo = IET.build_topology(); edges = [tuple(e) for e in topo["edges"]]
    inc = CIGM.incidence_table(topo)
    err = []; nonfinite = 0; conds = []
    for i in range(20000):
        d = json.loads((ALLDIR / f"{i:06d}.json").read_text("utf-8"))
        cd, o = d["camera_data"], d["objects"][0]
        g = _squash_to_grid(np.asarray(o["projected_cuboid"], float), cd["width"], cd["height"])
        tg = _edge_targets(g, edges)
        centre = torch.tensor(np.stack([t["centre"] for t in tg]), dtype=torch.float64)[None]
        direction = torch.tensor(np.stack([t["direction"] for t in tg]), dtype=torch.float64)[None]
        c, r, cond = CIGM.solve_corners(centre, direction, inc)
        cn = c[0].numpy()
        if not np.isfinite(cn).all(): nonfinite += 1; continue
        err.extend(np.hypot(*(cn - g).T).tolist()); conds.append(float(cond.median()))
        if i % 4000 == 0: st.beat("cigm-oracle", f"{i}/20000")
    a = np.array(err)
    gates = {"finite>=99.9%": (20000 - nonfinite) / 20000 >= 0.999,
             "median<0.5": float(np.median(a)) < 0.5, "p99<1.5": float(np.percentile(a, 99)) < 1.5,
             "no_nan": nonfinite == 0}
    atomic(OUT / "cigm_oracle.json", json.dumps(
        {"frames": 20000, "corner_points": len(a), "median": float(np.median(a)),
         "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
         "max": float(a.max()), "condition_median": float(np.median(conds)),
         "non_finite": nonfinite, "gates": {k: bool(v) for k, v in gates.items()},
         "CIGM_ORACLE_STATUS": "PASS" if all(gates.values()) else "FAIL"}, indent=1))
    log(f"[cigm] median {np.median(a):.4f} p99 {np.percentile(a,99):.4f} max {a.max():.4f} cell  "
        f"pass {all(gates.values())}")
    if not all(gates.values()): raise RuntimeError("HARD_BLOCKED_CIGM_ORACLE_PARITY")


def _rank(values):
    """Percentile rank within the train split.  NaN is never rank 0."""
    a = np.asarray(values, float)
    if not np.isfinite(a).all():
        raise RuntimeError("HARD_BLOCKED_DIFFICULTY_MISSING_VALUE")
    order = a.argsort(kind="stable")
    rank = np.empty(len(a), float)
    rank[order] = np.arange(len(a), dtype=float) / max(len(a) - 1, 1)
    return rank


def phase_subsets(st):
    """Difficulty is a rank over the train split, never an absolute threshold.

    The previous definition used visible_kp_count == 8 and a 60px side, chosen
    without checking the column's meaning or its distribution, and the easy pool
    came back empty.  A rank cannot be empty, and it is computed from train
    metadata alone before any model result exists.
    """
    import cv2
    idx = index_rows()
    train = split_rows("train")
    feats = []
    for k, r in enumerate(train):
        stem = r["index"]; m = idx[stem]
        d = json.loads((ALLDIR / f"{stem}.json").read_text("utf-8"))
        cd, o = d["camera_data"], d["objects"][0]
        w, h = cd["width"], cd["height"]
        pc = np.asarray(o["projected_cuboid"], float)
        inside = ((pc[:, 0] >= 0) & (pc[:, 0] < w) & (pc[:, 1] >= 0) & (pc[:, 1] < h))
        geom_in = int(inside.sum())
        border = bool(((pc[:, 0] < 4) | (pc[:, 0] > w - 4) | (pc[:, 1] < 4)
                       | (pc[:, 1] > h - 4)) & inside).__index__() if False else bool(
            (inside & ((pc[:, 0] < 4) | (pc[:, 0] > w - 4)
                       | (pc[:, 1] < 4) | (pc[:, 1] > h - 4))).any())
        mv = cv2.imread(str(DATA / m["mask_visible"]), cv2.IMREAD_GRAYSCALE)
        ma = cv2.imread(str(DATA / m["mask_amodal"]), cv2.IMREAD_GRAYSCALE)
        if mv is None or ma is None:
            raise RuntimeError("HARD_BLOCKED_DIFFICULTY_MISSING_VALUE: mask unreadable")
        av = float((mv > 127).sum()); aa = float((ma > 127).sum())
        occ = 1.0 - av / max(aa, 1.0)
        img = cv2.imread(str(ALLDIR / f"{stem}.png"))
        luma = float(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        size = float(m["bbox_vis_min_side_px"] or 0.0)
        tiny = 1.0 if str(m["tiny_warning"]).lower() in ("true", "1") else 0.0
        feats.append({"index": stem, "group": r["group_id"], "run": m["run"],
                      "geom_in": geom_in, "trunc": 1.0 if geom_in < 8 else 0.0,
                      "occ": occ, "size": size, "luma": luma,
                      "border": 1.0 if border else 0.0, "tiny": tiny,
                      "mode": m["diagnostic_mode"], "pallet": m["pallet_type"]})
        if k % 2000 == 0:
            st.beat("subsets", f"features {k}/{len(train)}")
    occ_r = _rank([f["occ"] for f in feats])
    size_r = _rank([f["size"] for f in feats])
    luma_r = _rank([f["luma"] for f in feats])
    for i, f in enumerate(feats):
        f["D"] = (3.0 * (8 - f["geom_in"]) / 4 + 2.0 * f["trunc"] + 2.0 * occ_r[i]
                  + 1.5 * (1 - size_r[i]) + 1.0 * (1 - luma_r[i])
                  + 1.0 * f["border"] + 0.5 * f["tiny"])
    feats.sort(key=lambda f: (-f["D"], hashlib.sha256(f["index"].encode()).hexdigest()))
    n = len(feats); h_end = int(round(0.45 * n)); m_end = h_end + int(round(0.35 * n))
    for i, f in enumerate(feats):
        f["difficulty"] = "hard" if i < h_end else ("medium" if i < m_end else "easy")
    pools = {k: [f for f in feats if f["difficulty"] == k] for k in ("hard", "medium", "easy")}
    log(f"[subsets] pools " + ", ".join(f"{k} {len(v)}" for k, v in pools.items()))

    def draw(quota, cap, seed, restrict=None):
        picked = []
        for level, want in quota.items():
            avail = [f for f in pools[level] if restrict is None or f["index"] in restrict]
            if len(avail) < want:
                raise RuntimeError(f"HARD_BLOCKED_SUBSET_QUOTA:{level} {len(avail)}<{want}")
            rng = random.Random(seed + hash(level) % 1000)
            # spread over appearance groups first, then fill deterministically
            bygroup = {}
            for f in avail:
                bygroup.setdefault(f["group"], []).append(f)
            for g in bygroup:
                bygroup[g].sort(key=lambda f: f["index"])
            take, used = [], {g: 0 for g in bygroup}
            groups = sorted(bygroup)
            while len(take) < want:
                progressed = False
                for g in groups:
                    if len(take) >= want:
                        break
                    if used[g] < cap and used[g] < len(bygroup[g]):
                        take.append(bygroup[g][used[g]]); used[g] += 1; progressed = True
                if not progressed:
                    raise RuntimeError(f"HARD_BLOCKED_SUBSET_QUOTA:{level} group cap {cap}")
            picked.extend(take)
            del rng
        return picked

    c6 = draw(QUOTA_6K, 50, 6)
    c6_ids = {f["index"] for f in c6}
    s2 = draw(QUOTA_2K, 20, 2, restrict=c6_ids)
    s2_ids = {f["index"] for f in s2}
    s512 = sorted(s2_ids)[:512]
    assert set(s512) <= s2_ids <= c6_ids <= {f["index"] for f in feats}
    hold = {r["index"] for r in split_rows("validation")} | {r["index"] for r in split_rows("untouched")}
    assert not (c6_ids & hold)
    for name, ids in (("smoke512", sorted(s512)), ("search2k", sorted(s2_ids)),
                      ("confirm6k", sorted(c6_ids))):
        with open(OUT / f"{name}_manifest.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["index"]); w.writerows([[i] for i in ids])
    with open(OUT / "difficulty_distribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "group", "run", "geom_in", "trunc",
                                          "occ", "size", "luma", "border", "tiny",
                                          "mode", "pallet", "D", "difficulty"])
        w.writeheader(); w.writerows(feats)
    atomic(OUT / "difficulty_definition.json", json.dumps(
        {"basis": "percentile rank within the train split; no absolute threshold",
         "formula": "3*(8-geom_in)/4 + 2*trunc + 2*occ_rank + 1.5*(1-size_rank) "
                    "+ 1*(1-luma_rank) + 1*border + 0.5*tiny",
         "proportions": {"hard": 0.45, "medium": 0.35, "easy": 0.20},
         "tie_break": "sha256(index)",
         "visible_kp_count_used": False,
         "missing_value_policy": "HARD_BLOCK, never substituted with an easy value",
         "pools": {k: len(v) for k, v in pools.items()}}, indent=1))
    atomic(OUT / "nested_subset_summary.json", json.dumps(
        {"pools": {k: len(v) for k, v in pools.items()},
         "smoke512": len(s512), "search2k": len(s2_ids), "confirm6k": len(c6_ids),
         "group_cap": {"search2k": 20, "confirm6k": 50},
         "nesting_exact": True, "holdout_inclusion": 0}, indent=1))
    log(f"[subsets] 512/{len(s2_ids)}/{len(c6_ids)} nested, holdout inclusion 0")


# ---------------------------------------------------------------- training
ARMS = ("E1", "E2")            # E3 is SKIPPED_OPTIONAL for this Round-1
LR_PEQ, LR_HEAD, WD = 3e-4, 1e-3, 1e-4
TARGET_SHARE = {"centre": 0.50, "orientation": 0.25, "length": 0.25,
                "support": 0.10, "incidence": 0.50}


def _a1():
    spec = importlib.util.spec_from_file_location("SHS", STAGE0 / "spatial_hcrm_screen.py")
    shs = importlib.util.module_from_spec(spec); sys.modules["SHS"] = shs
    spec.loader.exec_module(shs)
    return shs


def _loader(manifest_name, batch=BATCH, shuffle=True, seed=1):
    screen = mods()["screen"]
    opt = screen.canonical_options(); opt.data = [str(ALLDIR.resolve())]
    import pdg_stage1_dataset as DS
    base, _, _, _ = DS.build("A1", opt, taca_seed=seed)
    wanted = {r["index"] for r in csv.DictReader(open(OUT / manifest_name))}
    keep = [i for i, e in enumerate(base.imgs)
            if pathlib.Path(str(e[0])).stem in wanted]
    if len(keep) != len(wanted):
        raise RuntimeError(f"HARD_BLOCKED_SUBSET_JOIN {len(keep)} != {len(wanted)}")
    subset = torch.utils.data.Subset(base, keep)
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(subset, batch_size=batch, shuffle=shuffle,
                                       num_workers=4, generator=g, drop_last=True)


def edge_targets(corners, edges):
    """(B,8,2) grid corners -> per-role centre/direction/half-length/support."""
    i = torch.tensor([e[0] for e in edges], device=corners.device)
    j = torch.tensor([e[1] for e in edges], device=corners.device)
    p0, p1 = corners[:, i], corners[:, j]
    delta = p1 - p0
    length = delta.norm(dim=-1, keepdim=True)
    direction = delta / length.clamp_min(1e-6)
    centre = 0.5 * (p0 + p1)
    half = (0.5 * length).clamp_min(1e-3)
    inside0 = ((p0 >= 0) & (p0 < GRID)).all(-1)
    inside1 = ((p1 >= 0) & (p1 < GRID)).all(-1)
    # a segment with either endpoint inside, or that straddles the grid, is supported
    straddle = ((p0.min(-1).values < GRID) & (p1.max(-1).values >= 0)
                & (p0.max(-1).values >= 0) & (p1.min(-1).values < GRID))
    support = (inside0 | inside1 | straddle).float()
    regression = support * (length.squeeze(-1) > 1e-4).float()
    return {"centre": centre, "direction": direction, "half_length": half,
            "support": support, "mask": regression}


def edge_losses(pred, tgt):
    m = tgt["mask"][..., None]
    n = m.sum().clamp_min(1.0)
    import physical_edge_query as PEQmod
    centre = (torch.nn.functional.smooth_l1_loss(pred["centre"], tgt["centre"],
                                                 reduction="none") * m).sum() / (2 * n)
    orient = PEQmod.orientation_loss(pred["direction"], tgt["direction"], tgt["mask"])
    length = (torch.nn.functional.smooth_l1_loss(
        pred["half_length"].log(), tgt["half_length"].log(), reduction="none") * m).sum() / n
    support = torch.nn.functional.binary_cross_entropy_with_logits(
        pred["support_logit"].squeeze(-1), tgt["support"])
    return {"centre": centre, "orientation": orient, "length": length, "support": support}


def calibrate_v2(median: dict) -> tuple[dict, dict]:
    """Raw normalisation, no lower clamp.

    The previous rule clipped lambda into [1e-3, 100].  With a belief anchor near
    0.003 and edge medians near 30, four of five lambdas hit the floor -- and the
    floor raised them, so centre carried 26x its intended weight and the target
    shares stopped meaning anything.  The safeguard is now a block, not a clip:
    an out-of-range lambda stops the run instead of silently rewriting the
    objective.
    """
    anchor = median["belief"]
    lam, rows = {}, {}
    for k, share in TARGET_SHARE.items():
        raw = share * anchor / max(median[k], 1e-9)
        if not np.isfinite(raw) or not (1e-8 <= raw <= 1e4):
            raise RuntimeError(f"HARD_BLOCKED_LOSS_SCALE_PATHOLOGY:{k}={raw:g}")
        lam[k] = float(raw)
        realized = lam[k] * median[k] / max(anchor, 1e-12)
        rows[k] = {"median": median[k], "lambda": lam[k],
                   "contribution": lam[k] * median[k], "realized_share": realized,
                   "target_share": share,
                   "relative_error": abs(realized / share - 1.0)}
    worst = max(r["relative_error"] for r in rows.values())
    report = {"anchor": "frozen A1 base belief, masked MSE over corner channels 0:8",
              "median_belief_anchor": anchor, "components": rows,
              "lower_clamp": "removed", "numerical_bound": "1e-8 <= lambda <= 1e4, HARD_BLOCK",
              "max_relative_error": worst,
              "CALIBRATION_V2": "PASS" if worst <= 0.01 else "FAIL",
              "frozen": True}
    if worst > 0.01:
        raise RuntimeError(f"HARD_BLOCKED_CALIBRATION_FIDELITY:{worst:.3e}")
    return lam, report


def build_arm(arm, seed):
    import physical_edge_query as PEQmod, edge_guided_corner_fusion as EG
    import spatial_hcrm as HC
    torch.manual_seed(seed)
    mod = {"peq": PEQmod.PhysicalEdgeQueryHead().to(DEV),
           "egcr": EG.EdgeGuidedCornerResidual().to(DEV)}
    groups = [{"params": list(mod["peq"].parameters()), "lr": LR_PEQ},
              {"params": list(mod["egcr"].parameters()), "lr": LR_HEAD}]
    if arm == "E2":
        mod["hcrm"] = HC.build("H2", seed).to(DEV)
        groups.append({"params": list(mod["hcrm"].parameters()), "lr": LR_HEAD})
    return mod, torch.optim.AdamW(groups, weight_decay=WD)


def arm_forward(arm, mod, feature, base_belief, inc, edges,
                permutation=None, zero=False, gt_corners=None):
    import corner_incident_geometry as CG, edge_guided_corner_fusion as EG
    pred = mod["peq"](feature)
    centre, direction = pred["centre"], pred["direction"]
    if permutation is not None:
        centre, direction = centre[:, permutation], direction[:, permutation]
    if gt_corners is not None:
        t = edge_targets(gt_corners, edges)
        centre, direction = t["centre"], t["direction"]
    corners, residual, condition = CG.solve_corners(centre, direction, inc)
    proposals = CG.render_proposals(corners, GRID, 2.0)
    if zero:
        proposals = torch.zeros_like(proposals)
    edge_res = mod["egcr"](base_belief, proposals)
    if zero:
        edge_res = torch.zeros_like(edge_res)
    near = mod["hcrm"](feature) if arm == "E2" else None
    final = EG.compose(base_belief, edge_res, near)
    return {"pred": pred, "corners": corners, "condition": condition,
            "final": final, "edge_residual": edge_res, "proposals": proposals}


def run_training(st, manifest, steps, epochs, seed, tag):
    shs = _a1()
    a1 = shs.FrozenA1().to(DEV)
    before = a1.checksum()
    IET = mods()["IET"]; topo = IET.build_topology()
    edges = [tuple(e) for e in topo["edges"]]
    import corner_incident_geometry as CG
    inc = CG.incidence_table(topo)
    loader = _loader(manifest, seed=seed)
    arms = {a: build_arm(a, seed) for a in ARMS}
    forwards = {"a1": 0}
    lam = None
    trace = {a: {"steps": 0, "losses": [], "grad_peq": 0.0, "grad_egcr": 0.0,
                 "grad_hcrm": 0.0, "residual_max": 0.0} for a in ARMS}
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            if steps and step >= steps:
                break
            img = batch["img"].to(DEV)
            feature, base, _aff = a1(img)          # exactly one A1 forward per batch
            forwards["a1"] += 1
            feature, base = feature.detach(), base.detach()
            kp = batch["refine_keypoints"][:, :8].to(DEV).float()
            tgt = edge_targets(kp, edges)
            gt_belief = batch["beliefs"][:, :8].to(DEV).float()
            bmask = batch["belief_channel_mask"][:, :8].to(DEV).float()[:, :, None, None]
            valid = (batch["refine_keypoints_valid"][:, :8].to(DEV) > 0)
            inside = ((kp >= 0) & (kp < GRID)).all(-1) & valid
            for a in ARMS:
                mod, opt = arms[a]
                out = arm_forward(a, mod, feature, base, inc, edges)
                el = edge_losses(out["pred"], tgt)
                inc_mask = inside[..., None].float()
                el["incidence"] = (torch.nn.functional.smooth_l1_loss(
                    out["corners"], kp, reduction="none") * inc_mask).sum() / inc_mask.sum().clamp_min(1.0) / 2
                belief = (((out["final"][:, :8] - gt_belief) ** 2) * bmask).mean()
                if lam is None and step < 16:
                    # anchor is the frozen A1 base belief, identical for every arm,
                    # so E1 and E2 calibrate on one scale rather than on their own output
                    anchor = (((base[:, :8] - gt_belief) ** 2) * bmask).mean()
                    trace[a]["losses"].append({k: float(v) for k, v in el.items()}
                                              | {"belief": float(anchor)})
                    continue
                loss = belief + sum(lam[k] * el[k] for k in el)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"HARD_BLOCKED_NONFINITE_LOSS {a}")
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                trace[a]["steps"] += 1
                trace[a]["grad_peq"] = float(sum(
                    p.grad.abs().sum() for p in mod["peq"].parameters() if p.grad is not None))
                trace[a]["grad_egcr"] = float(sum(
                    p.grad.abs().sum() for p in mod["egcr"].parameters() if p.grad is not None))
                if a == "E2":
                    trace[a]["grad_hcrm"] = float(sum(
                        p.grad.abs().sum() for p in mod["hcrm"].parameters() if p.grad is not None))
                trace[a]["residual_max"] = max(trace[a]["residual_max"],
                                               float(out["edge_residual"].abs().max()))
            if lam is None and step >= 15:
                med = {k: float(np.median([r[k] for r in trace[ARMS[0]]["losses"]]))
                       for k in trace[ARMS[0]]["losses"][0]}
                lam, report = calibrate_v2(med)
                atomic(OUT / f"{tag}_lambda.json", json.dumps(report, indent=1))
                log(f"[{tag}] lambda {json.dumps({k: float(f'{v:.4g}') for k,v in lam.items()})}")
                log(f"[{tag}] CALIBRATION_V2 {'PASS' if report['CALIBRATION_V2']=='PASS' else 'FAIL'}"
                    f"  max share error {report['max_relative_error']:.2e}")
            step += 1
            if step % 20 == 0:
                st.beat(tag, f"step {step}")
        if steps and step >= steps:
            break
    if a1.checksum() != before:
        raise RuntimeError("HARD_BLOCKED_A1_CHANGED")
    return {"arms": arms, "trace": trace, "a1_forwards": forwards["a1"], "steps": step,
            "a1_unchanged": True, "lambda": lam, "a1": a1, "inc": inc, "edges": edges}


def phase_smoke(st):
    status = json.loads((OUT / "cigm_oracle.json").read_text())["CIGM_ORACLE_STATUS"]
    if status != "PASS":
        raise RuntimeError("CIGM_ORACLE_STATUS != PASS; refusing to instantiate modules")
    r = run_training(st, "smoke512_manifest.csv", steps=50 + 16, epochs=99, seed=1, tag="smoke")
    checks = {}
    for a in ARMS:
        t = r["trace"][a]
        checks[a] = {"steps>=50": t["steps"] >= 50, "grad_peq>0": t["grad_peq"] > 0,
                     "grad_egcr>0": t["grad_egcr"] > 0,
                     "grad_hcrm>0": (t["grad_hcrm"] > 0) if a == "E2" else True,
                     "residual_grew": t["residual_max"] > 0}
    # zero-init identity and passthrough on a fresh arm
    import edge_guided_corner_fusion as EG
    fresh, _ = build_arm("E1", 1)
    base = torch.randn(2, 9, GRID, GRID, device=DEV)
    out = arm_forward("E1", fresh, torch.randn(2, 128, GRID, GRID, device=DEV), base,
                      r["inc"], r["edges"])
    checks["zero_init"] = {"residual_exact_0": float(out["edge_residual"].abs().max()) == 0.0,
                           "centroid_delta_0": EG.assert_passthrough(out["final"], base)["centroid_max_abs"] == 0.0}
    ok = all(all(v is True or v for v in c.values()) for c in checks.values())
    # checkpoint round-trip
    d = WEIGHTS / "R1B" / "smoke"; d.mkdir(parents=True, exist_ok=True)
    torch.save({k: m.state_dict() for k, m in r["arms"]["E1"][0].items()}, d / "E1.pth")
    checks["checkpoint_saved"] = (d / "E1.pth").is_file()
    atomic(OUT / "smoke.json", json.dumps(
        {"steps": r["steps"], "a1_forwards_per_batch": 1, "a1_unchanged": r["a1_unchanged"],
         "checks": checks, "trace": {a: {k: v for k, v in r["trace"][a].items() if k != "losses"}
                                     for a in ARMS},
         "lambda": r["lambda"], "passed": bool(ok)}, indent=1))
    log(f"[smoke] {json.dumps(checks)}")
    if not ok:
        raise RuntimeError("HARD_BLOCKED_SMOKE")


def phase_round1(st):
    r = run_training(st, "search2k_manifest.csv", steps=0, epochs=1, seed=1, tag="round1")
    atomic(OUT / "round1_metrics.json", json.dumps(
        {"steps": r["steps"], "a1_forwards": r["a1_forwards"], "a1_unchanged": r["a1_unchanged"],
         "lambda": r["lambda"],
         "trace": {a: {k: v for k, v in r["trace"][a].items() if k != "losses"} for a in ARMS},
         "validation": "PENDING"}, indent=1))
    for a in ARMS:
        d = WEIGHTS / "R1B" / a; d.mkdir(parents=True, exist_ok=True)
        torch.save({k: m.state_dict() for k, m in r["arms"][a][0].items()}, d / "round1.pth")
    log(f"[round1] {r['steps']} steps, A1 forwards {r['a1_forwards']}, checkpoints saved")


def phase_decide(st):
    m = json.loads((OUT / "round1_metrics.json").read_text())
    trained = {a: m["trace"][a]["steps"] > 0 and m["trace"][a]["grad_peq"] > 0 for a in ARMS}
    decision = "ROUND1_TRAINED" if any(trained.values()) else "EDGE_QUERY_LOCALIZATION_FAIL"
    atomic(OUT / "final_decision.json", json.dumps(
        {"decision": decision, "arms_trained": trained,
         "validation_and_ablations": "PENDING",
         "note": "Round-1 gates need the validation pass; training wiring is complete"},
        indent=1))
    log(f"[decide] {decision}")


DRIVERS = {"eligibility": phase_eligibility, "mask-audit": phase_mask_audit,
           "edge-role": phase_edge_role, "target-parity": phase_target_parity,
           "cigm-oracle": phase_cigm_oracle, "subsets": phase_subsets,
           "smoke": phase_smoke, "round1": phase_round1, "decide": phase_decide}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=list(DRIVERS) + ["all", "resume", "status"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    st = State()
    if a.command == "status":
        for p in PHASES: print(f"{p:16s} {st.get(p)}")
        return
    if sha_file(A1_CKPT) != A1_SHA: raise RuntimeError("HARD_BLOCKED: A1 checkpoint changed")
    if sha_file(OUT / "paper_group_split.csv") != SPLIT_SHA:
        raise RuntimeError("HARD_BLOCKED: split changed")
    todo = list(PHASES) if a.command in ("all", "resume") else [a.command]
    for p in todo:
        if st.get(p) == "DONE": log(f"[{p}] DONE -- skip"); continue
        st.set(p, "RUNNING"); st.beat(p, "start"); t0 = time.time()
        try:
            DRIVERS[p](st)
        except Exception as e:
            st.set(p, "HARD_BLOCKED" if "HARD_BLOCKED" in str(e) else "FAILED", error=repr(e))
            atomic(OUT / "failure.json", json.dumps(
                {"phase": p, "exception": repr(e), "traceback": traceback.format_exc(),
                 "resume": f"python scripts/stage0/line/edge_mandatory_fast_search.py resume"}, indent=1))
            log(f"[{p}] FAILED: {e}"); sys.exit(1)
        st.set(p, "DONE", seconds=round(time.time() - t0, 1))
    log("[runner] all phases done")


if __name__ == "__main__":
    main()
