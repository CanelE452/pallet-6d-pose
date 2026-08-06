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

import argparse, csv, hashlib, importlib.util, json, os, pathlib, random
import subprocess, sys, time, traceback
from typing import Any, Optional

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE0 = ROOT / "scripts/stage0"
for _e in (STAGE0, ROOT / "Deep_Object_Pose/common", ROOT / "Deep_Object_Pose/train",
           ROOT / "challenge/scripts", ROOT / "scripts/data_prep/blender"):
    if str(_e) not in sys.path:
        sys.path.insert(0, str(_e))

OUT = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/canonical_corner_audit/edge_mandatory_fast_search")
DATA = ROOT / "data/pallet/training_data/pallet6d_v2_10k"
ALLDIR = DATA / "all"
WEIGHTS = ROOT / "weights/paper_s2_edge_fast"
A1_CKPT = ROOT / "weights/paper_s2_pdg/A1/epoch_003.pth"
A1_SHA = "00a0dcd8730e21d14b8a86e2f2a398650b78026006e4e358eabc438148fb9657"
SPLIT_SHA = "9a755438dcb55e0ff60415d5b2f861a29e60b23d921a2e0985a23eb2e214415f"

GRID, IMAGE, FACTOR = 50, 400, 8
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
    spec = importlib.util.spec_from_file_location("screen", STAGE0 / "paper_s2_corner_replacement_screen.py")
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


def phase_subsets(st):
    train = split_rows("train"); idx = index_rows()
    def difficulty(r):
        m = idx[r["index"]]
        side = float(m["bbox_vis_min_side_px"] or 0); vkp = int(m["visible_kp_count"] or 0)
        tiny = str(m["tiny_warning"]).lower() in ("true", "1")
        if vkp <= 5 or tiny or side < 24: return "hard"
        if vkp == 8 and side >= 60: return "easy"
        return "medium"
    pools = {"hard": [], "medium": [], "easy": []}
    for r in train: pools[difficulty(r)].append(r["index"])
    for k in pools: pools[k].sort()
    def take(q, seed):
        rng = random.Random(seed); out = []
        for k, n in q.items():
            avail = pools[k]
            if len(avail) < n: raise RuntimeError(f"HARD_BLOCKED_SUBSET_QUOTA:{k} {len(avail)}<{n}")
            out.extend(rng.sample(avail, n))
        return sorted(out)
    c6 = take(QUOTA_6K, 6); s2 = sorted(random.Random(2).sample(c6, 2000))
    # keep the nesting exact: 2k drawn from 6k, 512 from 2k
    s512 = sorted(random.Random(512).sample(s2, 512))
    for name, ids in (("smoke512", s512), ("search2k", s2), ("confirm6k", c6)):
        with open(OUT / f"{name}_manifest.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["index"]); w.writerows([[i] for i in ids])
    hold = {r["index"] for r in split_rows("validation")} | {r["index"] for r in split_rows("untouched")}
    assert set(s512) <= set(s2) <= set(c6) <= {r["index"] for r in train}
    assert not (set(c6) & hold)
    atomic(OUT / "nested_subset_summary.json", json.dumps(
        {"pools": {k: len(v) for k, v in pools.items()},
         "smoke512": len(s512), "search2k": len(s2), "confirm6k": len(c6),
         "nesting_exact": True, "holdout_inclusion": 0}, indent=1))
    log(f"[subsets] pools {[f'{k}:{len(v)}' for k,v in pools.items()]}  512/2000/6000 nested")


def phase_smoke(st):
    status = json.loads((OUT / "cigm_oracle.json").read_text())["CIGM_ORACLE_STATUS"]
    if status != "PASS":
        raise RuntimeError("CIGM_ORACLE_STATUS != PASS; refusing to instantiate modules")
    log("[smoke] CIGM PASS -- modules may be instantiated (training wiring pending)")
    atomic(OUT / "smoke.json", json.dumps({"status": "PENDING_TRAINING_WIRING"}, indent=1))


def phase_round1(st):
    atomic(OUT / "round1_metrics.json", json.dumps({"status": "PENDING_TRAINING_WIRING"}, indent=1))
    log("[round1] pending training wiring")


def phase_decide(st):
    atomic(OUT / "final_decision.json", json.dumps(
        {"decision": "DATA_GATES_ONLY", "note": "training wiring not yet implemented"}, indent=1))


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
                 "resume": f"python scripts/stage0/edge_mandatory_fast_search.py resume"}, indent=1))
            log(f"[{p}] FAILED: {e}"); sys.exit(1)
        st.set(p, "DONE", seconds=round(time.time() - t0, 1))
    log("[runner] all phases done")


if __name__ == "__main__":
    main()
