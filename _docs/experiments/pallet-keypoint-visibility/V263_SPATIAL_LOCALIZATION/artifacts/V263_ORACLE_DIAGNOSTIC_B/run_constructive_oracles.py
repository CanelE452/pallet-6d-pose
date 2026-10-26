"""SYN-only partial-certified four-cell oracle, unchanged V58 fixed post."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np


def guard(path):
    p = Path(path)
    if not p.is_absolute() or ".." in p.parts or any(t in str(p).lower() for t in ("v112", "cache", ".pyc")):
        raise ValueError("forbidden/nonabsolute path")
    for a in (p, *p.parents):
        if a.is_symlink():
            raise ValueError("symlink artifact/ancestor")
    return p


def checked(b):
    data = guard(b["path"]).read_bytes()
    assert len(data) == b["size_bytes"] and hashlib.sha256(data).hexdigest() == b["sha256"], b["path"]
    return data


def load(name, binding):
    m = types.ModuleType(name)
    m.__file__ = binding["path"]
    sys.modules[name] = m
    exec(compile(checked(binding), m.__file__, "exec"), m.__dict__)
    return m


def write_json(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def bind(path):
    b = guard(path).read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(b).hexdigest(), "size_bytes": len(b)}


def fixed_metrics(pose, g, angle):
    if not pose["available"]:
        return {"available": False, "success_post5cm5deg": False, "success_canonical5cm5deg": False}
    t = float(np.linalg.norm(np.asarray(pose["centroid"]) - np.asarray(g["GT_t"])) * 100)
    # All three comparators are fixed from declared geometry BEFORE prediction.
    # No predicted W/D branch is applied to the GT, and no minimum is selected.
    r_post = angle(g["GT_R_post_fixed"], pose["R_physical"])
    r_canonical = angle(g["GT_R_canonical"], pose["R_physical"])
    r_cf = angle(g["GT_R_CF"], pose["R_cf"])
    return {"available": True, "T_cm": t, "R_post_fixed_deg": r_post,
            "R_canonical_C1_deg": r_canonical, "R_CF_deg": r_cf,
            "success_post5cm5deg": t < 5 and r_post < 5,
            "success_canonical5cm5deg": t < 5 and r_canonical < 5,
            "T_gt50": t > 50, "R_post_gt45": r_post > 45,
            "canonical_near90": 80 <= r_canonical <= 100}


def summarize(rows):
    d = {"n": len(rows), "available": sum(r["available"] for r in rows),
         "success_post5cm5deg": sum(r["success_post5cm5deg"] for r in rows),
         "success_canonical5cm5deg": sum(r["success_canonical5cm5deg"] for r in rows),
         "fallback_n": sum(r["fallback_used"] for r in rows),
         "min6_not_met": sum(r["retained_native_points"] < 6 for r in rows)}
    for key in ("T_cm", "R_post_fixed_deg", "R_canonical_C1_deg", "R_CF_deg"):
        vals = [r[key] for r in rows if r["available"]]
        for suffix, fn in (("mean", np.mean), ("median", np.median),
                           ("P90", lambda x: np.percentile(x, 90)),
                           ("P95", lambda x: np.percentile(x, 95)), ("max", np.max)):
            d[key + "_" + suffix] = float(fn(vals)) if vals else None
    return d


def run(entry):
    assert entry["execution_authorized"] is True and entry["scope"] == "ORIGINAL_CONSTRUCTIVE_SOURCE110_ONLY"
    assert checked(entry["code"]) == Path(__file__).read_bytes()
    scope = json.loads(checked(entry["root_scope"]))
    assert scope["approved"] is True and scope["entry_contract"] == {k: v for k, v in entry.items() if k != "root_scope"}
    out = guard(entry["output_dir"])
    assert not out.exists(), "fresh output required"
    policy = json.loads(checked(entry["partial_oracle_policy"]))
    assert policy["centroid"]["valid9_index8"] is False
    original = load("v263_old_fixed_post_api", entry["baseline_api"])
    oracle = load("v263_four_input_api", entry["oracle_core"])
    backend_raw = {k: checked(v) for k, v in entry["backend_inputs"].items()}
    backend, evaluator = original.load_backend(entry["backend_inputs"], backend_raw)
    packet_complete = json.loads(checked(entry["packet_complete"]))
    assert packet_complete["PASS"] and packet_complete["include_V255"] is False
    assert packet_complete["outputs"]["groups"] == entry["groups"]
    groups = [json.loads(x) for x in checked(entry["groups"]).splitlines()]
    assert len(groups) == 880 and len({g["group_id"] for g in groups}) == 880
    assert len({g["base_index"] for g in groups}) == 110
    assert len({g["base_index"] for g in groups if g["fit"]}) == 86
    assert len({g["base_index"] for g in groups if not g["fit"]}) == 24
    assert sum(len(g["row_ids8"]) for g in groups) == 7040
    cv2.setNumThreads(1)
    # Explicit, prospectively fixed CV frame. GT_R_CF directly projects the CF
    # diagram to saved r. The old source-only Rx(pi) is not blindly reused.
    assert entry["source_basis_bool"] is False
    out.mkdir()
    write_json(out / "START.json", {"entry": entry, "no_gt_fitted_axis": True, "symmetry_order": "UNKNOWN_NO_MINIMIZATION"})
    start = time.monotonic()
    per_frame, initial, coverage, point_rows, pose_rows = [], [], [], [], []
    quarter = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    for gi, source_group in enumerate(groups):
        g = dict(source_group)
        q, L = np.asarray(g["q8"], float), np.asarray(g["L8"], float)
        available = np.asarray(g["original_available8"], bool)
        K, xyz, cf = np.asarray(g["K"], float), np.asarray(g["canonical_dimensions_xyz"], float), np.asarray(g["CF_dimensions_xyz"], float)
        A, Rcf, Rc, t = (np.asarray(g[k], float) for k in ("canonical_to_CF_rotation", "GT_R_CF", "GT_R_canonical", "GT_t"))
        X = np.asarray(g["native_xyz_CF_frame"], float)
        assert q.shape == (8, 2) and available.shape == (8,) and xyz.shape == cf.shape == (3,)
        assert np.all(xyz > 0) and np.isfinite(q).all()
        assert np.allclose(X, backend.base["cuboid"](*cf), rtol=0, atol=1e-12)
        assert np.allclose(Rcf, Rc @ A.T, rtol=0, atol=1e-12)
        assert np.allclose(np.asarray(g["native_xyz_canonical_by_CF"]) @ A.T, X, rtol=0, atol=1e-12)
        ps = g["native_proofs8"]
        assert len(ps) == 8 and [p["native_slot"] for p in ps] == list(range(8))
        r = np.asarray([p["r"] for p in ps], float)
        cam = X @ Rcf.T + t
        hp = cam @ K.T
        reproj_gap = float(np.max(np.abs(hp[:, :2] / hp[:, 2:] - r)))
        assert reproj_gap <= 1e-6, "declared exact CF basis does not project saved r"
        assert np.allclose(cf, xyz, rtol=0, atol=1e-6) or np.allclose(cf, xyz[[2, 1, 0]], rtol=0, atol=1e-6)
        Q_truth = np.eye(3) if abs(cf[0] - xyz[0]) < 1e-6 else quarter
        g["GT_R_post_fixed"] = (Rcf @ Q_truth).tolist()
        known = np.asarray([bool(p["identity_known"] and p["identity_correct"]) for p in ps], bool)
        raw_states = [p["point_combined_state"] for p in ps]
        states = [s if s in ("D", "E", "S") else "U" for s in raw_states]
        edge = np.asarray([bool(p["direct_geometric_2edge"]) for p in ps], bool)
        search = oracle.original_search_mask(q, L, r)
        constructed = oracle.four_inputs(q, available, r, states, known, edge, search)
        configs = constructed["configurations"]
        n0 = backend.infer_masked(configs["ORIGINAL"]["points9"], K, xyz, configs["ORIGINAL"]["valid9"], source=False)
        n0_projection = None
        if n0["available"]:
            xx = backend.base["cuboid"](*n0["cf_extents"])
            camera = xx @ np.asarray(n0["R_cf"]).T + np.asarray(n0["centroid"])
            hh = camera @ K.T
            n0_projection = (hh[:, :2] / hh[:, 2:]).tolist()
        initial.append({"group_id": g["group_id"], "group_index": g["group_index"], "row_ids8": g["row_ids8"],
                        "source_indices8": g["source_indices8"], "base_index": g["base_index"], "fit": g["fit"],
                        "original_pose": n0, "initial_pose": n0, "INITIAL_T_source": "ORIGINAL_FIXED_SOLVER_NOT_GT",
                        "valid9": configs["ORIGINAL"]["valid9"].tolist(), "q8": q.tolist(), "K": K.tolist(),
                        "projected_native8": n0_projection, "source_basis_bool": False,
                        "geometry_crop_requires_prediction_only": True, "solver_unavailable_2D_fallback_required": not n0["available"]})
        coverage.append({"group_id": g["group_id"], "base_index": g["base_index"], "fit": g["fit"],
                         "family": g["family"], "variant": g["variant"], "all_native": 8,
                         "original_available": int(available.sum()), "certified_known": int(constructed["certified_known8"].sum()),
                         "certified_D": int(constructed["certified_D8"].sum()),
                         "unknown_original_support_retained": int(constructed["unknown_original_support_retained8"].sum()),
                         "location_replaced": int(constructed["location_replaced8"].sum()),
                         "direct_geometric_2edge": int(edge.sum()), "in_original_search": int(search.sum()),
                         "projection_maxabs_px": reproj_gap, "photometric_observability": "UNKNOWN_NOT_MEASURED"})
        for j, proof in enumerate(ps):
            point_rows.append({"group_id": g["group_id"], "row_id": g["row_ids8"][j], "source_index": g["source_indices8"][j],
                               "native_slot": j, "fit": g["fit"], "base_index": g["base_index"], "family": g["family"],
                               "variant": g["variant"], "original_available": bool(available[j]), "raw_combined_state": raw_states[j],
                               "native_identity_certified": bool(known[j]), "direct_geometric_2edge": bool(edge[j]),
                               "in_original_search": bool(search[j]), "location_replaced": bool(constructed["location_replaced8"][j]),
                               "q_x": float(q[j, 0]), "q_y": float(q[j, 1]), "r_x": float(r[j, 0]), "r_y": float(r[j, 1])})
        for arm, inp in configs.items():
            p = n0 if arm == "ORIGINAL" else backend.infer_masked(inp["points9"], K, xyz, inp["valid9"], source=False)
            raw = fixed_metrics(p, g, evaluator.angle)
            fallback = not p["available"] and arm != "ORIGINAL"
            effective = n0 if fallback else p
            for policy_name, pose_metric, used in (("RAW_UNAVAILABLE_PRIMARY", raw, False),
                                                   ("KEEP_ORIGINAL_NO_POSE_SENSITIVITY", fixed_metrics(effective, g, evaluator.angle), fallback)):
                per_frame.append({"group_id": g["group_id"], "base_index": g["base_index"], "fit": g["fit"],
                                  "family": g["family"], "variant": g["variant"], "arm": arm, "fallback_policy": policy_name,
                                  "retained_native_points": int(inp["valid9"][:8].sum()), "fallback_used": used,
                                  "symmetry_order": "UNKNOWN", **pose_metric})
            pose_rows.append({"group_id": g["group_id"], "arm": arm, "finite9": inp["points9"].tolist(), "valid9": inp["valid9"].tolist(),
                              "raw_pose": p, "effective_keep_original_pose": effective,
                              "GT_R_post_fixed": g["GT_R_post_fixed"], "GT_R_canonical": g["GT_R_canonical"],
                              "GT_R_CF": g["GT_R_CF"], "GT_t": g["GT_t"], "Q_truth": Q_truth.tolist()})
        if (gi + 1) % 200 == 0 or gi + 1 == len(groups):
            print("CONSTRUCTIVE_ORACLES", gi + 1, "/880", flush=True)
    write_jsonl(out / "INITIAL_T_ORIGINAL.jsonl", initial)
    initial_binding = bind(out / "INITIAL_T_ORIGINAL.jsonl")
    write_json(out / "INITIAL_T_COMPLETE.json", {"completed": True, "PASS": True, "groups": 880, "rows": 7040,
        "source": "ORIGINAL_FIXED_SOLVER_NOT_GT", "output": initial_binding, "packet_groups": entry["groups"],
        "geometry_fallback_no_GT": True, "available": sum(x["initial_pose"]["available"] for x in initial)})
    write_csv(out / "PER_GROUP_FOUR_CONFIGURATIONS.csv", per_frame)
    write_csv(out / "CERTIFIED_COVERAGE880.csv", coverage)
    write_csv(out / "NATIVE_COVERAGE7040.csv", point_rows)
    write_jsonl(out / "ALL_FOUR_INPUTS_AND_POSES.jsonl", pose_rows)
    summaries, paired = [], []
    for pol in sorted({r["fallback_policy"] for r in per_frame}):
        for role in ("ALL110", "FIT86", "DEV24"):
            rr = [r for r in per_frame if r["fallback_policy"] == pol and (role == "ALL110" or r["fit"] == (role == "FIT86"))]
            for arm in configs:
                aa = [r for r in rr if r["arm"] == arm]
                summaries.append({"scope": role, "fallback_policy": pol, "arm": arm, **summarize(aa)})
                if arm != "ORIGINAL":
                    baseline = {r["group_id"]: r for r in rr if r["arm"] == "ORIGINAL"}
                    paired.append({"scope": role, "fallback_policy": pol, "arm": arm, "n": len(aa),
                        "post_wrong_to_correct": sum(not baseline[r["group_id"]]["success_post5cm5deg"] and r["success_post5cm5deg"] for r in aa),
                        "post_correct_to_wrong": sum(baseline[r["group_id"]]["success_post5cm5deg"] and not r["success_post5cm5deg"] for r in aa),
                        "canonical_wrong_to_correct": sum(not baseline[r["group_id"]]["success_canonical5cm5deg"] and r["success_canonical5cm5deg"] for r in aa),
                        "canonical_correct_to_wrong": sum(baseline[r["group_id"]]["success_canonical5cm5deg"] and not r["success_canonical5cm5deg"] for r in aa)})
    write_csv(out / "SUMMARY.csv", summaries)
    write_csv(out / "PAIRED_TRANSITIONS.csv", paired)
    names = ("START.json", "INITIAL_T_ORIGINAL.jsonl", "INITIAL_T_COMPLETE.json", "PER_GROUP_FOUR_CONFIGURATIONS.csv",
             "CERTIFIED_COVERAGE880.csv", "NATIVE_COVERAGE7040.csv", "ALL_FOUR_INPUTS_AND_POSES.jsonl", "SUMMARY.csv", "PAIRED_TRANSITIONS.csv")
    complete = {"completed": True, "PASS": True, "groups": 880, "native_rows": 7040, "base_count": 110,
        "FIT_bases": 86, "DEV_bases": 24, "fixed_solver_configurations": 4, "raw_pose_calls": 3520,
        "partial_certified_oracle": True, "unknown_visibility_keeps_original_support": True,
        "local_location_geometric_not_photometric_certified": True, "symmetry_order": "UNKNOWN_NO_SYMMETRY_MIN",
        "source_basis_bool": False, "GT_frame_fixed_not_predicted_WD": True,
        "metric_axes": ["post_fixed_Qtruth", "declared_physical_canonical_C1", "declared_CF"],
        "original_centroid9_present": False, "centroid9_placeholder": [0, 0], "centroid9_valid": False,
        "no_original_pose_substitution_with_GT": True, "new_training_or_CAL": 0,
        "raw_RGB_depth_masks_mesh_or_private_current_held": 0,
        "packet": entry["groups"], "partial_policy": entry["partial_oracle_policy"], "code": entry["code"],
        "INITIAL_T": initial_binding, "outputs": {n: bind(out / n) for n in names}, "seconds": time.monotonic() - start}
    write_json(out / "COMPLETE.json", complete)
    print(json.dumps({"PASS": True, "groups": 880, "seconds": complete["seconds"], "INITIAL_T": initial_binding}), flush=True)
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", required=True)
    args = parser.parse_args()
    run(json.loads(guard(args.entry).read_text()))
