"""Frozen saved-current utility: existing V58 post, references after pose seal."""
import argparse
import csv
import gzip
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
    if not p.is_absolute() or ".." in p.parts or any(x in str(p).lower() for x in ("v112", "cache", ".pyc")):
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
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def binding(path):
    data = guard(path).read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def json_from(data, path):
    return json.loads(gzip.decompress(data) if path.endswith(".gz") else data)


def has_binding(obj, candidate):
    if isinstance(obj, dict):
        return obj == candidate or any(has_binding(x, candidate) for x in obj.values())
    if isinstance(obj, list):
        return any(has_binding(x, candidate) for x in obj)
    return False


def run(entry):
    if entry.get("execution_authorized") is not True or entry.get("scope") != "FROZEN_CURRENT223_FIXED_POST_UTILITY_ONLY":
        raise ValueError("Root true entry required")
    assert checked(entry["code"]) == Path(__file__).read_bytes()
    approval = json.loads(checked(entry["root_scope"]))
    assert approval["approved"] is True
    assert approval["all4_checkpoints_and_logs_frozen"] is True
    assert approval["primary_current_predictions_frozen"] is True
    assert approval["allow_exact_late_reference_bindings_after_pose_seal"] is True
    assert approval["entry_contract"] == {k: v for k, v in entry.items() if k != "root_scope"}
    freeze = json.loads(checked(entry["all4_freeze"]))
    assert freeze["completed"] and freeze["PASS"] and freeze["all_learned_settings"] == 4
    assert freeze["optimizer_updates_each"] == 1000 and freeze["current_inputs_or_GT_used_for_selection"] == 0
    assert freeze["no_further_recipe_or_threshold_change_after_current"] is True
    models = entry["models"]
    assert ["ORIGINAL"] + [m["name"] for m in models] == freeze["current_primary_configurations"]
    assert len(models) == 3 and {m["mode"] for m in models} == {"LOC_ONLY", "VIS_ONLY", "JOINT"}
    pred_complete = json.loads(checked(entry["current_predictions_complete"]))
    predictions = {}
    for m in models:
        assert has_binding(pred_complete, m["predictions"]), "saved model predictions not closed by C COMPLETE"
        rows = [json.loads(x) for x in checked(m["predictions"]).splitlines()]
        assert len(rows) == 1784 and [r["query_index"] for r in rows] == list(range(1784))
        predictions[m["name"]] = rows
    # Only input metadata is decoded in this phase; original prepared data has
    # no case reference R/t. Its fixed cohort/symmetry metadata is not fitted.
    geometry_doc = json.loads(checked(entry["current_geometry"]))
    prepared_binding = entry["original_prepared_inputs"]
    assert geometry_doc["source"] == prepared_binding
    prepared = json_from(checked(prepared_binding), prepared_binding["path"])
    geom = geometry_doc["rows"]
    assert len(prepared) == len(geom) == 223
    assert [u["id"] for u in prepared] == [u["id"] for u in geom]
    original = load("v263_fixed_post_api_current", entry["baseline_api"])
    core = load("v263_current_input_core", entry["input_core"])
    raw = {k: checked(v) for k, v in entry["backend_inputs"].items()}
    backend, ev = original.load_backend(entry["backend_inputs"], raw)
    out = guard(entry["output_dir"])
    assert not out.exists(), "fresh output required"
    out.mkdir()
    start = time.monotonic()
    cv2.setNumThreads(1)
    pose_rows, parity, support_rows = [], [], []
    for fi, (u, g) in enumerate(zip(prepared, geom)):
        q9, K, xyz = np.asarray(u["initial_points"], float), np.asarray(u["K"], float), np.asarray(u["known_physical_dimensions_xyz_m"], float)
        assert np.array_equal(q9, np.asarray(g["original_initial_points"], float))
        assert np.array_equal(K, np.asarray(g["K"])) and np.array_equal(xyz, np.asarray(g["registered_dimensions_xyz_m"]))
        assert np.array_equal(np.asarray(u["native_xyz"]), np.asarray(g["native_xyz"]))
        orig_inp = core.make_post_inputs(q9, None, "ORIGINAL", fi, u["id"])
        n0 = backend.infer_masked(orig_inp["points9"], K, xyz, orig_inp["valid9"], source=False)
        gap, same = original.pose_gap(n0, u["initial_pose"])
        assert same and bool(n0["available"]) == bool(g["initial_pose_available"]), "unchanged all9 baseline mismatch"
        if n0["available"]:
            T = np.asarray(g["initial_T_physical"], float)
            assert np.max(np.abs(T[:3, :3] - np.asarray(n0["R_physical"]))) <= 1e-9
            assert np.max(np.abs(T[:3, 3] - np.asarray(n0["centroid"]))) <= 1e-9
            assert np.allclose(np.asarray(u["native_xyz"]), backend.base["cuboid"](*n0["cf_extents"]), rtol=0, atol=1e-12)
        parity.append({"frame_index": fi, "id": u["id"], "PASS": same, "max_pose_gap": gap,
                       "all9_original_solver_support": True, "source_basis_bool": False})
        arms = [("ORIGINAL", orig_inp)]
        for m in models:
            rows8 = predictions[m["name"]][8 * fi:8 * fi + 8]
            arms.append((m["name"], core.make_post_inputs(q9, rows8, m["mode"], fi, u["id"])))
        for name, inp in arms:
            raw_pose = n0 if name == "ORIGINAL" else backend.infer_masked(inp["points9"], K, xyz, inp["valid9"], source=False)
            fallback = not raw_pose["available"] and name != "ORIGINAL"
            final_pose = u["initial_pose"] if fallback else raw_pose
            pose_rows.append({"frame_index": fi, "id": u["id"], "group": u["group"], "partition": u["partition"],
                              "arm": name, "raw_pose": raw_pose, "final_pose": final_pose, "fallback": fallback,
                              "points9": inp["points9"].tolist(), "valid9": inp["valid9"].tolist(),
                              "original_centroid9": q9[8].tolist(), "centroid_preserved": True})
            support_rows.append({"frame_index": fi, "id": u["id"], "arm": name,
                                 "solver_native_support": int(inp["valid9"][:8].sum()),
                                 "localized_indices": json.dumps(inp["localized_native_indices"]),
                                 "excluded_indices": json.dumps(inp["excluded_native_indices"]),
                                 "RGB_available8": json.dumps(inp["RGB_classification_available8"]), "fallback": fallback})
    assert len(pose_rows) == 223 * 4
    write_jsonl(out / "ALL223_FIXED_POST_POSES.jsonl", pose_rows)
    write_csv(out / "ORIGINAL223_PARITY.csv", parity)
    write_csv(out / "SOLVER_SUPPORT_VS_RGB_AVAILABILITY.csv", support_rows)
    write_json(out / "POSES_FROZEN_BEFORE_REFERENCE_JOIN.json", {
        "completed": True, "PASS": True, "all4_head_freeze": entry["all4_freeze"],
        "current_prediction_complete": entry["current_predictions_complete"], "frames": 223, "pose_configurations": 4,
        "reference_case_R_t_reads_before_seal": 0, "poses": binding(out / "ALL223_FIXED_POST_POSES.jsonl"),
        "original_parity": binding(out / "ORIGINAL223_PARITY.csv"), "new_model_or_CAL_or_fit": 0})
    # Exact old V60/V50 reference joins only, after ALL new fixed-post poses.
    wb, sb = entry["references"]["wood"], entry["references"]["square_bundle"]
    wood = json_from(checked(wb), wb["path"])["references"]
    square_doc = json_from(checked(sb), sb["path"])
    square = {x["id"]: x["reference"] for x in square_doc["frames"] if x["group"] == "SQUARE_YOLO"}
    refs = {}
    for u in prepared:
        if u["group"].startswith("WOOD"):
            g = wood[u["partition"]][u["id"]]
            assert u["symmetry_order"] == 2
        else:
            g = square[u["id"].split(":")[-1]]
            assert u["symmetry_order"] == 4
        refs[u["id"]] = {"R": g["R"], "t": g["t"], "order": u["symmetry_order"]}
    results = []
    for p in pose_rows:
        metric = ev.metric(p["final_pose"], refs[p["id"]])
        results.append({"frame_index": p["frame_index"], "id": p["id"], "group": p["group"], "arm": p["arm"],
                        "fallback": p["fallback"], "excluded": 8 - sum(p["valid9"][:8]),
                        "reference_kind": "HISTORICAL_RECONSTRUCTED_GEOMETRIC_REFERENCE_NOT_INDEPENDENT_PHYSICAL_GT",
                        "symmetry_order": refs[p["id"]]["order"], **metric})
    summaries, paired = [], []
    for group in ["ALL223"] + sorted({p["group"] for p in results}):
        rr = [r for r in results if group == "ALL223" or r["group"] == group]
        baseline = {r["id"]: r for r in rr if r["arm"] == "ORIGINAL"}
        for arm in ["ORIGINAL"] + [m["name"] for m in models]:
            ar = [r for r in rr if r["arm"] == arm]
            summaries.append({"group": group, "arm": arm, **ev.summarize(ar)})
            if arm != "ORIGINAL":
                paired.append({"group": group, "arm": arm, "n": len(ar),
                    "wrong_to_correct": sum(not baseline[r["id"]]["success"] and r["success"] for r in ar),
                    "correct_to_wrong": sum(baseline[r["id"]]["success"] and not r["success"] for r in ar),
                    "T_gt50_entry": sum(not baseline[r["id"]].get("T_gt50", False) and r.get("T_gt50", False) for r in ar),
                    "R_gt45_entry": sum(not baseline[r["id"]].get("R_gt45", False) and r.get("R_gt45", False) for r in ar)})
    write_csv(out / "REFERENCE_RELATIVE_PER_FRAME.csv", results)
    write_csv(out / "REFERENCE_RELATIVE_SUMMARY.csv", summaries)
    write_csv(out / "REFERENCE_RELATIVE_PAIRED_TRANSITIONS.csv", paired)
    names = ("ALL223_FIXED_POST_POSES.jsonl", "ORIGINAL223_PARITY.csv", "SOLVER_SUPPORT_VS_RGB_AVAILABILITY.csv",
             "POSES_FROZEN_BEFORE_REFERENCE_JOIN.json", "REFERENCE_RELATIVE_PER_FRAME.csv", "REFERENCE_RELATIVE_SUMMARY.csv", "REFERENCE_RELATIVE_PAIRED_TRANSITIONS.csv")
    complete = {"completed": True, "PASS": True, "frames": 223, "original_N0_parity": 223,
        "max_original_pose_gap": max(p["max_pose_gap"] for p in parity), "pose_configurations": 4,
        "pose_eval_is_reference_relative": True, "symmetry_contract": "old WOOD C2 / SQUARE C4 unchanged; canonical separately reported",
        "success_rule": "same V60 strict T<5cm and symmetry-min R<5deg", "reference_pose_reads_after_all_pose_freeze": True,
        "source_basis_bool": False, "old_all9_finite_solver_support_preserved": True,
        "centroid9_original_value_and_true_preserved": True, "new_raw_RGB_or_model_forward_or_fitting_or_CAL": 0,
        "entry_code": entry["code"], "all4_freeze": entry["all4_freeze"], "prediction_complete": entry["current_predictions_complete"],
        "references": entry["references"], "outputs": {n: binding(out / n) for n in names}, "seconds": time.monotonic() - start}
    write_json(out / "COMPLETE.json", complete)
    print(json.dumps({"PASS": True, "frames": 223, "original_N0_parity": 223, "seconds": complete["seconds"]}), flush=True)
    return complete


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    args = ap.parse_args()
    run(json.loads(guard(args.entry).read_text()))
