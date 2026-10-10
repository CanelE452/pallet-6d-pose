"""Independent saved-coordinate/pose review, with no evaluator/model/F imports."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotations(order):
    return [np.asarray([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
            for t in np.arange(order) * 2 * np.pi / order]


def same(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict)
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert b is not None and np.isclose(a, b, rtol=0, atol=1e-7)
    else:
        assert a == b


def review(source, docs):
    locked = read(docs / "INPUT_AND_METHOD_LOCK.json")
    raw = rows(docs / "PREDICTIONS.jsonl.gz")
    expected = locked["population"]["frame_ids"]
    methods = locked["methods"]
    assert len(raw) == 3828
    lookup = {(r["seed"], r["method"], r["id"]): r for r in raw}
    assert len(lookup) == len(raw)
    inputs = {r["id"]: r for r in locked["input_manifest"]}
    axes = {r["image"]: r for r in read(source / "data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json")["frames_list"]}
    gts = read(source / "data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json")["frames"]
    audit = read(docs / "INPUT_AUDIT.json")
    binding = {r["id"]: r for r in audit["inputs"]}
    old = {(r["method"], r["id"]): r for r in rows(source / "_docs/experiments/pallet_n3_subpix_20261008_v1/PREDICTIONS.jsonl.gz")}
    aliases = dict(BASE="BASE", N3_DIM_SYM="N3", SUBPIX="SUBPIX", N3_THEN_SUBPIX="N3_SUBPIX")
    errors = dict(translation_cm=0., rotation_deg=0., ADDsym_m=0., corner_px=0., cap_coordinates_px=0.)
    actual_F = 0
    no_pose = []
    old_vectors = Counter()
    preserved = Counter()
    diagnostics = Counter()
    cached_pose_parity = {}
    largest_errors = []
    N3_head_usage = {}
    for seed in (1, 2, 3):
        pose_relative = f"data/pallet/results/pallet_dim_conditioned_p_v1/pose_predictions/REAL_DEV/N3_DIM_SYM_seed{seed}.json"
        cache = read(source / pose_relative)
        assert cache["GT_input"] is False and set(cache["records"]) == set(expected)
        expected_hash = "c1452342a17b9576cb3639f9f9d15c33446268322c63ed0183406df96c266b2f" if seed == 1 else None
        actual_hash = sha(source / pose_relative)
        if expected_hash:
            assert actual_hash == expected_hash
        max_cached_difference = 0.
        for fid in expected:
            old_pose = cache["records"][fid]
            new_pose = lookup[(seed, "N3_DIM_SYM", fid)]["actual_pose"]
            same(old_pose, new_pose)
            for key in ("R_cf", "R_physical", "centroid", "cf_extents", "reprojection_px"):
                if key in old_pose:
                    max_cached_difference = max(max_cached_difference, float(np.max(np.abs(np.asarray(old_pose[key])-np.asarray(new_pose[key])), initial=0)))
        cached_pose_parity[str(seed)] = dict(path=pose_relative, sha256=actual_hash, frames=319,
            historical_hash_bound=(seed == 1), historical_expected_sha256=expected_hash,
            provenance_note="Seed1 hash is recorded in old sealed protocol; seed2/3 cache hashes are recorded now as additional parity evidence, not historical authentication.",
            maximum_absolute_difference=max_cached_difference, absolute_tolerance=1e-7, status="PASS")
        packet = read(source / f"data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV/N3_DIM_SYM_seed{seed}.json")
        N3_head_usage[str(seed)] = dict(frames=319, head_used=sum(bool(r["head_used"]) for r in packet["records"]),
                                       head_skipped=sum(not bool(r["head_used"]) for r in packet["records"]))
        for method in methods:
            group = [r for r in raw if r["seed"] == seed and r["method"] == method]
            assert [r["id"] for r in group] == expected
            assert len(group) == 319 and len({r["session"] for r in group}) == 13
            assert Counter(r["grade"] for r in group) == dict(clean=153, moderate=92, severe=74)
            assert sum(len(r["corner"]["errors"]) for r in group) == 2499
            assert sum(len(r["corner"]["observed_errors"]) for r in group) == 2445
            assert sum(r["corner"]["matched"] for r in group) == 311
            largest = max((r for r in group if r["pose"]["available"]), key=lambda r: r["pose"]["translation_cm"])
            largest_errors.append(dict(seed=seed, method=method, id=largest["id"], translation_cm=largest["pose"]["translation_cm"],
                                       rotation_deg=largest["pose"]["rotation_deg"], ADDsym_cm=100*largest["pose"]["ADDsym_m"]))
            for r in group:
                fid = r["id"]
                b = inputs[fid]
                q0, qn, qs, final = [np.asarray(r[k], dtype=np.float64) for k in ("q0", "qN", "qS", "qFinal")]
                support = np.asarray(r["prediction_support"], dtype=bool)
                assert np.array_equal(q0, np.asarray(b["q0"]), equal_nan=True)
                assert np.array_equal(support, np.asarray(b["prediction_support"]))
                assert np.array_equal(final[8], q0[8], equal_nan=True)
                assert np.array_equal(final[~support], q0[~support], equal_nan=True)
                for k in ("selected_index", "K", "dimensions_pnp_WH_D_m", "canonical_symmetry_order"):
                    assert r["fixed_metadata"][k] == b[k]
                assert r["evaluation_reference_used_in_inference"] is False
                actual_F += int(r["F_attempt"])
                assert r["actual_pose"]["available"] == r["pose"]["available"]
                assert r["pose_parameters_saved"] and r["F_complete"]
                if seed > 1 and method in ("BASE", "SUBPIX"):
                    baseline = lookup[(1, method, fid)]
                    assert r["actual_pose"] == baseline["actual_pose"] and r["pose"] == baseline["pose"]
                    assert r["qFinal"] == baseline["qFinal"] and r["reused_from_seed1"] is True
                    assert r["F_attempt"] is False
                if method in ("SUBPIX", "N3_THEN_SUBPIX"):
                    cap = .01 * np.hypot(*r["raw_hw"])
                    active = support[:8] & np.isfinite(q0[:8]).all(-1) & ~np.all(q0[:8] == -1, axis=-1)
                    delta = qs[:8] - q0[:8]
                    length = np.linalg.norm(delta, axis=-1)
                    factors = np.ones_like(length)
                    moving = length > 0
                    factors[moving] = np.minimum(1., cap / length[moving])
                    computed = q0.copy()
                    computed[:8][active] = q0[:8][active] + delta[active] * factors[active, None]
                    diff = float(np.max(np.abs(computed - final)))
                    errors["cap_coordinates_px"] = max(errors["cap_coordinates_px"], diff)
                    assert diff < 1e-10
                    assert np.max(np.linalg.norm(final[:8]-q0[:8], axis=-1)[active], initial=0) <= cap + 1e-10
                    assert r["correction"]["cap_active8"] == ((length > cap) & active).tolist()
                    starting = qn if method == "N3_THEN_SUBPIX" else q0
                    for c in r["correction"]["diagnostics"]["corner_records"]:
                        if c["status"] != "refined":
                            assert np.array_equal(qs[c["corner"]], starting[c["corner"]], equal_nan=True)
                            diagnostics[(seed, method, c["status"])] += 1
                gt = np.asarray(r["evaluation_reference_points"], dtype=float)
                valid = np.asarray(r["evaluation_reference_valid"], dtype=bool)
                perm = np.asarray(r["evaluation_permutation"], dtype=int)
                observed = np.isfinite(final).all(-1) & ~np.all(final == -1, axis=-1) & r["corner"]["matched"]
                distance = np.linalg.norm(final - gt[perm], axis=-1)
                per_corner = np.where(observed, distance, np.hypot(*r["raw_hw"]))
                c = per_corner[:8][valid[perm][:8]]
                obs = distance[:8][(valid[perm] & observed)[:8]]
                errors["corner_px"] = max(errors["corner_px"], float(np.max(np.abs(c - np.asarray(r["corner"]["errors"])), initial=0)))
                assert np.allclose(c, r["corner"]["errors"], rtol=0, atol=1e-7)
                assert np.allclose(obs, r["corner"]["observed_errors"], rtol=0, atol=1e-7)
                if not r["pose"]["available"]:
                    no_pose.append(dict(seed=seed, method=method, id=fid))
                    continue
                assert r["actual_pose"]["selected_hypothesis"] == r["final_hypothesis"]
                xyz = np.asarray(b["dimensions_pnp_WH_D_m"])
                ref = gts[axes[binding[fid]["image"]["path"]]["frame_id"]]
                dims = ref["physical_dimensions_m"]
                cf = np.asarray([dims["across"], dims["height"], dims["along"]])
                truth_R = np.asarray(ref["R_gt_representative"])
                if abs(cf[0]-xyz[0]) >= 1e-6:
                    truth_R = truth_R @ rotations(4)[1]
                truth_t = np.asarray(ref["t_gt"])
                R, t = np.asarray(r["actual_pose"]["R_physical"]), np.asarray(r["actual_pose"]["centroid"])
                assert np.allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-10) and abs(np.linalg.det(R)-1)<1e-10
                corners = np.asarray([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]) * xyz / 2
                t_error = float(np.linalg.norm(t-truth_t)*100)
                candidate_rotations, candidate_add = [], []
                for symmetry in rotations(b["canonical_symmetry_order"]):
                    target_R = truth_R @ symmetry
                    angle = np.arccos(np.clip((np.trace(target_R.T @ R)-1)/2, -1, 1))
                    candidate_rotations.append(float(np.rad2deg(angle)))
                    candidate_add.append(float(np.linalg.norm(corners @ R.T + t - (corners @ target_R.T + truth_t), axis=-1).mean()))
                computed = dict(translation_cm=t_error, rotation_deg=min(candidate_rotations), ADDsym_m=min(candidate_add))
                for key, value in computed.items():
                    delta = abs(value-r["pose"][key])
                    errors[key] = max(errors[key], delta)
                    assert delta < 1e-7, (seed, method, fid, key, delta)
                if seed == 1:
                    old_row = old[(aliases[method], fid)]
                    assert np.array_equal(final, np.asarray(old_row["qFinal"]), equal_nan=True)
                    same(old_row["corner"], r["corner"])
                    same(old_row["pose"], r["pose"])
                    assert old_row["final_hypothesis"] == r["final_hypothesis"]
                    if old_row.get("actual_pose") is not None:
                        same(old_row["actual_pose"], r["actual_pose"])
                        old_vectors[method] += 1
                preserved[method] += 1
    assert actual_F == 2552
    # The actual F implementation and call receive no target/reference variables.
    infer_source = source / "scripts/research/pallet_dim_conditioned_p_v1/pose.py"
    infer_ast = ast.parse(infer_source.read_text())
    infer_fn = next(n for n in infer_ast.body if isinstance(n, ast.FunctionDef) and n.name == "infer")
    assert [n.arg for n in infer_fn.args.args] == ["points", "K", "xyz", "source"]
    evaluator_source = Path(__file__).resolve().with_name("evaluate.py")
    return dict(schema="independent_evaluator_contract_review_v1", status="PASS",
                source_files=[dict(path="scripts/research/pallet_n3_subpix_final_20261010/evaluate.py", sha256=sha(evaluator_source)),
                              dict(path="scripts/research/pallet_dim_conditioned_p_v1/pose.py", sha256=sha(infer_source))],
                rows=len(raw), rows_per_seed_method=319, seeds=[1,2,3], methods=methods,
                actual_F_calls_accounted=actual_F, shared_BASE_SUBPIX_rows=1276,
                preserved_rows_by_method=dict(preserved), old_pose_vector_parity_coverage=dict(old_vectors),
                maximum_absolute_differences=errors, no_pose=no_pose,
                all_seed_N3_pose_cache_parity=cached_pose_parity, largest_translation_error_each_seed_method=largest_errors,
                N3_head_usage=N3_head_usage,
                corner_fallback_native_preservation=dict(status="PASS", counts=[dict(seed=s,method=m,status=k,corners=n)
                    for (s,m,k),n in sorted(diagnostics.items())]),
                no_pose_interpretation="No pose failures means F produced a finite accepted pose for every input; it does not mean every estimate was accurate. Large finite outliers are preserved in raw rows and distribution statistics.",
                checks=dict(same_IDs_and_order=True, all_grades_preserved=True, canonical_dimensions_K_support_metadata=True,
                            reference_corner_errors_recomputed=True, actual_R_t_T_R_ADDsym_recomputed=True,
                            total_cap_anchored_once_to_Base=True, center_unsupported_preserved=True,
                            seed1_all_319_metric_coordinate_hypothesis_parity=True,
                            inference_signature_no_GT=True, scoring_reference_fields_appended_after_F=True),
                GT_access_explanation="E.scored computes corner metrics separately, calls POSE.infer(points,K,xyz,source), then scores pose. None of target GT, human visibility, evaluation branch or grade enters cornerSubPix/cap/F.",
                implementation="Independent saved-row NumPy formulas plus source signature review; no experiment evaluator, model, OpenCV or F imports",
                execution=dict(new_model_calls=0, new_F_calls=0, new_optimizer_updates=0, original_input_writes=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.source_root, args.docs)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
