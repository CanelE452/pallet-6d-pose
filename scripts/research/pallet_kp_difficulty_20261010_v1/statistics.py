"""Post-seal statistics and reference annotations for the eight frozen arms.

Run as ``python -m scripts.research.pallet_kp_difficulty_20261010_v1.statistics``.
No detector, head, optimizer, PnP, training or renderer is called.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

from .geometry_difficulty import (NEW, OLD, corners, dimension, image_shape,
                                 protected_check, read_rows, sha)

METRICS = {"translation_cm": ("translation_cm", 1., "cm"),
           "rotation_deg": ("rotation_deg", 1., "degree"),
           "ADDsym_cm": ("ADDsym_m", 100., "cm")}
COMPARATORS = ("BASE", "N3_SUBPIX", "BASE_NO_MASK_ROBUST", "BASE_GEOM_NOSELF_ROBUST",
               "N3_SUBPIX_NO_MASK_ROBUST", "N3_SUBPIX_GEOM_NOSELF_ROBUST",
               "GEOMETRY_ONLY", "IMAGE_NO_ROLE", "IMAGE_ROLE")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, data):
    assert not path.exists(), "Preserve completed follow-up outputs"
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def save_rows(path, rows):
    assert not path.exists(), "Preserve completed follow-up outputs"
    with path.open("wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as zipped:
            for row in rows:
                zipped.write((json.dumps(clean(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n").encode())


def bind(path, root):
    return {"path": str(path.relative_to(root)) if path.is_relative_to(root) else path.name,
            "sha256": sha(path), "bytes": path.stat().st_size}


def distribution(values, unit):
    a = np.asarray(values, dtype=np.float64)
    assert np.isfinite(a).all()
    return {"n": len(a), "mean": float(a.mean()) if len(a) else None,
            "sample_variance": float(a.var(ddof=1)) if len(a)>1 else None,
            "sample_std": float(a.std(ddof=1)) if len(a)>1 else None,
            "median": float(np.median(a)) if len(a) else None,
            "P90": float(np.quantile(a, .9)) if len(a) else None,
            "max": float(a.max()) if len(a) else None, "unit": unit, "ddof": 1}


def summary(rows):
    available = [r for r in rows if r["pose"]["available"]]
    fresh = [r for r in available if r.get("new_pose_estimated")]
    assert all(bool(r.get("no_pose")) != bool(r["pose"]["available"]) for r in rows)
    metrics = {scope: {m: distribution([r["pose"][field]*factor for r in rr], unit)
                       for m, (field, factor, unit) in METRICS.items()}
               for scope, rr in (("operational", available), ("new_pose", fresh))}
    return {"total_frames": len(rows), "pose_available": len(available),
            "new_pose_estimated": len(fresh),
            "fallback_used": sum(bool(r.get("fallback_used")) for r in rows),
            "no_pose": len(rows)-len(available),
            "hidden_reprojected": sum(bool(r.get("hidden_reprojected")) for r in rows),
            "available_ids": [r["id"] for r in available],
            "new_pose_ids": [r["id"] for r in fresh],
            "fallback_ids": [r["id"] for r in rows if r.get("fallback_used")],
            "no_pose_ids": [r["id"] for r in rows if not r["pose"]["available"]],
            "output_status_counts": dict(Counter(r["output_status"] for r in rows)),
            "failure_reasons": dict(Counter(r.get("solver", {}).get("reason", "historical_fixed")
                                            for r in rows if not r.get("new_pose_estimated"))),
            "metrics": metrics,
            "denominator_note": "All rows are retained. Operational error moments use available outputs, including fallback; complete failure count/IDs remain separate. No fabricated zero or finite failure penalty."}


def paired(a, b, ids, inverse, draws, scope):
    eligible = [i for i, fid in enumerate(ids) if a[fid]["pose"]["available"] and b[fid]["pose"]["available"]
                and (scope=="common_operational" or a[fid].get("new_pose_estimated"))
                and (scope!="both_new_pose" or b[fid].get("new_pose_estimated"))]
    counts = np.bincount(inverse[eligible], minlength=13)
    denominator = draws @ counts
    keep = denominator > 0
    results = {"denominator": 319, "common_frames": len(eligible),
               "common_ids": [ids[i] for i in eligible],
               "excluded_ids": [fid for i, fid in enumerate(ids) if i not in set(eligible)],
               "scope": scope, "bootstrap_nonempty_resamples": int(keep.sum()), "metrics": {}}
    for metric, (field, factor, unit) in METRICS.items():
        delta = np.asarray([(a[ids[i]]["pose"][field]-b[ids[i]]["pose"][field])*factor for i in eligible])
        totals = np.bincount(inverse[eligible], weights=delta, minlength=13)
        samples = (draws @ totals)[keep]/denominator[keep]
        results["metrics"][metric] = {**distribution(delta, unit),
            "CI95": np.quantile(samples, [.025, .975]).tolist() if len(samples) else None,
            "improved_frames": int((delta < -1e-9).sum()),
            "worsened_frames": int((delta > 1e-9).sum()),
            "unchanged_frames": int((np.abs(delta) <= 1e-9).sum()),
            "interpretation": "new minus comparator; CI of paired mean by repeated session draws, not error SD"}
    results["new_marginals"] = summary([a[ids[i]] for i in eligible])["metrics"]["operational"]
    results["comparator_marginals"] = summary([b[ids[i]] for i in eligible])["metrics"]["operational"]
    return results


def delta_outcome(t, r, fresh, available):
    if not available:
        return "no_pose"
    if not fresh:
        return "fallback"
    if t < -1e-9 and r < -1e-9:
        return "both_improved"
    if t > 1e-9 and r > 1e-9:
        return "both_worsened"
    return "mixed_or_equal"


def reference_annotations(new, controls, targets, old_audit):
    records = []
    for row in new:
        arm = "N3_SUBPIX" if row["method"].startswith("N3_SUBPIX") else "BASE"
        baseline = controls[arm][row["id"]]
        target = targets[row["id"]]
        inherited = old_audit[(arm+"_NO_MASK_ROBUST", row["id"])]
        phase = np.asarray(target["permutations"][baseline["corner"].get("branch", 0)][:8], int)
        assert phase.tolist() == inherited["permutation_native_to_canonical"]
        states = inherited["human_states_native"]
        gt = np.asarray(target["gt"], float)[phase]
        valid = np.asarray(target["valid"], bool)[phase] & np.isfinite(gt).all(1) & ~(gt==-1).all(1) & bool(target["matched"])
        def errors(points):
            q = np.asarray(points, float)[:8]
            known = valid & np.isfinite(q).all(1) & ~(q==-1).all(1)
            e = np.linalg.norm(q-gt, axis=1)
            e[~known] = np.nan
            return e, known
        e, known = errors(row["input_points"])
        before, before_known = errors(baseline["native_points"])
        after, after_known = errors(row["native_points"])
        correct = set(np.flatnonzero(known & (e<=8.)).tolist())
        known_ids = set(np.flatnonzero(known).tolist())
        pool = set(row["solver"]["used"])
        inliers = set(row["solver"].get("final_inliers", row["solver"].get("inliers", [])))
        hidden = set(row["hidden_initial"])
        human_hidden = {i for i, s in enumerate(states) if s=="SELF_OCCLUDED"}
        direct = {i for i, s in enumerate(states) if s=="DIRECT_VISIBLE"}
        label_known = {i for i, s in enumerate(states) if s!="UNANNOTATED"}
        wrong_mask = bool((hidden ^ human_hidden) & label_known)
        xyz = (row.get("actual_pose") or {}).get("cf_extents") or row["xyz"]
        layout = np.asarray(corners(xyz), float)[sorted(pool & correct)]
        singular = np.linalg.svd(layout-layout.mean(0), compute_uv=False) if len(layout) else np.array([])
        current, prior = row["pose"], baseline["pose"]
        t = current["translation_cm"]-prior["translation_cm"] if current["available"] and prior["available"] else None
        r = current["rotation_deg"]-prior["rotation_deg"] if current["available"] and prior["available"] else None
        item = {"id": row["id"], "session": row["session"], "method": row["method"],
            "reference_kind": "GEOMETRIC_PROXY", "GT_used_after_geometry_seal": True,
            "frozen_baseline_arm": arm, "permutation_native_to_canonical": phase.tolist(),
            "human_states_native": states, "reference_matched": bool(target["matched"]),
            "accuracy_threshold_px": 8., "reference_error_input_native_px": e,
            "reference_error_initial_native_px": before, "reference_error_output_native_px": after,
            "reference_valid_input_ids": sorted(known_ids),
            "paired_corner_valid_ids": np.flatnonzero(before_known & after_known).tolist(),
            "correct_input_native_ids": sorted(correct), "pool_ids": sorted(pool),
            "correct_pool_ids": sorted(correct & pool), "correct_pool_count": len(correct & pool),
            "wrong_pool_ids": sorted((pool & known_ids)-correct), "unknown_pool_ids": sorted(pool-known_ids),
            "final_inlier_ids": sorted(inliers), "correct_final_inlier_ids": sorted(correct & inliers),
            "correct_final_inlier_count": len(correct & inliers),
            "wrong_final_inlier_ids": sorted((inliers & known_ids)-correct),
            "unknown_final_inlier_ids": sorted(inliers-known_ids),
            "human_direct_pool_ids": sorted(direct & pool), "human_direct_correct_pool_ids": sorted(direct & pool & correct),
            "false_excluded_direct_ids": sorted(hidden & direct),
            "false_excluded_accurate_ids": sorted(hidden & correct),
            "false_retained_human_self_ids": sorted((pool & human_hidden)-hidden),
            "correct_pool_layout_m": layout, "correct_pool_shape_singular_values": singular,
            "correct_pool_shape_rank": int(np.sum(singular > singular[0]*1e-10)) if len(singular) and singular[0]>0 else 0,
            "correct_pool_image_layout": image_shape([row["input_points"][i] for i in sorted(pool & correct)]),
            "solver_local_jacobian": row["solver"].get("geometry", {}).get("jacobian"),
            "mask_applied": "NO_MASK" not in row["method"], "mask_wrong_on_known": wrong_mask,
            "initial_hidden_ids": sorted(hidden), "hidden_reprojected_ids": row["reprojected_ids"],
            "new_pose_estimated": row["new_pose_estimated"], "fallback_used": row["fallback_used"],
            "no_pose": row["no_pose"], "output_status": row["output_status"],
            "translation_cm": current.get("translation_cm"), "rotation_deg": current.get("rotation_deg"),
            "ADDsym_m": current.get("ADDsym_m"), "translation_delta_cm": t, "rotation_delta_deg": r,
            "paired_pose_outcome_vs_same_coordinate_initial": delta_outcome(t, r, row["new_pose_estimated"], current["available"]),
            "initial_dimension_index": dimension(row["initial_pose"].get("cf_extents"), row["xyz"]),
            "selected_dimension_index": dimension((row["actual_pose"] or {}).get("cf_extents"), row["xyz"])}
        assert set(item["correct_pool_ids"]) | set(item["wrong_pool_ids"]) | set(item["unknown_pool_ids"]) == pool
        records.append(item)
    return clean(records)


def corner_summary(records, only_new=False):
    pairs = defaultdict(list)
    frame_means = defaultdict(list)
    for row in records:
        if only_new and not row["new_pose_estimated"]:
            continue
        local = defaultdict(list)
        for i in row["paired_corner_valid_ids"]:
            categories = [row["human_states_native"][i]]
            if i in row["hidden_reprojected_ids"]:
                categories.append("ALGORITHM_REPROJECTED_IDS")
            if i in row["false_excluded_direct_ids"]:
                categories.append("DIRECT_VISIBLE_FALSE_EXCLUDED")
            a, b = row["reference_error_initial_native_px"][i], row["reference_error_output_native_px"][i]
            for category in categories:
                pairs[category].append((a, b))
                local[category].append((a, b))
        for category, values in local.items():
            frame_means[category].append(tuple(np.mean(values, axis=0)))
    result = {}
    for category, values in pairs.items():
        a, b = np.asarray(values).T
        fa, fb = np.asarray(frame_means[category]).T
        result[category] = {"corners": len(a), "frames": len(fa),
            "before": distribution(a, "px"), "after": distribution(b, "px"),
            "paired_delta": distribution(b-a, "px"),
            "before_frame_mean": distribution(fa, "px"), "after_frame_mean": distribution(fb, "px"),
            "improved": int((b<a-1e-9).sum()), "worsened": int((b>a+1e-9).sum()),
            "good5_to_bad10": int(((a<5)&(b>10)).sum()),
            "bad20_to_good10": int(((a>20)&(b<=10)).sum())}
    return result


def correspondence_summary(records):
    def subset_summary(rr):
        return {"frames": len(rr), "new_pose": sum(r["new_pose_estimated"] for r in rr),
            "fallback": sum(r["fallback_used"] for r in rr),
            "pose_outcomes": dict(Counter(r["paired_pose_outcome_vs_same_coordinate_initial"] for r in rr)),
            "correct_pool_count_histogram": dict(Counter(str(r["correct_pool_count"]) for r in rr)),
            "correct_final_inlier_count_histogram": dict(Counter(str(r["correct_final_inlier_count"]) for r in rr)),
            "translation_cm": distribution([r["translation_cm"] for r in rr if r["translation_cm"] is not None], "cm"),
            "rotation_deg": distribution([r["rotation_deg"] for r in rr if r["rotation_deg"] is not None], "degree")}
    ge4 = [r for r in records if r["correct_pool_count"]>=4]
    answer = {"all": subset_summary(records), "correct_pool_lt4": subset_summary([r for r in records if r["correct_pool_count"]<4]),
        "correct_pool_ge4": subset_summary(ge4),
        "correct_pool_ge4_object_rank_counts": dict(Counter(str(r["correct_pool_shape_rank"]) for r in ge4)),
        "pool_inlier_reference_partition_histogram": dict(Counter(
            f'pool{len(r["pool_ids"])}_correct{r["correct_pool_count"]}_inliers{len(r["final_inlier_ids"])}_correct_inliers{r["correct_final_inlier_count"]}_unknown{len(r["unknown_pool_ids"])}'
            for r in records)),
        "final_reference_inaccurate_inliers_total": sum(len(r["wrong_final_inlier_ids"]) for r in records),
        "final_unknown_inliers_total": sum(len(r["unknown_final_inlier_ids"]) for r in records),
        "false_excluded_accurate_total": sum(len(r["false_excluded_accurate_ids"]) for r in records),
        "unknown_pool_frames": sum(bool(r["unknown_pool_ids"]) for r in records)}
    if records[0]["mask_applied"]:
        answer["mask_known_relation"] = {
            "wrong_mask": subset_summary([r for r in records if r["mask_wrong_on_known"]]),
            "matching_known_human_self_states": subset_summary([r for r in records if not r["mask_wrong_on_known"]])}
    else:
        answer["mask_known_relation"] = {"applicable": False, "reason": "No mask applied; label mismatch is not a classifier error."}
    return answer


def select_cases(methods, annotations, ids):
    fixed = ["plastic_night_01:038630", "plastic_day_01:011497", "eval_pallet09:1778653806958839552"]
    details = []
    comparisons = [("IMAGE_ROLE_MATCH_MASS", "N3_SUBPIX"),
                   ("N3_SUBPIX_INITIAL_DIMENSION_PRIOR_GEOM_NOSELF", "N3_SUBPIX_GEOM_NOSELF_ROBUST"),
                   ("N3_SUBPIX_INITIAL_DIMENSION_PRIOR_GEOM_NOSELF", "N3_SUBPIX")]
    lookup = {(r["method"], r["id"]): r for r in annotations}
    for a, b in comparisons:
        eligible = [fid for fid in ids if methods[a][fid]["pose"]["available"] and methods[b][fid]["pose"]["available"]]
        order = sorted(eligible, key=lambda fid: (methods[a][fid]["pose"]["translation_cm"]-methods[b][fid]["pose"]["translation_cm"], fid))
        selections = [(fid, "fixed mechanism example") for fid in fixed]
        selections += [(fid, "three smallest translation deltas; ID tie-break") for fid in order[:3]]
        selections += [(fid, "three largest translation deltas; ID tie-break") for fid in order[-3:][::-1]]
        for fid, rule in selections:
            pa, pb = methods[a][fid]["pose"], methods[b][fid]["pose"]
            details.append({"id": fid, "method": a, "comparator": b, "selection": rule,
                "output_status": methods[a][fid]["output_status"],
                "translation_delta_cm": pa.get("translation_cm", 0)-pb.get("translation_cm", 0) if pa["available"] and pb["available"] else None,
                "rotation_delta_deg": pa.get("rotation_deg", 0)-pb.get("rotation_deg", 0) if pa["available"] and pb["available"] else None,
                "candidate_pose": pa, "comparator_pose": pb, "posthoc_correspondence": lookup[(a, fid)]})
    return {"selection_is_performance_illustration_only": True, "no_new_inference_or_pose": True,
            "fixed_cases": fixed, "cases": details}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--protected-state", type=Path)
    args = p.parse_args()
    began = time.monotonic()
    root = args.root.resolve()
    old = root/"_docs/experiments"/OLD
    doc = root/"_docs/experiments"/NEW
    output = args.output_dir or doc
    output.mkdir(parents=True, exist_ok=True)
    names = ("METRICS.json", "STATISTICS_RECEIPT.json", "POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz", "REVIEW_CASE_SELECTION.json")
    assert not any((output/n).exists() for n in names), "Preserve completed statistics"
    before = protected_check(root, args.protected_state)
    protocol = json.loads((doc/"PROTOCOL.json").read_text())
    execution = json.loads((doc/"CAUSAL_POSE_EXECUTION.json").read_text())
    sealed = doc/"CAUSAL_GEOMETRY_SEALED.jsonl.gz"
    scored = doc/"PREDICTIONS.jsonl.gz"
    assert execution["complete"] and execution["score_phase_after_geometry_seal"]
    assert sha(sealed)==execution["raw_geometry"]["sha256"] and sha(scored)==execution["scored"]["sha256"]
    new = read_rows(scored)
    inputs = json.loads((old/"INPUTS.json").read_text())["frames"]
    ids = [r["id"] for r in inputs]
    assert len(new)==2552 and len(ids)==len(set(ids))==319
    methods = defaultdict(dict)
    controls = read_rows(old/"FIXED_CONTROLS.jsonl.gz")
    originals = read_rows(old/"PREDICTIONS.jsonl.gz")+read_rows(old/"LEARNED_PREDICTIONS.jsonl.gz")
    for row in controls+originals+new:
        if row["method"] not in COMPARATORS and row["method"] not in protocol["new_methods"]:
            continue
        assert row["id"] not in methods[row["method"]]
        methods[row["method"]][row["id"]] = row
    assert len(methods)==17
    for method, data in methods.items():
        assert set(data)==set(ids), method
    units, inverse = np.unique([r["session"] for r in inputs], return_inverse=True)
    assert len(units)==13
    draws = np.random.default_rng(20260917).multinomial(13, np.full(13, 1/13), size=10000).astype(np.uint16)
    draw_sha = hashlib.sha256(draws.tobytes()).hexdigest()
    original_metrics = json.loads((old/"METRICS.json").read_text())
    assert draw_sha==original_metrics["bootstrap"]["draw_sha256"]
    summaries = {method: summary([data[fid] for fid in ids]) for method, data in methods.items()}
    for method in COMPARATORS:
        assert summaries[method]["metrics"]==original_metrics["methods"][method]["metrics"], method
    contrasts = [(a, b) for a in protocol["new_methods"] for b in COMPARATORS]
    contrasts += [("IMAGE_ROLE_MATCH_MASS_NO_MASK", "IMAGE_ROLE_MATCH_MASS"),
                  ("BASE_INITIAL_DIMENSION_PRIOR_GEOM_NOSELF", "BASE_INITIAL_DIMENSION_PRIOR_NO_MASK"),
                  ("N3_SUBPIX_INITIAL_DIMENSION_PRIOR_GEOM_NOSELF", "N3_SUBPIX_INITIAL_DIMENSION_PRIOR_NO_MASK")]
    paired_results = {a+"_minus_"+b: {scope: paired(methods[a], methods[b], ids, inverse, draws, scope)
                                    for scope in ("common_operational", "candidate_new_pose", "both_new_pose")}
                      for a, b in contrasts}
    # This is the first GT access in this script, after complete/hash-bound sealing.
    from ..pallet_observation_refiner_20261009_v1 import common as C
    C.source_modules()
    import cv2
    guarded = {name: getattr(cv2, name) for name in ("solvePnP", "solvePnPGeneric", "solvePnPRefineLM")}
    def forbidden(*a, **kw):
        raise AssertionError("Statistics must not solve a new pose")
    for name in guarded:
        setattr(cv2, name, forbidden)
    try:
        from scripts.research.pallet_training_free_compare_20261007_v1.common import load_real
        _, frames, targets, _, baseline_path = load_real()
        assert {r["id"] for r in frames}==set(ids)
        old_audit = {(r["method"], r["id"]): r for r in read_rows(old/"REAL_CORRESPONDENCE_ROWS.jsonl.gz")}
        annotations = reference_annotations(new, methods, targets, old_audit)
    finally:
        for name, fn in guarded.items():
            setattr(cv2, name, fn)
    ann_groups = defaultdict(list)
    for row in annotations:
        ann_groups[row["method"]].append(row)
    annotation_summary = {m: correspondence_summary(rr) for m, rr in ann_groups.items()}
    visible = {m: {"operational": corner_summary(rr), "new_pose": corner_summary(rr, True)} for m, rr in ann_groups.items()}
    selected = select_cases(methods, annotations, ids)
    save_rows(output/"POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz", annotations)
    write(output/"REVIEW_CASE_SELECTION.json", selected)
    primary = "IMAGE_ROLE_MATCH_MASS"
    secondary = "N3_SUBPIX_INITIAL_DIMENSION_PRIOR_GEOM_NOSELF"
    verdicts = {}
    for role, arm in (("primary", primary), ("secondary", secondary)):
        comparator = summaries["N3_SUBPIX"]["metrics"]["operational"]
        actual = summaries[arm]["metrics"]["operational"]
        paired_ci = paired_results[arm+"_minus_N3_SUBPIX"]["common_operational"]["metrics"]
        lower = all(actual[m]["mean"] is not None and actual[m]["mean"]<comparator[m]["mean"] for m in ("translation_cm", "rotation_deg"))
        ci_lower = all(paired_ci[m]["CI95"] is not None and paired_ci[m]["CI95"][1]<0 for m in ("translation_cm", "rotation_deg"))
        verdicts[role] = {"method": arm, "fixed_comparator": "N3_SUBPIX",
                         "both_operational_means_lower": lower, "both_paired_CI95_strictly_below_zero": ci_lower,
                         "new_pose_estimated": summaries[arm]["new_pose_estimated"],
                         "fallback_used": summaries[arm]["fallback_used"], "no_pose": summaries[arm]["no_pose"],
                         "not_independent_heldout_confirmation": True}
    after = protected_check(root, args.protected_state)
    metrics = {"schema": "kp_difficulty_metrics_v1", "complete": True,
        "new_methods": protocol["new_methods"], "fixed_comparators": list(COMPARATORS),
        "reference": "GEOMETRIC_PROXY: unchanged geometry-reconstructed DEV319; no independent physical GT",
        "population": {"frames": 319, "sessions": 13, "new_rows": 2552, "all_summary_arms": 17},
        "methods": summaries, "contrasts": paired_results, "verdicts": verdicts,
        "bootstrap": {"seed": 20260917, "resamples": 10000, "sessions": units.tolist(),
            "session_frame_counts": dict(Counter(r["session"] for r in inputs)),
            "algorithm": "NumPy default_rng multinomial session multiplicities; means weighted by retained frame counts",
            "dtype": "uint16", "draw_sha256": draw_sha, "original_draw_values_exact_by_digest": True,
            "numpy_version": np.__version__, "CI_quantiles": [.025, .975], "multiplicity_correction": False},
        "paired_scope_contract": {"common_operational": "Both output poses available, including fallback.",
            "candidate_new_pose": "Candidate produced a new pose; comparator output available. Historical fixed comparator is reused, not falsely counted as a new estimate.",
            "both_new_pose": "Both arms produced new poses. Empty against historical BASE/N3_SUBPIX by design.",
            "population_caveat": "Hypotheses were motivated by prior DEV319 diagnostics; this is post-hoc development, not new held-out validation."},
        "posthoc_correspondence": annotation_summary, "visibility_damage": visible,
        "reference_annotations": bind(output/"POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz", root),
        "case_selection": bind(output/"REVIEW_CASE_SELECTION.json", root),
        "uncertainty": "Sample variance/SD describe error dispersion; bootstrap CI describes paired mean uncertainty over repeated 13-session draws. Neither is seed variability or independent physical validation.",
        "corner_contract": "Distances use the frozen same-coordinate initial baseline phase; DIRECT_VISIBLE and SELF_OCCLUDED separated. Hidden references may be derived. Unknown inputs remain unknown. Report mean frame and mean corner separately. No success gate is applied from classification or hidden-corner error.",
        "zero_new_pose_in_statistics": True}
    write(output/"METRICS.json", metrics)
    public_sources = [doc/"PROTOCOL.json", doc/"CAUSAL_POSE_EXECUTION.json", sealed, scored,
                      old/"INPUTS.json", old/"FIXED_CONTROLS.jsonl.gz", old/"PREDICTIONS.jsonl.gz",
                      old/"LEARNED_PREDICTIONS.jsonl.gz", old/"REAL_CORRESPONDENCE_ROWS.jsonl.gz", old/"METRICS.json"]
    receipt = {"schema": "kp_difficulty_statistics_receipt_v1", "complete": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": [bind(path, root) for path in public_sources],
        "GT_inputs_opened_after_seal": True, "reference_target": C.binding(C.ROOT/"data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json"),
        "baseline_target_binding": C.binding(baseline_path),
        "rows": {"new_scored": 2552, "posthoc": len(annotations), "comparisons": len(contrasts), "bootstrap_scopes_per_pair": 3},
        "all_eight_methods_have_identical_319_ids": True, "original_comparator_metrics_exact": True,
        "bootstrap_original_draw_values_exact": True, "protected_before": before, "protected_after": after,
        "new_PnP_optimizer_calls": 0, "PnP_functions_guarded_against_calls": True,
        "new_detector_head_calls": 0, "new_training_updates": 0, "new_RGB_or_ray_renders": 0,
        "outputs": [bind(output/n, root) for n in names if n!="STATISTICS_RECEIPT.json"],
        "script": bind(Path(__file__).resolve(), root), "wall_seconds": time.monotonic()-began}
    write(output/"STATISTICS_RECEIPT.json", receipt)
    print(json.dumps({"complete": True, "methods": 17, "new_rows": 2552, "posthoc_rows": len(annotations),
                      "comparisons": len(contrasts), "wall_seconds": receipt["wall_seconds"], "verdicts": verdicts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
