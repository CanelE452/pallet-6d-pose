"""CAL-only total-error calibration for frozen image B/I measurements.

This module never loads RGB or TEST annotations.  Initial S queries and raw
coordinates are not inputs.  A fitted covariance is an empirical soft-loss
scale, not a physical confidence certificate or a license to move observations.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1"
N = BASE / "statewise_joint_ot_20261014_v2"
DEFAULT_OUTPUT = N / "nonsevere_v4_fix/calibration"
RECIPE = {
    "schema": "CAL_TOTAL_ERROR_RECIPE_V4_1",
    "covariance_scale": 1.0,
    "target_train_coverage": 0.80,
    "target_mahalanobis2": float(-2 * np.log(0.20)),
    "minimum_floor_px2": 1.0,
    "group_shrinkage_point_count": 20.0,
    "scene_scale_ratio_clip": [0.5, 2.0],
    "semantic_weight_clip": [0.20, 1.0],
    "semantic_pool_pseudo_count": 8.0,
    "prior_diagonal_shrinkage": 0.75,
    "prior_huber_radius": 3.0,
    "prior_min_std6": [float(np.deg2rad(1))] * 3 + [0.002, 0.002, 0.005],
    "floor_fit_population": "all current B/I observations on compatible CAL native geometry",
    "annotation_population": "human DIRECT only, reported separately; never substitutes reference",
    "floor_estimator": "session-balanced weighted 80% zero-centered residual ellipse coverage; isotropic addition",
    "groups": "B/I x corrected/fallback; group floor shrunk to state floor",
    "scale_feature": "saved upstream nonS-only query bbox diagonal; no S raw coordinates",
    "prior_coordinates": "log(R_reference_native R_coarse_native.T), tx/z delta, ty/z delta, log(z_reference/z_coarse)",
    "prior_weight_candidates_owned_by_root": [0.0, 0.25, 1.0, 4.0],
    "LOGO_scope": "uncertainty/prior fit only; upstream frozen state/BI models are conditional all-CAL fitted inputs",
    "physical_coverage_probability_guarantee": False,
    "TEST_labels_read": False,
    "S_coordinates_or_image_queries_read": False,
}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return _jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _quantile(x, w, q):
    x, w = np.asarray(x, float), np.asarray(w, float)
    if not len(x):
        return None
    order = np.argsort(x, kind="stable")
    cumulative = np.cumsum(w[order])
    return float(x[order[min(np.searchsorted(cumulative, q * cumulative[-1]), len(x) - 1)]])


def _cuboid(extents):
    a, b, c = np.asarray(extents, float) / 2
    return np.array([[-a, -b, -c], [a, -b, -c], [a, b, -c], [-a, b, -c],
                     [-a, -b, c], [a, -b, c], [a, b, c], [-a, b, c]])


def _project(xyz, R, t, K):
    cam = np.asarray(xyz, float) @ np.asarray(R, float).T + np.asarray(t, float)
    h = cam @ np.asarray(K, float).T
    return h[:, :2] / h[:, 2:]


def _group(record):
    corrected = record.get("source") != "VISIBLE_BACKBONE_FALLBACK"
    return f"{record['state']}:{'corrected' if corrected else 'fallback'}"


def _reference_alignment(row, label):
    """Geometry-only fixed mapping phase.  No estimated-pose/2D phase selection."""
    g = row["runtime"]["initial_geometry"]
    truth = label["target"]["truth"]
    xyz = np.asarray(g["native_xyz"], float)
    Q = np.asarray(g["physical_registration_Q"], float)
    R = np.asarray(truth["R"], float)
    body_R = np.asarray(truth["body_R"], float)
    Qref = body_R.T @ R
    mapping = np.asarray(label["metadata"]["native_to_reference_corner_ids"], int)
    mapped = _cuboid(truth["body_xyz"])[mapping] @ Qref
    gaps, rotations = [], []
    for phase in range(int(truth.get("order", 1))):
        S = Rotation.from_rotvec([0, phase * 2 * np.pi / int(truth.get("order", 1)), 0]).as_matrix()
        gaps.append(float(np.max(np.linalg.norm(xyz @ Q @ S.T - mapped, axis=1))))
        rotations.append(S)
    phase = int(np.argmin(gaps))
    return {
        "compatible": gaps[phase] < 1e-6,
        "geometry_gap_m": gaps[phase], "phase": phase,
        "phase_gaps_m": gaps,
        "R_reference_native": R @ rotations[phase] @ Q.T,
        "t_reference": np.asarray(truth["t"], float),
        "mapping": mapping, "xyz": xyz,
    }


def _prior_residual(row, alignment):
    g = row["runtime"]["initial_geometry"]
    center_xyz = alignment["xyz"].mean(axis=0)
    t0 = np.asarray(g["R_cf"], float) @ center_xyz + np.asarray(g["centroid"], float)
    tr = alignment["R_reference_native"] @ center_xyz + alignment["t_reference"]
    if min(t0[2], tr[2]) <= 0:
        raise ValueError("Nonpositive coarse/reference depth in CAL prior")
    omega = Rotation.from_matrix(alignment["R_reference_native"] @ np.asarray(g["R_cf"], float).T).as_rotvec()
    return np.r_[omega, tr[:2] / tr[2] - t0[:2] / t0[2], np.log(tr[2] / t0[2])]


def _cal_dataset(rows, annotations):
    rows = list(rows)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate CAL prediction IDs")
    if any(r.get("partition") != "CAL" for r in rows):
        raise ValueError("fit accepts CAL rows only; TEST/FINAL/severe rows forbidden")
    labels = annotations.get("rows", annotations)
    points, cases = [], []
    for row in rows:
        label = labels[row["id"]]
        if label["metadata"].get("partition") != "CAL":
            raise ValueError("Non-CAL annotation")
        runtime = row["runtime"]
        states = runtime["states"]
        initial_S = runtime["initial_geometry"]["initial_self_hidden"]
        alignment = _reference_alignment(row, label)
        scale = float(runtime.get("ot_domains", {}).get("nonself_prediction_scale_px", 0))
        if not np.isfinite(scale) or scale <= 0:
            # Only actual B/I image measurements are read, never raw native points.
            xy = np.asarray([m["xy"] for m in runtime["frozen_visible_measurements"]
                             if m["state"] in ("B", "I") and not initial_S[int(m["native_corner"])]], float)
            scale = float(np.linalg.norm(np.ptp(xy, axis=0))) if len(xy) > 1 else 1.0
        case = {"id": row["id"], "session": row["session"], "compatible": alignment["compatible"],
                "geometry_gap_m": alignment["geometry_gap_m"], "phase": alignment["phase"],
                "phase_gaps_m": alignment["phase_gaps_m"], "nonself_scale_px": scale,
                "initial_S_ids": [i for i, x in enumerate(initial_S) if x],
                "independent_physical_pose_truth": bool(label["metadata"].get("independent_physical_pose_truth", False)),
                "prior_residual6": _prior_residual(row, alignment) if alignment["compatible"] else None,
                "fit_eligible": alignment["compatible"],
                "exclusion_reason": None if alignment["compatible"] else "fixed physical/native 3D corner registration incompatible; no reference residual fit"}
        cases.append(case)
        for measurement in runtime["frozen_visible_measurements"]:
            k = int(measurement["native_corner"])
            # Gate before accessing ANY measurement coordinate or covariance.
            if states[k] == "S" or initial_S[k]:
                raise ValueError("Initial S entered frozen B/I measurements")
            if measurement["state"] not in ("B", "I") or states[k] != measurement["state"]:
                raise ValueError("Only actual B/I observations can be calibrated")
            q = np.asarray(measurement["xy"], float)
            cov = np.asarray(measurement["covariance_px2"], float)
            if cov.shape != (2, 2) or np.linalg.eigvalsh(cov).min() <= 0:
                raise ValueError("Observation covariance must be SPD")
            hstate = label["metadata"]["human_states_native"][k]
            ref_error = None
            if alignment["compatible"]:
                # Project only this nonS native vertex; no S image coordinate is formed.
                ref = _project(alignment["xyz"][[k]], alignment["R_reference_native"], alignment["t_reference"], row["K"])[0]
                ref_error = q - ref
            ann_error = None
            ref_id = int(alignment["mapping"][k])
            if hstate == "DIRECT_VISIBLE" and label["target"]["valid"][ref_id]:
                ann_error = q - np.asarray(label["target"]["gt"][ref_id], float)
            points.append({"id": row["id"], "session": row["session"], "native_corner": k,
                           "state": measurement["state"], "group": _group(measurement),
                           "source": measurement.get("source"), "xy": q, "covariance_px2": cov,
                           "reference_error_xy": ref_error, "annotation_error_xy": ann_error,
                           "human_state": hstate, "human_DIRECT": hstate == "DIRECT_VISIBLE",
                           "compatible": alignment["compatible"], "nonself_scale_px": scale})
    return points, cases


def _point_weights(points):
    """Each represented session equal, then frame equal, then point equal."""
    by_session = {}
    for p in points:
        by_session.setdefault(p["session"], {}).setdefault(p["id"], 0)
        by_session[p["session"]][p["id"]] += 1
    return np.asarray([1 / (len(by_session) * len(by_session[p["session"]]) * by_session[p["session"]][p["id"]])
                       for p in points], float)


def _floor(points, scale_reference):
    if not points:
        return None
    errors = np.asarray([p["reference_error_xy"] for p in points], float)
    covariance = np.asarray([p["covariance_px2"] for p in points], float)
    ratio = np.clip(np.asarray([p["nonself_scale_px"] for p in points]) / scale_reference,
                    *RECIPE["scene_scale_ratio_clip"])
    weights = _point_weights(points)
    def qmah(floor):
        total = covariance + floor * ratio[:, None, None] ** 2 * np.eye(2)
        mah2 = np.einsum("ni,nij,nj->n", errors, np.linalg.inv(total), errors)
        return _quantile(mah2, weights, RECIPE["target_train_coverage"])
    lo, hi = RECIPE["minimum_floor_px2"], 1.0
    while qmah(hi) > RECIPE["target_mahalanobis2"] and hi < 1e8:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if qmah(mid) > RECIPE["target_mahalanobis2"]:
            lo = mid
        else:
            hi = mid
    return float(max(RECIPE["minimum_floor_px2"], hi))


def _semantic_probability(points, pooled):
    if not points:
        return pooled
    w = _point_weights(points)
    empirical = float(np.dot(w, [p["human_DIRECT"] for p in points]) / w.sum())
    # The effective count is explicitly descriptive; no binomial CI is claimed.
    n = len(points)
    pseudo = RECIPE["semantic_pool_pseudo_count"]
    return float(np.clip((n * empirical + pseudo * pooled) / (n + pseudo), *RECIPE["semantic_weight_clip"]))


def _fit_prior(cases):
    eligible = [c for c in cases if c["fit_eligible"]]
    x = np.asarray([c["prior_residual6"] for c in eligible], float)
    sessions = {c["session"] for c in eligible}
    counts = {s: sum(c["session"] == s for c in eligible) for s in sessions}
    w = np.asarray([1 / (len(sessions) * counts[c["session"]]) for c in eligible])
    median = np.asarray([_quantile(x[:, j], w, .5) for j in range(6)])
    mad = np.asarray([_quantile(np.abs(x[:, j] - median[j]), w, .5) for j in range(6)])
    std = np.maximum(np.sqrt((1.4826 * mad) ** 2 + median ** 2), RECIPE["prior_min_std6"])
    norm = np.sqrt(np.sum((x / std) ** 2, axis=1))
    robust = np.minimum(1.0, RECIPE["prior_huber_radius"] / np.maximum(norm, 1e-12))
    covariance = np.einsum("n,ni,nj->ij", w * robust, x, x) / np.sum(w * robust)
    shrink = RECIPE["prior_diagonal_shrinkage"]
    covariance = (1 - shrink) * covariance + shrink * np.diag(np.diag(covariance))
    covariance += np.diag(np.maximum(np.asarray(RECIPE["prior_min_std6"]) ** 2 - np.diag(covariance), 0))
    eig = np.linalg.eigvalsh(covariance)
    covariance += np.eye(6) * max(0, 1e-10 - eig.min())
    return {"covariance6": covariance, "median_bias6": median, "MAD6": mad,
            "fit_count": len(x), "session_balanced": True, "zero_center_second_moment": True,
            "robust_weights": [{"id": c["id"], "weight": float(r), "standardized_norm": float(n)}
                               for c, r, n in zip(eligible, robust, norm)],
            "min_eigenvalue": float(np.linalg.eigvalsh(covariance).min())}


def _fit(points, cases):
    eligible = [p for p in points if p["compatible"]]
    if not eligible:
        raise ValueError("No compatible CAL B/I residuals")
    compatible_cases = [c for c in cases if c["fit_eligible"]]
    scale_reference = float(np.median([c["nonself_scale_px"] for c in compatible_cases]))
    state_floor, state_weights, groups = {}, {}, {}
    for state in ("B", "I"):
        selected = [p for p in eligible if p["state"] == state]
        state_floor[state] = _floor(selected, scale_reference) or _floor(eligible, scale_reference)
        state_weights[state] = _semantic_probability(selected, _semantic_probability(eligible, 1.0))
        for source_class in ("corrected", "fallback"):
            key = f"{state}:{source_class}"
            subset = [p for p in selected if p["group"] == key]
            raw_floor = _floor(subset, scale_reference)
            shrink = len(subset) / (len(subset) + RECIPE["group_shrinkage_point_count"])
            groups[key] = {"count": len(subset), "session_count": len({p["session"] for p in subset}),
                           "raw_floor_px2": raw_floor,
                           "floor_px2": shrink * raw_floor + (1 - shrink) * state_floor[state] if raw_floor is not None else state_floor[state],
                           "shrinkage_weight": shrink,
                           "reliability_weight": _semantic_probability(subset, state_weights[state]),
                           "human_state_counts": {s: sum(p["human_state"] == s for p in subset) for s in sorted({p["human_state"] for p in subset})}}
    prior = _fit_prior(cases)
    return _jsonable({"schema": "STATEWISE_UNCERTAINTY_CAL_V4_1", "recipe": RECIPE,
                      "fit_partition": "CAL", "fit_ids": [c["id"] for c in cases],
                      "compatible_fit_ids": [c["id"] for c in compatible_cases],
                      "incompatible_ids": [c["id"] for c in cases if not c["fit_eligible"]],
                      "reference_scale_px": scale_reference, "covariance_scale": 1.0,
                      "state_BI_total_error_floor_px2": state_floor, "state_loss_weights": state_weights,
                      "groups": groups, "prior_covariance6": prior["covariance6"], "prior_fit": prior,
                      "TEST_labels_read": False, "S_coordinate_inputs_used": False,
                      "runtime_GT_human_or_compatibility_inputs": False,
                      "physical_probability_calibrated": False})


def load_uncertainty_model(path):
    model = json.loads(Path(path).read_text())
    if model.get("schema") != "STATEWISE_UNCERTAINTY_CAL_V4_1" or model.get("fit_partition") != "CAL":
        raise ValueError("Unrecognized CAL calibration model")
    return model


def apply_calibrated_measurements(source_visible_records, model, states=None, nonself_scale_px=None):
    """Pure runtime transform; coordinate and conditional covariance stay immutable.

    Optional nonself_scale_px MUST come from the nonS-only upstream context.
    No image, human label, reference coordinate or compatibility flag is accepted.
    """
    result = []
    scale = model["reference_scale_px"] if nonself_scale_px is None else float(nonself_scale_px)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("NonS scale must be finite and positive")
    ratio = float(np.clip(scale / model["reference_scale_px"], *RECIPE["scene_scale_ratio_clip"]))
    for source in source_visible_records:
        k, state = int(source["native_corner"]), source["state"]
        # S is rejected before reading its coordinates/covariance or copying it.
        if state not in ("B", "I") or (states is not None and states[k] != state):
            raise ValueError("apply accepts actual nonS B/I measurements only")
        record = copy.deepcopy(source)
        key = _group(source)
        group = model["groups"].get(key)
        floor = float(group["floor_px2"] if group else model["state_BI_total_error_floor_px2"][state]) * ratio ** 2
        cov = np.asarray(source["covariance_px2"], float)
        record["fit_covariance_px2"] = (float(model["covariance_scale"]) * cov + floor * np.eye(2)).tolist()
        record["fit_reliability_weight"] = float(group["reliability_weight"] if group else model["state_loss_weights"][state])
        record["uncertainty_calibration"] = {"schema": model["schema"], "group": key,
                                              "additional_isotropic_floor_px2": floor, "nonself_scale_ratio": ratio,
                                              "xy_changed": False, "observed_covariance_changed": False,
                                              "fit_covariance_replaces_pooled_floor": True,
                                              "physical_confidence_guarantee": False}
        result.append(record)
    return result


def solver_config_from_model(model, prior_weight=0.0):
    return {"covariance_scale": float(model["covariance_scale"]),
            # Model stores isotropic scalar; current pure solver validates a PSD matrix.
            "state_BI_total_error_floor_px2": {s: (float(v) * np.eye(2)).tolist()
                for s, v in model["state_BI_total_error_floor_px2"].items()},
            "state_loss_weights": copy.deepcopy(model["state_loss_weights"]),
            "prior_covariance6": copy.deepcopy(model["prior_covariance6"]), "prior_weight": float(prior_weight)}


def _evaluate(points, cases, model):
    details = []
    for p in points:
        measurement = {k: p[k] for k in ("native_corner", "state", "xy", "covariance_px2", "source")}
        calibrated = apply_calibrated_measurements([measurement], model, nonself_scale_px=p["nonself_scale_px"])[0]
        effective = np.asarray(calibrated["fit_covariance_px2"])
        d = {k: p[k] for k in ("id", "session", "native_corner", "state", "group", "human_state", "compatible")}
        d["fit_reliability_weight"] = calibrated["fit_reliability_weight"]
        d["observed_std_max_px"] = float(np.sqrt(np.linalg.eigvalsh(p["covariance_px2"]).max()))
        d["effective_std_max_px"] = float(np.sqrt(np.linalg.eigvalsh(effective).max()))
        for target in ("reference", "annotation"):
            e = p[f"{target}_error_xy"]
            d[f"{target}_error_xy"] = e
            d[f"{target}_error_px"] = float(np.linalg.norm(e)) if e is not None else None
            for name, cov in (("conditional", p["covariance_px2"]), ("effective", effective)):
                d[f"{target}_{name}_Mah"] = float(np.sqrt(e @ np.linalg.solve(cov, e))) if e is not None else None
        d["annotation_reference_discrepancy_px"] = float(np.linalg.norm(p["reference_error_xy"] - p["annotation_error_xy"])) if p["reference_error_xy"] is not None and p["annotation_error_xy"] is not None else None
        details.append(d)
    summary = {}
    for target in ("reference", "annotation"):
        for scope, selected in (("all", details), ("compatible", [d for d in details if d["compatible"]]),
                                ("incompatible", [d for d in details if not d["compatible"]]),
                                ("human_DIRECT", [d for d in details if d["human_state"] == "DIRECT_VISIBLE"])):
            valid = [d for d in selected if d[f"{target}_error_px"] is not None]
            if not valid:
                continue
            group = {"count": len(valid), "frame_count": len({d["id"] for d in valid}),
                     "mean_error_px": float(np.mean([d[f"{target}_error_px"] for d in valid])),
                     "median_error_px": float(np.median([d[f"{target}_error_px"] for d in valid])),
                     "max_error_px": float(max(d[f"{target}_error_px"] for d in valid))}
            for covname in ("conditional", "effective"):
                mahs = np.asarray([d[f"{target}_{covname}_Mah"] for d in valid])
                frame_max = [max(d[f"{target}_{covname}_Mah"] for d in valid if d["id"] == rid) for rid in sorted({d["id"] for d in valid})]
                group[covname] = {"median_Mah": float(np.median(mahs)),
                                  "point_coverage": {str(t): float(np.mean(mahs <= t)) for t in (1, 2, 3, 4, 6, 9, 12)},
                                  "frame_all_point_coverage": {str(t): float(np.mean(np.asarray(frame_max) <= t)) for t in (1, 2, 3, 4, 6, 9, 12)}}
            summary[f"{target}:{scope}"] = group
    semantic = [d for d in details if d["compatible"]]
    summary["semantic_reliability"] = {"count": len(semantic),
        "Brier_descriptive": float(np.mean([(d["fit_reliability_weight"] - (d["human_state"] == "DIRECT_VISIBLE")) ** 2 for d in semantic])) if semantic else None,
        "hard_visibility_filter_applied": False}
    return _jsonable({"summary": summary, "points": details})


def _detail_summary(details):
    result = {}
    scopes = {"all": details}
    scopes.update({"compatible": [d for d in details if d["compatible"]],
                   "incompatible": [d for d in details if not d["compatible"]]})
    scopes.update({g: [d for d in details if d["group"] == g] for g in sorted({d["group"] for d in details})})
    scopes.update({"human:" + s: [d for d in details if d["human_state"] == s] for s in sorted({d["human_state"] for d in details})})
    for scope, group in scopes.items():
        for target in ("reference", "annotation"):
            valid = [d for d in group if d[f"{target}_error_px"] is not None]
            if not valid:
                continue
            errors = np.asarray([d[f"{target}_error_xy"] for d in valid])
            report = {"count": len(valid), "frame_count": len({d["id"] for d in valid}),
                      "mean_error_px": float(np.mean([d[f"{target}_error_px"] for d in valid])),
                      "median_error_px": float(np.median([d[f"{target}_error_px"] for d in valid])),
                      "unweighted_mean_bias_xy_px": errors.mean(0),
                      "unweighted_zero_center_error_second_moment_px2": errors.T @ errors / len(errors)}
            for covname in ("conditional", "effective"):
                mahs = np.asarray([d[f"{target}_{covname}_Mah"] for d in valid])
                frame_max = np.asarray([max(d[f"{target}_{covname}_Mah"] for d in valid if d["id"] == rid) for rid in sorted({d["id"] for d in valid})])
                report[covname] = {"point_coverage": {str(t): float(np.mean(mahs <= t)) for t in (1, 2, 3, 4, 6, 9, 12)},
                                   "frame_all_point_coverage": {str(t): float(np.mean(frame_max <= t)) for t in (1, 2, 3, 4, 6, 9, 12)},
                                   "median_Mah": float(np.median(mahs))}
            result[f"{target}:{scope}"] = report
    discrepancies = [d["annotation_reference_discrepancy_px"] for d in details if d.get("annotation_reference_discrepancy_px") is not None]
    result["DIRECT_annotation_reference_discrepancy"] = {"count": len(discrepancies),
        "mean_px": float(np.mean(discrepancies)) if discrepancies else None,
        "max_px": float(max(discrepancies)) if discrepancies else None}
    return _jsonable(result)


def _contract_checks(rows, model, report):
    checks = {"CAL_only_fit_ids": all(r["partition"] == "CAL" for r in rows),
              "LOGO_train_heldout_disjoint": all(not set(f["train_ids"]) & set(f["heldout_ids"]) for f in report["LOGO_folds"]),
              "LOGO_model_fit_ids_equal_fold_train_ids": all(set(f["model"]["fit_ids"]) == set(f["train_ids"]) for f in report["LOGO_folds"]),
              "prior_covariance6_SPD": bool(np.linalg.eigvalsh(model["prior_covariance6"]).min() > 0)}
    unchanged, spd, nonS = True, True, True
    for row in rows:
        records = row["runtime"]["frozen_visible_measurements"]
        before = json.dumps(_jsonable(records), sort_keys=True)
        result = apply_calibrated_measurements(records, model, states=row["runtime"]["states"],
            nonself_scale_px=row["runtime"]["ot_domains"]["nonself_prediction_scale_px"])
        unchanged &= before == json.dumps(_jsonable(records), sort_keys=True)
        for source, output in zip(records, result):
            unchanged &= source["xy"] == output["xy"] and source["covariance_px2"] == output["covariance_px2"]
            spd &= np.linalg.eigvalsh(output["fit_covariance_px2"]).min() > 0
            nonS &= output["state"] in ("B", "I") and not row["runtime"]["initial_geometry"]["initial_self_hidden"][source["native_corner"]]
    checks.update({"source_records_xy_and_covariance_immutable": bool(unchanged),
                   "all_effective_covariances_SPD": bool(spd), "initialS_never_transformed": bool(nonS)})
    # Malformed coordinates prove that S is rejected before any numeric use.
    try:
        apply_calibrated_measurements([{"native_corner": 0, "state": "S", "xy": "NEVER_READ", "covariance_px2": "NEVER_READ"}], model)
        checks["S_rejected_before_coordinates"] = False
    except ValueError as error:
        checks["S_rejected_before_coordinates"] = "actual nonS B/I" in str(error)
    config = solver_config_from_model(model)
    checks["exported_state_floor_matrix_PSD"] = all(np.asarray(v).shape == (2, 2) and np.linalg.eigvalsh(v).min() >= 0 for v in config["state_BI_total_error_floor_px2"].values())
    try:
        import importlib
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        _configuration = importlib.import_module("scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_uncertain_solver_v4")._configuration
        _configuration(config)
        checks["actual_solver_config_accepts_export"] = True
    except ImportError:
        checks["actual_solver_config_accepts_export"] = None
    except Exception as error:
        checks["actual_solver_config_accepts_export"] = False
        checks["actual_solver_error"] = repr(error)
    checks["all_required_checks_pass"] = all(value is not False for value in checks.values())
    return _jsonable(checks)


def _prior_evaluation(cases, model):
    covariance = np.asarray(model["prior_covariance6"], float)
    details = []
    for c in cases:
        if not c["fit_eligible"]:
            continue
        e = np.asarray(c["prior_residual6"], float)
        details.append({"id": c["id"], "session": c["session"], "residual6": e,
                        "Mah6": float(np.sqrt(e @ np.linalg.solve(covariance, e)))})
    return _jsonable({"count": len(details), "cases": details,
        "coordinate_order": RECIPE["prior_coordinates"],
        "Mah6_empirical_coverage": {str(t): float(np.mean([d["Mah6"] <= t for d in details])) for t in (1, 2, 3, 4, 6, 9, 12)},
        "physical_probability_guarantee": False})


def _coverage_plot(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    details = report["LOGO_heldout_points"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, target, title in zip(axes, ("reference", "annotation"),
                               ("Fixed-native derived reference", "Human DIRECT annotation")):
        valid = [d for d in details if d[f"{target}_effective_Mah"] is not None]
        for covname, color in (("conditional", "#d95f02"), ("effective", "#1b9e77")):
            values = np.sort([d[f"{target}_{covname}_Mah"] for d in valid])
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=covname, color=color)
        ax.set(xlim=(0, 12), ylim=(0, 1), xlabel="Mahalanobis radius (sensitivity only)", ylabel="Empirical point coverage", title=f"{title} (n={len(valid)})")
        ax.axvline(3, color="grey", lw=.8, ls="--")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("CAL leave-one-session-out uncertainty fit; upstream heads remain frozen\nNo physical probability guarantee / no TEST labels")
    fig.savefig(output / "CAL_LOGO_COVERAGE.png", dpi=180)
    fig.savefig(output / "CAL_LOGO_COVERAGE.svg")
    plt.close(fig)


def _readme(output, model, report):
    summary = report["LOGO_aggregate"]
    ref, ann = summary.get("reference:all", {}), summary.get("annotation:all", {})
    lines = ["# CAL 전용 전체 오차 불확실성 모델", "",
        f"CAL {report['CAL_frames']}건 중 고정 3D 대응이 맞는 {report['compatible_frames']}건으로 추가 분산과 prior covariance를 fit했다. 대응 불일치 {report['incompatible_frames']}건은 제외 사유와 DIRECT annotation 오차를 별도 공개한다. TEST label/RGB를 읽지 않는다.", "",
        "원 B/I `xy`와 `covariance_px2`는 유지한다. solver용 `fit_covariance_px2`는 원 조건부 covariance에 CAL에서 추정한 isotropic floor를 더한 값이다. `fit_reliability_weight`는 CAL human DIRECT 빈도를 부드럽게 반영하며 관측을 삭제하지 않는다.", "",
        "원 조건부 covariance는 선 교차 resampling과 detector 2/3px 가정의 오차를 담는다. 새 floor는 zero-centered 전체 재투영 오차로 fit하여 semantic/대응/model bias를 일부 포함한다. 각 frame/session의 상관관계, 모델 부정확성, 가시성 착오가 모두 제거된다는 뜻은 아니다.", "",
        "추정법은 session→frame→point 동일 가중 80% residual ellipse coverage다. B/I × corrected/fallback 그룹은 상태 pooled floor로 shrink한다. 큰 오차를 삭제하지 않고 원 오차 분포/상위 사례를 보고한다. 80% quantile 추정은 극단 오류 영향이 제한되는 robust fit이며 모든 오류를 덮는 hard bound가 아니다.", "",
        "DIRECT annotation 오차와 fixed-native derived reference 재투영 오차는 별도다. 후자는 독립 물리 계측 GT가 아니다. 경험적 Mahalanobis 3 coverage를 물리적 99% 보장으로 해석할 수 없다.", "",
        "LOGO는 held-out session을 제외해 scale normalization, floor, reliability, prior covariance를 모두 다시 fit한다. upstream state/BI head는 기존 all-CAL fit 입력이므로 전체 파이프라인의 완전 독립 LOGO가 아니다.", "",
        f"pooled floor(px²): B={model['state_BI_total_error_floor_px2']['B']:.4f}, I={model['state_BI_total_error_floor_px2']['I']:.4f}.", "",
        f"LOGO reference point count={ref.get('count')}; Mah≤3 conditional={ref.get('conditional', {}).get('point_coverage', {}).get('3')}, effective={ref.get('effective', {}).get('point_coverage', {}).get('3')}.",
        f"LOGO DIRECT annotation point count={ann.get('count')}; Mah≤3 conditional={ann.get('conditional', {}).get('point_coverage', {}).get('3')}, effective={ann.get('effective', {}).get('point_coverage', {}).get('3')}.", "",
        "prior는 fixed-native geometry mapping으로 정한 reference와 coarse pose의 camera tangent/normalized translation 오차를 사용한다. MAD+bias scale, Huber weighted zero-centered second moment, diagonal shrinkage로 SPD covariance를 만든다. prior_weight [0,0.25,1,4] 선택은 별도 root CAL 검증 범위다.", "",
        "런타임 apply API는 B/I records, frozen model, state 목록, nonS-only bbox scale만 받는다. human/reference/GT compatibility 조건으로 기능을 바꾸지 않는다. 초기 S는 좌표 접근 전에 차단한다. override covariance를 적용하면 pooled floor를 다시 더하지 않는다.", "",
        "[MODEL.json](MODEL.json), [LOGO 보고서](CAL_LOGO_REPORT.json), [coverage 그림](CAL_LOGO_COVERAGE.png), [API 검증](API_CHECKS.json).", ""]
    (output / "README_KO.md").write_text("\n".join(lines))


def fit_cal_uncertainty_model(cal_prediction_rows, isolated_cal_annotations, output_dir=None, input_binding=None):
    """Fit full CAL and uncertainty-only leave-one-session-out models.

    Every fold refits floor, scale feature normalization, reliability AND prior
    covariance without the held-out session.  Upstream predictors stay frozen.
    """
    cal_prediction_rows = list(cal_prediction_rows)
    points, cases = _cal_dataset(cal_prediction_rows, isolated_cal_annotations)
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        if (output / "MODEL.json").exists() or (output / "PROTOCOL.json").exists():
            raise FileExistsError("Preserve completed CAL calibration/protocol: " + str(output))
        output.mkdir(parents=True, exist_ok=True)
        # Recipe/input manifest are committed before any fitted parameter exists.
        _write(output / "PROTOCOL.json", {"recipe": RECIPE, "input_binding": input_binding or {},
            "source_sha256": _sha(__file__), "CAL_ids": [c["id"] for c in cases],
            "protocol_written_before_fit": True})
    full = _fit(points, cases)
    folds, heldout = [], []
    for session in sorted({c["session"] for c in cases}):
        train_cases = [c for c in cases if c["session"] != session]
        train_points = [p for p in points if p["session"] != session]
        held_points = [p for p in points if p["session"] == session]
        fold_model = _fit(train_points, train_cases)
        evaluation = _evaluate(held_points, [c for c in cases if c["session"] == session], fold_model)
        for d in evaluation["points"]:
            d["heldout_session"] = session
            heldout.append(d)
        folds.append({"heldout_session": session, "train_ids": [c["id"] for c in train_cases],
                      "heldout_ids": [c["id"] for c in cases if c["session"] == session],
                      "model": fold_model, "heldout_evaluation": evaluation,
                      "heldout_prior_evaluation": _prior_evaluation([c for c in cases if c["session"] == session], fold_model)})
    report = {"schema": "CAL_TOTAL_ERROR_LOGO_REPORT_V4_1", "recipe": RECIPE,
              "cases": cases, "CAL_frames": len(cases),
              "compatible_frames": sum(c["compatible"] for c in cases),
              "incompatible_frames": sum(not c["compatible"] for c in cases),
              "current_BI_points": len(points), "compatible_BI_points": sum(p["compatible"] for p in points),
              "full_fit_resubstitution": _evaluate(points, cases, full), "LOGO_folds": folds,
              "LOGO_heldout_points": heldout,
              "LOGO_aggregate": _detail_summary(heldout),
              "full_fit_prior_resubstitution": _prior_evaluation(cases, full),
              "large_reference_errors_unfiltered": sorted([{k: p[k] for k in ("id", "native_corner", "state", "group", "human_state")}
                    | {"error_px": float(np.linalg.norm(p["reference_error_xy"]))} for p in points if p["reference_error_xy"] is not None], key=lambda x: -x["error_px"])[:20],
              "incompatible_annotation_diagnostics": _evaluate([p for p in points if not p["compatible"]], cases, full),
              "interpretation": "Empirical soft-loss scale. Derived reference is not independent physical metrology. Sigma3 is not calibrated physical 99% proof. Correlation and model mismatch may remain.",
              "input_binding": input_binding or {}, "TEST_labels_read": False,
              "RGB_opened": False, "S_coordinate_inputs_used": False,
              "runtime_groundtruth_compatibility_branch": False}
    if output_dir is not None:
        _write(output / "MODEL.json", full)
        _write(output / "CAL_LOGO_REPORT.json", report)
        _write(output / "SOLVER_CONFIG_EXAMPLE.json", solver_config_from_model(full, prior_weight=0.0))
        _write(output / "API_CHECKS.json", _contract_checks(cal_prediction_rows, full, report))
        _coverage_plot(output, report)
        _readme(output, full, report)
        (output / Path(__file__).name).write_text(Path(__file__).read_text())
        _write(output / "SEAL.json", {"files": {p.name: _sha(p) for p in output.iterdir() if p.is_file()},
                                      "source_sha256": _sha(__file__), "TEST_labels_read": False})
    return full, report


def _load_cal_rows(path):
    path = Path(path)
    if path.name.endswith(".jsonl.gz") or path.suffix == ".jsonl":
        result = []
        with (gzip.open if path.suffix == ".gz" else open)(path, "rt") as f:
            for line in f:
                row = json.loads(line)
                # Filter prediction-only rows before any measurement/geometry access.
                if row.get("partition") == "CAL" and row.get("cohort", "PRIMARY_NONSEVERE") == "PRIMARY_NONSEVERE":
                    result.append(row)
        return result
    value = json.loads(path.read_text())
    rows = value.get("rows", value) if isinstance(value, dict) else value
    rows = list(rows.values()) if isinstance(rows, dict) else rows
    return [r for r in rows if r.get("partition") == "CAL" and r.get("cohort", "PRIMARY_NONSEVERE") == "PRIMARY_NONSEVERE"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=N / "evaluation_v3/PREDICTIONS.jsonl.gz")
    parser.add_argument("--cal-annotations", type=Path, default=BASE / "OOF_pose_v1/ISOLATED_CAL_SCORING.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "preliminary_v3")
    args = parser.parse_args()
    labels = json.loads(args.cal_annotations.read_text())
    if any(labels.get(k) for k in ("TEST_rows", "FINAL_rows", "severe_rows")):
        raise ValueError("Annotation input must be isolated CAL only")
    rows = _load_cal_rows(args.predictions)
    model, report = fit_cal_uncertainty_model(rows, labels, args.output,
        {"predictions": str(args.predictions), "predictions_sha256": _sha(args.predictions),
         "CAL_annotations": str(args.cal_annotations), "CAL_annotations_sha256": _sha(args.cal_annotations)})
    print(json.dumps({"output": str(args.output), "CAL_frames": report["CAL_frames"],
                      "compatible_frames": report["compatible_frames"], "groups": model["groups"],
                      "state_floor_px2": model["state_BI_total_error_floor_px2"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
