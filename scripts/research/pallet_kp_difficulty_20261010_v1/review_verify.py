#!/usr/bin/env python3
"""Independent public-file arithmetic and contract verifier; Python >= 3.9.

No experiment module, detector, model, OpenCV, NumPy, private dataset or GPU is
loaded. Checks are recomputed on every invocation. A PASS verifies the described
public arithmetic/contracts, not physical ground truth or an inference rerun.
"""
import argparse
import base64
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import zlib


NAME = "pallet_kp_difficulty_20261010_v1"
OLD = "pallet_observation_refiner_20261009_v1"
ARMS = ("GEOMETRY_ONLY", "IMAGE_NO_ROLE", "IMAGE_ROLE")
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
SIGNS = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))
METRICS = {"translation_cm": ("translation_cm", 1.),
           "rotation_deg": ("rotation_deg", 1.), "ADDsym_cm": ("ADDsym_m", 100.)}
ROLES = ("BOUNDARY", "INTERNAL", "UNAVAILABLE")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def close(a, b, tolerance=1e-8):
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance * max(1., abs(a), abs(b))


def equal(a, b, tolerance=1e-8, label="value"):
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), label + " keys differ")
        for key in a:
            equal(a[key], b[key], tolerance, label + "." + key)
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), label + " length differs")
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, tolerance, label + "[" + str(i) + "]")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        require(isinstance(b, (int, float)) and close(a, b, tolerance), label + " differs")
    else:
        require(a == b, label + " differs")


def quantile(values, probability):
    if not values:
        return None
    a = sorted(values)
    x = (len(a) - 1) * probability
    i = int(math.floor(x))
    return a[i] + (a[min(i + 1, len(a) - 1)] - a[i]) * (x - i)


def distribution(values, detailed=False):
    n = len(values)
    mean = math.fsum(values) / n if n else None
    variance = math.fsum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else None
    result = dict(n=n, mean=mean, sample_variance=variance,
                  sample_std=math.sqrt(variance) if variance is not None else None,
                  max=max(values) if n else None)
    if detailed:
        result["min"] = min(values) if n else None
        result.update({"P" + str(round(p * 100)).zfill(2): quantile(values, p)
                       for p in (.01, .1, .25, .5, .75, .9, .99)})
    else:
        result.update(median=quantile(values, .5), P90=quantile(values, .9))
    return result


def compare_distribution(values, stored, detailed=False):
    calculated = distribution(values, detailed)
    for key, value in calculated.items():
        require(key in stored and close(value, stored[key], 2e-8), "distribution " + key + " differs")
    return calculated


def finite_point(point):
    return isinstance(point, list) and len(point) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)


def valid_seed(point):
    return finite_point(point) and point != [-1, -1]


def in_frame(point, hw):
    return valid_seed(point) and 0 <= point[0] < hw[1] and 0 <= point[1] < hw[0]


def probability(scores):
    # Separate spatial normalization and log odds, independent of the producer's
    # joint66 normalization. Both represent exactly the same probability law.
    top = max(scores[:65])
    spatial = [math.exp(s - top) for s in scores[:65]]
    total = math.fsum(spatial)
    conditional = [s / total for s in spatial]
    log_mass = top + math.log(total)
    odds = log_mass - scores[65]
    pmatch = 1. / (1. + math.exp(-odds)) if odds >= 0 else math.exp(odds) / (1. + math.exp(odds))
    return pmatch, conditional, log_mass, odds


def scalar_decode(seed, choices):
    """Independent 2x2 covariance eigensystem, without an SVD library."""
    lines = {}
    for edge, (a, b) in enumerate(EDGES):
        if not valid_seed(seed[a]) or not valid_seed(seed[b]):
            continue
        d = [seed[b][j] - seed[a][j] for j in range(2)]
        length = math.hypot(*d)
        if length <= 1e-6:
            continue
        normal = [-d[1] / length, d[0] / length]
        samples = []
        for k in range(7):
            index, fraction = edge * 7 + k, (k + 1) / 8
            if choices[index] < 65:
                xy = [(1 - fraction) * seed[a][j] + fraction * seed[b][j] + (choices[index] - 32) * normal[j] for j in range(2)]
                samples.append((index, xy))
        if len(samples) < 2:
            continue
        pts = [p for _, p in samples]
        center = [math.fsum(p[j] for p in pts) / len(pts) for j in range(2)]
        xx = math.fsum((p[0] - center[0]) ** 2 for p in pts)
        xy = math.fsum((p[0] - center[0]) * (p[1] - center[1]) for p in pts)
        yy = math.fsum((p[1] - center[1]) ** 2 for p in pts)
        largest = (xx + yy + math.hypot(xx - yy, 2 * xy)) / 2
        if math.sqrt(max(0., largest)) < 1e-6:
            continue
        theta = .5 * math.atan2(2 * xy, xx - yy)
        n = [-math.sin(theta), math.cos(theta)]
        lines[edge] = dict(normal=n, offset=math.fsum(n[j] * center[j] for j in range(2)), queries=[i for i, _ in samples])
    corners = {}
    for corner in range(8):
        incident = [e for e in sorted(lines) if corner in EDGES[e]]
        pairs = []
        for i, first in enumerate(incident):
            for second in incident[i + 1:]:
                a, b = lines[first]["normal"], lines[second]["normal"]
                det = a[0] * b[1] - a[1] * b[0]
                if abs(det) > 1e-6:
                    pairs.append((-abs(det), first, second, det))
        if pairs:
            _, first, second, det = min(pairs)
            a, b = lines[first]["normal"], lines[second]["normal"]
            c, d = lines[first]["offset"], lines[second]["offset"]
            corners[corner] = [(c * b[1] - a[1] * d) / det, (a[0] * d - c * b[0]) / det]
    return lines, corners


class Review:
    def __init__(self, root, output):
        self.root, self.output = root, output
        self.doc = root / "_docs" / "experiments" / NAME
        self.old = root / "_docs" / "experiments" / OLD
        self.inputs = {}
        self.checks = []
        self.data = {}
        self.counts = Counter()

    def path(self, path):
        absolute = Path(path)
        if not absolute.is_absolute():
            absolute = self.root / absolute
        absolute = absolute.resolve()
        require(absolute.is_relative_to(self.root), "public input path escapes --root")
        require(absolute.is_file(), "missing public file: " + str(absolute.relative_to(self.root)))
        relative = str(absolute.relative_to(self.root))
        if relative not in self.inputs:
            self.inputs[relative] = dict(path=relative, sha256=sha(absolute), bytes=absolute.stat().st_size)
        return absolute

    def json(self, path):
        return json.loads(self.path(path).read_text(encoding="utf-8"))

    def rows(self, path):
        with gzip.open(self.path(path), "rt", encoding="utf-8") as stream:
            for line in stream:
                require(line.strip(), "blank raw row")
                yield json.loads(line)

    def check(self, name, function):
        start = time.perf_counter()
        try:
            details = function()
            self.checks.append(dict(name=name, passed=True, details=details,
                                    seconds=time.perf_counter() - start))
        except Exception as exc:
            self.checks.append(dict(name=name, passed=False, error=type(exc).__name__ + ": " + str(exc),
                                    seconds=time.perf_counter() - start))

    def prior(self):
        prior = self.json(self.doc / "PRIOR_PUBLICATION_BINDINGS.json")
        rows = prior["protected_files"]
        require(len(rows) == 111 and len({r["path"] for r in rows}) == 111, "expected 111 unique protected files")
        for row in rows:
            p = self.path(row["path"])
            require(p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], "protected original changed: " + row["path"])
        return dict(original_immutable_files=111, original_publication_unchanged=True,
                    declared_baseline_commit=prior["baseline_commit"], signed_provenance_verified=False)

    def load_population(self):
        original = self.json(self.old / "INPUTS.json")
        frames = {f["id"]: f for f in original["frames"]}
        require(len(original["frames"]) == len(frames) == 319, "original population is not 319 unique frames")
        require(len({f["session"] for f in frames.values()}) == 13, "expected 13 sessions")
        protocol = self.json(self.doc / "PROTOCOL.json")
        methods = protocol["new_methods"]
        if isinstance(methods, dict):
            methods = list(methods)
        if methods and isinstance(methods[0], dict):
            methods = [m.get("name", m.get("method")) for m in methods]
        require(len(methods) == len(set(methods)) == 8, "protocol requires eight unique new methods")
        self.data.update(frames=frames, ids=sorted(frames), new_methods=methods)
        return dict(frames=319, sessions=13, new_methods=methods)

    def raw_pose_contract(self):
        frames, ids = self.data["frames"], self.data["ids"]
        sealed = {(r["method"], r["id"]): r for r in self.rows(self.doc / "CAUSAL_GEOMETRY_SEALED.jsonl.gz")}
        predictions = list(self.rows(self.doc / "PREDICTIONS.jsonl.gz"))
        require(len(predictions) == len(sealed) == 2552, "expected 2552 unique raw/scored rows")
        per_arm, pairs = defaultdict(set), set()
        statuses, projection_count = defaultdict(Counter), 0
        max_projection_error = 0.
        for row in predictions:
            pair = row["method"], row["id"]
            require(pair not in pairs and pair in sealed, "duplicate or unsealed prediction")
            pairs.add(pair)
            per_arm[row["method"]].add(row["id"])
            before = {k: v for k, v in row.items() if k not in ("pose", "corner")}
            equal(before, sealed[pair], 0., "sealed geometry")
            f = frames[row["id"]]
            require(row["session"] == f["session"], "session changed")
            for key in ("K", "xyz", "raw_hw", "selected_index"):
                equal(row[key], f[key], 0., key)
            require(not row["inference_GT_input"] and not row["oracle"], "deployable inference declares GT input")
            require(not row["reprojections_reused_as_observations"] and not row["no_match_points_filled_from_Base"], "forbidden feedback or no-match fill")
            new, fallback, no_pose = (row[k] for k in ("new_pose_estimated", "fallback_used", "no_pose"))
            require(sum(bool(v) for v in (new, fallback, no_pose)) == 1, "pose status not mutually exclusive")
            require(row["pose_available"] == (new or fallback), "availability differs from operational status")
            expected_status = "NEW_POSE" if new else "BASELINE_FALLBACK" if fallback else "NO_POSE"
            require(row["output_status"] == expected_status, "status name differs")
            statuses[row["method"]][expected_status] += 1
            original_points = f["points"][row["coordinate_branch"]]
            equal(row["native_points"][8], original_points[8], 0., "center seed")
            require(bool(row["pose"]["available"]) == bool(row["pose_available"]), "metric availability differs")
            actual = row["actual_pose"] or {}
            hidden = set(row["hidden_initial"])
            require(set(row["excluded"]) == hidden, "fit mask differs from fixed initial hidden IDs")
            require(set(row["reprojected_ids"]) == (hidden if new else set()), "hidden replacement IDs differ")
            require(not (hidden & set(row["solver"].get("fit_input_ids", []))), "hidden coordinate entered final fit")
            if "INITIAL_DIMENSION_PRIOR" in row["method"]:
                require(row["solver"].get("prior_used") is True, "dimension prior absent")
                prior = row["solver"]["initial_dimension_prior"]
                require(prior["rotation_translation_residual_prior"] is False, "unexpected pose residual prior")
                equal(prior["cf_extents"], row["initial_pose"]["cf_extents"], 0., "dimension prior")
                if new:
                    equal(actual["cf_extents"], row["initial_pose"]["cf_extents"], 0., "selected locked dimensions")
                equal(row["input_points"], original_points, 0., "dimension intervention changed coordinates")
            else:
                require(row["coordinate_branch"] == "BASE", "learned fallback branch must be BASE")
                selected = set(row.get("selected_corner_ids", []))
                if not selected:
                    selected = {i for i in range(8) if finite_point(row["input_points"][i])}
                for i in range(8):
                    require(finite_point(row["input_points"][i]) == (i in selected), "learned observation sparsity differs")
            if fallback:
                equal(row["native_points"], original_points, 0., "fallback coordinates")
                require(not row["hidden_reprojected"], "fallback declares a new hidden projection")
            if new:
                require(actual.get("available") is True and len(row["solver"].get("final_inliers", [])) >= 4,
                        "new pose lacks four final inliers")
                require(row["solver"]["residual_threshold_px"] == 8.0 and row["solver"]["refit_count"] <= 3,
                        "solver protocol changed")
                R, t, extents = actual["R_cf"], actual["centroid"], actual["cf_extents"]
                for i in hidden:
                    X = [SIGNS[i][j] * extents[j] / 2 for j in range(3)]
                    camera = [math.fsum(R[a][b] * X[b] for b in range(3)) + t[a] for a in range(3)]
                    image = [math.fsum(row["K"][a][b] * camera[b] for b in range(3)) for a in range(3)]
                    require(image[2] != 0, "zero projection denominator")
                    xy = [image[0] / image[2], image[1] / image[2]]
                    for a in range(2):
                        max_projection_error = max(max_projection_error, abs(xy[a] - row["native_points"][i][a]))
                        require(close(xy[a], row["native_points"][i][a], 1e-8), "hidden final coordinate is not final-pose projection")
                    projection_count += 1
                for i in range(8):
                    if i not in hidden and finite_point(row["input_points"][i]):
                        equal(row["native_points"][i], row["input_points"][i], 0., "selected nonhidden observation")
        require(set(per_arm) == set(self.data["new_methods"]), "raw methods differ from protocol")
        require(all(sorted(v) == ids for v in per_arm.values()), "an arm omitted a frame")
        execution = self.json(self.doc / "CAUSAL_POSE_EXECUTION.json")
        require(execution["complete"] and execution["GT_reads_blocked_before_seal"] and execution["score_phase_after_geometry_seal"], "execution seal contract missing")
        require(execution["new_pose_paths"] == execution["scored_rows"] == 2552, "execution counts differ")
        for key in ("raw_geometry", "scored"):
            binding = execution[key]
            p = self.path(binding["path"])
            require(sha(p) == binding["sha256"] and p.stat().st_size == binding["bytes"], "execution raw binding differs")
        self.data.update(predictions=predictions, new_rows={k: sealed[k] for k in pairs}, statuses=statuses)
        self.counts["public_projection_replays"] += projection_count
        return dict(rows=2552, identical_population_per_arm=True, statuses=dict(statuses),
                    hidden_projection_checks=projection_count, max_absolute_projection_difference_px=max_projection_error,
                    numerical_projection_is_not_physical_GT_verification=True)

    def unlocked(self):
        rows = list(self.rows(self.doc / "UNLOCKED_REPLAY_ROWS.jsonl.gz"))
        methods = {r["method"] for r in rows}
        originals = {(r["method"], r["id"]): r for r in self.rows(self.old / "OBSERVATIONS.jsonl.gz") if r["method"] in methods}
        require(len(rows) == len(originals) == 1276 and len(methods) == 4, "unlocked population differs")
        seen = set()
        keys = ("native_points", "pose_available", "new_pose_estimated", "fallback_used", "no_pose", "hidden_initial", "reprojected_ids", "output_status")
        for row in rows:
            key = row["method"], row["id"]
            require(key not in seen and key in originals, "unlocked duplicate/missing frame")
            seen.add(key)
            original = originals[key]
            for name in keys:
                equal(row[name], original[name], 0., "unlocked " + name)
            for name in ("R_cf", "R_physical", "centroid", "cf_extents", "inliers", "fit_input_ids"):
                equal((row["actual_pose"] or {}).get(name), (original["actual_pose"] or {}).get(name), 0., "unlocked actual " + name)
        self.data["original_unlocked"] = originals
        profiles = list(self.rows(self.doc / "DIMENSION_PROFILE_ROWS.jsonl.gz"))
        require(len(profiles) == len({(r["id"], r["coordinate_branch"], r["mask"]) for r in profiles}) == 1276, "dimension profile population differs")
        for row in profiles:
            require(row["coordinate_branch"] in ("BASE", "N3_SUBPIX") and row["mask"] in ("NO_MASK", "GEOM_NOSELF"), "dimension profile branch differs")
            for candidate in row["pre_refit_best_per_dimension"]:
                if candidate is not None and candidate["candidate_count"]:
                    require(candidate["pre_refit_only"] is True, "profile falsely declares final refit")
            if row["locked_final_cf_extents"] is not None:
                equal(row["locked_final_cf_extents"], row["initial_cf_extents"], 0., "profile locked dimensions")
        return dict(unlocked_replay_rows=1276, numerical_final_output_parity=True, dimension_profile_rows=1276,
                    profile_is_pre_refit_only=True, new_pose_calls_in_this_check=0)

    def metrics(self):
        metric = self.json(self.doc / "METRICS.json")
        wanted = set(metric["methods"])
        lookup = {(r["method"], r["id"]): r for r in self.data["predictions"]}
        for name in ("FIXED_CONTROLS.jsonl.gz", "PREDICTIONS.jsonl.gz", "LEARNED_PREDICTIONS.jsonl.gz"):
            for r in self.rows(self.old / name):
                if r["method"] in wanted:
                    key = r["method"], r["id"]
                    require(key not in lookup, "duplicate metric input identity")
                    lookup[key] = r
        require(len(wanted) == 17 and len(lookup) == 17 * 319, "expected 17 complete metric arms")
        checked_distributions = 0
        for method, stored in metric["methods"].items():
            arm = [lookup[(method, i)] for i in self.data["ids"]]
            require(len({r["session"] for r in arm}) == 13, "metric session population differs")
            for flag in ("pose_available", "new_pose_estimated", "fallback_used", "no_pose", "hidden_reprojected"):
                require(sum(bool(r[flag]) for r in arm) == stored[flag], "operational count differs for " + method + "/" + flag)
            require(stored["total_frames"] == 319, "metric total denominator differs")
            for scope, selected in (("operational", [r for r in arm if r["pose_available"]]),
                                    ("new_pose", [r for r in arm if r["new_pose_estimated"]])):
                for name, (field, scale) in METRICS.items():
                    values = [r["pose"][field] * scale for r in selected]
                    require(all(math.isfinite(v) for v in values), "nonfinite available pose metric")
                    compare_distribution(values, stored["metrics"][scope][name])
                    checked_distributions += 1
        contrasted = 0
        for key, scopes in metric["contrasts"].items():
            a, b = key.split("_minus_", 1)
            require(a in wanted and b in wanted, "contrast has unknown method")
            for scope, stored in scopes.items():
                ids = []
                for identity in self.data["ids"]:
                    ra, rb = lookup[a, identity], lookup[b, identity]
                    keep = ra["pose_available"] and rb["pose_available"]
                    if scope == "candidate_new_pose":
                        keep = keep and ra["new_pose_estimated"]
                    elif scope == "both_new_pose":
                        keep = keep and ra["new_pose_estimated"] and rb["new_pose_estimated"]
                    else:
                        require(scope == "common_operational", "unknown contrast scope")
                    if keep:
                        ids.append(identity)
                require(ids == stored["common_ids"] and stored["denominator"] == 319 and stored["common_frames"] == len(ids), "contrast population differs")
                for name, (field, scale) in METRICS.items():
                    va = [lookup[a, i]["pose"][field] * scale for i in ids]
                    vb = [lookup[b, i]["pose"][field] * scale for i in ids]
                    delta = [x - y for x, y in zip(va, vb)]
                    summary = stored["metrics"][name]
                    compare_distribution(delta, summary)
                    compare_distribution(va, stored["new_marginals"][name])
                    compare_distribution(vb, stored["comparator_marginals"][name])
                    require(summary["improved_frames"] == sum(v < -1e-9 for v in delta), "paired improvement count differs")
                    require(summary["worsened_frames"] == sum(v > 1e-9 for v in delta), "paired worsening count differs")
                    require(summary["unchanged_frames"] == sum(abs(v) <= 1e-9 for v in delta), "paired unchanged count differs")
                    ci = summary["CI95"]
                    require((ci is None and not ids) or (len(ci) == 2 and all(math.isfinite(v) for v in ci) and ci[0] <= ci[1]), "CI shape invalid")
                    checked_distributions += 3
                contrasted += 1
        self.data.update(metric_lookup=lookup, metrics=metric)
        csv_path = self.doc / "METRICS.csv"
        with self.path(csv_path).open(newline="", encoding="utf-8") as stream:
            table = list(csv.DictReader(stream))
        require(len(table) == 102 and len({(r["method"], r["scope"], r["metric"]) for r in table}) == 102, "metric CSV population differs")
        for row in table:
            summary = metric["methods"][row["method"]]["metrics"][row["scope"]][row["metric"]]
            for field, cell in row.items():
                if field in ("method", "scope", "metric"):
                    continue
                expected = summary[field]
                if expected is None:
                    require(cell == "", "metric CSV null cell differs")
                elif isinstance(expected, str):
                    require(cell == expected, "metric CSV unit differs")
                else:
                    require(close(float(cell), expected, 1e-12), "metric CSV numeric cell differs")
        self.counts["raw_pose_metric_distributions"] += checked_distributions
        return dict(methods=17, raw_rows=5423, metric_distributions=checked_distributions,
                    CSV_rows=102, CSV_numeric_and_null_cells_checked=True,
                    contrast_scopes=contrasted, mean_variance_SD_median_P90_recomputed=True,
                    bootstrap_CI_numeric_recomputed=False, physical_GT_recomputed=False)

    def correspondence(self):
        predictions = {(r["method"], r["id"]): r for r in self.data["predictions"]}
        inherited = {(r["method"], r["id"]): r for r in self.rows(self.old / "REAL_CORRESPONDENCE_ROWS.jsonl.gz")
                     if r["method"] in ("BASE_NO_MASK_ROBUST", "N3_SUBPIX_NO_MASK_ROBUST")}
        count, seen = 0, set()
        for row in self.rows(self.doc / "POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz"):
            key = row["method"], row["id"]
            require(key not in seen and key in predictions, "posthoc duplicate or unknown row")
            seen.add(key)
            pred = predictions[key]
            require(row["GT_used_after_geometry_seal"] and row["accuracy_threshold_px"] == 8.0, "posthoc contract differs")
            require(row["pool_ids"] == sorted(pred["solver"]["used"]), "posthoc pool not linked to actual solver used IDs")
            require(row["final_inlier_ids"] == sorted(pred["solver"].get("final_inliers", pred["solver"].get("inliers", []))), "posthoc final inliers not linked to solver")
            require(row["correct_pool_count"] == len(row["correct_pool_ids"]) and row["correct_final_inlier_count"] == len(row["correct_final_inlier_ids"]), "posthoc correct correspondence count differs from IDs")
            branch = pred["coordinate_branch"]
            require(row["frozen_baseline_arm"] == branch, "posthoc baseline branch differs")
            old = inherited[branch + "_NO_MASK_ROBUST", row["id"]]
            for field in ("human_states_native", "permutation_native_to_canonical", "reference_matched"):
                equal(row[field], old[field], 0., "posthoc inherited " + field)
            equal(row["reference_error_initial_native_px"], old["reference_error_input_native_px"], 1e-8, "posthoc frozen baseline corner errors")
            errors = row["reference_error_input_native_px"]
            correct = {i for i, error in enumerate(errors) if error is not None and math.isfinite(error) and error <= 8.0}
            require(sorted(correct) == row["correct_input_native_ids"], "accuracy label arithmetic differs")
            for pool, good, wrong, unknown in (("pool_ids", "correct_pool_ids", "wrong_pool_ids", "unknown_pool_ids"),
                                               ("final_inlier_ids", "correct_final_inlier_ids", "wrong_final_inlier_ids", "unknown_final_inlier_ids")):
                ids = set(row[pool])
                partition = [set(row[k]) for k in (good, wrong, unknown)]
                require(partition[0] == ids & correct, "correct correspondence subset differs")
                require(set.union(*partition) == ids and sum(len(s) for s in partition) == len(ids), "correspondence categories do not partition pool")
                require(partition[2] == {i for i in ids if errors[i] is None}, "unknown reference points misclassified")
            states = row["human_states_native"]
            direct = {i for i, state in enumerate(states) if state == "DIRECT_VISIBLE"}
            self_hidden = {i for i, state in enumerate(states) if state == "SELF_OCCLUDED"}
            known = {i for i, state in enumerate(states) if state != "UNANNOTATED"}
            hidden, pool = set(pred["hidden_initial"]), set(row["pool_ids"])
            require(row["mask_wrong_on_known"] == bool((hidden ^ self_hidden) & known), "posthoc human mask relation differs")
            for field, ids in (("human_direct_pool_ids", direct & pool), ("human_direct_correct_pool_ids", direct & pool & correct),
                               ("false_excluded_direct_ids", hidden & direct), ("false_excluded_accurate_ids", hidden & correct),
                               ("false_retained_human_self_ids", (pool & self_hidden) - hidden)):
                require(row[field] == sorted(ids), "posthoc human state subset differs")
            require(row["initial_hidden_ids"] == sorted(hidden) and row["hidden_reprojected_ids"] == pred["reprojected_ids"], "posthoc hidden IDs differ")
            for flag in ("new_pose_estimated", "fallback_used", "no_pose", "output_status"):
                equal(row[flag], pred[flag], 0., "posthoc " + flag)
            for field in ("translation_cm", "rotation_deg", "ADDsym_m"):
                require(close(row[field], pred["pose"].get(field)), "posthoc pose metric differs")
            prior = self.data["metric_lookup"][branch, row["id"]]["pose"]
            delta = []
            for field, target in (("translation_cm", "translation_delta_cm"), ("rotation_deg", "rotation_delta_deg")):
                value = pred["pose"][field] - prior[field] if pred["pose"]["available"] and prior["available"] else None
                require(close(value, row[target], 1e-10), "posthoc paired delta not linked to frozen baseline")
                delta.append(value)
            outcome = ("no_pose" if not pred["pose_available"] else "fallback" if not pred["new_pose_estimated"] else
                       "both_improved" if all(v < -1e-9 for v in delta) else "both_worsened" if all(v > 1e-9 for v in delta) else "mixed_or_equal")
            require(row["paired_pose_outcome_vs_same_coordinate_initial"] == outcome, "posthoc paired pose outcome differs")
            count += 1
        require(count == 2552, "posthoc frame omitted")
        return dict(rows=count, correspondence_accuracy_partition_recomputed=True,
                    solver_pool_and_inlier_ID_join_checked=True, frozen_human_states_and_baseline_deltas_join_checked=True,
                    stored_reference_error_only=True, physical_reference_error_recomputed=False)

    def bootstrap(self):
        binding = self.json(self.doc / "BOOTSTRAP_BINDING.json")
        public = binding["public_counts"]
        path = self.path(public["path"])
        require(sha(path) == public["sha256"] and path.stat().st_size == public["bytes"], "bootstrap public binding differs")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            draws = json.load(stream)
        sessions = sorted({f["session"] for f in self.data["frames"].values()})
        require(draws["shape"] == [10000, 13] and draws["sessions"] == sessions == binding["sessions"], "bootstrap session ordering/shape differs")
        require(draws["serialized_raw_dtype"] == "uint16_little_endian" and draws["serialized_raw_order"] == "C", "bootstrap byte order differs")
        matrix = draws["counts"]
        require(len(matrix) == 10000 and all(len(row) == 13 and all(type(v) is int and 0 <= v <= 13 for v in row) and sum(row) == 13 for row in matrix), "bootstrap count matrix invalid")
        raw = b"".join(struct.pack("<13H", *row) for row in matrix)
        expected = draws["serialized_raw_sha256"]
        require(hashlib.sha256(raw).hexdigest() == expected == binding["uint16_little_endian_raw_sha256"] == self.data["metrics"]["bootstrap"]["draw_sha256"], "bootstrap draw raw digest differs")
        require(draws["new_draws_generated"] == binding["new_random_draws"] == 0, "unexpected new bootstrap draws")
        lookup = self.data["metric_lookup"]
        session_index = {name: i for i, name in enumerate(sessions)}
        cache = {}
        checked = mathematical = 0
        for key, scopes in self.data["metrics"]["contrasts"].items():
            a, b = key.split("_minus_", 1)
            for scope, stored in scopes.items():
                ids = stored["common_ids"]
                cache_key = a, b, tuple(ids)
                if cache_key not in cache:
                    population = [0] * 13
                    per_session = [[[] for _ in METRICS] for _ in sessions]
                    for identity in ids:
                        j = session_index[lookup[a, identity]["session"]]
                        population[j] += 1
                        for k, (field, scale) in enumerate(METRICS.values()):
                            per_session[j][k].append((lookup[a, identity]["pose"][field] - lookup[b, identity]["pose"][field]) * scale)
                    sums = [[math.fsum(values) for values in group] for group in per_session]
                    means = [[] for _ in METRICS]
                    if ids:
                        for draw in matrix:
                            denominator = sum(n * w for n, w in zip(population, draw))
                            if denominator:
                                for k in range(len(METRICS)):
                                    means[k].append(math.fsum(sums[j][k] * draw[j] for j in range(13)) / denominator)
                        mathematical += len(matrix)
                    cache[cache_key] = means
                means = cache[cache_key]
                require(stored["bootstrap_nonempty_resamples"] == len(means[0]), "bootstrap retained draw denominator differs")
                for k, name in enumerate(METRICS):
                    calculated = [quantile(means[k], .025), quantile(means[k], .975)] if means[k] else None
                    equal(calculated, stored["metrics"][name]["CI95"], 2e-8, "public paired CI95")
                    checked += 1
        self.counts["fixed_bootstrap_resample_mean_replays"] += mathematical
        self.counts["paired_CI95_recomputed"] += checked
        return dict(existing_draws=10000, existing_sessions=13, paired_metric_CIs=checked,
                    frozen_draw_digest=expected, new_random_draws=0,
                    independent_scalar_CI95_recomputed=True)

    def runtime(self):
        runtime = self.json(self.doc / "RUNTIME.json")
        rows = list(self.rows(self.doc / "RUNTIME_ROWS.jsonl.gz"))
        arms = runtime["arms"]
        require(len(arms) == 4 and len(rows) == 600 and runtime["complete"] and runtime["status"] == "DONE", "official runtime not complete")
        require(runtime["frames"] == 26 and runtime["panel_sessions"] == 13 and runtime["warmup_per_arm"] == 20 and runtime["repeats"] == 5 and runtime["measured_per_arm"] == 130, "runtime panel contract differs")
        panel = {r["id"]: r for r in runtime["panel"]}
        require(len(panel) == 26 and Counter(p["session"] for p in panel.values()) == Counter({s: 2 for s in {p["session"] for p in panel.values()}}), "runtime panel requires two frames per session")
        counts = Counter()
        per_arm, measured = defaultdict(Counter), defaultdict(Counter)
        actual_calls = Counter()
        for row in rows:
            require(row["arm"] in arms and row["id"] in panel and row["session"] == panel[row["id"]]["session"], "runtime identity differs")
            phase = row["phase"]
            require(phase in ("warmup", "measured"), "unknown runtime phase")
            per_arm[row["arm"]][phase] += 1
            if phase == "measured":
                measured[row["arm"]][row["id"]] += 1
            require(row["GT_canary_active"] and row["parity_status"] == "PASS" and row["RAW_cached_bitexact"] and row["candidate_center_score_box_preserved"], "runtime canary/parity failed")
            fields = ("detector", "correction", "initial_pose", "robust_and_reprojection")
            require(all(math.isfinite(row[k + "_ms"]) and row[k + "_ms"] >= 0 for k in ("full",) + fields), "invalid runtime interval")
            require(abs(row["full_ms"] - math.fsum(row[k + "_ms"] for k in fields)) < 1e-5, "full pipeline interval differs from contiguous recorded intervals")
            require(row["final_points_parity"]["max_abs_px"] <= row["final_points_parity"]["atol_px"], "runtime coordinate parity exceeds tolerance")
            require(all(v <= row["final_pose_parity"]["atol"] for v in row["final_pose_parity"]["max_abs_by_field"].values()), "runtime pose parity exceeds tolerance")
            require(row["learned_initial_pose_in_correction_interval"] == (row["arm"] == "IMAGE_ROLE_MATCH_MASS"), "learned pose interval contract differs")
            frame = self.data["frames"][row["id"]]
            equal(row["raw_points"], frame["points"]["BASE"], 0., "runtime fresh raw detector parity")
            require(row["selected_index"] == frame["selected_index"], "runtime detector candidate selection differs")
            if row["arm"] in ("BASE", "N3_SUBPIX"):
                expected_points = frame["points"][row["arm"]]
                expected_pose = self.data["original_unlocked"][row["arm"] + "_NO_MASK_ROBUST", row["id"]]["initial_pose"]
            else:
                expected = self.data["new_rows"][row["arm"], row["id"]]
                expected_points, expected_pose = expected["native_points"], expected["actual_pose"]
                require(row["output_status"] == expected["output_status"], "runtime operational status parity differs")
            equal(row["final_points"], expected_points, 1e-7, "runtime final public coordinate parity")
            for field in ("R_cf", "R_physical", "centroid", "cf_extents"):
                equal((row["actual_pose"] or {}).get(field), (expected_pose or {}).get(field), 1e-7, "runtime actual public pose parity")
            actual_calls.update(row["call_counts"])
            counts["detector_calls"] += 1
            counts["initial_pose_calls"] += 1
            counts["robust_solver_calls"] += int(row["arm"] in arms[2:])
            counts["learned_observation_calls"] += int(row["arm"] == "IMAGE_ROLE_MATCH_MASS")
            counts["feature_role_pose_calls"] += int(row["arm"] == "IMAGE_ROLE_MATCH_MASS")
        for arm in arms:
            require(per_arm[arm] == Counter(warmup=20, measured=130), "runtime warmup/measurement count differs")
            require(measured[arm] == Counter({identity: 5 for identity in panel}), "runtime per-frame repeat count differs")
            for stage in ("full", "detector", "correction", "initial_pose", "robust_and_reprojection"):
                values = [r[stage + "_ms"] for r in rows if r["arm"] == arm and r["phase"] == "measured"]
                calculated = distribution(values)
                stored = runtime["summaries"][arm][stage]
                keys = dict(n="n", mean="mean_ms", sample_variance="sample_variance_ms2", sample_std="sample_std_ms", median="median_ms", P90="p90_ms")
                for source, target in keys.items():
                    require(close(calculated[source], stored[target], 1e-8), "runtime raw distribution differs")
        for name, value in (counts + actual_calls).items():
            require(runtime["execution"][name] == value, "runtime recorded call count differs: " + name)
        forwards = runtime["model_forwards"]
        require(forwards["detector"] - forwards["detector_internal_initialization_calls"] == 600 and forwards["fixed_N3"] == 300 and forwards["learned"] == 150 and forwards["duplicate_N3_backbone"] == 0, "runtime forward accounting differs")
        require(not runtime["GT_inference_access"] and not runtime["cached_coordinate_replay_used_for_timing"] and not runtime["other_benchmarks_parallel"], "runtime execution contract differs")
        binding = runtime["raw_rows"]
        p = self.path(binding["path"])
        require(sha(p) == binding["sha256"] and p.stat().st_size == binding["bytes"], "runtime raw binding differs")
        self.counts["runtime_record_distribution_replays"] += 20
        return dict(rows=600, warmups=80, measured_calls=520, frames=26, sessions=13,
                    stored_full_path_statistics_and_call_counts_recomputed=True,
                    timed_inference_repeated=False, new_detector_or_model_calls=0)

    def real_mass(self):
        old = self.rows(self.old / "LEARNED_OBSERVATIONS.jsonl.gz")
        new = self.rows(self.doc / "MATCH_MASS_OBSERVATIONS.jsonl.gz")
        rows = accepted = corners = 0
        identity = set()
        observations = {}
        totals = defaultdict(lambda: defaultdict(Counter))
        for original, row in zip(old, new):
            require(row["original_method"] == original["method"] and row["method"] == original["method"] + "_MATCH_MASS" and row["id"] == original["id"], "real mass row identity differs")
            key = row["method"], row["id"]
            require(key not in identity, "duplicate real mass observation")
            identity.add(key)
            require(row["candidate_logits_shape"] == original["candidate_logits_shape"] == [84, 66], "real logits shape differs")
            raw = base64.b64decode(row["candidate_logits_f32_base64"], validate=True)
            require(row["candidate_logits_f32_base64"] == original["candidate_logits_f32_base64"], "sealed logits altered")
            require(len(raw) == 84 * 66 * 4 and hashlib.sha256(raw).hexdigest() == row["raw_logits_sha256"] == original["raw_logits_sha256"], "real raw logits hash differs")
            scores = struct.unpack("<" + "f" * (84 * 66), raw)
            require(all(math.isfinite(v) for v in scores), "nonfinite real logits")
            for value, hashkey in ((original, "unencoded_semantic_sha256"), (row, "unencoded_semantic_sha256")):
                decoded = {k: v for k, v in value.items() if k not in ("candidate_logits_f32_base64", "candidate_logits_shape", hashkey)}
                decoded["queries"] = [dict(q, candidate_logits=list(scores[i * 66:(i + 1) * 66])) for i, q in enumerate(value["queries"])]
                require(digest(decoded) == value[hashkey], "lossless semantic binding differs")
            require(row["source_unencoded_semantic_sha256"] == original["unencoded_semantic_sha256"], "original semantic reference differs")
            for name in ("original_base_points", "feature_initial_pose", "feature_hidden_initial", "predicted_roles"):
                equal(row[name], original[name], 0., "unchanged mass input " + name)
            require(not row["GT_input"] and not row["new_pose_solver_executed"] and row["raw_logits_unchanged"], "stored-logit replay declared new inference")
            support = defaultdict(list)
            original_support = defaultdict(list)
            for i, (qold, q) in enumerate(zip(original["queries"], row["queries"])):
                s = scores[i * 66:(i + 1) * 66]
                pmatch, conditional, lse, _ = probability(s)
                spatial = max(range(65), key=lambda k: s[k])
                old_choice = max(range(66), key=lambda k: s[k])
                a, b = q["endpoints"]
                seed = row["original_base_points"]
                valid = valid_seed(seed[a]) and valid_seed(seed[b]) and math.dist(seed[a], seed[b]) > 1e-6
                old_choice = old_choice if valid else 65
                choice = spatial if valid and lse > s[65] else 65
                require(q["chosen_candidate"] == choice and q["no_match"] == (choice == 65), "fixed mass decision differs")
                require(qold["chosen_candidate"] == old_choice and qold["no_match"] == (old_choice == 65), "old66 decision differs")
                require(close(q["match_posterior_probability"], pmatch, 1e-12), "stored real Pmatch differs")
                for field in ("query", "edge", "endpoints", "fraction", "center", "normal"):
                    equal(q[field], qold[field], 0., "query geometry " + field)
                if choice < 65:
                    xy = [q["center"][j] + (choice - 32) * q["normal"][j] for j in range(2)]
                    equal(xy, q["selected_xy"], 1e-10, "fixed bin coordinate")
                    support[q["edge"]].append((i, xy))
                    accepted += 1
                else:
                    require(q["selected_xy"] is None, "no-match has a fabricated coordinate")
                if old_choice < 65:
                    original_support[q["edge"]].append((i, qold["selected_xy"]))
                self.counts["real_mass_probability_queries"] += 1
            for value, selected_support in ((original, original_support), (row, support)):
                self.verify_lines_corners(value, selected_support)
            for label, value, selected_support in (("old", original, original_support), ("new", row, support)):
                tally = totals[original["method"]][label]
                hw = self.data["frames"][row["id"]]["raw_hw"]
                tally.update(frames=1, accepted_queries=sum(len(v) for v in selected_support.values()),
                             lines=len(value["lines"]), corners=len(value["corners"]),
                             four_raw_corner_frames=int(len(value["corners"]) >= 4),
                             four_in_image_corner_frames=int(sum(in_frame(c["xy"], hw) for c in value["corners"]) >= 4))
                for edge, samples in selected_support.items():
                    role = value["predicted_roles"][edge].index(max(value["predicted_roles"][edge]))
                    if role == 0:
                        tally["accepted_boundary_queries"] += len(samples)
                    elif role == 1:
                        tally["accepted_internal_queries"] += len(samples)
            observations[key] = {c["id"]: c["xy"] for c in row["corners"]}
            require(row["selected_queries"] == sum(len(v) for v in support.values()), "selected query count differs")
            rows += 1
            corners += len(row["corners"])
        require(rows == len(identity) == 957 and next(old, None) is None and next(new, None) is None, "real mass population differs")
        checks = self.json(self.doc / "MATCH_MASS_DECODE_CHECKS.json")
        for arm in ARMS:
            for label in ("old", "new"):
                expected = {k: totals[arm][label][k] for k in checks["counts"][arm][label]}
                require(expected == checks["counts"][arm][label], "mass aggregate count differs")
        self.data["mass_observations"] = observations
        # Explicit vector negative controls, without a learned model.
        for none, spatial_total, expected_old, expected_new in ((.4, .6, 65, 0), (.7, .3, 65, 65)):
            p = [spatial_total / 2, spatial_total / 2] + [1e-30] * 63 + [none]
            s = [math.log(v) for v in p]
            old_choice = max(range(66), key=lambda k: s[k])
            new_choice = max(range(65), key=lambda k: s[k]) if probability(s)[2] > s[65] else 65
            require((old_choice, new_choice) == (expected_old, expected_new), "mass unit vector failed")
        return dict(rows=rows, queries=rows * 84, accepted_queries=accepted, decoded_corners=corners,
                    old66_and_fixed_mass_query_decisions_recomputed=True,
                    TLS_and_intersections_checked_with_scalar_covariance=True,
                    unit_vectors=2, new_model_or_pose_calls=0)

    def sparse_inputs(self):
        observations = self.data["mass_observations"]
        originals = self.data["original_unlocked"]
        checked = 0
        for row in self.data["predictions"]:
            branch = row["coordinate_branch"]
            mask = "NO_MASK" if row["method"].endswith("NO_MASK") else "GEOM_NOSELF"
            initial = originals[branch + "_" + mask + "_ROBUST", row["id"]]
            equal(row["initial_pose"], initial["initial_pose"], 0., "frozen initial pose provenance")
            equal(row["hidden_initial"], initial["hidden_initial"], 0., "frozen initial hidden mask provenance")
            if "_MATCH_MASS" not in row["method"]:
                continue
            observation_method = row["method"].removesuffix("_NO_MASK")
            corners = observations[observation_method, row["id"]]
            for i in range(8):
                point = row["input_points"][i]
                if i in corners:
                    equal(point, corners[i], 0., "PnP input is not the selected sealed corner")
                else:
                    require(not finite_point(point) and all(v is None for v in point), "unobserved corner filled from Base")
                checked += 1
        return dict(learned_input_coordinate_checks=checked, exact_sealed_observation_link=True,
                    unobserved_inputs_are_null=True, frozen_initial_pose_and_mask_same_coordinate=True)

    def learned_difficulty(self):
        raw = self.rows(self.old / "LEARNED_OBSERVATIONS.jsonl.gz")
        difficulty = self.rows(self.doc / "LEARNED_DIFFICULTY_ROWS.jsonl.gz")
        summary = self.json(self.doc / "LEARNED_DIFFICULTY.json")
        queries, stages, bottlenecks = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter)), defaultdict(Counter)
        count = 0
        for original, row in zip(raw, difficulty):
            require(original["method"] == row["method"] and original["id"] == row["id"] and row["raw_logits_sha256"] == original["raw_logits_sha256"], "learned difficulty raw identity differs")
            hw = self.data["frames"][row["id"]]["raw_hw"]
            logits = struct.unpack("<" + "f" * (84 * 66), base64.b64decode(original["candidate_logits_f32_base64"], validate=True))
            counts = Counter()
            selected = defaultdict(list)
            for i, query in enumerate(row["queries"]):
                source = original["queries"][i]
                scores = logits[i * 66:(i + 1) * 66]
                chosen = max(range(66), key=lambda k: scores[k])
                spatial = max(range(65), key=lambda k: scores[k])
                peak, second = sorted(scores, reverse=True)[:2]
                seed = original["original_base_points"]
                a, b = source["endpoints"]
                valid = valid_seed(seed[a]) and valid_seed(seed[b]) and math.dist(seed[a], seed[b]) > 1e-6
                accepted = valid and chosen < 65
                inside = sum(in_frame([source["center"][j] + offset * source["normal"][j] for j in range(2)], hw) for offset in range(-32, 33))
                expected = dict(valid=valid, raw_argmax=chosen, argmax_is_none=chosen == 65,
                                best_spatial_bin=spatial, best_spatial_offset_px=spatial - 32,
                                best_spatial_logit=scores[spatial], none_logit=scores[65],
                                spatial_minus_none_margin=scores[spatial] - scores[65],
                                winner_runnerup_gap=peak - second, weak_gap_diagnostic=peak - second <= .1,
                                selected=accepted, selected_xy=source["selected_xy"],
                                center_in_image=in_frame(source["center"], hw), candidate_count_in_image=inside,
                                selected_in_image=in_frame(source["selected_xy"], hw) if accepted else None)
                for key, value in expected.items():
                    equal(value, query[key], 1e-10, "learned difficulty query " + key)
                counts.update(queries=1, valid_queries=int(valid), accepted_queries=int(accepted),
                              argmax_none_queries=int(chosen == 65), invalid_queries=int(not valid),
                              centers_out_of_image=int(not expected["center_in_image"]),
                              accepted_out_of_image=int(accepted and not expected["selected_in_image"]),
                              candidate_locations_out_of_image=65 - inside, weak_gap_queries=int(peak - second <= .1))
                if accepted:
                    selected[query["edge"]].append(i)
            require(dict(counts) == row["query_counts"], "learned difficulty query totals differ")
            hidden = set(row["fixed_initial_hidden_ids"])
            corners = {c["id"]: c["xy"] for c in original["corners"]}
            eligible = sorted(i for i, xy in corners.items() if in_frame(xy, hw))
            after = sorted(set(eligible) - hidden)
            require(row["raw_corner_ids"] == sorted(corners) and row["eligible_roi_ids"] == eligible and row["eligible_roi_after_fixed_mask_ids"] == after, "learned difficulty graph/ROI IDs differ")
            expected_counts = dict(accepted_queries=counts["accepted_queries"], edges_with_two_or_more_support_queries=sum(len(v) >= 2 for v in selected.values()),
                                   stored_lines=len(original["lines"]), raw_corners=len(corners),
                                   corners_after_fixed_mask=len(set(corners) - hidden), eligible_roi_corners=len(eligible), eligible_roi_after_fixed_mask=len(after))
            require(row["counts"] == expected_counts, "learned difficulty stage count differs")
            stage = ("BELOW_FOUR_RAW_CORNERS" if len(corners) < 4 else "ROI_BELOW_FOUR" if len(eligible) < 4 else
                     "FIXED_MASK_BELOW_FOUR" if len(after) < 4 else "FOUR_REMAINING_STORED_NEW_POSE" if row["stored_new_pose_estimated"] else "FOUR_REMAINING_STORED_SOLVER_FAILURE")
            require(row["observation_bottleneck"] == stage and not row["new_pose_solver_executed"] and not row["no_match_filled_with_initial_coordinates"], "learned difficulty bottleneck differs")
            queries[row["method"]].update(counts)
            bottlenecks[row["method"]][stage] += 1
            for key, value in expected_counts.items():
                stages[row["method"]][key][str(value)] += 1
            count += 1
        require(count == 957 and next(raw, None) is None and next(difficulty, None) is None, "learned difficulty population differs")
        for arm in ARMS:
            require(dict(queries[arm]) == summary["methods"][arm]["query_counts"] and dict(bottlenecks[arm]) == summary["methods"][arm]["bottleneck_counts"], "learned difficulty method aggregate differs")
            for key, histogram in stages[arm].items():
                require(dict(histogram) == summary["methods"][arm]["stage_count_histograms"][key], "learned difficulty stage histogram differs")
        self.counts["learned_difficulty_query_replays"] += count * 84
        return dict(rows=count, raw_queries=count * 84, argmax_margin_ROI_graph_mask_and_stage_totals_recomputed=True,
                    descriptive_thresholds_are_not_selection_rules=True, new_pose_calls=0)

    def verify_lines_corners(self, value, supports):
        line_edges = set()
        lines = {}
        for line in value["lines"]:
            edge = line["edge"]
            require(edge not in line_edges and edge in supports and len(supports[edge]) >= 2, "line lacks two independent query locations")
            line_edges.add(edge)
            samples = supports[edge]
            require(line["queries"] == [i for i, _ in samples], "line support query identity differs")
            pts = [point for _, point in samples]
            equal(pts, line["support_points"], 1e-10, "line support coordinate")
            n = line["normal"]
            require(close(math.fsum(x * x for x in n), 1., 1e-9), "TLS normal not unit length")
            center = [math.fsum(p[j] for p in pts) / len(pts) for j in range(2)]
            require(close(line["offset"], math.fsum(center[j] * n[j] for j in range(2)), 1e-9), "TLS offset differs")
            xx = math.fsum((p[0] - center[0]) ** 2 for p in pts)
            xy = math.fsum((p[0] - center[0]) * (p[1] - center[1]) for p in pts)
            yy = math.fsum((p[1] - center[1]) ** 2 for p in pts)
            eigen_min = ((xx + yy) - math.hypot(xx - yy, 2 * xy)) / 2
            residual_sse = math.fsum((n[0] * p[0] + n[1] * p[1] - line["offset"]) ** 2 for p in pts)
            require(abs(residual_sse - max(0., eigen_min)) <= 1e-6 * max(1., xx + yy), "normal is not a TLS minimizer")
            require(close(line["residual_rms_px"], math.sqrt(residual_sse / len(pts)), 1e-7), "TLS RMS differs")
            tangent = [-n[1], n[0]]
            along = [math.fsum(tangent[j] * p[j] for j in range(2)) for p in pts]
            require(close(line["support_length_px"], max(along) - min(along), 1e-8), "line support length differs")
            lines[edge] = line
        expected_edges = set()
        for edge, samples in supports.items():
            if len(samples) >= 2:
                pts = [p for _, p in samples]
                cx = math.fsum(p[0] for p in pts) / len(pts)
                cy = math.fsum(p[1] for p in pts) / len(pts)
                xx = math.fsum((p[0] - cx) ** 2 for p in pts)
                xy = math.fsum((p[0] - cx) * (p[1] - cy) for p in pts)
                yy = math.fsum((p[1] - cy) ** 2 for p in pts)
                eigen_max = ((xx + yy) + math.hypot(xx - yy, 2 * xy)) / 2
                if math.sqrt(max(0., eigen_max)) >= 1e-6:
                    expected_edges.add(edge)
        require(line_edges == expected_edges, "line generation eligibility differs")
        expected_corners = {}
        for corner in range(8):
            incident = [line for edge, line in lines.items() if corner in EDGES[edge]]
            pairs = []
            for i, first in enumerate(incident):
                for second in incident[i + 1:]:
                    dot = abs(math.fsum(first["normal"][j] * second["normal"][j] for j in range(2)))
                    pairs.append((dot, first, second))
            if not pairs:
                continue
            _, first, second = min(pairs, key=lambda p: p[0])
            a, b = first["normal"], second["normal"]
            det = a[0] * b[1] - a[1] * b[0]
            if abs(det) <= 1e-6:
                continue
            c, d = first["offset"], second["offset"]
            point = [(c * b[1] - a[1] * d) / det, (a[0] * d - c * b[0]) / det]
            expected_corners[corner] = (point, [first["edge"], second["edge"]])
        require({c["id"] for c in value["corners"]} == set(expected_corners), "decoded corner identities differ")
        for corner in value["corners"]:
            expected, edges = expected_corners[corner["id"]]
            equal(corner["xy"], expected, 1e-7, "line intersection")
            require(corner["edges"] == edges, "corner chosen incident pair differs")

    def source(self):
        ceiling = {r["index"]: r for r in self.rows(self.doc / "SOURCE_CEILING_ROWS.jsonl.gz")}
        require(len(ceiling) == 1024 and set(ceiling) == set(range(1024)), "source ceiling population differs")
        family_partition, target_counts = {}, Counter()
        for row in ceiling.values():
            family = row["family"]
            require(family not in family_partition or family_partition[family] == row["partition"], "family crosses partitions")
            family_partition[family] = row["partition"]
            counts = Counter()
            by_role = defaultdict(Counter)
            for i in range(84):
                target = row["targets"]
                lo, hi, weight, valid = (target[k][i] for k in ("lo", "hi", "weight", "valid"))
                label = "ignore" if not valid else "no_match" if lo == 65 else "positive"
                if label == "positive":
                    require(0 <= lo <= hi < 65 and 0 <= weight <= 1, "positive interpolation target invalid")
                else:
                    require(lo == hi == 65 and weight == 0, "none/ignore placeholder differs")
                counts[label] += 1
                by_role[ROLES[row["predicted_role_query_ids"][i]]][label] += 1
            require(dict(counts) == row["target_counts"], "source target count differs")
            for role in ROLES:
                require({k: by_role[role][k] for k in ("positive", "no_match", "ignore")} == row["target_counts_by_predicted_role"][role], "source role target count differs")
            target_counts.update(counts)
            require(not row["source_pose_fitted"] and row["predicted_initial_H"] is None, "source diagnostic declares extra pose/mask inference")
            targets = row["targets"]
            ideal = [targets["lo"][i] if targets["weight"][i] <= .5 else targets["hi"][i] for i in range(84)]
            ideal = [bin_id if targets["valid"][i] else 65 for i, bin_id in enumerate(ideal)]
            require(ideal == row["ideal_choices"], "supplied target ceiling choices differ")
            lines, corners = scalar_decode(row["frozen_selected_points"], ideal)
            require(len(lines) == row["ideal"]["line_count"] and len(corners) == row["ideal"]["corner_count"], "source ideal graph count differs")
            for item in row["ideal"]["corners"]:
                equal(item["xy"], corners[item["id"]], 1e-7, "supplied target ceiling corner")
            inside = sorted(i for i, xy in corners.items() if in_frame(xy, row["raw_hw"]))
            hidden = set(row["source_annotation_oracle_H"])
            require(inside == row["before_mask_in_frame_corner_ids"] and sorted(set(corners) - hidden) == row["after_source_annotation_oracle_H_corner_ids"] and sorted(set(inside) - hidden) == row["after_source_annotation_oracle_H_in_frame_corner_ids"], "source ideal mask/ROI graph differs")
        ceiling_summary = self.json(self.doc / "SOURCE_CEILING.json")["summary"]
        for partition in ("all", "train", "calibration", "source_test"):
            rows = [r for r in ceiling.values() if partition == "all" or r["partition"] == partition]
            stored = ceiling_summary[partition]
            require(stored["families"] == len(rows), "source ceiling summary population differs")
            require(stored["target_counts"] == {k: sum(r["target_counts"][k] for r in rows) for k in ("positive", "no_match", "ignore")}, "source ceiling aggregate targets differ")
            for field, count in (("ideal_before_mask_ge4", sum(r["ideal"]["corner_count"] >= 4 for r in rows)),
                                 ("ideal_before_mask_in_frame_ge4", sum(len(r["before_mask_in_frame_corner_ids"]) >= 4 for r in rows)),
                                 ("after_source_annotation_oracle_H_in_frame_ge4", sum(len(r["after_source_annotation_oracle_H_in_frame_corner_ids"]) >= 4 for r in rows))):
                require(stored[field] == count, "source ceiling four-corner summary differs")
        self.counts["source_scalar_graph_assemblies"] += 1024
        groups, decisions = defaultdict(lambda: defaultdict(list)), defaultdict(Counter)
        label_graph = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
        count = positive = 0
        identities = set()
        for row in self.rows(self.doc / "SOURCE_MATCH_MASS_LOGITS.jsonl.gz"):
            key = row["arm"], row["partition"], row["index"]
            require(key not in identities and row["arm"] in ARMS and row["partition"] in ("calibration", "source_test"), "source logit identity invalid")
            identities.add(key)
            truth = ceiling[row["index"]]
            require(row["id"] == truth["id"] and row["family"] == truth["family"] and row["partition"] == truth["partition"], "source target identity differs")
            require(row["source_ceiling_row"] == dict(file="SOURCE_CEILING_ROWS.jsonl.gz", line=row["index"] + 1, sha256=digest(truth)), "source semantic row hash differs")
            raw = zlib.decompress(base64.b64decode(row["logits_f32_zlib_base64"], validate=True))
            require(len(raw) == 84 * 66 * 4 and row["logits_shape"] == [84, 66] and hashlib.sha256(raw).hexdigest() == row["logits_uncompressed_sha256"], "source raw logits binding differs")
            logits = struct.unpack("<" + "f" * (84 * 66), raw)
            require(all(math.isfinite(v) for v in logits), "source nonfinite logit")
            per_decoder_counts = {name: Counter() for name in row["decoders"]}
            for i in range(84):
                s = logits[i * 66:(i + 1) * 66]
                p, conditional, _, odds = probability(s)
                require(close(p, row["Pmatch"][i], 1e-12) and close(odds, row["log_match_mass_minus_none"][i], 1e-12), "source Pmatch differs")
                lo, hi, weight, valid = (truth["targets"][k][i] for k in ("lo", "hi", "weight", "valid"))
                label = "IGNORE" if not valid else "NONE" if lo == 65 else "POSITIVE"
                role = ROLES[truth["predicted_role_query_ids"][i]]
                group = row["partition"], row["arm"], label, role
                entropy = -math.fsum(x * math.log(x) for x in conditional if x)
                fields = dict(Pmatch=p, conditional_candidate_entropy_nats=entropy,
                              conditional_candidate_entropy_bits=entropy / math.log(2),
                              conditional_effective_candidates=math.exp(entropy),
                              maximum_conditional_candidate_probability=max(conditional), none_probability=1 - p)
                if label == "POSITIVE":
                    mass = math.fsum(conditional[b] for b in {lo, hi})
                    weighted = (1 - weight) * conditional[lo] + weight * conditional[hi]
                    fields.update(two_target_bin_joint_probability=p * mass,
                                  two_target_bin_conditional_match_probability=mass,
                                  supplied_weighted_target_joint_probability=p * weighted,
                                  supplied_weighted_target_conditional_match_probability=weighted)
                    positive += 1
                for field, value in fields.items():
                    groups[group][field].append(value)
                seed = truth["frozen_selected_points"]
                a, b = EDGES[i // 7]
                query_valid = valid_seed(seed[a]) and valid_seed(seed[b]) and math.dist(seed[a], seed[b]) > 1e-6
                original = max(range(66), key=lambda k: s[k]) if query_valid else 65
                mass_choice = max(range(65), key=lambda k: s[k]) if query_valid and odds > 0 else 65
                decisions[group].update(queries=1, old_66way_match=int(original < 65),
                                        fixed_mass_match=int(mass_choice < 65), old_none_mass_match=int(original == 65 and mass_choice < 65))
                for name, expected in (("ORIGINAL_66_CLASS_MAP", original), ("FIXED_BINARY_EXISTENCE_MAP", mass_choice)):
                    require(row["decoders"][name]["choices"][i] == expected, "source decoder choice differs")
                    per_decoder_counts[name].update(selected_queries=int(expected < 65), positive=int(label == "POSITIVE"),
                                                    positive_accepted=int(label == "POSITIVE" and expected < 65),
                                                    no_match=int(label == "NONE"), no_match_false_accepted=int(label == "NONE" and expected < 65),
                                                    ignored=int(label == "IGNORE"), ignored_accepted_without_truth=int(label == "IGNORE" and expected < 65))
                self.counts["source_probability_queries"] += 1
            for name, counts in per_decoder_counts.items():
                for field, expected in counts.items():
                    require(row["decoders"][name][field] == expected, "source decoder label count differs")
                choice = row["decoders"][name]["choices"]
                line, corner = scalar_decode(truth["frozen_selected_points"], choice)
                require(sorted(line) == row["decoders"][name]["decoded_line_edges"] and sorted(corner) == row["decoders"][name]["decoded_corner_ids"], "source decoder graph differs")
                for item in row["decoders"][name]["decoded_corners"]:
                    equal(item["xy"], corner[item["id"]], 1e-7, "source decoded corner")
                positive_only = [bin_id if truth["targets"]["valid"][i] and truth["targets"]["lo"][i] < 65 else 65 for i, bin_id in enumerate(choice)]
                _, corners = scalar_decode(truth["frozen_selected_points"], positive_only)
                label_graph[row["arm"]][row["partition"]][name].update(frames=1, positive_only_ge4=int(len(corners) >= 4), positive_only_corners_total=len(corners), **{"corners_" + str(len(corners)): 1})
            count += 1
        require(count == len(identities) == 768, "source logits population differs")
        stored = self.json(self.doc / "SOURCE_LOGIT_DIFFICULTY.json")
        require(stored["source_logit_rows"] == 768 and stored["probability_query_replays"] == 64512 and stored["positive_query_target_probability_replays"] == positive, "source probability execution count differs")
        require(len(stored["groups"]) == len(groups) == 36, "source logit group population differs")
        for group in stored["groups"]:
            key = group["partition"], group["arm"], group["supplied_label"], group["predicted_role"]
            require(key in groups and dict(decisions[key]) == group["counts"], "source group decision count differs")
            require(set(groups[key]) == set(group["distributions"]), "source distribution fields differ")
            for field, values in groups[key].items():
                compare_distribution(values, group["distributions"][field], True)
        graph = self.json(self.doc / "SOURCE_MATCH_MASS_LABEL_GRAPH.json")
        for arm in ARMS:
            for partition in ("calibration", "source_test"):
                for name in ("ORIGINAL_66_CLASS_MAP", "FIXED_BINARY_EXISTENCE_MAP"):
                    require(dict(label_graph[arm][partition][name]) == graph["summary"][arm][partition][name], "positive-only source graph aggregate differs")
        self.counts["source_scalar_graph_assemblies"] += 768 * 4
        self.data["ceiling"] = ceiling
        return dict(source_target_rows=1024, source_logit_rows=768, probability_queries=64512,
                    supplied_positive_targets=positive, groups=36, target_counts=dict(target_counts),
                    family_split_disjoint=True, independent_probability_entropy_and_two_bin_mass=True,
                    full_and_positive_only_source_graphs_recomputed=True,
                    labels_not_revalidated_from_private_mesh=True, new_head_forwards=0)

    def ray(self):
        summary = self.json(self.doc / "SOURCE_RAY_VALIDATION.json")["summary"]
        ceiling = self.data["ceiling"]
        count = rays = eligible = rescues = mismatch = closest = 0
        cached, reproduced, role_rescue, edge_rescue = Counter(), Counter(), Counter(), Counter()
        positive_errors, rescued_errors = [], []
        seen = set()
        for row in self.rows(self.doc / "SOURCE_RAY_VALIDATION_ROWS.jsonl.gz"):
            require(row["index"] not in seen and row["partition"] == "source_test", "ray diagnostic identity differs")
            seen.add(row["index"])
            require(digest(ceiling[row["index"]]) == row["source_ceiling_semantic_sha256"], "ray source ceiling row binding differs")
            tolerance = row["depth_tolerance_m"]
            require(close(tolerance, row["physical_diagonal_m"] * .001, 1e-12), "ray depth tolerance differs")
            require(len(row["queries"]) == 84, "ray row missing a query")
            require([q["query"] for q in row["queries"]] == list(range(84)), "ray query ID ordering differs")
            for q in row["queries"]:
                target = ceiling[row["index"]]["targets"]
                i = q["query"]
                expected_target = "IGNORE" if not target["valid"][i] else "NONE" if target["lo"][i] == 65 else "POSITIVE"
                require(q["cached_target"] == expected_target and q["edge"] == i // 7, "ray cached target/query geometry identity differs")
                cached[q["cached_target"]] += 1
                reproduced[q["reproduced_target"]] += 1
                mismatch += int(q["cached_target"] != q["reproduced_target"])
                if q["ray_eligible"]:
                    eligible += 1
                    closest += 1
                    require(close(q["source_camera_Z"], math.fsum(row["R"][2][j] * q["source_X"][j] for j in range(3)) + row["t"][2], 1e-12), "ray source point camera depth differs")
                    require(len(q["ray_depths_camera_Z"]) == len(q["ray_depth_minus_source_Z_m"]) == 3, "ray offset variants differ")
                    rays += 3
                    agreements = []
                    for depth, delta in zip(q["ray_depths_camera_Z"], q["ray_depth_minus_source_Z_m"]):
                        if depth is None:
                            require(delta is None, "infinite ray has finite delta")
                            agreements.append(False)
                        else:
                            require(close(delta, depth - q["source_camera_Z"], 1e-12), "stored ray depth subtraction differs")
                            agreements.append(abs(delta) <= tolerance)
                    require(agreements == q["ray_source_depth_agreement"], "ray depth agreement differs")
                    if q["cached_target"] == "POSITIVE" and q["ray_depth_minus_source_Z_m"][0] is not None:
                        positive_errors.append(abs(q["ray_depth_minus_source_Z_m"][0]))
                    rescue = (not agreements[0]) and any(agreements[1:])
                    require(rescue == q["fixed_normal_offset_rescues_source_depth"], "fixed ray rescue arithmetic differs")
                    strong = rescue and q["cached_target"] == "NONE" and q["actual_mesh_point_within_original_tolerance"] and q["in_image"] and q["supplied_visible_mask_support_3x3"]
                    require(strong == q["mesh_and_mask_supported_original_miss_rescued"], "mesh/mask rescue criterion differs")
                    if strong:
                        rescues += 1
                        role_rescue[str(q["predicted_role"])] += 1
                        edge_rescue[str(q["edge"])] += 1
                        rescued_errors.append(min(abs(x) for x in q["ray_depth_minus_source_Z_m"][1:] if x is not None and abs(x) < tolerance))
                else:
                    require(not q.get("fixed_normal_offset_rescues_source_depth", False), "ineligible query declares ray rescue")
            count += 1
        require(count == 128 and len(seen) == 128, "ray source population differs")
        expected = dict(families=count, queries=count * 84, rays=rays, ray_eligible_queries=eligible,
                        source_query_closest_points=closest, label_mismatches=mismatch,
                        cached_NONE_fixed_offset_rescued=rescues, cached_targets=dict(cached), reproduced_targets=dict(reproduced),
                        cached_NONE_rescued_by_predicted_role=dict(role_rescue), cached_NONE_rescued_by_physical_edge=dict(edge_rescue))
        for key, value in expected.items():
            equal(value, summary[key], 0., "ray summary " + key)
        for values, stored in ((positive_errors, summary["original_positive_depth_error_m"]), (rescued_errors, summary["rescued_offset_abs_depth_error_m"])):
            for key, expected_value in dict(n=len(values), mean=math.fsum(values) / len(values), median=quantile(values, .5), P90=quantile(values, .9), P99=quantile(values, .99), max=max(values)).items():
                require(close(expected_value, stored[key], 1e-8), "ray error quantile differs")
        self.counts["stored_ray_result_classifications"] += count * 84
        return dict(rows=128, stored_ray_results=rays, stored_classifications_recomputed=True,
                    mesh_ray_execution_repeated=False, private_mesh_unavailable_to_this_verifier=True,
                    fixed_offset_rescues=rescues, target_labels_unchanged=True)

    def manifest(self, required):
        path = self.doc / "REVIEW_MANIFEST.json"
        if not path.exists():
            require(not required, "--require-manifest: REVIEW_MANIFEST.json missing")
            return dict(present=False, required=False, hashes_checked=0)
        manifest = self.json(path)
        entries = manifest["files"]
        require(len(entries) == len({r["path"] for r in entries}), "duplicate manifest path")
        hashes = {r["path"]: r for r in entries}
        manifest_relative = str(path.relative_to(self.root))
        check_relative = str((self.doc / "REVIEW_CHECKS.json").relative_to(self.root))
        require(manifest_relative not in hashes and check_relative not in hashes, "manifest/check hash cycle")
        for row in entries:
            p = self.path(row["path"])
            require(p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], "review manifest binding differs: " + row["path"])
        for name in self.inputs:
            if "/" + NAME + "/" in name and name not in (manifest_relative, check_relative):
                require(name in hashes, "used new public input absent from manifest: " + name)
        return dict(present=True, hashes_checked=len(entries), all_used_new_inputs_bound=True,
                    signed_provenance_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3], help="repository root or this experiment's _docs directory")
    parser.add_argument("--output", type=Path, help="default: new experiment REVIEW_CHECKS.json; existing research files protected")
    parser.add_argument("--require-manifest", action="store_true", help="fail when the new review manifest is absent")
    args = parser.parse_args()
    root = args.root.resolve()
    if root.name == NAME and root.parent.name == "experiments" and root.parent.parent.name == "_docs":
        root = root.parents[2]
    doc = root / "_docs" / "experiments" / NAME
    nominal_default = doc / "REVIEW_CHECKS.json"
    nominal_output = args.output if args.output else nominal_default
    require(not nominal_default.is_symlink() and not nominal_output.is_symlink(), "refusing a symlink output target")
    default_output = nominal_default.resolve()
    require(default_output.is_relative_to(root), "default output directory escapes repository root")
    output = nominal_output.resolve()
    script = Path(__file__).resolve()
    # Protect every repository input/code/artifact, including files not yet read.
    # Only this verifier's own new check receipt may be replaced in the repo.
    require(output != script and output.name not in ("REVIEW_MANIFEST.json", "PRIOR_PUBLICATION_BINDINGS.json"), "protected output path")
    require(output.suffix == ".json", "verification receipt must have .json extension")
    require(output == default_output or not output.exists(), "only this verifier's own default receipt may be overwritten")
    if output.exists():
        require(output.is_file() and not output.is_symlink(), "unsafe output target")
    review = Review(root, output)
    start = time.perf_counter()
    for name, function in (("original_111_publication_bindings", review.prior),
                           ("fixed_319_frame_population", review.load_population),
                           ("sealed_pose_status_and_projection_contract", review.raw_pose_contract),
                           ("original_unlocked_output_parity_and_dimension_profile", review.unlocked),
                           ("raw_pose_statistics_and_paired_populations", review.metrics),
                           ("frozen_public_bootstrap_paired_CI95", review.bootstrap),
                           ("posthoc_correspondence_partitions", review.correspondence),
                           ("real_stored_logit_match_mass_and_TLS", review.real_mass),
                           ("learned_sparse_input_and_fixed_initial_provenance", review.sparse_inputs),
                           ("learned_difficulty_raw_argmax_ROI_graph_counts", review.learned_difficulty),
                           ("source_labels_and_logit_probability_statistics", review.source),
                           ("stored_actual_mesh_ray_result_arithmetic", review.ray),
                           ("actual_runtime_public_record_arithmetic", review.runtime)):
        review.check(name, function)
    review.check("publication_manifest", lambda: review.manifest(args.require_manifest))
    passed = all(c["passed"] for c in review.checks)
    result = dict(schema="kp_difficulty_public_review_checks_v1", generated_utc=datetime.now(timezone.utc).isoformat(),
                  complete=True, passed=passed, fresh_public_arithmetic=True,
                  python_minimum="3.9", python_version=sys.version.split()[0],
                  checks=review.checks, inputs=list(review.inputs.values()),
                  implementation=dict(path=str(script.relative_to(root)) if script.is_relative_to(root) else script.name, sha256=sha(script), bytes=script.stat().st_size),
                  execution=dict(wall_seconds=time.perf_counter() - start, mathematical_replays=dict(review.counts),
                                 new_detector_forwards=0, new_head_forwards=0, GPU_calls=0, new_PnP_calls=0,
                                 new_training_updates=0, actual_mesh_ray_calls=0),
                  limits=["Physical ground truth, private mesh rays, model execution and wall-clock timings are not independently rerun.",
                          "Pose metrics and correspondence errors are public stored reference measurements; arithmetic is recomputed.",
                          "Paired CI95 values are reproduced from the public frozen draw counts, with no new random draws.",
                          "Hash bindings establish internal file consistency, not signed provenance or the truth of an execution claim."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=passed, checks=len(review.checks), failed=[c["name"] for c in review.checks if not c["passed"]], output=str(output)), ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        sys.exit(1)
