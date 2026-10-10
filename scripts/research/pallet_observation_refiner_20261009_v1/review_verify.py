#!/usr/bin/env python3
"""Recompute checks from the public research bundle with Python's standard library.

This is an artifact-consistency verifier, not a detector/training replay or an
independent measurement of private physical ground truth. It never imports the
experiment package, reads private dependencies, or trusts saved verification
results. Exit status 0 means all available required checks passed; 1 means failure.
"""
import argparse
import base64
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


EXPERIMENT = "pallet_observation_refiner_20261009_v1"
METHODS = (
    "BASE", "SUBPIX", "N3", "N3_SUBPIX",
    "BASE_NO_MASK_STANDARD", "BASE_NO_MASK_ROBUST",
    "BASE_GEOM_NOSELF_STANDARD", "BASE_GEOM_NOSELF_ROBUST",
    "BASE_ORACLE_NOSELF_ROBUST", "BASE_ORACLE_VISIBLE_ROBUST",
    "N3_SUBPIX_NO_MASK_STANDARD", "N3_SUBPIX_NO_MASK_ROBUST",
    "N3_SUBPIX_GEOM_NOSELF_STANDARD", "N3_SUBPIX_GEOM_NOSELF_ROBUST",
    "N3_SUBPIX_ORACLE_NOSELF_ROBUST", "N3_SUBPIX_ORACLE_VISIBLE_ROBUST",
    "SHARED_BOUNDARY_GEOM_ROBUST", "GEOMETRY_ONLY", "IMAGE_NO_ROLE", "IMAGE_ROLE",
    "IMAGE_ROLE_NO_MASK_ROBUST", "IMAGE_ROLE_STANDARD", "IMAGE_ROLE_POINT_LINE",
)
LEARNED = ("GEOMETRY_ONLY", "IMAGE_NO_ROLE", "IMAGE_ROLE")
ARMS = ("BASE", "N3_SUBPIX", "N3_SUBPIX_GEOM_NOSELF_ROBUST", "IMAGE_ROLE")
FIELDS = {"translation_cm": ("translation_cm", 1., "cm"),
          "rotation_deg": ("rotation_deg", 1., "degree"),
          "ADDsym_cm": ("ADDsym_m", 100., "cm")}
STAGES = ("full", "detector", "correction", "initial_pose", "robust_and_reprojection")
STAT_KEYS = ("n", "mean", "sample_variance", "sample_std", "median", "P90", "max")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a, b, atol=1e-8, rtol=1e-10):
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, atol, rtol) for x, y in zip(a, b))
    return a == b


def distribution(values):
    values = sorted(values)
    n = len(values)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("nonfinite metric")
    mean = math.fsum(values) / n if n else None
    variance = math.fsum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else None
    def quantile(p):
        if not n:
            return None
        x = (n - 1) * p
        i = math.floor(x)
        return values[i] + (values[min(i + 1, n - 1)] - values[i]) * (x - i)
    return dict(n=n, mean=mean, sample_variance=variance,
                sample_std=math.sqrt(variance) if variance is not None else None,
                median=quantile(.5), P90=quantile(.9), max=values[-1] if n else None)


def pose_parameters(pose):
    if not pose or not pose.get("available"):
        return None
    keys = ("R_cf", "R_physical", "centroid", "cf_extents", "selected_hypothesis")
    return {key: pose.get(key) for key in keys}


def project(pose, K):
    w, h, d = pose["cf_extents"]
    a, b, c = w / 2, h / 2, d / 2
    X = ((-a, -b, -c), (a, -b, -c), (a, b, -c), (-a, b, -c),
         (-a, -b, c), (a, -b, c), (a, b, c), (-a, b, c))
    output = []
    for point in X:
        camera = [math.fsum(pose["R_cf"][i][j] * point[j] for j in range(3))
                  + pose["centroid"][i] for i in range(3)]
        pixel = [math.fsum(K[i][j] * camera[j] for j in range(3)) for i in range(3)]
        if camera[2] <= 0 or abs(pixel[2]) < 1e-12:
            raise ValueError("nonpositive corner depth or degenerate projection")
        output.append([pixel[0] / pixel[2], pixel[1] / pixel[2]])
    return output


class Review:
    def __init__(self, repo, doc):
        self.repo, self.doc = repo, doc
        self.checks, self.problems, self.inputs = [], [], {}
        self.issue_counts = Counter()

    def require(self, condition, group, detail):
        if not condition:
            self.issue_counts[group] += 1
            if self.issue_counts[group] <= 20:
                self.problems.append(dict(check=group, detail=detail))

    def check(self, name, **details):
        n = self.issue_counts[name]
        self.checks.append(dict(name=name, status="FAIL" if n else "PASS",
                                details=dict(details, failures=n)))

    def register(self, path):
        path = path.resolve()
        key = str(path.relative_to(self.repo))
        if key not in self.inputs:
            self.inputs[key] = dict(path=key, sha256=sha(path), bytes=path.stat().st_size)
        return path

    def load(self, name):
        return json.loads(self.register(self.doc / name).read_text(encoding="utf-8"))

    def rows(self, name):
        with gzip.open(self.register(self.doc / name), "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"{name}:{line_number}: empty raw row")
                yield json.loads(line)

    def run(self, frame_id=None, require_manifest=False):
        inputs = self.load("INPUTS.json")
        frames = {r["id"]: r for r in inputs["frames"]}
        ids = set(frames)
        sessions = Counter(r["session"] for r in frames.values())
        protocol = self.load("PROTOCOL.json")["population"]
        self.require(len(inputs["frames"]) == len(ids) == 319 and len(sessions) == 13,
                     "population", "INPUTS must contain 319 unique IDs and 13 sessions")
        self.require(protocol["frames"] == 319 and protocol["sessions"] == 13
                     and len(protocol["ids"]) == 319 and set(protocol["ids"]) == ids,
                     "population", "PROTOCOL population differs from INPUTS")
        self.require(inputs["GT_input"] is False, "population", "INPUTS GT_input flag")
        if frame_id:
            self.require(frame_id in ids, "population", f"unknown frame ID: {frame_id}")
        self.check("population", frames=len(ids), session_counts=dict(sessions))

        rows = defaultdict(dict)
        raw_counts = {}
        projection_corners = projection_frames = missing_historical_pose = 0
        for filename, expected_rows in (("FIXED_CONTROLS.jsonl.gz", 1276),
                                       ("PREDICTIONS.jsonl.gz", 4147),
                                       ("LEARNED_PREDICTIONS.jsonl.gz", 1914)):
            count = 0
            for row in self.rows(filename):
                count += 1
                method, fid = row["method"], row["id"]
                label = f"{method}/{fid}"
                self.require(method in METHODS and fid in ids and fid not in rows[method],
                             "raw_identity", label + " duplicate or unknown identity")
                if fid not in frames:
                    continue
                f = frames[fid]
                self.require(row["session"] == f["session"], "raw_identity", label + " session")
                available = row["pose"].get("available", False)
                self.require(type(available) is bool and row["pose_available"] == available
                             and row["no_pose"] == (not available), "output_accounting", label + " availability flags")
                for key in ("new_pose_estimated", "fallback_used", "hidden_reprojected"):
                    self.require(type(row[key]) is bool, "output_accounting", label + "/" + key)
                if filename != "FIXED_CONTROLS.jsonl.gz":
                    new, fallback = row["new_pose_estimated"], row["fallback_used"]
                    self.require(sum((new, fallback, not available)) == 1 and
                                 row["output_status"] == ("NEW_POSE" if new else "BASELINE_FALLBACK" if fallback else "NO_POSE"),
                                 "output_accounting", label + " status partition")
                    self.require(close(row["K"], f["K"]) and close(row["xyz"], f["xyz"])
                                 and row["raw_hw"] == f["raw_hw"] and row["selected_index"] == f["selected_index"],
                                 "geometry_contract", label + " input calibration/dimensions/selection")
                    hidden = set(row["hidden_initial"])
                    excluded = set(row["excluded"])
                    self.require(hidden <= excluded and all(type(i) is int and 0 <= i < 8 for i in hidden | excluded),
                                 "geometry_contract", label + " hidden IDs not excluded")
                    self.require(row["reprojections_reused_as_observations"] is False,
                                 "geometry_contract", label + " reprojections reused")
                    expected_reprojection = hidden if new else set()
                    self.require(set(row["reprojected_ids"]) == expected_reprojection
                                 and row["hidden_reprojected"] == bool(expected_reprojection),
                                 "geometry_contract", label + " hidden replacement flags")
                    if new:
                        solver = row["solver"]
                        fit_ids = set(solver.get("fit_input_ids", [])) | set(solver.get("used", []))
                        self.require(not (fit_ids & excluded), "geometry_contract", label + " excluded input reached fit")
                        if method == "IMAGE_ROLE_POINT_LINE":
                            self.require(solver.get("same_edge_point_line_double_count") is False
                                         and not (set(solver["line_edges"]) & set(solver["consumed_edges"]))
                                         and not (set(solver["point_ids"]) & hidden),
                                         "geometry_contract", label + " line duplication/hidden point")
                        if hidden:
                            projected = project(row["actual_pose"], row["K"])
                            projection_frames += 1
                            for corner in hidden:
                                projection_corners += 1
                                self.require(close(projected[corner], row["native_points"][corner], 1e-7, 0),
                                             "geometry_contract", label + f" hidden projection {corner}")
                    branch = "N3_SUBPIX" if method.startswith("N3_SUBPIX_") else "BASE"
                    self.require(close(row["native_points"][8], f["points"][branch][8], 1e-8, 0),
                                 "geometry_contract", label + " center changed")
                    if fallback and method != "SHARED_BOUNDARY_GEOM_ROBUST":
                        self.require(close(row["native_points"], f["points"][branch], 1e-8, 0),
                                     "geometry_contract", label + " fallback changed original coordinates")
                elif not row.get("actual_pose"):
                    missing_historical_pose += 1
                if available:
                    for field, _, _ in FIELDS.values():
                        value = row["pose"].get(field)
                        self.require(isinstance(value, (float, int)) and math.isfinite(value) and value >= 0,
                                     "raw_metric_values", label + "/" + field)
                # Retain public fields needed for later comparisons, not duplicated solver alternatives.
                retained = ("id", "session", "method", "pose", "native_points", "pose_available",
                            "new_pose_estimated", "fallback_used", "no_pose", "hidden_reprojected",
                            "hidden_initial", "reprojected_ids", "output_status", "input_points",
                            "observation_raw_logits_sha256", "selected_queries", "selected_corner_ids", "all_no_match")
                compact = {key: row.get(key) for key in retained}
                compact["actual_pose"] = pose_parameters(row.get("actual_pose"))
                compact["initial_pose"] = pose_parameters(row.get("initial_pose"))
                rows[method][fid] = compact
            raw_counts[filename] = count
            self.require(count == expected_rows, "raw_identity", f"{filename}: {count} != {expected_rows}")
        self.require(set(rows) == set(METHODS), "raw_identity", "exact 23 methods")
        for method, arm in rows.items():
            self.require(set(arm) == ids and Counter(r["session"] for r in arm.values()) == sessions,
                         "raw_identity", method + " population differs")
        self.check("raw_identity", files=raw_counts, methods=len(rows), rows=sum(raw_counts.values()))
        self.check("output_accounting", statuses={m: dict(Counter(r["output_status"] for r in rr.values())) for m, rr in rows.items()})
        self.check("raw_metric_values", metrics=[*FIELDS], units={m: f[2] for m, f in FIELDS.items()})
        self.check("geometry_contract", hidden_projection_frames=projection_frames,
                   hidden_projection_corners=projection_corners, historical_rows_without_Rt=missing_historical_pose,
                   projection="stored R_cf, centroid, cf_extents and K; pinhole; eight canonical cuboid corner signs")

        metrics = self.load("METRICS.json")["methods"]
        self.require(set(metrics) == set(METHODS), "statistics_json", "METRICS methods")
        recomputed = {}
        for method in METHODS:
            arm = list(rows[method].values())
            summary = metrics[method]
            flag_ids = {"available_ids": [r["id"] for r in arm if r["pose"]["available"]],
                        "new_pose_ids": [r["id"] for r in arm if r["new_pose_estimated"] and r["pose"]["available"]],
                        "fallback_ids": [r["id"] for r in arm if r["fallback_used"]],
                        "no_pose_ids": [r["id"] for r in arm if not r["pose"]["available"]]}
            counts = dict(total_frames=len(arm), pose_available=len(flag_ids["available_ids"]),
                          new_pose_estimated=len(flag_ids["new_pose_ids"]), fallback_used=len(flag_ids["fallback_ids"]),
                          no_pose=len(flag_ids["no_pose_ids"]), hidden_reprojected=sum(r["hidden_reprojected"] for r in arm))
            for key, value in counts.items():
                self.require(summary[key] == value, "statistics_json", method + "/" + key)
            for key, value in flag_ids.items():
                self.require(len(summary[key]) == len(value) and set(summary[key]) == set(value),
                             "statistics_json", method + "/" + key)
            for scope in ("operational", "new_pose"):
                selected = [r for r in arm if r["pose"]["available"] and (scope != "new_pose" or r["new_pose_estimated"])]
                for metric, (field, factor, unit) in FIELDS.items():
                    stats = distribution([r["pose"][field] * factor for r in selected])
                    recomputed[(method, scope, metric)] = stats
                    expected = summary["metrics"][scope][metric]
                    for key in STAT_KEYS:
                        self.require(close(stats[key], expected[key]), "statistics_json", f"{method}/{scope}/{metric}/{key}")
                    self.require(expected["unit"] == unit and expected["ddof"] == 1,
                                 "statistics_json", f"{method}/{scope}/{metric}/units-ddof")
        self.check("statistics_json", groups=len(recomputed), quantile="linear interpolation at (n-1)*p", variance="sample, ddof=1")

        csv_path = self.register(self.doc / "METRICS.csv")
        csv_keys = set()
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["method"], row["scope"], row["metric"])
                self.require(key in recomputed and key not in csv_keys, "statistics_csv", str(key) + " identity")
                csv_keys.add(key)
                if key not in recomputed:
                    continue
                for field in STAT_KEYS:
                    value = None if row[field] == "" else float(row[field])
                    self.require(close(recomputed[key][field], value), "statistics_csv", str(key) + "/" + field)
                self.require(row["unit"] == FIELDS[key[2]][2] and int(row["ddof"]) == 1,
                             "statistics_csv", str(key) + " units/ddof")
                for field in ("total_frames", "new_pose_estimated", "fallback_used", "no_pose"):
                    self.require(int(row[field]) == metrics[key[0]][field], "statistics_csv", str(key) + "/" + field)
        self.require(csv_keys == set(recomputed), "statistics_csv", "CSV group coverage")
        self.check("statistics_csv", groups=len(csv_keys))

        observation_keys = set()
        queries_checked = logits_checked = semantic_checked = 0
        for encoded in self.rows("LEARNED_OBSERVATIONS.jsonl.gz"):
            method, fid = encoded["method"], encoded["id"]
            key, label = (method, fid), f"{method}/{fid}"
            self.require(method in LEARNED and fid in ids and key not in observation_keys,
                         "learned_observations", label + " identity")
            observation_keys.add(key)
            self.require(encoded["session"] == frames[fid]["session"] and encoded["GT_input"] is False
                         and close(encoded["original_base_points"], frames[fid]["points"]["BASE"]),
                         "learned_observations", label + " input identity")
            corner_ids = [c["id"] for c in encoded["corners"]]
            self.require(len(corner_ids) == len(set(corner_ids)) and
                         all(type(c["id"]) is int and 0 <= c["id"] < 8 and len(set(c["edges"])) == 2 for c in encoded["corners"]),
                         "learned_observations", label + " corner identity")
            queries = encoded["queries"]
            if encoded["detector_available"]:
                raw = base64.b64decode(encoded["candidate_logits_f32_base64"], validate=True)
                self.require(encoded["candidate_logits_shape"] == [84, 66] and len(raw) == 84 * 66 * 4,
                             "learned_observations", label + " tensor shape")
                floats = struct.unpack("<" + "f" * (len(raw) // 4), raw)
                self.require(all(math.isfinite(v) for v in floats) and hashlib.sha256(raw).hexdigest() == encoded["raw_logits_sha256"],
                             "learned_observations", label + " raw logit hash/finite")
                self.require(len(queries) == 84 and [q["query"] for q in queries] == list(range(84)),
                             "learned_observations", label + " query coverage")
                decoded = {k: v for k, v in encoded.items() if k not in
                           ("candidate_logits_f32_base64", "candidate_logits_shape", "unencoded_semantic_sha256")}
                decoded["queries"] = []
                for i, q in enumerate(queries):
                    scores = floats[i * 66:(i + 1) * 66]
                    chosen = max(range(len(scores)), key=scores.__getitem__)
                    a, b = (encoded["original_base_points"][j] for j in q["endpoints"])
                    valid = all(v is not None and math.isfinite(v) for v in a + b) and a != [-1, -1] and b != [-1, -1]
                    valid = valid and math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-6
                    if valid:
                        length = math.hypot(b[0] - a[0], b[1] - a[1])
                        center = [a[j] * (1 - q["fraction"]) + b[j] * q["fraction"] for j in range(2)]
                        normal = [-(b[1] - a[1]) / length, (b[0] - a[0]) / length]
                        self.require(close(q["center"], center, 1e-7, 0) and close(q["normal"], normal, 1e-8, 0),
                                     "learned_observations", label + f" query {i} original-coordinate geometry")
                    no_match = chosen == 65 or not valid
                    expected_xy = None if no_match else [q["center"][j] + (chosen - 32) * q["normal"][j] for j in range(2)]
                    self.require(q["chosen_candidate"] == chosen and q["no_match"] == no_match
                                 and close(q["selected_xy"], expected_xy, 1e-8, 0),
                                 "learned_observations", label + f" query {i} argmax/no-match/coordinate")
                    decoded["queries"].append(dict(q, candidate_logits=list(scores)))
                semantic = hashlib.sha256(json.dumps(decoded, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                self.require(semantic == encoded["unencoded_semantic_sha256"], "learned_observations", label + " semantic hash")
                semantic_checked += 1
                logits_checked += len(floats)
                queries_checked += len(queries)
            selected = sum(not q["no_match"] for q in queries)
            # Historical field name means no fitted line; one isolated selected
            # query can be present without the two-query support needed for TLS.
            self.require(encoded["selected_queries"] == selected and encoded["all_no_match"] == (len(encoded["lines"]) == 0),
                         "learned_observations", label + " query count")
            lines = {line["edge"]: line for line in encoded["lines"]}
            self.require(len(lines) == len(encoded["lines"]), "learned_observations", label + " duplicate physical edge")
            for line in lines.values():
                support = line["support_points"]
                chosen_queries = line["queries"]
                self.require(len(support) >= 2 and len(support) == len(chosen_queries)
                             and all(not queries[i]["no_match"] and queries[i]["edge"] == line["edge"]
                                     and close(queries[i]["selected_xy"], p, 1e-8, 0) for i, p in zip(chosen_queries, support)),
                             "learned_observations", label + f" edge {line['edge']} selected support")
                normal, offset = line["normal"], line["offset"]
                self.require(close(math.hypot(*normal), 1., 1e-8, 0), "learned_observations", label + " unit line normal")
                residuals = [math.fsum(normal[j] * p[j] for j in range(2)) - offset for p in support]
                rms = math.sqrt(math.fsum(r * r for r in residuals) / len(residuals))
                tangent_coords = [-normal[1] * p[0] + normal[0] * p[1] for p in support]
                self.require(close(rms, line["residual_rms_px"], 1e-7, 0)
                             and close(max(tangent_coords) - min(tangent_coords), line["support_length_px"], 1e-7, 0),
                             "learned_observations", label + " line residual/support span")
            for corner in encoded["corners"]:
                a, b = (lines[e] for e in corner["edges"])
                determinant = a["normal"][0] * b["normal"][1] - a["normal"][1] * b["normal"][0]
                self.require(abs(determinant) > 1e-6 and all(corner["id"] in line["endpoints"] for line in (a, b)),
                             "learned_observations", label + " physical incident edge intersection")
                if abs(determinant) > 1e-6:
                    xy = [(a["offset"] * b["normal"][1] - a["normal"][1] * b["offset"]) / determinant,
                          (a["normal"][0] * b["offset"] - a["offset"] * b["normal"][0]) / determinant]
                    self.require(close(xy, corner["xy"], 1e-7, 1e-10), "learned_observations", label + " corner intersection coordinate")
            prediction = rows[method][fid]
            self.require(prediction["observation_raw_logits_sha256"] == encoded["raw_logits_sha256"]
                         and prediction["selected_queries"] == selected
                         and prediction["selected_corner_ids"] == corner_ids
                         and prediction["all_no_match"] == encoded["all_no_match"],
                         "learned_observations", label + " prediction-observation linkage")
            for corner in encoded["corners"]:
                self.require(close(prediction["input_points"][corner["id"]], corner["xy"], 1e-8, 0),
                             "learned_observations", label + f" selected corner {corner['id']}")
        self.require(observation_keys == {(m, fid) for m in LEARNED for fid in ids},
                     "learned_observations", "exact 957 model-frame observation keys")
        seal = self.load("OBSERVATION_SEAL.json")
        self.require(seal["complete"] is True and seal["records"]["sha256"] == sha(self.doc / "LEARNED_OBSERVATIONS.jsonl.gz"),
                     "learned_observations", "observation seal file hash")
        self.check("learned_observations", rows=len(observation_keys), queries=queries_checked,
                   float32_logits=logits_checked, lossless_semantic_hashes=semantic_checked,
                   all_no_match_definition="no fitted physical-edge line; isolated selected queries can remain",
                   limitation="validates stored inference decisions and linkage, not image-to-head replay")

        runtime = self.load("RUNTIME.json")
        runtime_rows = list(self.rows("RUNTIME_ROWS.jsonl.gz"))
        panel = {r["id"]: r for r in runtime["panel"]}
        self.require(tuple(runtime["arms"]) == ARMS and len(panel) == len(runtime["panel"]) == 26
                     and Counter(r["session"] for r in panel.values()) == {s: 2 for s in sessions},
                     "runtime_identity", "four arms and 26-frame two-per-session panel")
        self.require(runtime["complete"] is True and runtime["statistics_official"] is True
                     and runtime["warmup_per_arm"] == 20 and runtime["repeats"] == 5
                     and runtime["measured_per_arm"] == 130 and runtime["max_pipeline_calls"] == 600,
                     "runtime_identity", "runtime completeness/budget")
        self.require(len(runtime_rows) == 600, "runtime_identity", "600 full path rows")
        call_counts = Counter()
        parity_rows = 0
        for arm in ARMS:
            rr = [r for r in runtime_rows if r["arm"] == arm]
            warm = [r for r in rr if r["phase"] == "warmup"]
            measured = [r for r in rr if r["phase"] == "measured"]
            self.require(len(rr) == 150 and len(warm) == 20 and len(measured) == 130
                         and {r["warmup_index"] for r in warm} == set(range(20))
                         and Counter((r["repeat"], r["id"]) for r in measured)
                         == {(repeat, fid): 1 for repeat in range(5) for fid in panel},
                         "runtime_identity", arm + " exact warmup/measured identities")
            for r in rr:
                fid = r["id"]
                self.require(fid in panel and r["session"] == frames[fid]["session"], "runtime_identity", arm + "/" + fid)
                self.require(all(isinstance(r[s + "_ms"], (int, float)) and math.isfinite(r[s + "_ms"]) and r[s + "_ms"] >= 0 for s in STAGES)
                             and close(r["full_ms"], math.fsum(r[s + "_ms"] for s in STAGES[1:]), 1e-6, 0),
                             "runtime_statistics", arm + "/" + fid + " actual interval values/sum")
                call_counts.update(r["call_counts"])
                if arm in ("BASE", "N3_SUBPIX"):
                    expected_points = frames[fid]["points"][arm]
                    expected_pose = rows[arm + "_NO_MASK_STANDARD"][fid]["initial_pose"]
                else:
                    expected_points = rows[arm][fid]["native_points"]
                    expected_pose = rows[arm][fid]["actual_pose"]
                actual = pose_parameters(r["actual_pose"])
                pose_equal = ((actual is None and expected_pose is None) or
                              (actual is not None and expected_pose is not None and
                               all(close(actual[k], expected_pose[k], 1e-6, 1e-9) for k in actual)))
                self.require(close(r["final_points"], expected_points, 1e-7, 0) and pose_equal,
                             "runtime_output_parity", arm + "/" + fid + " stored output vs full319 output")
                self.require(r["GT_canary_active"] is True and r["RAW_cached_bitexact"] is True
                             and r["candidate_center_score_box_preserved"] is True,
                             "runtime_output_parity", arm + "/" + fid + " recorded execution flags")
                parity_rows += 1
            for stage in STAGES:
                actual = distribution([r[stage + "_ms"] for r in measured])
                saved = runtime["summaries"][arm][stage]
                mapping = dict(n="n", mean="mean_ms", sample_variance="sample_variance_ms2",
                               sample_std="sample_std_ms", median="median_ms", P90="p90_ms")
                for key, saved_key in mapping.items():
                    self.require(close(actual[key], saved[saved_key]), "runtime_statistics", arm + "/" + stage + "/" + key)
                self.require(saved["ddof"] == 1, "runtime_statistics", arm + "/" + stage + " ddof")
        self.require(set(r["arm"] for r in runtime_rows) == set(ARMS), "runtime_identity", "unknown runtime arm")
        for name, count in call_counts.items():
            self.require(runtime["execution"][name] == count, "runtime_identity", name + " runtime execution count")
        self.require(runtime["execution"]["pipeline_calls_started"] == runtime["execution"]["pipeline_calls_complete"]
                     == runtime["execution"]["detector_calls"] == len(runtime_rows),
                     "runtime_identity", "600 detector/pipeline calls")
        self.check("runtime_identity", rows=len(runtime_rows), measured=520, warmup=80, operation_counts=dict(call_counts))
        self.check("runtime_statistics", groups=len(ARMS) * len(STAGES), units="milliseconds; variance milliseconds squared")
        self.check("runtime_output_parity", rows=parity_rows,
                   limitation="compares saved full-forward outputs; execution boundaries/canary are recorded evidence, not rerun")

        loss_path = self.doc / "REVIEW_LOSS_MASK_CHECK.json"
        if loss_path.exists():
            loss = self.load("REVIEW_LOSS_MASK_CHECK.json")
            fields = loss["fields"]
            self.require(set(fields) == {"lo", "hi", "weight", "valid"} and
                         all(len(values) == 16 and all(len(row) == 84 for row in values) for values in fields.values()),
                         "loss_mask_targets", "exact public first-batch 16x84 targets")
            counts = Counter(images=16, positive=0, no_match=0, true_ignore=0)
            for i in range(16):
                for j in range(84):
                    lo, hi, weight, valid = (fields[k][i][j] for k in ("lo", "hi", "weight", "valid"))
                    self.require(type(lo) is int and type(hi) is int and type(valid) is bool
                                 and 0 <= lo <= hi <= 65 and math.isfinite(weight) and 0 <= weight <= 1,
                                 "loss_mask_targets", f"target {i}/{j} types/ranges")
                    if not valid:
                        counts["true_ignore"] += 1
                    elif lo == hi == 65:
                        counts["no_match"] += 1
                    else:
                        counts["positive"] += 1
                        self.require(lo < 65 and hi < 65, "loss_mask_targets", f"positive {i}/{j} bin")
                    if not valid or lo == hi == 65:
                        self.require(lo == hi == 65 and weight == 0., "loss_mask_targets", f"none/ignore {i}/{j} encoding")
            self.require(dict(counts) == loss["counts"] == dict(images=16, positive=249, no_match=635, true_ignore=460),
                         "loss_mask_targets", "independent positive/none/ignore counts")
            self.require(close(math.log(66), loss["uniform_logits_loss"], 1e-12, 0),
                         "loss_mask_targets", "analytic uniform-logits CE loss log66")
            self.require(loss["real_GT_used"] is False and loss["model_head_forwards"] == 0
                         and loss["detector_forwards"] == 0 and loss["training_updates"] == 0,
                         "loss_mask_targets", "recorded loss-only execution scope")
            source = (self.repo / loss["loss_source"]["path"]).resolve()
            self.require(source.is_relative_to(self.repo) and source.is_file(), "loss_mask_targets", "loss source path")
            if source.is_relative_to(self.repo) and source.is_file():
                self.require(sha(self.register(source)) == loss["loss_source"]["sha256"], "loss_mask_targets", "loss source SHA256")
            self.check("loss_mask_targets", independent_counts=dict(counts), analytic_uniform_logits_loss=math.log(66),
                       limitation="stdlib verifies stored target masks/counts/ranges and analytic loss; actual autograd derivative checks require the optional PyTorch rerun")
        else:
            self.checks.append(dict(name="loss_mask_targets", status="NOT_PRESENT", details=dict(required=False)))

        manifest_path = self.doc / "REVIEW_MANIFEST.json"
        if manifest_path.exists():
            manifest = self.load("REVIEW_MANIFEST.json")
            paths = set()
            for entry in manifest["files"]:
                relative = entry["path"]
                path = (self.repo / relative).resolve()
                confined = not Path(relative).is_absolute() and path.is_relative_to(self.repo)
                self.require(confined and relative not in paths, "manifest", relative + " path/duplicate")
                paths.add(relative)
                if not confined:
                    continue
                self.require(path.is_file(), "manifest", relative + " missing")
                if path.is_file():
                    self.require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
                                 "manifest", relative + " bytes/SHA256")
            for name in ("REVIEW_CHECKS.json", "REVIEW_MANIFEST.json"):
                self.require(not any(Path(p).name == name for p in paths), "manifest", name + " must be excluded from hash cycle")
            original = self.load("PUBLICATION_PRECHECK.json")["files"]
            mutable = {"report.py", "RESULT_KO.md", "README.md", "REPRODUCE.md", "REPORT_NUMBERS.json"}
            immutable = [entry for entry in original if Path(entry["path"]).name not in mutable]
            declared = manifest["original_research_payload"]
            binding = lambda entry: {key: entry[key] for key in ("path", "sha256", "bytes")}
            expected_bindings = {entry["path"]: binding(entry) for entry in immutable}
            declared_bindings = {entry["path"]: binding(entry) for entry in declared}
            current_bindings = {entry["path"]: binding(entry) for entry in manifest["files"]}
            self.require(len(immutable) == len(expected_bindings) == len(declared) == len(declared_bindings) == 86,
                         "manifest", "exact 86 unique immutable original research payload paths")
            self.require(declared_bindings == expected_bindings and
                         all(current_bindings.get(path) == entry for path, entry in expected_bindings.items()),
                         "manifest", "immutable path/SHA256/bytes differ from PUBLICATION_PRECHECK or current bindings")
            self.require(manifest["original_research_payload_unchanged"] is True,
                         "manifest", "original_research_payload_unchanged must be true")
            expected_inputs = set(self.inputs) - {str(manifest_path.relative_to(self.repo))}
            self.require(expected_inputs <= paths, "manifest", "manifest missing a numerically verified input")
            self.check("manifest", files=len(paths), algorithm="SHA256", original_immutable_files=len(declared_bindings),
                       scope="public files only; original bindings checked against PUBLICATION_PRECHECK; manifest is an integrity index, not a signed authenticity proof")
        else:
            self.require(not require_manifest, "manifest", "REVIEW_MANIFEST.json is required but absent")
            self.checks.append(dict(name="manifest", status="FAIL" if require_manifest else "NOT_PRESENT",
                                    details=dict(required=require_manifest)))
        detail = None
        if frame_id in frames:
            detail = dict(input=frames[frame_id], methods={m: rows[m][frame_id] for m in METHODS})
        return detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="repository root or this experiment's result directory; default: script repository")
    parser.add_argument("--output", type=Path, help="JSON output; default: result directory/REVIEW_CHECKS.json")
    parser.add_argument("--require-manifest", action="store_true", help="fail if REVIEW_MANIFEST.json is absent")
    parser.add_argument("--frame-id", help="include one frame's public numerical inputs and all23 outputs; still checks all319")
    args = parser.parse_args()
    root = (args.root or Path(__file__).resolve().parents[3]).resolve()
    if (root / "INPUTS.json").is_file():
        doc = root
        repo = root.parents[2]
    else:
        repo, doc = root, root / "_docs" / "experiments" / EXPERIMENT
    output = (args.output or doc / "REVIEW_CHECKS.json").resolve()
    review = Review(repo, doc)
    frame_detail, complete = None, False
    try:
        frame_detail = review.run(args.frame_id, args.require_manifest)
        complete = True
    except Exception as error:
        review.problems.append(dict(check="execution", detail=f"{type(error).__name__}: {error}"))
        review.checks.append(dict(name="execution", status="FAIL", details=dict(error_type=type(error).__name__)))
    passed = complete and not review.problems and not any(c["status"] == "FAIL" for c in review.checks)
    result = dict(schema="public_review_checks_v1", complete=complete, passed=passed,
                  generated_utc=datetime.now(timezone.utc).isoformat(),
                  implementation=dict(path=str(Path(__file__).resolve().relative_to(repo)) if Path(__file__).resolve().is_relative_to(repo) else Path(__file__).name,
                                      sha256=sha(Path(__file__).resolve()), python=sys.version.split()[0], dependencies="Python standard library only"),
                  checks=review.checks, problems=review.problems, verified_inputs=list(review.inputs.values()),
                  limits=["Stored pose-error rows are independently aggregated; private target poses, physical annotation correctness and image-derived error measurements are not reestablished.",
                          "Projection checks use public stored estimated poses, calibration and cuboid dimensions; they do not establish pose accuracy against ground truth.",
                          "Historical controls with missing saved R/t cannot receive projection or full pose-output checks.",
                          "Logit checks establish stored argmax/no-match/coordinate decisions and lossless storage, not a detector or learned model replay.",
                          "Runtime checks recompute statistics and compare stored outputs; actual timing, GPU synchronization and isolation are recorded evidence, not independently rerun.",
                          "Bootstrap confidence intervals, physical mesh supervision and training/checkpoint authenticity are outside this standalone verifier's scope.",
                          "A manifest hash is integrity evidence relative to that manifest, not an independently signed provenance assertion."],
                  frame_detail=frame_detail)
    # Refuse to overwrite research inputs, source, or manifest even with --output.
    protected = {str((repo / p).resolve()) for p in review.inputs}
    if (str(output) in protected or output == Path(__file__).resolve() or output.name == "REVIEW_MANIFEST.json"
            or (output.exists() and output.is_relative_to(repo) and output.name != "REVIEW_CHECKS.json")):
        print("Refusing to overwrite a verified research input/source/manifest", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=passed, complete=complete, checks=len(review.checks),
                          failures=len(review.problems), output=str(output)), ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
