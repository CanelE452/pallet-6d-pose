#!/usr/bin/env python3
"""Public-only decomposition of saved query/line/corner observation difficulty.

No model import, forward, training, image read, ground-truth read or pose solve.
Diagnostic thresholds label all saved rows; they never select configurations or
replace absent correspondences. Completed output files are never overwritten.
"""
import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import time


OLD = "pallet_observation_refiner_20261009_v1"
NEW = "pallet_kp_difficulty_20261010_v1"
ARMS = ("GEOMETRY_ONLY", "IMAGE_NO_ROLE", "IMAGE_ROLE")
ROLES = ("BOUNDARY_PROXY", "INTERNAL_PROXY", "UNAVAILABLE")
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
DIAGNOSTICS = dict(weak_argmax_gap=0.1, acute_intersection_angle_deg=5.,
                   short_line_support_px=8., large_extrapolation_over_span=5.)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def finite_xy(point):
    return len(point) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)


def in_image(point, h, w):
    return finite_xy(point) and 0 <= point[0] < w and 0 <= point[1] < h


def close(a, b, tolerance=1e-7):
    if a is None or b is None:
        return a is b
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y, tolerance) for x, y in zip(a, b))
    return math.isclose(a, b, abs_tol=tolerance, rel_tol=1e-10)


def distribution(values):
    a = sorted(values)
    n = len(a)
    if not n:
        return dict(n=0, mean=None, sample_variance=None, sample_std=None, median=None, P90=None, min=None, max=None)
    mean = math.fsum(a) / n
    var = math.fsum((x - mean) ** 2 for x in a) / (n - 1) if n > 1 else None
    def q(p):
        x = (n - 1) * p
        i = math.floor(x)
        return a[i] + (a[min(i + 1, n - 1)] - a[i]) * (x - i)
    return dict(n=n, mean=mean, sample_variance=var, sample_std=math.sqrt(var) if var is not None else None,
                median=q(.5), P90=q(.9), min=a[0], max=a[-1])


def analyze(encoded, frame, prediction):
    h, w = frame["raw_hw"]
    points = encoded["original_base_points"]
    assert close(points, frame["points"]["BASE"])
    assert encoded["candidate_logits_shape"] == [84, 66]
    blob = base64.b64decode(encoded["candidate_logits_f32_base64"], validate=True)
    assert len(blob) == 84 * 66 * 4
    logits = struct.unpack("<" + "f" * (84 * 66), blob)
    assert all(math.isfinite(v) for v in logits)
    assert sha_bytes(blob) == encoded["raw_logits_sha256"]
    decoded = {k: v for k, v in encoded.items() if k not in
               ("candidate_logits_f32_base64", "candidate_logits_shape", "unencoded_semantic_sha256")}
    decoded["queries"] = []
    queries, counters = [], Counter()
    for i, q in enumerate(encoded["queries"]):
        assert q["query"] == i and tuple(q["endpoints"]) == EDGES[q["edge"]]
        scores = list(logits[i * 66:(i + 1) * 66])
        chosen = max(range(66), key=scores.__getitem__)
        spatial = max(range(65), key=scores.__getitem__)
        ranked = sorted(scores, reverse=True)
        gap, margin = ranked[0] - ranked[1], scores[spatial] - scores[65]
        a, b = (points[j] for j in q["endpoints"])
        valid = finite_xy(a) and finite_xy(b) and a != [-1, -1] and b != [-1, -1]
        valid = valid and math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-6
        selected = valid and chosen < 65
        expected_xy = None if not selected else [q["center"][j] + (chosen - 32) * q["normal"][j] for j in range(2)]
        assert chosen == q["chosen_candidate"] and q["no_match"] == (not selected)
        assert close(expected_xy, q["selected_xy"])
        role_vector = encoded["predicted_roles"][q["edge"]]
        assert len(role_vector) == 3 and role_vector.count(1.) == 1 and sum(role_vector) == 1.
        role = ROLES[role_vector.index(1.)]
        inside_candidates = sum(in_image([q["center"][j] + offset * q["normal"][j] for j in range(2)], h, w)
                                for offset in range(-32, 33))
        query = dict(query=i, edge=q["edge"], endpoints=q["endpoints"], fraction=q["fraction"],
                     valid=valid, role=role, center=q["center"], center_in_image=in_image(q["center"], h, w),
                     candidate_count_in_image=inside_candidates, raw_argmax=chosen, argmax_is_none=chosen == 65,
                     best_spatial_bin=spatial, best_spatial_offset_px=spatial - 32,
                     best_spatial_logit=scores[spatial], none_logit=scores[65],
                     spatial_minus_none_margin=margin, winner_runnerup_gap=gap,
                     weak_gap_diagnostic=gap <= DIAGNOSTICS["weak_argmax_gap"],
                     selected=selected, selected_xy=q["selected_xy"],
                     selected_in_image=in_image(q["selected_xy"], h, w) if selected else None)
        queries.append(query)
        counters.update(dict(queries=1, valid_queries=int(valid), accepted_queries=int(selected),
                             argmax_none_queries=int(chosen == 65), invalid_queries=int(not valid),
                             centers_out_of_image=int(not query["center_in_image"]),
                             accepted_out_of_image=int(selected and not query["selected_in_image"]),
                             candidate_locations_out_of_image=65 - inside_candidates,
                             weak_gap_queries=int(query["weak_gap_diagnostic"])))
        decoded["queries"].append(dict(q, candidate_logits=scores))
    assert len(queries) == 84
    semantic = sha_bytes(json.dumps(decoded, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    assert semantic == encoded["unencoded_semantic_sha256"]
    lines = {line["edge"]: line for line in encoded["lines"]}
    assert len(lines) == len(encoded["lines"])
    edge_rows = []
    for e, endpoints in enumerate(EDGES):
        accepted = [q["query"] for q in queries[e * 7:(e + 1) * 7] if q["selected"]]
        line = lines.get(e)
        if line:
            assert line["queries"] == accepted and len(accepted) >= 2
            assert all(close(p, queries[i]["selected_xy"]) for p, i in zip(line["support_points"], accepted))
        edge_rows.append(dict(edge=e, endpoints=list(endpoints), role=queries[e * 7]["role"],
                              accepted_queries=len(accepted), accepted_query_ids=accepted,
                              at_least_two_support_queries=len(accepted) >= 2,
                              stored_line_available=line is not None,
                              support_length_px=line["support_length_px"] if line else None,
                              residual_rms_px=line["residual_rms_px"] if line else None,
                              short_support_diagnostic=bool(line and line["support_length_px"] < DIAGNOSTICS["short_line_support_px"])))
    hidden = set(prediction["hidden_initial"])
    eligible, corner_rows = [], []
    for corner in encoded["corners"]:
        first, second = (lines[e] for e in corner["edges"])
        n1, n2 = first["normal"], second["normal"]
        det = abs(n1[0] * n2[1] - n1[1] * n2[0])
        norm1, norm2 = math.hypot(*n1), math.hypot(*n2)
        cosine = min(1., max(0., abs(n1[0] * n2[0] + n1[1] * n2[1]) / (norm1 * norm2)))
        angle = math.degrees(math.acos(cosine))
        condition = math.sqrt((1 + cosine) / (1 - cosine)) if cosine < 1 else None
        extrap = []
        for line in (first, second):
            normal = line["normal"]
            tangent = [-normal[1], normal[0]]
            point_u = sum(corner["xy"][j] * tangent[j] for j in range(2))
            support_u = [sum(p[j] * tangent[j] for j in range(2)) for p in line["support_points"]]
            distance = max(min(support_u) - point_u, point_u - max(support_u), 0.)
            ratio = distance / max(line["support_length_px"], 1e-12)
            extrap.append(dict(edge=line["edge"], distance_beyond_support_px=distance,
                               distance_over_support_span=ratio,
                               large_extrapolation_diagnostic=ratio > DIAGNOSTICS["large_extrapolation_over_span"]))
        roi = in_image(corner["xy"], h, w) and corner["xy"] != [-1, -1]
        if roi:
            eligible.append(corner["id"])
        corner_rows.append(dict(id=corner["id"], xy=corner["xy"], edges=corner["edges"],
                                acute_line_angle_deg=angle, abs_normal_determinant=det,
                                unit_normal_system_condition_number=condition,
                                acute_angle_diagnostic=angle < DIAGNOSTICS["acute_intersection_angle_deg"],
                                in_image=roi, excluded_by_fixed_initial_mask=corner["id"] in hidden,
                                eligible_after_fixed_mask=roi and corner["id"] not in hidden,
                                extrapolation=extrap))
    corner_ids = [c["id"] for c in corner_rows]
    assert len(corner_ids) == len(set(corner_ids))
    assert set(eligible) == set(prediction["solver"]["eligible"])
    assert prediction["observation_raw_logits_sha256"] == encoded["raw_logits_sha256"]
    assert prediction["selected_corner_ids"] == corner_ids
    assert prediction["selected_queries"] == sum(q["selected"] for q in queries)
    after_mask = [i for i in corner_ids if i not in hidden]
    after_roi_mask = [i for i in eligible if i not in hidden]
    stage = ("BELOW_FOUR_RAW_CORNERS" if len(corner_ids) < 4 else
             "ROI_BELOW_FOUR" if len(eligible) < 4 else
             "FIXED_MASK_BELOW_FOUR" if len(after_roi_mask) < 4 else
             "FOUR_REMAINING_STORED_NEW_POSE" if prediction["new_pose_estimated"] else
             "FOUR_REMAINING_STORED_SOLVER_FAILURE")
    return dict(schema="learned_difficulty_frame_v1", id=encoded["id"], session=encoded["session"], method=encoded["method"],
                raw_hw=frame["raw_hw"], raw_logits_sha256=encoded["raw_logits_sha256"],
                lossless_semantic_sha256=semantic, raw_logit_and_semantic_hash_verified=True,
                query_counts=dict(counters), queries=queries, edges=edge_rows, corners=corner_rows,
                counts=dict(accepted_queries=sum(q["selected"] for q in queries),
                            edges_with_two_or_more_support_queries=sum(e["at_least_two_support_queries"] for e in edge_rows),
                            stored_lines=len(lines), raw_corners=len(corner_ids), corners_after_fixed_mask=len(after_mask),
                            eligible_roi_corners=len(eligible), eligible_roi_after_fixed_mask=len(after_roi_mask)),
                fixed_initial_hidden_ids=sorted(hidden), feature_initial_hidden_ids=encoded["feature_hidden_initial"],
                feature_vs_final_mask_differs=set(encoded["feature_hidden_initial"]) != hidden,
                raw_corner_ids=corner_ids, eligible_roi_ids=eligible, eligible_roi_after_fixed_mask_ids=after_roi_mask,
                observation_bottleneck=stage, stored_output_status=prediction["output_status"],
                stored_new_pose_estimated=prediction["new_pose_estimated"],
                stored_solver_reason=prediction["solver"]["reason"],
                new_pose_solver_executed=False, no_match_filled_with_initial_coordinates=False)


def sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def summarize(rows):
    result = {}
    for arm in ARMS:
        rr = [r for r in rows if r["method"] == arm]
        edge = {e: Counter() for e in range(12)}
        role = {name: Counter() for name in ROLES}
        counters = Counter()
        for r in rr:
            counters.update(r["query_counts"])
            for q in r["queries"]:
                role[q["role"]].update(dict(queries=1, accepted=int(q["selected"]), argmax_none=int(q["argmax_is_none"])))
            for e in r["edges"]:
                edge[e["edge"]].update(dict(accepted_queries=e["accepted_queries"], line_frames=int(e["stored_line_available"]),
                                             two_support_frames=int(e["at_least_two_support_queries"])))
        query_values = [q for r in rr for q in r["queries"]]
        corner_values = [c for r in rr for c in r["corners"]]
        extrap = [e for c in corner_values for e in c["extrapolation"]]
        result[arm] = dict(frames=len(rr), sessions=len({r["session"] for r in rr}), query_counts=dict(counters),
                           roles={k: dict(v) for k, v in role.items()}, edges={str(k): dict(v) for k, v in edge.items()},
                           corner_id_counts=dict(Counter(c["id"] for c in corner_values)),
                           stage_count_histograms={key: dict(Counter(r["counts"][key] for r in rr)) for key in rr[0]["counts"]},
                           bottleneck_counts=dict(Counter(r["observation_bottleneck"] for r in rr)),
                           feature_vs_final_mask_difference_frames=sum(r["feature_vs_final_mask_differs"] for r in rr),
                           accepted_normal_offset_px=distribution([q["raw_argmax"] - 32 for q in query_values if q["selected"]]),
                           spatial_minus_none_margin={"accepted": distribution([q["spatial_minus_none_margin"] for q in query_values if q["selected"]]),
                                                      "rejected": distribution([q["spatial_minus_none_margin"] for q in query_values if not q["selected"]])},
                           winner_runnerup_gap=distribution([q["winner_runnerup_gap"] for q in query_values]),
                           intersection_abs_determinant=distribution([c["abs_normal_determinant"] for c in corner_values]),
                           intersection_acute_angle_deg=distribution([c["acute_line_angle_deg"] for c in corner_values]),
                           extrapolation_length_px=distribution([e["distance_beyond_support_px"] for e in extrap]),
                           extrapolation_over_support_span=distribution([e["distance_over_support_span"] for e in extrap]),
                           four_raw_corner_case_ids=[r["id"] for r in rr if r["counts"]["raw_corners"] >= 4],
                           acute_angle_case_ids=sorted({r["id"] for r in rr if any(c["acute_angle_diagnostic"] for c in r["corners"])}),
                           large_extrapolation_case_ids=sorted({r["id"] for r in rr if any(e["large_extrapolation_diagnostic"] for c in r["corners"] for e in c["extrapolation"])}),
                           case_ids_by_bottleneck={stage: [r["id"] for r in rr if r["observation_bottleneck"] == stage]
                                                   for stage in sorted({r["observation_bottleneck"] for r in rr})})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3], help="repository root; Python>=3.9")
    parser.add_argument("--output-dir", type=Path, help="new result directory; completed outputs cannot be overwritten")
    args = parser.parse_args()
    repo = args.root.resolve()
    old = repo / "_docs" / "experiments" / OLD
    output_dir = (args.output_dir or repo / "_docs" / "experiments" / NEW).resolve()
    raw_path, summary_path = (output_dir / name for name in ("LEARNED_DIFFICULTY_ROWS.jsonl.gz", "LEARNED_DIFFICULTY.json"))
    for path in (raw_path, summary_path, raw_path.with_suffix(raw_path.suffix + ".pending"), summary_path.with_suffix(".json.pending")):
        if path.exists():
            raise FileExistsError(f"Preserve completed or pending output: {path.name}")
    start = time.monotonic()
    paths = [old / name for name in ("INPUTS.json", "LEARNED_OBSERVATIONS.jsonl.gz", "LEARNED_PREDICTIONS.jsonl.gz")]
    bindings = [dict(path=str(p.relative_to(repo)), sha256=sha(p), bytes=p.stat().st_size) for p in paths]
    inputs = json.loads(paths[0].read_text())["frames"]
    frames = {f["id"]: f for f in inputs}
    assert len(frames) == len(inputs) == 319 and len({f["session"] for f in inputs}) == 13
    predictions = {(r["method"], r["id"]): r for r in read_rows(paths[2]) if r["method"] in ARMS}
    rows, seen = [], set()
    for encoded in read_rows(paths[1]):
        key = (encoded["method"], encoded["id"])
        assert key not in seen and encoded["detector_available"] is True and encoded["GT_input"] is False
        seen.add(key)
        assert encoded["session"] == frames[encoded["id"]]["session"]
        rows.append(analyze(encoded, frames[encoded["id"]], predictions[key]))
    assert seen == {(arm, fid) for arm in ARMS for fid in frames} and len(rows) == 957
    assert all(sha(path) == binding["sha256"] for path, binding in zip(paths, bindings)), "Original input changed during analysis"
    methods = summarize(rows)
    summary = dict(schema="learned_difficulty_v1", complete=True, generated_utc=datetime.now(timezone.utc).isoformat(),
                   population=dict(frames=319, sessions=13, models=3, rows=957), methods=methods,
                   diagnostic_thresholds=DIAGNOSTICS,
                   thresholds_used_for="diagnostic case labels only; all957 rows retained; no model/solver/configuration selection",
                   input_bindings=bindings,
                   implementation=dict(path=str(Path(__file__).resolve().relative_to(repo)), sha256=sha(Path(__file__).resolve()), dependencies="Python>=3.9 standard library"),
                   execution=dict(new_detector_forwards=0, new_head_forwards=0, new_training_updates=0, new_pose_solves=0,
                                  image_reads=0, private_truth_reads=0, stored_observation_rows=957,
                                  decoded_logits=957 * 84 * 66, wall_seconds=time.monotonic() - start),
                   limits=["Uses stored logits/coordinates/estimated masks only; physical edge visibility or location correctness is not independently established.",
                           "Four eligible corners are a count condition, not proof of sufficient placement, consensus or pose accuracy.",
                           "Query-to-corner coverage is different from conditional accuracy on selected source queries.",
                           "Endpoint extrapolation is expected from interior1/8..7/8 queries; large amplification is a fragility diagnostic, not an error proof.",
                           "BOUNDARY_PROXY/INTERNAL_PROXY come from a projected cuboid hull and are not actual pallet mesh ownership labels.",
                           "Stored pose outcomes are copied as labels; no new PnP, pose evaluation, threshold tuning or absent-correspondence fill is performed."])
    output_dir.mkdir(parents=True, exist_ok=True)
    pending_raw = raw_path.with_suffix(raw_path.suffix + ".pending")
    with gzip.open(pending_raw, "wt", encoding="utf-8", compresslevel=6) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    summary["raw_rows"] = dict(path=str(raw_path.relative_to(repo)) if raw_path.is_relative_to(repo) else raw_path.name,
                               sha256=sha(pending_raw), bytes=pending_raw.stat().st_size, rows=len(rows))
    pending_summary = summary_path.with_suffix(".json.pending")
    pending_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending_raw.replace(raw_path)
    pending_summary.replace(summary_path)
    print(json.dumps({arm: dict(bottlenecks=value["bottleneck_counts"], roles=value["roles"],
                               four_raw_corner_case_ids=value["four_raw_corner_case_ids"],
                               acute_angle_case_ids=value["acute_angle_case_ids"],
                               large_extrapolation_case_ids=value["large_extrapolation_case_ids"])
                      for arm, value in methods.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
