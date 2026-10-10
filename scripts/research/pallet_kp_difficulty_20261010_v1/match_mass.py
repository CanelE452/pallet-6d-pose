#!/usr/bin/env python3
"""Separate, fixed posterior-mass decoder for previously sealed real logits.

Existence compares logsumexp of the65 location scores to the one none score.
An accepted observation still picks one location bin by argmax. No coordinate
averaging, Base fill, model forward, pose solve, GT read or threshold tuning.
The original66-way MAP decoder is replayed independently before publication.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import struct
import time

import numpy as np


OLD = "pallet_observation_refiner_20261009_v1"
NEW = "pallet_kp_difficulty_20261010_v1"
ARMS = ("GEOMETRY_ONLY", "IMAGE_NO_ROLE", "IMAGE_ROLE")
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
POLICY = dict(existence="logsumexp(scores[:65]) > scores[65]", match_posterior_threshold=0.5,
              equality="reject", location="argmax(scores[:65]); first maximum; one original1px bin",
              query_validity="finite non-sentinel seed endpoints and length>1e-6, unchanged",
              line="TLS with>=2 selected valid queries; first singular value>1e-6, unchanged",
              corner="most orthogonal incident line pair; abs normal determinant>1e-6, unchanged",
              coordinate_averaging=False, initial_coordinate_fill=False, parameter_tuning=False)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def logsumexp(scores):
    largest = max(scores)
    return largest + math.log(math.fsum(math.exp(v - largest) for v in scores))


def choices(scores, mass=False):
    spatial = max(range(65), key=scores.__getitem__)
    lse = logsumexp(scores[:65])
    delta = lse - scores[65]
    posterior = 1 / (1 + math.exp(-delta)) if delta >= 0 else math.exp(delta) / (1 + math.exp(delta))
    chosen = spatial if mass and delta > 0 else 65 if mass else max(range(66), key=scores.__getitem__)
    return chosen, spatial, lse, posterior


def decode(encoded, logits, mass=False):
    """Mirror the frozen topology/geometry decoder without importing its modules."""
    seed = np.asarray(encoded["original_base_points"], float)
    queries, points, choice, valid_queries = [], [], [], []
    for i, original in enumerate(encoded["queries"]):
        a, b = original["endpoints"]
        vector = seed[b] - seed[a]
        valid = bool(np.isfinite(vector).all() and np.linalg.norm(vector) > 1e-6
                     and not np.any(np.all(seed[[a, b]] == -1, axis=1)))
        chosen, spatial, lse, posterior = choices(logits[i], mass)
        center, normal = np.asarray(original["center"]), np.asarray(original["normal"])
        candidate = center[None] + np.arange(-32, 33)[:, None] * normal[None]
        selected = chosen < 65 and valid
        q = {k: v for k, v in original.items() if k not in ("chosen_candidate", "no_match", "selected_xy")}
        q.update(chosen_candidate=chosen, no_match=not selected,
                 selected_xy=candidate[chosen].tolist() if selected else None)
        if mass:
            q.update(source_66way_chosen_candidate=original["chosen_candidate"],
                     source_66way_no_match=original["no_match"],
                     match_logsumexp=lse, none_logit=float(logits[i, 65]),
                     match_posterior_probability=posterior,
                     best_spatial_candidate=spatial)
        queries.append(q)
        points.append(candidate)
        choice.append(chosen)
        valid_queries.append(valid)
    lines = []
    for edge, (a, b) in enumerate(EDGES):
        ids = [i for i in range(edge * 7, (edge + 1) * 7) if valid_queries[i] and choice[i] < 65]
        if len(ids) < 2:
            continue
        support = np.asarray([points[i][choice[i]] for i in ids])
        center = support.mean(0)
        _, singular, v = np.linalg.svd(support - center, full_matrices=False)
        if singular[0] < 1e-6:
            continue
        tangent = v[0]
        normal = np.array([-tangent[1], tangent[0]])
        offset = float(normal @ center)
        along = (support - center) @ tangent
        lines.append(dict(edge=edge, endpoints=[a, b], normal=normal.tolist(), offset=offset,
                          support_length_px=float(along.max() - along.min()), queries=ids,
                          support_points=support.tolist(),
                          residual_rms_px=float(np.sqrt(np.mean((support @ normal - offset) ** 2))),
                          correlated_observation_source="one physical edge; sampled queries are not separate 3D corner IDs"))
    lookup = {line["edge"]: line for line in lines}
    corners = []
    for corner in range(8):
        incident = [e for e, (a, b) in enumerate(EDGES) if corner in (a, b) and e in lookup]
        pairs = []
        for a, b in combinations(incident, 2):
            matrix = np.asarray([lookup[a]["normal"], lookup[b]["normal"]])
            determinant = float(np.linalg.det(matrix))
            if abs(determinant) > 1e-6:
                pairs.append((-abs(determinant), a, b, matrix))
        if not pairs:
            continue
        _, a, b, matrix = min(pairs, key=lambda p: p[:3])
        xy = np.linalg.solve(matrix, [lookup[a]["offset"], lookup[b]["offset"]])
        if not np.isfinite(xy).all():
            continue
        corners.append(dict(id=corner, xy=xy.tolist(), edges=[a, b],
                            source="two selected physical-edge support line intersection",
                            extrapolation_possible=True, correlated_edges=[a, b]))
    return dict(corners=corners, lines=lines, queries=queries,
                selected_queries=sum(not q["no_match"] for q in queries), all_no_match=len(lines) == 0)


def parity(original, reproduced):
    maximum = 0.
    signs = 0
    def equal_numbers(a, b):
        nonlocal maximum
        x, y = np.asarray(a, float), np.asarray(b, float)
        assert x.shape == y.shape
        difference = float(np.abs(x - y).max()) if x.size else 0.
        maximum = max(maximum, difference)
        assert difference <= 1e-7, ("old decoder numerical parity", difference)
    assert original["selected_queries"] == reproduced["selected_queries"]
    assert original["all_no_match"] == reproduced["all_no_match"]
    for a, b in zip(original["queries"], reproduced["queries"]):
        assert a["query"] == b["query"] and a["chosen_candidate"] == b["chosen_candidate"] and a["no_match"] == b["no_match"]
        if a["selected_xy"] is None:
            assert b["selected_xy"] is None
        else:
            equal_numbers(a["selected_xy"], b["selected_xy"])
    assert len(original["lines"]) == len(reproduced["lines"])
    for a, b in zip(original["lines"], reproduced["lines"]):
        assert a["edge"] == b["edge"] and a["endpoints"] == b["endpoints"] and a["queries"] == b["queries"]
        equal_numbers(a["support_points"], b["support_points"])
        # A TLS normal and its negation represent the same physical line.
        sign = 1. if np.dot(a["normal"], b["normal"]) >= 0 else -1.
        signs += sign == -1.
        equal_numbers(a["normal"], [sign * n for n in b["normal"]])
        equal_numbers(a["offset"], sign * b["offset"])
        equal_numbers(a["support_length_px"], b["support_length_px"])
        equal_numbers(a["residual_rms_px"], b["residual_rms_px"])
    assert len(original["corners"]) == len(reproduced["corners"])
    for a, b in zip(original["corners"], reproduced["corners"]):
        assert a["id"] == b["id"] and a["edges"] == b["edges"]
        equal_numbers(a["xy"], b["xy"])
    return maximum, int(signs)


def units():
    result = []
    for name, probs, none, expected in (
            ("split_match_mass_0.6_none_0.4", [.3, .3] + [0.] * 63, .4, (65, 0)),
            ("match_mass_0.3_none_0.7", [.15, .15] + [0.] * 63, .7, (65, 65))):
        scores = [math.log(p) if p else -math.inf for p in probs] + [math.log(none)]
        old, _, _, _ = choices(scores, False)
        new, _, _, posterior = choices(scores, True)
        assert (old, new) == expected and math.isclose(posterior, sum(probs), abs_tol=1e-12)
        result.append(dict(case=name, none_probability=none, nonzero_spatial_probabilities=probs[:2],
                           match_probability=sum(probs), old_66way_choice=old, new_mass_choice=new,
                           recovered_match_posterior=posterior, passed=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3], help="repository root")
    parser.add_argument("--output-dir", type=Path, help="new/empty output directory")
    args = parser.parse_args()
    repo = args.root.resolve()
    old = repo / "_docs" / "experiments" / OLD
    output = (args.output_dir or repo / "_docs" / "experiments" / NEW).resolve()
    raw, checks_path = output / "MATCH_MASS_OBSERVATIONS.jsonl.gz", output / "MATCH_MASS_DECODE_CHECKS.json"
    pending_raw, pending_checks = raw.with_suffix(".gz.pending"), checks_path.with_suffix(".json.pending")
    for path in (raw, checks_path, pending_raw, pending_checks):
        if path.exists():
            raise FileExistsError(f"Preserve completed/pending observations: {path.name}")
    start = time.monotonic()
    paths = [old / "LEARNED_OBSERVATIONS.jsonl.gz", old / "INPUTS.json"]
    bindings = [dict(path=str(p.relative_to(repo)), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]
    frames = {f["id"]: f for f in json.loads(paths[1].read_text())["frames"]}
    unit_checks = units()
    decoded_rows, identities = [], set()
    replay_difference = replay_signs = 0
    counts = {a: dict(old=Counter(), new=Counter()) for a in ARMS}
    with gzip.open(paths[0], "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["method"], row["id"])
            assert key not in identities and row["method"] in ARMS and row["id"] in frames
            identities.add(key)
            assert row["GT_input"] is False and row["detector_available"] is True
            assert row["session"] == frames[row["id"]]["session"]
            assert np.array_equal(row["original_base_points"], frames[row["id"]]["points"]["BASE"])
            blob = base64.b64decode(row["candidate_logits_f32_base64"], validate=True)
            assert row["candidate_logits_shape"] == [84, 66] and len(blob) == 84 * 66 * 4
            assert hashlib.sha256(blob).hexdigest() == row["raw_logits_sha256"]
            logits = np.asarray(struct.unpack("<" + "f" * (84 * 66), blob), float).reshape(84, 66)
            assert np.isfinite(logits).all()
            semantic = {k: v for k, v in row.items() if k not in
                        ("candidate_logits_f32_base64", "candidate_logits_shape", "unencoded_semantic_sha256")}
            semantic["queries"] = [dict(q, candidate_logits=logits[i].tolist()) for i, q in enumerate(row["queries"])]
            assert digest(semantic) == row["unencoded_semantic_sha256"]
            old_decode = decode(row, logits, False)
            difference, signs = parity(row, old_decode)
            replay_difference, replay_signs = max(replay_difference, difference), replay_signs + signs
            new_decode = decode(row, logits, True)
            transformed = {k: v for k, v in row.items() if k not in ("corners", "lines", "queries", "selected_queries", "all_no_match", "unencoded_semantic_sha256")}
            transformed.update(schema="match_mass_observation_v1", original_method=row["method"],
                               method=row["method"] + "_MATCH_MASS", **new_decode,
                               decode_policy=POLICY, source_unencoded_semantic_sha256=row["unencoded_semantic_sha256"],
                               raw_logits_unchanged=True, initial_coordinates_not_used_to_fill_absence=True,
                               new_pose_solver_executed=False)
            new_semantic = {k: v for k, v in transformed.items() if k not in
                            ("candidate_logits_f32_base64", "candidate_logits_shape")}
            new_semantic["queries"] = [dict(q, candidate_logits=logits[i].tolist()) for i, q in enumerate(transformed["queries"])]
            transformed["unencoded_semantic_sha256"] = digest(new_semantic)
            decoded_rows.append(transformed)
            h, w = frames[row["id"]]["raw_hw"]
            for phase, observation in (("old", old_decode), ("new", new_decode)):
                cc = counts[row["method"]][phase]
                cc.update(dict(frames=1, accepted_queries=observation["selected_queries"],
                               lines=len(observation["lines"]), corners=len(observation["corners"]),
                               four_raw_corner_frames=int(len(observation["corners"]) >= 4),
                               four_in_image_corner_frames=int(sum(0 <= c["xy"][0] < w and 0 <= c["xy"][1] < h for c in observation["corners"]) >= 4)))
                for q in observation["queries"]:
                    role = row["predicted_roles"][q["edge"]].index(1.)
                    cc["accepted_boundary_queries" if role == 0 else "accepted_internal_queries" if role == 1 else "accepted_unavailable_queries"] += not q["no_match"]
    assert identities == {(a, fid) for a in ARMS for fid in frames} and len(decoded_rows) == 957
    assert all(sha(p) == b["sha256"] for p, b in zip(paths, bindings)), "Original sealed input changed"
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(pending_raw, "wt", encoding="utf-8", compresslevel=6) as f:
        for row in decoded_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    checks = dict(schema="match_mass_decode_checks_v1", complete=True, passed=True,
                  generated_utc=datetime.now(timezone.utc).isoformat(), policy=POLICY,
                  method_mapping={a: a + "_MATCH_MASS" for a in ARMS}, units=unit_checks,
                  original_66way_decoder_replay=dict(rows=957, exact_query_line_corner_identities=True,
                                                   query_support_corner_max_absolute_difference=replay_difference,
                                                   tolerance=1e-7, TLS_normal_sign_equivalences=replay_signs),
                  original_logits_and_semantic_hash_verified_rows=957,
                  counts={a: {phase: dict(c) for phase, c in phases.items()} for a, phases in counts.items()},
                  input_bindings=bindings,
                  observation_binding=dict(path=str(raw.relative_to(repo)) if raw.is_relative_to(repo) else raw.name,
                                           sha256=sha(pending_raw), bytes=pending_raw.stat().st_size, rows=957),
                  implementation=dict(path=str(Path(__file__).resolve().relative_to(repo)), sha256=sha(Path(__file__).resolve()),
                                      dependencies="Python>=3.9 and NumPy; no experiment module imports"),
                  execution=dict(new_detector_forwards=0, new_head_forwards=0, GPU_calls=0, new_pose_solves=0,
                                 new_training_updates=0, image_reads=0, private_GT_reads=0,
                                 old_decoder_replays=957, new_decoder_calls=957,
                                 wall_seconds_through_observation_write=time.monotonic() - start),
                  limits=["This is a predeclared decoder intervention on previously sealed logits, not an independently trained model.",
                          "More accepted queries/corners do not establish correct physical boundaries, consensus or pose improvement.",
                          "Location remains one spatial argmax bin; accepted modes are not averaged.",
                          "Physical visibility, initial-mask accuracy and posterior calibration are not established by this decoder check.",
                          "Final masks and pose evaluation occur separately; no hidden reprojection or Base fill is performed here."])
    pending_checks.write_text(json.dumps(checks, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending_raw.replace(raw)
    pending_checks.replace(checks_path)
    print(json.dumps(dict(passed=True, replay_max_difference=replay_difference, counts=checks["counts"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
