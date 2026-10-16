"""One-shot synthetic CPU checks for the frozen endpoint-heldout wrapper.

Run only after parent review/GO. No detector, head, training, ray, real image,
real reference or latency path is executed. Synthetic truth only constructs
the fixture; no true pose is supplied to either numeric solver.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import inspect
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
import cv2
import numpy as np

from . import common as C
from . import solver as S
from ..pallet_cornerwise_independent_20261010_v4 import pose as P
from ..pallet_partial_line_independent_20261010_v5 import solver as V5
from ..pallet_observation_refiner_20261009_v1.solver import Candidate, cuboid, project

SIZE = (1280, 960)
ATOL = 1e-6


def fixture(square=False, front=False):
    K = np.array([[600., 0., 640.], [0., 600., 480.], [0., 0., 1.]])
    xyz = np.array([1.1, .11, 1.1 if square else 1.3])
    r = np.zeros(3) if front else np.array([.3, .25, .1])
    t = np.array([0., .1, 4.])
    q = np.vstack([project(cuboid(*xyz), cv2.Rodrigues(r)[0], t, K), [641.25, 481.75]])
    return K, xyz, q, np.r_[r, t]


def observation(q, edges=(6,), corners=()):
    lines = []
    for edge in edges:
        a, b = V5.EDGES[edge]
        tangent = (q[b] - q[a]) / np.linalg.norm(q[b] - q[a])
        normal = np.array([-tangent[1], tangent[0]])
        support = np.array([(1-u)*q[a] + u*q[b] for u in (.25, .5, .75)])
        lines.append(dict(edge=edge, endpoints=[a, b], normal=normal.tolist(),
            offset=float(normal @ q[a]), support_points=support.tolist(),
            query_radii_px=[1., 1., 1.], queries=[edge*7+j for j in range(3)],
            parameter_covariance=[[.001, 0.], [0., .1]]))
    records = []
    for k in corners:
        incident = [edge for edge in edges if k in V5.EDGES[edge]]
        assert len(incident) >= 2
        records.append(dict(id=k, xy=q[k].tolist(), edges=incident[:2], radius_px=.5))
    return dict(lines=lines, corners=records, partial_lines=[], queries=[], selected_queries=3*len(lines))


class AuditedBank(P.PoseBank):
    def __init__(self, *args, **kwargs):
        self.trace, self.solve_requests = [], []
        super().__init__(*args, **kwargs)

    def _generic(self, ids, dim, phase):
        key = (int(dim), tuple(sorted(ids)))
        self.trace.append(dict(kind="generic", ids=list(ids), dimension=dim, phase=phase,
                               cached=key in self.numeric_cache))
        return super()._generic(ids, dim, phase)

    def _lm(self, candidate, ids):
        self.trace.append(dict(kind="LM", ids=list(ids), dimension=candidate.dim))
        return super()._lm(candidate, ids)

    def solve(self, excluded=(), hidden=(), robust=True):
        before = len(self.trace)
        solved = super().solve(excluded=excluded, hidden=hidden, robust=robust)
        blocked = set(excluded) | set(hidden)
        for event in self.trace[before:]:
            if event["kind"] == "LM" or event.get("phase") != "subset":
                assert not set(event["ids"]) & blocked, event
        audit_packet(solved, blocked)
        self.solve_requests.append(dict(excluded=list(excluded), hidden=list(hidden), robust=robust))
        return solved


def audit_packet(solved, blocked):
    for key in ("used", "fit_input_ids", "final_inliers", "generator_ids"):
        assert not set(solved.get(key, [])) & set(blocked), key
    for row in solved.get("all_candidate_solutions", []):
        for key in ("generator_ids", "actual_fit_input_ids", "inlier_ids"):
            assert not set(row[key]) & set(blocked), key
        assert len(row["residuals_used_px"]) == len(solved["used"])
    for key in ("prior_used", "initial_pose_used", "initial_projection_used",
                "initial_dimension_prior_used", "known_dimension_constraint_used",
                "excluded_image_coordinates_used_for_scoring", "excluded_image_coordinates_used_for_equivalence",
                "reprojected_points_reused_as_observations"):
        assert solved[key] is False, key


def audit_validation(result):
    v = result["endpoint_validation"]
    assert [r["edge"] for r in v["records"]] == v["original_unused_edges"]
    assert v["attempted_pose_calls"] == len(v["records"])
    assert v["retained_edges"] == sorted(set(v["original_unused_edges"]) - set(v["rejected_edges"]))
    assert not set(v["consumed_edges"]) & set(v["rejected_edges"])
    assert result["line_edges"] == v["retained_edges"]
    for record in v["records"]:
        blocked = set(record["hidden"]) | set(record["temporary_excluded"]) | set(record["endpoints"])
        assert record["attempted_pose_calls"] == 1
        assert set(record["validation_excluded"]) == blocked
        assert set(record["validator_temporary_excluded"]) == set(record["temporary_excluded"]) | set(record["endpoints"])
        audit_packet(record["solved"], blocked)
        assert record["bank_delta"] == {k: record["bank_after"][k] - record["bank_before"][k] for k in record["bank_after"]}
        if record["decision"] == "UNVERIFIED_RETAIN":
            assert record["edge"] in v["retained_edges"]
        else:
            assert record["solved"]["available"]
            q = np.asarray(record["solved"]["projected"])
            line = record["source_line"]
            d = q[record["endpoints"]] @ np.asarray(line["normal"]) - line["offset"]
            assert np.allclose(d, record["signed_endpoint_distances_px"], atol=0, rtol=0)
            rms = float(np.sqrt(np.mean(d**2)))
            assert record["endpoint_RMS_px"] == rms
            assert record["decision"] == ("SUPPORTED_RETAIN" if rms <= 8 else "CONTRADICTED_REJECT")
    aggregate = {k: sum(delta[k] for delta in v["bank_delta"].values()) for k in v["validation_bank_delta"]}
    for key, value in aggregate.items():
        assert result["operation_counts"][key] == value, key
    assert v["original_observation_preserved"]
    assert v["observed_line_used_for_validation_pose_choice"] is False
    assert v["validation_projections_added_to_observations"] is False


def same_pose(a, b):
    for key in ("available", "state", "used", "fit_input_ids", "final_inliers", "selected_hypothesis"):
        assert a.get(key) == b.get(key), key
    for key in ("R_cf", "R_physical", "centroid", "cf_extents", "projected", "residuals_used_px"):
        if a.get(key) is not None:
            assert np.allclose(a[key], b[key], rtol=0, atol=ATOL), key
    for key in ("truncated_sse_px2", "sse_px2"):
        if key in a:
            assert abs(a[key] - b[key]) <= ATOL**2, key


@contextmanager
def forbidden_reads():
    import builtins
    import io
    saved = [(builtins, "open", builtins.open), (io, "open", io.open), (Path, "open", Path.open)]
    tokens = ("TARGETS", "GROUND_TRUTH", "READY_SOURCE", "FEATURES.NPY", "BEST.PT", "LAST.PT", "IMAGE_ROLE.PT",
              "PREDICTIONS.JSON", "OBSERVATIONS.JSON", "INPUTS.JSON", "STATIC_VISIBILITY")
    attempts = []
    def wrap(fn):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and any(s in str(path).upper() for s in tokens):
                attempts.append(str(path)); raise AssertionError("Forbidden cache/model/GT read")
            return fn(path, *args, **kwargs)
        return call
    for obj, name, fn in saved:
        setattr(obj, name, wrap(fn))
    try:
        yield attempts
    finally:
        for obj, name, fn in saved:
            setattr(obj, name, fn)


def run():
    cv2.setNumThreads(1)
    checks, actual = [], Counter()
    names = ("solvePnP", "solvePnPGeneric", "solvePnPRefineLM", "projectPoints")
    original = {name: getattr(cv2, name) for name in names}
    original_optimizer = V5.least_squares
    for name, fn in original.items():
        def counted(*args, _name=name, _fn=fn, **kwargs):
            actual[_name] += 1
            return _fn(*args, **kwargs)
        setattr(cv2, name, counted)
    def optimizer(*args, **kwargs):
        actual["scipy_least_squares"] += 1
        return original_optimizer(*args, **kwargs)
    V5.least_squares = optimizer

    def check(name, fn):
        before = actual.copy()
        try:
            with forbidden_reads() as reads:
                details = fn() or {}
            assert not reads
            checks.append(dict(name=name, status="PASS", details=C.finite(details), actual_calls=dict(actual-before),
                               forbidden_read_attempts=reads))
        except Exception as error:
            checks.append(dict(name=name, status="FAIL", error=type(error).__name__, message=str(error),
                               traceback=traceback.format_exc(), actual_calls=dict(actual-before)))

    def API_and_bank_guards():
        for fn in (S.EndpointValidatedPointLineBank, S.EndpointValidatedPointLineBank.solve):
            assert "initial_pose" not in inspect.signature(fn).parameters
            assert "known_dimension_index" not in inspect.signature(fn).parameters
        K, xyz, q, _ = fixture()
        b = P.PoseBank(q, K, xyz, SIZE)
        engine = S.EndpointValidatedPointLineBank(q, K, xyz, SIZE, native_points=q, native_bank=b, final_bank=b)
        assert engine.native_bank is engine.final_bank
        try:
            S.EndpointValidatedPointLineBank(q, K, xyz, SIZE, native_points=q, native_bank=b,
                                             final_bank=P.PoseBank(q, K, xyz, SIZE))
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate same-coordinate numeric banks were accepted")
        constrained = P.PoseBank(q, K, xyz, SIZE, known_dimension_index=0,
            known_dimension_provenance=dict(source="synthetic-only declaration", independent_of_initial_pose=True))
        try:
            S.EndpointValidatedPointLineBank(q, K, xyz, SIZE, native_points=q, native_bank=constrained)
        except ValueError:
            pass
        else:
            raise AssertionError("A dimension-constrained native validator was accepted")
        assert S.POLICY["endpoint_RMS_threshold_px"] == 8.
        return dict(no_initial_pose_API=True, no_known_dimension=True, same_coordinate_bank_reuse_required=True)

    def actual_both_endpoint_exclusion():
        K, xyz, q, _ = fixture(); obs = observation(q, edges=(0, 6, 9))
        b = AuditedBank(q, K, xyz, SIZE)
        result = S.EndpointValidatedPointLineBank(q, K, xyz, SIZE, native_points=q, native_bank=b, final_bank=b).solve(
            obs, hidden=(7,), excluded=(6,))
        audit_validation(result)
        assert len(b.solve_requests) == 3
        assert any(r["solved"]["available"] for r in result["endpoint_validation"]["records"])
        for request, record in zip(b.solve_requests, result["endpoint_validation"]["records"]):
            assert request["excluded"] == record["validator_temporary_excluded"]
            assert request["hidden"] == [7] and request["robust"] is True
        return dict(requests=b.solve_requests, scoring_generator_and_LM_exclusion=True,
                    global_unused_subset_build_distinguished=True)

    def supported_vs_contradicted_no_line_pose_choice():
        K, xyz, q, _ = fixture(); obs = observation(q)
        answers = []
        for shift in (0., 100.):
            current = copy.deepcopy(obs)
            line = current["lines"][0]
            line["offset"] += shift
            line["support_points"] = (np.asarray(line["support_points"]) + shift*np.asarray(line["normal"])).tolist()
            before = S.observation_binding(current)
            result = S.EndpointValidatedPointLineBank(q, K, xyz, SIZE, native_points=q).solve(current)
            audit_validation(result)
            assert S.observation_binding(current) == before
            record = result["endpoint_validation"]["records"][0]
            assert record["solved"]["available"], record["solved"]["state"]
            assert record["decision"] == ("SUPPORTED_RETAIN" if shift == 0 else "CONTRADICTED_REJECT")
            if shift:
                assert result["line_edges"] == [] and result["empty_line_pool_exact_point_delegate"]
            answers.append(result)
        same_pose(answers[0]["endpoint_validation"]["records"][0]["solved"],
                  answers[1]["endpoint_validation"]["records"][0]["solved"])
        return dict(edge=6, shifts_px=[0, 100], identical_validation_pose_and_score=True,
                    known_contradiction_rejects_only_line=True)

    def insufficient_validator_retains_and_final_can_solve():
        K, xyz, q, _ = fixture(); native = q.copy(); native[[3, 5, 6, 7]] = np.nan
        obs = observation(q, edges=(0,))
        result = S.EndpointValidatedPointLineBank(q, K, xyz, SIZE, native_points=native).solve(obs)
        audit_validation(result)
        record = result["endpoint_validation"]["records"][0]
        assert record["decision"] == "UNVERIFIED_RETAIN"
        assert record["solved"]["state"] == "INSUFFICIENT_OBSERVATIONS"
        assert len(record["solved"]["used"]) == 2
        assert record["bank_delta"]["generic_calls"] == record["bank_delta"]["lm_calls"] == 0
        assert result["available"], result["state"]
        sparse = S.EndpointValidatedPointLineBank(native, K, xyz, SIZE, native_points=native).solve(obs, hidden=(4,))
        assert not sparse["available"] and sparse["state"] == "SUPPLEMENTAL_LINE_UNAVAILABLE"
        assert sparse["endpoint_validation"]["records"][0]["decision"] == "UNVERIFIED_RETAIN"
        return dict(validator_remaining_points=2, attempted_request=1, unavailable_line_retained=True,
                    sufficient_final_actual_points_can_still_solve=True, sparse_final_explicitly_unavailable=True)

    def missing_witness_retains():
        class MissingWitness(AuditedBank):
            def solve(self, *args, **kwargs):
                result = super().solve(*args, **kwargs)
                if result["available"]:
                    result.pop("projected")
                return result
        K, xyz, q, _ = fixture(); b = MissingWitness(q, K, xyz, SIZE)
        result = S.EndpointValidatedPointLineBank(q, K, xyz, SIZE, native_points=q, native_bank=b, final_bank=b).solve(observation(q))
        record = result["endpoint_validation"]["records"][0]
        assert record["solved"]["available"]
        assert record["decision"] == "UNVERIFIED_RETAIN" and record["endpoint_RMS_px"] is None
        assert result["line_edges"] == [6] and result["endpoint_validation"]["rejected_edges"] == []
        return dict(actual_pose_call=True, deliberately_missing_projection=True, unknown_retained=True)

    def consumed_support_and_raw_immutability():
        K, xyz, q, _ = fixture()
        obs = observation(q, edges=(0, 8, 9), corners=(0,))
        # Only edge 9 is unused. Shift it as a coherent bad observed line;
        # preserve both support lines used by the actual adopted corner 0.
        line = obs["lines"][-1]
        line["offset"] += 100
        line["support_points"] = (np.asarray(line["support_points"]) + 100*np.asarray(line["normal"])).tolist()
        obs["lines"].append(copy.deepcopy(obs["lines"][-1]))
        saved = copy.deepcopy(obs)
        result = S.EndpointValidatedPointLineBank(q, K, xyz, SIZE, native_points=q).solve(obs, adopted_boundary_corner_ids=(0,))
        audit_validation(result)
        v = result["endpoint_validation"]
        assert v["consumed_edges"] == [0, 8]
        assert v["original_unused_edges"] == [9] and v["attempted_pose_calls"] == 1
        assert v["rejected_edges"] == [9] and v["filtered_raw_line_edges"] == [0, 8]
        assert result["empty_line_pool_exact_point_delegate"]
        assert obs == saved and obs["corners"][0]["edges"] == [0, 8]
        return dict(consumed_unvalidated_and_preserved=[0, 8], one_duplicate_edge_validation=9,
                    original_query_corner_line_records_unchanged=True, rejected_all_duplicate_records=True)

    def invalid_support_does_not_generate_validator():
        K, xyz, q, _ = fixture(); obs = observation(q, edges=(0, 8), corners=(0,))
        line = obs["lines"][0]
        line["support_points"][0] = (np.asarray(line["support_points"][0]) + 30*np.asarray(line["normal"])).tolist()
        b = AuditedBank(q, K, xyz, SIZE); before = actual.copy()
        result = S.EndpointValidatedPointLineBank(q, K, xyz, SIZE, native_points=q, native_bank=b, final_bank=b).solve(
            obs, adopted_boundary_corner_ids=(0,))
        assert result["state"] == "INVALID_OBSERVATION_CONTRACT"
        assert result["reason"] == "adopted_boundary_has_inconsistent_final_support_line"
        assert result["endpoint_validation"]["records"] == [] and not b.solve_requests
        assert not (actual-before)
        return dict(invalid_consumed_support_fails_closed=True, no_endpoint_pose_or_optimizer_called=True)

    def shared_cache_and_nonoverlapping_counts():
        K, xyz, q, _ = fixture(); b = AuditedBank(q, K, xyz, SIZE)
        engine = S.EndpointValidatedPointLineBank(q, K, xyz, SIZE, native_points=q, native_bank=b, final_bank=b)
        obs = observation(q, edges=tuple(sorted(V5.SOURCE_REGISTRY_EDGE_IDS)))
        first = engine.solve(obs); audit_validation(first)
        assert len(first["endpoint_validation"]["records"]) == 8
        old = b.ledger.copy()
        second = engine.solve(obs, hidden=(7,)); audit_validation(second)
        assert b.ledger["subsets_considered"] == old["subsets_considered"] == 70*len(b.dims)
        assert b.ledger["subset_generic_calls"] <= 70*len(b.dims)
        assert second["endpoint_validation"]["unique_numeric_bank_count"] == 1
        v = second["endpoint_validation"]
        assert v["bank_aliases"] == {"native": "native", "final": "native"}
        for key in b.ledger:
            assert v["bank_delta"]["native"][key] == v["validation_bank_delta"][key] + v["final_bank_delta"][key]
            assert v["validation_bank_delta"][key] == sum(r["bank_delta"][key] for r in v["records"])
        misses = [event for event in b.trace if event["kind"] == "generic" and not event["cached"] and len(event["ids"]) == 4]
        assert len({(event["dimension"], tuple(event["ids"])) for event in misses}) == len(misses)
        assert len(misses) <= 70*len(b.dims)
        return dict(two_masks_same_bank=True, ledger=dict(b.ledger), actual_four_ID_numeric_misses=len(misses),
                    total_Generic_can_include_separate_five_or_six_ID_refit_problems=True,
                    validation_and_final_ledger_deltas_added_once=True)

    def excluded_coordinate_invariance():
        K, xyz, q, _ = fixture(); obs = observation(q); answers = []
        for values in (q[[6, 7]], np.array([[20., 30.], [1230., 910.]]), np.full((2, 2), -1.), np.full((2, 2), np.nan)):
            changed = q.copy(); changed[[6, 7]] = values
            answer = S.EndpointValidatedPointLineBank(changed, K, xyz, SIZE, native_points=changed).solve(
                obs, hidden=(7,), excluded=(6,))
            audit_validation(answer)
            record = answer["endpoint_validation"]["records"][0]
            assert record["decision"] == "SUPPORTED_RETAIN" and record["solved"]["used"] == list(range(6))
            assert answer["available"], answer["state"]
            assert np.array_equal(np.asarray(answer["points_final"])[6], changed[6], equal_nan=True)
            assert np.array_equal(np.asarray(answer["points_final"])[8], changed[8])
            assert np.array_equal(np.asarray(answer["points_final"])[7], np.asarray(answer["projected"])[7])
            if answers:
                same_pose(answers[0], answer)
                same_pose(answers[0]["endpoint_validation"]["records"][0]["solved"], record["solved"])
            answers.append(answer)
        return dict(variants=4, H=[7], caller_temporary=[6], edge_endpoints=[6, 7],
                    active_numeric_pose_and_endpoint_decision_invariant=True,
                    unused_global_bank_counts_may_change=True,
                    hidden_replaced_temporary_and_center_preserved=True, no_projection_refit=True)

    def actual_planar_four_branches_and_five():
        K, xyz, q, _ = fixture(square=True); cases = []
        for H in ((4, 5), (5,)):
            result = S.EndpointValidatedPointLineBank(q, K, xyz, SIZE, native_points=q).solve(observation(q), hidden=H)
            audit_validation(result)
            record = result["endpoint_validation"]["records"][0]
            solved = record["solved"]
            assert len(solved["used"]) == (4 if len(H) == 2 else 5)
            assert solved["all_candidate_solutions"]
            assert solved["state"] in ("NEW_POSE", "AMBIGUOUS_PNP")
            if len(H) == 2:
                assert any(c["generator"].startswith("IPPE_PLANE") for c in solved["all_candidate_solutions"])
                assert len(solved["all_candidate_solutions"]) >= 2
            cases.append(dict(H=list(H), used=solved["used"], state=solved["state"],
                              actual_numeric_branches=len(solved["all_candidate_solutions"])))
        return dict(actual_numeric_cases=cases, known_dimension_not_supplied=True, rank_is_not_global_uniqueness=True)

    def controlled_ambiguity_retains():
        K, xyz, q, z = fixture(front=True)
        b = AuditedBank(q, K, xyz, SIZE)
        branches = []
        for dx in (-.02, .02):
            zz = z.copy(); zz[3] += dx
            branches.append(Candidate(0, (0, 1, 2, 4), 0, "CONTROLLED_TIE", zz[:3].reshape(3, 1),
                zz[3:].reshape(3, 1), project(b.models[0], cv2.Rodrigues(zz[:3])[0], zz[3:], K)))
        b.hypotheses = branches; b._build = lambda: None
        b._generic = lambda *args, **kwargs: []
        b._lm = lambda *args, **kwargs: None
        # A separate selected-coordinate bank lets the unchanged final solver
        # execute its real numeric path; the controlled validator is isolated.
        selected = q.copy(); selected[0] += [1e-3, 0.]
        final = P.PoseBank(selected, K, xyz, SIZE)
        result = S.EndpointValidatedPointLineBank(selected, K, xyz, SIZE,
            native_points=q, native_bank=b, final_bank=final).solve(observation(q))
        audit_validation(result)
        record = result["endpoint_validation"]["records"][0]
        assert record["solved"]["state"] == "AMBIGUOUS_PNP" and record["solved"]["unresolved_ambiguity"]
        assert record["decision"] == "UNVERIFIED_RETAIN" and result["line_edges"] == [6]
        return dict(controlled_numeric_tie_candidates=2, validator_actual_Generic_LM=False,
                    validator_local_rank_calculated=True, actual_final_numeric_path=True,
                    ambiguous_validator_is_not_no_match_or_frame_veto=True)

    try:
        for name, fn in (("prior_free_API_and_shared_bank_guards", API_and_bank_guards),
            ("actual_both_endpoints_H_and_temporary_absent_from_active_numeric_paths", actual_both_endpoint_exclusion),
            ("same_native_pose_support_or_contradiction_without_line_pose_choice", supported_vs_contradicted_no_line_pose_choice),
            ("insufficient_endpoint_validation_retains_line_and_sufficient_final_points", insufficient_validator_retains_and_final_can_solve),
            ("missing_projection_witness_is_unverified_retained", missing_witness_retains),
            ("consumed_support_raw_immutability_and_duplicate_edge_once", consumed_support_and_raw_immutability),
            ("invalid_consumed_support_prevents_validation_and_fit", invalid_support_does_not_generate_validator),
            ("shared_four_ID_cache_and_unique_bank_operation_accounting", shared_cache_and_nonoverlapping_counts),
            ("wrong_inframe_sentinel_NaN_endpoint_coordinate_invariance", excluded_coordinate_invariance),
            ("actual_planar_four_multiple_branches_and_five_point_validation", actual_planar_four_branches_and_five),
            ("controlled_equal_score_physical_ambiguity_remains_unknown", controlled_ambiguity_retains)):
            check(name, fn)
    finally:
        for name, fn in original.items():
            setattr(cv2, name, fn)
        V5.least_squares = original_optimizer
    return dict(schema="endpoint_validation_CPU_checks_v6", complete=True,
        passed=all(c["status"] == "PASS" for c in checks), checks=checks, test_groups=len(checks),
        actual_calls=dict(actual), actual_model_forwards=0, actual_training_updates=0,
        actual_rays=0, actual_real_pose_fits=0, actual_latency_measurements=0,
        code={name: C.binding(path) for name, path in dict(solver=Path(S.__file__), test=Path(__file__),
            frozen_v4_pose=Path(P.__file__), frozen_v5_solver=Path(V5.__file__),
            original_solver=Path(sys.modules[Candidate.__module__].__file__)).items()},
        limits=["synthetic fixtures do not certify real IMAGE_ROLE boundary ownership or improved accuracy",
                "rank six and finite numerical tie handling do not prove global pose uniqueness",
                "a wrong native consensus can support a wrong line or contradict a correct line",
                "unavailable endpoint validation intentionally retains unknown lines",
                "H and image-observation proposal generation are outside this conditional numeric exclusion check"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output")
    args = p.parse_args()
    expected = C.DOC / "ENDPOINT_VALIDATION_CHECKS.json"
    output = Path(args.output) if args.output else expected
    started = C.DOC / "ENDPOINT_VALIDATION_CHECKS_STARTED.json"
    if (output.resolve() != expected.resolve() or output.is_symlink() or C.DOC.is_symlink()
            or output.exists() or started.exists()):
        raise RuntimeError("Preserve the one-shot CPU attempt; only the new exclusive receipt is allowed")
    C.write_new(started, dict(schema="endpoint_validation_CPU_started_v6", actual_model_forwards=0,
        code={"solver": C.binding(S.__file__), "test": C.binding(__file__)}))
    result = run()
    C.write_new(output, result)
    print("ENDPOINT_VALIDATION_CHECKS", "PASS" if result["passed"] else "FAIL", result["test_groups"], result["actual_calls"])
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
