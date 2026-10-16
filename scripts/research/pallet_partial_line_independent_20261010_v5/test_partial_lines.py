"""Frozen mathematical CPU checks; run only after parent review/GO.

No model, image, source target, GPU, training, ray, real fit or timing input.
Synthetic truth constructs the test fixture only; no solver sees that pose.
Controlled ties are labeled separately from actual numerical branch tests.
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
from ..pallet_observation_refiner_20261009_v1.solver import Candidate, cuboid, project

SIZE = (1280, 960)
ATOL = 1e-6


def fixture(square=False):
    K = np.array([[600., 0., 640.], [0., 600., 480.], [0., 0., 1.]])
    xyz = np.array([1.1, .11, 1.1 if square else 1.3])
    r = np.array([.3, .25, .1]); t = np.array([0., .1, 4.])
    q = np.vstack([project(cuboid(*xyz), cv2.Rodrigues(r)[0], t, K), [641.25, 481.75]])
    return K, xyz, q, np.r_[r, t]


def observation(q, edges=tuple(sorted(S.SOURCE_REGISTRY_EDGE_IDS)), corners=()):
    lines = []
    for e in edges:
        a, b = S.EDGES[e]; pa, pb = q[a], q[b]
        tangent = (pb - pa) / np.linalg.norm(pb - pa)
        normal = np.array([-tangent[1], tangent[0]])
        support = np.array([(1-u)*pa + u*pb for u in (.25, .5, .75)])
        lines.append(dict(edge=e, endpoints=[a, b], normal=normal.tolist(), offset=float(normal@pa),
            support_points=support.tolist(), query_radii_px=[1., 1., 1.], queries=[e*7+j for j in range(3)],
            support_length_px=float(np.linalg.norm(pb-pa)), parameter_covariance=[[.001, 0.], [0., .1]]))
    records = []
    for k in corners:
        incident = [e for e in edges if k in S.EDGES[e]]
        assert len(incident) >= 2
        records.append(dict(id=k, xy=q[k].tolist(), edges=incident[:2], radius_px=.5))
    # This deliberately disagrees with the full line pool. The C2 solver must
    # reconstruct unused factors from actually adopted corners, not this field.
    return dict(lines=lines, corners=records, partial_lines=[])


def candidate(dim, ids, z, model, K, label="CONTROLLED"):
    return Candidate(dim, tuple(ids), 0, label, z[:3].reshape(3, 1), z[3:].reshape(3, 1),
                     project(model, cv2.Rodrigues(z[:3])[0], z[3:], K))


def line_factors(obs):
    return [dict(edge=l["edge"], normal=np.array(l["normal"]), offset=l["offset"]) for l in obs["lines"]]


def same(a, b, exact=False):
    for k in ("available", "state", "used", "fit_input_ids", "final_inliers", "selected_hypothesis"):
        assert a.get(k) == b.get(k), k
    for k in ("R_cf", "R_physical", "centroid", "cf_extents", "projected", "residuals_used_px"):
        if a.get(k) is not None:
            assert np.allclose(a[k], b[k], rtol=0, atol=0 if exact else ATOL), k
    for k in ("truncated_sse_px2", "sse_px2"):
        if k in a:
            assert abs(a[k] - b[k]) <= (0 if exact else ATOL**2), k


def audit_result(result, hidden=(), excluded=()):
    blocked = set(hidden) | set(excluded)
    for key in ("used", "fit_input_ids", "final_inliers", "generator_ids"):
        assert set(result.get(key, [])).isdisjoint(blocked), key
    for row in result.get("all_candidate_solutions", []):
        assert set(row["generator_ids"]).isdisjoint(blocked)
        assert set(row["actual_fit_input_ids"]).isdisjoint(blocked)
        if "same_scoring_line_edges" in row:
            assert row["same_scoring_line_edges"] == result["line_edges"]
            assert row["same_scoring_point_ids"] == result["used"]
            assert row["total_inlier_factor_count"] == row["point_inlier_count"] + row["line_inlier_count"]
            assert not set(row["actual_fit_line_edges"]) & set(result["consumed_edges"])
    for attempt in result.get("optimizer_attempts", []):
        assert set(attempt["fit_point_ids"]).isdisjoint(blocked)
        assert len(attempt["fit_point_ids"]) >= 4
        assert len(attempt["fit_line_edges"]) == len(set(attempt["fit_line_edges"]))
    for key in ("prior_used", "initial_pose_used", "initial_projection_used", "initial_dimension_prior_used",
                "known_dimension_constraint_used", "excluded_image_coordinates_used_for_scoring",
                "excluded_image_coordinates_used_for_equivalence", "reprojected_points_reused_as_observations"):
        assert result[key] is False, key
    if result["available"]:
        assert len(result["final_inliers"]) >= 4
        assert result["global_uniqueness_proven"] is False


@contextmanager
def forbidden_reads():
    import builtins
    import io
    names = ("TARGETS", "GROUND_TRUTH", "READY_SOURCE", "FEATURES.NPY", "IMAGE_ROLE.PT", "BEST.PT", "LAST.PT",
             "PREDICTIONS.JSON", "OBSERVATIONS.JSON", "INPUTS.JSON")
    saved = [(builtins, "open", builtins.open), (io, "open", io.open), (Path, "open", Path.open)]
    attempts = []
    def wrap(fn):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and any(n in str(path).upper() for n in names):
                attempts.append(str(path)); raise AssertionError("Forbidden model/cache/GT read")
            return fn(path, *args, **kwargs)
        return call
    for obj, name, fn in saved:
        setattr(obj, name, wrap(fn))
    try:
        yield attempts
    finally:
        for obj, name, fn in saved:
            setattr(obj, name, fn)


def run(attempt=1):
    cv2.setNumThreads(1)
    checks, actual = [], Counter()
    names = ("solvePnP", "solvePnPGeneric", "solvePnPRefineLM", "projectPoints")
    original = {k: getattr(cv2, k) for k in names}
    original_optimizer = S.least_squares
    for name, fn in original.items():
        def counted(*args, _name=name, _fn=fn, **kwargs):
            actual[_name] += 1
            return _fn(*args, **kwargs)
        setattr(cv2, name, counted)
    def optimizer(*args, **kwargs):
        actual["scipy_least_squares"] += 1
        return original_optimizer(*args, **kwargs)
    S.least_squares = optimizer

    def check(name, fn):
        before = actual.copy()
        try:
            with forbidden_reads() as reads:
                details = fn() or {}
            assert not reads
            checks.append(dict(name=name, status="PASS", details=C.finite(details),
                actual_calls=dict(actual-before), forbidden_read_attempts=reads))
        except Exception as error:
            checks.append(dict(name=name, status="FAIL", error=type(error).__name__, message=str(error),
                traceback=traceback.format_exc(), actual_calls=dict(actual-before)))

    def analytic_jacobian():
        K, xyz, q, z = fixture(); model = cuboid(*xyz)
        obs = observation(q); lines = line_factors(obs)
        res, J, uv = S.residual_jacobian(model, z, K, q, (0, 1, 2, 4, 6), lines)
        numeric = np.empty_like(J)
        h = 1e-6
        for k in range(6):
            step = np.zeros(6); step[k] = h
            numeric[:, k] = (S.residual_jacobian(model, z+step, K, q, (0, 1, 2, 4, 6), lines)[0]
                             - S.residual_jacobian(model, z-step, K, q, (0, 1, 2, 4, 6), lines)[0]) / (2*h)
        assert np.max(np.abs(res)) < ATOL
        assert np.allclose(uv, q[:8], rtol=0, atol=ATOL)
        assert np.allclose(J, numeric, rtol=1e-5, atol=1e-5)
        return dict(max_abs_J_difference=float(np.max(np.abs(J-numeric))), scalar_rows=len(res),
                    rotation_units="radians", translation_units="meters", residual_units="pixels")

    def adopted_and_duplicates():
        K, xyz, q, _ = fixture(); obs = observation(q, corners=(0, 1))
        bank = S.PointLineBank(q, K, xyz, SIZE)
        ls, consumed, _ = bank._lines(obs, (0,), set(range(8)))
        assert consumed == [0, 8]
        assert [l["edge"] for l in ls] == [2, 4, 6, 9, 10, 11]
        assert 9 in [l["edge"] for l in ls]  # Raw but unadopted corner 1 cannot consume edge 9.
        dupe = copy.deepcopy(obs); dupe["lines"].append(copy.deepcopy(dupe["lines"][0]))
        l2, c2, _ = bank._lines(dupe, (0,), set(range(8)))
        assert len(l2) == len(ls) and c2 == consumed
        bad = copy.deepcopy(dupe); bad["lines"][-1]["offset"] += 1
        before = bank.counts.copy(); rejected = bank.solve(bad, adopted_boundary_corner_ids=(0,))
        assert rejected["state"] == "INVALID_OBSERVATION_CONTRACT" and bank.counts == before
        bad = copy.deepcopy(obs)
        bad["lines"][0]["support_points"][0] = (np.asarray(bad["lines"][0]["support_points"][0], float)
                                                  + 30*np.asarray(bad["lines"][0]["normal"], float)).tolist()
        rejected = bank.solve(bad, adopted_boundary_corner_ids=(0,))
        assert rejected["state"] == "INVALID_OBSERVATION_CONTRACT"
        assert rejected["reason"] == "adopted_boundary_has_inconsistent_final_support_line"
        bad = copy.deepcopy(obs); bad["lines"][0]["endpoints"] = [1, 0]
        assert bank.solve(bad)["state"] == "INVALID_OBSERVATION_CONTRACT"
        return dict(actual_adopted=[0], consumed=consumed, recovered_unadopted_edge=9,
                    raw_partial_lines_ignored=True, inconsistent_duplicate_rejected=True,
                    consumed_support_inconsistency_rejected=True)

    def support_replication():
        K, xyz, q, z = fixture(); obs = observation(q); doubled = copy.deepcopy(obs)
        for l in doubled["lines"]:
            for key in ("support_points", "query_radii_px", "queries"):
                l[key] *= 2
        bank = S.PointLineBank(q, K, xyz, SIZE)
        la, _, _ = bank._lines(obs, (), set(range(8)))
        lb, _, _ = bank._lines(doubled, (), set(range(8)))
        c = candidate(0, range(4), z, bank.bank.models[0], K)
        a, b = bank._score(c, tuple(range(8)), la, True), bank._score(c, tuple(range(8)), lb, True)
        assert a[0] == b[0] and a[1:3] == b[1:3] and len(la) == len(lb) == 8
        assert np.array_equal(a[4], b[4])
        return dict(unique_line_factors=8, support_queries_before=24, after=48, score_unchanged=True)

    def hidden_invariance():
        K, xyz, q, _ = fixture(); obs = observation(q)
        original_obs = copy.deepcopy(obs); results = []
        for value in (q[[6, 7]], np.array([[777., 555.], [999., 444.]]), np.full((2, 2), -1.), np.full((2, 2), np.nan)):
            changed = q.copy(); changed[[6, 7]] = value
            r = S.PointLineBank(changed, K, xyz, SIZE).solve(obs, hidden=(7,), excluded=(6,))
            audit_result(r, (7,), (6,)); assert r["available"], r["state"]
            assert 11 in r["line_edges"]  # Edge endpoint 7 belongs to H.
            assert np.array_equal(np.asarray(r["points_final"])[8], q[8])
            assert np.array_equal(np.asarray(r["points_final"])[6], changed[6], equal_nan=True)
            assert np.allclose(np.asarray(r["points_final"])[7], np.asarray(r["projected"])[7], atol=0, rtol=0)
            results.append(r)
        for r in results[1:]: same(results[0], r)
        assert obs == original_obs
        return dict(H=[7], temporary=[6], variants=4, eligible_bank_counts_may_change=True,
                    accepted_fit_and_score_invariant=True, visible_H_endpoint_line_retained=11,
                    no_reprojection_refit=True)

    def low_point_and_degenerate():
        K, xyz, q, _ = fixture(); obs = observation(q)
        sparse = q.copy(); sparse[3:8] = np.nan
        bank = S.PointLineBank(sparse, K, xyz, SIZE); before = actual.copy()
        r = bank.solve(obs)
        assert r["state"] == "SUPPLEMENTAL_LINE_UNAVAILABLE" and not r["available"]
        assert not (actual-before) and bank.counts["point_line_optimizer_calls"] == 0
        bad = q.copy(); bad[:8] = [640., 480.]
        r = S.PointLineBank(bad, K, xyz, SIZE).solve(obs)
        assert r["state"] == "DEGENERATE_OBSERVATIONS"
        return dict(actual_points=3, unique_lines=8, no_numeric_initializer_calls=True,
                    independent_global_PnL_not_implemented=True)

    def modeled_rank_guard():
        K, xyz, q, z = fixture(); model = cuboid(*xyz)
        parallel = line_factors(observation(q, edges=(0, 2, 4)))
        noisy = copy.deepcopy(parallel)
        for l, angle in zip(noisy, (.01, -.02, .03)):
            n = l["normal"]; c, s = np.cos(angle), np.sin(angle)
            l["normal"] = np.array([[c, -s], [s, c]]) @ n
        geo = S.geometry_info(model, z, K, q, (), noisy)
        assert geo["modeled_normal_lines_only"]["numerical_rank"] == 5
        assert geo["observed_normal_lines_only"]["numerical_rank"] == 6
        two = S.geometry_info(model, z, K, q, (), line_factors(observation(q, edges=(0, 9, 11))))
        assert two["modeled_normal_lines_only"]["numerical_rank"] == 6
        assert geo["global_unique_pose_proven"] is False
        return dict(parallel_edges=[0, 2, 4], noisy_observed_rank=6, exact_modeled_rank=5,
                    two_family_edges=[0, 9, 11], two_family_modeled_rank=6, local_only=True)

    def empty_pool_delegate():
        K, xyz, q, _ = fixture()
        details = []
        for robust in (True, False):
            old = P.PoseBank(q, K, xyz, SIZE).solve(hidden=(7,), robust=robust)
            new = S.PointLineBank(q, K, xyz, SIZE).solve(dict(lines=[], corners=[]), hidden=(7,), robust=robust)
            same(old, new, exact=True); audit_result(new, (7,))
            assert new["empty_line_pool_exact_point_delegate"] is True
            assert new["point_line_operation_counts"]["point_line_optimizer_calls"] == 0
            details.append(dict(robust=robust, state=new["state"], exact_Rt_score=True))
        # Both lines are consumed, which must also delegate exactly.
        obs = observation(q, edges=(0, 8), corners=(0,))
        old = P.PoseBank(q, K, xyz, SIZE).solve()
        new = S.PointLineBank(q, K, xyz, SIZE).solve(obs, adopted_boundary_corner_ids=(0,))
        same(old, new, exact=True); assert new["consumed_edges"] == [0, 8]
        return dict(cases=details, all_lines_consumed_exact_delegate=True)

    def line_outlier_not_frame_veto():
        K, xyz, q, _ = fixture(); obs = observation(q)
        line = obs["lines"][1]
        shift = 100 * np.array(line["normal"])
        line["offset"] += 100
        line["support_points"] = (np.array(line["support_points"]) + shift).tolist()
        r = S.PointLineBank(q, K, xyz, SIZE).solve(obs); audit_result(r)
        assert r["available"], r["state"]
        assert line["edge"] in r["line_edges"] and line["edge"] not in r["final_line_inliers"]
        assert np.allclose(r["projected"], q[:8], rtol=0, atol=ATOL)
        for row in r["all_candidate_solutions"]:
            assert row["same_scoring_line_edges"] == r["line_edges"]
        return dict(corrupted_semantic_edge=line["edge"], remains_in_same_pool=True,
                    excluded_as_line_outlier=True, pose_available=True)

    def finite_four_five():
        K, xyz, q, _ = fixture(square=True); obs = observation(q)
        cases = []
        for ids in ((0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 3, 4)):
            sparse = q.copy(); sparse[sorted(set(range(8))-set(ids))] = np.nan
            r = S.PointLineBank(sparse, K, xyz, SIZE).solve(obs); audit_result(r)
            assert r["eligible_candidate_count"] > 0
            assert r["available"], r["state"]
            assert np.allclose(r["projected"], q[:8], rtol=0, atol=ATOL)
            if ids == (0, 1, 2, 3):
                assert any("IPPE" in c["generator"] for c in r["all_candidate_solutions"])
                assert len(r["all_candidate_solutions"]) >= 2
            cases.append(dict(ids=list(ids), state=r["state"], numerical_branches=r["eligible_candidate_count"],
                              final_points=len(r["final_inliers"])))
        return dict(actual_numeric_cases=cases, square_registry_dimensions_deduplicated=True)

    def masks_outliers_and_cache():
        K, xyz, q, _ = fixture(); obs = observation(q); cases = []
        for drop, bad in (((7,), ()), ((6, 7), ()), ((), (1,)), ((), (1, 5)), ((7,), (1,)), ((6, 7), (1, 5))):
            changed = q.copy()
            for k in bad: changed[k] += [80., -50.]
            bank = S.PointLineBank(changed, K, xyz, SIZE)
            r = bank.solve(obs, hidden=drop); audit_result(r, drop)
            assert r["available"], r["state"]
            assert set(bad).isdisjoint(r["final_inliers"])
            assert np.allclose(r["projected"], q[:8], rtol=0, atol=ATOL)
            misses = bank.bank.ledger["generic_cache_misses"]
            again = bank.solve(obs, hidden=drop, excluded=(0,))
            audit_result(again, drop, (0,))
            assert bank.bank.ledger["generic_cache_misses"] == misses
            assert misses <= 140
            cases.append(dict(dropped_good=list(drop), retained_bad=list(bad), state=r["state"],
                              correct_point_inliers=r["final_inliers"], numeric_four_ID_misses=misses,
                              later_mask_reuses_bank=True, later_mask_state=again["state"]))
        return dict(cases=cases, recoverability_claim="these fixed mathematical fixtures only; not arbitrary wrong consensus")

    def controlled_tie():
        K, xyz, q, z = fixture(); obs = observation(q)
        # A controlled symmetric *point-only* tie delegates unchanged v4 and
        # preserves its explicit ambiguity. A separate line-scored tie retains
        # only a Y-normal line, insensitive to horizontal translation at t_z.
        q = q.copy(); q[:8] = project(cuboid(*xyz), np.eye(3), np.array([0., .1, 4.]), K)
        z = np.array([0., 0., 0., 0., .1, 4.])
        obs = observation(q, edges=(0,))
        bank = S.PointLineBank(q, K, xyz, SIZE)
        model = bank.bank.models[0]
        zs = []
        for dx in (-.02, .02):
            zz = z.copy(); zz[3] += dx; zs.append(zz)
        bank.bank.hypotheses = [candidate(0, (0, 1, 2, 4), zz, model, K) for zz in zs]
        bank.bank._build = lambda: None
        saved = S.least_squares
        def unavailable(*args, **kwargs): raise ValueError("controlled optimizer absence")
        S.least_squares = unavailable
        try: r = bank.solve(obs)
        finally: S.least_squares = saved
        audit_result(r)
        assert r["state"] == "AMBIGUOUS_PNP" and r["unresolved_ambiguity"] and not r["available"]
        assert len(r["alternatives"]) == 1 and r["alternatives"][0]["numerical_objective_tie"]
        return dict(controlled_candidates=2, actual_PnP_generation=False, actual_optimizer=False,
                    rank_local_only=True, state=r["state"])

    def API_and_policy():
        assert "initial_pose" not in inspect.signature(S.PointLineBank.solve).parameters
        assert "initial_pose" not in inspect.signature(S.PointLineBank).parameters
        assert S.POLICY["minimum_actual_point_inliers"] == 4
        assert S.POLICY["maximum_refit_starts"] == 3
        assert S.POLICY["source_cal_supported_edges"] == [0, 2, 4, 8, 9, 10, 11]
        K, xyz, q, _ = fixture()
        constrained = P.PoseBank(q, K, xyz, SIZE, known_dimension_index=0,
            known_dimension_provenance=dict(source="synthetic declaration only", independent_of_initial_pose=True))
        try: S.PointLineBank(q, K, xyz, SIZE, bank=constrained)
        except ValueError: pass
        else: raise AssertionError("constrained point bank accepted")
        return dict(prior_or_initial_start_not_in_API=True, independently_known_dimension_not_used=True,
                    total_factor_tradeoff_disclosed=True, no_global_uniqueness_claim=True)

    try:
        for name, fn in (("exact_observed_forward_jacobian", analytic_jacobian),
            ("adopted_edges_duplicate_and_final_support_contract", adopted_and_duplicates),
            ("query_replication_one_semantic_factor", support_replication),
            ("hidden_and_heldout_coordinate_invariance", hidden_invariance),
            ("low_actual_point_and_degenerate_unavailable", low_point_and_degenerate),
            ("modeled_normal_rank_avoids_noisy_parallel_phantom", modeled_rank_guard),
            ("empty_unused_line_pool_exact_frozen_point_delegate", empty_pool_delegate),
            ("bad_line_is_outlier_not_frame_veto", line_outlier_not_frame_veto),
            ("actual_exact_four_planar_nonplanar_and_five", finite_four_five),
            ("wrong_mask_outliers_remaining_support_and_shared_cache", masks_outliers_and_cache),
            ("controlled_physical_tie_is_unavailable", controlled_tie),
            ("explicit_prior_free_API_and_fixed_policy", API_and_policy)):
            check(name, fn)
    finally:
        for name, fn in original.items(): setattr(cv2, name, fn)
        S.least_squares = original_optimizer
    return dict(schema="partial_line_independent_CPU_checks_v5", complete=True, attempt=attempt,
        prior_failed_receipt=C.binding(C.DOC/"PARTIAL_LINE_CHECKS.json") if attempt == 2 else None,
        passed=all(c["status"] == "PASS" for c in checks), checks=checks, actual_calls=dict(actual),
        test_groups=len(checks), actual_model_forwards=0, actual_training_updates=0, actual_rays=0,
        actual_real_pose_fits=0, actual_latency_measurements=0,
        code={k: C.binding(path) for k, path in dict(solver=Path(S.__file__), test=Path(__file__),
            v4_pose=Path(P.__file__), original_solver=Path(sys.modules[Candidate.__module__].__file__)).items()},
        limits=["synthetic mathematical fixtures do not establish real boundary ownership or accuracy",
                "rank six is local; finite branch and numerical tie checks do not prove global uniqueness",
                "controlled tie fixture does not represent an additional actual PnP solve",
                "wrong-consensus recovery is not guaranteed for arbitrary corrupted observations",
                "H/proposal generation remains outside this conditional numeric independence test"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    p.add_argument("--output")
    args = p.parse_args()
    suffix = "" if args.attempt == 1 else "_REPAIR1"
    expected = C.DOC/("PARTIAL_LINE_CHECKS"+suffix+".json")
    output = Path(args.output) if args.output else expected
    if output.resolve() != expected.resolve() or output.is_symlink():
        raise RuntimeError("Only the exclusive new-phase CPU receipt is allowed")
    started = C.DOC/("PARTIAL_LINE_CHECKS_STARTED"+suffix+".json")
    if output.exists() or started.exists() or C.DOC.is_symlink():
        raise RuntimeError("Preserve the CPU test attempt; refusing overwrite")
    failed_binding = None
    if args.attempt == 2:
        failed = C.read(C.DOC/"PARTIAL_LINE_CHECKS.json")
        if not failed["complete"] or failed["passed"]:
            raise RuntimeError("Repair attempt requires the preserved actual complete failed receipt")
        for key, name in (("solver", "solver.py"), ("test", "test_partial_lines.py")):
            C.bound(C.DOC/"cpu_attempts/01"/name, failed["code"][key], "original failed CPU code")
        C.bound(C.DOC/"cpu_attempts/01/PARTIAL_LINE_CHECKS.json", C.binding(C.DOC/"PARTIAL_LINE_CHECKS.json"), "original failed CPU receipt")
        failed_binding = C.binding(C.DOC/"PARTIAL_LINE_CHECKS.json")
    C.write_new(started, dict(schema="partial_line_CPU_started_v5", actual_model_forwards=0,
        attempt=args.attempt, prior_failed_receipt=failed_binding,
        code={"solver": C.binding(S.__file__), "test": C.binding(__file__)}))
    result = run(args.attempt); C.write_new(output, result)
    print("PARTIAL_LINE_CHECKS", "PASS" if result["passed"] else "FAIL", result["test_groups"], result["actual_calls"])
    if not result["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
