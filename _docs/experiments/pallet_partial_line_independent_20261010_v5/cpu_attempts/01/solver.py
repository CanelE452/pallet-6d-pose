"""Supplement unused IMAGE_ROLE lines with prior-free finite point hypotheses.

This C2 ablation changes only the final representation of the same observations.
It does not change selection, make points from lines, or initialize from N3 pose.
At least four actual point inliers are required.  One semantic line is one
consensus factor, although its normalized residual has two scalar components.
The total-factor policy can prefer four point plus seven line inliers to seven
point inliers; it is not a guarantee of preserving the larger point consensus.
Actual real-image physical boundary ownership remains unverified.
"""
from __future__ import annotations

import copy
import json

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..pallet_cornerwise_independent_20261010_v4 import pose as P
from ..pallet_observation_refiner_20261009_v1.solver import Candidate, _shape, project


EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
SOURCE_REGISTRY_EDGE_IDS = frozenset((0, 2, 4, 6, 8, 9, 10, 11))
POLICY = dict(
    schema="supplemental_observation_only_point_line_policy_v5",
    point_inlier_threshold_px=8., line_inlier_RMS_threshold_px=8.,
    minimum_actual_points=4, minimum_actual_point_inliers=4,
    line_residual="two projected registry endpoint to observed line distances divided by sqrt(2)",
    consensus_unit="one distinct actual corner ID or one distinct semantic edge ID",
    score="total point+edge inlier count first, truncated factor SSE second",
    tradeoff="4 point + 7 line inliers may outrank 7 point + 0 line; no point-count-preservation claim",
    ordinary_score="untruncated point+edge factor SSE",
    line_weights_fixed=True, same_factor_pool_for_all_hypotheses=True,
    loss="soft_l1", f_scale_px=8., max_nfev=50, maximum_refit_starts=3,
    numerical_rank_relative_threshold=1e-10,
    numerical_score_tie_px2=1e-8,
    numerical_pose_equivalence=copy.deepcopy({k: v for k, v in P.POLICY.items() if k.startswith("equivalent_")}),
    initial_pose_prior=False, initial_pose_start=False, initial_dimension_prior=False,
    numeric_initializer="all valid cached four-point SQPnP/IPPE branches, both registry dimensions",
    duplicate_policy="remove incident edges of actually adopted allowed boundary corners; one factor per remaining edge",
    inconsistent_duplicate_edge="fail closed; no optimization",
    source_registry_supported_edges=sorted(SOURCE_REGISTRY_EDGE_IDS),
    source_cal_supported_edges=[0, 2, 4, 8, 9, 10, 11],
    edge_model="existing source wire registry; edge 6 lacks CAL support; independent real-image ownership not verified",
    empty_unused_line_pool="delegate unchanged v4 point solve exactly; no new SciPy refinement",
    H_endpoint_policy="visible line retained even if a registry endpoint is in H; H image coordinates unused",
    rank="exact forward Jacobian, plus modeled-normal geometry Jacobian with current modeled normals held fixed",
    rank_is_local_only=True, global_unique_pose_proven=False,
    fewer_than_four="SUPPLEMENTAL_LINE_UNAVAILABLE; no independent global PnL or local initial-pose rescue",
    hidden_reprojection="after accepted R,t only; never refit projections",
)


def _camera(K):
    """The frozen no-distortion projectPoints Jacobian uses pinhole K."""
    return bool(np.array_equal(K[2], [0., 0., 1.]) and K[0, 1] == 0. and K[1, 0] == 0.)


def projection_jacobian(model, z, K):
    """Exact OpenCV forward projection and its rotation/translation columns."""
    q, J = cv2.projectPoints(np.asarray(model, float), np.asarray(z[:3], float),
                            np.asarray(z[3:], float), np.asarray(K, float), None)
    return q.reshape(-1, 2), J[:, :6].reshape(-1, 2, 6)


def residual_jacobian(model, z, K, points, point_ids, lines, *, modeled_normals=False):
    q, J = projection_jacobian(model, z, K)
    residual, rows = [], []
    for k in point_ids:
        residual.extend(q[k] - points[k])
        rows.extend(J[k])
    for line in lines:
        a, b = EDGES[line["edge"]]
        normal = np.asarray(line["normal"], float)
        if modeled_normals:
            direction = q[b] - q[a]
            length = float(np.linalg.norm(direction))
            if not np.isfinite(length) or length <= 1e-12:
                raise ValueError("projected_model_edge_degenerate")
            normal = np.array([-direction[1], direction[0]]) / length
        # Modeled-normal rows are a local geometry diagnostic. They are not
        # the derivative of a moving-normal residual objective.
        residual.extend((q[[a, b]] @ normal - line["offset"]) / np.sqrt(2))
        rows.extend([normal @ J[a] / np.sqrt(2), normal @ J[b] / np.sqrt(2)])
    return np.asarray(residual, float), np.asarray(rows, float).reshape(-1, 6), q


def rank_info(matrix):
    matrix = np.asarray(matrix, float).reshape(-1, 6)
    if not len(matrix) or not np.isfinite(matrix).all():
        return dict(numerical_rank=0, singular_values=[], condition_number=None,
                    normalized_column_scale=[], scalar_rows=len(matrix), local_only=True)
    scale = np.maximum(np.linalg.norm(matrix, axis=0), 1e-300)
    singular = np.linalg.svd(matrix / scale, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * POLICY["numerical_rank_relative_threshold"]))
    return dict(numerical_rank=rank, singular_values=singular.tolist(),
                condition_number=float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
                normalized_column_scale=scale.tolist(), scalar_rows=len(matrix),
                normalization="unit L2 Jacobian columns", local_only=True)


def geometry_info(model, z, K, points, point_ids, lines, *, projection_counter=None):
    if projection_counter is not None:
        projection_counter()
    _, observed, _ = residual_jacobian(model, z, K, points, point_ids, lines)
    if projection_counter is not None:
        projection_counter()
    _, modeled, _ = residual_jacobian(model, z, K, points, point_ids, lines, modeled_normals=True)
    split = 2 * len(point_ids)
    return dict(observed_normal_joint=rank_info(observed),
                modeled_normal_joint=rank_info(modeled), point_only=rank_info(modeled[:split]),
                observed_normal_lines_only=rank_info(observed[split:]),
                modeled_normal_lines_only=rank_info(modeled[split:]),
                point_ids=list(point_ids), line_edges=[l["edge"] for l in lines],
                modeled_normals_held_fixed_for_derivative=True,
                global_unique_pose_proven=False)


class PointLineBank:
    def __init__(self, points, K, xyz, image_size=(640, 480), *, bank=None):
        if bank is None:
            bank = P.PoseBank(points, K, xyz, image_size=image_size)
        if not isinstance(bank, P.PoseBank) or bank.known_dimension_index is not None:
            raise ValueError("A matching prior-free, unconstrained v4 PoseBank is required")
        if not (np.array_equal(bank.points, np.asarray(points, float), equal_nan=True)
                and np.array_equal(bank.K, np.asarray(K, float))
                and np.array_equal(bank.xyz, np.asarray(xyz, float))
                and bank.image_size == tuple(image_size)):
            raise ValueError("Point bank numeric inputs differ")
        self.bank = bank
        self.points, self.K, self.xyz = bank.points, bank.K, bank.xyz
        self.counts = dict(point_line_optimizer_calls=0, optimizer_residual_calls=0,
                           optimizer_jacobian_calls=0, optimizer_projectPoints_calls=0,
                           geometry_diagnostic_calls=0, diagnostic_projectPoints_calls=0)

    def _lines(self, observation, adopted, U):
        corners = {int(c["id"]): c for c in observation.get("corners", [])}
        if len(corners) != len(observation.get("corners", [])):
            raise ValueError("duplicate_boundary_corner_ID")
        consumed = set()
        for k in adopted:
            if k not in U:
                continue
            c = corners.get(k)
            if c is None or not np.array_equal(np.asarray(c["xy"], float), self.points[k]):
                raise ValueError("adopted_boundary_coordinate_not_actual_observation")
            edges = tuple(int(e) for e in c["edges"])
            if len(edges) != 2 or len(set(edges)) != 2 or any(e < 0 or e >= 12 or k not in EDGES[e] for e in edges):
                raise ValueError("invalid_adopted_boundary_incident_edges")
            consumed.update(edges)
        seen, accepted, checks = {}, [], []
        for raw in observation.get("lines", []):
            edge = int(raw["edge"])
            if edge < 0 or edge >= 12:
                raise ValueError("invalid_semantic_edge_ID")
            # Consistent duplicate records collapse; inconsistent geometry,
            # support, identity or covariance fails closed before any optimizer.
            identity = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if edge in seen:
                if identity != seen[edge]:
                    raise ValueError("inconsistent_duplicate_semantic_edge")
                checks.append(dict(edge=edge, accepted=False, reason="CONSISTENT_DUPLICATE_COLLAPSED"))
                continue
            seen[edge] = identity
            if tuple(int(k) for k in raw["endpoints"]) != EDGES[edge]:
                raise ValueError("semantic_edge_endpoint_identity_mismatch")
            if edge not in SOURCE_REGISTRY_EDGE_IDS:
                if edge in consumed:
                    raise ValueError("adopted_boundary_uses_source_registry_unsupported_edge")
                checks.append(dict(edge=edge, accepted=False, reason="SOURCE_REGISTRY_UNSUPPORTED_EDGE"))
                continue
            normal = np.asarray(raw["normal"], float)
            support = np.asarray(raw["support_points"], float)
            radii = np.asarray(raw["query_radii_px"], float)
            queries = [int(v) for v in raw["queries"]]
            norm = float(np.linalg.norm(normal))
            if (normal.shape != (2,) or not np.isfinite(normal).all() or norm <= 0 or
                    support.ndim != 2 or support.shape[1] != 2 or not np.isfinite(support).all() or
                    radii.shape != (len(support),) or not np.isfinite(radii).all() or (radii < 0).any() or
                    len(queries) != len(support) or len(set(queries)) < 3 or
                    any(q // 7 != edge or q < 0 or q >= 84 for q in queries)):
                raise ValueError("invalid_final_line_support_contract")
            offset = float(raw["offset"]) / norm
            normal = normal / norm
            errors = np.abs(support @ normal - offset)
            good = bool(np.isfinite(offset) and np.all(errors <= radii + 1e-9))
            if edge in consumed and not good:
                raise ValueError("adopted_boundary_has_inconsistent_final_support_line")
            reason = ("CONSUMED_BY_ACTUALLY_ADOPTED_BOUNDARY" if edge in consumed else
                      "accepted" if good else "FINAL_LINE_CONSENSUS_INCONSISTENT")
            checks.append(dict(edge=edge, accepted=good and edge not in consumed, reason=reason,
                support_query_ids=queries, absolute_support_residuals_px=errors.tolist(), query_radii_px=radii.tolist(), tolerance_px=1e-9))
            if good and edge not in consumed:
                accepted.append(dict(edge=edge, normal=normal, offset=offset,
                                     support_query_ids=queries, source_line=copy.deepcopy(raw)))
        if not consumed <= set(seen):
            raise ValueError("adopted_boundary_has_missing_support_line")
        accepted.sort(key=lambda l: l["edge"])
        return accepted, sorted(consumed), checks

    def _score(self, candidate, U, lines, robust):
        point = np.linalg.norm(candidate.projected[list(U)] - self.points[list(U)], axis=1)
        edge = []
        for line in lines:
            d = candidate.projected[list(EDGES[line["edge"]])] @ line["normal"] - line["offset"]
            edge.append(float(np.sqrt(np.mean(d ** 2))))
        edge = np.asarray(edge, float)
        point_ids = tuple(k for k, e in zip(U, point) if e <= 8.)
        edge_ids = tuple(l["edge"] for l, e in zip(lines, edge) if e <= 8.)
        norms = np.r_[point, edge]
        loss = float(np.minimum(norms ** 2, 64.).sum()) if robust else float((norms ** 2).sum())
        key = ((-len(point_ids) - len(edge_ids), loss) if robust else (loss,)) + (
            candidate.dim, candidate.ids, candidate.solution, candidate.generator)
        return key, point_ids, edge_ids, point, edge

    def solve(self, observation, adopted_boundary_corner_ids=(), hidden=(), excluded=(), robust=True):
        before_bank, before = self.bank.ledger.copy(), self.counts.copy()
        H, temporary = P._ids(hidden, "hidden"), P._ids(excluded, "excluded")
        adopted = P._ids(adopted_boundary_corner_ids, "adopted_boundary_corner_ids")
        blocked = tuple(sorted(set(H) | set(temporary)))
        U = tuple(k for k in self.bank.eligible if k not in blocked)
        result = dict(available=False, pose_available=False, new_pose_estimated=False,
            no_pose=True, fallback_used=False, state="SUPPLEMENTAL_LINE_UNAVAILABLE",
            reason="fewer_than_four_actual_point_observations", output_status="NO_NEW_POSE",
            eligible=list(self.bank.eligible), used=list(U), excluded=list(blocked),
            hidden=list(H), actual_self_hidden_ids=list(H), temporary_excluded=list(temporary),
            ineligible=copy.deepcopy(self.bank.ineligible), input_hash=self.bank.digest,
            solver="OBSERVATION_ONLY_FINITE_POINT_LINE", policy=copy.deepcopy(POLICY),
            prior_used=False, initial_pose_used=False, initial_projection_used=False,
            initial_dimension_prior_used=False, known_dimension_constraint_used=False,
            excluded_image_coordinates_used_for_scoring=False,
            excluded_image_coordinates_used_for_equivalence=False,
            equivalence_uses_all_eight_model_predictions=True,
            initial_pose_start_used=False, global_uniqueness_proven=False,
            inliers=[], input_inliers=[], final_inliers=[], fit_input_ids=[],
            final_line_inliers=[], fit_line_edges=[], line_edges=[], consumed_edges=[],
            R_cf=None, R_physical=None, centroid=None, cf_extents=None, projected=None,
            points_final=self.points.tolist(), hidden_reprojected=False,
            hidden_initial_excluded=True, reprojected_points_reused_as_observations=False,
            same_edge_point_line_double_count=False, derived_line_points=0,
            candidate_count=0, eligible_candidate_count=0, refit_count=0,
            multiple_solutions=False, unresolved_ambiguity=False,
            all_candidate_solutions=[], alternatives=[], per_dimension_best=[], optimizer_attempts=[],
            geometry=dict(image=_shape(self.points[list(U)]) if U else None))

        def finish():
            result["operation_counts"] = {k: self.bank.ledger[k] - before_bank[k] for k in self.bank.ledger}
            result["operation_counts"].update({k: self.counts[k] - before[k] for k in self.counts})
            result["hypothesis_bank_counts"] = self.bank.ledger.copy()
            result["point_line_operation_counts"] = self.counts.copy()
            return result

        try:
            lines, consumed, checks = self._lines(observation, adopted, set(U))
        except (KeyError, TypeError, ValueError, IndexError) as error:
            result.update(state="INVALID_OBSERVATION_CONTRACT", reason=str(error))
            return finish()
        result.update(line_edges=[l["edge"] for l in lines], consumed_edges=consumed,
            line_contract_checks=checks, actual_adopted_boundary_ids=[k for k in adopted if k in U],
            factor_pool=dict(point_ids=list(U), line_edges=[l["edge"] for l in lines],
                actual_point_factors=len(U), unique_line_factors=len(lines),
                line_support_queries=sum(len(l["support_query_ids"]) for l in lines),
                scalar_residuals=2 * (len(U) + len(lines))))
        if not lines:
            supplemental = {key: copy.deepcopy(result[key]) for key in (
                "line_edges", "consumed_edges", "line_contract_checks", "actual_adopted_boundary_ids", "factor_pool",
                "final_line_inliers", "fit_line_edges", "same_edge_point_line_double_count", "derived_line_points")}
            result = self.bank.solve(hidden=H, excluded=temporary, robust=robust)
            result.update(supplemental, C2_policy=copy.deepcopy(POLICY), C2_solver="OBSERVATION_ONLY_FINITE_POINT_LINE",
                empty_line_pool_exact_point_delegate=True, selected_pose_refined_with_lines=False,
                supplemental_line_state="SUPPLEMENTAL_LINE_UNAVAILABLE" if len(U) < 4 else "EMPTY_LINE_POOL_POINT_DELEGATE",
                point_inlier_count=len(result["final_inliers"]), unique_line_inlier_count=0,
                total_inlier_factor_count=len(result["final_inliers"]), optimizer_attempts=[])
            return finish()
        if len(U) < 4:
            return finish()
        if not _camera(self.K):
            result.update(state="UNSUPPORTED_CAMERA_CONTRACT", reason="forward_Jacobian_requires_frozen_zero_skew_pinhole_K")
            return finish()
        if result["geometry"]["image"]["numerical_rank"] < 2:
            result.update(state="DEGENERATE_OBSERVATIONS", reason="degenerate_actual_point_layout")
            return finish()
        self.bank._build()
        candidates = [c for c in self.bank.hypotheses if set(c.ids) <= set(U)]
        result.update(candidate_count=len(self.bank.hypotheses), eligible_candidate_count=len(candidates))
        if not candidates:
            result.update(state="NUMERICAL_GENERATION_FAILURE", reason="no_valid_finite_point_hypothesis")
            return finish()
        history = {}

        def diagnostic(c):
            key, point_ids, edge_ids, rp, rl = self._score(c, U, lines, robust)
            actual_points, actual_edges = history.get(id(c), (c.ids, ()))
            return dict(dimension_index=c.dim, dimensions=self.bank.dims[c.dim].tolist(),
                generator=c.generator, generator_ids=list(c.ids), solution_index=c.solution,
                point_inlier_count=len(point_ids), line_inlier_count=len(edge_ids),
                total_inlier_factor_count=len(point_ids) + len(edge_ids),
                inlier_ids=list(point_ids), line_inlier_edges=list(edge_ids),
                actual_fit_input_ids=list(actual_points), actual_fit_line_edges=list(actual_edges),
                point_residual_norms_px=rp.tolist(), line_residual_RMS_px=rl.tolist(),
                line_endpoint_signed_residuals_px=[(c.projected[list(EDGES[l["edge"]])] @ l["normal"] - l["offset"]).tolist() for l in lines],
                same_scoring_point_ids=list(U), same_scoring_line_edges=[l["edge"] for l in lines],
                truncated_sse_px2=float(np.minimum(np.r_[rp, rl] ** 2, 64.).sum()),
                sse_px2=float((np.r_[rp, rl] ** 2).sum()),
                R_cf=P._rotation(c).tolist(), R_physical=P._physical_rotation(c).tolist(),
                centroid=c.tvec.reshape(3).tolist(), projected=c.projected.tolist())

        viable = [c for c in candidates if len(self._score(c, U, lines, robust)[1]) >= 4]
        ranked = sorted(viable, key=lambda c: self._score(c, U, lines, robust)[0])
        starts = []
        for c in ranked:
            if not any(self.bank._equivalent(c, other) for other in starts):
                starts.append(c)
            if len(starts) == 3:
                break
        refined = []
        for seed in starts:
            _, ip, il, _, _ = self._score(seed, U, lines, robust)
            fit_points = ip if robust else U
            fit_edges = il if robust else tuple(l["edge"] for l in lines)
            fit_lines = [l for l in lines if l["edge"] in fit_edges]
            assert set(fit_points).isdisjoint(blocked) and len(fit_points) >= 4
            z0 = np.r_[seed.rvec.reshape(3), seed.tvec.reshape(3)]
            self.counts["point_line_optimizer_calls"] += 1
            result["refit_count"] += 1

            def residual(z):
                self.counts["optimizer_residual_calls"] += 1
                self.counts["optimizer_projectPoints_calls"] += 1
                return residual_jacobian(self.bank.models[seed.dim], z, self.K, self.points, fit_points, fit_lines)[0]

            def jacobian(z):
                self.counts["optimizer_jacobian_calls"] += 1
                self.counts["optimizer_projectPoints_calls"] += 1
                return residual_jacobian(self.bank.models[seed.dim], z, self.K, self.points, fit_points, fit_lines)[1]

            attempt = dict(dimension_index=seed.dim, generator_ids=list(seed.ids),
                fit_point_ids=list(fit_points), fit_line_edges=list(fit_edges), initial_pose_used=False)
            try:
                fit = least_squares(residual, z0, jac=jacobian, loss="soft_l1", f_scale=8., max_nfev=50)
                attempt.update(success=bool(fit.success), scipy_status=int(fit.status), nfev=int(fit.nfev),
                    njev=int(fit.njev) if fit.njev is not None else None, soft_l1_cost=float(fit.cost))
                if fit.success and self.bank._valid(fit.x[:3], fit.x[3:], seed.dim):
                    R = cv2.Rodrigues(fit.x[:3])[0]
                    c = Candidate(seed.dim, tuple(fit_points), seed.solution, seed.generator + "+POINT_LINE",
                        fit.x[:3].reshape(3, 1), fit.x[3:].reshape(3, 1),
                        project(self.bank.models[seed.dim], R, fit.x[3:], self.K))
                    refined.append(c)
                    history[id(c)] = (tuple(fit_points), tuple(fit_edges))
                else:
                    attempt["rejected"] = "optimizer_failure_or_invalid_pose"
            except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
                attempt.update(success=False, error=type(error).__name__, message=str(error))
            result["optimizer_attempts"].append(attempt)
        pool = candidates + refined
        result["all_candidate_solutions"] = [diagnostic(c) for c in pool]
        for dim in range(len(self.bank.dims)):
            group = [c for c in pool if c.dim == dim]
            if group:
                result["per_dimension_best"].append(diagnostic(min(group, key=lambda c: self._score(c, U, lines, robust)[0])))
        viable = [c for c in pool if len(self._score(c, U, lines, robust)[1]) >= 4]
        if not viable:
            result.update(state="INSUFFICIENT_POINT_CONSENSUS", reason="fewer_than_four_actual_point_inliers")
            return finish()
        ranked = sorted(viable, key=lambda c: self._score(c, U, lines, robust)[0])
        chosen = ranked[0]
        key, ip, il, rp, rl = self._score(chosen, U, lines, robust)
        fit_points, fit_edges = history.get(id(chosen), (chosen.ids, ()))
        result.update(best_candidate=diagnostic(chosen), input_inliers=list(fit_points), fit_input_ids=list(fit_points),
            final_inliers=list(ip), inliers=list(ip), final_line_inliers=list(il), fit_line_edges=list(fit_edges),
            point_inlier_count=len(ip), unique_line_inlier_count=len(il), total_inlier_factor_count=len(ip) + len(il),
            residuals_used_px=rp.tolist(), line_residual_RMS_px=rl.tolist(),
            line_endpoint_signed_residuals_px=result["best_candidate"]["line_endpoint_signed_residuals_px"],
            weak_four_point_consensus=len(ip) == 4,
            selected_pose_refined_with_lines=bool(fit_edges))
        self.counts["geometry_diagnostic_calls"] += 1
        try:
            def diagnostic_projection():
                self.counts["diagnostic_projectPoints_calls"] += 1
            geo = geometry_info(self.bank.models[chosen.dim], np.r_[chosen.rvec.reshape(3), chosen.tvec.reshape(3)],
                self.K, self.points, ip, [l for l in lines if l["edge"] in il], projection_counter=diagnostic_projection)
        except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
            result.update(state="NUMERICAL_GEOMETRY_FAILURE", reason=str(error))
            return finish()
        result["point_line_geometry"] = geo
        result["geometry"].update(object=_shape(self.bank.models[chosen.dim][list(ip)]), jacobian=geo["modeled_normal_joint"])
        if geo["observed_normal_joint"]["numerical_rank"] < 6 or geo["modeled_normal_joint"]["numerical_rank"] < 6:
            result.update(state="NUMERICAL_RANK_DEFICIENT", reason="joint_observed_or_modeled_normal_rank_below_six")
            return finish()
        objective_index = 1 if robust else 0
        for other in ranked[1:]:
            if self.bank._equivalent(chosen, other):
                continue
            other_key = self._score(other, U, lines, robust)[0]
            tied = (not robust or other_key[0] == key[0]) and abs(other_key[objective_index] - key[objective_index]) <= 1e-8
            info = diagnostic(other)
            info["numerical_objective_tie"] = bool(tied)
            info["inlier_factor_count_difference_from_selected"] = len(ip) + len(il) - info["total_inlier_factor_count"]
            info["objective_difference_from_selected_px2"] = float(other_key[objective_index] - key[objective_index])
            result["alternatives"].append(info)
            result["multiple_solutions"] = True
            result["unresolved_ambiguity"] |= bool(tied)
        if result["unresolved_ambiguity"]:
            result.update(state="AMBIGUOUS_PNP", reason="unresolved_equal_score_physical_point_line_pose")
            return finish()
        output = self.points.copy()
        if H:
            output[list(H)] = chosen.projected[list(H)]
        assert set(fit_points).isdisjoint(blocked)
        assert np.array_equal(output[8:], self.points[8:], equal_nan=True)
        keep = sorted(set(temporary) - set(H))
        assert np.array_equal(output[keep], self.points[keep], equal_nan=True)
        result.update(available=True, pose_available=True, new_pose_estimated=True, no_pose=False,
            state="NEW_POSE", reason="new_observation_only_point_line_pose", output_status="NEW_POSE",
            R_cf=P._rotation(chosen).tolist(), R_physical=P._physical_rotation(chosen).tolist(),
            centroid=chosen.tvec.reshape(3).tolist(), cf_extents=self.bank.dims[chosen.dim].tolist(),
            dimension_index=chosen.dim, selected_hypothesis=self.bank.names[chosen.dim],
            projected=chosen.projected.tolist(), points_final=output.tolist(), hidden_reprojected=bool(H),
            reprojection_px=float(rp.mean()), truncated_sse_px2=float(np.minimum(np.r_[rp, rl] ** 2, 64.).sum()),
            sse_px2=float((np.r_[rp, rl] ** 2).sum()), generator=chosen.generator, generator_ids=list(chosen.ids))
        return finish()
