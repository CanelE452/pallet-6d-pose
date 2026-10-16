"""Finite observation-only PnP for corner-held-out validation.

This separate solver never reads an initial pose, its projections, or its
dimension choice.  A lazy bank retains the original finite four-ID generators
and every valid returned branch.  Only actual allowed image correspondences
score a branch or enter a refit.  The supplied hidden IDs and temporary excluded
IDs have different output meanings: both are excluded from fitting, but only
the physical hidden IDs are replaced after a new pose has been accepted.

Numerically tied, physically different poses are an explicit unavailable result.
Rank six is a local numerical condition and is never a uniqueness certificate.
"""
from __future__ import annotations

import copy
import operator

import cv2
import numpy as np

from ..pallet_observation_refiner_20261009_v1.solver import (
    HypothesisBank, MAX_REFITS, RESIDUAL_PX, _shape,
)


POLICY = dict(
    observation_residual_px=RESIDUAL_PX,
    maximum_refit_starts=MAX_REFITS,
    minimum_correspondences=4,
    initial_pose_prior=False,
    initial_rotation_translation_input=False,
    initial_projection_input=False,
    initial_dimension_choice_input=False,
    independently_known_dimension="optional explicit constructor index and provenance",
    dimension_hypotheses="all deduplicated registry dimensions unless explicitly known",
    generators="unchanged SQPnPGeneric / exact-plane IPPE; every valid returned solution",
    generic_cache="same numeric bank and exact (dimension, fit-ID tuple), across phases and masks",
    robust_score="descending inlier count then truncated squared residual over allowed U only",
    ordinary_score="squared residual over allowed U only",
    numerical_score_tie_px2=1e-8,
    equivalent_projection_max_abs_px=1e-5,
    equivalent_physical_rotation_max_abs=1e-5,
    equivalent_translation_relative_diagonal=1e-5,
    numerical_equivalence_not_physical_symmetry="physical R/t also checked; ADD symmetry is not a pose identity certificate",
    tie_order="dimension index, lexicographic generator IDs, returned solution index, generator",
    ambiguity="same inlier count and numerically tied objective; distinct physical R/t or all-eight model projection",
    ambiguity_uses_excluded_image_coordinates=False,
    global_uniqueness_proven=False,
    hidden_policy="exclude H before scoring/refit; replace only H after final R,t; never refit projections",
    temporary_exclusion_policy="exclude from generators/scoring/refit; do not replace excluded-only coordinates",
    fallback="none inside solver; caller may separately return a declared unchanged baseline",
)


def _ids(values, label):
    try:
        result = tuple(sorted(set(operator.index(v) for v in values)))
    except (TypeError, ValueError) as error:
        raise ValueError(label + " must contain integer corner IDs") from error
    if any(i < 0 or i >= 8 for i in result):
        raise ValueError(label + " must contain only corner IDs 0..7")
    return result


def _rotation(candidate):
    return cv2.Rodrigues(candidate.rvec)[0]


def _physical_rotation(candidate):
    # Preserve the frozen old solver's registry-to-physical convention.
    Q = (np.eye(3) if candidate.dim == 0 else
         np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]))
    return _rotation(candidate) @ Q


class PoseBank(HypothesisBank):
    """Immutable observations with reusable finite numerical hypotheses.

    ``known_dimension_index`` is optional external knowledge, never copied from
    an initial pose.  Its provenance is a caller declaration, not an assertion
    that this module can independently validate the provenance of that fact.
    """

    def __init__(self, points, K, xyz, image_size=(640, 480), *,
                 known_dimension_index=None, known_dimension_provenance=None):
        if np.asarray(points).shape != (9, 2):
            raise ValueError("Expected eight corners and preserved center, shape (9,2)")
        size = np.asarray(image_size, float)
        if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError("image_size must be positive finite (width,height)")
        self.image_size = tuple(image_size)
        super().__init__(points, K, xyz, image_size=self.image_size)
        self.numeric_cache = {}
        self.ledger.update(generic_cache_hits=0, generic_cache_misses=0)
        if known_dimension_index is None:
            if known_dimension_provenance is not None:
                raise ValueError("Dimension provenance requires an explicit dimension index")
            self.known_dimension_index = None
            self.known_dimension_provenance = None
        else:
            index = operator.index(known_dimension_index)
            if index < 0 or index >= len(self.dims):
                raise ValueError("known_dimension_index is not a deduplicated registry index")
            if not isinstance(known_dimension_provenance, dict) or not (
                    known_dimension_provenance.get("independent_of_initial_pose") is True
                    and isinstance(known_dimension_provenance.get("source"), str)
                    and known_dimension_provenance["source"].strip()):
                raise ValueError("Known dimension needs an explicit independent source declaration")
            self.known_dimension_index = index
            self.known_dimension_provenance = copy.deepcopy(known_dimension_provenance)

    def _generic(self, ids, dim, phase):
        """Cache only identical numeric problems; the mask is not a cache key."""
        ids = tuple(sorted(ids))
        key = (int(dim), tuple(ids))
        if key in self.numeric_cache:
            self.ledger["generic_cache_hits"] += 1
            return list(self.numeric_cache[key])
        self.ledger["generic_cache_misses"] += 1
        values = super()._generic(ids, dim, phase)
        self.numeric_cache[key] = tuple(values)
        return list(values)

    def _equivalent(self, first, second):
        """Numerical duplicate only; never infer an object symmetry."""
        projection = float(np.max(np.abs(first.projected - second.projected)))
        rotation = float(np.max(np.abs(_physical_rotation(first) - _physical_rotation(second))))
        diagonal = max(float(np.linalg.norm(self.xyz)), 1e-300)
        translation = float(np.linalg.norm(first.tvec.reshape(3) - second.tvec.reshape(3))) / diagonal
        return (projection < POLICY["equivalent_projection_max_abs_px"] and
                rotation < POLICY["equivalent_physical_rotation_max_abs"] and
                translation < POLICY["equivalent_translation_relative_diagonal"])

    def solve(self, excluded=(), hidden=(), robust=True):
        """Solve only actual U; excluded-only IDs never become projected outputs."""
        before = self.ledger.copy()
        temporary = _ids(excluded, "excluded")
        H = _ids(hidden, "hidden")
        blocked = tuple(sorted(set(temporary) | set(H)))
        allowed = set(self.eligible) - set(blocked)
        U = tuple(i for i in self.eligible if i in allowed)
        result = dict(
            available=False, pose_available=False, new_pose_estimated=False,
            no_pose=True, fallback_used=False, state="INSUFFICIENT_OBSERVATIONS",
            reason="insufficient_observations", output_status="NO_NEW_POSE",
            eligible=list(self.eligible), used=list(U), excluded=list(blocked),
            temporary_excluded=list(temporary), temporary_exclusion_is_self_occlusion=False,
            hidden=list(H), actual_self_hidden_ids=list(H), ineligible=copy.deepcopy(self.ineligible),
            inliers=[], input_inliers=[], final_inliers=[], fit_input_ids=[],
            input_hash=self.digest, points_final=self.points.tolist(),
            R_cf=None, R_physical=None, centroid=None, cf_extents=None, projected=None,
            solver="OBSERVATION_ONLY_FINITE_SUBSET_ROBUST" if robust else "OBSERVATION_ONLY_STANDARD",
            residual_threshold_px=RESIDUAL_PX, candidate_count=0,
            eligible_candidate_count=0, refit_count=0, refit_solution_count=0,
            LM_solution_count=0, hidden_reprojected=False, hidden_initial_excluded=True,
            reprojected_points_reused_as_observations=False,
            prior_used=False, initial_pose_used=False, initial_projection_used=False,
            initial_dimension_prior_used=False, excluded_image_coordinates_used_for_scoring=False,
            excluded_image_coordinates_used_for_equivalence=False,
            equivalence_uses_all_eight_model_predictions=True,
            known_dimension_constraint_used=self.known_dimension_index is not None,
            known_dimension_index=self.known_dimension_index,
            known_dimension_provenance=copy.deepcopy(self.known_dimension_provenance),
            known_dimension_provenance_independently_verified=False,
            multiple_solutions=False, unresolved_ambiguity=False,
            global_uniqueness_proven=False, alternatives=[], per_dimension_best=[],
            all_candidate_solutions=[], numerical_tied_solution_count=0,
            weak_four_point_consensus=len(U) == 4,
            geometry=dict(image=_shape(self.points[list(U)]) if U else None),
            policy=copy.deepcopy(POLICY),
        )

        def finish():
            result["operation_counts"] = {k: self.ledger[k] - before[k] for k in self.ledger}
            result["hypothesis_bank_counts"] = self.ledger.copy()
            return result

        if len(U) < 4:
            return finish()
        object_ranks = [_shape(self.models[d][list(U)])["numerical_rank"] for d in range(len(self.dims))]
        if result["geometry"]["image"]["numerical_rank"] < 2 or max(object_ranks) < 2:
            result.update(state="DEGENERATE_OBSERVATIONS", reason="degenerate_observation_layout")
            return finish()
        if robust:
            self._build()
            candidates = [c for c in self.hypotheses if set(c.ids) <= allowed]
            result["candidate_count"] = len(self.hypotheses)
        else:
            if U not in self.standard_cache:
                self.standard_cache[U] = [c for d in range(len(self.dims))
                                          for c in self._generic(U, d, "standard")]
            candidates = list(self.standard_cache[U])
            result["candidate_count"] = len(candidates)
        result["eligible_candidate_count"] = len(candidates)
        if not candidates:
            result.update(state="NUMERICAL_GENERATION_FAILURE",
                          reason="degenerate_or_numerical_generation_failure")
            return finish()
        fit_history = {}
        regenerated = []
        refined = []

        def diagnostic(candidate):
            key, inliers, residual = self._score(candidate, U, robust)
            return dict(dimension_index=candidate.dim, dimensions=self.dims[candidate.dim].tolist(),
                selected_hypothesis=self.names[candidate.dim],
                dimension_constraint_allowed=self.known_dimension_index is None or candidate.dim == self.known_dimension_index,
                generator=candidate.generator, generator_ids=list(candidate.ids),
                solution_index=candidate.solution, inlier_count=len(inliers), inlier_ids=list(inliers),
                actual_fit_input_ids=list(fit_history.get(id(candidate), candidate.ids)),
                score_count=len(inliers) if robust else len(U),
                objective_px2=float(key[1] if robust else key[0]),
                truncated_sse_px2=float(np.minimum(residual ** 2, RESIDUAL_PX ** 2).sum()),
                sse_px2=float((residual ** 2).sum()), residuals_used_px=residual.tolist(),
                R_cf=_rotation(candidate).tolist(), R_physical=_physical_rotation(candidate).tolist(),
                centroid=candidate.tvec.reshape(3).tolist(), projected=candidate.projected.tolist())

        dimension_candidates = [c for c in candidates if self.known_dimension_index is None or
                                c.dim == self.known_dimension_index]
        result["dimension_allowed_candidate_count"] = len(dimension_candidates)
        if not dimension_candidates:
            result["all_candidate_solutions"] = [diagnostic(c) for c in candidates]
            for dim in range(len(self.dims)):
                group = [c for c in candidates if c.dim == dim]
                if group:
                    result["per_dimension_best"].append(diagnostic(min(group, key=lambda c: self._score(c, U, robust)[0])))
            result.update(state="KNOWN_DIMENSION_GENERATION_FAILURE",
                          reason="no_valid_solution_for_explicit_known_dimension")
            return finish()
        ranked = sorted(dimension_candidates, key=lambda c: self._score(c, U, robust)[0])
        if robust:
            starts = []
            for candidate in ranked:
                if any(self._equivalent(candidate, seed) for seed in starts):
                    continue
                starts.append(candidate)
                if len(starts) == MAX_REFITS:
                    break
            for candidate in starts:
                ids = self._score(candidate, U, True)[1]
                if len(ids) < 4:
                    continue
                assert set(ids) <= allowed and not set(ids) & set(blocked)
                result["refit_count"] += 1
                branches = self._generic(ids, candidate.dim, "refit")
                regenerated.extend(branches)
                seed = min(branches, key=lambda c: self._score(c, U, True)[0]) if branches else candidate
                fitted = self._lm(seed, ids)
                if fitted is not None:
                    refined.append(fitted)
                    fit_history[id(fitted)] = ids
        else:
            result["refit_count"] = 1
            fitted = self._lm(ranked[0], U)
            if fitted is not None:
                refined.append(fitted)
                fit_history[id(fitted)] = U
        result["refit_solution_count"] = len(regenerated)
        result["LM_solution_count"] = len(refined)
        # Keep every returned numeric branch, including refit generators that
        # were not selected as an LM start. Repeated cache references are not
        # additional numeric solutions and are represented once here.
        pool = []
        seen = set()
        for candidate in candidates + regenerated + refined:
            if id(candidate) not in seen:
                pool.append(candidate)
                seen.add(id(candidate))

        result["all_candidate_solutions"] = [diagnostic(c) for c in pool]
        for dim in range(len(self.dims)):
            group = [c for c in pool if c.dim == dim]
            if group:
                result["per_dimension_best"].append(diagnostic(min(group, key=lambda c: self._score(c, U, robust)[0])))
        accepted = [c for c in pool if self.known_dimension_index is None or c.dim == self.known_dimension_index]
        ranked = sorted(accepted, key=lambda c: self._score(c, U, robust)[0])
        chosen = ranked[0]
        score, inliers, residual = self._score(chosen, U, robust)
        result["best_candidate"] = diagnostic(chosen)
        fit_ids = tuple(fit_history.get(id(chosen), chosen.ids))
        assert set(fit_ids) <= allowed and not set(fit_ids) & set(blocked)
        result.update(input_inliers=list(fit_ids), fit_input_ids=list(fit_ids),
                      final_inliers=list(inliers), inliers=list(inliers),
                      residuals_used_px=residual.tolist(),
                      weak_four_point_consensus=len(inliers) == 4 if robust else len(U) == 4)
        if robust and len(inliers) < 4:
            result.update(state="INSUFFICIENT_CONSENSUS", reason="insufficient_consensus")
            return finish()
        rank_ids = inliers if robust else U
        jacobian = self._jacobian(chosen, rank_ids)
        result["geometry"].update(object=_shape(self.models[chosen.dim][list(rank_ids)]), jacobian=jacobian)
        if jacobian["numerical_rank"] < 6:
            result.update(state="NUMERICAL_RANK_DEFICIENT", reason="numerical_rank_deficient")
            return finish()
        objective_index = 1 if robust else 0
        for candidate in ranked[1:]:
            other_score = self._score(candidate, U, robust)[0]
            same_count = not robust or other_score[0] == score[0]
            tied = same_count and abs(other_score[objective_index] - score[objective_index]) <= POLICY["numerical_score_tie_px2"]
            if tied:
                result["numerical_tied_solution_count"] += 1
            if self._equivalent(candidate, chosen):
                continue
            info = diagnostic(candidate)
            info.update(numerical_objective_tie=bool(tied),
                projection_max_abs_from_selected_px=float(np.max(np.abs(candidate.projected - chosen.projected))),
                physical_rotation_max_abs_from_selected=float(np.max(np.abs(_physical_rotation(candidate) - _physical_rotation(chosen)))))
            result["alternatives"].append(info)
            result["multiple_solutions"] = True
            if tied:
                result["unresolved_ambiguity"] = True
        if result["unresolved_ambiguity"]:
            result.update(state="AMBIGUOUS_PNP", reason="unresolved_equal_score_physical_pose")
            return finish()
        output = self.points.copy()
        if H:
            output[list(H)] = chosen.projected[list(H)]
        assert np.array_equal(output[8:], self.points[8:], equal_nan=True)
        assert np.array_equal(output[list(set(temporary) - set(H))],
                              self.points[list(set(temporary) - set(H))], equal_nan=True)
        result.update(available=True, pose_available=True, new_pose_estimated=True, no_pose=False,
            state="NEW_POSE", reason="new_pose_estimated", output_status="NEW_POSE",
            R_cf=_rotation(chosen).tolist(), R_physical=_physical_rotation(chosen).tolist(),
            centroid=chosen.tvec.reshape(3).tolist(), cf_extents=self.dims[chosen.dim].tolist(),
            dimension_index=chosen.dim, selected_hypothesis=self.names[chosen.dim],
            projected=chosen.projected.tolist(), points_final=output.tolist(), hidden_reprojected=bool(H),
            reprojection_px=float(residual.mean()),
            truncated_sse_px2=float(np.minimum(residual ** 2, RESIDUAL_PX ** 2).sum()),
            sse_px2=float((residual ** 2).sum()), generator=chosen.generator,
            generator_ids=list(chosen.ids))
        return finish()


def refine(points, K, xyz, hidden=(), image_size=(640, 480), robust=True, *,
           excluded=(), bank=None, known_dimension_index=None, known_dimension_provenance=None):
    """Actual observations to an explicit new-pose or no-new-pose result."""
    if bank is None:
        bank = PoseBank(points, K, xyz, image_size=image_size,
                        known_dimension_index=known_dimension_index,
                        known_dimension_provenance=known_dimension_provenance)
    else:
        if not isinstance(bank, PoseBank):
            raise TypeError("bank must be the matching observation-only PoseBank")
        if not (np.array_equal(bank.points, np.asarray(points, float), equal_nan=True)
                and np.array_equal(bank.K, np.asarray(K, float))
                and np.array_equal(bank.xyz, np.asarray(xyz, float))
                and bank.image_size == tuple(image_size)):
            raise ValueError("Shared bank numeric inputs differ")
        if known_dimension_index is not None and bank.known_dimension_index != known_dimension_index:
            raise ValueError("Shared bank dimension constraint differs")
        if known_dimension_provenance is not None and bank.known_dimension_provenance != known_dimension_provenance:
            raise ValueError("Shared bank dimension provenance differs")
    return bank.solve(excluded=excluded, hidden=hidden, robust=robust)
