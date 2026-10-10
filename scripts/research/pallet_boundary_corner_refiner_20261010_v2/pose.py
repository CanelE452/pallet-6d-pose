"""Finite consensus with an explicit fixed-initial-pose identity prior.

This is a separate method. It does not change the published finite PnP solver.
All observations, including incorrect ones, retain their supplied corner IDs.
The prior's projections select a pose basin; they never become fit residuals.
"""
from __future__ import annotations

import copy
import cv2
import numpy as np

from ..pallet_observation_refiner_20261009_v1.solver import (
    HypothesisBank, RESIDUAL_PX, MAX_REFITS, _shape, cuboid, project,
)

POLICY = dict(
    observation_residual_px=RESIDUAL_PX,
    maximum_refit_starts=MAX_REFITS,
    minimum_correspondences=4,
    initial_dimension_prior=True,
    initial_projection_rms_radius_px=8.0,
    initial_projection_ids=list(range(8)),
    prior_projection_is_observation=False,
    prior_rotation_translation_residual=False,
    numerical_score_tie_px2=1e-8,
    equivalent_projection_max_abs_px=1e-5,
    ambiguity="equal inlier count and numerically tied objective, distinct all-eight projection",
    tie_order="prior projection RMS, dimension index, subset IDs, solution index, generator",
    global_uniqueness_proven=False,
    generators="unchanged SQPnPGeneric / exact-plane IPPE, all returned solutions",
    dimensions_generated="both registry dimension hypotheses, once per same-coordinate bank",
    other_dimension_used_for_acceptance=False,
    hidden_policy="exclude initial H from generators/fit; replace H after final R,t; never refit projections",
    fallback="driver returns the full frozen N3 coordinates/pose with a distinct status",
)


def _rms(a, b):
    return float(np.sqrt(np.mean(np.sum((np.asarray(a) - np.asarray(b)) ** 2, axis=1))))


class PoseBank(HypothesisBank):
    """Shared numeric bank; masks never regenerate its four-ID hypotheses."""

    def __init__(self, points, K, xyz, image_size=(640, 480)):
        if np.asarray(points).shape != (9, 2):
            raise ValueError("Expected eight corner observations and preserved center, shape (9,2)")
        self.image_size = tuple(image_size)
        super().__init__(points, K, xyz, image_size=self.image_size)

    def _prior(self, initial):
        if not isinstance(initial, dict):
            return None, "INITIAL_POSE_INVALID"
        if not initial.get("available"):
            return None, "INITIAL_POSE_UNAVAILABLE"
        try:
            R = np.asarray(initial["R_cf"], dtype=float)
            t = np.asarray(initial["centroid"], dtype=float)
            dims = np.asarray(initial["cf_extents"], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None, "INITIAL_POSE_INVALID"
        if (R.shape != (3, 3) or t.shape != (3,) or dims.shape != (3,)
                or not all(np.isfinite(v).all() for v in (R, t, dims))
                or not np.allclose(R @ R.T, np.eye(3), atol=1e-6, rtol=0)
                or abs(np.linalg.det(R) - 1.) > 1e-6):
            return None, "INITIAL_POSE_INVALID"
        matches = [i for i, d in enumerate(self.dims)
                   if np.allclose(d, dims, atol=1e-9, rtol=0)]
        if len(matches) != 1:
            return None, "INITIAL_DIMENSION_UNAVAILABLE"
        dim = matches[0]
        if (((R @ self.models[dim].T).T + t)[:, 2] <= 1e-9).any():
            return None, "INITIAL_POSE_INVALID"
        projected = project(self.models[dim], R, t, self.K)
        if not np.isfinite(projected).all():
            return None, "INITIAL_POSE_INVALID"
        return dict(dim=dim, projected=projected,
                    cf_extents=dims.tolist(), initial_label=initial.get("selected_hypothesis")), None

    def _ranked(self, candidates, U, robust, prior):
        ranked = sorted(candidates, key=lambda c: self._score(c, U, robust)[0])
        if not ranked:
            return ranked
        best = self._score(ranked[0], U, robust)[0]
        objective_index = 1 if robust else 0
        count = -best[0] if robust else None
        tied = []
        remainder = []
        for candidate in ranked:
            score = self._score(candidate, U, robust)[0]
            same_count = not robust or -score[0] == count
            if same_count and abs(score[objective_index] - best[objective_index]) <= POLICY["numerical_score_tie_px2"]:
                tied.append(candidate)
            else:
                remainder.append(candidate)
        tied.sort(key=lambda c: (_rms(c.projected, prior["projected"]),
                                 c.dim, c.ids, c.solution, c.generator))
        return tied + remainder

    def solve(self, initial_pose, hidden=(), robust=True):
        before = self.ledger.copy()
        H = tuple(sorted(set(int(i) for i in hidden)))
        if any(i < 0 or i >= 8 for i in H):
            raise ValueError("Only corner IDs 0..7 may be hidden")
        U = tuple(i for i in self.eligible if i not in H)
        result = dict(
            available=False, pose_available=False, new_pose_estimated=False,
            no_pose=True, fallback_used=False, state="INSUFFICIENT_OBSERVATIONS",
            reason="insufficient_observations", output_status="NO_NEW_POSE",
            eligible=list(self.eligible), used=list(U), excluded=list(H), hidden=list(H),
            ineligible=self.ineligible, inliers=[], input_inliers=[], final_inliers=[],
            fit_input_ids=[], input_hash=self.digest, points_final=self.points.tolist(),
            R_cf=None, R_physical=None, centroid=None, cf_extents=None, projected=None,
            solver="INITIAL_IDENTITY_FINITE_SUBSET_ROBUST" if robust else "INITIAL_IDENTITY_STANDARD",
            residual_threshold_px=RESIDUAL_PX, candidate_count=0,
            eligible_candidate_count=0, prior_accepted_candidate_count=0, refit_count=0,
            hidden_reprojected=False, hidden_initial_excluded=True,
            reprojected_points_reused_as_observations=False,
            prior_used=True, prior_projection_used_as_observation=False,
            multiple_solutions=False, unresolved_ambiguity=False,
            global_uniqueness_proven=False, alternatives=[], per_dimension_best=[],
            weak_four_point_consensus=len(U) == 4,
            geometry=dict(image=_shape(self.points[list(U)]) if U else None),
            policy=copy.deepcopy(POLICY),
        )

        def finish():
            result["operation_counts"] = {k: self.ledger[k] - before[k] for k in self.ledger}
            result["hypothesis_bank_counts"] = self.ledger.copy()
            return result

        prior, failure = self._prior(initial_pose)
        if failure:
            result.update(state=failure, reason=failure.lower(), prior_used=False)
            return finish()
        result["initial_identity_prior"] = dict(
            origin="frozen N3 initial pose, no truth input",
            dimension_index=prior["dim"], cf_extents=prior["cf_extents"],
            initial_label=prior["initial_label"], projected=prior["projected"].tolist(),
            radius_px=POLICY["initial_projection_rms_radius_px"],
            projection_ids=list(range(8)), residual_prior=False,
            projection_is_fit_observation=False, inherited_initial_errors_possible=True,
        )
        if len(U) < 4:
            return finish()
        if robust:
            self._build()
            candidates = [c for c in self.hypotheses if set(c.ids) <= set(U)]
            result["candidate_count"] = len(self.hypotheses)
        else:
            if U not in self.standard_cache:
                self.standard_cache[U] = [c for d in range(len(self.dims))
                                          for c in self._generic(U, d, "standard")]
            candidates = self.standard_cache[U]
            result["candidate_count"] = len(candidates)
        result["eligible_candidate_count"] = len(candidates)
        if not candidates:
            result.update(state="NUMERICAL_GENERATION_FAILURE",
                          reason="degenerate_or_numerical_generation_failure")
            return finish()

        def accepted(c):
            return (c.dim == prior["dim"] and
                    _rms(c.projected, prior["projected"]) <= POLICY["initial_projection_rms_radius_px"])

        def diagnostic(c):
            key, inliers, residual = self._score(c, U, robust)
            return dict(dimension_index=c.dim, dimensions=self.dims[c.dim].tolist(),
                        generator=c.generator, generator_ids=list(c.ids), solution_index=c.solution,
                        inlier_count=len(inliers), inlier_ids=list(inliers),
                        truncated_sse_px2=float(np.minimum(residual ** 2, RESIDUAL_PX ** 2).sum()),
                        sse_px2=float((residual ** 2).sum()),
                        prior_projection_rms_px=_rms(c.projected, prior["projected"]),
                        same_initial_dimension=c.dim == prior["dim"], accepted_by_prior=accepted(c),
                        R_cf=cv2.Rodrigues(c.rvec)[0].tolist(),
                        centroid=c.tvec.reshape(3).tolist(), projected=c.projected.tolist())

        # Opposing dimensions remain diagnostics. They are never fit acceptance candidates.
        for dim in range(len(self.dims)):
            group = [c for c in candidates if c.dim == dim]
            if group:
                result["per_dimension_best"].append(diagnostic(min(group, key=lambda c: self._score(c, U, robust)[0])))
        eligible = [c for c in candidates if accepted(c)]
        result["prior_accepted_candidate_count"] = len(eligible)
        result["dimension_rejected_candidate_count"] = sum(c.dim != prior["dim"] for c in candidates)
        result["projection_prior_rejected_candidate_count"] = sum(
            c.dim == prior["dim"] and not accepted(c) for c in candidates)
        if not eligible:
            result.update(state="BRANCH_JUMP_REJECTED", reason="initial_identity_prior_rejected_all_candidates")
            return finish()
        ranked = self._ranked(eligible, U, robust, prior)
        refined = []
        history = {}
        if robust:
            starts = []
            for candidate in ranked:
                if any(np.max(np.abs(candidate.projected - s.projected)) < 1e-6 for s in starts):
                    continue
                starts.append(candidate)
                if len(starts) == MAX_REFITS:
                    break
            for candidate in starts:
                ids = self._score(candidate, U, True)[1]
                if len(ids) < 4:
                    continue
                result["refit_count"] += 1
                regenerated = self._generic(ids, candidate.dim, "refit")
                regenerated = [c for c in regenerated if accepted(c)]
                seed = self._ranked(regenerated, U, True, prior)[0] if regenerated else candidate
                fitted = self._lm(seed, ids)
                if fitted is not None and accepted(fitted):
                    refined.append(fitted)
                    history[id(fitted)] = ids
            ranked = self._ranked(eligible + refined, U, True, prior)
        else:
            result["refit_count"] = 1
            fitted = self._lm(ranked[0], U)
            if fitted is not None and accepted(fitted):
                history[id(fitted)] = U
                ranked = self._ranked(eligible + [fitted], U, False, prior)
        chosen = ranked[0]
        score, inliers, residual = self._score(chosen, U, robust)
        result["best_candidate"] = diagnostic(chosen)
        result.update(input_inliers=list(history.get(id(chosen), chosen.ids)),
                      final_inliers=list(inliers), inliers=list(inliers),
                      residuals_used_px=residual.tolist(),
                      weak_four_point_consensus=len(inliers) == 4 if robust else len(U) == 4)
        result["fit_input_ids"] = result["input_inliers"]
        if robust and len(inliers) < 4:
            result.update(state="INSUFFICIENT_CONSENSUS", reason="insufficient_consensus")
            return finish()
        J = self._jacobian(chosen, inliers if robust else U)
        result["geometry"].update(object=_shape(self.models[chosen.dim][list(inliers if robust else U)]), jacobian=J)
        if J["numerical_rank"] < 6:
            result.update(state="NUMERICAL_RANK_DEFICIENT", reason="numerical_rank_deficient")
            return finish()
        objective_index = 1 if robust else 0
        for candidate in ranked[1:]:
            if np.max(np.abs(candidate.projected - chosen.projected)) < POLICY["equivalent_projection_max_abs_px"]:
                continue
            info = diagnostic(candidate)
            other_score = self._score(candidate, U, robust)[0]
            same_count = not robust or other_score[0] == score[0]
            tied = same_count and abs(other_score[objective_index] - score[objective_index]) <= POLICY["numerical_score_tie_px2"]
            info.update(numerical_objective_tie=bool(tied),
                        projection_rms_from_selected_px=_rms(candidate.projected, chosen.projected))
            result["multiple_solutions"] = True
            if len(result["alternatives"]) < 3:
                result["alternatives"].append(info)
            if tied:
                result.update(state="AMBIGUOUS_PNP", reason="unresolved_equal_score_pose_basin",
                              unresolved_ambiguity=True)
                return finish()
        R = cv2.Rodrigues(chosen.rvec)[0]
        Q = np.eye(3) if chosen.dim == 0 else np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        output = self.points.copy()
        if H:
            output[list(H)] = chosen.projected[list(H)]
        assert not set(H) & set(result["fit_input_ids"])
        assert np.array_equal(output[8:], self.points[8:], equal_nan=True)
        result.update(available=True, pose_available=True, new_pose_estimated=True, no_pose=False,
                      state="NEW_POSE", reason="new_pose_estimated", output_status="NEW_POSE",
                      R_cf=R.tolist(), R_physical=(R @ Q).tolist(), centroid=chosen.tvec.reshape(3).tolist(),
                      cf_extents=self.dims[chosen.dim].tolist(), selected_hypothesis=self.names[chosen.dim],
                      projected=chosen.projected.tolist(), points_final=output.tolist(),
                      hidden_reprojected=bool(H), prior_projection_rms_px=_rms(chosen.projected, prior["projected"]),
                      reprojection_px=float(residual.mean()),
                      truncated_sse_px2=float(np.minimum(residual ** 2, RESIDUAL_PX ** 2).sum()),
                      sse_px2=float((residual ** 2).sum()), generator=chosen.generator,
                      generator_ids=list(chosen.ids))
        return finish()


def refine(points, K, xyz, initial_pose, hidden, image_size, robust=True, *, bank=None):
    """Sparse actual observations -> new pose, or an explicit no-new-pose state.

    A caller that compares masks supplies the same ``PoseBank``. No fallback
    observations or initial pose coordinates are manufactured by this function.
    """
    if bank is None:
        bank = PoseBank(points, K, xyz, image_size=image_size)
    else:
        if not isinstance(bank, PoseBank):
            raise TypeError("bank must be the matching PoseBank")
        if not (np.array_equal(bank.points, np.asarray(points, float), equal_nan=True)
                and np.array_equal(bank.K, np.asarray(K, float))
                and np.array_equal(bank.xyz, np.asarray(xyz, float))
                and bank.image_size == tuple(image_size)):
            raise ValueError("Shared bank numeric inputs differ")
    return bank.solve(initial_pose, hidden=hidden, robust=robust)
