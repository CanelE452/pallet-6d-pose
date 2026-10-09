"""Frozen finite-subset PnP, with no truth/visibility or confidence inputs.

OpenCV 4.9.0 documented SQPnP and generic IPPE are standard algorithms.
Exact planes use IPPE after an orthonormal plane-coordinate transformation;
nonplanes use SQPnP. Every returned solution is retained. IPPE_SQUARE is never
used. Distortion remains None, matching the historical evaluation contract.
The 8 px threshold and global top-three refits were fixed before real scoring.
"""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

import cv2
import numpy as np

RESIDUAL_PX = 8.0
MAX_REFITS = 3
POLICY = dict(residual_px=RESIDUAL_PX, max_refits=MAX_REFITS,
              exact_plane="IPPE in orthonormal local coordinates, all solutions",
              nonplane="SQPnPGeneric, all solutions", distortion="None",
              tie_break="dimension index, subset lexicographic, solution index",
              initial_pose_prior=False, robust_extra_all_pool_candidate=False,
              reference="https://github.com/opencv/opencv/blob/4.9.0/modules/calib3d/doc/solvePnP.markdown")


def cuboid(width, height, depth):
    a, b, c = width / 2., height / 2., depth / 2.
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
                     [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]], dtype=np.float64)


def project(model, R, t, K):
    xyz = (np.asarray(R) @ np.asarray(model).T).T + np.asarray(t).reshape(3)
    uvw = (np.asarray(K) @ xyz.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def visibility(model, R, t, margin_deg=2.0):
    """Cuboid proxy only; no claim of actual mesh or external occlusion."""
    camera = -np.asarray(R).T @ np.asarray(t).reshape(3)
    rays = camera[None, :] - np.asarray(model)
    cosines = np.sign(model) * rays / np.linalg.norm(rays, axis=1)[:, None]
    best = cosines.max(1)
    margin = np.sin(np.deg2rad(margin_deg))
    return best < -margin, best > margin, best


def _shape(points):
    p = np.asarray(points, np.float64)
    s = np.linalg.svd(p - p.mean(0), compute_uv=False)
    rel = s / max(float(s[0]), 1e-300)
    return dict(singular_values=s.tolist(), relative_singular_values=rel.tolist(),
                numerical_rank=int(np.sum(rel > 1e-10)))


@dataclass
class Candidate:
    dim: int
    ids: tuple
    solution: int
    generator: str
    rvec: np.ndarray
    tvec: np.ndarray
    projected: np.ndarray


class HypothesisBank:
    """A single frame's immutable numeric inputs and lazy shared hypothesis bank.

    `xyz` is registry (width,height,depth) in meters. `image_size` is (w,h).
    At most 70 four-ID subsets per registered dimension are considered once.
    A mask never changes hypothesis coordinates, K, or dimensions. Only allowed
    generators and scoring IDs U change. Returned coordinates preserve center 8.
    """
    def __init__(self, points, K, xyz, image_size=(640, 480)):
        self.points = np.asarray(points, np.float64).copy()
        self.K = np.asarray(K, np.float64).copy()
        self.xyz = np.asarray(xyz, np.float64).copy()
        if self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 8:
            raise ValueError("points must contain at least eight two-dimensional corners")
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all():
            raise ValueError("K must be finite 3x3")
        if self.xyz.shape != (3,) or not np.isfinite(self.xyz).all() or (self.xyz <= 0).any():
            raise ValueError("xyz must be positive finite (width,height,depth)")
        w, h = image_size
        p = self.points[:8]
        finite = np.isfinite(p).all(1)
        sentinel = (p == -1).all(1)
        inframe = (p[:,0] >= 0) & (p[:,0] < w) & (p[:,1] >= 0) & (p[:,1] < h)
        self.eligible = tuple(np.flatnonzero(finite & ~sentinel & inframe).tolist())
        self.ineligible = dict(nonfinite=np.flatnonzero(~finite).tolist(),
                               sentinel=np.flatnonzero(sentinel).tolist(),
                               out_of_frame=np.flatnonzero(finite & ~sentinel & ~inframe).tolist())
        self.dims = [self.xyz.copy()]
        self.names = ["REGISTRY_WD"]
        if abs(self.xyz[0] - self.xyz[2]) >= 1e-9:
            self.dims.append(self.xyz[[2,1,0]].copy())
            self.names.append("REGISTRY_DW")
        self.models = [cuboid(*d) for d in self.dims]
        self.hypotheses = None
        self.standard_cache = {}
        self.ledger = dict(generic_calls=0, subset_generic_calls=0,
                           standard_generic_calls=0, refit_generic_calls=0,
                           lm_calls=0, subsets_considered=0, returned_solutions=0,
                           invalid_solutions=0, generic_errors=0, geometry_rejected=0)
        self.digest = hashlib.sha256(b"".join(v.tobytes() for v in
                                    [self.points, self.K, self.xyz]) + str(image_size).encode()).hexdigest()

    def _valid(self, rvec, tvec, dim):
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return False
        R = cv2.Rodrigues(rvec)[0]
        if not np.isfinite(R).all() or abs(np.linalg.det(R) - 1.) > 1e-6:
            return False
        return bool((((R @ self.models[dim].T).T + tvec.reshape(3))[:,2] > 1e-9).all())

    def _generic(self, ids, dim, phase):
        ids = tuple(ids)
        X = self.models[dim][list(ids)]
        q = np.ascontiguousarray(self.points[list(ids)], np.float64)
        if _shape(X)["numerical_rank"] < 2 or _shape(q)["numerical_rank"] < 2:
            self.ledger["geometry_rejected"] += 1
            return []
        center = X.mean(0)
        _, s, vt = np.linalg.svd(X - center, full_matrices=True)
        exact_plane = s[-1] <= max(s[0], 1.) * 1e-12
        flag = cv2.SOLVEPNP_SQPNP
        A = np.eye(3)
        local = np.ascontiguousarray(X)
        generator = "SQPNP"
        if exact_plane:
            # A columns are the local axes in object coordinates, det(A)=+1.
            first, second = vt[0], vt[1]
            A = np.column_stack([first, second, np.cross(first, second)])
            local = np.ascontiguousarray((X - center) @ A)
            local[:,2] = 0.
            flag = cv2.SOLVEPNP_IPPE
            generator = "IPPE_PLANE"
        self.ledger["generic_calls"] += 1
        self.ledger[phase + "_generic_calls"] += 1
        try:
            ret = cv2.solvePnPGeneric(local, q, self.K, None, flags=flag)
        except cv2.error:
            self.ledger["generic_errors"] += 1
            return []
        out = []
        if not ret[0]:
            return out
        for j, (rvec, tvec) in enumerate(zip(ret[1], ret[2])):
            self.ledger["returned_solutions"] += 1
            rvec = np.asarray(rvec, np.float64).reshape(3,1)
            tvec = np.asarray(tvec, np.float64).reshape(3,1)
            if exact_plane:
                R = cv2.Rodrigues(rvec)[0] @ A.T
                rvec = cv2.Rodrigues(R)[0]
                tvec = tvec - R @ center.reshape(3,1)
            if not self._valid(rvec, tvec, dim):
                self.ledger["invalid_solutions"] += 1
                continue
            R = cv2.Rodrigues(rvec)[0]
            pr = project(self.models[dim], R, tvec, self.K)
            out.append(Candidate(dim, ids, j, generator, rvec.copy(), tvec.copy(), pr))
        return out

    def _build(self):
        if self.hypotheses is not None:
            return
        self.hypotheses = []
        for dim in range(len(self.dims)):
            for ids in itertools.combinations(self.eligible, 4):
                self.ledger["subsets_considered"] += 1
                self.hypotheses.extend(self._generic(ids, dim, "subset"))

    def _score(self, candidate, ids, robust):
        residual = np.linalg.norm(candidate.projected[list(ids)] - self.points[list(ids)], axis=1)
        inliers = tuple(i for i, r in zip(ids, residual) if r <= RESIDUAL_PX)
        loss = float(np.minimum(residual ** 2, RESIDUAL_PX ** 2).sum()) if robust else float((residual ** 2).sum())
        key = ((-len(inliers), loss) if robust else (loss,)) + (candidate.dim, candidate.ids, candidate.solution, candidate.generator)
        return key, inliers, residual

    def _lm(self, candidate, fit_ids):
        self.ledger["lm_calls"] += 1
        try:
            rv, tv = cv2.solvePnPRefineLM(self.models[candidate.dim][list(fit_ids)],
                         self.points[list(fit_ids)], self.K, None,
                         candidate.rvec.copy(), candidate.tvec.copy())
        except cv2.error:
            return None
        if not self._valid(rv, tv, candidate.dim):
            return None
        pr = project(self.models[candidate.dim], cv2.Rodrigues(rv)[0], tv, self.K)
        return Candidate(candidate.dim, tuple(fit_ids), candidate.solution,
                         candidate.generator + "+LM", rv, tv, pr)

    def _jacobian(self, candidate, fit_ids):
        _, jac = cv2.projectPoints(self.models[candidate.dim][list(fit_ids)],
                           candidate.rvec, candidate.tvec, self.K, None)
        J = jac[:, :6]
        norms = np.linalg.norm(J, axis=0)
        J = J / np.maximum(norms, 1e-300)[None,:]
        s = np.linalg.svd(J, compute_uv=False)
        rank = int(np.sum(s > s[0] * 1e-10))
        return dict(normalization="unit L2 Jacobian columns, rotation and translation",
                    singular_values=s.tolist(), numerical_rank=rank,
                    condition_number=float(s[0] / s[-1]) if s[-1] > 0 else None,
                    weak_condition=bool(s[-1] < s[0] * 1e-4))

    def solve(self, excluded=(), robust=True, hidden=()):
        before = self.ledger.copy()
        hidden = tuple(sorted(set(int(i) for i in hidden)))
        excluded = tuple(sorted(set(int(i) for i in excluded) | set(hidden)))
        if any(i < 0 or i >= 8 for i in excluded):
            raise ValueError("Only physical corner IDs 0..7 may be excluded/reprojected")
        U = tuple(i for i in self.eligible if i not in excluded)
        result = dict(available=False, pose_available=False, new_pose_estimated=False,
                      fallback_used=False, no_pose=True, reason="insufficient_observations",
                      eligible=list(self.eligible), used=list(U), excluded=list(excluded),
                      hidden=list(hidden), ineligible=self.ineligible, inliers=[],
                      input_inliers=[], final_inliers=[], fit_input_ids=[],
                      hidden_reprojected=False, input_hash=self.digest,
                      solver="FINITE_SUBSET_ROBUST" if robust else "STANDARD",
                      residual_threshold_px=RESIDUAL_PX, prior_used=False,
                      candidate_count=0, eligible_candidate_count=0, refit_count=0,
                      points_final=self.points.tolist(), R_cf=None, R_physical=None,
                      centroid=None, cf_extents=None, projected=None,
                      geometry=dict(image=_shape(self.points[list(U)]) if U else None),
                      weak_four_point_consensus=len(U)==4, multiple_solutions=False)

        def finish():
            result["operation_counts"] = {k:self.ledger[k]-before[k] for k in self.ledger}
            result["hypothesis_bank_counts"] = self.ledger.copy()
            return result

        if len(U) < 4:
            return finish()
        if robust:
            self._build()
            allowed = set(U)
            candidates = [c for c in self.hypotheses if set(c.ids) <= allowed]
        else:
            if U not in self.standard_cache:
                self.standard_cache[U] = [c for d in range(len(self.dims))
                                             for c in self._generic(U, d, "standard")]
            candidates = self.standard_cache[U]
        result["candidate_count"] = len(self.hypotheses or []) if robust else len(candidates)
        result["eligible_candidate_count"] = len(candidates)
        if not candidates:
            result["reason"] = "degenerate_or_numerical_generation_failure"
            return finish()
        ranked = sorted(candidates, key=lambda c:self._score(c,U,robust)[0])
        refined = []
        fit_history = {}
        if robust:
            # Equivalent projected hypotheses are not independent alternative starts.
            starts = []
            for c in ranked:
                if any(c.dim==s.dim and np.max(np.abs(c.projected-s.projected)) < 1e-6 for s in starts):
                    continue
                starts.append(c)
                if len(starts) == MAX_REFITS:
                    break
            for c in starts:
                initial_inliers = self._score(c,U,True)[1]
                if len(initial_inliers) < 4:
                    continue
                result["refit_count"] += 1
                regenerated = self._generic(initial_inliers, c.dim, "refit")
                if regenerated:
                    seed = min(regenerated, key=lambda h:self._score(h,U,True)[0])
                else:
                    seed = c
                final = self._lm(seed, initial_inliers)
                if final is not None:
                    refined.append(final)
                    fit_history[id(final)] = initial_inliers
            ranked = sorted(candidates + refined, key=lambda c:self._score(c,U,True)[0])
        else:
            seed = ranked[0]
            result["refit_count"] = 1
            final = self._lm(seed,U)
            if final is not None:
                ranked = [final] + ranked[1:]
                fit_history[id(final)] = U
        chosen = ranked[0]
        key, inliers, residual = self._score(chosen,U,robust)
        result["input_inliers"] = list(fit_history.get(id(chosen), chosen.ids))
        result["fit_input_ids"] = result["input_inliers"]
        result["final_inliers"] = result["inliers"] = list(inliers)
        result["residuals_used_px"] = residual.tolist()
        result["weak_four_point_consensus"] = len(inliers)==4 if robust else len(U)==4
        if robust and len(inliers) < 4:
            result["reason"] = "insufficient_consensus"
            return finish()
        fit_ids = inliers if robust else U
        jac = self._jacobian(chosen,fit_ids)
        result["geometry"].update(object=_shape(self.models[chosen.dim][list(fit_ids)]), jacobian=jac)
        if jac["numerical_rank"] < 6:
            result["reason"] = "numerical_rank_deficient"
            return finish()
        alternatives = []
        for c in ranked[1:]:
            if c.dim==chosen.dim and np.max(np.abs(c.projected-chosen.projected)) < 1e-5:
                continue
            altkey, altin, _ = self._score(c,U,robust)
            alternatives.append(dict(dimensions=self.dims[c.dim].tolist(),
                     generator=c.generator, generator_ids=list(c.ids),
                     inlier_count=len(altin), truncated_sse_px2=float(np.minimum(
                         np.linalg.norm(c.projected[list(U)]-self.points[list(U)],axis=1)**2,
                         RESIDUAL_PX**2).sum()),
                     sse_px2=float((np.linalg.norm(c.projected[list(U)]-self.points[list(U)],axis=1)**2).sum()),
                     R_cf=cv2.Rodrigues(c.rvec)[0].tolist(), centroid=c.tvec.reshape(3).tolist()))
            if len(alternatives)==2:
                break
        R = cv2.Rodrigues(chosen.rvec)[0]
        Q = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]) if chosen.dim else np.eye(3)
        output = self.points.copy()
        if hidden:
            output[list(hidden)] = chosen.projected[list(hidden)]
        assert not set(hidden) & set(result["fit_input_ids"])
        assert not set(excluded) & set(U)
        if len(output)>8:
            assert np.array_equal(output[8:],self.points[8:],equal_nan=True)
        score_sse = float(np.minimum(residual**2,RESIDUAL_PX**2).sum())
        alternative = alternatives[0] if alternatives else None
        result.update(available=True, pose_available=True, new_pose_estimated=True,
                      no_pose=False, reason="new_pose_estimated", R_cf=R.tolist(),
                      R_physical=(R@Q).tolist(), centroid=chosen.tvec.reshape(3).tolist(),
                      cf_extents=self.dims[chosen.dim].tolist(),
                      selected_hypothesis=self.names[chosen.dim],
                      projected=chosen.projected.tolist(), points_final=output.tolist(),
                      hidden_reprojected=bool(hidden), hidden_initial_excluded=True,
                      reprojection_px=float(residual.mean()),
                      truncated_sse_px2=score_sse, sse_px2=float((residual**2).sum()),
                      generator=chosen.generator, generator_ids=list(chosen.ids),
                      alternatives=alternatives, multiple_solutions=bool(alternatives),
                      alternative_inlier_gap=len(inliers)-alternative["inlier_count"] if alternative else None,
                      alternative_score_gap_px2=alternative["truncated_sse_px2"]-score_sse if alternative else None,
                      reprojected_points_reused_as_observations=False)
        return finish()
