"""One local C2 refinement of actual sparse ROLE points and unused lines.

The sealed initial N3 pose and an available point-only pose supply optimizer
starts, never residual priors or observations. Both registered dimensions are
tried for every available start. This is local estimation, not independent
PnP/PnL initialization or a certificate of global uniqueness.

The unchanged V5 factor validator and exact forward/model-normal Jacobians
are reused. Its four-point solver is deliberately not called. A finite local
optimizer result must have rank six in both Jacobians of the actual factor
pool. Eight-pixel inliers are diagnostics, not a replacement four-point gate.
"""
from __future__ import annotations

import copy
import hashlib
import json

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..pallet_cornerwise_independent_20261010_v4 import pose as P
from ..pallet_partial_line_independent_20261010_v5 import solver as V5
from ..pallet_observation_refiner_20261009_v1.solver import Candidate


POLICY = dict(
    schema='same_sparse_ROLE_local_point_line_policy_v9',
    loss='soft_l1', f_scale_px=8., max_nfev=50,
    point_weight=1., line_weight=1 / np.sqrt(2),
    score='minimum whole fixed factor-pool soft_l1 cost; SSE/inliers diagnostic only',
    soft_l1_cost='64*sum(sqrt(1+(scalar_residual/8)**2)-1)',
    inlier_diagnostic_threshold_px=8., minimum_actual_points=0,
    no_minimum_point_inlier_gate=True,
    minimum_scalar_residuals=6,
    numerical_rank_relative_threshold=V5.POLICY['numerical_rank_relative_threshold'],
    numerical_score_tie_px2=P.POLICY['numerical_score_tie_px2'],
    numerical_equivalence={k: v for k, v in P.POLICY.items() if k.startswith('equivalent_')},
    candidate_acceptance='optimizer success, finite, all-eight positive depth, observed and modeled-normal actual-pool rank6',
    rank_is_local_only=True, global_unique_pose_proven=False,
    optimizer_starts='sealed initial N3 and available sealed point-only pose; each remapped to both registry dimensions',
    maximum_start_sources=2, maximum_dimension_starts=4,
    initial_pose_start=True, initial_pose_residual_prior=False,
    initial_dimension_choice_is_not_a_constraint=True,
    duplicate_policy=V5.POLICY['duplicate_policy'],
    source_registry_supported_edges=list(V5.POLICY['source_registry_supported_edges']),
    source_CAL_support_required_by_packet=True,
    same_factor_pool_for_all_starts=True,
    H_endpoint_policy=V5.POLICY['H_endpoint_policy'],
    missing_sparse_filled_for_numeric_fit=False,
    independent_PnP_or_PnL=False, random_starts=False,
    hidden_reprojection='NEW final pose only; never fit projected points or lines',
    real_physical_boundary_ownership_independently_certified=False,
    initial_rank='diagnostic only; it never vetoes an optimizer start',
    empty_unused_line_pool='keep original v1 C2 point-local objective if scalar factors>=6; report separately from line effects',
)


def _copy(value):
    return copy.deepcopy(value)


def _serial(value):
    if isinstance(value, np.ndarray):
        return _serial(value.tolist())
    if isinstance(value, np.generic):
        return _serial(value.item())
    if isinstance(value, dict):
        return {str(k): _serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serial(v) for v in value]
    return value


def _binding(value):
    data = json.dumps(_serial(value), sort_keys=True, separators=(',', ':'), allow_nan=True).encode()
    return dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                definition='semantic sorted JSON; not file-byte provenance')


def soft_l1_cost(residual):
    residual = np.asarray(residual, dtype=float)
    return float(64. * np.sum(np.sqrt(1. + (residual / 8.) ** 2) - 1.))


class LocalPointLineBank:
    """One immutable observation packet; no models, images, truth or files."""

    def __init__(self, points, K, xyz, image_size=(640, 480)):
        self.factor_validator = V5.PointLineBank(points, K, xyz, image_size=image_size)
        self.bank = self.factor_validator.bank
        self.points, self.K, self.xyz = self.bank.points, self.bank.K, self.bank.xyz
        self.counts = dict(local_solve_calls=0, start_sources_examined=0,
            local_start_entries=0, local_optimizer_calls=0, local_optimizer_completions=0,
            local_start_cache_hits=0, optimizer_residual_calls=0, optimizer_jacobian_calls=0,
            optimizer_projectPoints_calls=0, geometry_diagnostic_calls=0,
            diagnostic_projectPoints_calls=0, candidate_Rodrigues_calls=0,
            seed_Rodrigues_calls=0, candidate_validation_calls=0,
            hidden_reprojection_assignments=0)

    def _seed(self, raw, source):
        witness = dict(source=source, supplied=raw is not None, accepted=False,
                       available=bool(isinstance(raw, dict) and raw.get('available')),
                       reason='POSE_UNAVAILABLE', original_pose=_copy(raw))
        if not witness['available']:
            return None, witness
        try:
            Rcf, Rphysical = np.asarray(raw['R_cf'], float), np.asarray(raw['R_physical'], float)
            centroid, extents = np.asarray(raw['centroid'], float), np.asarray(raw['cf_extents'], float)
            if Rcf.shape != (3, 3) or Rphysical.shape != (3, 3) or centroid.shape != (3,) or extents.shape != (3,):
                raise ValueError('SEED_POSE_SHAPE')
            if not all(np.isfinite(a).all() for a in (Rcf, Rphysical, centroid, extents)):
                raise ValueError('SEED_POSE_NONFINITE')
            for rotation in (Rcf, Rphysical):
                if (abs(np.linalg.det(rotation) - 1.) > 1e-6 or
                        np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-6):
                    raise ValueError('SEED_ROTATION_INVALID')
            matches = [i for i, dims in enumerate(self.bank.dims) if np.max(np.abs(extents - dims)) <= 1e-6]
            if len(matches) != 1:
                raise ValueError('SEED_REGISTRY_EXTENTS_INVALID')
            d = matches[0]
            Q = np.eye(3) if d == 0 else np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
            if np.max(np.abs(Rphysical - Rcf @ Q)) > 1e-6:
                raise ValueError('SEED_PHYSICAL_ROTATION_CONVENTION')
            witness.update(accepted=True, reason='AVAILABLE_POSE_USED_AS_LOCAL_START_ONLY',
                           recorded_seed_dimension_index=d,
                           all_registry_dimensions_will_be_tried=True,
                           residual_prior=False, pose_as_observation=False)
            return (Rphysical, centroid), witness
        except (KeyError, TypeError, ValueError) as error:
            witness['reason'] = str(error)
            return None, witness

    def _geometry(self, model, z, point_ids, lines):
        self.counts['geometry_diagnostic_calls'] += 1
        def count_projection():
            self.counts['diagnostic_projectPoints_calls'] += 1
        info = V5.geometry_info(model, z, self.K, self.points, point_ids, lines,
                                projection_counter=count_projection)
        self.counts['diagnostic_projectPoints_calls'] += 1
        residual, observed, projected = V5.residual_jacobian(model, z, self.K, self.points, point_ids, lines)
        self.counts['diagnostic_projectPoints_calls'] += 1
        _, modeled, _ = V5.residual_jacobian(model, z, self.K, self.points, point_ids, lines, modeled_normals=True)
        info.update(observed_normal_J=observed.tolist(), modeled_normal_J=modeled.tolist(),
                    actual_scalar_residuals_px=residual.tolist())
        return info, residual, projected

    def solve(self, observation, adopted_boundary_corner_ids=(), hidden=(), *,
              initial_pose=None, point_result=None):
        before = self.counts.copy()
        self.counts['local_solve_calls'] += 1
        H = P._ids(hidden, 'hidden')
        adopted = P._ids(adopted_boundary_corner_ids, 'adopted_boundary_corner_ids')
        U = tuple(k for k in self.bank.eligible if k not in H)
        result = dict(available=False, pose_available=False, new_pose_estimated=False,
            no_pose=True, fallback_used=False, state='INSUFFICIENT_LOCAL_FACTORS',
            reason='insufficient_actual_point_line_factors', output_status='NO_NEW_POSE',
            solver='LOCAL_POINT_LINE', policy=_copy(POLICY), eligible=list(self.bank.eligible),
            used=list(U), excluded=list(H), hidden=list(H), actual_self_hidden_ids=list(H),
            temporary_excluded=[], ineligible=_copy(self.bank.ineligible), input_hash=self.bank.digest,
            fit_input_ids=[], input_inliers=[], final_inliers=[], inliers=[],
            fit_line_edges=[], final_line_inliers=[], line_edges=[], consumed_edges=[],
            all_candidate_solutions=[], optimizer_attempts=[], start_source_witnesses=[],
            alternatives=[], per_dimension_best=[], candidate_count=0, refit_count=0,
            multiple_solutions=False, unresolved_ambiguity=False,
            numerical_tied_solution_count=0, numerical_duplicate_solution_count=0,
            R_cf=None, R_physical=None, centroid=None, cf_extents=None, projected=None,
            points_final=self.points.tolist(), point_line_geometry=None,
            prior_used=False, initial_pose_used=False, initial_pose_start_used=False,
            initial_projection_used=False, initial_dimension_prior_used=False,
            known_dimension_constraint_used=False, independent_PnP_or_PnL=False,
            local_initialization_only=True, rank_is_local_only=True, global_uniqueness_proven=False,
            hidden_initial_excluded=True, hidden_reprojected=False,
            reprojected_points_reused_as_observations=False,
            same_edge_point_line_double_count=False, derived_line_points=0,
            excluded_image_coordinates_used_for_scoring=False,
            excluded_image_coordinates_used_for_equivalence=False,
            equivalence_uses_all_eight_model_predictions=True,
            inlier_counts_are_diagnostic_only=True, no_minimum_point_inlier_gate=True,
            observation_binding=_binding(observation), initial_pose_residual_prior=False,
            local_refinement_estimated=False,
            final_factor_pool_is_same_for_every_start=True,
            source_line_support_is_not_real_ownership_certificate=True)

        def finish():
            result['operation_counts'] = {k: self.counts[k] - before[k] for k in self.counts}
            result['hypothesis_bank_counts'] = self.bank.ledger.copy()
            result['local_bank_counts'] = self.counts.copy()
            result['initial_pose_used'] = bool(result['initial_pose_start_used'])
            return result

        try:
            lines, consumed, checks = self.factor_validator._lines(observation, adopted, set(U))
        except (KeyError, TypeError, ValueError, IndexError) as error:
            result.update(state='INVALID_OBSERVATION_CONTRACT', reason=str(error))
            return finish()
        edges = tuple(l['edge'] for l in lines)
        result.update(line_edges=list(edges), consumed_edges=consumed, line_contract_checks=checks,
            actual_adopted_boundary_ids=[k for k in adopted if k in U],
            factor_pool=dict(point_ids=list(U), line_edges=list(edges), actual_point_factors=len(U),
                unique_line_factors=len(lines), scalar_residuals=2 * (len(U) + len(lines)),
                line_support_queries=sum(len(l['support_query_ids']) for l in lines)),
            factor_pool_binding=_binding(dict(points=self.points[list(U)], point_ids=U,
                lines=[dict(edge=l['edge'], normal=l['normal'], offset=l['offset'],
                            support_query_ids=l['support_query_ids']) for l in lines])),
            registered_dimensions=[dict(index=i, name=self.bank.names[i], extents=d.tolist())
                                   for i, d in enumerate(self.bank.dims)])
        if 2 * (len(U) + len(lines)) < 6:
            return finish()
        if not V5._camera(self.K):
            result.update(state='UNSUPPORTED_CAMERA_CONTRACT', reason='forward_Jacobian_requires_frozen_zero_skew_pinhole_K')
            return finish()

        seeds = []
        for source, raw in (('SEALED_INITIAL_N3', initial_pose), ('SEALED_AVAILABLE_POINT_ONLY', point_result)):
            self.counts['start_sources_examined'] += 1
            seed, witness = self._seed(raw, source)
            result['start_source_witnesses'].append(witness)
            if seed is not None:
                seeds.append((source, seed))
        if not seeds:
            result.update(state='NO_LOCAL_INITIALIZATION', reason='no_available_valid_sealed_optimizer_start')
            return finish()

        # Cache only exactly identical dimension/start vectors for this fixed
        # factor pool. Every logical entry and reused fit are still witnessed.
        cache, candidates, failures = {}, [], []
        for source, (Rphysical, centroid) in seeds:
            for dim, model in enumerate(self.bank.models):
                Q = np.eye(3) if dim == 0 else np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
                self.counts['seed_Rodrigues_calls'] += 1
                z0 = np.r_[cv2.Rodrigues(Rphysical @ Q.T)[0].reshape(3), centroid]
                self.counts['local_start_entries'] += 1
                entry = dict(start_index=len(result['optimizer_attempts']), source=source, dimension_index=dim,
                    registry_name=self.bank.names[dim], initial_z=z0.tolist(), initial_R_physical=Rphysical.tolist(),
                    fit_point_ids=list(U), fit_line_edges=list(edges), factor_pool_binding=_copy(result['factor_pool_binding']),
                    initial_pose_residual_prior=False, actual_optimizer_called=False, cache_reused=False,
                    success=False, accepted=False, rejected=None)
                result['initial_pose_start_used'] = True
                key = (dim, z0.astype('<f8').tobytes())
                if key in cache:
                    self.counts['local_start_cache_hits'] += 1
                    cached_index, cached = cache[key]
                    entry.update(_copy(cached), start_index=entry['start_index'], source=source,
                        actual_optimizer_called=False, cache_reused=True, cache_source_start_index=cached_index)
                    result['optimizer_attempts'].append(entry)
                    continue
                try:
                    # A collapsed modeled edge can make this diagnostic
                    # unavailable at the start while the observed-normal
                    # residual/Jacobian used by the optimizer stays finite.
                    # Only the final two rank tests are acceptance gates.
                    try:
                        initial_geometry, _, _ = self._geometry(model, z0, U, lines)
                        entry['initial_geometry'] = initial_geometry
                        entry['initial_geometry_unavailable'] = None
                    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
                        entry['initial_geometry'] = None
                        entry['initial_geometry_unavailable'] = dict(type=type(error).__name__, message=str(error))
                    def residual(z):
                        self.counts['optimizer_residual_calls'] += 1
                        self.counts['optimizer_projectPoints_calls'] += 1
                        return V5.residual_jacobian(model, z, self.K, self.points, U, lines)[0]

                    def jacobian(z):
                        self.counts['optimizer_jacobian_calls'] += 1
                        self.counts['optimizer_projectPoints_calls'] += 1
                        return V5.residual_jacobian(model, z, self.K, self.points, U, lines)[1]

                    entry['actual_optimizer_called'] = True
                    self.counts['local_optimizer_calls'] += 1
                    result['refit_count'] += 1
                    fit = least_squares(residual, z0, jac=jacobian, loss='soft_l1', f_scale=8., max_nfev=50)
                    self.counts['local_optimizer_completions'] += 1
                    entry.update(success=bool(fit.success), scipy_status=int(fit.status), nfev=int(fit.nfev),
                        njev=int(fit.njev) if fit.njev is not None else None, reported_soft_l1_cost=float(fit.cost),
                        final_z=np.asarray(fit.x, float).tolist())
                    self.counts['candidate_validation_calls'] += 1
                    if not fit.success or not self.bank._valid(np.asarray(fit.x[:3]), np.asarray(fit.x[3:]), dim):
                        entry['rejected'] = 'OPTIMIZER_FAILURE_OR_NONFINITE_OR_NONPOSITIVE_DEPTH'
                        failures.append(entry['rejected'])
                    else:
                        geometry, rr, projected = self._geometry(model, np.asarray(fit.x, float), U, lines)
                        if (not np.isfinite(rr).all() or not np.isfinite(projected).all() or
                                not np.isfinite(np.asarray(geometry['observed_normal_J'])).all() or
                                not np.isfinite(np.asarray(geometry['modeled_normal_J'])).all()):
                            raise ValueError('NONFINITE_FINAL_ACTUAL_FACTOR_GEOMETRY')
                        cost = soft_l1_cost(rr)
                        if not np.isfinite(fit.cost) or not np.isclose(cost, fit.cost, rtol=1e-9, atol=1e-8):
                            raise ValueError('OPTIMIZER_WHOLE_POOL_COST_MISMATCH')
                        self.counts['candidate_Rodrigues_calls'] += 1
                        Rcf = cv2.Rodrigues(np.asarray(fit.x[:3]))[0]
                        pnorm = np.linalg.norm(projected[list(U)] - self.points[list(U)], axis=1)
                        lnormal = [projected[list(V5.EDGES[l['edge']])] @ l['normal'] - l['offset'] for l in lines]
                        lnorm = np.array([float(np.sqrt(np.mean(d ** 2))) for d in lnormal])
                        ip = [k for k, d in zip(U, pnorm) if d <= 8.]
                        il = [e for e, d in zip(edges, lnorm) if d <= 8.]
                        try:
                            inlier_geometry, _, _ = self._geometry(model, np.asarray(fit.x, float), ip,
                                [line for line in lines if line['edge'] in il])
                            inlier_geometry_unavailable = None
                        except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
                            inlier_geometry = None
                            inlier_geometry_unavailable = dict(type=type(error).__name__, message=str(error))
                        camera_depth = ((Rcf @ model.T).T + np.asarray(fit.x[3:]))[:, 2]
                        witness = dict(candidate_index=len(result['all_candidate_solutions']), start_index=entry['start_index'],
                            source=source, dimension_index=dim, selected_hypothesis=self.bank.names[dim],
                            R_cf=Rcf.tolist(), R_physical=(Rcf @ Q).tolist(), centroid=np.asarray(fit.x[3:]).tolist(),
                            cf_extents=self.bank.dims[dim].tolist(), final_z=np.asarray(fit.x).tolist(),
                            projected=projected.tolist(), model_camera_depths_m=camera_depth.tolist(),
                            fit_input_ids=list(U), fit_line_edges=list(edges), factor_pool_binding=_copy(result['factor_pool_binding']),
                            scalar_residuals_px=rr.tolist(), residuals_used_px=pnorm.tolist(),
                            line_residual_RMS_px=lnorm.tolist(), line_endpoint_signed_residuals_px=[d.tolist() for d in lnormal],
                            final_inliers=ip, final_line_inliers=il, inlier_count=len(ip),
                            unique_line_inlier_count=len(il), total_inlier_factor_count=len(ip) + len(il),
                            soft_l1_cost=cost, reported_soft_l1_cost=float(fit.cost),
                            sse_px2=float(rr @ rr),
                            factor_sse_px2=float(np.square(np.r_[pnorm, lnorm]).sum()),
                            factor_truncated_sse_px2=float(np.minimum(np.square(np.r_[pnorm, lnorm]), 64.).sum()),
                            point_line_geometry=geometry, local_only=True, global_uniqueness_proven=False,
                            diagnostic_inlier_geometry=inlier_geometry,
                            diagnostic_inlier_geometry_unavailable=inlier_geometry_unavailable,
                            diagnostic_inlier_scalar_residuals=2 * (len(ip) + len(il)),
                            diagnostic_inlier_support_rank6=bool(inlier_geometry is not None and
                                inlier_geometry['observed_normal_joint']['numerical_rank'] == 6 and
                                inlier_geometry['modeled_normal_joint']['numerical_rank'] == 6),
                            inlier_geometry_is_acceptance_gate=False,
                            optimizer_success=True, accepted=False, initial_pose_residual_prior=False)
                        rank_ok = (geometry['observed_normal_joint']['numerical_rank'] == 6 and
                                   geometry['modeled_normal_joint']['numerical_rank'] == 6)
                        witness['accepted'] = rank_ok
                        witness['rejected'] = None if rank_ok else 'FINAL_ACTUAL_FACTOR_RANK_BELOW_SIX'
                        result['all_candidate_solutions'].append(witness)
                        entry.update(candidate_index=witness['candidate_index'], accepted=rank_ok,
                                     rejected=witness['rejected'], final_geometry=geometry)
                        if rank_ok:
                            candidates.append((Candidate(dim, U, witness['candidate_index'], 'LOCAL_POINT_LINE',
                                np.asarray(fit.x[:3]).reshape(3, 1), np.asarray(fit.x[3:]).reshape(3, 1), projected), witness))
                        else:
                            failures.append(witness['rejected'])
                except (ValueError, TypeError, np.linalg.LinAlgError, cv2.error) as error:
                    entry.update(rejected='LOCAL_NUMERICAL_FAILURE', error=type(error).__name__, message=str(error))
                    failures.append(entry['rejected'])
                result['optimizer_attempts'].append(entry)
                cache[key] = (entry['start_index'], _copy(entry))
        result.update(candidate_count=len(result['all_candidate_solutions']),
                      viable_candidate_count=len(candidates),
                      local_start_cache=dict(unique_numeric_starts=len(cache), logical_start_entries=len(result['optimizer_attempts']),
                                            reuse_count=self.counts['local_start_cache_hits'] - before['local_start_cache_hits']))
        if not candidates:
            rank_only = failures and all('RANK_BELOW_SIX' in reason for reason in failures)
            result.update(state='NUMERICAL_RANK_DEFICIENT' if rank_only else 'LOCAL_NUMERICAL_FAILURE',
                          reason='no_viable_local_actual_factor_solution')
            return finish()
        ranked = sorted(candidates, key=lambda cw: (cw[1]['soft_l1_cost'], cw[0].dim, cw[1]['start_index']))
        chosen, best = ranked[0]
        result['best_candidate'] = _copy(best)
        for dim in range(len(self.bank.dims)):
            group = [w for c, w in ranked if c.dim == dim]
            if group:
                result['per_dimension_best'].append(_copy(group[0]))
        for other, witness in ranked[1:]:
            equivalent = self.bank._equivalent(chosen, other)
            delta = witness['soft_l1_cost'] - best['soft_l1_cost']
            tied = abs(delta) <= POLICY['numerical_score_tie_px2']
            alternate = _copy(witness)
            alternate.update(numerically_equivalent_to_selected=equivalent,
                             numerical_objective_tie=tied, objective_difference_from_selected_px2=float(delta))
            result['alternatives'].append(alternate)
            result['numerical_duplicate_solution_count'] += int(equivalent)
            result['numerical_tied_solution_count'] += int(tied)
            if not equivalent:
                result['multiple_solutions'] = True
                result['unresolved_ambiguity'] |= tied
        if result['unresolved_ambiguity']:
            result.update(state='AMBIGUOUS_LOCAL_POINT_LINE', reason='distinct_physical_local_solutions_with_equal_whole_pool_soft_l1_cost')
            return finish()
        output = self.points.copy()
        if H:
            output[list(H)] = chosen.projected[list(H)]
            self.counts['hidden_reprojection_assignments'] += len(H)
        if not set(U).isdisjoint(H) or not np.array_equal(output[8], self.points[8], equal_nan=True):
            raise AssertionError('Hidden initial coordinate entered fit or detector center changed')
        result.update(available=True, pose_available=True, new_pose_estimated=True, no_pose=False,
            state='NEW_POSE', reason='new_local_point_line_pose', output_status='NEW_POSE',
            local_refinement_estimated=True,
            R_cf=best['R_cf'], R_physical=best['R_physical'], centroid=best['centroid'],
            cf_extents=best['cf_extents'], dimension_index=chosen.dim, selected_hypothesis=best['selected_hypothesis'],
            projected=best['projected'], points_final=output.tolist(), hidden_reprojected=bool(H),
            fit_input_ids=list(U), input_inliers=list(U), final_inliers=best['final_inliers'], inliers=best['final_inliers'],
            fit_line_edges=list(edges), final_line_inliers=best['final_line_inliers'],
            point_line_geometry=best['point_line_geometry'], soft_l1_cost=best['soft_l1_cost'],
            diagnostic_inlier_geometry=best['diagnostic_inlier_geometry'],
            diagnostic_inlier_geometry_unavailable=best['diagnostic_inlier_geometry_unavailable'],
            diagnostic_inlier_scalar_residuals=best['diagnostic_inlier_scalar_residuals'],
            diagnostic_inlier_support_rank6=best['diagnostic_inlier_support_rank6'],
            inlier_geometry_is_acceptance_gate=False,
            sse_px2=best['sse_px2'], factor_sse_px2=best['factor_sse_px2'],
            truncated_sse_px2=best['factor_truncated_sse_px2'], residuals_used_px=best['residuals_used_px'],
            line_residual_RMS_px=best['line_residual_RMS_px'],
            point_inlier_count=len(best['final_inliers']), unique_line_inlier_count=len(best['final_line_inliers']),
            total_inlier_factor_count=best['total_inlier_factor_count'], generator='LOCAL_POINT_LINE', generator_ids=[],
            fit_has_fewer_than_four_actual_points=len(U) < 4,
            accepted_rank_scope='whole actual point+unused line pool;8px inliers diagnostic only')
        return finish()
