"""Fresh native-observation objective and unchanged joint guards for v17.

No pose, image, label or file is fetched here.  Authenticated image-line
factors replace their old vertex-normal factors once.  Complete BI data and
the uncertain archived tangent information keep their fixed covariance and
reliability.  The whole-mask cost is deliberately outside this objective.
This helper evaluates saved proposals and also supplies analytic derivatives
for a later saved-start continuous solver.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import numpy as np

from . import statewise_top_normal_outline_solver_v4_16 as PREP
from . import statewise_boundary_normal_solver_v4_4 as N
from . import statewise_archived_BI_tangent_solver_v4_15 as T
from . import projected_native_edge_factor_v4_17 as EDGE

V, J, P = PREP.V, PREP.J, PREP.P


def _ref(value):
    return int(value['native_corner']), int(value['line_index'])


class NativeEdgeProblem:
    def __init__(self, xyz, K, states, visible, domains, image_hw, coarse_pose, *,
                 config, boundary_observations, u_regions, edge_bundle):
        records = T.archive_legacy_B_tangent_predictions(
            boundary_observations, visible, states, config)
        prepared = PREP._prepare(xyz, K, states, visible, domains, image_hw,
            coarse_pose, config, records, u_regions)
        (self.xyz, self.K, self.states, self.measurements, self.domains,
         self.image_hw, self.cfg, self.boundary, self.regions, self.Q,
         self.prior, self.requested_prior_weight) = prepared
        if self.cfg['prior_weight'] != 0:
            raise ValueError('This native-image objective has no coarse-pose prior')
        self.scale = max(float(np.linalg.norm(np.ptp(self.xyz, axis=0))), 1e-3)
        self.edges = []
        for edge in edge_bundle['edges']:
            k, mate = edge['native_corner'], edge['native_neighbor']
            if (isinstance(k, (bool, np.bool_)) or isinstance(mate, (bool, np.bool_))
                    or not isinstance(k, (int, np.integer)) or not isinstance(mate, (int, np.integer))
                    or not 0 <= k < 8 or not 0 <= mate < 8):
                raise ValueError('Valid native edge endpoint IDs required')
            # Do not deepcopy/read noise, provenance or coordinates until S
            # and non-BI endpoint gates have passed, even for malformed input.
            if self.states[k] not in ('B', 'I') or self.states[mate] == 'S':
                raise ValueError('Only BI anchors with a non-initial-S model mate are admitted')
            self.edges.append(deepcopy(edge))
        supplied_refs = [_ref(r) for r in edge_bundle['replaces_normal_factor_refs']]
        if len(supplied_refs) != len(set(supplied_refs)):
            raise ValueError('An old normal factor may be replaced only once')
        self.replaced = set(supplied_refs)
        by_corner = {b['native_corner']: b for b in self.boundary}
        edge_refs = []
        for edge in self.edges:
            k, mate = edge['native_corner'], edge['native_neighbor']
            refs = [_ref(r) for r in edge['replaces_normal_factor_refs']]
            if not refs or len(refs) != len(set(refs)):
                raise ValueError('Every new edge replaces distinct authenticated old normals')
            old_variances, old_weights = [], []
            for corner, index in refs:
                old = by_corner.get(corner)
                if old is None or not 0 <= index < len(old['lines']):
                    raise ValueError('Edge replacement does not identify an existing BI line')
                line = old['lines'][index]
                neighbor = line['provenance'].get('native_neighbor')
                if {corner, neighbor} != {k, mate}:
                    raise ValueError('A new edge may replace only the same physical native edge')
                old_variances.append(line['normal_variance_px2'])
                old_weights.append(old['fit_reliability_weight'])
            if (float(edge['normal_variance_px2']) < max(old_variances) - 1e-10
                    or not np.isclose(edge['fit_reliability_weight'], min(old_weights), atol=1e-12, rtol=0)):
                raise ValueError('Shared-edge noise/reliability cannot be strengthened silently')
            if np.count_nonzero(np.abs(self.xyz[k] - self.xyz[mate]) > 1e-9) != 1:
                raise ValueError('A named native edge must join adjacent cuboid vertices')
            edge_refs.extend(refs)
        if len(edge_refs) != len(set(edge_refs)) or set(edge_refs) != self.replaced:
            raise ValueError('Replacement audit and deduplicated edge factors disagree')
        self.remaining_normals = []
        for b in self.boundary:
            keep = [line for index, line in enumerate(b['lines'])
                    if (b['native_corner'], index) not in self.replaced]
            if keep:
                self.remaining_normals.append({**b, 'lines': keep})
        partial = {b['native_corner'] for b in self.boundary}
        self.complete = [m for m in self.measurements if m['native_corner'] not in partial]
        choices = [(list(d['outside_canvas_side']) if isinstance(d['outside_canvas_side'], tuple)
                    else [d['outside_canvas_side']] if d['outside_canvas_side'] else list(V._SIDES))
                   if d['state'] == 'T' else [None] for d in self.domains]
        if int(np.prod([len(c) for c in choices], dtype=int)) > self.cfg['maximum_union_branches']:
            raise ValueError('Original T-union budget exceeded')
        self.sides = list(product(*choices))

    def objective(self, p):
        """Return native robust value, exact derivative and per-factor costs."""
        delta = self.cfg['huber_scale']; value = 0.; gradient = np.zeros(6); costs = []
        residual, jacobian = J.visible_residual_and_jacobian(
            p, self.xyz, self.K, self.complete, self.scale)
        for m, r, j in zip(self.complete, residual.reshape((-1, 2)), jacobian.reshape((-1, 2, 6))):
            denominator = np.sqrt(1. + float(r @ r) / delta**2)
            weight = m['fit_reliability_weight']
            extra = float(weight * 2. * delta**2 * (denominator - 1.))
            value += extra; gradient += (2. * weight * r / denominator) @ j
            costs.append(dict(kind='COMPLETE_BI', native_corner=m['native_corner'],
                weighted_robust_cost=extra))
        for b in self.remaining_normals:
            extra, derivative = N._normal_objective_pair(p, self.xyz, self.K, [b],
                self.scale, b['fit_reliability_weight'], delta)
            value += extra; gradient += derivative
            costs.append(dict(kind='UNREPLACED_PARTIAL_NORMAL', native_corner=b['native_corner'],
                weighted_robust_cost=extra))
        edge_audits = []
        for edge in self.edges:
            extra, derivative, audit = EDGE.projected_edge_value_and_gradient(
                p, self.xyz, self.K, self.scale, edge,
                weight=edge['fit_reliability_weight'], huber_scale=delta)
            value += extra; gradient += derivative; edge_audits.append(audit)
            costs.append(dict(kind='PROJECTED_NATIVE_EDGE', native_corner=edge['native_corner'],
                native_neighbor=edge['native_neighbor'], weighted_robust_cost=extra))
        tangent_records = T._tangent_records(self.boundary, 4.)
        tangent_residual, tangent_jacobian = T._tangent_residual_and_jacobian(
            p, self.xyz, self.K, self.boundary, self.scale, 4.)
        for record, r, j in zip(tangent_records, tangent_residual, tangent_jacobian):
            denominator = np.sqrt(1. + r*r / delta**2); weight = record['reliability']
            extra = float(weight * 2. * delta**2 * (denominator - 1.))
            value += extra; gradient += 2. * weight * r / denominator * j
            costs.append(dict(kind='UNCERTAIN_ARCHIVED_TANGENT', native_corner=record['native_corner'],
                line_index=record['line_index'], weighted_robust_cost=extra))
        return float(value), gradient, dict(native_costs=costs, projected_edge_factor_audit=edge_audits,
            replaced_old_normal_factor_count=len(self.replaced),
            complete_BI_likelihood_count=len(self.complete), coarse_pose_prior_loss=0.,
            global_outline_objective_weight=0., S_image_terms=0, U_image_terms=0,
            old_covariances_and_reliabilities_unchanged=True,
            line_intercept_not_duplicated=True, tangent_variance_inflation=4.)

    def constraints(self, p, sides):
        return P._joint_constraint_pair(p, self.xyz, self.K, self.measurements,
            self.domains, sides, self.image_hw, self.scale, self.cfg,
            self.boundary, self.regions)

    def evaluate(self, R, t):
        """Freshly verify a saved proposal; never trust its old accepted flag."""
        R, t = np.asarray(R, float), np.asarray(t, float)
        if R.shape != (3, 3) or t.shape != (3,) or not V._valid_rotation(R) or not np.isfinite(t).all():
            return dict(accepted=False, objective=None, native_costs=[], reason='invalid saved R/t')
        p = V._pose_vector(R, t, self.scale)
        try:
            value, _, parts = self.objective(p)
            q, depth = V.project_native(self.xyz, R, t, self.K)
            branches = [(self.constraints(p, sides)[0], sides) for sides in self.sides]
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return dict(accepted=False, objective=None, native_costs=[], reason=str(error))
        finite = [(slack, sides) for slack, sides in branches if np.isfinite(slack).all()]
        if not finite:
            return dict(accepted=False, objective=value, **parts, reason='nonfinite original hard guards')
        slack, sides = max(finite, key=lambda x: float(np.min(x[0])))
        feasible = bool(np.isfinite(value) and np.isfinite(q).all()
            and np.min(depth) >= self.cfg['positive_depth_m'] - 1e-10
            and np.min(slack) >= -self.cfg['feasibility_tolerance'])
        camera = -R.T @ t
        visible = P._visible_audit(q, self.measurements, camera, self.image_hw, self.cfg)
        domains = V._domain_audit(q, self.domains, sides, self.image_hw, P._helper_config(self.cfg))
        regions = P.U._u_audit(q, self.regions, self.cfg['feasibility_tolerance'])
        boundary = P._boundary_audit(p, self.xyz, self.K, self.boundary,
            self.image_hw, self.scale, self.cfg, 1.)
        scalar = (all(v['visible_support_guards_pass'] for v in visible)
            and all(v['feasible'] for v in domains + regions + boundary))
        if feasible != bool(scalar and np.isfinite(value) and np.isfinite(q).all()
                and np.min(depth) >= self.cfg['positive_depth_m'] - 1e-10):
            raise RuntimeError('Vector and independent scalar current guards disagree')
        return dict(accepted=feasible, objective=value, **parts,
            R=R.tolist(), t=t.tolist(), R_physical=(R @ self.Q).tolist(),
            projected_native=q.tolist(), minimum_joint_slack=float(np.min(slack)),
            visible_protection_audit=visible, joint_constraint_audit=domains,
            U_region_constraint_audit=regions, boundary_observation_audit=boundary,
            all_original_hard_guards_recomputed=True)


def prepare_native_edge_problem(xyz, K, states, visible, domains, image_hw, coarse_pose, *,
                               config, boundary_observations, u_regions, edge_bundle):
    return NativeEdgeProblem(xyz, K, states, visible, domains, image_hw, coarse_pose,
        config=config, boundary_observations=boundary_observations,
        u_regions=u_regions, edge_bundle=edge_bundle)
