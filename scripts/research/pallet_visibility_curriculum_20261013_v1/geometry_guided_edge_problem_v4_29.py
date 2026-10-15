"""Opt-in actual-edge replacement of archived B/I normal information.

The base is exactly the v26 prepartial cost proxy with no projected-edge
factors. No new owner preserves that measurement state. An authenticated new
owner replaces its old complete/partial normal and archived tangent costs;
one selected direction retains only the archived q's fixed measured-line
tangent, while two finite, directed nonparallel lines remove that point cost.
All coordinates, noise, sources and source archives remain immutable.

Selected physical edges are counted once. New edge owners replace only their
old partial locality guards, with finite observed-support coverage and ordered
incident support of the projected physical segment. O/T/U, depth, canvas and
visible geometry retain their frozen primitives. No image or file is read.
"""
from __future__ import annotations

from copy import copy, deepcopy
from itertools import combinations
import hashlib
import json

import numpy as np

from . import prepartial_RGB_point_preservation_v4_26 as BASELINE
from . import projected_native_edge_factor_v4_17 as EDGE

N = BASELINE.N
V, J, P = N.V, N.J, N.P
OBSERVATION_FAMILY = 'FULL_PROVENANCE_MULTIGUIDE_ACTUAL_NMS_ORDERED_NATIVE_EDGES_V29'
POPULATION = 'V29_RUNTIME_ACTIVE_DETERMINISTIC_GRAPH_WINNER_ONE_ACTUAL_LINE_PER_PHYSICAL_EDGE'
SOURCE_POLICIES = ('ALL_BI_REPLACE', 'RGB_NONFALLBACK_PRESERVE')
FALLBACK = 'VISIBLE_BACKBONE_FALLBACK'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _id(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < 8:
        raise ValueError('Native corner integer 0..7 required')
    return int(value)


def _unit(value, name):
    array = np.asarray(value, float)
    if array.shape != (2,) or not np.isfinite(array).all() or not np.isclose(array@array, 1., atol=1e-7, rtol=0):
        raise ValueError(name+' must be a fixed finite unit image direction')
    return array


def _archive(base, k):
    # Caller has already performed the permanent state/S gate.
    return base.prepartial_bound_source_archives[str(k)]


def _finite_actual_pair(first, second, image_hw):
    normals = np.asarray([first['normal_unit_xy'], second['normal_unit_xy']])
    sine = abs(float(np.linalg.det(normals)))
    if sine < .25:
        return dict(passed=False, reason='selected_directions_near_parallel', intersection_sine=sine)
    point = np.linalg.solve(normals, -np.asarray([first['offset_px'], second['offset_px']]))
    gaps = []
    for line in (first, second):
        n = np.asarray(line['normal_unit_xy']);t = np.array([-n[1], n[0]])
        lo, hi = line['actual_support_tangent_interval_px'];s = float(t@point)
        gaps.append(max(lo-s, s-hi, 0.))
    h, w = image_hw
    finite = max(gaps) <= 11. + 1e-9
    canvas = bool(0 <= point[0] < w and 0 <= point[1] < h)
    return dict(passed=bool(finite and canvas), reason='finite_actual_support_pass' if finite and canvas else 'finite_actual_support_or_canvas_failed',
        intersection_sine=sine, diagnostic_intersection_xy=point.tolist(), original_actual_support_gaps_px=gaps,
        support_margin_px=11., old_q_12px_cap_applied=False, precise_corner_created=False)


def plan_geometry_guided_replacement(base, bundle, *, source_policy):
    """Pure source/owner ledger; the bundle must come from v29 authentication."""
    if not isinstance(base, BASELINE.PrepartialRGBPointPreservingNativeProblem):
        raise ValueError('Exact empty-edge v26 prepartial cost baseline required')
    if source_policy not in SOURCE_POLICIES:
        raise ValueError('Explicit v29 source policy required')
    if bundle.get('observation_family') != OBSERVATION_FAMILY or bundle.get('population') != POPULATION:
        raise ValueError('New deterministic selected-edge observation generation required')
    hidden = [s=='S' for s in base.states]
    groups = [];seen = set();owners = {}
    for supplied in bundle['groups']:
        pair = tuple(sorted((_id(supplied['native_corner']), _id(supplied['native_neighbor']))))
        if pair in seen:
            raise ValueError('One selected factor per physical native edge')
        seen.add(pair)
        if hidden[pair[0]] or hidden[pair[1]]:
            raise ValueError('S physical endpoints excluded before edge fields')
        retained = []
        for owner in supplied['activation_owners']:
            k = _id(owner['native_corner'])
            if hidden[k] or base.states[k] not in ('B', 'I'):
                raise ValueError('Only nonS B/I may own image measurements')
            if k not in pair:
                raise ValueError('Only a physical endpoint may own an edge')
            sign=owner.get('fixed_tangent_sign')
            if sign not in (-1,1,None):
                raise ValueError('Fixed authenticated order sign must be +1/-1 or explicitly ambiguous')
            if sign is None and owner.get('order_ambiguous') is not True:
                raise ValueError('Missing order is an explicit model-only ambiguity, never pose-selected')
            if sign is not None and owner.get('order_provenance') not in (
                    'IMMUTABLE_BI_ENDPOINT_HEMISPHERE','SELECTED_GENERATING_GUIDE_UNIQUE_ORDER'):
                raise ValueError('Order sign requires immutable BI or generating-guide provenance')
            source = _archive(base, k)['original_measurement_source']
            if source_policy=='RGB_NONFALLBACK_PRESERVE' and source!=FALLBACK:
                continue
            retained.append(deepcopy(owner))
        if not retained:
            continue
        group = deepcopy(supplied);group['activation_owners'] = retained
        groups.append(group)
        for owner in retained:
            owners.setdefault(owner['native_corner'], []).append(group)
    plans = [];keep_by_owner = {}
    for k, incident in sorted(owners.items()):
        pairs = []
        for first, second in combinations(incident, 2):
            check = _finite_actual_pair(first, second, base.image_hw)
            # This is the selected graph's separately recomputed ordered pair,
            # not a sharp corner pixel or an old-q displacement constraint.
            ordered = [a for a in bundle.get('ordered_corner_pairs', [])
                if a['native_corner']==k and set(a['component_ids'])=={first['component_id'], second['component_id']}]
            directed = bool(ordered and all(a['passed'] for a in ordered))
            pairs.append(dict(first=first['component_id'], second=second['component_id'],
                              directed_order_pass=directed, **check))
        rank2 = any(p['passed'] and p['directed_order_pass'] for p in pairs)
        if rank2:
            valid_pairs=[p for p in pairs if p['passed'] and p['directed_order_pass']]
            best_pair=min(valid_pairs,key=lambda p:(sum(g['image_quality_energy'] for g in incident
                if g['component_id'] in (p['first'],p['second'])),p['first'],p['second']))
            chosen=sorted([g for g in incident if g['component_id'] in (best_pair['first'],best_pair['second'])],
                          key=lambda g:g['component_id'])
            mode = 'TWO_DIRECTION_LINES_NO_Q_COST'
        else:
            # Ambiguous/near-parallel incident support retains a single fixed
            # measured direction; no invented second information dimension.
            chosen = [min(incident, key=lambda g:(g['image_quality_energy'], g['component_id']))]
            mode = 'ONE_SELECTED_LINE_ARCHIVED_Q_TANGENT'
        keep_by_owner[k] = {g['component_id'] for g in chosen}
        archive = deepcopy(_archive(base, k))
        entry = dict(native_corner=k, state=base.states[k], mode=mode,
                     source_archive=archive, selected_component_ids=sorted(keep_by_owner[k]),
                     incident_pair_audit=pairs, new_precise_KP=False)
        if not rank2:
            n = _unit(chosen[0]['normal_unit_xy'], 'selected measured normal')
            entry.update(kind='ARCHIVED_Q_SELECTED_MEASURED_LINE_TANGENT',
                         selected_line_tangent_unit_xy=[-float(n[1]), float(n[0])],
                         selected_line_ref=chosen[0]['component_id'])
        plans.append(entry)
    active = []
    for g in groups:
        retained = [o for o in g['activation_owners'] if g['component_id'] in keep_by_owner[o['native_corner']]]
        if retained:
            active.append({**g, 'activation_owners':retained,
                'fit_reliability_weight':min(float(_archive(base, o['native_corner'])['original_fit_reliability_weight']) for o in retained)})
    owner_ids = sorted(owners)
    return dict(schema='GEOMETRY_GUIDED_EDGE_REPLACEMENT_PLAN_V29', source_policy=source_policy,
        observation_family=OBSERVATION_FAMILY, population=POPULATION, active_groups=active,
        corner_plans=plans, new_owner_ids=owner_ids,
        ledger=dict(physical_edge_factors_once=len(active), archived_selected_line_tangents_once=sum('tangent_unit_xy' in p or 'selected_line_tangent_unit_xy' in p for p in plans),
                    new_precise_2D_KPs=0, old_owner_normal_and_T4_removed_once=True,
                    no_new_owner_preserves_v26_measurement_state=True,
                    RGB_q_and_shared_other_owner_lines_can_remain_correlated=True),
        old_q_source_covariance_reliability_immutable=True)


class GeometryGuidedEdgeProblem:
    def __init__(self, base, bundle, *, source_policy, tangent_gamma):
        self.plan = plan_geometry_guided_replacement(base, bundle, source_policy=source_policy)
        gamma = float(tangent_gamma)
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError('Separately fitted finite nonnegative tangent gamma required')
        self.tangent_gamma = gamma;self.original_base = base;self.base = copy(base)
        owners = set(self.plan['new_owner_ids'])
        self.base.complete = [m for m in base.complete if m['native_corner'] not in owners]
        self.base.cost_boundary = [b for b in base.cost_boundary if b['native_corner'] not in owners]
        self.base.remaining_normals = [b for b in base.remaining_normals if b['native_corner'] not in owners]
        self.guard_boundary = [b for b in base.boundary if b['native_corner'] not in owners]
        self.groups = self.plan['active_groups'];self.tangents = []
        for plan in self.plan['corner_plans']:
            if plan['mode']!='ONE_SELECTED_LINE_ARCHIVED_Q_TANGENT':continue
            archive=plan['source_archive'];axis=_unit(plan['selected_line_tangent_unit_xy'], 'fixed measured-line tangent')
            covariance=np.asarray(archive['original_effective_fit_covariance_px2'],float)
            a=float(axis@covariance@axis)
            if not np.isfinite(a) or a<=0:raise ValueError('Original positive effective covariance required')
            self.tangents.append(dict(native_corner=plan['native_corner'],axis=axis,
                q=np.asarray(archive['original_observed_xy'],float),variance=1.+gamma*a,
                reliability=float(archive['original_fit_reliability_weight']),selected_line_ref=plan['selected_line_ref']))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def objective(self, p):
        value, gradient, parts = self.base.objective(p)
        parts = deepcopy(parts);costs = parts['native_costs'];edge_audits = []
        for g in self.groups:
            extra, derivative, audit = EDGE.projected_edge_value_and_gradient(p,self.xyz,self.K,self.scale,g,
                weight=g['fit_reliability_weight'],huber_scale=self.cfg['huber_scale'])
            value+=extra;gradient+=derivative;edge_audits.append(audit)
            costs.append(dict(kind='PROJECTED_NATIVE_EDGE',native_corner=g['native_corner'],
                native_neighbor=g['native_neighbor'],weighted_robust_cost=extra))
        if self.tangents:
            xy,_,jacobian,_=J.projection_and_jacobian(p,self.xyz,self.K,self.scale)
            delta=self.cfg['huber_scale']
            for t in self.tangents:
                k=t['native_corner'];r=float(t['axis']@(xy[k]-t['q']))/np.sqrt(t['variance'])
                j=t['axis']@jacobian[k]/np.sqrt(t['variance']);den=np.sqrt(1.+r*r/delta**2)
                extra=float(t['reliability']*2.*delta**2*(den-1.))
                value+=extra;gradient+=2.*t['reliability']*r/den*j
                costs.append(dict(kind='ARCHIVED_Q_SELECTED_MEASURED_LINE_TANGENT',native_corner=k,
                    selected_line_ref=t['selected_line_ref'],variance_px2=t['variance'],weighted_robust_cost=extra))
        parts.update(v29_source_policy=self.plan['source_policy'],v29_observation_family=OBSERVATION_FAMILY,
            v29_cost_plan=self.plan,projected_edge_factor_audit=edge_audits,
            old_owner_image_bounds_replaced_by_actual_finite_support=bool(self.plan['new_owner_ids']),
            old_owner_image_bounds_replaced_count=len(self.plan['new_owner_ids']),
            coarse_pose_prior_loss=0.,S_image_terms=0,U_image_terms=0,new_precise_KPs=0,
            graph_selected_one_factor_per_physical_edge=True,
            tangent_axis_is_fixed_selected_measured_line_not_pose_dependent=True)
        return float(value),gradient,parts

    def _actual_support_constraints(self,p):
        if not self.groups:return np.empty(0),np.empty((0,6)),[]
        q,_,jac, _=J.projection_and_jacobian(p,self.xyz,self.K,self.scale)
        values=[];rows=[];audits=[]
        for g in self.groups:
            k,mate=g['native_corner'],g['native_neighbor'];n=np.asarray(g['normal_unit_xy']);axis=np.array([-n[1],n[0]])
            s=np.asarray([axis@q[k],axis@q[mate]]);indices=[k,mate];lo,hi=np.argsort(s)
            actual=np.asarray(g['actual_inlier_xy'],float);coords=actual@axis
            lower,upper=float(coords.min()),float(coords.max());margin=11.
            finite=[lower-s[lo]+margin,s[hi]-upper+margin]
            finite_j=[-axis@jac[indices[lo]],axis@jac[indices[hi]]]
            directed=[]
            for owner in g['activation_owners']:
                anchor=owner['native_corner'];other=mate if anchor==k else k
                sign=owner['fixed_tangent_sign']
                if sign is None:
                    directed.append(dict(native_corner=anchor,order_ambiguous=True,
                        hard_order_guard_applied=False,slack_px=None))
                    continue
                sign=float(sign)
                ordered=np.sort(sign*coords);required=int(np.ceil(.6*len(ordered)-1e-12))
                cut=float(ordered[-required]);slack=cut+1.-sign*float(axis@q[anchor])
                orientation=float(sign*axis@(q[other]-q[anchor]))
                directed.append(dict(native_corner=anchor,slack_px=slack,
                    fixed_tangent_sign=sign,order_provenance=owner['order_provenance'],
                    hard_order_guard_applied=True,projected_endpoint_order_slack_px=orientation,
                    actual_support_fraction=float(np.mean(sign*(coords-float(axis@q[anchor]))>=-1.))))
                values.append(slack);rows.append(-sign*axis@jac[anchor])
                values.append(orientation);rows.append(sign*axis@(jac[other]-jac[anchor]))
            values.extend(finite);rows.extend(finite_j)
            audits.append(dict(component_id=g['component_id'],physical_native_edge=sorted([k,mate]),
                finite_actual_support_slacks_px=list(map(float,finite)),directed_support=directed,
                passed=bool(min([*finite,*[v for x in directed if x['hard_order_guard_applied']
                    for v in (x['slack_px'],x['projected_endpoint_order_slack_px'])]])>=-self.cfg['feasibility_tolerance']),
                margin_px=margin,uses_original_actual_support_not_rounded_exclusion=True))
        return np.asarray(values),np.asarray(rows).reshape(-1,6),audits

    def constraints(self,p,sides):
        values,jacobian=P._joint_constraint_pair(p,self.xyz,self.K,self.measurements,self.domains,sides,
            self.image_hw,self.scale,self.cfg,self.guard_boundary,self.regions)
        extra,derivative,_=self._actual_support_constraints(p)
        return np.r_[values,extra],np.vstack([jacobian,derivative])

    def evaluate(self,R,t):
        R,t=np.asarray(R,float),np.asarray(t,float)
        if R.shape!=(3,3) or t.shape!=(3,) or not V._valid_rotation(R) or not np.isfinite(t).all():
            return dict(accepted=False,objective=None,native_costs=[],reason='invalid saved R/t')
        p=V._pose_vector(R,t,self.scale)
        try:
            value,_,parts=self.objective(p);q,depth=V.project_native(self.xyz,R,t,self.K)
            branches=[(self.constraints(p,sides)[0],sides)for sides in self.sides]
            _,_,support=self._actual_support_constraints(p)
        except(ValueError,np.linalg.LinAlgError,FloatingPointError)as error:
            return dict(accepted=False,objective=None,native_costs=[],reason=str(error))
        finite=[(slack,sides)for slack,sides in branches if np.isfinite(slack).all()]
        if not finite:return dict(accepted=False,objective=value,**parts,reason='nonfinite current hard guards')
        slack,sides=max(finite,key=lambda a:float(np.min(a[0])))
        feasible=bool(np.isfinite(value)and np.isfinite(q).all()and np.min(depth)>=self.cfg['positive_depth_m']-1e-10
            and np.min(slack)>=-self.cfg['feasibility_tolerance'])
        camera=-R.T@t
        visible=P._visible_audit(q,self.measurements,camera,self.image_hw,self.cfg)
        domains=V._domain_audit(q,self.domains,sides,self.image_hw,P._helper_config(self.cfg))
        regions=P.U._u_audit(q,self.regions,self.cfg['feasibility_tolerance'])
        boundary=P._boundary_audit(p,self.xyz,self.K,self.guard_boundary,self.image_hw,self.scale,self.cfg,1.)
        scalar=all(a['visible_support_guards_pass']for a in visible)and all(a['feasible']for a in domains+regions+boundary)and all(a['passed']for a in support)
        if feasible!=bool(scalar and np.isfinite(value)and np.isfinite(q).all()and np.min(depth)>=self.cfg['positive_depth_m']-1e-10):
            raise RuntimeError('V29 scalar and vector hard guards disagree')
        return dict(accepted=feasible,objective=value,**parts,R=R.tolist(),t=t.tolist(),R_physical=(R@self.Q).tolist(),
            projected_native=q.tolist(),minimum_joint_slack=float(np.min(slack)),visible_protection_audit=visible,
            joint_constraint_audit=domains,U_region_constraint_audit=regions,boundary_observation_audit=boundary,
            actual_edge_support_constraint_audit=support,all_current_hard_guards_recomputed=True)
