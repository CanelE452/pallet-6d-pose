"""Pure, opt-in preservation of archived nonfallback prepartial B/I points.

Current source24 authentication/support/topology/overlap policy is reused.
An existing RGB/unknown-source B/I point that was previously made partial is
restored as its exact original complete 2D likelihood. Its own old normal,
archived tangent and edge-activation ownership are removed from COST only.
All original partial interval/geometry/canvas/O/T/U/depth guards remain.

Only exact VISIBLE_BACKBONE_FALLBACK keeps the old partial/revocation policy.
A separately fitted CAL observation-family model is mandatory. No image,
file, target, reference, inference, point search or parameter fit is used.
"""
from __future__ import annotations

from collections import Counter
from copy import copy, deepcopy
import types

import numpy as np

from . import source_preserving_complete_bi_evidence_v4_24 as SOURCE


NOISE_OBSERVATION_FAMILY = 'PREPARTIAL_RGB_POINT_PRESERVING_SUPPORT_BALANCED_NMS17C_MIXTURES_V26'
SOURCE_POLICY = 'RESTORE_NONFALLBACK_PREPARTIAL_BI_2D_EXACT_FALLBACK_ONLY_REVOKES_V26'
PROTOTYPE_OVERLAP_POLICY = SOURCE.PROTOTYPE_OVERLAP_POLICY
FALLBACK_SOURCE = SOURCE.FALLBACK_SOURCE
OLD, N, M, OBS, BROAD = SOURCE, SOURCE.N, SOURCE.M, SOURCE.OBS, SOURCE.BROAD


class PrepartialRGBPointPreservingNativeProblem:
    """Objective policy proxy with an unchanged full-boundary guard delegate.

    The cost view is built at EACH objective call from current self.complete
    and self.remaining_normals: the frozen 20d constructor modifies these on
    a shallow copy. A frozen view would silently undo its point revocations.
    """
    def __init__(self, base, archives, restored):
        self.__dict__.update(base.__dict__)
        self.guard_base = base
        self.restored_prepartial_complete_ids = tuple(sorted(restored))
        self.restored_prepartial_source_archives = {
            str(k): deepcopy(archives[k]) for k in self.restored_prepartial_complete_ids}
        self.prepartial_bound_source_archives = {str(k):deepcopy(v) for k,v in archives.items()}
        self.prepartial_bound_source_archives_sha256 = SOURCE._digest(self.prepartial_bound_source_archives)
        partial = {b['native_corner'] for b in base.boundary}
        self.complete = [m for m in base.measurements
            if m['native_corner'] not in partial or m['native_corner'] in restored]
        self.cost_boundary = [b for b in base.boundary if b['native_corner'] not in restored]
        self.remaining_normals = [b for b in base.remaining_normals if b['native_corner'] not in restored]
        self.prepartial_preservation_audit = dict(schema='PREPARTIAL_RGB_POINT_COST_POLICY_V26',
            source_policy=SOURCE_POLICY, restored_prepartial_complete_ids=list(self.restored_prepartial_complete_ids),
            restored_source_counts=dict(Counter(archives[k]['original_measurement_source'] for k in restored)),
            original_full_partial_guard_ids=[b['native_corner'] for b in base.boundary],
            partial_cost_ids=[b['native_corner'] for b in self.cost_boundary],
            source_q_observedSigma_fitSigma_reliability_exact=True,
            unknown_nonempty_source_retains_complete_q=True,
            old_normal_and_archived_tangent_removed_for_restored_owners=True,
            original_boundary_for_guards_unchanged=True, new_local_queries=0,
            source_policy_is_not_new_measurement_or_total_error_calibration=True)

    def objective(self,p):
        view=copy(self)
        view.boundary=self.cost_boundary
        # No new guard or observation is constructed. The frozen implementation
        # reads this call's self.complete and self.remaining_normals exactly.
        value,gradient,parts=N.NativeEdgeProblem.objective(view,p)
        parts.update(prepartial_RGB_point_preservation_audit=self.prepartial_preservation_audit,
            restored_prepartial_complete_ids=list(self.restored_prepartial_complete_ids),
            restored_full_BI_q_is_archived_original_not_newly_inferred=True,
            hard_guard_boundary_count=len(self.boundary),cost_boundary_count=len(self.cost_boundary),
            original_full_partial_guards_preserved=True)
        return value,gradient,parts

    def constraints(self,p,sides):
        return self.guard_base.constraints(p,sides)

    def evaluate(self,R,t):
        checked=self.guard_base.evaluate(R,t)
        if checked.get('objective') is None:return checked
        try:
            p=N.V._pose_vector(np.asarray(R),np.asarray(t),self.scale)
            value,_,parts=self.objective(p)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError) as error:
            return dict(accepted=False,objective=None,native_costs=[],reason=str(error))
        return {**checked,**parts,'objective':value,
            'accepted':bool(checked['accepted'] and np.isfinite(value)),
            'guard_delegate_is_original_full_boundary':True}


def prepare_prepartial_native_problem(base, *, visible_measurements, initial_self_hidden):
    """Validate every original source/q/noise archive, then expose cost policy."""
    if isinstance(base,PrepartialRGBPointPreservingNativeProblem):
        raise ValueError('Apply the prepartial cost policy exactly once')
    if base.edges or base.replaced or hasattr(base,'groups'):
        raise ValueError('Unmodified old-normal native base required')
    hidden=np.asarray(initial_self_hidden,bool)
    if hidden.shape!=(8,) or list(hidden)!=[s=='S' for s in base.states]:
        raise ValueError('Exact permanent initial S gate required')
    for m in base.measurements:
        k=BROAD._id(m['native_corner'])
        if hidden[k] or base.states[k] not in ('B','I'):
            raise ValueError('Authoritative BI/S gate must precede measurement fields')
    records=[]
    for record in visible_measurements:
        k=BROAD._id(record['native_corner'])
        if hidden[k] or base.states[k] not in ('B','I'):continue
        records.append(record)
    archives=BROAD._archive_measurements(base,records,hidden)
    partial=set()
    for b in base.boundary:
        k=BROAD._id(b['native_corner'])
        if hidden[k] or base.states[k] not in ('B','I'):
            raise ValueError('Only authoritative BI partial records allowed before line fields')
        if k not in archives or k in partial:raise ValueError('Unique bound partial BI archive required')
        partial.add(k)
    restored={k for k in partial if archives[k]['original_measurement_source']!=FALLBACK_SOURCE}
    return PrepartialRGBPointPreservingNativeProblem(base,archives,restored)


def prepare_prepartial_RGB_evidence(base,visible_measurements,*,initial_self_hidden=None):
    """Compatibility alias for the CAL preparation actor."""
    if initial_self_hidden is None:initial_self_hidden=[s=='S' for s in base.states]
    return prepare_prepartial_native_problem(base,visible_measurements=visible_measurements,
        initial_self_hidden=initial_self_hidden)


_AUTH=types.FunctionType(SOURCE._AUTH.__code__,
    {**SOURCE._AUTH.__globals__,'NOISE_OBSERVATION_FAMILY':NOISE_OBSERVATION_FAMILY},'_AUTH_prepartial')
_AUTHENTICATE=types.FunctionType(SOURCE._AUTHENTICATE.__code__,
    {**SOURCE._AUTHENTICATE.__globals__,'_AUTH':_AUTH,
     'NOISE_OBSERVATION_FAMILY':NOISE_OBSERVATION_FAMILY},'_AUTHENTICATE_prepartial')
_SOURCE_AUTH=types.FunctionType(SOURCE.authenticate_complete_bi_image_chain_groups.__code__,
    {**SOURCE.authenticate_complete_bi_image_chain_groups.__globals__,'_AUTHENTICATE':_AUTHENTICATE},'_SOURCE_AUTH_prepartial')


def _require_proxy(base):
    if not isinstance(base,PrepartialRGBPointPreservingNativeProblem):
        raise ValueError('Prepare the explicit prepartial native cost policy first')


def authenticate_complete_bi_image_chain_groups(base,observation,*,junction_proposals,
        visible_measurements,initial_self_hidden,image_binding,initial_projection_xy,
        normal_variance_component_px2,angular_variance_component_sine2,
        noise_provenance,normal_angle_correlation=0.,calibration_preparation=False):
    _require_proxy(base)
    # The proxy retains its original full boundary here, including restored
    # owners. All old references are authenticated before any cost filtering.
    bundle=_SOURCE_AUTH(base,observation,junction_proposals=junction_proposals,
        visible_measurements=visible_measurements,initial_self_hidden=initial_self_hidden,
        image_binding=image_binding,initial_projection_xy=initial_projection_xy,
        normal_variance_component_px2=normal_variance_component_px2,
        angular_variance_component_sine2=angular_variance_component_sine2,
        noise_provenance=noise_provenance,normal_angle_correlation=normal_angle_correlation,
        calibration_preparation=calibration_preparation)
    if SOURCE._digest(SOURCE._bound_archives(base,bundle))!=base.prepartial_bound_source_archives_sha256:
        raise ValueError('Current source/q/noise changed after prepartial preparation')
    bundle.update(schema='PREPARTIAL_RGB_PRESERVING_TYPED_AUTHENTICATION_V26',
        source_policy=SOURCE_POLICY,prepartial_preservation_audit=deepcopy(base.prepartial_preservation_audit),
        restored_prepartial_complete_ids=list(base.restored_prepartial_complete_ids),
        original_full_boundary_normal_refs_authenticated_before_cost_filter=True,
        separately_fitted_new_population_CAL_required=True)
    return bundle


def plan_complete_bi_replacement(base,bundle,*,overlap_policy):
    _require_proxy(base)
    if bundle.get('source_policy')!=SOURCE_POLICY:
        raise ValueError('Exact prepartial preservation generation required')
    archives=SOURCE._bound_archives(base,bundle)
    if (SOURCE._digest(archives)!=base.prepartial_bound_source_archives_sha256
            or bundle.get('restored_prepartial_complete_ids')!=list(base.restored_prepartial_complete_ids)):
        raise ValueError('Restored source/q/noise archive or ID binding changed')
    original=SOURCE.plan_complete_bi_replacement(base,{**bundle,'source_policy':SOURCE.SOURCE_POLICY},overlap_policy=overlap_policy)
    restored=set(base.restored_prepartial_complete_ids)
    active=[];removed=[];owner_removals=[]
    for group in original['active_groups']:
        owners=[]
        for owner in group['activation_owners']:
            k=owner['native_corner']
            if k in restored:
                owner_removals.append(dict(physical_native_edge=group['physical_native_edge'],owner=deepcopy(owner),
                    reason='restored_complete_RGB_or_unknown_q_removes_own_partial_edge_cost'))
            else:owners.append(deepcopy(owner))
        if not owners:
            removed.append(group['physical_native_edge']);continue
        refs=[deepcopy(r) for r in group['replaces_normal_factor_refs'] if r['native_corner'] not in restored]
        active.append({**deepcopy(group),'activation_owners':owners,
            'guard_authorized_original_normal_factor_refs':deepcopy(group['replaces_normal_factor_refs']),
            'replaces_normal_factor_refs':refs})
    refs=[r for group in active for r in group['replaces_normal_factor_refs']]
    if len(refs)!=len({(r['native_corner'],r['line_index']) for r in refs}):
        raise ValueError('Surviving old normal costs replaced once')
    if any(r['native_corner'] in restored for r in refs):raise ValueError('Restored owner normal cost must be removed')
    pairs={tuple(g['physical_native_edge']) for g in active}
    for record in original['corner_plans']:
        if record['native_corner'] in restored and (record['point_cost_revoked'] or 'tangent_kind' in record):
            raise ValueError('A restored complete nonfallback point cannot be revoked or receive tangent cost')
        if any(tuple(pair) not in pairs for pair in record['active_physical_edges']):
            raise ValueError('Surviving complete owner lost an active physical edge')
    ledger={**original['ledger'],
        'original_complete_q_factors_once':len(base.complete)-len(original['revoked_complete_ids']),
        'unique_physical_edge_factors_once':len(active),
        'restored_prepartial_complete_q_factors_once':len(restored),
        'restored_owner_old_normal_costs':0,'restored_owner_old_archived_tangent_costs':0,
        'restored_RGB_q_and_other_owner_edge_may_remain_correlated':True,
        'hard_guard_partial_record_count':len(base.boundary),
        'partial_cost_record_count':len(base.cost_boundary)}
    return {**original,'schema':'PREPARTIAL_RGB_PRESERVING_COMPLETE_BI_REPLACEMENT_V26',
        'source_policy':SOURCE_POLICY,'active_groups':active,'replaces_normal_factor_refs':refs,'ledger':ledger,
        'restored_prepartial_complete_ids':list(base.restored_prepartial_complete_ids),
        'restored_prepartial_source_archives':deepcopy(base.restored_prepartial_source_archives),
        'prepartial_preservation_audit':deepcopy(base.prepartial_preservation_audit),
        'prepartial_owner_removals':owner_removals,'prepartial_ownerless_withheld_physical_edges':removed,
        'full_original_boundary_used_for_authentication_and_hard_guards':True,
        'current_source24_finite_topology_overlap_policy_unchanged':True,
        'new_observation_population_requires_separate_CAL_model':True}


class CompleteBIImageChainProblem(SOURCE.CompleteBIImageChainProblem):
    __init__=types.FunctionType(SOURCE.CompleteBIImageChainProblem.__init__.__code__,
        {**SOURCE.CompleteBIImageChainProblem.__init__.__globals__,
         'plan_complete_bi_replacement':plan_complete_bi_replacement},'__init__')

    def objective(self,p):
        value,gradient,parts=super().objective(p)
        parts.update(prepartial_RGB_point_preservation_generation='V26',
            restored_prepartial_complete_ids=self.plan['restored_prepartial_complete_ids'],
            restored_original_2D_cost_once_old_partial_costs_removed=True,
            replaced_old_normal_factor_count=len(self.plan['replaces_normal_factor_refs']),
            original_full_partial_hard_guards_preserved=True,
            separate_CAL_observation_population_required=True)
        return value,gradient,parts
