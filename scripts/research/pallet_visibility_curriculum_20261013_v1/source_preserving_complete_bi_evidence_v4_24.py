"""Opt-in source-preserving controller for immutable v20d evidence.

The exact old authentication, finite11/shift12/rank2/topology/overlap tests
run before this provenance gate. Only a complete VISIBLE_BACKBONE_FALLBACK
self-anchor may revoke its 2D likelihood. RGB-derived and unknown complete
sources retain their exact q/covariance/reliability. A physical edge remains
active once if another eligible owner or an existing partial-normal ref
survives. This is a conditional correspondence, not independent evidence.

Pure module: no image/file/GT reads, no fit, no source mutation. A separate
CAL observation-family model is mandatory for this changed active population.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import types

import numpy as np

from . import complete_bi_image_chain_evidence_v4_20d as OLD


NOISE_OBSERVATION_FAMILY = 'SOURCE_PRESERVING_SUPPORT_BALANCED_COMPLETE_PARTIAL_NMS17C_MIXTURES_V24'
SOURCE_POLICY = 'EXACT_VISIBLE_BACKBONE_FALLBACK_SELF_OWNER_ONLY_REVOKES_COMPLETE_2D_V24'
FALLBACK_SOURCE = 'VISIBLE_BACKBONE_FALLBACK'
PROTOTYPE_OVERLAP_POLICY = OLD.PROTOTYPE_OVERLAP_POLICY
N, M, OBS, BROAD = OLD.N, OLD.M, OLD.OBS, OLD.BROAD


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def _bound_archives(base, bundle):
    """Validate source and all q/noise archives without reading hidden entries."""
    hidden = np.asarray([state == 'S' for state in base.states], bool)
    required = {m['native_corner'] for m in base.measurements}
    archives = {}
    for k in sorted(required):
        if hidden[k] or base.states[k] not in ('B', 'I'):
            raise ValueError('Only authoritative nonS BI archives can enter source planning')
        archive = bundle['frozen_measurement_archives'][str(k)]
        record = archive['source_record']
        if (archive['native_corner'] != k or archive['state'] != base.states[k]
                or not isinstance(archive.get('original_measurement_source'), str)
                or archive['original_measurement_source'] != record.get('source')):
            raise ValueError('Original source field must exactly match the frozen source record')
        archives[str(k)] = archive
    reconstructed = BROAD._archive_measurements(base,
        [archives[str(k)]['source_record'] for k in sorted(required)], hidden)
    if any(archives[str(k)] != reconstructed[k] for k in required):
        raise ValueError('Complete source/q/covariance/reliability archive changed')
    return archives


# Only the observation-family provenance changes in the old authenticator;
# all pixel/TLS/support/typed/topology checks and mixture priors are identical.
_AUTH = types.FunctionType(OLD._AUTH.__code__,
    {**OLD._AUTH.__globals__, 'NOISE_OBSERVATION_FAMILY': NOISE_OBSERVATION_FAMILY},
    'authenticate_source_preserving_population')
_AUTHENTICATE = types.FunctionType(OLD.authenticate_complete_bi_image_chain_groups.__code__,
    {**OLD.authenticate_complete_bi_image_chain_groups.__globals__, '_AUTH': _AUTH,
     'NOISE_OBSERVATION_FAMILY': NOISE_OBSERVATION_FAMILY},
    'authenticate_source_preserving_complete_BI_groups')


def authenticate_complete_bi_image_chain_groups(base, observation, *, junction_proposals,
        visible_measurements, initial_self_hidden, image_binding, initial_projection_xy,
        normal_variance_component_px2, angular_variance_component_sine2,
        noise_provenance, normal_angle_correlation=0., calibration_preparation=False):
    bundle = _AUTHENTICATE(base, observation, junction_proposals=junction_proposals,
        visible_measurements=visible_measurements, initial_self_hidden=initial_self_hidden,
        image_binding=image_binding, initial_projection_xy=initial_projection_xy,
        normal_variance_component_px2=normal_variance_component_px2,
        angular_variance_component_sine2=angular_variance_component_sine2,
        noise_provenance=noise_provenance, normal_angle_correlation=normal_angle_correlation,
        calibration_preparation=calibration_preparation)
    archives = _bound_archives(base, bundle)
    bundle.update(schema='SOURCE_PRESERVING_TYPED_FINITE_TOPOLOGY_AUTHENTICATION_V24',
        source_policy=SOURCE_POLICY, bound_source_archives_sha256=_digest(archives),
        original_authentication_and_all_component_priors_unchanged=True)
    return bundle


def plan_complete_bi_replacement(base, bundle, *, overlap_policy):
    """Run the full old plan, then remove only disallowed source owners."""
    if bundle.get('source_policy') != SOURCE_POLICY:
        raise ValueError('Explicit source-preserving authentication required')
    archives = _bound_archives(base, bundle)
    if _digest(archives) != bundle.get('bound_source_archives_sha256'):
        raise ValueError('Authenticated original source archive binding changed')
    original = OLD.plan_complete_bi_replacement(base, bundle, overlap_policy=overlap_policy)
    eligible = {r['native_corner'] for r in original['corner_plans']
        if r['original_measurement_archive']['original_measurement_source'] == FALLBACK_SOURCE}
    plans, rejections, source_counts = [], [], Counter()
    for proposal in original['corner_plans']:
        record = deepcopy(proposal)
        k = record['native_corner']
        source = record['original_measurement_archive']['original_measurement_source']
        admitted = k in eligible
        source_counts[source] += 1
        record['source_preservation_audit'] = dict(original_measurement_source=source,
            exact_fallback_source_eligible=admitted,
            old_finite_typed_plan_would_revoke=proposal['point_cost_revoked'],
            original_q_covariance_reliability_source_exact=True,
            source_policy_selected_without_GT_ID_or_error=True)
        if not admitted:
            record['active_physical_edges'] = []
            record['point_cost_revoked'] = False
            for key in list(record):
                if key.startswith('tangent_'):
                    del record[key]
            record['reason'] = 'original_complete_2D_retained_by_exact_source_policy'
            if proposal['point_cost_revoked']:
                rejections.append(dict(native_corner=k, reason='nonfallback_source_cannot_revoke_complete_2D',
                    original_measurement_source=source,
                    old_finite_typed_edges=proposal['active_physical_edges']))
        plans.append(record)
    active, removed_groups, owner_removals = [], [], []
    for original_group in original['active_groups']:
        owners = []
        for owner in original_group['activation_owners']:
            kind = owner['kind']
            if kind == 'EXISTING_PARTIAL_NORMAL_REF':
                owners.append(deepcopy(owner))
            elif kind == 'ROLE_CORRECT_SELF_COMPLETE_ANCHOR':
                if owner['native_corner'] in eligible:
                    owners.append(deepcopy(owner))
                else:
                    owner_removals.append(dict(physical_native_edge=original_group['physical_native_edge'],
                        owner=deepcopy(owner), reason='source_preservation_owner_removed'))
            else:
                raise ValueError('Unknown activation-owner kind must fail closed')
        if owners:
            active.append({**deepcopy(original_group), 'activation_owners': owners})
        else:
            removed_groups.append(original_group['physical_native_edge'])
    normal_refs = [r for group in active for r in group['replaces_normal_factor_refs']]
    if len(normal_refs) != len({(r['native_corner'], r['line_index']) for r in normal_refs}):
        raise ValueError('Retained physical edges must replace each old normal exactly once')
    active_pairs = {tuple(group['physical_native_edge']) for group in active}
    if len(active_pairs) != len(active):
        raise ValueError('One factor per physical edge required after owner filtering')
    revoked = [r['native_corner'] for r in plans if r['point_cost_revoked']]
    if not set(revoked) <= eligible:
        raise ValueError('Only exact fallback complete measurements may be revoked')
    for record in plans:
        if any(tuple(pair) not in active_pairs for pair in record['active_physical_edges']):
            raise ValueError('A surviving complete owner must retain every admitted edge')
    ledger = {**original['ledger'],
        'original_complete_q_factors_once': len(base.complete) - len(revoked),
        'archived_complete_tangent_factors_once': sum('tangent_kind' in r for r in plans),
        'unique_physical_edge_factors_once': len(active),
        'exact_fallback_complete_sources': len(eligible),
        'nonfallback_complete_q_retained': sum(r['native_corner'] not in eligible for r in plans),
        'complete_sources': dict(source_counts)}
    return {**original, 'schema': 'SOURCE_PRESERVING_COMPLETE_BI_REPLACEMENT_V24',
        'active_groups': active, 'corner_plans': plans, 'revoked_complete_ids': revoked,
        'replaces_normal_factor_refs': normal_refs, 'ledger': ledger,
        'main_replacement_rejections': original['main_replacement_rejections'] + rejections,
        'source_policy': SOURCE_POLICY, 'bound_source_archives_sha256': bundle['bound_source_archives_sha256'],
        'source_owner_removals': owner_removals, 'ownerless_withheld_physical_edges': removed_groups,
        'source_filter_runs_after_unchanged_all_alternative_finite_typed_topology_plan': True,
        'RGB_and_unknown_complete_sources_retain_exact_2D_cost': True,
        'surviving_owner_or_oldpartial_keeps_physical_edge_once': True,
        'all_authentication_inventory_groups_preserved': True,
        'source_preservation_is_not_independent_measurement_calibration': True}


class CompleteBIImageChainProblem(OLD.CompleteBIImageChainProblem):
    __init__ = types.FunctionType(OLD.CompleteBIImageChainProblem.__init__.__code__,
        {**OLD.CompleteBIImageChainProblem.__init__.__globals__,
         'plan_complete_bi_replacement': plan_complete_bi_replacement}, '__init__')

    def objective(self, p):
        value, gradient, audit = super().objective(p)
        audit.update(source_preserving_complete_BI_generation='V24',
            original_RGB_complete_q_likelihood_retained=True,
            exact_fallback_source_only_complete_revocation=True)
        return value, gradient, audit
