"""Pure positive-input native factory and zero-component CAL metadata.

No old task/spec, baseline pose, proposal bank, initialization or external
pose-guide input exists. The caller separately verifies the original input
ancestry and source/byte seals, then supplies only a freshly constructed
clean family plus these named physics/observation/guard inputs. This module
does not establish that ancestry merely from an attestation. It performs no
file/image/reference read, PnP, optimization, label decode or numeric fitting.
"""
from collections import Counter
from copy import deepcopy
import numpy as np

from . import geometry_guided_edge_problem_v4_32 as G
from . import geometry_guided_edge_authentication_v4_32 as AUTH
from . import fresh_observed_BI_pnp_proposals_v4_32 as FRESH

SCHEMA = 'POSITIVE_CLEAN_NATIVE_AND_ZERO_COMPONENT_METADATA_FACTORY_V32'
FAMILY_KEYS = frozenset(('name', 'native_xyz', 'cf_extents', 'physical_registration_Q',
    'coarse_pose', 'coarse_model_center_camera_xyz', 'initializations_imported_from_legacy_task'))
COARSE_KEYS = frozenset(('R', 't', 'physical_registration_Q', 'R_physical',
    'source', 'independent_of_cached_pose_bank'))
BINDING_KEYS = frozenset(('path', 'sha256', 'verified_by_caller',
    'independent_initial_geometry_and_BI_q'))


def _source_binding(binding):
    if (not isinstance(binding, dict) or set(binding) != BINDING_KEYS
            or not isinstance(binding['path'], str) or not binding['path']
            or binding['sha256'] != G.INITIAL_SOURCE_CLOSURE_SHA256
            or binding['verified_by_caller'] is not True
            or binding['independent_initial_geometry_and_BI_q'] is not True):
        raise ValueError('Caller-verified exact independent initial-input closure required')
    return deepcopy(binding)


def _family(family):
    if (not isinstance(family, dict) or set(family) != FAMILY_KEYS
            or family['name'] not in ('ORIGINAL_WD', 'SWAPPED_WD')
            or family['initializations_imported_from_legacy_task'] is not False):
        raise ValueError('Positive clean-family fields only; no legacy bank or initialization')
    coarse = family['coarse_pose']
    if (not isinstance(coarse, dict) or set(coarse) != COARSE_KEYS
            or coarse['source'] != FRESH.COARSE_SOURCE
            or coarse['independent_of_cached_pose_bank'] is not True):
        raise ValueError('Only independent original coarse attestation, no old selected pose')
    xyz = G._array(family['native_xyz'], (8, 3), 'Clean family native XYZ')
    Q = FRESH._rotation(family['physical_registration_Q'], 'Clean family Q')
    R = FRESH._rotation(coarse['R'], 'Clean family coarse R')
    t = G._array(coarse['t'], (3,), 'Clean family coarse native t')
    if (not np.array_equal(np.asarray(coarse['physical_registration_Q'], float), Q)
            or not np.allclose(coarse['R_physical'], R @ Q, atol=1e-12, rtol=0)
            or not np.allclose(family['cf_extents'], np.ptp(xyz, axis=0), atol=1e-12, rtol=0)
            or not np.allclose(family['coarse_model_center_camera_xyz'], R @ xyz.mean(axis=0) + t,
                               atol=1e-12, rtol=0)):
        raise ValueError('Clean family native basis/physical Q/center/extents mismatch')
    return xyz, Q, R, t


def _BI_records(states, hidden, visible_measurements):
    if (len(states) != 8 or any(s not in ('B', 'I', 'O', 'T', 'S', 'U') for s in states)
            or len(hidden) != 8 or any(not isinstance(v, (bool, np.bool_)) for v in hidden)
            or [bool(v) for v in hidden] != [s == 'S' for s in states]):
        raise ValueError('Eight authoritative states and exact original S gates required')
    admitted, seen = [], set()
    for record in visible_measurements:
        k = G._id(record['native_corner'])
        if hidden[k] or states[k] not in ('B', 'I'):
            continue  # No access/copy/hash of excluded q/source/covariance.
        if k in seen:
            raise ValueError('Unique original BI source archive required')
        seen.add(k)
        admitted.append(deepcopy(record))
    return admitted


def native_baseline(*, family, K, states, visible_measurements, domains, image_hw,
                    config, boundary_merged, u_regions, initial_self_hidden,
                    initial_source_binding):
    """Construct the frozen no-new-owner objective with all old guards.

    Nonfallback prepartial q2D is restored exactly as in V26; exact fallback
    partial retains old normal+T4. Boundary/OT/U/canvas/depth guards retain
    the same mathematical constructor. Only the existing pose-prior weight
    is forced to zero. No start generation/selection is performed here.
    """
    binding = _source_binding(initial_source_binding)
    xyz, Q, _, _ = _family(family)
    visible = _BI_records(states, initial_self_hidden, visible_measurements)
    original = G.N.prepare_native_edge_problem(xyz, K, states, visible, domains,
        image_hw, family['coarse_pose'], config={**config, 'prior_weight': 0.},
        boundary_observations=boundary_merged, u_regions=u_regions,
        edge_bundle=dict(edges=[], replaces_normal_factor_refs=[]))
    if not np.allclose(original.Q, Q, atol=1e-12, rtol=0):
        raise ValueError('Native baseline and clean-family physical Q differ')
    result = G.BASELINE.prepare_prepartial_native_problem(original,
        visible_measurements=visible, initial_self_hidden=initial_self_hidden)
    result.clean_positive_input_audit = dict(schema=SCHEMA,
        initial_source_binding=binding, family_name=family['name'],
        source_binding_is_attestation_not_input_byte_verification=True,
        prior_weight=0., old_task_or_pose_bank_or_selected_pose_inputs=0,
        original_BI_q_source_covariance_reliability_preserved=True,
        original_boundary_OT_U_canvas_depth_guards_preserved=True,
        new_image_queries_PnP_start_selection_fits_or_labels=0)
    return result


def _initial_geometry(base, family, geometry):
    if family['name'] != 'ORIGINAL_WD':
        raise ValueError('CAL preparation binds ORIGINAL clean family only')
    _, _, rotation, translation = _family(family)
    center = rotation @ np.asarray(family['native_xyz']).mean(axis=0) + translation
    if (not np.array_equal(np.asarray(geometry['native_xyz']), np.asarray(family['native_xyz']))
            or not np.array_equal(np.asarray(geometry['physical_registration_Q']), np.asarray(family['physical_registration_Q']))
            or not np.allclose(geometry['R_cf'], rotation, atol=1e-12, rtol=0)
            or not np.allclose(geometry['centroid'], center, atol=1e-12, rtol=0)):
        raise ValueError('Exact independent ORIGINAL native geometry/model center required')
    binding = G.bind_initial_geometry(base, geometry)
    # Never copy a supplied S projection slot while producing prepared rows.
    return {**binding['geometry'], 'initial_projection_xy': binding['initial_projection_xy']}


def prepare_one(*, identifier, partition, session, family, K, states,
                visible_measurements, domains, image_hw, config, boundary_merged,
                u_regions, initial_self_hidden, initial_source_binding,
                initial_geometry, image_binding, observation):
    """Create both V32 CAL populations with floors only, before labels.

    Full CAL46 inventory/fixed44 reference alignment, input-byte equality and
    immutable seals are responsibilities of a separate declared operator.
    Empty selected populations remain explicit rows in both policy profiles.
    """
    if partition != 'CAL':
        raise ValueError('CAL-only metadata gate before any coordinate payload')
    if not all(isinstance(value, str) and value for value in (identifier, session)):
        raise ValueError('String CAL identifier/session metadata required')
    base = native_baseline(family=family, K=K, states=states,
        visible_measurements=visible_measurements, domains=domains, image_hw=image_hw,
        config=config, boundary_merged=boundary_merged, u_regions=u_regions,
        initial_self_hidden=initial_self_hidden, initial_source_binding=initial_source_binding)
    geometry = _initial_geometry(base, family, initial_geometry)
    bundle = AUTH.authenticate_geometry_guided_edges(base, observation,
        image_binding=image_binding, initial_geometry=geometry,
        normal_variance_component_px2=0., angular_variance_component_sine2=0.,
        noise_provenance=dict(fit_partition='CAL', observation_family=G.OBSERVATION_FAMILY,
            noise_population=G.POPULATION, selection_profile=G.SELECTION_PROFILE,
            raw_observation_generation=G.RAW_OBSERVATION_GENERATION,
            mode='ZERO_COMPONENT_PREFIT_METADATA_NOT_FITTED'), calibration_preparation=True)
    profiles, metadata = {}, {}
    for policy in G.SOURCE_POLICIES:
        plan = G.plan_geometry_guided_replacement(base, bundle, source_policy=policy)
        groups, tangents = [], []
        for group in plan['active_groups']:
            component = deepcopy(group)
            component['mixture_component_prior'] = 1.
            groups.append(dict(physical_native_edge=group['physical_native_edge'],
                reliability_weight=group['fit_reliability_weight'], components=[component],
                selected_chain_ref=deepcopy(group['selected_chain_ref']),
                activation_owners=deepcopy(group['activation_owners'])))
        for record in plan['corner_plans']:
            if record['mode'] != 'ONE_SELECTED_LINE_ARCHIVED_Q_TANGENT':
                continue
            archive = record['source_archive']
            tangents.append(dict(kind='ARCHIVED_Q_SELECTED_MEASURED_LINE_TANGENT',
                native_corner=record['native_corner'], state=record['state'],
                source=archive['original_measurement_source'], archived_q_xy=archive['original_observed_xy'],
                selected_line_tangent_unit_xy=record['selected_line_tangent_unit_xy'],
                effective_fit_covariance_px2=archive['original_effective_fit_covariance_px2'],
                fit_reliability_weight=archive['original_fit_reliability_weight'],
                selected_line_ref=record['selected_line_ref']))
        profiles[policy] = dict(groups=groups, tangent_records=tangents,
            runtime_plan_digest=G._digest(plan), selection_profile=G.SELECTION_PROFILE)
        metadata[policy] = dict(groups=len(groups), tangents=len(tangents), owners=len(plan['new_owner_ids']),
            modes=dict(Counter(record['mode'] for record in plan['corner_plans'])), ledger=plan['ledger'])
    prepared = dict(id=identifier, partition='CAL', session=session, states=list(states),
        initial_geometry=geometry, observation_generation=G.OBSERVATION_FAMILY,
        raw_observation_generation=AUTH.PRODUCER_GENERATION, noise_population=G.POPULATION,
        selection_profile=G.SELECTION_PROFILE, profiles=profiles)
    return prepared, dict(id=identifier, session=session, profiles=metadata,
        authenticated_groups=len(bundle['groups']), skipped=bundle['skipped'],
        cross_native_overlap_withheld=bundle['cross_native_support_overlap_withheld_component_ids'],
        clean_positive_input_audit=base.clean_positive_input_audit)
