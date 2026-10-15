"""Selected-pose V32 audit from positive packets and a separate e archive.

No original task, raw guide bank, reference, RGB, pose generator or optimizer
is accepted.  Selected native costs use the reviewed positive replay helper
and independent V29 scalar equations.  O/S/C use their frozen pure kernels.
Stored contender totals and start hashes are checked; unseen complete seed
banks/start rotations are not regenerated or certified by this audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
from itertools import zip_longest
from pathlib import Path

import numpy as np

from . import common as C
from . import audit_geometry_guided_edges_v4_29 as INDEPENDENT
from . import bounded_o_localization_prior_v4_21 as O
from . import bounded_coarse_visibility_prior_v4_20 as S
from . import global_foreground_containment_v4_25 as CONTAINMENT
from . import statewise_partial_BI_outline_solver_v4_13 as CONFIG


B = C.ROOT / '_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
OUTPUT = B / 'clean_fresh_BI_pose_eval_v4_32_r2'
MODEL = B / 'CAL_geometry_guided_edge_noise_v4_32'
CENTRAL = B / 'initial_only_observed_ray_method_plan_v4_32/PREDECLARATION.json'
HELPER = B / 'clean_native_runtime_equation_audit_v4_32/independent_positive_native_replay_v1.py'
HELPER_SHA256 = '6d181262ee5537906e8bd31854c926fd0b7d66e4053cd855f7cb93e3c0f2c90f'
spec = importlib.util.spec_from_file_location('positive_selected_native_replay32', HELPER)
REPLAY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REPLAY)
POLICIES = ('ALL_BI_REPLACE', 'RGB_NONFALLBACK_PRESERVE')
FAMILIES = ('ORIGINAL_WD', 'SWAPPED_WD')
NEW_ARMS = ('CLEAN_FRESH_BI_ALL_BI_NATIVE', 'CLEAN_FRESH_BI_ALL_BI_CONTAINMENT',
            'CLEAN_FRESH_BI_SOURCE_NATIVE', 'CLEAN_FRESH_BI_SOURCE_CONTAINMENT')
ARMS = ('SAME_FROZEN_E', *NEW_ARMS)
POLICY_ARMS = dict(zip(POLICIES, (NEW_ARMS[:2], NEW_ARMS[2:])))
PARTITIONS = dict(CAL=46, TEST=49, REMAINDER=143)
RAW_GENERATION = 'FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29'
COARSE_SOURCE = 'INDEPENDENT_INITIAL_COARSE_PREDICTOR'
GEOMETRY_FIELDS = ('native_xyz', 'R_cf', 'centroid', 'physical_registration_Q',
                   'K', 'image_hw', 'initial_self_hidden')
START_CONFIG = dict(maximum_native_ranked_starts=6,
    minimum_rotation_separation_deg=2., minimum_translation_separation_ratio=.01)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def digest(value):
    return INDEPENDENT._digest(C.finite(value))


def near(first, second):
    return bool(np.isclose(first, second, atol=1e-8, rtol=1e-8))


def _tags(value, *, model=False):
    return (value.get('observation_generation' if model else 'observation_family') == REPLAY.FAMILY
        and value.get('noise_population', value.get('population')) == REPLAY.POPULATION
        and value.get('selection_profile') == REPLAY.PROFILE
        and value.get('raw_observation_generation') == RAW_GENERATION)


def original_BI_archive(source):
    """Same named PnP binding archive, gating before coordinate payload."""
    records = []
    for record in source['visible_measurements']:
        k = record['native_corner']
        if source['initial_self_hidden'][k] or source['states'][k] not in ('B', 'I'):
            continue
        item = dict(native_corner=k, state=source['states'][k], xy=deepcopy(record['xy']),
                    source=record['source'])
        for key in ('covariance_px2', 'fit_covariance_px2', 'fit_reliability_weight'):
            if key in record:
                item[key] = deepcopy(record[key])
        records.append(item)
    return sorted(records, key=lambda record: record['native_corner'])


def geometry_families(packet):
    """Reconstruct only the two known physical bases, never a seed or PnP."""
    source = packet['clean_input']
    xyz = np.asarray(source['native_xyz'], float)
    coarse = source['initial_coarse_pose']
    R, t = np.asarray(coarse['R'], float), np.asarray(coarse['t'], float)
    Q = np.asarray(source['physical_registration_Q'], float)
    REPLAY._rotation(R); REPLAY._rotation(Q)
    if xyz.shape != (8, 3) or t.shape != (3,) or not np.isfinite(xyz).all() or not np.isfinite(t).all():
        raise ValueError('Positive eight-vertex geometry and native translation required')
    mean = xyz.mean(axis=0); extents = np.ptp(xyz-mean, axis=0); center = R@mean+t
    if np.any(extents <= 0) or not np.allclose(xyz-mean, C.cuboid(*extents), atol=1e-9, rtol=0):
        raise ValueError('Known ordered cuboid required; no alternate body fit')
    turn = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    result = {}
    for name in FAMILIES:
        if name == 'ORIGINAL_WD':
            body, rotation, translation, registration = xyz.copy(), R.copy(), t.copy(), Q.copy()
        else:
            registration = turn@Q; rotation = R@turn.T
            new_mean = registration@Q.T@mean
            body = C.cuboid(*extents[[2, 1, 0]])+new_mean
            translation = center-rotation@new_mean
        pose = dict(R=rotation.tolist(), t=translation.tolist(),
            physical_registration_Q=registration.tolist(), R_physical=(rotation@registration).tolist(),
            source=COARSE_SOURCE, independent_of_cached_pose_bank=True)
        family = dict(name=name, native_xyz=body.tolist(), physical_registration_Q=registration.tolist(),
                      coarse_pose=pose)
        REPLAY.check_initial_physical_body(packet, family)
        result[name] = family
    return result


def fresh_input_digest(source, family):
    camera = np.asarray(source['K'], float); camera = camera/camera[2, 2]
    return digest(dict(native_xyz=family['native_xyz'], K=camera.tolist(), states=source['states'],
        initial_self_hidden=source['initial_self_hidden'], admitted_BI=original_BI_archive(source),
        initial_coarse_R=family['coarse_pose']['R'], initial_coarse_t=family['coarse_pose']['t'],
        physical_Q=family['physical_registration_Q']))


def check_model(model, policy, *, tangent=False):
    if model.get('model') is None and model.get('status', '').startswith('MODEL_UNAVAILABLE'):
        if model.get('old_model_substituted') is not False:
            raise ValueError('Unavailable new model may not substitute an old model')
        return False
    # Full available files are bare models; the sixteen LOGO entries are
    # {status, model} records from the frozen fitter's actual public schema.
    model = model.get('model', model)
    schema = ('CAL_INITIAL_ONLY_OBSERVED_NATIVE_RAY_TANGENT_SCALE_V32' if tangent
              else 'CAL_INITIAL_ONLY_OBSERVED_NATIVE_RAY_EDGE_NOISE_V32')
    if (not _tags(model, model=True) or model.get('schema') != schema
            or model.get('fit_partition') != 'CAL' or model.get('source_policy') != policy
            or model.get('newly_fitted_selection_population') is not True
            or model.get('raw_observation_generation_unchanged') is not True
            or any(model.get('prior_V'+str(v)+'_fitted_model_relabelled') is not False for v in (29, 30, 31))
            or model.get('model_digest_sha256') != digest({k:v for k,v in model.items() if k != 'model_digest_sha256'})):
        raise ValueError('Exact newly fitted V32 source-policy model required')
    values = ([model['gamma']] if tangent else
              [model['normal_variance_component_px2'], model['angular_variance_component_sine2']])
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError('Finite nonnegative new variance parameters required')
    if tangent and (model['pixel_floor_variance_px2'] != 1. or model['bias_correction_px'] != 0.):
        raise ValueError('Original fixed pixel floor and zero tangent bias required')
    return True


def _plan_binding_checks(packet, plan, policy, family_name, check, models=None):
    source, contract = packet['clean_input'], packet['source_contract']
    bundle = packet['typed_bundles'][policy][family_name]
    check(_tags(plan) and plan['source_policy'] == policy, 'current policy/family/population/profile plan')
    check(bundle['observation_family'] == REPLAY.FAMILY and bundle['population'] == REPLAY.POPULATION
          and bundle['selection_profile'] == REPLAY.PROFILE and bundle['producer_generation'] == RAW_GENERATION,
          'typed positive bundle population')
    check(plan['initial_model_ray_binding'] == bundle['initial_model_ray_binding'], 'same positive initial geometry/ray binding')
    check(plan['association_configuration'] == bundle['association_configuration'], 'same frozen initial-only query configuration')
    check(bundle['image_binding']['sha256'] == packet['identifiers']['image_sha256'], 'same original image metadata')
    for group in bundle['groups']:
        k, mate = group['native_corner'], group['native_neighbor']
        if (any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v < 8 for v in (k,mate))
                or source['states'][k] not in ('B','I') or source['initial_self_hidden'][k]
                or source['states'][mate] == 'S' or source['initial_self_hidden'][mate]):
            raise ValueError('Permanent state/S gate before selected line/query payload')
    available = contract['model_availability'][policy]
    check(isinstance(available, bool) and contract['old_model_substituted'] is False, 'explicit new model availability; no substitution')
    if models is not None:
        check(models[policy]['available'] == available, 'policy availability exact current full model pair')
    if not available:
        check(not plan['active_groups'] and not plan['new_owner_ids'] and not bundle['groups']
              and contract['tangent_parameters'][policy] is None
              and contract['line_noise_provenance'][policy] is None
              and contract['unavailable_policy_tangent_gamma'] == 0., 'unavailable policy uniformly empty/gamma0')
        return
    tangent = contract['tangent_parameters'][policy]; noise = contract['line_noise_provenance'][policy]
    for value in (tangent, noise):
        check(_tags(value) and value['source_policy'] == policy and value['fit_partition'] == 'CAL', 'per-policy new CAL exporter provenance')
    check(tangent['pixel_floor_variance_px2'] == 1. and tangent['gamma'] >= 0., 'separate measured-axis gamma and fixed floor')
    if models is not None:
        line, gamma = models[policy]['line'], models[policy]['tangent']
        check(noise['model_digest_sha256'] == line['model_digest_sha256']
              and tangent['model_digest_sha256'] == gamma['model_digest_sha256']
              and tangent['gamma'] == gamma['gamma'], 'runtime exact separately fitted policy model digests/gamma')
    supplied = {group['component_id']: group for group in bundle['groups']}
    check(len(supplied) == len(bundle['groups']), 'unique authenticated bundle component IDs')
    for group in plan['active_groups']:
        component = supplied.get(group['component_id'])
        check(component is not None, 'active component bound to positive AUTH bundle')
        if component is None:
            continue
        same = lambda value: {k:v for k,v in value.items() if k not in ('activation_owners', 'fit_reliability_weight')}
        check(same(group) == same(component), 'active TLS/support/query/model payload immutable')
        check(all(owner in component['activation_owners'] for owner in group['activation_owners']), 'shared owner subset from authenticated component')
        check(group['noise_provenance'] == noise, 'active line exact policy noise provenance')
        n = np.asarray(group['normal_unit_xy']); actual = np.asarray(group['actual_inlier_xy'])
        tangent_axis = np.array([-n[1], n[0]])
        coordinates = actual@tangent_axis; span = float(np.ptp(coordinates))
        rms = float(np.sqrt(np.mean((actual@n+group['offset_px'])**2)))
        normal_floor = max(1., rms)
        sine_floor = np.sqrt(2.*normal_floor**2/(span**2+2.*normal_floor**2))
        check(near(group['actual_inlier_span_px'], span)
              and np.allclose(group['actual_support_tangent_interval_px'], [coordinates.min(), coordinates.max()], atol=1e-8, rtol=0)
              and np.isclose(group['normal_floor_px'], normal_floor, atol=1e-7, rtol=1e-7)
              and np.isclose(group['angular_sine_floor'], sine_floor, atol=1e-7, rtol=1e-7),
              'exact observed support span and RMS-derived covariance floors')
        if models is not None:
            line = models[policy]['line']
            fn = float(group['normal_floor_px']); fs = float(group['angular_sine_floor'])
            check(near(group['normal_variance_px2'], line['normal_variance_component_px2']+fn*fn)
                  and near(group['angular_variance_rad2'], line['angular_variance_component_sine2']+fs*fs)
                  and group['normal_angle_correlation'] == 0., 'CAL line components plus exact runtime floors once')


def _readout_checks(packet, entry, family, native, plan, check):
    source = packet['clean_input']; readout = entry['readout']
    states, hidden = source['states'], source['initial_self_hidden']
    R, t = np.asarray(entry['R']), np.asarray(entry['t']); xyz = np.asarray(family['native_xyz'])
    hom = (xyz@R.T+t)@np.asarray(source['K']).T; projected = hom[:, :2]/hom[:, 2:3]
    complete = sorted(c['native_corner'] for c in native['native_costs'] if c['kind'] == 'COMPLETE_BI')
    owners = sorted(plan['new_owner_ids']); archive = {r['native_corner']:r for r in source['visible_measurements']}
    check(readout['states'] == states and readout['current_complete_BI_ids'] == complete
          and readout['current_active_owner_ids'] == owners, 'readout original states and current measurement IDs')
    check(readout['u_regions'] == packet['physics']['u_regions'] and readout['U_range_only'] is True,
          'U original approximate regions only')
    check(readout['S_image_raw_2D_or_region_terms'] == 0 and readout['new_full_2D_corner_measurements'] == 0
          and readout['OT_solved'] is False and readout['OT_unique_solution_claimed'] is False,
          'no S image point or O/T unique-point claim')
    check(readout['used_in_accepted_pose'] == entry['accepted']
          and readout['accepted_fresh_or_refined_pose'] == entry['accepted'], 'accepted versus fallback readout flags')
    check(readout['S_model_projection_ids'] == [k for k,h in enumerate(hidden) if h], 'permanent initial S model projection IDs')
    for k, state in enumerate(states):
        point, origin = readout['reported_points'][k], readout['point_sources'][k]
        if hidden[k] or state == 'S':
            check(point is not None and np.allclose(point, projected[k], atol=1e-8, rtol=0)
                  and origin == 'MODEL_PNP_PROJECTION_ONLY_NO_S_IMAGE_TERM', 'S model-only PnP projection')
        elif state in ('U', 'O', 'T'):
            check(point is None and origin == 'APPROXIMATE_REGION_ONLY_NO_POINT_CORRESPONDENCE', 'U/O/T no precise image point')
        elif k in owners:
            check(point is None and origin == 'ACTUAL_NATIVE_EDGE_CONSTRAINTS_NO_PRECISE_2D_KP', 'line owner not fabricated as keypoint')
        elif k in complete:
            check(point == archive[k]['xy'] and origin == archive[k]['source'], 'retained complete original q/source exact')
        else:
            check(point is None and origin == 'ARCHIVED_PARTIAL_NORMAL_AND_UNCERTAIN_TANGENT_NO_PRECISE_KP', 'fallback partial has no precise point')


def _selection_checks(selection, accepted, policy, containment, weight, check):
    check(selection['old_full_mask_cost_or_numeric_tie_break_used'] is False
          and selection['e_numeric_pose_prior_or_fallback'] is False
          and selection['native_Pareto_audit_not_prefilter'] is True, 'stored selector has no e/full-mask/Pareto routing')
    check(near(selection['containment_weight'], weight if containment else 0.), 'same fixed selector containment weight')
    candidates = selection['candidates']; names = [c['name'] for c in candidates]
    check(len(names) == len(set(names)), 'unique stored feasible contender names')
    for candidate in candidates:
        values = [candidate[k] for k in ('native_energy', 'regularizer_energy', 'global_containment_cost',
                                         'weighted_containment_objective', 'total_energy')]
        check(np.isfinite(values).all() and min(values) >= 0., 'finite nonnegative stored feasible contender energies')
        check(near(candidate['weighted_containment_objective'], selection['containment_weight']*candidate['global_containment_cost'])
              and near(candidate['total_energy'], candidate['native_energy']+candidate['regularizer_energy']+candidate['weighted_containment_objective']),
              'stored contender N+O+S+lambdaC once')
        check(candidate['family'] in FAMILIES and candidate['name'].startswith(candidate['family']+'/'), 'stored contender current WD label')
    if not accepted:
        check(not candidates and selection['selected_name'] is None and selection['selected_family'] is None,
              'fallback only when no stored feasible contender')
        return None
    check(bool(candidates), 'accepted selection has feasible contenders')
    if not candidates:
        return None
    minimum = min(c['total_energy'] for c in candidates)
    ties = [c for c in candidates if abs(c['total_energy']-minimum) <=
            32*np.finfo(float).eps*max(1., abs(minimum), abs(c['total_energy']))]
    winner = min(ties, key=lambda c:c['index'])
    check(selection['selected_name'] == winner['name'] and selection['selected_family'] == winner['family'],
          'minimum stored feasible total; stable original index on numeric ties')
    return next((c for c in candidates if c['name'] == selection['selected_name']), None)


def audit_row(packet, comparison, row, *, models=None):
    """Pure one-row check. No IO or regeneration of fresh proposals occurs."""
    before = digest(dict(packet=packet, comparison=comparison, row=row)); checks = []; components = []
    def check(condition, reason):
        checks.append(dict(passed=bool(condition), reason=reason))
    if set(packet) != REPLAY.WORKER_KEYS or packet['schema'] != REPLAY.WORKER_SCHEMA:
        raise ValueError('Exact positive seven-key packet required before numeric payload')
    ids = packet['identifiers']; source = packet['clean_input']; contract = packet['source_contract']
    check(ids == source['identifiers'] == row['identifiers'] and row['id'] == comparison['id'] == ids['id']
          and row['partition'] == comparison['partition'] == ids['partition'], 'exact ordered positive/e/new identity')
    check(row['image_sha256'] == ids['image_sha256'], 'original RGB SHA metadata')
    check(set(row['arms']) == set(ARMS) and digest(row['arms']['SAME_FROZEN_E']) == digest(comparison['prediction']),
          'parent e prediction canonical exact; five arms complete')
    check(comparison['legacy_e_pose_semantics'] == row['legacy_e_pose_semantics']
          and comparison['legacy_e_pose_semantics']['numeric_worker_exposure'] is False
          and row['e_comparison_parent_only_no_numeric_exposure'] is True,
          'parent-only e comparison convention retained')
    check(row['source_contract'] == contract and _tags(contract) and contract['old_model_substituted'] is False,
          'exact positive input contract and new population')
    binding = source['initial_source_binding']
    check(binding == contract['initial_source_binding'] and binding['verified_by_caller'] is True
          and binding['independent_initial_geometry_and_BI_q'] is True and _sha(binding['sha256']),
          'same explicit caller-verified independent original source binding')
    check(row['states'] == source['states'] and row['initial_self_hidden'] == source['initial_self_hidden'], 'permanent state/S gates unchanged')
    check(row['parent_only_e_comparison_not_supplied_to_worker'] is True and row['baseline_inputs'] == 0
          and row['cached_pose_or_external_guide_inputs'] == 0 and row['row_retained'] is True,
          'positive worker source architecture flags')
    if contract['ready'] is False:
        valid_coarse = False
        try:
            family = geometry_families(packet)['ORIGINAL_WD']; pose = source['initial_coarse_pose']
            scale = max(float(np.linalg.norm(np.ptp(family['native_xyz'], axis=0))), 1e-3)
            depth = (np.asarray(family['native_xyz'])@np.asarray(pose['R']).T+pose['t'])[:, 2]
            valid_coarse = bool(pose.get('source') == COARSE_SOURCE
                and pose.get('independent_of_cached_pose_bank') is True and np.all(depth > 1e-9*scale))
            if 'physical_registration_Q' in pose:
                valid_coarse &= bool(np.array_equal(pose['physical_registration_Q'], source['physical_registration_Q']))
            if 'R_physical' in pose:
                valid_coarse &= bool(np.allclose(pose['R_physical'], np.asarray(pose['R'])@np.asarray(source['physical_registration_Q']), atol=1e-10, rtol=0))
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError):
            pass
        check(not valid_coarse, 'no-pose input independently lacks valid positive own coarse geometry')
        check(row['invalid_initial_coarse'] is True and row['fresh_seed_audit'] == {}, 'invalid coarse retained before any other physics')
        for arm in NEW_ARMS:
            entry = row['arms'][arm]
            check(entry['accepted'] is False and entry['pose_available'] is False
                  and entry['actual_pose'] is None and entry['R'] is None and entry['t'] is None
                  and entry['old_method_or_baseline_fallback_used'] is False and entry['row_retained'] is True,
                  'invalid coarse arm retained without method fallback')
            check(entry['readout']['reported_points'] == [None]*8
                  and entry['readout']['continuous_SE3_search_performed'] is False,
                  'no-pose readout has no invented points')
        check(digest(dict(packet=packet, comparison=comparison, row=row)) == before, 'no-pose audit inputs immutable')
        return dict(checks=checks, components=components, no_pose=True, accepted=0, fallback=0)
    check(contract['ready'] is True, 'boolean ready contract')
    families = geometry_families(packet)
    task = REPLAY.positive_task(packet)
    check(row['legacy_evaluator_run_or_refiner_wrapper_used'] is False and row['old_reference_start_problem_used'] is False,
          'no old evaluator or reference-start wrapper')
    for name, family in families.items():
        bank = row['fresh_seed_audit'][name]
        check(bank['input_binding_sha256'] == fresh_input_digest(source, family)
              and bank['old_pose_bank_inputs'] == 0 and isinstance(bank['proposal_count'], int)
              and bank['proposal_count'] >= 1, 'fresh geometric bank original BI input digest; no bank regeneration')
    regularizers = packet['regularizers']; cfg = CONFIG._outline_config(regularizers['outline_configuration'])
    weight = float(cfg['outline_factor_weight']) if regularizers['outline_observation'].get('supported') else 0.
    raw_cache = {name: INDEPENDENT.prepare_raw(task, family) for name,family in families.items()}
    accepted = fallback = 0
    for policy, arms in POLICY_ARMS.items():
        for containment, arm in zip((False, True), arms):
            entry = row['arms'][arm]; is_accepted = entry['accepted']
            check(entry['method'] == arm and entry['source_policy'] == policy and entry['pose_available'] is True,
                  'fixed arm/policy and pose availability')
            name = entry['WD_family']; family = families[name]
            xyz = np.asarray(family['native_xyz']); Q = np.asarray(family['physical_registration_Q'])
            R, t = np.asarray(entry['R']), np.asarray(entry['t']); REPLAY._rotation(R)
            center = R@xyz.mean(axis=0)+t; actual = entry['actual_pose']
            check(np.array_equal(actual['R_cf'], R) and np.array_equal(actual['native_t'], t)
                  and np.allclose(actual['centroid'], center, atol=1e-12, rtol=0)
                  and np.allclose(actual['model_center_camera_xyz'], center, atol=1e-12, rtol=0)
                  and np.array_equal(actual['physical_registration_Q'], Q)
                  and np.allclose(actual['R_physical'], R@Q, atol=1e-12, rtol=0)
                  and actual['WD_family'] == name
                  and np.allclose(actual['cf_extents'], np.ptp(xyz, axis=0), atol=1e-12, rtol=0)
                  and actual['centroid_semantics'] == 'GEOMETRIC_MODEL_CENTER_NOT_NATIVE_ORIGIN_TRANSLATION',
                  'explicit native t, model center, WD and physical RQ')
            check(entry['new_noise_availability'] == contract['model_availability'][policy]
                  and entry['same_policy_native_endpoints_retained'] == containment,
                  'policy noise availability and same-policy endpoint retention flags')
            check(np.array_equal(entry['physical_registration_Q'], Q)
                  and np.allclose(entry['R_physical'], R@Q, atol=1e-12, rtol=0)
                  and np.allclose(entry['centroid'], center, atol=1e-12, rtol=0), 'top-level registered physical pose')
            check(set(entry['new_geometry_edge_cost_plans']) == set(FAMILIES), 'both family plans retained')
            for family_name in FAMILIES:
                _plan_binding_checks(packet, entry['new_geometry_edge_cost_plans'][family_name],
                                     policy, family_name, check, models)
                refinement = entry['family_refinements'][family_name]
                start = refinement['start_selection']; search = refinement['search']
                check(start['configuration'] == START_CONFIG and 1 <= len(start['selected_start_names']) <= 7
                      and search['starts_per_T_branch'] == len(start['selected_start_names']), 'stored native6+coarse start budget')
                check(_sha(refinement['start_packet_sha256'])
                      and search['same_clean_start_packet_sha256'] == refinement['start_packet_sha256']
                      and search['cached_or_baseline_pose_inputs'] == 0
                      and search['legacy_reference_start_problem_used'] is False, 'fresh start hash/search source preservation')
                other = row['arms'][arms[0]]['family_refinements'][family_name]
                check(refinement['start_packet_sha256'] == other['start_packet_sha256']
                      and start == other['start_selection'], 'same-policy native/C exact stored start packet and selection')
            selection = entry['candidate_selection']
            chosen = _selection_checks(selection, is_accepted, policy, containment, weight, check)
            plan = entry['new_geometry_edge_cost_plans'][name]
            if not is_accepted:
                fallback += 1; original = families['ORIGINAL_WD']['coarse_pose']
                check(name == 'ORIGINAL_WD' and np.array_equal(R, original['R']) and np.array_equal(t, original['t'])
                      and entry['selected_endpoint_origin'] == 'OWN_INDEPENDENT_INITIAL_COARSE_FALLBACK', 'failed arm exact own original coarse fallback only')
                check(entry['solution']['objective'] is None and not entry['solution']['native_costs'], 'fallback has no accepted likelihood claim')
            else:
                accepted += 1
            gamma = contract['tangent_parameters'][policy]['gamma'] if contract['model_availability'][policy] else 0.
            raw = raw_cache[name]
            if is_accepted:
                native = INDEPENDENT.recompute_native(task, family, R, t, plan, gamma, raw=raw)
                checks.extend(dict(passed=bool(ok), reason=reason) for ok,reason in native['checks'])
            else:
                # An own-coarse fallback makes no feasible likelihood claim.
                # A collapsed trial edge must not discard that retained row.
                partial = {b['native_corner'] for b in raw.boundary}
                original = {m['native_corner']:m['source'] for m in source['visible_measurements']}
                native = dict(native_costs=[dict(kind='COMPLETE_BI', native_corner=m['native_corner'])
                    for m in raw.measurements if m['native_corner'] not in plan['new_owner_ids']
                    and (m['native_corner'] not in partial or original[m['native_corner']] != INDEPENDENT.FALLBACK)])
            _readout_checks(packet, entry, family, native, plan, check)
            check(entry['fallback_audit'] == dict(own_independent_original_coarse_only=not is_accepted,
                old_method_or_baseline_fallback_used=False, cached_pose_inputs=0), 'fallback provenance flags exact')
            if not is_accepted:
                continue
            check(native['accepted'] is True, 'selected independent original/new support guard acceptance')
            solution = entry['solution']
            check(np.allclose(solution['R'], R, atol=1e-12, rtol=0) and np.allclose(solution['t'], t, atol=1e-12, rtol=0)
                  and solution['accepted'] is True, 'selected solution pose bound to exported result')
            first = INDEPENDENT.cost_buckets(native['native_costs']); saved = INDEPENDENT.cost_buckets(solution['native_costs'])
            check(first.keys() == saved.keys() and all(near(first[k], saved[k]) for k in first), 'independent native factor identities/costs once')
            p = np.r_[REPLAY.Rotation.from_matrix(R).as_rotvec(), t/raw.scale]
            for region in regularizers['o_regions']:
                k = region['native_corner']
                if source['states'][k] != 'O' or source['initial_self_hidden'][k]:
                    raise ValueError('Authoritative nonS O gate before region payload')
            if regularizers['original_S_prepared'] is not None:
                if not all(source['initial_self_hidden'][r['native_corner']] and source['states'][r['native_corner']] == 'S'
                           for r in regularizers['original_S_prepared']['eligible_initial_S']):
                    raise ValueError('Permanent S gate before model prior payload')
            ov,_,oa = O.bounded_o_localization_value_and_gradient(p, raw.xyz, raw.K, raw.scale,
                regularizers['o_regions'], **regularizers['o_parameters'])
            if regularizers['original_S_prepared'] is None:
                sv, sa = 0., dict(S_image_or_2D_terms=0)
            else:
                rebound = S.rebind_bounded_visibility_trial_geometry(regularizers['original_S_prepared'], raw.xyz)
                sv,_,sa = S.coarse_visibility_prior_value_and_gradient(p, raw.xyz, raw.scale, rebound)
            priors = solution['regularizer_costs']
            check([v['kind'] for v in priors] == ['O_COARSE_LOCALIZATION_PRIOR', 'S_COARSE_VISIBILITY_PRIOR']
                  and near(priors[0]['weighted_cost'], ov) and near(priors[1]['weighted_cost'], sv), 'only frozen O/S scalar costs; pose prior0')
            check(oa['O_point_terms'] == 0 and sa['S_image_or_2D_terms'] == 0, 'bounded priors create no O/S image points')
            shape = 0.
            if containment and weight > 0:
                shape,_,ca = CONTAINMENT.outline_value_and_gradient(p, raw.xyz, raw.K, raw.scale,
                    regularizers['outline_observation'], cfg)
                check(ca['valid_projection'] and ca['model_to_observed_factor_weight'] == 0.
                      and ca['inside_model_hull_cost_zero'], 'conditional foreground containment only')
            total = native['objective']+ov+sv+(weight*shape if containment else 0.)
            check(near(solution['native_observation_objective'], native['objective'])
                  and near(solution['regularizer_objective'], ov+sv) and near(solution['objective'], total), 'selected N+O+S+lambdaC sum once')
            if containment:
                check(near(solution['global_containment_cost'], shape)
                      and near(solution['weighted_containment_objective'], weight*shape), 'stored containment scalar/weight once')
            if chosen is not None:
                check(chosen['family'] == name and near(chosen['native_energy'], native['objective'])
                      and near(chosen['regularizer_energy'], ov+sv)
                      and near(chosen['weighted_containment_objective'], weight*shape if containment else 0.)
                      and near(chosen['total_energy'], total), 'selected optimizer and selector identical current components')
            components.append(dict(id=ids['id'], arm=arm, family=name, native=native['objective'],
                O=float(ov), S=float(sv), weighted_containment=float(weight*shape if containment else 0.),
                total=float(total), absolute_total_difference=abs(total-solution['objective']),
                owner_ids=native['owner_ids'], minimum_actual_support_slack=native['minimum_actual_support_slack']))
    check(digest(dict(packet=packet, comparison=comparison, row=row)) == before, 'audit inputs immutable')
    return dict(checks=checks, components=components, no_pose=False, accepted=accepted, fallback=fallback)


def joined_rows(packets, comparisons, predictions, records, indices, ids):
    """Ordered streaming join, including zeros/failures; no worker spool decode."""
    if len(ids) != 238 or len(set(ids)) != 238:
        raise ValueError('All238 unique fixed IDs required')
    counts = Counter(); count = 0
    for packet, comparison, row, record, index in zip_longest(packets, comparisons, predictions, records, indices):
        if any(value is None for value in (packet, comparison, row, record, index)):
            raise ValueError('All238 ordered streams required including no-pose/fallback rows')
        identifiers = packet['identifiers']
        if (count >= 238 or any(value != ids[count] for value in
                (identifiers['id'], comparison['id'], row['id'], record['id'], index['id']))
                or any(value != identifiers['partition'] for value in
                (comparison['partition'], row['partition'], record['partition'], index['partition']))
                or record['positive_packet_sha256'] != digest(packet)
                or record['ready'] != packet['source_contract']['ready'] or index['arm_count'] != 4):
            raise ValueError('Positive/e/prediction canonical identity, partition or packet digest mismatch')
        # The final journal adds only e and its two explicit comparison fields
        # to the sealed four-arm worker record. Rebuild that same JSON value.
        worker = {key:value for key,value in row.items()
                  if key not in ('e_comparison_parent_only_no_numeric_exposure', 'legacy_e_pose_semantics')}
        worker['arms'] = {name:row['arms'][name] for name in NEW_ARMS}
        if digest(worker) != index['canonical_row_sha256']:
            raise ValueError('Merged new result differs from sealed four-arm canonical worker record')
        count += 1; counts[identifiers['partition']] += 1
        yield packet, comparison, row
    if count != 238 or counts != Counter(PARTITIONS):
        raise ValueError('Every CAL46/TEST49/REMAINDER143 row retained; severe0')


def _verify(*maps):
    union = {}
    for mapping in maps:
        for relative, expected in mapping.items():
            if relative in union and union[relative] != expected:
                raise ValueError('Conflicting source/input binding: '+relative)
            union[relative] = expected
            if C.sha(C.ROOT/relative) != expected:
                raise ValueError('Frozen binding changed: '+relative)
    return union


def _load_models(folder):
    seal = C.read(folder/'MODEL_SEAL.json')
    if (seal['fit_partition'] != 'CAL' or seal['CAL_inventory'] != 46 or seal['compatible_inventory'] != 44
            or not _tags(seal, model=True)):
        raise ValueError('Exact new CAL46/compatible44 model seal required')
    _verify(seal['source_sha256'], seal['input_sha256'])
    expected_assets = {'CAL_LOGO_REPORT.json'} | {policy+'/'+name for policy in POLICIES
        for name in ('MODEL.json', 'TANGENT_MODEL.json', 'CAL_ERRORS.jsonl.gz')}
    if set(seal['assets']) != expected_assets:
        raise ValueError('Four full model files, two saved error journals and one LOGO report required')
    for relative, expected in seal['assets'].items():
        if C.sha(folder/relative) != expected:
            raise ValueError('Sealed current model asset changed')
    if C.sha(folder/'PREPARATION_SEAL.json') != seal['preparation_seal_sha256']:
        raise ValueError('New CAL model preparation link changed')
    report = C.read(folder/'CAL_LOGO_REPORT.json'); models = {}
    for policy in POLICIES:
        full = dict(line=C.read(folder/policy/'MODEL.json'), tangent=C.read(folder/policy/'TANGENT_MODEL.json'))
        states = [check_model(full[kind], policy, tangent=kind == 'tangent') for kind in ('line', 'tangent')]
        models[policy] = {**full, 'available':all(states)}
        entry = report[policy]
        if entry['inventory46'] != 46 or entry['compatible44'] != 44 or len(entry['LOGO_folds']) != 4:
            raise ValueError('Four CAL held-session model/status pairs required')
        held = []
        for fold in entry['LOGO_folds']:
            held.append(fold['heldout_session'])
            for kind in ('line', 'tangent'):
                check_model(fold[kind], policy, tangent=kind == 'tangent')
        if len(set(held)) != 4:
            raise ValueError('Four distinct held sessions required')
    return models, seal


def _source_map():
    paths = [Path(__file__), Path(__file__).with_name('test_audit_clean_fresh_BI_poses_v4_32.py'), HELPER]
    paths += [Path(module.__file__) for module in (C, INDEPENDENT, INDEPENDENT.KERNEL, INDEPENDENT.N,
                                                   O, S, CONTAINMENT, CONFIG)]
    return {str(path.resolve().relative_to(C.ROOT)):C.sha(path) for path in paths}


def main(*, root_independent_audit_GO=False, expected_prediction_seal_sha256=None,
         expected_central_plan_sha256=None, expected_model_seal_sha256=None, output=OUTPUT):
    if root_independent_audit_GO is not True:
        raise PermissionError('Separate ROOT all238 independent audit GO before IO')
    if not all(_sha(value) for value in (expected_prediction_seal_sha256,
            expected_central_plan_sha256, expected_model_seal_sha256)):
        raise ValueError('Explicit prediction/central/new-model seal SHA before IO')
    output = Path(output).resolve(); dest = output/'independent_root_audit_v1'
    if dest.exists():
        raise FileExistsError('Preserve every audit attempt')
    if (C.sha(output/'PREDICTION_SEAL.json') != expected_prediction_seal_sha256
            or C.sha(CENTRAL) != expected_central_plan_sha256
            or C.sha(MODEL/'MODEL_SEAL.json') != expected_model_seal_sha256
            or C.sha(HELPER) != HELPER_SHA256):
        raise ValueError('Exact caller-pinned current seals and reviewed helper required')
    protocol = C.read(output/'PROTOCOL.json'); inputs = C.read(output/'INPUT_SEAL.json')
    pred = C.read(output/'PREDICTION_SEAL.json'); ids = protocol['ordered_ids']
    if (len(ids) != 238 or len(set(ids)) != 238 or protocol['arms'] != list(ARMS)
            or protocol['central_plan_sha256'] != expected_central_plan_sha256
            or protocol['new_model_seal_sha256'] != expected_model_seal_sha256
            or inputs['rows'] != 238 or inputs['ordered_ids'] != ids or inputs['arms'] != list(ARMS)
            or inputs['arm_count'] != 5
            or protocol['partitions'] != PARTITIONS
            or protocol['e_comparison_sha256'] != inputs['e_comparison_sha256']
            or protocol['positive_worker_schema'] != REPLAY.WORKER_SCHEMA
            or protocol['worker_keys'] != sorted(REPLAY.WORKER_KEYS)
            or protocol['full_pose_TP_topnormal_prior_weights'] != 0.
            or pred['rows'] != 238 or pred['ordered_ids'] != ids or pred['arms'] != list(ARMS)
            or pred['arm_count'] != 5 or pred['all_declared_arms_complete'] is not True
            or pred['before_fixed_reference_scoring'] is not True or pred['no_population_drop'] is not True):
        raise ValueError('Same canonical all238 positive inputs and complete five-arm predictions required')
    execution_plan = protocol['execution_plan']
    if C.sha(C.ROOT/execution_plan['path']) != execution_plan['sha256']:
        raise ValueError('Exact separately declared positive execution plan required')
    bindings = [('PROTOCOL.json', inputs['protocol_sha256']), ('INPUTS.jsonl.gz', inputs['input_sha256']),
        ('E_COMPARISON.jsonl.gz', inputs['e_comparison_sha256']), ('PROTOCOL.json', pred['protocol_sha256']),
        ('INPUT_SEAL.json', pred['input_seal_sha256']), ('PREDICTIONS.jsonl.gz', pred['predictions_sha256']),
        ('E_COMPARISON.jsonl.gz', pred['separate_e_comparison_sha256']),
        ('PREDICTION_EXECUTION_PROTOCOL.json', pred['execution_protocol_sha256']),
        ('PREDICTION_ROW_INDEX.json', pred['row_index_sha256'])]
    for relative, expected in bindings:
        if C.sha(output/relative) != expected:
            raise ValueError('Sealed positive/prediction link changed: '+relative)
    maps = [protocol[key] for key in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256')]
    _verify(*maps)
    if len(protocol['protected_user_sha256']) != 6 or len(protocol['original_CAL_file_sha256']) != 53:
        raise ValueError('Protected6 and immutable CAL53 maps required')
    execution = C.read(output/'PREDICTION_EXECUTION_PROTOCOL.json')
    if (execution['positive_worker_keys'] != sorted(REPLAY.WORKER_KEYS)
            or execution['e_numeric_worker_inputs'] != 0 or execution['reference_or_score_inputs'] != 0):
        raise ValueError('Positive numerical worker execution boundary required')
    models, _ = _load_models(MODEL)
    sources = _source_map(); dest.mkdir()
    C.write(dest/'PLAN.json', dict(schema='INDEPENDENT_POSITIVE_SELECTED_AUDIT_PLAN_V32',
        UTC=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        central_plan_sha256=expected_central_plan_sha256, model_seal_sha256=expected_model_seal_sha256,
        prediction_seal_sha256=expected_prediction_seal_sha256, rows=238, arms=list(ARMS),
        numeric_packet_prediction_decodes_before_this_plan=0, input_streams=['INPUTS','E_COMPARISON','PREDICTIONS'],
        original_tasks_rawguidebanks_RGB_reference_PnP_optimizer_inputs=0,
        scope='chosen native+O/S/C arithmetic/guards, stored feasible minima and source hashes; no full freshbank/start-pose regeneration'))
    records = protocol['records']; index = C.read(output/'PREDICTION_ROW_INDEX.json')['entries']
    if [r['id'] for r in records] != ids or [r['id'] for r in index] != ids:
        raise ValueError('Exact ordered positive records and worker row index required')
    failures = []; components = []; count = checks = accepted = fallback = no_pose = 0; partitions = Counter()
    try:
        for packet, comparison, row in joined_rows(C.rows(output/'INPUTS.jsonl.gz'),
                C.rows(output/'E_COMPARISON.jsonl.gz'), C.rows(output/'PREDICTIONS.jsonl.gz'), records, index, ids):
            result = audit_row(packet, comparison, row, models=models)
            for check in result['checks']:
                checks += 1
                if not check['passed']:
                    failures.append(dict(id=ids[count], **check))
            components.extend(result['components']); accepted += result['accepted']; fallback += result['fallback']
            no_pose += int(result['no_pose']); partitions[packet['identifiers']['partition']] += 1; count += 1
            if count % 20 == 0:
                print(dict(rows=count, checks=checks, failures=len(failures)), flush=True)
    except BaseException as error:
        C.write(dest/'AUDIT_FAILURE_V1.json', dict(rows_completed=count, checks=checks, failures=failures,
            exception=type(error).__name__, reason=str(error), pending_ids=ids[count:],
            attempt_preserved=True, missing_or_mismatched_rows_not_dropped=True,
            complete_zero_failure_audit_seal_created=False))
        raise
    if count != 238 or partitions != Counter(PARTITIONS):
        raise ValueError('Every CAL46/TEST49/REMAINDER143 row retained; severe0')
    _verify(*maps, sources)
    for relative, expected in bindings:
        if C.sha(output/relative) != expected:
            raise ValueError('Sealed positive/prediction bytes changed during audit: '+relative)
    C.save_rows(dest/'SELECTED_POSE_COMPONENTS.jsonl.gz', components)
    prediction_binding = dict(path=str((output/'PREDICTION_SEAL.json').relative_to(C.ROOT)),
                              sha256=expected_prediction_seal_sha256)
    report = dict(schema='INDEPENDENT_POSITIVE_CLEAN_FRESH_BI_AUDIT_V32', rows=count, checks=checks,
        failure_count=len(failures), failures=failures, partition_counts=dict(partitions),
        accepted_arms=accepted, fallback_arms=fallback, no_pose_rows=no_pose,
        maximum_absolute_scalar_difference=max((r['absolute_total_difference'] for r in components), default=0.),
        prediction_seal=prediction_binding, source_sha256=sources, central_plan_sha256=expected_central_plan_sha256,
        model_seal_sha256=expected_model_seal_sha256, positive_native_helper_sha256=HELPER_SHA256,
        new_consumer_planner_objective_or_PnP_optimizer_calls=0, reference_RGB_or_NN_calls=0,
        full_unseen_seed_bank_or_start_rotations_regenerated=False, selected_pose_arithmetic_only=True,
        old_V17_guard_primitives_and_independent_V29_scalar_reused=True,
        scope='selected-pose native/O/S/C arithmetic and guards; stored feasible contender minima/source hashes; no graph/native identity/global optimum/accuracy certification',
        default_adoption=False)
    C.write(dest/'AUDIT.json', report)
    C.write(dest/'AUDIT_SEAL.json', dict(receipt_sha256=C.sha(dest/'AUDIT.json'),
        components_sha256=C.sha(dest/'SELECTED_POSE_COMPONENTS.jsonl.gz'), plan_sha256=C.sha(dest/'PLAN.json'),
        source_sha256=sources, prediction_seal=prediction_binding))
    print(dict(rows=count, checks=checks, failures=len(failures)), flush=True)
    if failures:
        raise SystemExit(1)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-independent-audit-GO', action='store_true')
    parser.add_argument('--prediction-seal-sha256')
    parser.add_argument('--central-plan-sha256')
    parser.add_argument('--model-seal-sha256')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(root_independent_audit_GO=args.root_independent_audit_GO,
         expected_prediction_seal_sha256=args.prediction_seal_sha256,
         expected_central_plan_sha256=args.central_plan_sha256,
         expected_model_seal_sha256=args.model_seal_sha256, output=args.output)
