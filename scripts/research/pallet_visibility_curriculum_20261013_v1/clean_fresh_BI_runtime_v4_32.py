"""Pure clean geometry/seed boundary for the proposed V32 evaluator.

This is not a full evaluator or an ancestry verifier. The caller first checks
the exact independent-initial-input receipt, then supplies only these named
inputs. No legacy task, baseline, pose bank, old initialization or external
guide argument exists. Original hardguard/range inputs and new authenticated
line evidence stay explicit inputs to the caller's pure problem factory;
this module does not copy an opaque old task into a numerical worker.
"""
from __future__ import annotations

from copy import deepcopy
from types import FunctionType

import numpy as np

from . import fresh_observed_BI_pnp_proposals_v4_32 as FRESH
from . import initial_only_observed_ray_association_v4_32 as ADMISSION
from . import statewise_two_wd_family_v4_9 as WD
from . import refine_native_edge_pose_v4_17 as REFINE


SCHEMA = 'CLEAN_GEOMETRY_AND_FRESH_BI_SEEDS_RUNTIME_V32'
START_SCHEMA = 'CLEAN_FRESH_NATIVE_SIX_DIVERSE_PLUS_COARSE_STARTS_V32'
START_CONFIGURATION = dict(maximum_native_ranked_starts=6,
    minimum_rotation_separation_deg=2., minimum_translation_separation_ratio=.01)
_COARSE_KEYS = frozenset(('R', 't', 'source', 'independent_of_cached_pose_bank',
    'physical_registration_Q', 'R_physical'))
_BINDING_KEYS = frozenset(('path', 'sha256', 'verified_by_caller',
    'independent_initial_geometry_and_BI_q'))
_IDENTIFIER_KEYS = frozenset(('id', 'partition', 'session', 'image', 'image_sha256'))


def _hash(value, key):
    return FRESH._digest({k: v for k, v in value.items() if k != key})


def _source_binding(value):
    if (not isinstance(value, dict) or set(value) != _BINDING_KEYS
            or not isinstance(value['path'], str) or not value['path']
            or value['sha256'] != ADMISSION.INITIAL_SOURCE_CLOSURE_SHA256
            or value['verified_by_caller'] is not True
            or value['independent_initial_geometry_and_BI_q'] is not True):
        raise ValueError('Exact caller-verified independent initial-input closure required')
    return deepcopy(value)


def _identifiers(value):
    if value is None:
        return {}
    if (not isinstance(value, dict) or not set(value) <= _IDENTIFIER_KEYS
            or any(not isinstance(v, str) for v in value.values())):
        raise ValueError('Only string output/order metadata allowed; no runtime routing fields')
    return deepcopy(value)


def _abstention(reason, identifiers, binding):
    result = dict(schema=SCHEMA, ready=False, accepted=False, pose_available=False,
        reason=reason, identifiers=identifiers, initial_source_binding=binding,
        families={}, fresh_seed_banks={}, row_retained=True,
        old_method_or_baseline_fallback_used=False, cached_pose_inputs=0)
    result['clean_task_sha256'] = _hash(result, 'clean_task_sha256')
    return result


def _families(xyz, coarse, Q):
    """Use the frozen centered WD construction, preserving a nonzero origin."""
    mean = xyz.mean(axis=0)
    R = np.asarray(coarse['R']); t = np.asarray(coarse['t'])
    center = R @ mean + t
    centered_pose = dict(R=R.tolist(), t=center.tolist(),
        physical_registration_Q=Q.tolist(), R_physical=(R @ Q).tolist())
    generated = WD.make_wd_families(xyz - mean, centered_pose, initializations=None)
    result = {}
    for item in generated:
        name = item['name']; Qf = np.asarray(item['physical_registration_Q'])
        Rf = np.asarray(item['coarse_pose']['R'])
        mean_f = Qf @ Q.T @ mean
        body = np.asarray(item['native_xyz']) + mean_f
        tf = center - Rf @ mean_f
        if name == 'ORIGINAL_WD':
            body, Rf, tf = xyz.copy(), R.copy(), t.copy()
        initial = dict(R=Rf.tolist(), t=tf.tolist(), physical_registration_Q=Qf.tolist(),
            R_physical=(Rf @ Qf).tolist(), source=FRESH.COARSE_SOURCE,
            independent_of_cached_pose_bank=True)
        result[name] = dict(name=name, native_xyz=body.tolist(),
            cf_extents=np.ptp(body, axis=0).tolist(), physical_registration_Q=Qf.tolist(),
            coarse_pose=initial, coarse_model_center_camera_xyz=center.tolist(),
            initializations_imported_from_legacy_task=False)
        if (not np.allclose(Rf @ Qf, R @ Q, atol=1e-10, rtol=0)
                or not np.allclose(Rf @ body.mean(axis=0) + tf, center, atol=1e-10, rtol=0)
                or WD._unordered_gap(body @ Rf.T + tf, xyz @ R.T + t) > 1e-9):
            raise ValueError('Fresh family must preserve independent initial physical body pose')
    return result


def prepare_clean_fresh_BI_task(native_xyz, K, states, initial_self_hidden,
        visible_measurements, initial_coarse_pose, physical_registration_Q, *,
        initial_source_binding, identifiers=None):
    """Construct geometry/seeds only; no legacy runtime task is accepted.

    Invalid independent coarse R/t/cheirality produces a retained explicit
    no-pose abstention. A missing/false provenance receipt remains a failclosed
    preparation error. All old/fresh observation hard guards must subsequently
    be supplied to the caller's current pure native problem, without relaxing
    them. Identifiers have no influence on candidate generation or selection.
    """
    binding, metadata = _source_binding(initial_source_binding), _identifiers(identifiers)
    if not isinstance(initial_coarse_pose, dict) or not set(initial_coarse_pose) <= _COARSE_KEYS:
        raise ValueError('Clean independent coarse fields only; no old initialization/bank payload')
    xyz = FRESH._array(native_xyz, (8, 3), 'native_xyz')
    camera = FRESH._array(K, (3, 3), 'K')
    Q = FRESH._rotation(physical_registration_Q, 'physical Q')
    if (len(states) != 8 or any(not isinstance(v, str) or v not in FRESH.STATES for v in states)
            or len(initial_self_hidden) != 8
            or any(not isinstance(v, (bool, np.bool_)) for v in initial_self_hidden)):
        raise ValueError('Eight fixed states and original boolean S gates required')
    states = list(states); hidden = [bool(v) for v in initial_self_hidden]
    try:
        if (initial_coarse_pose.get('source') != FRESH.COARSE_SOURCE
                or initial_coarse_pose.get('independent_of_cached_pose_bank') is not True):
            raise ValueError('independent initial coarse provenance attestation missing')
        R = FRESH._rotation(initial_coarse_pose['R'], 'independent initial coarse R')
        t = FRESH._array(initial_coarse_pose['t'], (3,), 'independent initial coarse native t')
        if ('physical_registration_Q' in initial_coarse_pose and not np.allclose(
                FRESH._rotation(initial_coarse_pose['physical_registration_Q'], 'initial Q'), Q,
                atol=1e-12, rtol=0)):
            raise ValueError('independent initial Q mismatch')
        if ('R_physical' in initial_coarse_pose and not np.allclose(
                FRESH._rotation(initial_coarse_pose['R_physical'], 'initial physical R'), R @ Q,
                atol=1e-7, rtol=0)):
            raise ValueError('independent initial physical R mismatch')
        FRESH._proposal(R, t, xyz, Q, 'OWN', FRESH.COARSE_SOURCE, {})
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        return _abstention('invalid_independent_initial_coarse: ' + str(error), metadata, binding)
    # Gate before copying/hashing xy/source/covariance. Excluded S/U/O/T
    # measurement payloads never enter the resulting clean input.
    admitted = FRESH._measurements(visible_measurements, states, hidden)
    coarse = dict(R=R.tolist(), t=t.tolist(), physical_registration_Q=Q.tolist(),
        R_physical=(R @ Q).tolist(), source=FRESH.COARSE_SOURCE,
        independent_of_cached_pose_bank=True)
    families = _families(xyz, coarse, Q)
    banks = {}
    for name, family in families.items():
        banks[name] = FRESH.generate_fresh_observed_BI_pnp_proposals(
            family['native_xyz'], camera, states, hidden, admitted,
            family['coarse_pose'], family['physical_registration_Q'])
    result = dict(schema=SCHEMA, ready=True, accepted=False, pose_available=False,
        identifiers=metadata, initial_source_binding=binding,
        states=states, initial_self_hidden=hidden,
        spec=dict(K=camera.tolist(), states=states, visible=admitted),
        families=families, fresh_seed_banks=banks, row_retained=True,
        current_hardguards_and_typed_evidence_must_be_attached_by_caller=True,
        source_binding_is_caller_attestation_not_ancestry_verification=True,
        old_method_or_baseline_fallback_used=False, cached_pose_inputs=0)
    result['clean_task_sha256'] = _hash(result, 'clean_task_sha256')
    return result


def _validate(clean):
    if (clean.get('schema') != SCHEMA
            or clean.get('clean_task_sha256') != _hash(clean, 'clean_task_sha256')):
        raise ValueError('Constructed clean geometry/seed input changed')
    _source_binding(clean['initial_source_binding'])
    if clean.get('cached_pose_inputs') != 0 or clean.get('old_method_or_baseline_fallback_used') is not False:
        raise ValueError('No cached or baseline pose input allowed')


def _problem_binding(problem, clean, family_name):
    family = clean['families'][family_name]
    for name, expected in (('xyz', family['native_xyz']), ('K', clean['spec']['K']),
                           ('Q', family['physical_registration_Q'])):
        if not np.allclose(np.asarray(getattr(problem, name)), expected, atol=1e-12, rtol=0):
            raise ValueError('Current native problem ' + name + ' differs from clean physics')
    if list(problem.states) != clean['states']:
        raise ValueError('Current native states differ from clean original states')


def select_clean_fresh_native_starts(problem, clean, family_name):
    """Rank only fresh seeds using the explicit new current native core."""
    _validate(clean)
    if not clean['ready']:
        return dict(schema=START_SCHEMA, abstained=True, reason=clean['reason'], starts=[],
            clean_task_sha256=clean['clean_task_sha256'], family_name=family_name)
    _problem_binding(problem, clean, family_name)
    family = clean['families'][family_name]; bank = clean['fresh_seed_banks'][family_name]
    if any(p['source'] not in (FRESH.COARSE_SOURCE, FRESH.FRESH_SOURCE) for p in bank['proposals']):
        raise ValueError('Only freshly generated geometric/coarse proposal origins allowed')
    starts, selection = REFINE.choose_native_starts(problem, bank['proposals'], family['coarse_pose'],
        configuration=START_CONFIGURATION)
    packet = dict(schema=START_SCHEMA, abstained=False, family_name=family_name,
        clean_task_sha256=clean['clean_task_sha256'], starts=FRESH._plain(starts),
        selection=FRESH._plain(selection), fresh_PnP_input_sha256=bank['input_binding_sha256'],
        native_ranked_start_budget=6, independent_coarse_always_considered=True,
        legacy_reference_start_problem_used=False, cached_or_baseline_pose_inputs=0)
    packet['start_packet_sha256'] = _hash(packet, 'start_packet_sha256')
    return packet


def refine_clean_fresh_native_problem(problem, clean, family_name, start_packet):
    """Reuse pure continuous math with an authenticated fixed fresh start set.

    Native and containment callers may reuse the same source-policy packet.
    No parent evaluator _run, old reference_start_problem, or old _REFINER is
    called. The caller still owns the exact active objective/constraint class.
    """
    _validate(clean)
    if not clean['ready']:
        return dict(abstained=True, accepted=False, pose_available=False, solutions=[],
            candidates=[], reason=clean['reason'], row_retained=True,
            old_method_or_baseline_fallback_used=False)
    _problem_binding(problem, clean, family_name)
    if (start_packet.get('schema') != START_SCHEMA or start_packet.get('abstained') is not False
            or start_packet.get('clean_task_sha256') != clean['clean_task_sha256']
            or start_packet.get('family_name') != family_name
            or start_packet.get('start_packet_sha256') != _hash(start_packet, 'start_packet_sha256')):
        raise ValueError('Exact same clean-family fresh start packet required')
    family = clean['families'][family_name]; bank = clean['fresh_seed_banks'][family_name]
    allowed = {p['name']: p for p in bank['proposals']}
    allowed['OWN_INITIAL_COARSE'] = family['coarse_pose']
    if (not start_packet['starts'] or len(start_packet['starts']) > 7
            or start_packet['selection']['configuration'] != START_CONFIGURATION
            or start_packet['selection']['selected_start_names'] != [s['name'] for s in start_packet['starts']]
            or start_packet['fresh_PnP_input_sha256'] != bank['input_binding_sha256']):
        raise ValueError('At most six native starts plus own independent coarse')
    scale = max(float(np.linalg.norm(np.ptp(family['native_xyz'], axis=0))), 1e-3)
    for start in start_packet['starts']:
        expected = allowed.get(start['name'])
        # Native evaluate() may round-trip R through its rotvec chart. Permit
        # only floating-point noise from that roundtrip, never a new pose.
        if (expected is None or not np.allclose(start['R'], expected['R'], atol=2e-12, rtol=0)
                or not np.allclose(start['t'], expected['t'], atol=2e-12*scale, rtol=0)):
            raise ValueError('Start is not a generated fresh seed or independent coarse')

    def fixed(_problem, _proposals, _coarse_pose, *, configuration=None):
        starts = deepcopy(start_packet['starts'])
        for start in starts:
            start['R'] = np.asarray(start['R']); start['t'] = np.asarray(start['t'])
        return starts, deepcopy(start_packet['selection'])

    refine = FunctionType(REFINE.refine_native_edge_pose.__code__,
        {**REFINE.refine_native_edge_pose.__globals__, 'choose_native_starts': fixed},
        'refine_only_clean_fresh_BI_starts')
    refine.__kwdefaults__ = deepcopy(REFINE.refine_native_edge_pose.__kwdefaults__)
    result = refine(problem, bank['proposals'], family['coarse_pose'], configuration=START_CONFIGURATION)
    # The frozen numerical kernel's old objective labels do not describe the
    # caller's composed O/S/C objective. Preserve math, report scope explicitly.
    for key in ('mask_objective', 'pose_prior', 'new_P3P_or_global_starts'):
        result['search'].pop(key, None)
    result['search'].update(cached_or_baseline_pose_inputs=0,
        legacy_reference_start_problem_used=False, fresh_SQPNP_calls=bank['SQPNP_calls'],
        same_clean_start_packet_sha256=start_packet['start_packet_sha256'],
        exact_current_objective_and_guards_owned_by_problem=True)
    return result
