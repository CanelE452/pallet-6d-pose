"""V31 q-anchored, signed-native-ray population; unchanged V29 physics.

The ORIGINAL observer geometry is evidence provenance, never the trial WD
geometry or a pose prior. Its nonS model rays and the immutable prepartial BI
archives are rebound before planning. Legacy v29/v30 dictionary names below
are internal wire aliases only. No file, image or label is read here.
"""
from copy import deepcopy
import numpy as np

from . import geometry_guided_edge_problem_v4_29 as PARENT
from . import geometry_guided_edge_problem_v4_30 as V30
from . import observed_anchor_native_ray_association_v4_31 as ASSOCIATION

BASELINE, EDGE, N = PARENT.BASELINE, PARENT.EDGE, PARENT.N
V, J, P = PARENT.V, PARENT.J, PARENT.P
SOURCE_POLICIES, FALLBACK = PARENT.SOURCE_POLICIES, PARENT.FALLBACK
RAW_OBSERVATION_GENERATION = V30.RAW_OBSERVATION_GENERATION
OBSERVATION_FAMILY = 'OBSERVED_Q_ANCHORED_FIXED_NATIVE_RAY_ACTUAL_EDGES_V31'
POPULATION = 'V31_Q_ANCHORED_SIGNED_RAY_RUNTIME_ACTIVE_ONE_LINE_PER_PHYSICAL_EDGE'
SELECTION_PROFILE = ASSOCIATION.SELECTION_PROFILE
GEOMETRY_DIGEST_FIELDS = ('native_xyz', 'R_cf', 'centroid', 'physical_registration_Q',
                          'K', 'image_hw', 'initial_self_hidden')
GEOMETRY_BINDING_SCHEMA = 'ORIGINAL_NON_S_MODEL_RAYS_AND_PREPARTIAL_BI_ARCHIVES_V31'
_digest, _id, _unit = PARENT._digest, PARENT._id, PARENT._unit
_archive, _finite_actual_pair = PARENT._archive, PARENT._finite_actual_pair
bound_BI_points = V30.bound_BI_points


def _array(value, shape, name):
    array = np.asarray(value, float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(name + ' must have its fixed finite shape')
    return array


def bind_initial_geometry(base, geometry, *, expected_digest=None):
    """Bind ORIGINAL model rays, skipping supplied S projection slots entirely.

    The 7-field digest deliberately excludes all 2D projection slots. Only
    nonS MODEL projections are checked against native R/t/K, and copied; no
    O/T/U observed coordinate is accepted. ``centroid`` is model center, so
    native t is center-R*mean(X), including nonzero native-coordinate origins.
    A changed trial WD is allowed; it must retain the same physical body.
    """
    if not isinstance(base, BASELINE.PrepartialRGBPointPreservingNativeProblem):
        raise ValueError('Exact prepartial native baseline required')
    hidden = [s == 'S' for s in base.states]
    if list(geometry['initial_self_hidden']) != hidden:
        raise ValueError('Initial S gate must bind before model projection fields')
    xyz = _array(geometry['native_xyz'], (8, 3), 'Original native XYZ')
    R = _array(geometry['R_cf'], (3, 3), 'Original native R')
    Q = _array(geometry['physical_registration_Q'], (3, 3), 'Original physical Q')
    K = _array(geometry['K'], (3, 3), 'Original K')
    center = _array(geometry['centroid'], (3,), 'Original model center')
    if any(not np.allclose(a.T @ a, np.eye(3), atol=1e-7, rtol=0)
           or not np.isclose(np.linalg.det(a), 1., atol=1e-7, rtol=0) for a in (R, Q)):
        raise ValueError('Original native R and physical Q must be rotations')
    if (not np.array_equal(K, base.K) or tuple(geometry['image_hw']) != tuple(base.image_hw)):
        raise ValueError('Original image canvas/camera must bind the trial problem')
    # A native-channel/WD basis change preserves the unordered physical body.
    original_body = xyz @ Q
    trial_body = np.asarray(base.xyz) @ np.asarray(base.Q)
    matches = np.all(np.isclose(original_body[:, None, :], trial_body[None, :, :],
                                atol=1e-7, rtol=0), axis=2)
    if not np.all(matches.sum(axis=0) == 1) or not np.all(matches.sum(axis=1) == 1):
        raise ValueError('Original source and trial WD must describe the same physical body')
    source = {key: deepcopy(geometry[key]) for key in GEOMETRY_DIGEST_FIELDS}
    digest = _digest(source)
    if expected_digest is not None and digest != expected_digest:
        raise ValueError('Exact original 7-field observer geometry digest required')
    native_t = center - R @ xyz.mean(axis=0)
    supplied = geometry['initial_projection_xy']
    projection = [None] * 8
    for k, is_hidden in enumerate(hidden):
        if is_hidden:
            continue  # Do not even index a supplied S model/observed slot.
        camera = R @ xyz[k] + native_t
        hom = K @ camera
        value = supplied[k]
        if camera[2] <= 1e-9 or abs(hom[2]) <= 1e-9:
            if value is not None:
                raise ValueError('Nonpositive-depth original model ray must be missing')
            continue
        q = _array(value, (2,), 'Original nonS MODEL projection')
        if not np.allclose(q, hom[:2] / hom[2], atol=1e-6, rtol=0):
            raise ValueError('Original nonS ray must bind its original native model geometry')
        projection[k] = q.tolist()
    points = bound_BI_points(base)
    archives = [dict(native_corner=k, xy=deepcopy(points[k]['record']['xy']),
                     source=points[k]['record']['source']) for k in sorted(points)]
    return dict(schema=GEOMETRY_BINDING_SCHEMA, geometry=source,
        initial_projection_xy=projection, initial_geometry_digest=digest,
        initial_geometry_digest_fields=list(GEOMETRY_DIGEST_FIELDS),
        original_BI_archives=archives, original_BI_archive_digest=_digest(archives),
        S_projection_slots_read=0, unknown_observed_coordinate_slots_read=0,
        original_geometry_not_trial_WD_or_pose=True)


def _bound_geometry(base, bundle):
    binding = bundle['initial_model_ray_binding']
    if binding.get('schema') != GEOMETRY_BINDING_SCHEMA:
        raise ValueError('Explicit V31 original model-ray binding required')
    geometry = {**binding['geometry'], 'initial_projection_xy': binding['initial_projection_xy']}
    checked = bind_initial_geometry(base, geometry, expected_digest=bundle['initial_geometry_digest'])
    if checked != binding:
        raise ValueError('Original geometry/rays and exact immutable BI archive binding changed')
    return geometry


def _query_node(group, geometry_digest):
    """Reconstruct admission from exact authenticated raw ref, not its verdict."""
    k, mate = _id(group['native_corner']), _id(group['native_neighbor'])
    reference = group['selected_chain_ref']
    if (_id(reference['native_corner']) != k or _id(reference['native_neighbor']) != mate
            or any(isinstance(reference[key], (bool, np.bool_))
                   or not isinstance(reference[key], (int, np.integer)) or reference[key] < 0
                   for key in ('ray_index', 'chain_index'))):
        raise ValueError('Exact raw directed ray/chain reference required')
    guide = reference['guide_reference']
    if (guide.get('guide_reference_id') != reference['guide_reference_id']
            or _digest({key: value for key, value in guide.items()
                        if key != 'guide_reference_id'}) != reference['guide_reference_id']
            or _id(guide['native_corner']) != k or _id(guide['native_neighbor']) != mate
            or guide.get('original_geometry_digest') != geometry_digest):
        raise ValueError('Original actual-query dictionary/digest/native IDs/geometry required')
    direction = _array(group['generating_guide_direction_xy'], (2,), 'Actual query direction')
    if not np.array_equal(direction, np.asarray(guide['actual_search_direction_xy'], float)):
        raise ValueError('Authenticated line and exact actual query direction disagree')
    return dict(native_edge=[k, mate], native_corner=k, native_neighbor=mate,
        normal=group['normal_unit_xy'], offset=group['offset_px'],
        line_record=dict(guide_reference=guide, generating_guide_direction_xy=direction.tolist()))


def validate_noise_population(provenance):
    if (provenance.get('fit_partition') != 'CAL'
            or provenance.get('observation_family') != OBSERVATION_FAMILY
            or provenance.get('noise_population') != POPULATION
            or ('population' in provenance and provenance['population'] != POPULATION)
            or provenance.get('selection_profile') != SELECTION_PROFILE
            or provenance.get('raw_observation_generation') != RAW_OBSERVATION_GENERATION):
        raise ValueError('Separate V31 family/profile/population/raw-generation CAL noise required')


def assess_bound_line(base, node, configuration, *, geometry):
    return ASSOCIATION.assess_observed_anchor_native_ray_association(node,
        bound_BI_points(base), base.states, [s == 'S' for s in base.states],
        configuration, geometry=geometry)


def _parent_bundle(base, bundle):
    if (bundle.get('observation_family') != OBSERVATION_FAMILY
            or bundle.get('population') != POPULATION
            or bundle.get('selection_profile') != SELECTION_PROFILE
            or bundle.get('producer_generation') != RAW_OBSERVATION_GENERATION):
        raise ValueError('Separate V31 family/population/profile and unchanged raw V29 generation required')
    cfg = bundle['association_configuration']
    reach = float(cfg['normal_search_radius_px'])
    if not np.isfinite(reach) or reach <= 0:
        raise ValueError('Original positive configured normal reach required')
    # S gates precede any line/ref payload, including malformed core callers.
    for group in bundle['groups']:
        k, mate = _id(group['native_corner']), _id(group['native_neighbor'])
        if base.states[k] not in ('B', 'I') or base.states[mate] == 'S':
            raise ValueError('State/S gates precede actual query/line payload')
    geometry = _bound_geometry(base, bundle)
    checks = []
    for group in bundle['groups']:
        validate_noise_population(group['noise_provenance'])
        node = _query_node(group, bundle['initial_geometry_digest'])
        check = assess_bound_line(base, node, cfg, geometry=geometry)
        if not check['passes']:
            raise ValueError('V31 original BI q query/native ray association failed: ' + check['reason'])
        checks.append(dict(component_id=group['component_id'], **check))
    return ({**bundle, 'observation_family': PARENT.OBSERVATION_FAMILY,
             'population': PARENT.POPULATION}, checks)


def _tag_plan(plan, bundle, checks):
    plan.update(observation_family=OBSERVATION_FAMILY, population=POPULATION,
        raw_observation_generation=RAW_OBSERVATION_GENERATION, selection_profile=SELECTION_PROFILE,
        association_configuration=deepcopy(bundle['association_configuration']),
        initial_model_ray_binding=deepcopy(bundle['initial_model_ray_binding']),
        original_BI_query_and_native_ray_association_recomputed=checks,
        old_V29_or_V30_noise_relabelled=False, numerical_factor_and_guard_math='UNCHANGED_V29',
        legacy_dictionary_keys_are_internal_wire_aliases=True)
    return plan


def empty_geometry_guided_evidence(base, *, initial_geometry, image_binding,
                                    association_configuration, reason):
    """Explicit uniform no-evidence path, including unavailable NEW noise.

    No old noise or selected line is substituted. The frozen no-owner baseline
    objective/guards remain exact, with a separately named V31 abstention.
    """
    if not isinstance(reason, str) or not reason:
        raise ValueError('Explicit uniform V31 abstention reason required')
    if (tuple(image_binding['image_hw']) != tuple(base.image_hw)
            or not isinstance(image_binding.get('sha256'), str)
            or len(image_binding['sha256']) != 64):
        raise ValueError('Exact image binding required even for empty V31 evidence')
    binding = bind_initial_geometry(base, initial_geometry)
    return dict(schema='EXPLICIT_EMPTY_GEOMETRY_GUIDED_EVIDENCE_V31',
        observation_family=OBSERVATION_FAMILY, population=POPULATION,
        producer_generation=RAW_OBSERVATION_GENERATION, selection_profile=SELECTION_PROFILE,
        association_configuration=deepcopy(association_configuration), groups=[], ordered_corner_pairs=[],
        initial_geometry_digest=binding['initial_geometry_digest'], initial_model_ray_binding=binding,
        image_binding=deepcopy(image_binding), evidence_availability='UNAVAILABLE_UNIFORM_ABSTENTION',
        abstention_reason=reason, old_noise_substitution=False, new_image_queries=0,
        new_precise_KPs=0, permanent_S_U_local_queries=0)


def plan_geometry_guided_replacement(base, bundle, *, source_policy):
    converted, checks = _parent_bundle(base, bundle)
    plan = PARENT.plan_geometry_guided_replacement(base, converted, source_policy=source_policy)
    return _tag_plan(plan, bundle, checks)


class GeometryGuidedEdgeProblem(PARENT.GeometryGuidedEdgeProblem):
    def __init__(self, base, bundle, *, source_policy, tangent_gamma):
        converted, checks = _parent_bundle(base, bundle)
        super().__init__(base, converted, source_policy=source_policy, tangent_gamma=tangent_gamma)
        _tag_plan(self.plan, bundle, checks)

    def objective(self, p):
        value, gradient, parts = super().objective(p)
        parts.update(v29_observation_family=OBSERVATION_FAMILY,
            v31_observation_family=OBSERVATION_FAMILY, v31_selection_profile=SELECTION_PROFILE,
            v31_cost_plan=self.plan)
        return value, gradient, parts
