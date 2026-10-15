"""Pure metadata construction, no actual CAL rows or numeric labels."""
from copy import deepcopy
import numpy as np
import pytest

from . import prepare_geometry_guided_native_evidence_v4_31 as M
from .test_geometry_guided_edge_authentication_v4_31 import fixture


def rows():
    base, observation, kwargs, _ = fixture()
    geometry = kwargs['initial_geometry']
    family = dict(name='ORIGINAL_WD', native_xyz=base.xyz.tolist(),
        physical_registration_Q=base.Q.tolist(),
        coarse_pose=dict(R=geometry['R_cf'], t=geometry['centroid']))
    visible = [deepcopy(a['source_record']) for a in base.prepartial_bound_source_archives.values()]
    task = dict(id='synthetic:cal', partition='CAL', session='synthetic', states=base.states,
        initial_self_hidden=geometry['initial_self_hidden'], image_sha256='a'*64,
        spec=dict(K=base.K.tolist(), states=base.states, visible=visible, domains=[], image_hw=base.image_hw,
                  config=dict(prior_weight=0., analytic_jacobian=True)), boundary_merged=[], u_regions=[],
        families=dict(ORIGINAL_WD=family),
        complete_BI_evidence=dict(initial_projection_xy=deepcopy(geometry['initial_projection_xy'])))
    row = dict(id=task['id'], partition='CAL', session=task['session'], states=task['states'],
        visible_measurements=deepcopy(visible), domains=[], u_regions=[], image_sha256='a'*64,
        initial_geometry=deepcopy(geometry), observation=observation, selection_profile=M.G.SELECTION_PROFILE)
    return task, row


def test_two_policy_zero_component_metadata_and_measured_axes_before_labels():
    task, row = rows(); before = deepcopy((task, row))
    prepared, metadata = M.prepare_one(task, row)
    assert prepared['observation_generation'] == M.G.OBSERVATION_FAMILY
    assert prepared['noise_population'] == M.G.POPULATION
    assert prepared['raw_observation_generation'] == M.G.RAW_OBSERVATION_GENERATION
    assert prepared['selection_profile'] == M.G.SELECTION_PROFILE
    assert metadata['profiles']['ALL_BI_REPLACE']['groups'] == 1
    assert metadata['profiles']['RGB_NONFALLBACK_PRESERVE']['groups'] == 0
    for policy, profile in prepared['profiles'].items():
        for group in profile['groups']:
            component = group['components'][0]
            provenance = component['noise_provenance']
            assert provenance['mode'] == 'ZERO_COMPONENT_PREFIT_METADATA_NOT_FITTED'
            assert provenance['fit_partition'] == 'CAL' and provenance['noise_population'] == M.G.POPULATION
            assert provenance['raw_observation_generation'] == M.G.RAW_OBSERVATION_GENERATION
            assert component['normal_variance_px2'] == component['normal_floor_px']**2
            assert component['angular_variance_rad2'] == component['angular_sine_floor']**2
            assert group['components'][0]['mixture_component_prior'] == 1.
        for tangent in profile['tangent_records']:
            component = next(g['components'][0] for g in profile['groups']
                             if g['components'][0]['component_id'] == tangent['selected_line_ref'])
            assert abs(np.asarray(tangent['selected_line_tangent_unit_xy']) @ component['normal_unit_xy']) < 1e-12
    assert (task, row) == before


def test_zero_selected_observation_kept_in_both_profiles():
    task, row = rows()
    row['observation']['selected_chain_refs'] = []
    row['observation']['joint_graph']['selected_line_names'] = []
    prepared, metadata = M.prepare_one(task, row)
    assert set(prepared['profiles']) == set(M.G.SOURCE_POLICIES)
    assert all(not p['groups'] and not p['tangent_records'] for p in prepared['profiles'].values())
    assert metadata['authenticated_groups'] == 0


@pytest.mark.parametrize('change', ['q', 'rotation', 'center', 'U', 'image', 'projection', 'profile', 'geometry_digest'])
def test_original_join_or_new_profile_change_rejects_before_planning(change):
    task, row = rows()
    if change == 'q': row['visible_measurements'][0]['xy'][0] += 1.
    elif change == 'rotation': row['initial_geometry']['R_cf'][0][0] += .01
    elif change == 'center': row['initial_geometry']['centroid'][2] += .01
    elif change == 'U': row['u_regions'] = [{'native_corner': 0}]
    elif change == 'image': row['image_sha256'] = 'b'*64
    elif change == 'projection': row['initial_geometry']['initial_projection_xy'][0][0] += 1.
    elif change == 'profile': row['selection_profile'] = M.PARENT.G.SELECTION_PROFILE
    else: row['observation']['initial_geometry_digest'] = '0'*64
    with pytest.raises(ValueError): M.prepare_one(task, row)


def test_primary_gate_precedes_coordinate_payload_and_parent_globals_unchanged():
    with pytest.raises(ValueError, match='CAL-only'):
        M.prepare_one(dict(partition='TEST'), dict(partition='TEST'))
    assert M.PARENT.native_baseline.__globals__['G'] is M.PARENT.G
    assert M.native_baseline.__globals__['G'] is M.G
    assert M.PARENT.G.OBSERVATION_FAMILY != M.G.OBSERVATION_FAMILY
