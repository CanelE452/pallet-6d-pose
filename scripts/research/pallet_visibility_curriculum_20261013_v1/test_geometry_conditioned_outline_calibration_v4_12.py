from copy import deepcopy

import numpy as np
import pytest

from . import geometry_conditioned_outline_calibration_v4_12 as G


def _records():
    rows, geometry = [], {}
    for session, dimensions, bias, count in [('plastic_a', [1.1, .11, 1.3], .01, 10),
                                               ('plastic_b', [1.3, .11, 1.1], .01, 4),
                                               ('wood', [.8, .14, .59], -.02, 10)]:
        for k in range(count):
            identity = f'{session}:{k}'
            values = np.array([bias-.001, bias+.001])
            rows.append(dict(id=identity, partition='CAL', session=session, supported=True,
                             mask_diagonal_px=100., signed_distances_normalized=values.tolist(),
                             signed_distances_px=(100*values).tolist()))
            geometry[identity] = dimensions
    return rows, geometry


def test_TEST_rejected_before_residual_access():
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Forbidden residual accessed')
    row = dict(id='test', partition='TEST', signed_distances_normalized=Poison())
    with pytest.raises(ValueError, match='CAL only'):
        G.fit_geometry_outline_model([row], {'test': [1., 2., 3.]})


def test_geometry_key_swap_invariant_and_not_session_material_route():
    assert G.geometry_signature([1.1, .11, 1.3]) == G.geometry_signature([1.3, .11, 1.1])
    rows, geometry = _records()
    for row in rows:
        row['material'] = 'misleading_material_label'
    model = G.fit_geometry_outline_model(rows, geometry)
    group = model['geometry_groups'][G.geometry_signature([1.1, .11, 1.3])]
    assert group['supported_train_frame_count'] == 14
    assert group['fit_sessions'] == ['plastic_a', 'plastic_b']
    with pytest.raises(ValueError, match='positive'):
        G.geometry_signature([1., 0., 2.])


def test_single_session_requires_ten_and_is_disclosed():
    rows, geometry = _records()
    model = G.fit_geometry_outline_model(rows, geometry)
    group = model['geometry_groups'][G.geometry_signature([.8, .14, .59])]
    assert group['group_mean_available'] and group['single_session_group']
    assert not group['independently_held_session_validation_available']
    rows = rows[:-1]
    geometry.pop('wood:9')
    model = G.fit_geometry_outline_model(rows, geometry)
    params = G.geometry_outline_parameters(model, [.8, .14, .59], 100.)
    assert params['group_mean_fallback'] and params['mean_source'] == 'shared_global'


def test_LOGO_exclusion_and_missing_group_global_fallback():
    rows, geometry = _records()
    model = G.fit_geometry_outline_model(rows, geometry, heldout_session='wood')
    assert not any(identity.startswith('wood:') for identity in model['fit_ids'])
    group = model['geometry_groups'][G.geometry_signature([.8, .14, .59])]
    assert group['supported_train_frame_count'] == 0 and not group['group_mean_available']
    assert G.geometry_outline_parameters(model, [.8, .14, .59], 100.)['group_mean_fallback']


def test_all_profiles_share_exact_global_sigma_and_only_mean_changes():
    rows, geometry = _records()
    model = G.fit_geometry_outline_model(rows, geometry)
    global_model = G.GLOBAL.fit_outline_noise_model(rows)
    assert model['shared_global_model'] == global_model
    baseline = G.GLOBAL.outline_noise_parameters(global_model, 100.)
    params = [G.geometry_outline_parameters(model, [.8, .14, .59], 100., profile=p) for p in G.PROFILES]
    assert all(p['total_normal_sigma_px'] == baseline['total_normal_sigma_px'] for p in params)
    assert params[0]['outward_mask_bias_px'] == 0
    assert params[1]['outward_mask_bias_px'] == baseline['outward_mask_bias_px']
    assert params[2]['outward_mask_bias_px'] == pytest.approx(-2.)
    unknown = G.geometry_outline_parameters(model, [2., 3., 4.], 100.)
    assert unknown['group_mean_fallback'] and unknown['outward_mask_bias_px'] == baseline['outward_mask_bias_px']


def test_frame_equal_mean_despite_boundary_pixel_counts_and_tamper_rejected():
    rows, geometry = _records()
    wood = [r for r in rows if r['session'] == 'wood']
    wood[0]['signed_distances_normalized'] = [10.] * 1000
    wood[0]['signed_distances_px'] = [1000.] * 1000
    model = G.fit_geometry_outline_model(rows, geometry)
    group = model['geometry_groups'][G.geometry_signature([.8, .14, .59])]
    assert group['bias_ratio'] == pytest.approx(-.02)
    altered = deepcopy(model)
    altered['geometry_groups'][G.geometry_signature([.8, .14, .59])]['bias_ratio'] = 4.
    with pytest.raises(ValueError, match='digest changed'):
        G.geometry_outline_parameters(altered, [.8, .14, .59], 100.)


def test_unsupported_records_inventory_without_coordinate_read():
    rows, geometry = _records()
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Unsupported residual accessed')
    rows.append(dict(id='unsupported', partition='CAL', session='plastic_a', supported=False,
                     signed_distances_normalized=Poison()))
    geometry['unsupported'] = [1.1, .11, 1.3]
    model = G.fit_geometry_outline_model(rows, geometry)
    assert model['unsupported_CAL_ids'] == ['unsupported']
    assert model['fit_frame_count'] == 24


def test_inconsistent_normalization_and_unknown_profiles_rejected():
    rows, geometry = _records()
    rows[0]['signed_distances_px'][0] += 1
    with pytest.raises(ValueError, match='normalized/pixel'):
        G.fit_geometry_outline_model(rows, geometry)
    rows, geometry = _records()
    model = G.fit_geometry_outline_model(rows, geometry)
    with pytest.raises(ValueError, match='Unknown'):
        G.geometry_outline_parameters(model, [.8, .14, .59], 100., profile='frame_specific')
