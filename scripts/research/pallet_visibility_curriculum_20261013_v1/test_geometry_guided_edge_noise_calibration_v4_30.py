from copy import deepcopy
import numpy as np
import pytest

from . import geometry_guided_edge_noise_calibration_v4_30 as M
from . import test_geometry_guided_edge_noise_calibration_v4_29 as FIXTURE


def records(policy='ALL_BI_REPLACE'):
    result = FIXTURE.records(policy)
    for row in result:
        row.update(observation_generation=M.OBSERVATION_FAMILY,
            raw_observation_generation=M.RAW_OBSERVATION_GENERATION,
            noise_population=M.POPULATION, selection_profile=M.SELECTION_PROFILE)
    return result


@pytest.fixture(scope='module')
def models():
    parent = M.PARENT
    globals_before = {name: {k: id(v) for k, v in getattr(parent, name).__globals__.items()}
        for name in ('fit_geometry_guided_line_model', 'fit_geometry_guided_tangent_model',
            'mixture_noise_parameters', 'tangent_variance_parameters')}
    recipes = deepcopy((parent.RECIPE, parent.TANGENT_RECIPE,
        M.LINE.RECIPE, M.TANGENT.RECIPE))
    old_records, new_records = FIXTURE.records(), records()
    before = deepcopy((old_records, new_records))
    answer = dict(old_line=parent.fit_geometry_guided_line_model(old_records,
        source_policy='ALL_BI_REPLACE'),
        line=M.fit_geometry_guided_line_model(new_records, source_policy='ALL_BI_REPLACE'),
        old_tangent=parent.fit_geometry_guided_tangent_model(old_records,
            source_policy='ALL_BI_REPLACE'),
        tangent=M.fit_geometry_guided_tangent_model(new_records,
            source_policy='ALL_BI_REPLACE'))
    assert (old_records, new_records) == before
    assert recipes == (parent.RECIPE, parent.TANGENT_RECIPE,
        M.LINE.RECIPE, M.TANGENT.RECIPE)
    assert globals_before == {name: {k: id(v) for k, v in getattr(parent, name).__globals__.items()}
        for name in globals_before}
    return answer


def test_raw_generation_unchanged_but_selection_population_and_recipe_distinct():
    assert M.RAW_OBSERVATION_GENERATION == M.PARENT.RAW_OBSERVATION_GENERATION
    assert M.OBSERVATION_FAMILY != M.PARENT.OBSERVATION_FAMILY
    assert M.POPULATION != M.PARENT.POPULATION
    for tangent in (False, True):
        recipe = M.recipe_for('ALL_BI_REPLACE', tangent=tangent)
        assert recipe['selection_profile'] == M.CORE.SELECTION_PROFILE
        assert recipe['raw_observation_generation'] == M.RAW_OBSERVATION_GENERATION
    assert M.SCHEMA != M.PARENT.SCHEMA
    assert M.TANGENT_SCHEMA != M.PARENT.TANGENT_SCHEMA


def test_physical_likelihood_values_and_gradients_are_identical():
    groups = [[dict(error=np.array([2., .12]), floor=np.array([1., .002]),
        mixture_component_prior=1.)], [dict(error=np.array([-8., -.03]),
        floor=np.array([2., .004]), mixture_component_prior=1.)]]
    for variances in ([0., 0.], [4., .01], [30., .2]):
        a, ga = M.mixture_noise_nll_and_gradient(variances, groups, [.4, .6])
        b, gb = M.PARENT.mixture_noise_nll_and_gradient(variances, groups, [.4, .6])
        assert a == b and np.array_equal(ga, gb)
    for gamma in (0., 1., 4.):
        assert M.scalar_tangent_nll_and_gradient(gamma, [2., -8.], [3., 7.], [.4, .6]) == \
            M.PARENT.scalar_tangent_nll_and_gradient(gamma, [2., -8.], [3., 7.], [.4, .6])
    assert M.prepared_tangent is M.PARENT.prepared_tangent
    assert M.conditional_covariance_floors is M.PARENT.conditional_covariance_floors


def test_same_synthetic_line_population_has_bit_exact_parameters_and_optimizer_searches(models):
    for key in ('normal_variance_component_px2', 'angular_variance_component_sine2',
            'training_mixture_nll', 'optimizer_searches', 'selected_optimizer_start_index',
            'fit_ids', 'session_weights', 'optimizer_conditioning_variance_scales'):
        assert models['line'][key] == models['old_line'][key]
    assert models['line']['selection_profile'] == M.SELECTION_PROFILE
    assert models['line']['schema'] == M.SCHEMA


def test_same_synthetic_tangent_population_has_bit_exact_gamma_and_optimizer_searches(models):
    for key in ('gamma', 'training_NLL', 'optimizer_searches',
            'selected_optimizer_start_index', 'fit_ids'):
        assert models['tangent'][key] == models['old_tangent'][key]
    assert models['tangent']['selection_profile'] == M.SELECTION_PROFILE
    assert models['tangent']['schema'] == M.TANGENT_SCHEMA


def test_profile_family_policy_and_new_recipe_propagate_to_runtime_exports(models):
    line = M.mixture_noise_parameters(models['line'], source_policy='ALL_BI_REPLACE')
    provenance = line['noise_provenance']
    assert provenance['observation_family'] == M.CORE.OBSERVATION_FAMILY
    assert provenance['selection_profile'] == M.CORE.SELECTION_PROFILE
    assert provenance['noise_population'] == M.POPULATION
    assert provenance['raw_observation_generation'] == M.RAW_OBSERVATION_GENERATION
    assert provenance['source_policy'] == 'ALL_BI_REPLACE'
    tangent = M.calibrated_archived_tangent_variance(models['tangent'], [0., 1.],
        [[4., 1.], [1., 9.]], source_policy='ALL_BI_REPLACE')
    assert tangent['tangent_variance_px2'] == 1. + 9. * models['tangent']['gamma']
    assert tangent['old_x4_engineering_control_variance_px2'] == 36.
    assert tangent['selection_profile'] == M.SELECTION_PROFILE


def test_V29_models_and_wrong_profile_or_policy_fail_closed(models):
    for fn, old, new in ((M.mixture_noise_parameters, 'old_line', 'line'),
            (M.tangent_variance_parameters, 'old_tangent', 'tangent')):
        with pytest.raises(ValueError, match='V30'):
            fn(models[old], source_policy='ALL_BI_REPLACE')
        with pytest.raises(ValueError):
            fn(models[new], source_policy='RGB_NONFALLBACK_PRESERVE')
        for value in (None, 'V29', 'another-selector'):
            changed = deepcopy(models[new]); changed['selection_profile'] = value
            with pytest.raises(ValueError, match='selection profile'):
                fn(changed, source_policy='ALL_BI_REPLACE')
    changed = deepcopy(models['line']); changed['normal_variance_component_px2'] += 1.
    with pytest.raises(ValueError):
        M.mixture_noise_parameters(changed, source_policy='ALL_BI_REPLACE')


@pytest.mark.parametrize('field,value', [('selection_profile', 'V29'),
    ('noise_population', M.PARENT.POPULATION),
    ('observation_generation', M.PARENT.OBSERVATION_FAMILY),
    ('raw_observation_generation', M.OBSERVATION_FAMILY), ('partition', 'TEST')])
def test_new_fits_reject_old_or_wrong_selection_metadata_before_numeric_payload(field, value):
    class Poison(dict):
        def __getitem__(self, key):
            raise AssertionError('Wrong population numeric payload read')
    rows = records(); rows[0][field] = value
    rows[0]['groups'] = [Poison()]; rows[0]['tangents'] = [Poison()]
    for fn in (M.fit_geometry_guided_line_model, M.fit_geometry_guided_tangent_model):
        with pytest.raises(ValueError, match='V30'):
            fn(rows, source_policy='ALL_BI_REPLACE')


def test_heldout_and_incompatible_payloads_are_not_read():
    class Poison(dict):
        def __getitem__(self, key):
            raise AssertionError('Held/incompatible numeric payload read')
    rows = records(); rows[0]['compatible'] = False
    for i in (0, 2, 3):
        rows[i]['groups'] = [Poison()]; rows[i]['tangents'] = [Poison()]
    for fn in (M.fit_geometry_guided_line_model, M.fit_geometry_guided_tangent_model):
        model = fn(rows, source_policy='ALL_BI_REPLACE', heldout_session='b')
        assert model['fit_ids'] == ['1'] and model['excluded_heldout_ids'] == ['2', '3']
