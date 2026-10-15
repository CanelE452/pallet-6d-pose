"""Synthetic noise-facade tests only; no actual CAL/error/model/row IO."""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import pytest

from . import geometry_guided_edge_noise_calibration_v4_31 as M
from . import geometry_guided_edge_noise_calibration_v4_30 as OLD30
from . import test_geometry_guided_edge_noise_calibration_v4_29 as FIXTURE


def records(policy='ALL_BI_REPLACE',module=M):
    rows=FIXTURE.records(policy)
    for row in rows:
        row.update(observation_generation=module.OBSERVATION_FAMILY,
            raw_observation_generation=module.RAW_OBSERVATION_GENERATION,
            noise_population=module.POPULATION,selection_profile=module.SELECTION_PROFILE)
    return rows


@pytest.fixture(scope='module')
def fitted():
    # These are four hardcoded scalar-error fixtures, not actual CAL records.
    parents=(M.PARENT,OLD30,M.LINE,M.TANGENT)
    paths=[Path(p.__file__)for p in parents]
    source_before={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
    recipes=deepcopy([(p.RECIPE,getattr(p,'TANGENT_RECIPE',None))for p in parents])
    fnames=('fit_geometry_guided_line_model','fit_geometry_guided_tangent_model',
        'mixture_noise_parameters','tangent_variance_parameters')
    globals_before={id(fn):{k:id(v)for k,v in fn.__globals__.items()}
        for p in (M.PARENT,OLD30)for name in fnames if(fn:=getattr(p,name,None))is not None}
    output={}
    for policy in M.SOURCE_POLICIES:
        old29=FIXTURE.records(policy);old30=records(policy,OLD30);new=records(policy)
        before=deepcopy((old29,old30,new))
        output[policy]={}
        for name,module,rows in [('old29',M.PARENT,old29),('old30',OLD30,old30),('new',M,new)]:
            output[policy][name+'_line']=module.fit_geometry_guided_line_model(rows,source_policy=policy)
            output[policy][name+'_tangent']=module.fit_geometry_guided_tangent_model(rows,source_policy=policy)
        assert (old29,old30,new)==before
    assert recipes==[(p.RECIPE,getattr(p,'TANGENT_RECIPE',None))for p in parents]
    assert source_before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
    assert globals_before=={id(fn):{k:id(v)for k,v in fn.__globals__.items()}
        for p in (M.PARENT,OLD30)for name in fnames if(fn:=getattr(p,name,None))is not None}
    return output


def test_strict_CORE31_constants_no_temporary_fallback_and_separate_recipe():
    assert M.OBSERVATION_FAMILY==M.CORE.OBSERVATION_FAMILY=='OBSERVED_Q_ANCHORED_FIXED_NATIVE_RAY_ACTUAL_EDGES_V31'
    assert M.POPULATION==M.CORE.POPULATION=='V31_Q_ANCHORED_SIGNED_RAY_RUNTIME_ACTIVE_ONE_LINE_PER_PHYSICAL_EDGE'
    assert M.SELECTION_PROFILE==M.CORE.SELECTION_PROFILE==M.CORE.ASSOCIATION.SELECTION_PROFILE
    assert M.RAW_OBSERVATION_GENERATION==M.PARENT.RAW_OBSERVATION_GENERATION==OLD30.RAW_OBSERVATION_GENERATION
    assert M.SCHEMA not in (M.PARENT.SCHEMA,OLD30.SCHEMA)
    assert M.TANGENT_SCHEMA not in (M.PARENT.TANGENT_SCHEMA,OLD30.TANGENT_SCHEMA)
    for policy in M.SOURCE_POLICIES:
        for tangent in (False,True):
            r=M.recipe_for(policy,tangent=tangent)
            assert r['source_policy']==policy and r['selection_profile']==M.SELECTION_PROFILE
            assert r['raw_observation_generation']==M.RAW_OBSERVATION_GENERATION
    assert not M.RECIPE['finite_alongline_native_identity_certified']


def test_numerical_line_and_scalar_tangent_functions_gradients_floors_identical():
    groups=[[dict(error=np.array([2.,.12]),floor=np.array([1.,.002]),mixture_component_prior=1.)],
        [dict(error=np.array([-8.,-.03]),floor=np.array([2.,.004]),mixture_component_prior=1.)]]
    for variances in ([0.,0.],[4.,.01],[30.,.2]):
        result,gradient=M.mixture_noise_nll_and_gradient(variances,groups,[.4,.6])
        old,oldgradient=M.PARENT.mixture_noise_nll_and_gradient(variances,groups,[.4,.6])
        assert result==old and np.array_equal(gradient,oldgradient)
    for gamma in (0.,1.,4.):
        assert M.scalar_tangent_nll_and_gradient(gamma,[2.,-8.],[3.,7.],[.4,.6])==M.PARENT.scalar_tangent_nll_and_gradient(gamma,[2.,-8.],[3.,7.],[.4,.6])
    for span,rms in ((11.,0.),(18.,.3),(75.,1.25)):
        assert np.array_equal(M.conditional_covariance_floors(span,rms),M.PARENT.conditional_covariance_floors(span,rms))
    assert M.prepared_tangent is M.PARENT.prepared_tangent


@pytest.mark.parametrize('policy',M.SOURCE_POLICIES)
def test_line_parameters_optimizer_budget_weights_and_conditioning_bitexact(fitted,policy):
    model=fitted[policy]['new_line']
    for old in ('old29_line','old30_line'):
        for key in ('normal_variance_component_px2','angular_variance_component_sine2','training_mixture_nll',
                'optimizer_searches','selected_optimizer_start_index','fit_ids','session_weights',
                'optimizer_conditioning_variance_scales','fit_weight_sum'):
            assert model[key]==fitted[policy][old][key]
    assert len(model['optimizer_searches'])==3


@pytest.mark.parametrize('policy',M.SOURCE_POLICIES)
def test_gamma_optimizer_parameters_and_selected_actual_axis_bitexact(fitted,policy):
    model=fitted[policy]['new_tangent']
    for old in ('old29_tangent','old30_tangent'):
        for key in ('gamma','training_NLL','optimizer_searches','selected_optimizer_start_index','fit_ids'):
            assert model[key]==fitted[policy][old][key]
    assert len(model['optimizer_searches'])==3
    assert model['tangent_axis']=='SELECTED_ACTUAL_MEASURED_LINE_TANGENT'


@pytest.mark.parametrize('policy',M.SOURCE_POLICIES)
def test_one_final_digest_covers_all_numerics_newly_fitted_flags_and_exports(fitted,policy):
    for kind in ('line','tangent'):
        model=fitted[policy]['new_'+kind]
        assert model['model_digest_sha256']==M._digest({k:v for k,v in model.items()if k!='model_digest_sha256'})
        assert model['newly_fitted_selection_population']is True
        assert model['prior_V29_fitted_model_relabelled']is model['prior_V30_fitted_model_relabelled']is False
        assert len([k for k in model if k=='model_digest_sha256'])==1
    result=M.mixture_noise_parameters(fitted[policy]['new_line'],source_policy=policy)
    provenance=result['noise_provenance']
    for key,value in [('observation_family',M.OBSERVATION_FAMILY),('selection_profile',M.SELECTION_PROFILE),
            ('raw_observation_generation',M.RAW_OBSERVATION_GENERATION),('noise_population',M.POPULATION),('source_policy',policy)]:
        assert provenance[key]==value
    assert result['normal_angle_correlation']==0.
    tangent=M.calibrated_archived_tangent_variance(fitted[policy]['new_tangent'],[0.,1.],[[4.,1.],[1.,9.]],source_policy=policy)
    assert tangent['tangent_variance_px2']==1.+9.*fitted[policy]['new_tangent']['gamma']
    assert tangent['old_x4_engineering_control_variance_px2']==36.
    assert tangent['selection_profile']==M.SELECTION_PROFILE and tangent['source_policy']==policy


def test_old_V29_V30_models_wrong_policy_profile_and_numeric_mutation_rejected(fitted):
    policy='ALL_BI_REPLACE';models=fitted[policy]
    for fn,kind in ((M.mixture_noise_parameters,'line'),(M.tangent_variance_parameters,'tangent')):
        for version in ('old29','old30'):
            with pytest.raises(ValueError,match='V31'):
                fn(models[version+'_'+kind],source_policy=policy)
        with pytest.raises(ValueError,match='V31'):
            fn(models['new_'+kind],source_policy='RGB_NONFALLBACK_PRESERVE')
        for field,value in [('selection_profile',OLD30.SELECTION_PROFILE),('observation_generation',OLD30.OBSERVATION_FAMILY),
                ('noise_population',OLD30.POPULATION),('raw_observation_generation','wrong'),
                ('newly_fitted_selection_population',False),('prior_V30_fitted_model_relabelled',True)]:
            changed=deepcopy(models['new_'+kind]);changed[field]=value
            with pytest.raises(ValueError):fn(changed,source_policy=policy)
    changed=deepcopy(models['new_line']);changed['normal_variance_component_px2']+=1.
    with pytest.raises(ValueError):M.mixture_noise_parameters(changed,source_policy=policy)


@pytest.mark.parametrize('field,value', [('selection_profile',OLD30.SELECTION_PROFILE),
    ('noise_population',OLD30.POPULATION),('observation_generation',M.PARENT.OBSERVATION_FAMILY),
    ('raw_observation_generation',M.OBSERVATION_FAMILY),('source_policy','RGB_NONFALLBACK_PRESERVE'),('partition','TEST')])
def test_all_row_tags_checked_before_numeric_payload(field,value):
    class Poison(dict):
        def __getitem__(self,key):raise AssertionError('Wrong population numeric payload accessed')
    rows=records();rows[0][field]=value;rows[0]['groups']=[Poison()];rows[0]['tangents']=[Poison()]
    for fn in (M.fit_geometry_guided_line_model,M.fit_geometry_guided_tangent_model):
        with pytest.raises(ValueError,match='V31'):
            fn(rows,source_policy='ALL_BI_REPLACE')


def test_heldout_and_incompatible_payloads_excluded_and_fits_still_match_parent():
    class Poison(dict):
        def __getitem__(self,key):raise AssertionError('Heldout/incompatible numeric payload accessed')
    rows=records();rows[0]['compatible']=False
    for i in (0,2,3):rows[i]['groups']=[Poison()];rows[i]['tangents']=[Poison()]
    old=FIXTURE.records();old[0]['compatible']=False
    for i in (0,2,3):old[i]['groups']=[Poison()];old[i]['tangents']=[Poison()]
    for fn,parent,key in ((M.fit_geometry_guided_line_model,M.PARENT.fit_geometry_guided_line_model,'training_mixture_nll'),
            (M.fit_geometry_guided_tangent_model,M.PARENT.fit_geometry_guided_tangent_model,'training_NLL')):
        result=fn(rows,source_policy='ALL_BI_REPLACE',heldout_session='b')
        expected=parent(old,source_policy='ALL_BI_REPLACE',heldout_session='b')
        assert result['fit_ids']==['1'] and result['excluded_heldout_ids']==['2','3']
        assert result[key]==expected[key] and result['optimizer_searches']==expected['optimizer_searches']


@pytest.mark.parametrize('state,hidden',[('S',False),('U',False),('O',False),('T',False),('B',True)])
def test_inherited_tangent_state_gates_precede_any_excluded_q_fields(state,hidden):
    class Record(dict):
        def __getitem__(self,key):raise AssertionError('Excluded q/covariance accessed')
    assert M.prepared_tangent(Record(native_corner=0),[state]*8,[hidden]*8)is None


def test_unknown_source_policy_rejected_before_record_iteration():
    class Records:
        def __iter__(self):raise AssertionError('Wrong-policy data inspected')
    for fn in (M.fit_geometry_guided_line_model,M.fit_geometry_guided_tangent_model):
        with pytest.raises(ValueError,match='V31 source policy'):
            fn(Records(),source_policy='not-declared')


@pytest.mark.parametrize('policy',M.SOURCE_POLICIES)
def test_real_exporter_kwargs_connect_to_AUTH31_core_with_floors_once(fitted,policy):
    from . import geometry_guided_edge_authentication_v4_31 as AUTH
    from .test_geometry_guided_edge_authentication_v4_31 import fixture as auth_fixture
    base,observation,kwargs,_=auth_fixture(partial=True)
    noise=M.mixture_noise_parameters(fitted[policy]['new_line'],source_policy=policy)
    arguments={**kwargs,**noise,'expected_source_policy':policy,'calibration_preparation':False}
    bundle=AUTH.authenticate_geometry_guided_edges(base,observation,**arguments)
    assert bundle['groups']
    for group in bundle['groups']:
        assert group['noise_provenance']['noise_population']==M.POPULATION
        assert group['noise_provenance']['source_policy']==policy
        assert group['normal_variance_px2']==noise['normal_variance_component_px2']+group['normal_floor_px']**2
        assert group['angular_variance_rad2']==noise['angular_variance_component_sine2']+group['angular_sine_floor']**2
    tangent=M.tangent_variance_parameters(fitted[policy]['new_tangent'],source_policy=policy)
    problem=M.CORE.GeometryGuidedEdgeProblem(base,bundle,source_policy=policy,tangent_gamma=tangent['gamma'])
    assert problem.plan['selection_profile']==M.SELECTION_PROFILE
    assert problem.plan['original_BI_query_and_native_ray_association_recomputed']
    wrong=next(p for p in M.SOURCE_POLICIES if p!=policy)
    with pytest.raises(ValueError):
        AUTH.authenticate_geometry_guided_edges(base,observation,**{**arguments,'expected_source_policy':wrong})
