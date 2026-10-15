from copy import deepcopy
import numpy as np
import pytest
from . import geometry_guided_edge_noise_calibration_v4_29 as M

def records(policy='ALL_BI_REPLACE'):
 result=[]
 for i,(session,e) in enumerate([('a',2.),('a',5.),('b',8.),('b',12.)]):
  floor=M.conditional_covariance_floors(18.,.3)
  component=dict(normal_error_px=e,angular_sine_error=.02*(i+1),actual_inlier_span_px=18.,normal_floor_px=1.,conditional_variance_floor=floor.tolist(),support_id='s',fit_variant_id='f',component_id='s:f',mixture_component_prior=1.)
  tangent=dict(native_corner=0,state='B',source='VISIBLE_BACKBONE_FALLBACK',source_group='fallback',signed_tangent_error_px=e,original_variance_axis_px2=9.,exact_saved_old_tangent_variance_px2=36.)
  result.append(dict(id=str(i),partition='CAL',session=session,compatible=True,observation_generation=M.OBSERVATION_FAMILY,noise_population=M.POPULATION,source_policy=policy,groups=[dict(physical_native_edge=[0,1],components=[component])],tangents=[tangent]))
 return result

def test_line_nll_gradient_and_floor_once():
 r=records();groups=[]
 for row in r:
  c=row['groups'][0]['components'][0];groups.append([dict(error=np.array([c['normal_error_px'],c['angular_sine_error']]),floor=np.array(c['conditional_variance_floor']),mixture_component_prior=1.)])
 v=np.array([5.,.002]);w=np.full(4,.25);_,g=M.mixture_noise_nll_and_gradient(v,groups,w)
 for j,step in enumerate([1e-5,1e-8]):
  d=np.zeros(2);d[j]=step
  fd=(M.mixture_noise_nll_and_gradient(v+d,groups,w)[0]-M.mixture_noise_nll_and_gradient(v-d,groups,w)[0])/(2*step)
  assert np.isclose(fd,g[j],rtol=2e-6,atol=1e-7)
 assert np.array_equal(M.conditional_covariance_floors(18.,.3),M.LINE.conditional_covariance_floors(18.,.3))

def test_new_line_model_policy_digest_and_no_parent_mutation():
 old=deepcopy(M.LINE.RECIPE);r=records();before=deepcopy(r)
 model=M.fit_geometry_guided_line_model(r,source_policy='ALL_BI_REPLACE')
 p=M.mixture_noise_parameters(model,source_policy='ALL_BI_REPLACE')
 assert model['schema']==M.SCHEMA and p['noise_provenance']['observation_family']==M.OBSERVATION_FAMILY
 assert model['fit_frame_count']==4 and model['fit_physical_edge_count']==4 and model['fit_weight_sum']==1.
 assert M.LINE.RECIPE==old and r==before
 with pytest.raises(ValueError):M.mixture_noise_parameters(model,source_policy='RGB_NONFALLBACK_PRESERVE')
 broken=deepcopy(model);broken['normal_variance_component_px2']+=1.
 with pytest.raises(ValueError):M.mixture_noise_parameters(broken,source_policy='ALL_BI_REPLACE')

def test_new_family_old_model_and_multiple_components_rejected():
 r=records();r[0]['observation_generation']='OLD_17C'
 with pytest.raises(ValueError):M.fit_geometry_guided_line_model(r,source_policy='ALL_BI_REPLACE')
 r=records();r[0]['groups'][0]['components']*=2
 with pytest.raises(ValueError):M.fit_geometry_guided_line_model(r,source_policy='ALL_BI_REPLACE')
 r=records();r[0]['partition']='TEST'
 with pytest.raises(ValueError):M.fit_geometry_guided_line_model(r,source_policy='ALL_BI_REPLACE')

class Poison(dict):
 def __getitem__(self,key):raise AssertionError('Excluded label/group fields must not be read')

def test_heldout_and_incompatible_excluded_before_error_fields():
 r=records();r[0]['groups']=[Poison()];r[0]['compatible']=False
 r[2]['groups']=[Poison()];r[3]['groups']=[Poison()]
 model=M.fit_geometry_guided_line_model(r,source_policy='ALL_BI_REPLACE',heldout_session='b')
 assert model['fit_ids']==['1'] and model['excluded_heldout_ids']==['2','3']
 assert model['single_session_scale_caveat']

def test_tangent_new_gamma_population_marginal_and_orthogonal_axis():
 r=records();model=M.fit_geometry_guided_tangent_model(r,source_policy='ALL_BI_REPLACE')
 p=M.calibrated_archived_tangent_variance(model,[0.,1.],[[4.,1.],[1.,9.]],source_policy='ALL_BI_REPLACE')
 assert p['original_variance_axis_px2']==9. and p['tangent_variance_px2']==1.+9.*model['gamma']
 assert p['old_x4_engineering_control_variance_px2']==36.
 assert model['tangent_axis']=='SELECTED_ACTUAL_MEASURED_LINE_TANGENT'
 with pytest.raises(ValueError):M.tangent_variance_parameters(model,source_policy='RGB_NONFALLBACK_PRESERVE')

@pytest.mark.parametrize('state,hidden',[('S',False),('U',False),('O',False),('T',False),('B',True)])
def test_state_gate_before_tangent_coordinates(state,hidden):
 class Record(dict):
  def __getitem__(self,key):raise AssertionError('Excluded image-coordinate archive touched')
 rec=Record(native_corner=0)
 assert M.prepared_tangent(rec,[state]*8,[hidden]*8) is None

def test_active_tangent_archive_immutable():
 r=dict(native_corner=0,state='I',kind='ARCHIVED_Q_SELECTED_MEASURED_LINE_TANGENT',archived_q_xy=[4.,6.],selected_line_tangent_unit_xy=[0.,1.],effective_fit_covariance_px2=[[4.,1.],[1.,9.]],fit_reliability_weight=.7,source='VISIBLE_BACKBONE_FALLBACK',selected_line_ref='edge0/chain1')
 before=deepcopy(r);p=M.prepared_tangent(r,['I']*8,[False]*8)
 assert p['original_variance_axis_px2']==9. and p['selected_line_ref']=='edge0/chain1' and r==before

def test_tangent_scalar_gradient_and_exact_old4_sensitivity():
 e=np.array([1.,4.,8.]);a=np.array([2.,5.,9.]);w=np.array([.2,.3,.5]);gamma=.7
 v,g=M.scalar_tangent_nll_and_gradient(gamma,e,a,w);step=1e-5
 fd=(M.scalar_tangent_nll_and_gradient(gamma+step,e,a,w)[0]-M.scalar_tangent_nll_and_gradient(gamma-step,e,a,w)[0])/(2*step)
 assert np.isclose(g,fd,rtol=1e-7)
 assert not np.array_equal(4*a,1+4*a)
