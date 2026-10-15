from copy import deepcopy
import numpy as np
import pytest
from . import calibrate_geometry_guided_edge_noise_v4_29 as D
from . import test_geometry_guided_edge_noise_calibration_v4_29 as T
M=D.M

def prepared():
 floor=M.conditional_covariance_floors(18.,1.)
 edge=dict(native_corner=0,native_neighbor=1,normal_unit_xy=[1.,0.],offset_px=-3.,support_anchor_xy=[3.,7.],actual_inlier_span_px=18.,normal_floor_px=1.,normal_variance_px2=floor[0],angular_variance_rad2=floor[1],normal_angle_correlation=0.,support_id='support',fit_variant_id='fit',component_id='support:fit',mixture_component_prior=1.,selected_chain_ref={'ray_index':0,'chain_index':1,'chain_id':'chain','guide_reference_id':'guide'})
 tangent=dict(native_corner=0,state='B',kind='ARCHIVED_Q_SELECTED_MEASURED_LINE_TANGENT',source='VISIBLE_BACKBONE_FALLBACK',archived_q_xy=[4.,9.],selected_line_tangent_unit_xy=[0.,1.],effective_fit_covariance_px2=[[4.,0.],[0.,9.]],fit_reliability_weight=.8,selected_line_ref='support:fit')
 group=dict(physical_native_edge=[0,1],components=[edge],selected_chain_ref=deepcopy(edge['selected_chain_ref']),reliability_weight=.8)
 return dict(id='x',partition='CAL',session='one',states=['B']*8,initial_geometry={'initial_self_hidden':[False]*8},observation_generation=M.OBSERVATION_FAMILY,raw_observation_generation=M.RAW_OBSERVATION_GENERATION,profiles={p:dict(groups=[deepcopy(group)],tangent_records=[deepcopy(tangent)],runtime_plan_digest='sha')for p in M.SOURCE_POLICIES})

def test_exact_selected_normal_tangent_and_input_unchanged():
 r=prepared();before=deepcopy(r);saved=D.validate_prepared_row(r)
 assert r==before
 for policy in M.SOURCE_POLICIES:
  point=saved['profiles'][policy]['tangents'][0]
  assert point['original_variance_axis_px2']==9. and point['selected_line_tangent_unit_xy']==[0.,1.]

def test_original_guide_axis_not_orthogonal_is_rejected():
 r=prepared();r['profiles']['ALL_BI_REPLACE']['tangent_records'][0]['selected_line_tangent_unit_xy']=[.6,.8]
 with pytest.raises(ValueError,match='orthogonal'):D.validate_prepared_row(r)

def test_selected_exact_ref_floors_and_duplicates_are_bound():
 for kind in ('ref','floor','duplicate'):
  r=prepared();p=r['profiles']['ALL_BI_REPLACE'];g=p['groups'][0]
  if kind=='ref':g['components'][0]['selected_chain_ref']['chain_id']='another'
  elif kind=='floor':g['components'][0]['angular_variance_rad2']*=2
  else:p['groups'].append(deepcopy(g))
  with pytest.raises(ValueError):D.validate_prepared_row(r)

class Excluded(dict):
 def __getitem__(self,key):
  if key in ('normal_unit_xy','observed_points','support_anchor_xy'):raise AssertionError('S line fields touched')
  return super().__getitem__(key)

def test_S_gate_before_line_fields():
 r=prepared();r['states'][0]='S';r['profiles']['ALL_BI_REPLACE']['groups'][0]['components'][0]=Excluded(native_corner=0,native_neighbor=1)
 with pytest.raises(ValueError,match='Excluded'):D.validate_prepared_row(r)

def test_fit_requires_explicit_go_before_io(tmp_path):
 with pytest.raises(PermissionError):D.fit(tmp_path/'absent')

def test_zero_support_missing_model_and_single_session_caveat():
 r=T.records();
 for row in r:row['groups']=[];row['tangents']=[]
 s=D._fit_or_status(r,'ALL_BI_REPLACE','tangent',{})
 assert s['model']is None and not s['old_model_substituted']
 r=T.records();s=D._fit_or_status(r,'ALL_BI_REPLACE','line',{},'b')
 assert s['status']=='FITTED_SINGLE_SESSION_DESCRIPTIVE'
 assert s['model']['fit_ids']==['0','1']

def test_LOGO_description_uses_new_axis_population_not_old_model():
 r=T.records();line=D._fit_or_status(r,'ALL_BI_REPLACE','line',{},'b');tangent=D._fit_or_status(r,'ALL_BI_REPLACE','tangent',{},'b')
 desc=D._heldout_description(r,'b',line,tangent)
 assert desc['heldout_supported_line_frames']==desc['heldout_supported_tangent_frames']==2
 assert np.isfinite(desc['heldout_line_NLL']) and np.isfinite(desc['heldout_tangent_NLL'])


def test_raw_generation_distinct_from_noise_family_is_required():
 r=prepared();r["raw_observation_generation"]=M.OBSERVATION_FAMILY
 with pytest.raises(ValueError,match="New-generation"):D.validate_prepared_row(r)
