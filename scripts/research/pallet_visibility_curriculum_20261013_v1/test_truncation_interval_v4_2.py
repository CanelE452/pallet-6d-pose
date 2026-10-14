"""Pure admission and actual synthetic-image observation contract canaries."""
from copy import deepcopy

import numpy as np
import pytest

from .truncation_interval_v4_2 import (fit_truncation_interval_model,
    load_truncation_interval_model, evaluate_interval_coverage,
    disk_canvas_relation, apply_truncation_interval)
from .statewise_interval_api_v4_2 import StatewiseIntervalV4_2
from .statewise_bias_api_v4_1 import StatewiseBiasV4_1


class GatedCoordinates:
    def __init__(self, values, excluded):
        self.values, self.excluded, self.accessed = values, set(excluded), []
    def __getitem__(self, k):
        if k in self.excluded: raise AssertionError('Excluded coordinate accessed')
        self.accessed.append(k); return self.values[k]


def cal_frame(fid='a', session='A', error=10., scale=100.):
    return dict(id=fid,partition='CAL',session=session,physical_native_compatible=True,
        initial_self_hidden=[True,False,False,False,False,False,False,False],
        human_states_native=['DIRECT_VISIBLE','DIRECT_VISIBLE','EXTERNAL_OCCLUDED',
            'SELF_OCCLUDED','UNANNOTATED','DIRECT_VISIBLE','DIRECT_VISIBLE','DIRECT_VISIBLE'],
        valid_native=[True,True,True,True,False,False,False,False],
        raw_points=[[0.,0.]]*8,gt_native=[[0.,0.],[error,0.],[0.,error/2]]+[[0.,0.]]*5,
        nonself_prediction_scale_px=scale)


def interval_model(ratio=.1):
    return fit_truncation_interval_model([cal_frame(error=ratio*100.)])


def test_all_valid_direct_and_external_untrimmed_max_and_early_exclusion():
    a=cal_frame();a['raw_points']=GatedCoordinates(a['raw_points'],[0,3,4,5,6,7])
    a['gt_native']=GatedCoordinates(a['gt_native'],[0,3,4,5,6,7])
    b=cal_frame('b','B',error=54.4750754221928)
    model=fit_truncation_interval_model([a,b])
    assert model['radius_ratio']==pytest.approx(.544750754221928)
    assert model['eligible_point_count']==4 and model['max_error_point']['id']=='b'
    assert a['raw_points'].accessed==[1,2] and a['gt_native'].accessed==[1,2]
    assert not model['initial_S_coordinates_accessed'] and not model['future_coverage_guarantee']
    assert 'target_xy' not in model['max_error_point']
    assert evaluate_interval_coverage(model,[a,b])['covered_points']==4


def test_LOGO_and_incompatible_rows_skip_before_any_coordinate_access():
    a=cal_frame();b=cal_frame('b','B',error=40.)
    bad=cal_frame('bad','C');bad['physical_native_compatible']=False
    bad['raw_points']=GatedCoordinates([],range(8));bad['gt_native']=GatedCoordinates([],range(8))
    m=fit_truncation_interval_model([a,b,bad],heldout_session='B')
    assert m['radius_ratio']==.1 and m['fit_ids']==['a','bad']
    assert m['compatible_ids']==['a'] and m['excluded_native_incompatible_ids']==['bad']
    assert evaluate_interval_coverage(m,[b])['covered_points']==0


def test_TEST_batch_rejected_before_any_coordinates_and_model_tamper_rejected():
    a=cal_frame();a['partition']='TEST';a['raw_points']=GatedCoordinates([],range(8));a['gt_native']=GatedCoordinates([],range(8))
    with pytest.raises(ValueError,match='TEST'):fit_truncation_interval_model([a])
    with pytest.raises(ValueError,match='TEST'):evaluate_interval_coverage(interval_model(),[a])
    m=interval_model();m['radius_ratio']*=.1
    with pytest.raises(ValueError,match='binding'):load_truncation_interval_model(m)


@pytest.mark.parametrize('center,radius,retained',[
    ([30.,30.],5.,False),([-5.,30.],5.,False),([-5.,30.],4.999,True),
    ([105.,30.],4.,True),([30.,-5.],4.,True),([30.,85.],4.,True),
    ([-.8,-.8],1.,True),([-.8,-.8],1.2,False),([100.,80.],0.,False),
])
def test_closed_canvas_disk_all_sides_and_Euclidean_corner(center,radius,retained):
    assert disk_canvas_relation(center,radius,[80,100])['whole_disk_outside']==retained


def gate_fixture():
    states=['S','B','I','O','T','T','T','U']
    q=GatedCoordinates([[np.nan,np.nan],[20.,20.],[30.,20.],[40.,20.],
        [101.,40.],[101.,40.],[-10.,40.],[50.,40.]],[0,1,2,3,5,7])
    records=[dict(native_id=k,state=s,physical_truncation_certified=k==5,
        outside_canvas_side='right' if k==5 else []) for k,s in enumerate(states)]
    return states,q,records


def test_conditional_T_certified_T_and_S_precedence_without_new_queries():
    states,q,records=gate_fixture();before=deepcopy(records)
    out=apply_truncation_interval(states,q,[80,100],100.,interval_model(.05),records,[True]+[False]*7)
    assert out['states']==['S','B','I','O','U','T','T','U'] and records==before
    assert q.accessed==[4,6] and out['records'][5]['outside_canvas_side']=='right'
    assert out['records'][4]['abstained'] and out['records'][4]['reason']=='CAL_localization_disk_intersects_canvas'
    assert out['audit']['demoted_T_to_U_ids']==[4] and out['audit']['RGB_reads']==0
    assert not out['audit']['S_raw_coordinates_accessed']


def test_missing_conditional_prediction_cannot_admit_hard_T():
    states,q,records=gate_fixture();q.values[4]=[-1.,-1.]
    out=apply_truncation_interval(states,q,[80,100],100.,interval_model(.05),records,[True]+[False]*7)
    assert out['states'][4]=='U' and out['audit']['records'][0]['raw_prediction_missing']


def image_fixture():
    from .test_statewise_joint_api_v4 import fixture
    from .state_visibility_v2 import StateVisibilityModel
    from .state_visibility_v4 import infer_state_visibility_v4
    from .ot_domains_v2 import fit_external_o_error_model
    base,image,q,K,xyz,coarse,uncertainty=fixture()
    image=image[:,:370].copy()
    classifier=StateVisibilityModel(dict(feature_family='TEST_ONLY',visible_threshold=.8,
        abstention_margin=.05,external_O_enabled=True),{},None)
    calls=[]
    def producer(image,raw,hidden,roles,projected,model,**kwargs):
        def features(image,raw,ids,radius,backbone,**kwargs):
            assert all(not hidden[k] and 0<=raw[k,0]<image.shape[1] and 0<=raw[k,1]<image.shape[0] for k in ids)
            calls.append(ids.tolist())
            return ids[:,None].astype(float),np.zeros((len(ids),1))
        return infer_state_visibility_v4(image,raw,hidden,roles,projected,classifier,
            feature_extractor=features,
            probability_predictor=lambda local,spatial,model:np.where(local[:,0]==min(local[:,0]),.1,.99),**kwargs)
    ot=fit_external_o_error_model([dict(split='CAL',state='O',external_label_confirmed=True,
        label_source='HUMAN',raw_xy=[10.,10.],target_xy=[12.,10.],image_hw=[80,100],
        scale_px=100.,native_corner=0,record_id='synthetic',session='synthetic')])
    kwargs=dict(ot_model=ot,uncertainty_model=uncertainty,state_producer=producer,config=deepcopy(base.config))
    old=StatewiseBiasV4_1(**kwargs)
    new=StatewiseIntervalV4_2(truncation_model=interval_model(.6),**kwargs)
    return old,new,image,q,K,xyz,coarse,calls


def test_actual_synthetic_RGB_observe_rebuilds_domains_preserves_BI_Q_ninth_and_no_OOB_query(monkeypatch):
    import cv2
    old,new,image,q,K,xyz,coarse,calls=image_fixture()
    real=cv2.getRectSubPix
    def guarded(array,size,center,*args,**kwargs):
        assert 0<=center[0]<image.shape[1] and 0<=center[1]<image.shape[0]
        return real(array,size,center,*args,**kwargs)
    monkeypatch.setattr(cv2,'getRectSubPix',guarded)
    kwargs=dict(pallet_mask=np.ones(image.shape[:2]))
    a=old.observe(image,q,K,xyz,coarse,**kwargs);b=new.observe(image,q,K,xyz,coarse,**kwargs)
    assert 'T' in a['states'] and 'O' in a['states'] and b['truncation_interval_guard']['demoted_T_to_U_ids']
    for key in ['initial_geometry','frozen_visible_measurements','visible_correction','mask_audit','original_coarse_pose','original_raw_points']:
        assert a[key]==b[key],key
    assert a['original_raw_points'][8]==q[8].tolist()
    assert np.allclose(a['initial_geometry']['physical_registration_Q'],np.asarray(coarse['R_cf']).T@np.asarray(coarse['R_physical']))
    assert a['ot_domains']['nonself_prediction_scale_px']==b['ot_domains']['nonself_prediction_scale_px']
    assert b['ot_domains']['domains_sha256']!=a['ot_domains']['domains_sha256']
    assert b['truncation_interval_guard']['O_numeric_geometry_unchanged']
    cached=new.adapt_observation(image,a)
    assert cached==b and len(calls)==2
    changed=q.copy();changed[:8][np.array(a['initial_geometry']['initial_self_hidden'],bool)]=[1e9,-1e9]
    c=new.observe(image,changed,K,xyz,coarse,**kwargs)
    for key in ['states','frozen_visible_measurements','visible_correction','ot_domains','mask_audit','truncation_interval_guard']:
        assert b[key]==c[key],key


def test_solve_wrapper_only_changes_metadata_and_requires_gated_observations(monkeypatch):
    old,new,image,q,K,xyz,coarse,_=image_fixture()
    original=old.observe(image,q,K,xyz,coarse,pallet_mask=np.ones(image.shape[:2]))
    with pytest.raises(ValueError,match='observe or adapt'):new.solve_observations(original)
    observed=new.adapt_observation(image,original);frozen=deepcopy(observed);called=[]
    def unchanged_math(self,observation):
        called.append((self,deepcopy(observation),deepcopy(self.config)))
        return deepcopy(observation)|{'schema':'V4_1','method':'V4_1'}
    monkeypatch.setattr(StatewiseBiasV4_1,'solve_observations',unchanged_math)
    out=new.solve_observations(observed)
    assert len(called)==1 and called[0][0] is new and called[0][1]==frozen and observed==frozen
    assert out['frozen_visible_measurements']==observed['frozen_visible_measurements']
    assert out['truncation_model_binding']['radius_ratio']==.6
    assert not out['truncation_model_binding']['future_coverage_guarantee']
    assert out['schema']=='STATEWISE_INTERVAL_CAL_BIAS_RGB_V4_2'


def test_certified_T_side_survives_full_observation_adapter_and_changed_image_binding_rejected():
    old,new,image,q,K,xyz,coarse,_=image_fixture();kwargs=dict(pallet_mask=np.ones(image.shape[:2]))
    first=old.observe(image,q,K,xyz,coarse,**kwargs)
    k=next(i for i,s in enumerate(first['states']) if s=='T')
    cert={k:dict(independent_of_initial_pose=True,source='synthetic_independent_crop',outside_canvas_side='right')}
    result=new.observe(image,q,K,xyz,coarse,independent_truncation=cert,**kwargs)
    d=next(d for d in result['ot_domains']['domains'] if d['native_corner']==k)
    assert result['states'][k]=='T' and d['outside_canvas_side']=='right'
    changed=image.copy();changed[0,0,0]^=1
    with pytest.raises(ValueError,match='binding'):new.adapt_observation(changed,first)


def test_cached_crop_requires_unchanged_retained_T_source_validity_halfplanes():
    from .test_statewise_joint_api_v4 import fixture
    from .ot_domains_v2 import apply_real_crop
    old,new,_,q,K,xyz,coarse,_=image_fixture()
    _,source,_,_,_,_,_=fixture()
    crop=apply_real_crop(source,K,q,[0,0,370,480],source_inframe_native_ids=range(8))
    image=crop['image_bgr'];old_obs=old.observe(image,crop['raw_points'],crop['K'],xyz,coarse,
        pallet_mask=np.ones(image.shape[:2]),crop_transform=crop['transform'])
    new.truncation_model=interval_model(0.)
    assert any(d.get('source_crop_validity_intersection') for d in old_obs['ot_domains']['domains'] if d['state']=='T')
    with pytest.raises(RuntimeError,match='retained T numeric geometry'):
        new.adapt_observation(image,old_obs)
    good=new.adapt_observation(image,old_obs,crop_transform=crop['transform'])
    assert good['truncation_interval_guard']['retained_T_numeric_geometry_and_crop_halfplanes_unchanged']
    assert good['truncation_interval_guard']['retained_T_ids']
