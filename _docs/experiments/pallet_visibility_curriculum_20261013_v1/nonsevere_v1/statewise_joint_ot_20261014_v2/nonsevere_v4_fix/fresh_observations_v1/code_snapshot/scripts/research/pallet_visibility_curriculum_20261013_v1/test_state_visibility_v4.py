import numpy as np
from .state_visibility_v2 import StateVisibilityModel
from .state_visibility_v4 import geometry_state_gates_v4, infer_state_visibility_v4


def fixture():
    image=np.zeros((60,80,3),np.uint8)
    q=np.array([[20.,20.],[30.,20.],[90.,20.],[40.,40.],
                [10.,50.],[25.,30.],[20.,10.],[30.,30.],[40.,30.]])
    hidden=np.array([True,False,False,False,False,False,False,False])
    projected=q[:8].copy();projected[[0,1,2,3],0]=90.
    roles=['I','B','B','I','B','B','I','B']
    return image,q,hidden,roles,projected


def test_available_nonself_query_survives_coarse_T_but_S_and_certified_T_keep_precedence():
    image,q,hidden,roles,projected=fixture()
    evidence={3:dict(independent_of_initial_pose=True,source='independent_fixture',
                    outside_canvas_side='right')}
    records,ids=geometry_state_gates_v4(image,q,hidden,roles,projected,
                                       independent_truncation=evidence)
    assert records[0]['state']=='S' and 0 not in ids
    assert 1 in ids and records[1]['coarse_T_overridden_by_available_image_query']
    assert records[2]['state']=='T' and 2 not in ids
    assert records[3]['state']=='T' and records[3]['physical_truncation_certified'] and 3 not in ids
    assert np.array_equal(q[0],[20.,20.])


def test_direct_inference_erases_S_before_any_injected_feature_code_and_is_invariant():
    image,q,hidden,roles,projected=fixture()
    model=StateVisibilityModel(dict(feature_family='TEST_ONLY',visible_threshold=.8,
        abstention_margin=.05,external_O_enabled=True),{},None)
    def features(image,raw,ids,radius,backbone,**kwargs):
        assert np.array_equal(raw[0],[-1.,-1.]) and 0 not in ids
        return ids[:,None].astype(float),np.zeros((len(ids),1))
    def predict(local,spatial,model):
        return np.where(local[:,0]==4,.1,.99)
    arguments=dict(pallet_mask=np.ones(image.shape[:2]),feature_extractor=features,
                   probability_predictor=predict)
    result=infer_state_visibility_v4(image,q,hidden,roles,projected,model,**arguments)
    assert result['states'][1]=='B' and result['states'][4]=='O'
    changed=q.copy();changed[0]=[1e8,-1e9]
    again=infer_state_visibility_v4(image,changed,hidden,roles,projected,model,**arguments)
    assert result==again and np.array_equal(q[0],[20.,20.])
