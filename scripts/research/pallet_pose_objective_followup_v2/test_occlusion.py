import numpy as np
from .occlusion import random_plan, cover

def test_plan_shared_and_bounded():
    p=np.array([[200,200],[400,200],[400,400],[200,400],[220,220],[380,220],[380,380],[220,380],[300,300]])
    mask=np.ones(9,bool); applied=0
    for seed in range(200):
        a=random_plan(p,mask,[200,200,400,400],(640,640),seed)
        assert a==random_plan(p,mask,[200,200,400,400],(640,640),seed)
        if a['applied']:
            applied+=1
            assert 1<=len(a['covered'])<=6 and a['remaining']>=2
            assert a['bbox_fraction']<=.31 and 8 not in a['covered']
    assert applied>20

def test_empty_supervision_does_not_generate_targets():
    p=np.zeros((9,2)); mask=np.zeros(9,bool)
    for seed in range(10): assert not random_plan(p,mask,[0,0,640,640],(640,640),seed)['applied']
