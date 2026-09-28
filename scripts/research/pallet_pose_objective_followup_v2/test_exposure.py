import numpy as np
from .occlusion import random_plan
from .exposure_student import frequent_plan

def test_only_gate_changes():
    points=np.array([[200,200],[400,200],[400,400],[200,400],[220,220],[380,220],[380,380],[220,380],[300,300]])
    mask=np.ones(9,bool); extra=0
    for seed in range(300):
        a=random_plan(points,mask,[200,200,400,400],(640,640),seed)
        b=frequent_plan(points,mask,[200,200,400,400],(640,640),seed)
        for k in ('size','fill_seed','area_fraction','aspect'): assert a[k]==b[k]
        if a['scheduled']:
            assert {k:v for k,v in b.items() if k!='previous_scheduled'}==a
        extra+=b['applied'] and not a['applied']
    assert extra>20
