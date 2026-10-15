import pytest
from . import diagnose_CAL_guide_anchor_association_v4_29 as P


class Poison:
    def __array__(self,*args,**kwargs):raise AssertionError('S/U coordinate read')


def test_q_distance_state_and_initial_S_gate_before_coordinates():
    visible=[dict(native_corner=0,xy=[8.,12.],source='RGB'),
        dict(native_corner=1,xy=Poison()),dict(native_corner=2,xy=Poison()),dict(native_corner=3,xy=Poison())]
    states=['B','U','S','I']+['S']*4;hidden=[False,False,True,True]+[True]*4
    result=P.q_distances([0.,1.],-10.,states,hidden,visible)
    assert len(result)==1 and result[0]['signed_normal_distance_px']==2.


def test_measured_line_sign_only_changes_signed_distance_not_absolute_reach():
    row=[dict(native_corner=0,xy=[7.,50.],source='VISIBLE_BACKBONE_FALLBACK')]
    a=P.q_distances([0.,1.],-4.,['B']+['S']*7,[False]+[True]*7,row)
    b=P.q_distances([0.,-1.],4.,['B']+['S']*7,[False]+[True]*7,row)
    assert a[0]['signed_normal_distance_px']==-b[0]['signed_normal_distance_px']==46.
