import numpy as np
from .identity_rank import pair_errors,calibrate
from .identity_rank_features import QUARTER,HALF


def test_target_error_respects_c2_but_not_quarter_turn():
    q=np.array([[0,0],[100,0],[100,20],[0,20],[10,50],[90,50],[90,60],[10,60],[50,30]],float)
    good=pair_errors(q,q,np.ones(9,bool))
    half=pair_errors(q[HALF],q,np.ones(9,bool))
    wrong=pair_errors(q[QUARTER],q,np.ones(9,bool))
    # Wrong-class C2 members tie in this symmetric fixture; their per-corner
    # vectors may differ, while the class score must be invariant.
    np.testing.assert_allclose(good.mean(-1),half.mean(-1),atol=1e-5,rtol=0)
    assert good[0].max()==0 and good[1].min()>20
    assert wrong[1].max()==0 and wrong[0].min()>20


def test_calibration_counts_synthetic_swaps_separately_and_can_abstain():
    # One correct native candidate and one wrong quarter-assigned candidate.
    errors=np.tile(np.array([[2.]*8,[100.]*8]),(10,1,1))
    y=np.zeros(10,int);eligible=np.ones(10,bool)
    good=calibrate(np.tile([.92,.08],(10,1)),errors,y,eligible)
    assert good['threshold']==.9
    assert good['selected']['clean']['damaged']==0
    assert good['selected']['clean']['recovered']==0
    assert good['selected']['combined']['recovered']==80
    bad=calibrate(np.tile([.08,.92],(10,1)),errors,y,eligible)
    assert bad['threshold'] is None and bad['pair_accuracy']==0
