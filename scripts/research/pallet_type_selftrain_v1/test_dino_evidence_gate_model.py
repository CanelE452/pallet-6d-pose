import numpy as np
import torch
from . import dino_evidence_gate_model as G


def test_features_finite_and_gt_free():
    torch.set_num_threads(2);z=torch.zeros(1,9,192,144);z[:,:,60,80]=10
    old=torch.full((1,9,2),100.);v=torch.ones(1,9,dtype=torch.bool)
    x,q=G.features(z,z,old,v,torch.tensor([300.]))
    assert x.shape==(1,9,16) and torch.isfinite(x).all()
    assert (x[:,:,2]>0).all() and (x[:,:,9]==0).all()
    torch.testing.assert_close(q[0,0],torch.tensor([320.,240.]),atol=.1,rtol=0)


def test_rejected_points_center_invalid_bit_exact():
    old=np.arange(18,dtype=np.float64).reshape(9,2)/7;new=old+100
    score=np.ones(9);score[1]=.1;v=np.ones(9,bool);v[2]=False
    out,mask=G.choose(old,new,score,.5,v)
    assert not mask[1] and not mask[2] and not mask[8]
    np.testing.assert_array_equal(out[~mask],old[~mask]);np.testing.assert_array_equal(out[mask],new[mask])


def test_calibration_rejects_high_scoring_bad_changes():
    s=np.array([.9,.8,.2,.1]);before=np.array([1.,2.,50.,60.]);after=np.array([100.,100.,1.,1.])
    best,rows=G.calibrate(s,before,after,np.ones(4,bool))
    assert best['accepted']==0 and best['clean']['damaged']==0


def test_calibration_accepts_separable_repair():
    s=np.array([.1,.2,.9,.95]);before=np.array([1.,2.,50.,60.]);after=np.array([100.,100.,1.,1.])
    best,_=G.calibrate(s,before,after,np.ones(4,bool))
    assert best['accepted']==2 and best['clean']['recovered']==2 and best['clean']['damaged']==0


def test_invalid_points_do_not_make_features_nan():
    z=torch.randn(1,9,192,144);old=torch.full((1,9,2),float('nan'));v=torch.zeros(1,9,dtype=torch.bool)
    x,_=G.features(z,z,old,v,torch.tensor([300.]))
    assert torch.isfinite(x).all() and not x[:,:,13].any() and not x[:,:,15].any()
