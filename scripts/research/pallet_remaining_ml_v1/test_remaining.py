import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import torch
import reweight as W

def test_meta_weights_match_exact_virtual_step_derivative():
    theta=torch.tensor([1.,-1.],requires_grad=True)
    targets=torch.tensor([[0.,0.],[2.,-2.],[2.,0.]])
    losses=((theta-targets)**2).sum(1)/2
    clean=((theta-torch.tensor([0.,0.]))**2).sum()/2
    gm=torch.autograd.grad(clean,theta,retain_graph=True)[0]
    align=torch.stack([torch.dot(torch.autograd.grad(l,theta,retain_graph=True)[0],gm) for l in losses])
    eps=torch.zeros(3,requires_grad=True)
    proxy=theta-.01*torch.autograd.grad((losses*eps).sum(),theta,create_graph=True)[0]
    val=(proxy**2).sum()/2
    exact=-torch.autograd.grad(val,eps)[0]/.01
    torch.testing.assert_close(exact,align)
    torch.testing.assert_close(W.weights('meta_weight',losses,align),torch.tensor([3.,0.,0.]))

def test_zero_alignment_fallback_and_detach():
    x=torch.tensor([2.,3.],requires_grad=True)
    assert not W.weights('meta_weight',x,x).requires_grad
    assert W.weights('meta_weight',x,-x).sum()==0
    assert W.weights('uniform',x,-x).sum()==2
    torch.testing.assert_close(W.weights('hard_loss',x,-x),torch.tensor([.8,1.2]))

def test_batch_slicing_reindexes_multiple_objects_and_nested_predictions():
    batch=dict(batch_idx=torch.tensor([0.,1.,1.,2.]),cls=torch.zeros(4,1),bboxes=torch.arange(16).reshape(4,4),
        keypoints=torch.zeros(4,9,3),img=torch.zeros(3,3,8,8))
    one=W.single_target(batch,1)
    assert one['bboxes'].shape==(2,4) and one['batch_idx'].tolist()==[0.,0.]
    pred={'one2one':{'kpts':torch.arange(12).reshape(3,4), 'feats':[torch.zeros(3,8,2,2)]}}
    sliced=W.slice_predictions(pred,1)
    assert sliced['one2one']['kpts'].tolist()==[[4,5,6,7]]
    assert sliced['one2one']['feats'][0].shape==(1,8,2,2)

def test_proxy_parameter_contract():
    params=W.proxy_parameters(W.S.load_model())
    assert len(params)==12 and sum(p.numel() for _,p in params)==7452

def test_reject_all_when_calibration_is_unsafe():
    import selective as G
    out=G.select_threshold(np.arange(12),np.ones(12,bool),np.ones(12,bool))
    assert out['threshold'] is None and out['coverage']==0 and out['risk'] is None

def test_threshold_cannot_split_tied_scores_or_accept_invalid():
    import selective as G
    score=np.zeros(12);bad=np.array([False]*10+[True]*2)
    assert G.select_threshold(score,bad,np.ones(12,bool))['threshold'] is None
    valid=np.array([True]*10+[False]*2)
    out=G.select_threshold(score,bad,valid)
    assert out['accepted']==10 and out['risk']==0

def test_aurc_ties_are_permutation_invariant():
    import selective as G
    a=G.curve([0,0,1,1],[False,True,False,True],[True]*4)
    b=G.curve([0,0,1,1],[True,False,True,False],[True]*4)
    assert a==b and a['aurc']==.5

def test_missing_pose_has_fixed_features_no_truth_input():
    import inspect
    import selective as G
    out=G.prediction_features(None,np.eye(3),(1.3,1.1,.11),640,480)
    assert not out['valid'] and len(out['features'])==38
    assert list(inspect.signature(G.prediction_features).parameters)==['pred','camera','dimensions','width','height']
