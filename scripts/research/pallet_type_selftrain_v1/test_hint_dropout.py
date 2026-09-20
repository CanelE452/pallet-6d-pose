import numpy as np
import pytest
import torch
from .hint_dropout import mask_one,without_own_hint,predict_crop
from .test_recovery_pseudo_denoise import fixture
from scripts.research.pallet_sensors_submission_v1.prior_model import gaussian


def test_mask_preserves_points_targets_and_supervision():
    x=fixture();p=x['points'].copy();t=x['target'].copy();v=x['valid'].copy();s=x['target_valid'].copy()
    y=mask_one(x,np.random.default_rng(1),force=True)
    hidden=v&~y['valid'];assert hidden.sum()==1 and not hidden[[2,8]].any()
    for key,old in [('points',p),('target',t),('target_valid',s)]:
        np.testing.assert_array_equal(x[key],old);np.testing.assert_array_equal(y[key],old)
    np.testing.assert_array_equal(x['valid'],v)
    assert y['target_valid'][hidden].all()
    a=gaussian(torch.tensor(y['points'])[None],torch.tensor(y['valid'])[None])
    assert torch.equal(a[:,hidden],torch.zeros_like(a[:,hidden]))


def test_mask_is_deterministic_balanced_no_eligible_is_noop():
    x=fixture();rng=np.random.default_rng(77)
    counts=[int((x['valid']&~mask_one(x,rng)['valid']).sum()) for _ in range(1000)]
    assert set(counts)=={0,1} and 430<sum(counts)<570
    a=mask_one(x,np.random.default_rng(4),True);b=mask_one(x,np.random.default_rng(4),True)
    np.testing.assert_array_equal(a['valid'],b['valid'])
    x['target_valid'][:]=False
    np.testing.assert_array_equal(mask_one(x,rng,True)['valid'],x['valid'])


def test_blind_inference_masks_exactly_one_hint_and_keeps_center():
    valid=torch.ones(2,9,dtype=torch.bool);valid[0,6]=False
    v=without_own_hint(valid,3)
    assert not v[:,3].any() and v[:,8].all() and valid[:,3].all()
    torch.testing.assert_close(v[:,[0,1,2,4,5,6,7,8]],valid[:,[0,1,2,4,5,6,7,8]])
    with pytest.raises(ValueError):without_own_hint(valid,8)


def test_each_output_comes_from_own_blind_pass_not_other_channel():
    class Fake:
        def __init__(self):self.calls=[]
        def __call__(self,rgb,points,valid):
            self.calls.append(valid.clone());q=torch.full((len(points),9,96,72),-100.)
            for j in range(9):
                for b in range(len(points)):q[b,j,10,20 if valid[b,j] else 30]=0
            return q
    model=Fake();points=torch.zeros(1,9,2);points[:,8]=torch.tensor([12.,13.]);v=torch.ones(1,9,dtype=torch.bool)
    out=predict_crop(model,torch.zeros(1,3,384,288),points,v,blind=True)
    assert len(model.calls)==8
    for j,mask in enumerate(model.calls):
        assert (~mask).sum()==1 and not mask[0,j]
    torch.testing.assert_close(out[:,:8],torch.tensor([[[120.,40.]]]).expand(1,8,2))
    torch.testing.assert_close(out[:,8],points[:,8]);assert v.all()
