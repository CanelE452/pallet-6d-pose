import pytest,torch
from torch import nn
from pointline_v4.adapter_helpers import capture_native_features,find_unique_hough_module,explicit_feature_affine,content_mask_from_raw_rectangle,from_verified_v2_batch
from conftest import make_observation


class HoughFeatureFusion(nn.Module):
    def forward(self,x):return [v+1 for v in x]


def test_feature_capture_no_forward_of_its_own_and_hooks_removed():
    m=HoughFeatureFusion();features=[torch.randn(1,2,5,5),torch.randn(1,3,3,3),torch.randn(1,4,2,2)]
    with capture_native_features(m) as c:
        assert c.calls==0;m(features)
    assert c.calls==1 and torch.equal(c.features[0],features[0]) and not m._forward_pre_hooks


def test_multiple_capture_calls_rejected_and_cleaned():
    m=HoughFeatureFusion();f=[torch.ones(1,1,2,2)]*3
    with pytest.raises(ValueError,match='one actual forward'):
        with capture_native_features(m):m(f);m(f)
    assert not m._forward_pre_hooks


def test_live_tap_preserves_feature_gradient():
    m=HoughFeatureFusion();f=[torch.ones(1,1,2,2,requires_grad=True) for _ in range(3)]
    with capture_native_features(m,detach=False,cpu=False) as c:m(f)
    sum(x.sum() for x in c.features).backward();assert f[0].grad.sum()==4 and f[1].grad.sum()==4


def test_find_requires_unique_verified_class():
    with pytest.raises(ValueError):find_unique_hough_module(nn.Linear(2,2))
    m=nn.Sequential(HoughFeatureFusion());assert isinstance(find_unique_hough_module(m),HoughFeatureFusion)
    with pytest.raises(ValueError):find_unique_hough_module(nn.Sequential(HoughFeatureFusion(),HoughFeatureFusion()))


def test_explicit_affine_and_content_mask():
    a=torch.tensor([[[2.,0.,4.],[0.,2.,4.]]]);out=explicit_feature_affine(a,stride_xy=(2.,2.),index_offset_xy=(-.5,-.5))
    assert torch.equal(out,torch.tensor([[[1.,0.,1.5],[0.,1.,1.5]]]))
    mask=content_mask_from_raw_rectangle(out,torch.tensor([[3,3]]),(6,6))
    assert mask.sum()==4 and mask[0,0,2:4,2:4].all()


def test_v2_mapping_rejects_extra_gt_and_maps_actual_keys():
    o=make_observation();old=dict(p4=o.features[1],baseline_points=o.baseline,layouts=o.layouts,point_conf=o.point_conf,
        diagonal=o.diagonal,raw_to_input_affine=o.raw_to_feature[0],input_shape_hw=torch.tensor([[40,40]]),
        line_h=o.line_h,peak_logits=o.line_logits,peak_valid=o.line_valid)
    args=dict(native_features=o.features,affines=o.raw_to_feature,content_masks=o.content_valid,
              prediction_valid=o.point_valid,candidate_valid=o.candidate_valid)
    mapped=from_verified_v2_batch(old,**args);mapped.validate((2,3));assert torch.equal(mapped.layouts,o.layouts)
    with pytest.raises(ValueError):from_verified_v2_batch({**old,'gt':o.baseline},**args)
