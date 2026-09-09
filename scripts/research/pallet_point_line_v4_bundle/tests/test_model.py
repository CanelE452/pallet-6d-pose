from dataclasses import fields
import pytest,torch
from pointline_v4.model import EvidenceVerifier,Observation,ARMS
from pointline_v4.objective import candidate_targets,quality_loss
from conftest import make_observation,clone_observation


def model(arm):
    torch.manual_seed(7)
    return EvidenceVerifier(arm,channels=(2,3),visual=4,width=16,layers=1).eval()


@pytest.mark.parametrize('arm',ARMS)
def test_all_arms_forward_and_loss_gradient(arm):
    o=make_observation(batch=2,requires_grad=True);m=model(arm);c,p=m(o)
    assert c.shape==(2,4) and p.shape==(2,4,8)
    target=candidate_targets(o.layouts,o.baseline+.3,o.point_valid,o.diagonal)
    loss,_=quality_loss(c,p,target,o.candidate_valid);loss.backward()
    assert m.quality[-1].weight.grad.abs().sum()>0
    assert all(x.grad.abs().sum()>0 for x in o.features)


def test_same_registered_state_shapes_and_initial_weights():
    models=[model(a) for a in ARMS];states=[m.state_dict() for m in models]
    assert len({sum(p.numel() for p in m.parameters()) for m in models})==1
    for state in states[1:]:
        assert all(torch.equal(state[k],states[0][k]) for k in state)


def test_no_gt_fields_in_inference_contract():
    o=make_observation();d={f.name:getattr(o,f.name) for f in fields(o)};d['gt']=o.baseline
    with pytest.raises(ValueError):Observation.from_mapping(d)
    with pytest.raises(TypeError):model('H')(d)


@pytest.mark.parametrize('arm',('P','S','H'))
def test_anchor_free_scores_invariant_to_baseline_id_order(arm):
    o=make_observation();n=clone_observation(o);order=torch.tensor([4,5,6,7,0,1,2,3,8])
    n.baseline=n.baseline[:,order];n.point_conf=n.point_conf[:,order];n.point_valid=n.point_valid[:,order]
    n.layouts[:,0]=n.baseline
    # Fixed candidate layouts1..K: score cannot depend on baseline slot IDs.
    assert torch.allclose(model(arm)(o)[0][:,1:],model(arm)(n)[0][:,1:],atol=2e-6,rtol=0)


def test_same_id_control_is_sensitive_to_baseline_id_order():
    o=make_observation();n=clone_observation(o);order=torch.tensor([4,5,6,7,0,1,2,3,8])
    n.baseline=n.baseline[:,order];n.point_conf=n.point_conf[:,order];n.layouts[:,0]=n.baseline
    assert not torch.allclose(model('HA')(o)[0][:,1:],model('HA')(n)[0][:,1:],atol=1e-7,rtol=0)


@pytest.mark.parametrize('arm',ARMS)
def test_candidate_order_equivariance(arm):
    o=make_observation();n=clone_observation(o);order=torch.tensor([0,3,1,2])
    n.layouts=n.layouts[:,order];n.candidate_valid=n.candidate_valid[:,order]
    assert torch.allclose(model(arm)(o)[0][:,order],model(arm)(n)[0],atol=2e-6,rtol=0)


@pytest.mark.parametrize('arm',('P','S'))
def test_point_segment_controls_ignore_explicit_hough(arm):
    o=make_observation();n=clone_observation(o);n.line_h[:,:,0,2]+=25;n.line_logits+=7
    assert torch.equal(model(arm)(o)[0],model(arm)(n)[0])


def test_hough_arm_reads_explicit_line_evidence():
    o=make_observation();n=clone_observation(o);n.line_h[...,2]+=20
    assert not torch.allclose(model('H')(o)[0],model('H')(n)[0],atol=1e-7,rtol=0)


def test_point_control_does_not_query_disjoint_edge_interior():
    o=make_observation(candidates=1);n=clone_observation(o);n.features[0][:,:,12:15,18:21]+=50
    assert torch.equal(model('P')(o)[0],model('P')(n)[0])
    assert not torch.allclose(model('S')(o)[0],model('S')(n)[0],atol=1e-7,rtol=0)


def test_missing_prediction_only_identity_allowed():
    o=make_observation();o.point_valid[:,3]=False
    with pytest.raises(ValueError):model('H')(o)
    o.candidate_valid[:,1:]=False;assert torch.isfinite(model('H')(o)[0]).all()


def test_centroid_or_identity_mutation_rejected():
    o=make_observation();o.layouts[:,1,8,0]+=1
    with pytest.raises(ValueError):model('H')(o)
    o=make_observation();o.layouts[:,0,0,0]+=1
    with pytest.raises(ValueError):model('H')(o)


def test_mixed_precision_export_rejected_explicitly():
    o=make_observation();o.raw_to_feature=(o.raw_to_feature[0].double(),o.raw_to_feature[1])
    with pytest.raises(TypeError,match='FP32'):model('H')(o)
