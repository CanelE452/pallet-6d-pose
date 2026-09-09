from types import SimpleNamespace
import copy
import torch
from scripts.research.pallet_dht_coupling_v2.integration import build_model, DIAGNOSTIC_STEPS
from scripts.research.pallet_dht_coupling_v2.gradient_surgery import parameter_dependencies, project_from_sum


def test_frozen_sampler_and_seed_are_reused():
    from scripts.research.pallet_dht_coupling_v2 import train
    from scripts.research.pallet_dht_joint_v1 import train as original
    assert train.ManifestDataset is original.ManifestDataset
    assert train.EpochSampler is original.EpochSampler
    assert train.seed_for is original.seed_for
    assert DIAGNOSTIC_STEPS == (2,16,256,1750,3499,3500,5249,6998)
    for seed in [1,2,3]:
        assert list(train.EpochSampler(32,seed,True)) == list(original.EpochSampler(32,seed,True))


def test_actual_architecture_gradient_capture_and_balanced_schedule():
    torch.set_num_threads(1)
    torch.manual_seed(623)
    model=build_model('balanced_pcgrad').cpu().train()
    model.args.epochs=2
    # Open the residual on this generated-tensor test so the point objective
    # actually reaches the line head/reduce instead of only output projections.
    with torch.no_grad():
        for layer in model.model[-1].hough.outputs:
            layer.weight.fill_(.001)
    model.criterion=model.init_criterion()
    criterion=model.criterion
    assert abs(criterion.stock.o2m-.8)<1e-12
    image=torch.rand(2,3,64,64)
    points=torch.tensor([[.25,.25,1],[.75,.25,1],[.75,.75,1],[.25,.75,1],
                         [.3,.3,1],[.7,.3,1],[.7,.7,1],[.3,.7,1],[.5,.5,1]],dtype=torch.float32)
    batch={'img':image,'batch_idx':torch.tensor([0.,1.]),'cls':torch.zeros(2,1),
           'bboxes':torch.tensor([[.5,.5,.6,.6],[.5,.5,.6,.6]]),'keypoints':points.unsqueeze(0).repeat(2,1,1)}
    pred=model(image)
    criterion.capture_gradients=True;criterion.next_step=2
    losses,items=criterion(pred,batch)
    assert losses.shape==items.shape==(8,) and torch.isfinite(losses).all()
    assert criterion.last_scalars['line_coefficient']==.1
    pending=criterion.pending
    # Differentiate the pre-concatenation task scalars. Slicing a concatenated
    # loss vector creates artificial zero-grad graph edges to the other task.
    reference_a=torch.autograd.grad(pending['task_a'],criterion.parameters,retain_graph=True,allow_unused=True)
    reference_b=torch.autograd.grad(pending['task_b'],criterion.parameters,retain_graph=True,allow_unused=True)
    assert pending['main_present']==[g is not None for g in reference_a]
    for actual,reference in zip(pending['auxiliary'],reference_b):
        assert (actual is None)==(reference is None)
        if actual is not None:torch.testing.assert_close(actual,reference,atol=0,rtol=0)
    losses.sum().backward()
    total=[p.grad for p in criterion.parameters]
    _,stats,mask,derived=project_from_sum(total,pending['auxiliary'],pending['main_present'])
    for a,b,keep in zip(derived,reference_a,mask):
        if keep:torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)
    assert any(mask)
    assert all(not keep for n,keep in zip(criterion.names,mask) if '.hough.outputs.' in n or '.hough.spatial_mix.' in n)
    criterion.pending=None;criterion.update()
    assert abs(criterion.stock.o2m-.1)<1e-12
    assert abs(criterion.weight*criterion.stock.o2m/.8-.0125)<1e-12
    # A final checkpoint's class must remain importable via the package path.
    model.criterion=None
    assert type(model).__module__=='scripts.research.pallet_dht_coupling_v2.integration'


def test_initial_forward_same_for_all_arms():
    torch.set_num_threads(1)
    image=torch.linspace(0,1,3*64*64).reshape(1,3,64,64)
    reference=None
    for arm in ['balanced','pcgrad','balanced_pcgrad','incidence']:
        torch.manual_seed(17)
        model=build_model(arm).eval()
        with torch.no_grad():output=model(image)[0]
        if reference is None:reference=output
        else:torch.testing.assert_close(output,reference,atol=0,rtol=0)


def test_incidence_reuses_unchanged_stock_assignment_and_loss():
    from ultralytics.utils.loss import E2ELoss, PoseLoss26
    torch.set_num_threads(1);torch.manual_seed(741)
    model=build_model('incidence').train();model.args.epochs=2
    image=torch.rand(2,3,64,64)
    points=torch.tensor([[.25,.25,1],[.75,.25,1],[.75,.75,1],[.25,.75,1],
                         [.3,.3,1],[.7,.3,1],[.7,.7,1],[.3,.7,1],[.5,.5,1]],dtype=torch.float32)
    batch={'img':image,'batch_idx':torch.tensor([0.,1.]),'cls':torch.zeros(2,1),
           'bboxes':torch.tensor([[.5,.5,.6,.6],[.5,.5,.6,.6]]),'keypoints':points.unsqueeze(0).repeat(2,1,1)}
    pred=model(image)
    stock_loss,stock_items=E2ELoss(model,PoseLoss26)(pred,batch)
    criterion=model.init_criterion()
    loss,items=criterion(pred,batch)
    torch.testing.assert_close(loss[:6],stock_loss,atol=0,rtol=0)
    torch.testing.assert_close(items[:6],stock_items,atol=0,rtol=0)
    assert loss.shape==(8,) and torch.isfinite(loss).all()
    assert criterion.stock.one2many.incidence_assignment is None
    assert criterion.stock.one2one.incidence_assignment is None
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_ema_criterion_can_have_no_trainable_parameters():
    torch.set_num_threads(1)
    model=build_model('balanced').eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    criterion=model.init_criterion()
    assert criterion.parameters==() and criterion.names==()
    assert not criterion.capture_gradients


def test_fresh_trainer_resume_callback_accepts_none():
    from scripts.research.pallet_dht_coupling_v2.train import CouplingTrainer
    trainer=CouplingTrainer.__new__(CouplingTrainer)
    trainer.resume_path=None
    trainer.resume_training(None)
