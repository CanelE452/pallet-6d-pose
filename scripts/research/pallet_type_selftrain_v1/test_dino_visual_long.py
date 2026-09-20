import torch
from . import dino_visual_long as L


def test_scope_changes_only_destination_and_count():
    old=(L.D.PHASE,L.D.DOC,L.D.RAW,L.D.STEPS,L.D.M)
    with L.scope():
        assert L.D.PHASE=='dino_visual_long' and L.D.STEPS==5000 and L.D.M is L.V.V
    assert (L.D.PHASE,L.D.DOC,L.D.RAW,L.D.STEPS,L.D.M)==old


def test_adam_resume_preserves_moments_and_next_update():
    torch.manual_seed(1);a=torch.nn.Linear(4,2);opt=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.0001)
    x=torch.randn(3,4);y=torch.randn(3,2)
    def step(m,o):
        o.zero_grad();loss=(m(x)-y).square().mean();loss.backward();o.step()
    for _ in range(3):step(a,opt)
    import copy
    state=copy.deepcopy(a.state_dict());optimizer=copy.deepcopy(opt.state_dict())
    b=torch.nn.Linear(4,2);b.load_state_dict(state);other=torch.optim.AdamW(b.parameters(),lr=.001,weight_decay=.0001);other.load_state_dict(optimizer)
    step(a,opt);step(b,other)
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
