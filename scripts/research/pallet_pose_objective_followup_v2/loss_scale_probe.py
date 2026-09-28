"""CPU-only single-batch joint-gradient magnitude check; no optimizer/fit."""
import time
from pathlib import Path
import torch
from . import common as C
from .loss_signal import dataset_fixture, load_model, make_criterion, attach_signal_hooks, normalized_smooth_l1


def main():
    start=time.monotonic();torch.set_num_threads(4)
    p,batch,*_=dataset_fixture('MAIN')
    model=load_model(C.checkpoint('PLASTIC','R0'),p['args'],'cpu')
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.criterion=make_criterion(model)
    batch['img']=batch['img'].float()/255
    parameters=[p for p in model.parameters() if p.requires_grad]
    with attach_signal_hooks(model.criterion) as hook:
        predictions=model(batch['img']); loss,_=model.criterion(predictions,batch)
        original=loss.sum()/len(batch['img'])
        supplement=0.
        for name,row in hook.rows.items():
            pred,gt,mask,area=row['location_inputs'];b=row['branch']
            weight=model.criterion.o2m if name=='one2many' else model.criterion.o2o
            supplement=supplement+normalized_smooth_l1(pred,gt,mask,area,b.keypoint_loss.sigmas)*b.hyp.pose*weight
        def gradient(value):
            result=torch.autograd.grad(value,parameters,allow_unused=True,retain_graph=True)
            return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten().double()
                for p,g in zip(parameters,result)])
        g0=gradient(original);g1=gradient(supplement)
        coefficient=C.read(C.DOC/'LOSS_SIGNAL_AUDIT.json')['calibration']['coefficient']
        addon=coefficient*g1;total=g0+addon
        direct=gradient(original+coefficient*supplement)
        assert torch.allclose(direct,total,atol=1e-7,rtol=1e-5)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    result=dict(kind='CPU_FROZEN_MAIN_R0_CLEAR_BATCH4',fits=0,optimizer_updates=0,optimizer_constructed=False,
        coefficient=coefficient,original_full_objective_head_norm=float(g0.norm()),
        supplement_head_norm=float(addon.norm()),combined_head_norm=float(total.norm()),
        combined_over_original=float(total.norm()/g0.norm()),
        original_supplement_cosine=float((g0@addon)/(g0.norm()*addon.norm())),
        exact_direction_residual_ratio=float((addon-(g0@addon)/(g0@g0)*g0).norm()/addon.norm()),
        all_state_tensors_exact=len(before),direct_gradient_sum_checked=True,
        caveat='One frozen CPU TRAIN batch only; not gradient-matched fit, not constant scale across later steps. Addon is not exactly collinear with original gradient.',
        diagnostic=C.bind(C.DOC/'LOSS_SIGNAL_AUDIT.json'),implementation=C.bind(Path(__file__)),
        seconds=time.monotonic()-start)
    C.save(C.DOC/'LOSS_SCALE_PROBE.json',result,True);print(result,flush=True)


if __name__=='__main__':main()
