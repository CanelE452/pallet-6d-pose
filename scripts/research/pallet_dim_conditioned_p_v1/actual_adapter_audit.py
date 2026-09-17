"""Additional actual TRAIN-cache adapter audit before any performance evaluation."""
import numpy as np
import torch
import dcp_env as E
from data import PaperData
from refiner import model,forward,train_loss,local_phase

def main():
    if (E.DOC/'ACTUAL_ADAPTER_TESTS.json').exists():
        assert E.read(E.DOC/'ACTUAL_ADAPTER_TESTS.json')['PASS'];print('ACTUAL_ADAPTER_ALREADY_COMPLETE');return
    torch.set_num_threads(2);data=PaperData();config=E.read(E.DOC/'MODEL_AND_PARAMETER_AUDIT.json')['config'];a=data.arrays
    rows=data.train_rows[:4096];t={k:torch.from_numpy(np.array(a[k][rows])) for k in ['points','point_valid','gt_points','gt_valid','boxes']};diag=(t['boxes'][:,2:]-t['boxes'][:,:2]).norm(dim=-1)
    _,_,g,cost=local_phase(t['points'],t['point_valid'],t['gt_points'],t['gt_valid'],torch.from_numpy(data.side['permutations'][rows]),torch.from_numpy(data.side['group_valid'][rows]),diag)
    assert (cost.min(-1).values<=cost[:,0]+1e-7).all();changed=np.flatnonzero(g.numpy()>0);assert len(changed)
    chosen=np.array([rows[np.flatnonzero(data.side['order'][rows]==1)[0]],rows[np.flatnonzero(data.side['order'][rows]==2)[0]],rows[changed[0]]]);results={}
    for seed in [1,2,3]:
        torch.manual_seed(seed);base=model('N0_BASE_REPLAY',config);b0=data.batch(chosen,'N0_BASE_REPLAY',device='cpu');before=forward(base,b0)
        for arm in ['N2_DIM_ONLY','N3_DIM_SYM','N4_META_SYM']:
            torch.manual_seed(seed);head=model(arm,config);b=data.batch(chosen,arm,device='cpu');out=forward(head,b)
            assert torch.equal(out['logits'],before['logits']);assert torch.equal(out['points'],before['points'])
            assert all(torch.equal(v,head.state_dict()[k]) for k,v in base.state_dict().items())
            optimizer=torch.optim.AdamW(head.parameters(),lr=1e-5,weight_decay=1e-4);sym=arm!='N2_DIM_ONLY';value=train_loss(out,b,sym);value.backward()
            first=float(head.metadata_scorer[-1].weight.grad.norm());assert first>0;encoder_first=float(head.metadata_encoder[0].weight.grad.norm());assert encoder_first==0
            optimizer.step();optimizer.zero_grad(set_to_none=True);train_loss(forward(head,b),b,sym).backward();second=float(head.metadata_encoder[0].weight.grad.norm());assert second>0
            results[f'{arm}_seed{seed}']=dict(initial_logit_max_abs_delta=0,base_state_exact=True,first_scorer_gradient_norm=first,first_encoder_gradient_norm=encoder_first,second_encoder_gradient_norm=second)
    assert E.sha(E.R0)==E.R0_SHA
    E.write(E.DOC/'ACTUAL_ADAPTER_TESTS.json',dict(PASS=True,time=E.now(),actual_train_rows=chosen.tolist(),results=results,
      heldout_DEV_used=False,R0_not_in_optimizer_or_graph=True,R0_sha256=E.sha(E.R0),smoke_or_main_state_used=False,
      diagnostic_CPU_optimizer_updates=9,diagnostic_second_backward_without_update=9,discarded_models=True,
      note='Additional adapter test after main launch, before performance evaluation. Nine one-step CPU test models only; not training fits or selected checkpoints.'))
    print('ACTUAL_ADAPTER_PASS',results,flush=True)
if __name__=='__main__':main()
