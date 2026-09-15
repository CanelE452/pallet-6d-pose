"""Disposable full-batch resource test; at most one nominal16 optimizer update."""
import gc,time
import numpy as np
import torch
from env import *
from prior_model import PoseFixPallet9,TFAdam
from prior_data import SourceRGB
def run():
    if complete('SMOKE_COMPLETE'):return
    before=gpu();assert not before['foreign_compute'],before
    assert torch.cuda.is_available(),'CUDA unavailable; no CPU full training fallback'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(4)
    start=now();src=SourceRGB();rows=src.data.train_rows[:16];cpu=src.batch(rows,'cpu');attempts=[]
    chosen=None
    for micro in (16,8,4):
        model=PoseFixPallet9();model.load_tf(RAW/'tf_initial_seed1.npz');model.cuda().train();opt=TFAdam(model.parameters());torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        try:
            opt.zero_grad(set_to_none=True)
            for j in range(0,16,micro):
                b={k:v[j:j+micro].cuda() for k,v in cpu.items()};logits=model(b['rgb'],b['points'],b['valid']);loss,parts=model.losses(logits,b['target'],b['target_valid']);assert torch.isfinite(loss)
                (loss*(micro/16)).backward()
                del logits,loss,b
            grad=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)).item());assert np.isfinite(grad) and grad>0
            old=model.out.weight.detach().clone();opt.step();change=float((model.out.weight-old).norm().item());assert change>0
            torch.cuda.synchronize();seconds=time.perf_counter()-t;chosen=micro
            attempts.append(dict(microbatch=micro,status='PASS',nominal_batch=16,seconds_per_update=seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,gradient_norm=grad,head_update_norm=change,BN_updates=int(model.stem.bn.num_batches_tracked)))
            assert all(int(m.num_batches_tracked)==16//micro for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d))
            break
        except torch.cuda.OutOfMemoryError as e:
            attempts.append(dict(microbatch=micro,status='CUDA_OOM',error=str(e),optimizer_updates=0))
        finally:
            del model,opt;gc.collect();torch.cuda.empty_cache()
    assert chosen is not None,attempts
    freeze(DOC/'RESOURCE_AMENDMENT.json',dict(nominal_batch=16,microbatch=chosen,gradient_accumulation=16//chosen,BN_equivalent_to_batch16=chosen==16,BN_policy='train-mode BN per physical microbatch; one running-stat update per forward; explicit resource amendment if micro<16',activation_checkpointing=False,AMP=False,TF32=False,clipping='NOT_USED',attempts=attempts,before=before,after=gpu(),estimated_18000_update_hours=attempts[-1]['seconds_per_update']*18000/3600,estimate_scope='single cold actual update including transfer; not a completion guarantee'))
    receipt('SMOKE_COMPLETE',[HERE/'prior_model.py',HERE/'resource_probe.py',RAW/'tf_initial_seed1.npz'],[DOC/'RESOURCE_AMENDMENT.json'],start,disposable_optimizer_updates=1,main_state_reused=False,seed_restarts=0)
    print('GPU_RESOURCE_READY',attempts[-1],flush=True)
