"""Matched-real-exposure source replay; no pseudo labels or final-model writes."""
import gc
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from scripts.research.pallet_posefix_replay_v1 import core as N
from scripts.research.pallet_posefix_large_error_v1 import train as O
from scripts.research.pallet_sensors_submission_v1.prior_model import TFAdam,expectation


def combined_micro_loss(parts,branch,micro=2,batch_size=8):
    # Original regularization exactly once over all real microbatches.
    loss=parts['heatmap']+parts['coordinate']
    if branch=='real':loss=loss+parts['L2']
    elif branch!='source':raise ValueError(branch)
    return loss*(micro/batch_size)


def real_rows(items,order,step):
    result=[]
    for j,row in enumerate(order['real_rows'][step]):
        x=dict(items[int(row)]);x['points']=order['real_points'][step,j].copy();x['valid']=order['real_valid'][step,j].copy()
        result.append(x)
    return result


@torch.no_grad()
def source_probe(model,items):
    model.eval(); rng=np.random.default_rng(7104);result={}
    for mode in ('clean','stress'):
        rows=items if mode=='clean' else [O.corrupted(x,rng,True) for x in items]
        initial=[];final=[]
        for start in range(0,len(rows),2):
            xs=rows[start:start+2];b=O.batch(xs)
            q=expectation(model(b['rgb'],b['points'],b['valid'])).cpu().numpy()
            for x,p in zip(xs,q):
                mask=x['target_valid'];scale=x['matrix'][0,0]
                assert abs(scale-x['matrix'][1,1])<1e-9
                initial.extend((np.linalg.norm(x['points'][mask]-x['target'][mask],axis=-1)/scale).tolist())
                final.extend((np.linalg.norm(p[mask]-x['target'][mask],axis=-1)/scale).tolist())
        row=O.error_summary(initial,final);a,b=np.asarray(initial),np.asarray(final)
        row.update(P90_px=float(np.quantile(b,.9)),good=int((a<5).sum()),damaged=int(((a<5)&(b>10)).sum()),
                   source_frames=len(rows),units='original prepared RGB pixels, not crop pixels',fixed_index=True)
        result[mode]=row
    return result


def inputs():
    from scripts.research.pallet_posefix_replay_v1.source import SourceData
    oldsupport=N.E.bound(N.C.DOC/'TRAIN_SUPPORT.json');real=O.items();N.F.verify(oldsupport)
    source=SourceData();order=dict(np.load(N.RAW/'ORDERS.npz'))
    expected=N.make_real_order(real)
    for k,v in expected.items():assert np.array_equal(order[k],v),k
    selected=np.unique(np.r_[order['source_rows'].ravel(),order['held_rows']])
    lock=N.E.read(N.DOC/'INPUT_LOCK.json')
    assert N.source_cache_signature(source.data,selected)==lock['source_cache_values']
    assert set(order['source_rows'].ravel())<=set(source.train_rows)
    assert set(order['held_rows'])<=set(source.held_rows)
    for r in lock['source_images']:N.F.verify(r['image'])
    held=[source.item(int(i)) for i in order['held_rows']]
    return real,source,order,held


def before():
    N.verify();N.setup();N.E.gpu();real,source,order,held=inputs()
    dst=N.DOC/'BEFORE.json'
    if dst.exists():print('BEFORE_ALREADY_COMPLETE',flush=True);return
    original=N.C.load_model()
    real_probe=O.probe(original,real)
    old=N.E.read(N.C.DOC/'FIT.json')['before']
    for mode in real_probe:
        for key in ('input_mean_px','output_mean_px','output_median_px','PCK10'):
            assert abs(real_probe[mode][key]-old[mode][key])<1e-3,(mode,key)
    prior=source_probe(original,held);del original;gc.collect();torch.cuda.empty_cache()
    adapted=N.C.load_finetuned();real_only=source_probe(adapted,held)
    N.freeze(dst,dict(complete=True,PRIOR_SOURCE=prior,REAL_ONLY_SOURCE=real_only,
        real_probe=real_probe,old_probe_numeric_parity_atol_px=.001,
        input_lock=N.E.bound(N.DOC/'INPUT_LOCK.json'),
        model_bindings=[N.E.bound(N.C.PRIOR_CK),N.E.read(N.C.DOC/'FIT.json')['checkpoint']]))
    print('BEFORE',N.E.read(dst),flush=True)


def run():
    N.verify();N.setup();N.E.gpu()
    if (N.DOC/'FIT.json').exists():
        N.F.verify(N.E.read(N.DOC/'FIT.json')['checkpoint']);print('TRAIN_REUSED',flush=True);return
    assert (N.DOC/'BEFORE.json').exists(),'Run before source retention probe first'
    real,source,order,held=inputs()
    torch.manual_seed(1);torch.cuda.manual_seed_all(1)
    model=N.C.load_model().requires_grad_(True)
    optimizer=TFAdam(model.parameters(),lr=1e-4)
    model.train()
    for m in model.modules():
        if isinstance(m,torch.nn.BatchNorm2d):m.eval()
    buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    gs=np.random.default_rng(7103);history=[];start_step=0;elapsed=0.
    resume=N.RAW/'resume.pt';protocol_sha=N.E.sha(N.DOC/'PROTOCOL.json')
    if resume.exists():
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==protocol_sha
        model.load_state_dict(ck['model_state_dict']);optimizer.load_state_dict(ck['optimizer_state_dict'])
        history=ck['history'];start_step=ck['step'];gs.bit_generator.state=ck['source_rng'];elapsed=ck['elapsed_seconds']
        del ck
    assert not (N.RAW/'last300.pt').exists() or start_step==300,'Final checkpoint without resumable state; inspect before proceeding'
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=[pool.submit(source.item,int(i)) for i in order['source_rows'][start_step]] if start_step<300 else []
        for idx in range(start_step,300):
            source_items=[f.result() for f in pending]
            pending=[pool.submit(source.item,int(i)) for i in order['source_rows'][idx+1]] if idx+1<300 else []
            assert all(x['partition']=='train' for x in source_items)
            rows_by_branch=dict(real=real_rows(real,order,idx),source=[O.corrupted(x,gs) for x in source_items])
            optimizer.zero_grad(set_to_none=True);details={}
            for branch,rows in rows_by_branch.items():
                parts=[]
                for j in range(0,8,2):
                    b=O.batch(rows[j:j+2]);q=model(b['rgb'],b['points'],b['valid'])
                    _,detail=model.losses(q,b['target'],b['target_valid'])
                    loss=combined_micro_loss(detail,branch);assert torch.isfinite(loss)
                    loss.backward();parts.append({k:float(v.detach()) for k,v in detail.items()})
                    del b,q,detail,loss
                details[branch]={k:float(np.mean([p[k] for p in parts])) for k in parts[0]}
            norm=float(torch.stack([p.grad.square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt())
            assert np.isfinite(norm);update=float(optimizer.step());assert np.isfinite(update) and update>0
            rec=dict(step=idx+1,gradient_norm=norm,update_norm=update,loss=details,
                     real_supervised=sum(int(x['target_valid'].sum()) for x in rows_by_branch['real']),
                     source_supervised=sum(int(x['target_valid'].sum()) for x in rows_by_branch['source']))
            history.append(rec)
            if (idx+1)%100==0:
                N.save(resume,dict(step=idx+1,protocol_sha256=protocol_sha,model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),source_rng=gs.bit_generator.state,history=history,
                    elapsed_seconds=elapsed+time.monotonic()-start))
            if idx==0 or (idx+1)%50==0:
                status=dict(stage='TRAINING',**rec,elapsed_seconds=elapsed+time.monotonic()-start,gpu=N.E.gpu())
                N.write(N.DOC/'STATUS.json',status);print(status,flush=True)
    assert all(torch.equal(buffers[k],v) for k,v in model.named_buffers())
    real_probe=O.probe(model,real);source_after=source_probe(model,held);noisy=real_probe['noisy_manual']
    sanity=bool(noisy['output_mean_px']<=10 and noisy['recovery_rate'] is not None and noisy['recovery_rate']>=.5 and noisy['output_mean_px']<=.5*noisy['input_mean_px'])
    path=N.RAW/'last300.pt'
    if not path.exists():N.save(path,dict(complete=True,step=300,protocol_sha256=protocol_sha,model_state_dict=model.state_dict()))
    N.freeze(N.RAW/'TRAIN_STEPS.json',history)
    N.freeze(N.DOC/'FIT.json',dict(complete=True,step=300,checkpoint=N.E.bound(path),trainability_pass=sanity,
        real_probe=real_probe,source_probe=source_after,BN_buffers_bit_exact=True,
        real_input_order_bit_exact=True,real_exposures=2400,synthetic_exposures=2400,L2_applications_per_step=1,
        elapsed_seconds=elapsed+time.monotonic()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        source_train_supervised_corners=sum(x['source_supervised'] for x in history),
        real_train_supervised_corners=sum(x['real_supervised'] for x in history),auto_promoted=False))
    N.write(N.DOC/'STATUS.json',dict(stage='TRAIN_COMPLETE',trainability_pass=sanity))
    N.verify();print('TRAIN_COMPLETE',N.E.read(N.DOC/'FIT.json'),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['before','train']);args=p.parse_args()
    before() if args.stage=='before' else run()
