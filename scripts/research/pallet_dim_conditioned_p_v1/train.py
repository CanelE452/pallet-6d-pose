"""Fixed-budget paired small-head fits with exact-state resume, no real selection."""
import argparse,hashlib,time,io,unittest
import numpy as np
import torch
import dcp_env as E
from data import PaperData
from refiner import model,forward,train_loss,specification

def bindings():
    return [E.bound(E.HERE/f) for f in ['refiner.py','data.py','train.py']]+[E.bound(E.DOC/f) for f in ['TRAIN_PROTOCOL_LOCK.json','DIM_NORMALIZATION_LOCK.json','SOURCE_BINDINGS.json']]

def run_one(data,arm,seed,steps,smoke=False):
    lock=E.read(E.DOC/'TRAIN_PROTOCOL_LOCK.json');config=lock['config'];opt=lock['optimizer'];dim,sym=specification(arm)
    destination=E.RAW/('smoke' if smoke else 'runs')/f'{arm}_seed{seed}';destination.mkdir(parents=True,exist_ok=True)
    completion=E.DOC/('smoke' if smoke else 'fits')/f'{arm}_seed{seed}.json'
    if completion.exists():
        c=E.read(completion);assert c['complete'] and E.sha(destination/'last.pt')==c['checkpoint']['sha256'];return c
    E.gpu();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    head=model(arm,config).cuda().train()
    base_state={k:v for k,v in head.state_dict().items() if not k.startswith('metadata_')};initial=E.state_sha(base_state)
    torch.manual_seed(seed);ref=model('N0_BASE_REPLAY',config).cuda();assert initial==E.state_sha(ref.state_dict());del ref
    optimizer=torch.optim.AdamW(head.parameters(),lr=opt['lr'],betas=tuple(opt['betas']),weight_decay=opt['weight_decay'])
    order=data.order(seed) if hasattr(data,'order') else np.load(E.C.BRAW/f'order_seed{seed}.npy')
    assert order.shape==(6000,16);order=order[:steps]
    bhash=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest();locked=bindings()
    start_step=0;prior_seconds=0;history=[]
    if (destination/'last.pt').exists():
        ck=torch.load(destination/'last.pt',map_location='cpu',weights_only=False)
        assert ck['bindings']==locked and ck['batch_order_sha256']==bhash and ck['initial_base_sha256']==initial
        head.load_state_dict(ck['model_state_dict']);optimizer.load_state_dict(ck['optimizer_state_dict'])
        torch.set_rng_state(ck['torch_rng_state']);torch.cuda.set_rng_state_all(ck['cuda_rng_state'])
        start_step=ck['step'];prior_seconds=ck['elapsed_seconds'];history=ck['history']
    begin=time.monotonic();first_audit=None;probe=[]
    for step in range(start_step+1,steps+1):
        batch=data.batch(order[step-1],arm);optimizer.zero_grad(set_to_none=True)
        lr=E.old('train').learning_rate(step,dict(steps=6000,optimizer=opt))
        for g in optimizer.param_groups:g['lr']=lr
        out=forward(head,batch)
        if step==1 and dim:assert torch.equal(out['logits'],out['base_logits'])
        value=train_loss(out,batch,sym);assert torch.isfinite(value);value.backward()
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),opt['gradient_clip_norm'],error_if_nonfinite=True)
        if step==1:
            gradients={k:float(p.grad.norm()) if p.grad is not None else None for k,p in head.named_parameters()}
            assert gradients['adapt3.0.weight']>0 and gradients['adapt4.0.weight']>0
            if dim:assert gradients['metadata_scorer.2.weight']>0
            first_audit=dict(gradients=gradients,metadata_encoder_zero_first_step_expected=bool(dim),initial_base_sha256=initial,
              optimizer_only_refiner=True,R0_not_instantiated_in_training=True,zero_effect_step0=bool(dim))
            E.write(destination/'FIRST_STEP_AUDIT.json',first_audit)
        optimizer.step()
        if step==1 or step%100==0 or step==steps:
            snapshot=E.gpu();row=dict(step=step,loss=float(value.detach()),lr=lr,gradient_norm=float(norm),elapsed_seconds=prior_seconds+time.monotonic()-begin,gpu=snapshot)
            history.append(row);E.write(E.DOC/'TRAIN_PROGRESS.json',dict(arm=arm,seed=seed,smoke=smoke,**row))
            print('DCP_TRAIN',arm,seed,step,steps,round(row['loss'],6),round(row['elapsed_seconds'],1),flush=True)
        if arm.startswith('S') and step in [1000,3000,6000]:
            with torch.no_grad():
                pb=data.batch(data.train_rows[:16],arm);pv=float(train_loss(forward(head,pb),pb,sym))
            probe.append(dict(step=step,train_only_loss=pv));E.write(destination/f'TRAIN_PROBE_{step}.json',probe[-1])
        if step%500==0 or step==steps:
            ck=dict(complete=step==steps,smoke=smoke,step=step,expected_steps=steps,arm=arm,seed=seed,config=config,
              model_state_dict=head.state_dict(),optimizer_state_dict=optimizer.state_dict(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
              baseline_checkpoint_sha256=E.R0_SHA,initial_base_sha256=initial,batch_order_sha256=bhash,bindings=locked,
              elapsed_seconds=prior_seconds+time.monotonic()-begin,history=history)
            tmp=destination/'last.pending.pt';torch.save(ck,tmp);tmp.replace(destination/'last.pt')
    assert E.sha(E.R0)==E.R0_SHA and all(torch.isfinite(p).all() for p in head.parameters())
    assert E.state_sha({k:v for k,v in head.state_dict().items() if not k.startswith('metadata_')})!=initial
    c=dict(complete=True,arm=arm,seed=seed,steps=steps,smoke=smoke,checkpoint=E.bound(destination/'last.pt'),initial_base_sha256=initial,batch_order_sha256=bhash,
      elapsed_seconds=prior_seconds+time.monotonic()-begin,exposures=steps*16,history=history,first_step=E.read(destination/'FIRST_STEP_AUDIT.json'),
      final_step_only=True,R0_hash_unchanged=True,real_DEV_access=False,trainable_params=sum(p.numel() for p in head.parameters()),
      torch_version=torch.__version__,cudnn_version=torch.backends.cudnn.version(),TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32)
    E.write(completion,c);del head,optimizer,out,value,batch;torch.cuda.empty_cache();return c

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['tests','smoke','paper','square','mixed']);args=p.parse_args()
    torch.set_num_threads(4)
    if args.stage=='tests':
        out=io.StringIO();r=unittest.TextTestRunner(stream=out,verbosity=2).run(unittest.defaultTestLoader.discover(str(E.HERE),pattern='test_*.py'))
        E.write(E.DOC/'REGRESSION_TESTS.json',dict(PASS=r.wasSuccessful(),tests=r.testsRun,log=out.getvalue(),time=E.now()))
        print(out.getvalue());assert r.wasSuccessful();return
    assert E.read(E.DOC/'REGRESSION_TESTS.json')['PASS'];E.verify(E.read(E.DOC/'SOURCE_BINDINGS.json')['files']);E.gpu();assert torch.cuda.is_available()
    if args.stage in ['smoke','paper']:data=PaperData()
    else:
        from square_data import SquareData,MixedData
        data=SquareData() if args.stage=='square' else MixedData()
    if args.stage=='smoke':
        completed=[run_one(data,a,1,100,True) for a in ['N0_BASE_REPLAY','N4_META_SYM']]
        E.write(E.DOC/'SMOKE_COMPLETE.json',dict(complete=True,updates=200,runs=completed,performance_verdict=False));return
    assert E.read(E.DOC/'SMOKE_COMPLETE.json')['complete']
    arms=E.ARMS if args.stage=='paper' else E.SQUARE_ARMS if args.stage=='square' else E.MIXED_ARMS
    fits=[run_one(data,arm,seed,6000) for arm in arms for seed in [1,2,3]]
    E.write(E.DOC/f'{args.stage.upper()}_TRAINING_COMPLETE.json',dict(complete=True,fits=len(fits),updates=6000*len(fits),runs=fits))
    print('TRAINING_COMPLETE',args.stage,len(fits),flush=True)
if __name__=='__main__':main()
