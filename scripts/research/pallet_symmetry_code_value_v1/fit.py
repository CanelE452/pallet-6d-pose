"""Twelve paired 6000-step fits, four discarded 50-step smokes; no other updates."""
import argparse,io,time,unittest
import numpy as np
import torch
import cv_env as E
from code_adapter import model,TrainingData,forward,train_loss

def tests():
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(E.HERE),pattern='test_*.py')
    # Reuse original mathematical regressions without invoking their writers.
    for name in ['test_refiner','test_evaluation','test_pose']:
        module=E.D.C.module('cv_reused_'+name,E.DCP_CODE/(name+'.py'));suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    payload=dict(PASS=result.wasSuccessful(),tests=result.testsRun,log=stream.getvalue(),time=E.now(),optimizer_updates=0)
    E.write(E.DOC/'REGRESSION_TESTS.json',payload);E.write(E.DOC/'TEST_RESULTS.json',payload);print(stream.getvalue(),flush=True);assert result.wasSuccessful()
def bindings():
    return [E.bound(E.HERE/f) for f in ['cv_env.py','code_adapter.py','fit.py']]+[E.bound(E.DOC/f) for f in ['PROTOCOL_LOCK.json','SOURCE_BINDING.json','CAPACITY_PARITY.json']]+[E.bound(E.DCP_CODE/f) for f in ['refiner.py','data.py','square_data.py']]
def one(data,arm,seed,smoke=False):
    lock=E.protocol();steps=50 if smoke else 6000;root=E.RAW/('smoke' if smoke else 'runs')/f'{arm}_seed{seed}';root.mkdir(parents=True,exist_ok=True)
    marker=E.DOC/('smoke' if smoke else 'fits')/f'{arm}_seed{seed}.json'
    if marker.exists():
        c=E.read(marker);assert c['complete'];E.verify([c['checkpoint']]);return c
    E.gpu();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);head=model().cuda().train()
    initial=E.protocol()['initial'][str(seed)];state=torch.load(E.ROOT/initial['file']['path'],map_location='cpu',weights_only=False)
    assert E.state_sha(head.state_dict())==initial['full_state_sha256'];head.load_state_dict(state)
    opt=lock['optimizer'];optimizer=torch.optim.AdamW(head.parameters(),lr=opt['lr'],betas=tuple(opt['betas']),weight_decay=opt['weight_decay'])
    order=np.array(data.order(seed)[:steps]);oh=E.order_sha(order);bound=bindings();start=0;elapsed=0;history=[]
    ckpath=root/'last.pt'
    if ckpath.exists():
        ck=torch.load(ckpath,map_location='cpu',weights_only=False);assert ck['bindings']==bound and ck['order_sha256']==oh
        head.load_state_dict(ck['state']);optimizer.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start=ck['step'];elapsed=ck['elapsed_seconds'];history=ck['history']
    begin=time.monotonic()
    for step in range(start+1,steps+1):
        b=data.batch(order[step-1],arm);optimizer.zero_grad(set_to_none=True);lr=E.D.old('train').learning_rate(step,dict(steps=6000,optimizer=opt))
        for g in optimizer.param_groups:g['lr']=lr
        out=forward(head,b)
        if step==1:assert torch.equal(out['logits'],out['base_logits'])
        value=train_loss(out,b,True);assert torch.isfinite(value);value.backward()
        gn=torch.nn.utils.clip_grad_norm_(head.parameters(),opt['gradient_clip_norm'],error_if_nonfinite=True)
        if step==1:
            assert float(head.metadata_scorer[-1].weight.grad.norm())>0
            E.write(root/'FIRST_STEP.json',dict(initial_state_sha256=E.state_sha(state),zero_effect_initial_logits=True,only_refiner_optimizer=True,
              context_first5=b['context'][:,:5].cpu().tolist(),context_last3=b['context'][:,5:].cpu().tolist(),scorer_gradient=float(head.metadata_scorer[-1].weight.grad.norm())))
        optimizer.step()
        if step==1 or step%100==0 or step==steps:
            row=dict(step=step,loss=float(value.detach()),gradient_norm=float(gn),lr=lr,elapsed_seconds=elapsed+time.monotonic()-begin,gpu=E.gpu());history.append(row)
            E.write(E.DOC/'TRAIN_PROGRESS.json',dict(arm=arm,seed=seed,smoke=smoke,**row));print('CODE_TRAIN',arm,seed,step,steps,round(row['loss'],6),round(row['elapsed_seconds'],1),flush=True)
        if step%500==0 or step==steps:
            ck=dict(complete=step==steps,step=step,arm=arm,seed=seed,smoke=smoke,state=head.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
              initial_state_sha256=initial['full_state_sha256'],order_sha256=oh,bindings=bound,history=history,elapsed_seconds=elapsed+time.monotonic()-begin,R0_sha256=E.D.R0_SHA)
            tmp=root/'last.pending.pt';torch.save(ck,tmp);tmp.replace(ckpath)
    assert all(torch.isfinite(v).all() for v in head.state_dict().values());assert E.sha(E.D.R0)==E.D.R0_SHA
    c=dict(complete=True,arm=arm,seed=seed,steps=steps,smoke=smoke,checkpoint=E.bound(ckpath),initial_state_sha256=initial['full_state_sha256'],order_sha256=oh,
      params=sum(p.numel() for p in head.parameters()),elapsed_seconds=elapsed+time.monotonic()-begin,history=history,R0_frozen=True,DEV_training=False,
      FP32=True,AMP=False,torch_version=torch.__version__,TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32)
    E.write(marker,c);del head,optimizer,out,b,value;torch.cuda.empty_cache();return c
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['tests','smoke','A','B']);args=p.parse_args();torch.set_num_threads(4)
    if args.stage=='tests':tests();return
    assert E.read(E.DOC/'REGRESSION_TESTS.json')['PASS'];E.gpu();assert torch.cuda.is_available()
    if args.stage=='smoke':
        fits=[]
        for track in ['A','B']:
            data=TrainingData(track)
            for arm in E.ARMS[track]:fits.append(one(data,arm,1,True))
            del data
        for track in ['A','B']:
            pair=[c for c in fits if c['arm'].startswith(track)];assert len({c['initial_state_sha256'] for c in pair})==1 and len({c['order_sha256'] for c in pair})==1
        E.write(E.DOC/'SMOKE_COMPLETE.json',dict(complete=True,updates=200,fits=fits,discarded=True,performance_verdict=False));return
    assert E.read(E.DOC/'SMOKE_COMPLETE.json')['complete']
    if args.stage=='B':assert E.read(E.DOC/'A_C1C2_RESULTS.json')['complete']
    data=TrainingData(args.stage);fits=[one(data,arm,s) for arm in E.ARMS[args.stage] for s in [1,2,3]]
    E.write(E.DOC/f'{args.stage}_TRAINING_COMPLETE.json',dict(complete=True,fits=fits,updates=36000));print('TRAIN_COMPLETE',args.stage,flush=True)
if __name__=='__main__':main()
