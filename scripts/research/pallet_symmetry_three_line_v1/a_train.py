"""Bounded paired full-detector training, immutable recipe and exact-state resume."""
import argparse,copy,hashlib,json,os,random,signal,time
from collections import Counter
import cv2,numpy as np,torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
import env as E
from a_data import records,FixedDataset,order_indices,collate,cuda,worker_init
from a_loss import GenericSymmetryPoseLoss

STOP=False
def request_stop(*_):
    global STOP
    STOP=True
    print('STOP requested: checkpoint after current optimizer step',flush=True)

def configure(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False

def bn_fixed(model):
    model.train()
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()

def bn_state(model):
    return {n:b.detach().clone() for n,b in model.named_buffers() if n.endswith(('running_mean','running_var','num_batches_tracked'))}

def make_model(seed):
    configure(seed)
    model=YOLO(str(E.R0)).model.cuda().float();model.args=get_cfg();model.requires_grad_(True)
    bn_fixed(model)
    return model

def atomic_save(value,path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.pending');torch.save(value,tmp);tmp.replace(path)

def verify_binding(binding):
    for r in binding:
        path=E.Path(r['path']);path=path if path.is_absolute() else E.ROOT/path
        assert E.sha(path)==r['sha256'],r['path']

def lock():
    assert E.read(E.DOC/'A/JOINT_WIRING_AUDIT.json')['PASS']
    verify_binding(E.read(E.DOC/'SOURCE_BINDING.json')['files'])
    paths=[E.HERE/n for n in ['a_train.py','a_loss.py','a_joint.py','a_data.py']]
    paths += [E.RAW/'A/rect_target_sidecar.json',E.RAW/'A/square_annotation_target_view.json',E.RAW/'A/square_membership.json']
    sampler={}
    for cohort in ['RECT','SQUARE']:
        rows=records(cohort,'train')
        for seed in [1,2,3]:
            order=order_indices(len(rows),seed)
            sampler[f'{cohort}_{seed}']=dict(train_images=len(rows),sample_count=len(order),
              index_sequence_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
              first_batch_ids=[rows[i]['id'] for i in order[:16]],
              all_train_membership_SHA=hashlib.sha256('\n'.join(r['id'] for r in rows).encode()).hexdigest())
    E.freeze(E.DOC/'A/JOINT_TRAINING_LOCK.json',dict(
      approval='User: 일단 그렇게 해봐; paused before training; user: 다시해줘 resumes approved joint choice.',
      objective='min over one allowed g per GT row of weighted two-head STOCK pose/visibility/RLE loss, with per-head batch-global RLE clamp intact',
      optimizer='AdamW',lr=1e-4,betas=[.9,.999],weight_decay=1e-4,updates=2000,physical_batch=16,
      precision='FP32; CUDA matmul and cuDNN TF32 disabled',AMP=False,EMA=False,augmentation=False,
      BN='fixed statistics, affine trainable',clip_norm=10,parameter_set='all full-detector parameters requires_grad=True in both arms',
      E2E_head_weights=[.8,.2],E2E_head_schedule='constant native initial weights; step-based recipe has no epoch callback or added decay',
      loss_scale='stock returned per-batch loss summed unchanged; no extra normalization',
      selector='certified separable lower bound, else exact one-hot MILP with zero gap; failure stops fit',
      selection_numeric_tolerance='FP64 reduction of FP32 per-anchor costs; reconstruction tested against stock, atol/rtol 2e-5; selected loss/gradient uses stock directly',
      sampler=sampler,bindings=[E.bound(p) for p in paths],probe='fixed first 16 TRAIN records at updates100/500/1000/2000',
      checkpoint='raw last weights; resume carries optimizer, RNG, sampler offset and branch histograms',
      groups='actual 3D-derived C1/C2; separately approved task-C4 square',new_square_view_shared_by_both_arms=True,
      no_P_or_DHT_in_A=True,FINAL_access=False))

def fit(cohort,arm,seed,smoke=False):
    steps=3 if smoke else 2000;stage='smoke' if smoke else 'runs';name=f'{cohort}_{arm}_seed{seed}'
    root=E.RAW/f'A/{stage}/{name}';doc=E.DOC/f'A/{stage}/{name}.json';resume=root/'resume.pt'
    if doc.exists() and E.read(doc).get('status')=='COMPLETE':return E.read(doc)
    E.gpu();verify_binding(E.read(E.DOC/'A/JOINT_TRAINING_LOCK.json')['bindings'])
    model=make_model(seed);params=[p for p in model.parameters() if p.requires_grad]
    criterion=GenericSymmetryPoseLoss(model,arm=='EQUIV')
    optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),weight_decay=1e-4)
    initial_bn=bn_state(model);initial_state_SHA=E.R0_SHA
    offset=0;hist=Counter();methods=Counter();elapsed_before=0.;trace=[]
    if resume.exists():
        c=torch.load(resume,map_location='cuda',weights_only=False)
        assert c['cohort']==cohort and c['arm']==arm and c['seed']==seed and c['recipe_SHA']==E.sha(E.DOC/'A/JOINT_TRAINING_LOCK.json')
        model.load_state_dict(c['state_dict']);optimizer.load_state_dict(c['optimizer']);offset=c['step']
        random.setstate(c['python_rng']);np.random.set_state(c['numpy_rng']);torch.set_rng_state(c['torch_rng'].cpu());torch.cuda.set_rng_state(c['cuda_rng'].cpu())
        hist.update(c['branches']);methods.update(c['methods']);elapsed_before=c['elapsed_seconds'];trace=c['trace']
    elif doc.exists():raise RuntimeError(f'Existing incomplete run without resumable checkpoint: {doc}')
    indices=order_indices(len(records(cohort,'train')),seed,steps)[offset*16:]
    loader=DataLoader(FixedDataset(cohort,indices=indices),batch_size=16,shuffle=False,num_workers=4,pin_memory=True,
      collate_fn=collate,worker_init_fn=worker_init,persistent_workers=False,
      generator=torch.Generator().manual_seed(seed))
    probe=None
    if not smoke:probe=cuda(collate([FixedDataset(cohort)[i] for i in range(16)]))
    start=time.monotonic();step=offset;status='RUNNING';torch.cuda.reset_peak_memory_stats()
    def checkpoint(state):
        elapsed=elapsed_before+time.monotonic()-start
        atomic_save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),step=step,cohort=cohort,arm=arm,seed=seed,
          python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
          recipe_SHA=E.sha(E.DOC/'A/JOINT_TRAINING_LOCK.json'),branches=dict(hist),methods=dict(methods),elapsed_seconds=elapsed,trace=trace),resume)
        audit=dict(status=state,stage=stage,cohort=cohort,arm=arm,seed=seed,optimizer_updates=step,training_forward_images=step*16,
          main_training=not smoke,physical_batch=16,AMP=False,EMA=False,initial_R0_SHA=initial_state_SHA,
          BN_unchanged=all(torch.equal(b,dict(model.named_buffers())[n]) for n,b in initial_bn.items()),
          head_weights=[criterion.stock.o2m,criterion.stock.o2o],branch_histogram=dict(hist),selector_methods=dict(methods),
          parameter_count=sum(p.numel() for p in params),trainable_names_SHA=hashlib.sha256('\n'.join(n for n,p in model.named_parameters() if p.requires_grad).encode()).hexdigest(),
          elapsed_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),trace=trace,
          resume_checkpoint=E.bound(resume),sampler_lock=f'{cohort}_{seed}',GPU=E.gpu())
        assert audit['BN_unchanged'];E.write(doc,audit)
        E.write(E.DOC/'A/LIVE_STATUS.json',dict(time=E.now(),pid=os.getpid(),run=name,stage=stage,status=state,updates=step,target=steps))
        return audit
    checkpoint('RUNNING')
    try:
        for raw in loader:
            if STOP:status='PAUSED';break
            batch=cuda(raw);assert len(batch['img'])==16
            before=torch.nn.utils.parameters_to_vector(params).detach().clone()
            optimizer.zero_grad(set_to_none=True);pred=model(batch['img']);loss,items=criterion(pred,batch);total=loss.sum()
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite loss before update')
            total.backward();pre=torch.nn.utils.clip_grad_norm_(params,10,error_if_nonfinite=True)
            post=torch.nn.utils.clip_grad_norm_(params,float('inf'),error_if_nonfinite=True)
            optimizer.step();step+=1
            delta=float(torch.linalg.vector_norm(torch.nn.utils.parameters_to_vector(params).detach()-before))
            if not np.isfinite(delta) or delta==0:raise FloatingPointError('Invalid or zero parameter update')
            hist.update(criterion.last_audit['branches']);methods.update([criterion.last_audit['method']])
            trace.append(dict(step=step,loss=float(total.detach()),one2one_components=items.tolist(),gradient_pre=float(pre),gradient_post=float(post),
              parameter_delta_L2=delta,lr=[g['lr'] for g in optimizer.param_groups],batch_ids=batch['ids'],
              selector=criterion.last_audit['method'],branches=criterion.last_audit['branches']))
            del pred,loss,total,before,batch,raw
            if not smoke and step in [100,500,1000,2000]:
                with torch.no_grad():pl,_=criterion(model(probe['img']),probe)
                trace[-1]['fixed_train_probe_loss']=float(pl.sum())
            if step%100==0 or step==steps or STOP:
                state='PAUSED' if STOP else 'COMPLETE' if step==steps else 'RUNNING'
                checkpoint(state);print('A',name,step,'/',steps,'loss',round(trace[-1]['loss'],4),'seconds',round(time.monotonic()-start,1),flush=True)
            if STOP:status='PAUSED';break
        if step==steps:
            status='COMPLETE'
            if not smoke:
                path=root/'last.pt';cpu_model=copy.deepcopy(model).cpu().float()
                atomic_save(dict(model=cpu_model,ema=None,train_args=vars(model.args),epoch=-1,updates=step,cohort=cohort,arm=arm,seed=seed,
                  source_R0_SHA=E.R0_SHA,recipe_SHA=E.sha(E.DOC/'A/JOINT_TRAINING_LOCK.json')),path)
                del cpu_model
        result=checkpoint(status)
        if not smoke and status=='COMPLETE':result['last_weights']=E.bound(root/'last.pt');E.write(doc,result)
        return result
    except Exception as error:
        result=checkpoint('NUMERICAL_OR_EXECUTION_FAILURE');result['error']=repr(error);E.write(doc,result)
        raise
    finally:
        del model,optimizer,criterion,loader,params,probe;torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','main'],required=True);args=parser.parse_args()
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    configure(1);E.gpu();lock();results=[]
    if args.stage=='smoke':
        for cohort in ['RECT','SQUARE']:
            for arm in ['INDEXED','EQUIV']:
                results.append(fit(cohort,arm,1,True))
                if STOP:return
        assert sum(r['optimizer_updates'] for r in results)==12
        E.write(E.DOC/'A/SMOKE_AUDIT.json',dict(PASS=True,updates=12,main_updates=0,physical_batch=16,runs=results))
    else:
        assert E.read(E.DOC/'A/SMOKE_AUDIT.json')['PASS']
        for cohort in ['RECT','SQUARE']:
            for seed in [1,2,3]:
                for arm in ['INDEXED','EQUIV']:
                    results.append(fit(cohort,arm,seed))
                    if STOP:return
        E.write(E.DOC/'A/MAIN_TRAINING_COMPLETE.json',dict(complete=True,fits=12,optimizer_updates=sum(r['optimizer_updates'] for r in results),
          training_forward_images=sum(r['training_forward_images'] for r in results),probe_forward_images=12*4*16,
          runs=[E.bound(E.DOC/f"A/runs/{r['cohort']}_{r['arm']}_seed{r['seed']}.json") for r in results]))
    print('A TRAIN STAGE COMPLETE',args.stage,flush=True)
if __name__=='__main__':main()
