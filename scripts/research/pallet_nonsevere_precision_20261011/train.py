"""Prelocked two-seed, five-arm screen; no REAL access during fitting."""
import argparse,copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
import precision_model as M

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def state_sha(s):
    h=hashlib.sha256()
    for k,v in sorted(s.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if hasattr(v,'tolist'):return clean(v.tolist())
    if isinstance(v,float) and not math.isfinite(v):return None
    return v
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(clean(v),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def configure():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False

def setup(args,seed,device='cuda'):
    env,refiner,Data,Base,targets,decode,loss=M.source_modules(args.source_root)
    p=env.C.BRAW/f'runs/seed{seed}/last.pt';ck=torch.load(p,map_location='cpu')
    assert ck['complete'] and ck['step']==6000
    base=Base(**ck['config']);base.load_state_dict(ck['model_state_dict'])
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=M.NativeEvidence(base).to(device)
    data=Data();data.side={k:data.side[k] for k in data.side.files}
    for value in data.side.values():value.flags.writeable=False
    order=np.load(env.C.BRAW/f'order_seed{seed}.npy',mmap_mode='r')[:3000]
    assert order.shape==(3000,16) and np.isin(order,data.train_rows).all()
    assert all(data.source['records'][int(data.indices[r])]['source_kind']=='synthetic' for r in np.unique(order))
    return env,refiner,data,model,order,p,targets,decode,loss

def batch_metadata(data,rows,device='cuda'):
    gains=torch.as_tensor(np.array(data.arrays['gain'][rows],copy=True),dtype=torch.float32,device=device)
    hw=torch.tensor([data.source['records'][int(data.indices[r])]['raw_shape_hw'] for r in rows],device=device)
    return gains,hw

def components(output,batch,data,rows,refiner,targets,decode,loss,geometry):
    gt,valid,branch,_=refiner.local_phase(output['points_raw'],output['point_valid'],batch['gt_points'],batch['gt_valid'],batch['permutations'],batch['group_valid'],output['box_diagonal'])
    gain,hw=batch_metadata(data,rows)
    eligible,_=M.aux_mask(output,gt,valid,gain,hw,targets)
    ce=loss(output,gt,valid)
    coord,error=M.coordinate_loss(output,gt,eligible,gain,decode)
    if geometry is None:
        geo=ce*0
    else:
        # A24 is already paired with the single initial-point whole-object branch.
        assert np.array_equal(geometry['branch'][rows],branch.detach().cpu().numpy())
        A=torch.as_tensor(np.array(geometry['A24'][rows],copy=True),device=error.device,dtype=error.dtype).detach()
        ok=torch.as_tensor(np.array(geometry['valid'][rows],copy=True),device=error.device).bool()
        # Cache correspondence support may be a subset; ineligible columns zero.
        mask=torch.as_tensor(np.array(geometry['eligible'][rows],copy=True),device=error.device).bool()
        assert torch.equal(mask,eligible), 'Geometry/coordinate eligibility differs'
        weighted=(A*error.reshape(-1,1,16)).reshape(-1,8,3,16)
        per=torch.linalg.vector_norm(weighted.reshape(-1,8,48),dim=-1).mean(-1)
        geo=(per*ok).sum()/ok.sum().clamp_min(1)
    return ce,coord,geo,dict(eligible=int(eligible.sum()),valid_corners=int(valid[:,:8].sum()),geo_frames=0 if geometry is None else int(ok.sum()))

def geometry_data(args):
    p=Path(args.private_dir)/'GEOMETRY_CACHE.npz'
    if p.exists():
        cache=np.load(p)
        result={k:cache[k] for k in cache.files};cache.close()
        for value in result.values():value.flags.writeable=False
        return result
    return None

def calibrate(args):
    env,refiner,data,model,order,path,targets,decode,loss=setup(args,1)
    geom=geometry_data(args)
    norm=[];parity=[];probe_rows=[];probe_logits=[];probe_support=[];start=time.monotonic()
    model.eval();torch.cuda.reset_peak_memory_stats()
    for j,rows in enumerate(order[:32]):
        batch=data.batch(rows,'N3_DIM_SYM');tiles=M.native_tiles(data,rows,batch,model.base.displacements)
        with torch.no_grad():
            bare=model.base(*(batch[k] for k in ('p3','p4','points','boxes','point_valid','input_shape')),lam=0)
            output=model(batch,False);rgb_output=model(batch,True,tiles)
            assert torch.equal(bare['logits'],output['logits']) and torch.equal(output['logits'],rgb_output['logits'])
        output['logits']=output['logits'].detach().requires_grad_(True)
        probe_rows.append(np.asarray(rows));probe_logits.append(output['logits'].detach().cpu().numpy());probe_support.append(output['point_support'].cpu().numpy())
        ce,coord,geo,count=components(output,batch,data,rows,refiner,targets,decode,loss,geom)
        grads=[torch.autograd.grad(v,output['logits'],retain_graph=True)[0] for v in (ce,coord,geo)]
        vals=[float(g.norm()) for g in grads];assert all(math.isfinite(v) for v in vals)
        norm.append(dict(batch=j,CE=vals[0],COORD=vals[1],GEO=vals[2],**count))
        parity.append(True)
    avg={k:float(np.mean([r[k] for r in norm])) for k in ('CE','COORD','GEO')}
    assert avg['CE']>0 and avg['COORD']>1e-12
    weights=dict(COORD=.25*avg['CE']/avg['COORD'],GEO=.25*avg['CE']/avg['GEO'] if avg['GEO']>1e-12 else None)
    probe_path=Path(args.private_dir)/'TRAIN_DECODER_LOGITS.npz';assert not probe_path.exists()
    np.savez_compressed(probe_path,rows=np.concatenate(probe_rows),logits=np.concatenate(probe_logits),point_support=np.concatenate(probe_support))
    torch.cuda.synchronize()
    write(Path(args.public_dir)/'AUX_WEIGHT_LOCK.json',dict(status='TRAIN_ONLY_ONE_TIME_LOCK',weights=weights,gradient_target_ratio=.25,minibatches=32,batch_size=16,gradient_measure='L2 wrt actual222 logits; ratio of mean norms',zero_guard='norm<=1e-12 disables geometry arms; nonfinite stops affected arm',gradient_records=norm,mean_norms=avg,shared_step0_logits_exact=all(parity),all_arm_capacity=sum(p.numel() for p in model.parameters()),original_corrector_parameters=sum(p.numel() for p in model.base.parameters()),additional_parameters=sum(p.numel() for p in model.parameters())-sum(p.numel() for p in model.base.parameters()),real_reads=0,optimizer_updates=0,actual_device_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),source_checkpoint_sha256=sha(path)))
    print('AUXILIARY_WEIGHTS_LOCKED',weights,flush=True)

def lr_at(step,opt):
    if step<=opt['warmup_steps']:return opt['lr']*step/opt['warmup_steps']
    x=(step-opt['warmup_steps'])/(3000-opt['warmup_steps'])
    return opt['lr']*(opt['cosine_final_lr_fraction']+(1-opt['cosine_final_lr_fraction'])*.5*(1+math.cos(math.pi*x)))

def train(args):
    doc=Path(args.public_dir);private=Path(args.private_dir)
    lock=read(doc/'PLAN_LOCK.json');weights=read(doc/'AUX_WEIGHT_LOCK.json')['weights']
    assert lock['status']=='LOCKED_BEFORE_FITS'
    for name,value in lock['training_code_sha256'].items():assert sha(Path(__file__).parent/name)==value
    geom=geometry_data(args)
    arms=list(M.ARMS) if geom is not None and weights['GEO'] is not None else ['M0','M1','M3']
    receipts=[];gpu_seconds=read(doc/'AUX_WEIGHT_LOCK.json')['actual_device_seconds'];common_steps=None
    for seed in (1,2):
        env,refiner,data,template,order,path,targets,decode,loss=setup(args,seed)
        models={a:copy.deepcopy(template) for a in arms};initial=state_sha(template.state_dict());del template
        opt=read(env.DOC/'TRAIN_PROTOCOL_LOCK.json')['optimizer']
        optimizers={a:torch.optim.AdamW(m.parameters(),lr=opt['lr'],betas=tuple(opt['betas']),weight_decay=opt['weight_decay']) for a,m in models.items()}
        assert all(state_sha(m.state_dict())==initial for m in models.values())
        start=time.monotonic();history=[];target_steps=3000 if common_steps is None else common_steps
        torch.cuda.reset_peak_memory_stats()
        for step,rows in enumerate(order,1):
            if step>target_steps:break
            batch=data.batch(rows,'N3_DIM_SYM')
            tiles=M.native_tiles(data,rows,batch,models[arms[0]].base.displacements) if any(a in arms for a in ('M3','M4')) else None
            logged=dict(seed=seed,step=step,arms={})
            for arm,m in models.items():
                m.train();optimizer=optimizers[arm];optimizer.zero_grad(set_to_none=True)
                lr=lr_at(step,opt)
                for g in optimizer.param_groups:g['lr']=lr
                output=m(batch,arm in ('M3','M4'),tiles)
                ce,coord,geo,count=components(output,batch,data,rows,refiner,targets,decode,loss,geom)
                value=ce+(weights['COORD']*coord if arm!='M0' else 0)+(weights['GEO']*geo if arm in ('M2','M4') else 0)
                ratios={}
                if step in (100,1000,target_steps):
                    gradients=[torch.autograd.grad(v,output['logits'],retain_graph=True)[0].norm() for v in (ce,coord,geo)]
                    ratios=dict(coord_to_CE=float(weights['COORD']*gradients[1]/gradients[0].clamp_min(1e-12)),geo_to_CE=None if weights['GEO'] is None else float(weights['GEO']*gradients[2]/gradients[0].clamp_min(1e-12)))
                assert torch.isfinite(value),('NONFINITE',seed,arm,step)
                value.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),opt['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step()
                if step==1 or step%100==0 or step==target_steps:
                    logged['arms'][arm]=dict(loss=float(value.detach()),CE=float(ce.detach()),coord=float(coord.detach()),geo=float(geo.detach()),gradient_norm=float(gn),lr=lr,gradient_ratios=ratios,**count)
            if step==1 or step%100==0 or step==target_steps:
                torch.cuda.synchronize();logged['device_seconds']=time.monotonic()-start;history.append(logged);print('SCREEN_TRAIN',json.dumps(logged),flush=True)
            if seed==1 and step==200:
                elapsed=time.monotonic()-start
                # Fixed budget algorithm: reserve1000s inference/benchmark and15% safety.
                per=elapsed/200
                capacity=int((6200-gpu_seconds)/(2*per*1.15))
                common_steps=min(3000,max(200,(capacity//100)*100));target_steps=common_steps
                write(doc/'THROUGHPUT_RECEIPT.json',dict(status='MEASURED_FIRST200_ACTUAL_UPDATES',seed=seed,arms=arms,updates_per_arm=200,actual_device_seconds=elapsed,seconds_per_matched_arm_round=per,common_last_step=common_steps,step_rule='min(3000,floor100((6200-calibration)/(2*round_seconds*1.15))); minimum200',schedule='original100-step warmup/cosine horizon3000 preserved even if budgetlimited',no_restarts=True,total_updates_so_far=200*len(arms)))
            if step==target_steps:break
            assert gpu_seconds+time.monotonic()-start<7200,'DEVICE_BUDGET_GUARD'
        torch.cuda.synchronize();elapsed=time.monotonic()-start;gpu_seconds+=elapsed
        ckpts={}
        for arm,m in models.items():
            p=private/f'checkpoints/{arm}_seed{seed}.pt';p.parent.mkdir(exist_ok=True)
            assert not p.exists()
            torch.save(dict(arm=arm,seed=seed,step=step,state_dict=m.state_dict(),config=m.base.__dict__.get('config',read(env.DOC/'TRAIN_PROTOCOL_LOCK.json')['config']),initial_checkpoint_sha256=sha(path),order_sha256=hashlib.sha256(order[:step].astype('<i8').tobytes()).hexdigest(),plan_sha256=sha(doc/'PLAN_LOCK.json'),aux_weights=weights),p)
            ckpts[arm]=dict(private_namespace=str(p.relative_to(private)),sha256=sha(p))
        receipt=dict(seed=seed,status='DONE' if step==3000 else 'BUDGET_LIMITED',arms=arms,updates_per_arm=step,total_updates=step*len(arms),batch=16,shared_initial_state_sha256=initial,original_checkpoint_sha256=sha(path),last_checkpoint_only=True,unique_train_frames=int(len(np.unique(order[:step]))),order_sha256=hashlib.sha256(order[:step].astype('<i8').tobytes()).hexdigest(),actual_device_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),history=history,checkpoints=ckpts,backbone_training_updates=0,protected_or_Severe_REAL_training_reads=0,source_checkpoint_unchanged=True)
        write(doc/f'TRAIN_RECEIPT_seed{seed}.json',receipt);receipts.append(receipt)
        del models,optimizers,data;torch.cuda.empty_cache()
    assert len({r['updates_per_arm'] for r in receipts})==1
    write(doc/'TRAIN_RECEIPTS.json',dict(status='DONE' if common_steps==3000 else 'BUDGET_LIMITED',planned_arms=list(M.ARMS),executed_arms=arms,blocked_arms=[a for a in M.ARMS if a not in arms],seeds=[1,2],common_steps=common_steps,total_updates=sum(r['total_updates'] for r in receipts),actual_device_seconds=gpu_seconds,device_budget_seconds=7200,FP32=True,AMP=False,matmul_TF32=False,cudnn_TF32=False,real_selection=False,receipts=[f'TRAIN_RECEIPT_seed{s}.json' for s in (1,2)]))
    print('ALL_MATCHED_FITS_COMPLETE',common_steps,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','train']);p.add_argument('--source-root',required=True);p.add_argument('--private-dir',required=True);p.add_argument('--public-dir',required=True)
    a=p.parse_args();configure();(calibrate if a.stage=='calibrate' else train)(a)
