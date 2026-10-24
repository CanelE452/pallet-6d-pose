"""Source-only fixed-budget v65 head with compatible-state partial likelihood."""
import argparse,hashlib,json,os,random,shutil,time
from pathlib import Path
import numpy as np
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from query_global_amodal_head import HeadConfig,QueryGlobalAmodalHead,parameter_count,save_checkpoint,load_model

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def partial_nll(logits,allowed):
    """-log P(any compatible joint), retaining correlation of exact/near."""
    if logits.shape[1]!=6 or allowed.shape!=logits.shape[:1]+logits.shape[2:]:raise ValueError('six-joint aligned partial labels')
    bits=torch.arange(6,device=logits.device).reshape((1,6)+((1,)*(logits.ndim-2)))
    permitted=(allowed[:,None].to(torch.long)&(1<<bits))!=0
    # Invalid cells carry63; zero is rejected so logsumexp never sees all-inf.
    if not bool(permitted.any(1).all()):raise ValueError('zero compatible mask')
    return torch.logsumexp(logits,1)-torch.logsumexp(logits.masked_fill(~permitted,-torch.inf),1)

def domain_loss(loss,allowed,valid,codes,*,per_target=False):
    info=valid&(allowed!=63)&(allowed!=0)
    terms=[]
    if per_target:
        denom=info.flatten(1).sum(1);good=denom>0
        means=(loss*info).flatten(1).sum(1)/denom.clamp_min(1)
        for c in torch.unique(codes):
            ix=(codes==c)&good
            if bool(ix.any()):terms.append(means[ix].mean())
    else:
        for c in torch.unique(codes):
            ix=(codes==c)
            if bool(info[ix].any()):terms.append(loss[ix][info[ix]].mean())
    # A sampled target may contain no global information. Preserve and ignore
    # it rather than invent a label or divide by zero.
    return torch.stack(terms).mean()if terms else loss.sum()*0

def load_cache(p):
    with np.load(p,allow_pickle=False)as z:d={k:z[k]for k in z.files}
    n=len(d['x']);t=len(d['global_rgb'])
    shapes=dict(x=(n,65,9,9),rgb_native=(n,75,9,9),sam=(n,2,9,9),context=(n,64),box_grid=(n,4,9,9),
        dense_allowed=(n,9,9),dense_valid=(n,9,9),source=(n,),sample_id=(n,),group_id=(n,),image_sha256=(n,),target_index=(n,),
        global_rgb=(t,3,64,64),global_sam=(t,1,64,64),global_prompt=(t,1,64,64),global_allowed=(t,64,64),global_valid=(t,64,64),
        global_id=(t,),global_image_sha256=(t,),global_source=(t,),global_box=(t,4),global_index64=(n,9,9),global_index8=(n,9,9),global_query_valid=(n,9,9))
    for k,s in shapes.items():
        if k not in d or d[k].shape!=s:raise ValueError((k,s))
    for k in ['dense_allowed','global_allowed']:
        if d[k].dtype!=np.uint8 or not np.isin(d[k],np.arange(1,64)).all():raise ValueError('compatible bitmask1..63 required')
    for k in ['dense_valid','global_valid','global_query_valid']:
        if d[k].dtype!=bool:raise ValueError('Boolean native validity required')
    for k in ['dense','global']:
        if not(d[k+'_allowed'][~d[k+'_valid']]==63).all():raise ValueError('invalid cells must be uninformed63')
    for k in ['x','rgb_native','sam','context','box_grid','global_rgb','global_sam','global_box']:
        if not np.isfinite(d[k]).all():raise ValueError('nonfinite '+k)
    if d['rgb_native'].dtype!=np.uint8 or d['global_rgb'].dtype!=np.uint8:raise ValueError('uint8 RGB required')
    if np.abs(d['sam']).max()>4.000001 or np.abs(d['global_sam']).max()>4.000001 or np.abs(d['box_grid']).max()>4.000001:raise ValueError('normalized input channel range')
    ti=d['target_index'];valid=d['global_query_valid'];i64=d['global_index64'];i8=d['global_index8']
    if ti.dtype!=np.int64 or ti.min()<0 or ti.max()>=t or len(np.unique(ti))!=t:raise ValueError('complete target table mapping')
    if not np.array_equal(d['global_image_sha256'][ti],d['image_sha256'])or not np.array_equal(d['global_source'][ti],d['source']):raise ValueError('target identity/source mapping')
    if i64.min()<0 or i64.max()>=4096 or i8.min()<0 or i8.max()>=64:raise ValueError('global index bounds')
    if not np.array_equal(i8[valid],((i64//64)//8*8+(i64%64)//8)[valid]):raise ValueError('global coarse axes')
    if(i64[~valid]!=0).any()or(i8[~valid]!=0).any():raise ValueError('invalid global index placeholder')
    return d

def smoke():
    torch.manual_seed(12)
    z=torch.randn(100,6,requires_grad=True);a=torch.randint(1,64,(100,));loss=partial_nll(z,a)
    p=torch.softmax(z,1);ref=[]
    for i,m in enumerate(a.tolist()):ref.append(-torch.log(p[i,[j for j in range(6)if m&(1<<j)]].sum()))
    assert torch.allclose(loss,torch.stack(ref),atol=4e-7)
    full=torch.full((100,),63,dtype=torch.long);zero=partial_nll(z,full)
    assert torch.equal(zero,torch.zeros_like(zero));zero.sum().backward();assert torch.equal(z.grad,torch.zeros_like(z))
    y=torch.randint(0,6,(100,));one=partial_nll(z,1<<y)
    assert torch.allclose(one,torch.nn.functional.cross_entropy(z,y,reduction='none'),atol=4e-7)
    ls=torch.tensor([1.,3.,4.]);aa=torch.tensor([1,1,1]);vv=torch.ones(3,dtype=torch.bool);cc=torch.tensor([0,0,1])
    assert domain_loss(ls,aa,vv,cc)==3
    return dict(PASS=True,partial_mask_probability_sum=True,unknown63_loss_and_gradient_exact_zero=True,onehot_CE_parity=True,domain_normalization=True,new_training=0,new_PnP=0)

def fit(args):
    out=Path(args.out);assert not out.exists(),'preserve existing run'
    approved=json.loads(Path(args.input_lock).read_text())
    for b in approved['bindings']:
        if sha(b['path'])!=b['sha256']:raise ValueError('frozen input/code binding changed '+b['path'])
    if sha(__file__)!=approved['trainer_sha256']or sha(Path(__file__).with_name('query_global_amodal_head.py'))!=approved['model_sha256']:
        raise ValueError('fit entry code differs from frozen input lock')
    run=approved['runs'][args.candidate]
    for role in ['train','val','protocol','denylist']:
        b=run[role]
        if Path(getattr(args,role)).resolve()!=Path(b['path']).resolve()or sha(b['path'])!=b['sha256']:
            raise ValueError('unapproved source '+role)
    if out.resolve()!=Path(run['out']).resolve():raise ValueError('unapproved output run directory')
    protocol=json.loads(Path(args.protocol).read_text());protocol_sha=sha(args.protocol)
    if protocol['updates']!=args.updates or protocol['validation_every']!=args.val_every or protocol['seed']!=args.seed:raise ValueError('fixed protocol budget/seed')
    tr,va=load_cache(args.train),load_cache(args.val);denied=set(json.loads(Path(args.denylist).read_text())['image_sha256'])
    for d in [tr,va]:
        if set(d['image_sha256'])&denied:raise ValueError('private image in source cache')
    for key in ['group_id','sample_id','image_sha256','global_id','global_image_sha256']:
        if set(tr[key])&set(va[key]):raise ValueError('source TRAIN/VAL overlap '+key)
    sources=sorted(set(tr['source']));v_sources=sorted(set(va['source']))
    if sources!=sorted(protocol['candidates'][args.candidate]['training_sources'])or v_sources!=sorted(protocol['validation_sources']):raise ValueError('candidate/source set mismatch')
    torch.set_num_threads(1);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    model=QueryGlobalAmodalHead(HeadConfig()).to(args.device);assert parameter_count(model)==95508
    out.mkdir(parents=True);(out/'code').mkdir()
    for p in [Path(__file__),Path(__file__).with_name('query_global_amodal_head.py')]:shutil.copy2(p,out/'code'/p.name)
    shutil.copy2(args.protocol,out/'PROTOCOL_AT_START.json')
    audit=dict(arguments=vars(args),input_lock_sha256=sha(args.input_lock),protocol_sha256=protocol_sha,train_sha256=sha(args.train),val_sha256=sha(args.val),
        parameters=parameter_count(model),code={p.name:sha(p)for p in (out/'code').iterdir()},
        source_only=True,current223_used=False,new_PnP=0,runtime=dict(executable=os.sys.executable,torch=str(torch.__version__),CUDA=torch.version.cuda),
        fitting_sources=sources,validation_sources=v_sources,sampling='stratified source, with replacement; rotate remainder allocation per update')
    write(out/'RUN_STARTED.json',audit);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(args.seed);pools={s:np.flatnonzero((tr['source']==s)&tr['dense_valid'][:,4,4]&(tr['dense_allowed'][:,4,4]!=63))for s in sources}
    if any(len(p)==0 for p in pools.values()):raise ValueError('empty informative source training pool')
    source_codes={s:i for i,s in enumerate(sorted(set(sources)|set(v_sources)))}
    def tensor(d,k,ix,dtype=torch.float32):return torch.as_tensor(np.ascontiguousarray(d[k][ix]),device=args.device,dtype=dtype)
    def inputs(d,ix):
        ts,inv=np.unique(d['target_index'][ix],return_inverse=True)
        return dict(x=tensor(d,'x',ix),rgb_native=tensor(d,'rgb_native',ix)/255,sam=tensor(d,'sam',ix),context=tensor(d,'context',ix),box_grid=tensor(d,'box_grid',ix),
          global_input=torch.cat((tensor(d,'global_rgb',ts)/255,tensor(d,'global_sam',ts),tensor(d,'global_prompt',ts)),1),
          target_inverse=torch.as_tensor(inv,device=args.device,dtype=torch.long),global_index64=tensor(d,'global_index64',ix,torch.long),global_index8=tensor(d,'global_index8',ix,torch.long),global_query_valid=tensor(d,'global_query_valid',ix,torch.bool)),ts
    def validation():
        model.eval();outs=[]
        with torch.inference_mode():
            for b in range(0,len(va['x']),256):outs.append(model(**inputs(va,np.arange(b,min(len(va['x']),b+256)))[0]).cpu().numpy())
        logits=np.concatenate(outs).astype(np.float64);logp=logits-np.logaddexp.reduce(logits,axis=1)[:,None]
        masks=va['dense_allowed'][:,4,4];valid=va['dense_valid'][:,4,4]&(masks!=63)
        permit=(masks[:,None]&(1<<np.arange(6))[None])!=0
        nll=-np.logaddexp.reduce(np.where(permit,logp,-np.inf),axis=1)
        by={s:float(nll[valid&(va['source']==s)].mean())for s in v_sources}
        if not np.isfinite(list(by.values())).all():raise ValueError('empty/invalid informative validation source')
        return float(np.mean(list(by.values()))),by,logits
    best=float('inf');best_update=None;history=[];start=time.monotonic();losses=[]
    for update in range(1,args.updates+1):
        if sha(args.protocol)!=protocol_sha:raise RuntimeError('protocol changed while fitting')
        model.train();q,r=divmod(256,len(sources));chunks=[]
        for j,s in enumerate(sources):
            take=q+int((j-(update-1))%len(sources)<r);chunks.append(rng.choice(pools[s],size=take,replace=True))
        ix=np.concatenate(chunks);rng.shuffle(ix);inp,ts=inputs(tr,ix)
        center,dense,glob=model(**inp,return_dense=True)
        allowed=tensor(tr,'dense_allowed',ix,torch.long);valid=tensor(tr,'dense_valid',ix,torch.bool)
        codes=torch.as_tensor([source_codes[s]for s in tr['source'][ix]],device=args.device)
        ga=tensor(tr,'global_allowed',ts,torch.long);gv=tensor(tr,'global_valid',ts,torch.bool)
        gc=torch.as_tensor([source_codes[s]for s in tr['global_source'][ts]],device=args.device)
        lc=domain_loss(partial_nll(center,allowed[:,4,4]),allowed[:,4,4],valid[:,4,4],codes)
        ld=domain_loss(partial_nll(dense,allowed),allowed,valid,codes)
        lg=domain_loss(partial_nll(glob,ga),ga,gv,gc,per_target=True)
        loss=lc+.25*ld+.25*lg
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite partial likelihood')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        if update%args.val_every==0 or update==args.updates:
            metric,by,z=validation();improved=metric<best
            if improved:
                best,best_update=metric,update
                save_checkpoint(out/'BEST_PARTIAL_HEAD.pt',model,updates=update,selection_partial_NLL=best,selection_by_source=by,threshold=.5)
            row=dict(update=update,train_loss=float(np.mean(losses)),VAL_partial_NLL=metric,by_source=by,saved_best=improved,seconds=time.monotonic()-start)
            history.append(row);write(out/'PROGRESS.json',row);print(json.dumps(row),flush=True);losses=[]
    model,payload=load_model(out/'BEST_PARTIAL_HEAD.pt',args.device);_,by,z=validation()
    np.savez_compressed(out/'VALIDATION_SOURCE_OUTPUTS.npz',joint_logits=z,sample_id=va['sample_id'],source=va['source'],dense_center_allowed=va['dense_allowed'][:,4,4],center_valid=va['dense_valid'][:,4,4])
    write(out/'HISTORY.json',history)
    complete={**audit,'completed':True,'optimizer_updates':args.updates,'best_update':best_update,'best_source_partial_NLL':best,'best_by_source':by,'checkpoint_sha256':sha(out/'BEST_PARTIAL_HEAD.pt'),'threshold':.5,'seconds':time.monotonic()-start,'eval_data_used':False,'PnP_used':False}
    if sha(args.protocol)!=protocol_sha:raise RuntimeError('protocol changed before freeze')
    write(out/'MODEL_FREEZE.json',complete);print('COMPLETE',json.dumps(complete),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true')
    for k in ['train','val','out','protocol','denylist','candidate','input-lock']:p.add_argument('--'+k)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=20261022);p.add_argument('--updates',type=int,default=2000);p.add_argument('--val-every',type=int,default=100)
    a=p.parse_args()
    if a.smoke:print(json.dumps(smoke()))
    else:fit(a)
