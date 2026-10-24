"""V67 one-factor objective control: partial NLL + known exact-visible BCE.

Architecture, source batches, budget, and partial-NLL checkpoint selector are
unchanged from frozen V66. No actual cache is accessed by --smoke.
"""
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

def visible_marginal_nll(logits,allowed):
    """Stable exact-visible BCE and its known-fact mask, with no new labels.

    Only allowed==1 certifies exact V (joint VV). A bitmask without bit0
    certifies exact nonvisible, even if E/O remains partial. Odd masks>1
    retain V as a possibility and cannot supervise this marginal.
    """
    if logits.shape[1]!=6 or allowed.shape!=logits.shape[:1]+logits.shape[2:]:raise ValueError('six-joint aligned partial labels')
    if not bool(((allowed>=1)&(allowed<=63)).all()):raise ValueError('compatible masks must be1..63')
    positive=allowed==1;known=positive|((allowed.to(torch.long)&1)==0)
    denominator=torch.logsumexp(logits,1)
    positive_nll=denominator-logits[:,0]
    negative_nll=denominator-torch.logsumexp(logits[:,1:],1)
    value=torch.where(positive,positive_nll,negative_nll)
    # Unknown cells have exact zero value/gradient before domain reduction.
    return torch.where(known,value,torch.zeros_like(value)),known

def visible_domain_loss(logits,allowed,valid,codes,*,per_target=False):
    loss,known=visible_marginal_nll(logits,allowed)
    return domain_loss(loss,allowed,valid&known,codes,per_target=per_target)

def objective_terms(center,dense,glob,allowed,valid,codes,ga,gv,gc,*,visible_bce_weight=1.):
    if visible_bce_weight not in (0.,1.):raise ValueError('Only frozen lambda1 or lambda0 parity control is supported')
    lc=domain_loss(partial_nll(center,allowed[:,4,4]),allowed[:,4,4],valid[:,4,4],codes)
    ld=domain_loss(partial_nll(dense,allowed),allowed,valid,codes)
    lg=domain_loss(partial_nll(glob,ga),ga,gv,gc,per_target=True)
    partial=lc+.25*ld+.25*lg
    if visible_bce_weight==0.:
        # Do not even add zero*BCE: this preserves the exact V66 scalar graph.
        zero=partial.new_zeros(())
        return dict(total=partial,partial=partial,visible_bce=zero,partial_center=lc,partial_local=ld,partial_global=lg,
                    visible_center=zero,visible_local=zero,visible_global=zero)
    bc=visible_domain_loss(center,allowed[:,4,4],valid[:,4,4],codes)
    bd=visible_domain_loss(dense,allowed,valid,codes)
    bg=visible_domain_loss(glob,ga,gv,gc,per_target=True)
    visible=bc+.25*bd+.25*bg
    return dict(total=partial+visible,partial=partial,visible_bce=visible,partial_center=lc,partial_local=ld,partial_global=lg,
                visible_center=bc,visible_local=bd,visible_global=bg)

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

def visible_smoke():
    """CPU leaf-logit math only: no cache, checkpoint, model forward or fit."""
    import importlib.util
    base=smoke();torch.set_num_threads(1);torch.manual_seed(20261023)
    a=torch.arange(1,64,dtype=torch.long);z=torch.randn(63,6,dtype=torch.float64,requires_grad=True)
    loss,known=visible_marginal_nll(z,a);p=z.softmax(1)[:,0]
    expected_known=(a==1)|((a&1)==0)
    assert torch.equal(known,expected_known)and int(known.sum())==32
    reference=torch.where(a==1,-p.log(),-torch.log1p(-p))
    assert torch.allclose(loss[known],reference[known],atol=1e-12,rtol=1e-12)
    assert torch.equal(loss[~known],torch.zeros_like(loss[~known]))
    gradient=torch.autograd.grad(loss.sum(),z)[0]
    assert torch.equal(gradient[~known],torch.zeros_like(gradient[~known]))
    extreme=torch.tensor([[-1000.,1000.,1000.,1000.,1000.,1000.],[1000.,-1000.,-1000.,-1000.,-1000.,-1000.]],requires_grad=True)
    value,_=visible_marginal_nll(extreme,torch.tensor([1,2]));g=torch.autograd.grad(value.sum(),extreme)[0]
    assert torch.isfinite(value).all()and torch.isfinite(g).all()
    # A completely invalid or ambiguous batch must contribute exact zero.
    zz=torch.randn(5,6,dtype=torch.float64,requires_grad=True);aa=torch.tensor([1,2,3,63,5]);cc=torch.tensor([0,0,1,1,2])
    empty=visible_domain_loss(zz,aa,torch.zeros(5,dtype=torch.bool),cc)
    assert empty==0 and torch.equal(torch.autograd.grad(empty,zz)[0],torch.zeros_like(zz))
    # Independent manual reductions, including an empty source/target.
    zz=torch.randn(4,6,2,2,dtype=torch.float64);aa=torch.tensor([[[1,2],[3,63]],[[4,1],[63,5]],[[1,6],[7,8]],[[63,63],[63,63]]]);vv=torch.ones_like(aa,dtype=torch.bool);vv[0,0,1]=False;cc=torch.tensor([0,0,1,2])
    vals,kk=visible_marginal_nll(zz,aa);eligible=vv&kk
    for per_target in (False,True):
        terms=[]
        for source in (0,1,2):
            members=[i for i in range(4)if cc[i]==source and bool(eligible[i].any())]
            if not members:continue
            if per_target:terms.append(torch.stack([vals[i][eligible[i]].mean()for i in members]).mean())
            else:terms.append(torch.cat([vals[i][eligible[i]]for i in members]).mean())
        expected=torch.stack(terms).mean();actual=visible_domain_loss(zz,aa,vv,cc,per_target=per_target)
        assert torch.equal(actual,expected)
    # Load only the historical trainer CODE for a direct lambda0 graph replay.
    oldpath=Path(__file__).resolve().parents[2]/'query_visibility_controlled_real_20261022_v66/model/fit_partial_head.py'
    spec=importlib.util.spec_from_file_location('v66_frozen_loss_reference',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    values=[torch.randn(7,6,dtype=torch.float64),torch.randn(7,6,9,9,dtype=torch.float64),torch.randn(3,6,64,64,dtype=torch.float64)]
    za=[x.clone().requires_grad_()for x in values];zb=[x.clone().requires_grad_()for x in values]
    aa=torch.randint(1,64,(7,9,9));vv=torch.rand(7,9,9)>.1;ga=torch.randint(1,64,(3,64,64));gv=torch.rand(3,64,64)>.1;cc=torch.tensor([0,0,0,1,1,2,2]);gc=torch.tensor([0,1,2])
    legacy=old.domain_loss(old.partial_nll(za[0],aa[:,4,4]),aa[:,4,4],vv[:,4,4],cc)+.25*old.domain_loss(old.partial_nll(za[1],aa),aa,vv,cc)+.25*old.domain_loss(old.partial_nll(za[2],ga),ga,gv,gc,per_target=True)
    current=objective_terms(*zb,aa,vv,cc,ga,gv,gc,visible_bce_weight=0.)['total']
    assert torch.equal(legacy,current)
    grad_old=torch.autograd.grad(legacy,za);grad_new=torch.autograd.grad(current,zb)
    for x,y in zip(grad_old,grad_new):assert torch.equal(x,y)
    # Lambda1 doubles an isolated known-visible center likelihood, with every
    # auxiliary cell intentionally uninformed; no optimizer update is made.
    pos=[torch.randn(2,6,dtype=torch.float64),torch.randn(2,6,9,9,dtype=torch.float64),torch.randn(2,6,64,64,dtype=torch.float64)]
    masks=torch.full((2,9,9),63,dtype=torch.long);masks[:,4,4]=1;valid=torch.ones_like(masks,dtype=torch.bool);global_masks=torch.full((2,64,64),63,dtype=torch.long);global_valid=torch.ones_like(global_masks,dtype=torch.bool);codes=torch.tensor([0,1])
    t0=objective_terms(*pos,masks,valid,codes,global_masks,global_valid,codes,visible_bce_weight=0.)
    t1=objective_terms(*pos,masks,valid,codes,global_masks,global_valid,codes,visible_bce_weight=1.)
    assert torch.equal(t1['total'],2*t0['total'])
    return dict(**base,known_visible_masks_all63_verified=True,known_visible_positive_masks=1,known_nonvisible_masks=31,unknown_odd_masks_excluded=31,
        marginal_BCE_reference_atol=1e-12,unknown_and_invalid_gradient_exact_zero=True,extreme_logit_loss_and_grad_finite=True,
        per_source_and_unique_target_manual_normalization_exact=True,lambda0_V66_scalar_and_all3_leaf_logit_gradient_bit_exact=True,
        lambda0_gradient_max_difference=0.,mock_logits_unchanged=True,isolated_visible_lambda1_loss_exactly_doubled=True,
        baseline_trainer_code_sha256=sha(oldpath),baseline_model_code_SHA_identity=sha(Path(__file__).with_name('query_global_amodal_head.py'))==sha(oldpath.with_name('query_global_amodal_head.py')),
        actual_cache_or_checkpoint_or_current_reads=0,model_forward_calls=0,optimizer_updates=0,GPU_calls=0)

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
    if args.visible_bce_weight!=1. or protocol['visible_bce_weight']!=1.:raise ValueError('V67 fitting is locked to the single lambda1 objective control')
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
        fitting_sources=sources,validation_sources=v_sources,sampling='stratified source, with replacement; rotate remainder allocation per update',
        visible_bce_weight=1.,visible_BCE_facts='allowed==1 positive; bit0 absent negative; other odd masks and invalid excluded',
        checkpoint_selection='Unchanged V66 macro informative-center partial NLL; visible BCE never used for checkpoint selection')
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
    best=float('inf');best_update=None;history=[];start=time.monotonic();losses=[];partial_losses=[];visible_losses=[]
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
        terms=objective_terms(center,dense,glob,allowed,valid,codes,ga,gv,gc,visible_bce_weight=args.visible_bce_weight)
        loss=terms['total']
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite partial likelihood')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()));partial_losses.append(float(terms['partial'].detach()));visible_losses.append(float(terms['visible_bce'].detach()))
        if update%args.val_every==0 or update==args.updates:
            metric,by,z=validation();improved=metric<best
            if improved:
                best,best_update=metric,update
                save_checkpoint(out/'BEST_PARTIAL_HEAD.pt',model,updates=update,selection_partial_NLL=best,selection_by_source=by,threshold=.5,visible_bce_weight=1.)
            row=dict(update=update,train_loss=float(np.mean(losses)),train_partial_NLL=float(np.mean(partial_losses)),train_visible_BCE=float(np.mean(visible_losses)),VAL_partial_NLL=metric,by_source=by,saved_best=improved,seconds=time.monotonic()-start)
            history.append(row);write(out/'PROGRESS.json',row);print(json.dumps(row),flush=True);losses=[];partial_losses=[];visible_losses=[]
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
    p.add_argument('--visible-bce-weight',type=float,default=1.,choices=(1.,))
    a=p.parse_args()
    if a.smoke:print(json.dumps(visible_smoke()))
    else:fit(a)
