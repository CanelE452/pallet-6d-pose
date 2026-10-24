"""One locked V75 source pretrain. No current/public-real/model-backbone input.

GT is loaded only for sampling, loss and diagnostics. Forward gets exactly six
input-cache tensors. Never silently resume/restart a run after optimizer step1.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,hashlib,importlib.util,json,random,shutil,sys,time
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'PRETRAIN_PROTOCOL.json'
PROTOCOL_SHA='ecd30714b7c3ec015f29a97dfb8ef7360aeb0cd121400ef2c397535556007b09'
SOURCES=('G38','P0','TEX')
FIELDS=('p3_roi','local_rgb96','local_valid96','target_rgb128','target_valid128','context42')
RUN=ROOT/'runs/SOURCE_PRETRAIN_V1'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(p.resolve()),sha256=sha(p))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def model_module(binding):
    check(binding);spec=importlib.util.spec_from_file_location('v75_frozen_native_model',binding['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def protocol():
    assert sha(PROTOCOL)==PROTOCOL_SHA
    p=load(PROTOCOL)
    for b in p['bindings'].values():check(b)
    assert p['optimizer']==dict(name='AdamW',lr=.0003,weight_decay=.001,betas=[.9,.999],updates=1000,clip_grad_norm=5.,LR_schedule='constant',initialization_seed=20261023)
    assert p['sampling']['batch_queries']==90 and p['sampling']['per_source_class']==10
    assert p['checkpoint_selection']['evaluate_every_updates']==50
    return p


def sample(pools,rng):
    return np.concatenate([rng.choice(pools[(s,c)],10,replace=True)for s in range(3)for c in range(3)])


def objective(pred,labels,primary,offset,square,weight=.1):
    """GT-only loss boundary: D0/E1/S2; F3/U4/invalid have no class loss.

    Location target is a virtual-reference proxy, never physical cue gold.
    Native9 centers=(col/row-4)/9; sigma=.75cells; valid cells only.
    """
    logits=pred['proxy_logits'];response=pred['reference_response_map'];valid=pred['response_input_valid'].bool()
    known=primary.bool()&(labels>=0)&(labels<3)
    zero=logits.sum()*0+response.sum()*0
    ce=F.cross_entropy(logits[known],labels[known]) if known.any() else zero
    eligible=known&(labels==0)&square.bool()&torch.isfinite(offset).all(1)
    location=zero
    if eligible.any():
        vm=valid[eligible];assert vm.flatten(1).any(1).all()
        axis=(torch.arange(9,device=response.device,dtype=response.dtype)-4)/9
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        delta=torch.stack((xx,yy),-1)[None]-offset[eligible,None,None,:].to(response.dtype)
        gaussian=-.5*((delta*9/.75)**2).sum(-1)
        target=F.softmax(gaussian.masked_fill(~vm,float('-inf')).flatten(1),1)
        logpred=F.log_softmax(response[eligible].masked_fill(~vm,-1e4).flatten(1),1)
        location=-(target*logpred).sum(1).mean()
    return dict(loss=ce+weight*location,primary_CE=ce,reference_location_CE=location,primary_count=int(known.sum()),auxiliary_count=int(eligible.sum()))


def generated():
    p=protocol();out=ROOT/'GENERATED_TRAINER_TESTS.json';assert not out.exists()
    torch.set_num_threads(1);torch.manual_seed(20261023)
    src=np.repeat(np.arange(3),12);y=np.tile(np.repeat(np.arange(3),4),3)
    pools={(s,c):np.flatnonzero((src==s)&(y==c))for s in range(3)for c in range(3)}
    rng=np.random.default_rng(20261023)
    for _ in range(64):
        ids=sample(pools,rng);assert len(ids)==90 and all(np.sum((src[ids]==s)&(y[ids]==c))==10 for s in range(3)for c in range(3))
    logits=torch.randn(8,3,requires_grad=True);response=torch.randn(8,9,9,requires_grad=True)
    valid=torch.ones(8,9,9,dtype=torch.bool);valid[:,:2,:2]=False
    labels=torch.tensor([0,1,2,0,3,4,0,0]);primary=torch.tensor([1,1,1,1,1,1,0,1],dtype=torch.bool)
    square=torch.tensor([1,1,1,0,1,1,1,1],dtype=torch.bool)
    offset=torch.zeros(8,2);offset[7]=torch.tensor([.22,-.13])
    result=objective(dict(proxy_logits=logits,reference_response_map=response,response_input_valid=valid),labels,primary,offset,square)
    assert result['primary_count']==5 and result['auxiliary_count']==2
    result['loss'].backward();assert torch.isfinite(logits.grad).all() and torch.isfinite(response.grad).all()
    assert not logits.grad[[4,5,6]].any() and not response.grad[[1,2,3,4,5,6]].any()
    assert not response.grad[:,:2,:2].any() and response.grad[[0,7]].abs().sum()>0
    # Known symmetric target peaks at center and its weighted mass sums to1.
    axis=(torch.arange(9)-4)/.75;g=-.5*(axis[:,None]**2+axis[None,:]**2);t=F.softmax(g.flatten(),0).reshape(9,9)
    assert int(t.argmax())==40 and abs(float(t.sum())-1)<1e-6
    m=model_module(p['bindings']['model_code']).NativeContextProxyHead()
    assert tuple(__import__('inspect').signature(m.forward).parameters)==FIELDS and sum(x.numel()for x in m.parameters())==41328
    write(out,dict(completed=True,PASS=True,code=bind(Path(__file__)),protocol=bind(PROTOCOL),balanced9strata_batches_tested=64,batch90_counts_per_stratum=10,
        Gaussian_sigma_cells=.75,aux_only_D_primary_local_square=True,E_S_F_U_unmatched_aux_gradient_zero=True,F_U_invalid_class_gradient_zero=True,
        invalid_response_cells_loss_gradient_zero=True,forward_fields=list(FIELDS),GT_forward_fields=0,source_arrays_RGB_features_read=0,model_source_forward=0,optimizer_updates=0,current_public_real_PnP=0))
    print(json.dumps(dict(generated=bind(out))),flush=True)


def freeze():
    p=protocol();proof=load(ROOT/'GENERATED_TRAINER_TESTS.json');assert proof['PASS'] and proof['code']==bind(Path(__file__))
    f=ROOT/'TRAINER_ENTRYLOCK.json';assert not f.exists()
    write(f,dict(schema='V75_ONE_SOURCE_FIT_ENTRYLOCK',approved=True,protocol=bind(PROTOCOL),trainer=bind(Path(__file__)),generated=bind(ROOT/'GENERATED_TRAINER_TESTS.json'),
        bindings=p['bindings'],run_directory=str(RUN),source_order=list(SOURCES),class_order=[0,1,2],sampler_order='sourceG38/P0/TEX thenclass0/1/2,10replacementqueries each; freshnpdefault_rng20261023 beforeupdate1.',
        evaluation='FP32torchcrossentropy perrow; aggregatefloat64per-nineheldoutstrata. Strictlessbest,earliestties; update0noteligible.',
        auxiliary_aggregation='mean over batchprimaryDandlocal-squareeligible;0ifnone,weight.1. No sourcephysical-cue GT claim.',
        runtime=dict(executable=str(Path(sys.executable).resolve()),torch=str(torch.__version__),CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],torch_deterministic_algorithms=False,CPU_threads=1),
        restart_after_first_optimizer_update=False,real_or_current_or_PnP_or_new_backbone_allowed=False))
    print(json.dumps(dict(entry=bind(f))),flush=True)


def load_source(p,device):
    manifest=load(p['bindings']['input_manifest']['path']);seal=load(p['bindings']['input_seal']['path']);assert manifest['completed'] and seal['PASS']
    for b in [seal['manifest'],seal['lineage'],*seal['all_chunk_SHA']]:check(b)
    names=['p3_roi','rgb96','crop_valid96','target_rgb128','target_valid128','context42','query_valid','L'];parts={k:[]for k in names}
    for d in manifest['chunks']:
        with np.load(d['file']['path'],allow_pickle=False) as z:
            for k in names:parts[k].append(z[k])
    data={k:np.ascontiguousarray(np.concatenate(v))for k,v in parts.items()};del parts
    with np.load(p['bindings']['GT']['path'],allow_pickle=False) as z:gt={k:z[k]for k in z.files}
    index=[json.loads(s)for s in Path(seal['lineage']['path']).read_text().splitlines()]
    assert len(index)==5120 and np.array_equal(gt['frame_index'],np.arange(5120)) and np.array_equal(gt['query_valid'],data['query_valid'])
    assert all(r['cache_row']==gt['cache_row'][i] and r['id']==gt['sample_id'][i] and r['source']==gt['source'][i] and r['partition']==gt['partition'][i]for i,r in enumerate(index))
    y=gt['proxy_state8'].reshape(-1).astype(np.int64);primary=gt['primary_DES_loss_mask'].reshape(-1)
    assert np.array_equal(primary,(gt['matched'][:,None]&data['query_valid']&(gt['proxy_state8']<3)&(gt['proxy_state8']>=0)).reshape(-1))
    assert np.all(gt['human_semantic_visibility8']==-1) and np.all(gt['target_pixel_presence8']==-1)
    source=np.repeat(np.array([SOURCES.index(x)for x in gt['source']]),8);train=np.repeat(gt['partition']=='train',8);heldout=np.repeat(gt['partition']=='heldout',8)
    assert np.sum(primary&train)==29353 and np.sum(primary&heldout)==7358 and not np.any(train&heldout)
    pools={(s,c):np.flatnonzero(primary&train&(source==s)&(y==c))for s in range(3)for c in range(3)}
    valpools={(s,c):np.flatnonzero(primary&heldout&(source==s)&(y==c))for s in range(3)for c in range(3)}
    assert all(len(x)>0 for x in list(pools.values())+list(valpools.values()))
    gpu=dict(p3_roi=torch.from_numpy(data['p3_roi'].reshape(-1,65,9,9)).to(device),local_rgb96=torch.from_numpy(data['rgb96'].reshape(-1,3,96,96)).to(device),
        local_valid96=torch.from_numpy(data['crop_valid96'].reshape(-1,1,96,96)).to(device),target_rgb128=torch.from_numpy(data['target_rgb128']).to(device),
        target_valid128=torch.from_numpy(data['target_valid128']).to(device),context42=torch.from_numpy(data['context42'].reshape(-1,42)).to(device))
    labels=torch.as_tensor(y,device=device);pm=torch.as_tensor(primary,device=device)
    offset=torch.as_tensor(gt['query_to_reference_offset_over_L8'].reshape(-1,2),dtype=torch.float32,device=device);square=torch.as_tensor(gt['reference_in_local_square8'].reshape(-1),device=device)
    meta=dict(y=y,primary=primary,source=source,train=train,heldout=heldout,index=index,L=np.repeat(data['L'],8),
        offset=gt['query_to_reference_offset_over_L8'].reshape(-1,2),square=gt['reference_in_local_square8'].reshape(-1),pools=pools,valpools=valpools)
    del data
    return gpu,dict(labels=labels,primary=pm,offset=offset,square=square),meta


def forward(model,gpu,ids):
    ix=torch.as_tensor(ids,dtype=torch.long,device=gpu['p3_roi'].device);frame=ix//8
    values={k:v[frame if k in ('target_rgb128','target_valid128') else ix]for k,v in gpu.items()}
    assert tuple(values)==FIELDS
    return model(**values)


def predict(model,gpu,ids,labels,full=False):
    model.eval();zs=[];ns=[];maps=[];ats=[];av=[]
    with torch.inference_mode():
        for i in range(0,len(ids),128):
            ix=ids[i:i+128];out=forward(model,gpu,ix);zs.append(out['proxy_logits'].cpu().numpy())
            valid_label=labels[torch.as_tensor(ix,device=labels.device)].clamp(0,2)
            ns.append(F.cross_entropy(out['proxy_logits'],valid_label,reduction='none').cpu().numpy())
            if full:maps.append(out['reference_response_map'].cpu().numpy());ats.append(out['reference_attention_weights'].cpu().numpy());av.append(out['input_available'].cpu().numpy())
    result=dict(logits=np.concatenate(zs),NLL_F32=np.concatenate(ns))
    if full:result.update(response=np.concatenate(maps),attention=np.concatenate(ats),input_available=np.concatenate(av))
    return result


def val_metric(pred,ids,meta):
    where=np.full(40960,-1,np.int64);where[ids]=np.arange(len(ids))
    by={f'{SOURCES[s]}|{c}':float(pred['NLL_F32'][where[ix]].astype(np.float64).mean())for (s,c),ix in meta['valpools'].items()}
    return float(np.mean(list(by.values()))),by


def class_metrics(logits,nll,ids,meta):
    y=meta['y'][ids];pred=np.argmax(logits[ids],1);cm=np.zeros((3,3),np.int64)
    np.add.at(cm,(y,pred),1);precision=[];recall=[]
    for c in range(3):
        precision.append(float(cm[c,c]/cm[:,c].sum())if cm[:,c].sum()else None);recall.append(float(cm[c,c]/cm[c].sum())if cm[c].sum()else None)
    shift=logits[ids].astype(float);prob=np.exp(shift-np.logaddexp.reduce(shift,axis=1)[:,None]);pv=prob[:,0]>=.5
    return dict(N=len(ids),confusion_rows_GT_cols_pred_DES=cm.tolist(),class_precision=precision,class_recall=recall,accuracy=float(np.trace(cm)/max(1,cm.sum())),mean_NLL_FP32=float(nll[ids].astype(float).mean()),
        binary_D_threshold=.5,D_recall=float(pv[y==0].mean())if np.any(y==0)else None,nonD_absence_recall=float((~pv[y!=0]).mean())if np.any(y!=0)else None)


def report(scores,meta):
    result={}
    for partition in ('train','heldout'):
        mask=meta['primary']&meta[partition];ids=np.flatnonzero(mask);by={SOURCES[s]:class_metrics(scores['logits'],scores['NLL_F32'],np.flatnonzero(mask&(meta['source']==s)),meta)for s in range(3)}
        aux=mask&(meta['y']==0)&meta['square']&np.isfinite(meta['offset']).all(1)
        peak=scores['attention'].reshape(-1,81).argmax(1);peakxy=np.stack((peak%9-4,peak//9-4),1)/9
        axis=(np.arange(9)-4)/9;yy,xx=np.meshgrid(axis,axis,indexing='ij');expected=(scores['attention'][...,None]*np.stack((xx,yy),-1)).sum((1,2))
        def quant(v):return {str(q):float(np.quantile(v,q))for q in [.5,.75,.9,.95,.99]}if len(v)else None
        err=np.linalg.norm(peakxy[aux]-meta['offset'][aux],axis=1)
        result[partition]=dict(overall=class_metrics(scores['logits'],scores['NLL_F32'],ids,meta),per_source=by,all_retained_slots=int(meta[partition].sum()),primary_eligible_slots=int(mask.sum()),
            primary_known_rate=float(mask.sum()/meta[partition].sum()),auxiliary_D_primary_local_square=int(aux.sum()),D_primary=int(np.sum(mask&(meta['y']==0))),
            D_primary_reference_outside_local_square=int(np.sum(mask&(meta['y']==0)&~meta['square'])),
            auxiliary_peak_error_over_L=quant(err),auxiliary_peak_error_native_px=quant(err*meta['L'][aux]),
            auxiliary_attention_mean_error_over_L=quant(np.linalg.norm(expected[aux]-meta['offset'][aux],axis=1)),
            proxy_scope='Sourcevirtualreference/ray/convexproxy, notphysicalsemanticgold; U/F/unmatched/invalidloss0.')
    return result


def save_scores(path,scores):
    np.savez_compressed(path,proxy_logits=scores['logits'].reshape(5120,8,3),NLL_primary_FP32=scores['NLL_F32'].reshape(5120,8),
        reference_response_map=scores['response'].reshape(5120,8,9,9),reference_attention_weights=scores['attention'].reshape(5120,8,9,9),input_available=scores['input_available'].reshape(5120,8))


def run():
    p=protocol();entry=load(ROOT/'TRAINER_ENTRYLOCK.json');assert entry['approved'] and entry['trainer']==bind(Path(__file__)) and entry['protocol']==bind(PROTOCOL)
    check(entry['generated']);assert not RUN.exists(),'Onefitonly; existing run must be preserved and resolved byroot, never silentlyrestarted'
    assert Path(sys.executable).resolve()==Path(p['runtime']['python']).resolve() and str(torch.__version__)==p['runtime']['torch'] and torch.cuda.is_available()
    RUN.mkdir(parents=True);(RUN/'code_snapshot').mkdir()
    for b in [entry['trainer'],entry['protocol'],p['bindings']['model_code']]:shutil.copy2(b['path'],RUN/'code_snapshot'/Path(b['path']).name)
    write(RUN/'RUN_STARTED.json',dict(protocol=bind(PROTOCOL),entry=bind(ROOT/'TRAINER_ENTRYLOCK.json'),trainer=bind(Path(__file__)),optimizer_updates=0,current_public_real_PnP_reads=0))
    torch.set_num_threads(1);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023);np.random.seed(20261023);random.seed(20261023)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False)  # protocol makes no exactGPUbackward-bit claim
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();gpu,gt,meta=load_source(p,'cuda:0')
    model=model_module(p['bindings']['model_code']).NativeContextProxyHead().cuda();assert sum(x.numel()for x in model.parameters())==41328
    trainpools=meta['pools'];validids=np.flatnonzero(meta['primary']&meta['heldout']);allids=np.arange(40960)
    poolcounts={f'{SOURCES[s]}|{c}':dict(TRAIN=len(trainpools[(s,c)]),heldout=len(meta['valpools'][(s,c)]))for s in range(3)for c in range(3)}
    preids=sample(trainpools,np.random.default_rng(20261023));model.train();out=forward(model,gpu,preids)
    preloss=objective(out,**{k:v[torch.as_tensor(preids,device='cuda')]for k,v in gt.items()});assert torch.isfinite(preloss['loss']);preloss['loss'].backward();model.zero_grad(set_to_none=True)
    write(RUN/'SOURCE_GPU_PREFLIGHT.json',dict(PASS=True,optimizer_updates=0,source9strata=poolcounts,loss=float(preloss['loss'].detach()),auxiliary_count=preloss['auxiliary_count'],peak_memory_bytes=torch.cuda.max_memory_allocated(),forward_fields=list(FIELDS),GT_forward=False))
    torch.save({k:v.detach().cpu()for k,v in model.state_dict().items()},RUN/'UNTRAINED_UPDATE0.pt')
    baseline=predict(model,gpu,allids,gt['labels'],full=True);save_scores(RUN/'BASELINE_UPDATE0_SOURCE_SCORES.npz',baseline)
    baseline_metrics=report(baseline,meta);baseline_val,_=val_metric(dict(NLL_F32=baseline['NLL_F32'][validids]),validids,meta)
    write(RUN/'BASELINE_UPDATE0_RESULTS.json',dict(update=0,eligible_for_checkpoint_selection=False,heldout9strata_macroNLL=baseline_val,metrics=baseline_metrics));del baseline
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001,betas=(.9,.999));rng=np.random.default_rng(20261023);history=[];best=float('inf');beststep=None
    for step in range(1,1001):
        ids=sample(trainpools,rng);ix=torch.as_tensor(ids,device='cuda');model.train();opt.zero_grad(set_to_none=True)
        result=objective(forward(model,gpu,ids),**{k:v[ix]for k,v in gt.items()});assert torch.isfinite(result['loss'])
        result['loss'].backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(grad);opt.step()
        if step==1:
            first=dict(update=1,loss=float(result['loss'].detach()),primary_CE=float(result['primary_CE'].detach()),location_CE=float(result['reference_location_CE'].detach()),auxiliary_count=result['auxiliary_count'],entry=bind(ROOT/'TRAINER_ENTRYLOCK.json'),current_public_real_PnP=0,restart_allowed=False)
            write(RUN/'FIRST_UPDATE.json',first);print(json.dumps(dict(FIRST_UPDATE=first)),flush=True)
        if step%50==0:
            val=predict(model,gpu,validids,gt['labels']);metric,by=val_metric(val,validids,meta);saved=metric<best
            if saved:best=metric;beststep=step;torch.save({k:v.detach().cpu()for k,v in model.state_dict().items()},RUN/'BEST_PROXY_HEAD.pt')
            rec=dict(update=step,train_loss=float(result['loss'].detach()),primary_CE=float(result['primary_CE'].detach()),reference_location_CE=float(result['reference_location_CE'].detach()),auxiliary_count=result['auxiliary_count'],heldout9strata_macroNLL=metric,stratumNLL=by,saved=saved,best_update=beststep,seconds=time.monotonic()-start)
            history.append(rec);write(RUN/'HISTORY.json',history);print(json.dumps(rec),flush=True)
    model.load_state_dict(torch.load(RUN/'BEST_PROXY_HEAD.pt',map_location='cuda',weights_only=True));model.eval().requires_grad_(False)
    freeze=dict(completed=True,phase='SOURCE_ONLY_PHASE1_PRETRAIN',optimizer_updates=1000,best_update=beststep,best_heldout9strata_macroNLL=best,checkpoint=bind(RUN/'BEST_PROXY_HEAD.pt'),
        model_code=p['bindings']['model_code'],trainer_code=bind(Path(__file__)),protocol=bind(PROTOCOL),entry=bind(ROOT/'TRAINER_ENTRYLOCK.json'),source_bindings=p['bindings'],
        parameter_count=41328,proxy_order=p['labels'],argmax3_fixed=True,binary_D_threshold=.5,update0_not_selection_eligible=True,
        actual_source_fit=True,current_or_public_real_fit_selection=False,new_backbone_inference=0,PnP=0,physical_native8_semantic_gold=False,real_transfer_solved=False)
    write(RUN/'MODEL_FREEZE.json',freeze)
    selected=predict(model,gpu,allids,gt['labels'],full=True);save_scores(RUN/'SELECTED_SOURCE_SCORES.npz',selected);metrics=report(selected,meta)
    write(RUN/'SOURCE_RESULTS.json',dict(completed=True,source_freeze=bind(RUN/'MODEL_FREEZE.json'),selected_update=beststep,selection_heldout_macroNLL=best,baseline0=bind(RUN/'BASELINE_UPDATE0_RESULTS.json'),selected_metrics=metrics,
        strata_counts=poolcounts,scope='Phase1conditionalvirtualreferenceproxy only; historicalupstreamheldout; notcurrent orpublicreal transfer proof.',source_GT_forward=False,current_public_real_reads_or_forward=0,new_backbone_PnP=0))
    write(RUN/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=1000,best_update=beststep,model_freeze=bind(RUN/'MODEL_FREEZE.json'),history=bind(RUN/'HISTORY.json'),source_results=bind(RUN/'SOURCE_RESULTS.json'),baseline0_scores=bind(RUN/'BASELINE_UPDATE0_SOURCE_SCORES.npz'),selected_scores=bind(RUN/'SELECTED_SOURCE_SCORES.npz'),
        peak_GPU_memory_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start,current_public_real_reads_or_forward=0,new_backbone_PnP=0))
    print(json.dumps(dict(COMPLETE=bind(RUN/'FIT_COMPLETE.json'),freeze=bind(RUN/'MODEL_FREEZE.json'),best_update=beststep,bestNLL=best)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['generated','freeze','run']);args=parser.parse_args();{'generated':generated,'freeze':freeze,'run':run}[args.stage]()
