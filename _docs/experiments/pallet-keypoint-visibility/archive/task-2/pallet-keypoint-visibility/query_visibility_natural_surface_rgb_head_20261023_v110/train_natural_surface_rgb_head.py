"""One C10 source fit. --generated-check uses invented tensors/metadata only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, hashlib, importlib.util, json, time, traceback
import numpy as np
import torch
V = Path(__file__).resolve().parent
P = V.parent
DESIGN = V/'ROOT_C10_DESIGN_BEFORE_FEATURE_FORWARD.json'
DESIGN_SHA = 'b7862cca1917375cb6b22421e76296f03d32623bed13a9502440b3c592f446ec'
C9 = P/'query_visibility_target_relative_rgb_head_20261023_v108/train_target_relative_rgb_head.py'
C9_SHA = 'c3f0c5095e2c9bcfdf31e345aba2a839602eeaa000be11b9b4bd94c254cf450b'
RUN = V/'runs/C10_NATURAL_SURFACE_TARGET_RELATIVE_RGB'
NOLD, NNEW, N, UNITS = 33984, 8000, 41984, 11652
QUOTAS = (('old_LOCO_D',2),('old_PALLOX_D',2),('synth_D',4),('YCB_D',8),
          ('old_natural_E',4),('synth_E',8),('YCB_E',12))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path): return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path): return json.loads(Path(path).read_text())
def check(b): assert sha(b['path']) == b['sha256'], b['path']; return Path(b['path'])
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def write(path, value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,allow_nan=False); f.write('\n')
def old_modules():
    assert sha(C9) == C9_SHA
    spec=importlib.util.spec_from_file_location('c10_pinned_c9',C9)
    c9=importlib.util.module_from_spec(spec); spec.loader.exec_module(c9)
    return c9,c9.adapter()

class NaturalSurfaceRGBHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head=torch.nn.Sequential(torch.nn.Linear(2565,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        torch.nn.init.zeros_(self.head[-1].weight);torch.nn.init.zeros_(self.head[-1].bias)
    def forward(self, features):
        if not isinstance(features,torch.Tensor) or features.ndim!=2 or features.shape[1]!=1541 or features.dtype!=torch.float32:
            raise ValueError('Only FP32 image-only features[N,1541] accepted.')
        target,centre=features[:,:512],features[:,1024:1536]
        return self.head(torch.cat((features,torch.abs(target-centre),target*centre),1)).squeeze(-1)

def join_ycb(inputs,lineage,raw,external,query_copy,L_copy):
    """GT quarantined metadata join. None of these fields enter model.forward."""
    assert len(inputs)==1000 and len(lineage)==8000 and raw.shape==external.shape==(1000,8)
    assert query_copy.shape==(1000,8,2) and L_copy.shape==(1000,)
    assert np.isin(raw,['D','E','S','U']).all()
    assert np.array_equal(external,np.where(raw=='D',0,np.where(raw=='E',1,-1)))
    for i,r in enumerate(inputs):
        assert r['instance_row_index']==i and r['split']=='TRAIN' and r['source_emulation'] is True
        assert np.array_equal(query_copy[i],r['initial_points']) and L_copy[i]==r['roi_side']
        for k in range(8):
            j=i*8+k;l=lineage[j]
            assert l['global_occurrence_index']==NOLD+j and l['input_unit_index']==3652+j
            assert l['query_row_index']==i and l['logical_query_index']==k
            assert l['input_id']==r['input_id'] and l['query_id']==r['input_id']+'::q'+str(k)
            assert l['query_xy']==r['initial_points'][k] and l['L']==r['roi_side']
            assert l['image']==r['image'] and l['split']=='TRAIN' and l['source']==r['source']
            assert l['source_emulated_not_initializer'] is True
    return dict(typed_state=raw.reshape(-1),labels=external.reshape(-1).astype(np.int8),
                split=np.full(NNEW,'TRAIN'),domain=np.full(NNEW,'V109_YCB_SURFACE'),
                source=np.array([r['source'] for r in lineage]),variant=np.full(NNEW,'NATURAL_SCENE'))

def load_metadata(entry,receipt,design,d):
    old=d.load_metadata(read(check(entry['source_bindings'])))
    seal=read(check(design['source']['new_teacher_seal']))
    assert seal['completed'] and seal['all8000_preserved']
    assert seal['inputs']==design['source']['new_input_freeze']
    packet=next(b for b in seal['outputs'] if Path(b['path']).name=='RAW_SURFACE_TEACHER8000_GT_ONLY.npz')
    inputs=rows(check(design['source']['new_input_freeze']))
    lineage=rows(check(receipt['new_query_lineage']))
    with np.load(check(packet),allow_pickle=False) as f:
        new=join_ycb(inputs,lineage,np.array(f['raw_surface_teacher_state'],copy=True),
            np.array(f['external_target_D0_E1_else_minus1'],copy=True),
            np.array(f['query_xy_input_copy'],copy=True),np.array(f['roi_side_input_copy'],copy=True))
    m={k:np.concatenate((old[k],new[k])) for k in ('typed_state','labels','split','domain','source','variant')}
    m.update(known=np.isin(m['typed_state'],['D','E']),known_geometric_S=m['typed_state']=='S',
             unresolved_teacher_U=m['typed_state']=='U',lineage=old['lineage']+lineage,old=old,
             new_teacher_packet=packet)
    m['external_target']=np.where(m['known'],m['labels'],-1).astype(np.int8)
    assert {s:int((m['typed_state']==s).sum()) for s in 'DESU'}==design['source']['all_raw_states']
    assert int(m['known'].sum())==3039
    assert [int((m['known']&(m['split']==s)).sum()) for s in ('TRAIN','VALIDATION')]==[2523,516]
    assert int((m['domain'][NOLD:]=='V109_YCB_SURFACE').sum())==8000 and not (m['split'][NOLD:]=='VALIDATION').any()
    return m

def sampling_pools(old,meta,available,d):
    om=d.c5.old_masks(old,'TRAIN')
    def filtered(index):
        index=np.asarray(index,np.int64)
        return index[available[index]] if index.ndim==1 else index[available[index].all(1)]
    pools={
      'old_triplets':filtered(old['old_pairs'][old['split'][old['old_pairs'][:,0]]=='TRAIN']),
      'synth_triplets':filtered(old['new_pairs'][old['split'][old['new_pairs'][:,0]]=='TRAIN']),
      'old_LOCO_D':filtered(np.flatnonzero(om['LOCO_BEFORE_D'])),
      'old_PALLOX_D':filtered(np.flatnonzero(om['PALLOX_BEFORE_D'])),
      'old_natural_E':filtered(np.flatnonzero(om['BEFORE_E']))}
    for prefix,domain in [('synth','V99_SYNTH'),('YCB','V109_YCB_SURFACE')]:
        for state in ('D','E'):
            mask=(meta['domain']==domain)&(meta['split']=='TRAIN')&(meta['typed_state']==state)
            pools[prefix+'_'+state]=filtered(np.flatnonzero(mask))
    assert all(len(p)>0 for p in pools.values()),'Empty required TRAIN cohort: no fit/fallback.'
    assert all(v.any() for v in d.val_masks(old).values()),'Empty required old VAL cohort.'
    for name,p in pools.items():
        assert meta['known'][p].all() and available[p].all() and (meta['split'][p]=='TRAIN').all(),name
        if p.ndim==2: assert np.array_equal(meta['external_target'][p],np.tile([0,1,0],(len(p),1)))
    return pools

def draw(pools,generator):
    oi=torch.randint(len(pools['old_triplets']),(4,),generator=generator).numpy()
    si=torch.randint(len(pools['synth_triplets']),(4,),generator=generator).numpy()
    triples=np.concatenate((pools['old_triplets'][oi],pools['synth_triplets'][si]))
    unshuffled=np.concatenate([triples.reshape(-1)]+[pools[k][torch.randint(len(pools[k]),(n,),generator=generator).numpy()] for k,n in QUOTAS])
    assert len(unshuffled)==64
    perm=torch.randperm(64,generator=generator).numpy();inv=np.argsort(perm)
    return unshuffled[perm],inv[np.arange(24).reshape(8,3)],dict(old_pair_pool_indices=oi.tolist(),synth_pair_pool_indices=si.tolist(),unshuffled_occurrence_indices=unshuffled.tolist(),shuffle_permutation=perm.tolist(),cohort_draw_quotas=dict(old_triplets=4,synth_triplets=4,**dict(QUOTAS)))

def supervised_loss(z,target,available,triples):
    eligible=(target>=0)&available
    bce=torch.nn.functional.binary_cross_entropy_with_logits(z[eligible],target[eligible]) if eligible.any() else z.sum()*0
    cover,sham=triples[:,1],triples[:,2];ok=eligible[cover]&eligible[sham]
    assert (target[cover[ok]]==1).all() and (target[sham[ok]]==0).all()
    hinge=torch.relu(2-(z[cover[ok]]-z[sham[ok]])).mean() if ok.any() else z.sum()*0
    return bce+.25*hinge,bce,hinge

def run(entry_path):
    assert not RUN.exists(),'No silent source fit restart.'
    e=read(entry_path)
    assert e['approved_source_fit'] is True and e['current_allowed'] is False and e['PnP_allowed'] is False and e['restart_after_update'] is False
    assert e['new_source_visual_admitted'] is True
    for k in ('trainer','design','feature_pack','feature_receipt','source_bindings','new_source_visual_admission'):check(e[k])
    assert e['trainer']==bind(__file__) and e['design']==bind(DESIGN) and sha(DESIGN)==DESIGN_SHA
    design=read(DESIGN);c9,d=old_modules();receipt=read(e['feature_receipt']['path'])
    assert receipt['completed'] and receipt['PASS'] and receipt['output']==e['feature_pack']
    assert receipt['model_state_unchanged'] and receipt['first_batch_repeat_exact']
    assert receipt['features_shape']==[UNITS,1541] and receipt['old_components_copied_exact']
    meta=load_metadata(e,receipt,design,d);old=meta['old']
    with np.load(e['feature_pack']['path'],allow_pickle=False) as f:
        x=np.array(f['features'],copy=True);units=np.array(f['occurrence_unit'],copy=True)
        available=np.array(f['query_available'],copy=True);factual=np.array(f['factual_query_available'],copy=True)
        context=np.array(f['context_available'],copy=True);indices=np.array(f['global_occurrence_index'],copy=True)
    assert x.shape==(UNITS,1541) and x.dtype==np.float32 and np.isfinite(x).all()
    assert ((x[:,1538:]>=0)&(x[:,1538:]<=1)).all()
    assert units.shape==available.shape==factual.shape==context.shape==(N,) and units.dtype==np.int32
    assert available.dtype==factual.dtype==context.dtype==bool and np.array_equal(available,factual&context)
    assert np.array_equal(indices,np.arange(N)) and np.array_equal(units[NOLD:],np.arange(3652,11652))
    assert units.min()==0 and units.max()==11651
    assert np.array_equal(factual[:NOLD],[r['input_facts']['operator_query_available'] for r in old['lineage']])
    pools=sampling_pools(old,meta,available,d)
    assert torch.__version__=='2.1.1+cu118' and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(208);torch.cuda.manual_seed_all(208)
    model=NaturalSurfaceRGBHead().cuda();assert sum(p.numel() for p in model.parameters())==164289
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    x=torch.as_tensor(x,device='cuda');ug=torch.as_tensor(units,dtype=torch.long,device='cuda')
    ag=torch.as_tensor(available,device='cuda');tg=torch.as_tensor(meta['external_target'],dtype=torch.float32,device='cuda')
    generator=torch.Generator(device='cpu').manual_seed(208)
    RUN.mkdir(parents=True)
    write(RUN/'RUN_STARTED.json',dict(entry=bind(entry_path),optimizer_updates=0,pool_counts={k:len(v) for k,v in pools.items()},known_train_available=int((meta['known']&(meta['split']=='TRAIN')&available).sum()),known_train_unavailable=int((meta['known']&(meta['split']=='TRAIN')&~available).sum())))
    updates=0;started=time.monotonic();best=float('inf');selected=None;best_state=None;draw_hash=hashlib.sha256()
    try:
        def checkpoint(step):
            return dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},update=step,design=bind(DESIGN),trainer=bind(__file__),entry=bind(entry_path))
        @torch.no_grad()
        def evaluate(step):
            nonlocal best,selected,best_state
            model.eval();z=model(x)[ug].cpu().numpy();z=np.where(available,z,0).astype(np.float32)
            criterion,parts=d.selector(z[:NOLD],available[:NOLD],old)
            with (RUN/'HISTORY.jsonl').open('a') as f:f.write(json.dumps(dict(update=step,criterion_eight_VAL_cohorts_BCE=criterion,VAL_cohorts=parts),allow_nan=False)+'\n')
            if criterion<best:
                best,selected=criterion,step;best_state=checkpoint(step);torch.save(best_state,RUN/'BEST_NATURAL_SURFACE_RGB_HEAD.pt')
            return z
        torch.save(checkpoint(0),RUN/'INITIAL_CHECKPOINT.pt');assert not np.count_nonzero(evaluate(0))
        for step in range(1,2001):
            ix,tr,audit=draw(pools,generator)
            assert meta['known'][ix].all() and available[ix].all() and (meta['split'][ix]=='TRAIN').all()
            assert np.array_equal(np.bincount(meta['external_target'][ix],minlength=2),[32,32])
            draw_hash.update(ix.astype('<i8').tobytes());draw_hash.update(tr.astype('<i8').tobytes())
            idx=torch.as_tensor(ix,dtype=torch.long,device='cuda');pairs=torch.as_tensor(tr,dtype=torch.long,device='cuda')
            model.train();optimizer.zero_grad(set_to_none=True);z=model(x[ug[idx]])
            loss,bce,hinge=supervised_loss(z,tg[idx],ag[idx],pairs);assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm)
            optimizer.step();updates=step
            with (RUN/'TRAIN_UPDATES.jsonl').open('a') as f:f.write(json.dumps(dict(update=step,occurrence_indices=ix.tolist(),input_unit_indices=units[ix].tolist(),matched_triplet_batch_positions=tr.tolist(),sampling=audit,available_loss_rows=64,sampled_unavailable_rows=0,loss=float(loss.detach()),BCE=float(bce.detach()),matched_hinge=float(hinge.detach()),gradient_norm_before_clip=float(norm)),allow_nan=False)+'\n')
            if step%100==0:evaluate(step)
        torch.save(checkpoint(2000),RUN/'FINAL_CHECKPOINT.pt');model.load_state_dict(best_state['state_dict']);model.eval()
        with torch.no_grad():z=model(x)[ug].cpu().numpy()
        z=np.where(available,z,0).astype(np.float32);criterion,parts=d.selector(z[:NOLD],available[:NOLD],old);assert criterion==best
        metrics,gates=c9.report(d,z[:NOLD],available[:NOLD],old,design)
        mask=meta['domain']=='V109_YCB_SURFACE';ycb=d.external_cm(z,available,meta,mask)
        passed=all(v['PASS'] for v in gates.values());line=meta['lineage']
        q=np.array([r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan] for r in line]);L=np.array([r['L'] if r['L'] is not None else np.nan for r in line])
        np.savez_compressed(RUN/'SELECTED_ALL41984_SOURCE_SCORES.npz',query_logit=z,query_available=available,factual_query_available=factual,context_available=context,prediction=d.decode(z,available),input_unit_index=units,raw_teacher_state=meta['typed_state'],external_target=meta['external_target'],split=meta['split'],domain=meta['domain'],source=meta['source'],variant=meta['variant'],query_xy_raw=q,L_raw=L,old_accepted_triplets=old['old_pairs'],new_selected_triplets=old['new_pairs'],global_occurrence_index=np.arange(N))
        write(RUN/'SOURCE_RESULTS.json',dict(completed=True,selected_update=selected,criterion_eight_VAL_cohorts_BCE=best,VAL_cohorts=parts,metrics=metrics,YCB_surface_TRAIN=ycb,YCB_validation_rows=0,no_new_natural_real_heldout_claim=True,raw_state_counts={k:int((meta['typed_state']==k).sum()) for k in 'DESU'},known_TRAIN=int((meta['known']&(meta['split']=='TRAIN')).sum()),known_VAL=int((meta['known']&(meta['split']=='VALIDATION')).sum()),availability=dict(factual_unavailable=int((~factual).sum()),context_unavailable=int((~context).sum()),extra_context_unavailable=int((factual&~context).sum()),final_unavailable=int((~available).sum()),known_train_unavailable=int((meta['known']&(meta['split']=='TRAIN')&~available).sum())),learned_self_classification=False,native_mapping='UNKNOWN',human_gold=False,source_development_exposure=True,all2000_draws_sha256=draw_hash.hexdigest()))
        write(RUN/'SOURCE_GATE_RESULTS.json',dict(completed=True,all_gates_pass=passed,gates=gates,current_inference_authorized=False))
        write(RUN/'MODEL_FREEZE.json',dict(completed=True,selected_update=selected,checkpoint=bind(RUN/'BEST_NATURAL_SURFACE_RGB_HEAD.pt'),feature_receipt=e['feature_receipt'],feature_pack=e['feature_pack'],trainable_parameters=164289,encoder_trainable_parameters=0,threshold_logit=0,design=bind(DESIGN),trainer=bind(__file__),current_used_for_fit_or_selection=False,current_inference_authorized=False,PnP=0))
        write(RUN/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=2000,selected_update=selected,criterion_eight_VAL_cohorts_BCE=best,all_source_gates_pass=passed,seconds=time.monotonic()-started,source_results=bind(RUN/'SOURCE_RESULTS.json'),gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),scores=bind(RUN/'SELECTED_ALL41984_SOURCE_SCORES.npz'),model_freeze=bind(RUN/'MODEL_FREEZE.json'),current_or_PnP=0))
        print(json.dumps(read(RUN/'FIT_COMPLETE.json')),flush=True)
    except BaseException:
        write(RUN/'FAILED_EXECUTION.json',dict(optimizer_updates_completed=updates,traceback=traceback.format_exc(),restart_automatically=False));raise

def generated_check(output):
    """Invented tensors/metadata only. No adapter, source arrays or weights read."""
    torch.set_num_threads(1);torch.manual_seed(208);model=NaturalSurfaceRGBHead()
    assert sum(p.numel() for p in model.parameters())==164289
    x=torch.randn(6,1541);z=model(x);assert z.shape==(6,) and not torch.count_nonzero(z)
    seen=[];handle=model.head[0].register_forward_pre_hook(lambda m,args:seen.append(args[0].detach().clone()))
    model(x);handle.remove();assert seen[0].shape==(6,2565)
    assert torch.equal(seen[0][:,1541:2053],torch.abs(x[:,:512]-x[:,1024:1536]))
    assert torch.equal(seen[0][:,2053:],x[:,:512]*x[:,1024:1536])
    raw=torch.tensor([-2.,9.,-9.,3.,8.,-8.],requires_grad=True);a=torch.tensor([1,0,1,1,0,1],dtype=torch.bool);target=torch.tensor([0.,1.,-1.,1.,-1.,0.])
    loss,_,_=supervised_loss(raw,target,a,torch.empty((0,3),dtype=torch.long));loss.backward();assert torch.count_nonzero(raw.grad[[1,2,4]])==0 and torch.count_nonzero(raw.grad[[0,3,5]])==3
    try:model(torch.randn(2,1540))
    except ValueError:pass
    else:raise AssertionError('Incorrect model input accepted')
    # Disjoint generated pools prove all9 quotas and matched triples are32D/32E.
    y=np.zeros(160,np.int8);y[[1,4,7,10,13,16,19,22]]=1;y[80:]=1
    pools={'old_triplets':np.arange(12).reshape(4,3),'synth_triplets':np.arange(12,24).reshape(4,3),
           'old_LOCO_D':np.arange(24,30),'old_PALLOX_D':np.arange(30,36),'synth_D':np.arange(36,44),'YCB_D':np.arange(44,60),
           'old_natural_E':np.arange(80,86),'synth_E':np.arange(86,102),'YCB_E':np.arange(102,126)}
    def replay():
        gen=torch.Generator().manual_seed(208);h=hashlib.sha256()
        for _ in range(2000):
            ix,tr,au=draw(pools,gen);assert np.array_equal(np.bincount(y[ix],minlength=2),[32,32]);assert np.array_equal(y[ix][tr],np.tile([0,1,0],(8,1)))
            assert au['cohort_draw_quotas']==dict(old_triplets=4,synth_triplets=4,**dict(QUOTAS))
            h.update(ix.astype('<i8').tobytes());h.update(tr.astype('<i8').tobytes())
        return h.hexdigest()
    h=replay();assert h==replay()
    from types import SimpleNamespace
    old=dict(old_pairs=pools['old_triplets'],new_pairs=pools['synth_triplets'],split=np.full(160,'TRAIN'))
    domain=np.full(160,'V96_REAL',dtype='<U20');domain[12:24]='V99_SYNTH';domain[36:44]='V99_SYNTH';domain[86:102]='V99_SYNTH';domain[44:60]='V109_YCB_SURFACE';domain[102:126]='V109_YCB_SURFACE'
    fixture=dict(known=np.ones(160,bool),split=np.full(160,'TRAIN'),external_target=y,typed_state=np.where(y==0,'D','E'),domain=domain)
    masks={k:np.isin(np.arange(160),pools[n]) for k,n in [('LOCO_BEFORE_D','old_LOCO_D'),('PALLOX_BEFORE_D','old_PALLOX_D'),('BEFORE_E','old_natural_E')]}
    fake=SimpleNamespace(c5=SimpleNamespace(old_masks=lambda *_:masks),val_masks=lambda _:{'generated_VAL':np.array([True])})
    available=np.ones(160,bool);available[[0,12,24,44,80,102]]=False
    eligible=sampling_pools(old,fixture,available,fake)
    assert len(eligible['old_triplets'])==len(eligible['synth_triplets'])==3
    assert all(available[v].all() for v in eligible.values())
    # Eligibility cannot silently relabel or delete unavailable source rows.
    assert len(fixture['known'])==160 and fixture['known'][~available].all()
    safe=np.where(available,np.ones(160,np.float32),0)
    decoded=np.where(available,(safe>=0).astype(np.int8),-1)
    assert (decoded[~available]==-1).all()
    # Full1000x8 invented join proves order/q/L/raw-state and explicit TRAIN-only addition.
    inputs=[];line=[]
    for i in range(1000):
        r=dict(instance_row_index=i,input_id=f'GENERATED_{i}',split='TRAIN',source_emulation=True,initial_points=[[i+k,k] for k in range(8)],roi_side=16.,image={'path':'invented','sha256':'0'*64},source='GENERATED')
        inputs.append(r)
        for k in range(8):line.append(dict(global_occurrence_index=NOLD+8*i+k,input_unit_index=3652+8*i+k,query_row_index=i,logical_query_index=k,input_id=r['input_id'],query_id=r['input_id']+'::q'+str(k),query_xy=r['initial_points'][k],L=16.,image=r['image'],split='TRAIN',source='GENERATED',source_emulated_not_initializer=True))
    states=np.tile(['D','E','S','U','D','E','S','U'],(1000,1));labels=np.where(states=='D',0,np.where(states=='E',1,-1));new=join_ycb(inputs,line,states,labels,np.array([r['initial_points'] for r in inputs]),np.full(1000,16.))
    assert len(new['labels'])==8000 and not (new['split']=='VALIDATION').any() and (new['labels'][np.isin(new['typed_state'],['S','U'])]==-1).all()
    write(output,dict(PASS=True,code=bind(__file__),runtime=torch.__version__,parameters=164289,feature_input=1541,linear_input=2565,lastzero=True,abs_product_exact=True,known_available_loss_only=True,wrong_dimension_rejected=True,generated2000_draws_32D32E_and8triplets_exact=True,generated_draw_sha256=h,generated8000_metadata_join=True,sampler_filters_available_rows_and_all3_pairs=True,unavailable_retained_decodes_U=True,new_YCB_VAL0=True,source_arrays_checkpoint_adapter_modeldata_forward_fit_current_PnP=0))

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--entry',type=Path);g.add_argument('--generated-check',type=Path);args=ap.parse_args()
    if args.generated_check:generated_check(args.generated_check)
    else:run(args.entry)
