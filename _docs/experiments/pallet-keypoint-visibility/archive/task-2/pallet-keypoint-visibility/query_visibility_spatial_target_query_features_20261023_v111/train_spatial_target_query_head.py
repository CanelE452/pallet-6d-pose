"""C11 single-source fit using pinned C10 metadata/sampling/loss and old report math."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
V=Path(__file__).resolve().parent;P=V.parent
DESIGN=V/'ROOT_C11_DESIGN_BEFORE_SOURCE_FORWARD.json'
DESIGN_SHA='f60d3d84ecc0fd3305864aae7e3f834969e1170e857f71541f8f900c06382daa'
C10=P/'query_visibility_natural_surface_rgb_head_20261023_v110/train_natural_surface_rgb_head.py'
C10_SHA='3baff03b0c73b17fff8cbad4f1ba96b9c5b134bc50538b96fb0a27827ddfc6e0'
EXPECTED_DRAW_SHA='8d6ec2eecbdd0937e710173328cea8e0d4211d21b11d722faea5d5c6fb8cf680'
RUN=V/'runs/C11_DINOV2_SPATIAL_TARGET_QUERY'
NOLD,N,UNITS=33984,41984,11652

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def pinned_c10():
    assert sha(C10)==C10_SHA
    s=importlib.util.spec_from_file_location('c11_exact_c10',C10);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class SpatialTargetQueryHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head=torch.nn.Sequential(torch.nn.Linear(3464,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        torch.nn.init.zeros_(self.head[-1].weight);torch.nn.init.zeros_(self.head[-1].bias)
    def forward(self,features):
        if not isinstance(features,torch.Tensor) or features.ndim!=2 or features.shape[1]!=1928 or features.dtype!=torch.float32:
            raise ValueError('Only FP32 frozen image features[N,1928] accepted.')
        T,Q,C,P=features[:,:384],features[:,384:768],features[:,1152:1536],features[:,1536:1920]
        x=torch.cat((features,torch.abs(T-C),T*C,torch.abs(Q-P),Q*P),1)
        return self.head(x).squeeze(-1)

def run(entry_path):
    assert not RUN.exists(),'No silent restart after source fit.'
    e=read(entry_path)
    assert e['approved_source_fit'] is True and e['current_allowed'] is False and e['PnP_allowed'] is False and e['restart_after_update'] is False
    assert e['new_source_visual_admitted'] is True
    for k in ('trainer','design','feature_pack','feature_receipt','source_bindings','new_source_visual_admission'):check(e[k])
    assert e['trainer']==bind(__file__) and e['design']==bind(DESIGN) and sha(DESIGN)==DESIGN_SHA
    design=read(DESIGN);c10=pinned_c10();c9,d=c10.old_modules()
    c10_design=read(check(design['source']['C10_design']));c10_receipt=read(check(design['source']['C10_feature_receipt']))
    receipt=read(e['feature_receipt']['path'])
    assert receipt['completed'] and receipt['PASS'] and receipt['output']==e['feature_pack']
    assert receipt['model_state_unchanged'] and receipt['first_batch_repeat_exact'] and receipt['features_shape']==[UNITS,1928]
    assert receipt['new_query_lineage']==c10_receipt['new_query_lineage']
    assert e['new_source_visual_admission']==design['source']['teacher_admission']
    meta=c10.load_metadata(e,c10_receipt,c10_design,d);old=meta['old']
    with np.load(e['feature_pack']['path'],allow_pickle=False) as f:
        features=np.array(f['features'],copy=True);units=np.array(f['occurrence_unit'],copy=True)
        available=np.array(f['query_available'],copy=True);factual=np.array(f['factual_query_available'],copy=True)
        context=np.array(f['context_available'],copy=True);indices=np.array(f['global_occurrence_index'],copy=True)
    assert features.shape==(UNITS,1928) and features.dtype==np.float32 and np.isfinite(features).all()
    assert ((features[:,1924:]>=0)&(features[:,1924:]<=1)).all()
    assert np.isin(features[:,1927],[0,1]).all()
    assert units.shape==available.shape==factual.shape==context.shape==(N,) and units.dtype==np.int32
    assert available.dtype==factual.dtype==context.dtype==bool and np.array_equal(available,factual&context)
    assert np.array_equal(indices,np.arange(N)) and np.array_equal(units[NOLD:],np.arange(3652,11652))
    # Exact C10 flags/map; Q-outside is a feature fact, never a new availability gate.
    with np.load(check(c10_receipt['output']),allow_pickle=False) as f:
        for k,a in [('occurrence_unit',units),('query_available',available),('factual_query_available',factual),('context_available',context),('global_occurrence_index',indices)]:assert np.array_equal(a,f[k]),k
    pools=c10.sampling_pools(old,meta,available,d)
    assert torch.__version__=='2.1.1+cu118' and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(211);torch.cuda.manual_seed_all(211)
    model=SpatialTargetQueryHead().cuda();assert sum(p.numel() for p in model.parameters())==221825
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    q_inside=features[:,1927][units].astype(bool)
    x=torch.as_tensor(features,device='cuda');ug=torch.as_tensor(units,dtype=torch.long,device='cuda')
    ag=torch.as_tensor(available,device='cuda');tg=torch.as_tensor(meta['external_target'],dtype=torch.float32,device='cuda')
    generator=torch.Generator(device='cpu').manual_seed(208)
    RUN.mkdir(parents=True)
    write(RUN/'RUN_STARTED.json',dict(entry=bind(entry_path),optimizer_updates=0,pool_counts={k:len(a) for k,a in pools.items()},same_C10_flags_and_units=True,Q_outside_additional_U=0))
    updates=0;started=time.monotonic();best=float('inf');selected=None;best_state=None;draw_hash=hashlib.sha256()
    try:
        def checkpoint(step):
            return dict(state_dict={k:a.detach().cpu().clone() for k,a in model.state_dict().items()},update=step,design=bind(DESIGN),trainer=bind(__file__),entry=bind(entry_path))
        @torch.no_grad()
        def evaluate(step):
            nonlocal best,selected,best_state
            model.eval();z=model(x)[ug].cpu().numpy();z=np.where(available,z,0).astype(np.float32)
            criterion,parts=d.selector(z[:NOLD],available[:NOLD],old)
            with (RUN/'HISTORY.jsonl').open('a') as f:f.write(json.dumps(dict(update=step,criterion_eight_VAL_cohorts_BCE=criterion,VAL_cohorts=parts),allow_nan=False)+'\n')
            if criterion<best:
                best,selected=criterion,step;best_state=checkpoint(step);torch.save(best_state,RUN/'BEST_SPATIAL_TARGET_QUERY_HEAD.pt')
            return z
        torch.save(checkpoint(0),RUN/'INITIAL_CHECKPOINT.pt');assert not np.count_nonzero(evaluate(0))
        for step in range(1,2001):
            ix,tr,audit=c10.draw(pools,generator)
            assert meta['known'][ix].all() and available[ix].all() and (meta['split'][ix]=='TRAIN').all()
            assert np.array_equal(np.bincount(meta['external_target'][ix],minlength=2),[32,32])
            draw_hash.update(ix.astype('<i8').tobytes());draw_hash.update(tr.astype('<i8').tobytes())
            idx=torch.as_tensor(ix,dtype=torch.long,device='cuda');pairs=torch.as_tensor(tr,dtype=torch.long,device='cuda')
            model.train();optimizer.zero_grad(set_to_none=True);z=model(x[ug[idx]])
            loss,bce,hinge=c10.supervised_loss(z,tg[idx],ag[idx],pairs);assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm)
            optimizer.step();updates=step
            with (RUN/'TRAIN_UPDATES.jsonl').open('a') as f:f.write(json.dumps(dict(update=step,occurrence_indices=ix.tolist(),input_unit_indices=units[ix].tolist(),matched_triplet_batch_positions=tr.tolist(),sampling=audit,available_loss_rows=64,sampled_unavailable_rows=0,loss=float(loss.detach()),BCE=float(bce.detach()),matched_hinge=float(hinge.detach()),gradient_norm_before_clip=float(norm)),allow_nan=False)+'\n')
            if step%100==0:evaluate(step)
        assert draw_hash.hexdigest()==EXPECTED_DRAW_SHA,'C10 identical2000 source draws required.'
        torch.save(checkpoint(2000),RUN/'FINAL_CHECKPOINT.pt');model.load_state_dict(best_state['state_dict']);model.eval()
        with torch.no_grad():z=model(x)[ug].cpu().numpy()
        z=np.where(available,z,0).astype(np.float32);criterion,parts=d.selector(z[:NOLD],available[:NOLD],old);assert criterion==best
        metrics,gates=c9.report(d,z[:NOLD],available[:NOLD],old,c10_design)
        ycb=d.external_cm(z,available,meta,meta['domain']=='V109_YCB_SURFACE');passed=all(g['PASS'] for g in gates.values())
        line=meta['lineage'];q=np.array([r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan] for r in line]);L=np.array([r['L'] if r['L'] is not None else np.nan for r in line])
        np.savez_compressed(RUN/'SELECTED_ALL41984_SOURCE_SCORES.npz',query_logit=z,query_available=available,factual_query_available=factual,context_available=context,context_Q_inside=q_inside,prediction=d.decode(z,available),input_unit_index=units,raw_teacher_state=meta['typed_state'],external_target=meta['external_target'],split=meta['split'],domain=meta['domain'],source=meta['source'],variant=meta['variant'],query_xy_raw=q,L_raw=L,old_accepted_triplets=old['old_pairs'],new_selected_triplets=old['new_pairs'],global_occurrence_index=np.arange(N))
        write(RUN/'SOURCE_RESULTS.json',dict(completed=True,selected_update=selected,criterion_eight_VAL_cohorts_BCE=best,VAL_cohorts=parts,metrics=metrics,YCB_surface_TRAIN=ycb,YCB_validation_rows=0,no_new_natural_real_heldout_claim=True,raw_state_counts={k:int((meta['typed_state']==k).sum()) for k in 'DESU'},known_TRAIN=2523,known_VAL=516,availability=dict(factual_unavailable=int((~factual).sum()),context_unavailable=int((~context).sum()),extra_context_unavailable=int((factual&~context).sum()),final_unavailable=int((~available).sum()),known_train_unavailable=int((meta['known']&(meta['split']=='TRAIN')&~available).sum()),Q_outside_additional_U=0),learned_self_classification=False,native_mapping='UNKNOWN',human_gold=False,source_development_exposure=True,all2000_draws_sha256=draw_hash.hexdigest(),same_C10_sampler_trace=True))
        write(RUN/'SOURCE_GATE_RESULTS.json',dict(completed=True,all_gates_pass=passed,gates=gates,current_inference_authorized=False))
        write(RUN/'MODEL_FREEZE.json',dict(completed=True,selected_update=selected,checkpoint=bind(RUN/'BEST_SPATIAL_TARGET_QUERY_HEAD.pt'),encoder_weights=design['encoder']['weight'],feature_receipt=e['feature_receipt'],feature_pack=e['feature_pack'],trainable_parameters=221825,encoder_trainable_parameters=0,threshold_logit=0,design=bind(DESIGN),trainer=bind(__file__),current_used_for_fit_or_selection=False,current_inference_authorized=False,PnP=0))
        write(RUN/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=2000,selected_update=selected,criterion_eight_VAL_cohorts_BCE=best,all_source_gates_pass=passed,seconds=time.monotonic()-started,source_results=bind(RUN/'SOURCE_RESULTS.json'),gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),scores=bind(RUN/'SELECTED_ALL41984_SOURCE_SCORES.npz'),model_freeze=bind(RUN/'MODEL_FREEZE.json'),current_or_PnP=0))
        print(json.dumps(read(RUN/'FIT_COMPLETE.json')),flush=True)
    except BaseException:
        write(RUN/'FAILED_EXECUTION.json',dict(optimizer_updates_completed=updates,traceback=traceback.format_exc(),restart_automatically=False));raise

def generated_check(output):
    """Generated FP32 tensors/pools only; no metadata/feature/weights body loads."""
    torch.set_num_threads(1);torch.manual_seed(211);model=SpatialTargetQueryHead();c10=pinned_c10()
    assert sum(p.numel() for p in model.parameters())==221825
    x=torch.randn(6,1928);assert model(x).shape==(6,) and not torch.count_nonzero(model(x))
    seen=[];hook=model.head[0].register_forward_pre_hook(lambda m,args:seen.append(args[0].detach().clone()))
    model(x);hook.remove();a=seen[0];assert a.shape==(6,3464)
    T,Q,C,P=x[:,:384],x[:,384:768],x[:,1152:1536],x[:,1536:1920]
    assert torch.equal(a[:,1928:2312],torch.abs(T-C)) and torch.equal(a[:,2312:2696],T*C)
    assert torch.equal(a[:,2696:3080],torch.abs(Q-P)) and torch.equal(a[:,3080:],Q*P)
    raw=torch.tensor([-2.,9.,-9.,3.,8.,-8.],requires_grad=True);available=torch.tensor([1,0,1,1,0,1],dtype=torch.bool);target=torch.tensor([0.,1.,-1.,1.,-1.,0.])
    loss,_,_=c10.supervised_loss(raw,target,available,torch.empty((0,3),dtype=torch.long));loss.backward()
    assert torch.count_nonzero(raw.grad[[1,2,4]])==0 and torch.count_nonzero(raw.grad[[0,3,5]])==3
    try:model(torch.randn(2,1927))
    except ValueError:pass
    else:raise AssertionError('Incorrect feature dimension accepted.')
    try:model({'features':x,'reference':torch.zeros(6,2)})
    except ValueError:pass
    else:raise AssertionError('GT dictionary accepted.')
    y=np.zeros(160,np.int8);y[[1,4,7,10,13,16,19,22]]=1;y[80:]=1
    pools={'old_triplets':np.arange(12).reshape(4,3),'synth_triplets':np.arange(12,24).reshape(4,3),'old_LOCO_D':np.arange(24,30),'old_PALLOX_D':np.arange(30,36),'synth_D':np.arange(36,44),'YCB_D':np.arange(44,60),'old_natural_E':np.arange(80,86),'synth_E':np.arange(86,102),'YCB_E':np.arange(102,126)}
    g1=torch.Generator().manual_seed(208);g2=torch.Generator().manual_seed(208);h=hashlib.sha256()
    for _ in range(2000):
        ix,tr,audit=c10.draw(pools,g1);jx,jr,_=c10.draw(pools,g2)
        assert np.array_equal(ix,jx) and np.array_equal(tr,jr)
        assert np.array_equal(np.bincount(y[ix],minlength=2),[32,32]) and np.array_equal(y[ix][tr],np.tile([0,1,0],(8,1)))
        h.update(ix.astype('<i8').tobytes());h.update(tr.astype('<i8').tobytes())
    write(output,dict(PASS=True,code=bind(__file__),pinned_C10=bind(C10),runtime=torch.__version__,parameters=221825,feature_input=1928,linear_input=3464,lastzero=True,all4_relations_and_descriptor_order_exact=True,known_available_loss_only=True,GTdict_and_wrong_dimension_rejected=True,same_C10_generated2000_draws=True,generated_draw_SHA256=h.hexdigest(),expected_actual_C10_draw_SHA256=EXPECTED_DRAW_SHA,Q_feature_no_additional_availability_gate=True,source_arrays_checkpoints_feature_body_or_adapter_metadata_reads=0,real_data_model_forward_fit_current_PnP=0))

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--entry',type=Path);g.add_argument('--generated-check',type=Path);args=ap.parse_args()
    if args.generated_check:generated_check(args.generated_check)
    else:run(args.entry)
