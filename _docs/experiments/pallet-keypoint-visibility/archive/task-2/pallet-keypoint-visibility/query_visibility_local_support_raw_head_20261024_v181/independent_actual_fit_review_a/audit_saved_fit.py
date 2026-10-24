"""V181 saved-only fit audit. Actual logs/checkpoints/outputs require both-completed root trigger."""
from pathlib import Path
import argparse,hashlib,json,math,sys,time
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent
PROTOCOL=H/'PROTOCOL_BEFORE_ACTUAL_FIT_LOG_OR_OUTPUT_BODIES.json';CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
RUN=V/'runs/C50_C51_LOCAL_SUPPORT_FIXED8000_v2'
ARMS=['C50_RAW96_DUAL_SUPPORT_OWNER_AUX','C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX']
TRAIN=['SYNTH_TRAIN4096','PUBLIC_TRAIN1480','CF3816'];N={'SYNTH_TRAIN4096':4096,'PUBLIC_TRAIN1480':1480,'CF3816':3816,'SYNTH_VAL2048':2048}
META=['row_id','original_global_index','native_slot','query_xy','L','original_query_available']
def require(ok,message):
    if not bool(ok):raise AssertionError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):
    path=Path(b['path']).resolve();require(path.is_relative_to(V),'Audit only frozen V181 files')
    require(sha(path)==b['sha256'],'SHA '+str(path));return path
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def make_pools(cache):
    s,p,c=(cache[k] for k in TRAIN);av=s['available'];support=s['target']['support_target'];soft=s['target']['owner_soft']
    events=[]
    for col in [0,1]:events.append([np.array([i for i in range(len(av)) if av[i] and support[i,col]==v],np.int64) for v in [1,0]])
    require(all(len(a)>0 for e in events for a in e),'Nonempty support classes')
    eligible=av&np.isfinite(soft).all(1);conditional=[]
    for col in range(3):
        weight=np.zeros(len(av),np.float64);weight[eligible]=soft[eligible,col];require((weight>=0).all() and weight.sum()>0,'Nonzero raw source class mass')
        conditional.append(weight/weight.sum(dtype=np.float64))
    hard=[]
    for domain,classes in [(p,range(3)),(c,range(2))]:
        require((domain['target']['support_target']==-1).all(),'Public/CF owner never end supervision')
        hard.append([np.array([i for i,v in enumerate(domain['target']['owner_target']) if domain['available'][i] and v==cl],np.int64) for cl in classes])
    return events,conditional,hard

def expected_draw(rng,pool):
    events,conditional,hard=pool;d={}
    for name,event in zip(['visible','foreign'],events):d[name]=np.concatenate([rng.choice(x,16,replace=True) for x in event])
    d['synthetic_owner']=np.concatenate([rng.choice(len(p),16,replace=True,p=p) for p in conditional])
    d['public_owner']=np.concatenate([rng.choice(x,16,replace=True) for x in hard[0]])
    d['CF_owner']=np.concatenate([rng.choice(x,16,replace=True) for x in hard[1]])
    require([len(v) for v in d.values()]==[32,32,48,48,32],'Fixed192 event/class draw order')
    return d

def audit_log(rows,pool,steps=8000):
    require(len(rows)==steps,'Exactly fixed update count');rng=np.random.default_rng(1810);h=hashlib.sha256();max_mean=0.
    for update,row in enumerate(rows,1):
        require(row['update']==update,'Update sequence1..8000');desired=expected_draw(rng,pool)
        require(list(row['draw_indices'])==list(desired),'Fixed draw key order')
        for key,arr in desired.items():
            actual=np.asarray(row['draw_indices'][key],np.int64);require(np.array_equal(actual,arr),'Independent draw/pool membership '+key)
            h.update(key.encode());h.update(arr.astype('<i8').tobytes())
        require(row['cumulative_draw_sha']==h.hexdigest(),'Independent cumulative draw digest')
        loss=row['five_losses'];require(len(loss)==5 and all(math.isfinite(v) and v>=0 for v in loss),'Finite five nonnegative components')
        mean=math.fsum(loss)/5;total=row['total_loss'];diff=abs(total-mean);max_mean=max(max_mean,diff)
        require(math.isfinite(total) and diff<=4*np.finfo(np.float32).eps*max(1.,abs(mean)),'FP32 five-equal-mean consistency, no new forward')
        require(math.isfinite(row['grad_norm']) and row['grad_norm']>=0,'Finite recorded gradient norm')
    return dict(steps=steps,draw_sha=h.hexdigest(),all_draws_and_pool_memberships_exact=True,all_loss_gradnorm_finite=True,max_recorded_mean_rounding_difference=max_mean)

def state_sha(state):
    # The frozen trainer canonical digest is a serialization contract; recompute from saved tensors.
    h=hashlib.sha256()
    for k in sorted(state):
        a=state[k].detach().cpu().contiguous();h.update(str(k).encode());h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()

def audit_checkpoint(b,arm,record,protocol_binding):
    packet=torch.load(check(b),map_location='cpu',weights_only=True)
    require(packet['update']==8000 and packet['seed']==1810 and packet['arm']==arm and packet['use_sam']==(arm==ARMS[1]) and packet['protocol']==protocol_binding,'Final-only checkpoint metadata')
    state=packet['state_dict'];require(len(state)==30 and sum(t.numel() for t in state.values())==115157,'All30 original parameter tensors/115157 parameters')
    require(all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in state.values()),'All final parameter tensors FP32 finite')
    require(state_sha(state)==record['final_state_sha'],'Saved final state independently rehashed')
    opt=packet['optimizer_state'];require(len(opt['param_groups'])==1 and len(opt['state'])==30,'One AdamW optimizer/all30 parameters')
    group=opt['param_groups'][0];require(group['lr']==3e-4 and group['weight_decay']==1e-4 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and not group['amsgrad'],'Frozen AdamW hyperparameters')
    require(len(group['params'])==30 and len(set(group['params']))==30 and set(group['params'])==set(opt['state']),'Every original parameter has one optimizer state')
    for pid,param in zip(group['params'],state.values()):
        moment=opt['state'][pid];require(set(moment)=={'step','exp_avg','exp_avg_sq'} and float(moment['step'])==8000.,'Every optimizer state reached8000')
        require(moment['exp_avg'].shape==moment['exp_avg_sq'].shape==param.shape and moment['exp_avg'].dtype==moment['exp_avg_sq'].dtype==torch.float32,'Moment dtype/shapes')
        require(torch.isfinite(moment['exp_avg']).all() and torch.isfinite(moment['exp_avg_sq']).all() and (moment['exp_avg_sq']>=0).all(),'Finite valid final AdamW moments')
    return dict(final_state_sha=record['final_state_sha'],parameters=115157,optimizer_state_parameters30=True,every_optimizer_step8000=True,all_final_tensors_finite=True)

def audit_output(binding,original):
    n=len(original['row_id'])
    with np.load(check(binding),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    require(set(a)==set(META)|{'raw_support_logits','raw_owner_logits','support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1','support_order','owner_order'},'Exact final output schema')
    for k in META:require(exact(a[k],original[k]),'Every original output metadata byte '+k)
    require(a['support_order'].tolist()==['visible_end_support','foreign_front_support'] and a['owner_order'].tolist()==['TARGET','FOREIGN','BACKGROUND'],'Independent fixed output semantics')
    for key,dtype,shape in [('raw_support_logits',np.float32,(n,2)),('raw_owner_logits',np.float32,(n,3)),('support_prediction',np.int8,(n,2)),('owner_prediction',np.int8,(n,)),('regional_support_proxy_D0_E1_Uminus1',np.int8,(n,))]:require(a[key].dtype==dtype and a[key].shape==shape,'Saved schema '+key)
    av=original['original_query_available']
    for i in range(n):
        if not av[i]:require(np.isnan(a['raw_support_logits'][i]).all() and np.isnan(a['raw_owner_logits'][i]).all() and (a['support_prediction'][i]==-1).all() and a['owner_prediction'][i]==a['regional_support_proxy_D0_E1_Uminus1'][i]==-1,'Original U first, no resurrection')
        else:
            s=a['raw_support_logits'][i];o=a['raw_owner_logits'][i];require(np.isfinite(s).all() and np.isfinite(o).all(),'Finite available output')
            visible=int(float(s[0])>0);foreign=int(float(s[1])>0);best=max(range(3),key=lambda k:float(o[k]));proxy=1 if foreign else 0 if visible else -1
            require(a['support_prediction'][i].tolist()==[visible,foreign] and int(a['owner_prediction'][i])==best and int(a['regional_support_proxy_D0_E1_Uminus1'][i])==proxy,'Independent scalar decisions, >0/firsttie/priority')
    return dict(rows=n,original_inputU=int((~av).sum()),all_original_metadata_bytes_preserved=True,all_scalar_predictions_U_first_exact=True)

def invented():
    def hard(labels):return dict(available=np.ones(len(labels),bool),target=dict(support_target=np.full((len(labels),2),-1,np.int8),owner_target=np.array(labels,np.int8)))
    toy={'SYNTH_TRAIN4096':dict(available=np.array([1,1,1,0],bool),target=dict(support_target=np.array([[1,0],[0,1],[-1,-1],[-1,-1]],np.int8),owner_soft=np.array([[.2,.1,0],[.1,.2,.7],[0,0,0],[np.nan]*3],np.float64))),'PUBLIC_TRAIN1480':hard([0,1,2]),'CF3816':hard([0,1])}
    pool=make_pools(toy);r=np.random.default_rng(1810);h=hashlib.sha256();logs=[]
    for step in range(1,4):
        d=expected_draw(r,pool)
        for k,a in d.items():h.update(k.encode());h.update(a.astype('<i8').tobytes())
        logs.append(dict(update=step,draw_indices={k:v.tolist() for k,v in d.items()},cumulative_draw_sha=h.hexdigest(),five_losses=[1.]*5,total_loss=1.,grad_norm=1.))
    result=audit_log(logs,pool,3);checks={'actual_log_auditor_replays_exact_draws_and_digest':result['steps']==3,'partial_known_CDF_mass_no_row_normalization':np.allclose(pool[1][0],[2/3,1/3,0,0],rtol=0,atol=1e-15),'unknown_U_and_zero_known_mass_never_drawn':all(2 not in row['draw_indices']['synthetic_owner'] and 3 not in row['draw_indices']['synthetic_owner'] for row in logs)}
    bad=json.loads(json.dumps(logs));bad[0]['draw_indices']['visible'][0]=3
    try:audit_log(bad,pool,3)
    except AssertionError:checks['actual_log_auditor_rejects_illegal_U_draw']=True
    else:checks['actual_log_auditor_rejects_illegal_U_draw']=False
    state={'a':torch.tensor([1.,2.])};digest=state_sha(state);state['a'][0]=2;checks['actual_saved_state_digest_detects_tensor_change']=digest!=state_sha(state)
    require(all(checks.values()),'Invented auditor checks');return {k:bool(v) for k,v in checks.items()}

def prepare():
    checks=invented();write(CHECKS,dict(completed=True,PASS=True,invented_only=True,code=bind(__file__),checks=checks,number_checks=len(checks),actual_log_checkpoint_or_output_body_reads=0,CUDA_initialized=torch.cuda.is_initialized()))
    p=dict(completed=True,prepared_only=True,code=bind(__file__),checks=bind(CHECKS),
        source_fit_protocol=bind(V/'source_fit_preparation/PROTOCOL_BEFORE_NEW_RAW_CACHE_OR_FIT_BODIES.json'),source_fit_entry=bind(V/'ROOT_ACTUAL_C50_C51_SOURCE_FIT_ENTRY_v2.json'),
        training_cache_complete=bind(V/'source_raw96_cache_v1/COMPLETE.json'),evaluation_cache_complete=bind(V/'synthetic_VAL2048_raw96_cache_v1/COMPLETE.json'),
        previous_static_review=bind(V/'independent_runtime_pool_repair_review_a/REVIEW.json'),steps_each=8000,arms=ARMS,output_run=str(RUN),
        audit='Original cachetargets→independent deterministic8000draw reconstruction perarm; allrecordedlossfinite andfive-mean consistency; all30AdamWstatesstep8000; savedfinalstatehash; all2freezebarrier; finaloutputs all11440rows/arm metadata+scalarU/support/owner/proxy decisions.',
        limitations='No actual loss/model replay or initialization-state checkpoint exists in this contract: initialsamehash and noVALpre-freezeaccess use sealedreceipts+reviewedcodepath. Saved finalstate digest is recomputed from checkpoint tensors, with no model construction/forward. Actual head performance outside this audit.',
        actual_log_checkpoint_or_output_reads_during_prepare=0,new_model_or_SAM_or_fit=0,selected_model=None,fully_solved=False)
    write(PROTOCOL,p);write(H/'ROOT_AUDIT_TRIGGER_TEMPLATE_APPROVED_FALSE.json',dict(approved_saved_fit_audit=False,all2_completed=False,fit_complete=None,code=p['code'],protocol=bind(PROTOCOL)))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],actual_fit_log_checkpoint_or_output_reads=0))

def run(trigger):
    e=read(trigger);require(e.get('approved_saved_fit_audit') is True and e.get('all2_completed') is True,'Separate root both-completed audit trigger')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL),'Frozen auditor binding');p=read(PROTOCOL);require(p['code']==bind(__file__) and p['checks']==bind(CHECKS),'Frozen auditor preparation')
    require(Path(sys.executable).resolve()==Path('/home/minjae/anaconda3/envs/pallet-yolo26/bin/python').resolve() and np.__version__=='1.26.4' and torch.__version__=='2.1.1+cu118','Pinned CPU audit runtime')
    complete=read(check(e['fit_complete']));require(complete['completed'] and complete['PASS'] and complete['all2_frozen_before_outputs_or_VAL_access'] and complete['updates_each']==8000 and complete['total_updates']==16000,'Actual both-fixed8000 complete')
    require(Path(e['fit_complete']['path']).resolve()==RUN/'COMPLETE.json' and complete['protocol']==p['source_fit_protocol'],'Exactly repaired frozen run, no other attempt')
    original={};cache={};tc=read(check(p['training_cache_complete']));vc=read(check(p['evaluation_cache_complete']))
    for domain,d in {**tc['domains'],**vc['domains']}.items():
        with np.load(check(d['inputs']),allow_pickle=False) as z:meta={k:z[k].copy() for k in META}
        require(len(meta['row_id'])==N[domain],'All original output rows');original[domain]=meta
        if domain in TRAIN:
            with np.load(check(d['targets']),allow_pickle=False) as z:target={k:z[k].copy() for k in z.files}
            cache[domain]=dict(available=meta['original_query_available'],target=target)
    pools=make_pools(cache);freeze=read(check(complete['all2_freeze']));require(freeze['completed'] and freeze['validation_cache_or_targets_read'] is False and freeze['saved_outputs_emitted'] is False,'Both freeze-before-VAL/output receipt')
    require(set(complete['arms'])==set(ARMS) and set(freeze['checkpoints'])==set(ARMS),'Both fixed arms retained');result={};start=time.monotonic()
    for arm in ARMS:
        r=complete['arms'][arm];require(r['updates']==8000 and r['checkpoint']==freeze['checkpoints'][arm],'Final checkpoint binding at barrier')
        log=audit_log(lines(check(r['step_losses'])),pools);require(log['draw_sha']==r['draw_sha']==complete['same_draw_sha']==freeze['same_draw_sha'],'Actual same draw digest all8000')
        require(r['initial_state_sha']==complete['same_initial_state_sha']==freeze['same_initial_state_sha'],'Recorded initial states identical')
        ck=audit_checkpoint(r['checkpoint'],arm,r,p['source_fit_protocol']);require(set(r['outputs'])==set(N),'All four source domains outputs retained')
        outputs={name:audit_output(r['outputs'][name],original[name]) for name in N};result[arm]=dict(draws=log,checkpoint=ck,outputs=outputs)
        print(json.dumps(dict(arm=arm,all8000draws_and11440outputrows_PASS=True,seconds=time.monotonic()-start)),flush=True)
    report=dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),trigger=bind(trigger),actual_complete=e['fit_complete'],all2_freeze=complete['all2_freeze'],arms=result,
        same_initial_state_hash_recorded=True,all16000updates_and_each_30_optimizer_steps8000_verified=True,all22880_saved_output_rows_scalar_decisions_and_metadata_verified=True,
        limitations=p['limitations'],new_model_forward_SAM_or_fit=0,private_publicheld_V112_reads=0,selected_model=None,fully_solved=False,seconds=time.monotonic()-start)
    write(H/'REVIEW.json',report);print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'))))
if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--trigger',type=Path);args=ap.parse_args();prepare() if args.prepare else run(args.trigger)
