"""V183 saved-only fit audit. Actual logs/checkpoints/outputs require both-completed root trigger.
No head/model construction or forward, no evaluation targets. CPU saved tensors only.
"""
from pathlib import Path
import argparse,hashlib,json,math,sys,time
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
ORIGINAL=P/'query_visibility_local_support_raw_head_20261024_v181'
PROTOCOL=H/'PROTOCOL_BEFORE_ACTUAL_FIT_LOG_OR_OUTPUT_BODIES.json';CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
RUN=V/'runs/C52_C53_JITTER_PAIRED_SOURCE_FIXED8000'
ARMS=['C52_RAW96_JITTER_PAIRED_SOURCE','C53_RAW96_SAM_JITTER_PAIRED_SOURCE']
TRAIN=['SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816'];N={'SYNTH_AUG65536':65536,'PUBLIC_TRAIN1480':1480,'CF3816':3816,'COMMON4096':4096,'SYNTH_VAL2048':2048}
EXPECTED_U={'SYNTH_AUG65536':1264,'PUBLIC_TRAIN1480':194,'CF3816':360,'COMMON4096':96,'SYNTH_VAL2048':8}
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
    path=Path(b['path']).resolve();require(any(path.is_relative_to(root) for root in [V,ORIGINAL,P/'query_visibility_source_same_query_causal_diagnosis_20261024_v182']),'Only exact bound V181/V182/V183 caches or saved outputs')
    require(not any(s in str(path).lower() for s in ['_v112','_v169','_v177','_v178','current223','/private_','source_validation']),'No private/current/publicheld/V112')
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
    with np.load(check(binding),allow_pickle=False) as z:a={k:z[k] for k in z.files}
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
    toy={'SYNTH_AUG65536':dict(available=np.array([1,1,1,0],bool),target=dict(support_target=np.array([[1,0],[0,1],[-1,-1],[-1,-1]],np.int8),owner_soft=np.array([[.2,.1,0],[.1,.2,.7],[0,0,0],[np.nan]*3],np.float64))),'PUBLIC_TRAIN1480':hard([0,1,2]),'CF3816':hard([0,1])}
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
    meta=dict(row_id=np.array(['invented/a','invented/b','invented/u']),original_global_index=np.arange(3,dtype=np.int64),native_slot=np.arange(3,dtype=np.int8),query_xy=np.array([[1.,2.],[3.,4.],[np.nan,np.nan]],np.float64),L=np.array([1.,2.,np.nan]),original_query_available=np.array([True,True,False]))
    packet=dict(**meta,raw_support_logits=np.array([[0.,-0.],[1.,1.],[np.nan,np.nan]],np.float32),raw_owner_logits=np.array([[0.,0.,0.],[1.,2.,2.],[np.nan]*3],np.float32),support_prediction=np.array([[0,0],[1,1],[-1,-1]],np.int8),owner_prediction=np.array([0,1,-1],np.int8),regional_support_proxy_D0_E1_Uminus1=np.array([-1,1,-1],np.int8),support_order=np.array(['visible_end_support','foreign_front_support']),owner_order=np.array(['TARGET','FOREIGN','BACKGROUND']))
    path=H/'INVENTED_OUTPUT_ONLY.npz';np.savez_compressed(path,**packet);checks['actual_scalar_output_auditor_zero_firsttie_nonexclusive_and_U']=audit_output(bind(path),meta)['rows']==3
    packet['owner_prediction'][2]=0;badpath=H/'INVENTED_BAD_U_OUTPUT_ONLY.npz';np.savez_compressed(badpath,**packet)
    try:audit_output(bind(badpath),meta)
    except AssertionError:checks['actual_scalar_output_auditor_rejects_resurrected_U']=True
    else:checks['actual_scalar_output_auditor_rejects_resurrected_U']=False
    require(all(checks.values()),'Invented auditor checks');return {k:bool(v) for k,v in checks.items()}

def prepare():
    checks=invented();write(CHECKS,dict(completed=True,PASS=True,invented_only=True,code=bind(__file__),checks=checks,number_checks=len(checks),actual_log_checkpoint_or_output_body_reads=0,CUDA_initialized=torch.cuda.is_initialized()))
    p=dict(completed=True,prepared_only=True,code=bind(__file__),checks=bind(CHECKS),
        source_fit_protocol=bind(V/'source_fit_preparation/PROTOCOL_BEFORE_AUGMENTED_CACHE_OR_FIT_BODIES.json'),source_fit_code=bind(V/'source_fit_preparation/train_jitter_paired_source.py'),scope=bind(V/'ROOT_SCOPE_BEFORE_NEW_AUGMENTATION_OR_FIT.json'),
        previous_static_review=bind(V/'independent_trainer_review_a/REVIEW.json'),steps_each=8000,arms=ARMS,output_run=str(RUN),expected_rows=N,expected_original_inputU=EXPECTED_U,
        actual_dependencies='Read completed augmentation/publicCF/evaluation descriptors only after root both-completed audit trigger. Preparation reads no actual fit logs/checkpoints/output or VAL target bodies.',
        audit='TRAIN65536+public253+CF314 targets independently reconstruct8000draws/arm, allpoolmemberships/cumulativehash andfinitefive-meanlosses; all30AdamWstatesstep8000; final115157statehash; bothfreeze; all76976outputrows/arm metadata andscalarU/support/owner/proxy. Evaluationtargets never read.',
        limitations='No actual loss/model replay or initialization-state checkpoint exists in this contract: initialsamehash and noCOMMON/VALpre-freezeaccess use sealedreceipts+reviewedcodepath, not OS access trace. Saved finalstate digest is recomputed from checkpoint tensors without model construction/forward. No performance/qualification inference.',
        actual_log_checkpoint_or_output_reads_during_prepare=0,new_model_or_SAM_or_fit=0,selected_model=None,fully_solved=False)
    write(PROTOCOL,p);write(H/'ROOT_AUDIT_TRIGGER_TEMPLATE_APPROVED_FALSE.json',dict(approved_saved_fit_audit=False,all2_completed=False,fit_complete=None,code=p['code'],protocol=bind(PROTOCOL)))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],actual_fit_log_checkpoint_or_output_reads=0))

def run(trigger):
    e=read(trigger);require(e.get('approved_saved_fit_audit') is True and e.get('all2_completed') is True,'Separate root both-completed audit trigger')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL),'Frozen auditor binding');p=read(PROTOCOL);require(p['code']==bind(__file__) and p['checks']==bind(CHECKS),'Frozen auditor preparation')
    require(Path(sys.executable).resolve()==Path('/home/minjae/anaconda3/envs/pallet-yolo26/bin/python').resolve() and np.__version__=='1.26.4' and torch.__version__=='2.1.1+cu118','Pinned CPU audit runtime')
    complete=read(check(e['fit_complete']));require(complete['completed'] and complete['PASS'] and complete['all2_frozen_before_outputs_or_evaluation_access'] and complete['updates_each']==8000 and complete['total_updates']==16000,'Actual both-fixed8000 complete')
    require(Path(e['fit_complete']['path']).resolve()==RUN/'COMPLETE.json' and complete['protocol']==p['source_fit_protocol'] and complete['code']==p['source_fit_code'],'Exact frozen run/code, no other attempt')
    original={};cache={};aug=read(check(complete['augmentation_cache_complete']));tc=read(check(complete['original_training_cache_complete']));vc=read(check(complete['evaluation_cache_complete']))
    require(aug['domain']=='SYNTH_AUG65536' and aug['rows']==65536 and aug['original_aug_inputU']==1264 and aug['SAM_arrays_present'] is True,'Exact training augmentation completion')
    descriptors=read(check(complete['common_evaluation_descriptor']));require(descriptors['domain']=='COMMON4096','Exact postfreeze COMMON descriptor')
    records={'SYNTH_AUG65536':aug,'PUBLIC_TRAIN1480':tc['domains']['PUBLIC_TRAIN1480'],'CF3816':tc['domains']['CF3816'],'SYNTH_VAL2048':vc['domains']['SYNTH_VAL2048'],'COMMON4096':dict(inputs=descriptors['raw_inputs8'],metadata=descriptors['metadata'])}
    for domain,d in records.items():
        with np.load(check(d['inputs']),allow_pickle=False) as z:meta={k:z[k] for k in META}
        require(len(meta['row_id'])==N[domain] and int((~meta['original_query_available']).sum())==EXPECTED_U[domain],'All output rows/factualU')
        rr=lines(check(d['metadata']));require(len(rr)==N[domain],'Full rows metadata count')
        for i,r in enumerate(rr):
            require(r['row_id']==meta['row_id'][i] and r['original_global_index']==int(meta['original_global_index'][i]) and r['native_slot']==int(meta['native_slot'][i]) and r['original_query_available']==bool(meta['original_query_available'][i]),'Input metadataidentity')
            q=np.asarray([np.nan,np.nan] if r['query_xy'] is None else r['query_xy'],np.float64);L=np.asarray(np.nan if r['L'] is None else r['L'],np.float64)
            require(exact(q,meta['query_xy'][i]) and exact(L,np.asarray(meta['L'][i])),'Input metadata qL')
        original[domain]=meta
        if domain in TRAIN:
            require(all(r['split']==('TRAIN' if domain=='SYNTH_AUG65536' else 'SOURCE_DEVELOPMENT') for r in rr),'Training split exact')
            with np.load(check(d['targets']),allow_pickle=False) as z:target={k:z[k] for k in z.files}
            cache[domain]=dict(available=meta['original_query_available'],target=target)
        else:require(all(r['split']==('TRAIN' if domain=='COMMON4096' else 'VALIDATION') for r in rr),'Evaluationonly split')
    # COMMON SAM metadata belongs to the same fixed diagnostic input; targets binding never opened.
    with np.load(check(descriptors['sam_inputs']),allow_pickle=False) as z:
        for k in META:require(exact(z[k],original['COMMON4096'][k]),'COMMON exactSAM/meta merge '+k)
    pools=make_pools(cache);require([len(a) for a in pools[2][0]]==[53,52,148] and [len(a) for a in pools[2][1]]==[157,157],'Unchanged owner-only public253/CF314 pools')
    freeze=read(check(complete['all2_freeze']));require(freeze['completed'] and freeze['validation_cache_or_targets_read'] is False and freeze['common_diagnostic_inputs_or_targets_read'] is False and freeze['saved_outputs_emitted'] is False,'Both freeze-before-COMMON/VAL/output receipt')
    require(set(complete['arms'])==set(ARMS) and set(freeze['checkpoints'])==set(ARMS),'Both fixed arms retained');result={};start=time.monotonic()
    for arm in ARMS:
        r=complete['arms'][arm];require(r['updates']==8000 and r['checkpoint']==freeze['checkpoints'][arm],'Final checkpoint binding at barrier')
        log=audit_log(lines(check(r['step_losses'])),pools);require(log['draw_sha']==r['draw_sha']==complete['same_draw_sha']==freeze['same_draw_sha'],'Actual same draw digest all8000')
        require(r['initial_state_sha']==complete['same_initial_state_sha']==freeze['same_initial_state_sha'],'Recorded initial states identical')
        ck=audit_checkpoint(r['checkpoint'],arm,r,p['source_fit_protocol']);require(set(r['outputs'])==set(N),'All five source domains outputs retained')
        outputs={name:audit_output(r['outputs'][name],original[name]) for name in N};result[arm]=dict(draws=log,checkpoint=ck,outputs=outputs)
        print(json.dumps(dict(arm=arm,all8000draws_and76976outputrows_PASS=True,seconds=time.monotonic()-start)),flush=True)
    report=dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),trigger=bind(trigger),actual_complete=e['fit_complete'],all2_freeze=complete['all2_freeze'],arms=result,
        same_initial_state_hash_recorded=True,all16000updates_and_each_30_optimizer_steps8000_verified=True,all153952_saved_output_rows_scalar_decisions_and_metadata_verified=True,original_inputU_perarm=EXPECTED_U,evaluation_target_body_reads=0,
        limitations=p['limitations'],new_model_forward_SAM_or_fit=0,private_publicheld_V112_reads=0,selected_model=None,fully_solved=False,seconds=time.monotonic()-start)
    write(H/'REVIEW.json',report);print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'))))
if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--trigger',type=Path);args=ap.parse_args();prepare() if args.prepare else run(args.trigger)
