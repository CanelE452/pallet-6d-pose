"""Independent saved TRAIN-only fit audit/readout. No model/producer import/forward."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,os,stat,sys,time,traceback
import numpy as np
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PROTO=H/'PROTOCOL_BEFORE_SOURCE_FIT_OUTPUT_BODIES.json'
RUN=V/'runs/C54_C55_DIRECT_END_SOURCE_FIXED8000'
ARMS=('C54_RAW96_DIRECT_END','C55_RAW96_PLUS_FROZEN_SAM_DIRECT_END')
DOMAINS={'SYNTH_AUG65536':65536,'PUBLIC_TRAIN1480':1480,'CF3816':3816,'DIRECT_CONTROL':2792}
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
TARGET5=('support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass')
OUTKEYS=set(META)|{'raw_support_logits','raw_owner_logits','raw_direct_end_external_logit','support_prediction','owner_prediction','direct_end_prediction_D0_E1_Uminus1','regional_support_proxy_D0_E1_Uminus1','support_order','owner_order','direct_end_order'}

def need(v,msg):
    if not bool(v):raise AssertionError(msg)
def safe(value):
    p=Path(value);need(p.is_absolute() and '..' not in p.parts and all('v112' not in q.casefold() for q in p.parts),'Safe lexical name before access')
    need(p.is_relative_to(P) and not any(q in str(p).casefold() for q in ('current223','_v169/','_v177/','_v178/','_v184/')),'Bound source project only')
    return p
def opened(path):
    p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for q in p.parts[1:-1]:
            end=os.open(q,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=end
        last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(last).st_mode),'Regular nofollow file');return os.fdopen(last,'rb')
def raw(path):
    with opened(path) as f:return f.read()
def bind(path):return dict(path=str(path),sha256=hashlib.sha256(raw(path)).hexdigest())
def checked(b):
    z=raw(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Metadata/log SHA drift');return z
def jd(b):return json.loads(checked(b))
def jl(b):return [json.loads(x) for x in checked(b).splitlines() if x.strip()]
def write(path,x):
    with path.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def arrays(b,keys,allow,strict=False):
    need(allow.get(b['path'])==b,'Exact binary whitelist before header/hash')
    with opened(b['path']) as f:
        h=hashlib.sha256()
        for z in iter(lambda:f.read(1<<20),b''):h.update(z)
        need(h.hexdigest()==b['sha256'],'Container SHA drift');f.seek(0)
        with np.load(f,allow_pickle=False) as z:
            need(set(z.files)==set(keys) if strict else set(keys)<=set(z.files),'Saved schema')
            return {k:z[k].copy() for k in keys}

def parameter_shapes():
    d={};incoming=8
    for i,out in enumerate((16,32,32,48,48,64)):
        c=3*i;d['encoder.%d.weight'%c]=(out,incoming,3,3);d['encoder.%d.bias'%c]=(out,);d['encoder.%d.weight'%(c+1)]=(out,);d['encoder.%d.bias'%(c+1)]=(out,);incoming=out
    for name,shape in [('shared.1',(64,576)),('support_head',(2,64)),('owner_head',(3,64)),('direct_end_head',(1,64))]:d[name+'.weight']=shape;d[name+'.bias']=(shape[0],)
    return d

def state_digest(state):
    h=hashlib.sha256()
    for key in sorted(state,key=str):
        a=state[key].detach().cpu().contiguous();h.update(str(key).encode());h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()

def checkpoint(binding,reported,protocol,arm,allow):
    need(allow.get(binding['path'])==binding,'Exact final checkpoint whitelist');checked(binding)
    import torch
    need(torch.__version__=='2.1.1+cu118','Pinned CPU checkpoint runtime')
    with opened(binding['path']) as f:z=torch.load(f,map_location='cpu',weights_only=True)
    need(set(z)=={'state_dict','optimizer_state','update','seed','arm','use_sam','protocol'} and z['update']==8000 and z['seed']==1810 and z['arm']==arm and z['use_sam']==(arm==ARMS[1]) and z['protocol']==protocol,'Final checkpoint envelope')
    shapes=parameter_shapes();sd=z['state_dict'];need(set(sd)==set(shapes) and sum(a.numel() for a in sd.values())==115222,'Exact32tensor/115222 state schema')
    for k,a in sd.items():need(a.dtype==torch.float32 and tuple(a.shape)==shapes[k] and bool(torch.isfinite(a).all()),'Finite final tensor '+k)
    need(state_digest(sd)==reported,'Reported final fullstate hash')
    opt=z['optimizer_state'];need(len(opt['param_groups'])==1,'Single inherited AdamW parameter group');group=opt['param_groups'][0]
    need(group['lr']==3e-4 and group['weight_decay']==1e-4 and len(group['params'])==32 and len(set(group['params']))==32,'AdamW fixed params')
    for key,ident in zip(shapes,group['params']):
        s=opt['state'][ident];need(set(s)=={'step','exp_avg','exp_avg_sq'},'AdamW final state members')
        need(float(s['step'])==8000,'Every parameter final8000 AdamW updates')
        for field in ('exp_avg','exp_avg_sq'):need(tuple(s[field].shape)==shapes[key] and s[field].dtype==torch.float32 and bool(torch.isfinite(s[field]).all()),'Finite AdamW moment')
    del z
    return dict(PASS=True,parameters=115222,tensors32=32,update8000=True,reported_fullstate_hash_exact=True)

def pools(dom):
    s,p,c=(dom[k] for k in ('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816'));av=s['x']['original_query_available'];y=s['y'];support=[]
    for j in range(2):support.append([np.flatnonzero(av&(y['support_target'][:,j]==label)) for label in (1,0)])
    mass=y['owner_soft'];eligible=av&np.isfinite(mass).all(1);w=np.where(eligible[:,None],np.nan_to_num(mass,nan=0),0.)
    need((w.sum(0)>0).all(),'Original owner class masses positive');conditional=[w[:,j]/w[:,j].sum(dtype=np.float64) for j in range(3)]
    hard=[]
    for cache,classes in ((p,range(3)),(c,range(2))):
        need((cache['y']['support_target']==-1).all(),'Owner labels not support targets');hard.append([np.flatnonzero(cache['x']['original_query_available']&(cache['y']['owner_target']==k)) for k in classes])
    need([len(q) for q in hard[0]]==[53,52,148] and [len(q) for q in hard[1]]==[157,157],'Unchanged public253/CF314')
    need(all(len(q)>0 for pair in support for q in pair),'Original support classes nonempty')
    return dict(support=support,conditional=conditional,public=hard[0],CF=hard[1])

def old_draw(rng,p):
    d={}
    d['visible']=np.concatenate([rng.choice(q,16,replace=True) for q in p['support'][0]])
    d['foreign']=np.concatenate([rng.choice(q,16,replace=True) for q in p['support'][1]])
    d['synthetic_owner']=np.concatenate([rng.choice(len(q),16,replace=True,p=q) for q in p['conditional']])
    d['public_owner']=np.concatenate([rng.choice(q,16,replace=True) for q in p['public']])
    d['CF_owner']=np.concatenate([rng.choice(q,16,replace=True) for q in p['CF']])
    return d

def loss_record(r,step):
    need(r['update']==step,'Exactly sequential1..8000 updates')
    parts=np.asarray(r['old_five_losses'],np.float64);need(parts.shape==(5,) and np.isfinite(parts).all() and (parts>=0).all(),'Five finite nonnegative losses')
    base=float(r['old_five_mean']);direct=float(r['direct_loss']);total=float(r['total_loss']);norm=float(r['grad_norm'])
    need(all(math.isfinite(q) for q in (base,direct,total,norm)) and min(base,direct,total,norm)>=0,'Finite loss/gradient norm logs')
    # Frozen arithmetic-only float32 reduction bound, not a data criterion.
    ideal=math.fsum(map(float,parts))/5;bound=8*np.finfo(np.float32).eps*max(1,abs(ideal),math.fsum(abs(float(q)) for q in parts)/5)
    need(abs(base-ideal)<=bound,'Unchanged arithmetic mean of5 within FP32 rounding')
    need(np.float32(base)+np.float32(direct)==np.float32(total),'Exact FP32 additive direct weight1 total')

def audit_log(binding,pool,dpool,oldhash,directhash):
    rng=np.random.default_rng(1810);drng=np.random.default_rng(1811);dh=hashlib.sha256();ddh=hashlib.sha256();fh=hashlib.sha256();n=0
    with opened(binding['path']) as f:
        for line in f:
            fh.update(line);r=json.loads(line);n+=1;loss_record(r,n);draw=old_draw(rng,pool);di=np.concatenate([drng.choice(q,16,replace=True) for q in dpool])
            need(list(r['original192_draws'])==list(draw),'Original draw domain order')
            for name,q in draw.items():need(np.array_equal(np.asarray(r['original192_draws'][name]),q),'Every original192 exact RNG draw');dh.update(name.encode());dh.update(q.astype('<i8').tobytes())
            need(np.array_equal(np.asarray(r['direct32_draws']),di),'Every direct16+16 RNG1811 draw');ddh.update(di.astype('<i8').tobytes())
            need(r['old_draw_sha']==dh.hexdigest() and r['direct_draw_sha']==ddh.hexdigest(),'Every cumulative draw hash')
    need(n==8000 and fh.hexdigest()==binding['sha256'] and dh.hexdigest()==oldhash and ddh.hexdigest()==directhash,'All8000 log bytes/final draws hash')
    return dict(PASS=True,updates8000=n,old192_RNG_exact=True,direct16_16_RNG_exact=True,loss_arithmetic_and_finite_logged_gradient_norm=True)

def metric(truth,pred):
    cm=[[0,0,0],[0,0,0]];unknown=0
    for t,p in zip(truth,pred):
        t=int(t);p=int(p);need(t in (-1,0,1) and p in (-1,0,1),'D/E/U values')
        if t<0:unknown+=1
        else:cm[t][p if p>=0 else 2]+=1
    classes=[]
    for c in range(2):
        tp=cm[c][c];truthn=sum(cm[c]);predn=cm[0][c]+cm[1][c]
        classes.append(dict(class_D0_E1=c,truth=truthn,predicted_knowntruth=predn,correct=tp,precision=tp/predn if predn else None,recall=tp/truthn if truthn else None))
    return dict(rows=len(truth),known=sum(sum(q) for q in cm),truth_UNKNOWN=unknown,CM_D_E_by_D_E_U=cm,classes=classes,known_abstentions=sum(q[2] for q in cm))

def selftest():
    r=dict(update=1,old_five_losses=[.1,.2,.3,.4,.5],old_five_mean=float(np.float32(.3)),direct_loss=float(np.float32(.7)),total_loss=1.,grad_norm=6.);loss_record(r,1)
    for key,value in [('update',2),('grad_norm',float('nan')),('total_loss',1.1)]:
        bad=dict(r);bad[key]=value
        try:loss_record(bad,1);raise RuntimeError('Bad log fixture passed')
        except AssertionError:pass
    need(sum(math.prod(q) for q in parameter_shapes().values())==115222 and len(parameter_shapes())==32,'Independent parameter schema')
    a=metric([0,0,1,1,-1,-1],[0,-1,1,0,1,-1]);need(a['CM_D_E_by_D_E_U']==[[1,0,1],[1,1,0]] and a['truth_UNKNOWN']==2 and a['known']==4 and a['classes'][0]['precision']==.5,'Unknown denominator and scalar CM')
    toy=dict(support=[[np.arange(2),np.arange(2,4)],[np.arange(4,6),np.arange(6,8)]],conditional=[np.ones(8)/8]*3,public=[np.arange(2),np.arange(2,4),np.arange(4,6)],CF=[np.arange(2),np.arange(2,4)])
    r1=np.random.default_rng(1810);r2=np.random.default_rng(1810);extra=np.random.default_rng(1811)
    for _ in range(3):
        d1=old_draw(r1,toy);extra.choice(9,32);d2=old_draw(r2,toy);need(sum(len(z) for z in d1.values())==192 and all(np.array_equal(d1[k],d2[k]) for k in d1),'Separated direct RNG')
    return dict(completed=True,PASS=True,invented_only=True,checks=7,actual_source_output_log_checkpoint_or_model_reads=0)

def run(entry_path):
    e=json.loads(raw(entry_path));need(e['approved_actual_independent_sourcefit_audit'] is True and e['both_models_completed'] is True and e['source_only'] is True and e['automatic_retry'] is False,'Postterminal root entry')
    p=json.loads(raw(PROTO));need(e['code']==p['code']==bind(Path(__file__).absolute()) and e['protocol']==bind(PROTO),'Frozen independent verifier')
    need(sys.executable==p['runtime']['executable'] and np.__version__==p['runtime']['numpy'],'Pinned independent runtime')
    jd(e['root_audit_scope']);fitentry=jd(p['fit_entry']);scope=jd(p['fit_scope']);done=jd(e['complete'])
    need(e['complete']['path']==str(RUN/'COMPLETE.json') and done['completed'] is True and done['PASS'] is True and done['source_only'] is True and done['entry']==p['fit_entry'],'Exact completed source fit')
    need(done['code']==fitentry['code'] and done['protocol']==fitentry['protocol'] and done['updates_each']==8000 and done['total_updates']==16000 and done['all2_frozen_before_outputs'] is True and set(done['arms'])==set(ARMS),'Frozen fit source/protocol/bothfinal')
    barrier=jd(done['all2_freeze']);need(barrier['completed'] is True and barrier['saved_outputs_emitted'] is False and barrier['evaluation_inputs_targets_or_scores_read'] is False,'All2-before-output barrier')
    need(barrier['checkpoints']=={arm:done['arms'][arm]['checkpoint'] for arm in ARMS},'Barrier final checkpoint bindings')
    need(done['training_caches']==dict(augmentation=fitentry['augmentation_complete'],public_CF=fitentry['original_public_CF_complete'],direct=fitentry['direct_cache_complete']),'Exact actual fit TRAIN bindings')
    need(len(done['same_initial_state_sha'])==64 and all(c in '0123456789abcdef' for c in done['same_initial_state_sha']),'Paired initial state SHA format')
    descriptors=scope['exact_source_TRAIN_array_and_metadata_whitelist_inherited_not_rehashed'];need(set(descriptors)==set(DOMAINS),'Only original TRAIN/direct domains')
    allowed={b['path']:b for d in descriptors.values() for k,b in d.items() if k in ('inputs','targets')}
    for arm in ARMS:
        a=done['arms'][arm]
        need(set(a['outputs'])==set(DOMAINS) and a['updates']==8000,'All source outputs, fixed8000')
        need(a['checkpoint']['path']==str(RUN/arm/'FINAL_CHECKPOINT.pt') and a['draws_and_losses']['path']==str(RUN/arm/'ALL8000_DRAWS_AND_LOSSES.jsonl'),'Only exact new final/log paths')
        allowed[a['checkpoint']['path']]=a['checkpoint']
        for domain,b in a['outputs'].items():need(b['path']==str(RUN/arm/('FINAL_'+domain+'_OUTPUTS.npz')),'Exact saved source output path');allowed[b['path']]=b
    out=H/'actual_review_v1';out.mkdir(exist_ok=False);start=time.monotonic();domains={};results={};readout=[]
    try:
        for name,n in DOMAINS.items():
            d=descriptors[name];x=arrays(d['inputs'],META,allowed);rows=jl(d['metadata']);need(len(rows)==n and len(set(x['row_id'].tolist()))==n,'Full source metadata rows')
            for i,r in enumerate(rows):
                if name!='DIRECT_CONTROL':need(r['split']==('TRAIN' if name=='SYNTH_AUG65536' else 'SOURCE_DEVELOPMENT'),'Original sourceTRAIN roles only')
                for key in META:
                    val=np.asarray([np.nan,np.nan] if r[key] is None else r[key],np.float64) if key=='query_xy' else r[key]
                    if key=='L' and val is None:val=np.nan
                    if key in ('query_xy','L'):need(np.array_equal(np.asarray(val),np.asarray(x[key][i]),equal_nan=True),'Original ordered q/L')
                    else:need(val==x[key][i],'Original ordered identity/availability')
            y=arrays(d['targets'],META+('direct_end_external_target','authority') if name=='DIRECT_CONTROL' else TARGET5,allowed)
            if name=='DIRECT_CONTROL':
                for key in META:need(eq(x[key],y[key]),'Direct sidecar META6 byte identity')
            domains[name]=dict(x=x,y=y,rows=rows)
        need([int((~domains[k]['x']['original_query_available']).sum()) for k in DOMAINS]==[1264,194,360,60],'All original U denominators')
        pool=pools(domains);dt=domains['DIRECT_CONTROL']['y']['direct_end_external_target'];dav=domains['DIRECT_CONTROL']['x']['original_query_available'];dpool=[np.flatnonzero(dav&(dt==c)) for c in (0,1)]
        need([len(q) for q in dpool]==[204,102] and int((dt<0).sum())==2486 and (dt[~dav]==-1).all(),'Exact306 direct known+2486unknown')
        need(done['same_initial_state_sha']==barrier['same_initial_state_sha'] and all(a['initial_state_sha']==done['same_initial_state_sha'] for a in done['arms'].values()),'Paired same initialization receipt')
        for arm in ARMS:
            a=done['arms'][arm];logs=audit_log(a['draws_and_losses'],pool,dpool,a['old_draw_sha'],a['direct_draw_sha']);ck=checkpoint(a['checkpoint'],a['final_state_sha'],done['protocol'],arm,allowed);perdomain={}
            for name,n in DOMAINS.items():
                z=arrays(a['outputs'][name],OUTKEYS,allowed,True);cache=domains[name];x=cache['x'];av=x['original_query_available']
                for key in META:need(eq(z[key],x[key]),'All savedsource META6 bytes/native/U exact')
                for key,shape in [('raw_support_logits',(n,2)),('raw_owner_logits',(n,3)),('raw_direct_end_external_logit',(n,))]:need(z[key].dtype==np.float32 and z[key].shape==shape and np.isfinite(z[key][av]).all() and np.isnan(z[key][~av]).all(),'Original U first/finite raw logits '+key)
                for key,expect in [('support_order',['visible_end_support','foreign_front_support']),('owner_order',['TARGET','FOREIGN','BACKGROUND']),('direct_end_order',['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1'])]:need(z[key].tolist()==expect,'Output task order')
                sp=np.full((n,2),-1,np.int8);op=np.full(n,-1,np.int8);dp=np.full(n,-1,np.int8);proxy=np.full(n,-1,np.int8)
                sp[av]=(z['raw_support_logits'][av]>0).astype(np.int8);op[av]=z['raw_owner_logits'][av].argmax(1).astype(np.int8);dp[av]=(z['raw_direct_end_external_logit'][av]>=0).astype(np.int8);proxy[av]=np.where(sp[av,1]==1,1,np.where(sp[av,0]==1,0,-1))
                for key,expected in [('support_prediction',sp),('owner_prediction',op),('direct_end_prediction_D0_E1_Uminus1',dp),('regional_support_proxy_D0_E1_Uminus1',proxy)]:need(eq(z[key],expected),'Independent saved decoder exact '+key)
                perdomain[name]=dict(rows=n,inputU=int((~av).sum()),direct_predictions=dict(Counter(map(int,dp))))
                if name=='DIRECT_CONTROL':
                    y=cache['y'];strata={'ALL':np.ones(n,bool)}
                    for authority in sorted(set(y['authority'].tolist())):strata['authority:'+authority]=y['authority']==authority
                    for native in range(8):strata['native:'+str(native)]=x['native_slot']==native
                    for domain in sorted({r['domain'] for r in cache['rows']}):strata['source_domain:'+domain]=np.array([r['domain']==domain for r in cache['rows']])
                    perdomain[name]['TRAIN_APPARENT_DIRECT_ONLY']={k:metric(dt[m],dp[m]) for k,m in strata.items()}
                    for i,r in enumerate(cache['rows']):readout.append(dict(arm=arm,row_id=str(x['row_id'][i]),original_global_index=int(x['original_global_index'][i]),native_slot=int(x['native_slot'][i]),original_query_available=bool(av[i]),authority=str(y['authority'][i]),source_domain=r['domain'],variant=r['variant'],target=int(dt[i]),direct_prediction=int(dp[i]),direct_rawlogit=float(z['raw_direct_end_external_logit'][i]) if av[i] else None,truth_UNKNOWN_excluded_from_CM=bool(dt[i]<0),source_TRAIN_apparent_not_validation=True))
            results[arm]=dict(logs=logs,checkpoint=ck,source_outputs=perdomain)
        need(all(done['arms'][arm]['old_draw_sha']==done['same_old192_draw_sha']==barrier['old_same_draw_sha'] and done['arms'][arm]['direct_draw_sha']==done['same_direct32_draw_sha']==barrier['direct_same_draw_sha'] for arm in ARMS),'Both full8000 draw streams same')
        # Static trainer ordering plus file creation ordering corroborate barrier.
        barrier_time=safe(done['all2_freeze']['path']).stat().st_mtime_ns
        for arm in ARMS:
            need(safe(done['arms'][arm]['checkpoint']['path']).stat().st_mtime_ns<=barrier_time,'Final checkpoint before barrier')
            for b in done['arms'][arm]['outputs'].values():need(safe(b['path']).stat().st_mtime_ns>=barrier_time,'Source saved outputs after bothfinal barrier')
        with (out/'ALL5584_DIRECT_TRAIN_READOUT.jsonl').open('x') as f:
            for r in readout:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        write(out/'SOURCE_TRAIN_APPARENT_METRICS.json',dict(completed=True,PASS=True,not_validation_or_real_improvement=True,known306=306,truthUNKNOWN2486=2486,models=results,choice_or_threshold_or_model_selection=False))
        report=dict(completed=True,PASS=True,code=bind(Path(__file__).absolute()),protocol=bind(PROTO),actual_entry=bind(entry_path),source_fit_complete=e['complete'],all16000_log_rows_verified=True,all_old192_and_direct32_RNG_exact=True,both_final_checkpoints115222_finite_statehash_exact=True,all2_before_sourceoutput_barrier_preserved=True,all_four_domains_source_outputs_META6_and_U_exact=True,direct_known306_unknown2486_separate=True,metrics=bind(out/'SOURCE_TRAIN_APPARENT_METRICS.json'),direct_rows=bind(out/'ALL5584_DIRECT_TRAIN_READOUT.jsonl'),seconds=time.monotonic()-start,source_training_readout_not_validation_or_real_improvement=True,model_forward_or_fit_or_encoder_or_evaluation_reads=0,model_instantiations=0,CPU_final_checkpoint_loads_only=2,automatic_retry=False,limits=['Logged five component means checked with prospective8eps float32 arithmetic bound; additive total exact FP32. Per-step direct CE/gradient tensors cannot be recreated without forbidden fullfit/forward replay; recorded finite gradient norms and frozen trainer checks are the evidence.','Same initialization is cross-bound sealed hash receipt; no production model is instantiated to recreate initial weights.','No model/threshold/step/data selection or claim of real generalization; direct weak authorities remain distinct.'])
        write(out/'REVIEW.json',report);print(json.dumps(dict(completed=True,PASS=True,review=bind(out/'REVIEW.json'),seconds=report['seconds'])),flush=True)
    except BaseException:
        write(out/'FAILURE.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.selftest:print(json.dumps(selftest()))
    elif a.entry:run(a.entry)
    else:ap.error('--selftest or --entry')
