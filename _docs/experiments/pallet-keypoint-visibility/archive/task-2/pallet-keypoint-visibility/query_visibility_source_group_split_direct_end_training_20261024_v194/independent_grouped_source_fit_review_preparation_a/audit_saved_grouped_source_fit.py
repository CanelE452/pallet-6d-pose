"""Independent saved grouped source-fit audit. No model/producer import/forward."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,os,stat,sys,time,traceback
import numpy as np
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PROTO=H/'PROTOCOL_BEFORE_GROUPED_SOURCE_FIT_OUTPUT_BODIES.json'
RUN=V/'runs/C56_C57_DIRECT_END_GROUPED_SOURCE_FIXED8000'
ARMS=('C56_RAW96_DIRECT_END_GROUPED','C57_RAW96_PLUS_FROZEN_SAM_DIRECT_END_GROUPED')
DOMAINS={'SYNTH_AUG65536':65536,'PUBLIC_TRAIN1480':1480,'CF3816':3816,'DIRECT_CONTROL':3144}
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
TARGET5=('support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass')
ROWKEYS=META+('raw_support_logits','raw_owner_logits','raw_direct_end_external_logit','support_prediction','owner_prediction','direct_end_prediction_D0_E1_Uminus1','regional_support_proxy_D0_E1_Uminus1')
ORDERS=('support_order','owner_order','direct_end_order')
OUTKEYS=set(ROWKEYS+ORDERS)
def need(v,msg):
    if not bool(v):raise AssertionError(msg)

def safe(value):
    p=Path(value);need(p.is_absolute() and '..' not in p.parts and all('v112' not in q.casefold() for q in p.parts),'Safe lexical name before access')
    need(p.is_relative_to(P) and not any(q in str(p).casefold() for q in ('current223','_v169/','_v177/','_v178/','_v184/','_v191/','_v195/','_v182/','synthetic_val')),'Bound source project only')
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

def checkpoint(binding,reported,protocol,arm,allow,source_split):
    need(allow.get(binding['path'])==binding,'Exact final checkpoint whitelist');checked(binding)
    import torch
    need(torch.__version__=='2.1.1+cu118','Pinned CPU checkpoint runtime')
    with opened(binding['path']) as f:z=torch.load(f,map_location='cpu',weights_only=True)
    need(set(z)=={'state_dict','optimizer_state','update','seed','arm','use_sam','protocol','source_split'} and z['update']==8000 and z['seed']==1810 and z['arm']==arm and z['use_sam']==(arm==ARMS[1]) and z['protocol']==protocol and z['source_split']==source_split,'Final checkpoint envelope')
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

def metadata_exact(rows,x,domain):
    n=DOMAINS[domain];need(len(rows)==n==len(x['row_id']) and len(set(x['row_id'].tolist()))==n,'All ordered original occurrence IDs; source indices may repeat')
    need(x['original_global_index'].dtype==np.int64 and x['native_slot'].dtype==np.int8 and x['query_xy'].dtype==np.float64 and x['L'].dtype==np.float64 and x['original_query_available'].dtype==bool,'Original META6 types')
    for i,r in enumerate(rows):
        if domain!='DIRECT_CONTROL':need(r['split']==('TRAIN'if domain=='SYNTH_AUG65536'else'SOURCE_DEVELOPMENT'),'Source-only original split')
        for key in META:
            value=np.asarray([np.nan,np.nan]if r[key]is None else r[key],np.float64)if key=='query_xy'else r[key]
            if key=='L'and value is None:value=np.nan
            need(np.array_equal(np.asarray(value),np.asarray(x[key][i]),equal_nan=True)if key in('query_xy','L')else value==x[key][i],'Ordered source metadata '+key)

def split_masks(split,domains):
    components=jl(split['outputs']['ALL_COMPONENTS_AND_CANONICAL_ORIGIN_NODES.jsonl']);roles={}
    for r in components:
        cid=hashlib.sha256('\n'.join(sorted(r['origin_nodes'])).encode()).hexdigest();role='HOLDOUT'if int(hashlib.sha256(('pallet_visibility_source_groups_v194_20261024\n'+cid).encode()).hexdigest()[:16],16)%100<20 else'TRAIN'
        need(cid==r['component_id']and r['role']==role and cid not in roles,'Fixed component SHA/source role');roles[cid]=role
    masks={};aligned={}
    for domain,cache in domains.items():
        plan=jl(split['outputs'][domain+'_SPLIT_ROWS.jsonl']);need(len(plan)==DOMAINS[domain],'All source role rows retained')
        if domain=='DIRECT_CONTROL':
            bykey={tuple(r['control_join_key']):r for r in plan};need(len(bykey)==len(plan),'Unique direct logical occurrence keys')
            rr=[];used=set()
            for i,m in enumerate(cache['rows']):
                key=tuple(m['control_join_key']);need(key==(m['domain'],m['attempt_id'],m['variant'],m['native_slot'])and key in bykey and key not in used,'Exact canonical direct key; no order alias');used.add(key);r=bykey[key]
                need(m['cache_index']==i and m['source_packet']==r['source_packet']and m['authority']==r['authority'],'Exact direct packet/authority')
                target=int(cache['y']['direct_end_external_target'][i]);need(target==r['stored_weak_direct_target']and(target<0 or m['root_admitted']is True),'Root-only selected direct label')
                rr.append(dict(r,source_split_cache_index=r['cache_index'],cache_index=i))
            need(used==set(bykey),'No missing/additional direct split occurrence')
        else:rr=plan
        mask=np.zeros(len(rr),bool)
        for i,(m,r)in enumerate(zip(cache['rows'],rr)):
            need(r['cache_index']==i and r['role']==roles.get(r['component_id']),'Component TRAIN/HOLDOUT exact')
            for key in META:
                if key in('query_xy','L'):
                    aa=np.asarray([np.nan,np.nan]if key=='query_xy'and m[key]is None else np.nan if m[key]is None else m[key],np.float64);bb=np.asarray([np.nan,np.nan]if key=='query_xy'and r[key]is None else np.nan if r[key]is None else r[key],np.float64);need(np.array_equal(aa,bb,equal_nan=True),'Split q/L exact')
                else:need(r[key]==m[key],'Split original META6 exact')
            mask[i]=r['role']=='TRAIN'
        masks[domain]=mask;aligned[domain]=rr
    return masks,aligned

def filtered_pools(dom,masks):
    s,p,c=(dom[k]for k in('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816'));av=s['x']['original_query_available'];y=s['y'];train=masks['SYNTH_AUG65536']
    support=[[np.flatnonzero(av&train&(y['support_target'][:,j]==label))for label in(1,0)]for j in range(2)]
    mass=y['owner_soft'];eligible=av&np.isfinite(mass).all(1);w=np.where(eligible[:,None],np.nan_to_num(mass,nan=0),0.)
    need((w.sum(0)>0).all()and ((mass[np.isfinite(mass).all(1)]>=0).all()),'Original partial known soft masses')
    conditional=[]
    for j in range(3):
        # Preserve the declared inherited full-pool normalization operation,
        # then zero HOLDOUT and normalize only retained TRAIN class mass.
        original=w[:,j]/w[:,j].sum(dtype=np.float64);weights=np.where(train,original,0.);total=weights.sum(dtype=np.float64);need(total>0,'Nonempty TRAIN soft class');conditional.append(weights/total)
    hard=[]
    for cache,domain,classes,expected in((p,'PUBLIC_TRAIN1480',range(3),[53,52,148]),(c,'CF3816',range(2),[157,157])):
        need((cache['y']['support_target']==-1).all(),'Public/CF remain owner-only')
        full=[np.flatnonzero(cache['x']['original_query_available']&(cache['y']['owner_target']==k))for k in classes];need([len(v)for v in full]==expected,'Original full weak owner denominators')
        hard.append([q[masks[domain][q]]for q in full])
    need(all(len(q)>0 for group in support+hard for q in group),'Every TRAIN support/hard-owner class remains nonempty, no fallback')
    direct=dom['DIRECT_CONTROL'];dt=direct['y']['direct_end_external_target'];dav=direct['x']['original_query_available'];dm=masks['DIRECT_CONTROL']
    need(dt.dtype==np.int8 and np.isin(dt,[-1,0,1]).all()and(dt[~dav]==-1).all(),'Direct weak UNKNOWN/U retained')
    dpool=[np.flatnonzero(dav&dm&(dt==k))for k in(0,1)];need(all(len(v)>0 for v in dpool),'Both TRAIN direct classes')
    return dict(support=support,conditional=conditional,public=hard[0],CF=hard[1]),dpool

def audit_log(binding,pool,dpool,oldhash,directhash,masks):
    rng=np.random.default_rng(1810);drng=np.random.default_rng(1811);dh=hashlib.sha256();ddh=hashlib.sha256();fh=hashlib.sha256();n=0
    mapping={'visible':'SYNTH_AUG65536','foreign':'SYNTH_AUG65536','synthetic_owner':'SYNTH_AUG65536','public_owner':'PUBLIC_TRAIN1480','CF_owner':'CF3816'}
    with opened(binding['path'])as f:
        for line in f:
            fh.update(line);r=json.loads(line);n+=1;loss_record(r,n);need(r['all_drawn_rows_TRAIN']is True,'Actual perstep TRAIN assertion')
            draw=old_draw(rng,pool);di=np.concatenate([drng.choice(q,16,replace=True)for q in dpool]);need(list(r['original192_draws'])==list(draw),'Fixed five-domain draw order')
            for name,q in draw.items():
                need(np.array_equal(np.asarray(r['original192_draws'][name]),q)and masks[mapping[name]][q].all(),'Exact RNG and every old draw TRAIN');dh.update(name.encode());dh.update(q.astype('<i8').tobytes())
            need(np.array_equal(np.asarray(r['direct32_draws']),di)and masks['DIRECT_CONTROL'][di].all(),'Exact direct RNG1811 and every direct draw TRAIN');ddh.update(di.astype('<i8').tobytes())
            need(r['old_draw_sha']==dh.hexdigest()and r['direct_draw_sha']==ddh.hexdigest(),'All cumulative paired draw hashes')
    need(n==8000 and fh.hexdigest()==binding['sha256']and dh.hexdigest()==oldhash and ddh.hexdigest()==directhash,'Full8000 log/source SHA/finaldraw')
    return dict(PASS=True,updates=8000,all192_plus32_RNG_draws_exact=True,all_drawn_rows_TRAIN=True,logged_loss_arithmetic_and_gradient_norm_finite=True)

def validate_output(z,x):
    n=len(x['row_id']);av=x['original_query_available']
    for key in META:need(eq(z[key],x[key]),'Saved output META6 bytes/U exact')
    for key,shape in [('raw_support_logits',(n,2)),('raw_owner_logits',(n,3)),('raw_direct_end_external_logit',(n,))]:need(z[key].dtype==np.float32 and z[key].shape==shape and np.isfinite(z[key][av]).all()and np.isnan(z[key][~av]).all(),'Finite eligible/rawNaN U '+key)
    for key,expected in [('support_order',['visible_end_support','foreign_front_support']),('owner_order',['TARGET','FOREIGN','BACKGROUND']),('direct_end_order',['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1'])]:need(z[key].tolist()==expected,'Distinct task order')
    sp=np.full((n,2),-1,np.int8);op=np.full(n,-1,np.int8);dp=np.full(n,-1,np.int8);proxy=np.full(n,-1,np.int8)
    for i in range(n):
        if av[i]:
            sp[i]=[int(float(v)>0)for v in z['raw_support_logits'][i]];values=[float(v)for v in z['raw_owner_logits'][i]];op[i]=max(range(3),key=lambda k:values[k]);dp[i]=int(float(z['raw_direct_end_external_logit'][i])>=0);proxy[i]=1 if sp[i,1]==1 else 0 if sp[i,0]==1 else -1
    for key,expected in [('support_prediction',sp),('owner_prediction',op),('direct_end_prediction_D0_E1_Uminus1',dp),('regional_support_proxy_D0_E1_Uminus1',proxy)]:need(eq(z[key],expected),'Independent scalar saved decoder '+key)
    return dp

def audit_subsets(a,domain,full,masks,aligned,allowed):
    mask=masks[domain];n=len(mask);need(set(a['source_cohorts'][domain])=={'TRAIN','HOLDOUT'},'Both source cohorts preserved')
    all_indices=[];counts={}
    for role,take in [('TRAIN',mask),('HOLDOUT',~mask)]:
        indices=np.flatnonzero(take);all_indices.extend(map(int,indices));r=a['source_cohorts'][domain][role];z=arrays(r['outputs'],OUTKEYS,allowed,True)
        for key in ROWKEYS:need(eq(z[key],full[key][indices]),'Every cohort row/score/prediction exact subset')
        for key in ORDERS:need(eq(z[key],full[key]),'Cohort task orders unchanged')
        line=jl(r['metadata']);need(len(line)==len(indices)==r['rows']and int((~z['original_query_available']).sum())==r['original_inputU'],'Full cohort rows/U')
        for j,i in enumerate(indices):
            q=line[j];source=aligned[domain][i];need(q['cache_index']==int(i)and q['row_id']==str(full['row_id'][i])and q['component_id']==source['component_id']and q['role']==role and q['original_global_index']==int(full['original_global_index'][i])and q['native_slot']==int(full['native_slot'][i])and q['original_query_available']==bool(full['original_query_available'][i])and q['direct_target_authority']==source.get('authority')and q['control_join_key']==source.get('control_join_key'),'Cohort full lineage/authority exact')
        counts[role]=dict(rows=len(indices),inputU=r['original_inputU'])
    need(len(all_indices)==n and sorted(all_indices)==list(range(n)),'TRAIN/HOLDOUT disjoint union equals full original output')
    return counts

def fresh_review(path):
    p=safe(path);need(p==V/'independent_grouped_source_fit_review_a/actual_review_v1','Fresh fixed actual audit output')
    cur=Path(p.anchor)
    for part in p.parts[1:-1]:
        cur=cur/part
        try:mode=cur.lstat().st_mode
        except FileNotFoundError:cur.mkdir();mode=cur.lstat().st_mode
        need(stat.S_ISDIR(mode),'Ordinary audit output ancestors')
    p.mkdir(exist_ok=False);return p

def file_time(b):
    with opened(b['path'])as f:return os.fstat(f.fileno()).st_mtime_ns

def cohort_counts(cache,mask):
    result={};x,y=cache['x'],cache['y']
    for role,take in [('TRAIN',mask),('HOLDOUT',~mask)]:
        ii=np.flatnonzero(take);r=dict(rows=len(ii),original_inputU=int((~x['original_query_available'][ii]).sum()),native={str(k):int((x['native_slot'][ii]==k).sum())for k in range(8)})
        if 'direct_end_external_target'in y:
            target=y['direct_end_external_target'][ii];r.update(direct_D=int((target==0).sum()),direct_E=int((target==1).sum()),direct_UNKNOWN=int((target<0).sum()),known_by_authority={str(authority):int(((y['authority'][ii]==authority)&(target>=0)).sum())for authority in np.unique(y['authority'])})
        result[role]=r
    return result

def run(entry_path):
    e=json.loads(raw(entry_path));need(e['approved_actual_independent_grouped_sourcefit_audit']is True and e['both_models_completed']is True and e['source_only']is True and e['automatic_retry']is False,'Separate postterminal root audit entry')
    p=json.loads(raw(PROTO));need(e['code']==p['code']==bind(Path(__file__).absolute())and e['protocol']==bind(PROTO),'Frozen saved-fit auditor')
    need(sys.executable==p['runtime']['executable']and np.__version__==p['runtime']['numpy'],'Pinned saved audit runtime')
    jd(e['root_audit_scope']);done=jd(e['complete']);fitentry=jd(done['entry'])
    need(e['complete']['path']==str(RUN/'COMPLETE.json')and done['completed']is True and done['PASS']is True and done['source_only']is True and done['code']==p['fit_code']and done['protocol']==p['fit_protocol'],'Exact completed grouped fit')
    need(done['steps_each']==8000 and done['total_updates']==16000 and done['all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs']is True and set(done['arms'])==set(ARMS),'All2 final fixed8000 barrier')
    need(done['code']==fitentry['code']and done['protocol']==fitentry['protocol']and done['source_split']==p['split_complete']==fitentry['split_complete'],'Frozen actual recipe/split binding')
    barrier=jd(done['all2_freeze']);need(barrier['completed']is True and barrier['source_TRAIN_or_HOLDOUT_prediction_emitted']is False and barrier['protected_real_COMMON_VAL_access']is False and barrier['checkpoint_or_arm_selection']is False,'All2-before-any-source-output barrier')
    need(barrier['checkpoints']=={arm:done['arms'][arm]['checkpoint']for arm in ARMS},'Barrier final checkpoint bindings')
    need(done['training_caches']==dict(augmentation=fitentry['augmentation_complete'],public_CF=fitentry['original_public_CF_complete'],direct=fitentry['direct_cache_complete']),'Exact actual source cache bindings')
    aug=jd(done['training_caches']['augmentation']);orig=jd(done['training_caches']['public_CF']);direct=jd(done['training_caches']['direct']);split=jd(done['source_split'])
    descriptors={'SYNTH_AUG65536':aug,'PUBLIC_TRAIN1480':orig['domains']['PUBLIC_TRAIN1480'],'CF3816':orig['domains']['CF3816'],'DIRECT_CONTROL':direct}
    allowed={b['path']:b for d in descriptors.values()for key,b in d.items()if key in('inputs','targets')}
    for arm in ARMS:
        a=done['arms'][arm];need(a['updates']==8000 and set(a['outputs'])==set(DOMAINS)and set(a['source_cohorts'])==set(DOMAINS),'All domains full and both cohorts')
        need(a['checkpoint']['path']==str(RUN/arm/'FINAL_CHECKPOINT.pt')and a['draws_and_losses']['path']==str(RUN/arm/'ALL8000_DRAWS_AND_LOSSES.jsonl'),'Only exact final checkpoints/logs')
        allowed[a['checkpoint']['path']]=a['checkpoint']
        for domain,b in a['outputs'].items():
            need(b['path']==str(RUN/arm/('FINAL_'+domain+'_OUTPUTS.npz')),'Exact full source output path');allowed[b['path']]=b
            for role in('TRAIN','HOLDOUT'):
                q=a['source_cohorts'][domain][role];need(q['outputs']['path']==str(RUN/arm/('FINAL_'+domain+'_'+role+'_OUTPUTS.npz'))and q['metadata']['path']==str(RUN/arm/(domain+'_'+role+'_ROW_COMPONENTS.jsonl')),'Exact source cohort artifacts');allowed[q['outputs']['path']]=q['outputs']
    out=fresh_review(p['output_directory']);start=time.monotonic();domains={};results={};readout={}
    try:
        for domain,n in DOMAINS.items():
            descriptor=descriptors[domain];x=arrays(descriptor['inputs'],META,allowed);rows=jl(descriptor['metadata']);metadata_exact(rows,x,domain)
            y=arrays(descriptor['targets'],META+('direct_end_external_target','authority')if domain=='DIRECT_CONTROL'else TARGET5,allowed)
            if domain=='DIRECT_CONTROL':
                for key in META:need(eq(y[key],x[key]),'Direct target META6 byte identity')
                for i,r in enumerate(rows):need(str(y['authority'][i])==r['authority']and(int(y['direct_end_external_target'][i])<0 or r['root_admitted']is True),'Separate root direct weak authority')
            domains[domain]=dict(x=x,y=y,rows=rows)
        need([int((~domains[k]['x']['original_query_available']).sum())for k in DOMAINS]==[1264,194,360,112],'Every full original inputU denominator')
        masks,aligned=split_masks(split,domains);pool,dpool=filtered_pools(domains,masks);dt=domains['DIRECT_CONTROL']['y']['direct_end_external_target'];dm=masks['DIRECT_CONTROL']
        need(done['cohort_denominators']=={name:cohort_counts(cache,masks[name])for name,cache in domains.items()},'Every full TRAIN/HOLDOUT native/U/authority denominator')
        need([int((dt==k).sum())for k in(0,1)]==[248,124]and[len(q)for q in dpool]==[218,109]and[int(((dt==k)&~dm).sum())for k in(0,1)]==[30,15],'Full372 weak labels/TRAIN327/HOLDOUT45')
        need(all(r['domain']=='SYNTH_FINITE_END'for i,r in enumerate(domains['DIRECT_CONTROL']['rows'])if not dm[i]and dt[i]>=0),'All45 source heldout direct labels synthetic, real-end holdout0')
        initial=done['same_initial_state_sha'];need(len(initial)==64 and all(c in'0123456789abcdef'for c in initial)and barrier['same_initial_state_sha']==initial and all(a['initial_state_sha']==initial for a in done['arms'].values()),'Same initial state cross-bound sealed receipt')
        for arm in ARMS:
            a=done['arms'][arm];logs=audit_log(a['draws_and_losses'],pool,dpool,a['old_draw_sha'],a['direct_draw_sha'],masks);ck=checkpoint(a['checkpoint'],a['final_state_sha'],done['protocol'],arm,allowed,done['source_split']);perdomain={}
            for domain,n in DOMAINS.items():
                z=arrays(a['outputs'][domain],OUTKEYS,allowed,True);cache=domains[domain];dp=validate_output(z,cache['x']);cohorts=audit_subsets(a,domain,z,masks,aligned,allowed)
                perdomain[domain]=dict(rows=n,inputU=int((~cache['x']['original_query_available']).sum()),cohorts=cohorts,all_full_and_subset_output_bytes_exact=True)
                if domain=='DIRECT_CONTROL':
                    strata={'ALL':np.ones(n,bool),'TRAIN':dm,'HOLDOUT':~dm};readout[arm]={name:metric(dt[take],dp[take])for name,take in strata.items()}
                    readout[arm]['by_authority']={str(authority):{role:metric(dt[take&(cache['y']['authority']==authority)],dp[take&(cache['y']['authority']==authority)])for role,take in strata.items()}for authority in np.unique(cache['y']['authority'])}
            results[arm]=dict(logs=logs,checkpoint=ck,source_output_domains=perdomain)
        need(all(a['old_draw_sha']==done['same_filtered_old192_draw_sha']==barrier['same_filtered_old192_draw_sha']and a['direct_draw_sha']==done['same_direct32_draw_sha']==barrier['same_direct32_draw_sha']for a in done['arms'].values()),'Both paired full8000 exact draw streams')
        bt=file_time(done['all2_freeze'])
        for a in done['arms'].values():
            need(file_time(a['checkpoint'])<=bt,'Final checkpoint before barrier')
            for b in list(a['outputs'].values())+[q['outputs']for domains_ in a['source_cohorts'].values()for q in domains_.values()]:need(file_time(b)>=bt,'Every source output after all2 final barrier')
        write(out/'SOURCE_GROUPED_DIRECT_READOUT.json',dict(completed=True,PASS=True,source_only=True,models=readout,full372_train327_holdout45=True,heldout_direct45_synthetic_only_real_end_holdout0=True,source_design_historically_known_not_freshblind=True,C54_C55_not_retroactively_validated=True,natural_real_generalization_or_native_gold=False,selected_model=None,adopted=False,fully_solved=False))
        report=dict(completed=True,PASS=True,code=bind(Path(__file__).absolute()),protocol=bind(PROTO),actual_entry=bind(entry_path),source_fit_complete=e['complete'],source_split=done['source_split'],all16000_log_rows_verified=True,all192_and_direct32_RNG_exact_and_TRAIN_only=True,both_final_checkpoints115222_finite_statehash_and_AdamW8000_exact=True,all2_before_TRAIN_or_HOLDOUT_prediction_barrier=True,all147952_full_output_rows_META6_U_and_scalar_decoders_exact=True,all_full_vs_TRAIN_HOLDOUT_disjoint_union_and_every_output_byte_exact=True,full_inputU=[1264,194,360,112],direct_weak_known_D_E=[248,124],TRAIN_D_E=[218,109],HOLDOUT_D_E=[30,15],direct_holdout_synthetic_only_real_end_holdout0=True,results=results,source_readout=bind(out/'SOURCE_GROUPED_DIRECT_READOUT.json'),seconds=time.monotonic()-start,model_instantiations=0,model_forward_fit_or_encoder_or_evaluation_calls=0,CPU_final_checkpoint_loads_only=2,automatic_retry=False,limits=['Logged five component means use prospective8eps FP32 bound and additive direct total exactFP32. Perstep CE/gradient tensors cannot be recreated without forbidden model forward/fit; finite logged gradients and frozen source code are checked.','Initial state is paired sealed SHA receipt, not regenerated by instantiating a source model.','Full source target containers are read for authority guards and independent draw reconstruction. Only TRAIN pools enter any reconstructed draw; no source score-based choice.','Grouped heldout direct45 labels are synthetic only, realend holdout0, historical design known. This is not retrospective independent validation of C54/C55 or natural-image/native-gold/transfer success.'])
        write(out/'REVIEW.json',report);print(json.dumps(dict(completed=True,PASS=True,review=bind(out/'REVIEW.json'),seconds=report['seconds'])),flush=True)
    except BaseException:
        write(out/'FAILURE.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',type=Path,required=True);a=ap.parse_args();run(a.entry)

