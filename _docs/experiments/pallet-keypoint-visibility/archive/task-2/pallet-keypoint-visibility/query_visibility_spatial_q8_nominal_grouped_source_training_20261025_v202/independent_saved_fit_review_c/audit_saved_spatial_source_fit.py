"""Independent V202 saved fit audit; no model construction, forward or input-pixel NPZ."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,math,os,stat,sys,time,traceback
import numpy as np
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PROTO=H/'PROTOCOL_BEFORE_SAVED_SPATIAL_SOURCE_FIT_BODIES.json'
RUN=V/'runs/C60_C61_SPATIAL_Q8_NOMINAL_SOURCE_FIXED4000'
ARMS=('C60_RAW96_NATIVE_MASK_ZERO_XY_NOMINAL_CONTROL','C61_RAW96_NATIVE_MASK_Q8_XY_NOMINAL')
DOMAINS={'SYNTH_AUG65536':65536,'PUBLIC_TRAIN1480':1480,'CF3816':3816,'DIRECT_CONTROL':3144}
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
TARGET5=('support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass')
ROWKEYS=META+('raw_support_logits','raw_owner_logits','raw_direct_end_external_logit','support_prediction','owner_prediction','direct_end_prediction_D0_E1_Uminus1','regional_support_proxy_D0_E1_Uminus1')
ORDERS=('support_order','owner_order','direct_end_order');OUTKEYS=set(ROWKEYS+ORDERS)


def need(v,msg):
    if not bool(v):raise AssertionError(msg)

def safe(value):
    p=Path(value);need(p.is_absolute() and '..' not in p.parts and all('v112' not in q.casefold() for q in p.parts),'Safe lexical name before access')
    need(p.is_relative_to(P) and not any(q in str(p).casefold() for q in ('current223','_v169/','_v177/','_v178/','_v184/','_v191/','_v195/','_v200/','_v182/','_v191/','_v178/','public_held','common4096','synthetic_val')),'Bound source project only')
    return p

def opened(path):
    p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for q in p.parts[1:-1]:
            end=os.open(q,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=end
        last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(last).st_mode),'Regular nofollow file');return os.fdopen(last,'rb')

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

def state_digest(state):
    h=hashlib.sha256()
    for key in sorted(state,key=str):
        a=state[key].detach().cpu().contiguous();h.update(str(key).encode());h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()

def old_draw(rng,p):
    d={}
    d['visible']=np.concatenate([rng.choice(q,16,replace=True) for q in p['support'][0]])
    d['foreign']=np.concatenate([rng.choice(q,16,replace=True) for q in p['support'][1]])
    d['synthetic_owner']=np.concatenate([rng.choice(len(q),16,replace=True,p=q) for q in p['conditional']])
    d['public_owner']=np.concatenate([rng.choice(q,16,replace=True) for q in p['public']])
    d['CF_owner']=np.concatenate([rng.choice(q,16,replace=True) for q in p['CF']])
    return d

def loss_record(r,step):
    need(r['update']==step,'Exactly sequential1..4000 updates')
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
    need(n==4000 and fh.hexdigest()==binding['sha256']and dh.hexdigest()==oldhash and ddh.hexdigest()==directhash,'Full4000 log/source SHA/finaldraw')
    return dict(PASS=True,updates=4000,all192_plus32_RNG_draws_exact=True,all_drawn_rows_TRAIN=True,logged_loss_arithmetic_and_gradient_norm_finite=True)

def validate_output(z,x):
    n=len(x['row_id']);av=x['original_query_available']
    meta_values_exact(z,x)
    for key,shape in [('raw_support_logits',(n,2)),('raw_owner_logits',(n,3)),('raw_direct_end_external_logit',(n,))]:need(z[key].dtype==np.float32 and z[key].shape==shape and np.isfinite(z[key][av]).all()and np.isnan(z[key][~av]).all(),'Finite eligible/rawNaN U '+key)
    for key,expected in [('support_order',['visible_end_support','foreign_front_support']),('owner_order',['TARGET','FOREIGN','BACKGROUND']),('direct_end_order',['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1'])]:need(z[key].tolist()==expected,'Distinct task order')
    sp=np.full((n,2),-1,np.int8);op=np.full(n,-1,np.int8);dp=np.full(n,-1,np.int8);proxy=np.full(n,-1,np.int8)
    for i in range(n):
        if av[i]:
            sp[i]=[int(float(v)>0)for v in z['raw_support_logits'][i]];values=[float(v)for v in z['raw_owner_logits'][i]];op[i]=max(range(3),key=lambda k:values[k]);dp[i]=int(float(z['raw_direct_end_external_logit'][i])>=0);proxy[i]=1 if sp[i,1]==1 else 0 if sp[i,0]==1 else -1
    for key,expected in [('support_prediction',sp),('owner_prediction',op),('direct_end_prediction_D0_E1_Uminus1',dp),('regional_support_proxy_D0_E1_Uminus1',proxy)]:need(eq(z[key],expected),'Independent scalar saved decoder '+key)
    return dp

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

def independent_native_priors(cache,mask):
    x=cache['x'];y=cache['y']['direct_end_external_target'];rr=cache['rows'];n=len(y)
    need(mask.dtype==bool and mask.shape==(n,),'All-row TRAIN mask for independent prior')
    need(x['native_slot'].dtype==np.int8 and np.isin(x['native_slot'],range(8)).all(),'Preserved native8')
    totals=[sum(bool(mask[i])and bool(x['original_query_available'][i])and int(y[i])==k for i in range(n))for k in(0,1)];need(all(totals),'Both TRAIN classes')
    records=[]
    for domain in ['ALL']+sorted({r['domain']for r in rr}):
        for native in range(8):
            ids=[i for i in range(n)if mask[i]and int(x['native_slot'][i])==native and(domain=='ALL'or rr[i]['domain']==domain)]
            D=sum(bool(x['original_query_available'][i])and int(y[i])==0 for i in ids);E=sum(bool(x['original_query_available'][i])and int(y[i])==1 for i in ids)
            d=D/totals[0];e=E/totals[1]
            records.append(dict(domain=domain,native_slot=native,TRAIN_rows=len(ids),original_inputU=sum(not bool(x['original_query_available'][i])for i in ids),direct_D=D,direct_E=E,direct_UNKNOWN=sum(int(y[i])<0 for i in ids),raw_known_E_fraction=E/(D+E)if D+E else None,balanced_draw_expected_E_given_domain_native=e/(d+e)if d+e else None,raw_D_equals_2E=D==2*E if D+E else None))
    return dict(completed=True,TRAIN_only=True,TRAIN_known_class_totals=dict(D=totals[0],E=totals[1]),rows=records,not_finite_draw_realization=True,not_public_or_owner_auxiliary_prior=True,architecture_or_data_selection=False)

def canonical_census(value):
    if isinstance(value,dict):return {k:canonical_census(v)for k,v in value.items()}
    if isinstance(value,list):
        result=[canonical_census(v)for v in value]
        return sorted(result,key=lambda q:q['identity'])if result and all(isinstance(q,dict)and'identity'in q for q in result)else result
    return value

def raw(path):
    need(Path(path).suffix.casefold() not in ('.npz','.npy','.pt','.pth','.safetensors','.bin'),'No generic binary/input access')
    with opened(path)as f:return f.read()

def metadata_x(rows):
    return dict(row_id=np.asarray([r['row_id']for r in rows]),original_global_index=np.asarray([r['original_global_index']for r in rows],np.int64),native_slot=np.asarray([r['native_slot']for r in rows],np.int8),query_xy=np.asarray([[np.nan,np.nan]if r['query_xy']is None else r['query_xy']for r in rows],np.float64),L=np.asarray([np.nan if r['L']is None else r['L']for r in rows],np.float64),original_query_available=np.asarray([r['original_query_available']for r in rows],bool))

def meta_values_exact(actual,expected):
    for key in META:
        a,b=actual[key],expected[key]
        if key=='row_id':need(a.dtype.kind=='U'and a.shape==b.shape and np.array_equal(a,b),'Exact original row_id Unicode values; source storage width not re-read')
        else:need(a.dtype==b.dtype and a.shape==b.shape and(np.array_equal(a,b,equal_nan=True)if a.dtype.kind=='f'else np.array_equal(a,b)),'Original META6 value/dtype '+key)

def parameter_shapes():
    shapes={};incoming=8
    for i,out in enumerate((16,32,32,48,48,64)):
        j=i*3;shapes['encoder.%d.weight'%j]=(out,incoming,3,3);shapes['encoder.%d.bias'%j]=(out,);shapes['encoder.%d.weight'%(j+1)]=(out,);shapes['encoder.%d.bias'%(j+1)]=(out,);incoming=out
    for name,shape in [('shared.1',(64,576)),('support_head',(2,64)),('owner_head',(3,64)),('direct_end_head',(1,64)),('spatial_residual.0',(32,96)),('spatial_residual.2',(1,32))]:shapes[name+'.weight']=shape;shapes[name+'.bias']=(shape[0],)
    return shapes

def checkpoint_envelope(z,protocol,arm,split):
    need(set(z)=={'state_dict','optimizer_state','update','seed','arm','use_sam','spatial_q8','protocol','source_split'},'Owned checkpoint envelope only')
    need(z['update']==4000 and z['seed']==1810 and z['arm']==arm and z['use_sam']is False and type(z['spatial_q8'])is bool and z['spatial_q8']==(arm==ARMS[1])and z['protocol']==protocol and z['source_split']==split,'Fixed final4000 RAW/spatial flags')

def checkpoint(binding,reported,protocol,arm,allowed,split):
    need(allowed.get(binding['path'])==binding,'Exact owned final checkpoint whitelist')
    import torch
    need(torch.__version__=='2.1.1+cu118','Pinned CPU load runtime')
    with opened(binding['path'])as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
        need(h.hexdigest()==binding['sha256'],'Checkpoint byte SHA');f.seek(0);z=torch.load(f,map_location='cpu',weights_only=True)
    checkpoint_envelope(z,protocol,arm,split);shapes=parameter_shapes();sd=z['state_dict']
    need(set(sd)==set(shapes)and len(sd)==36 and sum(t.numel()for t in sd.values())==118359,'Exact118359/36state tensors')
    for k,t in sd.items():need(t.device.type=='cpu'and t.dtype==torch.float32 and tuple(t.shape)==shapes[k]and bool(torch.isfinite(t).all()),'Finite CPU state '+k)
    need(state_digest(sd)==reported,'Final state hash exact');opt=z['optimizer_state'];need(len(opt['param_groups'])==1,'Single AdamW group');group=opt['param_groups'][0]
    need(group['lr']==3e-4 and group['weight_decay']==1e-4 and tuple(group['betas'])==(.9,.999)and group['eps']==1e-8 and group['amsgrad']is False and len(group['params'])==36 and len(set(group['params']))==36,'Fixed AdamW configuration')
    need(set(opt['state'])==set(group['params']),'Every owned parameter optimizer state')
    for k,ident in zip(shapes,group['params']):
        state=opt['state'][ident];need(set(state)=={'step','exp_avg','exp_avg_sq'}and float(state['step'])==4000,'All36 optimizer4000')
        for field in ('exp_avg','exp_avg_sq'):need(tuple(state[field].shape)==shapes[k]and state[field].dtype==torch.float32 and bool(torch.isfinite(state[field]).all()),'Finite optimizer moment')
        if arm==ARMS[0]and k=='spatial_residual.0.weight':need(bool((state['exp_avg'][:,80:]==0).all())and bool((state['exp_avg_sq'][:,80:]==0).all()),'ZeroXY16 control moments exactzero; weights may decay')
    del z
    return dict(PASS=True,parameters=118359,state_tensors=36,update4000=True,use_sam=False,spatial_q8=arm==ARMS[1],final_hash_exact=True)

def nominal_and_context(domains,masks,census):
    need(census['completed']is True and census['PASS']is True and census['binary_header_pixel_weight_model_target_array_or_score_body_reads']==0,'Completed target-free context census')
    eligibility={};result={}
    for domain,cache in domains.items():
        rows=cache['rows'];eligible=[]
        for r in rows:
            if domain=='SYNTH_AUG65536':
                need(r['family']in('SELF_SCENE15','BEFORE_ANCHOR4'),'Original augmentation family')
                if r['family']=='SELF_SCENE15':need(type(r['state_index'])is int and 0<=r['state_index']<15 and(r['state_index']==0)==(r['state_name']=='NOMINAL'),'Exact SELF nominal state')
                else:need(r['state_index']==0 and r['state_name']=='NOMINAL','BEFORE all retained')
                eligible.append(r['state_index']==0)
            else:eligible.append(True)
        eligible=np.asarray(eligible,bool);eligibility[domain]=eligible
        mapping=jl(census['row_mappings'][domain]);need(len(mapping)==len(rows),'All context mapping rows')
        for i,(r,m)in enumerate(zip(rows,mapping)):
            need(m['cache_index']==i and m['row_id']==r['row_id']and m['native_slot']==r['native_slot']and m['original_query_available']==r['original_query_available']and m['fit_nominal_eligible']==bool(eligible[i])and m['immutable_TRAIN_role']==bool(masks[domain][i]),'Context exact original row/native/U/role/fitjoin')
            need(len(m['context_FP32_native_mask_xy_sha256'])==64 and all(q in'0123456789abcdef'for q in m['context_FP32_native_mask_xy_sha256']),'Frozen context hash receipt shape')
        fit=masks[domain]&eligible;result[domain]=fit
    return result,eligibility

def audit_subsets(arm,domain,full,masks,aligned,allowed,eligible):
    mask=masks[domain];seen=[];counts={}
    need(set(arm['source_cohorts'][domain])=={'TRAIN','HOLDOUT'},'Both immutable cohorts')
    for role,take in [('TRAIN',mask),('HOLDOUT',~mask)]:
        indices=np.flatnonzero(take);seen+=indices.tolist();r=arm['source_cohorts'][domain][role];packet=arrays(r['outputs'],OUTKEYS,allowed,True)
        for key in ROWKEYS:need(eq(packet[key],full[key][indices]),'Subset exact saved bytes '+key)
        for key in ORDERS:need(eq(packet[key],full[key]),'Same task orders')
        lineage=jl(r['metadata']);need(len(lineage)==len(indices)==r['rows']and int((~packet['original_query_available']).sum())==r['original_inputU'],'Cohort allrow/U')
        for j,i in enumerate(indices):
            x=lineage[j];origin=aligned[domain][i]
            need(x['cache_index']==int(i)and x['row_id']==str(full['row_id'][i])and x['component_id']==origin['component_id']and x['role']==role and x['original_global_index']==int(full['original_global_index'][i])and x['native_slot']==int(full['native_slot'][i])and x['original_query_available']==bool(full['original_query_available'][i])and x['direct_target_authority']==origin.get('authority')and x['control_join_key']==origin.get('control_join_key'),'Original role cohort lineage')
            need(x['fit_nominal_eligible']==bool(eligible[domain][i])and x['fit_draw_eligible']==bool(mask[i]and eligible[domain][i]),'Excluded TRAIN jitter remains TRAIN but not fiteligible')
        counts[role]=dict(rows=len(indices),inputU=r['original_inputU'])
    need(sorted(seen)==list(range(len(mask))),'Disjoint exhaustive original cohorts, not fit complement');return counts

def exact_metric(target,pred,available,ii):
    counts=Counter((int(target[i]),int(pred[i]))for i in ii);cm=[[counts[y,q]for q in (0,1,-1)]for y in (0,1)];D=sum(cm[0]);E=sum(cm[1]);known=D+E;inputU=sum(not bool(available[i])for i in ii);unknown=sum(int(target[i])<0 for i in ii)
    for i in ii:need(int(target[i])in(-1,0,1)and int(pred[i])in(-1,0,1)and(bool(available[i])or int(target[i])==int(pred[i])==-1),'Known/U semantics')
    div=lambda n,d:n/d if d else None
    return dict(rows=len(ii),known=known,D=D,E=E,target_UNKNOWN=unknown,original_inputU=inputU,available_target_UNKNOWN=unknown-inputU,CM_D_E_by_D_E_U=cm,D_precision=div(cm[0][0],cm[0][0]+cm[1][0]),E_precision=div(cm[1][1],cm[0][1]+cm[1][1]),D_recall=div(cm[0][0],D),E_recall=div(cm[1][1],E),known_prediction_U=cm[0][2]+cm[1][2],known_prediction_U_rate=div(cm[0][2]+cm[1][2],known),UNKNOWN_target_prediction_counts={str(q):counts[-1,q]for q in(-1,0,1)})

def paired_correctness(target,preds,ii):
    counts=Counter(tuple(int(preds[a][i])==int(target[i])for a in ARMS)for i in ii if target[i]>=0)
    return dict(known=sum(counts.values()),both_correct=counts[True,True],both_wrong=counts[False,False],recovered_by_XY=counts[False,True],damaged_by_XY=counts[True,False],net_correct_change=counts[False,True]-counts[True,False])

def independent_triplets(rows,aligned,target,preds):
    grouped=defaultdict(dict)
    for i,row in enumerate(rows):
        key=(row['domain'],row['attempt_id'],int(row['native_slot']));variant=row['variant'];need(variant in('BEFORE','COVER','SHAM')and variant not in grouped[key],'Unique physical native variant');grouped[key][variant]=i
    records=[]
    for key,variants in sorted(grouped.items()):
        present=[v for v in('BEFORE','COVER','SHAM')if v in variants];indices=[variants[v]for v in present];missing=[v for v in('BEFORE','COVER','SHAM')if v not in variants]
        need(len({aligned[i]['role']for i in indices})==1 and len({aligned[i]['component_id']for i in indices})==1,'Physical attempt stable source role')
        truth=[int(target[i])for i in indices];known=any(t>=0 for t in truth)
        if known:need(not missing and truth==[0,1,0],'Only complete known D/E/D')
        else:need(all(t==-1 for t in truth),'Incomplete/complete unknown never imputed')
        record=dict(domain=key[0],attempt_id=key[1],native_slot=key[2],role=aligned[indices[0]]['role'],indices=indices,known=known,targets=truth,present_variants=present,missing_variants=missing,complete=not missing,predictions={a:[int(preds[a][i])for i in indices]for a in ARMS})
        record['all3_correct']={a:known and record['predictions'][a]==[0,1,0]for a in ARMS};records.append(record)
    summary={}
    for role in('FULL','TRAIN','HOLDOUT'):
        selected=[x for x in records if role=='FULL'or x['role']==role];known=[x for x in selected if x['known']];pc=Counter(tuple(x['all3_correct'][a]for a in ARMS)for x in known)
        summary[role]=dict(physical_all_native_attempt_groups=len(selected),physical_present_variant_rows=sum(len(x['indices'])for x in selected),complete_native_triplets=sum(x['complete']for x in selected),partial_UNKNOWN_native_attempt_groups=sum(not x['complete']for x in selected),known_D_E_D_triplets=len(known),UNKNOWN_groups=len(selected)-len(known),missing_variant_counts={v:sum(v in x['missing_variants']for x in selected)for v in('BEFORE','COVER','SHAM')},models={a:dict(all3_correct=sum(x['all3_correct'][a]for x in known),all3_accuracy=sum(x['all3_correct'][a]for x in known)/len(known)if known else None)for a in ARMS},paired=dict(both_correct=pc[True,True],both_wrong=pc[False,False],recovered_by_XY=pc[False,True],damaged_by_XY=pc[True,False],net_correct_change=pc[False,True]-pc[True,False]))
    return summary,records

def source_metrics(cache,aligned,preds):
    target=cache['y']['direct_end_external_target'];av=cache['x']['original_query_available'];rows=cache['rows'];authority=cache['y']['authority'];summary={}
    for role in('FULL','TRAIN','HOLDOUT'):
        ii=[i for i,r in enumerate(aligned)if role=='FULL'or r['role']==role]
        def cell(jj):return dict(models={a:exact_metric(target,preds[a],av,jj)for a in ARMS},paired_correctness=paired_correctness(target,preds,jj))
        summary[role]=dict(all=cell(ii),by_domain={d:cell([i for i in ii if rows[i]['domain']==d])for d in sorted(set(r['domain']for r in rows))},by_native={str(k):cell([i for i in ii if rows[i]['native_slot']==k])for k in range(8)},by_authority={str(a):cell([i for i in ii if str(authority[i])==str(a)])for a in sorted(set(authority.tolist()))})
    triplets,records=independent_triplets(rows,aligned,target,preds);return dict(cohorts=summary,triplets=triplets),records

def fresh_review(path):
    out=safe(path);need(out==H/'actual_review_v1','Fixed fresh independent output')
    out.mkdir(exist_ok=False);return out

def compare_evaluation(complete,own,triplet_records,p):
    need(complete['completed']is True and complete['PASS']is True and complete['code']==p['evaluator_code']and complete['protocol']==p['evaluator_protocol'],'Exact completed saved-only evaluator')
    stored=jd(complete['outputs']['metrics']);need(stored['completed']is True and stored['PASS']is True and stored['cohorts']==own['cohorts'],'All independently recomputed role/domain/native/authority CM/P/R/U/UNKNOWN and row paired cells')
    for role,record in own['triplets'].items():
        actual=stored['triplets'][role]
        for field,value in record.items():need(actual[p['evaluation_triplet_key_map'].get(field,field)]==value,'Independent triplet cohort field '+role+'/'+field)
    ledger=jl(complete['outputs']['triplets']);need(len(ledger)==len(triplet_records),'All known and incompleteUNKNOWN attempt-native groups retained')
    # Exact logical identity/truth/predictions, no physical pixel/GT claim.
    for a,b in zip(ledger,triplet_records):
        for key in ('domain','attempt_id','native_slot','role','indices','known','targets','predictions','all3_correct'):need(a[key]==b[key],'Saved triplet ledger '+key)
        for key in ('present_variants','missing_variants','complete'):need(a[p['evaluation_triplet_ledger_key_map'].get(key,key)]==b[key],'Saved partial group ledger '+key)
    return dict(PASS=True,all_cohort_metric_cells_exact=True,all_triplet_metrics_and_partialUNKNOWN_ledger_exact=True,metrics=complete['outputs']['metrics'],triplets=complete['outputs']['triplets'])

def run(entry_path):
    e=json.loads(raw(entry_path));need(e['approved_actual_independent_spatial_sourcefit_audit']is True and e['both_models_completed']is True and e['source_only']is True and e['automatic_retry']is False,'Separate postterminal root entry required')
    p=json.loads(raw(PROTO));need(e['code']==p['code']==bind(Path(__file__).absolute())and e['protocol']==bind(PROTO),'Exact frozen independent audit')
    jd(e['root_audit_scope']);tests=jd(p['checks']);need(tests['completed']is True and tests['PASS']is True and tests['code']==e['code'],'Completed invented preparation')
    need(sys.executable==p['runtime']['executable']and np.__version__==p['runtime']['numpy'],'Pinned NumPy audit runtime')
    done=jd(e['complete']);need(e['complete']['path']==str(RUN/'COMPLETE.json')and done['completed']is True and done['PASS']is True and done['source_only']is True and done['parameters_each']==118359 and done['steps_each']==4000 and done['total_updates']==8000 and done['code']==p['fit_code']and done['protocol']==p['fit_protocol']and done['spatial_head']==p['spatial_head']and done['context_adapter']==p['context_adapter'],'Exact completed two final4000 fit')
    need(done['all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs']is True and set(done['arms'])==set(ARMS)and done['full_component_roles_preserved']is True and done['nominal_fit_eligibility_applied_to_all_loss_pools']is True,'Both final source barrier and immutable roles')
    fitentry=jd(done['entry']);need(fitentry['code']==done['code']and fitentry['protocol']==done['protocol']and fitentry['split_complete']==done['source_split']==p['split_complete'],'Actual root recipe/split unchanged')
    barrier=jd(done['all2_freeze']);need(barrier['completed']is True and barrier['checkpoints']=={a:done['arms'][a]['checkpoint']for a in ARMS}and barrier['source_TRAIN_or_HOLDOUT_prediction_emitted']is False and barrier['protected_real_COMMON_VAL_access']is False and barrier['checkpoint_or_arm_selection']is False,'All2 closed before saved outputs')
    ev=jd(e['source_evaluation_complete']);need(ev['source_fit_complete']==e['complete'],'Evaluator closes exactly this fit')
    need(ev['code']==p['evaluator_code']and ev['protocol']==p['evaluator_protocol']and ev['completed']is True and ev['PASS']is True,'Completed final evaluator after bothfreeze')
    need(done['training_caches']==p['training_caches']==dict(augmentation=fitentry['augmentation_complete'],public_CF=fitentry['original_public_CF_complete'],direct=fitentry['direct_cache_complete']),'Fixed source cache descriptors')
    aug=jd(done['training_caches']['augmentation']);orig=jd(done['training_caches']['public_CF']);direct=jd(done['training_caches']['direct']);split=jd(done['source_split']);context=jd(done['context_metadata_census'])
    need(done['context_metadata_census']==p['context_census']and context['code']==p['context_adapter']and done['context_input_bindings']==context['input_bindings']==fitentry['context_input_bindings']==p['context_input_bindings'],'Bound context metadata closure')
    descriptors={'SYNTH_AUG65536':aug,'PUBLIC_TRAIN1480':orig['domains']['PUBLIC_TRAIN1480'],'CF3816':orig['domains']['CF3816'],'DIRECT_CONTROL':direct}
    targets={name:d['targets']for name,d in descriptors.items()};need(targets==e['source_target_bindings'],'Only explicit root-bound four small TARGET5/directtarget containers')
    allowed={b['path']:b for b in targets.values()}
    for arm in ARMS:
        a=done['arms'][arm];need(a['updates']==4000 and a['use_sam']is False and type(a['spatial_q8'])is bool and a['spatial_q8']==(arm==ARMS[1])and set(a['outputs'])==set(DOMAINS)and set(a['source_cohorts'])==set(DOMAINS),'Two fixed RAW flags/all original domains')
        need(a['checkpoint']['path']==str(RUN/arm/'FINAL_CHECKPOINT.pt')and a['draws_and_losses']['path']==str(RUN/arm/'ALL4000_DRAWS_AND_LOSSES.jsonl'),'Only declared final checkpoints/logs');allowed[a['checkpoint']['path']]=a['checkpoint']
        for name,binding in a['outputs'].items():
            need(binding['path']==str(RUN/arm/('FINAL_'+name+'_OUTPUTS.npz')),'Full saved output path');allowed[binding['path']]=binding
            for role in('TRAIN','HOLDOUT'):
                subset=a['source_cohorts'][name][role];need(subset['outputs']['path']==str(RUN/arm/('FINAL_'+name+'_'+role+'_OUTPUTS.npz'))and subset['metadata']['path']==str(RUN/arm/(name+'_'+role+'_ROW_COMPONENTS.jsonl')),'Only fixed source cohort outputs');allowed[subset['outputs']['path']]=subset['outputs']
    need(len(allowed)==30 and not any(b['path']in allowed for d in descriptors.values()for k,b in d.items()if k=='inputs'),'Whitelist4targets+2ownedcheckpoints+24predictions; input NPZ excluded')
    out=fresh_review(p['output_directory']);start=time.monotonic();domains={};results={};preds={}
    try:
        for name,n in DOMAINS.items():
            d=descriptors[name];rows=jl(d['metadata']);x=metadata_x(rows);metadata_exact(rows,x,name);keys=META+('direct_end_external_target','authority')if name=='DIRECT_CONTROL'else TARGET5;y=arrays(d['targets'],keys,allowed,True)
            if name=='DIRECT_CONTROL':
                meta_values_exact(y,x);x={k:y[k]for k in META}
                for i,r in enumerate(rows):need(str(y['authority'][i])==r['authority']and(y['direct_end_external_target'][i]<0 or r['root_admitted']is True),'Original admitted direct weak authority only')
            else:
                need(y['support_target'].dtype==np.int8 and y['support_target'].shape==(n,2)and np.isin(y['support_target'],[-1,0,1]).all()and(y['support_target'][~x['original_query_available']]==-1).all(),'Original support targets/U')
                need(y['owner_soft'].dtype==np.float64 and y['owner_soft'].shape==(n,3),'Original raw softmass shape/type')
            domains[name]=dict(x=x,y=y,rows=rows)
        need([int((~domains[k]['x']['original_query_available']).sum())for k in DOMAINS]==[1264,194,360,112],'Original source inputU unchanged')
        masks,aligned=split_masks(split,domains);fitmasks,eligible=nominal_and_context(domains,masks,context);pool,dpool=filtered_pools(domains,fitmasks)
        need(sum(int(m.sum())for m in masks.values())==64160 and sum(int(m.sum())for m in fitmasks.values())==14880 and sum(int((fitmasks[d]&~domains[d]['x']['original_query_available']).sum())for d in domains)==697,'Exact original role64160/fit14880/U697')
        validation=jd(e['fit_input_validation']);need(e['fit_input_validation']['path']==str(RUN/'FIT_INPUT_VALIDATION.json')and validation['completed']is True and validation['PASS']is True and validation['full_component_roles_preserved']is True,'Root-bound input validation')
        need(validation['fit_nominal_eligibility_counts']=={d:dict(full_rows=len(masks[d]),component_TRAIN=int(masks[d].sum()),nominal_eligible=int(eligible[d].sum()),fit_rows=int(fitmasks[d].sum()))for d in domains},'Every declared role/fit count independently exact')
        need(done['cohort_denominators']=={d:cohort_counts(cache,masks[d])for d,cache in domains.items()},'Every original role/domain/native/U/authority denominator')
        dc=domains['DIRECT_CONTROL'];target=dc['y']['direct_end_external_target'];dm=masks['DIRECT_CONTROL'];need([int((target==k).sum())for k in(0,1)]==[248,124]and[len(x)for x in dpool]==[218,109]and[int(((target==k)&~dm).sum())for k in(0,1)]==[30,15],'Full372/TRAIN327/H45 direct labels')
        need(all(r['domain']=='SYNTH_FINITE_END'for i,r in enumerate(dc['rows'])if not dm[i]and target[i]>=0),'HOLD45 synthetic known only, realend0')
        prior=independent_native_priors(dc,dm);need(canonical_census(jd(done['native_class_prior_census']))==canonical_census(prior),'Independent TRAIN raw/balanced native prior arithmetic')
        initial=done['same_initial_state_sha'];need(len(initial)==64 and barrier['same_initial_state_sha']==initial and all(a['initial_state_sha']==initial for a in done['arms'].values()),'Paired sealed initial-state receipt; no initialization rerun')
        for arm in ARMS:
            a=done['arms'][arm];logs=audit_log(a['draws_and_losses'],pool,dpool,a['old_draw_sha'],a['direct_draw_sha'],fitmasks);ck=checkpoint(a['checkpoint'],a['final_state_sha'],done['protocol'],arm,allowed,done['source_split']);outputs={}
            for name in DOMAINS:
                full=arrays(a['outputs'][name],OUTKEYS,allowed,True);dp=validate_output(full,domains[name]['x']);cohorts=audit_subsets(a,name,full,masks,aligned,allowed,eligible);outputs[name]=dict(rows=DOMAINS[name],inputU=int((~domains[name]['x']['original_query_available']).sum()),cohorts=cohorts)
                if name=='DIRECT_CONTROL':preds[arm]=dp
            results[arm]=dict(logs=logs,checkpoint=ck,outputs=outputs)
        need(all(a['old_draw_sha']==done['same_filtered_old192_draw_sha']==barrier['same_filtered_old192_draw_sha']and a['direct_draw_sha']==done['same_direct32_draw_sha']==barrier['same_direct32_draw_sha']for a in done['arms'].values()),'Both full4000 paired old192/direct32 hashes')
        own,triplet_records=source_metrics(dc,aligned['DIRECT_CONTROL'],preds);comparison=compare_evaluation(ev,own,triplet_records,p)
        need([own['triplets'][r]['known_D_E_D_triplets']for r in('FULL','TRAIN','HOLDOUT')]==[124,109,15],'Known complete DED denominators only')
        bt=file_time(done['all2_freeze'])
        for a in done['arms'].values():
            need(file_time(a['checkpoint'])<=bt,'Final checkpoint mtime no later than barrier')
            for binding in list(a['outputs'].values())+[z['outputs']for records in a['source_cohorts'].values()for z in records.values()]:need(file_time(binding)>=bt,'Source output mtime no earlier than barrier')
        write(out/'SOURCE_GROUPED_DIRECT_READOUT.json',dict(completed=True,PASS=True,source_only=True,**own,physical_native_attempt_groups=len(triplet_records),incomplete_UNKNOWN_groups=sum(not r['complete']for r in triplet_records),no_imputation=True,source_HOLDOUT_historically_exposed_not_blind=True,source_known_HOLD45_synthetic_only=True,nativephysicalgold=False,adopted=False,goal_solved=False))
        report=dict(completed=True,PASS=True,code=bind(Path(__file__).absolute()),protocol=bind(PROTO),entry=bind(entry_path),fit_complete=e['complete'],source_evaluation_complete=e['source_evaluation_complete'],results=results,all8000_log_rows_seed1810_1811_replayed=True,fitmask_is_originalTRAIN_intersect_nominal=True,excluded_TRAIN_not_relabelled_HOLD=True,all147952_FULL_output_rows_META_decoders_exact=True,all24_FULL_TRAIN_HOLDOUT_packets_exact_original_roles=True,owned_CPU_checkpoint_loads=2,model_instantiation_forward_fit_CUDA_calls=0,input_RGB_SAM_feature_NPZ_header_or_bytes_reads=0,source_inputU=[1264,194,360,112],physical_source_rows_each=73976,full_rows_twoarms=147952,cohort_copies_not_new_observations=True,direct372_TRAIN327_HOLD45=True,metric_comparison=comparison,readout=bind(out/'SOURCE_GROUPED_DIRECT_READOUT.json'),seconds=time.monotonic()-start,automatic_retry=False,limits=p['limits'])
        write(out/'REVIEW.json',report);print(json.dumps(dict(completed=True,PASS=True,review=bind(out/'REVIEW.json'),seconds=report['seconds'])),flush=True)
    except BaseException:
        write(out/'FAILURE.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',type=Path,required=True);run(ap.parse_args().entry)
