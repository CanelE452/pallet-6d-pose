#!/usr/bin/env python3
"""Independent V251 SOURCE saved-fit QA; no producer imports or neural replay."""
import argparse,ast,csv,datetime,hashlib,io,json,math,os,stat,traceback
from pathlib import Path
import numpy as np
import torch
MAX_DIFF=0.;NUMERIC_COMPARISONS=0
VIEWS=("full_CLS","fixed_centre_Q","centre_CLS")
ARMS=("CONTROL_NO_REFERENCE","TREAT_REFERENCE")
CORE_SHA="5988479c4ccfa82a50b91ddc82d235c0009cc78280845830d0ddc4868d53cbf4"
DRAW_SHA="20a1fc74ea15a04a16fe29f8e5b9a114b4629677710ed43c1628b2a1ac00b734"

def path(p):
    p=Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any('v112' in s.lower() or 'cache' in s.lower() or s.lower().endswith('.pyc') for s in p.parts)
    for a in [*reversed(p.parents),p]:
        try: st=os.lstat(a)
        except FileNotFoundError: continue
        assert not stat.S_ISLNK(st.st_mode),str(a)
    return p

def raw(p):
    fd=os.open(path(p),os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f: return f.read()

def binding(p):
    b=raw(p)
    return {'path':str(path(p)),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}

def checked(b):
    assert set(b)=={'path','sha256','size_bytes'} and binding(b['path'])==b,b
    return path(b['path'])

def load_json(b): return json.loads(raw(checked(b)))

def csv_rows(p): return list(csv.DictReader(io.StringIO(raw(p).decode('utf-8'),newline='')))

def same(a,b,where='root'):
    global MAX_DIFF,NUMERIC_COMPARISONS
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),(where,'keys',set(a)^set(b))
        for k in a: same(a[k],b[k],where+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert isinstance(b,(list,tuple)) and len(a)==len(b),(where,'length')
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,where+'/'+str(i))
    elif a is None or isinstance(a,(str,bool)):
        assert type(a)==type(b) and a==b,(where,a,b)
    elif isinstance(a,(int,np.integer)):
        assert not isinstance(b,bool) and int(a)==b,(where,a,b)
    else:
        assert isinstance(b,(int,float)) and not isinstance(b,bool),(where,a,b)
        d=abs(float(a)-float(b));MAX_DIFF=max(MAX_DIFF,d);NUMERIC_COMPARISONS+=1
        assert math.isfinite(float(a)) and math.isfinite(float(b)) and d<=1e-12,(where,a,b,d)

def normalized(v):
    x=np.asarray(v,dtype=np.float64)
    if not np.isfinite(x).all():return None
    n=math.sqrt(float(x@x))
    return x/n if math.isfinite(n) and n>0 else None

def selector(nodes,orig,node_support,active,prototypes):
    norm=[normalized(v) for v in nodes];bank=[normalized(v) for v in prototypes]
    mask=[bool(orig[i] and node_support[i] and norm[i] is not None and i!=active) for i in range(8)]
    slots=np.flatnonzero(mask)
    r={'bank_index':None,'bank_scores':[None]*4,'top2_slots_by_bank':[[] for _ in range(4)],'valid_other_nodes':len(slots),'candidate_mask':mask,'selector_gap':None,'status':'UNKNOWN'}
    if any(v is None for v in bank):r['reason']='INVALID_FIXED_BANK';return r
    if len(slots)<2:r['reason']='FEWER_THAN_TWO_OTHER_NODES';return r
    scores=[];tops=[]
    for p in bank:
        cos=np.asarray([float(norm[int(s)]@p) for s in slots]);order=np.lexsort((slots,-cos))
        scores.append(float((cos[order[0]]+cos[order[1]])/2));tops.append([int(slots[order[0]]),int(slots[order[1]])])
    order=np.lexsort((np.arange(4),-np.asarray(scores)))
    r.update(bank_index=int(order[0]),bank_scores=scores,top2_slots_by_bank=tops,selector_gap=float(scores[order[0]]-scores[order[1]]),status='SUPPORTED',reason=None)
    return r

def state_digest(state):
    h=hashlib.sha256()
    for name in sorted(state):
        a=state[name].detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()

def positive_auc(rows):
    ds=[(r['target'],r['selected_family_E_logit']) for r in rows if r['available']]
    d=sum(y==0 for y,s in ds);e=sum(y==1 for y,s in ds)
    if not d or not e:return {'auc':None,'D_supported':d,'E_supported':e,'reason':'BOTH_CLASSES_REQUIRED'}
    ds.sort(key=lambda z:z[1]);rank=0.;i=0
    while i<len(ds):
        j=i+1
        while j<len(ds) and ds[j][1]==ds[i][1]:j+=1
        rank+=((i+1+j)/2)*sum(y==1 for y,s in ds[i:j]);i=j
    return {'auc':(rank-e*(e+1)/2)/(e*d),'D_supported':d,'E_supported':e,'E_direction':'POSITIVE_RAW_SELECTED_FAMILY_LOGIT','ties':'half'}

def reduce_source(rows):
    result={}
    for fam in (0,1):
        for role in ('FIT','DEV'):
            sub=[r for r in rows if r['family']==fam and r['role']==role]
            cm=[[sum(r['target']==y and r['prediction']==p for r in sub) for p in ('D','E','U')] for y in (0,1)]
            m={'rows':len(sub),'known_DE':len(sub),'CM_D_E_by_D_E_U':cm,'target_UNKNOWN':0,'original_input_U':sum(not r['available'] for r in sub)}
            for name,k in (('D',0),('E',1)):
                truth=sum(cm[k]);pred=sum(cm[y][k] for y in (0,1));tp=cm[k][k]
                m[name]={'truth':truth,'predicted':pred,'precision':tp/pred if pred else None,'recall':tp/truth if truth else None}
            a,b=m['D']['recall'],m['E']['recall']
            m.update(family=fam,role=role,CM_D_E=[row[:2] for row in cm],precisionE=m['E']['precision'],recallD=a,recallE=b,BR=None if a is None or b is None else (a+b)/2,AUC=positive_auc(sub),U=sum(r['prediction']=='U' for r in sub))
            result[f'family{fam}_{role}']=m
    return result

def mean_br(m):
    rr=[m[f'family{f}_DEV'][s]['recall'] for f in (0,1) for s in ('D','E')]
    return None if any(x is None for x in rr) else sum(rr)/4

def calculate_gate(base,control,treat):
    old=[base[f'family{f}_DEV'][s]['recall'] for f in (0,1) for s in ('D','E')]
    new=[treat[f'family{f}_DEV'][s]['recall'] for f in (0,1) for s in ('D','E')]
    if any(x is None for x in old+new):legacy={'PASS':False,'reason':'Missing fixed SOURCE class denominator','current_admission':False}
    else:
        gains=[(sum(new[2*f:2*f+2])-sum(old[2*f:2*f+2]))/2 for f in (0,1)]
        gain=100*sum(gains)/2;drop=100*max(x-y for x,y in zip(old,new));yd=new[3];sd,se=new[:2]
        legacy={'PASS':gain>=3 and drop<=2 and yd>=.5 and min(sd,se)>=.9,'mean_family_DEV_balanced_recall_gain_pp':gain,'max_family_class_recall_drop_pp':drop,'YCB_DEV_E_recall':yd,'SYN_DEV_D_recall':sd,'SYN_DEV_E_recall':se,'fixed_min_gain_pp':3,'fixed_max_class_drop_pp':2,'fixed_YCB_DEV_E_floor':.5,'fixed_SYN_DEV_D_E_floor':.9,'current_admission':False,'automatic_retry':False,'new_current_inference':0}
    mc,mt=mean_br(control),mean_br(treat);gain=None if mc is None or mt is None else (mt-mc)*100
    return {'PASS':bool(legacy['PASS'] and gain is not None and gain>=1),'historical_SOURCE_TREAT_vs_BASE_FROZEN_V239':legacy,'matched_TREAT_minus_CONTROL_mean_family_BR_gain_pp':gain,'matched_min_gain_pp':1.,'matched_PASS':gain is not None and gain>=1,'public_pixel80_gate_applied':False,'current_admission':False,'automatic_current_execution':False}

def paired(a,b):
    out=[]
    for fam in (0,1):
        for target in (0,1):
            ids=[i for i,r in enumerate(a) if r['family']==fam and r['role']=='DEV' and r['target']==target]
            truth='D' if target==0 else 'E'
            out.append({'family':fam,'teacher_state':truth,'rows':len(ids),'recovered':sum(a[i]['prediction']!=truth and b[i]['prediction']==truth for i in ids),'damaged':sum(a[i]['prediction']==truth and b[i]['prediction']!=truth for i in ids),'unchanged':sum(a[i]['prediction']==b[i]['prediction'] for i in ids)})
    return out

def main(args):
    cb={'path':str(path(args.complete)),'sha256':'33a9b72d368a231e83d6515fab14760b122ff35ca0fd04f3c3d7bbefe917cd92','size_bytes':13317}
    c=load_json(cb);assert c['completed'] is True and c['PASS'] is True
    e=load_json(c['entry']);scope=load_json(c['root_scope'])
    assert scope['approved'] is True and scope['entry_contract']=={k:v for k,v in e.items() if k!='root_scope'}
    for k in ('code','root_protocol','implementation_protocol','utility_code','head_module','selector_code','feature_complete','frozen_core_complete'):checked(c[k])
    inputs={k:checked(v) for k,v in c['features'].items()}
    for k in ('samples_npz','geometry_npz','YCB_prototypes'):checked(c[k])
    outputs=['summary','freeze','reference_banks','SYN_reference_support','reference_inputs','reference_selection','draw_ledger','BASE_FROZEN_V239_CSV']
    for k in outputs:checked(c[k])
    for k in ('checkpoints','logs','outputs'):
        for b in c[k].values():checked(b)
    freeze=load_json(c['freeze']);assert freeze['completed'] is True and freeze['PASS'] is True
    assert freeze['checkpoints']==c['checkpoints'] and freeze['logs']==c['logs'] and freeze['bank']==c['reference_banks'] and freeze['draw_ledger']==c['draw_ledger']
    assert freeze['new_DEV_core_reference_or_head_outputs_before_freeze']==0 and freeze['new_updates_each']==1000 and freeze['core_updates']==0
    initqa=load_json(freeze['initial_QA']);globalqa=load_json(freeze['global_QA'])
    assert initqa['completed'] is True and initqa['PASS'] is True and globalqa['completed'] is True and globalqa['PASS'] is True
    assert all(initqa['initial_total_logits_exact_frozen_core'].values()) and initqa['residual_output_weight_bias_zero'] is True
    assert c['frozen_core_state_sha256_before']==c['frozen_core_state_sha256_after']==freeze['core_state_sha256']==CORE_SHA
    x=np.load(inputs['ORIGINAL_ALL_FEATURES.npy'],mmap_mode='r',allow_pickle=False)
    nodes=np.load(inputs['NODE_CLS.npy'],mmap_mode='r',allow_pickle=False)
    group=np.load(inputs['SAMPLE_GROUP_INDEX.npy'],allow_pickle=False);slot=np.load(inputs['SAMPLE_NODE_INDEX.npy'],allow_pickle=False)
    assert x.dtype==np.float32 and x.shape==(7617,1186) and nodes.dtype==np.float32 and nodes.shape==(1767,8,384)
    with np.load(path(c['samples_npz']['path']),allow_pickle=False) as z:data={k:z[k] for k in ('condition','fractions','target','family','fit','available','row_id')}
    with np.load(path(c['geometry_npz']['path']),allow_pickle=False) as z:oa=z['original_available8'];na=z['node_available']
    with np.load(path(c['YCB_prototypes']['path']),allow_pickle=False) as z:yp=z['prototypes'];ys=z['supported'];views=z['view']
    assert tuple(views)==VIEWS and yp.shape==(3,4,384) and yp.dtype==np.float64
    assert data['target'].shape==(7617,) and np.isin(data['target'],[0,1]).all() and int(data['fit'].sum())==5876
    assert not set(group[data['fit']].tolist())&set(group[~data['fit']].tolist())
    assert np.array_equal(oa[group,slot]&na[group,slot],data['available'])
    assert x[:,-32:].tobytes()==data['condition'].tobytes() and x[:,1152:1154].tobytes()==data['fractions'].tobytes()
    assert x[:,:384].tobytes()==nodes[group,slot].tobytes()
    syn=data['family']==0;assert np.array_equal(data['condition'][syn,:8].argmax(1),slot[syn])
    # New SYN supervised FIT-D statistics; sequential ascending index sum, independent loop.
    proto=np.zeros((3,8,384),np.float64);support=np.zeros((3,8),bool);stats=[]
    for address in range(8):
        ids=np.flatnonzero(data['fit']&data['available']&syn&(data['target']==0)&(slot==address))
        for v,name in enumerate(VIEWS):
            accepted=[];acc=np.zeros(384,np.float64)
            for i in ids:
                vec=np.asarray(x[i,v*384:(v+1)*384],np.float64)
                if normalized(vec) is not None:acc+=vec;accepted.append(int(i))
            n=None if not accepted else normalized(acc/len(accepted))
            if n is not None:proto[v,address]=n;support[v,address]=True
            stats.append({'family':0,'view':name,'original_native_address':address,'FIT_D_available_rows':len(ids),'finite_nonzero_reference_rows':len(accepted),'sample_indices':accepted,'supported':bool(support[v,address]),'physical_canonical_anatomy':'NOT_CERTIFIED','pallet_type_or_material_filter':False})
    with np.load(path(c['reference_banks']['path']),allow_pickle=False) as z:
        same(proto.tolist(),z['SYN_prototypes'].tolist(),'SYN24banks');assert np.array_equal(support,z['SYN_supported'])
        assert yp.tobytes()==z['YCB_prototypes'].tobytes() and np.array_equal(ys,z['YCB_supported'])
        bank_byte_exact=proto.tobytes()==z['SYN_prototypes'].tobytes()
    same(stats,load_json(c['SYN_reference_support'])['rows'],'SYN24support')
    refs=np.zeros((7617,9),np.float32);records=[]
    for i in range(7617):
        fam=int(data['family'][i]);g=int(group[i]);a=int(slot[i]);detail=None
        if fam==0:b=a;other=(a+1)%8;p=proto;s=support
        else:
            detail=selector(nodes[g],oa[g],na[g],a,yp[0]);b=detail['bank_index'] if ys[0].all() else None
            other=None if b is None else (b+1)%4;p=yp;s=ys
        oks=[]
        for v in range(3):
            q=normalized(x[i,v*384:(v+1)*384]);p1=None if b is None else normalized(p[v,b]);p2=None if other is None else normalized(p[v,other])
            ok=bool(data['available'][i]) and b is not None and q is not None and p1 is not None and p2 is not None and bool(s[v,b]) and bool(s[v,other])
            if ok:refs[i,v]=float(q@p1);refs[i,3+v]=float(q@p2);refs[i,6+v]=1.
            oks.append(bool(ok))
        records.append({'sample_index':i,'family':fam,'original_active_slot':a,'selected_bank_index':b,'next_bank_index':other,'per_view_supported':oks,'choice':'ORIGINAL_NATIVE_ADDRESS' if fam==0 else 'V250_INPUT_ONLY_OTHER_NODES','selector_UNKNOWN':b is None,'YCB_selector_detail':detail,'SYN_next_is_not_wrong_object_or_negative_gold':fam==0})
    saved_ref=np.load(path(c['reference_inputs']['path']),allow_pickle=False)
    assert saved_ref.shape==refs.shape and saved_ref.dtype==np.float32
    ref_max=float(np.max(np.abs(refs-saved_ref)));assert refs.tobytes()==saved_ref.tobytes(),('reference9 byte inequality',ref_max)
    saved_records=[json.loads(line) for line in raw(c['reference_selection']['path']).splitlines()]
    same(records,saved_records,'reference_selection7617')
    ref_support={}
    for fam in (0,1):
        for role in ('FIT','DEV'):
            rr=[r for r in records if r['family']==fam and bool(data['fit'][r['sample_index']])==(role=='FIT')];n=len(rr)
            all3=sum(all(r['per_view_supported']) for r in rr);unknown=sum(r['selector_UNKNOWN'] for r in rr)
            ref_support[f'family{fam}_{role}']={'family':fam,'role':role,'original_rows':n,'all3_supported':all3,'all3_supported_fraction':all3/n if n else None,'selector_UNKNOWN':unknown,'selector_UNKNOWN_fraction':unknown/n if n else None,'per_view_supported':{name:sum(r['per_view_supported'][v] for r in rr) for v,name in enumerate(VIEWS)},'reference_UNKNOWN_does_not_change_model_availability':True}
    pools=[np.flatnonzero(data['fit']&data['available']&(data['family']==fam)&(data['target']==y)) for fam in (0,1) for y in (0,1)]
    rng=np.random.Generator(np.random.PCG64(17));draw=np.stack([np.concatenate([rng.choice(p,8,replace=True) for p in pools]) for _ in range(1000)]).astype(np.int64)
    assert hashlib.sha256(draw.astype('<i8').tobytes()).hexdigest()==DRAW_SHA==c['draws_sha256']
    ledger=[json.loads(line) for line in raw(c['draw_ledger']['path']).splitlines()];assert len(ledger)==1000
    logs={}
    for arm in ARMS:
        logs[arm]=[json.loads(line) for line in raw(c['logs'][arm]['path']).splitlines()];assert len(logs[arm])==1000
        for t in range(1000):
            l=logs[arm][t];d=ledger[t]
            assert l['update']==d['update']==t+1 and l['sample_indices']==d['sample_indices']==draw[t].tolist()
            assert l['draws_sha256']==d['draws_sha256']==DRAW_SHA and l['all_fit_available_known'] is True and d['all_fit_available_known'] is True
            assert l['core_optimizer_updates']==0 and l['reference_input']==('ZERO' if arm==ARMS[0] else 'INPUT_SELECTED')
            assert all(math.isfinite(l[k]) for k in ('family0_BCE','family1_BCE','total_loss'))
            assert abs(l['total_loss']-(l['family0_BCE']+l['family1_BCE'])/2)<=1e-6
    # Only two new residual checkpoints decoded; no frozen-core inference or core CP decode.
    cp_checks={};initials=[]
    shape={'net.0.weight':(32,73),'net.0.bias':(32,),'net.2.weight':(2,32),'net.2.bias':(2,)}
    for arm in ARMS:
        cp=torch.load(path(c['checkpoints'][arm]['path']),weights_only=True,map_location='cpu')
        assert cp['arm']==arm and cp['updates']==1000 and cp['core_optimizer_updates']==0 and cp['trainable_parameters']==2434
        assert cp['frozen_core_state_sha256']==CORE_SHA and cp['frozen_core_checkpoint_sha256']==c['frozen_core_checkpoint']['sha256']
        assert cp['draws_sha256']==DRAW_SHA and cp['reference_banks']==c['reference_banks'] and cp['threshold_raw_logit']==0
        for key in ('residual_state_dict','initial_residual_state_dict'):
            state=cp[key];assert set(state)==set(shape)
            for name,v in state.items():assert v.dtype==torch.float32 and tuple(v.shape)==shape[name] and bool(torch.isfinite(v).all())
            assert sum(v.numel() for v in state.values())==2434
        ini=cp['initial_residual_state_dict'];fin=cp['residual_state_dict'];initials.append(ini)
        assert int(torch.count_nonzero(ini['net.2.weight']))==int(torch.count_nonzero(ini['net.2.bias']))==0
        assert state_digest(ini)==cp['initial_residual_sha256']==c['initial_residual_sha256'][arm]==initqa['same_initial_residual_sha256']
        assert state_digest(fin)==c['final_residual_sha256'][arm]
        opt=cp['optimizer_state_dict'];assert len(opt['state'])==4 and len(opt['param_groups'])==1
        pg=opt['param_groups'][0];assert pg['lr']==.001 and pg['weight_decay']==.0001 and len(pg['params'])==4
        for st in opt['state'].values():
            assert float(st['step'])==1000
            assert all(bool(torch.isfinite(v).all()) for v in st.values() if isinstance(v,torch.Tensor))
        cp_checks[arm]={'new_residual_parameters':2434,'optimizer_tensor_states':4,'each_step':1000,'finite':True,'initial_sha256':state_digest(ini),'final_sha256':state_digest(fin),'core_state_hash_matches_metadata':True}
    assert all(torch.equal(initials[0][k],initials[1][k]) for k in shape)
    all_rows={};metrics={}
    for arm,b in [('BASE_FROZEN_V239',c['BASE_FROZEN_V239_CSV']),*c['outputs'].items()]:
        rr=csv_rows(b['path']);assert len(rr)==7617;typed=[]
        for i,s in enumerate(rr):
            assert int(s['sample_index'])==i and s['row_id']==str(data['row_id'][i]) and int(s['target'])==data['target'][i] and int(s['family'])==data['family'][i]
            assert s['role']==('FIT' if data['fit'][i] else 'DEV') and s['available'] in ('True','False') and (s['available']=='True')==bool(data['available'][i])
            z=[float(s['logit0']),float(s['logit1'])];selected=float(s['selected_family_E_logit']);assert all(math.isfinite(v) for v in z+[selected])
            assert selected==z[int(data['family'][i])]
            pred='U' if not data['available'][i] else ('E' if selected>=0 else 'D');assert pred==s['prediction']
            typed.append({'sample_index':i,'row_id':s['row_id'],'family':int(s['family']),'role':s['role'],'target':int(s['target']),'available':s['available']=='True','logit0':z[0],'logit1':z[1],'selected_family_E_logit':selected,'prediction':pred})
        all_rows[arm]=typed;metrics[arm]=reduce_source(typed)
    gate=calculate_gate(metrics['BASE_FROZEN_V239'],metrics[ARMS[0]],metrics[ARMS[1]])
    summary={'completed':True,'PASS':True,'BASE_FROZEN_V239':metrics['BASE_FROZEN_V239'],'arms':{a:metrics[a] for a in ARMS},'SOURCE_gate':gate,'reference_support':ref_support,'SYN_MAIN_vs_YCB_associated_r_AUX_separate':True,'all3views_joint_input_no_best_view':True,'current_admission':False,'exposed_SOURCE_DEV_not_blind':True}
    same(summary,load_json(c['summary']),'full_SOURCE_RESULTS')
    for k in ('core_optimizer_updates','new_DEV_core_reference_or_head_outputs_before_bothfinalfreeze','new_raw_RGB_depth_annotation_official_encoder_weights_or_current_held_reads','new_encoder_forward'):assert c[k]==0
    assert c['current_admission'] is False and c['original_rows']==15040 and c['original_masked_S_U']==7423
    for k in outputs:checked(c[k])
    for key in ('checkpoints','logs'):
        for b in c[key].values():checked(b)
    out=path(args.out);assert not out.exists();out.mkdir()
    report={'schema':'V251_INDEPENDENT_ACTUAL_REFERENCE_FIT_QA_C_V1','completed':True,'PASS':True,'fit_complete':cb,'entry':c['entry'],'root_scope':c['root_scope'],'root_protocol':c['root_protocol'],'producer_code':c['code'],'checker':binding(__file__),'freeze':c['freeze'],'checkpoint_QA':cp_checks,'bank_reference_QA':{'SYN_bank_vectors':24,'new_SYN_bank_byte_exact':bank_byte_exact,'new_SYN_bank_FIT_D_only':True,'new_SYN_bank_support_rows':len(stats),'YCB_bank_bytes_unchanged':True,'reference9_rows':7617,'reference9_FP32_byte_exact':True,'reference9_maxabs':ref_max,'all_selection_records_and_structural_policy_verified':7617,'original_source_qL_native_U_features_unchanged':True,'source_FIT_DEV_group_overlap':0},'draw_log_QA':{'draws_shape':[1000,32],'draw_sha256':DRAW_SHA,'ledger_rows':1000,'logs_each':1000,'both_logs_ids_exact_common_PCG17_ledger':True,'only_FIT_available_known_draws':True,'loss_records_finite_and_family_mean':True},'metrics':metrics,'gates':gate,'reference_support':ref_support,'paired_DEV':{'TREAT_vs_BASE':paired(all_rows['BASE_FROZEN_V239'],all_rows[ARMS[1]]),'TREAT_vs_CONTROL':paired(all_rows[ARMS[0]],all_rows[ARMS[1]]),'CONTROL_vs_BASE':paired(all_rows['BASE_FROZEN_V239'],all_rows[ARMS[0]])},'maximum_numeric_comparison_difference':MAX_DIFF,'numeric_values_compared':NUMERIC_COMPARISONS,'independence':'Independent sequential FP64 SYN bank, lexsort structural selector, reference9 and average-rank Mann-Whitney AUROC/CM/gates. No producer import/call and no training/optimizer updates. Only new residual CP2 tensors decoded; full core/3-arm neural output replay delegated to A.','temporal_and_core_limits':'Exact freeze bindings/static code order and before/after/core hashes reconciled. This saved audit does not independently prove historical read timing or re-execute frozen-core state; A full CPU inference replay is separate.','access':{'SOURCE_frozen_feature_arrays_and_nonpixel_samples_geometry':True,'SOURCE_DEV_nonpixel_metadata_and_saved_outputs_read':True,'new_residual_CP2_weights_only_CPU':True,'frozen_core_CP_decoded':False,'RGB_raw_GT_archives_official_encoderweights_current_protected_held':0,'producer_operational_imports_calls_or_training':0,'new_core_or_head_forward':0,'Git_network':0},'limitations':['SYN original intended-native-address witness and YCB associated-r AUX/nativeUNKNOWN remain distinct.','No material/class/CAD/physical-native gold claim; fixed-next SYN address is not a negative gold label.','Repeated SOURCE DEV and existing source lineage limitations are not independent deployment success.','SOURCE gate and matched effect are arithmetic only; current_admissionfalse/model choiceNone.'],'current_admission':False,'selected_model':None}
    dest=out/'REVIEW.json';dest.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completed':True,'PASS':True,'REVIEW':binding(dest),'gates':gate,'reference9_byte_exact':True,'SYN_bank_byte_exact':bank_byte_exact},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--complete',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a)

