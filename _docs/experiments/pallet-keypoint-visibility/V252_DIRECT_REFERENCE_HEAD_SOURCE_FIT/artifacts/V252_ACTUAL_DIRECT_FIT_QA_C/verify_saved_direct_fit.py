#!/usr/bin/env python3
"""Independent V252 SOURCE direct-head saved QA; no producer import or inference."""
import argparse,ast,csv,datetime,hashlib,io,json,math,os,stat
from pathlib import Path
import numpy as np
import torch
MAX_DIFF=0.;NUMERIC_COMPARISONS=0
VIEWS=("full_CLS","fixed_centre_Q","centre_CLS")
ARMS=("CONTROL_GENERIC_REFERENCE","TREAT_TARGET_REFERENCE")
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
            m.update(family=fam,role=role,CM_D_E=[row[:2] for row in cm],precisionD=m['D']['precision'],precisionE=m['E']['precision'],recallD=a,recallE=b,BR=None if a is None or b is None else (a+b)/2,AUC=positive_auc(sub),U=sum(r['prediction']=='U' for r in sub),U_rate=sum(r['prediction']=='U' for r in sub)/len(sub) if sub else None)
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
    return {'PASS':bool(legacy['PASS'] and gain is not None and gain>=1),'historical_SOURCE_TREAT_vs_BASE_FROZEN_V239':legacy,'matched_TREAT_minus_CONTROL_mean_family_BR_gain_pp':gain,'matched_min_gain_pp':1.,'matched_PASS':gain is not None and gain>=1,'Pallox80_gate_applied':False,'current_admission':False,'automatic_current_execution':False}

def paired(a,b):
    out=[]
    for fam in (0,1):
        for target in (0,1):
            ids=[i for i,r in enumerate(a) if r['family']==fam and r['role']=='DEV' and r['target']==target]
            truth='D' if target==0 else 'E'
            out.append({'family':fam,'teacher_state':truth,'rows':len(ids),'recovered':sum(a[i]['prediction']!=truth and b[i]['prediction']==truth for i in ids),'damaged':sum(a[i]['prediction']==truth and b[i]['prediction']!=truth for i in ids),'unchanged':sum(a[i]['prediction']==b[i]['prediction'] for i in ids)})
    return out

def paired_all(a,b):
    out={}
    for family in (0,1):
        for role in ('FIT','DEV'):
            ix=[i for i,r in enumerate(a) if r['family']==family and r['role']==role]
            delta=np.array([b[i]['selected_family_E_logit']-a[i]['selected_family_E_logit'] for i in ix],np.float64)
            truth=lambda i:'D' if a[i]['target']==0 else 'E'
            out[f'family{family}_{role}']={'rows':len(ix),'prediction_changes':sum(a[i]['prediction']!=b[i]['prediction'] for i in ix),'recoveries':sum(a[i]['prediction']!=truth(i) and b[i]['prediction']==truth(i) for i in ix),'damages':sum(a[i]['prediction']==truth(i) and b[i]['prediction']!=truth(i) for i in ix),'E_logit_delta_mean':float(delta.sum()/len(delta)) if len(delta) else None,'E_logit_delta_abs_max':float(np.abs(delta).max()) if len(delta) else None}
    return out

def main(args):
    cb={'path':str(path(args.complete)),'sha256':args.complete_sha,'size_bytes':11524}
    c=load_json(cb);assert c['completed'] is True and c['PASS'] is True
    e=load_json(c['entry']);scope=load_json(c['root_scope'])
    assert scope['approved'] is True and scope['entry_contract']=={k:v for k,v in e.items() if k!='root_scope'}
    assert c['inputs']==e['inputs']
    for k in ('code','root_protocol','implementation_protocol'):checked(c[k])
    inputs={k:checked(b) for k,b in c['inputs'].items()}
    for k in ('summary','freeze','generic_banks','generic_bank_support','generic_reference_inputs','target_reference_inputs','target_banks_provenance','draw_ledger'):checked(c[k])
    for name in ('checkpoints','logs','outputs'):
        for b in c[name].values():checked(b)
    freeze=load_json(c['freeze']);assert freeze['completed'] is True and freeze['PASS'] is True
    assert freeze['checkpoints']==c['checkpoints'] and freeze['logs']==c['logs'] and freeze['generic_bank']==c['generic_banks'] and freeze['draw_ledger']==c['draw_ledger']
    assert freeze['BASE_CSV_bytes_or_interpretation_before_freeze']==freeze['new_DEV_head_outputs_before_freeze']==0
    iq=load_json(freeze['initial_QA']);assert iq['completed'] is True and iq['PASS'] is True and all(iq['initial_zero_outputs_allFIT'].values())
    x=np.load(inputs['original_features'],mmap_mode='r',allow_pickle=False)
    assert x.shape==(7617,1186) and x.dtype==np.float32
    with np.load(inputs['samples'],allow_pickle=False) as z:d={k:z[k] for k in ('condition','fractions','target','family','fit','available','row_id')}
    assert d['target'].shape==(7617,) and np.isin(d['target'],[0,1]).all() and d['fit'].sum()==5876 and (~d['fit']).sum()==1741
    assert x[:,-32:].tobytes()==d['condition'].tobytes() and x[:,1152:1154].tobytes()==d['fractions'].tobytes()
    assert not np.count_nonzero(d['condition'][d['family']==1])
    targetref=np.load(inputs['V251_saved_reference_inputs'],mmap_mode='r',allow_pickle=False)
    assert targetref.shape==(7617,9) and targetref.dtype==np.float32 and np.isfinite(targetref).all()
    assert c['target_reference_inputs']==c['inputs']['V251_saved_reference_inputs'] and c['target_banks_provenance']==c['inputs']['V251_saved_banks']
    # Enforce the prospectively fixed FULL raw FIT-D pools before bank comparison.
    banks=np.zeros((2,3,384),np.float64);supported=np.zeros((2,3),bool);bankrows=[];counts={}
    for family in (0,1):
        ids=np.flatnonzero(d['fit']&d['available']&(d['family']==family)&(d['target']==0));counts[str(family)]=len(ids)
        for v,view in enumerate(VIEWS):
            acc=np.zeros(384,np.float64)
            for i in ids:
                rawvec=np.asarray(x[i,v*384:(v+1)*384],np.float64)
                assert normalized(rawvec) is not None,('fixed full FIT-D descriptor unsupported',family,v,int(i))
                acc+=rawvec
            unit=normalized(acc/len(ids));assert unit is not None
            banks[family,v]=unit;supported[family,v]=True
            bankrows.append({'family':family,'view':view,'FIT_available_D_rows':len(ids),'accepted':len(ids),'sample_indices':list(map(int,ids)),'supported':True,'raw_stored_descriptor_equal_weight':True,'full_FIT_D_self_inclusion':True,'DEV_used':False,'address_or_object_specific_filter':False})
    support_packet=load_json(c['generic_bank_support']);same(bankrows,support_packet['rows'],'six_full_FIT_D_pools')
    assert support_packet['full_FIT_D_self_inclusion'] is True and support_packet['no_DEV_or_target_bank_averaging'] is True
    with np.load(path(c['generic_banks']['path']),allow_pickle=False) as z:
        same(banks.tolist(),z['prototypes'].tolist(),'generic6banks');assert np.array_equal(supported,z['supported']) and tuple(z['view'])==VIEWS and z['family'].tolist()==[0,1]
        bank_byte_exact=banks.tobytes()==z['prototypes'].tobytes()
    generic=np.zeros((7617,9),np.float32)
    for i in range(7617):
        family=int(d['family'][i])
        for v in range(3):
            q=normalized(x[i,v*384:(v+1)*384]);p=normalized(banks[family,v])
            if d['available'][i] and supported[family,v] and q is not None and p is not None:
                cs=np.float32(q@p);generic[i,v]=generic[i,3+v]=cs;generic[i,6+v]=1
    savedgeneric=np.load(path(c['generic_reference_inputs']['path']),allow_pickle=False)
    assert savedgeneric.dtype==np.float32 and savedgeneric.shape==(7617,9) and generic.tobytes()==savedgeneric.tobytes()
    for ref in (generic,targetref):
        assert np.isin(ref[:,6:],[0.,1.]).all()
        for v in range(3):assert not np.count_nonzero(ref[ref[:,6+v]==0,v]) and not np.count_nonzero(ref[ref[:,6+v]==0,3+v])
    refs={ARMS[0]:generic,ARMS[1]:targetref};reference_support={}
    for arm,ref in refs.items():
        reference_support[arm]={}
        for family in (0,1):
            for role in ('FIT','DEV'):
                ids=np.flatnonzero((d['family']==family)&(d['fit']==(role=='FIT')));n=len(ids);all3=int((ref[ids,6:]==1).all(1).sum())
                reference_support[arm][f'family{family}_{role}']={'family':family,'role':role,'original_rows':n,'all3_reference_supported':all3,'all3_supported_fraction':all3/n if n else None,'per_view_supported':{v:int((ref[ids,6+j]==1).sum()) for j,v in enumerate(VIEWS)},'reference_missing_does_not_change_model_U':True}
    pools=[np.flatnonzero(d['fit']&d['available']&(d['family']==f)&(d['target']==y)) for f in (0,1) for y in (0,1)]
    rng=np.random.Generator(np.random.PCG64(17));draw=np.stack([np.concatenate([rng.choice(p,8,replace=True) for p in pools]) for _ in range(1000)]).astype(np.int64)
    assert hashlib.sha256(draw.astype('<i8').tobytes()).hexdigest()==DRAW_SHA==c['draws_sha256']
    ledger=[json.loads(s) for s in raw(c['draw_ledger']['path']).splitlines()];assert len(ledger)==1000
    for arm in ARMS:
        log=[json.loads(s) for s in raw(c['logs'][arm]['path']).splitlines()];assert len(log)==1000
        for i,(l,dd) in enumerate(zip(log,ledger)):
            assert l['update']==dd['update']==i+1 and l['sample_indices']==dd['sample_indices']==draw[i].tolist()
            assert l['draws_sha256']==dd['draws_sha256']==DRAW_SHA and l['all_fit_available_known'] is True and dd['all_fit_available_known'] is True
            assert all(math.isfinite(l[k]) for k in ('family0_BCE','family1_BCE','total_loss')) and abs(l['total_loss']-(l['family0_BCE']+l['family1_BCE'])/2)<=1e-6
    shape={'net.0.weight':(32,41),'net.0.bias':(32,),'net.2.weight':(2,32),'net.2.bias':(2,)};initials=[];cp_checks={}
    for arm in ARMS:
        cp=torch.load(path(c['checkpoints'][arm]['path']),weights_only=True,map_location='cpu')
        assert cp['arm']==arm and cp['updates']==1000 and cp['parameters']==1410 and cp['input_width']==41 and cp['old_core_used'] is False and cp['threshold_raw_logit']==0 and cp['generic_bank']==c['generic_banks']
        for key in ('model_state_dict','initial_model_state_dict'):
            state=cp[key];assert set(state)==set(shape)
            for name,v in state.items():assert v.dtype==torch.float32 and tuple(v.shape)==shape[name] and bool(torch.isfinite(v).all())
            assert sum(v.numel() for v in state.values())==1410
        ini=cp['initial_model_state_dict'];final=cp['model_state_dict'];initials.append(ini)
        assert int(torch.count_nonzero(ini['net.2.weight']))==int(torch.count_nonzero(ini['net.2.bias']))==0
        assert state_digest(ini)==cp['initial_model_sha256']==c['initial_model_sha256'][arm]==iq['initial_model_sha256']
        assert state_digest(final)==c['final_model_sha256'][arm]
        opt=cp['optimizer_state_dict'];assert len(opt['state'])==4 and len(opt['param_groups'])==1
        pg=opt['param_groups'][0];assert pg['lr']==.001 and pg['weight_decay']==.0001 and len(pg['params'])==4
        for st in opt['state'].values():assert float(st['step'])==1000 and all(bool(torch.isfinite(v).all()) for v in st.values() if isinstance(v,torch.Tensor))
        cp_checks[arm]={'parameters':1410,'input_width':41,'state_tensors':4,'optimizer_states':4,'steps_each':1000,'initial_sha256':state_digest(ini),'final_sha256':state_digest(final),'old_core_used':False,'finite':True}
    assert all(torch.equal(initials[0][k],initials[1][k]) for k in shape)
    rows_by_arm={};metrics={}
    for arm,b in [('BASE_FROZEN_V239',c['BASE_FROZEN_V239_CSV']),*c['outputs'].items()]:
        rows=csv_rows(b['path']);assert len(rows)==7617;typed=[]
        for i,s in enumerate(rows):
            assert int(s['sample_index'])==i and s['row_id']==str(d['row_id'][i]) and int(s['family'])==d['family'][i] and int(s['target'])==d['target'][i]
            assert s['role']==('FIT' if d['fit'][i] else 'DEV') and s['available'] in ('True','False') and (s['available']=='True')==bool(d['available'][i])
            z=[float(s['logit0']),float(s['logit1'])];selected=float(s['selected_family_E_logit']);assert all(math.isfinite(x) for x in z+[selected]) and selected==z[int(d['family'][i])]
            pred='U' if not d['available'][i] else ('E' if selected>=0 else 'D');assert pred==s['prediction']
            typed.append({'sample_index':i,'row_id':s['row_id'],'family':int(s['family']),'role':s['role'],'target':int(s['target']),'available':s['available']=='True','logit0':z[0],'logit1':z[1],'selected_family_E_logit':selected,'prediction':pred})
        rows_by_arm[arm]=typed;metrics[arm]=reduce_source(typed)
    gate=calculate_gate(metrics['BASE_FROZEN_V239'],metrics[ARMS[0]],metrics[ARMS[1]])
    effect=paired_all(rows_by_arm[ARMS[0]],rows_by_arm[ARMS[1]])
    summary={'completed':True,'PASS':True,'BASE_FROZEN_V239':metrics['BASE_FROZEN_V239'],'arms':{a:metrics[a] for a in ARMS},'SOURCE_gate':gate,'paired_effects':effect,'reference_support':reference_support,'SYN_MAIN_vs_YCB_associated_r_AUX_separate':True,'all3views_joint_no_best_view':True,'generic_FIT_D_self_inclusion':True,'current_admission':False,'reused_SOURCE_DEV_not_blind':True}
    same(summary,load_json(c['summary']),'SOURCE_RESULTS')
    assert c['current_admission'] is False and c['new_DEV_head_outputs_before_bothfinalfreeze']==0 and c['old_core_CP_module_hidden_logits_or_preds_forward_inputs']==0
    assert c['original_rows']==15040 and c['original_masked_S_U']==7423 and c['full_FIT_D_self_inclusion'] is True
    for key in ('checkpoints','logs'):
        for b in c[key].values():checked(b)
    checked(c['freeze']);checked(c['generic_banks'])
    out=path(args.out);assert not out.exists();out.mkdir()
    report={'schema':'V252_INDEPENDENT_ACTUAL_DIRECT_REFERENCE_FIT_QA_C_V1','completed':True,'PASS':True,'fit_complete':cb,'entry':c['entry'],'root_scope':c['root_scope'],'root_protocol':c['root_protocol'],'producer_code':c['code'],'checker':binding(__file__),'freeze':c['freeze'],'generic_bank_QA':{'bank_vectors':6,'full_FIT_D_pool_counts':counts,'all_six_accepted_equals_full_pool':True,'ascending_exact_indices_and_no_DEV':True,'all_pool_descriptors_finite_nonzero':True,'raw_FP64_bank_byte_exact':bank_byte_exact,'own_FIT_D_included_as_frozen_policy':True,'no_target_bank_average':True,'generic_ref9_ALL7617_byte_exact':True,'original_condition32_and_fractions_byte_exact':True,'TREAT_ref9_exact_ancestor_cache_binding':c['target_reference_inputs']},'checkpoint_QA':cp_checks,'draw_log_QA':{'draw_shape':[1000,32],'draw_sha256':DRAW_SHA,'ledger1000_and_both_logs1000_indices_exact':True,'only_FIT_available_known':True,'loss_finite_family_mean':True},'metrics':metrics,'gates':gate,'paired_effects':effect,'reference_support':reference_support,'maximum_numeric_comparison_difference':MAX_DIFF,'numeric_values_compared':NUMERIC_COMPARISONS,'independence':'No producer import/call. Independent raw sequential FP64 bank means, genericcos/ref9, CM/precision/recall/U/average-rank Mann-Whitney AUC/gates and paired effects. New CP2 tensor/optimizer-state validation only; no neural replay/optimization.','temporal_limit':'Saved exact freeze bindings and previously audited code order are consistent with Root observed execution; this is not independent retrospective proof of read timing.','access':{'SOURCE_frozen_features_and_nonpixel_samples_and_ref9':True,'new_CP2_weights_only_CPU':True,'old_core_CP_module_hidden_decode_or_forward':0,'raw_RGB_GT_archives_DINO_weights_current_held':0,'producer_import_or_fit_optimizer_or_headforward':0,'Git_network':0},'limitations':['Generic CONTROL is supervised fullFIT-D reference and includes ownFIT-D samples; not unsupervised/no-reference.','Target-specific versus generic contrast includes representation/self-inclusion effects; not a pure causal identity test.','SYN original native-address MAIN and YCB associated-r AUX/nativeSELFUNKNOWN remain distinct; no physical native gold.','Repeated SOURCE DEV gate computation is not current success, adoption, new threshold/view/model selection.'],'current_admission':False,'selected_model':None}
    dest=out/'REVIEW.json';dest.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completed':True,'PASS':True,'REVIEW':binding(dest),'full_FIT_D_pool_counts':counts,'banks_byte_exact':bank_byte_exact,'reference9_byte_exact':True,'gates':gate},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--complete',required=True);p.add_argument('--complete-sha',required=True);p.add_argument('--out',required=True);args=p.parse_args();main(args)
