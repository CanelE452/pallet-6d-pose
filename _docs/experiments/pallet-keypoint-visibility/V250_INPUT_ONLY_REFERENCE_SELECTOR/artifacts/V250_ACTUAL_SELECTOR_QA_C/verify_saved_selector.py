#!/usr/bin/env python3
"""V250 SOURCE saved-packet QA. Independent NumPy arithmetic; no producer import."""
import argparse, ast, csv, datetime, hashlib, io, json, math, os, stat
from pathlib import Path
import numpy as np

VIEWS=('full_CLS','fixed_centre_Q','centre_CLS')
STRATA=('all','<=1px','>1_to_3px','<=3px','>3px','unknown')
MAX_DIFF=0.; NUMERIC_COMPARISONS=0

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

def independent_selector(nodes,orig,supported,active,bank,bank_supported):
    unit_nodes=[normalized(x) for x in nodes]
    eligible=[bool(orig[k] and supported[k] and unit_nodes[k] is not None and k!=active) for k in range(8)]
    slots=[k for k in range(8) if eligible[k]]
    p=[normalized(x) for x in bank]
    out={'bank_index':None,'bank_scores':[None]*4,'top2_slots_by_bank':[[] for _ in range(4)],'valid_other_nodes':len(slots),'candidate_mask':eligible,'selector_gap':None,'status':'UNKNOWN'}
    if any(x is None for x in p):out['reason']='INVALID_FIXED_BANK'
    elif len(slots)<2:out['reason']='FEWER_THAN_TWO_OTHER_NODES'
    else:
        values=[];tops=[]
        for q in p:
            cs=np.asarray([float(unit_nodes[k]@q) for k in slots],np.float64)
            rank=np.lexsort((np.asarray(slots),-cs))
            tops.append([slots[int(rank[0])],slots[int(rank[1])]])
            values.append(float((cs[rank[0]]+cs[rank[1]])/2.))
        order=np.lexsort((np.arange(4),-np.asarray(values)))
        out.update(bank_index=int(order[0]),bank_scores=values,top2_slots_by_bank=tops,selector_gap=float(values[order[0]]-values[order[1]]),status='SUPPORTED',reason=None)
    if not bool(bank_supported.all()):out.update(bank_index=None,status='UNKNOWN',reason='UNSUPPORTED_FIXED_BANK',selector_gap=None)
    return out

def parse_csv(rows,kind):
    ints={'sample_index','group_index','active_slot','target','object_id','logical_aux_slot','nominal_bank_index'}
    optional_int={'selected_bank_index','fixed_next_bank_index','selected_object_id'}
    booleans={'selector_supported','query_descriptor_supported','original_query_available','supported','auto_supported'}
    floats={'cosine_selected_bank','cosine_next_bank','selected_minus_next_margin','q_r_distance_px','oracle_cosine_nominal','oracle_cosine_other','oracle_margin'}
    out=[]
    for r in rows:
        d={}
        for k,v in r.items():
            if k in ints:d[k]=int(v)
            elif k in optional_int:d[k]=None if v=='' else int(v)
            elif k in booleans:assert v in ('True','False');d[k]=v=='True'
            elif k=='selection_correct':assert v in ('','True','False');d[k]=None if v=='' else v=='True'
            elif k in floats:d[k]=None if v=='' else float(v)
            elif k=='support_reasons':d[k]=json.loads(v)
            else:d[k]=v
        out.append(d)
    return out

def rank_auc(rows,field):
    data=[(r['target'],-r[field]) for r in rows if r[field] is not None]
    d=sum(y==0 for y,s in data);e=sum(y==1 for y,s in data)
    if not d or not e:return {'auc':None,'D_supported':d,'E_supported':e,'reason':'BOTH_CLASSES_REQUIRED'}
    ordered=sorted(data,key=lambda z:z[1]);positive_rank=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][1]==ordered[i][1]:j+=1
        positive_rank+=((i+1+j)/2)*sum(y==1 for y,s in ordered[i:j]);i=j
    return {'auc':(positive_rank-e*(e+1)/2)/(e*d),'D_supported':d,'E_supported':e,'reason':None,'ties':'half'}

def distribution(rows,field,target):
    a=np.asarray([r[field] for r in rows if r['target']==target and r[field] is not None],np.float64)
    if not len(a):return {'n':0,'mean':None,'quantiles':None}
    q=np.percentile(a,[0,5,25,50,75,95,100],method='linear')
    return {'n':len(a),'mean':float(np.sum(a)/len(a)),'quantiles':dict(zip(('min','p05','p25','median','p75','p95','max'),map(float,q)))}

def selection(rows):
    n=len(rows);ss=[r for r in rows if r['selected_bank_index'] is not None]
    c=sum(r['selected_bank_index']==r['nominal_bank_index'] for r in ss)
    return {'rows':n,'D':sum(r['target']==0 for r in rows),'E':sum(r['target']==1 for r in rows),'selector_supported':len(ss),'selector_UNKNOWN':n-len(ss),'selector_UNKNOWN_rate':(n-len(ss))/n if n else None,'correct':c,'accuracy_supported':c/len(ss) if ss else None}

def metrics(rows):
    out={'rows':len(rows),'D':sum(r['target']==0 for r in rows),'E':sum(r['target']==1 for r in rows),'query_UNKNOWN':sum(not r['auto_supported'] for r in rows)}
    for name,a,o in [('negative_cosine','cosine_selected_bank','oracle_cosine_nominal'),('negative_margin','selected_minus_next_margin','oracle_margin')]:
        supported=[r for r in rows if r[a] is not None];intersection=[r for r in supported if r[o] is not None];oracle=[r for r in rows if r[o] is not None]
        out[name]={'auto':rank_auc(supported,a),'same_support_auto':rank_auc(intersection,a),'same_support_oracle':rank_auc(intersection,o),'full_oracle':rank_auc(oracle,o),'same_support_rows':len(intersection),'original_denominator':len(rows)}
    fields=('cosine_selected_bank','cosine_next_bank','selected_minus_next_margin','oracle_cosine_nominal','oracle_margin')
    out['distributions']={f:{'D':distribution(rows,f,0),'E':distribution(rows,f,1)} for f in fields}
    out['selection']=selection(rows)
    return out

def in_stratum(r,s):
    q=r['q_r_distance_px']
    if s=='all':return True
    if s=='unknown':return q is None
    if q is None:return False
    return {'<=1px':q<=1,'>1_to_3px':1<q<=3,'<=3px':q<=3,'>3px':q>3}[s]

def main(args):
    complete_binding={'path':str(path(args.complete)),'sha256':'52823d695115cd68d900a17499a3c58aa844432a0391e7b14c00af5bba95ae79','size_bytes':9092}
    c=load_json(complete_binding);assert c['completed'] is True and c['PASS'] is True
    e=load_json(c['entry']);scope=load_json(e['root_scope'])
    assert scope['approved'] is True and scope['entry_contract']=={k:v for k,v in e.items() if k!='root_scope'}
    assert c['inputs']==e['inputs'] and c['metadata_provenance']==e['metadata_provenance']
    checked(c['code']);checked(c['root_protocol']);checked(e['implementation_protocol'])
    paths={k:checked(b) for k,b in c['inputs'].items()}
    for b in c['metadata_provenance'].values():checked(b)
    for b in c['outputs'].values():checked(b)
    freeze=load_json(c['prediction_freeze'])
    assert freeze['completed'] is True and freeze['PASS'] is True and freeze['target_fit_object_id_oracle_semantic_decodes_so_far']==0
    assert freeze['outputs']=={'selector':c['outputs']['SELECTOR1521_INPUT_ONLY.jsonl'],'query_scores':c['outputs']['QUERY4563_INPUT_ONLY_SCORES.csv']}
    code=ast.parse(raw(c['code']['path']))
    run=next(x for x in code.body if isinstance(x,ast.FunctionDef) and x.name=='run')
    late=[x.lineno for x in ast.walk(run) if isinstance(x,ast.Subscript) and isinstance(x.slice,ast.Constant) and x.slice.value in ('target','fit','object_id')]
    assert min(late)>261
    f=np.load(paths['ORIGINAL_ALL_FEATURES.npy'],mmap_mode='r',allow_pickle=False)
    n=np.load(paths['NODE_CLS.npy'],mmap_mode='r',allow_pickle=False)
    gi=np.load(paths['SAMPLE_GROUP_INDEX.npy'],allow_pickle=False);ni=np.load(paths['SAMPLE_NODE_INDEX.npy'],allow_pickle=False)
    assert f.shape==(7617,1186) and f.dtype==np.float32 and n.shape==(1767,8,384) and n.dtype==np.float32
    assert gi.shape==ni.shape==(7617,) and gi.dtype==ni.dtype==np.int64
    with np.load(paths['samples.npz'],allow_pickle=False) as z:family=z['family'];rid=z['row_id'];av=z['available']
    with np.load(paths['INPUT_GEOMETRY.npz'],allow_pickle=False) as z:oa=z['original_available8'];na=z['node_available']
    with np.load(paths['PROTOTYPES.npz'],allow_pickle=False) as z:p=z['prototypes'];ps=z['supported'];views=z['view']
    assert family.shape==av.shape==rid.shape==(7617,) and family.dtype==np.int8 and av.dtype==np.bool_ and rid.dtype.kind=='U'
    assert oa.shape==na.shape==(1767,8) and oa.dtype==na.dtype==np.bool_ and not (na&~oa).any()
    assert p.shape==(3,4,384) and p.dtype==np.float64 and ps.shape==(3,4) and ps.dtype==np.bool_ and tuple(views)==VIEWS
    saved_sel=[json.loads(s) for s in raw(c['outputs']['SELECTOR1521_INPUT_ONLY.jsonl']['path']).splitlines()]
    saved_score=parse_csv(csv_rows(c['outputs']['QUERY4563_INPUT_ONLY_SCORES.csv']['path']),'query')
    ids=np.flatnonzero(family==1);assert len(ids)==1521
    selectors=[];scores=[]
    for i0 in ids:
        i=int(i0);g=int(gi[i]);a=int(ni[i]);assert 0<=g<1767 and 0<=a<8
        assert f[i,:384].tobytes()==n[g,a].tobytes()
        s={'sample_index':i,'row_id':str(rid[i]),'group_index':g,'active_slot':a,'original_query_available':bool(av[i]),**independent_selector(n[g],oa[g],na[g],a,p[0],ps[0])}
        selectors.append(s)
        for v,name in enumerate(VIEWS):
            b=s['bank_index'];nb=None if b is None else (b+1)%4;q=normalized(f[i,v*384:(v+1)*384]);reasons=[]
            if b is None:reasons.append('SELECTOR_UNKNOWN')
            if not av[i]:reasons.append('ORIGINAL_QUERY_UNAVAILABLE')
            if q is None:reasons.append('QUERY_DESCRIPTOR_UNSUPPORTED')
            nom=None if b is None else normalized(p[v,b]);other=None if nb is None else normalized(p[v,nb])
            if b is not None and (not ps[v,b] or nom is None):reasons.append('SELECTED_PROTOTYPE_UNSUPPORTED')
            if nb is not None and (not ps[v,nb] or other is None):reasons.append('NEXT_PROTOTYPE_UNSUPPORTED')
            cs=co=margin=None
            if not reasons:cs=float(q@nom);co=float(q@other);margin=cs-co
            scores.append({'sample_index':i,'row_id':str(rid[i]),'view':name,'selected_bank_index':b,'fixed_next_bank_index':nb,'selector_supported':b is not None,'query_descriptor_supported':q is not None,'original_query_available':bool(av[i]),'supported':not reasons,'cosine_selected_bank':cs,'cosine_next_bank':co,'selected_minus_next_margin':margin,'support_reasons':reasons})
    same(selectors,saved_sel,'all1521_selector');same(scores,saved_score,'all4563_scores')
    # Late semantic join; independent selector computation above has no teacher inputs.
    with np.load(paths['samples.npz'],allow_pickle=False) as z:target=z['target'];fit=z['fit']
    with np.load(paths['PROTOTYPES.npz'],allow_pickle=False) as z:objects=z['object_id']
    assert objects.dtype==np.int64 and objects.tolist()==sorted(set(objects.tolist())) and len(objects)==4
    oracle=csv_rows(paths['V249_DEV_ORACLE_SCORES.csv']);sm={r['sample_index']:r for r in selectors};qm={(r['sample_index'],r['view']):r for r in scores}
    expected_dev={int(i) for i in ids if not fit[i]};assert len(expected_dev)==397 and sum(target[i]==0 for i in expected_dev)==340 and sum(target[i]==1 for i in expected_dev)==57
    old={};comp=[];checks=0;objmap={int(o):j for j,o in enumerate(objects)}
    for r in oracle:
        i=int(r['sample_index']);v=r['view'];key=(i,v);assert key not in old and i in expected_dev and v in VIEWS
        assert r['row_id']==str(rid[i]) and r['role']=='DEV' and int(r['target'])==int(target[i]) and (r['available']=='True')==bool(av[i])
        assert r['associated_r_state']==('D' if target[i]==0 else 'E');old[key]=r
    assert set(old)=={(i,v) for i in expected_dev for v in VIEWS}
    for i in sorted(expected_dev):
        for v in VIEWS:
            r=old[(i,v)];s=qm[(i,v)];obj=int(r['object_id']);b=objmap[obj]
            assert int(r['fixed_other_object_id'])==int(objects[(b+1)%4])
            val=lambda k:None if r[k]=='' else float(r[k])
            oc,oo,om=val('cosine_nominal_target'),val('cosine_fixed_other_target'),val('nominal_minus_other_margin')
            if s['selected_bank_index']==b and s['supported']:
                same([s['cosine_selected_bank'],s['cosine_next_bank'],s['selected_minus_next_margin']],[oc,oo,om],'correct_bank_oracle');checks+=1
            comp.append({'sample_index':i,'row_id':str(rid[i]),'view':v,'role':'DEV','target':int(target[i]),'associated_r_state':r['associated_r_state'],'object_id':obj,'scene':r['scene'],'frame_id':r['frame_id'],'logical_aux_slot':int(r['logical_aux_slot']),'q_r_distance_px':val('q_r_distance_px'),'nominal_bank_index':b,'selected_bank_index':s['selected_bank_index'],'selected_object_id':None if s['selected_bank_index'] is None else int(objects[s['selected_bank_index']]),'selection_correct':None if s['selected_bank_index'] is None else s['selected_bank_index']==b,'auto_supported':s['supported'],'cosine_selected_bank':s['cosine_selected_bank'],'cosine_next_bank':s['cosine_next_bank'],'selected_minus_next_margin':s['selected_minus_next_margin'],'oracle_cosine_nominal':oc,'oracle_cosine_other':oo,'oracle_margin':om,'support_reasons':s['support_reasons']})
    same(comp,parse_csv(csv_rows(c['outputs']['DEV1191_AUTO_ORACLE_COMPARISON.csv']['path']),'comparison'),'DEV1191')
    tables=[]
    for v in VIEWS:
        for st in STRATA:
            sr=[r for r in comp if r['view']==v and in_stratum(r,st)]
            for obj in [None,*map(int,objects)]:
                rows=sr if obj is None else [r for r in sr if r['object_id']==obj]
                tables.append({'view':v,'q_r_stratum':st,'object_id':obj,**metrics(rows)})
    table_packet={'tables':tables,'fixed_views':list(VIEWS),'fixed_q_r_strata':list(STRATA)}
    same(table_packet,load_json(c['outputs']['DESCRIPTIVE_TABLES.json']),'all90_tables')
    unique=[r for r in comp if r['view']==VIEWS[0]]
    counts={'YCB_prediction_queries':len(selectors),'input_only_view_scores':len(scores),'DEV_queries':len(expected_dev),'DEV_D':340,'DEV_E':57,'DEV_view_rows':len(comp),'correctbank_oracle_numeric_checks':checks}
    summary={'counts':counts,'selection_pooled':selection(unique),'selection_by_state':{name:selection([r for r in unique if r['target']==y]) for name,y in [('D',0),('E',1)]},'pooled_all_by_view':{v:next(t for t in tables if t['view']==v and t['q_r_stratum']=='all' and t['object_id'] is None) for v in VIEWS},'decision':'DESCRIPTIVE_SOURCE_DIAGNOSTIC_NO_MODEL_OR_THRESHOLD_SELECTION','current_admission':False}
    same(summary,load_json(c['outputs']['SUMMARY.json']),'SUMMARY');same(counts,c['counts'],'COMPLETEcounts')
    assert c['new_reference_fit']==c['optimizer_updates']==c['new_encoder_or_head_forward']==c['current_or_held_payloads']==0 and c['current_admission'] is False
    out=path(args.out);assert not out.exists();out.mkdir()
    report={'schema':'V250_INDEPENDENT_ACTUAL_SELECTOR_SAVED_MATH_QA_C_V1','completed':True,'PASS':True,'created_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':complete_binding,'entry':c['entry'],'root_scope':e['root_scope'],'root_protocol':c['root_protocol'],'producer_code':c['code'],'inputs':c['inputs'],'outputs':c['outputs'],'prediction_freeze':c['prediction_freeze'],'independent_checker':binding(__file__),'counts':counts,'selection':summary['selection_pooled'],'selection_by_state':summary['selection_by_state'],'all_views':summary['pooled_all_by_view'],'all_strata_object_metric_tables_verified':len(tables),'maximum_absolute_numeric_difference':MAX_DIFF,'numeric_values_compared':NUMERIC_COMPARISONS,'math_independence':'No producer import/call. Separately implemented FP64 normalization, lexsort top2/tie selector, query scores and average-rank Mann-Whitney AUROC; producer uses pairwise AUROC. Full saved rows/tables recomputed.','freeze_verification':'Static code order and exact saved freeze/output bindings verified. This is consistent with observed Root execution; a retrospective file check alone does not independently prove historical read timing.','packet_access':{'samples_members':['family','row_id','available','target','fit'],'geometry_members':['original_available8','node_available'],'prototype_members':['prototypes','supported','view','object_id'],'numeric_YCB_source_population':1521,'source_derived_teacher_metadata_late_join':True,'raw_RGB_GT_archives_models_weights_current_or_held':0,'producer_operational_imports_or_calls':0,'optimizer_encoder_head_forward':0,'Git_network':0},'limitations':['SOURCE FIT supervised banks, not unsupervised; no new bank fitting in this diagnostic.','Closed four-bank nominal teacher-object selection, not native visibility accuracy or deployment correctness.','Associated-r labels, GT-jitter q, repeated exposed SOURCE DEV and narrow object/scene composition remain.','No model/data/architecture/threshold/best-view decision; current_admission false.'],'current_admission':False,'selected_model':None,'administrative_intake_failures':binding(path(__file__).parent/'METADATA_INTAKE_ADMIN_FAILURES.json')}
    dest=out/'REVIEW.json';dest.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completed':True,'PASS':True,'REVIEW':binding(dest),'counts':counts,'selection':summary['selection_pooled'],'numeric_maxabs':MAX_DIFF},ensure_ascii=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--complete',required=True);a.add_argument('--out',required=True);args=a.parse_args();main(args)
