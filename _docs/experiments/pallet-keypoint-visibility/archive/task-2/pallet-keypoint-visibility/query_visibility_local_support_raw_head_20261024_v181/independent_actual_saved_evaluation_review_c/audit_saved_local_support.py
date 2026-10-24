"""Independent scalar review of saved source scores/targets. No producer import.

Reads only root-completed score NPZs, separate target NPZs and text metadata.
Never opens input pixels/features, model checkpoints, GT masks/depth or weights.
"""
from pathlib import Path
import hashlib,json,math,time,traceback
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
COMPLETE=ROOT/'saved_local_support_source_evaluation_v1/COMPLETE.json'
ARMS=('C50_RAW96_DUAL_SUPPORT_OWNER_AUX','C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX')
DOMAINS={'SYNTH_TRAIN4096':4096,'PUBLIC_TRAIN1480':1480,'CF3816':3816,'SYNTH_VAL2048':2048}
USED={}
COUNTS={'numeric_or_boolean_leaves':0,'other_leaves':0,'metric_stratum_blocks':0,'decoded_rows':0,'ledger_rows':0,'max_abs_numeric_difference':0.}

def need(ok,msg):
    if not bool(ok): raise AssertionError(msg)

def digest(p):
    p=Path(p).resolve()
    need(p.is_relative_to(ROOT.parent),'Outside exact project source lineage')
    need(not any(x in str(p).lower() for x in ('_v112','_v169','_v177','_v178','current223','/private_')),'Forbidden source lineage')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()

def bound(p):return {'path':str(Path(p).resolve()),'sha256':digest(p)}
def verified(b):
    p=Path(b['path']);need(digest(p)==b['sha256'],'Source SHA '+str(p));USED[str(p.resolve())]=b['sha256'];return p
def readj(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def npz(b,keys):
    with np.load(verified(b),allow_pickle=False) as z:
        return {k:z[k].copy() for k in keys}
def divide(a,b):return None if b==0 else a/b

def compare(got,want,path='results'):
    if isinstance(want,dict):
        need(isinstance(got,dict) and set(got)==set(want),'Dict keys '+path)
        for k,v in want.items():compare(got[k],v,path+'/'+str(k))
    elif isinstance(want,list):
        need(isinstance(got,list) and len(got)==len(want),'List '+path)
        for i,v in enumerate(want):compare(got[i],v,path+'/'+str(i))
    elif isinstance(want,(bool,int,float)):
        need(got==want,'Numeric mismatch '+path+': '+str(got)+' != '+str(want))
        if isinstance(want,(int,float)) and not isinstance(want,bool):
            COUNTS['max_abs_numeric_difference']=max(COUNTS['max_abs_numeric_difference'],abs(got-want))
        COUNTS['numeric_or_boolean_leaves']+=1
    else:
        need(got==want,'Leaf mismatch '+path);COUNTS['other_leaves']+=1

def categorical(truth,pred,nclasses):
    need(len(truth)==len(pred),'Truth/pred length')
    matrix=[[0 for _ in range(nclasses+1)] for _ in range(nclasses)]
    unk=[0 for _ in range(nclasses+1)]
    for y,p in zip(truth,pred):
        y=int(y);p=int(p)
        need(-1<=y<nclasses and -1<=p<nclasses,'Category bounds')
        column=nclasses if p==-1 else p
        if y==-1:unk[column]+=1
        else:matrix[y][column]+=1
    support=[sum(r) for r in matrix]
    out={'CM':matrix,'truth_classes':list(range(nclasses)),
         'prediction_columns':list(range(nclasses))+['U'],'rows_all':len(truth),
         'truth_known':sum(support),'truth_unknown':sum(unk),'class_support':support,
         'precision':[divide(matrix[k][k],sum(matrix[j][k] for j in range(nclasses))) for k in range(nclasses)],
         'recall':[divide(matrix[k][k],support[k]) for k in range(nclasses)],
         'prediction_U_among_known':sum(matrix[k][-1] for k in range(nclasses)),
         'prediction_counts_among_unknown_truth':unk}
    if nclasses==2:
        out.update(TN=matrix[0][0],FP=matrix[0][1],FN_without_positive_U=matrix[1][0],TP=matrix[1][1],U_per_truth=[matrix[0][2],matrix[1][2]])
    return out

def owner_mass(y,amb,invalid,pred):
    cells=[[[] for _ in range(4)] for _ in range(3)]
    aa=[];ii=[];missing=0
    for weights,a,v,p in zip(y,amb,invalid,pred):
        p=int(p);need(-1<=p<=2,'Owner class')
        if all(math.isnan(float(x)) for x in weights):
            need(math.isnan(float(a)) and math.isnan(float(v)),'Unobserved owner ancillary fields')
            missing+=1;continue
        floats=[float(x) for x in weights]+[float(a),float(v)]
        need(all(math.isfinite(x) and 0<=x<=1 for x in floats),'Nonnegative finite partial owner mass')
        need(abs(math.fsum(floats)-1.)<=2e-12,'Known+ambiguous+invalid owner mass balance')
        col=3 if p==-1 else p
        for k in range(3):cells[k][col].append(floats[k])
        aa.append(float(a));ii.append(float(v))
    cm=[[math.fsum(v) for v in row] for row in cells]
    sums=[math.fsum(v) for v in cm]
    return {'CM_T_F_BG_to_T_F_BG_U':cm,'rows_all':len(pred),'rows_owner_truth_unobserved_or_missing':missing,
            'raw_known_class_mass':sums,'ambiguous_mass':math.fsum(aa),'invalid_mass':math.fsum(ii),
            'precision':[divide(cm[k][k],math.fsum(cm[j][k] for j in range(3))) for k in range(3)],
            'recall':[divide(cm[k][k],sums[k]) for k in range(3)],
            'prediction_U_known_mass':math.fsum(r[3] for r in cm),
            'no_cutoff_or_row_renormalization':True,'not_native_visibility':True}

def code_joint(v):
    out=[]
    for visible,foreign in v:
        out.append(-1 if visible<0 or foreign<0 else int(visible)+2*int(foreign))
    return out

def proxy_truth(v):
    out=[]
    for visible,foreign in v:
        out.append(-1 if visible<0 or foreign<0 or (visible==0 and foreign==0) else 1 if foreign==1 else 0)
    return out

def native_association(rr,pred):
    roles=['D','E','S','U'];m=[[0,0,0] for _ in range(4)];missing=0
    for r,p in zip(rr,pred):
        role=r.get('original_semantic_native_role')
        if role not in roles:missing+=1;continue
        m[roles.index(role)][2 if p<0 else int(p)]+=1
    return {'declared_semantic_rows':roles,'regional_proxy_columns':['D','E','U'],'counts':m,'rows_no_original_native_truth':missing,
            'authority':'Original CF/V4 declared semantic association only; physical initializer native identity is uncertified. This is separate from fixed support truth and is not a relabel or success gate.'}

def grouping(rr):
    g={'ALL':list(range(len(rr)))}
    for native in range(8):g['native:'+str(native)]=[i for i,r in enumerate(rr) if int(r['native_slot'])==native]
    for field in ('material','aspect','geometry_family','foreign_family','variant'):
        values=[]
        for r in rr:
            original=r['original']
            values.append(original.get('variant','UNDECLARED') if field=='variant' else original.get('factors',{}).get(field,'UNDECLARED'))
        for value in sorted(set(values),key=str):g[field+':'+str(value)]=[i for i,x in enumerate(values) if x==value]
    for field in ('original_semantic_native_role','reference_error_stratum','position_band'):
        values=[str(r[field]) if r.get(field) is not None else 'NOT_AVAILABLE' if field=='original_semantic_native_role' else 'missing' for r in rr]
        for value in sorted(set(values)):g[field+':'+value]=[i for i,x in enumerate(values) if x==value]
    return g

def summarize(rr,available,y,support,owner,proxy):
    result={}
    for name,ids in grouping(rr).items():
        part=[rr[i] for i in ids];av=[bool(available[i]) for i in ids]
        target=[y['support_target'][i].tolist() for i in ids]
        sp=[support[i] for i in ids];ow=[owner[i] for i in ids];px=[proxy[i] for i in ids]
        traits={}
        for k,label in enumerate(('visible_end_support','foreign_front_support')):
            known_available_truth=[int(target[i][k]) for i,a in enumerate(av) if a]
            available_pred=[int(sp[i][k]) for i,a in enumerate(av) if a]
            traits[label]={'primary_known_available':categorical(known_available_truth,available_pred,2),
                           'secondary_known_all':categorical([int(t[k]) for t in target],[int(p[k]) for p in sp],2)}
        truth=proxy_truth(target)
        block={'rows_all':len(ids),'original_input_available':sum(av),'original_inputU':len(ids)-sum(av),**traits,
          'joint4':categorical(code_joint(target),code_joint(sp),4),
          'owner_soft':owner_mass([y['owner_soft'][i] for i in ids],[y['owner_ambiguous_mass'][i] for i in ids],[y['owner_invalid_mass'][i] for i in ids],ow),
          'owner_hard_weak_only':categorical([int(y['owner_target'][i]) for i in ids],ow,3),
          'regional_proxy_D0_E1':categorical(truth,px,2),
          'original_native_semantic_proxy_association':native_association(part,px),
          'proxy_truth_NEITHER_or_UNKNOWN_rows':sum(x<0 for x in truth),'source_support_not_physical_native_gold':True}
        result[name]=block;COUNTS['metric_stratum_blocks']+=1
    return result

def scalar_decode(x):
    support=[];owner=[];proxy=[]
    for i,av in enumerate(x['original_query_available']):
        rs=[float(v) for v in x['raw_support_logits'][i]];ro=[float(v) for v in x['raw_owner_logits'][i]]
        if not av:
            need(all(math.isnan(v) for v in rs+ro),'U raw NaN first')
            support.append([-1,-1]);owner.append(-1);proxy.append(-1)
        else:
            need(all(math.isfinite(v) for v in rs+ro),'Available raw finite')
            s=[int(v>0.) for v in rs]
            # Python max returns the first index on exact ties.
            o=max(range(3),key=lambda k:ro[k])
            support.append(s);owner.append(o);proxy.append(1 if s[1] else 0 if s[0] else -1)
    need(np.array_equal(np.array(support,np.int8),x['support_prediction']),'Stored support decoder')
    need(np.array_equal(np.array(owner,np.int8),x['owner_prediction']),'Stored owner decoder')
    need(np.array_equal(np.array(proxy,np.int8),x['regional_support_proxy_D0_E1_Uminus1']),'Stored proxy decoder')
    COUNTS['decoded_rows']+=len(owner)
    return support,owner,proxy

def normq(q):return np.asarray([np.nan,np.nan] if q is None else q,np.float64)
def normL(L):return float('nan') if L is None else float(L)

def identity(x,rr,n):
    need(len(rr)==n and x['query_xy'].shape==(n,2) and x['L'].shape==(n,),'Complete metadata row count')
    for key,dt in [('query_xy',np.float64),('L',np.float64),('original_global_index',np.int64),('native_slot',np.int8),('original_query_available',np.bool_),('raw_support_logits',np.float32),('raw_owner_logits',np.float32),('support_prediction',np.int8),('owner_prediction',np.int8),('regional_support_proxy_D0_E1_Uminus1',np.int8)]:
        need(x[key].dtype==dt,'Saved dtype '+key)
    need(x['raw_support_logits'].shape==(n,2) and x['raw_owner_logits'].shape==(n,3),'Saved raw shapes')
    need(len(set(str(v) for v in x['row_id']))==n,'Unique occurrence row IDs')
    for i,r in enumerate(rr):
        need(str(x['row_id'][i])==r['row_id'] and int(x['original_global_index'][i])==r['original_global_index'] and int(x['native_slot'][i])==r['native_slot'],'ID ordering')
        need(np.array_equal(x['query_xy'][i],normq(r['query_xy']),equal_nan=True),'q unchanged')
        L=normL(r['L']);need((math.isnan(L) and math.isnan(float(x['L'][i]))) or L==float(x['L'][i]),'L unchanged')
        need(bool(x['original_query_available'][i])==r['original_query_available'],'Original U unchanged')

def native_metadata(rr,diagnostic):
    for r in rr:
        if r['split'] in ('TRAIN','VALIDATION'):
            d=diagnostic[r['original_global_index']]
            need(d['original_occurrence_id']==r['row_id'] and d['native_slot']==r['native_slot'] and d['split']==r['split'],'Source diagnostic exact ID')
            need(np.array_equal(normq(d['query_xy']),normq(r['query_xy']),equal_nan=True) and d['L']==r['L'],'Source diagnostic qL')
            e=d['q_reference_error_px']
            r.update(original_semantic_native_role=d['original_semantic_state'],reference_error_stratum='missing' if e is None or not math.isfinite(e) else '<=3px' if e<=3 else '>3px',position_band=d['position_band'])
        else:r.update(original_semantic_native_role='NOT_AVAILABLE',reference_error_stratum='missing',position_band='missing')

def v61_result(binding,packets):
    keys=['global_index','occurrence_id','native_slot','query_xy','L','query_available','raw_visible_logit','prediction_D0_E1_Uminus1','head_forwarded','native_query_valid','ROI_any_valid','native_continuous_valid','native_nearest_valid']
    z=npz(binding,keys);need(len(z['global_index'])==6144,'v61 original full6144')
    need(z['raw_visible_logit'].dtype==np.float32,'v61 FP32 raw')
    own=[];av=[]
    for i in range(6144):
        need(int(z['global_index'][i])==i,'v61 original index order')
        valid=bool(z['head_forwarded'][i] and z['native_query_valid'][i] and z['ROI_any_valid'][i])
        need(bool(z['query_available'][i])==valid,'v61 original availability rule')
        need(bool(z['native_query_valid'][i])==bool(z['native_continuous_valid'][i] and z['native_nearest_valid'][i]),'v61 original native validity')
        p=-1 if not valid else 0 if float(z['raw_visible_logit'][i])>=0. else 1
        need(p==int(z['prediction_D0_E1_Uminus1'][i]),'v61 saved threshold0 prediction')
        own.append(p);av.append(valid)
    result={}
    for domain in ('SYNTH_TRAIN4096','SYNTH_VAL2048'):
        x,y,rr=packets[domain];ids=[int(i) for i in x['original_global_index']]
        for j,i in enumerate(ids):
            need(str(z['occurrence_id'][i])==rr[j]['row_id'] and int(z['native_slot'][i])==rr[j]['native_slot'],'v61 exact original occurrence/native')
            need(np.array_equal(z['query_xy'][i],x['query_xy'][j],equal_nan=True),'v61 same q')
            a=float(z['L'][i]);b=float(x['L'][j]);need(a==b or math.isnan(a) and math.isnan(b),'v61 same L')
        support=[[-1,-1] if not av[i] else [int(own[i]==0),int(own[i]==1)] for i in ids]
        m=summarize(rr,[av[i] for i in ids],y,support,[-1]*len(ids),[own[i] for i in ids])
        for name,ii in grouping(rr).items():
            block=m[name]
            block['historical_v61_available']=block.pop('original_input_available');block['historical_v61_U']=block.pop('original_inputU')
            block['original_V181_query_available']=sum(bool(x['original_query_available'][j]) for j in ii)
            block['original_V181_inputU']=len(ii)-block['original_V181_query_available']
            del block['owner_soft'];del block['owner_hard_weak_only'];block['owner3_output_not_defined']=True
        result[domain]=m
    return {'meaning':'Historical pixel-presence binary sign shown against experimental regional-end targets only as a task-mismatch diagnostic. v61 cannot predict BOTH, has its own original availability, and has no owner3 output. No equivalence or native certification claimed.','baseline':result}

def main():
    started=time.monotonic();complete=readj(COMPLETE);USED[str(COMPLETE)]=digest(COMPLETE)
    need(complete['completed'] and complete['PASS'] and complete['all2_frozen_before_readout'],'Actual completed all2 source evaluation')
    protocol=readj(verified(complete['protocol']));fit=readj(verified(complete['fit_complete']))
    need(fit['PASS'] and fit['total_updates']==16000 and fit['all2_frozen_before_outputs_or_VAL_access'],'Fixed final fits')
    freeze=readj(verified(complete['all2_freeze']));need(freeze['completed'] and not freeze['saved_outputs_emitted'] and not freeze['validation_cache_or_targets_read'],'All2 barrier')
    train=readj(verified(complete['training_cache_complete']));val=readj(verified(complete['evaluation_cache_complete']))
    caches={**train['domains'],**val['domains']};need(set(caches)==set(DOMAINS),'Four full source domains')
    dd=rows(verified(protocol['source_diagnostic_metadata_inherited_binding_not_read_or_hashed']))
    need(len(dd)==6144 and [r['original_global_index'] for r in dd]==list(range(6144)),'Original diagnostic census6144')
    diagnostic={r['original_global_index']:r for r in dd};targetdata={};metadata={}
    targetkeys=['support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass']
    for domain,n in DOMAINS.items():
        targetdata[domain]=npz(caches[domain]['targets'],targetkeys)
        y=targetdata[domain];need(y['support_target'].dtype==np.int8 and y['support_target'].shape==(n,2) and np.isin(y['support_target'],[-1,0,1]).all(),'Support target schema')
        need(y['owner_soft'].dtype==np.float64 and y['owner_soft'].shape==(n,3) and y['owner_target'].dtype==np.int8 and np.isin(y['owner_target'],[-1,0,1,2]).all(),'Owner target schema')
        metadata[domain]=rows(verified(caches[domain]['metadata']));need(len(metadata[domain])==n,'Target metadata full row count')
        expected='VALIDATION' if domain=='SYNTH_VAL2048' else 'TRAIN' if domain=='SYNTH_TRAIN4096' else 'SOURCE_DEVELOPMENT'
        need(all(r['split']==expected for r in metadata[domain]),'Fixed original splits')
        native_metadata(metadata[domain],diagnostic)
    expected_metrics={};expected_ledgers={};coverage={};first_packets={}
    keys=['raw_support_logits','raw_owner_logits','support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1','support_order','owner_order','row_id','original_global_index','native_slot','query_xy','L','original_query_available']
    for arm in ARMS:
        need(fit['arms'][arm]['updates']==8000,'Each final8000')
        expected_metrics[arm]={};expected_ledgers[arm]=[];coverage[arm]={}
        for domain,n in DOMAINS.items():
            x=npz(fit['arms'][arm]['outputs'][domain],keys);y=targetdata[domain];rr=metadata[domain]
            identity(x,rr,n)
            need(x['support_order'].tolist()==['visible_end_support','foreign_front_support'] and x['owner_order'].tolist()==['TARGET','FOREIGN','BACKGROUND'],'Output task order')
            sp,op,proxy=scalar_decode(x)
            expected_metrics[arm][domain]=summarize(rr,x['original_query_available'],y,sp,op,proxy)
            coverage[arm][domain]={'rows':n,'input_U':sum(not bool(v) for v in x['original_query_available'])}
            for i,r in enumerate(rr):
                expected_ledgers[arm].append(dict(domain=domain,row_id=r['row_id'],original_global_index=r['original_global_index'],native_slot=r['native_slot'],split=r['split'],query_xy=r['query_xy'],L=r['L'],original_query_available=r['original_query_available'],support_target=y['support_target'][i].tolist(),support_prediction=sp[i],owner_target=int(y['owner_target'][i]),owner_prediction=op[i],regional_support_proxy=proxy[i],new_native_visibility_or_quality_label=False))
            if arm==ARMS[0]:first_packets[domain]=(x,y,rr)
    expected_metrics['historical_v61_separate_fixed_diagnostic']=v61_result(protocol['v61_metadata_only_readiness']['inherited_output_binding_not_opened_or_rehashed'],first_packets)
    # Stored summary/ledgers are first opened only after own raw-score arithmetic.
    stored=readj(verified(complete['summary']))
    compare(stored['results'],expected_metrics)
    need(stored['training_domains_apparent_only'] and stored['syntheticVAL_reused_research_split'] and stored['task_and_label_authority_separated'] and stored['selected_model'] is None and stored['fully_solved'] is False,'Interpretation limits')
    for arm in ARMS:
        ledger=rows(verified(complete['row_ledgers'][arm]));need(len(ledger)==11440,'Full original11440 ledger')
        compare(ledger,expected_ledgers[arm],'ledger/'+arm);COUNTS['ledger_rows']+=len(ledger)
    for path,s in USED.items():need(digest(path)==s,'Read asset changed during audit')
    review={'completed':True,'PASS':True,'code':bound(__file__),'evaluation_complete':bound(COMPLETE),'evaluation_protocol':complete['protocol'],'fit_complete':complete['fit_complete'],'frozen_barrier':complete['all2_freeze'],
        'checks':[{'name':n,'PASS':True} for n in ['All2x11440 raw-score/target/ROWS identity and originalqL/native/split/U','Independent scalar strict>0/firsttie owner/foreignpriority with Ufirst NaN','Every domain/ALL/native8/factor/native-role/referror/position stratum block','Primaryknownavailable vssecondaryknownall supportCM/P/R/U/unknown','Nonexclusivejoint4 and BOTHknown conditionalregionalproxy with NEITHER/U retained','Partialowner mass softCM/P/R plus ambiguity/invalid/missing without renormalization','Historicalv61 original6144 sameqL/ID/native/ownavailability/>=0 sign separate task','Stored11440 rowledgers eacharm and all summary scalar leaves','All consumed score/target/text bindings unchanged; inputpixels/features/checkpoints not opened']],
        'counts':COUNTS,'coverage':coverage,'bindings_verified_before_and_after':len(USED),'consumed_bindings':[{'path':p,'sha256':s} for p,s in sorted(USED.items())],
        'producer_function_import_or_reuse':0,'new_model_forward_fit_or_threshold_selection':0,'raw_RGB_mask_depth_input_features_or_checkpoint_bodies_read':0,'private_or_publicheld_bodies_read':0,
        'limitations':['Saved arithmetic consistency only, not physical GT/native identity or generalization certification.','TRAIN metrics are apparent fit. Synthetic VALIDATION was historically reused; no model/data/threshold choice is made.','v61 is a pixel-presence sign diagnostic, not an owner3 or nonexclusive end-support classifier.'],
        'selected_model':None,'adopted':False,'fully_solved':False,'seconds':time.monotonic()-started}
    (HERE/'REVIEW.json').write_text(json.dumps(review,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'completed':True,'PASS':True,'counts':COUNTS,'review':bound(HERE/'REVIEW.json')}))

if __name__=='__main__':
    try:main()
    except BaseException:
        (HERE/'FAILED_REVIEW_PRESERVED.json').write_text(json.dumps({'completed':False,'traceback':traceback.format_exc(),'automatic_retry':False},indent=2)+'\n')
        raise
