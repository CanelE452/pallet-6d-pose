"""Independent SOURCE prototype/cosine and rank-sum AUC QA; no producer import.
Only COMPLETE-declared three inputs, saved outputs and entry/scope metadata are read.
"""
import argparse,csv,hashlib,io,json,math,os,stat,traceback
from pathlib import Path
from collections import Counter
import numpy as np

COLS=('row_id','sample_index','role','scene','frame_id','object_id','aux_slot','associated_r_state','q_x','q_y','r_x_GT_diagnostic_only','r_y_GT_diagnostic_only','q_reference_distance_px','L')
VIEWS={'full_CLS':(0,384),'fixed_centre_Q':(384,768),'centre_CLS':(768,1152)}
STRATA=('all','<=1px','>1_to_3px','<=3px','>3px','unknown')

def need(ok,message):
    if not ok:raise ValueError(message)

def raw(p):
    p=Path(p);need(p.is_absolute() and '..' not in p.parts,'absolute path')
    need(not any(x in str(p).casefold() for x in ('v112','cache','.pyc','current','held','private','holdout','oldval','v177','v178','v182')),'lexical before FS')
    for a in reversed(p.parents):need(stat.S_ISDIR(os.lstat(a).st_mode),'symlink ancestor')
    need(stat.S_ISREG(os.lstat(p).st_mode),'nonregular/symlink leaf')
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def compare(a,b,label=''):
    if isinstance(a,dict):
        need(set(a)==set(b),'dictionary schema '+label)
        for k in a:compare(a[k],b[k],label+'/'+str(k))
    elif isinstance(a,list):
        need(len(a)==len(b),'list shape '+label)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,label+'/'+str(i))
    elif isinstance(a,float):need(isinstance(b,(int,float)) and math.isfinite(b) and abs(a-b)<=2e-13,'FP64 result '+label)
    else:need(type(a) is type(b) and a==b,'value '+label)

def number(v):
    if v is None or str(v).strip() in ('','UNKNOWN','null','None','NaN','nan'):return None
    x=float(v);return x if math.isfinite(x) else None

def stats(v):
    a=np.asarray([x for x in v if x is not None],np.float64)
    if not len(a):return {'n':0,'mean':None,'quantiles':None}
    return {'n':len(a),'mean':float(np.mean(a)),'quantiles':dict(zip(('min','p05','p25','median','p75','p95','max'),[float(x) for x in np.percentile(a,[0,5,25,50,75,95,100])]))}

def rank_auc(rr,field,lower_is_E=True):
    direction=-1 if lower_is_E else 1
    items=sorted(((direction*r[field],r['target']) for r in rr if r[field] is not None),key=lambda x:x[0])
    nE=sum(y==1 for _,y in items);nD=len(items)-nE
    if not nE or not nD:return {'auc':None,'D_supported':nD,'E_supported':nE,'reason':'BOTH_CLASSES_REQUIRED'}
    ranks=0.;start=0
    while start<len(items):
        end=start+1
        while end<len(items) and items[end][0]==items[start][0]:end+=1
        ranks+=((start+1+end)/2)*sum(y==1 for _,y in items[start:end]);start=end
    return {'auc':(ranks-nE*(nE+1)/2)/(nE*nD),'D_supported':nD,'E_supported':nE,'reason':None,'ties':'half'}

def in_stratum(d,s):
    if s=='all':return True
    if d is None:return s=='unknown'
    return {'<=1px':d<=1,'>1_to_3px':1<d<=3,'<=3px':d<=3,'>3px':d>3,'unknown':False}[s]

def parse_like(row,example):
    need(set(row)==set(example),'CSV schema');out={}
    for k,v in example.items():
        s=row[k]
        if v is None:need(s=='','CSV null');out[k]=None
        elif isinstance(v,bool):need(s in ('True','False'),'CSV bool');out[k]=s=='True'
        elif isinstance(v,(list,dict)):out[k]=json.loads(s)
        elif isinstance(v,int):out[k]=int(s)
        elif isinstance(v,float):out[k]=float(s)
        else:out[k]=s
    return out

def run(path,sha,appendix=None):
    cb=raw(path);need(hashlib.sha256(cb).hexdigest()==sha,'COMPLETE SHA');c=json.loads(cb);need(c['completed'] is True and c['PASS'] is True,'complete PASS')
    need(c['code']['sha256']=='01f0f81d1add4cada7791103b40fa8b0b3619ad3ce74ab825d3dfe0298a7b91e','sealed V2 code')
    admitted=[*c['inputs'].values(),*c['outputs'].values(),c['entry'],c['root_scope']];white={(b['path'],b['sha256']) for b in admitted};reads=[]
    def load(b):
        need((b['path'],b['sha256']) in white,'undeclared artifact');v=raw(b['path']);need(hashlib.sha256(v).hexdigest()==b['sha256'] and len(v)==b['size_bytes'],'input/output SHA size');reads.append(b);return v
    e=json.loads(load(c['entry']));scope=json.loads(load(c['root_scope']));contract=dict(e);contract.pop('root_scope')
    need(e['approved_SOURCE_oracle_reference_diagnostic'] is True and e['automatic_retry'] is False and scope['approved'] is True and scope['entry_contract']==contract and e['inputs']==c['inputs'],'exact Root entry/three-input scope')
    need(set(c['inputs'])=={'original_features','samples','YCB_metadata'},'three inputs')
    feature_bytes=load(c['inputs']['original_features']);features=np.load(io.BytesIO(feature_bytes),allow_pickle=False);del feature_bytes
    need(features.dtype==np.float32 and features.shape==(7617,1186),'known feature packet')
    with np.load(io.BytesIO(load(c['inputs']['samples'])),allow_pickle=False) as z:samples={k:z[k] for k in ('target','family','fit','available','row_id')}
    need(all(x.shape==(7617,) for x in samples.values()) and samples['target'].dtype==samples['family'].dtype==np.int8 and samples['fit'].dtype==samples['available'].dtype==bool and samples['row_id'].dtype.kind=='U','five nonpixel fields')
    reader=csv.DictReader(io.StringIO(load(c['inputs']['YCB_metadata']).decode(),newline=''));need(set(COLS)<=set(reader.fieldnames),'14 selected columns')
    selected=[{k:r[k] for k in COLS} for r in reader];rows=[]
    for r in selected:
        i=int(r['sample_index']);y=int(r['associated_r_state']=='E');need(r['associated_r_state'] in ('D','E') and r['role'] in ('FIT','DEV'),'associated-r/role')
        need(0<=i<7617 and samples['family'][i]==1 and int(samples['target'][i])==y and str(samples['row_id'][i])==r['row_id'] and bool(samples['fit'][i])==(r['role']=='FIT'),'row_id/sample_index/target/role join')
        q=[number(r[k]) for k in ('q_x','q_y')];ref=[number(r[k]) for k in ('r_x_GT_diagnostic_only','r_y_GT_diagnostic_only')];distance=None if None in q+ref else float(np.sqrt((q[0]-ref[0])**2+(q[1]-ref[1])**2))
        if distance is not None:need(number(r['q_reference_distance_px']) is not None and math.isclose(distance,number(r['q_reference_distance_px']),rel_tol=1e-6,abs_tol=1e-5),'saved q-r distance')
        rows.append({'row_id':r['row_id'],'sample_index':i,'role':r['role'],'scene':r['scene'],'frame_id':r['frame_id'],'object_id':int(r['object_id']),'logical_aux_slot':int(r['aux_slot']),'associated_r_state':r['associated_r_state'],'target':y,'available':bool(samples['available'][i]),'q_xy':q,'r_xy_GT_diagnostic_only':ref,'q_r_distance_px':distance,'L':number(r['L'])})
    rows.sort(key=lambda r:r['sample_index']);need(len(rows)==len({r['row_id'] for r in rows})==1521 and {r['sample_index'] for r in rows}==set(np.flatnonzero(samples['family']==1).tolist()),'all1521 identity closure')
    fit=[r for r in rows if r['role']=='FIT'];dev=[r for r in rows if r['role']=='DEV'];need((len(fit),len(dev),sum(r['target']==0 for r in fit),sum(r['target']==1 for r in fit),sum(r['target']==0 for r in dev),sum(r['target']==1 for r in dev))==(1124,397,841,283,340,57),'SOURCE census')
    objects=sorted({r['object_id'] for r in fit});wrong={o:objects[(j+1)%len(objects)] if len(objects)>1 else None for j,o in enumerate(objects)}
    with np.load(io.BytesIO(load(c['outputs']['PROTOTYPES.npz'])),allow_pickle=False) as z:
        need(set(z.files)=={'prototypes','supported','object_id','view'},'prototype schema');saved_p=z['prototypes'];saved_ok=z['supported'];need(z['object_id'].tolist()==objects and z['view'].tolist()==list(VIEWS),'prototype order')
    need(saved_p.dtype==np.float64 and saved_p.shape==(3,len(objects),384) and saved_ok.dtype==bool and saved_ok.shape==(3,len(objects)),'FP64 prototype shapes')
    support=[];scored=[];proto_differences=[];byteexact=True
    for vi,(view,(lo,hi)) in enumerate(VIEWS.items()):
        bank={}
        for oi,obj in enumerate(objects):
            declared=[r for r in fit if r['object_id']==obj and r['target']==0];candidates=[r for r in declared if r['available']];good=[];vectors=[]
            for r in candidates:
                v=np.array(features[r['sample_index'],lo:hi],dtype=np.float64)
                if np.isfinite(v).all() and np.linalg.norm(v)>0:good.append(r);vectors.append(v)
            p=None;reason='NO_FINITE_NONZERO_FIT_D_REFERENCE'
            if good:
                # Explicit ascending FP64 prefix sum, independently of producer code.
                mean=np.add.accumulate(np.vstack([np.zeros((1,384),np.float64),vectors]),axis=0)[-1]/len(good);n=float(np.linalg.norm(mean))
                if math.isfinite(n) and n>0:p=mean/n;reason=None
                else:reason='ZERO_OR_NONFINITE_PROTOTYPE_MEAN'
            need(saved_ok[vi,oi]==(p is not None),'prototype availability');bank[obj]=p
            if p is not None:
                delta=float(np.max(np.abs(p-saved_p[vi,oi])));need(delta<=2e-14,'independent FP64 FIT-D prototype');proto_differences.append(delta);byteexact &= p.tobytes()==saved_p[vi,oi].tobytes()
            else:need(np.count_nonzero(saved_p[vi,oi])==0,'unsupported prototype zero')
            support.append({'view':view,'object_id':obj,'fixed_other_object_id':wrong[obj],'FIT_D_total_rows':len(declared),'FIT_D_original_unavailable':len(declared)-len(candidates),'FIT_D_candidates_available':len(candidates),'FIT_D_descriptor_nonfinite_or_zero':len(candidates)-len(good),'FIT_D_rows_used':len(good),'distinct_scenes':len({r['scene'] for r in good}),'distinct_scene_frames':len({(r['scene'],r['frame_id']) for r in good}),'logical_aux_slot_counts':dict(Counter(str(r['logical_aux_slot']) for r in good)),'prototype_supported':p is not None,'unsupported_reason':reason,'prototype_FP64_sha256':None if p is None else hashlib.sha256(saved_p[vi,oi].tobytes()).hexdigest()})
        for r in dev:
            v=np.array(features[r['sample_index'],lo:hi],dtype=np.float64);n=float(np.linalg.norm(v)) if np.isfinite(v).all() else float('nan');supported=r['available'] and math.isfinite(n) and n>0;o=wrong.get(r['object_id']);p=bank.get(r['object_id']);other=bank.get(o);a=b=m=None;reasons=[]
            if not r['available']:reasons.append('ORIGINAL_INPUT_UNAVAILABLE')
            elif not supported:reasons.append('QUERY_DESCRIPTOR_NONFINITE_OR_ZERO')
            if p is None:reasons.append('NOMINAL_OBJECT_PROTOTYPE_UNSUPPORTED')
            if o is None:reasons.append('NO_FIXED_DIFFERENT_FIT_OBJECT')
            elif other is None:reasons.append('FIXED_OTHER_PROTOTYPE_UNSUPPORTED')
            if supported:
                unit=v/n
                if p is not None:a=float(unit@p)
                if other is not None:b=float(unit@other)
                if a is not None and b is not None:m=a-b
            scored.append(dict(r,view=view,fixed_other_object_id=o,query_descriptor_supported=supported,nominal_prototype_supported=p is not None,other_prototype_supported=other is not None,cosine_nominal_target=a,cosine_fixed_other_target=b,nominal_minus_other_margin=m,support_reasons=reasons,reference_selection_is_teacher_oracle=True,native_mapping='UNKNOWN'))
    for name,computed in [('DEV397_DESCRIPTOR_SCORES.csv',scored),('PROTOTYPE_SUPPORT.csv',support)]:
        saved=list(csv.DictReader(io.StringIO(load(c['outputs'][name]).decode(),newline='')));need(len(saved)==len(computed),'saved row counts')
        for i,(r,want) in enumerate(zip(saved,computed)):compare(want,parse_like(r,want),name+'/'+str(i))
    tables=[]
    for view in VIEWS:
        for stratum in STRATA:
            for obj in [None]+sorted({r['object_id'] for r in dev}):
                group=[r for r in scored if r['view']==view and in_stratum(r['q_r_distance_px'],stratum) and (obj is None or r['object_id']==obj)];d=[r for r in group if r['target']==0];e2=[r for r in group if r['target']==1];margins=[r['nominal_minus_other_margin'] for r in d if r['nominal_minus_other_margin'] is not None]
                distributions={key:{'D':stats(r[key] for r in d),'E':stats(r[key] for r in e2)} for key in ('cosine_nominal_target','cosine_fixed_other_target','nominal_minus_other_margin')}
                tables.append({'view':view,'q_r_stratum':stratum,'object_id':obj,'rows':len(group),'D':len(d),'E':len(e2),'query_supported':sum(r['query_descriptor_supported'] for r in group),'distributions':distributions,'E_AUROC_negative_nominal_cosine':rank_auc(group,'cosine_nominal_target'),'E_AUROC_negative_margin':rank_auc(group,'nominal_minus_other_margin'),'D_nominal_gt_other':{'supported':len(margins),'count':sum(x>0 for x in margins),'ties':sum(x==0 for x in margins),'fraction':sum(x>0 for x in margins)/len(margins) if margins else None}})
    stored_tables=json.loads(load(c['outputs']['DESCRIPTIVE_TABLES.json']));compare({'completed':True,'PASS':True,'tables':tables},stored_tables,'all distributions/rank-AUC/strata')
    summary=json.loads(load(c['outputs']['SUMMARY.json']));counts={'known_YCB':1521,'FIT':1124,'DEV':397,'FIT_D':841,'FIT_E':283,'DEV_D':340,'DEV_E':57,'descriptor_views':3,'DEV_view_rows':1191,'unknown_q_r_rows_DEV':sum(r['q_r_distance_px'] is None for r in dev)}
    compare(counts,c['counts'],'complete census');compare(counts,summary['counts'],'summary census');need(summary['nominal_FIT_objects']==objects and summary['fixed_wrong_object_cyclic_next']=={str(k):v for k,v in wrong.items()},'wrong target fixed mapping')
    compare({t['view']:t for t in tables if t['q_r_stratum']=='all' and t['object_id'] is None},summary['pooled_all'],'pooled summary')
    need(summary['no_best_view_or_threshold_selected'] is True and summary['reference_identity_is_oracle'] is True and summary['current_admission'] is False,'no selection/admission')
    portable=load(c['outputs']['PORTABLE_REPORT_KO.md']).decode();need('reference-bank statistics fitted on SOURCE FIT-D' in portable and 'No view is selected' in portable,'source fitting and no selection disclosed')
    need(c['SOURCE_derived_teacher_metadata_read'] is True and c['SOURCE_FIT_reference_statistics_computed'] is True and all(c[k]==0 for k in ('new_raw_RGB_depth_mesh_or_annotation_archive_reads','new_weights_or_model_forward','current_or_held_payloads','optimizer_updates')),'scope disclosure')
    result={'completed':True,'PASS':True,'diagnostic_complete':{'path':str(path),'sha256':sha,'size_bytes':len(cb)},'checked_bindings':reads,'counts':counts,'prototype_max_abs_difference':max(proto_differences,default=0.),'prototype_byteexact':bool(byteexact),'three_views_pooled':summary['pooled_all'],'DEV_object_class_counts':{str(o):{'D':sum(r['object_id']==o and r['target']==0 for r in dev),'E':sum(r['object_id']==o and r['target']==1 for r in dev)} for o in sorted({r['object_id'] for r in dev})},'all1191_scores_and14metadata_join_PASS':True,'all_tables_perobject_strata_distribution_rank_sum_AUC_PASS':True,'independent_method':'No producer import/call. FP64 ascending prefix sums and unit-L2, new cosine arithmetic, average-rank Mann-Whitney AUC with half ties instead of producer pairwise comparisons.','SOURCE_FIT_statistics_recomputed':True,'optimizer_model_forward_RGB_raw_GT_weights_CP_current_or_held_access':0,'current_admission':False,'model_data_architecture_threshold_choice':0,'limits':'Teacher identity oracle, source-fitted reference statistics, associated-r/nativeUNKNOWN, exposed/reused SOURCEDEV; AUC does not certify physical/native gold, causal target use or deployable reference selection.'}
    if appendix is not None:
        scope_b,results_b=appendix
        for bound in appendix:
            v=raw(bound['path']);need(hashlib.sha256(v).hexdigest()==bound['sha256'],'posthoc metadata pin');bound['size_bytes']=len(v);white.add((bound['path'],bound['sha256']))
        post_scope=json.loads(load(scope_b));saved_post=json.loads(load(results_b))
        need(post_scope['posthoc'] is True and post_scope['source_score_read_authorized'] is True and post_scope['current_admission'] is False,'separate posthoc SOURCE-only scope')
        white.update((v['path'],v['sha256']) for v in post_scope['SOURCE_baseline_inputs'].values());post_tables=[];devmap={r['sample_index']:r for r in dev}
        for arm,binding in post_scope['SOURCE_baseline_inputs'].items():
            all_rows=list(csv.DictReader(io.StringIO(load(binding).decode(),newline='')));need(len(all_rows)==7617,'saved main CSV census');joined=[]
            for r in all_rows:
                if int(r['family'])!=1 or r['role']!='DEV':continue
                i=int(r['sample_index']);meta=devmap[i];need(r['row_id']==meta['row_id'] and int(r['target'])==meta['target'] and r['available']=='True','same397 baseline scalar join')
                value=float(r['selected_family_logit']);need(math.isfinite(value) and r['prediction']==('E' if value>=0 else 'D'),'unchanged raw E-logit/zero decode')
                joined.append(dict(meta,raw_E_logit=value,prediction=r['prediction']))
            need(len(joined)==len({r['sample_index'] for r in joined})==397,'all397 baseline rows')
            for stratum in STRATA:
                for obj in [None]+sorted({r['object_id'] for r in dev}):
                    group=[r for r in joined if in_stratum(r['q_r_distance_px'],stratum) and (obj is None or r['object_id']==obj)]
                    cm=[[sum(r['target']==y and r['prediction']==p for r in group) for p in ('D','E')] for y in (0,1)]
                    post_tables.append({'arm':arm,'q_r_stratum':stratum,'object_id':obj,'D':sum(r['target']==0 for r in group),'E':sum(r['target']==1 for r in group),'raw_E_logit_AUROC':rank_auc(group,'raw_E_logit',False)['auc'],'CM_at_original_zero_logit':cm})
        compare({'completed':True,'PASS':True,'posthoc_SOURCE_only':True,'score_direction_fixed':'E positive raw selected_family_logit','threshold_changed':False,'model_choice':False,'current_admission':False,'tables':post_tables},saved_post,'separate posthoc raw rank')
        result['posthoc_baseline_rank_appendix']={'completed':True,'PASS':True,'posthoc':True,'scope':scope_b,'saved_results':results_b,'same397_row_join_perobject_allstrata_PASS':True,'independent_average_rank_AUC_and_original_zero_CM_PASS':True,'pooled_all_and_LE3':[x for x in post_tables if x['object_id'] is None and x['q_r_stratum'] in ('all','<=3px')],'original_prototype_recipe_unchanged':True,'old_METADATA_model_score_columns_used':False,'separate_V248_saved_SOURCE_CSV_scores_used':True,'threshold_or_model_selection':False}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--complete',required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',required=True);p.add_argument('--appendix-scope');p.add_argument('--appendix-scope-sha');p.add_argument('--appendix-results');p.add_argument('--appendix-results-sha');a=p.parse_args();out=Path(a.out).absolute();out.mkdir()
    try:
        appendix=None
        if a.appendix_scope:
            need(all([a.appendix_scope_sha,a.appendix_results,a.appendix_results_sha]),'exact posthoc metadata bindings required');appendix=({'path':str(Path(a.appendix_scope).absolute()),'sha256':a.appendix_scope_sha},{'path':str(Path(a.appendix_results).absolute()),'sha256':a.appendix_results_sha})
        result=run(Path(a.complete).absolute(),a.sha256,appendix);code=raw(Path(__file__).absolute());result['checker_code']={'path':str(Path(__file__).absolute()),'sha256':hashlib.sha256(code).hexdigest(),'size_bytes':len(code)}
        with (out/'REVIEW.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        b=raw(out/'REVIEW.json');print(json.dumps({'completed':True,'PASS':True,'REVIEW':{'path':str(out/'REVIEW.json'),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}}))
    except BaseException:
        with (out/'FAILED_EXECUTION.json').open('x') as f:json.dump({'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()},f,indent=2)
        raise
