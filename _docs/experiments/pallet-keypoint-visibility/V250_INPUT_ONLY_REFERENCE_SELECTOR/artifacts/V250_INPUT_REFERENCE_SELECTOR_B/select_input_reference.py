#!/usr/bin/env python3
"""V250 fixed input-only reference selection; Root-authorized CPU diagnostic only."""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np

ROOT_PROTOCOL_SHA='af9ddfbce63992575eac86f073e2418be08021df72cddcdd6716fda0720b921b'
ROOT_PROTOCOL_SIZE=9431
VIEWS=('full_CLS','fixed_centre_Q','centre_CLS')
STRATA=('all','<=1px','>1_to_3px','<=3px','>3px','unknown')
INPUT_KEYS=('ORIGINAL_ALL_FEATURES.npy','NODE_CLS.npy','SAMPLE_GROUP_INDEX.npy','SAMPLE_NODE_INDEX.npy','samples.npz','INPUT_GEOMETRY.npz','PROTOTYPES.npz','V249_DEV_ORACLE_SCORES.csv')
META_KEYS=('original_feature_complete','V249_oracle_complete')


def guard(path):
    p=Path(path)
    if not p.is_absolute() or '..' in p.parts:
        raise ValueError('Absolute path without traversal required')
    if any(('v112' in x.lower() or 'cache' in x.lower() or x.lower().endswith('.pyc')) for x in p.parts):
        raise ValueError('Forbidden path, rejected before filesystem access')
    for a in reversed([p,*p.parents]):
        if a.is_symlink(): raise ValueError('Symlink artifact or ancestor forbidden')
    return p


def binding(path):
    p=guard(path); h=hashlib.sha256(); size=0
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b);size+=len(b)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':size}


def checked(b):
    if set(b)!= {'path','sha256','size_bytes'}: raise ValueError('Exact binding schema required')
    if binding(b['path'])!=b: raise ValueError('Binding mismatch: '+b['path'])
    return guard(b['path'])


def read_json(path):
    return json.loads(guard(path).read_text(encoding='utf-8'))


def write_json(path,value):
    guard(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path,rows,fields=None):
    fields=fields or list(rows[0])
    with guard(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,separators=(',',':'),ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def unit(x):
    x=np.asarray(x,np.float64)
    if not np.isfinite(x).all(): return None
    norm=float(np.linalg.norm(x))
    return x/norm if np.isfinite(norm) and norm>0 else None


def select_reference_bank(node_cls8,original_available8,node_available8,active_slot,prototype_full_CLS4):
    """Exactly five input-only arguments. No target, role, object ID or teacher query."""
    x=np.asarray(node_cls8);oa=np.asarray(original_available8);na=np.asarray(node_available8);p=np.asarray(prototype_full_CLS4)
    if x.shape!=(8,384) or p.shape!=(4,384) or oa.shape!=(8,) or na.shape!=(8,): raise ValueError('Selector shape')
    if oa.dtype!=np.bool_ or na.dtype!=np.bool_ or not 0<=int(active_slot)<8: raise ValueError('Selector mask/slot')
    normalized=[unit(a) for a in x]
    mask=oa & na & np.asarray([a is not None for a in normalized],bool)
    mask[int(active_slot)]=False
    slots=np.flatnonzero(mask).tolist()
    out={'bank_index':None,'bank_scores':[None]*4,'top2_slots_by_bank':[[] for _ in range(4)],'valid_other_nodes':len(slots),'candidate_mask':mask.tolist(),'selector_gap':None,'status':'UNKNOWN'}
    pu=[unit(a) for a in p]
    if any(a is None for a in pu): out['reason']='INVALID_FIXED_BANK';return out
    if len(slots)<2: out['reason']='FEWER_THAN_TWO_OTHER_NODES';return out
    scores=[];top=[]
    for proto in pu:
        ranked=sorted(((float(np.dot(normalized[s],proto)),s) for s in slots),key=lambda z:(-z[0],z[1]))
        scores.append(float(np.mean([ranked[0][0],ranked[1][0]],dtype=np.float64)))
        top.append([ranked[0][1],ranked[1][1]])
    order=sorted(range(4),key=lambda b:(-scores[b],b))
    out.update(bank_index=order[0],bank_scores=scores,top2_slots_by_bank=top,selector_gap=float(scores[order[0]]-scores[order[1]]),status='SUPPORTED',reason=None)
    return out


def predict_input_only(features,node_cls,group_index,node_index,family,row_id,available,original_available,node_available,prototypes,prototype_supported):
    """No fit/target/object-ID/oracle arguments: all known YCB selected by family only."""
    ids=np.flatnonzero(family==1);selectors=[];scores=[]
    for i in ids:
        g=int(group_index[i]);a=int(node_index[i])
        if np.asarray(features[i,:384]).tobytes()!=np.asarray(node_cls[g,a]).tobytes(): raise ValueError('Active CLS byte-sharing failure')
        s=select_reference_bank(node_cls[g],original_available[g],node_available[g],a,prototypes[0])
        if not bool(prototype_supported[0].all()):
            s.update(bank_index=None,status='UNKNOWN',reason='UNSUPPORTED_FIXED_BANK',selector_gap=None)
        s={'sample_index':int(i),'row_id':str(row_id[i]),'group_index':g,'active_slot':a,'original_query_available':bool(available[i]),**s}
        selectors.append(s)
        for vi,view in enumerate(VIEWS):
            b=s['bank_index'];other=None if b is None else (b+1)%4
            q=unit(features[i,vi*384:(vi+1)*384]);reason=[]
            if b is None: reason.append('SELECTOR_UNKNOWN')
            if not bool(available[i]): reason.append('ORIGINAL_QUERY_UNAVAILABLE')
            if q is None: reason.append('QUERY_DESCRIPTOR_UNSUPPORTED')
            nom=None if b is None else unit(prototypes[vi,b]);alt=None if other is None else unit(prototypes[vi,other])
            if b is not None and (not prototype_supported[vi,b] or nom is None):reason.append('SELECTED_PROTOTYPE_UNSUPPORTED')
            if other is not None and (not prototype_supported[vi,other] or alt is None):reason.append('NEXT_PROTOTYPE_UNSUPPORTED')
            cs=co=margin=None
            if not reason:
                cs=float(np.dot(q,nom));co=float(np.dot(q,alt));margin=cs-co
            scores.append({'sample_index':int(i),'row_id':str(row_id[i]),'view':view,'selected_bank_index':b,'fixed_next_bank_index':other,'selector_supported':b is not None,'query_descriptor_supported':q is not None,'original_query_available':bool(available[i]),'supported':not reason,'cosine_selected_bank':cs,'cosine_next_bank':co,'selected_minus_next_margin':margin,'support_reasons':reason})
    return selectors,scores


def summarize(values):
    a=np.asarray([v for v in values if v is not None],np.float64)
    if not len(a): return {'n':0,'mean':None,'quantiles':None}
    if not np.isfinite(a).all(): raise ValueError('Finite scores only')
    return {'n':len(a),'mean':float(a.mean()),'quantiles':dict(zip(('min','p05','p25','median','p75','p95','max'),map(float,np.quantile(a,[0,.05,.25,.5,.75,.95,1]))))}


def auc_E(labels,scores):
    pairs=[(int(y),float(s)) for y,s in zip(labels,scores) if s is not None]
    pos=np.asarray([s for y,s in pairs if y==1],np.float64);neg=np.asarray([s for y,s in pairs if y==0],np.float64)
    if not len(pos) or not len(neg): return {'auc':None,'D_supported':len(neg),'E_supported':len(pos),'reason':'BOTH_CLASSES_REQUIRED'}
    value=sum(float((p>neg).sum())+.5*float((p==neg).sum()) for p in pos)/(len(pos)*len(neg))
    return {'auc':value,'D_supported':len(neg),'E_supported':len(pos),'reason':None,'ties':'half'}


def belongs(row,stratum):
    d=row['q_r_distance_px']
    return (stratum=='all' or (stratum=='unknown' and d is None) or
            (d is not None and ((stratum=='<=1px' and d<=1) or (stratum=='>1_to_3px' and 1<d<=3) or
                                (stratum=='<=3px' and d<=3) or (stratum=='>3px' and d>3))))


def number(s):
    if s is None or s=='': return None
    f=float(s)
    if not np.isfinite(f): raise ValueError('Nonfinite saved score')
    return f


def boolean(s):
    if s not in ('True','False'):raise ValueError('Saved boolean encoding')
    return s=='True'


def selection_table(rows):
    # Count each query once: caller supplies one view or de-duplicates it.
    supported=[r for r in rows if r['selected_bank_index'] is not None]
    return {'rows':len(rows),'D':sum(r['target']==0 for r in rows),'E':sum(r['target']==1 for r in rows),'selector_supported':len(supported),'selector_UNKNOWN':len(rows)-len(supported),'selector_UNKNOWN_rate':(len(rows)-len(supported))/len(rows) if rows else None,'correct':sum(r['selected_bank_index']==r['nominal_bank_index'] for r in supported),'accuracy_supported':sum(r['selected_bank_index']==r['nominal_bank_index'] for r in supported)/len(supported) if supported else None}


def metric_table(rows):
    result={'rows':len(rows),'D':sum(r['target']==0 for r in rows),'E':sum(r['target']==1 for r in rows),'query_UNKNOWN':sum(not r['auto_supported'] for r in rows)}
    for name,ac,oc in [('negative_cosine','cosine_selected_bank','oracle_cosine_nominal'),('negative_margin','selected_minus_next_margin','oracle_margin')]:
        auto=[r for r in rows if r[ac] is not None]
        same=[r for r in auto if r[oc] is not None]
        full=[r for r in rows if r[oc] is not None]
        result[name]={'auto':auc_E([r['target'] for r in auto],[-r[ac] for r in auto]),'same_support_auto':auc_E([r['target'] for r in same],[-r[ac] for r in same]),'same_support_oracle':auc_E([r['target'] for r in same],[-r[oc] for r in same]),'full_oracle':auc_E([r['target'] for r in full],[-r[oc] for r in full]),'same_support_rows':len(same),'original_denominator':len(rows)}
    result['distributions']={k:{state:summarize([r[k] for r in rows if r['target']==y]) for state,y in [('D',0),('E',1)]} for k in ['cosine_selected_bank','cosine_next_bank','selected_minus_next_margin','oracle_cosine_nominal','oracle_margin']}
    result['selection']=selection_table(rows)
    return result


def evaluate_frozen(selectors,scores,target,fit,available,row_id,object_ids,oracle_rows):
    if object_ids.dtype!=np.int64 or object_ids.shape!=(4,) or list(map(int,object_ids))!=sorted(set(map(int,object_ids))):raise ValueError('Frozen numeric bank order')
    banks={int(v):i for i,v in enumerate(object_ids)};sel={r['sample_index']:r for r in selectors};pred={(r['sample_index'],r['view']):r for r in scores}
    old={};out=[];nominal_check=0
    for r in oracle_rows:
        i=int(r['sample_index']);v=r['view'];key=(i,v)
        if key in old: raise ValueError('Duplicate oldoracle key')
        if i not in sel or v not in VIEWS or str(row_id[i])!=r['row_id'] or bool(fit[i]) or r['role']!='DEV' or int(target[i])!=int(r['target']) or bool(available[i])!=boolean(r['available']):raise ValueError('Late oracle identity/role mismatch')
        if r['associated_r_state']!=('D' if int(target[i])==0 else 'E'):raise ValueError('Teacher state mismatch')
        old[key]=r
    ids=sorted({i for i,v in old})
    expected=sorted(i for i in sel if not fit[i])
    if ids!=expected or len(ids)!=397 or len(old)!=1191 or sum(target[i]==0 for i in ids)!=340 or sum(target[i]==1 for i in ids)!=57:raise ValueError('Original DEV397/1191 denominator')
    if set(old)!={(i,v) for i in ids for v in VIEWS}:raise ValueError('Missing fixed view')
    for i in ids:
        for view in VIEWS:
            r=old[(i,view)];p=pred[(i,view)];obj=int(r['object_id']);b=banks[obj]
            if int(r['fixed_other_object_id'])!=int(object_ids[(b+1)%4]):raise ValueError('Old nextcyclic map')
            cs=number(r['cosine_nominal_target']);co=number(r['cosine_fixed_other_target']);om=number(r['nominal_minus_other_margin'])
            if p['selected_bank_index']==b and p['supported']:
                for a,z in [(p['cosine_selected_bank'],cs),(p['cosine_next_bank'],co),(p['selected_minus_next_margin'],om)]:
                    if z is None or abs(a-z)>1e-12:raise ValueError('Correct-selected bank must replay oldoracle abs1e-12')
                nominal_check+=1
            out.append({'sample_index':i,'row_id':str(row_id[i]),'view':view,'role':'DEV','target':int(target[i]),'associated_r_state':r['associated_r_state'],'object_id':obj,'scene':r['scene'],'frame_id':r['frame_id'],'logical_aux_slot':int(r['logical_aux_slot']),'q_r_distance_px':number(r['q_r_distance_px']),'nominal_bank_index':b,'selected_bank_index':p['selected_bank_index'],'selected_object_id':None if p['selected_bank_index'] is None else int(object_ids[p['selected_bank_index']]),'selection_correct':None if p['selected_bank_index'] is None else p['selected_bank_index']==b,'auto_supported':p['supported'],'cosine_selected_bank':p['cosine_selected_bank'],'cosine_next_bank':p['cosine_next_bank'],'selected_minus_next_margin':p['selected_minus_next_margin'],'oracle_cosine_nominal':cs,'oracle_cosine_other':co,'oracle_margin':om,'support_reasons':p['support_reasons']})
    tables=[]
    for view in VIEWS:
        vr=[r for r in out if r['view']==view]
        for st in STRATA:
            sr=[r for r in vr if belongs(r,st)]
            for obj in [None,*map(int,object_ids)]:
                rr=sr if obj is None else [r for r in sr if r['object_id']==obj]
                tables.append({'view':view,'q_r_stratum':st,'object_id':obj,**metric_table(rr)})
    unique=[r for r in out if r['view']==VIEWS[0]]
    selection_by_state={state:selection_table([r for r in unique if r['target']==y]) for state,y in [('D',0),('E',1)]}
    summary={'counts':{'YCB_prediction_queries':len(selectors),'input_only_view_scores':len(scores),'DEV_queries':len(ids),'DEV_D':340,'DEV_E':57,'DEV_view_rows':len(out),'correctbank_oracle_numeric_checks':nominal_check},'selection_pooled':selection_table(unique),'selection_by_state':selection_by_state,'pooled_all_by_view':{v:next(t for t in tables if t['view']==v and t['q_r_stratum']=='all' and t['object_id'] is None) for v in VIEWS},'decision':'DESCRIPTIVE_SOURCE_DIAGNOSTIC_NO_MODEL_OR_THRESHOLD_SELECTION','current_admission':False}
    return out,tables,summary


def shape(a,dtype,expected):
    if a.dtype!=np.dtype(dtype) or a.shape!=expected:raise ValueError('dtype/shape mismatch '+str((a.dtype,a.shape,dtype,expected)))


def authorize(entry_path,protocol_path,output_dir):
    e=read_json(entry_path)
    if e.get('approved_SOURCE_input_reference_selector') is not True or e.get('automatic_retry') is not False:raise ValueError('Root true entry required; false entry refused before payload access')
    if checked(e['code'])!=guard(__file__):raise ValueError('Self-code binding')
    original=read_json(checked(e['original_protocol']))
    if e['original_protocol']['sha256']!=ROOT_PROTOCOL_SHA or e['original_protocol']['size_bytes']!=ROOT_PROTOCOL_SIZE:raise ValueError('Original prospective protocol pin')
    if checked(e['root_protocol'])!=guard(protocol_path):raise ValueError('CLI protocol binding')
    pr=read_json(protocol_path);expected=copy.deepcopy(original)
    if set(pr['inputs'])!=set(INPUT_KEYS) or set(pr['metadata_provenance'])!=set(META_KEYS):raise ValueError('Exact closed inputs')
    for section in ('inputs','metadata_provenance'):
        for key,b in pr[section].items():
            if set(b)!=set(expected[section][key]) or {k:v for k,v in b.items() if k!='path'}!={k:v for k,v in expected[section][key].items() if k!='path'}:raise ValueError('Only binding.path relocation allowed')
            expected[section][key]['path']=b['path']
    if expected!=pr or e['inputs']!=pr['inputs'] or e['metadata_provenance']!=pr['metadata_provenance']:raise ValueError('Path-only protocol relocation')
    checked(e['implementation_protocol'])
    scope=read_json(checked(e['root_scope']))
    if scope.get('approved') is not True or scope.get('entry_contract')!={k:v for k,v in e.items() if k!='root_scope'}:raise ValueError('Exact Root scope required')
    out=guard(output_dir)
    if str(out)!=e['output_directory'] or out.exists():raise ValueError('Exact fresh output directory required')
    paths={k:checked(b) for k,b in pr['inputs'].items()}
    for b in pr['metadata_provenance'].values():checked(b) # opaque integrity only; no JSON follow
    return e,pr,paths,out


def run(entry_path,protocol_path,output_dir):
    e,pr,p,out=authorize(entry_path,protocol_path,output_dir)
    f=np.load(p['ORIGINAL_ALL_FEATURES.npy'],mmap_mode='r',allow_pickle=False);n=np.load(p['NODE_CLS.npy'],mmap_mode='r',allow_pickle=False)
    gi=np.load(p['SAMPLE_GROUP_INDEX.npy'],mmap_mode='r',allow_pickle=False);ni=np.load(p['SAMPLE_NODE_INDEX.npy'],mmap_mode='r',allow_pickle=False)
    shape(f,'float32',(7617,1186));shape(n,'float32',(1767,8,384));shape(gi,'int64',(7617,));shape(ni,'int64',(7617,))
    if not ((gi>=0)&(gi<1767)&(ni>=0)&(ni<8)).all():raise ValueError('Original map bounds')
    with np.load(p['samples.npz'],allow_pickle=False) as z:family=z['family'];row_id=z['row_id'];available=z['available']
    shape(family,'int8',(7617,));shape(available,'bool',(7617,))
    if row_id.shape!=(7617,) or row_id.dtype.kind!='U' or len(set(map(str,row_id)))!=7617:raise ValueError('Original unique Unicode row IDs')
    with np.load(p['INPUT_GEOMETRY.npz'],allow_pickle=False) as z:oa=z['original_available8'];na=z['node_available']
    shape(oa,'bool',(1767,8));shape(na,'bool',(1767,8))
    if (na&~oa).any():raise ValueError('Node support must retain original availability')
    with np.load(p['PROTOTYPES.npz'],allow_pickle=False) as z:proto=z['prototypes'];ps=z['supported'];views=z['view']
    shape(proto,'float64',(3,4,384));shape(ps,'bool',(3,4))
    if views.dtype.kind!='U' or views.shape!=(3,) or tuple(map(str,views))!=VIEWS:raise ValueError('Fixed descriptor view order')
    if int((family==1).sum())!=1521:raise ValueError('Original known YCB1521')
    selectors,scores=predict_input_only(f,n,gi,ni,family,row_id,available,oa,na,proto,ps)
    if len(selectors)!=1521 or len(scores)!=4563:raise ValueError('Prediction population')
    out.mkdir(parents=False,exist_ok=False)
    sp=out/'SELECTOR1521_INPUT_ONLY.jsonl';qp=out/'QUERY4563_INPUT_ONLY_SCORES.csv'
    with sp.open('w',encoding='utf-8') as dest:
        for r in selectors:dest.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    write_csv(qp,scores)
    prefrozen={'selector':binding(sp),'query_scores':binding(qp)}
    freeze=out/'PREDICTIONS_FROZEN_BEFORE_TARGET_ROLE_ID_OR_ORACLE_READ.json'
    write_json(freeze,{'completed':True,'PASS':True,'prediction_population':1521,'query_view_rows':4563,'outputs':prefrozen,'target_fit_object_id_oracle_semantic_decodes_so_far':0,'selection_API_arguments':['node_cls8','original_available8','node_available8','active_slot','prototype_full_CLS4'],'bank_statistics':'FROZEN_V249_SOURCE_FIT_D_SUPERVISED_ASSET_NOT_REFITTED'})
    freeze_binding=binding(freeze)
    # HARD BOUNDARY: these members / CSV are not decoded anywhere above this line.
    with np.load(p['samples.npz'],allow_pickle=False) as z:target=z['target'];fit=z['fit']
    with np.load(p['PROTOTYPES.npz'],allow_pickle=False) as z:objects=z['object_id']
    shape(target,'int8',(7617,));shape(fit,'bool',(7617,))
    with p['V249_DEV_ORACLE_SCORES.csv'].open(newline='',encoding='utf-8') as src:oracle=list(csv.DictReader(src))
    rows,tables,summary=evaluate_frozen(selectors,scores,target,fit,available,row_id,objects,oracle)
    write_csv(out/'DEV1191_AUTO_ORACLE_COMPARISON.csv',rows)
    write_json(out/'DESCRIPTIVE_TABLES.json',{'tables':tables,'fixed_views':list(VIEWS),'fixed_q_r_strata':list(STRATA)})
    write_json(out/'SUMMARY.json',summary)
    text='# V250 SOURCE 입력 기반 reference 선택 진단\n\n동결된 V249 SOURCE FIT-D bank를 재사용했습니다. 새 optimizer/encoder/head 학습이나 reference-bank 재적합은 없습니다. 입력의 다른 q8 full-CLS 노드만으로 bank를 선택하고, 정답·역할·명목 object ID·기존 oracle 점수는 1521개 선택과 4563개 점수 봉인 뒤 읽었습니다.\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n모든 세 descriptor view와 사전 고정 strata를 보고합니다. closed four-bank 명목 object 선택 정확도와 SOURCE visibility 순위 진단이며 native semantic 정확도·독립 일반화·실사 current 개선을 뜻하지 않습니다. q는 GT-jitter emulation, bank는 SOURCE 감독 자산, DEV는 반복 노출 자료입니다. UNKNOWN은 원래 분모에서 보존하며 E로 바꾸지 않습니다. threshold·best-view·모델 선택 및 current 채택은 없습니다.\n'
    guard(out/'PORTABLE_REPORT_KO.md').write_text(text,encoding='utf-8')
    for b in prefrozen.values():checked(b)
    checked(freeze_binding)
    names=['SELECTOR1521_INPUT_ONLY.jsonl','QUERY4563_INPUT_ONLY_SCORES.csv','PREDICTIONS_FROZEN_BEFORE_TARGET_ROLE_ID_OR_ORACLE_READ.json','DEV1191_AUTO_ORACLE_COMPARISON.csv','DESCRIPTIVE_TABLES.json','SUMMARY.json','PORTABLE_REPORT_KO.md']
    complete={'completed':True,'PASS':True,'scientific_status':'DESCRIPTIVE_ONLY_NO_ADOPTION','entry':binding(entry_path),'code':binding(__file__),'root_protocol':e['root_protocol'],'original_protocol':e['original_protocol'],'inputs':pr['inputs'],'metadata_provenance':pr['metadata_provenance'],'outputs':{name:binding(out/name) for name in names},'prediction_freeze':freeze_binding,'counts':summary['counts'],'new_reference_fit':0,'SOURCE_FIT_supervised_bank_reused':True,'SOURCE_derived_teacher_metadata_read_after_prediction_freeze':True,'descriptive_statistics_computed':True,'new_raw_RGB_depth_mesh_annotation_archive_or_weights_reads':0,'new_encoder_or_head_forward':0,'optimizer_updates':0,'current_or_held_payloads':0,'original_state_role_q_L_native_U_changes':0,'current_admission':False,'access_counters':{'opaque_input_bindings_verified':8,'opaque_metadata_provenance_bindings_verified':2,'samples_members_before_freeze':['family','row_id','available'],'samples_members_after_freeze':['target','fit'],'geometry_members':['original_available8','node_available'],'prototype_members_before_freeze':['prototypes','supported','view'],'prototype_members_after_freeze':['object_id'],'oldoracle_CSV_semantic_read_after_freeze':True,'all_known_SOURCE_nonpixel_metadata_before_freeze':7617,'YCB_prediction_population':1521}}
    write_json(out/'COMPLETE.json',complete)
    print(json.dumps({'completed':True,'PASS':True,'complete':binding(out/'COMPLETE.json'),'counts':summary['counts']},ensure_ascii=False))


def main():
    a=argparse.ArgumentParser();a.add_argument('--protocol',required=True);a.add_argument('--entry',required=True);a.add_argument('--output-dir',required=True);q=a.parse_args();run(q.entry,q.protocol,q.output_dir)


if __name__=='__main__':main()
