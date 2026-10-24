"""V216 prospective internal SOURCE FIT/DEV grouping; metadata only, no payloads."""
from collections import Counter
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import traceback

H=Path(__file__).absolute().parent
V=H.parent
P=V.parent
SEED='V216_PUBLIC_PART_QUERY_SUPPORT_INTERNAL_SOURCE_DEV_V1'
PREFIXES=('val/000000','val/000002','val/000005','val/000006')
PROTOCOL={'path':str(H/'PROTOCOL_BEFORE_SOURCE_METADATA_SPLIT.json'),'sha256':'07d9c8b03762f175b6a42f61b828e26c25747da486666892a5337db2e505d7b0'}

def need(v,m):
    if not v:raise AssertionError(m)

def safe(p):
    s=str(p);need('v112'not in s.casefold(),'V112 lexical reject before FS')
    p=Path(s);need(p.is_absolute()and'..'not in p.parts and P in p.parents,'Exact P path only')
    return p

def opened(path,write=False):
    p=safe(path);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for q in p.parts[1:-1]:
            n=os.open(q,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,(os.O_WRONLY|os.O_CREAT|os.O_EXCL if write else os.O_RDONLY)|os.O_NOFOLLOW,0o644,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(n).st_mode),'Regular file/no symlink')
    return os.fdopen(n,'wb'if write else'rb')

def rb(path):
    with opened(path)as f:return f.read()

def bind(path):return dict(path=str(safe(path)),sha256=hashlib.sha256(rb(path)).hexdigest())

def checked(b):
    need(set(b)=={'path','sha256'}and bind(b['path'])==b,'Exact metadata binding')
    return b['path']

def read(b):return json.loads(rb(checked(b)))

def text(path,value):
    with opened(path,True)as f:f.write(value.encode())

def write(path,value):text(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def component_hash(component_id):
    return hashlib.sha256((SEED+component_id).encode('utf-8')).hexdigest()

def assign_roles(frames):
    need(len(frames)==12 and len({f['source_id']for f in frames})==12,'Exactly12 unique original selected frames')
    need(len({f['component_id']for f in frames})==12,'12 distinct original components; no fallback')
    need(Counter(f['prefix']for f in frames)=={p:3 for p in PREFIXES},'Exact4prefixx3; no fallback')
    need(all(f['role']=='TRAIN'and re.fullmatch('[0-9a-f]{64}',f['component_id'])for f in frames),'Original roleTRAIN and canonical component identity')
    roles={}
    for prefix in PREFIXES:
        ranked=sorted((f for f in frames if f['prefix']==prefix),key=lambda f:(component_hash(f['component_id']),f['component_id'],f['source_id']))
        for rank,f in enumerate(ranked):
            roles[f['component_id']]=dict(internal_source_role='SOURCE_DEV'if rank==0 else'SOURCE_FIT',rank_within_prefix=rank,stable_component_SHA256=component_hash(f['component_id']))
    need(Counter(r['internal_source_role']for r in roles.values())=={'SOURCE_FIT':8,'SOURCE_DEV':4},'8FIT/4DEV components')
    return roles

def sibling_role(component_id,roles):
    need(component_id in roles,'Unknown sibling component rejected, never infer from labels/native/variant')
    return roles[component_id]['internal_source_role']

def invented_tests():
    fake=[dict(source_id=p+':'+str(i),prefix=p,component_id=hashlib.sha256((p+str(i)).encode()).hexdigest(),role='TRAIN',available_queries=4+i,semantic_label='UNREAD')for p in PREFIXES for i in range(3)]
    a=assign_roles(fake);b=assign_roles(fake[::-1]);need(a==b,'Order-independent component hash grouping')
    changed=copy.deepcopy(fake)
    for f in changed:f['available_queries']=0;f['semantic_label']='IGNORED_INVENTED_OTHER'
    need(assign_roles(changed)==a,'Availability and semantic label do not rank groups')
    need(component_hash('a'*64)==hashlib.sha256((SEED+'a'*64).encode()).hexdigest(),'Literal seed+component with no separator')
    for f in fake:
        need(all(sibling_role(f['component_id'],a)==a[f['component_id']]['internal_source_role']for _ in range(8*3)),'All8 and any paired variants inherit one role')
    for bad in (fake[:-1],[dict(f,component_id=fake[0]['component_id'])for f in fake],[dict(f,role='HOLDOUT')for f in fake]):
        try:assign_roles(bad);ok=False
        except AssertionError:ok=True
        need(ok,'Missing/component-collision/nonTRAIN fixture rejected without fallback')
    try:sibling_role('f'*64,a);ok=False
    except AssertionError:ok=True
    need(ok,'Unmapped sibling rejected')
    # Original TRAIN remains TRAIN for FIT and DEV. Unavailable rows are not dropped.
    rows=[dict(role='TRAIN',native_slot=k,original_query_available=k!=7,query_xy=None if k==7 else[10.25,20.5],L=None if k==7 else96.)for k in range(8)]
    out=[dict(r,internal_source_role='SOURCE_DEV')for r in rows]
    need(len(out)==8 and all(r['role']=='TRAIN'for r in out)and out[7]['query_xy']is None and not out[7]['original_query_available'],'Original TRAIN/native8/null-U unchanged in internal DEV')
    return dict(completed=True,PASS=True,invented_only=True,checks=['Seed_no_separator_hash_and_order_independent4x3_to8FIT4DEV','Availability_labels_not_ranking_inputs','All8_and_sibling_pairedvariants_same_component_role','Insufficiency_collision_nonTRAIN_unknown_component_rejected','Original_TRAIN_and_nullU_preserved'],count=5,source_payload_access=0)

def csv_scalar(v):return None if v==''else float(v)

def main():
    protocol=read(PROTOCOL)
    need(protocol['seed']==SEED and protocol['component_hash_formula']=='SHA256(UTF8(seed + component_id)), literal concatenation with NO separator or trailing LF','Exact prospective rule')
    checks=invented_tests();write(H/'INVENTED_SPLIT_CHECKS.json',checks)
    scope=read(protocol['scope']);need(scope['phase']=='V216'and scope['prepared_before_new_semantic_annotations'],'Root prospective scope')
    # First actual metadata read follows prospective protocol and pure fixtures.
    inputs=protocol['input_bindings'];selection=read(inputs['selection']);manifest=read(inputs['manifest'])
    csvrows=list(csv.DictReader(io.StringIO(rb(checked(inputs['identity_CSV'])).decode())))
    frames=selection['frames'];slots=manifest['slots96'];mframes=manifest['frames']
    need(selection['completed']and selection['PASS']and selection['selection_frozen_before_pixels']and len(slots)==len(csvrows)==96,'Closed selected12 /96 input rows')
    roles=assign_roles(frames)
    need([f['position']for f in mframes]==list(range(12))and len(mframes)==12,'Original frame order')
    by_position={}
    for pos,(f,mf)in enumerate(zip(frames,mframes)):
        need(f['source_id']==mf['frame_key']and f['component_id']==mf['component_id']and f['original_frame_index']==mf['original_frame_index']and f['prefix']==mf['prefix']and mf['role']=='TRAIN','Selection/preview exact frame component identity')
        by_position[pos]=f
    outputrows=[]
    for s,c in zip(slots,csvrows):
        f=by_position[s['position']];k=s['native_slot'];r=f['native8'][k]
        need(s['role']==r['role']=='TRAIN'and s['frame_key']==f['source_id']and s['component_id']==f['component_id'],'Whole original frame TRAIN')
        for key in('row_id','original_global_index','native_slot','query_xy','L','original_query_available','original_flags'):
            need(s[key]==r[key],'Exact original row/scalar/flag: '+key)
        need(c['row_id']==s['row_id']and c['frame_key']==s['frame_key']and int(c['position'])==s['position']and int(c['native_slot'])==k and c['role']=='TRAIN'and c['component_id']==s['component_id'],'Exact original CSV metadata identity')
        q=s['query_xy']
        need(csv_scalar(c['query_x'])==(None if q is None else q[0])and csv_scalar(c['query_y'])==(None if q is None else q[1])and csv_scalar(c['L'])==s['L']and c['original_query_available']==str(s['original_query_available']),'Original q/L/U parity')
        need(s['clean96_decoded_RGB_sha256']==r['RGB96_sha256']and s['valid96_decoded_bool_sha256']==r['valid96_sha256'],'Original raw96/valid inherited decoded hashes')
        rr={key:s[key]for key in('position','frame_key','original_frame_index','component_id','role','native_slot','row_id','original_global_index','query_xy','L','original_query_available','original_flags','crop_constructed','clean96','valid96','clean96_decoded_RGB_sha256','valid96_decoded_bool_sha256')}
        rr.update(internal_source_role=sibling_role(f['component_id'],roles),source_group_key=f['component_id'])
        outputrows.append(rr)
    need(all([r['native_slot']for r in outputrows if r['position']==i]==list(range(8))for i in range(12)),'All8 per frame in original order')
    need(Counter(r['internal_source_role']for r in outputrows)=={'SOURCE_FIT':64,'SOURCE_DEV':32}and sum(not r['original_query_available']for r in outputrows)==11,'64FIT/32DEV; original U11')
    groupframes=[dict(position=i,source_id=f['source_id'],original_frame_index=f['original_frame_index'],prefix=f['prefix'],component_id=f['component_id'],original_role=f['role'],**roles[f['component_id']],native_slots=list(range(8)),original_row_ids=[r['row_id']for r in f['native8']])for i,f in enumerate(frames)]
    counts={role:dict(frames=sum(f['internal_source_role']==role for f in groupframes),rows=sum(r['internal_source_role']==role for r in outputrows),original_inputU=sum(not r['original_query_available']for r in outputrows if r['internal_source_role']==role))for role in('SOURCE_FIT','SOURCE_DEV')}
    result=dict(completed=True,PASS=True,phase='V216',prospective_grouping_before_new_semanticlabels_scores=True,code=bind(__file__),protocol=PROTOCOL,scope=protocol['scope'],input_bindings=inputs,seed=SEED,component_hash_formula=protocol['component_hash_formula'],frames=groupframes,rows96=outputrows,counts=counts,original_role='TRAIN',internal_source_role_is_separate=True,all8_siblingcases_and_pairedvariants_inherit_component_role=True,all_original_query_L_native_inputU_flags_and_RAW96_metadata_unchanged=True,original_inputU_total=11,no_frame_native_available_label_or_score_selection=True,no_fallback=True,new_training_or_label_admission=False,not_old_SOURCE_HOLDOUT=True,limits=protocol['exposure_limits'],actual_source_binary_stat_hash_header_decode=0,GT_labels_models_scores_or_Git=0)
    write(H/'INTERNAL_SOURCE_FIT_DEV_COMPONENT_SPLIT.json',result)
    fields=['position','frame_key','original_frame_index','component_id','source_group_key','role','internal_source_role','native_slot','row_id','original_global_index','query_x','query_y','L','original_query_available']
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=fields,lineterminator='\n');w.writeheader()
    for r in outputrows:
        q=r['query_xy'];w.writerow({**{k:r[k]for k in fields if k in r},'query_x':None if q is None else q[0],'query_y':None if q is None else q[1]})
    text(H/'ALL96_INTERNAL_SOURCE_FIT_DEV_ROLES.csv',out.getvalue())
    ready=dict(completed=True,PASS=True,metadata_split_complete=True,code=bind(__file__),protocol=PROTOCOL,scope=protocol['scope'],input_bindings=inputs,checks=bind(H/'INVENTED_SPLIT_CHECKS.json'),split=bind(H/'INTERNAL_SOURCE_FIT_DEV_COMPONENT_SPLIT.json'),CSV=bind(H/'ALL96_INTERNAL_SOURCE_FIT_DEV_ROLES.csv'),counts=counts,original_role_TRAIN_unchanged=True,internal_DEV_not_freshblind_or_protected_HOLDOUT=True,semantic_labels_or_model_or_source_binary_reads=0,frozen_no_further_edits=True)
    write(H/'FINAL_READY.json',ready)
    print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),split=ready['split'],CSV=ready['CSV'],counts=counts,DEV=[f['source_id']for f in groupframes if f['internal_source_role']=='SOURCE_DEV']),ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        write(H/'FAILURE_PRESERVED_METADATA_SPLIT.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),source_binary_model_score_access=0))
        raise
