#!/usr/bin/env python3
"""Future root-entry reader: opaque identities first, selected TRAIN JSON values only."""
import argparse
import hashlib
import json
import os
import stat

D='/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_fixed_public_train_instance_part_GT_correspondence_20261025_v217'
H=D+'/fixed12_GT_payload_preparation_a'
OUT=D+'/fixed12_selected_GT_metadata_v1'
JSON_KINDS={'SCENE_GT_JSON','SCENE_GT_INFO_JSON','SCENE_CAMERA_JSON'}

def require(ok,message):
    if not ok:raise ValueError(message)

def opened(path,flags=os.O_RDONLY):
    require(isinstance(path,str) and 'v112' not in path.lower(),'Forbidden lexical path before FS')
    require(os.path.isabs(path) and os.path.normpath(path)==path,'Normalized absolute path required')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for c in path.strip('/').split('/')[:-1]:
            n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        r=os.open(path.rsplit('/',1)[1],flags|os.O_NOFOLLOW,0o644,dir_fd=fd)
        if not stat.S_ISREG(os.fstat(r).st_mode):os.close(r);raise ValueError('Regular file required')
        return r
    finally:os.close(fd)

def read_bytes(path,maxbytes=67108864):
    with os.fdopen(opened(path),'rb') as f:b=f.read(maxbytes+1)
    require(len(b)<=maxbytes,'Byte cap exceeded')
    return b

def digest(b):return hashlib.sha256(b).hexdigest()
def bind(path,b):return {'path':path,'sha256':digest(b),'size_bytes':len(b)}
def checked(b):
    require(set(b)>={'path','sha256'},'Missing binding identity')
    raw=read_bytes(b['path']);require(digest(raw)==b['sha256'],'Text SHA mismatch')
    return raw

def write_exclusive(path,obj):
    b=(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with os.fdopen(opened(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'wb') as f:f.write(b)
    return bind(path,b)

def make_output_directory(path):
    require(path==OUT and 'v112' not in path.lower(),'Fixed output path only')
    parent=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for c in D.strip('/').split('/'):
            n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent);os.close(parent);parent=n
        os.mkdir(path.rsplit('/',1)[1],dir_fd=parent)
    finally:os.close(parent)

def skip_ws(s,i):
    while i<len(s) and s[i] in ' \t\r\n':i+=1
    return i

def string_end(s,i):
    require(i<len(s) and s[i]=='"','String expected')
    i+=1
    while i<len(s):
        if s[i]=='\\':i+=2;continue
        if s[i]=='"':return i+1
        require(ord(s[i])>=32,'Unescaped control character')
        i+=1
    raise ValueError('Unterminated string')

def opaque_value_end(s,i):
    """Find structure boundary without JSON-decoding or interpreting the value."""
    require(i<len(s),'Missing value')
    if s[i]=='"':return string_end(s,i)
    if s[i] in '[{':
        stack=[s[i]];i+=1
        while i<len(s):
            c=s[i]
            if c=='"':i=string_end(s,i);continue
            if c in '[{':stack.append(c)
            elif c in ']}':
                require(stack and (stack[-1],c) in (('[',']'),('{','}')),'Unbalanced structure')
                stack.pop()
                if not stack:return i+1
            i+=1
        raise ValueError('Unterminated container')
    j=i
    while j<len(s) and s[j] not in ',}':j+=1
    require(bool(s[i:j].strip()),'Missing primitive value')
    return j

def select_top_level_values(text,allowed_keys,decode_value=json.loads):
    """Only allowlisted value slices reach a JSON value decoder; other values stay opaque."""
    require(isinstance(text,str) and isinstance(allowed_keys,set) and all(isinstance(k,str) for k in allowed_keys),'Explicit string key allowlist required')
    i=skip_ws(text,0);require(i<len(text) and text[i]=='{','Top-level object required');i+=1
    result={};seen=set();skipped=0
    i=skip_ws(text,i)
    if i<len(text) and text[i]=='}':i+=1
    else:
        while True:
            i=skip_ws(text,i);end=string_end(text,i)
            key=json.loads(text[i:end]);require(key not in seen,'Duplicate top-level key');seen.add(key)
            i=skip_ws(text,end);require(i<len(text) and text[i]==':','Colon required')
            i=skip_ws(text,i+1);end=opaque_value_end(text,i)
            if key in allowed_keys:result[key]=decode_value(text[i:end])
            else:skipped+=1
            i=skip_ws(text,end);require(i<len(text),'Missing object terminator')
            if text[i]=='}':i+=1;break
            require(text[i]==',','Comma required');i=skip_ws(text,i+1)
            require(i<len(text) and text[i]!='}','Trailing comma forbidden')
    require(skip_ws(text,i)==len(text),'Trailing content')
    require(set(result)==allowed_keys,'Selected frame key missing')
    return result,{'selected_value_count':len(result),'skipped_value_count':skipped,'skipped_values_JSON_decoded':0,'whole_file_semantic_JSON_validation':False}

def validate_all8_inputs(rows):
    require(len(rows)==96,'All96 required')
    groups={};ids=set()
    for r in rows:
        require(r['row_id'] not in ids,'Duplicate row identity');ids.add(r['row_id'])
        require(r['original_role']=='TRAIN','Outer TRAIN only')
        require(r['internal_source_role'] in ('SOURCE_FIT','SOURCE_DEV'),'Explicit internal role required')
        require(type(r['native_slot']) is int and 0<=r['native_slot']<8,'Native integer0..7')
        require(type(r['original_query_available']) is bool,'Original availability required')
        groups.setdefault(r['frame_position'],[]).append(r)
    require(len(groups)==12,'Exactly fixed12 groups')
    for rs in groups.values():
        require(sorted(r['native_slot'] for r in rs)==list(range(8)),'Exact all8 group membership')
        require(len({r['frame_key'] for r in rs})==1 and len({r['component_id'] for r in rs})==1,'Group identity mismatch')
        require(len({r['internal_source_role'] for r in rs})==1,'Mixed group roles')
    require(sum(not r['original_query_available'] for r in rows)==11,'Original U11 preserved')
    return groups

def gate_payload(record,declared,authorize_first_opaque):
    """Identity and permission checks happen before any payload FS call."""
    path=record['path'];require('v112' not in path.lower(),'Forbidden path before FS')
    require(path in declared,'Undeclared payload')
    require(record==declared[path],'Exact declared record required')
    require(record['kind'] in JSON_KINDS,'No image/depth/mask/CAD/NPZ decode in this reader')
    if record['sha256'] is None:require(path in authorize_first_opaque,'Missing prior SHA needs explicit first opaque intake authorization')
    else:require(len(record['sha256'])==64,'Expected raw SHA256 required')

def requested_records(payloads,kinds):
    require(isinstance(kinds,list) and len(kinds)==len(set(kinds)) and set(kinds)<=JSON_KINDS and 'SCENE_GT_JSON' in kinds,'Explicit JSON dependency branch including scene_gt required')
    result=[r for r in payloads if r['kind'] in kinds]
    require(len(result)==4*len(kinds),'Four exact SOURCE scenes per requested kind')
    return result

def run(entry_path):
    require(entry_path.startswith(D+'/') and 'v112' not in entry_path.lower(),'Entry inside exact new phase')
    entry=json.loads(read_bytes(entry_path))
    require(entry.get('approved_actual_selected_GT_JSON_read') is True,'Root actual entry required')
    require(entry.get('training_admission') is False,'No training admission')
    require(entry.get('output_directory')==OUT,'Fresh fixed output')
    require(entry['code']['path']==os.path.abspath(__file__),'Exact frozen reader')
    require(entry['protocol']['path']==H+'/PROTOCOL_BEFORE_ANY_PUBLIC_GT_PAYLOAD.json','Exact protocol path')
    require(entry['payload_whitelist']['path']==H+'/EXACT_PUBLIC_GT_PAYLOAD_WHITELIST.json','Exact whitelist path')
    require(entry['fixed12_frames']['path']==H+'/FIXED12_TRAIN_FRAME_AND_INSTANCE_CANDIDATE_METADATA.json','Exact frame selection path')
    require(entry['original96_inputs']['path']==H+'/ALL96_ORIGINAL_INPUTS_AND_GT_POINTERS.json','Exact original input metadata path')
    checked(entry['code']);checked(entry['protocol'])
    wl=json.loads(checked(entry['payload_whitelist']))
    fm=json.loads(checked(entry['fixed12_frames']))
    rows=json.loads(checked(entry['original96_inputs']))['rows96']
    validate_all8_inputs(rows)
    records=requested_records(wl['payloads'],entry['requested_scene_JSON_kinds'])
    declared={r['path']:r for r in records}
    require(len(declared)==len(records),'Unique requested paths')
    missing={r['path'] for r in records if r['sha256'] is None}
    first=set(entry.get('authorize_first_opaque_SHA_intake_exact_paths',[]))
    require(first==missing,'Exactly the predeclared missing-SHA paths; no auto discovery')
    held={tuple(x) for x in entry.get('forbidden_frame_keys',[])}
    allowed_by_scene={}
    for f in fm['frames12']:
        require(f['original_role']=='TRAIN','TRAIN before selected GT parsing')
        key=f['json_frame_key'];require((f['scene'],key) not in held,'Forbidden frame')
        allowed_by_scene.setdefault(f['scene'],set()).add(key)
    opaque={};bodies={}
    for r in records:
        gate_payload(r,declared,first)
        b=read_bytes(r['path']);actual=bind(r['path'],b)
        if r['sha256'] is not None:require(actual['sha256']==r['sha256'],'Known GT JSON SHA mismatch')
        if r.get('size_bytes') is not None:require(actual['size_bytes']==r['size_bytes'],'Recorded byte size mismatch')
        opaque[r['path']]={**actual,'prior_SHA_status':r['sha_status']};bodies[r['path']]=b
    make_output_directory(OUT)
    opaque_receipt=write_exclusive(OUT+'/ACTUAL_OPAQUE_BINDINGS_BEFORE_SELECTED_VALUES.json',{'completed':True,'PASS':True,'entry':bind(entry_path,read_bytes(entry_path)),'bindings':opaque,'GT_values_parsed_before_this_receipt':0,'first_opaque_intake_paths':sorted(first)})
    selected={};parse_stats={}
    for r in records:
        values,stats=select_top_level_values(bodies[r['path']].decode('utf-8'),allowed_by_scene[r['scene']])
        selected[(r['scene'],r['kind'])]=values;parse_stats[r['path']]=stats
    output_frames=[]
    for f in fm['frames12']:
        scene,key=f['scene'],f['json_frame_key']
        gt=selected[(scene,'SCENE_GT_JSON')][key]
        info=selected.get((scene,'SCENE_GT_INFO_JSON'),{}).get(key)
        cam=selected.get((scene,'SCENE_CAMERA_JSON'),{}).get(key)
        require(isinstance(gt,list),'Expected BOP selected-frame GT list')
        if info is not None:require(isinstance(info,list) and len(gt)==len(info),'Instance GT/info alignment')
        if cam is not None:require(isinstance(cam,dict),'Selected camera object required')
        candidates=f['obj1_candidate_gt_list_indices']
        actual=[i for i,g in enumerate(gt) if g.get('obj_id')==1]
        require(actual==candidates,'Candidate indices differ from saved source selection')
        output_frames.append({'frame_position':f['frame_position'],'frame_key':f['frame_key'],'original_role':'TRAIN','internal_source_role':f['internal_source_role'],'obj1_candidates':[{'gt_list_index':i,'GT':gt[i],'GT_info':info[i] if info is not None else None,'masks':f['candidate_masks'][str(i)]} for i in candidates],'camera':cam,'unrequested_info_or_camera':'UNKNOWN_NOT_REQUESTED' if info is None or cam is None else None,'selected_gt_instance_index':None,'selected_instance_correspondence':'UNKNOWN_PENDING_EXPLICIT_EVIDENCE','native_target_admission':False})
    a=write_exclusive(OUT+'/ALL12_SELECTED_TRAIN_FRAME_GT_METADATA.json',{'completed':True,'frames12':output_frames,'no_GT_best_candidate_selection':True,'nominal_projection_executed':False})
    c=write_exclusive(OUT+'/COMPLETE.json',{'completed':True,'PASS':True,'entry':bind(entry_path,read_bytes(entry_path)),'code':entry['code'],'protocol':entry['protocol'],'opaque_bindings_before_values':opaque_receipt,'selected_GT_metadata':a,'selection_statistics':parse_stats,'requested_scene_JSON_kinds':entry['requested_scene_JSON_kinds'],'frames':12,'native_slots_preserved':96,'inputU_preserved':11,'selected_instance_correspondence':'UNKNOWN','mask_depth_CAD_RGB_NPZ_reads':0,'projection_PnP_qL_or_permutation_changes':0,'training_admission':False})
    print(json.dumps(c,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
