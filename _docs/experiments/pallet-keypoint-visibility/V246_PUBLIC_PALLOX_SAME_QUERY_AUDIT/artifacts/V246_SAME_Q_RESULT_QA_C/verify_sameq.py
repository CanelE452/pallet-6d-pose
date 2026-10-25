"""Independent saved TRAIN same-q support QA; exact22 certificates, no raw GT/RGB/model.
No producer/helper imports. The four source Boolean fields remain conditional teacher evidence.
"""
import argparse,hashlib,io,json,math,os,stat,traceback
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np

H=Path(__file__).absolute().parent;R=H.parent
P=R.parent;V66=P/'query_visibility_controlled_real_20261022_v66'
TRAIN=R/'V246_ROOT_PALLOX_TRAIN/train_input_metadata_v1/COMPLETE.json'
TRAIN_SHA='8db146bc2a5b7acfd1f59539d9f4e3943f9ea550a734721e30ff795ea8fc807f'
PROBE_SHA='d95196bf6b3f406d1356e8dfa0bf532d0371de82b48dabfaf30beb5c4076ebd5'
PROTOCOL_SHA='0d12434e3330c3cdd79e9ae7281a5d377fc5a7e5273eb395843a4d0437d63326'
KEYS=('visible_positive','visible_negative','full_positive','full_negative')
JOINT=((0,0),(1,0),(1,1),(2,0),(2,1),(2,2))

def need(ok,msg):
    if not ok:raise ValueError(msg)

def ident(b):
    s=b['path'];need(not any(x in s.casefold() for x in ('v112','__pycache__','.pyc','current','private','held','holdout','oldval','source_val')),'lexical before FS')
    p=Path(s);need(p.is_absolute() and '..' not in p.parts,'literal absolute path')
    need(len(b['sha256'])==64 and all(c in '0123456789abcdef' for c in b['sha256']),'SHA syntax');return s,b['sha256']

def raw(path):
    p=Path(path)
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'symlink ancestor')
    need(stat.S_ISREG(os.lstat(p).st_mode),'symlink/nonregular leaf')
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def checked(b,white):
    need(ident(b) in {ident(x) for x in white},'not explicitly whitelisted')
    data=raw(b['path']);need(hashlib.sha256(data).hexdigest()==b['sha256'],'payload SHA')
    for key in ('size_bytes','bytes'):
        if key in b:need(len(data)==b[key],'payload bytes')
    return data

def region(fields):
    vp,vn,fp,fn=fields
    visible=1 if vp.any() else 0 if vn.all() else -1
    full=1 if fp.any() else 0 if fn.all() else -1
    possible=[]
    if visible!=0 and full!=0:possible.append(0)
    if visible!=1 and full!=0:possible.append(1)
    if visible!=1 and full!=1:possible.append(2)
    state=0 if vp.any() else 1 if vn.all() and fp.any() else -1
    need(state==(possible[0] if len(possible)==1 else -1),'independent conditional decision')
    return {'state':state,'possible_states':possible,'visible_status':visible,'full_status':full,'pixels':int(vp.size),
            'known_visible_pixels':int(vp.sum()),'known_nonvisible_pixels':int(vn.sum()),
            'known_full_positive_pixels':int(fp.sum()),'known_full_negative_pixels':int(fn.sum()),
            'unknown_visible_pixels':int((~vp&~vn).sum()),'unknown_full_pixels':int((~fp&~fn).sum())}

def answer(fields,xy):
    invalid={'valid':False,'exact_state':-1,'radius2_state':-1,'joint_known':False,'joint_state':-1,
             'joint_compatible_codes':[],'joint_compatible_mask':0,'semantic_self':None,'corner_semantic_truth':None}
    if not isinstance(xy,list) or len(xy)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in xy):return invalid
    h,w=fields[0].shape;x,y=(math.floor(x+.5) for x in xy)
    if not (0<=xy[0]<w and 0<=xy[1]<h and 0<=x<w and 0<=y<h):return invalid
    x0,x1=max(0,x-2),min(w,x+3);y0,y1=max(0,y-2),min(h,y+3)
    window=tuple(a[y0:y1,x0:x1] for a in fields)
    exact=region(tuple(a[y:y+1,x:x+1] for a in fields));near=region(window)
    rest=np.ones(window[0].shape,bool);rest[y-y0,x-x0]=False;other=region(tuple(a[rest] for a in window))
    pairs=set()
    for a in exact['possible_states']:
        for b in other['possible_states']:
            combined=0 if 0 in (a,b) else 1 if 1 in (a,b) else 2
            pairs.add(JOINT.index((a,combined)))
    codes=sorted(pairs);known=exact['state']!=-1 and near['state']!=-1
    joint=JOINT.index((exact['state'],near['state'])) if known else -1
    if known:need(codes==[joint],'joint completion consistency')
    vp,vn,fp,fn=window;allE=vn&fp&~vp
    return {'valid':True,'pixel':[x,y],'exact_state':exact['state'],'radius2_state':near['state'],
            'exact':exact,'radius2':near,'window_without_exact':other,'joint_known':known,'joint_state':joint,
            'joint_compatible_codes':codes,'joint_compatible_mask':sum(1<<c for c in codes),
            'radius2_strict_E_core':bool(near['pixels']==25 and allE.all()),'radius2_strict_O_core':False,
            'radius2_clipped_by_frame':near['pixels']!=25,'semantic_self':None,'corner_semantic_truth':None}

def write(name,value):
    p=H/name
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}

def run(entry_path):
    # Entry is owned administrative metadata, and is lexically checked before filesystem.
    ident({'path':str(Path(entry_path).absolute()),'sha256':'0'*64});e=json.loads(raw(entry_path))
    need(e['approved_saved_TRAIN_sameq_QA'] is True and e['automatic_retry'] is False,'Root true QA entry required')
    own=Path(__file__).absolute();need(e['code']['path']==str(own) and hashlib.sha256(raw(own)).hexdigest()==e['code']['sha256'],'frozen checker')
    mw=e['metadata_whitelist'];cw=e['certificate_whitelist'];need(len(cw)==len({ident(x) for x in cw})==22,'22 unique certificate bindings')
    scope=json.loads(checked(e['root_scope'],mw))
    need(scope['approved_saved_TRAIN_sameq_QA'] is True and {ident(x) for x in scope['certificate_whitelist']}=={ident(x) for x in cw},'Root22 payload scope')
    need({ident(x) for x in scope['metadata_whitelist']}|{ident(e['root_scope'])}=={ident(x) for x in mw},'closed Root metadata scope')
    def meta(b):return json.loads(checked(b,mw))
    tc=meta(e['train_complete']);qa=meta(e['train_metadata_QA'])
    need(ident(e['train_complete'])==(str(TRAIN),TRAIN_SHA) and tc['completed'] is True and tc['PASS'] is True,'pinned TRAIN extraction')
    need(qa['completed'] is True and qa['PASS'] is True and ident(qa['dataset_complete'])==ident(e['train_complete']),'independent TRAIN QA binding')
    frames=[json.loads(x) for x in checked(tc['outputs']['TRAIN44_FRAMES_INPUT_ONLY.jsonl'],mw).splitlines()]
    queries=[json.loads(x) for x in checked(tc['outputs']['TRAIN352_ORIGINAL_NATIVE_QUERIES_INPUT_ONLY.jsonl'],mw).splitlines()]
    need((len(frames),len(queries))==(44,352) and all(x['split']=='TRAIN_LAB' for x in frames+queries),'TRAIN only44/352')
    pairs={};fm={f['frame_input_id']:f for f in frames}
    for f in frames:
        need(f['view'] in ('before','after'),'original paired view');pair=pairs.setdefault(f['pair_id'],{})
        need(f['view'] not in pair,'unique frame view');pair[f['view']]=f
    need(len(pairs)==22 and all(set(x)=={'before','after'} for x in pairs.values()),'22 pair source join')
    c=meta(e['probe_complete']);need(c['completed'] is True and c['PASS'] is True and c['stage']=='probe','probe complete')
    need(c['code']['sha256']==PROBE_SHA and c['protocol']['sha256']==PROTOCOL_SHA and ident(c['train_complete'])==ident(e['train_complete']),'frozen producer authority')
    sc=meta(c['selected_metadata_complete']);need(sc['completed'] is True and sc['PASS'] is True and sc['stage']=='select','selected22 stage complete')
    need(ident(sc['train_complete'])==ident(e['train_complete']),'selected TRAIN source')
    records=[json.loads(x) for x in checked(sc['metadata'],mw).splitlines()];white=meta(sc['certificate_payload_whitelist'])
    need(len(records)==white['pair_count']==len(white['files'])==len(c['certificate_whitelist'])==22 and white['TRAIN_only'] is True and {ident(x) for x in white['files']}=={ident(x) for x in cw}=={ident(x) for x in c['certificate_whitelist']},'selected certificate exact closure')
    bypair={x['pair_id']:x for x in records};need(len(bypair)==22 and set(bypair)==set(pairs),'all/only original TRAIN pairs')
    resultrows=[json.loads(x) for x in checked(c['rows'],mw).splitlines()];summary=meta(c['summary']);fieldqa=meta(c['certificate_QA'])
    need(len(resultrows)==352,'all original query rows retained')
    results={};cert_checks=[];viewqa=[]
    for pid,pair in pairs.items():
        rec=bypair[pid];need(rec['split']=='TRAIN_LAB' and rec['tier']=='CONDITIONAL_TEACHER_NOT_GOLD','source role/conditional authority')
        need(rec['pixel_RGB_depth_alignment_certified'] is False and rec['native_depth_is_assumed_RGB_grid'] is True,'uncertainty flags preserved')
        expected=V66/'public_pallet/SYMMETRIC_ALL50_V2'/(pid.replace(':','__')+'.npz')
        need(rec['evidence']['path']==str(expected),'exact pair certificate path')
        data=checked(rec['evidence'],cw);cert_checks.append({'binding':rec['evidence'],'bytes_verified':len(data)})
        with np.load(io.BytesIO(data),allow_pickle=False) as z:
            for view,side in (('before','A'),('after','B')):
                f=pair[view];im=rec['files'][side]['image']
                need(im['path']==f['image'] and im['sha256']==f['image_sha256'] and rec[side+'_frame']==Path(im['path']).stem,'actual original source frame, image pointer inert')
                fields=tuple(z[k+'_'+side] for k in KEYS)
                need(all(a.dtype==np.bool_ and a.shape==(480,640) for a in fields),'only4 Boolean native fields')
                vp,vn,fp,fn=fields;need(not fn.any() and not (vp&vn).any() and not (vp&~fp).any(),'fullnegative/O disabled and no certificate conflict')
                empty=not any(a.any() for a in fields)
                need(rec['supplied_pose_and_K_identical'] or empty,'ineligible metadata has empty fields')
                viewqa.append({'pair_id':pid,'view':side,'native_boolean_fields':4,'shape':[480,640],'full_negative_pixels':0,'conflicts':0,'empty_certificate':empty,'pair_metadata_eligible':rec['supplied_pose_and_K_identical']})
                for q in queries:
                    if q['pair_id']!=pid or q['view']!=view:continue
                    truth=answer(fields,q['query_xy'])
                    if not q['original_query_continuous_valid']:need(not truth['valid'] and truth['exact_state']==truth['radius2_state']==-1,'original U retained')
                    reason=('PAIR_METADATA_INELIGIBLE_UNKNOWN' if not rec['supplied_pose_and_K_identical'] else 'EMPTY_CERTIFICATE_UNKNOWN' if empty else 'INPUT_OR_ROUNDED_QUERY_UNSUPPORTED_U' if not truth['valid'] else 'PARTIAL_OR_NO_CERTIFIED_SUPPORT_UNKNOWN' if truth['exact_state']==-1 else 'EXPLICIT_CONDITIONAL_SAME_PIXEL_SUPPORT')
                    results[q['query_id']]={'original_input_unchanged':q,'certificate_binding':rec['evidence'],'certificate_view':side,'conditional_pixel_truth':truth,'conditional_reason':reason,'native_visibility_label':None,'native_expected_part_correspondence':'UNKNOWN','training_admission':False}
    expected=[results[q['query_id']] for q in queries];need(resultrows==expected,'all352 complete truth/input/order/evidence/authority fields independently exact')
    need(all(q['native_semantic_state']=='UNKNOWN' and q['human_gold'] is False for q in queries),'native/Human authority unchanged')
    truths=[x['conditional_pixel_truth'] for x in expected]
    count={'rows':352,'frames':44,'pairs':22,'original_continuous_available':297,'original_input_U_retained':55,
           'exact1pixel_counts':dict(Counter(str(x['exact_state']) for x in truths)),'radius2_5x5_counts':dict(Counter(str(x['radius2_state']) for x in truths)),
           'helper_valid':sum(x['valid'] for x in truths),'radius2_strict_E_core':sum(x.get('radius2_strict_E_core',False) for x in truths),'radius2_clipped_by_frame':sum(x.get('radius2_clipped_by_frame',False) for x in truths)}
    need(sum(q['original_query_continuous_valid'] for q in queries)==297,'original297/55 census')
    for k,v in count.items():need(summary[k]==v,'independent summary '+k)
    need(summary['semantic_SELF'] is None and summary['native_labels_created']==0 and summary['training_admission'] is False,'summary no semantic label promotion')
    need(fieldqa['completed'] is True and fieldqa['PASS'] is True and fieldqa['rows']==viewqa,'all44 field QA independently exact')
    additional=sum(q['original_query_continuous_valid'] and not t['valid'] for q,t in zip(queries,truths))
    review={'completed':True,'PASS':True,'probe_complete':e['probe_complete'],'train_complete':e['train_complete'],'train_metadata_QA':e['train_metadata_QA'],'root_scope':e['root_scope'],'selected_metadata_complete':c['selected_metadata_complete'],'counts':count,'original_available_but_rounded_pixel_unsupported_U':additional,'all352_inputs_and_conditional_truth_exact':True,'certificate_closed_SHA_checks':cert_checks,'four_fields_per_view_loaded_only':True,'raw_RGB_depth_mask_mesh_pose_K_models_scores_current_or_held_access':0,'new_SOURCE_VAL_metadata_access':0,'source_native_or_SELF_labels_created':0,'native_expected_part_correspondence':'UNKNOWN','human_gold':False,'training_admission':False,'model_data_architecture_threshold_choices':0,'physical_gold_certified':False,'limits':'Recounts supplied conditional Boolean evidence at the same original q/window. Does not regenerate source certificates, prove RGB-depth registration, physical part/native correctness or model performance.'}
    print(json.dumps(write('REVIEW.json',review)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);a=p.parse_args()
    try:run(a.entry)
    except BaseException:
        write('FAILED_EXECUTION.json',{'completed':False,'PASS':False,'traceback':traceback.format_exc(),'automatic_retry':False});raise
