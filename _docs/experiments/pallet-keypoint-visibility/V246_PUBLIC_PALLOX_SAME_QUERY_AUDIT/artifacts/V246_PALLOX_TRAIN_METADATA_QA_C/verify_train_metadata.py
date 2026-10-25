"""Independent stdlib actual metadata QA. SOURCE_VAL input-only rows are parsed/discarded.
Root authorizes only the bound COMPLETE/entry/scope, two original JSONLs and five saved outputs.
No source image, GT, label, array, weight, model or original mixed packet is opened.
"""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,io,json,math,os,stat,traceback

H=Path(__file__).absolute().parent;R=H.parent
DATA=R/'V246_ROOT_PALLOX_TRAIN/train_input_metadata_v1'
COMPLETE={'path':str(DATA/'COMPLETE.json'),'sha256':'8db146bc2a5b7acfd1f59539d9f4e3943f9ea550a734721e30ff795ea8fc807f','size_bytes':5401}
FRAME_FIELDS={'frame_input_id','image','image_sha256','pair_id','raw_hw','seed_V84_reuse','session','source_family','split','view'}
QUERY_FIELDS={'query_id','frame_index','frame_input_id','pair_id','view','split','native_slot','query_xy','roi_side','original_query_continuous_valid','query_nearest_pixel_valid','query_rgb96_centre_domain_valid','reason','native_semantic_state','human_gold'}

def need(ok,msg):
    if not ok:raise ValueError(msg)

def read(binding):
    p=Path(binding['path']);need(p.is_absolute() and '..' not in p.parts,'literal path')
    need(not any(x in str(p).casefold() for x in ('v112','__pycache__','.pyc','current223','publicheld','source_holdout','oldval')),'lexical before FS')
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'ancestor symlink')
    need(stat.S_ISREG(os.lstat(p).st_mode),'leaf symlink/nonregular')
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:b=f.read()
    need((len(b),hashlib.sha256(b).hexdigest())==(binding['size_bytes'],binding['sha256']),'exact SHA/size')
    return b

def lines(data):
    f=io.BytesIO(data);rows=[]
    while f.tell()<len(data):
        offset=f.tell();raw=f.readline();need(raw.strip(),'no blank row')
        row=json.loads(raw);need(isinstance(row,dict),'object row')
        rows.append((row,raw,{'original_row_index':len(rows),'byte_offset':offset,'byte_length':len(raw),'row_sha256':hashlib.sha256(raw).hexdigest()}))
    need(f.tell()==len(data),'all metadata bytes covered');return rows

def write(name,obj):
    p=H/name
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}

def run():
    c=json.loads(read(COMPLETE));need(c['completed'] is True and c['PASS'] is True,'completed extraction')
    entry=json.loads(read(c['entry']));scope=json.loads(read(c['read_scope']))
    need(entry['approved_SOURCE_input_metadata_extraction'] is True and entry['automatic_retry'] is False,'actual Root gate')
    need(scope['approved_mixed_PUBLIC_SOURCE_INPUT_ONLY_metadata'] is True and scope['SOURCE_VAL_input_metadata_read_parse_discard_allowed'] is True and scope['RGB_GT_teacher_arrays_weights_scores_current_access'] is False,'transparent metadata scope')
    sources=c['input_metadata'];need(entry['input_metadata']==scope['input_metadata']==sources,'same two input binding closure')
    need(set(sources)=={'frames','queries'},'two original INPUT_ONLY files only')
    need(sources['frames']['sha256']=='a7691362fe47824cb9ecd66f41e6e030c94fae9985b8580618b5dc225d185b11' and sources['queries']['sha256']=='3fabf6a2b653feda0cb4cd8c8df14a157f103bcbe17eb71e37d13ad422978767','frozen original pins')
    outputs=c['outputs'];names={'TRAIN44_FRAMES_INPUT_ONLY.jsonl','TRAIN352_ORIGINAL_NATIVE_QUERIES_INPUT_ONLY.jsonl','ALL_SELECTED_TRAIN_ROW_SHA_INDEX_AND_OFFSETS.json','TRAIN44_RGB_WHITELIST_METADATA_ONLY.json','SUMMARY.json'}
    need(set(outputs)==names and all(Path(b['path'])==DATA/n for n,b in outputs.items()),'exact saved output leaves')
    saved={n:read(b) for n,b in outputs.items()}
    frames=lines(read(sources['frames']));queries=lines(read(sources['queries']))
    need((len(frames),len(queries))==(100,800),'full original input metadata census')
    need(all(set(f)==FRAME_FIELDS and f['split'] in ('TRAIN_LAB','VAL_HALL') for f,_,_ in frames),'FRAME10 input schema')
    need(all(set(q)==QUERY_FIELDS and q['split'] in ('TRAIN_LAB','VAL_HALL') and q['native_semantic_state']=='UNKNOWN' and q['human_gold'] is False for q,_,_ in queries),'QUERY15/no native gold')
    fr=[x for x in frames if x[0]['split']=='TRAIN_LAB'];qr=[x for x in queries if x[0]['split']=='TRAIN_LAB']
    need((len(fr),len(qr))==(44,352),'selected TRAIN44/352')
    need(saved['TRAIN44_FRAMES_INPUT_ONLY.jsonl']==b''.join(x[1] for x in fr),'TRAIN frame raw bytes/order unchanged')
    need(saved['TRAIN352_ORIGINAL_NATIVE_QUERIES_INPUT_ONLY.jsonl']==b''.join(x[1] for x in qr),'TRAIN query raw bytes/order unchanged')
    pair_roles=defaultdict(set)
    for f,_,_ in frames:pair_roles[f['pair_id']].add(f['split'])
    need(all(len(s)==1 for s in pair_roles.values()),'no source pair role overlap')
    pairs=Counter(f['pair_id'] for f,_,_ in fr);need(len(pairs)==22 and set(pairs.values())=={2},'22 paired TRAIN groups')
    need(len({f['frame_input_id'] for f,_,_ in fr})==44 and len({f['image'] for f,_,_ in fr})==44,'44 frame/image identity strings')
    for i,(q,_,_) in enumerate(queries):
        need((q['frame_index'],q['native_slot'])==(i//8,i%8),'original 100x8 global ordering')
        f=frames[i//8][0]
        need(all(q[k]==f[k] for k in ('frame_input_id','pair_id','view','split')) and q['query_id']==f['frame_input_id']+':native'+str(i%8),'frame-query source/role/native join')
    need(len({q['query_id'] for q,_,_ in qr})==352,'unique selected source query IDs')
    for q,_,_ in qr:
        f=frames[q['frame_index']][0];flags=tuple(q[k] for k in ('original_query_continuous_valid','query_nearest_pixel_valid','query_rgb96_centre_domain_valid'))
        need(all(type(x) is bool for x in flags),'original bool flags')
        if q['query_xy'] is None:need(q['roi_side'] is None and flags==(False,False,False) and q['reason']=='NO_DETECTION_U','absent input U retained')
        else:
            x,y=q['query_xy'];L=q['roi_side'];hh,ww=f['raw_hw']
            need(all(isinstance(z,(int,float)) and not isinstance(z,bool) and math.isfinite(z) for z in (x,y,L)) and L>0,'finite original q/L')
            need(flags==(0<=x<ww and 0<=y<hh,0<=math.floor(x+.5)<ww and 0<=math.floor(y+.5)<hh,-.5<=x<ww-.5 and -.5<=y<hh-.5),'input-only q validity definitions')
    available=sum(q['original_query_continuous_valid'] for q,_,_ in qr);need(available==297,'297 available/55 original unavailable')
    proof=json.loads(saved['ALL_SELECTED_TRAIN_ROW_SHA_INDEX_AND_OFFSETS.json'])
    need(proof['completed'] is True and proof['sources']==sources and proof['raw_selected_lines_byte_unchanged'] is True and proof['VAL_perrow_values_or_identifiers_written'] is False,'offset proof disclosure')
    need(proof['frames']==[dict(p,frame_input_id=f['frame_input_id']) for f,_,p in fr],'44 exact original ordinal/offset/length/SHA')
    need(proof['queries']==[dict(p,query_id=q['query_id'],frame_index=q['frame_index'],native_slot=q['native_slot']) for q,_,p in qr],'352 exact original ordinal/offset/length/SHA')
    rgb=json.loads(saved['TRAIN44_RGB_WHITELIST_METADATA_ONLY.json'])
    expected=[{'original_frame_index':p['original_row_index'],'frame_input_id':f['frame_input_id'],'pair_id':f['pair_id'],'session':f['session'],'split':f['split'],'view':f['view'],'raw_hw':f['raw_hw'],'image':{'path':f['image'],'sha256':f['image_sha256']}} for f,_,p in fr]
    need(rgb['completed'] is True and rgb['RGB_bodies_authorized_by_this_file'] is False and rgb['image_paths_stat_hash_header_decode']==0 and rgb['frames']==expected,'44 inert RGB bindings unchanged; no pointer followed')
    summary=json.loads(saved['SUMMARY.json']);need(summary==c['summary'],'summary binding and embedded scalar equality')
    need(summary['selected_TRAIN']=={'frames':44,'queries':352,'pairs':22,'original_continuous_available':297,'original_unavailable_or_OOB':55},'saved selected counts')
    need(summary['discarded_SOURCE_VAL_INPUT_ONLY_metadata']=={'frames':56,'queries':448,'values_or_perrow_identifiers_output':0} and summary['SOURCE_VAL_input_metadata_bytes_and_values_read'] is True,'honest SOURCE_VAL access')
    need(summary['training_admission'] is False and summary['native_semantic_state']=='UNKNOWN' and summary['human_gold'] is False and summary['no_added_labels_or_native_authority'] is True,'no label/admission promotion')
    result={'completed':True,'PASS':True,'dataset_complete':COMPLETE,'entry':c['entry'],'read_scope':c['read_scope'],'input_metadata':sources,'outputs':outputs,'original_raw_selected_lines_byte_exact':True,'all396_offset_index_length_SHA_exact':True,'all_original_q_L_U_native_source_ID_role_order_preserved':True,'TRAIN':{'frames':44,'queries':352,'pairs':22,'inert_unique_images':44,'original_available':297,'original_unavailable_or_OOB':55},'C_source_input_metadata_rows_read_and_parsed':{'frames':100,'queries':800},'C_SOURCE_VAL_input_metadata_read_then_discarded':{'frames':56,'queries':448,'perrow_values_or_IDs_in_QA_output':0},'native_semantic_state':'UNKNOWN','human_gold':False,'training_admission':False,'C_RGB_GT_teacher_arrays_NPZ_model_weights_visibilitylabels_scores_current_or_held_payload_access':0,'image_pointer_stat_hash_header_decode':0,'producer_code_import_or_execution':0,'model_data_architecture_or_training_choice':0,'scientific_files_modified':False,'checker_code':{'path':str(Path(__file__).absolute()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'limits':'INPUT_ONLY byte/schema/role lineage QA. Public SOURCE_VAL nonpixel metadata was explicitly authorized and parsed; zero-held-metadata access is not claimed. No RGB content/annotation/visibility authority verified.'}
    print(json.dumps(write('REVIEW.json',result)),flush=True)

if __name__=='__main__':
    try:run()
    except BaseException:
        write('FAILED_EXECUTION.json',{'completed':False,'PASS':False,'traceback':traceback.format_exc(),'automatic_retry':False});raise
