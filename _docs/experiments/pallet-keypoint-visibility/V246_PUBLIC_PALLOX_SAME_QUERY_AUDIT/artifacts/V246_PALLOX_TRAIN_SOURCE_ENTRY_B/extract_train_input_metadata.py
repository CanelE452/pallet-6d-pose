"""Root-scoped mixed PUBLIC SOURCE INPUT_ONLY -> unchanged TRAIN44/352.

Reads exactly two input-metadata JSONLs, never their image/tensor/GT pointers.
SOURCE VAL input metadata is read/validated/discarded and disclosed as counts.
No labels, model, inference, training, RGB/GT/NPZ or mixed pack access.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'publication_root_preparation'
H=R/'V246_PALLOX_TRAIN_SOURCE_ENTRY_B'
V94=P/'query_visibility_matched_query_scalar_20261023_v94'
SOURCES={
 'frames':{'path':str(P/'query_visibility_public_cue_grid_20261023_v89/inputs_metadata/FRAMES_INPUTS_ONLY.jsonl'),'sha256':'a7691362fe47824cb9ecd66f41e6e030c94fae9985b8580618b5dc225d185b11','size_bytes':48566},
 'queries':{'path':str(V94/'pallox_actual_initializer/FIXED100/ORIGINAL_NATIVE_QUERY800_INPUT_ONLY.jsonl'),'sha256':'3fabf6a2b653feda0cb4cd8c8df14a157f103bcbe17eb71e37d13ad422978767','size_bytes':420827}}
WRITERS={
 'initializer':{'path':str(V94/'pallox_actual_initializer_preparation/infer_fixed_pallox_queries.py'),'sha256':'8e87d71fc0fc56468b5fd4ce4f366211520d02286e15a3ddd96a9355d59b9ce2','size_bytes':13384},
 'pack_schema':{'path':str(V94/'pallox_actual_query_pack_preparation/build_actual_query_pack.py'),'sha256':'a2777bccdfbf667b51139a3d89b6b2f5db7499861c79dd7df6de73f5dc3713af','size_bytes':17543}}
FRAME_FIELDS={'frame_input_id','image','image_sha256','pair_id','raw_hw','seed_V84_reuse','session','source_family','split','view'}
QUERY_FIELDS={'query_id','frame_index','frame_input_id','pair_id','view','split','native_slot','query_xy','roi_side','original_query_continuous_valid','query_nearest_pixel_valid','query_rgb96_centre_domain_valid','reason','native_semantic_state','human_gold'}
REASONS={'NO_DETECTION_U','OOB_OR_INVALID_U','VALID_QUERY_NOT_A_VISIBILITY_LABEL'}

def lexical(p):
 text=str(p)
 if any(x in text.casefold() for x in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')):raise ValueError('Forbidden before filesystem')
 p=Path(text)
 if not p.is_absolute() or '..' in p.parts:raise ValueError('Absolute literal path required')
 return p

def safe(p):
 p=lexical(p)
 for q in reversed((p,*p.parents)):
  try:mode=os.lstat(q).st_mode
  except FileNotFoundError:continue
  if stat.S_ISLNK(mode):raise ValueError('Symlink rejected')
 return p

def read(p):
 with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def binding(p,data):return {'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data)}

def checked(b):
 data=read(b['path'])
 if hashlib.sha256(data).hexdigest()!=b['sha256'] or len(data)!=b['size_bytes']:raise ValueError('Exact binding changed')
 return data

def records(data):
 result=[];offset=0
 for index,line in enumerate(data.splitlines(keepends=True)):
  if not line.strip():raise ValueError('No blank/missing metadata row')
  row=json.loads(line.decode('utf-8'))
  if not isinstance(row,dict):raise ValueError('Object row required')
  result.append((row,line,{'original_row_index':index,'byte_offset':offset,'byte_length':len(line),'row_sha256':hashlib.sha256(line).hexdigest()}));offset+=len(line)
 if offset!=len(data):raise ValueError('Whole metadata byte coverage')
 return result

def validate_and_select(frame_rows,query_rows):
 if len(frame_rows)!=100 or len(query_rows)!=800:raise ValueError('Immutable100/800 source rows')
 train_frames=[];train_queries=[];frame_ids=set();pair_roles={}
 for i,(f,raw,proof) in enumerate(frame_rows):
  if set(f)!=FRAME_FIELDS or f['split'] not in ('TRAIN_LAB','VAL_HALL'):raise ValueError('Exact original frame schema/role')
  if f['frame_input_id'] in frame_ids:raise ValueError('Duplicate source frame ID')
  frame_ids.add(f['frame_input_id']);pair_roles.setdefault(f['pair_id'],set()).add(f['split'])
  if f['split']=='TRAIN_LAB':
   if not isinstance(f['session'],str) or not f['session'].casefold().endswith('_lab'):raise ValueError('Declared TRAIN lab session')
   p=lexical(f['image']);source=P/'query_visibility_controlled_real_20261022_v66/public_pallet/data'
   if not p.is_relative_to(source) or p.parent.name!='rgb':raise ValueError('TRAIN source RGB metadata pointer only')
   if len(f['image_sha256'])!=64 or any(c not in '0123456789abcdef' for c in f['image_sha256']):raise ValueError('RGB inert SHA syntax')
   if len(f['raw_hw'])!=2 or any(type(x) is not int or x<=0 for x in f['raw_hw']):raise ValueError('Original H/W')
   train_frames.append((f,raw,proof))
 for i,(q,raw,proof) in enumerate(query_rows):
  if set(q)!=QUERY_FIELDS or q['split'] not in ('TRAIN_LAB','VAL_HALL'):raise ValueError('Exact15field teacherless input query schema')
  if q['native_semantic_state']!='UNKNOWN' or q['human_gold'] is not False or q['reason'] not in REASONS:raise ValueError('No native label promotion or extra state')
  if type(q['frame_index']) is not int or type(q['native_slot']) is not int or q['frame_index']!=i//8 or q['native_slot']!=i%8:raise ValueError('Original global order/native8')
  f=frame_rows[q['frame_index']][0]
  if any(q[k]!=f[k] for k in ('frame_input_id','pair_id','view','split')) or q['query_id']!=q['frame_input_id']+':native'+str(q['native_slot']):raise ValueError('Original source identity join')
  if any(type(q[k]) is not bool for k in ('original_query_continuous_valid','query_nearest_pixel_valid','query_rgb96_centre_domain_valid')):raise ValueError('Original boolean support flags')
  if q['split']!='TRAIN_LAB':continue
  if q['query_xy'] is None:
   if q['roi_side'] is not None or q['reason']!='NO_DETECTION_U' or any(q[k] for k in ('original_query_continuous_valid','query_nearest_pixel_valid','query_rgb96_centre_domain_valid')):raise ValueError('Absent input U preserved')
  else:
   xy=q['query_xy'];L=q['roi_side'];height,width=f['raw_hw']
   if not isinstance(xy,list) or len(xy)!=2 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in xy) or not isinstance(L,(int,float)) or isinstance(L,bool) or not math.isfinite(L) or L<=0:raise ValueError('Original finite q/L')
   x,y=xy;continuous=0<=x<width and 0<=y<height;nearest=0<=math.floor(x+.5)<width and 0<=math.floor(y+.5)<height;centre=-.5<=x<width-.5 and -.5<=y<height-.5
   if (q['original_query_continuous_valid'],q['query_nearest_pixel_valid'],q['query_rgb96_centre_domain_valid'])!=(continuous,nearest,centre):raise ValueError('Original input-only validity')
   if q['reason']!=('VALID_QUERY_NOT_A_VISIBILITY_LABEL' if continuous else 'OOB_OR_INVALID_U'):raise ValueError('Original U reason')
  train_queries.append((q,raw,proof))
 if any(len(roles)!=1 for roles in pair_roles.values()):raise ValueError('Paired siblings cross role')
 if len(train_frames)!=44 or len(train_queries)!=352 or sum(q['original_query_continuous_valid'] for q,_,_ in train_queries)!=297:raise ValueError('Prospective TRAIN44/352/297 census')
 if len({f['pair_id'] for f,_,_ in train_frames})!=22:raise ValueError('Declared22TRAIN paired groups')
 return train_frames,train_queries

def run(entry_path):
 eb=read(entry_path);e=json.loads(eb)
 if e.get('approved_SOURCE_input_metadata_extraction') is not True or e.get('automatic_retry') is not False:raise PermissionError('Separate Root true metadata entry required')
 if e['input_metadata']!=SOURCES or e['writer_codes']!=WRITERS:raise ValueError('Exact closed two-input/two-code bindings')
 own=read(__file__)
 if e['code']!=binding(Path(__file__).absolute(),own):raise ValueError('Pinned extractor')
 scope=json.loads(checked(e['read_scope']))
 if scope.get('approved_mixed_PUBLIC_SOURCE_INPUT_ONLY_metadata') is not True or scope.get('SOURCE_VAL_input_metadata_read_parse_discard_allowed') is not True or scope.get('RGB_GT_teacher_arrays_weights_scores_current_access') is not False:raise ValueError('Transparent narrow SOURCE metadata scope required')
 if scope['input_metadata']!=SOURCES:raise ValueError('Scope exact two JSONLs')
 # Writer/schema hashes are verified first. No old writer run/import is called.
 for b in WRITERS.values():checked(b)
 source_bytes={key:checked(b) for key,b in SOURCES.items()}
 fr,qr=validate_and_select(records(source_bytes['frames']),records(source_bytes['queries']))
 out=safe(e['output_directory'])
 if out.parent!=R/'V246_ROOT_PALLOX_TRAIN' or out.name!='train_input_metadata_v1' or out.exists():raise ValueError('Fresh Root output only')
 out.mkdir();outputs={}
 def put(name,data):
  p=out/name
  with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(data)
  outputs[name]=binding(p,data)
 def js(name,obj):put(name,(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
 put('TRAIN44_FRAMES_INPUT_ONLY.jsonl',b''.join(raw for _,raw,_ in fr))
 put('TRAIN352_ORIGINAL_NATIVE_QUERIES_INPUT_ONLY.jsonl',b''.join(raw for _,raw,_ in qr))
 js('ALL_SELECTED_TRAIN_ROW_SHA_INDEX_AND_OFFSETS.json',{'completed':True,'sources':SOURCES,'frames':[dict(proof,frame_input_id=row['frame_input_id']) for row,_,proof in fr],'queries':[dict(proof,query_id=row['query_id'],frame_index=row['frame_index'],native_slot=row['native_slot']) for row,_,proof in qr],'raw_selected_lines_byte_unchanged':True,'VAL_perrow_values_or_identifiers_written':False})
 js('TRAIN44_RGB_WHITELIST_METADATA_ONLY.json',{'completed':True,'RGB_bodies_authorized_by_this_file':False,'frames':[{'original_frame_index':proof['original_row_index'],'frame_input_id':f['frame_input_id'],'pair_id':f['pair_id'],'session':f['session'],'split':f['split'],'view':f['view'],'raw_hw':f['raw_hw'],'image':{'path':f['image'],'sha256':f['image_sha256']}} for f,_,proof in fr],'image_paths_stat_hash_header_decode':0})
 summary={'completed':True,'PASS':True,'source_metadata_rows_read_and_parsed':{'frames':100,'queries':800},'selected_TRAIN':{'frames':44,'queries':352,'pairs':22,'original_continuous_available':297,'original_unavailable_or_OOB':55},'discarded_SOURCE_VAL_INPUT_ONLY_metadata':{'frames':56,'queries':448,'values_or_perrow_identifiers_output':0},'input_schema_fields':{'frames':10,'queries':15},'original_q_L_native_U_index_sourceID_byte_unchanged':True,'no_added_labels_or_native_authority':True,'native_semantic_state':'UNKNOWN','human_gold':False,'SOURCE_VAL_input_metadata_bytes_and_values_read':True,'RGB_GT_teacher_arrays_weights_visibilitylabels_scores_current_access':0,'no_held_RGB_GT_or_visibility_label_read':True,'third_lineage_or_pack_NPZ_read':0,'original_source_files_changed':False,'training_admission':False,'automatic_retry':False}
 js('SUMMARY.json',summary)
 js('COMPLETE.json',{'completed':True,'PASS':True,'schema':'V246_PUBLIC_PALLOX_TRAIN_INPUT_METADATA_V1','entry':binding(Path(entry_path).absolute(),eb),'code':binding(Path(__file__).absolute(),own),'read_scope':e['read_scope'],'writer_codes':WRITERS,'input_metadata':SOURCES,'outputs':dict(outputs),'summary':summary,'no_training_or_model_choice':True})
 print(json.dumps({'completed':True,'PASS':True,'COMPLETE':outputs['COMPLETE.json']}))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);run(ap.parse_args().entry)
