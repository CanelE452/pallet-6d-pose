"""Saved SOURCE TRAIN original-BEFORE point-state lookup; root-entry only. No NumPy/image/model/GT derivation."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,json,math,os,stat,traceback
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PROTOCOL=H/'PROTOCOL_BEFORE_SAME_TRAIN_GT_RECORDS_CF_V2.json';SELECTION=V/'preparation_a/SELECTED_SOURCE_TRAIN_BEFORE_GT.json';OUT=V/'source_train_native_point_grounding_v2'
STATES={'D','E','S','U'}
def need(x,m):
 if not x:raise AssertionError(m)
def lexical(p):
 p=Path(p);need(p.is_absolute()and'..'not in p.parts and'v112'not in str(p).casefold(),'Lexical name before filesystem');return p
def rb(p):
 p=lexical(p);need(p.suffix in('.json','.jsonl','.py'),'Only selected metadata/JSON; no binary/header access')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for c in p.parts[1:-1]:n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Ordinary metadata file')
 with os.fdopen(n,'rb')as f:return f.read()
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(rb(p)).hexdigest()}
def checked(b):
 z=rb(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Exact selected JSON/code/metadata binding');return json.loads(z)
def write(p,x):
 p=lexical(p)
 with p.open('x',encoding='utf-8')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rowwrite(p,rows):
 with lexical(p).open('x',encoding='utf-8')as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def before_scene(doc,scene_index):
 # Bounded structural contract, not a recursive alias search. Accept only a direct
 # scene object, direct list of scenes, or lists/direct BEFORE member in top dict.
 candidates=[]
 def add(x):
  if not isinstance(x,dict):return
  identity=('variant'in x or'scene_index'in x)
  if identity and x.get('variant','BEFORE')=='BEFORE'and x.get('scene_index',scene_index)==scene_index:candidates.append(x)
 if isinstance(doc,list):
  for x in doc:add(x)
 elif isinstance(doc,dict):
  add(doc)
  for value in doc.values():
   if isinstance(value,list):
    for x in value:add(x)
  if isinstance(doc.get('BEFORE'),(dict,list)):
   x=doc['BEFORE'];need(not isinstance(x,dict)or x.get('variant','BEFORE')=='BEFORE','BEFORE keyed scene cannot claim another variant');candidates.append(x)
 else:return None,'UNKNOWN_GT_TOP_CONTAINER_SCHEMA'
 unique={json.dumps(x,sort_keys=True):x for x in candidates}
 if len(unique)!=1:return None,'UNKNOWN_OR_AMBIGUOUS_ORIGINAL_BEFORE_SCENE_CONTAINER'
 return next(iter(unique.values())),None

def cf_records(doc,scene_index):
 scene,reason=before_scene(doc,scene_index)
 if scene is None:return None,reason
 # The observed original schema has several CF-indexed diagnostic lists.
 # Only the authoritative saved point-witness records_CF is admissible.
 if not isinstance(scene,dict):return None,'UNKNOWN_BEFORE_SCENE_NOT_OBJECT_WITH_EXPLICIT_RECORDS_CF'
 records=scene.get('records_CF')
 if not isinstance(records,list)or len(records)!=8 or not all(isinstance(r,dict)and'CF_index'in r and'canonical_index'in r for r in records):return None,'UNKNOWN_MISSING_OR_INVALID_EXPLICIT_RECORDS_CF'
 need(all(type(r['CF_index'])is int and type(r['canonical_index'])is int for r in records),'Integer saved native/canonical identities')
 need(sorted(r['CF_index']for r in records)==list(range(8)),'Unique saved CF native0..7, no fallback/permutation fit')
 need(sorted(r['canonical_index']for r in records)==list(range(8)),'Declared canonical0..7 bijection')
 for r in records:
  if'native_canonical_index'in r:need(type(r['native_canonical_index'])is int and r['native_canonical_index']==r['canonical_index'],'Saved canonical declarations agree')
 if'permCF_to_canonical'in scene:
  perm=scene['permCF_to_canonical']
  need(isinstance(perm,list)and len(perm)==8 and all(type(x)is int for x in perm)and sorted(perm)==list(range(8)),'Explicit scene CF-to-canonical permutation')
  need(all(perm[r['CF_index']]==r['canonical_index']for r in records),'Scene permutation equals saved CF declarations')
 # extract_one separately requires each record's canonical index to equal the
 # unchanged original selection's declared_CF_to_canonical for that slot.
 return {r['CF_index']:r for r in records},None

def saved_state(record,key,point0=False):
 if record is None or key not in record:return 'UNKNOWN','MISSING_SAVED_'+key
 value=record[key]
 if point0:
  if not isinstance(value,list)or not value:return 'UNKNOWN','MISSING_SAVED_POINT0_'+key
  value=value[0]
 if value not in STATES:return 'UNKNOWN','INVALID_SAVED_'+key
 return value,None

def saved_reference(record):
 r=None if record is None else record.get('projected_reference_xy')
 if not isinstance(r,list)or len(r)!=2 or any(type(x)not in(int,float)or not math.isfinite(x)for x in r):return None,'MISSING_OR_NONFINITE_SAVED_ANCHOR_PROJECTED_REFERENCE'
 return [float(x)for x in r],None

def locality(q,L,ref,available):
 # Algebraic inverse of original F64 crop sample index, NOT new 3D projection.
 if ref is None or not isinstance(q,list)or len(q)!=2 or any(type(x)not in(int,float)or not math.isfinite(x)for x in q)or type(L)not in(int,float)or not math.isfinite(L)or L<=0:return {'raw96_reference_index_xy':None,'inside_sample_center_extent':None,'inside_canvas_extent':None,'original_input_available':available,'reason':'MISSING_FINITE_ORIGINAL_Q_L_OR_SAVED_REFERENCE'}
 u=[47.5+96.0*(ref[i]-q[i])/L for i in range(2)]
 return {'raw96_reference_index_xy':u,'inside_sample_center_extent':all(0<=x<=95 for x in u),'inside_canvas_extent':all(-.5<=x<95.5 for x in u),'original_input_available':available,'original_input_U_preserved':not available,'not_GT_coordinate_error_or_local_pixel_visibility_proof':True}

def extract_one(original,record,reason):
 k=original['native_slot'];c=original['declared_CF_to_canonical']
 if record is not None:
  need(record['CF_index']==k and record['canonical_index']==c,'Exact declared native->canonical, never GT-best')
  if'native_canonical_index'in record:need(record['native_canonical_index']==c,'Saved canonical fields agree')
 reasons=[]if reason is None else[reason]
 t,tr=saved_state(record,'target_witness_states',True);s,sr=saved_state(record,'combined_witness_states',True);region,rr=saved_state(record,'state');base,br=saved_state(record,'target_alone_region_state');ref,fr=saved_reference(record)
 reasons.extend(x for x in(tr,sr,rr,br,fr)if x)
 return dict(original,source_variant='ORIGINAL_BEFORE_ONLY',canonical_index=c,point0_target_alone_state=t,point0_combined_state=s,region_state=region,target_alone_region_state=base,saved_anchor_projected_reference_xy=ref,original_crop_reference_locality=locality(original['query_xy'],original['L'],ref,original['original_query_available']),UNKNOWN_reasons=reasons,new_visibility_target_created=False,training_admission=False)

def run(entry_path):
 e=checked(bind(entry_path));need(e.get('approved_selected_TRAIN_saved_GT_JSON_lookup')is True and e.get('automatic_retry')is False,'Separate actual root entry required')
 p=checked(e['protocol']);need(e['code']==bind(__file__)==p['code']and e['protocol']==bind(PROTOCOL),'Frozen code/protocol')
 checked(e['scope']);need(e['scope']==p['scope'],'Exact prospective V2 schema-adapter scope');selection=checked(e['selection']);need(e['selection']==p['selection']==bind(SELECTION),'Frozen metadata-first selection')
 whitelist=selection['GT_json_whitelist_NOT_OPENED'];need(e['GT_json_whitelist']==whitelist,'Exact selected original TRAIN GT JSON paths+recorded SHA; no config/binary additions')
 need(selection['config_json_required']is False and e['output_directory']==str(OUT),'No config/projection dependency, fresh output')
 need(all(r['role']=='TRAIN'for r in selection['parents']+selection['original_BEFORE_native8']+selection['physical_control_rows']),'No HOLDOUT query/state export')
 for b in selection['input_metadata_bindings'].values():need(bind(b['path'])==b,'Original metadata immutable')
 for parent in selection['parents']:
  need(bind(parent['source_base_completion']['path'])==parent['source_base_completion'],'Original parent authority unchanged')
 out=lexical(OUT);need(not out.exists(),'No overwrite/retry');out.mkdir();write(out/'STARTED.json',{'entry':bind(entry_path),'completed':False,'automatic_retry':False})
 try:
  rows=[];schemas=[];opened=[]
  for parent in selection['parents']:
   b=parent['GT_json_binding_NOT_OPENED'];need(b in whitelist and lexical(b['path']).name=='SCENES_GT_ONLY.json','Whitelist before GT JSON open')
   doc=checked(b);opened.append(b);records,reason=cf_records(doc,parent['original_scene_index']);schemas.append({'base_index':parent['base_index'],'saved_CF_records_found':records is not None,'UNKNOWN_schema_reason':reason,'original_source_GT':b})
   native=sorted([r for r in selection['original_BEFORE_native8']if r['base_index']==parent['base_index']],key=lambda r:r['native_slot']);need([r['native_slot']for r in native]==list(range(8))and sorted(r['declared_CF_to_canonical']for r in native)==list(range(8)),'All8 original input/native/declared permutation')
   for r in native:rows.append(extract_one(r,None if records is None else records[r['native_slot']],reason))
  mapping=[]
  for r in selection['physical_control_rows']:
   mapping.append(dict(r,parent_lookup_key=[r['source_base_index'],r['native_slot']],saved_native_states_apply_to='ORIGINAL_BEFORE_PARENT_ONLY',changed_V189_COVER_SHAM_point_states='UNKNOWN_NOT_DERIVED_OR_ALIASED_FROM_FINITE_END'))
  need(len(rows)==selection['counts']['original_parent_native_rows']and len(mapping)==selection['counts']['TRAIN_constructed_physical_rows'],'All selected native8/control rows retained')
  rowwrite(out/'ALL_ORIGINAL_BEFORE_NATIVE_POINT_STATE_AND_LOCALITY.jsonl',rows);rowwrite(out/'ALL_CONSTRUCTED_TRAIN_CONTROL_ROWS_TO_BEFORE_PARENT.jsonl',mapping)
  summary={'completed':True,'PASS':True,'source_TRAIN_only':True,'selection_counts':selection['counts'],'original_inputU':sum(not r['original_query_available']for r in rows),'point0_target_alone_states':dict(Counter(r['point0_target_alone_state']for r in rows)),'point0_combined_states':dict(Counter(r['point0_combined_state']for r in rows)),'region_states':dict(Counter(r['region_state']for r in rows)),'schemas':schemas,'no_new_control_variant_native_visibility_labels':True,'no_new_labels_or_admission':True,'authority':'Declared procedural source point reference only; not public physical native gold or coordinate correction','actual_GT_JSON_reads':len(opened),'NPZ_NPY_RGB_depth_owner_weights_headers_or_new_projection_ray_model_reads_or_calls':0,'SOURCE_HOLDOUT_rows_analyzed_or_exported':0}
  write(out/'SUMMARY.json',summary)
  for b in opened:need(bind(b['path'])==b,'Original selected GT JSON unchanged after lookup')
  complete={'completed':True,'PASS':True,'code':bind(__file__),'protocol':e['protocol'],'entry':bind(entry_path),'selection':e['selection'],'GT_json_whitelist':whitelist,'outputs':{n:bind(out/n)for n in ('ALL_ORIGINAL_BEFORE_NATIVE_POINT_STATE_AND_LOCALITY.jsonl','ALL_CONSTRUCTED_TRAIN_CONTROL_ROWS_TO_BEFORE_PARENT.jsonl','SUMMARY.json')},'point_state_rows':len(rows),'physical_TRAIN_control_rows_mapped':len(mapping),'GT_json_files_read':len(opened),'new_point_or_region_labels_created':False,'training_admission':False,'model_fit_forward_PnP_new_projection_or_ray_calls':0,'original_q_L_native_U_and_source_bindings_unchanged':True}
  write(out/'COMPLETE.json',complete);print(json.dumps(complete,ensure_ascii=False))
 except BaseException as error:
  write(out/'FAILED_NO_AUTOMATIC_RETRY.json',{'completed':False,'PASS':False,'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'automatic_retry':False});raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--entry',type=Path,required=True);run(a.parse_args().entry)
