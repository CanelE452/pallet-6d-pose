"""Frozen canonical descriptors for V242 scene-CF inputs; preparation is inert.

Constructed input availability, including proof-UNKNOWN, controls extraction.
Weak targets and references never enter the feature equations or encoder.
"""
import argparse,hashlib,json,os,stat,sys,time,traceback,types
from pathlib import Path
import numpy as np
import torch

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V242_CF_FEATURE_PRODUCER_B'
CANONICAL={'path':str(R/'V239_CANONICAL_FEATURE_PRODUCER_B/build_canonical_source_features.py'),'sha256':'29a890a308f232743f65bc525e5aaee346b8702ae14c5db8a60b15d5bd367ff4'}
SAMPLES_SHA='837bbe7ff406e27c6ed2981493a83c2adab830cd5b2de498e4a48de456456e99'
OFFICIAL_SHA='f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb'
VENDOR='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
ROLES=('COVER','ADJACENT_SHAM')
DATA_NAMES={'node_rgb':'CF_NODE_RGB96.npy','context_rgb':'CF_CONTEXT_RGB224_F32.npy','parent_indices':'PARENT_INDICES.npy','constructed':'CONSTRUCTED.npy','node_valid':'GROUP_NODE_VALID96.npy','context_valid':'GROUP_CONTEXT_VALID224.npy','geometry':'GROUP_GEOMETRY.npz','pair_metadata':'PAIR_METADATA.jsonl'}
GEOMETRY_KEYS=('group_index','q8','L8','original_available8','node_available','node_fraction','context_fraction','box_xyxy','box_supported','context_supported')


def lexical(value):
 s=str(value)
 if any(t in s.casefold() for t in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')):raise ValueError('Forbidden lexical path before filesystem')
 p=Path(s)
 if not p.is_absolute() or '..' in p.parts:raise ValueError('Absolute path without traversal')
 return p


def safe(value):
 p=lexical(value)
 for q in reversed((p,*p.parents)):
  try:mode=os.lstat(q).st_mode
  except FileNotFoundError:continue
  if stat.S_ISLNK(mode):raise ValueError('No symlink ancestors')
 return p


def identity(b):
 if not isinstance(b,dict) or not {'path','sha256'}<=set(b)<={'path','sha256','size_bytes'}:raise ValueError('Exact binding schema')
 p=str(lexical(b['path']));d=b['sha256']
 if not isinstance(d,str) or len(d)!=64 or any(c not in '0123456789abcdef' for c in d):raise ValueError('SHA256 format')
 if 'size_bytes' in b and (type(b['size_bytes']) is not int or b['size_bytes']<0):raise ValueError('Size format')
 return p,d


def bound(b):return {k:b[k] for k in ('path','sha256','size_bytes') if k in b}


def read(path):
 with os.fdopen(os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()


def binding(path):
 p=safe(path);h=hashlib.sha256();n=0
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x);n+=len(x)
 return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':n}


def verify(b,white,pin=None):
 ident=identity(b)
 if ident not in {identity(x) for x in white} or (pin is not None and ident!=identity(pin)):raise ValueError('Closed whitelist before filesystem')
 actual=binding(b['path'])
 if identity(actual)!=ident or ('size_bytes' in b and b['size_bytes']!=actual['size_bytes']):raise ValueError('Pinned bytes changed')
 return safe(b['path'])


def write_json(path,obj):
 with os.fdopen(os.open(safe(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return binding(path)


def load_canonical(b,white):
 p=verify(b,white,CANONICAL);m=types.ModuleType('v242_frozen_canonical_math');m.__file__=str(p)
 exec(compile(read(p),str(p),'exec'),m.__dict__)
 return m


class IndexedRows:
 """A read-only indexing view; no whole CF array copy or new image sampling."""
 def __init__(self,array,rows,role=None):self.array=array;self.rows=np.asarray(rows,np.int64);self.role=role
 def __getitem__(self,key):
  parts=key if isinstance(key,tuple) else (key,)
  first=self.rows[parts[0]]
  return self.array[((first,)+parts[1:]) if self.role is None else ((first,self.role)+parts[1:])]


def parent_geometry(g,cache):
 return {k:np.array(g[k][cache],copy=True) for k in ('q8','L8','original_available8','node_available','node_fraction','context_fraction','box_xyxy')}


def join_parents(rows,indices,constructed,g,samples):
 n=len(indices)
 if n!=5406 or len(rows)!=n or indices.dtype!=np.int64 or constructed.dtype!=bool or constructed.shape!=(n,) or len(set(indices.tolist()))!=n:raise ValueError('All5406 original parents unchanged')
 cache=np.empty(n,np.int64);slot=np.empty(n,np.int64)
 for i,row in enumerate(rows):
  p=int(indices[i]);gi=int(row['group_cache_index']);si=int(row['active_slot']);original=row['original_input']
  if row['pair_index']!=i or row['parent_sample_index']!=p or bool(row['constructed'])!=bool(constructed[i]):raise ValueError('Exact pair row order/availability')
  if not 0<=p<7617 or not 0<=gi<1198 or not 0<=si<8 or row['group_index']!=int(g['group_index'][gi]):raise ValueError('Parent/group/slot indices')
  if not samples['fit'][p] or samples['target'][p]!=0 or not samples['available'][p] or str(samples['row_id'][p])!=original['row_id']:raise ValueError('Original SOURCE FIT available D unchanged')
  if original['q']!=g['q8'][gi,si].tolist() or original['L']!=g['L8'][gi,si] or not g['original_available8'][gi,si] or not g['node_available'][gi,si]:raise ValueError('Original active q/L/input support')
  if samples['family'][p]!=original['family_code']:raise ValueError('Original family preserved')
  if samples['family'][p]==0 and np.argmax(samples['condition'][p,:8])!=si:raise ValueError('Original SYN native condition')
  cache[i],slot[i]=gi,si
 return cache,slot


def extract_role(can,model,arrays,g,cache,slot,constructed,condition_samples,role,preprocess,precontext,crop,device):
 """Canonical math is called unchanged; active/node output shares one tensor."""
 selected=np.flatnonzero(constructed);allg=parent_geometry(g,cache)
 if not len(selected):return np.zeros((5406,1186),np.float32),np.zeros((5406,8,384),np.float32),np.zeros((5406,384),np.float32),0,0
 sg={k:v[selected] for k,v in allg.items()};sample={k:v[selected] for k,v in condition_samples.items()}
 pixel,context,node4,nodes,contexts=can.extract_canonical(model,IndexedRows(arrays['node_rgb'],selected,role),IndexedRows(arrays['node_valid'],cache[selected]),IndexedRows(arrays['context_rgb'],selected,role),IndexedRows(arrays['context_valid'],cache[selected]),sg,preprocess,precontext,crop,device,32)
 b,node=can.assemble_known(pixel,sample,np.arange(len(selected),dtype=np.int64),slot[selected],sg,node4)
 if np.count_nonzero(context[sg['context_fraction']==0]):raise ValueError('Unsupported original context remains zero')
 base_all=np.zeros((5406,1186),np.float32);node_all=np.zeros((5406,8,384),np.float32);context_all=np.zeros((5406,384),np.float32)
 base_all[selected]=b;node_all[selected]=node;context_all[selected]=context
 if base_all[selected,:384].tobytes()!=node_all[selected,slot[selected]].tobytes():raise ValueError('Canonical byte sharing; no tolerance or overwrite')
 if base_all[selected,1154:].tobytes()!=sample['condition'].tobytes():raise ValueError('Original condition32 byteexact')
 return base_all,node_all,context_all,nodes,contexts


def run(entry_path):
 e=json.loads(read(entry_path))
 if e.get('execution_authorized') is not True or e.get('source_only') is not True or e.get('training_authorized') is not False or e.get('automatic_retry') is not False:raise PermissionError('Separate exact Root CF feature entry required')
 if e['batch_size']!=32:raise ValueError('Canonical batch32 fixed')
 code,meta,payload=e['code_whitelist'],e['metadata_whitelist'],e['payload_whitelist']
 verify(e['code'],code,binding(__file__));verify(e['implementation_protocol'],code)
 protocol=json.loads(read(verify(e['root_protocol'],meta)));scope=json.loads(read(verify(e['root_scope'],meta)))
 if scope.get('approved_SOURCE_CF_feature_build') is not True:raise ValueError('Root feature-only scope required')
 complete=json.loads(read(verify(e['data_complete'],meta)));qa=json.loads(read(verify(e['data_QA'],meta)))
 if not all(d.get('completed') is True and d.get('PASS') is True for d in (complete,qa)) or complete['schema']!='V242_SCENE_CONSISTENT_CF_DATA_V1':raise ValueError('Saved data and independent QA must PASS')
 if identity(qa['dataset_complete'])!=identity(e['data_complete']):raise ValueError('Independent QA must bind this exact dataset')
 if complete['counts']['attempted']!=5406 or complete['all_original5406_attempted'] is not True:raise ValueError('No original parent filtering')
 if set(e['data_inputs'])!=set(DATA_NAMES):raise ValueError('Only closed CF feature inputs')
 for key,name in DATA_NAMES.items():
  if identity(e['data_inputs'][key])!=identity(bound(complete['outputs'][name])):raise ValueError('CF input not bound by COMPLETE')
 cfg=e['model_loader_config'];official=cfg['official_state']
 if official['sha256']!=OFFICIAL_SHA or cfg['asset_identity']['vendor_revision']!=VENDOR:raise ValueError('Same official frozen DINO asset')
 if {identity(x) for x in cfg['payload_whitelist']}!={identity(official)} or not {identity(x) for x in cfg['code_whitelist']}<={identity(x) for x in code}:raise ValueError('Closed official loader config')
 if e['samples_npz']['sha256']!=SAMPLES_SHA:raise ValueError('Original nonpixel SOURCE samples only')
 if {identity(x) for x in payload}!={identity(x) for x in [*e['data_inputs'].values(),e['samples_npz'],official]}:raise ValueError('No old prefix/CONTROL/raw image/GT payload')
 can=load_canonical(e['canonical_producer_code'],code)
 for key in can.PINS:
  if identity(e[key])!=can.PINS[key]:raise ValueError('Unchanged canonical code equations')
 out=safe(e['output_directory'])
 if out.parent!=R/'V242_ROOT_SCENE_CF' or out.name!='cf_canonical_features_v1' or out.exists():raise ValueError('Fresh immutable CF feature output only')
 out.mkdir();started=time.monotonic()
 try:
  paths={k:verify(v,payload) for k,v in e['data_inputs'].items()};sample_path=verify(e['samples_npz'],payload)
  arrays={k:np.load(paths[k],mmap_mode='r',allow_pickle=False) for k in ('node_rgb','context_rgb','parent_indices','constructed','node_valid','context_valid')}
  for k,shape,dtype in [('node_rgb',(5406,2,8,3,96,96),np.uint8),('context_rgb',(5406,2,3,224,224),np.float32),('parent_indices',(5406,),np.int64),('constructed',(5406,),bool),('node_valid',(1198,8,96,96),bool),('context_valid',(1198,224,224),bool)]:
   if arrays[k].shape!=shape or arrays[k].dtype!=dtype:raise ValueError('CF shape/dtype exact')
  with np.load(paths['geometry'],allow_pickle=False) as z:g={k:z[k] for k in GEOMETRY_KEYS}
  can.validate_geometry({k:g[k] for k in can.GEOMETRY_KEYS})
  if g['group_index'].dtype!=np.int64 or g['group_index'].shape!=(1198,) or len(set(g['group_index'].tolist()))!=1198:raise ValueError('Original sparse group index')
  with np.load(sample_path,allow_pickle=False) as z:samples=can.validate_samples({k:z[k] for k in can.SAMPLE_KEYS})
  if len(samples['target'])!=7617:raise ValueError('Original7617 samples')
  rows=[json.loads(line) for line in read(paths['pair_metadata']).decode().splitlines() if line.strip()]
  indices=arrays['parent_indices'];constructed=arrays['constructed'];cache,slot=join_parents(rows,indices,constructed,g,samples)
  data={k:v[indices] for k,v in samples.items()};pg=parent_geometry(g,cache);relative=can.relative_xy(pg,np.arange(5406,dtype=np.int64),slot)
  if not np.array_equal(relative[data['family']==0],data['condition'][data['family']==0,16:].reshape(-1,8,2)):raise ValueError('Original SYN relative condition unchanged')
  crop=can.load_code(e['crop_module'],code,'crop_modules',can.PINS['crop_module'])
  loader=can.load_code(e['model_loader_code'],code,'v242_inert_official_loader',can.PINS['model_loader_code'])
  preprocess=can.load_preprocessor(e['preprocess_CODE'],code,'preprocess_views',can.PINS['preprocess_CODE']);precontext=can.load_preprocessor(e['context_preprocess_CODE'],code,'preprocess_target',can.PINS['context_preprocess_CODE'])
  loader.configure_runtime();model=loader.load_backbone(cfg,e['device'],treatment=False);model.eval().requires_grad_(False)
  if any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Encoder fully frozen/eval')
  before=can.state_sha(model.state_dict());outputs={};sharing={}
  for role_i,role in enumerate(ROLES):
   base,node,context,nodes,contexts=extract_role(can,model,arrays,g,cache,slot,constructed,data,role_i,preprocess,precontext,crop,e['device'])
   for suffix,value in [('BASE1186',base),('NODE_CLS',node),('CONTEXT_CLS',context)]:
    name=role+'_'+suffix+'.npy';np.save(out/name,value,allow_pickle=False);outputs[name]={**binding(out/name),'shape':list(value.shape),'dtype':str(value.dtype)}
   use=np.flatnonzero(constructed)
   sharing[role]={'constructed_rows':len(use),'proof_UNKNOWN_is_not_a_feature_gate':True,'supported_nodes_processed':nodes,'context_rows_processed':contexts,'active_first384_byte_exact':base[use,:384].tobytes()==node[use,slot[use]].tobytes(),'original_condition32_byte_exact':base[use,1154:].tobytes()==data['condition'][use].tobytes(),'known_fractions_byte_exact':base[use,1152:1154].tobytes()==data['fractions'][use].tobytes(),'UNCONSTRUCTED_feature_nonzeros':int(np.count_nonzero(base[~constructed])+np.count_nonzero(node[~constructed])+np.count_nonzero(context[~constructed]))}
   if not sharing[role]['active_first384_byte_exact'] or not sharing[role]['original_condition32_byte_exact'] or not sharing[role]['known_fractions_byte_exact'] or sharing[role]['UNCONSTRUCTED_feature_nonzeros']:raise ValueError('Canonical common-source sharing/counts')
   del base,node,context
  for name,value in [('RELATIVE_XY8.npy',relative),('PARENT_INDICES.npy',np.asarray(indices)),('CONSTRUCTED.npy',np.asarray(constructed))]:
   np.save(out/name,value,allow_pickle=False);outputs[name]={**binding(out/name),'shape':list(value.shape),'dtype':str(value.dtype)}
  after=can.state_sha(model.state_dict())
  if before!=after:raise ValueError('Frozen encoder state changed')
  qafile=write_json(out/'CANONICAL_CF_SHARING_QA.json',{'completed':True,'PASS':True,'roles':sharing,'same_forward_construction':True,'feature_overwrite_or_tolerance_relaxation':False})
  result={'completed':True,'PASS':True,'schema':'V242_CANONICAL_CF_FEATURES_V1','entry':binding(entry_path),'code':e['code'],'implementation_protocol':e['implementation_protocol'],'root_protocol':e['root_protocol'],'root_scope':e['root_scope'],'data_complete':e['data_complete'],'data_QA':e['data_QA'],'data_inputs':e['data_inputs'],'samples_npz':e['samples_npz'],'canonical_code':e['canonical_producer_code'],'code_provenance':{k:e[k] for k in can.PINS},'official_base':official,'asset_identity':cfg['asset_identity'],'base_state_sha256_before':before,'base_state_sha256_after':after,'features':outputs,'outputs':outputs,'canonical_sharing_QA':sharing,'canonical_sharing_QA_binding':qafile,'rows':5406,'constructed':int(constructed.sum()),'unconstructed':int((~constructed).sum()),'all5406_original_order_retained':True,'original_condition32_preserved':True,'YCB_native_UNKNOWN_condition0':True,'new_forward_GT_references_owners_pose_ID':False,'GT_teacher_targets_used_as_features':False,'old_cached_prefix_or_CONTROL_payload':0,'classifier_optimizer_or_training':0,'SOURCE_DEV_or_current_or_held_payloads':0,'full_NODE_PIXEL_duplicate_files':0,'automatic_retry':False,'seconds':time.monotonic()-started}
  write_json(out/'COMPLETE.json',result);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json')}),flush=True)
 except BaseException:
  write_json(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'traceback':traceback.format_exc(),'automatic_retry':False});raise


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
