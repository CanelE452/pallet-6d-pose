"""V242 closed SOURCE FIT scene-CF driver. No payload I/O without Root entry.

Frozen V241 pure math owns construction. This adapter owns exact closed decoding,
streamed storage and provenance; reference/IDs/masks never become model features.
"""
import argparse
import ast
from collections import OrderedDict, Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types
import numpy as np
from PIL import Image

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'publication_root_preparation'
H = R / 'V242_SCENE_CF_DRIVER_B'
PROTOCOL = {'path': str(R/'V242_ROOT_SCENE_CF/PROTOCOL_BEFORE_ACTUAL_SCENE_CF_DATA.json'), 'sha256': '03177adde9dca4e91e04663bd0a7a1976d176a876c114fb67844acc6f27ecc2c'}
SCOPE = {'path':str(R/'V242_ROOT_SCENE_CF/READ_SCOPE_BEFORE_NATIVE_RGB.json'),'sha256':'48c865bc539f70d6c387f31f6fe8257ef43db2a50394161ef6caee166589a25c'}
METADATA = {
 'manifest': ('MANIFEST.json','e9a225cb66d339775700fb34823d3ef86755f2e9dd9559dd5787c4beb709f847'),
 'parents': ('PARENTS_ALL5406.jsonl','b439b10b866953b51706eb0195abc32e4de9be2ee15f4e51dd00313c43d21e3d'),
 'groups': ('PARENT_GROUPS_INPUT_ONLY.jsonl','c98785e6e5d26a9788491cd6400f9af5ab533538ef6d8e2a49795f3593ac4236'),
 'parent_images': ('PARENT_IMAGE_WHITELIST.json','bfdfa28241fece306b6435ca5ae1ba943865dffbf12cb5ac34d9999d33b3f44d'),
 'donors': ('DONOR64_CLOSED_METADATA.json','3c2270c787687244eb79718b6d34df272d9377370bc94d1234b3710cb9fe0c83'),
}
METADATA = {k: {'path':str(R/'V241_SCENE_CF_MANIFEST_C'/v[0]),'sha256':v[1]} for k,v in METADATA.items()}
DEPENDENCIES = {
 'scene_cf_math': {'path':str(R/'V241_SCENE_CF_PRODUCER_B/scene_cf_math.py'),'sha256':'77b88de4311ff38c3baa480feb179b15a6e6ad8d9ea409701b98c6794fbd92c8'},
 'crop90': {'path':str(P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'),'sha256':'6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'},
 'context107': {'path':str(P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'),'sha256':'aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3'},
 'v235_compositor': {'path':str(R/'V235_CF_COMPOSITOR_A/compositor.py'),'sha256':'d940ff8cb901482a77761dfac638ab473a5841d682ef0150f1ece0ab8c5fe4f4'},
 'scene_math': {'path':str(R/'V237_CONTEXT_COMPOSITION_A/context_scene_math.py'),'sha256':'f48545c907cf0705a1cd9ee408e54f520f7eef4563fb166f20a1821b3fa9b30f'},
}
ROLES = ('COVER','ADJACENT_SHAM')
COUNTS = {'parents':5406,'groups':1198,'images':804,'donors':64,'donor_PNG':125}
ARRAYS = {
 'CF_NODE_RGB96.npy': ('uint8',(5406,2,8,3,96,96)),
 'CF_CONTEXT_RGB224_F32.npy': ('float32',(5406,2,3,224,224)),
 'PARENT_INDICES.npy': ('int64',(5406,)),
 'PAIR_TARGETS.npy': ('int8',(5406,2)),
 'CONSTRUCTED.npy': ('bool',(5406,)), 'PROOF_PASS.npy': ('bool',(5406,)),
 'GROUP_NODE_VALID96.npy': ('bool',(1198,8,96,96)),
 'GROUP_CONTEXT_VALID224.npy': ('bool',(1198,224,224)),
 'PAIR_LAYER_RGB96.npy': ('uint8',(5406,2,3,96,96)),
 'PAIR_ALPHA_PACKBITS.npy': ('uint8',(5406,2,1152)),
}


def lexical(path):
 s = os.fspath(path)
 if not isinstance(s,str) or any(t in s.lower() for t in ('v112','__pycache__','.pyc')):
  raise ValueError('Forbidden lexical path before filesystem access')
 p = Path(s)
 if not p.is_absolute() or any(x in ('.','..') for x in p.parts):
  raise ValueError('Absolute normalized paths only')
 return p


def safe(path, must_exist=True):
 p=lexical(path)
 for a in reversed((p,)+tuple(p.parents)):
  try: m=os.lstat(a)
  except FileNotFoundError:
   if must_exist: raise
   continue
  if stat.S_ISLNK(m.st_mode): raise ValueError('Symlink forbidden: '+str(a))
 return p


def identity(binding):
 if not isinstance(binding,dict) or set(binding)-{'path','sha256','size_bytes'} or not {'path','sha256'}<=set(binding):
  raise ValueError('Closed binding keys')
 p=str(lexical(binding['path'])); digest=binding['sha256']
 if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
  raise ValueError('SHA256 format')
 if 'size_bytes' in binding and (type(binding['size_bytes']) is not int or binding['size_bytes']<0): raise ValueError('Binding size')
 return p,digest


def same(binding, expected):
 if identity(binding)!=identity(expected): raise ValueError('Exact binding differs')


@contextmanager
def verified(binding):
 path,digest=identity(binding); p=safe(path)
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  m=os.fstat(f.fileno())
  if not stat.S_ISREG(m.st_mode): raise ValueError('Regular files only')
  if 'size_bytes' in binding and m.st_size!=binding['size_bytes']: raise ValueError('Byte size differs')
  hh=hashlib.sha256()
  for block in iter(lambda:f.read(1024*1024),b''): hh.update(block)
  if hh.hexdigest()!=digest: raise ValueError('Opaque SHA differs: '+str(p))
  f.seek(0)
  yield f


def load_json(binding):
 with verified(binding) as f: return json.load(f)


def load_jsonl(binding):
 with verified(binding) as f: return [json.loads(line) for line in f if line.strip()]


def digest_array(a):
 return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(path, obj):
 p=safe(path,False)
 with p.open('x',encoding='utf-8') as f:
  json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


def file_binding(path):
 p=safe(path); hh=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): hh.update(block)
 return {'path':str(p),'sha256':hh.hexdigest(),'size_bytes':p.stat().st_size}


def payload_identity(image):
 if set(image)!={'path','sha256','format','member'}: raise ValueError('Closed image keys')
 identity({k:image[k] for k in ('path','sha256')})
 if (image['format'],image['member']) not in (('png',None),('npz','RGB_uint8')): raise ValueError('Only PNG or RGB_uint8 NPZ')
 return image['path'],image['sha256'],image['format'],image['member']


def payload_closure(images, donors):
 result={}
 def add(image):
  key=payload_identity(image); path=key[0]
  if path in result and payload_identity(result[path])!=key: raise ValueError('Conflicting payload identity')
  result[path]=image
 for item in images['images']: add(item['image'])
 for b in donors['exact_future_PNG_whitelist']:
  add({'path':b['path'],'sha256':b['sha256'],'format':'png','member':None})
 return list(result.values())


class Decoder:
 """Opaque full-file SHA first; only approved RGB/mask numerical decoding.

 Compressed NPZ container bytes may transit during SHA/decompression; only the
 RGB_uint8 member is numerically decoded. No member enumeration is performed.
 """
 def __init__(self,payloads,max_bytes=128*1024*1024):
  self.white={payload_identity(b)[0]:b for b in payloads}
  self.cache=OrderedDict(); self.bytes=0; self.maximum=int(max_bytes)
  self.accesses=Counter(); self.max_cached=0

 def get(self,image,mask=False,raw_hw=None):
  key=payload_identity(image)
  if key[0] not in self.white or payload_identity(self.white[key[0]])!=key: raise ValueError('Payload outside exact whitelist')
  cachekey=(key,bool(mask))
  if cachekey in self.cache:
   a=self.cache.pop(cachekey); self.cache[cachekey]=a
  else:
   with verified({'path':key[0],'sha256':key[1]}) as f:
    if key[2]=='npz':
     if mask: raise ValueError('No NPZ mask access')
     with np.load(f,allow_pickle=False) as z: a=z['RGB_uint8']
    else:
     with Image.open(f) as im: a=np.asarray(im if mask else im.convert('RGB')).copy()
   if mask:
    if a.ndim!=2 or min(a.shape)<1: raise ValueError('Donor visible mask rank')
    a=np.ascontiguousarray(a>0)
   else:
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<1: raise ValueError('Decoded original RGB uint8 HWC3')
    a=np.ascontiguousarray(a)
   self.accesses[(key[0],bool(mask))]+=1
   if a.nbytes<=self.maximum:
    while self.cache and self.bytes+a.nbytes>self.maximum:
     _,old=self.cache.popitem(last=False); self.bytes-=old.nbytes
    self.cache[cachekey]=a; self.bytes+=a.nbytes; self.max_cached=max(self.max_cached,self.bytes)
  if raw_hw is not None and tuple(a.shape[:2])!=tuple(raw_hw): raise ValueError('Declared raw_hw differs')
  return a


def load_math(bindings):
 modules={}
 for name,expected in DEPENDENCIES.items():
  same(bindings[name],expected)
  with verified(bindings[name]) as f: src=f.read().decode('utf-8')
  if name=='crop90':
   tree=ast.parse(src); fn=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='crop']
   if len(fn)!=1: raise ValueError('Pinned crop AST')
   ns={'np':np}; exec(compile(ast.Module(body=fn,type_ignores=[]),expected['path'],'exec'),ns)
   modules[name]=ns['crop']
  else:
   m=types.ModuleType('v242_pinned_'+name); exec(compile(src,expected['path'],'exec'),m.__dict__); modules[name]=m
 return modules


def validate_metadata(entry):
 same(entry['protocol'],PROTOCOL); load_json(entry['protocol'])
 if set(entry['metadata'])!=set(METADATA) or set(entry['code_dependencies'])!=set(DEPENDENCIES): raise ValueError('Closed code/metadata keys')
 for k,v in METADATA.items(): same(entry['metadata'][k],v)
 for k,v in DEPENDENCIES.items(): same(entry['code_dependencies'][k],v)
 docs={k:(load_jsonl(v) if k in ('parents','groups') else load_json(v)) for k,v in entry['metadata'].items()}
 parents,groups=docs['parents'],docs['groups']; images=docs['parent_images']; donors=docs['donors']
 if [len(parents),len(groups),len(images['images']),len(donors['donors']),len(donors['exact_future_PNG_whitelist'])]!=[5406,1198,804,64,125]: raise ValueError('Closed cardinalities')
 if len({p['parent_sample_index'] for p in parents})!=5406 or len({g['group_index'] for g in groups})!=1198: raise ValueError('Duplicate parent/group')
 if [p['parent_sample_index'] for p in parents]!=sorted(p['parent_sample_index'] for p in parents): raise ValueError('Original parent order')
 gg={g['group_index']:g for g in groups}; dd={d['donor_id']:d for d in donors['donors']}
 ii={payload_identity(x['image']) for x in images['images']}; seen=set(); families=Counter()
 for p in parents:
  original=p['original_input']; g=gg[p['group_index']]; d=dd[p['same_V235_donor_id']]
  if original['fit'] is not True or original['target']!=0 or original['original_state']!='D' or original['original_query_available'] is not True or original['crop_available'] is not True: raise ValueError('Only exact original FIT available D')
  if original['group_index']!=g['group_index'] or p['group_id']!=g['group_id'] or p['image']!=g['image']: raise ValueError('Actual image/group join')
  if payload_identity(g['image']) not in ii: raise ValueError('Parent image not closed')
  slot=p['active_slot']; q=np.asarray(g['q8'],np.float64); L=np.asarray(g['L8'],np.float64); av=np.asarray(g['original_available8'],bool)
  if q.shape!=(8,2) or L.shape!=(8,) or av.shape!=(8,) or not 0<=slot<8 or not av[slot]: raise ValueError('Original group eight slots')
  if not np.array_equal(np.asarray(original['q'],np.float64),q[slot]) or original['L']!=L[slot]: raise ValueError('Original q/L')
  if d['original_role']!='TRAIN' or d['internal_source_role']!='INTERNAL_SOURCE_FIT' or d['original_associated_r_state']!='D': raise ValueError('Only prior FIT donor')
  if identity(p['same_V235_donor_RGB'])!=identity(d['RGB']) or identity(p['same_V235_donor_visible_mask'])!=identity(d['visible_mask']): raise ValueError('Same exact donor')
  if p['same_V235_donor_object_id_sampling_only']!=d['object_id']: raise ValueError('Donor bookkeeping')
  if original['family_code']==1 and p['same_V235_base_object_id_sampling_only']==d['object_id']: raise ValueError('Original YCB different-object rule')
  if p.get('old_pixel_teacher_proof_inherited') is not False: raise ValueError('Old proof not admission')
  seen.add(g['group_index']);families[str(original['family_code'])]+=1
 if len(seen)!=1198 or sorted(families.values())!=[841,4565]: raise ValueError('All original parents/groups/families')
 closure=payload_closure(images,donors)
 if len(entry['payloads'])!=len(closure) or {payload_identity(x) for x in entry['payloads']}!={payload_identity(x) for x in closure}: raise ValueError('Exact actual payload closure required')
 return docs,closure


def layer_carriers(result,parent,donor_rgb,donor_mask,compositor,math):
 """One exact repeat of the unchanged proposal solely to retain QA carrier."""
 slot=parent['active_slot']; original=parent['original_input']; before=result['inputs']['BEFORE']
 prop=compositor.compose_counterfactual_pair(before['NODE_RGB96'][slot],before['NODE_VALID96'][slot],donor_rgb,donor_mask,np.asarray(original['q'],np.float64),np.float64(original['L']),np.asarray(parent['teacher_reference_xy'],np.float64),original['row_id'])
 if not prop['success'] or math.plain(prop['donor_plan'])!=math.plain(parent['same_V235_donor_plan']): raise ValueError('Repeated unchanged proposal differs')
 layers=np.zeros((2,3,96,96),np.uint8); alphas=np.zeros((2,96,96),bool)
 for i,role in enumerate(('COVER','SHAM')):
  alphas[i]=prop['alpha_'+role]; layers[i][:,alphas[i]]=prop[role][:,alphas[i]]
 return layers,alphas


def geometry_arrays(groups):
 n=len(groups)
 return {'group_index':np.asarray([g['group_index'] for g in groups],np.int64),
  'q8':np.asarray([g['q8'] for g in groups],np.float64),'L8':np.asarray([g['L8'] for g in groups],np.float64),
  'original_available8':np.asarray([g['original_available8'] for g in groups],bool),
  'node_available':np.zeros((n,8),bool),'node_fraction':np.zeros((n,8),np.float32),
  'context_fraction':np.zeros(n,np.float32),'box_xyxy':np.asarray([g['box_xyxy'] if g['box_xyxy'] is not None else [np.nan]*4 for g in groups],np.float64),
  'box_supported':np.zeros(n,bool),'context_supported':np.zeros(n,bool)}


def store_children(arrays,i,result,layers=None,alphas=None):
 """Input availability is independent of weak-label availability."""
 arrays['PAIR_TARGETS.npy'][i]=-1
 for name in ('CF_NODE_RGB96.npy','CF_CONTEXT_RGB224_F32.npy','PAIR_LAYER_RGB96.npy','PAIR_ALPHA_PACKBITS.npy'): arrays[name][i]=0
 constructed=bool(result['constructed']); passed=bool(result['proof'] and result['proof']['PASS'])
 if passed and not constructed: raise ValueError('Proof cannot precede construction')
 arrays['CONSTRUCTED.npy'][i]=constructed;arrays['PROOF_PASS.npy'][i]=passed
 if constructed:
  if layers is None or alphas is None: raise ValueError('Constructed carrier required')
  arrays['PAIR_LAYER_RGB96.npy'][i]=layers;arrays['PAIR_ALPHA_PACKBITS.npy'][i]=np.packbits(alphas.reshape(2,-1),axis=1,bitorder='big')
  for k,role in enumerate(ROLES):
   arrays['CF_NODE_RGB96.npy'][i,k]=result['inputs'][role]['NODE_RGB96']
   arrays['CF_CONTEXT_RGB224_F32.npy'][i,k]=result['inputs'][role]['CONTEXT_RGB224_F32']
  if passed: arrays['PAIR_TARGETS.npy'][i]=[1,0]
 return constructed,passed


def run(entry_path):
 # Entry itself is code-administration metadata, not permission to follow input
 # pointers. False entry refusal precedes all scope/code/data binding reads.
 epath=safe(entry_path)
 with epath.open('rb') as f: entry=json.load(f)
 if entry.get('execution_authorized') is not True: raise PermissionError('Root actual execution entry required; payload access 0')
 if entry.get('training_authorized') is not False: raise ValueError('Data only; no training')
 same(entry['code'],file_binding(__file__))
 same(entry['scope'],SCOPE); scope=load_json(entry['scope'])
 if scope.get('approved_SOURCE_scene_CF_data') is not True: raise ValueError('Root specific actual scope required')
 same(scope['protocol'],PROTOCOL)
 if scope['output_dir']!=entry['output_dir'] or len(scope['payloads'])!=len(entry['payloads']) or {payload_identity(x) for x in scope['payloads']}!={payload_identity(x) for x in entry['payloads']}: raise ValueError('Root closed scope/entry closure')
 docs,closure=validate_metadata(entry); modules=load_math(entry['code_dependencies'])
 math=modules['scene_cf_math']; crop=modules['crop90']; ctx=modules['context107']; comp=modules['v235_compositor']; lift=modules['scene_math']
 out=safe(entry['output_dir'],False)
 if out.parent!=R/'V242_ROOT_SCENE_CF' or out.name!='scene_pairs_v1': raise ValueError('Fixed fresh output path')
 if out.exists(): raise FileExistsError('Immutable output directory already exists')
 out.mkdir()
 try:
  arrays={name:np.lib.format.open_memmap(out/name,mode='w+',dtype=dtype,shape=shape) for name,(dtype,shape) in ARRAYS.items()}
  parents,groups=docs['parents'],docs['groups']; groupmap={g['group_index']:(i,g) for i,g in enumerate(groups)}
  donors={d['donor_id']:d for d in docs['donors']['donors']}; geom=geometry_arrays(groups); saved_groups=set(); decoder=Decoder(closure)
  counts=Counter({k:0 for k in ('attempted','constructed','proof_pass','constructed_unknown','unconstructed','COVER_zero_changed_native','ADJACENT_SHAM_zero_changed_native')}); byfamily={}
  with (out/'PAIR_METADATA.jsonl').open('x',encoding='utf-8') as f:
   for i,parent in enumerate(parents):
    gi,g=groupmap[parent['group_index']]; d=donors[parent['same_V235_donor_id']]; slot=parent['active_slot']
    base=decoder.get(g['image'],raw_hw=g['raw_hw'])
    donor_rgb=decoder.get({'path':d['RGB']['path'],'sha256':d['RGB']['sha256'],'format':'png','member':None})
    donor_mask=decoder.get({'path':d['visible_mask']['path'],'sha256':d['visible_mask']['sha256'],'format':'png','member':None},mask=True)
    result=math.compose_scene_pair(base,g,parent,donor_rgb,donor_mask,d['donor_id'],crop,ctx,comp,lift)
    before=result['inputs']['BEFORE']
    if gi not in saved_groups:
     arrays['GROUP_NODE_VALID96.npy'][gi]=before['NODE_VALID96'];arrays['GROUP_CONTEXT_VALID224.npy'][gi]=before['CONTEXT_VALID224']
     for key in ('node_available','node_fraction','context_fraction','context_supported'): geom[key][gi]=before[key]
     geom['box_supported'][gi]=bool(ctx.valid_box(g['box_xyxy']));saved_groups.add(gi)
    else:
     if not np.array_equal(arrays['GROUP_NODE_VALID96.npy'][gi],before['NODE_VALID96']) or not np.array_equal(arrays['GROUP_CONTEXT_VALID224.npy'][gi],before['CONTEXT_VALID224']): raise AssertionError('Shared factual valid differs')
    arrays['PARENT_INDICES.npy'][i]=parent['parent_sample_index']
    constructed=bool(result['constructed']); passed=bool(result['proof'] and result['proof']['PASS'])
    carrier_sha=[None,None];alpha_sha=[None,None];alpha_mass=[0,0];feature_sha={}
    layers=alphas=None
    if constructed:
     layers,alphas=layer_carriers(result,parent,donor_rgb,donor_mask,comp,math)
     for k,role in enumerate(ROLES):
      carrier_sha[k]=digest_array(layers[k]);alpha_sha[k]=digest_array(alphas[k]);alpha_mass[k]=int(alphas[k].sum())
    store_children(arrays,i,result,layers,alphas)
    for k,role in enumerate(ROLES):
     feature_sha[role]={'NODE_RGB96_sha256':digest_array(arrays['CF_NODE_RGB96.npy'][i,k]),'CONTEXT_RGB224_F32_sha256':digest_array(arrays['CF_CONTEXT_RGB224_F32.npy'][i,k])}
    reason=None if passed else (result.get('failure_reason') or result.get('proposal_failure_reason') or result['status'])
    row={'pair_index':i,'parent_sample_index':parent['parent_sample_index'],'group_index':g['group_index'],'group_cache_index':gi,'group_id':g['group_id'],'active_slot':slot,
     'original_input':parent['original_input'],'image':parent['image'],'teacher_reference_xy':parent['teacher_reference_xy'],'teacher_authority':parent['teacher_authority'],'reference_is_supervision_only':True,
     'donor_id':d['donor_id'],'donor_RGB':d['RGB'],'donor_visible_mask':d['visible_mask'],'same_V235_proposal_plan_sha256':hashlib.sha256(json.dumps(parent['same_V235_donor_plan'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
     'role_order':list(ROLES),'constructed':constructed,'proof_pass':passed,'status':result['status'],'unknown_reason':reason,'targets_D0_E1_Uminus1':arrays['PAIR_TARGETS.npy'][i].tolist(),
     'proposal_success':result.get('proposal_success',False),'proposal_failure_reason':result.get('proposal_failure_reason'),
     'input_available':[constructed,constructed],'target_known':[passed,passed],'expected_BEFORE_RGB96_CHW_sha256':parent['expected_BEFORE_RGB96_CHW_sha256'],'expected_BEFORE_valid96_bool_sha256':parent['expected_BEFORE_valid96_bool_sha256'],
     'actual_BEFORE_RGB96_CHW_sha256':digest_array(before['NODE_RGB96'][slot]),'actual_BEFORE_valid96_bool_sha256':digest_array(before['NODE_VALID96'][slot]),
     'native_scene_sha256':result.get('native_scene_sha256',{'BEFORE':digest_array(base),'COVER':None,'ADJACENT_SHAM':None}),'native_alpha_sha256_bool':{role:digest_array(result['native_alpha'][role]) if constructed else None for role in ROLES},'native_alpha_pixels':{role:int(result['native_alpha'][role].sum()) if constructed else 0 for role in ROLES},'proof':result['proof'],'changed_pixels':result.get('changed_pixels'),
     'carrier_RGB96_CHW_sha256':carrier_sha,'alpha_bool_sha256':alpha_sha,'alpha_pixels':alpha_mass,'alpha_packbits_bitorder':'big','saved_variant_features':feature_sha,
     'old_V235_pair_index':parent['old_V235_pair_index'],'old_V235_success_diagnostic_only':parent['old_V235_success_diagnostic_only'],'old_proof_inherited':False,'original_D_retained':True,
     'new_native_gold':False,'whole_intended_part_correspondence':'UNKNOWN','training_admission':False}
    f.write(json.dumps(math.plain(row),ensure_ascii=False,allow_nan=False)+'\n')
    counts['attempted']+=1;counts['constructed']+=int(constructed);counts['proof_pass']+=int(passed);counts['constructed_unknown']+=int(constructed and not passed);counts['unconstructed']+=int(not constructed)
    for role in ROLES:
     if constructed and result['changed_pixels'][role]['native']==0: counts[role+'_zero_changed_native']+=1
    fam=str(parent['original_input']['family_code']);byfamily.setdefault(fam,Counter());byfamily[fam]['attempted']+=1;byfamily[fam]['constructed']+=int(constructed);byfamily[fam]['proof_pass']+=int(passed)
    if (i+1)%50==0 or i+1==5406: print(json.dumps({'progress':i+1,'total':5406,'counts':dict(counts),'LRU_bytes':decoder.bytes}),flush=True)
    del result
  if len(saved_groups)!=1198: raise AssertionError('Every closed group saved')
  for a in arrays.values(): a.flush()
  np.savez(out/'GROUP_GEOMETRY.npz',**geom)
  outputs={}
  for name,(dtype,shape) in ARRAYS.items(): outputs[name]={**file_binding(out/name),'dtype':dtype,'shape':list(shape)}
  outputs['PAIR_METADATA.jsonl']={**file_binding(out/'PAIR_METADATA.jsonl'),'rows':5406}
  outputs['GROUP_GEOMETRY.npz']={**file_binding(out/'GROUP_GEOMETRY.npz'),'members':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in geom.items()}}
  complete={'completed':True,'PASS':True,'schema':'V242_SCENE_CONSISTENT_CF_DATA_V1','entry':file_binding(epath),'scope':entry['scope'],'protocol':entry['protocol'],'code':entry['code'],'metadata':entry['metadata'],'code_dependencies':entry['code_dependencies'],
   'outputs':outputs,'counts':dict(counts),'by_family':{k:dict(v) for k,v in byfamily.items()},'role_order':list(ROLES),'alpha_packbits_bitorder':'big','group_cache_order':'PARENT_GROUPS_INPUT_ONLY.jsonl order; original sparse group_index saved',
   'BEFORE_active5406_exact_SHA':True,'all_original5406_attempted':True,'constructed_UNKNOWN_inputs_preserved':True,'only_UNCONSTRUCTED_child_inputs_zero':True,'old_success_not_filter':True,
   'decoder':{'raw_image_LRU_max_bytes':decoder.maximum,'max_cached_bytes':decoder.max_cached,'decode_events':sum(decoder.accesses.values()),'NPZ_numeric_members':['RGB_uint8'],'NPZ_member_enumeration':False,'compressed_container_raw_bytes_transit_for_SHA_or_decompression':True,'PIL_version':Image.__version__},
   'training_executed':False,'classifier_or_encoder_executed':False,'native_gold':False,'training_admission':False,'GT_reference_forward_features':False,'SOURCE_DEV_payloads':0,'current_or_held_payloads':0,'automatic_retry':False}
  write_json(out/'COMPLETE.json',complete)
  print(json.dumps({'completed':True,'PASS':True,'COMPLETE':file_binding(out/'COMPLETE.json'),'counts':dict(counts)}),flush=True)
 except BaseException as exc:
  failed=out/'FAILED.json'
  if not failed.exists(): write_json(failed,{'completed':False,'PASS':False,'error_type':type(exc).__name__,'error':str(exc),'automatic_retry':False})
  raise


if __name__=='__main__':
 parser=argparse.ArgumentParser(); parser.add_argument('--entry',required=True)
 run(parser.parse_args().entry)
