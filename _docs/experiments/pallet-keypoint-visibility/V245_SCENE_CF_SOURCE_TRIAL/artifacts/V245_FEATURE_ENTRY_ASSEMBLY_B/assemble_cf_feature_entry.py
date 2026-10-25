"""Metadata-only assembler for the frozen V242 CF canonical feature producer.

Never stats/hashes/opens array, sample, official-weight or other payload paths.
Root executes this helper separately with an explicit true metadata plan.
"""
import argparse,copy,hashlib,json,os,stat
from pathlib import Path

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V245_FEATURE_ENTRY_ASSEMBLY_B'
ROOT=R/'V245_ROOT_SCENE_CF_ACTUAL'
PROTOCOL={'path':str(ROOT/'PROTOCOL_BEFORE_CF_FEATURES_AND_FIXED_SOURCE_FIT.json'),'sha256':'5c5eab43fe3fee635c42647acd05385e8ea9a32156fb7ce04010b9b54315e7c9'}
READY={'path':str(R/'V242_CF_FEATURE_PRODUCER_B/FINAL_READY_V2.json'),'sha256':'e046f3491fd2c60784bca76cd61ff4394f4d86886448af158081b32c16949f5a'}
TEMPLATE={'path':str(R/'V242_CF_FEATURE_PRODUCER_B/ROOT_FEATURE_ENTRY_FALSE_TEMPLATE_V2.json'),'sha256':'eb0d973ad663d2ab439c316e0e2fd6efc39028450a5026444219417f997aba9f'}
PRODUCER={'path':str(R/'V242_CF_FEATURE_PRODUCER_B/build_cf_features_v2.py'),'sha256':'c025749399f86fe6c0f09fed6ef1c2510bb44a45fc61a196cecb0446864f4335'}
OUTPUT=str(R/'V242_ROOT_SCENE_CF/cf_canonical_features_v1')
DATA_NAMES={'node_rgb':'CF_NODE_RGB96.npy','context_rgb':'CF_CONTEXT_RGB224_F32.npy','parent_indices':'PARENT_INDICES.npy','constructed':'CONSTRUCTED.npy','node_valid':'GROUP_NODE_VALID96.npy','context_valid':'GROUP_CONTEXT_VALID224.npy','geometry':'GROUP_GEOMETRY.npz','pair_metadata':'PAIR_METADATA.jsonl'}


def lexical(value):
 s=str(value)
 if any(t in s.casefold() for t in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')):raise ValueError('Forbidden before filesystem access')
 p=Path(s)
 if not p.is_absolute() or '..' in p.parts:raise ValueError('Absolute path without traversal required')
 return p


def safe_metadata(value):
 p=lexical(value)
 for q in reversed((p,*p.parents)):
  try:mode=os.lstat(q).st_mode
  except FileNotFoundError:continue
  if stat.S_ISLNK(mode):raise ValueError('Symlink metadata path forbidden')
 return p


def identity(b):
 if not isinstance(b,dict) or not {'path','sha256'}<=set(b)<={'path','sha256','size_bytes'}:raise ValueError('Exact binding keys')
 path=str(lexical(b['path']));sha=b['sha256']
 if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('SHA256 format')
 if 'size_bytes' in b and (type(b['size_bytes']) is not int or b['size_bytes']<0):raise ValueError('Nonnegative integer size')
 return path,sha


def bare(b):return {k:b[k] for k in ('path','sha256','size_bytes') if k in b}


def same(a,b):
 if identity(a)!=identity(b):raise ValueError('Exact binding identity differs')


def checked_json(b,pin=None):
 identity(b)
 if pin is not None:same(b,pin)
 p=safe_metadata(b['path'])
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:raw=f.read()
 if hashlib.sha256(raw).hexdigest()!=b['sha256'] or ('size_bytes' in b and len(raw)!=b['size_bytes']):raise ValueError('Metadata bytes differ')
 return json.loads(raw)


def union_bindings(*lists):
 """Lexical identity only: payload paths are not statted or followed."""
 out={};paths={}
 for values in lists:
  for raw in values:
   b=bare(raw);key=identity(b)
   if key[0] in paths and paths[key[0]]!=key[1]:raise ValueError('Conflicting path SHA')
   paths[key[0]]=key[1]
   if key in out and 'size_bytes' in b and 'size_bytes' in out[key] and b['size_bytes']!=out[key]['size_bytes']:raise ValueError('Conflicting declared byte size')
   if key not in out or 'size_bytes' in b:out[key]=b
 return list(out.values())


def construct_entry(template,plan,ready,dataset,qa,scope):
 if template.get('execution_authorized') is not False:raise ValueError('Frozen template must be false')
 same(ready['code'],PRODUCER);same(ready['false_entry'],TEMPLATE);same(template['code'],PRODUCER)
 if any(obj.get('completed') is not True or obj.get('PASS') is not True for obj in (ready,dataset,qa)):raise ValueError('Completed preparation/data/QA required')
 if scope.get('approved_SOURCE_CF_feature_build') is not True:raise ValueError('Root runtime feature-scope key required')
 same(scope['prospective_draw_plan'],plan['prospective_draw_plan'])
 if dataset['schema']!='V242_SCENE_CONSISTENT_CF_DATA_V1' or dataset['counts']['attempted']!=5406 or dataset['all_original5406_attempted'] is not True:raise ValueError('All5406 fixed source parent dataset')
 same(qa['dataset_complete'],plan['data_complete'])
 e=copy.deepcopy(template)
 if e['samples_npz']['sha256']!='837bbe7ff406e27c6ed2981493a83c2adab830cd5b2de498e4a48de456456e99' or e['model_loader_config']['official_state']['sha256']!='f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb':raise ValueError('Original samples/official config pins')
 if e['batch_size']!=32 or e['source_only'] is not True or e['training_authorized'] is not False or e['output_directory']!=OUTPUT:raise ValueError('No recipe/output change')
 e['execution_authorized']=True
 for name in ('root_protocol','root_scope','data_complete','data_QA'):e[name]=bare(plan[name])
 e['data_admission']=bare(plan['data_admission'])
 e['prospective_draw_plan']=bare(plan['prospective_draw_plan'])
 e['data_inputs']={k:bare(dataset['outputs'][name]) for k,name in DATA_NAMES.items()}
 cfg=e['model_loader_config']
 codes=[e[k] for k in ('code','implementation_protocol','canonical_producer_code','model_loader_code','crop_module','preprocess_CODE','context_preprocess_CODE')]
 # Preserve all inherited pins and explicitly close nested loader/vendor imports.
 codes+=cfg['code_whitelist']+cfg['backbone_code_bindings']+[cfg[k] for k in ('backbone_factory','prefix_runtime','adapter_module','crop_module')]
 e['code_whitelist']=union_bindings(e['code_whitelist'],codes)
 e['payload_whitelist']=union_bindings(list(e['data_inputs'].values()),[e['samples_npz'],cfg['official_state']])
 e['metadata_whitelist']=union_bindings([e[k] for k in ('root_protocol','root_scope','data_complete','data_QA','data_admission','prospective_draw_plan')])
 if not {identity(b) for b in cfg['code_whitelist']}<={identity(b) for b in e['code_whitelist']}:raise AssertionError('Nested loader closure')
 if {identity(b) for b in e['payload_whitelist']}!={identity(b) for b in [*e['data_inputs'].values(),e['samples_npz'],cfg['official_state']]}:raise AssertionError('Exact10 payload closure')
 if e['model_loader_config']!=template['model_loader_config']:raise AssertionError('Original loader config unchanged')
 return e


def write_json(path,obj):
 p=safe_metadata(path)
 if p.parent!=ROOT:raise ValueError('Root V245 metadata output folder only')
 with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 raw=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}


def run(plan_path):
 p=safe_metadata(plan_path)
 with p.open('rb') as f:plan=json.load(f)
 if plan.get('assembly_authorized') is not True:raise PermissionError('Root metadata assembly plan required')
 ready=checked_json(READY);template=checked_json(TEMPLATE)
 # Actual producer CODE is admin text; never import or execute it here.
 pp=safe_metadata(PRODUCER['path']);raw=pp.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=PRODUCER['sha256']:raise ValueError('Frozen producer CODE changed')
 same(plan['root_protocol'],PROTOCOL)
 docs={k:checked_json(plan[k]) for k in ('root_protocol','root_scope','data_complete','data_QA','data_admission','prospective_draw_plan')}
 if not isinstance(docs['prospective_draw_plan'],dict):raise ValueError('Frozen prospective draw plan must be metadata JSON')
 admission=docs['data_admission']
 if not isinstance(admission,dict):raise ValueError('Root admission must be metadata dict')
 for k,expected in plan.get('admission_expected_fields',{}).items():
  if admission.get(k)!=expected:raise ValueError('Root specified admission field differs')
 e=construct_entry(template,plan,ready,docs['data_complete'],docs['data_QA'],docs['root_scope'])
 result=write_json(plan['entry_output_path'],e)
 receipt={'completed':True,'PASS':True,'actual_feature_entry':result,'producer':PRODUCER,'ready':READY,'template':TEMPLATE,'Root_metadata_bindings':{k:plan[k] for k in docs},'code_whitelist_count':len(e['code_whitelist']),'payload_whitelist_count':len(e['payload_whitelist']),'metadata_whitelist_count':len(e['metadata_whitelist']),'nested_loader_whitelist_closed':True,'original_loader_config_unchanged':True,'original_feature_output_directory_unchanged':OUTPUT,'payload_leaf_stat_hash_header_or_body_access':0,'official_model_or_encoder_or_training':0,'feature_output_freshness_checked_here':False,'feature_execution_performed':False,'automatic_retry':False}
 rb=write_json(plan['receipt_output_path'],receipt)
 print(json.dumps({'completed':True,'PASS':True,'entry':result,'receipt':rb}),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);run(p.parse_args().plan)
