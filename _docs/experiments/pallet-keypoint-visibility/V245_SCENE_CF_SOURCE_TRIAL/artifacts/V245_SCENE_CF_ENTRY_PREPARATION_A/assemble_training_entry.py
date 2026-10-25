"""V245 metadata-only late entry assembler. Writes an entry, never runs training."""
import argparse,copy,hashlib,json,os,stat,struct
from pathlib import Path
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation');H=Path(__file__).absolute().parent
ROOT_PROTOCOL={'path':str(R/'V245_ROOT_SCENE_CF_ACTUAL/PROTOCOL_BEFORE_CF_FEATURES_AND_FIXED_SOURCE_FIT.json'),'sha256':'5c5eab43fe3fee635c42647acd05385e8ea9a32156fb7ce04010b9b54315e7c9'}
TEMPLATE={'path':str(R/'V243_SCENE_CF_TRAINER_A/ENTRY_FALSE_TEMPLATE_V2.json'),'sha256':'3c0e1809da30a3bc0a7582aa51cde0e4748d9737175c72ac942accfa7421a42a'}
PLAN={'path':str(H/'plan_v1/PLAN_MANIFEST.json'),'sha256':'4271eb15f9501514dc969050b87e25d806984dc548f6ba57b5c45715a0d0635b'}
PLAN_LEDGER={'path':str(H/'plan_v1/PROSPECTIVE_ORIGINAL_DRAW_AND_MATCHED_SOURCE_LEDGER1000.jsonl'),'sha256':'b2c1c3bff32981b2570f0330353c3c4f54fcfa4fb41d209b0dacfccfc1ac0dae'}
PLAN_SHA='50e547b4c07afdee690727d895278ad7b82fe80eab52551dd00b9f5eb7a5d0e3'
DRAW_SHA='20a1fc74ea15a04a16fe29f8e5b9a114b4629677710ed43c1628b2a1ac00b734'
CF_NAMES=[role+'_'+suffix+'.npy' for role in ('COVER','ADJACENT_SHAM') for suffix in ('BASE1186','NODE_CLS','CONTEXT_CLS')]+['RELATIVE_XY8.npy','PARENT_INDICES.npy','CONSTRUCTED.npy']


def lexical(value):
 p=Path(value)
 if any(t in str(p).casefold() for t in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')) or not p.is_absolute() or '..' in p.parts:raise ValueError('Forbidden lexical path before FS')
 return p


def safe(value):
 p=lexical(value)
 for a in reversed((p,*p.parents)):
  try:mode=os.lstat(a).st_mode
  except FileNotFoundError:continue
  if stat.S_ISLNK(mode):raise ValueError('Symlink ancestor')
 return p


def identity(b):
 if not isinstance(b,dict) or not {'path','sha256'}<=set(b):raise ValueError('Exact metadata binding required')
 p=str(lexical(b['path']));sha=b['sha256']
 if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('Full SHA256 required')
 return p,sha


def normalize(b):
 identity(b);return {k:b[k] for k in ('path','sha256','size_bytes') if k in b}


def exact(a,b):
 if identity(a)!=identity(b):raise ValueError('Exact metadata graph mismatch')


def read(p):
 with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()


def binding(p):
 b=read(p);return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def checked(b):
 identity(b);raw=read(b['path'])
 if hashlib.sha256(raw).hexdigest()!=b['sha256'] or ('size_bytes'in b and len(raw)!=b['size_bytes']):raise ValueError('Bound metadata bytes changed')
 return raw


def load(b):return json.loads(checked(b))


def require_PASS(v):
 if v.get('completed') is not True or v.get('PASS') is not True:raise ValueError('Upstream completed/PASS required')


def unique_bindings(values):
 out=[];seen=set()
 for b in values:
  b=normalize(b);key=identity(b)
  if key not in seen:out.append(b);seen.add(key)
 return out


def verify_sealed_plan(plan,raw):
 require_PASS(plan);exact(plan['ledger'],PLAN_LEDGER)
 if plan['source_plan_sha256']!=PLAN_SHA or plan['draws_sha256']!=DRAW_SHA or plan['updates']!=1000:raise ValueError('Unchanged prefeature Root prospective plan')
 rows=[json.loads(line) for line in raw.splitlines()];draw_hash=hashlib.sha256();h=hashlib.sha256()
 if len(rows)!=1000:raise ValueError('Exactly1000 updates')
 for n,row in enumerate(rows,1):
  if row['update']!=n or row['draws_sha256']!=DRAW_SHA or row['source_plan_sha256']!=PLAN_SHA:raise ValueError('Prospective ledger step/digest')
  for key in ('sample_indices','source_parent_indices','pair_indices','source_roles'):
   if len(row[key])!=32:raise ValueError('Fixed32 slot metadata')
  packed=struct.pack('<32q',*row['sample_indices']);draw_hash.update(packed);h.update(packed)
 if draw_hash.hexdigest()!=DRAW_SHA:raise ValueError('Original draw bytes changed')
 for row in rows:
  original=row['sample_indices'];parent=row['source_parent_indices'];pair=row['pair_indices'];roles=row['source_roles']
  h.update(struct.pack('<32q',*parent));h.update(struct.pack('<32q',*pair));h.update(struct.pack('<32b',*roles))
  for f in (0,1):
   start=f*16
   for j in (*range(4),*range(8,12)):
    i=start+j
    if parent[i]!=original[i] or pair[i]!=-1 or roles[i]!=-1:raise ValueError('First4 natural D/E unchanged')
   for j in range(4):
    d=start+4+j;e=start+12+j
    if pair[d]>=0:
     if pair[d]!=pair[e] or parent[d]!=original[d] or parent[e]!=original[d] or roles[d]!=1 or roles[e]!=0:raise ValueError('Matched SHAM/COVER same drawnD parent')
    elif not (pair[d]==pair[e]==-1 and parent[d]==original[d] and parent[e]==original[e] and roles[d]==roles[e]==-1):raise ValueError('Failed proof retains both natural slots')
 if h.hexdigest()!=PLAN_SHA:raise ValueError('Trainer-compatible semantic plan hash changed')
 return {'completed':True,'PASS':True,'source_plan_sha256':PLAN_SHA,'draws_sha256':DRAW_SHA,'ledger_rows':1000,'all_matched_and_failure_slots_valid':True}


def prepare(config_path):
 cfg=json.loads(read(config_path))
 if cfg.get('approved_to_write_actual_training_entry') is not True:raise PermissionError('Root entry-assembly approval required; this does not execute training')
 exact(cfg['root_protocol'],ROOT_PROTOCOL);root=load(ROOT_PROTOCOL)
 template=load(TEMPLATE);plan=load(PLAN);planQA=verify_sealed_plan(plan,checked(PLAN_LEDGER));scope=load(cfg['root_scope'])
 if scope.get('approved_SOURCE_scene_CF_training') is not True or scope.get('current_or_held_payloads_authorized') is not False:raise ValueError('Root SOURCE fit scope required')
 exact(scope['prospective_plan_manifest'],PLAN);exact(scope['prospective_training_ledger'],PLAN_LEDGER)
 if scope['prospective_source_plan_sha256']!=PLAN_SHA:raise ValueError('Root scope binds prefeature plan')
 exact(root['pins']['trainer'],template['code']);exact(root['pins']['training_implementation_protocol'],template['implementation_protocol']);exact(root['pins']['baseline_complete'],template['baseline_complete'])
 if root['fit_step']['original_draws_sha256']!=DRAW_SHA or root['fit_step']['updates']!=1000 or root['fit_step']['threshold_logit']!=0:raise ValueError('Fixed original recipe and no new schedule')
 data_b=root['pins']['data_complete'];data=load(data_b);require_PASS(data);dqa_b=root['pins']['data_QA'];dqa=load(dqa_b);require_PASS(dqa);exact(dqa['dataset_complete'],data_b)
 cqa_b=root['pins']['trainer_CODE_QA'];cqa=load(cqa_b);require_PASS(cqa)
 # This admission metadata is bound, not converted into any forward feature.
 visual_b=root['pins']['data_visual_admission'];load(visual_b)
 fcomplete_b=cfg['CF_feature_complete'];fqa_b=cfg['CF_feature_QA'];fc=load(fcomplete_b);fq=load(fqa_b);require_PASS(fc);require_PASS(fq)
 exact(fq['feature_complete'],fcomplete_b);exact(fc['data_complete'],data_b);exact(fc['data_QA'],dqa_b)
 if 'dataset_complete'in fq:exact(fq['dataset_complete'],data_b)
 if 'data_QA'in fq:exact(fq['data_QA'],dqa_b)
 exact(fc['code'],root['pins']['feature_producer'])
 if fc['schema']!='V242_CANONICAL_CF_FEATURES_V1' or fc['rows']!=5406 or fc['all5406_original_order_retained'] is not True:raise ValueError('All5406 fixed canonical CF packet')
 if set(fc['features'])!=set(CF_NAMES):raise ValueError('Exact9 CF vector/index outputs')
 for role in ('COVER','ADJACENT_SHAM'):
  q=fc['canonical_sharing_QA'][role]
  if any(q[key] is not True for key in ('active_first384_byte_exact','original_condition32_byte_exact','known_fractions_byte_exact')) or q['UNCONSTRUCTED_feature_nonzeros']!=0:raise ValueError('Required byte-sharing producer QA')
 for name in ('PARENT_INDICES.npy','CONSTRUCTED.npy'):
  if fc['features'][name]['sha256']!=data['outputs'][name]['sha256']:raise ValueError('CF index/constructed unchanged from dataset')
 e=copy.deepcopy(template);e['approved_scene_CF_source_training']=True;e['root_protocol']=normalize(ROOT_PROTOCOL);e['root_scope']=normalize(cfg['root_scope'])
 e['CF_data_complete']=normalize(data_b);e['CF_data_QA']=normalize(dqa_b);e['CF_feature_complete']=normalize(fcomplete_b);e['CF_feature_QA']=normalize(fqa_b)
 e['CF_features']={name:normalize(fc['features'][name]) for name in CF_NAMES};e['CF_proof_inputs']={name:normalize(data['outputs'][name]) for name in ('PROOF_PASS.npy','PAIR_TARGETS.npy')}
 exact(fc['samples_npz'],e['samples_npz'])
 e['device']=cfg.get('device','cuda')
 if e['device'] not in ('cpu','cuda'):raise ValueError('Explicit head device cpu/cuda')
 # Extra entry bindings are for independent post-run validation. Frozen trainer V2 is unchanged.
 e['prospective_plan_manifest']=normalize(PLAN);e['prospective_training_ledger']=normalize(PLAN_LEDGER);e['prospective_source_plan_sha256']=PLAN_SHA
 e['plan_admission_policy']='Root must compare actual source_plan_sha256 and whole actual ledger to sealed plan before scientific/model admission; mismatch invalidates run, no automatic retry.'
 e['metadata_whitelist']=unique_bindings([*e['metadata_whitelist'],ROOT_PROTOCOL,cfg['root_scope'],PLAN,PLAN_LEDGER,data_b,dqa_b,visual_b,cqa_b,fcomplete_b,fqa_b])
 e['payload_whitelist']=unique_bindings([*e['original_features'].values(),*e['CF_features'].values(),*e['CF_proof_inputs'].values(),e['samples_npz'],e['geometry_npz'],e['baseline_checkpoint'],e['baseline_log'],e['baseline_CSV']])
 e.pop('instructions',None)
 out=safe(cfg['output_entry'])
 if out.parent!=R/'V245_ROOT_SCENE_CF_ACTUAL' or out.suffix!='.json' or out.exists():raise ValueError('Fresh Root V245 entry only')
 with os.fdopen(os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:json.dump(e,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 receipt=out.with_name(out.stem+'_ASSEMBLY_RECEIPT.json')
 report={'completed':True,'PASS':True,'entry':binding(out),'config':binding(config_path),'helper':binding(Path(__file__).absolute()),'plan_QA':planQA,'plan_manifest':PLAN,'plan_ledger':PLAN_LEDGER,'prospective_source_plan_sha256':PLAN_SHA,'late_CF_feature_complete':fcomplete_b,'late_CF_feature_QA':fqa_b,'metadata_only':True,'numerical_feature_sample_weight_reads':0,'training_executed':False,'trainer_V2_modified':False,'automatic_retry':False}
 with receipt.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'completed':True,'PASS':True,'entry':binding(out),'receipt':binding(receipt),'training_executed':False}),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);prepare(p.parse_args().config)
