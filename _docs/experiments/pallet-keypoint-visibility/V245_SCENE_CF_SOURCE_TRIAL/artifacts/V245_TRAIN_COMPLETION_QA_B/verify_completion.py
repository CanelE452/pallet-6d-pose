"""Independent saved V243 SOURCE trial QA. No training, encoder or raw inputs.

Default scope: new head CPU weights_only + opaque baseline + logs/ledger/CSV.
Optional Root-approved CPU saved-head replay admits only six canonical arrays
and original input geometry; no DINO, prefix, RGB, GT or protected data.
"""
import argparse,csv,hashlib,io,json,math,os,stat,traceback,types
from pathlib import Path
import numpy as np
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V245_TRAIN_COMPLETION_QA_B'
ARMS=('BASELINE_V239_CANONICAL_CONTEXT','TREAT_SCENE_CF_MATCHED_DATA')
OLD_ARM='TREAT_CANONICAL_CONTEXT'
INITIAL_BASE_SHA='d1b8da9721a5492dcfb166cf78943eb5db18c5b14b8e8fed328ccab31c12648c'
INITIAL_FULL_SHA='196bf2dfb05f79658af04bbc753bbb377a3d5e51bb65eb4135da4d1c7faaea7d'
DRAW_SHA='20a1fc74ea15a04a16fe29f8e5b9a114b4629677710ed43c1628b2a1ac00b734'
PLAN_SHA='50e547b4c07afdee690727d895278ad7b82fe80eab52551dd00b9f5eb7a5d0e3'
LEDGER_SHA='b2c1c3bff32981b2570f0330353c3c4f54fcfa4fb41d209b0dacfccfc1ac0dae'
CODE_SHA='e18e6e601742dd2ad52cfbd417b5209852375d8a3d163687c7c2f0eb7b2d110d'
PROTOCOL_SHA='5c5eab43fe3fee635c42647acd05385e8ea9a32156fb7ce04010b9b54315e7c9'
OLD_COMPLETE={'path':str(R/'V239_ROOT_CANONICAL_CONTEXT/canonical_pair_source_fit_v1/COMPLETE.json'),'sha256':'1fd6880528d1bd440376c68c7079b9cbee59ce736f006fdaef4244e3310a4df9'}
PLAN_MANIFEST={'path':str(R/'V245_SCENE_CF_ENTRY_PREPARATION_A/plan_v1/PLAN_MANIFEST.json'),'sha256':'4271eb15f9501514dc969050b87e25d806984dc548f6ba57b5c45715a0d0635b'}
MODULE={'path':str(R/'V237_RELATIONAL_QUERY_DESIGN_B/relational_query_head.py'),'sha256':'dfd26ab7ccfa7037f0c3dca548d59c8574aa8b78c50699f70f01e03bf48935a9'}
BASE_SHAPES={'head.0.weight':(64,1186),'head.0.bias':(64,),'head.2.weight':(2,64),'head.2.bias':(2,)}
EXTRA_SHAPES={'relational.project.weight':(16,384),'relational.geometry_bias.weight':(1,2),'relational.readout.0.weight':(32,116),'relational.readout.0.bias':(32,),'relational.readout.2.weight':(2,32),'relational.readout.2.bias':(2,)}
FULL_SHAPES={**{'base.'+k:v for k,v in BASE_SHAPES.items()},**EXTRA_SHAPES}
FEATURE_NAMES=('ORIGINAL_ALL_FEATURES.npy','NODE_CLS.npy','CONTEXT_CLS.npy','SAMPLE_GROUP_INDEX.npy','SAMPLE_NODE_INDEX.npy','RELATIVE_XY8.npy')

def state_sha(state):
 h=hashlib.sha256()
 for name in sorted(state):
  a=state[name].detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes())
 return h.hexdigest()

def check_state(state,shapes):
 assert set(state)==set(shapes)
 for key,shape in shapes.items():
  a=state[key].detach().cpu().numpy();assert a.dtype==np.float32 and tuple(a.shape)==shape and np.isfinite(a).all()
 return sum(v.numel() for v in state.values())

def check_optimizer(cp,shapes):
 opt=cp['optimizer_state_dict'];assert len(opt['param_groups'])==1
 g=opt['param_groups'][0];ids=g['params'];assert len(ids)==len(shapes) and set(ids)==set(opt['state'])
 assert g['lr']==.001 and g['weight_decay']==.0001 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8
 for pid,shape in zip(ids,shapes.values()):
  s=opt['state'][pid];assert float(s['step'])==1000
  for k in ('exp_avg','exp_avg_sq'):
   a=s[k].detach().cpu().numpy();assert a.dtype==np.float32 and tuple(a.shape)==shape and np.isfinite(a).all()
 return len(ids)

def parse_csv(text):
 rows=[]
 for i,r in enumerate(csv.DictReader(io.StringIO(text))):
  assert int(r['sample_index'])==i and r['available'] in ('True','False')
  f=int(r['family']);y=int(r['target']);role=r['role'];available=r['available']=='True'
  assert f in (0,1) and y in (0,1) and role in ('FIT','DEV')
  z=np.asarray([float(r['logit0']),float(r['logit1'])],np.float32);assert np.isfinite(z).all()
  value=float(r['selected_family_logit']);assert value==float(r['logit'+str(f)])
  prediction='U' if not available else ('E' if value>=0 else 'D');assert r['prediction']==prediction
  rows.append({'sample_index':i,'row_id':r['row_id'],'family':f,'target':y,'role':role,'available':available,'prediction':prediction,'logits':z})
 assert len(rows)==7617 and len({r['row_id'] for r in rows})==7617 and all(r['available'] for r in rows)
 assert sum(r['role']=='FIT' for r in rows)==5876 and sum(r['role']=='DEV' for r in rows)==1741
 return rows

def replay_draws(rows,steps=1000):
 pools=[np.asarray([r['sample_index'] for r in rows if r['role']=='FIT' and r['available'] and r['family']==f and r['target']==y],np.int64) for f in (0,1) for y in (0,1)]
 assert all(len(p) for p in pools)
 rng=np.random.Generator(np.random.PCG64(17));draw=np.stack([np.concatenate([rng.choice(p,8,replace=True) for p in pools]) for _ in range(steps)])
 return draw,hashlib.sha256(draw.astype('<i8').tobytes()).hexdigest(),[len(p) for p in pools]

def lexical(value):
 s=str(value);assert not any(t in s.casefold() for t in ('v112','__pycache__','.pyc','current223','publicheld','private_evaluation','oldval','v177','v178','source_holdout','v182'))
 p=Path(s);assert p.is_absolute() and '..' not in p.parts;return p

def safe(value):
 p=lexical(value)
 for q in reversed((p,*p.parents)):
  try:mode=os.lstat(q).st_mode
  except FileNotFoundError:continue
  assert not stat.S_ISLNK(mode)
 return p

def raw(path):
 with os.fdopen(os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def identity(b):
 assert isinstance(b,dict) and {'path','sha256'}<=set(b)<={'path','sha256','size_bytes'}
 p=str(lexical(b['path']));d=b['sha256'];assert isinstance(d,str) and len(d)==64 and all(c in '0123456789abcdef' for c in d)
 if 'size_bytes' in b:assert type(b['size_bytes']) is int and b['size_bytes']>=0
 return p,d

def same(a,b):assert identity(a)==identity(b)

def parse_CF(text,baseline):
 rows=list(csv.DictReader(io.StringIO(text)));assert len(rows)==10812;pairs={}
 for i,r in enumerate(rows):
  role='COVER' if i<5406 else 'ADJACENT_SHAM';p=i%5406;parent=int(r['parent_sample_index']);old=baseline[parent]
  assert int(r['pair_index'])==p and r['variant']==role and r['role']=='FIT' and r['native_gold']=='False'
  assert old['role']=='FIT' and old['target']==0 and old['available'] and r['row_id']==old['row_id'] and int(r['family'])==old['family']
  assert r['constructed'] in ('True','False') and r['proof_PASS'] in ('True','False')
  made=r['constructed']=='True';proof=r['proof_PASS']=='True';assert not proof or made
  assert int(r['target'])==((1 if role=='COVER' else 0) if proof else -1)
  if made:
   z=[float(r[k]) for k in ('logit0','logit1')];assert all(math.isfinite(v) for v in z)
   value=float(r['selected_family_logit']);assert value==z[old['family']] and r['prediction']==('E' if value>=0 else 'D')
  else:assert not any(r[k] for k in ('logit0','logit1','selected_family_logit')) and r['prediction']=='U'
  item={'parent':parent,'family':old['family'],'constructed':made,'proof':proof}
  if role=='COVER':assert p not in pairs;pairs[p]=item
  else:assert pairs[p]==item
 assert len({v['parent'] for v in pairs.values()})==5406
 counts={'constructed':sum(v['constructed'] for v in pairs.values()),'proof_PASS':sum(v['proof'] for v in pairs.values()),'unconstructed':sum(not v['constructed'] for v in pairs.values())}
 assert counts=={'constructed':5367,'proof_PASS':5367,'unconstructed':39}
 assert [sum(v['family']==f and v['proof'] for v in pairs.values()) for f in (0,1)]==[4537,830]
 return pairs,counts

def audit_ledger(ledger,logs,baseline,pairs,manifest):
 draws,digest,pools=replay_draws(baseline);assert digest==DRAW_SHA and pools==[4565,187,841,283] and len(ledger)==len(logs)==1000
 h=hashlib.sha256();h.update(draws.astype('<i8').tobytes());matched=[0,0];failed=[0,0]
 for i,(row,log,ids) in enumerate(zip(ledger,logs,draws),1):
  assert row['update']==log['update']==i and row['sample_indices']==log['sample_indices']==ids.tolist()
  assert row['draws_sha256']==log['draws_sha256']==DRAW_SHA and row['source_plan_sha256']==log['source_plan_sha256']==PLAN_SHA
  assert row['all_fit_available_known'] is True and log['all_fit_available_known'] is True and log['family_counts']==[16,16]
  parent=np.asarray(row['source_parent_indices'],np.int64);pair=np.asarray(row['pair_indices'],np.int64);role=np.asarray(row['source_roles'],np.int8)
  assert parent.shape==pair.shape==role.shape==(32,)
  assert log['source_parent_indices']==parent.tolist() and log['pair_indices']==pair.tolist() and log['source_roles']==role.tolist()
  for f in (0,1):
   off=f*16
   for j in list(range(4))+list(range(8,12)):assert parent[off+j]==ids[off+j] and pair[off+j]==role[off+j]==-1
   for j in range(4):
    di,ei=off+4+j,off+12+j
    if pair[di]>=0:
     assert pair[di]==pair[ei] and role[di]==1 and role[ei]==0 and parent[di]==parent[ei]==ids[di]
     p=pairs[int(pair[di])];assert p['proof'] and p['parent']==int(ids[di]) and p['family']==f;matched[f]+=1
    else:assert parent[di]==ids[di] and parent[ei]==ids[ei] and pair[di]==pair[ei]==role[di]==role[ei]==-1;failed[f]+=1
  assert all(baseline[int(p)]['role']=='FIT' and baseline[int(p)]['available'] for p in parent)
  assert [baseline[int(p)]['family'] for p in parent]==[baseline[int(p)]['family'] for p in ids]
  h.update(parent.astype('<i8').tobytes());h.update(pair.astype('<i8').tobytes());h.update(role.tobytes())
  x,y,z=(float(log[k]) for k in ('family0_BCE','family1_BCE','total_loss'))
  assert all(math.isfinite(v) and v>=0 for v in (x,y,z)) and math.isclose(z,.5*(x+y),rel_tol=1e-6,abs_tol=1e-6)
 assert h.hexdigest()==PLAN_SHA and matched==[3974,3940] and failed==[26,60]
 assert manifest['source_plan_sha256']==PLAN_SHA and manifest['draws_sha256']==DRAW_SHA and manifest['updates']==1000
 for f in (0,1):assert manifest['family_counts'][str(f)]['matched_pairs_in_optimizer_draws']==matched[f] and manifest['family_counts'][str(f)]['failed_proposals_both_natural']==failed[f]
 return {'updates':1000,'draws_sha256':digest,'source_plan_sha256':h.hexdigest(),'FIT_class_pools':pools,'matched_by_family':matched,'failed_both_natural_by_family':failed}

def CPU_replay(e,cp,rows,old,checked):
 assert e['allow_saved_head_cpu_replay'] is True and e['cpu_atol']==e['cpu_rtol']==1e-5
 spec=e['cpu_replay'];same(spec['module'],MODULE);assert set(spec['features'])==set(FEATURE_NAMES)
 for name in FEATURE_NAMES:same(spec['features'][name],old['features'][name])
 same(spec['geometry'],old['geometry_npz'])
 import torch
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 p=checked(spec['module']);mod=types.ModuleType('v245_cpu_head');mod.__file__=str(p);exec(compile(raw(p),str(p),'exec'),mod.__dict__)
 head=mod.RelationalQueryHead().float();head.load_state_dict(cp['head_state_dict'],strict=True);head.eval().requires_grad_(False);before=state_sha(head.state_dict())
 f={name:np.load(checked(b),mmap_mode='r',allow_pickle=False) for name,b in spec['features'].items()}
 shapes=((7617,1186),(1767,8,384),(1767,384),(7617,),(7617,),(7617,8,2))
 for name,shape in zip(FEATURE_NAMES,shapes):assert f[name].shape==shape and f[name].dtype==(np.int64 if 'INDEX' in name else np.float32)
 with np.load(checked(spec['geometry']),allow_pickle=False) as z:g={k:z[k] for k in ('original_available8','node_available','context_fraction')}
 assert g['original_available8'].shape==g['node_available'].shape==(1767,8) and g['original_available8'].dtype==g['node_available'].dtype==bool
 assert g['context_fraction'].shape==(1767,) and g['context_fraction'].dtype==np.float32
 maximum=0.;count=0;attention_checks=0
 with torch.no_grad():
  for start in range(0,7617,64):
   ids=np.arange(start,min(start+64,7617));group=f['SAMPLE_GROUP_INDEX.npy'][ids];slot=f['SAMPLE_NODE_INDEX.npy'][ids]
   assert ((group>=0)&(group<1767)).all() and ((slot>=0)&(slot<8)).all()
   x=np.array(f['ORIGINAL_ALL_FEATURES.npy'][ids],copy=True);nodes=np.array(f['NODE_CLS.npy'][group],copy=True)
   assert x[:,:384].tobytes()==nodes[np.arange(len(ids)),slot].tobytes()
   args=(x,nodes,np.array(f['CONTEXT_CLS.npy'][group],copy=True),np.array(slot,copy=True),g['original_available8'][group],g['node_available'][group],g['context_fraction'][group]>0,np.array(f['RELATIVE_XY8.npy'][ids],copy=True))
   tensors=tuple(torch.from_numpy(np.array(a,copy=True)) for a in args)
   alignment=mod.verify_active_node_alignment(tensors[0],tensors[1],tensors[3],tensors[4],tensors[5],atol=1e-6);assert alignment['checked']==len(ids)
   logits,detail=head(*tensors);actual=logits.numpy();expected=np.stack([rows[int(i)]['logits'] for i in ids])
   attention=detail['attention_other7'].numpy();other=g['original_available8'][group]&g['node_available'][group];other[np.arange(len(ids)),slot]=False
   assert not np.count_nonzero(attention[~other]) and np.allclose(attention.sum(1),other.any(1).astype(np.float32),rtol=0,atol=1e-6);attention_checks+=len(ids)
   assert np.isfinite(actual).all() and np.allclose(actual,expected,atol=1e-5,rtol=1e-5)
   maximum=max(maximum,float(np.max(np.abs(actual-expected))))
   for i,z in zip(ids,actual):assert ('E' if z[rows[int(i)]['family']]>=0 else 'D')==rows[int(i)]['prediction'];count+=1
 assert count==7617 and before==state_sha(head.state_dict())
 return {'PASS':True,'predictions_exact':count,'binary_logits':15234,'max_abs_error':maximum,'atol':1e-5,'rtol':1e-5,'active_NODE_byte_exact':7617,'attention_gate_and_sum_rows':attention_checks,'batch_size':64,'head_state_unchanged':True,'DINO_or_training':0}

def run(entry_path):
 ep=safe(entry_path);e=json.loads(raw(ep))
 assert e.get('approved_actual_completion_QA') is True and e.get('new_final1000') is True and e.get('source_only') is True and e.get('automatic_retry') is False
 same(e['verifier'],{'path':str(safe(__file__)),'sha256':hashlib.sha256(raw(__file__)).hexdigest()})
 white={identity(b):b for b in e['whitelist']};assert len(white)==len(e['whitelist']);access=[];allow_arrays=set()
 if e.get('allow_saved_head_cpu_replay') is True:allow_arrays={identity(b) for b in [*e['cpu_replay']['features'].values(),e['cpu_replay']['geometry']]}
 out=safe(e['output_directory']);assert out.parent==H;out.mkdir(exist_ok=False)
 def checked(b):
  key=identity(b);assert key in white;p=safe(b['path'])
  assert p.suffix.lower() in ('.json','.jsonl','.csv','.pt') or key in allow_arrays or key==identity(MODULE)
  h=hashlib.sha256();n=0
  with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
   for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block);n+=len(block)
  assert h.hexdigest()==b['sha256']
  for bb in (b,white[key]):
   if 'size_bytes' in bb:assert bb['size_bytes']==n
  access.append({'path':str(p),'sha256':b['sha256'],'size_bytes':n});return p
 def js(b):return json.loads(raw(checked(b)))
 def completed(b):
  obj=js(b);assert obj['completed'] is True and obj['PASS'] is True;return obj
 try:
  c=completed(e['complete']);root=js(e['root_fit_entry']);same(c['entry'],e['root_fit_entry'])
  assert c['schema']=='V243_SCENE_CF_DATA_ONLY_HEAD_TRIAL_V1' and c['code']['sha256']==CODE_SHA and c['root_protocol']['sha256']==PROTOCOL_SHA
  same(e['baseline_complete'],OLD_COMPLETE);old=completed(e['baseline_complete']);same(c['baseline_complete'],e['baseline_complete'])
  same(e['prospective_plan'],PLAN_MANIFEST);manifest=completed(e['prospective_plan']);assert manifest['ledger']['sha256']==LEDGER_SHA
  same(root['prospective_plan_manifest'],e['prospective_plan']);assert root['prospective_training_ledger']['sha256']==LEDGER_SHA and root['prospective_source_plan_sha256']==PLAN_SHA
  same(c['root_protocol'],root['root_protocol']);same(c['root_scope'],root['root_scope'])
  protocol=js(c['root_protocol']);scope=js(c['root_scope'])
  assert protocol['pins']['trainer']['sha256']==CODE_SHA and protocol['fit_step']['updates']==1000 and protocol['fit_step']['threshold_logit']==0
  assert scope['approved_SOURCE_scene_CF_training'] is True and scope['current_or_held_payloads_authorized'] is False
  same(scope['prospective_plan_manifest'],e['prospective_plan']);assert scope['prospective_training_ledger']['sha256']==LEDGER_SHA and scope['prospective_source_plan_sha256']==PLAN_SHA
  assert root['source_only'] is True and root['automatic_retry'] is False and root['approved_scene_CF_source_training'] is True
  recipe=root['fixed_recipe'];assert recipe['steps']==1000 and recipe['seed']==recipe['draw_seed']==17 and recipe['batch_size']==32 and recipe['head_parameters']==86054 and recipe['head_lr']==.001 and recipe['weight_decay']==.0001 and recipe['threshold_logit']==0 and recipe['use_lora'] is False
  assert c['new_optimizer_updates']=={ARMS[0]:0,ARMS[1]:1000} and c['head_parameters']==dict.fromkeys(ARMS,86054) and c['new_optimizer_state_tensors']=={ARMS[0]:0,ARMS[1]:10}
  assert c['new_DEV_head_output_before_finalfreeze']==0 and c['DINO_LoRA_RGB_GT_protected_payload_reads']==0 and c['baseline_checkpoint_numerical_loads']==0 and c['native_gold'] is False and c['current_admission'] is False and c['automatic_retry'] is False
  assert c['draws_sha256']==DRAW_SHA and c['source_plan_sha256']==PLAN_SHA and c['same_initial_base_sha256']==INITIAL_BASE_SHA and c['initial_full_head_sha256']==INITIAL_FULL_SHA
  for k in ('original_feature_complete','CF_data_complete','CF_data_QA','CF_feature_complete','CF_feature_QA','samples_npz','geometry_npz','head_module'):same(c[k],root[k])
  for k in ('original_features','CF_features','CF_proof_inputs'):
   assert set(c[k])==set(root[k])
   for name in c[k]:same(c[k][name],root[k][name])
  for name in FEATURE_NAMES:same(c['original_features'][name],old['features'][name])
  same(c['head_module'],MODULE)
  reuse=completed(c['baseline_reuse']);freeze=completed(c['freeze']);qa=completed(c['global_feature_QA'])
  assert reuse['new_optimizer_updates']==reuse['numerical_checkpoint_loads']==reuse['baseline_scores_interpreted_before_new_finalfreeze']==0
  assert freeze['all_final_updates']==dict.fromkeys(ARMS,1000) and freeze['new_optimizer_updates']==c['new_optimizer_updates'] and freeze['new_DEV_head_output_before_finalfreeze']==0 and freeze['new_model_or_threshold_selection'] is False
  assert freeze['initial_full_head_sha256']==INITIAL_FULL_SHA and freeze['draws_sha256']==DRAW_SHA and freeze['source_plan_sha256']==PLAN_SHA
  same(freeze['baseline_reuse'],c['baseline_reuse']);same(freeze['global_feature_QA'],c['global_feature_QA']);same(freeze['draw_ledger'],c['draw_ledger'])
  assert qa['before_optimizer'] is True and qa['new_DEV_head_output']==0 and qa['original']['checked']==7617 and qa['original']['tolerance_used']==qa['original']['features_overwritten']==0
  for k in ('active_first384_byte_exact','original_condition32_byte_exact','original_fractions_byte_exact','original_group_slot_availability_exact'):assert qa['original'][k] is True
  assert qa['original']['source_group_FIT_DEV_overlap']==qa['original']['head_DEV_forward_before_QA']==0 and qa['original']['active_bytes_sha256']==qa['original']['node_bytes_sha256']
  assert qa['CF']['all_original_FIT_D_parents']==5406 and qa['CF']['constructed']==qa['CF']['proof_PASS']==5367 and qa['CF']['unconstructed']==39 and qa['CF']['constructed_UNKNOWN']==0
  for arm in ARMS:same(freeze['checkpoints'][arm],c['checkpoints'][arm]);same(freeze['logs'][arm],c['logs'][arm])
  for kind,table in [('checkpoint','checkpoints'),('log','logs'),('CSV','outputs')]:
   original=old[table][OLD_ARM];new=c[table][ARMS[0]];same(reuse['source_'+kind],original);same(reuse['copied'][kind],new)
   p=checked(new);assert new['sha256']==original['sha256'] and ('size_bytes'not in original or p.stat().st_size==original['size_bytes'])
  baseline=parse_csv(raw(checked(c['outputs'][ARMS[0]])).decode());newrows=parse_csv(raw(checked(c['outputs'][ARMS[1]])).decode())
  for a,b in zip(baseline,newrows):assert all(a[k]==b[k] for k in ('sample_index','row_id','family','target','role','available'))
  pairs,cfcounts=parse_CF(raw(checked(c['CF_FIT_output'])).decode(),baseline)
  assert c['draw_ledger']['sha256']==LEDGER_SHA
  ledger=[json.loads(x) for x in raw(checked(c['draw_ledger'])).decode().splitlines()];logs=[json.loads(x) for x in raw(checked(c['logs'][ARMS[1]])).decode().splitlines()]
  ledgerQA=audit_ledger(ledger,logs,baseline,pairs,manifest)
  import torch
  cp=torch.load(checked(c['checkpoints'][ARMS[1]]),weights_only=True,map_location='cpu')
  assert cp['arm']==ARMS[1] and cp['updates']==1000 and cp['head_parameters']==86054 and cp['use_lora'] is False and cp['adapter_parameters']==0
  for k in ('head_state_dict','initial_head_state_dict'):assert check_state(cp[k],FULL_SHAPES)==86054
  assert check_optimizer(cp,FULL_SHAPES)==10 and state_sha(cp['initial_head_state_dict'])==cp['initial_head_sha256']==INITIAL_FULL_SHA
  base={k[len('base.'):]:v for k,v in cp['initial_head_state_dict'].items() if k.startswith('base.')};assert state_sha(base)==INITIAL_BASE_SHA
  assert cp['same_initial_base_sha256']==INITIAL_BASE_SHA and cp['draws_sha256']==DRAW_SHA and cp['source_plan_sha256']==PLAN_SHA
  assert state_sha(cp['head_state_dict'])==c['final_head_sha256'][ARMS[1]]!=INITIAL_FULL_SHA
  same(cp['entry'],e['root_fit_entry'])
  for key in ('original_feature_complete','CF_feature_complete','CF_data_complete'):same(cp[key],c[key])
  assert c['final_head_sha256'][ARMS[0]]==old['final_head_sha256'][OLD_ARM]
  summary=completed(c['summary']);assert summary['all_original_rows']==15040 and summary['known_output_rows']==7617 and summary['masked_target_minus1_retained']==7423 and summary['current_admission'] is False
  replay={'executed':False}
  if e.get('allow_saved_head_cpu_replay') is True:replay=CPU_replay(e,cp,newrows,old,checked)
  result={'completed':True,'PASS':True,'complete':e['complete'],'root_fit_entry':e['root_fit_entry'],'prospective_plan':e['prospective_plan'],'new_TREAT_parameters':86054,'new_TREAT_updates':1000,'optimizer_states_at1000':10,'new_BASELINE_updates':0,'BASELINE_CP_numerical_loads':0,'baseline_CP_log_CSV_SHA_identity_vs_oldmetadata':True,'initial_full_head_sha256':INITIAL_FULL_SHA,'initial_base_sha256':INITIAL_BASE_SHA,'final_TREAT_state_sha256':state_sha(cp['head_state_dict']),'ledger':ledgerQA,'actual_whole_ledger_sha256':LEDGER_SHA,'final_freeze':c['freeze'],'global_feature_QA':c['global_feature_QA'],'natural_rows_each':7617,'FIT':5876,'DEV':1741,'CF_rows':10812,'CF_pair_counts':cfcounts,'optional_saved_head_CPU_replay':replay,'access':access,'limits':['No training, encoder, raw RGB/GT/prefix/current/held. Baseline checkpoint is opaque SHA; only new small checkpoint is CPU weights_only decoded.','Provenance/optimizer/count/freeze audit. Independent CM/P/R arithmetic is separate C QA.','Chronology combines reviewed code, saved receipts and exact artifact bindings; no gradient replay.','Actual trainer read original SOURCE DEV nonpixelmetadata; reused SOURCE DEV is exposed. This verifier reads no samples NPZ. Optional CPU replay admits only public saved SOURCE feature/geometry arrays afterfinalfreeze.'],'automatic_retry':False,'model_data_threshold_selection':False}
  p=out/'REVIEW.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'completed':True,'PASS':True,'REVIEW':{'path':str(p),'sha256':hashlib.sha256(raw(p)).hexdigest()}}))
 except BaseException:
  (out/'FAILED_REVIEW.json').write_text(json.dumps({'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()},indent=2)+'\n');raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
