"""Prospective saved-SOURCE TEXT-only matched plan; stdlib, no new RNG/arrays."""
import hashlib,json,os,stat,struct,traceback
from collections import Counter
from pathlib import Path
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation');H=Path(__file__).absolute().parent;OUT=H/'plan_v1'
PINS={
 'source_complete':(R/'V239_ROOT_CANONICAL_CONTEXT/canonical_pair_source_fit_v1/COMPLETE.json','1fd6880528d1bd440376c68c7079b9cbee59ce736f006fdaef4244e3310a4df9'),
 'dataset_complete':(R/'V242_ROOT_SCENE_CF/scene_pairs_v1/COMPLETE.json','12f973806eccdbeae25afea8f6e23fb43da738f2d412d7dbdbb82ea8d087d8bc'),
 'draw_ledger':(R/'V239_ROOT_CANONICAL_CONTEXT/canonical_pair_source_fit_v1/COMMON_DRAW_LEDGER1000.jsonl','c8965acb04108eaf0fb4418733d92ad895cf155a6ca03f427baee2f4b202d8c4'),
 'pair_metadata':(R/'V242_ROOT_SCENE_CF/scene_pairs_v1/PAIR_METADATA.jsonl','32c63e6325efc5410468508ec3109d737a4be1afa8833f4daaca295e60e39000')}
DRAW_SHA='20a1fc74ea15a04a16fe29f8e5b9a114b4629677710ed43c1628b2a1ac00b734'

def safe(value):
 p=Path(value)
 if any(t in str(p).casefold() for t in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')) or not p.is_absolute() or '..' in p.parts:raise ValueError('Forbidden lexical path before FS')
 for a in reversed((p,*p.parents)):
  try:mode=os.lstat(a).st_mode
  except FileNotFoundError:continue
  if stat.S_ISLNK(mode):raise ValueError('Symlink ancestor')
 return p

def read(p):
 with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def binding(p):
 raw=read(p);return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}

def checked(pin):
 p,sha=pin;raw=read(p)
 if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Exact saved metadata bytes changed')
 return raw

def exact(binding_,pin):
 if (binding_['path'],binding_['sha256'])!=(str(pin[0]),pin[1]):raise ValueError('Declared metadata graph mismatch')

def plan_step(indices,pair_by_parent):
 if len(indices)!=32 or any(type(x)is not int or x<0 or x>=7617 for x in indices):raise ValueError('Fixed original32 integer draw')
 parent=list(indices);pairs=[-1]*32;roles=[-1]*32
 for family in (0,1):
  start=family*16
  # Original8D8E position roles come from immutable reviewed V239 generator.
  for j in range(8):
   row=pair_by_parent[indices[start+j]]
   if row['family']!=family:raise ValueError('All original drawn D metadata must retain its source family')
  for j in range(4):
   d=start+4+j;e=start+12+j;row=pair_by_parent[indices[d]]
   if row['proof_pass']:
    parent[d]=parent[e]=indices[d];pairs[d]=pairs[e]=row['pair_index'];roles[d]=1;roles[e]=0
 return parent,pairs,roles

def digest(draws,plans):
 h=hashlib.sha256()
 for indices in draws:h.update(struct.pack('<32q',*indices))
 for parent,pairs,roles in plans:
  h.update(struct.pack('<32q',*parent));h.update(struct.pack('<32q',*pairs));h.update(struct.pack('<32b',*roles))
 return h.hexdigest()

def write_json(name,v):
 p=safe(OUT/name)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return binding(p)

def run():
 if safe(OUT).exists():raise ValueError('Fresh prospective output only; no retry')
 OUT.mkdir()
 try:
  raw={k:checked(pin) for k,pin in PINS.items()};source=json.loads(raw['source_complete']);data=json.loads(raw['dataset_complete'])
  if not all(v.get('completed') is True and v.get('PASS') is True for v in (source,data)):raise ValueError('Existing source/data COMPLETE required')
  exact(source['draw_ledger'],PINS['draw_ledger']);exact(data['outputs']['PAIR_METADATA.jsonl'],PINS['pair_metadata'])
  if data['counts']['attempted']!=5406 or data['all_original5406_attempted'] is not True or source['draws_sha256']!=DRAW_SHA:raise ValueError('Fixed source full-retention scope')
  parentmap={};pool=Counter()
  for i,line in enumerate(raw['pair_metadata'].splitlines()):
   row=json.loads(line);original=row['original_input'];parent=int(row['parent_sample_index']);f=int(original['family_code']);proof=row['proof_pass'];constructed=row['constructed']
   if row['pair_index']!=i or parent in parentmap or not 0<=parent<7617 or f not in (0,1) or type(proof)is not bool or type(constructed)is not bool:raise ValueError('Unique5406 original pair_index/sourceparent metadata')
   if original['fit'] is not True or original['target']!=0 or original['original_query_available'] is not True or original['crop_available'] is not True:raise ValueError('Original SOURCE FIT supported knownD only')
   if proof and (not constructed or row['targets_D0_E1_Uminus1']!=[1,0]):raise ValueError('Proof-pass role target metadata')
   if not proof and row['targets_D0_E1_Uminus1']!=[-1,-1]:raise ValueError('No UNKNOWN to E conversion')
   parentmap[parent]={'pair_index':i,'family':f,'proof_pass':proof,'constructed':constructed,'row_id':original['row_id']};pool[(f,'all')]+=1;pool[(f,'proof_PASS' if proof else 'proof_fail')]+=1
  if len(parentmap)!=5406 or pool[(0,'all')]!=4565 or pool[(1,'all')]!=841:raise ValueError('All originalD5406 family distribution')
  draws=[]
  for update,line in enumerate(raw['draw_ledger'].splitlines(),1):
   row=json.loads(line)
   if row['update']!=update or row['draws_sha256']!=DRAW_SHA or row['family_counts']!=[16,16] or row['all_fit_available_known'] is not True:raise ValueError('Original sealed1000 draw provenance')
   draws.append(row['sample_indices'])
  if len(draws)!=1000:raise ValueError('Exactly1000 original updates')
  drawbytes=b''.join(struct.pack('<32q',*d) for d in draws)
  if hashlib.sha256(drawbytes).hexdigest()!=DRAW_SHA:raise ValueError('Original PCG17 draws unchanged; no regeneration')
  plans=[plan_step(indices,parentmap) for indices in draws];plan_sha=digest(draws,plans)
  output=OUT/'PROSPECTIVE_ORIGINAL_DRAW_AND_MATCHED_SOURCE_LEDGER1000.jsonl';counts=Counter();unique={0:set(),1:set()}
  with output.open('x') as f:
   for update,(indices,(parents,pairs,roles)) in enumerate(zip(draws,plans),1):
    # Same insertion order, default JSON separators and newline as frozen V243V2 actual ledger.
    f.write(json.dumps({'update':update,'sample_indices':indices,'source_parent_indices':parents,'pair_indices':pairs,'source_roles':roles,'draws_sha256':DRAW_SHA,'source_plan_sha256':plan_sha,'all_fit_available_known':True})+'\n')
    for family in (0,1):
     start=family*16
     for j in range(4):
      d=start+4+j;e=start+12+j;counts[(family,'matched_attempts')]+=1
      if pairs[d]>=0:
       if pairs[d]!=pairs[e] or parents[d]!=parents[e] or roles[d]!=1 or roles[e]!=0:raise ValueError('Strict sameparent matched C/S roles')
       counts[(family,'matched_success')]+=1;unique[family].add(pairs[d])
      else:
       if parents[d]!=indices[d] or parents[e]!=indices[e] or roles[d]!=roles[e]!=-1:raise ValueError('Failure must retain both original natural slots')
       counts[(family,'failed_both_natural')]+=1
     if any(parents[start+j]!=indices[start+j] or roles[start+j]!=-1 for j in (*range(4),*range(8,12))):raise ValueError('First4 original natural D/E unchanged')
  family_counts={str(f):{'parent_pool':pool[(f,'all')],'proof_PASS_pool':pool[(f,'proof_PASS')],'proof_fail_pool':pool[(f,'proof_fail')],'last4_D_proposals':counts[(f,'matched_attempts')],'matched_pairs_in_optimizer_draws':counts[(f,'matched_success')],'COVER_draws':counts[(f,'matched_success')],'SHAM_draws':counts[(f,'matched_success')],'failed_proposals_both_natural':counts[(f,'failed_both_natural')],'natural_D_draws':4000+counts[(f,'failed_both_natural')],'natural_E_draws':4000+counts[(f,'failed_both_natural')],'distinct_matched_parents_drawn':len(unique[f])} for f in (0,1)}
  manifest={'completed':True,'PASS':True,'status':'PROSPECTIVE_SEALED_BEFORE_NEW_FEATURES_OR_FIT','inputs':{k:{'path':str(pin[0]),'sha256':pin[1],'size_bytes':len(raw[k])} for k,pin in PINS.items()},'code':binding(Path(__file__).absolute()),'ledger':binding(output),'updates':1000,'draw_shape':[1000,32],'draws_sha256':DRAW_SHA,'source_plan_sha256':plan_sha,'family_counts':family_counts,'new_RNG_calls':0,'numpy_or_array_libraries_imported':False,'original_sample_or_feature_or_weight_payload_reads':0,'new_feature_or_optimizer_or_DEV_head_execution':0,'class_order_authority':'Immutable reviewed V239 generator and its saved draw ledger; E row labels are not reinterpreted from new sample arrays here. Full original numeric metadata checks remain in frozen trainer.','plan_encoding':'sha256(all1000 draw rows little-endian int64 then perupdate parent int64 + pair int64 + role signedint8), matches frozen V243V2 plan_digest','actual_ledger_byte_identity_expected':True,'policy':'Original first4 D/E; final4 D proofPASS ownparentSHAM and corresponding E ownparentCOVER; failbothnatural; no successpoolchooser','runtime_admission':'Frozen trainer V2 unchanged. Root independently compares actual source_plan_sha256 and whole actual ledger to this sealed plan before result/model admission; mismatch invalidates run with no automatic retry.'}
  result=write_json('PLAN_MANIFEST.json',manifest);print(json.dumps({'completed':True,'PASS':True,'plan_manifest':result,'ledger':binding(output),'source_plan_sha256':plan_sha,'family_counts':family_counts}),flush=True)
 except BaseException:
  write_json('FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()});raise

if __name__=='__main__':run()
