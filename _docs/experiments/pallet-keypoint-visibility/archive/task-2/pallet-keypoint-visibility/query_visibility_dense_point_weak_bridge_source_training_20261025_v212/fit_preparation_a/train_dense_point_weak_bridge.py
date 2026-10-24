"""C62/C63: unchanged C61 head, admitted own-scene point loss, optional weak bridge.
No source work on import. Actual root entry + exact two admissions precede payloads.
"""
import argparse,hashlib,io,json,os,stat,sys,time,traceback,types
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
RUN=V/'runs/C62_C63_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIXED4000'
PROTOCOL=H/'PROTOCOL_BEFORE_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIT.json'
ARMS=(('C62_DENSE_POINT_RAW_Q8',False),('C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8',True))

def need(v,m):
 if not bool(v):raise AssertionError(m)
def safe(p):
 p=Path(p);need(p.is_absolute()and'..'not in p.parts and'v112'not in str(p).casefold(),'Lexical path before FS');need(p.is_relative_to(P),'Exact source project only');return p
def rb(p):
 p=safe(p);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for i,s in enumerate(p.parts[1:]):
   n=os.open(s,os.O_RDONLY|os.O_NOFOLLOW|(os.O_DIRECTORY if i<len(p.parts[1:])-1 else 0),dir_fd=fd);os.close(fd);fd=n
  need(stat.S_ISREG(os.fstat(fd).st_mode),'Regular file only')
  with os.fdopen(fd,'rb')as f:fd=-1;return f.read()
 finally:
  if fd>=0:os.close(fd)
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def checked(b):
 z=rb(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Exact bound SHA');return z
def meta(b):
 p=safe(b['path']);need(p.suffix in('.json','.jsonl')and p.name!='SCENES_GT_ONLY.json'and not p.name.startswith('BASE_CONFIGS'),'Metadata only; no original GT/config')
 z=checked(b);return json.loads(z)if p.suffix=='.json'else[json.loads(x)for x in z.splitlines()]
def module(b,name):
 need(Path(b['path']).suffix=='.py','Explicit code only');z=checked(b);m=types.ModuleType(name);m.__file__=b['path'];sys.modules[name]=m;exec(compile(z,b['path'],'exec'),m.__dict__);return m
def new(p,v):
 p=safe(p);need(p.is_relative_to(RUN),'Own fresh runtime only');z=(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for s in p.parts[1:-1]:
   n=os.open(s,os.O_RDONLY|os.O_NOFOLLOW|os.O_DIRECTORY,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644,dir_fd=fd)
  with os.fdopen(n,'wb')as f:f.write(z)
 finally:os.close(fd)
def fresh_run():
 cur=P
 for s in RUN.relative_to(P).parts[:-1]:
  cur/=s
  try:mode=cur.lstat().st_mode
  except FileNotFoundError:cur.mkdir();mode=cur.lstat().st_mode
  need(stat.S_ISDIR(mode),'Ordinary output ancestors')
 RUN.mkdir(exist_ok=False)

def split_old(split,old,adapter):
 roles={};masks={};aligned={}
 for r in meta(split['outputs']['ALL_COMPONENTS_AND_CANONICAL_ORIGIN_NODES.jsonl']):
  cid=hashlib.sha256('\n'.join(sorted(r['origin_nodes'])).encode()).hexdigest();role='HOLDOUT'if int(hashlib.sha256(('pallet_visibility_source_groups_v194_20261024\n'+cid).encode()).hexdigest()[:16],16)%100<20 else'TRAIN'
  need(cid==r['component_id']and role==r['role']and cid not in roles,'Immutable original component role');roles[cid]=role
 for d,cache in old.items():
  rr=meta(split['outputs'][d+'_SPLIT_ROWS.jsonl']);need(all(roles.get(r['component_id'])==r['role']for r in rr),'Exact role in every loss pool');masks[d]=adapter.fit_mask(cache['metadata'],rr);aligned[d]=rr
 return masks,aligned

def load_weak(plan,context_adapter,c,allowed):
 from PIL import Image
 rows=plan['rows'];n=len(rows);need(n==44 and plan['counts']==dict(rows=44,factual_D=8,SHAM_D=18,COVER_weak_E=18,distinct_frames=3,unique_effective_RGB_valid_hashes=44,unique_PNG_payloads=52),'Frozen44 weak metadata contract')
 images={}
 def png(b,kind):
  key=b['path'];need((key,b['sha256'])in allowed,'Exact root PNG whitelist')
  if key not in images:
   with Image.open(io.BytesIO(checked(b)))as im:
    need(im.format=='PNG'and im.size==(96,96)and im.mode==('RGB'if kind=='RGB'else'L'),'Original PNG schema; no conversion/resample');a=np.asarray(im).copy()
   if kind=='VALID':need(np.isin(a,[0,255]).all(),'Original binary validity');a=a==255
   need(a.dtype==(np.uint8 if kind=='RGB'else bool),'Decoded source dtype');images[key]=a
  return images[key]
 x=dict(rgb96=np.empty((n,96,96,3),np.uint8),valid96=np.empty((n,96,96),bool),sam_mask96=np.zeros((n,96,96),bool),sam_available96=np.zeros((n,96,96),bool),row_id=np.asarray([r['row_id']for r in rows]),original_global_index=np.asarray([r['original_global_index']for r in rows],np.int64),native_slot=np.asarray([r['native_slot']for r in rows],np.int8),query_xy=np.asarray([r['query_xy']for r in rows],np.float64),L=np.asarray([r['L']for r in rows],np.float64),original_query_available=np.asarray([r['original_query_available']for r in rows],bool))
 vv=[]
 for i,r in enumerate(rows):
  need(r['component_role']=='TRAIN'and r['original_query_available']is True,'No weak source-HOLDOUT or inputU label')
  x['rgb96'][i]=png(r['RGB_binding'],'RGB');x['valid96'][i]=png(r['valid_binding'],'VALID')
  need(c.ash(x['rgb96'][i])==r['rgb96_sha256_expected']and c.ash(x['valid96'][i])==r['valid96_sha256_expected'],'Actual weak PNG byte and decoded hash closure')
  frame=plan['frames'][r['frame_q8_key']];eight=frame['metadata8'];need([m['native_slot']for m in eight]==list(range(8)),'Same original q8 group')
  active=eight[r['native_slot']]
  for k in('query_xy','L','original_query_available','native_slot'):need(active[k]==r[k],'Weak original q8 active identity '+k)
  need(active['row_id']==r['source_row_id']and active['original_global_index']==r['source_original_global_index'],'Weak source row identity')
  vv.append(context_adapter.build_context_vectors(r['query_xy'],r['L'],r['native_slot'],r['original_query_available'],[m['query_xy']for m in eight],[m['L']for m in eight],[m['original_query_available']for m in eight]))
 context={k:np.stack([v[k]for v in vv])for k in('native_onehot8','other_mask8','other_mask8_bool','other_dxdy16')}
 context.update(native_slot=x['native_slot'].astype(np.int64),row_id=x['row_id'],original_query_available=x['original_query_available'],meta=[{k:r[k]for k in c.META}for r in rows],context_origin_group_id=[r['frame_q8_key']for r in rows],fit_nominal_eligible=np.ones(n,bool))
 need(len({x['rgb96'][i].tobytes()+x['valid96'][i].tobytes()for i in range(n)})==44,'Actual44 unique weak inputs; no deletion/fallback')
 return dict(x=c.validate_input10(x,n),metadata=rows),context,np.asarray([r['proposed_weak_target']for r in rows],np.int8)

def gather(old,weak,contexts,weakctx,draw,pi,wi,pure):
 g=pure['gather_batch_arrays'](old,draw,False);domains=('SYNTH_AUG65536',)*3+('PUBLIC_TRAIN1480','CF3816');names=('visible','foreign','synthetic_owner','public_owner','CF_owner')
 for k in g:
  chunks=[g[k],old['SYNTH_AUG65536']['x'][k][pi]]
  if wi is not None:chunks.append(weak['x'][k][wi])
  g[k]=np.concatenate(chunks)
 ctx={}
 for k in('native_slot','other_mask8','other_dxdy16'):
  chunks=[contexts[d][k][draw[name]]for d,name in zip(domains,names)]+[contexts['SYNTH_AUG65536'][k][pi]]
  if wi is not None:chunks.append(weakctx[k][wi])
  ctx[k]=np.concatenate(chunks)
 expected=np.concatenate([old[d]['x']['native_slot'][draw[name]]for d,name in zip(domains,names)]+[old['SYNTH_AUG65536']['x']['native_slot'][pi]]+([]if wi is None else[weak['x']['native_slot'][wi]]))
 need(np.array_equal(ctx['native_slot'],expected),'Same image draw/native/context take')
 return g,ctx

def fit(entry):
 e=json.loads(rb(entry));need(e['approved_dense_point_weak_bridge_source_fit']is True and e['source_only']is True and e['automatic_retry']is False,'Separate actual root fit entry before bodies')
 p=meta(e['protocol']);need(e['protocol']==bind(PROTOCOL)and e['code']==bind(__file__)==p['trainer_code'],'Frozen exact code/protocol')
 meta(e['root_scope']);need(e['weak_admission']==p['weak_admission'],'Exact frozen separate weak admission')
 need(e['output_directory']==str(RUN),'Fresh fixed run');scope=meta(p['scope']);meta(p['eligibility_amendment']);need(scope['fixed_fit_design_pending_admission']['steps_per_arm']==4000,'Predeclared fixed trial')
 qa=meta(e['independent_preflight']);need(qa['completed']is True and qa['PASS']is True and qa['code']==p['trainer_code']and qa['protocol']==e['protocol'],'Exact independent code preflight')
 checks=meta(p['checks']);need(checks['completed']and checks['PASS']and checks['trainer_code']==p['trainer_code']and checks['target_contract']==p['target_contract'],'Meaningful checks bind exact final code/contract')
 pa=meta(p['point_admission']);wa=meta(e['weak_admission']);need(pa['approved_prospective_SOURCE_point_teaching']is True and wa['completed']is True and wa['approved_SOURCE_weak44_bridge']is True,'Separate source target admissions')
 need(pa['eligibility_amendment']==p['eligibility_amendment']and pa['source_teacher_complete']==p['teacher_complete']and pa['independent_actual_QA']==p['teacher_QA'],'Exact admitted point authority')
 tq=meta(p['teacher_QA']);need(tq['completed']and tq['PASS'],'Independent saved teacher check');teacher=meta(p['teacher_complete']);need(teacher['completed']and teacher['PASS'],'Actual saved teacher')
 mapped=meta(teacher['outputs']['ALL7040_INPUT_SOURCE_POINT_MAPPING.jsonl']);need(teacher['outputs']['ALL7040_INPUT_SOURCE_POINT_MAPPING.jsonl']==pa['exact_teacher_mapping'],'Exact admitted mapping')
 weakplan=meta(p['weak_metadata']);need(wa['weak_metadata']==p['weak_metadata']and wa['counts']==dict(rows=44,D=26,E=18)and wa['weak_loss_weight']==.25,'Exact separate44 weak branch, not native gold')
 for b in p['source_bindings'].values():checked(b)
 oldproto=meta(p['inherited_protocol']);need(oldproto['context_input_bindings']==e['context_input_bindings']and p['old_cache_bindings']==oldproto['old_cache_bindings']and e['split_complete']==oldproto['split_complete'],'Unchanged old5 input/split/context authority')
 aug=meta(p['old_cache_bindings']['augmentation']);public=meta(p['old_cache_bindings']['public_CF']);records={'SYNTH_AUG65536':aug,**{d:public['domains'][d]for d in('PUBLIC_TRAIN1480','CF3816')}}
 expected={(r[k]['path'],r[k]['sha256'])for r in records.values()for k in('inputs','targets')}|{(r['path'],r['sha256'])for r in weakplan['payload_whitelist']}
 need(expected=={(r['path'],r['sha256'])for r in e['payload_whitelist']}and len(expected)==58,'Exact6 old NPZ +52 weak PNG payloads; no oldDIRECT/weights/GT')
 need(torch.__version__=='2.1.1+cu118'and np.__version__=='1.26.4'and torch.cuda.is_available(),'Pinned CUDA runtime')
 base=module(p['inherited_trainer'],'v212_inherited_c61');c=base.c;contract=module(p['target_contract'],'v212_target_contract')
 head=module(p['source_bindings']['head'],'v212_head');candidate=module(p['source_bindings']['candidate'],'v212_candidate');spatial=module(p['source_bindings']['spatial_head'],'v212_spatial');adapter=module(p['source_bindings']['mask_adapter'],'v212_masks');ca=module(p['source_bindings']['context_adapter'],'v212_q8');pure=base.pure_training(head)
 original_arrays=c.arrays
 def guarded_arrays(b,keys):
  need((b['path'],b['sha256'])in expected or safe(b['path']).is_relative_to(RUN),'No undeclared source array binding')
  return original_arrays(b,keys)
 c.arrays=guarded_arrays
 fresh_run();new(RUN/'STARTED.json',dict(entry=bind(entry),from_scratch=True,previous_checkpoint_loaded=False,source_only=True));start=time.monotonic()
 try:
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  old={d:base.load_old(r,base.OLD_ROWS[d],'TRAIN'if d=='SYNTH_AUG65536'else'SOURCE_DEVELOPMENT',pure)for d,r in records.items()}
  split=meta(e['split_complete']);masks,aligned=split_old(split,old,adapter);loaded=ca.load_bound_context_metadata();contexts=loaded['views'];need(loaded['input_bindings']==e['context_input_bindings'],'Exact original metadata q8')
  fitmasks={}
  for d,cache in old.items():
   base.validate_context_view(cache,contexts[d]);need(np.array_equal(contexts[d]['immutable_TRAIN_role'],masks[d]),'Immutable role equality')
   fitmasks[d]=masks[d]&contexts[d]['fit_nominal_eligible']
  masked=adapter.masked_old_pools(old,fitmasks,pure['pools_for']);need(masked['feasible'],'Old5 classes unchanged and feasible')
  point,point_indices,pointrows=contract.join_point_rows(old['SYNTH_AUG65536']['metadata'],mapped,masks['SYNTH_AUG65536'],contexts['SYNTH_AUG65536']['fit_nominal_eligible'])
  need(len(point_indices)==7040 and[(point==j).sum()for j in(0,1)]==[5853,243]and pa['eligibility_counts']==dict(all7040_retained=7040,loss_eligible=6096,D=5853,E=243,excluded_UNKNOWN_loss=944),'Prespecified root-admitted source eligibility; no data change')
  pp=contract.balanced_pools(point,old['SYNTH_AUG65536']['x']['original_query_available'],fitmasks['SYNTH_AUG65536'])
  weak,weakctx,wt=load_weak(weakplan,ca,c,expected);base.validate_context_view(weak,weakctx);wp=contract.balanced_pools(wt,weak['x']['original_query_available'],np.ones(44,bool))
  c.write_rows(RUN/'ALL7040_POINT_LOSS_TARGETS_AND_ELIGIBILITY.jsonl',pointrows)
  new(RUN/'FIT_INPUT_VALIDATION.json',dict(completed=True,PASS=True,point_all7040=7040,point_D=5853,point_E=243,point_excluded_UNKNOWN=944,weak_rows=44,weak_D=26,weak_E=18,old5_masked_counts=masked['counts'],fit_rows={d:int(v.sum())for d,v in fitmasks.items()},all_old5_masks_unchanged=True,original_U_qL_native_q8_unchanged=True,original_roles_not_relabelled=True,teacher_reference_or_region_or_alpha_not_model_input=True,no_old_DIRECT_NPZ_read=True,point_admission=p['point_admission'],weak_admission=e['weak_admission'],teacher_complete=p['teacher_complete'],weak_metadata=p['weak_metadata']))
  cls=spatial.make_spatial_q8_head_class(head,candidate);pending={};initial=None;oldhash=None;pointhash=None
  for arm,bridge in ARMS:
   folder=RUN/arm;folder.mkdir();torch.manual_seed(1810);torch.cuda.manual_seed_all(1810);model=cls(spatial_q8=True).float().cuda();ish=pure['object_sha'](model.state_dict());need(sum(x.numel()for x in model.parameters())==118359,'Same118359 C61 architecture')
   if initial is None:initial=ish
   else:need(ish==initial,'Paired same fresh init')
   opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(1810);prng=np.random.default_rng(1811);wrng=np.random.default_rng(1812);dh=hashlib.sha256();ph=hashlib.sha256();wh=hashlib.sha256();logpath=folder/'ALL4000_DRAWS_AND_LOSSES.jsonl'
   with logpath.open('x')as log:
    for update in range(1,4001):
     draw=pure['draw_step'](rng,masked['pools']);pi=np.concatenate([prng.choice(a,16,replace=True)for a in pp]);wi=np.concatenate([wrng.choice(a,8,replace=True)for a in wp])if bridge else None
     for key,idx in draw.items():need(fitmasks[{'visible':'SYNTH_AUG65536','foreign':'SYNTH_AUG65536','synthetic_owner':'SYNTH_AUG65536','public_owner':'PUBLIC_TRAIN1480','CF_owner':'CF3816'}[key]][idx].all(),'All old5 draws TRAIN/eligible');dh.update(key.encode());dh.update(idx.astype('<i8').tobytes())
     need(fitmasks['SYNTH_AUG65536'][pi].all()and np.isin(point[pi],[0,1]).all(),'Known TRAIN point draws');ph.update(pi.astype('<i8').tobytes())
     if wi is not None:wh.update(wi.astype('<i8').tobytes())
     g,ctx=gather(old,weak,contexts,weakctx,draw,pi,wi,pure);inp=base.assemble(g,head,False,'cuda');native,mask,xy=base.context_tensors(ctx,np.arange(len(inp)),'cuda');opt.zero_grad(set_to_none=True);a=model(inp,native,mask,xy)
     pt=torch.as_tensor(point[pi],device='cuda');wtarget=None if wi is None else torch.as_tensor(wt[wi],device='cuda');loss,aux,components,pl,wl=contract.objective(a,pt,wtarget,pure,bridge)
     need(torch.isfinite(loss),'Finite fixed loss');loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);need(torch.isfinite(norm),'Finite gradients');opt.step();need(all(torch.isfinite(x).all()for x in model.parameters()),'Finite state')
     log.write(json.dumps(dict(update=update,original192_draws={k:v.tolist()for k,v in draw.items()},point32_draws=pi.tolist(),weak16_draws=None if wi is None else wi.tolist(),old_five_losses=[float(x.detach())for x in components],old_five_mean=float(aux.detach()),point_loss=float(pl.detach()),weak_loss=float(wl.detach()),weak_loss_weight=.25 if bridge else 0.,total_loss=float(loss.detach()),grad_norm=float(norm),old_draw_sha=dh.hexdigest(),point_draw_sha=ph.hexdigest(),weak_draw_sha=wh.hexdigest(),all_drawn_rows_TRAIN=True),allow_nan=False)+'\n')
     if update%250==0:print(json.dumps(dict(arm=arm,update=update,seconds=time.monotonic()-start)),flush=True)
   if oldhash is None:oldhash,pointhash=dh.hexdigest(),ph.hexdigest()
   else:need((dh.hexdigest(),ph.hexdigest())==(oldhash,pointhash),'Paired old192/point32 draws unchanged by weak RNG')
   cp=folder/'FINAL_CHECKPOINT.pt';torch.save(dict(state_dict=pure['cpu_copy'](model.state_dict()),optimizer_state=pure['cpu_copy'](opt.state_dict()),update=4000,seed=1810,arm=arm,use_sam=False,spatial_q8=True,weak_bridge=bridge,weak_loss_weight=.25 if bridge else 0.,protocol=e['protocol'],source_split=e['split_complete'],direct_target_authority='ADMITTED_SOURCE_OWN_IMAGE_POINT0_PLUS_SEPARATE_WEAK_BRIDGE_IF_ENABLED'),cp)
   model.eval().requires_grad_(False);pending[arm]=dict(model=model,use_sam=False,spatial_q8=True,weak_bridge=bridge,weak_loss_weight=.25 if bridge else 0.,updates=4000,initial_state_sha=ish,final_state_sha=pure['object_sha'](model.state_dict()),checkpoint=c.bind(cp),draws_and_losses=c.bind(logpath),old_draw_sha=dh.hexdigest(),point_draw_sha=ph.hexdigest(),weak_draw_sha=wh.hexdigest());del opt
  barrier=RUN/'ALL2_FINAL4000_FROZEN_BEFORE_ANY_SOURCE_OUTPUTS.json';new(barrier,dict(completed=True,checkpoints={a:r['checkpoint']for a,r in pending.items()},same_initial_state_sha=initial,same_old192_draw_sha=oldhash,same_point32_draw_sha=pointhash,source_predictions_emitted=False,checkpoint_or_arm_selection=False))
  for arm,r in pending.items():
   model=r.pop('model');folder=RUN/arm;r['outputs']={};r['source_cohorts']={}
   for d,cache in old.items():
    full=base.save_outputs(model,cache,contexts[d],folder/('FINAL_'+d+'_OUTPUTS.npz'),False,head);r['outputs'][d]=full;r['source_cohorts'][d]=base.save_cohorts(full,masks[d],folder,d,aligned[d],contexts[d]['fit_nominal_eligible'])
   sx=c.arrays(r['outputs']['SYNTH_AUG65536'],base.ROW_OUTPUTS+base.ORDERS);packet={k:sx[k][point_indices]for k in base.ROW_OUTPUTS};packet.update({k:sx[k]for k in base.ORDERS});r['outputs']['POINT7040']=c.save_arrays(folder/'FINAL_POINT7040_OUTPUTS.npz',packet)
   r['outputs']['WEAK44']=base.save_outputs(model,weak,weakctx,folder/'FINAL_WEAK44_OUTPUTS.npz',False,head);need(pure['object_sha'](model.state_dict())==r['final_state_sha'],'Frozen weights unchanged after source outputs');del model
  for b in e['payload_whitelist']:checked(b)
  for b in p['source_bindings'].values():checked(b)
  base.check_binding_tree(loaded['input_bindings'])
  for b in(e['protocol'],p['trainer_code'],p['target_contract'],p['teacher_complete'],p['teacher_QA'],p['point_admission'],e['weak_admission'],p['weak_metadata'],e['independent_preflight']):checked(b)
  new(RUN/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,from_scratch=True,code=p['trainer_code'],protocol=e['protocol'],entry=bind(entry),head=p['source_bindings']['head'],candidate=p['source_bindings']['candidate'],spatial_head=p['source_bindings']['spatial_head'],context_adapter=p['source_bindings']['context_adapter'],target_contract=p['target_contract'],point_teacher=p['teacher_complete'],point_admission=p['point_admission'],weak_admission=e['weak_admission'],weak_metadata=p['weak_metadata'],parameters_each=118359,steps_each=4000,total_updates=8000,source_split=e['split_complete'],all2_freeze=bind(barrier),all2_frozen_before_any_SOURCE_outputs=True,arms=pending,same_initial_state_sha=initial,same_old192_draw_sha=oldhash,same_point32_draw_sha=pointhash,main_target_replaces_old_finite_end_DIRECT=True,old_five_aux_unchanged=True,primary_predeclared='C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8',selected_model=None,protected_evaluation_access=0,protected_real_COMMON_VAL_access=0,original_qL_native_U_changed=False,adopted=False,fully_solved=False,seconds=time.monotonic()-start))
  print('ALL2_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIT_COMPLETE',flush=True)
 except BaseException:
  new(RUN/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--entry',type=Path,required=True);v=a.parse_args();fit(v.entry)
