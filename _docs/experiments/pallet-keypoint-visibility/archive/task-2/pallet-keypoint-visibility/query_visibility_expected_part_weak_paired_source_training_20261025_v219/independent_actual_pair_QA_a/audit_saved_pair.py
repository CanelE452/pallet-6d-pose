"""Independent bounded SOURCE saved-pair QA; all actual reads require root --entry.
No original/selected NPZ, GT payload, private evaluation, fitting or optimizer step.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,io,json,math,os,stat,sys,types
import numpy as np
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_expected_part_weak_paired_source_training_20261025_v219';H=V/'independent_actual_pair_QA_a'
ARMS=('STRONG_ONLY_CONTROL','STRONG_PLUS_EXPECTED_PART_WEAK')
DOMAINS=('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816')
TASKS=('visible','foreign','synthetic_owner','public_owner','CF_owner');SIZES=(32,32,48,48,32)
AUTH_POINT='SYNTH_RENDERED_OWN_IMAGE_POINT0_TARGET_ALONE_D_COMBINED_DE'
AUTH_WEAK='ASSISTANT_CONSENSUS_EXPECTED_FINITE_PART_SOURCE_GT_QUERY'
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
MAXABS=1e-4

def need(v,m):
 if not bool(v):raise ValueError(m)
def ident(b):
 need(isinstance(b,dict)and isinstance(b.get('path'),str)and isinstance(b.get('sha256'),str)and len(b['sha256'])==64,'typed binding')
 return b['path'],b['sha256']
def same(a,b):return ident(a)==ident(b)
def lexical(p):
 p=Path(p);need(p.is_absolute()and '..'not in p.parts and 'v112'not in str(p).casefold(),'literal nonV112 before FS');return p
def raw(p):
 p=lexical(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(leaf,'rb')as f:need(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'regular file');return f.read()
def checked(b,allow,suffixes):
 key=ident(b);p=lexical(key[0]);need(p.suffix in suffixes,'declared file kind before FS')
 need(any(ident(x)==key for x in allow),'exact path/SHA whitelist before FS')
 z=raw(p);need(hashlib.sha256(z).hexdigest()==key[1]and('size_bytes'not in b or len(z)==b['size_bytes']),'byte hash/optional size');return z
def js(b,allow):
 z=checked(b,allow,('.json','.jsonl'))
 return [json.loads(x)for x in z.splitlines()if x.strip()]if Path(b['path']).suffix=='.jsonl'else json.loads(z)
def binding(p):
 z=raw(p);return dict(path=str(p),sha256=hashlib.sha256(z).hexdigest())
def write(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
 return binding(p)
def meta_equal(a,b):
 for k in META:need(a[k]==b[k],'exact original META6 '+k)
def context(q,L,k,av,q8,L8,av8):
 need(type(k)is int and 0<=k<8 and len(q8)==len(L8)==len(av8)==8 and type(av)is bool and all(type(x)is bool for x in av8)and av8[k]==av,'same original native/availability8')
 out=np.zeros(32,np.float32);out[k]=1
 if av:
  need(q is not None and len(q)==2 and all(math.isfinite(v)for v in q)and L is not None and math.isfinite(L)and L>0,'available active finite')
  for j in range(8):
   if j==k or not av8[j]:continue
   need(q8[j]is not None and len(q8[j])==2 and all(math.isfinite(v)for v in q8[j])and L8[j]is not None and math.isfinite(L8[j])and L8[j]>0,'available other finite')
   out[8+j]=1
   for axis in range(2):out[16+2*j+axis]=np.float32(max(-8.,min(8.,(np.float64(q8[j][axis])-np.float64(q[axis]))/np.float64(L))))
 return out

def fit_mask(rr,dev):
 return np.asarray([r.get('component_role')=='TRAIN'and r.get('nominal_fit_eligible')is True and r.get('lineage_bound')is True and isinstance(r.get('component_id'),str)and len(r['component_id'])==64 and r['component_id']not in dev for r in rr],bool)
def point_target(r):
 finite=lambda x:x is not None and len(x)==2 and all(math.isfinite(float(a))for a in x)
 q,L,ref=r.get('query_xy'),r.get('L'),r.get('saved_projected_reference_xy');loc=r.get('reference_locality_for_actual_input',{})
 goodq=finite(q)and L is not None and math.isfinite(float(L))and L>0
 if goodq and finite(ref):
  uv=np.asarray([47.5+96*(float(ref[j])-float(q[j]))/float(L)for j in range(2)])
  need(np.array_equal(uv,np.asarray(loc.get('raw96_reference_index_xy'))),'point reference address exact algebra')
  need(bool(np.all((uv>=-.5)&(uv<95.5)))==(loc.get('inside_crop_canvas_extent')is True),'point crop canvas flag')
 ok=(r.get('component_role')=='TRAIN'and r.get('mapping_status')=='EXACT_ORIGINAL_IMAGE_OWN_SCENE'and r.get('family')in('SELF_SCENE15','BEFORE_ANCHOR4')and r.get('state_index')==0 and r.get('state_name')=='NOMINAL'and r.get('original_query_available')is True and goodq and r.get('point0_target_alone_state')=='D'and r.get('point0_combined_state')in('D','E')and finite(ref)and loc.get('inside_original_frame_canvas_extent')is True and loc.get('inside_crop_canvas_extent')is True)
 return int(r['point0_combined_state']=='E')if ok else -1

def weak_rows(rows):
 need(len(rows)==360 and len({r['row_id']for r in rows})==360,'all360 weak records retained')
 groups={};dev=set()
 for r in rows:
  g=r['generated_input'];need(g['row_id']==r['row_id']and g['source_role']==r['source_role']and r['training_admission']is False and g['training_admission']is False,'weak identity/original nonadmission')
  need(g['family']=='PUBLIC_SOURCE_GT_QUERY_JITTER_V1'and g['q8_origin']=='GT_DERIVED_TRAINING_INITIALIZATION','GT-derived training query explicit')
  need(g['annotation_markers_in_training_RGB']is False,'unmarked weak training image')
  groups.setdefault(g['group_id'],[]).append(r)
  if r['source_role']=='SOURCE_DEV':dev.add(g['component_id'])
 out=[]
 for r in rows:
  g=r['generated_input'];gg=sorted(groups[g['group_id']],key=lambda x:x['generated_input']['native_slot']);need([x['generated_input']['native_slot']for x in gg]==list(range(8)),'weak same group8')
  q8=[x['generated_input']['generated_query_xy']for x in gg];L8=[x['generated_input']['L']for x in gg];av8=[x['generated_input']['generated_query_available']for x in gg]
  ctx=context(g['generated_query_xy'],g['L'],g['native_slot'],g['generated_query_available'],q8,L8,av8);need(np.array_equal(ctx,np.asarray(r['deployable_condition32'],np.float32)),'weak condition32 scalar recomputation')
  t=-1
  if r['weak_loss_candidate']is True:
   need(r['target_state']in('D_VISIBLE_EXPECTED_PART_WEAK','E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK')and g['generated_query_available']is True,'known expected-part D/E only')
   need(all(r['support_checks'][x]is True for x in ('query_available','nominal_target_in_frame','nominal_target_in_crop96','nominal_target_strict_valid')),'weak support checks separate from native gold')
   t=int(r['target_state']=='E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK');need(t==r['weak_target_E_binary'],'weak scalar target')
  fit=t>=0 and r['source_role']=='SOURCE_FIT';need(fit==(r['weak_loss_fit_candidate']is True),'DEV excluded from weak fit')
  out.append(dict(row_id=r['row_id'],source_role=r['source_role'],component_id=g['component_id'],native_slot=g['native_slot'],available=g['generated_query_available'],condition32=ctx,target=t,fit=fit,RGB96=g['RGB96'],valid96_RLE=g['valid96_RLE']))
 need(len(dev)==4 and Counter(x['target']for x in out if x['fit'])=={0:39,1:6}and Counter(x['target']for x in out if x['source_role']=='SOURCE_DEV'and x['target']>=0)=={0:9,1:3},'weak FIT45/DEV12 class counts')
 return out,dev

def indices(x,n,size):
 need(isinstance(x,list)and len(x)==size and all(type(v)is int and 0<=v<n for v in x),'typed bounded draw indices');return np.asarray(x,np.int64)
def log_audit(logs,metadata,point,weak,dev,steps,weight):
 need(len(logs)==2 and all(len(x)==steps for x in logs),'all2 exact4000 logs')
 masks={d:fit_mask(metadata[d],dev)for d in DOMAINS};pt=np.asarray(point,np.int8);wt=np.asarray([x['target']for x in weak],np.int8);wm=np.asarray([x['fit']for x in weak],bool)
 pp=[np.flatnonzero(masks[DOMAINS[0]]&(pt==k))for k in(0,1)];wp=[np.flatnonzero(wm&(wt==k))for k in(0,1)]
 need(all(len(x)>0 for x in pp+wp),'both metadata-only target classes')
 hashes=[];whashes=[]
 for arm,rr in enumerate(logs):
  prng=np.random.default_rng(1811);wrng=np.random.default_rng(1812);hh=hashlib.sha256();ww=hashlib.sha256()
  for update,r in enumerate(rr,1):
   need(r['update']==update and list(r['old192'])==list(TASKS),'update/key ordering')
   for name,size,d in zip(TASKS,SIZES,(DOMAINS[0],)*3+DOMAINS[1:]):
    ii=indices(r['old192'][name],len(metadata[d]),size);need(masks[d][ii].all(),'every old5 draw fit mask TRUE, no DEV/HOLD/nonNOMINAL/unbound')
    need(all(metadata[d][i]['original_query_available']is True for i in ii),'every old draw available')
    hh.update(name.encode());hh.update(ii.astype('<i8').tobytes())
   pi=indices(r['strong32'],len(point),32);want=np.concatenate([prng.choice(p,16,replace=True)for p in pp]);need(np.array_equal(pi,want),'point seed1811 exact replay');hh.update(pi.astype('<i8').tobytes())
   if arm==0:need(r['weak16']is None and r['weak']==0,'control weak bypass')
   else:
    wi=indices(r['weak16'],len(weak),16);want=np.concatenate([wrng.choice(p,8,replace=True)for p in wp]);need(np.array_equal(wi,want)and wm[wi].all(),'weak seed1812/FIT only exact replay');ww.update(wi.astype('<i8').tobytes())
   need(r['old_point_sha']==hh.hexdigest()and r['weak_sha']==ww.hexdigest(),'every rolling draw hash exact')
   need(r['all_fit_masks_valid']is True and len(r['old5'])==5,'producer all-mask proof crosschecked')
   vals=r['old5']+[r['strong'],r['weak'],r['total'],r['gradient_norm']];need(all(type(x)in(int,float)and math.isfinite(x)for x in vals),'all five/direct/weak/loss/grad finite')
   expected=sum(r['old5'])/5+r['strong']+(weight*r['weak']if arm else 0.)
   need(abs(r['total']-expected)<=1e-4*max(1.,abs(expected)),'saved FP32 objective composition')
   if arm:need(r['old192']==logs[0][update-1]['old192']and r['strong32']==logs[0][update-1]['strong32']and r['old_point_sha']==logs[0][update-1]['old_point_sha'],'all paired old192+strong32 draws exactequal')
  hashes.append(hh.hexdigest());whashes.append(ww.hexdigest())
 need(hashes[0]==hashes[1],'paired final draw hash')
 return dict(completed=True,PASS=True,updates_each=steps,log_rows=steps*2,all_old192_and_point32_draws_equal=True,old192_masks_sizes_hashes_and_paired_equality_exact=True,old192_classconditional_RNG_not_replayed_without_original_target_arrays=True,point_seed1811_all_draws_exact=True,weak_seed1812_all_draws_exact=True,weak_ONLY_second_arm_FIT39D6E=True,all_loss_masks_TRUE=True,finite_loss_and_gradient=True,old_point_draw_sha=hashes[0],weak_draw_sha_by_arm=whashes)

def state_sha(state):
 h=hashlib.sha256()
 for k,t in sorted(state.items()):
  a=t.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
 return h.hexdigest()
def optimizer_audit(state,parameters,steps):
 import torch
 need(len(state['param_groups'])==1,'one AdamW parameter group');ids=state['param_groups'][0]['params'];need(len(ids)==len(parameters)and len(set(ids))==len(ids)and set(state['state'])==set(ids),'all parameter optimizer states present')
 for i,p in zip(ids,parameters):
  r=state['state'][i];need(set(r)=={'step','exp_avg','exp_avg_sq'},'AdamW typed state keys')
  step=r['step'];need(isinstance(step,torch.Tensor)and step.numel()==1 and torch.isfinite(step).all()and step.item()==steps,'all optimizer steps4000')
  for key in ('exp_avg','exp_avg_sq'):need(isinstance(r[key],torch.Tensor)and r[key].dtype==torch.float32 and r[key].shape==p.shape and torch.isfinite(r[key]).all(),'finite matching AdamW tensors')
  need((r['exp_avg_sq']>=0).all(),'AdamW second moment nonnegative')
 return True

def score_agreement(actual,saved):
 a=np.asarray(actual,np.float64);s=np.asarray(saved,np.float64);need(a.shape==s.shape and np.isfinite(a).all()and np.isfinite(s).all(),'finite paired logits')
 err=float(np.max(np.abs(a-s)))if a.size else 0.;need(err<=MAXABS and np.array_equal(a>=0,s>=0),'fixed replay1e-4 and exact >=0 signs');return err

def rle_valid(runs):
 v=np.zeros(9216,bool)
 for start,length in runs:
  need(type(start)is int and type(length)is int and start>=0 and length>0 and start+length<=9216 and not v[start:start+length].any(),'nonoverlap bounded original valid RLE');v[start:start+length]=True
 return v.reshape(96,96)
def module(b,allow,name):
 s=checked(b,allow,('.py',));m=types.ModuleType(name);m.__file__=b['path'];sys.modules[name]=m;exec(compile(s,b['path'],'exec'),m.__dict__);return m

def run(entry_path):
 e=json.loads(raw(entry_path));need(e['approved_actual_pair_QA']is True and e['source_only']is True and e['automatic_retry']is False,'explicit root audit authorization')
 need(same(e['code'],binding(Path(__file__).absolute())),'exact frozen audit code');meta=e['metadata_whitelist'];code=e['code_whitelist'];weights=e['checkpoint_whitelist'];png=e['weak_PNG_whitelist']
 protocol=js(e['protocol'],meta);need(same(protocol['code'],e['code'])and protocol['actual_before_root_entry']is False,'frozen prospective protocol')
 fc=js(e['fit_complete'],meta);freeze=js(e['all2_freeze'],meta);fe=js(e['fit_entry'],meta);recipe=js(e['root_recipe'],meta)
 need(fc['completed']is True and fc['PASS']is True and fc['source_only']is True and fc['from_scratch']is True and fc['steps_each']==4000 and fc['parameters_each']==118359 and fc['DEV_not_fit']is True and fc['protected_access']==0 and fc['selected_arm']is None,'BOTH source finals fixed4000/fresh source-only')
 need(same(fc['entry'],e['fit_entry'])and same(fc['root_recipe'],e['root_recipe'])and same(fc['all2_freeze'],e['all2_freeze']),'fit closure metadata exact')
 need(freeze['completed']is True and freeze['PASS']is True and freeze['DEV_or_real_output_emitted']is False and freeze['selected_arm']is None and set(fc['arms'])==set(ARMS)and set(freeze['checkpoints'])==set(ARMS),'both-final barrier exists before any checkpoint/image/score read')
 need(fe['approved_source_pair_fit_or_probe']is True and fe['mode']=='FIXED_PAIRED_FIT'and fe['source_only']is True and fe['warmstart_checkpoint']is None and recipe['steps']==4000 and recipe['approved_frozen_recipe']is True and recipe['budget_selection_basis']=='THROUGHPUT_ONLY_NO_SCORE_OR_ARM_SELECTION','frozen root fullfit recipe')
 need(recipe['weak_weight']>0 and math.isfinite(recipe['weak_weight']),'prospective positive weak weight')
 for arm in ARMS:need(same(freeze['checkpoints'][arm],fc['arms'][arm]['checkpoint']),'all2 checkpoint binding exact')
 need(len(weights)==2 and {ident(x)for x in weights}=={ident(fc['arms'][a]['checkpoint'])for a in ARMS},'only two own final PT whitelisted')
 for x in weights:need(Path(x['path']).is_relative_to(V/'paired_source_fit_v1')and Path(x['path']).name=='FINAL_CHECKPOINT.pt','own V219 final checkpoint only')
 need(same(fe['weak_rows360'],e['weak_rows360'])and same(fe['point_metadata7040'],e['point_metadata7040']),'actual fit target metadata exact')
 for k in ('code','contract'):need(same(fc[k],fe[k]),'producer code/contract closure')
 producer_protocol=js(fe['protocol'],meta);need(same(producer_protocol['code'],fe['code'])and same(producer_protocol['contract'],fe['contract']),'producer frozen protocol exact')
 need(all(same(fe[k],protocol['producer'][k])for k in ('code','contract','protocol')),'prospectively pinned producer code/contract/protocol')
 need(all(same(producer_protocol['kernels'][k],fe['kernels'][k])and same(fe['kernels'][k],e['kernels'][k])for k in e['kernels']),'factory/kernel closure')
 need(set(e['kernels'])==set(protocol['producer']['kernels'])and all(same(e['kernels'][k],protocol['producer']['kernels'][k])for k in e['kernels']),'pinned factory kernel identities')
 extractor=js(e['selected_extractor_complete'],meta);need(extractor['completed']is True and extractor['PASS']is True,'selected extraction completed metadata; source arrays not reopened')
 lineage=js(fe['lineage_assembly_QA'],meta);need(lineage['completed']is True and lineage['PASS']is True and lineage['all_old_public_CF_descendants_of_new_DEV_excluded']is True and lineage['q8_input_provenance_verified_no_target_geometry_features']is True,'bound lineage QA')
 rr={d:js(fe['old_caches'][d]['metadata'],meta)for d in DOMAINS};weak,dev=weak_rows(js(e['weak_rows360'],meta));need(dev==set(fe['new_DEV_component_ids']),'same fixed DEV components')
 for d,rows in rr.items():
  need(len({r['row_id']for r in rows})==len(rows)and fit_mask(rows,dev).all(),'selected all rows fit masks TRUE')
  for r in rows:
   need(r['context_origin']=='ORIGINAL_INITIALIZER','old q8 input source, not target geometry');context(r['query_xy'],r['L'],r['native_slot'],r['original_query_available'],r['context_q8'],r['context_L8'],r['context_available8'])
 need([len(rr[d])for d in DOMAINS]==[7040,1208,3816],'filtered metadata denominator, no old HOLD/newDEV')
 points=js(e['point_metadata7040'],meta);need(len(points)==7040,'all7040 metadata retained');byid={r['row_id']:i for i,r in enumerate(rr[DOMAINS[0]])};target=np.full(7040,-1,np.int8);seen=set()
 for r in points:
  rid=r['row_id'];need(rid in byid and rid not in seen,'one point per source nominal row');seen.add(rid);m=rr[DOMAINS[0]][byid[rid]];meta_equal(m,r)
  need(r['source_teacher_key']==[m['source_image_original_global_index']//8,m['native_slot']]and r['source_image_original_global_index']==m['source_image_original_global_index']and r['anchor_original_global_index']==m['anchor_original_global_index'],'point actual image vs original anchor')
  target[byid[rid]]=point_target(r)
 need(seen==set(byid)and Counter(target.tolist())=={-1:944,0:5853,1:243}and sum(not r['original_query_available']for r in points)==64,'strong full7040/known6096/U64 unchanged')
 # Logs/outputs are opened only after the complete+barrier gates above.
 logs=[js(e['logs'][arm],meta)for arm in ARMS];la=log_audit(logs,rr,target,weak,dev,4000,recipe['weak_weight'])
 for i,arm in enumerate(ARMS):need(fc['arms'][arm]['old_point_draw_sha']==la['old_point_draw_sha']and fc['arms'][arm]['weak_draw_sha']==la['weak_draw_sha_by_arm'][i],'declared final draw hashes')
 need(freeze['same_old_point_draw_sha']==la['old_point_draw_sha']and len({fc['arms'][a]['initial']for a in ARMS})==1 and freeze['same_initial_state_sha']==fc['arms'][ARMS[0]]['initial'],'shared init/draw seal')
 saved={};strong_saved={};known=[r for r in weak if r['target']>=0];knownmap={r['row_id']:r for r in known}
 need(len(png)==57 and {ident(x)for x in png}=={ident(r['RGB96'])for r in known},'only canonical joined57 PNG paths/SHA, no other source images')
 for arm in ARMS:
  w=js(fc['arms'][arm]['weak57_outputs'],meta);sr=js(fc['arms'][arm]['strong_outputs'],meta);need(w['completed']is True and sr['completed']is True and w['no_primary_native_gold']is True and w['no_original_initializer_performance_claim']is True and sr['source_apparent_not_validation']is True,'source output authority limits')
  saved[arm]={r['row_id']:r for r in w['rows57']};strong_saved[arm]={r['row_id']:r for r in sr['rows']};need(len(saved[arm])==57 and set(saved[arm])==set(knownmap),'all weak57 saved IDs')
  expected={rr[DOMAINS[0]][i]['row_id']:int(t)for i,t in enumerate(target)if t>=0};need(len(strong_saved[arm])==6096 and set(strong_saved[arm])==set(expected),'all strong6096 saved IDs')
  for rid,r in strong_saved[arm].items():need(r['authority']==AUTH_POINT and r['source_role']=='TRAIN'and r['target']==expected[rid]and math.isfinite(r['raw_logit'])and r['prediction_D0_E1']==int(r['raw_logit']>=0),'strong target/role/fixed scalar decoder')
  for rid,r in saved[arm].items():
   m=knownmap[rid];need(r['authority']==AUTH_WEAK and r['source_role']==m['source_role']and r['target']==m['target']and math.isfinite(r['raw_logit'])and r['prediction_D0_E1']==int(r['raw_logit']>=0),'weak target/role/fixed scalar decoder')
 import torch
 from PIL import Image
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 base=module(e['kernels']['base_head'],code,'v219qa_base');candidate=module(e['kernels']['candidate'],code,'v219qa_candidate');spatial=module(e['kernels']['spatial_head'],code,'v219qa_spatial');factory=spatial.make_spatial_q8_head_class(base,candidate)
 torch.manual_seed(1810);initial=factory(spatial_q8=True).float();need(sum(p.numel()for p in initial.parameters())==118359 and len(initial.state_dict())==36,'factory exact118359/36');ish=state_sha(initial.state_dict());need(ish==freeze['same_initial_state_sha'],'fresh CPU seed1810 initialstate exact')
 pixels={}
 for r in known:
  need(r['available']is True,'no inputU forwarded');z=checked(r['RGB96'],png,('.png',));im=Image.open(io.BytesIO(z));need(im.format=='PNG'and im.mode=='RGB'and im.size==(96,96),'canonical RAW96 RGB PNG');im.load();pixels[r['row_id']]=(np.asarray(im).copy(),rle_valid(r['valid96_RLE']))
 results={}
 for i,arm in enumerate(ARMS):
  z=checked(fc['arms'][arm]['checkpoint'],weights,('.pt',));cp=torch.load(io.BytesIO(z),map_location='cpu',weights_only=True);need(cp['update']==4000 and cp['parameters']==118359 and cp['spatial_q8']is True and cp['use_sam']is False and cp['use_weak']==bool(i)and cp['weak_weight']==(recipe['weak_weight']if i else 0.),'typed own checkpoint arm/RAW/context/weak flags')
  need(same(cp['source_entry'],e['fit_entry'])and same(cp['recipe'],e['root_recipe']),'checkpoint sourceentry/recipe closure')
  model=factory(spatial_q8=True).float();need(set(cp['state_dict'])==set(model.state_dict()),'state keys exact')
  for k,p in model.state_dict().items():
   t=cp['state_dict'][k];need(isinstance(t,torch.Tensor)and t.dtype==torch.float32 and t.shape==p.shape and torch.isfinite(t).all(),'typed finite exact state')
  model.load_state_dict(cp['state_dict'],strict=True);model.eval().requires_grad_(False);need(state_sha(cp['state_dict'])==fc['arms'][arm]['final'],'state hash exact')
  optimizer_audit(cp['optimizer_state'],list(model.parameters()),4000);group=cp['optimizer_state']['param_groups'][0];need(group['lr']==recipe['learning_rate']and group['weight_decay']==recipe['weight_decay'],'AdamW recipe exact')
  pred=[]
  with torch.inference_mode():
   for start in range(0,57,32):
    batch=known[start:start+32];rgb=np.asarray([pixels[r['row_id']][0]for r in batch]);valid=np.asarray([pixels[r['row_id']][1]for r in batch]);ct=torch.from_numpy(np.asarray([r['condition32']for r in batch],np.float32))
    out=model(base.assemble_local_input(torch.from_numpy(rgb),torch.from_numpy(valid),use_sam=False),ct[:,:8].argmax(1).to(torch.int64),ct[:,8:16],ct[:,16:32]);pred.extend(out['direct_end_external_logit'].numpy().tolist())
  err=score_agreement(pred,[saved[arm][r['row_id']]['raw_logit']for r in known]);need(state_sha(model.state_dict())==fc['arms'][arm]['final'],'replay state unchanged')
  results[arm]=dict(PASS=True,parameters=118359,state_tensors=36,all_finite=True,optimizer_all_steps4000=True,checkpoint=fc['arms'][arm]['checkpoint'],weak57_CPU_replay_batch32_maxabs=err,weak57_all_saved_signs_exact=True,strong6096_saved_metadata_decoder_exact=True,weak57_authority_role_metadata_exact=True)
  del model,cp
 # Final rehash: opaque own weights/PNG/metadata only; original NPZ never opened.
 for bb in weights:checked(bb,weights,('.pt',))
 for bb in png:checked(bb,png,('.png',))
 out=lexical(e['output_directory']);need(out==H/'actual_review_v1','fixed fresh QA output')
 for parent in [out.parent,*out.parent.parents]:need(not stat.S_ISLNK(os.lstat(parent).st_mode),'output parent no symlink')
 out.mkdir(exist_ok=False)
 review=dict(completed=True,PASS=True,source_only=True,fit_complete=e['fit_complete'],all2_freeze=e['all2_freeze'],fit_entry=e['fit_entry'],selected_extractor_complete=e['selected_extractor_complete'],code=e['code'],protocol=e['protocol'],entry=binding(Path(entry_path)),log_checks=la,initial_seed1810_state_exact=True,all8000_saved_log_rows_verified=True,all_fit_rows_original_TRAIN_nominal_nonDEV_lineage_bound=True,source_metadata_rows={d:len(rr[d])for d in DOMAINS},strong_metadata_retained7040_known6096_unknown944_inputU64=True,weak_FIT45_D39_E6_DEV12_D9_E3=True,results=results,actual_replay_model_forward_calls=4,checkpoint_loads_weights_only_CPU=2,weak_PNG_decodes=57,original_or_selected_source_NPZ_reads=0,GT_or_protected_body_reads=0,training_optimizer_step_calls=0,old192_target_class_distribution_RNG_not_independently_rebuilt=True,old192_validity_claim='All masks, sizes, pair equality and hash exact; original target arrays were not reread, so original class-pool membership/conditional probabilities rely on frozen trainer+selected extractor metadata QA.',strong_logits_not_replayed_without_SOURCE_RGB=True,weak_DEV_internal_preexposed_not_freshblind=True,GT_derived_weak_q8_not_original_initializer_performance=True,weak_not_physical_native_gold=True,no_model_threshold_or_data_selection=True)
 rb=write(out/'REVIEW.json',review);print(json.dumps(dict(completed=True,PASS=True,review=rb,results=results),ensure_ascii=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);a=ap.parse_args();run(a.entry)
