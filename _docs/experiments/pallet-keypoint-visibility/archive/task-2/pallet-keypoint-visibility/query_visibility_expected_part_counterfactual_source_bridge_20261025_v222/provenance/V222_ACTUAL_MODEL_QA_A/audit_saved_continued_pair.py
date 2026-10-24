"""V222 saved continuation-pair QA; CPU replay only after explicit root entry and both-final seal.
No old NPZ/source GT/protected data, training, optimizer step or checkpoint selection.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, io, json, math, os, stat, sys, types
import numpy as np
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_expected_part_counterfactual_source_bridge_20261025_v222'
H=P/'publication_root_preparation/V222_ACTUAL_MODEL_QA_A'
ARMS=('C66CONTINUATION_CONTROL','C67PLUS_EXPECTED_PART_CF_WEAK')
DOMAINS=('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816')
TASKS=('visible','foreign','synthetic_owner','public_owner','CF_owner');SIZES=(32,32,48,48,32)
AUTH_POINT='SYNTH_RENDERED_OWN_IMAGE_POINT0_TARGET_ALONE_D_COMBINED_DE'
AUTH_WEAK='ASSISTANT_CONSENSUS_EXPECTED_FINITE_PART_SOURCE_GT_QUERY'
AUTH_CF='CONTROLLED_EXPECTED_FINITE_PART_WEAK_NOT_PHYSICAL_NATIVE_POINT_GOLD'
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
MAXABS=1e-4
STEPS=1100
PRODUCER={'code':'1e88783241417b97554bdc5b33225e0d6b8c2ae937d0f0288575acd1c26b3957','contract':'40fb7d9e1994d7cd1aa30d063c19d785789062b7e6176af9cdb3ed5bde973c94','protocol':'880483f45aa981da588c65454f98a54b0d1db2398a4591caf4c24a6de530b16e'}

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


def cf_metadata(rows,weak,original):
 need(len(rows)==144 and len({r['row_id']for r in rows})==144,'all144 unique CF status rows')
 wm={r['row_id']:r for r in weak};gm={r['row_id']:r['generated_input']for r in original};groups={};out=[]
 for r in rows:
  p=r['parent_row_id'];w=wm[p];g=gm[p];need(w['target']==0 and r['row_id']==p+'__'+r['variant'],'original D48 parent/variant')
  need(r['variant']in('BEFORE','COVER','SHAM')and r['authority']==AUTH_CF and r['native_point_DE_target']==-1 and r['training_admission']is False,'CF weak authority separate from native')
  need(r['source_role']==w['source_role']and r['component_id']==w['component_id']and r['native_slot']==w['native_slot'],'CF original role/component/native')
  need(r['query_xy']==g['generated_query_xy']and r['L']==g['L']and r['query_available']is g['generated_query_available'],'same original named-family q/L/U')
  need(np.array_equal(np.asarray(r['condition32'],np.float32),w['condition32'])and r['valid96_RLE']==w['valid96_RLE'],'same original condition32/valid')
  need(same(r['parent_RGB96_binding'],w['RGB96']),'exact inherited D parent PNG')
  groups.setdefault(p,{})[r['variant']]=r
  out.append(dict(row_id=r['row_id'],parent_row_id=p,variant=r['variant'],source_role=r['source_role'],component_id=r['component_id'],native_slot=r['native_slot'],query_xy=r['query_xy'],L=r['L'],available=r['query_available'],condition32=np.asarray(r['condition32'],np.float32),target=-1 if r['weak_finite_part_target_D0_E1']is None else r['weak_finite_part_target_D0_E1'],RGB96=r['RGB96'],valid96_RLE=r['valid96_RLE'],constructed=r['constructed']))
 need(len(groups)==48 and Counter(next(iter(g.values()))['source_role']for g in groups.values())=={'SOURCE_FIT':39,'SOURCE_DEV':9},'all48 original roles')
 for p,rr in groups.items():
  need(set(rr)=={'BEFORE','COVER','SHAM'},'three statuses per parent');need(rr['BEFORE']['weak_finite_part_target_D0_E1']==0 and isinstance(rr['BEFORE']['RGB96'],dict),'failed-parent BEFORE retained')
  if rr['COVER']['constructed']:
   need(all(rr[v]['constructed']and rr[v]['weak_finite_part_target_D0_E1']==t and rr[v]['input_available']is True for v,t in zip(('BEFORE','COVER','SHAM'),(0,1,0))),'constructed weak D/E/D')
  else:
   need(all(rr[v]['RGB96']is None and rr[v]['weak_finite_part_target_D0_E1']is None for v in ('COVER','SHAM')),'unconstructed variants UNKNOWN')
 return out,sorted(p for p,rr in groups.items()if rr['COVER']['constructed']and rr['BEFORE']['source_role']=='SOURCE_FIT')

def log_check(logs,metadata,point,weak,cfparents,dev,steps):
 need(all(len(x)==steps for x in logs),'both1100 complete logs');masks={d:fit_mask(metadata[d],dev)for d in DOMAINS}
 pp=[np.flatnonzero(np.asarray(point)==k)for k in(0,1)];wp=[np.asarray([i for i,r in enumerate(weak)if r['fit']and r['target']==k])for k in(0,1)]
 final=[]
 for arm,rr in enumerate(logs):
  prng=np.random.default_rng(1811);wrng=np.random.default_rng(1812);crng=np.random.default_rng(1813);hh=hashlib.sha256();ww=hashlib.sha256();ch=hashlib.sha256()
  for u,r in enumerate(rr,1):
   need(r['update']==u and list(r['old192'])==list(TASKS),'every update/key ordering')
   for task,size,d in zip(TASKS,SIZES,(DOMAINS[0],)*3+DOMAINS[1:]):
    ii=indices(r['old192'][task],len(metadata[d]),size);need(masks[d][ii].all()and all(metadata[d][i]['original_query_available']is True for i in ii),'all old5 draw indices fit/available/no DEV')
    hh.update(task.encode());hh.update(ii.astype('<i8').tobytes())
   pi=indices(r['strong32'],len(point),32);need(np.array_equal(pi,np.concatenate([prng.choice(p,16,replace=True)for p in pp])),'point1811 exact');hh.update(pi.astype('<i8').tobytes())
   wi=indices(r['naturalweak16'],len(weak),16);need(np.array_equal(wi,np.concatenate([wrng.choice(p,8,replace=True)for p in wp]))and all(weak[i]['fit']for i in wi),'both natural1812 exact/FIT39D6E');ww.update(wi.astype('<i8').tobytes())
   if arm:
    ci=indices(r['CF4parents'],len(cfparents),4);need(np.array_equal(ci,crng.choice(len(cfparents),4,replace=True))and r['CF_parent_row_ids']==[cfparents[i]for i in ci],'CF1813 exact4x3/only admitted39FIT');ch.update(ci.astype('<i8').tobytes())
   else:need(r['CF4parents']is None and r['CF_parent_row_ids']is None and r['CF_weighted_parent_BCE']==0,'control no CF loss')
   need(r['old_point_sha']==hh.hexdigest()and r['naturalweak_sha']==ww.hexdigest()and r['CF_sha']==ch.hexdigest(),'all3 rolling hashes independently exact')
   vals=r['old5']+[r[k]for k in ('strong','naturalweak','CF_weighted_parent_BCE','total','gradient_norm')];need(len(r['old5'])==5 and all(type(x)in(int,float)and math.isfinite(x)for x in vals)and r['all_fit_masks_valid']is True,'all loss/gradient finite and mask flag')
   value=sum(r['old5'])/5+r['strong']+.25*r['naturalweak']+(.25*r['CF_weighted_parent_BCE']if arm else 0.)
   need(abs(r['total']-value)<=1e-4*max(1.,abs(value)),'saved FP32 full loss composition')
   if arm:need(all(r[k]==logs[0][u-1][k]for k in('old192','strong32','naturalweak16','old_point_sha','naturalweak_sha')),'both common draws exactly equal')
  final.append(dict(old_point=hh.hexdigest(),natural=ww.hexdigest(),CF=ch.hexdigest()))
 need(final[0]['old_point']==final[1]['old_point']and final[0]['natural']==final[1]['natural'],'paired final hashes')
 return dict(PASS=True,updates_each=steps,all_log_rows=2*steps,all_common_old192_point32_natural16_equal=True,point1811_natural1812_CF1813_exact=True,CF_only39FIT_parents_fourtriplets_perstep=True,all_loss_masks_and_finite_gradient_logs=True,hashes_by_arm=final,old192_classconditional_RNG_not_rebuilt_without_target_NPZ=True,per_variant_training_logits_not_saved_loss_formula_bound_to_reviewed_code=True)

def metrics(rows):
 known=[r for r in rows if r['target']in(0,1)];cm=np.zeros((2,3),np.int64)
 for r in known:cm[r['target'],r['prediction']if r['prediction']in(0,1)else 2]+=1
 pr=[]
 for k in(0,1):
  tp=int(cm[k,k]);denp=int(cm[:,k].sum());denr=int(cm[k].sum());pr.append(dict(class_D0_E1=k,precision=None if not denp else tp/denp,recall=None if not denr else tp/denr))
 return dict(rows=len(rows),known=len(known),UNKNOWN=len(rows)-len(known),CM_truthDE_predDEU=cm.tolist(),correct=int(cm[0,0]+cm[1,1]),P_R=pr)

def reduce_output(rows):
 return dict(all=metrics(rows),by_role={s:metrics([r for r in rows if r['source_role']==s])for s in sorted({r['source_role']for r in rows})},by_native={str(k):metrics([r for r in rows if r['native_slot']==k])for k in range(8)})

def run(entry_path):
 e=json.loads(raw(entry_path));need(e['approved_actual_pair_QA']is True and e['source_only']is True and e['automatic_retry']is False,'separate root actual approval')
 need(same(e['code'],binding(Path(__file__).absolute())),'frozen QA code');meta=e['metadata_whitelist'];code=e['code_whitelist'];weights=e['checkpoint_whitelist'];png=e['CF_PNG_whitelist']
 protocol=js(e['protocol'],meta);need(same(protocol['code'],e['code'])and protocol['actual_before_root_entry']is False and protocol['replay_maxabs']==MAXABS,'frozen prospective replay contract')
 fc=js(e['fit_complete'],meta);freeze=js(e['all2_freeze'],meta);fe=js(e['fit_entry'],meta);recipe=js(e['root_recipe'],meta)
 need(fc['completed']is True and fc['PASS']is True and fc['source_only']is True and fc['from_scratch']is False and fc['optimizer_state_reset']is True and fc['steps_each']==STEPS and fc['parameters_each']==118359 and fc['protected_access']==0 and fc['DEV_not_fit']is True and fc['selected_arm']is None,'both1100 completed source-only continuation')
 need(all(same(fc[k],e[q])for k,q in(('entry','fit_entry'),('all2_freeze','all2_freeze'),('root_recipe','root_recipe'))),'fit metadata closure')
 need(fe['approved_source_pair_fit_or_probe']is True and fe['approved_load_exact_C65_into_both_arms']is True and fe['mode']=='FIXED_PAIRED_FIT'and recipe['approved_frozen_recipe']is True and recipe['steps']==STEPS and recipe['budget_selection_basis']=='THROUGHPUT_ONLY_NO_SCORE_OR_ARM_SELECTION','exact actual fit entry/recipe')
 need(recipe['natural_weak_weight']==recipe['CF_weak_weight']==.25 and recipe['CF_parents_per_step']==4,'fixed BCE recipe')
 need(freeze['completed']is True and freeze['PASS']is True and freeze['DEV_or_real_output_emitted']is False and freeze['selected_arm']is None and set(freeze['checkpoints'])==set(fc['arms'])==set(ARMS),'both final barrier before logs/outputs/weights/image reads')
 for k in ('code','contract','protocol'):need(fe[k]['sha256']==PRODUCER[k],'prospectively pinned producer V2 '+k)
 need(same(fc['code'],fe['code'])and same(fc['contract'],fe['contract']),'producer closure')
 need(len(weights)==2 and {ident(x)for x in weights}=={ident(fc['arms'][a]['checkpoint'])for a in ARMS},'only two own final PT')
 for arm in ARMS:need(same(freeze['checkpoints'][arm],fc['arms'][arm]['checkpoint']),'checkpoint seal exact')
 for b in weights:need(Path(b['path']).is_relative_to(V/'paired_source_fit_v1')and Path(b['path']).name=='FINAL_CHECKPOINT.pt','own V222 final weights only')
 parent=js(fe['reference_C65_fit_complete'],meta);c65=parent['arms']['STRONG_PLUS_EXPECTED_PART_WEAK'];need(parent['completed']is True and parent['PASS']is True and same(c65['checkpoint'],fe['warmstart_checkpoint'])and same(fe['warmstart_checkpoint'],fc['continued_from_same_C65_checkpoint']),'designated C65 source continuation, no old C65 PT decode')
 need(freeze['same_initial_state_sha']==c65['final']and all(fc['arms'][a]['initial']==c65['final']for a in ARMS),'both init tensor hashes exact trained C65')
 probe=js(fe['throughput_probe_complete'],meta);slow=max(probe['step_seconds_by_arm'].values());want=max(200,min(2000,100*math.floor((60/slow)/100)))
 need(probe['completed']is True and probe['PASS']is True and probe['steps_each']==32 and probe['checkpoint_saved']is False and same(probe['code'],fe['code'])and same(probe['C65_checkpoint'],fe['warmstart_checkpoint'])and want==STEPS,'disposable probe budget exact')
 rr={d:js(fe['old_caches'][d]['metadata'],meta)for d in DOMAINS};original=js(fe['weak_rows360'],meta);weak,dev=weak_rows(original)
 need(dev==set(fe['new_DEV_component_ids'])and [len(rr[d])for d in DOMAINS]==[7040,1208,3816]and all(fit_mask(rr[d],dev).all()for d in DOMAINS),'same fit-only metadata, all old5/norm DEV excluded')
 points=js(fe['point_metadata7040'],meta);byid={r['row_id']:i for i,r in enumerate(rr[DOMAINS[0]])};target=np.full(7040,-1,np.int8);seen=set()
 for r in points:
  need(r['row_id']in byid and r['row_id']not in seen,'unique7040 point identity');seen.add(r['row_id']);i=byid[r['row_id']];m=rr[DOMAINS[0]][i];meta_equal(m,r)
  need(r['source_teacher_key']==[m['source_image_original_global_index']//8,m['native_slot']]and r['source_image_original_global_index']==m['source_image_original_global_index']and r['anchor_original_global_index']==m['anchor_original_global_index'],'own-image teacher/anchor separate');target[i]=point_target(r)
 need(len(seen)==7040 and Counter(target.tolist())=={-1:944,0:5853,1:243},'full7040/known6096 preserved')
 cfr=js(fe['CF_rows144'],meta);cf,cfparents=cf_metadata(cfr,weak,original);need(len(cfparents)==39,'only39 successful FIT parents')
 admission=js(fe['CF_target_admission'],meta);need(admission['approved_CF_FIT_weak_only']is True and set(admission['allowed_parent_ids'])==set(cfparents),'exact root weak-only admission39')
 logs=[js(e['logs'][a],meta)for a in ARMS];la=log_check(logs,rr,target,weak,cfparents,dev,STEPS)
 for i,arm in enumerate(ARMS):
  a=fc['arms'][arm];need(a['old_point_draw_sha']==la['hashes_by_arm'][i]['old_point']and a['naturalweak_draw_sha']==la['hashes_by_arm'][i]['natural']and a['CF_draw_sha']==la['hashes_by_arm'][i]['CF'],'all final3 hashes exact')
 need(freeze['same_old_point_draw_sha']==la['hashes_by_arm'][0]['old_point']and freeze['same_naturalweak_draw_sha']==la['hashes_by_arm'][0]['natural'],'common draw seal')
 cfmap={r['row_id']:r for r in cf};knownweak={r['row_id']:r for r in weak if r['target']>=0};pointmap={rr[DOMAINS[0]][i]['row_id']:(i,int(t))for i,t in enumerate(target)if t>=0};saved={};reductions={}
 for arm in ARMS:
  rec=fc['arms'][arm];a=js(rec['CF_all144_outputs'],meta);w=js(rec['weak57_outputs'],meta);s=js(rec['strong_outputs'],meta)
  need(a['completed']is True and w['completed']is True and s['completed']is True,'all saved source outputs complete');saved[arm]={r['row_id']:r for r in a['rows144']};need(len(saved[arm])==144 and set(saved[arm])==set(cfmap),'all144 source IDs')
  cfout=[]
  for rid,r in saved[arm].items():
   m=cfmap[rid];need(r['authority']==AUTH_CF and r['native_point_DE_target']==-1 and r['source_role']==m['source_role']and r['component_id']==m['component_id']and r['native_slot']==m['native_slot']and r['query_xy']==m['query_xy']and r['L']==m['L']and r['query_available']==m['available']and r['target']==(None if m['target']<0 else m['target']),'CF saved authority/META/target')
   avail=m['RGB96']is not None;z=r['raw_logit'];p=r['prediction_D0_E1_Uminus1'];need(r['input_available']==avail and ((z is None and p==-1)if not avail else math.isfinite(z)and p==int(z>=0)),'CF fixed decoder/U');cfout.append(dict(**m,prediction=p))
  natural=[];wm={r['row_id']:r for r in w['rows57']};need(len(wm)==57 and set(wm)==set(knownweak),'natural57 exact IDs')
  for rid,r in wm.items():
   m=knownweak[rid];need(r['authority']==AUTH_WEAK and r['source_role']==m['source_role']and r['target']==m['target']and math.isfinite(r['raw_logit'])and r['prediction_D0_E1']==int(r['raw_logit']>=0),'natural authority/fixed decoder');natural.append(dict(**m,prediction=r['prediction_D0_E1']))
  strong=[];sm={r['row_id']:r for r in s['rows']};need(len(sm)==6096 and set(sm)==set(pointmap),'strong6096 exact IDs')
  for rid,r in sm.items():
   i,t=pointmap[rid];need(r['authority']==AUTH_POINT and r['source_role']=='TRAIN'and r['target']==t and math.isfinite(r['raw_logit'])and r['prediction_D0_E1']==int(r['raw_logit']>=0),'point authority/fixed decoder');strong.append(dict(target=t,prediction=r['prediction_D0_E1'],source_role='TRAIN',native_slot=rr[DOMAINS[0]][i]['native_slot']))
  paircounts={}
  for role in('SOURCE_FIT','SOURCE_DEV'):
   parents=sorted({m['parent_row_id']for m in cf if m['source_role']==role and m['constructed']});triplets=[]
   for p in parents:
    b,c,h=[saved[arm][p+'__'+v]for v in('BEFORE','COVER','SHAM')];triplets.append(dict(all_D_E_D_correct=[b['prediction_D0_E1_Uminus1'],c['prediction_D0_E1_Uminus1'],h['prediction_D0_E1_Uminus1']]==[0,1,0],cover_minus_before=c['raw_logit']-b['raw_logit'],sham_minus_before=h['raw_logit']-b['raw_logit'],SHAM_false_E=h['prediction_D0_E1_Uminus1']==1))
   paircounts[role]=dict(parents=len(parents),all_D_E_D_correct=sum(x['all_D_E_D_correct']for x in triplets),SHAM_false_E=sum(x['SHAM_false_E']for x in triplets),mean_cover_minus_before=None if not triplets else float(np.mean([x['cover_minus_before']for x in triplets])),mean_sham_minus_before=None if not triplets else float(np.mean([x['sham_minus_before']for x in triplets])))
  reductions[arm]=dict(strong6096=reduce_output(strong),natural57=reduce_output(natural),CF144=reduce_output(cfout),CF_triplets=paircounts)
 need({ident(b)for b in png}=={ident(r['RGB96'])for r in cf if r['RGB96']is not None},'only actual available CF144 PNG bindings')
 import torch
 from PIL import Image
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 kernels=fe['kernels'];base=module(kernels['base_head'],code,'v222qa_base');candidate=module(kernels['candidate'],code,'v222qa_candidate');spatial=module(kernels['spatial_head'],code,'v222qa_spatial');factory=spatial.make_spatial_q8_head_class(base,candidate)
 pixels={}
 for r in cf:
  if r['RGB96']is None:continue
  need(r['available']is True,'no inputU forward');z=checked(r['RGB96'],png,('.png',))
  with Image.open(io.BytesIO(z))as im:need(im.format=='PNG'and im.mode=='RGB'and im.size==(96,96),'saved CF clean96');im.load();rgb=np.asarray(im).copy()
  pixels[r['row_id']]=(rgb,rle_valid(r['valid96_RLE']))
 results={}
 for i,arm in enumerate(ARMS):
  cp=torch.load(io.BytesIO(checked(fc['arms'][arm]['checkpoint'],weights,('.pt',))),map_location='cpu',weights_only=True)
  need(cp['update']==STEPS and cp['parameters']==118359 and cp['spatial_q8']is True and cp['use_sam']is False and cp['use_weak']is True and cp['weak_weight']==.25 and cp['use_CF']==bool(i)and cp['CF_weight']==(.25 if i else 0.)and cp['CF_authority']==AUTH_CF,'typed final RAW/context/natural/CF flags')
  need(same(cp['source_entry'],e['fit_entry'])and same(cp['recipe'],e['root_recipe'])and same(cp['parent_C65_checkpoint'],fe['warmstart_checkpoint']),'checkpoint source/recipe/C65 closure');model=factory(spatial_q8=True).float();need(sum(p.numel()for p in model.parameters())==118359 and len(model.state_dict())==36,'same model118359/36')
  need(set(cp['state_dict'])==set(model.state_dict()),'exact typed state keys')
  for k,p in model.state_dict().items():
   z=cp['state_dict'][k];need(isinstance(z,torch.Tensor)and z.dtype==torch.float32 and z.shape==p.shape and torch.isfinite(z).all(),'finite FP32 state tensors')
  model.load_state_dict(cp['state_dict'],strict=True);model.eval().requires_grad_(False);need(state_sha(cp['state_dict'])==fc['arms'][arm]['final'],'final tensor hash');optimizer_audit(cp['optimizer_state'],list(model.parameters()),STEPS);group=cp['optimizer_state']['param_groups'][0];need(group['lr']==recipe['learning_rate']and group['weight_decay']==recipe['weight_decay'],'AdamW reset typed steps/hyperparameters')
  replay=[];order=[r for r in cf if r['RGB96']is not None]
  with torch.inference_mode():
   for start in range(0,len(order),32):
    batch=order[start:start+32];rgb=np.asarray([pixels[r['row_id']][0]for r in batch]);valid=np.asarray([pixels[r['row_id']][1]for r in batch]);ct=torch.from_numpy(np.asarray([r['condition32']for r in batch],np.float32));output=model(base.assemble_local_input(torch.from_numpy(rgb),torch.from_numpy(valid),use_sam=False),ct[:,:8].argmax(1).to(torch.int64),ct[:,8:16],ct[:,16:32]);replay.extend(output['direct_end_external_logit'].numpy().tolist())
  err=score_agreement(replay,[saved[arm][r['row_id']]['raw_logit']for r in order]);need(state_sha(model.state_dict())==fc['arms'][arm]['final'],'replay state unchanged');results[arm]=dict(PASS=True,checkpoint=fc['arms'][arm]['checkpoint'],parameters=118359,state_tensors=36,finite=True,AdamW_all_steps=STEPS,CF_CPU_replay_rows=len(order),CF_CPU_replay_batch32_maxabs=err,all_saved_signs_exact=True);del cp,model
 for b in weights:checked(b,weights,('.pt',))
 for b in png:checked(b,png,('.png',))
 out=lexical(e['output_directory']);need(out==H/'actual_review_v1','fixed own output');fd=os.open(out.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in out.parts[1:-1]:n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  os.mkdir(out.name,dir_fd=fd)
 finally:os.close(fd)
 result=write(out/'REVIEW.json',dict(completed=True,PASS=True,source_only=True,fit_complete=e['fit_complete'],all2_freeze=e['all2_freeze'],fit_entry=e['fit_entry'],code=e['code'],protocol=e['protocol'],entry=binding(Path(entry_path)),log_checks=la,source_reductions=reductions,checkpoint_replay=results,warm_C65_same_initial_tensor_hash_exact=True,optimizer_reset_supported_by_final_1100_step_counters_and_reviewed_code=True,source_metadata_rows={d:len(rr[d])for d in DOMAINS},strong7040_known6096_unknown944_retained=True,weak_FIT45_D39E6_DEV12_D9E3=True,CF_all144_native_targets_minus1=True,both_final_before_DEV_gate_and_checkpoint_bindings_exact=True,old_NPZ_or_GT_or_protected_reads=0,new_final_checkpoint_loads=2,CF_PNG_decodes=len(pixels),training_optimizer_step_calls=0,limits=['Old192 target-class/conditional-probability RNG is not independently rebuilt without old target NPZ; every index/mask/paired equality/rolling hash is checked.','Per-update individual logits/gradients are not saved: reviewed parent BCE formula, draw layout, scalar total composition and finite gradient logs are checked, without replaying training.','Strong6096 and natural57 logits are reduced and metadata/decoders checked, not RGB replayed. CF saved logits only are CPU replayed with fixed maxabs1e-4/sign-exact.','Freeze-before-DEV is checked from pinned producer code order and saved all2 seal/outputs graph; no independent process-event tracing.','SOURCE DEV is historically exposed/correlated; finite-part weak is not native-point or natural external-visibility gold. No checkpoint/model/threshold selection.']))
 print(json.dumps(dict(completed=True,PASS=True,review=result,checkpoint_replay=results),ensure_ascii=False))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--entry',required=True);run(parser.parse_args().entry)
