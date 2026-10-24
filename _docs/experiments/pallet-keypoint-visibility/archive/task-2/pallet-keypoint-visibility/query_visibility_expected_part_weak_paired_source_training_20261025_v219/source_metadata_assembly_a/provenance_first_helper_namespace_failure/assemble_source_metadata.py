"""Bound SOURCE metadata assembly only. Never stats/opens any binary binding."""
from pathlib import Path
from collections import Counter,defaultdict
import ast,argparse,csv,hashlib,json,math,os,stat,sys,types
import numpy as np
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_expected_part_weak_paired_source_training_20261025_v219';H=V/'source_metadata_assembly_a'
A=P/'query_visibility_query_jitter_paired_source_training_20261024_v183'
R=P/'query_visibility_local_support_raw_head_20261024_v181'
S=P/'query_visibility_source_group_split_direct_end_training_20261024_v194'
C=P/'publication_root_preparation/V219_TRAINING_PREPARATION_C'
Q=P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202'
T=P/'query_visibility_dense_source_native_point_teacher_20261025_v211'
D=P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212'
W=P/'query_visibility_public_train_intended_part_query_targets_20261025_v218'
DOMAINS=('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816');META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
def b(p,s):return dict(path=str(p),sha256=s)
B={
'scope':b(V/'ROOT_PROSPECTIVE_SOURCE_ONLY_WEAK_PAIRED_FIT_SCOPE.json','1c7e15525713f981410828c604dea47ce9c25226a482bd62babdb3b21bd1f3f7'),
'current_ready':b(C/'FINAL_READY.json','832e04ebc781ada27dc0d01cb9a4e60cec05bd3e9f3cdff1945fd199555c707a'),
'current_contract':b(C/'source_training_contract.py','fc0a161e8a4c1a413d528ce1e66eed1b2addf2d4593c632506280f10732451d7'),
'current_trainer':b(C/'train_strong_plus_expected_part_weak.py','11c40cb6edfb5260310e2370ff06f4ac04a30d3d69e3f892f3bcd9709696a06d'),
'current_protocol':b(C/'PROTOCOL_BEFORE_ANY_V219_SOURCE_PAYLOAD_OR_FIT.json','f3b769daf7f5419db9ff77a28718db49a804fecc152a7691bbe3f8eb18a39540'),
'false_entry':b(C/'ROOT_ACTUAL_V219_SOURCE_PAIR_ENTRY_FALSE_TEMPLATE.json','6fd89e1e9bc2bde06e8051a1c7d9a6d38fd9e769752cb20d098c6fee6e835cab'),
'minimum_payload':b(C/'MINIMUM_FUTURE_PAYLOAD_BINDINGS_NO_PAYLOAD_ACCESS.json','4fe1508fb50bc2fe910e98f856047c32fb6d36cfad117300956cc3b7b04dd0e9'),
'context_code':b(Q/'context_adapter_a/original_q8_context_adapter.py','b4c3134970004fcfb9351f35b16bc5a81c75ebd39cad9ee98784fca7059de5b6'),
'point_contract':b(D/'fit_preparation_a/source_target_contract.py','d8925908d00f4f120bd80bae7a691a1a3846c4e9096e2e111d71984843e0b738'),
'point_ready':b(D/'fit_preparation_a/FINAL_READY.json','1ef23fb991e042b26c1670154f1764a8dd34c7a6eec8e943bd68f6574163a073'),
'augmentation_complete':b(A/'augmentation_training_inputs_v1/COMPLETE.json','beadda20f8e9534e4c8efa2a0b479a3e0dc101188d4a66d7b899da3f4d3d746f'),
'public_CF_complete':b(R/'source_raw96_cache_v1/COMPLETE.json','b49f641551190aa27fd520acce51c5a1f0a9acc002b91c5b0b1709d4946cd3e4'),
'split_complete':b(S/'source_component_split_v1/COMPLETE.json','f2dd0929ad73e843f2d0044c296c4c54f618ead69899400e75b2b6bac486d445'),
'point_complete':b(T/'dense_source_point_teacher_v1/COMPLETE.json','76f204d49bef1ec26cac0f8115cb9711a0a7a8ac7a8049c6c35969cfaca3fa79'),
'point_rows':b(T/'dense_source_point_teacher_v1/ALL7040_INPUT_SOURCE_POINT_MAPPING.jsonl','aeb958d6982717a22fd777d25c3b675325ea1a46c8f644941d8f86f0ea4aa7a9'),
'weak_complete':b(W/'joined_weak_intended_part_targets_v1/COMPLETE.json','8252865691b60b25d4d05c89036c8e0300265c80b5a1161c5c15fc2d31592825'),
'weak_rows':b(W/'joined_weak_intended_part_targets_v1/ALL360_CHECKED_WEAK_TARGET_CANDIDATES.jsonl','52ac5bb8d0fb19a1329b9bdb7f7b56964fed954a2a4d50489e3dcb7d598d139d'),
'original_scenes':b(A/'augmentation_preparation_a/SCENE512_ORIGINALS_INPUT_ONLY.jsonl','f5942772e7b0072c398d2abb1c009c43276110d5945e3f4853095fe336c99285'),
'SYNTH_AUG65536':b(A/'augmentation_source_train_v1/AUG65536_ROWS.jsonl','33863d44b0648d498e1e913a0a8e708becb0edeb8230f8522e10ab84b350959d'),
'PUBLIC_TRAIN1480':b(R/'source_raw96_cache_v1/PUBLIC_TRAIN1480_ROWS.jsonl','08754bf88a98fcdc1a56cd0d84d6dd40804ff9de2f1ad7693a9e5bdd1fa31ff0'),
'CF3816':b(R/'source_raw96_cache_v1/CF3816_ROWS.jsonl','c0499c1940a3f51d74e98d61f0cb45c6d68fd974bc4eb875a4166b264763530c')}
for domain,sha in zip(DOMAINS,('7d695977ecab1f9f5659f787c3cc7626b95f34d4654186fc8dc23204d679fde9','b04d5d7b1779b3df346a3fe6db26074b7376733317212471eef838439d72e0cc','c69ea16911554e6ce1afd87f29bce4e1530bc4e76b3767f1b9dfa197fa38b0a9')):B[domain+'_roles']=b(S/'source_component_split_v1'/(domain+'_SPLIT_ROWS.jsonl'),sha)
ACCESS=[]
def need(ok,m):
 if not bool(ok):raise ValueError(m)
def lexical(p):
 p=Path(p);need(p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).casefold(),'literal nonV112 path before FS');return p
def raw_metadata(p):
 p=lexical(p);need(p.suffix in ('.json','.jsonl','.py','.md','.csv'),'metadata/code only before FS')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(leaf,'rb') as f:need(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'regular metadata file');return f.read()
def read(key,text=False):
 bb=B[key];z=raw_metadata(bb['path']);need(hashlib.sha256(z).hexdigest()==bb['sha256'],'metadata SHA '+key)
 x=z.decode() if text else ([json.loads(a)for a in z.splitlines()if a.strip()]if Path(bb['path']).suffix=='.jsonl' else json.loads(z))
 ACCESS.append(dict(key=key,**bb,size_bytes=len(z),physical_records=len(x)if isinstance(x,list)else None));return x
def binding(p):
 z=raw_metadata(p);return b(p,hashlib.sha256(z).hexdigest())
def write(p,x):
 lexical(p)
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
 return binding(p)
def jsonl(p,rows):
 lexical(p)
 with p.open('x')as f:
  for x in rows:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
 return binding(p)
def pure(key,names):
 s=read(key,True);tree=ast.parse(s);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
 need({n.name for n in nodes}==set(names),'exact pure function names')
 ns=dict(np=np,math=math,Counter=Counter,DOMAINS=DOMAINS+('DIRECT_CONTROL',))
 exec(compile(ast.Module(body=nodes,type_ignores=[]),B[key]['path'],'exec'),ns);return types.SimpleNamespace(**ns)
def meta(r):return {k:r[k]for k in META}
def eq(a,c):
 for k in META:
  need(a[k]==c[k], 'exact META6 '+k)
def fixture(cc,aa):
 dev={'d'*64};rr=[dict(component_role=r,nominal_fit_eligible=n,component_id=i,lineage_bound=l)for r,n,i,l in [('TRAIN',True,'a'*64,True),('HOLDOUT',True,'a'*64,True),('TRAIN',False,'a'*64,True),('TRAIN',True,'d'*64,True),('TRAIN',True,None,False)]]
 need(cc.fit_mask(rr,dev).tolist()==[True,False,False,False,False],'DEV/role/nominal/unbound mask fixture')
 out=cc.context32(None,None,7,False,[None]*8,[None]*8,[False]*8);need(out[7]==1 and not out[8:].any(),'U retained and spatial zero')
 q8=[[0.,0.],[100.,-100.]]+[[0.,0.]]*6;av=[True,True]+[False]*6
 z=cc.context32([0.,0.],1.,0,True,q8,[1.]*8,av);need(z[8]==0 and z[9]==1 and z[18:20].tolist()==[8.,-8.],'F64 clipped self-mask fixture')
 need(aa.fit_nominal_eligible('SYNTH_AUG65536',dict(family='SELF_SCENE15',state_index=0,state_name='NOMINAL'))and not aa.fit_nominal_eligible('SYNTH_AUG65536',dict(family='SELF_SCENE15',state_index=1,state_name='JITTER')),'nominal fixture')
 return dict(completed=True,PASS=True,checks=['role_DEV_nominal_unknown_lineage_exclusion','original_U_not_filled','F64_clip_selfmask','nominal_only_SELF'],actual_binary_reads=0)
def main():
 scope=read('scope');ready=read('current_ready');entry=read('false_entry');minimum=read('minimum_payload')
 read('current_trainer',True);read('current_protocol');read('point_ready')
 cc=pure('current_contract',['need','context32','weak_plan','fit_mask'])
 aa=pure('context_code',['require','same','native_index','bool_flag','q64','build_context_vectors','fit_nominal_eligible','meta6','assert_meta6','ordered8'])
 pp=pure('point_contract',['need','finite_pair','point_loss_target'])
 checks=fixture(cc,aa);checkbinding=write(H/'GENERATED_PURE_METADATA_CHECKS.json',checks)
 completed={k:read(k)for k in ['augmentation_complete','public_CF_complete','split_complete','point_complete','weak_complete']}
 need(all(x.get('completed')is True and x.get('PASS',True)is True for x in completed.values()),'completed input authorities')
 need(completed['augmentation_complete']['metadata']==B[DOMAINS[0]]and completed['augmentation_complete']['scene_metadata']==B['original_scenes'],'augmentation metadata receipt')
 old={DOMAINS[0]:{k:completed['augmentation_complete'][k]for k in ('inputs','targets')}}
 for d in DOMAINS[1:]:
  x=completed['public_CF_complete']['domains'][d];need(x['metadata']==B[d],'raw metadata receipt');old[d]={k:x[k]for k in ('inputs','targets')}
 for d in DOMAINS:need(completed['split_complete']['outputs'][d+'_SPLIT_ROWS.jsonl']==B[d+'_roles'],'split output receipt, not historical inputs.direct_rows')
 # Only explicitly selected metadata files are parsed. Binary fields are inert.
 rows={d:read(d)for d in DOMAINS};roles={d:read(d+'_roles')for d in DOMAINS};scenes=read('original_scenes');points=read('point_rows');weakrows=read('weak_rows')
 weak,dev=cc.weak_plan(weakrows);need(dev==set(entry['new_DEV_component_ids']),'exact fixed four DEV components')
 rolemap={}
 for d in DOMAINS:
  rolemap[d]={r['row_id']:r for r in roles[d]};need(len(rolemap[d])==len(roles[d])==len(rows[d]),'one role per full original occurrence')
  for i,r in enumerate(rows[d]):
   z=rolemap[d][r['row_id']];eq(r,z);need(z['role']in('TRAIN','HOLDOUT')and z['domain']==d and z['cache_index']==i,'ordered receipt plus authoritative row_id/META6')
 original={};scene8={}
 for s in scenes:
  mm=[];rr=s['original_input_metadata8'];need(len(rr)==8,'scene original8')
  for k,r in enumerate(rr):
   need(r['native_slot']==k and r['original_global_index']==s['scene_index']*8+k and r['split']=='TRAIN','original scene/native')
   q8=r['original_initial_points8'];need(q8==rr[0]['original_initial_points8']and aa.same(r['original_query_xy'],None if q8 is None else q8[k]),'scene q8 membership')
   need(r['original_global_index']not in original,'unique original source');original[r['original_global_index']]=r
   mm.append(dict(row_id=r['original_occurrence_id'],original_global_index=r['original_global_index'],native_slot=k,query_xy=r['original_query_xy'],L=r['original_L'],original_query_available=r['original_input_flags']['query_available']))
  scene8[s['scene_index']]=aa.ordered8(mm)
 need(len(original)==4096 and len(scene8)==512,'full original512 metadata container')
 public8=defaultdict(list);public={};public_component={};frame_to_component={}
 for r in rows[DOMAINS[1]]:
  o=r['original'];need(o['split']=='SOURCE_DEVELOPMENT'and aa.same(r['query_xy'],o['query_xy'])and r['L']==o['L']and r['original_query_available']==o['native_continuous_valid'],'public original scalars')
  key=(o['source_id'],r['native_slot']);need(key not in public,'unique public source native');public[key]=r
  z=rolemap[DOMAINS[1]][r['row_id']];public_component[key]=z['component_id'];public8[o['frame_index']].append(meta(r))
  if o['source_id']in frame_to_component:need(frame_to_component[o['source_id']]==z['component_id'],'sameframe component')
  frame_to_component[o['source_id']]=z['component_id']
 public8={f:aa.ordered8(rr)for f,rr in public8.items()};cf8=defaultdict(list)
 for r in rows[DOMAINS[2]]:
  o=r['original'];p=public[(o['source_id'],r['native_slot'])]
  need(aa.same(r['query_xy'],p['query_xy'])and r['L']==p['L']and r['original_query_available']==p['original_query_available'],'CF exact public qL/U')
  need(o['original_query_id']==p['row_id']and o['original_frame_index']==p['original']['frame_index'],'CF original parent identity')
  need(rolemap[DOMAINS[2]][r['row_id']]['component_id']==public_component[(o['source_id'],r['native_slot'])],'CF connected component equals parent public')
  cf8[r['sam_group_id']].append(meta(r))
 cf8={g:aa.ordered8(rr)for g,rr in cf8.items()};need(len(public8)==185 and len(cf8)==477,'full public/CF group8 counts')
 # Verify GT-generated weak family refers to the same original input lineage,
 # without making its q8 the original-initializer context of the old caches.
 weakframes={}
 for r in weakrows:
  g=r['generated_input'];o=g['original_input'];p=public[(o['frame_key'],g['native_slot'])];eq(o,p)
  cid=public_component[(o['frame_key'],g['native_slot'])];need(cid==g['component_id']==o['component_id']and rolemap[DOMAINS[1]][p['row_id']]['role']==o['original_role']=='TRAIN','weak original public lineage/outerTRAIN')
  need((r['source_role']=='SOURCE_DEV')==(cid in dev),'weak internal role exact fixed component')
  weakframes[o['frame_key']]=dict(source_id=o['frame_key'],component_id=cid,internal_source_role=r['source_role'],original_frame_index=o['original_frame_index'])
 assembled={};census=[];role_census=[];fitindices={};context_groups={};dev_desc=[]
 for d,rr in rows.items():
  out=[]
  for i,r in enumerate(rr):
   z=rolemap[d][r['row_id']];k=aa.native_index(r['native_slot']);cid=z['component_id'];need(isinstance(cid,str)and len(cid)==64,'component binding')
   if d==DOMAINS[0]:
    a=original[r['anchor_original_global_index']];im=original[r['source_image_original_global_index']]
    need(a['native_slot']==im['native_slot']==k and r['scene_index']==im['scene_index'],'actual image/anchor native')
    need(r['original_flags_ref']==[a['scene_index'],k]and r['image_flags_ref']==[im['scene_index'],k]and r['original_query_available']==a['original_input_flags']['query_available'],'exact anchor availability/flags')
    want=None if a['original_query_xy']is None else [a['original_query_xy'][0]+r['dx'],a['original_query_xy'][1]+r['dy']]
    need(aa.same(r['query_xy'],want)and r['L']==(None if a['original_L']is None else a['original_L']*r['L_scale']),'exact saved active transform')
    if r['family']=='SELF_SCENE15':need(a['original_global_index']==im['original_global_index'],'SELF own anchor')
    else:need(a['variant']=='BEFORE'and r['anchor_scene_index']==a['scene_index']and r['dx']==r['dy']==0 and r['L_scale']==1,'BEFORE fixed original anchor')
    origin=scene8[a['scene_index']];originid='SYNTH_ORIGINAL_SCENE_%04d'%a['scene_index'];sourceid=None
    extra={key:r[key]for key in ('family','state_name','state_index','source_image_original_global_index','anchor_original_global_index','scene_index','base_index','variant')}
   else:
    o=r['original'];f=o['frame_index']if d==DOMAINS[1]else o['original_frame_index'];origin=public8[f];originid='PUBLIC_FRAME_%04d'%f;sourceid=o['source_id']
    p=public[(sourceid,k)];need(origin[k]==meta(p),'same original source group8')
    extra=dict(source_image_original_global_index=p['original_global_index'],anchor_original_global_index=p['original_global_index'],original_frame_index=f,source_id=sourceid,source_image_index_semantics='ORIGINAL_PUBLIC_PARENT_QUERY_INDEX_NOT_CF_VARIANT_OCCURRENCE')
    if d==DOMAINS[2]:extra.update(pair_id=o['pair_id'],variant=o['variant'],source_occurrence_row_id=r['row_id'])
   need(origin[k]['original_query_available']==r['original_query_available'],'active original availability')
   q8=[x['query_xy']for x in origin];L8=[x['L']for x in origin];av8=[x['original_query_available']for x in origin]
   context=cc.context32(r['query_xy'],r['L'],k,r['original_query_available'],q8,L8,av8)
   ref=aa.build_context_vectors(r['query_xy'],r['L'],k,r['original_query_available'],q8,L8,av8)
   need(np.array_equal(context,np.concatenate([ref['native_onehot8'],ref['other_mask8'],ref['other_dxdy16']])),'current contract same exact V202 information')
   m=dict(meta(r),**extra,cache_index=i,component_id=cid,component_role=z['role'],nominal_fit_eligible=aa.fit_nominal_eligible(d,r),lineage_bound=True,context_origin='ORIGINAL_INITIALIZER',context_origin_group_id=originid,context_q8=q8,context_L8=L8,context_available8=av8,context_FP32_native_mask_xy_sha256=hashlib.sha256(context.tobytes()).hexdigest())
   out.append(m)
  assembled[d]=out;mask=cc.fit_mask(out,dev);available=np.asarray([r['original_query_available']for r in out],bool);train=np.asarray([r['component_role']=='TRAIN'for r in out]);nom=np.asarray([r['nominal_fit_eligible']for r in out]);dc=np.asarray([r['component_id']in dev for r in out])
  need(not mask[dc].any()and not mask[~train].any()and not mask[~nom].any(),'all shared old5 pool/norm masks exclude DEV/HOLD/jitter')
  fitindices[d]=dict(full_metadata_rows=len(out),fit_indices=np.flatnonzero(mask).tolist(),fit_available_indices=np.flatnonzero(mask&available).tolist(),source_HOLDOUT_indices=np.flatnonzero(~train).tolist(),new_DEV_component_indices=np.flatnonzero(dc).tolist(),TRAIN_non_nominal_indices=np.flatnonzero(train&~nom).tolist())
  counts=dict(domain=d,full_metadata_rows=len(out),original_inputU=int((~available).sum()),original_TRAIN_rows=int(train.sum()),original_HOLDOUT_rows=int((~train).sum()),nominal_eligible_all=int(nom.sum()),original_TRAIN_nominal_rows=int((train&nom).sum()),original_TRAIN_non_nominal_rows=int((train&~nom).sum()),new_DEV_component_rows=int(dc.sum()),new_DEV_original_TRAIN_rows=int((dc&train).sum()),new_DEV_original_HOLDOUT_rows=int((dc&~train).sum()),new_DEV_original_TRAIN_nominal_rows=int((dc&train&nom).sum()),fit_rows=int(mask.sum()),fit_inputU=int((mask&~available).sum()),fit_available_rows=int((mask&available).sum()),unknown_lineage_rows=0,full_original_group8_count=len(out)//8)
  census.append(counts)
  for role in ('TRAIN','HOLDOUT'):
   for n in (False,True):
    sel=np.asarray([r['component_role']==role and r['nominal_fit_eligible']==n for r in out]);role_census.append(dict(domain=d,component_role=role,nominal_fit_eligible=n,rows=int(sel.sum()),original_inputU=int((sel&~available).sum()),new_DEV_component_rows=int((sel&dc).sum()),fit_rows=int((sel&mask).sum())))
  for cid in sorted(dev):dev_desc.append(dict(domain=d,component_id=cid,physical_rows=sum(r['component_id']==cid for r in out),original_TRAIN_rows=sum(r['component_id']==cid and r['component_role']=='TRAIN'for r in out),fit_rows=sum(r['component_id']==cid and bool(mask[i])for i,r in enumerate(out))))
 # Strong7040 remains a metadata ledger, including its unavailable/S/outside rows.
 syn=assembled[DOMAINS[0]];byid={r['row_id']:i for i,r in enumerate(syn)};mask=cc.fit_mask(syn,dev);seen=set();known=Counter();reasons=Counter();state=Counter();unknown=0;pu=0
 pointfit=[]
 for r in points:
  rid=r['row_id'];need(rid in byid and rid not in seen,'unique point row in current source cache');seen.add(rid);i=byid[rid];m=syn[i];eq(r,m)
  need(r['component_id']==m['component_id']and r['component_role']==m['component_role'],'point original role/component')
  need(r['source_teacher_key']==[m['source_image_original_global_index']//8,m['native_slot']]and r['source_image_original_global_index']==m['source_image_original_global_index']and r['anchor_original_global_index']==m['anchor_original_global_index'],'actual image teacher key, separate q anchor')
  t,why=pp.point_loss_target(r);state[r['point0_combined_state']]+=1;reasons.update(why);pu+=not r['original_query_available'];unknown+=t<0
  if mask[i]and t>=0:known['D'if t==0 else'E']+=1;pointfit.append(i)
 need(len(points)==7040 and seen=={r['row_id']for i,r in enumerate(syn)if mask[i]},'point seen exact new masked SYNTH fit rows')
 need(census[0]['new_DEV_component_rows']==0 and census[0]['fit_rows']==7040,'no synthetic newDEV exclusion; all original TRAINnominal7040')
 need(dict(known)=={'D':5853,'E':243}and unknown==944 and pu==64,'curated strong eligibility unchanged')
 outputs={}
 for d,rr in assembled.items():outputs[d]=jsonl(H/(d+'_ASSEMBLED_ROWS.jsonl'),rr);old[d]['metadata']=outputs[d]
 indices=write(H/'SOURCE_FIT_AND_EXCLUSION_ROW_INDICES.json',dict(completed=True,PASS=True,physical_metadata_only=True,domains=fitindices,strong_known_fit_indices=pointfit,note='Indices reference original full cache order. Root must separately choose/authorize selective binary extraction; no binary was opened.'))
 fields=list(census[0]);p=H/'FULL_CONTAINER_AND_FIT_CENSUS.csv'
 with p.open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(census)
 censusb=binding(p);roleb=write(H/'PER_ROLE_NOMINAL_AND_DEV_CENSUS.json',role_census)
 inert=[old[d][k]for d in DOMAINS for k in ('inputs','targets')];need(len(inert)==6 and len({x['path']for x in inert})==6 and all(Path(x['path']).suffix=='.npz'for x in inert),'exact inert old6, no filesystem access')
 wp=[x for x in weak if x['target']>=0];expected={x['RGB96']['path']:x['RGB96']['sha256']for x in wp};declared={x['binding']['path']:x['binding']['sha256']for x in minimum['weak_RGB96_known57']};need(expected==declared and len(expected)==57,'inert weak57 exact metadata whitelist')
 pointsummary=dict(physical_retained_rows=7040,inputU_retained=pu,recorded_combined_states=dict(state),known_loss_D_E=[known['D'],known['E']],known_loss_rows=sum(known.values()),unknown_loss_rows=unknown,excluded_reasons_overlapping=dict(reasons),seen_equals_SYNTH_original_TRAIN_nominal_and_new_fitmask=True,new_DEV_SYNTH_exclusion_rows=0,actual_image_teacher_keys_exact=True,new_label_array_created=False)
 qa=dict(completed=True,PASS=True,kind='METADATA_ASSEMBLY_NO_BINARY_PAYLOAD_ACCESS',scope=B['scope'],code=binding(Path(__file__).absolute()),protocol=binding(H/'PROTOCOL_BEFORE_METADATA_ASSEMBLY.json'),checks=checkbinding,input_metadata_bindings=B,physical_metadata_accesses=ACCESS,domain_census=census,per_role_census=roleb,full_container_census_csv=censusb,new_DEV_component_ids=sorted(dev),new_DEV_public_frames=[x for x in weakframes.values()if x['internal_source_role']=='SOURCE_DEV'],all_new_weak_frame_lineage=list(weakframes.values()),new_DEV_descendants_by_domain_component=dev_desc,all_old_public_CF_descendants_of_new_DEV_excluded=True,q8_input_provenance_verified_no_target_geometry_features=True,all_metadata_rows_lineage_bound=True,metadata_META6_and_all8_group_identity_exact=True,old5_all_support_soft_public_CF_pools_and_synthetic_owner_normalization_use_same_C_fit_mask_by_frozen_code=True,actual_target_pool_counts_or_mass_normalization_not_read_or_computed=True,physical_old_metadata_includes_original_source_HOLDOUT=True,physical_metadata_read_is_not_fit_admission=True,binary_container_HOLDOUT_members_not_decoded=True,old6_NPZ_bindings_inert_only=inert,assembled_metadata=outputs,row_indices=indices,point7040=B['point_rows'],strong_point_summary=pointsummary,weak360=B['weak_rows'],weak_known57=dict(FIT_D39_E6=45,DEV_D9_E3=12,weak_q8_origin='GT_DERIVED_TRAINING_INITIALIZATION',not_original_initializer=True,authority=cc.WEAK_AUTHORITY),source_targets_and_inputs_authority_separate=True,old_METADATA_roles_unchanged=True,source_binary_stat_hash_header_read_decode_count=0,raw_GT_BOP_camera_image_mask_depth_reads=0,weights_model_forward_fit_Git_calls=0,actual_fit_or_probe_approved=False)
 qab=write(H/'LINEAGE_ASSEMBLY_QA.json',qa)
 entry['old_caches']=old;entry['point_metadata7040']=B['point_rows'];entry['expected_strong_D_E_counts']=[known['D'],known['E']];entry['lineage_assembly_QA']=qab
 entry['metadata_whitelist']=[qab,B['point_rows'],B['weak_rows'],*outputs.values()]
 entry['binary_whitelist']=inert+[x['binding']for x in minimum['weak_RGB96_known57']];need(entry['approved_source_pair_fit_or_probe']is False and entry['root_recipe']is None and entry['strong_target_admission']is None and entry['weak_target_admission']is None,'actual approval still root-only')
 entryb=write(H/'ASSEMBLED_SOURCE_PAIR_ENTRY_FALSE_TEMPLATE.json',entry)
 final=dict(completed=True,PASS=True,metadata_assembly_complete=True,frozen=True,no_further_own_writes=True,actual_fit_approved=False,code=binding(Path(__file__).absolute()),protocol=binding(H/'PROTOCOL_BEFORE_METADATA_ASSEMBLY.json'),QA=qab,assembled_metadata=outputs,old_caches=old,point_metadata7040=B['point_rows'],weak_rows360=B['weak_rows'],expected_strong_D_E_counts=[known['D'],known['E']],row_indices=indices,census=censusb,checks=checkbinding,future_entry_false=entryb,root_required_before_payload=['approved actual source scope/entry','strong/weak target admissions','throughput-only root recipe','root choice of selective extraction versus physically shared member decode'],binary_accesses=0)
 finalb=write(H/'FINAL_READY.json',final)
 print(json.dumps(dict(completed=True,PASS=True,final=finalb,QA=qab,domain_census=census,strong=pointsummary,entry_false=entryb),ensure_ascii=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--assemble-metadata-only',action='store_true',required=True);ap.parse_args();main()
