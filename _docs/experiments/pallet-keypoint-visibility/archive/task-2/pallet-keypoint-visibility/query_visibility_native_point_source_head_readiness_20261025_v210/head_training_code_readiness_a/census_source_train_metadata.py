"""Metadata-only TRAIN census. Original GT/config/arrays/assets are never opened."""
import collections as C
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
H=Path(__file__).absolute().parent
V202=P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202'
PROTO=V202/'fit_preparation_b/PROTOCOL_BEFORE_SPATIAL_Q8_SOURCE_ARRAY_OR_FIT_BODIES.json'
SOURCE_SCENES={'path':str(P/'query_visibility_full_source_query_causal_census_20261024_v173/census_preparation/ALL768_SCENE_PATHS_BEFORE_MASKS.jsonl'),'sha256':'e12ae8fd3c908ed1dee5c99fa6915601b1b6e176ce6cefabc4292abbef1c1307'}
ACCESS=[]

def raw(p):
 p=Path(p)
 assert p.is_absolute() and '..'not in p.parts and 'v112'not in str(p).lower()
 assert p.suffix in ('.json','.jsonl','.py') and p.name!='SCENES_GT_ONLY.json' and not p.name.startswith('BASE_CONFIGS'), 'No original GT/config/asset access'
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for i,s in enumerate(p.parts[1:]):
   n=os.open(s,os.O_RDONLY|os.O_NOFOLLOW|(os.O_DIRECTORY if i<len(p.parts[1:])-1 else 0),dir_fd=fd);os.close(fd);fd=n
  assert stat.S_ISREG(os.fstat(fd).st_mode)
  with os.fdopen(fd,'rb')as f:fd=-1;z=f.read()
 finally:
  if fd>=0:os.close(fd)
 ACCESS.append({'path':str(p),'sha256':hashlib.sha256(z).hexdigest()})
 return z

def read(b):
 z=raw(b['path']);assert hashlib.sha256(z).hexdigest()==b['sha256'];return json.loads(z)if Path(b['path']).suffix=='.json'else[json.loads(x)for x in z.splitlines()]

def output(name,v):
 z=(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
 with (H/name).open('xb')as f:f.write(z)
 return {'path':str(H/name),'sha256':hashlib.sha256(z).hexdigest()}

def main():
 q=json.loads(raw(PROTO));b=q['context_input_bindings'];assert ACCESS[-1]['sha256']=='8df60dc146b5f026563341a1d959ecff20ef501333a14694c3b0794a2c0e07d8'
 domains=['SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816','DIRECT_CONTROL']
 roles={d:read(b[d+'_roles'])for d in domains}
 train={d:{r['row_id']:r for r in roles[d]if r['role']=='TRAIN'}for d in domains}
 population=[]
 for d in domains:
  rr=list(train[d].values())
  population.append({'domain':d,'TRAIN_rows':len(rr),'input_U':sum(not r['original_query_available']for r in rr),'native_counts':dict(sorted(C.Counter(r['native_slot']for r in rr).items())),'physical_shared_role_records_read':len(roles[d])})
 allaug=read(b['SYNTH_AUG65536']);aug=[r for r in allaug if r['row_id']in train['SYNTH_AUG65536']]
 assert len(aug)==56320 and all(r['native_slot']==train['SYNTH_AUG65536'][r['row_id']]['native_slot']for r in aug)
 bases={r['base_index']for r in aug};assert len(bases)==110
 assert all(r['row_id']in train['SYNTH_AUG65536']for r in allaug if r['base_index']in bases), 'No partially split original base'
 scenes=[r for r in read(b['original_scenes'])if r['base_index']in bases]
 assert len(scenes)==440 and {r['variant']for r in scenes}=={'BEFORE','COVER','NEAR_SHAM','FAR_SHAM'}
 smap={r['scene_index']:r for r in scenes}
 for base in bases:
  sr=[r for r in scenes if r['base_index']==base]
  assert len(sr)==4 and len({r['variant']for r in sr})==4
  for s in sr:assert [r['native_slot']for r in s['original_input_metadata8']]==list(range(8))
 eligible=[r for r in aug if r['family']=='BEFORE_ANCHOR4'or(r['family']=='SELF_SCENE15'and r['state_name']=='NOMINAL'and r['state_index']==0)]
 assert len(eligible)==7040
 source_inventory=[r for r in read(SOURCE_SCENES)if r['base_index']in bases and r['split']=='TRAIN']
 assert len(source_inventory)==440
 inv={(r['base_index'],r['variant']):r for r in source_inventory}
 gt=[]
 for base in sorted(bases):
  selected=[r for r in scenes if r['base_index']==base]
  refs=[];receipts=[]
  for s in sorted(selected,key=lambda x:x['variant']):
   i=inv[(base,s['variant'])];assert i['scene_buffer']==s['source_RGB_binding'] and i['raw_hw']==s['raw_hw']
   receipts.append(i['base_checkpoint'])
   refs.append({'scene_index':s['scene_index'],'variant':s['variant'],'source_RGB_binding_NOT_OPENED':s['source_RGB_binding'],'original_query_rows':8})
  assert all(x==receipts[0]for x in receipts)
  completion=read(receipts[0]);assert completion['completed']and completion['PASS']and completion['base_index']==base and completion['scenes']==4
  z=completion['GT_scenes'];assert Path(z['path']).name=='SCENES_GT_ONLY.json' and 'v112'not in z['path'].lower()
  gt.append({'base_index':base,'source_component_ids':sorted({train['SYNTH_AUG65536'][r['row_id']]['component_id']for r in aug if r['base_index']==base}),'base_completion':receipts[0],'GT_json_binding_NOT_OPENED':z,'selected_original_scenes':refs})
 # Current expanded direct rows are joined by authoritative control key, never historical cache_index.
 direct_roles={tuple(r['control_join_key']):r for r in train['DIRECT_CONTROL'].values()}
 direct=[r for r in read(b['DIRECT_CONTROL'])if tuple(r['control_join_key'])in direct_roles]
 assert len(direct)==2784 and len({tuple(r['control_join_key'])for r in direct})==2784
 for r in direct:
  z=direct_roles[tuple(r['control_join_key'])]
  for k in ('row_id','native_slot','query_xy','L','original_query_available','source_packet','authority'):assert r[k]==z[k],k
 known=[r for r in direct if direct_roles[tuple(r['control_join_key'])]['stored_weak_direct_target']in(0,1)]
 assert len(known)==327 and all(r['root_admitted']and r['original_query_available']for r in known)
 bydomain=[];native_exposure=[]
 for domain in ['SYNTH_FINITE_END','REAL_SEMANTIC_END_WEAK']:
  rr=[r for r in direct if r['domain']==domain];kk=[r for r in known if r['domain']==domain]
  attempts={r['attempt_id']for r in kk}
  for aid in attempts:
   tr=[r for r in kk if r['attempt_id']==aid];assert len(tr)==3 and {r['variant']:direct_roles[tuple(r['control_join_key'])]['stored_weak_direct_target']for r in tr}=={'BEFORE':0,'COVER':1,'SHAM':0}
  bydomain.append({'domain':domain,'physical_TRAIN_rows':len(rr),'known_D':sum(direct_roles[tuple(r['control_join_key'])]['stored_weak_direct_target']==0 for r in kk),'known_E':sum(direct_roles[tuple(r['control_join_key'])]['stored_weak_direct_target']==1 for r in kk),'complete_DED_triplets':len(attempts),'distinct_source_bases':len({r['source_base_index']for r in kk if r['source_base_index']is not None}),'distinct_source_frames':len({r['source_frame_index']for r in kk if r['source_frame_index']is not None}),'distinct_source_components':len({direct_roles[tuple(r['control_join_key'])]['component_id']for r in kk})})
  for k in range(8):
   subset=[r for r in kk if r['native_slot']==k]
   native_exposure.append({'domain':domain,'native_slot':k,'D':sum(direct_roles[tuple(r['control_join_key'])]['stored_weak_direct_target']==0 for r in subset),'E':sum(direct_roles[tuple(r['control_join_key'])]['stored_weak_direct_target']==1 for r in subset),'triplets':len({r['attempt_id']for r in subset}),'bases':len({r['source_base_index']for r in subset if r['source_base_index']is not None}),'frames':len({r['source_frame_index']for r in subset if r['source_frame_index']is not None})})
 synth_known=[r for r in known if r['domain']=='SYNTH_FINITE_END']
 canonical_counts=C.Counter(smap[r['source_base_index']*4]['original_input_metadata8'][r['native_slot']]['declared_CF_to_canonical']for r in synth_known if r['variant']=='COVER')
 factor_bases=[]
 for geometry,aspect,material in sorted({(s['factors']['geometry_family'],s['factors']['aspect'],s['factors']['material'])for s in scenes}):
  ss=[s for s in scenes if (s['factors']['geometry_family'],s['factors']['aspect'],s['factors']['material'])==(geometry,aspect,material)]
  factor_bases.append({'geometry':geometry,'aspect':aspect,'material':material,'TRAIN_bases':len({s['base_index']for s in ss}),'original_scenes':len(ss),'original_native_rows':len(ss)*8})
 future=output('SOURCE110_TRAIN_GT_BINDING_PROPOSAL_NOT_OPENED.json',{'completed':True,'metadata_proposal_only':True,'GT_payload_stat_hash_header_reads':0,'TRAIN_bases':110,'original_scenes':440,'original_native_slots':3520,'no_actual_GT_point_state_census_yet':True,'parents':gt})
 result={'completed':True,'PASS':True,'source_TRAIN_only':True,'population':population,'SYNTH_TRAIN':{'rows':56320,'families':dict(C.Counter(r['family']for r in aug)),'original_parent_bases':110,'original_scenes':440,'original_native_rows':3520,'variants':dict(C.Counter(s['variant']for s in scenes)),'C61_fit_eligible_rows':7040,'C61_fit_eligible_families':dict(C.Counter(r['family']for r in eligible)),'C61_fit_eligible_input_U':sum(not r['original_query_available']for r in eligible),'all110_bases_all4_scenes_same_TRAIN_role':True,'all_scene_RGB_origin_V129':all('query_visibility_registered_native_diversity_20261023_v129' in s['source_RGB_binding']['path']for s in scenes),'TARGET_SHIFT_in_current4views':False,'factor_cells':factor_bases},'existing_DIRECT_TRAIN':{'physical_rows':2784,'known':327,'unknown':2784-327,'by_domain':bydomain,'per_native':native_exposure,'synthetic_COVER_canonical_mapping_counts':dict(sorted(canonical_counts.items())),'authority':'Stored finite-end / real RGB semantic-weak targets; not newly admitted native point states.'},'future_GT_bindings':future,'metadata_reads':ACCESS,'original_GT_config_arrays_assets_headers_stat_hash_decode':0,'new_labels_training_or_forward':0,'source_HOLDOUT_or_protected_score_analysis':0}
 output('SOURCE_TRAIN_COVERAGE_CENSUS.json',result)
 buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=['domain','native_slot','D','E','triplets','bases','frames']);w.writeheader();w.writerows(native_exposure)
 with (H/'DIRECT_TRAIN_NATIVE_EXPOSURE.csv').open('x',encoding='utf8',newline='')as f:f.write(buf.getvalue())
 print(json.dumps({'PASS':True,'SYNTH_TRAIN_bases':110,'GT_bindings_proposed_not_opened':110,'C61_fit_eligible':7040,'DIRECT_domains':bydomain,'DIRECT_native':native_exposure,'canonical_COVER_counts':dict(sorted(canonical_counts.items()))},ensure_ascii=False))

if __name__=='__main__':main()
