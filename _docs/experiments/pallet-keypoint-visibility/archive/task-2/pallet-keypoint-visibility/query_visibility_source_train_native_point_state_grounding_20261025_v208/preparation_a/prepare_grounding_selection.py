"""V208 metadata-only pre-GT selection. Never opens source GT/config/binary paths."""
from pathlib import Path
import argparse,csv,hashlib,json,os,stat
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
D=P/'query_visibility_source_group_split_direct_end_training_20261024_v194'
META=('original_global_index','native_slot','query_xy','L','original_query_available')
SCOPE=V/'ROOT_PROSPECTIVE_SYNTH_TRAIN_POINT_STATE_GROUNDING_SCOPE.json'
INPUTS={
 'cache_complete':{'path':str(D/'direct_control_raw_sam_cache_v1/COMPLETE.json'),'sha256':'4bad28053a1ba96d20e64ab4b1039edfbbfb90afd3aac586b8d06d6548bbe197'},
 'cache_rows':{'path':str(D/'direct_control_raw_sam_cache_v1/ALL_CONSTRUCTED_ROWS.jsonl'),'sha256':'5746f0962f3a5cd55b7212ec1114612d05467364d9fda313e0a104305328424d'},
 'split_complete':{'path':str(D/'source_component_split_v1/COMPLETE.json'),'sha256':'f2dd0929ad73e843f2d0044c296c4c54f618ead69899400e75b2b6bac486d445'},
 'split_rows':{'path':str(D/'source_component_split_v1/DIRECT_CONTROL_SPLIT_ROWS.jsonl'),'sha256':'9ba6db7cfddad297d1fa060c55483d46a6e61eef1acb26cd0dc98fd32e51e046'},
 'synthetic_census':{'path':str(P/'query_visibility_sampled_domain_direct_end_20261024_v189/synthetic_sampled_end_controls_v1/ALL1024_END_CONTROL_ATTEMPTS.jsonl'),'sha256':'7adcd517279685136377bba9e7d22cda8fab6b3e50ab067013f725d50def359b'},
 'source_scene_inventory':{'path':str(P/'query_visibility_full_source_query_causal_census_20261024_v173/census_preparation/ALL768_SCENE_PATHS_BEFORE_MASKS.jsonl'),'sha256':'e12ae8fd3c908ed1dee5c99fa6915601b1b6e176ce6cefabc4292abbef1c1307'}}
SOURCE_INPUT=P/'query_visibility_frozen_sam_source_baseline_20261024_v180/ALL4096_TRAIN_INPUTS_METADATA_ONLY.jsonl'
def need(x,m):
 if not x:raise AssertionError(m)
def lexical(p):
 p=Path(p);need(p.is_absolute()and'..'not in p.parts and'v112'not in str(p).casefold(),'Lexical name before filesystem');return p
def rb(p):
 p=lexical(p);need(p.suffix in('.json','.jsonl','.py'),'Metadata/code only')
 need(p.name not in('SCENES_GT_ONLY.json','BASE_CONFIGS192_BEFORE_RGB.json'),'GT/config requires later root entry')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for c in p.parts[1:-1]:n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Ordinary metadata')
 with os.fdopen(n,'rb')as f:return f.read()
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(rb(p)).hexdigest()}
def read(b):
 z=rb(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Exact metadata binding');return json.loads(z)
def lines(b):
 z=rb(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Exact metadata binding');return [json.loads(s)for s in z.splitlines()if s.strip()]
def write(name,x):
 p=lexical(H/name)
 with p.open('x',encoding='utf-8')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def select_train(cache,roles):
 d={tuple(r['control_join_key']):r for r in roles};need(len(d)==len(roles),'Unique sidecar key');need({tuple(r['control_join_key'])for r in cache}==set(d),'Current owncache joins sidecar by control key, never cache_index')
 selected=[]
 for r in cache:
  s=d[tuple(r['control_join_key'])]
  if s['role']!='TRAIN' or r['domain']!='SYNTH_FINITE_END':continue
  need(all(r[k]==s[k]for k in META)and r['row_id']==s['row_id'],'Selected original META6 exact')
  selected.append((r,s))
 return selected

def run():
 cache=read(INPUTS['cache_complete']);split=read(INPUTS['split_complete'])
 need(cache['metadata']==INPUTS['cache_rows']and split['outputs']['DIRECT_CONTROL_SPLIT_ROWS.jsonl']==INPUTS['split_rows'],'Current V194 authorities; ignore split.inputs.direct_rows')
 allcache=lines(INPUTS['cache_rows']);allroles=lines(INPUTS['split_rows']);selected=select_train(allcache,allroles)
 ids={r['attempt_id']for r,s in selected};bases={r['source_base_index']for r,s in selected}
 census={r['attempt_id']:r for r in lines(INPUTS['synthetic_census'])if r['attempt_id']in ids}
 need(set(census)==ids,'Every selected constructed attempt has an original source binding')
 source_input=bind(SOURCE_INPUT)
 originals=[r for r in lines(source_input)if r['base_index']in bases and r['variant']=='BEFORE']
 sceneids={r['scene_index']for r in originals};scene={r['base_index']:r for r in lines(INPUTS['source_scene_inventory'])if r['scene_index']in sceneids and r['variant']=='BEFORE'}
 need(set(scene)==bases and len(originals)==8*len(bases),'All8 original BEFORE inputs for selected TRAIN parents')
 original_by={(r['base_index'],r['native_slot']):r for r in originals};parents=[];queries=[];selected_rows=[];attempts=[]
 for base in sorted(bases):
  aa=[r for r in census.values()if r['base_index']==base];need(aa,'Parent has constructed attempt')
  buffers={json.dumps(r['source_BEFORE_buffer'],sort_keys=True)for r in aa};need(len(buffers)==1,'Same original BEFORE payload per base');buffer=aa[0]['source_BEFORE_buffer']
  need(buffer==scene[base]['BEFORE_buffer'],'Exact parent source identity before GT')
  checkpoint=scene[base]['base_checkpoint'];need(Path(checkpoint['path']).name=='BASE_COMPLETE.json','Only bound completion metadata, no inferred GT path');record=read(checkpoint)
  need(record['base_index']==base and record['completed']is True and record['PASS']is True,'Bound selected parent receipt')
  gt=record['GT_scenes'];need(Path(gt['path']).name=='SCENES_GT_ONLY.json'and Path(gt['path']).parent==Path(buffer['path']).parent,'Literal distinct GT JSON from original receipt')
  component_ids=sorted({s['component_id']for r,s in selected if r['source_base_index']==base})
  p={'base_index':base,'original_BEFORE_payload_binding_only':buffer,'GT_json_binding_NOT_OPENED':gt,'source_base_completion':checkpoint,'original_scene_index':scene[base]['scene_index'],'role':'TRAIN','component_ids':component_ids,'constructed_attempt_ids':sorted(r['attempt_id']for r in aa),'original_native_rows':8};parents.append(p)
  for k in range(8):
   o=original_by[base,k];need(o['split']=='TRAIN','Original source partition TRAIN')
   # The immutable component role was filtered before touching this source parent.
   mm=[r for r,s in selected if r['source_base_index']==base and r['native_slot']==k]
   need(mm and all(r['query_xy']==o['original_query_xy']and r['L']==o['original_L']and r['original_query_available']==o['original_input_flags']['query_available']and r['original_input_flags']==o['original_input_flags']for r in mm),'All native original q/L/U/flags preserved across attempts/variants')
   queries.append({'base_index':base,'role':'TRAIN','component_ids':component_ids,'native_slot':k,'original_global_index':o['original_global_index'],'original_occurrence_id':o['original_occurrence_id'],'query_xy':o['original_query_xy'],'L':o['original_L'],'original_query_available':o['original_input_flags']['query_available'],'original_input_flags':o['original_input_flags'],'declared_CF_to_canonical':o['declared_CF_to_canonical'],'raw_hw':o['raw_hw'],'original_BEFORE_raw96_binding_only':{'saved_crop_chunk':o['saved_crop_chunk'],'saved_crop_row':o['saved_crop_row'],'saved_crop_keys':o['saved_crop_keys'],'RGB96_sha256':o['saved_rgb96_sha256_inherited'],'valid96_sha256':o['saved_valid96_sha256_inherited']},'source_BEFORE_buffer_binding_only':buffer,'GT_json_binding_NOT_OPENED':gt})
 for aid in sorted(ids):
  rr=[r for r,s in selected if r['attempt_id']==aid];cr=census[aid]
  variants=sorted({r['variant']for r in rr});need(all(sorted(r['native_slot']for r in rr if r['variant']==v)==list(range(8))for v in variants),'All8 for every physically present variant; missing variants not fabricated')
  attempts.append({'attempt_id':aid,'base_index':cr['base_index'],'original_requested_native_slot':cr['native_slot'],'role':'TRAIN','present_variants':variants,'original_BEFORE_payload_binding_only':cr['source_BEFORE_buffer'],'physical_native_rows':len(rr)})
 for r,s in selected:
  selected_rows.append({k:r[k]for k in ('row_id','control_join_key','attempt_id','variant','native_slot','source_base_index','original_global_index','original_occurrence_id','query_xy','L','original_query_available','original_input_flags','source_packet') }|{'role':'TRAIN','component_id':s['component_id']})
 need(len({(b['original_BEFORE_payload_binding_only']['path'],b['original_BEFORE_payload_binding_only']['sha256'])for b in parents})==len(parents),'Exact original BEFORE payload dedupe')
 selection={'completed':True,'PASS':True,'scope':bind(SCOPE),'input_metadata_bindings':INPUTS|{'original_input_metadata':source_input},'selection_rule':'Current V194 constructed rows JOIN fixed sidecar via control_join_key, then domain SYNTH_FINITE_END AND role TRAIN; no root_admitted/status/weaktarget filter. Deduplicate original BEFORE parent payloads.','physically_unpacked_shared_metadata':{'current_cache_rows':len(allcache),'current_split_rows':len(allroles),'source_census_rows':1024,'source_input_rows':4096,'source_scene_inventory_rows':768},'counts':{'TRAIN_constructed_physical_rows':len(selected_rows),'TRAIN_constructed_attempts':len(attempts),'dedup_original_BEFORE_parents':len(parents),'original_parent_native_rows':len(queries),'original_parent_inputU':sum(not r['original_query_available']for r in queries)},'parents':parents,'attempts':attempts,'original_BEFORE_native8':queries,'physical_control_rows':selected_rows,'GT_json_whitelist_NOT_OPENED':[x['GT_json_binding_NOT_OPENED']for x in parents],'config_json_required':False,'actual_GT_config_binary_header_stat_hash_decode':0,'no_SOURCE_HOLDOUT_rows_analyzed_or_exported':True,'no_new_labels_or_admission':True}
 write('SELECTED_SOURCE_TRAIN_BEFORE_GT.json',selection)
 print(json.dumps({'completed':True,'PASS':True,'counts':selection['counts'],'GTfiles_not_opened':len(parents),'selection':bind(H/'SELECTED_SOURCE_TRAIN_BEFORE_GT.json')},ensure_ascii=False))
if __name__=='__main__':run()
