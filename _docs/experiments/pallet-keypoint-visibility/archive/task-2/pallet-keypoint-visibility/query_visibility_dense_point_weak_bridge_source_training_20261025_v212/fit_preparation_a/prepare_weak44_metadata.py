"""Exact frozen SOURCE metadata joins only. No PNG/NPZ/header/stat/hash of assets."""
import collections
import hashlib
import json
import os
from pathlib import Path

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');H=Path(__file__).absolute().parent
V207=P/'query_visibility_source_train_witness_conditioned_controls_20261025_v207'
PROTO=P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202/fit_preparation_b/PROTOCOL_BEFORE_SPATIAL_Q8_SOURCE_ARRAY_OR_FIT_BODIES.json'

def rb(p):
 p=Path(p);assert p.is_absolute()and'..'not in p.parts and'v112'not in str(p).lower()and p.suffix in('.json','.jsonl','.py')
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for i,s in enumerate(p.parts[1:]):
   n=os.open(s,os.O_RDONLY|os.O_NOFOLLOW|(os.O_DIRECTORY if i<len(p.parts[1:])-1 else 0),dir_fd=fd);os.close(fd);fd=n
  with os.fdopen(fd,'rb')as f:fd=-1;return f.read()
 finally:
  if fd>=0:os.close(fd)

def bind(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def read(b):
 z=rb(b['path']);assert hashlib.sha256(z).hexdigest()==b['sha256'];return json.loads(z)if Path(b['path']).suffix=='.json'else[json.loads(x)for x in z.splitlines()]
def new(n,v):
 z=(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
 with(H/n).open('xb')as f:f.write(z)
 return dict(path=str(H/n),sha256=hashlib.sha256(z).hexdigest())

def main():
 p=json.loads(rb(PROTO));b=p['context_input_bindings']
 public=read(b['PUBLIC_TRAIN1480']);roles=read(b['PUBLIC_TRAIN1480_roles']);ri={r['row_id']:r for r in roles};byid={r['row_id']:r for r in public}
 complete_path=V207/'raw96_weak_witness_control_pilot_v1/COMPLETE.json';complete=json.loads(rb(complete_path));assert complete['completed']and complete['PASS']and complete['fit_admission']is False
 selection=read(complete['selection']);preview=read(complete['outputs']['FIXED8_PREVIEW_MANIFEST.json']);attempts=read(complete['outputs']['ALL80_ATTEMPTS.jsonl'])
 cases=selection['cases'];cards={r['position']:r for r in preview['cards']};success=[r for r in attempts if r['status']=='CONSTRUCTED_RAW96_CONTROL_NO_LABEL_ADMISSION'];assert len(cases)==8 and len(success)==18 and len(attempts)==80
 frames={};rows=[]
 for case in cases:
  m=byid[case['row_id']];role=ri[m['row_id']];assert role['role']=='TRAIN'and m['native_slot']==case['native_slot']and m['query_xy']==case['original_query_xy']and m['L']==case['original_L']and m['original_query_available']==case['original_query_available']is True
  fi=m['original']['frame_index'];eight=sorted([r for r in public if r['original']['frame_index']==fi],key=lambda r:r['native_slot']);assert[r['native_slot']for r in eight]==list(range(8))and all(ri[r['row_id']]['role']=='TRAIN'for r in eight)
  frame={'frame_index':fi,'source_id':m['original']['source_id'],'component_id':role['component_id'],'metadata8':[{k:r[k]for k in('row_id','original_global_index','native_slot','query_xy','L','original_query_available')}for r in eight]}
  if str(fi)in frames:assert frames[str(fi)]==frame
  frames[str(fi)]=frame;card=cards[case['position']];assert card['row_id']==case['row_id']and card['BEFORE']['sha256']==case['CLEAN96']['sha256']and card['VALID']['sha256']==case['VALID96']['sha256']
  common=dict(source_row_id=m['row_id'],source_original_global_index=m['original_global_index'],original_global_index=m['original_global_index'],native_slot=m['native_slot'],query_xy=m['query_xy'],L=m['L'],original_query_available=m['original_query_available'],source_frame_index=fi,frame_q8_key=str(fi),component_id=role['component_id'],component_role='TRAIN',original_flags={k:m['original'][k]for k in('native_continuous_valid','feature_constructed','nearest_valid','Q_inside')if k in m['original']},case_position=case['position'],valid_binding=card['VALID'],valid96_sha256_expected=m['valid96_sha256'],authority='ASSISTANT_SOURCE_LOCAL_VISIBLE_FEATURE_WEAK_NOT_NATIVE_GOLD',new_training_admission=False)
  rows.append(dict(common,row_id='V212_WEAK_CASE%02d_FACTUAL'%case['position'],variant='FACTUAL',attempt_id=None,RGB_binding=card['BEFORE'],rgb96_sha256_expected=m['rgb96_sha256'],proposed_weak_target=0))
  for a in sorted([x for x in success if x['case_position']==case['position']],key=lambda x:x['attempt_id']):
   assert a['original_meta']['row_id']==m['row_id']and a['original_meta']['original_query_xy']==m['query_xy']and a['original_meta']['original_L']==m['L']and a['proof']['actual_change_PASS']and a['proof']['SHAM_feature_RGB_unchanged']
   for variant,label in [('SHAM',0),('COVER',1)]:
    z=a['outputs'][variant+'_CLEAN96'];rows.append(dict(common,row_id='V212_WEAK_'+a['attempt_id']+'_'+variant,variant=variant,attempt_id=a['attempt_id'],RGB_binding=z,rgb96_sha256_expected=z['decoded_sha256'],proposed_weak_target=label,source_attempt_proof_binding=complete['outputs']['ALL80_ATTEMPTS.jsonl']))
 assert len(rows)==44 and len(frames)==3 and len({r['row_id']for r in rows})==44
 assert collections.Counter(r['variant']for r in rows)=={'FACTUAL':8,'SHAM':18,'COVER':18}
 assert len({(r['rgb96_sha256_expected'],r['valid96_sha256_expected'])for r in rows})==44,'Exact duplicate weak input; no deletion/fallback permitted'
 payloads={}
 for r in rows:
  for kind,key in [('WEAK_RGB','RGB_binding'),('WEAK_VALID','valid_binding')]:
   z=r[key];assert'v112'not in z['path'].lower()and Path(z['path']).is_relative_to(V207)
   value=dict(path=z['path'],sha256=z['sha256'],kind=kind)
   if z['path']in payloads:assert payloads[z['path']]==value
   payloads[z['path']]=value
 assert len(payloads)==52
 result=dict(completed=True,PASS=True,metadata_preparation_only=True,source_complete=bind(complete_path),selection=complete['selection'],attempts=complete['outputs']['ALL80_ATTEMPTS.jsonl'],preview=complete['outputs']['FIXED8_PREVIEW_MANIFEST.json'],source_PUBLIC_metadata=b['PUBLIC_TRAIN1480'],source_PUBLIC_roles=b['PUBLIC_TRAIN1480_roles'],frames=frames,rows=rows,payload_whitelist=list(payloads.values()),counts=dict(rows=44,factual_D=8,SHAM_D=18,COVER_weak_E=18,distinct_frames=3,unique_effective_RGB_valid_hashes=44,unique_PNG_payloads=52),original_qL_native_roles_and_original_q8_unchanged=True,failed_cases_keep_factual=True,no_duplicate_factual_by_donor=True,actual_source_asset_stat_hash_header_or_decode=0,training_admission=False)
 done=new('WEAK44_INPUT_METADATA_AND_FUTURE_PNG_WHITELIST.json',result);print(json.dumps(dict(PASS=True,weak_metadata=done,counts=result['counts'])))

if __name__=='__main__':main()
