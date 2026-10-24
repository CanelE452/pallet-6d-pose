"""Metadata-only V163 recovery; preserve original failure and every existing image."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, math, traceback
import numpy as np
from PIL import Image

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_train_native_part_witness_20261024_v163'
O=Path(__file__).resolve().parent
F=V/'evidence_v1'
R=V/'evidence_metadata_recovery_v2'
PREP=V/'evidence_preparation'
ASSETS={'card':('CARD.png',(1920,1400),'RGB'),'raw96':('RAW_RGB96.png',(96,96),'RGB'),'central32':('RAW_CENTRAL32.png',(32,32),'RGB'),'center2x2':('RAW_CENTER2X2.png',(2,2),'RGB'),'valid96':('RAW_VALID96.png',(96,96),'L'),'display96':('DISPLAY96_CENTER_OVERLAY.png',(288,288),'RGB'),'nominal_ref_diagnostic':('NOMINAL_REFERENCE_DIAGNOSTIC.png',(224,224),'RGB')}

def guard(p):
 p=Path(p).resolve();assert p.is_relative_to(V)
 assert not any(s in str(p).lower() for s in ('v112','current223','/private','/held'))
 return p
def sha(p):return hashlib.sha256(guard(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(guard(p)),'sha256':sha(p)}
def read(p):return json.loads(guard(p).read_text())
def verify(b):
 p=guard(b['path']);assert sha(p)==b['sha256'],str(p);return p
def native(x):
 if isinstance(x,np.generic):return native(x.item())
 if isinstance(x,np.ndarray):return native(x.tolist())
 if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [native(v) for v in x]
 if isinstance(x,float):assert math.isfinite(x)
 assert x is None or isinstance(x,(str,bool,int,float)),type(x)
 return x
def write(p,x):
 p=guard(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
 s=json.dumps(native(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
 with p.open('x') as f:f.write(s)
def writejl(p,rows):
 lines=[json.dumps(native(r),ensure_ascii=False,allow_nan=False) for r in rows]
 with guard(p).open('x') as f:f.write('\n'.join(lines)+'\n')
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def ref_box(ref,side=224):
 cx=int(math.floor(ref[0]+.5));cy=int(math.floor(ref[1]+.5))
 return [cx-side//2,cy-side//2,cx+side//2,cy+side//2]
def diagnostic_valid_count(box,hw):
 left,top,right,bottom=box;h,w=hw
 return int(max(0,min(w,right)-max(0,left))*max(0,min(h,bottom)-max(0,top)))

def inventory(data):
 rows=[];expected_dirs={c['case_id'] for c in data['cases']}
 assert {p.name for p in (F/'cases').iterdir()}==expected_dirs
 for c in data['cases']:
  d=F/'cases'/c['case_id'];assert {p.name for p in d.iterdir()}=={v[0] for v in ASSETS.values()}
  images={}
  for key,(name,wh,mode) in ASSETS.items():
   p=d/name
   with Image.open(p) as im:assert im.size==wh and im.mode==mode
   images[key]=dict(bind(p),dimensions=list(wh),mode=mode,bytes=p.stat().st_size)
  rows.append(dict(case_id=c['case_id'],images=images))
 contacts=[]
 for p in sorted((F/'contacts').iterdir()):
  assert p.name.startswith('GROUP_A_PAGE_') and p.suffix=='.png'
  with Image.open(p) as im:assert im.size==(1920,2800) and im.mode=='RGB'
  contacts.append(dict(bind(p),dimensions=[1920,2800],bytes=p.stat().st_size))
 assert len(rows)==390 and len(contacts)==65
 return dict(completed=True,PASS=True,cases=390,case_image_assets=2730,existing_A_contacts=65,missing_B_C_and_cross_contacts_preserved=True,images_never_presented_or_semantically_reviewed=True,inventory=rows,contacts=contacts)

def prepare():
 assert not (O/'PREPARATION_COMPLETE.json').exists()
 data=read(PREP/'BLINDED_INPUT390_AND_FRAME88_BINDINGS.json')
 original=read(PREP/'PREPARATION_COMPLETE.json');assert original['code']==bind(PREP/'render_native_part_evidence.py')
 failure=read(F/'FAILED.json');assert failure['error']=='Object of type int64 is not JSON serializable'
 assert not (F/'COMPLETE.json').exists() and not (F/'ALL390_BLINDED_EVIDENCE.jsonl').exists()
 # Preserve source code text and exact failure, while original evidence_v1 remains untouched.
 preserved=O/'provenance';preserved.mkdir(parents=True,exist_ok=True)
 for src in [PREP/'render_native_part_evidence.py',PREP/'PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json',F/'FAILED.json',F/'STARTED.json',F/'REVIEW_INPUT_GROUP_A.json']:
  target=preserved/src.name
  with target.open('xb') as out:out.write(src.read_bytes())
 write(O/'EXISTING390_ASSET_COMPLETENESS_AND_HASHES.json',inventory(data))
 cx,cy=np.floor(np.asarray([.2,-.2])+.5).astype(int);bad=[cx-112,cy-112,cx+112,cy+112]
 rejected=False
 try:json.dumps({'nominal_reference_diagnostic_raw_xyxy':bad})
 except TypeError:rejected=True
 sample={'box':bad,'nested':[np.int64(3),np.bool_(False),{'a':np.float64(.125)}],'array':np.array([[True,False]])}
 fixed=native(sample);encoded=json.dumps(fixed,allow_nan=False)
 checks={'reproduces_original_numpy_int64_JSON_failure':rejected,'recursive_native_scalars_roundtrip':json.loads(encoded)==fixed,'geometry_values_unchanged':fixed['box']==[-112,-112,112,112],'box_new_native_python_int':all(type(x) is int for x in ref_box([.2,-.2])),'negative_and_edge_padding_arithmetic':diagnostic_valid_count(ref_box([.2,-.2]),[720,1280])==112*112,'nonfinite_rejected':False,'original_code_and_failure_preserved':sha(preserved/'render_native_part_evidence.py')==original['code']['sha256'] and sha(preserved/'FAILED.json')==sha(F/'FAILED.json')}
 try:native({'bad':float('nan')})
 except AssertionError:checks['nonfinite_rejected']=True
 assert all(checks.values())
 write(O/'GENERATED_SERIALIZATION_REGRESSION_CHECKS.json',dict(completed=True,PASS=True,checks=checks,no_source_pixels_or_generated_review_images_viewed=True))
 protocol=dict(schema='V163_MINIMAL_EXISTING_IMAGE_METADATA_RECOVERY_V2',completed=True,code=bind(__file__),original_preparation=bind(PREP/'PREPARATION_COMPLETE.json'),original_code=original['code'],original_protocol=original['protocol'],failed_attempt=bind(F/'FAILED.json'),original_started=bind(F/'STARTED.json'),partial_invalid_GROUP_A_preserved=bind(F/'REVIEW_INPUT_GROUP_A.json'),inputs=original['inputs'],assignment=original['assignment'],asset_inventory=bind(O/'EXISTING390_ASSET_COMPLETENESS_AND_HASHES.json'),generated_checks=bind(O/'GENERATED_SERIALIZATION_REGRESSION_CHECKS.json'),diagnosis='ref_crop uses NumPy floor/astype(int) then returns list of int64 scalars; all390 images were saved and asserted before first reviewer manifest JSON write. No native/scientific observation produced.',permitted='Read existing generated PNG bytes/headers; decode saved raw96/central32/center2x2/valid96/nominal-reference PNGs only to reconstruct metadata and verify exact input-array hashes. Do not present/view semantic pixels. Write new versioned manifests only.',forbidden='No source RGB or NPZ reread, card rerender, source image/label/model/feature/weight reads, fit/encoder/PnP/qL changes, annotation or votes.',image_policy='Original2730 case assets and65 A contacts remain byte-identical, in place. No new images or B/C/cross contacts. All reviewers use complete individual-card manifests; missing contacts disclosed.',output_directory=str(R),criteria_unchanged=True,root_execution_entry_required=True,failure_policy='Single finalization attempt; preserve STARTED/FAILED and all partial new manifests; no automatic retry.')
 write(O/'PROTOCOL_BEFORE_METADATA_RECOVERY_EXECUTION.json',protocol)
 write(O/'ACTUAL_METADATA_RECOVERY_ENTRY_TEMPLATE.json',dict(approved_actual_metadata_recovery=False,source_only=True,code=bind(__file__),protocol=bind(O/'PROTOCOL_BEFORE_METADATA_RECOVERY_EXECUTION.json'),no_source_pixels_reread_no_rerender_no_models=True))
 write(O/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(O/'PROTOCOL_BEFORE_METADATA_RECOVERY_EXECUTION.json'),checks=bind(O/'GENERATED_SERIALIZATION_REGRESSION_CHECKS.json'),asset_inventory=bind(O/'EXISTING390_ASSET_COMPLETENESS_AND_HASHES.json'),entry_template=bind(O/'ACTUAL_METADATA_RECOVERY_ENTRY_TEMPLATE.json'),actual_recovery_executed=False,source_pixel_NPZ_or_semantic_evidence_views=0))
 print(json.dumps(bind(O/'PREPARATION_COMPLETE.json')),flush=True)

def execute(entrypath):
 entry=read(entrypath);assert entry['approved_actual_metadata_recovery'] is True and entry['source_only'] is True and entry['no_source_pixels_reread_no_rerender_no_models'] is True
 protocol=read(verify(entry['protocol']));assert entry['code']==protocol['code']==bind(__file__)
 assert not (R/'STARTED.json').exists()
 write(R/'STARTED.json',dict(completed=False,entry=bind(entrypath),protocol=entry['protocol'],UTC=datetime.now(timezone.utc).isoformat(),no_rerender=True))
 try:
  for key in ['original_code','original_protocol','failed_attempt','original_started','partial_invalid_GROUP_A_preserved']:verify(protocol[key])
  data=read(verify(protocol['inputs']));assignment=read(verify(protocol['assignment']));inv=read(verify(protocol['asset_inventory']));lookupinv={r['case_id']:r for r in inv['inventory']}
  records=[]
  for c in data['cases']:
   images={key:{'path':b['path'],'sha256':b['sha256']} for key,b in lookupinv[c['case_id']]['images'].items()}
   for b in images.values():verify(b)
   # Arrays are decoded only from already generated raw PNGs; no source assets or card pixels.
   a={key:np.asarray(Image.open(images[key]['path'])) for key in ['raw96','central32','center2x2','valid96','nominal_ref_diagnostic']}
   crop=a['raw96'];valid=a['valid96']>0
   assert set(np.unique(a['valid96'])).issubset({0,255})
   assert ah(crop)==c['RGB96_HWC_sha256'] and ah(crop.transpose(2,0,1))==c['RGB96_CHW_sha256'] and ah(valid)==c['valid96_sha256']
   assert np.array_equal(a['central32'],crop[32:64,32:64]) and np.array_equal(a['center2x2'],crop[47:49,47:49])
   box=ref_box(c['reference_xy']);v4=valid[47:49,47:49]
   records.append(dict(c,images=images,center4_valid_bits=v4.tolist(),center4_valid_count=int(v4.sum()),center4_allvalid=bool(v4.all()),valid96_count=int(valid.sum()),crop_edge_or_padding=bool(not valid.all()),central32_array_sha256=ah(a['central32']),center2x2_array_sha256=ah(a['center2x2']),nominal_reference_diagnostic_raw_xyxy=box,nominal_reference_diagnostic_valid_count=diagnostic_valid_count(box,c['raw_hw']),nominal_reference_diagnostic_array_sha256=ah(a['nominal_ref_diagnostic'])))
  assert len(records)==390;lookup={r['case_id']:r for r in records};contacts=[]
  for j,b in enumerate(inv['contacts']):
   verify(b);contacts.append(dict(group='A',page=j+1,case_ids=assignment['primary']['A'][2*j:2*j+2],image={'path':b['path'],'sha256':b['sha256']}))
  for g in 'ABC':
   ids=assignment['primary'][g];assert len(ids)==130
   write(R/f'REVIEW_INPUT_GROUP_{g}.json',dict(schema='V163_BLINDED_NATIVE_PART_INPUT',completed=True,group=g,count=130,cases=[lookup[cid] for cid in ids],contacts=contacts if g=='A' else [],cross40_case_ids=assignment['cross40'],historically_contextualized_assistant_weak=True,old_labels_or_outputs_exposed=False,metadata_recovered_from_existing_images=True,missing_contacts_disclosed=g!='A',individual_cards_complete=True))
  crossids=assignment['cross40'];assert len(crossids)==len(set(crossids))==40
  write(R/'BLIND_CROSS40_ALL_REVIEWERS.json',dict(completed=True,count=40,independent_reviewers=['A','B','C','ROOT_BEFORE_AGENT_VOTES'],cases=[lookup[cid] for cid in crossids],contacts=[],other_reviewer_votes_exposed=False,individual_cards_complete=True,missing_contacts_disclosed=True))
  writejl(R/'ALL390_BLINDED_EVIDENCE.jsonl',records)
  write(R/'ALL390_READOUT_INVENTORY.json',dict(completed=True,PASS=True,cases=390,frames=88,primary_groups={g:130 for g in 'ABC'},cross40=40,cards=390,case_assets=2730,existing_A_contacts=65,new_images=0,original_failed_directory_unchanged=True,original_array_hashes_and_saved_raw_pixel_relations_verified=True,unchanged_qL_native_selectedinstance_split=True,source_asset_reads=0,original_asset_inventory=protocol['asset_inventory']))
  # Verify preserved source evidence again after all metadata writes.
  for r in inv['inventory']:
   for b in r['images'].values():verify(b)
  for b in inv['contacts']:verify(b)
  write(R/'COMPLETE.json',dict(completed=True,PASS=True,entry=bind(entrypath),code=bind(__file__),protocol=entry['protocol'],original_render_failed=True,failure_scope='JSON metadata serialization only; all390 existing cards retained',original_failure=protocol['failed_attempt'],inventory=bind(R/'ALL390_READOUT_INVENTORY.json'),all390=bind(R/'ALL390_BLINDED_EVIDENCE.jsonl'),review_groups={g:bind(R/f'REVIEW_INPUT_GROUP_{g}.json') for g in 'ABC'},cross40=bind(R/'BLIND_CROSS40_ALL_REVIEWERS.json'),cases=390,frames=88,groups={g:130 for g in 'ABC'},source_RGB_NPZ_rereads=0,rerendered_or_modified_images=0,oldlabels_modeloutputs_votes_read=0,new_native_quality_owner_labels=0,automatic_training_admission=False))
  print(json.dumps(bind(R/'COMPLETE.json')),flush=True)
 except Exception as e:
  write(R/'FAILED.json',dict(completed=False,error=str(e),traceback=traceback.format_exc(),metadata_recovery_failure=True,automatic_retry=False));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','execute']);p.add_argument('--root-entry');a=p.parse_args()
 if a.mode=='prepare':prepare()
 else:assert a.root_entry;execute(a.root_entry)
