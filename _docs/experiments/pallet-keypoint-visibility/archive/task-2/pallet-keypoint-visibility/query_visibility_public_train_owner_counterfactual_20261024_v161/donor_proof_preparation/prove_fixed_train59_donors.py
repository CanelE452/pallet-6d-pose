"""V161 fixed59 donor proof. --prepare reads metadata/code + invented arrays only.

Actual pixel/NPZ proof requires separately bound approved root entry. No donor
assignment/search/replacement; no features, models, encoder, fitting or native
visibility labels. Imports only NumPy/Pillow and a single pinned pure AST function.
"""
from pathlib import Path
import argparse, ast, collections, datetime, hashlib, json, sys, traceback
import numpy as np
from PIL import Image, __version__ as PIL_VERSION
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DESIGN=V/'ROOT_SOURCE_GENERATION_DESIGN_BEFORE_PIXELS.json'
DESIGN_SHA='25f38eb0e8b098b681a8bb38e5a3f93086a1a24cdc9fdfa8f3300412382c332b'
V95=P/'query_visibility_same_q_counterfactual_20261023_v95'
CERTS=V95/'source_pair_intake/image_only_v2_exclusive_alpha/actual/SELECTED_DONOR_RGBA_CERTIFICATES.jsonl'
V95_SEAL=V95/'source_pair_intake/image_only_v2_exclusive_alpha/IMAGE_ONLY_INTAKE_SEAL.json'
PURE=P/'query_visibility_matched_query_scalar_20261023_v94/paired_source_preparation/same_q_pair_builder.py'
PURE_SHA='7450f610bd3f7567c2f87e9fe18197e5a064b64ba02c61f1a3e58d85ed3731fb'
PROTOCOL=H/'PROTOCOL_BEFORE_DONOR_PIXEL_OR_NPZ_BODIES.json'
FIXED=H/'FIXED_TRAIN59_DONOR_PROOF_INPUTS.json'
GENERATED=H/'GENERATED_DONOR_PROOF_CHECKS.json'
OUT=V/'donor_proof_v1'
OWNERLESS='Controlled augmentation opacity/native mask ownership proof only; not measured optical opacity, human/native visibility or coordinate gold.'
def safe(p):
 p=Path(p).resolve();s=str(p).lower()
 assert p.is_relative_to(P) and not any(x in s for x in ('_v112','/private_evaluation/','current223','/current_evaluation/'))
 return p
def sha(p):
 h=hashlib.sha256()
 with safe(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def meta_bind(p):
 p=safe(p);assert p.suffix in ('.json','.jsonl','.py')
 return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(safe(p).read_text())
def rows(p):return [json.loads(s) for s in safe(p).read_text().splitlines() if s]
def verify_meta(b):assert meta_bind(b['path'])['sha256']==b['sha256'];return safe(b['path'])
def write(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def array_meta(a):return dict(shape=list(a.shape),dtype=str(a.dtype),array_sha256=array_sha(a))
def runtime():return dict(executable=str(Path(sys.executable).resolve()),python=sys.version.split()[0],numpy=np.__version__,pillow=PIL_VERSION)
def erode1(a):
 assert a.ndim==2 and a.dtype==bool
 return np.lib.stride_tricks.sliding_window_view(np.pad(a,1,constant_values=False),(3,3)).all(axis=(-1,-2))
def pure_cutout():
 assert sha(PURE)==PURE_SHA
 nodes=[n for n in ast.parse(PURE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='prepare_cutout'];assert len(nodes)==1
 node=nodes[0];scope={'np':np,'Image':Image};exec(compile(ast.Module(body=[node],type_ignores=[]),str(PURE),'exec'),scope)
 return scope['prepare_cutout'],hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def expected_native(rgb,selected,other):
 assert rgb.dtype==np.uint8 and rgb.shape==(*selected.shape,3) and selected.dtype==other.dtype==bool and selected.shape==other.shape
 exclusive=selected&~other;core=erode1(exclusive);yy,xx=np.nonzero(core)
 assert len(xx)>0,'EMPTY_EXCLUSIVE_ERODE1_CORE'
 bbox=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1];x0,y0,x1,y1=bbox
 return exclusive,core,bbox,rgb[y0:y1,x0:x1].copy(),core[y0:y1,x0:x1].copy()
def validate_stored_arrays(rgb,selected,other,native,compact,certificate,prepare_cutout):
 exclusive,core,bbox,want_rgb,want_alpha=expected_native(rgb,selected,other)
 assert set(native)=={'selected_visible','other_visible_union','exclusive_alpha','erode1_exclusive_alpha'}
 for k,want in [('selected_visible',selected),('other_visible_union',other),('exclusive_alpha',exclusive),('erode1_exclusive_alpha',core)]:
  assert native[k].dtype==bool and native[k].shape==want.shape and np.array_equal(native[k],want),k
 assert not (core&other).any() and not (core&~selected).any()
 assert set(compact)=={'rgb','alpha'} and compact['rgb'].dtype==np.uint8 and compact['alpha'].dtype==bool
 assert np.array_equal(compact['rgb'],want_rgb) and np.array_equal(compact['alpha'],want_alpha)
 longest=max(want_alpha.shape);reproduced=prepare_cutout(rgb,exclusive,longest)
 assert reproduced is not None and reproduced['donor_native_bbox_xyxy']==bbox and reproduced['resized_hw']==list(want_alpha.shape)
 assert np.array_equal(reproduced['rgb'],compact['rgb']) and np.array_equal(reproduced['alpha'],compact['alpha'])
 assert certificate['native_bbox_xyxy']==bbox
 counts=dict(original_selected_visible_pixels=int(selected.sum()),original_conflicting_pixels=int((selected&other).sum()),excluded_conflicting_pixels=int((selected&other).sum()),retained_exclusive_alpha_pixels=int(exclusive.sum()),retained_erode1_pixels=int(core.sum()))
 ec=certificate['exclusive_alpha_certificate']
 for key,value in counts.items():assert ec[key]==value,(key,ec[key],value)
 assert ec['admitted'] and ec['complete_other_instance_masks'] and ec['every_eroded_native_alpha_pixel_selected_and_not_other']
 assert certificate['visible_pixels']==certificate['metadata_visible_pixels']==int(selected.sum())
 assert certificate['eroded_core_pixels']==int(erode1(selected).sum())
 assert certificate['selected_other_instance_visible_overlap_in_eroded_core']==int((erode1(selected)&other).sum())
 return dict(native_bbox_xyxy=bbox,counts=counts,raw_hw=list(selected.shape),rgb=array_meta(compact['rgb']),alpha=array_meta(compact['alpha']),native_arrays={k:array_meta(v) for k,v in native.items()},pinned_prepare_cutout_reproduced=True,independent_raw_RGB_bbox_and_core_relation_exact=True,scale_in_original_compact_production=1.0)
def generated_checks():
 fn,ast_sha=pure_cutout();rgb=np.arange(13*17*3,dtype=np.uint16).reshape(13,17,3).astype(np.uint8);selected=np.zeros((13,17),bool);selected[1:12,1:16]=True;other=np.zeros_like(selected);other[4:7,6:9]=True
 exclusive,core,bbox,crgb,alpha=expected_native(rgb,selected,other)
 native=dict(selected_visible=selected,other_visible_union=other,exclusive_alpha=exclusive,erode1_exclusive_alpha=core);compact=dict(rgb=crgb,alpha=alpha)
 cert=dict(native_bbox_xyxy=bbox,visible_pixels=int(selected.sum()),metadata_visible_pixels=int(selected.sum()),eroded_core_pixels=int(erode1(selected).sum()),selected_other_instance_visible_overlap_in_eroded_core=int((erode1(selected)&other).sum()),exclusive_alpha_certificate=dict(admitted=True,complete_other_instance_masks=True,every_eroded_native_alpha_pixel_selected_and_not_other=True,original_selected_visible_pixels=int(selected.sum()),original_conflicting_pixels=int((selected&other).sum()),excluded_conflicting_pixels=int((selected&other).sum()),retained_exclusive_alpha_pixels=int(exclusive.sum()),retained_erode1_pixels=int(core.sum())))
 proof=validate_stored_arrays(rgb,selected,other,native,compact,cert,fn);assert proof['pinned_prepare_cutout_reproduced']
 bad=0
 for field in native:
  n={k:v.copy() for k,v in native.items()};n[field][2,2]=~n[field][2,2]
  try:validate_stored_arrays(rgb,selected,other,n,compact,cert,fn)
  except AssertionError:bad+=1
 assert bad==4
 n={k:v.copy() for k,v in compact.items()};n['rgb'][0,0,0]^=1
 try:validate_stored_arrays(rgb,selected,other,native,n,cert,fn)
 except AssertionError:bad+=1
 n={k:v.copy() for k,v in compact.items()};n['alpha'][0,0]=~n['alpha'][0,0]
 try:validate_stored_arrays(rgb,selected,other,native,n,cert,fn)
 except AssertionError:bad+=1
 assert bad==6
 assert not erode1(np.ones((2,2),bool)).any() and not erode1(np.zeros((5,5),bool)).any()
 try:expected_native(rgb,selected,selected)
 except AssertionError:bad+=1
 assert bad==7
 return dict(completed=True,PASS=True,invented_arrays_only=True,pinned_prepare_cutout_ast_sha256=ast_sha,independent_exclusive_core_and_bbox_relation=True,all4_native_array_tampering_rejected=True,compact_RGB_and_alpha_tampering_rejected=True,empty_core_rejected=True,zero_outside_erosion_boundary_checked=True,source_pixel_NPZ_or_image_body_reads=0,model_feature_weight_calls=0)
def prepare():
 assert sha(DESIGN)==DESIGN_SHA;d=read(DESIGN)
 assert d['source_only'] and d['number_of_donors']==59 and not any(d[k] for k in ('training_allowed','encoder_allowed','initializer_allowed','current_allowed','V112_allowed','PnP_allowed','q_L_or_native_or_split_change_allowed','automatic_retry'))
 catalog=read(verify_meta(d['donor_catalog']))['catalog'];assert len(catalog)==59 and len(set(r['id'] for r in catalog))==59 and all(r['split']=='TRAIN' for r in catalog)
 seal=read(V95_SEAL);assert seal['completed'] and seal['PASS'];cb=meta_bind(CERTS);assert any(x['path']==cb['path'] and x['sha256']==cb['sha256'] for x in seal['files'])
 cert_by={r['id']:r for r in rows(CERTS)};inputs=[]
 for r in sorted(catalog,key=lambda r:r['id']):
  c=cert_by[r['id']];assert c['split']=='TRAIN' and c['foreign_opaque_cutout_admitted'] and c['replacement_allowed'] is False
  for key in r.keys()-{'compact_cutout_stat','native_alpha_proof_stat','future_MR_assignment_or_admission_performed'}:assert r[key]==c[key],(r['id'],key)
  assert c['scene_GT_owner_completeness']['complete'] and c['exclusive_alpha_certificate']['admitted']
  inputs.append(c)
 write(FIXED,dict(completed=True,source_only=True,count=59,donor_ids=[r['id'] for r in inputs],records=inputs,source_catalog=d['donor_catalog'],source_certificate=cb))
 g=dict(code=meta_bind(__file__),**generated_checks());write(GENERATED,g)
 p=dict(schema='V161_FIXED_TRAIN59_DONOR_PIXEL_PROOF_PROTOCOL',completed=True,prepared_only=True,code=meta_bind(__file__),design=meta_bind(DESIGN),catalog=d['donor_catalog'],source_V95_certificate=cb,source_V95_seal=meta_bind(V95_SEAL),fixed_inputs=meta_bind(FIXED),generated_checks=meta_bind(GENERATED),pure_prepare_cutout_source=meta_bind(PURE),pure_prepare_cutout_ast_sha256=g['pinned_prepare_cutout_ast_sha256'],runtime=runtime(),fixed_count=59,donor_selection_or_replacement=False,required_entry_approval='approved_actual_donor_pixel_proof',actual_output=str(OUT),output_schema=dict(complete='DONOR_PROOF_COMPLETE.json',records='PROVED_TRAIN59_DONORS.jsonl',record_keys=['donor_id','id','split','PASS','reason','compact_RGBA_cutout','native_alpha_proof','original_RGB','decoded_RGB_sha256','complete_mask_inventory','array_proof','runtime_cutout_contract','semantics']),runtime_cutout_contract=dict(file_keys=['rgb','alpha'],rgb='uint8[H,W,3] HWC RGB, contiguous when used',alpha='bool[H,W]; exact erode1(selected & ~union(all_other_visible)) within native nonzero-core bbox',original_compact_scale='1.0; pinned prepare_cutout longest equals native core bbox longside',builder_transform='Builder alone applies frozen V161 resize, conservative hard alpha and extra erode1; proof does not transform donors.'),semantics=OWNERLESS,source_pixel_or_NPZ_body_reads_during_preparation=0)
 write(PROTOCOL,p)
 template=dict(approved_actual_donor_pixel_proof=False,source_only=True,code=meta_bind(__file__),protocol=meta_bind(PROTOCOL),design=meta_bind(DESIGN),training_allowed=False,encoder_allowed=False,initializer_allowed=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,automatic_retry=False)
 write(H/'ROOT_ACTUAL_DONOR_PROOF_ENTRY_TEMPLATE_FALSE.json',template)
 write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=meta_bind(__file__),protocol=meta_bind(PROTOCOL),fixed_inputs=meta_bind(FIXED),generated_checks=meta_bind(GENERATED),entry_template=meta_bind(H/'ROOT_ACTUAL_DONOR_PROOF_ENTRY_TEMPLATE_FALSE.json'),source_pixel_NPZ_or_image_body_reads=0,actual_proof_performed=False))
 print(json.dumps(dict(completed=True,prepared_only=True,preparation=meta_bind(H/'PREPARATION_COMPLETE.json'),protocol=meta_bind(PROTOCOL))))
def actual(entry):
 e=read(entry);assert e['approved_actual_donor_pixel_proof'] is True and e['source_only'] is True
 assert not any(e[k] for k in ('training_allowed','encoder_allowed','initializer_allowed','current_allowed','V112_allowed','PnP_allowed','automatic_retry'))
 assert e['code']==meta_bind(__file__) and e['protocol']==meta_bind(PROTOCOL) and e['design']==meta_bind(DESIGN) and sha(DESIGN)==DESIGN_SHA and not OUT.exists()
 p=read(PROTOCOL);assert p['runtime']==runtime();verify_meta(p['fixed_inputs']);verify_meta(p['generated_checks']);verify_meta(p['catalog']);verify_meta(p['source_V95_certificate']);verify_meta(p['source_V95_seal']);verify_meta(p['pure_prepare_cutout_source'])
 fixed=read(FIXED);rr=fixed['records'];assert len(rr)==59 and len(set(r['id'] for r in rr))==59 and all(r['split']=='TRAIN' for r in rr)
 prepare_cutout,ast_sha=pure_cutout();assert ast_sha==p['pure_prepare_cutout_ast_sha256']
 OUT.mkdir();write(OUT/'STARTED.json',dict(completed=False,entry=meta_bind(entry),protocol=meta_bind(PROTOCOL),fixed_count=59,automatic_retry=False))
 # Only paths bound by the fixed TRAIN59 certificates can be opened as assets.
 allowed={}
 for r in rr:
  for b in [r['image'],r['mask'],r['native_alpha_proof'],r['compact_RGBA_cutout'],r['scene_GT_owner_completeness']['scene_gt']]+[x['mask'] for x in r['scene_GT_owner_completeness']['instance_masks']]:
   key=str(safe(b['path']));assert key not in allowed or allowed[key]['sha256']==b['sha256'];allowed[key]=b
 checked={};images={};masks={};scenes={}
 def verify_asset(b):
  path=safe(b['path']);assert str(path) in allowed and allowed[str(path)]['sha256']==b['sha256']
  if str(path) not in checked:assert sha(path)==b['sha256'],str(path);checked[str(path)]={'path':str(path),'sha256':b['sha256']}
  return path
 def load_mask(b,hw):
  path=verify_asset(b)
  if str(path) not in masks:
   with Image.open(path) as im:a=np.asarray(im).copy()
   assert a.dtype==np.uint8 and list(a.shape)==hw and set(np.unique(a).tolist()).issubset({0,255})
   masks[str(path)]=a>0
  assert list(masks[str(path)].shape)==hw;return masks[str(path)]
 results=[]
 for r in rr:
  result=dict(donor_id=r['id'],id=r['id'],split='TRAIN',PASS=False,reason=None,semantics=OWNERLESS,donor_search_or_replacement=False)
  try:
   rgbpath=verify_asset(r['image'])
   if str(rgbpath) not in images:
    with Image.open(rgbpath) as im:images[str(rgbpath)]=np.asarray(im.convert('RGB')).copy()
   rgb=images[str(rgbpath)];assert rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3
   assert array_sha(rgb)==r['image_decoded_RGB_sha256'];hw=list(rgb.shape[:2]);owner=r['scene_GT_owner_completeness'];scene_path=verify_asset(owner['scene_gt'])
   if str(scene_path) not in scenes:scenes[str(scene_path)]=read(scene_path)
   frame=scenes[str(scene_path)][str(r['frame_id'])];inventory=[dict(instance_id=i,object_id=int(x['obj_id'])) for i,x in enumerate(frame)]
   declared=owner['instance_masks'];assert [x['instance_id'] for x in declared]==list(range(len(frame)))==owner['expected_instance_ids']==owner['available_metadata_instance_ids']
   assert [x['object_id'] for x in declared]==[x['object_id'] for x in inventory] and owner['object_ids_match'] and owner['complete'] and owner['all_expected_visible_mask_files_exist']
   mask_dir=safe(r['mask']['path']).parent;expected_names={f"{r['frame_id']:06d}_{i:06d}.png" for i in range(len(frame))};actual_names={x.name for x in mask_dir.glob(f"{r['frame_id']:06d}_*.png")};assert actual_names==expected_names
   assert r['mask']['path']==declared[r['instance_id']]['mask']['path'] and r['object_id']==inventory[r['instance_id']]['object_id']
   selected=load_mask(r['mask'],hw);other=np.zeros_like(selected)
   for x in declared:
    a=load_mask(x['mask'],hw)
    if x['instance_id']!=r['instance_id']:other|=a
   assert r['other_instance_masks']==len(frame)-1
   with np.load(verify_asset(r['native_alpha_proof']),allow_pickle=False) as z:native={k:z[k] for k in z.files}
   with np.load(verify_asset(r['compact_RGBA_cutout']),allow_pickle=False) as z:compact={k:z[k] for k in z.files}
   proof=validate_stored_arrays(rgb,selected,other,native,compact,r,prepare_cutout)
   result.update(PASS=True,reason='ALL_FIXED_NATIVE_EXCLUSIVE_AND_COMPACT_RELATIONS_EXACT',compact_RGBA_cutout=r['compact_RGBA_cutout'],native_alpha_proof=r['native_alpha_proof'],original_RGB=r['image'],decoded_RGB_sha256=r['image_decoded_RGB_sha256'],complete_mask_inventory=dict(scene_gt=owner['scene_gt'],scene_id=r['scene_id'],frame_id=r['frame_id'],instances=[dict(**x,mask=declared[x['instance_id']]['mask']) for x in inventory],all_expected_frame_mask_files_exact=True),array_proof=proof,runtime_cutout_contract=p['runtime_cutout_contract'],source_certificate=p['source_V95_certificate'])
  except Exception:
   result.update(reason='FIXED_DONOR_PROOF_FAILED_NO_REPLACEMENT',traceback=traceback.format_exc())
  results.append(result)
 with (OUT/'PROVED_TRAIN59_DONORS.jsonl').open('x') as f:
  for r in results:f.write(json.dumps(r,allow_nan=False)+'\n')
 passed=all(r['PASS'] for r in results)
 complete=dict(schema='V161_FIXED_TRAIN59_DONOR_PIXEL_PROOF_COMPLETE',completed=True,PASS=passed,source_only=True,actual_pixel_proof=True,attempted=59,passed=sum(r['PASS'] for r in results),failed=sum(not r['PASS'] for r in results),all59_fixed_ids_retained=[r['donor_id'] for r in results]==fixed['donor_ids'],records=meta_bind(OUT/'PROVED_TRAIN59_DONORS.jsonl'),code=meta_bind(__file__),protocol=meta_bind(PROTOCOL),entry=meta_bind(entry),design=meta_bind(DESIGN),fixed_inputs=meta_bind(FIXED),source_catalog=p['catalog'],source_V95_certificate=p['source_V95_certificate'],runtime_cutout_contract=p['runtime_cutout_contract'],asset_hash_bindings=list(checked.values()),unique_hashed_source_assets=len(checked),no_donor_search_or_replacement=True,held_or_validation_donors_read=0,new_target_RGB_or_pair_construction=0,feature_weight_encoder_model_fit_PnP_calls=0,semantics=OWNERLESS,native_visibility_or_pointquality_labels_created=False,training_admitted=False,automatic_retry=False,selected_model=None,fully_solved=False)
 write(OUT/'DONOR_PROOF_COMPLETE.json',complete);print(json.dumps(dict(completed=True,PASS=passed,proof=meta_bind(OUT/'DONOR_PROOF_COMPLETE.json'),passed=complete['passed'],failed=complete['failed'])))
 if not passed:raise SystemExit('Fixed donor proof has preserved failures; no construction admission or automatic retry')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--entry',type=Path);args=parser.parse_args()
 if args.prepare and not args.entry:prepare()
 elif args.entry and not args.prepare:
  try:
   assert not (H/'FAILED_ACTUAL_DONOR_ENTRY_BEFORE_RUN.json').exists();actual(args.entry)
  except Exception:
   destination=OUT/'FAILED_EXECUTION.json' if OUT.exists() else H/'FAILED_ACTUAL_DONOR_ENTRY_BEFORE_RUN.json'
   if not destination.exists():write(destination,dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False))
   raise
 else:raise SystemExit('Use --prepare or separately root-approved --entry')
