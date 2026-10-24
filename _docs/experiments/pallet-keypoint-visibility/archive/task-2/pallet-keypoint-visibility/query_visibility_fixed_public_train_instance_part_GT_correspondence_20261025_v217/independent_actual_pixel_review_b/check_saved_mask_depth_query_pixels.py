"""Independent exact-whitelist V217 pixel QA. No producer import, no SOURCE GT reopening."""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import numpy as np
from PIL import Image

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
A=V/'fixed12_mask_depth_query_diagnostic_preparation_a_v2';O=V/'fixed12_mask_depth_query_diagnostics_v2';OUT=H/'actual_review_v1'
PROTO=H/'PROTOCOL_BEFORE_INDEPENDENT_SAVED_PIXEL_QA.json'
INPUTS={
 'producer_code':{'path':str(A/'diagnose_fixed12_masks_depth_queries.py'),'sha256':'78e896bced453cf7e00ab75354f6dc8f1b5c195bb1817b3dddbfb0e046e5e1f3'},
 'producer_protocol':{'path':str(A/'PROTOCOL_BEFORE_MASK_DEPTH_PIXEL_READS.json'),'sha256':'14579dccedf02556e54661e275f7c317cd79bf53ea869ecb657ea1605b1d29c5'},
 'raw42':{'path':str(V/'fixed12_mask_depth_query_diagnostic_preparation_a/EXACT42_KNOWN_MASK_DEPTH_PNG_BINDINGS.json'),'sha256':'b62082c1ff9a30f23b9e116b9bf276137e6063c6f066a49050bd32fcec73c22c'},
 'contexts12':{'path':str(A/'EXACT12_ORIGINAL_DERIVED_CONTEXT_PNG_BINDINGS.json'),'sha256':'b2f9e45758ab9c42f7831be17b28c749fbce462c4af72d4089b9c9f9ade1ef4a'},
 'complete':{'path':str(O/'COMPLETE.json'),'sha256':'407cff2c84fda8aedf54aa6f34c65b4b295eda5c2f867cec698d3b3cf1929453'},
 'root_scope':{'path':str(V/'ROOT_ACTUAL_FIXED12_MASK_DEPTH_QUERY_DIAGNOSTIC_SCOPE.json'),'sha256':'65d51e271f4fece6e2ee4b1f4c37cf604524fa4719d33a7d993e44ab345d0dc3'},
}
ALLOWED_META={Path(x['path'])for x in INPUTS.values()}|{PROTO,Path(__file__)}

def need(x,m):
 if not x:raise AssertionError(m)
def safe(p):
 p=Path(p);need(p.is_absolute()and'..'not in p.parts and'v112'not in str(p).casefold(),'V112/relative path before FS');return p
def parentfd(p):
 p=safe(p);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for q in p.parts[1:-1]:
   n=os.open(q,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  return p,fd
 except BaseException:os.close(fd);raise
def rb(p,allowed):
 p=safe(p);need(p in allowed,'Exact approved path membership before filesystem');p,fd=parentfd(p)
 try:n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Regular no-symlink file')
 with os.fdopen(n,'rb')as f:return f.read()
def new(p,z):
 p=safe(p);need(p.parent in(H,OUT),'New owned audit outputs only');p,fd=parentfd(p)
 try:n=os.open(p.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'wb')as f:f.write(z)
def mkdir_new(p):
 p,fd=parentfd(p)
 try:os.mkdir(p.name,dir_fd=fd)
 finally:os.close(fd)
def sha(z):return hashlib.sha256(z).hexdigest()
def enc(v):return(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def canonical(b):return{key:b[key]for key in('path','sha256','size_bytes')if key in b}
def checked_bytes(b,allowed):
 p=safe(b['path']);z=rb(p,allowed);need(sha(z)==b['sha256'],'Exact opaque SHA')
 if 'size_bytes'in b:need(type(b['size_bytes'])is int and len(z)==b['size_bytes'],'Exact optional size')
 return z
def read(b,allowed):return json.loads(checked_bytes(b,allowed))
def binding(p):return dict(path=str(p),sha256=sha(rb(p,ALLOWED_META)))

def metadata():
 protocol=read(INPUTS['producer_protocol'],ALLOWED_META);raw=read(INPUTS['raw42'],ALLOWED_META)['payloads'];ctx=read(INPUTS['contexts12'],ALLOWED_META)['frames12']
 inputs=protocol['inputs'];wanted=('fixed12_frames','original96_inputs','V216_joined96','V216_root_semantic','selected_GT_complete')
 for name in wanted:
  p=safe(inputs[name]['path']);need(P in p.parents and p.suffix=='.json','Declared scalar/selected metadata only');ALLOWED_META.add(p)
 frames=read(inputs['fixed12_frames'],ALLOWED_META)['frames12'];original=read(inputs['original96_inputs'],ALLOWED_META)['rows96'];joined=read(inputs['V216_joined96'],ALLOWED_META)['rows96'];semantic=read(inputs['V216_root_semantic'],ALLOWED_META)['rows96'];gtcomplete=read(inputs['selected_GT_complete'],ALLOWED_META)
 b=gtcomplete['selected_GT_metadata'];need(Path(b['path'])==V/'fixed12_selected_GT_metadata_v1/ALL12_SELECTED_TRAIN_FRAME_GT_METADATA.json','Only generated selected GT rows');ALLOWED_META.add(Path(b['path']));gt=read(b,ALLOWED_META)['frames12']
 need(len(raw)==len({r['path']for r in raw})==42 and collections.Counter(r['kind']for r in raw)=={'FULL_INSTANCE_MASK_PNG':15,'VISIBLE_INSTANCE_MASK_PNG':15,'DEPTH_PNG':12},'Exact42 mask/depth whitelist')
 need(len(ctx)==12 and len({r['binding']['path']for r in ctx})==12 and len(frames)==12 and len(original)==len(joined)==len(semantic)==96,'Source metadata12/96/12 closure')
 need(sum(not r['original_query_available']for r in original)==11 and collections.Counter(r['internal_source_role']for r in original)=={'SOURCE_FIT':64,'SOURCE_DEV':32},'Original unique U11 and FIT64 DEV32')
 jmap={r['input']['row_id']:r for r in joined}
 for r in original:
  old=jmap[r['row_id']]['input']
  need(r['original_role']==old['role']=='TRAIN','Original outer TRAIN')
  for key in('native_slot','query_xy','L','original_query_available','component_id','internal_source_role','original_flags'):need(r[key]==old[key],'Original qL/native/U/role metadata '+key)
 multiplicity={f['frame_position']:len(f['obj1_candidate_gt_list_indices'])for f in frames};expandedU=sum(not r['original_query_available']for r in original for _ in range(multiplicity[r['frame_position']]))
 need(sum(multiplicity.values())==15,'All15 candidates');return frames,original,{r['input']['row_id']:r for r in joined},{r['row_id']:r for r in semantic},{f['frame_position']:f for f in gt},raw,ctx,expandedU

def depth_scale(camera):
 x=camera.get('depth_scale');return float(x)if type(x)in(int,float)and math.isfinite(x)and x>0 else None
def decode(z):
 with Image.open(io.BytesIO(z))as im:need(im.format=='PNG','PNG only');a=np.asarray(im).copy()
 need(a.ndim==2 and np.issubdtype(a.dtype,np.number),'Numeric single channel only');return a
def areas(f,v,z,scale):
 if f.shape!=v.shape or f.shape!=z.shape:return None
 F=f>0;W=v>0;D=np.isfinite(z)&(z>0);fa=np.count_nonzero(F);va=np.count_nonzero(W);fd=np.count_nonzero(F&D);vd=np.count_nonzero(W&D);outside=np.count_nonzero(W&~F)
 return dict(height=int(f.shape[0]),width=int(f.shape[1]),full_area=int(fa),visible_area=int(va),visible_outside_full_area=int(outside),visible_subset_full=bool(outside==0),visible_full_area_ratio=float(va/fa)if fa else None,depth_positive_finite_pixels=int(np.count_nonzero(D)),depth_zero_or_invalid_pixels=int(D.size-np.count_nonzero(D)),full_mask_positive_depth_pixels=int(fd),visible_mask_positive_depth_pixels=int(vd),full_mask_positive_depth_fraction=float(fd/fa)if fa else None,visible_mask_positive_depth_fraction=float(vd/va)if va else None,depth_scale=scale,metric_depth_status='BOP_DEPTH_SCALE_AVAILABLE_MM_PER_RAW_UNIT'if scale is not None else'UNKNOWN_CAMERA_DEPTH_SCALE_UNAVAILABLE',full_dtype=str(f.dtype),visible_dtype=str(v.dtype),depth_dtype=str(z.dtype))
def query(q,available,f,v,z,scale):
 out=dict(measure='ORIGINAL_RAW_IMAGE_NEAREST_FLOOR_Q_PLUS_0_5_AND_CLIPPED_3X3_NOT_RAW96_SAMPLER',original_inputU=not available,query_xy_unchanged=q,finite_original_q=False,nearest_x=None,nearest_y=None,original_q_in_continuous_image_canvas=None,nearest_pixel_in_frame=False,full_mask_at_nearest=None,visible_mask_at_nearest=None,raw_depth_at_nearest=None,positive_finite_depth_at_nearest=None,scaled_depth_mm_at_nearest=None,window3x3_in_frame_samples=0,window3x3_full_pixels=None,window3x3_visible_pixels=None,window3x3_positive_depth_pixels=None,new_native_visibility_label='UNKNOWN',training_admission=False)
 if q is None or len(q)!=2 or not all(type(x)in(int,float)and math.isfinite(x)for x in q):return out
 h,w=f.shape;x,y=q;ix=int(math.floor(x+.5));iy=int(math.floor(y+.5));inside=0<=ix<w and 0<=iy<h
 out.update(finite_original_q=True,nearest_x=ix,nearest_y=iy,original_q_in_continuous_image_canvas=(-.5<=x<w-.5 and -.5<=y<h-.5),nearest_pixel_in_frame=inside)
 if not inside:return out
 value=float(z[iy,ix]);valid=math.isfinite(value)and value>0;points=[(xx,yy)for yy in range(iy-1,iy+2)for xx in range(ix-1,ix+2)if 0<=xx<w and 0<=yy<h]
 out.update(full_mask_at_nearest=bool(f[iy,ix]>0),visible_mask_at_nearest=bool(v[iy,ix]>0),raw_depth_at_nearest=value if math.isfinite(value)else None,positive_finite_depth_at_nearest=valid,scaled_depth_mm_at_nearest=value*scale if valid and scale is not None else None,window3x3_in_frame_samples=len(points),window3x3_full_pixels=sum(bool(f[yy,xx]>0)for xx,yy in points),window3x3_visible_pixels=sum(bool(v[yy,xx]>0)for xx,yy in points),window3x3_positive_depth_pixels=sum(bool(math.isfinite(float(z[yy,xx]))and z[yy,xx]>0)for xx,yy in points));return out
def info_context(info,height,width):
 out=dict(GT_info_context_only=info,GT_bbox_obj_oob_status='UNKNOWN_METADATA_MISSING',global_visib_fract_is_not_native_occlusion=True);box=info.get('bbox_obj')if isinstance(info,dict)else None
 if isinstance(box,list)and len(box)==4 and all(type(x)in(int,float)and math.isfinite(x)for x in box):
  x,y,w,h=box
  if w<0 or h<0:out['GT_bbox_obj_oob_status']='NEGATIVE_BBOX_SENTINEL_NO_GEOMETRIC_SUPPORT'
  else:out.update(GT_bbox_obj_oob_status='BBOX_PARTLY_OUTSIDE'if x<0 or y<0 or x+w>width or y+h>height else'BBOX_WITHIN_CANVAS',GT_bbox_obj_clipped_area_fraction=(max(0,min(width,x+w)-max(0,x))*max(0,min(height,y+h)-max(0,y))/(w*h)if w*h else None))
 return out
def equal(a,b,label):
 if isinstance(a,dict):need(isinstance(b,dict)and set(a)==set(b),label+' dict keys');[equal(a[k],b[k],label+'.'+k)for k in a]
 elif isinstance(a,list):need(isinstance(b,list)and len(a)==len(b),label+' list');[equal(x,y,label)for x,y in zip(a,b)]
 elif type(a)is float:need(type(b)in(int,float)and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),label+' float')
 else:need(a==b,label+' scalar')
def fixtures():
 f=np.array([[1,0,1],[1,1,0]],dtype=np.uint8);v=np.array([[1,1,0],[0,1,0]],dtype=np.uint8);z=np.array([[10,0,30],[0,50,0]],dtype=np.uint16)
 s=areas(f,v,z,2.);need(s['full_area']==4 and s['visible_area']==3 and s['visible_outside_full_area']==1 and s['full_mask_positive_depth_pixels']==3,'Independent area/nonexclusive/depth fixture')
 q=query([-.5,-.5],False,f,v,z,2.);need(q['nearest_x']==0 and q['window3x3_in_frame_samples']==4 and q['original_inputU']and q['scaled_depth_mm_at_nearest']==20,'U remains U, boundary nearest/clipped3x3')
 need(not query([2.5,0],True,f,v,z,2.)['nearest_pixel_in_frame']and not query([-.5001,0],True,f,v,z,2.)['nearest_pixel_in_frame'],'Half-open image edge OOB without clamp')
 need(query(None,False,f,v,z,None)['new_native_visibility_label']=='UNKNOWN'and areas(f,v,z[:1],None)is None,'Missing q/dimension failure retained')
 need(info_context({'bbox_obj':[-1,-1,-1,-1],'px_count_visib':9,'px_count_valid':2},2,3)['GT_bbox_obj_oob_status']=='NEGATIVE_BBOX_SENTINEL_NO_GEOMETRIC_SUPPORT','Negative bbox sentinel and nonexclusive GT counts preserved')
 need(sum(u*c for u,c in zip([4,2,5],[2,2,1]))==17 and sum([4,2,5])==11,'Expanded candidate U derived separately from original unique U')
 return dict(completed=True,PASS=True,count=6,invented_only=True,actual_pixels_or_results_read=0)

def prepare():
 metadata();checks=fixtures();new(H/'INVENTED_PIXEL_QA_CHECKS.json',enc(checks))
 e=dict(approved_actual_independent_pixel_QA=False,code=binding(__file__),protocol=binding(PROTO),inputs=INPUTS,root_counter_addendum=None,output_directory=str(OUT),actual_tool_view=False,automatic_retry=False)
 new(H/'ACTUAL_QA_ENTRY_APPROVED_FALSE_TEMPLATE.json',enc(e));ready=dict(completed=True,PASS=True,prepared_only=True,code=e['code'],protocol=e['protocol'],checks=dict(path=str(H/'INVENTED_PIXEL_QA_CHECKS.json'),sha256=sha(enc(checks))),false_entry=dict(path=str(H/'ACTUAL_QA_ENTRY_APPROVED_FALSE_TEMPLATE.json'),sha256=sha(enc(e))),actual_pixel_or_saved_result_body_reads=0,no_producer_import=True,frozen=True)
 new(H/'FINAL_READY.json',enc(ready));print(json.dumps(dict(completed=True,PASS=True,ready=dict(path=str(H/'FINAL_READY.json'),sha256=sha(enc(ready))),code=e['code'],protocol=e['protocol'])))

def run(entry_path):
 ALLOWED_META.add(safe(entry_path));e=read(dict(path=str(entry_path),sha256=sha(rb(entry_path,ALLOWED_META))),ALLOWED_META)
 need(e['approved_actual_independent_pixel_QA']is True and e['automatic_retry']is False and e['actual_tool_view']is False and e['inputs']==INPUTS and e['output_directory']==str(OUT),'Root authorized one exact actual numeric audit')
 need(e['code']==binding(__file__)and e['protocol']==binding(PROTO),'Frozen independent code/protocol');read(INPUTS['root_scope'],ALLOWED_META)
 if e['root_counter_addendum']is not None:ALLOWED_META.add(safe(e['root_counter_addendum']['path']));read(e['root_counter_addendum'],ALLOWED_META)
 frames,original,joined,semantic,gt,raw,ctx,expandedU=metadata();mkdir_new(OUT)
 try:
  complete=read(INPUTS['complete'],ALLOWED_META);need(complete['completed']and complete['PASS'],'Producer terminal complete')
  need(complete['code']['path']==INPUTS['producer_code']['path']and complete['code']['sha256']==INPUTS['producer_code']['sha256']and complete['protocol']['sha256']==INPUTS['producer_protocol']['sha256']and complete['source_mask_depth_files_decoded']==42 and complete['derived_context_PNGs_decoded']==12 and complete['original_JPEG_RGB_NPY_CAD_NPZ_model_projection_PnP_or_fit_reads']==0 and complete['training_admission']is False,'Exact producer closure and no additional source/model/projection work')
  allowed_output={Path(complete[key]['path'])for key in('opaque_SHA_QA','summary','all_rows','candidate_CSV','query_CSV','gallery')}|{Path(p['image']['path'])for p in complete['panels']}|{Path(p['image']['path'])for p in complete['original_context12']}
  need(all(p.parent==O for p in allowed_output),'Only own exact producer outputs');allowed_raw={Path(r['path'])for r in raw}|{Path(c['binding']['path'])for c in ctx};need(len(allowed_raw)==54,'Exact54 original approved PNGs')
  buffers={r['path']:checked_bytes(canonical(r),allowed_raw)for r in raw};buffers.update({c['binding']['path']:checked_bytes(c['binding'],allowed_raw)for c in ctx})
  opaque=read(complete['opaque_SHA_QA'],allowed_output);need(len(opaque['files42'])==42 and len(opaque['context12'])==12 and opaque['decoded_images_before_receipt']==0,'Producer opaque identity first receipt')
  for b in opaque['files42']+opaque['context12']:need(b['path']in buffers and sha(buffers[b['path']])==b['sha256'],'Opaque receipt matches independent source SHA')
  saved=read(complete['all_rows'],allowed_output);summary=read(complete['summary'],allowed_output);actual_c=saved['candidate_rows15'];actual_q=saved['candidate_native_rows120']
  need(len(actual_c)==15 and len(actual_q)==120 and len({(r['frame_position'],r['candidate_gt_list_index'],r['native_slot'])for r in actual_q})==120,'All15/120 distinct rows')
  copies={c['frame_position']:c['image']for c in complete['original_context12']};need(set(copies)==set(range(12))and len(complete['original_context12'])==12,'Exact12 context-copy identities');panels={(p['frame_position'],p['candidate_gt_list_index']):p for p in complete['panels']};expected_c=[];expected_q=[];failures=[];expected_panels=set();ctx_by={c['frame_position']:c['binding']for c in ctx}
  for frame in frames:
   pos=frame['frame_position'];rr=[r for r in original if r['frame_position']==pos];g=gt[pos];scale=depth_scale(g['camera']);depth=decode(buffers[frame['depth']['path']]);context_bytes=buffers[ctx_by[pos]['path']]
   need(checked_bytes(copies[pos],allowed_output)==context_bytes,'Original context byte copy exact')
   with Image.open(io.BytesIO(context_bytes))as im:need(im.mode=='RGB','Original derived RGB context only');context_hw=(im.height,im.width)
   for ix in frame['obj1_candidate_gt_list_indices']:
    paths=frame['candidate_masks'][str(ix)];full=decode(buffers[paths['full_mask']['path']]);vis=decode(buffers[paths['visible_mask']['path']]);base=dict(frame_position=pos,frame_key=frame['frame_key'],component_id=frame['component_id'],internal_source_role=frame['internal_source_role'],outer_role='TRAIN',candidate_gt_list_index=ix,selected_instance_correspondence='UNKNOWN',native_visibility_admission=False);measures=areas(full,vis,depth,scale)
    if measures is None:status='FAILED_DIMENSIONS_PRESERVED';measures=dict(diagnostic_error='DIMENSION_MISMATCH_NO_RESIZE_OR_REPAIR',full_shape=list(full.shape),visible_shape=list(vis.shape),depth_shape=list(depth.shape));failures.append({**base,**measures})
    else:
     status='DIAGNOSTIC_COMPLETE_NO_LABEL'
     if context_hw!=full.shape:status='DIAGNOSTIC_COMPLETE_CONTEXT_DIMENSIONS_UNALIGNED';measures['context_shape']=[*context_hw,3];failures.append({**base,**measures})
     else:expected_panels.add((pos,ix))
    info=next(c for c in g['obj1_candidates']if c['gt_list_index']==ix)['GT_info'];measures.update(info_context(info,*full.shape));expected_c.append({**base,'diagnostic_status':status,**measures})
    for r in rr:
     q=query(r['query_xy'],r['original_query_available'],full,vis,depth,scale)if measures.get('height')is not None else dict(measure='NOT_COMPUTED_DIMENSION_MISMATCH',original_inputU=not r['original_query_available'],new_native_visibility_label='UNKNOWN',training_admission=False)
     expected_q.append({**base,'row_id':r['row_id'],'native_slot':r['native_slot'],'query_xy':r['query_xy'],'L':r['L'],'original_query_available':r['original_query_available'],'original_flags':r['original_flags'],'diagnostic_status':status,'saved_V216_root_weak_part_state':semantic[r['row_id']]['root_weak_part_state'],'A_ROI_raw96':joined[r['row_id']]['reader_A_axes']['witness']['roi'],'C_ROI_raw96':joined[r['row_id']]['reader_C_axes']['witness']['roi'],'ROI_support_sampling':'NOT_COMPUTED_NO_RAW96_MASK_SAMPLER_REPLAY',**q})
  equal(expected_c,actual_c,'candidate15');equal(expected_q,actual_q,'query120');equal(failures,saved['dimension_failures_preserved'],'dimension failures')
  expected_summary=dict(completed=True,frames=12,original_native_rows=96,original_inputU=11,candidate_instances=15,candidate_native_rows=120,rows_with_original_U=expandedU,candidate_dimension_failures=len(failures),panels_generated=len(expected_panels),metric_depth_camera_available_frames=sum(depth_scale(g['camera'])is not None for g in gt.values()),selected_instance_correspondence='UNKNOWN_ALL_CANDIDATES',all_new_native_visibility_labels='UNKNOWN',training_admission=False,query_measure='nearest raw-image pixel floor(q+.5) and clipped 3x3; NOT RAW96 bilinear sampling',mask_absence_or_query_foreign_is_not_E=True,no_ROI_mapping_no_native_projection_no_q_change=True)
  equal(expected_summary,summary,'SUMMARY');need(len({r['row_id']for r in actual_q})==96,'Unique original96/U11 and derived expanded U separate')
  for b,rows in((complete['candidate_CSV'],actual_c),(complete['query_CSV'],actual_q)):
   csvrows=list(csv.DictReader(io.StringIO(checked_bytes(b,allowed_output).decode())));need(len(csvrows)==len(rows),'CSV denominator')
   for c,r in zip(csvrows,rows):
    for k,value in c.items():
     target=r.get(k);text=json.dumps(target,ensure_ascii=False)if isinstance(target,(dict,list))else''if target is None else str(target);need(value==text,'CSV value vs exact JSON '+k)
  need(set(panels)==expected_panels and len(panels)==summary['panels_generated'],'Every expected panel retained')
  for key,panel in panels.items():
   need(Path(panel['image']['path']).name=='FRAME_%02d_CANDIDATE_%06d_MASKS_DEPTH_VALID.png'%key,'Panel frame/candidate number in filename');data=checked_bytes(panel['image'],allowed_output)
   with Image.open(io.BytesIO(data))as im:need(im.format=='PNG'and im.mode=='RGB'and im.size==(1920,525),'Panel pixel dimensions/mode; root owns actual readable marker view')
  gallery=checked_bytes(complete['gallery'],allowed_output).decode();links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',gallery)
  need(len(links)==len(copies)+len(panels)and len(set(links))==len(links),'Exact context+panel gallery link count')
  for rel in links:need('/'not in rel and'v112'not in rel.casefold()and O/rel in allowed_output,'Closed local gallery target');rb(O/rel,allowed_output)
  for record in raw:need(sha(rb(record['path'],allowed_raw))==record['sha256'],'Source raw mask/depth immutable after QA')
  for record in ctx:need(sha(rb(record['binding']['path'],allowed_raw))==record['binding']['sha256'],'Source derived context immutable after QA')
  report=dict(completed=True,PASS=True,evaluation_complete=INPUTS['complete'],complete=INPUTS['complete'],code=e['code'],protocol=e['protocol'],entry=dict(path=str(entry_path),sha256=sha(rb(entry_path,ALLOWED_META))),source_original_raw_SHA42_exact=True,source_derived_context_SHA12_and_copies_exact=True,all54_source_hashes_unchanged_after_QA=True,candidates=15,candidate_native_rows=120,unique_original_rows=96,unique_original_U=11,expanded_candidate_U=expandedU,original_FIT64_DEV32_preserved=True,all_mask_depth_areas_support_and_scalar_query_floor3x3_exact=True,GT_bbox_sentinel_OOB_context_exact=True,all120_native_visibility_UNKNOWN=True,training_admission=False,CSV_JSON_values_exact=True,panel_count=len(panels),panel_bindings_and_pixel_dimensions_exact=True,actual_tool_views=0,panel_native_number_legibility='ROOT_ACTUAL_VIEW_LANE_NOT_OCR_OR_CLAIMED_BY_B',gallery_relative_links_exact=len(links),dimension_failures_retained=len(failures),source_GT_camera_CAD_RGBarray_NPZ_models_scores_protected_reads=0,no_producer_import_or_helper=True,RAW96_sampler_replay=False,root_counter_addendum=e['root_counter_addendum'],root14_was_assumption_not_producer_failure=True,automatic_retry=False)
  new(OUT/'REVIEW.json',enc(report));new(OUT/'KO.md',('고정 SOURCE12의 15후보/120행을 독립 mask/depth pixel 연산과 원 query floor(q+0.5)·경계3×3 순회로 확인했다. 원 고유96/U11과 후보중복 U'+str(expandedU)+'를 분리했다. 원 RAW96 bilinear sampler나 intended-native 가림 라벨의 검증은 아니다. 새 native 라벨은 전부 UNKNOWN, 학습 admission은 false다. 원42 mask/depth와12 derived context의 SHA 및 복제 bytes를 확인했다. 패널 '+str(len(panels))+'장의 binding/크기와 gallery 링크를 검사했으며 B tool 실제뷰/marker 가독성 주장은 0이다. 모델·GT 본체·보호셋·배열 입력을 열지 않았고 producer helper를 import하지 않았다.\n').encode());print(json.dumps(dict(completed=True,PASS=True,REVIEW=dict(path=str(OUT/'REVIEW.json'),sha256=sha(enc(report))),expanded_candidate_U=expandedU)))
 except BaseException as error:new(OUT/'FAILURE_PRESERVED.json',enc(dict(completed=False,PASS=False,error_type=type(error).__name__,error=str(error),automatic_retry=False)));raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--entry');a=parser.parse_args();prepare()if a.prepare else run(a.entry)
