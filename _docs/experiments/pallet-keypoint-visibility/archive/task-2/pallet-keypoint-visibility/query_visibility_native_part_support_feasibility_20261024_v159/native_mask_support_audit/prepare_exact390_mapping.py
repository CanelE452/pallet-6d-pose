"""Exact existing selected-instance/native mapping; metadata only, no RGB/mask reads."""
from pathlib import Path
import hashlib,json,re
from collections import Counter
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_native_part_support_feasibility_20261024_v159';O=Path(__file__).resolve().parent
S=P/'query_visibility_mr6d_full_public_source_20261024_v153';B=[]
def guard(p):
 p=Path(p);s=str(p).lower();assert p.is_absolute() and str(p).startswith(str(P)+'/')
 assert not any(t in s for t in ('v112','current223','/private'))
 assert p.suffix.lower() in ('.json','.jsonl','.py');return p
def bind(p,expected=None):
 p=guard(p);b=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 if expected:assert b['sha256']==expected,p
 if b not in B:B.append(b)
 return b
def read(p,expected=None):bind(p,expected);return json.loads(guard(p).read_text())
def selected_lines(p,key,values,expected):
 bind(p,expected);out=[];pat=re.compile('"'+key+'"\\s*:\\s*(\\d+)')
 for line in guard(p).read_text().splitlines():
  m=pat.search(line);assert m
  if int(m.group(1)) in values:out.append(json.loads(line))
 return out
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
scope=read(V/'ROOT_SCOPE_BEFORE_SOURCE_NATIVE_SUPPORT_BODIES.json','49ca7c44ac80611c8be988b9c30f020528d4090ee40a31f8825a405690c25be3')
seal=read(O/'CASE_AND_CALCULATION_SEAL_BEFORE_MASK_BODIES.json');ids=set(seal['global_indices']);frames=set(seal['frame_indices'])
source=selected_lines(scope['source_fixed_rows']['path'],'global_index',ids,scope['source_fixed_rows']['sha256'])
old=selected_lines(P/'query_visibility_mr6d_source_supervision_bridge_20261024_v154/ROOT_ALL1840_SEPARATE_QUERY_SUPERVISION.jsonl','global_index',ids,'e5c37048b35fc4194b0ec22acb1c045e514a09c8355385d839e7edd6c96baf68')
old={r['global_index']:r for r in old};assert len(source)==len(old)==390
refseal=read(S/'native_reference_preparation/provided_reference_v1/REFERENCE_PROJECTION_COMPLETE.json')
rb=refseal['references'];references=selected_lines(rb['path'],'frame_index',frames,rb['sha256'])
ref={(r['frame_index'],r['gt_instance_index'],r['native_slot']):r for r in references}
geomall=read(refseal['frames_geometry']['path'],refseal['frames_geometry']['sha256']);geom={r['frame_index']:r for r in geomall['frames'] if r['frame_index'] in frames}
intake=read(S/'full_source_v1/FULL_SOURCE_INTAKE_COMPLETE.json');sel=read(intake['selection']['path'],intake['selection']['sha256'])
selection={r['frame_index']:r for r in sel['frames'] if r['frame_index'] in frames}
dl=read(intake['download']['path'],intake['download']['sha256']);payload={r['relative_path']:r for r in dl['files']}
facts=read(intake['factual_summary']['path'],intake['factual_summary']['sha256']);factual={r['frame_index']:r for r in facts['frames'] if r['frame_index'] in frames}
def maskbind(rel):
 d=payload[rel];return dict(path=str(S/'full_source_v1/original'/rel),sha256=d['sha256'],bytes=d['bytes'],relative_path=rel)
rows=[];frameout=[]
for fi in sorted(frames):
 g=geom[fi];s=selection[fi];assert g['split']==s['split']=='SOURCE_DEVELOPMENT'
 masks=[]
 for inst in s['obj1_gt_instance_indices']:
  full=next(x['path'] for x in s['source_files']['full_masks'] if x['gt_instance_index']==inst)
  vis=next(x['path'] for x in s['source_files']['visible_masks'] if x['gt_instance_index']==inst)
  masks.append(dict(gt_instance_index=inst,full_mask=maskbind(full),visible_mask=maskbind(vis)))
 frameout.append(dict(frame_index=fi,source_id=s['source_id'],split=s['split'],raw_hw=g['raw_hw'],geometry=g,mask_bindings=masks,factual_summary=factual[fi]))
frame_by={r['frame_index']:r for r in frameout}
for r in sorted(source,key=lambda x:x['global_index']):
 i=r['global_index'];a=old[i];fi=r['frame_index'];gt=a['selected_gt_instance_index'];native=r['native_slot']
 assert a['split']==r['split']=='SOURCE_DEVELOPMENT' and a['original_input_valid'] and a['role_target']>=0 and gt is not None
 assert a['query_id']==r['query_id'] and a['query_xy']==r['original_query_xy'] and a['L']==r['original_L']
 assert a['role_state']==r['original_role_state'] and a['role_target']==r['original_role_target']
 refrow=ref[(fi,gt,native)];g=next(x for x in geom[fi]['instances'] if x['gt_instance_index']==gt)
 assert refrow['reference_xy']==g['dynamic_projected8'][native]
 mask=next(x for x in frame_by[fi]['mask_bindings'] if x['gt_instance_index']==gt)
 rows.append(dict(global_index=i,query_id=r['query_id'],frame_index=fi,source_id=r['source_id'],split=r['split'],native_slot=native,
  query_xy=r['original_query_xy'],L=r['original_L'],original_role_state=r['original_role_state'],original_role_target=r['original_role_target'],
  weak_visible_owner_support=r['final_visible_owner_support'],center4_valid=r['center4_valid'],selected_gt_instance_index=gt,
  native_name=refrow['native_name'],fixed_canonical_index=refrow['fixed_canonical_index'],reference_xy=refrow['reference_xy'],
  reference_model_XYZ_mm=refrow['reference_model_XYZ_mm'],reference_continuous_inframe=refrow['continuous_inframe'],
  reference_point_uncertainty=refrow['reference_point_uncertainty'],source_masks_not_exclusive_owner_truth=refrow['source_masks_not_exclusive_owner_truth'],
  selected_instance_mask_bindings=mask,other_instance_mask_bindings=[x for x in frame_by[fi]['mask_bindings'] if x['gt_instance_index']!=gt],
  selected_instance_geometry=g,original_supervision_selected_instance_reason=a['association_reason'],
  native_material_or_physical_point_certification='UNKNOWN_MASK_GEOMETRY_ONLY',point_quality_target=-1,training_admitted=False))
out=O/'ALL390_EXACT_SELECTED_NATIVE_REFERENCE_MAPPING.jsonl'
with out.open('x') as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
fo=O/'SELECTED88_TRAIN_FRAME_GEOMETRY_AND_MASK_BINDINGS.json';write(fo,dict(completed=True,frames=frameout))
complete=O/'EXACT390_MAPPING_COMPLETE_BEFORE_MASK_BODIES.json'
write(complete,dict(completed=True,rows=390,frames=88,roles=dict(Counter(r['original_role_state'] for r in rows)),
 mapping=bind(out),frame_geometry=bind(fo),source_bindings=B,code=bind(__file__),mask_or_RGB_body_reads=0,
 selected_instance_or_native_optimized_or_changed=False,fullsource_mixed_JSONL_policy='Only sealed390indices or88TRAINframe records decoded/used; no heldRGB or heldlabel records analyzed. Existing wholefilehashes verified for immutable binding.',new_labels_created=0))
print(json.dumps(dict(completed=True,mapping=bind(out),complete=bind(complete),rows=390,frames=88),indent=2))
