"""V107 metadata-only inventory. Does not open RGB/features/weights/GT."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
from target_context_math import query_extent,valid_box,target_context_grid,sample_target_context
V=Path(__file__).resolve().parent;P=V.parent
V96=P/'query_visibility_query_anchored_cue_20261023_v96';V99=P/'query_visibility_diverse_expected_cue_source_20261023_v99'
PATHS=dict(old_base=V96/'actual_query_anchored_image_intake/SOURCE960_DRIVER_LINEAGE.jsonl',old_occ=V96/'actual_query_anchored_image_intake/OCCURRENCES31680_LINEAGE.jsonl',old_pack=V96/'actual_source_pair_input_pack/PACK_MANIFEST.json',loco_proposals=P/'query_visibility_loco_native_context_20261023_v74/public_initializer/DECODED_RGB_V2_RUN/FROZEN_PROPOSALS.jsonl',pallox_proposals=P/'query_visibility_matched_query_scalar_20261023_v94/pallox_actual_initializer/FIXED100/FROZEN_PROPOSALS_ALL100.jsonl',new_occ=V99/'matching_feature_pack/INPUT_LINEAGE2304_INPUT_ONLY.jsonl',new_RGB=V99/'render_preparation/actual_corpus/FULL_RGB288_INPUT_ONLY.jsonl',new_q=V99/'geometry_query_preparation/QUERY_INPUTS96_INPUT_ONLY.jsonl')
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def jl(p,rs):
 with p.open('x')as f:
  for r in rs:f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n')
bases=rows(PATHS['old_base']);old=rows(PATHS['old_occ']);new=rows(PATHS['new_occ']);qnew=rows(PATHS['new_q']);images=rows(PATHS['new_RGB'])
assert[len(bases),len(old),len(new),len(qnew),len(images)]==[960,31680,2304,96,288]
oldpack=json.loads(PATHS['old_pack'].read_text());oldinput=rows(oldpack['input_lineage']['path']);assert len(oldinput)==31680
loco={r['input_id']:r for r in rows(PATHS['loco_proposals'])};pal={r['frame_input_id']:r for r in rows(PATHS['pallox_proposals'])}
defs=[];byid={};boxstats={};fallback=False
for i,base in enumerate(bases):
 assert base['base_row_index']==i
 key=base['source']+':'+base['input_id']
 if key not in byid:
  pr=(loco if base['source']=='LOCO'else pal)[base['input_id']];assert pr['image_sha256']==base['original_image_sha256']
  selected=pr['selected_index'];q8=[None]*8;box=None;reason='NO_SELECTED_INITIALIZER_CANDIDATE';kind='MISSING'
  if selected is not None:
   assert selected==int(np.argmax([c['score']for c in pr['candidates']]))==base['selected_candidate_index']
   ca=pr['candidates'][selected];q8=ca['keypoints_xy'][:8];rawbox=ca.get('box_xyxy')
   if rawbox is not None:
    if valid_box(rawbox):box=rawbox;kind='ORIGINAL_INITIALIZER_SELECTED_BOX';reason=None
    else:reason='INVALID_ORIGINAL_SELECTED_BOX_NO_SILENT_REPLACEMENT'
   else:
    box=query_extent(q8);kind='FROZEN_QUERY8_EXTENT_FALLBACK'if box else'MISSING';reason=None if box else'NO_USABLE_ORIGINAL_QUERY_EXTENT'
  d=dict(context_id=key,source=base['source'],original_input_id=base['input_id'],raw_hw=base['raw_hw'],target_box_xyxy=box,box_origin=kind,missing_reason=reason,original_query8_xy=q8,original_selected_candidate_index=selected,box_fixed_across_all_occurrence_variants=True,GT_box_used=False,initializer_not_rerun=True)
  byid[key]=d;defs.append(d)
 else:d=byid[key]
 k=base['native_slot_logical_only'];expected=d['original_query8_xy'][k]
 assert expected==base['query_xy']or(expected is None and base['query_xy']is None),(i,expected,base['query_xy'])
for r in qnew:
 key='V99:'+r['base_id'];box=query_extent(r['query8_xy']);assert box is not None
 d=dict(context_id=key,source='V99_SYNTH',original_input_id=r['base_id'],raw_hw=r['raw_hw'],target_box_xyxy=box,box_origin='FROZEN_QUERY8_EXTENT_FALLBACK',missing_reason=None,original_query8_xy=r['query8_xy'],original_selected_candidate_index=None,box_fixed_across_all_occurrence_variants=True,GT_box_used=False,source_query_emulated_not_initializer=True)
 byid[key]=d;defs.append(d)
occ=[]
for i,r in enumerate(old):
 base=bases[r['base_row_index']];d=byid[base['source']+':'+base['input_id']];assert r['query_xy']==base['query_xy']and r['L']==base['L']and r['occurrence_index']==i
 binding=r['edited_fullRGB_file']if r['full_rgb_changed']else base['fullRGB_file_binding']
 occ.append(dict(global_occurrence_index=i,domain='V96_REAL',occurrence_id=r['occurrence_id'],base_row_index=r['base_row_index'],native_slot_logical_only=r['native_slot_logical_only'],variant=r['variant'],donor_rank=r['donor_rank'],source=r['source'],raw_split=r['raw_split'],query_xy=r['query_xy'],L=r['L'],original_input_valid=r['original_input_valid'],operator_query_available=oldinput[i]['input_facts']['operator_query_available'],context_id=d['context_id'],target_context_available=d['target_box_xyxy']is not None,full_RGB=binding,full_RGB_format=r['edited_fullRGB_format']if r['full_rgb_changed']else base['fullRGB_format'],decoded_RGB_sha256=r['full_rgb_decoded_sha256'],raw_hw=r['raw_hw'],full_RGB_changed=r['full_rgb_changed']))
for j,r in enumerate(new):
 im=images[r['image_index']];q=qnew[r['base_index']];assert r['occurrence_index']==j and r['image_id']==im['image_id']and r['variant']==im['variant']and r['query_xy']==q['query8_xy'][r['aux_corner_index']]and r['L']==q['L'][r['aux_corner_index']]
 d=byid['V99:'+r['base_id']]
 occ.append(dict(global_occurrence_index=31680+j,domain='V99_SYNTH',occurrence_id=r['image_id']+'::AUX'+str(r['aux_corner_index']),base_row_index=r['base_index'],native_slot_logical_only=r['aux_corner_index'],variant=r['variant'],donor_rank=None,source='SYNTH_EXPECTED_CUE',raw_split=r['split'],query_xy=r['query_xy'],L=r['L'],original_input_valid=r['input_facts']['query_center_valid'],operator_query_available=r['input_facts']['operator_query_available'],context_id=d['context_id'],target_context_available=True,full_RGB=im['image'],full_RGB_format='PNG_RGB',decoded_RGB_sha256=im['decoded_RGB_sha256'],raw_hw=im['raw_hw'],full_RGB_changed=r['variant']!='BEFORE'))
assert len(occ)==33984 and len(defs)==216
unique={r['full_RGB']['path']for r in occ};assert all(Path(p).is_file()for p in unique)
# Generated-only crop/grid tests, no actual RGB body access.
a=np.arange(6*8*3,dtype=np.uint8).reshape(6,8,3);out,valid=sample_target_context(a,[0,0,8,6],(3,4));assert out.shape==(3,4,3)and valid.all()
g=target_context_grid([0,0,8,6],(3,4));assert np.array_equal(g[0,0],[1,1])and np.array_equal(g[-1,-1],[7,5])
_,v=sample_target_context(a,[-2,-2,2,2],(4,4));assert(~v).any()and v.any()
assert query_extent([[1,1]]*8)is None
# Size and geometric q-in-box diagnostics only; not visibility or ownership.
for src in('LOCO','PALLOX','V99_SYNTH'):
 ds=[d for d in defs if d['source']==src and d['target_box_xyxy']is not None]
 sizes=np.array([[d['target_box_xyxy'][2]-d['target_box_xyxy'][0],d['target_box_xyxy'][3]-d['target_box_xyxy'][1]]for d in ds])
 boxstats[src]=dict(contexts=len(ds),min_wh=sizes.min(0).tolist(),median_wh=np.median(sizes,axis=0).tolist(),max_wh=sizes.max(0).tolist())
paths=dict(definitions=V/'TARGET_CONTEXT_DEFINITIONS216_INPUT_ONLY.jsonl',occurrences=V/'OCCURRENCES33984_TARGET_CONTEXT_INPUT_ONLY.jsonl')
jl(paths['definitions'],defs);jl(paths['occurrences'],occ)
summary=dict(completed=True,PASS_metadata_join=True,scope='Metadata/code only: zero source RGB/pixel/feature/model/checkpoint/GT-body/private/current reads; no crop production, inference, fit, or PnP.',inputs={k:bind(p)for k,p in PATHS.items()},outputs={k:bind(p)for k,p in paths.items()},counts=dict(total_occurrences=len(occ),context_definitions=len(defs),exact_full_RGB_paths_available=len(occ),unique_full_RGB_file_paths=len(unique),with_input_derived_target_box=sum(r['target_context_available']for r in occ),missing_target_box=sum(not r['target_context_available']for r in occ),context_origins=dict(Counter(d['box_origin']for d in defs)),occurrence_context_origins=dict(Counter(byid[r['context_id']]['box_origin']for r in occ)),by_source={src:dict(rows=sum(r['source']==src for r in occ),missing=sum(r['source']==src and not r['target_context_available']for r in occ))for src in sorted({r['source']for r in occ})}),box_dimensions_native_px=boxstats,plan=dict(purpose='Prospective target-relative RGB context, not a target identity or correctness certificate.',box_choice='Exact original first-argmax selected initializer bbox when supplied and finite positive. Only if absent: finite original q8 min/max, no centroid, no GT. Invalid supplied bbox remains missing.',same_box_and_q_L='One context definition per original selected image/instance or synthetic base, reused exactly for every before/cover/sham and donor rank.',full_RGB='Read the exact corresponding fullRGB binding, including edited fullRGB; do not combine an edited local crop with an original whole-target context.',proposed_sampling='224x224 direct rectangular native bilinear grid lo+(j+.5)*span/M, integer native pixel-centres; no expansion, recenter, clipping, or aspect-preserving promise. Zero-neighbor padding; retain explicit valid mask. Future protocol must freeze this exact prospective recipe before feature extraction.',missing='Keep all rows; no-detection/no usable context remains missing, never a negative label. Original q availability is preserved independently.',no_feature_or_architecture_choice='This inventory does not select an encoder, head, threshold, loss, fit, or gate.'),generated_math_tests=dict(PASS=True,generated_arrays_only=True,grid_native_positions=True,unavailable_padding_mask=True,degenerate_extent_rejected=True),limits=['Initializer-selected boxes may identify a wrong target or a small part; they are not certified whole-target boxes. Their actual dimensions are reported rather than upgraded to object ownership.','V99 uses fixed synthetic projected-AABB-plus-jitter query extents, not actual initializer boxes. This input-source difference remains explicit.','A target crop can itself contain an occluder or multiple objects; context alone does not resolve hidden expected-cue or target identity.','No GT cue/reference/native binding/mask/pose selects the box. Native correspondence and human gold remain UNKNOWN/false.','C8 and earlier candidate artifacts/thresholds/gates are unchanged. This is representation-readiness inventory, not admission or evidence of causal improvement.'])
write(V/'RESULT.json',summary)
print(json.dumps(dict(result=bind(V/'RESULT.json'),counts=summary['counts'],box_dimensions=boxstats)))
