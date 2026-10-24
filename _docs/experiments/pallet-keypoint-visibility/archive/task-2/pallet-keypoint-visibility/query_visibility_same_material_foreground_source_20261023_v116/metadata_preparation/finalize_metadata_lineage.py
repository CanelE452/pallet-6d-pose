"""Metadata-only old-VAL SHA certificate and BEFORE feature/context readiness."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).resolve().parent;P=H.parent.parent
R=P/'query_visibility_query_anchored_cue_20261023_v96';C=P/'query_visibility_spatial_target_query_features_20261023_v111';T=P/'query_visibility_target_relative_rgb_head_20261023_v108'
def lines(p):return [json.loads(s)for s in p.read_text().splitlines()]
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':Path(p).stat().st_size}
def write(n,d,jsonl=False):
 p=H/n
 with p.open('x') as f:
  if jsonl:
   for r in d:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
  else:f.write(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 return bind(p)
def main():
 core=json.loads((H/'SEAL.json').read_text());bases=lines(H/'BASES28_INPUT_ONLY.jsonl');don=lines(H/'SELECTED16_DONORS_METADATA_ONLY.jsonl')
 for b in core['outputs'].values():assert bind(Path(b['path']))['sha256']==b['sha256']
 AP=R/'query_anchored_pair_preparation/DONOR_ASSIGNMENT960_K16_BEFORE_PIXELS.jsonl';a=lines(AP);oldval=sorted({d['image_sha256']for r in a for d in r['ranks']if d['split']=='VALIDATION'})
 assert len(oldval)==200 and not({d['image_sha256']for d in don}&set(oldval))
 oldbinding=write('OLD_VALIDATION_DONOR_RGB_SHA256_METADATA_ONLY.json',{'completed':True,'scope':'Unique stored RGB SHA256 values from all old locked VALIDATION donor assignments; metadata only, no payload hashes recomputed.','source':bind(AP),'unique_RGB_count':200,'RGB_SHA256':oldval,'all_selected16_disjoint':True})
 F={'V96_occurrence_lineage':R/'actual_source_pair_input_pack/ALL31680_INPUT_LINEAGE.jsonl','V96_unit_lineage':R/'actual_source_pair_input_pack/INPUT_UNITS_LINEAGE.jsonl','C9_original_contexts3652':T/'context_feature_preparation/UNIT_CONTEXTS3652_INPUT_ONLY.jsonl','C11_RGB_refs3652':C/'feature_preparation/OLD3652_RGB_UNIT_REFS_INPUT_ONLY.jsonl','C11_RGB_spatial_facts':C/'spatial_query_features/INPUT_RGB_AND_SPATIAL_FACTS_INPUT_ONLY.jsonl','C11_feature_receipt':C/'spatial_query_features/FEATURE_COMPLETE.json','C11_feature_protocol':C/'feature_preparation/FEATURE_PROTOCOL_BEFORE_SOURCE_FORWARD.json','context_math':P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py','local_math':P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'}
 binding={k:bind(p)for k,p in F.items()};occ=lines(F['V96_occurrence_lineage']);unit=lines(F['V96_unit_lineage']);ctx=lines(F['C9_original_contexts3652']);refs=lines(F['C11_RGB_refs3652']);facts=lines(F['C11_RGB_spatial_facts']);receipt=json.loads(F['C11_feature_receipt'].read_text());prot=json.loads(F['C11_feature_protocol'].read_text())
 assert receipt['completed'] and receipt['PASS'] and receipt['all_C10_mapping_availability_exact']
 assert prot['old_contexts']['sha256']==binding['C9_original_contexts3652']['sha256'] and prot['old_RGB_refs']['sha256']==binding['C11_RGB_refs3652']['sha256']
 out=[]
 for b in bases:
  j=b['base_index'];i=b['original_base_row_index'];r=b['original_row'];o=occ[33*i];u=o['input_unit_index'];c=ctx[u];f=facts[u];ref=refs[u]
  assert o['occurrence_index']==33*i and o['base_row_index']==i and o['variant']=='BEFORE' and u==i
  assert o['query_xy']==r['query_xy']==c['query_xy'] and o['L']==r['L']==c['L']
  assert c['input_unit_index']==f['input_unit_index']==ref['input_unit_index']==unit[u]['input_unit_index']==u
  assert c['full_RGB']['sha256']==b['source_lineage']['fullRGB_file_binding']['sha256'] and c['decoded_RGB_sha256']==b['source_lineage']['before_decoded_sha256']
  assert c['box_xyxy']==r['selected_box_xyxy']
  assert f['RGB96_sha256']==ref['rgb96_sha256'] and f['valid96_sha256']==ref['valid96_sha256']
  out.append({'base_index':j,'original_base_row_index':i,'V116_BEFORE_occurrence_index':33*j,'original_V96_C11_global_occurrence_index':33*i,'original_occurrence_id':o['occurrence_id'],'C11_spatial_feature_input_unit_index':u,'input_id':r['input_id'],'native_slot_logical_only':r['native_slot_logical_only'],'query_xy':r['query_xy'],'L':r['L'],'source':b['source'],'raw_split':b['raw_split'],'original_context_id':c['context_id'],'frozen_context_box_xyxy':c['box_xyxy'],'context_box_origin':'Exact original initializer selected_box_xyxy in old source input, matched against saved C9 UNIT_CONTEXTS; no expansion/recenter or GT box.','context_full_RGB':c['full_RGB'],'context_full_RGB_format':c['full_RGB_format'],'context_decoded_RGB_sha256':c['decoded_RGB_sha256'],'C11_feature_packet_recorded_binding':receipt['output'],'C11_feature_values_read_here':False,'BEFORE_RGB96_ref':ref['rgb96_ref'],'BEFORE_valid96_ref':ref['valid96_ref'],'saved_RGB96_sha256':f['RGB96_sha256'],'saved_valid96_sha256':f['valid96_sha256'],'saved_target_RGB224_float32_sha256':f['target_RGB224_float32_sha256'],'saved_Q_inside':f['Q_inside'],'original_input_facts':o['input_facts'],'changed_COVER_SHAM_rule':'Keep this exact q/L/context box and operators, but decode the actual new fullRGB and freshly compute context224 and matching T/Q (and all changed local descriptors). Never pair changed composite pixels with BEFORE target-context features. Exact unchanged fullRGB identity may reuse only under a later bound feature entry.'})
 assert len(out)==28
 line=write('BEFORE28_C11_FEATURE_CONTEXT_READINESS_INPUT_ONLY.jsonl',out,True)
 appendix={'completed':True,'scope':'Saved input metadata/protocol/receipt joins only; no feature/score/RGB/array bodies opened.','rows':28,'same_q_L_and_initial_context_box':True,'C11_feature_dimensions':1928,'original_units':[x['C11_spatial_feature_input_unit_index']for x in out],'original_global_occurrence_indices':[x['original_V96_C11_global_occurrence_index']for x in out],'global_occurrence_definition':'Original C11 common source order keeps old V96 global index33*original_base_row_index; unit mapping inherited unchanged from V96 through C10 as certified by C11 receipt. This is metadata/certificate reconciliation, not an independent array-mapping read.','context_operator':'Pinned V107rect224 unexpanded original selected bbox, exact same native pixel-centre bilinear/mean-fill pipeline; no new boxes chosen.','changed_fullRGB_requires_fresh_target_context':True,'BEFORE_reference_or_reuse_is_not_permission_for_changed_context_reuse':True,'no_actual_feature_execution_or_training_policy':True,'bindings':binding,'readiness_rows':line,'code':bind(Path(__file__))}
 app=write('FEATURE_CONTEXT_READINESS_APPENDIX.json',appendix)
 final={'completed':True,'PASS':True,'scope':'Exact root policy metadata16selection/28base/448assignment/924inputplan plus oldVAL SHA and metadata-only C11 context readiness. No payloads/model/fit.','core_seal':bind(H/'SEAL.json'),'receipt':core['receipt'],'policy':core['policy'],'outputs':core['outputs'],'old_VALIDATION_donor_RGB_SHA256':oldbinding,'feature_context_readiness':app,'feature_context_rows':line,'training_admission':False,'native_mapping':'UNKNOWN','human_gold':False,'new_root_entry_or_model_or_pixels_or_arrays_or_fit_or_current_V112':0,'created_UTC':datetime.now(timezone.utc).isoformat()}
 fin=write('METADATA_SELECTION_SEAL.json',final)
 print(json.dumps({'final_seal':fin,'oldVAL':oldbinding,'appendix':app,'context_rows':line}))
if __name__=='__main__':main()
