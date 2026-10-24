#!/usr/bin/env python3
"""One-time metadata-only fixed4 plan and pure geometry checks; no PNG access."""
import copy,importlib.util,json,os

P='/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility'
D=P+'/query_visibility_fixed_public_train_instance_part_GT_correspondence_20261025_v217'
H=D+'/fixed_query_crop_nominal_reference_visual_preparation_a'
spec=importlib.util.spec_from_file_location('fixed4_renderer',H+'/render_fixed4_query_reference_cards.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
INPUTS={
 'nominal_complete':{'path':D+'/nominal_candidate_query_projection_v1/COMPLETE.json','sha256':'fcd82b252b74c4f0843d9819fab02cff631eb9d13df778a315ed8dd0eef33982'},
 'nominal_rows120':{'path':D+'/nominal_candidate_query_projection_v1/ALL120_CANDIDATE_NATIVE_QUERY_DIAGNOSTICS.json','sha256':'e1c45b308b2a8263812043ce4bb78e4fff616506b3366bdbc174b7fa6d021e2e'},
 'pixel_complete':{'path':D+'/fixed12_mask_depth_query_diagnostics_v2/COMPLETE.json','sha256':'407cff2c84fda8aedf54aa6f34c65b4b295eda5c2f867cec698d3b3cf1929453'},
 'pixel_rows':{'path':D+'/fixed12_mask_depth_query_diagnostics_v2/ALL_DIAGNOSTIC_ROWS.json','sha256':'4708c8958e43436ff236807e75e6e7a407016bb245f0959fb97b0ce1f18585a3'},
 'original_V216_join':{'path':P+'/query_visibility_public_train_part_query_support_contract_20261025_v216/joined_part_query_support_v2/ALL96_JOINED_CANDIDATES.json','sha256':'ffebab4da32966ec1ae5299e6ecdd3a8ba66aaad7b03388c3ef2c52020f60675'}
}
def newj(name,x):
 z=(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();p=H+'/'+name
 try:
  with os.fdopen(m.opened(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'wb') as f:f.write(z)
 except FileExistsError:
  m.need(name=='FIXED4_CASES_AND_EXACT27_PNG_BINDINGS_BEFORE_PIXELS.json' and m.raw(p)==z,'Only exact preserved pre-check plan may already exist')
 return m.binding(p,z)
def bind(name):return m.binding(H+'/'+name,m.raw(H+'/'+name))
def reject(call):
 try:call()
 except (ValueError,OSError):return True
 raise AssertionError('Expected explicit rejection')
def main():
 x={k:json.loads(m.checked(v)) for k,v in INPUTS.items()}
 m.need(x['nominal_complete']['completed'] and x['nominal_complete']['PASS'] and x['pixel_complete']['PASS'],'Completed generated metadata')
 m.need(x['nominal_complete']['artifacts']['ALL120_CANDIDATE_NATIVE_QUERY_DIAGNOSTICS.json']==INPUTS['nominal_rows120'],'Nominal closure')
 m.need(x['pixel_complete']['all_rows']['path']==INPUTS['pixel_rows']['path'] and x['pixel_complete']['all_rows']['sha256']==INPUTS['pixel_rows']['sha256'],'Pixel closure')
 prs=x['nominal_rows120']['rows120'];pix=x['pixel_rows']['candidate_rows15'];jrs=x['original_V216_join']['rows96']
 m.need(len(prs)==120 and len(pix)==15 and len(jrs)==96,'No population change')
 cases=[];unique={}
 for frame,candidate in m.CASE_KEYS:
  selected=[r for r in prs if r['original_input']['frame_position']==frame and r['diagnostic_candidate_gt_list_index']==candidate]
  selected.sort(key=lambda r:r['native_slot']);m.need([r['native_slot'] for r in selected]==list(range(8)),'Fixed candidate all8 once')
  px=[r for r in pix if r['frame_position']==frame and r['candidate_gt_list_index']==candidate];m.need(len(px)==1,'Exact saved candidate dimensions')
  ctx=[r['image'] for r in x['pixel_complete']['original_context12'] if r['frame_position']==frame];m.need(len(ctx)==1,'Exact owned context')
  rows=[]
  for r in selected:
   o=r['original_input'];j=[v for v in jrs if v['input']['position']==frame and v['input']['native_slot']==r['native_slot']];m.need(len(j)==1,'Exact V216 original join')
   ji=j[0]['input']
   for k in ('frame_key','original_frame_index','component_id','source_group_key','internal_source_role','row_id','original_global_index','native_slot','query_xy','L','original_query_available','original_flags'):
    m.need(o[k]==ji[k],'Original identity field '+k)
   m.need(o['original_role']==ji['role']=='TRAIN' and r['selected_gt_instance_index'] is None and r['training_admission'] is False,'Role and unselected candidate')
   clean=j[0]['display_bindings']['clean96'];m.need(ctx[0]['sha256']==j[0]['display_bindings']['full_context'][0]['sha256'],'Owned context is original saved context')
   local=r['query_diagnostic']['RAW96_reference_address']
   if o['original_query_available'] and local is not None:m.need(local==m.crop_reference(r['nominal_reference_xy'],o['query_xy'],o['L']),'Saved reference address exact F64 algebra')
   if not o['original_query_available']:m.need(local is None,'U not repaired by finite nominal geometry')
   rows.append({**{k:copy.deepcopy(o[k]) for k in ('row_id','original_global_index','frame_key','component_id','internal_source_role','native_slot','query_xy','L','original_query_available','original_flags')},'nominal_reference_xy':r['nominal_reference_xy'],'nominal_reference_raw96_xy':local,'saved_query_diagnostic_status':r['query_diagnostic']['status'],'canonical_index':r['canonical_index'],'nominal_mapping_status':r['nominal_mapping_status'],'clean96':clean,'physical_native_gold':'UNKNOWN','training_admission':False})
   unique[clean['path']]=clean
  unique[ctx[0]['path']]=ctx[0]
  cases.append({'case_key':[frame,candidate],'frame_position':frame,'candidate_gt_list_index':candidate,'source_hw':[px[0]['height'],px[0]['width']],'context':ctx[0],'rows8':rows,'selected_gt_instance_index':None,'training_admission':False})
 plan={'completed':True,'prepared_before_PNG_reads':True,'selection_basis':'Root fixed frames0/6/10 all candidates; no result/nearest/GT-best selection','cases4':cases,'unique_PNG_bindings27':list(unique.values()),'input_metadata_bindings':INPUTS,'PNG_stat_hash_read_decode_during_preparation':0,'training_admission':False}
 m.validate_plan(plan);pb=newj('FIXED4_CASES_AND_EXACT27_PNG_BINDINGS_BEFORE_PIXELS.json',plan)
 checks=[]
 def test(name,v):m.need(v,name);checks.append(name)
 test('Reference_equals_q_is47_5',m.crop_reference([100.,200.],[100.,200.],192.)==[47.5,47.5])
 test('F64_original_qL_signed_offsets',m.crop_reference([120.,180.],[100.,200.],192.)==[57.5,37.5])
 test('Invalid_or_missing_scalars_remain_UNKNOWN',all(m.crop_reference(a,b,c) is None for a,b,c in [(None,[0,0],1),([0,0],None,1),([0,0],[0,0],None),([0,0],[0,0],0),([0,0],[0,0],-1)]))
 test('Continuous_canvas_halfopen',m.in_canvas([-.5,95.499],96,96) and not m.in_canvas([95.5,0],96,96))
 test('Nearest_display_pixelcenters_axis_scales',m.display_coordinate([47.5,47.5],(2,3),(10,20))==[105.5,163.5])
 ref=[200.,47.5];before=copy.deepcopy(ref);arrow=m.boundary_arrow(ref,96,96)
 test('Outside_arrow_displayonly_original_immutable',ref==before and arrow==[90.,47.5] and not m.in_canvas(ref,96,96))
 test('Unknown_reference_no_arrow',m.boundary_arrow(None,96,96) is None)
 bad=copy.deepcopy(plan);bad['cases4'][0]['case_key']=[0,99]
 test('No_candidate_or_query_selection_substitution',reject(lambda:m.validate_plan(bad)))
 bad=copy.deepcopy(plan);bad['cases4'][0]['rows8'][0]['nominal_reference_raw96_xy']=[47.5,47.5]
 test('Finite_q_does_not_promote_original_U',reject(lambda:m.validate_plan(bad)))
 bad=copy.deepcopy(plan);bad['unique_PNG_bindings27'][0]={**bad['unique_PNG_bindings27'][0],'sha256':'0'*64}
 test('PNG_binding_annotation_cannot_replace_exact_identity',reject(lambda:m.validate_plan(bad)))
 test('Lexical_forbidden_rejected_before_FS',reject(lambda:m.opened('/V112/does-not-exist')))
 test('Original_3frames24clean3contexts4cards_counts',len(unique)==27 and len({r['clean96']['path'] for c in cases for r in c['rows8']})==24 and len({c['context']['path'] for c in cases})==3)
 cb=newj('INVENTED_GEOMETRY_AND_METADATA_CHECKS.json',{'completed':True,'PASS':True,'count':len(checks),'checks':checks,'actual_PNG_payload_access':0,'SOURCE_projection_calls':0})
 proto={'completed':True,'prepared_only':True,'purpose_KO':'고정4 원query/crop와 저장된 불확실 nominal 참조의 표시 전용 비교','fixed_case_keys':list(map(list,m.CASE_KEYS)),'input_metadata_bindings':INPUTS,'fixed_cases':pb,'exact_PNG_whitelist':27,'PNG_class_counts':{'original_clean96':24,'owned_original_context':3},'actual_approval_before_any_PNG_stat_hash_read_decode':True,'coordinate_contract':{'full_context':'original q cyan; saved nominal reference magenta, original native IDs; original source H/W from generated pixel diagnostics','crop':'Use original clean96 byte copies, q center47.5. Available saved reference address delta*(96.0/L)+47.5; original-U local reference remains null UNKNOWN','display_scale':'nearest enlarge only on new canvas; independent rounded x/y pixel-center scale','out_of_bounds':'coordinate listed; boundary arrow is display only, never recenter or clamp source coordinates','no_new_projection_permutation_or_RAW_sampler_replay':True},'all_candidate_and_instance_authority':'Diagnostic candidate only; selected GT instance null; physical native gold UNKNOWN; no D/E/S annotation or training admission','output_directory':m.OUT,'report_scope':'Fixed4 explanatory cards, not all15 view or representative source population','original_images_inputs_qL_U_native_roles_unchanged':True,'source_GT_mask_depth_CAD_RGBarray_NPZ_model_score_reads':0,'training_admission':False,'automatic_retry':False}
 pr=newj('PROTOCOL_BEFORE_FIXED4_PNG_BODIES.json',proto)
 code=bind('render_fixed4_query_reference_cards.py')
 e={'approved_actual_fixed4_query_reference_cards':False,'code':code,'protocol':pr,'fixed_cases':pb,'input_metadata_bindings':INPUTS,'output_directory':m.OUT,'training_admission':False,'automatic_retry':False}
 eb=newj('ROOT_ACTUAL_FIXED4_VISUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',e)
 ready=newj('FINAL_READY.json',{'completed':True,'PASS':True,'prepared_only':True,'code':code,'preparer':bind('prepare_fixed4_metadata_and_ready.py'),'protocol':pr,'fixed_cases':pb,'checks':cb,'false_entry':eb,'input_metadata_bindings':INPUTS,'planned_output_directory':m.OUT,'planned_card_names':['FRAME_%02d_CANDIDATE_%06d_QUERY_REFERENCE_CARD.png'%k for k in m.CASE_KEYS],'actual_PNG_or_source_payload_reads':0,'projection_model_training_calls':0,'frozen_no_further_edits':True})
 print(json.dumps({'completed':True,'PASS':True,'ready':ready,'code':code,'protocol':pr,'fixed_cases':pb,'checks':cb},ensure_ascii=False))
if __name__=='__main__':main()
