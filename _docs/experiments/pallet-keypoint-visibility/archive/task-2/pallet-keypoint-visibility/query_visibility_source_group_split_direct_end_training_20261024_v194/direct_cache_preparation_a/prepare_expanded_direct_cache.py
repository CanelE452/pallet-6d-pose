"""Preparation only: bind code/text and existing completion pointers, never arrays."""
import ast, json, sys
import direct_cache_common as c
H,V,P=c.H,c.V,c.P
CODE=H/'build_expanded_direct_control_cache.py'
TEST=H/'test_invented_expanded_cache.py'
CHECKS=H/'GENERATED_INVENTED_EXPANDED_CACHE_CHECKS.json'
PROTOCOL=H/'PROTOCOL_BEFORE_EXPANDED_SOURCE_CONTROL_CACHE_BODIES.json'
PINNED='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'

def prepare():
    scope=c.read(c.SCOPE)
    prior_phase=P/'query_visibility_direct_end_paired_source_training_20261024_v190/ROOT_PHASE_RESULT.json'
    phase=c.read(prior_phase);old_binding=phase['source_input_cache'];old=c.read_bound(old_binding)
    c.need(old['completed'] is True and old['PASS'] is True,'Completed old cache metadata only')
    old_protocol=c.read_bound(old['protocol']);sam=c.read(c.SAM_PROTOCOL)
    c.need(old['SAM_checkpoint']==old_protocol['checkpoint']==sam['checkpoint'],'Inherited checkpoint pointer exact; no checkpoint read')
    checks=c.read(CHECKS);c.need(checks['completed'] is True and checks['PASS'] is True and checks['count']==12,'Invented checks completed')
    for file in (CODE,TEST,H/'direct_cache_common.py',c.PIXEL_HELPER,c.SAM_HELPER):
        with c.opened(file) as f:ast.parse(f.read())
    sources={
      'scope':c.bind(c.SCOPE),'common_code':c.bind(H/'direct_cache_common.py'),
      'old_cache_code':old['code'],'old_cache_protocol':old['protocol'],
      'old_common_code':old_protocol['source_bindings']['common_code'],
      'original_pixel_helper':old_protocol['source_bindings']['original_pixel_helper'],
      'original_SAM_helper':old_protocol['source_bindings']['original_SAM_helper'],
      'original_SAM_protocol':old_protocol['source_bindings']['original_SAM_protocol'],
      'SAM_source_pin':old_protocol['source_bindings']['SAM_source_pin']}
    for b in sources.values():c.checked(b)
    protocol=dict(completed=True,prepared_only=True,cache_code=c.bind(CODE),prepare_code=c.bind(__file__),test_code=c.bind(TEST),checks=c.bind(CHECKS),source_bindings=sources,
      root_scope=c.bind(c.SCOPE),old_cache_discovered_via_source_phase=c.bind(prior_phase),old_cache_complete=old_binding,
      source_controls={'SYNTH_FINITE_END':next(s['complete'] for s in old['source_controls'] if s['domain']=='SYNTH_FINITE_END'),'REAL_SEMANTIC_END_WEAK':scope['new_source_generation']},
      checkpoint=sam['checkpoint'],runtime=sam['runtime'],pinned_python=PINNED,output_directory=str(V/'direct_control_raw_sam_cache_v1'),
      population='V189 all1024 synthetic attempts unchanged, V193 all66 real attempts replaces old V189 real66 whole. All3270 variant statuses retained, all physically saved groups x8 retained; missing variants never fabricated. No split filtering in cache.',
      admission='Separate root admission binds each complete/census/independent source audit and original21-frame plan. Only complete verified triplets may receive 0/1/0 at original selected available native slot; other7/failure/UNKNOWN=-1. Old native/quality/owner remain -1. Neither broad support nor q-owner becomes direct-end truth.',
      identity=dict(canonical_variants=['BEFORE','COVER','SHAM'],real_packet_SHAM='ADJACENT_SHAM',group_id='SYNTH__attempt__variant or REAL__pair__variant',row_id='group_id/nativeN',control_join_key=['domain','attempt_id','canonical_variant','native_slot'],source_frame_identity='Exact original_real_frame_plan lookup; never originalglobal//8',original_metadata='All8 q float64/L/native/occurrence/global/availability/flags checked against original source rows/plan; repeated source global IDs allowed'),
      targets='Separate direct_end_external_target int8[N] and authority unicode[N], with META6; target-free INPUTS10 rgb96uint8/valid96bool/SAM96bool/available96bool+META6. RawRGB96/valid96 copied byteexact from original physical packet. No native/quality/owner labels in input.',
      SAM='Exact original official SAM1 ViT-B FP32 pinned code/checkpoint/runtime, all8 finite F64 q minmax box without clip/expand, inherited warn-finally/cumsum guard, strict image_logits>0, first new forward exact repeat, statehash unchanged. sample_sam is exact bound original V181 pure AST at original q64/L. InputU remains factual original availability.',
      reuse='Old cache full SAM RAW4 only after exact fullRGB decoded bytes SHA/shape/F64 box plus identical pinned source/checkpoint/full model state. Validate old exact12 packet/metadata/strictzero; copy raw4 into fresh metadata packet and reload saved bytes exactly. ChangedRGB/box must forward. No external SAM96 or old metadata alias; exact within-run raw4 reuse recorded. No old trained head/features used.',
      source_array_whitelist='Actual root entry must list exactly current admitted source packet bindings plus old SAM group packet bindings from old COMPLETE.SAM_groups. Compare finite set before NPZ header/body. Checkpoint separately exact inherited binding. Preparation reads neither census/source NPZ/SAM packet/checkpoint header/body nor their fresh asset SHA.',
      split='All rows retained. B component sidecars applied later by trainer through exact control_join_key+META6; cache does not apply TRAIN/HOLDOUT masks or change old3-domain caches.',
      completion_schema={'completed':'bool','PASS':'bool','source_only':True,'domain':'DIRECT_CONTROL','rows':'all physical groups*8','groups':'all physical groups','inputs':'binding INPUTS10','targets':'binding direct target+authority+META6','metadata':'binding ALL_CONSTRUCTED_ROWS.jsonl incl cache_index/control_join_key/source lineage','census':'binding ALL3270_VARIANT_CENSUS.jsonl','SAM_groups':'binding ALL_SAM_GROUPS.jsonl','admission':'root separate binding','real_source_replaces_old_whole':True,'old_cache_reference':'binding','component_role_masks_applied_to_training':False},
      actual_source_array_header_weight_SAM_model_fit_eval_calls=0,automatic_retry=False,selected_model=None,fully_solved=False)
    c.write(PROTOCOL,protocol)
    false=dict(approved_direct_control_cache=False,source_only=True,code=c.bind(CODE),protocol=c.bind(PROTOCOL),root_scope={'path':'ROOT_FILL_EXACT_ACTUAL_SOURCE_CACHE_SCOPE','sha256':'ROOT_FILL'},admission={'path':'ROOT_FILL_SEPARATE_ROOT_ADMISSION','sha256':'ROOT_FILL'},checkpoint=sam['checkpoint'],array_whitelist=[],output_directory=protocol['output_directory'],note='False template cannot execute; root must bind all source/proof/oldSAM packet whitelist and source admission before actual entry.')
    template=H/'ROOT_ACTUAL_EXPANDED_DIRECT_CACHE_ENTRY_TEMPLATE_APPROVED_FALSE.json';c.write(template,false)
    c.write(H/'OUTPUT_SCHEMA.json',dict(completed=True,prepared_only=True,completion=protocol['completion_schema'],input_keys=list(c.INPUTS),target_keys=['direct_end_external_target','authority']+list(c.META),META6=list(c.META),rows_extra_fields=['cache_index','domain','group_id','attempt_id','variant','source_variant','control_join_key','authority','source_packet','source_status','source_base_index','source_frame_index','source_case_id','source_donor_id','source_proof_binding','root_admitted','raw_RGB96_sha256','raw_valid96_sha256'],all_original_flags_retained=True,source_status_census_rows=3270,split_filtering=False))
    ready=dict(completed=True,PASS=True,prepared_only=True,FINAL_READY_FROZEN=True,code=c.bind(CODE),common=c.bind(H/'direct_cache_common.py'),protocol=c.bind(PROTOCOL),checks=c.bind(CHECKS),test_code=c.bind(TEST),prepare_code=c.bind(__file__),false_entry=c.bind(template),output_schema=c.bind(H/'OUTPUT_SCHEMA.json'),old_cache_metadata_only=old_binding,root_admission_and_actual_cache_scope_pending=True,source_binary_array_header_checkpoint_SAM_forward_model_fit_or_eval_calls=0,command=[PINNED,'-B',str(CODE),'--entry','ROOT_FILL_APPROVED_ENTRY_PATH'],output=protocol['output_directory'])
    c.write(H/'FINAL_READY.json',ready);print(json.dumps(ready,ensure_ascii=False),flush=True)

if __name__=='__main__':prepare()
