import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'query_visibility_matched_query_scalar_20261023_v94'
O=R/'conditional_source_fit_contract_preparation'
RULE='CLEAN_ALL_SAME_CERTIFIED_GROUP_KNOWNSET'
def bind(p):
 b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def write(p,x):
 assert not p.exists(),str(p);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def jl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
 inputpath=O/'ALL800_CONTRACT_ALTERNATIVE_DIAGNOSTICS_GT_ONLY.jsonl'
 crosspath=R/'pallox_actual_query_metadata_audit/ALL4832_QUERY_X_CUE_VIEW_AVAILABILITY_GT_ONLY.jsonl'
 framepath=P/'query_visibility_public_cue_grid_20261023_v89/inputs_metadata/FRAMES_INPUTS_ONLY.jsonl'
 write(O/'ROOT_REQUESTED_CLEAN_ALL_SAME_RULE_BEFORE_PROPOSAL_PACKET.json',dict(status='ROOT_REQUESTED_RULE_PROPOSAL_PENDING_FINAL_ADMISSION',rule=RULE,source_quantification=bind(O/'CONDITIONAL_SOURCE_FIT_CONTRACT_DRAFT_KO.json'),all800=bind(inputpath),cross4832=bind(crosspath),code=bind(Path(__file__)),
    exact_rule='q exists AND nativecontinuous AND nearest AND RGB-centre valid; nonempty FULL known cue set allD or allE; no locatedFULL-U; no separate definedPARTIAL competing group. Null/unlocated are kept as unknown conditioned-task limitations. No native/GTbest association.',
    target_codes={'D_CONDITIONAL':0,'E_CONDITIONAL':1,'U':-1},original800_retained=True,controlledV3_included=False,training_admission=False,current_forward_fit_PnP=0))
 rows=jl(inputpath);cross=jl(crosspath);frames={r['frame_input_id']:r for r in jl(framepath)}
 byq=defaultdict(list)
 for x in cross:byq[x['query_id']].append(x)
 out=[]
 for i,r in enumerate(rows):
  xs=byq[r['query_id']];f=frames[r['frame_input_id']]
  full=[x for x in xs if x['coverage']=='FULL' and x['teacher_label'] in [0,1]]
  states={x['teacher_label'] for x in full}
  expected=(r['input_metadata_available'] and bool(full) and len(states)==1 and not r['located_FULL_unknown_groups'] and not r['defined_PARTIAL_competing_groups'])
  state=r['variants'][RULE]['label_candidate'];known=state!='U';assert known==expected
  target=0 if state=='D_CONDITIONAL' else 1 if state=='E_CONDITIONAL' else -1
  if known:assert states=={target}
  record=dict(row_index=i,query_id=r['query_id'],frame_index=r['frame_index'],frame_input_id=r['frame_input_id'],pair_id=r['pair_id'],view=r['view'],split=r['split'],native_slot=r['native_slot'],original_q=r['original_q'],original_L=r['original_L'],image=f['image'],image_sha256=f['image_sha256'],raw_hw=f['raw_hw'],source_family=f['source_family'],seed_V84_reuse=f['seed_V84_reuse'],
   query_source='ACTUAL_FROZEN_INITIALIZER_FIRST_ARGMAX_NATIVE0_7',query_status=r['query_status'],input_metadata_available=r['input_metadata_available'],scalar_target_D0_E1_Uminus1=target,candidate_known=known,candidate_state=state,
   candidate_rule=RULE,known_FULL_original_count=r['original_full_known_count'],known_FULL_certified_group_count=r['certified_group_full_known_count'],known_FULL_cue_ids=r['full_known_cue_ids'],FULL_unknown_count=r['raw_counts']['FULL_unknown'],PARTIAL_known_count=r['raw_counts']['PARTIAL_known'],PARTIAL_unknown_count=r['raw_counts']['PARTIAL_unknown'],null_unlocated_count=r['raw_counts']['null_reference'],null_unlocated_certified_group_count=len(r['null_unlocated_groups']),
   located_FULL_unknown_groups=r['located_FULL_unknown_groups'],defined_PARTIAL_competing_groups=r['defined_PARTIAL_competing_groups'],null_unlocated_groups=r['null_unlocated_groups'],diagnostic_flags=r['diagnostic_flags'],
   source_GT_cue_view_indices=[x['cue_view_index'] for x in xs],retained_FULL_GT_cues=[{k:x[k] for k in ['cue_view_index','cue_id','corpus','teacher_label','teacher_state','cue_reference_xy_unchanged','cue_uncertainty_bbox_unchanged','certified_duplicate_group','duplicate_group_size','duplicate_weight']} for x in full],
   label_scope='CONDITIONED_RECORDED_FULL_KNOWN_PHYSICALCUE_SET_STATE_NOT_NATIVE_EXPECTEDPART_OR_CROP_GLOBAL_ABSENCE',native_expected_part_state='UNKNOWN',human_gold=False,official_gold=False,unknown_or_MISS_is_negative=False,training_admission=False,root_final_admission_pending=True,GT_only_do_not_forward=True))
  out.append(record)
 assert len(out)==800 and sum(r['candidate_known'] for r in out)==157
 expected={'TRAIN_LAB':Counter({0:56,1:9,-1:287}),'VAL_HALL':Counter({0:81,1:11,-1:356})}
 for s,c in expected.items():assert Counter(r['scalar_target_D0_E1_Uminus1'] for r in out if r['split']==s)==c
 p=O/'CLEAN_ALL_SAME_ALL800_PROPOSAL_GT_ONLY.jsonl';assert not p.exists();p.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in out))
 pairs=defaultdict(dict)
 for r in rows:pairs[r['pair_id']][(r['view'],r['native_slot'])]=(r['original_q'],r['original_L'])
 equal=[p for p,d in pairs.items() if all(d[('before',k)]==d[('after',k)] for k in range(8))]
 actual_equal=[p for p in equal if all(pairs[p][('before',k)][0] is not None for k in range(8))]
 stats=dict(status='GT_ONLY_PROPOSAL_PENDING_ROOT_FINAL_ADMISSION',rule=RULE,rows=800,candidate_known=157,counts_by_split={s:{'D0':c[0],'E1':c[1],'Uminus1':c[-1]} for s,c in expected.items()},input_metadata_available=sum(r['input_metadata_available'] for r in out),all800_q_L_ids_order_unchanged=True,
  null_unlocated_retained_all800=dict(rows_with_null=sum(r['null_unlocated_count']>0 for r in out),total_raw_null_slot_combinations=sum(r['null_unlocated_count'] for r in out),known_rows_with_null=sum(r['candidate_known'] and r['null_unlocated_count']>0 for r in out)),
  source_GT_bindings=dict(variant_ledger=bind(inputpath),cross4832=bind(crosspath),frames100=bind(framepath),teacher_contract=bind(P/'query_visibility_physical_cue_aux_20261023_v91/teacher_preparation/TEACHER_CONTRACT_BEFORE_BUILD.json')),
  original_before_after_query_pair_clarification=dict(equal_serialized_pairs=equal,actual_nonnull_sameqL_pairs=actual_equal,note='The one equal serialized pair in the earlier draft is all-null before/after; it is NOT a same-query valid natural contrast. Actual nonnull shared pairs0.'),
  boundary_and_source_limits='696 metadata-available, no actual input tensor QA performed here. One bin/environment split. Null slots remain UNKNOWN outside conditioned known set. No native binding/crop-global absence claim.',controlledV3_included=False,new_pixels_arrays_weights_models_current_forward_fit_PnP=0,training_admission=False)
 write(O/'CLEAN_ALL_SAME_ALL800_PROPOSAL_SUMMARY_KO.json',stats)
 write(O/'CLEAN_ALL_SAME_PROPOSAL_SEAL.json',dict(completed=True,PASS_metadata_quantification=True,status='PROPOSAL_NOT_TRAINING_ADMISSION',rule_lock=bind(O/'ROOT_REQUESTED_CLEAN_ALL_SAME_RULE_BEFORE_PROPOSAL_PACKET.json'),ledger=bind(p),summary=bind(O/'CLEAN_ALL_SAME_ALL800_PROPOSAL_SUMMARY_KO.json'),code=bind(Path(__file__)),original_modified=0,training_admission=False))
 print(json.dumps(dict(ledger=bind(p),summary=bind(O/'CLEAN_ALL_SAME_ALL800_PROPOSAL_SUMMARY_KO.json'),seal=bind(O/'CLEAN_ALL_SAME_PROPOSAL_SEAL.json'),counts=stats['counts_by_split']),ensure_ascii=False))

if __name__=='__main__':main()
