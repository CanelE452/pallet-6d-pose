"""V161 JSON-only derived readout. No image/NPZ/model/feature imports or reads.

Prepare is limited to frozen schema/code/assignment metadata. Extract requires
root's completion trigger; failed attempts and untested checks remain explicit.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json

H=Path(__file__).resolve().parent
V=H.parent
P=V.parent
DESIGN=V/'ROOT_SOURCE_GENERATION_DESIGN_BEFORE_PIXELS.json'
PROTOCOL=V/'pixel_builder_preparation/PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json'
ASSIGN=V/'metadata_assignment/ASSIGNMENTS159_BEFORE_PIXELS.jsonl'
SELECT=V/'metadata_assignment/FIXED12_VISUAL_SELECTION_BEFORE_PIXELS.json'
SOURCE=V/'source_pixels_v1'
VARIANTS=('BEFORE','COVER','ADJACENT_SHAM')
BEFORE_KEYS=('all8_RGB96_equal_saved_packet','all8_valid96_equal_saved_packet',
 'all8_construct_equal_saved_packet','saved_global_indices_equal','all8_RGB96_fact_hash',
 'all8_valid96_fact_hash','target224_original_hash','target_valid_original_hash',
 'all8_fractions_original_exact','selected_RGB96_anchor_hash',
 'all8_center4_original_equal','all8_context_original_equal')
UNCHANGED_KEYS=tuple(label+'_'+key+'_unchanged' for label in ('cover','sham')
 for key in ('valid96','target_valid','crop_constructed','native_continuous_valid',
 'nearest_valid','central4_available','context_available','fractions','query_xy','L','native_slot',
 'original_no_detection','original_finite_non_sentinel',
 'original_query_RGB96_centre_domain_valid','original_native_grid_any_valid','query_id'))
CENTER_KEYS=('center4_footprint_inside','all_contributing_raw_cover_alpha1',
 'all_contributing_raw_sham_alpha0','frozen_sampled_cover_alpha_center4_255',
 'frozen_sampled_sham_alpha_center4_0','sham_center4_RGB_bit_equal_before',
 'sham_some_selected_local_donor_support','cover_sham_equal_pasted_alpha_mass','positive_pasted_mass')
READS=[]


def safe(path):
 p=Path(path).resolve()
 assert p.is_relative_to(P) and p.suffix in ('.json','.jsonl','.py'),p
 assert not any(s in str(p).lower()for s in ('_v112','current223','/private/')),p
 return p


def bind(path):
 p=safe(path)
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def check(b):
 assert bind(b['path'])==b
 return safe(b['path'])


def read(path):
 p=safe(path);READS.append(str(p));return json.loads(p.read_text())


def rows(path):
 p=safe(path);READS.append(str(p));return[json.loads(x)for x in p.read_text().splitlines()if x]


def write(name,obj,jsonl=False):
 p=H/name
 with p.open('x')as f:
  if jsonl:
   for row in obj:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
  else:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return bind(p)


def tallies(records,section,keys):
 """Missing evidence is untested, never treated as passing or failing."""
 result={}
 for key in keys:
  values=[r.get(section,{}).get(key)for r in records]
  assert all(v is None or isinstance(v,bool)for v in values),(section,key)
  result[key]=dict(PASS=sum(v is True for v in values),FAIL=sum(v is False for v in values),
                   UNTESTED=sum(v is None for v in values))
 return dict(per_check=result,all_attempts_every_required_check_PASS=all(v['PASS']==len(records)for v in result.values()),
             all_tested_checks_PASS=all(v['FAIL']==0 for v in result.values()),
             attempts_all_required_checks_PASS=sum(all(r.get(section,{}).get(k)is True for k in keys)for r in records),
             missing_is_not_success=True)


def status(r):
 return dict(pair_id=r['pair_id'],PASS=r['PASS'],reason=r.get('reason','UNSPECIFIED'),
  before_parity_failed=[k for k,v in r.get('before_parity',{}).items()if v is False],
  proof_failed=[k for k,v in r.get('proof',{}).items()if v is False],
  geometry_reason=r.get('geometry',{}).get('reason'),
  exception_final_line=(r.get('traceback','').strip().splitlines()or[None])[-1],
  constructed_variants={name:not(r['variants'][name].get('constructed')is False)for name in VARIANTS},
  conditional_weak_owner_auxiliary_eligibility=r['weak_owner_auxiliary_eligibility_only'],
  no_new_owner_labels_or_training_admission=True)


def invented():
 rr=[dict(before_parity={'a':True},proof={'b':True}),
     dict(before_parity={'a':False},proof={}),dict(before_parity={},proof={'b':False})]
 t=tallies(rr,'before_parity',('a',))
 assert t['per_check']['a']==dict(PASS=1,FAIL=1,UNTESTED=1)
 assert not t['all_attempts_every_required_check_PASS']and not t['all_tested_checks_PASS']
 t=tallies([dict(proof={}),dict(proof={'b':True})],'proof',('b',))
 assert t['all_tested_checks_PASS']and not t['all_attempts_every_required_check_PASS']
 assert t['attempts_all_required_checks_PASS']==1
 assert 159*3==477 and 477!=46 and 477!=53
 return dict(completed=True,PASS=True,invented_JSON_records_only=True,
  tested_PASS_FAIL_UNTESTED_are_distinct=True,untested_not_counted_as_success=True,
  attempted477_variant_rows_not_independent_images=True,source_generated_payload_reads=0)


def prepare():
 p=read(PROTOCOL);d=read(DESIGN)
 assert p['design']==bind(DESIGN)and p['assignments']==bind(ASSIGN)and p['fixed12']==bind(SELECT)
 assert d['pairs_total']==159 and d['number_of_anchor_queries']==53 and d['number_of_anchor_frames']==46
 assert p['all159_attempts_retained']and p['training_admitted']is False
 assert p['output_schema']['proof_manifest']=='ALL159_PIXEL_INPUT_PROOFS.jsonl'
 generated=write('GENERATED_INVENTED_SUMMARY_CHECKS.json',dict(**invented(),code=bind(__file__)))
 plan=write('OUTPUT_PLAN_BEFORE_GENERATED_PAYLOADS.json',dict(
  completed=True,prepared_only=True,code=bind(__file__),design=bind(DESIGN),protocol=bind(PROTOCOL),
  assignments=bind(ASSIGN),fixed12=bind(SELECT),
  post_trigger_read_scope='source_pixels_v1 COMPLETE.json and bound ALL159_PIXEL_INPUT_PROOFS.jsonl only; inherited output paths/hashes are recorded without reading RGB/NPY/NPZ bytes.',
  root_trigger_required=True,no_progress_polling=True,
  denominator=dict(anchor_queries=53,source_frames=46,donors=59,pairs_attempted=159,planned_variant_rows=477,
                   variants_per_pair=list(VARIANTS),variant_rows_are_not_independent_images=True),
  required_summaries=['all159 attempts retained vs all159 proofs pass','PASS/FAIL/UNTESTED per BEFORE parity and unchanged/center proof check',
    '53 anchor and46 frame coverage including failures','59 fixed donor/class usage and conditional pass coverage',
    'all12 preselected distinct anchors retained including failures','native0..7 with absent2/3 explicit',
    'conditional owner auxiliary eligibility only; no new labels/admission/nativeE or primary success'],
  outputs=['SUMMARY.json','ALL159_DERIVED_ATTEMPT_OUTCOMES.jsonl','ALL53_ANCHOR_COVERAGE.json',
    'ALL59_DONOR_CLASS_COVERAGE.json','FIXED12_RETAINED_DERIVED_STATUS.json','EXTRACTION_COMPLETE.json'],
  generated_payloads_read_during_preparation=0,RGB_NPY_NPZ_feature_model_weight_reads_or_hashes=0,
  training_admitted=False,selected_model=None,fully_solved=False))
 receipt=write('PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,
  code=bind(__file__),plan=plan,generated=generated,metadata_reads=READS.copy(),
  source_generated_payload_reads=0,source_pixel_NPZ_feature_or_model_calls=0))
 print(json.dumps(dict(prepared=True,receipt=receipt)))


def extract(root_note):
 assert root_note and len(root_note.strip())>=20,'Explicit root source builder completion trigger required'
 prep=read(H/'PREPARATION_COMPLETE.json');assert check(prep['code'])==Path(__file__).resolve()
 p=read(PROTOCOL);check(p['assignments']);check(p['fixed12']);check(p['frame_inputs'])
 assert p['code']==bind(V/'pixel_builder_preparation/build_coherent_source_pixels.py')
 # Only after root barrier: final completion is the first generated body opened.
 complete=read(SOURCE/'COMPLETE.json')
 assert complete['completed'] and complete['all159_attempts_retained']
 assert complete['protocol']==bind(PROTOCOL) and complete['assignments']==bind(ASSIGN) and complete['fixed12']==bind(SELECT)
 rr=rows(check(complete['proofs']));aa=rows(ASSIGN);selection=read(SELECT)
 frame_metadata=read(check(p['frame_inputs']))['frames']
 assert len(rr)==len(aa)==159 and len({r['pair_id']for r in rr})==159
 assert [r['pair_id']for r in rr]==[a['pair_id']for a in aa]
 assert len({a['anchor_query_id']for a in aa})==53 and len({a['frame_id']for a in aa})==46
 assert len(frame_metadata)==46 and all(f['split']=='SOURCE_DEVELOPMENT'for f in frame_metadata)
 outcome=[]
 for r,a in zip(rr,aa):
  assert isinstance(r['PASS'],bool) and r['assignment_index']==a['assignment_index']
  for key in ('pair_id','pair_sha256','anchor_query_id','donor_id','native_slot','split'):assert r[key]==a[key]
  assert r['original_q']==a['query_xy'] and r['original_L']==a['L']
  assert r['attempted_once']is True and r['replacement_or_retry']is False and r['training_admitted']is False
  assert r['weak_owner_auxiliary_eligibility_only']==r['PASS']
  assert r['new_native_visibility_target']==r['new_point_quality_target']==[-1]*8
  assert set(r['variants'])==set(VARIANTS)
  if r['PASS']:
   assert all(r.get('before_parity',{}).get(k)is True for k in BEFORE_KEYS)
   assert all(r.get('proof',{}).get(k)is True for k in CENTER_KEYS+UNCHANGED_KEYS)
   assert all(v is True for v in r['proof'].values())
  s=status(r);outcome.append(dict(**s,anchor_query_id=a['anchor_query_id'],frame_id=a['frame_id'],frame_index=a['frame_index'],
   source_id=a['source_id'],split=a['split'],native_slot=a['native_slot'],donor_id=a['donor_id'],
   donor_object_id=a['donor']['object_id'],replica=a['replica'],scale=a['scale'],longside_pixels=a['longside_pixels']))
 passed=sum(r['PASS']for r in rr);failed=159-passed
 assert complete['passed_pixel_input_proofs']==passed and complete['failed_attempts_retained']==failed
 assert complete['PASS']==complete['all_proofs_pass']==(passed==159)
 assert not complete['training_admitted'] and not complete['original_labels_read_or_changed']
 assert complete['all_new_native_visibility_and_point_quality_UNKNOWN']
 reasons=Counter(r.get('reason','UNSPECIFIED')for r in rr)
 failed_keys=Counter(k for r in rr for k,v in r.get('proof',{}).items()if v is False)
 parity=tallies(rr,'before_parity',BEFORE_KEYS)
 unchanged=tallies(rr,'proof',UNCHANGED_KEYS)
 center=tallies(rr,'proof',CENTER_KEYS)
 by_anchor=[]
 for qid in sorted({a['anchor_query_id']for a in aa}):
  pairs=[r for r in outcome if r['anchor_query_id']==qid];assert len(pairs)==3
  by_anchor.append(dict(anchor_query_id=qid,frame_id=pairs[0]['frame_id'],native_slot=pairs[0]['native_slot'],
   attempted=3,passed=sum(r['PASS']for r in pairs),failed=sum(not r['PASS']for r in pairs),pair_ids=[r['pair_id']for r in pairs]))
 donor_rows=[]
 for donor_id in sorted({a['donor_id']for a in aa}):
  pairs=[r for r in outcome if r['donor_id']==donor_id]
  donor_rows.append(dict(donor_id=donor_id,donor_object_id=pairs[0]['donor_object_id'],split='TRAIN',
   attempted=len(pairs),passed=sum(r['PASS']for r in pairs),failed=sum(not r['PASS']for r in pairs)))
 assert len(donor_rows)==59 and Counter(r['attempted']for r in donor_rows)=={2:18,3:41}
 class_coverage={str(k):dict(donors=sum(d['donor_object_id']==k for d in donor_rows),
  attempted=sum(d['attempted']for d in donor_rows if d['donor_object_id']==k),
  conditional_eligible_pairs=sum(d['passed']for d in donor_rows if d['donor_object_id']==k),
  failed=sum(d['failed']for d in donor_rows if d['donor_object_id']==k))for k in sorted({d['donor_object_id']for d in donor_rows})}
 by_id={r['pair_id']:r for r in rr};fixed12=[]
 for item in selection['selected']:
  r=by_id[item['pair_id']]
  assert r['anchor_query_id']==item['anchor_query_id'] and r['pair_sha256']==item['pair_sha256']
  fixed12.append(dict(selection_rank=item['selection_rank'],selection_reason=item['selection_reason'],
   **status(r),anchor_query_id=item['anchor_query_id'],native_slot=item['native_slot'],variants=r['variants']))
 assert len(fixed12)==len({r['anchor_query_id']for r in fixed12})==12
 native={str(k):dict(anchors=sum(a['native_slot']==k for a in by_anchor),
  attempted=sum(r['native_slot']==k for r in outcome),passed=sum(r['native_slot']==k and r['PASS']for r in outcome),
  failed=sum(r['native_slot']==k and not r['PASS']for r in outcome))for k in range(8)}
 assert native['2']['attempted']==native['3']['attempted']==0
 constructed={name:sum(r['constructed_variants'][name]for r in outcome)for name in VARIANTS}
 outputs={}
 outputs['attempts']=write('ALL159_DERIVED_ATTEMPT_OUTCOMES.jsonl',outcome,jsonl=True)
 outputs['anchors']=write('ALL53_ANCHOR_COVERAGE.json',dict(completed=True,rows=by_anchor))
 outputs['donors']=write('ALL59_DONOR_CLASS_COVERAGE.json',dict(completed=True,rows=donor_rows,class_coverage=class_coverage))
 outputs['fixed12']=write('FIXED12_RETAINED_DERIVED_STATUS.json',dict(completed=True,original_selection=bind(SELECT),
  all12_original_pair_ids_and_distinct_anchors_retained=True,no_failure_replacement=True,rows=fixed12))
 summary=dict(schema='V161_SOURCE_ONLY_PIXEL_INPUT_PROOF_DERIVED_SUMMARY',completed=True,
  execution_and_attempt_completeness_PASS=True,all_pixel_input_proofs_PASS=(passed==159),
  source_complete_PASS_semantics='Builder COMPLETE.PASS requires every159 proof to pass; completed/retained is reported separately.',
  attempted_pairs=159,passed_pixel_input_proof_pairs=passed,failed_attempts_retained=failed,
  unique_source_anchor_queries=53,unique_source_frames=46,unique_fixed_TRAIN_donors=59,
  planned_variant_rows=477,constructed_variant_rows=sum(constructed.values()),constructed_by_variant=constructed,
  variant_rows_are_not_independent_images=True,independent_image_count_not_estimated=True,
  reason_counts=dict(reasons),failed_proof_key_counts=dict(failed_keys),
  failures=[x for x in outcome if not x['PASS']],before_all8_parity=parity,
  generated_qL_native_and_availability_unchanged_checks=unchanged,center_and_sham_input_proof_checks=center,
  all159_pair_metadata_original_q_L_native_split_and_assignment_unchanged=True,
  generated_array_claim_scope='Per-proof saved JSON declarations/check outcomes. This extractor does not open or independently rehash variant NPZ/RGB; separate pixel review is authoritative for those arrays.',
  donor_usage_histogram={'2':18,'3':41},donor_object_class_coverage=class_coverage,
  native_coverage=native,missing_anchor_native_slots=[2,3],absence_not_imputed_or_claimed_covered=True,
  anchors_with_any_conditional_eligible_pair=sum(a['passed']>0 for a in by_anchor),
  anchors_with_all3_proofs_PASS=sum(a['passed']==3 for a in by_anchor),
  donors_with_any_conditional_eligible_pair=sum(d['passed']>0 for d in donor_rows),
  conditional_weak_owner_auxiliary_eligibility=dict(pairs=passed,variant_rows_if_later_separately_admitted=3*passed,
   by_variant={name:passed for name in VARIANTS},new_owner_labels_created=False,training_admitted=False,
   meaning='Controlled pixel/input proof eligibility only; not new native E, coordinate quality, human gold, learned accuracy or admission.'),
  fixed12=dict(all12_original_selection_retained=True,passed=sum(x['PASS']for x in fixed12),failed=sum(not x['PASS']for x in fixed12),
   no_success_replacement=True,source_selection=bind(SELECT)),
  all159_declared_new_native_visibility_and_point_quality_UNKNOWN=True,original_labels_read_or_changed=False,
  original_WOOD_SQUARE_all_native8_full_goal_unchanged=True,full_goal_success=False,
  source_only=True,training_admitted=False,selected_model=None,adopted=False,fully_solved=False,
  actual_new_encoder_model_initializer_fit_PnP_calls=0,current_allowed=False,outputs=outputs)
 outputs['summary']=write('SUMMARY.json',summary)
 receipt=write('EXTRACTION_COMPLETE.json',dict(completed=True,PASS=True,root_completion_message=root_note,
  UTC=datetime.now(timezone.utc).isoformat(),code=bind(__file__),preparation=bind(H/'PREPARATION_COMPLETE.json'),
  source_complete=bind(SOURCE/'COMPLETE.json'),source_proofs=complete['proofs'],protocol=bind(PROTOCOL),
  assignments=bind(ASSIGN),fixed12=bind(SELECT),outputs=outputs,exact_JSON_reads=READS,
  source_RGB_NPY_NPZ_feature_weight_or_model_bodies_read_or_hashed=0,training_admitted=False,selected_model=None))
 print(json.dumps(dict(completed=True,passed=passed,failed=failed,summary=outputs['summary'],receipt=receipt)))


if __name__=='__main__':
 parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
 mode.add_argument('--prepare',action='store_true');mode.add_argument('--extract',action='store_true')
 parser.add_argument('--root-completion-note');args=parser.parse_args()
 if args.prepare:
  assert args.root_completion_note is None;prepare()
 else:extract(args.root_completion_note)
