"""V161 focused publication and JSON-only next-input mapping review."""
from pathlib import Path
from html.parser import HTMLParser
from collections import Counter
from datetime import datetime,timezone
import hashlib,json
V=Path(__file__).resolve().parents[1];P=V.parent;H=Path(__file__).resolve().parent
READS=[]
def bind(p):
 p=Path(p).resolve();assert p.suffix in ('.json','.jsonl','.py','.html')
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def check(b):
 assert bind(b['path'])==b;return Path(b['path'])
def read(p):READS.append(str(p));return json.loads(Path(p).read_text())
def rows(p):READS.append(str(p));return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
class Table(HTMLParser):
 def __init__(self):super().__init__();self.rows=[];self.row=None;self.cell=None
 def handle_starttag(self,t,a):
  if t=='tr':self.row=[]
  if t in ('th','td'):self.cell=''
 def handle_data(self,d):
  if self.cell is not None:self.cell+=d
 def handle_endtag(self,t):
  if t in ('td','th')and self.cell is not None:self.row.append(self.cell);self.cell=None
  if t=='tr'and self.row is not None:self.rows.append(self.row);self.row=None

def main():
 assert bind(V/'ROOT_READOUT_KO.html')['sha256']=='e8781df383421df9493565baef4c81659eb177f7aeb8c59e4052c138ddd66c20'
 assert bind(V/'ROOT_RESULTS_KO.json')['sha256']=='1b9f62b63bd73f0d500b25139c067756092dc0055bee09a7bb861c26b609ebdf'
 root=read(V/'ROOT_RESULTS_KO.json');summary=read(check(root['summary']));ind=read(check(root['independent_review']))
 mc=read(check(root['feature_input_metadata']));nxt=read(check(root['next_requirements']))
 for key,sk in [('attempted_pairs','attempted_pairs'),('passed_pairs','passed_pixel_input_proof_pairs'),('retained_failed_pairs','failed_attempts_retained'),('constructed_variant_records','constructed_variant_rows'),('original_anchor_queries','unique_source_anchor_queries'),('original_anchor_frames','unique_source_frames'),('fixed_TRAIN_donors','unique_fixed_TRAIN_donors'),('native_coverage','native_coverage')]:assert root[key]==summary[sk]
 assert root['execution_and_integrity_PASS']and not root['all159_pixel_proofs_PASS']and ind['PASS']and ind['integrity_PASS']
 assert ind['pair_attempts']==159 and ind['pixel_proof_passed']==157 and ind['pixel_proof_failures_retained']==2
 assert root['conditional_eligibility']==dict(pairs=157,variant_rows=471,additional_COVER_SHAM_rows_if_later_admitted=314,BEFORE_not_duplicated_in_prospective_owner_loss=True)
 html=(V/'ROOT_READOUT_KO.html').read_text();parser=Table();parser.feed(html);assert len(parser.rows)==9
 for row in parser.rows[1:]:
  k,*values=map(int,row);m=summary['native_coverage'][str(k)]
  assert values==[m[x]for x in ('anchors','attempted','passed','failed')]
 for text in ('COMPLETE.PASS','false','477개의 독립 실사로 세지 않습니다','native2/3이 없습니다','UNKNOWN','학습 편입이나 새로운 정답 등록은 하지 않았습니다','COVER157+SHAM157','BEFORE는','아직 새 encoder/학습을 실행하거나 편입을 승인한 것은 아닙니다','전체 목표는 계속 미해결'):assert text in html,text
 metadata=rows(check(mc['manifest']));proof=rows(check(mc['source_proof_manifest']))
 frames=read(check(mc['original_source_frame_metadata']))['frames'];assign=rows(V/'metadata_assignment/ASSIGNMENTS159_BEFORE_PIXELS.jsonl')
 pm={r['pair_id']:r for r in proof};am={r['pair_id']:r for r in assign};fm={r['frame_index']:r for r in frames}
 original_path=P/'query_visibility_native_slot_owner_factorial_20261024_v160/input_contract_audit/EXPECTED_SOURCE_INPUT_NATIVE_METADATA28976.jsonl'
 original={x['global_index']:x for x in rows(original_path)if x['family']=='MR1840'}
 policy_path=P/'query_visibility_mr6d_full_public_source_20261024_v153/initializer_feature_preparation/FEATURE_PROTOCOL_BEFORE_DINO_FORWARD.json'
 policy=read(policy_path)
 assert policy['availability']=='query_available is original native continuous flag only (V147-compatible metadata), nearest/context/central4/Qinside/feature_constructed remain separate; no new classifier U rule'
 assert len(metadata)==3816==477*8 and len({r['input_occurrence_id']for r in metadata})==3816
 checks=0
 for r in metadata:
  a=am[r['pair_id']];z=pm[r['pair_id']];f=fm[a['frame_index']];k=r['native_slot'];variant=r['variant']
  assert variant in ('BEFORE','COVER','ADJACENT_SHAM')and k in range(8)
  assert r['input_occurrence_id']==f"{r['pair_id']}/{variant}/native{k}"
  for out,src in [('original_query_id','query_id'),('original_global_index','global_index'),('query_xy','query_xy'),('L','L'),('feature_constructed','feature_constructed'),('nearest_valid','nearest_valid'),('central4_available','local_central4_available'),('context_available','context_available')]:
   assert r[out]==f[src][k];checks+=1
  assert r['original_frame_index']==a['frame_index']and r['source_id']==a['source_id']and r['split']==a['split']=='SOURCE_DEVELOPMENT'
  assert r['variant_payload']==z['variants'][variant]and r['original_context_box_xyxy']==f['context_box_xyxy']
  # Frozen source-specific MR availability: continuous only. Other input
  # diagnostics remain unchanged and do not introduce a new classifier U gate.
  assert r['query_available']==f['native_continuous_valid'][k]==original[r['original_global_index']]['query_available']
  candidate=bool(z['PASS']and variant in ('COVER','ADJACENT_SHAM')and k==a['native_slot'])
  assert r['future_owner_supervision_candidate_only']==candidate
  assert r['pair_pixel_proof_PASS']==z['PASS']and r['selected_anchor_native_slot']==a['native_slot']
  assert r['owner_target_not_created']and r['native_visibility_target']==r['point_quality_target']==-1 and r['training_admitted']is False
  assert 'owner_target' not in r;checks+=8
 assert Counter(r['variant']for r in metadata)=={'BEFORE':1272,'COVER':1272,'ADJACENT_SHAM':1272}
 assert Counter(r['native_slot']for r in metadata)=={k:477 for k in range(8)}
 assert sum(not r['query_available']for r in metadata)==root['all8_original_inputU']==mc['inputU']==360
 assert sum(r['future_owner_supervision_candidate_only']for r in metadata)==mc['future_additional_owner_COVER_SHAM_candidates']==314
 candidates=Counter(r['variant']for r in metadata if r['future_owner_supervision_candidate_only'])
 assert candidates=={'COVER':157,'ADJACENT_SHAM':157}
 assert all(r['query_available']for r in metadata if r['future_owner_supervision_candidate_only'])
 assert mc['training_admitted']is False and mc['metadata_only']and not mc['source_pixel_NPZ_feature_or_weight_bodies_read']and mc['encoder_or_model_calls']==0
 assert nxt['prospective_requirements_only']and not nxt['encoder_allowed_by_this_file']and not nxt['fit_allowed_by_this_file']and not nxt['training_admitted']
 assert any('TARGET210/FOREIGN209/BG148' in s for s in nxt['requirements_KO'])
 assert (53+candidates['ADJACENT_SHAM'],52+candidates['COVER'],148)==(210,209,148)and 253+314==567
 assert root['selected_model']is None and not root['adopted']and not root['fully_solved']and not root['new_owner_labels_or_training_admission']
 assert root['current223_reads']==root['V112_reads']==root['feature_extraction_calls']==root['model_training_updates']==root['PnP_q_L_changes']==0
 donor=read(V/'donor_proof_v1/DONOR_PROOF_COMPLETE.json');assert donor['PASS']and donor['attempted']==donor['passed']==59 and donor['unique_hashed_source_assets']==448
 receipt=dict(schema='V161_FOCUSED_FINAL_PUBLICATION_AND_NEXT_INPUT_METADATA_REVIEW',completed=True,PASS=True,UTC=datetime.now(timezone.utc).isoformat(),findings=[],
  publication=[bind(V/'ROOT_READOUT_KO.html'),bind(V/'ROOT_RESULTS_KO.json')],review_code=bind(__file__),
  exact_native_table_rows=8,metadata_occurrence_checks=checks,
  verified={'all159_retained_157PASS_2failed_and477_variant_records_match_summary_and_independent_review':True,
  'completed_integrity_PASS_distinct_from_all159_pixel_proofs_PASS_false':True,
  'native0_to7_counts_and_absent_anchor2_3_explicit':True,
  '477_variant_rows_not_independent_images_and53queries46frames_preserved':True,
  'weak_owner_eligibility_only_native_qualityUNKNOWN_no_admission_or_model_success':True,
  'full_WOOD_SQUARE_native8_goal_unchanged_source_only_no_selection':True,
  'all3816_exact159x3x8_original_q8_L_native_frame_source_split_and_payload_bindings':True,
  'MR_original_continuous_only_query_availability_exact_all3816_with0_discrepancies_inputU360':True,
  'nearest_center4_context_constructed_preserved_as_separate_diagnostics_without_new_gate':True,
  'candidate314_exactly_COVER157_plus_SHAM157_original_selected_anchor_slots_only':True,
  'no_BEFORE_or_failed_pair_candidate_duplicate':True,
  'future253_plus314_equals567_T210_F209_BG148_explicitly_prospective_no_labels_or_approval':True},
  counts=dict(pairs=159,passed=157,failed=2,variant_records=477,input_occurrences=3816,inputU=360,
   conditional_additional_candidates=314,conditional_COVER=157,conditional_SHAM=157,BEFORE_candidates=0,
   prospective_total_owner_if_separately_admitted=567,prospective_class_counts=[210,209,148]),
  dismissed_review_assumption=dict(initial_suspicion='A nearest/center4-invalid non-anchor row was suspected unavailable using a synthetic conjunction.',
   resolution='Reviewer assumption was incorrect. Frozen V153 MR protocol explicitly uses native_continuous_valid only; all3816 mapping rows match original V160 MR query_available exactly. No source or report repair or new U gate is warranted.',
   original_rule=policy['availability'],primary_policy=bind(policy_path),original_metadata=bind(original_path),discrepancies=0,source_or_report_changes_required=False),
  evidence=[root['summary'],root['independent_review'],root['feature_input_metadata'],mc['manifest'],mc['source_proof_manifest'],mc['original_source_frame_metadata'],root['next_requirements'],bind(V/'donor_proof_v1/DONOR_PROOF_COMPLETE.json')],
  image_review_scope='No image required/read in this focused pass; root separately viewed the first PNG and visual readout receipt is bound in published results.',
  exact_metadata_reads=READS,source_RGB_NPY_NPZ_feature_model_weight_or_private_current_V112_bodies_read_or_hashed=0,
  new_encoder_model_fit_initializer_PnP_calls=0,training_admitted=False,selected_model=None,
  limitations=['Publication and metadata consistency only; independent C receipt supplies actual pixel verification.','Future owner counts are arithmetic and planning requirements; no new targets or training admission created.'])
 output=H/'FINAL_PUBLICATION_REVIEW.json'
 with output.open('x')as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(dict(PASS=True,metadata_checks=checks,candidate_variant_counts=candidates,review=bind(output))))
if __name__=='__main__':main()
