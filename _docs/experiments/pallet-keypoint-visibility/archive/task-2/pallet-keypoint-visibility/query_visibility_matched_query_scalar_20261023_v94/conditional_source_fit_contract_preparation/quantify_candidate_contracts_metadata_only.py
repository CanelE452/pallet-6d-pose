import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'query_visibility_matched_query_scalar_20261023_v94'
O=R/'conditional_source_fit_contract_preparation'
A=R/'pallox_actual_query_metadata_audit'
Q=A/'ALL800_QUERY_METADATA_AND_CUE_AVAILABILITY_GT_ONLY.jsonl'
X=A/'ALL4832_QUERY_X_CUE_VIEW_AVAILABILITY_GT_ONLY.jsonl'
V91=P/'query_visibility_physical_cue_aux_20261023_v91'
def bind(p):
 b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def write(p,x):
 assert not p.exists(),str(p);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def jl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def count_bucket(n):return '0' if n==0 else '1' if n==1 else '>1'

VARIANTS={
 'RAW_SINGLE_FULL_KNOWNSET':'Input available AND exactly one original FULL known cue record. Competitors retained as explicit flags; not complete crop truth.',
 'RAW_ALL_SAME_FULL_KNOWNSET':'Input available AND nonempty original FULL known records all D or all E. Competitors retained; multiple originals may be seed duplicates.',
 'CLEAN_SINGLE_CERTIFIED_GROUP_KNOWNSET':'Input available AND exactly one knownFULL certified group, same known state, no locatedFULLunknown group and no PARTIAL group outside knownFULL set. Null/unlocated slots remain flagged and outside the conditioned known set.',
 'CLEAN_ALL_SAME_CERTIFIED_GROUP_KNOWNSET':'Same as CLEAN_SINGLE but >=1 certified groups allowed when every FULLknown record has the same state. Null/unlocated slots remain flagged.',
 'STRICT_COMPLETE_SINGLE_NO_NULL':'CLEAN_SINGLE plus zero null/unlocated groups. More conservative complete annotated-slot-set candidate, not native gold.',
 'STRICT_COMPLETE_ALL_SAME_NO_NULL':'CLEAN_ALL_SAME plus zero null/unlocated groups. More conservative complete annotated-slot-set candidate, not native gold.',
}
def summarize(rr):
 out={}
 for name in VARIANTS:
  hist=Counter(r['variants'][name]['label_candidate'] for r in rr)
  out[name]=dict(N=len(rr),D=hist['D_CONDITIONAL'],E=hist['E_CONDITIONAL'],U=hist['U'],known=hist['D_CONDITIONAL']+hist['E_CONDITIONAL'],query_IDs_known=[r['query_id'] for r in rr if r['variants'][name]['label_candidate']!='U'])
 return out
def compact(ss):return {k:{a:b for a,b in v.items() if a!='query_IDs_known'} for k,v in ss.items()}

def main():
 write(O/'SCOPE_AND_CANDIDATE_RULES_BEFORE_RECOUNT.json',dict(schema='V94_DRAFT_CONDITIONAL_KNOWNSET_QUERY_SCALAR_SOURCE_CONTRACT_POLICY',created_UTC=datetime.now(timezone.utc).isoformat(),status='DRAFT_CHOICES_NOT_FROZEN_NO_FIT_NO_ADMISSION',
  bindings=dict(root_scope=bind(R/'SCOPE_BEFORE_MATCHED_QUERY_CONTRACT.json'),actual_audit=bind(A/'FINAL_METADATA_AUDIT_SEAL.json'),queries800=bind(Q),cross4832=bind(X),teacher_contract=bind(V91/'teacher_preparation/TEACHER_CONTRACT_BEFORE_BUILD.json')),
  prior_exposure='Counts from the completed source metadata audit are already known. This compares explicit source-only label-contract alternatives; no blind or preregistered performance claim.',
  input_available='Saved q exists AND nativecontinuous valid AND nearest/RGBcentre valid; no actual crop/ROI availability assumed until future producer QA. Keep all unavailable IDs as U. No new crop resize/recenter or candidate choice.',
  definitions=VARIANTS,located_unknown='Teacher U with defined FULL uncertainty box. It competes with an expected known-set scalar and is never made D/E.',
  partial_competitor='A defined PARTIAL cue group outside the FULLknown group set; any D/E/U state competes. A second linked seed proposal PARTIAL for the SAME already FULLknown certified group is logged but not a distinct physical competitor.',
  null='No reference coordinate means its in/out-crop support cannot be certified. Complete-set/native task remains U. A CONDITIONED known-set proxy may explicitly exclude unlocated slots, but this is a target restriction, not proof no competing cue exists. Both alternatives counted.',
  certified_groups='Only existing V84 seed duplicate links, no new cue merging or native mapping. Original group members/boxes retained.',
  label_target='State of recorded FULL known physicalcue set conditioned on its presence in original q crop, NOT querypixel membership/native expectedpart/global cropabsence. No automatic association.',
  pixels_arrays_models_weights_current_forward_fit_PnP=0,training_admission=False))
 qrows=jl(Q);xrows=jl(X);assert len(qrows)==800 and len(xrows)==4832
 byq=defaultdict(list)
 for x in xrows:byq[x['query_id']].append(x)
 rr=[];group_duplicate_partial=0
 for q in qrows:
  xs=byq[q['query_id']];assert len(xs) in [6,8]
  input_ok=q['original_q'] is not None and q['nativecontinuous_valid'] and q['nearest_valid'] and q['RGBcentre_valid']
  fullknown=[x for x in xs if x['coverage']=='FULL' and x['teacher_label'] in [0,1]]
  states={x['teacher_label'] for x in fullknown};rawN=len(fullknown)
  groupmap=defaultdict(list)
  for x in xs:groupmap[x['certified_duplicate_group']].append(x)
  known_groups={x['certified_duplicate_group'] for x in fullknown}
  unknown_full_groups={x['certified_duplicate_group'] for x in xs if x['coverage']=='FULL' and x['teacher_label']==-1}
  partial_groups={x['certified_duplicate_group'] for x in xs if x['coverage']=='PARTIAL'}
  competing_partial=partial_groups-known_groups
  linked_partial=partial_groups&known_groups
  # Same original cue cannot itself be both FULL and PARTIAL. Such linked cases
  # are only the previously certified multiple proposals of a seed region.
  if linked_partial:
   assert q['seed_namespace_applicable']
   group_duplicate_partial+=len(linked_partial)
  null_groups={x['certified_duplicate_group'] for x in xs if x['reference_null']}
  knownD={x['certified_duplicate_group'] for x in fullknown if x['teacher_label']==0}
  knownE={x['certified_duplicate_group'] for x in fullknown if x['teacher_label']==1}
  mixed_group=knownD&knownE
  flags=[]
  if q['original_q'] is None:flags.append('NO_QUERY')
  if q['original_q'] is not None and not q['nativecontinuous_valid']:flags.append('NATIVE_CONTINUOUS_OOB_OR_INVALID')
  if q['original_q'] is not None and q['nativecontinuous_valid'] and not q['RGBcentre_valid']:flags.append('RGB_CENTRE_OR_NEAREST_INVALID')
  if not fullknown:flags.append('NO_FULL_KNOWN_CUE')
  if len(states)>1:flags.append('MIXED_D_E_FULL_KNOWN')
  if unknown_full_groups:flags.append('LOCATED_FULL_UNKNOWN_CUE')
  if competing_partial:flags.append('DEFINED_PARTIAL_COMPETING_CUE')
  if null_groups:flags.append('NULL_UNLOCATED_CUE_SCOPE_UNRESOLVED')
  if mixed_group:flags.append('CERTIFIED_GROUP_STATE_CONFLICT')
  same=bool(fullknown) and len(states)==1 and not mixed_group
  clean=input_ok and same and not unknown_full_groups and not competing_partial
  candidate='D_CONDITIONAL' if states=={0} else 'E_CONDITIONAL' if states=={1} else 'U'
  admits={
   'RAW_SINGLE_FULL_KNOWNSET':input_ok and rawN==1 and same,
   'RAW_ALL_SAME_FULL_KNOWNSET':input_ok and same,
   'CLEAN_SINGLE_CERTIFIED_GROUP_KNOWNSET':clean and len(known_groups)==1,
   'CLEAN_ALL_SAME_CERTIFIED_GROUP_KNOWNSET':clean,
   'STRICT_COMPLETE_SINGLE_NO_NULL':clean and len(known_groups)==1 and not null_groups,
   'STRICT_COMPLETE_ALL_SAME_NO_NULL':clean and not null_groups,
  }
  variants={k:dict(label_candidate=candidate if v else 'U',conditional_proxy_only=True,automatic_training_admission=False) for k,v in admits.items()}
  rr.append(dict(query_id=q['query_id'],frame_index=q['frame_index'],frame_input_id=q['frame_input_id'],pair_id=q['pair_id'],view=q['view'],split=q['split'],native_slot=q['native_slot'],original_q=q['original_q'],original_L=q['original_L'],query_status=q['query_status'],input_metadata_available=input_ok,
   original_full_known_count=rawN,certified_group_full_known_count=len(known_groups),canonical_full_known_count=q['canonical_known_FULL_count'],full_known_state_set=sorted(states),full_known_cue_ids=[x['cue_id'] for x in fullknown],
   located_FULL_unknown_groups=sorted(unknown_full_groups),defined_PARTIAL_competing_groups=sorted(competing_partial),same_certified_group_partial_proposal_NOT_distinct_competitor=sorted(linked_partial),null_unlocated_groups=sorted(null_groups),
   raw_counts=dict(FULL_unknown=sum(x['coverage']=='FULL' and x['teacher_label']==-1 for x in xs),PARTIAL_known=sum(x['coverage']=='PARTIAL' and x['teacher_label']!=-1 for x in xs),PARTIAL_unknown=sum(x['coverage']=='PARTIAL' and x['teacher_label']==-1 for x in xs),null_reference=sum(x['reference_null'] for x in xs)),
   diagnostic_flags=flags,variants=variants,native_expected_part_state='UNKNOWN',human_gold=False,scalar_label_admitted=None,training_admission=False,source_query_audit_row_unchanged=q))
 # All raw initial queries can change naturally across the two captures; never
 # present them as controlled matched-q pairs merely because a cue is fixed.
 pairs=defaultdict(dict)
 for r in rr:pairs[r['pair_id']][(r['view'],r['native_slot'])]=(r['original_q'],r['original_L'])
 pair_equal=[p for p,d in pairs.items() if all(d[('before',k)]==d[('after',k)] for k in range(8))]
 table=summarize(rr);split={s:summarize([r for r in rr if r['split']==s]) for s in ['TRAIN_LAB','VAL_HALL']}
 comparison=[]
 for name,ss in table.items():
  for scope,subset in [('ALL',rr),('TRAIN_LAB',[r for r in rr if r['split']=='TRAIN_LAB']),('VAL_HALL',[r for r in rr if r['split']=='VAL_HALL'])]:
   row={'variant':name,'scope':scope,**compact(summarize(subset))[name]};comparison.append(row)
 out=O/'ALL800_CONTRACT_ALTERNATIVE_DIAGNOSTICS_GT_ONLY.jsonl';assert not out.exists();out.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rr))
 with (O/'CONTRACT_ALTERNATIVE_COUNTS.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
 old_vs_new=[]
 for a,b in [('RAW_SINGLE_FULL_KNOWNSET','RAW_ALL_SAME_FULL_KNOWNSET'),('CLEAN_SINGLE_CERTIFIED_GROUP_KNOWNSET','CLEAN_ALL_SAME_CERTIFIED_GROUP_KNOWNSET'),('STRICT_COMPLETE_SINGLE_NO_NULL','STRICT_COMPLETE_ALL_SAME_NO_NULL')]:
  added=[r for r in rr if r['variants'][a]['label_candidate']=='U' and r['variants'][b]['label_candidate']!='U']
  old_vs_new.append(dict(single=a,all_same=b,extra_queries=len(added),extra_D=sum(r['variants'][b]['label_candidate']=='D_CONDITIONAL' for r in added),extra_E=sum(r['variants'][b]['label_candidate']=='E_CONDITIONAL' for r in added),query_ids=[r['query_id'] for r in added]))
 summary=dict(schema='V94_CONDITIONAL_PHYSICAL_KNOWNSET_SAME_QUERY_SCALAR_FIT_CONTRACT_DRAFT_KO',status='DRAFT_ONLY_ROOT_CHOICE_PENDING_NOT_TRAINING_ADMISSION',completed_metadata_quantification=True,
  counts=dict(all_queries=800,input_metadata_available=sum(r['input_metadata_available'] for r in rr),unavailable=sum(not r['input_metadata_available'] for r in rr),original_pairs=50,original_native_queries_pair_exact_shared=len(pair_equal),original_native_queries_pair_exact_shared_ids=pair_equal,
   raw_knownFULL_0_1_many=dict(Counter(count_bucket(r['original_full_known_count']) for r in rr)),certified_group_knownFULL_0_1_many=dict(Counter(count_bucket(r['certified_group_full_known_count']) for r in rr)),
   queries_with_locatedFULLunknown=sum(bool(r['located_FULL_unknown_groups']) for r in rr),queries_with_definedPARTIAL_competitor=sum(bool(r['defined_PARTIAL_competing_groups']) for r in rr),queries_with_null_unlocated=sum(bool(r['null_unlocated_groups']) for r in rr),
   queries_with_both_knownFULL_and_FULLunknown=sum(r['original_full_known_count']>0 and bool(r['located_FULL_unknown_groups']) for r in rr),queries_with_knownFULL_and_PARTIALcompetitor=sum(r['original_full_known_count']>0 and bool(r['defined_PARTIAL_competing_groups']) for r in rr),queries_with_knownFULL_and_null=sum(r['original_full_known_count']>0 and bool(r['null_unlocated_groups']) for r in rr),same_certified_seed_partial_proposals=group_duplicate_partial),
  variant_definitions=VARIANTS,variant_counts=compact(table),variant_counts_by_split={k:compact(v) for k,v in split.items()},single_vs_all_same=old_vs_new,
  separate_U_reason_flags=dict(Counter(flag for r in rr for flag in r['diagnostic_flags'])),U_reason_flags_overlap_not_partition=True,
  draft_interpretation_KO=[
   'Known-set 후보 라벨은 원래 q/L crop에 FULL로 들어온 기존 conditional physical-cue 기록 집합의 상태이다. r/uncertainty는 GT-side availability와 loss target 생성에만 사용하며 query/box/candidate/slot을 바꾸지 않는다. 모델 입력이나 output 위치 선택에는 GT r을 넣지 않는다.',
   '단일-cue-only는 FULL known certified group 수가 정확히1인 조건이다. All-known-same은 여러 FULL known group이 있어도 모두 D 또는 모두 E일 때 후보를 보존한다. 다수결/nearest/GTbest는 없다. 원raw8 seed와 certified6 group을 둘 다 보존하고 기존 링크 외 새로운 merge는 하지 않는다.',
   'FULL unknown 또는 별도의 defined PARTIAL cue는 clean single-scalar 후보에서 U 사유다. PARTIAL 상태가 알려져 있어도 이 draft는 자동 known 라벨로 보강하지 않는다. Null reference는 crop 안팎을 알 수 없으므로 complete-set/native visibility는 U이다.',
   'Null을 가진 row를 conditioned known-set proxy에 쓰려면 root가 명시적으로 목표를 recorded FULL known-set의 상태로 제한해야 한다. 이는 다른 미주석 visible cue/expected target의 부재 증명이 아니다. 그 제한을 하지 않는 strict complete-set 대안에서는 null 하나라도 U로 유지한다. 두 감독량을 모두 표로 보였으며 이 파일은 어느 대안도 채택하지 않는다.',
   'D/E라벨 후보라도 FULL은 selected detection/expected cue association의 증명이 아니다. query-local conditional known-set proxy이며 native expectedpart UNKNOWN/humanfalse를 유지한다. 특히 E-only는 crop 전체에 목재/다른 cue가 안 보인다는 라벨이 아니다.',
   'Actual before/after initializer q/L은 원칙적으로 독립이다. 같은 pair/cue의 상태변화만으로 SAME-q causal contrast라고 부르지 않는다. 향후 controlled cover/sham은 원q/L 고정·전후 RGB/features 일치·donor/capture split을 별도 동결해야 한다.',
  ],
  executable_fit_requirements_DRAFT_ONLY=[
   'Root가 variant/conditioned target 및 availability/UNKNOWN loss0를 사전 선택·동결한다. All800 입력·실패/U row를 보존하고 fit sampler에는 선택된 known loss만 쓴다. Original candidate firstargmax/slot/q/L은 불변이다.',
   '최종 actual same-q input producer가 source RGB96/P3/validity를 생성·검산한 뒤 입력 lock을 만든다. 이 단계의696은 metadata 가능성일 뿐 실제 tensor/ROI QA 통과 선언이 아니다.',
   'Train/sourceVAL/deployment에서 동일 crop-centred 단일 scalar 연산자를 사용한다. Region/mosaic로 학습한 뒤 q-centre로 바꾸거나 GT-best tile/output을 읽지 않는다.',
   'TRAIN_LAB352/VAL_HALL448 원래 source partition과 pair/cue/capture grouping을 유지한다. One physical bin/environment holdout이며 flat wooden/square pallet의 native 의미나 독립 object/site 일반화 성능을 보증하지 않는다.',
   'Certified seed duplicate IDs는 보존한다. 원proposal별로 loss를 중복 가산하지 않도록 기존1/n lineage 또는 query-level singleloss 방식을 fit 전에 명시한다. 이 비교는 새 loss weight를 선택하지 않는다.',
   '모델/init/업데이트수/샘플러/optimizer/threshold/checkpoint criterion/sourcegate를 source-only로 별도 동결한다. Validation에서 candidate threshold/radius/모델을 뒤늦게 고르지 않고 current223은 fit/선택에 사용하지 않는다.',
  ],
  scientific_risks_KO=[
   'All-known-same의 감독량 증가는 서로 다른 구조/거리의 cue를 같은 scalar로 묶는 범위 확대다. 관찰되지 않은 competing cue와 target identity를 무시하면 material-presence shortcut이나 wrong-target 오판을 학습할 수 있다.',
   'Single-only도 native expectedcue binding이 생기는 것은 아니다. Large window에 하나의 주석 cue만 있다는 것과 그 slot이 그 cue를 요구한다는 것은 다르다.',
   '자연 cardboard E는 AI-조건부 지역 관찰이며 RGB 정렬/subpixel/opaque mask gold가 아니다. Same physical bin의 외관과 배경·상자 shortcut을 분리하는 진단 및 향후 별도 flat-pallet source evidence가 필요하다.',
   'Null/unknown을 negative로 채우거나 MISS를 E로 만들지 않는다. 본문 D/E 후보를 human/currentnative D/E/S와 같다고 해석하지 않는다.',
  ],
  policy=bind(O/'SCOPE_AND_CANDIDATE_RULES_BEFORE_RECOUNT.json'),all800_diagnostics=bind(out),metadata_counts_only=True,new_pixels_arrays_weights_models_current_forward_fit_PnP=0,sourceGTlabels_modified=0,training_admission=False,native_expected_part_state='UNKNOWN',human_gold=False,goal_status='active',fully_solved=False)
 write(O/'CONDITIONAL_SOURCE_FIT_CONTRACT_DRAFT_KO.json',summary)
 write(O/'SEAL.json',dict(completed=True,PASS_metadata_quantification=True,status='DRAFT_ONLY_NO_FIT_OR_ADMISSION',policy=bind(O/'SCOPE_AND_CANDIDATE_RULES_BEFORE_RECOUNT.json'),code=bind(Path(__file__)),outputs=[bind(out),bind(O/'CONTRACT_ALTERNATIVE_COUNTS.csv'),bind(O/'CONDITIONAL_SOURCE_FIT_CONTRACT_DRAFT_KO.json')],new_pixels_arrays_models_weights_current_forward_fit_PnP=0,original_modified=0,training_admission=False))
 print(json.dumps(dict(draft=bind(O/'CONDITIONAL_SOURCE_FIT_CONTRACT_DRAFT_KO.json'),seal=bind(O/'SEAL.json'),counts=summary['counts'],variants=summary['variant_counts'],by_split=summary['variant_counts_by_split']),ensure_ascii=False))

if __name__=='__main__':main()
