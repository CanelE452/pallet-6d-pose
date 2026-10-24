"""Create exact root publication entries for each separately invoked, authorized step."""
import sys,json,hashlib
from pathlib import Path
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');R=P/'publication_root_preparation'
S=P/'query_visibility_expected_part_counterfactual_source_bridge_20261025_v222'
H=R/'V222_PUBLICATION_PREPARATION_C';A=R/'V222_PUBLICATION_ROOT_OPERATIONS'
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
PREFIX='_docs/experiments/pallet-keypoint-visibility/'
PUB=REPO/PREFIX;ARC='archive/task-2/pallet-keypoint-visibility/'+S.name+'/'
def bind(p):
 if 'v112' in str(p).lower():raise RuntimeError('forbidden path')
 b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(p.read_bytes())
def new(p,o):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
mode=sys.argv[1];assert mode in ('stage','verify','push','post')
scope=A/('ROOT_ACTUAL_'+mode.upper()+'_SCOPE.json')
new(scope,{'completed':True,'approved_actual_'+mode:True,
 'human_authorization':'User explicitly requested all trials/code/results/tables/visualizations published to main continuously; current goal remains active.',
 'source_only':'V222 C66/C67 finished; implementation QA PASS; SOURCE generalization not improved; no protected inference or model adoption.',
 'single_nonforce_push_only':True,'automatic_retry':False})
e={'approved_actual_operation':mode,'automatic_retry':False,'publisher':bind(H/'publish_completed_v222.py'),
 'runbook':bind(H/'RUNBOOK_KO.md'),'root_actual_scope':bind(scope),'initial_remote_main':'8e8cda5a8d6e0ce8cc02426fa633ceec36d02a9f','repository':str(REPO)}
if mode=='stage':
 fit=read(S/'paired_source_fit_v1/COMPLETE.json')
 ip='# V222 — 예상 부위 COVER/SHAM 감독: SOURCE 일반화 개선 미확인\n\n같은 C65에서 C66/C67을 각1,100 update 학습했다. 추가 감독은 FIT39묶음 모두 맞혔지만 CF DEV 전체 정답19/27→17/27, natural weak DEV11/12→9/12였다. DEV9묶음 세 상태 완전정답1→2만으로 성공이라 하지 않는다. GT-derived 공개 학습 query의 finite-part weak 감독이며 native점 gold나 생산 초기query 정합성을 증명하지 않았다. 현재223 새추론·PnP·모델채택0, 최신 실사 평가V220 유지.\n\n'
 for label,path in [('V222 보고서',ARC+'PORTABLE_REPORT_KO.md'),('분모별 CM·precision·recall 표',ARC+'saved_source_comparison_v2/SOURCE_FAMILIES_SEPARATE_CM_PRECISION_RECALL.csv'),('검증9묶음 실제 crop 전부',ARC+'saved_source_comparison_v2/PORTABLE_GALLERY_KO.md'),('원문 코드·실패·QA 보존 목록',ARC+'provenance/ARCHIVE_MANIFEST.json'),('완결 결과',ARC+'ROOT_PHASE_RESULT.json')]:ip+='- ['+label+']('+path+')\n'
 ip+='\n---\n\n'
 e.update(approved_actual_enumeration_or_copy=True,source_root=str(S),
  phase_result={'binding':bind(S/'ROOT_PHASE_RESULT.json'),'expected_fields':{'completed':True,'phase':'V222','scientific_outcome':'SOURCE_GENERALIZATION_NOT_IMPROVED'}},
  final_freeze={'binding':bind(S/'ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json'),'expected_fields':{'completed':True,'frozen':True,'all_scientific_phase_files_final_no_more_writes':True}},
  paired_fit_complete={'binding':bind(S/'paired_source_fit_v1/COMPLETE.json'),'expected_fields':{'completed':True,'PASS':True,'steps_each':1100,'source_only':True}},
  all2_freeze={'binding':bind(S/'paired_source_fit_v1/ALL2_FINAL_FROZEN_BEFORE_ANY_DEV_OR_REAL_OUTPUT.json'),'expected_fields':{'completed':True,'PASS':True,'DEV_or_real_output_emitted':False,'selected_arm':None}},
  owned_binary_whitelist=[fit['arms'][arm]['checkpoint'] for arm in ['C66CONTINUATION_CONTROL','C67PLUS_EXPECTED_PART_CF_WEAK']],
  excluded_RAW_paths=[],source_caps={'files':1200,'bytes':180*1024**2},
  history_before={name:bind(PUB/name if name=='EXPERIMENT_INDEX_THROUGH_V221_KO.md' else REPO/name) for name in ['EXPERIMENT_INDEX_THROUGH_V221_KO.md','readme.md',PREFIX+'README.md',PREFIX+'EXPERIMENT_INDEX_KO.md']},
  publication_prefixes={'index_utf8':ip,
   'root_readme_utf8':'V222 공개 SOURCE 대조학습·실패 결과·crop·코드/QA를 추가했습니다. SOURCE 일반화 개선 미확인, 현재223 새추론·모델채택0입니다. [최신 실험 인덱스]('+PREFIX+'EXPERIMENT_INDEX_THROUGH_V222_KO.md).\n\n',
   'docs_readme_utf8':'V222: 예상 부위 COVER/SHAM 감독을 실제 학습했으나 SOURCE 검증 개선은 확인되지 않았습니다. [표·전체 crop·실행/QA 원문](EXPERIMENT_INDEX_THROUGH_V222_KO.md). 최신 실사 비교는V220입니다.\n\n'},
  portable_pages=['EXPERIMENT_INDEX_THROUGH_V222_KO.md']+[ARC+x for x in ['PORTABLE_REPORT_KO.md','ROOT_REPORT_KO.md','saved_source_comparison_v1/PORTABLE_GALLERY_KO.md','saved_source_comparison_v2/PORTABLE_GALLERY_KO.md','expected_part_counterfactual_pairs_v1/PORTABLE_GALLERY_KO.md','expected_part_counterfactual_pairs_v1/index.html']],
  commit_message='docs: publish V222 paired source training, negative generalization results and crops')
else:
 d=read(R/'V222_PUBLICATION_STAGE_C/COMPLETE.json')
 e.update(stage_complete=bind(R/'V222_PUBLICATION_STAGE_C/COMPLETE.json'),push_plan=d['push_plan'],manifest=d['manifest'])
 if mode=='push':e.update(approved_single_nonforce_main_push=True,byte_QA=bind(R/'ROOT_V222_SOURCE_AND_PLAN_BYTE_QA.json'))
 if mode=='post':
  tx=R/'V222_MAIN_PUSH_ROOT_TRANSACTION/COMPLETE.json'
  e.update(transaction=bind(tx),expected_commit=read(tx)['commit'],HTTP_HEAD_targets=[
   {'role':'INDEX','path':PREFIX+'EXPERIMENT_INDEX_THROUGH_V222_KO.md'},
   {'role':'ROOT_REPORT','path':PREFIX+ARC+'PORTABLE_REPORT_KO.md'},
   {'role':'COUNTS','path':PREFIX+ARC+'saved_source_comparison_v2/SOURCE_FAMILIES_SEPARATE_CM_PRECISION_RECALL.csv'},
   {'role':'SUCCESS_CROP','path':PREFIX+ARC+'saved_source_comparison_v2/SOURCE_DEV_F05_C000000_S0_N4.png'},
   {'role':'FAILURE_CROP','path':PREFIX+ARC+'saved_source_comparison_v2/SOURCE_DEV_F05_C000000_S0_N1.png'},
   {'role':'MODEL_QA','path':PREFIX+ARC+'provenance/V222_ACTUAL_MODEL_QA_A/actual_review_v1/REVIEW.json'}])
entry=A/('ROOT_ACTUAL_'+mode.upper()+'_ENTRY.json');new(entry,e)
print(json.dumps({'completed':True,'entry':bind(entry),'mode':mode}))
