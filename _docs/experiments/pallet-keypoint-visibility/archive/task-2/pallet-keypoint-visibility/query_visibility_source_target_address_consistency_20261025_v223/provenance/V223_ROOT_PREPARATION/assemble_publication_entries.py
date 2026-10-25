"""Create exact root publication entries for each separately invoked, authorized step."""
import sys,json,hashlib
from pathlib import Path
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');R=P/'publication_root_preparation'
S=P/'query_visibility_source_target_address_consistency_20261025_v223'
H=R/'V223_PUBLICATION_PREPARATION_C';A=R/'V223_PUBLICATION_ROOT_OPERATIONS'
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
 'source_only':'V223 C68/C69 SOURCE split-target comparison finished; implementation QA PASS; outcome from frozen root report; no current223 inference or model adoption; metadata exception disclosed.',
 'single_nonforce_push_only':True,'automatic_retry':False})
e={'approved_actual_operation':mode,'automatic_retry':False,'publisher':bind(H/'publish_completed_v223.py'),
 'runbook':bind(H/'RUNBOOK_KO.md'),'root_actual_scope':bind(scope),'initial_remote_main':'f467431f01e8b3b60b1a45fa8814af0dd7dff142','repository':str(REPO)}
if mode=='stage':
 fit=read(S/'paired_source_fit_v1/COMPLETE.json')
 outcome=read(S/'ROOT_PHASE_RESULT.json')
 ip=outcome['publication_prefixes']['index_utf8']
 e.update(approved_actual_enumeration_or_copy=True,source_root=str(S),
  phase_result={'binding':bind(S/'ROOT_PHASE_RESULT.json'),'expected_fields':{'completed':True,'phase':'V223','scientific_outcome':outcome['scientific_outcome']}},
  final_freeze={'binding':bind(S/'ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json'),'expected_fields':{'completed':True,'frozen':True,'all_scientific_phase_files_final_no_more_writes':True}},
  paired_fit_complete={'binding':bind(S/'paired_source_fit_v1/COMPLETE.json'),'expected_fields':{'completed':True,'PASS':True,'steps_each':1100,'source_only':True}},
  all2_freeze={'binding':bind(S/'paired_source_fit_v1/ALL2_FINAL_FROZEN_BEFORE_ANY_DEV_OR_REAL_OUTPUT.json'),'expected_fields':{'completed':True,'PASS':True,'DEV_or_real_output_emitted':False,'selected_arm':None}},
  owned_binary_whitelist=[fit['arms'][arm]['checkpoint'] for arm in ['C68SHARED_POINT_PART_LOGIT','C69SEPARATE_POINT_PART_LOGITS']],
  excluded_RAW_paths=[],source_caps={'files':1200,'bytes':180*1024**2},
  history_before={name:bind(PUB/name if name=='EXPERIMENT_INDEX_THROUGH_V222_KO.md' else REPO/name) for name in ['EXPERIMENT_INDEX_THROUGH_V222_KO.md','readme.md',PREFIX+'README.md',PREFIX+'EXPERIMENT_INDEX_KO.md']},
  publication_prefixes=outcome['publication_prefixes'],
  portable_pages=['EXPERIMENT_INDEX_THROUGH_V223_KO.md']+[ARC+x for x in ['PORTABLE_REPORT_KO.md','ROOT_REPORT_KO.md','saved_source_comparison_v1/PORTABLE_GALLERY_KO.md']],
  commit_message='docs: publish V223 point-part routing contrast, source results and crops')
else:
 d=read(R/'V223_PUBLICATION_STAGE_C/COMPLETE.json')
 e.update(stage_complete=bind(R/'V223_PUBLICATION_STAGE_C/COMPLETE.json'),push_plan=d['push_plan'],manifest=d['manifest'])
 if mode=='push':e.update(approved_single_nonforce_main_push=True,byte_QA=bind(R/'ROOT_V223_SOURCE_AND_PLAN_BYTE_QA.json'))
 if mode=='post':
  tx=R/'V223_MAIN_PUSH_ROOT_TRANSACTION/COMPLETE.json'
  e.update(transaction=bind(tx),expected_commit=read(tx)['commit'],HTTP_HEAD_targets=[
   {'role':'INDEX','path':PREFIX+'EXPERIMENT_INDEX_THROUGH_V223_KO.md'},
   {'role':'ROOT_REPORT','path':PREFIX+ARC+'PORTABLE_REPORT_KO.md'},
   {'role':'COUNTS','path':PREFIX+ARC+'saved_source_comparison_v1/SOURCE_FAMILIES_SEPARATE_CM_PRECISION_RECALL.csv'},
   {'role':'SUCCESS_CROP','path':PREFIX+ARC+'saved_source_comparison_v1/SOURCE_DEV_F05_C000000_S0_N4.png'},
   {'role':'FAILURE_CROP','path':PREFIX+ARC+'saved_source_comparison_v1/SOURCE_DEV_F05_C000000_S0_N1.png'},
   {'role':'MODEL_QA','path':PREFIX+ARC+'provenance/V223_ACTUAL_MODEL_QA_A/actual_review_v1/REVIEW.json'}])
entry=A/('ROOT_ACTUAL_'+mode.upper()+'_ENTRY.json');new(entry,e)
print(json.dumps({'completed':True,'entry':bind(entry),'mode':mode}))
