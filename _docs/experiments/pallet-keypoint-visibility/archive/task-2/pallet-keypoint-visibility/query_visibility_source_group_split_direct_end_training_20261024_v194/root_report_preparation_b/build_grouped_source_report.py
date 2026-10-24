"""V194 source-only closed-summary report builder; stdlib, no model/array reader."""
from pathlib import Path
import argparse,copy,hashlib,html,json,math,os,stat,traceback
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PROTOCOL=H/'PROTOCOL_BEFORE_TERMINAL_GROUPED_SOURCE_REPORT_INPUTS.json'
SCOPE=H/'ROOT_REPORT_CODE_PREPARATION_SCOPE.json'
AUDITOR=V/'independent_grouped_source_fit_review_preparation_a/audit_saved_grouped_source_fit.py'
AUDIT_PROTO=AUDITOR.with_name('PROTOCOL_BEFORE_GROUPED_SOURCE_FIT_OUTPUT_BODIES.json')
FIT_PROTO=V/'grouped_fit_preparation_b/PROTOCOL_BEFORE_GROUPED_SOURCE_CACHE_ARRAY_OR_FIT_BODIES.json'
FIT_COMPLETE=V/'runs/C56_C57_DIRECT_END_GROUPED_SOURCE_FIXED8000/COMPLETE.json'
REVIEW=V/'independent_grouped_source_fit_review_a/actual_review_v1/REVIEW.json'
ARMS=('C56_RAW96_DIRECT_END_GROUPED','C57_RAW96_PLUS_FROZEN_SAM_DIRECT_END_GROUPED')
DOMAINS={'SYNTH_AUG65536':65536,'PUBLIC_TRAIN1480':1480,'CF3816':3816,'DIRECT_CONTROL':3144}
U={'SYNTH_AUG65536':1264,'PUBLIC_TRAIN1480':194,'CF3816':360,'DIRECT_CONTROL':112}
ROLES={'ALL':(3144,248,124),'TRAIN':(2784,218,109),'HOLDOUT':(360,30,15)}
GALLERY=dict(path=str(P/'query_visibility_source_end_pairs_opposite_sham_20261024_v193/source_pair_visual_readout_v2_caption_fit/PORTABLE_GALLERY_KO.md'),sha256='5158806b87ff697ce76092be4c207650c50d9fed7e419b28194c74b3e455c35a')
FIELDS=['ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html','ROOT_SOURCE_REPORT_BUILD_COMPLETE.json']

def need(x,msg):
 if not bool(x):raise AssertionError(msg)
def safe(path):
 p=Path(path);need(p.is_absolute()and '..'not in p.parts and all('v112'not in q.casefold()for q in p.parts),'Lexical forbidden-name/absolute-path check before access')
 need(p.is_relative_to(P)and not any(q in str(p).casefold()for q in ('current223','_v169/','_v177/','_v178/','_v184/','_v191/','_v195/','_v182/','synthetic_val')),'Exact source project paths only')
 return p

def opened(path):
 p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
  last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(last).st_mode),'Regular nofollow source metadata');return os.fdopen(last,'rb')
def bytes_text(path):
 p=safe(path);need(p.suffix in('.json','.py','.md','.html','.diff'),'Only text metadata/code, never arrays/log/checkpoints/images')
 with opened(p)as f:return f.read()
def bind(path):return dict(path=str(safe(path)),sha256=hashlib.sha256(bytes_text(path)).hexdigest())
def read(path):
 need(Path(path).suffix=='.json','JSON summaries/contracts only');return json.loads(bytes_text(path))
def jd(binding):
 need(set(binding)>={'path','sha256'},'Bound metadata object');need(bind(binding['path'])=={k:binding[k]for k in('path','sha256')},'Closed source JSON SHA');return read(binding['path'])
def write(path,value):
 p=safe(path);need(p.parent in(H,V),'Fixed preparation or report output only')
 with p.open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def text(path,value):
 p=safe(path);need(p.parent==V,'Fixed top source report output')
 with p.open('x')as f:f.write(value)
def rel(binding):return Path(os.path.relpath(safe(binding['path']),V)).as_posix()
def ratio(a,b):return a/b if b else None

def metric_row(model,role,authority,m):
 cm=m['CM_D_E_by_D_E_U'];need(len(cm)==2 and all(len(r)==3 for r in cm)and all(type(x)is int and x>=0 for r in cm for x in r),'2x3 integer D/E/U CM')
 nd,ne=map(sum,cm);known=nd+ne;need(type(m['rows'])is int and m['rows']>=known and m['known']==known and m['truth_UNKNOWN']==m['rows']-known and m['known_abstentions']==cm[0][2]+cm[1][2],'All rows/known/UNKNOWN/U conservation')
 expected=[dict(class_D0_E1=k,truth=sum(cm[k]),predicted_knowntruth=cm[0][k]+cm[1][k],correct=cm[k][k],precision=ratio(cm[k][k],cm[0][k]+cm[1][k]),recall=ratio(cm[k][k],sum(cm[k])))for k in(0,1)]
 need(m['classes']==expected,'Independent-audit P/R summaries match stored CM')
 return dict(model=model,cohort=role,authority=authority,N_all=m['rows'],known=known,truth_UNKNOWN=m['truth_UNKNOWN'],D=nd,E=ne,CM_D_E_by_D_E_U=cm,D_precision=expected[0]['precision'],D_recall=expected[0]['recall'],E_precision=expected[1]['precision'],E_recall=expected[1]['recall'],known_prediction_U=m['known_abstentions'])

def readout_tables(data):
 need(data['completed']is True and data['PASS']is True and data['source_only']is True and set(data['models'])==set(ARMS),'Both fixed source model readouts')
 need(data['full372_train327_holdout45']is True and data['heldout_direct45_synthetic_only_real_end_holdout0']is True and data['natural_real_generalization_or_native_gold']is False and data['selected_model']is None and data['adopted']is False,'Source-heldout authority, no adoption/generalization promotion')
 rows=[];authority=[]
 for arm in ARMS:
  a=data['models'][arm]
  for role,(n,d,e)in ROLES.items():
   r=metric_row(arm,role,'ALL_EXISTING_AUTHORITIES',a[role]);need((r['N_all'],r['D'],r['E'])==(n,d,e),'Frozen FULL/TRAIN/HOLDOUT denominator');rows.append(r)
  for auth,cohorts in a['by_authority'].items():
   for role in ROLES:authority.append(metric_row(arm,role,auth,cohorts[role]))
  for role in ROLES:
   rr=[r for r in authority if r['model']==arm and r['cohort']==role]
   need(rr and sum(r['N_all']for r in rr)==a[role]['rows']and sum(r['known']for r in rr)==a[role]['known'],'Authority stratification allrows and known totals')
   for i in range(2):
    for j in range(3):need(sum(r['CM_D_E_by_D_E_U'][i][j]for r in rr)==a[role]['CM_D_E_by_D_E_U'][i][j],'Authority CMs sum to full')
 return rows,authority

def domain_tables(done):
 rows=[];native=[]
 need(set(done['cohort_denominators'])==set(DOMAINS),'All4 source domains preserved')
 for domain,n in DOMAINS.items():
  cc=done['cohort_denominators'][domain];need(set(cc)=={'TRAIN','HOLDOUT'},'Both grouped source roles')
  need(sum(cc[k]['rows']for k in cc)==n and sum(cc[k]['original_inputU']for k in cc)==U[domain],'Full source rows/U')
  for role in('TRAIN','HOLDOUT'):
   r=cc[role];need(set(r['native'])==set(map(str,range(8)))and sum(r['native'].values())==r['rows'],'Full native8 denominator')
   rows.append(dict(domain=domain,cohort=role,N_all=r['rows'],original_input_U=r['original_inputU'],original_available=r['rows']-r['original_inputU']))
   for slot in range(8):native.append(dict(domain=domain,cohort=role,native_slot=slot,N_all=r['native'][str(slot)]))
 need(sum(DOMAINS.values())==73976,'73976 rows per arm /147952 full output rows')
 return rows,native

def fmt(x):
 if x is None:return 'UNKNOWN/미정의'
 if isinstance(x,float):return '%.6f'%x
 return str(x)
def mdtable(rows,cols):
 s='|'+'|'.join(cols)+'|\n|'+'|'.join(['---']*len(cols))+'|\n'
 return s+''.join('|'+'|'.join(fmt(r[k]).replace('|','/')for k in cols)+'|\n'for r in rows)
def htmltable(rows,cols):
 return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>'for k in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r[k]))+'</td>'for k in cols)+'</tr>'for r in rows)+'</table>'

def invented():
 tests={}
 def met(cm,unknown=0):
  n=sum(map(sum,cm));return dict(rows=n+unknown,known=n,truth_UNKNOWN=unknown,CM_D_E_by_D_E_U=cm,known_abstentions=cm[0][2]+cm[1][2],classes=[dict(class_D0_E1=k,truth=sum(cm[k]),predicted_knowntruth=cm[0][k]+cm[1][k],correct=cm[k][k],precision=ratio(cm[k][k],cm[0][k]+cm[1][k]),recall=ratio(cm[k][k],sum(cm[k])))for k in(0,1)])
 r=metric_row('TOY','TRAIN','AUTH',met([[1,1,1],[1,1,0]],2));tests['nonperfect_CM_U_unknown_P_R_separate']=r['N_all']==7 and r['truth_UNKNOWN']==2 and r['D_recall']==1/3 and r['E_recall']==.5 and r['known_prediction_U']==1
 r=metric_row('TOY','HOLDOUT','AUTH',met([[0,0,0],[0,0,0]],8));tests['empty_known_denominator_UNKNOWN_not_zero_accuracy']=r['known']==0 and r['D_precision']is None and r['E_recall']is None
 b=met([[1,0,0],[0,1,0]]);b['classes'][1]['recall']=.1
 try:metric_row('T','R','A',b);ok=False
 except AssertionError:ok=True
 tests['corrupt_reported_precision_recall_rejected']=ok
 b=met([[1,0,0],[0,1,0]]);b['truth_UNKNOWN']=9
 try:metric_row('T','R','A',b);ok=False
 except AssertionError:ok=True
 tests['unknown_denominator_drift_rejected']=ok
 populations={}
 for role,(n,d,e)in ROLES.items():populations[role]=met([[d-1,1,0],[1,e-1,0]],n-d-e)
 auth=copy.deepcopy(populations);data=dict(completed=True,PASS=True,source_only=True,models={a:dict(populations,by_authority={'AUTH':auth})for a in ARMS},full372_train327_holdout45=True,heldout_direct45_synthetic_only_real_end_holdout0=True,natural_real_generalization_or_native_gold=False,selected_model=None,adopted=False)
 rr,aa=readout_tables(data);tests['actual_source_roles_and_authority_adapter_nonperfect']=len(rr)==6 and len(aa)==6 and rr[2]['N_all']==360 and rr[2]['known']==45
 bad=copy.deepcopy(data);bad['models'][ARMS[0]]['by_authority']['AUTH']['TRAIN']=met([[218,0,0],[0,109,0]])
 try:readout_tables(bad);ok=False
 except AssertionError:ok=True
 tests['authority_coverage_not_silently_dropped']=ok
 try:safe(P/'forbidden_v112'/'unopened.json');ok=False
 except AssertionError:ok=True
 tests['forbidden_name_before_any_stat_or_read']=ok
 tests['portable_crossphase_source_gallery_relative']=rel(GALLERY).startswith('../query_visibility_source_end_pairs_opposite_sham_20261024_v193/')and '/home/'not in rel(GALLERY)
 need(all(tests.values()),'Invented report adapter checks')
 return dict(completed=True,PASS=True,invented_only=True,tests=tests,count=len(tests),actual_fit_output_or_cache_or_metric_or_array_or_model_reads=0)

def prepare():
 fit=read(FIT_PROTO);need(fit['training']['steps']==8000 and fit['training']['seed']==1810 and fit['training']['direct_rng_seed']==1811,'Frozen source recipe contract')
 tests=invented();write(H/'GENERATED_INVENTED_REPORT_CHECKS.json',tests)
 p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),checks=bind(H/'GENERATED_INVENTED_REPORT_CHECKS.json'),fit_protocol=bind(FIT_PROTO),auditor_code=bind(AUDITOR),auditor_protocol=bind(AUDIT_PROTO),fit_complete_path=str(FIT_COMPLETE),independent_review_path=str(REVIEW),source_split=fit['split_complete'],partition_QA=fit['partition_QA'],output_files=FIELDS,source_gallery=GALLERY,future_readout='Exact REVIEW.source_readout binding; models[arm].ALL/TRAIN/HOLDOUT and .by_authority[authority][role] metrics only.',constants=dict(domains=DOMAINS,inputU=U,rows_per_arm=73976,full_model_rows=147952,direct_roles=ROLES,triplets=dict(total124=124,synthetic97=97,real27=27),real_source_frames_before2_after12=[2,12]),no_arrays_models_checkpoints_logs_or_raw_inputs_reader=True,no_actual_readout_before_rootentry=True,report_is_source_TRAIN_and_grouped_HOLDOUT_only=True,not_natural_real_or_physical_native_gold=True,no_model_selection_or_adoption=True,source_design_previously_known=True,pending_separate_V195_benchmark=True)
 write(PROTOCOL,p)
 write(H/'ROOT_ACTUAL_GROUPED_SOURCE_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_saved_grouped_source_report=False,both_source_fits_and_independent_audit_completed=False,source_only=True,no_array_model_or_real_body_access=True,code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],source_fit_complete=dict(path=str(FIT_COMPLETE),sha256=None),independent_source_fit_review=dict(path=str(REVIEW),sha256=None),source_terminal_evidence=dict(path=None,sha256=None),root_confirmed_source_terminal_exit0=False,source_cache_audit=None))
 ready=dict(completed=True,PASS=True,FINAL_READY_FROZEN=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],approved_false_entry=bind(H/'ROOT_ACTUAL_GROUPED_SOURCE_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_source_fit_log_output_checkpoint_metric_cache_or_array_reads=0)
 write(H/'FINAL_READY.json',ready);print(json.dumps(ready))

def run(entry_path):
 e=read(entry_path);need(all(e.get(k)is True for k in('approved_saved_grouped_source_report','both_source_fits_and_independent_audit_completed','source_only','no_array_model_or_real_body_access','root_confirmed_source_terminal_exit0')),'Separate postterminal root report approval')
 p=read(PROTOCOL);need(e['code']==bind(__file__)==p['code']and e['protocol']==bind(PROTOCOL)and e['scope']==bind(SCOPE)==p['scope'],'Frozen report code/protocol')
 need(e['source_fit_complete']['path']==p['fit_complete_path']and e['independent_source_fit_review']['path']==p['independent_review_path'],'Exact final source COMPLETE and independent audit path')
 done=jd(e['source_fit_complete']);review=jd(e['independent_source_fit_review']);jd(e['source_terminal_evidence']);cache_review=jd(e['source_cache_audit'])
 need(done['completed']is True and done['PASS']is True and done['source_only']is True and done['from_scratch']is True and done['protocol']==p['fit_protocol']and done['steps_each']==8000 and done['total_updates']==16000 and set(done['arms'])==set(ARMS),'Both unchanged source8000 finals')
 need(done['all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs']is True and done['protected_real_COMMON_VAL_access']==0 and done['selected_model']is None and done['adopted']is False,'Closed source-only freeze/no adoption')
 need(review['completed']is True and review['PASS']is True and review['source_fit_complete']==e['source_fit_complete']and review['code']==p['auditor_code']and review['protocol']==p['auditor_protocol'],'Independent exact-fit audit closure')
 for k in('all16000_log_rows_verified','all192_and_direct32_RNG_exact_and_TRAIN_only','both_final_checkpoints115222_finite_statehash_and_AdamW8000_exact','all2_before_TRAIN_or_HOLDOUT_prediction_barrier','all147952_full_output_rows_META6_U_and_scalar_decoders_exact','all_full_vs_TRAIN_HOLDOUT_disjoint_union_and_every_output_byte_exact','direct_holdout_synthetic_only_real_end_holdout0'):need(review[k]is True,'Independent audit coverage '+k)
 need(cache_review['completed']is True and cache_review['PASS']is True,'Completed independent source cache audit')
 need(done['source_split']==p['source_split']and done['partition_QA']==p['partition_QA']and review['source_split']==p['source_split'],'Frozen grouped split provenance')
 split=jd(done['source_split']);partition=jd(done['partition_QA']);cache=jd(done['training_caches']['direct']);barrier=jd(done['all2_freeze']);source_entry=jd(done['entry'])
 need(split['completed']and split['PASS']and partition['completed']and partition['PASS']and cache['completed']and cache['PASS']and cache['rows']==3144 and cache['groups']==393 and cache['direct_known_counts']==[248,124]and cache['original_inputU']==112,'Full source split/cache metadata facts')
 need(source_entry['code']==done['code']and source_entry['protocol']==done['protocol']and barrier['completed']and barrier['checkpoints']=={a:done['arms'][a]['checkpoint']for a in ARMS},'Root actual source entry and both-state barrier')
 raw_readout=jd(review['source_readout']);metric_rows,authority_rows=readout_tables(raw_readout);domain_rows,native_rows=domain_tables(done)
 for a in ARMS:need(done['arms'][a]['updates']==8000 and done['arms'][a]['initial_state_sha']==done['same_initial_state_sha'],'Both fixed final8000 and same fresh initialization')
 for name in FIELDS:need(not (V/name).exists(),'Exclusive new top report output')
 evidence=dict(source_fit_complete=e['source_fit_complete'],source_fit_review=e['independent_source_fit_review'],source_direct_readout=review['source_readout'],source_terminal=e['source_terminal_evidence'],source_cache_audit=e['source_cache_audit'],direct_cache_complete=done['training_caches']['direct'],component_split=done['source_split'],partition_QA=done['partition_QA'],all2_freeze=done['all2_freeze'],source_fit_entry=done['entry'],source_fit_code=done['code'],source_fit_protocol=done['protocol'],source_auditor_code=p['auditor_code'],source_auditor_protocol=p['auditor_protocol'],prior_V193_source_gallery=p['source_gallery'],preserved_invented_fixture_failure=bind(V/'grouped_fit_preparation_b/provenance/BEFORE_INVENTED_TARGET_DTYPE_REPAIR/FAILURE.json'))
 result=dict(completed=True,PASS=True,phase='V194_GROUP_WITHHELD_SOURCE_DIRECT_END_FIT',source_only=True,from_scratch=True,arms=list(ARMS),fixed_recipe=dict(parameters115222=115222,updates_each8000=8000,seed1810=1810,direct_rng1811=1811,AdamW_lr=.0003,weight_decay=.0001,clip_grad_norm5=5,old_five_loss_mean_plus_balanced_direct32_weight1=True),direct_metrics=metric_rows,authority_metrics=authority_rows,all4_domain_cohort_rows_U=domain_rows,all_native8_cohort_denominators=native_rows,full_direct_counts=dict(allrows3144=3144,known372=372,D248=248,E124=124,UNKNOWN2772=2772,triplets124=124,synthetic_triplets97=97,real_weak_triplets27=27,not_independent_frames=True),source_grouped_split=dict(TRAIN_known327_D218_E109=True,HOLDOUT_known45_D30_E15=True,HOLDOUT_direct_synthetic_only=True,real_end_HOLDOUT_known=0,same_TRAIN_component_mask_for_all_old5_and_direct=True,no_seed_or_component_rebalance=True),source_real_controls=dict(prior_weak_triplets5=5,expanded_weak_triplets27=27,source_original_frames_before2_after12=[2,12],source_RGB_semantic_end_weak_not_nativephysicalgold=True,original_q_L_initializer_sampler_frozen=True,frozen_SAM_on_exact_changed_fullRGB_original_q8box=True),fit_audit_limits=review['limits'],evidence=evidence,source_design_historically_known_not_freshblind=True,old_C54_C55_not_retrospectively_validated=True,source_TRAIN_is_apparent=True,source_HOLDOUT_is_group_withheld_not_natural_real_generalization=True,pending_separate_V195_real_benchmark=True,no_C56_C57_real_improvement_claim=True,no_new_labels_or_primary_quality_gold=True,selected_model=None,adopted=False,fully_solved=False,goal_status='active_unsolved')
 write(V/'ROOT_RESULTS_KO.json',result)
 cols=['model','cohort','N_all','known','truth_UNKNOWN','D','E','CM_D_E_by_D_E_U','D_precision','D_recall','E_precision','E_recall','known_prediction_U']
 paragraphs=['# V194 원본 그룹 분리 source 학습 결과','', '같은 115222개 파라미터 모델 두 조건을 처음부터 각각 고정8000회 학습했다. RAW와 RAW+frozen SAM 조건의 구조·seed1810·AdamW(lr3e-4,wd1e-4,clip5)·기존5loss 평균과 별도 direct16D+16E 가중치1은 같다. 새로운 모델·문턱·정규화·checkpoint·arm 선택은 없다.','', '동일한 원본 component TRAIN 마스크를 기존5개 loss 풀과 direct 풀 모두에 적용했다. 전체 입력·UNKNOWN·U·q/L·native8은 보존했고 학습 인덱스만 제한했다. 양쪽 최종8000 상태를 먼저 동결한 뒤 source TRAIN/HOLDOUT 예측을 저장했으며 독립 감사가 draw16000개·checkpoint·출력 subset과 분모를 검산했다.','', 'Direct FULL은372개 약한 라벨(D248/E124), TRAIN327(D218/E109), HOLDOUT45(D30/E15)다. HOLDOUT45는 전부 합성이며 실사 끝점 holdout은0이다. source 설계가 이미 알려진 상태에서 새 fromscratch 모델에 원본 그룹을 withholding한 결과이며 fresh blind 검증이나 과거 C54/C55의 독립 검증이 아니다. TRAIN 수치는 학습자료 apparent 응답이다.','', '총124 triplet(합성97+공개 실사 약한 끝점27)은 독립124 frame이 아니다. 공개 실사 control은 기존5→27 triplet, 원source frame2→12로 늘었지만 동일 끝점/원본/물체·texture가 여러 row에서 재사용된다. 이 라벨은 source의 구성적/assistant weak end 권한이며 자연 실사 native 물리 gold가 아니다. 넓은 crop 내 팔레트 존재·점소속·native끝점 가시성은 같은 정답이 아니다.','', 'V195 동일 실사 benchmark는 별도 단계로 대기 중이다. 이 source 결과만으로 C56/C57의 실사 개선·모델 채택·전체 목표 해결을 주장하지 않는다.','', '## Source direct TRAIN/HOLDOUT와 FULL','',mdtable(metric_rows,cols),'','## 권한별 별도 집계','',mdtable(authority_rows,['model','cohort','authority','N_all','known','truth_UNKNOWN','CM_D_E_by_D_E_U','D_precision','D_recall','E_precision','E_recall']),'','## 전체4domain / native8 / 원U','', '각 arm 전체73976 rows(두 arm147952)를 유지한다. HOLDOUT/UNKNOWN은 삭제하거나 가짜0으로 채우지 않는다. 아래 분모는 두 arm이 공유한다.','',mdtable(domain_rows,['domain','cohort','N_all','original_input_U','original_available']),'',mdtable(native_rows,['domain','cohort','native_slot','N_all']),'','## 근거와 보존 기록','']
 for key,binding in evidence.items():paragraphs+=['- ['+key+']('+rel(binding)+') · SHA256 '+binding['sha256']]
 paragraphs+=['','독립 감사의 한계:','']+['- '+x for x in review['limits']]+['','학습/관측/실행 원문과 실패 기록은 보존한다. 이 보고서는 배열·checkpoint·모델·새 이미지·실사 평가를 읽거나 실행하지 않고 닫힌 JSON 요약만 표시했다. HTML은 생성했으며 브라우저 렌더/추가 이미지 실제 확인을 주장하지 않는다.','']
 md='\n'.join(paragraphs);need('/home/'not in md and 'file://'not in md,'Portable Markdown contains only relative source links');text(V/'ROOT_REPORT_KO.md',md);text(V/'PORTABLE_REPORT_KO.md',md)
 links='<ul>'+''.join('<li><a href="'+html.escape(rel(b))+'">'+html.escape(k)+'</a> · '+b['sha256']+'</li>'for k,b in evidence.items())+'</ul>'
 body='<!doctype html><meta charset="utf-8"><style>body{font:17px sans-serif;margin:24px}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #bbb;padding:5px}pre{white-space:pre-wrap}</style><h1>V194 source 원본 그룹 분리 학습</h1><p>Source TRAIN apparent와 group-withheld HOLDOUT만 보고한다. HOLDOUT direct45는 합성뿐이며 real-end holdout0이다. 별도 V195 실사 benchmark는 대기 중이며 새 실사 개선·채택·해결 주장은 없다.</p>'+htmltable(metric_rows,cols)+'<h2>Authority별</h2>'+htmltable(authority_rows,['model','cohort','authority','N_all','known','truth_UNKNOWN','CM_D_E_by_D_E_U'])+'<h2>전체 source 분모</h2>'+htmltable(domain_rows,['domain','cohort','N_all','original_input_U'])+'<h2>근거</h2>'+links+'<p><a href="PORTABLE_REPORT_KO.md">전체 portable 보고서 / native8 / 한계</a></p><pre>'+html.escape(md)+'</pre>'
 text(V/'ROOT_READOUT_KO.html',body)
 outputs={n:bind(V/n)for n in FIELDS[:-1]};write(V/'ROOT_SOURCE_REPORT_BUILD_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=e['protocol'],entry=bind(entry_path),inputs=evidence,outputs=outputs,no_array_checkpoint_log_model_pixel_or_evaluation_body_reads=True,HTML_browser_rendered=False,selected_model=None,adopted=False,fully_solved=False));print(json.dumps(dict(completed=True,PASS=True,complete=bind(V/'ROOT_SOURCE_REPORT_BUILD_COMPLETE.json'))))

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare()if a.prepare else run(a.entry)
