"""V195 closed JSON report only; prepare reads code/contracts + invented fixtures."""

from pathlib import Path
from html.parser import HTMLParser
import argparse,ast,copy,hashlib,html,json,os,re,stat,traceback
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent

INPUT_SCOPE=H/'PROSPECTIVE_REPORT_INPUT_SCOPE_BEFORE_SAVED_SUMMARY_READ.json'
PROTOCOL=H/'PROTOCOL_BEFORE_FINAL_REPORT_METADATA_AND_WRITES.json';CHECKS=H/'GENERATED_INVENTED_REPORT_CHECKS.json'
OUTPUTS=['ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html','ROOT_PHASE_RESULT.json']
INLINE=[0,2,256,300,840,996];MODELS=['v61','legacy','C52','C53','C54','C55','C56','C57'];NEW=['C56','C57']
SOURCE=P/'query_visibility_source_group_split_direct_end_training_20261024_v194'
EVAL_PROTOCOL=V/'evaluation_preparation_b/PROTOCOL_BEFORE_REAL_ANNOTATION_OR_SAVED_SCORE_BODIES.json'
VISUAL_PROTOCOL=V/'fixed_visual_preparation_b/PROTOCOL_BEFORE_SAVED_EVALUATION_OR_CROP_BODIES.json'
ROLES={'gallery_complete':V/'fixed_direct_end_real_visual_v1/COMPLETE.json','summary':V/'fixed_direct_end_real_visual_v1/SUMMARY_AND_TABLES.json','image_manifest':V/'fixed_direct_end_real_visual_v1/IMAGE_MANIFEST.json','evaluation_complete':V/'saved_real_head_evaluation_v1/COMPLETE.json','current_saved_summary':V/'saved_real_head_evaluation_v1/current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json','prediction_seal':V/'frozen_C56_C57_real_predictions_v1/PREDICTION_SEAL.json','independent_numeric':V/'independent_saved_result_review_preparation_a/actual_review_v1/REVIEW.json','root_visual':None,'source_fit_integrity':SOURCE/'independent_grouped_source_fit_review_a/actual_review_v1/REVIEW.json'}

def require(x,m):
 if not bool(x):raise AssertionError(m)

def safe(path):
 p=Path(path).absolute();require(P in p.parents and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts),'Explicit safe path before access');return p

def opened(path,write=False):
 p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for x in p.parts[1:-1]:
   nxt=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
  last=os.open(p.name,(os.O_WRONLY|os.O_CREAT|os.O_EXCL if write else os.O_RDONLY)|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 require(stat.S_ISREG(os.fstat(last).st_mode),'Ordinary file/no symlink');return os.fdopen(last,'wb' if write else 'rb')

def sha(p):
 h=hashlib.sha256()
 with opened(p)as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def bind(p):return dict(path=str(safe(p)),sha256=sha(p))

def checked(b):require(bind(b['path'])==b,'Binding changed');return safe(b['path'])

def read(p):
 with opened(p)as f:return json.load(f)

def text(p,s):
 with opened(p,True)as f:f.write(s.encode())

def write(p,d):text(p,json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def pct(v):return '미정' if v is None else '%.1f%%'%(100*v)

def cm(r):return [[r['D_D'],r['D_E'],r['D_U']],[r['E_D'],r['E_E'],r['E_U']]]

def short_rows(rows):
 return [{'모델':r['model'],'N':r['N'],'D행 [D,E,U]':str(cm(r)[0]),'E행 [D,E,U]':str(cm(r)[1]),'D 정밀도':pct(r['D_precision']),'D 재현율':pct(r['D_recall']),'E 정밀도':pct(r['E_precision']),'E 재현율':pct(r['E_recall']),'U':r['U'],'U율':pct(r['U']/r['N']) if r['N'] else '미정'} for r in rows]

def mdtable(rows):
 if not rows:return '없음\n'
 keys=list(rows[0]);s='|'+'|'.join(keys)+'|\n|'+'|'.join(['---']*len(keys))+'|\n'
 for row in rows:s+='|'+'|'.join(str(row[k]).replace('|','/') for k in keys)+'|\n'
 return s

def htmltable(rows):
 if not rows:return '<p>없음</p>'
 keys=list(rows[0]);return '<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</tbody></table>'

def rel(p):return Path(os.path.relpath(safe(p),V)).as_posix()
class Refs(HTMLParser):
 def __init__(self):super().__init__();self.refs=[]
 def handle_starttag(self,t,a):self.refs.extend(v for k,v in a if k in('href','src')and v)
def linkcheck(content,html_mode=False,pending=()):
 require('/home/'not in content and 'file://'not in content,'Portable paths only')
 if html_mode:parser=Refs();parser.feed(content);refs=parser.refs
 else:refs=re.findall(r'\]\(([^)]+)\)',content)
 for s in refs:
  require(not s.startswith('/')and ':'not in s and all('v112'not in q.casefold()for q in Path(s).parts),'Relative safe name before normalization/access')
  path=safe(os.path.normpath(str(V/s)))
  if s in pending:continue
  with opened(path):pass # Existence/type only; no image read/hash/header/decode.
 return refs

def validate_metric(r):
 matrix=cm(r);require(all(type(v)is int and v>=0 for row in matrix for v in row),'Integer2x3 CM')
 d,e=map(sum,matrix);require(r['N']==d+e and r['D_support']==d and r['E_support']==e and r['U']==r['D_U']+r['E_U'],'Fixed full truth denominator/U')
 ratio=lambda a,b:a/b if b else None
 expected=dict(D_precision=ratio(r['D_D'],r['D_D']+r['E_D']),D_recall=ratio(r['D_D'],d),E_precision=ratio(r['E_E'],r['D_E']+r['E_E']),E_recall=ratio(r['E_E'],e))
 require(all(r[k]==v for k,v in expected.items()),'Stored summary P/R matches CM')
 return True

def actual_view_count(qa):
 for key in('viewed_count','actual_tool_viewed_card_count','cards_actually_viewed'):
  if type(qa.get(key))is int:return qa[key]
 for key in('actual_viewed_cards','viewed_cards','actual_tool_viewed_cards'):
  if isinstance(qa.get(key),list):return len(qa[key])
 return 0

def validate(inputs,bindings):
 s=inputs['summary'];g=inputs['gallery_complete'];m=inputs['image_manifest'];ev=inputs['evaluation_complete'];audit=inputs['independent_numeric'];current=inputs['current_saved_summary'];pred=inputs['prediction_seal']
 for role in('gallery_complete','evaluation_complete','independent_numeric','root_visual','source_fit_integrity'):require(inputs[role]['completed']is True and inputs[role]['PASS']is True,'Actual completed QA '+role)
 require(s['completed']and s['all_models']==MODELS and s['selected_model']is None and s['adopted']is False and s['fully_solved']is False,'Eight fixed models, no adopted/best arm')
 require(s['source_evaluation']==bindings['evaluation_complete']and g['summary']==bindings['summary']and g['manifest']==bindings['image_manifest']and ev['prediction_seal']==bindings['prediction_seal'],'Summary/gallery/evaluation/prediction closure')
 require(audit['complete']==bindings['evaluation_complete']and audit['completed']and audit['PASS'],'Independent A exact-evaluation numerical receipt')
 require(inputs['root_visual']['gallery_complete']==bindings['gallery_complete']and actual_view_count(inputs['root_visual'])>0,'Root actual new-gallery image-tool QA')
 require(pred['completed']and pred['PASS']and pred['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads']and inputs['source_fit_integrity']['source_fit_complete']==pred['source_fit_complete'],'Both new predictions and exact grouped source-fit integrity')
 require(ev['models']==MODELS and ev['old_six_primary_metrics_exact']is True and ev['old_annotations_outputs_and_inputs_unchanged']is True,'Exact reused old6 baseline routes')
 require(s['current_primary']['N']==1237 and s['current_primary']['D']==1147 and s['current_primary']['E']==90 and s['current_all1784']['N']==1784 and s['current_all1784']['input_U']==45,'Original primary1237/current1784/U45')
 require(g['selected_unique_cases']==s['selected_images']==len(m['images'])==68 and g['all1784_META_qL_native_U_preserved']and g['all_selected_original_RGB96_valid96_png_parity'],'Same68 and original q/L/native/crop parity')
 require(len(s['current_primary']['rows'])==24 and {r['model']for r in s['current_primary']['rows']}==set(MODELS),'All8 by3 group table')
 for r in s['current_primary']['rows']:
  validate_metric(r);h=audit['current']['primary'][r['group']][r['model']]
  require(cm(r)==h['CM_D_E_by_D_E_U']and r['N']==h['N']and all(r[k]==h[k]for k in('D_precision','D_recall','E_precision','E_recall')),'Independent scalar headline vs summary')
 require(sum(r['all_original_slots']for r in s['current_all1784']['materials_native8'])==1784 and {r['native_slot']for r in s['current_all1784']['materials_native8']}==set(range(8)),'Full native8')
 require(current['self_all136']is True and current['self_pairedflag21']is True and current['reference_LE3_humanD461']is True and current['manual_near_D376']is True,'Separate frozen self/reference/manual cohorts')
 p=s['public_weak_owner_only'];require(p['rows']==360 and p['coverage']['known_center204']==204 and p['coverage']['unknown_center156']==156 and p['coverage']['original_input_U']==108 and p['legacy_baseline']['not_imputed']is True,'Public weak204/156 separate, legacy missing')
 return True

def compose(inputs,bindings):
 validate(inputs,bindings);s=inputs['summary'];pt=s['current_primary']['rows'];main=[r for r in pt if r['group']=='ALL'];by={r['model']:r for r in main};material=[r for r in pt if r['model']in NEW and r['group']!='ALL'];current=inputs['current_saved_summary'];native=s['current_all1784']['materials_native8'];weak=s['public_weak_headline_table']
 def self_table(cohort):
  out=[]
  for model in MODELS:
   rr=[r for r in s['self_separate']if r['cohort']==cohort and r['model']==model]
   out.append({'모델':model,'별도 human S N':sum(r['human_S_support']for r in rr),'D 응답':sum(r['visible_response']for r in rr),'binary hidden 응답':sum(r['binary_hidden_response']for r in rr),'미결정':sum(r['unresolved']for r in rr)})
  return out
 self21=self_table('paired_same_query_RGB');self136=self_table('same_slot_pipeline');require({r['별도 human S N']for r in self21}=={21}and {r['별도 human S N']for r in self136}=={136},'SELF21 D_vs_ES vs full136')
 ref=[]
 for tag in NEW:
  rr=[r for r in current['tables']['support_joint_owner_and_reference_associations']if r['group']=='ALL'and r['error_bin']=='LE3'and r['human_semantic_state']=='DIRECT_VISIBLE'and r['model']==tag];require(len(rr)==1 and rr[0]['N']==461,'Saved existing reference LE3 humanD461')
  r=rr[0];h=inputs['independent_numeric']['current']['all_registered_humanD_existing_reference_LE3_association'][tag];require((r['N'],r['primary_D'],r['primary_E'],r['primary_U'])==(h['N'],h['D'],h['E'],h['U']),'Reference counts separately independently audited')
  ref.append({'모델':tag,'기존reference≤3 humanD N':461,'D':r['primary_D'],'E':r['primary_E'],'U':r['primary_U']})
 material_rows=[{'모델':r['model'],'재질':'WOOD'if r['group']=='WOOD_PRIMARY105'else'SQUARE',**short_rows([r])[0]}for r in material]
 weakrows=[{'모델':r['model'],'T 정밀도':pct(r['T_precision_known']),'T 재현율':pct(r['T_recall_including_U']),'non-T 재현율':pct(r['nonT_recall_including_U']),'ownerF 정밀도':pct(r['ownerF_precision_known']),'ownerF 재현율':pct(r['ownerF_recall_including_U'])}for r in weak]
 captions={0:'동일 고정 원 입력과 모든8모델 출력을 함께 비교한다. 무효 padding을 자연 배경으로 해석하지 않는다.',2:'사전에 고정된 원 crop이다. D/E 응답과 기존 참조 연관을 분리하며 local crop만으로 실제 원인을 단정하지 않는다.',256:'이전과 같은 고정 ID다. 새 출력에 따라 사례를 바꾸거나 중심을 보정하지 않았다.',300:'기존 참조와 원 query의 관계는 설명용 연관이다. E 응답만으로 queried-end의 물리 외부가림을 인증하지 않는다.',840:'고정 SQUARE 사례다. 중심 owner/주변 팔레트 존재와 native 끝부위 가시성을 동일시하지 않는다.',996:'기존 고정 사례의 새 두 응답을 그대로 표시한다. 작은 참조 오차도 물리 pixel gold나 인과 분리 증거가 아니다.'}
 images={r['original_global_index']:r for r in inputs['image_manifest']['images']};inline=[dict(original_global_index=i,card=images[i]['card'],caption_KO=captions[i])for i in INLINE if i in images];missing=[i for i in INLINE if i not in images]
 lead='동일 primary1237(D1147/E90)에서 C56의 D/E 재현율은 '+pct(by['C56']['D_recall'])+'/'+pct(by['C56']['E_recall'])+', C57은 '+pct(by['C57']['D_recall'])+'/'+pct(by['C57']['E_recall'])+'다. 아래에 U와 두 재질·기존6모델을 함께 보존했다. 어느 arm도 채택하거나 best로 고르지 않았으며 전체 목표는 미해결이다.'
 limits=['C56/C57은 V194의 새 fromscratch 고정8000/115222 모델이며 source-only 학습에는 공개 실사와 YCB 기반 control도 포함된다. synthetic-only 학습이라고 부르지 않는다.',
 'C54/C55 대비 직접 control5→27 확대와 모든 loss 풀의 원본 component TRAIN 분리 등 여러 학습 조건이 함께 달라졌다. 이 평가로 특정 감독·SAM·분할·자료 증가의 단일 인과 효과를 분리하지 않는다.',
 'V194 direct FULL372(D248/E124),TRAIN327(D218/E109),HOLDOUT45(D30/E15)를 구분한다. HOLDOUT45는 전부 합성이며 real-end holdout0이다.124triplet(합성97+실사weak27)은124독립 frame이 아니다. 이미 알려진 source 설계 안의 grouped holdout이며 자연실사/unseen blind/nativephysical gold가 아니다.',
 '기존 인간 D/E/S 의미 권위를 유지한다. 점소속·넓은crop 팔레트 존재·broad support2·직접 weak끝점 감독을 native 물리 gold로 혼합하지 않는다. quality/owner/오차로 primary를 걸러내지 않았다.',
 'current223/publicheld45는 반복 개발 노출되어 fresh 독립 검증이 아니다. legacy의 과거 realCAL 이력을 보존하고 public legacy 부재를 새 forward로 채우지 않았다.',
 'primary U는1237행의 모델 응답 분모다. 전체1784 factual inputU45와 public360 inputU108/centerinvalid112/known204 UNKNOWN156은 서로 다른 분모다. U0이 uncertainty나 입력 가용성 문제 해결을 뜻하지 않는다.',
 'SELF는 D_vs_ES paired21과 all/same-slot136을 별도로 보고하며 D/E primary1237에 넣지 않는다. 차이115의 원인을 query오류·inputU·물리 원인으로 추측하지 않는다. reference≤3 humanD461은 manual-click near376과 다르다.',
 '그림은 V191에서 봉인된 동일68 ID다. 새 출력 기반 firstcase나 대체 사례가 없고 대표성/유병률을 주장하지 않는다. 실제 image-tool 확인 범위는 root QA에 기록된 수만 인용하며 HTML browser 렌더를 주장하지 않는다.']
 source_link='../'+SOURCE.name+'/PORTABLE_REPORT_KO.md'
 result=dict(completed=True,phase_scope='V195 grouped source heads frozen real benchmark and saved report only',models=MODELS,current_primary_N1237_D1147_E90=s['current_primary'],table_primary8=main,table_new_C56_C57_material4=material,table_all8_material24=pt,primary_U_rate_by_model={r['model']:r['U']/r['N']for r in main},original_current_input_U_rate=45/1784,current_all1784=s['current_all1784'],native8_material_full_denominators=native,self_tables_paired21=self21,self_tables_same_slot136=self136,self_membership_authority='paired_same_query_RGB_D_vs_ES vs same_slot_pipeline_D_vs_ES; primary D/E uses D_vs_E separately',reference_error_LE3_humanD461=ref,manual_near_D_distinct={'N':376,'WOOD':153,'SQUARE':223},public_weak_center_only=s['public_weak_owner_only'],table_public_weak7=weak,old_six_saved_routes_replayed_exact=True,source_training=dict(phase='V194',source_only_includes_public_real=True,from_scratch=True,parameters=115222,updates_each=8000,total_updates=16000,full_direct372_D248_E124=True,TRAIN327_D218_E109=True,HOLDOUT45_D30_E15_synthetic_only=True,real_end_holdout0=True,triplets124_not_independentframes=True,multiple_changed_conditions_not_causal_isolation=True),six_fixed_inline_cards=inline,static_inline_missing_ids_no_replacement=missing,actual_visual_scope=dict(root_viewed_count=actual_view_count(inputs['root_visual']),all68_or1784_actually_viewed=inputs['root_visual'].get('all68_or1784_actually_viewed',False),browser_rendered=False),evidence=bindings,limitations_KO=limits,source_summary_only_no_new_metric_kernel_or_data_run=True,selected_model=None,adopted=False,fully_solved=False,goal_status='active_unsolved')
 md=['# V195: 원본 그룹 분리 direct-end 모델의 동일 실사 평가','',lead,'','원 current223×8=1784, public45×8=360, 등록 WOOD105/SQUARE118과 native0..7의 원 q/L·crop·가용성을 유지했다. primary는 기존 동일-query 인간 D1147/E90 총1237행이다. 원입력U45/1784와 primary 모델 U/N을 구분한다.','','## 동일8모델 primary CM/P/R/U','','CM 행은 인간 D/E, 열은 모델 D/E/U다. 정밀도는 해당 결정 예측, 재현율은 U를 포함한 해당 인간 전체 분모다. 기존 E 응답은 물리 원인 인증이 아니다.','',mdtable(short_rows(main)),'','## 새 두 모델의 WOOD/SQUARE 결과','',mdtable(material_rows),'','[모든8모델·재질 표](fixed_direct_end_real_visual_v1/PRIMARY_ALL8_CM_PRECISION_RECALL_U.csv) · [native8 전체분모](fixed_direct_end_real_visual_v1/ALL_ORIGINAL_NATIVE_MATERIAL_DENOMINATORS.csv)','','## self와 참조 연관은 별도다','','### SELF paired D_vs_ES21','',mdtable(self21),'','### SELF all/same-slot136','',mdtable(self136),'','차이115는 기존 paired eligibility 밖이며, 제외 사유를 새 원인으로 추정하지 않는다. 두 표는 learned-self classifier가 아니다.','',mdtable(ref),'','기존 reference≤3px humanD461과 manual-click near376(WOOD153/SQUARE223)은 다른 집단이다. 참조 연관을 물리 좌표 gold나 인과 증거로 바꾸지 않았다.','','## Public360 weak center-owner만','','known204/UNKNOWN156, inputU108/center4invalid112와 v61 자체 가용성을 유지한다. public owner/crop 약관찰을 native 외부가림 정답으로 승격하지 않으며 legacy는 기존 경로 부재로 그대로 둔다.','',mdtable(weakrows),'','## 이전과 동일한 고정 그림','', '동일68 ID 중 아래 사전 고정0/2/256/300/840/996가 있으면 표시한다. 새 출력에 따라 보충·교체하지 않는다. clean96는 원 픽셀이고 cyan q는 표시 사본에만 있다.','']
 for item in inline:md+=['### original '+str(item['original_global_index']),'',item['caption_KO'],'',f'![fixed original {item["original_global_index"]}]({rel(item["card"]["path"])})','']
 md+=['[동일68 portable 갤러리](fixed_direct_end_real_visual_v1/PORTABLE_GALLERY_KO.md) · [고정 ID 및 기존 membership](fixed_direct_end_real_visual_v1/FIXED_CELL_SELECTION.csv)','','## Source 학습과 실제 평가의 구분','','[V194 원본 그룹 분리 source 보고서]('+source_link+')를 별도로 보존한다. source TRAIN apparent와 grouped HOLDOUT은 이 실사 평가의 재점수/독립 증거가 아니다. 공개 실사 source가 포함되며 synthetic-only라고 부르지 않는다.','','## 한계와 보존 기록','']+['- '+x for x in limits]+['','[독립 A 수치 감사]('+rel(bindings['independent_numeric']['path'])+') · [root 실제 그림 QA]('+rel(bindings['root_visual']['path'])+') · [self guard 행정 수정](SELF_COHORT_GUARD_REPAIR_RECEIPT.json) · [원 실패 보존](provenance/BEFORE_SELF_COHORT_D_VS_ES_GUARD_REPAIR/FAILED_SAVED_EVALUATOR_ORIGINAL.json)','','[예측 코드](prediction_preparation_b/infer_both_frozen_grouped_direct_end_heads.py) · [평가 코드](evaluation_preparation_b/evaluate_saved_grouped_direct_end_real.py) · [고정 그림 코드](fixed_visual_preparation_b/render_fixed_grouped_direct_end_real.py) · [보고서 코드](report_preparation_b/build_top_level_readout.py)','','[수치·근거 JSON](ROOT_RESULTS_KO.json) · [phase 범위](ROOT_PHASE_RESULT.json) · [오프라인 HTML](ROOT_READOUT_KO.html)','']
 md='\n'.join(md)
 h='<h1>V195: grouped source direct-end frozen real benchmark</h1><p>'+html.escape(lead)+'</p><p>동일 primary1237 D1147/E90; 전체1784 inputU45는 다른 분모다. 새 두 모델을 best로 고르지 않았고 과거6모델 저장 경로는 그대로다.</p>'+htmltable(short_rows(main))+'<h2>C56/C57 재질</h2>'+htmltable(material_rows)+'<h2>SELF paired21 / all136</h2>'+htmltable(self21)+htmltable(self136)+'<h2>기존 reference≤3px humanD461 (manual376 별도)</h2>'+htmltable(ref)+'<h2>Public weak204/UNKNOWN156만</h2>'+htmltable(weakrows)+'<h2>이전 동일 고정 사례</h2>'
 for item in inline:h+='<figure><img src="'+html.escape(rel(item['card']['path']))+'"><figcaption>'+html.escape(str(item['original_global_index'])+': '+item['caption_KO'])+'</figcaption></figure>'
 h+='<h2>해석 한계</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>'for x in limits)+'</ul><p><a href="PORTABLE_REPORT_KO.md">전체 portable 보고서·표·코드·QA</a> · <a href="fixed_direct_end_real_visual_v1/PORTABLE_GALLERY_KO.md">동일68 갤러리</a> · <a href="'+html.escape(source_link)+'">별도 V194 source 보고서</a></p>'
 h='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport"content="width=device-width,initial-scale=1"><style>body{max-width:1600px;margin:30px auto;padding:0 24px;font:18px/1.6 system-ui}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border:1px solid #bbb;padding:7px}img{max-width:100%;height:auto}figure{margin:24px 0}</style><body>'+h+'</body></html>'
 return result,md,h
def invented_fixture():
 bindings={k:dict(path=str((V if k!='source_fit_integrity'else SOURCE)/('INVENTED_'+k+'.json')),sha256='a'*64)for k in ROLES}
 metrics=[]
 for model in MODELS:
  for group,d,e in [('ALL',1147,90),('WOOD_PRIMARY105',616,29),('SQUARE118',531,61)]:
   errors=2 if group=='ALL' else 1;r=dict(group=group,model=model,N=d+e,D_support=d,E_support=e,D_D=d-errors,D_E=errors,D_U=0,E_D=errors,E_E=e-errors,E_U=0,U=0);r.update(D_precision=r['D_D']/(r['D_D']+r['E_D']),D_recall=r['D_D']/d,E_precision=r['E_E']/(r['E_E']+r['D_E']),E_recall=r['E_E']/e);metrics.append(r)
 primary={group:{r['model']:dict(N=r['N'],CM_D_E_by_D_E_U=cm(r),**{k:r[k]for k in('D_precision','D_recall','E_precision','E_recall')})for r in metrics if r['group']==group}for group in('ALL','WOOD_PRIMARY105','SQUARE118')}
 native=[dict(material=mat,native_slot=i,all_original_slots=105 if mat=='WOOD'else118,original_input_U=0,primary_D_E_N=0,human_D=0,human_E=0,human_S=0,human_OTHER_UNKNOWN=105 if mat=='WOOD'else118)for mat in('WOOD','SQUARE')for i in range(8)]
 selfrows=[]
 for model in MODELS:
  for cohort,counts in [('paired_same_query_RGB',[10,11]),('same_slot_pipeline',[68,68])]:
   for count in counts:selfrows.append(dict(cohort=cohort,model=model,human_S_support=count,visible_response=1,binary_hidden_response=count-2,unresolved=1))
 weak=[dict(model=m,T_precision_known=.5,T_recall_including_U=.4,nonT_recall_including_U=.3,ownerF_precision_known=None if m=='v61'else.2,ownerF_recall_including_U=None if m=='v61'else.1)for m in('v61','C52','C53','C54','C55','C56','C57')]
 public=dict(rows=360,coverage=dict(known_center204=204,unknown_center156=156,original_input_U=108),legacy_baseline=dict(not_imputed=True))
 summary=dict(completed=True,all_models=MODELS,selected_model=None,adopted=False,fully_solved=False,source_evaluation=bindings['evaluation_complete'],current_primary=dict(N=1237,D=1147,E=90,rows=metrics),current_all1784=dict(N=1784,input_U=45,materials_native8=native),selected_images=68,self_separate=selfrows,public_weak_owner_only=public,public_weak_headline_table=weak)
 ids=INLINE+[i for i in range(100)if i not in INLINE][:62];images=[dict(original_global_index=i,card=dict(path=str(V/'fixed_direct_end_real_visual_v1'/('CASE_%04d'%i)/'CARD.png'),sha256='b'*64))for i in ids]
 current=dict(self_all136=True,self_pairedflag21=True,reference_LE3_humanD461=True,manual_near_D376=True,tables=dict(support_joint_owner_and_reference_associations=[dict(group='ALL',error_bin='LE3',human_semantic_state='DIRECT_VISIBLE',model=t,N=461,primary_D=450,primary_E=10,primary_U=1)for t in NEW]))
 audit=dict(completed=True,PASS=True,complete=bindings['evaluation_complete'],current=dict(primary=primary,all_registered_humanD_existing_reference_LE3_association={t:dict(N=461,D=450,E=10,U=1)for t in NEW}))
 dummyfit=dict(path=str(SOURCE/'INVENTED_FIT_COMPLETE.json'),sha256='c'*64)
 inputs=dict(summary=summary,image_manifest=dict(completed=True,images=images),gallery_complete=dict(completed=True,PASS=True,summary=bindings['summary'],manifest=bindings['image_manifest'],selected_unique_cases=68,all1784_META_qL_native_U_preserved=True,all_selected_original_RGB96_valid96_png_parity=True),evaluation_complete=dict(completed=True,PASS=True,prediction_seal=bindings['prediction_seal'],models=MODELS,old_six_primary_metrics_exact=True,old_annotations_outputs_and_inputs_unchanged=True),independent_numeric=audit,root_visual=dict(completed=True,PASS=True,gallery_complete=bindings['gallery_complete'],viewed_count=2),source_fit_integrity=dict(completed=True,PASS=True,source_fit_complete=dummyfit),prediction_seal=dict(completed=True,PASS=True,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,source_fit_complete=dummyfit),current_saved_summary=current)
 return inputs,bindings

def invented():
 tests=[]
 def test(name,ok):require(ok,name);tests.append(dict(name=name,PASS=True))
 r=dict(model='TOY',N=5,D_support=3,E_support=2,D_D=1,D_E=1,D_U=1,E_D=1,E_E=1,E_U=0,D_precision=.5,D_recall=1/3,E_precision=.5,E_recall=.5,U=1);validate_metric(r)
 test('Nonperfect_CM_P_R_include_U_and_shared_MD_HTML_rows',short_rows([r])[0]['D行 [D,E,U]']=='[1, 1, 1]'and '20.0%'in htmltable(short_rows([r])))
 bad=dict(r,E_recall=0.)
 try:validate_metric(bad);ok=False
 except AssertionError:ok=True
 test('Corrupt_stored_precision_recall_rejected',ok)
 z=dict(r,D_D=0,D_E=0,D_U=3,E_D=0,E_E=0,E_U=2,U=5,D_precision=None,D_recall=0.,E_precision=None,E_recall=0.);validate_metric(z)
 test('All_U_not_reported_as_zero_undefined_precision',short_rows([z])[0]['D 정밀도']=='미정'and short_rows([z])[0]['U율']=='100.0%')
 inputs,bindings=invented_fixture();res,md,h=compose(inputs,bindings)
 test('Actual_compose_all8_material4_self21_136_public7',len(res['table_primary8'])==8 and len(res['table_new_C56_C57_material4'])==4 and len(res['table_public_weak7'])==7 and len(res['self_tables_paired21'])==8)
 test('No_best_arm_and_source_public_real_grouped_caveats',res['selected_model']is None and res['adopted']is False and res['source_training']['source_only_includes_public_real']and res['source_training']['HOLDOUT45_D30_E15_synthetic_only'])
 test('Reference461_manual376_and_actual_view_scope_distinct',len(res['reference_error_LE3_humanD461'])==2 and res['manual_near_D_distinct']['N']==376 and res['actual_visual_scope']['root_viewed_count']==2 and res['actual_visual_scope']['all68_or1784_actually_viewed']is False)
 for name,mutate in [('audit_CM',lambda a:a['independent_numeric']['current']['primary']['ALL']['C56']['CM_D_E_by_D_E_U'][0].__setitem__(0,0)),('root_view_not_completed',lambda a:a['root_visual'].__setitem__('PASS',False)),('wrong_source_fit',lambda a:a['source_fit_integrity'].__setitem__('source_fit_complete',dict(path='/invented/wrong.json',sha256='d'*64)))]:
  a=copy.deepcopy(inputs);mutate(a)
  try:validate(a,bindings);ok=False
  except AssertionError:ok=True
  test(name+'_fails_closed',ok)
 a=copy.deepcopy(inputs);a['image_manifest']['images'][0]['original_global_index']=99999;r,_,_=compose(a,bindings);test('Static_inline_ID_missing_records_no_fallback',r['static_inline_missing_ids_no_replacement']==[0]and len(r['six_fixed_inline_cards'])==5)
 test('Authored_MD_HTML_relative_no_private_absolute_paths','/home/'not in md and '/home/'not in h and 'C56'in md and 'C57'in h)
 try:safe(P/'blocked_v112'/'unopened.json');ok=False
 except AssertionError:ok=True
 test('Forbidden_lexical_name_before_any_access',ok)
 return dict(completed=True,PASS=True,invented_only=True,count=len(tests),checks=tests,actual_real_source_metric_score_label_image_array_header_or_weight_reads=0)

def prepare():
 read(INPUT_SCOPE);ep=read(EVAL_PROTOCOL);vp=read(VISUAL_PROTOCOL)
 require(ep['models']==MODELS and vp['evaluation_protocol']==bind(EVAL_PROTOCOL)and vp['fixed_prior_selection']['sha256']=='69f558908cd5efbbb692bf998f75b2c5e93b74691011207b496f2e5ea04b9769','Frozen eight-model evaluator/same68 visual contract')
 write(CHECKS,invented())
 p=dict(completed=True,prepared_only=True,code=bind(__file__),prospective_input_scope=bind(INPUT_SCOPE),future_inputs={k:dict(path=str(x)if x else None,sha256=None)for k,x in ROLES.items()},evaluation_code=ep['code'],evaluation_protocol=bind(EVAL_PROTOCOL),visual_code=vp['code'],visual_protocol=bind(VISUAL_PROTOCOL),checks=bind(CHECKS),outputs=OUTPUTS,inline_existing_fixed_indices=INLINE,source_scope='V194 source-only includes public real and YCB controls;327TRAIN/45synthetic heldout/372FULL/realheld0; multiple changed conditions vsC54C55, no causal model/data choice.',root_actual_visual_QA_required=dict(completed=True,PASS=True,gallery_complete='Exact gallery binding',viewed_count='Positive actual card image-tool count; never infer all68/1784 viewed'),independent_A_numeric_review_contract="REVIEW.complete exact evaluation binding; current.primary[group][model] headline and .all_registered_humanD_existing_reference_LE3_association copied from independent scalar audit.",actual_before_rootentry=0,no_arrays_models_images_labels_scores_or_rowledgers_reader=True,selected_model=None,adopted=False,fully_solved=False)
 write(PROTOCOL,p);t=H/'ROOT_ACTUAL_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(t,dict(approved_saved_benchmark_report_write=False,no_model_array_image_label_score_forward_fit_or_selection=True,code=p['code'],protocol=bind(PROTOCOL),scope=p['prospective_input_scope'],inputs=p['future_inputs'],outputs=OUTPUTS))
 ready=dict(completed=True,PASS=True,FINAL_READY_FROZEN=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],template=bind(t),actual_result_metric_label_score_image_array_header_weight_reads=0,actual_top_outputs_written=0)
 write(H/'FINAL_READY.json',ready);print(json.dumps(ready))

def run(entry_path):
 e=read(entry_path);require(e['approved_saved_benchmark_report_write']is True and e['no_model_array_image_label_score_forward_fit_or_selection']is True,'Separate postterminal root report entry')
 p=read(checked(e['protocol']));require(e['code']==bind(__file__)==p['code']and e['protocol']==bind(PROTOCOL)and e['scope']==bind(INPUT_SCOPE)==p['prospective_input_scope']and e['outputs']==p['outputs']==OUTPUTS,'Frozen code/input rules')
 require(set(e['inputs'])==set(ROLES),'All planned report roles exactly')
 for k,expected in p['future_inputs'].items():
  if expected['path']is not None:require(e['inputs'][k]['path']==expected['path'],'Exact future JSON path '+k)
  else:require(safe(e['inputs'][k]['path']).is_relative_to(V)and Path(e['inputs'][k]['path']).suffix=='.json','Dynamic root visual QA in exact V195')
 inputs={k:read(checked(b))for k,b in e['inputs'].items()};require(inputs['evaluation_complete']['code']==p['evaluation_code']and inputs['evaluation_complete']['protocol']==p['evaluation_protocol']and inputs['gallery_complete']['code']==p['visual_code']and inputs['gallery_complete']['protocol']==p['visual_protocol'],'Actual evaluation/gallery frozen code closure')
 result,md,h=compose(inputs,e['inputs']);require(not any((V/n).exists()for n in OUTPUTS),'Five exclusive new top reports')
 mdrefs=linkcheck(md,pending=OUTPUTS);hrefs=linkcheck(h,True,pending=OUTPUTS)
 result.update(report_code=p['code'],report_protocol=bind(PROTOCOL),entry=bind(entry_path),offline_QA_before_write=dict(MD_local_refs=len(mdrefs),HTML_local_refs=len(hrefs),browser_rendered=False,image_bodies_or_new_data_read=0))
 write(V/'ROOT_RESULTS_KO.json',result);text(V/'ROOT_REPORT_KO.md',md);text(V/'PORTABLE_REPORT_KO.md',md);text(V/'ROOT_READOUT_KO.html',h)
 phase=dict(completed=True,PASS=True,phase='V195_GROUPED_DIRECT_END_FROZEN_REAL_BENCHMARK',phase_scope_complete_not_goal_complete=True,models=MODELS,actual=dict(source_models=2,source_updates_each=8000,new_saved_prediction_rows4288=True,current1784=True,public360=True,primary1237_D1147_E90=True,fixed_gallery_cases68=True,root_actual_card_views=actual_view_count(inputs['root_visual']),independent_A_numeric_PASS=True),reports={n:bind(V/n)for n in OUTPUTS if n!='ROOT_PHASE_RESULT.json'},report_code=p['code'],report_protocol=bind(PROTOCOL),entry=bind(entry_path),evidence=e['inputs'],root_report_browser_rendered=False,source_fit_V194_phase_scope_separate=True,selected_model=None,adopted=False,fully_solved=False,goal_status='active_unsolved',new_model_data_threshold_architecture_selection=False)
 write(V/'ROOT_PHASE_RESULT.json',phase);linkcheck(md);linkcheck(h,True)
 print(json.dumps(dict(completed=True,PASS=True,outputs={n:bind(V/n)for n in OUTPUTS},MD_links=len(mdrefs),HTML_links=len(hrefs),browser_rendered=False)))

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare()if a.prepare else run(a.entry)
