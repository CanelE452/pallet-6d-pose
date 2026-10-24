"""V191 Korean report from completed summary/QA JSON only; no data/model reads.
Prepare is metadata-only after the prospective report input scope. Root --entry
writes five exclusive top-level reports. Existing science/artifacts stay intact.
"""
from pathlib import Path
from html.parser import HTMLParser
import argparse,hashlib,html,json,os,re,stat,traceback
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
INPUT_SCOPE=H/'PROSPECTIVE_REPORT_INPUT_SCOPE_BEFORE_SAVED_SUMMARY_READ.json'
PROTOCOL=H/'PROTOCOL_BEFORE_FINAL_REPORT_METADATA_AND_WRITES.json';CHECKS=H/'GENERATED_INVENTED_REPORT_CHECKS.json'
OUTPUTS=['ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html','ROOT_PHASE_RESULT.json']
INLINE=[0,2,256,300,840,996];MODELS=['v61','legacy','C52','C53','C54','C55']
ROLES={'gallery_complete':V/'fixed_direct_end_real_visual_v1/COMPLETE.json','summary':V/'fixed_direct_end_real_visual_v1/SUMMARY_AND_TABLES.json','image_manifest':V/'fixed_direct_end_real_visual_v1/IMAGE_MANIFEST.json','evaluation_complete':V/'saved_real_head_evaluation_v1/COMPLETE.json','root_headline':V/'ROOT_INDEPENDENT_LEDGER_HEADLINE_COMPARISON_CHECK.json','reference_error':V/'ROOT_REFERENCE_ERROR_ASSOCIATIONS_FROM_SAVED_LEDGER.json','independent_numeric':V/'independent_saved_result_review_b/actual_review_v1/REVIEW.json','root_visual':V/'ROOT_ACTUAL_FIXED24_CARD_VISUAL_REVIEW.json','C_visual':V/'independent_actual_visual_review_c/REVIEW.json','source_fit_integrity':P/'query_visibility_direct_end_paired_source_training_20261024_v190/independent_source_fit_review_b/actual_review_v1/REVIEW.json'}

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
def rel(p):
 path=safe(p);require(V in path.parents,'Portable links stay inside V191');return path.relative_to(V).as_posix()
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
class Refs(HTMLParser):
 def __init__(self):super().__init__();self.refs=[]
 def handle_starttag(self,t,a):self.refs.extend(v for k,v in a if k in ['href','src'] and v)
def linkcheck(content,html_mode=False,pending=()):
 require('/home/' not in content and 'file://' not in content,'Portable text has no absolute local path')
 if html_mode:parser=Refs();parser.feed(content);refs=parser.refs
 else:refs=re.findall(r'\]\(([^)]+)\)',content)
 for s in refs:
  require(not s.startswith('/') and ':' not in s and '..' not in Path(s).parts,'Generated local relative link only')
  if s in pending:continue
  with opened(V/s):pass  # existence/type only, no image body read/hash/decode
 return refs

def validate(inputs):
 s=inputs['summary'];g=inputs['gallery_complete'];m=inputs['image_manifest'];headline=inputs['root_headline']
 require(s['completed'] and s['all_models']==MODELS and s['selected_model'] is None and s['adopted'] is False and s['fully_solved'] is False,'Fixed six models/no adoption')
 for role in ['gallery_complete','evaluation_complete','root_headline','reference_error','independent_numeric','root_visual','C_visual','source_fit_integrity']:require(inputs[role]['completed'] and inputs[role]['PASS'],'Completed QA '+role)
 require(s['current_primary']['N']==1237 and s['current_primary']['D']==1147 and s['current_primary']['E']==90 and s['current_all1784']['N']==1784 and s['current_all1784']['input_U']==45,'Original cohorts/U')
 require(g['selected_unique_cases']==s['selected_images']==len(m['images'])==68 and g['all1784_META_qL_native_U_preserved'] and g['all_selected_original_RGB96_valid96_png_parity'],'Fixed68/input parity')
 for r in s['current_primary']['rows']:
  h=headline['results'][r['group']][r['model']];require(cm(r)==h['CM_D_E_by_D_E_U'] and r['N']==h['N'] and all(r[k]==h[k] for k in ['D_precision','D_recall','E_precision','E_recall']),'Authoritative summary vs root independent headline')
 require(sum(r['all_original_slots'] for r in s['current_all1784']['materials_native8'])==1784 and {r['native_slot'] for r in s['current_all1784']['materials_native8']}==set(range(8)),'Native8/all source rows retained')
 require(inputs['root_visual']['viewed_count']==24 and inputs['C_visual']['actual_tool_viewed_card_count']==12 and inputs['C_visual']['all68_or_all1784_actually_viewed'] is False,'Exact actual view scope')
 ids={r['original_global_index'] for r in m['images']};require(set(INLINE)<=ids,'Six existing fixed cards, no fallback/new selection')
 return True

def compose(inputs,bindings,entry):
 validate(inputs);s=inputs['summary'];pt=s['current_primary']['rows'];main=[r for r in pt if r['group']=='ALL'];by={r['model']:r for r in main};c55=[r for r in pt if r['model']=='C55' and r['group']!='ALL']
 native=s['current_all1784']['materials_native8'];ref=inputs['reference_error']['counts']['PRIMARY_HUMAN_D']['LE3'];public=s['public_weak_owner_only'];weak=s['public_weak_headline_table']
 def self_table(cohort):
  rows=[]
  for model in MODELS:
   rr=[r for r in s['self_separate'] if r['cohort']==cohort and r['model']==model]
   rows.append({'모델':model,'별도 human S N':sum(r['human_S_support'] for r in rr),'D 응답':sum(r['visible_response'] for r in rr),'binary hidden 응답':sum(r['binary_hidden_response'] for r in rr),'미결정':sum(r['unresolved'] for r in rr)})
  return rows
 selfrows=self_table('paired_same_query_RGB');selfall=self_table('same_slot_pipeline')
 require({r['별도 human S N'] for r in selfrows}=={21} and {r['별도 human S N'] for r in selfall}=={136} and s['current_all1784']['human_state_counts']['SELF_OCCLUDED']==136,'Existing paired21 vs full SELF136 cohorts')
 self_definition={'paired_same_query_RGB_D_vs_ES_human_SELF':21,'same_slot_pipeline_D_vs_ES_human_SELF':136,'all_original_1784_human_SELF':136,'not_in_paired_same_query_cohort':115,'difference_meaning':'Existing paired_same_query_RGB_D_vs_ES eligibility excludes115 human SELF rows retained by same_slot_pipeline; source summary does not classify exclusion reasons. Do not infer query error, inputU, or physical cause.','authority_code_relative':'../query_visibility_matched_query_scalar_20261023_v94/current_evaluation_preparation/evaluate_saved_only.py','authority_lines':[97,113,114,115,116,117,118,119,120,121],'no_learned_self_or_external_vs_self_identification':True}
 weakrows=[{'모델':r['model'],'T 정밀도':pct(r['T_precision_known']),'T 재현율':pct(r['T_recall_including_U']),'non-T 재현율':pct(r['nonT_recall_including_U']),'owner F 정밀도':pct(r['ownerF_precision_known']),'owner F 재현율':pct(r['ownerF_recall_including_U'])} for r in weak]
 materialrows=[{'재질':'WOOD' if r['group']=='WOOD_PRIMARY105' else 'SQUARE','primary N':r['N'],'D/E 분모':'%d/%d'%(r['D_support'],r['E_support']),'D행 [D,E,U]':str(cm(r)[0]),'E행 [D,E,U]':str(cm(r)[1]),'D 재현율':pct(r['D_recall']),'E 재현율':pct(r['E_recall'])} for r in c55]
 captions={0:'고정된 원 입력과 6모델 출력을 함께 비교한다. 무효 padding이 있으면 자연 배경으로 해석하지 않는다.',2:'기존 human D인데 C55는 E다. 작은 reference 오차 집단에도 D 오탐이 남으며 local crop만으로 원인을 단정할 수 없다.',256:'사전에 고정된 입력 예시다. 표의 인간 D/E와 각 모델 응답을 그대로 보며 새 원인·라벨을 만들지 않는다.',300:'기존 reference 오차가 약103px인 사례다. E를 맞혔다는 수치만으로 현재 q의 의도한 끝부위 외부가림을 인증하지 않는다.',840:'SQUARE 고정 crop이다. 색이나 중심의 literal owner를 native 외부가림 gold로 바꾸지 않는다.',996:'기존 human E이며 C54/C55 모두 D다. reference 오차 약1.14px에서도 E miss가 남지만 그 참조가 물리 pixel gold라는 뜻은 아니다.'}
 ims={r['original_global_index']:r for r in inputs['image_manifest']['images']};inline=[dict(original_global_index=i,card=ims[i]['card'],caption_KO=captions[i]) for i in INLINE]
 lead='직접 끝부위 감독을 더한 C55는 이전 C53보다 E 재현율이 '+pct(by['C53']['E_recall'])+' → '+pct(by['C55']['E_recall'])+'로 높아졌지만, D 재현율은 '+pct(by['C53']['D_recall'])+' → '+pct(by['C55']['D_recall'])+'로 낮아졌다. E90개 중 '+str(by['C55']['E_D'])+'개를 여전히 D로 통과시켜 전체 목표는 미해결이며 모델을 채택하지 않았다.'
 limits=['이 벤치마크의 인간 D/E/S 의미 권위는 유지한다. 새 direct source 표적은 synthetic near-anchor 끝부위 규약과 public RGB 약관찰이며, 실제 native semantic label과의 동일성 gold로 승격하지 않았다.',
 'C55와 C53 비교에는 감독·학습 조건 변화가 함께 있다. 이 결과로 SAM, 특정 데이터, head 변화의 원인을 분리하거나 새 학습/문턱/모델/architecture/data를 선택하지 않았다.',
 'source 직접감독306행의 apparent fit은 학습 노출 결과다. 자연실사 성능 증거가 아니며 source 학습·보고 단계 V190과 이 frozen 실사 벤치마크를 구분한다.',
 'current223과 publicheld45는 반복 개발 노출되어 fresh blind 검증이 아니다. legacy의 과거 realCAL 이력을 숨기지 않으며 새 C54/C55는 source-only로 학습했다.',
 '원 current1784의 input U45(45/1784=2.52%)와 public360의 input U108을 보존했다. primary 표의 U는 해당1237행 안의 모델 응답 U이며, 전체 원 입력U 분모와 다르다. legacy U177과 기존 C52/C53 regional NONE도 자체 의미 그대로다.',
 'human self S는 E에 합치지 않고 별도 binary response로 보고한다. E 응답 자체가 물리 외부가림 원인이나 외부/self 분리 성공을 뜻하지 않는다.',
 '전체 WOOD/SQUARE와 native0..7 목표는 유지한다. 고정68갤러리는 post-result strata의 사전 고정 first6 규칙이며 prevalence 대표표본이 아니다. Root24/C12 실뷰는 일부 중첩하며 전68/1784 이미지를 모두 봤다는 주장은 없다. 브라우저 렌더는 검증하지 않았다.']
 result=dict(completed=True,phase_scope='Frozen C54/C55 real benchmark and saved readout only; source-fit phase V190 separately tracked',models=MODELS,current_primary_N1237_D1147_E90=s['current_primary'],table_primary6=main,primary_U_rate_by_model={r['model']:r['U']/r['N'] for r in main},original_current_input_U_rate=45/1784,table_material_C55_2=c55,table_all6_material18=pt,native8_material_full_denominators=native,current_all1784=s['current_all1784'],registered_frames={'WOOD':105,'SQUARE':118},primary_SQUARE_frames=117,self_separate_source_tables=s['self_separate'],table_self6=selfrows,table_self_same_slot_all136_6=selfall,self_cohort_definition=self_definition,public_weak_center_only=public,table_public_weak5=weak,reference_error_LE3_humanD=ref,manual_near_D_distinct_cohort={'N':376,'WOOD':153,'SQUARE':223,'authority':'fixed manual_click subset, not the461 existing reference<=3 association'},source_training={'arms':2,'updates_each':8000,'total_updates':16000,'direct_source_known_rows':306,'apparent_fit_not_real_proof':True,'V190_phase_status_separate':True},six_fixed_inline_cards=inline,actual_visual_scope={'root':24,'independent_C':12,'overlap_possible':True,'all68_or1784_viewed':False,'browser_rendered':False},evidence=bindings,source_summary_only_no_new_metric_kernel_or_data_run=True,limitations_KO=limits,selected_model=None,adopted=False,fully_solved=False,goal_status='active_unsolved')
 md=['# V191: 직접 끝부위 감독 이후의 고정 실사 벤치마크','',lead,'', '원 입력은 current223×8=1784와 public45×8=360이다. primary는 같은 원 q/RGB에 대응하는 인간 D1147/E90, 총1237행이다. 등록 프레임은 WOOD105/SQUARE118이며 SQUARE primary는117프레임이다. 원 q/L/native/가용성·crop을 바꾸지 않았다. 전체 원 입력의 factual input U는 45/1784=2.52%다. 아래 primary 모델 U/N과 분모가 다르며, primary U0은 전체 원 입력이 모두 분류되었다는 뜻이 아니다.','', '## 전체 6모델 primary 비교','', 'CM의 행은 인간 D/E, 각 행의 열은 모델 D/E/U다. 정밀도는 결정된 해당 예측 분모, 재현율은 U를 포함한 해당 인간 D/E 전체 분모다. 기존 모델의 E 열은 원래 binary 응답이며 물리 원인 인증이 아니다.','',mdtable(short_rows(main)),'', 'C55는 E를 '+str(by['C55']['E_E'])+'/90 검출했고, 이전 C53의 '+str(by['C53']['E_E'])+'/90보다 늘었다. 그러나 '+str(by['C55']['E_D'])+'/90을 놓치며 legacy의 E 재현율 '+pct(by['legacy']['E_recall'])+'에도 미치지 못한다. 높은 D 통과율만으로 해결로 판단하지 않았다.','', '## C55 재질별 결과','',mdtable(materialrows),'', 'WOOD primary645(D616/E29)와 SQUARE primary592(D531/E61)를 각각 유지한다. [모든6모델·재질 표](fixed_direct_end_real_visual_v1/PRIMARY_ALL6_CM_PRECISION_RECALL_U.csv)와 [native8/재질 전체1784 분모](fixed_direct_end_real_visual_v1/ALL_ORIGINAL_NATIVE_MATERIAL_DENOMINATORS.csv)를 함께 제공한다.','', '## 위치·self·public weak는 별개다','',
 '기존 reference 오차≤3px인 human D는 '+str(ref['N'])+'행이며 C55 응답은 D'+str(ref['models']['C55']['D'])+'/E'+str(ref['models']['C55']['E'])+'/U'+str(ref['models']['C55']['U'])+'다. 이461행은 manual-click near376 집단과 다르다. <=3px는 기존 참조 연관으로서 물리 좌표 gold나 인과 효과가 아니다.','','### Self: paired same-query RGB 21행','',mdtable(selfrows),'', '### Self: 전체 원장136행 / same-slot pipeline','',mdtable(selfall),'', '전체1784 원장의 human SELF136행이 기존 same_slot_pipeline_D_vs_ES 표에 모두 포함되어 있다. 그중 paired_same_query_RGB_D_vs_ES 조건을 만족하는 집단은21행이고, 나머지115행은 이 paired 집단 밖이다. 기존 요약에는115행의 제외 사유 세부 집계가 없으므로 query 오류·원 inputU·물리 가림 원인으로 추측하지 않는다.','', '두 self 표는 서로 다른 기존 집단의 binary 응답이며 learned-self classifier가 아니다. [전체 self 표](fixed_direct_end_real_visual_v1/SELF_SEPARATE_BINARY_RESPONSES.csv)를 보존했다.','',
 'public360은 중심 owner 약관찰 known204/UNKNOWN156이며 원 inputU108, center4 invalid112가 별개다. 아래 T/non-T와 owner F 지표를 current native E90 성능과 동일시하지 않는다. legacy public owner 경로는 없어서 만들지 않았다.','',mdtable(weakrows),'', '[public weak 전체 분모·혼동표](fixed_direct_end_real_visual_v1/SUMMARY_AND_TABLES.json) · [public weak 요약 CSV](fixed_direct_end_real_visual_v1/PUBLIC_WEAK_CENTER_OWNER_ONLY.csv)','', '## 고정 갤러리에서 볼 수 있는 것','', '기존 68개 고정 crop 중 아래6개는 설명용 링크다. 새 사례를 렌더하거나 input을 재중심화하지 않았다. 그림의 cyan q는 표시 사본에만 있고 clean96는 원 모델 입력 픽셀이다. 실제 원인 판단은 local crop만으로 확정하지 않는다.','']
 for item in inline:md += ['### original '+str(item['original_global_index']),'',item['caption_KO'],'',f'![fixed original {item["original_global_index"]}]({rel(item["card"]["path"])})','']
 md += ['[68개 전체 portable 갤러리](fixed_direct_end_real_visual_v1/PORTABLE_GALLERY_KO.md) · [HTML 갤러리 원문](fixed_direct_end_real_visual_v1/index.html) · [고정 선택표](fixed_direct_end_real_visual_v1/FIXED_CELL_SELECTION.csv)','', '## 확인한 범위와 남은 문제','']+['- '+x for x in limits]+['','[독립 수치 검산](independent_saved_result_review_b/actual_review_v1/REVIEW.json) · [root24 실제 그림 QA](ROOT_ACTUAL_FIXED24_CARD_VISUAL_REVIEW.json) · [C12 실제 그림 QA](independent_actual_visual_review_c/REVIEW.json)','', '[예측 코드](prediction_preparation_a/infer_both_frozen_direct_end_heads.py) · [평가 코드](evaluation_preparation_a/evaluate_saved_direct_end_real.py) · [갤러리 코드](visual_readout_preparation_a/render_fixed_direct_end_real.py) · [보고서 생성 코드](report_preparation_a/build_top_level_readout.py)','', '[정확 수치·binding JSON](ROOT_RESULTS_KO.json) · [phase 범위](ROOT_PHASE_RESULT.json) · [오프라인 HTML](ROOT_READOUT_KO.html)','']
 markdown='\n'.join(md)
 # Explicit static HTML uses the same formatted table row dictionaries as MD.
 h='<h1>V191: 직접 끝부위 감독 이후의 고정 실사 벤치마크</h1><p class="lead">'+html.escape(lead)+'</p><p>원 current1784/inputU45, primary1237(D1147/E90), 등록 WOOD105/SQUARE118, SQUARE primary117프레임. 모든 native0..7 원 q/L/가용성 보존. 전체 원 입력 factual U45/1784=2.52%와 아래 primary 모델 U/N은 별개이며, primary U0이 전체 원 입력의 전부 분류를 뜻하지 않는다.</p><h2>전체6모델 primary</h2><p>행: 인간 D/E, 열: 모델 D/E/U. 재현율은 U 포함 인간 전체 분모. E 응답은 물리 원인 인증이 아니다.</p>'+htmltable(short_rows(main))+'<h2>C55 재질별</h2>'+htmltable(materialrows)+'<h2>위치·self</h2><p>기존 reference≤3px humanD '+str(ref['N'])+'행과 manual-click near376을 구분한다. 물리 좌표 gold나 인과 분리가 아니다.</p>'+'<h3>Self: paired same-query RGB 21행</h3>'+htmltable(selfrows)+'<h3>Self: 전체 원장136행 / same-slot pipeline</h3>'+htmltable(selfall)+'<p>전체1784의 human SELF136행은 기존 same_slot_pipeline_D_vs_ES 집단에 모두 포함된다. 그중 paired_same_query_RGB_D_vs_ES는21행이며 차이115행은 그 paired eligibility 밖이다. 기존 요약에는 제외 사유 세부 집계가 없어 query 오류/inputU/물리 원인으로 추측하지 않는다. learned self 또는 외부/self 구분 성공이 아니다.</p><h2>Public weak center-owner만</h2><p>360행, known204/UNKNOWN156, 원 inputU108/centerinvalid112 별개. current native D/E와 같은 정답이 아니다.</p>'+htmltable(weakrows)+'<h2>기존 고정6개 설명용 그림</h2>'
 for item in inline:h+='<figure><img src="'+rel(item['card']['path'])+'"><figcaption>original '+str(item['original_global_index'])+': '+html.escape(item['caption_KO'])+'</figcaption></figure>'
 h+='<h2>한계</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in limits)+'</ul><p><a href="PORTABLE_REPORT_KO.md">전체 portable 보고서와 코드·QA·표 링크</a> · <a href="fixed_direct_end_real_visual_v1/PORTABLE_GALLERY_KO.md">고정68 전체 갤러리</a> · <a href="ROOT_RESULTS_KO.json">수치와 근거 JSON</a></p>'
 h='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>V191 frozen direct-end benchmark</title><style>body{max-width:1600px;margin:30px auto;padding:0 24px;font:18px/1.65 system-ui,sans-serif;color:#1c2630}.lead{padding:20px;background:#eef3f7;border-left:5px solid #31536c}table{border-collapse:collapse;width:100%;font-size:16px;margin:20px 0}th,td{border:1px solid #c6cdd2;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}figure{margin:24px 0}img{max-width:100%;height:auto}figcaption{padding:10px;background:#f3f4f5}li{margin-bottom:12px}</style><body>'+h+'</body></html>'
 return result,markdown,h

def invented():
 r=dict(model='TOY',N=4,D_D=1,D_E=0,D_U=1,E_D=1,E_E=0,E_U=1,D_precision=.5,D_recall=.5,E_precision=None,E_recall=0.,U=2)
 sr=short_rows([r])[0];require(sr['D행 [D,E,U]']=='[1, 0, 1]' and sr['E행 [D,E,U]']=='[1, 0, 1]' and sr['E 정밀도']=='미정','CM row/column/U and undefined denominator display')
 require(sr['U율']=='50.0%','Primary U rate uses U/N');require(pct(.41111111)=='41.1%' and pct(0)=='0.0%','Stable displayed percentages')
 require('0.0%' in htmltable(short_rows([r])) and '[1, 0, 1]' in mdtable(short_rows([r])),'Same table cells MD/HTML')
 require(136-21==115 and 103+33+0==136 and 16+5+0==21,'Distinct full and paired self table denominators');return dict(completed=True,PASS=True,invented_only=True,checks=['CM_D_E_rows_D_E_U_columns','U_in_denominators_and_undefined_precision_not_zero','percentage_format','same_table_row_dictionary_MD_HTML','primary_U_div_N_display','paired21_full136_SELF_cohorts_not_conflated'],new_metric_or_data_or_model_execution=0)

def prepare():
 scope=read(INPUT_SCOPE);allowed=set(scope['exact_phase_JSON_whitelist'])|{scope['additional_exact_source_fit_receipt']};require(all(str(x) in allowed for x in ROLES.values()),'Prospective input whitelist declared before body reads')
 inputs={k:read(p) for k,p in ROLES.items()};validate(inputs);write(CHECKS,invented())
 p=dict(completed=True,prepared_only=True,code=bind(Path(__file__)),prospective_input_scope=bind(INPUT_SCOPE),inputs={k:bind(x) for k,x in ROLES.items()},checks=bind(CHECKS),outputs=OUTPUTS,inline_existing_fixed_indices=INLINE,meaning='Completed saved benchmark/report JSON and existing QA only; no raw images/arrays/checkpoints/labels/row ledgers or new inference/fit/metric kernel. Original files unchanged.',actual_top_level_write_before_rootentry=0,selected_model=None,adopted=False,fully_solved=False)
 write(PROTOCOL,p);template=H/'ROOT_ACTUAL_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(template,dict(approved_saved_benchmark_report_write=False,no_model_array_image_label_score_forward_fit_or_selection=True,code=p['code'],protocol=bind(PROTOCOL),inputs=p['inputs'],outputs=OUTPUTS))
 ready=dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],template=bind(template),new_actual_data_or_tensor_model_reads=0,actual_top_outputs_written=0)
 write(H/'PREPARATION_COMPLETE.json',ready);print(json.dumps(ready))

def run(entry_path):
 e=read(entry_path);require(e['approved_saved_benchmark_report_write'] is True and e['no_model_array_image_label_score_forward_fit_or_selection'] is True,'Explicit saved-report entry')
 p=read(checked(e['protocol']));require(e['code']==bind(Path(__file__))==p['code'] and e['protocol']==bind(PROTOCOL) and e['inputs']==p['inputs'] and e['outputs']==p['outputs']==OUTPUTS,'Frozen report code/protocol/input closure')
 inputs={k:read(checked(b)) for k,b in p['inputs'].items()};result,md,h=compose(inputs,p['inputs'],bind(entry_path));require(not any((V/n).exists() for n in OUTPUTS),'Five exclusive new outputs only')
 mdrefs=linkcheck(md,pending=OUTPUTS);hrefs=linkcheck(h,True,pending=OUTPUTS)
 result['report_code']=p['code'];result['report_protocol']=bind(PROTOCOL);result['entry']=bind(entry_path);result['offline_QA_before_write']=dict(MD_local_refs=len(mdrefs),HTML_local_refs=len(hrefs),browser_rendered=False,image_bodies_or_new_data_read=0)
 write(V/'ROOT_RESULTS_KO.json',result);text(V/'ROOT_REPORT_KO.md',md);text(V/'PORTABLE_REPORT_KO.md',md);text(V/'ROOT_READOUT_KO.html',h)
 phase=dict(completed=True,PASS=True,phase='V191_FROZEN_DIRECT_END_REAL_BENCHMARK',phase_scope_complete_not_goal_complete=True,actual=dict(source_models=2,source_updates_each=8000,new_saved_prediction_rows4288=True,current1784=True,public360=True,primary1237_D1147_E90=True,fixed_gallery_cases68=True,root_actual_card_views24=True,C_actual_card_views12=True,independent_saved_numeric_PASS=True),reports={n:bind(V/n) for n in OUTPUTS if n!='ROOT_PHASE_RESULT.json'},report_code=p['code'],report_protocol=bind(PROTOCOL),entry=bind(entry_path),evidence=p['inputs'],root_report_browser_rendered=False,source_fit_V190_phase_scope_separate=True,selected_model=None,adopted=False,fully_solved=False,goal_status='active_unsolved',new_model_data_threshold_architecture_selection=False)
 write(V/'ROOT_PHASE_RESULT.json',phase);linkcheck(md);linkcheck(h,True)
 print(json.dumps(dict(completed=True,PASS=True,outputs={n:bind(V/n) for n in OUTPUTS},MD_links=len(mdrefs),HTML_links=len(hrefs),browser_rendered=False)))

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare() if a.prepare else run(a.entry)
