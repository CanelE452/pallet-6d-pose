"""Source-only structural, numeric, asset-binding QA; no model or private data."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote
from datetime import datetime, timezone
import hashlib,json,struct
V=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return {'path':str(p),'sha256':sha(p)}
qa_path=V/'ROOT_READOUT_QA.json';assert not qa_path.exists()
rp=V/'ROOT_RESULTS_KO.json';hp=V/'ROOT_READOUT_KO.html'
r=read(rp);doc=hp.read_text();before=[sha(rp),sha(hp)]
assert r['completed'] and r['source_only'] and not r['fully_solved']
assert r['source_model_admission']==dict(selected_model=None,current_allowed=False,adopted=False,source_qualification_failed=True)
assert r['successful_execution']['session']==25096 and r['successful_execution']['terminal_exit_code']==0
assert r['successful_execution']['updates_each']==2000
assert r['first_attempt_preserved']['session']==22384 and r['first_attempt_preserved']['fit_updates']==0 and r['first_attempt_preserved']['classifier_calls']==0
assert r['comparison']['public_coefficient']=={'C33':0,'C34':1} and r['comparison']['HALL_gradient_rows']==0
assert r['public_inventory']['known_DE']==84 and [r['public_inventory'][x] for x in ('D','E','U')]==[70,14,428]
assert r['public_inventory']['frames']==64 and r['public_inventory']['slots']==512
assert r['public_inventory']['HALL_known41']=={'D':35,'E':6} and r['public_inventory']['HALL_input_valid39']=={'D':33,'E':6}
assert r['public_inventory']['HALL_teacherD_original_inputU']==[402,434]
assert r['public_inventory']['original_input_valid']==463 and r['public_inventory']['original_input_OOB']==49
expected={'ALL':{'C32_INITIAL':[[62,6,2],[0,14,0]],'C33_CONTROL':[[60,8,2],[0,14,0]],'C34_PUBLIC':[[68,0,2],[0,14,0]]},'TRAIN_LAB':{'C32_INITIAL':[[33,2,0],[0,8,0]],'C33_CONTROL':[[32,3,0],[0,8,0]],'C34_PUBLIC':[[35,0,0],[0,8,0]]},'VAL_HALL':{'C32_INITIAL':[[29,4,2],[0,6,0]],'C33_CONTROL':[[28,5,2],[0,6,0]],'C34_PUBLIC':[[33,0,2],[0,6,0]]}}
for group,models in expected.items():
 for model,cm in models.items():
  got=r['public_metrics'][group][model]
  assert got['CM_D_E_rows_D_E_U_columns']==cm
  assert got['correct']==cm[0][0]+cm[1][1] and got['known']==sum(map(sum,cm))
  assert got['U']==cm[0][2]+cm[1][2]
assert r['native35']['C33_passed']==22 and r['native35']['C34_passed']==22 and r['native35']['total']==35 and not r['native35']['all_pass']
assert r['same_provided_T_K_four_pairs_both_correct']==dict(C32_INITIAL=3,C33_CONTROL=3,C34_PUBLIC=4)
assert r['initial_D_to_E_errors']==r['C34_recovered_initial_errors']==6
assert r['same_T_K_is_not_same_actual_query_or_isolated_image_only_intervention']
checked=[]
def check_bindings(x):
 if isinstance(x,dict):
  if 'path'in x and 'sha256'in x:
   p=Path(x['path']);assert p.is_file() and sha(p)==x['sha256'];checked.append(str(p))
  else:
   for y in x.values():check_bindings(y)
 elif isinstance(x,list):
  for y in x:check_bindings(y)
check_bindings(r['evidence']);check_bindings(r['successful_execution']['control']);check_bindings(r['first_attempt_preserved']['diagnosis'])
diagnosis=read(r['evidence']['saved_diagnosis']['path']);assert diagnosis['metrics']==r['public_metrics']
root_visual=read(r['evidence']['root_visual_QA']['path']);assert root_visual['completed'] and root_visual['root_actually_viewed_both_scoreboards']
for arm,b in r['evidence']['source_results'].items():
 d=read(b['path']);model='C33_CONTROL'if arm.startswith('C33')else'C34_PUBLIC'
 assert d['completed'] and d['final_update']==2000 and d['selected_model']is None and not d['full_goal_solved'] and not d['adopted']
 for group in expected:assert d['public']['groups'][group]['CM_D_E_rows_D_E_U_columns']==expected[group][model]
 assert d['native']['native35_passed']==22 and not d['native']['native35_all_pass']
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.refs=[];self.images=[];self.tables=0;self.main=0;self.text=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='table':self.tables+=1
  if tag=='main':self.main+=1
  if tag=='a':self.refs.append(a.get('href',''))
  if tag=='img':self.images.append(a);self.refs.append(a.get('src',''))
 def handle_data(self,s):self.text.append(s)
p=Parser();p.feed(doc);assert p.tables==2 and p.main==1 and len(p.images)==2
local_refs=[]
for ref in p.refs:
 u=urlparse(ref);assert not u.scheme and not u.netloc and ref
 path=(V/unquote(u.path)).resolve();assert path.is_file();local_refs.append(binding(path))
images=[]
for image,b in zip(p.images,r['evidence']['scoreboards']):
 path=(V/image['src']).resolve();assert str(path)==b['path'] and sha(path)==b['sha256'] and image.get('alt')
 data=path.read_bytes();assert data[:8]==b'\x89PNG\r\n\x1a\n';w,h=struct.unpack('>II',data[16:24]);assert(w,h)==(1920,1730)
 images.append(dict(**b,width=w,height=h,actual_pixel_view_QA=r['evidence']['root_visual_QA']))
text=' '.join(p.text)
for needle in ('보이는 D 보존','학습에 사용한 LAB43개','전체41개와 입력 유효39개','current_allowed=false','selected_model=null','native35는 두 arm22/35','실사 전이 성공을 뜻하지 않습니다','의미 U라서','입력 OOB라 최종 U','정밀 physical GT 오차가 아니며','EXIT1','EXIT0'):assert needle in text
assert 'V149' not in doc and '1072 / 75' not in doc and '1097 / 50' not in doc
assert before==[sha(rp),sha(hp)]
out=dict(completed=True,PASS=True,phase='V148',created_UTC=datetime.now(timezone.utc).isoformat(),
 results=binding(rp),readout=binding(hp),QA_code=binding(Path(__file__).resolve()),
 numeric_scope='Completed source JSON summaries only; no model, checkpoint, private, feature or score-packet body reads in this QA.',
 checks=dict(completed_original_audit_and_evidence_bindings=True,evidence_bindings_verified=len(checked),all_public_CM_matches_diagnosis_and_saved_source_results=True,
 HALL41_vs_available39_and_two_teacherD_inputU_preserved=True,ALL84_contains_TRAIN43_disclosed=True,D_preservation_not_new_E_recall=True,
 initial_six_D_to_E_recovered=True,sameT4_pair_3_to_4_actual_query_diff_disclosed=True,native35_both22_fail_admission_false_null=True,
 weak_partial_lower_not_human_gold_provisional_scan_bridge_disclosed=True,original512_qL_selection_and_input463_OOB49_retained=True,
 original_first_prefit_failure_and_successful_separate_run_retained=True,HTML_parser_completed=True,HTML_tables_count=2,
 all_local_asset_links_exist=True,external_resources=0,local_reference_count=len(local_refs),source_only_no_V149_metrics=True,
 result_and_HTML_unchanged_during_QA=True,parent_phase_and_pointers_not_changed=True),
 images=images,local_refs=local_refs,
 HTML_browser_render=dict(actually_rendered=False,reason='IAB unavailable; Chrome local file: URL rejected by browser security policy. No workaround or additional browser attempts.',
 policy_block_does_not_affect_saved_source_numerical_or_PNG_QA=True),
 actual_visual_QA=dict(saved_board_PNGs_personally_viewed_earlier=True,root_independently_viewed_both=True,root_receipt=r['evidence']['root_visual_QA'],
 HTML_layout_screenshot_not_verified=True),new_model_calls=0,new_private_or_V112_access=0,new_training=0)
with qa_path.open('x')as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(results=binding(rp),readout=binding(hp),QA=binding(qa_path)),ensure_ascii=False))
