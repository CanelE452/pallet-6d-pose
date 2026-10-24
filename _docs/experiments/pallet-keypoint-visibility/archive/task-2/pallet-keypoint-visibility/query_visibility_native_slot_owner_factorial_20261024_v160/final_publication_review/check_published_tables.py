"""Read-only publication arithmetic check over sealed report/extracted JSON."""
from html.parser import HTMLParser
from pathlib import Path
import ast, hashlib, json
V=Path(__file__).resolve().parents[1]
H=Path(__file__).resolve().parent
class Tables(HTMLParser):
 def __init__(self):
  super().__init__();self.text_tag=None;self.buf='';self.heading='';self.tables=[];self.table=None;self.row=None;self.cell=None
 def handle_starttag(self,t,a):
  if t in ('h2','summary'):self.text_tag=t;self.buf=''
  if t=='table':self.table=[]
  if t=='tr':self.row=[]
  if t in ('td','th'):self.cell=''
 def handle_data(self,d):
  if self.text_tag:self.buf+=d
  if self.cell is not None:self.cell+=d
 def handle_endtag(self,t):
  if t==self.text_tag:self.heading=self.buf;self.text_tag=None
  if t in ('td','th') and self.cell is not None:self.row.append(self.cell);self.cell=None
  if t=='tr' and self.row is not None:self.table.append(self.row);self.row=None
  if t=='table':self.tables.append((self.heading,self.table));self.table=None

def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text())
def pct(v):return '—' if v is None else f'{v*100:.2f}%'
s=read(V/'factorial_source_readout/ARM_DATASET_SUMMARY.json')
t=read(V/'factorial_source_readout/FULL_NATIVE_MATERIAL_ASPECT_TABLES.json')
r=read(V/'ROOT_RESULTS_KO.json')
parser=Tables();html=(V/'ROOT_READOUT_KO.html').read_text();parser.feed(html)
armmap=dict(zip(('기준','키포인트 번호 추가','중심 표면 보조 추가','둘 다 추가'),s['arms']))
folds={'OLD6144':'VALIDATION','NEW20480':'VALIDATION','PALLOX512':'VAL_HALL','MR1840':'SOURCE_VALIDATION'}
main={'기존 합성 검증':'OLD6144','추가 합성 검증':'NEW20480','Pallox 공개 실사 검증':'PALLOX512','MR 공개 실사 · 분리 장면':'MR1840'}
checks=0;rows_checked=0;table_names=[];last_main_family=None
for title,table in parser.tables:
 table_names.append(title)
 conditional=title.startswith('원래 입력 유효 조건부 진단')
 if title in main:last_main_family=main[title]
 family=last_main_family if conditional else main.get(title,title.split(' · ')[0]);assert family in folds,title
 for row in table[1:]:
  armname,sep,group=row[0].partition(' · ');arm=armmap[armname]
  if title in main or conditional:metric=s['arms'][arm][family][folds[family]]['input_valid_diagnostic' if conditional else 'all_original_rows']
  else:
   selector=folds[family]+'::'+(group if group.startswith('native') else 'material_aspect::'+group)
   metric=t['arms'][arm][family][selector]['all_original_rows']
  assert ast.literal_eval(row[1])==metric['CM_D_E_rows_D_E_U_columns'],(title,row)
  checks+=1
  for actual,key in zip(row[2:6],('D_precision','D_recall','E_precision','E_recall')):
   assert actual==pct(metric[key]),(title,row,key,metric[key]);checks+=1
  expected=f"{metric['prediction_U_known']}/{metric['known_DE']} ({pct(metric['prediction_U_rate_known'])})"
  assert row[6]==expected,(title,row,expected);checks+=1;rows_checked+=1
assert rows_checked==192 and len(parser.tables)==14
for x in r['MR_held_full']:
 exact=s['arms'][x['arm']]['MR1840']['SOURCE_VALIDATION']['all_original_rows']
 assert all(x[k]==v for k,v in exact.items());checks+=len(exact)
for key,expected in [('all_source_rows_each',28976),('all_inputU_each',535),('known_DE_each',12949),('unknown_truth_each',16027),('all4_actual_updates_each',8000),('all4_actual_updates_total',32000)]:
 assert r[key]==expected;checks+=1
assert r['native35_passed']==[s['arms'][a]['gates']['native35_passed'] for a in s['arms']]
assert r['selected_model'] is None and not r['adopted'] and not r['fully_solved'] and not r['all_source_full_qualification_PASS']
review=read(V/'independent_saved_review/SAVED_SOURCE_EXECUTION_REVIEW.json')
assert review['PASS'] and review['C39_C38_source_scale_control_exact'] and r['C39_previous_C38_source_outputs_and_draws_exact']
a=read(V/'factorial_source_readout/ORIGINAL_GATE_AND_SEPARATE_AUXILIARY_DIAGNOSTICS.json')
for arm in s['arms']:
 m=s['arms'][arm]['MR1840']['SOURCE_DEVELOPMENT']['input_valid_diagnostic']['CM_D_E_rows_D_E_U_columns']
 assert m==[[341,0,0],[0,49,0]]
for arm in list(s['arms'])[2:]:
 assert a['arms'][arm]['owner_weak_TRAIN_only']['original']['CM_rows_truth_T_F_BG_columns_prediction_T_F_BG']==[[53,0,0],[0,52,0],[0,0,148]]
receipt=dict(completed=True,PASS=True,table_count=14,visible_data_rows_compared=rows_checked,numeric_checks=checks,
 exact_CMs_formatted_precision_recall_U_and_all_MR_root_JSON_fields=True,
 MR_weak_TRAIN_and_owner253_training_fit_claims_supported=True,
 actual_C39_C38_control_claim_supported_by_saved_independent_review=True,
 no_new_model_or_feature_or_checkpoint_or_RGB_or_private_reads=True,
 table_names=table_names,
 bindings=[bind(V/n)for n in ('ROOT_READOUT_KO.html','ROOT_RESULTS_KO.json','ROOT_SOURCE_RESULT_OVERVIEW.png','factorial_source_readout/ARM_DATASET_SUMMARY.json','factorial_source_readout/FULL_NATIVE_MATERIAL_ASPECT_TABLES.json','independent_saved_review/SAVED_SOURCE_EXECUTION_REVIEW.json')])
output=H/'VISIBLE_TABLE_AND_SUMMARY_CHECKS.json'
with output.open('x')as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'PASS':True,'rows':rows_checked,'checks':checks,'receipt':bind(output)}))
