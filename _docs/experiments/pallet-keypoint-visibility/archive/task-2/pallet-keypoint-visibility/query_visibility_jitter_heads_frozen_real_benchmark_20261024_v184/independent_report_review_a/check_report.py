"""Offline report-only QA. Never reads underlying metric/label/model packets."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,hashlib,json,os,stat
H=Path(__file__).resolve().parent;V=H.parent
FIELDS=['table_primary','table_material','table_manual_near376','table_public_weak_center']
COUNTS=[4,8,4,3]
class Document(HTMLParser):
 def __init__(self):super().__init__(convert_charrefs=True);self.tables=[];self.links=[];self.current=None;self.row=None;self.cell=None;self.text=[]
 def handle_starttag(self,tag,attrs):
  attrs=dict(attrs)
  if tag=='table':self.current=[];self.tables.append(self.current)
  elif tag=='tr' and self.current is not None:self.row=[];self.current.append(self.row)
  elif tag in ('td','th') and self.row is not None:self.cell=[]
  if tag=='a' and 'href' in attrs:self.links.append(dict(tag=tag,attribute='href',value=attrs['href']))
  if tag=='img' and 'src' in attrs:self.links.append(dict(tag=tag,attribute='src',value=attrs['src']))
 def handle_data(self,data):
  self.text.append(data)
  if self.cell is not None:self.cell.append(data)
 def handle_endtag(self,tag):
  if tag in ('td','th') and self.cell is not None:self.row.append(''.join(self.cell));self.cell=None
  elif tag=='tr':self.row=None
  elif tag=='table':self.current=None
def need(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def safe_exists(p):
 s=os.path.abspath(p);need('_v112' not in s.lower(),'Forbidden path before access');parts=s.split('/')[1:];cur='/';hops=0
 while parts:
  n=os.path.normpath(os.path.join(cur,parts.pop(0)));need('_v112' not in n.lower(),'Forbidden path component')
  try:a=os.lstat(n)
  except FileNotFoundError:return False
  if stat.S_ISLNK(a.st_mode):
   t=os.readlink(n);need('_v112' not in t.lower(),'Forbidden symlink target');t=os.path.normpath(t if os.path.isabs(t) else os.path.join(os.path.dirname(n),t));need('_v112' not in t.lower(),'Forbidden expansion');hops+=1;need(hops<40,'Symlink loop');parts=t.split('/')[1:]+parts;cur='/'
  else:cur=n
 return True
def compare(result,document):
 need(len(document.tables)==4,'Exactly4 requested report tables')
 for key,n,table in zip(FIELDS,COUNTS,document.tables):
  need(len(result[key])==n,'Requested JSON rowcount '+key)
  expected=[[str(cell) for cell in row] for row in result[key]]
  need(table[1:]==expected,'All HTML data cells equal JSON '+key)
 return sum(len(row) for table in document.tables for row in table[1:])
def prepare():
 toy={k:[[i,'A&B',None] for i in range(n)] for k,n in zip(FIELDS,COUNTS)}
 text=''.join('<table><tr><th>h</th></tr>'+''.join('<tr>'+''.join('<td>'+str(x).replace('&','&amp;')+'</td>' for x in r)+'</tr>' for r in toy[k])+'</table>' for k in FIELDS)
 d=Document();d.feed(text);need(compare(toy,d)==57,'Invented allcell parser parity/entity regression')
 d.tables[0][1][1]='wrong'
 try:compare(toy,d);reject=False
 except AssertionError:reject=True
 need(reject,'Invented singlecell mismatch rejected')
 p=H/'PREPARATION.json'
 with p.open('x') as f:json.dump(dict(completed=True,PASS=True,prepared_only=True,code=bind(Path(__file__)),checks=['HTML entity decoded correctly/all19rows exact','Single changed tablecell rejected'],report_bodies_read=0,underlying_metrics_models_labels_or_pixels_read=0,run_only_after_root_report_generated_message=True),f,indent=2);f.write('\n')
 print(json.dumps(bind(p)))
def run():
 need(not (H/'AUTOMATED_OFFLINE_CHECKS.json').exists(),'Fresh report-only QA receipt')
 jp=V/'ROOT_RESULTS_KO.json';hp=V/'ROOT_READOUT_KO.html';result=json.loads(jp.read_text());source=hp.read_text();d=Document();d.feed(source);cells=compare(result,d)
 need(result['completed'] and result['PASS'] and result['selected_model'] is None and result['adopted'] is False and result['fully_solved'] is False and result['goal_status']=='active_unsolved','Goal/adoption claims')
 need(result['original_current_slots']==1784 and result['original_current_available']==1739 and result['original_current_inputU']==45,'Original current inventory claims')
 need(result['primary_D1147_E90_N1237'] is True and len(result['primary_all'])==4 and len(result['primary_by_material'])==8,'Original primary cohort')
 need(result['current_publicheld_repeated_exposure_not_blind'] and result['current_publicheld_eval_only'] and result['threshold_model_architecture_or_training_data_choice'] is False,'Evaluation-only/reused exposure')
 need(result['training_definition_diagnosis']['public_CF_support_truth_all_UNKNOWN'] and result['training_definition_diagnosis']['causal_real_failure_explanation_established'] is False,'Source-only definition diagnosis separated from real causality')
 txt=''.join(d.text)
 for phrase in ['real-CAL','이번 C52/C53 학습·보정·문턱 선택에는 현재 자료를 쓰지 않았고','fresh blind 평가라고 부르지 않습니다','UNKNOWN','인과 원인이 확정된 것은 아닙니다','SELF 136개','known 204개(T24/F96/BG84), UNKNOWN 156개','사람 D/E 표','서로 다른 출력','어느 모델도 채택하지 않았습니다']:
  need(phrase in txt,'Required scope/limit phrase '+phrase)
 references=[]
 for link in d.links:
  url=urlsplit(link['value'])
  if url.scheme or url.netloc:continue
  path=unquote(url.path)
  if not path:continue
  candidate=Path(os.path.normpath(str(V/path)));need(safe_exists(candidate),'Missing local reference '+path);references.append(dict(**link,path=str(candidate)))
 receipt=dict(completed=True,PASS=True,review_kind='Automated offline report-only checks; companion final review supplies human prose assessment.',code=bind(Path(__file__)),results=bind(jp),report=bind(hp),all_JSON_HTML_table_cells_equal=True,table_rowcounts=dict(zip(FIELDS,COUNTS)),data_cells_compared=cells,local_references=len(references),unique_local_paths=len(set(r['path'] for r in references)),all_local_refs_exist=True,references=references,source_diagnosis_and_legacy_scope_phrases_present=True,model_selected=None,goal_solved=False,HTML_browser_rendered=False,metric_recalculation_or_underlying_model_label_score_pixel_reads=0)
 p=H/'AUTOMATED_OFFLINE_CHECKS.json'
 with p.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(bind(p)))
if __name__=='__main__':
 a=argparse.ArgumentParser();m=a.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--run',action='store_true');z=a.parse_args();prepare() if z.prepare else run()
