"""Independent saved48 observation/join/report QA. No source images/models/eval."""
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,csv,hashlib,io,json,os,re,stat
O=Path(__file__).parent;V=O.parent;S='part_visibility_D_E_S_UNKNOWN';Q='query_relation_on_near_off_UNKNOWN'
def need(x,m):
 if not x:raise AssertionError(m)
def rb(p):
 p=Path(p);need(p.is_absolute()and '..'not in p.parts and 'v112'not in str(p).casefold()and V in p.parents,'Only exact ownphase safe paths before FS')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for k in p.parts[1:-1]:n=os.open(k,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Regular no symlink')
 with os.fdopen(n,'rb')as f:return f.read()
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(rb(p)).hexdigest()}
def get(b):
 z=rb(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Saved binding drift');return json.loads(z)
def write(n,d):
 with (O/n).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def scalar(available,a,c):
 # Separate scalar rule implementation, no import/execution of root join/report.
 if available is False:return ('ORIGINAL_INPUT_U',None)
 vals=[a,c]
 for z in vals:
  need(z[S]in{'D','E','S','UNKNOWN'}and z[Q]in{'ON','NEAR','OFF','UNKNOWN'},'Every available slot reviewed')
  need(all(z[k]in{'YES','NO','UNKNOWN'}for k in ('target_instance_identified','intended_part_identified','location_error_possible')),'Observation field domain')
 if any(z[Q]=='OFF'or z['location_error_possible']=='YES'for z in vals):return ('LOCATION_ERROR_OR_OFF',None)
 states=(a[S],c[S])
 if states==('S','S'):return ('SELF_CONSENSUS',None)
 if states in {('D','D'),('E','E')}:
  ids=all(z['target_instance_identified']=='YES'and z['intended_part_identified']=='YES'for z in vals)
  relation=a[Q]==c[Q]and a[Q]in{'ON','NEAR'}
  evidence=all(z['location_error_possible']=='NO'and bool(z['short_visual_evidence'].strip())for z in vals)
  foreign=states[0]=='D'or all(bool(z['external_occluder_evidence'].strip())for z in vals)
  return ('ALIGNED_WEAK_CONSENSUS_CANDIDATE',states[0])if ids and relation and evidence and foreign else('INSUFFICIENT_ALIGNED_EVIDENCE',None)
 if states==('UNKNOWN','UNKNOWN'):return ('UNKNOWN_CONSENSUS',None)
 return ('DISAGREEMENT_OR_UNRESOLVED',None)
class HTML(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.rows=[];self.cells=None;self.text=None
 def handle_starttag(self,tag,attrs):
  self.links.extend(v for k,v in attrs if k in('href','src')and v)
  if tag=='tr':self.cells=[]
  if tag in('td','th'):self.text=''
 def handle_data(self,x):
  if self.text is not None:self.text+=x
 def handle_endtag(self,tag):
  if tag in('td','th')and self.text is not None:
   if self.cells is not None:self.cells.append(self.text)
   self.text=None
  if tag=='tr'and self.cells is not None:self.rows.append(self.cells);self.cells=None

def local_targets(name,text):
 if name.endswith('.html'):
  h=HTML();h.feed(text);urls=h.links
 else:urls=[a or b for a,b in re.findall(r'!?\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))\)',text)]
 n=0
 for u in urls:
  p=urlsplit(u)
  if p.scheme in('http','https','mailto')or not p.path:continue
  need(not p.scheme and not p.netloc and not p.path.startswith('/'),'Portable relative URL')
  path=Path(os.path.normpath(str(V/Path(name).parent/unquote(p.path))));need(V in path.parents and 'v112'not in str(path).casefold(),'No forbidden/outside local link')
  # Link existence only: NO image body/decode or source binding traversal.
  for q in reversed([path,*path.parents]):need(not stat.S_ISLNK(q.lstat().st_mode),'No linked symlink')
  need(stat.S_ISREG(path.lstat().st_mode),'Local linked file exists');n+=1
 return n

def main(cp):
 need(cp==V/'ROOT_REPORT_BUILD_COMPLETE.json','Exact root report completion only')
 need(bind(cp)['sha256']=='8d983866612c7f929ea23be6751066a718fc0f43f648973f5afa95633a9fe81c','Exact root report completed SHA');complete=json.loads(rb(cp));need(complete['completed']and complete['PASS']and complete['training_admission']is False,'Root report completed')
 need(complete['code']['path']==str(V/'build_root_source_observation_report.py'),'Exact root builder code binding');get_code=bind(Path(complete['code']['path']));need(get_code==complete['code'],'Final executed builder SHA')
 output={Path(x['path']).name:x for x in complete['outputs']};need(set(output)=={'PORTABLE_REPORT_KO.md','ROOT_REPORT_KO.md','PORTABLE_GALLERY_KO.md','ROOT_READOUT_KO.html','ROOT_RESULTS_KO.json'},'Exact five report outputs')
 for x in output.values():need(bind(Path(x['path']))==x,'Report output byte seal')
 result=get(output['ROOT_RESULTS_KO.json']);joined=get(result['joined']);manifest=get(joined['preview_manifest']);a=get(joined['observer_a']);c=get(joined['observer_c']);entry=get(joined['entry'])
 need(joined['completed']and joined['PASS']and entry['approved_saved_observation_join']is True and entry['training_admission']is False,'Actual join gate')
 expected=manifest['slots48'];need(len(expected)==48 and len(manifest['frames'])==6,'Full preview census')
 maps=[]
 for doc,observer in [(a,'A'),(c,'C')]:
  rr=doc['rows'];mm={x['row_id']:x for x in rr};need(len(rr)==len(mm)==48 and set(mm)=={x['row_id']for x in expected},'All48 unique observer IDs');maps.append(mm)
  need(all(x['observer']==observer and x['training_admission']is False and x['authority'].startswith('INDEPENDENT_ASSISTANT_SOURCE_TRAIN_WEAK')for x in rr),'Observer/weak authority/fitfalse')
 observed=[]
 for x in expected:
  aa,cc=(m[x['row_id']]for m in maps)
  for z in (aa,cc):
   need(all(z[k]==x[k]for k in ('row_id','frame_key','native_slot','query_xy','L','original_query_available')),'Original IDs/qL/native/U immutable')
   if not x['original_query_available']:need(z[S]=='UNKNOWN'and z[Q]=='UNKNOWN','Original U stays UNKNOWN')
  category,state=scalar(x['original_query_available'],aa,cc)
  observed.append({**{k:x[k]for k in ('row_id','frame_key','native_slot','query_xy','L','original_query_available')},'A_state':aa[S],'C_state':cc[S],'A_query_relation':aa[Q],'C_query_relation':cc[Q],'category':category,'consensus_state_candidate':state,'training_admission':False,'authority':'TWO_ASSISTANT_SOURCE_TRAIN_WEAK_NOT_GOLD'})
 need(observed==joined['rows'],'Every saved join row equals independent scalar rule')
 categories=dict(Counter(r['category']for r in observed));candidates=dict(Counter(r['consensus_state_candidate']for r in observed if r['consensus_state_candidate']))
 acount=dict(Counter(r['A_state']for r in observed));ccount=dict(Counter(r['C_state']for r in observed));need(joined['category_counts']==categories and joined['consensus_candidate_state_counts']==candidates and joined['observer_A_state_counts']==acount and joined['observer_C_state_counts']==ccount,'All raw/candidate denominators exact')
 need(joined['original_U']==sum(not r['original_query_available']for r in observed)==8 and all(x['native_slot']in range(8)for x in observed),'Original48/U8/native')
 need(all(sum(x['frame_key']==f['frame_key']for x in observed)==8 for f in manifest['frames']),'All six frames/native8')
 # CSV is original saved table, not a second label source.
 csvpath=V/'joined_source_observations_v1/JOINED_OBSERVATIONS48.csv';csvrows=list(csv.DictReader(io.StringIO(rb(csvpath).decode())))
 need(len(csvrows)==48 and all(rr=={k:(''if v is None else str(v))for k,v in x.items()}for rr,x in zip(csvrows,observed)),'Saved CSV all48 cells exact')
 # Root-requested descriptive axes, kept separate from prospective strict rule.
 equalD=[(maps[0][z['row_id']],maps[1][z['row_id']])for z in expected if maps[0][z['row_id']][S]=='D'and maps[1][z['row_id']][S]=='D']
 equalE=sum(maps[0][z['row_id']][S]=='E'and maps[1][z['row_id']][S]=='E'for z in expected)
 equalDrel=[(aa,cc)for aa,cc in equalD if aa[Q]in {'ON','NEAR'}and aa[Q]==cc[Q]]
 axes=dict(descriptive_after_observation_not_new_admission_rule=True,same_D=len(equalD),same_E=equalE,same_D_and_same_ON_or_NEAR=len(equalDrel),same_ON_or_NEAR_D_location_flag_disagreement=sum(aa['location_error_possible']!=cc['location_error_possible']for aa,cc in equalDrel),training_admission=False)
 need(result['separate_descriptive_visibility_query_axes']==axes,'Independent descriptive axes exact; not admission rule')
 perframe=[]
 for f in manifest['frames']:
  rr=[x for x in observed if x['frame_key']==f['frame_key']]
  perframe.append(dict(frame_key=f['frame_key'],slots=len(rr),U=sum(not z['original_query_available']for z in rr),A_states=dict(Counter(z['A_state']for z in rr)),C_states=dict(Counter(z['C_state']for z in rr)),categories=dict(Counter(z['category']for z in rr))))
 need(result['categories']==categories and result['consensus_D_candidates']==candidates.get('D',0)and result['consensus_E_candidates']==candidates.get('E',0)and result['perframe']==perframe and result['slots48']and result['original_U8'],'Root JSON every count exact')
 need(result['new_training_model_forward_admission_or_protected_data_access']is False and result['human_gold_or_model_accuracy_claim']is False and result['goal_solved']is False,'No fit/modelperformance/gold/adoption claim')
 md=rb(V/'PORTABLE_REPORT_KO.md').decode();need(rb(V/'ROOT_REPORT_KO.md')==md.encode(),'Portable/root MD byteidentity')
 table=dict((k,int(v))for k,v in re.findall(r'^\| ([A-Z_]+) \| ([0-9]+) \|$',md,re.M));need(table==categories,'MD all category table counts')
 need(f"D {candidates.get('D',0)}개, E {candidates.get('E',0)}개"in md,'MD headline candidates')
 need('독립 인간 평가·물리 native gold도 아니다'in md and '두 assistant의 일치가 정확성을 보증하지 않는다'in md and '새 학습·모델 평가·학습 채택은 수행하지 않았다'in md,'Weak authority and no performance claims explicit')
 need(f"같은 D 판단은 {axes['same_D']}개, 같은 E는 {axes['same_E']}개"in md and f"그중 {axes['same_ON_or_NEAR_D_location_flag_disagreement']}개는 location_error_possible 표시가 다르다"in md,'Descriptive axes wording exact');need('확인된 좌표 오류의 개수가 아니다'in md and '엄격한 준비 수를 전체 보임 부위의 수로 바꾸어 말하면 안 된다'in md,'Location candidate count and strict D are not physicalerror/visiblepart census');gallery=rb(V/'PORTABLE_GALLERY_KO.md').decode();need(len(re.findall(r'^\| [0-7] / (?:True|False) \|',gallery,re.M))==48,'MD gallery all48 rows')
 ht=rb(V/'ROOT_READOUT_KO.html').decode();parser=HTML();parser.feed(ht);obsrows=[z for z in parser.rows if z and re.fullmatch('[0-7] / (?:True|False)',z[0])]
 need(len(obsrows)==48 and all(len(z)==6 for z in obsrows),'HTML actual observation48 rows')
 for cells,x in zip(obsrows,observed):
  aa=maps[0][x['row_id']];cc=maps[1][x['row_id']]
  expectedcells=[f"{x['native_slot']} / {x['original_query_available']}",f"{x['A_state']} / {x['A_query_relation']} / {aa['location_error_possible']}",f"{x['C_state']} / {x['C_query_relation']} / {cc['location_error_possible']}",x['category'],aa['short_visual_evidence'],cc['short_visual_evidence']]
  need(cells==expectedcells,'HTML actual48 observer states/evidence/category exact')
 linkcount=sum(local_targets(name,rb(V/name).decode())for name in ('PORTABLE_REPORT_KO.md','ROOT_REPORT_KO.md','PORTABLE_GALLERY_KO.md','ROOT_READOUT_KO.html'))
 review=dict(completed=True,PASS=True,code=bind(Path(__file__)),root_report_complete=bind(cp),executed_report_builder=complete['code'],joined=result['joined'],observer_a=joined['observer_a'],observer_c=joined['observer_c'],join_actual_entry=joined['entry'],all48_original_rows_U8_qL_native_exact=True,independent_scalar_join_all48_exact=True,raw_A_counts=acount,raw_C_counts=ccount,categories=categories,strict_weak_candidate_counts=candidates,separate_descriptive_axes=axes,all_CSV_JSON_MD_HTML_counts_and48evidence_rows_exact=True,portable_local_links=linkcount,fit_admission_model_performance_gold_claim=False,original_source_images_or_arrays_GT_eval_models_read=0,generated_images_decoded=0,actual_toolviews=0,root_independent_sealing_precondition='Root actualentry procedural verification; distinct assistant values/row data do not independently prove blind or human-rater independence.',limits='Saved observer/join/report scalar+text QA only, no semantic relabel or adjudication.',goal_solved=False)
 write('REVIEW.json',review)
 text='V205 저장 관찰48행·원 q/L/native/U8·독립 scalar 결합·CSV/JSON/MD/HTML 표를 검산했다. PASS. 원상태 집계와 엄격 D/E 약합의 후보 집계는 별개이며 SELF/OFF/위치문제/UNKNOWN을 보존한다. 실제이미지·GT·평가·모델을 읽거나 관찰을 재판정하지 않았다. 관찰 합의는 source 약라벨 후보일 뿐 gold/성능/fit admission이 아니다.\n'
 with (O/'KO.md').open('x')as f:f.write(text)
 print(json.dumps(dict(completed=True,PASS=True,review=bind(O/'REVIEW.json'),KO=bind(O/'KO.md'),categories=categories,candidates=candidates,descriptive_axes=axes,local_links=linkcount),ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--complete',type=Path,required=True);a=p.parse_args();main(a.complete)
