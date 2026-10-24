"""Minimal root-entry-only self21/full136 presentation QA; no new metrics/data."""
from pathlib import Path
from html.parser import HTMLParser
import argparse,csv,hashlib,io,json,os,re,stat
H=Path(__file__).absolute().parent;V=H.parent
PROTOCOL=H/'PROTOCOL_V2_PRESERVATION_SHA_ADAPTER.json'
PRIOR_QA=dict(path=str(V/'independent_final_report_review_c/REVIEW.json'),sha256='02a1a759e24f1629fdde07a0cbebad8271bd7b786ae40185bd5f1db7df9149dd')
def need(v,m):
 if not v:raise AssertionError(m)
def raw(value):
 p=Path(value);need(V in p.parents and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts),'Exact safe report path')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for s in p.parts[1:-1]:
   nxt=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
  last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(last).st_mode),'Regular saved text only')
 with os.fdopen(last,'rb') as f:return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(raw(p)).hexdigest())
def checked(b):
 d=raw(b['path']);need(hashlib.sha256(d).hexdigest()==b['sha256'],'Bound report/metadata changed');return d
def obj(b):return json.loads(checked(b))
def write(p,o):
 with p.open('x') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n')
def preserved_record(item,d):
 need(hashlib.sha256(d).hexdigest()==item['preserved']['sha256']==item['original']['sha256'],'Original and preserved actualbyte SHA equality')
 for role in ['original','preserved']:
  if 'size_bytes' in item[role]:need(type(item[role]['size_bytes']) is int and len(d)==item[role]['size_bytes'],'Provided size equality')
 return True
def invariant(before,after,all_key):
 need(after['table_self6']==before['table_self6'],'Original paired21 table unchanged')
 allowed={'report_code','report_protocol','entry','limitations_KO'}
 need(all(after[k]==v for k,v in before.items() if k not in allowed),'All previous primary/material/U/ref/public/native/selection numbers unchanged')
 need(all_key not in before and all_key in after,'Separate new full136 table, never overwrite21')
 return True
def aggregate(source,cohort):
 result=[]
 for model in ['v61','legacy','C52','C53','C54','C55']:
  rr=[x for x in source if x['cohort']==cohort and x['model']==model]
  need(len(rr)==2 and {x['group'] for x in rr}=={'WOOD_PRIMARY105','SQUARE118'},'Both material source rows retained')
  result.append({'모델':model,'별도 human S N':sum(x['human_S_support'] for x in rr),'D 응답':sum(x['visible_response'] for x in rr),'binary hidden 응답':sum(x['binary_hidden_response'] for x in rr),'미결정':sum(x['unresolved'] for x in rr)})
 return result
def check_self(rows,n):
 need(len(rows)==6 and len({r['모델'] for r in rows})==6,'All six models')
 need(all(r['별도 human S N']==n and r['D 응답']+r['binary hidden 응답']+r['미결정']==n for r in rows),'Full self denominator and binary D/E/U conservation')
 return True
class Refs(HTMLParser):
 def __init__(self):super().__init__();self.refs=[]
 def handle_starttag(self,t,a):self.refs.extend(v for k,v in a if k in ['href','src'])
def run(path):
 e=json.loads(raw(path));need(e['approved_self_scope_report_QA'] is True,'Separate root actual QA entry')
 p=obj(e['protocol']);need(e['code']==bind(__file__)==p['code'] and e['prior_QA']==PRIOR_QA,'Frozen new QA and immutable prior28QA')
 scope=obj(e['root_scope']);need(scope.get('approved_independent_final_report_QA') is True or scope.get('approved_independent_self_report_QA') is True,'Explicit root updated scope')
 prior=obj(PRIOR_QA);need(prior['completed'] is True and prior['PASS'] is True and prior['check_count']==28,'Previous full28QA closed PASS')
 final={Path(b['path']).name:b for b in e['final_reports']};need(len(final)==5,'Exactly current5topreports');new={k:checked(b) for k,b in final.items()};r=json.loads(new['ROOT_RESULTS_KO.json']);phase=json.loads(new['ROOT_PHASE_RESULT.json'])
 preservation=obj(e['preservation_receipt']);old={}
 for item in preservation['records']:
  d=checked(item['preserved']);preserved_record(item,d);old[item['original']['path']]=d
 previous={Path(b['path']).name:b for b in prior['current_reports']}
 for name,b in previous.items():need(hashlib.sha256(old[b['path']]).hexdigest()==b['sha256'],'Before-self-repair5reports equal prior28QA bytes')
 before=json.loads(old[str(V/'ROOT_RESULTS_KO.json')]);allkey=e['all_self_table_key'];invariant(before,r,allkey)
 summary=obj(e['saved_summary']);source=summary['self_separate'];need(r['self_separate_source_tables']==source,'Stored self source table unchanged')
 paired=aggregate(source,'paired_same_query_RGB');full=aggregate(source,'same_slot_pipeline');check_self(paired,21);check_self(full,136)
 need(r['table_self6']==paired and r[allkey]==full,'Both displayed selftables exactly savedsummary scalar sums')
 csvrows=list(csv.DictReader(io.StringIO(checked(e['saved_self_csv']).decode())))
 for x in source:
  match=[z for z in csvrows if all(z[k]==str(x[k]) for k in ['cohort','group','model'])];need(len(match)==1 and all(int(match[0][k])==x[k] for k in ['human_S_support','visible_response','binary_hidden_response','unresolved']),'Saved summary/selfCSV agree, no new aggregation source')
 md=new['ROOT_REPORT_KO.md'].decode();htm=new['ROOT_READOUT_KO.html'].decode();need(new['ROOT_REPORT_KO.md']==new['PORTABLE_REPORT_KO.md'],'Portable/sourceMD exact')
 for txt in [md,htm]:need(all(t in txt for t in ['136','21','115','same_slot_pipeline','paired_same_query_RGB']) and ('이유' in txt or '사유' in txt) and ('추정' in txt or '제공' in txt),'Both cohort names/denominators and115unknown-exclusion caveat displayed')
 need('learned-self' in md and ('아니' in md or '않' in md),'Both selftables binary responses not learned-self success')
 for name,b in phase['reports'].items():need(final[name]==b,'Current phase rebinds current report hashes')
 need(r['selected_model'] is None and r['adopted'] is False and r['fully_solved'] is False and phase['goal_status']=='active_unsolved','No new model/adoption/goal claim')
 refs=Refs();refs.feed(htm);links=re.findall(r'\]\(([^)]+)\)',md)
 for rel in links+refs.refs:need(not rel.startswith('/') and not re.match(r'^[A-Za-z]+:',rel) and '..' not in Path(rel).parts and all('v112' not in s.casefold() for s in Path(rel).parts) and stat.S_ISREG((V/rel).lstat().st_mode),'Portable refs safe and exist; no body/decode')
 receipt=dict(completed=True,PASS=True,QA_code=bind(__file__),protocol=e['protocol'],actual_entry=bind(path),scope=e['root_scope'],prior28_QA=PRIOR_QA,final_reports=e['final_reports'],before_self_reports_unchanged_metrics=True,paired21_table_preserved=True,all136_table_matches_saved_summary_CSV=True,full136_plus_paired21_not_added_as_disjoint=True,remaining115_exclusion_reason_not_in_saved_summary=True,remaining115_not_attributed_to_coordinate_error_or_inputU=True,no_learned_self_or_external_gold_claim=True,summary=e['saved_summary'],self_CSV=e['saved_self_csv'],all_self_table_key=allkey,full_self_table=full,paired_self_table=paired,portable_links=dict(MD=len(links),HTML=len(refs.refs)),source_arrays_original_labels_images_model_new_evaluation_Git_reads_or_runs=0,image_decode_or_toolviews=0,new_model_threshold_or_admission=False,selected_model=None,adopted=False,fully_solved=False)
 write(H/'REVIEW.json',receipt);print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'))))
def prepare():
 row=lambda n:{'모델':'v61','별도 human S N':n,'D 응답':n-2,'binary hidden 응답':1,'미결정':1}
 before=dict(table_self6=[row(21)],primary={'N':1237,'D':1147,'E':90},public={'known':204,'U':156});after=dict(before,table_self_all136=[row(136)])
 checks=[('pairedtable_and_primary_unchanged',invariant(before,after,'table_self_all136'))]
 for name,change in [('primary_drift',dict(after,primary={'N':1236})),('paired_overwritten',dict(after,table_self6=[row(136)]))]:
  try:invariant(before,change,'table_self_all136');ok=False
  except AssertionError:ok=True
  checks.append((name+'_rejected',ok))
 tables=[dict(row(136),모델=m) for m in ['v61','legacy','C52','C53','C54','C55']];checks.append(('all136_binary_count_conservation',check_self(tables,136)))
 try:check_self([dict(x,미결정=0) for x in tables],136);ok=False
 except AssertionError:ok=True
 checks.append(('unknown_denominator_drift_rejected',ok))
 fixture=b'invented original bytes';hh=hashlib.sha256(fixture).hexdigest();record={'original':{'sha256':hh},'preserved':{'sha256':hh}};checks.append(('SHA_only_no_size_preservation_accepted',preserved_record(record,fixture)))
 try:preserved_record({'original':{'sha256':hh,'size_bytes':1},'preserved':{'sha256':hh}},fixture);ok=False
 except AssertionError:ok=True
 checks.append(('provided_size_drift_rejected',ok))
 result=dict(completed=True,PASS=all(x for _,x in checks),invented_only=True,checks=[dict(name=n,PASS=v) for n,v in checks],actual_regenerated_report_or_summary_CSV_reads=0)
 write(H/'GENERATED_INVENTED_SELF_QA_CHECKS_V2.json',result)
 protocol=dict(completed=True,prepared_only=True,code=bind(__file__),prior_QA=PRIOR_QA,schema_adapter='Rootexplicit e71cdd4f: original/preserved/actualbyte SHA remains exact; optional size checked only ifprovided. Firstfailure preserved; exactly one additional actual run.',inherited_code_binding=bind(H/'INHERITED_QA_CODE_BINDING.json'),scope='Newrootentry only: compare latest5topreports to before-selfpresentation preserved5fromprior28QA; all previous nonself metrics exact, table_self6 paired21 retained, new136table from same_slot_pipeline savedsummary/selfCSV;115 reason unknown, no attribution or learnedself/externalgold. No original labels/images/arrays/models/eval.',required_entry_fields=['approved_self_scope_report_QA','code','protocol','root_scope','prior_QA','final_reports','preservation_receipt','saved_summary','saved_self_csv','all_self_table_key'],actual_report_body_reads_before_entry=0)
 write(PROTOCOL,protocol);template=dict(approved_self_scope_report_QA=False,code=protocol['code'],protocol=bind(PROTOCOL),root_scope={'path':None,'sha256':None},prior_QA=PRIOR_QA,final_reports=[],preservation_receipt={'path':None,'sha256':None},saved_summary={'path':str(V/'fixed_direct_end_real_visual_v1/SUMMARY_AND_TABLES.json'),'sha256':'478fa6dcc69bbc5784f35f4bacbf6513a0dcc7860bdfc0a1265e3775303bc141'},saved_self_csv={'path':str(V/'fixed_direct_end_real_visual_v1/SELF_SEPARATE_BINARY_RESPONSES.csv'),'sha256':None},all_self_table_key=None)
 write(H/'ROOT_ACTUAL_SELF_SCOPE_QA_ENTRY_TEMPLATE_V2_APPROVED_FALSE.json',template);write(H/'PREPARATION_COMPLETE_V2.json',dict(completed=True,PASS=result['PASS'],code=protocol['code'],protocol=bind(PROTOCOL),checks=bind(H/'GENERATED_INVENTED_SELF_QA_CHECKS_V2.json'),false_entry=bind(H/'ROOT_ACTUAL_SELF_SCOPE_QA_ENTRY_TEMPLATE_V2_APPROVED_FALSE.json'),actual_regenerated_report_or_summary_CSV_reads=0));print(json.dumps(result))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--entry',type=Path);p=a.parse_args();prepare() if p.prepare else run(p.entry)
