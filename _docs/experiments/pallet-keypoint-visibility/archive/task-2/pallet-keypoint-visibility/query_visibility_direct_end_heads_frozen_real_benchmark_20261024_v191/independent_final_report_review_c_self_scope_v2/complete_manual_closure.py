"""Root-authorized manual QA closure: remaining report bindings/links only.

Not a third production QA rerun. The two preserved attempts already passed
numeric/table/CSV invariance before the over-specific caption string guard.
"""
from pathlib import Path
from html.parser import HTMLParser
import hashlib,json,os,re,stat
H=Path(__file__).absolute().parent;V=H.parent
def need(v,m):
 if not v:raise AssertionError(m)
def raw(p):
 p=Path(p);need(V in p.parents and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts),'Safe exact report path')
 fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for s in p.parts[1:-1]:
   nxt=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
  last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(last).st_mode),'Regular saved text')
 with os.fdopen(last,'rb') as f:return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(raw(p)).hexdigest())
def checked(b):
 d=raw(b['path']);need(hashlib.sha256(d).hexdigest()==b['sha256'],'Latest exact report unchanged');return d
e=json.loads(raw(H/'ROOT_ACTUAL_SELF_SCOPE_QA_ENTRY_V2.json'));final={Path(b['path']).name:b for b in e['final_reports']};contents={n:checked(b) for n,b in final.items()}
phase=json.loads(contents['ROOT_PHASE_RESULT.json']);need(all(final[n]==b for n,b in phase['reports'].items()),'Latest phase binds latest reports')
need(phase['completed'] is True and phase['PASS'] is True and phase['selected_model'] is None and phase['adopted'] is False and phase['fully_solved'] is False,'Goal remains unsolved/unadopted')
need(contents['ROOT_REPORT_KO.md']==contents['PORTABLE_REPORT_KO.md'],'Portable MD exact')
class Refs(HTMLParser):
 def __init__(self):super().__init__();self.refs=[]
 def handle_starttag(self,t,a):self.refs.extend(v for k,v in a if k in ['href','src'])
md=contents['ROOT_REPORT_KO.md'].decode();html=contents['ROOT_READOUT_KO.html'].decode();mr=re.findall(r'\]\(([^)]+)\)',md);hr=Refs();hr.feed(html)
for rel in mr+hr.refs:
 need(not rel.startswith('/') and not re.match(r'^[A-Za-z]+:',rel) and '..' not in Path(rel).parts and all('v112' not in s.casefold() for s in Path(rel).parts),'Local lexical guard')
 need(stat.S_ISREG((V/rel).lstat().st_mode),'Relative literal target exists, no decode')
need(len(mr)==24 and len(hr.refs)==9,'Expected24MD/9HTML refs')
receipt=dict(completed=True,PASS=True,review_method='Root explicitly approved manual closure after two administrative checker failures; no third full production QA rerun',root_scope=e['root_scope'],root_manual_closure_authorization='Parent /root message: already-passed numeric/preservation/CSV checks + direct semantic caption reading + remaining currentphase/reportbindings and24MD/9HTMLrefs; no thirdproduction rerun or new strictstringguard.',prior28_QA=e['prior_QA'],actual_V2_entry=bind(H/'ROOT_ACTUAL_SELF_SCOPE_QA_ENTRY_V2.json'),current_reports=e['final_reports'],first_failure=bind(H/'FIRST_ACTUAL_QA_FAILURE_PRESERVED.json'),second_failure=bind(H/'SECOND_ACTUAL_QA_FAILURE_PRESERVED.json'),failures_not_hidden=True,first_failure_detail='e89181: preservationmetadata omitted size_bytes; original/preserved actualSHA stillpresent. Rootapproved optional-size-only adapter, preserved originalattempt.',second_failure_detail='cfc3e2: all priorreport/byteSHA/nonselfmetrics/paired21/full136/CSV invariance assertions completed before captionguard. Guard accepted 추정/제공 but incorrectly omitted synonymous 추측; no report defect.',checks_completed_before_second_false_positive=['All preserved original actual-byte SHA values equal original/preserved metadata; optionalprovided sizes exact','Before-self5reports SHA equal prior28QA input reports','All existing nonself numeric fields including6primary/material/U/reference/public/native/68selection unchanged','Original table_self6 paired21 preserved exactly','New table_self_same_slot_all136_6 equals independent scalar sums from saved same_slot_pipeline selfsummary; full136 eachmodel D+binaryhidden+U conserved','Both summary and unchangedselfCSV cohort/group/model and4countcolumns exact'],paired_self21_preserved=True,full_self136_separate=True,full136_table='table_self_same_slot_all136_6',full136_all_WOOD_authority='Saved summary source rows, not a new label or inference',remaining115_reason_not_provided=True,remaining115_no_coordinate_U_or_physical_cause_attribution=True,manual_caption_semantic_PASS=True,manual_caption_read='HTML: 기존 요약에는 제외 사유 세부 집계가 없어 query 오류/inputU/물리 원인으로 추측하지 않는다. This explicitly denies invented causes; 추측하지 않는다 satisfies the intended no-causal-attribution condition. MD separately says 추측하지 않는다 and both self tables are binary responses, learned-self classifier가 아니다.',remaining_binding_QA_PASS=True,portable_relative_links=dict(MD=24,HTML=9),preservation=e['preservation_receipt'],saved_summary=e['saved_summary'],saved_self_csv=e['saved_self_csv'],manual_closure_check_code=bind(__file__),third_full_production_QA_rerun=False,new_strict_caption_guard=False,source_arrays_weights_models_GT_images_or_new_metric_eval_Git_network_reads_or_runs=0,actual_image_decode_or_view=0,scientific_reports_predictions_or_source_modified=False,selected_model=None,adopted=False,fully_solved=False,goal_status='active_unsolved')
with (H/'REVIEW.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,manual_closure=True,review=bind(H/'REVIEW.json'),remaining_MD_links=len(mr),remaining_HTML_links=len(hr.refs))))
