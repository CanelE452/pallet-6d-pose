"""Root exact allowlists + final scientific report -> final opaque build entry."""
from pathlib import Path
from collections import Counter
import hashlib,json,os,stat
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
F=R/'V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON'
def safe(p):
 p=Path(p);assert p.is_absolute() and '..' not in p.parts and not any(t in str(p).casefold() for t in ('v112','cache','.pyc'))
 for a in [*reversed(p.parents),p]:
  try:m=os.lstat(a).st_mode
  except FileNotFoundError:continue
  assert not stat.S_ISLNK(m),str(a)
 return p
def bind(p):
 p=safe(p);h=hashlib.sha256();n=0
 with p.open('rb') as s:
  for b in iter(lambda:s.read(1<<20),b''):h.update(b);n+=len(b)
 return dict(path=str(p),sha256=h.hexdigest(),size_bytes=n)
def read(p,sha=None):
 p=safe(p);b=p.read_bytes()
 if sha:assert hashlib.sha256(b).hexdigest()==sha
 return json.loads(b)
def write(p,j):
 p=safe(p)
 with p.open('x',encoding='utf-8') as s:json.dump(j,s,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);s.write('\n')
 return bind(p)
def run():
 h=R/'V261_PUBLICATION_INVENTORY_C'
 base=read(h/'ALLOWLIST_PROPOSAL.json','39acebfc383a277dae5cc9d55620da7637ca07c2a03aba8a8bb225be5aa2ae05')
 supplement=read(h/'PUBLICATION_SUPPLEMENT_V2.json','33600f310dd44d6558f493ad2bb72c4fa06a9c29043183c49c6788d6b568d941')
 rows={}
 def add(b,group,rel=None):
  b={k:b[k] for k in ('path','sha256','size_bytes')};p=safe(b['path'])
  rel=rel or 'artifacts/'+p.relative_to(R).as_posix()
  assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and not any(t in rel.casefold() for t in ('v112','cache','.pyc'))
  if rel in rows:assert rows[rel]['source']==b;return
  rows[rel]=dict(relative_path=rel,group=group,source=b)
 for row in base['files']+supplement['additional_files']:add(row['source'],row['group'],row['relative_path'])
 # Named immediate administrative directories; no traversal or payload decoding.
 for folder,group in [('V261_PUBLICATION_INVENTORY_C','publication_inventory'),('V261_MANUAL_NEAR_SAVED_LEDGER_B','saved_manual_reference_diagnosis')]:
  directory=safe(R/folder)
  for child in sorted(os.listdir(directory)):
   assert not any(t in child.casefold() for t in ('v112','cache','.pyc'))
   p=safe(directory/child)
   if stat.S_ISREG(os.lstat(p).st_mode):add(bind(p),group)
 manual=R/'V261_MANUAL_NEAR_SAVED_LEDGER_B/results_v1'
 manifest=read(manual/'OUTPUT_BINDINGS.json','a4380637de24da1e76c3206cdcd89c8db4b1711833bd3c495cca35ddbac925dd')
 add(bind(manual/'OUTPUT_BINDINGS.json'),'saved_manual_reference_diagnosis')
 for name,b in manifest['files'].items():add(dict(path=str(manual/name),**b),'saved_manual_reference_diagnosis')
 for version in (1,2):
  directory=F/('final_readout_v%d'%version);c=read(directory/'COMPLETE.json')
  add(bind(directory/'COMPLETE.json'),'final_report')
  for b in c['files']:
   rel=b['relative_path'] if version==2 else None
   add(b,'final_report',rel)
 for name in ['prepare_v261_final_fit.py','replay_v261_public_selection.py','prepare_v261_teacher_groups.py','build_v261_final_readout.py','assemble_v261_publication_entry.py']:add(bind(R/name),'root_execution_code')
 add(bind(F/'FINAL_SCIENTIFIC_STATUS.json'),'final_scientific_status')
 # The root scopes carry exact inputs; source entries already included in base.
 for name in ['ROOT_PUBLIC_FEATURE_VERIFICATION.json','ROOT_PUBLIC_FULL_SELECTION_REPLAY.json']:
  add(bind(F/name),'source_metadata_and_QA')
 e=read(R/'V261_FINAL_PUBLICATION_C/BUILD_ENTRY_FALSE_TEMPLATE.json')
 e['files']=sorted(rows.values(),key=lambda row:row['relative_path'])
 e['coverage_contract']={g:[row['source'] for row in e['files'] if row['group']==g] for g in sorted({row['group'] for row in e['files']})}
 e['scientific_status']=read(F/'FINAL_SCIENTIFIC_STATUS.json')
 e['scientific_status']['public_source_and_source_model_metrics_are_not_native_current_accuracy']=True
 e['approved_actual_publication_package']=True
 e['root_scope']=write(F/'ACTUAL_PUBLICATION_BUILD_SCOPE.json',dict(approved_actual_publication_package=True,entry_contract={k:v for k,v in e.items() if k!='root_scope'}))
 eb=write(F/'ACTUAL_PUBLICATION_BUILD_ENTRY.json',e)
 print(json.dumps(dict(completed=True,PASS=True,entry=eb,logical_files=len(rows),source_bytes=sum(row['source']['size_bytes'] for row in rows.values()),groups=dict(Counter(row['group'] for row in rows.values())))))
if __name__=='__main__':run()
