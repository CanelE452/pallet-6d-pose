"""Frozen completed V214/V215 SOURCE diagnosis and public TRAIN preview publication; opaque text/PNG, no Git."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,hashlib,json,os,posixpath,re,stat,tempfile
H=Path(__file__).absolute().parent;P=H.parent.parent
PHASE=P/'query_visibility_source_train_modality_reliance_diagnosis_20261025_v214'
PHASE215=P/'query_visibility_public_train_visible_part_coverage_expansion_20261025_v215'
PHASES=(PHASE,PHASE215);LABELS=('V214','V215')
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024');PREFIX='_docs/experiments/pallet-keypoint-visibility/';PUB=REPO/PREFIX
ARCHIVE_PREFIX='archive/task-2/pallet-keypoint-visibility/'
DELTA='publication/deltas/V215_20261025_v1/';INDEX='EXPERIMENT_INDEX_THROUGH_V215_KO.md';PRIOR='EXPERIMENT_INDEX_THROUGH_V213_KO.md'
PROTOCOL=H/'PROTOCOL_BEFORE_ACTUAL_V214_V215_ENUMERATION_OR_COPY.json'
ROOT_PREP_SCOPE={'path':str(P/'publication_root_preparation/ROOT_PROSPECTIVE_V214_V215_COMPLETED_PUBLICATION_PREPARATION_SCOPE.json'),'sha256':'17e23eda8e5b97c6ed8894afea606465a2c2e7e80cc2d2f3a95d6c1badb777cb'}
INHERITED={'path':str(P/'publication_root_preparation/V211_PUBLICATION_PREPARATION_C/stage_completed_v211.py'),'sha256':'624787eece70824d989a233d092fff877450a80a738eec053cd79be77dd7fc97'}
EXPECTED_PARENT='25d7864ef31508595d83b65b4c68a440b929cfc1'
LIMIT=100*1024**2;BATCH=480*1024**2;MAX_ORIGINAL_FILES=600;MAX_ORIGINAL_BYTES=300*1024**2
TEXT={'.py','.json','.jsonl','.csv','.tsv','.md','.txt','.html','.htm','.patch','.diff','.log','.svg'}
RAW={'.npz','.npy','.pt','.pth','.bin','.safetensors','.pkl','.pickle'}
REQUIRED_ROLES=tuple(label+'_'+role for label in LABELS for role in('ROOT_PHASE_RESULT','ROOT_FINAL_FREEZE'))
REPORTS=[ARCHIVE_PREFIX+root.name+'/PORTABLE_REPORT_KO.md'for root in PHASES]
RULES={'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----','AWS':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b','GitHub':r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b','OpenAI':r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b','Slack':r'\bxox[baprs]-[A-Za-z0-9-]{20,}\b'}

def need(x,m):
 if not x:raise AssertionError(m)

def safe(p):
 p=Path(p);need(p.is_absolute()and'..'not in p.parts and all('v112'not in q.casefold()for q in p.parts),'Forbidden path before filesystem access');return p

def parentfd(p,create=False):
 p=safe(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for q in p.parts[1:-1]:
   try:n=os.open(q,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
   except FileNotFoundError:
    need(create,'Missing parent');os.mkdir(q,dir_fd=fd);n=os.open(q,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
   os.close(fd);fd=n
  return p,fd
 except BaseException:os.close(fd);raise

def rb(p):
 p,fd=parentfd(p)
 try:n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Only regular file')
 with os.fdopen(n,'rb')as f:return f.read()

def digest(z):return hashlib.sha256(z).hexdigest()

def bind(p):return dict(path=str(p),sha256=digest(rb(p)))

def checked(z):need(bind(z['path'])==z,'SHA binding drift');return json.loads(rb(z['path']))

def enc(z):return(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()

def new(p,z):
 p,fd=parentfd(p,True)
 try:n=os.open(p.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'wb')as f:f.write(z)

def replace_readme(p,z,expected):
 need(str(p)==str(REPO/'readme.md'),'Only mutable README');need(digest(rb(p))==expected,'README changed since root entry')
 p,fd=parentfd(p)
 try:n=os.open(p.name,os.O_WRONLY|os.O_TRUNC|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'wb')as f:f.write(z)

def frozen_phase(q,phase_result=None):
 need(q.get('completed')is True and q.get('frozen')is True and q.get('all_scientific_phase_files_final_no_more_writes')is True and q.get('administrative_publication_receipts_written_elsewhere')is True,'Actual root final freeze fields')
 b=q.get('phase_result');need(isinstance(b,dict)and set(b)=={'path','sha256'}and re.fullmatch('[0-9a-f]{64}',b['sha256']),'Freeze phase_result binding schema');safe(b['path'])
 if phase_result is not None:need(b==phase_result,'Freeze exact phase_result role binding')
 return True

def inventory(root):
 root=safe(root);out=[];excluded=[];stack=[root]
 while stack:
  d=stack.pop()
  with os.scandir(d)as it:
   for q in sorted(it,key=lambda x:x.name):
    need('v112'not in q.name.casefold(),'Forbidden basename before stat')
    if q.name in {'matplotlib_cache','__pycache__'}:excluded.append(str(Path(q.path).relative_to(root)));continue
    mode=q.stat(follow_symlinks=False).st_mode;need(not stat.S_ISLNK(mode),'Source symlink refused')
    if stat.S_ISDIR(mode):stack.append(Path(q.path))
    else:
     need(stat.S_ISREG(mode),'Nonregular source refused');source=Path(q.path);suffix=source.suffix.casefold()
     if suffix in RAW:
      excluded.append(dict(path=str(source.relative_to(root)),reason='ALL source array/model/input/target binaries excluded before header/body/hash; zero binary whitelist'));continue
     need(suffix in TEXT|{'.png'},'Only own code/text/table/vector/PNG records; unexpected binary refused before body')
     need(q.stat(follow_symlinks=False).st_size<LIMIT,'Each Gitfile<100MiB before body access');out.append(source)
 return sorted(out),excluded

def hits(z):
 text=z.decode('utf-8');return{k:len(re.findall(v,text))for k,v in RULES.items()if re.search(v,text)}

class Attrs(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in('href','src')and v)

def links(doc,z):
 text=z.decode('utf-8')
 if Path(doc).suffix.lower()in('.html','.htm'):p=Attrs();p.feed(text);found=p.links
 else:found=re.findall(r'!?\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))\)',text);found=[a or b for a,b in found]
 out=[]
 for v in found:
  u=urlsplit(v)
  if u.scheme in('http','https','mailto','tel')or not u.path:continue
  need(not u.scheme and not u.netloc and not u.path.startswith('/'),'Portable local link must be relative')
  target=posixpath.normpath(posixpath.join(posixpath.dirname(doc),unquote(u.path)))
  need(not target.startswith('../')and target!='..','Link outside publication root');safe(PUB/target);out.append(target)
 return out

def existing_target(p):
 # Check ordinary file/directory components, never enumerate/follow a source binding.
 p=safe(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for i,q in enumerate(p.parts[1:]):
   flags=os.O_RDONLY|os.O_NOFOLLOW|(os.O_DIRECTORY if i<len(p.parts)-2 else 0);n=os.open(q,flags,dir_fd=fd);os.close(fd);fd=n
  mode=os.fstat(fd).st_mode;return stat.S_ISREG(mode)or stat.S_ISDIR(mode)
 finally:os.close(fd)

def original_destination(root,source):
 need(root in PHASES and root in safe(source).parents,'Exact owned source root')
 return ARCHIVE_PREFIX+root.name+'/'+str(source.relative_to(root))

def role_phase(role):
 need(role in REQUIRED_ROLES,'Exact completion role')
 return PHASES[LABELS.index(role.split('_',1)[0])]

def publication_headers(e):
 z=e['publication_prefixes'];need(set(z)=={'index_utf8','readme_utf8'},'Root exact prose keys')
 out=[]
 for key in ('index_utf8','readme_utf8'):
  value=z[key];need(isinstance(value,str)and value.strip(),'Root final prose required')
  data=value.encode();need(len(data)<65536 and not hits(data),'Bounded credential-free prose');out.append(data)
 return tuple(out)

def portable_pages(e):
 pages=e['portable_pages'];need(isinstance(pages,list)and len(pages)==len(set(pages))and 2<=len(pages)<=30,'Bounded unique root-selected portable pages')
 need({INDEX,*REPORTS}<=set(pages),'Index and both root PORTABLE pages mandatory')
 for rel in pages:
  need(rel==INDEX or any(rel.startswith(ARCHIVE_PREFIX+root.name+'/')for root in PHASES),'Pages exact two new phases only')
  need(Path(rel).suffix.lower()in{'.md','.html','.htm'}and posixpath.normpath(rel)==rel,'Portable page path')
  safe(PUB/rel)
 return set(pages)

def selftest():
 checks=[]
 def add(name,v):need(v,name);checks.append(name)
 add('SHA_kernel',digest(b'abc')=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
 for bad in('/invented/V112/a','/invented/../a'):
  try:safe(bad);ok=False
  except AssertionError:ok=True
  add('lexical_reject_before_FS_'+str(len(checks)),ok)
 with tempfile.TemporaryDirectory(dir=H,prefix='invented_')as t:
  d=Path(t);(d/'proof.json').write_bytes(b'{}');(d/'view.png').write_bytes(b'opaque invented PNG');(d/'table.tsv').write_bytes(b'a\tb\n');(d/'figure.svg').write_bytes(b'<svg/>')
  (d/'__pycache__').mkdir();(d/'__pycache__/unread').write_bytes(b'unread');(d/'raw.npz').write_bytes(b'never hashed or decoded')
  ff,ee=inventory(d);add('positive_text_PNG_and_binary_cache_excluded',set(p.name for p in ff)=={'proof.json','view.png','table.tsv','figure.svg'}and len(ee)==2)
  new(d/'exact.txt',b'exact');add('exclusive_byte_copy',rb(d/'exact.txt')==b'exact')
  try:new(d/'exact.txt',b'overwrite');ok=False
  except FileExistsError:ok=True
  add('no_overwrite',ok)
  (d/'link').symlink_to(d/'proof.json')
  try:inventory(d);ok=False
  except AssertionError:ok=True
  add('symlink_refused_before_read',ok)
 f=dict(completed=True,frozen=True,all_scientific_phase_files_final_no_more_writes=True,administrative_publication_receipts_written_elsewhere=True,phase_result=dict(path=str(PHASE/'ROOT_PHASE_RESULT.json'),sha256='a'*64))
 add('strict_authoritative_freeze',frozen_phase(f,f['phase_result']))
 try:frozen_phase(dict(f,frozen=False));ok=False
 except AssertionError:ok=True
 add('false_freeze_refused',ok)
 e=dict(publication_prefixes=dict(index_utf8='# SOURCE V214\n\n',readme_utf8='# SOURCE V214\n\n'),portable_pages=[INDEX,*REPORTS])
 ih,rh=publication_headers(e);add('root_prose_history_suffix',(ih+b'oldindex').endswith(b'oldindex')and(rh+b'oldREADME').endswith(b'oldREADME'))
 add('two_phases_4roles_pages',len(PHASES)==2 and len(REQUIRED_ROLES)==4 and portable_pages(e)=={INDEX,*REPORTS})
 add('same_named_source_files_disjoint',len({original_destination(root,root/'COMPLETE.json')for root in PHASES})==2)
 add('relative_sibling_link',links('archive/a/r.md',b'[prior](../b/r.md)')==['archive/b/r.md'])
 add('credential_format_only',hits(('ghp_'+'x'*40).encode())=={'GitHub':1})
 return dict(completed=True,PASS=True,invented_only=True,checks=checks,source_or_repository_body_access=0,Git_calls=0)

def prepare():
 q=selftest();new(H/'GENERATED_INVENTED_CHECKS.json',enc(q))
 p=dict(completed=True,prepared_only=True,code=bind(__file__),checks=bind(H/'GENERATED_INVENTED_CHECKS.json'),root_prospective_scope=ROOT_PREP_SCOPE,inherited_publisher=INHERITED,supersedes_unexecuted_V214_only_code={'path':str(P/'publication_root_preparation/V214_PUBLICATION_PREPARATION_C/stage_completed_v214.py'),'sha256':'83af06fa8eac5f9f7dad30ed6c2f7af327ff4be0b47b114757ce6688416c67a5'},
  source_roots=list(map(str,PHASES)),required_completion_roles=list(REQUIRED_ROLES),expected_parent=EXPECTED_PARENT,
  policy='Exactly final frozen V214 and V215 only; original byte archive, positive code/text/CSV/JSONL/PNG (SVG is XML text); all source NPZ/NPY/weights excluded before header/body/hash; binding pointers never followed.',
  byte_security_kernel='V211 functions need/safe/parentfd/rb/digest/bind/checked/enc/new/replace_readme/frozen_phase/inventory/hits/Attrs/links/existing_target/original_destination/role_phase unchanged.',
  strict_freeze_fields=dict(completed=True,frozen=True,all_scientific_phase_files_final_no_more_writes=True,administrative_publication_receipts_written_elsewhere=True,phase_result='Exact result-role binding'),
  required_actual_scope_fields=dict(approved_bounded_publication_copy_stage=True,approved_git_commit_or_push_by_child=False,completed_source_phases=[dict(path=str(root),freeze_sha256='ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE binding SHA')for root in PHASES],expected_parent_main=EXPECTED_PARENT,remote='https://github.com/CanelE452/pallet-6d-pose.git',known_ignored_cache_exclusions=[]),
  original_file_cap=MAX_ORIGINAL_FILES,original_byte_cap=MAX_ORIGINAL_BYTES,each_file_cap_strict=LIMIT,batch_cap_strict=BATCH,
  future_root_supplies='Exact final index/readme UTF8 prefixes, portable pages, actual scope, result/freeze and prior THROUGH_V213/readme SHA bindings. No current results assumed.',
  mandatory_portable_pages=[INDEX,*REPORTS],Git_add_commit_push=False,source_phase_writes=0,automatic_retry=False)
 new(PROTOCOL,enc(p))
 template=dict(approved_actual_V214_V215_stage=False,code=p['code'],protocol=bind(PROTOCOL),root_prospective_scope=ROOT_PREP_SCOPE,root_scope=None,
  source_roots=list(map(str,PHASES)),initial_remote_main=None,repository=str(REPO),completion_receipts=[],
  previous_index=dict(path=str(PUB/PRIOR),sha256=None),global_readme_before=dict(path=str(REPO/'readme.md'),sha256=None),
  publication_prefixes=dict(index_utf8=None,readme_utf8=None),portable_pages=[INDEX,*REPORTS],output_directory=str(P/'publication_root_preparation/V215_PUBLICATION_STAGE_C'),automatic_retry=False,Git_add_commit_push=False)
 new(H/'ROOT_ACTUAL_ENTRY_FALSE_TEMPLATE.json',enc(template))
 new(H/'FINAL_READY.json',enc(dict(completed=True,PASS=True,FINAL_READY_FROZEN=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],false_entry=bind(H/'ROOT_ACTUAL_ENTRY_FALSE_TEMPLATE.json'),actual_source_enumeration_copy=0,Git_calls=0)))
 print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),checks=len(q['checks']))))

def run(entry):
 e=json.loads(rb(entry));need(e['approved_actual_V214_V215_stage']is True and e['automatic_retry']is False and e['Git_add_commit_push']is False,'Root copy-stage authority only')
 p=checked(e['protocol']);need(e['code']==bind(__file__)==p['code']and e['protocol']==bind(PROTOCOL),'Frozen code/protocol');need(checked(p['checks'])['PASS']is True,'Invented preflight PASS')
 need(e['root_prospective_scope']==ROOT_PREP_SCOPE,'Exact prospective authority');prep=checked(ROOT_PREP_SCOPE);need(prep['approved_actual_enumeration_or_copy']is False,'Preparation remains copyfalse')
 scopebinding=e['root_scope'];need(isinstance(scopebinding,dict)and safe(scopebinding['path']).parent==P/'publication_root_preparation'and scopebinding!=ROOT_PREP_SCOPE,'Separate root actual scope required')
 scope=checked(scopebinding);need(scope['approved_bounded_publication_copy_stage']is True and scope['approved_git_commit_or_push_by_child']is False,'Root actual stage/no Git')
 declared=scope['completed_source_phases'];need([r['path']for r in declared]==e['source_roots']==list(map(str,PHASES)),'Exact V214 and V215 roots only')
 need(e['repository']==str(REPO)and e['initial_remote_main']==scope['expected_parent_main']==EXPECTED_PARENT,'Root declared exact parent')
 need(scope['remote']=='https://github.com/CanelE452/pallet-6d-pose.git','Exact repository remote')
 roles={}
 for r in e['completion_receipts']:
  need(r['role']not in roles,'Unique role');root=role_phase(r['role']);need(root in safe(r['binding']['path']).parents,'Exact source role')
  z=checked(r['binding']);need(r['expected_fields'],'Exact completion expectations')
  for key,value in r['expected_fields'].items():need(z[key]==value,'Completion field '+key)
  roles[r['role']]=r
 need(set(roles)==set(REQUIRED_ROLES),'Exactly four result/freeze roles')
 for label,root,decl in zip(LABELS,PHASES,declared):
  result=roles[label+'_ROOT_PHASE_RESULT']['binding'];freeze=roles[label+'_ROOT_FINAL_FREEZE']['binding']
  need(result['path']==str(root/'ROOT_PHASE_RESULT.json')and freeze['path']==str(root/'ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json')and freeze['sha256']==decl['freeze_sha256'],'Exact frozen phase bindings')
  need(checked(result).get('completed')is True,'Root phase result completed');frozen_phase(checked(freeze),result)
 pages=portable_pages(e);ih,rh=publication_headers(e)
 out=safe(e['output_directory']);need(out==P/'publication_root_preparation/V215_PUBLICATION_STAGE_C','Exact fresh admin stage');out.mkdir(exist_ok=False)
 new(out/'STARTED.json',enc(dict(completed=False,entry=bind(entry),automatic_retry=False)))
 try:
  need(e['previous_index']['path']==str(PUB/PRIOR)and e['global_readme_before']['path']==str(REPO/'readme.md'),'Exact prior index/README')
  oldindex=rb(e['previous_index']['path']);need(digest(oldindex)==e['previous_index']['sha256'],'Prior index unchanged')
  oldreadme=rb(e['global_readme_before']['path']);need(digest(oldreadme)==e['global_readme_before']['sha256'],'README before unchanged')
  inventory_by_root={};excluded={}
  for root in PHASES:inventory_by_root[root],excluded[root.name]=inventory(root)
  original=[(root,source)for root in PHASES for source in inventory_by_root[root]];need(original and len(original)<=MAX_ORIGINAL_FILES,'Bounded nonempty combined source count')
  original_bytes=sum(os.stat(source,follow_symlinks=False).st_size for _,source in original);need(original_bytes<=MAX_ORIGINAL_BYTES,'Combined source bytes cap before copy')
  records=[];physical=[];texts={};source_hashes={}
  def store(rel,data,source,category):
   need(len(data)<LIMIT,'Each Gitfile<100MiB');need(not hits(data)if Path(rel).suffix.lower()in TEXT else True,'Credential pattern, values withheld')
   new(PUB/rel,data);need(rb(PUB/rel)==data,'Exact destination bytes')
   physical.append(dict(path=PREFIX+rel,sha256=digest(data),size_bytes=len(data)))
   records.append(dict(source_root_id='P'if category=='original_scientific_record'else'PUBLICATION_ADMIN_CODE'if category=='publication_tool'else'DELTA_DERIVED',source_absolute=str(source),source_relative_path=str(source.relative_to(P))if P in source.parents else rel,source_sha256=digest(data),size_bytes=len(data),published_relative=rel,published_sha256=digest(data),status='PUBLISHED_RAW',category=category,exclusion_reason=None,gzipchunks=None))
   if rel in pages:texts[rel]=data
  for root,source in original:
   data=rb(source);source_hashes[str(source)]=digest(data);store(original_destination(root,source),data,source,'original_scientific_record')
  for source in(Path(__file__).absolute(),PROTOCOL,H/'GENERATED_INVENTED_CHECKS.json'):store('tools/v215_lean_publication/'+source.name,rb(source),source,'publication_tool')
  store(INDEX,ih+oldindex,PUB/INDEX,'versioned_index');need(rb(PUB/INDEX).endswith(oldindex),'Historical index suffix')
  store(DELTA+'GLOBAL_README_BEFORE_V215.md',oldreadme,PUB/(DELTA+'GLOBAL_README_BEFORE_V215.md'),'old_global_snapshot')
  manifest=dict(format='immutable-completed-phase-delta-v1',phase='V214_V215_COMPLETED_SOURCE_DIAGNOSIS_AND_PUBLIC_TRAIN_PREVIEW',source_roots=list(map(str,PHASES)),previous_index=e['previous_index'],root_entry=bind(entry),completion_receipts=e['completion_receipts'],entries=records,owned_binary_whitelist=[],excluded_cache_paths=excluded,source_bindings_followed=0,old_archive_or_manifest_paths_modified=False,no_second_closing_required=True)
  mz=enc(manifest);new(PUB/(DELTA+'DELTA_MANIFEST.json'),mz);physical.append(dict(path=PREFIX+DELTA+'DELTA_MANIFEST.json',sha256=digest(mz),size_bytes=len(mz)))
  need(set(texts)==pages,'Every root-selected portable entry present');linkrows=[]
  for doc,data in texts.items():
   targets=links(doc,data)
   for target in targets:need(existing_target(PUB/target),'Missing portable link '+doc+' -> '+target)
   linkrows.append(dict(page=doc,local_links=len(targets)))
  rootlinks=links('ROOT_REPO_README.md',rh)
  for target in rootlinks:need(existing_target(REPO/target),'New README target missing')
  linkrows.append(dict(page='readme.md newprefix only; oldsuffix preserved',local_links=len(rootlinks)))
  for root in PHASES:
   final,excluded_final=inventory(root);need(final==inventory_by_root[root]and excluded_final==excluded[root.name],'Final exact source inventories unchanged')
  newreadme=rh+oldreadme;need(newreadme.endswith(oldreadme),'Historical README suffix');replace_readme(REPO/'readme.md',newreadme,e['global_readme_before']['sha256']);physical.append(dict(path='readme.md',sha256=digest(newreadme),size_bytes=len(newreadme)))
  for source,h in source_hashes.items():need(digest(rb(source))==h,'Original source final SHA unchanged')
  for rec in physical:
   data=rb(REPO/rec['path']);need(digest(data)==rec['sha256']and len(data)==rec['size_bytes'],'Final physical SHA/size')
  need(len({r['path']for r in physical})==len(physical)and sum(r['size_bytes']for r in physical)<BATCH,'Single unique bounded batch')
  qa=dict(completed=True,PASS=True,entry=bind(entry),exact_phase_frozen=True,exact4_completion_roles=True,original_source_files=len(original),original_source_bytes=original_bytes,original_counts_by_phase={root.name:len(inventory_by_root[root])for root in PHASES},source_PNGs=sum(source.suffix.lower()=='.png'for _,source in original),all_original_to_destination_SHA_size_exact=True,source_original_final_SHA_unchanged=True,original_destination_identity_exact=True,all_declared_portable_local_links_resolve=True,link_pages=linkrows,local_links=sum(r['local_links']for r in linkrows),credential_format_detections=0,oldindex_and_README_suffix_preserved=True,source_bindings_followed=0,owned_models=0,new_prediction_NPZ=0,targets_inputs_NPZ=0,old_archive_mutations=0,model_array_image_decode=0,Git_calls=0,automatic_retry=False,ignored_ancestor_cache_paths_not_statted_read_hashed_or_modified=scope['known_ignored_cache_exclusions'])
  new(out/'QA.json',enc(qa))
  plan=dict(authorized_main_push=True,repository=str(REPO),remote=scope['remote'],branch='main',initial_remote_main=e['initial_remote_main'],manifest_sha256=digest(mz),batch_max_bytes=BATCH,batches=[dict(files=physical,message='Publish V214 modality diagnosis and V215 public TRAIN coverage preview')],scope='Exact completed V214/V215 source diagnosis and public TRAIN preview text/tables/PNG, THROUGH_V215 index and README prepend only; no new fit, weak-label admission, model selection, adoption, or extra closing cycle.')
  new(out/'ROOT_V215_PUSH_PLAN.json',enc(plan))
  complete=dict(completed=True,PASS=True,staged_only_not_pushed=True,entry=bind(entry),manifest=bind(PUB/(DELTA+'DELTA_MANIFEST.json')),QA=bind(out/'QA.json'),push_plan=bind(out/'ROOT_V215_PUSH_PLAN.json'),original_source_files=len(original),original_source_bytes=original_bytes,physical_files=len(physical),physical_bytes=sum(r['size_bytes']for r in physical),source_PNGs=qa['source_PNGs'],owned_models=0,new_prediction_NPZ=0,targets_inputs_NPZ=0,excluded_cache_paths=excluded,Git_calls=0,phase_writes=0)
  new(out/'COMPLETE.json',enc(complete));print(json.dumps(complete))
 except BaseException as error:
  new(out/'FAILURE_PRESERVED_NO_RETRY.json',enc(dict(completed=False,PASS=False,error_type=type(error).__name__,error=str(error),automatic_retry=False)));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);v=a.parse_args();prepare()if v.prepare else run(v.entry)
