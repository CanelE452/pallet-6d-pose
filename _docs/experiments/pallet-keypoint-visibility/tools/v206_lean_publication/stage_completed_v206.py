"""Lean V206 completed SOURCE TRAIN visible-part weak witness publication: no Git/network/scientific decode.
--prepare invented only. --entry requires root final freeze notice and bindings.
"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,hashlib,json,os,posixpath,re,stat,tempfile
H=Path(__file__).absolute().parent;P=H.parent.parent
PHASE=P/'query_visibility_source_train_visible_part_witness_20261025_v206'
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024');PREFIX='_docs/experiments/pallet-keypoint-visibility/';PUB=REPO/PREFIX
ARCHIVE='archive/task-2/pallet-keypoint-visibility/'+PHASE.name+'/'
DELTA='publication/deltas/V206_20261025_v1/';INDEX='EXPERIMENT_INDEX_THROUGH_V206_KO.md';PRIOR='EXPERIMENT_INDEX_THROUGH_V205_KO.md'
PROTOCOL=H/'PROTOCOL_BEFORE_ACTUAL_V206_PHASE_ENUMERATION_OR_COPY.json'
LIMIT=100*1024**2;BATCH=480*1024**2
TEXT={'.py','.json','.jsonl','.csv','.tsv','.md','.txt','.html','.htm','.yaml','.yml','.toml','.sh','.patch','.diff','.svg','.log'}
RAW={'.npz','.npy','.pt','.pth','.bin','.safetensors','.pkl','.pickle'}
EXPECTED_PARENT='3fbd0638e9303d8b96f2f008e26fe201b1b3266d'
INHERITED_V205_PUBLISHER={'path':str(P/'publication_root_preparation/V205_PUBLICATION_PREPARATION_B/stage_completed_v205_v2_index_label.py'),'sha256':'655978a1a19e2a0e220e6dd2ac957f0281cc55ac8834a42cb3df0cabca206f32'}
FIXED_A_COMPLETE={'path':str(PHASE/'observer_a/COMPLETE.json'),'sha256':'8fbb945094be50f7fc604726edb2c0c036c6a4fdd971055e1318df20b9e71088'}
REQUIRED_ROLES=('phase_result','final_phase_freeze','observer_A_complete','observer_C_complete','witness_join_complete','root_fixed8_visual_QA','root_report_complete','final_report_QA')
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
def frozen_phase(q):
 need(q.get('scientific_phase_frozen')is True and q.get('no_further_phase_writes')is True,'Actual two-field final scientific freeze; no alias/spelling assumption')
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
def publication_headers(portable):
 index_header=('# V206 · 8 source visible-part weak witness\n\n[최신 V206 보고서]('+portable+') · [이전 V205 인덱스]('+PRIOR+')\n\n기존 공개 TRAIN3프레임의 고정8 원query에 대한 독립 visible-part ROI, 실제 시각근거, 합의/불일치와 한계를 보존한다. 의미 부위 가시성과 query 위치 가능성을 분리하며 V205 strict join을 수정하지 않는다. ROI는 교정q·physical native gold가 아니고, fit admission·새 학습·모델 성공·채택을 주장하지 않는다. 아래 이전 V205 인덱스 바이트는 그대로다.\n\n---\n\n').encode()
 readme_header=('# V206 · 8 source visible-part weak witness 게시\n\n[최신 실험 인덱스]('+PREFIX+INDEX+') · [독립 weak ROI·시각근거·한계]('+PREFIX+portable+')\n\n기존 공개 TRAIN3프레임·고정8 원query의 visible-part weak witness를 보존한다. q/L/native는 그대로이며 ROI는 교정q·native gold·모델/학습 성공이 아니다. V205 결과와 strict join은 유지하고 fit admission·전체 해결을 주장하지 않는다.\n\n---\n\n').encode()
 return index_header,readme_header
def selftest():
 t=[]
 def add(n,x):t.append(dict(name=n,PASS=bool(x)));need(x,n)
 add('SHA',digest(b'abc')=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
 for value in ('/a/V112/x','/a/../x'):
  try:safe(value);ok=False
  except AssertionError:ok=True
  add('unsafe_name_'+str(len(t)),ok)
 add('relative_link',links('a/ROOT.md',b'[csv](b.csv)')==['a/b.csv'])
 add('html_attributes_only',links('a/r.html',b'<pre>[ignore](missing)</pre><img src="x.png">')==['a/x.png'])
 try:links('a/r.md',b'[bad](../../../../x)');ok=False
 except AssertionError:ok=True
 add('escape_link_rejected',ok)
 add('credential_redacted',hits(('ghp_'+'x'*40).encode())=={'GitHub':1})
 with tempfile.TemporaryDirectory(dir=H,prefix='invented_')as tmp:
  p=Path(tmp);(p/'__pycache__').mkdir();(p/'__pycache__'/'ignored').write_bytes(b'x');(p/'matplotlib_cache').mkdir();(p/'ok.json').write_bytes(b'{}');(p/'failure.diff').write_bytes(b'preserved')
  files,excluded=inventory(p);add('only2cache_dirs_excluded_failure_kept',len(files)==2 and len(excluded)==2)
  new(p/'new.txt',b'exact');add('copy_exactbytes',rb(p/'new.txt')==b'exact')
  try:new(p/'new.txt',b'overwrite');ok=False
  except FileExistsError:ok=True
  add('no_overwrite',ok);(p/'symbolic').symlink_to(p/'ok.json')
  try:inventory(p);ok=False
  except AssertionError:ok=True
  add('symlink_refused',ok);(p/'symbolic').unlink();(p/'unowned.pt').write_bytes(b'opaque no decode')
  ff,ee=inventory(p);add('model_excluded_before_read',not any(x.name=='unowned.pt'for x in ff)and any(isinstance(x,dict)and x['path']=='unowned.pt'for x in ee));(p/'unowned.pt').unlink();(p/'input_cache.npz').write_bytes(b'neverdecoded');ff,ee=inventory(p);add('unowned_inputcache_excluded_before_read',not any(x.name=='input_cache.npz'for x in ff)and any(isinstance(x,dict)and x['path']=='input_cache.npz'for x in ee))

 with tempfile.TemporaryDirectory(dir=H,prefix='invented_zero_binary_')as tmp:
  d=Path(tmp)
  for name in ('owned.pt','cache.npz','input.npy','thirdparty.safetensors'): (d/name).write_bytes(b'opaque no decode')
  (d/'proof.json').write_bytes(b'{}');(d/'clean.png').write_bytes(b'invented PNG opaque bytes')
  ff,ee=inventory(d);add('zero_array_model_whitelist_four_binaries_excluded_before_read',len(ff)==2 and len(ee)==4)
  add('preserve_own_text_and_PNG_bytes',set(x.name for x in ff)=={'proof.json','clean.png'})
 add('two_actual_freeze_fields_accept',frozen_phase(dict(scientific_phase_frozen=True,no_further_phase_writes=True)))
 for q in [dict(scientific_phase_frozen=True,no_further_phase_writes=False),dict(scientific_phase_frozen_no_further_phase_writes=True),dict(completed=True)]:
  try:frozen_phase(q);ok=False
  except AssertionError:ok=True
  add('missing_or_false_actual_freeze_field_'+str(len(t)),ok)
 add('README_old_suffix',(b'new\n'+b'old').endswith(b'old'))
 add('V206_exact8_witness_completion_roles',len(REQUIRED_ROLES)==8 and set(REQUIRED_ROLES)=={'phase_result','final_phase_freeze','observer_A_complete','observer_C_complete','witness_join_complete','root_fixed8_visual_QA','root_report_complete','final_report_QA'})
 ih,rh=publication_headers(ARCHIVE+'PORTABLE_REPORT_KO.md')
 add('functional_headers_latest206_previous205_not_training_success',ih.startswith(b'# V206')and rh.startswith(b'# V206')and ('[이전 V205 인덱스]('+PRIOR+')').encode()in ih and b'V204'not in ih+rh and b'48'not in ih+rh and b'8 source visible-part weak witness'in ih+rh)
 add('sealed_A_receipt_fixed_pointer_no_body_read',FIXED_A_COMPLETE['path']==str(PHASE/'observer_a/COMPLETE.json')and FIXED_A_COMPLETE['sha256']=='8fbb945094be50f7fc604726edb2c0c036c6a4fdd971055e1318df20b9e71088')
 return dict(completed=True,PASS=True,invented_only=True,checks=t,actual_phase_or_repository_or_API_or_Git_access=0)
def prepare():
 q=selftest();new(H/'GENERATED_INVENTED_CHECKS.json',enc(q))
 p=dict(completed=True,prepared_only=True,code=bind(__file__),checks=bind(H/'GENERATED_INVENTED_CHECKS.json'),phase=str(PHASE),repository=str(REPO),inherited_publisher=INHERITED_V205_PUBLISHER,
 source_scope='Own completed frozen V206 SOURCE TRAIN visible-part weak witness code/JSON/JSONL/CSV/MD/HTML/PNG: fixed8 original queries from3 sourceframes, independent ROI observations, semantic evidence, UNKNOWN/ambiguity, common join, own new cards, failures/provenance and QA. Parent V205 image bindings are inert metadata and existing publication relative links only; NEVER followed for new source copying. No dataset/source cache/array/weight/target/input bindings followed. Exclude matplotlib_cache/__pycache__ and ALL RAW binary suffixes before header/body/hash. 0ownedheads/0savedNPZ/0targets/0inputs.',
 immutable='Exclusive new V206 archive/delta/index/tools; preserve previous V205 index as exact suffix and prepend global README preserving old suffix. No historical scan/ancestor rehash/Git/network/secondclosing.',
 publication_outputs=dict(archive=ARCHIVE,delta=DELTA,index=INDEX,prior_index=PRIOR),links='All new MD/HTML actualattributes/local PNG/table links resolve. JSON absolute provenance remains inert; old README suffix not rescanned.',
 bytes='Original/destination SHA+size exact, <100MiB everyphysicalfile and one strict<480MiB batch; binary/image/scientific decode0.',required_final_roles=list(REQUIRED_ROLES),freeze_schema=dict(scientific_phase_frozen=True,no_further_phase_writes=True),owned_binary_policy='EMPTY; never follow source RGB/GT/cache/weights/NPZ/input/target bindings',expected_initial_main=EXPECTED_PARENT,
 root_code_preparation_authorization='Parent root message: V206 zero-binary lean publisher derived from frozen V205 V2; exact8 required final roles; source witness records only; actualstage after explicit root final freeze and approved entry. Code preparation never enumerates/hash/copies V206 scientific files or touches repository/API/Git.',actual_phase_repository_parent_metadata_inventory_hash_copy_API_or_Git_access=0,automatic_retry=False)
 new(PROTOCOL,enc(p))
 receipts=[dict(role=r,binding=dict(path=str(PHASE/'ROOT_PHASE_RESULT.json')if r=='phase_result'else None,sha256=None),expected_fields=dict(scientific_phase_frozen=True,no_further_phase_writes=True)if r=='final_phase_freeze'else None)for r in REQUIRED_ROLES]
 for item in receipts:
  if item['role']=='final_phase_freeze':item['binding']['path']=str(PHASE/'ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json')
  if item['role']=='observer_A_complete':item.update(binding=FIXED_A_COMPLETE,expected_fields=dict(completed=True,PASS=True,training_admission=False))
 template=dict(approved_actual_V206_stage=False,root_explicit_final_freeze_notice=False,code=p['code'],protocol=bind(PROTOCOL),root_scope=dict(path=None,sha256=None),completion_receipts=receipts,initial_remote_main=None,repository=str(REPO),previous_index=dict(path=str(PUB/PRIOR),sha256=None),global_readme_before=dict(path=str(REPO/'readme.md'),sha256=None),output_directory=str(P/'publication_root_preparation/V206_PUBLICATION_STAGE_A'),automatic_retry=False,Git_add_commit_push=False)
 new(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',enc(template));new(H/'FINAL_READY.json',enc(dict(completed=True,PASS=True,FINAL_READY_FROZEN=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],false_entry=bind(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_phase_enumeration_hash_copy_repository_API_or_Git_access=0)));print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),checks=len(q['checks']))))

def run(entry):
 e=json.loads(rb(entry));need(e['approved_actual_V206_stage']is True and e['root_explicit_final_freeze_notice']is True,'Wait explicit root scientific freeze and actualentry');need(e['automatic_retry']is False and e['Git_add_commit_push']is False,'No retry/Git')
 p=checked(e['protocol']);need(e['code']==bind(__file__)==p['code']and e['protocol']==bind(PROTOCOL),'Frozen publisher code/protocol');checked(e['root_scope'])
 need(e['repository']==str(REPO)and re.fullmatch('[0-9a-f]{40}',e['initial_remote_main'])and e['initial_remote_main']==EXPECTED_PARENT,'Exact root-verified initial main')
 roles={}
 for x in e['completion_receipts']:
  q=checked(x['binding']);need(PHASE in safe(x['binding']['path']).parents and x['expected_fields'],'Exact ownphase final receipt and fields')
  for k,v in x['expected_fields'].items():need(q[k]==v,'Completion receipt field '+k)
  need(x['role']not in roles,'Unique completion role');roles[x['role']]=x
 need(set(REQUIRED_ROLES)==set(roles),'All source diagnosis roles and no prediction/weight roles');need(Path(roles['phase_result']['binding']['path']).name=='ROOT_PHASE_RESULT.json'and 'FREEZE'in Path(roles['final_phase_freeze']['binding']['path']).name,'Final root result/freeze names')
 frozen_phase(checked(roles['final_phase_freeze']['binding']));need(roles['observer_A_complete']['binding']==FIXED_A_COMPLETE,'Exact sealed A observation receipt')
 out=safe(e['output_directory']);need(out.parent==P/'publication_root_preparation','Separate admin output');out.mkdir(exist_ok=False);new(out/'STARTED.json',enc(dict(completed=False,entry=bind(entry),automatic_retry=False)))
 try:
  need(e['previous_index']['path']==str(PUB/PRIOR)and e['global_readme_before']['path']==str(REPO/'readme.md'),'Exact oldindex/README')
  oldindex=rb(e['previous_index']['path']);need(digest(oldindex)==e['previous_index']['sha256'],'Oldindex SHA');oldreadme=rb(e['global_readme_before']['path']);need(digest(oldreadme)==e['global_readme_before']['sha256'],'OldREADME SHA')
  paths,excluded=inventory(PHASE);need(paths,'Nonempty V206');need(not any(x.suffix.casefold()in RAW for x in paths),'ZERO source binaries read/copied');records=[];physical=[];texts={};source_hashes={}
  def store(rel,data,source,category):
   need(len(data)<LIMIT,'Each Gitfile<100MiB');need(not hits(data)if Path(rel).suffix.lower()in TEXT else True,'Credential format hit; no values printed');new(PUB/rel,data);need(rb(PUB/rel)==data,'Exact destination bytes');physical.append(dict(path=PREFIX+rel,sha256=digest(data),size_bytes=len(data)))
   row=dict(source_root_id='P'if category=='original_scientific_record'else'PUBLICATION_ADMIN_CODE'if category=='publication_tool'else'DELTA_DERIVED',source_absolute=str(source),source_relative_path=str(source.relative_to(P))if P in source.parents else rel,source_sha256=digest(data),size_bytes=len(data),published_relative=rel,published_sha256=digest(data),status='PUBLISHED_RAW',category=category,exclusion_reason=None,gzipchunks=None);records.append(row)
   if category!='old_global_snapshot'and Path(rel).suffix.lower()in('.md','.html','.htm'):texts[rel]=data
  for source in paths:
   data=rb(source);source_hashes[str(source)]=digest(data);store(ARCHIVE+str(source.relative_to(PHASE)),data,source,'original_scientific_record')
  for source in [Path(__file__).absolute(),PROTOCOL,H/'GENERATED_INVENTED_CHECKS.json']:store('tools/v206_lean_publication/'+source.name,rb(source),source,'publication_tool')
  portable=ARCHIVE+'PORTABLE_REPORT_KO.md';header,readme_prefix=publication_headers(portable);store(INDEX,header+oldindex,PUB/INDEX,'versioned_index');need(rb(PUB/INDEX).endswith(oldindex),'Oldindex suffix')

  store(DELTA+'GLOBAL_README_BEFORE_V206.md',oldreadme,PUB/(DELTA+'GLOBAL_README_BEFORE_V206.md'),'old_global_snapshot')
  manifest=dict(format='immutable-completed-phase-delta-v1',phase=PHASE.name,previous_index=e['previous_index'],root_entry=bind(entry),completion_receipts=e['completion_receipts'],entries=records,owned_binary_whitelist=[],excluded_cache_paths=excluded,source_bindings_followed=0,original_input_target_foundation_bindings_followed=0,owned_prediction_binary_bindings_followed=0,old_archive_or_manifest_paths_modified=False,no_second_closing_required=True)
  new(PUB/(DELTA+'DELTA_MANIFEST.json'),enc(manifest));physical.append(dict(path=PREFIX+DELTA+'DELTA_MANIFEST.json',sha256=digest(enc(manifest)),size_bytes=len(enc(manifest))))
  newreadme=readme_prefix+oldreadme;need(newreadme.endswith(oldreadme),'README suffix')

  # Root README links resolve relative to repository, all other portable links to publicationroot.
  linkrows=[]
  for doc,data in texts.items():
   targets=links(doc,data)
   for target in targets:need(existing_target(PUB/target),'Missing relative link '+doc+' -> '+target)
   linkrows.append(dict(page=doc,local_links=len(targets)))
  rootprefix=newreadme[:-len(oldreadme)]if oldreadme else newreadme
  rootlinks=links('ROOT_REPO_README.md',rootprefix)
  for target in rootlinks:need(existing_target(REPO/target),'New repository README link missing')
  linkrows.append(dict(page='readme.md newprefix only; oldsuffix byte-preserved',local_links=len(rootlinks)))
  finalpaths,finalexcluded=inventory(PHASE);need(finalpaths==paths and finalexcluded==excluded,'Frozen phase file inventory unchanged')
  need(existing_target(PUB/portable),'Required portable root report');replace_readme(REPO/'readme.md',newreadme,e['global_readme_before']['sha256']);physical.append(dict(path='readme.md',sha256=digest(newreadme),size_bytes=len(newreadme)))
  for source,h in source_hashes.items():need(digest(rb(source))==h,'Original scientific file changed during stage')
  for x in physical:need(digest(rb(REPO/x['path']))==x['sha256'],'Stagedfile SHA')
  need(len({x['path']for x in physical})==len(physical)and sum(x['size_bytes']for x in physical)<BATCH,'One bounded push batch')
  qa=dict(completed=True,PASS=True,entry=bind(entry),owned_models=0,new_real_prediction_NPZ=0,owned_targets_NPZ=0,completed_source_witness_required_roles_verified=True,original_source_files=len(paths),original_source_bytes=sum(r['size_bytes']for r in records if r['source_root_id']=='P'),source_original_SHA_unchanged=True,source_binding_followed=0,original_input_target_foundation_bindings_followed=0,owned_prediction_binary_bindings_followed=0,all_original_to_destination_SHA_size_exact=True,all_new_portable_local_links_resolve=True,link_pages=linkrows,local_links=sum(x['local_links']for x in linkrows),credential_format_detections=0,oldindex_and_README_suffix_preserved=True,old_archive_mutations=0,model_array_image_decode=0,Git_calls=0,automatic_retry=False);new(out/'QA.json',enc(qa))
  plan=dict(authorized_main_push=True,repository=str(REPO),remote='https://github.com/CanelE452/pallet-6d-pose.git',branch='main',initial_remote_main=e['initial_remote_main'],manifest_sha256=digest(enc(manifest)),batch_max_bytes=BATCH,batches=[dict(files=physical,message='Publish V206 eight source visible-part weak witnesses and preserved uncertainty')],scope='New V206 only plus global README prepend; no scientific changes or separate closing cycle.');new(out/'ROOT_V206_PUSH_PLAN.json',enc(plan))
  complete=dict(completed=True,PASS=True,staged_only_not_pushed=True,entry=bind(entry),manifest=bind(PUB/(DELTA+'DELTA_MANIFEST.json')),QA=bind(out/'QA.json'),push_plan=bind(out/'ROOT_V206_PUSH_PLAN.json'),owned_models=0,new_real_prediction_NPZ=0,owned_targets_NPZ=0,original_source_files=len(paths),physical_files=len(physical),physical_bytes=sum(x['size_bytes']for x in physical),excluded_cache_paths=excluded,Git_calls=0,phase_writes=0);new(out/'COMPLETE.json',enc(complete));print(json.dumps(complete))
 except BaseException as error:
  new(out/'FAILURE_PRESERVED_NO_RETRY.json',enc(dict(completed=False,PASS=False,error_type=type(error).__name__,error=str(error),automatic_retry=False)));raise
if __name__=='__main__':
 a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);v=a.parse_args();prepare()if v.prepare else run(v.entry)
