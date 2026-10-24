"""Lean V202 completed-science publication: no Git/network/scientific decode.
--prepare invented only. --entry requires root final freeze notice and bindings.
"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,hashlib,json,os,posixpath,re,stat,tempfile
H=Path(__file__).absolute().parent;P=H.parent.parent
PHASE=P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202'
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024');PREFIX='_docs/experiments/pallet-keypoint-visibility/';PUB=REPO/PREFIX
ARCHIVE='archive/task-2/pallet-keypoint-visibility/'+PHASE.name+'/'
DELTA='publication/deltas/V202_20261025_v1/';INDEX='EXPERIMENT_INDEX_THROUGH_V202_KO.md';PRIOR='EXPERIMENT_INDEX_THROUGH_V201_KO.md'
PROTOCOL=H/'PROTOCOL_BEFORE_ACTUAL_V202_PHASE_ENUMERATION_OR_COPY.json'
LIMIT=100*1024**2;BATCH=480*1024**2
TEXT={'.py','.json','.jsonl','.csv','.tsv','.md','.txt','.html','.htm','.yaml','.yml','.toml','.sh','.patch','.diff','.svg','.log'}
RAW={'.npz','.npy','.pt','.pth','.bin','.safetensors','.pkl','.pickle'}
RUN=PHASE/'runs/C60_C61_SPATIAL_Q8_NOMINAL_SOURCE_FIXED4000'
ARMS=('C60_RAW96_NATIVE_MASK_ZERO_XY_NOMINAL_CONTROL','C61_RAW96_NATIVE_MASK_Q8_XY_NOMINAL')
DOMAINS=('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816','DIRECT_CONTROL')
INHERITED_V201_PUBLISHER={'path':'/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V201_PUBLICATION_PREPARATION_B/stage_completed_v201.py','sha256':'be907d38edf74d38800332b0a26d9f19483ebaefcd27c741f1065490bc5c09be'}
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
def owned_raw_whitelist(fit):
 # Pure metadata only: never follow original input/target/cache/weight bindings.
 need(fit['completed']is True and fit['PASS']is True and fit['source_only']is True and fit['from_scratch']is True,'Completed fromscratch sourceonly fit')
 need(fit['parameters_each']==118359 and fit['steps_each']==4000 and fit['total_updates']==8000,'Fixed2x4000/118359 parameters')
 need(fit['all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs']is True,'Bothfinalfreeze before any source predictions')
 freeze=fit['all2_freeze'];safe(freeze['path']);need(RUN in Path(freeze['path']).parents and re.fullmatch('[0-9a-f]{64}',freeze['sha256']),'Own freeze metadata binding')
 need(set(fit['arms'])==set(ARMS),'Exactly both prospectivelyfixed arms')
 result={}
 def add(binding,kind):
  p=safe(binding['path']);need(RUN in p.parents and re.fullmatch('[0-9a-f]{64}',binding['sha256']),'Own RUN binary exactbinding only')
  need(p.suffix.lower()in({'.pt','.pth'}if kind=='owned_checkpoint'else{'.npz'}),'Owned binary suffix')
  need(str(p)not in result,'Owned binary unique logicalpath, no alias/doublecount')
  result[str(p)]={'path':str(p),'sha256':binding['sha256'],'kind':kind}
 for i,name in enumerate(ARMS):
  arm=fit['arms'][name];need(arm['updates']==4000 and arm['use_sam']is False and arm['spatial_q8']is bool(i),'Final4000 RAW/native/contextflags')
  add(arm['checkpoint'],'owned_checkpoint');need(set(arm['outputs'])==set(DOMAINS),'Four FULL prediction domains')
  for name in DOMAINS:add(arm['outputs'][name],'source_prediction')
  need(set(arm['source_cohorts'])==set(DOMAINS),'Four source subset domains')
  for name in DOMAINS:
   cohorts=arm['source_cohorts'][name];need(set(cohorts)=={'TRAIN','HOLDOUT'},'Originalcomponent TRAIN/HOLDOUT savedcohorts')
   for role in ('TRAIN','HOLDOUT'):add(cohorts[role]['outputs'],'source_prediction')
 need(sum(x['kind']=='owned_checkpoint'for x in result.values())==2 and sum(x['kind']=='source_prediction'for x in result.values())==24,'Exactly2owned models/24sourcepredictions')
 return result

def invented_fit():
 # Binding-shaped authored fixture; paths are strings and are not read/statted.
 fit=dict(completed=True,PASS=True,source_only=True,from_scratch=True,parameters_each=118359,steps_each=4000,total_updates=8000,all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs=True,all2_freeze=dict(path=str(RUN/'ALL2_FINAL4000_FROZEN_BEFORE_SOURCE_TRAIN_OR_HOLDOUT_OUTPUTS.json'),sha256='a'*64),arms={})
 for i,name in enumerate(ARMS):
  base=RUN/name;outputs={d:dict(path=str(base/(d+'_FULL.npz')),sha256='b'*64)for d in DOMAINS}
  cohorts={d:{r:dict(outputs=dict(path=str(base/(d+'_'+r+'.npz')),sha256='c'*64),metadata=dict(path='/unused/input.jsonl',sha256='d'*64),rows=0,original_inputU=0)for r in ('TRAIN','HOLDOUT')}for d in DOMAINS}
  fit['arms'][name]=dict(use_sam=False,spatial_q8=bool(i),updates=4000,checkpoint=dict(path=str(base/'FINAL4000.pt'),sha256='e'*64),outputs=outputs,source_cohorts=cohorts)
 fit['original_input_cache']={'path':'/MUST_NOT_BE_FOLLOWED/input.npz','sha256':'f'*64}
 return fit

def inventory(root,owned_raw=None):
 root=safe(root);owned_raw={}if owned_raw is None else owned_raw;out=[];excluded=[];stack=[root]
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
     if suffix in RAW and str(source)not in owned_raw:
      need(suffix in {'.npz','.npy'},'Unowned model/foundation binary refused before body access')
      excluded.append(dict(path=str(source.relative_to(root)),reason='unowned source array or cache excluded before body/header/hash'));continue
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
  new(p/'new',b'exact');add('copy_exactbytes',rb(p/'new')==b'exact')
  try:new(p/'new',b'overwrite');ok=False
  except FileExistsError:ok=True
  add('no_overwrite',ok);(p/'symbolic').symlink_to(p/'ok.json')
  try:inventory(p);ok=False
  except AssertionError:ok=True
  add('symlink_refused',ok);(p/'symbolic').unlink();(p/'unowned.pt').write_bytes(b'opaque no decode')
  try:inventory(p);ok=False
  except AssertionError:ok=True
  add('unowned_model_rejected_before_read',ok);(p/'unowned.pt').unlink();(p/'input_cache.npz').write_bytes(b'neverdecoded');ff,ee=inventory(p);add('unowned_inputcache_excluded_before_read',not any(x.name=='input_cache.npz'for x in ff)and any(isinstance(x,dict)and x['path']=='input_cache.npz'for x in ee))
 with tempfile.TemporaryDirectory(dir=H,prefix='invented_owned_')as tmp:
  d=Path(tmp);model=d/'owned.pt';pred=d/'owned.npz';model.write_bytes(b'invented model opaque bytes');pred.write_bytes(b'invented prediction opaque bytes');allow={str(model):dict(kind='owned_checkpoint'),str(pred):dict(kind='source_prediction')};ff,ee=inventory(d,allow);add('only_exactowned_raw_allowed_no_decode',set(ff)=={model,pred}and not ee);new(d/'copy',rb(pred));add('owned_raw_bytes_roundtrip',rb(d/'copy')==rb(pred))
 fit=invented_fit();allow=owned_raw_whitelist(fit);add('exact2owned24sourceprediction_metadata_whitelist',len(allow)==26)
 for label,mutate in [('not_fromscratch',lambda x:x.update(from_scratch=False)),('not_sourceonly',lambda x:x.update(source_only=False)),('wrongfinalsteps',lambda x:x.update(steps_each=8000)),('barrier_false',lambda x:x.update(all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs=False)),('missing_prediction',lambda x:x['arms'][ARMS[0]]['outputs'].pop(DOMAINS[0])),('oldinput_as_prediction',lambda x:x['arms'][ARMS[0]]['outputs'].__setitem__(DOMAINS[0],dict(path='/outside/cache.npz',sha256='f'*64)))]:
  x=invented_fit();mutate(x)
  try:owned_raw_whitelist(x);ok=False
  except AssertionError:ok=True
  add('owned_closure_'+label,ok)
 add('README_old_suffix',(b'new\n'+b'old').endswith(b'old'))
 return dict(completed=True,PASS=True,invented_only=True,checks=t,actual_phase_or_repository_or_API_or_Git_access=0)
def prepare():
 q=selftest();new(H/'GENERATED_INVENTED_CHECKS.json',enc(q));need(q['PASS'],'Invented checks')
 p=dict(completed=True,prepared_only=True,code=bind(__file__),checks=bind(H/'GENERATED_INVENTED_CHECKS.json'),phase=str(PHASE),repository=str(REPO),source_scope='Only root-frozen V202; exactly2owned checkpoints+24sourceprediction bindings from completed ownfit. Archive original scientific files/failures/provenance. Exclude matplotlib_cache/__pycache__ and unowned npz/npy inputcache beforebody. Unowned model/foundation binaries fail. Original input/target/foundation binding values never followed. V112/symlinks/nonregular refused beforebody.',immutable='New exclusive V202 archive/delta/index/tools only; original phase and prior publication paths never overwritten; read oldV201 index once and preserve as newindex suffix, globalREADME only prepend exact oldbytes.',publication_outputs=dict(archive=ARCHIVE,delta=DELTA,index=INDEX,prior_index=PRIOR),links='All new source MD/HTML, newindex and newREADME prefix relative links resolve to ordinary files/directories. Exact oldREADME backup is provenance with original repository link base, not a portable page; oldsuffix bytes preserved, no historical link rescan. Actual HTML attrs only; JSON source references never followed.',bytes='Original SHA/size equals destination SHA/size; allphysical<100MiB; one strict480MiB batch. No compression needed; fail if bounds exceeded.',entry='Separate explicit root freeze notice required plus exact phase_result/finalfreeze/sourcefit/independentfitQA/finalscienceQA expectations, actualverified40hexparent with prefix3f6cca4; root scope binding; no Git',no_model_array_RGB_or_scientific_decode=True,actual_preparation_access=0,automatic_retry=False)
 p.update(inherited_publisher=INHERITED_V201_PUBLISHER,owned_binary_policy='Only completed fit COMPLETE arms.checkpoint/outputs/source_cohorts[domain][TRAIN|HOLDOUT].outputs: exactly2own118359 fromscratch final4000 checkpoints+24sourcepredictionNPZ. Original input/cache/target/foundation bindings never followed. Unowned source npz/npy cache excluded before body; other unowned models fail. No NPZheader/model/image decode.',required_final_roles=['phase_result','final_phase_freeze','source_fit_complete','independent_source_fit_review','final_scientific_QA'])
 new(PROTOCOL,enc(p));template=dict(approved_actual_V202_stage=False,root_explicit_final_freeze_notice=False,code=p['code'],protocol=bind(PROTOCOL),root_scope=dict(path=None,sha256=None),source_fit_complete=dict(path=str(RUN/'COMPLETE.json'),sha256=None),completion_receipts=[dict(role='phase_result',binding=dict(path=str(PHASE/'ROOT_PHASE_RESULT.json'),sha256=None),expected_fields=None),dict(role='final_phase_freeze',binding=dict(path=None,sha256=None),expected_fields=None),dict(role='source_fit_complete',binding=dict(path=str(RUN/'COMPLETE.json'),sha256=None),expected_fields=dict(completed=True,PASS=True,source_only=True,from_scratch=True,steps_each=4000,parameters_each=118359,total_updates=8000,all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs=True)),dict(role='independent_source_fit_review',binding=dict(path=None,sha256=None),expected_fields=None),dict(role='final_scientific_QA',binding=dict(path=None,sha256=None),expected_fields=None)],initial_remote_main=None,repository=str(REPO),previous_index=dict(path=str(PUB/PRIOR),sha256=None),global_readme_before=dict(path=str(REPO/'readme.md'),sha256=None),output_directory=str(P/'publication_root_preparation/V202_PUBLICATION_STAGE_A'),automatic_retry=False,Git_add_commit_push=False)
 new(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',enc(template));new(H/'FINAL_READY.json',enc(dict(completed=True,PASS=True,FINAL_READY_FROZEN=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],false_entry=bind(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_phase_enumeration_hash_copy_repository_API_or_Git_access=0)));print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),checks=len(q['checks']))))
def run(entry):
 e=json.loads(rb(entry));need(e['approved_actual_V202_stage']is True and e['root_explicit_final_freeze_notice']is True,'Wait explicit root scientific freeze and actualentry');need(e['automatic_retry']is False and e['Git_add_commit_push']is False,'No retry/Git')
 p=checked(e['protocol']);need(e['code']==bind(__file__)==p['code']and e['protocol']==bind(PROTOCOL),'Frozen publisher code/protocol');checked(e['root_scope'])
 need(e['repository']==str(REPO)and re.fullmatch('[0-9a-f]{40}',e['initial_remote_main'])and e['initial_remote_main'].startswith('3f6cca4'),'Exact root-verified initial main')
 roles={}
 for x in e['completion_receipts']:
  q=checked(x['binding']);need(PHASE in safe(x['binding']['path']).parents and x['expected_fields'],'Exact ownphase final receipt and fields')
  for k,v in x['expected_fields'].items():need(q[k]==v,'Completion receipt field '+k)
  need(x['role']not in roles,'Unique completion role');roles[x['role']]=x
 need({'phase_result','final_phase_freeze','source_fit_complete','independent_source_fit_review','final_scientific_QA'}<=roles.keys(),'Final result and freeze required');need(Path(roles['phase_result']['binding']['path']).name=='ROOT_PHASE_RESULT.json'and 'FREEZE'in Path(roles['final_phase_freeze']['binding']['path']).name,'Final root result/freeze names')
 need(e['source_fit_complete']==roles['source_fit_complete']['binding'] and e['source_fit_complete']['path']==str(RUN/'COMPLETE.json'),'Exact completed ownfit binding')
 fit=checked(e['source_fit_complete']);allowed_raw=owned_raw_whitelist(fit);checked(fit['all2_freeze'])
 out=safe(e['output_directory']);need(out.parent==P/'publication_root_preparation','Separate admin output');out.mkdir(exist_ok=False);new(out/'STARTED.json',enc(dict(completed=False,entry=bind(entry),automatic_retry=False)))
 try:
  need(e['previous_index']['path']==str(PUB/PRIOR)and e['global_readme_before']['path']==str(REPO/'readme.md'),'Exact oldindex/README')
  oldindex=rb(e['previous_index']['path']);need(digest(oldindex)==e['previous_index']['sha256'],'Oldindex SHA');oldreadme=rb(e['global_readme_before']['path']);need(digest(oldreadme)==e['global_readme_before']['sha256'],'OldREADME SHA')
  paths,excluded=inventory(PHASE,allowed_raw);need(paths,'Nonempty V202');need(set(allowed_raw)=={str(x)for x in paths if x.suffix.lower()in RAW},'Allandonly26 owned binary files present');records=[];physical=[];texts={};source_hashes={}
  def store(rel,data,source,category):
   need(len(data)<LIMIT,'Each Gitfile<100MiB');need(not hits(data)if Path(rel).suffix.lower()in TEXT else True,'Credential format hit; no values printed');new(PUB/rel,data);need(rb(PUB/rel)==data,'Exact destination bytes');physical.append(dict(path=PREFIX+rel,sha256=digest(data),size_bytes=len(data)))
   row=dict(source_root_id='P'if category=='original_scientific_record'else'PUBLICATION_ADMIN_CODE'if category=='publication_tool'else'DELTA_DERIVED',source_absolute=str(source),source_relative_path=str(source.relative_to(P))if P in source.parents else rel,source_sha256=digest(data),size_bytes=len(data),published_relative=rel,published_sha256=digest(data),status='PUBLISHED_RAW',category=category,exclusion_reason=None,gzipchunks=None);records.append(row)
   if category!='old_global_snapshot'and Path(rel).suffix.lower()in('.md','.html','.htm'):texts[rel]=data
  for source in paths:
   data=rb(source);need(str(source)not in allowed_raw or digest(data)==allowed_raw[str(source)]['sha256'],'Owned model/prediction exactfit SHA');source_hashes[str(source)]=digest(data);store(ARCHIVE+str(source.relative_to(PHASE)),data,source,'original_scientific_record')
  for source in [Path(__file__).absolute(),PROTOCOL,H/'GENERATED_INVENTED_CHECKS.json']:store('tools/v202_lean_publication/'+source.name,rb(source),source,'publication_tool')
  portable=ARCHIVE+'PORTABLE_REPORT_KO.md';header=('# V202 spatial q8 NOMINAL source 대조 학습\n\n[최신 V202 보고서]('+portable+') · [이전 V201 인덱스]('+PRIOR+')\n\n원 q/L/native8 및 weak/UNKNOWN을 유지한 두 RAW head 각4000 source 학습 기록이다. 결과로 실사 전이·물리 native gold·모델 채택·전체 목표 해결을 주장하지 않는다. 아래 이전 인덱스 바이트는 그대로다.\n\n---\n\n').encode();store(INDEX,header+oldindex,PUB/INDEX,'versioned_index');need(rb(PUB/INDEX).endswith(oldindex),'Oldindex suffix')
  store(DELTA+'GLOBAL_README_BEFORE_V202.md',oldreadme,PUB/(DELTA+'GLOBAL_README_BEFORE_V202.md'),'old_global_snapshot')
  manifest=dict(format='immutable-completed-phase-delta-v1',phase=PHASE.name,previous_index=e['previous_index'],root_entry=bind(entry),completion_receipts=e['completion_receipts'],entries=records,owned_binary_whitelist=list(allowed_raw.values()),source_fit_complete=e['source_fit_complete'],excluded_cache_paths=excluded,source_bindings_followed=0,original_input_target_foundation_bindings_followed=0,owned_fit_binary_bindings_followed=26,old_archive_or_manifest_paths_modified=False,no_second_closing_required=True)
  new(PUB/(DELTA+'DELTA_MANIFEST.json'),enc(manifest));physical.append(dict(path=PREFIX+DELTA+'DELTA_MANIFEST.json',sha256=digest(enc(manifest)),size_bytes=len(enc(manifest))))
  newreadme=('# V202 spatial q8 source 대조 학습 게시\n\n[최신 실험 인덱스]('+PREFIX+INDEX+') · [TRAIN 대응·한계]('+PREFIX+portable+')\n\n원 초기8점 공간 조건과 matched control 두 source head 각4000 학습의 기록이다. 실제 성능·한계는 보고서에 보존하며 실사 개선·채택·전체 해결을 자동 주장하지 않는다.\n\n---\n\n').encode()+oldreadme;need(newreadme.endswith(oldreadme),'README suffix')
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
  finalpaths,finalexcluded=inventory(PHASE,allowed_raw);need(finalpaths==paths and finalexcluded==excluded,'Frozen phase file inventory unchanged')
  need(existing_target(PUB/portable),'Required portable root report');replace_readme(REPO/'readme.md',newreadme,e['global_readme_before']['sha256']);physical.append(dict(path='readme.md',sha256=digest(newreadme),size_bytes=len(newreadme)))
  for source,h in source_hashes.items():need(digest(rb(source))==h,'Original scientific file changed during stage')
  for x in physical:need(digest(rb(REPO/x['path']))==x['sha256'],'Stagedfile SHA')
  need(len({x['path']for x in physical})==len(physical)and sum(x['size_bytes']for x in physical)<BATCH,'One bounded push batch')
  qa=dict(completed=True,PASS=True,entry=bind(entry),owned_models=2,source_prediction_NPZ=24,owned_targets_NPZ=0,completed_fromscratch_RAW_2x4000_118359_bothfreeze_verified=True,original_source_files=len(paths),original_source_bytes=sum(r['size_bytes']for r in records if r['source_root_id']=='P'),source_original_SHA_unchanged=True,source_binding_followed=0,original_input_target_foundation_bindings_followed=0,owned_fit_binary_bindings_followed=26,all_original_to_destination_SHA_size_exact=True,all_new_portable_local_links_resolve=True,link_pages=linkrows,local_links=sum(x['local_links']for x in linkrows),credential_format_detections=0,oldindex_and_README_suffix_preserved=True,old_archive_mutations=0,model_array_image_decode=0,Git_calls=0,automatic_retry=False);new(out/'QA.json',enc(qa))
  plan=dict(authorized_main_push=True,repository=str(REPO),remote='https://github.com/CanelE452/pallet-6d-pose.git',branch='main',initial_remote_main=e['initial_remote_main'],manifest_sha256=digest(enc(manifest)),batch_max_bytes=BATCH,batches=[dict(files=physical,message='Publish V202 spatial original-query context controlled source training and preserved limits')],scope='New V202 only plus global README prepend; no scientific changes or separate closing cycle.');new(out/'ROOT_V202_PUSH_PLAN.json',enc(plan))
  complete=dict(completed=True,PASS=True,staged_only_not_pushed=True,entry=bind(entry),manifest=bind(PUB/(DELTA+'DELTA_MANIFEST.json')),QA=bind(out/'QA.json'),push_plan=bind(out/'ROOT_V202_PUSH_PLAN.json'),owned_models=2,source_prediction_NPZ=24,owned_targets_NPZ=0,original_source_files=len(paths),physical_files=len(physical),physical_bytes=sum(x['size_bytes']for x in physical),excluded_cache_paths=excluded,Git_calls=0,phase_writes=0);new(out/'COMPLETE.json',enc(complete));print(json.dumps(complete))
 except BaseException as error:
  new(out/'FAILURE_PRESERVED_NO_RETRY.json',enc(dict(completed=False,PASS=False,error_type=type(error).__name__,error=str(error),automatic_retry=False)));raise
if __name__=='__main__':
 a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);v=a.parse_args();prepare()if v.prepare else run(v.entry)
