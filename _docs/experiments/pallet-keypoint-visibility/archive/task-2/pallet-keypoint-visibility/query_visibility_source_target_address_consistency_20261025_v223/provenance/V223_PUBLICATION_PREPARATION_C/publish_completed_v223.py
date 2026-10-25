"""V223 완결 SOURCE 실험의 원바이트 게시. Diagnostic+SOURCE paired fit; CODEONLY 준비; 실제 단계는 별도 root entry 필요."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote,quote
from types import SimpleNamespace
import argparse,concurrent.futures,datetime,hashlib,json,os,posixpath,re,stat,subprocess,tempfile,traceback,urllib.request
H=Path(__file__).absolute().parent;R=H.parent;P=R.parent
PHASE=P/'query_visibility_source_target_address_consistency_20261025_v223'
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
PREFIX='_docs/experiments/pallet-keypoint-visibility/';PUB=REPO/PREFIX
ARCHIVE='archive/task-2/pallet-keypoint-visibility/'+PHASE.name+'/'
DELTA='publication/deltas/V223_20261025_v1/'
INDEX='EXPERIMENT_INDEX_THROUGH_V223_KO.md';PRIOR='EXPERIMENT_INDEX_THROUGH_V222_KO.md'
EXPECTED_PARENT='f467431f01e8b3b60b1a45fa8814af0dd7dff142'
REMOTE='https://github.com/CanelE452/pallet-6d-pose.git'
STAGE=R/'V223_PUBLICATION_STAGE_C'
TEXT={'.py','.json','.jsonl','.csv','.tsv','.md','.txt','.html','.htm','.patch','.diff','.log','.svg'}
RAW={'.npz','.npy','.pt','.pth','.bin','.safetensors','.pkl','.pickle'}
PT_RELS=('paired_source_fit_v1/C68SHARED_POINT_PART_LOGIT/FINAL_CHECKPOINT.pt','paired_source_fit_v1/C69SEPARATE_POINT_PART_LOGITS/FINAL_CHECKPOINT.pt')
MUTABLE={'readme.md',PREFIX+'README.md',PREFIX+'EXPERIMENT_INDEX_KO.md'}
IGNORED={PREFIX+'tools/__pycache__/check_publication_qa.cpython-313.pyc',PREFIX+'tools/__pycache__/verify_published_results.cpython-313.pyc'}
LIMIT=100*1024**2;BATCH=480*1024**2
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

def enc(z):return(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()

def new(p,z):
 p,fd=parentfd(p,True)
 try:n=os.open(p.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'wb')as f:f.write(z)

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

def git(args, payload=None):
 r = subprocess.run(['git', *args], cwd=REPO, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
 need(r.returncode == 0, 'Git failed: '+str(args[:4])+' '+r.stderr.decode(errors='replace')[-3000:])
 return r.stdout

def names(args): return set(git(args).decode().split('\0'))-{''}

def tree_map(raw, index):
 result = {}
 for rec in raw.split(b'\0'):
  if not rec: continue
  meta, path = rec.decode().split('\t', 1)
  mode, middle, last = meta.split()
  need(mode == '100644' and (last == '0' if index else middle == 'blob'), 'Regular Git blob only')
  need(path not in result, 'Unique Git record')
  result[path] = middle if index else last
 return result

def verify_blobs(mapping, planned):
 # Read only the explicit owned index/tree blob IDs; no historic path traversal.
 oid_expect = {}
 for path, oid in mapping.items():
  expected = (planned[path]['sha256'], planned[path]['size_bytes'])
  need(oid not in oid_expect or oid_expect[oid] == expected, 'Shared blob content agrees')
  oid_expect[oid] = expected
 raw = git(['cat-file','--batch'], ''.join(x+'\n' for x in oid_expect).encode())
 offset = 0
 for oid, (sha, size) in oid_expect.items():
  end = raw.index(b'\n', offset)
  header = raw[offset:end].decode().split()
  need(header == [oid, 'blob', str(size)], 'Explicit blob header')
  offset = end+1; body = raw[offset:offset+size]; offset += size
  need(m.digest(body) == sha and raw[offset:offset+1] == b'\n', 'Explicit Git blob SHA256')
  offset += 1
 need(offset == len(raw), 'No extra blob payload')
 return len(mapping)


# 바이트·보안·Git blob kernel은 위 V220 함수와 AST 동일. 실제 model/image 해석은 없다.
m=SimpleNamespace(digest=digest)
def bind(path):return dict(path=str(safe(path)),sha256=digest(rb(path)))
def checked_bytes(b):
 need(isinstance(b,dict)and set(b)in({'path','sha256'},{'path','sha256','size_bytes'}),'Binding path/SHA/optional size only')
 need(type(b['sha256'])is str and re.fullmatch('[0-9a-f]{64}',b['sha256']),'SHA256 format')
 data=rb(safe(b['path']));need(digest(data)==b['sha256'],'Binding SHA drift')
 if 'size_bytes'in b:need(type(b['size_bytes'])is int and len(data)==b['size_bytes'],'Binding byte size')
 return data
def checked(b):return json.loads(checked_bytes(b))
def relpath(s):
 need(type(s)is str and s and not s.startswith('/')and posixpath.normpath(s)==s and '..'not in Path(s).parts and '\0'not in s and 'v112'not in s.casefold(),'Relative lexical guard before filesystem')
 return s
def admin(b):
 need(R in safe(b['path']).parents and Path(b['path']).suffix=='.json','Root admin JSON only');return checked(b)
def mkdir_new(path):
 p,fd=parentfd(path,True)
 try:os.mkdir(p.name,dir_fd=fd)
 finally:os.close(fd)
def entry_gate(path,mode):
 need(R in safe(path).parents,'Admin entry only');e=json.loads(rb(path))
 need(e.get('approved_actual_operation')==mode and e.get('automatic_retry')is False,'Separate exact root actual operation; no retry')
 need(e.get('publisher')==bind(__file__)and e.get('runbook')==bind(H/'RUNBOOK_KO.md'),'Frozen publisher/runbook')
 s=admin(e['root_actual_scope']);need(s.get('approved_actual_'+mode)is True,'Root scope exact operation')
 need(e.get('initial_remote_main')==EXPECTED_PARENT and e.get('repository')==str(REPO),'Exact parent/clone')
 return e
def expect_receipt(x,path):
 need(x['binding']['path']==str(path)and isinstance(x['expected_fields'],dict)and x['expected_fields'],'Exact receipt and actual field expectations')
 z=checked(x['binding'])
 for k,v in x['expected_fields'].items():need(k in z and z[k]==v and (type(z[k])is bool if type(v)is bool else True),'Actual receipt field '+k)
 need(z.get('completed')is True,'Completed root receipt');return z
def closure(e):
 need(e['source_root']==str(PHASE),'One exact owned phase; never follow source pointers')
 phase=expect_receipt(e['phase_result'],PHASE/'ROOT_PHASE_RESULT.json')
 freeze=expect_receipt(e['final_freeze'],PHASE/'ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json')
 need(freeze.get('phase_result')==e['phase_result']['binding'],'Final freeze binds exact result')
 # Receipt schemas are explicit actual entry fields, rather than a guessed generic frozen key.
 ef=e['final_freeze']['expected_fields']
 need((ef.get('frozen')is True and ef.get('all_scientific_phase_files_final_no_more_writes')is True)or(ef.get('scientific_phase_frozen')is True and ef.get('no_further_phase_writes')is True)or ef.get('scientific_phase_frozen_no_further_phase_writes')is True,'Positive actual freeze schema')
 fit=expect_receipt(e['paired_fit_complete'],PHASE/'paired_source_fit_v1/COMPLETE.json')
 barrier=expect_receipt(e['all2_freeze'],PHASE/'paired_source_fit_v1/ALL2_FINAL_FROZEN_BEFORE_ANY_DEV_OR_REAL_OUTPUT.json')
 need(len(e['all2_freeze']['expected_fields'])>=2,'Explicit completed + actual final barrier field')
 owned=e['owned_binary_whitelist'];need(isinstance(owned,list)and len(owned)==2,'Exactly two new owned checkpoints')
 om={}
 def bindings(z):
  if isinstance(z,dict):
   if {'path','sha256'}<=set(z):yield(z['path'],z['sha256'])
   for v in z.values():yield from bindings(v)
  elif isinstance(z,list):
   for v in z:yield from bindings(v)
 graph=set(bindings(fit))|set(bindings(barrier))
 for b in owned:
  need(set(b)=={'path','sha256','size_bytes'}and b['path']in{str(PHASE/r)for r in PT_RELS},'Only literal new C68/C69 PT paths')
  need(re.fullmatch('[0-9a-f]{64}',b['sha256'])and type(b['size_bytes'])is int and 0<b['size_bytes']<LIMIT,'Owned PT SHA/size')
  need((b['path'],b['sha256'])in graph,'PT opaque binding present in completed fit/barrier metadata')
  rel=str(safe(b['path']).relative_to(PHASE));need(rel not in om,'Unique owned PT');om[rel]=b
 need(set(om)==set(PT_RELS),'Both final new checkpoints only')
 excluded=e.get('excluded_RAW_paths',[]);need(isinstance(excluded,list)and len(excluded)==len(set(excluded))<=64,'Closed RAW exclusions')
 for rel in excluded:relpath(rel);need(Path(rel).suffix.casefold()in RAW and rel not in om,'Only excluded RAW leaves; no owned final exclusion')
 return om,set(excluded)
def inventory(owned,excluded):
 records=[];skips=[];seen=set();seenexcluded=set();stack=[PHASE]
 while stack:
  d=stack.pop();_,fd=parentfd(d/'__directory_guard_only__')
  try:
   with os.scandir(fd)as it:
    for q in sorted(it,key=lambda x:x.name):
     source=d/q.name;rel=relpath(str(source.relative_to(PHASE)))
     if q.name in {'__pycache__','matplotlib_cache'}:skips.append(dict(path=rel,reason='Cache ignored before stat/body'));continue
     suffix=source.suffix.casefold()
     if suffix in RAW and rel not in owned:
      need(rel in excluded,'Unexpected RAW: STOP before stat/hash/header/decode '+rel)
      skips.append(dict(path=rel,reason='Exact root RAW exclusion; no stat/hash/header/body'));seenexcluded.add(rel);continue
     mode=q.stat(follow_symlinks=False).st_mode;need(not stat.S_ISLNK(mode),'Source symlink rejected')
     if stat.S_ISDIR(mode):stack.append(source);continue
     need(stat.S_ISREG(mode)and suffix in TEXT|{'.png','.pt'},'Positive code/table/PNG/new2PT only')
     size=q.stat(follow_symlinks=False).st_size;need(size<LIMIT,'Each source file strictly below100MiB')
     if suffix=='.pt':need(rel in owned and size==owned[rel]['size_bytes'],'Owned checkpoint size before opaque hash');seen.add(rel)
     records.append((source,size))
  finally:os.close(fd)
 need(seen==set(PT_RELS)and seenexcluded==excluded,'Exact owned2 and excluded RAW census')
 return sorted(records),sorted(skips,key=lambda x:x['path'])
def physical_path(name,owned):
 relpath(name);need(name not in IGNORED and(name in MUTABLE or name.startswith(PREFIX)),'Exact publication prefix/cache guard')
 suffix=Path(name).suffix.casefold()
 if suffix in RAW:need(name in{PREFIX+ARCHIVE+r for r in owned},'Only whitelisted new2PT physical RAW')
 else:need(suffix in TEXT|{'.png'},'Positive physical suffix')
 return safe(REPO/name)
def git_parent(plan=None,clean=False):
 need(git(['remote','get-url','origin']).decode().strip()==REMOTE,'Public remote URL')
 parent=EXPECTED_PARENT if plan is None else plan['initial_remote_main']
 need(git(['rev-parse','HEAD']).decode().strip()==parent and git(['ls-remote','origin','refs/heads/main']).decode().split()[0]==parent,'HEAD/remote expected parent')
 need(not git(['diff','--cached','--name-only','-z']),'Initially empty index')
 if clean:
  need(not git(['status','--porcelain','--untracked-files=all']),'Initial normal clean tree')
  need(names(['ls-files','--others','--ignored','--exclude-standard','-z'])<=IGNORED,'No unrelated ignored additions')
def replace_exact(name,data,before_sha):
 need(name in MUTABLE,'Only root/docs README/canonical index may change')
 path=REPO/name;need(digest(rb(path))==before_sha,'Mutable prefix original unchanged')
 tmp=path.with_name(path.name+'.v223_publication_tmp');new(tmp,data);p,fd=parentfd(tmp)
 try:os.replace(p.name,path.name,src_dir_fd=fd,dst_dir_fd=fd)
 finally:os.close(fd)
def copy_authority(e):
 need(e.get('approved_actual_enumeration_or_copy')is True,'Explicit actual inventory/copy true')
 return True
def stage(e,entry):
 copy_authority(e);owned,excluded=closure(e)
 git_parent(clean=True);mkdir_new(STAGE);new(STAGE/'STARTED.json',enc(dict(entry=bind(entry),automatic_retry=False)))
 original,skips=inventory(owned,excluded);cap=e.get('source_caps',dict(files=1200,bytes=180*1024**2))
 need(0<len(original)<=cap['files']<=1200 and sum(s for _,s in original)<=cap['bytes']<=180*1024**2,'Bounded actual source inventory')
 pages=e['portable_pages'];need(isinstance(pages,list)and len(pages)==len(set(pages))and 2<=len(pages)<=40,'Root portable MD/HTML list')
 need(INDEX in pages and ARCHIVE+'PORTABLE_REPORT_KO.md'in pages,'Index and portable report')
 for x in pages:relpath(x);need((x==INDEX or x.startswith(ARCHIVE))and Path(x).suffix.casefold()in{'.md','.html','.htm'},'Own MD/HTML only, no JSON pages')
 prefixes=e['publication_prefixes'];need(set(prefixes)=={'index_utf8','root_readme_utf8','docs_readme_utf8'},'Root exact final prose, no metric inference')
 for x in prefixes.values():need(isinstance(x,str)and x.strip()and len(x.encode())<65536 and not hits(x.encode()),'Bounded root final prose')
 before=e['history_before'];need(set(before)=={PRIOR,'readme.md',PREFIX+'README.md',PREFIX+'EXPERIMENT_INDEX_KO.md'},'Exactly four historical bindings')
 history={}
 for name,b in before.items():
  path=PUB/PRIOR if name==PRIOR else REPO/name;need(b['path']==str(path),'Exact existing history path');history[name]=checked_bytes(b)
 rows=[];physical=[];texts={};sourcehashes={}
 def store(rel,data,source,category):
  relpath(rel);need(len(data)<LIMIT and not hits(data)if Path(rel).suffix.casefold()in TEXT else len(data)<LIMIT,'File cap/credential format')
  dest=PUB/rel;new(dest,data);need(rb(dest)==data,'Destination exact bytes')
  physical.append(dict(path=PREFIX+rel,sha256=digest(data),size_bytes=len(data)))
  rows.append(dict(category=category,source_absolute=str(source),published_relative=rel,source_sha256=digest(data),published_sha256=digest(data),size_bytes=len(data),status='PUBLISHED_RAW'))
  if rel in pages:texts[rel]=data
 for source,size in original:
  data=rb(source);need(len(data)==size,'Inventory size stable')
  rel=str(source.relative_to(PHASE))
  if rel in owned:need(digest(data)==owned[rel]['sha256'],'New owned PT exact opaque SHA')
  sourcehashes[str(source)]=digest(data);store(ARCHIVE+rel,data,source,'original_scientific_record')
 for source in (Path(__file__).absolute(),H/'RUNBOOK_KO.md',H/'GENERATED_INVENTED_CHECKS.json',H/'FINAL_READY.json',H/'ROOT_INPUT_SCHEMA_BEFORE_ACTUAL_FREEZE.json'):
  store('tools/v223_publication/'+source.name,rb(source),source,'publication_tool')
 store(DELTA+'ROOT_ACTUAL_STAGE_ENTRY.json',rb(entry),Path(entry),'root_actual_stage_authority')
 store(DELTA+'ROOT_ACTUAL_STAGE_SCOPE.json',checked_bytes(e['root_actual_scope']),Path(e['root_actual_scope']['path']),'root_actual_stage_authority')
 for name,data in history.items():store(DELTA+'HISTORY_BEFORE/'+('PRIOR_THROUGH_V223.md'if name==PRIOR else name.replace('/','__')),data,PUB/DELTA,'historical_snapshot')
 ip=prefixes['index_utf8'].encode();store(INDEX,ip+history[PRIOR],PUB/INDEX,'versioned_index')
 # Only new-prefix links are followed; historical suffix stays opaque, no old-tree crawl.
 texts[INDEX]=ip
 updates={'readme.md':prefixes['root_readme_utf8'].encode(),PREFIX+'README.md':prefixes['docs_readme_utf8'].encode(),PREFIX+'EXPERIMENT_INDEX_KO.md':ip}
 for name,prefix in updates.items():
  old=history[name];data=prefix+old;need(data.endswith(old),'Exact history suffix')
  replace_exact(name,data,before[name]['sha256']);physical.append(dict(path=name,sha256=digest(data),size_bytes=len(data)))
 need(set(texts)==set(pages),'Every declared portable page present')
 linkrows=[]
 for doc,data in texts.items():
  targets=links(doc,data)
  for target in targets:need(existing_target(PUB/target),'Missing relative link '+doc+' -> '+target)
  linkrows.append(dict(page=doc,local_links=len(targets)))
 for name,prefix in updates.items():
  targets=links(name,prefix)
  for target in targets:need(existing_target(REPO/target),'New global/doc prefix target missing')
  linkrows.append(dict(page=name+' new prefix only',local_links=len(targets)))
 manifest=dict(format='completed-source-owned-byte-delta-v1',phase='V223',source_roots=[str(PHASE)],entry=bind(entry),phase_result=e['phase_result'],final_freeze=e['final_freeze'],paired_fit_complete=e['paired_fit_complete'],all2_freeze=e['all2_freeze'],owned_binary_whitelist=e['owned_binary_whitelist'],excluded_RAW_paths=sorted(excluded),excluded_cache_paths=skips,history_before=before,entries=rows,source_bindings_followed=0,opaque_model_copies=2,no_model_array_image_decode=True)
 mz=enc(manifest);new(PUB/(DELTA+'DELTA_MANIFEST.json'),mz);physical.append(dict(path=PREFIX+DELTA+'DELTA_MANIFEST.json',sha256=digest(mz),size_bytes=len(mz)))
 final,finalskips=inventory(owned,excluded);need(final==original and finalskips==skips,'Final exact phase inventory unchanged')
 for path,sha in sourcehashes.items():need(digest(rb(path))==sha,'Final original SHA unchanged')
 for row in physical:
  data=rb(physical_path(row['path'],owned));need(digest(data)==row['sha256']and len(data)==row['size_bytes'],'Every final destination SHA/size')
 need(len({x['path']for x in physical})==len(physical)and sum(x['size_bytes']for x in physical)<BATCH,'Single unique batch strictly below480MiB')
 qa=dict(completed=True,PASS=True,entry=bind(entry),source_original_SHA_unchanged=True,all_original_destination_bytes_exact=True,owned_models=2,array_input_model_decode=0,RAW_arrays_published=0,source_bindings_followed=0,source_files=len(original),source_bytes=sum(s for _,s in original),source_PNGs=sum(p.suffix.casefold()=='.png'for p,_ in original),link_pages=linkrows,all_declared_links_resolve=True,credential_format_detections=0,history_suffix_preserved=True,ignored_cache_paths_not_opened=sorted(IGNORED),Git_mutations=0,scientific_phase_writes=0)
 new(STAGE/'QA.json',enc(qa))
 message=e['commit_message'];need(isinstance(message,str)and message.strip()and '\n'not in message,'One exact root commit title')
 plan=dict(authorized_main_push=True,repository=str(REPO),remote=REMOTE,branch='main',initial_remote_main=EXPECTED_PARENT,manifest=bind(PUB/(DELTA+'DELTA_MANIFEST.json')),owned_binary_whitelist=e['owned_binary_whitelist'],batches=[dict(message=message,files=physical)],batch_max_bytes=BATCH)
 new(STAGE/'ROOT_V223_PUSH_PLAN.json',enc(plan))
 result=dict(completed=True,PASS=True,entry=bind(entry),manifest=plan['manifest'],QA=bind(STAGE/'QA.json'),push_plan=bind(STAGE/'ROOT_V223_PUSH_PLAN.json'),original_source_files=len(original),source_PNGs=qa['source_PNGs'],original_source_bytes=qa['source_bytes'],physical_files=len(physical),physical_bytes=sum(x['size_bytes']for x in physical),owned_new_checkpoints=2,RAW_arrays_published=0,source_bindings_followed=0,model_image_array_decode=0,Git_mutations=0,staged_only=True)
 new(STAGE/'COMPLETE.json',enc(result));return result

def closed_plan(e):
 need(e['stage_complete']['path']==str(STAGE/'COMPLETE.json'),'Exact stage complete')
 done=admin(e['stage_complete']);need(done.get('completed')is True and done.get('PASS')is True and done['owned_new_checkpoints']==2 and done['RAW_arrays_published']==0,'Completed opaque2PT/zeroarray stage')
 need(e['push_plan']==done['push_plan']and e['manifest']==done['manifest'],'Exact closed stage graph')
 plan=admin(e['push_plan']);need(plan['repository']==str(REPO)and plan['remote']==REMOTE and plan['initial_remote_main']==EXPECTED_PARENT and plan['branch']=='main'and len(plan['batches'])==1,'One exact main plan')
 need(e['manifest']['path']==str(PUB/(DELTA+'DELTA_MANIFEST.json')),'Only own manifest before body')
 manifest=checked(e['manifest']);need(plan['manifest']==e['manifest']and manifest['source_roots']==[str(PHASE)]and manifest['source_bindings_followed']==0,'Closed source manifest')
 stageentry=admin(done['entry']);owned,_=closure(stageentry);need(manifest['owned_binary_whitelist']==plan['owned_binary_whitelist']==stageentry['owned_binary_whitelist'],'Same2PT across stage/manifest/plan')
 planned={}
 for rec in plan['batches'][0]['files']:
  need(set(rec)=={'path','sha256','size_bytes'}and rec['path']not in planned,'Unique physical schema');physical_path(rec['path'],owned)
  need(re.fullmatch('[0-9a-f]{64}',rec['sha256'])and type(rec['size_bytes'])is int and 0<=rec['size_bytes']<LIMIT,'Physical SHA/size')
  planned[rec['path']]=rec
 need(len(planned)==done['physical_files']and sum(x['size_bytes']for x in planned.values())==done['physical_bytes']<BATCH,'Plan actual counts/bytes')
 need({x for x in planned if Path(x).suffix.casefold()in RAW}=={PREFIX+ARCHIVE+r for r in owned},'Exactly new2PT physical RAW')
 return done,plan,manifest,owned,planned

def exact_bytes(done,manifest,owned,planned):
 sources=[x for x in manifest['entries']if x['category']=='original_scientific_record'];seen=set()
 for rec in sources:
  source=safe(rec['source_absolute']);need(PHASE in source.parents,'Only owned phase original')
  rel=str(source.relative_to(PHASE));need(rec['published_relative']==ARCHIVE+rel,'Exact source/destination identity')
  suffix=source.suffix.casefold();need(suffix in TEXT|{'.png'}or(rel in owned and suffix=='.pt'),'No undeclared RAW original before body')
  a=rb(source);b=rb(PUB/rec['published_relative']);need(a==b and digest(a)==rec['source_sha256']==rec['published_sha256']and len(a)==rec['size_bytes'],'Every original opaque byte/SHA/size')
  if rel in owned:need(digest(a)==owned[rel]['sha256']and len(a)==owned[rel]['size_bytes'],'Owned2PT actual original binding')
  name=PREFIX+rec['published_relative'];need(name in planned and planned[name]['sha256']==rec['source_sha256']and planned[name]['size_bytes']==rec['size_bytes'],'Original-to-plan closure');seen.add(str(source))
 need(len(seen)==len(sources)==done['original_source_files'],'No duplicate or missing original')
 for name,rec in planned.items():
  data=rb(physical_path(name,owned));need(digest(data)==rec['sha256']and len(data)==rec['size_bytes'],'Every full-plan destination SHA/size')
 for name,binding in manifest['history_before'].items():
  snapshot=PUB/(DELTA+'HISTORY_BEFORE/'+('PRIOR_THROUGH_V223.md'if name==PRIOR else name.replace('/','__')));old=rb(snapshot);need(digest(old)==binding['sha256'],'Original history snapshot bytes')
  dest=PUB/INDEX if name==PRIOR else REPO/name;need(rb(dest).endswith(old),'Every original history exact suffix preserved')
 return len(sources)
def working_delta(planned):
 changed=names(['diff','--name-only','-z','HEAD'])|names(['ls-files','--others','--exclude-standard','-z'])
 ignored=names(['ls-files','--others','--ignored','--exclude-standard','-z'])
 need(changed|(ignored-IGNORED)==set(planned),'Whole delta exactly plan; known cache names never opened')
def verify(e,entry):
 done,plan,manifest,owned,planned=closed_plan(e);n=exact_bytes(done,manifest,owned,planned);git_parent(plan);working_delta(planned)
 result=dict(completed=True,PASS=True,entry=bind(entry),code=bind(__file__),plan=e['push_plan'],stage_complete=e['stage_complete'],manifest=e['manifest'],originals=n,physical_files=len(planned),physical_bytes=done['physical_bytes'],owned_new_checkpoints=2,RAW_arrays_published=0,opaque_SHA_only=True,all_original_destination_and_plan_bytes_exact=True,originals_final_SHA_unchanged=True,whole_delta_exact_plan=True,index_empty=True,initial_main=EXPECTED_PARENT,ignored_cache_paths_not_opened=sorted(IGNORED),Git_mutations=0,model_array_image_decode=0)
 new(R/'ROOT_V223_SOURCE_AND_PLAN_BYTE_QA.json',enc(result));return result

def push(e,entry):
 need(e.get('approved_single_nonforce_main_push')is True,'Separate root push authority')
 qa=admin(e['byte_QA']);need(qa.get('completed')is True and qa.get('PASS')is True and qa['plan']==e['push_plan'],'Exact byte QA PASS before push')
 done,plan,manifest,owned,planned=closed_plan(e);exact_bytes(done,manifest,owned,planned);git_parent(plan);working_delta(planned)
 out=R/'V223_MAIN_PUSH_ROOT_TRANSACTION';mkdir_new(out);seq=0
 def save(stage,**kwargs):
  nonlocal seq;seq+=1;rec=dict(completed=False,stage=stage,entry=bind(entry),plan=e['push_plan'],byte_QA=e['byte_QA'],**kwargs);new(out/('%02d_%s.json'%(seq,stage)),enc(rec));return rec
 try:
  save('STARTED')
  force={p for p in planned if Path(p).suffix.casefold()in{'.png','.pt','.log'}};normal=set(planned)-force
  need({p for p in force if p.endswith('.pt')}=={PREFIX+ARCHIVE+r for r in owned}and not(force&IGNORED),'Force only declared PNG/new2PT/LOG leaves, never directories/cache')
  for label,paths in [('NORMAL',normal),('EXPLICIT_PNG_PT_LOG',force)]:
   if not paths:continue
   file=out/(label+'.paths');new(file,b''.join(p.encode()+b'\0'for p in sorted(paths)))
   git(['-c','core.autocrlf=false','add',*(['--force']if label=='EXPLICIT_PNG_PT_LOG'else[]),'--pathspec-from-file='+str(file),'--pathspec-file-nul'])
  git(['-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'])
  need(names(['diff','--cached','--name-only','-z'])==set(planned),'Exact staged names')
  index=tree_map(git(['ls-files','--stage','-z','--',*planned]),True);need(set(index)==set(planned),'Exact index path set');verify_blobs(index,planned);save('INDEX_BYTES_VERIFIED')
  message=out/'COMMIT_MESSAGE.txt';new(message,(plan['batches'][0]['message']+'\n').encode());git(['-c','gc.auto=0','commit','--quiet','--file='+str(message)])
  commit=git(['rev-parse','HEAD']).decode().strip();save('COMMITTED_PENDING_VERIFICATION',commit=commit)
  need(git(['rev-parse','HEAD^']).decode().strip()==EXPECTED_PARENT and names(['diff-tree','--no-commit-id','--name-only','-r','-z','HEAD'])==set(planned),'Single exact commit/parent/delta')
  tree=tree_map(git(['ls-tree','-r','-z','HEAD','--',*planned]),False);need(tree==index,'Exact committed index blob map');verify_blobs(tree,planned)
  need(not git(['status','--porcelain','--untracked-files=all'])and not git(['diff','--cached','--name-only','-z']),'Committed clean tree/index')
  save('COMMITTED_BYTES_VERIFIED',commit=commit);print(json.dumps(dict(event='PUSH_START',commit=commit)),flush=True)
  git(['-c','pack.window=0','-c','pack.compression=1','-c','pack.threads=2','push','origin','HEAD:refs/heads/main'])
  need(git(['ls-remote','origin','refs/heads/main']).decode().split()[0]==git(['rev-parse','HEAD']).decode().strip()==commit and not git(['status','--porcelain','--untracked-files=all']),'Remote/HEAD/clean verified')
  result=dict(completed=True,PASS=True,commit=commit,plan=e['push_plan'],byte_QA=e['byte_QA'],nonforce=True,physical_files=len(planned),physical_bytes=done['physical_bytes'],owned_new_checkpoints=2,RAW_arrays_published=0);new(out/'COMPLETE.json',enc(result));return result
 except BaseException as error:save('FAILURE_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise

def post(e,entry):
 done,plan,manifest,owned,planned=closed_plan(e);tx=admin(e['transaction']);need(tx.get('completed')is True and tx.get('PASS')is True and tx['plan']==e['push_plan'],'Exact completed push')
 commit=e['expected_commit'];need(re.fullmatch('[0-9a-f]{40}',commit)and tx['commit']==commit,'Exact pushed commit')
 n=exact_bytes(done,manifest,owned,planned)
 need(git(['rev-parse','HEAD']).decode().strip()==git(['ls-remote','origin','refs/heads/main']).decode().split()[0]==commit,'Post HEAD/remote main exact')
 need(not git(['status','--porcelain','--untracked-files=all'])and not git(['diff','--cached','--name-only','-z']),'Post normal clean/index empty')
 targets=e['HTTP_HEAD_targets'];need(isinstance(targets,list)and len(targets)==6 and len({x['path']for x in targets})==6 and len({x['role']for x in targets})==6,'Six unique HEAD targets')
 targetmap={x['role']:x['path']for x in targets};need(targetmap.get('INDEX')==PREFIX+INDEX and targetmap.get('ROOT_REPORT')==PREFIX+ARCHIVE+'PORTABLE_REPORT_KO.md','Index/root report HEAD targets')
 for x in targets:need(set(x)=={'role','path'}and x['path']in planned and Path(x['path']).suffix.casefold()not in RAW,'Exact non-RAW planned HEAD files')
 def head(x):
  url='https://github.com/CanelE452/pallet-6d-pose/blob/'+commit+'/'+quote(x['path'],safe='/')
  req=urllib.request.Request(url,method='HEAD',headers={'User-Agent':'Codex-V223-publication-QA'})
  with urllib.request.urlopen(req,timeout=30)as response:status=response.status
  need(status==200,'Immutable HTTP HEAD200');return dict(**x,url=url,status=status,response_body_read=False)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6)as pool:http=list(pool.map(head,targets))
 result=dict(completed=True,PASS=True,entry=bind(entry),commit=commit,plan=e['push_plan'],transaction=e['transaction'],source_original_files_rehashed=n,all_source_and_destination_SHA_unchanged=True,physical_files=len(planned),physical_bytes=done['physical_bytes'],HTTP_HEAD=http,working_tree_clean=True,index_empty=True,owned_new_checkpoints=2,RAW_arrays_published=0,source_bindings_followed=0,model_array_image_decode=0,scientific_phase_writes=0,latest_real_evaluation_unchanged='V220',selected_arm=None,adopted=False,fully_solved=False,goal_status='active')
 new(R/'ROOT_FINAL_V223_MAIN_PUBLICATION_LOCAL_RECEIPT.json',enc(result));return result

def selftest():
 global PHASE
 tests=[]
 def rejects(f):
  try:f();return False
  except AssertionError:return True
 tests.append(dict(name='lexical_V112_before_FS',PASS=rejects(lambda:safe('/not_opened/V112/x.png'))))
 fake=dict(approved_actual_operation='stage',approved_actual_enumeration_or_copy=False,automatic_retry=False,owned_binary_whitelist=[dict(path=str(PHASE/r),sha256='a'*64,size_bytes=12)for r in PT_RELS],excluded_RAW_paths=[])
 roundtrip=json.loads(enc(fake));tests.append(dict(name='stored_actual_schema_false_missing_reject_exact_two_PT',PASS=roundtrip==fake and rejects(lambda:copy_authority(roundtrip))and rejects(lambda:copy_authority({}))and copy_authority(dict(roundtrip,approved_actual_enumeration_or_copy=True))and len({b['path']for b in roundtrip['owned_binary_whitelist']})==2))
 owned={r:fake['owned_binary_whitelist'][i]for i,r in enumerate(PT_RELS)}
 tests.append(dict(name='PT2_only_no_inherited_PT_NPZ_cache',PASS=all(physical_path(PREFIX+ARCHIVE+r,owned)for r in owned)and rejects(lambda:physical_path(PREFIX+ARCHIVE+'C65_INITIAL.pt',owned))and rejects(lambda:physical_path(PREFIX+ARCHIVE+'inputs.npz',owned))and all(rejects(lambda q=q:physical_path(q,owned))for q in IGNORED)))
 paths={PREFIX+ARCHIVE+'opaque.png',PREFIX+ARCHIVE+'declared_stdout.log',PREFIX+ARCHIVE+'table.csv'}|{PREFIX+ARCHIVE+r for r in PT_RELS};force={p for p in paths if Path(p).suffix.casefold()in{'.png','.pt','.log'}}
 with tempfile.TemporaryDirectory(prefix='v223_publisher_invented_')as t:
  d=Path(t);data=b'column\r\n1\r\n';new(d/'source.csv',data);new(d/'dest.csv',rb(d/'source.csv'))
  (d/'symlink.csv').symlink_to(d/'source.csv')
  try:rb(d/'symlink.csv');symlink_refused=False
  except OSError:symlink_refused=True
  tests.append(dict(name='force_exact_declared_PNG_PT_LOG_only_CRLF_bytecopy_nofollow',PASS=len(force)==4 and not(force&IGNORED)and rb(d/'dest.csv')==data and symlink_refused))
 # 새 inventory의 포함 PT/제외 RAW/cache 분기는 외부 임시 toy tree에서만 실행한다.
 saved=PHASE
 with tempfile.TemporaryDirectory(prefix='v223_inventory_invented_')as t:
  PHASE=Path(t)
  try:
   for rel in PT_RELS:new(PHASE/rel,b'invented opaque checkpoint bytes')
   new(PHASE/'note.json',b'{}');new(PHASE/'excluded.npz',b'invented opaque excluded bytes')
   new(PHASE/'__pycache__'/'unread',b'unread')
   found,skips=inventory({r:{'size_bytes':len(b'invented opaque checkpoint bytes')}for r in PT_RELS},{'excluded.npz'})
   need({str(x.relative_to(PHASE))for x,_ in found}==set(PT_RELS)|{'note.json'}and{r['path']for r in skips}=={'excluded.npz','__pycache__'},'Invented inventory exactPT/excludedRAW/cache')
  finally:PHASE=saved
 need(all(x['PASS']for x in tests),'Invented functional guard tests');return dict(completed=True,PASS=True,invented_only=True,checks=tests,source_phase_enumeration_hash_copy=0,clone_Git_network_access=0)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('stage','verify','push','post'));ap.add_argument('--entry');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
 if args.self_test:print(json.dumps(selftest(),ensure_ascii=False));return
 need(args.mode is not None and args.entry is not None,'Mode and separate root actual entry required');e=entry_gate(args.entry,args.mode)
 try:result={'stage':stage,'verify':verify,'push':push,'post':post}[args.mode](e,args.entry)
 except BaseException as error:
  failure=R/('V223_'+args.mode.upper()+'_FAILURE_NO_RETRY.json');new(failure,enc(dict(completed=False,PASS=False,entry=bind(args.entry),error_type=type(error).__name__,error=str(error),automatic_retry=False)));raise
 print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
