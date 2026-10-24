"""Lean frozen V207 + V208 SOURCE data/definition publication: no Git/network/scientific decode.
--prepare invented only. --entry requires root final freeze notice and bindings.
"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,hashlib,json,os,posixpath,re,stat,tempfile
H=Path(__file__).absolute().parent;P=H.parent.parent
PHASE207=P/'query_visibility_source_train_witness_conditioned_controls_20261025_v207'
PHASE208=P/'query_visibility_source_train_native_point_state_grounding_20261025_v208'
PHASES=(PHASE207,PHASE208)
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024');PREFIX='_docs/experiments/pallet-keypoint-visibility/';PUB=REPO/PREFIX
ARCHIVE_PREFIX='archive/task-2/pallet-keypoint-visibility/'
DELTA='publication/deltas/V208_20261025_v1/';INDEX='EXPERIMENT_INDEX_THROUGH_V208_KO.md';PRIOR='EXPERIMENT_INDEX_THROUGH_V206_KO.md'
PROTOCOL=H/'PROTOCOL_BEFORE_ACTUAL_V207_V208_ENUMERATION_OR_COPY.json'
ROOT_SCOPE={'path':str(P/'publication_root_preparation/ROOT_PROSPECTIVE_V207_V208_COMPLETED_PUBLICATION_SCOPE.json'),'sha256':'6304a00cfda7bbd46e87cd631faea3105863edb0664e280ea70f73e7f697b67e'}
LIMIT=100*1024**2;BATCH=480*1024**2;MAX_ORIGINAL_FILES=450;MAX_ORIGINAL_BYTES=40000000
TEXT={'.py','.json','.jsonl','.csv','.md','.txt','.html','.patch','.log'}
RAW={'.npz','.npy','.pt','.pth','.bin','.safetensors','.pkl','.pickle'}
EXPECTED_PARENT='71bb3815f6ea556603eb4ec466813e9a390a770d'
INHERITED_V206_PUBLISHER={'path':str(P/'publication_root_preparation/V206_PUBLICATION_PREPARATION_A/stage_completed_v206.py'),'sha256':'dcd1bee6cc3aba47f2b72174efce35b21dddce42ebee252a2eedb989c2968130'}
REQUIRED_ROLES=('V207_ROOT_PHASE_RESULT','V207_ROOT_FINAL_FREEZE','V207_ACTUAL_CONTROLS','V207_ROOT_FIXED8_ACTUAL_VISUAL_QA','V207_INDEPENDENT_SAVED_PIXEL_QA','V207_ROOT_REPORT_BUILD','V208_ROOT_PHASE_RESULT','V208_ROOT_FINAL_FREEZE','V208_ACTUAL_V2_POINT_LOOKUP','V208_INDEPENDENT_SAVED_GT_QA','V208_ROOT_REPORT_BUILD','V208_ROOT_FINAL_REPORT_QA')
REPORT207=ARCHIVE_PREFIX+PHASE207.name+'/PORTABLE_REPORT_KO.md'
REPORT208=ARCHIVE_PREFIX+PHASE208.name+'/PORTABLE_REPORT_KO.md'
GALLERY207=ARCHIVE_PREFIX+PHASE207.name+'/raw96_weak_witness_control_pilot_v1/PORTABLE_GALLERY_KO.md'
CSV207=ARCHIVE_PREFIX+PHASE207.name+'/raw96_weak_witness_control_pilot_v1/ALL80_STATUS.csv'
CSV208=ARCHIVE_PREFIX+PHASE208.name+'/NATIVE328_POINT_REGION_LOCALITY.csv'
MANDATORY_PORTABLE_PAGES={INDEX,REPORT207,REPORT208,GALLERY207}
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
def publication_headers():
 index_header=('# V207 + V208 · SOURCE 제어·판정 정의 진전\n\n[V207 RAW96 weak witness controls]('+REPORT207+') · [V207 전체80 gallery]('+GALLERY207+') · [80상태 CSV]('+CSV207+')\n\n[V208 point0·region 저장값]('+REPORT208+') · [native328 CSV]('+CSV208+') · [이전 V206 인덱스]('+PRIOR+')\n\nV207은 기존 공개 TRAIN3프레임·8사례에서 80시도를 모두 보존했고 18쌍 구성/62실패를 기록했다. 이는 post-sampler RAW96 weak-feature control이며 native E나 학습 admission이 아니다. V208은 original BEFORE41부모·native328의 저장 point/region 정의를 조회했고 point D·region U123을 분리했다. annotation 오류 수나 모델 원인을 인증하지 않는다. 두 단계 모두 새 모델 개선·채택을 주장하지 않으며 최신 실제 실사 비교는 V203 그대로다. 아래 이전 V206 인덱스는 바이트 그대로다.\n\n---\n\n').encode()
 readme_header=('# V207 + V208 · SOURCE 제어·판정 정의 진전 게시\n\n[최신 V208 통합 인덱스]('+PREFIX+INDEX+') · [V207 제어 gallery]('+PREFIX+GALLERY207+') · [V208 source point/region 진단]('+PREFIX+REPORT208+')\n\n18쌍/62실패와 저장 point D·region U123/328을 포함해 방법·시도·검증·한계를 보존한다. post-sampler weak control과 합성 내부 point 상태는 실사 native gold·학습 성공이 아니다. 최신 실제 실사 비교 V203은 그대로이며 모델 개선·전체 해결을 주장하지 않는다.\n\n---\n\n').encode()
 return index_header,readme_header

def role_phase(role):
 need(role in REQUIRED_ROLES,'Unknown completion role')
 return PHASE207 if role.startswith('V207_')else PHASE208

def original_destination(root,source):
 need(root in PHASES and root in safe(source).parents,'Exact owned source root')
 return ARCHIVE_PREFIX+root.name+'/'+str(source.relative_to(root))

def selftest():
 t=[]
 def add(n,x):t.append(dict(name=n,PASS=bool(x)));need(x,n)
 add('stable_SHA',digest(b'abc')=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
 for bad in ('/invented/V112/a','/invented/../a'):
  try:safe(bad);ok=False
  except AssertionError:ok=True
  add('lexical_reject_before_FS_'+str(len(t)),ok)
 with tempfile.TemporaryDirectory(dir=H,prefix='invented_')as tmp:
  d=Path(tmp);(d/'proof.json').write_bytes(b'{}');(d/'failure.patch').write_bytes(b'kept');(d/'view.png').write_bytes(b'opaque invented PNG')
  (d/'__pycache__').mkdir();(d/'__pycache__'/'ignore').write_bytes(b'ignored')
  for name in ('old.npz','array.npy','weights.pt'):(d/name).write_bytes(b'opaque no decode')
  ff,ee=inventory(d);add('positive_records_failurepatch_and_PNG_preserved_zero_arrays',set(x.name for x in ff)=={'proof.json','failure.patch','view.png'}and len(ee)==4)
  new(d/'exact.txt',b'exact');add('exclusive_copy_exactbytes',rb(d/'exact.txt')==b'exact')
  try:new(d/'exact.txt',b'overwrite');ok=False
  except FileExistsError:ok=True
  add('exclusive_no_overwrite',ok)
  (d/'link').symlink_to(d/'proof.json')
  try:inventory(d);ok=False
  except AssertionError:ok=True
  add('symlink_refused_before_read',ok);(d/'link').unlink();(d/'unexpected.svg').write_bytes(b'notwhitelisted')
  try:inventory(d);ok=False
  except AssertionError:ok=True
  add('unlisted_extension_fails_closed',ok)
 add('same_named_records_two_exact_destinations_disjoint',original_destination(PHASE207,PHASE207/'COMPLETE.json')!=original_destination(PHASE208,PHASE208/'COMPLETE.json'))
 add('all12_completion_roles_exact_split',len(REQUIRED_ROLES)==12 and sum(role_phase(r)==PHASE207 for r in REQUIRED_ROLES)==6 and sum(role_phase(r)==PHASE208 for r in REQUIRED_ROLES)==6)
 frozen=dict(completed=True,frozen=True,all_scientific_phase_files_final_no_more_writes=True,administrative_publication_receipts_written_elsewhere=True,phase_result=dict(path=str(PHASE207/'ROOT_PHASE_RESULT.json'),sha256='a'*64))
 add('actual_authoritative_freeze_fields_accept',frozen_phase(frozen,frozen['phase_result']))
 try:frozen_phase(dict(frozen,frozen=False));ok=False
 except AssertionError:ok=True
 add('missing_or_false_freeze_rejected',ok)
 try:frozen_phase(dict(scientific_phase_frozen=True,no_further_phase_writes=True));ok=False
 except AssertionError:ok=True
 add('legacy_alias_freeze_not_authority_rejected',ok)
 try:frozen_phase(frozen,dict(path=str(PHASE208/'ROOT_PHASE_RESULT.json'),sha256='b'*64));ok=False
 except AssertionError:ok=True
 add('freeze_phase_result_cross_role_mismatch_rejected',ok)
 add('relative_parent_links_preserve_published_sibling',links('archive/a/r.md',b'[prior](../b/r.md)')==['archive/b/r.md'])
 add('credential_format_detection_without_secret_output',hits(('ghp_'+'x'*40).encode())=={'GitHub':1})
 ih,rh=publication_headers()
 add('header_latest208_both207_prior206_data_not_model_claim',ih.startswith(b'# V207 + V208')and rh.startswith(b'# V207 + V208')and PRIOR.encode()in ih and REPORT207.encode()in ih and REPORT208.encode()in ih and GALLERY207.encode()in ih and CSV208.encode()in ih and b'V205'not in ih+rh)
 add('previous_index_and_README_suffix',(ih+b'oldindex').endswith(b'oldindex')and(rh+b'oldreadme').endswith(b'oldreadme'))
 return dict(completed=True,PASS=True,invented_only=True,checks=t,actual_source_phase_or_repository_or_API_or_Git_access=0)

def prepare():
 q=selftest();new(H/'GENERATED_INVENTED_CHECKS.json',enc(q))
 p=dict(completed=True,prepared_only=True,code=bind(__file__),checks=bind(H/'GENERATED_INVENTED_CHECKS.json'),root_scope=ROOT_SCOPE,source_roots=list(map(str,PHASES)),repository=str(REPO),inherited_publisher=INHERITED_V206_PUBLISHER,revision='V2 root-authoritative freeze schema adapter only; V1 prepared but never staged and remains immutable. Actual freeze.phase_result exact-crossbinds same phase role.',preserved_v1_ready=dict(path=str(P/'publication_root_preparation/V208_PUBLICATION_PREPARATION_A/FINAL_READY.json'),sha256='55a0bf855601c15be5a71b25364022c06606970da2d82a83e70162305619f357'),
 source_scope='Only two exact completed frozen V207/V208 roots; never follows any nested source binding. Positive extensions .py/.json/.jsonl/.csv/.md/.html/.png/.log/.patch/.txt. Retain failures/UNKNOWN/history. Exclude caches and every NPZ/NPY/model binary before body/header/hash. No image/scientific decode.',
 required_completion_roles=list(REQUIRED_ROLES),freeze_fields=dict(completed=True,frozen=True,all_scientific_phase_files_final_no_more_writes=True,administrative_publication_receipts_written_elsewhere=True),expected_initial_main=EXPECTED_PARENT,budget=dict(max_original_files=MAX_ORIGINAL_FILES,max_original_bytes=MAX_ORIGINAL_BYTES,max_batch_bytes=BATCH,max_each_file_bytes_strict=LIMIT),
 source_to_destination='P/<exact phase>/<relative> -> archive/task-2/pallet-keypoint-visibility/<same exact phase>/<same relative>; original byteSHA/size and final rehash, no phase writes or old archive changes.',
 portable_policy='Declared root reports/gallery/index entry pages only are link-audited; original provenance MD is preserved byteexact without treating it as an entry. Root supplies exact portable pages and mandatory four must be present. Both CSVs are direct relative links from new index. No logical compressed links; all small artifacts RAW.',
 old_suffix_policy='Prepend current README and new THROUGH_V208 index; exact previous THROUGH_V206 index bytes form new index suffix. No old inventory/manifests/ancestor scans or extra closing cycle.',
 owned_models_arrays_targets_inputs=0,original_asset_bindings_followed=0,Git_or_API_calls=0,automatic_retry=False,actual_phase_enumeration_hash_copy_started=False)
 new(PROTOCOL,enc(p))
 conventional={'V207_ROOT_PHASE_RESULT':PHASE207/'ROOT_PHASE_RESULT.json','V207_ROOT_FINAL_FREEZE':PHASE207/'ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json','V208_ROOT_PHASE_RESULT':PHASE208/'ROOT_PHASE_RESULT.json','V208_ROOT_FINAL_FREEZE':PHASE208/'ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json','V208_ACTUAL_V2_POINT_LOOKUP':PHASE208/'source_train_native_point_grounding_v2/COMPLETE.json','V208_INDEPENDENT_SAVED_GT_QA':PHASE208/'independent_actual_saved_gt_review_b_v2/REVIEW.json','V208_ROOT_REPORT_BUILD':PHASE208/'source_report_preparation_a/REPORT_BUILD_COMPLETE.json','V208_ROOT_FINAL_REPORT_QA':PHASE208/'ROOT_FINAL_SAVED_POINT_REPORT_QA.json'}
 receipts=[dict(role=r,binding=dict(path=str(conventional[r])if r in conventional else None,sha256=None),expected_fields=dict(completed=True,frozen=True,all_scientific_phase_files_final_no_more_writes=True,administrative_publication_receipts_written_elsewhere=True)if r.endswith('ROOT_FINAL_FREEZE')else None)for r in REQUIRED_ROLES]
 template=dict(approved_actual_V207_V208_stage=False,root_explicit_both_final_freeze_notice=False,code=p['code'],protocol=bind(PROTOCOL),root_scope=ROOT_SCOPE,source_roots=list(map(str,PHASES)),completion_receipts=receipts,initial_remote_main=None,repository=str(REPO),previous_index=dict(path=str(PUB/PRIOR),sha256=None),global_readme_before=dict(path=str(REPO/'readme.md'),sha256=None),portable_pages=sorted(MANDATORY_PORTABLE_PAGES),output_directory=str(P/'publication_root_preparation/V208_PUBLICATION_STAGE_A'),automatic_retry=False,Git_add_commit_push=False)
 new(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',enc(template))
 new(H/'FINAL_READY.json',enc(dict(completed=True,PASS=True,FINAL_READY_FROZEN=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],false_entry=bind(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_phase_repository_inventory_hash_copy_API_or_Git_access=0)))
 print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),checks=len(q['checks']))))

def run(entry):
 e=json.loads(rb(entry));need(e['approved_actual_V207_V208_stage']is True and e['root_explicit_both_final_freeze_notice']is True,'Wait both root final freezes and actual entry')
 need(e['automatic_retry']is False and e['Git_add_commit_push']is False,'No retry/Git')
 p=checked(e['protocol']);need(e['code']==bind(__file__)==p['code']and e['protocol']==bind(PROTOCOL),'Frozen code/protocol')
 need(e['root_scope']==ROOT_SCOPE,'Exact prospective scope');scope=checked(ROOT_SCOPE)
 need(scope['source_roots']==e['source_roots']==list(map(str,PHASES)),'Exact two roots, no aliases')
 need(e['repository']==str(REPO)and e['initial_remote_main']==EXPECTED_PARENT,'Exact root-verified initial main')
 roles={}
 for x in e['completion_receipts']:
  need(x['role']not in roles,'Unique completion role');phase=role_phase(x['role']);need(phase in safe(x['binding']['path']).parents,'Role belongs to exact phase')
  q=checked(x['binding']);need(x['expected_fields'],'Root expected completion fields required')
  for k,v in x['expected_fields'].items():need(q[k]==v,'Completion field '+k)
  roles[x['role']]=x
 need(set(roles)==set(REQUIRED_ROLES),'Exactly12 completion roles')
 for label,phase in [('V207',PHASE207),('V208',PHASE208)]:
  need(roles[label+'_ROOT_PHASE_RESULT']['binding']['path']==str(phase/'ROOT_PHASE_RESULT.json'),'Exact root phase result')
  need(roles[label+'_ROOT_FINAL_FREEZE']['binding']['path']==str(phase/'ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json'),'Exact root phase freeze')
  frozen_phase(checked(roles[label+'_ROOT_FINAL_FREEZE']['binding']),roles[label+'_ROOT_PHASE_RESULT']['binding'])
 pages=set(e['portable_pages']);need(MANDATORY_PORTABLE_PAGES<=pages,'Both reports/gallery/index required')
 for page in pages:need(page==INDEX or any(page.startswith(ARCHIVE_PREFIX+z.name+'/')and Path(page).suffix.lower()in('.md','.html')for z in PHASES),'Portable entry owned exact2 roots')
 out=safe(e['output_directory']);need(out==P/'publication_root_preparation/V208_PUBLICATION_STAGE_A','Exact separate admin output')
 out.mkdir(exist_ok=False);new(out/'STARTED.json',enc(dict(completed=False,entry=bind(entry),automatic_retry=False)))
 try:
  need(e['previous_index']['path']==str(PUB/PRIOR)and e['global_readme_before']['path']==str(REPO/'readme.md'),'Exact old index/README')
  oldindex=rb(e['previous_index']['path']);need(digest(oldindex)==e['previous_index']['sha256'],'OldindexSHA')
  oldreadme=rb(e['global_readme_before']['path']);need(digest(oldreadme)==e['global_readme_before']['sha256'],'OldREADME SHA')
  inventory_by_root={};excluded={}
  for root in PHASES:inventory_by_root[root],excluded[root.name]=inventory(root)
  pairs=[(root,source)for root in PHASES for source in inventory_by_root[root]]
  need(pairs and len(pairs)<=MAX_ORIGINAL_FILES,'Bounded nonempty original count')
  original_bytes=sum(os.stat(s,follow_symlinks=False).st_size for _,s in pairs);need(original_bytes<=MAX_ORIGINAL_BYTES,'Original bytebudget before read/copy')
  records=[];physical=[];texts={};source_hashes={}
  def store(rel,data,source,category):
   need(len(data)<LIMIT,'Each Gitfile<100MiB');need(not hits(data)if Path(rel).suffix.lower()in TEXT else True,'Credential format hit, values withheld')
   new(PUB/rel,data);need(rb(PUB/rel)==data,'Exact destination bytes')
   physical.append(dict(path=PREFIX+rel,sha256=digest(data),size_bytes=len(data)))
   row=dict(source_root_id='P'if category=='original_scientific_record'else'PUBLICATION_ADMIN_CODE'if category=='publication_tool'else'DELTA_DERIVED',source_absolute=str(source),source_relative_path=str(source.relative_to(P))if P in source.parents else rel,source_sha256=digest(data),size_bytes=len(data),published_relative=rel,published_sha256=digest(data),status='PUBLISHED_RAW',category=category,exclusion_reason=None,gzipchunks=None);records.append(row)
   if rel in pages:texts[rel]=data
  for root,source in pairs:
   data=rb(source);source_hashes[str(source)]=digest(data);store(original_destination(root,source),data,source,'original_scientific_record')
  for source in [Path(__file__).absolute(),PROTOCOL,H/'GENERATED_INVENTED_CHECKS.json']:store('tools/v208_lean_publication/'+source.name,rb(source),source,'publication_tool')
  header,readme_prefix=publication_headers();store(INDEX,header+oldindex,PUB/INDEX,'versioned_index');need(rb(PUB/INDEX).endswith(oldindex),'Oldindex suffix')
  store(DELTA+'GLOBAL_README_BEFORE_V208.md',oldreadme,PUB/(DELTA+'GLOBAL_README_BEFORE_V208.md'),'old_global_snapshot')
  manifest=dict(format='immutable-completed-phase-delta-v1',phase='V207_AND_V208_COMPLETED_SOURCE_ARTIFACTS',source_roots=list(map(str,PHASES)),previous_index=e['previous_index'],root_entry=bind(entry),completion_receipts=e['completion_receipts'],entries=records,owned_binary_whitelist=[],excluded_cache_paths=excluded,source_bindings_followed=0,old_archive_or_manifest_paths_modified=False,no_second_closing_required=True)
  mz=enc(manifest);new(PUB/(DELTA+'DELTA_MANIFEST.json'),mz);physical.append(dict(path=PREFIX+DELTA+'DELTA_MANIFEST.json',sha256=digest(mz),size_bytes=len(mz)))
  newreadme=readme_prefix+oldreadme;need(newreadme.endswith(oldreadme),'README suffix')
  need(set(texts)==pages,'Every declared portable page published');linkrows=[]
  for doc,data in texts.items():
   targets=links(doc,data)
   for target in targets:need(existing_target(PUB/target),'Missing portable link '+doc+' -> '+target)
   linkrows.append(dict(page=doc,local_links=len(targets)))
  for csvrel in (CSV207,CSV208):need(existing_target(PUB/csvrel),'Both numeric CSVs directly accessible')
  rootlinks=links('ROOT_REPO_README.md',readme_prefix)
  for target in rootlinks:need(existing_target(REPO/target),'New README target missing')
  linkrows.append(dict(page='readme.md newprefix only; oldsuffix byte-preserved',local_links=len(rootlinks)))
  for root in PHASES:
   pp,ee=inventory(root);need(pp==inventory_by_root[root]and ee==excluded[root.name],'Frozen source inventory unchanged')
  replace_readme(REPO/'readme.md',newreadme,e['global_readme_before']['sha256']);physical.append(dict(path='readme.md',sha256=digest(newreadme),size_bytes=len(newreadme)))
  for source,h in source_hashes.items():need(digest(rb(source))==h,'Original scientific SHA changed')
  for x in physical:need(digest(rb(REPO/x['path']))==x['sha256'],'Physical file SHA')
  need(len({x['path']for x in physical})==len(physical)and sum(x['size_bytes']for x in physical)<BATCH,'One bounded unique batch')
  qa=dict(completed=True,PASS=True,entry=bind(entry),required12_completion_roles_verified=True,both_source_phases_frozen=True,original_source_files=len(pairs),original_source_bytes=original_bytes,original_counts_by_phase={root.name:len(inventory_by_root[root])for root in PHASES},all_original_to_destination_SHA_size_exact=True,source_original_final_SHA_unchanged=True,original_destination_identity_exact=True,all_declared_portable_local_links_resolve=True,link_pages=linkrows,local_links=sum(x['local_links']for x in linkrows),credential_format_detections=0,oldindex_and_README_suffix_preserved=True,source_bindings_followed=0,owned_models=0,new_prediction_NPZ=0,targets_inputs_NPZ=0,old_archive_mutations=0,model_array_image_decode=0,Git_calls=0,automatic_retry=False);new(out/'QA.json',enc(qa))
  plan=dict(authorized_main_push=True,repository=str(REPO),remote='https://github.com/CanelE452/pallet-6d-pose.git',branch='main',initial_remote_main=e['initial_remote_main'],manifest_sha256=digest(mz),batch_max_bytes=BATCH,batches=[dict(files=physical,message='Publish V207 source witness controls and V208 point-region grounding')],scope='Exact completed V207+V208 source records plus README prepend and V208 index; no scientific changes or second closing cycle.');new(out/'ROOT_V208_PUSH_PLAN.json',enc(plan))
  complete=dict(completed=True,PASS=True,staged_only_not_pushed=True,entry=bind(entry),manifest=bind(PUB/(DELTA+'DELTA_MANIFEST.json')),QA=bind(out/'QA.json'),push_plan=bind(out/'ROOT_V208_PUSH_PLAN.json'),owned_models=0,new_prediction_NPZ=0,targets_inputs_NPZ=0,original_source_files=len(pairs),physical_files=len(physical),physical_bytes=sum(x['size_bytes']for x in physical),excluded_cache_paths=excluded,Git_calls=0,phase_writes=0);new(out/'COMPLETE.json',enc(complete));print(json.dumps(complete))
 except BaseException as error:
  new(out/'FAILURE_PRESERVED_NO_RETRY.json',enc(dict(completed=False,PASS=False,error_type=type(error).__name__,error=str(error),automatic_retry=False)));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);v=a.parse_args();prepare()if v.prepare else run(v.entry)
