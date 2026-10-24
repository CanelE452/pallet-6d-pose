"""Exact future frozen V212/V213 paired source/real publication; opaque bytes only, no Git."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,hashlib,json,os,posixpath,re,stat,tempfile
H=Path(__file__).absolute().parent;P=H.parent.parent
PHASE212=P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212'
PHASE213=P/'query_visibility_dense_point_heads_frozen_real_benchmark_20261025_v213'
PHASES=(PHASE212,PHASE213)
LABELS=('V212','V213')
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024');PREFIX='_docs/experiments/pallet-keypoint-visibility/';PUB=REPO/PREFIX
ARCHIVE_PREFIX='archive/task-2/pallet-keypoint-visibility/'
DELTA='publication/deltas/V213_20261025_v1/';INDEX='EXPERIMENT_INDEX_THROUGH_V213_KO.md';PRIOR='EXPERIMENT_INDEX_THROUGH_V211_KO.md'
PROTOCOL=H/'PROTOCOL_BEFORE_ACTUAL_V212_V213_ENUMERATION_OR_COPY.json'
ROOT_SCOPE={'path':str(P/'publication_root_preparation/ROOT_PROSPECTIVE_V212_V213_COMPLETED_PUBLICATION_PREPARATION_SCOPE.json'),'sha256':'9beda36a88964fbac59bdf396eba16efe96e58f4aaaf3c06a3388e4a2cf82666'}
INHERITED={'path':str(P/'publication_root_preparation/V211_PUBLICATION_PREPARATION_C/stage_completed_v211.py'),'sha256':'624787eece70824d989a233d092fff877450a80a738eec053cd79be77dd7fc97'}
EXPECTED_PARENT='be5b2585eda1efc7fccc3e9f1ae6b40bfc1d06c5'
LIMIT=100*1024**2;BATCH=480*1024**2;MAX_ORIGINAL_FILES=600;MAX_ORIGINAL_BYTES=300*1024**2
TEXT={'.py','.json','.jsonl','.csv','.md','.txt','.html','.patch','.log'}
RAW={'.npz','.npy','.pt','.pth','.bin','.safetensors','.pkl','.pickle'}
REQUIRED_ROLES=tuple(label+'_'+role for label in LABELS for role in ('ROOT_PHASE_RESULT','ROOT_FINAL_FREEZE'))
REPORTS=[ARCHIVE_PREFIX+z.name+'/PORTABLE_REPORT_KO.md'for z in PHASES]
MANDATORY_PORTABLE_PAGES={INDEX,*REPORTS}
ARMS=('C62_DENSE_POINT_RAW_Q8','C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8')
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

def validate_owned_whitelists(documents):
 need(len(documents)==2,'Exactly two future owned-output whitelists');owned={}
 for label,root,doc in zip(LABELS,PHASES,documents):
  need(doc.get('completed')is True and doc.get('phase')==label and isinstance(doc.get('owned_files'),list),'Exact phase owned whitelist')
  counts={'OWN_NEW_CHECKPOINT':0,'OWN_NEW_PREDICTION_NPZ':0};byarm={(category,arm):0 for category in counts for arm in ARMS}
  for record in doc['owned_files']:
   need(set(record)=={'path','sha256','size_bytes','category'},'Explicit path/SHA/size/category only')
   source=safe(record['path']);need(root in source.parents and source.parent.name in ARMS,'Only own same-phase C62/C63 output')
   need(re.fullmatch('[0-9a-f]{64}',record['sha256'])and type(record['size_bytes'])is int and 0<record['size_bytes']<LIMIT,'Owned SHA/size schema')
   category=record['category'];need(category in counts,'No input/target/inherited-model category')
   if category=='OWN_NEW_CHECKPOINT':need(root==PHASE212 and source.name=='FINAL_CHECKPOINT.pt','Only new two source final checkpoints')
   else:need(source.suffix=='.npz'and re.fullmatch('FINAL_[A-Z0-9_]+_OUTPUTS[.]npz',source.name),'Only own final prediction OUTPUTS packets')
   need(str(source)not in owned,'No duplicate owned binary');owned[str(source)]=record;counts[category]+=1;byarm[category,source.parent.name]+=1
  need(counts==({'OWN_NEW_CHECKPOINT':2,'OWN_NEW_PREDICTION_NPZ':22}if label=='V212'else{'OWN_NEW_CHECKPOINT':0,'OWN_NEW_PREDICTION_NPZ':4}),'Exact 2 new PT +26 prediction NPZ distribution')
  need(all(byarm['OWN_NEW_CHECKPOINT',arm]==(1 if label=='V212'else 0)and byarm['OWN_NEW_PREDICTION_NPZ',arm]==(11 if label=='V212'else 2)for arm in ARMS),'Both arms complete, no arm selection')
 return owned

def inventory(root,owned=None):
 owned={}if owned is None else owned
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
     if suffix in RAW and str(source)not in owned:
      excluded.append(dict(path=str(source.relative_to(root)),reason='Non-whitelisted source/input/target/model binary excluded before header/body/hash'));continue
     if str(source)in owned:need(q.stat(follow_symlinks=False).st_size==owned[str(source)]['size_bytes'],'Owned binary size exactly root whitelist')
     else:need(suffix in TEXT|{'.png'},'Positive code/text/PNG policy only')
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

def publication_headers(entry):
 # Final result prose is explicitly supplied and reviewed by root after reports.
 result=[]
 for key in('index_prefix_text','readme_prefix_text'):
  text=entry.get(key);need(isinstance(text,str)and text.startswith('# ')and text.endswith('\n\n---\n\n'),'Root final reviewed prefix required')
  result.append(text.encode())
 return tuple(result)

def role_phase(role):
 need(role in REQUIRED_ROLES,'Exact completion role')
 return PHASES[LABELS.index(role.split('_',1)[0])]

def selftest():
 docs=[]
 for label,root in zip(LABELS,PHASES):
  rows=[]
  for arm in ARMS:
   if label=='V212':rows.append(dict(path=str(root/'invented'/arm/'FINAL_CHECKPOINT.pt'),sha256='a'*64,size_bytes=123,category='OWN_NEW_CHECKPOINT'))
   for k in range(11 if label=='V212'else 2):rows.append(dict(path=str(root/'invented'/arm/('FINAL_SOURCE_%02d_OUTPUTS.npz'%k)),sha256='b'*64,size_bytes=234,category='OWN_NEW_PREDICTION_NPZ'))
  docs.append(dict(completed=True,phase=label,owned_files=rows))
 need(len(validate_owned_whitelists(docs))==28,'Exact28 owned binary declarations accepted without filesystem');tests=['exact2PT26predictionNPZ']
 for kind in('input','outside','V112','inherited','duplicate'):
  fake=json.loads(json.dumps(docs));r=fake[0]['owned_files'][0]
  if kind=='input':r.update(category='OWN_NEW_PREDICTION_NPZ',path=str(PHASE212/ARMS[0]/'INPUTS.npz'))
  elif kind=='outside':r['path']=str(P/'outside'/ARMS[0]/'FINAL_CHECKPOINT.pt')
  elif kind=='V112':r['path']=str(PHASE212/'V112'/ARMS[0]/'FINAL_CHECKPOINT.pt')
  elif kind=='inherited':r['category']='INHERITED_MODEL'
  else:fake[0]['owned_files'][1]=dict(r)
  rejected=False
  try:validate_owned_whitelists(fake)
  except AssertionError:rejected=True
  need(rejected,'Reject '+kind+' before filesystem');tests.append(kind+'_rejected')
 return dict(completed=True,PASS=True,invented_only=True,checks=tests,actual_source_repository_or_API_access=0)


def prepare():
 q=selftest();new(H/'GENERATED_INVENTED_CHECKS.json',enc(q))
 p=dict(completed=True,prepared_only=True,code=bind(__file__),checks=bind(H/'GENERATED_INVENTED_CHECKS.json'),root_scope=ROOT_SCOPE,inherited_publisher=INHERITED,
  source_roots=list(map(str,PHASES)),required_completion_roles=list(REQUIRED_ROLES),expected_parent=EXPECTED_PARENT,
  policy='Exact2 future frozen roots only. Positive inherited text/PNG plus root-explicit2newPT26newpredictionNPZ; opaque bytecopy/hash only. Never deserialize, no source/input/target/GT/inheritedweight binding follows.',
  owned_schema=dict(completed=True,phase='V212 or V213',owned_files=[dict(path='absolute samephase own finaloutput',sha256='64 lowerhex',size_bytes='positive integer <100MiB',category='OWN_NEW_CHECKPOINT or OWN_NEW_PREDICTION_NPZ')]),
  binary_counts=dict(V212=dict(OWN_NEW_CHECKPOINT=2,OWN_NEW_PREDICTION_NPZ=22),V213=dict(OWN_NEW_CHECKPOINT=0,OWN_NEW_PREDICTION_NPZ=4)),
  before_actual='Separate root actualscope/entry; exact two final phase/freeze barriers, owned whitelists, root-reviewed final prefix texts and portablepage list. No actual enumeration/hash/copy during prep.',
  old_suffix_policy='Exact previous THROUGH_V211 index and current README preserved as suffix; new THROUGH_V213 only.',
  original_file_cap=MAX_ORIGINAL_FILES,original_byte_cap=MAX_ORIGINAL_BYTES,each_file_cap_strict=LIMIT,batch_cap_strict=BATCH,Git_mutations=False,automatic_retry=False,
  preparation_edit_failure_preserved=bind(H/'PREPARATION_INDENTATION_FAILURE_PRESERVED/RECORD.json'))
 new(PROTOCOL,enc(p))
 template=dict(approved_actual_V212_V213_stage=False,code=p['code'],protocol=bind(PROTOCOL),root_scope=ROOT_SCOPE,root_actual_scope=None,
  source_roots=list(map(str,PHASES)),initial_remote_main=None,repository=str(REPO),completion_receipts=[],owned_output_whitelists=[dict(path=str(z/'ROOT_OWNED_OUTPUT_BINARY_PUBLICATION_WHITELIST.json'),sha256=None)for z in PHASES],
  previous_index=dict(path=str(PUB/PRIOR),sha256=None),global_readme_before=dict(path=str(REPO/'readme.md'),sha256=None),portable_pages=sorted(MANDATORY_PORTABLE_PAGES),index_prefix_text=None,readme_prefix_text=None,
  output_directory=str(P/'publication_root_preparation/V213_PUBLICATION_STAGE_C'),automatic_retry=False,Git_add_commit_push=False)
 new(H/'ROOT_ACTUAL_ENTRY_FALSE_TEMPLATE.json',enc(template))
 new(H/'FINAL_READY.json',enc(dict(completed=True,PASS=True,FINAL_READY_FROZEN=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],false_entry=bind(H/'ROOT_ACTUAL_ENTRY_FALSE_TEMPLATE.json'),actual_source_repository_enumeration_hash_copy=0,actualbinary_header_or_decode=0,Git_calls=0)))
 print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),checks=len(q['checks']))))


def run(entry):
 e=json.loads(rb(entry));need(e['approved_actual_V212_V213_stage']is True and e['automatic_retry']is False and e['Git_add_commit_push']is False,'Root copy-stage authority only')
 p=checked(e['protocol']);need(e['code']==bind(__file__)==p['code']and e['protocol']==bind(PROTOCOL),'Frozen code/protocol');checked(p['checks']);need(e['root_scope']==ROOT_SCOPE,'Exact root scope')
 prospective=checked(ROOT_SCOPE);need(prospective['prepared_code_only']is True and prospective['approved_actual_enumeration_or_copy']is False,'Prep scope is not body authority');need(e['root_actual_scope']is not None,'Separate actual root scope');scope=checked(e['root_actual_scope']);need(scope['approved_bounded_publication_copy_stage']is True and scope['child_Git_mutations_authorized']is False,'Root actual stage/no Git')
 declared=scope['completed_source_phases'];need([z['path']for z in declared]==e['source_roots']==list(map(str,PHASES)),'Exact two roots')
 need(e['repository']==str(REPO)and e['initial_remote_main']==scope['expected_parent_main']==EXPECTED_PARENT,'Root declared parent main')
 roles={}
 for r in e['completion_receipts']:
  need(r['role']not in roles,'Unique role');root=role_phase(r['role']);need(root in safe(r['binding']['path']).parents,'Exact source role')
  q=checked(r['binding']);need(r['expected_fields'],'Exact completion expectations')
  for key,value in r['expected_fields'].items():need(q[key]==value,'Completion field '+key)
  roles[r['role']]=r
 need(set(roles)==set(REQUIRED_ROLES),'Exactly four result/freeze roles')
 for label,root,decl in zip(LABELS,PHASES,declared):
  result=roles[label+'_ROOT_PHASE_RESULT']['binding'];freeze=roles[label+'_ROOT_FINAL_FREEZE']['binding']
  need(result['path']==str(root/'ROOT_PHASE_RESULT.json')and freeze['path']==str(root/'ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json')and freeze['sha256']==decl['freeze_sha256'],'Root exact frozen phase')
  frozen_phase(checked(freeze),result)
 pages=set(e['portable_pages']);need(MANDATORY_PORTABLE_PAGES<=pages,'Two reports and newindex mandatory');need(all(page==INDEX or any(page.startswith(ARCHIVE_PREFIX+z.name+'/')and Path(page).suffix.lower()in('.md','.html')for z in PHASES)for page in pages),'Only owned portable pages');need(len(e['owned_output_whitelists'])==2,'Two owned whitelist bindings');docs=[]
 for root,b in zip(PHASES,e['owned_output_whitelists']):
  need(b['path']==str(root/'ROOT_OWNED_OUTPUT_BINARY_PUBLICATION_WHITELIST.json'),'Exact root-owned whitelist');docs.append(checked(b))
 owned=validate_owned_whitelists(docs)
 out=safe(e['output_directory']);need(out==P/'publication_root_preparation/V213_PUBLICATION_STAGE_C','Fresh separate admin stage');out.mkdir(exist_ok=False)
 new(out/'STARTED.json',enc(dict(completed=False,entry=bind(entry),automatic_retry=False)))
 try:
  need(e['previous_index']['path']==str(PUB/PRIOR)and e['global_readme_before']['path']==str(REPO/'readme.md'),'Exact prior index/README')
  oldindex=rb(e['previous_index']['path']);need(digest(oldindex)==e['previous_index']['sha256'],'Prior index unchanged')
  oldreadme=rb(e['global_readme_before']['path']);need(digest(oldreadme)==e['global_readme_before']['sha256'],'README before unchanged')
  inventory_by_root={};excluded={}
  for root in PHASES:inventory_by_root[root],excluded[root.name]=inventory(root,owned)
  pairs=[(root,source)for root in PHASES for source in inventory_by_root[root]]
  need(set(owned)<={str(source)for _,source in pairs},'Every owned whitelist binary encountered');need(pairs and len(pairs)<=MAX_ORIGINAL_FILES,'Bounded nonempty source count')
  original_bytes=sum(os.stat(source,follow_symlinks=False).st_size for _,source in pairs);need(original_bytes<=MAX_ORIGINAL_BYTES,'Bounded source bytes before copy')
  records=[];physical=[];texts={};source_hashes={}
  def store(rel,data,source,category):
   need(len(data)<LIMIT,'Each physical Gitfile<100MiB');need(not hits(data)if Path(rel).suffix.lower()in TEXT else True,'Credential pattern, values withheld')
   new(PUB/rel,data);need(rb(PUB/rel)==data,'Exact destination bytes')
   physical.append(dict(path=PREFIX+rel,sha256=digest(data),size_bytes=len(data)))
   records.append(dict(source_root_id='P'if category=='original_scientific_record'else'PUBLICATION_ADMIN_CODE'if category=='publication_tool'else'DELTA_DERIVED',source_absolute=str(source),source_relative_path=str(source.relative_to(P))if P in source.parents else rel,source_sha256=digest(data),size_bytes=len(data),published_relative=rel,published_sha256=digest(data),status='PUBLISHED_RAW',category=category,exclusion_reason=None,gzipchunks=None))
   if rel in pages:texts[rel]=data
  for root,source in pairs:
   data=rb(source);need(digest(data)==owned[str(source)]['sha256']and len(data)==owned[str(source)]['size_bytes']if str(source)in owned else True,'Root owned SHA/size exact');source_hashes[str(source)]=digest(data);store(original_destination(root,source),data,source,'original_scientific_record')
  for source in(Path(__file__).absolute(),PROTOCOL,H/'GENERATED_INVENTED_CHECKS.json'):store('tools/v213_lean_publication/'+source.name,rb(source),source,'publication_tool')
  ih,rh=publication_headers(e);store(INDEX,ih+oldindex,PUB/INDEX,'versioned_index');need(rb(PUB/INDEX).endswith(oldindex),'History index suffix')
  store(DELTA+'GLOBAL_README_BEFORE_V213.md',oldreadme,PUB/(DELTA+'GLOBAL_README_BEFORE_V213.md'),'old_global_snapshot')
  manifest=dict(format='immutable-completed-phase-delta-v1',phase='V212_V213_COMPLETED_SOURCE_FIT_AND_EXPOSED_REAL_BENCHMARK',source_roots=list(map(str,PHASES)),previous_index=e['previous_index'],root_entry=bind(entry),completion_receipts=e['completion_receipts'],entries=records,owned_binary_whitelist=list(owned.values()),owned_whitelist_bindings=e['owned_output_whitelists'],excluded_cache_paths=excluded,source_bindings_followed=0,old_archive_or_manifest_paths_modified=False,no_second_closing_required=True)
  mz=enc(manifest);new(PUB/(DELTA+'DELTA_MANIFEST.json'),mz);physical.append(dict(path=PREFIX+DELTA+'DELTA_MANIFEST.json',sha256=digest(mz),size_bytes=len(mz)))
  need(set(texts)==pages,'Every portable entry present');linkrows=[]
  for doc,data in texts.items():
   targets=links(doc,data)
   for target in targets:need(existing_target(PUB/target),'Missing portable link '+doc+' -> '+target)
   linkrows.append(dict(page=doc,local_links=len(targets)))
  # All final tables/galleries are checked through root-declared portable pages.
  rootlinks=links('ROOT_REPO_README.md',rh)
  for target in rootlinks:need(existing_target(REPO/target),'New README target missing')
  linkrows.append(dict(page='readme.md newprefix only; oldsuffix preserved',local_links=len(rootlinks)))
  for root in PHASES:
   pp,ee=inventory(root,owned);need(pp==inventory_by_root[root]and ee==excluded[root.name],'Final source inventory unchanged')
  newreadme=rh+oldreadme;need(newreadme.endswith(oldreadme),'README historical suffix');replace_readme(REPO/'readme.md',newreadme,e['global_readme_before']['sha256']);physical.append(dict(path='readme.md',sha256=digest(newreadme),size_bytes=len(newreadme)))
  for source,h in source_hashes.items():need(digest(rb(source))==h,'Original source final SHA unchanged')
  for rec in physical:need(digest(rb(REPO/rec['path']))==rec['sha256']and len(rb(REPO/rec['path']))==rec['size_bytes'],'Final physical SHA/size')
  need(len({r['path']for r in physical})==len(physical)and sum(r['size_bytes']for r in physical)<BATCH,'Single unique bounded batch')
  qa=dict(completed=True,PASS=True,entry=bind(entry),all2_scientific_phases_frozen=True,exact4_completion_roles=True,original_source_files=len(pairs),original_source_bytes=original_bytes,
   original_counts_by_phase={root.name:len(inventory_by_root[root])for root in PHASES},source_PNGs=sum(source.suffix.lower()=='.png'for _,source in pairs),
   all_original_to_destination_SHA_size_exact=True,source_original_final_SHA_unchanged=True,original_destination_identity_exact=True,
   all_declared_portable_local_links_resolve=True,link_pages=linkrows,local_links=sum(r['local_links']for r in linkrows),credential_format_detections=0,
   oldindex_and_README_suffix_preserved=True,source_bindings_followed=0,owned_models=2,new_prediction_NPZ=26,targets_inputs_NPZ=0,old_archive_mutations=0,
   model_array_image_decode=0,Git_calls=0,automatic_retry=False,ignored_ancestor_cache_paths_not_statted_read_hashed_or_modified=scope['known_ignored_cache_exclusions'])
  new(out/'QA.json',enc(qa))
  plan=dict(authorized_main_push=True,repository=str(REPO),remote=scope['remote'],branch='main',initial_remote_main=e['initial_remote_main'],manifest_sha256=digest(mz),batch_max_bytes=BATCH,
   batches=[dict(files=physical,message='Publish V212 paired point-weak source fit and V213 frozen exposed real benchmark')],scope='Exact completed V212/V213 records and root-owned2checkpoint26prediction packets, V213 index/README; no sourceinput or inheritedmodel publication, no modelselection/adoption/goal-success claim or extra closing cycle.')
  new(out/'ROOT_V213_PUSH_PLAN.json',enc(plan))
  complete=dict(completed=True,PASS=True,staged_only_not_pushed=True,entry=bind(entry),manifest=bind(PUB/(DELTA+'DELTA_MANIFEST.json')),QA=bind(out/'QA.json'),push_plan=bind(out/'ROOT_V213_PUSH_PLAN.json'),
   original_source_files=len(pairs),original_source_bytes=original_bytes,physical_files=len(physical),physical_bytes=sum(r['size_bytes']for r in physical),source_PNGs=qa['source_PNGs'],owned_models=2,new_prediction_NPZ=26,targets_inputs_NPZ=0,excluded_cache_paths=excluded,Git_calls=0,phase_writes=0)
  new(out/'COMPLETE.json',enc(complete));print(json.dumps(complete))
 except BaseException as error:
  new(out/'FAILURE_PRESERVED_NO_RETRY.json',enc(dict(completed=False,PASS=False,error_type=type(error).__name__,error=str(error),automatic_retry=False)));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);v=a.parse_args();prepare()if v.prepare else run(v.entry)
