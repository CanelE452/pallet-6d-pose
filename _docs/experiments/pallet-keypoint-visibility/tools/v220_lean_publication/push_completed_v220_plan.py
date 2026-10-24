"""Execute one root-verified, already authorized publication plan. No discovery/force."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, subprocess, datetime, traceback, os, stat, types

H = Path(__file__).absolute().parent
R = H.parent
P = R.parent
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
PREFIX = '_docs/experiments/pallet-keypoint-visibility/'
IGNORED = {
 PREFIX+'tools/__pycache__/check_publication_qa.cpython-313.pyc',
 PREFIX+'tools/__pycache__/verify_published_results.cpython-313.pyc',
}

PROTO=Path(__file__).absolute().parent/'PROTOCOL_BEFORE_ACTUAL_V220_ROOT_TRANSACTIONS.json'
STAGER=R/'V220_PUBLICATION_PREPARATION_C/stage_completed_v220.py'
STAGER_SHA='82eb2b65ae8f04f6b1b26c5677493aa681fa3ca5934a98decca616c7c57d6bb7'
m=None
def _admin_bytes(path):
 p=Path(path);need(p.is_absolute()and'..'not in p.parts and'v112'not in str(p).casefold()and R in p.parents,'Root admin path before FS')
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Regular admin file/no symlink')
 with os.fdopen(n,'rb')as f:return f.read()
def _binding(path):return dict(path=str(path),sha256=hashlib.sha256(_admin_bytes(path)).hexdigest())
def _checked(binding):
 need(set(binding)=={'path','sha256'},'Strict publication two-key binding')
 data=_admin_bytes(binding['path']);need(hashlib.sha256(data).hexdigest()==binding['sha256'],'Exact admin SHA');return json.loads(data)
def _enter(path,operation):
 e=json.loads(_admin_bytes(path));need(e.get('approved_actual_operation')==operation and e.get('automatic_retry')is False,'Explicit future root actual operation')
 need(e['code']==_binding(__file__)and e['protocol']==_binding(PROTO),'Exact helper/protocol closure')
 need(e['stager']==dict(path=str(STAGER),sha256=STAGER_SHA),'Exact reviewed V220 stager')
 source=_admin_bytes(STAGER);need(hashlib.sha256(source).hexdigest()==STAGER_SHA,'Stager opaque SHA before execution')
 ns={'__file__':str(STAGER),'__name__':'reviewed_v220_stager_root_runtime'};exec(compile(source,str(STAGER),'exec'),ns)
 module=types.SimpleNamespace(**ns);need(module.PHASE.name=='query_visibility_expected_part_weak_frozen_real_benchmark_20261025_v220','Single exact scientific phase')
 return e,module

def need(ok, message):
 if not ok: raise AssertionError(message)

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def git(args, payload=None):
 r = subprocess.run(['git', *args], cwd=REPO, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
 need(r.returncode == 0, 'Git failed: '+str(args[:4])+' '+r.stderr.decode(errors='replace')[-3000:])
 return r.stdout

def names(args): return set(git(args).decode().split('\0'))-{''}

def modeless_path(s):
 need(type(s) is str and (s.startswith(PREFIX) or s == 'readme.md'), 'Publication prefix')
 need(not s.startswith('/') and '..' not in Path(s).parts and 'v112' not in s.casefold(), 'Lexical path guard before filesystem')
 need(Path(s).suffix.casefold()not in m.RAW,'Zero RAW before planned file read/Git add')
 return m.safe(REPO/s)

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

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--entry',type=Path,required=True);a=ap.parse_args()
 global m
 entry,m=_enter(a.entry,'push');plan=m.checked(entry['plan']);qa=m.checked(entry['independent_byte_QA'])
 need(entry['approved_single_nonforce_main_push'] is True and qa['completed'] is True and qa['PASS'] is True, 'Root actual verified push authority')
 need(plan['authorized_main_push'] is True and plan['repository']==str(REPO) and plan['branch']=='main' and len(plan['batches'])==1, 'One exact plan')
 need(plan['remote']=='https://github.com/CanelE452/pallet-6d-pose.git' and plan['initial_remote_main']=='7b967ec82475562dd1b1d7e82d0ff842bfa549f7', 'Exact repository and parent')
 need(qa['plan']==entry['plan'], 'Independent QA exact plan')
 out=R/'V220_MAIN_PUSH_ROOT_TRANSACTION';out.mkdir(exist_ok=False)
 state=dict(completed=False,entry=m.bind(a.entry),code=m.bind(__file__),plan=entry['plan'],independent_byte_QA=entry['independent_byte_QA'],stage='STARTED',UTC=now())
 def save():
  target=out/'STATE.json';temporary=out/'STATE.json.tmp';temporary.write_bytes(m.enc(state));temporary.replace(target)
 save()
 try:
  need(entry['manifest']['path']==str(REPO/PREFIX/m.DELTA/'DELTA_MANIFEST.json'),'Exact own manifest before body');manifest=m.checked(entry['manifest']);need(entry['manifest']['sha256']==plan['manifest_sha256'],'Exact manifest bound to plan');m.zero_raw_manifest(manifest);files=plan['batches'][0]['files'];planned={x['path']:x for x in files}
  need(len(files)==len(planned) and sum(x['size_bytes']for x in files)<plan['batch_max_bytes'], 'Unique bounded batch')
  for name,rec in planned.items():
   data=m.rb(modeless_path(name));need(len(data)==rec['size_bytes'] and m.digest(data)==rec['sha256'], 'Root pre-add planned byte identity')
  need(git(['remote','get-url','origin']).decode().strip()==plan['remote'], 'Remote URL')
  parent=plan['initial_remote_main'];need(git(['rev-parse','HEAD']).decode().strip()==parent, 'Expected local parent')
  need(git(['ls-remote','origin','refs/heads/main']).decode().split()[0]==parent, 'Expected remote parent')
  need(not git(['diff','--cached','--name-only','-z']), 'Initially empty index')
  changed=names(['diff','--name-only','-z','HEAD'])|names(['ls-files','--others','--exclude-standard','-z'])
  ignored=names(['ls-files','--others','--ignored','--exclude-standard','-z'])
  need((changed | (ignored-IGNORED))==set(planned), 'Whole working delta exactly plan with two known ignored exceptions')
  pathspec=out/'EXACT_PLAN.paths';pathspec.write_bytes(b''.join(s.encode()+b'\0'for s in planned))
  git(['-c','core.autocrlf=false','add','--force','--pathspec-from-file='+str(pathspec),'--pathspec-file-nul'])
  need(names(['diff','--cached','--name-only','-z'])==set(planned), 'Staged names exactly plan')
  index=tree_map(git(['ls-files','--stage','-z','--',*planned]),True)
  need(set(index)==set(planned),'Index path set');state['verified_index_blobs']=verify_blobs(index,planned);state['stage']='INDEX_ALL_SHA256_VERIFIED';save()
  message=out/'COMMIT_MESSAGE.txt';message.write_text(plan['batches'][0]['message']+'\n')
  git(['-c','gc.auto=0','commit','--quiet','--file='+str(message)])
  commit=git(['rev-parse','HEAD']).decode().strip();state['commit']=commit;state['stage']='COMMITTED_PENDING_VERIFICATION';save()
  need(git(['rev-parse','HEAD^']).decode().strip()==parent,'Single commit parent')
  need(names(['diff-tree','--no-commit-id','--name-only','-r','-z','HEAD'])==set(planned),'Committed names exactly plan')
  tree=tree_map(git(['ls-tree','-r','-z','HEAD','--',*planned]),False)
  need(tree==index,'Committed and staged blob mappings exact');state['verified_committed_blobs']=verify_blobs(tree,planned)
  need(not git(['status','--porcelain','--untracked-files=all']),'Clean working tree after commit')
  state['stage']='ALL_COMMITTED_BYTES_VERIFIED_BEFORE_NONFORCE_PUSH';save()
  print(json.dumps(dict(event='PUSH_START',commit=commit,files=len(planned))),flush=True)
  git(['-c','pack.window=0','-c','pack.compression=1','-c','pack.threads=2','push','origin','HEAD:refs/heads/main'])
  need(git(['ls-remote','origin','refs/heads/main']).decode().split()[0]==commit,'Remote main exact commit')
  need(git(['rev-parse','HEAD']).decode().strip()==commit and not git(['status','--porcelain','--untracked-files=all']),'Final local HEAD and clean tree')
  state.update(completed=True,PASS=True,stage='REMOTE_MAIN_COMMIT_AND_CLEAN_VERIFIED',nonforce_push=True,physical_files=len(planned),UTC=now());save()
  print(json.dumps(state),flush=True)
 except BaseException:
  state.update(stage='FAILED_NO_AUTOMATIC_RETRY',traceback=traceback.format_exc(),UTC=now());save();raise

if __name__=='__main__':main()
