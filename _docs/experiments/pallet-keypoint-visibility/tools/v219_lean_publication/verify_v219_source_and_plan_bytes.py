"""Independent root verification of closed publication list; opaque file bytes only."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, subprocess, argparse, os, stat, types
H=Path(__file__).absolute().parent
R=H.parent
P=R.parent

PROTO=Path(__file__).absolute().parent/'PROTOCOL_BEFORE_ACTUAL_V219_ROOT_TRANSACTIONS.json'
STAGER=R/'V219_PUBLICATION_PREPARATION_C/stage_completed_v219.py'
STAGER_SHA='dfbb719cf2f1de8443136bd6dc9096838ef655454728e16244fbb88269a873a4'
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
 need(e['stager']==dict(path=str(STAGER),sha256=STAGER_SHA),'Exact reviewed V219 stager')
 source=_admin_bytes(STAGER);need(hashlib.sha256(source).hexdigest()==STAGER_SHA,'Stager opaque SHA before execution')
 ns={'__file__':str(STAGER),'__name__':'reviewed_v219_stager_root_runtime'};exec(compile(source,str(STAGER),'exec'),ns)
 module=types.SimpleNamespace(**ns);need(module.PHASE.name=='query_visibility_expected_part_weak_paired_source_training_20261025_v219','Single exact scientific phase')
 return e,module
C=STAGER
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
PREFIX='_docs/experiments/pallet-keypoint-visibility/'
ROOTS=[P/'query_visibility_expected_part_weak_paired_source_training_20261025_v219']

def need(x,message):
 if not x:raise AssertionError(message)

def main():
 global m
 ap=argparse.ArgumentParser();ap.add_argument('--entry',type=Path,required=True);a=ap.parse_args();entry,m=_enter(a.entry,'byte_QA')
 stage=R/'V219_PUBLICATION_STAGE_C';complete=json.loads(m.rb(stage/'COMPLETE.json'));need(complete['completed']and complete['PASS'],'Stage actual complete')
 plan_binding=m.bind(stage/'ROOT_V219_PUSH_PLAN.json');need(entry['plan']==plan_binding,'Future root exact staged plan');plan=m.checked(plan_binding)
 need(plan['initial_remote_main']=='5b764d4d90479a1e5530d71575511d85d8809181','Exact V219 parent')
 need(complete['manifest']['path']==str(REPO/PREFIX/m.DELTA/'DELTA_MANIFEST.json'),'Exact own manifest before body');manifest=m.checked(complete['manifest']);need(complete['manifest']['sha256']==plan['manifest_sha256'],'Manifest exact plan')
 entry=m.checked(complete['entry']);need(entry['code']==m.bind(C),'Exact reviewed publisher')
 for rec in entry['completion_receipts']:
  need(any(str(root)==str(Path(rec['binding']['path']).parent)for root in ROOTS),'Exact phase role path')
  doc=m.checked(rec['binding'])
  for k,v in rec['expected_fields'].items():need(doc[k]==v,'Phase final fields')
 owned,_=m.manifest_models(manifest);need(manifest['owned_model_whitelist']==entry['owned_model_whitelist']and manifest['excluded_RAW_input_paths']==entry['excluded_RAW_input_paths'],'Entry/manifest RAW policy closure');sources=[]
 for rec in manifest['entries']:
  if rec['category']!='original_scientific_record':continue
  source=m.safe(rec['source_absolute']);root=next((r for r in ROOTS if r in source.parents),None);need(root is not None,'Exact original source parent')
  expected='archive/task-2/pallet-keypoint-visibility/'+root.name+'/'+str(source.relative_to(root));need(rec['published_relative']==expected,'Original destination relative identity')
  need(source.suffix.casefold()not in m.RAW or str(source)in owned,'Only owned new PT2 source RAW before opaque body');body=m.rb(source);target=m.rb(REPO/(PREFIX+expected));need(body==target and len(body)==rec['size_bytes']and m.digest(body)==rec['source_sha256']==rec['published_sha256'],'Every original source/destination SHA size byte identity')
  sources.append(rec)
 need(len(sources)==complete['original_source_files'],'Source file count')
 binary_sources={x['source_absolute']for x in sources if Path(x['source_absolute']).suffix.casefold()in m.RAW}
 need(binary_sources==set(owned),'Only new PT2, no input/target/prediction array source entries')
 files=plan['batches'][0]['files'];allowed=m.physical_models(files,manifest);need(len(files)==len({x['path']for x in files})==complete['physical_files'],'Full unique plan')
 for rec in files:
  name=rec['path'];need((name.startswith(PREFIX)or name=='readme.md')and not name.startswith('/')and '..'not in Path(name).parts and 'v112'not in name.casefold(),'Plan lexical guard before body')
  need(Path(name).suffix.casefold()not in m.RAW or name in allowed,'Only exact PT2 physical RAW before opaque body');data=m.rb(REPO/name);need(len(data)==rec['size_bytes']and m.digest(data)==rec['sha256'],'All full-plan destination bytes')
 def git(args):
  r=subprocess.run(['git',*args],cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.PIPE);need(r.returncode==0,'Read-only Git call');return r.stdout
 def names(args):return set(git(args).decode().split('\0'))-{''}
 changed=names(['diff','--name-only','-z','HEAD'])|names(['ls-files','--others','--exclude-standard','-z'])
 ignored=names(['ls-files','--others','--ignored','--exclude-standard','-z'])
 cache={PREFIX+'tools/__pycache__/check_publication_qa.cpython-313.pyc',PREFIX+'tools/__pycache__/verify_published_results.cpython-313.pyc'}
 need((changed|(ignored-cache))=={x['path']for x in files},'Whole Git delta exact plan with known two ignored exceptions')
 need(not git(['diff','--cached','--name-only','-z']),'Empty index before root transaction')
 need(git(['rev-parse','HEAD']).decode().strip()==plan['initial_remote_main'],'Local parent exact')
 need(git(['ls-remote','origin','refs/heads/main']).decode().split()[0]==plan['initial_remote_main'],'Remote parent exact')
 result=dict(completed=True,PASS=True,code=m.bind(__file__),plan=plan_binding,stage_complete=m.bind(stage/'COMPLETE.json'),manifest=complete['manifest'],original_source_files=len(sources),all_source_original_and_destination_byte_identity=True,full_plan_files=len(files),full_plan_size_bytes=sum(x['size_bytes']for x in files),all_plan_SHA256_and_size_verified=True,owned_new_checkpoints=2,owned_binary_whitelist=manifest['owned_binary_whitelist'],owned_new_prediction_NPZ=0,inputs_or_targets_NPZ=0,source_or_destination_model_array_decode=0,all2_phase_result_and_freeze_roles_exact=True,whole_Git_delta_exact_plan=True,initial_index_empty=True,expected_parent_main=plan['initial_remote_main'],known_ignored_cache_exclusions_not_opened=sorted(cache),selected_model=None,adopted=False,fully_solved=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
 target=R/'ROOT_V219_INDEPENDENT_SOURCE_AND_FULL_PLAN_BYTE_QA.json';m.new(target,m.enc(result));print(json.dumps(dict(completed=True,PASS=True,result=m.bind(target),files=len(files),originals=len(sources))))

if __name__=='__main__':main()
