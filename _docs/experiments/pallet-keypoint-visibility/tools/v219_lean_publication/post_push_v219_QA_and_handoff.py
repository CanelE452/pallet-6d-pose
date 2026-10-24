"""V219 closed-list post-push byte/HEAD QA and local handoff; actual entry required."""
import argparse
import ast
import concurrent.futures
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import types
import urllib.parse
import urllib.request

H=Path(__file__).absolute().parent;R=H.parent;P=R.parent
PHASE=P/'query_visibility_expected_part_weak_paired_source_training_20261025_v219'
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
PREFIX='_docs/experiments/pallet-keypoint-visibility/'
ARCHIVE=PREFIX+'archive/task-2/pallet-keypoint-visibility/'+PHASE.name+'/'
INDEX=PREFIX+'EXPERIMENT_INDEX_THROUGH_V219_KO.md'
PROTO=H/'PROTOCOL_BEFORE_ACTUAL_V219_ROOT_TRANSACTIONS.json'
OUT=H/'actual_post_push_v1'
HANDOFF=P/'ACTIVE_HANDOFF_KO.json'
INHERITED={'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V218_PUBLICATION_PREPARATION_B_V2/post_push_v218_QA_and_handoff.py', 'sha256': '019bcbf9fa533929f1cb102a1a5e6629538de5a2b9a111a1336fc8a684f69316'}
STAGER={'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V219_PUBLICATION_PREPARATION_C/stage_completed_v219.py', 'sha256': 'dfbb719cf2f1de8443136bd6dc9096838ef655454728e16244fbb88269a873a4'}
RAW={'.npz','.npy','.pt','.pth','.bin','.safetensors','.pkl','.pickle'}
HEAD_ROLES={'INDEX','ROOT_REPORT'}
REPORT_ROLES={'latest_source_report','latest_source_report_QA','latest_report','latest_phase_seal','root_source_review'}

def need(v,message):
 if not v:raise AssertionError(message)

def safe(path):
 s=str(path);need('v112'not in s.casefold(),'V112 lexical rejection before filesystem')
 p=Path(s);need(p.is_absolute()and'..'not in p.parts and(P in p.parents or REPO in p.parents or p==REPO),'Exact project/publication path')
 return p

def parentfd(path):
 p=safe(path);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  return p,fd
 except BaseException:os.close(fd);raise

def rb(path):
 p,fd=parentfd(path)
 try:n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Regular file, no symlink')
 with os.fdopen(n,'rb')as f:return f.read()

def new(path,data):
 p,fd=parentfd(path)
 try:n=os.open(p.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'wb')as f:f.write(data)

def mkdir_new(path):
 p,fd=parentfd(path)
 try:os.mkdir(p.name,dir_fd=fd)
 finally:os.close(fd)

def digest(data):return hashlib.sha256(data).hexdigest()
def enc(value):return(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def bind(path):return dict(path=str(safe(path)),sha256=digest(rb(path)))

def identity(binding):
 need(isinstance(binding,dict)and set(binding)in({'path','sha256'},{'path','sha256','size_bytes'}),'Only path/SHA and optional size binding')
 need(type(binding['path'])is str and type(binding['sha256'])is str and re.fullmatch('[0-9a-f]{64}',binding['sha256']),'Binding path/SHA types')
 safe(binding['path'])
 if 'size_bytes'in binding:need(type(binding['size_bytes'])is int and binding['size_bytes']>=0,'Nonnegative integer size, not bool')
 return(binding['path'],binding['sha256'])

def checked_bytes(binding):
 identity(binding);data=rb(binding['path']);need(digest(data)==binding['sha256'],'Exact SHA binding')
 if 'size_bytes'in binding:need(len(data)==binding['size_bytes'],'Exact recorded byte size')
 return data

def checked(binding):return json.loads(checked_bytes(binding))
def same(a,b):return identity(a)==identity(b)

def admin_json(binding):
 identity(binding);p=safe(binding['path']);need(R in p.parents and p.suffix=='.json','Exact root-admin JSON binding before body read')
 return checked(binding)

def entry_gate(e):
 need(e.get('approved_actual_post_push_QA_and_handoff')is True and e.get('root_verified_push_completed')is True,'Separate root post-push authority')
 need(e.get('automatic_retry')is False and e.get('training_execution_by_post_QA')is False and e.get('model_array_image_GT_score_decode')is False,'No retries, training or binary interpretation')
 need(e.get('output_directory')==str(OUT)and e.get('handoff_path')==str(HANDOFF),'Exact fresh output and sole mutable handoff path')
 need(type(e.get('expected_commit'))is str and re.fullmatch('[0-9a-f]{40}',e['expected_commit']),'Exact verified pushed main commit')

def plan_records(plan,allowed):
 need(len(plan['batches'])==1,'Exactly one closed payload batch')
 files=plan['batches'][0]['files'];need(files and sum(r['size_bytes']for r in files)<480*1024**2,'Nonempty batch below480MiB')
 out={}
 for rec in files:
  need(set(rec)=={'path','sha256','size_bytes'},'Strict physical path/SHA/size schema')
  name=rec['path'];need(type(name)is str and(name.startswith(PREFIX)or name=='readme.md')and'v112'not in name.casefold()and not name.startswith('/')and'..'not in Path(name).parts,'Closed plan lexical path')
  need((Path(name).suffix.casefold()not in RAW or name in allowed)and re.fullmatch('[0-9a-f]{64}',rec['sha256'])and type(rec['size_bytes'])is int and 0<=rec['size_bytes']<100*1024**2,'Zero RAW, valid physical SHA/size')
  need(name not in out,'No duplicated physical plan path');out[name]=rec
 return out



def head_targets(targets,physical):
 need(isinstance(targets,list)and len(targets)==6 and len({r['role']for r in targets})==6 and HEAD_ROLES<={r['role']for r in targets},'Six explicit future named HEAD targets')
 need(all(set(r)=={'role','path'}for r in targets)and len({r['path']for r in targets})==6,'Unique exact target paths')
 for rec in targets:
  path=rec['path'];need(path in physical and Path(path).suffix.casefold()not in RAW,'HEAD target in exact plan')
  if rec['role']=='INDEX':need(path==INDEX,'Exact V219 index')
  elif rec['role']=='ROOT_REPORT':need(path==ARCHIVE+'PORTABLE_REPORT_KO.md','Exact V219 portable report')
  else:need(path.startswith(ARCHIVE)and Path(path).suffix in('.png','.csv','.md','.json','.html','.txt'),'Only own exact future portable artifact')
 return targets

def research_summary(value):
 fields={'status','source_updates_each','source_total_updates','parameters_each','weak_known_rows','weak_FIT_rows','weak_DEV_rows','diagnostic_results','root_source_status','training_admission_this_phase'}
 need(isinstance(value,dict)and set(value)==fields,'Explicit actual SOURCE training summary')
 need(type(value['source_updates_each'])is int and value['source_updates_each']>0 and value['source_total_updates']==2*value['source_updates_each']and value['parameters_each']==118359,'Actual paired source budget/capacity')
 need(value['weak_known_rows']==57 and value['weak_FIT_rows']==45 and value['weak_DEV_rows']==12 and value['training_admission_this_phase']is True and isinstance(value['diagnostic_results'],dict),'Actual admitted57 weak SOURCE diagnostic, not native gold')
 return copy.deepcopy(value)

def handoff_value(old,e,publication,utc):
 value=research_summary(e['research_summary']);reports=e['report_bindings']
 need(set(reports)==REPORT_ROLES,'Explicit final report and semantic pointers')
 for binding in reports.values():identity(binding)
 real={key:copy.deepcopy(old[key])for key in('latest_frozen_real_benchmark','latest_current_evaluation')}
 need(all('v213'in json.dumps(v,ensure_ascii=False).casefold()for v in real.values()),'Latest real benchmark remains V213')
 h=copy.deepcopy(old);h['updated_UTC']=utc;h['goal_status']='active';h['fully_solved']=False
 h['previous_publication_current']=h.get('publication_current');h['publication_current']=publication
 value.update(phase='V219',training_updates_this_phase=value['source_total_updates'],source_model_execution_occurred=True,protected_evaluation_reads_this_phase=0,public_training_query8_is_GT_derived=True,no_additional_GT_pose_reference_ROI_mask_deployment_channels=True,all360_candidates_automatically_admitted=False,native_point_gold=False,new_owned_checkpoints_published=2,root_source_review=reports['root_source_review'],original_role_TRAIN_preserved=True,DEV_excluded_from_all_loss_pool_normalization=True,fully_solved=False)
 h['research_current']=value
 for key in('latest_source_report','latest_source_report_QA','latest_report','latest_phase_seal'):h[key]=reports[key]
 h['latest_index']=publication['index_url']
 need(all(h[key]==v for key,v in real.items()),'Exact original latest real evaluation fields preserved')
 return h

def replace_handoff(data,expected):
 need(same(bind(HANDOFF),expected),'Canonical handoff unchanged since root entry')
 tmp=HANDOFF.with_name('ACTIVE_HANDOFF_KO.json.v219_post_push_tmp');new(tmp,data)
 src,fd=parentfd(tmp);dst,other=parentfd(HANDOFF)
 try:os.replace(src.name,dst.name,src_dir_fd=fd,dst_dir_fd=other)
 finally:os.close(fd);os.close(other)

def load_stager():
 source=checked_bytes(STAGER);ns={'__file__':STAGER['path'],'__name__':'v219_frozen_stager_postpush_only'}
 exec(compile(source,STAGER['path'],'exec'),ns)
 return types.SimpleNamespace(**ns)

def run(entry_path):
 e=json.loads(rb(entry_path));entry_gate(e)
 need(same(e['code'],bind(__file__))and same(e['protocol'],bind(PROTO))and same(e['stager'],STAGER),'Frozen helper/protocol/stager closure')
 scope=admin_json(e['root_actual_scope']);need(scope.get('approved_actual_post_push_QA_and_handoff')is True,'Separate prospective root actual scope')
 checked_bytes(INHERITED);m=load_stager();need(m.PHASE==PHASE and m.REPO==REPO and m.PREFIX==PREFIX,'Single V219 inherited byte kernel')
 mkdir_new(OUT);new(OUT/'STARTED.json',enc(dict(completed=False,entry=bind(entry_path),automatic_retry=False)))
 try:
  state=admin_json(e['root_transaction']);need(state.get('completed')is True and state.get('PASS')is True and state['commit']==e['expected_commit'],'Completed exact root push transaction')
  done=admin_json(e['stage_complete']);need(done['completed']is True and done['PASS']is True and done['owned_models']==2 and done['new_prediction_NPZ']==done['targets_inputs_NPZ']==0,'Completed exact PT2 / zero input or prediction NPZ V219 stage')
  need(same(done['push_plan'],e['push_plan'])and same(done['manifest'],e['manifest']),'Actual stage plan/manifest exact graph')
  need(e['manifest']['path']==str(REPO/PREFIX/m.DELTA/'DELTA_MANIFEST.json'),'Exact V219 published delta manifest before body read')
  plan=admin_json(e['push_plan']);manifest=checked(e['manifest']);stageqa=admin_json(done['QA'])
  need(stageqa['PASS']is True and plan['authorized_main_push']is True and plan['repository']==str(REPO)and plan['remote']=='https://github.com/CanelE452/pallet-6d-pose.git'and plan['branch']=='main','Exact stage QA and authorized publication repository')
  need(plan['initial_remote_main']==e['expected_parent']==m.EXPECTED_PARENT and plan['manifest_sha256']==e['manifest']['sha256'],'Exact staged parent and manifest SHA')
  allowed=m.physical_models(plan['batches'][0]['files'],manifest);owned,_=m.manifest_models(manifest);physical=plan_records(plan,allowed);need(len(physical)==done['physical_files']and sum(r['size_bytes']for r in physical.values())==done['physical_bytes'],'Exact staged physical count/bytes')
  need(manifest['source_roots']==[str(PHASE)]and len(manifest['owned_binary_whitelist'])==2 and manifest['source_bindings_followed']==0,'Only frozen V219 originals, no source following')
  originals=[r for r in manifest['entries']if r['category']=='original_scientific_record'];need(len(originals)==done['original_source_files']and len({r['source_absolute']for r in originals})==len(originals),'Exact original closed count, no scan')
  original_map={}
  for rec in originals:
   source=safe(rec['source_absolute']);need(PHASE in source.parents and(source.suffix.casefold()not in RAW or str(source)in owned),'Owned non-RAW source before read')
   data=rb(source);need(digest(data)==rec['source_sha256']and len(data)==rec['size_bytes'],'Frozen original source SHA/size after push')
   published=PREFIX+rec['published_relative'];need(published in physical and physical[published]['sha256']==rec['source_sha256']and physical[published]['size_bytes']==rec['size_bytes'],'Original-to-physical exact byte closure')
   original_map[str(source)]=dict(path=str(source),sha256=rec['source_sha256'],size_bytes=rec['size_bytes'])
  for rec in physical.values():
   data=rb(REPO/rec['path']);need(digest(data)==rec['sha256']and len(data)==rec['size_bytes'],'Every physical plan byte SHA/size')
  for binding in e['report_bindings'].values():
   need(binding['path']in original_map and same(binding,original_map[binding['path']]),'Final report/semantic pointer covered by original manifest');checked_bytes(binding)
  head_targets(e['HTTP_HEAD_targets'],physical);research_summary(e['research_summary'])
  # No Git command mutates the repository. Validate the ordinary cwd before subprocess use.
  p,fd=parentfd(REPO/'readme.md');os.close(fd)
  def git(args):
   r=subprocess.run(['git',*args],cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   need(r.returncode==0,'Read-only Git QA');return r.stdout
  commit=e['expected_commit'];need(git(['rev-parse','HEAD']).decode().strip()==commit,'Exact HEAD')
  remote=git(['ls-remote','origin','refs/heads/main']).decode().split();need(len(remote)==2 and remote[0]==commit,'Exact remote main')
  need(not git(['status','--porcelain','--untracked-files=all'])and not git(['diff','--cached','--name-only','-z']),'Clean working tree and empty index')
  def head(rec):
   url='https://github.com/CanelE452/pallet-6d-pose/blob/'+commit+'/'+urllib.parse.quote(rec['path'],safe='/')
   request=urllib.request.Request(url,method='HEAD',headers={'User-Agent':'Codex-V219-publication-QA'})
   with urllib.request.urlopen(request,timeout=30)as response:status=response.status
   need(status==200,'Immutable portable HTTP HEAD200')
   return dict(role=rec['role'],path=rec['path'],url=url,HTTP_HEAD_status=status,response_content_read=False)
  with concurrent.futures.ThreadPoolExecutor(max_workers=6)as pool:http=list(pool.map(head,e['HTTP_HEAD_targets']))
  utc=datetime.datetime.now(datetime.timezone.utc).isoformat();index=next(r['url']for r in http if r['role']=='INDEX')
  need(e['handoff_before']['path']==str(HANDOFF),'Exact canonical handoff before any body read');oldbytes=checked_bytes(e['handoff_before']);old=json.loads(oldbytes)
  qa=dict(completed=True,PASS=True,commit=commit,entry=bind(entry_path),code=bind(__file__),protocol=bind(PROTO),source_original_files_rehashed=len(originals),source_original_SHA_unchanged=True,physical_files_rehashed=len(physical),physical_destination_SHA_unchanged=True,remote_main_and_HEAD_exact=True,working_tree_clean=True,index_empty=True,portable_HTTP_HEAD=http,source_bindings_followed=0,model_array_image_GT_score_decode=0,scientific_phase_writes=0,Git_writes=0,goal_status='active',fully_solved=False,UTC=utc)
  qp=OUT/'POST_PUSH_ORIGINALS_AND_PORTABLE_HTTP_QA.json';new(qp,enc(qa))
  receipt=dict(completed=True,PASS=True,commit=commit,remote='https://github.com/CanelE452/pallet-6d-pose.git',branch='main',index_url=index,publication_stage=e['stage_complete'],root_transaction=e['root_transaction'],independent_byte_QA=state['independent_byte_QA'],post_push_QA=bind(qp),push_plan=e['push_plan'],original_scientific_files=len(originals),physical_git_files=len(physical),physical_git_bytes=sum(r['size_bytes']for r in physical.values()),PNG_files=done['source_PNGs'],new_checkpoints=2,new_prediction_NPZ=0,scientific_phase_freezes=manifest['completion_receipts'],goal_status='active',fully_solved=False,no_second_closing_commit_required=True,UTC=utc)
  rp=OUT/'FINAL_MAIN_PUBLICATION_LOCAL_RECEIPT.json';new(rp,enc(receipt))
  publication=dict(status='MAIN_PUSHED_VERIFIED',phases=['V219'],commit=commit,index_url=index,receipt=bind(rp),post_push_QA=bind(qp))
  updated=handoff_value(old,e,publication,utc);new(OUT/'ACTIVE_HANDOFF_BEFORE_V219_PUBLICATION.json',oldbytes)
  replace_handoff(enc(updated),e['handoff_before'])
  new(OUT/'ACTIVE_HANDOFF_UPDATE_RECEIPT.json',enc(dict(completed=True,PASS=True,handoff=bind(HANDOFF),publication=bind(rp),latest_frozen_real_benchmark_unchanged=True,latest_current_evaluation_unchanged=True,goal_status='active',fully_solved=False,UTC=utc)))
  new(OUT/'COMPLETE.json',enc(dict(completed=True,PASS=True,entry=bind(entry_path),post_push_QA=bind(qp),receipt=bind(rp),handoff_update=bind(OUT/'ACTIVE_HANDOFF_UPDATE_RECEIPT.json'),automatic_retry=False)))
  print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'),receipt=bind(rp),post_push_QA=bind(qp),commit=commit)))
 except BaseException as error:
  new(OUT/'FAILURE_PRESERVED_NO_AUTOMATIC_RETRY.json',enc(dict(completed=False,PASS=False,error_type=type(error).__name__,error=str(error),automatic_retry=False)));raise



if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--entry',required=True);args=parser.parse_args();run(args.entry)
