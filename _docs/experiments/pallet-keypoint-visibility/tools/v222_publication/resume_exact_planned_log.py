"""Manually resume V222 after one ignored declared .log, preserving original failure/index."""
import argparse,importlib.util,hashlib,json,concurrent.futures,urllib.request
from pathlib import Path
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V222_PUBLICATION_PREPARATION_C';O=Path(__file__).absolute().parent
original=H/'publish_completed_v222.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='c026896d5271525eaa5860cb4118ab46fea5d5fe9e67649199297f12d230c726'
spec=importlib.util.spec_from_file_location('v222_original_preserved_publisher',original);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
LOG=m.PREFIX+m.ARCHIVE+'provenance/V222_ACTUAL_MODEL_QA_A/ACTUAL_EXECUTION_STDOUT.log'
CODE=m.PREFIX+'tools/v222_publication/resume_exact_planned_log.py'
NOTE=m.PREFIX+m.DELTA+'V222_PUBLICATION_RECOVERY.json'
def b(path):return m.bind(path)
def j(path):return json.loads(m.rb(path))
def newj(path,obj):m.new(path,m.enc(obj))
def baseline():
 e=m.entry_gate(R/'V222_PUBLICATION_ROOT_OPERATIONS/ROOT_ACTUAL_PUSH_ENTRY.json','push')
 done,plan,manifest,owned,planned=m.closed_plan(e)
 m.exact_bytes(done,manifest,owned,planned)
 return e,done,plan,manifest,owned,planned
def repo_before():
 m.need(m.git(['remote','get-url','origin']).decode().strip()==m.REMOTE,'Exact public remote')
 m.need(m.git(['rev-parse','HEAD']).decode().strip()==m.EXPECTED_PARENT==m.git(['ls-remote','origin','refs/heads/main']).decode().split()[0],'No commit or remote push happened yet')
def verify_all(planned,owned):
 for name,rec in planned.items():
  data=m.rb(m.physical_path(name,owned));m.need(m.digest(data)==rec['sha256'] and len(data)==rec['size_bytes'],'All baseline/extension exact bytes')
def push():
 e,done,plan,manifest,owned,planned=baseline();repo_before();m.working_delta(planned)
 failure=j(R/'V222_PUSH_FAILURE_NO_RETRY.json')
 m.need(failure['PASS'] is False and failure['automatic_retry'] is False and LOG in failure['error'],'Exact original ignored-log failure preserved')
 staged=m.names(['diff','--cached','--name-only','-z']);expect=set(planned)-{x for x in planned if Path(x).suffix in {'.png','.pt'}}-{LOG}
 m.need(staged==expect and len(staged)==185,'Exactly existing 185 normal index entries; do not reset or restage failed operation')
 oldindex=m.tree_map(m.git(['ls-files','--stage','-z','--',*staged]),True);m.verify_blobs(oldindex,planned)
 ignored=m.names(['ls-files','--others','--ignored','--exclude-standard','-z'])-m.IGNORED
 m.need(ignored=={x for x in planned if Path(x).suffix in {'.png','.pt'}}|{LOG},'Only declared PNG/PT plus one exact log ignored')
 m.need('.gitignore:270:*.log' in m.git(['check-ignore','-v','--',LOG]).decode(),'Actual log ignore rule')
 recovery={'completed':True,'kind':'MANUAL_CONTINUATION_AFTER_ONE_DECLARED_LOG_IGNORED',
  'original_failure':failure,'original_stage_complete':e['stage_complete'],'original_plan':e['push_plan'],'original_byte_QA':e['byte_QA'],
  'unchanged_scientific_source_files':done['original_source_files'],'unchanged_baseline_files':len(planned),
  'exact_ignored_log':LOG,'actual_ignore_rule':'.gitignore:270:*.log','retained_staged_normal_files':len(staged),
  'correction':'Keep original index and failure. Force-add only already planned PNG/new2PT leaves and this one declared log. Add this recovery record and exact recovery code. No generic log/cache force-add.',
  'new_admin_files':[CODE,NOTE],'code':b(__file__),'automatic_retry':False,'original_publisher_rerun':False,
  'scientific_phase_writes':0,'models_or_predictions_changed':False,'protected_reads':0,'goal_solved':False}
 m.new(m.REPO/CODE,m.rb(__file__));newj(m.REPO/NOTE,recovery)
 for name in [CODE,NOTE]:
  data=m.rb(m.REPO/name);planned[name]={'path':name,'sha256':m.digest(data),'size_bytes':len(data)}
 newj(O/'RECOVERY_EXACT_605_FILE_PLAN.json',{'completed':True,'baseline':e['push_plan'],'baseline_manifest':e['manifest'],
  'files':list(planned.values()),'baseline_count':done['physical_files'],'extension_count':2,'total_count':len(planned),
  'total_bytes':sum(x['size_bytes'] for x in planned.values()),'original_source_files':done['original_source_files']})
 verify_all(planned,owned);m.working_delta(planned)
 m.git(['-c','core.autocrlf=false','add','--',CODE,NOTE])
 force=sorted({x for x in planned if Path(x).suffix in {'.png','.pt'}}|{LOG})
 pathspec=O/'EXACT_DECLARED_PNG_PT_AND_ONE_LOG.paths';m.new(pathspec,b''.join(x.encode()+b'\0' for x in force))
 m.git(['-c','core.autocrlf=false','add','--force','--pathspec-from-file='+str(pathspec),'--pathspec-file-nul'])
 m.git(['-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'])
 m.need(m.names(['diff','--cached','--name-only','-z'])==set(planned),'Exact605 index names')
 index=m.tree_map(m.git(['ls-files','--stage','-z','--',*planned]),True);m.need(set(index)==set(planned),'Exact605 indexed blobs');m.verify_blobs(index,planned)
 newj(O/'INDEX_AND_FULL_BYTES_VERIFIED.json',{'completed':True,'PASS':True,'plan':b(O/'RECOVERY_EXACT_605_FILE_PLAN.json'),'index_files':len(index),'source_unchanged':True,'forced_exact_declared_files':len(force)})
 message=O/'COMMIT_MESSAGE.txt';m.new(message,(plan['batches'][0]['message']+'\n').encode());m.git(['-c','gc.auto=0','commit','--quiet','--file='+str(message)])
 commit=m.git(['rev-parse','HEAD']).decode().strip()
 newj(O/'COMMITTED_PENDING_VERIFICATION.json',{'completed':False,'commit':commit,'plan':b(O/'RECOVERY_EXACT_605_FILE_PLAN.json')})
 m.need(m.git(['rev-parse','HEAD^']).decode().strip()==m.EXPECTED_PARENT and m.names(['diff-tree','--no-commit-id','--name-only','-r','-z','HEAD'])==set(planned),'Exact single605 commit/parent')
 tree=m.tree_map(m.git(['ls-tree','-r','-z','HEAD','--',*planned]),False);m.need(tree==index,'Exact committed index map');m.verify_blobs(tree,planned)
 m.need(not m.git(['status','--porcelain','--untracked-files=all']) and not m.git(['diff','--cached','--name-only','-z']),'Clean committed tree/index')
 print(json.dumps({'event':'PUSH_START','commit':commit,'files':len(planned)}),flush=True)
 m.git(['-c','pack.window=0','-c','pack.compression=1','-c','pack.threads=2','push','origin','HEAD:refs/heads/main'])
 m.need(m.git(['ls-remote','origin','refs/heads/main']).decode().split()[0]==commit==m.git(['rev-parse','HEAD']).decode().strip(),'HEAD remote exact')
 result={'completed':True,'PASS':True,'commit':commit,'plan':b(O/'RECOVERY_EXACT_605_FILE_PLAN.json'),
  'original_plan':e['push_plan'],'original_failure':b(R/'V222_PUSH_FAILURE_NO_RETRY.json'),'physical_files':len(planned),
  'physical_bytes':sum(x['size_bytes'] for x in planned.values()),'original_source_files':done['original_source_files'],
  'owned_new_checkpoints':2,'RAW_arrays_published':0,'nonforce':True,'automatic_retry':False,'original_publisher_rerun':False}
 newj(O/'COMPLETE.json',result);return result
def post():
 e,done,plan,manifest,owned,baseplanned=baseline();tx=j(O/'COMPLETE.json');rp=j(O/'RECOVERY_EXACT_605_FILE_PLAN.json');planned={x['path']:x for x in rp['files']}
 m.need(tx['PASS'] is True and len(planned)==len(baseplanned)+2==605 and set(planned)-set(baseplanned)=={CODE,NOTE},'Closed605 plan')
 for name,row in baseplanned.items():m.need(planned[name]==row,'Original603 plan unchanged')
 verify_all(planned,owned);commit=tx['commit']
 m.need(m.git(['rev-parse','HEAD']).decode().strip()==commit==m.git(['ls-remote','origin','refs/heads/main']).decode().split()[0],'HEAD remote exact')
 m.need(not m.git(['status','--porcelain','--untracked-files=all']) and not m.git(['diff','--cached','--name-only','-z']),'Post clean tree/index')
 tree=m.tree_map(m.git(['ls-tree','-r','-z','HEAD','--',*planned]),False);m.need(set(tree)==set(planned),'All605 committed names');m.verify_blobs(tree,planned)
 targetpaths=[('INDEX',m.PREFIX+m.INDEX),('ROOT_REPORT',m.PREFIX+m.ARCHIVE+'PORTABLE_REPORT_KO.md'),
  ('COUNTS',m.PREFIX+m.ARCHIVE+'saved_source_comparison_v2/SOURCE_FAMILIES_SEPARATE_CM_PRECISION_RECALL.csv'),
  ('SUCCESS_CROP',m.PREFIX+m.ARCHIVE+'saved_source_comparison_v2/SOURCE_DEV_F05_C000000_S0_N4.png'),
  ('FAILURE_CROP',m.PREFIX+m.ARCHIVE+'saved_source_comparison_v2/SOURCE_DEV_F05_C000000_S0_N1.png'),('PUBLICATION_RECOVERY',NOTE)]
 def head(item):
  role,name=item;m.need(name in planned and Path(name).suffix not in m.RAW,'Exact planned nonRAW HEAD')
  url='https://github.com/CanelE452/pallet-6d-pose/blob/'+commit+'/'+m.quote(name,safe='/')
  req=urllib.request.Request(url,method='HEAD',headers={'User-Agent':'Codex-V222-recovery-QA'})
  with urllib.request.urlopen(req,timeout=30) as res:status=res.status
  m.need(status==200,'HTTP HEAD200');return {'role':role,'path':name,'url':url,'status':status,'body_read':False}
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:http=list(pool.map(head,targetpaths))
 result={'completed':True,'PASS':True,'commit':commit,'plan':b(O/'RECOVERY_EXACT_605_FILE_PLAN.json'),'transaction':b(O/'COMPLETE.json'),
  'original_plan':e['push_plan'],'original_failure_preserved':b(R/'V222_PUSH_FAILURE_NO_RETRY.json'),
  'source_original_files_rehashed':len([x for x in manifest['entries'] if x['category']=='original_scientific_record']),
  'all_source_and_destination_SHA_unchanged':True,'physical_files':len(planned),'physical_bytes':rp['total_bytes'],
  'all_Git_blob_SHA_exact':True,'HTTP_HEAD':http,'working_tree_clean':True,'index_empty':True,
  'owned_new_checkpoints':2,'RAW_arrays_published':0,'scientific_phase_writes':0,'latest_real_evaluation_unchanged':'V220',
  'selected_arm':None,'adopted':False,'fully_solved':False,'goal_status':'active'}
 newj(R/'ROOT_FINAL_V222_MAIN_PUBLICATION_LOCAL_RECEIPT.json',result);return result
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['push','post'],required=True);args=ap.parse_args()
try:print(json.dumps({'push':push,'post':post}[args.mode](),ensure_ascii=False))
except BaseException as error:
 newj(O/('FAILURE_'+args.mode.upper()+'_PRESERVED.json'),{'completed':False,'error_type':type(error).__name__,'error':str(error),'automatic_retry':False});raise
