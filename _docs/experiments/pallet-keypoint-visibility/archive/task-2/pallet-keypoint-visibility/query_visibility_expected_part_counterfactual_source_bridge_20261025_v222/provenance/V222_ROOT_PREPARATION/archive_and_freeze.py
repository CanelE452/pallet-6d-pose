"""Preserve bounded V222 prep/QA records and freeze completed SOURCE phase once."""
import os,stat,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'publication_root_preparation';S=P/'query_visibility_expected_part_counterfactual_source_bridge_20261025_v222'
FOLDERS=['V222_COUNTERFACTUAL_PRODUCER_C','V222_COUNTERFACTUAL_PRODUCER_EXECUTION_C',
 'V222_TARGET_AUTHORITY_REVIEW_A','V222_ACTUAL_PIXEL_QA_A','V222_TRAINER_CODE_QA_A',
 'V222_PAIRED_BRIDGE_TRAINING_B','V222_PAIRED_BRIDGE_TRAINING_B_V2',
 'V222_ACTUAL_MODEL_QA_A','V222_ROOT_PREPARATION','V222_PUBLICATION_PREPARATION_C','V222_REPORT_QA_B']
def safe(p):
 p=Path(p)
 if not p.is_absolute() or '..' in p.parts or 'v112' in str(p).lower():raise RuntimeError('lexical guard')
 return p
def parent(p):
 p=safe(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for part in p.parts[1:-1]:
   n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  return p,fd
 except BaseException:os.close(fd);raise
def rb(p):
 p,fd=parent(p)
 try:n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 if not stat.S_ISREG(os.fstat(n).st_mode):raise RuntimeError('regular file only')
 with os.fdopen(n,'rb') as f:return f.read()
def sha(b):return hashlib.sha256(b).hexdigest()
def bind(p):
 b=rb(p);return {'path':str(p),'sha256':sha(b),'size_bytes':len(b)}
def new(p,b):
 safe(p);p.parent.mkdir(parents=True,exist_ok=True);p,fd=parent(p)
 try:n=os.open(p.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 with os.fdopen(n,'wb') as f:f.write(b)
def jnew(p,o):new(p,(json.dumps(o,ensure_ascii=False,indent=2)+'\n').encode())
def files(root,allow):
 todo=[root];out=[]
 while todo:
  d=todo.pop();safe(d)
  with os.scandir(d) as it:
   for ent in sorted(it,key=lambda x:x.name):
    if ent.name in {'__pycache__','matplotlib_cache'}:continue
    p=safe(d/ent.name)
    if p.suffix.lower() in {'.npz','.npy','.pth','.safetensors','.pkl','.pickle','.bin'}:raise RuntimeError('unexpected RAW before stat')
    mode=ent.stat(follow_symlinks=False).st_mode
    if stat.S_ISDIR(mode):todo.append(p)
    elif stat.S_ISREG(mode) and p.suffix.lower() in allow:out.append(p)
    else:raise RuntimeError('unknown leaf/symlink '+str(p))
 return sorted(out)
qa=R/'V222_REPORT_QA_B/REVIEW.json'
assert json.loads(rb(qa))['PASS'] is True
records=[]
for folder in FOLDERS:
 for source in files(R/folder,{'.py','.json','.md','.csv','.txt','.log','.patch','.diff'}):
  rel=source.relative_to(R);dest=S/'provenance'/rel;b=rb(source);new(dest,b)
  records.append({'original':str(source),'preserved_relative':str(dest.relative_to(S)), 'sha256':sha(b),'size_bytes':len(b)})
jnew(S/'provenance/ARCHIVE_MANIFEST.json',{'completed':True,'original_bytes_preserved':True,'source_folders':FOLDERS,
 'records':records,'count':len(records),'external_source_payloads_followed':0,'old_weights_arrays_copied':0,
 'notes':'Only owned completed V222 prep/QA records. V1 schema and graphic failures preserved. Original absolute binding pointers retained, not followed.'})
phase={'completed':True,'phase':'V222','scientific_outcome':'SOURCE_GENERALIZATION_NOT_IMPROVED',
 'scientific_result':bind(S/'ROOT_RESULT.json'),'report':bind(S/'PORTABLE_REPORT_KO.md'),
 'actual_fit':bind(S/'paired_source_fit_v1/COMPLETE.json'),'all2_freeze':bind(S/'paired_source_fit_v1/ALL2_FINAL_FROZEN_BEFORE_ANY_DEV_OR_REAL_OUTPUT.json'),
 'data_pixel_QA':bind(S/'provenance/V222_ACTUAL_PIXEL_QA_A/actual_review_v1/REVIEW.json'),
 'actual_model_log_QA':bind(S/'provenance/V222_ACTUAL_MODEL_QA_A/actual_review_v1/REVIEW.json'),
 'saved_report_QA':bind(S/'provenance/V222_REPORT_QA_B/REVIEW.json'),
 'provenance_archive':bind(S/'provenance/ARCHIVE_MANIFEST.json'),
 'current223_new_inference':False,'latest_real_benchmark':'V220','selected_model':None,'adopted':False,'fully_solved':False,'goal_status':'active','PnP':0}
jnew(S/'ROOT_PHASE_RESULT.json',phase)
inventory=[]
owned={'paired_source_fit_v1/C66CONTINUATION_CONTROL/FINAL_CHECKPOINT.pt','paired_source_fit_v1/C67PLUS_EXPECTED_PART_CF_WEAK/FINAL_CHECKPOINT.pt'}
for p in files(S,{'.py','.json','.jsonl','.md','.csv','.txt','.log','.patch','.diff','.png','.html','.svg','.pt'}):
 if p.suffix=='.pt':assert str(p.relative_to(S)) in owned
 b=rb(p);inventory.append({'relative_path':str(p.relative_to(S)),'sha256':sha(b),'size_bytes':len(b)})
assert len([x for x in inventory if x['relative_path'].endswith('.pt')])==2
jnew(S/'ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json',{'completed':True,'phase':'V222',
 'frozen':True,'all_scientific_phase_files_final_no_more_writes':True,'recorded_UTC':datetime.now(timezone.utc).isoformat(),
 'phase_result':{k:v for k,v in bind(S/'ROOT_PHASE_RESULT.json').items() if k!='size_bytes'},'files_before_this_freeze':inventory,
 'file_count_before_this_freeze':len(inventory),'bytes_before_this_freeze':sum(x['size_bytes'] for x in inventory),
 'new_owned_models':2,'model_selected':None,'adopted':False,'fully_solved':False,'goal_status':'active'})
print(json.dumps({'completed':True,'archived':len(records),'freeze':bind(S/'ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json'),'source_count_including_freeze':len(inventory)+1}))
