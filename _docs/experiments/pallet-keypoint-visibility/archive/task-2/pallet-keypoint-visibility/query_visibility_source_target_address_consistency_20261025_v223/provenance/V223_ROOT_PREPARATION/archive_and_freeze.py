"""Preserve bounded V223 prep/QA records and freeze completed SOURCE phase once."""
import os,stat,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'publication_root_preparation';S=P/'query_visibility_source_target_address_consistency_20261025_v223'
FOLDERS=['V223_TARGET_ADDRESS_A','V223_INPUT_TARGET_CONSISTENCY_B','V223_PUBLIC_SOURCE_COVERAGE_C',
 'V223_SPLIT_TARGET_TRAINING_B','V223_SPLIT_TARGET_TRAINER_QA_A','V223_ACTUAL_MODEL_QA_A',
 'V223_ROOT_PREPARATION','V223_PUBLICATION_PREPARATION_C','V223_REPORT_QA_B']
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
qa=R/'V223_REPORT_QA_B/REVIEW.json'
assert json.loads(rb(qa))['PASS'] is True
records=[]
for folder in FOLDERS:
 for source in files(R/folder,{'.py','.json','.md','.csv','.txt','.log','.patch','.diff'}):
  rel=source.relative_to(R);dest=S/'provenance'/rel;b=rb(source);new(dest,b)
  records.append({'original':str(source),'preserved_relative':str(dest.relative_to(S)), 'sha256':sha(b),'size_bytes':len(b)})
jnew(S/'provenance/ARCHIVE_MANIFEST.json',{'completed':True,'original_bytes_preserved':True,'source_folders':FOLDERS,
 'records':records,'count':len(records),'external_source_payloads_followed':0,'old_weights_arrays_copied':0,
 'notes':'Only owned completed V223 prep/QA records. Metadata scope exception and completed source audits preserved. Original absolute binding pointers retained, not followed.'})
phase={'completed':True,'phase':'V223','scientific_outcome':'SPLIT_ROUTING_NOT_JOINT_SOURCE_GAIN',
 'scientific_result':bind(S/'ROOT_RESULT.json'),'report':bind(S/'PORTABLE_REPORT_KO.md'),
 'actual_fit':bind(S/'paired_source_fit_v1/COMPLETE.json'),'all2_freeze':bind(S/'paired_source_fit_v1/ALL2_FINAL_FROZEN_BEFORE_ANY_DEV_OR_REAL_OUTPUT.json'),
 'source_address_QA':bind(S/'provenance/V223_TARGET_ADDRESS_A/actual_audit_v1/SUMMARY.json'),
 'actual_model_log_QA':bind(S/'provenance/V223_ACTUAL_MODEL_QA_A/actual_review_v1/REVIEW.json'),
 'saved_report_QA':bind(S/'provenance/V223_REPORT_QA_B/REVIEW.json'),
 'provenance_archive':bind(S/'provenance/ARCHIVE_MANIFEST.json'),
 'current223_new_inference':False,'protected_metadata_exception_rows':360,'exception_values_used_for_design_or_fit':False,'latest_real_benchmark':'V220','selected_model':None,'adopted':False,'fully_solved':False,'goal_status':'active','PnP':0}
index='# V223 — 점·부위 출력 분리: SOURCE 공동 개선 미확인\n\n'
index+='같은 구조/초기값/데이터/1100update에서 strong 감독 출력 경로만 대조했다. CF DEV 정답17/27→17/27, 세 상태 완전정답2/9→1/9, natural DEV9/12동일이다. 입력6297행의 exact opposite-label group0을 확인했고 shared control은 이전C67SOURCElogit전부재현했다. GT-derived finite-part weak이며 실제nativegold/생산query전이/실사개선은 미입증이다. 현재223새추론·PnP·모델채택0, 최신실사V220유지. 공개보류metadata360행 eager-parse 예외는기록했고설계/학습사용0이며phase전체metadata접근0주장은하지않는다.\n\n'
archive='archive/task-2/pallet-keypoint-visibility/'+S.name+'/'
for label,path in [('V223 보고서',archive+'PORTABLE_REPORT_KO.md'),('CM·두클래스 P/R 표',archive+'saved_source_comparison_v1/SOURCE_FAMILIES_SEPARATE_CM_PRECISION_RECALL.csv'),('DEV9묶음 실제 crop',archive+'saved_source_comparison_v1/PORTABLE_GALLERY_KO.md'),('코드·실행·QA·범위예외 원문',archive+'provenance/ARCHIVE_MANIFEST.json')]:index+='- ['+label+']('+path+')\n'
index+='\n---\n\n'
phase['publication_prefixes']={'index_utf8':index,
 'root_readme_utf8':'V223 점·부위 출력 분리 실험/표/crop/QA를 추가했습니다. SOURCE 공동개선 미확인, 현재223 새추론·모델채택0입니다. [최신 실험 인덱스](_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_THROUGH_V223_KO.md).\n\n',
 'docs_readme_utf8':'V223: 점·부위 출력 분리 대조와 입력일관성 검사를 완료했습니다. [표·전체crop·실행/QA·범위예외](EXPERIMENT_INDEX_THROUGH_V223_KO.md). 최신실사비교는V220이며 이번 SOURCE 공동개선은확인되지않았습니다.\n\n'}
jnew(S/'ROOT_PHASE_RESULT.json',phase)
inventory=[]
owned={'paired_source_fit_v1/C68SHARED_POINT_PART_LOGIT/FINAL_CHECKPOINT.pt','paired_source_fit_v1/C69SEPARATE_POINT_PART_LOGITS/FINAL_CHECKPOINT.pt'}
for p in files(S,{'.py','.json','.jsonl','.md','.csv','.txt','.log','.patch','.diff','.png','.html','.svg','.pt'}):
 if p.suffix=='.pt':assert str(p.relative_to(S)) in owned
 b=rb(p);inventory.append({'relative_path':str(p.relative_to(S)),'sha256':sha(b),'size_bytes':len(b)})
assert len([x for x in inventory if x['relative_path'].endswith('.pt')])==2
jnew(S/'ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json',{'completed':True,'phase':'V223',
 'frozen':True,'all_scientific_phase_files_final_no_more_writes':True,'recorded_UTC':datetime.now(timezone.utc).isoformat(),
 'phase_result':{k:v for k,v in bind(S/'ROOT_PHASE_RESULT.json').items() if k!='size_bytes'},'files_before_this_freeze':inventory,
 'file_count_before_this_freeze':len(inventory),'bytes_before_this_freeze':sum(x['size_bytes'] for x in inventory),
 'new_owned_models':2,'model_selected':None,'adopted':False,'fully_solved':False,'goal_status':'active'})
print(json.dumps({'completed':True,'archived':len(records),'freeze':bind(S/'ROOT_FINAL_SCIENTIFIC_PHASE_FREEZE.json'),'source_count_including_freeze':len(inventory)+1}))
