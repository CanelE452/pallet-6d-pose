"""Explicit V247 manifest-only copy/stage/blob checks. Commit/push remain Root's.

Adapted from the reviewed V246 publisher. No broad staging, remote operation,
model execution, dependency pointer following or history rewrite.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V247_PUBLICATION_PREPARATION_B'
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
BASE='_docs/experiments/pallet-keypoint-visibility'
NAME='V247_FROZEN_SAM_PUBLIC_TRAIN_DIAGNOSTIC'
PARENT_PREFIX='087ea4be20476c5b5f8550cfe8c17035a6dc9547'
REC=R/'V247_MAIN_PREPARATION.json'

def safe(rel):
 assert isinstance(rel,str)
 assert not any(s in rel.casefold() for s in ('v112','__pycache__','.pyc','cache'))
 p=PurePosixPath(rel);assert not p.is_absolute() and '..' not in p.parts and '\\' not in rel
 return rel

def sha(b):return hashlib.sha256(b).hexdigest()

def guarded(p):
 p=Path(p);assert p.is_absolute() and '..' not in p.parts
 assert not any(s in str(p).casefold() for s in ('v112','__pycache__','.pyc','cache'))
 for ancestor in reversed((p,*p.parents)):assert not ancestor.is_symlink()
 return p

def git(*args):return subprocess.check_output(['git',*args],cwd=guarded(REPO))

def jsave(p,obj):
 with guarded(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')

def prepare(entry_path):
 eb=guarded(entry_path).read_bytes();e=json.loads(eb)
 assert e.get('approved_publication_prepare') is True and e.get('automatic_retry') is False
 assert e['publisher']['path']==str(H/'publish_v247_main.py') and e['publisher']['sha256']==sha(guarded(__file__).read_bytes())
 parent=e['parent_commit'];assert len(parent)==40 and parent.startswith(PARENT_PREFIX) and all(c in '0123456789abcdef' for c in parent)
 assert git('rev-parse','HEAD').decode().strip()==parent
 branch=subprocess.run(['git','symbolic-ref','-q','--short','HEAD'],cwd=guarded(REPO),check=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 assert (branch.returncode==0 and branch.stdout.decode().strip()=='main') or (branch.returncode==1 and not branch.stdout.strip())
 # Detached HEAD is permitted only after the same exact parent SHA check above;
 # no checkout, switch or branch mutation is performed.
 assert not git('status','--porcelain=v1','-uno').strip() and not git('diff','--cached','--name-only','-z')
 package=R/'V247_PUBLICATION_PACKAGE';assert e['package_root']==str(package);mb=guarded(package/'MANIFEST.json').read_bytes();manifest=json.loads(mb)
 assert sha(mb)==e['manifest_sha256'] and manifest['schema']=='V247_FROZEN_SAM_PUBLIC_TRAIN_DIAGNOSTIC_PUBLICATION'
 status=manifest['scientific_status'];assert status['new_training_updates']==status['native_labels_created']==status['current_or_held_payloads']==0 and status['final_outputs_ready'] is True and status['model_adopted'] is False and status['training_admission'] is False
 dest=REPO/BASE/NAME;assert not guarded(dest).exists();copied=[]
 def put(rel,data):
  p=guarded(REPO/safe(rel));assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
  copied.append({'path':rel,'sha256':sha(data),'bytes':len(data)})
 names=[safe(row['path']) for row in manifest['files']];assert len(names)==len(set(names)) and 'MANIFEST.json' not in names
 for row in manifest['files']:
  assert set(row)=={'path','sha256','bytes'} and type(row['bytes']) is int and 0<=row['bytes']<95_000_000
  assert isinstance(row['sha256'],str) and len(row['sha256'])==64 and all(c in '0123456789abcdef' for c in row['sha256'])
  rel=safe(row['path']);p=guarded(package/rel);b=p.read_bytes();assert (len(b),sha(b))==(row['bytes'],row['sha256'])
  put(f'{BASE}/{NAME}/{rel}',b)
 put(f'{BASE}/{NAME}/MANIFEST.json',mb)
 prefixes={
  'readme.md':'## 팔레트 가림: V247 공개 TRAIN 동결 SAM 진단\n\n[보고서·고정 사례·원 query readout·코드·검산](_docs/experiments/pallet-keypoint-visibility/V247_FROZEN_SAM_PUBLIC_TRAIN_DIAGNOSTIC/README_KO.md). 기존 SAM2.1과 원 q8 minmax prompt로 공개 TRAIN44프레임/352query의 segmentation proposal 근거를 기록했습니다. mask support/outside는 native D/E/SELF 정답이나 실사 성능 개선을 뜻하지 않으며, 새 학습·현재 평가 추론·PnP는0입니다. 이전 기록은 아래에 보존합니다.\n\n',
  BASE+'/README.md':'## V247: 공개 TRAIN 원 q8 prompt의 동결 SAM 진단\n\n[실제 실행·readout·고정 gallery·재검산](V247_FROZEN_SAM_PUBLIC_TRAIN_DIAGNOSTIC/README_KO.md). 원 q/L/native/U와 prompt/runtime 지원을 구분하고, logit>0 exact/5×5 proposal 근거를 별도로 기록합니다. teacher는 모델 입력이 아니며 native 라벨·모델 채택·새 학습 admission을 만들지 않았습니다.\n\n',
  BASE+'/EXPERIMENT_INDEX_KO.md':'## V247 · SAM2.1 공개 SOURCE same-query proposal 진단\n\n[보고서·전체352행·chosen logits·실패 기록·검산](V247_FROZEN_SAM_PUBLIC_TRAIN_DIAGNOSTIC/README_KO.md). 공식 동결 모델의 class-agnostic mask와 조건부 교사/의도한 nativepart 권한을 분리합니다. 새 학습0·현재 평가 사용0·모델 미채택이며 기존 기록은 아래에 유지합니다.\n\n'}
 assert e['registry_paths']==list(prefixes)
 entries=[]
 for rel,prefix in prefixes.items():
  p=guarded(REPO/safe(rel));old=p.read_bytes();assert old==git('show',parent+':'+rel)
  pre=prefix.encode();p.write_bytes(pre+old);assert p.read_bytes()[len(pre):]==old
  entries.append({'path':rel,'sha256':sha(pre+old),'bytes':len(pre+old),'old_sha256':sha(old),'old_bytes_preserved_as_suffix':True})
 jsave(REC,{'completed':True,'PASS':True,'parent_commit':parent,'manifest_sha256':sha(mb),'entry':{'path':str(guarded(entry_path)),'sha256':sha(eb)},'copied':copied,'entry_points':entries,'immutable_package_manifest_preserved':True,'two_ignored_cache_paths_not_touched':True,'new_model_or_payload_execution':0,'normal_nonforce_push_only_external_Root':True})
 print(json.dumps({'completed':True,'PASS':True,'new_files':len(copied),'entry_points':len(entries)}))

def check(stage=False,commit=False):
 rec=json.loads(guarded(REC).read_bytes());parent=rec['parent_commit'];assert parent.startswith(PARENT_PREFIX)
 rows=rec['copied']+rec['entry_points'];names=[safe(row['path']) for row in rows];assert len(names)==len(set(names))
 if commit:
  assert git('rev-parse','HEAD^').decode().strip()==parent
  changed=set(git('diff-tree','--no-commit-id','--name-only','-r','-z','HEAD').decode().strip('\0').split('\0'))
 else:
  assert git('rev-parse','HEAD').decode().strip()==parent
  for row in rows:
   b=guarded(REPO/row['path']).read_bytes();assert (len(b),sha(b))==(row['bytes'],row['sha256'])
  if stage:
   assert not git('diff','--cached','--name-only','-z')
   allow=guarded(R/'V247_STAGE_ALLOWLIST.nul')
   with allow.open('xb') as f:f.write(b''.join(name.encode()+b'\0' for name in names))
   subprocess.run(['git','add','-f','--pathspec-from-file='+str(allow),'--pathspec-file-nul'],cwd=guarded(REPO),check=True)
  changed=set(git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0'))
 assert changed==set(names),(changed-set(names),set(names)-changed)
 for row in rows:
  revision=('HEAD:' if commit else ':')+row['path'];b=git('show',revision);assert (len(b),sha(b))==(row['bytes'],row['sha256'])
 mode='COMMIT' if commit else 'STAGED'
 jsave(R/f'V247_{mode}_QA.json',{'completed':True,'PASS':True,'files_verified':len(rows),'all_blob_SHA_and_bytes_exact':True,'manifest_sha256':rec['manifest_sha256'],'explicit_allowlist_only':True,'revision':git('rev-parse','HEAD').decode().strip(),'force_push':False})
 print(json.dumps({'completed':True,'PASS':True,'mode':mode,'files_verified':len(rows)}))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('prepare','stage','check-commit'));ap.add_argument('--entry');args=ap.parse_args()
 if args.mode=='prepare':assert args.entry;prepare(args.entry)
 else:check(stage=args.mode=='stage',commit=args.mode=='check-commit')
