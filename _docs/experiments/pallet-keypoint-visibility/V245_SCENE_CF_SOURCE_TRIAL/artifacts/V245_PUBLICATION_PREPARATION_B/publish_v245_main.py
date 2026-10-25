"""Explicit V245 copy/stage/blob checks. Commit and normal push remain Root's.

Adapted from the reviewed V244 publisher. No broad staging, remote operation,
model execution, dependency pointer following or history rewrite.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V245_PUBLICATION_PREPARATION_B'
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
BASE='_docs/experiments/pallet-keypoint-visibility'
NAME='V245_SCENE_CF_SOURCE_TRIAL'
PARENT_PREFIX='49560f404'
REC=R/'V245_MAIN_PREPARATION.json'

def safe(rel):
 assert isinstance(rel,str)
 assert not any(s in rel.casefold() for s in ('v112','__pycache__','.pyc'))
 p=PurePosixPath(rel);assert not p.is_absolute() and '..' not in p.parts and '\\' not in rel
 return rel

def sha(b):return hashlib.sha256(b).hexdigest()

def guarded(p):
 p=Path(p);assert p.is_absolute() and '..' not in p.parts
 assert not any(s in str(p).casefold() for s in ('v112','__pycache__','.pyc'))
 for ancestor in reversed((p,*p.parents)):assert not ancestor.is_symlink()
 return p

def git(*args):return subprocess.check_output(['git',*args],cwd=guarded(REPO))

def jsave(p,obj):
 with guarded(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')

def prepare(entry_path):
 eb=guarded(entry_path).read_bytes();e=json.loads(eb)
 assert e.get('approved_publication_prepare') is True and e.get('automatic_retry') is False
 assert e['publisher']['path']==str(H/'publish_v245_main.py') and e['publisher']['sha256']==sha(guarded(__file__).read_bytes())
 parent=e['parent_commit'];assert len(parent)==40 and parent.startswith(PARENT_PREFIX) and all(c in '0123456789abcdef' for c in parent)
 assert git('rev-parse','HEAD').decode().strip()==parent
 assert git('symbolic-ref','--short','HEAD').decode().strip()=='main'
 assert not git('status','--porcelain=v1','-uno').strip() and not git('diff','--cached','--name-only','-z')
 package=R/'V245_PUBLICATION_PACKAGE';mb=guarded(package/'MANIFEST.json').read_bytes();manifest=json.loads(mb)
 assert sha(mb)==e['manifest_sha256'] and manifest['schema']=='V245_SCENE_CF_SOURCE_TRIAL_PUBLICATION'
 status=manifest['scientific_status'];assert status['new_training_updates']==1000 and status['pipeline_QA_PASS'] is True and status['SOURCE_gate_PASS'] is False and status['model_adopted'] is False and status['current_or_held_payloads']==0
 dest=REPO/BASE/NAME;assert not guarded(dest).exists();copied=[]
 def put(rel,data):
  p=guarded(REPO/safe(rel));assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
  copied.append({'path':rel,'sha256':sha(data),'bytes':len(data)})
 names=[safe(row['path']) for row in manifest['files']];assert len(names)==len(set(names)) and 'MANIFEST.json' not in names
 for row in manifest['files']:
  rel=safe(row['path']);p=guarded(package/rel);b=p.read_bytes();assert (len(b),sha(b))==(row['bytes'],row['sha256'])
  put(f'{BASE}/{NAME}/{rel}',b)
 put(f'{BASE}/{NAME}/MANIFEST.json',mb)
 # Optional existing Root/C receipts are explicit, never discovered by traversal.
 for binding in e.get('public_receipts',[]):
  p=guarded(binding['path']);assert p.parent==R and p.suffix=='.json' and p.name.startswith('V245')
  b=p.read_bytes();assert sha(b)==binding['sha256'] and ('size_bytes' not in binding or len(b)==binding['size_bytes'])
  put(f'{BASE}/{p.name}',b)
 prefixes={
  'readme.md':'## 팔레트 가림: V245 공개 SOURCE 장면 대조 결과\n\n[보고서·고정 사례·코드·독립 검산](_docs/experiments/pallet-keypoint-visibility/V245_SCENE_CF_SOURCE_TRIAL/README_KO.md). 같은 원 장면의 COVER/ADJACENT_SHAM을 추가한 고정 SOURCE 학습 1,000회를 완료했습니다. 파이프라인 검산은 통과했으나 사전 성능 기준은 통과하지 못했으며 모델은 채택하지 않았습니다. current/held 새 추론·PnP는0입니다. 이전 기록은 아래에 보존합니다.\n\n',
  BASE+'/README.md':'## V245: 같은 장면 CF 추가의 SOURCE 대조\n\n[실제 결과·제한·고정 gallery·재생 코드](V245_SCENE_CF_SOURCE_TRIAL/README_KO.md). 기존 context baseline은 원본 바이트 재사용·새 업데이트0, treatment는 동일 seed17 초기값에서1,000회 학습했습니다. 사전 SOURCE 기준 미통과이며 현재 실사 개선이나 native gold 인증을 뜻하지 않습니다.\n\n',
  BASE+'/EXPERIMENT_INDEX_KO.md':'## V245 · 실제 SOURCE CF 특징 및 고정 학습 대조\n\n[보고서·표·동결 체크포인트·특징·검산](V245_SCENE_CF_SOURCE_TRIAL/README_KO.md). 원 q/L/q8·분리·교사 권한을 유지한 SOURCE-only 대조입니다. 파이프라인 PASS와 성능 기준 FAIL을 구분하고, 모델 미채택·current/held 새 접근0을 보존합니다. 기존 결과는 아래에 유지합니다.\n\n'}
 entries=[]
 for rel,prefix in prefixes.items():
  p=guarded(REPO/safe(rel));old=p.read_bytes();assert old==git('show',parent+':'+rel)
  pre=prefix.encode();p.write_bytes(pre+old);assert p.read_bytes()[len(pre):]==old
  entries.append({'path':rel,'sha256':sha(pre+old),'bytes':len(pre+old),'old_sha256':sha(old),'old_bytes_preserved_as_suffix':True})
 jsave(REC,{'completed':True,'PASS':True,'parent_commit':parent,'manifest_sha256':sha(mb),'entry':{'path':str(guarded(entry_path)),'sha256':sha(eb)},'copied':copied,'entry_points':entries,'two_ignored_cache_paths_not_touched':True,'new_model_or_payload_execution':0,'normal_nonforce_push_only_external_Root':True})
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
   allow=guarded(R/'V245_STAGE_ALLOWLIST.nul')
   with allow.open('xb') as f:f.write(b''.join(name.encode()+b'\0' for name in names))
   subprocess.run(['git','add','-f','--pathspec-from-file='+str(allow),'--pathspec-file-nul'],cwd=guarded(REPO),check=True)
  changed=set(git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0'))
 assert changed==set(names),(changed-set(names),set(names)-changed)
 for row in rows:
  revision=('HEAD:' if commit else ':')+row['path'];b=git('show',revision);assert (len(b),sha(b))==(row['bytes'],row['sha256'])
 mode='COMMIT' if commit else 'STAGED'
 jsave(R/f'V245_{mode}_QA.json',{'completed':True,'PASS':True,'files_verified':len(rows),'all_blob_SHA_and_bytes_exact':True,'manifest_sha256':rec['manifest_sha256'],'explicit_allowlist_only':True,'revision':git('rev-parse','HEAD').decode().strip(),'force_push':False})
 print(json.dumps({'completed':True,'PASS':True,'mode':mode,'files_verified':len(rows)}))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('prepare','stage','check-commit'));ap.add_argument('--entry');args=ap.parse_args()
 if args.mode=='prepare':assert args.entry;prepare(args.entry)
 else:check(stage=args.mode=='stage',commit=args.mode=='check-commit')
