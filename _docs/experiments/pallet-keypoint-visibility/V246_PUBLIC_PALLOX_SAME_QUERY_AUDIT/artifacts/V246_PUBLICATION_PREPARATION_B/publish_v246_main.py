"""Explicit V246 copy/stage/blob checks. Commit and normal push remain Root's.

Adapted from the reviewed V244 publisher. No broad staging, remote operation,
model execution, dependency pointer following or history rewrite.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V246_PUBLICATION_PREPARATION_B'
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
BASE='_docs/experiments/pallet-keypoint-visibility'
NAME='V246_PUBLIC_PALLOX_SAME_QUERY_AUDIT'
PARENT_PREFIX='bff8cc8460166b023ba98735d9d53fe7090591aa'
REC=R/'V246_MAIN_PREPARATION.json'

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
 assert e['publisher']['path']==str(H/'publish_v246_main.py') and e['publisher']['sha256']==sha(guarded(__file__).read_bytes())
 parent=e['parent_commit'];assert len(parent)==40 and parent.startswith(PARENT_PREFIX) and all(c in '0123456789abcdef' for c in parent)
 assert git('rev-parse','HEAD').decode().strip()==parent
 branch=subprocess.run(['git','symbolic-ref','-q','--short','HEAD'],cwd=guarded(REPO),check=False,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 assert (branch.returncode==0 and branch.stdout.decode().strip()=='main') or (branch.returncode==1 and not branch.stdout.strip())
 # Detached HEAD is permitted only after the same exact parent SHA check above;
 # no checkout, switch or branch mutation is performed.
 assert not git('status','--porcelain=v1','-uno').strip() and not git('diff','--cached','--name-only','-z')
 package=R/'V246_PUBLICATION_PACKAGE';mb=guarded(package/'MANIFEST.json').read_bytes();manifest=json.loads(mb)
 assert sha(mb)==e['manifest_sha256'] and manifest['schema']=='V246_PUBLIC_PALLOX_SAME_QUERY_AUDIT_PUBLICATION'
 status=manifest['scientific_status'];assert status['new_training_updates']==status['native_labels_created']==status['current_or_held_payloads']==0 and status['pipeline_QA_PASS'] is True and status['same_q_probe_complete'] is True and status['model_adopted'] is False and status['training_admission'] is False and status['SOURCE_VAL_input_metadata_read_disclosed'] is True
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
  p=guarded(binding['path']);assert p.parent==R and p.suffix=='.json' and p.name.startswith('V246')
  b=p.read_bytes();assert sha(b)==binding['sha256'] and ('size_bytes' not in binding or len(b)==binding['size_bytes'])
  put(f'{BASE}/{p.name}',b)
 # Small explicit post-package corrections stay outside immutable MANIFEST.
 supplement=[]
 for binding in e.get('additional_public_files',[]):
  assert isinstance(binding,dict) and {'path','sha256'}<=set(binding)<={'path','sha256','size_bytes'}
  p=Path(binding['path']);assert p.is_absolute() and p.is_relative_to(R) and p.suffix in ('.py','.json');p=guarded(p)
  assert p.name not in {row['basename'] for row in supplement}
  digest=binding['sha256'];assert isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
  b=p.read_bytes();assert sha(b)==digest and len(b)<95_000_000
  if 'size_bytes' in binding:assert type(binding['size_bytes']) is int and binding['size_bytes']==len(b)
  destination=f'{BASE}/V246_PUBLICATION_PREPARATION_SUPPLEMENT/{p.name}'
  put(destination,b);supplement.append({'basename':p.name,'source_path_provenance_only':str(p),'path':destination,'sha256':digest,'bytes':len(b)})
 prefixes={
  'readme.md':'## 팔레트 가림: V246 공개 Pallox 원 query 조건부 근거 감사\n\n[보고서·고정 SOURCE 사례·코드·검산](_docs/experiments/pallet-keypoint-visibility/V246_PUBLIC_PALLOX_SAME_QUERY_AUDIT/README_KO.md). 기존 공개 TRAIN44프레임/352원 query를 보존하고, 선택22쌍의 기존 조건부 인증으로 같은 픽셀 근거를 진단했습니다. 새 학습·새 모델·현재 평가 추론은0이며 native gold나 실사 성능 개선을 뜻하지 않습니다. 이전 기록은 아래에 보존합니다.\n\n',
  BASE+'/README.md':'## V246: 공개 TRAIN의 원 query를 바꾸지 않은 조건부 진단\n\n[결과·출처·SOURCE 사례·재검산](V246_PUBLIC_PALLOX_SAME_QUERY_AUDIT/README_KO.md). 입력 메타데이터의 SOURCE VAL 읽기/폐기는 투명하게 기록하며, 인증 배열은 선택 TRAIN22쌍만 사용했습니다. 원 U와 native UNKNOWN을 보존하고 라벨·학습 admission을 만들지 않았습니다.\n\n',
  BASE+'/EXPERIMENT_INDEX_KO.md':'## V246 · Pallox TRAIN 원 query 동일주소 조건부 감사\n\n[보고서·352행·22인증·실패 원본·재현 코드](V246_PUBLIC_PALLOX_SAME_QUERY_AUDIT/README_KO.md). 같은 픽셀/5×5 조건부 V/E/U 근거와 native D/E/SELF 권한을 분리합니다. 새 학습0·모델 미채택·현재 평가 사용0이며 기존 기록은 아래에 유지합니다.\n\n'}
 entries=[]
 for rel,prefix in prefixes.items():
  p=guarded(REPO/safe(rel));old=p.read_bytes();assert old==git('show',parent+':'+rel)
  pre=prefix.encode();p.write_bytes(pre+old);assert p.read_bytes()[len(pre):]==old
  entries.append({'path':rel,'sha256':sha(pre+old),'bytes':len(pre+old),'old_sha256':sha(old),'old_bytes_preserved_as_suffix':True})
 jsave(REC,{'completed':True,'PASS':True,'parent_commit':parent,'manifest_sha256':sha(mb),'entry':{'path':str(guarded(entry_path)),'sha256':sha(eb)},'copied':copied,'entry_points':entries,'additional_public_files':supplement,'immutable_package_manifest_preserved':True,'two_ignored_cache_paths_not_touched':True,'new_model_or_payload_execution':0,'normal_nonforce_push_only_external_Root':True})
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
   allow=guarded(R/'V246_STAGE_ALLOWLIST.nul')
   with allow.open('xb') as f:f.write(b''.join(name.encode()+b'\0' for name in names))
   subprocess.run(['git','add','-f','--pathspec-from-file='+str(allow),'--pathspec-file-nul'],cwd=guarded(REPO),check=True)
  changed=set(git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0'))
 assert changed==set(names),(changed-set(names),set(names)-changed)
 for row in rows:
  revision=('HEAD:' if commit else ':')+row['path'];b=git('show',revision);assert (len(b),sha(b))==(row['bytes'],row['sha256'])
 mode='COMMIT' if commit else 'STAGED'
 jsave(R/f'V246_{mode}_QA.json',{'completed':True,'PASS':True,'files_verified':len(rows),'all_blob_SHA_and_bytes_exact':True,'manifest_sha256':rec['manifest_sha256'],'explicit_allowlist_only':True,'revision':git('rev-parse','HEAD').decode().strip(),'force_push':False})
 print(json.dumps({'completed':True,'PASS':True,'mode':mode,'files_verified':len(rows)}))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('prepare','stage','check-commit'));ap.add_argument('--entry');args=ap.parse_args()
 if args.mode=='prepare':assert args.entry;prepare(args.entry)
 else:check(stage=args.mode=='stage',commit=args.mode=='check-commit')
