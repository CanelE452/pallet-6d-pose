"""Stage only the verified V256 publication package and three index prefixes."""
from pathlib import Path
import hashlib,json,subprocess

R=Path(__file__).absolute().parent
PKG=R/'V256_PUBLICATION_PACKAGE_V2'
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
DEST=Path('_docs/experiments/pallet-keypoint-visibility/V256_MATCHED_NATIVE_DATA_PAIR')
PARENT='a82d76f37545ab7b6bba84b6eea0e8a8414e98c0'

def safe(p):
 p=Path(p)
 if not p.is_absolute() or '..' in p.parts or any(t in str(p).lower() for t in ('v112','cache','.pyc')):raise ValueError(str(p))
 for a in reversed([p,*p.parents]):
  if a.is_symlink():raise ValueError('Artifact symlink '+str(a))
 return p
def read(p):return safe(p).read_bytes()
def sha(b):return hashlib.sha256(b).hexdigest()
def git(*args,inp=None):
 r=subprocess.run(['git',*args],cwd=safe(REPO),input=inp,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
 return r.stdout
def write_new(p,b):
 p=safe(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)

def main():
 assert git('rev-parse','HEAD').decode().strip()==PARENT
 assert not git('status','--porcelain','-uno')
 manifest_bytes=read(PKG/'MANIFEST.json');m=json.loads(manifest_bytes);assert m['completed'] and m['PASS']
 cpu=json.loads(read(R/'V256_ROOT_NATIVE_DATA_FIT/ACTUAL_PACKAGED_FULL_RESTORE_QA.json'))
 assert cpu['completed'] and cpu['PASS']
 assert cpu['package_manifest']=={'path':str(PKG/'MANIFEST.json'),'sha256':sha(manifest_bytes),'size_bytes':len(manifest_bytes)}
 assert not safe(REPO/DEST).exists()
 changed={};preserved={}
 for row in m['files']:
  rel=Path(row['relative_path'])
  assert not rel.is_absolute() and '..' not in rel.parts
  b=read(PKG/rel);assert sha(b)==row['sha256'] and len(b)==row['size_bytes']
  changed[str(DEST/rel)]=b
 changed[str(DEST/'MANIFEST.json')]=read(PKG/'MANIFEST.json')
 for name in ('ACTUAL_PACKAGED_FULL_RESTORE_QA.json','ACTUAL_MAIN_PUBLICATION_ENTRY.json'):
  changed[str(DEST/'PUBLICATION_RECEIPTS'/name)]=read(R/'V256_ROOT_NATIVE_DATA_FIT'/name)
 refs={
  'readme.md':('V256：새 합성 가림 데이터의 동일 모델 비교', '_docs/experiments/pallet-keypoint-visibility/V256_MATCHED_NATIVE_DATA_PAIR/README_KO.md'),
  '_docs/experiments/pallet-keypoint-visibility/README.md':('V256：기존 데이터와 새 합성 가림 대조의 동일 모델 비교','V256_MATCHED_NATIVE_DATA_PAIR/README_KO.md'),
  '_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_KO.md':('V256：가림 대조 데이터 추가 효과와 공개 실사 미해결 기록','V256_MATCHED_NATIVE_DATA_PAIR/README_KO.md'),
 }
 for rel,(title,url) in refs.items():
  old=read(REPO/rel);prefix=('> ['+title+']('+url+') — 동일 모델·초기값·1,000회 학습으로 새 합성 데이터 추가 효과를 비교했습니다. 합성 가림 recall은 증가했지만 precision·FIT 대조 패턴은 악화했고 공개 YCB 가림은 두 모델 모두 0/57입니다. 모델 채택은 하지 않았습니다.\n\n').encode()
  changed[rel]=prefix+old;preserved[rel]={'old_sha256':sha(old),'old_size_bytes':len(old),'prefix_size_bytes':len(prefix)}
 for rel,b in changed.items():
  if rel in refs:safe(REPO/rel).write_bytes(b)
  else:write_new(REPO/rel,b)
 for rel,v in preserved.items():
  tail=read(REPO/rel)[v['prefix_size_bytes']:];assert sha(tail)==v['old_sha256'] and len(tail)==v['old_size_bytes']
 allow=sorted(changed)
 git('add','-f','--pathspec-from-file=-','--pathspec-file-nul',inp=b'\0'.join(s.encode() for s in allow)+b'\0')
 staged=git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0');assert sorted(staged)==allow
 for rel,b in changed.items():assert git('show',':'+rel)==b
 git('-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check')
 receipt={'completed':True,'PASS':True,'parent':PARENT,'files':[{ 'relative_path':p,'sha256':sha(changed[p]),'size_bytes':len(changed[p])} for p in allow],'registry_preserved_suffixes':preserved,'force_push':False}
 write_new(R/'V256_STAGED_BLOB_QA.json',(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode())
 print(json.dumps({'completed':True,'PASS':True,'changed_files':len(allow),'package_leaves':len(m['files']),'registry_suffixes_preserved':True}))

if __name__=='__main__':main()
