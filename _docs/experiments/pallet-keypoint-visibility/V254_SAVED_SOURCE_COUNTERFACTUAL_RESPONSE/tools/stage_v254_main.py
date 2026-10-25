"""Stage only the verified V254 publication package and three index prefixes."""
from pathlib import Path
import hashlib,json,subprocess

R=Path(__file__).absolute().parent
PKG=R/'V254_PUBLICATION_PACKAGE'
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
DEST=Path('_docs/experiments/pallet-keypoint-visibility/V254_SAVED_SOURCE_COUNTERFACTUAL_RESPONSE')
PARENT='49256c1cd794f214940efa7ae79fcb0628b32a8a'

def safe(p):
 p=Path(p)
 if not p.is_absolute() or '..' in p.parts or any(t in str(p).lower() for t in ('v112','cache','.pyc')):raise ValueError(str(p))
 for a in reversed([p,*p.parents]):
  if a.is_symlink():raise ValueError('Artifact symlink '+str(a))
 return p
def read(p):return safe(p).read_bytes()
def sha(b):return hashlib.sha256(b).hexdigest()
def git(*args,inp=None):
 r=subprocess.run(['git',*args],cwd=REPO,input=inp,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
 return r.stdout
def write_new(p,b):
 p=safe(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)

def main():
 assert git('rev-parse','HEAD').decode().strip()==PARENT
 assert not git('status','--porcelain','-uno')
 manifest_bytes=read(PKG/'MANIFEST.json');m=json.loads(manifest_bytes);assert m['completed'] and m['PASS']
 cpu=json.loads(read(R/'V254_ROOT_SAVED_CF_RESPONSE/ACTUAL_PACKAGED_CPU_REPLAY_QA.json'))
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
 for name in ('ACTUAL_PACKAGED_CPU_REPLAY_QA.json','ACTUAL_MAIN_PUBLICATION_ENTRY.json'):
  changed[str(DEST/'PUBLICATION_RECEIPTS'/name)]=read(R/'V254_ROOT_SAVED_CF_RESPONSE'/name)
 refs={
  'readme.md':('V254：저장된 동일 query 가림 대조 반응', '_docs/experiments/pallet-keypoint-visibility/V254_SAVED_SOURCE_COUNTERFACTUAL_RESPONSE/README_KO.md'),
  '_docs/experiments/pallet-keypoint-visibility/README.md':('V254：합성 Source의 고정 좌표 COVER·SHAM 반응','V254_SAVED_SOURCE_COUNTERFACTUAL_RESPONSE/README_KO.md'),
  '_docs/experiments/pallet-keypoint-visibility/EXPERIMENT_INDEX_KO.md':('V254：같은 원 query의 저장 COVER·SHAM 반응 진단','V254_SAVED_SOURCE_COUNTERFACTUAL_RESPONSE/README_KO.md'),
 }
 for rel,(title,url) in refs.items():
  old=read(REPO/rel);prefix=('> ['+title+']('+url+') — 새 학습 없이 저장된 세 모델의 같은 초기 좌표 BEFORE·COVER·SHAM 반응과 누락을 검산합니다. 현재 평가 데이터와 PnP는 사용하지 않았습니다.\n\n').encode()
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
 write_new(R/'V254_STAGED_BLOB_QA.json',(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode())
 print(json.dumps({'completed':True,'PASS':True,'changed_files':len(allow),'package_leaves':len(m['files']),'registry_suffixes_preserved':True}))

if __name__=='__main__':main()
