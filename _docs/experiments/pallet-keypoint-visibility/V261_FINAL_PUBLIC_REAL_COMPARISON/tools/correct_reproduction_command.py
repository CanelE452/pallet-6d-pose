"""One documentation-only amendment, preserving the sealed pre-amendment bytes."""
import hashlib,json,os,stat,subprocess
from pathlib import Path
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
F=R/'V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON';P=R/'V261_FINAL_PUBLICATION_PACKAGE'
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
DEST='_docs/experiments/pallet-keypoint-visibility/V261_FINAL_PUBLIC_REAL_COMPARISON'
def safe(p):
 p=Path(p);assert p.is_absolute() and '..' not in p.parts and not any(t in str(p).casefold() for t in ('v112','cache','.pyc'))
 for a in [*reversed(p.parents),p]:
  try:m=os.lstat(a).st_mode
  except FileNotFoundError:continue
  assert not stat.S_ISLNK(m),str(a)
 return p
def raw(p):return safe(p).read_bytes()
def sha(b):return hashlib.sha256(b).hexdigest()
def bind(p):
 p=safe(p);h=hashlib.sha256();n=0
 with p.open('rb') as s:
  for b in iter(lambda:s.read(1<<20),b''):h.update(b);n+=len(b)
 return dict(path=str(p),sha256=h.hexdigest(),size_bytes=n)
def jb(j):return (json.dumps(j,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
def git(*args,inp=None):return subprocess.run(['git','--literal-pathspecs',*args],cwd=safe(REPO),input=inp,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
def amend():
 old=raw(P/'MANIFEST.json');assert sha(old)=='e8f09fedfdbdeafe57d1cd5b3c7e7cfa2714ae4fa62da55a9ff4acc1bcb2d348'
 m=json.loads(old);before=raw(P/'README_KO.md');assert raw(REPO/DEST/'README_KO.md')==before
 assert before.count(b'--package . --output-dir')==1
 after=before.replace(b'--package . --output-dir',b'--package "$PWD" --output-dir')
 h=safe(F/'publication_documentation_correction_v1');assert not h.exists();h.mkdir()
 for name,data in [('BEFORE_MANIFEST.json',old),('BEFORE_README_KO.md',before),('CORRECTED_README_KO.md',after)]: (h/name).write_bytes(data)
 newbinding=bind(h/'CORRECTED_README_KO.md')
 original=next(a for a in m['artifacts'] if a['relative_path']=='README_KO.md')['original_source_binding_inert']
 for b in m['coverage_contract']['final_report']:
  if b==original:b.update(newbinding)
 for row in m['files']:
  if row['relative_path']=='README_KO.md':row.update(sha256=sha(after),size_bytes=len(after))
 for a in m['artifacts']:
  if a['relative_path']=='README_KO.md':a.update(original_sha256=sha(after),original_size_bytes=len(after),original_source_binding_inert=newbinding)
 additions=[('archive/documentation_correction_v1/BEFORE_MANIFEST.json',h/'BEFORE_MANIFEST.json'),('archive/documentation_correction_v1/BEFORE_README_KO.md',h/'BEFORE_README_KO.md'),('tools/correct_reproduction_command.py',Path(__file__).absolute())]
 sources=[]
 for rel,source in additions:
  b=bind(source);target=safe(P/rel);assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw(source));sources.append(b)
  m['files'].append(dict(relative_path=rel,sha256=b['sha256'],size_bytes=b['size_bytes']))
  m['artifacts'].append(dict(relative_path=rel,group='documentation_correction',encoding='identity',original_sha256=b['sha256'],original_size_bytes=b['size_bytes'],transport_files=[rel],compressed_stream_sha256=None,compressed_stream_bytes=None,original_source_binding_inert=b))
 m['files'].sort(key=lambda x:x['relative_path']);m['file_count']=len(m['files']);m['logical_artifact_count']=len(m['artifacts']);m['total_bytes']=sum(row['size_bytes'] for row in m['files']);m['group_counts']['documentation_correction']=len(additions);m['coverage_contract']['documentation_correction']=sources
 m['postbuild_documentation_amendment']=dict(reason='README relative --package . violates absolute-only restorer API; replaced only that argument with quoted $PWD',before_manifest_sha256=sha(old),before_readme_sha256=sha(before),after_readme_sha256=sha(after),corrected_source_binding=newbinding,pre_amendment_bytes_preserved=True,model_or_features_or_predictions_or_threshold_changes=0)
 safe(P/'README_KO.md').write_bytes(after);safe(P/'MANIFEST.json').write_bytes(jb(m))
 rels=['README_KO.md','MANIFEST.json',*[rel for rel,_ in additions]]
 for rel in rels:
  dest=safe(REPO/DEST/rel);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw(P/rel))
 git('add','-f','--pathspec-from-file=-','--pathspec-file-nul',inp=b'\0'.join((DEST+'/'+rel).encode() for rel in rels)+b'\0')
 result=dict(completed=True,PASS=True,manifest=bind(P/'MANIFEST.json'),archive=str(h),changed_and_added_paths=rels,model_or_training_or_predictions_changes=0)
 (h/'AMENDMENT_RECEIPT.json').write_bytes(jb(result));print(json.dumps(result))
def verify():
 m=json.loads(raw(P/'MANIFEST.json'));expected={DEST+'/'+row['relative_path']:(row['sha256'],row['size_bytes']) for row in m['files']};expected[DEST+'/MANIFEST.json']=(sha(raw(P/'MANIFEST.json')),len(raw(P/'MANIFEST.json')))
 e=json.loads(raw(F/'ACTUAL_PUBLICATION_STAGE_ENTRY.json'));tails={}
 for rel,b in e['registry_prefixes'].items():
  old=git('show','HEAD:'+rel).stdout;prefix=b['utf8'].encode();current=raw(REPO/rel);assert current==prefix+old
  expected[rel]=(sha(current),len(current));tails[rel]=dict(old_sha256=sha(old),old_size_bytes=len(old),prefix_size_bytes=len(prefix))
 staged=git('diff','--cached','--name-only','-z').stdout.decode().split('\0');assert staged[-1]=='' and set(staged[:-1])==set(expected) and len(staged)-1==len(expected)
 # One long-lived Git reader verifies every index blob without thousands of subprocesses.
 proc=subprocess.Popen(['git','cat-file','--batch'],cwd=safe(REPO),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 try:
  for rel,(digest,size) in sorted(expected.items()):
   assert '\n' not in rel and '\r' not in rel
   proc.stdin.write((':'+rel+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob' and int(header[2])==size
   h=hashlib.sha256();left=size
   while left:
    data=proc.stdout.read(min(left,1<<20));assert data;h.update(data);left-=len(data)
   assert proc.stdout.read(1)==b'\n' and h.hexdigest()==digest
   wb=bind(REPO/rel);assert wb['sha256']==digest and wb['size_bytes']==size
  proc.stdin.close();assert proc.wait(timeout=60)==0
 finally:
  if proc.poll() is None:proc.kill()
 git('diff','--cached','--check','--',*e['whitespace_check_relative_paths'])
 restore=R/'V261_FULL_RESTORE_V2/FULL_RESTORE_QA.json';qa=json.loads(raw(restore));assert qa['completed'] and qa['PASS'] and qa['package_manifest_sha256']==sha(raw(P/'MANIFEST.json')) and qa['logical_artifacts_restored']==len(m['artifacts'])
 result=dict(completed=True,PASS=True,staged_files=len(expected),staged_tree=git('write-tree').stdout.decode().strip(),parent_HEAD=git('rev-parse','HEAD').stdout.decode().strip(),manifest=bind(P/'MANIFEST.json'),full_restore_QA=bind(restore),all_staged_index_and_working_bytes_verified=True,registry_tails=tails,original_stage_QA=bind(F/'publication_stage_v1/STAGED_QA.json'),documentation_amendment=bind(F/'publication_documentation_correction_v1/AMENDMENT_RECEIPT.json'),scientific_artifacts_or_results_changed=False)
 out=F/'FINAL_AMENDED_STAGE_QA.json';assert not out.exists();out.write_bytes(jb(result));print(json.dumps(result))
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['amend','verify']);args=ap.parse_args();amend() if args.mode=='amend' else verify()
