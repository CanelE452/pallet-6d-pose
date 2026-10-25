"""Package exact V252 source diagnostic directories; never follow provenance pointers."""
from pathlib import Path
import hashlib,json,os,re,argparse
R=Path(__file__).absolute().parent
H=R/'V252_ROOT_DIRECT_REFERENCE'
P=R/'V252_PUBLICATION_PACKAGE'
DIRS={
 'V252_ROOT_DIRECT_REFERENCE':('matched_direct_reference_v1',),
 'V252_DEPENDENCY_CLOSURE_A':(),
 'V252_DIRECT_REFERENCE_TRAINER_B':(),
 'V252_PREVIOUS_PROGRESS_QA_C':(),
 'V252_DIRECT_TRAINER_CODE_QA_C':(),
 'V252_ACTUAL_DIRECT_FIT_QA_C':('actual_review_v1',),
 'V252_PUBLIC_REPLAY_A':(),
}

def safe(p):
 p=Path(p)
 if not p.is_absolute() or '..' in p.parts or any(x in str(p).lower() for x in ('v112','cache','.pyc')):raise ValueError(str(p))
 for a in reversed([p,*p.parents]):
  if a.is_symlink():raise ValueError('Artifact symlink')
 return p
def read(p):return safe(p).read_bytes()
def sha(b):return hashlib.sha256(b).hexdigest()
def verify(p):
 p=safe(p);m=json.loads(read(p/'MANIFEST.json'));assert m['completed'] and m['PASS']
 for row in m['files']:
  rel=Path(row['relative_path']);assert not rel.is_absolute() and '..' not in rel.parts
  b=read(p/rel);assert sha(b)==row['sha256'] and len(b)==row['size_bytes']
 links=0
 for n in ('README_KO.md','ROOT_READOUT_KO.html'):
  f=p/n;t=read(f).decode()
  rr=re.findall(r'(?:href|src)=[\"\']([^\"\']+)',t) if f.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',t)
  for link in rr:
   if link.startswith(('http:','https:','#','mailto:')):continue
   assert safe(f.parent/link.split('#')[0]).is_file(),link
   links+=1
 result={'completed':True,'PASS':True,'leaves':len(m['files']),'bytes':sum(r['size_bytes'] for r in m['files']),'entrypoint_links':links,'original_path_fallback':False}
 print(json.dumps(result));return result
def build():
 assert json.loads(read(R/'V252_ACTUAL_DIRECT_FIT_QA_C/REPORT_REVIEW.json'))['PASS']
 assert not safe(P).exists();P.mkdir();rows=[];seen=set()
 def take(src,rel):
  assert rel not in seen;seen.add(rel);b=read(src);d=safe(P/rel);d.parent.mkdir(parents=True,exist_ok=True)
  with d.open('xb') as f:f.write(b)
  rows.append({'relative_path':rel,'sha256':sha(b),'size_bytes':len(b),'original_path':str(src)})
 def direct(d,prefix,children=()):
  for name in sorted(os.listdir(safe(d))):
   src=safe(d/name)
   if src.is_dir():
    assert name in children,name
    direct(src,prefix+'/'+name)
   elif src.is_file():
    assert src.suffix in ('.py','.md','.json','.jsonl','.csv','.npz','.npy','.png','.html','.pt'),name
    take(src,prefix+'/'+name)
   else:raise ValueError('Nonregular leaf')
 for name,ch in DIRS.items():direct(R/name,'artifacts/'+name,ch)
 for name in ('SOURCE_COMPARISON.csv','SOURCE_GATE.csv','MODEL_DEV_RECALL.png','SOURCE_CONFUSION_MATRICES.png','ROOT_READOUT_KO.html'):
  take(H/name,name)
 take(H/'PUBLIC_README_KO.md','README_KO.md')
 for name in ('build_v252_readout.py','package_v252_results.py','stage_v252_main.py'):take(R/name,'tools/'+name)
 m={'schema':'V252_DIRECT_REFERENCE_HEAD_SOURCE_FIT_PUBLICATION_V1','completed':True,'PASS':True,'files':rows,'phase':'ACTUAL_MATCHED_DIRECT_REFERENCE_SOURCE_FIT','SOURCE_FIT_SYN_reference_statistics_fitted':True,'SOURCE_FIT_YCB_supervised_reference_bank_reused':True,'old_core_forward':0,'trainable_direct_head_parameters_each':1410,'matched_optimizer_updates_each':1000,'total_new_optimizer_updates':2000,'new_DINO_encoder_forward':0,'current_or_held_inference':0,'PnP':0,'model_adopted':False,'goal_status':'active_unsolved','absolute_metadata_paths':'Inert provenance; public saved-model inference replay explicitly rebinds only pinned repository-relative SOURCE inputs and exact package leaves.'}

 with safe(P/'MANIFEST.json').open('x') as f:json.dump(m,f,ensure_ascii=False,indent=2);f.write('\n')
 verify(P)

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--build',action='store_true');g.add_argument('--verify',type=Path);a=ap.parse_args()
 if a.build:build()
 else:verify(a.verify)
