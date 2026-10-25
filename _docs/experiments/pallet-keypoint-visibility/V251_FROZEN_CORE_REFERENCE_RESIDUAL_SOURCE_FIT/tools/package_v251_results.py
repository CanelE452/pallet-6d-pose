"""Package exact V251 source diagnostic directories; never follow provenance pointers."""
from pathlib import Path
import hashlib,json,os,re,argparse
R=Path(__file__).absolute().parent
H=R/'V251_ROOT_REFERENCE_RESIDUAL'
P=R/'V251_PUBLICATION_PACKAGE'
DIRS={
 'V251_ROOT_REFERENCE_RESIDUAL':('matched_reference_residual_v1','reference_effect_posthoc_v1'),
 'V251_REFERENCE_DEPLOYMENT_CLOSURE_A':(),
 'V251_SMALL_HEAD_READINESS_B':(),
 'V251_PREVIOUS_PROGRESS_QA_C':(),
 'V251_FROZEN_CORE_REFERENCE_TRAINER_B':(),
 'V251_REFERENCE_TRAINER_CODE_QA_C':(),
 'V251_ACTUAL_REFERENCE_FIT_QA_C':('actual_review_v1',),
 'V251_PUBLIC_REPLAY_A':(),
 'V251_REFERENCE_EFFECT_POSTHOC_A':(),
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
 assert json.loads(read(R/'V251_ACTUAL_REFERENCE_FIT_QA_C/REPORT_REVIEW.json'))['PASS']
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
 for name in ('build_v251_readout.py','package_v251_results.py','stage_v251_main.py'):take(R/name,'tools/'+name)
 m={'schema':'V251_FROZEN_CORE_REFERENCE_RESIDUAL_SOURCE_FIT_PUBLICATION_V1','completed':True,'PASS':True,'files':rows,'phase':'ACTUAL_MATCHED_SOURCE_FIT_WITH_FROZEN_CORE','SOURCE_FIT_SYN_reference_statistics_fitted':True,'SOURCE_FIT_YCB_supervised_reference_bank_reused':True,'core_frozen_parameters':86054,'trainable_residual_parameters_each':2434,'matched_optimizer_updates_each':1000,'total_new_optimizer_updates':2000,'new_DINO_encoder_forward':0,'current_or_held_inference':0,'PnP':0,'model_adopted':False,'goal_status':'active_unsolved','absolute_metadata_paths':'Inert provenance; public saved-model inference replay explicitly rebinds only pinned repository-relative SOURCE inputs and exact package leaves.'}

 with safe(P/'MANIFEST.json').open('x') as f:json.dump(m,f,ensure_ascii=False,indent=2);f.write('\n')
 verify(P)

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--build',action='store_true');g.add_argument('--verify',type=Path);a=ap.parse_args()
 if a.build:build()
 else:verify(a.verify)
