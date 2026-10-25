"""Package explicit new V249 SOURCE diagnostic directories. No model/git mutation."""
from pathlib import Path
import argparse, hashlib, json, os, re
R=Path(__file__).absolute().parent
H=R/'V249_ROOT_TARGET_RGB_DIAGNOSTIC'
P=R/'V249_PUBLICATION_PACKAGE'
DIRS={
 'V249_ROOT_TARGET_RGB_DIAGNOSTIC':('oracle_rgb_reference_diagnostic_v1',),
 'V249_TARGET_IDENTITY_CODE_A':(),
 'V249_SOURCE_TARGET_READINESS_B':(),
 'V249_RGB_REFERENCE_DIAGNOSTIC_B':(),
 'V249_PREVIOUS_PROGRESS_AND_LIMITS_QA_C':(),
 'V249_RGB_REFERENCE_CODE_QA_C':(),
 'V249_PUBLISHED_DEPENDENCIES_A':(),
 'V249_PUBLIC_REPLAY_A':(),
 'V249_ROOT_PUBLIC_REPLAY_V1':(),
 'V249_ACTUAL_SOURCE_QA_C':('actual_review_v1',),
}

def safe(p):
 p=Path(p)
 if not p.is_absolute() or '..' in p.parts or any(x in str(p).lower() for x in ('v112','__pycache__','.pyc')):raise ValueError(str(p))
 if any(a.is_symlink() for a in [p,*p.parents]):raise ValueError('Symlink artifact')
 return p

def read(p):return safe(p).read_bytes()
def sha(b):return hashlib.sha256(b).hexdigest()

def verify(p):
 p=safe(p);m=json.loads(read(p/'MANIFEST.json'))
 for row in m['files']:
  rel=Path(row['relative_path'])
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Relative package leaf only')
  b=read(p/rel)
  if sha(b)!=row['sha256'] or len(b)!=row['size_bytes']:raise ValueError('Package leaf mismatch')
 links=0
 for name in ('README_KO.md','ROOT_READOUT_KO.html'):
  f=p/name;s=read(f).decode();rr=re.findall(r'(?:href|src)=[\"\']([^\"\']+)',s) if f.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',s)
  for link in rr:
   if link.startswith(('http:','https:','#','mailto:')):continue
   if not safe(f.parent/link.split('#')[0]).is_file():raise ValueError('Broken public entrypoint link '+link)
   links+=1
 result={'completed':True,'PASS':True,'leaves':len(m['files']),'bytes':sum(r['size_bytes'] for r in m['files']),'entrypoint_links':links,'original_pointer_fallback':False,'model_execution':0}
 print(json.dumps(result));return result

def build():
 qa=json.loads(read(R/'V249_ACTUAL_SOURCE_QA_C/REPORT_REVIEW.json'))
 if qa.get('PASS') is not True:raise ValueError('Final report QA required')
 if safe(P).exists():raise ValueError('Fresh package only')
 P.mkdir();files=[];seen=set()
 def take(src,rel):
  if rel in seen:raise ValueError('Duplicate destination')
  seen.add(rel);b=read(src);dst=safe(P/rel);dst.parent.mkdir(parents=True,exist_ok=True)
  with dst.open('xb') as f:f.write(b)
  files.append({'relative_path':rel,'original_path':str(src),'sha256':sha(b),'size_bytes':len(b)})
 def direct(d,prefix,children=()):
  for name in sorted(os.listdir(safe(d))):
   src=safe(d/name)
   if src.is_dir():
    if name not in children:raise ValueError('Undeclared child '+name)
    direct(src,prefix+'/'+name)
   elif src.is_file():
    if src.suffix not in ('.py','.md','.json','.jsonl','.csv','.npz','.npy','.png','.html'):raise ValueError('Unexpected leaf')
    take(src,prefix+'/'+name)
   else:raise ValueError('Nonregular artifact')
 for name,ch in DIRS.items():direct(R/name,'artifacts/'+name,ch)
 for name in ('AUC_COMPARISON.csv','ORACLE_REFERENCE_AUC.png','REFERENCE_SCORE_DISTRIBUTIONS.png','REFERENCE_MARGIN_DISTRIBUTIONS.png','ROOT_READOUT_KO.html'):
  take(H/name,name)
 take(H/'PUBLIC_README_KO.md','README_KO.md')
 for name in ('build_v249_readout.py','diagnose_v249_saved_baseline_rank.py','package_v249_results.py'):take(R/name,'tools/'+name)
 m={'schema':'V249_EXPLICIT_SOURCE_ORACLE_DIAGNOSTIC_PUBLICATION_V1','completed':True,'PASS':True,'files':files,
    'phase':'SOURCE_FIT_RGB_REFERENCE_TEACHER_IDENTITY_ORACLE_DIAGNOSTIC','SOURCE_FIT_statistics_computed':True,'optimizer_updates':0,'model_forward':0,
    'current_or_held_inference':0,'PnP':0,'model_adopted':False,'reference_retrieval_deployable':False,
    'absolute_metadata_paths':'Provenance only; package verifier never follows original pointers.',
    'primary_replay':'Explicit public-repo wrapper follows pinned repository-relative SOURCE3 inputs. No RGB/encoder/weight needed.',
    'posthoc':'V248 baseline rank appendix was defined after primary scores, before reading its baseline scores; kept separate.'}
 with safe(P/'MANIFEST.json').open('x') as f:json.dump(m,f,ensure_ascii=False,indent=2);f.write('\n')
 verify(P)

if __name__=='__main__':
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--build',action='store_true');g.add_argument('--verify',type=Path);a=parser.parse_args()
 if a.build:build()
 else:verify(a.verify)
