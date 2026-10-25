"""Package the bounded V254 saved Source CF response audit and six replay inputs."""
from pathlib import Path
import argparse,hashlib,json,os,re,stat

R=Path(__file__).absolute().parent
H=R/'V254_ROOT_SAVED_CF_RESPONSE'
P=R/'V254_PUBLICATION_PACKAGE'
TREE={
 'V254_ROOT_SAVED_CF_RESPONSE':{'saved_cf_response_v1':{}},
 'V254_SAME_QUERY_CF_READINESS_A':{},
 'V254_REVIEW_RECONCILIATION_A':{},
 'V254_SAVED_CF_RESPONSE_B':{},
 'V254_PREVIOUS_PROGRESS_AND_CF_QA_C':{'actual_review_v1':{}},
 'V254_PUBLIC_REPLAY_ROOT':{},
 'V254_VISUAL_READOUT_QA_A':{},
}

def safe(p):
 p=Path(p)
 if not p.is_absolute() or '..' in p.parts or any(w in str(p).lower() for w in ('v112','cache','.pyc')):raise ValueError(str(p))
 for a in [*reversed(p.parents),p]:
  try:t=os.lstat(a)
  except FileNotFoundError:continue
  if stat.S_ISLNK(t.st_mode):raise ValueError('Artifact symlink')
 return p
def read(p):return safe(p).read_bytes()
def sha(b):return hashlib.sha256(b).hexdigest()
def obj(p):return json.loads(read(p))

def verify(pkg):
 pkg=safe(pkg);m=obj(pkg/'MANIFEST.json');assert m['completed'] and m['PASS']
 for row in m['files']:
  rel=Path(row['relative_path']);assert not rel.is_absolute() and '..' not in rel.parts
  b=read(pkg/rel);assert sha(b)==row['sha256'] and len(b)==row['size_bytes']
 links=0
 for name in ('README_KO.md','ROOT_READOUT_KO.html'):
  p=pkg/name;t=read(p).decode()
  found=re.findall(r'(?:href|src)=[\"\']([^\"\']+)',t) if p.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',t)
  for target in found:
   if target.startswith(('http:','https:','#','mailto:')):continue
   assert safe(p.parent/target.split('#')[0]).is_file(),target;links+=1
 print(json.dumps({'completed':True,'PASS':True,'leaves':len(m['files']),'bytes':sum(x['size_bytes'] for x in m['files']),'links':links}))

def build():
 assert obj(H/'saved_cf_response_v1/COMPLETE.json')['PASS']
 assert obj(R/'V254_PREVIOUS_PROGRESS_AND_CF_QA_C/REPORT_REVIEW.json')['PASS']
 assert not safe(P).exists();P.mkdir();rows=[];seen=set()
 def take(src,rel):
  assert rel not in seen;seen.add(rel);b=read(src);p=safe(P/rel);p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b)
  rows.append({'relative_path':rel,'sha256':sha(b),'size_bytes':len(b),'original_path':str(src)})
 def direct(d,prefix,children):
  for name in sorted(os.listdir(safe(d))):
   src=safe(d/name)
   if src.is_dir():
    assert name in children,name;direct(src,prefix+'/'+name,children[name])
   elif src.is_file():
    assert src.suffix in ('.py','.md','.json','.jsonl','.csv','.npz','.png','.html'),name
    take(src,prefix+'/'+name)
   else:raise ValueError('Nonregular artifact')
 for name,ch in TREE.items():direct(R/name,'artifacts/'+name,ch)
 deps={'schema':'V254_EXACT_SIX_PUBLIC_SAVED_INPUT_REPLAY_V1','inputs6':{},'core_outputs':[]}
 protocol=obj(H/'PROTOCOL_BEFORE_SOURCE_PAIR_ROWS.json')
 for key,row in protocol['inputs6'].items():
  src=Path(row['path']);b=read(src);assert len(b)==row['size_bytes'] and sha(b)==row['sha256']
  rel='inputs/'+src.name;take(src,rel);deps['inputs6'][key]={'relative_path':rel,'sha256':sha(b),'size_bytes':len(b)}
 code=R/'V254_SAVED_CF_RESPONSE_B/audit_saved_cf.py';b=read(code)
 deps['code']={'relative_path':'artifacts/V254_SAVED_CF_RESPONSE_B/audit_saved_cf.py','sha256':sha(b),'size_bytes':len(b)}
 complete=obj(H/'saved_cf_response_v1/COMPLETE.json')
 for name,row in complete['outputs'].items():
  if name=='COMPLETE.json':continue
  src=H/'saved_cf_response_v1'/name;b=read(src)
  assert sha(b)==row['sha256'] and len(b)==row['size_bytes']
  deps['core_outputs'].append({'name':name,'relative_path':'artifacts/V254_ROOT_SAVED_CF_RESPONSE/saved_cf_response_v1/'+name,'sha256':sha(b),'size_bytes':len(b)})
 depbytes=(json.dumps(deps,ensure_ascii=False,indent=2)+'\n').encode();q=safe(P/'runtime_dependencies.json')
 with q.open('xb') as f:f.write(depbytes)
 rows.append({'relative_path':'runtime_dependencies.json','sha256':sha(depbytes),'size_bytes':len(depbytes)})
 for name in ('PAIR_SUPPORT.png','SAVED_CF_RESPONSE.png','FIXED_PREDICTION_PATTERNS.png','ROOT_READOUT_KO.html'):take(H/name,name)
 take(H/'PUBLIC_README_KO.md','README_KO.md')
 for name in ('build_v254_readout.py','package_v254_results.py','stage_v254_main.py'):take(R/name,'tools/'+name)
 m={'schema':'V254_SAVED_SOURCE_COUNTERFACTUAL_RESPONSE_PUBLICATION_V1','completed':True,'PASS':True,'files':rows,'phase':'ACTUAL_FROZEN_SAVED_SOURCE_CF_RESPONSE','whole_Source_inputs_contain_DEV_bytes':True,'primary_response':'all-member FIT only; all candidate, unmatched and S/U denominators retained','new_training':0,'new_model_forward_or_CP':0,'RGB_or_original_GT_decode':0,'new_bank_or_threshold_choice':0,'current_or_held_payload_read':0,'current_or_held_used_for_recipe_choice':False,'PnP':0,'model_adopted':False,'goal_status':'active_unsolved','limits':'Replay regenerates saved Source response arithmetic; no original renderer, teacher, encoder or private-evaluation reproduction. Serialized original paths are inert.'}
 with safe(P/'MANIFEST.json').open('x') as f:json.dump(m,f,ensure_ascii=False,indent=2);f.write('\n')
 verify(P)

if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--build',action='store_true');g.add_argument('--verify',type=Path);a=p.parse_args()
 if a.build:build()
 else:verify(a.verify)
