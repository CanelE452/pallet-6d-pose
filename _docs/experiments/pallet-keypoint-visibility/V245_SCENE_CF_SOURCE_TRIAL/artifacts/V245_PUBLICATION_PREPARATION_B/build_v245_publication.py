"""Explicit V245 publication; Root entry required. No model execution or Git.

Only the eight named V245 directories and exact COMPLETE-bound outputs are
copied. All files retain identity encoding. Absolute pointers are provenance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=R/'V245_PUBLICATION_PREPARATION_B'
OUT=R/'V245_PUBLICATION_PACKAGE'
TOPS=('V245_ROOT_SCENE_CF_ACTUAL','V245_SCENE_CF_ENTRY_PREPARATION_A',
 'V245_FEATURE_ENTRY_ASSEMBLY_B','V245_ENTRY_AND_REPORT_PREPARATION_C',
 'V245_SOURCE_REPORT_C','V245_FIXED_GALLERY_A','V245_QUERY_ERROR_JOIN_A',
 'V245_TRAIN_COMPLETION_QA_B')
FROOT=R/'V242_ROOT_SCENE_CF/cf_canonical_features_v1'
TROOT=R/'V243_ROOT_SCENE_CF_TRAINING/scene_CF_source_fit_v1'
FEATURE_COMPLETE={'path':str(FROOT/'COMPLETE.json'),'sha256':'4fa17f8b96ece58680ebdeab59baca5c6a6ec900822f856c48dbc85a6baa4775','size_bytes':16569}
FIT_COMPLETE={'path':str(TROOT/'COMPLETE.json'),'sha256':'1b3a0bf4c6e0581643a8863d127a55b1b2ea7866879ecbb40560ba7cf2805c66','size_bytes':15809}
FEATURE_QA_CODE={'path':str(R/'V242_CF_FEATURE_PACKET_QA_C/verify_features.py'),'sha256':'7541bdb58d6582a301df5857e7053240759021bc4a28c6c7a12f5e1b37288a17'}
FEATURE_QA={'path':str(R/'V242_CF_FEATURE_PACKET_QA_C/actual_review_v1/REVIEW.json'),'sha256':'15fb0b188bd9ecafb180c2b0f649f6058ae4c8e67d0cedf815c89c9802ce7a08','size_bytes':8714}
FEATURE_NAMES={r+'_'+k+'.npy' for r in ('COVER','ADJACENT_SHAM') for k in ('BASE1186','NODE_CLS','CONTEXT_CLS')}|{'RELATIVE_XY8.npy','PARENT_INDICES.npy','CONSTRUCTED.npy'}
TEXT_SUFFIXES={'.json','.jsonl','.csv','.md','.html','.py','.txt','.log'}
CAP=95_000_000

def lexical(p):
 p=Path(p);assert p.is_absolute() and '..' not in p.parts
 assert not any(v in str(p).casefold() for v in ('v112','__pycache__','.pyc','current223','publicheld','private_evaluation','oldval','v177','v178','source_holdout','v182'))
 return p

def safe(p):
 p=lexical(p)
 for a in reversed((p,*p.parents)):assert not a.is_symlink()
 return p

def sha(b):return hashlib.sha256(b).hexdigest()

def read(p):
 with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def identity(b):
 assert isinstance(b,dict) and {'path','sha256'}<=set(b)
 p=lexical(b['path']);d=b['sha256'];assert len(d)==64 and all(c in '0123456789abcdef' for c in d)
 if 'size_bytes' in b:assert type(b['size_bytes']) is int and b['size_bytes']>=0
 return str(p),d

def checked(b):
 identity(b)
 # Metadata cap is checked before any leaf payload access.
 if 'size_bytes' in b:assert b['size_bytes']<CAP
 data=read(b['path']);assert len(data)<CAP and sha(data)==b['sha256']
 if 'size_bytes' in b:assert len(data)==b['size_bytes']
 return data

def leaves(root):
 for name in sorted(os.listdir(safe(root))):
  # Reject before touching the candidate leaf; never descend into a cache.
  if any(v in name.casefold() for v in ('v112','__pycache__','.pyc')):continue
  p=lexical(root/name)
  if p.suffix.lower() in ('.npy','.npz','.pt','.pth','.safetensors'):raise ValueError('Payload outside explicit COMPLETE-bound allowlist: '+str(p))
  p=safe(p)
  if p.is_dir():yield from leaves(p)
  else:
   assert p.suffix.lower() in TEXT_SUFFIXES|{'.png'} and p.is_file()
   yield p

def write(rel,b):
 rp=PurePosixPath(rel);assert not rp.is_absolute() and '..' not in rp.parts and '\\' not in rel
 p=safe(OUT/rel);p.parent.mkdir(parents=True,exist_ok=True)
 with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(b)

def jsave(rel,obj):write(rel,(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode())

def main(entry_path):
 eb=read(entry_path);e=json.loads(eb)
 assert e.get('approved_publication_copy') is True and e.get('automatic_retry') is False
 assert identity(e['builder'])==(str(H/'build_v245_publication.py'),sha(read(__file__)))
 f=json.loads(checked(FEATURE_COMPLETE));t=json.loads(checked(FIT_COMPLETE))
 assert f['completed'] and f['PASS'] and t['completed'] and t['PASS']
 assert set(f['features'])==FEATURE_NAMES and t['CF_feature_complete']['sha256']==FEATURE_COMPLETE['sha256']
 assert t['CF_feature_QA']['sha256']==FEATURE_QA['sha256']
 assert t['new_optimizer_updates']=={'BASELINE_V239_CANONICAL_CONTEXT':0,'TREAT_SCENE_CF_MATCHED_DATA':1000}
 readme=checked(e['README']);status_bytes=checked(e['status']);status=json.loads(status_bytes)
 assert status['final_outputs_ready'] is True and status['pipeline_QA_PASS'] is True and status['SOURCE_gate_PASS'] is False
 assert status['new_training_updates']==1000 and status['current_or_held_payloads']==0 and status['model_adopted'] is False
 # None of these inert references are followed for copy, stat, hash or discovery.
 excluded=[{'category':'CF_input_not_published','binding':b} for b in f['data_inputs'].values()]
 excluded += [{'category':'original_dependency_not_recopied','binding':b} for b in [f['samples_npz'],f['official_base'],*t['original_features'].values(),t['geometry_npz'],*t['CF_proof_inputs'].values()]]
 excluded_paths={identity(x['binding'])[0] for x in excluded}
 assert not safe(OUT).exists();OUT.mkdir();source_map=[];copied={}
 def copy(src,dst,binding=None):
  src=lexical(src);assert str(src) not in excluded_paths
  if dst in copied:
   assert copied[dst]==str(src);return
  data=checked(binding) if binding else read(src);assert len(data)<CAP
  write(dst,data);copied[dst]=str(src)
  source_map.append({'path':dst,'source_path_provenance_only':str(src),'source_sha256':sha(data),'source_bytes':len(data),'encoding':'identity'})
 def artifact(b,root):
  p=lexical(b['path']);assert p.parent==root
  copy(p,'artifacts/'+p.relative_to(R).as_posix(),b)
 for top in TOPS:
  for src in leaves(R/top):copy(src,'artifacts/'+src.relative_to(R).as_posix())
 artifact(FEATURE_COMPLETE,FROOT)
 for name,b in f['features'].items():
  assert Path(b['path']).name==name and Path(b['path']).suffix=='.npy';artifact(b,FROOT)
 artifact(f['canonical_sharing_QA_binding'],FROOT)
 artifact(FIT_COMPLETE,TROOT)
 fit_leaves=[*t['checkpoints'].values(),*t['logs'].values(),*t['outputs'].values(),t['CF_FIT_output'],t['summary'],t['freeze'],t['global_feature_QA'],t['draw_ledger'],t['baseline_reuse']]
 for b in fit_leaves:artifact(b,TROOT)
 for b in (FEATURE_QA_CODE,FEATURE_QA):
  p=lexical(b['path']);copy(p,'artifacts/'+p.relative_to(R).as_posix(),b)
 # Optional small history is explicit; no parent V244 package is traversed.
 if e.get('include_previous_publication_QA') is True:
  for src in leaves(R/'V244_PUBLICATION_VERIFY_C'):copy(src,'artifacts/'+src.relative_to(R).as_posix())
 for b in e.get('history_bindings',[]):
  p=lexical(b['path']);assert p.is_relative_to(R) and p.name.startswith('V244') and p.suffix in TEXT_SUFFIXES
  copy(p,'artifacts/'+p.relative_to(R).as_posix(),b)
 for src in leaves(H):copy(src,'artifacts/'+src.relative_to(R).as_posix())
 for b in e.get('portable_tools',[]):
  p=lexical(b['path']);assert p.parent==R and p.suffix=='.py'
  copy(p,p.name,b)
 write('README_KO.md',readme)
 write('STATUS.json',status_bytes)
 for dst,b,data in [('README_KO.md',e['README'],readme),('STATUS.json',e['status'],status_bytes)]:source_map.append({'path':dst,'source_path_provenance_only':b['path'],'source_sha256':sha(data),'source_bytes':len(data),'encoding':'identity'})
 notice=R/'V240_PUBLICATION_NOTICES.md'
 copy(notice,'THIRD_PARTY_NOTICES.md')
 jsave('LARGE_CACHE_EXCLUSIONS.json',{'saved_bindings_only_no_leaf_access':True,'files':excluded,'note':'Exclusions and absolute pointers are provenance only. No recursive V242 parent scan.'})
 jsave('PUBLICATION_SOURCE_MAP.json',{'files':source_map,'encoding_policy':'identity: no gzip or relinking','absolute_pointers_are_provenance_only':True,'excluded_cache_leaf_access':0,'private_current_held_payloads':0,'old_frozen_sources_changed':False})
 rows=[]
 # Package itself is a newly owned, closed output; this is not source discovery.
 for root,dirs,files in os.walk(safe(OUT),followlinks=False):
  for name in dirs:safe(Path(root)/name)
  for name in sorted(files):
   p=safe(Path(root)/name);data=read(p);rows.append({'path':p.relative_to(OUT).as_posix(),'sha256':sha(data),'bytes':len(data)})
 rows.sort(key=lambda x:x['path'])
 jsave('MANIFEST.json',{'schema':'V245_SCENE_CF_SOURCE_TRIAL_PUBLICATION','files':rows,'file_count':len(rows),'total_bytes':sum(x['bytes'] for x in rows),'scientific_status':status,'entry':{'path':str(lexical(entry_path)),'sha256':sha(eb)},'feature_complete':FEATURE_COMPLETE,'fit_complete':FIT_COMPLETE,'publication_is_not_model_success':True})
 print(json.dumps({'completed':True,'PASS':True,'files':len(rows),'bytes':sum(x['bytes'] for x in rows),'manifest_sha256':sha(read(OUT/'MANIFEST.json'))}))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);main(ap.parse_args().entry)
