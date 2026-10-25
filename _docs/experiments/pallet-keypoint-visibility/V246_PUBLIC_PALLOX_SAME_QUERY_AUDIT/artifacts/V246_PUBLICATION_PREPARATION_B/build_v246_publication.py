"""Root-only V246 publication copy; exact owned tops and TRAIN22 certificates.

No model/teacher computation, original metadata traversal, image decode or Git.
Original absolute pointers remain inert; source-map supplies portable paths.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'publication_root_preparation';H=R/'V246_PUBLICATION_PREPARATION_B';OUT=R/'V246_PUBLICATION_PACKAGE'
SELECT={'path':str(R/'V246_ROOT_PALLOX_TRAIN/train_pair_certificate_metadata_v2/COMPLETE.json'),'sha256':'15533dcad2a45de544d17b56d737170fb64fe869792c343d39d6ec33540fb488','size_bytes':2896}
PROBE={'path':str(R/'V246_ROOT_PALLOX_TRAIN/same_q_conditional_probe_v2/COMPLETE.json'),'sha256':'65adef16f5c39f9a2b96d52a567ad582498e98956ff9318570c69687b65c7659','size_bytes':12289}
CERT_ROOT=P/'query_visibility_controlled_real_20261022_v66/public_pallet/SYMMETRIC_ALL50_V2'
HELPER={'path':str(P/'query_visibility_controlled_real_20261022_v66/contract/paired_truth.py'),'sha256':'649c93b98085289bb1aa797b3beba141423f5ffbeb055b5c7f2f571e208f0c14'}
CAP=95_000_000
TEXT={'.json','.jsonl','.csv','.md','.html','.py','.txt','.log'}
FORBIDDEN=('v112','__pycache__','.pyc','current223','publicheld','private_evaluation','oldval','v177','v178','source_holdout','v182')

def lexical(p):
 p=Path(p);assert p.is_absolute() and '..' not in p.parts and not any(x in str(p).casefold() for x in FORBIDDEN)
 return p

def safe(p):
 p=lexical(p)
 for a in reversed((p,*p.parents)):assert not a.is_symlink()
 return p

def sha(b):return hashlib.sha256(b).hexdigest()

def read(p):
 with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def identity(b):
 p=lexical(b['path']);d=b['sha256'];assert len(d)==64 and all(c in '0123456789abcdef' for c in d)
 return str(p),d

def checked(b):
 identity(b)
 for key in ('bytes','size_bytes'):
  if key in b:assert type(b[key]) is int and 0<=b[key]<CAP
 raw=read(b['path']);assert len(raw)<CAP and sha(raw)==b['sha256']
 for key in ('bytes','size_bytes'):
  if key in b:assert len(raw)==b[key]
 return raw

def rejected_name(name):return any(x in name.casefold() for x in (*FORBIDDEN,'cache'))

def leaves(root):
 for name in sorted(os.listdir(safe(root))):
  if rejected_name(name):continue  # before stat/symlink/leaf access
  p=lexical(root/name)
  if p.suffix.lower() in ('.npy','.npz','.pt','.pth','.safetensors'):raise ValueError('Only explicit external TRAIN22 certificates may be binary arrays')
  p=safe(p)
  if p.is_dir():yield from leaves(p)
  else:
   assert p.is_file() and p.suffix.lower() in TEXT|{'.png'}
   yield p

def write(rel,raw):
 rp=PurePosixPath(rel);assert not rp.is_absolute() and '..' not in rp.parts and '\\' not in rel
 p=safe(OUT/rel);p.parent.mkdir(parents=True,exist_ok=True)
 with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:f.write(raw)

def jsave(rel,obj):write(rel,(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode())

def main(entry_path):
 entry_bytes=read(entry_path);e=json.loads(entry_bytes)
 assert e.get('approved_publication_copy') is True and e.get('explicit_TRAIN22_certificates_authorized') is True and e.get('automatic_retry') is False
 assert identity(e['builder'])==(str(H/'build_v246_publication.py'),sha(read(__file__)))
 selected=json.loads(checked(SELECT));probe=json.loads(checked(PROBE))
 assert selected['completed'] is True and selected['PASS'] is True and selected['stage']=='select'
 assert probe['completed'] is True and probe['PASS'] is True and probe['stage']=='probe'
 assert identity(probe['selected_metadata_complete'])==identity(SELECT) and probe['native_labels_created']==0 and probe['training_admission'] is False
 white=json.loads(checked(selected['certificate_payload_whitelist']))
 certs=white['files'];assert white['TRAIN_only'] is True and white['pair_count']==22 and len(certs)==22 and len({identity(b) for b in certs})==22
 assert {identity(b) for b in certs}=={identity(b) for b in probe['certificate_whitelist']}
 for b in certs:
  p=lexical(b['path']);assert p.parent==CERT_ROOT and p.suffix=='.npz'
  if 'original_relative_path' in b:assert b['original_relative_path']=='SYMMETRIC_ALL50_V2/'+p.name
 readme=checked(e['README']);status_bytes=checked(e['status']);status=json.loads(status_bytes)
 for key in ('final_outputs_ready','pipeline_QA_PASS','same_q_probe_complete','SOURCE_VAL_input_metadata_read_disclosed'):assert status[key] is True
 assert status['new_training_updates']==status['current_or_held_payloads']==status['native_labels_created']==0 and status['model_adopted'] is False and status['training_admission'] is False
 assert not safe(OUT).exists()
 # First-level enumeration admits only V246 names; reject cache names before stat.
 roots=[]
 for name in sorted(os.listdir(safe(R))):
  if not name.startswith('V246') or name=='V246_PUBLICATION_PACKAGE' or rejected_name(name):continue
  if name.endswith(('.nul',)) or name in ('V246_MAIN_PREPARATION.json','V246_STAGED_QA.json','V246_COMMIT_QA.json'):continue
  roots.append(safe(R/name))
 OUT.mkdir();source_map=[];copied={}
 def copy(src,dst,b=None):
  p=lexical(src)
  if dst in copied:assert copied[dst]==str(p);return
  raw=checked(b) if b else read(p);assert len(raw)<CAP
  write(dst,raw);copied[dst]=str(p);source_map.append({'path':dst,'source_path_provenance_only':str(p),'source_sha256':sha(raw),'source_bytes':len(raw),'encoding':'identity'})
 for root in roots:
  if root.is_dir():
   for src in leaves(root):copy(src,'artifacts/'+src.relative_to(R).as_posix())
  else:
   assert root.suffix.lower() in TEXT
   copy(root,'artifacts/'+root.relative_to(R).as_posix())
 for b in certs:copy(b['path'],'certificates/SYMMETRIC_ALL50_V2/'+Path(b['path']).name,b)
 copy(HELPER['path'],'dependencies/paired_truth.py',HELPER)
 for b in e.get('portable_tools',[]):
  p=lexical(b['path']);assert p.parent==R and p.suffix=='.py'
  copy(p,p.name,b)
 for dst,b,raw in [('README_KO.md',e['README'],readme),('STATUS.json',e['status'],status_bytes)]:
  write(dst,raw);source_map.append({'path':dst,'source_path_provenance_only':b['path'],'source_sha256':sha(raw),'source_bytes':len(raw),'encoding':'identity'})
 notice=read(R/'V240_PUBLICATION_NOTICES.md')
 write('THIRD_PARTY_NOTICES.md',notice+b'\nV246 includes selected KU Leuven Pallox public-source derivative certificates and four already-generated SOURCE gallery PNG copies. Dataset: Yanming Wu and Eric Demeester, DOI10.48804/DJLDFS, recorded CC-BY-4.0. No official weights, original mixed source manifests, raw depth/mesh/pose or held RGB are redistributed by this builder. Conditional teacher support is not native gold.\n')
 paths={}
 for row in source_map:paths.setdefault(row['source_path_provenance_only'],row['path'])
 jsave('PUBLICATION_SOURCE_MAP.json',{'files':source_map,'original_path_to_package_relative_path':paths,'encoding_policy':'identity; no gzip or source relinking','absolute_pointers_are_provenance_only':True,'portable_original_path_fallback_allowed':False,'exact22certificate_only':True,'original_source_or_hugecache_pointer_following':0,'old_frozen_sources_changed':False})
 jsave('EXCLUSIONS.json',{'original_mixed_metadata_not_copied':True,'raw_RGB_GT_teacher_pointers_not_followed':True,'exact22certificate_only':True,'no_weights_or_other_arrays':True,'R_scope':'first-level names prefix V246 only, cache/v112/pycache rejected before stat; no recursive P','note':'Owned gallery PNG copies only. Certificate NPZs are the explicit derived TRAIN22 allowlist, not raw GT or full mixed source data.'})
 rows=[]
 for root,dirs,files in os.walk(safe(OUT),followlinks=False):
  for name in dirs:safe(Path(root)/name)
  for name in sorted(files):
   p=safe(Path(root)/name);raw=read(p);rows.append({'path':p.relative_to(OUT).as_posix(),'sha256':sha(raw),'bytes':len(raw)})
 rows.sort(key=lambda x:x['path'])
 jsave('MANIFEST.json',{'schema':'V246_PUBLIC_PALLOX_SAME_QUERY_AUDIT_PUBLICATION','files':rows,'file_count':len(rows),'total_bytes':sum(row['bytes'] for row in rows),'scientific_status':status,'entry':{'path':str(lexical(entry_path)),'sha256':sha(entry_bytes)},'select_complete':SELECT,'probe_complete':PROBE,'publication_is_not_model_success':True})
 print(json.dumps({'completed':True,'PASS':True,'files':len(rows),'bytes':sum(row['bytes'] for row in rows),'certificate_count':22,'manifest_sha256':sha(read(OUT/'MANIFEST.json'))}))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);main(ap.parse_args().entry)
