#!/usr/bin/env python3
"""Additional saved-mask diagnostic only: continuous ref four raw96 taps."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=Path(__file__).parent;S=R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/pairs_v1'
expected_complete='8958c47fc5cf65d592346d5b9671c592ad470e799cfdcdb926e6e2788ca2fc13'
access=[]
def safe(p):
 p=Path(p);assert p.is_absolute() and '..' not in p.parts
 for word in ('v112','__pycache__','current223','publicheld','oldval','source_holdout','v177','v178','v182'):assert word not in str(p).lower()
 assert p.parent==S
 for q in [p]+list(p.parents):assert not q.is_symlink()
 return p
def checked(b):
 p=safe(b['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==b['sha256']
 if 'size_bytes' in b:assert len(raw)==b['size_bytes']
 access.append({'path':str(p),'sha256':b['sha256'],'size_bytes':len(raw)})
 return p
p=safe(S/'COMPLETE.json');raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected_complete;c=json.loads(raw)
assert c['PASS'] is True and c['completed'] is True
paths={n:checked(c['outputs'][n]) for n in ('PAIR_METADATA.jsonl','success.npy','PAIR_ALPHA_PACKBITS.npy','parent_indices.npy')}
success=np.load(paths['success.npy'],allow_pickle=False);parents=np.load(paths['parent_indices.npy'],allow_pickle=False)
pack=np.load(paths['PAIR_ALPHA_PACKBITS.npy'],mmap_mode='r',allow_pickle=False)
assert success.shape==parents.shape==(5406,) and pack.dtype==np.uint8 and pack.shape==(5406,2,1152)
counts={'successful_pairs':0,'COVER_all4_true':0,'SHAM_all4_false':0,'both_pass':0,'continuous_taps_outside96':0};failure=[];byfamily={}
for j,line in enumerate(paths['PAIR_METADATA.jsonl'].read_text().splitlines()):
 if not success[j]:continue
 m=json.loads(line);assert m['pair_index']==j and m['parent_sample_index']==int(parents[j]) and m['success'] is True
 q=m['original_query_xy'];r=m['teacher_reference_xy'];L=m['original_L']
 u=(float(r[0])-float(q[0]))/float(L)*96.+47.5;v=(float(r[1])-float(q[1]))/float(L)*96.+47.5
 x=math.floor(u);y=math.floor(v);tapxy=[[x,y],[x+1,y],[x,y+1],[x+1,y+1]]
 inside=all(0<=xx<96 and 0<=yy<96 for xx,yy in tapxy)
 a=np.unpackbits(np.asarray(pack[j]),axis=1,bitorder='big').reshape(2,96,96).astype(bool)
 vals=[[bool(a[k,yy,xx]) for xx,yy in tapxy] for k in (0,1)] if inside else [None,None]
 cp=inside and all(vals[0]);sp=inside and not any(vals[1])
 counts['successful_pairs']+=1;counts['COVER_all4_true']+=int(cp);counts['SHAM_all4_false']+=int(sp);counts['both_pass']+=int(cp and sp);counts['continuous_taps_outside96']+=int(not inside)
 f=byfamily.setdefault(str(m['family']),{'successful_pairs':0,'COVER_all4_true':0,'SHAM_all4_false':0,'both_pass':0})
 f['successful_pairs']+=1;f['COVER_all4_true']+=int(cp);f['SHAM_all4_false']+=int(sp);f['both_pass']+=int(cp and sp)
 for k,(role,passed) in enumerate((('COVER_E',cp),('SHAM_D',sp))):
  if not passed:failure.append({'pair_index':j,'parent_sample_index':int(parents[j]),'row_id':m['row_id'],'family':m['family'],'role':role,'continuous_RAW96_u':u,'continuous_RAW96_v':v,'tap_xy':json.dumps(tapxy),'alpha_flags':json.dumps(vals[k]),'in96':inside,'original_nearest3x3_teacher_pass':True})
assert counts['successful_pairs']==5367
out=H/'CONTINUOUS_REFERENCE_4TAP_FAILURES.csv';assert not out.exists()
with out.open('w',newline='') as f:
 fields=['pair_index','parent_sample_index','row_id','family','role','continuous_RAW96_u','continuous_RAW96_v','tap_xy','alpha_flags','in96','original_nearest3x3_teacher_pass'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(failure)
d={'completed':True,'PASS':True,'meaning':'Execution/arithmetic QA passed; geometric condition counts are reported separately, no automatic target/data change.','root_request':'Additional continuous refF64 itself four raw96 tap support audit after original source-nearest3x3 validation.','complete':{'path':str(S/'COMPLETE.json'),'sha256':expected_complete},'counts':counts,'by_family':byfamily,'failed_role_rows':len(failure),'failed_unique_parents':len({r['parent_sample_index'] for r in failure}),'all_continuous_ref_conditions_pass':not failure,'failure_CSV':{'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'size_bytes':out.stat().st_size},'access':access,'limits':['No original RGB/valid/GT/donor/prefix/model/current/held read. Only same saved metadata/success/parent/packed-alpha outputs consumed.','Query/L/reference provenance was independently matched to original SOURCE metadata in prior actual_review_v2; this diagnostic uses those sealed values.','Original 3x3 teacher remains the declared construction rule. Additional continuous-ref support is separate evidence, not new target authority or automatic row selection.','No row deletion, target change, regeneration, forward, fit, threshold or model selection.']}
p=H/'CONTINUOUS_REFERENCE_4TAP_QA.json';assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'completed':True,'counts':counts,'by_family':byfamily,'failed_role_rows':len(failure),'failed_unique_parents':d['failed_unique_parents'],'all_continuous_ref_conditions_pass':not failure,'review_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
