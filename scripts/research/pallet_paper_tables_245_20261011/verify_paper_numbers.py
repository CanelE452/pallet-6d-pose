"""Independent source-cell and altitude-bin audit of manuscript numbers."""
from __future__ import annotations
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
DOC=Path(os.environ.get('PALLET_PAPER_OUTPUT',ROOT/'_docs/experiments/pallet_paper_tables_245_20261011')).resolve()
COUNT=0
MAX_ERROR=0.

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(actual,expected,name):
 global COUNT,MAX_ERROR
 COUNT+=1
 if isinstance(expected,(str,list,bool)):assert actual==expected,(name,actual,expected);return
 if actual is None or expected is None:assert actual is None and expected is None,(name,actual,expected);return
 error=abs(float(actual)-float(expected));MAX_ERROR=max(MAX_ERROR,error);assert error<=1e-10,(name,actual,expected,error)

def csv_rows(path):
 with path.open() as f:return list(csv.DictReader(f))
def source_cell(cell,metrics):
 filename,location=cell.split(':',1)
 if filename=='METRICS.json':
  value=metrics
  for key in location.split('.'):value=value[key]
  return value
 if filename=='TABLE_2.csv':
  backbone,method,column=location.split('/')
  rr=[r for r in csv_rows(DOC/filename) if r['backbone']==backbone and r['method']==method and r['scope']=='RECT245']
 elif filename=='TABLE_1.csv':
  scope,backbone,method,column=location.split('/')
  rr=[r for r in csv_rows(DOC/filename) if r['scope']==scope and r['backbone']==backbone and r['method']==method]
 else:raise ValueError(cell)
 assert len(rr)==1,cell
 return float(rr[0][column]) if rr[0][column] else None

def run():
 lock=read(DOC/'PAPER_NUMBERS_VERIFY_SOURCE_LOCK.json');assert lock['status'].startswith('LOCKED')
 for binding in lock['current_bindings']:assert sha(ROOT/binding['path'])==binding['sha256']
 assert not (DOC/'PAPER_NUMBERS_VERIFICATION.json').exists()
 numbers=read(DOC/'PAPER_NUMBERS.json');metrics=read(DOC/'METRICS.json')
 for backbone,p in numbers['N3_PCK10_increase_percentage_points'].items():
  a,b=[source_cell(cell,metrics) for cell in p['source_cells']]
  check(p['value'],100*(a-b),'N3_PCK_gain.'+backbone)
 for method,p in numbers['YOLO_success_RECT245_percent'].items():check(p['value'],source_cell(p['source_cell'],metrics),'YOLO_success.'+method)
 for backbone,p in numbers['square_success_Base_to_Two_stage_percent'].items():
  check(p['BASE'],source_cell(p['source_cells'][0],metrics),'Square_BASE.'+backbone)
  check(p['N3_THEN_SUBPIX'],source_cell(p['source_cells'][1],metrics),'Square_SEQ.'+backbone)
 frame=metrics['scopes']['RECT245']['groups']['YOLO::BASE']['frame_values'];p=numbers['RECT245_pallet_diagonal_px']
 diag=np.array([r['pallet_diagonal_px'] for r in frame]);assert np.isfinite(diag).all() and (diag>0).all()
 median=float(np.percentile(diag,50))
 check(p['median_px'],median,'diagonal.median');check(p['ten_px_percent_of_median'],10/median*100,'diagonal.10px_percent');check(p['frames'],245,'diagonal.N')
 assert p['source_cell']=='METRICS.json:scopes.RECT245.groups.YOLO::BASE.frame_values.pallet_diagonal_px'
 for group,p in numbers['confusion_share_among_R_gt45_percent'].items():
  fv=metrics['scopes']['RECT245']['groups'][group]['frame_values'];c=float(np.array([r['confusion_rate'] for r in fv]).sum());den=float(np.array([r['rotation_gt45_rate'] for r in fv]).sum())
  check(p['confusion_seed_mean_sum'],c,'R45share.num.'+group);check(p['R_gt45_seed_mean_sum'],den,'R45share.den.'+group)
  check(p['value'],100*c/den if den else None,'R45share.percent.'+group)
  assert p['source_cells']==[f'METRICS.json:scopes.RECT245.groups.{group}.frame_values.confusion_rate',f'METRICS.json:scopes.RECT245.groups.{group}.frame_values.rotation_gt45_rate']
 path=ROOT/'_docs/experiments/pallet_wd_hypothesis_diag_20261010/STAGE1_ROWS_REAL.jsonl.gz'
 ids=set(metrics['scopes']['RECT245']['ids']);rows=[]
 with gzip.open(path,'rt') as f:
  for line in f:
   r=json.loads(line)
   if r['method']=='N3_THEN_SUBPIX' and r['id'] in ids and r.get('backbone','YOLO')=='YOLO':rows.append(r)
 assert len(rows)==735 and len({(r['seed'],r['id']) for r in rows})==735
 labels=('<5','5–10','10–20','20–30','≥30');edges=(5,10,20,30);buckets=defaultdict(list);oracle_R_worse=0
 for row in rows:
  altitude=float(row['elevation_deg']);j=sum(altitude>=edge for edge in edges);buckets[j].append(row)
  expected_hyp={'CF_WIDTH':'long-face-front','CF_DEPTH':'short-face-front'}[row['gt_physical_long_axis']]
  check(row['hypOracle'],expected_hyp,'oracle.GTparity')
  check(row['oracle_diagnostic_only'],True,'oracle.diagnostic');check(row['inference_reference_inputs'],False,'oracle.no_inference_GT')
  assert row['oracle_definition']=='GT parity supplied; same predicted coordinates and8-corner final solver; no min-error candidate search'
  assert row['actual_pose']['ORACLE']==row['candidates'][expected_hyp]['actual_pose'],'Oracle must reuse the supplied-parity candidate'
  if row['pose']['ORACLE']['available'] and row['pose']['S0']['available'] and row['pose']['ORACLE']['rotation_deg']>row['pose']['S0']['rotation_deg']:oracle_R_worse+=1
 table=numbers['altitude_confusion_bins'];assert table['oracle']=='GT-parity reference_long_axis; not minimum error choice'
 assert table['source']=='_docs/experiments/pallet_wd_hypothesis_diag_20261010/STAGE1_ROWS_REAL.jsonl.gz'
 total=0
 for j,p in enumerate(table['bins']):
  bucket=buckets[j];frameids=sorted({r['id'] for r in bucket});check(p['label'],labels[j],'altbin.label');check(p['frames'],len(frameids),'altbin.N');check(p['frame_ids'],frameids,'altbin.IDs');total+=len(frameids)
  assert len(bucket)==3*len(frameids)
  for arm in ('S0','ORACLE'):
   hits=sum(1 for r in bucket if r['pose'][arm]['available'] and r['pose'][arm]['rotation_deg']>45 and abs(r['pose'][arm]['yaw_deg'])>=60)
   check(p[arm]['confusions'],hits,'altbin.'+arm+'.hits');check(p[arm]['prediction_count'],len(bucket),'altbin.'+arm+'.den')
   check(p[arm]['confusion_percent'],100*hits/len(bucket) if bucket else None,'altbin.'+arm+'.percent')
 check(total,245,'altbin.allframes')
 receipt=dict(status='PASS',comparisons=COUNT,max_absolute_difference=MAX_ERROR,tolerance=1e-10,
  checks=['PAPER_NUMBERS exact METRICS/CSV source cells','all245 diagonal percentile50 and10px percent','allroutes confusion/R45 numerator-denominator','five altitude bins independently counted735rows','GT-parity axis mapping/candidate reuse, never min-R selection'],
  oracle_R_worse_than_S0_rows=oracle_R_worse,rows=735,frames=245,
  source_sha256=sha(Path(__file__)),source_lock_sha256=sha(DOC/'PAPER_NUMBERS_VERIFY_SOURCE_LOCK.json'),
  inputs={name:sha(DOC/name) for name in ('PAPER_NUMBERS.json','METRICS.json','TABLE_1.csv','TABLE_2.csv')},
  existing_STAGE1_ROWS_sha256=sha(path),additional_model_forwards=0,additional_PnP_calls=0)
 with (DOC/'PAPER_NUMBERS_VERIFICATION.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
 print('PAPER_NUMBERS_SOURCE_AUDIT_PASS',COUNT,MAX_ERROR,oracle_R_worse,flush=True)
if __name__=='__main__':run()
