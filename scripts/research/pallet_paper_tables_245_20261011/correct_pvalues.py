"""Repair floating equality in discrete paired bootstrap p values.

The preregistered formula, saved poses, coordinates, draw sequence and CIs stay
fixed. Integer seed decisions retain exact equality in the null comparison.
All initial statistical outputs are archived before the canonical p table is
updated. This is a numerical correction, with every changed cell recorded.
"""
from __future__ import annotations
from collections import defaultdict
import copy
import json
from pathlib import Path
import shutil
import numpy as np
from . import stats as S
from . import tables as T

DISCRETE=('PCK10','success_rate','confusion_rate','ADDsym_AUC')
ARCHIVE='STATS_NUMERIC_DIAGNOSTIC'
LOCK='NUMERICAL_AMENDMENT_SOURCE_LOCK.json'
THRESHOLDS=np.linspace(0,.1,1001)

def contribution(row,metric):
 p=row['pose'];available=bool(p.get('available'))
 if metric=='PCK10':return int(row['corner']['pck10_hits']) if available else 0
 if metric=='success_rate':return int(available and p['translation_cm']<5 and p['rotation_deg']<5)
 if metric=='confusion_rate':return int(available and p['rotation_deg']>45 and abs(p['yaw_deg'])>=60)
 if metric=='ADDsym_AUC':
  if not available:return 0
  hit=float(p['ADDsym_normalized'])<=THRESHOLDS
  # The original 1001-point trapezoid has exactly 2000 half-step units.
  return 2*int(hit.sum())-int(hit[0])-int(hit[-1])
 raise ValueError(metric)

def ratio_frame_counts(group_left,group_right,ids,metric):
 seeds=sorted({s for s,i in group_left});assert seeds==[1,2,3]
 assert {s for s,i in group_right}==set(seeds)
 numerator=[];denominator=[]
 for i in ids:
  numerator.append(sum(contribution(group_left[s,i],metric)-contribution(group_right[s,i],metric) for s in seeds))
  if metric=='PCK10':
   d=[int(group_left[s,i]['corner']['pck10_denominator']) for s in seeds]
   assert len(set(d))==1 and all(group_right[s,i]['corner']['pck10_denominator']==d[0] for s in seeds)
   denominator.append(d[0]*len(seeds))
  else:denominator.append(len(seeds)*(2000 if metric=='ADDsym_AUC' else 1))
 return np.asarray(numerator,dtype=np.int64),np.asarray(denominator,dtype=np.int64)

def exact_null_p(numerator,denominator,sessions,weights):
 labels=sorted(set(sessions));sessions=np.asarray(sessions)
 sums=np.array([numerator[sessions==s].sum() for s in labels],dtype=np.int64)
 counts=np.array([denominator[sessions==s].sum() for s in labels],dtype=np.int64)
 assert len(labels)==13 and weights.shape==(10000,13)
 sample_num=weights@sums;sample_den=weights@counts;keep=sample_den>0
 n=int(numerator.sum());d=int(denominator.sum());assert d>0
 bound=max(abs(int(sample_num.min())),abs(int(sample_num.max())))*d+2*abs(n)*int(sample_den.max())
 assert bound<np.iinfo(np.int64).max,'Integer cross products would overflow'
 if n>0:events=(sample_num<=0)|(sample_num*d>=2*n*sample_den)
 elif n<0:events=(sample_num>=0)|(sample_num*d<=2*n*sample_den)
 else:events=np.ones(len(sample_num),bool)
 count=int(events[keep].sum());b=int(keep.sum())
 return dict(delta=n/d,p_bootstrap=(1+count)/(b+1),full_integer_numerator=n,full_integer_denominator=d,
  valid_draws=b,tail_draws=count,int64_product_bound=bound,
  formula='positiveobs: drawN<=0 OR drawN*fullD>=2*fullN*drawD; negativeobs: drawN>=0 OR <=; zeroobs alltrue')

def preserve_initial():
 archive=S.DOC/ARCHIVE;assert not archive.exists(),'Initial diagnostic archive already exists'
 archive.mkdir()
 names=['METRICS.json','PAIRED.json','FAILURES.json','PAPER_NUMBERS.json','STATISTICS_EXECUTION.json','TABLES_EXECUTION.json','SUPPLEMENT_319.json','SEED_TABLES.csv']
 names +=[stem+extension for stem in ('TABLE_1','TABLE_2','TABLE_3','SUPP','SUPPLEMENT_319_TABLE_1','SUPPLEMENT_319_TABLE_2','PAIRED_TABLE_1','PAIRED_TABLE_2','PAIRED_TABLE_3') for extension in ('.csv','.md')]
 bindings=[]
 for name in names:
  path=S.DOC/name
  if path.is_file():
   target=archive/name;shutil.copy2(path,target);bindings.append(dict(path=ARCHIVE+'/'+name,sha256=S.sha(target),bytes=target.stat().st_size))
 for name in ('stats.py','tables.py','verify_stats.py'):
  path=Path(__file__).parent/name;target=archive/('ORIGINAL_'+name+'.txt');shutil.copy2(path,target)
  bindings.append(dict(path=ARCHIVE+'/'+target.name,sha256=S.sha(target),bytes=target.stat().st_size))
 shutil.copy2(S.DOC/'STATS_SOURCE_LOCK.json',archive/'ORIGINAL_STATS_SOURCE_LOCK.json')
 with (archive/'ARCHIVE_MANIFEST.json').open('x') as f:json.dump(dict(status='INITIAL_OUTPUTS_PRESERVED',files=bindings),f,ensure_ascii=False,indent=2);f.write('\n')
 return bindings

def run():
 lock=S.read(S.DOC/LOCK);assert lock['status'].startswith('LOCKED')
 assert not (S.DOC/'NUMERICAL_AMENDMENT.json').exists()
 for binding in lock['current_bindings']:assert S.sha(S.ROOT/binding['path'])==binding['sha256'],binding['path']
 assert S.sha(S.DOC/'METHOD_LOCK.json')==lock['method_lock_sha256']
 old=S.read(S.DOC/'PAIRED.json');new=copy.deepcopy(old)
 raw={p:S.rows(S.DOC/(p+'_ROWS.jsonl.gz')) for p in ('RECT','SQUARE')}
 metrics=S.read(S.DOC/'METRICS.json');grouped={}
 for population,rr in raw.items():
  groups=defaultdict(dict)
  for row in rr:
   key=row['backbone']+'::'+row['method'];k=(row['seed'],row['id']);assert k not in groups[key];groups[key][k]=row
  grouped[population]=groups
 rng=np.random.default_rng(20260917)
 weights=np.concatenate([rng.multinomial(13,np.full(13,1/13),size=100) for _ in range(100)]).astype(np.int64)
 import hashlib
 draw_sha=hashlib.sha256(weights.astype('<u2').tobytes()).hexdigest();assert draw_sha==S.REAL_DRAW_SHA
 before_hash=S.sha(S.DOC/'PAIRED.json');had_paired_tables=(S.DOC/'PAIRED_TABLE_1.csv').is_file();initial_bindings=preserve_initial();changes=[];proofs=[]
 for family,records in new['tables'].items():
  for j,record in enumerate(records):
   metric=record['metric'];scope=record['scope'];population='SQUARE' if scope=='SQUARE118' else 'RECT'
   if metric not in DISCRETE or (population=='SQUARE' and metric=='confusion_rate'):continue
   ids=metrics['scopes'][scope]['ids'];groups=grouped[population];left=groups[record['backbone']+'::'+record['left']];right=groups[record['backbone']+'::'+record['right']]
   num,den=ratio_frame_counts(left,right,ids,metric)
   if population=='RECT':
    sessions=[left[1,i]['session'] for i in ids];proof=exact_null_p(num,den,sessions,weights)
   else:proof=dict(delta=int(num.sum())/int(den.sum()),full_integer_numerator=int(num.sum()),full_integer_denominator=int(den.sum()))
   assert abs(proof['delta']-record['delta'])<=1e-12,'This repair must not change the effect estimand'
   for field in ('delta','p_bootstrap'):
    if field not in proof:continue
    previous=record.get(field);value=proof[field]
    if previous!=value:changes.append(dict(table=family,record_index=j,scope=scope,backbone=record['backbone'],left=record['left'],right=record['right'],metric=metric,field=field,before=previous,after=value))
    record[field]=value
   record['discrete_exact_rational_bootstrap']=population=='RECT'
   record['full_integer_numerator']=proof['full_integer_numerator'];record['full_integer_denominator']=proof['full_integer_denominator']
   proofs.append(dict(table=family,record_index=j,metric=metric,scope=scope,**proof))
  previous_holm=[r['p_bootstrap_holm'] for r in records]
  S.adjust_family(records)
  for j,(previous,r) in enumerate(zip(previous_holm,records)):
   if previous!=r['p_bootstrap_holm']:changes.append(dict(table=family,record_index=j,scope=r['scope'],backbone=r['backbone'],left=r['left'],right=r['right'],metric=r['metric'],field='p_bootstrap_holm',before=previous,after=r['p_bootstrap_holm']))
 # CIs and all t values are byte-for-byte equal; all other source cells stay fixed.
 for family,records in new['tables'].items():
  for a,b in zip(records,old['tables'][family]):
   for key in ('CI95','p_t','p_t_holm','p_t_holm_family_size','t_test','per_seed_delta'):
    assert a.get(key)==b.get(key),('Unchanged field',family,key)
 new['discrete_bootstrap_arithmetic']='Exact integer seed counts and inclusive rational null comparison; p formula/draws/CI/estimands fixed'
 pending=S.DOC/'PAIRED.json.pending'
 with pending.open('x') as f:json.dump(S.clean(new),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 pending.replace(S.DOC/'PAIRED.json')
 if had_paired_tables:
  for stem in ('PAIRED_TABLE_1','PAIRED_TABLE_2','PAIRED_TABLE_3'):
   for extension in ('.csv','.md'):
    path=S.DOC/(stem+extension)
    if path.is_file():path.unlink() # exact initial bytes copied to immutable diagnostic archive
  T.paired_tables(new)
 report=dict(status='COMPLETE_NUMERICAL_REPAIR',reason='Floating summation order changed equality decisions for discrete bootstrap tails; zero confusion difference did not yield p=1',
  source_lock_sha256=S.sha(S.DOC/LOCK),current_bindings=lock['current_bindings'],method_lock_sha256=lock['method_lock_sha256'],
  original_statistics_source_lock_sha256=S.sha(S.DOC/'STATS_SOURCE_LOCK.json'),preserved_initial_outputs=initial_bindings,
  original_verifier_archived_path=ARCHIVE+'/ORIGINAL_verify_stats.py.txt',original_verifier_sha256=S.sha(S.DOC/ARCHIVE/'ORIGINAL_verify_stats.py.txt'),
  original_PAIRED_sha256=before_hash,corrected_PAIRED_sha256=S.sha(S.DOC/'PAIRED.json'),changed_cells=changes,proofs=proofs,
  exact_comparison='abs(drawN/drawD-fullN/fullD)>=abs(fullN/fullD), evaluated by integer cross multiplication',
  exact_zero_difference='fullN=0 => every draw is in two-sided tail, p=1',
  AUC_integer_definition='2000halfstepunits from fixed1001threshold-indicators; sum3seeds denominator6000 perframe',
  bootstrap_draws_sha256=draw_sha,all_t_p_and_CI_exactly_unchanged=True,all_METRICS_and_main_table_cells_exactly_unchanged=True,
  additional_model_forwards=0,additional_PnP_calls=0,additional_coordinate_changes=0,setting_or_row_changes=0)
 S.write('NUMERICAL_AMENDMENT.json',report)
 print('NUMERICAL_P_REPAIR_COMPLETE',len(changes),'changed cells',sum(r['field']=='p_bootstrap' for r in changes),'p cells',flush=True)
if __name__=='__main__':run()
