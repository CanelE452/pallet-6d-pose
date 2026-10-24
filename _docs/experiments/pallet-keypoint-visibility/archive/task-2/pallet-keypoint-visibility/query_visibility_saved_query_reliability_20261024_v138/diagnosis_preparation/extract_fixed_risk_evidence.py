"""Read completed V138 summary/table only; fixed declared risk order, no fitting."""
import argparse, hashlib, json
from pathlib import Path

H=Path(__file__).resolve().parent
V=H.parent
SRC=V/'saved_observability_diagnosis_v1'
OUT=V/'saved_risk_interpretation_v1'
SEAL_SHA='0872ad6105050a4b78dbe87c29c9bf1d2fe22375478a3efd365bb811f0ce14ed'
RISKS=('risk_low_detector_confidence','risk_low_native_keypoint_confidence','risk_local_invalid_fraction','risk_central32_invalid_fraction','risk_context_invalid_fraction','risk_input_unavailable','risk_native_bounds_invalid','risk_query_outside_pred_bbox_normalized','risk_global_edge_concurrence_rms','risk_native_heldout_edge_rms')
LABELS=('1-detector confidence','1-native keypoint confidence','local invalid fraction','central32 invalid fraction','context invalid fraction','input unavailable','native bounds invalid','query outside bbox / scale','global edge concurrence RMS','native heldout edge RMS')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def key(r):return tuple(r.get(k)for k in('corpus','fold','teacher','availability_scope','target','GT_error_diagnostic_stratum','signal'))
def run():
 sealpath=SRC/'DIAGNOSIS_SEAL.json';assert sha(sealpath)==SEAL_SHA
 seal=read(sealpath);assert seal['completed']and seal['PASS']and seal['all26624_preserved']
 result=read(check(seal['results']));assert result['completed']and result['PASS']and result['no_sign_or_feature_or_threshold_or_model_selection']
 assert result['fixed_primary_signal_order']==list(RISKS)
 rows=[json.loads(x)for x in check(seal['full_table']).read_text().splitlines()if x]
 assert len(rows)==15168==result['records']
 keep=[r for r in rows if r['signal']in RISKS and r['fold']in('TRAIN','VALIDATION')and (r['teacher']in('ALL','D','E')or r['target']=='EXTERNAL_TEACHER_E1_D0')]
 index={key(r):r for r in keep};assert len(index)==len(keep)
 tables=[]
 for corpus in('OLD_NATIVE','CONTINUOUS_NATIVE'):
  for fold in('TRAIN','VALIDATION'):
   table=[]
   for signal,label in zip(RISKS,LABELS):
    cells={}
    for teacher in('D','E'):
     for target,short in(('GT_ERROR_GT3','position_gt3'),('C29_DE_MISTAKE_WHEN_INPUT_AVAILABLE','available_mistake'),('C29_KNOWN_FAILURE_INCLUDING_INPUT_U','known_failure')):
      r=index[(corpus,fold,teacher,'ALL_ORIGINAL',target,None,signal)]
      cells[short+'_'+teacher]=r['auc']
    for stratum,short in(('ALL_GT_ERROR','external_ED_all'),('GT_ERROR_LE3','external_ED_LE3')):
     r=index[(corpus,fold,'ALL_WITH_SU_OUTSIDE_TARGET','ALL_ORIGINAL','EXTERNAL_TEACHER_E1_D0',stratum,signal)]
     cells[short]=r['auc']
    table.append(dict(signal=signal,label=label,cells=cells))
   tables.append(dict(corpus=corpus,fold=fold,rows=table))
 OUT.mkdir()
 write(OUT/'PRIMARY_FIXED_ORDER_EVIDENCE.json',dict(completed=True,PASS=True,source_only=True,diagnosis_seal=binding(sealpath),code=binding(__file__),accounting=result['accounting'],fixed_risk_order=list(RISKS),tables=tables,all_selected_tables=keep,
  signs_unchanged=True,no_best_risk_or_sign_or_model_selection=True,no_numerical_thresholds_exported=True,no_model_fit_RGB_private_or_GT_body_reads=True,
  limitations=['Input-unavailable and bounds flags partly define prediction-U; this is not independent failure discovery.','Positive fixed-direction AUC below .5 remains below .5; no direction flipping.','Raw qerror is source GT-only diagnosis, never an input filter or new rejection rule.','TRAIN bins retain ties, empty bins, undefined inputs and no-TRAIN-reference bins.','Source variants share bases and are correlated; these are point associations, not causal or transfer estimates.']))
 def fmt(a):
  v=a['AUC_fixed_higher_score_positive'];return 'NA'if v is None else f'{v:.3f}'
 lines=['All ten declared risks, fixed higher-risk direction. Values are tied-rank AUC, with missing-input and class counts retained in the accompanying JSON. E/D is teacher external E1 versus D0. No numerical cuts or new model decisions.','']
 for t in tables:
  lines +=[t['corpus']+' / '+t['fold'],'','| Declared input signal | q>3 D | q>3 E | available mistake D | available mistake E | known failure D | known failure E | E/D all | E/D ≤3 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for row in t['rows']:
   cols=['position_gt3_D','position_gt3_E','available_mistake_D','available_mistake_E','known_failure_D','known_failure_E','external_ED_all','external_ED_LE3']
   lines.append('| '+row['label']+' | '+' | '.join(fmt(row['cells'][c])for c in cols)+' |')
  lines+=['']
 (OUT/'PRIMARY_FIXED_ORDER_TABLE.md').write_text('\n'.join(lines))
 write(OUT/'EXTRACTION_SEAL.json',dict(completed=True,PASS=True,diagnosis_seal=binding(sealpath),code=binding(__file__),evidence=binding(OUT/'PRIMARY_FIXED_ORDER_EVIDENCE.json'),table=binding(OUT/'PRIMARY_FIXED_ORDER_TABLE.md'),no_model_or_fit_or_private_or_newselection=True))
 print(json.dumps(dict(completed=True,PASS=True,seal=binding(OUT/'EXTRACTION_SEAL.json'),tables=4,rows_each=10,full_selected_records=len(keep))))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run-completed',action='store_true');a=ap.parse_args()
 if a.run_completed:run()
 else:raise SystemExit('Use --run-completed only after root diagnosis completion notice.')
