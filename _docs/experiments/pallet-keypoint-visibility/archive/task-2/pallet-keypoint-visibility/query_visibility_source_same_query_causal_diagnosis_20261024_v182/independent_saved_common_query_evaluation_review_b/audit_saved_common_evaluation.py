"""Independent scalar-loop verification of V182 saved TRAIN diagnostic metrics.
No import of head, producer or evaluator; no RGB, features, weights or forward.
Run --complete only after root reports both inference and evaluation terminal.
"""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,math,sys
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent
VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM');SUPPORT=('visible_end','foreign_front')
RAWCOMPLETE=V/'common_query_diagnostic_v1/COMPLETE.json'
def sha(p):
 p=Path(p).resolve();assert not any(x in str(p).lower() for x in ('_v112','current223','private','_v169','_v177','_v178','synthetic_val','final_synth_val'))
 assert p.suffix not in ('.pth','.pt','.ckpt')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert bind(b['path'])==b,b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ratio(a,b):return None if b==0 else a/b
def scalar_metric(truth,pred,indices):
 cm=[[0,0,0],[0,0,0]];unknown=[0,0,0]
 for i in indices:
  t=int(truth[i]);p=int(pred[i]);assert t in (-1,0,1) and p in (-1,0,1)
  if t<0:unknown[2 if p<0 else p]+=1
  else:cm[t][2 if p<0 else p]+=1
 tn,fp,u0=cm[0];fn,tp,u1=cm[1];known=sum(sum(x) for x in cm)
 return dict(rows=len(indices),known=known,unknown=sum(unknown),CM_NO_YES_by_NO_YES_U=cm,positive_precision=ratio(tp,tp+fp),positive_recall_including_U=ratio(tp,tp+fn+u1),known_prediction_U=u0+u1,known_U_rate=ratio(u0+u1,known),unknown_truth_prediction_counts_NO_YES_U=unknown)
def delta_summary(values):
 # Producer reports FP32 differences/mean/median, so preserve its declared FP32
 # representation while computing all memberships/sign counts independently.
 a=np.asarray(values,np.float32)
 return dict(n=len(values),mean=None if not values else float(np.mean(a,dtype=np.float32)),median=None if not values else float(np.median(a)),positive_count=sum(float(x)>0 for x in values),zero_count=sum(float(x)==0 for x in values),negative_count=sum(float(x)<0 for x in values))
def scalar_pairs(truth,pred,raw,pairs,av):
 result={}
 for j,name in enumerate(SUPPORT):
  cells={}
  for ta in (-1,0,1):
   for tb in (-1,0,1):
    n=0;nav=0;bothcorrect=0;cm=[[0]*3 for _ in range(3)];d=[]
    for left,right in pairs:
     if int(truth[left,j])!=ta or int(truth[right,j])!=tb:continue
     n+=1;bothav=bool(av[left] and av[right]);nav+=bothav
     pa,pb=int(pred[left,j]),int(pred[right,j]);cm[2 if pa<0 else pa][2 if pb<0 else pb]+=1
     bothcorrect+=bool(bothav and pa==ta and pb==tb)
     if math.isfinite(float(raw[left,j])) and math.isfinite(float(raw[right,j])):d.append(np.float32(raw[right,j]-raw[left,j]))
    cells[f'{ta}_to_{tb}']=dict(pairs=n,both_input_available=nav,prediction_transition_NO_YES_U_by_NO_YES_U=cm,both_predictions_correct=bothcorrect if ta>=0 and tb>=0 else None,right_minus_left_logit=delta_summary(d))
  result[name]=cells
 canonical=available=correct=0
 for left,right in pairs:
  if tuple(int(x) for x in truth[left])==(1,0) and tuple(int(x) for x in truth[right])==(0,1):
   canonical+=1;bothav=bool(av[left] and av[right]);available+=bothav
   correct+=bool(bothav and all(int(pred[left,j])==int(truth[left,j]) and int(pred[right,j])==int(truth[right,j]) for j in range(2)))
 result['joint_visible_only_to_foreign_only']=dict(pairs=canonical,both_available=available,both_supports_correct_at_both_images=correct,all_other_or_unknown_pairs=len(pairs)-canonical,not_a_success_gate=True)
 return result

def invented():
 c={}
 def ok(k,v):assert v,k;c[k]=True
 t=[0,0,1,1,1,-1];p=[0,1,1,0,-1,1];m=scalar_metric(t,p,list(range(6)))
 ok('CM_2by3_keeps_predictionU_and_unknown_truth_separate',m['CM_NO_YES_by_NO_YES_U']==[[1,1,0],[1,1,1]] and m['unknown_truth_prediction_counts_NO_YES_U']==[0,1,0])
 ok('precision_called_positive_and_recall_includes_positiveU',m['positive_precision']==.5 and m['positive_recall_including_U']==1/3)
 ok('zero_support_returns_null_rates',scalar_metric([-1],[-1],[0])['positive_precision'] is None and scalar_metric([-1],[-1],[0])['known_U_rate'] is None)
 truth=np.array([[1,0],[0,1],[-1,-1],[-1,-1]],np.int8);pred=truth.copy();raw=np.array([[2,-2],[-3,3],[np.nan,np.nan],[np.nan,np.nan]],np.float32);av=np.array([1,1,0,0],bool)
 z=scalar_pairs(truth,pred,raw,[(0,1),(2,3)],av)
 ok('canonical_joint_exact_nonexclusive_state_pair',z['joint_visible_only_to_foreign_only']['both_supports_correct_at_both_images']==1)
 ok('right_minus_left_sign_correct',z['foreign_front']['0_to_1']['right_minus_left_logit']['mean']==5. and z['visible_end']['1_to_0']['right_minus_left_logit']['mean']==-5.)
 ok('unknown_transition_is_retained_not_scored',z['foreign_front']['-1_to_-1']['pairs']==1 and z['foreign_front']['-1_to_-1']['both_predictions_correct'] is None)
 pred[1]=[-1,-1];z=scalar_pairs(truth,pred,raw,[(0,1)],av)
 ok('predictionU_cannot_count_as_paired_success',z['joint_visible_only_to_foreign_only']['both_supports_correct_at_both_images']==0 and z['foreign_front']['0_to_1']['prediction_transition_NO_YES_U_by_NO_YES_U'][0][2]==1)
 return c

def compare_tree(expected,actual,path='root'):
 assert type(expected)==type(actual) or isinstance(expected,(int,float)) and isinstance(actual,(int,float)),(path,type(expected),type(actual))
 if isinstance(expected,dict):
  assert set(expected)==set(actual),(path,'keys',set(expected)^set(actual))
  for k in expected:compare_tree(expected[k],actual[k],path+'/'+k)
 elif isinstance(expected,list):
  assert len(expected)==len(actual),(path,'length')
  for i,(a,b) in enumerate(zip(expected,actual)):compare_tree(a,b,path+f'/{i}')
 else:assert expected==actual,(path,expected,actual)

def audit(cp):
 cp=Path(cp).resolve();assert cp.parent.parent==V and cp.name=='COMPLETE.json'
 complete=read(cp);assert complete['completed'] and complete['PASS'] and complete['source_TRAIN_diagnostic_only'] and not complete['fit_admission'] and complete['selected_model'] is None and not complete['fully_solved']
 entry=read(check(complete['entry']));assert entry['approved_saved_only_evaluation'] is True
 assert complete['code']==entry['code'];check(entry['code']);check(entry['scope']);check(entry['self_checks'])
 assert complete['raw_complete']==entry['raw_complete']==bind(RAWCOMPLETE) and complete['prediction_complete']==entry['prediction_complete']
 prediction=read(check(entry['prediction_complete']));assert prediction['completed'] and prediction['PASS'] and prediction.get('new_training_updates',0)==0
 assert set(entry['arm_outputs'])==set(prediction['arms']) and len(entry['arm_outputs'])==2
 for arm,b in entry['arm_outputs'].items():assert b==prediction['arms'][arm]['outputs']
 source=read(RAWCOMPLETE);assert source['PASS'] and source['original_COVERanchors1024_byteparity']
 metadata=lines(check(source['metadata']));assert len(metadata)==4096 and all(r['split']=='TRAIN' and not r['fit_admission'] and not r['production_query_replay'] for r in metadata)
 byidentity={};groups=defaultdict(list)
 for i,r in enumerate(metadata):
  assert r['diagnostic_row_index']==i and r['row_id']==f"V182_BASE{r['base_index']:03d}/{r['variant']}/native{r['native_slot']}"
  key=(r['base_index'],r['variant'],r['native_slot']);assert key not in byidentity;byidentity[key]=i
  groups['ALL'].append(i);groups['variant:'+r['variant']].append(i);groups['native:'+str(r['native_slot'])].append(i)
  for k in ('geometry_family','material','aspect','foreign_family'):groups[k+':'+r['source_image_original_metadata']['factors'][k]].append(i)
 assert len(byidentity)==4096 and {r['base_index'] for r in metadata}==set(range(128))
 av=np.array([r['original_query_available'] for r in metadata],bool);assert int((~av).sum())==96
 identity=dict(row_id=np.array([r['row_id'] for r in metadata]),original_global_index=np.array([r['diagnostic_row_index'] for r in metadata],np.int64),native_slot=np.array([r['native_slot'] for r in metadata],np.int8),query_xy=np.array([[np.nan,np.nan] if r['query_xy'] is None else [np.nan if x is None else x for x in r['query_xy']] for r in metadata],np.float64),L=np.array([np.nan if r['L'] is None else r['L'] for r in metadata],np.float64),original_query_available=av)
 outputs={}
 for arm,b in entry['arm_outputs'].items():
  with np.load(check(b),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
  for k,v in identity.items():assert same(a[k],v),(arm,k)
  assert a['raw_support_logits'].dtype==a['raw_owner_logits'].dtype==np.float32 and a['raw_support_logits'].shape==(4096,2) and a['raw_owner_logits'].shape==(4096,3)
  for i in range(4096):
   assert a['support_prediction'][i].tolist()==[int(x>0) for x in a['raw_support_logits'][i]] if av[i] else (a['support_prediction'][i]==-1).all()
   if av[i]:assert np.isfinite(a['raw_support_logits'][i]).all() and np.isfinite(a['raw_owner_logits'][i]).all() and int(a['owner_prediction'][i])==int(np.argmax(a['raw_owner_logits'][i]))
   else:assert np.isnan(a['raw_support_logits'][i]).all() and np.isnan(a['raw_owner_logits'][i]).all() and int(a['owner_prediction'][i])==-1
  outputs[arm]=a
 # Only after both frozen output packets are verified: read the separate support target.
 with np.load(check(source['targets']),allow_pickle=False) as z:truth=z['support_target'].copy()
 assert truth.shape==(4096,2) and truth.dtype==np.int8 and np.isin(truth,[-1,0,1]).all() and (truth[~av]==-1).all()
 reported=read(check(complete['metrics']));assert reported['completed'] and reported['PASS'] and reported['support_order']==list(SUPPORT) and reported['training_data_only'] and not reported['heldout_performance'] and not reported['physical_native_or_real_transfer_gold'] and reported['no_model_or_threshold_choice']
 result={}
 for arm,a in outputs.items():
  strata={name:{name2:scalar_metric(truth[:,j],a['support_prediction'][:,j],ii) for j,name2 in enumerate(SUPPORT)} for name,ii in groups.items()}
  paired={}
  for right in VARIANTS[1:]:
   pp=[(byidentity[b,'BEFORE',k],byidentity[b,right,k]) for b in range(128) for k in range(8)]
   paired['BEFORE_to_'+right]=scalar_pairs(truth,a['support_prediction'],a['raw_support_logits'],pp,av)
  result[arm]=dict(strata=strata,paired=paired)
 compare_tree(result,reported['arms'])
 ledger=lines(check(complete['rows']));assert len(ledger)==4096
 for i,(r,l) in enumerate(zip(metadata,ledger)):
  for k in ('row_id','diagnostic_row_index','base_index','variant','native_slot','query_xy','L','source_image_original_global_index','anchor_original_global_index','original_query_available'):assert l[k]==r[k]
  assert l['support_truth']==truth[i].tolist() and set(l['responses'])==set(outputs)
  for arm,a in outputs.items():
   x=l['responses'][arm];assert x['support_prediction']==a['support_prediction'][i].tolist() and x['owner_prediction']==int(a['owner_prediction'][i])
   assert x['raw_support_logits']==[float(v) if math.isfinite(float(v)) else None for v in a['raw_support_logits'][i]]
 for b in entry['arm_outputs'].values():check(b)
 check(source['targets']);check(source['metadata']);check(entry['prediction_complete']);check(complete['metrics']);check(complete['rows'])
 return dict(completed=True,PASS=True,code=bind(__file__),actual_evaluation_complete=bind(cp),root_entry=complete['entry'],prediction_complete=entry['prediction_complete'],raw_complete=entry['raw_complete'],metrics=complete['metrics'],rows=complete['rows'],rows4096=True,common_input_U96=True,arms2=len(outputs),strata_per_arm=len(groups),support_metrics_per_arm=2*len(groups),paired_truth_transition_cells_per_arm=3*2*9,all3_canonical_joint_cells_per_arm=True,all_metrics_and_transition_cells_exact=True,all4096_saved_ledger_rows_exact=True,no_producer_or_evaluator_import=True,independent_method='Scalar membership/count loops for every CM, UNKNOWN/U/P/R, 9x2x3 truth-transition cells, paired successes/sign counts. Pair matching uses explicit(base,variant,native) identity rather than row cube. Raw difference/mean/median retains declared FP32 arithmetic; exact JSON value comparison.',limitations='SourceTRAIN same-query diagnostic only; not independent heldout/native physical/real-transfer gold. Original RGB/features/weights/model not opened; frozen-head/runtime provenance bound through completed saved receipts.',new_model_forward_training_or_threshold_changes=0,source_TRAIN_only=True,training_admission=False,selected_model=None,fully_solved=False)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--complete',type=Path);a=ap.parse_args()
 if a.prepare and a.complete is None:
  tests=invented();write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),invented_checks=tests,actual_prediction_or_target_bodies_read=0,requires_root_actual_completion_notification=True));print(json.dumps({'prepared':True,'receipt':bind(H/'PREPARATION_COMPLETE.json')}))
 elif a.complete is not None and not a.prepare:
  assert not (H/'REVIEW.json').exists();r=audit(a.complete);write(H/'REVIEW.json',r);print(json.dumps({'completed':True,'PASS':True,'review':bind(H/'REVIEW.json')}))
 else:raise SystemExit('Use --prepare or --complete after root terminal notification only.')
if __name__=='__main__':main()
