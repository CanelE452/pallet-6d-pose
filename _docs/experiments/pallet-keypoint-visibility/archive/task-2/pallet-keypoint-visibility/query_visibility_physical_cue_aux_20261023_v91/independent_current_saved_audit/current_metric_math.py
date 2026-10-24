"""Pure independent saved-current metric math. No IO/model imports or globals."""
from collections import Counter
import math
D='DIRECT_VISIBLE';E='EXTERNAL_OCCLUDED';S='SELF_OCCLUDED'
TASKS={'D_vs_E':(D,E),'D_vs_S':(D,S),'D_vs_ES':(D,E,S)}

def truth_bool(v):return v is True or v=='True'
def authoritative(z,valid,geometry=None):
 if not valid:return 'OTHER_UNRESOLVED'
 if geometry is True:return 'ABSENT_NONVISIBLE'
 if z is None or not math.isfinite(z):raise ValueError('Validfinite rawzE required')
 return 'VISIBLE'if z<0 else'ABSENT_NONVISIBLE'
def baseline_call(row,method,task,p=None,valid=True,geometry=None):
 if method=='v61':return 'VISIBLE'if truth_bool(row['present_at_fixed_0_5'])else'ABSENT_NONVISIBLE'
 if method=='legacy':
  label=row['legacy_stored_state']
  if label in('B','I'):return 'VISIBLE'
  if label=='O':return 'ABSENT_NONVISIBLE'
  if label=='S':return 'OUT_OF_TASK_GEOMETRY_S'if task=='D_vs_E'else'ABSENT_NONVISIBLE'
  return 'ABSTAIN'if label=='U'else'OTHER_UNRESOLVED'
 if not valid:return 'OTHER_UNRESOLVED'
 if geometry is True:return 'ABSENT_NONVISIBLE'
 if p is None:return 'OTHER_UNRESOLVED'
 if not math.isfinite(p)or not 0<=p<=1:raise ValueError('Frozenprobability')
 return 'VISIBLE'if p>=.5 else'ABSENT_NONVISIBLE'
def selected(rows,cohort,group,task):
 flag=cohort+'_'+('D_vs_ES'if task=='D_vs_S'else task)
 return[r for r in rows if r['group']==group and truth_bool(r[flag])and r['human_state']in TASKS[task]]
def manual_near(rows):return[r for r in rows if r['annotation_coordinate_source']=='manual_click'and r['human_state']==D and truth_bool(r['query_valid'])and float(r['initial_reference_error_px'])<=3]
def error_bin(e):
 e=float(e)
 if not math.isfinite(e):return 'UNKNOWN_ERROR'
 return 'px_le3'if e<=3 else'px_3_to10'if e<=10 else'px_10_to30'if e<=30 else'px_gt30'
def calculate(rows,predictions):
 assert len(rows)==len(predictions)
 cm=Counter((r['human_state']==D,p)for r,p in zip(rows,predictions));pos=sum(r['human_state']==D for r in rows);neg=len(rows)-pos
 tp,fn,fp,tn=cm[True,'VISIBLE'],cm[True,'ABSENT_NONVISIBLE'],cm[False,'VISIBLE'],cm[False,'ABSENT_NONVISIBLE']
 up,un=cm[True,'ABSTAIN'],cm[False,'ABSTAIN'];sp,sn=cm[True,'OUT_OF_TASK_GEOMETRY_S'],cm[False,'OUT_OF_TASK_GEOMETRY_S'];op,on=cm[True,'OTHER_UNRESOLVED'],cm[False,'OTHER_UNRESOLVED']
 assert tp+fn+up+sp+op==pos and fp+tn+un+sn+on==neg
 ratio=lambda a,b:a/b if b else None
 return dict(N=len(rows),frames=len({r['id']for r in rows}),visible_support=pos,occluded_support=neg,TP=tp,FN_decided=fn,FP=fp,TN=tn,U_visible=up,U_occluded=un,out_of_task_S_visible=sp,out_of_task_S_occluded=sn,other_unresolved_visible=op,other_unresolved_occluded=on,visible_precision=ratio(tp,tp+fp),visible_recall_all_GT=ratio(tp,pos),occluded_precision=ratio(tn,tn+fn),occluded_recall_all_GT=ratio(tn,neg),abstention_rate=ratio(up+un,len(rows)),out_of_task_S_rate=ratio(sp+sn,len(rows)),unresolved_rate=ratio(up+un+sp+sn+op+on,len(rows)),decision_coverage=ratio(tp+fn+fp+tn,len(rows)),correct_fraction_all_GT=ratio(tp+tn,len(rows)),selective_accuracy=ratio(tp+tn,tp+fn+fp+tn))
def paired_change(rows,old,new):
 c=Counter()
 for r,a,b in zip(rows,old,new):
  if b not in('VISIBLE','ABSENT_NONVISIBLE'):c['new_unresolved']+=1;continue
  gt='VISIBLE'if r['human_state']==D else'ABSENT_NONVISIBLE'
  if a not in('VISIBLE','ABSENT_NONVISIBLE'):c['baseline_unresolved_new_correct'if b==gt else'baseline_unresolved_new_wrong']+=1
  elif a==gt and b==gt:c['both_correct']+=1
  elif a==gt:c['baseline_correct_new_wrong']+=1
  elif b==gt:c['baseline_wrong_new_correct']+=1
  else:c['both_wrong']+=1
 keys=('both_correct','baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong','baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved');assert sum(c.values())==len(rows)
 return {k:c[k]for k in keys}
