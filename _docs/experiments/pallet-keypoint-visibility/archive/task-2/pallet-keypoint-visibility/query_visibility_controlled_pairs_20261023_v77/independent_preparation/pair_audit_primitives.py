"""Future source-only V77 independent audit primitives; NumPy, no model I/O.

Real artifact adapters will be bound after generator/trainer schemas freeze.
This module never reads paths on import and performs no model inference.
"""
import hashlib,json
from pathlib import Path
import numpy as np

def stable_bce(logit,label):return np.logaddexp(0,np.asarray(logit,float))-np.asarray(label)*np.asarray(logit,float)
def probability(logit):return np.exp(-np.logaddexp(0,-np.asarray(logit,float)))
def _same(a,b):return np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True)

def assert_opaque_mask_identity(M0_before,M4_before,alpha,M0_after,M4_after):
 arrays=[np.asarray(a)for a in [M0_before,M4_before,alpha,M0_after,M4_after]]
 assert len({a.shape for a in arrays})==1 and arrays[0].ndim==2
 assert all(np.isin(a,[0,1]).all()for a in arrays),'Normalize only with frozen source-specific raw threshold before audit'
 f,v,o,fa,va=[a.astype(bool)for a in arrays]
 assert not(v&~f).any() and not(va&~fa).any()
 assert np.array_equal(f,fa) and np.array_equal(va,v&~o)
 return {'same_M0':True,'after_M4_equals_before_and_not_alpha':True,'alpha_is_opaque_binary':True,'after_vis_subset_before':True}

def assert_opaque_rgb_identity(before,after,donor_canvas,alpha):
 a,b,c=[np.asarray(x)for x in [before,after,donor_canvas]];m=np.asarray(alpha,bool)
 assert a.dtype==b.dtype==c.dtype==np.uint8 and a.shape==b.shape==c.shape==(*m.shape,3)
 assert np.array_equal(a[~m],b[~m])and np.array_equal(b[m],c[m])
 return {'RGB_unpainted_exact':True,'RGB_painted_exact_donor_canvas':True}

def assert_teacher_order(before_label,after_label):
 a=np.asarray(before_label);b=np.asarray(after_label);assert a.shape==b.shape and np.isin(a,[-1,0,1]).all()and np.isin(b,[-1,0,1]).all()
 # With identical M0/r and M4after subset, frozen V2 can only lose visibility.
 assert not((a==0)&(b!=0)).any(),'KnownC cannot becomeV/U under this frozen teacher if M0/r are identical'
 assert not((a==-1)&(b==1)).any(),'Removal cannot create knownVISIBLE'
 return {'knownC_staysC':True,'no_unknown_to_visible':True}

def assert_fixed_geometry(before,after):
 keys=['q8','r8_GT_only','L','box','context42','native_slot','self_proxy_GT_only','query_center_valid','local_valid','target_valid','ROI_native_valid']
 for k in keys:assert _same(before[k],after[k]),k
 for k in ['source','split','parent_id']:assert before[k]==after[k],k
 return {'all_fixed_geometry_and_canvas_validity_exact':True,'keys':keys}

def audit_donor_split(records):
 """Normalized metadata-only schema; real adapter must bind original fields."""
 for r in records:
  assert r['split']in ['train','val']and r['donor_split']==r['split']
  assert r['image_sha256']!=r['donor_image_sha256']
  assert r['donor_source']=='YCB_PUBLIC_REAL'
  if r['source']=='YCB_PUBLIC_REAL':assert r['object_id']!=r['donor_object_id']
 return {'rows':len(records),'same_partition':True,'same_RGB_excluded':True,'YCB_same_object_id_excluded':True}

def audit_batch96(source,label,component,pair_id=None):
 s=np.asarray(source);y=np.asarray(label);c=np.asarray(component);assert len(s)==len(y)==len(c)==96
 assert np.isin(y,[0,1]).all()
 counts={str(k):int((c==k).sum())for k in np.unique(c)}
 assert counts=={'natural_original':32,'after_nonanchor':32,'matched_pair':32}
 for name in ['SYNTHETIC','YCB_PUBLIC_REAL']:
  for v in [0,1]:
   assert int(((s==name)&(y==v)).sum())==24
   for cc in counts:assert int(((s==name)&(y==v)&(c==cc)).sum())==8
 if pair_id is not None:
  ids=np.asarray(pair_id)
  for name in ['SYNTHETIC','YCB_PUBLIC_REAL']:
   ix=np.flatnonzero((s==name)&(c=='matched_pair'));assert len(ix)==16
   # Pair member order is beforeV then afterC, adjacent per frozen sampler.
   assert np.array_equal(y[ix],np.tile([1,0],8))
   assert np.array_equal(ids[ix][0::2],ids[ix][1::2])
 return {'batch96':True,'component_counts':counts,'each_source_class':24,'each_source_class_component':8}

def audit_natural_selector(history,metric_key='natural4strata_macroBCE',strata_key='natural_stratumBCE'):
 assert [r['update']for r in history]==list(range(100,2001,100))
 best=float('inf');step=None
 for r in history:
  assert len(r[strata_key])==4
  v=float(np.mean(list(r[strata_key].values())));assert v==r[metric_key]
  saved=v<best
  if saved:best=v;step=r['update']
  assert r['saved']==saved and r['best_update']==step
 return {'best_update':step,'natural4strata_macroBCE':best,'strict_lower_earliest_ties':True,'update0_excluded':True,'controlled_VAL_used_for_selection':False}

def binary_metric(z,y,m):
 z=np.asarray(z);y=np.asarray(y);m=np.asarray(m,bool);assert z.shape==y.shape==m.shape
 assert np.isin(y[m],[0,1]).all()and np.isfinite(z[m]).all()
 pred=z>=0;cm=np.zeros((2,2),int);np.add.at(cm,(y[m].astype(int),pred[m].astype(int)),1)
 def div(a,b):return float(a/b)if b else None
 return {'N':int(m.sum()),'CM_rows_CV_cols_CV':cm.tolist(),'V_precision':div(cm[1,1],cm[:,1].sum()),'V_recall':div(cm[1,1],cm[1].sum()),'C_precision':div(cm[0,0],cm[:,0].sum()),'C_recall':div(cm[0,0],cm[0].sum()),'BCE_known_primary_only':float(stable_bce(z[m],y[m]).mean())if m.any()else None}

def paired_saved_metrics(before_z,after_z,before_y,after_y,before_primary,after_primary,anchor,source,split,control_success):
 bz,az,by,ay,bp,ap=[np.asarray(x)for x in [before_z,after_z,before_y,after_y,before_primary,after_primary]]
 assert bz.shape==az.shape==by.shape==ay.shape==bp.shape==ap.shape and bz.ndim==2 and bz.shape[1]==8
 n=len(bz);ak=np.asarray(anchor,int);assert ak.shape==(n,)and ((ak>=0)&(ak<8)).all();rows=np.arange(n);anchor_mask=np.zeros((n,8),bool);anchor_mask[rows,ak]=True
 assert np.all(bp[rows,ak]&(by[rows,ak]==1)),'Every attempted pair anchor is originalprimaryVISIBLE'
 success=np.asarray(control_success,bool);assert success.shape==(n,)
 valid=success&ap[rows,ak]&(ay[rows,ak]==0);before_correct=bz[rows,ak]>=0;after_correct=az[rows,ak]<0
 pbefore=probability(bz);pafter=probability(az);out=[]
 for s in ['SYNTHETIC','YCB_PUBLIC_REAL']:
  for part in ['train','val']:
   base=(np.asarray(source)==s)&(np.asarray(split)==part);pair=base&valid;nm=base[:,None]&ap&~anchor_mask
   counts={str(v):int((base&(ay[rows,ak]==v)).sum())for v in [-1,0,1]}
   ctr=nm&(ay==1)&pair[:,None];anchor_predC=(az[rows,ak]<0)[:,None]
   mixed_instances=np.flatnonzero(pair&(ctr.any(1)))
   stability=base[:,None]&~anchor_mask&bp&ap&(by==1)&(ay==1)
   both=before_correct&after_correct
   contrast=pbefore[rows,ak]-pafter[rows,ak]
   out.append({'source':s,'split':part,'attempted_pairs':int(base.sum()),'control_failures':int((base&~success).sum()),'failed_or_unknown_or_invalid_anchor_rows':int((base&~valid).sum()),'anchor_after_teacher_counts':counts,'known_admitted_V_to_C_pairs':int(pair.sum()),'both_correct_known_pair':int((pair&both).sum()),'before_V_correct_known_pair':int((pair&before_correct).sum()),'after_C_correct_known_pair':int((pair&after_correct).sum()),'both_correct_per_attempted_diagnostic':float((pair&both).sum()/base.sum())if base.any()else None,'p_before_gt_p_after_known_pair':int((pair&(contrast>0)).sum()),'equal_probability_known_pair':int((pair&(contrast==0)).sum()),'mean_p_before_minus_after_known_pair':float(contrast[pair].mean())if pair.any()else None,'after_nonanchor_metrics':binary_metric(az,ay,nm),
    'same_after_C_anchor_V_nonanchor_query_pairs':int(ctr.sum()),'same_after_both_correct':int((ctr&anchor_predC&(az>=0)).sum()),'same_after_pV_nonanchor_gt_anchor':int((ctr&(pafter>pafter[rows,ak,None])).sum()),'same_after_mixed_instances':len(mixed_instances),'all_mixed_after_queries_correct_instances':int(sum(bool(after_correct[i]and np.all(az[i,ctr[i]]>=0))for i in mixed_instances)),
    'nonanchor_V_to_V_stability_N':int(stability.sum()),'nonanchor_V_before_correct_after_wrong':int((stability&(bz>=0)&(az<0)).sum()),'nonanchor_V_before_wrong_after_correct':int((stability&(bz<0)&(az>=0)).sum()),'nonanchor_V_mean_abs_probability_change':float(abs(pbefore-pafter)[stability].mean())if stability.any()else None})
 return out

def mock_tests():
 full=np.ones((7,9),bool);vis=full.copy();alpha=np.zeros_like(full);alpha[2:5,3:7]=1
 assert_opaque_mask_identity(full,vis,alpha,full,vis&~alpha)
 rgb=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3);donor=np.full_like(rgb,111);after=np.where(alpha[...,None],donor,rgb);assert_opaque_rgb_identity(rgb,after,donor,alpha)
 assert_teacher_order(np.array([1,1,-1,0]),np.array([0,-1,0,0]))
 source=[];label=[];component=[];pid=[]
 for cc in ['natural_original','after_nonanchor']:
  for s in ['SYNTHETIC','YCB_PUBLIC_REAL']:
   for c in [0,1]:source += [s]*8;label += [c]*8;component += [cc]*8;pid += [-1]*8
 for s in ['SYNTHETIC','YCB_PUBLIC_REAL']:
  for i in range(8):source += [s,s];label += [1,0];component += ['matched_pair']*2;pid += [i,i]
 batch=audit_batch96(source,label,component,pid)
 zz=np.linspace(-3,3,96);loss=stable_bce(zz,np.array(label));assert np.isclose(loss.mean(),np.mean([loss[np.array(component)==c].mean()for c in ['natural_original','after_nonanchor','matched_pair']]))
 bz=np.full((4,8),2.);az=bz.copy();az[:,0]=-2;by=np.ones((4,8),int);ay=by.copy();ay[:,0]=0;pm=np.ones((4,8),bool)
 results=paired_saved_metrics(bz,az,by,ay,pm,pm,np.zeros(4,int),np.array(['SYNTHETIC','SYNTHETIC','YCB_PUBLIC_REAL','YCB_PUBLIC_REAL']),np.array(['train','val','train','val']),np.ones(4,bool))
 failed=paired_saved_metrics(bz,az,by,ay,pm,pm,np.zeros(4,int),np.array(['SYNTHETIC','SYNTHETIC','YCB_PUBLIC_REAL','YCB_PUBLIC_REAL']),np.array(['train','val','train','val']),np.zeros(4,bool))
 assert all(r['known_admitted_V_to_C_pairs']==0 and r['control_failures']==1 and r['failed_or_unknown_or_invalid_anchor_rows']==1 for r in failed)
 assert all(r['both_correct_known_pair']==1 and r['same_after_both_correct']==7 and r['nonanchor_V_before_correct_after_wrong']==0 for r in results)
 history=[];best=float('inf');step=None
 for update in range(100,2001,100):
  value=1. if update>=200 else 2.;saved=value<best
  if saved:best=value;step=update
  history.append({'update':update,'natural4strata_macroBCE':value,'natural_stratumBCE':dict.fromkeys(['S0','S1','Y0','Y1'],value),'saved':saved,'best_update':step})
 selected=audit_natural_selector(history);assert selected['best_update']==200
 return {'PASS':True,'batch':batch,'mean_BCE_equal_three_component_weights':True,'mask_RGB_algebra':True,'paired_and_same_after_controls':True,'control_failure_with_actual_teacherC_excluded':True,'teacher_visibility_monotonicity':True,'natural_only_earliest_selection':selected,'model_source_current_forward_fit_PnP':0}

if __name__=='__main__':
 out=Path(__file__).parent/'GENERATED_INDEPENDENT_AUDIT_PRIMITIVE_TESTS.json';assert not out.exists();r=mock_tests();r['code']={'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
