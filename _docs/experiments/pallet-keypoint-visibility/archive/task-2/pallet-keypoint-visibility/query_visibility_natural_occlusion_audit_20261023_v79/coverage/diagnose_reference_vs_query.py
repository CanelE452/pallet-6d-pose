"""Additive saved-coverage tables; no mask/RGB/model/current reads."""
from pathlib import Path
import csv,hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent;OUT=HERE/'results';P=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def write(p,d):
 with open(p,'x')as f:f.write(json.dumps(d,indent=2)+'\n')
seal=json.loads((OUT/'COVERAGE_SEAL.json').read_text());assert seal['PASS']and sha(Path(seal['per_query']['path']))==seal['per_query']['sha256']
names=['id','source','split','domain','teacher_label8','primary_loss_mask','q_center_support_state','q_center_M0','q_center_M4','q_round_raw3x3_valid','q_round_raw3x3_M0','q_round_raw3x3_M4','class_counts','valid_counts','q_reference_error_px_GT_only','reference_offset_L_GT_only','rule8']
with np.load(seal['per_query']['path'],allow_pickle=False)as z:a={k:z[k]for k in names}
teacher=a['teacher_label8'];center=a['q_center_support_state'];tables=[]
def table(mask):
 cm=np.zeros((3,4),np.int64)
 for ri,truth in enumerate((-1,0,1)):
  for ci,state in enumerate((-1,0,1,2)):cm[ri,ci]=np.sum(mask&(teacher==truth)&(center==state))
 cref=mask&(teacher==0);vref=mask&(teacher==1)
 return dict(N=int(mask.sum()),rows_reference_teacher=[-1,0,1],columns_q_support_state=[-1,0,1,2],CM=cm.tolist(),reference_C_with_q_VISIBLE=int((cref&(center==1)).sum()),reference_C_with_q_COVERED=int((cref&(center==0)).sum()),reference_C_with_q_OUTSIDE=int((cref&(center==2)).sum()),reference_C_with_q_INVALID=int((cref&(center==-1)).sum()),reference_V_with_q_COVERED=int((vref&(center==0)).sum()))
for domain in(0,1):
 for source in('SYNTHETIC','YCB_PUBLIC_REAL'):
  for split in('train','val'):
   base=np.broadcast_to(((a['domain']==domain)&(a['source']==source)&(a['split']==split))[:,None],teacher.shape)
   for pop,popmask in [('ALL',np.ones(teacher.shape,bool)),('PRIMARY_UNCHANGED',a['primary_loss_mask'])]:tables.append(dict(domain=domain,source=source,split=split,population=pop,counts=table(base&popmask)))
fixed=P/'query_visibility_controlled_pairs_20261023_v77/source_failure_audit/FIXED_COHORT_IDS.csv';assert sha(fixed)=='328caa6779d7dbdd4e715a64220332a0d0a78357fbc2a295265d37bf97b2e38d'
with open(fixed,newline='')as f:rr=[r for r in csv.DictReader(f)if r['cohort']=='STRONG_AND_ERROR_LE3PX_AND_CENTER_CELL']
lookup={str(v):i for i,v in enumerate(a['id'])};mask=np.zeros(teacher.shape,bool)
for r in rr:
 i=lookup[r['id']];k=int(r['native_k']);assert a['domain'][i]==0 and a['split'][i]=='val'and a['source'][i]=='YCB_PUBLIC_REAL'and a['primary_loss_mask'][i,k]
 assert a['q_reference_error_px_GT_only'][i,k]<=3 and np.max(abs(a['reference_offset_L_GT_only'][i,k]))<=1/18 and str(a['rule8'][i,k]).startswith('STRONG')
 mask[i,k]=True
assert mask.sum()==535 and(mask&(teacher==1)).sum()==459 and(mask&(teacher==0)).sum()==76
details=[]
for i,k in zip(*np.nonzero(mask)):
 c=a['class_counts'][i,k,1,2];count=int(c.sum())
 details.append(dict(id=str(a['id'][i]),native_k=int(k),reference_teacher=int(teacher[i,k]),q_center_state=int(center[i,k]),q_center_M0=int(a['q_center_M0'][i,k]),q_center_M4=int(a['q_center_M4'][i,k]),q_raw3x3_valid=int(a['q_round_raw3x3_valid'][i,k].sum()),q_raw3x3_V=int((a['q_round_raw3x3_M4'][i,k]==1).sum()),q_raw3x3_C=int(((a['q_round_raw3x3_M0'][i,k]==1)&(a['q_round_raw3x3_M4'][i,k]==0)).sum()),RGB96_native_cached_intersection_V=int(c[0]),RGB96_native_cached_intersection_C=int(c[1]),RGB96_native_cached_intersection_O=int(c[2]),RGB96_intersection_V_fraction=float(c[0]/count)if count else None))
cp=OUT/'FIXED_535_REFERENCE_VS_Q_CENTER.csv'
with open(cp,'x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
out=OUT/'REFERENCE_TEACHER_VS_Q_CENTER.json'
write(out,dict(completed=True,PASS=True,q_state_codes={'-1':'INVALID','0':'M0_AND_NOT_M4_COVERED_SUPPORT','1':'M4_VISIBLE','2':'NOT_M0_OUTSIDE_SUPPORT'},tables=tables,fixed535=table(mask),fixed535_csv=bind(cp),coverage_seal=bind(OUT/'COVERAGE_SEAL.json'),fixed_ids=bind(fixed),code=bind(Path(__file__)),scope='Raw query-pixel vs frozen reference-local teacher cross-tab only. No label/admission change, semantic E/S inference, model/error selection or causal explanation.',mask_RGB_features_scores_current_reads=0,new_forward_fit_PnP=0))
write(OUT/'FINAL_COVERAGE_SEAL.json',dict(completed=True,PASS=True,coverage_seal=bind(OUT/'COVERAGE_SEAL.json'),qcenter_tables=bind(out),qcenter_csv=bind(cp),code=bind(Path(__file__)),new_current_forward_fit_PnP=0))
print(json.dumps(bind(OUT/'FINAL_COVERAGE_SEAL.json')))
