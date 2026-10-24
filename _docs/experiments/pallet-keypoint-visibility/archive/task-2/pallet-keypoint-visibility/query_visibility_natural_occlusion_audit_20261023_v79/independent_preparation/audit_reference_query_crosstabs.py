from pathlib import Path
import json,csv,hashlib
import numpy as np
R=Path(__file__).resolve().parent;V=R.parent;D=V/'coverage/results'
def b(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
p=D/'REFERENCE_TEACHER_VS_Q_CENTER.json';report=json.loads(p.read_text())
with np.load(D/'COVERAGE_PER_QUERY.npz',allow_pickle=False)as z:a={k:z[k]for k in ['id','source','split','domain','teacher_label8','primary_loss_mask','q_center_support_state']}
y=a['teacher_label8'];q=a['q_center_support_state'];assert np.isin(y,[-1,0,1]).all()and np.isin(q,[-1,0,1,2]).all()
def cm(m):
 out=np.zeros((3,4),int);np.add.at(out,(y[m]+1,q[m]+1),1);return out
for row in report['tables']:
 mask=np.broadcast_to(((a['domain']==row['domain'])&(a['source']==row['source'])&(a['split']==row['split']))[:,None],y.shape).copy()
 if row['population']=='PRIMARY_UNCHANGED':mask&=a['primary_loss_mask']
 c=cm(mask);pp=row['counts'];assert pp['N']==int(mask.sum())and pp['CM']==c.tolist()
 for key,value in dict(reference_C_with_q_VISIBLE=c[1,2],reference_C_with_q_COVERED=c[1,1],reference_C_with_q_OUTSIDE=c[1,3],reference_C_with_q_INVALID=c[1,0],reference_V_with_q_COVERED=c[2,1]).items():assert pp[key]==value
idx={str(x):i for i,x in enumerate(a['id'])}
with Path(report['fixed_ids']['path']).open()as f:ids=[x for x in csv.DictReader(f)if x['cohort']=='STRONG_AND_ERROR_LE3PX_AND_CENTER_CELL']
m=np.zeros(y.shape,bool)
for row in ids:m[idx[row['id']],int(row['native_k'])]=1
c=cm(m);assert int(m.sum())==535 and report['fixed535']['CM']==c.tolist()and c.tolist()==[[0,0,0,0],[0,76,0,0],[0,0,459,0]]
out=V/'independent_audits/REFERENCE_QUERY_CROSSTAB_AUDIT.json';assert not out.exists();out.write_text(json.dumps(dict(PASS=True,code=b(Path(__file__)),source_crosstabs=b(p),all_source_split_domain_population_tables_recomputed=len(report['tables']),fixed535_reference_C76_query_C76_reference_V459_query_V459=True,fixed535_q_vs_reference_state_mismatch=0,scope='Source mask component correspondence only, not human semantic native8 labels. On this fixed strong/near/central subset, query-pixel versus reference-local label mismatch is zero; this does not explain model errors there. Broader-source O/teacherV mismatches remain typed and are not relabelled.',RGB_model_score_current_forward_fit_PnP=0),indent=2)+'\n');print(json.dumps(b(out)))
