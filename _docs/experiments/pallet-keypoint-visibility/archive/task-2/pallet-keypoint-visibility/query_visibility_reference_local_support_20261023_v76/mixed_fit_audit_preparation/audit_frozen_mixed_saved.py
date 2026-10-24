"""Independent future V76 saved-source audit. NumPy only; no model/torch imports.

Default is metadata/math mock. Actual mode requires explicit saved output,
frozen source inputs and GT paths; input chunks expose only coords/validity.
RGB/P3 tensors are never decoded. U NLL placeholders never enter metrics.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
R=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def load(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def save(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def bce(z,y):return np.logaddexp(0,z)-y*z

def metric(z,nll,y,m):
 yy=y[m].astype(int);pred=(z[m]>=0).astype(int);cm=np.zeros((2,2),np.int64);np.add.at(cm,(yy,pred),1)
 def div(a,b):return float(a/b)if b else None
 return {'N':int(m.sum()),'confusion_rows_GT_cols_pred_COVERED_VISIBLE':cm.tolist(),'visible_precision':div(cm[1,1],cm[:,1].sum()),'visible_recall':div(cm[1,1],cm[1].sum()),'covered_precision':div(cm[0,0],cm[:,0].sum()),'covered_recall':div(cm[0,0],cm[0].sum()),'mean_BCE_FP32':float(nll[m].astype(np.float64).mean())if m.any()else None}

def mock():
 y=np.array([0,1,-1,1,0]);z=np.array([-2.,3.,999.,-1.,2.]);m=y>=0;n=bce(z,np.clip(y,0,1));a=metric(z,n,y,m)
 assert a['N']==4 and a['confusion_rows_GT_cols_pred_COVERED_VISIBLE']==[[1,1],[1,1]]
 assert a['visible_precision']==a['visible_recall']==a['covered_recall']==.5
 assert np.allclose(bce(np.array([0.,0.]),np.array([0,1])),np.log(2))
 assert np.isfinite(bce(np.array([-1000.,1000.]),np.array([0,1]))).all()
 assert metric(np.zeros(2),np.full(2,np.log(2)),np.array([0,1]),np.ones(2,bool))['confusion_rows_GT_cols_pred_COVERED_VISIBLE']==[[0,1],[0,1]]
 save(R/'MOCK_AUDITOR_TESTS.json',{'PASS':True,'code':bind(Path(__file__)),'U_excluded':True,'fixed_zero_logit_visible_at_half':True,'extreme_logit_BCE_finite':True,'source_or_current_data_read':0,'model_forward_fit_PnP':0})
 print('MOCK_PASS')

def read_input_facts(descriptor):
 facts={k:[]for k in ['query8_xy','query_center_valid','ROI_anyvalid','L']};locals=[];targets=[];lineage=[]
 d=load(descriptor)
 for pack in d['packs']:
  for name in ['manifest','seal','lineage']:check(pack[name])
  man=load(pack['manifest']['path']);assert man['completed'];lineage.extend(json.loads(s)for s in Path(pack['lineage']['path']).read_text().splitlines())
  for chunk in man['chunks']:
   check(chunk['file'])
   with np.load(chunk['file']['path'],allow_pickle=False)as z:
    for k in facts:facts[k].append(z[k])
    qn=len(z['L']);locals.append(z['crop_valid96'].reshape(qn,8,-1).any(-1));targets.append(z['target_valid128'].reshape(qn,-1).any(-1))
 return {k:np.concatenate(v)for k,v in facts.items()},np.concatenate(locals),np.concatenate(targets),lineage

def actual(a):
 out=a.output;out.mkdir(parents=True,exist_ok=False)
 protocol=load(a.protocol);freeze=load(a.freeze);hist=load(a.history);published=load(a.results)
 assert freeze['completed']and freeze['visible_probability_threshold']==.5 and freeze['current_used_for_fit_selection']is False and freeze['PnP']==0
 assert freeze['optimizer_updates']==2000 and freeze['parameter_count']==41237
 for key in ['checkpoint','model_code','protocol','trainer','entry','source_inputs']:check(freeze[key])
 assert Path(freeze['protocol']['path']).resolve()==a.protocol.resolve() and Path(freeze['source_inputs']['path']).resolve()==a.source_inputs.resolve()
 for v in protocol['bindings'].values():check(v)
 assert Path(protocol['bindings']['teacher_GT']['path']).resolve()==a.teacher_GT.resolve()
 with np.load(a.teacher_GT,allow_pickle=False)as z:gt={k:z[k]for k in ['id','source','split','label8','native_GTref8','near_boundary8','rule8','convex_self_proxy8_separate','rho_px']}
 f=len(gt['id']);assert f==5203 and gt['label8'].shape==(f,8)
 facts,local,target,lineage=read_input_facts(a.source_inputs)
 assert len(lineage)==f and all(r['id']==gt['id'][i]and {'validation':'val'}.get(r['split'],r['split'])==gt['split'][i]for i,r in enumerate(lineage))
 L=facts['L'];assert np.isfinite(L).all()and(L>0).all()
 offset=(gt['native_GTref8']-facts['query8_xy'])/L[:,None,None]
 sq=np.isfinite(offset).all(-1)&(offset>=-.5).all(-1)&(offset<.5).all(-1)
 available=facts['ROI_anyvalid']&local&target[:,None];known=np.isin(gt['label8'],[0,1]);primary=known&facts['query_center_valid']&available&sq
 with np.load(a.scores,allow_pickle=False)as z:
  for k in ['id','source','split']:assert np.array_equal(z[k],gt[k])
  assert np.array_equal(z['primary_loss_mask'],primary)and np.array_equal(z['reference_in_local_square_GT_only'],sq)and np.array_equal(z['input_available'],available)
  logits=z['support_logit'];nll=z['NLL_FP32'];assert logits.shape==nll.shape==(f,8)
 assert np.isfinite(logits).all()and np.isfinite(nll).all()
 independent=bce(logits.astype(float),np.clip(gt['label8'],0,1));diff=abs(independent[primary]-nll[primary]);assert diff.max()<1e-5
 # Fixed chosen source history only. No new selection occurs.
 assert [h['update']for h in hist]==list(range(100,2001,100));best=float('inf');step=None
 for h in hist:
  assert len(h['stratumBCE'])==4 and float(np.mean(list(h['stratumBCE'].values())))==h['heldout4strata_macroBCE']
  yes=h['heldout4strata_macroBCE']<best
  if yes:best=h['heldout4strata_macroBCE'];step=h['update']
  assert yes==h['saved']and step==h['best_update']
 assert step==freeze['best_update']==published['best_update']and best==freeze['best_heldout4strata_macroBCE']==published['selection_macroBCE']
 rows=[];strata=[]
 # Entire continuous disk containment is a diagnostic, never an admission edit.
 half=L[:,None]/2;delta=gt['native_GTref8']-facts['query8_xy'];rho=gt['rho_px'][:,None]
 disk_full=np.isfinite(delta).all(-1)&(delta-rho[:,:,None]>=-half[:,:,None]).all(-1)&(delta+rho[:,:,None]<half[:,:,None]).all(-1)
 for source in ['SYNTHETIC','YCB_PUBLIC_REAL']:
  for split in ['train','val']:
   base=np.broadcast_to(((gt['source']==source)&(gt['split']==split))[:,None],(f,8));m=base&primary
   met=metric(logits,nll,gt['label8'],m);pub=published['source_metrics'][source+'|'+split]
   assert met==pub['metrics'],(source,split,met,pub['metrics'])
   subs={}
   for name,c in {'near_boundary':gt['near_boundary8'],'not_near_boundary':~gt['near_boundary8'],'strong_center':np.char.startswith(gt['rule8'],'STRONG'),'fallback':~np.char.startswith(gt['rule8'],'STRONG'),'convex_self_hidden':gt['convex_self_proxy8_separate']==0,'convex_self_may_visible':gt['convex_self_proxy8_separate']==1}.items():
    subs[name]=metric(logits,nll,gt['label8'],m&c);assert subs[name]==pub['conditional_subgroups'][name]
   categories={'teacher_U':base&~known,'known_query_invalid':base&known&~facts['query_center_valid'],'known_query_valid_input_unavailable':base&known&facts['query_center_valid']&~available,'known_query_valid_available_ref_MISS':base&known&facts['query_center_valid']&available&~sq,'admitted':m}
   counts={k:int(v.sum())for k,v in categories.items()};assert sum(counts.values())==base.sum()
   row={'source':source,'split':split,'metrics':met,'admission_exclusive_counts':counts,'all_slots':int(base.sum()),'known':int((base&known).sum()),'teacher_known_reference_outside_square':int((base&known&~sq).sum()),'conditional_subgroups':subs,'admitted_teacher_disk_full_in_local_crop':int((m&disk_full).sum()),'admitted_teacher_disk_partial_in_local_crop':int((m&~disk_full).sum()),'self_hidden_local_VISIBLE_admitted':int((m&(gt['convex_self_proxy8_separate']==0)&(gt['label8']==1)).sum())};rows.append(row)
   for c in [0,1]:
    mm=m&(gt['label8']==c);strata.append({'source':source,'split':split,'class':c,'N':int(mm.sum()),'mean_BCE_FP32':float(nll[mm].astype(float).mean())})
 macro=float(np.mean([r['mean_BCE_FP32']for r in strata if r['split']=='val']))
 report={'schema':'V76_INDEPENDENT_FROZEN_MIXED_SAVED_AUDIT','PASS':True,'source_rows':f,'slots':f*8,'primary_admission_reconstructed_exact':True,'published_metrics_subgroups_exact':True,'checkpoint':freeze['checkpoint'],'best_update':step,'history_macroBCE':best,'saved_allslot_macroBCE':macro,'saved_minus_history_macroBCE':macro-best,'NLL_FP64_vs_saved_FP32_maxdiff_primary_only':float(diff.max()),'source_metrics':rows,'four_strata':strata,'input_arrays_decoded':['query8_xy','query_center_valid','ROI_anyvalid','L','crop_valid96','target_valid128'],'RGB_P3_arrays_decoded':0,'saved_U_NLL_used':False,'teacher_disk_containment_used_to_change_admission':False,'model_current_forward_fit_PnP':0,'scope':'Conditional source reference-local support, not human semantic DES, exactquerypixel or public initializer/endtoend performance.','bindings':[bind(p)for p in [a.scores,a.teacher_GT,a.protocol,a.source_inputs,a.freeze,a.history,a.results,Path(__file__)]]}
 save(out/'INDEPENDENT_SAVED_RESULT_AUDIT.json',report);save(out/'AUDIT_SEAL.json',{'bindings':[bind(p)for p in sorted(out.iterdir())if p.is_file()],'run_files_modified':False});print(json.dumps({'PASS':True,'report':bind(out/'INDEPENDENT_SAVED_RESULT_AUDIT.json')}))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mock',action='store_true')
 for k in ['scores','teacher_GT','protocol','source_inputs','freeze','history','results','output']:p.add_argument('--'+k.replace('_','-'),dest=k,type=Path)
 a=p.parse_args()
 if a.mock:mock();return
 if any(getattr(a,k)is None for k in ['scores','teacher_GT','protocol','source_inputs','freeze','history','results','output']):p.error('Actual mode requires all explicit saved-source paths')
 actual(a)
if __name__=='__main__':main()
