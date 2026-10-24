"""Prospectively specified source-only saved diagnostics; no model execution."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent;RUN=R/'runs/SOURCE_DENSE_AUX_V1'
V76=P/'query_visibility_reference_local_support_20261023_v76';V77=P/'query_visibility_controlled_pairs_20261023_v77';V78=P/'query_visibility_dino_query_support_20261023_v78';V79=P/'query_visibility_natural_occlusion_audit_20261023_v79'
def load(p):return json.loads(Path(p).read_text())
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p,j):
 with open(p,'x')as f:f.write(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def arrays(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def metrics(z,y,m):
 c=np.zeros((2,2),np.int64);np.add.at(c,(y[m].astype(int),(z[m]>=0).astype(int)),1)
 def d(a,b):return float(a/b)if b else None
 return dict(N=int(m.sum()),CM_rows_GT_CV_cols_pred_CV=c.tolist(),C_precision=d(c[0,0],c[:,0].sum()),C_recall=d(c[0,0],c[0].sum()),V_precision=d(c[1,1],c[:,1].sum()),V_recall=d(c[1,1],c[1].sum()))
def main():
 assert load(RUN/'FIT_COMPLETE.json')['completed']; assert not(R/'saved_analysis').exists()
 out=R/'saved_analysis';out.mkdir()
 old=arrays(V77/'runs/PAIRED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz');dense=arrays(RUN/'SELECTED_SOURCE_SCORES.npz')
 for k in ['id','source','split','domain','primary_loss_mask','input_available','after_original_row_index','after_anchor_k','pair_good']:
  assert np.array_equal(old[k],dense[k]),k
 gt0=arrays(V76/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz');gt1=arrays(V77/'pair_generation/AFTER_GT_ONLY.npz');y=np.r_[gt0['label8'],gt1['label8']]
 assert y.shape==(10002,8);primary=dense['primary_loss_mask'];domain=dense['domain'];src=dense['source'];split=dense['split']
 paths={'V76':V77/'frozen_V76_source_baseline/SOURCE_SCORES.npz','V77':V77/'runs/PAIRED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz','V78':V78/'runs/SOURCE_DINO_V1/SELECTED_SOURCE_SCORES.npz','V80':RUN/'SELECTED_SOURCE_SCORES.npz'}
 fixed=list(csv.DictReader((V79/'coverage/results/FIXED_535_REFERENCE_VS_Q_CENTER.csv').open()));where={str(x):i for i,x in enumerate(dense['id'])};strict=np.zeros_like(primary)
 for row in fixed:strict[where[row['id']],int(row['native_k'])]=True
 assert strict.sum()==535 and np.sum(strict&(y==0))==76 and np.sum(strict&(y==1))==459 and (primary[strict]).all()
 rows=[];pairs=[];strictrows=[]
 for name,path in paths.items():
  z=arrays(path)
  for k in ['id','source','split','domain','primary_loss_mask','input_available']:assert np.array_equal(z[k],dense[k]),(name,k)
  logit=z['support_logit'];assert np.isfinite(logit).all()
  for dom in(0,1):
   for source in('SYNTHETIC','YCB_PUBLIC_REAL'):
    for part in('train','val'):
     raw=((domain==dom)&(src==source)&(split==part))[:,None];m=raw&primary
     rows.append(dict(model=name,domain=dom,source=source,split=part,**metrics(logit,y,m),all_slots=int(np.broadcast_to(raw,y.shape).sum()),teacher_U=int(np.sum(raw&(y==-1))),input_invalid=int(np.sum(raw&~dense['input_available']))))
  strictrows.append(dict(model=name,**metrics(logit,y,strict)))
  ai=np.arange(4799);oi=old['after_original_row_index'];ak=old['after_anchor_k'];bz=logit[oi,ak];az=logit[5203+ai,ak]
  for source in('SYNTHETIC','YCB_PUBLIC_REAL'):
   for part in('train','val'):
    mask=old['pair_good']&(src[5203:]==source)&(split[5203:]==part)
    pairs.append(dict(model=name,source=source,split=part,N=int(mask.sum()),before_V=int(np.sum(mask&(bz>=0))),after_C=int(np.sum(mask&(az<0))),both=int(np.sum(mask&(bz>=0)&(az<0))),both_rate=float(np.mean((bz[mask]>=0)&(az[mask]<0)))))
  del z
 public=next(x for x in rows if x['model']=='V80'and x['domain']==0 and x['source']=='YCB_PUBLIC_REAL'and x['split']=='val')
 pair=next(x for x in pairs if x['model']=='V80'and x['source']=='YCB_PUBLIC_REAL'and x['split']=='val')
 limits={'C_precision':.5,'C_recall':.8,'V_recall':.8};checks={k:dict(value=public[k],minimum=v,PASS=bool(public[k]is not None and public[k]>=v))for k,v in limits.items()}
 checks['controlled_pair_both']=dict(value=pair['both_rate'],minimum=.7,PASS=pair['both_rate']>=.7)
 passed=all(x['PASS']for x in checks.values())
 write(R/'SOURCE_ADMISSION_DECISION.json',dict(completed=True,source_candidate_pass=passed,checks=checks,fit=bind(RUN/'FIT_COMPLETE.json'),decisions=bind(R/'EXPERIMENT_DECISIONS_BEFORE_GT_BUILD.json'),threshold=.5,current223_access=False,current_evaluation_automatically_executed=False,fully_solved=False,PnP=0))
 write(out/'PRIMARY_SOURCE_COMPARISON.json',dict(completed=True,rows=rows,pairs=pairs,strict535=strictrows,all_primary_masks_ids_input_available_exact=True,bindings={k:bind(v)for k,v in paths.items()},current223_forward_fit_PnP=0))
 # Conditional auxiliary maps are source-only diagnostics. Argmax ties C,V,O;
 # this is never a new primary threshold, readout, selector or adoption rule.
 ap=RUN/'SELECTED_AUX_LOGITS.npz';aux=arrays(ap);z=aux['aux_mask_logits'].astype(np.float64)
 assert z.shape==(10002,8,2,9,9)and np.isfinite(z).all()
 g=arrays(R/'dense_gt/DENSE_GT9.npz')
 for k in ['id','source','split','domain']:assert np.array_equal(aux[k],dense[k])and np.array_equal(g[k],dense[k])
 s=np.exp(-np.logaddexp(0.,-z));a=s[:,:,0];v=s[:,:,1];prob=np.stack([a*(1-v),a*v,1-a],2);pred=prob.argmax(2)
 assert np.allclose(prob.sum(2),1.,atol=2e-15)
 valid=g['valid9'];yy=np.full(valid.shape,2,np.uint8);yy[g['expected_support_M0']]=0;yy[g['visible_target_M4']]=1
 auxrows=[]
 for dom in(0,1):
  for source in('SYNTHETIC','YCB_PUBLIC_REAL'):
   for part in('train','val'):
    for pop in('ALL_QUERIES','PRIMARY'):
     take=((domain==dom)&(src==source)&(split==part))[:,None,None,None]&valid
     if pop=='PRIMARY':take=take&primary[:,:,None,None]
     c=np.zeros((3,3),np.int64);np.add.at(c,(yy[take],pred[take]),1)
     auxrows.append(dict(domain=dom,source=source,split=part,population=pop,Ncells=int(take.sum()),CM_rows_GT_CVO_cols_pred_CVO=c.tolist(),GT_pixel_count=c.sum(1).tolist(),correct_per_GT=[float(c[i,i]/c[i].sum())if c[i].sum()else None for i in range(3)]))
 write(out/'AUXILIARY_MAP_DIAGNOSTICS.json',dict(completed=True,channel_names=['pM0','pVisibleGivenM0'],joint_probability_rule='pV=pM0*pS,pC=pM0*(1-pS),pO=1-pM0',argmax_tie_order=['C','V','O'],rows=auxrows,GTonlyvalid9=True,source_saved_auxiliary_logits=bind(ap),loss_GT=bind(R/'dense_gt/SEAL.json'),auxiliary_is_not_primary_prediction=True,no_primary_threshold_checkpoint_changes=True,current223_forward_fit_PnP=0))
 np.savez_compressed(out/'FIXED_NATURAL_FOUR_MAPS.npz',id=np.asarray([c['id']for c in load(V79/'natural_failure_gallery/GALLERY_METADATA.json')['cases']]),**{key:np.stack([x[where[c['id']],int(c['native_k'])]for c in load(V79/'natural_failure_gallery/GALLERY_METADATA.json')['cases']])for key,x in [('GT_class_CVO',yy),('valid9',valid),('predicted_class_CVO',pred),('joint_probability_CVO',prob.astype(np.float32))]})
 write(out/'SEAL.json',dict(completed=True,code=bind(__file__),primary=bind(out/'PRIMARY_SOURCE_COMPARISON.json'),auxiliary=bind(out/'AUXILIARY_MAP_DIAGNOSTICS.json'),fixed_four_maps=bind(out/'FIXED_NATURAL_FOUR_MAPS.npz'),gate=bind(R/'SOURCE_ADMISSION_DECISION.json'),fully_solved=False,current223_forward_fit_PnP=0))
 print(json.dumps(dict(gate=bind(R/'SOURCE_ADMISSION_DECISION.json'),source_candidate_pass=passed,strict=next(x for x in strictrows if x['model']=='V80'))),flush=True)
if __name__=='__main__':main()
