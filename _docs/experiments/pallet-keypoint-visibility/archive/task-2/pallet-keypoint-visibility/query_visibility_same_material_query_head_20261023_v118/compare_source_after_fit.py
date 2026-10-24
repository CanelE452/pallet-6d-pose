"""Saved PUBLIC-source comparison. Run only after the once-fixed C13 fit."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
V=Path(__file__).resolve().parent;P=V.parent
S=P/'query_visibility_same_material_foreground_source_20261023_v116'
C12=P/'query_visibility_material_augmented_query_head_20261023_v114'
def bind(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def stat(z,a,ix):
 ix=np.asarray(ix);zz=z[ix];aa=a[ix];pr=np.where(aa,(zz>=0).astype(int),-1)
 return dict(BEFORE=float(zz[0]),COVER=float(zz[1]),SHAM=float(zz[2]),pred_BEFORE_COVER_SHAM=pr.tolist(),
  cover_minus_sham=float(np.float64(zz[1])-np.float64(zz[2])),margin_ge2=bool(aa[1:].all() and np.float32(zz[1]-zz[2])>=2),
  cover_E=bool(pr[1]==1),before_D=bool(pr[0]==0),sham_D=bool(pr[2]==0),all3_correct=bool(np.array_equal(pr,[0,1,0])),U=int((~aa).sum()))
def summarize(rs):
 out={}
 for name in ['ALL','TRAIN','VALIDATION','VALIDATION_WOOD16','VALIDATION_OTHER']:
  r=[x for x in rs if name=='ALL' or x['split']==name or (name=='VALIDATION_WOOD16' and x['split']=='VALIDATION' and x['donor_object_id']==16) or (name=='VALIDATION_OTHER' and x['split']=='VALIDATION' and x['donor_object_id']!=16)]
  if not r:continue
  group=dict(N=len(r),base_queries=len(set(x['original_query_id'] for x in r)),base_images=len(set(x['base_image_sha256'] for x in r)),donors=len(set(x['donor_id'] for x in r)),models={})
  for model in ['C11','C12','C13']:
   if model not in r[0]:continue
   group['models'][model]={k:sum(bool(x[model].get(k, x[model].get('absolute_cover_ge0',False) if k=='cover_E' else False)) for x in r) for k in ['cover_E','before_D','sham_D','all3_correct','margin_ge2']}
   # Existing C11/C12 records encode branch decisions rather than individual convenience fields.
   if model!='C13':
    for key,col,val in [('before_D',0,0),('cover_E',1,1),('sham_D',2,0)]:group['models'][model][key]=sum(x[model]['pred_BEFORE_COVER_SHAM'][col]==val for x in r)
   group['models'][model]['U_branch_responses']=sum(sum(p==-1 for p in x[model]['pred_BEFORE_COVER_SHAM']) for x in r)
   group['models'][model]['mean_cover_E_BCE']=float(np.mean([np.logaddexp(0.,-x[model]['COVER']) for x in r]))
   group['models'][model]['mean_relative_hinge']=float(np.mean([max(0.,2-(x[model]['COVER']-x[model]['SHAM'])) for x in r]))
  out[name]=group
 return out
def main(score_path,out):
 assert not out.exists()
 with np.load(score_path,allow_pickle=False) as f:
  z=f['query_logit'].copy();a=f['query_available'].copy();q=f['query_xy_raw'].copy();L=f['L_raw'].copy()
  state=f['raw_teacher_state'].copy();split=f['split'].copy();domain=f['domain'].copy()
 assert z.shape==a.shape==(46108,) and np.isfinite(z).all()
 old=rows(C12/'public_source_regression_inventory/ALL194_ACCEPTED_TRIPLETS.jsonl')
 assert len(old)==194
 for r in old:
  ix=r['triple_global_indices'];assert np.array_equal(state[ix],['D','E','D'])
  assert np.array_equal(q[ix],np.tile(r['query_xy'],(3,1))) and np.array_equal(L[ix],np.repeat(r['L'],3))
  r['C13']=stat(z,a,ix)
 inputs=rows(S/'corpus_build_v1/OCCURRENCES924_INPUT_ONLY.jsonl');gt=rows(S/'corpus_build_v1/OCCURRENCES924_GT_ONLY.jsonl')
 feas=rows(S/'corpus_build_v1/FEASIBILITY448_GT_ONLY.jsonl');new=[]
 for r in feas:
  if not r['pair_ready']:continue
  bi=r['base_index'];dr=r['donor_rank'];loc=[bi*33,bi*33+1+2*dr,bi*33+2+2*dr];ix=np.array(loc)+45184
  assert [inputs[i]['variant'] for i in loc]==['BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM']
  assert [gt[i]['candidate_aux_hidden_label_GT_only'] for i in loc]==[0,1,0]
  assert np.array_equal(state[ix],['D','E','D']) and (split[ix]=='TRAIN').all()
  assert np.array_equal(q[ix],np.tile(r['original_q'],(3,1))) and np.array_equal(L[ix],np.repeat(r['original_L'],3))
  b=inputs[loc[0]]
  new.append(dict(attempt_index=r['attempt_index'],triple_global_indices=ix.tolist(),original_query_id=b['occurrence_id'],base_image_sha256=b['source_original_image_sha256'],donor_id=r['donor_id'],donor_object_id=16,split='TRAIN',C13=stat(z,a,ix)))
 assert len(new)==62
 source_cm={};pred=np.where(a,(z>=0).astype(int),-1)
 for dm in np.unique(domain):
  source_cm[str(dm)]={}
  for sp in np.unique(split):
   mask=(domain==dm)&(split==sp)
   if not mask.any():continue
   cm=np.array([[int((mask&(state==st)&(pred==pp)).sum()) for pp in [0,1,-1]] for st in ['D','E']])
   counts=cm.sum(1);known=int(counts.sum())
   source_cm[str(dm)][str(sp)]=dict(total_occurrences=int(mask.sum()),typed_states={str(k):int((mask&(state==k)).sum()) for k in 'DESU'},
    factual_U_all_states=int((mask&~a).sum()),rows_D_E_cols_D_E_U=cm.tolist(),
    recall={k:float(cm[i,i]/counts[i]) if counts[i] else None for i,k in enumerate(['D','E'])},
    precision={k:float(cm[i,i]/cm[:,i].sum()) if cm[:,i].sum() else None for i,k in enumerate(['D','E'])},
    known_DE_U_rate=float(cm[:,2].sum()/known) if known else None)
 summary=dict(completed=True,scope='Saved source only after frozen once-fit; diagnostic, no re-selection.',scores=bind(score_path),source_domain_split_confusions=source_cm,old194=summarize(old),new62_TRAIN_only=summarize(new),
  old_C12E_C13D_regressions=sum(r['C12']['pred_BEFORE_COVER_SHAM'][1]==1 and not r['C13']['cover_E'] for r in old if r['split']=='VALIDATION'),
  old_C12D_C13E_fixes=sum(r['C12']['pred_BEFORE_COVER_SHAM'][1]==0 and r['C13']['cover_E'] for r in old if r['split']=='VALIDATION'),
  old_VAL_margin_pass_but_C13_cover_D=sum(r['C13']['margin_ge2'] and not r['C13']['cover_E'] for r in old if r['split']=='VALIDATION'),
  original_q_L_and_folds_exact=True,current223_V112_fit_or_PnP=0,natural_transfer_proven=False)
 out.mkdir()
 for name,rr in [('ALL194_OLD_WITH_C13.jsonl',old),('ALL62_NEW_WITH_C13.jsonl',new)]:
  (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rr))
 (out/'RESULT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--scores',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args();main(x.scores,x.output)
