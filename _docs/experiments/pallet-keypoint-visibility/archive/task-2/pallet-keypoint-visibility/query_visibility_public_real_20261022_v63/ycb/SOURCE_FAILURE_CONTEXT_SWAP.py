#!/usr/bin/env python3
"""Source-only CPU ownership counterfactual and paired stored baseline diagnostics."""
import csv,hashlib,importlib.util,json,pathlib,sys,time
from collections import defaultdict
import cv2,numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=pathlib.Path(__file__).resolve().parent.parent;OUT=ROOT/'ycb';C=ROOT/'source_cache/v2'
torch.set_num_threads(1);cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-80,80)))
def boolstr(x):return x=='True'
def metric(y,p,pred=None):
 y=np.asarray(y).astype(bool);p=np.asarray(p);pred=p>=.5 if pred is None else np.asarray(pred).astype(bool);tp=int((y&pred).sum());fp=int((~y&pred).sum());fn=int((y&~pred).sum());tn=int((~y&~pred).sum())
 return {'N':len(y),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/(tp+fp)if tp+fp else None,'recall':tp/(tp+fn)if tp+fn else None,'FPR':fp/(fp+tn)if fp+tn else None,'AP':float(average_precision_score(y,p))if y.any()else None,'AUC':float(roc_auc_score(y,p))if len(set(y))==2 else None}
def stats(x):
 x=np.asarray(x);return {'min':float(x.min()),'p10':float(np.quantile(x,.1)),'median':float(np.median(x)),'p90':float(np.quantile(x,.9)),'max':float(x.max()),'mean':float(x.mean())}
def load_snapshot(run):
 f=ROOT/f'runs/{run}/code/query_head.py';name='source_diagnostic_'+run;spec=importlib.util.spec_from_file_location(name,f);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);model,payload=mod.load_model(ROOT/f'runs/{run}/BEST_QUERY_HEAD.pt','cpu');model.eval();return model,payload,sha(f)
def predict(model,data,context):
 values=[]
 for st in range(0,len(data['y']),128):
  sl=slice(st,st+128);x=torch.from_numpy(data['x'][sl].astype(np.float32));ctx=torch.from_numpy(context[sl].astype(np.float32));rgb=torch.from_numpy(data['rgb'][sl].astype(np.float32)/255)if model.config.rgb_branch else None
  with torch.no_grad():values.append(model(x,rgb=rgb,context=ctx).numpy())
 return np.concatenate(values)

def main():
 start=time.time();report={'scope':'Public source-only stored data plus explicitly authorized CPU source context-swap inference; no private files or scores','private_data_read':False,'GPU_used':False,'training_or_download':False,'active_code_or_cache_modified':False};rows=list(csv.DictReader((C/'YCB_VALIDATION_QUERIES.csv').open()));z=np.load(C/'YCB_VALIDATION_RGB.npz');data={k:z[k]for k in z.files};z.close();assert np.array_equal(data['sample_id'],[r['sample_id']for r in rows]);N=len(rows)
 groups=defaultdict(list);frames=defaultdict(list)
 for i,r in enumerate(rows):groups[r['id']].append(i)
 for fid,ix in groups.items():frames[data['group_id'][ix[0]]].append(fid)
 other=np.empty(N,int);other_id=np.empty(N,dtype='<U24');other_y=np.zeros(N,int);purity=np.zeros(N,float);ctxs=[];group_frames=[];group_ids=[]
 for fid,ix in groups.items():
  peers=frames[data['group_id'][ix[0]]];assert len(peers)==2;peer=next(x for x in peers if x!=fid);jx=groups[peer];other[ix]=jx[0];other_id[ix]=peer
  mp=pathlib.Path(rows[jx[0]]['mask_vis']).resolve();assert mp.is_relative_to((OUT/'data/train_real').resolve());mask=cv2.imread(str(mp),cv2.IMREAD_UNCHANGED)>0
  for i in ix:px,py=np.floor([float(rows[i]['query_x'])+.5,float(rows[i]['query_y'])+.5]).astype(int);other_y[i]=int(mask[py,px])
  valid=data['dense_valid'][ix];pur=float((data['dense_y'][ix]*valid).sum()/max(valid.sum(),1));purity[ix]=pur
  ctxs.append(data['context'][ix[0]]);group_frames.append(data['group_id'][ix[0]]);group_ids.append(fid)
 swap=data['context'][other];changed=data['y'].astype(int)!=other_y
 report['counterfactual_definition']='Replace only 64D prototype by other selected target in the same public image; x, RGB, query, scale unchanged. Counterfactual truth is other target visible mask at the original query pixel. No fitting or threshold adjustment.'
 report['counterfactual_rows']=N;report['rows_with_changed_ownership_truth']=int(changed.sum());report['target_context_visible_fraction']=stats(purity[::8]);report['original_target_presence_prevalence']=float(data['y'].mean());report['other_target_presence_prevalence']=float(other_y.mean())
 ctx=np.array(ctxs);centered=ctx-ctx.mean(0);rng=np.random.default_rng(20261022);random_other=[];paired_other=[]
 for j,fid in enumerate(group_ids):
  paired_other.append(group_ids.index(next(x for x in frames[group_frames[j]]if x!=fid)))
  choices=[k for k in range(len(group_ids))if group_frames[k]!=group_frames[j]];random_other.append(int(rng.choice(choices)))
 def cosine(a,b):return (a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
 report['prototype_similarity']={'same_image_different_target_raw_cosine':stats(cosine(ctx,ctx[paired_other])),'different_image_raw_cosine':stats(cosine(ctx,ctx[random_other])),'same_image_different_target_centered_cosine':stats(cosine(centered,centered[paired_other])),'different_image_centered_cosine':stats(cosine(centered,centered[random_other]))}
 outputs={};report['models']={}
 for run in ['MIXED_CONTEXT_V2','MIXED_CONTEXT_RGB_V3']:
  model,payload,code_sha=load_snapshot(run);base=predict(model,data,data['context']);counter=predict(model,data,swap);p=sigmoid(base);ps=sigmoid(counter);delta=p-ps;direction=delta*(data['y']-other_y)
  saved=np.load(ROOT/f'runs/{run}/VAL_SOURCE_OUTPUTS.npz');ss=saved['source']=='ycbv';replay_error=float(np.max(np.abs(p-saved['probability'][ss])));saved.close()
  report['models'][run]={'checkpoint_sha256':sha(ROOT/f'runs/{run}/BEST_QUERY_HEAD.pt'),'snapshot_code_sha256':code_sha,'source_output_replay_max_abs':replay_error,'correct_context_metrics':metric(data['y'],p),'swapped_context_vs_other_target_truth':metric(other_y,ps),'swapped_context_vs_original_truth':metric(data['y'],ps),'context_change_abs_probability':stats(np.abs(delta)),'probability_shift_above_.05_fraction':float((np.abs(delta)>.05).mean()),'prediction_flip_fraction':float(((p>=.5)!=(ps>=.5)).mean()),'changed_truth_subset':{'N':int(changed.sum()),'correct_direction_fraction':float((direction[changed]>0).mean()),'correct_direction_above_.05_fraction':float((direction[changed]>.05).mean()),'mean_probability_margin_for_correct_ownership':float(direction[changed].mean()),'own_target_correct_original_and_other_target_correct_swapped':float(((p[changed]>=.5)==data['y'][changed]).__and__((ps[changed]>=.5)==other_y[changed]).mean())}}
  outputs[run]=(p,ps,delta)
 # Stored v61 scores and a source-only SAM subset must be paired by query ID.
 with np.load(C/'MIXED_VALIDATION.npz')as m:ids=m['sample_id'].copy()
 with np.load(ROOT/'source_comparison/V61/SOURCE_OUTPUTS.npz')as v:vp=v['probability'].copy();vi=v['index'].copy()
 vd={ids[j]:p for j,p in zip(vi,vp)};v61=np.array([vd[r['sample_id']]for r in rows]);outputs['V61']=(v61,v61,np.zeros(N))
 samrows=[r for r in csv.DictReader((ROOT/'sam/SOURCE_QUERY_SCORED.csv').open())if r['source']=='YCB'and int(r['radius'])==0];sam={r['query_id']:r for r in samrows};ix=np.array([i for i,r in enumerate(rows)if r['sample_id']in sam]);assert len(ix)==800
 report['paired_SAM_subset']={'queries':len(ix),'label_agreement':all(int(boolstr(sam[rows[i]['sample_id']]['target_visible_support']))==int(data['y'][i])for i in ix),'metrics':{}}
 sp=np.array([float(sam[rows[i]['sample_id']]['visible_support_logit'])for i in ix]);spred=sp>=0
 report['paired_SAM_subset']['metrics']['MobileSAM_box']=metric(data['y'][ix],sp,spred)
 for run,(p,_,_)in outputs.items():report['paired_SAM_subset']['metrics'][run]=metric(data['y'][ix],p[ix])
 table=[]
 state=np.array(['visible'if int(r['presence_target']) else'external_proxy'if boolstr(r['external_proxy'])else'outside_support'for r in rows]);boundary=data['boundary'].astype(bool);sides=np.array([float(r['roi_side'])for r in rows]);seen_train={int(x)for x in json.loads((OUT/'LABEL_AND_SPLIT_AUDIT.json').read_text())['object_instance_counts_by_split']['train']};oid=np.array([int(r['object_id'])for r in rows]);seen=np.array([x in seen_train for x in oid])
 strata={'ALL':np.ones(N,bool),'boundary':boundary,'nonboundary':~boundary,'seen_object':seen,'unseen_object':~seen}
 for s in np.unique(state):strata['state='+s]=state==s
 for o in sorted(set(oid)):strata['object='+str(o)]=oid==o
 for name,select in [('ROI<=32',sides<=32),('ROI32..48',(sides>32)&(sides<=48)),('ROI48..64',(sides>48)&(sides<=64)),('ROI>64',sides>64),('context_visible<.25',purity<.25),('context_visible.25...5',(purity>=.25)&(purity<.5)),('context_visible>=.5',purity>=.5)]:strata[name]=select
 for run,(p,_,_)in outputs.items():
  for name,select in strata.items():
   if select.any():table.append({'model':run,'stratum':name,**metric(data['y'][select],p[select])})
 for name,select in strata.items():
  subset=np.array([k for k,j in enumerate(ix)if select[j]])
  if len(subset):table.append({'model':'SAM_paired_subset','stratum':name,**metric(data['y'][ix[subset]],sp[subset],spred[subset])})
 report['YCB_native_ROI_side_distribution']=stats(sides);report['YCB_P3_native_stride']=8/(640/840);report['YCB_P3_cell_spacing_native']=stats(sides/9);report['YCB_RGB_center_receptive_width_native_for_two3x3conv']=stats(5*sides/64)
 report['source_synth_contract']='M4 >127 visible target support; M0 full support. Values1/2 in synthetic mask background are absent, per existing v63 source contract. Query presence remains nearest integer pixel; source self semantic is unavailable.'
 # Cache spatial label density, with no need to read external synthetic files.
 for split in ['TRAIN','VALIDATION']:
  report['cache_label_distributions_'+split]={}
  for source in ['SYNTH','YCB']:
   with np.load(C/f'{source}_{split}.npz')as x:
    valid=x['dense_valid'];occ=(x['dense_y']*valid).sum((1,2))/np.maximum(valid.sum((1,2)),1);y=x['y'].astype(bool)
    report['cache_label_distributions_'+split][source]={'query_prevalence':float(y.mean()),'boundary_fraction':float(x['boundary'].mean()),'ROI_visible_fraction':stats(occ),'positive_query_ROI_visible_fraction':stats(occ[y]),'negative_query_ROI_visible_fraction':stats(occ[~y])}
 report['seconds']=time.time()-start
 with (OUT/'SOURCE_FAILURE_STATE_METRICS.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 queryrows=[]
 for i,r in enumerate(rows):
  item=dict(r);item.update(other_target_id=str(other_id[i]),other_target_y=int(other_y[i]),ownership_truth_changed=bool(changed[i]),context_visible_fraction=float(purity[i]))
  for run,(p,ps,delta)in outputs.items():item[run+'_p']=float(p[i]);item[run+'_swapped_p']=float(ps[i]);item[run+'_own_minus_swap']=float(delta[i])
  queryrows.append(item)
 with (OUT/'SOURCE_FAILURE_CONTEXT_SWAP_QUERIES.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(queryrows[0]));w.writeheader();w.writerows(queryrows)
 (OUT/'SOURCE_FAILURE_CONTEXT_SWAP.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
