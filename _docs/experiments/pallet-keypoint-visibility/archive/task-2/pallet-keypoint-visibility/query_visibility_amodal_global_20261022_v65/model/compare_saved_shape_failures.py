"""Compare frozen v64/v65 SOURCE scores; never opens RGB/masks or runs a model.

The six preselected qualitative cases are fixed by the v64 review's exact saved
output SHA. Their changes are descriptive and cannot select a model/threshold.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
V64=ROOT.parent/'query_visibility_multitarget_20261022_v64'
PAIRS=np.array(((0,0),(1,0),(1,1),(2,0),(2,1),(2,2)),np.int64)
NAMES=('VISIBLE_TARGET','EXTERNAL_SUPPORT_PROXY','OUTSIDE_FULL_SUPPORT','UNKNOWN')


def sha256(path):
 d=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()


def metadata(cache,freeze):
 digest=sha256(cache)
 allowed={freeze['train_sha256'],freeze['val_sha256']}
 allowed.update(v for k,v in freeze.get('source_provenance',{}).get('content',{}).get('bindings',{}).items()
                if str(k).startswith('source_cache/')and str(k).endswith('_GLOBAL.npz'))
 if digest not in allowed:raise AssertionError('Source cache is not bound by its frozen run')
 with np.load(cache,allow_pickle=False)as z:
  return {k:z[k]for k in ('sample_id','image_sha256','source','joint','dense_valid')}


def read_outputs(path,cache):
 with np.load(path,allow_pickle=False)as z:data={k:z[k]for k in z.files}
 ix=data['index'];p=data['joint_probability'];n=len(ix)
 if ix.shape!=(n,)or not np.issubdtype(ix.dtype,np.integer)or len(np.unique(ix))!=n or(ix<0).any()or(ix>=len(cache['sample_id'])).any():
  raise AssertionError('Invalid/duplicate output-to-source cache indices')
 if p.shape!=(n,6)or not np.isfinite(p).all()or(p<0).any()or not np.allclose(p.sum(1),1,atol=1e-10):
  raise AssertionError('Invalid six-joint saved probabilities')
 if not cache['dense_valid'][ix,4,4].all():raise AssertionError('Invalid scalar label in saved output')
 truth=cache['joint'][ix,4,4]
 if not np.array_equal(truth,data['truth_joint'])or not np.array_equal(PAIRS[truth],data['truth_classes']):raise AssertionError('Saved source labels changed')
 if not np.array_equal(data['source'],cache['source'][ix]):raise AssertionError('Saved source domain identity changed')
 for k in ('sample_id','image_sha256'):
  if k in data and not np.array_equal(data[k],cache[k][ix]):raise AssertionError('Saved query identity mismatch')
 data.update({k:cache[k][ix].astype(str)for k in ('sample_id','image_sha256','source')})
 if len(set(data['sample_id']))!=n:raise AssertionError('Duplicate source sample IDs')
 data['truth_classes']=PAIRS[truth]
 data['p_visible']=np.stack((p[:,0],p[:,0]+p[:,1]+p[:,3]),1)
 data['p_external']=np.stack((p[:,1]+p[:,2],p[:,2]+p[:,4]),1)
 data['p_full']=np.stack((p[:,:3].sum(1),p[:,:5].sum(1)),1)
 data['logical']=np.where(data['p_visible']>=.5,0,np.where(data['p_full']>=.5,1,2))
 data['confidence']=np.where((p.max(1)>=.5)[:,None],PAIRS[p.argmax(1)],3)
 data['max_joint']=p.max(1)
 return data


def transition(old,new,states=4):
 cm=np.zeros((states,states),np.int64);np.add.at(cm,(old,new),1);return cm.tolist()


def summary(old,new,mask):
 out={'N':int(mask.sum()),'scales':{}}
 if not mask.any():return out
 for scale,name in enumerate(('exact','radius2')):
  truth=old['truth_classes'][mask,scale];report={}
  for key in ('p_visible','p_external','p_full'):
   a,b=old[key][mask,scale],new[key][mask,scale];delta=b-a
   report[key]={'v64_mean':float(a.mean()),'v65_mean':float(b.mean()),'mean_delta':float(delta.mean()),
                'delta_quantiles_0_25_50_75_100':np.quantile(delta,[0,.25,.5,.75,1]).tolist(),
                'increased':int((delta>0).sum()),'decreased':int((delta<0).sum())}
  for decoder in ('logical','confidence'):
   a,b=old[decoder][mask,scale],new[decoder][mask,scale]
   ca,cb=a==truth,b==truth
   report[decoder]={'transition_v64rows_v65columns':transition(a,b),
    'v64_correct':int(ca.sum()),'v65_correct':int(cb.sum()),'newly_correct':int((~ca&cb).sum()),
    'newly_wrong':int((ca&~cb).sum()),'both_wrong':int((~ca&~cb).sum()),
    'v64_unknown':int((a==3).sum()),'v65_unknown':int((b==3).sum()),
    'became_unknown':int(((a!=3)&(b==3)).sum()),'became_decided':int(((a==3)&(b!=3)).sum())}
  out['scales'][name]=report
 return out


def case_score(data,i):
 p=data['joint_probability'][i]
 return {'joint_probability':p.tolist(),'p_visible_exact':float(data['p_visible'][i,0]),
  'p_external_exact':float(data['p_external'][i,0]),'p_full_exact':float(data['p_full'][i,0]),
  'p_visible_radius2':float(data['p_visible'][i,1]),'p_external_radius2':float(data['p_external'][i,1]),
  'p_full_radius2':float(data['p_full'][i,1]),'max_joint':float(p.max()),
  'logical_states':[NAMES[v]for v in data['logical'][i]],
  'confidence_states':[NAMES[v]for v in data['confidence'][i]],'confidence_U':bool(p.max()<.5)}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--v64-outputs',type=Path,default=V64/'runs/MIXED_MULTITARGET_UNWEIGHTED/VAL_SOURCE_OUTPUTS.npz')
 p.add_argument('--v64-cache',type=Path,default=V64/'source_cache/MIXED_VALIDATION_WITH_SAM.npz')
 p.add_argument('--v64-freeze',type=Path,default=V64/'runs/MIXED_MULTITARGET_UNWEIGHTED/MODEL_FREEZE.json')
 p.add_argument('--v65-outputs',type=Path,default=ROOT/'runs/MIXED_GLOBAL_UNWEIGHTED/VAL_SOURCE_OUTPUTS.npz')
 p.add_argument('--v65-cache',type=Path,default=ROOT/'source_cache/MIXED_VALIDATION_GLOBAL.npz')
 p.add_argument('--v65-freeze',type=Path,default=ROOT/'runs/MIXED_GLOBAL_UNWEIGHTED/MODEL_FREEZE.json')
 p.add_argument('--cases',type=Path,default=V64/'source_audit/FULL_SHAPE_FAILURES/REVIEW_CASES.json')
 p.add_argument('--evidence',type=Path,default=V64/'source_audit/YCB_VALIDATION_QUERY_EVIDENCE.csv')
 p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if any(a.out.with_suffix(s).exists()for s in ('.json','.csv')):raise FileExistsError(a.out)
 review=json.loads(a.cases.read_text())
 if not review['source_only']or review['current_private_data_read']or not review['selection_not_representative']:raise AssertionError('Expected fixed SOURCE qualitative selection')
 if sha256(a.v64_outputs)!=review['source_outputs_sha256']:raise AssertionError('v64 scores differ from preselected six-case review binding')
 freezes=[json.loads(path.read_text())for path in (a.v64_freeze,a.v65_freeze)]
 if any(not f.get('completed')or f.get('eval_data_used')or f.get('PnP_used')for f in freezes):raise AssertionError('Completed SOURCE-only frozen runs required')
 old=read_outputs(a.v64_outputs,metadata(a.v64_cache,freezes[0]));new=read_outputs(a.v65_outputs,metadata(a.v65_cache,freezes[1]))
 if len(old['sample_id'])!=4942 or set(old['sample_id'])!=set(new['sample_id']):raise AssertionError('Exactly same4942 source queries required')
 newlookup={sid:i for i,sid in enumerate(new['sample_id'])};order=np.array([newlookup[s]for s in old['sample_id']])
 new={k:v[order]if isinstance(v,np.ndarray)and len(v)==len(order)else v for k,v in new.items()}
 for key in ('source','image_sha256','truth_classes'):
  if not np.array_equal(old[key],new[key]):raise AssertionError('Matched source identities/labels disagree')
 with a.evidence.open(newline='')as f:evidence={r['sample_id']:r for r in csv.DictReader(f)}
 strict=np.array([evidence.get(s,{}).get('strict_external_corroborated_5x5')=='1'for s in old['sample_id']])
 if int(strict.sum())!=review['strict_corroborated_E_population']:raise AssertionError('Fixed strict source cohort changed')
 if {s:int((old['source']==s).sum())for s in ('SYNTH','YCB')}!={'SYNTH':1742,'YCB':3200}:raise AssertionError('Source population changed')
 groups={'ALL':np.ones(4942,bool),'SYNTH':old['source']=='SYNTH','YCB':old['source']=='YCB',
  'YCB_strict_external_corroborated_5x5':strict}
 for source in ('SYNTH','YCB'):
  for state,code in (('V',0),('E',1),('O',2)):
   groups[f'{source}_truth_exact_{state}']=(old['source']==source)&(old['truth_classes'][:,0]==code)
 selected=[];lookup={sid:i for i,sid in enumerate(old['sample_id'])}
 for case in review['cases']:
  i=lookup[case['sample_id']]
  if old['image_sha256'][i]!=case['image_sha256']or old['truth_classes'][i,0]!=1 or not strict[i]:raise AssertionError('Fixed review source case contract changed')
  if abs(old['p_visible'][i,0]-case['p_visible'])>1e-12 or abs(old['p_full'][i,0]-case['p_full'])>1e-12:raise AssertionError('Fixed case saved scores changed')
  before,after=case_score(old,i),case_score(new,i)
  delta={key:after[key]-before[key]for key in ('p_visible_exact','p_external_exact','p_full_exact','p_visible_radius2','p_external_radius2','p_full_radius2','max_joint')}
  selected.append({'sample_id':case['sample_id'],'object_id':case['object_id'],'image_sha256':case['image_sha256'],
   'truth_exact':'EXTERNAL_SUPPORT_PROXY','target_visible_fraction':case['target_visible_fraction_in_frame_full'],
   'v64':before,'v65':after,'delta_v65_minus_v64':delta,
   'directional_checks':{'visible_probability_decreased':delta['p_visible_exact']<0,'full_probability_increased':delta['p_full_exact']>0,
    'external_probability_increased':delta['p_external_exact']>0,'v65_logical_E':new['logical'][i,0]==1,
    'v65_confident_E':new['confidence'][i,0]==1}})
 result={'schema':'SOURCE_SAVED_V64_V65_SHAPE_FAILURE_COMPARISON','N':4942,
  'fixed_six_selection_nonrepresentative':True,'fixed_six_not_used_for_selection':True,
  'classes_in_transition_order':NAMES,'decoder_thresholds':{'visible':.5,'full':.5,'max_joint_confidence':.5},
  'caveat':'E is source full-mask invisible-support proxy. Logical E is not confidence-qualified. Probability changes alone do not establish physical visibility.',
  'fixed_cases':selected,'matched_population':{name:summary(old,new,mask)for name,mask in groups.items()},
  'bindings':{name:{'path':str(path.resolve()),'sha256':sha256(path)}for name,path in vars(a).items()if isinstance(path,Path)and name!='out'},
  'code_sha256':sha256(__file__),'new_model_inference':False,'optimizer_updates':0,'RGB_or_masks_opened':False,
  'current_private_data_read':False,'threshold_or_model_selection':False,'PnP_calls':0}
 # Convert NumPy booleans from the descriptive fixed-case checks before JSON.
 for case in selected:case['directional_checks']={k:bool(v)for k,v in case['directional_checks'].items()}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 records=[]
 for c in selected:
  row={k:c[k]for k in ('sample_id','object_id','truth_exact','target_visible_fraction')}
  for version in ('v64','v65'):
   row.update({version+'_'+k:v for k,v in c[version].items()if k!='joint_probability'and not isinstance(v,list)})
   row[version+'_logical_exact']=c[version]['logical_states'][0];row[version+'_confidence_exact']=c[version]['confidence_states'][0]
  row.update({'delta_'+k:v for k,v in c['delta_v65_minus_v64'].items()});records.append(row)
 with a.out.with_suffix('.csv').open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 print(json.dumps({'output':str(a.out.with_suffix('.json')),'matched_queries':4942,'fixed_cases':len(selected),'new_model_inference':False}))

if __name__=='__main__':main()
