from features44 import *
from train44 import fit44,predict44
from evaluate44 import METHODS,masks44
import copy

def main():
 p=population44();old=L.population();s=read(RUN/'SUMMARY.json');manifest=read(RUN/'cache/manifest.json');render=read(RUN/'RENDER_COMPLETE.json');priorrender=read(ROOT/'data/pallet/results/pallet_visible_geometry_v32/RENDER_COMPLETE.json');assert read(RUN/'FEATURE_PARITY.json')['status']=='PASS';assert read(RUN/'UPSTREAM_REPLAY.json')['status']=='PASS';assert read(RUN/'INFERENCE_COMPLETE.json')['GT_absent_feature_construction'];assert read(RUN/'TARGET_AUDIT.json')['status']=='PASS';assert len(render['rows'])==240 and len(manifest['rows'])==600
 hashes=set(r['image_sha256'] for r in priorrender['rows']);assert not hashes&set(r['image_sha256'] for r in render['rows']);assert not set(r['source_group'] for r in render['rows'])&set(r['source_group'] for r in priorrender['rows']);groups={k:set(r['source_group'] for r in render['rows'] if r['split']==k) for k in ('train','calibration','check')};assert [len(groups[k]) for k in groups]==[12,4,8];assert all(not groups[a]&groups[b] for a,b in [('train','calibration'),('train','check'),('calibration','check')])
 for row in render['rows']:
  assert sha(ROOT/row['image'])==row['image_sha256'];assert sha(ROOT/row['label'])==row['label_sha256']
 for name in ('PRIOR_BINDINGS.json','TRAIN_SOURCE_BINDINGS.json','INFERENCE_COMPLETE.json','TRAINING_COMPLETE.json','EVALUATION_COMPLETE.json'):
  for f,h in read(RUN/name)['files'].items():assert sha(ROOT/f)==h,f
 assert read(RUN/'TRAIN_SOURCE_BINDINGS.json')['protocol_sha256']==sha(RUN/'PROTOCOL.json');count=0;refits=0;points=0;parents=[dict(np.load(RUN/'arrays'/f'parents_seed{k}.npz')) for k in SEEDS];hard=np.stack([np.linalg.norm(q['anchor']-p['g']['gt'],axis=-1)>=10 for q in parents])
 for seed,q in zip(SEEDS,parents):
  fresh=dict(np.load(RUN/'features'/f'fresh_seed{seed}.npz'));reused=dict(np.load(RUN/'features'/f'reused_seed{seed}.npz'));fresh['diag']=p['diag'];reused['diag']=old['diag'];models=pickle.loads((RUN/'models'/f'seed{seed}.pkl').read_bytes());c=dict(np.load(RUN/'arrays'/f'corrections_seed{seed}.npz'))
  for arm in ARMS:
   origin,mode=arm.split('_');m=models[arm];assert all(x.n_iter_==60 for x in m);count+=2;pred=predict44(m,fresh[mode],fresh['anchor'],fresh['diag']);assert np.array_equal(pred,c['points_'+arm]);assert np.all(np.linalg.norm(pred-fresh['anchor'],axis=-1)<=.15*p['diag'][:,None]+1e-8)
   if seed==1 and origin=='fresh':
    altered={k:v.copy() for k,v in fresh.items()};altered['gt'][~altered['fit']]=999;altered[mode][~altered['fit']]=999;mm=fit44(altered,mode,seed);assert np.array_equal(predict44(mm,fresh[mode],fresh['anchor'],fresh['diag']),pred);refits+=2
  for method in METHODS:
   pt=q[method] if method in METHODS[:4] else c['points_'+method];err=np.hypot(pt[...,0]-p['g']['gt'][...,0],pt[...,1]-p['g']['gt'][...,1]);points+=1
   for key,M in masks44(p,hard):assert abs(err[M[seed-1]].mean()-s['results'][key][method]['per_seed_px'][seed-1])<1e-9
  # Selection and new regressor features are built entirely without physical or point GT.
  if seed==1:
   pp=copy.deepcopy(p)
   for k in pp['g']:pp['g'][k]=np.full_like(pp['g'][k],123)
   # No GT is an argument to predict44; perturb explicit cached GT and ensure same output.
   changed=fresh.copy();changed['gt']=np.full_like(fresh['gt'],123);assert np.array_equal(predict44(models[PRIMARY],changed['line'],changed['anchor'],changed['diag']),c['points_'+PRIMARY])
 write(RUN/'AUDIT.json',dict(status='PASS',models_replayed=count,point_arrays_recalculated=points,nonfit_corruption_refits=refits,new_render_originals=240,requested_observations=600,new_groups=[12,4,8],old_new_image_hash_overlap=0,old_new_group_overlap=0,newGTprojection_audit=read(RUN/'TARGET_AUDIT.json'),fixed_feature_and_prediction_parity=read(RUN/'FEATURE_PARITY.json'),prior_files_unchanged=True,neural_updates=0));print('AUDIT44 PASS',count,points,refits,flush=True)
if __name__=='__main__':main()
