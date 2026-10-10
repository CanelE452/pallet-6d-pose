"""Freeze input bindings and actually reproduce old Square 2D results; no F/model."""
from collections import Counter
import math
import numpy as np
from PIL import Image
from . import common as C

def truth():
 snapshot=C.read(C.resolve(C.SNAPSHOT));groups=C.read(C.resolve(C.SYMMETRY))['objects']
 g=next(x for x in groups if x['object_type']=='plastic_standard_110x110x15')
 assert g['group_order']==4 and all(p[8]==8 for p in g['permutations'])
 from scripts.evaluation.green_saved_labels_v1 import annotation_arrays
 records=snapshot['records'];assert len(records)==119 and len({r['id'] for r in records})==119
 original={p.stem for p in (C.SOURCE/'outputs/annotations/0918_dataset_square').glob('*.json')}
 assert original=={r['id'] for r in records}
 modes={'manual_declared':[],'manual_in_frame':[]};inputs=[];count=Counter();combos=Counter()
 for r in records:
  assert r['canonical_WDH_m']==[1.1,1.1,.15] and r['session']=='0918_dataset'
  ann=C.read(C.verify_binding(r['annotation']));obj=ann['objects'][0]
  assert ann['schema_version']=='real_pallet_gt_v2' and len(ann['objects'])==1
  assert obj['object_type']=='plastic_standard_110x110x15'
  dims=obj['physical_dimensions_m'];assert [dims['x'],dims['z'],dims['y']]==[1.1,1.1,.15]
  im=C.verify_binding(r['image'])
  with Image.open(im) as p: assert list(p.size[::-1])==r['original_hw']==[480,640];p.verify()
  camera=ann['camera_data'];assert [camera['height'],camera['width']]==r['original_hw']
  intr=camera['intrinsics'];K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],float)
  assert np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0
  gt,known=annotation_arrays(ann);_,manual=annotation_arrays(ann,True)
  inside=np.isfinite(gt).all(1)&(gt[:,0]>=0)&(gt[:,0]<640)&(gt[:,1]>=0)&(gt[:,1]<480)
  v=manual&inside;indices=np.flatnonzero(v[:8]).tolist();count[len(indices)]+=1;combos[tuple(indices)]+=1
  assert len(indices)==r['manual_in_frame_corners']
  assert int(manual[:8].sum())==r['manual_corners']
  bb=np.r_[gt[known&inside].min(0),gt[known&inside].max(0)]
  for name,mask in [('manual_declared',manual),('manual_in_frame',v)]:
   modes[name].append(dict(id=r['id'],session=r['session'],hw=r['original_hw'],gt=gt,valid=mask,box=bb,
    permutations=g['permutations'],material='plastic',occlusion='unclassified'))
  inputs.append(dict(id=r['id'],session=r['session'],raw_hw=r['original_hw'],K=K,dimensions_pnp_WH_D_m=C.XYZ,
   manual_declared_indices=np.flatnonzero(manual[:8]).tolist(),manual_in_frame_indices=indices,
   manual_points8=gt[:8],image=r['image'],annotation=r['annotation'],eligible=len(indices)>=4))
 assert count=={3:1,4:30,5:52,6:35,7:1}
 assert sum(len(r['manual_in_frame_indices']) for r in inputs)==600
 assert sum(len(r['manual_declared_indices']) for r in inputs)==602
 assert [r['id'] for r in inputs if not r['eligible']]==['029844']
 faces=[{0,1,2,3},{4,5,6,7},{0,1,4,5},{2,3,6,7},{0,3,4,7},{1,2,5,6}]
 assert all(not any(set(r['manual_in_frame_indices'])<=f for f in faces) for r in inputs if r['eligible'])
 return inputs,modes,dict(count),{'/'.join(map(str,k)):v for k,v in combos.items()}

def main():
 assert not C.DOC.exists(), 'Output namespace already exists; preserve it and STOP'
 C.DOC.mkdir(parents=True)
 C.write('METHOD_LOCK.json',dict(status='LOCKED_BEFORE_REFERENCE_AND_6D',frames=119,primary_eligible_frames=118,
  reference_input_fields=['manual_click corner0..7 in frame','K','registered WH_D'],reference_predicted_inputs=False,
  manual_minimum=4,reference_solver='unchanged SQPnP -> RefineLM',dimensions_pnp_WH_D_m=C.XYZ,
  residual_gate=dict(median_mean_residual_px_max=3.,fraction_mean_over5_px_max=.1),
  LOO='record only for >=5 manual points; C4 minimum rotation and centroid cm; no filtering',
  subpix=dict(winSize=[5,5],zeroZone=[-1,-1],maxCount=40,epsilon=.001),
  final_cap=dict(fraction_raw_diagonal=.01,reference='original Base',applied_once_at_end=True,center8_preserved=True),
  paths=list(C.METHODS),backbones=list(C.BACKBONES),seeds=list(C.SEEDS),
  scopes=['ALL118','reference_residual_mean_le5','manual4','manual_ge5'],
  bootstrap=dict(level='paired frame',resamples=10000,seed=20260917,descriptive_only=True,cluster_bootstrap=False),
  success='translation_cm < 5 and rotation_deg < 5',symmetry='proper C4; center8 fixed',
  fixed_F_sources=[dict(path=k,sha256=v) for k,v in C.F_SOURCE_HASHES.items()],
  new_training_updates=0,new_model_forwards=0,new_synthetic_images=0,raw_RGB_public=False))
 inputs,modes,count,combos=truth();payloads=C.raw_payloads();track={};comparisons=0;maxdiff=0.
 def bind(b):
  p=C.verify_binding(b);track[str(C.binding(p)['path'])]=C.binding(p);return p
 def compare(a,b,label):
  nonlocal comparisons,maxdiff
  if isinstance(b,dict):
   assert set(a)==set(b),(label,'keys')
   for k in b:compare(a[k],b[k],label+'/'+str(k))
  elif isinstance(b,list):
   assert len(a)==len(b),label
   for i,(x,y) in enumerate(zip(a,b)):compare(x,y,label+'/'+str(i))
  elif isinstance(b,(int,float)) and not isinstance(b,bool):
   delta=abs(float(a)-float(b));maxdiff=max(maxdiff,delta);comparisons+=1
   assert delta<=1e-7,(label,delta)
  else: assert a==b,(label,a,b)
 for rel,expected in C.F_SOURCE_HASHES.items():
  assert C.sha(C.SOURCE/rel)==expected and C.sha(C.ROOT/rel)==expected
  track[rel]=C.binding(C.SOURCE/rel)
 from scripts.research.pallet_n3_completion_v3 import square_yolo as Y,evaluation as E
 for backbone,payload in payloads.items():
  rawpath=C.resolve(C.RAW+'/SQUARE_YOLO_PREDICTIONS.json' if backbone=='yolo' else C.RAW+f'/predictions/{backbone}_GREEN0918_119.json')
  track[str(C.binding(rawpath)['path'])]=C.binding(rawpath)
  oldpath=C.resolve(C.OLD_DOC+f'/SQUARE_{backbone.upper()}_RESULTS.json');old=C.read(oldpath);track[str(C.binding(oldpath)['path'])]=C.binding(oldpath)
  assert payload['complete'] and old['complete']
  if backbone=='yolo':
   assert not payload['GT_input'] and not payload['annotation_input']
   for b in payload['bindings'].values():
    if isinstance(b,dict) and 'path'in b:bind(b)
    elif isinstance(b,dict):
     for v in b.values():bind(v)
   actual=Y.evaluate_predictions(payload['predictions'],modes)
   for mode in modes:
    compare(actual['modes'][mode]['summary'],old['modes'][mode]['summary'],backbone+'/'+mode+'/summary')
    compare(actual['modes'][mode]['families'],old['modes'][mode]['families'],backbone+'/'+mode+'/families')
  else:
   assert not payload['GT_inputs'] and not payload['camera_inputs']
   for k in ('population','protocol','selection','training','registry'):bind(payload[k])
   training=C.read(bind(payload['training']))
   for receipt in training['runs']:
    bind(receipt['checkpoint'])
    if isinstance(receipt.get('base_checkpoint'),dict):bind(receipt['base_checkpoint'])
   protocol=C.read(bind(payload['protocol']))
   def checkpoints(x):
    if isinstance(x,dict):
     if 'path'in x and 'sha256'in x and str(x['path']).endswith(('.pt','.pth')):bind(x)
     else:
      for v in x.values():checkpoints(v)
    elif isinstance(x,list):
     for v in x:checkpoints(v)
   checkpoints(protocol)
   for mode in modes:
    actual=E.evaluate_payloads([(None,payload)],modes[mode],backbone=backbone,include_pose=False,expected_frames=None)
    for method in ('base','n3_seed1','n3_seed2','n3_seed3'):
     compare(actual['methods'][method]['result']['corner'],old['modes'][mode]['methods'][method]['result']['corner'],backbone+'/'+mode+'/'+method+'/corner')
     compare(actual['methods'][method]['result']['corner_rows'],old['modes'][mode]['methods'][method]['result']['corner_rows'],backbone+'/'+mode+'/'+method+'/rows')
  fixed=C.fixed_predictions(payload,backbone);assert {k[0] for k in fixed}=={r['id'] for r in inputs}
  for r in inputs:
   base=fixed[(r['id'],'BASE')]
   for seed in C.SEEDS:
    n=fixed[(r['id'],f'N3_DIM_SYM_seed{seed}')]
    assert np.array_equal(base['support'],n['support'])
    assert np.array_equal(base['points'][8],n['points'][8],equal_nan=True)
 C.write('INPUT_AUDIT.json',dict(status='PASS',frames=119,sessions=1,session='0918_dataset',eligible_frames=118,
  manual_in_frame_counts=count,manual_index_combinations=combos,eligible_single_face_only=0,
  inputs=inputs,bound_files=list(track.values()),snapshot=C.binding(C.resolve(C.SNAPSHOT)),symmetry=C.binding(C.resolve(C.SYMMETRY)),
  namespace_initially_absent=True,original_source_modified=False,reference_F_calls=0,model_forwards=0))
 C.write('TWO_D_PARITY.json',dict(status='PASS',frames=119,manual_declared=602,manual_in_frame=600,
  backbones=list(C.BACKBONES),actual_recalculated=True,numeric_comparisons=comparisons,max_absolute_difference=maxdiff,
  tolerance=1e-7,F_calls=0,model_forwards=0,checkpoint_bytes_verified=True,
  preflight_source_sha256=C.sha(__file__)))
 print('SQUARE_INPUT_AND_2D_PARITY_PASS',comparisons,maxdiff,flush=True)
if __name__=='__main__':main()
