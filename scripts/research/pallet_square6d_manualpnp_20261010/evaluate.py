"""Existing fixed corners + raw-image subpixel correction, then unchanged square F."""
from collections import Counter
import time
import cv2
import numpy as np
from . import common as C

def inference_coordinates(gray, base_points, n3_points, prediction_support):
 from scripts.research.pallet_training_free_compare_20261007_v1.methods import correct,cap_points
 q0=np.asarray(base_points,float);qN=np.asarray(n3_points,float);support=np.asarray(prediction_support,bool)
 assert np.array_equal(q0[8],qN[8],equal_nan=True)
 h,w=gray.shape
 sub,sub_diag=correct(gray,q0,support,'SUBPIX')
 seq,seq_diag=correct(gray,qN,support,'SUBPIX')
 qSub=cap_points(q0,sub,w,h,support);qSeq=cap_points(q0,seq,w,h,support)
 return [('BASE',q0,q0,None),('N3_DIM_SYM',qN,qN,None),
         ('SUBPIX',qSub,sub,sub_diag),('N3_THEN_SUBPIX',qSeq,seq,seq_diag)]

def main():
 assert C.read(C.DOC/'REFERENCE_GATE.json')['proceed_to_evaluation'] is True
 assert C.read(C.DOC/'TWO_D_PARITY.json')['status']=='PASS'
 assert not (C.DOC/'COORDINATES_SEAL.json').exists() and not (C.DOC/'PREDICTIONS.jsonl.gz').exists()
 started=time.monotonic();cv2.setNumThreads(1)
 input_audit=C.read(C.DOC/'INPUT_AUDIT.json');eligible=[r for r in input_audit['inputs'] if r['eligible']]
 assert len(eligible)==118
 payloads=C.raw_payloads();fixed={b:C.fixed_predictions(p,b) for b,p in payloads.items()}
 saved=[];subpix_calls=0
 for r in eligible:
  image=cv2.imread(str(C.verify_binding(r['image'])));assert image is not None and list(image.shape[:2])==r['raw_hw']
  gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  for b in C.BACKBONES:
   base=fixed[b][(r['id'],'BASE')]
   for s in C.SEEDS:
    n3=fixed[b][(r['id'],f'N3_DIM_SYM_seed{s}')]
    assert np.array_equal(base['support'],n3['support'])
    generated=inference_coordinates(gray,base['points'],n3['points'],base['support'])
    for method,q,native,diag in generated:
     support=np.asarray(base['support']);q0=np.asarray(base['points']);qN=np.asarray(n3['points'])
     assert np.array_equal(q[8],q0[8],equal_nan=True)
     if method in ('SUBPIX','N3_THEN_SUBPIX'):
      usable=support[:8]&np.isfinite(q0[:8]).all(1)&~(q0[:8]==-1).all(1)
      assert np.max(np.linalg.norm(q[:8][usable]-q0[:8][usable],axis=1),initial=0.)<=.01*np.hypot(*r['raw_hw'])+1e-10
      subpix_calls+=diag['algorithm_corner_calls']
     saved.append(dict(id=r['id'],session=r['session'],backbone=b,seed=s,method=method,
      q0=q0,qN=qN,qS=native,qFinal=q,prediction_support=support,detected=base['detected'],
      fixed_metadata=dict(K=r['K'],dimensions_pnp_WH_D_m=C.XYZ,raw_hw=r['raw_hw'],base=base['metadata'],N3=n3['metadata']),
      correction=diag,evaluation_reference_inputs=False))
 C.write_rows('COORDINATES_SEALED.jsonl.gz',saved)
 code=[C.binding(p) for p in PathFiles()]
 C.write('COORDINATES_SEAL.json',dict(status='SEALED_BEFORE_6D_REFERENCE_ACCESS',rows=len(saved),frames=118,
  sha256=C.sha(C.DOC/'COORDINATES_SEALED.jsonl.gz'),method_lock_sha256=C.sha(C.DOC/'METHOD_LOCK.json'),
  source=code,model_forwards=0,GT_coordinates_used_by_correction=False,new_training_updates=0))
 # The saved reference R/t are loaded only after final prediction coordinates are sealed.
 reference=C.read(C.DOC/'SQUARE_REFERENCE_POSES.json');ref={r['id']:r for r in reference['frames'] if r['eligible']}
 F=C.pose_module();counts=Counter();original={name:getattr(cv2,name) for name in ('solvePnP','solvePnPRefineLM')}
 def wrap(name):
  def call(*args,**kwargs):counts[name]+=1;return original[name](*args,**kwargs)
  return call
 output=[]
 try:
  for name in original:setattr(cv2,name,wrap(name))
  for index,r in enumerate(saved):
   q=np.asarray(r['qFinal'],float);K=np.asarray(r['fixed_metadata']['K'],float)
   before=counts.copy();pose=F.infer(q if r['detected'] else None,K,C.XYZ,False)
   g=ref[r['id']]
   truth=None if not g['available'] else dict(R=np.asarray(g['R']),t=np.asarray(g['t']),xyz=C.XYZ,
    body_R=np.asarray(g['R']),body_xyz=C.XYZ,order=4)
   metric=F.metric((r['id'],pose,truth)) if truth is not None else dict(id=r['id'],available=False,reason='REFERENCE_UNAVAILABLE')
   row=dict(r,actual_pose=pose,pose=metric,reference_available=g['available'],manual_count=g['manual_count'],
    reference_residual_mean_px=g.get('residual_mean_px'),F_attempt=True,F_complete=True,
    PnP_counts={name:counts[name]-before[name] for name in original},final_hypothesis=pose.get('selected_hypothesis'),
    inference_reference_access=False,reference_sha256=C.sha(C.DOC/'SQUARE_REFERENCE_POSES.json'))
   output.append(row)
   if (index+1)%300==0:print('SQUARE_F',index+1,len(saved),flush=True)
 finally:
  for name,function in original.items():setattr(cv2,name,function)
 assert len(output)==4248
 C.write_rows('PREDICTIONS.jsonl.gz',output)
 assert all(C.sha(C.resolve(x['path'],x['sha256']))==x['sha256'] for x in input_audit['bound_files'])
 for r in input_audit['inputs']:
  C.verify_binding(r['image']);C.verify_binding(r['annotation'])
 C.write('EXECUTION.json',dict(status='COMPLETE',eligible_frames=118,rows=len(output),backbones=list(C.BACKBONES),
  methods=list(C.METHODS),seeds=list(C.SEEDS),actual_F_calls=len(output),actual_PnP_calls=dict(counts),
  actual_subpix_corner_calls=subpix_calls,elapsed_seconds=time.monotonic()-started,
  PREDICTIONS_sha256=C.sha(C.DOC/'PREDICTIONS.jsonl.gz'),COORDINATES_sha256=C.sha(C.DOC/'COORDINATES_SEALED.jsonl.gz'),
  existing_input_SHA_unchanged=True,new_model_forwards=0,new_training_updates=0,new_synthetic_images=0,
  raw_RGB_published=False,reference_solver_calls=reference['reference_solver_calls']))
 print('SQUARE_EVALUATION_COMPLETE',len(output),dict(counts),flush=True)

def PathFiles():
 from pathlib import Path
 here=Path(__file__).parent
 return [here/'common.py',here/'evaluate.py',C.ROOT/'scripts/research/pallet_training_free_compare_20261007_v1/methods.py']
if __name__=='__main__':main()
