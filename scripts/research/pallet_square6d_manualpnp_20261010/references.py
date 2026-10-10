"""Manual-only square reference; prediction artifacts are never opened here."""
from collections import Counter
import argparse, math
from pathlib import Path
import cv2
import numpy as np
from . import common as C

def reference_fit(manual_xy, K, corner_indices):
 """Only directly observed manual xy, camera K and fixed dimensions enter fit."""
 indices=np.asarray(corner_indices,int);xy=np.asarray(manual_xy,float);camera=np.asarray(K,float)
 assert len(indices)>=4 and len(set(indices))==len(indices) and np.all((indices>=0)&(indices<8))
 assert xy.shape==(len(indices),2) and np.isfinite(xy).all() and camera.shape==(3,3)
 F=C.pose_module();model=F.cuboid(*C.XYZ)
 points=np.full((8,2),np.nan);points[indices]=xy;usable=np.zeros(8,bool);usable[indices]=True
 try:
  result=F.solve(model,points,camera,usable)
 except cv2.error as exc:return dict(available=False,reason='OpenCV:'+str(exc).split('\n')[0])
 if result is None:return dict(available=False,reason='NO_SOLUTION')
 R,t,mean=result
 if not np.isfinite(R).all() or not np.isfinite(t).all():return dict(available=False,reason='NONFINITE_POSE')
 if t[2]<=0:return dict(available=False,reason='TZ_NONPOSITIVE')
 rvec=cv2.Rodrigues(R)[0];projected=cv2.projectPoints(model,rvec,t,camera,None)[0].reshape(8,2)
 errors=np.linalg.norm(projected[indices]-xy,axis=1)
 assert abs(float(errors.mean())-mean)<=1e-7
 return dict(available=True,R=R,t=t,manual_corner_indices=indices,manual_xy=xy,
  projected_corners8=projected,residual_mean_px=float(errors.mean()),residual_max_px=float(errors.max()),
  residual_per_manual_corner_px=errors,tz_positive=True)

def rotation_c4(R,G):
 vals=[]
 for i in range(4):
  a=i*math.pi/2;Q=np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
  vals.append(math.degrees(math.acos(float(np.clip((np.trace((G@Q).T@R)-1)/2,-1,1)))))
 return min(vals)

def private_sheet(records, private):
 private=Path(private).resolve()
 assert not private.is_relative_to(C.ROOT.resolve()),'Raw RGB review must remain outside repository'
 private.mkdir(parents=True,exist_ok=True);names=[]
 for start in range(0,len(records),20):
  rr=records[start:start+20];sheet=np.full((len(rr)*270,640,3),255,np.uint8)
  for j,r in enumerate(rr):
   image=cv2.imread(str(C.verify_binding(r['image'])))
   assert image is not None
   for corner,xy in zip(r['manual_corner_indices'],r['manual_xy']):
    x,y=np.rint(xy).astype(int);cv2.circle(image,(x,y),4,(0,0,255),-1);cv2.putText(image,str(corner),(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
   proj=np.asarray(r['projected_corners8']);edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
   for a,b in edges:cv2.line(image,tuple(np.rint(proj[a]).astype(int)),tuple(np.rint(proj[b]).astype(int)),(0,220,0),1)
   image=cv2.resize(image,(320,240));sheet[j*270:j*270+240,:320]=image
   label=f"{r['id']} manual={len(r['manual_corner_indices'])} mean={r['residual_mean_px']:.3f}px max={r['residual_max_px']:.3f}px"
   cv2.putText(sheet,label,(5,j*270+259),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
  name=f'SQUARE_REFERENCE_REVIEW_{start//20+1:02d}.png';path=private/name
  assert not path.exists();assert cv2.imwrite(str(path),sheet);names.append(dict(file=name,sha256=C.sha(path)))
 return names

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--private-dir',type=Path,required=True);args=parser.parse_args()
 assert C.read(C.DOC/'INPUT_AUDIT.json')['status']=='PASS' and C.read(C.DOC/'TWO_D_PARITY.json')['status']=='PASS'
 assert not (C.DOC/'SQUARE_REFERENCE_POSES.json').exists()
 cv2.setNumThreads(1);inputs=C.read(C.DOC/'INPUT_AUDIT.json')['inputs'];result=[];calls=0
 for r in inputs:
  if not r['eligible']:
   result.append(dict(id=r['id'],eligible=False,available=False,reason='FEWER_THAN4_MANUAL_IN_FRAME',manual_corner_indices=r['manual_in_frame_indices']));continue
  indices=r['manual_in_frame_indices'];manual=np.asarray(r['manual_points8'],float)[indices]
  fit=reference_fit(manual,r['K'],indices);calls+=1
  fit.update(id=r['id'],eligible=True,manual_count=len(indices),K=r['K'],dimensions_pnp_WH_D_m=C.XYZ,
   image=r['image'],annotation=r['annotation'],prediction_inputs=False)
  loo=[]
  if fit['available'] and len(indices)>=5:
   for remove in range(len(indices)):
    sub_indices=indices[:remove]+indices[remove+1:];sub_xy=np.delete(manual,remove,axis=0)
    candidate=reference_fit(sub_xy,r['K'],sub_indices);calls+=1
    if candidate['available']:
     loo.append(dict(removed_corner=indices[remove],available=True,
      rotation_change_c4_deg=rotation_c4(np.asarray(candidate['R']),np.asarray(fit['R'])),
      translation_change_cm=float(np.linalg.norm(np.asarray(candidate['t'])-np.asarray(fit['t']))*100)))
    else:loo.append(dict(removed_corner=indices[remove],available=False,reason=candidate['reason']))
  fit.update(LOO=loo,LOO_applicable=len(indices)>=5,LOO_used_as_filter=False,
   LOO_max_rotation_change_c4_deg=max([x['rotation_change_c4_deg'] for x in loo if x['available']],default=None),
   LOO_max_translation_change_cm=max([x['translation_change_cm'] for x in loo if x['available']],default=None),
   LOO_failed_count=sum(not x['available'] for x in loo))
  result.append(fit)
 eligible=[r for r in result if r['eligible']];available=[r for r in eligible if r['available']]
 residuals=[r['residual_mean_px'] for r in available]
 median=float(np.median(residuals)) if residuals else None
 over=sum(x>5 for x in residuals)/len(eligible)
 passed=bool(residuals and median<=3 and over<=.1)
 status='PASS' if passed else 'STOP_RESIDUAL_GATE'
 C.write('SQUARE_REFERENCE_POSES.json',dict(status=status,reference_type='manual_keypoint_PnP_reference_NOT_independent_physical_measurement',
  total_frames=119,eligible_frames=118,reference_available=len(available),reference_failures=len(eligible)-len(available),
  excluded_ids=['029844'],frames=result,median_mean_residual_px=median,fraction_mean_residual_over5_px=over,
  residual_gate_denominator=118,reference_solver_calls=calls,reference_solver='unchanged SQPnP -> RefineLM',
  C4_proper=True,center8_not_fitted=True,reference_prediction_inputs=False,LOO_diagnostic_only=True,
  reference_source_sha256=C.sha(__file__),input_audit_sha256=C.sha(C.DOC/'INPUT_AUDIT.json'),
  two_D_parity_sha256=C.sha(C.DOC/'TWO_D_PARITY.json')))
 C.write('REFERENCE_GATE.json',dict(status=status,median_mean_residual_px=median,fraction_mean_residual_over5_px=over,
  thresholds=dict(median_max=3,fraction_max=.1),eligible_denominator=118,available=len(available),
  unavailable_ids=[r['id'] for r in eligible if not r['available']],proceed_to_evaluation=passed,source_convention='CF0123',
  generated_reference_solver_calls=calls,model_forwards=0,new_training_updates=0))
 print('SQUARE_REFERENCE_GATE',status,'median_px',median,'fraction_mean_over5',over,'available',len(available),flush=True)
 if not passed:return
 receipt=private_sheet(available,args.private_dir)
 C.write('PRIVATE_REVIEW_RECEIPT.json',dict(status='GENERATED_PRIVATE_ONLY',location_argument='--private-dir',
  private_filenames=receipt,raw_RGB_published=False,blocks_evaluation=False,human_approval_required=False))
if __name__=='__main__':main()
