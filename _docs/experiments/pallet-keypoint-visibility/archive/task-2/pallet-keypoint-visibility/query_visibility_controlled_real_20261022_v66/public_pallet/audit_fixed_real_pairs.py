#!/usr/bin/env python3
"""Supplied-geometry-only source RGBD pairing audit; no PnP/optimization."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,hashlib,json,time
import cv2,numpy as np
from numba import njit
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data';cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def stats(a):
 a=np.asarray(a,float).reshape(-1);a=a[np.isfinite(a)];return dict(N=int(a.size),median=float(np.median(a)),mean=float(a.mean()),p05=float(np.percentile(a,5)),p10=float(np.percentile(a,10)),p90=float(np.percentile(a,90)),p95=float(np.percentile(a,95)),min=float(a.min()),max=float(a.max()))if a.size else dict(N=0)
def mask(path):
 a=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);assert a.shape==(480,640)and set(np.unique(a))<={0,255};return a>0
def erode2(m):return cv2.erode(m.astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
def mesh(path):
 vv=[];ff=[]
 for line in Path(path).open():
  if line.startswith('v '):vv.append([float(x)for x in line.split()[1:4]])
  elif line.startswith('f '):
   face=[int(t.split('/')[0])-1 for t in line.split()[1:]]
   for j in range(1,len(face)-1):ff.append([face[0],face[j],face[j+1]])
 return np.array(vv,float),np.array(ff,np.int64)

@njit(cache=True)
def render_depth(v,f,K,T,h=480,w=640):
 cam=v@T[:3,:3].T+T[:3,3];uv=np.empty((len(v),2));uv[:,0]=K[0,0]*cam[:,0]/cam[:,2]+K[0,2];uv[:,1]=K[1,1]*cam[:,1]/cam[:,2]+K[1,2];z=np.full((h,w),np.inf,np.float32)
 for tri in f:
  i,j,k=tri;z0,z1,z2=cam[i,2],cam[j,2],cam[k,2]
  if min(z0,z1,z2)<=1e-6:continue
  x0,y0=uv[i];x1,y1=uv[j];x2,y2=uv[k];den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
  if abs(den)<1e-9:continue
  xmin=max(0,int(np.ceil(min(x0,x1,x2))));xmax=min(w-1,int(np.floor(max(x0,x1,x2))));ymin=max(0,int(np.ceil(min(y0,y1,y2))));ymax=min(h-1,int(np.floor(max(y0,y1,y2))))
  if xmax<xmin or ymax<ymin:continue
  for yy in range(ymin,ymax+1):
   for xx in range(xmin,xmax+1):
    b0=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den;b1=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den;b2=1-b0-b1
    if b0>=-1e-7 and b1>=-1e-7 and b2>=-1e-7:
     depth=1/(b0/z0+b1/z1+b2/z2)
     if depth<z[yy,xx]:z[yy,xx]=depth
 return z

def overlap(full,teacher):
 inter=int((full&teacher).sum());union=int((full|teacher).sum());return dict(mesh_pixels=int(full.sum()),teacher_pixels=int(teacher.sum()),IoU=inter/union if union else None,teacher_covered_by_mesh=inter/int(teacher.sum())if teacher.any()else None,mesh_unexplained_by_teacher=int((full&~teacher).sum()))
def pose_checks(T):return dict(rotation_orthonormal_max_abs=float(np.max(np.abs(T[:3,:3].T@T[:3,:3]-np.eye(3)))),rotation_det=float(np.linalg.det(T[:3,:3])),bottom_row=T[3].tolist())
def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def visual(pair,arrays,index,canvas):
 a,b,core,stable,front,strictmesh=arrays;overlay=b.astype(float);overlay[stable]=.4*overlay[stable]+.6*np.array([0,255,80]);overlay[front]=.4*overlay[front]+.6*np.array([255,0,180]);overlay[front&strictmesh]=.3*overlay[front&strictmesh]+.7*np.array([255,100,0]);d=ImageDraw.Draw(canvas);y=60+index*270
 d.text((12,y),pair['session']+' '+pair['before_frame']+' -> '+pair['after_frame']+' (supplied pose identical)',font=font(19),fill='black')
 for col,image in enumerate([a,b,overlay.astype(np.uint8)]):canvas.paste(Image.fromarray(image).resize((320,240),Image.Resampling.BILINEAR),(12+col*390,y+24))
 text=f"core {pair['counts']['before_positive_core_bothdepth']}; stable {pair['counts']['stable_visible_20mm']}; front50 {pair['counts']['foreground_50mm']}; front+mesh {pair['counts']['foreground_50mm_strict_mesh']}"
 d.text((12,y+247),text,font=font(16),fill='black')

if __name__=='__main__':
 started=time.monotonic();assert not(ROOT/'FIXED_SIX_PAIR_DIAGNOSTIC.json').exists();protocol=ROOT/'PAIRED_DEPTH_DIAGNOSTIC_PROTOCOL.json';assert protocol.exists()
 lock=json.loads((ROOT/'SPLIT_LOCK_BEFORE_FIT.json').read_text());frames=lock['frames'];groups=defaultdict(list)
 for r in frames:groups[(r['session'],sha(DATA/r['pose']))].append(r)
 pairs=[];unpaired=[]
 for (session,digest),rr in sorted(groups.items()):
  rr=sorted(rr,key=lambda r:int(r['frame_id']))
  if len(rr)==2 and int(rr[0]['frame_id'])%2==0 and int(rr[1]['frame_id'])==int(rr[0]['frame_id'])+1:
   pairs.append(dict(pair_id=session+':'+rr[0]['frame_id']+'_'+rr[1]['frame_id'],session=session,split=rr[0]['split'],before_frame=rr[0]['frame_id'],after_frame=rr[1]['frame_id'],before=rr[0],after=rr[1],supplied_pose_sha256=digest,K_sha256=sha(DATA/rr[0]['K']),pairing_evidence='same-session exactly identical supplied pose; consecutive even/odd ids, paper documents two captures per position',before_after_direction='PROVISIONAL_EVEN_UNOBSTRUCTED_ODD_CARDBOARD; fixed six reviewed visually/depth'))
  else:unpaired.append(dict(session=session,pose_sha256=digest,frames=[r['frame_id']for r in rr]))
 pairs.sort(key=lambda p:(p['session'],int(p['before_frame'])))
 write(ROOT/'EXACT_POSE_PAIR_MANIFEST.json',dict(protocol_sha256=sha(protocol),split_lock_sha256=sha(ROOT/'SPLIT_LOCK_BEFORE_FIT.json'),pairs=pairs,unpaired_groups=unpaired,counts_by_split=dict(Counter(p['split']for p in pairs)),pairing_uses_model_scores_or_private_data=False))
 selected=[next(p for p in pairs if p['session']==s)for s in sorted(set(p['session']for p in pairs))];assert len(selected)==6
 assert all(p['before_frame']=='0000'and p['after_frame']=='0001'for p in selected),'initial sample must remain fixed0000/0001'
 write(ROOT/'FIXED_SIX_PAIR_SELECTION.json',dict(selection='First exact-pose consecutive even/odd pair per session; locked before paired pixel statistics',pair_ids=[p['pair_id']for p in selected],protocol_sha256=sha(protocol)))
 v,f=mesh(DATA/'assets/mesh/pallox.obj');canvas=Image.new('RGB',(1180,60+6*270+40),'white');d=ImageDraw.Draw(canvas);d.text((12,8),'Actual fixed paired RGBD: before / after / depth evidence (no model inference)',font=font(23),fill='black');d.text((12,36),'Green=stable visible; magenta=closer50mm candidate; orange=also before-mesh agreement. Other pixels UNKNOWN.',font=font(16),fill='black');diagnostics=[]
 for i,p in enumerate(selected):
  b,a=p['before'],p['after'];rgbb=np.array(Image.open(DATA/b['image']).convert('RGB'));rgba=np.array(Image.open(DATA/a['image']).convert('RGB'));mb=mask(DATA/b['mask_teacher']);ma=mask(DATA/a['mask_teacher']);Db=cv2.imread(str(DATA/b['depth']),cv2.IMREAD_UNCHANGED);Da=cv2.imread(str(DATA/a['depth']),cv2.IMREAD_UNCHANGED);assert Db.dtype==Da.dtype==np.uint16
  T=np.loadtxt(DATA/b['pose']);Ta=np.loadtxt(DATA/a['pose']);K=np.loadtxt(DATA/b['K']);Ka=np.loadtxt(DATA/a['K']);assert np.array_equal(T,Ta)and np.array_equal(K,Ka)
  depth_direct=render_depth(v,f,K,T);depth_inverse=render_depth(v,f,K,np.linalg.inv(T));full=np.isfinite(depth_direct);inverse=np.isfinite(depth_inverse)
  both=(Db>0)&(Da>0);core=erode2(mb)&both;common=erode2(mb)&erode2(ma)&both;delta=Db.astype(float)-Da.astype(float);residual=Da.astype(float)-Db.astype(float);mesh_resid=depth_direct.astype(float)*1000-Db.astype(float);mesh_agree=full&(Db>0)&(np.abs(mesh_resid)<=np.maximum(20,.01*Db.astype(float)));stable=core&ma&(np.abs(residual)<=20);front=core&(delta>=50)
  # Identity rigidity is checked from supplied matrices, never fitted.
  yy,xx=np.where(common);step=max(1,len(xx)//2000);xx=xx[::step];yy=yy[::step];zz=Db[yy,xx].astype(float)/1000;points=np.stack([(xx-K[0,2])*zz/K[0,0],(yy-K[1,2])*zz/K[1,1],zz],1);relative=Ta@np.linalg.inv(T);warped=points@relative[:3,:3].T+relative[:3,3];uv=warped@Ka.T;uv=uv[:,:2]/uv[:,2,None];error=np.linalg.norm(uv-np.stack([xx,yy],1),axis=1)
  record=dict(pair_id=p['pair_id'],session=p['session'],split=p['split'],before_frame=p['before_frame'],after_frame=p['after_frame'],pose_checks=pose_checks(T),pose_exact_equal=True,K_exact_equal=True,relative_pose_max_abs_from_identity=float(np.max(np.abs(relative-np.eye(4)))),provided_pose_identity_reprojection_error_px=stats(error),direct_pose_projection=overlap(full,mb),inverse_pose_projection=overlap(inverse,mb),mesh_depth_minus_before_mm_on_teacher_core=stats(mesh_resid[erode2(mb)&(Db>0)&full]),after_minus_before_depth_mm_on_common_teacher_core=stats(residual[common]),absolute_depth_residual_mm_on_common_teacher_core=stats(np.abs(residual[common])),mean_RGB_absolute_difference_on_common_teacher_core=stats(np.abs(rgbb.astype(float)-rgba.astype(float)).mean(2)[common]),
              counts=dict(before_mask_positive=int(mb.sum()),after_mask_positive=int(ma.sum()),before_positive_core_bothdepth=int(core.sum()),common_positive_core_bothdepth=int(common.sum()),stable_visible_20mm=int(stable.sum()),strict_mesh_agreement_on_before_core=int((core&mesh_agree).sum()),foreground_50mm=int(front.sum()),foreground_50mm_strict_mesh=int((front&mesh_agree).sum()),foreground_100mm=int((core&(delta>=100)).sum()),foreground_200mm=int((core&(delta>=200)).sum())),
              full_mesh_validated_as_human_pixel_GT=False,before_mask_provenance='MANUAL_PROMPT_SAM_TEACHER',no_mask_zero_negative_label=True,actual_before_after_direction_manual_review='PENDING',unknown_other_pixels=True)
  diagnostics.append(record);visual(record,(rgbb,rgba,core,stable,front,mesh_agree),i,canvas)
  np.savez_compressed(ROOT/f"FIXED_PAIR_{p['session']}.npz",before_positive_core=core,stable_visible_candidate=stable,depth_front_50mm_candidate=front,before_mesh_depth_agreement=mesh_agree,depth_difference_before_minus_after_mm=delta.astype(np.float32),render_depth_before_m=depth_direct,common_teacher_positive_core=common)
  print('FIXED_PAIR',p['session'],record['counts'],'direct/inverse IoU',record['direct_pose_projection']['IoU'],record['inverse_pose_projection']['IoU'],flush=True)
 canvas.save(ROOT/'FIXED_SIX_PAIRED_RGBD_CONTACT_SHEET.png');result=dict(schema='FIXED_PUBLIC_PALLOX_SIX_PAIR_DIAGNOSTIC_V1',protocol_sha256=sha(protocol),source_only=True,current_private_images_labels_or_scores_read=False,GPU_used=False,training=False,model_inference=False,PnP_calls=0,pose_or_alignment_optimization=False,thresholds_are_prespecified_source_diagnostic_only=True,all_catalog_pair_counts=dict(Counter(p['split']for p in pairs)),unpaired_group_count=len(unpaired),mesh=dict(vertices=len(v),triangles=len(f),bounds_min=v.min(0).tolist(),bounds_max=v.max(0).tolist(),extent=np.ptp(v,axis=0).tolist(),physical_shape='Tall wooden pallet bin, not flat pallet'),fixed_pairs=diagnostics,code_sha256=sha(__file__),seconds=time.monotonic()-started);write(ROOT/'FIXED_SIX_PAIR_DIAGNOSTIC.json',result);print('SIX PAIR AUDIT DONE',result['seconds'],flush=True)
