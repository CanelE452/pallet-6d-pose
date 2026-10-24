"""Independent saved GT-only MR6D reference arithmetic; no producer imports.

Cuboid face scores use analytic axis normals. Rear indices use exact sign
flips, not nearest-prediction matching. No query, model, masks or RGB reads.
"""
from pathlib import Path
import ast,hashlib,json,math
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
O=V/'native_reference_audit/provided_reference_v1'
SRC=P/'query_visibility_mr6d_public_native_intake_20261024_v151';D=SRC/'provided_metadata'
SEAL=O/'REFERENCE_PROJECTION_COMPLETE.json';SEAL_SHA='7d2bf532e30d2c2680e36d3c63058e94c3b1ba8b52176713fdedf66af7a7e3f1'
READS={}
def bind(p):
 p=Path(p).resolve();assert '_v112'not in str(p).lower()and '_v149'not in str(p).lower()
 assert p.suffix.lower()not in('.png','.jpg','.jpeg','.pt','.pth','.npz','.npy')
 b=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest());READS[str(p)]=b;return b
def read(p):bind(p);return json.loads(Path(p).read_text())
def checked(b):assert bind(b['path'])==b,b;return Path(b['path'])
def lines(p):bind(p);return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,a):
 assert not p.exists(),p;p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def close(a,b,tol=1e-9):
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape and np.allclose(a,b,atol=tol,rtol=0),(a,b)
 return float(np.max(np.abs(a-b)))if a.size else 0.

def main():
 assert bind(SEAL)['sha256']==SEAL_SHA
 seal=read(SEAL);assert seal['completed']and seal['PASS']and seal['query_or_model_outputs_read']is False
 selection=read(checked(seal['selection']));assert len(selection['selected_frames'])==10
 rr=lines(checked(seal['references']));cc=lines(checked(seal['centroids']));saved=read(checked(seal['frames_and_geometry']))['frames']
 assert len(rr)==104 and len(cc)==13 and len(saved)==10
 original=checked(seal['exact_original_V74_AST']['original_code']);tree=ast.parse(original.read_text());constants={}
 for n in tree.body:
  if isinstance(n,ast.Assign):
   for target in n.targets:
    if isinstance(target,ast.Name)and target.id in('FIXED_ZUP_SIGNS','VERTICAL_FIXED_FACES','NATIVE_NAMES'):
     constants[target.id]=ast.literal_eval(n.value.args[0]if isinstance(n.value,ast.Call)else n.value)
 signs=np.array([[-1,-1,1],[1,-1,1],[1,-1,-1],[-1,-1,-1],[-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]],np.int8)
 faces=np.array([[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2]])
 assert constants['FIXED_ZUP_SIGNS']==signs.tolist()and constants['VERTICAL_FIXED_FACES']==faces.tolist()
 names=constants['NATIVE_NAMES'];mi=read(D/'models/models_info.json')['1']
 mn=np.array([mi['min_x'],mi['min_y'],mi['min_z']],np.float64);extent=np.array([mi['size_x'],mi['size_y'],mi['size_z']],np.float64)
 center=mn+extent/2;fixed=center+signs*extent/2;close(extent,[800,1200,144],0);close(center,[0,0,0],0)
 # Actual ASCII mesh body bounds, independent of previous producer helpers.
 mesh=SRC/'factual_sample_v1/original/models/obj_000001.ply';text=mesh.read_text();bind(mesh)
 head,body=text.split('end_header',1);assert 'format ascii' in head
 nv=int(next(l.split()[2]for l in head.splitlines()if l.startswith('element vertex ')))
 vertices=np.array([[float(v)for v in l.split()[:3]]for l in body.strip().splitlines()[:nv]])
 close(vertices.min(0),mn,1e-6);close(vertices.max(0)-vertices.min(0),extent,1e-6)
 maxerr=dict(model=0.,camera=0.,pixel=0.,camera_inverse=0.,face_score=0.,centroid_pixel=0.)
 records=[];ri=ci=0;inframe_count=positive_count=proxy_count=0;margins=[];LR=[]
 for fi,(s,f)in enumerate(zip(selection['selected_frames'],saved)):
  source_id=f"{s['split']}/{s['scene']}:{s['frame_key']}";ident=f"V152_MR6D_{s['split']}_{s['scene']}_{int(s['frame_key']):06d}"
  assert f['frame_index']==fi and f['id']==ident and f['source_id']==source_id and f['official_split']==s['split']
  gt=read(D/s['split']/s['scene']/'scene_gt.json')[s['frame_key']];cam=read(D/s['split']/s['scene']/'scene_camera.json')[s['frame_key']]
  validation=read(SRC/'factual_sample_v1/SOURCE_GEOMETRY_AND_INSTANCE_MASK_VALIDATION.json')['frames'][fi];HH,W=validation['raw_RGB_HW']
  assert validation['scene']==s['scene']and validation['frame_key']==s['frame_key']
  K=np.array(cam['cam_K'],np.float64).reshape(3,3);assert np.isfinite(K).all()
  assert [a['gt_instance_index']for a in f['instances']]==s['obj1_gt_instance_indices']
  for i,inst in zip(s['obj1_gt_instance_indices'],f['instances']):
   g=gt[i];assert g['obj_id']==1
   R=np.array(g['cam_R_m2c'],np.float64).reshape(3,3);t=np.array(g['cam_t_m2c'],np.float64)
   assert np.isfinite(R).all()and np.isfinite(t).all()
   xyz=(R@fixed.T).T+t;image=(K@xyz.T).T;xy=image[:,:2]/image[:,2,None]
   # Exact inverse of the supplied finite R, without orthogonalizing/fitting it.
   camera=-np.linalg.inv(R)@t;ray=camera-center
   scores=np.array([-ray[1],ray[1],-ray[0],ray[0]])/np.linalg.norm(ray)
   front=int(np.argmax(scores));top=[j for j in faces[front]if signs[j,2]==1];top.sort(key=lambda j:xy[j,0])
   index={tuple(v):j for j,v in enumerate(signs.tolist())}
   bottom=lambda j:index[tuple(signs[j]*[1,1,-1])]
   front4=[top[0],top[1],bottom(top[1]),bottom(top[0])]
   depthaxis=1 if front<2 else 0
   rear4=[]
   for j in front4:
    sign=signs[j].copy();sign[depthaxis]*=-1;rear4.append(index[tuple(sign)])
   perm=np.array(front4+rear4);assert sorted(perm.tolist())==list(range(8))
   # A cuboid vertex is proxy-visible if any adjacent axis-aligned face faces the camera.
   fixed_proxy=np.any(signs*(camera[None,:]-fixed)>0,axis=1);proxy=fixed_proxy[perm]
   dp=xy[perm];dc=xyz[perm];inside=(dc[:,2]>0)&(dp[:,0]>=0)&(dp[:,0]<W)&(dp[:,1]>=0)&(dp[:,1]<HH)
   margin=float(np.sort(scores)[-1]-np.sort(scores)[-2]);lr=float(dp[1,0]-dp[0,0]);margins.append(margin);LR.append(lr)
   maxerr['camera_inverse']=max(maxerr['camera_inverse'],close(camera,inst['camera_in_object_coordinates_exact_linear_inverse']))
   maxerr['face_score']=max(maxerr['face_score'],close(scores,inst['normal_face_scores'],1e-12))
   close(R,inst['provided_R_m2c'],0);close(t,inst['provided_t_m2c_mm'],0);close(K,inst['provided_K'],0)
   assert inst['dynamic_to_fixed_permutation']==perm.tolist()and inst['chosen_face_index']==front
   assert inst['chosen_face_fixed_indices']==faces[front].tolist()
   close(dp,inst['dynamic_projected8']);assert inst['inframe8']==inside.tolist()and inst['convex_self_visible_proxy8_only']==proxy.tolist()
   close(margin,inst['front_runner_margin'],1e-12);close(lr,inst['front_top_imageX_separation_px'])
   assert inst['semantic_states_initial']==['U_UNREVIEWED']*8
   centroid_cam=R@center+t;centroid_h=K@centroid_cam;centroid_xy=centroid_h[:2]/centroid_h[2]
   cr=cc[ci];ci+=1
   assert (cr['frame_index'],cr['id'],cr['source_id'],cr['gt_instance_index'])==(fi,ident,source_id,i)
   close(cr['centroid_model_XYZ_mm'],center,0);maxerr['centroid_pixel']=max(maxerr['centroid_pixel'],close(centroid_xy,cr['centroid_reference_xy']))
   close(cr['camera_Z_mm'],centroid_cam[2]);assert cr['separately_projected_3D_centroid_not_mean_2D_corners']
   for k in range(8):
    r=rr[ri];ri+=1
    assert (r['frame_index'],r['id'],r['source_id'],r['gt_instance_index'],r['native_slot'])==(fi,ident,source_id,i,k)
    assert r['obj_id']==1 and r['source']=='MR6D'and r['split']=='SOURCE_DEVELOPMENT'and r['official_split']==s['split']and r['scene']==s['scene']and r['frame_key']==s['frame_key']
    assert r['native_name']==names[k]and r['fixed_canonical_index']==int(perm[k])and r['dynamic_to_fixed_permutation']==perm.tolist()
    maxerr['model']=max(maxerr['model'],close(fixed[perm[k]],r['reference_model_XYZ_mm'],0));maxerr['camera']=max(maxerr['camera'],close(dc[k],r['reference_camera_XYZ_mm']))
    maxerr['pixel']=max(maxerr['pixel'],close(dp[k],r['reference_xy']))
    assert r['continuous_inframe']==bool(inside[k])and r['positive_camera_Z']==bool(dc[k,2]>0)and r['convex_self_visible_proxy_only']==bool(proxy[k])
    assert r['semantic_state']=='U_UNREVIEWED'and r['human_gold']is False and r['train_ready']is False
    close(scores,r['front_face_scores'],1e-12);close(margin,r['front_runner_margin'],1e-12);close(lr,r['front_top_imageX_separation_px'])
   inframe_count+=int(inside.sum());positive_count+=int((dc[:,2]>0).sum());proxy_count+=int(proxy.sum())
   records.append(dict(frame_index=fi,id=ident,source_id=source_id,gt_instance_index=i,permutation_dynamic_slot_to_fixed=perm.tolist(),
    chosen_vertical_face=front,front_runner_margin=margin,front_top_LR_px=lr,nominal_inframe=int(inside.sum()),nominal_OOF=int((~inside).sum()),
    cuboid_self_visible_proxy=int(proxy.sum()),camera_exact_inverse_mm=camera.tolist(),provided_Rtranspose_inverse_difference_mm=float(np.linalg.norm(camera-(-R.T@t))),
    projected_centroid_minus_mean_projected_corners_px=float(np.linalg.norm(centroid_xy-xy.mean(0)))))
 assert ri==104 and ci==13 and len(records)==13
 assert seal['counts']==dict(frames=10,instances=13,native_references=104,centroids=13,nominal_inframe=inframe_count,nominal_OOF=104-inframe_count)
 close(min(margins),seal['minimum_front_runner_margin'],1e-12);close(min(LR),seal['minimum_front_top_LR_px'])
 result=dict(completed=True,PASS=True,reference_seal=bind(SEAL),all_source_instance_IDs_original_indices_and_all104_roles_verified=True,
  counts=dict(frames=10,instances=13,native_references=104,centroids=13,inframe=inframe_count,OOF=104-inframe_count,positive_camera_Z=positive_count,
   cuboid_self_visible_proxy=proxy_count,semantic_U_UNREVIEWED=104),maximum_absolute_recalculation_errors=maxerr,
  minimum_front_runner_margin=min(margins),minimum_front_top_LR_px=min(LR),instance_results=records,
  permutation_direction='dynamic[k] = canonical[perm[k]]; rear slots are exact depth-axis sign flips of the ordered FRONT slots.',
  raw_model_bounds_mm=[vertices.min(0).tolist(),vertices.max(0).tolist()],source_spec_Z_up_not_image_up=True,
  visual_review_performed=False,limits=['The computations certify saved nominal reference arithmetic, not actual <=3px physical corner accuracy.',
   'Camera inverse uses the original non-orthogonalized supplied R; R-transpose difference is recorded without fitting/repair.',
   'Positive front/LR margins here avoid actual ties; historical renderer writer tie policy remains uncertified.',
   'Cuboid self-visible proxy is separate from mesh/semantic D/E/S. All104 semantic states remain U_UNREVIEWED.',
   'Source nominal dimensions and pose have low-centimeter uncertainty; multiple source instance masks are not exclusive ownership gold.',
   'This reference audit reads no original initializer q, model outputs, native semantic proposals or private data.'],
  calls=dict(producer_helper_import_or_exec=0,query_or_model_output_reads=0,learned_model_fit_forward=0,private_or_forbidden_phase_access=0,
   RGB_depth_mask_body_reads=0,pose_orientation_fitting=0,native_semantic_labels_created=0))
 write(H/'NUMERIC_RECALCULATION.json',result)
 for b in list(READS.values()):assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()==b['sha256']
 write(H/'REVIEW.json',dict(completed=True,PASS=True,code=bind(__file__),reference_seal=bind(SEAL),result=bind(H/'NUMERIC_RECALCULATION.json'),
  original_provided_T_K_CAD_reference_files_unchanged=True,native_labels_certified=False,calls=result['calls']))
 print(json.dumps(dict(completed=True,PASS=True,receipt=bind(H/'REVIEW.json'),counts=result['counts'],maxerrors=maxerr,min_margin=min(margins),min_LR=min(LR)),indent=2))

if __name__=='__main__':main()
