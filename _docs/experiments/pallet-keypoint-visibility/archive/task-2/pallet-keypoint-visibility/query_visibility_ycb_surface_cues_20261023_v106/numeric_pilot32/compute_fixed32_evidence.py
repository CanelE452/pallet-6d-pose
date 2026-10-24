"""32 fixed source-surface points: target-alone ray and native pixel evidence, no labels."""
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode=True
import cv2
import numpy as np
import trimesh
H=Path(__file__).resolve().parent;V=H.parent
EPS=np.finfo(np.float64).eps
TOLS=(0.,64*EPS,1e-10)
POLICY=dict(fixed_points='Exactly the32 existing V106 candidate vertex IDs and lexfirst4 pilot poses; no reselection/relocation.',
 ray='Double-sided Moller-Trumbore, original triangle order; camera origin0, normalized camera ray. Reject abs(det)<=64*FP64eps*||edge1||*||edge2||; distance>0.',
 barycentric_acceptance='Report three fixed sensitivity settings0,64*FP64eps(primary arithmetic tolerance),1e-10; no semantic visibility threshold or label chosen.',
 tie='Distance tie tolerance=256*FP64eps*max(1,target_ray_range_mm,max_vertex_camera_range_mm). Preserve hit triangle IDs, expected-vertex incident membership, exact minimum ties and bounded ties.',
 pixels='3x3 offsets[-1,0,1] around floor(original projected r+.5), retain every OOB/zero-depth/overlap/unassigned sample. Source raw depth*depth_scale ->mm.',
 owner='Complete scene_gt list, all corresponding mask_visib PNGs; own mask and own mask_visib separate. >0 source raster membership, no M0 gate.',
 differences='Raw observed depth(mm)-fixed surface cameraZ and -target-alone first-hit cameraZ at that integer pixel ray; camera range distances separately preserved.',
 interpretation='Numeric evidence only, no D/E/S training label or source admission. No-hit is not declared a hole; non-watertight topology and ray sensitivity stay explicit.',
 forbidden=['RGB','model','fit','private/current','PnP','reference snap or crop re-selection'])
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def ray_hits(vertices,faces,direction,target_range,incident):
 d=np.asarray(direction,dtype=np.float64);d=d/np.linalg.norm(d)
 a=vertices[faces[:,0]];e1=vertices[faces[:,1]]-a;e2=vertices[faces[:,2]]-a
 h=np.cross(np.broadcast_to(d,e2.shape),e2);det=(e1*h).sum(1)
 detbound=64*EPS*np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1);nonparallel=abs(det)>detbound
 inv=np.zeros_like(det);inv[nonparallel]=1/det[nonparallel]
 s=-a;u=inv*(s*h).sum(1);q=np.cross(s,e1);v=inv*(q*d).sum(1);t=inv*(e2*q).sum(1)
 tie=256*EPS*max(1.,float(target_range),float(np.linalg.norm(vertices,axis=1).max()))
 results=[]
 for tol in TOLS:
  valid=nonparallel&(u>=-tol)&(v>=-tol)&(u+v<=1+tol)&(t>0)
  ix=np.flatnonzero(valid);ordered=ix[np.lexsort((ix,t[ix]))]
  if not len(ordered):results.append(dict(barycentric_tolerance=tol,hit_count=0,first_hit_range_mm=None,first_hit_camera_Z_mm=None,no_hit=True));continue
  first=int(ordered[0]);ties=ordered[abs(t[ordered]-t[first])<=tie];p=d*t[first]
  results.append(dict(barycentric_tolerance=tol,hit_count=len(ordered),no_hit=False,
   first_hit_range_mm=float(t[first]),first_hit_camera_Z_mm=float(p[2]),first_hit_xyz_camera_mm=p.tolist(),
   target_range_minus_first_hit_range_mm=float(target_range-t[first]),
   first_triangle_index=first,first_triangle_barycentric_uv=[float(u[first]),float(v[first])],
   exact_minimum_triangle_tie_count=int((t[ordered]==t[first]).sum()),bounded_first_hit_triangle_indices=ties.tolist(),
   numerical_tie_bound_mm=tie,first_hit_is_incident_to_expected_vertex=bool(any(int(x)in incident for x in ties)),
   incident_expected_triangle_hit_indices=[int(x)for x in ordered if int(x)in incident],
   first5_intersections=[dict(triangle_index=int(x),ray_range_mm=float(t[x]),cameraZ_mm=float(d[2]*t[x]))for x in ordered[:5]]))
 return dict(ray_direction_camera_unit=d.tolist(),target_ray_range_mm=float(target_range),determinant_rejected_triangle_count=int((~nonparallel).sum()),sensitivity=results)
def main():
 assert not(H/'FIXED32_NUMERIC_EVIDENCE.json').exists()
 write(H/'NUMERIC_POLICY_BEFORE_DEPTH_MASK_PIXELS.json',dict(completed=True,policy=POLICY,geometry=bind(V/'SURFACE_CANDIDATE_RESULT.json'),visual_observations=bind(V/'fixed4_actual_rgb_review/ACTUAL_FIXED4_VISUAL_OBSERVATIONS.json')))
 geometry=read(V/'SURFACE_CANDIDATE_RESULT.json');prior={x['path']:x for x in read(geometry['prior_depth_owner_byte_bindings']['path'])['files']}
 frames={};out=[];all_bindings={}
 for obj in geometry['objects']:
  pilot=obj['fixed_pilot'];scene=Path(pilot['depth']).parent.parent;key=(str(scene),pilot['frame_id'])
  if key not in frames:
   gt=read(scene/'scene_gt.json')[str(pilot['frame_id'])];camera=read(scene/'scene_camera.json')[str(pilot['frame_id'])]
   depth=cv2.imread(pilot['depth'],cv2.IMREAD_UNCHANGED);assert depth is not None and depth.dtype==np.uint16 and depth.shape==tuple(pilot['raw_hw'])
   paths=[scene/'mask_visib'/f'{pilot["frame_id"]:06d}_{i:06d}.png'for i in range(len(gt))]
   owners=[]
   for path in paths:
    b=bind(path);assert b['sha256']==prior[str(path)]['sha256'];all_bindings[str(path)]=b
    mask=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);assert mask is not None and mask.ndim==2 and mask.shape==depth.shape and set(np.unique(mask)).issubset({0,255});owners.append(mask>0)
   db=bind(Path(pilot['depth']));assert db['sha256']==prior[pilot['depth']]['sha256'];all_bindings[pilot['depth']]=db
   for filename in('scene_gt.json','scene_camera.json'):all_bindings[str(scene/filename)]=bind(scene/filename)
   frames[key]=dict(gt=gt,camera=camera,depth=depth,owners=np.stack(owners),owner_paths=[str(p)for p in paths])
  frame=frames[key];gt=frame['gt'];own=pilot['instance_id'];assert gt[own]['obj_id']==obj['object_id']
  assert gt[own]['cam_R_m2c']==pilot['cam_R_m2c']and gt[own]['cam_t_m2c']==pilot['cam_t_m2c_mm']
  assert frame['camera']['cam_K']==pilot['K']and frame['camera']['depth_scale']==pilot['depth_scale_to_mm']
  full=cv2.imread(pilot['mask_full'],cv2.IMREAD_UNCHANGED);assert full is not None and full.shape==frame['depth'].shape and set(np.unique(full)).issubset({0,255});full=full>0;all_bindings[pilot['mask_full']]=bind(Path(pilot['mask_full']))
  mesh=trimesh.load(obj['model']['path'],process=False,skip_materials=True);assert bind(Path(obj['model']['path']))==obj['model']
  xyz=np.asarray(mesh.vertices,dtype=np.float64);faces=np.asarray(mesh.faces,dtype=np.int64);R=np.asarray(pilot['cam_R_m2c']).reshape(3,3);t=np.asarray(pilot['cam_t_m2c_mm']);K=np.asarray(pilot['K']).reshape(3,3);invK=np.linalg.inv(K);vertices=xyz@R.T+t
  boundary=int((np.bincount(mesh.edges_unique_inverse)==1).sum());point_records=[]
  for c,projection in zip(obj['candidates'],pilot['projected_surface_candidates']):
   assert c['aux_surface_id']==projection['aux_surface_id']and np.array_equal(xyz[c['original_mesh_vertex_index']],c['xyz_mm'])
   target=R@np.array(c['xyz_mm'])+t;rr=np.linalg.norm(target);ray=ray_hits(vertices,faces,target,rr,set(c['incident_triangle_indices']))
   uv=K@target;uv=uv[:2]/uv[2];assert np.max(abs(uv-projection['projected_xy']))<1e-10
   r=np.asarray(projection['projected_xy']);centre=np.floor(r+.5).astype(int);samples=[]
   for dy in(-1,0,1):
    for dx in(-1,0,1):
     x,y=(centre+np.array([dx,dy])).tolist();valid=0<=x<pilot['raw_hw'][1]and 0<=y<pilot['raw_hw'][0]
     cell=dict(offset_xy=[dx,dy],pixel_xy=[x,y],in_bounds=valid)
     if not valid:cell.update(depth_raw_uint16=None,depth_mm=None,own_full=None,own_visible=None,other_owner_count=None,all_visible_owner_indices=None);samples.append(cell);continue
     raw=int(frame['depth'][y,x]);z=raw*pilot['depth_scale_to_mm'];ids=np.flatnonzero(frame['owners'][:,y,x]).tolist();other=[i for i in ids if i!=own]
     pixel_ray=ray_hits(vertices,faces,invK@np.array([x,y,1.]),rr,set(c['incident_triangle_indices']));first=pixel_ray['sensitivity'][1]['first_hit_camera_Z_mm']
     cell.update(depth_raw_uint16=raw,depth_valid=raw>0,depth_mm=z,own_full=bool(full[y,x]),own_visible=bool(frame['owners'][own,y,x]),
      other_owner_count=len(other),all_visible_owner_count=len(ids),all_visible_owner_indices=ids,all_visible_owner_object_ids=[gt[i]['obj_id']for i in ids],
      own_other_overlap=own in ids and bool(other),unassigned_visible_owner=len(ids)==0,
      observed_depth_minus_fixed_surface_Z_mm=None if raw==0 else float(z-target[2]),
      observed_depth_minus_pixel_target_first_hit_Z_mm=None if raw==0 or first is None else float(z-first),target_alone_integer_pixel_ray=pixel_ray)
     samples.append(cell)
   valid=[s for s in samples if s['in_bounds']];valid_depth=[s for s in valid if s['depth_valid']]
   point_records.append(dict(aux_surface_id=c['aux_surface_id'],original_mesh_vertex_index=c['original_mesh_vertex_index'],fixed_xyz_model_mm=c['xyz_mm'],fixed_projected_r_xy=projection['projected_xy'],camera_Z_mm=float(target[2]),camera_range_mm=float(rr),nearest_pixel_xy=centre.tolist(),
    continuous_point_in_image=bool(0<=r[0]<pilot['raw_hw'][1]and 0<=r[1]<pilot['raw_hw'][0]),target_alone_fixed_point_ray=ray,all9_native_samples=samples,
    counts=dict(samples=9,in_bounds=len(valid),OOB=9-len(valid),valid_depth=len(valid_depth),zero_depth=len(valid)-len(valid_depth),own_full=sum(s['own_full']for s in valid),own_visible=sum(s['own_visible']for s in valid),
      unique_other_owner_only=sum(s['other_owner_count']==1 and not s['own_visible']for s in valid),multiple_other_owners=sum(s['other_owner_count']>1 for s in valid),own_other_overlap=sum(s['own_other_overlap']for s in valid),unassigned=sum(s['unassigned_visible_owner']for s in valid)),
    raw_nonzero_depth_minus_fixed_surface_Z_mm=[s['observed_depth_minus_fixed_surface_Z_mm']for s in valid_depth],D_E_S_training_label=None))
  out.append(dict(object_id=obj['object_id'],pilot_pose=dict(scene_id=pilot['scene_id'],frame_id=pilot['frame_id'],instance_id=own,raw_hw=pilot['raw_hw'],K=pilot['K'],depth_scale_to_mm=pilot['depth_scale_to_mm']),
    complete_owner_annotation_instances=[dict(instance_index=i,object_id=g['obj_id'],visible_mask=frame['owner_paths'][i])for i,g in enumerate(gt)],expected_scene_GT_instance_count=len(gt),loaded_visible_owner_mask_count=len(frame['owners']),
    mesh=dict(model=obj['model'],watertight=bool(mesh.is_watertight),winding_consistent=bool(mesh.is_winding_consistent),boundary_unique_edges=boundary,non_watertight_does_not_localize_a_hole=True),all8=point_records))
 result=dict(completed=True,policy=bind(H/'NUMERIC_POLICY_BEFORE_DEPTH_MASK_PIXELS.json'),code=bind(Path(__file__)),geometry=bind(V/'SURFACE_CANDIDATE_RESULT.json'),
  official_docs=dict(URL='https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md',consulted='2026-10-23',support='Lines247-259:16-bit depth, silhouette/visible silhouette masks and depth_scale to mm; lines269-274:model-to-camera pose/projection; lines279-285:render/depth-derived GT-info and tolerance caveat.'),
  source_bindings=list(all_bindings.values()),all4_instances=out,all32_fixed_points=True,all288_projected_neighbour_cells=True,all_source_points_unchanged=True,
  interpretation='Raw target-alone geometry versus independently observed source scene evidence; no D/E/S labels or admission. Never equate amodal-only pixels, nohit, positive depth gap or missing junction with a semantic external label.',
  RGB_pixels_read=0,model_fit_private_current_PnP=0,training_admission=False,human_gold=False,native_mapping='UNKNOWN')
 write(H/'FIXED32_NUMERIC_EVIDENCE.json',result)
 write(H/'COMPLETE.json',dict(completed=True,terminal_exit_expected=0,output=bind(H/'FIXED32_NUMERIC_EVIDENCE.json'),code=bind(Path(__file__)),policy=bind(H/'NUMERIC_POLICY_BEFORE_DEPTH_MASK_PIXELS.json'),points=32,neighbour_cells=288,unique_depth_frames=len(frames),labels_admitted=False))
 print(json.dumps(dict(completed=True,evidence=bind(H/'FIXED32_NUMERIC_EVIDENCE.json'),complete=bind(H/'COMPLETE.json'))))
if __name__=='__main__':main()
