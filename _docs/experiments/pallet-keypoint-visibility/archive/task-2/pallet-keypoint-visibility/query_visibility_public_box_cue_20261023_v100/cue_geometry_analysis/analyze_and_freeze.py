from pathlib import Path
import hashlib,json,itertools,sys
import numpy as np
import trimesh
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_public_box_cue_20261023_v100'
A=V/'cue_geometry_analysis'; O=V/'fixed_public_visual_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
policy={'scope':'Geometry diagnostic followed by fixed public TRAIN RGB cue feasibility only; no new labels or training admission.', 'root_scope':bind(V/'ROOT_SCOPE_BEFORE_PUBLIC_BOX_GEOMETRY.json'),'intake_seal':bind(V/'geometry_intake/INTAKE_SEAL.json'),'source_metadata':bind(P/'query_visibility_public_real_20261022_v63/ycb/instances.jsonl'),'object_ids':[2,3,7,8], 'selection':'Lexicographically first (scene_id,frame_id,instance_id) TRAIN instance per fixed object; record every unavailable VAL object, do not replace it. No image/mask/score selection.', 'corner_order':'AUX_AABB index=4*bitX+2*bitY+bitZ, bit0=min,bit1=max; object-frame official models_info AABB; NOT pallet native8 and NOT snapped to mesh.', 'geometry_diagnostics':'Nearest triangle surface and nearest vertex distances only, no physical corner certification. Fixed vertex-radius diagnostics at1,2,5mm; nearest point not substituted for AABB.', 'crop':'All8 corners each of4 fixed instances; 96 native samples at r+(column/row+.5-48), nearest floor(value+.5), black outside. Raw and marked separate. No crop recenter/snap.', 'forbidden':['M0/M4/depth as visibility certification','PnP','model/fit','current/private'], 'training_admission':False,'human_gold':False,'native_pallet8_mapping':'UNKNOWN'}
write(A/'POLICY_BEFORE_GEOMETRY_AND_RGB.json',policy)
info=json.loads((V/'geometry_intake/models_info.json').read_text())
rows=[json.loads(s) for s in (P/'query_visibility_public_real_20261022_v63/ycb/instances.jsonl').read_text().splitlines()]
out=[];cases=[]
for oid in policy['object_ids']:
 path=V/'geometry_intake'/f'obj_{oid:06}.ply';mesh=trimesh.load(path,process=False)
 assert isinstance(mesh,trimesh.Trimesh)
 verts=np.asarray(mesh.vertices,dtype=np.float64);faces=np.asarray(mesh.faces,dtype=np.int64)
 assert np.isfinite(verts).all() and faces.min()>=0 and faces.max()<len(verts)
 m=info[str(oid)];lo=np.array([m['min_'+a] for a in 'xyz']);hi=lo+np.array([m['size_'+a] for a in 'xyz'])
 bits=np.array(list(itertools.product([0,1],repeat=3)));corners=lo+(hi-lo)*bits
 closest,dist,triangle=trimesh.proximity.closest_point_naive(mesh,corners)
 cr=[]
 for k,c in enumerate(corners):
  vd=np.linalg.norm(verts-c,axis=1); vv=[]
  for radius in [1.,2.,5.]:
   inside=verts[vd<=radius]; vv.append({'radius_mm':radius,'vertex_count':len(inside),'coordinate_range_mm':None if not len(inside) else [inside.min(0).tolist(),inside.max(0).tolist()]})
  cr.append({'aux_aabb_index':k,'bits_xyz':bits[k].tolist(),'original_AABB_xyz_mm':c.tolist(),'nearest_surface_xyz_mm_diagnostic':closest[k].tolist(),'nearest_triangle_index_diagnostic':int(triangle[k]),'nearest_surface_distance_mm':float(dist[k]),'nearest_vertex_distance_mm':float(vd.min()),'fixed_vertex_radius_diagnostics':vv,'physical_corner_semantic_association':'UNKNOWN'})
 actual=np.array([verts.min(0),verts.max(0)]); official=np.array([lo,hi])
 out.append({'object_id':oid,'model':bind(path),'vertices':len(verts),'triangles':len(faces),'units':'mm per BOP model contract','actual_mesh_AABB_mm':actual.tolist(),'official_AABB_mm':official.tolist(),'bounds_max_abs_difference_mm':float(abs(actual-official).max()),'corners':cr,'nearest_surface_distance_mm_range':[float(dist.min()),float(dist.max())]})
 train=sorted([r for r in rows if r['object_id']==oid and r['split']=='train'],key=lambda r:(r['scene_id'],r['frame_id'],r['instance_id']))
 val=[r for r in rows if r['object_id']==oid and r['split']=='val']
 r=train[0]; R=np.asarray(r['cam_R_m2c']).reshape(3,3);t=np.asarray(r['cam_t_m2c']);K=np.asarray(r['K']).reshape(3,3)
 cam=corners@R.T+t; xy=(cam@K.T)[:,:2]/cam[:,2,None];cp=closest@R.T+t;p=(cp@K.T)[:,:2]/cp[:,2,None]
 assert np.all(cam[:,2]>0)
 # Fixed primary index from geometry only; all8 remain in the visual sheet.
 primary=int(np.lexsort((np.arange(8),cam[:,2]))[0])
 case={'case_id':f'OBJ{oid:02d}_LEXFIRST_TRAIN','object_id':oid,'scene_id':r['scene_id'],'frame_id':r['frame_id'],'instance_id':r['instance_id'],'split':'TRAIN','source_instance_count_TRAIN':len(train),'source_instance_count_VAL':len(val),'VAL_status':'MISSING_FIXED_OBJECT' if not val else 'PRESENT_NOT_SELECTED','image':{'path':r['image'],'sha256':r['image_sha256']},'raw_hw':[r['height'],r['width']],'K':r['K'],'cam_R_m2c':r['cam_R_m2c'],'cam_t_m2c_mm':r['cam_t_m2c'],'projection_formula':'camera=R_m2c@model_xyz_mm+t_m2c_mm; image=K@camera/cameraZ; no optimization','geometry_primary_index_min_cameraZ':primary,'all8_preserved':True,'corners':[dict(c,projected_AABB_r_xy=xy[k].tolist(),camera_Z_mm=float(cam[k,2]),nearest_surface_projection_xy_diagnostic=p[k].tolist(),surface_projection_offset_px=float(np.linalg.norm(p[k]-xy[k]))) for k,c in enumerate(cr)]}
 cases.append(case)
write(A/'GEOMETRY_CORNER_DISTANCE_DIAGNOSTICS.json',{'completed':True,'policy':bind(A/'POLICY_BEFORE_GEOMETRY_AND_RGB.json'),'objects':out,'runtime':{'python':sys.version,'numpy':np.__version__,'trimesh':trimesh.__version__},'training_admission':False})
write(O/'FIXED_CASE_SELECTION_BEFORE_RGB.json',{'completed':True,'policy':bind(A/'POLICY_BEFORE_GEOMETRY_AND_RGB.json'),'geometry':bind(A/'GEOMETRY_CORNER_DISTANCE_DIAGNOSTICS.json'),'instances':cases,'all_cases':4,'all_projected_slots':32,'unique_RGBs':len({c['image']['sha256'] for c in cases}),'VAL_fixed_objects_missing':4,'RGB_pixels_read_by_this_script':0})
print(json.dumps({'completed':True,'geometry':bind(A/'GEOMETRY_CORNER_DISTANCE_DIAGNOSTICS.json'),'selection':bind(O/'FIXED_CASE_SELECTION_BEFORE_RGB.json'),'surface_distance_ranges':[(o['object_id'],o['nearest_surface_distance_mm_range'],o['bounds_max_abs_difference_mm']) for o in out]}))
