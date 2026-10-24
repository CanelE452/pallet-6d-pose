"""V93 asset-local geometry inventory only. No render, semantic labels, native refs, learned models."""
from pathlib import Path
from collections import Counter
import hashlib,json,datetime,sys
import numpy as np
from pxr import Usd,UsdGeom,UsdShade,Gf,Sdf
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
SCOPE=ROOT/'SCOPE_BEFORE_ACTUAL_MESH_GEOMETRY.json'
BUDGET=32*1024*1024

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):p=Path(p);return {'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size}
def write(p,j):
 with Path(p).open('x')as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def finite_list(a):return np.asarray(a).tolist()if np.isfinite(a).all()else None

def generated():
 A=np.array([[2.,0,0,0],[0,-3.,0,0],[0,0,4.,0],[5.,7.,11.,1.]])
 pts=np.array([[1.,2.,3.],[-4.,0.,2.]])
 got=np.c_[pts,np.ones(len(pts))]@A
 expected=np.array([[7.,1.,23.],[-3.,7.,19.]])
 assert np.array_equal(got[:,:3],expected)and np.all(got[:,3]==1)and np.linalg.det(A[:3,:3])<0
 gf=Gf.Matrix4d(tuple(tuple(float(v)for v in row)for row in A))
 for i,p in enumerate(pts):assert np.array_equal(np.array(gf.Transform(Gf.Vec3d(*p))),expected[i])
 n=np.array([[0.,1.,0.]])@np.linalg.inv(A[:3,:3]).T
 assert np.allclose(n,[[0.,-1/3,0.]])
 write(HERE/'GENERATED_TRANSFORM_AND_INDEX_TESTS.json',{'completed':True,'PASS':True,'code':bind(__file__),'Gf_rowvector_translation_nonuniform_reflection_exact':True,'normal_inverse_transpose_checked':True,'source_geometry_read':0,'render_labels_model_current':0})

def main():
 scope=json.loads(SCOPE.read_text());assert sha(SCOPE)=='c6d63fe1f5ba19b62998522d310fb964f8d4f6bd6a2a6d2bf4c3f28109901552'
 src=Path(scope['source_asset']['path']);assert sha(src)==scope['source_asset']['sha256']
 generated()
 stage=Usd.Stage.Open(str(src),load=Usd.Stage.LoadNone);assert stage
 time=Usd.TimeCode.Default();mpu=float(UsdGeom.GetStageMetersPerUnit(stage));assert np.isfinite(mpu)and mpu>0
 up=str(UsdGeom.GetStageUpAxis(stage));cache=UsdGeom.XformCache(time)
 prims=sorted([p for p in stage.Traverse()if p.IsA(UsdGeom.Mesh)],key=lambda p:str(p.GetPath()))
 assert len(prims)==308
 mats=sorted(str(p.GetPath())for p in stage.Traverse()if p.IsA(UsdShade.Material));matidx={p:i for i,p in enumerate(mats)}
 points_local=[];points_world=[];normals_local=[];normals_world=[];triangles=[];tri_local=[];tri_part=[];tri_material=[];tri_source_face=[];tri_valid=[];area2=[]
 voff=[0];foff=[0];noff=[0];xf=[];meta=[];problems=Counter();transform_err=0.
 missing=[];orientation=[];mesh_geometry_valid=[]
 for part_id,prim in enumerate(prims):
  mesh=UsdGeom.Mesh(prim);path=str(prim.GetPath());point_value=mesh.GetPointsAttr().Get(time);face_value=mesh.GetFaceVertexCountsAttr().Get(time);index_value=mesh.GetFaceVertexIndicesAttr().Get(time)
  if point_value is None or face_value is None or index_value is None:
   missing.append({'part_id':part_id,'prim_path':path,'status':'U_MISSING_GEOMETRY'});p=np.empty((0,3),np.float64);cs=np.empty(0,np.int64);inds=np.empty(0,np.int64)
  else:p=np.asarray(point_value,dtype=np.float64).reshape(-1,3);cs=np.asarray(face_value,dtype=np.int64);inds=np.asarray(index_value,dtype=np.int64)
  M=np.asarray(cache.GetLocalToWorldTransform(prim),dtype=np.float64).reshape(4,4);xf.append(M)
  assert np.isfinite(M).all();linear=M[:3,:3];det=float(np.linalg.det(linear));singular=abs(det)<1e-15
  ph=np.c_[p,np.ones(len(p))]@M;w=ph[:,3];affine=bool(np.all(w==1))
  pw=np.full((len(p),3),np.nan,np.float64)
  wok=np.isfinite(w)&(w!=0);pw[wok]=ph[wok,:3]/w[wok,None]
  pw*=mpu
  for k in sorted(set([0,len(p)-1]))if len(p)else[]:
   independent=np.array(cache.GetLocalToWorldTransform(prim).Transform(Gf.Vec3d(*p[k])),np.float64)*mpu
   if np.isfinite(independent).all()and np.isfinite(pw[k]).all():transform_err=max(transform_err,float(np.max(np.abs(independent-pw[k]))))
  orient=str(mesh.GetOrientationAttr().Get(time)or'rightHanded');orientation.append(orient)
  ns=mesh.GetNormalsAttr().Get(time);n=np.empty((0,3),np.float64)if ns is None else np.asarray(ns,np.float64).reshape(-1,3)
  nw=np.full_like(n,np.nan)
  if not singular and len(n):
   nw=n@np.linalg.inv(linear).T;lens=np.linalg.norm(nw,axis=1);ok=np.isfinite(lens)&(lens>0);nw[ok]/=lens[ok,None];nw[~ok]=np.nan
  valid_index_count=bool(cs.sum()==len(inds));assert valid_index_count,(path,'counts/indexlength')
  assert np.all(cs==3),(path,'UNEXPECTED_NONTRIANGLE_NEEDS_SEPARATE_TRIANGULATION_POLICY')
  localfaces=inds.reshape(-1,3);facegood=(localfaces>=0).all(1)&(localfaces<len(p)).all(1)
  af=np.full(len(localfaces),np.nan,np.float64)
  if facegood.any():
   t=pw[localfaces[facegood]];af[facegood]=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
  facegood&=np.isfinite(af)&(af>0)
  bound_material,relation=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial();material_path=str(bound_material.GetPath())if bound_material else None
  mat_id=matidx.get(material_path,-1);fm=np.full(len(localfaces),mat_id,np.int32)
  subsets=UsdGeom.Subset.GetAllGeomSubsets(mesh);assert not subsets,(path,'UNEXPECTED_SUBSET_NEEDS_SEPARATE_MATERIAL_POLICY')
  points_local.append(p);points_world.append(pw);normals_local.append(n);normals_world.append(nw)
  triangles.append(localfaces+voff[-1]);tri_local.append(localfaces);tri_part.append(np.full(len(localfaces),part_id,np.int32));tri_material.append(fm);tri_source_face.append(np.arange(len(localfaces),dtype=np.int32));tri_valid.append(facegood);area2.append(af)
  meshok=bool(len(p)>0 and len(localfaces)>0 and np.isfinite(p).all()and np.isfinite(pw).all()and affine and not singular)
  mesh_geometry_valid.append(meshok)
  counts={'points_nonfinite':int((~np.isfinite(p)).sum()),'world_nonfinite':int((~np.isfinite(pw)).sum()),'normal_missing':int(ns is None or len(n)==0),'normal_nonfinite_local':int((~np.isfinite(n)).sum()),'normal_nonfinite_world':int((~np.isfinite(nw)).sum()),'normal_zero_length':int((np.linalg.norm(n,axis=1)==0).sum())if len(n)else0,'negative_determinant':int(det<0),'singular_transform':int(singular),'non_affine':int(not affine),'invalid_face_indices':int((~((localfaces>=0).all(1)&(localfaces<len(p)).all(1))).sum()),'repeated_vertex_faces':int(((localfaces[:,0]==localfaces[:,1])|(localfaces[:,0]==localfaces[:,2])|(localfaces[:,1]==localfaces[:,2])).sum()),'exact_zero_area_faces':int((af==0).sum()),'nonfinite_area_faces':int((~np.isfinite(af)).sum()),'material_unknown':int(mat_id<0)}
  problems.update(counts)
  meta.append({'part_id_asset_local_only':part_id,'prim_path':path,'status':'GEOMETRY_AVAILABLE'if meshok else'U_GEOMETRY_OR_TRANSFORM','semantic_role':'UNKNOWN','native8_association':'UNKNOWN','points_range':[voff[-1],voff[-1]+len(p)],'triangles_range':[foff[-1],foff[-1]+len(localfaces)],'normals_range':[noff[-1],noff[-1]+len(n)],'local_to_world_rowmatrix':M.tolist(),'determinant_linear':det,'negative_handed_transform':bool(det<0),'USD_orientation':orient,'normals_interpolation':str(mesh.GetNormalsInterpolation()),'material_path':material_path,'material_id':mat_id,'double_sided':bool(mesh.GetDoubleSidedAttr().Get(time)),'subdivision_scheme':str(mesh.GetSubdivisionSchemeAttr().Get(time)),'hole_indices':list(mesh.GetHoleIndicesAttr().Get(time)or[]),'time_samples':{'points':mesh.GetPointsAttr().GetNumTimeSamples(),'xform_ops':sum(op.GetAttr().GetNumTimeSamples()for op in UsdGeom.Xformable(prim).GetOrderedXformOps())},'problems':counts,'world_bounds_m':[finite_list(pw.min(0)),finite_list(pw.max(0))]if len(pw)else None})
  voff.append(voff[-1]+len(p));foff.append(foff[-1]+len(localfaces));noff.append(noff[-1]+len(n))
 arrays={'vertices_local':np.concatenate(points_local),'vertices_world_m':np.concatenate(points_world),'normals_local_authored':np.concatenate(normals_local),'normals_world_covector_unit':np.concatenate(normals_world),'triangles_global':np.concatenate(triangles).astype(np.int64),'triangles_mesh_local':np.concatenate(tri_local).astype(np.int64),'triangle_asset_local_part_id':np.concatenate(tri_part),'triangle_material_id':np.concatenate(tri_material),'triangle_source_face_local':np.concatenate(tri_source_face),'triangle_valid_numeric':np.concatenate(tri_valid),'triangle_double_area_m2':np.concatenate(area2),'mesh_vertex_offsets':np.asarray(voff,np.int64),'mesh_triangle_offsets':np.asarray(foff,np.int64),'mesh_normal_offsets':np.asarray(noff,np.int64),'mesh_local_to_world_row':np.asarray(xf,np.float64),'mesh_geometry_available':np.asarray(mesh_geometry_valid,bool),'mesh_prim_path':np.asarray([str(p.GetPath())for p in prims]),'mesh_orientation':np.asarray(orientation),'material_prim_path':np.asarray(mats),'meters_per_stage_unit':np.asarray(mpu,np.float64),'up_axis':np.asarray(up)}
 rawbytes=sum(x.nbytes for x in arrays.values());assert rawbytes<=BUDGET,(rawbytes,BUDGET)
 v=arrays['vertices_world_m'];finite=v[np.isfinite(v).all(1)];bounds=[finite.min(0).tolist(),finite.max(0).tolist()]
 extent=float(np.max(finite.max(0)-finite.min(0)));near_eps=1e-12*extent*extent
 near=int(np.count_nonzero((arrays['triangle_double_area_m2']>0)&(arrays['triangle_double_area_m2']<=near_eps)))
 # Material references: asset paths/bytes only. No image decode or network.
 refs=[]
 for prim in stage.Traverse():
  for attr in prim.GetAttributes():
   tn=str(attr.GetTypeName())
   if tn not in('asset','asset[]'):continue
   value=attr.Get(time);vals=list(value)if tn=='asset[]'and value is not None else[value]
   for val in vals:
    if not isinstance(val,Sdf.AssetPath):continue
    authored=val.path;resolved=val.resolvedPath
    if not authored:continue
    nonlocal_uri='://'in authored
    candidate=Path(resolved)if resolved else src.parent/authored
    exists=not nonlocal_uri and candidate.is_file()
    refs.append({'prim_path':str(prim.GetPath()),'attribute':attr.GetName(),'authored_asset_path':authored,'resolved_asset_path':resolved or None,'candidate_local_path':str(candidate)if not nonlocal_uri else None,'status':'LOCAL_BYTES_BOUND_NOT_DECODED'if exists else'U_UNRESOLVED_NONLOCAL'if nonlocal_uri else'U_MISSING_LOCAL_REFERENCE','binding':bind(candidate)if exists else None})
 layer_refs=[]
 for layer in stage.GetUsedLayers():
  for ref in layer.GetExternalReferences():
   q=Path(ref)if Path(ref).is_absolute()else Path(layer.realPath).parent/ref
   layer_refs.append({'layer':layer.identifier,'reference':ref,'local_exists':q.is_file(),'binding':bind(q)if q.is_file()else None})
 payloads=[{'prim_path':str(p.GetPath()),'loaded':p.IsLoaded(),'active':p.IsActive(),'status':'UNLOADED_PAYLOAD_U'if not p.IsLoaded()else'LOADED_METADATA'}for p in stage.TraverseAll()if p.HasPayload()]
 topboard=sorted([m for m in meta if any(seg.startswith('Topboard')or seg.startswith('Large_Topboard')for seg in m['prim_path'].split('/'))],key=lambda m:m['prim_path'])
 candidate=topboard[0]if topboard else None
 npz=HERE/'SCENE3_GEOMETRY_ASSET_LOCAL_ONLY.npz';assert not npz.exists();np.savez_compressed(npz,**arrays);assert npz.stat().st_size<=BUDGET
 with np.load(npz,allow_pickle=False)as zz:
  assert set(zz.files)==set(arrays)
  for k,v in arrays.items():assert zz[k].dtype==v.dtype and np.array_equal(zz[k],v,equal_nan=True)if v.dtype.kind in'fc'else np.array_equal(zz[k],v),k
 assert sha(src)==scope['source_asset']['sha256']
 report={'schema':'V93_ACTUAL_SCENE3_GEOMETRY_INVENTORY_SAFE_EXPORT_V1','completed':True,'PASS':True,'scope':bind(SCOPE),'code':bind(__file__),'generated':bind(HERE/'GENERATED_TRANSFORM_AND_INDEX_TESTS.json'),'source_asset':bind(src),'runtime':{'executable':sys.executable,'USD_version':str(Usd.GetVersion()),'numpy':np.__version__},'stage':{'meters_per_unit':mpu,'meters_per_unit_authored':stage.HasAuthoredMetadata('metersPerUnit'),'up_axis':up,'up_axis_authored':stage.HasAuthoredMetadata('upAxis'),'time':'Default','load_policy':'LoadNone: external payload availability kept explicit','world_conversion':'Local homogeneous ROW vector [x,y,z,1] @ USDlocalToWorldRow, dividew then multiply declared metersPerUnit. No Y/Z conversion, centering, extent normalization or axis swap.','raw_matrix_column_convention':'Column-vector consumers must transpose mesh_local_to_world_row.','bounds_world_m':bounds,'GF_Transform_probe_max_abs_m':transform_err},'counts':{'meshes':len(prims),'vertices':len(v),'triangles':len(arrays['triangles_global']),'materials':len(mats),'faces_declared_triangle':len(arrays['triangles_global']),'nontriangle_faces':0,'missing_geometry_meshes':len(missing),'near_degenerate_positive_area_faces':near,'near_degenerate_area2_threshold_m2':near_eps,'numeric_valid_triangles':int(arrays['triangle_valid_numeric'].sum())},'problems':dict(problems),'missing_geometry_U':missing,'external_payloads':payloads,'external_layer_references':layer_refs,'material_asset_references':refs,'per_mesh':meta,'lexicographic_Topboard_candidate':{'part_id_asset_local_only':candidate['part_id_asset_local_only'],'prim_path':candidate['prim_path'],'semantic_role':'UNKNOWN','native8_association':'UNKNOWN','candidate_by_node_name_only':True,'geometry_available':candidate['status']=='GEOMETRY_AVAILABLE'}if candidate else None,'export':{'binding':bind(npz),'array_raw_bytes':rawbytes,'archive_bytes':npz.stat().st_size,'budget_bytes':BUDGET,'all_arrays_no_object_dtype':all(v.dtype.kind!='O'for v in arrays.values()),'roundtrip_exact_including_NaN':True,'schema':{k:{'shape':list(v.shape),'dtype':str(v.dtype)}for k,v in arrays.items()}},'triangulation_policy':'All actual faces are triangles; source order preserved. No fan/ear clipping or geometric face editing performed. Numeric invalid/zero-area triangles retained with explicit triangle_valid_numeric flag.','normals_policy':'Authored local normals/interpolation preserved. World covectors inverse-transpose of local-rowlinear normalized where valid; nonfinite/missing retained as explicit diagnostics. Triangles NOT flipped for negative determinant or left-handed orientation; consumers must account for recorded orientation/transform.','material_and_transparency_limit':'Material references do not establish opaque semantic parts. Missing shader/texture references stayU; no rendered appearance or photorealism verified.','provenance_and_license':'Scoped local models_usd DATASET/_scale_report/_shader_report and release ATTRIBUTION do not map this scene_3 mesh SHA to an original author/license. Source provenance and reuse license remain UNKNOWN; no network or new data intake.','no_historical_substitution':'Separate local auxiliaryasset, not certified old1003 model; partIDs are sortedprim IDs and not nativecorner/semanticpart gold.','actions':{'actual_mesh_points_faces_normals_read':True,'native_refs_or_queries_read':0,'native8_snap_or_association':0,'new_render':0,'new_labels':0,'learned_model_or_weights':0,'RGB_texture_decode':0,'current223':0,'fit_PnP':0,'source_asset_modified':0},'fully_solved':False,'goal_status':'active'}
 write(HERE/'GEOMETRY_INVENTORY.json',report)
 write(HERE/'EXPORT_SEAL.json',{'completed':True,'PASS':True,'scope':bind(SCOPE),'code':bind(__file__),'report':bind(HERE/'GEOMETRY_INVENTORY.json'),'safe_NPZ':bind(npz),'generated':bind(HERE/'GENERATED_TRANSFORM_AND_INDEX_TESTS.json'),'source_asset':bind(src),'no_render_label_fit_current_PnP':True})
 print(json.dumps({'report':bind(HERE/'GEOMETRY_INVENTORY.json'),'seal':bind(HERE/'EXPORT_SEAL.json'),'NPZ':bind(npz),'counts':report['counts'],'problems':report['problems'],'Topboard_candidate':report['lexicographic_Topboard_candidate']}))
if __name__=='__main__':main()
