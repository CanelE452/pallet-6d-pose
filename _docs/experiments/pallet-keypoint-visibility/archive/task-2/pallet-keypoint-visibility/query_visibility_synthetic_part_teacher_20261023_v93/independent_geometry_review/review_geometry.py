"""Independent sourceUSD to safeNPZ reconciliation; no exporter import or render."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,datetime,re
import numpy as np
from pxr import Usd,UsdGeom,UsdShade,Gf
O=Path(__file__).resolve().parent
R=O.parent
G=Path('/home/minjae/Documents/github/pallet-pose')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()

def bind(p):
    return {'path':str(Path(p).resolve()),'sha256':sha(p)}

def read(p):
    return json.loads(Path(p).read_text())

def write(p,data):
    with Path(p).open('x') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def sanitize(x):
    if x is None or isinstance(x,(str,int,float,bool)):
        return x
    if isinstance(x,dict):
        return {str(k):sanitize(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [sanitize(v)for v in x]
    return str(x)

policy=read(O/'POLICY_BEFORE_GEOMETRY_REPLAY.json')
for descriptor in policy['bindings'].values():
    assert sha(descriptor['path'])==descriptor['sha256']
report=read(policy['bindings']['export_report']['path'])
source=Path(report['source_asset']['path'])
assert sha(source)==report['source_asset']['sha256']
with np.load(policy['bindings']['safe_NPZ']['path'],allow_pickle=False) as z:
    arrays={key:z[key]for key in z.files}
assert all(a.dtype.kind!='O'for a in arrays.values())
assert sum(a.nbytes for a in arrays.values())==report['export']['array_raw_bytes']
assert sum(a.nbytes for a in arrays.values())<32_000_000
for key,schema in report['export']['schema'].items():
    assert arrays[key].shape==tuple(schema['shape'])and str(arrays[key].dtype)==schema['dtype']
    if arrays[key].dtype.kind in 'fc':
        assert np.isfinite(arrays[key]).all(),key
stage=Usd.Stage.Open(str(source),load=Usd.Stage.LoadNone)
assert stage
mpu=float(UsdGeom.GetStageMetersPerUnit(stage))
up=str(UsdGeom.GetStageUpAxis(stage))
assert mpu==float(arrays['meters_per_stage_unit'])==1.
assert up==str(arrays['up_axis'])=='Y'
meshes=sorted([p for p in stage.Traverse()if p.IsA(UsdGeom.Mesh)],key=lambda p:str(p.GetPath()))
paths=[str(p.GetPath())for p in meshes]
assert paths==arrays['mesh_prim_path'].tolist()and len(paths)==308
selection=sorted([(hashlib.sha256(path.encode('utf-8')).hexdigest(),path,i)for i,path in enumerate(paths)])[:3]
write(O/'FIXED_MIN_SHA3_MESHES.json',{'policy':bind(O/'POLICY_BEFORE_GEOMETRY_REPLAY.json'),'selection':[{'sha256_of_prim_path_UTF8':s,'prim_path':p,'part_id_asset_local':i}for s,p,i in selection],'geometry_or_semantic_quality_selection':False,'prior_exporter_exposure_acknowledged':True})
selected={i for digest,path,i in selection}
material_paths=sorted(str(p.GetPath())for p in stage.Traverse()if p.IsA(UsdShade.Material))
assert arrays['material_prim_path'].tolist()==material_paths
material_index={path:i for i,path in enumerate(material_paths)}
xcache=UsdGeom.XformCache(Usd.TimeCode.Default())
v=arrays['vertices_world_m']
tri=arrays['triangles_global']
parts=arrays['triangle_asset_local_part_id']
assert v.shape==(172656,3)and tri.shape==(110478,3)
assert tri.min()>=0 and tri.max()<len(v)
assert arrays['triangle_valid_numeric'].all()
assert (np.diff(arrays['mesh_vertex_offsets'])>0).all()
assert (np.diff(arrays['mesh_triangle_offsets'])>0).all()
assert arrays['mesh_geometry_available'].all()
fresh_min=np.full(3,np.inf)
fresh_max=np.full(3,-np.inf)
scalar_results=[]
all_exact={'source_local_points':0,'source_triangles_local_global':0,'source_matrix':0,'source_normal_arrays':0,'triangle_partIDs':0,'triangle_materialIDs':0}
for i,prim in enumerate(meshes):
    mesh=UsdGeom.Mesh(prim)
    raw=np.asarray(mesh.GetPointsAttr().Get(),np.float64)
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),np.int64)
    indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),np.int64)
    assert np.all(counts==3)and counts.sum()==len(indices)
    localtri=indices.reshape(-1,3)
    lo,hi=map(int,arrays['mesh_vertex_offsets'][i:i+2])
    flo,fhi=map(int,arrays['mesh_triangle_offsets'][i:i+2])
    nlo,nhi=map(int,arrays['mesh_normal_offsets'][i:i+2])
    assert hi-lo==len(raw)and fhi-flo==len(localtri)
    assert np.array_equal(arrays['vertices_local'][lo:hi],raw)
    assert np.array_equal(arrays['triangles_mesh_local'][flo:fhi],localtri)
    assert np.array_equal(tri[flo:fhi],localtri+lo)
    assert (parts[flo:fhi]==i).all()
    assert np.array_equal(arrays['triangle_source_face_local'][flo:fhi],np.arange(len(localtri)))
    assert (tri[flo:fhi]>=lo).all()and(tri[flo:fhi]<hi).all()
    M=xcache.GetLocalToWorldTransform(prim)
    Mnp=np.asarray(M,np.float64)
    assert np.array_equal(arrays['mesh_local_to_world_row'][i],Mnp)
    ph=np.concatenate([raw,np.ones((len(raw),1),np.float64)],axis=1)@Mnp
    fresh_world=ph[:,:3]/ph[:,3,None]*mpu
    assert np.array_equal(fresh_world,v[lo:hi])
    fresh_min=np.minimum(fresh_min,fresh_world.min(0))
    fresh_max=np.maximum(fresh_max,fresh_world.max(0))
    normals=np.asarray(mesh.GetNormalsAttr().Get(),np.float64).reshape(-1,3)
    assert nhi-nlo==len(normals)and np.array_equal(arrays['normals_local_authored'][nlo:nhi],normals)
    material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    material_path=str(material.GetPath())if material else None
    expected_material=material_index.get(material_path,-1)
    assert (arrays['triangle_material_id'][flo:fhi]==expected_material).all()
    assert arrays['mesh_orientation'][i]==str(mesh.GetOrientationAttr().Get())=='rightHanded'
    for key in all_exact:
        all_exact[key]+=1
    if i in selected:
        scalar=np.asarray([tuple(M.Transform(Gf.Vec3d(float(p[0]),float(p[1]),float(p[2]))))for p in raw],np.float64)*mpu
        delta=np.abs(scalar-v[lo:hi])
        bound=1e-12*np.maximum(1.,np.abs(v[lo:hi]))
        assert (delta<=bound).all()
        scalar_results.append({'part_id_asset_local':i,'prim_path':paths[i],'source_vertices_replayed':len(raw),'source_faces_and_material_ids_exact':len(localtri),'material_path':material_path,'material_id':expected_material,'Gf_scalar_max_abs_m':float(delta.max()),'Gf_scalar_RMS_m':float(np.sqrt(np.mean(delta**2))),'all_vertices_within_prefixed_bound':True,'semantic_role':'UNKNOWN','native8_association':'UNKNOWN'})
assert np.array_equal(fresh_min,v.min(0))and np.array_equal(fresh_max,v.max(0))
assert np.array_equal(np.asarray(report['stage']['bounds_world_m']),np.stack([fresh_min,fresh_max]))
# Official extents-based world-bound is recorded separately because authored extents
# may have their own float precision and need not equal pointwise bounds bit-for-bit.
bb=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy],useExtentsHint=False)
box=bb.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
box_arr=np.asarray([tuple(box.GetMin()),tuple(box.GetMax())],np.float64)*mpu
bbox_diag={'official_extents_world_bounds_m':box_arr.tolist(),'pointwise_bounds_m':[fresh_min.tolist(),fresh_max.tolist()],'max_abs_difference_m':float(np.max(np.abs(box_arr-np.stack([fresh_min,fresh_max])))),'interpretation':'Extents-based diagnostic only; authoritative export parity uses actual source points/transform, without recentering/scaling.'}
# Original asset provenance/custom metadata; do not query textures or models.
custom=[]
for prim in stage.Traverse():
    asset=prim.GetAssetInfo()
    data=prim.GetCustomData()
    if asset or data:
        custom.append({'prim_path':str(prim.GetPath()),'assetInfo':sanitize(asset),'customData':sanitize(data)})
root_metadata={'customLayerData':sanitize(stage.GetRootLayer().customLayerData),'stage_pseudoroot_metadata':sanitize(stage.GetPseudoRoot().GetAllMetadata()),'prims_with_assetInfo_or_customData':custom}
known=[source.parent/'DATASET.md',source.parent/'README.md',source.parent/'SOURCES.txt',source.parent/'LICENSE',source.parent/'license.txt',source.parent/'_scale_report.txt',source.parent/'_shader_report.txt',G/'README.md',G/'data/pallet/release/attribution/ATTRIBUTION.md']
files=[]
for path in known:
    if path.is_file():
        text=path.read_text(errors='replace')
        matched=[{'line':i+1,'text':line}for i,line in enumerate(text.splitlines())if re.search(r'scene_3|sources\.txt|license|licen[cs]e|attribution|author|저작|출처',line,re.I)]
        files.append({'path':str(path),'binding':bind(path),'status':'READ_SCOPED_LOCAL_DOCUMENT','matching_lines':matched[:60]})
    else:
        files.append({'path':str(path),'status':'NOT_PRESENT_AT_THIS_DOCUMENTED_PATH'})
provenance={'schema':'V93_BOUNDED_SOURCE_PROVENANCE_METADATA_REVIEW_V1','completed':True,'root_USD_metadata':root_metadata,'document_inventory':files,'conclusion':'No verified exact sourceSHA→original author/license/release manifest mapping established in this bounded local scope. USDcustomdata and generic attribution are not proof of a scene_3 reuse license. RemainsUNKNOWN; not a worldwide/no-file assertion.','no_network':True,'RGB_texture_decode':0,'training_admission':False}
write(O/'SOURCE_PROVENANCE_METADATA_REVIEW.json',provenance)
write(O/'GEOMETRY_AUDIT.json',{'schema':'V93_INDEPENDENT_ACTUAL_USD_SAFE_NPZ_REVIEW_V1','completed':True,'PASS':True,'blockers':[],'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'policy':bind(O/'POLICY_BEFORE_GEOMETRY_REPLAY.json'),'auditor':bind(__file__),'source_USD':bind(source),'export_seal':policy['bindings']['export_seal'],'safe_NPZ':policy['bindings']['safe_NPZ'],'fixed3':bind(O/'FIXED_MIN_SHA3_MESHES.json'),'full_source_USD_NPZ_exact_checks_mesh_counts':all_exact,'all_shapes_finite_numeric_and_index_part_material_ranges_PASS':True,'all_source_world_vertices_exact':True,'all_source_pointwise_bounds_exact':True,'orientation_all308':'rightHanded','no_scale_axis_or_native_ref_change':True,'source_units_authored_metersPerUnit':mpu,'up_axis':up,'scalar_Gf_fixed3':scalar_results,'scalar_Gf_bound':'1e-12*max(1,abs(worldcoordinate)) percoordinate fixed in policy before replay','official_BBox_diagnostic':bbox_diag,'provenance_review':bind(O/'SOURCE_PROVENANCE_METADATA_REVIEW.json'),'한국어_판정':'원본USD와NPZ의 좌표·행렬·삼각형·part/materialID 및 bounds가 일치했습니다. minSHA3mesh의 모든정점을 독립scalarGf로 재생해 사전수치범위 안에 있음을 확인했습니다. part의미/native대응/실제팔레트 여부는 render검토 전 UNKNOWN입니다.','limits':['Auditor previously authored export and has prior asset exposure; separately implemented arithmetic, not independent blinded annotation.','Source author/license remainsUNKNOWN in bounded metadata. No semanticpart or appearance certified.','No manifold/watertightness, physical scale calibration, shading or real transfer claim.','No render pixels, texture/image decode, nativequery/reference, learnedmodel/weights or current read.'],'actions':{'actual_source_geometry':True,'original_or_export_files_modified':0,'render':0,'RGB_texture_decode':0,'semantic_labels':0,'learned_model_weights':0,'fit':0,'current223':0,'PnP':0},'goal_status':'active','fully_solved':False})
write(O/'SEAL.json',{'completed':True,'PASS':True,'policy':bind(O/'POLICY_BEFORE_GEOMETRY_REPLAY.json'),'code':bind(__file__),'report':bind(O/'GEOMETRY_AUDIT.json'),'source_provenance':bind(O/'SOURCE_PROVENANCE_METADATA_REVIEW.json'),'fixed3':bind(O/'FIXED_MIN_SHA3_MESHES.json'),'source_export_unchanged':True,'render_labels_fit_current_PnP':0})
print(json.dumps({'report':bind(O/'GEOMETRY_AUDIT.json'),'seal':bind(O/'SEAL.json'),'selected_scalar_meshes':scalar_results,'BBox_diag':bbox_diag},ensure_ascii=False))
