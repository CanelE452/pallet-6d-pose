"""Model-only AUX surface points for official wood/foam assets. No image or label input."""
from pathlib import Path
import hashlib,itertools,json,sys
sys.dont_write_bytecode=True
import numpy as np
import trimesh
V=Path(__file__).resolve().parent;P=V.parent
ROOT=V/'ROOT_MATERIAL_SOURCE_GEOMETRY_BEFORE_INTAKE.json'
ROOT_SHA='0ac19ad283aa7080129243ac1f91fa4e8165a5757c3f7f0d1d1fd46f63566ccf'
ALGORITHM=P/'query_visibility_ycb_surface_cues_20261023_v106/build_surface_candidates.py'
ALGORITHM_SHA='a90f94237ffa8e3e5a029e9caa73b4d574e0e323783e8dc81ef86fd4ce47588c'
ORIGINAL_GEOMETRY_BLOCK_SHA256='aec1af919e863387f4bac0ab05f926af086f7c183bcc81a787e60cf28bca2203'
INSTANCES=P/'query_visibility_public_real_20261022_v63/ycb/instances.jsonl'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def derive(path,oid):
    bits=np.array(list(itertools.product([0,1],repeat=3)))
    mesh = trimesh.load(path, process=False)
    xyz = np.asarray(mesh.vertices, dtype=np.float64)
    face = np.asarray(mesh.faces, dtype=np.int64)
    assert xyz.ndim == 2 and xyz.shape[1] == 3 and np.isfinite(xyz).all()
    assert face.shape[1] == 3 and face.min() >= 0 and face.max() < len(xyz)
    mean = xyz.mean(0)
    eigen, basis = np.linalg.eigh(np.cov(xyz.T))
    order = np.argsort(-eigen, kind='stable');eigen=eigen[order];basis=basis[:,order]
    for j in range(3):
        if basis[np.argmax(abs(basis[:,j])),j] < 0:basis[:,j] *= -1
    if np.linalg.det(basis) < 0:basis[:,2] *= -1
    local = (xyz-mean)@basis;lo=local.min(0);hi=local.max(0);mid=(lo+hi)/2
    corners = lo+(hi-lo)*bits
    incident = [[] for _ in range(len(xyz))]
    normals = np.cross(xyz[face[:,1]]-xyz[face[:,0]], xyz[face[:,2]]-xyz[face[:,0]])
    norm = np.linalg.norm(normals,axis=1);good=norm>0;normals[good]/=norm[good,None]
    for fi,ff in enumerate(face):
        if good[fi]:
            for vi in ff:incident[int(vi)].append(fi)
    candidates=[]
    for k,bit in enumerate(bits):
        region=np.all(np.where(bit==0,local<=mid,local>=mid),axis=1)
        indices=np.flatnonzero(region)
        if len(indices)==0:
            candidates.append(dict(aux_surface_id=f'YCB{oid:02d}_SURFACE{k}',region_bits=bit.tolist(),xyz_mm=None,reason='EMPTY_OCTANT'));continue
        d2=((local[indices]-corners[k])**2).sum(1);vi=int(indices[np.argmin(d2)])
        distances=np.linalg.norm(xyz-xyz[vi],axis=1);near=xyz[distances<=5.]
        ns=normals[incident[vi]]
        angle=float(np.rad2deg(np.arccos(np.clip((ns@ns.T).min(),-1,1)))) if len(ns) else None
        candidates.append(dict(aux_surface_id=f'YCB{oid:02d}_SURFACE{k}',region_bits=bit.tolist(),
            original_mesh_vertex_index=vi,xyz_mm=xyz[vi].tolist(),on_mesh_by_construction=True,
            PCA_OBB_corner_xyz_mm=(mean+basis@corners[k]).tolist(),OBB_corner_to_vertex_mm=float(np.sqrt(d2.min())),
            octant_vertex_count=len(indices),local_5mm_vertex_count=len(near),local_5mm_coordinate_range_mm=[near.min(0).tolist(),near.max(0).tolist()],
            incident_triangle_indices=incident[vi],incident_normal_max_angle_deg=angle,
            geometry_spread_is_not_annotation_uncertainty=True,physical_corner_or_part_identity='UNKNOWN',native8_mapping='UNKNOWN'))

    edge_multiplicity=np.bincount(mesh.edges_unique_inverse)
    unique_xyz=len(np.unique(xyz,axis=0))
    canonical_faces=np.sort(face,axis=1)
    ratios=[None if eigen[j+1]==0 else float(eigen[j]/eigen[j+1])for j in range(2)]
    return dict(object_id=oid,model=bind(path),vertices=len(xyz),triangles=len(face),
                mesh_watertight=bool(mesh.is_watertight),mesh_winding_consistent=bool(mesh.is_winding_consistent),
                native_AABB_bounds_mm=[xyz.min(0).tolist(),xyz.max(0).tolist()],
                pca_origin_mm=mean.tolist(),PCA_basis_columns=basis.tolist(),PCA_basis_determinant=float(np.linalg.det(basis)),
                covariance_eigenvalues_mm2=eigen.tolist(),PCA_axis_eigenvalue_ratios=ratios,
                axis_symmetry_caution='Near-equal eigenvalues imply orientation sensitivity. Original mesh hash fixes this candidate version; no semantic axis claim.',
                geometry_quality=dict(all_vertices_finite=True,all_triangle_indices_in_range=True,
                    exact_zero_area_triangle_count=int((~good).sum()),exact_duplicate_vertex_count=len(xyz)-unique_xyz,
                    unused_vertex_count=len(xyz)-len(np.unique(face)),duplicate_triangle_vertex_sets=len(face)-len(np.unique(canonical_faces,axis=0)),
                    boundary_unique_edges=int((edge_multiplicity==1).sum()),nonmanifold_unique_edges=int((edge_multiplicity>2).sum()),
                    no_mesh_repair_or_vertex_reordering=True),
                candidates=candidates,all8_regions_retained=True,
                unique_selected_vertices=len({x['original_mesh_vertex_index']for x in candidates if x['xyz_mm']is not None}),
                empty_octants=sum(x['xyz_mm']is None for x in candidates),
                OBB_corner_to_vertex_mm_range=[min(x['OBB_corner_to_vertex_mm']for x in candidates if x['xyz_mm']is not None),max(x['OBB_corner_to_vertex_mm']for x in candidates if x['xyz_mm']is not None)],
                physical_corner_or_part_identity='UNKNOWN',native8_mapping='UNKNOWN')
def main():
 assert sha(ROOT)==ROOT_SHA and sha(ALGORITHM)==ALGORITHM_SHA and not(V/'SURFACE_CANDIDATE_RESULT.json').exists()
 root=read(ROOT);assert bind(INSTANCES)==root['instances']
 intake=read(V/'geometry_intake/GEOMETRY_INTAKE_RECEIPT.json');assert intake['completed']and intake['PASS']
 models={int(Path(x['archive_entry']['name']).stem.split('_')[1]):x for x in intake['selected_members']}
 assert set(models)=={16,21}
 for item in models.values():check(item['file'])
 rule=dict(name='AUX_SURFACE_CUE_PCA_OCTANT_VERTEX_V1',object_ids=[16,21],coordinate_units='BOP model mm',
           basis='Exact V106: covariance eigenvectors descending; largest component positive; third axis flipped for positive handedness.',
           regions='Exact V106: eight half-space octants of PCA-box midpoint, lower/upper bitXYZ.',
           candidate='Exact V106: nearest ORIGINAL vertex to each PCA-box corner inside its octant; exact first original vertex index tie.',
           diagnostic_neighborhood_radius_mm=5.0,pruning='None; retain all8 including empty or duplicate.',
           geometry_operation_source=bind(ALGORITHM),source_block_sha256=ORIGINAL_GEOMETRY_BLOCK_SHA256,
           image_label_model_score_selection=False,native_mapping='UNKNOWN',human_gold=False,training_admission=False)
 write(V/'SURFACE_CANDIDATE_POLICY_BEFORE_GEOMETRY.json',dict(completed=True,root=bind(ROOT),code=bind(__file__),intake=bind(V/'geometry_intake/GEOMETRY_INTAKE_RECEIPT.json'),rule=rule,all_source_pixels_read=0))
 source=[json.loads(x)for x in INSTANCES.read_text().splitlines()]
 objects=[]
 for oid in(16,21):
  o=derive(Path(models[oid]['file']['path']),oid)
  rows=[x for x in source if x['object_id']==oid]
  counts={s:sum(x['split']==s for x in rows)for s in sorted({x['split']for x in rows})}
  o['source_inventory_metadata_only']=dict(instances_by_original_split=counts,scene_ids=sorted({x['scene_id']for x in rows}),unique_RGB_metadata_frames=len({(x['scene_id'],x['frame_id'])for x in rows}))
  objects.append(o)
 result=dict(completed=True,root=bind(ROOT),policy=bind(V/'SURFACE_CANDIDATE_POLICY_BEFORE_GEOMETRY.json'),code=bind(__file__),
             official_model_intake=bind(V/'geometry_intake/GEOMETRY_INTAKE_RECEIPT.json'),source_instances=bind(INSTANCES),construction_rule=rule,objects=objects,
             fixed_original_mesh_points=16,no_image_check_before_point_freeze=True,
             original_V100_rejected_offsurface_AABB_unchanged=True,
             limitations=['These are original scanned mesh surface vertices, not certified physical junctions or pallet corners.',
                          'PCA/octant and5mm neighborhood diagnostics are fixed geometry definitions, not human localization uncertainty.',
                          'Non-watertight mesh, axis symmetry, scan topology and target-alone visibility need separate evidence; no D/E/S/U teacher has been generated.',
                          'Object16 has existing TRAIN/VAL metadata; object21 has no existing sourceVAL. No scene/fold or point replacement.',
                          'V112 scene holdout remains excluded from training and selection.'],
             RGB_depth_mask_pixel_reads=0,learned_model_forward_fit_private_current_PnP=0,new_teacher_labels=False,
             native_mapping='UNKNOWN',human_gold=False,training_admission=False,
             runtime=dict(python=sys.version,numpy=np.__version__,trimesh=trimesh.__version__))
 write(V/'SURFACE_CANDIDATE_RESULT.json',result)
 write(V/'GEOMETRY_SEAL.json',dict(completed=True,PASS=True,result=bind(V/'SURFACE_CANDIDATE_RESULT.json'),policy=result['policy'],code=result['code'],intake=result['official_model_intake'],source_geometry_only=True,no_teacher_labels_or_training_admission=True))
 print(json.dumps(dict(completed=True,result=bind(V/'SURFACE_CANDIDATE_RESULT.json'),seal=bind(V/'GEOMETRY_SEAL.json'),objects=[dict(object_id=o['object_id'],vertices=o['vertices'],triangles=o['triangles'],unique_points=o['unique_selected_vertices'],empty_octants=o['empty_octants'],watertight=o['mesh_watertight'],OBB_corner_distance=o['OBB_corner_to_vertex_mm_range'],source=o['source_inventory_metadata_only'])for o in objects])))
if __name__=='__main__':main()
