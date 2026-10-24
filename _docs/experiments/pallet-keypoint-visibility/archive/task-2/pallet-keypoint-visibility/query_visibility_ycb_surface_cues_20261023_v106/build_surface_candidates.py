"""Fixed model-only AUX surface candidates; geometry/metadata, no RGB or label generation."""
from pathlib import Path
import hashlib
import itertools
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import trimesh

V = Path(__file__).resolve().parent
P = V.parent
OLD = P/'query_visibility_public_box_cue_20261023_v100'
INSTANCES = P/'query_visibility_public_real_20261022_v63/ycb/instances.jsonl'
OLD_BINDINGS = P/'query_visibility_public_external_ray_20261023_v87/corpus_build_v1/SOURCE_DEPTH_MASK_BYTE_BINDINGS.json'
IDS = (2, 3, 7, 8)
RULE = dict(name='AUX_SURFACE_CUE_PCA_OCTANT_VERTEX_V1', object_ids=list(IDS),
    coordinate_units='BOP model mm', basis='Vertex covariance eigenvectors in descending eigenvalue order; axis sign makes largest absolute component positive; third axis flipped if needed for right-handed basis.',
    regions='Eight PCA-box half-space octants around the PCA box midpoint, bitXYZ0/1 = lower/upper. Empty region retains null.',
    candidate='Nearest original mesh vertex to each PCA box corner within its octant, squared Euclidean model-mm distance, exact first original vertex index tie.',
    diagnostic_neighborhood_radius_mm=5.0, pruning='None. All eight AUX regions retained, including empty/duplicate candidates.',
    semantic_claim='Actual scanned mesh surface point only; corner/junction/part identity remains UNKNOWN until separate actual RGB evidence.',
    historical_V100='Original rejected AABB definition unchanged; these are new AUX IDs, not corrected native8.',
    native_pallet_mapping='UNKNOWN', human_gold=False, training_admission=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def load(path):
    return json.loads(Path(path).read_text())


def main():
    assert not (V/'SURFACE_CANDIDATE_RESULT.json').exists()
    rows = [json.loads(x) for x in INSTANCES.read_text().splitlines()]
    byte_map = {x['path']: x for x in load(OLD_BINDINGS)['files']}
    old = load(OLD/'cue_geometry_analysis/GEOMETRY_CORNER_DISTANCE_DIAGNOSTICS.json')
    old_obj = {x['object_id']: x for x in old['objects']}
    bits = np.array(list(itertools.product([0, 1], repeat=3)))
    objects = []
    scene_cache = {}
    for oid in IDS:
        path = OLD/'geometry_intake'/f'obj_{oid:06d}.ply'
        assert bind(path)['sha256'] == old_obj[oid]['model']['sha256']
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
        selected=sorted([r for r in rows if r['object_id']==oid],key=lambda r:(r['scene_id'],r['frame_id'],r['instance_id']))
        counts={};complete_owners=0;own_complete=0;depth_complete=0;projection_oob=0;projection_front=0;scenes=set();frames=set();metadata_bindings={}
        for r in selected:
            counts[r['split']]=counts.get(r['split'],0)+1
            scene=Path(r['image']).parent.parent;sid=str(scene);scenes.add(r['scene_id']);frames.add((r['scene_id'],r['frame_id']))
            if sid not in scene_cache:
                scene_cache[sid]=(load(scene/'scene_gt.json'),load(scene/'scene_camera.json'))
            gt,cameras=scene_cache[sid];gts=gt[str(r['frame_id'])];cam=cameras[str(r['frame_id'])]
            assert gts[r['instance_id']]['obj_id']==oid
            assert np.array_equal(gts[r['instance_id']]['cam_R_m2c'],r['cam_R_m2c']) and np.array_equal(gts[r['instance_id']]['cam_t_m2c'],r['cam_t_m2c'])
            assert np.array_equal(cam['cam_K'],r['K']) and cam['depth_scale']==r['depth_scale']
            own_complete+=int(all(Path(r[key]).is_file() for key in ('mask_vis','mask_full')))
            depth_complete+=int(Path(r['depth']).is_file() and r['depth'] in byte_map)
            owners=[scene/'mask_visib'/f"{r['frame_id']:06d}_{i:06d}.png" for i in range(len(gts))]
            complete_owners+=int(all(p.is_file() and str(p) in byte_map for p in owners))
            for name in ('scene_gt.json','scene_camera.json','scene_gt_info.json'):
                metadata_bindings[str(scene/name)]=None
            R=np.array(r['cam_R_m2c']).reshape(3,3);t=np.array(r['cam_t_m2c']);K=np.array(r['K']).reshape(3,3)
            points=np.array([c['xyz_mm'] for c in candidates if c['xyz_mm'] is not None]);camera=points@R.T+t;uv=(camera@K.T)[:,:2]/camera[:,2,None]
            projection_front+=int((camera[:,2]>0).sum());projection_oob+=int(((uv[:,0]<0)|(uv[:,0]>=r['width'])|(uv[:,1]<0)|(uv[:,1]>=r['height'])).sum())
        r=selected[0];R=np.array(r['cam_R_m2c']).reshape(3,3);t=np.array(r['cam_t_m2c']);K=np.array(r['K']).reshape(3,3)
        pilot=[]
        for c in candidates:
            if c['xyz_mm'] is None:pilot.append(dict(aux_surface_id=c['aux_surface_id'],projected_xy=None));continue
            camera=R@np.array(c['xyz_mm'])+t;uv=K@camera;uv=uv[:2]/uv[2]
            pilot.append(dict(aux_surface_id=c['aux_surface_id'],projected_xy=uv.tolist(),camera_Z_mm=float(camera[2])))
        objects.append(dict(object_id=oid,model=bind(path),vertices=len(xyz),triangles=len(face),
            mesh_watertight=bool(mesh.is_watertight),mesh_winding_consistent=bool(mesh.is_winding_consistent),
            pca_origin_mm=mean.tolist(),PCA_basis_columns=basis.tolist(),covariance_eigenvalues_mm2=eigen.tolist(),
            PCA_axis_eigenvalue_ratios=(eigen[:-1]/eigen[1:]).tolist(),axis_symmetry_caution='Near-equal PCA eigenvalues can make orientation unstable; original asset hash fixes this version.',
            candidates=candidates,unique_selected_vertices=len({c['original_mesh_vertex_index'] for c in candidates if c['xyz_mm'] is not None}),
            prior_AABB_surface_distance_mm=old_obj[oid]['nearest_surface_distance_mm_range'],
            source_inventory=dict(instances_by_existing_split=counts,scene_ids=sorted(scenes),unique_frames=len(frames),own_full_visible_mask_files=own_complete,
                depth_files_with_existing_V87_SHA_binding=depth_complete,complete_same_frame_all_instance_visible_masks_with_existing_binding=complete_owners,
                all_candidate_projections_front_of_camera=projection_front,total_projections=len(selected)*8,continuous_OOB_projections=projection_oob,
                metadata_bindings=[bind(Path(p)) for p in sorted(metadata_bindings)]),
            fixed_pilot=dict(selection='Lexicographically first existing instance, no image/visibility filtering',scene_id=r['scene_id'],frame_id=r['frame_id'],instance_id=r['instance_id'],split=r['split'],
                image=dict(path=r['image'],sha256=r['image_sha256']),mask_visible=r['mask_vis'],mask_full=r['mask_full'],depth=r['depth'],depth_scale_to_mm=r['depth_scale'],
                raw_hw=[r['height'],r['width']],K=r['K'],cam_R_m2c=r['cam_R_m2c'],cam_t_m2c_mm=r['cam_t_m2c'],projected_surface_candidates=pilot)))
    result=dict(completed=True,construction_rule=RULE,code=bind(__file__),source_instances=bind(INSTANCES),
        prior_V100_geometry=bind(OLD/'cue_geometry_analysis/GEOMETRY_CORNER_DISTANCE_DIAGNOSTICS.json'),
        original_V100_rejected_AABB_unchanged=True,prior_depth_owner_byte_bindings=bind(OLD_BINDINGS),objects=objects,
        visibility_evidence=dict(available='Official GT pose/K + mm mesh; scene depth*depth_scale to mm; intended visible/amodal mask; complete same-frame visible-instance masks per scene_gt.',
            self_visibility_required='Target-alone mesh ray first-hit must match the fixed surface candidate within separately frozen geometric/raster uncertainty; own target mask membership alone cannot distinguish another target surface in front.',
            D_candidate='Only self-visible surface projection, valid consistent observed depth and unique intended visible owner support can become a conditional surface-visible proposal; RGB correspondence still requires review.',
            E_candidate='Only self-visible expected surface, clearly nearer observed depth and unique different annotated instance visible owner support can become conditional external occlusion; missing target pixels alone do not suffice.',
            known_geometric_self='Earlier intersection with the same target mesh is geometric S/outside learned D/E; do not label external E.',
            unresolved='No depth, range tie/quantization, owner overlap/unassigned, off-image, uncertain ray/edge/registration or unrecognizable cue remain U.',
            not_yet_computed='No target-alone ray tests or depth/mask pixel sampling, no actual visual cue review, no 2D uncertainty estimates or D/E/S labels in this artifact.',
            sampling='Future source q/L/operator must be prospectively frozen separately; GT projection is label/evidence, not silently substituted for actual initializer q.'),
        limitations=['All1100 cached instances of these four objects are TRAIN; fixed existing VAL contains zero of these IDs, so this alone supplies no independent source-heldout assessment.',
            'Model surface ownership is certified only in mesh geometry; distinct physical corner/part appearance and transfer to flat wooden pallet are UNKNOWN.',
            'PCA box-to-vertex construction intentionally changes AUX reference definition, never the original virtual/native reference.',
            '5mm neighbor ranges and normal angles are geometry diagnostics, not human localization uncertainty or selected visibility thresholds.'],
        RGB_pixels_read=0,depth_mask_pixels_read=0,new_downloads=0,model_fit_private_PnP=0,training_admission=False,human_gold=False,native_mapping='UNKNOWN',
        runtime=dict(python=sys.version,numpy=np.__version__,trimesh=trimesh.__version__))
    with (V/'SURFACE_CANDIDATE_RESULT.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(completed=True,result=bind(V/'SURFACE_CANDIDATE_RESULT.json'),objects=[dict(object_id=o['object_id'],unique_surface_points=o['unique_selected_vertices'],counts=o['source_inventory']['instances_by_existing_split']) for o in objects])))


if __name__=='__main__':main()
