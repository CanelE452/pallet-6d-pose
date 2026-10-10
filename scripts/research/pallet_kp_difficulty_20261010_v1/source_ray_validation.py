"""Fixed actual-mesh query-ray audit of the existing source-test128 targets.

The two 0.05-pixel normal offsets diagnose first-surface sensitivity at an
exact mesh edge. They do not create new labels, images or trained models.
Protocol and immutable input hashes are written before the first ray cast.
"""
from collections import Counter
from pathlib import Path
import time
import numpy as np
import cv2
from .source_ceiling import C,S,DOC,EDGES,query_geometry,stats
from scripts.research.pallet_observation_refiner_20261009_v1.solver import project

EPS=.05
SHIFTS=(0.,EPS,-EPS)


def kind(valid,lo):
    return 'IGNORE' if not valid else 'POSITIVE' if lo<65 else 'NONE'


def support(visible,position):
    h,w=visible.shape;x,y=position
    if not (0<=x<w and 0<=y<h):return False
    ix,iy=int(round(x-.5)),int(round(y-.5))
    return bool((visible[max(0,iy-1):min(h,iy+2),max(0,ix-1):min(w,ix+2)]>127).any())


def setup_protocol():
    assert not (DOC/'SOURCE_RAY_PROTOCOL.json').exists(),'Preserve a started ray audit'
    families=S.selected_families();sources=list(C.iter_rows(DOC/'SOURCE_CEILING_ROWS.jsonl.gz'))[896:1024]
    mesh_path=C.SCRATCH/'actual_mesh/scene.usd.npz';meta_path=mesh_path.with_suffix('.json')
    assert mesh_path.exists() and meta_path.exists(),'Audit must reuse the already composed actual mesh'
    evidence=C.read(meta_path);asset=C.ROOT/evidence['asset']['path']
    assert C.sha(asset)==evidence['asset']['sha256']
    inputs=[]
    for source in sources:
        paths=S.locate(families[source['index']])
        inputs.append(dict(id=source['id'],index=source['index'],source_ceiling_semantic_sha256=C.digest(source),
            source_annotation=C.binding(paths['label']),visible_mask=C.binding(paths['visible']),
            source_RGB=source['source_RGB_binding']))
    oldcode=Path(S.__file__).parent
    protocol=dict(schema='fixed_source_test128_actual_mesh_query_ray_protocol_v1',created_before_first_cast=True,
        split='unchanged source_test indices896..1023',families=128,query_count=128*84,
        max_rays=128*84*3,ray_shifts_original_pixels=list(SHIFTS),
        offset_definition='exact projected actual physical segment / initial query normal; plus and minus 0.05 raw image pixel along that same normal',
        original_query_guards='unchanged query valid; abs determinant>=1e-8; projected segment fraction in[0,1]; abs normal offset<=32px',
        perspective_correct_segment='u3d=(fraction/z_b)/((1-fraction)/z_a+fraction/z_b)',
        physical_tolerance='closest point distance <=1e-5 * physical diagonal; float32 mesh and closest query as original',
        depth_tolerance='abs t_hit minus source edge camera Z <=0.001 * physical diagonal, as original',
        depth_units='Ray camera Z direction is1; t_hit is camera Z, not Euclidean range',
        supplied_visible_mask_test='unchanged 3x3 any>127 around round(x-.5),round(y-.5)',
        actual_mesh='already composed scene.usd triangle mesh, same normalized axes and dimensions as original targets; no cuboid/hull replaces physical geometry',
        ray_precision='Open3D CPU float32 mesh/rays reproduces original; fixed +/-0.05px diagnostic first surfaces, no claim of double precision',
        max_closest_query_points=128*84,max_physical_edge_closest_points=128*84,
        expected_scene_builds=128,source_target_labels_unchanged=True,diagnostic_offsets_not_supervision=True,
        new_RGB=0,new_training_updates=0,new_head_forwards=0,new_detector_forwards=0,new_PnP_calls=0,
        no_full_image_mask_or_depth_render=True,no_threshold_sweep=True,
        bindings=dict(mesh_cache=C.binding(mesh_path),mesh_cache_metadata=C.binding(meta_path),actual_asset=C.binding(asset),
            source_ceiling=C.binding(DOC/'SOURCE_CEILING_ROWS.jsonl.gz'),
            source_no_ray_geometry=C.binding(DOC/'SOURCE_NO_RAY_GEOMETRY_ROWS.jsonl.gz'),
            original_target_code=C.binding(oldcode/'training.py'),source_geometry_code=C.binding(oldcode/'source_audit.py'),
            original_query_code=C.binding(oldcode/'model.py'),audit_code=C.binding(Path(__file__))),inputs=inputs)
    C.write(DOC/'SOURCE_RAY_PROTOCOL.json',protocol)
    return families,sources,protocol


def run():
    import open3d as o3d
    cv2.setNumThreads(1);begin=time.monotonic();families,sources,protocol=setup_protocol()
    v,tri,mesh=S.mesh_normalized();records=[];rays_count=closest_count=edge_closest_count=0
    for frame,source in enumerate(sources):
        paths=S.locate(families[source['index']]);ann=C.read(paths['label']);g=S.geometry(families[source['index']],ann)
        scene=S.ray_scene(v,tri,g['dims']);uv=project(g['X'],g['R'],g['t'],g['K'])
        visible=cv2.imread(str(paths['visible']),0);assert visible is not None and list(visible.shape)==g['hw']
        points=np.asarray(source['frozen_selected_points']);centers,normals,qvalid=query_geometry(points)
        diagonal=float(np.linalg.norm(g['dims']));tol=.001*diagonal;targets=source['targets'];rows=[];sourceids=[];xs=[]
        for i in range(84):
            edge=i//7;a,b=EDGES[edge];matrix=np.column_stack([normals[i],-(uv[b]-uv[a])])
            row=dict(query=i,edge=edge,cached_target=kind(targets['valid'][i],targets['lo'][i]),
                predicted_role=source['predicted_role_query_ids'][i],query_valid=bool(qvalid[i]))
            rows.append(row)
            if not qvalid[i] or abs(np.linalg.det(matrix))<1e-8:
                row.update(reproduced_target='IGNORE',ray_eligible=False);continue
            offset,fraction=np.linalg.solve(matrix,uv[a]-centers[i]);pos=centers[i]+offset*normals[i]
            row.update(offset_px=float(offset),fraction=float(fraction),position=pos.tolist(),normal=normals[i].tolist())
            if not (0<=fraction<=1 and abs(offset)<=32):
                row.update(reproduced_target='NONE',ray_eligible=False);continue
            za=(g['X'][a]@g['R'].T+g['t'])[2];zb=(g['X'][b]@g['R'].T+g['t'])[2]
            u=(fraction/zb)/((1-fraction)/za+fraction/zb);X=(1-u)*g['X'][a]+u*g['X'][b]
            row.update(ray_eligible=True,source_X=X.tolist(),source_camera_Z=float((X@g['R'].T+g['t'])[2]),
                supplied_visible_mask_support_3x3=support(visible,pos),in_image=bool(0<=pos[0]<g['hw'][1] and 0<=pos[1]<g['hw'][0]))
            sourceids.append(i);xs.append(X)
        if sourceids:
            xs32=np.asarray(xs,np.float32);cp=scene.compute_closest_points(o3d.core.Tensor(xs32));closest_count+=len(xs)
            cp_positions=cp['points'].numpy();dist=np.linalg.norm(cp_positions-xs32,axis=1);actual=dist<=1e-5*diagonal
            xy=np.asarray([rows[i]['position'] for i in sourceids]);ns=normals[sourceids]
            image_positions=np.stack([xy+shift*ns for shift in SHIFTS],axis=1)
            K=g['K'];direction=np.stack([(image_positions[...,0]-K[0,2])/K[0,0],(image_positions[...,1]-K[1,2])/K[1,1],np.ones(image_positions.shape[:2])],axis=-1)@g['R']
            origin=-g['R'].T@g['t'];rays=np.concatenate([np.broadcast_to(origin,direction.shape),direction],axis=-1).astype(np.float32)
            hit=scene.cast_rays(o3d.core.Tensor(rays.reshape(-1,6)));rays_count+=len(sourceids)*3
            depths=hit['t_hit'].numpy().reshape(-1,3);primitives=hit['primitive_ids'].numpy().reshape(-1,3)
            bary=hit['primitive_uvs'].numpy().reshape(-1,3,2);hn=hit['primitive_normals'].numpy().reshape(-1,3,3)
            cpp=cp['primitive_ids'].numpy()
            for j,i in enumerate(sourceids):
                row=rows[i];expected=row['source_camera_Z'];depthdiff=depths[j].astype(float)-expected
                close=np.isfinite(depths[j])&(np.abs(depthdiff)<=tol)
                row.update(actual_mesh_point_within_original_tolerance=bool(actual[j]),closest_distance_m=float(dist[j]),
                    closest_point_primitive_id=int(cpp[j]),closest_point=cp_positions[j].tolist(),
                    ray_depths_camera_Z=depths[j].tolist(),ray_depth_minus_source_Z_m=depthdiff.tolist(),
                    ray_primitive_ids=[int(x) if x!=4294967295 else None for x in primitives[j]],
                    ray_primitive_uvs=bary[j].tolist(),ray_primitive_normals=hn[j].tolist(),
                    ray_source_depth_agreement=close.tolist(),original_finite=bool(np.isfinite(depths[j,0])),
                    original_hit_in_front_of_source=bool(np.isfinite(depths[j,0]) and depthdiff[0]<-tol),
                    fixed_normal_offset_rescues_source_depth=bool(not close[0] and close[1:].any()),
                    mesh_and_mask_supported_original_miss_rescued=bool(actual[j] and row['in_image'] and row['supplied_visible_mask_support_3x3'] and not close[0] and close[1:].any()))
                row['reproduced_target']=('IGNORE' if not actual[j] or not row['in_image'] else
                    'POSITIVE' if close[0] and row['supplied_visible_mask_support_3x3'] else 'NONE')
                if row['reproduced_target']=='POSITIVE':
                    p=float((np.asarray(row['position'])-centers[i])@normals[i])+32;lower=int(np.clip(np.floor(p),0,64))
                    row.update(reproduced_lo=lower,reproduced_hi=min(64,lower+1),reproduced_weight=p-lower)
        edge_dist=S.physical_samples(scene,g['X']);edge_closest_count+=84;edge_physical=(edge_dist<1e-5*diagonal).any(1)
        for row in rows:
            row['whole_edge_has_physical_sample']=bool(edge_physical[row['edge']])
            if not edge_physical[row['edge']]:row['reproduced_target']='IGNORE'
            row['cached_label_reproduced']=bool(row['cached_target']==row['reproduced_target'])
        records.append(dict(id=source['id'],index=source['index'],partition='source_test',source_ceiling_semantic_sha256=C.digest(source),
            K=g['K'].tolist(),R=g['R'].tolist(),t=g['t'].tolist(),dimensions=g['dims'].tolist(),raw_hw=g['hw'],
            physical_diagonal_m=diagonal,depth_tolerance_m=tol,source_projection_error_px=g['projection_error_px'],
            physical_edge_sample_distances_m=edge_dist.tolist(),queries=rows,scene_build_index=frame+1,frame_rays=len(sourceids)*3))
        C.write(DOC/'SOURCE_RAY_PROGRESS.json',dict(stage='fixed_query_ray_audit',frames=frame+1,rays=rays_count,
            source_query_closest_points=closest_count,physical_edge_closest_points=edge_closest_count,
            protocol_sha256=C.sha(DOC/'SOURCE_RAY_PROTOCOL.json'),seconds=time.monotonic()-begin))
        if (frame+1)%16==0:print('SOURCE_RAY_VALIDATION',frame+1,128,rays_count,round(time.monotonic()-begin,2),flush=True)
    qq=[q for r in records for q in r['queries']];eligible=[q for q in qq if q['ray_eligible']]
    available=[q for q in eligible if q['actual_mesh_point_within_original_tolerance'] and q['in_image'] and q['supplied_visible_mask_support_3x3'] and q['whole_edge_has_physical_sample']]
    cachednone=[q for q in available if q['cached_target']=='NONE'];rescued=[q for q in cachednone if q['fixed_normal_offset_rescues_source_depth']]
    summary=dict(families=128,queries=len(qq),rays=rays_count,source_query_closest_points=closest_count,
        physical_edge_closest_points=edge_closest_count,scene_builds=128,cached_targets=dict(Counter(q['cached_target'] for q in qq)),
        reproduced_targets=dict(Counter(q['reproduced_target'] for q in qq)),label_mismatches=sum(not q['cached_label_reproduced'] for q in qq),
        label_mismatch_pairs=dict(Counter(q['cached_target']+'->'+q['reproduced_target'] for q in qq if not q['cached_label_reproduced'])),
        ray_eligible_queries=len(eligible),physical_in_frame_supplied_mask_supported=len(available),
        cached_NONE_physical_in_frame_mask_supported=len(cachednone),cached_NONE_fixed_offset_rescued=len(rescued),
        cached_NONE_rescued_by_predicted_role=dict(Counter(q['predicted_role'] for q in rescued)),
        cached_NONE_rescued_by_physical_edge=dict(Counter(q['edge'] for q in rescued)),
        cached_NONE_original_infinite=sum(not q['original_finite'] for q in cachednone),
        cached_NONE_original_front_surface=sum(q['original_hit_in_front_of_source'] for q in cachednone),
        cached_NONE_unrescued_original_front_surface=sum(q['original_hit_in_front_of_source'] for q in cachednone if not q['fixed_normal_offset_rescues_source_depth']),
        cached_NONE_unrescued_original_infinite=sum(not q['original_finite'] for q in cachednone if not q['fixed_normal_offset_rescues_source_depth']),
        original_positive_depth_error_m=stats([abs(q['ray_depth_minus_source_Z_m'][0]) for q in eligible if q['cached_target']=='POSITIVE']),
        rescued_offset_abs_depth_error_m=stats([min(abs(d) for d,close in zip(q['ray_depth_minus_source_Z_m'][1:],q['ray_source_depth_agreement'][1:]) if close) for q in rescued]))
    C.save_rows(DOC/'SOURCE_RAY_VALIDATION_ROWS.jsonl.gz',records)
    output=dict(schema='actual_mesh_exact_edge_float32_ray_sensitivity_audit_v1',summary=summary,
        interpretation='A fixed offset hit at source depth on the actual mesh demonstrates local first-surface sensitivity of the exact boundary ray. It is not a new label, complete visibility oracle or retrained result. Unrescued misses remain unresolved; front surfaces may indicate real occlusion.',
        source_target_labels_unchanged=True,original9000updates_unchanged=True,new_RGB=0,new_head_forwards=0,new_detector_forwards=0,new_PnP_calls=0,new_training_updates=0,
        additional_auxiliary_rays=rays_count,protocol=C.binding(DOC/'SOURCE_RAY_PROTOCOL.json'),rows=C.binding(DOC/'SOURCE_RAY_VALIDATION_ROWS.jsonl.gz'),
        audit_code=C.binding(Path(__file__)),wall_seconds=time.monotonic()-begin)
    C.write(DOC/'SOURCE_RAY_VALIDATION.json',output)
    import json
    print('SOURCE_RAY_VALIDATION_COMPLETE',json.dumps(C.finite(summary)),flush=True)


if __name__=='__main__':run()
