"""Actual RGB/mask/mesh audit and source-only preparation, never hull targets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import zipfile

import cv2
import numpy as np

from . import common as C

SOURCE = 'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
G38 = 'data/pallet/training_data/paper_release/v2_prod40k_clean_merged'
P0 = 'challenge/yolo_pose_one_model/datasets/_raw_legacy_v1v2_p0_10k'
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def selected_families():
    rows = C.read(C.ROOT/SOURCE)['records']
    pp = [r for r in rows if r['source']=='P0' and r['partition']=='train']
    pp.sort(key=lambda r: hashlib.sha256(('observation-source-20261009:'+r['scenario_id']).encode()).hexdigest())
    return [dict(r, observation_partition=('train' if i<768 else 'calibration' if i<896 else 'source_test'))
            for i,r in enumerate(pp[:1024])]

def locate(row):
    if row['source']=='G38':
        label = C.ROOT/row['renderer_annotation_locator_provenance_only']
    else:
        ident=row['id'].split('__',1)[1]; shard,frame=ident.rsplit('_',1)
        label=C.ROOT/P0/shard/'labels'/(frame+'_label.json')
    frame=label.name.removesuffix('_label.json')
    return dict(label=label, rgb=label.parent.parent/'rgb'/(frame+'_rgb.png'),
                visible=label.parent.parent/'mask_visible'/(frame+'.png'),
                amodal=label.parent.parent/'mask_amodal'/(frame+'.png'))

def mesh_normalized(asset='scene.usd'):
    """Read the actual USD triangles, including composed transforms."""
    cached=C.SCRATCH/'actual_mesh'/(asset+'.npz');meta=cached.with_suffix('.json')
    if cached.exists() and meta.exists():
        evidence=C.read(meta);assert C.sha(C.ROOT/evidence['asset']['path'])==evidence['asset']['sha256']
        with np.load(cached) as arrays:return arrays['vertices'],arrays['triangles'],evidence
    from pxr import Usd, UsdGeom
    path=C.ROOT/'data/pallet/raw_data/models_usd'/asset
    stage=Usd.Stage.Open(str(path)); cache=UsdGeom.XformCache(); vertices=[]; faces=[]; count=0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh): continue
        mesh=UsdGeom.Mesh(prim); v=np.asarray(mesh.GetPointsAttr().Get(),np.float64)
        matrix=np.asarray(cache.GetLocalToWorldTransform(prim))
        v=(np.column_stack([v,np.ones(len(v))])@matrix)[:,:3]
        ii=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()); cc=np.asarray(mesh.GetFaceVertexCountsAttr().Get()); start=0
        for n in cc:
            f=ii[start:start+n]; start+=n
            for k in range(1,n-1): faces.append([int(f[0])+count,int(f[k])+count,int(f[k+1])+count])
        vertices.append(v);count+=len(v)
    v=np.concatenate(vertices); lo=v.min(0);hi=v.max(0);v=(v-(lo+hi)/2)/(hi-lo)
    if asset=='scene.usd': v=v[:,[1,2,0]]*np.array([1.,-1.,-1.])
    else: v=v*np.array([1.,-1.,-1.])
    evidence=dict(asset=C.binding(path),vertices=count,triangles=len(faces),
        stage_meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),stage_up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        world_bbox_min=lo.tolist(),world_bbox_max=hi.tolist(),normalization='actual mesh bbox centered and scaled per axis',
        canonical_axis_map='[USD Y, -USD Z, -USD X]' if asset=='scene.usd' else '[USD X,-USD Y,-USD Z]')
    cached.parent.mkdir(parents=True,exist_ok=True);tri=np.asarray(faces,np.uint32)
    np.savez(cached,vertices=v,triangles=tri);C.write(meta,evidence)
    return v,tri,evidence

def ray_scene(vertices, triangles, dims):
    import open3d as o3d
    scene=o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor((vertices*np.asarray(dims)).astype(np.float32)),
        o3d.core.Tensor(triangles)))
    return scene

def render_mask(scene,K,R,t,hw):
    import open3d as o3d
    h,w=hw;y,x=np.mgrid[:h,:w]
    direction=np.stack([(x+.5-K[0,2])/K[0,0],(y+.5-K[1,2])/K[1,1],np.ones_like(x)],-1)@R
    origin=-R.T@t
    rays=np.concatenate([np.broadcast_to(origin,direction.shape),direction],-1).astype(np.float32)
    hit=scene.cast_rays(o3d.core.Tensor(rays))
    return np.isfinite(hit['t_hit'].numpy())

def geometry(row, ann=None):
    from .solver import cuboid,project
    ann=ann or C.read(locate(row)['label']);o=ann['objects'][0];intr=ann['camera_data']['intrinsics']
    K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
    T=np.array(o['pose_transform']);dim=o['dimensions_m'];dims=np.array([dim['width'],dim['height'],dim['depth']])
    mapped=frozenset(o['perm_v4'][:2]);width={frozenset(e) for e in [(0,1),(2,3),(4,5),(6,7)]}
    if mapped not in width: dims=dims[[2,1,0]]
    corners=cuboid(*dims);actual=np.array(o['projected_cuboid']);q=project(corners,T[:3,:3],T[:3,3],K)
    distance=np.linalg.norm(actual[:,None]-q[None],axis=-1);ids=distance.argmin(1)
    assert len(set(ids.tolist()))==8
    err=float(distance[np.arange(8),ids].max());assert err<.05
    return dict(K=K,R=T[:3,:3],t=T[:3,3],dims=dims,X=corners[ids],projection_error_px=err,
                asset=o['source_asset'],hw=[ann['camera_data']['height'],ann['camera_data']['width']])

def physical_samples(scene,X):
    import open3d as o3d
    q=np.asarray([[(1-u)*X[a]+u*X[b] for u in np.arange(1,8)/8] for a,b in EDGES],np.float32)
    cp=scene.compute_closest_points(o3d.core.Tensor(q.reshape(-1,3)))['points'].numpy()
    dist=np.linalg.norm(cp-q.reshape(-1,3),axis=-1).reshape(12,7)
    return dist

def audit_archived_tex(archive):
    """Inspect current archive bytes, without treating TEX as a controlled variant."""
    start=time.monotonic();rows=C.read(C.ROOT/SOURCE)['records'];tex={r['scenario_id']:r for r in rows if r['source']=='TEX'}
    records=[]
    with zipfile.ZipFile(archive) as z:
        for family in selected_families()[:32]:
            row=tex[family['scenario_id']];ident=row['id'].split('__',1)[1];shard,frame=ident.rsplit('_',1)
            members=dict(label=f'{shard}/labels/{frame}_label.json',rgb=f'{shard}/rgb/{frame}_rgb.png',
                visible=f'{shard}/mask_visible/{frame}.png',amodal=f'{shard}/mask_amodal/{frame}.png')
            raw={k:z.read(member) for k,member in members.items()};ann=json.loads(raw['label']);g=geometry(row,ann)
            decoded={k:cv2.imdecode(np.frombuffer(raw[k],np.uint8),cv2.IMREAD_COLOR if k=='rgb' else cv2.IMREAD_GRAYSCALE) for k in ['rgb','visible','amodal']}
            assert all(a is not None and list(a.shape[:2])==g['hw'] for a in decoded.values())
            prepared=cv2.imread(str(row['image']));pad=row['reflect_pad_px'];assert np.array_equal(prepared[pad:-pad,pad:-pad],decoded['rgb'])
            v,a=decoded['visible'],decoded['amodal'];p0=geometry(family)
            records.append(dict(id=row['id'],family=row['scenario_id'],hw=g['hw'],asset=g['asset'],
                evidence=[dict(archive=Path(archive).name,member=member,sha256=hashlib.sha256(raw[k]).hexdigest(),bytes=len(raw[k])) for k,member in members.items()],
                projection_error_px=g['projection_error_px'],visible_area=int((v>127).sum()),amodal_area=int((a>127).sum()),
                visible_outside_amodal=int(((v>127)&(a<=127)).sum()),rgb_matches_prepared_center=True,
                K=g['K'].tolist(),R=g['R'].tolist(),t=g['t'].tolist(),fixed_dims=g['dims'].tolist(),
                P0_same_family_fixed_dims=p0['dims'].tolist(),same_family_does_not_mean_controlled_appearance_only_variant=True))
    C.write(C.DOC/'TEX_ARCHIVE_SAMPLE_AUDIT.json',dict(schema='current_archived_source_samples_v1',frames=32,records=records,
        new_RGB=0,source_training_changed=False,additional_detector_or_head_calls=0,wall_seconds=time.monotonic()-start))
    print('ARCHIVED_TEX_AUDIT_COMPLETE',len(records),time.monotonic()-start,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--tex-archive',required=True);p.add_argument('--archived-only',action='store_true');args=p.parse_args()
    if args.archived_only:
        audit_archived_tex(args.tex_archive);return
    cv2.setNumThreads(1);start=time.monotonic();manifest=C.read(C.ROOT/SOURCE);selected=selected_families()
    bindings=[C.binding(C.ROOT/SOURCE)];sources={}
    for source,root in [('G38',C.ROOT/G38),('P0',C.ROOT/P0)]:
        files=[f for f in root.rglob('*') if f.is_file()]
        sources[source]=dict(files=len(files),suffix_counts=dict(Counter(f.suffix for f in files)),
            directories=dict(Counter(str(f.parent.relative_to(root)) for f in files)))
    archive=Path(args.tex_archive);z=zipfile.ZipFile(archive);members=z.infolist()
    sources['TEX']=dict(archive_filename=archive.name,bytes=archive.stat().st_size,
        files=len(members),suffix_counts=dict(Counter(Path(f.filename).suffix for f in members)),
        directories=dict(Counter('/'.join(f.filename.split('/')[:-1]) for f in members)))
    rows=[r for r in manifest['records'] if r['source']=='G38']
    rows.sort(key=lambda r:hashlib.sha256(('observation-extra-audit:'+r['id']).encode()).hexdigest())
    # The 128 P0 families are part of the locked 1,024 training pilot population.
    panel=selected[:128]+rows[:32];vertices,faces,meshinfo=mesh_normalized();records=[]
    for i,row in enumerate(panel):
        paths=locate(row);ann=C.read(paths['label']);g=geometry(row,ann);rgb=cv2.imread(str(paths['rgb']));v=cv2.imread(str(paths['visible']),0);a=cv2.imread(str(paths['amodal']),0)
        assert rgb is not None and v is not None and a is not None
        assert list(rgb.shape[:2])==g['hw']==list(v.shape)==list(a.shape)
        prepared=cv2.imread(str(row['image']));pad=row['reflect_pad_px']
        assert np.array_equal(prepared[pad:-pad,pad:-pad],rgb)
        b=[C.binding(path) for path in paths.values()];bindings.extend(b)
        record=dict(id=row['id'],family=row['scenario_id'],source=row['source'],asset=g['asset'],
            evidence=b,rgb_matches_prepared_center=True,hw=g['hw'],projection_error_px=g['projection_error_px'],
            visible_area=int((v>127).sum()),amodal_area=int((a>127).sum()),visible_outside_amodal=int(((v>127)&(a<=127)).sum()),
            mask_visible_value_counts={str(k):int(n) for k,n in zip(*np.unique(v,return_counts=True))},
            mask_amodal_value_counts={str(k):int(n) for k,n in zip(*np.unique(a,return_counts=True))},
            scene_seeds_available=bool(ann['objects'][0].get('scene_placement_v2',{}).get('stage_seeds')),
            exact_boundary_mapping=False)
        if g['asset']=='scene.usd':
            scene=ray_scene(vertices,faces,g['dims']);mask=render_mask(scene,g['K'],g['R'],g['t'],g['hw']);aa=a>127
            record.update(mesh_mask_iou=float((mask&aa).sum()/max(1,(mask|aa).sum())),mesh_mask_pixel_disagreement=int((mask!=aa).sum()))
            dist=physical_samples(scene,g['X']);record['actual_mesh_edge_sample_dist_m']=dist.tolist()
            record['physical_queries']=(dist<1e-5*np.linalg.norm(g['dims'])).tolist()
            record['exact_boundary_mapping']=record['mesh_mask_iou']>=.995
        records.append(record)
        if (i+1)%32==0:print('SOURCE_PANEL',i+1,len(panel),flush=True)
    exact=[r for r in records if r['source']=='P0'];assert len(exact)==128
    passed=all(r['exact_boundary_mapping'] and r['visible_outside_amodal']==0 for r in exact)
    families=[dict(id=r['id'],family=r['scenario_id'],partition=r['observation_partition'],source=r['source'],
        raw={k:str(v.relative_to(C.ROOT)) for k,v in locate(r).items()}) for r in selected]
    C.write(C.DOC/'SOURCE_FAMILY_SPLIT.json',dict(schema='new_source_family_split_v1',records=families,
        family_count=1024,counts=dict(Counter(r['partition'] for r in families)),
        all_variants_share_partition=True,existing_TEX_variants_not_separate_families=True,
        selection_uses_real_GT=False,selection_uses_source_scores=False,scene_family_split_hash=C.digest(families)))
    C.write(C.DOC/'SYNTH_SUPERVISION_AUDIT.json',dict(schema='new_actual_asset_audit_v1',
        inspected_source_records=len(manifest['records']),source_record_counts=dict(Counter(r['source'] for r in manifest['records'])),
        inventory=sources,panel_basic_families=128,additional_G38_audit_frames=32,panel=records,
        mesh=meshinfo,physical_corner_vertical_edges='IGNORED: rounded-corner cuboid vertical edges are not physical mesh boundaries',
        saved_object_masks=True,saved_depth=False,saved_normals=False,saved_face_ID=False,
        available_CPU_depth_and_primitive_ID_from_actual_scene_usd_mesh=True,
        missing_GLBS=['woodpallet_block_jtoastie_ccby.glb','eur_pallet_bk_cc0.glb'],
        renderer={'Blender_executable':False,'bpy':False,'isaacsim':False,'pxr':True,'open3d':True,
            'same_RGB_scene_reproduction':False,'reason':'Windows generator/background/GLBs not present; seeds do not serialize full scene'},
        exact_P0_existing_RGB_supervision_valid=passed,source_only_128_pass=passed,
        bindings=bindings,generated_RGB=0,raycast_auxiliary_masks=sum('mesh_mask_iou' in r for r in records),
        wall_seconds=time.monotonic()-start))
    C.write(C.DOC/'SOURCE_SUPERVISION_STATUS.json',dict(status='VALID_SOURCE_SUBSET' if passed else 'BLOCKED',
        usable_source='P0 actual scene.usd geometry + paired delivered masks',
        unavailable_other_assets_do_not_block_valid_source_subset=True,
        targets='physical edge intersections/depth visibility; unsupported vertical bounding edges IGNORE',
        preparation='existing original RGB reused; new RGB0 because exact supervision is available',
        four_controlled_variants='unrepresented in the selected existing P0 source; source coverage limitation',
        planned_fits=3,planned_updates_per_fit=3000,planned_batch=16,new_RGB_so_far=0,
        formal_updates_so_far=0,source_family_split=C.binding(C.DOC/'SOURCE_FAMILY_SPLIT.json')))
    print('SOURCE_AUDIT_COMPLETE',passed,time.monotonic()-start,flush=True)

if __name__=='__main__':main()
