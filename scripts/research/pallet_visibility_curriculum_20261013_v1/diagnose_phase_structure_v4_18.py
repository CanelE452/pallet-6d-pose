"""Available mesh layouts and stored WD poses; no RGB/pose/label fitting.

USD meshes are transformed to their composed world coordinates and then
centered/reordered by measured bounding-box extent solely for illustration.
This is NOT a recovered asset-to-real physical registration. No structural
CAD is overlaid onto a real image under an assumed registration.
"""
from __future__ import annotations

import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')

from pathlib import Path
import json

import cv2
import numpy as np
import trimesh
from pxr import Usd,UsdGeom

from . import common as C


BOUNDARY=C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
OUTPUT=BOUNDARY/'phase_structure_evidence_v4_18'
OBSERVATIONS=BOUNDARY/'native_edge_observations_v4_17b/OBSERVATIONS.jsonl.gz'
OLD95=BOUNDARY/'global_outline_all95_eval_v4_10'
REMAINDER=C.DOC/'nonsevere_v1/remainder_after_test49_v1/pose_evaluation_v4_14'
CASES=('wood_night_01:031426','wood_night_01:031344','eval_noapril:1775201428535961600')
ASSETS=tuple(C.ROOT/'data/pallet/raw_data/models_usd'/name
    for name in ('scene.usd','scene_1.usd','scene_2.usd','scene_3.usd'))+(C.ROOT/'data/pallet/scan_cleanup/pallet_full.obj',)
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def _binding(path):
    return dict(path=str(Path(path).relative_to(C.ROOT)),sha256=C.sha(path))


def _mesh(path):
    """Composed USD transform matters: scene.usd actual thin axis is Z."""
    if path.suffix=='.obj':
        mesh=trimesh.load(str(path),force='mesh',process=False)
        vertices=np.asarray(mesh.vertices,float);faces=np.asarray(mesh.faces,int)
        metadata=dict(format='OBJ',mesh_count=1,unit='Export coordinate units; no new physical registration inferred')
    else:
        stage=Usd.Stage.Open(str(path));cache=UsdGeom.XformCache();vparts=[];fparts=[];offset=0;count=0
        for prim in stage.Traverse():
            mesh=UsdGeom.Mesh(prim)
            if not mesh:continue
            points=np.asarray(mesh.GetPointsAttr().Get(),float)
            if not len(points):continue
            transform=np.asarray(cache.GetLocalToWorldTransform(prim),float)
            world=(np.c_[points,np.ones(len(points))]@transform)[:,:3]
            indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int)
            sizes=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int);cursor=0;triangles=[]
            for size in sizes:
                face=indices[cursor:cursor+size];cursor+=size
                for j in range(1,len(face)-1):triangles.append((face[0],face[j],face[j+1]))
            vparts.append(world);fparts.append(np.asarray(triangles,int)+offset);offset+=len(world);count+=1
        vertices=np.vstack(vparts);faces=np.vstack(fparts)
        metadata=dict(format='USD',mesh_count=count,stage_up_axis=str(UsdGeom.GetStageUpAxis(stage)),
            stage_meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),composed_world_transforms_applied=True,
            stage_units_not_assumed_to_equal_training_randomized_physical_size=True)
    extent=np.ptp(vertices,axis=0);thin=int(np.argmin(extent));foot=[k for k in range(3)if k!=thin]
    short,long=sorted(foot,key=lambda k:extent[k]);order=[short,thin,long]
    display=(vertices-(vertices.min(0)+vertices.max(0))/2.)[:,order]/extent[long]
    permutation=np.eye(3)[:,order]
    if np.linalg.det(permutation)<0:display[:,0]*=-1
    metadata.update(source=_binding(path),vertices=len(vertices),triangles=len(faces),raw_world_extent=extent.tolist(),
        smallest_extent_axis='XYZ'[thin],display_raw_axis_order=['XYZ'[k]for k in order],
        display_only_longest_extent_normalization=True,asset_to_real_physical_registration_verified=False,
        training_asset_name_listed_in_existing_mesh_audits=path.name in ('scene.usd','scene_1.usd'),
        exact_training_mesh_membership_currently_verified=False,
        exact_real_object_design_match_verified=False)
    return trimesh.Trimesh(display,faces,process=False),metadata


def _camera(position,up):
    z=np.asarray(position,float);z/=np.linalg.norm(z);up=np.asarray(up,float)
    x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x)
    pose=np.eye(4);pose[:3,:3]=np.column_stack((x,y,z));pose[:3,3]=np.asarray(position,float)
    return pose


def _asset_views(mesh):
    import pyrender
    renderer=pyrender.OffscreenRenderer(420,340)
    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=(.62,.66,.7,1.),metallicFactor=0.,roughnessFactor=1.,doubleSided=True)
    scene=pyrender.Scene(bg_color=[.97,.98,1.,1.],ambient_light=[.55,.55,.55])
    scene.add(pyrender.Mesh.from_trimesh(mesh,material=material,smooth=False))
    views=[]
    for position,up in (([0.,2.,0.],[0.,0.,-1.]),([0.,.12,2.],[0.,1.,0.]),([2.,.12,0.],[0.,1.,0.])):
        pose=_camera(position,up)
        camera=scene.add(pyrender.OrthographicCamera(xmag=.65,ymag=.53),pose=pose)
        light=scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.),pose=pose)
        color,depth=renderer.render(scene)
        # Crop only the rendered illustration; retain its complete geometry.
        yy,xx=np.where(depth>0)
        if len(xx):color=color[max(0,yy.min()-10):min(len(color),yy.max()+11),max(0,xx.min()-10):min(color.shape[1],xx.max()+11)]
        views.append(color)
        scene.remove_node(camera);scene.remove_node(light)
    renderer.delete()
    return views


def _project(points,R,t,K):
    camera=np.asarray(points)@np.asarray(R).T+np.asarray(t)
    if np.any(camera[:,2]<=0):raise ValueError('Stored model projects behind the camera')
    pixels=camera@np.asarray(K).T
    return pixels[:,:2]/pixels[:,2:]


def _pair_hull_iou(first,second):
    a=cv2.convexHull(np.asarray(first,np.float32));b=cv2.convexHull(np.asarray(second,np.float32))
    intersection,_=cv2.intersectConvexConvex(a,b)
    return float(intersection/max(cv2.contourArea(a)+cv2.contourArea(b)-intersection,1e-12))


def _case_figure(row,observation,path):
    import matplotlib.pyplot as plt
    image_path=C.ROOT/row['image']
    if C.sha(image_path)!=row['image_sha256']:raise ValueError('Original RGB changed')
    rgb=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
    families=row['families'];projections=[];audits=[]
    for family in families:
        solution=family['solution']
        xy=_project(family['native_xyz'],solution['R'],solution['t'],observation['initial_geometry']['K'])
        projections.append(xy)
        Q=np.asarray(family['physical_registration_Q']);R=np.asarray(solution['R']);physical=R@Q
        if not np.allclose(physical,solution['R_physical'],atol=1e-10,rtol=0):raise ValueError('Physical/native pose basis differs')
        audits.append(dict(family=family['name'],accepted=solution['accepted'],R=solution['R'],t=solution['t'],
            R_physical=solution['R_physical'],native_xyz=family['native_xyz'],physical_registration_Q=family['physical_registration_Q'],
            projected_native_model_xy=xy.tolist(),pose_is_stored_old_G10_no_new_fit=True))
    allxy=np.vstack(projections);lo=np.maximum(allxy.min(0)-18,0);hi=np.minimum(allxy.max(0)+18,[rgb.shape[1],rgb.shape[0]])
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax in axes:
        ax.imshow(rgb);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_aspect('equal')
    axes[0].set_title('Actual original RGB crop\nNo contrast change / no GT markers')
    for ax,family,xy,color in zip(axes[1:],families,projections,('#00b9dc','#ff8c00')):
        for i,j in EDGES:ax.plot(xy[[i,j],0],xy[[i,j],1],color=color,lw=1.1)
        for measurement in observation['visible_measurements']:
            k=measurement['native_corner'];q=measurement['xy']
            ax.scatter([q[0]],[q[1]],marker='x',s=25,c='#e83d76',zorder=6)
            ax.annotate(str(k),q,xytext=(3,3),textcoords='offset points',fontsize=7,color='#ff296d')
        ax.set_title(family['name']+' stored cuboid model\nPink x: unchanged visible B/I observations')
    iou=_pair_hull_iou(*projections)
    fig.suptitle(row['id']+f' | projected cuboid-hull pair IoU={iou:.3f}\nNo candidate chosen by GT; internal CAD registration is unverified',fontsize=10)
    fig.savefig(path,dpi=160);plt.close(fig)
    return dict(id=row['id'],image=_binding(image_path),states=observation['states'],families=audits,
        projected_cuboid_hull_pair_IoU=iou,display_crop_xyxy=[*lo.tolist(),*hi.tolist()],figure=path.name,
        GT_pose_or_corner_reads=0,new_pose_fits=0,structural_CAD_projected_as_registered_model=False)


def main(output=OUTPUT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve all previous diagnostics')
    oldseal=C.read(OLD95/'PREDICTION_SEAL.json');rseal=C.read(REMAINDER/'PREDICTION_SEAL.json')
    if C.sha(OLD95/'PREDICTIONS.jsonl.gz')!=oldseal['predictions_sha256'] or C.sha(REMAINDER/'PREDICTIONS.jsonl.gz')!=rseal['predictions_sha256']:
        raise ValueError('Existing published prediction seals required')
    observations={r['id']:r for r in C.rows(OBSERVATIONS)if r['id']in CASES}
    cases={}
    for r in C.rows(OLD95/'PREDICTIONS.jsonl.gz'):
        if r['id']in CASES:cases[r['id']]=r
    for r in C.rows(REMAINDER/'PREDICTIONS.jsonl.gz'):
        if r['id']in CASES:cases[r['id']]=r['old_global_outline']
    if set(cases)!=set(CASES) or set(observations)!=set(CASES):raise ValueError('All three fixed diagnostic examples required')
    sources=[_binding(Path(__file__))]
    inputs=[_binding(p)for p in (*ASSETS,OBSERVATIONS,OLD95/'PREDICTION_SEAL.json',REMAINDER/'PREDICTION_SEAL.json',
        C.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json',
        C.ROOT/'data/pallet/raw_data/models_usd/_scale_report.txt',C.ROOT/'data/pallet/scan_cleanup/_rebuild_pallet_full.py')]
    output.mkdir(parents=True)
    C.write(output/'PROTOCOL.json',dict(sources=sources,inputs=inputs,cases=list(CASES),
        pose_source='Stored old G10 per-family published solutions; no GT selector/no new fit',
        mesh_pose_registration='Unverified; separate mesh illustrations only, never force a CAD into a real RGB image',
        scene_2_3_exact_training_GLB_membership_verified=False,new_NN_calls=0,new_pose_fits=0,new_GT_decodes=0))
    fig,axes=plt.subplots(len(ASSETS),3,figsize=(12,13),constrained_layout=True);metadata=[]
    for index,asset in enumerate(ASSETS):
        mesh,info=_mesh(asset);views=_asset_views(mesh);metadata.append(info)
        for column,(view,title)in enumerate(zip(views,('Top / footprint','Long-axis side','Short-axis side'))):
            axes[index,column].imshow(view);axes[index,column].axis('off')
            axes[index,column].set_title(asset.name+' | '+title,fontsize=9)
    fig.suptitle('Available model structure, dimension-normalized illustration only\nExact real design / canonical physical registration NOT verified; rows are different assets',fontsize=11)
    fig.savefig(output/'available_asset_layouts.png',dpi=160);plt.close(fig)
    results=[_case_figure(cases[ident],observations[ident],output/(ident.replace(':','__')+'.png'))for ident in CASES]
    report=dict(schema='PHASE_STRUCTURE_EVIDENCE_DIAGNOSTIC_V4_18',available_assets=metadata,cases=results,
        unavailable_training_GLB=['woodpallet_block_jtoastie_ccby.glb','eur_pallet_bk_cc0.glb'],
        actual_known_dimensions_are_cuboid_only=True,exact_real_wood_structure_mesh_verified=False,
        exact_real_plastic_scan_structure_requires_cleanup_and_registration_review=True,
        findings=[
            'A cuboid-only objective contains no deck-plank/grate/side-opening layout term, even when such structure is visible in RGB.',
            'This does not prove that rectangular cuboid silhouette can never resolve W/D; shape, perspective and pose uncertainty determine whether two fits overlap.',
            'A structurally verified object model could add a phase cue; generic training CAD shape and stored hypothesis Q are not independent evidence of the real instance phase.',
            'Two wood examples are small dark targets; apparent plank/slot lines require actual image support and uncertainty, not assumed CAD correspondence.',
            'No CAD mesh was assigned to a real frame or used to choose, optimize or score a pose.'],
        sources=sources,inputs=inputs,new_GT_decodes=0,new_pose_fits=0,new_NN_calls=0,new_training=0,
        default_pipeline_or_observations_changed=0)
    C.write(output/'DIAGNOSTIC.json',report)
    for binding in sources+inputs:
        if C.sha(C.ROOT/binding['path'])!=binding['sha256']:raise ValueError('Bound source/input changed')
    C.write(output/'DIAGNOSTIC_SEAL.json',dict(sources=sources,inputs=inputs,
        artifacts=[_binding(p)for p in output.iterdir()if p.is_file()]))
    print(json.dumps(dict(output=str(output),pair_IoU={r['id']:r['projected_cuboid_hull_pair_IoU']for r in results})))


if __name__=='__main__':
    main()
