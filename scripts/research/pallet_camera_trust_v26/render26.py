"""Actual CAD meshes rendered at controlled cameras with depth and object masks."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from common26 import *
import numpy as np
import cv2,trimesh,pyrender
from pxr import Usd,UsdGeom
import sys
sys.path.insert(0,str(ROOT))
from scripts.annotate.convert_to_camera_facing_v4 import compute_perm_v4
WIDTH,HEIGHT=640,480
K=np.array([[615.,0,320.],[0,615.,240.],[0,0,1.]])
YAWS=(-30,-10,0,10,30)
HEIGHTS=(.12,.4,.9)
COLORS=((.03,.35,.18,1),(.22,.24,.28,1),(.64,.39,.12,1))

def load_mesh(asset):
    stage=Usd.Stage.Open(str(asset));cache=UsdGeom.XformCache(Usd.TimeCode.Default());vertices=[];faces=[];offset=0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):continue
        mesh=UsdGeom.Mesh(prim);p=np.asarray(mesh.GetPointsAttr().Get(),float);T=np.asarray(cache.GetLocalToWorldTransform(prim));v=(np.c_[p,np.ones(len(p))]@T)[:,:3];counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());ix=np.asarray(mesh.GetFaceVertexIndicesAttr().Get());tri=[];start=0
        for n in counts:
            face=ix[start:start+n];tri.extend([[face[0],face[j],face[j+1]] for j in range(1,n-1)]);start+=n
        tri=np.asarray(tri,int)
        if np.linalg.det(T[:3,:3])<0:tri=tri[:,::-1]
        vertices.append(v);faces.append(tri+offset);offset+=len(v)
    v=np.concatenate(vertices);f=np.concatenate(faces);extent=np.ptp(v,axis=0);height_axis=int(extent.argmin());order=[a for a in range(3) if a!=height_axis]+[height_axis];A=np.eye(3)[order];v=v@A.T
    if np.linalg.det(A)<0:f=f[:,::-1]
    scale=1.2/max(np.ptp(v,axis=0)[:2]);v*=scale;v[:,:2]-=(v[:,:2].min(0)+v[:,:2].max(0))/2;v[:,2]-=v[:,2].min()
    mesh=trimesh.Trimesh(v,f,process=False);mesh.fix_normals(multibody=True)
    return mesh,dict(asset=str(asset.relative_to(ROOT)),asset_sha256=sha(asset),world_height_axis=height_axis,axis_order=order,uniform_scale=scale,dimensions_xyz=np.ptp(v,axis=0).tolist(),vertices=len(v),triangles=len(f))

def pose_camera(yaw,height):
    angle=np.deg2rad(yaw);position=np.array([2.6*np.sin(angle),-2.6*np.cos(angle),height]);target=np.array([0,0,.075]);forward=target-position;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward);T=np.eye(4);T[:3,:3]=np.column_stack((right,up,-forward));T[:3,3]=position
    return T

def project(vertices,T):
    camera=(vertices-T[:3,3])@T[:3,:3]@np.diag([1,-1,-1]);q=camera@K.T
    assert (camera[:,2]>0).all()
    return q[:,:2]/q[:,2,None],camera[:,2]

def annotations(mesh,T):
    mn,mx=mesh.bounds;world=np.array([[x,y,z] for z in (mn[2],mx[2]) for y in (mn[1],mx[1]) for x in (mn[0],mx[0])]);uv,depth=project(world,T);perm=compute_perm_v4(world,uv);assert perm is not None;perm=perm[:8];world=world[perm];uv=uv[perm];depth=depth[perm]
    assert np.allclose(world[[0,1,4,5],2],mx[2]) and np.allclose(world[[2,3,6,7],2],mn[2]);assert uv[0,0]<=uv[1,0] and uv[4,0]<=uv[5,0]
    return world,uv,depth

def register():
    if (RUN/'PROTOCOL.json').exists():return
    write(RUN/'PROTOCOL.json',dict(camera_yaw_deg=YAWS,camera_height_m=HEIGHTS,radial_distance_m=2.6,assets=['scene.usd','scene_1.usd'],appearance_contexts=3,clean_cameras=90,physical_occluder_states=2,rendered_RGB=180,image_conditions=['clean','crop_left20','crop_right20','physical_panel'],expected_observations=360,seeds=[1,2,3],new_training=False,
        hypothesis='Camera/observed-face state changes parent point residuals and line evidence; neither detection confidence nor geometrically good line alone guarantees a useful point correction.',
        geometry='USD world transforms applied to vertices, shortest measured axis becomes upright z, uniform scale max footprint1.2m. Existing camera-facing-v4 permutation, no GT matching in model inference. Dimensions are fixed object xyz with height last. Actual depth and isolated object depth support visibility; bounding cuboid edges are only geometric role references, not all physical mesh edges.',
        evaluation='Frozen YOLO/V18 and V25 models on controlled new renders; prior real297/9sessions reusedDEV and V23new128 for point/line reliability diagnosis. No real fitting. Primary models see no GT; oracle diagnostics separately named.',
        controls='At same asset/color/distance, paired yaw and camera height. Physical foreground panel changes actual visibility. Crops anchored to clean predicted box with fallback, not GT.',
        domain_limit='New OpenGL PBR raster renderer with flat asset materials, not identical Isaac background/texture distribution. Report detection misses and per-camera coverage; do not infer deployment accuracy from this grid.',
        budget='180raster renders plus derived360observations and3seed frozen inference. Existing-cache line trust/oracle decomposition. No model-training sweep.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (KIT/'render26.py',KIT/'common26.py',ROOT/'scripts/annotate/convert_to_camera_facing_v4.py')}))
    record('V26 사전 등록: 실제 USD2메시×3외관×5카메라 방위×3높이, 가림판2상태의180렌더 및360관측, 세 시드 고정모델 진단. 점/선 신뢰도와 oracle는 분리. 신규 학습 없음.')

def main(smoke=False):
    register();renderer=pyrender.OffscreenRenderer(WIDTH,HEIGHT);rows=[];assets=[]
    for ai,name in enumerate(('scene.usd','scene_1.usd')):
        mesh,meta=load_mesh(ROOT/'data/pallet/raw_data/models_usd'/name);assets.append(meta)
        for ci,color in enumerate(COLORS):
            scene=pyrender.Scene(bg_color=[.68,.72,.77,1],ambient_light=[.35,.35,.35]);material=pyrender.MetallicRoughnessMaterial(baseColorFactor=color,metallicFactor=.05,roughnessFactor=.8,doubleSided=True);node=scene.add(pyrender.Mesh.from_trimesh(mesh,material=material,smooth=False))
            ground=trimesh.creation.box(extents=[20,20,.025]);ground.apply_translation([0,0,-.03]);scene.add(pyrender.Mesh.from_trimesh(ground,material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.46,.44,.4,1],roughnessFactor=1)))
            lightpose=np.eye(4);lightpose[:3,3]=[2,-3,5];scene.add(pyrender.PointLight(color=np.ones(3),intensity=40),pose=lightpose)
            cam=scene.add(pyrender.IntrinsicsCamera(fx=K[0,0],fy=K[1,1],cx=K[0,2],cy=K[1,2],znear=.01,zfar=30))
            panel=trimesh.creation.box(extents=[.4,.06,.24]);panel.apply_translation([-.25,-mesh.extents[1]/2-.12,.12]);panelmesh=pyrender.Mesh.from_trimesh(panel,material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.6,.55,.42,1],roughnessFactor=1))
            for yaw in YAWS:
                for height in HEIGHTS:
                    if smoke and (ai,ci,yaw,height)!=(0,0,0,.4):continue
                    stem=f'a{ai}_c{ci}_yaw{yaw:+03d}_h{height:.2f}';T=pose_camera(yaw,height);scene.set_pose(cam,T);world,uv,depth=annotations(mesh,T)
                    isolated=pyrender.Scene();isolated.add(pyrender.Mesh.from_trimesh(mesh,material=material,smooth=False));isolated.add(pyrender.IntrinsicsCamera(fx=615,fy=615,cx=320,cy=240,znear=.01,zfar=30),pose=T);_,object_depth=renderer.render(isolated)
                    for state in ('clean','panel'):
                        if state=='panel':panelnode=scene.add(panelmesh)
                        rgb,z=renderer.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL);fid=stem+'_'+state;folder=RUN/('smoke' if smoke else 'renders');folder.mkdir(parents=True,exist_ok=True);image=folder/(fid+'.png');cv2.imwrite(str(image),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));visible=(object_depth>0)&(np.abs(z-object_depth)<.003);cv2.imwrite(str(folder/(fid+'_visible.png')),visible.astype(np.uint8)*255);np.savez_compressed(folder/(fid+'_depth.npz'),scene=z,object=object_depth)
                        row=dict(id=fid,source_group=f'a{ai}_c{ci}',asset=name,appearance=ci,yaw_deg=yaw,camera_height_m=height,state=state,image=str(image.relative_to(ROOT)),padded=False,K=K.tolist(),dims=mesh.extents.tolist(),session=f'a{ai}_c{ci}',camera_pose_opengl=T.tolist(),points_world=world.tolist(),points=uv.tolist(),point_depth=depth.tolist(),valid=[True]*8,image_sha256=sha(image),visible_pixels=int(visible.sum()),object_pixels=int((object_depth>0).sum()),visible_fraction=float(visible.sum()/max((object_depth>0).sum(),1)),depth_path=str((folder/(fid+'_depth.npz')).relative_to(ROOT)));rows.append(row)
                        if state=='panel':scene.remove_node(panelnode)
                    print('RENDER26',stem,flush=True)
    renderer.delete();write(RUN/('SMOKE.json' if smoke else 'RENDER_COMPLETE.json'),dict(complete=True,rows=rows,assets=assets,camera_contract='OpenGL camera columns right/up/back; projected with CV diag(1,-1,-1). Camera-facing IDs recomputed per camera from world cuboid, then frozen through crop/occlusion.'))
    if not smoke:assert len(rows)==180;record('카메라/물리 가림판 실제 렌더180개 완료. RGB·깊이·실제 물체 가시 마스크·카메라 외부파라미터·corner permutation 기록.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
