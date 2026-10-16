# Renderer inherited from unseen_dimensions_v1; only scene seed and requested count change.
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys
from common import *
OUT=OUT/'mean_risk_followup'
sys.path.insert(0,str(KIT.parent/'pallet_camera_trust_v26'))
import render26 as R
import numpy as np, cv2, pyrender, trimesh

def camera(yaw,height,distance,targetz):
    a=np.deg2rad(yaw);pos=np.array([distance*np.sin(a),-distance*np.cos(a),height])
    f=np.array([0,0,targetz])-pos;f/=np.linalg.norm(f);r=np.cross(f,[0,0,1]);r/=np.linalg.norm(r)
    T=np.eye(4);T[:3,:3]=np.column_stack((r,np.cross(r,f),-f));T[:3,3]=pos;return T

def main():
    cfg=register();assert not (OUT/'RENDER_COMPLETE.json').exists()
    renderer=pyrender.OffscreenRenderer(640,480);rows=[];assets=[];audit=[]
    folder=OUT/'images';folder.mkdir(exist_ok=True)
    for ai,name in enumerate(ASSETS):
        original,meta=R.load_mesh(ROOT/'data/pallet/raw_data/models_usd'/name);assets.append(meta)
        splits=['train','val','test'] if ai<2 else ['test']
        for split in splits:
            count=cfg[{'train':'train_groups_per_seen_asset','val':'val_groups_per_seen_asset','test':'test_groups_per_asset'}[split]]
            for group in range(count):
                # Match parameter distributions across assets, distinct group seeds across splits.
                rng=np.random.default_rng(851000+10000*['train','val','test'].index(split)+group)
                mesh=original.copy();mesh.vertices*=rng.uniform([.75,.75,.7],[1.25,1.25,1.3])
                color=[*rng.uniform(.12,.65,3),1]
                material=pyrender.MetallicRoughnessMaterial(baseColorFactor=color,roughnessFactor=.8,doubleSided=True)
                pm=pyrender.Mesh.from_trimesh(mesh,material=material,smooth=False)
                scene=pyrender.Scene(bg_color=[*rng.uniform(.3,.8,3),1],ambient_light=rng.uniform(.2,.5,3));scene.add(pm)
                floor=trimesh.creation.box(extents=[20,20,.02]);floor.apply_translation([0,0,-.02])
                scene.add(pyrender.Mesh.from_trimesh(floor,material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[*rng.uniform(.2,.6,3),1])))
                lp=np.eye(4);lp[:3,3]=[rng.uniform(-3,3),-3,4];scene.add(pyrender.PointLight(color=np.ones(3),intensity=float(rng.uniform(25,65))),pose=lp)
                cam=scene.add(pyrender.IntrinsicsCamera(fx=615,fy=615,cx=320,cy=240,znear=.01,zfar=30))
                for view,yaw0 in enumerate([-45,0,45,90]):
                    yaw=yaw0+rng.uniform(-7,7)+(180 if group%2 else 0)
                    T=camera(yaw,float(rng.uniform(.15,1.1)),float(rng.uniform(2.1,3.1)),mesh.extents[2]/2)
                    scene.set_pose(cam,T);world,uv,z=R.annotations(mesh,T)
                    isolated=pyrender.Scene();isolated.add(pm)
                    isolated.add(pyrender.IntrinsicsCamera(fx=615,fy=615,cx=320,cy=240,znear=.01,zfar=30),pose=T)
                    object_depth=renderer.render(isolated,flags=pyrender.RenderFlags.DEPTH_ONLY)
                    center=mesh.bounds.mean(0);X=world-center
                    rotation=(T[:3,:3]@np.diag([1,-1,-1])).T;translation=rotation@(center-T[:3,3])
                    q=(X@rotation.T+translation)@R.K.T
                    audit.append(float(np.linalg.norm(q[:,:2]/q[:,2:]-uv,axis=-1).max()))
                    panel=trimesh.creation.box(extents=[rng.uniform(.25,.55),.04,rng.uniform(.15,.3)])
                    a=np.deg2rad(yaw);rot=trimesh.transformations.rotation_matrix(a,[0,0,1]);panel.apply_transform(rot)
                    front=np.array([np.sin(a),-np.cos(a)])*(np.linalg.norm(mesh.extents[:2])/2+.12)
                    front+=np.array([np.cos(a),np.sin(a)])*rng.uniform(-.2,.2)
                    panel.apply_translation([*front,.14])
                    panelnode=None;clean=None
                    for state in cfg['conditions']:
                        if state=='panel':panelnode=scene.add(pyrender.Mesh.from_trimesh(panel,material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.5,.4,.3,1])))
                        if state!='crop25':rgb,depth=renderer.render(scene)
                        if state=='panel':scene.remove_node(panelnode)
                        if state=='clean':clean=rgb.copy()
                        hidden=float(((object_depth>0)&(depth<object_depth-.005)&(depth>0)).sum()/max((object_depth>0).sum(),1)) if state=='panel' else 0.
                        # Fixed image crop, never placed from GT or predicted box.
                        left=160 if state=='crop25' and group%2==0 else 0
                        right=480 if state=='crop25' and group%2 else 640
                        if state=='crop25':rgb=clean[:,left:right]
                        K=R.K.copy();K[0,2]-=left;points=uv-np.array([left,0])
                        fid=f'a{ai}_{split}_g{group:03d}_v{view}_{state}';path=folder/(fid+'.png')
                        cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                        h,w=rgb.shape[:2];inside=(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
                        rows.append(dict(id=fid,asset=name,asset_index=ai,split=split,seen=ai<2,
                            group=f'{split}_g{group:03d}',asset_group=f'a{ai}_{split}_g{group:03d}',view=view,state=state,
                            image=str(path.relative_to(ROOT)),sha256=sha(path),raw_hw=[h,w],K=K.tolist(),dims_xyz=mesh.extents.tolist(),
                            gt=points.tolist(),Xcf=X.tolist(),R=rotation.tolist(),t=translation.tolist(),inside=inside.tolist(),
                            object_occluded_fraction=hidden,
                            box=[float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]))
                if (group+1)%8==0:print('RENDER',name,split,group+1,'rows',len(rows),flush=True)
    renderer.delete();assert max(audit)<1e-8
    assert len(rows)==576,len(rows)
    write(OUT/'RENDER_COMPLETE.json',dict(rows=rows,assets=assets,gt_projection_max_px=max(audit),count=len(rows)))
    print('RENDER COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
