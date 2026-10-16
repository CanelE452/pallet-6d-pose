import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_visible_geometry_v32'))
from context32 import *
from context44 import RUN, register
import render26 as R26
from geometry32 import creases,labels
import numpy as np,cv2,pyrender,trimesh

def camera(yaw,height,distance,targetz):
    a=np.deg2rad(yaw);pos=np.array([distance*np.sin(a),-distance*np.cos(a),height]);forward=np.array([0,0,targetz])-pos;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward);T=np.eye(4);T[:3,:3]=np.column_stack((right,up,-forward));T[:3,3]=pos;return T

def main():
    register()
    if (RUN/'RENDER_COMPLETE.json').exists():return
    renderer=pyrender.OffscreenRenderer(640,480);rows=[];assets=[]
    folder=RUN/'renders';folder.mkdir(exist_ok=True)
    for ai,name in enumerate(('scene.usd','scene_1.usd')):
        mesh,meta=R26.load_mesh(ROOT/'data/pallet/raw_data/models_usd'/name);segments,edge_meta=creases(mesh);meta.update(edge_meta);assets.append(meta)
        np.save(RUN/f'physical_segments_asset{ai}.npy',segments)
        for group in range(12):
            rng=np.random.default_rng(440000+ai*100+group);color=np.clip(np.array(R26.COLORS[group%3])+rng.uniform(-.06,.06,4),.015,.95);color[3]=1
            scene=pyrender.Scene(bg_color=[*rng.uniform(.35,.8,3),1],ambient_light=rng.uniform(.18,.45,3));mat=pyrender.MetallicRoughnessMaterial(baseColorFactor=color,metallicFactor=0.,roughnessFactor=float(rng.uniform(.45,.95)),doubleSided=True);pm=pyrender.Mesh.from_trimesh(mesh,material=mat,smooth=False);scene.add(pm)
            ground=trimesh.creation.box(extents=[20,20,.025]);ground.apply_translation([0,0,-.03]);scene.add(pyrender.Mesh.from_trimesh(ground,material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[*rng.uniform(.22,.6,3),1],roughnessFactor=1)))
            lightpose=np.eye(4);lightpose[:3,3]=[rng.uniform(-3,3),-3,4];scene.add(pyrender.PointLight(color=np.ones(3),intensity=float(rng.uniform(25,65))),pose=lightpose)
            cam=scene.add(pyrender.IntrinsicsCamera(fx=615,fy=615,cx=320,cy=240,znear=.01,zfar=30))
            for view,yaw in enumerate((-40,-20,0,20,40)):
                height=(.15,.35,.55,.8,1.05)[view]+rng.uniform(-.04,.04);distance=float(rng.uniform(2.,2.9));T=camera(yaw+rng.uniform(-4,4),height,distance,mesh.extents[2]/2);scene.set_pose(cam,T);world,uv,z=R26.annotations(mesh,T)
                isolated=pyrender.Scene();isolated.add(pm);isolated.add(pyrender.IntrinsicsCamera(fx=615,fy=615,cx=320,cy=240,znear=.01,zfar=30),pose=T);_,obj=renderer.render(isolated)
                panel=trimesh.creation.box(extents=[rng.uniform(.2,.5),.045,rng.uniform(.12,.26)]);panel.apply_translation([rng.uniform(-.35,.35),-mesh.extents[1]/2-.1,.12]);panelmesh=pyrender.Mesh.from_trimesh(panel,material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.5,.42,.3,1]))
                for state in ('clean','panel'):
                    fid=f'v44_a{ai}_g{group:02d}_v{view}_{state}';image=folder/(fid+'.png');labelpath=folder/(fid+'_geometry.json');depthpath=folder/(fid+'_depth.npz')
                    if state=='panel':node=scene.add(panelmesh)
                    rgb,depth=renderer.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
                    if state=='panel':scene.remove_node(node)
                    cv2.imwrite(str(image),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));np.savez_compressed(depthpath,object=obj,scene=depth)
                    lab=labels(segments,world,T,R26.K,obj,depth);write(labelpath,lab)
                    rows.append(dict(id=fid,source_group=f'v44_a{ai}_g{group:02d}',camera_group=f'v44_a{ai}_g{group:02d}_v{view}',split='train' if group<6 else ('calibration' if group<8 else 'check'),asset=name,state=state,image=str(image.relative_to(ROOT)),padded=False,K=R26.K.tolist(),dims=mesh.extents.tolist(),session=f'v44_a{ai}_g{group:02d}',camera_pose_opengl=T.tolist(),points_world=world.tolist(),points=uv.tolist(),valid=[True]*8,image_sha256=sha(image),label=str(labelpath.relative_to(ROOT)),label_sha256=sha(labelpath),depth_path=str(depthpath.relative_to(ROOT)),visible_samples=lab['physical_visible_samples'],visible_endpoints=lab['physical_visible_endpoints']))
                print('RENDER44',ai,group,view,'segments',len(segments),flush=True)
    renderer.delete();assert len(rows)==240;write(RUN/'RENDER_COMPLETE.json',dict(complete=True,rows=rows,assets=assets));print('RENDER44 COMPLETE',flush=True)

if __name__=='__main__':main()
