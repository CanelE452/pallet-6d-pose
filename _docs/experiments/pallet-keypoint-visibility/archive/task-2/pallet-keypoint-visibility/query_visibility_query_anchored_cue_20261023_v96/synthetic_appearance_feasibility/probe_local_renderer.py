"""Existing-library generated geometry only. No source geometry/model/current."""
import os
os.environ['PYOPENGL_PLATFORM']='egl'
from pathlib import Path
import json,time,traceback
H=Path(__file__).resolve().parent
start=time.monotonic();result={'source_geometry_read':False,'learned_model_current_fit_PnP':0,'new_download_or_install':False,'runtime':__import__('sys').executable,'platform':'egl'}
try:
 import numpy as np,pyrender,trimesh
 from OpenGL import GL
 scene=pyrender.Scene(ambient_light=[.3,.3,.3],bg_color=[.75,.8,.85,1.])
 node=scene.add(pyrender.Mesh.from_trimesh(trimesh.creation.box(extents=[1,1,1]),material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.5,.3,.1,1],metallicFactor=0,roughnessFactor=.9),smooth=False))
 pose=np.eye(4);pose[2,3]=3;scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(50)),pose=pose);scene.add(pyrender.DirectionalLight(color=[1,1,1],intensity=3),pose=pose)
 renderer=pyrender.OffscreenRenderer(64,64);rgb,depth=renderer.render(scene);mask,_=renderer.render(scene,flags=pyrender.RenderFlags.SEG,seg_node_map={node:[3,2,1]})
 assert rgb.shape==(64,64,3)and depth.shape==(64,64)and(depth>0).any();assert np.array_equal(mask[depth>0][0],[3,2,1])
 result.update(PASS=True,pyrender=pyrender.__version__,trimesh=trimesh.__version__,GL_vendor=GL.glGetString(GL.GL_VENDOR).decode(),GL_renderer=GL.glGetString(GL.GL_RENDERER).decode(),GL_version=GL.glGetString(GL.GL_VERSION).decode(),generated_foreground_pixels=int((depth>0).sum()),segmentation_separate_pass=True)
 renderer.delete()
except BaseException:result.update(PASS=False,error=traceback.format_exc())
result['seconds']=time.monotonic()-start
(H/'GENERATED_EGL_CAPABILITY_PROBE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
