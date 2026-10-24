import os
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']='/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
import pyrender,trimesh,numpy as np,json
from OpenGL import GL
from pathlib import Path
H=Path(__file__).resolve().parent
class ExactIDRenderer(pyrender.OffscreenRenderer):
 def render(self,*a,**k):
  self._platform.make_current();GL.glDisable(GL.GL_DITHER)
  return super().render(*a,**k)
def run(cls):
 scene=pyrender.Scene(bg_color=[.72,.75,.78,1],ambient_light=[.5,.5,.5]);nodes=[]
 for i in range(308):
  m=trimesh.creation.box(extents=[.9,.9,.01]);m.apply_translation([(i%22)-10.5,(i//22)-6.5,0]);nodes.append(scene.add(pyrender.Mesh.from_trimesh(m)))
 pose=np.eye(4);pose[2,3]=20;scene.add(pyrender.OrthographicCamera(xmag=11.5,ymag=7.5,znear=.1,zfar=100),pose=pose)
 rr=cls(660,420);rr.render(scene)
 enc,dep=rr.render(scene,flags=pyrender.RenderFlags.SEG,seg_node_map={n:[(i+1)%256,(i+1)//256,0]for i,n in enumerate(nodes)})
 vals,cnt=np.unique(enc.reshape(-1,3),axis=0,return_counts=True);ids=vals[:,0].astype(int)+256*vals[:,1].astype(int)+65536*vals[:,2].astype(int)-1
 bad=(ids<-1)|(ids>=308)
 output=dict(renderer=cls.__name__,colors=len(vals),bad_colors=vals[bad].tolist(),bad_color_counts=cnt[bad].tolist(),IDs_present=len(set(ids[~bad].tolist())-{-1}))
 rr.delete();return output
if __name__=='__main__':
 results=[run(pyrender.OffscreenRenderer),run(ExactIDRenderer)]
 p=H/'GENERATED_SEGMENTATION_ENCODING_PROBE.json';p.write_text(json.dumps(dict(generated_only=True,results=results),indent=2)+'\n');print(json.dumps(results))
