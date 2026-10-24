"""One procedural appearance prototype from actual immutable scene3 geometry.
RGB never uses part IDs/palette. Separate SEG passes are auxiliary ownership only.
"""
import os
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']='/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
from pathlib import Path
import hashlib,importlib.util,json,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import pyrender,trimesh
from OpenGL import GL
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
G=P/'query_visibility_synthetic_part_teacher_20261023_v93/geometry_inventory'
MATH=P/'query_visibility_synthetic_part_teacher_20261023_v93/raster_preparation/cpu_part_raster.py'
OUT=H/'ACTUAL_ONE_PROTOTYPE_V2';SIZE=640;SEED=20261023

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,o):
 with Path(p).open('x')as f:json.dump(o,f,indent=2,allow_nan=False);f.write('\n')
def camera(bounds):
 s=importlib.util.spec_from_file_location('v93_camera_geometry_only',MATH);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 # Fixed overall-bound three-quarter view, chosen before appearance raster.
 c=m.camera_from_bounds(bounds,[1.25,.95,1.4],size=SIZE,fov_degrees=50)
 pose=np.eye(4);pose[:3,:3]=c['R'].T@np.diag([1,-1,-1]);pose[:3,3]=c['eye_world']
 return c,pose

def wood_texture():
 rng=np.random.default_rng(SEED);u,v=np.meshgrid(np.linspace(0,1,1024),np.linspace(0,1,512))
 warp=.016*np.sin(2*np.pi*u*2)+.005*np.sin(2*np.pi*u*6+.7)+.003*np.sin(2*np.pi*u*17+2*v)
 grain=np.sin(2*np.pi*(v+warp)*82)+.35*np.sin(2*np.pi*(v+.4*warp)*181)
 broad=np.sin(2*np.pi*v*4+1.6*np.sin(2*np.pi*u*.7))
 noise=rng.normal(0,1,u.shape)
 scalar=8*grain+8*broad+2.7*noise
 # Same nonsemantic procedural texture on every part, no part-ID random seed.
 base=np.array([173.,136.,94.]);rgb=base+scalar[...,None]*np.array([1.,.9,.7])
 for cu,cv in [(0.27,.33),(.73,.78)]:
  radial=np.sqrt(((u-cu)/.09)**2+((v-cv)/.032)**2)
  knot=np.exp(-radial*radial*.45)*(20+8*np.sin(radial*16));rgb-=knot[...,None]*np.array([1.,.85,.65])
 return np.floor(np.clip(rgb,0,255)+.5).astype(np.uint8)

def material(double_sided,texture):
 return pyrender.MetallicRoughnessMaterial(name='ProceduralWood_TrainPrototypeOnly',baseColorTexture=pyrender.Texture(source=texture,source_channels='RGB'),baseColorFactor=[1,1,1,1],metallicFactor=0.,roughnessFactor=.88,emissiveFactor=[0,0,0],alphaMode='OPAQUE',doubleSided=double_sided)

def textured_part(xyz,tri,part,orientation,mat):
 faces=tri.copy()
 if orientation=='leftHanded':faces=faces[:,[0,2,1]]
 verts=xyz[faces].reshape(-1,3);faces=np.arange(len(verts)).reshape(-1,3)
 bounds=np.stack([verts.min(0),verts.max(0)]);ext=np.maximum(bounds[1]-bounds[0],1e-9);long=int(np.argmax(ext))
 normal=np.cross(verts[faces[:,1]]-verts[faces[:,0]],verts[faces[:,2]]-verts[faces[:,0]])
 dominant=np.argmax(np.abs(normal),axis=1);uv=np.zeros((len(verts),2),np.float32)
 phase=float(np.mod(bounds[0]@np.array([.137,.173,.331]),1.))
 for j in range(3):
  ff=np.flatnonzero(dominant==j)
  if not len(ff):continue
  options=[a for a in range(3)if a!=long and a!=j]
  secondary=options[0]if options else int(np.argsort(ext)[-2])
  vi=faces[ff].reshape(-1)
  uv[vi,0]=((verts[vi,long]-bounds[0,long])/ext[long])*2.5+phase
  uv[vi,1]=(verts[vi,secondary]-bounds[0,secondary])/ext[secondary]+phase*.41
 mesh=trimesh.Trimesh(vertices=verts,faces=faces,process=False)
 mesh.visual=trimesh.visual.texture.TextureVisuals(uv=uv)
 return pyrender.Mesh.from_trimesh(mesh,material=mat,smooth=False)

class ExactIDRenderer(pyrender.OffscreenRenderer):
 def render(self,*args,**kwargs):
  self._platform.make_current();GL.glDisable(GL.GL_DITHER)
  return super().render(*args,**kwargs)

def run():
 inv=json.loads((G/'GEOMETRY_INVENTORY.json').read_text());ex=json.loads((G/'EXPORT_SEAL.json').read_text())
 npz=Path(ex['safe_NPZ']['path']);assert sha(npz)==ex['safe_NPZ']['sha256']and ex['completed']and ex['PASS']
 cam,pose=camera(inv['stage']['bounds_world_m']);assert not OUT.exists()
 protocol=dict(completed=True,code=b(__file__),geometry=b(npz),raw_USD=inv['source_asset'],geometry_inventory=b(G/'GEOMETRY_INVENTORY.json'),geometry_export=b(G/'EXPORT_SEAL.json'),camera_math_only=b(MATH),
  chosen_before_actual_appearance=True,one_RGB_prototype=True,size=[SIZE,SIZE],camera={k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in cam.items()},
  appearance='Uniform non-ID procedural anisotropic wood from seed20261023; geometry-long-axis UV, position phase, matte roughness .88/metal0/emission0. Original red/black/high-emission MDL shading is not used. Neutral matte floor and warm directional/ambient light. No public photo texture.',
  source_geometry='All308 original meshes/110478 triangles; declared metres and Y-up unchanged. No physical-scale calibration or geometry simplification. USD orientation/doubleSided respected.',
  textures_fold='TRAIN_PROTOTYPE_ONLY namespace/seeds; no VAL texture or scene generated. This single prototype is not a train/VAL corpus.',
  auxiliary='Separate SEG passes encode asset-local mesh IDs (ID+1 RGB bytes). RGB appearance shader never uses IDs. Part152 alone support/visible mask is geometry-only; native association/semantic role UNKNOWN, no human D/E/S labels. GPU depth-tie ambiguity is not independently certified.',
  infrastructure_retry='Original failed SEG-range attempt preserved under provenance/FAILED_SEGMENTATION_RANGE_V1; V2 uses same camera/geometry/appearance and explicitly disables GL_DITHER. Both attempts are one appearance specification, not independent samples or outcome selection.',
  runtime='Existing pallet-pose Python3.10,pyrender0.1.45,trimesh4.2.2,NVIDIA EGL pinned10_nvidia.json. No installations/downloads.',
  no_original_query_ref_public_pixels_current_model_fit_PnP=True,training_admission=False)
 write(H/'ONE_PROTOTYPE_PROTOCOL_V2_BEFORE_APPEARANCE.json',protocol)
 OUT.mkdir();assets=OUT/'procedural_assets/TRAIN_PROTOTYPE_ONLY';assets.mkdir(parents=True)
 tex=wood_texture();Image.fromarray(tex).save(assets/'procedural_wood_seed20261023.png')
 scene=pyrender.Scene(ambient_light=[.22,.24,.27],bg_color=[.72,.75,.78,1.]);nodes=[];meshes=[]
 with np.load(npz,allow_pickle=False)as z:
  xyz=z['vertices_world_m'];tri=z['triangles_global'];pid=z['triangle_asset_local_part_id'];assert len(tri)==110478 and len(xyz)==172656
  assert np.array_equal(np.unique(pid),np.arange(308));assert str(z['up_axis'])=='Y'and float(z['meters_per_stage_unit'])==1
  mats={ds:material(ds,tex)for ds in [False,True]}
  for part in range(308):
   info=inv['per_mesh'][part];mesh=textured_part(xyz,tri[pid==part],part,info['USD_orientation'],mats[info['double_sided']]);meshes.append(mesh);nodes.append(scene.add(mesh,name=f'asset_component_{part}'))
 bounds=np.asarray(inv['stage']['bounds_world_m']);center=bounds.mean(0);radius=cam['radius']
 floor=trimesh.creation.box(extents=[radius*7,radius*.003,radius*7]);floor.apply_translation([center[0],bounds[0,1]-radius*.0016,center[2]])
 floor_material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.47,.45,.41,1.],metallicFactor=0,roughnessFactor=1.,emissiveFactor=[0,0,0])
 floor_node=scene.add(pyrender.Mesh.from_trimesh(floor,material=floor_material,smooth=False),name='procedural_neutral_floor')
 camera_object=pyrender.PerspectiveCamera(yfov=np.deg2rad(50),aspectRatio=1.,znear=cam['near'],zfar=cam['distance']+radius*20)
 scene.add(camera_object,pose=pose,name='fixed_geometry_only_camera')
 lightpose=pose.copy();scene.add(pyrender.DirectionalLight(color=[1.,.95,.86],intensity=3.2),pose=lightpose,name='fixed_warm_key')
 renderer=ExactIDRenderer(SIZE,SIZE);started=time.monotonic()
 # Only this pass creates the appearance RGB. No selected-part coloring.
 rgb,depth=renderer.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
 Image.fromarray(rgb).save(OUT/'APPEARANCE_RGB.png')
 np.save(OUT/'APPEARANCE_DEPTH_F32.npy',depth.astype(np.float32),allow_pickle=False)
 seg_map={n:[(i+1)%256,(i+1)//256,0]for i,n in enumerate(nodes)};seg_map[floor_node]=[0,0,0]
 encoded,seg_depth=renderer.render(scene,flags=pyrender.RenderFlags.SEG,seg_node_map=seg_map)
 Image.fromarray(encoded).save(OUT/'AUX_RAW_ENCODED_PART_IDS_RGB.png')
 owner=encoded[:,:,0].astype(np.int32)+256*encoded[:,:,1].astype(np.int32)+65536*encoded[:,:,2].astype(np.int32)-1
 legal=(owner==-1)|((owner>=0)&(owner<308))
 vals,cnt=np.unique(encoded[~legal],axis=0,return_counts=True)
 write(OUT/'AUX_ENCODING_GUARD.json',dict(PASS=bool(legal.all()),invalid_pixels=int((~legal).sum()),invalid_colors=vals.tolist(),invalid_counts=cnt.tolist(),dither_disabled=True))
 assert legal.all(), 'saved encoded AUX invalid; RGB retained independently'
 alone=pyrender.Scene(bg_color=[0,0,0,1]);alone_node=alone.add(meshes[152]);alone.add(camera_object,pose=pose)
 alone_rgb,alone_depth=renderer.render(alone,flags=pyrender.RenderFlags.SEG,seg_node_map={alone_node:[153,0,0]})
 expected=np.all(alone_rgb==[153,0,0],axis=2);visible=owner==152
 # Save geometry support separately; no automatic semantic class or query target.
 Image.fromarray(rgb).save(OUT/'APPEARANCE_RGB.png');Image.fromarray((owner+1).astype(np.uint16)).save(OUT/'AUX_ASSET_LOCAL_PART_IDS16.png')
 Image.fromarray((visible*255).astype(np.uint8)).save(OUT/'AUX_PART152_VISIBLE_MASK.png');Image.fromarray((expected*255).astype(np.uint8)).save(OUT/'AUX_PART152_ALONE_SUPPORT.png')
 np.savez_compressed(OUT/'AUX_OWNERSHIP_AND_DEPTH.npz',asset_local_part_id=owner.astype(np.int16),target_depth=depth.astype(np.float32),ownership_depth=seg_depth.astype(np.float32),part152_expected_support=expected,part152_visible=visible,part152_alone_depth=alone_depth.astype(np.float32),camera_K=cam['K'],camera_R=cam['R'],camera_t=cam['t'])
 # A readable contact is diagnostic. Raw RGB remains separately unmarked.
 contact=Image.new('RGB',(1320,760),'#f0f2f5');d=ImageDraw.Draw(contact);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
 d.text((20,12),'ONE actual mesh prototype | procedural wood, no part-ID colors in RGB',font=font,fill='#182537');contact.paste(Image.fromarray(rgb),(20,55))
 d.text((690,62),'Separate auxiliary ownership masks',font=font,fill='#182537')
 for title,mask,y in [('Part152 alone support (not native gold)',expected,105),('Part152 visible in full target',visible,390)]:
  d.text((690,y),title,font=font,fill='#182537');contact.paste(Image.fromarray((mask*255).astype(np.uint8)).convert('RGB').resize((256,256),Image.Resampling.NEAREST),(690,y+28))
 d.text((20,710),'TRAIN_PROTOTYPE_ONLY | native mapping UNKNOWN | no query replacement, fitting, current data or PnP',font=font,fill='#182537');contact.save(OUT/'APPEARANCE_AND_SEPARATE_OWNERSHIP_CONTACT.png')
 stats=dict(completed=True,PASS=True,actual_appearance_RGB_passes=1,auxiliary_ownership_passes=2,geometry_meshes=308,geometry_triangles=110478,
  target_visible_pixels=int((owner>=0).sum()),visible_asset_components=int(len(np.unique(owner[owner>=0]))),part152_alone_pixels=int(expected.sum()),part152_visible_pixels=int(visible.sum()),part152_visible_outside_alone_pixel_count=int((visible&~expected).sum()),
  selected_role_and_native_mapping='UNKNOWN',no_human_native_gold=True,training_admission=False,gl_renderer='NVIDIA RTX3080 existing EGL',seconds=time.monotonic()-started,
  geometry_input_unchanged=(sha(npz)==ex['safe_NPZ']['sha256']),raw_USD_input_unchanged=(sha(inv['source_asset']['path'])==inv['source_asset']['sha256']),
  procedural_texture=b(assets/'procedural_wood_seed20261023.png'),protocol=b(H/'ONE_PROTOTYPE_PROTOCOL_V2_BEFORE_APPEARANCE.json'),outputs=[b(x)for x in sorted(OUT.glob('*'))if x.is_file()],
  limits=['Procedural matte wood is an explicit appearance override, not recovered original MDL or a photorealism guarantee.','All components use the same wood shader, so unverified hardware material identity is not certified.','GPU segmentation ownership boundaries/depth ties are not independently ambiguity-audited; masks are auxiliary prototype output, not admitted labels.','Single asset/camera/prototype; no corpus, native mapping, transfer, semanticD/E/S, or training admission.'],
  public_texture_downloads_installs_model_training_current_queries_PnP=0)
 renderer.delete();write(OUT/'PROTOTYPE_RESULT.json',stats);write(OUT/'PROTOTYPE_SEAL.json',dict(completed=True,PASS=True,result=b(OUT/'PROTOTYPE_RESULT.json'),protocol=b(H/'ONE_PROTOTYPE_PROTOCOL_V2_BEFORE_APPEARANCE.json'),files=[b(x)for x in sorted(OUT.rglob('*'))if x.is_file()],source_only=True,native_mapping='UNKNOWN',training_admission=False,new_model_current_fit_PnP=0))
 print(json.dumps(stats),flush=True)
if __name__=='__main__':run()
