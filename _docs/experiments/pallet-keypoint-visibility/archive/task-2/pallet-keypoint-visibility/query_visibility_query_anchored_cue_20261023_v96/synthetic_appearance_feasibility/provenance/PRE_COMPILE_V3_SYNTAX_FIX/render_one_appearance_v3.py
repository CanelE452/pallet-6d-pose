"""Actual source appearance prototype; procedural texture, separate geometry masks.
V1/V2 failed artifacts remain immutable. No native points, labels or models.
"""
import os
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']='/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
from pathlib import Path
import json,time,hashlib,importlib.util
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import pyrender,trimesh
from OpenGL import GL
from appearance_v3_math import update_nearest_two,final_ownership,generated
H=Path(__file__).resolve().parent;P=H.parent.parent
G=P/'query_visibility_synthetic_part_teacher_20261023_v93/geometry_inventory'
MATH=P/'query_visibility_synthetic_part_teacher_20261023_v93/raster_preparation/cpu_part_raster.py'
BASE=H/'render_one_appearance.py';OUT=H/'ACTUAL_ONE_PROTOTYPE_V3';SIZE=640

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,o):
 with Path(p).open('x')as f:json.dump(o,f,indent=2,allow_nan=False);f.write('\n')
def load_code(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class ExactIDRenderer(pyrender.OffscreenRenderer):
 def render(self,*a,**k):
  self._platform.make_current();GL.glDisable(GL.GL_DITHER);return super().render(*a,**k)

def run():
 start=time.monotonic();assert not OUT.exists()
 old=load_code('v96_pure_wood_camera_uv',BASE);cpu=load_code('v93_geometry_raster_only',MATH)
 inv=json.loads((G/'GEOMETRY_INVENTORY.json').read_text());ex=json.loads((G/'EXPORT_SEAL.json').read_text())
 npz=Path(ex['safe_NPZ']['path']);assert sha(npz)==ex['safe_NPZ']['sha256']and ex['completed']and ex['PASS']
 cam,pose=old.camera(inv['stage']['bounds_world_m']);tex=old.wood_texture()
 proof=generated();write(H/'GENERATED_APPEARANCE_V3_MATH_TESTS.json',dict(**proof,code=b(__file__),math=b(H/'appearance_v3_math.py')))
 protocol=dict(completed=True,code=b(__file__),math=b(H/'appearance_v3_math.py'),base_pure_camera_texture_UV_helper=b(BASE),geometry_raster_helper=b(MATH),geometry=b(npz),inventory=b(G/'GEOMETRY_INVENTORY.json'),export=b(G/'EXPORT_SEAL.json'),raw_USD=inv['source_asset'],generated=b(H/'GENERATED_APPEARANCE_V3_MATH_TESTS.json'),
  scope='One actual RGB appearance specification from original 308-part mesh; auxiliary asset-local ownership, no training admission. No queries, native association, public pixels, models, fit, current or PnP.',
  revision='V2 preserved: near-black result traced to pyrender integer baseColorFactor [1,1,1,1] normalized to1/255. V3 corrects to explicit float [1.,1.,1.,1.]; camera/geometry/texture/light/floor unchanged. Not a camera/material search or additional corpus sample.',
  RGB_material='Same 1024x512 procedural wood uint8 RGB allparts, no ID-based albedo/seed; long-axis UV with position phase; float white material factor; metal0/roughness.88/emission0; authored MDL not used.',
  illumination=dict(ambient=[.22,.24,.27],directional_color=[1.,.95,.86],directional_intensity=3.2,light_pose='fixed camera pose',floor='neutral matte geometry below asset'),
  camera={k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in cam.items()},
  ownership='Independent V93 CPU integer-center reciprocal-depth raster at SAME camera; one part at a time, original winding/doubleSided, global nearest and second distinct-part minimum.64eps tie -> U(-2), background-1. Original scale unchanged. GPU SEG invalid byte colors remain U(-2), no rounding/snapping; all code-valid SEG owner boundaries dilated1px also marked U for comparison. CPU masks remain auxiliary, not admitted labels.',
  pixel_convention='640x640, integer arraypixel centers0..639, principal319.5; pyrender symmetric FOV50 projection has same center rays. RGB multisampling differs from categorical CPU center samples at edges; disagreements reported.',
  selected_asset_local_part=152,semantic_role='UNKNOWN',native8_mapping='UNKNOWN',training_admission=False,textures_fold='TRAIN_PROTOTYPE_ONLY procedural namespace, no VAL asset/view/corpus',new_downloads_installs_model_current_fit_PnP=0)
 protocolpath=H/'ONE_PROTOTYPE_PROTOCOL_V3_BEFORE_APPEARANCE.json';write(protocolpath,protocol)
 OUT.mkdir();assets=OUT/'procedural_assets/TRAIN_PROTOTYPE_ONLY';assets.mkdir(parents=True);Image.fromarray(tex).save(assets/'procedural_wood_seed20261023.png')
 mats={ds:pyrender.MetallicRoughnessMaterial(name='non_ID_procedural_wood',baseColorTexture=pyrender.Texture(source=tex,source_channels='RGB'),baseColorFactor=[1.,1.,1.,1.],metallicFactor=0.,roughnessFactor=.88,emissiveFactor=[0.,0.,0.],alphaMode='OPAQUE',doubleSided=ds)for ds in [False,True]}
 assert np.array_equal(mats[False].baseColorFactor,np.ones(4,np.float32))
 scene=pyrender.Scene(ambient_light=[.22,.24,.27],bg_color=[.72,.75,.78,1.]);nodes=[];meshes=[]
 with np.load(npz,allow_pickle=False)as z:
  xyz=z['vertices_world_m'];tri=z['triangles_global'];pid=z['triangle_asset_local_part_id'];assert len(tri)==110478 and len(xyz)==172656
  assert np.array_equal(np.unique(pid),np.arange(308));assert str(z['up_axis'])=='Y'and float(z['meters_per_stage_unit'])==1
  for part in range(308):
   info=inv['per_mesh'][part];mesh=old.textured_part(xyz,tri[pid==part],part,info['USD_orientation'],mats[info['double_sided']]);meshes.append(mesh);nodes.append(scene.add(mesh,name=f'asset_component_{part}'))
 bounds=np.asarray(inv['stage']['bounds_world_m']);center=bounds.mean(0);radius=cam['radius']
 floor=trimesh.creation.box(extents=[radius*7,radius*.003,radius*7]);floor.apply_translation([center[0],bounds[0,1]-radius*.0016,center[2]])
 floor_material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.47,.45,.41,1.],metallicFactor=0.,roughnessFactor=1.,emissiveFactor=[0.,0.,0.])
 floor_node=scene.add(pyrender.Mesh.from_trimesh(floor,material=floor_material,smooth=False),name='procedural_neutral_floor')
 camera_object=pyrender.PerspectiveCamera(yfov=np.deg2rad(50),aspectRatio=1.,znear=cam['near'],zfar=cam['distance']+radius*20)
 scene.add(camera_object,pose=pose);scene.add(pyrender.DirectionalLight(color=[1.,.95,.86],intensity=3.2),pose=pose)
 rr=ExactIDRenderer(SIZE,SIZE)
 rgb,depth=rr.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
 Image.fromarray(rgb).save(OUT/'APPEARANCE_RGB.png');print('Actual natural RGB saved',flush=True)
 segmap={n:[(i+1)%256,(i+1)//256,0]for i,n in enumerate(nodes)};segmap[floor_node]=[0,0,0]
 encoded,segdepth=rr.render(scene,flags=pyrender.RenderFlags.SEG,seg_node_map=segmap);Image.fromarray(encoded).save(OUT/'AUX_GPU_RAW_ENCODED_PART_IDS_RGB.png')
 gpu=encoded[:,:,0].astype(np.int32)+256*encoded[:,:,1].astype(np.int32)+65536*encoded[:,:,2].astype(np.int32)-1
 legal=(gpu==-1)|((gpu>=0)&(gpu<308));gpu_guarded=gpu.copy();gpu_guarded[~legal]=-2
 boundary=np.zeros(gpu.shape,bool);boundary[1:]|=gpu[1:]!=gpu[:-1];boundary[:-1]|=gpu[:-1]!=gpu[1:];boundary[:,1:]|=gpu[:,1:]!=gpu[:,:-1];boundary[:,:-1]|=gpu[:,:-1]!=gpu[:,1:]
 from scipy.ndimage import binary_dilation
 boundary=binary_dilation(boundary,iterations=1);gpu_guarded[boundary]=-2
 rr.delete()
 # Stream independent true geometry, not rendered ID colors. <=tens ofMB live buffers.
 vcam=cpu.world_to_camera(xyz,cam);best=np.full((SIZE,SIZE),np.inf);second=best.copy();owner_raw=np.full(best.shape,-1,np.int16);stats=[];expected=None
 for part in range(308):
  sel=pid==part;tpart=tri[sel];info=inv['per_mesh'][part];ori=-1 if info['USD_orientation']=='leftHanded'else1
  dd,ff,st=cpu.rasterize_parts(vcam,tpart,np.zeros(len(tpart),np.int32),np.flatnonzero(sel).astype(np.int32),np.full(len(tpart),info['double_sided'],bool),np.full(len(tpart),ori,np.int8),cam,1)
  update_nearest_two(best,second,owner_raw,dd[0],part);stats.append(st)
  if part==152:expected=np.isfinite(dd[0]).copy();alone_depth=np.where(expected,dd[0],np.nan).astype(np.float32)
  del dd,ff
  if part%50==0:print('Geometry ownership parts',part,flush=True)
 owner,amb=final_ownership(best,second,owner_raw);assert expected is not None
 visible=owner==152;assert not np.any(visible&~expected)
 comparable=(gpu_guarded!=-2)&~amb
 disagreement=comparable&(owner!=gpu_guarded)
 foreground=owner_raw>=0
 masked_rgb=rgb[foreground]
 np.savez_compressed(OUT/'AUX_GEOMETRY_OWNERSHIP_AND_DEPTH.npz',asset_local_part_id=owner,nearest_part_id_raw=owner_raw,ownership_ambiguity=amb,target_depth=np.where(np.isfinite(best),best,np.nan).astype(np.float32),second_distinct_part_depth=np.where(np.isfinite(second),second,np.nan).astype(np.float32),part152_alone_support=expected,part152_visible=visible,part152_alone_depth=alone_depth,gpu_encoded_owner_guarded=gpu_guarded.astype(np.int16),gpu_encoding_legal=legal,gpu_boundary_U=boundary,gpu_CPU_comparable=comparable,gpu_CPU_owner_disagreement=disagreement,RGB_renderer_depth=depth.astype(np.float32),GPU_seg_depth=segdepth.astype(np.float32),camera_K=cam['K'],camera_R=cam['R'],camera_t=cam['t'])
 Image.fromarray((expected*255).astype(np.uint8)).save(OUT/'AUX_PART152_ALONE_SUPPORT.png');Image.fromarray((visible*255).astype(np.uint8)).save(OUT/'AUX_PART152_VISIBLE_MASK.png');Image.fromarray((amb*255).astype(np.uint8)).save(OUT/'AUX_GEOMETRY_AMBIGUITY_U.png')
 # Diagnostic contact keeps RGB unmarked; ownership images are physically separate.
 c=Image.new('RGB',(1320,780),'#edf1f5');dr=ImageDraw.Draw(c);f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20);sm=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
 dr.text((20,12),'ACTUAL USD geometry + procedural wood | RGB has no ownership palette',font=f,fill='#182537');c.paste(Image.fromarray(rgb),(20,55))
 for title,mask,y in [('Separate part152 alone support',expected,85),('Separate part152 visible ownership',visible,390)]:
  dr.text((690,y),title,font=f,fill='#182537');c.paste(Image.fromarray((mask*255).astype(np.uint8)).convert('RGB').resize((256,256),Image.Resampling.NEAREST),(690,y+30))
 for line,y in [('Asset-local part ID is auxiliary; semantic role/native8 UNKNOWN. No gold labels or training admission.',714),('Same original geometry/scale; procedural material override. GPU edge/ID ambiguity retained as U.',744)]:dr.text((20,y),line,font=sm,fill='#182537')
 c.save(OUT/'APPEARANCE_AND_SEPARATE_OWNERSHIP_CONTACT.png')
 vals,cnt=np.unique(encoded[~legal],axis=0,return_counts=True)
 result=dict(completed=True,PASS=True,appearance_rendered=True,source_geometry_meshes=308,source_triangles=110478,prototype_dimensions=[SIZE,SIZE],
  appearance_fix=dict(texture_dtype=str(tex.dtype),texture_min=tex.reshape(-1,3).min(0).tolist(),texture_max=tex.reshape(-1,3).max(0).tolist(),texture_mean=tex.mean((0,1)).tolist(),material_baseColorFactor=mats[False].baseColorFactor.tolist(),V2_integer_factor=[1/255]*4,emission=[0.,0.,0.],metal0_roughness088=True),
  foreground_rgb_percentile_5_50_95=np.percentile(masked_rgb,[5,50,95],axis=0).tolist(),foreground_pixels=int(foreground.sum()),visible_asset_components=int(len(np.unique(owner[owner>=0]))),part152_alone_pixels=int(expected.sum()),part152_visible_pixels=int(visible.sum()),part152_visible_outside_alone_N=int((visible&~expected).sum()),geometry_ambiguous_U_pixels=int(amb.sum()),
  GPU_color_guard=dict(invalid_N=int((~legal).sum()),invalid_colors=vals.tolist(),counts=cnt.tolist(),invalid_policy='U(-2), never coerced',additional_boundary_U_N=int(boundary.sum()),comparable_N=int(comparable.sum()),CPU_owner_disagreement_N=int(disagreement.sum()),comparison_disagreement_policy='Retain separate arrays and limits; no mask snapping or RGB changes'),
  raster_stats=dict(input_triangles=sum(x['input_faces']for x in stats),near_straddling=sum(x['near_straddling_faces']for x in stats),zero_area=sum(x['zero_area_faces']for x in stats),culling='original USD orientation/doubleSided, same as appearance winding'),
  geometry_input_unchanged=sha(npz)==ex['safe_NPZ']['sha256'],raw_USD_input_unchanged=sha(inv['source_asset']['path'])==inv['source_asset']['sha256'],
  semantic_role_native_mapping='UNKNOWN',human_semantic_gold=False,training_admission=False,renderer='existing pyrender/NVIDIA EGL, no downloads/install',elapsed_seconds=time.monotonic()-start,
  limits=['One procedural wood appearance specification, not a photorealism or transfer result.','Authored MDL/shader and material identity are not recovered; wood override applies to every component, including possible hardware.','Geometric ownership is asset-local; GPU RGB MSAA and CPU integer-center coverage may differ at boundaries.','No native8/corner correspondence, human cue labels, query replacements, fitting, public/current inference or PnP.','Single asset/camera/light/texture namespace; not an independently split source corpus.'],outputs=[b(x)for x in sorted(OUT.rglob('*'))if x.is_file()],protocol=b(protocolpath),generated=b(H/'GENERATED_APPEARANCE_V3_MATH_TESTS.json'),new_model_fit_current_PnP=0)
 write(OUT/'PROTOTYPE_RESULT.json',result);write(OUT/'PROTOTYPE_SEAL.json',dict(completed=True,PASS=True,result=b(OUT/'PROTOTYPE_RESULT.json'),protocol=b(protocolpath),code=b(__file__),files=[b(x)for x in sorted(OUT.rglob('*'))if x.is_file()],native_mapping='UNKNOWN',training_admission=False))
 print(json.dumps(result),flush=True)
if __name__=='__main__':run()
