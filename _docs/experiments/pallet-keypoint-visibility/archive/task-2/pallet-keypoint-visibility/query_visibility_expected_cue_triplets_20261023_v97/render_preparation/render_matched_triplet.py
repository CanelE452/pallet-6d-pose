"""One matched photometric triplet; actual execution is root-entry gated.
RGB is separate from all cue/ownership outputs. Sealed V96 is read-only.
"""
import os,sys
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']='/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
os.environ['__GL_SHADER_DISK_CACHE']='0'
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,time
import numpy as np
from matched_render_math import camera_arrays,project,external_geometry
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
OLD=P/'query_visibility_query_anchored_cue_20261023_v96'
AP=OLD/'synthetic_appearance_feasibility';SOURCE_PROTOCOL=AP/'ONE_PROTOTYPE_PROTOCOL_V3_BEFORE_APPEARANCE.json'
SOURCE_PROTOCOL_SHA='2f657a682c5e42b5e94b855944c9873f402e754c5de52e105eb2fdd0bf754a36'
REFERENCES=OLD/'synthetic_query_cue_feasibility/SMALL_ASSOCIATION_MEASUREMENTS.json'
CROP_CODE=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
CROP_CODE_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
SCOPE=V/'ROOT_SCOPE_BEFORE_EXPECTED_CUE_RGB.json';PROTOCOL=H/'RENDER_PROTOCOL_BEFORE_ACTUAL_RGB.json';OUT=H/'actual_matched_rgb_triplet'
STAGES=('BEFORE','COVER','SHAM')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def jsonable(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return{k:jsonable(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return[jsonable(v)for v in x]
 if isinstance(x,np.generic):return x.item()
 return x

def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def crop_operator():
 assert sha(CROP_CODE)==CROP_CODE_SHA
 tree=ast.parse(CROP_CODE.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='crop')
 namespace={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(CROP_CODE),'exec'),namespace)
 return namespace['crop'],hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()

def make_protocol(cue_descriptor):
 """Only frozen metadata is read; no source geometry arrays or pixel render."""
 assert sha(SOURCE_PROTOCOL)==SOURCE_PROTOCOL_SHA
 prior=read(SOURCE_PROTOCOL);refs=read(REFERENCES);scope=read(SCOPE);cue=read(cue_descriptor)
 assert cue['completed']and cue['fixed_target_aux7']['aux_corner_index']==7
 contract=read(check(cue['contract']));assert contract['native8_mapping']=='UNKNOWN'and contract['human_gold']is False
 assert len(cue['references'])==8
 for old,new in zip(refs['references'],cue['references']):
  assert old['aux_corner_index']==new['aux_corner_index']and old['original_aux_q']==new['original_aux_q']and old['L']==new['L']
 assert cue['camera']==prior['camera']
 support=cue['fixed_target_aux7'];camera=prior['camera'];geom=external_geometry(camera,support['projected_support_disk_center_xy'],support['radius_px'],support['min_patch_camera_z'])
 crop,astsha=crop_operator();assert not OUT.exists()
 return dict(completed=True,source_only=True,scope=bind(SCOPE),code=bind(__file__),math=bind(H/'matched_render_math.py'),generated=bind(H/'GENERATED_RENDER_TESTS.json'),cue_descriptor=bind(cue_descriptor),cue_geometry=cue['cue_geometry'],cue_teacher_code=cue['teacher_code'],cue_contract=cue['contract'],previous_appearance_protocol=bind(SOURCE_PROTOCOL),previous_appearance_RGB=dict(path=str(AP/'ACTUAL_ONE_PROTOTYPE_V3/APPEARANCE_RGB.png'),sha256='69ba7c725bfd3d70b54cb3e976e41569636e861356af5dd5af42951c1cf74d59'),geometry=prior['geometry'],inventory=prior['inventory'],old_texture_camera_UV_code=prior['base_pure_camera_texture_UV_helper'],geometry_raster_code=prior['geometry_raster_helper'],crop_code=bind(CROP_CODE),crop_AST_sha256=astsha,
  camera=camera,references=[dict(aux_corner_index=r['aux_corner_index'],original_q=r['original_aux_q'],L=r['L'])for r in refs['references']],fixed_aux_index=7,external_geometry=jsonable(geom),
  target_material='Exact V96 V3 procedural wood/UV/float-white factor,metal0/roughness.88/emission0; no part-ID color or cue highlight',external_material='One shared procedural matte brown cardboard texture from seed20261023+104,metal0/roughness.92/emission0; same mesh/material in cover/sham, no ID palette',
  lighting='Exact V96 V3 ambient .22/.24/.27, warm directional1/.95/.86 intensity3.2 at camera pose; same neutralfloor. Shadows included; no changes between stages.',
  foreign_policy='Camera-aligned opaque cuboid. Front at.6*minimum localpatch cameraZ, thickness.05frontZ. Frontdiskbbox radius+3px. SHAM fixed+x by frontwidth+2px at identicaldepth,orientation,shape andmaterial; no search/retry. Placement uses GT cue geometry only for this source-control prototype.',
  ownership_policy='Independent V93 CPU triangle raster, no RGB color decoding. Original target nearest/secondpart depths once, same buffers reused for foreign cover/sham,64eps differentowner tiesU. Wholepart ownership is auxiliary only; B local endpoint face/edge witness teacher is the cue contract.',
  query_policy='All8 original source-emulated AUX virtualq/L exact; no snapping to physicalcue, no deployedinitializer/native8 claim. RGB96 exact pinned nativebilinear; categorical masks nearest only. Failed/partial/unresolved geometry staysexplicit in B teacher result, no alternatecue/object/camera.',
  output_directory=str(OUT),stages=list(STAGES),expected_photometric_passes=3,training_admission=False,native8_mapping='UNKNOWN',human_gold=False,new_ML_model_fit_current_PnP=0)

def authorize(entry):
 e=read(entry);assert e['root_approved_actual_rgb_triplet']is True and e['source_only']is True
 assert all(e[k]is False for k in('training_allowed','current_allowed','PnP_allowed','query_reselection_allowed'))
 for k in('code','protocol','generated','cue_descriptor','cue_contract','root_authorization'):check(e[k])
 assert e['code']==bind(__file__)and Path(e['protocol']['path'])==PROTOCOL and read(e['generated']['path'])['PASS']
 p=read(PROTOCOL);assert p['code']==e['code']and p['generated']==e['generated']and p['cue_descriptor']==e['cue_descriptor']and p['cue_contract']==e['cue_contract']
 for k in('geometry','inventory','old_texture_camera_UV_code','geometry_raster_code','crop_code','cue_geometry','cue_teacher_code','previous_appearance_protocol','previous_appearance_RGB','cue_contract'):check(p[k])
 assert p['native8_mapping']=='UNKNOWN'and p['training_admission']is False and not OUT.exists()
 return e,p

def cardboard_texture():
 rng=np.random.default_rng(20261127);u,v=np.meshgrid(np.arange(256),np.arange(256));noise=rng.normal(0,1.6,(256,256));fiber=1.7*np.sin(v*.47)+.8*np.sin(u*.71)
 rgb=np.array([151.,125.,89.])+noise[...,None]+fiber[...,None]*np.array([1.,.9,.7])
 return np.floor(np.clip(rgb,0,255)+.5).astype(np.uint8)

def raster_target(cpu,xyz,tri,pid,inv,camera):
 from matched_render_math import update_nearest_two,final_ownership
 size=camera['size'];best=np.full((size,size),np.inf);second=best.copy();owner=np.full(best.shape,-1,np.int16)
 vcam=cpu.world_to_camera(xyz,camera)
 for part in range(308):
  take=pid==part;tt=tri[take];info=inv['per_mesh'][part];ori=-1 if info['USD_orientation']=='leftHanded' else 1
  dd,ff,_=cpu.rasterize_parts(vcam,tt,np.zeros(len(tt),np.int32),np.flatnonzero(take).astype(np.int32),np.full(len(tt),info['double_sided'],bool),np.full(len(tt),ori,np.int8),camera,1)
  update_nearest_two(best,second,owner,dd[0],part);del dd,ff
 return best,second,owner

def run(entry):
 e,p=authorize(entry) # Sourcepixels/geometry/shader construction only below entry.
 import pyrender,trimesh
 from PIL import Image,ImageDraw,ImageFont
 from OpenGL import GL
 from matched_render_math import update_nearest_two,final_ownership
 start=time.monotonic();OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),code=bind(__file__),render_passes=0))
 old=module(p['old_texture_camera_UV_code']['path'],'v97_exact_old_wood_uv');cpu=module(p['geometry_raster_code']['path'],'v97_exact_cpu_raster')
 inv=read(p['inventory']['path']);cue_descriptor=read(p['cue_descriptor']['path']);cue_payload=read(p['cue_geometry']['path']);camera=p['camera'];cam={k:np.array(v,np.float64)if k in('K','R','t','eye_world','target_world','world_up','direction')else v for k,v in camera.items()}
 _,pose=old.camera(inv['stage']['bounds_world_m']);tex=old.wood_texture();size=int(cam['size']);scene=pyrender.Scene(ambient_light=[.22,.24,.27],bg_color=[.72,.75,.78,1.])
 mats={ds:pyrender.MetallicRoughnessMaterial(baseColorTexture=pyrender.Texture(source=tex,source_channels='RGB'),baseColorFactor=[1.,1.,1.,1.],metallicFactor=0.,roughnessFactor=.88,emissiveFactor=[0.,0.,0.],alphaMode='OPAQUE',doubleSided=ds)for ds in(False,True)}
 with np.load(p['geometry']['path'],allow_pickle=False)as z:xyz=z['vertices_world_m'];tri=z['triangles_global'];pid=z['triangle_asset_local_part_id']
 for part in range(308):
  info=inv['per_mesh'][part];scene.add(old.textured_part(xyz,tri[pid==part],part,info['USD_orientation'],mats[info['double_sided']]))
 bounds=np.asarray(inv['stage']['bounds_world_m']);center=bounds.mean(0);radius=cam['radius'];floor=trimesh.creation.box(extents=[radius*7,radius*.003,radius*7]);floor.apply_translation([center[0],bounds[0,1]-radius*.0016,center[2]])
 fm=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.47,.45,.41,1.],metallicFactor=0.,roughnessFactor=1.,emissiveFactor=[0.,0.,0.]);scene.add(pyrender.Mesh.from_trimesh(floor,material=fm,smooth=False))
 scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(50),aspectRatio=1.,znear=cam['near'],zfar=cam['distance']+radius*20),pose=pose);scene.add(pyrender.DirectionalLight(color=[1.,.95,.86],intensity=3.2),pose=pose)
 class ExactRGBRenderer(pyrender.OffscreenRenderer):
  def render(self,*args,**kwargs):
   self._platform.make_current();GL.glDisable(GL.GL_DITHER);return super().render(*args,**kwargs)
 renderer=ExactRGBRenderer(size,size);images=[];RGBdepth=[];node=None;g=p['external_geometry'];foreign_vertices={s:np.array(g['cover_vertices_world']if s=='COVER'else g['sham_vertices_world'],np.float64)for s in('COVER','SHAM')};ftri=np.array(g['triangles'],np.int64)
 cardboard=cardboard_texture();Image.fromarray(cardboard).save(OUT/'PROCEDURAL_CARDBOARD_TEXTURE.png')
 external=trimesh.Trimesh(vertices=foreign_vertices['COVER'],faces=ftri,process=False)
 # Face-expanded UVs, identical texture on every face; unrelated to GT/ID.
 external=external.copy();external.visual=trimesh.visual.texture.TextureVisuals(uv=np.stack((np.arange(8)%2,(np.arange(8)//2)%2),axis=1).astype(float))
 material=pyrender.MetallicRoughnessMaterial(baseColorTexture=pyrender.Texture(source=cardboard,source_channels='RGB'),baseColorFactor=[1.,1.,1.,1.],metallicFactor=0.,roughnessFactor=.92,emissiveFactor=[0.,0.,0.],alphaMode='OPAQUE',doubleSided=False)
 for stage in STAGES:
  if stage=='COVER':node=scene.add(pyrender.Mesh.from_trimesh(external,material=material,smooth=False))
  if stage=='SHAM':T=np.eye(4);T[:3,3]=np.array(g['sham_world_translation']);scene.set_pose(node,T)
  rgb,dep=renderer.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL);images.append(rgb);RGBdepth.append(dep);Image.fromarray(rgb).save(OUT/f'{stage}_RGB.png');print(stage+' RGB saved',flush=True)
 renderer.delete()
 crop,astsha=crop_operator();assert astsha==p['crop_AST_sha256'];q=np.array([r['original_q']for r in p['references']],np.float64);L=np.array([r['L']for r in p['references']],np.float64);crops=[];valids=[]
 for si,rgb in enumerate(images):
  cc=[];vv=[]
  for k in range(8):
   a,bv=crop(rgb,q[k],L[k],96);cc.append(a);vv.append(bv);Image.fromarray(a.transpose(1,2,0)).save(OUT/f'{STAGES[si]}_AUX{k}_RGB96_UNMARKED.png')
  crops.append(cc);valids.append(vv)
 np.savez_compressed(OUT/'MATCHED_RGB96_INPUT_ONLY.npz',rgb96=np.array(crops,np.uint8),crop_valid96=np.array(valids,bool),query8_xy=q,L=L,stage=np.array(STAGES))
 # Owner/depth buffers never enter RGB or modelinputs.
 best,second,owner0=raster_target(cpu,xyz,tri,pid,inv,cam);target_digest=hashlib.sha256(best.tobytes()+second.tobytes()+owner0.tobytes()).hexdigest();own=[];depth=[];ambiguity=[]
 for stage in STAGES:
  a=best.copy();b=second.copy();o=owner0.copy()
  if stage!='BEFORE':
   fv=cpu.world_to_camera(foreign_vertices[stage],cam);d,ff,_=cpu.rasterize_parts(fv,ftri,np.zeros(12,np.int32),np.arange(12,dtype=np.int32),np.zeros(12,bool),np.ones(12,np.int8),cam,1);update_nearest_two(a,b,o,d[0],308)
  ids,amb=final_ownership(a,b,o);own.append(ids);depth.append(a);ambiguity.append(amb)
 assert target_digest==hashlib.sha256(best.tobytes()+second.tobytes()+owner0.tobytes()).hexdigest()
 np.savez_compressed(OUT/'AUX_OWNERSHIP_GEOMETRY_ONLY.npz',part_id=np.array(own,np.int16),ambiguity=np.array(ambiguity,bool),depth=np.array(depth,np.float64),RGB_depth=np.array(RGBdepth,np.float32),camera_K=cam['K'],camera_R=cam['R'],camera_t=cam['t'],foreign_cover_vertices_world=foreign_vertices['COVER'],foreign_sham_vertices_world=foreign_vertices['SHAM'],foreign_triangles=ftri)
 # Local endpoint witness teacher, separate from wholepart ownership.
 teacher=module(p['cue_teacher_code']['path'],'v97_frozen_endpoint_teacher');orientation=np.array([1 if x['USD_orientation']=='rightHanded'else-1 for x in inv['per_mesh']])[pid];ds=np.array([x['double_sided']for x in inv['per_mesh']])[pid]
 target_scene=teacher.RayScene(xyz[tri],pid,orientation,ds);states={}
 for stage in STAGES:
  combined=None if stage=='BEFORE'else teacher.RayScene(np.concatenate((xyz[tri],foreign_vertices[stage][ftri])),np.r_[pid,np.full(12,308,np.int32)],np.r_[orientation,np.ones(12)],np.r_[ds,np.zeros(12,bool)])
  states[stage]=[teacher.evaluate_endpoint(c,cam,target_scene,combined)for c in cue_payload['references']]
 write(OUT/'EXPECTED_LOCAL_CUE_GEOMETRY_GT_ONLY.json',dict(completed=True,stages=states,wholepart_mask_used_for_cue_state=False,geometry_teacher_only=True,training_admission=False,native8_mapping='UNKNOWN',human_gold=False))
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
 contact=Image.new('RGB',(1500,580),'#edf1f5');dr=ImageDraw.Draw(contact)
 for j,stage in enumerate(STAGES):
  dr.text((j*500+12,10),stage+' | photometric RGB, no ID colors',font=font,fill='#182537');contact.paste(Image.fromarray(images[j]).resize((480,480),Image.Resampling.LANCZOS),(j*500+10,45))
 dr.text((12,540),'One fixed object, fixed AUX7 local cue, all8 q/L unchanged | AUX/native8 mapping UNKNOWN; not training admission',font=small,fill='#182537');contact.save(OUT/'MATCHED_FULL_RGB_CONTACT.png')
 cc=Image.new('RGB',(760,8*235+60),'#edf1f5');draw=ImageDraw.Draw(cc)
 for k in range(8):
  draw.text((10,k*235+8),'AUX'+str(k)+' q/L fixed; localcue geometry only',font=small,fill='#182537')
  for j,stage in enumerate(STAGES):
   a=Image.fromarray(np.array(crops[j][k]).transpose(1,2,0)).resize((192,192),Image.Resampling.NEAREST);cc.paste(a,(j*250+15,k*235+34));draw.text((j*250+15,k*235+215),stage+' '+states[stage][k]['state'],font=small,fill='#182537')
 draw.text((10,8*235+10),'Raw crops unmarked; AUX indices are not deployed native8. No learned model/current/fit/PnP.',font=small,fill='#182537');cc.save(OUT/'ALL8_UNCHANGED_QUERY_RGB96_CONTACT.png')
 previous=np.array(Image.open(p['previous_appearance_RGB']['path']).convert('RGB'),copy=True)
 result=dict(completed=True,PASS_execution=True,protocol=bind(PROTOCOL),entry=bind(entry),appearance_RGB_passes=3,one_camera_one_material_light=True,all8original_qL_unchanged=True,expected_local_cue_geometry_used=True,wholepart_mask_substitution=False,previous_BEFORE_RGB_exact=bool(np.array_equal(images[0],previous)),previous_BEFORE_diff_channel_N=int(np.sum(images[0]!=previous)),unchanged_target_depth_layers_sha256=target_digest,states={stage:[r['state']for r in states[stage]]for stage in STAGES},fixed_aux_index=7,target_aux7_states={stage:states[stage][7]['state']for stage in STAGES},native8_mapping='UNKNOWN',human_gold=False,training_admission=False,elapsed_seconds=time.monotonic()-start,limits=['Purposeful onecamera sourcecontrol, no blind benchmark or human semantic gold.','Externalobject placement is cue-geometry-guided and may float; procedural appearance has limited realism.','Finite localface/edge/endpoint witnesses define this AUX teacher; they do not certify every continuous visual cue or deployed nativepart.','SHAM sameobject may alter shadows or other image evidence; no automatic causal or training admission.'],outputs=[bind(f)for f in sorted(OUT.iterdir())if f.is_file()],new_learned_model_fit_private_current_PnP=0)
 write(OUT/'RESULT.json',result);write(OUT/'RENDER_SEAL.json',dict(completed=True,PASS_execution=True,result=bind(OUT/'RESULT.json'),protocol=bind(PROTOCOL),files=[bind(f)for f in sorted(OUT.iterdir())if f.is_file()],training_admission=False,new_current_fit_PnP=0));print(json.dumps(result),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--freeze',type=Path);ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.freeze:write(PROTOCOL,make_protocol(a.freeze));print(json.dumps(bind(PROTOCOL)))
 elif a.entry:run(a.entry)
 else:raise SystemExit('Choose metadata-only --freeze CUE_DESCRIPTOR or root-approved --entry')
