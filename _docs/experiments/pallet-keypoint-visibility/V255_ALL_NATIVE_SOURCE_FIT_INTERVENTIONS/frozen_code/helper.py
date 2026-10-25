"""Inert root-entry CPU renderer of six NEW registered synthetic scenes. No model/crops."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;POLICY=V/'ROOT_RENDER_POLICY_BEFORE_PIXELS.json'
POLICY_SHA='fc314702da18a0ff1e36e77fc20ee6db8de8d535b79b71a8473463d41179aaf6'
PROTOCOL=H/'RENDER_PROTOCOL_BEFORE_PIXELS.json';OUT=V/'actual_render_v1';VARIANTS=('BEFORE','COVER','SHAM')
LIGHT=np.array([-.35,-1.,-.5]);LIGHT/=np.linalg.norm(LIGHT)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def digest(*arrays):
 h=hashlib.sha256()
 for a in arrays:h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(np.ascontiguousarray(a).tobytes())
 return h.hexdigest()
def world_points(depth,camera):
 yy,xx=np.indices(depth.shape);z=np.where(np.isfinite(depth),depth,0.)
 cam=np.stack([(xx-camera['K'][0,2])*z/camera['K'][0,0],(yy-camera['K'][1,2])*z/camera['K'][1,1],z],-1)
 return (cam-camera['t'])@camera['R']
def shade(buffer,vertices,triangles,style,camera):
 """Linear procedural albedo/Lambert, then explicit sRGB; no part-ID palette."""
 xyz=vertices[triangles];normals=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);normals/=np.linalg.norm(normals,axis=1,keepdims=True)
 valid=np.isfinite(buffer['depth'])&(buffer['part_id']>=0);fid=np.maximum(buffer['face_id'],0);point=world_points(buffer['depth'],camera);normal=normals[fid]
 # Two-sided opaque shading orientation only; ownership/depth/ties never changed.
 normal=np.where((np.sum(normal*(camera['eye_world']-point),axis=-1)<0)[...,None],-normal,normal)
 grain=.89+.07*np.sin(95*point[...,2]+2*np.sin(9*point[...,0]))+.04*np.sin(410*point[...,2]+6*point[...,0])
 if style=='WOOD_BOARD_BLOCK':albedo=np.array([.60,.36,.16])*grain[...,None]
 elif style=='POLYMER_RIB_LEG':albedo=np.array([.055,.075,.095])*(.985+.015*np.sin(80*point[...,0])*np.sin(73*point[...,2]))[...,None]
 else:raise ValueError(style)
 albedo=np.where((buffer['part_id']==999)[...,None],np.array([.035,.16,.62]),albedo)
 lambert=.42+.58*np.maximum(0,np.sum(normal*LIGHT,axis=-1));linear=np.clip(albedo*lambert[...,None],0,1)
 srgb=np.where(linear<=.0031308,12.92*linear,1.055*np.power(linear,1/2.4)-.055);rgb=np.floor(255*np.clip(srgb,0,1)+.5).astype(np.uint8)
 rgb[~valid]=np.array([225,226,228],np.uint8);return rgb
def nearest_pixels(xy,size):
 finite=np.isfinite(xy).all(-1);ix=np.zeros(xy.shape,np.int32);ix[finite]=np.floor(xy[finite]+.5).astype(np.int32)
 inside=finite&(ix[...,0]>=0)&(ix[...,0]<size)&(ix[...,1]>=0)&(ix[...,1]<size);return ix,inside
def witness_diagnostics(data,camera,raster,teacher,buffer,vertices,triangles,owners,records):
 points=data['native_witness_points'];pc=raster.world_to_camera(points.reshape(-1,3),camera).reshape(points.shape);xy=raster.project_camera(pc.reshape(-1,3),camera).reshape(8,37,2);nearest,inside=nearest_pixels(xy,256)
 po=np.full((8,37),-1,np.int32);pf=po.copy();pd=np.full((8,37),np.nan,np.float64);pa=np.ones((8,37),bool);ro=po.copy();rf=po.copy();rd=pd.copy();ray_U=np.ones((8,37),bool)
 for i,j in np.ndindex(8,37):
  if inside[i,j]:
   x,y=nearest[i,j];po[i,j]=buffer['part_id'][y,x];pf[i,j]=buffer['face_id'][y,x];pd[i,j]=buffer['depth'][y,x];pa[i,j]=buffer['ambiguity'][y,x]
  hit=teacher.first_hit(camera['eye_world'],points[i,j],vertices[triangles],owners)
  if hit['status']=='HIT':
   ro[i,j]=hit['owner'];rf[i,j]=min(hit['face_ids']);direction=points[i,j]-camera['eye_world'];direction/=np.linalg.norm(direction);location=camera['eye_world']+direction*hit['distance'];rd[i,j]=raster.world_to_camera(location[None],camera)[0,2];ray_U[i,j]=False
 comparable=inside&~pa&~ray_U;agreement=po==ro;summary=[]
 patchxy=raster.project_camera(raster.world_to_camera(data['native_patch_quads'].reshape(-1,3),camera),camera).reshape(8,3,4,2)
 for i,r in enumerate(records):
  active=np.array(r['active_witness_indices']);mask=comparable[i];finite_depth=mask&np.isfinite(pd[i])&np.isfinite(rd[i]);d=pd[i]-rd[i];summary.append(dict(native_index=i,continuous_region_state=r['state'],all37_nearest_OOB=int((~inside[i]).sum()),nearest_ambiguous=int((inside[i]&pa[i]).sum()),nearest_background=int((inside[i]&(po[i]<0)).sum()),nearest_nonfinite_depth=int((inside[i]&~np.isfinite(pd[i])).sum()),continuous_ray_U=int(ray_U[i].sum()),certain_owner_comparable=int(mask.sum()),certain_owner_disagreements=int((mask&~agreement[i]).sum()),finite_depth_comparable=int(finite_depth.sum()),active_unique_nearest_pixels=len(set(map(tuple,nearest[i,active][inside[i,active]].tolist()))),projected_patch_spans_xy=(patchxy[i].max(1)-patchxy[i].min(1)).tolist(),max_abs_depth_difference_between_different_rays=float(np.max(np.abs(d[finite_depth])))if finite_depth.any()else None))
 arrays=dict(native8_xy_continuous=xy[:,0],witness_xy_continuous=xy,witness_nearest_xy=nearest,witness_nearest_in_image=inside,witness_nearest_owner=po,witness_nearest_triangle_id=pf,witness_nearest_depth=pd,witness_nearest_ambiguity=pa,witness_continuous_owner=ro,witness_continuous_triangle_id=rf,witness_continuous_depth=rd,witness_continuous_U=ray_U,witness_owner_agreement=agreement)
 return arrays,summary
def authorize(entry):
 e=read(entry);assert e['approved_source_render']is True and all(e[k]is False for k in('model_forward','fit','current223','V112','PnP','automatic_retry'))
 for k in('code','protocol','generated','policy'):check(e[k])
 assert sha(POLICY)==POLICY_SHA and e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['policy']==bind(POLICY)
 p=read(PROTOCOL);policy=read(POLICY);proof=read(e['generated']['path']);assert proof['PASS']and proof['code']==e['code']and p['generated']==e['generated']and p['code']==e['code']and p['policy']==e['policy']
 for k in('geometry_result','geometry_code','root_geometry_review','raster_code'):assert p[k]==policy[k];check(p[k])
 assert p['geometry_design']==read(p['geometry_result']['path'])['design'];check(p['geometry_design'])
 assert p['meshes']==policy['meshes']
 for mesh in p['meshes']:check(mesh)
 return e,p
def save_png(path,rgb):
 from PIL import Image
 Image.fromarray(rgb).save(path);return bind(path)
def displays(buffer,rgb,xy,records,depthrange):
 from PIL import Image,ImageDraw
 owner=buffer['part_id'];value=np.maximum(owner,0).astype(np.uint32);part=np.stack([(value*73+50)%200+30,(value*137+20)%200+30,(value*41+90)%200+30],-1).astype(np.uint8);part[owner<0]=245;part[buffer['ambiguity']]=[255,0,255]
 finite=np.isfinite(buffer['depth']);norm=np.where(finite,(buffer['depth']-depthrange[0])/(depthrange[1]-depthrange[0]),0);level=np.floor(np.clip(norm,0,1)*255+.5).astype(np.uint8);depth=np.repeat(level[...,None],3,-1);depth[~finite]=245;depth[buffer['ambiguity']]=[255,0,255]
 marked=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(marked);colors={'D':'lime','E':'red','S':'cyan','U':'orange'}
 for i,(x,y)in enumerate(xy):
  state=records[i]['state'];draw.ellipse((x-3,y-3,x+3,y+3),outline=colors[state],width=1);draw.text((x+4,y-7),str(i)+state,fill=colors[state],stroke_width=1,stroke_fill='black')
 return dict(PART_DISPLAY_ONLY=part,DEPTH_DISPLAY_ONLY=depth,NATIVE_DISPLAY_ONLY=np.asarray(marked))
def contact(images,labels,path,title):
 from PIL import Image,ImageDraw
 board=Image.new('RGB',(3*280,50+2*295),'white');draw=ImageDraw.Draw(board);draw.text((12,8),title,fill='black');draw.text((12,26),'NEW procedural assets; labels from continuous registered geometry; display only.',fill='black')
 for i,(rgb,label)in enumerate(zip(images,labels)):
  x=12+(i%3)*280;y=50+(i//3)*295;board.paste(Image.fromarray(rgb),(x,y+22));draw.text((x,y),label,fill='black')
 board.save(path);return bind(path)
def run(entry):
 e,p=authorize(entry);assert not OUT.exists(),'Preserve previous attempt; explicit new run needed.'
 assert np.__version__==p['runtime']['NumPy']and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
 raster=module(p['raster_code']['path'],'v128_v93_raster');teacher=module(p['geometry_code']['path'],'v128_registered_teacher');source=read(p['geometry_result']['path']);design=read(p['geometry_design']['path']);assert [r['mesh']for r in source['results']]==p['meshes']
 OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),raw_render_only=True,no_query_crops=True));start=time.monotonic();outputs=[];scene_rows=[];contacts={k:[]for k in('RGB','PART_DISPLAY_ONLY','DEPTH_DISPLAY_ONLY','NATIVE_DISPLAY_ONLY')};labels=[]
 try:
  for item in source['results']:
   style=item['style'];check(item['mesh'])
   with np.load(item['mesh']['path'],allow_pickle=False)as f:data={k:np.array(f[k],copy=True)for k in f.files}
   vertices=data['vertices'];tri=data['triangles'];pid=data['part_owner'];F=len(tri);camera=dict(size=256,K=data['camera_K'],R=data['camera_R'],t=data['camera_t'],eye_world=np.array(design['camera_eye'],np.float64),near=.001)
   assert np.array_equal(camera['K'][:2,2],[127.5,127.5])and np.allclose(camera['R']@camera['R'].T,np.eye(3),atol=1e-14)and np.linalg.det(camera['R'])>0
   vc=raster.world_to_camera(vertices,camera);part_count=len(item['parts']);d,f,stats=raster.rasterize_parts(vc,tri,pid,np.arange(F,dtype=np.int32),np.ones(F,bool),np.ones(F,np.int8),camera,part_count);layer_before=digest(d,f);base=raster.resolve_layers(d,f)
   layerfile=OUT/(style+'_TARGET_LAYERS.npz');np.savez_compressed(layerfile,part_depth=d,global_triangle_id=f,camera_K=camera['K'],camera_R=camera['R'],camera_t=camera['t']);outputs.append(bind(layerfile));before_rgb=None;before_buffer=None
   allvc=raster.world_to_camera(np.concatenate([vertices,data['foreign_cover_vertices'],data['foreign_sham_vertices']]),camera);depthrange=(float(allvc[:,2].min()),float(allvc[:,2].max()))
   for variant in VARIANTS:
    fv=None if variant=='BEFORE'else data['foreign_'+variant.lower()+'_vertices'];combined_vertices=vertices;combined_tri=tri;combined_owners=pid;foreign_stats=None
    if fv is None:buf={k:v.copy()for k,v in base.items()}
    else:
     fc=raster.world_to_camera(fv,camera);ft=data['foreign_triangles'];fd,ff,foreign_stats=raster.rasterize_parts(fc,ft,np.zeros(12,np.int32),F+np.arange(12,dtype=np.int32),np.ones(12,bool),np.ones(12,np.int8),camera,1);buf=raster.resolve_layers(d,f,foreign=(fd[0],ff[0],999));combined_vertices=np.concatenate([vertices,fv]);combined_tri=np.concatenate([tri,ft+len(vertices)]);combined_owners=np.r_[pid,np.full(12,999,np.int32)]
    assert digest(d,f)==layer_before
    records=teacher.region_teacher(data,camera,raster,fv);assert records==item['records'][variant],'Continuous teacher drift; never coerce from raster pixels.'
    rgb=shade(buf,combined_vertices,combined_tri,style,camera);unchanged=None
    if before_rgb is None:before_rgb=rgb.copy();before_buffer=buf
    else:
     unchanged=(buf['part_id']>=0)&(buf['part_id']!=999)&(buf['part_id']==before_buffer['part_id'])&(buf['face_id']==before_buffer['face_id'])&(buf['depth']==before_buffer['depth']);assert np.array_equal(rgb[unchanged],before_rgb[unchanged])
    diag,summary=witness_diagnostics(data,camera,raster,teacher,buf,combined_vertices,combined_tri,combined_owners,records);stem=style+'_'+variant;npz=OUT/(stem+'_BUFFERS.npz');np.savez_compressed(npz,RGB_uint8=rgb,camera_depth=buf['depth'],part_id=buf['part_id'],global_triangle_id=buf['face_id'],ambiguity=buf['ambiguity'],tie_count=buf['tie_part_count'],camera_K=camera['K'],camera_R=camera['R'],camera_t=camera['t'],continuous_region_states=np.array([r['state']for r in records]),**diag);outputs.append(bind(npz))
    pics=displays(buf,rgb,diag['native8_xy_continuous'],records,depthrange);pics['RGB']=rgb;imagefiles={}
    for key,pixels in pics.items():imagefiles[key]=save_png(OUT/(stem+'_'+key+'.png'),pixels);outputs.append(imagefiles[key]);contacts[key].append(pixels)
    label=style.replace('_BOARD_BLOCK','').replace('_RIB_LEG','')+' '+variant;labels.append(label);row=dict(style=style,variant=variant,buffers=bind(npz),images=imagefiles,triangle_namespace='Scene-global: target0..F-1,foreignF..F+11; owners are separate targetpart0..N-1/foreign999/background-1.',target_triangle_count=F,target_part_count=part_count,foreign_triangle_count=0 if fv is None else 12,target_layers=bind(layerfile),immutable_target_layer_digest_before=layer_before,immutable_target_layer_digest_after=digest(d,f),same_target_surface_RGB_exact=True,same_target_surface_pixels=int(unchanged.sum())if unchanged is not None else int((buf['part_id']>=0).sum()),ambiguous_pixels=int(buf['ambiguity'].sum()),foreign_pixels=int((buf['part_id']==999).sum()),target_raster_stats=stats,foreign_raster_stats=foreign_stats,continuous_region_records=records,nearest_pixel_diagnostic=summary);scene_rows.append(row)
   del d,f,data
  for key,pictures in contacts.items():outputs.append(contact(pictures,labels,OUT/(key+'_6_SCENES_CONTACT.png'),key+' / 256px integer-centre camera'))
  record=OUT/'SCENES6_AND_WITNESS_DIAGNOSTICS.json';write(record,dict(completed=True,scenes=scene_rows,nearest_pixel_does_not_relabel_continuous_teacher=True));outputs.append(bind(record))
  for k in('geometry_result','geometry_code','raster_code'):check(p[k])
  for mesh in p['meshes']:check(mesh)
  receipt=OUT/'RENDER_COMPLETE.json';write(receipt,dict(completed=True,PASS=True,entry=bind(entry),code=bind(__file__),protocol=bind(PROTOCOL),policy=e['policy'],scenes=6,size=256,outputs=outputs,target_rasters=2,foreign_rasters=4,teacher_exact_original=True,target_layers_immutable=True,same_target_RGB_bitexact=True,no_query_crops_created=True,RGB_no_part_ID_palette=True,nearest_pixel_aliasing_reported_not_coerced=True,training_admission=False,photorealism_or_transfer_claim=False,legacy_primary_or_real_native_identity_claim=False,seconds=time.monotonic()-start,model_encoder_fit_current_V112_PnP=0));print(json.dumps(dict(completed=True,PASS=True,receipt=bind(receipt),scenes=6,teacher_exact=True,target_layers_immutable=True,seconds=time.monotonic()-start)),flush=True)
 except BaseException:write(OUT/'FAILED_EXECUTION.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
def prepare():
 assert sha(POLICY)==POLICY_SHA;policy=read(POLICY);g=H/'GENERATED_RENDER_TESTS.json';proof=read(g);assert proof['PASS']and proof['code']==bind(__file__)
 p=dict(completed=True,code=bind(__file__),policy=bind(POLICY),generated=bind(g),runtime=dict(executable=sys.executable,NumPy=np.__version__),output=str(OUT),linear_albedo=dict(wood=[.60,.36,.16],polymer=[.055,.075,.095],foreign=[.035,.16,.62]),light_toward_world=LIGHT.tolist(),ambient=.42,directional=.58,display='Separate part/depth/native images and four6-scenecontacts; no crops.',all_actual_mesh_payload_RGB_render_model_calls=0)
 for k in('geometry_result','geometry_code','root_geometry_review','raster_code','meshes'):p[k]=policy[k]
 p['geometry_design']=read(p['geometry_result']['path'])['design']
 write(PROTOCOL,p);template=dict(approved_source_render=False,model_forward=False,fit=False,current223=False,V112=False,PnP=False,automatic_retry=False,code=p['code'],protocol=bind(PROTOCOL),generated=p['generated'],policy=p['policy']);write(H/'ROOT_RENDER_ENTRY_TEMPLATE.json',template);print(json.dumps(dict(code=p['code'],protocol=bind(PROTOCOL),template=bind(H/'ROOT_RENDER_ENTRY_TEMPLATE.json'),command=sys.executable+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_RENDER_ENTRY.json',actual_render=0)),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
 if args.prepare:prepare()
 elif args.entry:run(args.entry)
 else:raise SystemExit('Use --prepare or --entry ROOT_ACTUAL_RENDER_ENTRY.json')
