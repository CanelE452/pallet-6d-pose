"""Generated preparation first; actual scene mesh arrays/raster only under root-approved entry."""
from pathlib import Path
import argparse,hashlib,json,os,time,resource,traceback
import numpy as np
import cpu_part_raster as r
ROOT=Path(__file__).resolve().parent
V93=ROOT.parent
INVENTORY=V93/'geometry_inventory/GEOMETRY_INVENTORY.json'
EXPORT=V93/'geometry_inventory/EXPORT_SEAL.json'
SCOPE=V93/'SCOPE_BEFORE_ACTUAL_MESH_GEOMETRY.json'
OUT=V93/'raster_prototype/FIXED_PART152_SIX_CAMERAS'
DIRECTIONS=[('plusX',[1,0,0]),('minusX',[-1,0,0]),('plusY',[0,1,0]),('minusY',[0,-1,0]),('plusZ',[0,0,1]),('minusZ',[0,0,-1])]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while True:
   b=f.read(1<<20)
   if not b:break
   h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def write(p,o):
 assert not Path(p).exists(),str(p);Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check(d):assert sha(d['path'])==d['sha256'];return Path(d['path'])
def js(p):return json.loads(Path(p).read_text())
def json_camera(c):return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in c.items()}
def simple_camera(size=8):return dict(size=size,K=np.array([[8.,0.,(size-1)/2],[0.,8.,(size-1)/2],[0.,0.,1.]]),near=.1,R=np.eye(3),t=np.zeros(3))
def raster(v,t,p,cam,nparts,ds=True,ori=1,faces=None):
 n=len(t);return r.rasterize_parts(v,t,p,np.arange(n,dtype=np.int32)if faces is None else faces,np.full(n,ds,bool),np.full(n,ori,np.int8),cam,nparts)
def generated():
 cam=simple_camera();xy=np.array([[1.5,1.5],[5.5,1.5],[5.5,5.5],[1.5,5.5]]);v=r.unproject_xy(xy,2.,cam);t=np.array([[0,1,2],[0,2,3]]);p=np.zeros(2,np.int32)
 d,f,s=raster(v,t,p,cam,1);A=r.resolve_layers(d,f,[0]);assert np.sum(A['part_id']==0)==16 and np.all(A['depth'][A['part_id']==0]==2.) and not A['ambiguity'].any()
 # RightHanded front is negative screen area; reverse winding is culled before positive-order sampling.
 tf=t[:,[0,2,1]]
 df,ff,_=raster(v,tf,p,cam,1,ds=False);db,fb,_=raster(v,t,p,cam,1,ds=False)
 dl,fl,_=raster(v,t,p,cam,1,ds=False,ori=-1)
 assert np.isfinite(df).sum()==16 and np.isfinite(db).sum()==0 and np.array_equal(df,dl)
 # Camera handedness/frustum and metres preserved for all six views.
 bounds=np.array([[-1.,-2.,-3.],[1.,2.,3.]])
 for _,direction in DIRECTIONS:
  c=r.camera_from_bounds(bounds,direction);w=r.world_to_camera(r.box_corners(bounds),c);screen=r.project_camera(w,c)
  assert np.min(w[:,2])>c['near'] and np.all(screen>=0) and np.all(screen<=255)
  assert abs(np.linalg.det(c['R'])-1)<1e-14
 # Perspective interpolation checked directly against independent camera-ray/plane intersection.
 tri=np.array([[-.4,-.3,1.],[.4,-.2,3.],[-.2,.5,2.]])
 dd,_,_=raster(tri,np.array([[0,1,2]]),[0],cam,1);normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);maxerr=0.;points=0
 for yy,xx in zip(*np.where(np.isfinite(dd[0]))):
  ray=np.array([(xx-cam['K'][0,2])/8,(yy-cam['K'][1,2])/8,1.]);expected=float(normal@tri[0]/(normal@ray));maxerr=max(maxerr,abs(expected-dd[0,yy,xx]));points+=1
 assert points and maxerr<1e-12
 # Near straddle clips; no negative-depth fragments. Degenerate and entirely-behind faces remain empty.
 ncam=dict(cam,near=1.);cliptri=np.array([[-.2,-.2,.5],[.5,-.2,2.],[0,.5,2.]])
 clipped=r.clip_near(cliptri,1.);assert len(clipped)==4 and np.all(clipped[:,2]>=1.)
 dc,_,stat=raster(cliptri,np.array([[0,1,2]]),[0],ncam,1);assert np.isfinite(dc).any() and np.nanmin(np.where(np.isfinite(dc),dc,np.nan))>=1. and stat['near_straddling_faces']==1
 dz,_,_=raster(np.zeros((3,3)),np.array([[0,1,2]]),[0],cam,1);assert not np.isfinite(dz).any()
 # Same-part repeated/diagonal faces never ownership U; cross-part coplanar ties do.
 d2=np.repeat(d,2,axis=0);f2=np.repeat(f,2,axis=0);tie=r.resolve_layers(d2,f2,[7,3]);tie_rev=r.resolve_layers(d2[::-1],f2[::-1],[3,7])
 assert np.all(tie['ambiguity'][A['part_id']==0]) and np.all(tie['part_id'][A['part_id']==0]==3)
 assert all(np.array_equal(tie[k],tie_rev[k],equal_nan=True)for k in tie)
 dr,fr,_=raster(v,np.concatenate([t,t]),np.zeros(4,np.int32),cam,1,faces=np.array([3,4,1,2],np.int32));same=r.resolve_layers(dr,fr,[0]);assert not same['ambiguity'].any()
 # PS per-part banks stay immutable; partial foreign plane uses same raster math; fresh and reuse exact.
 front=r.unproject_xy(np.array([[1.5,1.5],[3.5,1.5],[3.5,5.5],[1.5,5.5]]),1.,cam)
 fd,ff,_=raster(front,t,p,cam,1,faces=np.array([100,101],np.int32))
 base=d.copy();basef=f.copy();PV=r.resolve_layers(d,f,[0],foreign=(fd[0],ff[0],308));assert np.array_equal(base,d)and np.array_equal(basef,f)
 allv=np.vstack([v,front]);allt=np.vstack([t,t+4]);allp=np.array([0,0,308,308]);freshd,freshf,_=raster(allv,allt,allp,cam,309,faces=np.array([0,1,100,101],np.int32));fresh=r.resolve_layers(freshd,freshf)
 assert all(np.array_equal(PV[k],fresh[k],equal_nan=True)for k in PV)
 part=r.partition_selected(A,A,PV,selected=0,foreign=308);assert part['foreign_blocked'].sum()==8 and part['V_selected_with_foreign'].sum()==8 and part['unknown'].sum()==0
 # Cross-part depth U cannot turn into self-hidden.
 amb_part=r.partition_selected(A,tie,tie,selected=0,foreign=308);assert amb_part['self_hidden_other_target'].sum()==0 and amb_part['unknown'].sum()==16
 # Negative fixtures ensure raw subset failures are rejected rather than masked by intersections.
 empty={k:v.copy()for k,v in A.items()};empty['part_id'][:]=-1
 for pa,ps,pv in [(empty,A,A),(A,empty,A)]:
  try:r.partition_selected(pa,ps,pv,selected=0,foreign=308)
  except AssertionError:pass
  else:raise AssertionError('raw monotonicity violation accepted')
 noowner={k:v.copy()for k,v in A.items()};noowner['part_id'][:]=-1
 unknown_bg=r.partition_selected(A,noowner,noowner,selected=0,foreign=308)
 assert unknown_bg['unknown'].sum()==16 and not unknown_bg['self_hidden_other_target'].any()
 d.flags.writeable=False;f.flags.writeable=False
 readonly=r.resolve_layers(d,f,[0]);assert np.array_equal(readonly['part_id'],A['part_id'])
 proof={'completed':True,'PASS':True,'code':bind(__file__),'raster_math':bind(ROOT/'cpu_part_raster.py'),'integer_pixel_centres_and_top_left_square16':True,'culling_right_left_and_doubleSided_generated':True,'all6_right_handed_camera_frustum':True,'perspective_scalar_ray_points':points,'perspective_scalar_max_abs_depth':maxerr,'near_clip_degenerate_behind':True,'samepart_ties_not_U_crosspart_ties_U':True,'tie_owner_order_independent':True,'PV_fresh_vs_reused_all_arrays_exact':True,'PS_base_unchanged':True,'partition_monotonicity_and_ambiguous_not_self':True,'raw_subset_violation_negative_fixtures_rejected':True,'PS_background_remains_U_not_SELF':True,'readonly_layer_resolve':True,'actual_mesh_array_load_or_mesh_pixel_render':0,'new_model_current_fit_PnP':0}
 write(ROOT/'GENERATED_RASTER_TESTS.json',proof);print(json.dumps(proof,indent=2))

def freeze():
 assert js(ROOT/'GENERATED_RASTER_TESTS.json')['PASS'];export=js(EXPORT);assert export['completed']and export['PASS'];inv=js(check(export['report']));m=inv['per_mesh'][152]
 assert m['part_id_asset_local_only']==152 and m['triangles_range'][1]-m['triangles_range'][0]==12
 cameras=[dict(name=name,**json_camera(r.camera_from_bounds(inv['stage']['bounds_world_m'],direction)))for name,direction in DIRECTIONS]
 protocol={'schema':'V93_FIXED_PART_CPU_RASTER_BEFORE_ACTUAL_PIXELS_V1','code':bind(__file__),'math_code':bind(ROOT/'cpu_part_raster.py'),'generated':bind(ROOT/'GENERATED_RASTER_TESTS.json'),'scope':bind(SCOPE),'export':bind(EXPORT),'inventory':export['report'],'safe_NPZ':export['safe_NPZ'],'fixed_part':m,'selected_part_id':152,'foreign_part_id':308,'all308_target_meshes_retained':True,'camera_order':cameras,'camera_rule':'AABBcenter/halfdiagonalR; eye=center+axis*1.3R/sin25deg; target=center; upY except +/-Y upZ; CVright/down/forwarddet+1. Kf=128/tan25,c127.5,integer0..255pixels; near=R*1e-4. No physical scale calibration.','raster_rule':'F64 original worldmetres unchanged. Respect declared USD right/left orientation and doubleSided BEFORE positivecoverageordering. Near SutherlandHodgman clip. Integerpixel topLeft dy<0 or dy==0&dx>0. z=1/sum(screen_bary/z_i).','tie_rule':'Per-part exact nearest z, same-part exact face ties min globalfaceID. Resolve allparts against finalmin with64eps*max(1,abs(z)); differenteligibleparts only ambiguityU, minpartID display winner. Samepart triangleedge is not U.','PA_PS_PV':'PS per-part isolated buffers once/camera. PA=selectedisolatedlayer. PS=alltargetlayers. PV=read-only same layers plus opaque foreignlayer; re-resolve finalmin (NOT literal nearest-owner summary update). PS summary and layer digests preserved before/after. Per-camera banks released before next camera; no layers stored.','foreign_rule':'Project unchangedselectedworldAABB8 to nativeview; rectangle[xmin-2,xmin+.6*(xmax-xmin)+2]x[ymin-2,ymax+2], z=.5*selectedAABBcenter_camera_z; two double-sided triangles; geometry-only no rendered-mask selection.','partition':'Assert raw certain PSselected subset PAselected and raw certain PVselected subset PSselected before intersections; save violationcounts0. On PAselectedcertain A: certain other-targetPS=>SELFproxy; certainselectedPS S then certainforeignPV=>FOREIGNproxy; certainselectedPV=>VISIBLEproxy; all remaining PA-support=>U. OutsideA is NOT_IN_PA, not physicalabsence. No human D/E/S labels generated.','shading':'Opaque flat diagnostic normals and deterministic partpalette; selectedgreen/foreignred highlighting only for inspection, not trainingRGB or certified material. Material MDL unresolved; no photorealism claim.','output':str(OUT),'preserved_all_six_no_drop':True,'output_budget_bytes':100000000,'peak_RSS_goal_bytes':1000000000,'buffer_bytes':308*256*256*(8+4),'new_native_queries_labels_model_current_fit_PnP':0}
 write(ROOT/'RASTER_PROTOCOL_BEFORE_ACTUAL_PIXELS.json',protocol)
 write(ROOT/'ROOT_RASTER_ENTRYLOCK_TEMPLATE.json',{'schema':'V93_ROOT_APPROVED_FIXED6_RASTER_ENTRY_V1','approved':False,'geometry_only':True,'code':bind(__file__),'math_code':bind(ROOT/'cpu_part_raster.py'),'protocol':bind(ROOT/'RASTER_PROTOCOL_BEFORE_ACTUAL_PIXELS.json'),'generated':bind(ROOT/'GENERATED_RASTER_TESTS.json'),'independent_review':{'path':'ROOT_FILL','sha256':'ROOT_FILL'},'export':bind(EXPORT),'model':False,'current':False,'fit':False,'PnP':False,'output':str(OUT)})
 print(json.dumps({'protocol':bind(ROOT/'RASTER_PROTOCOL_BEFORE_ACTUAL_PIXELS.json'),'template':bind(ROOT/'ROOT_RASTER_ENTRYLOCK_TEMPLATE.json')},indent=2))

def shade(result,normals):
 ids=result['part_id'];faces=result['face_id'];rgb=np.full((*ids.shape,3),235,np.uint8);good=ids>=0
 palette=np.array([[65+(i*67)%145,65+(i*97)%145,65+(i*43)%145]for i in range(309)],np.float64);palette[152]=[45,185,95];palette[308]=[225,55,45]
 if np.any(good):
  norm=np.zeros((*ids.shape,3),np.float64);target=good&(faces<len(normals));norm[target]=normals[faces[target]];norm[good&~target]=[0,0,-1]
  light=np.array([-.2,-.4,-1.]);light/=np.linalg.norm(light);intensity=.25+.75*np.abs(norm@light)
  rgb[good]=np.clip(np.floor(palette[ids[good]]*intensity[good,None]+.5),0,255).astype(np.uint8)
 rgb[result['ambiguity']]=[230,40,210]
 return rgb

def draw_figure(name,cam,PA,PS,PV,masks,rgbs,counts):
 from PIL import Image,ImageDraw,ImageFont
 fontpath='/usr/share/fonts/truetype/nanum/NanumGothic.ttf';font=ImageFont.truetype(fontpath,23);small=ImageFont.truetype(fontpath,19)
 im=Image.new('RGB',(1740,1390),'#f4f6f9');d=ImageDraw.Draw(im)
 text=f'Part152 / {name} | diagnostic geometry only | native semantic mapping UNKNOWN'
 d.text((30,20),text,font=font,fill='#15253a')
 d.text((30,58),'선택 part 강조색은 보기용입니다. 실제 material/학습 RGB/가림 gold가 아닙니다. MDL reference 미확인.',font=small,fill='#253448')
 for j,(mode,obj,rgb)in enumerate(zip(['PA: selected part alone','PS: all target parts','PV: target + foreign308'],[PA,PS,PV],rgbs)):
  x=30+j*570;d.text((x,100),mode,font=font,fill='#15253a');im.paste(Image.fromarray(rgb).resize((540,540),Image.Resampling.NEAREST),(x,137))
  mask=np.full((256,256,3),235,np.uint8)
  if j<2:mask[(obj['part_id']==152)&~obj['ambiguity']]=[35,175,95];mask[obj['ambiguity']]=[230,40,210]
  else:
   mask[masks['self_hidden_other_target']]=[225,175,40];mask[masks['foreign_blocked']]=[225,55,45];mask[masks['V_selected_with_foreign']]=[35,175,95];mask[masks['unknown']]=[230,40,210]
  im.paste(Image.fromarray(mask).resize((540,540),Image.Resampling.NEAREST),(x,705))
  d.text((x,1255),['PA A certain support','PS S certain support','A partition: green visible / red foreign'][j],font=small,fill='#253448')
 d.text((30,1290),f"A={counts['A']}  S={counts['S']}  V={counts['V']}  SELFproxy={counts['SELF']}  FOREIGNproxy={counts['FOREIGN']}  U={counts['U']} | pixel centres integer; c=127.5",font=small,fill='#15253a')
 d.text((30,1320),f"R={cam['radius']:.6f} m   distance={cam['distance']:.6f} m   near={cam['near']:.8f} m   declared metres retained, no physical scale correction",font=small,fill='#253448')
 d.text((30,1350),'노랑=other-target self proxy / 자홍=수치 ownership ambiguity 또는 미해결. Outside PA는 물리적 부재가 아닙니다.',font=small,fill='#253448')
 return im

def run(entry_path):
 e=js(entry_path);assert e['schema']=='V93_ROOT_APPROVED_FIXED6_RASTER_ENTRY_V1' and e['approved']is True and e['geometry_only']is True
 assert all(e[k]is False for k in ['model','current','fit','PnP'])
 assert e['code']['sha256']==sha(__file__)and e['math_code']['sha256']==sha(ROOT/'cpu_part_raster.py')
 p=js(check(e['protocol']));check(e['code']);check(e['math_code']);assert p['code']['sha256']==sha(__file__)and p['math_code']['sha256']==sha(ROOT/'cpu_part_raster.py')
 assert js(check(e['generated']))['PASS'] and js(check(e['independent_review']))['PASS'];assert e['export']['sha256']==p['export']['sha256']
 ex=js(check(e['export']));assert ex['completed']and ex['PASS'];inv=js(check(p['inventory']));npz=check(p['safe_NPZ'])
 assert e['output']==str(OUT)and not OUT.exists();OUT.mkdir(parents=True)
 # First actual mesh-array body load, after all root/review/export guards.
 z=np.load(npz,allow_pickle=False);vertices=z['vertices_world_m'];tri=z['triangles_global'];pid=z['triangle_asset_local_part_id'];faceids=np.arange(len(tri),dtype=np.int32)
 assert vertices.shape==(172656,3)and tri.shape==(110478,3)and np.isfinite(vertices).all()and np.unique(pid).size==308
 assert float(z['meters_per_stage_unit'])==1. and str(z['up_axis'])=='Y'
 assert np.allclose(np.stack([vertices.min(0),vertices.max(0)]),inv['stage']['bounds_world_m'],atol=1e-12,rtol=0)
 ori=np.array([1 if x['USD_orientation']=='rightHanded'else -1 for x in inv['per_mesh']],np.int8);ds=np.array([x['double_sided']for x in inv['per_mesh']],bool)
 assert np.array_equal(z['mesh_orientation'],np.array([x['USD_orientation']for x in inv['per_mesh']]))
 selectedbounds=np.array(inv['per_mesh'][152]['world_bounds_m']);assert np.sum(pid==152)==12
 results=[];start=time.time()
 for name,direction in DIRECTIONS:
  t0=time.time();cam=r.camera_from_bounds(inv['stage']['bounds_world_m'],direction);vc=r.world_to_camera(vertices,cam);assert np.min(vc[:,2])>cam['near']
  normal=np.cross(vc[tri[:,1]]-vc[tri[:,0]],vc[tri[:,2]]-vc[tri[:,0]]);normal/=np.linalg.norm(normal,axis=1)[:,None]
  depth,face,stats=r.rasterize_parts(vc,tri,pid,faceids,ds[pid],ori[pid],cam,308);assert stats['near_straddling_faces']==0
  depth.flags.writeable=False;face.flags.writeable=False
  PA=r.resolve_layers(depth[152:153],face[152:153],[152]);PS=r.resolve_layers(depth,face)
  PSdigest={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in PS.items()}
  layer_digest={'depth':hashlib.sha256(memoryview(depth).cast('B')).hexdigest(),'face':hashlib.sha256(memoryview(face).cast('B')).hexdigest()}
  foreign=r.foreign_rectangle(selectedbounds,cam);fd,ff,fs=r.rasterize_parts(foreign['vertices_camera'],foreign['triangles'],[0,0],[len(tri),len(tri)+1],[True,True],[1,1],cam,1)
  PV=r.resolve_layers(depth,face,foreign=(fd[0],ff[0],308));assert all(hashlib.sha256(v.tobytes()).hexdigest()==PSdigest[k]for k,v in PS.items())
  layer_digest_after={'depth':hashlib.sha256(memoryview(depth).cast('B')).hexdigest(),'face':hashlib.sha256(memoryview(face).cast('B')).hexdigest()};assert layer_digest==layer_digest_after
  masks=r.partition_selected(PA,PS,PV);counts={k:int(masks[v].sum())for k,v in [('A','A_selected_alone'),('S','S_selected_in_target'),('V','V_selected_with_foreign'),('SELF','self_hidden_other_target'),('FOREIGN','foreign_blocked'),('U','unknown')]}
  rgbs=[shade(obj,normal)for obj in [PA,PS,PV]]
  arrays={f'{tag}_{k}':v for tag,obj in [('PA',PA),('PS',PS),('PV',PV)]for k,v in obj.items()}
  arrays.update(masks);arrays.update({f'{tag}_RGB_diagnostic_only':rgb for tag,rgb in zip(['PA','PS','PV'],rgbs)})
  arrays.update(camera_K=cam['K'],camera_R_world_to_cv=cam['R'],camera_t_world_to_cv=cam['t'],foreign_vertices_camera=foreign['vertices_camera'],foreign_rectangle_xy=foreign['rectangle_xy'])
  np.savez_compressed(OUT/f'{name}.npz',**arrays)
  draw_figure(name,cam,PA,PS,PV,masks,rgbs,counts).save(OUT/f'{name}.png')
  lo=foreign['projected_part_AABB'][:2];hi=foreign['projected_part_AABB'][2:];area=float(np.prod(hi-lo));inter=float(np.prod(np.maximum(0,np.minimum(hi,255)-np.maximum(lo,0))))
  item={'camera':name,'camera_matrices':json_camera(cam),'selected_world_AABB_m':selectedbounds.tolist(),'part_AABB_projected_bbox':foreign['projected_part_AABB'].tolist(),'AABB_continuous_projected_area_px2':area,'AABB_continuous_inframe_fraction':inter/area if area>0 else None,'foreign_rectangle_xy':foreign['rectangle_xy'].tolist(),'foreign_depth_camera_z':foreign['z'],'counts':counts,'PS_ambiguity_pixels':int(PS['ambiguity'].sum()),'PV_ambiguity_pixels':int(PV['ambiguity'].sum()),'PS_raster':stats,'foreign_raster':fs,'status':'U_NO_SELECTED_PART_SAMPLE'if counts['A']==0 else 'GEOMETRY_DIAGNOSTIC_ONLY','PS_unchanged':True,'PS_summary_digest_before_after':PSdigest,'immutable_layer_digest_before':layer_digest,'immutable_layer_digest_after':layer_digest_after,'raw_monotonicity_violation_counts':{'S_outside_A':int(masks['raw_S_outside_A']),'V_outside_S':int(masks['raw_V_outside_S'])},'NPZ':bind(OUT/f'{name}.npz'),'PNG':bind(OUT/f'{name}.png'),'seconds':time.time()-t0}
  results.append(item);print(json.dumps({'camera':name,'counts':counts,'seconds':item['seconds']}),flush=True)
  del depth,face,PA,PS,PV,fd,ff,masks,rgbs,arrays,normal,vc
 size=sum(q.stat().st_size for q in OUT.iterdir()if q.is_file());peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 assert size<100000000 and peak<1000000000,(size,peak)
 write(OUT/'GEOMETRY_DIAGNOSTIC_RESULTS.json',{'completed':True,'PASS':True,'protocol':e['protocol'],'export':e['export'],'results':results,'all6_retained':True,'source_pixel_mesh_raster_cameras':6,'perpart_layers_saved':False,'peak_RSS_bytes':peak,'output_bytes_before_JSON':size,'seconds':time.time()-start,'part_role_and_native8':'UNKNOWN','human_DES_labels_produced':0,'model_training_current_PnP':0})
 write(OUT/'RASTER_OUTPUT_SEAL.json',{'completed':True,'PASS':True,'entry':bind(entry_path),'code':bind(__file__),'math_code':bind(ROOT/'cpu_part_raster.py'),'protocol':e['protocol'],'export':e['export'],'cameras':6,'selected_part':152,'foreign_part':308,'files':[bind(q)for q in sorted(OUT.iterdir())if q.is_file()],'new_model_training_current_PnP':0})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['generated','freeze','run']);p.add_argument('--entry',type=Path);a=p.parse_args()
 if a.stage=='generated':generated()
 elif a.stage=='freeze':freeze()
 else:
  assert a.entry
  try:run(a.entry)
  except BaseException as e:
   if OUT.exists()and not(OUT/'HALTED_FAILURE.json').exists():write(OUT/'HALTED_FAILURE.json',{'completed':False,'error':repr(e),'traceback':traceback.format_exc(),'no_silent_retry':True})
   raise
