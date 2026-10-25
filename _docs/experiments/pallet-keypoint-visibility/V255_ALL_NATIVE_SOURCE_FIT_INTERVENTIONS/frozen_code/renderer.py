"""Root-entry-only 192-base registered source corpus renderer; no crops or models."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_CORPUS_POLICY_BEFORE_PIXELS.json';POLICY_SHA='43a0a3989793a6e1ead3e2cf2b7897c8a7e7dfc7c7d21f46a7e29c471a3663ed'
CORE=H/'corpus_math.py';PROTOCOL=H/'CORPUS_PROTOCOL_BEFORE_PIXELS.json';CONFIGS=H/'BASE_CONFIGS192_BEFORE_RGB.json';OUT=V/'actual_corpus_v1'
SOURCE=P/'query_visibility_spatial_stencil_control_20261023_v127/registered_native_teacher_feasibility/RESULT.json'
HELPER=P/'query_visibility_registered_native_render_20261023_v128/render_preparation/render_registered_six_once.py'
VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def mod(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def camera_arrays(c):return {k:np.asarray(v,np.float64)if k in('K','R','t','eye_world')else v for k,v in c.items()}
def diagnostics(data,camera,raster,teacher,buf,vertices,triangles,owners,records,perm,helper):
 points=data['native_witness_points'];xy=raster.project_camera(raster.world_to_camera(points.reshape(-1,3),camera),camera).reshape(8,37,2);ix,inside=helper.nearest_pixels(xy,512);shape=(8,37);po=np.full(shape,-1,np.int32);pf=po.copy();pd=np.full(shape,np.nan);amb=np.ones(shape,bool);ro=po.copy();rd=pd.copy();ru=np.ones(shape,bool);summary=[]
 for i,j in np.ndindex(8,37):
  if inside[i,j]:x,y=ix[i,j];po[i,j]=buf['part_id'][y,x];pf[i,j]=buf['face_id'][y,x];pd[i,j]=buf['depth'][y,x];amb[i,j]=buf['ambiguity'][y,x]
  hit=teacher.first_hit(camera['eye_world'],points[i,j],vertices[triangles],owners)
  if hit['status']=='HIT':
   ro[i,j]=hit['owner'];direction=points[i,j]-camera['eye_world'];direction/=np.linalg.norm(direction);location=camera['eye_world']+direction*hit['distance'];rd[i,j]=raster.world_to_camera(location[None],camera)[0,2];ru[i,j]=False
 patchxy=raster.project_camera(raster.world_to_camera(data['native_patch_quads'].reshape(-1,3),camera),camera).reshape(8,3,4,2)
 for cf,canonical in enumerate(perm):
  mask=inside[canonical]&~amb[canonical]&~ru[canonical];finite=mask&np.isfinite(pd[canonical])&np.isfinite(rd[canonical]);active=np.array(records[canonical]['active_witness_indices']);summary.append(dict(CF_index=cf,canonical_index=canonical,state=records[canonical]['state'],projected_patch_spans_xy=(patchxy[canonical].max(1)-patchxy[canonical].min(1)).tolist(),nearest_OOB=int((~inside[canonical]).sum()),nearest_ambiguous=int((inside[canonical]&amb[canonical]).sum()),nearest_background=int((inside[canonical]&(po[canonical]<0)).sum()),owner_disagreements=int((mask&(po[canonical]!=ro[canonical])).sum()),active_unique_nearest_pixels=len(set(map(tuple,ix[canonical,active][inside[canonical,active]].tolist()))),finite_depth_comparable=int(finite.sum()),max_depth_difference_between_different_rays=float(np.max(np.abs(pd[canonical,finite]-rd[canonical,finite])))if finite.any()else None))
 arrays=dict(witness_xy_CF=xy[perm],witness_nearest_xy_CF=ix[perm],witness_nearest_in_image_CF=inside[perm],witness_nearest_owner_CF=po[perm],witness_nearest_triangle_CF=pf[perm],witness_nearest_depth_CF=pd[perm],witness_nearest_ambiguity_CF=amb[perm],witness_continuous_owner_CF=ro[perm],witness_continuous_depth_CF=rd[perm],witness_continuous_U_CF=ru[perm])
 return arrays,summary
def authorize(entry):
 e=read(entry);assert e['approved_source_corpus_render']is True and all(e[k]is False for k in('model_forward','fit','current223','V112','PnP','automatic_retry'))
 for k in('code','core_code','protocol','policy','base_configs','generated'):check(e[k])
 assert sha(POLICY)==POLICY_SHA and e['code']==bind(__file__)and e['core_code']==bind(CORE)and e['protocol']==bind(PROTOCOL)and e['policy']==bind(POLICY)and e['base_configs']==bind(CONFIGS)
 p=read(PROTOCOL);assert p['code']==e['code']and p['core_code']==e['core_code']and p['generated']==e['generated']and read(e['generated']['path'])['PASS']
 for k in('geometry_result','geometry_code','geometry_design','raster_code','reviewed_renderer_helper'):check(p[k])
 for b in p['meshes']:check(b)
 assert np.__version__==p['runtime']['NumPy']and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
 packet=read(CONFIGS);assert packet['completed']and len(packet['rows'])==192
 return e,p,packet['rows']
def run(entry):
 e,p,configs=authorize(entry);assert not OUT.exists(),'Preserve completed bases/failure; explicit new attempt needed.'
 from PIL import Image
 core=mod(CORE,'v129_geometry_material_math');raster=mod(p['raster_code']['path'],'v129_pinned_V93');teacher=mod(p['geometry_code']['path'],'v129_registered_teacher');helper=mod(p['reviewed_renderer_helper']['path'],'v129_pinned_V128');source=read(p['geometry_result']['path']);templates=[]
 for mesh in p['meshes']:
  with np.load(mesh['path'],allow_pickle=False)as f:templates.append({k:np.array(f[k],copy=True)for k in('vertices','triangles','part_owner','face_axis','face_side','native_anchor3D','native_primary_part_id','native_incident_part_ids','native_patch_quads','native_edge_samples','native_witness_points')})
 OUT.mkdir();(OUT/'bases').mkdir();(OUT/'geometry').mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),bases=192,scenes=768,automatic_retry=False,source_only=True));started=time.monotonic();geometry_cache={};target_cache=None;inputs=[];gts=[];base_receipts=[];geometry_outputs=[];outputs=[];target_rasters=0;foreign_rasters=0
 try:
  for cfg in configs:
   b=cfg['base_index'];assert b==len(base_receipts);baseout=OUT/'bases'/cfg['base_id'];baseout.mkdir();group=cfg['geometry_group_id']
   if group not in geometry_cache:
    data=core.scaled_data(templates[cfg['template_index']],cfg);file=OUT/'geometry'/(group+'.npz');np.savez_compressed(file,**data,template_to_canonical=np.diag(np.r_[cfg['positive_scale_xyz'],1.]),dimensions_xyz=np.array(cfg['dimensions_xyz_m']));geometry_cache[group]=(data,bind(file));geometry_outputs.append(bind(file))
   data,geometry=geometry_cache[group];vertices=data['vertices'];tri=data['triangles'];pid=data['part_owner'];F=len(tri);camera=camera_arrays(cfg['camera']);perm=cfg['permCF_to_canonical'];native_xy=raster.project_camera(raster.world_to_camera(data['native_anchor3D'],camera),camera);assert np.array_equal(native_xy[perm],np.array(cfg['native8_CF_xy']))
   if target_cache is None or target_cache[0]!=cfg['target_layer_group_id']:
    target_cache=None;d,f,stats=raster.rasterize_parts(raster.world_to_camera(vertices,camera),tri,pid,np.arange(F,dtype=np.int32),np.ones(F,bool),np.ones(F,np.int8),camera,len(source['results'][cfg['template_index']]['parts']));target_cache=(cfg['target_layer_group_id'],d,f,stats);target_rasters+=1
   _,d,f,stats=target_cache;layer_before=helper.digest(d,f);basebuf=raster.resolve_layers(d,f);foreign,translations,placement=core.foreign_placements(data,cfg);baseinputs=[];basegts=[];before_rgb=None;before_buf=None
   fp=baseout/'FOREIGN_GEOMETRY.npz';np.savez_compressed(fp,triangles=teacher.BOX_TRI,cover_vertices=foreign['COVER'],near_sham_vertices=foreign['NEAR_SHAM'],far_sham_vertices=foreign['FAR_SHAM']);outputs.append(bind(fp))
   for variant_index,variant in enumerate(VARIANTS):
    scene_index=4*b+variant_index;sid='V129_SCENE_%04d'%scene_index;fv=None if variant=='BEFORE'else foreign[variant];scene_vertices=vertices;scene_triangles=tri;scene_owners=pid;translation=np.zeros(3)
    if fv is None:buf={k:v.copy()for k,v in basebuf.items()}
    else:
     fd,ff,_=raster.rasterize_parts(raster.world_to_camera(fv,camera),teacher.BOX_TRI,np.zeros(12,np.int32),F+np.arange(12,dtype=np.int32),np.ones(12,bool),np.ones(12,np.int8),camera,1);foreign_rasters+=1;buf=raster.resolve_layers(d,f,foreign=(fd[0],ff[0],999));scene_vertices=np.concatenate([vertices,fv]);scene_triangles=np.concatenate([tri,teacher.BOX_TRI+len(vertices)]);scene_owners=np.r_[pid,np.full(12,999,np.int32)];translation=translations[variant]
    assert helper.digest(d,f)==layer_before;records=teacher.region_teacher(data,camera,raster,fv);records_CF=core.CF_records(records,perm,data,cfg['camera']);states=[r['state']for r in records_CF];rgb=core.shade(buf,scene_vertices,scene_triangles,cfg,translation)
    if before_rgb is None:before_rgb=rgb.copy();before_buf=buf;unchanged=int((buf['part_id']>=0).sum())
    else:
     mask=(buf['part_id']>=0)&(buf['part_id']!=999)&(buf['part_id']==before_buf['part_id'])&(buf['face_id']==before_buf['face_id'])&(buf['depth']==before_buf['depth']);assert np.array_equal(rgb[mask],before_rgb[mask]);unchanged=int(mask.sum())
    diag,summary=diagnostics(data,camera,raster,teacher,buf,scene_vertices,scene_triangles,scene_owners,records,perm,helper);bf=baseout/(variant+'_BUFFERS.npz');np.savez_compressed(bf,RGB_uint8=rgb,camera_depth=buf['depth'],part_id=buf['part_id'],global_triangle_id=buf['face_id'],ambiguity=buf['ambiguity'],tie_count=buf['tie_part_count'],native8_CF_xy=native_xy[perm],native8_canonical_xy=native_xy,permCF_to_canonical=np.array(perm,np.int32),state8_CF=np.array(states),camera_K=camera['K'],camera_R=camera['R'],camera_t=camera['t'],**diag);png=baseout/(variant+'_RGB.png');Image.fromarray(rgb).save(png);outputs.extend([bind(bf),bind(png)])
    identity=dict(scene_index=scene_index,base_index=b,id=sid,base_id=cfg['base_id'],split=cfg['split'],variant=variant);ir=dict(identity,style='__'.join(cfg['factors'][k]for k in('geometry_family','aspect','material')),factors=cfg['factors'],full_RGB=bind(bf),full_RGB_format='NPZ_RGB_UINT8_HWC',full_RGB_key='RGB_uint8',decoded_RGB_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),raw_hw=[512,512],raw_RGB_PNG=bind(png),reference_member='native8_CF_xy',source_base_config=dict(file=e['base_configs'],row_index=b),source_emulated_not_initializer=True)
    gr=dict(identity,state8_CF=states,records_CF=records_CF,records_canonical=records,permCF_to_canonical=perm,native8_CF_xy=native_xy[perm].tolist(),camera=cfg['camera'],geometry=geometry,foreign_geometry=bind(fp),placement=placement,nearest_pixel_diagnostic_CF=summary,raw_buffers=bind(bf),target_layers_immutable=True,target_layer_digest=layer_before,target_raster_stats=stats,same_target_surface_RGB_exact=True,same_target_surface_pixels=unchanged,ambiguity_pixels=int(buf['ambiguity'].sum()),foreign_pixels=int((buf['part_id']==999).sum()),training_admitted=False,constructed_registration_not_legacy_or_real_native_gold=True)
    baseinputs.append(ir);basegts.append(gr)
   assert helper.digest(d,f)==layer_before
   ip=baseout/'SCENES_INPUT_ONLY.json';gp=baseout/'SCENES_GT_ONLY.json';write(ip,dict(completed=True,rows=baseinputs));write(gp,dict(completed=True,rows=basegts));bc=baseout/'BASE_COMPLETE.json';write(bc,dict(completed=True,PASS=True,base_index=b,base_id=cfg['base_id'],config=dict(file=e['base_configs'],row_index=b),input_scenes=bind(ip),GT_scenes=bind(gp),geometry=geometry,foreign_geometry=bind(fp),target_layers_before=layer_before,target_layers_after=helper.digest(d,f),scenes=4,no_retry=True));base_receipts.append(bind(bc));inputs.extend(baseinputs);gts.extend(basegts);outputs.extend([bind(ip),bind(gp),bind(bc)])
   print(json.dumps(dict(bases_done=b+1,total_bases=192,scenes_done=len(inputs),seconds=round(time.monotonic()-started,2),last_base=cfg['base_id'],checkpoint=bind(bc),first_preview=baseinputs[0]['raw_RGB_PNG']if b<2 else None)),flush=True)
  assert len(inputs)==len(gts)==768 and [r['scene_index']for r in inputs]==list(range(768))and target_rasters==96 and foreign_rasters==576
  ip=OUT/'SCENES_INPUT_ONLY.json';gp=OUT/'SCENES_GT_ONLY.json';write(ip,dict(completed=True,rows=inputs));write(gp,dict(completed=True,rows=gts));outputs.extend([bind(ip),bind(gp)]);counts={fold:{s:sum(r['state8_CF'].count(s)for r in gts if r['split']==fold)for s in'DESU'}for fold in('TRAIN','VALIDATION')}
  for mesh in p['meshes']:check(mesh)
  result=OUT/'RENDER_COMPLETE.json';write(result,dict(completed=True,PASS=True,entry=bind(entry),code=e['code'],core_code=e['core_code'],protocol=e['protocol'],policy=e['policy'],base_configs=e['base_configs'],input_scenes=bind(ip),GT_scenes=bind(gp),outputs=outputs+geometry_outputs,base_checkpoints=base_receipts,bases=192,scenes=768,target_rasters=target_rasters,foreign_rasters=foreign_rasters,size=512,state_counts_before_query_expansion=counts,all_variants_retained=True,same_target_RGB_bitexact=True,continuous_teacher_not_relabelled_by_pixels=True,no_queries_or_crops=True,training_admitted=False,seconds=time.monotonic()-started,models_fit_current_V112_PnP=0));print(json.dumps(dict(completed=True,PASS=True,receipt=bind(result),bases=192,scenes=768,seconds=round(time.monotonic()-started,2))),flush=True)
 except BaseException:write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(base_index=cfg['base_index']if'cfg'in locals()else None,completed_bases=len(base_receipts),traceback=traceback.format_exc(),preserve_every_completed_base=True,automatic_retry=False));raise
def prepare():
 assert sha(POLICY)==POLICY_SHA;policy=read(POLICY);source=read(SOURCE);core=mod(CORE,'v129_prepare_math');g=H/'GENERATED_CORPUS_TESTS.json';proof=read(g);assert proof['PASS']and proof['code']==bind(__file__)and proof['core_code']==bind(CORE)
 meshes=[r['mesh']for r in source['results']];configs=core.make_configs(policy,meshes);write(CONFIGS,dict(completed=True,policy=bind(POLICY),code=bind(__file__),core_code=bind(CORE),rows=configs,actual_mesh_payload_RGB_model_calls=0))
 p=dict(completed=True,code=bind(__file__),core_code=bind(CORE),generated=bind(g),policy=bind(POLICY),base_configs=bind(CONFIGS),geometry_result=bind(SOURCE),geometry_code=source['code'],geometry_design=source['design'],meshes=meshes,raster_code=source['V93_geometry_projection_and_raster_code'],reviewed_renderer_helper=bind(HELPER),runtime=dict(executable=sys.executable,NumPy=np.__version__),output=str(OUT),base_count=192,scene_count=768,target_raster_groups=96,variants=list(VARIANTS),storage='Per-scene compressed buffers/RGBPNG; no saved512 per-part layers; one active target-layer group; per-base checkpoints; no automatic retry',actual_RGB_render_model_calls=0);write(PROTOCOL,p)
 template=dict(approved_source_corpus_render=False,model_forward=False,fit=False,current223=False,V112=False,PnP=False,automatic_retry=False,code=p['code'],core_code=p['core_code'],protocol=bind(PROTOCOL),policy=p['policy'],base_configs=p['base_configs'],generated=p['generated']);write(H/'ROOT_RENDER_ENTRY_TEMPLATE.json',template);print(json.dumps(dict(code=p['code'],core_code=p['core_code'],base_configs=p['base_configs'],protocol=bind(PROTOCOL),template=bind(H/'ROOT_RENDER_ENTRY_TEMPLATE.json'),command=sys.executable+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_CORPUS_ENTRY.json',actual_render=0)),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare:prepare()
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --prepare or --entry ROOT_ACTUAL_CORPUS_ENTRY.json')
