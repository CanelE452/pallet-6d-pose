from pathlib import Path
from PIL import Image
import json,numpy as np,hashlib,zlib
R=Path(__file__).resolve().parent;Y=R.parents[2]/'query_visibility_public_real_20261022_v63/ycb'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
rows=[json.loads(x)for x in (Y/'instances.jsonl').read_text().splitlines()];unique={r['image']:r for r in rows};scenes={}
for sc in sorted({r['scene_id']for r in rows}):
 rr=[r for r in unique.values()if r['scene_id']==sc];sd=Y/f'data/train_real/{sc:06d}'
 scenes[str(sc)]={'split':rr[0]['split'],'frames':len(rr),'RGB_file_count':len(list((sd/'rgb').glob('*'))),'depth_PNG_count':len(list((sd/'depth').glob('*.png'))),'mask_visib_count':len(list((sd/'mask_visib').glob('*.png'))),'mask_full_count':len(list((sd/'mask').glob('*.png'))),'scene_camera_exists':(sd/'scene_camera.json').exists(),'scene_gt_exists':(sd/'scene_gt.json').exists(),'all_selected_depth_paths_exist':all(Path(r['depth']).is_file()for r in rr),'depth_scale':sorted({r['depth_scale']for r in rr})}
# Frozen lexicographically first TRAIN instance, independent of pixel quality or model outputs.
r=sorted((x for x in rows if x['split']=='train'),key=lambda x:(x['scene_id'],x['frame_id'],x['instance_id']))[0]
sc=Y/f'data/train_real/{r["scene_id"]:06d}';camfile=sc/'scene_camera.json';gtfile=sc/'scene_gt.json';frame=str(r['frame_id']);camera=json.loads(camfile.read_text())[frame];gt=json.loads(gtfile.read_text())[frame][r['instance_id']]
assert gt['obj_id']==r['object_id']
assert camera['cam_K']==r['K'] and camera['depth_scale']==r['depth_scale']
assert gt['cam_R_m2c']==r['cam_R_m2c'] and gt['cam_t_m2c']==r['cam_t_m2c']
depth=np.asarray(Image.open(r['depth']));vis=np.asarray(Image.open(r['mask_vis']))>0;full=np.asarray(Image.open(r['mask_full']))>0
assert depth.shape==vis.shape==full.shape==(480,640) and depth.dtype==np.uint16
K=np.array(camera['cam_K']).reshape(3,3);t=np.array(gt['cam_t_m2c']);uv=K@t;uv=uv[:2]/uv[2];ij=np.floor(uv+.5).astype(int);scale=camera['depth_scale'];valid=depth>0;targetvalid=vis&valid
entries={x['name']:x for x in map(json.loads,(Y/'selected_archive_entries.jsonl').read_text().splitlines())}
old={x['relative_path']:x for x in map(json.loads,(Y/'SELECTED_FILES_SHA256.jsonl').read_text().splitlines())};checks=[]
for path in [Path(r['image']),Path(r['depth']),Path(r['mask_vis']),Path(r['mask_full']),camfile,gtfile]:
 b=path.read_bytes();rel=str(path.relative_to(Y/'data'));e=entries[rel];prior=old['data/'+rel];actualsha=hashlib.sha256(b).hexdigest();crc=zlib.crc32(b)&0xffffffff
 checks.append({'path':str(path),'SHA256':actualsha,'CRC32':crc,'official_archive_size':e['size'],'official_archive_CRC32':e['crc32'],'stored_SHA_match':actualsha==prior['sha256'],'archive_size_CRC_match':len(b)==e['size'] and crc==e['crc32']})
 assert checks[-1]['stored_SHA_match'] and checks[-1]['archive_size_CRC_match']
result={'schema':'V74_EXISTING_YCB_PUBLIC_LOCAL_AVAILABILITY_AND_ONE_DEPTH_AUDIT','source':'BOP YCB-V official train_real, pinned HF revision 5c2c4aa229800355648cd268040aa814f8dc94f0','official_lineage':bind(Y/'SOURCES_AND_LABELS.json'),'original_intake_code':bind(Y/'intake_ycbv.py'),'official_repo_metadata':bind(Y/'official_hf_repo_metadata.json'),
 'local_inventory':{'frames':len(unique),'instances':len(rows),'scenes':scenes,'RGB':1000,'scene_depth':1000,'mask_visib':4200,'mask_full':4200},
 'one_instance_selection':'Lexicographically first existing TRAIN scene/frame/instance; not selected by source scores or image quality.','one_instance':{'scene_id':r['scene_id'],'frame_id':r['frame_id'],'instance_id':r['instance_id'],'object_id':r['object_id'],'RGB_path':r['image'],'scene_depth_path':r['depth'],'mask_vis_path':r['mask_vis'],'mask_full_path':r['mask_full'],'scene_camera':str(camfile),'scene_gt':str(gtfile),'depth_dtype':str(depth.dtype),'depth_shape':list(depth.shape),'depth_scale_mm_per_stored_unit':scale,'raw_nonzero_min_max':[int(depth[valid].min()),int(depth[valid].max())],'full_frame_valid_depth_pixels':int(valid.sum()),'full_frame_invalid_depth_pixels':int((~valid).sum()),'visible_mask_pixels':int(vis.sum()),'visible_mask_valid_depth_pixels':int(targetvalid.sum()),'visible_mask_invalid_depth_pixels':int((vis&~valid).sum()),'visible_depth_mm_quantiles':np.quantile(depth[targetvalid]*scale,[0,.01,.5,.99,1]).tolist(),'vis_subset_full':bool(not(vis&~full).any()),'model_origin_t_m2c_mm':t.tolist(),'model_origin_projection_xy':uv.tolist(),'model_origin_nearest_pixel_xy':ij.tolist(),'observed_depth_mm_at_origin_projection':float(depth[ij[1],ij[0]]*scale),'model_origin_pixel_in_visible_mask':bool(vis[ij[1],ij[0]]),'projection_note':'Origin depth is the object-model origin, not a surface depth or semantic visibility certificate.'},
 'unit_and_direction_contract':{'depth_mm':'raw uint16 depth * scene_camera.depth_scale; zero is invalid','model_units':'BOP YCB models converted from meters to millimeters; GT converted correspondingly (official local dataset_info).','object_pose':'cam_R_m2c @ X_model_mm + cam_t_m2c_mm gives camera point. cam_K gives pixel projection.','mask_instance_join':'mask path IMID_GTID where GTID indexes scene_gt[IMID] list, not obj_id. Official mask silhouette and visible silhouette are separate; no semantic native8 annotation.'},
 'one_instance_integrity_checks':checks,'missing_before_tiny_metadata_intake':['models_info.json and mesh files not found in existing local YCB intake or project file inventory. No object bounds inferred from pallet dimensions.'],
 'no_new_RGB_depth_download':True,'depth_is_original_official_scene_not_composite':True,'RGB_decoded':0,'model_current_fit_PnP':0,'code':bind(Path(__file__))}
p=R/'LOCAL_AVAILABILITY_AND_ONE_DEPTH_AUDIT.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'report':bind(p),'one_instance':result['one_instance']}))
