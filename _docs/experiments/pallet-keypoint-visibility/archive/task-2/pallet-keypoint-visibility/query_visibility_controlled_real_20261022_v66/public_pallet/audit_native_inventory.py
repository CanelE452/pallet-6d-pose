#!/usr/bin/env python3
"""Full official catalog format/integrity/exact-evaluation-hash exclusion audit."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,hashlib,json
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data';V63=ROOT.parents[1]/'query_visibility_public_real_20261022_v63';cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
 out=ROOT/'NATIVE_FORMAT_AND_EXCLUSION_AUDIT.json';assert not out.exists();catalog=[json.loads(s)for s in(ROOT/'ALL_DOWNLOAD_INVENTORY.jsonl').read_text().splitlines()];assert len(catalog)==len({r['file_id']for r in catalog})==1234
 denied_path=V63/'contract/EVALUATION_SHA256_DENYLIST.json';denied=set(json.loads(denied_path.read_text())['image_sha256']);RGB=[];counts=Counter();maskvalues=Counter();poseerrors=[];depthnonzero=[];intrinsics={};byhash=defaultdict(list)
 for r in catalog:
  path=DATA/r['relative_path'];assert path.exists()and path.stat().st_size==r['bytes']
  if'/rgb/'in r['relative_path']:
   with Image.open(path)as im:assert im.size==(640,480)and im.mode=='RGB'
   assert sha(path)==r['sha256'];RGB.append(dict(relative_path=r['relative_path'],sha256=r['sha256'],MD5=r['MD5'],bytes=r['bytes'],image=str(path)));byhash[r['sha256']].append(r['relative_path']);counts['RGB']+=1
  elif'/depth/'in r['relative_path']:
   d=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);assert d.shape==(480,640)and d.dtype==np.uint16;depthnonzero.append(float((d>0).mean()));counts['depth_uint16_mm']+=1
  elif'/masks/'in r['relative_path']:
   m=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);assert m.shape==(480,640)and m.dtype==np.uint8;values=tuple(int(x)for x in np.unique(m));assert set(values)<={0,255};maskvalues[str(values)]+=1;counts['SAM_teacher_mask']+=1
  elif'/annotated_poses/'in r['relative_path']:
   T=np.loadtxt(path);assert T.shape==(4,4)and np.isfinite(T).all();poseerrors.append(dict(path=r['relative_path'],orthogonality=float(np.max(np.abs(T[:3,:3].T@T[:3,:3]-np.eye(3)))),det=float(np.linalg.det(T[:3,:3])),lastrow_error=float(np.max(np.abs(T[3]-[0,0,0,1])))));counts['supplied_pose']+=1
  elif r['relative_path'].endswith('/cam_K.txt'):
   K=np.loadtxt(path);assert K.shape==(3,3)and np.isfinite(K).all();intrinsics[r['relative_path']]=K.tolist()
 overlap=[r for r in RGB if r['sha256']in denied];assert not overlap
 lock=json.loads((ROOT/'SPLIT_LOCK_BEFORE_FIT.json').read_text());manifest=[]
 for r in lock['frames']:
  rr=dict(id=r['id'],split=r['split'],session=r['session'],frame_id=r['frame_id'],image=str(DATA/r['image']),depth_mm=str(DATA/r['depth']),mask_SAM_teacher=str(DATA/r['mask_teacher']),mask_full=None,pose_ob_in_cam_supplied=str(DATA/r['pose']),K_file=str(DATA/r['K']),mesh=str(DATA/'assets/mesh/pallox.obj'),source_DOI='10.48804/DJLDFS',source_version='1.0',license='CC-BY-4.0',image_sha256=sha(DATA/r['image']),visible_mask_independent_human_GT=False,semantic_self_GT='UNKNOWN',mask_zero_is_outside_or_external_GT=False,admission='LABEL_QUALITY_PENDING_NO_FIT')
  manifest.append(rr)
 with(ROOT/'FULLY_JOINED_SOURCE_MANIFEST.jsonl').open('w')as f:
  for r in manifest:f.write(json.dumps(r)+'\n')
 with(ROOT/'ALL_CANDIDATE_RGB_SHA256.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(RGB[0]));w.writeheader();w.writerows(sorted(RGB,key=lambda r:r['relative_path']))
 summary=dict(schema='OFFICIAL_PALLOX_NATIVE_INTAKE_INTEGRITY_V1',source_only=True,current_private_RGB_GT_scores_read=False,training=False,model_inference=False,PnP=False,catalog_files=len(catalog),catalog_bytes=sum(r['bytes']for r in catalog),all_official_MD5_checked_by_download=True,all_file_SHA256_inventory=sha(ROOT/'ALL_DOWNLOAD_INVENTORY.jsonl'),counts=dict(counts),mask_unique_values=dict(maskvalues),RGB_shape=[480,640,3],depth_shape=[480,640],depth_dtype='uint16',depth_units='mm',depth_nonzero_fraction=dict(min=min(depthnonzero),median=float(np.median(depthnonzero)),max=max(depthnonzero)),pose_max_orthogonality_error=max(r['orthogonality']for r in poseerrors),pose_det_range=[min(r['det']for r in poseerrors),max(r['det']for r in poseerrors)],pose_max_lastrow_error=max(r['lastrow_error']for r in poseerrors),K_by_session=intrinsics,RGB_denylist_overlap_count=len(overlap),RGB_denylist_overlap=overlap,denylist_sha256=sha(denied_path),denied_unique_images=len(denied),public_internal_identical_RGB_groups=[v for v in byhash.values()if len(v)>1],fully_joined=284,split_counts=dict(Counter(r['split']for r in manifest)),incomplete_RGB_excluded=38,official_metadata_sha256=sha(ROOT/'DATASET_V1_OFFICIAL.json'),protocol_split_sha256=sha(ROOT/'SPLIT_LOCK_BEFORE_FIT.json'),all_candidate_RGB_inventory_sha256=sha(ROOT/'ALL_CANDIDATE_RGB_SHA256.csv'),fully_joined_source_manifest_sha256=sha(ROOT/'FULLY_JOINED_SOURCE_MANIFEST.jsonl'),geometry_direction='README explicit ob_in_cam equation; fixed-six direct/inverse projection audited separately; no poses optimized',mask_provenance='Official paper manually selected prompt points then SAM; no evidence of independent manual per-pixel correction',full_mask_GT='NOT_PROVIDED; projected mesh is a separate source proxy, not automatically validated full/outside truth')
 out.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
