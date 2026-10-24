#!/usr/bin/env python3
"""Attach official instance boxes and verify SHA split exclusion and mask pixel accounting."""
import argparse,hashlib,json,pathlib
from collections import defaultdict
ROOT=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--incremental',action='store_true');args=ap.parse_args()
suffix='_incremental' if args.incremental else ''
allrows=[];meta={};differences=[];sha_splits=defaultdict(set);counts=defaultdict(lambda:defaultdict(int))
for split in ['train','val']:
 p=ROOT/f'{split}_instances{suffix}.jsonl'
 if not p.exists():continue
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 for r in rows:
  scene=r['scene_id'];key=f'{scene:06d}'
  if scene not in meta:meta[scene]=json.loads((ROOT/'data'/'train_real'/key/'scene_gt_info.json').read_text())
  i=meta[scene][str(r['frame_id'])][r['instance_id']]
  r['bbox_obj']=i['bbox_obj'];r['bbox_visib']=i['bbox_visib'];r['bop_gt_info']=i
  r['official_revision']='5c2c4aa229800355648cd268040aa814f8dc94f0'
  r['boundary_caution']='Full/visible masks are rendered/depth-derived; mesh/pose/RGB alignment is uncertain near raster boundaries.'
  if r['vis_pixels']!=i['px_count_visib']:differences.append({'scene_id':scene,'frame_id':r['frame_id'],'instance':r['instance_id'],'field':'visible_pixels','mask_count':r['vis_pixels'],'gt_info_count':i['px_count_visib']})
  # px_count_all can include silhouette outside the image and may exceed the raster full mask.
  if r['full_pixels']>i['px_count_all']:differences.append({'scene_id':scene,'frame_id':r['frame_id'],'instance':r['instance_id'],'field':'full_pixels_gt_info_less_than_inframe_mask','mask_count':r['full_pixels'],'gt_info_count':i['px_count_all']})
  sha_splits[r['image_sha256']].add(split);counts[split][str(r['object_id'])]+=1
 tmp=p.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows));tmp.rename(p);allrows.extend(rows)
assert all(len(x)==1 for x in sha_splits.values()),'exact RGB duplicates across train/val'
deny=set(json.loads((ROOT.parent/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256']);assert not deny&set(sha_splits)
if not args.incremental:
 out=ROOT/'instances.jsonl';out.write_text(''.join(json.dumps(r)+'\n' for r in allrows))
 selected=[json.loads(x) for x in (ROOT/'selected_archive_entries.jsonl').read_text().splitlines()]
 checks=[]
 for r in selected:
  p=ROOT/'data'/r['name'];assert p.exists()
  checks.append({'relative_path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size,'crc32_official_archive':r['crc32'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 (ROOT/'SELECTED_FILES_SHA256.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in checks))
 print(json.dumps({'selected_files':len(checks),'selected_bytes':sum(r['size_bytes'] for r in checks)}))
audit={'mode':suffix or 'final','image_sha256_train_val_disjoint':True,'image_sha256_eval_denylist_disjoint':True,'eval_images_opened':False,'object_instance_counts_by_split':dict(counts),'mask_vs_official_gt_info_differences':differences}
(ROOT/f'LABEL_AND_SPLIT_AUDIT{suffix}.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({'instance_rows':len(allrows),'unique_rgb':len(sha_splits),'gt_info_disagreements':len(differences),'object_ids':dict(counts)}))
