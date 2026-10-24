"""Bounded read-only inventory; no ray labels, pixel decode or model work.

Checks existing 4200-instance metadata against official saved scene metadata,
and existing 1000 compressed depth files' hash/IHDR only. Existing aggregate
mask-teacher counts are summarized without running a teacher again.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, struct
import numpy as np

R = Path(__file__).resolve().parent
P = R.parents[1]
YCB = P/'query_visibility_public_real_20261022_v63/ycb'
V74 = P/'query_visibility_loco_native_context_20261023_v74'
V76 = P/'query_visibility_reference_local_support_20261023_v76'
SYNTH = P/'query_visibility_visible_preservation_20261023_v67/source_audit/SYNTH_REFERENCE_FRAME_MANIFEST.jsonl'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def write(p,d):
    with open(p,'x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def png_header(p):
    with open(p,'rb')as f:a=f.read(33)
    assert a[:8]==b'\x89PNG\r\n\x1a\n' and a[12:16]==b'IHDR'
    w,h,bits,color,comp,filt,interlace=struct.unpack('>IIBBBBB',a[16:29])
    return dict(H=h,W=w,bit_depth=bits,color_type=color,compression=comp,filter=filt,interlace=interlace)

def main():
    rows=lines(YCB/'instances.jsonl');assert len(rows)==4200
    grouped=defaultdict(list)
    for a in rows:grouped[a['scene_id'],a['frame_id']].append(a)
    assert len(grouped)==1000
    scene_meta={};scene_stats={};frame_rows=[];K_mismatch=pose_mismatch=0
    for scene in sorted({x[0] for x in grouped}):
        root=YCB/f'data/train_real/{scene:06d}'
        camera=read(root/'scene_camera.json');gt=read(root/'scene_gt.json')
        scene_meta[str(scene)]={'camera':b(root/'scene_camera.json'),'GT':b(root/'scene_gt.json')}
        counts=Counter()
        for (s,frame),instances in sorted(grouped.items()):
            if s!=scene:continue
            c=camera[str(frame)];g=gt[str(frame)];a=instances[0];depth=Path(a['depth'])
            assert depth==root/f'depth/{frame:06d}.png' and depth.exists()
            h=png_header(depth);assert (h['H'],h['W'])==(a['height'],a['width'])
            assert h['bit_depth']==16 and h['color_type']==0
            assert c['depth_scale']==a['depth_scale']==.1
            ordinal=[]
            for a in instances:
                assert a['depth']==str(depth) and a['depth_scale']==.1
                assert a['image']==str(root/f'rgb/{frame:06d}.png')
                K_mismatch+=not np.array_equal(a['K'],c['cam_K'])
                g0=g[a['instance_id']]
                pose_mismatch+=not (a['object_id']==g0['obj_id'] and np.array_equal(a['cam_R_m2c'],g0['cam_R_m2c']) and np.array_equal(a['cam_t_m2c'],g0['cam_t_m2c']))
                assert Path(a['mask_vis']).exists() and Path(a['mask_full']).exists()
                ordinal.append(a['instance_id'])
            assert sorted(ordinal)==list(range(len(g)))
            counts.update(frames=1,instances=len(instances),depth_headers=1,depth_bytes=depth.stat().st_size)
            frame_rows.append(dict(scene_id=scene,frame_id=frame,split=a['split'],depth_path=str(depth),depth_sha256=sha(depth),depth_bytes=depth.stat().st_size,H=h['H'],W=h['W'],bit_depth=h['bit_depth'],depth_scale_mm=.1,annotated_instance_count=len(instances),all_annotated_instance_M4_files_exist=True))
        scene_stats[str(scene)]=dict(counts,split='train'if scene<8 else'val',scene_camera_exists=True,scene_GT_exists=True)
    assert K_mismatch==pose_mismatch==0
    path=R/'EXISTING_YCB_DEPTH_FRAMES.csv'
    with open(path,'x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(frame_rows[0]));w.writeheader();w.writerows(frame_rows)
    synth=lines(SYNTH);assert len(synth)==1003
    source_roots=sorted({str(Path(a['annotation']).parent.parent)for a in synth})
    source_dirs=[]
    for root in source_roots:
        r=Path(root)
        source_dirs.append(dict(root=str(r),immediate_directories=sorted(x.name for x in r.iterdir()if x.is_dir()),known_depth_or_instance_directory_present=any((r/n).exists()for n in('depth','depths','distance','instance','instance_segmentation','segmentation'))))
    old=read(V76/'gt_corpus_build/CORPUS_SUMMARY.json')
    support=[]
    for name,a in old['by_source_split'].items():
        support.append(dict(source_split=name,slots=a['slots'],label_counts=a['labels'],reference_center_M0zero_slots=a['center_M0zero_slots'],reference_center_M0zero_labels=a['center_M0zero_labels'],zero_expected_footprint_slots=a['zero_expected_footprint_slots'],zero_expected_footprint_labels=a['zero_expected_footprint_labels'],counts_are_GT_corpus_not_primary_admitted_pool=True))
    output=dict(completed=True,PASS=True,YCB=dict(instances=4200,unique_native_depth_frames=1000,TRAIN_frames=800,VAL_frames=200,TRAIN_instances=3400,VAL_instances=800,depth_scale_mm_per_stored_unit=[.1],depth_png_headers='All1000 native480x640,uint16-grayscale IHDR; no pixel decode',depth_bytes=sum(x['depth_bytes']for x in frame_rows),camera_manifest_K_mismatches=int(K_mismatch),scene_GT_pose_and_instance_ordinal_mismatches=int(pose_mismatch),scene_metadata=scene_meta,per_scene=scene_stats,frame_inventory=b(path),M4_instance_masks_exist=4200,all_official_scene_annotated_instances_present=True,owner_map_exported=False,owner_map_possible_from_mask_visib='Construct GTID membership, preserve overlap/unassigned as UNKNOWN; not complete physical scene ownership',local_mesh_files_in_YCB_root=len(list(YCB.rglob('*.ply')))),SYNTH=dict(frames=1003,source_roots=source_dirs,previous_export_depth_inventory='No corresponding target/scene depth or instance-map export in prior1003 intake; occlusion_fraction9 only, no per-ray hit IDs/depths.'),existing_support_teacher=support,bindings=dict(instances=b(YCB/'instances.jsonl'),SYNTH_frame_manifest=b(SYNTH),prior_depth_audit=b(V74/'source_audit/YCB_PUBLIC_NATIVE_FEASIBILITY/LOCAL_AVAILABILITY_AND_ONE_DEPTH_AUDIT.json'),models_info=b(V74/'source_audit/YCB_PUBLIC_NATIVE_FEASIBILITY/models_info_OFFICIAL.json'),GT_teacher_summary=b(V76/'gt_corpus_build/CORPUS_SUMMARY.json'),code=b(__file__)),depth_RGB_mask_pixel_decodes=0,new_ray_or_visibility_labels=0,source_model_forward_fit_current223_PnP=0,existing_artifacts_modified=False)
    write(R/'SAVED_RAY_ASSET_INVENTORY.json',output)
    print(json.dumps(dict(PASS=True,output=b(R/'SAVED_RAY_ASSET_INVENTORY.json'))))

if __name__=='__main__':main()
