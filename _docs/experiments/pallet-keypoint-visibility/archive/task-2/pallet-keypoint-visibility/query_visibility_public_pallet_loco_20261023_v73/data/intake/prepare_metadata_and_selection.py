"""Actual COCO schema audit + fixed metadata-only 20-image selection, before RGB."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,re
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):
    p=OUT/name;assert not p.exists(),p;p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def subset(image):
    m=re.search(r'/subset-(\d+)/',image['path']);assert m,image;return int(m[1])
def main():
    a=json.loads((OUT/'annotations/loco-all-v1.json').read_text());cats={c['id']:c['name'] for c in a['categories']}
    assert cats=={3:'small_load_carrier',5:'forklift',7:'pallet',10:'stillage',11:'pallet_truck'}
    images={i['id']:i for i in a['images']};anns={x['id']:x for x in a['annotations']}
    assert len(images)==len(a['images'])==5097 and len(anns)==len(a['annotations'])==151428
    by_image=defaultdict(list)
    for r in a['annotations']:assert r['image_id'] in images;by_image[r['image_id']].append(r)
    sub_report={};subimage=set();subanns=set()
    for file in sorted((OUT/'annotations').glob('loco-sub*.json')):
        data=json.loads(file.read_text());ids={i['id'] for i in data['images']};annids={x['id'] for x in data['annotations']}
        assert not subimage&ids and not subanns&annids;subimage|=ids;subanns|=annids
        assert all(images[i['id']]==i for i in data['images']) and all(anns[x['id']]==x for x in data['annotations'])
        catcounts=Counter(cats[x['category_id']] for x in data['annotations'])
        sub_report[file.name]=dict(images=len(ids),annotations=len(annids),categories=dict(catcounts),pallet_images=sum(any(x['category_id']==7 for x in by_image[i]) for i in ids),dataset_ids=sorted({i['dataset_id'] for i in data['images']}),source_subsets=sorted({subset(i) for i in data['images']}),official_split_name='VALIDATION' if '-val' in file.name else 'TRAIN')
    assert subimage==set(images) and subanns==set(anns)
    annkeys=Counter(tuple(sorted(x)) for x in a['annotations']);imagekeys=Counter(tuple(sorted(x)) for x in a['images'])
    rect=0;bbox_bad=0;bbox_outside=0
    for r in a['annotations']:
        seg=r['segmentation'];im=images[r['image_id']];b=r['bbox']
        if len(seg)==1 and len(seg[0])==8:
            xx=seg[0][::2];yy=seg[0][1::2]
            if len(set(xx))==len(set(yy))==2:rect+=1
        bbox_bad+=int(len(b)!=4 or b[2]<=0 or b[3]<=0)
        bbox_outside+=int(b[0]<0 or b[1]<0 or b[0]+b[2]>im['width'] or b[1]+b[3]>im['height'])
    assert rect==151428 and all(r['isbbox'] is True for r in a['annotations'])
    assert all(r['metadata']=={} for r in a['annotations'])
    report=dict(schema='V73_ACTUAL_OFFICIAL_COCO_SCHEMA_AUDIT',PASS=True,all_images=5097,all_annotations=151428,categories=cats,category_annotation_counts=dict(Counter(cats[x['category_id']] for x in a['annotations'])),pallet_annotated_images=sum(any(r['category_id']==7 for r in by_image[i]) for i in images),
      subset_counts=sub_report,all_equals_disjoint_union_of_five_subsets=True,
      actual_annotation_key_signatures=[dict(keys=list(k),N=v) for k,v in annkeys.items()],actual_image_key_signatures=[dict(keys=list(k),N=v) for k,v in imagekeys.items()],
      bbox=dict(nonpositive_or_wrong_length=bbox_bad,outside_image_bounds_exact_no_tolerance=bbox_outside),
      segmentation=dict(field_present=True,all_151428_four_corner_axis_aligned_rectangles=True,isbbox_true=151428,metadata_empty=151428,usable_visible_or_amodal_instance_silhouette=False),
      keypoints_or_native8_visibility_fields_present=False,external_self_occlusion_fields_present=False,
      absence_evidence='Inspected every actual annotation/image key; all annotationmetadata empty. Rectangularsegmentation with isbboxTrue is bboxexport, not pixelvisiblemask. No native8/keypoint/occlusionlabel is supplied by thesefiles.',
      README_difference=dict(README_annotated_images=5593,actual_COCO_images=5097,README_annotations=152421,actual_COCO_annotations=151428,do_not_silently_change_official_files=True),
      current_RGB_GT_scores_weights_model_forward_fit_PnP=0)
    write('ACTUAL_COCO_SCHEMA_AUDIT.json',report)
    selected=[]
    for ss in range(1,6):
        pool=[im for im in a['images'] if subset(im)==ss and any(r['category_id']==7 for r in by_image[im['id']])]
        def order(im):return hashlib.sha256(f"LOCO-v1|{ss}|{im['id']}|{im['path']}".encode()).hexdigest()
        assert len(pool)>=4
        for rank,im in enumerate(sorted(pool,key=order)[:4]):
            selected.append(dict(selection_index=len(selected),source_subset=ss,source_rank=rank,image_id=im['id'],official_image_metadata=im,archive_member=im['path'].lstrip('/'),selection_SHA256=order(im),pallet_annotation_ids=[r['id'] for r in by_image[im['id']] if r['category_id']==7],pallet_bboxes_xywh=[r['bbox'] for r in by_image[im['id']] if r['category_id']==7]))
    assert len(selected)==20
    write('FIXED_20_RGB_SELECTION_LOCK.json',dict(schema='V73_METADATA_ONLY_PALLET_RGB_SELECTION_LOCK_V1',frozen_before_any_RGB_decode_or_view=True,rule='Four lowest SHA256(UTF8 LOCO-v1|subset|imageid|officialpath) among officialbbox-pallet-annotated images per eachsubset1..5. No size/occlusion/appearance/model filtering. No replacement afterRGBreview.',
      source_annotation_path=str(OUT/'annotations/loco-all-v1.json'),source_annotation_sha256=sha(OUT/'annotations/loco-all-v1.json'),number=20,per_subset=4,selected=selected,
      model_or_GT_corner_cue_selection=False,category_bbox_metadata_used_to_select_target_class=True,
      group_split_note='Officialsubsetfile split is lineage only. No new independentcaptureclaim or trainadmission; full sourcegroup/scenecontract pending.',training_admission='HOLD_PENDING_NATIVE8_CUE_SCENE_INDEPENDENT_CONTRACT'))
    print(json.dumps(dict(schema_audit=str(OUT/'ACTUAL_COCO_SCHEMA_AUDIT.json'),categories=report['category_annotation_counts'],pallet_images=report['pallet_annotated_images'],subset_counts=sub_report,selection=str(OUT/'FIXED_20_RGB_SELECTION_LOCK.json')),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
