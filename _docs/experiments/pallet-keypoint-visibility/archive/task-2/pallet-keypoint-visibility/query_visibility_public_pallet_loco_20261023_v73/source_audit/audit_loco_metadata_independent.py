"""Independent official COCO + ZIP-directory audit; never decodes image bytes."""
from pathlib import Path,PurePosixPath
import csv
import hashlib
import json
from collections import Counter,defaultdict
from itertools import product
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/intake'
OUT=ROOT/'source_audit'
EXPECTED_SHA='f3629d989071b824edc63e000f45ccc5826cfbb201a338f16ed87ab3e10bcdcf'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def bind(path):return dict(path=str(path),sha256=sha(path))
def load(path):return json.loads(Path(path).read_text())
def norm(path):
    p=PurePosixPath(path)
    assert '..' not in p.parts,path
    return str(p).lstrip('/')


def rectangle(seg):
    if not isinstance(seg,list) or len(seg)!=1 or len(seg[0])!=8:return False
    points=list(zip(seg[0][::2],seg[0][1::2]));xs={x for x,y in points};ys={y for x,y in points}
    return len(xs)==len(ys)==2 and set(points)==set(product(xs,ys))


def shape_payload(ann):
    return json.dumps({k:v for k,v in ann.items() if k not in ['id','image_id','color']},sort_keys=True,separators=(',',':'))


def output_csv(path,rows,fields):
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    result=OUT/'INDEPENDENT_METADATA_INVENTORY_AUDIT.json'
    assert not result.exists(),'No overwrite'
    archive=SRC/'dataset.zip'
    assert archive.stat().st_size==769055500 and sha(archive)==EXPECTED_SHA
    with zipfile.ZipFile(archive) as z:
        # Central-directory metadata only. z.open/read/extract/testzip forbidden.
        infos=z.infolist();files=[x for x in infos if not x.is_dir()]
        names=[norm(x.filename) for x in files]
        assert len(names)==len(set(names))
        jpg=[x for x in names if PurePosixPath(x).suffix.lower() in ['.jpg','.jpeg']]
        assert len(jpg)==5593 and len(files)==5593
        zipmeta=dict(entries=len(infos),directories=len(infos)-len(files),JPEGs=len(jpg),other_files=0,
            compressed_payload_bytes=sum(x.compress_size for x in files),uncompressed_payload_bytes=sum(x.file_size for x in files),
            directory_paths_unique=True,CRC_read_from_central_directory_only=True,actual_CRC_or_image_decode_performed=False)
    allpath=SRC/'annotations/loco-all-v1.json';allc=load(allpath)
    images=allc['images'];annotations=allc['annotations'];categories=allc['categories']
    assert len(images)==5097 and len(annotations)==151428
    byid={im['id']:im for im in images};bypath={norm(im['path']):im for im in images}
    assert len(byid)==len(bypath)==5097
    assert len({a['id'] for a in annotations})==len(annotations)
    assert all(a['image_id'] in byid for a in annotations)
    assert all(PurePosixPath(im['path']).name==im['file_name'] for im in images)
    missing=sorted(set(bypath)-set(jpg));unannotated=sorted(set(jpg)-set(bypath));assert not missing and len(unannotated)==496
    category_counts=Counter(a['category_id'] for a in annotations)
    category_images=defaultdict(set)
    shape=defaultdict(Counter)
    for ann in annotations:
        path=norm(byid[ann['image_id']]['path']);shape[path][shape_payload(ann)]+=1
        category_images[ann['category_id']].add(path)
    subset_records=[];union=set();union_annotations=0;id_changed=0;dataset_changed=0
    subset_files=sorted((SRC/'annotations').glob('loco-sub*-v1-*.json'));assert len(subset_files)==5
    for path in subset_files:
        doc=load(path);si={im['id']:im for im in doc['images']};sp={norm(im['path']):im for im in doc['images']}
        assert len(si)==len(sp)==len(doc['images'])
        assert not union.intersection(sp),'Subsets overlap image paths'
        assert doc['categories']==categories
        local=defaultdict(Counter)
        for ann in doc['annotations']:
            assert ann['image_id'] in si
            local[norm(si[ann['image_id']]['path'])][shape_payload(ann)]+=1
        for key,im in sp.items():
            assert key in bypath and im['width']==bypath[key]['width'] and im['height']==bypath[key]['height'] and im['file_name']==bypath[key]['file_name']
            assert local[key]==shape[key],key
        changed=sum(im['id']!=bypath[key]['id'] for key,im in sp.items())
        dchanged=sum(im.get('dataset_id')!=bypath[key].get('dataset_id') for key,im in sp.items())
        subset_records.append(dict(file=bind(path),images=len(sp),annotations=len(doc['annotations']),geometry_and_nonID_payload_multiset_differences=0,image_ID_changed_vs_all=changed,dataset_ID_changed_vs_all=dchanged))
        id_changed+=changed;dataset_changed+=dchanged;union|=set(sp);union_annotations+=len(doc['annotations'])
    assert union==set(bypath) and union_annotations==151428
    fields=Counter(k for ann in annotations for k in ann)
    image_fields=Counter(k for im in images for k in im)
    bbox_flags=Counter(str(a.get('isbbox')) for a in annotations)
    assert bbox_flags=={'True':151380,'False':48}
    exceptions=[a for a in annotations if a.get('isbbox') is False]
    seg_formats=Counter(type(a['segmentation']).__name__ for a in annotations)
    poly_shapes=Counter()
    for a in annotations:
        seg=a['segmentation']
        poly_shapes[','.join(str(len(poly)//2) for poly in seg) if isinstance(seg,list) else 'RLE_OR_OTHER']+=1
    category_names={c['id']:c['name'] for c in categories}
    exact_missing=['keypoints','num_keypoints','visibility','visible','occluded','occlusion','occlusion_state','external_occlusion','self_occlusion','amodal_mask','visible_mask','track_id','sequence_id','pose','depth','reference_xy','uncertainty_bbox']
    absent={key:dict(annotation_rows_with_field=fields[key],image_rows_with_field=image_fields[key],category_rows_with_field=sum(key in c for c in categories)) for key in exact_missing}
    assert all(not any(v.values()) for v in absent.values())
    assert all(a['metadata']=={} for a in annotations) and all(c['metadata']=={} for c in categories)
    unsupported_fields=set(fields)-{'id','image_id','category_id','segmentation','area','bbox','iscrowd','isbbox','color','metadata'}
    assert not unsupported_fields
    unrows=[dict(archive_path=x,subset=PurePosixPath(x).parts[1],state='U_UNANNOTATED_NOT_BACKGROUND',capture_family='UNCONFIRMED_FROM_PATH_ONLY') for x in unannotated]
    unfile=OUT/'INDEPENDENT_UNANNOTATED_496.csv';output_csv(unfile,unrows,list(unrows[0]))
    erows=[dict(annotation_id=a['id'],image_id=a['image_id'],archive_path=norm(byid[a['image_id']]['path']),category_id=a['category_id'],category=category_names[a['category_id']],isbbox=False,single_four_corner_axis_aligned=rectangle(a['segmentation']),polygon_vertex_counts=json.dumps([len(p)//2 for p in a['segmentation']]),interpretation='Official isbboxFalse exception; not certified visible/amodal alpha or semantic D/E/S') for a in exceptions]
    efile=OUT/'INDEPENDENT_ISBBOX_FALSE_48.csv';output_csv(efile,erows,list(erows[0]))
    current_audit=SRC/'ACTUAL_COCO_SCHEMA_AUDIT.json';published=load(current_audit)
    report=dict(schema='V73_INDEPENDENT_OFFICIAL_COCO_ZIP_METADATA_INVENTORY_V1',completed=True,PASS=True,
        code=bind(__file__),archive=bind(archive),archive_bytes=archive.stat().st_size,archive_directory=zipmeta,
        annotations_all=bind(allpath),all_images=5097,all_annotations=151428,all_annotated_paths_in_archive=True,
        filename_basename_matches_path=5097,unannotated_archive_RGBs_retained_U=496,unannotated_by_subset=dict(Counter(r['subset'] for r in unrows)),
        unannotated_not_negative_or_occluded_or_self=True,unannotated_manifest=bind(unfile),all_images_have_annotations=sum(bool(shape[x]) for x in bypath),
        category_counts={category_names[k]:dict(annotations=v,images=len(category_images[k])) for k,v in category_counts.items()},
        subset_union_equals_all_paths=True,subset_annotations_sum=union_annotations,subset_payload_multiset_differences=0,subsets=subset_records,
        image_ID_changed_all_vs_subset=id_changed,dataset_ID_changed_all_vs_subset=dataset_changed,
        ID_interpretation='Same source path and identical annotation payloads have different exported IDs. image_id/dataset_id are export linkage keys, not a validated capture-family/session partition; path directories are only candidate lineage until verified.',
        image_fields=dict(image_fields),annotation_fields=dict(fields),explicit_absent_fields=absent,annotation_and_category_metadata_all_empty=True,
        segmentation_formats=dict(seg_formats),polygon_vertex_count_patterns=dict(poly_shapes),isbbox_counts=dict(bbox_flags),
        single_four_corner_axis_aligned_rectangles=sum(rectangle(a['segmentation']) for a in annotations),
        isbboxFalse_single_four_corner_rectangles=sum(rectangle(a['segmentation']) for a in exceptions),isbboxFalse_by_category=dict(Counter(category_names[a['category_id']] for a in exceptions)),isbboxFalse_manifest=bind(efile),
        annotation_interpretation='COCO instance category/bbox/segmentation exists. Segmentation is predominantly a bbox rectangle; 48 isbboxFalse rows remain explicit exceptions, not proof that all masks are bboxes or that exception masks certify opaque pixels. No canonical keypoints, reference semantics, visible/amodal decomposition, or external/self occlusion labels.',
        existing_intake_schema_audit=bind(current_audit),existing_intake_schema_audit_keys=list(published),
        sample_lock=bind(SRC/'FIXED_20_RGB_SELECTION_LOCK.json'),sample_lock_used_for_selection=False,
        image_bytes_opened_from_archive=0,images_extracted_or_decoded_or_viewed=0,current_RGB_GT_weights_model_forward_fit_PnP=0,old_intake_or_V72_files_modified=False,
        admission='Metadata feasibility only. No semantic-cue, opaque cutout, split-family, or model-training admission conferred.')
    write=lambda path,value:Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write(result,report)
    print(json.dumps(dict(PASS=True,result=bind(result),counts=report['category_counts'],ID_changed=id_changed,dataset_ID_changed=dataset_changed,single4rect=report['single_four_corner_axis_aligned_rectangles'],exceptions4rect=report['isbboxFalse_single_four_corner_rectangles'])))


if __name__=='__main__':main()
