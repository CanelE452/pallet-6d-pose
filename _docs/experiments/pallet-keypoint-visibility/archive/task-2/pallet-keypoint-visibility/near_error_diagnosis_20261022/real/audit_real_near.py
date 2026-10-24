"""Frozen-output/RGB/annotation diagnostics; never creates pixel ground truth."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RUNS = PROJECT.parent
REPO = Path('/home/minjae/Documents/github/pallet-pose')
SOURCES = {
    'manifest': PROJECT/'evaluation_preparation_20261022/PREPARED_QUERY_MANIFEST.csv',
    'features_metadata': RUNS/'external_ray_proxy_head_20261022_v60/REAL_FEATURE_METADATA.json',
    'sidecar': REPO/'_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json',
    'review_manifest': REPO/'_docs/experiments/pallet_static_registry_review_20261003_v1/review/STATIC_REVIEW_MANIFEST.json',
    'roi_source': RUNS/'external_ray_proxy_head_20261022_v60/head_and_roi.py',
    'human_review_instructions': REPO/'_docs/experiments/pallet_static_registry_review_20261003_v1/CORNER_VISIBILITY_MINIMAL_KO.md',
}


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with open(HERE/name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(name, value):
    (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


rows = read_csv(SOURCES['manifest'])
metadata = {r['id']:r for r in json.loads(SOURCES['features_metadata'].read_text())}
sidecar = json.loads(SOURCES['sidecar'].read_text())
human = {(r['population'], r['frame_id'], int(r['corner_id'])):r for r in sidecar['rows']}
review = json.loads(SOURCES['review_manifest'].read_text())
near = []
images = {}
for r in rows:
    if r['human_state'] != 'DIRECT_VISIBLE' or not r['initial_reference_error_px'] or float(r['initial_reference_error_px']) > 3:
        continue
    fid = r['id']
    pop, source_id = ('DEV319', fid) if r['group'] == 'WOOD_PRIMARY105' else ('GREEN0918', fid.split(':')[1])
    h = human[(pop, source_id, int(r['reference_corner_id']))]
    assert h['category'] == 'DIRECT_VISIBLE'
    assert r['legacy_direct_score_comparable_location'] == 'True'
    assert r['paired_same_query_RGB_D_vs_E'] == 'True'
    if fid not in images:
        images[fid] = np.asarray(Image.open(r['image']).convert('RGB'))
    image = images[fid]
    qx, qy, rx, ry = [float(r[k]) for k in ('query_x','query_y','human_reference_x','human_reference_y')]
    x, y = math.floor(qx+.5), math.floor(qy+.5)
    target_x, target_y = math.floor(rx+.5), math.floor(ry+.5)
    local = image[max(0,y-1):min(image.shape[0],y+2),max(0,x-1):min(image.shape[1],x+2)].astype(float)
    lum = local@np.array([.2126,.7152,.0722])
    gain = metadata[fid]['gain']
    r = dict(r)
    r.update(
        rounded_query_x=x, rounded_query_y=y,
        rounded_reference_x=target_x, rounded_reference_y=target_y,
        same_nearest_pixel=(x,y)==(target_x,target_y),
        reference_source=h['reference_source'],
        annotation_coordinate_source=h['label_evidence'].get('annotation_coordinate_source','UNAVAILABLE'),
        reference_coordinate_origin=h['label_evidence'].get('reference_coordinate_origin','UNAVAILABLE'),
        label_input_status=h['label_evidence'].get('human_input_status','UNAVAILABLE'),
        human_identity_confirmed=h['label_evidence'].get('human_identity_confirmed','UNAVAILABLE'),
        model_prediction_exposure=h['label_evidence'].get('model_prediction_exposure','UNAVAILABLE'),
        reference_overlay_exposure=h['label_evidence'].get('reference_overlay_exposure','UNAVAILABLE'),
        gain=gain,
        nominal_P3_stride_raw_px=8/gain,
        error_in_nominal_P3_stride=float(r['initial_reference_error_px'])/(8/gain),
        roi_sampling_step_raw_px=float(r['v61_roi_side_raw_px'])/9,
        local_3x3_luminance_mean=float(lum.mean()),
        local_3x3_luminance_std=float(lum.std()),
        local_3x3_luminance_min=float(lum.min()),
        local_3x3_luminance_max=float(lum.max()),
        local_RGB_stat_status='descriptive only; never segmentation or presence ground truth',
    )
    near.append(r)
write_csv('NEAR_DIRECT_ROWS_WITH_PROVENANCE.csv',near)

strata=[]
for grouping in ('all','recording','native_corner','error_bin','same_nearest_pixel','annotation_coordinate_source','side_bin','local_luminance_bin'):
    for group in ('WOOD_PRIMARY105','SQUARE118'):
        cohort = [r for r in near if r['group']==group]
        def level(r):
            if grouping=='all': return 'ALL'
            if grouping=='error_bin':
                e=float(r['initial_reference_error_px'])
                return '[0,0.5]' if e<=.5 else '(0.5,1]' if e<=1 else '(1,2]' if e<=2 else '(2,3]'
            if grouping=='side_bin':
                s=float(r['v61_roi_side_raw_px'])
                return '<32' if s<32 else '[32,64)' if s<64 else '[64,128)' if s<128 else '>=128'
            if grouping=='local_luminance_bin':
                b=r['local_3x3_luminance_mean']
                return '<50' if b<50 else '[50,100)' if b<100 else '[100,150)' if b<150 else '>=150'
            return str(r[grouping])
        for v in sorted({level(r) for r in cohort}):
            rr = [r for r in cohort if level(r)==v]
            tp = sum(r['present_at_fixed_0_5']=='True' for r in rr)
            strata.append(dict(group=group,stratification=grouping,stratum=v,n=len(rr),v61_present=tp,v61_absent=len(rr)-tp,semantic_D_recall=tp/len(rr),median_saved_probability=float(np.median([float(r['presence_probability']) for r in rr])),median_side_raw_px=float(np.median([float(r['v61_roi_side_raw_px']) for r in rr])),median_local_luminance=float(np.median([r['local_3x3_luminance_mean'] for r in rr]))))
write_csv('NEAR_DIRECT_STRATA.csv',strata)

summary={
    'scope':'saved outputs, original RGB and historical human semantic labels only',
    'new_training':0,'new_model_forward':0,'new_PnP':0,'new_real_pixel_ground_truth':0,
    'near_direct_n':len(near),
    'same_nearest_pixel_n':sum(r['same_nearest_pixel'] for r in near),
    'same_nearest_pixel_v61_absent':sum(r['same_nearest_pixel'] and r['present_at_fixed_0_5']=='False' for r in near),
    'all_near_paired_same_query_RGB':True,
    'groups':{},
    'label_contract':{
        'human_target':'Existing reference-corner semantic visibility, not occupancy segmentation of a pixel.',
        'v61_target':'Final visible target M4 occupancy at nearest initial-query pixel.',
        'near_distance':'Distance to existing human reference location; <=3px does not certify which physical instance owns the nearest pixel.',
        'real_pixel_reference':'No verified real target-instance pixel mask exists in these sources; exact-pixel transfer accuracy and mismatch fraction are unidentifiable.',
        'source_provenance':'Visibility sidecar uses old frozen coordinates and does not create a new independent 6D or subpixel reference.',
        'review_rules':review['review_rules'],
    },
    'architecture_evidence':{
        'input':'Frozen YOLO P3 64 channels plus valid channel, bilinear sampled to 9x9.',
        'head':'two Conv3x3/GELU, global average pooling, linear scalar; no raw RGB central-pixel branch',
        'interpretation':'Nominal P3 stride and context pooling make <=3 raw pixels sub-feature-scale. This is a capability concern, not a proved failure cause.',
    },
    'visual_review':'16 examples directly inspected via view_image; 12 deterministic strata and 4 low-error supplements. See VISUAL_REVIEW.csv. Diagnostic comments are not pixel GT.',
}
for g in ('WOOD_PRIMARY105','SQUARE118'):
    rr=[r for r in near if r['group']==g]
    pp=[r for r in rr if r['present_at_fixed_0_5']=='True']
    summary['groups'][g]={
        'n':len(rr),'v61_present':len(pp),'v61_absent':len(rr)-len(pp),'semantic_D_recall':len(pp)/len(rr),
        'same_nearest_pixel_n':sum(r['same_nearest_pixel'] for r in rr),
        'same_nearest_pixel_v61_absent':sum(r['same_nearest_pixel'] and r['present_at_fixed_0_5']=='False' for r in rr),
        'reference_sources':dict(Counter(r['reference_source'] for r in rr)),
        'annotation_coordinate_sources':dict(Counter(r['annotation_coordinate_source'] for r in rr)),
        'manual_coordinate_only':{
            'n':sum(r['annotation_coordinate_source']=='manual_click' for r in rr),
            'v61_present':sum(r['annotation_coordinate_source']=='manual_click' and r['present_at_fixed_0_5']=='True' for r in rr),
            'v61_absent':sum(r['annotation_coordinate_source']=='manual_click' and r['present_at_fixed_0_5']=='False' for r in rr),
            'same_nearest_pixel_n':sum(r['annotation_coordinate_source']=='manual_click' and r['same_nearest_pixel'] for r in rr),
            'same_nearest_pixel_v61_absent':sum(r['annotation_coordinate_source']=='manual_click' and r['same_nearest_pixel'] and r['present_at_fixed_0_5']=='False' for r in rr),
        },
        'human_identity_confirmed':dict(Counter(str(r['human_identity_confirmed']) for r in rr)),
        'error_P3_stride_percentiles':np.percentile([r['error_in_nominal_P3_stride'] for r in rr],[0,25,50,75,100]).tolist(),
        'P3_stride_raw_px_values':sorted({r['nominal_P3_stride_raw_px'] for r in rr}),
        'feature_valid_sample_count':dict(Counter(r['roi_valid_sample_count'] for r in rr)),
    }
write_json('REAL_NEAR_AUDIT.json',summary)
write_json('REAL_SOURCE_BINDINGS.json',{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in SOURCES.items()})
print(json.dumps(summary,ensure_ascii=False,indent=2))
