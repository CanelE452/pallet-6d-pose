"""Independent saved-only SAM contract audit: no RGB/GT/model imports/forward."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
from collections import Counter
import numpy as np

V=Path(__file__).resolve().parents[1]
SEAL=V/'segmentation/outputs/SAM_OUTPUT_SEAL.json'
SEAL_SHA='54bb5519a9f9ea36ad7cb2f85dfd0e90c3a788bad0924342889fa980f2466bd9'
OUTPUT=V/'source_audit/INDEPENDENT_SAVED_SAM_OUTPUT_AUDIT.json'
ORIGINAL7={'input_id','image','image_sha256','raw_hw','initial_points','roi_side','query_box_xyxy'}
ASSOCIATION_EXTRA={'rgb_input_id','status','reason','selected_detection_index','selected_detector_query_index',
    'selected_detection_score','association_iou','selected_box_xyxy_raw','selected_box_xyxy_prompt'}
SAM_EXTRA={'mask_status','mask_reason','mask','predicted_quality','selected_mask_index'}
READS=[]

def safe(path):
    p=Path(path)
    assert p.suffix not in {'.png','.jpg','.jpeg','.pt','.pth','.safetensors'},'RGB/weights reads prohibited'
    assert 'private_evaluation' not in p.parts and 'current223' not in str(p).lower()
    return p

def sha(path):
    p=safe(path);READS.append(str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def check(b):assert sha(b['path'])==b['sha256'],b['path']
def read(path):p=safe(path);READS.append(str(p));return json.loads(p.read_text())
def lines(path):p=safe(path);READS.append(str(p));return [json.loads(x) for x in p.read_text().splitlines() if x]
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))

def scalar_iou(a,b):
    if not(b[2]>b[0] and b[3]>b[1]):return -1.
    iw=max(0.,min(a[2],b[2])-max(a[0],b[0]));ih=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    inter=iw*ih;union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.

def expected_association(original,d):
    a=original['query_box_xyxy'];n=len(d['scores'])
    if a is None or not all(x is not None and math.isfinite(x) for x in a) or a[2]<=a[0] or a[3]<=a[1]:return 'INVALID_INITIAL_QUERY_BOX',None,None,None
    if not n:return 'NO_FIXED_THRESHOLD_DETECTIONS',None,None,None
    vals=[scalar_iou(a,[float(x) for x in b]) for b in d['boxes_xyxy']]
    j=min(range(n),key=lambda z:(-vals[z],-float(d['scores'][z]),int(d['detector_query_index'][z])))
    if vals[j]<.1:return 'MAX_INPUT_BOX_IOU_BELOW_0P1',None,None,None
    h,w=original['raw_hw'];b=np.array(d['boxes_xyxy'][j],np.float32,copy=True)
    b[[0,2]]=np.clip(b[[0,2]],0,w);b[[1,3]]=np.clip(b[[1,3]],0,h)
    if not (np.isfinite(b).all() and b[2]>b[0] and b[3]>b[1]):return 'INVALID_CLIPPED_SAM_PROMPT',None,None,None
    return 'OK',j,vals[j],b

def csv_float(s):return None if s=='' else float(s)

def main():
    assert sha(SEAL)==SEAL_SHA
    s=read(SEAL);assert s['completed'] is True and s['PASS'] is True
    assert s['source_only'] is True and s['current223'] is False and s['fit'] is False and s['PnP']==0
    # Metadata and immutable code only. Never hash/read the checkpoint body.
    for key in ['entrylock','decisions','detector_output_seal','associations','per_RGB','runtime','code','helper','generated','official_output_contract','output_masks','summary']:check(s[key])
    for b in s['source_inputs'].values():check(b)
    assert s['checkpoint']['sha256']=='a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'
    entry=read(s['entrylock']['path']);assert entry['approved'] is True and entry['allow_source_SAM_forward'] is True
    assert entry['bindings']['checkpoint']==s['checkpoint']
    assert entry['bindings']['code']==s['code'] and entry['bindings']['helper']==s['helper']
    detector=read(s['detector_output_seal']['path']);assert detector['completed'] is True and detector['PASS'] is True
    assert detector['source_inputs']==s['source_inputs'] and detector['associations']==s['associations'] and detector['per_RGB']==s['per_RGB']
    originals=lines(s['source_inputs']['manifest']['path']);unique=lines(s['source_inputs']['unique_manifest']['path'])
    assocs=lines(s['associations']['path']);rgbs=lines(s['per_RGB']['path']);rows=lines(s['output_masks']['path']);summary=read(s['summary']['path'])
    assert len(originals)==len(assocs)==len(rows)==1035 and len(unique)==len(rgbs)==435
    assert len({r['input_id'] for r in rows})==1035
    byrgb={r['rgb_input_id']:r for r in rgbs};ubyid={r['input_id']:r for r in unique};assert set(byrgb)==set(ubyid)
    detections={}
    for rgb in rgbs:
        u=ubyid[rgb['rgb_input_id']]
        assert all(rgb[k]==u[k] for k in ['image','image_sha256','raw_hw'])
        assert len(rgb['decoded_RGB_SHA256'])==64
        check(rgb['detections']);READS.append(rgb['detections']['path'])
        with np.load(safe(rgb['detections']['path']),allow_pickle=False) as z:
            detections[rgb['rgb_input_id']]={k:z[k] for k in ['boxes_xyxy','scores','detector_query_index']}
    outputs_by={b['path']:b for b in s['outputs']};assert len(outputs_by)==len(s['outputs'])
    for b in s['outputs']:check(b)
    qp=Path(s['output_masks']['path']).parent/'QUERY_MASK_SAMPLES.csv'
    assert str(qp) in outputs_by
    READS.append(str(qp))
    with qp.open() as f:qrows=list(csv.DictReader(f))
    assert len(qrows)==8280
    masks=[];qualities=[];predictions=Counter();validity=Counter();hold_reasons=Counter();associated_rgbs=set();zero=0
    for i,(original,a,row) in enumerate(zip(originals,assocs,rows)):
        assert set(original)==ORIGINAL7 and set(a)==ORIGINAL7|ASSOCIATION_EXTRA and set(row)==set(a)|SAM_EXTRA
        assert {k:a[k] for k in ORIGINAL7}==original and {k:row[k] for k in a}==a
        rgb=byrgb[a['rgb_input_id']];assert all(a[k]==rgb[k] for k in ['image','image_sha256','raw_hw'])
        d=detections[a['rgb_input_id']];reason,j,iou,prompt=expected_association(original,d)
        assert a['reason']==reason
        known=reason=='OK';mask=None
        if known:
            assert a['status']=='ASSOCIATED' and a['selected_detection_index']==j
            assert a['selected_detector_query_index']==int(d['detector_query_index'][j])
            assert a['selected_detection_score']==float(d['scores'][j])
            assert a['selected_box_xyxy_raw']==d['boxes_xyxy'][j].tolist() and a['selected_box_xyxy_prompt']==prompt.tolist()
            assert a['association_iou']==iou
            assert row['mask_status']=='OK' and row['mask_reason']=='SINGLE_BOX_NATIVE_VISIBLE_MASK'
            assert row['selected_mask_index']==0 and math.isfinite(row['predicted_quality'])
            assert row['mask']==outputs_by[row['mask']['path']]
            READS.append(row['mask']['path'])
            with np.load(safe(row['mask']['path']),allow_pickle=False) as z:
                assert z.files==['visible_mask'];mask=z['visible_mask']
            assert mask.dtype==np.bool_ and list(mask.shape)==original['raw_hw']
            masks.append(row['mask']);qualities.append(row['predicted_quality']);zero+=int(not mask.any())
            associated_rgbs.add(a['rgb_input_id'])
        else:
            assert a['status']=='U' and row['mask_status']=='U'
            assert all(a[k] is None for k in ['selected_detection_index','selected_detector_query_index','selected_detection_score','association_iou','selected_box_xyxy_raw','selected_box_xyxy_prompt'])
            assert row['mask_reason']=='NO_ASSOCIATED_BOX:'+reason
            assert row['mask'] is None and row['predicted_quality'] is None and row['selected_mask_index'] is None
            hold_reasons[reason]+=1
        h,w=original['raw_hw']
        for k,q in enumerate(original['initial_points']):
            csvrow=qrows[i*8+k]
            assert int(csvrow['input_index'])==i and int(csvrow['native_corner'])==k and csvrow['input_id']==original['input_id']
            assert csv_float(csvrow['query_x'])==q[0] and csv_float(csvrow['query_y'])==q[1]
            assert csvrow['mask_status']==row['mask_status']
            finite=all(v is not None and math.isfinite(v) for v in q)
            px,py=(math.floor(q[0]+.5),math.floor(q[1]+.5)) if finite else (None,None)
            good=finite and 0<=q[0]<w and 0<=q[1]<h and 0<=px<w and 0<=py<h
            assert csv_float(csvrow['pixel_x'])==px and csv_float(csvrow['pixel_y'])==py
            assert csvrow['native_query_valid']==str(bool(good))
            expected='U' if not known or not good else ('VISIBLE_MASK' if mask[py,px] else 'NOT_VISIBLE_MASK')
            assert csvrow['mask_prediction']==expected
            assert csvrow['visible_mask_value']==('' if expected=='U' else str(expected=='VISIBLE_MASK'))
            predictions[expected]+=1;validity[(a['status'],bool(good))]+=1
    assert len(masks)==823 and 1035-len(masks)==212 and len(associated_rgbs)==345
    assert summary['associated']==823 and summary['unassociated']==212 and summary['embedding_calls']==345
    assert summary['zero_visible_masks']==zero and summary['query_predictions']==dict(predictions)
    assert summary['state_before']==summary['state_after']
    assert all(summary[k] is False for k in ['multimask_output','return_logits','quality_selection','points','postprocessing','compile','current223','fit'])
    assert summary['selected_mask_index']==0 and summary['mask_threshold']==0 and summary['PnP']==0
    expected_paths={s['output_masks']['path'],str(qp),s['summary']['path']}|{b['path'] for b in masks}
    assert set(outputs_by)==expected_paths,'Every mask/manifest/queryCSV/summary covered once by seal'
    # Code-bound decode certificates: raw pixels are intentionally not reread here.
    source=safe(s['code']['path']).read_text();tree=ast.parse(source)
    pred_calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='predictor' and n.func.attr=='predict']
    assert len(pred_calls)==1
    kw={k.arg:ast.literal_eval(k.value) for k in pred_calls[0].keywords if k.arg in ['point_coords','point_labels','multimask_output','return_logits']}
    assert kw==dict(point_coords=None,point_labels=None,multimask_output=False,return_logits=False)
    assert "require(hashlib.sha256(image.tobytes()).hexdigest()==rgb['decoded_RGB_SHA256']" in source
    assert source.index("require(hashlib.sha256(image.tobytes()).hexdigest()==rgb['decoded_RGB_SHA256']")<source.index('predictor.set_image(image)')
    certpath=V/'source_audit/SAM_DECODE_GUARD_CERTIFICATES.csv'
    with certpath.open('x') as f:
        fields=['rgb_input_id','image_sha256','raw_H','raw_W','detector_decoded_RGB_SHA256','successful_code_requires_same_SHA_before_set_image','independent_raw_RGB_recheck']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for rgb in rgbs:
            if rgb['rgb_input_id'] in associated_rgbs:
                writer.writerow(dict(rgb_input_id=rgb['rgb_input_id'],image_sha256=rgb['image_sha256'],raw_H=rgb['raw_hw'][0],raw_W=rgb['raw_hw'][1],detector_decoded_RGB_SHA256=rgb['decoded_RGB_SHA256'],successful_code_requires_same_SHA_before_set_image=True,independent_raw_RGB_recheck=False))
    failed=V/'segmentation/provenance/FAILED_BINARY_FLOAT_API_V1/PRESERVATION_SEAL.json'
    fs=read(failed);check(fs['failed_receipt']);failure=read(fs['failed_receipt']['path'])
    assert failure['control_flow_counts']==dict(source_RGB_decodes=1,set_image_calls=1,predict_calls_attempted=1,successful_mask_conversions=0,saved_masks=0)
    result=dict(completed=True,PASS=True,blockers=[],audit_code=bind(__file__),sam_output_seal=bind(SEAL),
        counts=dict(targets=1035,unique_RGB=435,associated_native_bool_masks=823,unassociated_U=212,embedding_calls_successful_run=345,query_slots=8280,zero_visible_masks=zero),
        query_predictions=dict(predictions),query_validity_by_association={str(k):v for k,v in validity.items()},U_reasons=dict(hold_reasons),
        invariants=dict(original7_fields_q_L_box_and_order_exact=True,all_selected_boxes_scores_ordinals_and_U_reasons_independently_replayed=True,all_mask_file_SHAs_and_exact_native_bool_shapes=True,complete_output_seal_coverage=True,all8280_scalar_floor_plus_half_samples_match=True,quality_never_selects_mask=True,multimaskFalse_selected_index0=True,no_match_has_no_mask_not_zero_mask=True,model_state_before_after_saved_identical=True),
        predicted_quality=dict(count=len(qualities),minimum=min(qualities),maximum=max(qualities),used_for_selection=False),
        decoded_RGB_certificates=dict(associated_RGB=345,unassociated_only_RGB_not_decoded_in_SAM=90,certificate_csv=bind(certpath),evidence='Successful sealed code requires raw file SHA/HW and exact detector decoded uint8HWC SHA equality before each set_image. Independent audit verifies code/order/associated RGB set; it does not reopen RGB. No separate SAM per-RGB decoded digest was persisted.',independently_redecoded_RGB=False),
        attempts=dict(failed_first=dict(preservation=bind(failed),predict_calls=1,set_image_calls=1,saved_masks=0,reason='Technical binary float32 API guard mismatch, not model-quality failure',post_failure_state_hash_available=False),successful_completed=dict(predict_calls=823,set_image_calls=345,saved_masks=823,state_unchanged=True),total_phase=dict(SAM_predict_calls=824,set_image_calls=346),this_audit=dict(new_forward=0,new_model_or_weights_reads=0,raw_RGB_reads=0,GT_reads=0,current223_reads=0,fit=0,PnP=0)),
        semantics='Only visible-mask proxy and U. NOT_VISIBLE_MASK is not an external/self-occlusion class; no human-semantic metrics computed.',
        saved_file_read_count=len(READS),saved_unique_files_read=len(set(READS)))
    # all zero-less masks do not excuse the no-match-U check; all rows retained.
    assert result['decoded_RGB_certificates']['unassociated_only_RGB_not_decoded_in_SAM']==435-345
    with OUTPUT.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(PASS=True,path=str(OUTPUT),sha256=sha(OUTPUT),counts=result['counts'],query_predictions=result['query_predictions'])))

if __name__=='__main__':main()
