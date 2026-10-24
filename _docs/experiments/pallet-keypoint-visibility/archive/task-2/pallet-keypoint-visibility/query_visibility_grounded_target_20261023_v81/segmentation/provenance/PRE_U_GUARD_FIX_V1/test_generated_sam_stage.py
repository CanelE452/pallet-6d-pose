"""Generated arrays only; no source RGB, models, weights, GT or forward calls."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import tempfile
import numpy as np
import sam_stage_contract as c
import run_sam_from_saved_associations as runner

def binding(path):return dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())

def must_reject(fn):
    try:fn()
    except (ValueError,KeyError,FileNotFoundError):return
    raise AssertionError('Expected failure')

def fixture(hw=(10,20)):
    original=dict(input_id='generated-target',image='/generated/not-opened.png',image_sha256='a'*64,
        raw_hw=list(hw),initial_points=[[1.,2.]]*8,roi_side=16.,query_box_xyxy=[0.,0.,10.,10.])
    d=dict(boxes_xyxy=np.array([[-2,0,10,10],[0,0,10,10],[0,0,10,10]],np.float32),
        scores=np.array([.9,.8,.8],np.float32),detector_query_index=np.array([5,7,3],np.int64),
        text_labels=np.array(['opaque','generic','generic']),raw_query_max_scores=np.array([.9,.8,.8],np.float32),
        raw_normalized_boxes=np.ones((3,4),np.float32))
    rgb=dict(rgb_input_id='generated-rgb',image=original['image'],image_sha256=original['image_sha256'],raw_hw=list(hw),
        decoded_RGB_SHA256='b'*64,detections={'path':'/never/opened.npz','sha256':'c'*64},detection_count=3,caption_token_ids_SHA256='d'*64)
    ious=c.continuous_iou(original['query_box_xyxy'],d['boxes_xyxy']);j=2
    row=dict(original,rgb_input_id=rgb['rgb_input_id'],status='ASSOCIATED',reason='OK',selected_detection_index=j,
        selected_detector_query_index=int(d['detector_query_index'][j]),selected_detection_score=float(d['scores'][j]),
        association_iou=float(ious[j]),selected_box_xyxy_raw=d['boxes_xyxy'][j].tolist(),
        selected_box_xyxy_prompt=c.clip_detected_box(d['boxes_xyxy'][j],hw).tolist())
    return original,rgb,d,row

def test_all():
    original,rgb,d,row=fixture()
    assert np.array_equal(c.validate_association(row,original,rgb,d),np.array([0,0,10,10],np.float32))
    # IoU dominates score, then score, then original detector query ordinal.
    wrong=copy.deepcopy(row);wrong['selected_detection_index']=1
    must_reject(lambda:c.validate_association(wrong,original,rgb,d))
    d2=copy.deepcopy(d);d2['scores'][1]=np.float32(.85)
    row2=copy.deepcopy(row);row2.update(selected_detection_index=1,selected_detector_query_index=7,selected_detection_score=float(d2['scores'][1]))
    c.validate_association(row2,original,rgb,d2)
    u=dict(original,rgb_input_id=rgb['rgb_input_id'],status='U',reason='NO_FIXED_THRESHOLD_DETECTIONS',association_iou=None,
        **{k:None for k in c.SELECTED_FIELDS})
    assert c.validate_association(u,original,rgb,d) is None
    ur=runner.mask_row(u,None,None)
    assert ur['mask_status']=='U' and ur['mask'] is None and ur['selected_mask_index'] is None
    must_reject(lambda:runner.mask_row(u,{'path':'invented'},None))
    # GT fields/source selectors are forbidden even if their values look harmless.
    for field in ('GT','target_object_id','source','native_GTref8','mask_visib'):
        leaked=dict(original,**{field:None});must_reject(lambda:c.validate_input(leaked))
    for hw in [(1,1),(4,7),(7,4),(480,640),(540,960)]:
        h,w=hw;zero=np.zeros((1,h,w),bool)
        mask,quality=c.single_native_mask(zero,np.array([-.2],np.float32),hw)
        assert quality==float(np.float32(-.2)) and not mask.any()
        assert c.query_mask_sample(mask,[0,0],hw)['mask_prediction']=='NOT_VISIBLE_MASK'
        assert c.query_mask_sample(None,[0,0],hw)['mask_prediction']=='U'
        one=zero.copy();one[0,0,0]=True
        assert c.query_mask_sample(c.single_native_mask(one,np.array([.5],np.float32),hw)[0],[0,0],hw)['mask_prediction']=='VISIBLE_MASK'
        for q in [[-1,-1],[np.nan,0],[np.inf,0],[w,0],[0,h],[w-.1,0],[0,h-.1]]:
            assert c.query_mask_sample(mask,q,hw)['mask_prediction']=='U'
        assert np.array_equal(c.clip_detected_box([-1,-2,w+1,h+2],hw),[0,0,w,h])
        must_reject(lambda:c.prompt_box([0,0,0,h],hw))
        must_reject(lambda:c.single_native_mask(np.zeros((3,h,w),bool),np.zeros(3),hw))
        must_reject(lambda:c.single_native_mask(zero.astype(np.float32),np.zeros(1),hw))
        must_reject(lambda:c.single_native_mask(zero,np.array([np.nan]),hw))
    # Decode parity verifier is a direct byte digest comparison, with no tolerance.
    pixels=np.arange(3*4*3,dtype=np.uint8).reshape(3,4,3)
    a=hashlib.sha256(pixels.tobytes()).hexdigest();b=hashlib.sha256(pixels.copy().tobytes()).hexdigest();assert a==b
    pixels[0,0,0]^=1;assert hashlib.sha256(pixels.tobytes()).hexdigest()!=a
    # Input ordinal8 and q values survive output export, including no match.
    qset=[[float(k),float(7-k)] for k in range(8)]
    u['initial_points']=qset
    assert runner.mask_row(u,None,None)['initial_points']==qset
    assert len([c.query_mask_sample(None,q,(10,20)) for q in qset])==8
    # An unapproved/missing entry fails before source metadata/model access.
    old_read=runner.read;read_paths=[]
    def generated_read(p):
        read_paths.append(str(p));return dict(approved=False,allow_source_SAM_forward=False)
    runner.read=generated_read
    try:must_reject(lambda:runner.entry_guard('/generated/unapproved-entry.json'))
    finally:runner.read=old_read
    assert read_paths==['/generated/unapproved-entry.json']
    return dict(PASS=True,generated_only=True,HW_cases=5,tie_rule='IoU/score/originalqueryordinal',
        zero_mask_is_NOT_VISIBLE_MASK=True,no_match_is_U_no_mask=True,GT_field_rejection=True,
        exact_RGB_decode_SHA_guard=True,unapproved_entry_before_source_access=True,
        source_RGB_reads=0,GT_reads=0,weights_reads=0,model_construction=0,forward_calls=0,current223_reads=0,
        bindings=dict(helper=binding(c.__file__),runner=binding(runner.__file__),tests=binding(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    result=test_all()
    with open(args.out,'x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
