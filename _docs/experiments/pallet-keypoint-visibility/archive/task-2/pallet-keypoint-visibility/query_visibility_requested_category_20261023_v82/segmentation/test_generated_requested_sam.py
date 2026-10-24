"""Generated request traces/masks only: no source, GT, weights or model."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import numpy as np
import requested_sam_contract as c
import run_sam_from_saved_associations as runner

def binding(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def reject(fn):
    try:fn()
    except (AssertionError,ValueError,KeyError):return
    raise AssertionError('Malformed generated request accepted')

def fixture():
    target=dict(input_id='generated-input',image='/generated/no-file.png',image_sha256='a'*64,raw_hw=[10,20],initial_points=[[1.,2.]]*8,roi_side=16.,query_box_xyxy=[0.,0.,10.,10.],requested_category='pallet')
    req=dict(request_id='V82_REQUEST_000000',**{k:target[k] for k in c.REQUEST_FIELDS if k!='request_id'})
    boxes=np.array([[-2,0,10,10],[0,0,10,10],[0,0,10,10]],np.float32)
    scores=np.array([.9,.8,.8],np.float32)
    prob=np.repeat(scores[:,None],4,axis=1)
    d=dict(raw_active_logits=np.log(prob/(1-prob)).astype(np.float32),raw_active_probabilities=prob,active_token_indices=np.arange(4,dtype=np.int64),input_ids=np.array([[101,1,2,102]],np.int64),attention_mask=np.ones((1,4),np.int64),raw_normalized_boxes=np.ones((3,4),np.float32),raw_native_boxes_xyxy=boxes,raw_query_max_scores=scores,raw_query_ordinals=np.arange(3,dtype=np.int64),boxes_xyxy=boxes.copy(),scores=scores.copy(),detector_query_index=np.arange(3,dtype=np.int64),text_labels=np.array(['pallet']*3))
    rgb=dict(req,caption=c.caption(req),request_key_sha256=c.request_key(req),decoded_RGB_SHA256='b'*64,detections={'path':'/never/read.npz','sha256':'c'*64},detection_count=3,padding_guard=dict(PASS=True,raw_shape=[1,3,256],input_token_length=4,active_token_count=4,finite_active_logits=True,all_inactive_exact_negative_inf=True,finite_FP32_boxes=True,full_everyquery_FP32_active_logits_and_same_tensor_probs=True))
    legacy_target={k:v for k,v in target.items() if k!='requested_category'}
    ov=c.legacy.continuous_iou(target['query_box_xyxy'],boxes);j=1
    row=dict(target,request_id=req['request_id'],caption=c.caption(req),request_key_sha256=c.request_key(req),status='ASSOCIATED',reason='OK',selected_detection_index=j,selected_detector_query_index=j,selected_detection_score=float(scores[j]),association_iou=float(ov[j]),selected_box_xyxy_raw=boxes[j].tolist(),selected_box_xyxy_prompt=[0.,0.,10.,10.])
    return target,req,rgb,d,row

def test_all():
    target,req,rgb,d,row=fixture()
    assert np.array_equal(c.validate_association(row,target,req,rgb,d),[0,0,10,10])
    for key in ('reference_xy','GT','source','object_id','mask','pose'):
        reject(lambda:c.validate_input(dict(target,**{key:None})))
    for key in c.TRACE_FIELDS:
        bad={k:v for k,v in d.items() if k!=key}
        reject(lambda:c.validate_full_trace(bad,rgb))
    for key in ('raw_active_logits','raw_active_probabilities'):
        bad=copy.deepcopy(d);bad[key]=bad[key].astype(np.float16)
        reject(lambda:c.validate_full_trace(bad,rgb))
    for key in ('requested_category','caption','request_key_sha256'):
        bad=dict(row,**{key:'incorrect'});reject(lambda:c.validate_association(bad,target,req,rgb,d))
    bad=copy.deepcopy(d);bad['scores'][0]=.5;reject(lambda:c.validate_full_trace(bad,rgb))
    bad=copy.deepcopy(d);bad['active_token_indices'][1]=3;reject(lambda:c.validate_full_trace(bad,rgb))
    bad=copy.deepcopy(d);bad['raw_query_ordinals']=bad['raw_query_ordinals'][::-1];reject(lambda:c.validate_full_trace(bad,rgb))
    selected=('selected_detection_index','selected_detector_query_index','selected_detection_score','association_iou','selected_box_xyxy_raw','selected_box_xyxy_prompt')
    u=dict(row,status='U',reason='NO_FIXED_THRESHOLD_DETECTIONS',**{k:None for k in selected})
    reject(lambda:c.validate_association(u,target,req,rgb,d))
    empty=copy.deepcopy(d);empty['raw_active_probabilities'][:]=.1;empty['raw_active_logits'][:]=np.float32(np.log(.1/.9));empty['raw_query_max_scores'][:]=.1
    for k in ('boxes_xyxy','scores','detector_query_index','text_labels'):empty[k]=empty[k][:0]
    ergb=dict(rgb,detection_count=0)
    assert c.validate_association(u,target,req,ergb,empty) is None
    for box,reason in ((None,'INVALID_INITIAL_QUERY_BOX'),([15,5,19,9],'MAX_INPUT_BOX_IOU_BELOW_0P1')):
        orig=dict(target,query_box_xyxy=box);out=dict(u,query_box_xyxy=box,reason=reason)
        assert c.validate_association(out,orig,req,rgb,d) is None
    off=copy.deepcopy(d);off['boxes_xyxy'][:]=[-10,0,-1,10];off['raw_native_boxes_xyxy'][:]=[-10,0,-1,10]
    orig=dict(target,query_box_xyxy=[-10,0,-1,10]);out=dict(u,query_box_xyxy=orig['query_box_xyxy'],reason='INVALID_CLIPPED_SAM_PROMPT')
    assert c.validate_association(out,orig,req,rgb,off) is None
    assert runner.mask_row(u,None,None)['mask'] is None
    reject(lambda:runner.mask_row(u,dict(path='fake'),None))
    other=dict(rgb,request_id='V82_REQUEST_000001',requested_category='mug',caption='mug.')
    assert len(c.group_rgb_requests([rgb,other]))==1
    reject(lambda:c.group_rgb_requests([rgb,dict(other,decoded_RGB_SHA256='d'*64)]))
    reject(lambda:c.group_rgb_requests([rgb,dict(other,raw_hw=[11,20])]))
    for hw in ((1,1),(4,7),(480,640),(540,960)):
        h,w=hw;binary=(np.arange(h*w).reshape(1,h,w)%2).astype(np.float32)
        mask,_=c.single_native_mask(binary,np.array([.1],np.float32),hw)
        assert mask.dtype==np.bool_ and np.array_equal(mask,binary[0].astype(bool))
        for dtype in (np.float64,np.float16,np.bool_,np.uint8):reject(lambda:c.single_native_mask(binary.astype(dtype),np.ones(1),hw))
        for value in (.5,np.nan,np.inf,-1,2):
            bad=binary.copy();bad[0,0,0]=value;reject(lambda:c.single_native_mask(bad,np.ones(1),hw))
        assert c.query_mask_sample(None,[0,0],hw)['mask_prediction']=='U'
        assert c.query_mask_sample(mask,[w-.1,0],hw)['mask_prediction']=='U'
    previous=runner.read;paths=[]
    def fake_read(p):paths.append(str(p));return dict(approved=False,allow_source_SAM_forward=False)
    runner.read=fake_read
    try:reject(lambda:runner.entry_guard('/generated/unapproved.json'))
    finally:runner.read=previous
    assert paths==['/generated/unapproved.json']
    return dict(PASS=True,generated_only=True,full_thirteen_field_trace_before_normalization=True,caption_category_request_key_join=True,same_RGB_different_caption_one_embedding=True,decode_or_HW_mismatch_rejected=True,all_four_U_reasons_and_fabricated_U_rejection=True,exact_binary_F32_API_and_native_query_sampler_reused=True,unapproved_before_source=True,source_RGB_GT_weights_model_forward_fit_current_PnP=0,bindings=dict(runner=binding(runner.__file__),helper=binding(c.__file__),legacy_helper=binding(c.OLD),tests=binding(__file__)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    result=test_all()
    with open(a.out,'x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
