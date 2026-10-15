"""Pure injected-model contracts; no checkpoints, real images or arrays read."""
from copy import deepcopy
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from . import capture_generic_instance_mask_candidates_v1 as M


class Detector:
    def __init__(self, boxes):
        self.boxes = np.asarray(boxes, float).reshape(-1, 6)
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        assert {k:v for k,v in kwargs.items() if k != 'source'} == M.CONFIGURATION
        return [SimpleNamespace(boxes=SimpleNamespace(data=self.boxes), names={0:'person',56:'chair',60:'dining table'})]


class SAM:
    def __init__(self, masks, qualities=None):
        self.masks = np.asarray(masks).copy()
        self.qualities = np.asarray([.7,.9,.8] if qualities is None else qualities, float)
        self.images, self.calls = [], []

    def set_image(self, image, **kwargs):
        assert kwargs == {'image_format':'BGR'}
        self.images.append(image.copy())

    def predict(self, **kwargs):
        assert set(kwargs) == {'box','multimask_output'} and kwargs['multimask_output'] is True
        self.calls.append(deepcopy(kwargs))
        return self.masks.copy(), self.qualities.copy(), None


def fixture(*, overlap=False, quality=None):
    image = np.arange(10*12*3,dtype=np.uint8).reshape(10,12,3)
    target = np.zeros((10,12));target[6:9,8:11]=1
    masks = np.zeros((3,10,12),np.uint8)
    masks[0,1:3,1:3]=1;masks[1,2:5,2:5]=1;masks[2,3:6,3:6]=1
    if overlap:masks[1,7,9]=1
    saved=[]
    def save(index, values, scores):
        saved.append((index,values.copy(),scores.copy()))
        return {'path':f'fake_{index}.npz','sha256':'a'*64,'bytes':100}
    return image,target,SAM(masks,quality),saved,save


def test_two_boxes_share_one_encoder_all_three_masks_and_scores_preserved():
    image,target,sam,saved,save=fixture()
    detector=Detector([[1,1,6,7,.6,56],[2,1,7,8,.8,0]])
    result=M.infer_case(image,target,detector,sam,case_id='fixture',save_candidates=save)
    assert len(detector.calls)==1 and len(sam.images)==1 and len(sam.calls)==2
    assert result['SAM_image_encoder_calls']==1 and result['SAM_prompt_decoder_calls']==2
    assert result['returned_detections']==2 and len(saved)==2
    for row,(index,masks,quality)in zip(result['detections'],saved):
        np.testing.assert_array_equal(masks,sam.masks)
        np.testing.assert_array_equal(quality,sam.qualities)
        assert len(row['alternatives'])==3 and row['selected_mask_index']==1
        assert row['instance_id']==f'fixture/DETECTION_{index:03d}'
        assert not row['other_object_confirmed'] and not row['foreground_occlusion_confirmed']


def test_detector_order_class_confidence_and_original_box_are_not_reordered():
    image,target,sam,saved,save=fixture()
    boxes=[[2,3,7,8,.4,60],[1,1,6,7,.99,56]]
    result=M.infer_case(image,target,Detector(boxes),sam,case_id='fixture',save_candidates=save)
    assert [r['detector_class_id']for r in result['detections']]==[60,56]
    for row,box in zip(result['detections'],boxes):
        assert row['detector_box_xyxy']==box[:4] and row['detector_confidence']==box[4]


def test_empty_detection_retains_unavailable_row_without_SAM_encoder_or_fallback():
    image,target,sam,saved,save=fixture()
    result=M.infer_case(image,target,Detector([]),sam,case_id='fixture',save_candidates=save)
    assert result['status']=='UNAVAILABLE' and result['unavailable_reason']=='no_detections'
    assert result['detections']==[] and result['zero_rows_retained']
    assert not sam.images and not sam.calls and not saved


def test_low_confidence_invalid_class_and_box_are_retained_but_not_prompted():
    image,target,sam,saved,save=fixture()
    boxes=[[1,1,7,8,.2,56],[2,2,2,7,.8,56],[0,0,9,9,.9,99],[0,0,9,9,np.nan,56]]
    result=M.infer_case(image,target,Detector(boxes),sam,case_id='fixture',save_candidates=save)
    assert len(result['detections'])==4 and result['status']=='UNAVAILABLE'
    assert all(r['status']=='UNKNOWN' and not r['available'] for r in result['detections'])
    assert result['detections'][3]['detector_confidence']is None
    assert not sam.images and not saved


def test_prompt_clipping_is_audited_without_changing_original_box():
    image,target,sam,saved,save=fixture()
    result=M.infer_case(image,target,Detector([[-3,-4,20,15,.5,56]]),sam,case_id='fixture',save_candidates=save)
    row=result['detections'][0]
    assert row['detector_box_xyxy']==[-3.,-4.,20.,15.]
    assert row['SAM_box_xyxy']==[0.,0.,12.,10.] and row['prompt_box_canvas_clipped']
    np.testing.assert_array_equal(sam.calls[0]['box'],[0,0,12,10])


def test_any_target_overlap_is_only_suspected_target_not_occluder_truth():
    image,target,sam,saved,save=fixture(overlap=True)
    result=M.infer_case(image,target,Detector([[1,1,8,9,.8,56]]),sam,case_id='fixture',save_candidates=save)
    row=result['detections'][0]
    assert row['status']=='suspected_target' and row['available']
    overlap=row['alternatives'][1]['target_overlap']
    assert overlap['intersection_pixels']==1 and overlap['candidate_pixels']==10
    assert overlap['IoU']==pytest.approx(1/18)
    assert not row['other_object_confirmed'] and not row['supplied_to_existing_pipeline']


def test_no_overlap_is_candidate_with_unknown_identity_and_depth():
    image,target,sam,saved,save=fixture()
    row=M.infer_case(image,target,Detector([[1,1,8,9,.8,56]]),sam,case_id='fixture',save_candidates=save)['detections'][0]
    assert row['status']=='CANDIDATE' and row['status_reason']=='no_target_overlap_does_not_prove_other_instance'
    assert not row['foreground_occlusion_confirmed']


@pytest.mark.parametrize('change',['empty_target','empty_masks','nonfinite_quality'])
def test_uncertain_mask_context_or_quality_retains_all_alternatives_as_unknown(change):
    image,target,sam,saved,save=fixture()
    if change=='empty_target':target[:]=0
    elif change=='empty_masks':sam.masks[:]=0
    else:sam.qualities[0]=np.nan
    result=M.infer_case(image,target,Detector([[1,1,8,9,.8,56]]),sam,case_id='fixture',save_candidates=save)
    row=result['detections'][0]
    assert len(saved)==1 and len(row['alternatives'])==3 and row['status']=='UNKNOWN'
    assert result['status']=='UNAVAILABLE' and result['other_object_or_depth_confirmations']==0


def test_quality_tie_keeps_first_and_all_unchosen_masks_remain():
    image,target,sam,saved,save=fixture(quality=[.8,.8,.8])
    row=M.infer_case(image,target,Detector([[1,1,8,9,.8,56]]),sam,case_id='fixture',save_candidates=save)['detections'][0]
    assert row['selected_mask_index']==0 and len(row['alternatives'])==3
    assert len(set(x['mask_pixel_sha256']for x in row['alternatives']))==3


def test_original_image_target_and_model_arrays_immutable_even_mutating_callback():
    image,target,sam,saved,save=fixture();before=(image.copy(),target.copy(),sam.masks.copy())
    def mutating(index,masks,quality):
        masks[:]=0;quality[:]=0
        return {'path':'fake','sha256':'a'*64}
    result=M.infer_case(image,target,Detector([[1,1,8,9,.8,56]]),sam,case_id='fixture',save_candidates=mutating)
    for actual,expected in zip((image,target,sam.masks),before):np.testing.assert_array_equal(actual,expected)
    assert result['detections'][0]['alternatives'][1]['mask_pixel_sha256']==M.mask_pixel_sha256(sam.masks[1])


def test_encoder_is_reset_for_each_frame_not_reused_across_images():
    image,target,sam,saved,save=fixture();detector=Detector([[1,1,8,9,.8,56]])
    for name in ('frame0','frame1'):M.infer_case(image,target,detector,sam,case_id=name,save_candidates=save)
    assert len(sam.images)==2 and len(sam.calls)==2


def test_no_keypoint_state_pose_or_precise_prompt_input_exists():
    assert set(inspect.signature(M.infer_case).parameters)=={'image_bgr','target_mask','detector','predictor','case_id','save_candidates'}
    class Poison:
        def __array__(self,*args,**kwargs):raise AssertionError('S/U payload must not be touched')
    image,target,sam,saved,save=fixture()
    with pytest.raises(TypeError):M.infer_case(image,target,Detector([]),sam,case_id='fixture',save_candidates=save,raw_points=Poison())


def test_maximum_detector_budget_is_failclosed_not_silently_truncated():
    image,target,sam,saved,save=fixture()
    with pytest.raises(ValueError,match='declared20'):
        M.infer_case(image,target,Detector([[1,1,8,9,.8,56]]*21),sam,case_id='fixture',save_candidates=save)
    assert not sam.images and not saved


def test_wrong_SAM_count_or_nonbinary_masks_fail_before_archive():
    image,target,sam,saved,save=fixture();sam.masks=sam.masks[:2]
    with pytest.raises(ValueError,match='three'):
        M.infer_case(image,target,Detector([[1,1,8,9,.8,56]]),sam,case_id='fixture',save_candidates=save)
    assert not saved
    image,target,sam,saved,save=fixture();sam.masks=sam.masks.astype(float);sam.masks[0,0,0]=.5
    with pytest.raises(ValueError,match='binary'):
        M.infer_case(image,target,Detector([[1,1,8,9,.8,56]]),sam,case_id='fixture',save_candidates=save)
    assert not saved


def test_GO_guards_precede_any_IO_or_model_load(monkeypatch):
    def poison(*args,**kwargs):raise AssertionError('No IO')
    monkeypatch.setattr(M.C,'read',poison);monkeypatch.setattr(M.C,'sha',poison)
    monkeypatch.setattr(M,'_load_models',poison)
    with pytest.raises(PermissionError):M.prepare()
    with pytest.raises(PermissionError):M.run(expected_plan_sha256='a'*64)
    with pytest.raises(ValueError):M.run(expected_plan_sha256=None,root_actual_inference_GO=True)


def test_metadata_whitelist_extracts_only_fixed_bindings_and_rejects_extra_cases():
    chunks=[]
    for i,ident in enumerate(M.CASES):
        chunks.append('"id":'+json.dumps(ident)+',"image_binding":'+json.dumps(dict(path=f'im{i}',sha256='a'*64))+',"mask_binding":'+json.dumps(dict(path=f'mask{i}',sha256='b'*64))+',"unused_numeric_reference":POISON_NOT_DECODED')
    text='['+','.join(chunks)+']'
    rows=M.extract_case_bindings(text)
    assert tuple(r['id']for r in rows)==M.CASES
    assert set(rows[0])=={'id','partition','image_binding','target_mask_binding'}
    with pytest.raises(ValueError):M.extract_case_bindings(text+','+chunks[0])
    with pytest.raises(ValueError):M.extract_case_bindings(text.replace(M.CASES[0],'TEST:wrong'))


def test_pixel_digest_binds_shape_values_and_rejects_probability_mask():
    mask=np.eye(4,dtype=np.uint8)
    assert M.mask_pixel_sha256(mask)==M.mask_pixel_sha256(mask.astype(bool))
    assert M.mask_pixel_sha256(mask)!=M.mask_pixel_sha256(mask.reshape(2,8))
    with pytest.raises(ValueError):M.mask_pixel_sha256(mask*.5)
