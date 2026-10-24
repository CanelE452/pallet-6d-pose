"""GT-free saved-detection/box/mask schema checks; no model or file I/O."""
import numpy as np

INPUT_FIELDS={'input_id','image','image_sha256','raw_hw','initial_points','roi_side','query_box_xyxy'}
ASSOCIATION_EXTRA={'rgb_input_id','status','reason','selected_detection_index',
    'selected_detector_query_index','selected_detection_score','association_iou',
    'selected_box_xyxy_raw','selected_box_xyxy_prompt'}
RGB_FIELDS={'rgb_input_id','image','image_sha256','raw_hw','decoded_RGB_SHA256',
            'detections','detection_count','caption_token_ids_SHA256'}
SELECTED_FIELDS=('selected_detection_index','selected_detector_query_index',
    'selected_detection_score','selected_box_xyxy_raw','selected_box_xyxy_prompt')

def require(condition,message):
    if not condition:raise ValueError(message)

def native_hw(value):
    a=np.asarray(value)
    require(a.shape==(2,) and np.all(a.astype(np.int64)==a) and (a>0).all(),'Positive native H/W required')
    return tuple(map(int,a))

def prompt_box(box,hw):
    h,w=native_hw(hw);a=np.asarray(box,dtype=np.float32)
    require(a.shape==(4,) and np.isfinite(a).all(),'Finite float32 XYXY prompt required')
    require(0<=a[0]<a[2]<=w and 0<=a[1]<a[3]<=h,'SAM prompt must be clipped positive XYXY in native bounds')
    return a

def clip_detected_box(box,hw):
    h,w=native_hw(hw);a=np.asarray(box,dtype=np.float32)
    require(a.shape==(4,) and np.isfinite(a).all(),'Finite detector box required')
    a=a.copy();a[[0,2]]=np.clip(a[[0,2]],0,w);a[[1,3]]=np.clip(a[[1,3]],0,h)
    return prompt_box(a,hw)

def continuous_iou(box,boxes):
    a=np.asarray(box,np.float64);z=np.asarray(boxes,np.float32).astype(np.float64)
    require(a.shape==(4,) and z.ndim==2 and z.shape[1]==4,'IoU XYXY shape')
    require(np.isfinite(a).all() and np.isfinite(z).all(),'Finite box IoU')
    wh=np.maximum(np.minimum(a[2:],z[:,2:])-np.maximum(a[:2],z[:,:2]),0)
    intersection=wh[:,0]*wh[:,1]
    area_a=np.prod(np.maximum(a[2:]-a[:2],0));areas=np.prod(np.maximum(z[:,2:]-z[:,:2],0),axis=1)
    union=area_a+areas-intersection
    good=(z[:,2]>z[:,0])&(z[:,3]>z[:,1])
    out=np.divide(intersection,union,out=np.zeros_like(intersection),where=good&(union>0))
    out[~good]=-1
    return out

def validate_detections(d):
    required={'boxes_xyxy','scores','detector_query_index','text_labels',
              'raw_query_max_scores','raw_normalized_boxes'}
    require(set(d)==required,'Detection arrays must match GT-free whitelist')
    boxes,scores,ordinal,labels=[np.asarray(d[k]) for k in ('boxes_xyxy','scores','detector_query_index','text_labels')]
    n=len(scores)
    require(boxes.shape==(n,4) and boxes.dtype==np.float32,'Native detector box F32[D,4]')
    require(scores.shape==(n,) and scores.dtype==np.float32,'Detector scores F32[D]')
    require(ordinal.shape==(n,) and ordinal.dtype==np.int64 and (ordinal>=0).all(),'Detector ordinal int64[D]')
    require(labels.shape==(n,) and labels.dtype.kind=='U','Detection text labels unicode[D]')
    require(np.isfinite(boxes).all() and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all(),'Finite detector boxes/scores')
    require(len(np.unique(ordinal))==n,'One saved detection per original query ordinal')
    require(np.asarray(d['raw_query_max_scores']).dtype==np.float32 and np.isfinite(d['raw_query_max_scores']).all(),'Raw detection score audit F32')
    require(np.asarray(d['raw_normalized_boxes']).ndim==2 and np.asarray(d['raw_normalized_boxes']).shape[1]==4 and np.asarray(d['raw_normalized_boxes']).dtype==np.float32 and np.isfinite(d['raw_normalized_boxes']).all(),'Raw box audit F32[N,4]')
    return n

def validate_input(row):
    require(set(row)==INPUT_FIELDS,'Input row whitelist')
    require(isinstance(row['input_id'],str) and bool(row['input_id']),'Opaque input_id')
    require(isinstance(row['image'],str) and isinstance(row['image_sha256'],str) and len(row['image_sha256'])==64,'RGB path/SHA')
    native_hw(row['raw_hw'])
    q=np.asarray(row['initial_points'],np.float64)
    require(q.shape==(8,2),'Exactly unchanged ordered8queries')
    require(np.isfinite(float(row['roi_side'])) and float(row['roi_side'])>0,'Positive unchanged L')
    if row['query_box_xyxy'] is not None:
        box=np.asarray(row['query_box_xyxy'],np.float64)
        require(box.shape==(4,),'Initial query box shape; nonfinite boxes remain U')
    return q

def validate_association(row,original,rgb,d):
    validate_input(original)
    require(set(row)==INPUT_FIELDS|ASSOCIATION_EXTRA,'Association GT-free whitelist')
    require({k:row[k] for k in INPUT_FIELDS}==original,'Original q/L/box/identity changed')
    require(set(rgb)==RGB_FIELDS,'PerRGB detection metadata whitelist')
    for k in ('image','image_sha256','raw_hw'):require(row[k]==rgb[k],'Association RGB identity mismatch')
    require(row['rgb_input_id']==rgb['rgb_input_id'],'Association RGB ID mismatch')
    n=validate_detections(d)
    require(n==rgb['detection_count'],'Detection count mismatch')
    require(row['status'] in ('ASSOCIATED','U'),'Association status')
    box=original['query_box_xyxy'];b=np.asarray(box,np.float64) if box is not None else np.empty(0)
    reason=None;expected_j=None;iou=None;expected_prompt=None
    if b.shape!=(4,) or not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1]:reason='INVALID_INITIAL_QUERY_BOX'
    elif n==0:reason='NO_FIXED_THRESHOLD_DETECTIONS'
    else:
        iou=continuous_iou(b,d['boxes_xyxy'])
        order=np.lexsort((d['detector_query_index'],-d['scores'].astype(np.float64),-iou));expected_j=int(order[0])
        if float(iou[expected_j])<.1:reason='MAX_INPUT_BOX_IOU_BELOW_0P1'
        else:
            try:expected_prompt=clip_detected_box(d['boxes_xyxy'][expected_j],row['raw_hw'])
            except ValueError:reason='INVALID_CLIPPED_SAM_PROMPT'
    if reason is not None:
        require(row['status']=='U' and row['reason']==reason,'Saved U must match frozen input-only association reason')
        require(all(row[k] is None for k in SELECTED_FIELDS) and row['association_iou'] is None,'No-match must preserve no mask prompt')
        return None
    require(row['status']=='ASSOCIATED' and row['reason']=='OK','Valid saved association cannot be silently U')
    j=row['selected_detection_index']
    require(isinstance(j,int) and 0<=j<n,'Selected detection index')
    require(int(d['detector_query_index'][j])==row['selected_detector_query_index'],'Selected original query ordinal changed')
    require(float(d['scores'][j])==row['selected_detection_score'],'Saved detector score changed')
    raw=np.asarray(row['selected_box_xyxy_raw'],np.float32)
    require(np.array_equal(raw,d['boxes_xyxy'][j]),'Selected raw detector box changed')
    prompt=prompt_box(row['selected_box_xyxy_prompt'],row['raw_hw'])
    require(np.array_equal(prompt,clip_detected_box(raw,row['raw_hw'])),'Prompt clipping changed')
    require(expected_j==j and np.array_equal(prompt,expected_prompt),'Frozen input-only association rule not preserved')
    require(float(row['association_iou'])==float(iou[j]),'Float64 association IoU changed')
    return prompt

def single_native_mask(masks,quality,hw):
    h,w=native_hw(hw);a=np.asarray(masks);s=np.asarray(quality)
    require(a.shape==(1,h,w) and a.dtype==np.bool_,'multimaskFalse requires exactly one native bool mask')
    require(s.shape==(1,) and np.isfinite(s).all(),'One predicted quality required; no mask selection')
    return np.ascontiguousarray(a[0]),float(s[0])

def query_mask_sample(mask,q,hw):
    """Exact original q only. Mask0 is NOT_VISIBLE_MASK, no causal E/S label."""
    h,w=native_hw(hw);q=np.asarray(q,np.float64)
    require(q.shape==(2,),'Query XY')
    finite=bool(np.isfinite(q).all())
    px,py=map(int,np.floor(q+.5)) if finite else (None,None)
    valid=bool(finite and 0<=q[0]<w and 0<=q[1]<h and 0<=px<w and 0<=py<h)
    if mask is None or not valid:
        return dict(pixel_x=px,pixel_y=py,native_query_valid=valid,visible_mask_value=None,mask_prediction='U')
    require(np.asarray(mask).shape==(h,w) and np.asarray(mask).dtype==np.bool_,'Native bool query mask')
    value=bool(mask[py,px])
    return dict(pixel_x=px,pixel_y=py,native_query_valid=valid,visible_mask_value=value,
                mask_prediction='VISIBLE_MASK' if value else 'NOT_VISIBLE_MASK')
