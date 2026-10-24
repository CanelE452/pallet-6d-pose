"""V82 strict request adapter around the immutable successful V81 contract.

No model/RGB/weights I/O. The full detector trace must validate before its six
legacy association arrays are passed to the exact V81 validator.
"""
from pathlib import Path
import hashlib
import importlib.util
import numpy as np

OLD=Path(__file__).resolve().parents[2]/'query_visibility_grounded_target_20261023_v81/segmentation/sam_stage_contract.py'
OLD_SHA='34cb8d3f36eef91d61742a057e0fb974b9b9cc6429da706891303984dd873006'
assert hashlib.sha256(OLD.read_bytes()).hexdigest()==OLD_SHA
spec=importlib.util.spec_from_file_location('pinned_v81_sam_contract',OLD)
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
require=legacy.require
single_native_mask=legacy.single_native_mask
query_mask_sample=legacy.query_mask_sample
INPUT_FIELDS=legacy.INPUT_FIELDS|{'requested_category'}
REQUEST_FIELDS={'request_id','image','image_sha256','raw_hw','requested_category'}
REQUEST_OUTPUT_FIELDS=REQUEST_FIELDS|{'caption','request_key_sha256','decoded_RGB_SHA256','detections','detection_count','padding_guard'}
ASSOCIATION_FIELDS=INPUT_FIELDS|(legacy.ASSOCIATION_EXTRA-{'rgb_input_id'})|{'request_id','caption','request_key_sha256'}
TRACE_FIELDS={'raw_active_logits','raw_active_probabilities','active_token_indices','input_ids','attention_mask','raw_normalized_boxes','raw_native_boxes_xyxy','raw_query_max_scores','raw_query_ordinals','boxes_xyxy','scores','detector_query_index','text_labels'}
LEGACY_ARRAY_FIELDS={'boxes_xyxy','scores','detector_query_index','text_labels','raw_query_max_scores','raw_normalized_boxes'}
CATEGORIES=('master chef can','cracker box','sugar box','tomato soup can','mustard bottle','tuna fish can','pudding box','gelatin box','potted meat can','banana','pitcher base','bleach cleanser','bowl','mug','power drill','wood block','scissors','large marker','large clamp','extra large clamp','foam brick','pallet')

def caption(row):return row['requested_category']+'.'
def request_key(row):return hashlib.sha256((row['image_sha256']+'\n'+caption(row)).encode('utf-8')).hexdigest()

def validate_input(row):
    require(set(row)==INPUT_FIELDS,'Exact eight-field model input whitelist')
    require(row['requested_category'] in CATEGORIES,'Frozen requested category')
    return legacy.validate_input({k:v for k,v in row.items() if k!='requested_category'})

def validate_request(req,rgb):
    require(set(req)==REQUEST_FIELDS and set(rgb)==REQUEST_OUTPUT_FIELDS,'Request input/output whitelist')
    require(all(req[k]==rgb[k] for k in REQUEST_FIELDS),'Request identity or category changed')
    require(rgb['caption']==caption(req) and rgb['request_key_sha256']==request_key(req),'Exact caption/cache-key changed')
    require(req['requested_category'] in CATEGORIES,'Canonical requested category')
    legacy.native_hw(req['raw_hw'])
    return True

def group_rgb_requests(requests):
    """Embedding identity is actual file SHA, never image/category request ID."""
    by={}
    for row in requests:
        key=row['image_sha256']
        if key in by:
            require(all(by[key][k]==row[k] for k in ('image','image_sha256','raw_hw','decoded_RGB_SHA256')),'Same RGB decode differs across captions')
        else:by[key]={k:row[k] for k in ('image','image_sha256','raw_hw','decoded_RGB_SHA256')}
    return list(by.values())

def validate_full_trace(d,rgb):
    require(set(d)==TRACE_FIELDS,'Full thirteen-field detector trace required before legacy normalization')
    a={k:np.asarray(v) for k,v in d.items()}
    log,prob=a['raw_active_logits'],a['raw_active_probabilities']
    require(log.ndim==2 and log.shape==prob.shape and log.dtype==prob.dtype==np.float32,'All-query active logits/probabilities FP32[Q,A]')
    q,n=log.shape;require(q>0 and n>0,'Nonempty raw query/token dimensions')
    require(np.isfinite(log).all() and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all(),'Finite active logits/probabilities')
    ids,mask,indices=a['input_ids'],a['attention_mask'],a['active_token_indices']
    require(ids.dtype==mask.dtype==indices.dtype==np.int64,'Token/index integer64 arrays')
    require(ids.ndim==2 and ids.shape[0]==1 and ids.shape==mask.shape and 0<ids.shape[1]<=256,'Token tensor shape')
    require(np.isin(mask,[0,1]).all() and np.array_equal(indices,np.flatnonzero(mask[0])) and len(indices)==n,'Complete active-token positions including specials')
    require(np.array_equal(a['raw_query_ordinals'],np.arange(q,dtype=np.int64)),'Every original raw query ordinal preserved')
    require(a['raw_query_ordinals'].dtype==np.int64,'Raw query ordinal int64')
    for k in ('raw_normalized_boxes','raw_native_boxes_xyxy'):
        require(a[k].dtype==np.float32 and a[k].shape==(q,4) and np.isfinite(a[k]).all(),'All raw boxes FP32[Q,4]')
    require(a['raw_query_max_scores'].dtype==np.float32 and np.array_equal(a['raw_query_max_scores'],prob.max(1)),'Scores exactly from saved same-tensor active probabilities')
    keep=np.flatnonzero(a['raw_query_max_scores']>np.float32(.4))
    require(np.array_equal(a['detector_query_index'],keep),'Strict >.4 original ordinal selection')
    require(np.array_equal(a['scores'],a['raw_query_max_scores'][keep]),'Kept score parity')
    require(np.array_equal(a['boxes_xyxy'],a['raw_native_boxes_xyxy'][keep]),'Kept native box parity')
    cert=rgb['padding_guard']
    require(set(cert)=={'PASS','raw_shape','input_token_length','active_token_count','finite_active_logits','all_inactive_exact_negative_inf','finite_FP32_boxes','full_everyquery_FP32_active_logits_and_same_tensor_probs'},'Padding certificate schema')
    require(cert['PASS'] is True and cert['raw_shape']==[1,q,256] and cert['input_token_length']==ids.shape[1] and cert['active_token_count']==n,'Padding certificate counts')
    require(all(cert[k] is True for k in ('finite_active_logits','all_inactive_exact_negative_inf','finite_FP32_boxes','full_everyquery_FP32_active_logits_and_same_tensor_probs')),'Actual full-output padding guard certificate')
    # This is called only after the bound independent detector audit is PASS.
    # Its raw text decode is independently audited before this normalization.
    minimal={k:a[k] for k in LEGACY_ARRAY_FIELDS}
    count=legacy.validate_detections(minimal)
    require(count==rgb['detection_count'],'Request detection count')
    return minimal

def validate_association(row,original,request,rgb,d):
    validate_input(original);validate_request(request,rgb)
    require(set(row)==ASSOCIATION_FIELDS,'Exact V82 association field whitelist')
    require({k:row[k] for k in INPUT_FIELDS}==original,'Original q/L/box/category/order unchanged')
    require(row['request_id']==request['request_id'] and row['caption']==caption(original) and row['request_key_sha256']==request_key(original),'Association caption/request key changed')
    require(all(original[k]==request[k] for k in ('image','image_sha256','raw_hw','requested_category')),'Association request identity')
    minimal=validate_full_trace(d,rgb)
    old_original={k:v for k,v in original.items() if k!='requested_category'}
    old_row={k:v for k,v in row.items() if k not in ('requested_category','request_id','caption','request_key_sha256')}
    old_row['rgb_input_id']=row['request_id']
    old_rgb={k:rgb[k] for k in ('image','image_sha256','raw_hw','decoded_RGB_SHA256','detections','detection_count')}
    old_rgb.update(rgb_input_id=row['request_id'],caption_token_ids_SHA256=hashlib.sha256(np.asarray(d['input_ids'],np.int64).tobytes()).hexdigest())
    return legacy.validate_association(old_row,old_original,old_rgb,minimal)
