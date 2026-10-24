"""Independent metadata/saved-probability math. No model, file, or source I/O."""
import hashlib
import math
import numpy as np

ORIGINAL7={'input_id','image','image_sha256','raw_hw','initial_points','roi_side','query_box_xyxy'}
TARGET8=ORIGINAL7|{'requested_category'}
UNIQUE5={'request_id','image','image_sha256','raw_hw','requested_category'}
PACKET_KEYS={'raw_active_logits','raw_active_probabilities','active_token_indices','input_ids','attention_mask',
    'raw_normalized_boxes','raw_native_boxes_xyxy','raw_query_max_scores','raw_query_ordinals',
    'boxes_xyxy','scores','detector_query_index','text_labels'}

def require(test,message):
    if not test:raise ValueError(message)

def validate_target(original,row,ontology):
    require(set(original)==ORIGINAL7 and set(row)==TARGET8,'Exact input-only field whitelists')
    require({k:row[k] for k in ORIGINAL7}==original,'Original7 q/L/box/identity changed')
    require(row['requested_category'] in ontology,'Requestedcategory canonical ontology only')
    require(np.asarray(row['initial_points'],np.float64).shape==(8,2),'Native8 order')
    h,w=row['raw_hw'];require(type(h) is int and type(w) is int and h>0 and w>0,'Positive integer native HW')
    require(math.isfinite(row['roi_side']) and row['roi_side']>0,'Original positive L')
    require(row['query_box_xyxy'] is None or np.asarray(row['query_box_xyxy']).shape==(4,),'Original querybox shape')
    require(isinstance(row['image_sha256'],str) and len(row['image_sha256'])==64,'Original RGB SHA')
    return row['requested_category']+'.'

def request_tuple(row):return row['image_sha256'],row['requested_category']+'.'
def request_key_sha256(row):
    rgb,caption=request_tuple(row)
    return hashlib.sha256((rgb+'\n'+caption).encode('utf-8')).hexdigest()

def expected_unique(rows):
    seen={};unique=[];mapping=[]
    for row in rows:
        key=request_tuple(row)
        if key not in seen:
            rid='V82_REQUEST_%06d'%len(unique);seen[key]=rid
            unique.append(dict(request_id=rid,**{k:row[k] for k in ['image','image_sha256','raw_hw','requested_category']}))
        mapping.append(seen[key])
    return unique,mapping

def validate_full_logits(full,attention,boxes):
    a=np.asarray(full);m=np.asarray(attention);b=np.asarray(boxes)
    require(a.ndim==3 and a.shape[0]==1 and a.shape[2]==256 and a.dtype==np.float32,'All-query F32[1,Q,256] logits')
    require(m.ndim==2 and m.shape[0]==1 and 0<m.shape[1]<=256 and np.isin(m,[0,1]).all(),'Attention mask')
    require(b.shape==(1,a.shape[1],4) and b.dtype==np.float32 and np.isfinite(b).all(),'Rawboxes finite F32')
    active=np.flatnonzero(m[0]).astype(np.int64);require(len(active)>0,'At least one active token')
    unused=np.ones(256,bool);unused[active]=False
    require(np.isfinite(a[0,:,active]).all(),'Active logits finite')
    require(np.isneginf(a[0,:,unused]).all(),'All nonactive logits exactly -inf')
    return active

def validate_packet(d,hw):
    require(set(d)==PACKET_KEYS,'Complete every-query raw active trace schema')
    logits=np.asarray(d['raw_active_logits']);probs=np.asarray(d['raw_active_probabilities']);active=np.asarray(d['active_token_indices'])
    ids=np.asarray(d['input_ids']);attention=np.asarray(d['attention_mask']);raw=np.asarray(d['raw_normalized_boxes'])
    require(logits.ndim==2 and logits.dtype==np.float32 and np.isfinite(logits).all(),'Rawactive logits finite F32[Q,A]')
    Q,A=logits.shape
    require(Q>0 and A>0 and probs.shape==(Q,A) and probs.dtype==np.float32 and np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all(),'Exact active probabilities F32[Q,A]')
    require(ids.ndim==2 and ids.shape[0]==1 and ids.dtype==np.int64 and attention.shape==ids.shape and attention.dtype==np.int64,'Saved full tokenizer ids/mask')
    require(0<ids.shape[1]<=256 and np.isin(attention,[0,1]).all(),'Tokenizer input length/mask')
    require(active.dtype==np.int64 and active.shape==(A,) and np.array_equal(active,np.flatnonzero(attention[0])),'Every active position exactly once and in source order')
    require(np.array_equal(d['raw_query_ordinals'],np.arange(Q,dtype=np.int64)),'Every original query ordinal saved')
    require(raw.shape==(Q,4) and raw.dtype==np.float32 and np.isfinite(raw).all(),'Every rawnormalized box F32[Q,4]')
    native=np.asarray(d['raw_native_boxes_xyxy']);require(native.shape==(Q,4) and native.dtype==np.float32 and np.isfinite(native).all(),'Every native box F32[Q,4]')
    maximum=np.max(probs,axis=1)
    require(np.asarray(d['raw_query_max_scores']).dtype==np.float32 and np.array_equal(maximum,d['raw_query_max_scores']),'Exact same-tensor active probability maxscores')
    keep=np.flatnonzero(maximum>np.float32(.4)).astype(np.int64)
    require(np.asarray(d['detector_query_index']).dtype==np.int64 and np.array_equal(d['detector_query_index'],keep),'Strict .4 keeps original ordinal order, no NMS')
    require(np.asarray(d['scores']).dtype==np.float32 and np.array_equal(d['scores'],maximum[keep]),'Kept score exact')
    require(np.asarray(d['boxes_xyxy']).dtype==np.float32 and np.array_equal(d['boxes_xyxy'],native[keep]),'Kept native boxes exact')
    require(np.asarray(d['text_labels']).shape==(len(keep),) and np.asarray(d['text_labels']).dtype.kind=='U','All kept labels, including empty labels, retained')
    # No inference: normalized-to-native arithmetic can be checked separately
    # against official FP32 Torch helper on identical device if needed.
    h,w=hw;require(h>0 and w>0,'Native HW')
    full_logits=np.full((1,Q,256),-np.inf,np.float32);full_logits[0][:,active]=logits
    full_probs=np.zeros((Q,256),np.float32);full_probs[:,active]=probs
    token_sets=[]
    for q in keep:
        positions=np.flatnonzero(full_probs[q]>np.float32(.3));positions=positions[(positions!=0)&(positions!=255)]
        require((positions<len(ids[0])).all(),'Text replay cannot index omitted/invalid token')
        token_sets.append([int(ids[0,j]) for j in positions])
    return dict(keep=keep,token_ids=token_sets,full_logits=full_logits,full_probabilities=full_probs)

def associate(initial,hw,boxes,scores,ordinals):
    """Independent Python scalar IoU and stable input-only association."""
    a=np.asarray(initial,np.float64) if initial is not None else np.empty(0)
    if a.shape!=(4,) or not np.isfinite(a).all() or a[2]<=a[0] or a[3]<=a[1]:return 'INVALID_INITIAL_QUERY_BOX',None,None
    if len(scores)==0:return 'NO_FIXED_THRESHOLD_DETECTIONS',None,None
    overlaps=[]
    for box in boxes:
        b=[float(x) for x in box]
        if b[2]<=b[0] or b[3]<=b[1]:overlaps.append(-1.);continue
        inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
        union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        overlaps.append(inter/union if union>0 else 0.)
    j=min(range(len(scores)),key=lambda k:(-overlaps[k],-float(scores[k]),int(ordinals[k])))
    if overlaps[j]<.1:return 'MAX_INPUT_BOX_IOU_BELOW_0P1',None,None
    h,w=hw;prompt=np.asarray(boxes[j],np.float32).copy();prompt[[0,2]]=np.clip(prompt[[0,2]],0,w);prompt[[1,3]]=np.clip(prompt[[1,3]],0,h)
    if not (np.isfinite(prompt).all() and prompt[2]>prompt[0] and prompt[3]>prompt[1]):return 'INVALID_CLIPPED_SAM_PROMPT',None,None
    return 'OK',j,prompt

def validate_integrity(loading,repeat,state_before,state_after):
    for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']:require(k in loading and loading[k]==[],'Complete pretrained loading')
    require(repeat is True,'First exact request repeat')
    require(state_before==state_after,'Frozen state unchanged')
