"""Generated CPU tensors/metadata only; no model, source packets or outputs."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import copy
import hashlib
import json
import numpy as np
import torch
import independent_saved_grounding_math as m

OFFICIAL=Path('/home/minjae/anaconda3/lib/python3.13/site-packages/transformers/models/grounding_dino/processing_grounding_dino.py')
TRANSFORMS=OFFICIAL.parents[2]/'image_transforms.py'

def binding(f):return dict(path=str(Path(f).resolve()),sha256=hashlib.sha256(Path(f).read_bytes()).hexdigest())
def reject(fn):
    try:fn()
    except (ValueError,AssertionError,KeyError,IndexError):return
    raise AssertionError('Expected rejection')

def no_annotations(node):
    node=copy.deepcopy(node);node.returns=None
    for a in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]:a.annotation=None
    if node.args.vararg:node.args.vararg.annotation=None
    if node.args.kwarg:node.args.kwarg.annotation=None
    return node

def official_pure_processor():
    tree=ast.parse(OFFICIAL.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GroundingDinoProcessor')
    method=no_annotations(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='post_process_grounded_object_detection'))
    phrase=no_annotations(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_phrases_from_posmap'))
    t=ast.parse(TRANSFORMS.read_text());corners=no_annotations(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_center_to_corners_format_torch'))
    ns={'torch':torch,'DictWithDeprecationWarning':dict}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[phrase,corners],type_ignores=[])),str(OFFICIAL),'exec'),ns)
    ns['center_to_corners_format']=ns['_center_to_corners_format_torch']
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(OFFICIAL),'exec'),ns)
    def decode(self,rows):return [' '.join(str(int(x)) for x in row) for row in rows]
    return type('GeneratedNoModelProcessor',(),dict(post_process_grounded_object_detection=ns[method.name],batch_decode=decode))(),ns

def packet_fixture(ns):
    active=np.array([0,1,2,4,255],np.int64);attention=np.zeros((1,256),np.int64);attention[0,active]=1;ids=np.arange(256,dtype=np.int64)[None,:]+1000
    desired=torch.tensor([[.9,.3,.2,.1,.8],[.1,.4,.2,.2,.1],[.1,.6,.3,.1,.1],[.1,.2,.2,.6,.1],[.1,.2,.2,.2,.1],[.1,.9,.9,.1,.1]],dtype=torch.float32)
    raw_logits=torch.logit(desired);full=torch.full((1,6,256),float('-inf'),dtype=torch.float32);full[0,:,torch.tensor(active)]=raw_logits
    raw_boxes=torch.tensor([[[.3,.3,.2,.2],[.5,.5,.1,.1],[.7,.7,.2,.2],[.3,.3,.2,.2],[.2,.4,.1,.3],[.3,.3,.2,.2]]],dtype=torch.float32)
    all_probs=full.sigmoid()[0];maxscores=all_probs.max(-1).values;keep=torch.nonzero(maxscores>.4).flatten()
    hw=(10,20);native=ns['_center_to_corners_format_torch'](raw_boxes)[0]*torch.tensor([20,10,20,10],dtype=torch.float32)
    pos=all_probs[keep]>.3;tokens=ns['get_phrases_from_posmap'](pos,torch.from_numpy(ids[0]));labels=np.array([' '.join(str(int(x)) for x in row) for row in tokens],dtype=str)
    d=dict(raw_active_logits=raw_logits.numpy().copy(),raw_active_probabilities=all_probs[:,active].numpy().copy(),active_token_indices=active,input_ids=ids,attention_mask=attention,
        raw_normalized_boxes=raw_boxes[0].numpy().copy(),raw_native_boxes_xyxy=native.numpy().copy(),raw_query_max_scores=maxscores.numpy().copy(),raw_query_ordinals=np.arange(6,dtype=np.int64),
        boxes_xyxy=native[keep].numpy().copy(),scores=maxscores[keep].numpy().copy(),detector_query_index=keep.numpy().copy(),text_labels=labels)
    return d,full,raw_boxes,hw

def tests():
    torch.set_num_threads(1)
    original=dict(input_id='generated0',image='/generated/not-read.png',image_sha256='a'*64,raw_hw=[10,20],initial_points=[[1.,2.]]*8,roi_side=16.,query_box_xyxy=[0,0,10,10])
    row=dict(original,requested_category='pallet');assert m.validate_target(original,row,['pallet','mug'])=='pallet.'
    for key in ['GTmask','object_id','native_GTref8','pose','visibility']:
        reject(lambda:m.validate_target(original,dict(row,**{key:None}),['pallet']))
    reject(lambda:m.validate_target(original,dict(row,initial_points=[[2.,2.]]*8),['pallet']))
    reject(lambda:m.validate_target(original,dict(row,requested_category='Pallet'),['pallet']))
    r2=dict(row,input_id='generated1',requested_category='mug');r3=dict(row,input_id='generated2')
    uniq,mapping=m.expected_unique([row,r2,r3]);assert len(uniq)==2 and mapping==['V82_REQUEST_000000','V82_REQUEST_000001','V82_REQUEST_000000']
    assert all(set(z)==m.UNIQUE5 for z in uniq)
    assert m.request_key_sha256(row)!=m.request_key_sha256(r2) and m.request_key_sha256(row)==m.request_key_sha256(r3)
    processor,ns=official_pure_processor();d,full,boxes,hw=packet_fixture(ns)
    active=m.validate_full_logits(full.numpy(),d['attention_mask'],boxes.numpy());assert np.array_equal(active,d['active_token_indices'])
    replay=m.validate_packet(d,hw)
    official=processor.post_process_grounded_object_detection(SimpleNamespace(logits=full,pred_boxes=boxes),torch.from_numpy(d['input_ids']),threshold=.4,text_threshold=.3,target_sizes=[hw])[0]
    assert np.array_equal(official['scores'].numpy(),d['scores']) and np.array_equal(official['boxes'].numpy(),d['boxes_xyxy']) and official['text_labels']==d['text_labels'].tolist()
    assert [' '.join(map(str,z)) for z in replay['token_ids']]==official['text_labels']
    assert np.array_equal(replay['full_logits'],full.numpy())
    # Literal probability ties exercise strict saved thresholds independent of
    # a possible sigmoid inverse-rounding difference. No threshold tolerance.
    tied=copy.deepcopy(d);tied['raw_active_probabilities'][:]=np.float32(.1)
    tied['raw_active_probabilities'][0,1]=np.float32(.4)
    tied['raw_active_probabilities'][1,1]=np.nextafter(np.float32(.4),np.float32(1))
    tied['raw_active_probabilities'][1,2]=np.float32(.3)
    tied['raw_active_probabilities'][1,3]=np.nextafter(np.float32(.3),np.float32(1))
    tied['raw_query_max_scores']=tied['raw_active_probabilities'].max(1)
    kept=np.array([1],np.int64);tied.update(detector_query_index=kept,scores=tied['raw_query_max_scores'][kept],boxes_xyxy=tied['raw_native_boxes_xyxy'][kept],text_labels=np.array(['generated']))
    tr=m.validate_packet(tied,hw);assert np.array_equal(tr['keep'],kept) and tr['token_ids']==[[1001,1004]]
    # Active CLS/last contrastive token can drive confidence while phrase empty;
    # this is still a kept detection, never textual/GT class filtered.
    empty=copy.deepcopy(d);empty['raw_active_probabilities'][:]=.1;empty['raw_active_probabilities'][0,[0,4]]=.9
    empty['raw_query_max_scores']=empty['raw_active_probabilities'].max(1);empty.update(detector_query_index=np.array([0],np.int64),scores=empty['raw_query_max_scores'][:1],boxes_xyxy=empty['raw_native_boxes_xyxy'][:1],text_labels=np.array(['']))
    assert m.validate_packet(empty,hw)['token_ids']==[[]]
    # Duplicate boxes from different original queries remain (no NMS).
    assert len(d['boxes_xyxy'])>=3
    for key in ['raw_active_logits','raw_active_probabilities','raw_normalized_boxes']:
        for value in [np.nan,np.inf,-np.inf]:
            bad=copy.deepcopy(d);bad[key].flat[0]=value;reject(lambda:m.validate_packet(bad,hw))
        bad=copy.deepcopy(d);bad[key]=bad[key].astype(np.float16);reject(lambda:m.validate_packet(bad,hw))
    for key in ['active_token_indices','raw_query_ordinals']:
        bad=copy.deepcopy(d);bad[key]=bad[key][::-1].copy();reject(lambda:m.validate_packet(bad,hw))
    bad=copy.deepcopy(d);bad['active_token_indices']=bad['active_token_indices'][:-1];reject(lambda:m.validate_packet(bad,hw))
    bad=copy.deepcopy(d);bad['raw_query_max_scores'][0]=.01;reject(lambda:m.validate_packet(bad,hw))
    bad=copy.deepcopy(d);bad['detector_query_index']=bad['detector_query_index'][::-1].copy();reject(lambda:m.validate_packet(bad,hw))
    badfull=full.numpy().copy();badfull[0,0,3]=0;reject(lambda:m.validate_full_logits(badfull,d['attention_mask'],boxes.numpy()))
    badfull=full.numpy().copy();badfull[0,0,1]=-np.inf;reject(lambda:m.validate_full_logits(badfull,d['attention_mask'],boxes.numpy()))
    assert m.associate([0,0,10,1],hw,np.array([[0,0,1,1]],np.float32),np.array([.5],np.float32),np.array([4],np.int64))[0]=='OK'
    for initial,bs,reason in [(None,[[0,0,1,1]],'INVALID_INITIAL_QUERY_BOX'),([0,0,10,10],[],'NO_FIXED_THRESHOLD_DETECTIONS'),([0,0,10,10],[[19,9,20,10]],'MAX_INPUT_BOX_IOU_BELOW_0P1'),([-10,0,-1,10],[[-10,0,-1,10]],'INVALID_CLIPPED_SAM_PROMPT')]:
        b=np.asarray(bs,np.float32).reshape(-1,4);assert m.associate(initial,hw,b,np.full(len(b),.5,np.float32),np.arange(len(b),dtype=np.int64))[0]==reason
    b=np.array([[0,0,10,10]]*3,np.float32)
    assert m.associate([0,0,10,10],hw,b,np.array([.6,.9,.9],np.float32),np.array([0,8,2],np.int64))[1]==2
    good={k:[] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']};m.validate_integrity(good,True,'sha','sha')
    for key in good:
        bad=copy.deepcopy(good);bad[key]=['bad'];reject(lambda:m.validate_integrity(bad,True,'sha','sha'))
    reject(lambda:m.validate_integrity(good,False,'sha','sha'));reject(lambda:m.validate_integrity(good,True,'a','b'))
    return dict(PASS=True,generated_only=True,Torch=torch.__version__,device='CPU',model_constructors=0,model_forward=0,source_packets_or_outputs_read=0,RGB_GTmask_weights_current223_reads=0,fit=0,PnP=0,
        tests=['original7+category whitelist','RGB+caption dedupe','fullquery active-F32 rawlogits+same-tensorprobs','noncontiguousactiveindices+padding guards','official AST full postprocessor exact scores/boxes/tokenids/text','strict .4/.3 ties and nextafter','special-only emptylabel retained','noNMS fullqueryorder','IoU .1 inclusive and score/ordinaltie','all U reasons','loading4empty/firstrepeat/state'],
        bindings=dict(helper=binding(m.__file__),tests=binding(__file__),official_processor=binding(OFFICIAL),official_box_transform=binding(TRANSFORMS)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();result=tests()
    with open(a.out,'x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
