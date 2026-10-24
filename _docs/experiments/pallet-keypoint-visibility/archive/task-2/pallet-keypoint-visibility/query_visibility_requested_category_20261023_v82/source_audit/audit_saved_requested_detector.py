"""Future root-invoked saved detector audit. No RGB/mask/model/weight reader.

Actual output access requires the completed seal SHA supplied by root. Saved
active probabilities, not a CPU sigmoid approximation, govern exact replay.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math
import os
import sys
import numpy as np
import independent_saved_grounding_math as m

R=Path(__file__).resolve().parent;V=R.parent;OLD=V.parent/'query_visibility_grounded_target_20261023_v81'
OUT=V/'detector/outputs';SEAL=OUT/'DETECTOR_OUTPUT_SEAL.json'
CODE_SHA='6dce5e9e48511760deb3311ec59c678762ea0410c0f86ace7b9e5971fc950fbb'
INPUT_AUDIT_SHA='403af6a2e0821bafdedf4b4451a9ab767ad38a5b3ab44fe1e7d960c9bcab3779'
CODE_AUDIT_SHA='517fbee222407636bce69a03bc5bfe8be0a738279576761dd8484c2020f7728b'
READS=[]

def safe(path):
    p=Path(path)
    assert p.suffix.lower() in {'.json','.jsonl','.npz','.py','.txt'},'No RGB/masks/weights allowed'
    assert 'private_evaluation' not in p.parts and '/segmentation/outputs/masks/' not in str(p)
    assert 'current223' not in str(p).lower()
    return p
def sha(path):p=safe(path);READS.append(str(p));return hashlib.sha256(p.read_bytes()).hexdigest()
def binding(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def read(path):p=safe(path);READS.append(str(p));return json.loads(p.read_text())
def lines(path):p=safe(path);READS.append(str(p));return [json.loads(x) for x in p.read_text().splitlines() if x]

def native_boxes_fp32(raw,hw):
    h,w=hw;b=np.empty_like(raw,dtype=np.float32)
    b[:,0]=raw[:,0]-np.float32(.5)*raw[:,2];b[:,1]=raw[:,1]-np.float32(.5)*raw[:,3]
    b[:,2]=raw[:,0]+np.float32(.5)*raw[:,2];b[:,3]=raw[:,1]+np.float32(.5)*raw[:,3]
    return b*np.asarray([w,h,w,h],np.float32)

def exact_iou(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float32).astype(np.float64)
    inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return float(inter/union) if union>0 else 0.

def main(seal_sha):
    assert len(seal_sha)==64 and all(c in '0123456789abcdef' for c in seal_sha),'Root-supplied final seal SHA required'
    assert sha(SEAL)==seal_sha
    assert sha(R/'INDEPENDENT_INPUT_PACKET_METADATA_AUDIT.json')==INPUT_AUDIT_SHA
    assert sha(R/'INDEPENDENT_DETECTOR_CODE_REVIEW_BEFORE_FORWARD.json')==CODE_AUDIT_SHA
    pre=read(R/'INDEPENDENT_PREPARATION_LOCK_BEFORE_OUTPUTS.json')
    for b in pre['files']:check(b)
    for k in ['protocol','criteria','logging_addendum']:check(pre[k])
    s=read(SEAL);assert s['completed'] is True and s['PASS'] is True and s['current223_fit_PnP']==0
    assert s['code']['sha256']==CODE_SHA
    for k in ['entrylock','decisions','logging_addendum','code','assets','runtime','per_request','associations','summary','caption_gate','loading_info','first_repeat','postprocess_contract','guard_contract']:check(s[k])
    for b in s['source_inputs'].values():check(b)
    entry=read(s['entrylock']['path']);assert entry['code']==s['code'] and entry['source_inputs']==s['source_inputs']
    assert entry['model_directory']==str(OLD/'model_assets')
    inputs=lines(s['source_inputs']['manifest']['path']);requests=lines(s['source_inputs']['unique_manifest']['path']);mapping=lines(s['source_inputs']['target_to_request']['path'])
    rows=lines(s['per_request']['path']);assocs=lines(s['associations']['path']);summary=read(s['summary']['path'])
    assert len(inputs)==len(requests)==len(mapping)==len(rows)==len(assocs)==1035
    unique,expected_map=m.expected_unique(inputs);assert unique==requests
    assert len({r['image_sha256'] for r in requests})==435
    # Only tokenizer text assets are read; no safetensors or model constructor.
    assets=read(s['assets']['path'])
    tokenizer_names={'tokenizer_config.json','special_tokens_map.json','added_tokens.json','vocab.txt','tokenizer.json'}
    tokenizer_bindings=[b for b in assets['files'] if Path(b['path']).name in tokenizer_names]
    for b in tokenizer_bindings:check(b)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    sys.path.insert(0,str(OLD/'runtime/overlay'))
    import transformers
    from transformers import BertTokenizer
    assert transformers.__version__=='4.57.6'
    tokenizer=BertTokenizer.from_pretrained(str(OLD/'model_assets'),local_files_only=True,trust_remote_code=False)
    assert type(tokenizer).__name__=='BertTokenizer'
    capgate=read(s['caption_gate']['path']);assert capgate['PASS'] is True and capgate['use_fast'] is False and capgate['before_first_source_RGB_decode'] is True
    gate={g['caption']:g for g in capgate['captions']};assert len(gate)==len(capgate['captions'])
    assert set(gate)=={r['requested_category']+'.' for r in requests}
    # Decode identity is compared to the prior saved detector certificate only;
    # no source image is reopened by this audit.
    prior_seal=read(OLD/'detector/outputs/DETECTOR_OUTPUT_SEAL.json');check(prior_seal['per_RGB'])
    prior={r['image_sha256']:r for r in lines(prior_seal['per_RGB']['path'])}
    payloads={};allQ=0;allactive=0;detections=0;emptylabels=0;decodeby={};caption_lengths=Counter()
    rowkeys=m.UNIQUE5|{'caption','request_key_sha256','decoded_RGB_SHA256','detections','detection_count','padding_guard'}
    for req,row in zip(requests,rows):
        assert set(row)==rowkeys and {k:row[k] for k in m.UNIQUE5}==req
        caption=req['requested_category']+'.';assert row['caption']==caption and row['request_key_sha256']==m.request_key_sha256(req)
        assert row['decoded_RGB_SHA256']==prior[req['image_sha256']]['decoded_RGB_SHA256']
        if req['image_sha256'] in decodeby:assert decodeby[req['image_sha256']]==row['decoded_RGB_SHA256']
        decodeby[req['image_sha256']]=row['decoded_RGB_SHA256']
        check(row['detections']);READS.append(row['detections']['path'])
        with np.load(safe(row['detections']['path']),allow_pickle=False) as z:d={k:z[k] for k in z.files}
        replay=m.validate_packet(d,req['raw_hw'])
        assert np.array_equal(native_boxes_fp32(d['raw_normalized_boxes'],req['raw_hw']),d['raw_native_boxes_xyxy']),'Allquery normalized->native FP32 exact'
        tok=tokenizer(caption,return_tensors='np')
        assert np.array_equal(tok['input_ids'],d['input_ids']) and np.array_equal(tok['attention_mask'],d['attention_mask'])
        assert gate[caption]['input_ids']==d['input_ids'].tolist() and gate[caption]['attention_mask']==d['attention_mask'].tolist()
        decoded=tokenizer.batch_decode(replay['token_ids']);assert decoded==d['text_labels'].tolist(),'Exact official token-to-text replay'
        Q,A=d['raw_active_logits'].shape;T=d['input_ids'].shape[1]
        assert row['padding_guard']==dict(PASS=True,raw_shape=[1,Q,256],input_token_length=T,active_token_count=A,finite_active_logits=True,all_inactive_exact_negative_inf=True,finite_FP32_boxes=True,full_everyquery_FP32_active_logits_and_same_tensor_probs=True)
        assert row['detection_count']==len(replay['keep'])
        payloads[req['request_id']]=d;allQ+=Q;allactive+=Q*A;detections+=len(replay['keep']);emptylabels+=sum(x=='' for x in decoded);caption_lengths[T]+=1
    assert len(decodeby)==435
    reqby={r['request_id']:r for r in requests};assoccounts=Counter()
    extra={'request_id','caption','request_key_sha256','status','reason','selected_detection_index','selected_detector_query_index','selected_detection_score','association_iou','selected_box_xyxy_raw','selected_box_xyxy_prompt'}
    for i,(target,link,a) in enumerate(zip(inputs,mapping,assocs)):
        assert set(a)==m.TARGET8|extra and {k:a[k] for k in m.TARGET8}==target
        rid=expected_map[i];req=reqby[rid]
        assert link==dict(input_id=target['input_id'],request_id=rid,image_sha256=target['image_sha256'],requested_category=target['requested_category'],caption=target['requested_category']+'.',request_key_sha256=m.request_key_sha256(target))
        assert a['request_id']==rid and a['caption']==link['caption'] and a['request_key_sha256']==link['request_key_sha256']
        assert all(req[k]==target[k] for k in ['image','image_sha256','raw_hw','requested_category'])
        d=payloads[rid];reason,j,prompt=m.associate(target['query_box_xyxy'],target['raw_hw'],d['boxes_xyxy'],d['scores'],d['detector_query_index'])
        assert a['reason']==reason
        if reason=='OK':
            assert a['status']=='ASSOCIATED' and a['selected_detection_index']==j
            assert a['selected_detector_query_index']==int(d['detector_query_index'][j]) and a['selected_detection_score']==float(d['scores'][j])
            assert a['selected_box_xyxy_raw']==d['boxes_xyxy'][j].tolist() and a['selected_box_xyxy_prompt']==prompt.tolist()
            assert a['association_iou']==exact_iou(target['query_box_xyxy'],d['boxes_xyxy'][j])
        else:
            assert a['status']=='U'
            assert all(a[k] is None for k in ['selected_detection_index','selected_detector_query_index','selected_detection_score','association_iou','selected_box_xyxy_raw','selected_box_xyxy_prompt'])
        assoccounts[reason]+=1
    loading=read(s['loading_info']['path']);repeat=read(s['first_repeat']['path'])
    assert loading['PASS'] is True and loading['before_first_source_RGB_decode'] is True
    m.validate_integrity(loading['output_loading_info'],repeat['PASS'],summary['state_SHA256_before'],summary['state_SHA256_after'])
    assert repeat['logits_boxes_probs_maxscores_order_labels_bitexact'] is True and repeat['official_postprocessor_bitexact'] is True and repeat['state_unchanged'] is True and repeat['source_forward_calls']==2
    assert summary['completed'] is True and summary['target_instances']==1035 and summary['query_slots']==8280 and summary['unique_RGB']==435 and summary['unique_RGB_caption_requests']==1035
    assert summary['association_reasons']==dict(assoccounts) and summary['associated']==assoccounts['OK'] and summary['held_U']==1035-assoccounts['OK']
    assert summary['source_forward_calls']==summary['expected_forward_calls']==1036
    assert summary['first_repeat_bitexact'] is True and summary['state_unchanged'] is True and summary['FP32_noautocast_TF32false'] is True and summary['custom_kernels_disabled'] is True
    assert summary['NMS'] is False and summary['original_q_L_box_order_unchanged'] is True and summary['requested_category_side_information'] is True
    assert summary['GT_instance_or_reference_forward'] is False and summary['source_provenance_forward'] is False and summary['current223']==summary['fit']==summary['PnP']==0
    result=dict(completed=True,PASS=True,blockers=[],detector_output_seal=binding(SEAL),audit_code=binding(__file__),independent_helper=binding(m.__file__),preparation_lock=binding(R/'INDEPENDENT_PREPARATION_LOCK_BEFORE_OUTPUTS.json'),input_audit=binding(R/'INDEPENDENT_INPUT_PACKET_METADATA_AUDIT.json'),code_audit=binding(R/'INDEPENDENT_DETECTOR_CODE_REVIEW_BEFORE_FORWARD.json'),
        counts=dict(targets=1035,unique_RGB=435,unique_RGB_caption_requests=1035,source_forward_calls=1036,all_raw_queries=allQ,active_logit_and_probability_cells=allactive,kept_detections=detections,kept_empty_text_labels=emptylabels,association_reasons=dict(assoccounts),associated=assoccounts['OK'],U=1035-assoccounts['OK'],caption_token_length_distribution=dict(caption_lengths)),
        verified=dict(all13_arrays_exact_schema_dtypes=True,every_original_query_and_all_attended_special_tokens=True,saved_probability_max_and_strict_dot4_keep_bitexact=True,strict_dot3_phrase_token_and_bound_slow_tokenizer_decode_exact=True,no_CPU_sigmoid_reinterpretation_or_threshold_tolerance=True,all_raw_normalized_to_native_FP32_boxes_exact=True,padding_certificate_fullshape_attention_and_active_data_consistent=True,all1035_original8_geometry_categories_and_order_unchanged=True,all1035_input_IoU_score_ordinal_prompt_U_reasons_exact=True,all435_decode_metadata_equal_prior_saved_RGB_certificate=True,first_repeat_loading4lists_and_full_state_checks_match=True),
        limitations='Inactive/padded raw logits are not persisted; their exact -inf is certified by bound successful producer guard before postprocess, not independently remeasured from reconstruction. Raw logits were checked complete/F32/finite; independent score/text replay uses the saved exact same-tensor probabilities, without CPU sigmoid. A same-CUDA logit->probability numeric rerun is optional and not performed here. No raw RGB decode or GT/mask/weight/model read was performed.',
        changed_input_contract='Requested class is provided category side information, YCB metadata-oracle and SYNTH constant pallet; not predicted class or same-input comparison. No category/GT chooses a posthoc output.',scope=dict(RGB_reads=0,GT_masks_or_references_read=0,weight_bytes_read=0,model_constructors_or_forward=0,tokenizer_constructor_only=True,current223_reads=0,fit=0,PnP=0,prior_artifacts_modified=0),tokenizer_assets=tokenizer_bindings,read_metadata_and_saved_array_paths=sorted(set(READS)))
    path=R/'INDEPENDENT_DETECTOR_OUTPUT_AUDIT.json'
    with path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(PASS=True,path=str(path),sha256=sha(path),counts=result['counts'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal-sha',required=True);args=p.parse_args()
    main(args.seal_sha)
