"""Requested-category source detector. No RGB/weights/model load in generated/freeze.

The frozen V81 geometry/padding/loading helpers are reused after an exact hash
check. Only processor RGB and requested-category text tensors reach the model.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import numpy as np

R = Path(__file__).resolve().parent
V = R.parent
OLD = V.parent / 'query_visibility_grounded_target_20261023_v81'
OLD_CODE = OLD / 'detector/run_fixed_grounding.py'
OLD_SHA = '45aefb9ec230ed44dbe9a87f4d9869a2e23407d66292d6be4e59796059a3bc0d'
DECISIONS = V / 'EXPERIMENT_DECISIONS_BEFORE_INPUTS_AND_FORWARD.json'
DECISION_SHA = 'ab89f3885627cdd319b93fb6ead6357950a10f7fa51d751798ddd7cabba5af28'
ADDENDUM = V / 'ACTIVE_PROBABILITY_TRACE_ADDENDUM_BEFORE_FORWARD.json'
REVIEW = OLD / 'next_category_review/EXACT_FROZEN_V82_PROTOCOL_REVIEW.json'
REVIEW_SHA = 'bb28b2d639fcdd893718c5164902683d990fbfb5817d006947ea3fcad3d5df6c'
RUNTIME = OLD / 'runtime/RUNTIME_IMPORT_ONLY_SEAL.json'
RUNTIME_SHA = '13fb5d64dcde87314fb37badd2fba15d452a85083bbd29d74384671cc8b83933'
ASSETS = OLD / 'model_assets/ASSET_SEAL.json'
ASSET_SHA = '37ead31a70f2deab84c04e4c8a1da084d92b50190ecdac53ffe3ca5799c1921d'
LOGIT_CONTRACT = OLD / 'detector/OFFICIAL_LOGIT_AND_LOADING_GUARD_CONTRACT.json'
MANIFEST = V / 'input_preparation/INPUTS_RGB_ONLY.jsonl'
UNIQUE = V / 'input_preparation/UNIQUE_RGB_CATEGORY_INPUTS_ONLY.jsonl'
MAP = V / 'input_preparation/TARGET_TO_REQUEST_INPUT_ONLY.jsonl'
INPUT_SEAL = V / 'input_preparation/SEAL.json'
INPUT_SEAL_SHA = '0bc3ed7a20e976c99792e09caf6ee64968437d7aab9352266c5df2ed6cb81de4'
PROOF = R / 'GENERATED_REQUESTED_DETECTOR_TESTS.json'
POST_CONTRACT = R / 'OFFICIAL_POSTPROCESS_TRACE_CONTRACT.json'
ENTRY = R / 'DETECTOR_ENTRYLOCK.json'
OUT = R / 'outputs'
INPUT_FIELDS = {'input_id', 'image', 'image_sha256', 'raw_hw', 'initial_points', 'roi_side', 'query_box_xyxy', 'requested_category'}
REQUEST_FIELDS = {'request_id', 'image', 'image_sha256', 'raw_hw', 'requested_category'}
MAP_FIELDS = {'input_id', 'request_id', 'image_sha256', 'requested_category', 'caption', 'request_key_sha256'}

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()

def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def read(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def check(b): assert sha(b['path']) == b['sha256'], b['path']
def write(p, value):
    with open(p, 'x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
def write_lines(p, rows):
    with open(p, 'x') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

assert sha(OLD_CODE) == OLD_SHA
spec = importlib.util.spec_from_file_location('frozen_v81_detector_helpers', OLD_CODE)
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

def decisions():
    assert sha(DECISIONS) == DECISION_SHA and sha(REVIEW) == REVIEW_SHA
    p = read(DECISIONS)
    assert p['detector']['box_threshold_strict'] == .4 and p['detector']['text_threshold_strict'] == .3
    assert p['association']['minimum_IoU_inclusive'] == .1
    assert read(ADDENDUM)['base_protocol_SHA256'] == DECISION_SHA
    return p

def caption(row): return row['requested_category'] + '.'
def request_key(row): return hashlib.sha256((row['image_sha256'] + '\n' + caption(row)).encode('utf-8')).hexdigest()

def validate_input(row, request_only=False):
    assert set(row) == (REQUEST_FIELDS if request_only else INPUT_FIELDS), 'GT/unknown input field'
    category = row['requested_category']
    allowed = read(DECISIONS)['fixed_category_map_YCB_obj_ids_1_through_21'] + ['pallet']
    assert isinstance(category, str) and category in allowed
    if request_only:
        assert isinstance(row['request_id'], str) and row['request_id']
        base = {k: row[k] for k in ('image', 'image_sha256', 'raw_hw')}
        old.validate_input(dict(base, input_id=row['request_id']), True)
    else:
        old.validate_input({k: val for k, val in row.items() if k != 'requested_category'})
    return row

def validate_manifests(targets, requests, mapping, expected=1035):
    assert len(targets) == len(mapping) == expected and len({r['input_id'] for r in targets}) == expected
    by_key = {}; by_id = {}
    for i, row in enumerate(requests):
        validate_input(row, True)
        assert row['request_id'] == f'V82_REQUEST_{i:06d}'
        key = request_key(row); assert key not in by_key
        by_key[key] = row; by_id[row['request_id']] = row
    observed_order = []
    for target, link in zip(targets, mapping):
        validate_input(target); assert set(link) == MAP_FIELDS
        key = request_key(target); req = by_key[key]
        assert all(target[k] == req[k] for k in ('image', 'image_sha256', 'raw_hw', 'requested_category'))
        assert link == dict(input_id=target['input_id'], request_id=req['request_id'], image_sha256=target['image_sha256'], requested_category=target['requested_category'], caption=caption(target), request_key_sha256=key)
        if key not in observed_order: observed_order.append(key)
    assert observed_order == list(by_key), 'Unique tuple first-appearance order'
    return by_key

def associate(row, req, boxes, scores, query_index):
    validate_input(row)
    result = old.associate({k: v for k, v in row.items() if k != 'requested_category'}, req['request_id'], boxes, scores, query_index)
    result['request_id'] = result.pop('rgb_input_id')
    result.update(requested_category=row['requested_category'], caption=caption(row), request_key_sha256=request_key(row))
    return result

def postprocess_with_trace(processor, outputs, input_ids, target_sizes):
    """Pinned official source-equivalent; trace uses the SAME sigmoid tensor.

    Mirrors processing_grounding_dino.py post_process_grounded_object_detection
    lines233-273. No model hook, monkeypatch, or alternate threshold arithmetic.
    """
    import torch
    from transformers.image_transforms import center_to_corners_format
    from transformers.models.grounding_dino.processing_grounding_dino import get_phrases_from_posmap
    batch_probs = torch.sigmoid(outputs.logits)
    batch_scores = torch.max(batch_probs, dim=-1)[0]
    batch_boxes = center_to_corners_format(outputs.pred_boxes)
    img_h = torch.Tensor([i[0] for i in target_sizes])
    img_w = torch.Tensor([i[1] for i in target_sizes])
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(batch_boxes.device)
    batch_boxes = batch_boxes * scale_fct[:, None, :]
    results = []
    for idx, (scores, boxes, probs) in enumerate(zip(batch_scores, batch_boxes, batch_probs)):
        keep = scores > .4
        label_ids = get_phrases_from_posmap(probs[keep] > .3, input_ids[idx])
        results.append(dict(scores=scores[keep], boxes=boxes[keep], text_labels=processor.batch_decode(label_ids)))
    return results, batch_probs, batch_scores, batch_boxes

def assert_results_equal(a, b):
    import torch
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x['scores'], y['scores']) and torch.equal(x['boxes'], y['boxes'])
        assert x['text_labels'] == y['text_labels']

def generated():
    decisions()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    sys.path.insert(0, str(OLD / 'runtime/overlay'))
    import torch
    from types import SimpleNamespace
    from transformers.models.grounding_dino.processing_grounding_dino import GroundingDinoProcessor
    torch.set_num_threads(1)
    class GeneratedProcessor:
        def batch_decode(self, ids): return ['|'.join(map(str, row)) for row in ids]
    proc = GeneratedProcessor()
    tests = 0
    for length in (7, 77, 256):
        ids = torch.arange(length).repeat(2, 1)
        mask = torch.ones((2, length), dtype=torch.int64); mask[1, 2] = 0
        logit = torch.full((2, 12, 256), -torch.inf)
        active = torch.zeros((2, 256), dtype=torch.bool); active[:, :length] = mask.bool()
        values = torch.linspace(-3., 3., int(active.sum()) * 12)
        logit[active[:, None, :].expand_as(logit)] = values
        # Exact/adjacent strict probability thresholds and all-special tokens.
        for q, p in enumerate((.3, .4, .5, .9)):
            logit[0, q, 1] = torch.logit(torch.tensor(p, dtype=torch.float32))
        boxes = torch.linspace(.1, .9, 96).reshape(2, 12, 4)
        output = SimpleNamespace(logits=logit, pred_boxes=boxes)
        old.validate_output_arrays(logit.numpy(), mask.numpy(), boxes.numpy())
        got, probs, scores, native = postprocess_with_trace(proc, output, ids, [(30, 40), (50, 60)])
        official = GroundingDinoProcessor.post_process_grounded_object_detection(proc, output, ids, threshold=.4, text_threshold=.3, target_sizes=[(30, 40), (50, 60)])
        assert_results_equal(got, official)
        for b in range(2):
            indices = torch.nonzero(active[b]).flatten()
            packed = logit[b][:, indices].numpy(); packed_probs = probs[b][:, indices].numpy()
            assert packed.dtype == packed_probs.dtype == np.float32 and packed.shape == (12, len(indices))
            assert np.array_equal(packed_probs.max(1), scores[b].numpy())
            assert not probs[b][:, ~active[b]].any()
        tests += 1
        for pos, bad in [((0, 0, 1), np.nan), ((0, 0, 1), np.inf), ((0, 0, 1), -np.inf), ((1, 0, 2), 0.)]:
            malformed = logit.numpy().copy(); malformed[pos] = bad
            try: old.validate_output_arrays(malformed, mask.numpy(), boxes.numpy())
            except AssertionError: pass
            else: raise AssertionError('malformed padding accepted')
    empty = SimpleNamespace(logits=torch.full((1, 4, 256), -10.), pred_boxes=torch.ones((1, 4, 4)))
    got, *_ = postprocess_with_trace(proc, empty, torch.arange(256)[None], [(30, 40)])
    assert not len(got[0]['scores']) and got[0]['text_labels'] == []
    row = dict(input_id='generated', image='/generated/no-file.png', image_sha256='0'*64, raw_hw=[30, 40], initial_points=[[i, i+.25] for i in range(8)], roi_side=16., query_box_xyxy=[10., 10., 20., 20.], requested_category='pallet')
    other = dict(row, input_id='generated2', requested_category='mug')
    assert request_key(row) != request_key(other)
    reqs = [dict(request_id=f'V82_REQUEST_{i:06d}', **{k: r[k] for k in REQUEST_FIELDS if k != 'request_id'}) for i, r in enumerate((row, other))]
    mapping = [dict(input_id=r['input_id'], request_id=req['request_id'], image_sha256=r['image_sha256'], requested_category=r['requested_category'], caption=caption(r), request_key_sha256=request_key(r)) for r, req in zip((row, other), reqs)]
    validate_manifests([row, other], reqs, mapping, 2)
    a = associate(row, reqs[0], [[10,10,20,20]]*3, [.6,.9,.9], [2,6,4])
    assert a['selected_detection_index'] == 2 and a['selected_detector_query_index'] == 4
    assert associate(row, reqs[0], np.empty((0,4)), [], [])['reason'] == 'NO_FIXED_THRESHOLD_DETECTIONS'
    assert associate(dict(row, query_box_xyxy=None), reqs[0], [[10,10,20,20]], [.9], [0])['reason'] == 'INVALID_INITIAL_QUERY_BOX'
    assert associate(row, reqs[0], [[0,0,1,1]], [.9], [0])['reason'] == 'MAX_INPUT_BOX_IOU_BELOW_0P1'
    outside = dict(row, query_box_xyxy=[-10,-10,-1,-1])
    assert associate(outside, reqs[0], [[-10,-10,-1,-1]], [.9], [0])['reason'] == 'INVALID_CLIPPED_SAM_PROMPT'
    boundary = dict(row, query_box_xyxy=[0,0,1,10])
    assert associate(boundary, reqs[0], [[0,0,1,1]], [.5], [0])['status'] == 'ASSOCIATED'
    assert all(associate(outside, reqs[0], [[-10,-10,-1,-1]], [.9], [0])[k] is None for k in ('selected_detection_index','selected_detector_query_index','selected_detection_score','association_iou','selected_box_xyxy_raw','selected_box_xyxy_prompt'))
    for k in ('GT', 'reference_xy', 'mask', 'source', 'object_id', 'pose'):
        try: validate_input(dict(row, **{k:'forbidden'}))
        except AssertionError: pass
        else: raise AssertionError(k)
    old.validate_loading_info({k:[] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')})
    write(PROOF, dict(PASS=True, code=bind(__file__), decisions=bind(DECISIONS), logging_addendum=bind(ADDENDUM), old_helpers=bind(OLD_CODE), postprocess_contract=bind(POST_CONTRACT), generated_official_postprocess_bitexact_batches=tests, generated_attention_77_256_padding_malformed_rejection=True, generated_full_allquery_active_logits_probs_and_special_tokens=True, generated_empty_detection_and_strict_thresholds=True, generated_request_category_key_order_geometry_ties_GT_rejection=True, generated_all_four_U_reasons_null_selected_fields_and_inclusive_0P1=True, source_RGB_GT_weights_model_tokenizer_constructors_forward_fit_current_PnP=0))
    print('GENERATED_PASS', flush=True)

def freeze():
    p = decisions(); proof = read(PROOF)
    assert proof['PASS'] and proof['code'] == bind(__file__)
    for k in ('decisions', 'logging_addendum', 'old_helpers', 'postprocess_contract'): check(proof[k])
    assert sha(RUNTIME) == RUNTIME_SHA and sha(ASSETS) == ASSET_SHA and sha(INPUT_SEAL) == INPUT_SEAL_SHA
    assert read(RUNTIME)['PASS'] and read(ASSETS)['completed'] and read(ASSETS)['PASS']
    for f in read(ASSETS)['files']: check(f)
    for f in read(RUNTIME)['module_sources']: check(f)
    assert sha(OLD / 'model_assets/model.safetensors') == p['detector']['weights_SHA256']
    seal = read(INPUT_SEAL); assert seal['completed'] and seal['PASS']
    for f in seal['files']: check(f)  # Hash only; GT provenance body is not opened.
    check(read(LOGIT_CONTRACT)['official_modeling_source'])
    contract = read(POST_CONTRACT); check(contract['official_processor_source'])
    value = dict(completed=True, code=bind(__file__), generated=bind(PROOF), decisions=bind(DECISIONS), logging_addendum=bind(ADDENDUM), protocol_review=bind(REVIEW), old_helpers=bind(OLD_CODE), guard_contract=bind(LOGIT_CONTRACT), postprocess_contract=bind(POST_CONTRACT), assets=bind(ASSETS), runtime=bind(RUNTIME), source_inputs=dict(manifest=bind(MANIFEST), unique_manifest=bind(UNIQUE), target_to_request=bind(MAP), seal=bind(INPUT_SEAL)), model_directory=str(OLD / 'model_assets'), output_directory=str(OUT), processor_use_fast=False, forward_only='Processor RGB/pixelmask and exact requested_category+period text tensors.', forward_excluded='q/L/roughbox/GT/reference/masks/instance IDs/source provenance.', parent_must_review_and_explicitly_invoke_run=True, source_RGB_weights_model_forward_fit_current_PnP=0)
    write(ENTRY, value); write(R / 'DETECTOR_ENTRYLOCK_TEMPLATE.json', dict(value, template_only=True, execution_requires_root_approved_entry=True))
    print(json.dumps(bind(ENTRY)), flush=True)

def run():
    p = decisions(); e = read(ENTRY)
    assert e['completed'] and e['code'] == bind(__file__)
    for k in ('generated','decisions','logging_addendum','protocol_review','old_helpers','guard_contract','postprocess_contract','assets','runtime'): check(e[k])
    for b in e['source_inputs'].values(): check(b)
    check(read(e['guard_contract']['path'])['official_modeling_source'])
    check(read(e['postprocess_contract']['path'])['official_processor_source'])
    assert not OUT.exists(), 'Preserve failed outputs/entry; no silent restart'
    for b in read(ASSETS)['files']: check(b)
    rt = read(RUNTIME)
    for b in rt['module_sources']: check(b)
    assert Path(sys.executable).resolve() == Path(rt['python']).resolve()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.path.insert(0, str(OLD / 'runtime/overlay'))
    import importlib.metadata
    import random
    import torch
    from PIL import Image
    from transformers import GroundingDinoConfig, GroundingDinoProcessor, GroundingDinoForObjectDetection
    for package, version in rt['versions'].items(): assert importlib.metadata.version(package) == version
    assert torch.cuda.is_available()
    torch.set_num_threads(1); torch.manual_seed(20261023); torch.cuda.manual_seed_all(20261023)
    np.random.seed(20261023); random.seed(20261023)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    targets, requests, mapping = lines(MANIFEST), lines(UNIQUE), lines(MAP)
    by_key = validate_manifests(targets, requests, mapping)
    assert len(requests) == 1035 and len({x['image_sha256'] for x in requests}) == 435
    OUT.mkdir(); (OUT/'detections').mkdir()
    write(OUT/'EXECUTION_STARTED.json', dict(entrylock=bind(ENTRY), target_count=len(targets), unique_requests=len(requests), source_forward_calls=0, fit_current_PnP=0))
    proc = GroundingDinoProcessor.from_pretrained(e['model_directory'], local_files_only=True, trust_remote_code=False, use_fast=False)
    assert type(proc.image_processor).__name__ == 'GroundingDinoImageProcessor' and type(proc.tokenizer).__name__ == 'BertTokenizer'
    config = GroundingDinoConfig.from_pretrained(e['model_directory'], local_files_only=True, trust_remote_code=False); config.disable_custom_kernels=True
    model, loading = GroundingDinoForObjectDetection.from_pretrained(e['model_directory'], config=config, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32, use_safetensors=True, output_loading_info=True)
    old.validate_loading_info(loading)
    write(OUT/'PRETRAINED_LOADING_INFO.json', dict(PASS=True, output_loading_info=loading, before_first_source_RGB_decode=True))
    model=model.cuda().eval().requires_grad_(False)
    assert model.config.disable_custom_kernels and all(x.dtype == torch.float32 for x in model.parameters())
    tokens = {}
    for req in requests:
        cap=caption(req)
        if cap not in tokens:
            tok=proc.tokenizer(cap, return_tensors='pt')
            assert tok['input_ids'].shape[0]==1 and 0<tok['input_ids'].shape[1]<=256
            tokens[cap]={k:v for k,v in tok.items()}
    write(OUT/'CAPTION_TOKEN_GATE.json', dict(PASS=True, use_fast=False, before_first_source_RGB_decode=True, captions=[dict(caption=cap, input_ids=t['input_ids'].tolist(), attention_mask=t['attention_mask'].tolist()) for cap,t in tokens.items()]))
    state_before=old.state_sha(model); start=time.monotonic(); rows=[]; detections={}; forwards=0
    for n,req in enumerate(requests):
        assert sha(req['image'])==req['image_sha256']
        with Image.open(req['image']) as im: array=np.asarray(im.convert('RGB')).copy()
        assert array.dtype==np.uint8 and list(array.shape)==[*req['raw_hw'],3]
        cap=caption(req); inputs=proc(images=array,text=cap,return_tensors='pt')
        assert set(inputs)<={'pixel_values','pixel_mask','input_ids','attention_mask','token_type_ids'}
        for k in ('input_ids','attention_mask'): assert torch.equal(inputs[k],tokens[cap][k])
        inputs=inputs.to('cuda')
        with torch.inference_mode():
            output=model(**inputs); forwards+=1
            old.validate_output_arrays(output.logits.cpu().numpy(),inputs['attention_mask'].cpu().numpy(),output.pred_boxes.cpu().numpy())
            result,probs,maxscores,nativeboxes=postprocess_with_trace(proc,output,inputs['input_ids'],[tuple(req['raw_hw'])])
            if n==0:
                repeat=model(**inputs); forwards+=1
                assert torch.equal(output.logits,repeat.logits) and torch.equal(output.pred_boxes,repeat.pred_boxes)
                again,prob2,score2,box2=postprocess_with_trace(proc,repeat,inputs['input_ids'],[tuple(req['raw_hw'])])
                assert_results_equal(result,again); assert torch.equal(probs,prob2) and torch.equal(maxscores,score2) and torch.equal(nativeboxes,box2)
                official=proc.post_process_grounded_object_detection(output,inputs['input_ids'],threshold=.4,text_threshold=.3,target_sizes=[tuple(req['raw_hw'])])
                assert_results_equal(result,official)
                write(OUT/'FIRST_REQUEST_REPEAT.json',dict(PASS=True,logits_boxes_probs_maxscores_order_labels_bitexact=True,official_postprocessor_bitexact=True,state_unchanged=old.state_sha(model)==state_before,source_forward_calls=2))
            active=torch.nonzero(inputs['attention_mask'][0].bool()).flatten()
            keep=torch.nonzero(maxscores[0]>.4).flatten()
            boxes=result[0]['boxes'].cpu().numpy(); scores=result[0]['scores'].cpu().numpy(); qi=keep.cpu().numpy()
            rawlog=output.logits[0][:,active].cpu().numpy(); rawprob=probs[0][:,active].cpu().numpy()
            assert rawlog.dtype==rawprob.dtype==np.float32 and rawlog.shape==rawprob.shape==(output.logits.shape[1],len(active))
            assert np.array_equal(rawprob.max(1),maxscores[0].cpu().numpy())
            path=OUT/'detections'/(req['request_id']+'.npz')
            np.savez_compressed(path,raw_active_logits=rawlog,raw_active_probabilities=rawprob,active_token_indices=active.cpu().numpy(),input_ids=inputs['input_ids'].cpu().numpy(),attention_mask=inputs['attention_mask'].cpu().numpy(),raw_normalized_boxes=output.pred_boxes[0].cpu().numpy(),raw_native_boxes_xyxy=nativeboxes[0].cpu().numpy(),raw_query_max_scores=maxscores[0].cpu().numpy(),raw_query_ordinals=np.arange(rawlog.shape[0],dtype=np.int64),boxes_xyxy=boxes,scores=scores,detector_query_index=qi,text_labels=np.asarray(result[0]['text_labels'],dtype=str))
        row=dict(req,caption=cap,request_key_sha256=request_key(req),decoded_RGB_SHA256=hashlib.sha256(array.tobytes()).hexdigest(),detections=bind(path),detection_count=len(scores),padding_guard=dict(PASS=True,raw_shape=list(output.logits.shape),input_token_length=int(inputs['input_ids'].shape[1]),active_token_count=len(active),finite_active_logits=True,all_inactive_exact_negative_inf=True,finite_FP32_boxes=True,full_everyquery_FP32_active_logits_and_same_tensor_probs=True))
        rows.append(row); detections[request_key(req)]=(boxes,scores,qi)
        if (n+1)%25==0: print(json.dumps(dict(requests_completed=n+1,total=len(requests),source_forward_calls=forwards)),flush=True)
    state_after=old.state_sha(model); assert state_before==state_after
    assert sha(OLD/'model_assets/model.safetensors')==p['detector']['weights_SHA256']
    associated=[associate(row,by_key[request_key(row)],*detections[request_key(row)]) for row in targets]
    assert all(all(a[k]==t[k] for k in INPUT_FIELDS) for a,t in zip(associated,targets))
    write_lines(OUT/'REQUEST_DETECTIONS.jsonl',rows); write_lines(OUT/'ASSOCIATIONS.jsonl',associated)
    counts={reason:sum(x['reason']==reason for x in associated) for reason in sorted({x['reason'] for x in associated})}
    write(OUT/'DETECTOR_SUMMARY.json',dict(completed=True,target_instances=len(targets),query_slots=8280,unique_RGB=435,unique_RGB_caption_requests=len(requests),association_reasons=counts,associated=sum(x['status']=='ASSOCIATED' for x in associated),held_U=sum(x['status']=='U' for x in associated),source_forward_calls=forwards,expected_forward_calls=len(requests)+1,first_repeat_bitexact=True,state_SHA256_before=state_before,state_SHA256_after=state_after,state_unchanged=True,FP32_noautocast_TF32false=True,custom_kernels_disabled=True,NMS=False,original_q_L_box_order_unchanged=True,requested_category_side_information=True,GT_instance_or_reference_forward=False,source_provenance_forward=False,peak_GPU_memory_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start,current223=0,fit=0,PnP=0,fully_solved=False))
    write(OUT/'DETECTOR_OUTPUT_SEAL.json',dict(completed=True,PASS=True,entrylock=bind(ENTRY),decisions=bind(DECISIONS),logging_addendum=bind(ADDENDUM),code=bind(__file__),assets=bind(ASSETS),runtime=bind(RUNTIME),source_inputs=e['source_inputs'],per_request=bind(OUT/'REQUEST_DETECTIONS.jsonl'),associations=bind(OUT/'ASSOCIATIONS.jsonl'),summary=bind(OUT/'DETECTOR_SUMMARY.json'),caption_gate=bind(OUT/'CAPTION_TOKEN_GATE.json'),loading_info=bind(OUT/'PRETRAINED_LOADING_INFO.json'),first_repeat=bind(OUT/'FIRST_REQUEST_REPEAT.json'),postprocess_contract=bind(POST_CONTRACT),guard_contract=bind(LOGIT_CONTRACT),current223_fit_PnP=0,fully_solved=False))
    print(json.dumps(bind(OUT/'DETECTOR_OUTPUT_SEAL.json')),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=('generated','freeze','run'))
    {'generated':generated,'freeze':freeze,'run':run}[ap.parse_args().stage]()
