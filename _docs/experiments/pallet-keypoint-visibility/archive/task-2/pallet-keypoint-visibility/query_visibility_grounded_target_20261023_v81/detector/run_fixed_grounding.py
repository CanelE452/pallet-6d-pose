"""Fixed-caption source GroundingDINO pilot; generated/freeze do not infer.

Only run, explicitly invoked by the parent after review, opens source RGB or
loads detector weights. Source identity/query/box metadata is never forwarded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import time
import numpy as np

R = Path(__file__).resolve().parent
V = R.parent
DECISIONS = V / 'EXPERIMENT_DECISIONS_BEFORE_ASSETS_AND_SOURCE.json'
DECISION_SHA = '34e406371c3452f7013d6f871568f218f69d9e668e1ea8115919b11d0472040e'
RUNTIME = V / 'runtime/RUNTIME_IMPORT_ONLY_SEAL.json'
RUNTIME_SHA = '13fb5d64dcde87314fb37badd2fba15d452a85083bbd29d74384671cc8b83933'
ASSETS = V / 'model_assets/ASSET_SEAL.json'
ASSET_SHA = '37ead31a70f2deab84c04e4c8a1da084d92b50190ecdac53ffe3ca5799c1921d'
TOKEN_PROOF = V / 'ROOT_TOKENIZER_PREFLIGHT.json'
TOKEN_PROOF_SHA = '6ec46ae0a75a648f649214e782733fd61790eeed0395c49ac3d2453494bc3476'
INPUT_SEAL = V / 'input_preparation/SEAL.json'
MANIFEST = V / 'input_preparation/INPUTS_RGB_ONLY.jsonl'
UNIQUE = V / 'input_preparation/UNIQUE_RGB_INPUTS_ONLY.jsonl'
ENTRY = R / 'DETECTOR_ENTRYLOCK_V2.json'
LOGIT_CONTRACT = R / 'OFFICIAL_LOGIT_AND_LOADING_GUARD_CONTRACT.json'
OUT = R / 'outputs'
INPUT_FIELDS = {'input_id', 'image', 'image_sha256', 'raw_hw', 'initial_points', 'roi_side', 'query_box_xyxy'}
RGB_FIELDS = {'input_id', 'image', 'image_sha256', 'raw_hw'}
ASSOC_EXTRA = {'rgb_input_id', 'status', 'reason', 'selected_detection_index',
               'selected_detector_query_index', 'selected_detection_score',
               'association_iou', 'selected_box_xyxy_raw', 'selected_box_xyxy_prompt'}

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

def decisions():
    assert sha(DECISIONS) == DECISION_SHA
    p = read(DECISIONS)
    assert p['frozen'] and p['scope']['expected_unique_RGB'] == 435
    assert p['grounding']['box_threshold'] == .4 and p['grounding']['text_threshold'] == .3
    assert p['grounding']['fixed_caption'] == '. '.join(p['grounding']['fixed_phrases_in_order']) + '.'
    assert len(p['grounding']['fixed_phrases_in_order']) == 22
    return p

def validate_input(row, rgb_only=False):
    assert set(row) == (RGB_FIELDS if rgb_only else INPUT_FIELDS), 'GT/unknown field in source input'
    assert isinstance(row['input_id'], str) and row['input_id']
    assert isinstance(row['image'], str) and Path(row['image']).is_absolute()
    assert isinstance(row['image_sha256'], str) and len(row['image_sha256']) == 64
    assert all(c in '0123456789abcdef' for c in row['image_sha256'])
    hw = row['raw_hw']
    assert len(hw) == 2 and all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in hw)
    if not rgb_only:
        assert np.asarray(row['initial_points'], dtype=np.float64).shape == (8, 2)
        assert math.isfinite(row['roi_side']) and row['roi_side'] > 0
        assert row['query_box_xyxy'] is None or np.asarray(row['query_box_xyxy'], dtype=np.float64).shape == (4,)
    return row

def validate_manifests(targets, rgbs, target_count=1035, rgb_count=435):
    assert len(targets) == target_count and len(rgbs) == rgb_count
    assert len({x['input_id'] for x in targets}) == len(targets)
    assert len({x['input_id'] for x in rgbs}) == len(rgbs)
    by_sha = {}
    for row in rgbs:
        validate_input(row, True)
        assert row['image_sha256'] not in by_sha
        by_sha[row['image_sha256']] = row
    for row in targets:
        validate_input(row)
        rgb = by_sha[row['image_sha256']]
        assert all(row[k] == rgb[k] for k in ('image', 'image_sha256', 'raw_hw'))
    assert set(by_sha) == {x['image_sha256'] for x in targets}
    return by_sha

def validate_output_arrays(logits, attention_mask, pred_boxes):
    """Official contrastive head intentionally pads/masks logits with -inf.

    Validation is independent of box/text thresholds and never changes logits.
    Invalid attention positions and padding beyond input_len must all be -inf.
    """
    logits = np.asarray(logits)
    attention_mask = np.asarray(attention_mask)
    pred_boxes = np.asarray(pred_boxes)
    assert logits.ndim == 3 and logits.shape[-1] == 256 and logits.dtype == np.float32
    assert attention_mask.ndim == 2 and attention_mask.shape[0] == logits.shape[0]
    assert 0 < attention_mask.shape[1] <= 256 and np.isin(attention_mask, [0, 1]).all()
    assert pred_boxes.shape == (*logits.shape[:2], 4) and pred_boxes.dtype == np.float32
    assert np.isfinite(pred_boxes).all()
    valid = np.zeros((logits.shape[0], 256), dtype=bool)
    valid[:, :attention_mask.shape[1]] = attention_mask.astype(bool)
    valid = np.broadcast_to(valid[:, None, :], logits.shape)
    assert np.isfinite(logits[valid]).all(), 'Nonfinite active-token contrastive logit'
    assert np.isneginf(logits[~valid]).all(), 'Nonactive/padded logits must be exactly -inf'
    return True

def validate_loading_info(info):
    assert isinstance(info, dict)
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        assert key in info and info[key] == [], f'Incomplete pretrained loading: {key}'
    return True

def valid_box(box):
    if box is None: return False
    a = np.asarray(box, dtype=np.float64)
    return a.shape == (4,) and bool(np.isfinite(a).all() and a[2] > a[0] and a[3] > a[1])

def ious(input_box, boxes):
    a = np.asarray(input_box, np.float64)
    b = np.asarray(boxes, np.float32).astype(np.float64)
    good = np.isfinite(b).all(1) & (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    inter = np.maximum(0, np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2])).prod(1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    out = np.zeros(len(b), np.float64)
    np.divide(inter, area_a + area_b - inter, out=out, where=good & ((area_a + area_b - inter) > 0))
    out[~good] = -1
    return out

def associate(input_row, rgb_id, boxes, scores, query_index):
    validate_input(input_row)
    boxes = np.asarray(boxes, np.float32)
    scores = np.asarray(scores, np.float32)
    query_index = np.asarray(query_index, np.int64)
    assert boxes.shape == (len(scores), 4) and query_index.shape == scores.shape
    assert np.isfinite(scores).all() and len(np.unique(query_index)) == len(query_index)
    result = dict(input_row, rgb_input_id=rgb_id, status='U', reason='INVALID_INITIAL_QUERY_BOX',
                  selected_detection_index=None, selected_detector_query_index=None,
                  selected_detection_score=None, association_iou=None,
                  selected_box_xyxy_raw=None, selected_box_xyxy_prompt=None)
    if not valid_box(input_row['query_box_xyxy']): return result
    if not len(scores): result['reason'] = 'NO_FIXED_THRESHOLD_DETECTIONS'; return result
    overlap = ious(input_row['query_box_xyxy'], boxes)
    # Continuous un-clipped detector boxes; actual FP32 score then original
    # model query ordinal. Labels/text/GT/source identity never participate.
    order = np.lexsort((query_index, -scores.astype(np.float64), -overlap))
    selected = int(order[0])
    if overlap[selected] < .1: result['reason'] = 'MAX_INPUT_BOX_IOU_BELOW_0P1'; return result
    h, w = input_row['raw_hw']
    prompt = boxes[selected].copy()
    prompt[[0, 2]] = np.clip(prompt[[0, 2]], 0, w)
    prompt[[1, 3]] = np.clip(prompt[[1, 3]], 0, h)
    if not valid_box(prompt): result['reason'] = 'INVALID_CLIPPED_SAM_PROMPT'; return result
    result.update(status='ASSOCIATED', reason='OK', selected_detection_index=selected,
                  selected_detector_query_index=int(query_index[selected]),
                  selected_detection_score=float(scores[selected]), association_iou=float(overlap[selected]),
                  selected_box_xyxy_raw=boxes[selected].tolist(), selected_box_xyxy_prompt=prompt.tolist())
    assert set(result) == INPUT_FIELDS | ASSOC_EXTRA
    return result

def generated():
    p = decisions()
    row = dict(input_id='opaque', image='/generated/no-file.png', image_sha256='0' * 64,
               raw_hw=[30, 40], initial_points=[[i, i + .25] for i in range(8)],
               roi_side=16., query_box_xyxy=[10., 10., 20., 20.])
    assert associate(row, 'rgb', [[10, 10, 20, 20]], [.5], [7])['association_iou'] == 1
    a = associate(row, 'rgb', [[10, 10, 20, 20]] * 3, [.6, .9, .9], [2, 6, 4])
    assert a['selected_detection_index'] == 2 and a['selected_detector_query_index'] == 4
    assert a['selected_detection_score'] == float(np.float32(.9))
    # Exactly inclusive association 0.1, unlike strict detector thresholds.
    boundary = dict(row, query_box_xyxy=[0., 0., 1., 10.])
    assert associate(boundary, 'rgb', [[0, 0, 1, 1]], [.5], [0])['status'] == 'ASSOCIATED'
    assert associate(row, 'rgb', [[0, 0, 1, 1]], [.9], [0])['status'] == 'U'
    assert associate(row, 'rgb', np.empty((0, 4)), [], [])['reason'] == 'NO_FIXED_THRESHOLD_DETECTIONS'
    for box in (None, [None] * 4, [0, 0, 0, 1], [0, 1, 2, 0]):
        assert associate(dict(row, query_box_xyxy=box), 'rgb', [[10, 10, 20, 20]], [.5], [0])['status'] == 'U'
    clipped = associate(dict(row, query_box_xyxy=[0, 0, 10, 10]), 'rgb', [[-2, -3, 10, 10]], [.5], [1])
    assert clipped['selected_box_xyxy_prompt'] == [0., 0., 10., 10.]
    outside = associate(dict(row, query_box_xyxy=[-10, -10, -1, -1]), 'rgb', [[-10, -10, -1, -1]], [.5], [0])
    assert outside['reason'] == 'INVALID_CLIPPED_SAM_PROMPT'
    for key in ('reference_xy', 'mask', 'GT_class', 'source', 'annotation', 'quality_label'):
        try: validate_input(dict(row, **{key: 'forbidden'}))
        except AssertionError: pass
        else: raise AssertionError(key)
    rgbs = [dict(input_id='r', image=row['image'], image_sha256=row['image_sha256'], raw_hw=row['raw_hw'])]
    validate_manifests([row], rgbs, 1, 1)
    rng = np.random.default_rng(20261023)
    for _ in range(300):
        lo = rng.uniform(-10, 20, (12, 2)); hi = lo + rng.uniform(.1, 20, (12, 2))
        bs = np.c_[lo, hi].astype(np.float32); sc = rng.uniform(.4, 1., 12).astype(np.float32)
        qi = rng.permutation(12).astype(np.int64); ov = ious(row['query_box_xyxy'], bs)
        want = sorted(range(12), key=lambda i: (-ov[i], -float(sc[i]), int(qi[i])))[0]
        got = associate(row, 'r', bs, sc, qi)
        if ov[want] >= .1 and valid_box(np.clip(bs[want], [0, 0, 0, 0], [40, 30, 40, 30])):
            assert got['selected_detection_index'] == want
        else: assert got['status'] == 'U'
    for length in (77, 256):
        mask = np.ones((2, length), np.int64)
        mask[1, min(12, length - 1)] = 0  # Real in-input attention padding.
        logits = np.full((2, 3, 256), -np.inf, np.float32)
        valid = np.zeros((2, 256), bool); valid[:, :length] = mask.astype(bool)
        logits[np.broadcast_to(valid[:, None, :], logits.shape)] = .5
        boxes = np.full((2, 3, 4), .25, np.float32)
        assert validate_output_arrays(logits, mask, boxes)
        valid_pos = (0, 0, 0)
        invalid_pos = (1, 0, min(12, length - 1))
        for pos, bad in [(valid_pos, np.nan), (valid_pos, np.inf), (valid_pos, -np.inf),
                         (invalid_pos, .0), (invalid_pos, np.inf), (invalid_pos, np.nan)]:
            malformed = logits.copy(); malformed[pos] = bad
            try: validate_output_arrays(malformed, mask, boxes)
            except AssertionError: pass
            else: raise AssertionError(('Malformed logit accepted', length, pos, bad))
        malformed_boxes = boxes.copy(); malformed_boxes[0, 0, 0] = np.nan
        try: validate_output_arrays(logits, mask, malformed_boxes)
        except AssertionError: pass
        else: raise AssertionError('Malformed predicted box accepted')
    good_info = {key: [] for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}
    assert validate_loading_info(good_info)
    for key in good_info:
        for info in ({**good_info, key: ['bad']}, {k: v for k, v in good_info.items() if k != key}):
            try: validate_loading_info(info)
            except AssertionError: pass
            else: raise AssertionError(('Loading guard accepted', key))
    contract = read(LOGIT_CONTRACT)
    check(contract['official_modeling_source'])
    write(R / 'GENERATED_DETECTOR_TESTS_V3.json', dict(PASS=True, code=bind(__file__), decisions=bind(DECISIONS), guard_contract=bind(LOGIT_CONTRACT),
          generated_geometry_ties_invalid_boxes_clipping_and_300_random_replays=True,
          generated_77_and_256_token_finite_active_negative_infinite_padding_attention_mask_and_malformed_rejections=True,
          generated_loading_info_required_four_empty_lists_and_malformed_rejections=True,
          strict_detector_thresholds=dict(box=.4, text=.3), association_inclusive_IoU=.1,
          GT_whitelist_rejection=True, original_q_L_never_modified=True,
          source_RGB_GT_weights_tokenizer_or_model_constructors_forward_fit_current_PnP=0))
    print('GENERATED_PASS', flush=True)

def freeze():
    p = decisions()
    proof = read(R / 'GENERATED_DETECTOR_TESTS_V3.json')
    assert proof['PASS'] and proof['code'] == bind(__file__)
    assert proof['guard_contract'] == bind(LOGIT_CONTRACT)
    check(read(LOGIT_CONTRACT)['official_modeling_source'])
    assert sha(RUNTIME) == RUNTIME_SHA and read(RUNTIME)['PASS']
    assert sha(ASSETS) == ASSET_SHA
    assets = read(ASSETS); assert assets['completed'] and assets['PASS']
    for file in assets['files']: check(file)
    assert sha(V / 'model_assets/model.safetensors') == p['grounding']['weight_SHA256']
    source = read(INPUT_SEAL); assert source['completed'] and source['PASS']
    for file in source['files']: check(file)
    rt = read(RUNTIME)
    for module in rt['module_sources']: check(module)
    assert sha(TOKEN_PROOF) == TOKEN_PROOF_SHA
    token_proof = read(TOKEN_PROOF)
    assert token_proof['PASS'] and token_proof['use_fast'] is False and token_proof['caption'] == p['grounding']['fixed_caption']
    assert token_proof['token_count'] == 77 and len(token_proof['token_ids'][0]) == 77
    write(ENTRY, dict(completed=True, code=bind(__file__), generated=bind(R / 'GENERATED_DETECTOR_TESTS_V3.json'), guard_contract=bind(LOGIT_CONTRACT),
          decisions=bind(DECISIONS), assets=bind(ASSETS), runtime=bind(RUNTIME),
          root_tokenizer_preflight=bind(TOKEN_PROOF), processor_use_fast=False,
          source_inputs=dict(manifest=bind(MANIFEST), unique_manifest=bind(UNIQUE), seal=bind(INPUT_SEAL)),
          output_directory=str(OUT), model_directory=str(V / 'model_assets'),
          forward_only_fields='Processor-produced RGB tensor/pixelmask plus globally constant caption token tensors.',
          forward_rejects='GT/reference/semantic class/source identity/provenance/query/roughbox.',
          parent_must_review_and_explicitly_invoke_run=True, current223=False, fit=False, PnP=0))
    print(json.dumps(bind(ENTRY)), flush=True)

def state_sha(model):
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        h.update(key.encode()); h.update(str(tuple(tensor.shape)).encode()); h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def run():
    # Fail closed before source manifest bodies, model imports or source RGB.
    p = decisions(); e = read(ENTRY)
    assert e['completed'] and e['code'] == bind(__file__)
    for key in ('generated', 'decisions', 'assets', 'runtime', 'root_tokenizer_preflight', 'guard_contract'): check(e[key])
    check(read(e['guard_contract']['path'])['official_modeling_source'])
    for val in e['source_inputs'].values(): check(val)
    assert not OUT.exists(), 'Preserve previous/failed outputs; no silent restart'
    assets = read(e['assets']['path'])
    for val in assets['files']: check(val)
    rt = read(e['runtime']['path'])
    for val in rt['module_sources']: check(val)
    assert Path(sys.executable).resolve() == Path(rt['python']).resolve()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.path.insert(0, str(V / 'runtime/overlay'))
    import importlib.metadata
    import random
    import torch
    from PIL import Image
    from transformers import GroundingDinoConfig, GroundingDinoProcessor, GroundingDinoForObjectDetection
    for package, version in rt['versions'].items(): assert importlib.metadata.version(package) == version
    assert torch.cuda.is_available()
    torch.set_num_threads(1); torch.manual_seed(20261023); torch.cuda.manual_seed_all(20261023)
    np.random.seed(20261023); random.seed(20261023)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    targets, rgbs = lines(MANIFEST), lines(UNIQUE)
    rgb_by_sha = validate_manifests(targets, rgbs)
    OUT.mkdir(); (OUT / 'detections').mkdir()
    write(OUT / 'EXECUTION_STARTED.json', dict(entrylock=bind(ENTRY), original_target_count=len(targets), unique_RGB_count=len(rgbs), source_forward_calls=0, fit_current_PnP=0))
    model_dir = e['model_directory']
    processor = GroundingDinoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False, use_fast=False)
    assert type(processor.image_processor).__name__ == 'GroundingDinoImageProcessor' and type(processor.tokenizer).__name__ == 'BertTokenizer'
    config = GroundingDinoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    config.disable_custom_kernels = True
    model, loading_info = GroundingDinoForObjectDetection.from_pretrained(model_dir, config=config, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32, use_safetensors=True, output_loading_info=True)
    validate_loading_info(loading_info)
    write(OUT / 'PRETRAINED_LOADING_INFO.json', dict(PASS=True, output_loading_info=loading_info, all_four_required_lists_empty=True, before_first_source_RGB_decode=True, config_custom_kernels_disabled=True, guard_contract=bind(LOGIT_CONTRACT)))
    model = model.cuda().eval().requires_grad_(False)
    assert model.config.disable_custom_kernels and all(x.dtype == torch.float32 for x in model.parameters())
    caption = p['grounding']['fixed_caption']
    caption_ids = processor.tokenizer(caption, return_tensors='pt')['input_ids']
    assert caption_ids.shape[0] == 1 and caption_ids.shape[1] <= 256
    assert caption_ids.tolist() == read(e['root_tokenizer_preflight']['path'])['token_ids']
    caption_list = caption_ids[0].tolist(); caption_sha = hashlib.sha256(np.asarray(caption_list, np.int64).tobytes()).hexdigest()
    write(OUT / 'CAPTION_TOKEN_GATE.json', dict(PASS=True, caption=caption, token_ids=caption_list, token_count=len(caption_list), token_ids_SHA256=caption_sha, root_preflight_exact=True, use_fast=False, fixed_all_22_phrases=True, before_first_source_RGB_decode=True))
    state_before = state_sha(model); start = time.monotonic(); rows = []; detections = {}; forward_count = 0
    for n, rgb in enumerate(rgbs):
        # The hash/shape checks are mandatory, with no JPEG tolerance repair.
        assert sha(rgb['image']) == rgb['image_sha256'], rgb['input_id']
        with Image.open(rgb['image']) as image: array = np.asarray(image.convert('RGB')).copy()
        assert list(array.shape) == [*rgb['raw_hw'], 3] and array.dtype == np.uint8
        inputs = processor(images=array, text=caption, return_tensors='pt')
        assert set(inputs) <= {'pixel_values', 'pixel_mask', 'input_ids', 'attention_mask', 'token_type_ids'}
        assert torch.equal(inputs['input_ids'], caption_ids)
        inputs = inputs.to('cuda')
        with torch.inference_mode():
            output = model(**inputs); forward_count += 1
            assert output.logits.dtype == torch.float32 and output.pred_boxes.dtype == torch.float32
            validate_output_arrays(output.logits.cpu().numpy(), inputs['attention_mask'].cpu().numpy(), output.pred_boxes.cpu().numpy())
            if n == 0:
                repeat = model(**inputs); forward_count += 1
                assert torch.equal(output.logits, repeat.logits) and torch.equal(output.pred_boxes, repeat.pred_boxes)
                write(OUT / 'FIRST_RGB_REPEAT.json', dict(PASS=True, logits_maxdiff=0., boxes_maxdiff=0., same_inputs=True, source_forward_calls=2, state_unchanged=state_sha(model) == state_before))
            result = processor.post_process_grounded_object_detection(output, inputs['input_ids'], threshold=.4, text_threshold=.3, target_sizes=[tuple(rgb['raw_hw'])])[0]
            query_max = output.logits.sigmoid().max(-1).values[0]
            kept = torch.nonzero(query_max > .4).flatten()
            assert torch.equal(result['scores'], query_max[kept])
            boxes = result['boxes'].cpu().numpy(); scores = result['scores'].cpu().numpy(); qi = kept.cpu().numpy()
            text_labels = list(result['text_labels'])
            assert boxes.shape == (len(scores), 4) and len(text_labels) == len(scores)
            assert boxes.dtype == scores.dtype == np.float32 and np.isfinite(boxes).all()
            path = OUT / 'detections' / (rgb['input_id'] + '.npz')
            np.savez_compressed(path, boxes_xyxy=boxes, scores=scores, detector_query_index=qi, text_labels=np.asarray(text_labels, dtype=str), raw_query_max_scores=query_max.cpu().numpy(), raw_normalized_boxes=output.pred_boxes[0].cpu().numpy())
        record = dict(rgb_input_id=rgb['input_id'], image=rgb['image'], image_sha256=rgb['image_sha256'], raw_hw=rgb['raw_hw'], decoded_RGB_SHA256=hashlib.sha256(array.tobytes()).hexdigest(), detections=bind(path), detection_count=len(scores), caption_token_ids_SHA256=caption_sha)
        rows.append(record); detections[rgb['image_sha256']] = (boxes, scores, qi)
        if (n + 1) % 25 == 0: print(json.dumps(dict(unique_RGB_completed=n + 1, total=435, source_forward_calls=forward_count)), flush=True)
    state_after = state_sha(model); assert state_after == state_before
    assert sha(V / 'model_assets/model.safetensors') == p['grounding']['weight_SHA256']
    associated = [associate(row, rgb_by_sha[row['image_sha256']]['input_id'], *detections[row['image_sha256']]) for row in targets]
    assert len(associated) == 1035 and all(all(a[k] == t[k] for k in INPUT_FIELDS) for a, t in zip(associated, targets))
    write_lines(OUT / 'RGB_DETECTIONS.jsonl', rows)
    write_lines(OUT / 'ASSOCIATIONS.jsonl', associated)
    counts = {reason: sum(x['reason'] == reason for x in associated) for reason in sorted({x['reason'] for x in associated})}
    write(OUT / 'DETECTOR_SUMMARY.json', dict(completed=True, target_instances=1035, query_slots=8280, unique_RGB=435, detections=sum(x['detection_count'] for x in rows), association_reasons=counts, associated=sum(x['status'] == 'ASSOCIATED' for x in associated), held_U=sum(x['status'] == 'U' for x in associated), caption_token_count=len(caption_list), unchanged_caption_every_image=True, original_queries_L_boxes_and_order_unchanged=True, precision='FP32/noautocast/TF32false', custom_kernels_disabled=True, NMS=False, state_SHA256_before=state_before, state_SHA256_after=state_after, state_unchanged=True, first_repeat_bitexact=True, source_forward_calls=forward_count, expected_forward_calls=436, peak_GPU_memory_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic() - start, GT_or_per_image_class_forward=False, source_provenance_forward=False, current223=0, fit=0, PnP=0, fully_solved=False))
    write(OUT / 'DETECTOR_OUTPUT_SEAL.json', dict(completed=True, PASS=True, entrylock=bind(ENTRY), decisions=bind(DECISIONS), code=bind(__file__), assets=bind(ASSETS), runtime=bind(RUNTIME), source_inputs=e['source_inputs'], per_RGB=bind(OUT / 'RGB_DETECTIONS.jsonl'), associations=bind(OUT / 'ASSOCIATIONS.jsonl'), summary=bind(OUT / 'DETECTOR_SUMMARY.json'), caption_gate=bind(OUT / 'CAPTION_TOKEN_GATE.json'), loading_info=bind(OUT / 'PRETRAINED_LOADING_INFO.json'), guard_contract=bind(LOGIT_CONTRACT), first_repeat=bind(OUT / 'FIRST_RGB_REPEAT.json'), current223_fit_PnP=0, fully_solved=False))
    print(json.dumps(bind(OUT / 'DETECTOR_OUTPUT_SEAL.json')), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('generated', 'freeze', 'run'))
    {'generated': generated, 'freeze': freeze, 'run': run}[ap.parse_args().stage]()
