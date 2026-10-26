"""Frozen three-arm full-frame encoder/head inference; query readout after maps.

No current annotation, target M4, new fit, sliding window, query crop or PnP.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import sys
import types
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

HERE = Path(__file__).absolute().parent
ARMS = ('EVA_STANDARD', 'EVA_CSA', 'EVA_DECLIP')


def safe(p):
    p = Path(p)
    assert p.is_absolute() and not any(t in str(p).lower() for t in ('v112', 'cache', '.pyc', '..')), str(p)
    for a in (p, *p.parents): assert not a.is_symlink(), str(a)
    return p


def binding(p):
    p = safe(p); h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return dict(path=str(p), sha256=h.hexdigest(), size_bytes=p.stat().st_size)


def read(b):
    p = safe(b['path']); raw = p.read_bytes()
    assert len(raw) == b['size_bytes'] and hashlib.sha256(raw).hexdigest() == b['sha256'], str(p)
    return raw


def module(b, name):
    m = types.ModuleType(name); m.__file__ = b['path']; sys.modules[name] = m
    exec(compile(read(b), m.__file__, 'exec'), m.__dict__)
    return m


def write(p, obj):
    with safe(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def all_buffer_sha(model):
    h = hashlib.sha256()
    for name, t in sorted(model.named_buffers()):
        x = t.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


def sample_queries(maps, queries):
    scores = np.zeros(len(queries), np.float32)
    for i, row in enumerate(queries):
        if not row['available']: continue
        xy = np.asarray(row['original_q0'], np.float64); h, w = row['raw_hw']
        assert xy.shape == (2,) and np.isfinite(xy).all()
        grid = torch.tensor((xy + .5) / np.array([w,h]) * 2 - 1, dtype=torch.float32).reshape(1,1,1,2)
        field = torch.from_numpy(maps[int(row['frame_index'])]).reshape(1,1,32,32)
        scores[i] = float(F.grid_sample(field, grid, mode='bilinear', padding_mode='border', align_corners=False)[0,0,0,0])
    return scores


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    freeze = json.loads(safe(HERE / 'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json').read_bytes())
    assert freeze['PASS'] and freeze['threshold'] == .5
    policy = json.loads(read(freeze['protocol']))
    assert tuple(policy['train_arms']) == ARMS
    assert binding(Path(__file__).absolute()) == policy['inference_code']
    encoder_api = module(policy['encoder_code'], 'v269_current_frozen_encoder')
    assert tuple(encoder_api.ARMS) == ARMS
    head_api = module(freeze['model_code'], 'v269_current_frozen_head')
    io_api = module(policy['current_input_helper'], 'v269_current_full_RGB_io')
    source_features = json.loads(read(policy['source_feature_complete']))
    assert source_features['PASS'] and source_features['frames'] == 1003
    assert source_features['encoder_code'] == policy['encoder_code']
    inputs = json.loads(read(policy['current_input_complete']))
    assert inputs['PASS'] and inputs['accessible_verified_frames'] == 223
    frames = [json.loads(x) for x in read(inputs['frames']).splitlines()]
    assert [r['frame_index'] for r in frames] == list(range(223))
    out = safe(HERE / 'current_inference_v1'); assert not out.exists(); out.mkdir()
    replay = safe(HERE / 'fullframe_replay_v1'); assert not replay.exists(); replay.mkdir()
    write(out / 'BEFORE_CURRENT_ENCODER_FORWARDS.json', dict(completed=True,PASS=True,
          configuration=binding(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json'), inference_code=policy['inference_code'],
          encoder_code=policy['encoder_code'], source_feature_complete=policy['source_feature_complete'],
          current_input_complete=policy['current_input_complete'], frames=inputs['frames'],
          current_GT_or_annotation_reads=0, query_sidecar_reads=0, query_neural_inputs=0, new_optimizers=0,
          unused_query_fields_may_exist_in_frame_metadata=True,
          readout='EntireRGB512, frozen dense feature -> CPU NumPyFP16 -> FP32 maskhead, full32logits before query read.'))
    images = np.empty((223,512,512,3), np.uint8)
    for i, frame in enumerate(frames):
        rgb = io_api.read_rgb(frame)  # rechecks encoded/decoded RGB SHA and raw H/W
        assert tuple(rgb.shape[:2]) == tuple(frame['raw_hw'])
        images[i] = np.asarray(Image.fromarray(rgb).resize((512,512), Image.Resampling.BICUBIC))
    fixture_index = np.array([0,56,112,168,222], np.int64)
    fixture = dict(frame_index=fixture_index)
    payload = dict(frame_index=np.arange(223, dtype=np.int64)); proofs = {}; device = 'cuda:0'
    for arm in ARMS:
        encoder, encoder_proof = encoder_api.load_visual(arm, device=device)
        before = encoder_api.state_sha(encoder); buffers_before = all_buffer_sha(encoder)
        syn_proof = source_features['encoder_proofs'][arm]
        assert before == syn_proof['state_before'] and buffers_before == syn_proof['named_buffers_before']
        head = head_api.FullMaskHead().to(device).eval().requires_grad_(False)
        cp = torch.load(io.BytesIO(read(freeze['checkpoints'][arm])), map_location='cpu')
        assert cp['arm'] == arm and cp['steps'] == 1000 and cp['threshold'] == .5
        assert cp['model_code_sha256'] == freeze['model_code']['sha256'] and cp['feature_key'] == 'tokens' + arm
        head.load_state_dict(cp['state_dict'], strict=True); head_before = encoder_api.state_sha(head)
        raw = np.empty((223,32,32), np.float32)
        fixture_tokens = np.empty((5,1024,512), np.float16)
        first_repeat_exact = False
        with torch.inference_mode():
            for first in range(0,223,2):
                stop = min(first+2,223)
                normalized, _ = encoder_api.preprocess_fullframe(images[first:stop])
                dense = encoder(normalized.to(device=device, dtype=torch.float32))
                # Identical NumPy half roundtrip used for frozen SYN NPY features.
                half = dense.cpu().numpy().astype(np.float16)
                x = torch.as_tensor(half, device=device, dtype=torch.float32)
                logits = head(x)
                assert logits.shape == (stop-first,1,32,32) and bool(torch.isfinite(logits).all())
                if first == 0:
                    assert torch.equal(logits,head(x)); first_repeat_exact = True
                raw[first:stop] = logits[:,0].cpu().numpy()
                for k, frame_index in enumerate(fixture_index):
                    if first <= frame_index < stop: fixture_tokens[k] = half[frame_index-first]
                if stop % 50 == 0 or stop == 223:
                    print(json.dumps(dict(arm=arm,full_frames=stop,total=223)),flush=True)
        assert encoder_api.state_sha(encoder) == before and all_buffer_sha(encoder) == buffers_before
        assert encoder_api.state_sha(head) == head_before
        payload[arm + '_logit'] = raw
        fixture['tokens' + arm] = fixture_tokens; fixture[arm + '_logit'] = raw[fixture_index]
        proofs[arm] = dict(encoder=encoder_proof, encoder_state_before=before, encoder_state_after=encoder_api.state_sha(encoder),
                          all_named_buffers_before=buffers_before, all_named_buffers_after=all_buffer_sha(encoder),
                          identical_frozen_SYN_encoder_state_and_buffers=True, checkpoint=freeze['checkpoints'][arm],
                          head_state_sha256=head_before, first_head_batch_repeat_exact=first_repeat_exact,
                          full_frame_encoder_forwards=223, full_frame_head_forwards=223, stored_full223_features=False,
                          FP16_storage_before_head_identical_to_SYN=True, current_GT_inputs=0, query_neural_inputs=0, new_optimizer_updates=0)
        del encoder, head; torch.cuda.empty_cache()
    maps_path = safe(HERE / 'CURRENT_FRAME_MASKS.npz'); assert not maps_path.exists()
    np.savez_compressed(maps_path, **payload)
    fixture_path = safe(replay / 'FORWARD_INPUTS5.npz'); np.savez_compressed(fixture_path, **fixture)
    write(out / 'ALL_FULL_FRAME_MAPS_SAVED_BEFORE_QUERY_READ.json', dict(completed=True,PASS=True,
          frame_masks=binding(maps_path),forward_replay_inputs=binding(fixture_path),frames=223,arms=list(ARMS),
          current_GT_or_annotation_reads=0,query_sidecar_reads=0,query_neural_inputs=0,new_optimizers=0,
          unused_query_fields_may_exist_in_frame_metadata=True))
    # Existing query/native/availability sidecar is first read after all maps seal.
    queries = [json.loads(x) for x in read(inputs['queries_manifest']).splitlines()]
    assert len(queries) == 1784 and sum(row['available'] for row in queries) == 1739
    sampled = {arm:sample_queries(payload[arm + '_logit'], queries) for arm in ARMS}
    predictions = []
    for i, row in enumerate(queries):
        assert row['global_index'] == i and row['frame_index'] == i // 8 and row['native_slot'] == i % 8
        value = {key:row[key] for key in ('global_index','frame_index','frame_id','row_id','native_slot','available')}
        for arm in ARMS:
            if not row['available']:
                value[arm] = dict(p_target=None,E_logit=None,p_E=None,pred='U',
                                  interpretation='MASK_PRESENCE_PROXY_E_NOT_CAUSE')
                continue
            score = float(sampled[arm][i]); p = float(1 / (1 + np.exp(-np.clip(score,-80,80))))
            value[arm] = dict(p_target=p,E_logit=-score,p_E=1-p,
                              pred='U' if not row['available'] else 'D' if score>=0 else 'E',
                              interpretation='MASK_PRESENCE_PROXY_E_NOT_CAUSE')
        predictions.append(value)
    prediction_path = safe(HERE / 'CURRENT_ALL1784_PREDICTIONS.jsonl')
    with prediction_path.open('x') as f:
        for row in predictions: f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    write(HERE / 'CURRENT_PREDICTIONS_FROZEN.json',dict(completed=True,PASS=True,
          configuration=binding(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json'),predictions=binding(prediction_path),
          frame_masks=binding(maps_path),forward_replay_inputs=binding(fixture_path),
          current_frame_inputs=inputs['frames'],current_query_inputs=inputs['queries_manifest'],
          current_input_complete=policy['current_input_complete'],source_feature_complete=policy['source_feature_complete'],
          inference_code=binding(Path(__file__).absolute()),encoder_code=policy['encoder_code'],head_code=freeze['model_code'],
          model_proofs=proofs,rows=1784,frames=223,supported=1739,original_input_U=45,
          fixed_threshold=.5,sample_padding_mode='border',sample_align_corners=False,
          query_sampling='Original(q+.5)/HW, bilinear raw32logit border alignFalse then sigmoid.',
          full_frame_maps_saved_before_query_sidecar_read=binding(out/'ALL_FULL_FRAME_MAPS_SAVED_BEFORE_QUERY_READ.json'),
          unused_query_fields_may_exist_in_frame_metadata=True,query_neural_inputs=0,
          original_query_native_order_unchanged=True,current_GT_annotation_or_real_fit_reads=0,
          new_optimizer_updates=0,new_PnP=0,mask_presence_is_not_external_occlusion_cause=True,
          public_COCO_DeCLIP_dense_adaptation_history_disclosed=True,
          whole_image_scope='All223completeRGB512, no slide/crop; three frozen official encoders; identical SYN-trained32maskheads.'))
    print(json.dumps(dict(completed=True,PASS=True,freeze=binding(HERE/'CURRENT_PREDICTIONS_FROZEN.json'))),flush=True)


if __name__ == '__main__':
    main()
