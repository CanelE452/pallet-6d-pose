"""Three frozen official EVA/DeCLIP encoders on original SYN1003 full RGBs.

Input manifest contains original RGB bindings only; no M4, native/query or real
image input is loaded. All entire images undergo the same512 stretch and norm.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import sys
import time
import types
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).absolute().parent
OLD = HERE.parent / 'V267_FULL_IMAGE_CLIP_VISIBLE_MASK'
ENCODER_SHA = '1531607974dff5fd9ba0ae68fe41f915085b2d44109e276777bceed190fcac60'
AUDIT_SHA = 'ef2a2e0ac722df7a2f9cfc84d7333726107df20b1b39feba0db7ce673da2d483'
RGB_SHA = '22b4ea7f28ed8de8b13e83d6a83aa5774a26b6e41da135b3920dccea022e3a75'


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


def write(p, obj):
    with safe(p).open('x') as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')


def buffer_sha(model):
    h = hashlib.sha256()
    for name, t in sorted(model.named_buffers()):
        x = t.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


def main():
    start = time.monotonic(); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    source = safe(HERE / 'fullframe_eva_declip_v269.py'); raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ENCODER_SHA
    api = types.ModuleType('v269_sealed_encoder'); api.__file__ = str(source); sys.modules[api.__name__] = api
    exec(compile(raw, str(source), 'exec'), api.__dict__)
    smoke = json.loads(safe(HERE / 'ENCODER_ARTIFICIAL_SMOKE.json').read_bytes())
    assert smoke['PASS'] and smoke['encoder_code']['sha256'] == ENCODER_SHA
    design = json.loads(safe(HERE / 'DESIGN_BEFORE_DATA_FORWARDS.json').read_bytes())
    auditpath = safe(OLD / 'data_audit/COMPLETE.json'); audit_raw = auditpath.read_bytes()
    assert hashlib.sha256(audit_raw).hexdigest() == AUDIT_SHA
    audit = json.loads(audit_raw)
    assert audit['PASS'] and audit['frames'] == 1003 and audit['TRAIN_VAL'] == [768,235]
    manifest = audit['outputs']['FULL_FRAME_RGB_INPUTS.jsonl']; assert manifest['sha256'] == RGB_SHA
    path = safe(manifest['path']); manifest_raw = path.read_bytes()
    assert len(manifest_raw) == manifest['size_bytes'] and hashlib.sha256(manifest_raw).hexdigest() == RGB_SHA
    frames = [json.loads(x) for x in manifest_raw.splitlines()]
    assert [x['frame_index'] for x in frames] == list(range(1003))
    assert sum(x['split'] == 'train' for x in frames) == 768 and sum(x['split'] == 'validation' for x in frames) == 235
    out = safe(HERE / 'syn_fullframe_features_v1'); assert not out.exists(); out.mkdir()
    write(out / 'BEFORE_SYN_FIRST_DATA_FORWARD.json', dict(completed=True, PASS=True,
          code=binding(Path(__file__).absolute()), encoder_code=binding(source), artificial_smoke=binding(HERE / 'ENCODER_ARTIFICIAL_SMOKE.json'),
          root_design=binding(HERE / 'DESIGN_BEFORE_DATA_FORWARDS.json'), input_manifest=manifest,
          official_assets=binding(HERE / 'OFFICIAL_ASSETS_COMPLETE.json'), arms=list(api.ARMS),
          first_data_forward_not_started=True, new_optimizers=0, mask_query_pose_or_current_inputs=0,
          output_shape=[1003,1024,512], output_dtype='float16', batch_size=2,
          shared_transform='Entire original RGB -> PIL bicubic512 stretch -> /255 -> official CLIP mean/std.',
          normalized_feature_readout='Official norm/projection512; CLS excluded; per-token L2 normalization.'))
    with safe(out / 'FRAME_ORDER.jsonl').open('xb') as f: f.write(manifest_raw)
    # Decode/hash once; retain only complete resized uint8 frames in bounded host RAM.
    images = np.empty((1003,512,512,3), np.uint8)
    for i, frame in enumerate(frames):
        b = frame['RGB']; encoded = safe(b['path']).read_bytes()
        assert len(encoded) == b['size_bytes'] and hashlib.sha256(encoded).hexdigest() == b['sha256']
        with Image.open(io.BytesIO(encoded)) as im:
            assert tuple(reversed(im.size)) == tuple(frame['raw_hw'])
            images[i] = np.asarray(im.convert('RGB').resize((512,512), Image.Resampling.BICUBIC))
    print(json.dumps(dict(all1003_RGB_SHA_verified=True, decode_seconds=time.monotonic()-start)), flush=True)
    proofs = {}; outputs = {}; device = 'cuda:0'
    for arm in api.ARMS:
        model, proof = api.load_visual(arm, device=device)
        state_before, buffers_before = api.state_sha(model), buffer_sha(model)
        target = safe(out / ('TOKENS_' + arm + '.npy'))
        array = np.lib.format.open_memmap(target, mode='w+', dtype=np.float16, shape=(1003,1024,512))
        arm_started = time.monotonic(); repeated_first = False; norm_max = 0.
        with torch.inference_mode():
            for first in range(0,1003,2):
                stop = min(first+2,1003)
                # Each stored image is a full512 frame; same-size PIL resize is identity.
                batch, _ = api.preprocess_fullframe(images[first:stop])
                values = model(batch.to(device=device, dtype=torch.float32))
                assert values.shape == (stop-first,1024,512) and bool(torch.isfinite(values).all())
                if first == 0:
                    repeat = model(batch.to(device=device, dtype=torch.float32))
                    assert torch.equal(values,repeat); repeated_first = True
                norm_max = max(norm_max, float((values.norm(dim=-1)-1).abs().max()))
                array[first:stop] = values.cpu().numpy().astype(np.float16)
                if stop % 100 == 0 or stop == 1003:
                    elapsed = time.monotonic()-arm_started
                    print(json.dumps(dict(arm=arm, full_frames=stop, total=1003, elapsed_seconds=elapsed,
                                          estimated_remaining_seconds=elapsed/stop*(1003-stop))), flush=True)
        array.flush(); del array
        state_after, buffers_after = api.state_sha(model), buffer_sha(model)
        assert state_after == state_before and buffers_after == buffers_before
        proof.update(first_actual_batch_repeat_exact=repeated_first, actual_full_frame_forwards=1003,
                     first_batch_repeat_forward_rows=2, per_token_FP32_norm_max_abs_error=norm_max,
                     state_before=state_before, state_after=state_after, named_buffers_before=buffers_before,
                     named_buffers_after=buffers_after, all_persistent_and_nonpersistent_buffers_unchanged=True,
                     elapsed_seconds=time.monotonic()-arm_started, raw_RGB_SHA_verified_frames=1003,
                     original_frame_order_preserved=True, mask_or_query_or_pose_or_current_reads=0)
        proofs[arm] = proof; outputs['tokens' + arm] = binding(target)
        del model; torch.cuda.empty_cache()
    assert proofs['EVA_STANDARD']['state_before'] == proofs['EVA_CSA']['state_before']
    write(out / 'COMPLETE.json', dict(completed=True,PASS=True,frames=1003,train_val=[768,235],
          code=binding(Path(__file__).absolute()), encoder_code=binding(source),
          source_audit=binding(auditpath), source_RGB_manifest=manifest, frame_order=binding(out/'FRAME_ORDER.jsonl'),
          before_first_data_forward=binding(out/'BEFORE_SYN_FIRST_DATA_FORWARD.json'), artificial_smoke=binding(HERE/'ENCODER_ARTIFICIAL_SMOKE.json'),
          outputs=outputs,encoder_proofs=proofs,output_shape=[1003,1024,512],output_dtype='float16',
          actual_forward_frame_counts={a:1003 for a in api.ARMS},first_batch_repeat_frames_each=2,
          source_RGB_SHA_verified_frames=1003,new_optimizer_updates=0,original_M4_query_native_pose_or_current_reads=0,
          released_DeCLIP_public_COCO_pretraining_disclosed=True, elapsed_seconds=time.monotonic()-start))
    print(json.dumps(dict(completed=True,PASS=True,complete=binding(out/'COMPLETE.json'))),flush=True)


if __name__ == '__main__':
    main()
