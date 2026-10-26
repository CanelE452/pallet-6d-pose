"""Frozen full-frame CLIP features from the original 1003 SYN RGB images only.

Only the input-only FULL_FRAME_RGB_INPUTS sidecar is read.  Mask targets,
queries, pose, visibility annotations and current223 files are never opened.
Image metadata enters provenance only; the encoder receives whole RGB arrays.
"""
from pathlib import Path
import collections
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
EXPECTED_ENCODER = '88a5ddad13bb8a285a18a47f7ac1c7ee13a94fe79d00a7c1a7bae37a798c9556'
EXPECTED_INPUTS = '22b4ea7f28ed8de8b13e83d6a83aa5774a26b6e41da135b3920dccea022e3a75'
EXPECTED_SOURCE = '17e702ac534acb941c01da397fd4752eb8e76e16820771889cf4de64887675af'
EXPECTED_WEIGHTS = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
N = 1003
BATCH = 2


def safe(path):
    p = Path(path)
    if not p.is_absolute() or any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError('Symlink: ' + str(a))
    return p


def binding(path, expected=None):
    p = safe(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    result = dict(path=str(p), sha256=h.hexdigest(), size_bytes=p.stat().st_size)
    if expected is not None:
        assert result['sha256'] == expected, result
    return result


def write_json(path, obj):
    with safe(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')


def decode_rgb(row):
    path = safe(row['RGB']['path'])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['RGB']['sha256'], str(path)
    assert len(raw) == row['RGB']['size_bytes'], str(path)
    with Image.open(io.BytesIO(raw)) as image:
        mode = image.mode
        rgb = np.array(image.convert('RGB'), copy=True)
    assert rgb.dtype == np.uint8 and rgb.shape == tuple(row['raw_hw']) + (3,)
    digest = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    return rgb, digest, mode


def main():
    started = time.monotonic()
    code = binding(Path(__file__).absolute())
    encoder = binding(HERE / 'fullframe_clip_v267.py', EXPECTED_ENCODER)
    source = binding(HERE / 'OFFICIAL_SOURCE_PROVENANCE.json', EXPECTED_SOURCE)
    source_obj = json.loads(safe(source['path']).read_bytes())
    for name in ('clip/clip_model.py', 'clip/clip_surgery_model.py'):
        bound = source_obj['source_files'][name]
        assert binding(bound['path'], bound['sha256']) == bound
    weight = binding(HERE / 'pretrained/ViT-B-16.pt', EXPECTED_WEIGHTS)
    download = binding(HERE / 'pretrained/DOWNLOAD_PROVENANCE.json')
    audit = binding(HERE / 'data_audit/COMPLETE.json')
    audit_obj = json.loads(safe(audit['path']).read_bytes())
    inputs = binding(HERE / 'data_audit/FULL_FRAME_RGB_INPUTS.jsonl', EXPECTED_INPUTS)
    assert audit_obj['PASS'] and audit_obj['frames'] == N and audit_obj['TRAIN_VAL'] == [768, 235]
    assert audit_obj['outputs']['FULL_FRAME_RGB_INPUTS.jsonl'] == inputs
    rows = [json.loads(line) for line in safe(inputs['path']).read_bytes().splitlines()]
    assert len(rows) == N and [r['frame_index'] for r in rows] == list(range(N))
    assert collections.Counter(r['split'] for r in rows) == {'train': 768, 'validation': 235}
    for row in rows:
        assert set(row) == {'RGB', 'frame_index', 'id', 'raw_hw', 'scope', 'split'}
        assert row['scope'] == 'UNCROPPED_FULL_FRAME; no query/box/pose mask input'
        safe(row['RGB']['path'])
    out = safe(HERE / 'syn_fullframe_features_v1')
    assert not out.exists(), str(out)
    protocol = dict(completed=True, pre_forward=True, code=code, encoder_code=encoder,
                    official_source_provenance=source, generic_weights=weight, weight_download=download,
                    input_only_RGB_manifest=inputs, audit_complete=audit, frames=N, batch_size=BATCH,
                    arms=['PLAIN_CLIP', 'CLIP_SURGERY_ARCH'], full_input=True,
                    preprocess='Complete frame PIL RGB -> bicubic512x512 square -> CLIP mean/std; no crop',
                    output='L2-normalized1024x512 patch tokens float16; CLS retained separately but not mask-decoder input',
                    existing_source_order_unchanged=True, targets_queries_pose_or_current_read=False,
                    official_text_feature_surgery=False, optimizer_updates=0, output_directory=str(out))
    protocol_path = HERE / 'SYN_FEATURE_PROTOCOL_BEFORE_FORWARD.json'
    write_json(protocol_path, protocol)
    # Byte-pinned execution of the dedicated encoder only, after the protocol.
    p = safe(encoder['path'])
    module = types.ModuleType('v267_fullframe_syn_encoder')
    module.__file__ = str(p)
    sys.modules[module.__name__] = module
    exec(compile(p.read_bytes(), str(p), 'exec'), module.__dict__)
    assert module.ARMS == ('PLAIN_CLIP', 'CLIP_SURGERY_ARCH')
    out.mkdir()
    write_json(out / 'STARTED.json', dict(protocol=binding(protocol_path), code=code, frames=N,
                                        SYN_only=True, current_reads=0, optimizer_updates=0))
    frame_order = [None] * N
    proofs = {}
    filenames = {'PLAIN_CLIP': ('TOKENS_PLAIN.npy', 'CLS_PLAIN.npy'),
                 'CLIP_SURGERY_ARCH': ('TOKENS_CS.npy', 'CLS_CS.npy')}
    try:
        for arm in module.ARMS:
            arm_started = time.monotonic()
            model, proof = module.load_visual(weight['path'], arm, 'cuda:0')
            before = module.state_sha(model)
            tokens_path, cls_path = [safe(out / name) for name in filenames[arm]]
            assert not tokens_path.exists() and not cls_path.exists()
            features = np.lib.format.open_memmap(tokens_path, mode='w+', dtype=np.float16, shape=(N, 1024, 512))
            cls_features = np.lib.format.open_memmap(cls_path, mode='w+', dtype=np.float16, shape=(N, 512))
            repeat_exact = False
            for start in range(0, N, BATCH):
                selected = rows[start:start + BATCH]
                images = []
                for row in selected:
                    rgb, rgb_sha, mode = decode_rgb(row)
                    index = row['frame_index']
                    order = dict(row, decoded_RGB_sha256=rgb_sha, decoded_original_mode=mode,
                                 decoded_RGB_shape=list(rgb.shape))
                    if frame_order[index] is None:
                        frame_order[index] = order
                    else:
                        assert frame_order[index] == order, ('Cross-arm input mismatch', index)
                    images.append(rgb)
                # The entire RGB arrays are the sole model input.  No q/pose/GT.
                result = module.extract_fullframe_tokens(model, images, device='cuda:0', batch_size=BATCH)
                assert result['dense_tokens'].shape == (len(images), 1024, 512)
                assert np.array_equal(result['raw_hw'], np.asarray([r['raw_hw'] for r in selected]))
                if start == 0:
                    repeat = module.extract_fullframe_tokens(model, images, device='cuda:0', batch_size=BATCH)
                    assert np.array_equal(result['dense_tokens'], repeat['dense_tokens'])
                    assert np.array_equal(result['cls_token'], repeat['cls_token'])
                    repeat_exact = True
                features[start:start + len(images)] = result['dense_tokens'].astype(np.float16)
                cls_features[start:start + len(images)] = result['cls_token'].astype(np.float16)
                if (start // BATCH + 1) % 25 == 0 or start + BATCH >= N:
                    print(json.dumps(dict(arm=arm, frames=min(start + BATCH, N), total=N,
                                          elapsed_seconds=time.monotonic() - arm_started)), flush=True)
            features.flush()
            cls_features.flush()
            assert np.isfinite(features).all() and np.isfinite(cls_features).all()
            del features, cls_features
            after = module.state_sha(model)
            assert before == after and repeat_exact
            proofs[arm] = dict(encoder=proof, frozen_state_before=before, frozen_state_after=after,
                               state_unchanged=True, first_batch_repeat_FP32_byte_exact=True,
                               frames=N, encoded_input_SHA_validated=N, decoded_full_RGB_only=True,
                               outputs={name: binding(out / name) for name in filenames[arm]},
                               elapsed_seconds=time.monotonic() - arm_started)
            write_json(out / (arm + '_COMPLETE.json'), proofs[arm])
            del model
            print(json.dumps(dict(arm_completed=arm, proof=proofs[arm])), flush=True)
        assert len({p['encoder']['pretrained_visual_state_sha256'] for p in proofs.values()}) == 1
        frame_path = safe(out / 'FRAME_ORDER.jsonl')
        with frame_path.open('x') as f:
            f.write(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in frame_order))
        names = ['TOKENS_PLAIN.npy', 'TOKENS_CS.npy', 'CLS_PLAIN.npy', 'CLS_CS.npy', 'FRAME_ORDER.jsonl',
                 'PLAIN_CLIP_COMPLETE.json', 'CLIP_SURGERY_ARCH_COMPLETE.json', 'STARTED.json']
        complete = dict(completed=True, PASS=True, schema='V267_ORIGINAL_SYN1003_FULLFRAME_FROZEN_FEATURES_V1',
                        protocol=binding(protocol_path), code=code, encoder_code=encoder,
                        input_RGB_manifest=inputs, official_source_provenance=source, generic_weights=weight,
                        frames=N, TRAIN_VAL=[768, 235], token_shape=[N, 1024, 512], token_dtype='float16',
                        arms=proofs, outputs={name: binding(out / name) for name in names},
                        same_fullframe_encoded_and_decoded_RGB_both_arms=True,
                        no_KP_crop_or_pose_or_mask_as_encoder_input=True, official_text_feature_surgery=False,
                        target_or_query_sidecar_reads=0, current_or_real_reads=0, optimizer_updates=0,
                        elapsed_seconds=time.monotonic() - started,
                        runtime=dict(torch=torch.__version__, numpy=np.__version__, GPU=torch.cuda.get_device_name(0)))
        write_json(out / 'COMPLETE.json', complete)
        print(json.dumps(dict(PASS=True, completed=True, complete=binding(out / 'COMPLETE.json'))), flush=True)
    except BaseException as error:
        write_json(out / 'FAILURE.json', dict(completed=False, PASS=False, error_type=type(error).__name__,
                                             error=str(error), preserve_existing_outputs=True,
                                             optimizer_updates=0, current_reads=0))
        raise


if __name__ == '__main__':
    main()
