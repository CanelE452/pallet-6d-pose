"""Original SYN1003 whole RGB256, untouched M4 soft targets and GT bands.

No M0/query/pose/current reads.  Only the already-frozen V267 CS head is
forwarded to obtain fixed32 coarse context; no encoder or optimizer is run.
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
from torch.nn import functional as F

HERE = Path(__file__).absolute().parent
PREV = HERE.parent/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK'


def safe(path):
    p = Path(path)
    assert p.is_absolute() and not any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')), str(p)
    for a in reversed((p, *p.parents)):
        assert not a.is_symlink(), str(a)
    return p


def binding(path, expected=None):
    p = safe(path); h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):h.update(block)
    b = dict(path=str(p), sha256=h.hexdigest(), size_bytes=p.stat().st_size)
    assert expected is None or b['sha256'] == expected
    return b


def checked(bound):
    assert binding(bound['path']) == bound
    return safe(bound['path'])


def write_json(path, obj):
    with safe(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def module(bound, name):
    p = checked(bound); m = types.ModuleType(name); m.__file__ = str(p); sys.modules[name] = m
    exec(compile(p.read_bytes(), str(p), 'exec'), m.__dict__)
    return m


def main():
    started = time.monotonic(); torch.set_num_threads(1)
    freeze_path = PREV/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json'
    freeze_bound = binding(freeze_path, '7da54bae6f02e70e33641ff78782325e981c7755a3f69327e85aa51ec9895b84')
    freeze = json.loads(safe(freeze_path).read_bytes())
    audit = json.loads(safe(PREV/'data_audit/COMPLETE.json').read_bytes())
    inputs_bound, targets_bound = [audit['outputs'][n] for n in ('FULL_FRAME_RGB_INPUTS.jsonl', 'VISIBLE_MASK_TARGETS.jsonl')]
    inputs = [json.loads(x) for x in checked(inputs_bound).read_bytes().splitlines()]
    targets = [json.loads(x) for x in checked(targets_bound).read_bytes().splitlines()]
    assert len(inputs) == len(targets) == 1003
    assert [r['frame_index'] for r in inputs] == [r['frame_index'] for r in targets] == list(range(1003))
    assert all(r['id'] == t['id'] and r['split'] == t['split'] for r, t in zip(inputs, targets))
    old_policy = json.loads(checked(freeze['protocol']).read_bytes())
    tokens_bound = old_policy['source_bindings']['tokensCS']
    draws_bound = old_policy['source_bindings']['draws']
    out = safe(HERE/'syn_refine_inputs_v1'); assert not out.exists(); out.mkdir()
    model_bound = binding(HERE/'model_mask_refine_v268.py')
    write_json(out/'PREPARATION_BEFORE_INPUT_ARRAYS.json', dict(completed=True, code=binding(Path(__file__).absolute()),
        model_code=model_bound, frozen_V267=freeze_bound, RGB_inputs=inputs_bound, M4_targets=targets_bound,
        source_tokens=tokens_bound, source_draws=draws_bound, source_coarse_head_code=freeze['model_code'],
        source_coarse_checkpoint=freeze['checkpoints']['CLIP_SURGERY_ARCH'],
        RGB_transform='Whole original RGB ->PIL bicubic512 ->avgpool2/255 ->float16 RGB256[0,1]',
        M4_transform='uint8>127 ->torch bilinear512 align_corners=False antialias=True ->avgpool2 ->float16 soft256',
        boundary='soft256>=.5; two-sided GT dilation minus erosion radius2 with outside-frame0; weights only, targets untouched',
        no_query_M0_pose_or_real_reads=True, encoder_forwards=0, optimizer_updates=0))
    m = module(model_bound, 'v268_input_model_math')
    rgb = np.lib.format.open_memmap(safe(out/'RGB256_FP16.npy'), mode='w+', dtype=np.float16, shape=(1003, 3, 256, 256))
    mask = np.zeros((1003, 1, 256, 256), np.float16)
    band = np.zeros((1003, 1, 256, 256), bool)
    fit = np.asarray([r['split'] == 'train' for r in inputs], bool)
    assert fit.sum() == 768 and (~fit).sum() == 235
    frame_order = []
    for i, (r, t) in enumerate(zip(inputs, targets)):
        image_path = checked(r['RGB'])
        with Image.open(io.BytesIO(image_path.read_bytes())) as im:
            raw = np.asarray(im.convert('RGB')); assert raw.shape == tuple(r['raw_hw'])+(3,)
            resized = np.array(im.convert('RGB').resize((512,512), Image.Resampling.BICUBIC), copy=True)
        x = torch.from_numpy(resized).permute(2, 0, 1)[None].float()
        rgb[i] = (F.avg_pool2d(x, 2, 2)/255).numpy()[0].astype(np.float16)
        with Image.open(io.BytesIO(checked(t['M4']).read_bytes())) as im:a = np.array(im)
        assert a.dtype == np.uint8 and a.shape == tuple(r['raw_hw']) and int((a > 127).sum()) == t['M4_area']
        y = torch.from_numpy((a > 127).astype(np.float32))[None, None]
        y = F.avg_pool2d(F.interpolate(y, (512,512), mode='bilinear', align_corners=False, antialias=True), 2, 2).clamp(0,1)
        mask[i] = y.numpy()[0].astype(np.float16)
        # Derive the band from precisely the stored FP16 target representation.
        band[i] = m.boundary_band(torch.from_numpy(mask[i:i+1].astype(np.float32))).numpy()[0]
        frame_order.append(dict(frame_index=i, id=r['id'], split=r['split'], raw_hw=r['raw_hw'], RGB=r['RGB'], M4=t['M4']))
    rgb.flush(); assert np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1; del rgb
    np.savez_compressed(safe(out/'MASK_TARGETS.npz'), mask=mask, boundary_band=band, fit=fit, frame_ids=np.arange(1003,dtype=np.int32))
    draws = np.load(checked(draws_bound), allow_pickle=False)
    assert draws.shape == (1000, 8) and fit[draws].all()
    safe(out/'FIT_DRAWS1000x8.npy').write_bytes(checked(draws_bound).read_bytes())
    old = module(freeze['model_code'], 'v268_frozen_V267_CS_head')
    coarse_head = old.FullMaskHead().eval().requires_grad_(False)
    cp = torch.load(io.BytesIO(checked(freeze['checkpoints']['CLIP_SURGERY_ARCH']).read_bytes()), map_location='cpu')
    coarse_head.load_state_dict(cp['state_dict'], strict=True)
    before = old.state_sha(coarse_head.state_dict())
    tokens = np.load(checked(tokens_bound), mmap_mode='r', allow_pickle=False)
    coarse = old.infer_mask(coarse_head, tokens, device='cpu', batch_size=32)['mask_logits']
    assert coarse.shape == (1003,1,32,32) and np.isfinite(coarse).all() and before == old.state_sha(coarse_head.state_dict())
    np.save(safe(out/'COARSE_LOGITS_CS.npy'), coarse)
    safe(out/'FRAME_ORDER.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in frame_order))
    frequency = float(mask[fit].astype(np.float64).mean())
    names = ['PREPARATION_BEFORE_INPUT_ARRAYS.json', 'RGB256_FP16.npy', 'MASK_TARGETS.npz', 'FIT_DRAWS1000x8.npy', 'COARSE_LOGITS_CS.npy', 'FRAME_ORDER.jsonl']
    write_json(out/'COMPLETE.json', dict(completed=True, PASS=True, code=binding(Path(__file__).absolute()),
        model_code=model_bound, frozen_V267=freeze_bound, source_tokens=tokens_bound, source_draws=draws_bound,
        outputs={n:binding(out/n) for n in names}, frames=1003, FIT_DEV=[768,235],
        RGB_shape=[1003,3,256,256], mask_shape=[1003,1,256,256], boundary_radius=2,
        FIT_foreground_frequency=frequency, positive_weight=.5/frequency, negative_weight=.5/(1-frequency),
        boundary_tokens_FIT_DEV=[int(band[fit].sum()),int(band[~fit].sum())],
        coarse_head_state_before=before, coarse_head_state_after=old.state_sha(coarse_head.state_dict()),
        frozen_coarse_head_forward_frames=1003, encoder_forwards=0, optimizer_updates=0,
        current_or_real_reads=0, M0_or_query_or_pose_reads=0, elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(PASS=True,complete=binding(out/'COMPLETE.json'))), flush=True)


if __name__ == '__main__':
    main()
