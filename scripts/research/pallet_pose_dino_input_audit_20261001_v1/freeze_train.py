"""Freeze input-only DINO evidence on the existing eligible TRAIN population.

No fit, error target, VAL quality, real reference, new pose solve or download.
Full FP16 patch tokens are retained so descriptor extraction can be audited.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as TF
from scripts.research.pallet_sensors_submission_v1.posefix_contract_math import axis_aligned_crop_matrix
from . import visual_features as V

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_pose_dino_input_audit_20261001_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
OLD_DOC = ROOT / '_docs/experiments/pallet_pose_residual_direction_audit_20261001_v1'
PARENT_RAW = ROOT / 'data/pallet/results/pallet_pose_union_selection_20261001_v1'
BACKBONE = ROOT / '_docs/experiments/pallet_type_selftrain_v1/selftrain_recovery_v1/dino_localization/BACKBONE.json'
ASSETS = ROOT / 'outputs/pallet_type_selftrain_v1/selftrain_recovery_v1/dino_localization_assets'
MODELS = ('R0', 'DIVERSE251_s1', 'DIVERSE251_s2', 'DIVERSE251_s3')
HYP = ('long-face-front', 'short-face-front')
MEAN = np.array([123.68, 116.78, 103.94], np.float32)
READS = set()
WRITES = set()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def verify(b):
    p = ROOT / b['path']
    assert sha(p) == b['sha256'], b['path']
    if 'bytes' in b:
        assert p.stat().st_size == b['bytes'], b['path']


def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def wide_matrix(box):
    m = axis_aligned_crop_matrix(box)
    m[:2, 2] += [144., 192.]
    return m


def crop_rgb(image, matrix):
    bgr = cv2.warpAffine(image, matrix[:2], (576, 768), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Preserve the old wide extractor's float32 subtract/add round trip.
    return (bgr[:, :, ::-1].astype(np.float32) - MEAN).transpose(2, 0, 1)


def load_backbone(binding):
    b = read(ROOT / binding['path'])
    for entry in b['code'] + [b['checkpoint'], b['license'], b['loader']]:
        verify(entry)
    sys.path.insert(0, str(ASSETS / 'dinov2'))
    from dinov2.hub.backbones import dinov2_vits14
    # Loading the exact local state directly prohibits a network fallback.
    model = dinov2_vits14(pretrained=False)
    state = torch.load(ROOT / b['checkpoint']['path'], map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).cuda()
    assert model.embed_dim == 384 and model.patch_size == 14
    assert sum(p.numel() for p in model.parameters()) == b['parameters']
    assert not any(p.requires_grad for p in model.parameters())
    return model


@torch.no_grad()
def extract(model, rgb):
    x = torch.as_tensor((rgb + MEAN[:, None, None])[None], device='cuda') / 255
    x = TF.interpolate(x, size=(784, 588), mode='bilinear', align_corners=False)
    mean = torch.tensor([.485, .456, .406], device=x.device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=x.device)[None, :, None, None]
    z = model.forward_features((x - mean) / std)['x_norm_patchtokens']
    z = z.reshape(1, 56, 42, 384).permute(0, 3, 1, 2).contiguous()
    assert torch.isfinite(z).all()
    return z[0].cpu().numpy().astype(np.float16)


def install_guard(allowed):
    allowed = {Path(p).resolve() for p in allowed}
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to(ROOT):
            return
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing:
            assert p.is_relative_to(RAW), ('INPUT_ONLY_WRITE_DENIED', str(p))
            WRITES.add(str(p.relative_to(ROOT)))
            return
        if p.suffix in ('.py', '.pyc'):
            return
        assert p in allowed or p.is_relative_to(RAW), ('INPUT_ONLY_FILE_DENIED', str(p))
        READS.add(str(p.relative_to(ROOT)))
    sys.addaudithook(hook)


def seal():
    old = read(OLD_DOC / 'AUDIT_PROTOCOL.json')
    for b in old['inputs'].values():
        verify(b)
    paths = [Path(__file__), Path(V.__file__), ROOT / 'scripts/research/pallet_sensors_submission_v1/posefix_contract_math.py']
    p = dict(schema='pallet_pose_dino_input_audit_v1', created_at=datetime.now(timezone.utc).isoformat(),
        previous_input_audit=bind(OLD_DOC / 'AUDIT_PROTOCOL.json'), inputs=old['inputs'],
        backbone=bind(BACKBONE), codes=[bind(x) for x in paths],
        frames=2598, train_only=True, original_candidate_validity_preserved=True,
        crop=dict(height=768, width=576, old_expansion=1.25, old_shape=[384, 288], shift_xy=[144, 192],
                  box='current frozen R0 selected box; shared by every whole-pose candidate', additional_padding=0),
        tokens=dict(shape=[384, 56, 42], dtype='float16', resize_hw=[784, 588],
                    inference_dtype='float32', batch=1, pretrained_weights_updated=False,
                    matmul_allow_tf32=False, cudnn_allow_tf32=False, cudnn_benchmark=False,
                    normalization='ImageNet mean/std after old PoseFix RGB mean subtract/add'),
        descriptor='Mean of raw frozen384 patch channels at supported8 physical-pose projections plus support_count/8; no center, no L2 token normalization. Bilinear half-pixel grid with border clamping; geometric support is not visibility.',
        output_dim=385, normalize='R0 original valid TRAIN descriptors float32 mean/std floor1e-6; candidate-minus-anchor after float32 normalization castfloat64',
        full_tokens_retained=True, old_cache_reuse=False,
        no_fit=True, no_label_values=True, no_VAL_features=True, no_real_features=True,
        no_new_pose=True, no_download=True, stable_joint_improvement_achieved=False)
    save(DOC / 'INPUT_PROTOCOL.json', p)
    print('INPUT_PROTOCOL_SEALED', bind(DOC / 'INPUT_PROTOCOL.json'), flush=True)


def freeze():
    start = time.monotonic()
    p = read(DOC / 'INPUT_PROTOCOL.json')
    assert p['train_only'] and p['frames'] == 2598 and p['output_dim'] == 385
    for b in [p['previous_input_audit'], p['backbone'], *p['inputs'].values(), *p['codes']]:
        verify(b)
    metadata = read(ROOT / p['inputs']['metadata']['path'])
    contract = read(ROOT / p['inputs']['source_contract']['path'])
    eligible = set(contract['fit_eligibility']['eligible_ids']['TRAIN'])
    indices = np.array([i for i, r in enumerate(metadata) if r['id'] in eligible], np.int64)
    rows = [metadata[i] for i in indices]
    assert len(rows) == len(eligible) == 2598 and all(r['split'] == 'TRAIN' for r in rows)
    predlock = read(ROOT / p['inputs']['source_predictions_lock']['path'])
    assert predlock['complete'] and predlock['frames'] == 5120
    assert predlock['models'] == list(MODELS)
    r0receipt = read(ROOT / predlock['receipts']['R0']['path'])
    verify(predlock['receipts']['R0'])
    assert r0receipt['complete'] and r0receipt['model'] == 'R0' and r0receipt['frames'] == 5120
    assert r0receipt['protocol'] == predlock['protocol']
    predictions = [r0receipt['files'][i] for i in indices]
    b = read(BACKBONE)
    allowed = [DOC / 'INPUT_PROTOCOL.json', BACKBONE,
               *[ROOT / e['path'] for e in p['inputs'].values()],
               *[ROOT / e['path'] for e in b['code'] + [b['checkpoint'], b['license'], b['loader']]],
               *[ROOT / r['image']['path'] for r in rows], *[ROOT / e['path'] for e in predictions]]
    install_guard(allowed)
    with np.load(ROOT / p['inputs']['features']['path'], allow_pickle=False) as z:
        assert z['ids'].tolist() == [r['id'] for r in metadata]
        assert z['split'][indices].tolist() == ['TRAIN'] * len(rows)
        masks = {m: z[m + '_valid'][indices].copy() for m in MODELS}
        anchor = z['R0_GEO_index'][indices].copy()
    poses = read(ROOT / p['inputs']['poses']['path'])
    assert poses['ids'] == [r['id'] for r in metadata]
    for m in MODELS:
        assert masks[m].shape == (2598, 2) and masks[m].dtype == bool
        assert np.array_equal(masks[m].any(1), masks['R0'].any(1))
        assert int(masks[m].sum()) == 5194
    assert int((~masks['R0'].any(1)).sum()) == 1
    RAW.mkdir(parents=True, exist_ok=True)
    token_path = RAW / 'TRAIN_TOKENS.npy'
    assert not token_path.exists(), 'Do not restart a partial extraction without inspecting its process and incident.'
    assert not (RAW / 'TRAIN_APPEARANCE_INPUTS.json').exists()
    tokens = np.lib.format.open_memmap(token_path, mode='w+', dtype=np.float16, shape=(2598, 384, 56, 42))
    matrices = np.zeros((2598, 3, 3), np.float64)
    values = {m: np.zeros((2598, 2, 385), np.float32) for m in MODELS}
    support = {m: np.zeros((2598, 2, 8), bool) for m in MODELS}
    frame_receipts = []
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_backbone(p['backbone'])
    forwards = 0
    for j, (row, pb) in enumerate(zip(rows, predictions)):
        verify(pb)
        pred = read(ROOT / pb['path'])
        assert pred['id'] == row['id'] and pred['model'] == 'R0'
        assert pred['protocol_sha'] == predlock['protocol']['sha256']
        assert pred['checkpoint_sha'] == r0receipt['checkpoint']['sha256']
        frame = dict(id=row['id'], source_index=int(indices[j]), image=row['image'], prediction=pb,
                     image_forward=False, all_original_candidates_invalid=not bool(masks['R0'][j].any()))
        if not masks['R0'][j].any():
            tokens[j] = 0
        else:
            verify(row['image'])
            # Read bytes through Python so image access is visible to the audit hook.
            image = cv2.imdecode(np.frombuffer((ROOT / row['image']['path']).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            assert image is not None and list(image.shape[:2]) == row['hw']
            selection = pred['prediction']['selected_index']
            assert selection is not None
            selected = pred['prediction']['candidates'][selection]
            matrix = wide_matrix(selected['box_xyxy'])
            matrices[j] = matrix
            rgb = crop_rgb(image, matrix)
            feature = extract(model, rgb)
            forwards += 1
            tokens[j] = feature
            frame.update(image_forward=True, crop_rgb_sha=array_sha(rgb), token_sha=array_sha(feature),
                         matrix=matrix.tolist())
            for m in MODELS:
                hypotheses = {h['name']: h for h in poses['records'][m][row['id']]['hypotheses']}
                for k, name in enumerate(HYP):
                    if not masks[m][j, k]:
                        continue
                    result = V.describe(hypotheses[name]['pose'], row['K'], matrix, row['hw'], feature)
                    values[m][j, k] = result['descriptor']
                    support[m][j, k] = result['support8']
        frame_receipts.append(frame)
        if (j + 1) % 100 == 0 or j == len(rows) - 1:
            tokens.flush()
            print('DINO_TRAIN_INPUT', j + 1, '/', len(rows), 'forwards', forwards, 'seconds', round(time.monotonic() - start, 1), flush=True)
    tokens.flush()
    del tokens
    mean = values['R0'][masks['R0']].mean(0)
    std = np.maximum(values['R0'][masks['R0']].std(0), np.float32(1e-6))
    assert mean.dtype == std.dtype == np.float32
    arrays = dict(ids=np.asarray([r['id'] for r in rows]), source_index=indices, anchor_index=anchor,
                  crop_matrices=matrices, mean385=mean, std385=std)
    summaries = {}
    for m in MODELS:
        assert np.isfinite(values[m]).all() and not values[m][~masks[m]].any()
        assert np.array_equal(values[m][:, :, -1], support[m].sum(2).astype(np.float32) / 8)
        arrays[m + '_appearance385'] = values[m]
        arrays[m + '_valid'] = masks[m]
        arrays[m + '_support8'] = support[m]
        s = support[m].sum(2)[masks[m]]
        summaries[m] = dict(valid_candidates=int(masks[m].sum()), descriptor_sha=array_sha(values[m]),
            valid_sha=array_sha(masks[m]), support_sha=array_sha(support[m]),
            supported_corner_histogram={str(k): int((s == k).sum()) for k in range(9)})
    path = RAW / 'TRAIN_APPEARANCE.npz'
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)
    # No fitting or candidate selection is performed by this process.
    result = dict(complete=True, input_construction_pass=True, source_TRAIN_only=True,
        protocol=bind(DOC / 'INPUT_PROTOCOL.json'), tokens=bind(token_path), descriptors=bind(path),
        frames=2598, image_forwards=forwards, original_allinvalid_rows=1, fits_executed=0,
        torch_version=torch.__version__, numpy_version=np.__version__, opencv_version=cv2.__version__,
        cuda_device=torch.cuda.get_device_name(0),
        precision=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                       cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                       cudnn_benchmark=torch.backends.cudnn.benchmark),
        source_label_values_read=False, source_VAL_features_extracted=False, real_features_extracted=False,
        new_PnP_solves=0, original_candidate_validity_preserved=True,
        normalization=dict(source='R0_original_valid_TRAIN_candidates', count=5194, dtype='float32',
            std_floor=1e-6, mean385=mean.tolist(), std385=std.tolist(), array_sha=array_sha(np.stack([mean, std]))),
        models=summaries, frame_receipts=frame_receipts, read_paths=sorted(READS), write_paths=sorted(WRITES),
        elapsed_seconds=time.monotonic() - start, stable_joint_improvement_achieved=False)
    assert forwards == 2597
    # The receipt is outside RAW and must be explicitly permitted by the hook.
    save(RAW / 'TRAIN_APPEARANCE_INPUTS.json', result)
    print('DINO_TRAIN_INPUT_COMPLETE', dict(frames=2598, image_forwards=forwards, fits=0), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['seal', 'freeze'])
    args = parser.parse_args()
    seal() if args.action == 'seal' else freeze()
