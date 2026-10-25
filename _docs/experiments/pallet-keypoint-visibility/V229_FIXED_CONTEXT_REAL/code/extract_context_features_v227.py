"""V227 동결 DINO 특징 추출. --entry 승인 전에는 payload/model 접근이 없다.

원 TRAIN151/q8/q/L/availability와 기존 weak368·후보 FIT/DEV를 그대로 보존한다.
GT/ROI/새 query/학습/추론 평가를 추가하지 않는다. SOURCE feature-only runner다.
"""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import time
import traceback

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'publication_root_preparation'
PROTOCOL = R / 'V227_ROOT_CONTEXT_PROBE/PROTOCOL_BEFORE_CONTEXT_FEATURES_OR_PROBE_FIT.json'
PROTOCOL_SHA = '8f92b3da3d6e66df4e1bac2c7121ef975d2eb5f6b2c19ebf9215977881399462'
MANIFEST = R / 'V227_ROOT_CONTEXT_PROBE/ORIGINAL_TRAIN151_CONTEXT_INPUT_BINDINGS.jsonl'
MANIFEST_SHA = '2f3d7834cabfa833cfcca95a1f01326e6000b5461d02157f00e413fb96e35298'
WEIGHT = P / 'query_visibility_spatial_target_query_features_20261023_v111/assets/dinov2_vits14_reg4_pretrain.pth'
WEIGHT_SHA = 'f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb'
VENDOR = Path('/home/minjae/Documents/github/pallet-pose/outputs/pallet_type_selftrain_v1/selftrain_recovery_v1/dino_localization_assets/dinov2')
VENDOR_REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
SOURCES = {
    'local_crop': (P / 'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py', '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b', ['crop']),
    'local_preprocess': (P / 'query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py', 'cc1e2c331d41aaea5b3d42d144c9abbc40231dd3784d24f52dcc58f41d607578', ['preprocess_views']),
    'context_math': (P / 'query_visibility_target_context_rgb_20261023_v107/target_context_math.py', 'aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3', ['query_extent', 'valid_box', 'target_context_grid', 'sample_target_context']),
    'context_preprocess': (P / 'query_visibility_target_relative_rgb_head_20261023_v108/context_feature_preparation/prepare_target_context_features.py', '8deae91931c12bcc1692c98ce74fa5e703cfb0d57da980b126d5ca45ca74873e', ['preprocess_target']),
    'query_sample': (P / 'query_visibility_spatial_target_query_features_20261023_v111/feature_preparation/prepare_spatial_query_features.py', 'cbf2f1cad553673b727c39fbe5c9c4263dfbdac35dba0aeea086f956e74ff19e', ['sample_Q_raw', 'state_sha']),
}
VENDOR_PINS = {
    'LICENSE': '600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7',
    'MODEL_CARD.md': '70ca59606bee0a5fbb1baec80e7e29a93cd7cfbe26ca1910c52a852c4aab09d0',
    'hubconf.py': 'c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64',
    'dinov2/hub/backbones.py': '871fca671b12a9ff02e810654baf509e97ccf461bf8196ce5ddeefff2fd87d3e',
    'dinov2/hub/utils.py': '579613e3d7b82c2a387eddc215ad9f076b8740f9618229628395fe5755dbd131',
    'dinov2/models/vision_transformer.py': '7799a260f2d7d0fe197331d08502fb8c542f9b7424723650f6a39b64fa2639ea',
    'dinov2/layers/attention.py': '79c7be7a452b3aad96698ec38d5d5150b9f4d8ac084fa93324510dc9f624775d',
    'dinov2/layers/block.py': '60c0ac7dfa4474be313fabfa5a23d82faf6f0cecd4e720a88be35de9788cb636',
}
OUTPUT_NAMES = {'features': 'CONTEXT_FEATURES.npz', 'input_facts': 'INPUT_FEATURE_FACTS.json', 'complete': 'COMPLETE.json'}
ROLES = ('SOURCE_FIT_CANDIDATE', 'SOURCE_DEV_CANDIDATE')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_path(value):
    # V112는 어떤 filesystem 호출보다 먼저 문자열에서 차단한다.
    s = str(value)
    require('v112' not in s.lower(), 'Forbidden V112 path before filesystem')
    p = Path(s)
    require(p.is_absolute() and '..' not in p.parts, 'Absolute non-parent path required')
    return p


def parent_guard(p):
    p = safe_path(p)
    current = Path('/')
    for part in p.parts[1:-1]:
        current = current / part
        fd = os.open(str(current), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        os.close(fd)
    return p


def read_bytes(p):
    p = parent_guard(p)
    fd = os.open(str(p), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), 'Regular file required')
        with os.fdopen(fd, 'rb', closefd=False) as f:
            return f.read()
    finally:
        os.close(fd)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def binding(p):
    b = read_bytes(p)
    return {'path': str(safe_path(p)), 'sha256': digest(b), 'size_bytes': len(b)}


def checked(b, expected=None):
    require(isinstance(b, dict) and set(b) <= {'path', 'sha256', 'size_bytes'}, 'Binding schema')
    require({'path', 'sha256'} <= set(b), 'Binding requires path/SHA')
    p = safe_path(b['path'])
    if expected is not None:
        require(str(p) == str(expected[0]) and b['sha256'] == expected[1], 'Pinned binding identity')
    data = read_bytes(p)
    require(digest(data) == b['sha256'], 'File SHA mismatch: ' + str(p))
    if 'size_bytes' in b:
        require(len(data) == b['size_bytes'], 'File size mismatch')
    return data


def write_json(p, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    fd = os.open(str(parent_guard(p)), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        require(os.write(fd, data) == len(data), 'Output write incomplete')
    finally:
        os.close(fd)


def pure_operators(np, torch, F):
    # 기존 producer 모듈을 import/실행하지 않고 명시된 순수 함수만 재사용한다.
    ns = {'np': np, 'torch': torch, 'F': F, 'MEAN': (.485, .456, .406), 'STD': (.229, .224, .225), 'hashlib': hashlib}
    for path, sha, names in SOURCES.values():
        tree = ast.parse(checked({'path': str(path), 'sha256': sha}).decode())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        require({n.name for n in functions} == set(names), 'Exact pure functions required')
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path) + '::pure_only', 'exec'), ns)
    return ns


def validate_metadata(frames, np):
    require(len(frames) == 151, 'Exactly151 original TRAIN frames')
    seen_frames, seen_rows, eligible = set(), set(), []
    for ci, r in enumerate(frames):
        fi = r['original_frame_index']
        require(isinstance(fi, int) and fi not in seen_frames, 'Unique original frame index')
        seen_frames.add(fi)
        require(r['original_component_role'] == 'TRAIN' and r['candidate_role'] in ROLES, 'Original TRAIN/candidate role')
        require(r['training_admission'] is False and r['GT_pose_ROI_mask_or_reference_input'] is False, 'No input target sidecar')
        require(r['native_slots'] == list(range(8)) and len(r['queries']) == 8, 'Original native8 order')
        q8 = np.asarray(r['query_xy'], np.float64)
        av = np.asarray(r['original_query_available'], bool)
        L8 = np.asarray(r['L_by_native'], np.float64)
        require(q8.shape == (8, 2) and av.shape == L8.shape == (8,), 'Original q8/availability/L schema')
        for k, q in enumerate(r['queries']):
            require(q['row_id'] not in seen_rows, 'Unique original row identity')
            seen_rows.add(q['row_id'])
            require(q['native_slot'] == k and q['original_frame_index'] == fi and q['component_id'] == r['component_id'], 'Original query join')
            require(q['source_id'] == r['source_id'] and q['original_component_role'] == 'TRAIN', 'Source identity/role')
            require(q['proposed_role'] == r['candidate_role'] and q['training_admission'] is False, 'Immutable candidate role/admission')
            require(q['query_xy'] == r['query_xy'][k] and q['L'] == r['L_by_native'][k] and q['original_query_available'] == r['original_query_available'][k], 'No q/L/U change')
            if q['existing_available_D_E'] is True:
                require(q['original_query_available'] is True and np.isfinite(q8[k]).all() and not (q8[k] == [-1, -1]).all() and np.isfinite(L8[k]) and L8[k] > 0, 'Eligible original input finite')
                require((q['existing_role_state'], q['existing_role_target']) in [('D', 0), ('E', 1)], 'Preserve existing weak D/E')
                eligible.append((ci, q))
        # Decoder payloads are not touched here; whitelist identities are lexical only.
        safe_path(r['canonical_RGB_NPY']['path'])
        require(str(r['canonical_RGB_NPY']['path']).endswith('.npy'), 'Canonical NPY only')
    require(len(eligible) == 368, 'Exactly existing eligible weak368')
    counts = {(role, target): sum(q['proposed_role'] == role and q['existing_role_target'] == target for _, q in eligible) for role in ROLES for target in [0, 1]}
    require(counts == {(ROLES[0], 0): 57, (ROLES[0], 1): 8, (ROLES[1], 0): 267, (ROLES[1], 1): 36}, 'Fixed65/303 candidate split')
    fit_components = {r['component_id'] for r in frames if r['candidate_role'] == ROLES[0]}
    dev_components = {r['component_id'] for r in frames if r['candidate_role'] == ROLES[1]}
    require(not fit_components & dev_components, 'Candidate components disjoint')
    return eligible


def condition32(r, q, np):
    k = q['native_slot']
    q8 = np.asarray(r['query_xy'], np.float64)
    available = np.asarray(r['original_query_available'], bool).copy()
    available[k] = False
    require(np.isfinite(q8[available]).all() and not np.any((q8[available] == [-1, -1]).all(1)), 'Available other q8 must be finite')
    xy = np.zeros((8, 2), np.float64)
    xy[available] = np.clip((q8[available] - np.asarray(q['query_xy'], np.float64)) / np.float64(q['L']), -8, 8)
    native = np.eye(8, dtype=np.float32)[k]
    return np.concatenate((native, available.astype(np.float32), xy.astype(np.float32).reshape(16)))


def template():
    return {'approved_actual_context_feature_extraction': False, 'source_TRAIN_only': True, 'protected_access_allowed': False, 'fit_allowed': False, 'PnP_allowed': False, 'automatic_retry': False,
            'code': {'path': str(Path(__file__).absolute()), 'sha256': None}, 'protocol': {'path': str(PROTOCOL), 'sha256': PROTOCOL_SHA}, 'input_manifest': {'path': str(MANIFEST), 'sha256': MANIFEST_SHA},
            'runtime': {'executable': '/home/minjae/anaconda3/envs/pallet-yolo26/bin/python', 'torch': '2.1.1+cu118', 'numpy': '1.26.4', 'CUDA': '11.8', 'device': 'cuda'},
            'vendor': {'path': str(VENDOR), 'revision': VENDOR_REVISION, 'files': [{'path': str(VENDOR / p), 'sha256': h} for p, h in VENDOR_PINS.items()]}, 'weights': {'path': str(WEIGHT), 'sha256': WEIGHT_SHA},
            'output_dir': None, 'output_bindings': {k: None for k in OUTPUT_NAMES}}


def run(entry_path):
    entry_raw = read_bytes(entry_path)
    e = json.loads(entry_raw)
    require(e['approved_actual_context_feature_extraction'] is True and e['source_TRAIN_only'] is True, 'Separate true root entry required')
    require(all(e[k] is False for k in ['protected_access_allowed', 'fit_allowed', 'PnP_allowed', 'automatic_retry']), 'Source feature-only gate')
    require(e['code']['path'] == str(Path(__file__).absolute()), 'Exact extractor code')
    checked(e['code'])
    p = json.loads(checked(e['protocol'], (PROTOCOL, PROTOCOL_SHA)))
    require(p['phase'] == 'V227', 'Pinned root protocol')
    manifest_raw = checked(e['input_manifest'], (MANIFEST, MANIFEST_SHA))
    frames = [json.loads(line) for line in manifest_raw.splitlines() if line]
    require(e['runtime'] == template()['runtime'], 'Exact pinned runtime contract')
    require(e['vendor'] == template()['vendor'] and e['weights']['path'] == str(WEIGHT) and e['weights']['sha256'] == WEIGHT_SHA, 'Pinned vendor/weight metadata')
    out = safe_path(e['output_dir'])
    require(out.parent == PROTOCOL.parent and out.name not in {'', '.', '..'}, 'Fresh output must be direct child of root probe folder')
    require(e['output_bindings'] == {k: str(out / name) for k, name in OUTPUT_NAMES.items()}, 'Exact output path closure')
    parent_guard(out)
    require(not os.path.lexists(out), 'No overwrite/retry')
    os.environ['XFORMERS_DISABLED'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sys.dont_write_bytecode = True
    import numpy as np
    import torch
    from torch.nn import functional as F
    require(sys.executable == e['runtime']['executable'] and str(torch.__version__) == e['runtime']['torch'] and np.__version__ == e['runtime']['numpy'] and torch.version.cuda == e['runtime']['CUDA'], 'Actual runtime identity')
    require(torch.cuda.is_available(), 'Pinned CUDA runtime required')
    eligible = validate_metadata(frames, np)
    op = pure_operators(np, torch, F)
    # All root/code/manifest/runtime/vendor identities are settled before payload bytes.
    for b in e['vendor']['files']:
        checked(b)
    weights_raw = checked(e['weights'], (WEIGHT, WEIGHT_SHA))
    os.mkdir(out)
    started = time.monotonic()
    write_json(out / 'STARTED.json', {'entry': {'path': str(safe_path(entry_path)), 'sha256': digest(entry_raw)}, 'code': e['code'], 'protocol': e['protocol'], 'input_manifest': e['input_manifest'], 'no_automatic_retry': True})
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        require(not any(k == 'dinov2' or k.startswith('dinov2.') for k in sys.modules), 'No preloaded alternate vendor')
        sys.path.insert(0, str(VENDOR))
        from dinov2.hub.backbones import dinov2_vits14_reg
        model = dinov2_vits14_reg(pretrained=False)
        state = torch.load(io.BytesIO(weights_raw), map_location='cpu', weights_only=True)
        require(tuple(state['register_tokens'].shape) == (1, 4, 384), 'Pinned reg4 shape')
        model.load_state_dict(state, strict=True)
        del state, weights_raw
        model = model.cuda().float().eval().requires_grad_(False)
        require(sum(t.numel() for t in model.parameters()) == 22058112, 'Exact frozen encoder')
        before = op['state_sha'](model)
        norm = lambda t: F.normalize(t.float(), p=2, dim=-1, eps=1e-12)
        context_cls = np.zeros((151, 384), np.float32)
        tokens = np.zeros((151, 256, 384), np.float32)
        locals_by_id, fixed_by_id, normalized_by_id, facts, context_available = {}, {}, {}, [], []
        calls = 0
        first_repeat = False
        for ci, r in enumerate(frames):
            rgb_raw = checked(r['canonical_RGB_NPY'])
            rgb = np.load(io.BytesIO(rgb_raw), allow_pickle=False)
            require(rgb.dtype == np.uint8 and list(rgb.shape) == r['raw_hw'] + [3] and rgb.flags.c_contiguous, 'Exact canonical uint8 HWC')
            require(digest(rgb.tobytes()) == r['image']['decoded_RGB_sha256'], 'Decoded full RGB SHA')
            box = op['query_extent'](r['query_xy'])
            these = [q for j, q in eligible if j == ci]
            require(box is not None or not these, 'Eligible original query requires finite q8 context')
            if box is None:
                context_available.append(False)
                facts.append({'original_frame_index': r['original_frame_index'], 'context_index': ci, 'canonical_RGB_NPY': r['canonical_RGB_NPY'], 'context_available': False, 'eligible_queries': 0})
                continue
            target_rgb, target_valid = op['sample_target_context'](rgb, box)
            require(bool(target_valid.any()) or not these, 'Eligible context needs supported sample')
            local, valid = [], []
            for q in these:
                a, v = op['crop'](rgb, q['query_xy'], q['L'], 96)
                local.append(a)
                valid.append(v)
            tv = op['preprocess_target'](torch.as_tensor(target_rgb.transpose(2, 0, 1)[None], device='cuda'), torch.as_tensor(target_valid[None, None], device='cuda'))
            views = tv
            if these:
                lv, _ = op['preprocess_views'](torch.as_tensor(np.stack(local), device='cuda'), torch.as_tensor(np.stack(valid), device='cuda'))
                views = torch.cat((tv, lv[:len(these)]), 0)  # central32 branch intentionally not encoded
            with torch.inference_mode():
                z = model.forward_features(views)
                calls += 1
                if not first_repeat:
                    z2 = model.forward_features(views)
                    calls += 1
                    require(all(torch.equal(z[k], z2[k]) for k in ['x_norm_clstoken', 'x_norm_patchtokens']), 'First batch repeat exact')
                    first_repeat = True
                require(z['x_norm_clstoken'].shape == (len(views), 384) and z['x_norm_patchtokens'].shape == (len(views), 256, 384), 'CLS/token output shape')
                c = norm(z['x_norm_clstoken'][:1])
                t = norm(z['x_norm_patchtokens'][:1])
                context_cls[ci] = c[0].cpu().numpy() if target_valid.any() else 0
                tokens[ci] = t[0].cpu().numpy() if target_valid.any() else 0
                if these:
                    ll = norm(z['x_norm_clstoken'][1:])
                    fq, inside = op['sample_Q_raw'](t.expand(len(these), -1, -1), [q['query_xy'] for q in these], [box] * len(these), [bool(target_valid.any())] * len(these))
                    require(bool(inside.all()), 'Eligible original queries inside original q8 box')
                    fq = norm(fq)
                    b = np.asarray(box, np.float64)
                    for j, q in enumerate(these):
                        qn = (np.asarray(q['query_xy'], np.float64) - b[:2]) / (b[2:] - b[:2])
                        require(np.isfinite(qn).all() and (qn >= 0).all() and (qn <= 1).all(), 'Normalized active q')
                        locals_by_id[q['row_id']] = ll[j].cpu().numpy()
                        fixed_by_id[q['row_id']] = fq[j].cpu().numpy()
                        normalized_by_id[q['row_id']] = qn.astype(np.float32)
            context_available.append(bool(target_valid.any()))
            facts.append({'original_frame_index': r['original_frame_index'], 'context_index': ci, 'canonical_RGB_NPY': r['canonical_RGB_NPY'], 'decoded_RGB_sha256': r['image']['decoded_RGB_sha256'], 'context_box_xyxy': box, 'context_available': bool(target_valid.any()), 'context_valid_fraction': float(target_valid.mean()), 'eligible_queries': len(these), 'original_query_rows': [q['row_id'] for q in these], 'original96_RGB_SHA': [digest(a.tobytes()) for a in local], 'original96_valid_SHA': [digest(v.tobytes()) for v in valid]})
            if (ci + 1) % 25 == 0:
                print(json.dumps({'frames': ci + 1, 'total': 151, 'encoder_calls': calls}), flush=True)
        packet = {'contextCLS': context_cls, 'tokens': tokens,
                  'localCLS': np.stack([locals_by_id[q['row_id']] for _, q in eligible]),
                  'fixedQ': np.stack([fixed_by_id[q['row_id']] for _, q in eligible]),
                  'condition32': np.stack([condition32(frames[ci], q, np) for ci, q in eligible]),
                  'target': np.asarray([q['existing_role_target'] for _, q in eligible], np.int8),
                  'fit_bool': np.asarray([q['proposed_role'] == ROLES[0] for _, q in eligible], bool),
                  'row_id': np.asarray([q['row_id'] for _, q in eligible], dtype=str),
                  'native': np.asarray([q['native_slot'] for _, q in eligible], np.int8),
                  'normalized_active_q': np.stack([normalized_by_id[q['row_id']] for _, q in eligible]),
                  'frame_index': np.asarray([q['original_frame_index'] for _, q in eligible], np.int64),
                  'context_index': np.asarray([ci for ci, _ in eligible], np.int64),
                  'context_frame_index': np.asarray([r['original_frame_index'] for r in frames], np.int64)}
        require(all(np.isfinite(v).all() for v in packet.values() if v.dtype.kind == 'f'), 'All finite output features')
        require(packet['fit_bool'].sum() == 65 and np.array_equal(packet['frame_index'], packet['context_frame_index'][packet['context_index']]), 'Final identity/split mapping')
        after = op['state_sha'](model)
        require(before == after and all(not t.requires_grad for t in model.parameters()) and all(not m.training for m in model.modules()), 'Encoder unchanged/frozen')
        # No original JPEG/image pointer is followed. Canonical NPYs alone were decoded.
        for r in frames:
            checked(r['canonical_RGB_NPY'])
        checked(e['weights'], (WEIGHT, WEIGHT_SHA))
        checked(e['input_manifest'], (MANIFEST, MANIFEST_SHA))
        fp = out / OUTPUT_NAMES['features']
        fd = os.open(str(parent_guard(fp)), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **packet)
        write_json(out / OUTPUT_NAMES['input_facts'], {'completed': True, 'PASS': True, 'frames': facts, 'original_query_and_role_metadata_unchanged': True, 'input_manifest': e['input_manifest']})
        result = {'completed': True, 'PASS': True, 'code': e['code'], 'protocol': e['protocol'], 'entry': {'path': str(safe_path(entry_path)), 'sha256': digest(entry_raw)}, 'input_manifest': e['input_manifest'], 'runtime': e['runtime'], 'vendor': e['vendor'], 'weights': e['weights'], 'features': binding(fp), 'input_facts': binding(out / OUTPUT_NAMES['input_facts']), 'shapes': {k: list(v.shape) for k, v in packet.items()}, 'dtype': {k: str(v.dtype) for k, v in packet.items()}, 'frames': 151, 'queries': 368, 'FIT': 65, 'DEV': 303, 'context_available': sum(context_available), 'encoder_calls': calls, 'first_batch_repeat_exact': first_repeat, 'encoder_state_before': before, 'encoder_state_after': after, 'frozen_state_and_input_weight_hashes_unchanged': True, 'normalization': 'Per-descriptor/token unit L2 only; no learned or FIT/DEV statistics. fixedQ bilinear reads normalized tokens then unit L2.', 'authority': p['data_authority'], 'original_q_L_native_U_and_candidate_roles_unchanged': True, 'use_sam': False, 'GT_ROI_reference_input': False, 'new_fit_or_PnP_or_protected_calls': 0, 'no_selection': True, 'seconds': time.monotonic() - started}
        write_json(out / OUTPUT_NAMES['complete'], result)
        print(json.dumps({'completed': True, 'PASS': True, 'COMPLETE': binding(out / OUTPUT_NAMES['complete'])}), flush=True)
    except BaseException:
        write_json(out / 'FAILED_EXECUTION.json', {'completed': False, 'PASS': False, 'traceback': traceback.format_exc(), 'no_automatic_retry': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry')
    parser.add_argument('--entry-template', action='store_true')
    args = parser.parse_args()
    require(bool(args.entry) != args.entry_template, 'Use --entry or --entry-template')
    if args.entry_template:
        print(json.dumps(template(), ensure_ascii=False, indent=2))
    else:
        run(args.entry)
