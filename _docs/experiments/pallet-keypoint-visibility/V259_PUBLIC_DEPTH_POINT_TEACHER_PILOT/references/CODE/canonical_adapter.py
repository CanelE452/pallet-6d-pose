"""In-memory V255 crop/canonical adapter. No scene driver, model load or training."""
import ast
import hashlib
import json
import math
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn.functional as F

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'publication_root_preparation'
PINS = {
    'local_crop': (P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py', '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'),
    'context_crop': (P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py', 'aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3'),
    'condition': (R/'V219_TRAINING_PREPARATION_C/source_training_contract.py', 'fc0a161e8a4c1a413d528ce1e66eed1b2addf2d4593c632506280f10732451d7'),
    'preprocess': (P/'query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py', 'cc1e2c331d41aaea5b3d42d144c9abbc40231dd3784d24f52dcc58f41d607578'),
    'context_preprocess': (P/'query_visibility_target_relative_rgb_head_20261023_v108/context_feature_preparation/prepare_target_context_features.py', '8deae91931c12bcc1692c98ce74fa5e703cfb0d57da980b126d5ca45ca74873e'),
    'crop_features': (R/'V233_DETERMINISTIC_CROP_FIX_A/crop_modules.py', '44f54dfcbf1671a401e748e66ec30463096e6c24b0779f07dda85d7c9f38ae88'),
    'canonical': (R/'V239_CANONICAL_FEATURE_PRODUCER_B/build_canonical_source_features.py', '29a890a308f232743f65bc525e5aaee346b8702ae14c5db8a60b15d5bd367ff4'),
}
GEOMETRY_KEYS = ('q8', 'L8', 'original_available8', 'node_available', 'node_fraction', 'context_fraction', 'box_xyxy')
SAMPLE_KEYS = ('full_view', 'central_view', 'fractions', 'central4', 'condition', 'target', 'family', 'fit', 'available', 'row_id')
FEATURE_KEYS = ('ORIGINAL_ALL_FEATURES.npy', 'NODE_CLS.npy', 'CONTEXT_CLS.npy',
                'SAMPLE_GROUP_INDEX.npy', 'SAMPLE_NODE_INDEX.npy', 'RELATIVE_XY8.npy')


def guard(path):
    p = Path(path)
    if not p.is_absolute() or any(s == '..' or any(t in s.lower() for t in ('v112', 'cache', '.pyc')) for s in p.parts):
        raise ValueError('Forbidden CODE artifact path')
    for a in reversed((p,) + tuple(p.parents)):
        if a.is_symlink(): raise ValueError('Symlink CODE artifact/ancestor')
    return p


def pure_code(binding, expected_sha, names):
    """Load only exact pure function AST. Never run an inherited old driver."""
    p = guard(binding['path'])
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha or binding['sha256'] != expected_sha:
        raise ValueError('Pinned pure CODE SHA')
    if binding.get('size_bytes') is not None and len(raw) != binding['size_bytes']:
        raise ValueError('Pinned pure CODE bytes')
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names): raise ValueError('Exact pure definitions')
    ns = {'np': np, 'math': math, 'torch': torch, 'F': F, 'json': json, 'hashlib': hashlib,
          'MEAN': (.485, .456, .406), 'STD': (.229, .224, .225)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), ns)
    return types.SimpleNamespace(**{name: ns[name] for name in names})


def load_kernels(code_bindings):
    if set(code_bindings) != set(PINS): raise ValueError('Closed pure CODE bindings')
    wanted = {
        'local_crop': ('crop',),
        'context_crop': ('valid_box', 'target_context_grid', 'sample_target_context'),
        'condition': ('need', 'context32'),
        'preprocess': ('preprocess_views',),
        'context_preprocess': ('preprocess_target',),
        'crop_features': ('crop_features', 'fixed_center_q_read'),
        'canonical': ('state_sha', 'relative_xy', 'node_pixel_batch', 'extract_canonical', 'assemble_known'),
    }
    return types.SimpleNamespace(**{k: pure_code(code_bindings[k], PINS[k][1], wanted[k]) for k in PINS})


def sample_scene(rgb, q8, L8, original_available8, box_xyxy, kernels):
    """Input-only actual RGB. Teacher IDs/pose/r/owner/semantic masks are absent."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('Actual scene uint8 HWC RGB')
    q = np.asarray(q8, np.float64); L = np.asarray(L8, np.float64)
    av = np.asarray(original_available8)
    if q.shape != (8, 2) or L.shape != (8,) or av.dtype != bool or av.shape != (8,):
        raise ValueError('Original F64 q8/L8 and bool availability8')
    if not np.isfinite(q[av]).all() or not np.isfinite(L[av]).all() or not (L[av] > 0).all():
        raise ValueError('Original available q/L finite, no replacement')
    nodes = np.zeros((8, 3, 96, 96), np.uint8)
    valid = np.zeros((8, 96, 96), bool)
    for k in range(8):
        # Same V237 behavior: finite [-1,-1]/unavailable nodes still have raw crops.
        if np.isfinite(q[k]).all() and math.isfinite(float(L[k])) and L[k] > 0:
            nodes[k], vv = kernels.local_crop.crop(rgb, q[k], float(L[k]), 96)
            valid[k] = vv[0]
    central4 = valid[:, 47:49, 47:49].all((1, 2))
    supported = av & central4 & valid.any((1, 2))
    context = np.zeros((3, 224, 224), np.float32)
    cv = np.zeros((224, 224), bool)
    if kernels.context_crop.valid_box(box_xyxy):
        chw, cv = kernels.context_crop.sample_target_context(rgb, box_xyxy, (224, 224))
        context = np.ascontiguousarray(chw.transpose(2, 0, 1))
    box = np.full(4, np.nan, np.float64) if box_xyxy is None else np.asarray(box_xyxy, np.float64)
    if box.shape != (4,): raise ValueError('Original box shape; no invented box')
    geometry = {'q8': q.copy(), 'L8': L.copy(), 'original_available8': av.copy(),
                'node_available': supported, 'node_fraction': valid.mean((1, 2)).astype(np.float32),
                'context_fraction': np.float32(cv.mean()), 'box_xyxy': box.copy()}
    return {'node_rgb': nodes, 'node_valid': valid, 'context_rgb': context,
            'context_valid': cv, 'geometry': geometry}


def stack_scenes(sampled):
    if not sampled: raise ValueError('At least one declared scene')
    return {**{key: np.stack([s[key] for s in sampled]) for key in ('node_rgb', 'node_valid', 'context_rgb', 'context_valid')},
            'geometry': {key: np.stack([s['geometry'][key] for s in sampled]) for key in GEOMETRY_KEYS}}


def make_source_metadata(inputs, admitted_target8, row_id8, fit_groups, kernels):
    """Caller-supplied target admission, not a new teacher/variant-name label rule.

    The future Root contract must close original point_loss_target/locality or
    its prospective equivalent before supplying admitted_target8. Raw renderer
    states and reference coordinates stay separate; they are never forwarded.
    """
    g = inputs['geometry']; G = len(g['q8'])
    if admitted_target8.dtype != np.int8 or admitted_target8.shape != (G, 8) or not np.isin(admitted_target8, [-1, 0, 1]).all():
        raise ValueError('Separate Root-admitted int8 target8')
    if row_id8.shape != (G, 8) or row_id8.dtype.kind != 'U' or len(set(row_id8.ravel().tolist())) != G * 8:
        raise ValueError('Unique declared all8 row IDs')
    if fit_groups.dtype != bool or fit_groups.shape != (G,): raise ValueError('Original group roles')
    known_group, known_slot, conditions, records = [], [], [], []
    for gi in range(G):
        for k in range(8):
            available = bool(g['original_available8'][gi, k] and g['node_available'][gi, k])
            original_target = int(admitted_target8[gi, k])
            target = original_target if available else -1
            si = len(known_group) if target in (0, 1) else None
            if si is not None:
                known_group.append(gi); known_slot.append(k)
                conditions.append(kernels.condition.context32(g['q8'][gi, k].tolist(), float(g['L8'][gi, k]), k,
                    bool(g['original_available8'][gi, k]), g['q8'][gi].tolist(), g['L8'][gi].tolist(), g['original_available8'][gi].tolist()))
            records.append({'row_id': str(row_id8[gi, k]), 'full_row_index': 8 * gi + k,
                'group_index': gi, 'active_slot': k, 'native_slot': k, 'family_code': 0,
                'fit': bool(fit_groups[gi]), 'original_query_available': bool(g['original_available8'][gi, k]),
                'crop_available': available, 'admitted_teacher_target_before_input_mask': original_target,
                'target': target, 'sample_index': si, 'loss_mask': si is not None,
                'mask_reason': 'ORIGINAL_INPUT_OR_LOCAL_SUPPORT_U' if not available else ('TEACHER_S_U_OR_UNADMITTED' if target == -1 else None)})
    group = np.asarray(known_group, np.int64); slot = np.asarray(known_slot, np.int64)
    n = len(group)
    if n == 0: raise ValueError('No supported known rows; do not fabricate source classes')
    rgb = torch.from_numpy(np.array(inputs['node_rgb'][group, slot], copy=True))
    valid = torch.from_numpy(np.array(inputs['node_valid'][group, slot, None], copy=True))
    _, fractions = kernels.preprocess.preprocess_views(rgb, valid)
    central4 = valid[:, :, 47:49, 47:49].reshape(n, -1).all(1).numpy()
    ids = np.asarray([r['row_id'] for r in records if r['sample_index'] is not None])
    data = {'full_view': 2 * np.arange(n, dtype=np.int64), 'central_view': 2 * np.arange(n, dtype=np.int64) + 1,
            'fractions': fractions.numpy(), 'central4': central4,
            'condition': np.stack(conditions).astype(np.float32, copy=False),
            'target': admitted_target8[group, slot].copy(), 'family': np.zeros(n, np.int8),
            'fit': fit_groups[group].copy(), 'available': np.ones(n, bool), 'row_id': ids}
    return data, group, slot, records


def extract_packet(inputs, data, group, slot, encoder, kernels, device, expected_encoder_state_sha256):
    """Future Root-only encoder call. No model/weight loader or optimizer here."""
    if any(p.requires_grad for p in encoder.parameters()) or any(m.training for m in encoder.modules()):
        raise ValueError('Fully frozen eval encoder required')
    before = kernels.canonical.state_sha(encoder.state_dict())
    if before != expected_encoder_state_sha256:
        raise ValueError('Root-pinned official encoder state required before forward')
    pixel, context, node4, supported, contexts = kernels.canonical.extract_canonical(encoder,
        inputs['node_rgb'], inputs['node_valid'], inputs['context_rgb'], inputs['context_valid'], inputs['geometry'],
        kernels.preprocess.preprocess_views, kernels.context_preprocess.preprocess_target,
        kernels.crop_features, device, batch_size=32)
    base, nodes = kernels.canonical.assemble_known(pixel, data, group, slot, inputs['geometry'], node4)
    after = kernels.canonical.state_sha(encoder.state_dict())
    if before != after: raise ValueError('Frozen encoder changed')
    features = {'ORIGINAL_ALL_FEATURES.npy': base, 'NODE_CLS.npy': nodes, 'CONTEXT_CLS.npy': context,
                'SAMPLE_GROUP_INDEX.npy': group.copy(), 'SAMPLE_NODE_INDEX.npy': slot.copy(),
                'RELATIVE_XY8.npy': kernels.canonical.relative_xy(inputs['geometry'], group, slot)}
    proof = {'completed': True, 'active_first384_byte_exact': base[:, :384].tobytes() == nodes[group, slot].tobytes(),
             'condition32_byte_exact': base[:, -32:].tobytes() == data['condition'].tobytes(),
             'frozen_encoder_state_sha256_before': before, 'frozen_encoder_state_sha256_after': after,
             'groups': len(nodes), 'known': len(base), 'supported_nodes': supported, 'context_supported': contexts,
             'new_classifier_or_optimizer_updates': 0, 'teacher_fields_forwarded': False}
    return {'features': features, 'samples': data, 'geometry': inputs['geometry'], 'canonical_QA': proof}


def merge_fit_packets(packets):
    """Caller supplies already scoped FIT-only packets; bytecopy + explicit remap.

    In particular the function never opens an old whole-source feature file and
    never selects or decodes SOURCE DEV rows itself. IO remains in a future
    Root-bound driver after the upstream schema/counts/protocol are frozen.
    """
    if not packets: raise ValueError('Explicit FIT packets required')
    feature_parts = {key: [] for key in FEATURE_KEYS}; geometry_parts = {k: [] for k in GEOMETRY_KEYS}
    sample_parts = {key: [] for key in SAMPLE_KEYS}; provenance = []; N = G = 0
    for pi, packet in enumerate(packets):
        f, d, g = packet['features'], packet['samples'], packet['geometry']
        n, count = len(d['target']), len(g['q8'])
        if set(f) != set(FEATURE_KEYS) or set(d) != set(SAMPLE_KEYS) or set(g) != set(GEOMETRY_KEYS):
            raise ValueError('Exact canonical packet schemas')
        if not d['fit'].all(): raise ValueError('FIT-only merge; DEV input not admitted by this API')
        expected = {'ORIGINAL_ALL_FEATURES.npy': (n, 1186), 'NODE_CLS.npy': (count, 8, 384),
                    'CONTEXT_CLS.npy': (count, 384), 'SAMPLE_GROUP_INDEX.npy': (n,),
                    'SAMPLE_NODE_INDEX.npy': (n,), 'RELATIVE_XY8.npy': (n, 8, 2)}
        for key in FEATURE_KEYS:
            dtype = np.int64 if key in ('SAMPLE_GROUP_INDEX.npy', 'SAMPLE_NODE_INDEX.npy') else np.float32
            if f[key].shape != expected[key] or f[key].dtype != dtype: raise ValueError('Canonical feature shape/dtype')
        if f['ORIGINAL_ALL_FEATURES.npy'][:, -32:].tobytes() != d['condition'].tobytes():
            raise ValueError('Original condition32 bytes')
        groups, slots = f['SAMPLE_GROUP_INDEX.npy'], f['SAMPLE_NODE_INDEX.npy']
        if groups.dtype != np.int64 or slots.dtype != np.int64 or ((groups < 0) | (groups >= count)).any() or ((slots < 0) | (slots >= 8)).any():
            raise ValueError('Original local index bounds')
        if f['ORIGINAL_ALL_FEATURES.npy'][:, :384].tobytes() != f['NODE_CLS.npy'][groups, slots].tobytes():
            raise ValueError('Input packet canonical byte-sharing')
        for key in FEATURE_KEYS:
            value = np.array(f[key], copy=True)
            if key == 'SAMPLE_GROUP_INDEX.npy': value += G
            feature_parts[key].append(value)
        for key in SAMPLE_KEYS: sample_parts[key].append(np.array(d[key], copy=True))
        for key in GEOMETRY_KEYS: geometry_parts[key].append(np.array(g[key], copy=True))
        for i in range(n):
            provenance.append({'packet_index': pi, 'source_local_sample_index': i, 'assembled_sample_index': N + i,
                'source_local_group_index': int(groups[i]), 'assembled_group_index': G + int(groups[i]),
                'source_row_id': str(d['row_id'][i]), 'source_full_view': int(d['full_view'][i]),
                'source_central_view': int(d['central_view'][i])})
        N += n; G += count
    features = {key: np.concatenate(parts) for key, parts in feature_parts.items()}
    data = {key: np.concatenate(parts) for key, parts in sample_parts.items()}
    geometry = {key: np.concatenate(parts) for key, parts in geometry_parts.items()}
    data['full_view'] = 2 * np.arange(N, dtype=np.int64); data['central_view'] = data['full_view'] + 1
    if len(set(data['row_id'].tolist())) != N: raise ValueError('New/old row IDs must keep distinct namespaces')
    if features['ORIGINAL_ALL_FEATURES.npy'][:, :384].tobytes() != features['NODE_CLS.npy'][features['SAMPLE_GROUP_INDEX.npy'], features['SAMPLE_NODE_INDEX.npy']].tobytes():
        raise ValueError('Merged active/node sharing remains byteexact')
    return {'features': features, 'samples': data, 'geometry': geometry, 'index_provenance': provenance}


def numeric_route_differences(new, saved):
    """FIT BEFORE route test only. No tolerance, overwriting or performance gate."""
    if new.shape != saved.shape or new.dtype != saved.dtype: raise ValueError('Exact parity comparison schema')
    delta = np.asarray(new, np.float64) - np.asarray(saved, np.float64)
    if not np.isfinite(delta).all(): raise ValueError('Finite route diagnostics')
    return {'byte_exact': new.tobytes() == saved.tobytes(), 'shape': list(new.shape),
            'max_abs': float(np.max(np.abs(delta), initial=0)), 'squared_l2': float(np.sum(delta * delta)),
            'threshold_or_model_selection': False, 'overwritten': False}
