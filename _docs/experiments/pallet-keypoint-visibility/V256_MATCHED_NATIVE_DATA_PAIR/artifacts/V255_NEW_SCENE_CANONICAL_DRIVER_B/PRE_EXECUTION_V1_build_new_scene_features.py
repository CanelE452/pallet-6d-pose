"""Root-only new SOURCE FIT target compilation and frozen canonical extraction."""
import argparse
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback
import types

import numpy as np
import torch

POLICY_SHA = '0c0bc375c5e07cf7640bcb76bfd8efeda09e5a3d58f0c40c78094ba6dbc373a4'
GENERATOR_SHA = '090f541d80b725f52e2969c43096f6d0c6a7e7f89ab3028bb93e62cd24b150a5'
ADAPTER_SHA = '5377a1ee71e09aff9b95cb781967bbaf8a4061f2d667500528855ce79c996e88'
READER_SHA = '038bb84396a1db7932b70a8b1d91b7d7d020f4b05c0597e96082772b563db654'
POINT_SHA = 'd8925908d00f4f120bd80bae7a691a1a3846c4e9096e2e111d71984843e0b738'
LOADER_SHA = 'ae6e8ceb66ef29a52633ab98511e23a659d7f5b7e8268d300a53b8f4f12d71b1'
OFFICIAL_SHA = 'f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb'
OFFICIAL_STATE_SHA = 'bb4c7a80aa873bf3d34c3c1d44a28cee331503d7ccf52aba85f2d23edb65c3c3'
VENDOR_REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'


def guard(path, fresh=False):
    p = Path(path)
    if not p.is_absolute() or any(s == '..' or any(t in s.lower() for t in ('v112', 'cache', '.pyc')) for s in p.parts):
        raise ValueError('Forbidden artifact path')
    for a in reversed((p,) + tuple(p.parents)):
        if a.is_symlink(): raise ValueError('Symlink artifact/ancestor')
    if fresh and p.exists(): raise ValueError('Fresh output required; no automatic restart')
    return p


def identity(b):
    return (b['path'], b['sha256'], b.get('size_bytes'))


def read_bound(b):
    raw = guard(b['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != b['sha256'] or (b.get('size_bytes') is not None and len(raw) != b['size_bytes']):
        raise ValueError('Exact artifact binding')
    return raw


def binding(path):
    p = guard(path)
    h = hashlib.sha256(); size = 0
    with p.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data); size += len(data)
    return {'path': str(p), 'sha256': h.hexdigest(), 'size_bytes': size}


def load_module(b, expected, name):
    raw = read_bound(b)
    if b['sha256'] != expected: raise ValueError('Pinned implementation CODE')
    m = types.ModuleType(name); m.__file__ = b['path']; sys.modules[name] = m
    exec(compile(raw, b['path'], 'exec'), m.__dict__)
    return m


def load_teacher_helpers(adapter, code_bindings):
    reader = adapter.pure_code(code_bindings['reader'], READER_SHA, ('extract', 'locality'))
    # extract() uses the pinned reader's five-state vocabulary, not a new label.
    reader.extract.__globals__['STATES'] = ('D', 'E', 'S', 'U', 'UNKNOWN')
    point = adapter.pure_code(code_bindings['point_contract'], POINT_SHA, ('need', 'finite_pair', 'point_loss_target'))
    return reader, point


def stable(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def input_identity(scene, meta, parent, adapter):
    """New actual image identity first; old anchor identity is separate/inert."""
    reasons = []
    if scene.get('source_role') != 'FIT' or meta.get('source_role') != 'FIT' or parent.get('source_base_role') != 'FIT' or parent.get('identity_PASS') is not True:
        reasons.append('NOT_ADMITTED_SOURCE_FIT_PARENT')
    base = scene.get('base_index')
    if base != parent.get('base_index') or meta.get('base_index') != base or meta.get('base_id') != scene.get('base_id'):
        reasons.append('BASE_IDENTITY')
    for key in ('variant', 'target_native_slot', 'RGB_sha256_HWC'):
        if meta.get(key) != scene.get(key): reasons.append('NEW_SCENE_' + key + '_IDENTITY')
    if meta.get('image') != scene.get('image'):
        reasons.append('ACTUAL_NEW_IMAGE_BINDING')
    if scene.get('image', {}).get('format') != 'npz' or scene.get('image', {}).get('member') != 'RGB_uint8_HWC':
        reasons.append('EXACT_NEW_RGB_MEMBER')
    variant, intervention = scene.get('variant'), scene.get('target_native_slot')
    if variant == 'BEFORE':
        expected_id = str(scene.get('base_id')) + '_BEFORE'
        if intervention is not None: reasons.append('BEFORE_INTERVENTION_INDEX')
    else:
        expected_id = str(scene.get('base_id')) + '_NATIVE%s_%s' % (intervention, variant)
        if variant not in ('COVER', 'NEAR_SHAM', 'FAR_SHAM') or type(intervention) is not int or not 0 <= intervention < 8:
            reasons.append('INTERVENTION_INDEX_OR_VARIANT')
    if scene.get('scene_id') != expected_id: reasons.append('NEW_SCENE_ID_NAMESPACE')
    qmeta = meta.get('original_before_input', {})
    original = parent.get('before_input', {})
    for key in ('q8', 'L8', 'original_available8', 'box_xyxy', 'raw_hw', 'render_raw_hw_declared_separately'):
        try:
            equal = adapter_fp(qmeta.get(key)) == adapter_fp(original.get(key))
        except (TypeError, ValueError): equal = False
        if not equal: reasons.append('ORIGINAL_BEFORE_' + key + '_CHANGED')
    for key in ('group_id', 'group_index', 'image'):
        if qmeta.get(key) != original.get(key): reasons.append('ORIGINAL_BEFORE_' + key + '_CHANGED')
    perm = meta.get('permCF_to_canonical')
    if not isinstance(perm, list) or any(type(v) is not int for v in perm) or sorted(perm) != list(range(8)) or perm != original.get('originalpermCF_to_canonical'):
        reasons.append('ORIGINAL_NATIVE_PERMUTATION')
    shape = meta.get('RGB_shape')
    if meta.get('RGB_dtype') != 'uint8' or not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3 or shape[:2] != qmeta.get('render_raw_hw_declared_separately'):
        reasons.append('DECLARED_NEW_RGB_SHAPE')
    if variant == 'BEFORE' and scene.get('RGB_sha256_HWC') != original.get('expected_BEFORE_decoded_RGB_sha256'):
        reasons.append('BEFORE_RGB_PARITY')
    return sorted(set(reasons))


def adapter_fp(v):
    """Nullable decoded values, exact little-endian numeric bytes."""
    import struct
    if v is None: return b'N'
    if isinstance(v, list): return b'L' + struct.pack('<I', len(v)) + b''.join(adapter_fp(x) for x in v)
    if isinstance(v, bool): return b'B' + bytes([v])
    if type(v) not in (int, float): raise ValueError('Typed input value')
    return b'F' + struct.pack('<d', float(v))


def compile_scene_targets(scene, meta, parent, adapter, reader, point):
    """Teacher-only compilation. No image pixels or model feature API here."""
    global_reasons = input_identity(scene, meta, parent, adapter)
    qmeta = meta.get('original_before_input', {})
    fresh = meta.get('fresh_records_CF')
    perm = meta.get('permCF_to_canonical')
    shape = meta.get('RGB_shape', [0, 0, 0])
    image_input = {k: scene['image'][k] for k in ('path', 'sha256', 'format', 'member')}
    group_id = hashlib.sha256(json.dumps([image_input, qmeta.get('q8'), qmeta.get('L8'),
        qmeta.get('original_available8')], ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    rows = []; target = np.full(8, -1, np.int8)
    for j in range(8):
        reasons = list(global_reasons)
        record = fresh[j] if isinstance(fresh, list) and len(fresh) == 8 else None
        identity_ok = isinstance(record, dict) and record.get('CF_index') == j and isinstance(perm, list) and len(perm) == 8 and record.get('canonical_index') == perm[j] and record.get('native_canonical_index') == perm[j]
        if not identity_ok: reasons.append('MISSING_OR_WRONG_OWN_SCENE_CF_CANONICAL_RECORD')
        evidence = reader.extract(record if isinstance(record, dict) else None)
        for field, state_key in (('point0_target_alone_states8', 'point0_target_alone_state'), ('point0_combined_states8', 'point0_combined_state')):
            values = meta.get(field)
            if not isinstance(values, list) or len(values) != 8 or values[j] != evidence[state_key]:
                reasons.append('FRESH_POINT0_AGGREGATE_RECORD_IDENTITY')
        try:
            q, L, av = qmeta['q8'][j], qmeta['L8'][j], qmeta['original_available8'][j]
        except (KeyError, TypeError, IndexError):
            q, L, av = None, None, False; reasons.append('MISSING_ORIGINAL_INPUT')
        locality = reader.locality(q, L, av, evidence['saved_projected_reference_xy'], shape[:2])
        identity_reasons = list(reasons)
        if identity_reasons:
            y, old_reasons = -1, []
        else:
            compatibility = dict(evidence, component_role='TRAIN', mapping_status='EXACT_ORIGINAL_IMAGE_OWN_SCENE',
                family='BEFORE_ANCHOR4', state_name='NOMINAL', state_index=0,
                original_query_available=av, query_xy=q, L=L,
                reference_locality_for_actual_input=locality)
            y, old_reasons = point.point_loss_target(compatibility)
        reasons.extend(old_reasons)
        reasons = sorted(set(reasons))
        if reasons: y = -1
        target[j] = y
        rows.append({'row_id': 'V255:%s:NATIVE%d' % (scene['scene_id'], j),
            'scene_id': scene['scene_id'], 'new_scene_index': scene['scene_index'],
            'group_id': group_id, 'native_slot': j, 'logical_aux_slot': None,
            'base_index': scene['base_index'], 'generation_family': 'V255_NEW_OWN_SCENE_ORIGINAL_BEFORE_INPUT',
            'active_native_slot': j, 'intervention_target_native_slot': scene['target_native_slot'],
            'variant': scene['variant'], 'source_role': 'FIT',
            'new_teacher_key': {'scene_id': scene['scene_id'], 'RGB_sha256_HWC': scene['RGB_sha256_HWC'], 'active_CF_index': j},
            'own_image_record_identity_PASS': not identity_reasons,
            'own_image_mapping_reasons': identity_reasons,
            'declared_canonical_index_teacher_only': None if not isinstance(record, dict) else record.get('canonical_index'),
            'original_query_xy': q, 'original_L': L, 'original_query_available': av,
            'q': q, 'L': L, 'original_state': evidence['point0_combined_state'],
            'original_component_role': 'TRAIN', 'internal_source_role': 'INTERNAL_SOURCE_FIT',
            'point0_target_alone_state': evidence['point0_target_alone_state'],
            'point0_combined_state': evidence['point0_combined_state'],
            'projected_reference_xy_teacher_only': evidence['saved_projected_reference_xy'],
            'reference_locality_teacher_only': locality,
            'compatibility_namespace_only': {'applied_after_own_image_proof': not identity_reasons,
                'family': None if identity_reasons else 'BEFORE_ANCHOR4',
                'state_name': None if identity_reasons else 'NOMINAL', 'state_index': None if identity_reasons else 0,
                'not_historical_source_row_or_4base_variant_key': True},
            'point_loss_target_before_local_input_support': int(y), 'target_reasons': reasons,
            'no_q_replacement_or_variant_label_inference': True})
    return target, rows, global_reasons


def jwrite(path, value):
    p = guard(path)
    if p.exists(): raise ValueError('Immutable output file')
    p.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return binding(p)


def make_samples_batched(inputs, target8, compiled_rows, adapter, kernels):
    """Same metadata math as adapter, bounded32 preprocessing instead of all-N."""
    geometry = inputs['geometry']; group = []; slot = []; cond = []; ids = []; targets = []
    for gi in range(len(target8)):
        for j in range(8):
            r = compiled_rows[8 * gi + j]
            available = bool(geometry['original_available8'][gi, j] and geometry['node_available'][gi, j])
            target = int(target8[gi, j]) if available else -1
            r.update(group_index=gi, active_slot=j, family_code=0, fit=True,
                     crop_available=available, target=target, sample_index=None, loss_mask=False)
            if not available: r['target_reasons'] = sorted(set(r['target_reasons'] + ['RAW96_INPUT_SUPPORT_UNAVAILABLE']))
            if target not in (0, 1): continue
            r['sample_index'] = len(group); r['loss_mask'] = True
            group.append(gi); slot.append(j); ids.append(r['row_id']); targets.append(target)
            cond.append(kernels.condition.context32(geometry['q8'][gi, j].tolist(), float(geometry['L8'][gi, j]), j,
                bool(geometry['original_available8'][gi, j]), geometry['q8'][gi].tolist(),
                geometry['L8'][gi].tolist(), geometry['original_available8'][gi].tolist()))
    group = np.asarray(group, np.int64); slot = np.asarray(slot, np.int64); n = len(group)
    if not n: raise ValueError('No admitted supported D/E; no fabricated fallback')
    fractions = np.zeros((n, 2), np.float32); central = np.zeros(n, bool)
    for start in range(0, n, 32):
        gi, j = group[start:start + 32], slot[start:start + 32]
        rgb = torch.from_numpy(np.array(inputs['node_rgb'][gi, j], copy=True))
        valid = torch.from_numpy(np.array(inputs['node_valid'][gi, j, None], copy=True))
        views, f = kernels.preprocess.preprocess_views(rgb, valid)
        fractions[start:start + len(gi)] = f.numpy()
        central[start:start + len(gi)] = valid[:, :, 47:49, 47:49].reshape(len(gi), -1).all(1).numpy()
        del views, rgb, valid
    data = {'full_view': 2 * np.arange(n, dtype=np.int64), 'central_view': 2 * np.arange(n, dtype=np.int64) + 1,
        'fractions': fractions, 'central4': central, 'condition': np.stack(cond).astype(np.float32, copy=False),
        'target': np.asarray(targets, np.int8), 'family': np.zeros(n, np.int8),
        'fit': np.ones(n, bool), 'available': np.ones(n, bool), 'row_id': np.asarray(ids)}
    return data, group, slot


def authorize(entry_path):
    e = json.loads(guard(entry_path).read_bytes())
    if e.get('execution_authorized') is not True:
        raise PermissionError('Root true entry required before any new data/weight access')
    assert read_bound(e['code']) == guard(__file__).read_bytes()
    assert Path(e['code']['path']) == Path(__file__)
    pb = read_bound(e['root_policy']); assert hashlib.sha256(pb).hexdigest() == POLICY_SHA
    policy = json.loads(pb)
    scope = json.loads(read_bound(e['root_scope']))
    assert scope.get('approved') is True and scope['entry_contract'] == {k: v for k, v in e.items() if k != 'root_scope'}
    rt = policy['runtime']
    assert sys.executable == rt['executable']
    assert platform.python_version() == rt['python'] and np.__version__ == rt['numpy'] and torch.__version__ == rt['torch']
    assert rt['device'] == 'cuda:0' and rt['batch_size'] == 32
    complete = json.loads(read_bound(e['stage2_complete']))
    qa = json.loads(read_bound(e['stage2_QA']))
    assert complete.get('completed') is True and complete.get('PASS') is True and complete['code']['sha256'] == GENERATOR_SHA
    assert qa.get('completed') is True and qa.get('PASS') is True
    assert identity(complete['before_input_manifest']) == identity(e['stage1_parent_manifest'])
    assert guard(e['output_dir'], fresh=True).is_relative_to(guard(e['root_policy']['path']).parent)
    return e, policy, complete


def run(entry_path):
    e, policy, upstream = authorize(entry_path)
    cb = e['code_bindings']
    adapter = load_module(cb['adapter'], ADAPTER_SHA, 'v255_frozen_canonical_adapter')
    kernels = adapter.load_kernels({k: cb[k] for k in adapter.PINS})
    reader, point = load_teacher_helpers(adapter, cb)
    parents_doc = json.loads(read_bound(e['stage1_parent_manifest']))
    assert parents_doc['all_parent_identity_PASS'] is True
    parents = {p['base_index']: p for p in parents_doc['parents']}
    assert len(parents) == len(parents_doc['parents']) == upstream['counts']['selected_original_FIT_bases']
    scenes = [json.loads(line) for line in read_bound(upstream['outputs']['SCENES.jsonl']).splitlines() if line]
    manifest = json.loads(read_bound(upstream['outputs']['MANIFEST.json']))
    white = {identity(b) for b in manifest['files']}
    G = len(scenes)
    assert G == upstream['counts']['scenes'] and [r['scene_index'] for r in scenes] == list(range(G))
    assert len({r['scene_id'] for r in scenes}) == G
    assert {r['base_index'] for r in scenes} == set(parents)
    assert collections.Counter(r['base_index'] for r in scenes if r['variant'] == 'BEFORE') == {b: 1 for b in parents}
    stage2_root = guard(e['stage2_complete']['path']).parent
    for r in scenes:
        for key in ('image', 'metadata'):
            p = guard(r[key]['path'])
            assert p.is_relative_to(stage2_root / 'scenes') and identity(r[key]) in white
    out = guard(e['output_dir'], fresh=True); out.mkdir(parents=True)
    jwrite(out / 'STARTED.json', {'entry': binding(entry_path), 'automatic_retry': False, 'new_training': False})
    try:
        arrays = {
            'node_rgb': np.lib.format.open_memmap(out / 'NODE_RGB96.npy', mode='w+', dtype=np.uint8, shape=(G, 8, 3, 96, 96)),
            'node_valid': np.lib.format.open_memmap(out / 'NODE_VALID96.npy', mode='w+', dtype=bool, shape=(G, 8, 96, 96)),
            'context_rgb': np.lib.format.open_memmap(out / 'CONTEXT_RGB224_F32.npy', mode='w+', dtype=np.float32, shape=(G, 3, 224, 224)),
            'context_valid': np.lib.format.open_memmap(out / 'CONTEXT_VALID224.npy', mode='w+', dtype=bool, shape=(G, 224, 224)),
        }
        geom_parts = {k: [] for k in adapter.GEOMETRY_KEYS}; targets = np.full((G, 8), -1, np.int8)
        full_rows = []; input_records = []; record_identity_fail = 0
        for gi, scene in enumerate(scenes):
            meta = json.loads(read_bound(scene['metadata']))
            target, rows, failures = compile_scene_targets(scene, meta, parents[scene['base_index']], adapter, reader, point)
            if failures: raise ValueError('New-scene/parent input identity: ' + ','.join(failures))
            record_identity_fail += sum(not r['own_image_record_identity_PASS'] for r in rows)
            image_raw = read_bound(scene['image'])
            with np.load(io.BytesIO(image_raw), allow_pickle=False) as z:
                rgb = z['RGB_uint8_HWC']  # Single admitted member only; no .files/discovery.
            assert rgb.dtype == np.uint8 and rgb.ndim == 3 and list(rgb.shape) == meta['RGB_shape']
            assert hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest() == scene['RGB_sha256_HWC']
            before = meta['original_before_input']
            sampled = adapter.sample_scene(rgb, before['q8'], before['L8'], before['original_available8'], before['box_xyxy'], kernels)
            for key in arrays: arrays[key][gi] = sampled[key]
            for key in geom_parts: geom_parts[key].append(sampled['geometry'][key])
            targets[gi] = target; full_rows.extend(rows)
            input_records.append({'group_index': gi, 'scene_id': scene['scene_id'], 'image': scene['image'],
                'metadata': scene['metadata'], 'RGB_sha256_HWC': scene['RGB_sha256_HWC'],
                'original_parent_base_index': scene['base_index'], 'target_native_slot_teacher_only': scene['target_native_slot']})
            if (gi + 1) % 100 == 0: print(json.dumps({'stage': 'new_source_crop_and_targets', 'scenes': gi + 1, 'total': G}), flush=True)
        for value in arrays.values(): value.flush()
        geometry = {k: np.stack(v) for k, v in geom_parts.items()}
        inputs = dict(arrays, geometry=geometry)
        data, group, slot = make_samples_batched(inputs, targets, full_rows, adapter, kernels)
        rows_path = out / 'ALL_NEW_SOURCE_ROWS.jsonl'
        with rows_path.open('x') as f:
            for row in full_rows: f.write(stable(row) + '\n')
        compiler_qa = {'completed': True, 'PASS': record_identity_fail == 0, 'scenes': G, 'all_native_rows': len(full_rows),
            'known_supported_rows': len(group), 'masked_rows': len(full_rows) - len(group),
            'record_identity_failures': record_identity_fail, 'all_scene_parent_input_identity': True,
            'point0_states': dict(sorted(collections.Counter(r['point0_combined_state'] for r in full_rows).items())),
            'loss_targets': dict(sorted(collections.Counter(r['target'] for r in full_rows).items())),
            'reasons': dict(sorted(collections.Counter(x for r in full_rows for x in r['target_reasons']).items())),
            'original_U_rows_preserved': sum(not r['original_query_available'] for r in full_rows),
            'new_scene_namespace_not_old_integer_alias': True, 'variant_used_to_assign_label': False,
            'new_native_or_real_gold_created': False, 'teacher_r_or_ID_forwarded': False}
        cq = jwrite(out / 'TARGET_COMPILER_QA.json', compiler_qa)
        if record_identity_fail: raise ValueError('Fresh own-scene CF record identity failed; no encoder forward')
        np.savez(out / 'samples.npz', **data); np.savez(out / 'INPUT_GEOMETRY.npz', **geometry)
        jwrite(out / 'INPUT_SCENE_BINDINGS.json', {'completed': True, 'rows': input_records})
        cfg = e['model_loader_config']
        assert cfg['official_state']['sha256'] == cfg['asset_identity']['sha256'] == OFFICIAL_SHA
        assert cfg['asset_identity']['vendor_revision'] == VENDOR_REVISION
        assert {identity(b)[:2] for b in cfg['code_whitelist']} <= {identity(b)[:2] for b in e['code_whitelist']}
        assert cfg.get('approved_official_backbone_load') is True
        loader = load_module(cb['model_loader'], LOADER_SHA, 'v255_official_frozen_loader')
        loader.configure_runtime()
        encoder = loader.load_backbone(cfg, policy['runtime']['device'], treatment=False)
        encoder.eval().requires_grad_(False)
        packet = adapter.extract_packet(inputs, data, group, slot, encoder, kernels, policy['runtime']['device'], OFFICIAL_STATE_SHA)
        qa = packet['canonical_QA']
        assert qa['active_first384_byte_exact'] is True and qa['condition32_byte_exact'] is True
        assert qa['frozen_encoder_state_sha256_before'] == qa['frozen_encoder_state_sha256_after'] == OFFICIAL_STATE_SHA
        assert qa['known'] == len(data['target']) and qa['groups'] == G
        # The pinned extract_canonical refuses completion unless its repeated
        # first supported-node batch was byte-exact; no new tolerance is added.
        qa['first_node_batch_repeat_byte_exact'] = True
        features = {}
        for name, value in packet['features'].items():
            np.save(out / name, value, allow_pickle=False); features[name] = binding(out / name)
        canonical_qa = jwrite(out / 'CANONICAL_BYTE_SHARING_AND_FROZEN_ENCODER_QA.json', packet['canonical_QA'])
        census = {'completed': True, 'selected_original_FIT_bases': len(parents), 'new_scenes': G,
            'all_new_native_rows': 8 * G, 'known_rows': len(group), 'masked_rows': 8 * G - len(group),
            'supported_input_nodes_including_teacher_S_U': int(geometry['node_available'].sum()),
            'new_source_family': 0, 'all_fit': bool(data['fit'].all()), 'old_source_feature_payload_reads': 0,
            'classifier_head_optimizer_training_updates': 0, 'current_held_private_reads': 0}
        census_b = jwrite(out / 'CENSUS.json', census)
        names = ['NODE_RGB96.npy', 'NODE_VALID96.npy', 'CONTEXT_RGB224_F32.npy', 'CONTEXT_VALID224.npy',
                 'INPUT_GEOMETRY.npz', 'samples.npz', 'ALL_NEW_SOURCE_ROWS.jsonl', 'INPUT_SCENE_BINDINGS.json',
                 'TARGET_COMPILER_QA.json', 'CANONICAL_BYTE_SHARING_AND_FROZEN_ENCODER_QA.json', 'CENSUS.json', *features]
        outputs = {name: binding(out / name) for name in names}
        manifest_b = jwrite(out / 'MANIFEST.json', {'files': list(outputs.values()), 'treewalk': False})
        complete = {'completed': True, 'PASS': True, 'code': e['code'], 'entry': binding(entry_path),
            'root_policy': e['root_policy'], 'root_scope': e['root_scope'], 'implementation_protocol': e['implementation_protocol'],
            'stage2_complete': e['stage2_complete'], 'stage2_QA': e['stage2_QA'], 'stage1_parent_manifest': e['stage1_parent_manifest'],
            'code_bindings': cb, 'features': features, 'samples_npz': outputs['samples.npz'],
            'geometry_npz': outputs['INPUT_GEOMETRY.npz'], 'sample_map': outputs['ALL_NEW_SOURCE_ROWS.jsonl'],
            'outputs': outputs, 'manifest': manifest_b, 'counts': census, 'target_compiler_QA': cq,
            'canonical_sharing_QA_binding': canonical_qa, 'canonical_sharing_QA': packet['canonical_QA'],
            'official_base': cfg['official_state'], 'asset_identity': cfg['asset_identity'],
            'base_state_sha256_before': packet['canonical_QA']['frozen_encoder_state_sha256_before'],
            'base_state_sha256_after': packet['canonical_QA']['frozen_encoder_state_sha256_after'],
            'runtime': policy['runtime'], 'old_source_payload_reads': 0, 'classifier_optimizer_training': 0,
            'teacher_target_admission':'Root-prospective registered own-scene point0 only; not literalq/wholepart/real-native gold',
            'new_model_or_threshold_selection': False, 'automatic_retry': False}
        jwrite(out / 'COMPLETE.json', complete)
        print(json.dumps({'completed': True, 'PASS': True, 'COMPLETE': binding(out / 'COMPLETE.json')}), flush=True)
        return complete
    except BaseException:
        jwrite(out / 'FAILED_EXECUTION.json', {'completed': False, 'PASS': False, 'automatic_retry': False,
            'partial_outputs_preserved': True, 'new_classifier_training': False, 'traceback': traceback.format_exc()})
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--entry', required=True)
    run(ap.parse_args().entry)


if __name__ == '__main__': main()
