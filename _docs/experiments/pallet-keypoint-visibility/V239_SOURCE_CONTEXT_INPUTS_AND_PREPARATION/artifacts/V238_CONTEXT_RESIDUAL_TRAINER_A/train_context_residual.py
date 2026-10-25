"""V238 SOURCE-only fixed-DINO context residual; explicit Root entry only.

Import is inert. CONTROL is byte reused, new treatment is exactly1000 updates.
Frozen DEV context features are permitted; new-head DEV output is post-freeze.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types
import numpy as np
import torch
from torch.nn import functional as F

R = Path(__file__).absolute().parent.parent
CONTROL = 'CONTROL_LOCAL_CROP'
TREAT = 'TREAT_CONTEXT_RESIDUAL'
PINS = {
    'utility': (str(R/'V235_DATA_ONLY_TRAINER_A/train_data_only_counterfactual.py'), '7a118b4f650e0521edc57bb23dfc0f55cdd5a137bde8b8e0133e5968fa99cacb'),
    'source_trainer': (str(R/'V233_DETERMINISTIC_CROP_FIX_A/train_crop_adaptation.py'), 'ae6e8ceb66ef29a52633ab98511e23a659d7f5b7e8268d300a53b8f4f12d71b1'),
    'crop_module': (str(R/'V233_DETERMINISTIC_CROP_FIX_A/crop_modules.py'), '44f54dfcbf1671a401e748e66ec30463096e6c24b0779f07dda85d7c9f38ae88'),
    'prefix_runtime': (str(R/'V232_PREFIX_TRAIN_CODE_A/prefix_runtime.py'), '5b29e75eaa59c5be6091714a7999ada5272c3090fd4681f80979221984018be4'),
    'relational_module': (str(R/'V237_RELATIONAL_QUERY_DESIGN_B/relational_query_head.py'), 'dfd26ab7ccfa7037f0c3dca548d59c8574aa8b78c50699f70f01e03bf48935a9'),
    'context_preprocess_CODE': (str(R.parent/'query_visibility_target_relative_rgb_head_20261023_v108/context_feature_preparation/prepare_target_context_features.py'), '8deae91931c12bcc1692c98ce74fa5e703cfb0d57da980b126d5ca45ca74873e'),
    'root_protocol': (str(R/'V238_ROOT_CONTEXT_RESIDUAL/PROTOCOL_BEFORE_NEW_FEATURES_OR_FIT.json'), '17b3b00b1c8b801dc536460f555d40ff8df37f6fe2355955a5e1ad66ecb29ba3'),
    'input_admission': (str(R/'V238_ROOT_CONTEXT_RESIDUAL/INPUT_ADMISSION_BEFORE_FEATURES_OR_FIT.json'), '4a48a43616f1a1bb3b29bfd6cc1f2b6a5a1976ed01a2b0119b26de0fef67b751'),
}
GEOMETRY_KEYS = ('node_available', 'node_fraction', 'context_fraction', 'q8', 'L8', 'original_available8', 'box_xyxy')


def safe(value):
    text = str(value)
    if any(x in text.casefold() for x in ('v112', '__pycache__', '.pyc', 'current223', 'publicheld', 'oldval', 'v177', 'v178', 'source_holdout', 'v182', 'private_evaluation')):
        raise ValueError('Forbidden path before filesystem access')
    if '..' in Path(text).parts:
        raise ValueError('Parent path component forbidden')
    path = Path(os.path.abspath(text))
    for part in reversed((path, *path.parents)):
        if part.is_symlink():
            raise ValueError('Symlink rejected before leaf access')
    return path


def read(path):
    with os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        return stream.read()


def identity(b):
    return str(safe(b['path'])), b['sha256']


def verify(b, whitelist, pin=None):
    if identity(b) not in {identity(x) for x in whitelist} or (pin is not None and identity(b) != pin):
        raise ValueError('Outside explicit path/SHA whitelist or pinned code')
    digest = hashlib.sha256(); size = 0
    with os.fdopen(os.open(safe(b['path']), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block); size += len(block)
    if digest.hexdigest() != b['sha256'] or ('size_bytes' in b and b['size_bytes'] != size):
        raise ValueError('Bound bytes changed')
    return safe(b['path'])


def binding(path):
    raw = read(path)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def write_json(path, obj):
    with os.fdopen(os.open(safe(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'w') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    return binding(path)


def load_code(b, whitelist, name, pin):
    path = verify(b, whitelist, pin)
    module = types.ModuleType(name); module.__file__ = str(path); sys.modules[name] = module
    exec(compile(read(path), str(path), 'exec'), module.__dict__)
    return module


def load_context_preprocess(e):
    path = verify(e['context_preprocess_CODE'], e['code_whitelist'], PINS['context_preprocess_CODE'])
    nodes = [n for n in ast.parse(read(path)).body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_target']
    if len(nodes) != 1:
        raise ValueError('Exactly pinned V108 preprocess_target AST only')
    namespace = {'torch': torch, 'MEAN': (.485, .456, .406), 'STD': (.229, .224, .225)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['preprocess_target']


def validate_geometry(g, count):
    if any(k not in g for k in GEOMETRY_KEYS):
        raise ValueError('Missing saved input geometry')
    for key in ('node_available', 'original_available8'):
        if g[key].dtype != bool or g[key].shape != (count, 8):
            raise ValueError('Exact original/support bool8')
    for key, shape in [('node_fraction', (count, 8)), ('context_fraction', (count,))]:
        if g[key].dtype != np.float32 or g[key].shape != shape or not np.isfinite(g[key]).all() or not ((g[key] >= 0) & (g[key] <= 1)).all():
            raise ValueError('FP32 fixed validity fractions')
    for key, shape in [('q8', (count, 8, 2)), ('L8', (count, 8)), ('box_xyxy', (count, 4))]:
        if g[key].dtype != np.float64 or g[key].shape != shape:
            raise ValueError('Original F64 geometry')
    if np.any(g['node_available'] & ~g['original_available8']):
        raise ValueError('Support cannot promote original input U')
    return g


def map_samples(rows, data, geometry):
    """Pure SOURCE metadata join; no inferred labels, nearest index or GT input."""
    n = len(data['target']); groups = len(geometry['q8'])
    group = np.full(n, -1, np.int64); slot = np.full(n, -1, np.int64)
    group_roles = {}; row_ids = set()
    for row in rows:
        if row['row_id'] in row_ids:
            raise ValueError('Duplicate full SOURCE row ID')
        row_ids.add(row['row_id'])
        gi, si = int(row['group_index']), int(row['active_slot'])
        if not (0 <= gi < groups and 0 <= si < 8):
            raise ValueError('Saved group/node index only')
        role = bool(row['fit'])
        if gi in group_roles and group_roles[gi] != role:
            raise ValueError('A group cannot cross original FIT/DEV')
        group_roles[gi] = role
        q = np.asarray([np.nan, np.nan] if row['q'] is None else row['q'], np.float64)
        L = np.float64(np.nan if row['L'] is None else row['L'])
        if q.shape != (2,) or not np.array_equal(q, geometry['q8'][gi, si], equal_nan=True) or not np.array_equal(L, geometry['L8'][gi, si], equal_nan=True):
            raise ValueError('Original q/L differs from group slot; no correction')
        if bool(row['original_query_available']) != bool(geometry['original_available8'][gi, si]) or bool(row['crop_available']) != bool(geometry['node_available'][gi, si]):
            raise ValueError('Original/crop availability equality')
        i = row['sample_index']
        if i is None:
            if int(row['target']) != -1:
                raise ValueError('Unmapped S/U remain target-1')
            continue
        i = int(i)
        if not 0 <= i < n or group[i] >= 0:
            raise ValueError('Each known sample exactly once')
        if row['row_id'] != str(data['row_id'][i]) or int(row['family_code']) != int(data['family'][i]) or int(row['target']) != int(data['target'][i]) or role != bool(data['fit'][i]) or bool(row['crop_available']) != bool(data['available'][i]):
            raise ValueError('Exact known sample metadata identity')
        group[i], slot[i] = gi, si
    if (group < 0).any():
        raise ValueError('Missing known sample; no fallback mapping')
    return group, slot


def relative_xy(geometry, group, slot):
    q = geometry['q8'][group]; L = geometry['L8'][group, slot]
    active = q[np.arange(len(group)), slot]
    avail = geometry['original_available8'][group]
    if not np.isfinite(active).all() or not np.isfinite(L).all() or not (L > 0).all() or not np.isfinite(q[avail]).all():
        raise ValueError('Available original q and active L finite; no substitute')
    out = np.zeros((len(group), 8, 2), np.float64)
    np.divide(q-active[:, None, :], L[:, None, None], out=out, where=avail[:, :, None])
    out = np.clip(out, -8, 8).astype(np.float32)
    out[~avail] = 0; out[np.arange(len(group)), slot] = 0
    return out


def source_gate(control, treatment):
    gain = []; drops = []
    for family in (0, 1):
        a, b = control[f'family{family}_DEV'], treatment[f'family{family}_DEV']
        old = [a[k]['recall'] for k in ('D', 'E')]
        new = [b[k]['recall'] for k in ('D', 'E')]
        if any(x is None for x in old+new):
            return {'PASS': False, 'reason': 'Missing prospective SOURCE class denominator', 'current_admission': False}
        gain.append((sum(new)-sum(old))/2)
        drops.extend([o-v for o, v in zip(old, new)])
    gain_pp = 100*sum(gain)/2; drop_pp = 100*max(drops)
    ycb_e = treatment['family1_DEV']['E']['recall']
    passed = gain_pp >= 3 and drop_pp <= 2 and ycb_e >= .5
    return {'PASS': passed, 'mean_family_DEV_balanced_recall_gain_pp': gain_pp,
            'max_family_class_recall_drop_pp': drop_pp, 'YCB_DEV_E_recall': ycb_e,
            'fixed_min_gain_pp': 3, 'fixed_max_class_drop_pp': 2, 'fixed_YCB_DEV_E_floor': .5,
            'current_admission': False, 'new_current_inference': 0, 'automatic_retry': False}


def paired_changes(control_rows, treatment_rows):
    result = {}
    for family in (0, 1):
        for role in ('FIT', 'DEV'):
            take = [(a, b) for a, b in zip(control_rows, treatment_rows) if a['family'] == family and a['role'] == role]
            recovered = damaged = 0
            for a, b in take:
                truth = 'D' if a['target'] == 0 else 'E'
                recovered += a['prediction'] != truth and b['prediction'] == truth
                damaged += a['prediction'] == truth and b['prediction'] != truth
            result[f'family{family}_{role}'] = {'rows': len(take), 'recovered': recovered, 'damaged': damaged, 'net': recovered-damaged}
    return result


def extract_context_features(model, node_rgb, node_valid, context_rgb, context_valid, geometry, preprocess, preprocess_context, device):
    """Input-only frozen features, including supported S/U nodes; labels unused."""
    g = len(geometry['q8'])
    node = np.zeros((g, 8, 384), np.float32); context = np.zeros((g, 384), np.float32)
    for start in range(0, g, 32):
        vv = np.array(node_valid[start:start+32], copy=True)
        expected = geometry['original_available8'][start:start+32] & vv[:, :, 47:49, 47:49].all((2, 3)) & vv.any((2, 3))
        fractions = vv.mean((2, 3)).astype(np.float32)
        cv = np.array(context_valid[start:start+32], copy=True)
        if not np.array_equal(expected, geometry['node_available'][start:start+32]) or not np.array_equal(fractions, geometry['node_fraction'][start:start+32]) or not np.array_equal(cv.mean((1, 2)).astype(np.float32), geometry['context_fraction'][start:start+32]):
            raise ValueError('Exact saved geometry/validity mask override')
    supported = np.argwhere(geometry['node_available'])
    with torch.no_grad():
        for start in range(0, len(supported), 32):
            gi, si = supported[start:start+32].T
            pix = torch.from_numpy(np.array(node_rgb[gi, si], copy=True))
            vv = torch.from_numpy(np.array(node_valid[gi, si, None], copy=True))
            views, fractions = preprocess(pix, vv)
            if not torch.equal(fractions[:, 0], torch.from_numpy(geometry['node_fraction'][gi, si])):
                raise ValueError('Original node full-fraction parity')
            cls = model.forward_features(views[:len(gi)].to(device))['x_norm_clstoken']
            node[gi, si] = F.normalize(cls, p=2, dim=1, eps=1e-12).cpu().numpy()
            if start % 1024 == 0:
                print(json.dumps({'stage': 'frozen_node_CLS', 'nodes': min(start+32, len(supported)), 'total': len(supported)}), flush=True)
        contexts = np.flatnonzero(geometry['context_fraction'] > 0)
        for start in range(0, len(contexts), 32):
            gi = contexts[start:start+32]
            pix = torch.from_numpy(np.array(context_rgb[gi], copy=True))
            vv = torch.from_numpy(np.array(context_valid[gi, None], copy=True))
            x = preprocess_context(pix, vv)
            cls = model.forward_features(x.to(device))['x_norm_clstoken']
            context[gi] = F.normalize(cls, p=2, dim=1, eps=1e-12).cpu().numpy()
        if not np.isfinite(node).all() or not np.isfinite(context).all():
            raise ValueError('Finite unit-L2 frozen descriptors')
    return node, context, len(supported)


def node_args(indices, group, slot, relative, geometry, node, context, device):
    gi = group[indices]
    return (torch.as_tensor(node[gi], device=device), torch.as_tensor(context[gi], device=device),
            torch.as_tensor(slot[indices], device=device),
            torch.as_tensor(geometry['original_available8'][gi], device=device),
            torch.as_tensor(geometry['node_available'][gi], device=device),
            torch.as_tensor(geometry['context_fraction'][gi] > 0, device=device),
            torch.as_tensor(relative[indices], device=device))


def alignment(module, x, args):
    node, _, slot, available, supported, _, _ = args
    qa = module.verify_active_node_alignment(x, node, slot, available, supported, atol=1e-6)
    if qa['checked'] != len(x):
        raise ValueError('Alignment must cover every original known active row')
    return qa


def run(entry_path):
    e = json.loads(read(entry_path))
    if e.get('approved_source_context_residual_training') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False:
        raise ValueError('Explicit new Root SOURCE entry required')
    if e['steps'] != 1000 or e['threshold_logit'] != 0 or e['active_alignment_atol'] != 1e-6 or e.get('use_V235_counterfactuals') is not False:
        raise ValueError('Fixed protocol1000/zero/1e-6/no-new-CF')
    code, meta, payload = e['code_whitelist'], e['metadata_whitelist'], e['payload_whitelist']
    verify(e['code'], code, (str(safe(__file__)), binding(__file__)['sha256']))
    verify(e['root_protocol'], meta, PINS['root_protocol']); verify(e['root_scope'], meta)
    verify(e['implementation_protocol'], code)
    protocol = json.loads(read(e['root_protocol']['path']))
    admission = json.loads(read(verify(e['input_admission'], meta, PINS['input_admission'])))
    input_qa = json.loads(read(verify(e['independent_input_QA'], meta)))
    if not all(x.get('completed') is True and x.get('PASS') is True for x in (admission, input_qa)) or identity(e['independent_input_QA']) != identity(admission['independent_QA']):
        raise ValueError('Root input admission and exact independent input QA PASS')
    if admission['training_protocol_sha256'] != e['root_protocol']['sha256'] or identity(admission['input_complete']) != identity(e['input_complete']):
        raise ValueError('Input admission protocol/data closure')
    m = {k: json.loads(read(verify(e[k], meta))) for k in ('source_complete', 'input_complete', 'input_manifest')}
    old, inp, manifest = m['source_complete'], m['input_complete'], m['input_manifest']
    if not all(x.get('completed') is True and x.get('PASS') is True for x in m.values()):
        raise ValueError('SOURCE completed metadata required')
    if inp['new_V235_counterfactuals'] != 0 or inp['new_current_or_held_payload'] != 0 or inp['known7617_original_RGB96_and_valid_SHA_exact'] != 7617:
        raise ValueError('Original actual scenes and exact old crops only')
    if e['input_complete']['sha256'] != protocol['data']['V237_input_complete_sha256'] or e['input_manifest']['sha256'] != protocol['data']['direct_group_manifest_sha256']:
        raise ValueError('Root fixed input metadata closure')
    if identity(e['sample_map']) != identity(manifest['all_rows']):
        raise ValueError('Exact ALL15040 sample map')
    outputs = {Path(b['path']).name: b for b in inp['outputs']}
    for key, filename in [('node_rgb', 'NODE_RGB96.npy'), ('node_valid', 'NODE_VALID96.npy'), ('context_rgb', 'CONTEXT_RGB224_F32.npy'), ('context_valid', 'CONTEXT_VALID224.npy'), ('geometry_npz', 'INPUT_GEOMETRY.npz')]:
        if identity(e[key]) != identity(outputs[filename]):
            raise ValueError('Input body binding differs from sealed original scene build')
    for key, container in [('control_checkpoint', 'checkpoints'), ('control_log', 'logs'), ('control_csv', 'outputs')]:
        if identity(e[key]) != identity(old[container][CONTROL]):
            raise ValueError('Original CONTROL byte reuse only')
    for key in ('prefix_npy', 'samples_npz'):
        if identity(e[key]) != identity(old[key]):
            raise ValueError('Original prefix/sample packet equality')
    utility = load_code(e['utility'], code, 'v238_inert_v235_helpers', PINS['utility'])
    source, crop, runtime = utility.load_sources(e)
    relational = load_code(e['relational_module'], code, 'v238_relational_head', PINS['relational_module'])
    preprocess = utility.load_preprocess(e); preprocess_context = load_context_preprocess(e)
    source.configure_runtime()
    config = e['model_loader_config']
    if identity(config['official_state']) != identity(old['official_base']) or config['asset_identity'] != old['asset_identity']:
        raise ValueError('Same original frozen official asset')
    paths = {k: verify(e[k], payload) for k in ('control_checkpoint', 'control_log', 'control_csv', 'prefix_npy', 'samples_npz', 'node_rgb', 'node_valid', 'context_rgb', 'context_valid', 'geometry_npz')}
    rows = [json.loads(line) for line in read(verify(e['sample_map'], meta)).decode().splitlines()]
    if len(rows) != 15040:
        raise ValueError('All15040 original rows conserved')
    prefix = np.load(paths['prefix_npy'], mmap_mode='r', allow_pickle=False)
    if prefix.dtype != np.float32 or prefix.ndim != 3 or tuple(prefix.shape[1:]) != (261, 384):
        raise ValueError('Original prefix header')
    with np.load(paths['samples_npz'], allow_pickle=False) as z:
        data = source.validate_samples({k: z[k] for k in z.files}, len(prefix))
    draws, eligible = source.fixed_draws(data)
    fit = np.flatnonzero(eligible); dev = np.flatnonzero(~data['fit'] & data['available'])
    if len(data['target']) != 7617 or not data['available'].all() or len(fit) != 5876 or len(dev) != 1741:
        raise ValueError('Fixed7617 available /5876FIT/1741DEV; never force mismatched census')
    draw_sha = hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    with np.load(paths['geometry_npz'], allow_pickle=False) as z:
        geometry = validate_geometry({k: z[k] for k in GEOMETRY_KEYS}, 1767)
    group, slot = map_samples(rows, data, geometry)
    relative = relative_xy(geometry, group, slot)
    syn = data['family'] == 0
    if not np.array_equal(relative[syn], data['condition'][syn, 16:].reshape(-1, 8, 2)):
        raise ValueError('Original SYN condition relative geometry unchanged')
    node_rgb = np.load(paths['node_rgb'], mmap_mode='r', allow_pickle=False)
    node_valid = np.load(paths['node_valid'], mmap_mode='r', allow_pickle=False)
    context_rgb = np.load(paths['context_rgb'], mmap_mode='r', allow_pickle=False)
    context_valid = np.load(paths['context_valid'], mmap_mode='r', allow_pickle=False)
    for value, shape, dtype in [(node_rgb, (1767, 8, 3, 96, 96), np.uint8), (node_valid, (1767, 8, 96, 96), bool), (context_rgb, (1767, 3, 224, 224), np.float32), (context_valid, (1767, 224, 224), bool)]:
        if value.shape != shape or value.dtype != dtype:
            raise ValueError('Exact original input layout')
    control = torch.load(paths['control_checkpoint'], weights_only=True, map_location='cpu')
    records = [json.loads(line) for line in read(paths['control_log']).decode().splitlines()]
    control_rows, expected = utility.parse_control_rows(paths['control_csv'], data)
    initial = source.cpu_copy(control['initial_head_state_dict']); initial_sha = source.tensor_state_sha(initial)
    if draw_sha != old['draws_sha256'] or initial_sha != old['head_initial_sha256']:
        raise ValueError('Same original1000 draws and initial76098 head')
    out = safe(e['output_directory'])
    if not out.is_relative_to(R/'V238_ROOT_CONTEXT_RESIDUAL') or out.exists():
        raise ValueError('Fresh owned V238 output only')
    out.mkdir(); started = time.monotonic()
    write_json(out/'STARTED.json', {'entry': binding(entry_path), 'source_only': True, 'steps': 1000, 'automatic_retry': False})
    try:
        model = source.load_backbone(config, e['device'], treatment=False)
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Official encoder frozen; no LoRA')
        base_sha = source.tensor_state_sha(model.state_dict(), base=True)
        reuse_config = {'official_state': config['official_state'], 'reuse_control': {'original_entry': old['control_reuse']['original_entry']}}
        source.validate_control_reuse(control, records, initial, initial_sha, draws, draw_sha, base_sha, reuse_config)
        copies = {}
        for name, key, suffix in [('checkpoint', 'control_checkpoint', '_FINAL1000.pt'), ('log', 'control_log', '_TRAIN1000.jsonl'), ('CSV', 'control_csv', '_ALL_SOURCE_ROWS.csv')]:
            copies[name] = utility.copy_bound(paths[key], out/(CONTROL+suffix))
            if copies[name]['sha256'] != e[key]['sha256']:
                raise ValueError('CONTROL bytes changed')
        reuse = write_json(out/'CONTROL_REUSE_QA.json', {'completed': True, 'PASS': True, 'original': {k: e[k] for k in ('control_checkpoint', 'control_log', 'control_csv')}, 'copied': copies, 'new_CONTROL_optimizer_updates': 0, 'same_original_initial_head_sha256': initial_sha, 'same_original_draws_sha256': draw_sha})
        node, context, supported_nodes = extract_context_features(model, node_rgb, node_valid, context_rgb, context_valid, geometry, preprocess, preprocess_context, e['device'])
        features = {}
        for name, values in [('NODE_CLS.npy', node), ('CONTEXT_CLS.npy', context), ('SAMPLE_GROUP_INDEX.npy', group), ('SAMPLE_NODE_INDEX.npy', slot), ('RELATIVE_XY8.npy', relative)]:
            np.save(safe(out/name), values, allow_pickle=False); features[name] = binding(out/name)
        store = source.PrefixViews(prefix, data)
        chunks = []; max_alignment = 0.
        for first in range(0, len(fit), 64):
            indices = fit[first:first+64]
            x = source.features_for(indices, store, model, runtime, None, e['device'], False).detach()
            qa = alignment(relational, x, node_args(indices, group, slot, relative, geometry, node, context, e['device']))
            max_alignment = max(max_alignment, qa['max_abs']); chunks.append(x)
        fit_features = torch.cat(chunks); del chunks
        np.save(safe(out/'ORIGINAL_FIT_FEATURES.npy'), fit_features.cpu().numpy(), allow_pickle=False)
        features['ORIGINAL_FIT_FEATURES.npy'] = binding(out/'ORIGINAL_FIT_FEATURES.npy')
        fit_map = np.full(len(eligible), -1, np.int64); fit_map[fit] = np.arange(len(fit))
        alignment_fit = write_json(out/'FIT_ACTIVE_DESCRIPTOR_QA_BEFORE_OPTIMIZER.json', {'completed': True, 'PASS': True, 'checked': len(fit), 'max_abs': max_alignment, 'atol': 1e-6, 'rtol': 0, 'mismatches_ignored_or_overwritten': 0})
        torch.manual_seed(17)
        head = relational.RelationalQueryHead().float().to(e['device'])
        head.load_original_initial_base(initial)
        if sum(p.numel() for p in head.parameters()) != 86054 or source.tensor_state_sha(head.base.state_dict()) != initial_sha:
            raise ValueError('Exact original initial base76098 plus residual9956')
        if torch.count_nonzero(head.relational.readout[-1].weight) or torch.count_nonzero(head.relational.readout[-1].bias):
            raise ValueError('Residual final layer zero init')
        initial_full = source.cpu_copy(head.state_dict()); initial_full_sha = source.tensor_state_sha(initial_full)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        log_path = out/(TREAT+'_TRAIN1000.jsonl'); ledger_path = out/'ORIGINAL_DRAW_LEDGER1000.jsonl'
        with log_path.open('x') as log, ledger_path.open('x') as ledger:
            for step, indices in enumerate(draws):
                if not eligible[indices].all() or (fit_map[indices] < 0).any():
                    raise ValueError('Only original known available FIT loss')
                x = fit_features[torch.as_tensor(fit_map[indices], device=e['device'])]
                optimizer.zero_grad(set_to_none=True)
                logits, _ = head(x, *node_args(indices, group, slot, relative, geometry, node, context, e['device']))
                loss, parts = crop.family_loss(logits, torch.as_tensor(data['target'][indices], device=e['device']), torch.as_tensor(data['family'][indices], device=e['device']))
                if not torch.isfinite(loss):
                    raise ValueError('Finite two-referent objective')
                loss.backward(); optimizer.step()
                item = {'update': step+1, 'sample_indices': indices.tolist(), 'draws_sha256': draw_sha, 'family_counts': [16, 16], 'all_fit_available_known': True}
                ledger.write(json.dumps(item, allow_nan=False)+'\n')
                item.update(family0_BCE=float(parts[0].detach()), family1_BCE=float(parts[1].detach()), total_loss=float(loss.detach()))
                log.write(json.dumps(item, allow_nan=False)+'\n')
                if (step+1) % 100 == 0:
                    print(json.dumps({'stage': 'context_residual_fit', 'updates': step+1, 'fixed_final': 1000}), flush=True)
        if len(optimizer.state) != 10 or any(float(s['step']) != 1000 or any(not torch.isfinite(v).all() for v in s.values() if isinstance(v, torch.Tensor)) for s in optimizer.state.values()):
            raise ValueError('All10 finite head optimizer states at final1000')
        if any(not torch.isfinite(p).all() for p in head.parameters()) or source.tensor_state_sha(model.state_dict(), base=True) != base_sha:
            raise ValueError('Finite head and unchanged frozen encoder')
        checkpoint = {'arm': TREAT, 'updates': 1000, 'use_lora': False, 'head_parameters': 86054, 'base_head_parameters': 76098, 'residual_parameters': 9956, 'adapter_parameters': 0, 'head_state_dict': source.cpu_copy(head.state_dict()), 'optimizer_state_dict': source.cpu_copy(optimizer.state_dict()), 'initial_head_state_dict': initial_full, 'initial_head_sha256': initial_full_sha, 'original_initial_base_state_dict': initial, 'original_initial_base_sha256': initial_sha, 'official_base': config['official_state'], 'base_state_sha256_before_after': base_sha, 'original_draws_sha256': draw_sha, 'entry': binding(entry_path), 'source_referents': {'0': 'SYN intended-native witness0 MAIN', '1': 'YCB associated-r AUX/nativeUNKNOWN'}}
        cp_path = out/(TREAT+'_FINAL1000.pt'); torch.save(checkpoint, cp_path)
        final_sha = source.tensor_state_sha(head.state_dict())
        freeze = write_json(out/'FINAL1000_FROZEN_BEFORE_NEW_HEAD_DEV_FORWARD.json', {'completed': True, 'PASS': True, 'checkpoints': {CONTROL: copies['checkpoint'], TREAT: binding(cp_path)}, 'CONTROL_reuse_QA': reuse, 'new_CONTROL_optimizer_updates': 0, 'new_TREAT_optimizer_updates': 1000, 'new_head_DEV_forward_before_freeze': 0, 'same_original_draws_sha256': draw_sha, 'original_initial_base_sha256': initial_sha, 'FIT_active_descriptor_QA': alignment_fit, 'original_prefix_numeric_rows_before_freeze_only_FIT': True, 'original_FIT_sample_indices': sorted(store.prebarrier_sample_indices), 'frozen_NODE_context_features_may_include_DEV': True, 'new_V235_CF': 0})
        store.final_barrier = True
        head.eval().requires_grad_(False)
        control_head = crop.CropFamilyHead().float().to(e['device']); control_head.load_state_dict(control['head_state_dict'], strict=True); control_head.eval().requires_grad_(False)
        all_indices = np.flatnonzero(data['available']); scores = {}; replayed = hashlib.sha256(); reference_hash = hashlib.sha256(); dev_checked = 0; dev_max = 0.
        original_all_features = np.zeros((len(data['target']), 1186), np.float32)
        with torch.no_grad():
            for first in range(0, len(all_indices), 64):
                indices = all_indices[first:first+64]
                x = source.features_for(indices, store, model, runtime, None, e['device'], False)
                original_all_features[indices] = x.cpu().numpy()
                actual = control_head(x).cpu().numpy(); expected_batch = expected[indices]
                if actual.tobytes() != expected_batch.tobytes():
                    write_json(out/'CONTROL_REPLAY_FAILURE.json', {'completed': False, 'PASS': False, 'indices': indices.tolist(), 'max_abs': float(np.max(np.abs(actual-expected_batch))), 'no_tolerance_relaxation_or_retraining': True})
                    raise ValueError('Original CONTROL both-logit byte replay mismatch')
                replayed.update(actual.tobytes()); reference_hash.update(expected_batch.tobytes())
                take_dev = ~data['fit'][indices]
                if take_dev.any():
                    selected = indices[take_dev]
                    qa = alignment(relational, x[torch.as_tensor(take_dev, device=e['device'])], node_args(selected, group, slot, relative, geometry, node, context, e['device']))
                    dev_checked += qa['checked']; dev_max = max(dev_max, qa['max_abs'])
                # DEV within this batch has passed descriptor QA before its first head output.
                z, _ = head(x, *node_args(indices, group, slot, relative, geometry, node, context, e['device']))
                z = z.cpu().numpy()
                if not np.isfinite(z).all():
                    raise ValueError('Finite frozen available logits')
                scores.update({int(i): value for i, value in zip(indices, z)})
        if dev_checked != 1741:
            raise ValueError('Every DEV active descriptor checked, no silent subset')
        np.save(safe(out/'ORIGINAL_ALL_FEATURES.npy'), original_all_features, allow_pickle=False)
        features['ORIGINAL_ALL_FEATURES.npy'] = binding(out/'ORIGINAL_ALL_FEATURES.npy')
        alignment_dev = write_json(out/'DEV_ACTIVE_DESCRIPTOR_QA_AFTER_FREEZE.json', {'completed': True, 'PASS': True, 'checked': dev_checked, 'max_abs': dev_max, 'atol': 1e-6, 'rtol': 0, 'FIT_plus_DEV_unique': len(fit)+dev_checked, 'new_DEV_logits_before_own_descriptor_check': 0})
        replay = write_json(out/'CONTROL_ALL_SOURCE_REPLAY_QA.json', {'completed': True, 'PASS': True, 'rows': len(data['target']), 'both_logits_byte_exact': True, 'actual_FP32_bytes_sha256': replayed.hexdigest(), 'original_CSV_FP32_bytes_sha256': reference_hash.hexdigest(), 'original_features_for_batch64_schedule': True})
        treatment_rows = []
        for i in range(len(data['target'])):
            f = int(data['family'][i]); z = scores.get(i); value = None if z is None else float(z[f])
            treatment_rows.append({'sample_index': i, 'row_id': str(data['row_id'][i]), 'family': f, 'role': 'FIT' if data['fit'][i] else 'DEV', 'available': bool(data['available'][i]), 'target': int(data['target'][i]), 'logit0': None if z is None else float(z[0]), 'logit1': None if z is None else float(z[1]), 'selected_family_logit': value, 'prediction': 'U' if value is None else 'E' if value >= 0 else 'D'})
        csv_path = out/(TREAT+'_ALL_SOURCE_ROWS.csv')
        with csv_path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(treatment_rows[0])); writer.writeheader(); writer.writerows(treatment_rows)
        reductions = {CONTROL: source.reduce_rows(control_rows), TREAT: source.reduce_rows(treatment_rows)}
        result = write_json(out/'SOURCE_RESULTS.json', {'completed': True, 'PASS': True, 'arms': reductions, 'paired_changes': paired_changes(control_rows, treatment_rows), 'SOURCE_gate': source_gate(reductions[CONTROL], reductions[TREAT]), 'two_referents_not_merged': True, 'original7617_preserved': True, 'masked7423_retained_in_full15040_ledger': True, 'current_admission': False, 'new_current_inference': 0, 'automatic_retry': False})
        if source.tensor_state_sha(head.state_dict()) != final_sha or source.tensor_state_sha(model.state_dict(), base=True) != base_sha:
            raise ValueError('Final state changed while producing saved outputs')
        complete = {'completed': True, 'PASS': True, 'entry': binding(entry_path), 'code': e['code'], 'implementation_protocol': e['implementation_protocol'], 'root_protocol': e['root_protocol'], 'root_scope': e['root_scope'], 'input_admission': e['input_admission'], 'independent_input_QA': e['independent_input_QA'], 'source_complete': e['source_complete'], 'input_complete': e['input_complete'], 'input_manifest': e['input_manifest'], 'sample_map': e['sample_map'], 'relational_module': e['relational_module'], 'source_trainer': e['source_trainer'], 'crop_module': e['crop_module'], 'prefix_runtime': e['prefix_runtime'], 'preprocess_CODE': e['preprocess_CODE'], 'context_preprocess_CODE': e['context_preprocess_CODE'], 'official_base': config['official_state'], 'asset_identity': config['asset_identity'], 'original_draws_sha256': draw_sha, 'original_initial_base_sha256': initial_sha, 'initial_head_sha256': initial_full_sha, 'base_state_sha256_before_after': base_sha, 'checkpoints': {CONTROL: copies['checkpoint'], TREAT: binding(cp_path)}, 'logs': {CONTROL: copies['log'], TREAT: binding(log_path)}, 'draw_source_ledger': binding(ledger_path), 'features': features, 'supported_input_nodes_including_S_U_if_available': supported_nodes, 'CONTROL_reuse_QA': reuse, 'CONTROL_replay_QA': replay, 'FIT_active_descriptor_QA': alignment_fit, 'DEV_active_descriptor_QA': alignment_dev, 'freeze': freeze, 'outputs': {CONTROL: copies['CSV'], TREAT: binding(csv_path)}, 'summary': result, 'new_CONTROL_optimizer_updates': 0, 'new_TREAT_optimizer_updates': 1000, 'head_parameters': {CONTROL: 76098, TREAT: 86054}, 'new_DINO_or_LoRA_parameters': 0, 'new_V235_CF': 0, 'new_head_DEV_forward_before_freeze': 0, 'automatic_retry': False, 'current_admission': False, 'actual_seconds': time.monotonic()-started}
        write_json(out/'COMPLETE.json', complete)
        print(json.dumps({'completed': True, 'PASS': True, 'COMPLETE': binding(out/'COMPLETE.json')}), flush=True)
    except BaseException:
        write_json(out/'FAILED_EXECUTION.json', {'completed': False, 'PASS': False, 'automatic_retry': False, 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--entry', required=True)
    run(parser.parse_args().entry)
