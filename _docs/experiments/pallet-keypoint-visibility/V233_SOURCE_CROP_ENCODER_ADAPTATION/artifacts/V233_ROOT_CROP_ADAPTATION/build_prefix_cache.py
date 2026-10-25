"""Cache fixed first11 blocks from declared common-local-crop SOURCE inputs."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ['XFORMERS_DISABLED'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from torch.nn import functional as F


def safe(path):
    path = Path(path)
    if 'v112' in str(path).lower() or '__pycache__' in path.parts:
        raise ValueError('Forbidden path')
    for current in (path, *path.parents):
        if current.is_symlink():
            raise ValueError('No symlink in declared path')
    return path


def digest_file(path):
    hasher = hashlib.sha256()
    with safe(path).open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def pinned_bytes(binding):
    raw = safe(binding['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('Changed binding: ' + binding['path'])
    return raw


def load_code(name, binding):
    raw = pinned_bytes(binding)
    sys.path.insert(0, str(safe(binding['path']).parent))
    module_spec = importlib.util.spec_from_file_location(name, binding['path'])
    module = importlib.util.module_from_spec(module_spec)
    exec(compile(raw, binding['path'], 'exec'), module.__dict__)
    return module


def json_write(path, value):
    safe(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n')


def state_digest(backbone):
    hasher = hashlib.sha256()
    for name, value in sorted(backbone.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        hasher.update(name.encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True)
    args = parser.parse_args()
    entry_bytes = safe(args.entry).read_bytes()
    entry = json.loads(entry_bytes)
    if not entry.get('approved_SOURCE_prefix_cache'):
        raise ValueError('Closed root entry required')
    config = entry['model_loader_config']
    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    device = torch.device('cuda:0')
    for binding in config['code_whitelist']:
        pinned_bytes(binding)
    trainer = load_code('v233_trainer_for_prefix', entry['trainer_code'])
    runtime = load_code('v233_prefix_runtime', config['prefix_runtime'])
    # Reuse one exact tensor-only preprocessing function, never its old run().
    code = pinned_bytes(entry['preprocess_CODE'])
    tree = ast.parse(code)
    function = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'preprocess_views']
    if len(function) != 1:
        raise ValueError('Exact pure preprocessing function required')
    namespace = {'torch': torch, 'F': F, 'MEAN': (.485, .456, .406),
                 'STD': (.229, .224, .225)}
    exec(compile(ast.Module(body=function, type_ignores=[]),
                 entry['preprocess_CODE']['path'], 'exec'), namespace)
    preprocess = namespace['preprocess_views']
    data_bindings = entry['data_bindings']
    for binding in data_bindings.values():
        if digest_file(binding['path']) != binding['sha256']:
            raise ValueError('Changed newly materialized crop data')
    rgb = np.load(safe(data_bindings['rgb96']['path']), mmap_mode='r', allow_pickle=False)
    valid = np.load(safe(data_bindings['valid96']['path']), mmap_mode='r', allow_pickle=False)
    with np.load(safe(data_bindings['samples']['path']), allow_pickle=False) as packet:
        sample_ids = packet['row_id'].copy()
        full_view = packet['full_view'].copy()
        central_view = packet['central_view'].copy()
    count = len(rgb)
    assert rgb.shape == (count, 3, 96, 96) and rgb.dtype == np.uint8
    assert valid.shape == (count, 1, 96, 96) and valid.dtype == bool
    assert len(sample_ids) == count
    assert np.array_equal(full_view, 2 * np.arange(count))
    assert np.array_equal(central_view, 2 * np.arange(count) + 1)
    out = safe(entry['output_directory'])
    out.mkdir(exist_ok=False)
    json_write(out / 'STARTED.json', {
        'entry_sha256': hashlib.sha256(entry_bytes).hexdigest(),
        'samples': count, 'views': 2 * count, 'batch_samples': entry['batch_samples'],
        'stage': 'before official model load and SOURCE prefix forward',
    })
    backbone = trainer.load_backbone(config, device, treatment=False)
    state_before = state_digest(backbone)
    destination = safe(out / 'prefix.npy')
    cache = np.lib.format.open_memmap(destination, mode='w+', dtype=np.float32,
                                     shape=(2 * count, 261, 384))
    start = time.time()
    equivalence = None
    for first in range(0, count, entry['batch_samples']):
        last = min(count, first + entry['batch_samples'])
        pixels = torch.from_numpy(np.array(rgb[first:last], copy=True))
        support = torch.from_numpy(np.array(valid[first:last], copy=True))
        views, _ = preprocess(pixels, support)
        views = views.to(device)
        ids = ([str(x) + '/FULL96' for x in sample_ids[first:last]]
               + [str(x) + '/CENTRE32' for x in sample_ids[first:last]])
        packet = runtime.extract_prefix(backbone, views, {'view_ids': ids})
        prefix = packet['prefix_tokens']
        batch = last - first
        if equivalence is None:
            with torch.no_grad():
                original = backbone.forward_features(views)
                split = runtime.last_block_features(backbone, prefix.to(device),
                                                    adapter_grad=False)
            keys = ('x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm')
            exact = {key: torch.equal(original[key], split[key]) for key in keys}
            maximum = {key: float((original[key] - split[key]).abs().max()) for key in keys}
            if not all(exact.values()):
                json_write(out / 'EQUIVALENCE_FAILURE.json', {'exact': exact, 'max_abs': maximum})
                raise ValueError('Actual same-batch vendor forward vs prefix split differs')
            equivalence = {'views': len(views), 'exact': exact, 'max_abs': maximum,
                           'same_batch_and_preprocessed_pixels': True}
            json_write(out / 'ACTUAL_VENDOR_EQUIVALENCE.json', equivalence)
            del original, split
        positions = np.arange(first, last)
        cache[2 * positions] = prefix[:batch].numpy()
        cache[2 * positions + 1] = prefix[batch:].numpy()
        if first == 0 or last == count or (first // entry['batch_samples']) % 20 == 0:
            cache.flush()
            print(json.dumps({'prefix_samples': last, 'total': count,
                              'elapsed_seconds': round(time.time() - start, 2)}), flush=True)
        del views, packet, prefix, pixels, support
    cache.flush()
    del cache
    state_after = state_digest(backbone)
    if state_before != state_after:
        raise ValueError('Fixed prefix backbone state changed')
    completion = {
        'completed': True, 'PASS': True, 'samples': count, 'views': 2 * count,
        'shape': [2 * count, 261, 384], 'dtype': 'float32',
        'entry_sha256': hashlib.sha256(entry_bytes).hexdigest(),
        'actual_vendor_equivalence': equivalence,
        'backbone_state_before': state_before,
        'backbone_state_after': state_after,
        'backbone_state_unchanged': True,
        'prefix': {'path': str(destination), 'sha256': digest_file(destination),
                   'size_bytes': destination.stat().st_size},
        'elapsed_seconds': time.time() - start,
        'SOURCE_fit_and_internal_DEV_fixed_pretrained_prefixes': True,
        'trained_model_DEV_forward': 0, 'optimizer_steps': 0,
        'current_or_protected_payload': 0, 'new_PnP_or_query_correction': 0,
    }
    json_write(out / 'COMPLETE.json', completion)
    print(json.dumps({'completed': True, 'PASS': True,
                      'samples': count, 'views': 2 * count,
                      'elapsed_seconds': round(completion['elapsed_seconds'], 2)}), flush=True)


if __name__ == '__main__':
    main()
