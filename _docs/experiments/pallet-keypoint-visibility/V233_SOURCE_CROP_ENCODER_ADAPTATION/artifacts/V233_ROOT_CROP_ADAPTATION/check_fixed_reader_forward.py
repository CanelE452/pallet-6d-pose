"""Prove a deterministic fixed-centre repair preserves SOURCEFIT features."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types

os.environ['XFORMERS_DISABLED'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch


def safe(path):
    path = Path(path)
    if 'v112' in str(path).lower() or '__pycache__' in path.parts:
        raise ValueError('Forbidden path')
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Symlink rejected')
    return path


def verify(binding):
    sha = hashlib.sha256()
    with safe(binding['path']).open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            sha.update(chunk)
    if sha.hexdigest() != binding['sha256']:
        raise ValueError('Changed declared binding: ' + binding['path'])
    return safe(binding['path'])


def load_code(name, binding):
    path = verify(binding)
    sys.path.insert(0, str(path.parent))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True)
    args = parser.parse_args()
    entry_bytes = safe(args.entry).read_bytes()
    entry = json.loads(entry_bytes)
    if not entry.get('approved_source_FIT_forward_compatibility_only'):
        raise ValueError('Closed mechanical repair entry required')
    config = entry['original_loader_config']
    for binding in config['code_whitelist']:
        verify(binding)
    original_trainer = load_code('v233_original_for_repair_check', entry['original_trainer'])
    old_crop = load_code('v233_original_crop_for_repair_check', entry['original_crop'])
    new_crop = load_code('v233_fixed_crop_for_repair_check', entry['repaired_crop'])
    prefix_runtime = load_code('v233_prefix_for_repair_check', config['prefix_runtime'])
    adapter_module = load_code('v233_adapter_for_repair_check', config['adapter_module'])
    prefix_path = verify(entry['prefix'])
    sample_path = verify(entry['samples'])
    control_path = verify(entry['completed_control'])
    original_trainer.configure_runtime()
    checkpoint = torch.load(control_path, map_location='cpu', weights_only=True)
    if checkpoint['updates'] != 1000 or checkpoint['use_lora']:
        raise ValueError('Only completed original CONTROL1000')
    prefix = np.load(prefix_path, mmap_mode='r', allow_pickle=False)
    with np.load(sample_path, allow_pickle=False) as archive:
        data = original_trainer.validate_samples({key: archive[key] for key in archive.files}, len(prefix))
    eligible = data['fit'] & data['available'] & (data['target'] >= 0)
    indices = np.flatnonzero(eligible)
    if len(indices) != 5876:
        raise ValueError('Original full SOURCEFIT population changed')
    store = original_trainer.PrefixViews(prefix, data)
    device = torch.device('cuda')
    backbone = original_trainer.load_backbone(config, device, treatment=False)
    if backbone._v233_original_base_sha256 != checkpoint['base_state_sha256_before_after']:
        raise ValueError('Original completed CONTROL official base mismatch')
    head = old_crop.CropFamilyHead().float().to(device)
    head.load_state_dict(checkpoint['head_state_dict'], strict=True)
    head.eval().requires_grad_(False)
    before_head = original_trainer.tensor_state_sha(head.state_dict())
    output = safe(entry['output_directory'])
    output.mkdir(exist_ok=False)
    feature_old = hashlib.sha256()
    feature_new = hashlib.sha256()
    logits_old = hashlib.sha256()
    logits_new = hashlib.sha256()
    checked = 0
    with torch.no_grad():
        for start in range(0, len(indices), 64):
            rows = indices[start:start + 64]
            selected, inverse, count = store.fetch(rows, device)
            parts = prefix_runtime.last_block_features(backbone, selected, adapter_grad=False)
            full = {key: value[torch.as_tensor(inverse[:count], device=device)]
                    for key, value in parts.items() if isinstance(value, torch.Tensor)}
            centre = {key: value[torch.as_tensor(inverse[count:], device=device)]
                      for key, value in parts.items() if isinstance(value, torch.Tensor)}
            arguments = (full, centre,
                         torch.as_tensor(data['fractions'][rows], device=device),
                         torch.as_tensor(data['central4'][rows], device=device),
                         torch.as_tensor(data['condition'][rows], device=device),
                         adapter_module)
            old = old_crop.crop_features(*arguments)
            new = new_crop.crop_features(*arguments)
            if not torch.equal(old, new):
                raise ValueError('SOURCEFIT feature changed: maxabs=' + str(float((old-new).abs().max())))
            old_value = head(old)
            new_value = head(new)
            if not torch.equal(old_value, new_value):
                raise ValueError('Completed CONTROL logits changed')
            feature_old.update(old.cpu().contiguous().numpy().tobytes())
            feature_new.update(new.cpu().contiguous().numpy().tobytes())
            logits_old.update(old_value.cpu().contiguous().numpy().tobytes())
            logits_new.update(new_value.cpu().contiguous().numpy().tobytes())
            checked += count
            if start == 0 or start % 1024 == 0 or checked == len(indices):
                print(json.dumps({'SOURCEFIT_exact_checked': checked, 'total': len(indices)}), flush=True)
    if feature_old.digest() != feature_new.digest() or logits_old.digest() != logits_new.digest():
        raise ValueError('Numeric equality did not preserve literal FP32 feature/logit bytes')
    if (original_trainer.tensor_state_sha(backbone.state_dict(), base=True)
            != checkpoint['base_state_sha256_before_after']):
        raise ValueError('Base changed during compatibility check')
    if original_trainer.tensor_state_sha(head.state_dict()) != before_head:
        raise ValueError('Completed CONTROL head changed')
    used = sorted(store.prebarrier_sample_indices)
    if used != indices.tolist() or store.final_barrier:
        raise ValueError('Only all eligible FIT samples may be read before barrier')
    result = {
        'completed': True, 'PASS': True, 'entry_sha256': hashlib.sha256(entry_bytes).hexdigest(),
        'all_SOURCEFIT_samples': checked, 'feature_dimensions': 1186,
        'feature_Tensor_equal': True, 'feature_max_abs': 0,
        'old_feature_FP32_bytes_sha256': feature_old.hexdigest(),
        'new_feature_FP32_bytes_sha256': feature_new.hexdigest(),
        'completed_control_logits_Tensor_equal': True, 'logits_max_abs': 0,
        'old_logits_FP32_bytes_sha256': logits_old.hexdigest(),
        'new_logits_FP32_bytes_sha256': logits_new.hexdigest(),
        'completed_control': entry['completed_control'],
        'old_crop': entry['original_crop'], 'new_crop': entry['repaired_crop'],
        'official_base_and_control_head_unchanged': True,
        'fit_samples_checked': used, 'SOURCEDEV_forward': 0, 'optimizer_steps': 0,
        'current_or_held_payload_or_forward': 0, 'deterministic_algorithms': True,
    }
    path = output / 'COMPLETE.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'completed': True, 'PASS': True, 'SOURCEFIT_samples': checked,
                      'complete_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
