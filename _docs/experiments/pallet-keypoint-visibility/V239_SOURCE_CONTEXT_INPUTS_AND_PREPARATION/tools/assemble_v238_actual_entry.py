"""Assemble one exact SOURCE trial only after a pinned independent CODE PASS."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H = R / 'V238_ROOT_CONTEXT_RESIDUAL'


def safe(p):
    s = str(p)
    assert not any(x in s.lower() for x in ('v112', '__pycache__', '.pyc', 'current223', 'publicheld', 'oldval', 'v177', 'v178', 'source_holdout', 'v182', 'private_evaluation'))
    return Path(os.path.abspath(s))


def read(p):
    return safe(p).read_bytes()


def bind(p):
    p = safe(p); b = read(p)
    return {'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)}


def write(p, d):
    with safe(p).open('x') as f:
        json.dump(d, f, ensure_ascii=False, indent=2); f.write('\n')


def collect_bindings(value):
    found = []
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            found.append((str(safe(value['path'])), value['sha256']))
        for x in value.values():
            found.extend(collect_bindings(x))
    elif isinstance(value, list):
        for x in value:
            found.extend(collect_bindings(x))
    return found


def deduplicate(items):
    found = {}
    for b in items:
        p = str(safe(b['path']))
        if p in found:
            assert found[p]['sha256'] == b['sha256']
        else:
            found[p] = b
    return list(found.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--qa', required=True)
    parser.add_argument('--qa-sha', required=True)
    args = parser.parse_args()
    qb = bind(args.qa); assert qb['sha256'] == args.qa_sha
    qa = json.loads(read(args.qa)); assert qa.get('completed') is True and qa.get('PASS') is True
    e = json.loads(read(args.template))
    assert e['approved_source_context_residual_training'] is False
    assert (str(safe(e['code']['path'])), e['code']['sha256']) in collect_bindings(qa)
    assert bind(e['code']['path'])['sha256'] == e['code']['sha256']
    old_path = R / 'V235_ROOT_PUBLIC_COUNTERFACTUAL/ACTUAL_DATA_ONLY_HEAD_FIT_ENTRY.json'
    assert bind(old_path)['sha256'] == 'cba9789738e36327e9bbae1fe84e6e1d26ea04511ae0900d1e585e7c4a870a2c'
    old = json.loads(read(old_path))
    for key in ('control_checkpoint', 'control_log', 'control_csv', 'prefix_npy', 'samples_npz'):
        e[key] = old[key]
    config = old['model_loader_config']
    config['code_whitelist'] = deduplicate([config[k] for k in ('backbone_factory', 'prefix_runtime', 'adapter_module', 'crop_module')] + config['backbone_code_bindings'])
    config['approved_official_backbone_load'] = True
    config['approved_source_local_crop_paired_adaptation'] = False
    e['model_loader_config'] = config
    e['approved_source_context_residual_training'] = True
    e['created_UTC'] = datetime.now(timezone.utc).isoformat()
    e['independent_trainer_CODE_QA'] = qb
    e['output_directory'] = str(H / 'context_residual_source_fit_v1')
    assert not safe(e['output_directory']).exists()
    scope = {
        'version': 'V238_ROOT_ACTUAL_FEATURES_AND_FIXED1000_FIT_SCOPE',
        'created_UTC': e['created_UTC'], 'approved': True, 'source_only': True,
        'new_control_optimizer_updates': 0, 'new_treatment_optimizer_updates': 1000,
        'root_protocol': e['root_protocol'], 'independent_CODE_QA': qb,
        'input_admission': e['input_admission'], 'input_complete': e['input_complete'],
        'old_assembly_metadata': bind(old_path),
        'scope': 'Only exact listed original SOURCE prefix/samples/control and V237 input nodes/context/geometry plus pinned official DINO/code. No raw GT, masks, depth, source-image pointer following or newV235CF. New head DEV outputs only after final freeze and complete descriptor barrier.',
        'current_or_held_allowed': False, 'PnP_allowed': False, 'new_download_or_install': False,
        'automatic_retry': False, 'output_directory': e['output_directory'],
    }
    scope_path = H / 'ACTUAL_FEATURE_FIT_SCOPE.json'; write(scope_path, scope)
    e['root_scope'] = bind(scope_path)
    e['code_whitelist'] = deduplicate(e['code_whitelist'] + config['code_whitelist'])
    e['metadata_whitelist'] = deduplicate(e['metadata_whitelist'] + [e['root_scope'], qb])
    e['payload_whitelist'] = deduplicate(e['payload_whitelist'] + [e[k] for k in ('control_checkpoint', 'control_log', 'control_csv', 'prefix_npy', 'samples_npz')] + config['payload_whitelist'])
    for b in e['code_whitelist'] + e['metadata_whitelist'] + e['payload_whitelist']:
        safe(b['path'])
    path = H / 'ACTUAL_SOURCE_CONTEXT_RESIDUAL_FIT_ENTRY.json'; write(path, e)
    print(json.dumps({'entry': bind(path), 'code': e['code'], 'scope': bind(scope_path), 'features_or_fit_run_by_assembler': 0}))


if __name__ == '__main__':
    main()
