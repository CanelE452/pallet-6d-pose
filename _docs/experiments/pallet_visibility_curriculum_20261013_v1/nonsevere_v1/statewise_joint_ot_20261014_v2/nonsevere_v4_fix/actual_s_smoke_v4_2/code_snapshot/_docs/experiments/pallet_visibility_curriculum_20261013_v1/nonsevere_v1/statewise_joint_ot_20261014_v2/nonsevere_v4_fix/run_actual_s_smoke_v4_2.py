"""Authorized real-CAL S-carrier invariance audit; never reads targets/TEST."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
sys.path.insert(0, str(ROOT))
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import torch

from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_evaluation_v4 as E
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_uncertain_solver_v4_bias as B
from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_interval_api_v4_2 import StatewiseIntervalV4_2
from scripts.research.pallet_visibility_curriculum_20261013_v1 import evaluation_interval_v4_2 as I
from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_masks_v2 import StatewiseMaskProvider
from scripts.research.pallet_visibility_curriculum_20261013_v1.state_visibility_v2 import load_state_model
from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_uncertainty_calibration_v4 import load_uncertainty_model, solver_config_from_model

N = E.FIX
OWN = N
IDS = ['eval_night08:1779449488068910592', 'eval_outside:1778651519235162880', 'wood_184309:000432', 'eval_cad:1778653041187027200']


def digest(value):
    return hashlib.sha256(json.dumps(C.finite(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_binding(value):
    a = np.ascontiguousarray(value)
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': hashlib.sha256(a.tobytes()).hexdigest()}


def verify(bindings):
    for relative, expected in bindings.items():
        if C.sha(ROOT / relative) != expected:
            raise ValueError('Binding changed: ' + relative)


class TraceProvider:
    def __init__(self, actual):
        self.actual, self.calls = actual, []

    def __call__(self, image, **kwargs):
        bundle = self.actual(image, **kwargs)
        self.calls.append({'prompt_native_ids': list(kwargs['prompt_native_ids']),
            'prompt_points': np.asarray(kwargs['prompt_points']).tolist(),
            'prompt_box_xyxy': deepcopy(kwargs['prompt_box_xyxy']),
            'image': array_binding(image), 'K': array_binding(kwargs['K']),
            'initial_projection': array_binding(kwargs['initial_projection_xy']),
            'state_Tiny_mask': array_binding(bundle['state_pallet_mask']),
            'correction_MobileSAM_mask': array_binding(bundle['pallet_mask']),
            'actual_provider_provenance': deepcopy(bundle['provenance'])})
        return bundle


def normalized(runtime, hidden):
    result = deepcopy(runtime)
    for native_id in hidden:
        result['original_raw_points'][native_id] = ['EXCLUDED_S_CARRIER', 'EXCLUDED_S_CARRIER']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OWN / 'actual_s_smoke_v4_2')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != OWN.resolve() or output.exists():
        raise ValueError('New task-owned output directory required')
    import subprocess
    free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True).strip().splitlines()[0])
    if free < 4096: raise RuntimeError('Root concurrent GPU gate requires at least4096MiB free')
    selected, fixed_model, fixed_config, interval, interval_bindings = I.load_interval_configuration()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Authorized real-provider CUDA run required')
    choice_dir = N / 'bias_selection_v4_1'
    selection_seal = C.read(choice_dir / 'SELECTION_SEAL.json')
    selection_path = choice_dir / 'SELECTED_CONFIG.json'
    if C.sha(selection_path) != selection_seal['selected_config_sha256']:
        raise ValueError('Final CAL choice seal mismatch')
    choice = C.read(selection_path)
    if choice['TEST_files_opened'] or selection_seal['TEST_files_opened'] or not selection_seal['before_new_TEST_score']:
        raise ValueError('CAL-only selection must precede new TEST score')
    if choice['selected_prior_weight'] != 1. or choice['selected_bias_strength'] != 1.:
        raise ValueError('Root-authorized lambda1 beta1 choice required')
    model_path = ROOT / choice['calibration_model']['path']
    if C.sha(model_path) != choice['calibration_model']['sha256']:
        raise ValueError('Final CAL model changed')
    model = load_uncertainty_model(model_path)
    config = B._configuration(choice['solver_configuration'])
    for key, value in solver_config_from_model(model, 1.).items():
        if digest(config[key]) != digest(value):
            raise ValueError('Model/config coupling mismatch: ' + key)
    if config['prior_bias6'] != model['prior_fit']['median_bias6'] or config['bias_strength'] != 1.:
        raise ValueError('Final CAL mean/beta mismatch')
    if [config[k] for k in ('max_p3p_triples', 'global_rotation_starts', 'max_joint_iterations', 'maximum_union_branches')] != [56, 24, 250, 256]:
        raise ValueError('Full frozen search limits required')
    if choice != selected or config != fixed_config or model != fixed_model:
        raise ValueError('Smoke selected/model/config differs from strict interval loader')
    observation_path = N / 'fresh_observations_v2/CAL_OBSERVATIONS.jsonl.gz'
    obs_seal_path = observation_path.parent / 'OBSERVATION_SEAL.json'
    if C.sha(observation_path) != C.read(obs_seal_path)['CAL_observations_sha256']:
        raise ValueError('CAL input cache seal mismatch')
    with gzip.open(observation_path, 'rt') as f:
        cal = {r['id']: r for r in (json.loads(line) for line in f)}
    if len(cal) != 46 or any(r['partition'] != 'CAL' for r in cal.values()):
        raise ValueError('Only isolated CAL46 prediction inputs allowed')
    runtime_descriptor, paths, model_bindings = E._runtime_models()
    provider = StatewiseMaskProvider.from_configs(C.read(paths['mask_descriptor']), paths['boundary_mask_config'], 'cuda')
    source = {**choice['source_sha256'], **I.source_bindings()}
    for path in (Path(__file__), ROOT / 'scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_bias_api_v4_1.py'):
        source[str(path.relative_to(ROOT))] = C.sha(path)
    for path in provider.sam_vendor.rglob('*.py'):
        source[str(path.resolve().relative_to(ROOT))] = C.sha(path)
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT):
                source[str(path.relative_to(ROOT))] = C.sha(path)
    input_paths = [selection_path, choice_dir / 'SELECTION_SEAL.json', model_path, observation_path, obs_seal_path,
        N / 'TRUNCATION_INTERVAL_FIX_PLAN_V1.json', N / 'FIX_PROTOCOL_V1.json', N / 'CAL_MEDIAN_BIAS_AMENDMENT_V1.json']
    inputs = {str(p.relative_to(ROOT)): C.sha(p) for p in input_paths}
    for b in interval_bindings.values():
        if isinstance(b, dict) and 'path' in b: inputs[b['path']] = b['sha256']
    inputs.update(model_bindings)
    for fid in IDS:
        p = ROOT / cal[fid]['image']
        if C.sha(p) != cal[fid]['image_sha256']:
            raise ValueError('Real CAL image changed')
        inputs[cal[fid]['image']] = C.sha(p)
    protected = C.read(N / 'FIX_PROTOCOL_V1.json')['protected_user_sha256']
    verify(source); verify(inputs); verify(protected)
    output.mkdir()
    input_records = [{k: deepcopy(cal[fid][k]) for k in ('id', 'partition', 'image', 'image_sha256', 'K', 'raw_hw', 'initial_pose')}
        | {'raw_points': deepcopy(cal[fid]['runtime']['original_raw_points']),
            'native_xyz': deepcopy(cal[fid]['runtime']['initial_geometry']['native_xyz']),
            'initial_S_ids': [i for i, v in enumerate(cal[fid]['runtime']['initial_geometry']['initial_self_hidden']) if v]}
        for fid in IDS]
    C.write(output / 'INPUT_RECORDS.json', input_records)
    C.write(output / 'PROTOCOL.json', {'schema': 'REAL_CAL_S_RAW_INVARIANCE_V4_2', 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'ids': IDS, 'CAL_frames': 4, 'full_refine_calls': 8, 'device': 'cuda', 'workers': 1, 'CPU_threads': 1,
        'GPU_free_MiB_before_initialization': free, 'actual_provider_no_mask_or_feature_injection': True, 'lambda': 1., 'beta': 1., 'solver_configuration': config,
        'runtime_versions': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'cv2': cv2.__version__},
        'source_sha256': source, 'input_model_image_sha256': inputs, 'protected_user_sha256': protected,
        'input_records_sha256': C.sha(output / 'INPUT_RECORDS.json'),
        'input_quantity_hashes': {r['id']: {k: digest(r[k]) for k in ('raw_points', 'K', 'native_xyz', 'initial_pose')} for r in input_records},
        'known_native_reference_incompatible_CAL_stress_id': 'wood_184309:000432',
        'GT_coordinates_or_poses_used_for_inference': False, 'TEST_files_read': False,
        'CAL_acceptance_receipts_loaded_only_as_runtime_guard': True, 'interval_model_bindings': interval_bindings,
        'original_three_recipes_plus_fixed_T_to_U_CAL_case': True, 'pose_accuracy_scoring': False,
        'original_S_raw_metadata_difference_expected': True, 'normalized_full_output_except_S_raw_carrier_exact_required': True})
    for relative, expected in source.items():
        dest = output / 'code_snapshot' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
        if C.sha(dest) != expected:
            raise ValueError('Source snapshot mismatch')
    head = load_state_model(str(paths['state_model']), 'cuda')
    trace = TraceProvider(provider)
    start, results = time.monotonic(), []
    for record in input_records:
        image = cv2.imread(str(ROOT / record['image']))
        if image is None or list(image.shape[:2]) != record['raw_hw']:
            raise ValueError('Actual CAL image canvas mismatch')
        hidden = record['initial_S_ids']
        if not hidden:
            raise ValueError('Nonvacuous S test required')
        baseline_raw = np.asarray(record['raw_points'], float)
        changed = baseline_raw.copy()
        for k in hidden:
            changed[k] = [100000000. + 1000 * k, -1000000000. - 1000 * k]
        if not all(np.array_equal(baseline_raw[k], changed[k]) for k in range(len(changed)) if k not in hidden):
            raise ValueError('NonS/center carrier changed')
        api = StatewiseIntervalV4_2(head, C.read(paths['OT_model']), trace, uncertainty_model=model, truncation_model=interval,
            config={'solver': config, 'prediction_source': 'N3_SUBPIX', 'record_id': record['id'], 'device': 'cuda'})
        for variant, raw in [('BASELINE', baseline_raw), ('S_RAW_PERTURBED', changed)]:
            before_calls = len(trace.calls)
            runtime = api.refine(image, raw, record['K'], record['native_xyz'], deepcopy(record['initial_pose']))
            if len(trace.calls) != before_calls + 1:
                raise ValueError('Fresh actual provider call missing')
            row = {'id': record['id'], 'variant': variant, 'initial_S_ids': hidden, 'input_raw_sha256': digest(raw.tolist()),
                'runtime': runtime, 'actual_provider_trace': deepcopy(trace.calls[-1]), 'S_contract': E.S_contract_audit(runtime)}
            results.append(row)
            C.write(output / 'LIVE_PROGRESS.json', {'full_refine_completed': len(results), 'total': 8,
                'last_id': record['id'], 'variant': variant, 'elapsed_seconds': time.monotonic() - start})
            print(json.dumps({'completed': len(results), 'id': record['id'], 'variant': variant,
                'accepted': runtime['accepted'], 'BI': len(runtime['frozen_visible_measurements']), 'states': ''.join(runtime['states'])}), flush=True)
    verify(source); verify(inputs); verify(protected)
    C.save_rows(output / 'PREDICTIONS.jsonl.gz', results)
    C.write(output / 'PREDICTION_SEAL.json', {'schema': 'REAL_CAL_S_INVARIANCE_PREDICTION_SEAL_V4_2',
        'rows': 8, 'source_sha256': source, 'predictions_sha256': C.sha(output / 'PREDICTIONS.jsonl.gz'),
        'protocol_sha256': C.sha(output / 'PROTOCOL.json'), 'input_records_sha256': C.sha(output / 'INPUT_RECORDS.json'),
        'bindings_pre_and_post_match': True, 'no_GT_coordinate_or_pose_lookup_for_inference': True, 'no_TEST_reads': True,
        'before_pairwise_output_audit': True, 'elapsed_seconds': time.monotonic() - start})
    checks = []
    for a, b in zip(results[::2], results[1::2]):
        hidden = a['initial_S_ids']
        equal = digest(normalized(a['runtime'], hidden)) == digest(normalized(b['runtime'], hidden))
        checks.append({'id': a['id'], 'initial_S_ids': hidden, 'normalized_entire_runtime_exact': equal,
            'provider_Tiny_SAM_masks_prompts_quality_bindings_exact': a['actual_provider_trace'] == b['actual_provider_trace'],
            'both_S_zero_contracts_pass': a['S_contract']['passed'] and b['S_contract']['passed'],
            'only_expected_raw_carrier_hash_changes': a['input_raw_sha256'] != b['input_raw_sha256'],
            'BI_count': len(a['runtime']['frozen_visible_measurements']), 'OT_count': len(a['runtime']['ot_domains']['domains']),
            'accepted': a['runtime']['accepted'], 'ambiguity': a['runtime']['ambiguity_audit']['material_ambiguity'],
            'T_to_U_ids': a['runtime']['truncation_interval_guard']['demoted_T_to_U_ids'],
            'interval_model_binding_exact': a['runtime']['truncation_model_binding'] == b['runtime']['truncation_model_binding']})
    if not any(c['T_to_U_ids'] for c in checks): raise RuntimeError('Required actual T_to_U path absent')
    passed = all(all(c[k] for k in ('normalized_entire_runtime_exact', 'provider_Tiny_SAM_masks_prompts_quality_bindings_exact',
        'both_S_zero_contracts_pass', 'only_expected_raw_carrier_hash_changes', 'interval_model_binding_exact')) for c in checks)
    C.write(output / 'AUDIT.json', {'schema': 'REAL_CAL_S_INVARIANCE_AUDIT_V4_2', 'passed': passed, 'pairs': checks,
        'actual_provider_calls': len(trace.calls), 'Tiny_fullframe_calls': len(trace.calls),
        'MobileSAM_forward_calls': sum(c['actual_provider_provenance']['mask_forward_calls'] for c in trace.calls),
        'prediction_seal_sha256': C.sha(output / 'PREDICTION_SEAL.json'), 'GT_coordinates_or_pose_lookup_performed': False, 'CAL_acceptance_receipt_checked': True,
        'S_scope': 'fixed initial geometric S; not proof that all actual human SELF were correctly detected',
        'method_performance_or_target_completion_claim': False})
    print(json.dumps({'audit_passed': passed, 'output': str(output), 'pairs': checks}), flush=True)
    I.load_interval_configuration()
    C.write(output / 'COMPLETION_RECEIPT.json', {'source_sha256': source, 'input_sha256': inputs,
        'prediction_seal_sha256': C.sha(output / 'PREDICTION_SEAL.json'), 'audit_sha256': C.sha(output / 'AUDIT.json'),
        'actual_RGB_full_API_calls': 8, 'TEST_files_read': False, 'parameter_changes': 0, 'passed': passed})
    if not passed:
        raise RuntimeError('Semantic invariance audit failed; sealed outputs preserved')


if __name__ == '__main__':
    main()
