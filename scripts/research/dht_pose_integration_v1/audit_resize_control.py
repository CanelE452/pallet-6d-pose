"""CPU-only audit of actual source preprocessing and the GT-free resize control."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from resize_control import correct_legacy_resize_coordinates, resize_control_parameters

ROOT = Path(__file__).resolve().parents[3]
PREPROCESS = ROOT / 'scripts/stage0/eval_harness/eval_pvnet_heads.py'
CANVAS = ROOT / 'data/pallet/eval_results/stage16_truncation_addon/capturecad_b2_eval/eval_capturecad_b2.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_functions(path, names, namespace):
    """Run only named pure source functions/constants, without module side effects."""
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    if not all(name in namespace for name in names):
        raise RuntimeError(f'Missing requested source functions/constants in {path}')


def audit(run_dir):
    run_dir = Path(run_dir)
    namespace = {'np': np, 'cv2': cv2, 'torch': torch}
    source_functions(PREPROCESS, ['preprocess', 'MEAN', 'STD'], namespace)
    source_functions(CANVAS, ['pad_frame', 'belief_to_orig_pad'], namespace)
    rng, maximum_error, cases = np.random.default_rng(2909), 0.0, []
    for width, height in ((640, 480), (960, 540), (720, 480), (560, 560),
                          (480, 640), (777, 431)):
        # Generated zero pixels only. No dataset images/GT/predicted accuracy.
        dummy = np.zeros((height, width, 3), np.uint8)
        padded_canvas = namespace['pad_frame'](dummy, 100)
        tensor, nw, nh, sc = namespace['preprocess'](padded_canvas)
        params = resize_control_parameters(width, height, tensor.shape[1:])
        assert tuple(tensor.shape[1:]) == (3, nh, nw)
        bw, bh, case_error = nw // 8, nh // 8, 0.0
        for _ in range(32):
            belief = rng.uniform(-10, 80, (9, 2)) + 0.4395
            legacy = np.array([namespace['belief_to_orig_pad'](
                bx, by, bw, bh, nw, nh, sc, 100, width, height) for bx, by in belief])
            corrected, mask = correct_legacy_resize_coordinates(
                legacy, np.ones(9, bool), width, height, tensor.shape[1:])
            network_points = belief * [nw / bw, nh / bh]
            direct_inverse = network_points * [width / nw, height / nh]
            direct_inverse = direct_inverse * [(width + 200) / width,
                                               (height + 200) / height] - 100
            case_error = max(case_error, float(np.abs(corrected - direct_inverse).max()))
            assert mask.all()
        maximum_error = max(maximum_error, case_error)
        cases.append({'width': width, 'height': height, 'input_shape_chw': [3, nh, nw],
                      'scale_xy': params['scale_xy'].tolist(),
                      'translation_xy': params['translation_xy'].tolist(),
                      'n_arbitrary_points': 32 * 9,
                      'max_roundtrip_error_px': case_error})
    if maximum_error > 1e-9:
        raise AssertionError(f'Resize inverse mismatch: {maximum_error}')
    baseline_path = run_dir / 'BASELINE_DOPE.json'
    baseline = json.loads(baseline_path.read_text())
    if baseline.get('complete') is not True:
        raise ValueError('DOPE baseline metadata is incomplete')
    shapes = Counter()
    for record in baseline['records']:
        resize_control_parameters(record['width'], record['height'], record['input_shape_chw'])
        shapes[(record['width'], record['height'], *record['input_shape_chw'])] += 1
    report = {
        'schema': 'dope_actual_resize_geometry_audit_v1', 'status': 'PASS',
        'audit_only': 'GT-free source geometry and shape check, no accuracy evaluation',
        'source_function_execution': 'AST-extracted actual preprocess, MEAN, STD, pad_frame, belief_to_orig_pad; avoids unrelated module side effects',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (
            PREPROCESS, CANVAS, Path(__file__), Path(__file__).with_name('resize_control.py'),
            Path(__file__).with_name('test_resize_control.py'),
            Path(__file__).with_name('DOPE_RESIZE_DERIVATION.md'))},
        'baseline_sha256': sha(baseline_path), 'baseline_metadata_records_checked': len(baseline['records']),
        'baseline_shape_counts': [{'width': k[0], 'height': k[1], 'input_shape_chw': list(k[2:]), 'count': v}
                                  for k, v in sorted(shapes.items())],
        'cases': cases, 'maximum_roundtrip_error_px': maximum_error,
        'total_arbitrary_points_checked': sum(c['n_arbitrary_points'] for c in cases),
        'affine_640x480': {'scale_xy': [100 / 99, 1], 'translation_xy': [100 / 99, 0],
                          'example_legacy_x': [0, 320, 640],
                          'example_delta_x_px': [(x + 100) / 99 for x in (0, 320, 640)]},
        'preserved': ['semantic corner IDs', 'confidence rule', 'nonfinite missing points',
                      'canonical peak decoder and +0.4395', 'padding', 'existing pixel-center convention'],
        'center': 'center is coordinate-corrected along with corners; fusion preserves that corrected center',
        'limit': 'Exact inverse of actual resize sizes within legacy decoder convention; not a new half-pixel or receptive-field correction',
    }
    output = run_dir / 'DOPE_RESIZE_AUDIT.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'output': str(output.resolve()),
                      'maximum_roundtrip_error_px': maximum_error,
                      'baseline_metadata_records_checked': len(baseline['records'])}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    audit(parser.parse_args().run_dir)
