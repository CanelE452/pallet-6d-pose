"""Saved SOURCE-only table; no model, RGB, certificate-array or evaluation reader."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def checked(binding):
    p = Path(binding['path'])
    if not p.is_absolute() or '..' in p.parts or any(x in str(p).casefold() for x in ('v112', '__pycache__', '.pyc')):
        raise ValueError('Forbidden explicit bound path')
    if any(x.is_symlink() for x in [*p.parents, p]):
        raise ValueError('Symlink bound artifact')
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('Changed binding')
    if 'size_bytes' in binding and len(raw) != binding['size_bytes']:
        raise ValueError('Changed size')
    return raw


def main(scope_path, out_path):
    sp = Path(scope_path)
    # The scope itself is supplied explicitly by Root, never discovered from data.
    if any(x in str(sp).casefold() for x in ('v112', '__pycache__', '.pyc')) or any(x.is_symlink() for x in [*sp.parents, sp]):
        raise ValueError('Unsafe scope')
    scope = json.loads(sp.read_bytes())
    assert scope['approved'] is True
    complete = json.loads(checked(scope['prediction_freeze']))
    assert complete['PASS'] is True and complete['model_state_byte_unchanged'] is True
    assert complete['all_rows'] == scope['all_rows']
    predicted = [json.loads(x) for x in checked(scope['all_rows']).splitlines()]
    truth = [json.loads(x) for x in checked(scope['saved_TRAIN_conditional_truth']).splitlines()]
    assert len(predicted) == len(truth) == 352
    names = ['conditional_V', 'conditional_E', 'UNKNOWN']
    result = {}
    for truth_key, support_key in [('exact_state', 'exact_mask_support'), ('radius2_state', 'radius2_mask_support')]:
        matrix = [[0, 0, 0] for _ in range(3)]
        for p, t in zip(predicted, truth):
            assert p['original_input_unchanged'] == t['original_input_unchanged']
            assert p['native_visibility_label'] is None and t['native_visibility_label'] is None
            assert p['semantic_SELF'] is None and p['training_admission'] is False
            state = t['conditional_pixel_truth'].get(truth_key, -1)
            assert state in (-1, 0, 1)  # No new O or SELF truth is admitted.
            row = 2 if state == -1 else state
            val = p[support_key]
            assert val is None or type(val) is bool
            col = 2 if val is None else (0 if val else 1)
            assert (val is None) == (not p['prediction_available'])
            matrix[row][col] += 1
        result[truth_key.removesuffix('_state')] = {
            'rows': names, 'columns': ['SAM_MASK_SUPPORT', 'OUTSIDE_SAM_PROPOSAL', 'PREDICTION_U'],
            'counts': matrix, 'row_totals': list(map(sum, matrix)),
        }
    output = {
        'schema': 'V247_ROOT_SAVED_CONDITIONAL_PIXEL_TABLES_V1',
        'completed': True, 'PASS': True, 'queries': 352,
        'prediction_freeze': scope['prediction_freeze'],
        'all_rows': scope['all_rows'], 'saved_TRAIN_conditional_truth': scope['saved_TRAIN_conditional_truth'],
        'tables': result, 'model_calls': complete['calls'],
        'interpretation': 'Conditional pixel-certificate strata against SAM segmentation support; not native-keypoint correctness/visibility accuracy or confirmed instance correspondence.',
        'native_labels_created': 0, 'new_training_updates': 0, 'PnP': 0,
        'current_or_held_payloads': 0, 'model_adopted': False, 'training_admission': False,
    }
    out = Path(out_path)
    if not out.is_absolute() or '..' in out.parts or any(x in str(out).casefold() for x in ('v112', '__pycache__', '.pyc')) or any(x.is_symlink() for x in [*out.parents, out]):
        raise ValueError('Unsafe output')
    with os.fdopen(os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'PASS': True, 'tables': result, 'output': str(out)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    main(args.scope, args.out)
