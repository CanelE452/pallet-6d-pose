"""Replay both saved-output arithmetic checks using only copied package leaves."""
import argparse
import hashlib
import json
from pathlib import Path
import types


def safe(p):
    p = Path(p).absolute()
    assert '..' not in p.parts
    assert not any(s in str(p).casefold() for s in ('v112', '__pycache__', '.pyc'))
    for a in reversed((p, *p.parents)):
        assert not a.is_symlink()
    return p


def verify(root):
    root = safe(root)
    tasks = [
        ('own_branch', 'artifacts/V244_SAVED_OUTPUT_QA_C/verify_saved_branches.py',
         '2a0381d04c6ead62d2e8a995402146d06afe028038c346b55bc5759cc895697a',
         'artifacts/V244_OWN_BRANCH_DIAGNOSTIC_A/actual_diagnosis_v1'),
        ('old_pairs', 'artifacts/V244_SAVED_OUTPUT_QA_C/pair_review_v1/verify_saved_pairs.py',
         'b30b7a49cc764b62cac7d500da8a553853cf4cbfbc511c6eb157123cce0dd3c8',
         'artifacts/V244_ROOT_FROZEN_BEHAVIOR/v235_pair_replay_v1')]
    results = {}
    for name, code, digest, data in tasks:
        path = safe(root / code)
        body = path.read_bytes()
        assert hashlib.sha256(body).hexdigest() == digest
        module = types.ModuleType('v244_portable_' + name)
        module.__file__ = str(path)
        exec(compile(body, str(path), 'exec'), module.__dict__)
        value = module.verify(str(safe(root / data)))
        assert value['completed'] and value['PASS']
        results[name] = {'PASS': True, 'checker_sha256': digest}
    return {'completed': True, 'PASS': True, 'checks': results,
            'CP_features_encoder_images_GT_or_original_pointer_follow': 0,
            'new_training_or_model_choice': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', default=str(Path(__file__).absolute().parent))
    parser.add_argument('--output-json')
    args = parser.parse_args()
    result = verify(args.package)
    if args.output_json:
        with safe(args.output_json).open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
    print(json.dumps(result))
