"""Replay sealed invented V241 tests from a package in a temporary project.

Run with python -B. No original SOURCE pointers/environment overrides are used
for input discovery. Only seven literal package files are read. No permanent
output, SOURCE data, model, GT, training or frozen-summary modification.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

# package path, SHA256, size, temporary-project relative destination.
FILES = (
 ('artifacts/V241_SCENE_CF_PRODUCER_B/scene_cf_math.py',
  '77b88de4311ff38c3baa480feb179b15a6e6ad8d9ea409701b98c6794fbd92c8', 8844,
  'project/publication_root_preparation/V241_SCENE_CF_PRODUCER_B/scene_cf_math.py'),
 ('artifacts/V241_SCENE_CF_PRODUCER_B/test_scene_cf_math.py',
  'cc0500eeaa33baa638c35b6945990a2b16ad98d6c428863b3c650511924761ff', 6866,
  'project/publication_root_preparation/V241_SCENE_CF_PRODUCER_B/test_scene_cf_math.py'),
 ('artifacts/V241_SCENE_CF_PRODUCER_B/INVENTED_SCENE_CHECKS.json',
  'b0aa93d77e068acae2118d6755307dce69ad07a6c3c0e28fe5cb3a8818d59204', 605,
  'expected/INVENTED_SCENE_CHECKS.json'),
 ('dependencies/project/query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py',
  '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b', 4715,
  'project/query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'),
 ('dependencies/project/query_visibility_target_context_rgb_20261023_v107/target_context_math.py',
  'aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3', 2204,
  'project/query_visibility_target_context_rgb_20261023_v107/target_context_math.py'),
 ('dependencies/preparation/V235_CF_COMPOSITOR_A/compositor.py',
  'd940ff8cb901482a77761dfac638ab473a5841d682ef0150f1ece0ab8c5fe4f4', 9160,
  'project/publication_root_preparation/V235_CF_COMPOSITOR_A/compositor.py'),
 ('dependencies/preparation/V237_CONTEXT_COMPOSITION_A/context_scene_math.py',
  'f48545c907cf0705a1cd9ee408e54f520f7eef4563fb166f20a1821b3fa9b30f', 7813,
  'project/publication_root_preparation/V237_CONTEXT_COMPOSITION_A/context_scene_math.py'),
)


def lexical(value):
    text = str(value)
    if any(bad in text.casefold() for bad in ('v112', '__pycache__', '.pyc')):
        raise ValueError('Forbidden lexical path before filesystem access')
    path = Path(text)
    if '..' in path.parts:
        raise ValueError('Parent traversal forbidden')
    return Path(os.path.abspath(text))


def safe(value):
    path = lexical(value)
    for part in reversed((path, *path.parents)):
        if part.is_symlink():
            raise ValueError('Symlink path forbidden')
    return path


def read(value):
    with os.fdopen(os.open(safe(value), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        return stream.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    args = parser.parse_args()
    if not sys.dont_write_bytecode:
        raise RuntimeError('Use python -B to disable bytecode output')
    package = safe(args.package)
    checked = {}
    for relative, expected_sha, expected_size, destination in FILES:
        data = read(package / relative)
        if len(data) != expected_size or hashlib.sha256(data).hexdigest() != expected_sha:
            raise ValueError('Sealed package bytes differ: ' + relative)
        checked[destination] = data
    expected = json.loads(checked['expected/INVENTED_SCENE_CHECKS.json'])
    with tempfile.TemporaryDirectory(prefix='v241-invented-replay-', dir=safe('/tmp')) as temporary:
        root = safe(temporary)
        for destination, data in checked.items():
            target = lexical(root / destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            target = safe(target)
            with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as stream:
                stream.write(data)
        work = root / 'project/publication_root_preparation/V241_SCENE_CF_PRODUCER_B'
        test = work / 'test_scene_cf_math.py'
        # Exact original test writes its summary only inside the temporary tree.
        # The frozen package summary is copied separately under expected/.
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(safe(test)), run_name='__main__')
        generated = json.loads(read(work / 'INVENTED_SCENE_CHECKS.json'))
        if generated != expected:
            raise AssertionError('Generated invented summary differs from sealed expected JSON')
        if generated.get('completed') is not True or generated.get('PASS') is not True:
            raise AssertionError('Invented test did not report PASS')
    print(json.dumps({'completed': True, 'PASS': True,
        'sealed_package_files_verified': len(FILES),
        'exact_original_test_replayed': True,
        'generated_summary_equals_published_JSON': True,
        'invented_scene_groups': expected['invented_scene_groups'],
        'permanent_outputs': 0, 'actual_SOURCE_or_models_or_training': 0}))


if __name__ == '__main__':
    main()
