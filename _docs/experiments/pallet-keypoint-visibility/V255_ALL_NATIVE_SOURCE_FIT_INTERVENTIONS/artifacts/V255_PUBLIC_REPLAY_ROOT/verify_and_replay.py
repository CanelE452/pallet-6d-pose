"""Verify all packaged artifacts and optionally replay the first sealed FIT base.

Original absolute paths in saved metadata are inert. Only relative package
bindings are followed. Replay requires NumPy 2.3.5, the original render runtime.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import stat
import sys
import types


def safe(path):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts or any(
            x in str(p).casefold() for x in ('v112', 'cache', '.pyc')):
        raise ValueError('Forbidden artifact path')
    for a in [*reversed(p.parents), p]:
        mode = os.lstat(a).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError('Artifact symlink')
    return p


def main(package, replay=False):
    package = safe(Path(package).absolute())

    def raw(rel):
        p = Path(rel)
        assert not p.is_absolute() and '..' not in p.parts
        return safe(package / p).read_bytes()

    manifest = json.loads(raw('MANIFEST.json'))
    assert manifest['completed'] and manifest['PASS']
    files = {r['relative_path']: r for r in manifest['files']}
    assert len(files) == len(manifest['files'])

    def bound(binding):
        rel = binding['relative_path']
        assert rel in files
        b = raw(rel)
        assert len(b) == binding['size_bytes']
        assert hashlib.sha256(b).hexdigest() == binding['sha256']
        return b

    for r in manifest['files']:
        bound(r)
    result = {'completed': True, 'PASS': True,
              'package_manifest_sha256': hashlib.sha256(raw('MANIFEST.json')).hexdigest(),
              'all_artifact_bindings_verified': len(files),
              'original_absolute_path_reads': 0,
              'new_model_training_or_forward': 0, 'replayed_base': None}
    if not replay:
        return result
    import numpy as np
    assert np.__version__ == '2.3.5', 'Exact original NumPy required for pixel replay'
    deps = json.loads(raw('runtime_dependencies.json'))

    def module(binding, name):
        code = bound(binding)
        m = types.ModuleType(name)
        m.__file__ = str(package / binding['relative_path'])
        m.__package__ = None
        sys.modules[name] = m
        exec(compile(code, m.__file__, 'exec'), m.__dict__)
        return m

    sys.dont_write_bytecode = True
    producer = module(deps['generator'], 'v255_public_generator')
    modules = {k: module(b, 'v255_public_' + k)
               for k, b in deps['frozen_codes'].items()}
    configs = json.loads(bound(deps['selected_configs']))['rows']
    cfg = min(configs, key=lambda c: int(c['base_index']))
    scene_rows = [json.loads(line) for line in raw('dataset/SCENES.jsonl').splitlines()]
    rows = [r for r in scene_rows if r['base_index'] == cfg['base_index']]
    by_key = {(r['variant'], r['target_native_slot']): r for r in rows}
    assert len(rows) == len(by_key)

    def metadata(row):
        rel = 'dataset/scenes/' + row['scene_id'] + '.json'
        assert files[rel]['sha256'] == row['metadata']['sha256']
        return json.loads(raw(rel))

    before = metadata(by_key[('BEFORE', None)])['original_before_input']
    mesh_binding = deps['meshes'][int(cfg['template_index'])]
    with np.load(io.BytesIO(bound(mesh_binding)), allow_pickle=False) as z:
        mesh = {key: z[key].copy() for key in producer.MESH_KEYS}
    emitted = set()

    def emit(meta, rgb):
        key = (meta['variant'], meta['target_native_slot'])
        assert key in by_key and key not in emitted
        emitted.add(key)
        row = by_key[key]
        saved = metadata(row)
        assert all(saved[k] == value for k, value in meta.items()), key
        pixel_rel = 'dataset/scenes/' + row['scene_id'] + '.npz'
        assert files[pixel_rel]['sha256'] == row['image']['sha256']
        with np.load(io.BytesIO(raw(pixel_rel)), allow_pickle=False) as z:
            assert z.files == ['RGB_uint8_HWC']
            assert np.array_equal(rgb, z['RGB_uint8_HWC']), key
        assert hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest() == row['RGB_sha256_HWC']

    replayed = producer.render_base_interventions(mesh, cfg, before, modules, emit)
    assert emitted == set(by_key)
    trials = [json.loads(line) for line in raw('dataset/ALL_NATIVE_TRIALS.jsonl').splitlines()]
    trials = [r for r in trials if r['base_index'] == cfg['base_index']]
    assert trials == replayed['trials']
    result.update(replayed_base=cfg['base_id'],
                  RGB_and_fresh_teacher_scenes_exact=len(emitted),
                  all_native_trials_exact=len(trials),
                  NumPy=np.__version__,
                  replay_limit='First minimum-index SourceFIT base, not all 86 full RGB rerenders.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--replay-first-base', action='store_true')
    args = p.parse_args()
    print(json.dumps(main(args.package, args.replay_first_base), ensure_ascii=False, indent=2))
