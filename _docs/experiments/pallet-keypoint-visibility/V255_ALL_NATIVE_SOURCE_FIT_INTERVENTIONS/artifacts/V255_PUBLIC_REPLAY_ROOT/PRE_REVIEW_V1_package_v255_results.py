"""Publish the entire sealed native-intervention dataset, not selected examples."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import stat

R = Path(__file__).absolute().parent
H = R / 'V255_ROOT_NATIVE_COVERAGE'
P = R / 'V255_PUBLICATION_PACKAGE'
TREE = {
    'V255_ROOT_NATIVE_COVERAGE': {'source_fit_parent_manifest_v1': {}},
    'V255_NATIVE_BALANCED_PRODUCER_A': {},
    'V255_SOURCE_FIT_MANIFEST_B': {},
    'V255_REFERENCE_PRESERVATION_B': {},
    'V255_NEXT_INTERVENTION_QA_C': {
        'actual_review_v1': {}, 'actual_native_review_v1': {}},
    'V255_NATIVE_VISUAL_QA_A': {'gallery_v1': {}},
    'V255_PUBLIC_PALLET_METADATA_ROOT': {},
    'V255_PUBLIC_REPLAY_ROOT': {},
}


def safe(path):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts or any(
            s in str(p).casefold() for s in ('v112', 'cache', '.pyc')):
        raise ValueError(str(p))
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError('Artifact symlink')
    return p


def read(path):
    return safe(path).read_bytes()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def obj(path):
    return json.loads(read(path))


def verify(pkg):
    pkg = safe(pkg)
    manifest = obj(pkg / 'MANIFEST.json')
    assert manifest['completed'] and manifest['PASS']
    for row in manifest['files']:
        rel = Path(row['relative_path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        raw = read(pkg / rel)
        assert len(raw) == row['size_bytes'] and sha(raw) == row['sha256']
    print(json.dumps({'completed': True, 'PASS': True,
                      'files': len(manifest['files']),
                      'bytes': sum(r['size_bytes'] for r in manifest['files'])}))


def build():
    complete = obj(H / 'native_balanced_scenes_v1/COMPLETE.json')
    assert complete['PASS'] and complete['completed']
    qa = obj(R / 'V255_NEXT_INTERVENTION_QA_C/actual_native_review_v1/REVIEW.json')
    assert qa['PASS'] and qa['completed']
    assert not safe(P).exists()
    P.mkdir()
    rows = []
    seen = set()

    def take(src, rel, expected=None):
        assert rel not in seen, rel
        seen.add(rel)
        raw = read(src)
        if expected is not None:
            assert sha(raw) == expected['sha256']
            assert len(raw) == expected['size_bytes']
        target = safe(P / rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(raw)
        rows.append({'relative_path': rel, 'sha256': sha(raw),
                     'size_bytes': len(raw), 'original_path': str(src)})
        return {'relative_path': rel, 'sha256': sha(raw),
                'size_bytes': len(raw)}

    def direct(directory, prefix, allowed_children):
        for name in sorted(os.listdir(safe(directory))):
            path = safe(directory / name)
            if path.is_dir():
                if directory == H and name == 'native_balanced_scenes_v1':
                    continue  # Read only the exact immutable manifest below.
                assert name in allowed_children, str(path)
                direct(path, prefix + '/' + name, allowed_children[name])
            else:
                assert path.is_file()
                assert path.suffix in ('.json', '.jsonl', '.py', '.md',
                                       '.csv', '.png', '.html', '.npz')
                take(path, prefix + '/' + name)

    for name, children in TREE.items():
        direct(R / name, 'artifacts/' + name, children)
    dataset_root = H / 'native_balanced_scenes_v1'
    manifest_binding = complete['outputs']['MANIFEST.json']
    manifest_raw = read(manifest_binding['path'])
    assert sha(manifest_raw) == manifest_binding['sha256']
    assert len(manifest_raw) == manifest_binding['size_bytes']
    data_manifest = json.loads(manifest_raw)
    for binding in data_manifest['files']:
        source = Path(binding['path'])
        rel = source.relative_to(dataset_root)
        take(source, 'dataset/' + rel.as_posix(), binding)
    take(dataset_root / 'MANIFEST.json', 'dataset/MANIFEST.json', manifest_binding)
    take(dataset_root / 'COMPLETE.json', 'dataset/COMPLETE.json')
    deps = {'schema': 'V255_PORTABLE_RENDER_REPLAY_INPUTS_V1',
            'frozen_codes': {}, 'meshes': []}
    for key, binding in complete['code_bindings'].items():
        deps['frozen_codes'][key] = take(
            Path(binding['path']), 'frozen_code/' + key + '.py', binding)
    for i, binding in enumerate(complete['mesh_payload_bindings']):
        deps['meshes'].append(take(
            Path(binding['path']), 'geometry/template_' + str(i) + '.npz', binding))
    for key, filename in (
            ('selected_configs', 'SELECTED_FIT_CONFIGS.json'),
            ('before_inputs', 'SOURCE_FIT_PARENT_MANIFEST.json')):
        rel = 'artifacts/V255_ROOT_NATIVE_COVERAGE/source_fit_parent_manifest_v1/' + filename
        binding = next(x for x in rows if x['relative_path'] == rel)
        deps[key] = {k: binding[k] for k in ('relative_path', 'sha256', 'size_bytes')}
    deps['generator'] = next({k: x[k] for k in ('relative_path', 'sha256', 'size_bytes')}
                             for x in rows if x['relative_path'] ==
                             'artifacts/V255_NATIVE_BALANCED_PRODUCER_A/build_native_balanced_scenes.py')
    raw = (json.dumps(deps, ensure_ascii=False, indent=2) + '\n').encode()
    q = safe(P / 'runtime_dependencies.json')
    q.write_bytes(raw)
    rows.append({'relative_path': q.name, 'sha256': sha(raw), 'size_bytes': len(raw)})
    for name in ('README_KO.md', 'ROOT_READOUT_KO.html'):
        take(H / ('PUBLIC_' + name if name == 'README_KO.md' else name), name)
    for name in ('package_v255_results.py', 'stage_v255_main.py'):
        take(R / name, 'tools/' + name)
    manifest = {'schema': 'V255_ALL_NATIVE_SOURCE_FIT_RENDER_PUBLICATION_V1',
                'completed': True, 'PASS': True, 'files': rows,
                'all_generated_RGB_and_metadata_included': True,
                'dataset_counts': complete['counts'],
                'new_model_training_or_forward': 0,
                'model_improvement_proven': False,
                'current_or_held_payload_access': False,
                'goal_status': 'active_unsolved',
                'serialized_original_paths': 'Inert provenance; portable replay uses relative bindings only.'}
    (P / 'MANIFEST.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    verify(P)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--build', action='store_true')
    group.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.build:
        build()
    else:
        verify(args.verify)
