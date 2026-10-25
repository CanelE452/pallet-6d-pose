"""Bind named completed artifacts for the lossless V256 publication builder."""
from pathlib import Path
import collections
import hashlib
import json
import os
import stat

R = Path(__file__).absolute().parent
H = R / 'V256_ROOT_NATIVE_DATA_FIT'
TREE = {
    'V255_NEW_CANONICAL_ADAPTER_B': {},
    'V255_NEW_SCENE_CANONICAL_DRIVER_B': {},
    'V255_ROOT_NATIVE_FEATURES': {},
    'V255_FEATURE_TARGET_QA_C': {'actual_review_v1': {}},
    'V256_ROOT_NATIVE_DATA_FIT': {},
    'V256_MATCHED_DATA_TRAINER_A': {},
    'V256_TRAINER_CODE_REVIEW_B': {},
    'V256_ACTUAL_RESULT_QA_B': {'actual_review_v1': {}},
    'V256_RESULT_CHECKER_CODE_QA_A': {},
    'V256_PUBLICATION_PREP_A': {},
    'V256_PUBLICATION_CODE_REVIEW_B': {},
    'V256_SOURCE_PAIR_READOUT_C': {'publication_v2': {}},
}
SKIP_DIRS = {'new_source_canonical_features_v1',
             'matched_existing_crop_data_pair_v1', 'results_v1'}


def safe(path):
    p = Path(path)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for q in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(q).st_mode
        except FileNotFoundError:
            continue
        assert not stat.S_ISLNK(mode)
    return p


def bind(path):
    path = safe(path)
    s = hashlib.sha256()
    n = 0
    with path.open('rb') as f:
        while chunk := f.read(1048576):
            s.update(chunk)
            n += len(chunk)
    return {'path': str(path), 'sha256': s.hexdigest(), 'size_bytes': n}


def obj(path):
    return json.loads(safe(path).read_bytes())


def put(path, value):
    path = safe(path)
    assert not path.exists()
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return bind(path)


def main():
    assert bind(H / 'matched_existing_crop_data_pair_v1/COMPLETE.json')['sha256'] == 'bd2804063937d6ac43a197f2d592d92e610086985e4c3cf5d3bb7a908e105d1d'
    assert bind(R / 'V255_ROOT_NATIVE_FEATURES/new_source_canonical_features_v1/COMPLETE.json')['sha256'] == 'd83617666b00f70ed3205d3b352329f30ae253509fea6fe55aed4f6fcf3604ef'
    assert bind(R / 'V256_ACTUAL_RESULT_QA_B/actual_review_v1/REVIEW.json')['sha256'] == 'f3c6cb4bcd944a496461a4edc1754a0f0424e38a56545f33688225d21c49e635'
    fit = obj(H / 'matched_existing_crop_data_pair_v1/COMPLETE.json')
    features = obj(R / 'V255_ROOT_NATIVE_FEATURES/new_source_canonical_features_v1/COMPLETE.json')
    qa = obj(R / 'V256_ACTUAL_RESULT_QA_B/actual_review_v1/REVIEW.json')
    assert fit['completed'] and fit['PASS'] and features['completed'] and features['PASS']
    assert qa['completed'] and qa['PASS']
    files = []
    seen = set()

    def take(source, relative, group='attempts_code_protocols_QA', encoding='raw'):
        assert relative not in seen, relative
        seen.add(relative)
        if not isinstance(source, dict):
            source = bind(source)
        else:
            safe(source['path'])
        files.append({'source': source, 'relative_path': relative,
                      'group': group, 'encoding': encoding})

    def direct(directory, prefix, children):
        for name in sorted(os.listdir(safe(directory))):
            path = safe(directory / name)
            if path.is_dir():
                if name in SKIP_DIRS:
                    continue
                assert name in children, str(path)
                direct(path, prefix + '/' + name, children[name])
            else:
                assert path.is_file()
                assert path.suffix in ('.py', '.json', '.jsonl', '.md', '.csv', '.html', '.png')
                if name.startswith('ACTUAL_PUBLICATION_'):
                    continue  # These late scope/receipt files cannot hash themselves.
                take(path, prefix + '/' + name)

    for name, children in TREE.items():
        direct(R / name, 'artifacts/' + name, children)
    for name in ('STAGE3_CODE_REVIEW.json', 'ROOT_ENTRY_RECONCILIATION_REVIEW.json'):
        take(R / 'V255_NEXT_INTERVENTION_QA_C' / name,
             'artifacts/V255_NEXT_INTERVENTION_QA_C/' + name)
    for packet, root, prefix in (
            (features, R / 'V255_ROOT_NATIVE_FEATURES/new_source_canonical_features_v1', 'new_features'),
            (fit, H / 'matched_existing_crop_data_pair_v1', 'matched_fit')):
        mb = packet.get('manifest')
        assert bind(Path(mb['path'])) == mb
        manifest = obj(Path(mb['path']))
        for binding in manifest['files']:
            name = Path(binding['path']).relative_to(root).as_posix()
            group = 'labels_samples_allrows' if prefix == 'new_features' else 'paired_results_metrics'
            encoding = 'raw'
            if prefix == 'new_features':
                if name in ('NODE_RGB96.npy', 'NODE_VALID96.npy', 'CONTEXT_RGB224_F32.npy', 'CONTEXT_VALID224.npy'):
                    group, encoding = 'raw_views', 'gzip'
                elif name in features['features']:
                    group = 'canonical_features'
            else:
                for key, category in (('checkpoints', 'paired_checkpoints'),
                                      ('logs', 'paired_train_logs'), ('draw_ledgers', 'paired_draws')):
                    if binding in fit[key].values():
                        group = category
            take(binding, prefix + '/' + name, group, encoding)
        take(mb, prefix + '/MANIFEST.json')
        take(root / 'COMPLETE.json', prefix + '/COMPLETE.json')
    take(R / 'V255_ROOT_NATIVE_FEATURES/new_source_canonical_features_v1/STARTED.json',
         'new_features/STARTED.json')
    old_path = R / 'V239_ROOT_CANONICAL_CONTEXT/canonical_source_features_v1/COMPLETE.json'
    old = obj(old_path)
    take(old_path, 'original_source/COMPLETE.json')
    take(old['features']['ORIGINAL_ALL_FEATURES.npy'], 'original_source/ORIGINAL_ALL_FEATURES.npy')
    take(old['samples_npz'], 'original_source/samples.npz')
    take(R / 'V233_DETERMINISTIC_CROP_FIX_A/crop_modules.py', 'frozen_code/crop_modules.py')
    feature_entry = obj(R / 'V255_ROOT_NATIVE_FEATURES/ACTUAL_NEW_SOURCE_CANONICAL_FEATURE_ENTRY.json')
    for i, binding in enumerate(feature_entry['code_whitelist']):
        take(binding, 'frozen_code/extraction/' + str(i).zfill(2) + '_' + Path(binding['path']).name)
    public = R / 'V255_PUBLICATION_PACKAGE'
    for b in obj(public / 'MANIFEST.json')['files']:
        prefix = 'artifacts/V255_NATIVE_VISUAL_QA_A/gallery_v1/'
        if b['relative_path'].startswith(prefix):
            take({'path': str(public / b['relative_path']), 'sha256': b['sha256'], 'size_bytes': b['size_bytes']},
                 'references/V255_GALLERY/' + b['relative_path'][len(prefix):],
                 'visuals' if b['relative_path'].endswith('.png') else 'attempts_code_protocols_QA')
    report = R / 'V256_SOURCE_PAIR_READOUT_C/results_v1'
    take(H / 'PUBLIC_README_KO_FINAL.md', 'README_KO.md', 'reports')
    take(H / 'ROOT_READOUT_KO.html', 'ROOT_READOUT_KO.html', 'reports')
    for name in ('PORTABLE_REPORT_KO.md', 'ROOT_READOUT_KO.html'):
        take(report / name, 'artifacts/V256_SOURCE_PAIR_READOUT_C/results_v1/' + name)
    take(report / 'COMPLETE.json', 'artifacts/V256_SOURCE_PAIR_READOUT_C/results_v1/COMPLETE.json')
    for name in sorted(os.listdir(safe(report))):
        path = safe(report / name)
        if path.is_file() and name not in ('PORTABLE_REPORT_KO.md', 'ROOT_READOUT_KO.html'):
            take(path, 'tables/' + name, 'paired_results_metrics')
    for name in sorted(os.listdir(safe(report / 'tables'))):
        path = safe(report / 'tables' / name)
        assert path.is_file() and path.suffix in ('.csv', '.json', '.jsonl')
        take(path, 'tables/' + name, 'paired_results_metrics')
    for name in sorted(os.listdir(safe(report / 'figures'))):
        path = safe(report / 'figures' / name)
        assert path.is_file() and path.suffix == '.png'
        take(path, 'figures/' + name, 'visuals')
    take(R / 'prepare_v256_publication_entry.py', 'tools/prepare_v256_publication_entry.py')
    take(R / 'stage_v256_main.py', 'tools/stage_v256_main.py')
    entry = obj(R / 'V256_PUBLICATION_PREP_A/ENTRY_FALSE_TEMPLATE.json')
    groups = sorted({x['group'] for x in files})
    entry.update(approved_actual_publication_package=True, files=files,
                 coverage_contract={g: [x['source'] for x in files if x['group'] == g] for g in groups},
                 single_file_limit_bytes=90 * 1024 * 1024,
                 output_directory=str(R / 'V256_PUBLICATION_PACKAGE'))
    scope = {'approved_actual_publication_package': True,
             'entry_contract': {k: v for k, v in entry.items() if k != 'root_scope'},
             'all_completed_feature_and_fit_manifests_included': True,
             'all_new_attempts_in_named_directories_included': True,
             'protected_current_or_held_data': False, 'goal_status': 'active_unsolved'}
    entry['root_scope'] = put(H / 'ACTUAL_PUBLICATION_SCOPE.json', scope)
    eb = put(H / 'ACTUAL_PUBLICATION_ENTRY.json', entry)
    print(json.dumps({'entry': eb, 'files': len(files),
                      'coverage_counts': dict(collections.Counter(x['group'] for x in files))}))


if __name__ == '__main__':
    main()
