"""Copy an explicit V247 SOURCE evidence closure; never follow dataset pointers."""
import hashlib
import json
import os
from pathlib import Path

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'publication_root_preparation'
OUT = R / 'V247_PUBLICATION_PACKAGE'
CAP = 95_000_000


def guard(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(s in str(p).casefold() for s in ('v112', '__pycache__', '.pyc'))
    assert not any(x.is_symlink() for x in [*p.parents, p])
    return p


def data(p):
    with os.fdopen(os.open(guard(p), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        b = f.read(CAP)
        assert not f.read(1) and len(b) < CAP
    return b


def main():
    root = R / 'V247_ROOT_PUBLIC_TRAIN_SAM'
    scope = json.loads(data(root / 'READ_SCOPE_BEFORE_V247_PUBLICATION_COPY.json'))
    assert scope['approved'] is True and scope['package_root'] == str(OUT)
    assert not guard(OUT).exists()
    OUT.mkdir()
    leaves, source_map = {}, []

    def put(relative, b):
        part = Path(relative)
        assert not part.is_absolute() and '..' not in part.parts and relative != 'MANIFEST.json'
        dest = guard(OUT / relative)
        assert len(b) < CAP
        if relative in leaves:
            assert data(dest) == b
            return
        assert not dest.exists()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f:
            f.write(b)
        leaves[relative] = {'path': relative, 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}

    def copy(p, relative, binding=None):
        p = guard(p)
        b = data(p)
        digest = hashlib.sha256(b).hexdigest()
        if binding:
            assert binding['path'] == str(p) and binding['sha256'] == digest
            assert binding.get('size_bytes', binding.get('bytes', len(b))) == len(b)
        put(relative, b)
        source_map.append({'path': relative, 'encoding': 'identity', 'source_path_provenance_only': str(p), 'source_sha256': digest, 'source_bytes': len(b)})

    def owned_firstlevel(folder, suffixes, target=None):
        folder = guard(folder)
        for p in sorted(folder.iterdir(), key=lambda x: x.name):
            # Names-only discovery inside a newly owned folder. Reject before stat/read.
            if any(s in p.name.casefold() for s in ('v112', '__pycache__', '.pyc')):
                continue
            if p.suffix.lower() not in suffixes:
                continue
            relative = str(Path(target or folder.name) / p.name)
            copy(p, relative)

    for name in scope['owned_folders']:
        owned_firstlevel(R / name, {'.py', '.json', '.md', '.txt', '.csv'})
    visual = R / 'V247_FIXED4_SAM_VISUALS_A/gallery_v1'
    owned_firstlevel(visual, {'.py', '.json', '.png', '.html', '.csv'}, 'V247_FIXED4_SAM_VISUALS_A/gallery_v1')

    complete = json.loads(data(root / 'frozen_source_sam_v1/COMPLETE.json'))
    assert complete['PASS'] is True
    for name in ['STARTED.json', 'COMPLETE.json']:
        copy(root / 'frozen_source_sam_v1' / name, 'V247_ROOT_PUBLIC_TRAIN_SAM/frozen_source_sam_v1/' + name)
    for binding in [complete['all_rows'], complete['query_CSV']]:
        p = Path(binding['path'])
        assert p.parent == root / 'frozen_source_sam_v1'
        copy(p, 'V247_ROOT_PUBLIC_TRAIN_SAM/frozen_source_sam_v1/' + p.name, binding)
    masks = [x['output'] for x in complete['frame_outputs'] if x['output'] is not None]
    assert len(masks) == 42
    for binding in masks:
        p = Path(binding['path'])
        assert p.parent == root / 'frozen_source_sam_v1/masks' and p.suffix == '.npz'
        copy(p, 'V247_ROOT_PUBLIC_TRAIN_SAM/frozen_source_sam_v1/masks/' + p.name, binding)

    entry = json.loads(data(root / 'ACTUAL_INDEPENDENT_SAVED_SAM_QA_ENTRY.json'))
    assert entry['approved_saved_SOURCE_SAM_QA'] is True
    dependencies = {
        'TRAIN_INPUT_METADATA_COMPLETE.json': entry['train_complete'],
        'TRAIN_INPUT_METADATA_QA.json': entry['train_metadata_QA'],
        'TRAIN352_SAME_Q_CONDITIONAL_PIXEL_TRUTH.jsonl': entry['conditional_truth'],
        'TRAIN44_FRAMES_INPUT_ONLY.jsonl': complete['inputs']['frames'],
        'TRAIN352_ORIGINAL_NATIVE_QUERIES_INPUT_ONLY.jsonl': complete['inputs']['queries'],
    }
    for name, binding in dependencies.items():
        copy(Path(binding['path']), 'DEPENDENCIES/' + name, binding)
    for name, p in {
        'RUNTIME_SEAL.json': P / 'foundation_runtime_sam21_20261023/RUNTIME_SEAL.json',
        'INTEGRATION.json': P / 'foundation_runtime_sam21_20261023/INTEGRATION.json',
        'ORIGINAL87_CODE_MANIFEST.json': P / 'foundation_runtime_sam21_20261023/provenance/SAM2_REPO_MANIFEST.json',
        'V68_extract_source_masks.py': P / 'query_visibility_sam21_frozen_20261023_v68/extract_source_masks.py',
        'V180_run_frozen_sam_source.py': P / 'query_visibility_frozen_sam_source_baseline_20261024_v180/runner_preparation/run_frozen_sam_source.py',
    }.items():
        copy(p, 'CODE_METADATA_DEPENDENCIES/' + name)
    copy(R / 'V246_ROOT_ACTUAL_MAIN_PUSH_RECEIPT.json', 'PREVIOUS_MAIN_RECEIPT.json')
    copy(R / 'verify_v247_published_sam.py', 'verify_v247_published_sam.py')
    copy(Path(__file__).absolute(), 'build_v247_publication.py')
    copy(root / 'PUBLIC_README_KO.md', 'README_KO.md')
    copy(root / 'PUBLIC_STATUS.json', 'STATUS.json')
    mapping = {'schema': 'V247_PUBLICATION_IDENTITY_SOURCE_MAP_V1', 'files': source_map, 'original_paths_are_provenance_only': True, 'original_source_fallback_allowed': False}
    put('PUBLICATION_SOURCE_MAP.json', (json.dumps(mapping, ensure_ascii=False, indent=2) + '\n').encode())
    status = json.loads(data(root / 'PUBLIC_STATUS.json'))
    manifest = {'schema': 'V247_FROZEN_SAM_PUBLIC_TRAIN_DIAGNOSTIC_PUBLICATION', 'scientific_status': status, 'files': list(leaves.values()), 'saved_chosen_masks': 42, 'SOURCE_frames': 44, 'original_queries': 352, 'undeclared_original_dataset_body_pointers_followed': False, 'model_weights_or_original_mixed_rows_copied': False, 'original_paths_are_inert': True}
    mb = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    with guard(OUT / 'MANIFEST.json').open('xb') as f:
        f.write(mb)
    print(json.dumps({'completed': True, 'PASS': True, 'package_root': str(OUT), 'manifest_sha256': hashlib.sha256(mb).hexdigest(), 'leaves': len(leaves), 'leaf_bytes': sum(x['bytes'] for x in leaves.values()), 'saved_chosen_masks': 42, 'identity_rows': len(source_map)}))


if __name__ == '__main__':
    main()
