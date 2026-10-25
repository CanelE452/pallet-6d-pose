"""Root explicit-directory byte-only publication allowlist, V259."""
from pathlib import Path
import hashlib
import json

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
DIRECTORIES = (
    'V258_NATIVE_ADDRESS_DEFINITION_A',
    'V258_PUBLIC_HELD_DIGEST_IDENTITY_B',
    'V258_PUBLIC_RGBD_PROJECTION_C',
    'V258_ROOT_PUBLIC_POINT_TEACHER',
    'V258_ROOT_PUBLIC_POINT_TEACHER/public_0p55_hall_0000_v1',
    'V258_ROOT_PUBLIC_POINT_TEACHER/public_frame_geometry_v1',
    'V258_ROOT_PUBLIC_POINT_TEACHER/public_frame_geometry_v2',
    'V258_PUBLIC_VIRTUAL_QUERY_DIAGNOSIS_C',
    'V258_PUBLIC_VIRTUAL_QUERY_DIAGNOSIS_C/observations_v1',
    'V258_PUBLIC_VIRTUAL_QUERY_DIAGNOSIS_C/observations_v1/query_crops',
    'V259_ROOT_PUBLIC_OCCLUSION_PILOT',
    'V259_ROOT_PUBLIC_OCCLUSION_PILOT/raw',
    'V259_ROOT_PUBLIC_OCCLUSION_PILOT/public_frame_geometry_v1',
    'V259_ROOT_PUBLIC_OCCLUSION_PILOT/public_aux_label_pilot_v1',
    'V259_PUBLIC_SURFACE_LABEL_COMPILER_B',
    'V259_PUBLIC_PILOT_QA_A',
    'V259_PUBLIC_PILOT_QA_A/actual_review_v1',
    'V259_PUBLIC_LABEL_VISUAL_REPAIR_C',
    'V259_PUBLIC_LABEL_VISUAL_REPAIR_C/figures_v2',
    'V260_PUBLIC_POINT_FEATURE_TRANSPORT_B',
    'V259_PUBLICATION_HELPER_A',
)
ROOT_CODE = (
    'download_v258_public_frame.py',
    'diagnose_v258_public_frame.py',
    'collect_v259_occluded_public_pilot.py',
    'diagnose_v259_public_occlusion.py',
    'compile_v259_public_label_pilot.py',
    'collect_v259_publication.py',
)
ALIASES = {
    'README_KO.md': 'V259_ROOT_PUBLIC_OCCLUSION_PILOT/PUBLIC_README_KO.md',
    'ROOT_READOUT_KO.html': 'V259_ROOT_PUBLIC_OCCLUSION_PILOT/ROOT_READOUT_KO.html',
    'figures/PUBLIC_AUX_LABELS_OVERVIEW_V2.png': 'V259_PUBLIC_LABEL_VISUAL_REPAIR_C/figures_v2/PUBLIC_AUX_LABELS_OVERVIEW_V2.png',
    'figures/PUBLIC_R_Q_CROP_DIAGNOSTIC_V2.png': 'V259_PUBLIC_LABEL_VISUAL_REPAIR_C/figures_v2/PUBLIC_R_Q_CROP_DIAGNOSTIC_V2.png',
    'references/OFFICIAL_API_RESPONSE.json': 'V255_PUBLIC_PALLET_METADATA_ROOT/OFFICIAL_API_RESPONSE.json',
    'references/OFFICIAL_README.md': 'V255_PUBLIC_PALLET_METADATA_ROOT/OFFICIAL_README.md',
    'assets/pallox.obj': 'V257_ROOT_PUBLIC_PALLET_GEOMETRY/pallox.obj',
    'references/CODE/canonical_adapter.py': 'V255_NEW_CANONICAL_ADAPTER_B/canonical_adapter.py',
    'references/CODE/build_canonical_source_features.py': 'V239_CANONICAL_FEATURE_PRODUCER_B/build_canonical_source_features.py',
    'references/CODE/source_training_contract.py': 'V219_TRAINING_PREPARATION_C/source_training_contract.py',
    'references/CODE/crop_modules.py': 'V233_DETERMINISTIC_CROP_FIX_A/crop_modules.py',
    'references/CODE/build_new_scene_features.py': 'V255_NEW_SCENE_CANONICAL_DRIVER_B/build_new_scene_features.py',
}


def safe(path):
    p = Path(path)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for ancestor in reversed([p, *p.parents]):
        assert not ancestor.is_symlink(), str(ancestor)
    return p


def binding(path):
    p = safe(path)
    h, size = hashlib.sha256(), 0
    with p.open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
            size += len(chunk)
    return dict(path=str(p), sha256=h.hexdigest(), size_bytes=size)


def main():
    rows, used = [], set()

    def add(path, rel):
        assert rel not in used
        used.add(rel)
        rows.append(dict(source=binding(path), relative_path=rel))

    allowed_dirs = {safe(R / name) for name in DIRECTORIES}
    for name in DIRECTORIES:
        p = safe(R / name)
        for child in sorted(p.iterdir()):
            safe(child)
            if child.is_dir():
                assert child in allowed_dirs, f'Unadmitted directory: {child}'
            else:
                # The entry is created only after the source allowlist is sealed.
                assert child.name != 'ACTUAL_PUBLICATION_ENTRY.json'
                add(child, 'artifacts/' + str(child.relative_to(R)))
    for name in ROOT_CODE:
        add(R / name, 'tools/' + name)
    for rel, name in ALIASES.items():
        add(R / name, rel)
    extra_code = {
        'spatial_input_math.py': 'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py',
        'prepare_frozen_rgb_features.py': 'query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py',
        'target_context_math.py': 'query_visibility_target_context_rgb_20261023_v107/target_context_math.py',
    }
    for rel, name in extra_code.items():
        add(R.parent / name, 'references/CODE/' + rel)
    helper = binding(R / 'V259_PUBLICATION_HELPER_A/publish_public_point_teacher_v2.py')
    assert helper['sha256'] == '50f8387c2dd624f36a86da31b1ea2ba9810e23c6f60a49f6e7a88d8ec98aba2a'
    entry = dict(
        RootapprovedPublicPointTeacherPublication=True,
        helper_sha256=helper['sha256'],
        output_directory=str(R / 'V259_PUBLIC_POINT_TEACHER_PACKAGE'),
        repo_directory='/home/minjae/Documents/pallet-visibility-publication-20261024',
        destination_relative_path='_docs/experiments/pallet-keypoint-visibility/V259_PUBLIC_DEPTH_POINT_TEACHER_PILOT',
        parent_commit='3a60c7571019ed59c864736b7e7f09c01513e8e2',
        stage_receipt_directory=str(R / 'V259_ROOT_PUBLIC_OCCLUSION_PILOT/publication_stage_v1'),
        source_files=sorted(rows, key=lambda row: row['relative_path']),
        scope='Byte-only publication of public-source V258/V259 actual diagnostics and V260 CODE audit. No private/protected case pixels, classifier forward, training, threshold choice, or PnP.',
    )
    path = safe(R / 'V259_ROOT_PUBLIC_OCCLUSION_PILOT/ACTUAL_PUBLICATION_ENTRY.json')
    with path.open('x') as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(entry=binding(path), files=len(rows), bytes=sum(row['source']['size_bytes'] for row in rows)), ensure_ascii=False))


if __name__ == '__main__':
    main()
