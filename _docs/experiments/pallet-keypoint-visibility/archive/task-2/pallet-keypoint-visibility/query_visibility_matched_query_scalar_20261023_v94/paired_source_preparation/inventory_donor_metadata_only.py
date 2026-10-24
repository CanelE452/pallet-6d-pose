"""Project already-local YCB JSONL metadata; never opens RGB/mask/depth bodies."""
from collections import Counter
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
P = ROOT.parent.parent
YCB = P / 'query_visibility_public_real_20261022_v63/ycb'


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def records(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def write(path, obj):
    assert not path.exists()
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def run():
    policy = ROOT / 'DONOR_POLICY_BEFORE_DONOR_BODIES.json'
    write(policy, {'completed': True, 'stage': 'METADATA_ONLY_NO_DONOR_BODY_OR_ADMISSION',
                   'TRAIN': {'scenes': list(range(8)), 'physical_YCB_object_IDs': list(range(1, 22, 2))},
                   'VALIDATION': {'scenes': [8, 9], 'physical_YCB_object_IDs': list(range(2, 22, 2))},
                   'selection': 'One minSHA(seed|targetinputid|logicalslot|one_donor|donorid|RGBsha), no pixel/quality/feasibility filtering or replacement. Unusable chosen donor => afterU, never substitute.',
                   'meaning': 'Same physical YCB object appears across scenes, so disjoint scene IDs alone are insufficient; object families are explicitly disjoint too.',
                   'metadata_pool': 'All4200 existing V63 records retained; only matching split and odd/even family are eligible candidates. No native source mask/RGB body opened.',
                   'scope': 'GT-support-guided controlled augmentation, not GT-free generation or natural-foreign/human-native gold.'})
    index = YCB / 'SELECTED_FILES_SHA256.jsonl'
    official = {str((YCB / r['relative_path']).resolve()): r['sha256'] for r in records(index)}
    out = []; objectsets = {}; imagesets = {}; metadata_zero = Counter()
    sources = [('TRAIN', YCB / 'train_instances.jsonl', set(range(8))),
               ('VALIDATION', YCB / 'SOURCE_VAL_LOCKED_200.jsonl', {8, 9})]
    for split, path, scenes in sources:
        objectsets[split] = set(); imagesets[split] = set()
        for r in records(path):
            assert r['scene_id'] in scenes
            descriptors = {}
            for key, sourcekey in [('image', 'image'), ('visible_alpha_source', 'mask_vis')]:
                fp = Path(r[sourcekey]).resolve()
                assert fp.is_relative_to((YCB / 'data/train_real').resolve())
                assert str(fp) in official
                # Existence/stat only. Do not hash or decode any pixel body here.
                assert fp.is_file()
                descriptors[key] = {'path': str(fp), 'sha256': official[str(fp)],
                                    'bytes_stat_only': fp.stat().st_size}
            assert descriptors['image']['sha256'] == r['image_sha256']
            family_ok = (split == 'TRAIN' and r['object_id'] % 2 == 1) or (split == 'VALIDATION' and r['object_id'] % 2 == 0)
            row = dict(id=f"YCB:{r['scene_id']:06d}:{r['frame_id']:06d}:{r['instance_id']:06d}",
                       split=split, scene_id=r['scene_id'], frame_id=r['frame_id'], instance_id=r['instance_id'],
                       object_id=r['object_id'], image_sha256=r['image_sha256'],
                       raw_hw=[r['height'], r['width']], metadata_vis_pixels=r['vis_pixels'],
                       family_eligible=family_ok,
                       foreign_opaque_cutout_admitted=False, pixel_and_opacity_checks='PENDING_ACTUAL_INTAKE_ENTRY',
                       official_revision=r.get('official_revision'), license=r.get('license'), **descriptors)
            out.append(row)
            if r['vis_pixels'] == 0:
                metadata_zero[split] += 1
            if family_ok:
                objectsets[split].add(r['object_id']); imagesets[split].add(r['image_sha256'])
    assert len(out) == 4200 and not objectsets['TRAIN'].intersection(objectsets['VALIDATION'])
    assert not imagesets['TRAIN'].intersection(imagesets['VALIDATION'])
    dest = ROOT / 'YCB4200_DONOR_METADATA_CANDIDATES_NOT_ADMITTED.jsonl'
    assert not dest.exists()
    dest.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in out))
    write(ROOT / 'DONOR_METADATA_INVENTORY.json', {
        'completed': True, 'PASS': True, 'stage': 'LOCATED_AND_SCHEMA_MAPPED_NOT_PIXEL_ADMITTED',
        'code': binding(Path(__file__)), 'policy': binding(policy),
        'source_metadata': [binding(index)] + [binding(x[1]) for x in sources],
        'projected_metadata': binding(dest), 'all_rows_retained': len(out),
        'source_counts': dict(Counter(r['split'] for r in out)),
        'family_eligible_counts': dict(Counter(r['split'] for r in out if r['family_eligible'])),
        'physical_object_families': {k: sorted(v) for k, v in objectsets.items()},
        'eligible_crosssplit_physical_object_ID_intersection': 0,
        'eligible_crosssplit_image_SHA_intersection': 0,
        'zero_metadata_visible_rows_retained': dict(metadata_zero),
        'source_rgb_mask_file_bodies_hashed_or_decoded': 0,
        'source_depth_pose_or_fullmask_arrays_read': 0,
        'actual_source_transform_feasibility_verified': False,
        'remaining': ['Freeze exact one-donor assignment to original160 slot metadata before donor bodies.',
                      'Actual selected donor RGB/mask SHA, dimensions, opaque-core/support and visible boundary sanity.',
                      'Keep any unusable chosen donor/U without success-based replacement.',
                      'LOCO selected-target/cue consensus and capture-family admission remain separate.'],
        'new_model_forward_fit_current_PnP': 0})


if __name__ == '__main__':
    run()
