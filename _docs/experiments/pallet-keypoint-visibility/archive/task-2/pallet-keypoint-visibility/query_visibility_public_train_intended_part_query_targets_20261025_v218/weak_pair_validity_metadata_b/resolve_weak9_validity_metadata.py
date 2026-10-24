#!/usr/bin/env python3
"""Read exact JSON only; copy VALID bindings and test conservative grid bounds.

No data pointer is opened, statted, hashed or decoded. Analytical grid bounds
are explicitly not a replay of the stored VALID96 bitmap.
"""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
H = P/'query_visibility_public_train_intended_part_query_targets_20261025_v218/weak_pair_validity_metadata_b'
A = H.parent/'intended_part_targets_a'
R = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
PREVIEW = ('_docs/experiments/pallet-keypoint-visibility/archive/task-2/pallet-keypoint-visibility/'
           'query_visibility_public_train_visible_part_coverage_expansion_20261025_v215/'
           'fixed12_public_TRAIN_input_previews_v1')


def rb(p):
    p = Path(p)
    assert p.is_absolute() and 'v112' not in str(p).casefold()
    cur = Path(p.anchor)
    for part in p.parts[1:]:
        cur = cur/part
        assert not stat.S_ISLNK(os.lstat(cur).st_mode)
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        return f.read()


def checked(p, expected):
    b = rb(p)
    assert hashlib.sha256(b).hexdigest() == expected
    return json.loads(b)


def bind(p):
    return {'path': str(p), 'sha256': hashlib.sha256(rb(p)).hexdigest()}


def put(name, obj):
    assert 'v112' not in str(H/name).casefold()
    with os.fdopen(os.open(H/name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644), 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def rectangle(b):
    assert len(b) == 4 and all(type(x) is int and 0 <= x < 96 for x in b)
    x0, y0, x1, y1 = b
    assert x0 <= x1 and y0 <= y1
    return {(x, y) for y in range(y0, y1+1) for x in range(x0, x1+1)}


def conservative_grid_proof(points, q, L, hw):
    """Sufficient full four-tap source bounds, independent of stored valid bits."""
    assert points and len(q) == 2 and L > 0
    height, width = hw
    assert type(height) is int and type(width) is int and height > 1 and width > 1
    bad = []
    coords = []
    for x, y in sorted(points, key=lambda p: (p[1], p[0])):
        sx = float(q[0]) + (float(x)-47.5)*float(L)/96.0
        sy = float(q[1]) + (float(y)-47.5)*float(L)/96.0
        assert math.isfinite(sx) and math.isfinite(sy)
        x0, y0 = math.floor(sx), math.floor(sy)
        if not (0 <= x0 and x0+1 < width and 0 <= y0 and y0+1 < height):
            bad.append([x, y])
        coords.append((sx, sy))
    return {'pixels': len(points), 'all_conservative_four_taps_in_frame': not bad,
            'conservative_outside_count': len(bad), 'first_outside_sample_indices': bad[:8],
            'source_coordinate_extent_xyxy': [min(x for x,y in coords), min(y for x,y in coords),
                                              max(x for x,y in coords), max(y for x,y in coords)]}


def run():
    complete_path = R/PREVIEW/'COMPLETE.json'
    manifest_path = R/PREVIEW/'ANNOTATOR_INPUT_MANIFEST.json'
    complete = checked(complete_path, 'e19a3573217cf44d7c53ace4bc8b275b2b335345fee7d0e95103e59f5fb8ad1c')
    manifest = checked(manifest_path, 'd7982e3c61c85c82c56d630a8316b459504fbb99ebeb957f155f29cc43b13d18')
    assert complete['manifest']['sha256'] == bind(manifest_path)['sha256']
    assert complete['all_original_fullRGB_RAW96_valid96_decoded_PNG_bytes_exact'] is True
    assert len(manifest['slots96']) == 96
    by_id = {r['row_id']: r for r in manifest['slots96']}
    frames = {r['position']: r for r in manifest['frames']}
    targets_path = A/'ALL9_EXPLICIT_FINITE_LOCAL_PART_TARGETS.json'
    plans_path = A/'ALL9_FINITE_WITNESS_COVER_SHAM_EXAMPLE_PLANS.json'
    targets = checked(targets_path, '51844ddd5c142206a93a0c68c2880186a8a7d45507c29ec838ccaeb7fc5aa9bc')['targets9']
    plans = checked(plans_path, '70446a5c3d992f84e552bd17456a1545144d637f0407f9adf0b21168aed2bb06')['plans9']
    pp = {r['row_id']: r for r in plans}
    assert len(targets) == len(pp) == 9
    rows = []
    for target in targets:
        original = target['input']; row_id = original['row_id']; slot = by_id[row_id]; plan = pp[row_id]
        assert slot['position'] == original['position']
        for key in ('query_xy', 'L', 'native_slot', 'original_query_available', 'original_flags', 'component_id', 'frame_key'):
            assert slot[key] == original[key], key
        assert slot['original_query_available'] is True
        assert slot['clean96'] == target['clean96_binding']
        assert slot['original_raw_byte_identity'] is True
        frame = frames[slot['position']]
        hw = frame['source_RGB']['raw_hw']
        support = set()
        for y, x0, x1 in target['finite_target_support']['positive_runs_y_x0_x1']:
            assert type(y) is int and 0 <= y < 96 and 0 <= x0 <= x1 < 96
            support.update((x, y) for x in range(x0, x1+1))
        assert len(support) == target['finite_target_support']['positive_pixels']
        cover_bbox = plan['geometry_plan']['COVER_bbox_inclusive']
        sham_bbox = plan['geometry_plan']['SHAM_bbox_inclusive']
        cover = rectangle(cover_bbox)
        sham = rectangle(sham_bbox) if sham_bbox is not None else None
        assert support <= cover
        if sham is not None:
            assert not (support & sham) and len(cover) == len(sham)
        q, L = slot['query_xy'], slot['L']
        rows.append({'row_id': row_id, 'target_id': target['target_id'], 'position': slot['position'],
                     'native_slot': slot['native_slot'], 'internal_source_role': original['internal_source_role'],
                     'original_query_available': True, 'original_query_xy': q, 'original_L': L,
                     'raw_hw_from_manifest': hw, 'clean96': slot['clean96'],
                     'valid96': slot['valid96'], 'valid96_decoded_bool_sha256': slot['valid96_decoded_bool_sha256'],
                     'original_raw_byte_identity_saved_proof': True,
                     'finite_support': conservative_grid_proof(support, q, L, hw),
                     'COVER_bbox_inclusive': cover_bbox,
                     'COVER_full_rectangle': conservative_grid_proof(cover, q, L, hw),
                     'SHAM_bbox_inclusive': sham_bbox,
                     'SHAM_full_rectangle': conservative_grid_proof(sham, q, L, hw) if sham is not None else None,
                     'stored_VALID96_bitmap_replayed': False,
                     'actual_RGB_changes_or_pair_authority_established': False,
                     'training_admission': False})
    result = {'completed': True, 'PASS': True, 'original_metadata_manifest': complete['manifest'],
              'read_archive_manifest': bind(manifest_path), 'read_archive_preview_COMPLETE': bind(complete_path),
              'source_target_bindings': [bind(targets_path), bind(plans_path)], 'rows9': rows,
              'validity_proof_level': 'ORIGINAL_VALID96_BINDINGS_RESOLVED_AND_CONSERVATIVE_F64_GRID_BOUNDS_ONLY_NOT_BITMAP_REPLAY',
              'formula': 'source_xy = original_q + (RAW96_index - 47.5) * original_L / 96; require floor coordinate and floor+1 within original H,W for all four taps.',
              'dimensions_authority': 'saved preview frames[position].source_RGB.raw_hw; no PNG/header access',
              'binary_pointers_followed_or_statted_hashed_decoded': 0,
              'training_admission': False}
    put('EXACT9_ORIGINAL_VALID96_BINDINGS_AND_ANALYTICAL_GRID_PROOFS.json', result)
    summary = {'completed': True, 'PASS': True, 'targets': len(rows), 'valid96_bindings_resolved': len(rows),
               'target_support_all_conservative_valid': sum(r['finite_support']['all_conservative_four_taps_in_frame'] for r in rows),
               'COVER_full_rectangle_all_conservative_valid': sum(r['COVER_full_rectangle']['all_conservative_four_taps_in_frame'] for r in rows),
               'SHAM_full_rectangle_all_conservative_valid': sum(bool(r['SHAM_full_rectangle'] and r['SHAM_full_rectangle']['all_conservative_four_taps_in_frame']) for r in rows),
               'stored_VALID96_bitmap_replayed': False, 'binary_reads': 0,
               'future_pair_rule': 'Use bound actual VALID96 bitmap after root image entry. Require every actual alpha-positive COVER/SHAM sample valid; if no legal SHAM keep factual/failure without q/ROI edits. Analytical bbox result is not a substitute for donor-alpha/actual RGB change proof.',
               'proof': bind(H/'EXACT9_ORIGINAL_VALID96_BINDINGS_AND_ANALYTICAL_GRID_PROOFS.json')}
    put('SUMMARY.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    run()
