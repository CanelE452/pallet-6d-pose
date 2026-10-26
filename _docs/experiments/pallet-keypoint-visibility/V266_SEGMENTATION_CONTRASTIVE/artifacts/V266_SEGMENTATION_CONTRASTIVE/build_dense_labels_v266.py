"""SYN-only dense ownership packet, no RGB decode or model/real-data access.

Class order TARGET0 / FOREIGN1 / BACKGROUND2 is fixed before optimizer work.
Source owner labels describe visible raster surfaces, not native D/E states.
"""
from pathlib import Path
import collections
import hashlib
import io
import json
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).absolute().parent
ROOT = HERE.parent
SOURCE = ROOT / 'V263_LOCALIZATION_DATA_A/actual_label_packet_v2/GROUPS_WITH_NATIVE_GEOMETRIC_PROOF.jsonl'
SIDE = ROOT / 'V263_SPATIAL_HEAD_C/source_spatial_features_geometry_v1/CROP_SIDE.npy'
SUPPORT = ROOT / 'V263_SPATIAL_HEAD_C/source_spatial_features_geometry_v1/TOKEN_SUPPORT.npy'
RGB_MANIFEST = ROOT / 'V263_SPATIAL_HEAD_C/source_spatial_features_geometry_v1/RGB_ACCESS_WHITELIST.json'
LABEL_COMPLETE = ROOT / 'V263_LOCALIZATION_DATA_A/actual_label_packet_v2/COMPLETE.json'
FEATURE_COMPLETE = ROOT / 'V263_SPATIAL_HEAD_C/source_spatial_features_geometry_v1/COMPLETE.json'
EXPECTED = {
    str(SOURCE): '98f317f66f1c51d56daf55dbf452076ec3c92d159fd524fbba3d1ccc70e98726',
    str(SIDE): '130d4ccfaeea9207c15566d543747ee3d413a08f40e6fb05be8ec5f0b90b8172',
    str(SUPPORT): '9060f6a038091ca1dce5b6b1a9d8bfad61070b6c6207c7f726b47aec4386da29',
    str(RGB_MANIFEST): '153c1dc1681f1920ff322941c7e373b93e271b145e6be828f8e41009f90ee23a',
    str(LABEL_COMPLETE): 'b63f3934b5471ae444f275e50951e15c89fffecd7db0dc491fb10b53adc1db8d',
    str(FEATURE_COMPLETE): '3ec0b27cc20552ef7e8edc1c09b6f72a825eb343147f16708e2c6cb924f6271d',
}
N = 7040


def safe(value):
    p = Path(value)
    if not p.is_absolute() or '..' in p.parts or any(x in str(p).lower() for x in ('v112', 'cache', '.pyc')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError(str(a))
    return p


def binding(value, expected=None):
    p = safe(value)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    b = {'path': str(p), 'sha256': h.hexdigest(), 'size_bytes': p.stat().st_size}
    if expected is not None:
        assert b['sha256'] == expected, b
    return b


def write_json(path, data):
    p = safe(path)
    assert not p.exists(), str(p)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def sample_owner(onehot, q, side, available):
    """Literal V90 crop-coordinate/bilinear law, without uint8 rounding."""
    result = np.zeros((3, 96, 96), np.float64)
    valid = np.zeros((96, 96), bool)
    if not available:
        return result, valid
    assert np.isfinite(q).all() and np.isfinite(side) and side > 0
    h, w = onehot.shape[1:]
    u = (np.arange(96, dtype=np.float64) + .5) / 96 - .5
    xx, yy = np.meshgrid(q[0] + u * side, q[1] + u * side)
    ix = np.floor(xx).astype(np.int64)
    iy = np.floor(yy).astype(np.int64)
    dx, dy = xx - ix, yy - iy
    for ax, ay, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                           (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        x, y = ix + ax, iy + ay
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        result[:, inside] += onehot[:, y[inside], x[inside]] * weight[inside]
    valid = (xx >= -.5) & (xx < w-.5) & (yy >= -.5) & (yy < h-.5)
    result[:, ~valid] = 0
    return result, valid


def main():
    started = time.time()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = safe(HERE / 'data_v1')
    assert not out.exists(), str(out)
    inputs = {p.name: binding(p, EXPECTED[str(p)]) for p in
              (SOURCE, SIDE, SUPPORT, RGB_MANIFEST, LABEL_COMPLETE, FEATURE_COMPLETE)}
    groups = [json.loads(l) for l in safe(SOURCE).read_text().splitlines()]
    sides = np.load(safe(SIDE), allow_pickle=False)
    expected_support = np.load(safe(SUPPORT), allow_pickle=False)
    manifest = {x['path']: x for x in json.loads(safe(RGB_MANIFEST).read_text())}
    assert len(groups) == 880 and sides.shape == (N,) and expected_support.shape == (N, 16, 16)
    by_image = collections.OrderedDict()
    source_bases = sorted({int(g['base_index']) for g in groups})
    base_index = {b: i for i, b in enumerate(source_bases)}
    assert len(source_bases) == 110
    soft = np.zeros((N, 3, 16, 16), np.float16)
    mass = np.zeros((N, 16, 16), np.float16)
    base_id = np.full(N, -1, np.int32)
    fit = np.zeros(N, bool)
    seen = np.zeros(N, bool)
    derived_support = np.zeros((N, 16, 16), bool)
    row_manifest = [None] * N
    for g in groups:
        assert g['group_index'] >= 0
        by_image.setdefault(g['image']['path'], []).append(g)
        for slot, index in enumerate(g['source_indices8']):
            assert not seen[index]
            seen[index] = True
            base_id[index] = base_index[int(g['base_index'])]
            fit[index] = bool(g['fit'])
            row_manifest[index] = {'source_index': index, 'group_index': g['group_index'],
                 'row_id': g['row_ids8'][slot], 'native_slot': slot,
                 'base_id': int(base_id[index]), 'source_base_index': int(g['base_index']),
                 'fit': bool(g['fit']), 'variant': g['variant'], 'image': g['image'],
                 'q0': g['q8'][slot], 'crop_side': float(sides[index]),
                 'original_available': bool(g['original_available8'][slot])}
    assert seen.all() and len(by_image) == 440 and fit.sum() == 5504
    for b in range(110):
        assert np.unique(fit[base_id == b]).size == 1
    fold_counts = {k: collections.Counter() for k in ('FIT', 'DEV')}
    file_bindings = []
    part_values = set()
    fold_hashes = {k: set() for k in ('FIT', 'DEV')}
    fold_paths = {k: set() for k in ('FIT', 'DEV')}
    variants = collections.Counter()
    max_mass_float32_excess = 0.0
    for image_number, (image_path, image_groups) in enumerate(by_image.items(), 1):
        first = image_groups[0]
        assert all(g['fit'] == first['fit'] and g['base_index'] == first['base_index'] for g in image_groups)
        fold = 'FIT' if first['fit'] else 'DEV'
        variants[first['variant']] += 1
        b = binding(image_path, first['image']['sha256'])
        assert manifest[image_path]['sha256'] == b['sha256']
        assert manifest[image_path]['size_bytes'] == b['size_bytes']
        file_bindings.append(b)
        fold_hashes[fold].add(b['sha256']); fold_paths[fold].add(image_path)
        with np.load(io.BytesIO(safe(image_path).read_bytes()), allow_pickle=False) as z:
            assert {'part_id', 'camera_depth', 'ambiguity', 'tie_count', 'global_triangle_id'} <= set(z.files)
            owner = np.array(z['part_id'], copy=True)
            depth = np.array(z['camera_depth'], copy=True)
            ambiguity = np.array(z['ambiguity'], copy=True).astype(bool)
            ties = np.array(z['tie_count'], copy=True)
        assert owner.shape == depth.shape == ambiguity.shape == ties.shape == (512, 512)
        values = set(np.unique(owner).tolist()); part_values.update(values)
        assert values <= set(range(-1, 19)) | {999}
        known = ~ambiguity & (ties < 2)
        known &= ((owner < 0) & ~np.isfinite(depth)) | ((owner >= 0) & np.isfinite(depth) & (depth > 0))
        masks = ((owner >= 0) & (owner != 999), owner == 999, owner < 0)
        onehot = np.stack([a & known for a in masks]).astype(np.float64)
        c = fold_counts[fold]
        c['images'] += 1; c['pixels_all'] += owner.size
        c['source_unknown_pixels'] += int((~known).sum())
        c['ambiguity_pixels'] += int(ambiguity.sum()); c['tie_ge2_pixels'] += int((ties >= 2).sum())
        for name, mk in zip(('TARGET0', 'FOREIGN1', 'BACKGROUND2'), masks):
            c[name + '_source_known_pixels'] += int((mk & known).sum())
        indices = []; crops = []
        for g in image_groups:
            assert g['image']['sha256'] == b['sha256']
            for slot, index in enumerate(g['source_indices8']):
                crop, valid = sample_owner(onehot, np.asarray(g['q8'][slot], np.float64),
                                           float(sides[index]), bool(g['original_available8'][slot]))
                derived_support[index] = valid.reshape(16, 6, 16, 6).any((1, 3))
                indices.append(index); crops.append(crop)
        x = torch.from_numpy(np.stack(crops).astype(np.float32))
        resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
        pooled = F.avg_pool2d(resized, kernel_size=14, stride=14)
        km = pooled.sum(1)
        target = torch.where(km[:, None] > 0, pooled / km[:, None].clamp_min(1e-12), torch.zeros_like(pooled))
        assert bool(torch.isfinite(target).all())
        max_mass_float32_excess = max(max_mass_float32_excess, float(km.max()) - 1.0)
        assert bool((km >= -1e-7).all()) and bool((km <= 1+5e-5).all()), (float(km.min()), float(km.max()))
        soft[indices] = target.numpy().astype(np.float16)
        # Float32 pooling accumulates 196 values: remove only numerical excess.
        mass[indices] = km.clamp(0, 1).numpy().astype(np.float16)
        if image_number % 80 == 0:
            print(json.dumps({'dense_GT_images_done': image_number, 'total': 440}), flush=True)
    assert np.array_equal(expected_support, derived_support)
    assert not (fold_hashes['FIT'] & fold_hashes['DEV']) and not (fold_paths['FIT'] & fold_paths['DEV'])
    assert np.isfinite(soft).all() and np.isfinite(mass).all()
    code_binding = binding(Path(__file__).absolute())
    out.mkdir()
    np.savez_compressed(safe(out / 'dense_labels.npz'), soft_target=soft, known_mass=mass,
                         base_id=base_id, fit=fit)
    safe(out / 'ROW_MANIFEST.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in row_manifest))
    write_json(out / 'SOURCE_BUFFER_BINDINGS.json', {'buffers': file_bindings, 'RGB_members_decoded': 0})
    token_census = {}
    for is_fit, fold in ((True, 'FIT'), (False, 'DEV')):
        m = fit == is_fit
        classmass = soft[m].astype(np.float64) * mass[m, None].astype(np.float64)
        token_census[fold] = {'rows': int(m.sum()), 'supported_rows': int(expected_support[m].reshape(m.sum(), -1).any(1).sum()),
             'tokens_total': int(m.sum() * 256), 'tokens_known_mass_positive': int((mass[m] > 0).sum()),
             'tokens_unknown_mass_zero': int((mass[m] == 0).sum()),
             'known_mass_sum': float(mass[m].astype(np.float64).sum()),
             'class_weighted_mass_TARGET0_FOREIGN1_BACKGROUND2': classmass.sum((0, 2, 3)).tolist(),
             'hard_argmax_class_counts_TARGET0_FOREIGN1_BACKGROUND2':
                  [int(((soft[m].argmax(1) == k) & (mass[m] > 0)).sum()) for k in range(3)]}
    label_codebook = {'0': 'TARGET_INSTANCE_VISIBLE_SURFACE', '1': 'FOREIGN_VISIBLE_SURFACE', '2': 'BACKGROUND_NO_GEOMETRY',
         'UNKNOWN': 'Not assigned a class; zero known_mass. Ambiguity/ties/baddepth/invalid/OOB ignored.',
         'native_visibility': 'Original D/E/S/U labels remain separate. Foreign pixels are not automatically native E.'}
    visual_cases = []
    for is_fit, variant in ((False, 'BEFORE'), (False, 'COVER'), (False, 'NEAR_SHAM'),
                             (True, 'BEFORE'), (True, 'COVER'), (True, 'FAR_SHAM')):
        ids = [i for i, r in enumerate(row_manifest) if r['fit'] == is_fit and r['variant'] == variant]
        visual_cases.append(min(ids))
    from PIL import Image, ImageDraw
    board = Image.new('RGB', (960, 680), 'white')
    draw = ImageDraw.Draw(board)
    draw.text((16, 10), 'SYN dense ownership targets only (no RGB). TARGET green / FOREIGN red / BG blue / UNKNOWN black', fill='black')
    palette = np.array([[60, 190, 95], [225, 65, 65], [80, 110, 210]], np.float32)
    for panel, index in enumerate(visual_cases):
        r = row_manifest[index]
        rgb = np.einsum('chw,cd->hwd', soft[index].astype(np.float32), palette)
        rgb[mass[index] == 0] = 0
        tile = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((288, 288), Image.Resampling.NEAREST)
        x, y = 16 + panel % 3 * 320, 56 + panel // 3 * 312
        board.paste(tile, (x, y))
        draw.text((x, y-18), f'row{index} {"FIT" if r["fit"] else "DEV"} {r["variant"]} native{r["native_slot"]}', fill='black')
    board.save(safe(out / 'SYN_DENSE_TARGETS_FIXED6.png'))
    census = {'completed': True, 'PASS': True, 'schema': 'V266_DENSE_OWNERSHIP_SYN_V1',
         'code': code_binding, 'inputs': inputs, 'all_source_buffer_bindings': binding(out / 'SOURCE_BUFFER_BINDINGS.json'),
         'label_codebook': label_codebook, 'arrays': {'soft_target': [N, 3, 16, 16], 'known_mass': [N, 16, 16],
              'base_id': [N], 'fit': [N]}, 'array_dtypes': {'soft_target': 'float16', 'known_mass': 'float16', 'base_id': 'int32', 'fit': 'bool'},
         'row_order': 'Unchanged V263 source_index 0..7039', 'part_id_values': sorted(part_values),
         'source_pixel_census': {k: dict(v) for k, v in fold_counts.items()}, 'token_mass_census': token_census,
         'variants_unique_images': dict(variants), 'base_id_to_source_base_index': source_bases,
         'split_proof': {'base_count_FIT_DEV': [86, 24], 'image_path_count_FIT_DEV': [344, 96],
             'encoded_buffer_sha_unique_FIT_DEV': [len(fold_hashes['FIT']), len(fold_hashes['DEV'])],
             'base_overlap': 0, 'image_path_overlap': 0, 'buffer_sha_overlap': 0, 'row_fit_counts': [5504, 1536]},
         'crop': {'q': 'original group q8', 'L': 'stored V263 geometry CROP_SIDE.npy; no recomputation',
             'source_coordinate': 'q+((arange(96)+0.5)/96-0.5)*L',
             'sampling': 'four-neighbor bilinear one-hot times known, invalid outside[-0.5,w-0.5) ignored; no label rounding',
             'resize': '96->224 bilinear align_corners=False antialias=True',
             'token_pool': 'avg_pool2d kernel14 stride14 ->16x16',
             'normalization': 'soft_target=pooled_class_mass/known_mass where known_mass>0; unknown zero',
             'label_mass_support_may_extend_across_token_boundary_due_to_resize': True},
         'support_replay': {'derived_96_to_6x6_any_equals_existing_TOKEN_SUPPORT': True,
             'mismatched_tokens': 0, 'known_mass_vs_existing_support_mismatched_tokens': int(((mass > 0) != expected_support).sum()),
             'max_soft_target_sum_error_after_FP16': float(np.abs(soft.sum(1).astype(np.float64)[mass > 0]-1).max())},
         'floating_point': {'max_pooled_mass_excess_before_clip': max_mass_float32_excess,
             'known_mass_clipped_to_closed_unit_interval_after_soft_target_normalization': True,
             'prep_runtime_correction': 'Initial run stopped before data_v1 creation at overly strict 1e-6 float32 pooled mass assertion; tolerance5e-5 and explicit mass clip only. No models/optimizers/current reads.'},
         'fixed6_before_any_model_outputs': visual_cases,
         'RGB_members_decoded': 0, 'current_or_real_reads': 0, 'model_forwards': 0, 'optimizer_updates': 0,
         'elapsed_seconds': time.time()-started, 'runtime': {'python': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__}}
    write_json(out / 'DATA_CENSUS.json', census)
    names = ('dense_labels.npz', 'ROW_MANIFEST.jsonl', 'SOURCE_BUFFER_BINDINGS.json', 'SYN_DENSE_TARGETS_FIXED6.png', 'DATA_CENSUS.json')
    complete = {'completed': True, 'PASS': True, 'schema': census['schema'], 'code': code_binding,
           'outputs': {n: binding(out/n) for n in names}, 'arrays': census['arrays'], 'label_codebook': label_codebook,
           'rows': N, 'RGB_members_decoded': 0, 'current_or_real_reads': 0, 'model_forwards': 0, 'optimizer_updates': 0,
           'elapsed_seconds': time.time()-started}
    write_json(out / 'COMPLETE.json', complete)
    print(json.dumps({'completed': True, 'PASS': True, 'complete': binding(out/'COMPLETE.json'),
                     'packet': binding(out/'dense_labels.npz'), 'token_mass_census': token_census}), flush=True)


if __name__ == '__main__':
    main()
