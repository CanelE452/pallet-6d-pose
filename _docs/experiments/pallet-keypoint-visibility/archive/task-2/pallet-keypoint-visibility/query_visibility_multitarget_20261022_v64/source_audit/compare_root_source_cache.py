#!/usr/bin/env python3
"""Independent v64 source-cache comparison, CPU, public YCB only.

Evidence labels are a separately built, immutable reconstruction from the
official source masks. RGB and input-box grids are reconstructed here directly
from source files/registered SAM input metadata, without importing the builder.
"""
from pathlib import Path
import csv, hashlib, json, time
from collections import defaultdict
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
V63 = ROOT.parent / 'query_visibility_public_real_20261022_v63'
DATA = (V63 / 'ycb/data/train_real').resolve()
REPORT = OUT / 'ROOT_SOURCE_CACHE_INDEPENDENT_COMPARISON.json'
PER_QUERY = OUT / 'ROOT_SOURCE_CACHE_RGB_BOX_CHECKS.csv'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def csvrows(p):
    with Path(p).open(newline='') as f:
        return list(csv.DictReader(f))

def check_source(p):
    p = Path(p).resolve()
    assert p.is_relative_to(DATA), p
    return p

def native_grid(row):
    # Deliberately use the evidence reconstruction's center-of-cell expression.
    q = np.array([float(row['query_x']), float(row['query_y'])])
    side = float(row['roi_side'])
    axis = ((np.arange(9) + 0.5) / 9 - 0.5) * side
    dx, dy = np.meshgrid(axis, axis)
    xy = q + np.stack([dx, dy], axis=-1)
    px = np.floor(xy + .5).astype(np.int64)
    return xy, px

def source_patch_values(rgb, pixels):
    # Direct pixel gather. RGB channels are outermost, then row-major5x5.
    offsets = np.array([(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)])
    loc = pixels[..., None, :] + offsets
    h, w = rgb.shape[:2]
    inside = (loc[..., 0] >= 0) & (loc[..., 0] < w) & (loc[..., 1] >= 0) & (loc[..., 1] < h)
    values = rgb[np.clip(loc[..., 1], 0, h - 1), np.clip(loc[..., 0], 0, w - 1)].copy()
    values[~inside] = 0
    return values.transpose(3, 2, 0, 1).reshape(75, 9, 9)

def compare(split):
    started = time.monotonic()
    cache = ROOT / f'source_cache/YCB_{split}.npz'
    evidence = OUT / f'YCB_{split}_DENSE_EVIDENCE.npz'
    rows_path = OUT / f'YCB_{split}_QUERY_EVIDENCE.csv'
    old = V63 / f'source_cache/v2/YCB_{split}.npz'
    bindings = {str(p): sha(p) for p in [cache, evidence, rows_path, old]}
    rows = csvrows(rows_path)
    with np.load(cache, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    with np.load(evidence, allow_pickle=False) as z:
        ev = {k: z[k] for k in ['sample_id', 'visible_r0', 'full_r0', 'visible_r2', 'full_r2', 'valid', 'image_sha256']}
    n = len(rows)
    assert n == len(a['y'])
    assert np.array_equal(a['sample_id'], ev['sample_id'])
    assert np.array_equal(a['sample_id'].astype(str), [r['sample_id'] for r in rows])
    assert np.array_equal(a['image_sha256'], ev['image_sha256'])
    valid = ev['valid']
    exact = np.where(ev['visible_r0'], 0, np.where(ev['full_r0'], 1, 2))
    near = np.where(ev['visible_r2'], 0, np.where(ev['full_r2'], 1, 2))
    expected_classes = np.stack([exact, near], axis=1)
    valid2 = np.broadcast_to(valid[:, None], a['classes'].shape)
    mismatch = {
        'dense_valid': int((a['dense_valid'] != valid).sum()),
        'exact_visible_valid_cells': int(((a['classes'][:, 0] == 0) != ev['visible_r0'])[valid].sum()),
        'exact_full_valid_cells': int(((a['classes'][:, 0] != 2) != ev['full_r0'])[valid].sum()),
        'radius2_visible_valid_cells': int(((a['classes'][:, 1] == 0) != ev['visible_r2'])[valid].sum()),
        'radius2_full_valid_cells': int(((a['classes'][:, 1] != 2) != ev['full_r2'])[valid].sum()),
        'classes_valid_cells': int((a['classes'][valid2] != expected_classes[valid2]).sum()),
        'center_exact_classes_all_queries': int((a['classes'][:, 0, 4, 4] != exact[:, 4, 4]).sum()),
        'center_radius2_classes_all_queries': int((a['classes'][:, 1, 4, 4] != near[:, 4, 4]).sum()),
        'center_visible_vs_original_y': int(((a['classes'][:, 0, 4, 4] == 0) != a['y'].astype(bool)).sum()),
    }
    table = np.array([[0, -1, -1], [1, 2, -1], [3, 4, 5]])
    joint = table[exact, near]
    mismatch['joint_valid_cells'] = int((a['joint'][valid] != joint[valid]).sum())
    mismatch['invalid_joint_values_on_valid'] = int((a['joint'][valid] < 0).sum())
    preserved = {}
    with np.load(old, allow_pickle=False) as z:
        for k in ['x', 'y', 'valid', 'source', 'id', 'sample_id', 'native_corner', 'context', 'group_id', 'image_sha256']:
            preserved[k] = bool(np.array_equal(a[k], z[k], equal_nan=True) if np.issubdtype(a[k].dtype, np.number) else np.array_equal(a[k], z[k]))

    # All VAL queries and prespecified systematic256 TRAIN queries, without scores.
    rgb_indices = set(range(n)) if split == 'VALIDATION' else set(np.linspace(0, n - 1, 256, dtype=int).tolist())
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['image']].append(i)
    rgb_bad_values = rgb_bad_cells = rgb_bad_centers = 0
    rgb_values_checked = rgb_cells_checked = 0
    box_bad_values = 0
    box_max_abs = 0.
    records = []
    for frame_no, (path, ix) in enumerate(groups.items()):
        selected = [i for i in ix if i in rgb_indices]
        rgb = None
        if selected:
            path = check_source(path)
            assert sha(path) == rows[selected[0]]['image_sha256']
            rgb = np.asarray(Image.open(path).convert('RGB'))
            assert rgb.shape == (480, 640, 3)
        for i in ix:
            row = rows[i]
            inp = INPUTS[row['id']]
            assert inp['source'] == 'YCB' and inp['image_sha256'] == row['image_sha256']
            assert np.array_equal(np.asarray(inp['query'][int(row['query_index'])]), [float(row['query_x']), float(row['query_y'])])
            assert float(inp['side']) == float(row['roi_side'])
            xy, px = native_grid(row)
            x0, y0, x1, y1 = np.array(inp['box'], float)
            bw, bh = x1 - x0, y1 - y0
            expected_box = np.empty((4, 9, 9), np.float32)
            expected_box[0] = (xy[..., 0] - x0) / (bw / 2) - 1
            expected_box[1] = (xy[..., 1] - y0) / (bh / 2) - 1
            expected_box[2] = np.log2(float(row['roi_side']) / bw)
            expected_box[3] = np.log2(float(row['roi_side']) / bh)
            expected_box = np.clip(expected_box, -4, 4)
            difference = np.abs(a['box_grid'][i] - expected_box)
            box_max_abs = max(box_max_abs, float(difference.max()))
            box_bad_values += int((difference > 1e-6).sum())
            record = {'split': split, 'sample_id': row['sample_id'], 'box_grid_values_checked': 324, 'box_grid_max_abs_difference': float(difference.max()), 'rgb_checked': int(i in rgb_indices)}
            if i in rgb_indices:
                expected_rgb = source_patch_values(rgb, px)
                diff = a['rgb_native'][i] != expected_rgb
                bad_values = int(diff.sum())
                bad_cells = int(diff.any(axis=0).sum())
                center_bad = int((a['rgb_native'][i, [12, 37, 62]] != expected_rgb[[12, 37, 62]]).sum())
                rgb_bad_values += bad_values
                rgb_bad_cells += bad_cells
                rgb_bad_centers += center_bad
                rgb_values_checked += int(expected_rgb.size)
                rgb_cells_checked += 81
                record.update(rgb_patch_value_mismatches=bad_values, rgb_patch_cell_mismatches=bad_cells, rgb_center_channel_mismatches=center_bad)
            records.append(record)
        if (frame_no + 1) % 200 == 0:
            print(split, 'frames', frame_no + 1, '/', len(groups), 'seconds', round(time.monotonic() - started, 2), flush=True)
    assert all(sha(p) == h for p, h in bindings.items()), 'source cache changed during independent audit'
    result = dict(split=split, queries=n, valid_native9_cells=int(valid.sum()), label_mismatches=mismatch,
                  original_fields_byte_value_equal=preserved,
                  RGB=dict(queries_checked=len(rgb_indices), selection='all queries' if split == 'VALIDATION' else 'fixed linspace256 cache rows selected without scores', native9_cells_checked=rgb_cells_checked, uint8_values_checked=rgb_values_checked, patch_value_mismatches=rgb_bad_values, patch_cell_mismatches=rgb_bad_cells, center_channel_mismatches=rgb_bad_centers, includes_out_of_frame_zero_padding=True),
                  box_grid=dict(queries_checked=n, values_checked=n*324, values_error_gt_1e_minus6=box_bad_values, max_absolute_difference=box_max_abs, compared_to='registered source SAM INPUTS.jsonl same box, query and ROI side'),
                  input_sha256=bindings, input_sha256_preserved=True, seconds=time.monotonic()-started)
    return result, records

if __name__ == '__main__':
    assert not REPORT.exists() and not PER_QUERY.exists(), 'preserve audit outputs'
    start = time.monotonic()
    sam_input = V63 / 'sam_source/INPUTS.jsonl'
    sam_hash = sha(sam_input)
    INPUTS = {r['id']: r for r in [json.loads(s) for s in sam_input.read_text().splitlines()] if r['source'] == 'YCB'}
    summary = dict(schema='YCB_ROOT_SOURCE_CACHE_INDEPENDENT_COMPARISON_V64_V1', source_only=True, private_files_or_scores_read=False, GPU_used=False, training=False, PnP=False,
                   labels='separate immutable official-mask evidence reconstruction; compare labels only where dense_valid1, all centers',
                   invalid_labels='root caches store nearest-clipped labels even for invalid grid cells; these are intentionally excluded from label comparison and must be excluded from every dense loss',
                   RGB='original source RGB; floor(xy+.5) centers, all native9 cells, channel*(25)+dy*5+dx, black padding outside source canvas',
                   box_provenance_caveat='The new grid consumes only the registered RGB-side SAM input rough box, query and ROI side. However, public YCB source rough boxes were generated from official bbox_obj with deterministic0.9..1.1 scale jitter; calling the public source box GT-free would be inaccurate. No mask values/classes enter new box_grid computation. Deployment requires its own available initial rough box.',
                   source_SAM_inputs=dict(path=str(sam_input),sha256=sam_hash), splits={})
    records=[]
    for split in ['VALIDATION', 'TRAIN']:
        result, rr = compare(split)
        summary['splits'][split] = result
        records.extend(rr)
        print(json.dumps(result, indent=2), flush=True)
    assert sha(sam_input)==sam_hash
    summary['seconds']=time.monotonic()-start
    summary['all_checks_pass']=all(not any(s['label_mismatches'].values()) and all(s['original_fields_byte_value_equal'].values()) and s['RGB']['patch_value_mismatches']==0 and s['box_grid']['values_error_gt_1e_minus6']==0 for s in summary['splits'].values())
    with PER_QUERY.open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in records for k in r))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    summary['per_query_checks']=dict(path=str(PER_QUERY),sha256=sha(PER_QUERY))
    REPORT.write_text(json.dumps(summary,indent=2)+'\n')
    print('AUDIT COMPLETE',summary['all_checks_pass'],flush=True)
