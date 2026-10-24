"""One fixed public instance: isolated/visible mask support near virtual r.

This is GT-only local-support diagnosis. No query/reference relocation, model
input generation, training, inference, current data or source corpus creation.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / 'SUBFOOTPRINT_COMPARISON'
YCB = HERE.parents[2] / 'query_visibility_public_real_20261022_v63/ycb'


def bound(path):
    p = Path(path)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}


def save(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def boundary(mask):
    p = np.pad(mask, 1)
    eroded = np.logical_and.reduce([p[i:i + mask.shape[0], j:j + mask.shape[1]]
                                   for i in range(3) for j in range(3)])
    return mask & ~eroded


def main():
    assert not OUT.exists()
    OUT.mkdir()
    reference_path = HERE / 'ONE_INSTANCE_V2/ONE_INSTANCE_NATIVE_REFERENCES_GT_ONLY.npz'
    previous_path = HERE / 'ONE_INSTANCE_V2/ONE_INSTANCE_FEASIBILITY.json'
    previous = json.loads(previous_path.read_text())
    with np.load(reference_path, allow_pickle=False) as g:
        ref = g['reference_projected8']
        self_visible = g['convex_self_visible_proxy8']
    np.testing.assert_array_equal(ref, np.asarray(previous['native_reference8_xy']))
    radius = max(4., .05 * float(np.ptp(ref, axis=0).max()))
    sigma = radius / 2.
    manifest = YCB / 'train_instances.jsonl'
    row = next(r for r in map(json.loads, manifest.read_text().splitlines())
               if (r['scene_id'], r['frame_id'], r['instance_id']) == (0, 1, 0))
    assert row['object_id'] == 8
    protocol = {
        'schema': 'FIXED_ONE_INSTANCE_REFERENCE_LOCAL_SUPPORT_DIAGNOSTIC_V1',
        'instance': [0, 1, 0, 8], 'reference_and_query_relocation': False,
        'footprint': 'Native integer-pixel centers inside disk ||p-r||<=rho, clipped toimagebounds',
        'rho': 'max(4px,.05*max(ptp(projected8,axis0)))', 'rho_px': radius, 'sigma_px': sigma,
        'weight': 'exp(-||p-r||^2/(2*sigma^2)), peak1; not unit-sum normalized',
        'mass_units': 'native pixel-equivalents; A=sum(weight*Mfull),V=sum(weight*Mvis)',
        'support_adequate': 'Mfull positivepixelcount>=4 and weightedA>=1',
        'decision': 'adequate and V/A>=.9 -> LOCAL_EXPECTED_SUPPORT_VISIBLE_PROXY; adequate and V/A<=.1 -> LOCAL_EXPECTED_SUPPORT_COVERED_PROXY; else U',
        'self_convex': 'Separate field, never merged with support state',
        'not_semantic_gold_or_pixelcenter_presence': True,
        'bindings': [bound(x) for x in [Path(__file__), reference_path, previous_path, manifest,
                                       row['image'], row['mask_full'], row['mask_vis']]],
        'RGB_body_read_authorized_scope': 'Exactly this one public frame; earlier read0 report unchanged',
        'new_forward_fit_current_PnP': 0,
    }
    # Hash-reading a public RGB before loading is provenance-only, not input use.
    save(OUT / 'PROTOCOL.json', protocol)
    full = np.asarray(Image.open(row['mask_full'])) > 0
    visible = np.asarray(Image.open(row['mask_vis'])) > 0
    assert full.shape == visible.shape == (480, 640)
    assert not (visible & ~full).any()
    equal = np.array_equal(full, visible)
    h, w = full.shape
    yy, xx = np.indices((h, w), dtype=float)
    records = []; disks = []
    for k, r in enumerate(ref):
        distance2 = (xx - r[0]) ** 2 + (yy - r[1]) ** 2
        disk = distance2 <= radius ** 2
        weights = np.exp(-distance2 / (2 * sigma ** 2)) * disk
        a_count = int((full & disk).sum()); v_count = int((visible & disk).sum())
        a_mass = float(weights[full].sum()); v_mass = float(weights[visible].sum())
        ratio = v_mass / a_mass if a_mass > 0 else None
        adequate = a_count >= 4 and a_mass >= 1.
        state = 'U_INSUFFICIENT_EXPECTED_SUPPORT'
        if adequate:
            state = ('LOCAL_EXPECTED_SUPPORT_VISIBLE_PROXY' if ratio >= .9 else
                     'LOCAL_EXPECTED_SUPPORT_COVERED_PROXY' if ratio <= .1 else 'U_PARTIAL_VISIBLE_SUPPORT')
        px, py = np.floor(r + .5).astype(int)
        center_valid = bool(0 <= r[0] < w and 0 <= r[1] < h and 0 <= px < w and 0 <= py < h)
        rec = {'native_index': k, 'reference_x': float(r[0]), 'reference_y': float(r[1]),
            'rho_px': radius, 'sigma_px': sigma, 'valid_footprint_pixel_count': int(disk.sum()),
            'expected_support_pixel_count_A': a_count, 'visible_support_pixel_count_V': v_count,
            'weighted_expected_mass_A_pixel_equivalents': a_mass,
            'weighted_visible_mass_V_pixel_equivalents': v_mass,
            'weighted_visible_over_expected_V_A': ratio, 'support_adequate': adequate,
            'local_support_proxy': state,
            'Mfull_at_reference_nearestpixel': bool(full[py, px]) if center_valid else None,
            'Mvis_at_reference_nearestpixel': bool(visible[py, px]) if center_valid else None,
            'convex_self_visible_proxy_separate': bool(self_visible[k]),
            'prior_depth_ray_proxy_separate': previous['combined_weak_reference_proxy8'][k],
            'human_semantic_visibility': 'UNKNOWN', 'querypixel_presence': 'NOT_THIS_TARGET'}
        if equal:
            assert a_mass == v_mass and a_count == v_count
            assert ratio is None or ratio == 1.
            assert state != 'LOCAL_EXPECTED_SUPPORT_COVERED_PROXY'
        records.append(rec); disks.append(disk)
    assert records[6]['prior_depth_ray_proxy_separate'] == 'E_EXTERNAL_DEPTH_PROXY'
    assert records[6]['local_support_proxy'] in ['LOCAL_EXPECTED_SUPPORT_VISIBLE_PROXY', 'U_INSUFFICIENT_EXPECTED_SUPPORT']
    csv_path = OUT / 'PER_REFERENCE_SUPPORT.csv'
    with csv_path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    # One public RGB body, drawn solely for saved source GT diagnosis.
    rgb = np.asarray(Image.open(row['image']).convert('RGB'))
    assert rgb.shape == (480, 640, 3)
    expected = rgb.copy(); modal = rgb.copy()
    expected[full] = (.6 * expected[full] + .4 * np.array([25, 210, 255])).astype(np.uint8)
    modal[visible] = (.6 * modal[visible] + .4 * np.array([45, 225, 85])).astype(np.uint8)
    expected[boundary(full)] = [15, 235, 255]
    modal[boundary(visible)] = [70, 255, 100]
    canvas = Image.new('RGB', (1280, 820), '#101820')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), 'Fixed YCB-V instance: virtual-reference local support', fill='white', font=font(25))
    draw.text((20, 49), f'GT diagnostic only | scene0 frame1 inst0 obj8 | rho={radius:.3f}px, sigma={sigma:.3f}px | Mfull=Mvis', fill='#c4d0d9', font=font(17))
    canvas.paste(Image.fromarray(expected), (0, 110)); canvas.paste(Image.fromarray(modal), (640, 110))
    draw.text((15, 84), 'Isolated expected Mfull (cyan) + fixed r footprints', fill='#65edff', font=font(18))
    draw.text((655, 84), 'Visible target Mvis (green) + fixed r footprints', fill='#8eff9e', font=font(18))
    for panel_x in [0, 640]:
        for k, r in enumerate(ref):
            color = '#b8ff5a' if records[k]['support_adequate'] else '#ffc859'
            x, y = r[0] + panel_x, r[1] + 110
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
            draw.line((x - 3, y, x + 3, y), fill='white', width=1)
            draw.line((x, y - 3, x, y + 3), fill='white', width=1)
            draw.text((x + radius + 1, y - 10), str(k), fill='white', font=font(16), stroke_width=2, stroke_fill='black')
    for k, r in enumerate(ref):
        x0 = k * 160
        patch_side = int(np.ceil(radius * 4)) + 1
        left, top = np.floor(r - patch_side / 2).astype(int)
        patch = Image.fromarray(modal).crop((left, top, left + patch_side, top + patch_side)).resize((128, 128), Image.Resampling.NEAREST)
        local = ImageDraw.Draw(patch)
        cx = (r[0] - left) * 128 / patch_side; cy = (r[1] - top) * 128 / patch_side
        rad = radius * 128 / patch_side
        local.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), outline='yellow', width=2)
        local.line((cx - 4, cy, cx + 4, cy), fill='white', width=1)
        local.line((cx, cy - 4, cx, cy + 4), fill='white', width=1)
        canvas.paste(patch, (x0 + 16, 614))
        short = 'VISIBLE support' if records[k]['support_adequate'] else 'U: low support'
        draw.text((x0 + 8, 592), f'k{k}: {short}', fill='#9cffaf' if records[k]['support_adequate'] else '#ffc859', font=font(13))
        draw.text((x0 + 8, 747), f'Apx={records[k]["expected_support_pixel_count_A"]}  Awt={records[k]["weighted_expected_mass_A_pixel_equivalents"]:.2f}', fill='white', font=font(13))
        ratio = records[k]['weighted_visible_over_expected_V_A']
        draw.text((x0 + 8, 766), 'V/A=' + (f'{ratio:.3f}' if ratio is not None else 'undefined'), fill='white', font=font(13))
        if k == 6:
            draw.text((x0 + 8, 786), 'Prior ray E; kept separate', fill='#ff9c8a', font=font(11))
    image_path = OUT / 'REFERENCE_SUPPORT_OVERLAY.png'
    canvas.save(image_path)
    result = {'completed': True, 'scope': 'Same fixed public instance only, r/q unchanged; source GT diagnostic, no network input',
        'instance': [0, 1, 0, 8], 'rho_px': radius, 'sigma_px': sigma,
        'Mfull_equals_Mvis': equal,
        'local_support_counts': {state: sum(r['local_support_proxy'] == state for r in records)
            for state in sorted(set(r['local_support_proxy'] for r in records))},
        'all_positive_A_ratios_are_one': all(r['weighted_visible_over_expected_V_A'] in [None, 1.] for r in records),
        'ray_E_slot6_support_result': records[6], 'per_reference': records,
        'self_convex_not_merged': True,
        'center_mask0_never_forced_to_E': True,
        'native_human_semantic_gold_claimed': False,
        'RGB_body_reads_in_this_new_diagnostic': 1,
        'earlier_read0_reports_or_seals_changed': False,
        'new_forward_fit_current_PnP': 0,
        'outputs': [bound(csv_path), bound(image_path)],
        'bindings': [bound(OUT / 'PROTOCOL.json')],
    }
    result_path = OUT / 'SUBFOOTPRINT_COMPARISON.json'
    save(result_path, result)
    save(OUT / 'SEAL.json', {'completed': True, 'bindings': [bound(x) for x in
        [Path(__file__), OUT / 'PROTOCOL.json', result_path, csv_path, image_path]],
        'new_forward_fit_current_PnP': 0, 'old_artifacts_changed': False})
    print(json.dumps({'rho_px': radius, 'sigma_px': sigma, 'counts': result['local_support_counts'],
        'slot6': records[6], 'result': bound(result_path), 'image': str(image_path)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
