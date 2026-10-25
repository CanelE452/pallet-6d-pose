"""Independent saved public-pilot grid/ledger QA; never imports the producer."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import io
import json
import stat
import sys
import numpy as np
from PIL import Image

ENTRY_SHA = 'fd93307a093ab42ad73949b9fc1dae0a4c94345750296e28ce117177a4066149'
COMPLETE_SHA = '0537fe52cc78f94044c4dcd6d94aa6a54783e51a6ca5bd5d668dd597a18cfe8f'
POLICY_SHA = '793d580f602768c02ddebedf174f6d84d6b1d0dcbc04e71e33cb6afedf72f3e8'
ELIGIBILITY = ('full_window_in_image', 'observed_positive_finite',
               'render_positive_finite', 'observed_stable', 'render_stable',
               'sam_all_target', 'sam_all_background', 'depth_eligible',
               'D', 'E', 'known')
REASONS = ('image_edge', 'observed_invalid', 'render_invalid',
           'observed_discontinuity', 'render_discontinuity', 'sam_window_mixed',
           'D_residual_not_close', 'E_residual_not_closer')
OFFSETS = ((0, 0), (3, 0), (-3, 0), (0, 3), (0, -3),
           (6, 0), (-6, 0), (0, 6), (0, -6))


def safe(value):
    p = Path(value)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(x in str(p).casefold() for x in ('v112', 'cache', '.pyc'))
    for ancestor in reversed([p, *p.parents]):
        try:
            assert not stat.S_ISLNK(ancestor.lstat().st_mode), str(ancestor)
        except FileNotFoundError:
            pass
    return p


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pinned(path, sha):
    p = safe(path)
    data = p.read_bytes()
    assert digest(data) == sha, str(p)
    return data


def bound(binding):
    data = pinned(binding['path'], binding['sha256'])
    assert len(data) == binding['size_bytes'], binding['path']
    return data


def npy(data):
    return np.load(io.BytesIO(data), allow_pickle=False)


def window_truth(x, y, observed, rendered, sam, residual):
    """Direct 121-pixel tests at one r; no producer window/vector helpers."""
    height, width = sam.shape
    flags = dict.fromkeys(ELIGIBILITY, False)
    flags['full_window_in_image'] = 5 <= x < width - 5 and 5 <= y < height - 5
    if flags['full_window_in_image']:
        o = observed[y-5:y+6, x-5:x+6]
        t = rendered[y-5:y+6, x-5:x+6]
        s = sam[y-5:y+6, x-5:x+6]
        flags['observed_positive_finite'] = bool(np.all(np.isfinite(o) & (o > 0)))
        flags['render_positive_finite'] = bool(np.all(np.isfinite(t) & (t > 0)))
        flags['observed_stable'] = flags['observed_positive_finite'] and float(o.max()-o.min()) <= .04
        flags['render_stable'] = flags['render_positive_finite'] and float(t.max()-t.min()) <= .04
        flags['sam_all_target'] = bool(np.all(s))
        flags['sam_all_background'] = not bool(np.any(s))
    depth_ok = all(flags[k] for k in ('full_window_in_image',
        'observed_positive_finite', 'render_positive_finite', 'observed_stable', 'render_stable'))
    close = bool(abs(residual[y, x]) <= .05)
    closer = bool(residual[y, x] <= -.15)
    flags['depth_eligible'] = depth_ok
    flags['D'] = depth_ok and flags['sam_all_target'] and close
    flags['E'] = depth_ok and flags['sam_all_background'] and closer
    assert not (flags['D'] and flags['E'])
    flags['known'] = flags['D'] or flags['E']
    inside = flags['full_window_in_image']
    why = {
        'image_edge': not inside,
        'observed_invalid': inside and not flags['observed_positive_finite'],
        'render_invalid': inside and not flags['render_positive_finite'],
        'observed_discontinuity': inside and flags['observed_positive_finite'] and not flags['observed_stable'],
        'render_discontinuity': inside and flags['render_positive_finite'] and not flags['render_stable'],
        'sam_window_mixed': inside and not flags['sam_all_target'] and not flags['sam_all_background'],
        'D_residual_not_close': depth_ok and flags['sam_all_target'] and not close,
        'E_residual_not_closer': depth_ok and flags['sam_all_background'] and not closer,
    }
    label = 0 if flags['D'] else 1 if flags['E'] else -1
    assert any(why.values()) == (label == -1)
    return label, flags, why


def full_saved_map_checks(labels, e, r, residual):
    """Full saved-map algebra/counts; NOT full independent window recomputation."""
    assert labels.dtype == np.int8 and labels.shape == (480, 640)
    assert set(np.unique(labels).tolist()) <= {-1, 0, 1}
    assert set(e) == set(ELIGIBILITY) and set(r) == set(REASONS)
    assert all(a.shape == labels.shape and a.dtype == bool for a in [*e.values(), *r.values()])
    inside = np.zeros(labels.shape, bool)
    inside[5:-5, 5:-5] = True
    assert np.array_equal(e['full_window_in_image'], inside)
    depth = inside & e['observed_positive_finite'] & e['render_positive_finite'] & e['observed_stable'] & e['render_stable']
    assert np.array_equal(e['depth_eligible'], depth)
    assert np.array_equal(e['D'], depth & e['sam_all_target'] & (np.abs(residual) <= .05))
    assert np.array_equal(e['E'], depth & e['sam_all_background'] & (residual <= -.15))
    assert np.array_equal(labels == 0, e['D']) and np.array_equal(labels == 1, e['E'])
    assert np.array_equal(e['known'], labels >= 0)
    expected = {
        'image_edge': ~inside,
        'observed_invalid': inside & ~e['observed_positive_finite'],
        'render_invalid': inside & ~e['render_positive_finite'],
        'observed_discontinuity': inside & e['observed_positive_finite'] & ~e['observed_stable'],
        'render_discontinuity': inside & e['render_positive_finite'] & ~e['render_stable'],
        'sam_window_mixed': inside & ~e['sam_all_target'] & ~e['sam_all_background'],
        'D_residual_not_close': depth & e['sam_all_target'] & ~(np.abs(residual) <= .05),
        'E_residual_not_closer': depth & e['sam_all_background'] & ~(residual <= -.15),
    }
    for k in REASONS:
        assert np.array_equal(r[k], expected[k]), k
    union = np.zeros(labels.shape, bool)
    for a in r.values():
        union |= a
    assert np.array_equal(union, labels == -1)
    return {'height': 480, 'width': 640, 'total_pixels': labels.size,
            'labels': {'D': int((labels == 0).sum()), 'E': int((labels == 1).sum()), 'U': int((labels == -1).sum())},
            'eligibility': {k: int(a.sum()) for k, a in e.items()},
            'reasons': {k: int(a.sum()) for k, a in r.items()}}


def run(entry_path, complete_path):
    eb = pinned(entry_path, ENTRY_SHA)
    cb = pinned(complete_path, COMPLETE_SHA)
    entry, complete = json.loads(eb), json.loads(cb)
    assert entry['approved_public_aux_label_prototype'] and not entry['training_or_model_inference']
    assert complete['completed'] and complete['PASS'] and complete['entry_sha256'] == ENTRY_SHA
    assert [c['id'] for c in entry['cases']] == ['0000', '0011']
    assert entry['policy']['sha256'] == POLICY_SHA
    policy = json.loads(bound(entry['policy']))
    assert policy['parameters'] == {
        'radius_pixels': 5, 'observed_window_max_minus_min_m': .04,
        'render_window_max_minus_min_m': .04, 'D_abs_observed_minus_render_at_r_max_m': .05,
        'E_observed_minus_render_at_r_max_m': -.15, 'reference_grid_stride': 8,
        'reference_grid_offset': 4, 'q_offsets_xy': [list(o) for o in OFFSETS], 'visual_local_L_pixels': 96}
    files = complete['files']
    assert len(files) == 12 and len({f['path'] for f in files}) == 12
    output = {Path(f['path']).name: bound(f) for f in files}
    assert json.loads(output['STARTED.json']) == {'entry': entry, 'entry_sha256': ENTRY_SHA}
    summary = json.loads(output['SUMMARY.json'])
    assert summary['completed'] and summary['PASS']
    results = []
    for case, saved_case in zip(entry['cases'], summary['cases'], strict=True):
        ident = json.loads(bound(case['identity_QA']))
        assert ident['completed'] and ident['PASS'] and ident['exact_encoded_byte_hits'] == 0
        assert ident['public_RGB_binding']['sha256'] == case['RGB']['sha256']
        bound(case['RGB'])  # Opaque identity only: no RGB pixel decode or view.
        for key in ('geometry_entry', 'geometry_complete'):
            bound(case[key])  # Metadata only; nested pointers never followed.
        depth = np.asarray(Image.open(io.BytesIO(bound(case['DEPTH']))))
        mask = np.asarray(Image.open(io.BytesIO(bound(case['SAM']))))
        assert depth.shape == mask.shape == (480, 640)
        assert np.issubdtype(depth.dtype, np.integer) and depth.min() >= 0 and depth.max() <= 65535
        assert set(np.unique(mask).tolist()) <= {0, 255}
        sam = mask == 255
        observed = depth.astype(np.float64) / 1000
        observed[depth == 0] = np.nan
        rendered = npy(bound(case['T_render_depth']))
        old_residual = npy(bound(case['T_raw_residual']))
        assert rendered.shape == (480, 640) and rendered.dtype == np.float64
        residual = observed - rendered
        assert np.array_equal(old_residual, residual, equal_nan=True)
        cid = case['id']
        labels = npy(output[cid + '_SURFACE_AUX_LABELS_R.npy'])
        with np.load(io.BytesIO(output[cid + '_ELIGIBILITY_REASONS.npz']), allow_pickle=False) as pack:
            assert set(pack.files) == {'eligible_'+k for k in ELIGIBILITY} | {'reason_'+k for k in REASONS}
            e = {k: pack['eligible_'+k] for k in ELIGIBILITY}
            reasons = {k: pack['reason_'+k] for k in REASONS}
        full = full_saved_map_checks(labels, e, reasons, residual)
        saved_full = saved_case['full_pixel_counts']
        for k, v in full.items():
            assert saved_full[k] == v, (cid, k)
        for k in ('SELF_inferred', 'U_is_model_abstention', 'native_gold', 'raw_residual_bias_correction_applied'):
            assert saved_full[k] is False
        assert saved_full['reason_maps_overlap_do_not_sum_as_U_denominator'] is True
        assert saved_full['policy_sha256'] == POLICY_SHA
        refs = []
        for y in range(4, 480, 8):
            for x in range(4, 640, 8):
                label, flags, why = window_truth(x, y, observed, rendered, sam, residual)
                assert int(labels[y, x]) == label, (cid, x, y, 'label')
                for k in ELIGIBILITY:
                    assert bool(e[k][y, x]) == flags[k], (cid, x, y, k)
                for k in REASONS:
                    assert bool(reasons[k][y, x]) == why[k], (cid, x, y, k)
                v = float(residual[y, x])
                refs.append({'reference_grid_index': len(refs), 'r_xy': [x, y],
                    'r_derived_label': label, 'r_derived_state': {-1:'U', 0:'D', 1:'E'}[label],
                    'r_depth_eligible': flags['depth_eligible'], 'r_known': label >= 0,
                    'r_raw_observed_minus_render_m': v if np.isfinite(v) else None,
                    'r_raw_residual_finite': bool(np.isfinite(v)),
                    'r_exclusion_reasons': [k for k in REASONS if why[k]],
                    'native_slot': None, 'SELF_state': 'UNKNOWN'})
        actual_refs = [json.loads(line) for line in io.BytesIO(output[cid + '_REFERENCE_ALL_4800.jsonl'])]
        assert len(refs) == len(actual_refs) == 4800 and actual_refs == refs
        changes = defaultdict(Counter)
        count = 0
        for line in io.BytesIO(output[cid + '_QUERY_ALL_43200.jsonl']):
            row = json.loads(line)
            reference, k = refs[count // 9], count % 9
            x, y = reference['r_xy']; dx, dy = OFFSETS[k]; qx, qy = x+dx, y+dy
            inside = 0 <= qx < 640 and 0 <= qy < 480
            present = bool(sam[qy, qx]) if inside else None
            want = dict(reference, record_id=f'r_y{y}_x{x}_q{k}', q_offset_index=k,
                q_offset_xy=[dx, dy], q_xy=[qx, qy], q_inside_image=inside,
                q_sam_target_present=present, q_pixel_DE_gold=None,
                r_label_copied_as_query_label=False, q_is_public_emulation=True,
                q_is_actual_initializer=False, q_local_L_pixels=96, U_is_model_abstention=False)
            assert row == want, (cid, count, 'query record mismatch')
            bucket = str(max(abs(dx), abs(dy)))
            if inside:
                changes[bucket]['inside'] += 1
                if reference['r_derived_label'] == 0 and present is False:
                    changes[bucket]['r_D_q_SAM_background'] += 1
                if reference['r_derived_label'] == 1 and present is True:
                    changes[bucket]['r_E_q_SAM_target'] += 1
            count += 1
        assert count == 43200
        gc = {str(k): v for k, v in sorted(Counter(r['r_derived_label'] for r in refs).items())}
        qc = {k: dict(v) for k, v in changes.items()}
        assert saved_case['case'] == cid and saved_case['reference_rows'] == 4800 and saved_case['query_rows'] == count
        assert saved_case['reference_grid_counts_D_E_U'] == gc
        assert saved_case['r_vs_q_SAM_discordance_by_max_offset_pixels'] == qc
        results.append({'case': cid, 'independently_recomputed_grid_rows': 4800,
            'independently_recomputed_query_rows': count, 'grid_counts': gc,
            'saved_full_map_counts_verified_by_algebra_not_full_rerender': full,
            'r_vs_q_SAM_discordance_by_max_offset_pixels': qc,
            'all_grid_eligibility_and_reason_values_exact': True,
            'all_raw_residual_pixels_exact_equal_nan': True,
            'original_depth_dtype': str(depth.dtype), 'T_render_dtype': str(rendered.dtype)})
        print(json.dumps({'independent_case_PASS': cid, 'grid_counts': gc, 'queries': count}), flush=True)
    assert summary['current_native_gold_or_model_accuracy_claim'] is False
    assert summary['training_or_model_forward'] == 0 and summary['current223_or_protected_case_reads'] == 0
    assert summary['q_offsets_are_emulated_not_fresh_initializer'] and summary['r_teacher_state_and_q_SAM_presence_are_separate']
    return {'completed': True, 'PASS': True,
        'entry': {'path': str(safe(entry_path)), 'sha256': ENTRY_SHA, 'size_bytes': len(eb)},
        'dataset_complete': {'path': str(safe(complete_path)), 'sha256': COMPLETE_SHA, 'size_bytes': len(cb)},
        'policy': entry['policy'], 'all_12_output_bindings_hash_and_size_verified': files,
        'cases': results, 'production_compiler_import_or_call': False,
        'scope': 'Independent original-depth/render/SAM grid truth plus all r/q ledgers; full saved maps algebra/count consistency.',
        'limitations': ['No independent full-pixel window recompilation or geometry re-render.',
          'This is policy-derived public surface AUX, not physical native/SELF gold or query-pixel D/E gold.',
          'U is teacher exclusion, never model abstention; E reference counts are not model correct counts/recall.',
          'Any coincident historical YCB count is unrelated and supplies no evidence here.',
          'Identity QA encoded-byte no-hit is metadata only and does not prove full capture disjointness.'],
        'access': {'public_RGB_opaque_hash_only': 2, 'RGB_pixels_decoded': 0,
          'public_depth_and_SAM_decoded': 4, 'saved_T_depth_and_residual_loaded': 4,
          'external_inert_metadata_pointers_followed': 0, 'models_or_weights_or_training': 0,
          'current_or_held_or_SourceDEV_case_reads': 0, 'Git_mutations': 0}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True)
    parser.add_argument('--complete', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = safe(args.output_dir)
    out.mkdir(exist_ok=False)
    try:
        result = run(args.entry, args.complete)
        code = safe(Path(__file__).absolute()).read_bytes()
        result['reproducer'] = {'path': str(Path(__file__).absolute()), 'sha256': digest(code), 'size_bytes': len(code)}
        result['runtime'] = {'python': sys.version, 'numpy': np.__version__}
        data = (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode()
        p = safe(out / 'REVIEW.json'); p.write_bytes(data)
        print(json.dumps({'completed': True, 'PASS': True, 'path': str(p), 'sha256': digest(data), 'size_bytes': len(data)}), flush=True)
    except BaseException as exc:
        data = {'completed': False, 'PASS': False, 'error_type': type(exc).__name__, 'error': str(exc), 'automatic_retry': False}
        safe(out / 'FAILURE.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
