"""Compile public surface AUX labels and full emulated-query ledgers, no model."""
from pathlib import Path
import collections
import hashlib
import io
import json
import stat
import sys
import types


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for a in reversed([p, *p.parents]):
        try:
            assert not stat.S_ISLNK(a.lstat().st_mode), str(a)
        except FileNotFoundError:
            pass
    return p


def read_bound(binding):
    b = safe(Path(binding['path'])).read_bytes()
    assert len(b) == binding['size_bytes'] and hashlib.sha256(b).hexdigest() == binding['sha256']
    return b


def run(entry_path):
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eb = safe(entry_path).read_bytes()
    e = json.loads(eb)
    assert e['approved_public_aux_label_prototype'] is True and e['training_or_model_inference'] is False
    assert e['driver_sha256'] == hashlib.sha256(safe(Path(__file__)).read_bytes()).hexdigest()
    policy = json.loads(read_bound(e['policy']))
    assert e['policy']['sha256'] == '793d580f602768c02ddebedf174f6d84d6b1d0dcbc04e71e33cb6afedf72f3e8'
    toy = json.loads(read_bound(e['compiler_toy_QA']))
    assert toy['PASS']
    code = read_bound(e['compiler_code'])
    module = types.ModuleType('closed_v259_surface_label_compiler')
    module.__file__ = e['compiler_code']['path']
    sys.modules[module.__name__] = module
    exec(compile(code, module.__file__, 'exec'), module.__dict__)
    out = safe(Path(e['output_directory']))
    assert not out.exists()
    out.mkdir()
    generated = []

    def json_file(name, obj):
        p = safe(out / name)
        with p.open('x') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        generated.append(p)

    def jsonlines(name, records):
        p = safe(out / name)
        with p.open('x') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
        generated.append(p)

    json_file('STARTED.json', {'entry': e, 'entry_sha256': hashlib.sha256(eb).hexdigest()})
    try:
        summaries, contexts, selections = [], [], []
        for case in e['cases']:
            ident = json.loads(read_bound(case['identity_QA']))
            assert ident['PASS'] and ident['exact_encoded_byte_hits'] == 0
            assert ident['public_RGB_binding']['sha256'] == case['RGB']['sha256']
            rgb_image = Image.open(io.BytesIO(read_bound(case['RGB'])))
            assert rgb_image.mode == 'RGB'
            rgb = np.asarray(rgb_image)
            depth = np.asarray(Image.open(io.BytesIO(read_bound(case['DEPTH']))))
            sam = np.asarray(Image.open(io.BytesIO(read_bound(case['SAM']))))
            assert rgb.shape == (480, 640, 3) and depth.shape == sam.shape == (480, 640)
            assert np.issubdtype(depth.dtype, np.integer) and depth.min() >= 0 and depth.max() <= 65535
            assert set(np.unique(sam).tolist()) <= {0, 255}
            observed = depth.astype(np.float64) / 1000
            observed[depth == 0] = np.nan
            rendered = np.load(io.BytesIO(read_bound(case['T_render_depth'])), allow_pickle=False)
            previous_residual = np.load(io.BytesIO(read_bound(case['T_raw_residual'])), allow_pickle=False)
            compiled = module.compile_surface_labels(observed, rendered, sam == 255, policy)
            labels = compiled['labels']
            assert labels.shape == (480, 640) and labels.dtype == np.int8
            assert set(np.unique(labels).tolist()) <= {-1, 0, 1}
            assert np.array_equal(compiled['raw_center_residual_m'], previous_residual, equal_nan=True)
            refs = list(module.reference_grid_records(compiled, policy))
            queries = list(module.emulated_query_records(compiled, sam == 255, policy))
            assert len(refs) == 4800 and len(queries) == 43200
            prefix = case['id']
            jsonlines(prefix + '_REFERENCE_ALL_4800.jsonl', refs)
            jsonlines(prefix + '_QUERY_ALL_43200.jsonl', queries)
            p = safe(out / (prefix + '_SURFACE_AUX_LABELS_R.npy'))
            with p.open('xb') as f:
                np.save(f, labels, allow_pickle=False)
            generated.append(p)
            p = safe(out / (prefix + '_ELIGIBILITY_REASONS.npz'))
            with p.open('xb') as f:
                np.savez_compressed(f, **{'eligible_' + k: v for k, v in compiled['eligibility'].items()},
                                    **{'reason_' + k: v for k, v in compiled['reasons'].items()})
            generated.append(p)
            counts = collections.Counter(int(r['r_derived_label']) for r in refs)
            changes = collections.defaultdict(lambda: collections.Counter())
            for q in queries:
                dx, dy = int(q['q_xy'][0] - q['r_xy'][0]), int(q['q_xy'][1] - q['r_xy'][1])
                bucket = str(max(abs(dx), abs(dy)))
                if q['q_inside_image']:
                    changes[bucket]['inside'] += 1
                    if q['r_derived_label'] == 0 and q['q_sam_target_present'] is False:
                        changes[bucket]['r_D_q_SAM_background'] += 1
                    if q['r_derived_label'] == 1 and q['q_sam_target_present'] is True:
                        changes[bucket]['r_E_q_SAM_target'] += 1
            summaries.append({'case': prefix, 'full_pixel_counts': compiled['counts'],
                              'reference_grid_counts_D_E_U': {str(k): int(v) for k, v in sorted(counts.items())},
                              'reference_rows': len(refs), 'query_rows': len(queries),
                              'r_vs_q_SAM_discordance_by_max_offset_pixels': {k: dict(v) for k, v in changes.items()}})
            contexts.append((prefix, rgb, rendered, compiled, refs, sam))
            for label in (0, 1):
                candidates = [r for r in refs if r['r_derived_label'] == label]
                candidates.sort(key=lambda r: hashlib.sha256((prefix + '|' + str(r['r_xy'])).encode()).hexdigest())
                limit = 4 if prefix == '0000' and label == 1 else 1
                for record in candidates[:limit]:
                    selections.append((prefix, rgb, record, sam))
        fig, axes = plt.subplots(len(contexts), 2, figsize=(12, 5 * len(contexts)), squeeze=False, constrained_layout=True)
        for row, (prefix, rgb, rendered, compiled, refs, sam) in enumerate(contexts):
            axes[row, 0].imshow(rgb)
            for label, color, name in ((0, 'lime', 'D'), (1, 'orange', 'E')):
                points = np.asarray([r['r_xy'] for r in refs if r['r_derived_label'] == label], dtype=float).reshape(-1, 2)
                if len(points):
                    axes[row, 0].scatter(points[:, 0], points[:, 1], c=color, s=5, label=name + ' AUX')
            axes[row, 0].legend(loc='upper left')
            axes[row, 0].set_title('Public ' + prefix + ': confident reference points; no model predictions')
            im = axes[row, 1].imshow(compiled['raw_center_residual_m'], cmap='coolwarm', vmin=-1, vmax=0.1)
            fig.colorbar(im, ax=axes[row, 1], label='observed minus rendered Z (m), display clipped')
            axes[row, 1].set_title('Raw residual; unknown/invalid is never a successful model abstention')
            for ax in axes[row]:
                ax.set_axis_off()
        p = safe(out / 'PUBLIC_AUX_LABELS_OVERVIEW.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        generated.append(p)
        if selections:
            fig, axes = plt.subplots(len(selections), 3, figsize=(10, 3.1 * len(selections)), squeeze=False, constrained_layout=True)
            for row, (prefix, rgb, record, sam) in enumerate(selections):
                rx, ry = record['r_xy']
                state = 'D' if record['r_derived_label'] == 0 else 'E'
                for col, dx in enumerate((0, 3, 6)):
                    qx, qy = rx + dx, ry
                    ax = axes[row, col]
                    ax.imshow(rgb, interpolation='nearest')
                    ax.set_facecolor('black')
                    ax.set_xlim(qx - 48, qx + 48)
                    ax.set_ylim(qy + 48, qy - 48)
                    ax.plot(qx, qy, '+', color='cyan', markersize=10)
                    ax.plot(rx, ry, 'o', markerfacecolor='none', markeredgecolor='lime' if state == 'D' else 'orange', markersize=8)
                    presence = bool(sam[qy, qx] == 255) if 0 <= qx < 640 and 0 <= qy < 480 else None
                    ax.set_title(prefix + ' r=' + state + ', emulated q dx=' + str(dx) + ', q SAM=' + str(presence))
                    ax.set_axis_off()
            fig.suptitle('96-source-pixel visual windows: cyan q, ring r; no model input/accuracy claim')
            p = safe(out / 'PUBLIC_R_Q_CROP_DIAGNOSTIC.png')
            fig.savefig(p, dpi=150)
            plt.close(fig)
            generated.append(p)
        json_file('SUMMARY.json', {'completed': True, 'PASS': True, 'cases': summaries,
                                  'teacher_namespace': 'PUBLIC_REAL_SURFACE_DEPTH_SAM_AUX',
                                  'current_native_gold_or_model_accuracy_claim': False,
                                  'training_or_model_forward': 0, 'current223_or_protected_case_reads': 0,
                                  'q_offsets_are_emulated_not_fresh_initializer': True,
                                  'r_teacher_state_and_q_SAM_presence_are_separate': True,
                                  'boundary_and_virtual_native_semantic_visibility_remains_unresolved': True,
                                  'selection_rule_for_visuals': 'Per case/state SHA-ranked references, up to4 E from clearpilot if present, otherwise first1 per case/state; show dx0/3/6. All rows preserved.'})
        files = []
        for p in generated:
            b = safe(p).read_bytes()
            files.append({'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)})
        json_file('COMPLETE.json', {'completed': True, 'PASS': True, 'files': files,
                                   'entry_sha256': hashlib.sha256(eb).hexdigest(),
                                   'new_model_training_or_inference': 0})
        print(json.dumps({'completed': True, 'PASS': True, 'cases': summaries,
                          'outputs': len(files)}, allow_nan=False), flush=True)
    except BaseException as exc:
        json_file('FAILURE.json', {'completed': False, 'PASS': False, 'error': str(exc),
                                  'automatic_retry': False})
        raise


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry', type=Path, required=True)
    run(ap.parse_args().entry)
