"""Pinned public RGBD surface-AUX batch compiler; no download/model/training."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import io
import json
import platform
import stat
import types
import numpy as np
from PIL import Image, __version__ as PIL_VERSION

PROJECTION_SHA = '820803cfe9e2657a1f8fcedda18b17ef12471ae576ae2b099ef3f6dccd98116a'
COMPILER_SHA = '20e226a7fe8c2b2515c3c5be05d0b4ef785c6de4040625ce5c0b11cd3316746a'
POLICY_SHA = '793d580f602768c02ddebedf174f6d84d6b1d0dcbc04e71e33cb6afedf72f3e8'
INPUT_KEYS = {'RGB', 'DEPTH', 'POSE', 'SAM', 'K', 'OBJ'}


def safe(value):
    p = Path(value)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for a in reversed([p, *p.parents]):
        try:
            assert not stat.S_ISLNK(a.lstat().st_mode), str(a)
        except FileNotFoundError:
            pass
    return p


def check_binding(b):
    assert isinstance(b, dict) and {'path', 'sha256', 'size_bytes'} <= set(b)
    safe(b['path'])
    assert type(b['size_bytes']) is int and b['size_bytes'] >= 0
    assert isinstance(b['sha256'], str) and len(b['sha256']) == 64
    assert all(c in '0123456789abcdef' for c in b['sha256'])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bound(b):
    check_binding(b)
    data = safe(b['path']).read_bytes()
    assert len(data) == b['size_bytes'] and sha(data) == b['sha256'], b['path']
    return data


def binding(path):
    p = safe(path)
    h, size = hashlib.sha256(), 0
    with p.open('rb') as f:
        while block := f.read(1048576):
            h.update(block); size += len(block)
    return {'path': str(p), 'sha256': h.hexdigest(), 'size_bytes': size}


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                       default=json_default) + '\n').encode('utf-8')


def write_json(path, obj):
    p = safe(path)
    with p.open('xb') as f:
        f.write(json_bytes(obj))
    return binding(p)


def write_npz(path, arrays):
    p = safe(path)
    assert all(isinstance(a, np.ndarray) and not a.dtype.hasobject for a in arrays.values())
    with p.open('xb') as f:
        np.savez_compressed(f, **arrays)
    b = binding(p)
    b.update(encoding='lossless_npz', arrays={k: {'shape': list(a.shape), 'dtype': str(a.dtype)} for k, a in arrays.items()})
    return b


def write_gzip_records(path, records):
    p = safe(path)
    h, n, count = hashlib.sha256(), 0, 0
    with p.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=6, mtime=0) as stream:
            for row in records:
                line = (json.dumps(row, ensure_ascii=False, separators=(',', ':'),
                                   allow_nan=False, default=json_default) + '\n').encode('utf-8')
                h.update(line); n += len(line); count += 1
                stream.write(line)
    b = binding(p)
    b.update(encoding='gzip_jsonl', gzip_mtime=0, gzip_filename='', records=count,
             uncompressed_sha256=h.hexdigest(), uncompressed_size_bytes=n)
    return b


def module_from_binding(b, name, expected):
    assert b['sha256'] == expected
    source = bound(b)
    module = types.ModuleType(name)
    module.__file__ = b['path']
    exec(compile(source, b['path'], 'exec'), module.__dict__)
    return module


def no_geometry(hw, exc):
    outputs = {k: {'depth': None, 'support': None, 'observed_minus_rendered': None} for k in ('T', 'inverse_T')}
    reason = {'status': 'GEOMETRY_VALIDATION_UNKNOWN', 'error_type': type(exc).__name__, 'error': str(exc)}
    report = {'completed': False, 'interpretations': {k: {'render': dict(reason), 'continuous_depth_and_overlap': None} for k in outputs},
              'pose_hypotheses': ['T', 'inverse_T'], 'winner_selected': None,
              'registration_or_pose_correctness_certified': False}
    return outputs, report


def process_frame(frame, geom, compiler, policy, directory, shared):
    """All listed frames reach this API; labels never select a frame/pose."""
    root = safe(directory)
    root.mkdir()
    inputs = frame['inputs']
    # Verify every frame binding before using its decoded fields. Shared K/OBJ
    # are verified once per identical path/SHA/size and never changed thereafter.
    raw = {k: bound(inputs[k]) for k in ('RGB', 'DEPTH', 'POSE', 'SAM')}
    assets = {}
    for key in ('K', 'OBJ'):
        b = inputs[key]; identity = (key, b['path'], b['sha256'], b['size_bytes'])
        if identity not in shared:
            data = bound(b)
            shared[identity] = (np.loadtxt(io.StringIO(data.decode('utf-8')), dtype=np.float64)
                                if key == 'K' else geom.parse_obj_text(data))
        assets[key] = shared[identity]
    with Image.open(io.BytesIO(raw['RGB'])) as rgb:
        assert rgb.mode == 'RGB'
        rgb_hw = (rgb.height, rgb.width)  # Header only; RGB is not a teacher input.
    depth = np.asarray(Image.open(io.BytesIO(raw['DEPTH'])))
    sam_encoded = np.asarray(Image.open(io.BytesIO(raw['SAM'])))
    assert depth.ndim == 2 and depth.shape == sam_encoded.shape == rgb_hw and min(rgb_hw) > 0
    assert np.issubdtype(depth.dtype, np.integer) and depth.min() >= 0 and depth.max() <= 65535
    assert set(np.unique(sam_encoded).tolist()) <= {0, 255}
    sam = sam_encoded == 255
    observed = depth.astype(np.float64) / 1000
    observed[depth == 0] = np.nan
    T = np.loadtxt(io.StringIO(raw['POSE'].decode('utf-8')), dtype=np.float64)
    vertices, faces, mesh_report = assets['OBJ']
    try:
        outputs, geometry = geom.diagnostic_pair(vertices, faces, T, assets['K'], rgb_hw, observed, sam)
    except (ValueError, np.linalg.LinAlgError) as exc:
        outputs, geometry = no_geometry(rgb_hw, exc)
    main = outputs['T']
    main_missing = main['depth'] is None
    rendered = np.full(rgb_hw, np.nan, np.float64) if main_missing else main['depth']
    assert rendered.shape == rgb_hw and rendered.dtype == np.float64
    compiled = compiler.compile_surface_labels(observed, rendered, sam, policy)
    if not main_missing:
        assert np.array_equal(main['observed_minus_rendered'], compiled['raw_center_residual_m'], equal_nan=True)
    geometry_status = ('T_RENDER_UNAVAILABLE_ALL_U' if main_missing else
                       'T_NO_POSITIVE_SUPPORT_ALL_U' if not np.any(np.isfinite(rendered)) else 'T_RENDERED')
    if geometry_status != 'T_RENDERED':
        assert np.all(compiled['labels'] == -1)
    refs = compiler.reference_grid_records(compiled, policy)
    queries = compiler.emulated_query_records(compiled, sam, policy)
    assert len(queries) == len(refs) * 9
    maps = {'labels': compiled['labels']}
    maps.update({'eligible_' + k: a for k, a in compiled['eligibility'].items()})
    maps.update({'reason_' + k: a for k, a in compiled['reasons'].items()})
    files = {}
    files['T_GEOMETRY_MAPS.npz'] = write_npz(root / 'T_GEOMETRY_MAPS.npz', {
        'T_RENDER_DEPTH_M': rendered,
        'T_OBSERVED_MINUS_RENDER_M': compiled['raw_center_residual_m']})
    files['SURFACE_AUX_MAPS.npz'] = write_npz(root / 'SURFACE_AUX_MAPS.npz', maps)
    files['REFERENCE_ALL.jsonl.gz'] = write_gzip_records(root / 'REFERENCE_ALL.jsonl.gz', refs)
    files['QUERY_ALL.jsonl.gz'] = write_gzip_records(root / 'QUERY_ALL.jsonl.gz', queries)
    changes = defaultdict(Counter)
    for q in queries:
        dx, dy = q['q_offset_xy']; bucket = str(max(abs(dx), abs(dy)))
        if q['q_inside_image']:
            changes[bucket]['inside'] += 1
            if q['r_derived_label'] == 0 and q['q_sam_target_present'] is False:
                changes[bucket]['r_D_q_SAM_background'] += 1
            if q['r_derived_label'] == 1 and q['q_sam_target_present'] is True:
                changes[bucket]['r_E_q_SAM_target'] += 1
    summary = {'case': frame['frame_id'], 'full_pixel_counts': compiled['counts'],
        'reference_grid_counts_D_E_U': {str(k): int(v) for k, v in sorted(Counter(r['r_derived_label'] for r in refs).items())},
        'reference_rows': len(refs), 'query_rows': len(queries),
        'r_vs_q_SAM_discordance_by_max_offset_pixels': {k: dict(v) for k, v in changes.items()},
        'geometry_status': geometry_status, 'main_pose_convention': 'T object-to-camera; never selected from diagnostics',
        'inverse_T_full_arrays_saved': False, 'inverse_T_arrays_status': 'NOT_SAVED_DIAGNOSTIC_STATS_ONLY',
        'main_missing_render_is_explicit_NaN_not_fabricated_geometry': main_missing,
        'teacher_namespace': 'PUBLIC_REAL_SURFACE_DEPTH_SAM_AUX',
        'training_admission': False, 'native_gold': False, 'U_is_model_abstention': False}
    geometry.update(main_pose_used_for_labels='T', inverse_T_full_arrays_saved=False,
                    inverse_T_arrays_status='NOT_SAVED_DIAGNOSTIC_STATS_ONLY', geometry_status=geometry_status,
                    inputs=inputs, public_provenance=frame.get('provenance', {}), mesh_format=mesh_report)
    files['GEOMETRY_DIAGNOSTIC.json'] = write_json(root / 'GEOMETRY_DIAGNOSTIC.json', geometry)
    files['SUMMARY.json'] = write_json(root / 'SUMMARY.json', summary)
    complete = {'completed': True, 'PASS': True, 'frame_id': frame['frame_id'],
        'geometry_status': geometry_status, 'outputs': files, 'inputs': inputs,
        'all_reference_and_query_records_retained': True, 'training_admission': False,
        'geometry_or_native_gold_certified': False, 'compiler_policy_sha256': POLICY_SHA}
    complete_binding = write_json(root / 'COMPLETE.json', complete)
    return summary, geometry['interpretations'], complete_binding, files


def validate_entry(path):
    p = safe(path); raw = p.read_bytes(); entry = json.loads(raw)
    assert entry['approved_public_surface_batch_teacher'] is True
    assert entry['training_or_model_inference'] is False
    assert entry['driver_sha256'] == sha(safe(Path(__file__).absolute()).read_bytes())
    for key, expected in (('projection_code', PROJECTION_SHA), ('compiler_code', COMPILER_SHA), ('policy', POLICY_SHA)):
        check_binding(entry[key]); assert entry[key]['sha256'] == expected
    assert isinstance(entry['frames'], list) and entry['frames']
    ids = []
    for frame in entry['frames']:
        assert type(frame['frame_id']) is str and frame['frame_id']
        ids.append(frame['frame_id'])
        assert set(frame['inputs']) == INPUT_KEYS
        for b in frame['inputs'].values():
            check_binding(b)
        assert isinstance(frame.get('provenance', {}), dict)
    assert len(ids) == len(set(ids))
    safe(entry['output_directory'])
    return entry, raw, {'path': str(p), 'sha256': sha(raw), 'size_bytes': len(raw)}


def run(entry, raw, entry_binding):
    out = safe(entry['output_directory'])
    assert not out.exists(), 'Fresh output required; no automatic retry or overwrite.'
    out.mkdir()
    records, all_outputs = [], {}
    try:
        write_json(out / 'STARTED.json', {'entry': entry, 'entry_binding': entry_binding,
            'listed_frame_ids': [f['frame_id'] for f in entry['frames']], 'automatic_retry': False})
        geom = module_from_binding(entry['projection_code'], 'pinned_public_geometry', PROJECTION_SHA)
        compiler = module_from_binding(entry['compiler_code'], 'pinned_surface_labels', COMPILER_SHA)
        policy = json.loads(bound(entry['policy']))
        compiler.validate_policy(policy)
        frames_dir = safe(out / 'frames'); frames_dir.mkdir()
        shared = {}
        totals = {'labels': Counter(), 'grid_labels': Counter(), 'eligibility': Counter(), 'reasons': Counter(),
                  'reference_rows': 0, 'query_rows': 0}
        for index, frame in enumerate(entry['frames']):
            rel = f'frames/F{index:06d}'; directory = safe(out / rel)
            record = {'frame_index': index, 'frame_id': frame['frame_id'], 'relative_directory': rel,
                      'inputs': frame['inputs'], 'provenance': frame.get('provenance', {})}
            try:
                summary, interpretations, complete, files = process_frame(frame, geom, compiler, policy, directory, shared)
                record.update(completed=True, PASS=True, summary=summary, geometry_interpretations=interpretations, complete=complete)
                for name, b in files.items():
                    all_outputs[rel + '/' + name] = b
                all_outputs[rel + '/COMPLETE.json'] = complete
                totals['labels'].update(summary['full_pixel_counts']['labels'])
                totals['grid_labels'].update(summary['reference_grid_counts_D_E_U'])
                totals['eligibility'].update(summary['full_pixel_counts']['eligibility'])
                totals['reasons'].update(summary['full_pixel_counts']['reasons'])
                totals['reference_rows'] += summary['reference_rows']; totals['query_rows'] += summary['query_rows']
            except Exception as exc:
                if not directory.exists():
                    directory.mkdir()
                failure = {'completed': False, 'PASS': False, 'frame_id': frame['frame_id'],
                    'error_type': type(exc).__name__, 'error': str(exc), 'inputs': frame['inputs'],
                    'status': 'FRAME_INTEGRITY_OR_DECODE_OR_PROCESSING_FAILURE_NOT_U_LABELS',
                    'partial_outputs_preserved': True, 'automatic_retry': False, 'training_admission': False}
                fb = write_json(directory / 'FRAME_FAILURE.json', failure)
                all_outputs[rel + '/FRAME_FAILURE.json'] = fb
                record.update(completed=False, PASS=False, failure=fb, status=failure['status'])
            records.append(record)
            print(json.dumps({'completed_frames': index+1, 'listed_frames': len(entry['frames']),
                'frame_id': frame['frame_id'], 'frame_PASS': record['PASS'],
                'geometry_status': record.get('summary', {}).get('geometry_status')}), flush=True)
        okay = all(r['PASS'] for r in records)
        status_counts = dict(Counter(r.get('summary', {}).get('geometry_status', 'FRAME_FAILURE') for r in records))
        aggregate = {'completed': True, 'PASS': okay, 'listed_frames': len(entry['frames']),
            'frames_processed_without_integrity_failure': sum(r['PASS'] for r in records),
            'frame_status_counts': status_counts,
            'totals_over_compiled_frames_only': {k: dict(v) if isinstance(v, Counter) else v for k, v in totals.items()},
            'all_listed_frames_retained': True, 'policy_sha256': POLICY_SHA,
            'main_pose_convention': 'T fixed; inverse_T stats retained, no per-frame winner/pose selection',
            'inverse_T_full_arrays_saved': False, 'no_result_based_filtering': True,
            'teacher_namespace': 'PUBLIC_REAL_SURFACE_DEPTH_SAM_AUX', 'training_admission': False,
            'native_gold_or_model_accuracy_claim': False, 'U_is_model_abstention': False,
            'capture_groups_are_not_certified_independent_sessions': True,
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'Pillow': PIL_VERSION}}
        for name, value in (('FRAME_INDEX.json', {'frames': records}), ('SUMMARY.json', aggregate)):
            all_outputs[name] = write_json(out / name, value)
        all_outputs['STARTED.json'] = binding(out / 'STARTED.json')
        complete = {'completed': True, 'PASS': okay, 'entry': entry_binding, 'outputs': all_outputs,
            'listed_frames': len(records), 'all_listed_frame_ids_retained': True,
            'training_admission': False, 'model_or_training_or_downloads': 0,
            'projection_code': entry['projection_code'], 'compiler_code': entry['compiler_code'], 'policy': entry['policy']}
        cb = write_json(out / 'COMPLETE.json', complete)
        if not okay:
            write_json(out / 'FAILURE.json', {'completed': True, 'PASS': False,
                'reason': 'At least one frame integrity/decode/process failure; every listed frame was attempted once.',
                'frame_index': all_outputs['FRAME_INDEX.json'], 'automatic_retry': False})
        return {'completed': True, 'PASS': okay, 'COMPLETE': cb, 'listed_frames': len(records), 'status_counts': status_counts}
    except BaseException as exc:
        if not safe(out / 'FAILURE.json').exists():
            write_json(out / 'FAILURE.json', {'completed': False, 'PASS': False,
                'error_type': type(exc).__name__, 'error': str(exc),
                'listed_frame_ids': [f['frame_id'] for f in entry['frames']],
                'attempted_frame_ids': [r['frame_id'] for r in records],
                'partial_outputs_preserved': True, 'automatic_retry': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True)
    args = parser.parse_args()
    entry, raw, eb = validate_entry(args.entry)
    result = run(entry, raw, eb)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result['PASS']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
