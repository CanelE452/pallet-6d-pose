"""Compare only the two named public derived pilots; no geometry or model run."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat

import numpy as np

R = Path(__file__).absolute().parent.parent
NEW = {'path': str(R/'V261_ROOT_PUBLIC_BATCH_TEACHER/group_0p55m_hall_v1/COMPLETE.json'),
       'sha256': '23d8f7c4491e6ee308c4def34d97253e22ba3f2ce79feae58e267415309a10b3',
       'size_bytes': 177659}
OLD = {'path': str(R/'V259_ROOT_PUBLIC_OCCLUSION_PILOT/public_aux_label_pilot_v1/COMPLETE.json'),
       'sha256': '0537fe52cc78f94044c4dcd6d94aa6a54783e51a6ca5bd5d668dd597a18cfe8f',
       'size_bytes': 4140}
FRAMES = ('0000', '0011')


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def safe(path, exists=True):
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts and
         not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc')), 'Lexical path before FS')
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            need(not exists, 'Exact artifact absent')
            continue
        need(not stat.S_ISLNK(mode), 'No artifact symlink')
    return p


def raw(b):
    fd = os.open(safe(b['path']), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        data = f.read()
    need(len(data) == b['size_bytes'] and hashlib.sha256(data).hexdigest() == b['sha256'],
         'Exact declared file SHA256/size')
    return data


def binding(path):
    p = safe(path)
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        data = f.read()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def write(path, value):
    with safe(path, False).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def run(output):
    out = safe(output, False)
    need(not out.exists(), 'Fresh consistency output only')
    out.mkdir()
    results, inputs = [], [NEW, OLD]
    try:
        new, old = json.loads(raw(NEW)), json.loads(raw(OLD))
        need(new['completed'] is True and new['PASS'] is True and
             old['completed'] is True and old['PASS'] is True, 'Completed public producer receipts')
        index_binding = new['outputs']['FRAME_INDEX.json']
        index = json.loads(raw(index_binding))['frames']
        inputs.append(index_binding)
        selected = {r['frame_id']: r for r in index if r['frame_id'] in
                    {'0.55m_hall/'+stem for stem in FRAMES}}
        need(len(selected) == 2, 'Exactly named 0000 and0011 frame identities')
        old_files = {Path(b['path']).name: b for b in old['files']}
        for stem in FRAMES:
            frame_id = '0.55m_hall/'+stem
            frame = selected[frame_id]
            need(frame['completed'] is True and frame['PASS'] is True, 'Selected frame completed')
            directory = frame['relative_directory']
            need(directory == 'frames/F%06d' % frame['frame_index'], 'Literal FRAME_INDEX position mapping')
            gzip_binding = new['outputs'][directory+'/REFERENCE_ALL.jsonl.gz']
            maps_binding = new['outputs'][directory+'/SURFACE_AUX_MAPS.npz']
            original_rows = old_files[stem+'_REFERENCE_ALL_4800.jsonl']
            original_map = old_files[stem+'_SURFACE_AUX_LABELS_R.npy']
            inputs.extend([gzip_binding, maps_binding, original_rows, original_map])
            need(gzip_binding['encoding'] == 'gzip_jsonl' and
                 gzip_binding['gzip_mtime'] == 0 and gzip_binding['gzip_filename'] == '', 'Declared gzip contract')
            encoded = raw(gzip_binding)
            need(encoded[:3] == b'\x1f\x8b\x08' and encoded[4:8] == b'\0'*4 and
                 not (encoded[3] & 8), 'Actual gzip mtime0 and no filename header')
            decoded = gzip.decompress(encoded)
            need(len(decoded) == gzip_binding['uncompressed_size_bytes'] and
                 hashlib.sha256(decoded).hexdigest() == gzip_binding['uncompressed_sha256'],
                 'Exact gzip original-stream SHA256/size')
            prior_bytes = raw(original_rows)
            need(decoded == prior_bytes, 'Reference JSONL original bytes unchanged')
            rows = [json.loads(line) for line in decoded.splitlines()]
            prior = [json.loads(line) for line in prior_bytes.splitlines()]
            need(len(rows) == len(prior) == gzip_binding['records'] == 4800 and
                 all(type(a) is dict and type(b) is dict and a == b for a, b in zip(rows, prior)),
                 'Every reference dictionary, position and value exact')
            with np.load(io.BytesIO(raw(maps_binding)), allow_pickle=False) as packet:
                # Only the authorized labels member is decompressed; no other map is consumed.
                labels = packet['labels']
            old_labels = np.load(io.BytesIO(raw(original_map)), allow_pickle=False)
            need(labels.shape == old_labels.shape == (480, 640) and
                 labels.dtype == old_labels.dtype == np.dtype('int8'), 'Declared full label shape/type')
            need(labels.tobytes(order='C') == old_labels.tobytes(order='C'), 'Full label maps bitwise equal')
            counts = {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
            results.append({'frame_id': frame_id, 'new_frame_index': frame['frame_index'],
                            'new_reference_gzip': gzip_binding, 'old_reference_JSONL': original_rows,
                            'new_surface_maps': maps_binding, 'old_labels_NPY': original_map,
                            'reference_rows': len(rows), 'reference_dicts_exact': True,
                            'reference_original_bytes_exact': True,
                            'gzip_original_stream_sha256': hashlib.sha256(decoded).hexdigest(),
                            'gzip_original_stream_size_bytes': len(decoded),
                            'full_label_shape': list(labels.shape), 'full_label_dtype': str(labels.dtype),
                            'label_data_sha256': hashlib.sha256(labels.tobytes(order='C')).hexdigest(),
                            'full_label_pixels': int(labels.size), 'label_differences': 0,
                            'label_values_counts': counts})
        review = {'completed': True, 'PASS': True, 'code': binding(Path(__file__).absolute()),
                  'new_batch_complete': NEW, 'old_pilot_complete': OLD, 'FRAME_INDEX': index_binding,
                  'exact_read_bindings': inputs, 'frames': results,
                  'reference_dicts_verified': sum(r['reference_rows'] for r in results),
                  'label_pixels_bitwise_verified': sum(r['full_label_pixels'] for r in results),
                  'scope': 'Known two public pilots only. Consistency of saved batch adapter outputs; not an independent geometric teacher validation or all284 proof.',
                  'images_depth_pose_mesh_models_or_private_payloads_read': 0,
                  'other26_frame_payloads_or_other5_groups_read': 0,
                  'new_geometry_render_labels_training_or_threshold_decision': 0,
                  'native_gold_or_model_accuracy_claim': False}
        write(out/'REVIEW.json', review)
        answer = {'completed': True, 'PASS': True, 'REVIEW': binding(out/'REVIEW.json'),
                  'reference_dicts': review['reference_dicts_verified'],
                  'label_pixels': review['label_pixels_bitwise_verified']}
        print(json.dumps(answer, sort_keys=True), flush=True)
        return answer
    except BaseException as exc:
        write(out/'FAILURE.json', {'completed': False, 'PASS': False, 'error_type': type(exc).__name__,
              'error': str(exc), 'verified_frames_before_failure': results,
              'automatic_retry': False, 'exact_read_bindings': inputs})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
