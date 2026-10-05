"""Offline source/time audit and prediction-blind fixed review plan. No control imports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ISO = HERE / 'evaluation_checkout'
OUT = ROOT / 'data/pallet/results' / HERE.name
CAPTURE = ROOT / 'extracted/depth_cam/rec'
REFERENCE_COMMIT = '7e136fca834d97b52f63be696e4fcc9cbb8bd77e'
SOURCE_ROOTS = [ISO, ROOT, Path('C:/Users/DELL/Documents/GitHub/pallet-6d-pose'),
                Path('C:/Users/DELL/Documents/GitHub/pallet-6d-before'),
                Path('C:/Users/DELL/Documents/GitHub/pallet-6d-posebbefore')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value, frozen=False):
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists():
        if path.read_bytes() != data:
            raise RuntimeError('Refusing to replace frozen artifact: ' + str(path))
        return
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_bytes(data)
    tmp.replace(path)


def relative(path):
    p = Path(path).resolve()
    for i, root in enumerate(SOURCE_ROOTS):
        if p.is_relative_to(root.resolve()):
            return {'root_id': 'source_' + str(i), 'relative_path': p.relative_to(root.resolve()).as_posix()}
    return {'root_id': 'other_explicit_input', 'filename': p.name}


def modules():
    sys.path.insert(0, str(ISO))
    from scripts.research.pallet_n3_completion_v3 import lifter, lifter_run
    return lifter, lifter_run


def source_bindings(lr):
    records = []
    for name, rel, digest, size in lr.LEGACY_BINDINGS:
        candidates = []
        for root in SOURCE_ROOTS:
            path = root / rel
            if not path.is_file():
                continue
            observed = sha(path)
            # Windows checkout CRLF is not the historical source contract.
            # Restore only our isolated source copy to the exact committed blob.
            if root == ISO and observed != digest:
                blob = subprocess.run(['git', '-C', str(ISO), 'show', REFERENCE_COMMIT + ':' + rel.as_posix()],
                                      capture_output=True, check=False)
                if blob.returncode == 0 and len(blob.stdout) == size and hashlib.sha256(blob.stdout).hexdigest() == digest:
                    path.write_bytes(blob.stdout)
                    observed = sha(path)
            candidates.append({**relative(path), 'bytes': path.stat().st_size, 'sha256': observed,
                               'matches': observed == digest and path.stat().st_size == size})
        match = next((r for r in candidates if r['matches']), None)
        records.append({'name': name, 'relative_path': rel.as_posix(), 'expected_sha256': digest,
                        'expected_bytes': size, 'status': 'VERIFIED_COMPLETE' if match else 'BLOCKED_CONTRACT',
                        'candidates': candidates})
    return records


def choose_samples(rows, count=30):
    times = [float(r['camera_sensor_timestamp_ms']) for r in rows]
    if not times or any(not math.isfinite(t) for t in times) or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('Invalid sensor timeline')
    targets = [times[0] + (i + .5) * (times[-1] - times[0]) / count for i in range(count)]
    selected = []
    last = -1
    for bin_i, target in enumerate(targets):
        # A fixed duplicate rule: keep only first occurrence, never replace with nicer frame.
        index = min(range(len(times)), key=lambda i: (abs(times[i] - target), i))
        if index <= last:
            continue
        selected.append({'bin_index': bin_i, 'saved_frame_index': index, 'target_sensor_timestamp_ms': target,
                         'repeat_review': (len(selected) + 1) % 5 == 0})
        last = index
    return selected


def audit():
    started = time.perf_counter()
    lf, lr = modules()
    OUT.mkdir(parents=True, exist_ok=True)
    binding_records = source_bindings(lr)
    sessions = []
    allframes = []
    samples = []
    # Read timing and freeze identities before opening any prediction outputs.
    selections = {}
    for sid in lf.USABLE_SESSION_IDS:
        paths = lf.session_paths(CAPTURE, sid)
        rows = lf._read_timing_rows(paths['timing'])[35:]
        if len(rows) != lf.EXPECTED_FACTS[sid]['video_frames']:
            raise RuntimeError('Timing count drift: ' + sid)
        selections[sid] = choose_samples(rows)
    selection = {'schema_version': 'lifter_time_selection_v1', 'source_kind': 'machine_proposed',
                 'rule': '30 equal sensor-time bins; nearest stored frame to midpoint; tie earlier; keep first duplicate only',
                 'predictions_read_before_selection': False, 'legacy_analysis_exposure': True,
                 'source_video_bindings': {sid: lf.EXPECTED_FILES[sid]['raw_mp4'][1] for sid in lf.USABLE_SESSION_IDS},
                 'sessions': selections}
    dump(OUT / 'FIXED_REVIEW_SELECTION.json', selection, frozen=True)
    for sid in lf.USABLE_SESSION_IDS:
        paths = lf.session_paths(CAPTURE, sid)
        files = []
        for role, p in paths.items():
            n, digest = lf.EXPECTED_FILES[sid][role]
            files.append({'role': role, 'filename': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p),
                          'expected_bytes': n, 'expected_sha256': digest,
                          'matches': p.stat().st_size == n and sha(p) == digest})
        if not all(f['matches'] for f in files):
            dump(OUT / 'INPUT_DRIFT.json', {'session_id': sid, 'files': files})
            raise RuntimeError('Frozen input hash/size drift: ' + sid)
        rows_all = lf._read_timing_rows(paths['timing'])
        rows = rows_all[35:]
        meta = json.loads(paths['meta'].read_text(encoding='utf-8-sig'))
        camera = lr.camera_contract(meta)
        selected = {s['saved_frame_index']: s for s in selections[sid]}
        cap = cv2.VideoCapture(str(paths['raw_mp4']))
        if not cap.isOpened():
            raise RuntimeError('Raw failed to open: ' + sid)
        fps_diagnostic = cap.get(cv2.CAP_PROP_FPS)
        decoded = 0
        firstlast = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if decoded >= len(rows) or image.shape != (480, 640, 3):
                raise RuntimeError('Decode count/resolution drift: ' + sid)
            row = rows[decoded]
            frame_id = sid + ':' + str(decoded)
            rec = {'frame_id': frame_id, 'session_id': sid, 'saved_frame_index': decoded,
                   'timing_row_index': decoded + 35, 'frame_i': int(row['frame_i']),
                   'camera_frame_number': int(row['camera_frame_number']),
                   'camera_sensor_timestamp_ms': float(row[lf.SENSOR_TIME_FIELD]),
                   'camera_timestamp_domain': row['camera_timestamp_domain'],
                   'camera_input_host_mono_ms': float(row['camera_input_host_mono_ms']),
                   'width': 640, 'height': 480, 'raw_video_sha256': lf.EXPECTED_FILES[sid]['raw_mp4'][1],
                   'decoded_bgr_sha256': hashlib.sha256(image.tobytes()).hexdigest()}
            allframes.append(rec)
            if decoded in selected:
                imagepath = OUT / 'review/frames' / (sid + '_' + str(decoded).zfill(5) + '.png')
                imagepath.parent.mkdir(parents=True, exist_ok=True)
                good, png = cv2.imencode('.png', image)
                if not good:
                    raise RuntimeError('PNG encode failed')
                if imagepath.exists() and imagepath.read_bytes() != png.tobytes():
                    raise RuntimeError('Review image changed on repeat')
                imagepath.write_bytes(png.tobytes())
                roundtrip = cv2.imdecode(np.frombuffer(imagepath.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if not np.array_equal(roundtrip, image):
                    raise RuntimeError('PNG pixel roundtrip mismatch')
                samples.append({**rec, **selected[decoded], 'image_path': 'frames/' + imagepath.name,
                                'image_sha256': sha(imagepath), 'target_identity_status': 'human_confirmation_required'})
            if decoded in (0, len(rows)//2, len(rows)-1):
                firstlast.append(rec)
            decoded += 1
        cap.release()
        if decoded != len(rows):
            raise RuntimeError('Sequential decode did not reach expected last frame: ' + sid)
        times = [float(r[lf.SENSOR_TIME_FIELD]) for r in rows]
        numbers = [int(r['camera_frame_number']) for r in rows]
        gaps = [{'after_saved_frame_index': i, 'sensor_interval_ms': times[i+1]-times[i],
                 'missing_camera_frame_numbers': numbers[i+1]-numbers[i]-1}
                for i in range(len(rows)-1) if numbers[i+1]-numbers[i] > 1]
        sessions.append({'session_id': sid, 'status': 'VERIFIED_COMPLETE', 'files': files,
                         'sequentially_decoded_frames': decoded, 'timing_rows': len(rows_all),
                         'video_timing_row_offset': 35, 'mapping_basis': 'exact legacy file hashes + v3 recorded writer audit; first/middle/last checked',
                         'mapping_checkpoints': firstlast, 'sensor_duration_s': (times[-1]-times[0])/1000,
                         'camera_timestamp_domains': sorted({r['camera_timestamp_domain'] for r in rows}),
                         'duplicate_sensor_timestamps': sum(a == b for a,b in zip(times,times[1:])),
                         'duplicate_camera_frame_numbers': sum(a == b for a,b in zip(numbers,numbers[1:])),
                         'backwards_sensor_timestamps': sum(b < a for a,b in zip(times,times[1:])),
                         'backwards_camera_frame_numbers': sum(b < a for a,b in zip(numbers,numbers[1:])),
                         'camera_number_gaps': gaps, 'camera': camera,
                         'container_fps_diagnostic_only': fps_diagnostic, 'nominal_stream_fps': meta['stream_fps'],
                         'recording_model_filename': meta['model'], 'recording_model_hash_at_capture': None,
                         'dimensions_source': 'hash-bound session metadata and geometry registry; independent measurement/physical individual not reverified',
                         'pose_csv_source_kind': 'operational_model_prediction',
                         'origin': 'camera optical centre; offline centroid pose differs from operational front-plane alignment',
                         'independent_physical_reference': None})
        print('AUDIT', sid, decoded, 'samples', len(selected), flush=True)
    # Fifth unreadable video stays outside the primary denominator.
    badpath = lf.session_paths(CAPTURE, '175419')['raw_mp4']
    badcap = cv2.VideoCapture(str(badpath))
    bad_opened = badcap.isOpened()
    badread, _ = badcap.read()
    fifth_decoded = int(badread)
    if badread:
        while True:
            fifth_ok, _ = badcap.read()
            if not fifth_ok:
                break
            fifth_decoded += 1
    badcap.release()
    bad = {'session_id': '175419', 'filename': badpath.name, 'bytes': badpath.stat().st_size,
           'sha256': sha(badpath), 'matches_legacy_hash': sha(badpath) == lf.EXPECTED_FILES['175419']['raw_mp4'][1],
           'opened': bad_opened, 'first_frame_decoded': badread, 'sequentially_decoded_frames': fifth_decoded,
           'status': 'SEPARATE_MAPPING_REQUIRED' if badread else 'CORRUPT_RAW_MP4',
           'included_in_primary_denominator': False}
    # Contract describes source geometric IDs, not inferred physical visible vertices.
    signs = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
    contract = {'schema_version': 'lifter_corner_contract_v1', 'version': 'source_cuboid_ids_7e136fc_v1',
                'object_definition': 'Recorded plastic square pallet 1.10 x 1.10 x 0.15 m; eight canonical bounding cuboid corners, centroid origin, X width, Y down/height, Z depth. ID8 centroid is excluded.',
                'source': 'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py:cuboid + challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json',
                'source_kind': 'source_fact', 'source_commit': REFERENCE_COMMIT,
                'direct_click_policy': 'human_confirmation_required',
                'warning': 'Bounding cuboid corners may be virtual; do not click a nearby wood edge or infer occluded coordinates. Confirm physical correspondence or use definition_unconfirmed. Canonical orientation may require whole-object symmetry review; never reorder by screen position.',
                'corners': [{'id': i, 'name': 'ID ' + str(i), 'definition': f'centroid axes (X,Y,Z) sign {s}; metres {tuple(a*b for a,b in zip(s,(.55,.075,.55)))}',
                             'xyz_m': [a*b for a,b in zip(s,(.55,.075,.55))]} for i,s in enumerate(signs)]}
    dump(OUT / 'review/CORNER_CONTRACT.json', contract, frozen=True)
    plan = {'schema_version': 'lifter_evaluation_plan_v1', 'source_kind': 'machine_proposed',
            'source_commit': REFERENCE_COMMIT, 'selection_sha256': sha(OUT/'FIXED_REVIEW_SELECTION.json'),
            'stored_frame_count': len(allframes), 'fixed_review_frame_count': len(samples),
            'repeat_review_frame_count': sum(f['repeat_review'] for f in samples),
            'sample_rule': selection['rule'], 'repeat_rule': 'every fifth chronological selected frame within session',
            'frames': allframes, 'review_ids': [f['frame_id'] for f in samples],
            'methods': ['YOLO_R0','YOLO_R0_N3_seed1'], 'additional_training_runs': 0,
            'optimizer_updates': 0, 'gpu_pass_budget_s': 3600, 'full_saved_frame_passes': 1,
            'ground_truth_use_for_selection': False, 'prior_846_analysis_exposure': True,
            'visibility_not_used_to_replace_samples': True,
            'corner_accuracy_scope': 'fixed sensor-time sample human-confirmed directly visible corners only',
            'stop_intervals': [], 'stop_review_status': 'WAITING_HUMAN',
            'physical_reference_status': 'BLOCKED_REFERENCE'}
    dump(OUT / 'LIFTER_EVALUATION_PLAN.json', plan, frozen=True)
    manifest = {'schema_version': 'lifter_review_manifest_v1',
                'plan_sha256': sha(OUT/'LIFTER_EVALUATION_PLAN.json'),
                'corner_contract_sha256': sha(OUT/'review/CORNER_CONTRACT.json'), 'frames': samples}
    dump(OUT / 'review/MANIFEST.json', manifest, frozen=True)
    inputmap = {'schema_version': 'lifter_input_time_map_v1', 'source_kind': 'source_fact',
                'source_commit': REFERENCE_COMMIT, 'sessions': sessions, 'excluded_video': bad,
                'total_decoded_stored_frames': len(allframes),
                'decoder': {'opencv': cv2.__version__, 'format': 'uint8 BGR, sequential decode, lossless PNG samples',
                            'timing_from_nominal_or_container_fps': False},
                'source_bindings': binding_records,
                'clock_contract': 'RealSense camera_sensor_timestamp_ms; host monotonic separate origin, deltas only',
                'source_kind_registry': {'pallet_state_pose': 'operational_model_prediction', 'timing_pose': 'operational_model_prediction',
                                         'camera_sensor_timestamp_ms': 'sensor', 'control_seq': 'command', 'meta_intrinsics_dimensions': 'recorded_metadata'},
                'physical_accuracy_status': 'BLOCKED_REFERENCE', 'stop_jitter_status': 'WAITING_HUMAN',
                'capture_model_exposure': 'capture-specific training and physical individual overlap not independently verified; previously analysed sessions; no independent unseen-object claim',
                'audit_wall_seconds': time.perf_counter()-started}
    dump(OUT / 'LIFTER_INPUT_AND_TIME_MAP.json', inputmap, frozen=True)
    missing = [r['name'] for r in binding_records if r['status'] != 'VERIFIED_COMPLETE']
    receipt_path = ISO / '_docs/experiments/pallet_n3_completion_v3/LIFTER_YOLO_R0_N3_SEED1_COMPLETE.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    legacy_files = []
    for role in ('raw','plan'):
        binding = receipt[role]
        found = []
        for root in SOURCE_ROOTS:
            path = root/binding['path']
            if path.exists():
                found.append({**relative(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
                              'matches': sha(path)==binding['sha256'] and path.stat().st_size==binding['bytes']})
        legacy_files.append({'role': role, 'expected': binding, 'found': found})
    reuse = {'schema_version': 'lifter_legacy_reuse_audit_v1', 'status': 'BLOCKED_CONTRACT' if missing else 'READY_TO_REVIEW',
             'legacy_receipt_sha256': sha(receipt_path), 'legacy_receipt_source_commit': REFERENCE_COMMIT,
             'legacy_receipt_claim': {'sampled_frames': receipt['sampled_frames'], 'fresh_per_method': 832},
             'legacy_raw_verified_locally': all(any(f['matches'] for f in r['found']) for r in legacy_files),
             'legacy_files': legacy_files, 'missing_fixed_bindings': missing,
             'cache_reused_frames': 0, 'twenty_frame_equivalence_check': {'status': 'BLOCKED_CONTRACT', 'ran': False,
                'reason': 'requires hash-matched legacy raw and fixed Base/N3 weights'},
             'search_scope': ['isolated GitHub reference checkout','current workspace','three known GitHub pallet checkouts'],
             'absence_claim_scope': 'not found in inspected roots; not proof of global absence',
             'new_inference_status': 'BLOCKED_CONTRACT' if missing else 'READY_TO_REVIEW'}
    dump(OUT / 'LEGACY_REUSE_AUDIT.json', reuse)
    results = {'schema_version': 'lifter_results_v1', 'all_saved_frames': {'status': 'BLOCKED_CONTRACT' if missing else 'READY_TO_REVIEW',
                'denominator': len(allframes), 'inferred_frames': 0, 'fresh_output_rate': None},
               'fixed_visible_corners': {'status': 'WAITING_HUMAN', 'planned_frames': len(samples),
                'human_reviewed_frames': 0, 'reference_corners': None, 'median_error_px': None,'p90_error_px': None,'pck10_percent': None},
               'annotation_quality': {'status': 'WAITING_HUMAN','planned_repeat_frames':24,'median_repeat_difference_px':None,'p90_repeat_difference_px':None},
               'confirmed_stop_variation': {'status':'WAITING_HUMAN','intervals':None},
               'independent_physical_accuracy': {'status':'BLOCKED_REFERENCE','translation_error_m':None,'yaw_error_deg':None},
               'legacy_846': {'status':'RECEIPT_ONLY_NOT_REEXECUTED','sampled_frames':846,'reported_fresh_per_method':832,
                              'locally_verified_cache':False,'not_full_frame_results':True},
               'actual_cost': {'gpu_inference_seconds':0,'additional_training_runs':0,'optimizer_updates':0,
                               'audit_wall_seconds':time.perf_counter()-started}}
    dump(OUT/'LIFTER_RESULTS.json', results)
    print(json.dumps({'output_relative':OUT.relative_to(ROOT).as_posix(),'decoded_frames':len(allframes),
                      'review_frames':len(samples),'repeat_frames':24,'missing_fixed_bindings':missing},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['audit'])
    args = p.parse_args()
    audit()
