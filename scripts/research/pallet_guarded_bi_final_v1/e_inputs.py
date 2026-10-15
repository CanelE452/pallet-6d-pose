"""Predictor-only, fixed-vertex registration of frozen e into the new trial.

The frozen e pose is a model prediction, never a reference.  Original N3 /
SubPix image coordinates are permuted by physical vertex identity; e model
projections and mixed keypoint readouts are never used as observations.  No
legacy pipeline, scorer, model loader or image decoder is imported here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np


SCHEMA = 'GUARDED_BI_FIXED_E_PREDICTOR_INPUTS_V1'
BRANCH = 'E_FROZEN'
SOURCE_BRANCH = 'N3_SUBPIX'
SOURCE_ARM = 'SAME_FROZEN_E'
PARTITIONS = {'CAL': 46, 'TEST': 49, 'REMAINDER': 143}
SOURCE_FOLDER = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/'
    'statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/cached_native_edge_candidates_v4_17')
FIXED_HASHES = {
    'PREDICTIONS.jsonl.gz': 'decbf545281b101767377614091e828cd419d4c78d0c1abe384c722682eefc06',
    'PREDICTION_SEAL.json': '066729e65d5596aa08a92cb3792532e1bd2a14f590e50cc4396132a031007077',
    'PROTOCOL.json': '644bad30fed36d1263917e1e5164727feccc1e2ae7e22ca200dbdd666afffcc0',
}
VERTEX_ATOL_M = 1e-8
ROTATION_ATOL = 1e-7


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def _skip_value(text, start):
    """Skip JSON bytes without decoding unselected nested dictionaries."""
    i, n = start, len(text)
    while i < n and text[i].isspace():
        i += 1
    if i == n:
        raise ValueError('Missing JSON value')
    first = text[i]
    if first in '{[':
        stack, quoted, escaped = [first], False, False
        i += 1
        while i < n:
            char = text[i]
            if quoted:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in '{[':
                stack.append(char)
            elif char in '}]':
                if not stack or (stack.pop(), char) not in (('{','}'), ('[',']')):
                    raise ValueError('Unbalanced JSON container')
                if not stack:
                    return i+1
            i += 1
        raise ValueError('Unclosed JSON container')
    if first == '"':
        _, end = json.JSONDecoder().raw_decode(text, i)
        return end
    while i < n and text[i] not in ',}]' and not text[i].isspace():
        i += 1
    return i


def select_fields(text, wanted, *, raw_fields=()):
    """Decode only whitelisted fields of one JSON object, rejecting duplicates."""
    decoder, out, seen = json.JSONDecoder(), {}, set()
    i, n = 0, len(text)
    while i < n and text[i].isspace():
        i += 1
    if i == n or text[i] != '{':
        raise ValueError('A JSON object is required')
    i += 1
    while True:
        while i < n and text[i].isspace():
            i += 1
        if i < n and text[i] == '}':
            i += 1
            if text[i:].strip():
                raise ValueError('Trailing JSON data')
            return out
        key, i = decoder.raw_decode(text, i)
        if not isinstance(key, str) or key in seen:
            raise ValueError('Unique JSON object keys are required')
        seen.add(key)
        while i < n and text[i].isspace():
            i += 1
        if i == n or text[i] != ':':
            raise ValueError('Missing JSON colon')
        start = i+1
        end = _skip_value(text, start)
        if key in wanted:
            out[key] = text[start:end] if key in raw_fields else json.loads(text[start:end])
        i = end
        while i < n and text[i].isspace():
            i += 1
        if i < n and text[i] == ',':
            i += 1
            continue
        if i == n or text[i] != '}':
            raise ValueError('Missing JSON object terminator')


def cuboid(extents):
    a, b, c = np.asarray(extents, dtype=float)/2
    return np.array([[-a,-b,-c], [a,-b,-c], [a,b,-c], [-a,b,-c],
                     [-a,-b,c], [a,-b,c], [a,b,c], [-a,b,c]], dtype=float)


def _finite(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(name+' must have finite shape '+str(shape))
    return a


def _rotation(value, name):
    a = _finite(value, (3,3), name)
    if (not np.allclose(a.T@a, np.eye(3), atol=ROTATION_ATOL, rtol=0)
            or not np.isclose(np.linalg.det(a), 1, atol=ROTATION_ATOL, rtol=0)):
        raise ValueError(name+' must be a proper rotation')
    return a


def pose_geometry(pose, *, name):
    """Validate a registered model pose without changing any stored coordinate."""
    if pose.get('available') is not True:
        raise ValueError(name+' must have an explicit available frozen pose')
    extents = _finite(pose['cf_extents'], (3,), name+' dimensions')
    if np.any(extents <= 0):
        raise ValueError(name+' dimensions must be positive')
    R = _rotation(pose['R_cf'], name+' R_cf')
    physical = _rotation(pose['R_physical'], name+' R_physical')
    t = _finite(pose['centroid'], (3,), name+' center')
    Q = _rotation(R.T@physical, name+' derived registration Q')
    for key in ('physical_registration_Q', 'Q'):
        if key in pose and not np.allclose(_rotation(pose[key], name+' '+key), Q,
                                           atol=ROTATION_ATOL, rtol=0):
            raise ValueError(name+' stored and derived physical registration differ')
    if not np.allclose(R@Q, physical, atol=ROTATION_ATOL, rtol=0):
        raise ValueError(name+' native/physical rotation registration differs')
    return dict(native_xyz=cuboid(extents), physical_xyz=cuboid(extents)@Q,
                Q=Q, R_cf=R, centroid=t, R_physical=physical, cf_extents=extents)


def physical_vertex_bijection(source_pose, target_pose, physical_dimensions):
    """Return target-native -> source-native IDs by exact physical coordinates.

    Column-vector convention: X_physical=Q.T@X_cf, hence row vertices@Q.
    No nearest arbitrary assignment or symmetry phase is inferred: every
    target vertex must have exactly one match within the fixed metric tolerance.
    """
    source = pose_geometry(source_pose, name='source N3')
    target = pose_geometry(target_pose, name='frozen e')
    dimensions = _finite(physical_dimensions, (3,), 'known physical dimensions')
    if np.any(dimensions <= 0):
        raise ValueError('Known physical dimensions must be positive')
    known = cuboid(dimensions)
    def unique_matches(a, b, label):
        distances = np.linalg.norm(a[:,None,:]-b[None,:,:], axis=2)
        matches = distances <= VERTEX_ATOL_M
        if not np.all(matches.sum(axis=0) == 1) or not np.all(matches.sum(axis=1) == 1):
            raise ValueError(label+': no unique fixed physical vertex bijection')
        return np.argmax(matches, axis=1).tolist(), float(distances[matches].max())
    source_to_known, source_error = unique_matches(source['physical_xyz'], known, 'source dimensions')
    target_to_known, target_error = unique_matches(target['physical_xyz'], known, 'e dimensions')
    target_to_source, error = unique_matches(target['physical_xyz'], source['physical_xyz'], 'source/e')
    inverse = np.argsort(target_to_source).tolist()
    if [source_to_known[k] for k in target_to_source] != target_to_known:
        raise ValueError('Composed physical vertex registration is inconsistent')
    return dict(target_to_source_native_ids=target_to_source, source_to_target_native_ids=inverse,
        source_Q=source['Q'].tolist(), target_Q=target['Q'].tolist(),
        source_cf_extents=source['cf_extents'].tolist(), target_cf_extents=target['cf_extents'].tolist(),
        source_physical_vertex_xyz=source['physical_xyz'].tolist(),
        target_physical_vertex_xyz=target['physical_xyz'].tolist(),
        known_physical_dimensions_xyz_m=dimensions.tolist(),
        maximum_matching_error_m=max(error, source_error, target_error),
        tolerance_m=VERTEX_ATOL_M, bijection_unique=True,
        matching_rule='same fixed physical coordinate; no pose/image/GT fitted registration')


def attach_e_branch(frame, archived):
    """Pure own-E initialization; original frame and frozen pose stay unchanged."""
    before = digest(frame)
    for key in ('id', 'partition', 'session', 'image', 'image_sha256'):
        if archived[key] != frame[key]:
            raise ValueError('Frozen e / prior input identity mismatch: '+key)
    entry = archived['e']
    pose = entry['actual_pose']
    accepted = entry['accepted']
    fallback = pose.get('fallback')
    if not isinstance(accepted, bool) or not isinstance(fallback, bool) or accepted == fallback:
        raise ValueError('Frozen e acceptance/fallback contract is inconsistent')
    if entry['WD_family'] not in ('ORIGINAL_WD', 'SWAPPED_WD'):
        raise ValueError('Explicit e W/D family required')
    if BRANCH in frame['arms']:
        raise ValueError('Do not overwrite an existing E initialization')
    source = frame['arms'][SOURCE_BRANCH]
    points = np.asarray(source['native_points'], dtype=float)
    if points.shape not in ((8,2), (9,2)) or not np.isfinite(points).all():
        raise ValueError('Finite original N3 8/9 image coordinates required')
    mapping = physical_vertex_bijection(source['coarse_pose'], pose, frame['xyz'])
    ids = mapping['target_to_source_native_ids']
    reindexed = points.copy()
    reindexed[:8] = points[ids]
    def reindex_metadata(key):
        values = source[key]
        if len(values) != len(points):
            raise ValueError('Point metadata length differs: '+key)
        return [deepcopy(values[k]) for k in ids]+deepcopy(values[8:])
    conf = reindex_metadata('keypoints_conf')
    support = reindex_metadata('prediction_support')
    _finite(conf, (len(points),), 'original detector confidences')
    if not all(isinstance(v, bool) for v in support):
        raise ValueError('Original point support must be boolean')
    geometry = pose_geometry(pose, name='frozen e')
    K = _finite(frame['K'], (3,3), 'original K')
    if np.linalg.det(K) == 0:
        raise ValueError('Original camera matrix must be invertible')
    camera = geometry['native_xyz']@geometry['R_cf'].T+geometry['centroid']
    if np.any(camera[:,2] <= 0):
        raise ValueError('Frozen e geometry must retain positive camera depth')
    result = deepcopy(frame)
    result['arms'][BRANCH] = dict(native_points=reindexed.tolist(), keypoints_conf=conf,
        prediction_support=support, coarse_pose=deepcopy(pose),
        baseline_accepted=accepted, baseline_fallback=fallback,
        WD_family=entry['WD_family'], source_point_key='ORIGINAL_N3_SUBPIX_FIXED_PHYSICAL_VERTEX_REINDEXING',
        coarse_source='EXACT_FROZEN_E_MODEL_PREDICTION',
        confidence_source=source.get('confidence_source'),
        native_vertex_mapping=mapping,
        frozen_e_prediction_metadata=dict(method=entry['method'],
            fallback_audit=deepcopy(entry.get('fallback_audit')),
            hierarchical_correspondence_audit=deepcopy(entry.get('hierarchical_correspondence_audit'))),
        contract=dict(GT_inputs=False, frozen_e_pose_is_prediction=True,
            states_masks_and_BI_must_be_recomputed_from_own_E_geometry=True,
            model_projection_or_mixed_readout_used_as_observation=False,
            source_center9_preserved=len(points)==9,
            source_image_coordinates_unchanged_except_permutation=True))
    if len(points) == 9 and result['arms'][BRANCH]['native_points'][8] != source['native_points'][8]:
        raise ValueError('Original center9 must remain exact')
    if digest(frame) != before or result['arms'][BRANCH]['coarse_pose'] != pose:
        raise ValueError('Original frame or frozen e numeric pose was changed')
    canonical(result)
    return result


def read_frozen_e(source_root):
    """Stream only identity + e prediction fields from the pinned whole journal."""
    folder = Path(source_root)/SOURCE_FOLDER
    for name, expected in FIXED_HASHES.items():
        if sha(folder/name) != expected:
            raise ValueError('Fixed frozen e source bytes changed: '+name)
    seal = _read(folder/'PREDICTION_SEAL.json')
    protocol = _read(folder/'PROTOCOL.json')
    if (seal['rows'] != 238 or seal['predictions_sha256'] != FIXED_HASHES['PREDICTIONS.jsonl.gz']
            or seal['protocol_sha256'] != FIXED_HASHES['PROTOCOL.json']
            or seal.get('before_new_control_metric_read') is not True
            or seal['ordered_ids'] != protocol['ids']):
        raise ValueError('Complete before-score fixed e prediction seal required')
    rows, identities = [], set()
    top = {'id','partition','session','image','image_sha256','arms'}
    fields = {'actual_pose','accepted','WD_family','method','fallback_audit','hierarchical_correspondence_audit'}
    with gzip.open(folder/'PREDICTIONS.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = select_fields(line, top, raw_fields={'arms'})
            arms = select_fields(row.pop('arms'), {SOURCE_ARM}, raw_fields={SOURCE_ARM})
            if set(arms) != {SOURCE_ARM}:
                raise ValueError('Frozen e arm missing')
            row['e'] = select_fields(arms[SOURCE_ARM], fields)
            if row['id'] in identities:
                raise ValueError('Duplicate frozen e frame')
            identities.add(row['id'])
            rows.append(row)
    if ([r['id'] for r in rows] != seal['ordered_ids']
            or Counter(r['partition'] for r in rows) != Counter(PARTITIONS)):
        raise ValueError('Exact ordered original238 frozen e population required')
    return rows, dict(source_folder=str(SOURCE_FOLDER), source_file_sha256=deepcopy(FIXED_HASHES),
        original_prediction_rows=238, extracted_arm=SOURCE_ARM,
        decoded_prediction_fields=sorted(fields),
        GT_or_reference_or_score_numeric_decodes=0,
        archived_states_masks_readouts_and_solver_candidates_decoded=False)


def build_e_inputs(prior_input_dir, source_root, output_dir):
    """Create a new sealed input directory; return ``(frames, audit)``.

    Output frames preserve both original RAW/N3 branches and append E_FROZEN.
    Existing directories or files are never overwritten.
    """
    started, cpu = time.monotonic(), time.process_time()
    prior_input_dir, output_dir = Path(prior_input_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Preserve previous E input preparation: '+str(output_dir))
    prior_seal = _read(prior_input_dir/'INPUT_SEAL.json')
    for name, key in (('INPUTS.json','inputs_sha256'), ('INPUTS.jsonl.gz','inputs_gzip_sha256'),
                      ('CONTRACT.json','contract_sha256')):
        if sha(prior_input_dir/name) != prior_seal[key]:
            raise ValueError('Original minimal-BI input bytes changed: '+name)
    document = _read(prior_input_dir/'INPUTS.json')
    if document.get('reference_or_GT_inputs') is not False:
        raise ValueError('Predictor-only prior initialization required')
    with gzip.open(prior_input_dir/'INPUTS.jsonl.gz', 'rt') as stream:
        frames = [json.loads(line) for line in stream]
    if (len(frames) != 238 or [r['id'] for r in frames] != prior_seal['ordered_ids']
            or frames != document['frames']
            or [digest(r) for r in frames] != prior_seal['canonical_row_sha256']
            or Counter(r['partition'] for r in frames) != Counter(PARTITIONS)):
        raise ValueError('Exact fixed238 original predictor inputs required')
    archived, source_audit = read_frozen_e(source_root)
    if [r['id'] for r in archived] != prior_seal['ordered_ids']:
        raise ValueError('Frozen e / prior input ordered IDs differ')
    output = [attach_e_branch(frame, e) for frame,e in zip(frames, archived)]
    permutations = Counter(tuple(r['arms'][BRANCH]['native_vertex_mapping']['target_to_source_native_ids']) for r in output)
    prior_hashes = {name: sha(prior_input_dir/name) for name in
                   ('INPUT_SEAL.json','INPUTS.json','INPUTS.jsonl.gz','CONTRACT.json')}
    audit = dict(schema='GUARDED_BI_FIXED_E_INPUT_CONTRACT_V1', rows=238,
        partitions=PARTITIONS, branches=['RAW_YOLO','N3_SUBPIX',BRANCH],
        original_input_sha256=prior_hashes, frozen_e_source=source_audit,
        source_sha256={str(Path('scripts/research/pallet_guarded_bi_final_v1/e_inputs.py')):sha(__file__)},
        physical_vertex_permutations=[dict(target_to_source_native_ids=list(k),frames=v)
                                      for k,v in sorted(permutations.items())],
        nonidentity_permutation_frames=sum(v for k,v in permutations.items() if list(k)!=list(range(8))),
        maximum_physical_vertex_error_m=max(r['arms'][BRANCH]['native_vertex_mapping']['maximum_matching_error_m'] for r in output),
        physical_vertex_atol_m=VERTEX_ATOL_M, rotation_atol=ROTATION_ATOL,
        original_RAW_and_N3_input_branches_changed=False,
        E_pose_and_registration_exactly_preserved=True,
        source_image_coordinates_reindexed_only=True, center9_exactly_preserved=True,
        E_geometry_state_mask_BI_observations_reused=False,
        model_projected_or_mixed_readout_as_observation=False,
        new_model_or_seed_or_fit_or_GT_change=False, GT_inputs=False,
        GT_or_reference_or_score_document_opens=0, GPU_calls=0, PnP_calls=0,
        wall_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu)
    output_dir.mkdir(parents=True)
    doc = dict(schema=SCHEMA, ordered_ids=prior_seal['ordered_ids'],
               branches=audit['branches'], frames=output, reference_or_GT_inputs=False)
    _write(output_dir/'INPUTS.json', doc)
    with (output_dir/'INPUTS.jsonl.gz').open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as gz:
            for row in output:
                gz.write((canonical(row)+'\n').encode())
    _write(output_dir/'CONTRACT.json', audit)
    seal = dict(schema='GUARDED_BI_FIXED_E_PREDICTOR_INPUT_SEAL_V1', rows=238,
        partitions=PARTITIONS, arms=audit['branches'], ordered_ids=prior_seal['ordered_ids'],
        inputs_sha256=sha(output_dir/'INPUTS.json'), inputs_gzip_sha256=sha(output_dir/'INPUTS.jsonl.gz'),
        contract_sha256=sha(output_dir/'CONTRACT.json'),
        canonical_row_sha256=[digest(r) for r in output],
        prior_input_seal_sha256=prior_hashes['INPUT_SEAL.json'],
        frozen_e_prediction_sha256=FIXED_HASHES['PREDICTIONS.jsonl.gz'],
        source_sha256=audit['source_sha256'], GT_inputs=False, before_new_E_observation=True)
    _write(output_dir/'INPUT_SEAL.json', seal)
    return output, audit


def load_e_frames(output_dir, *, verify=True):
    folder = Path(output_dir)
    seal = _read(folder/'INPUT_SEAL.json')
    if verify:
        for name,key in (('INPUTS.json','inputs_sha256'), ('INPUTS.jsonl.gz','inputs_gzip_sha256'),
                         ('CONTRACT.json','contract_sha256')):
            if sha(folder/name) != seal[key]:
                raise ValueError('Sealed E initialization changed: '+name)
    with gzip.open(folder/'INPUTS.jsonl.gz', 'rt') as stream:
        frames = [json.loads(line) for line in stream]
    if ([r['id'] for r in frames] != seal['ordered_ids'] or len(frames) != 238
            or [digest(r) for r in frames] != seal['canonical_row_sha256']):
        raise ValueError('Sealed E population or coordinates changed')
    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-input-dir', required=True, type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    frames, audit = build_e_inputs(args.prior_input_dir, args.source_root, args.output_dir)
    print(json.dumps({k:audit[k] for k in ('rows','partitions','nonidentity_permutation_frames',
        'maximum_physical_vertex_error_m','GT_or_reference_or_score_document_opens','GPU_calls','PnP_calls')},
        ensure_ascii=False), flush=True)
