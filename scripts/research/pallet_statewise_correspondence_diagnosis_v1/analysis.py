"""Postseal, saved-pose analysis; stdlib/NumPy only, no inference imports.

The public entry point refuses all data IO without an explicit scoring GO.
It then verifies complete prediction AND candidate seals before decoding any
numeric reference. No NN, image decoder, optimizer, PnP, or selector is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np


DIAGNOSTICS = ('REOBS_MULTI', 'REOBS_SINGLE', 'TYPED_MULTI', 'TYPED_SINGLE')
BASELINES = ('R0', 'R1', 'N0', 'N1', 'E0', 'E1', 'POSEFIX_RAW',
             'N3_ONLY', 'SUBPIX_ONLY', 'RA', 'RB', 'NA', 'NB', 'EA', 'EB')
ARMS = BASELINES + DIAGNOSTICS
PARTITIONS = {'CAL': 46, 'TEST': 49, 'REMAINDER': 143}
METRICS = ('translation_cm', 'rotation_deg')
PRIMARY_CONTRASTS = (('TYPED_SINGLE', 'E0'), ('TYPED_SINGLE', 'TYPED_MULTI'),
                     ('TYPED_SINGLE', 'REOBS_SINGLE'))
SECONDARY_CONTRASTS = (
    ('REOBS_MULTI', 'E0'), ('REOBS_SINGLE', 'E0'), ('TYPED_MULTI', 'E0'),
    ('REOBS_SINGLE', 'REOBS_MULTI'), ('E1', 'E0'), ('EA', 'E0'), ('EB', 'E0'),
    ('R1', 'R0'), ('N1', 'N0'), ('E0', 'N0'), ('E0', 'POSEFIX_RAW'),
    ('N0', 'POSEFIX_RAW'), ('N1', 'POSEFIX_RAW'), ('N3_ONLY', 'R0'),
    ('SUBPIX_ONLY', 'R0'), ('N0', 'N3_ONLY'), ('N0', 'SUBPIX_ONLY'))
CONTRASTS = PRIMARY_CONTRASTS + SECONDARY_CONTRASTS
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260917
VIS = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1')
PF95 = VIS / 'posefix_sub_comparison_v1'
PF143 = VIS / 'remainder_after_test49_v1/posefix_baseline_supplement_v4_14'
REFERENCE_ANCHOR = VIS / ('statewise_joint_ot_20261014_v2/nonsevere_v4_fix/'
    'boundary_recovery_v1/geometry_guided_edge_eval_v4_30/SCORE_SEAL.json')
REFERENCE_ANCHOR_SHA256 = 'c2c85714f81dd90b78d46fd9e82541c9dc2ec1578e5e93c8f2209815ccf91d87'
REFERENCE_PATHS = {
    'CAL': VIS / 'OOF_pose_v1/ISOLATED_CAL_SCORING.json',
    'TEST': VIS / 'test_reused_v1/ISOLATED_TEST_SCORING.json',
    'REMAINDER': VIS / 'remainder_after_test49_v1/derived_reference_score_v4_14/PLAN.json'}
PF_CHECKPOINT = 'b9670d1a42db8d4e80ae1efe55eb237f05f81f7bd089cefe4051bbedd7d0a581'
PF_SUBPIX_ARMS = ('POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def journal(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _relative(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Explicit repository-relative source binding required')
    return Path(root)/p


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Full expected SHA256 required')


def rotation(value):
    R = np.asarray(value, float)
    if (R.shape != (3, 3) or not np.isfinite(R).all()
            or not np.allclose(R.T@R, np.eye(3), rtol=0, atol=1e-7)
            or not np.isclose(np.linalg.det(R), 1., rtol=0, atol=1e-7)):
        raise ValueError('Finite proper physical rotation required')
    return R


def point(value):
    t = np.asarray(value, float)
    if t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError('Finite center in meters required')
    return t


def normalize_entry(entry):
    entry = deepcopy(entry)
    pose = entry.get('actual_pose')
    if not isinstance(pose, dict) or type(pose.get('available')) is not bool:
        raise ValueError('Explicit available/unavailable actual_pose required')
    if pose['available']:
        if not all(k in pose for k in ('R_physical', 'centroid')):
            raise ValueError('Available pose requires physical R and center')
        R = rotation(pose['R_physical'])
        point(pose['centroid'])
        if 'R_cf' in pose and 'Q' in pose:
            if not np.allclose(rotation(pose['R_cf'])@rotation(pose['Q']), R,
                               rtol=0, atol=1e-7):
                raise ValueError('Registered native/physical pose mismatch')
    for key in ('accepted', 'fallback', 'technical_failure', 'solver_rejected', 'timeout'):
        value = entry.get(key)
        if value is not None and type(value) is not bool:
            raise ValueError(key+' must be boolean or explicit NA')
    if entry.get('accepted') is True and (entry.get('fallback') or not pose['available']):
        raise ValueError('Accepted cannot mean fallback or unavailable')
    solution = entry.get('solution') or {}
    audit = entry.get('audit') or {}
    ambiguity_record = solution.get('ambiguity') or {}
    ambiguity = ambiguity_record.get('detected')
    if ambiguity is not None and type(ambiguity) is not bool:
        raise ValueError('Ambiguity must be boolean or NA')
    status = {key: entry.get(key) for key in
              ('accepted', 'fallback', 'status', 'reason', 'technical_failure',
               'solver_rejected', 'timeout')}
    if status['status'] is None:
        status['status'] = solution.get('status')
    if status['reason'] is None:
        status['reason'] = solution.get('fallback_reason')
    status['ambiguous_but_accepted'] = (None if ambiguity is None or entry.get('accepted') is None
                                      else ambiguity and entry['accepted'])
    status['ambiguity_assessment_scope'] = (
        'SINGLE_START_LOCAL_RANK_NO_GLOBAL_ALTERNATIVE_TEST'
        if (entry.get('single_start_does_not_prove_global_uniqueness') is True
            or ambiguity_record.get('single_start_does_not_test_global_ambiguity') is True
            or audit.get('E0_start_only') is True)
        else 'RECORDED_EXISTING_CANDIDATE_SEARCH_NO_GLOBAL_UNIQUENESS_PROOF'
        if ambiguity_record else 'UNRECORDED')
    uses_e = entry.get('uses_e_initialization')
    if uses_e is not None and type(uses_e) is not bool:
        raise ValueError('Explicit E0 initialization flag must be boolean or NA')
    if audit.get('E0_start_only') is True:
        if uses_e is False:
            raise ValueError('E0-only start audit conflicts with explicit initialization flag')
        uses_e, uses_e_source = True, 'SEALED_AUDIT.E0_start_only'
    elif audit.get('exact_original_hierarchical_family_choice') is True:
        if uses_e is True:
            raise ValueError('Original hierarchical replay is not an E0-start replay')
        uses_e, uses_e_source = False, 'SEALED_AUDIT.exact_original_hierarchical_family_choice_ORIGINAL_PIPELINE'
    else:
        uses_e_source = 'SEALED_ENTRY.uses_e_initialization' if uses_e is not None else 'UNRECORDED'
    status.update(uses_e_initialization=uses_e, uses_e_initialization_source=uses_e_source,
                  start_policy_comparison_scope=('NOT_ISOLATED' if audit.get('start_policy_comparison_NOT_ISOLATED') is True
                                                 else 'NO_ADDITIONAL_ISOLATION_CLAIM'))
    restored = entry.get('restored_original_status')
    status.update(status_source=('SEALED_ORIGINAL_E0_METADATA' if restored else 'SAVED_PREDICTION_ENTRY'),
                  original_status_restored=restored is not None,
                  historical_common_accepted=None if restored is None else restored['historical_common']['accepted'],
                  historical_common_fallback=None if restored is None else restored['historical_common']['fallback'])
    return dict(actual_pose=pose, prediction_status=status, restored_original_status=deepcopy(restored))


def restore_e0_status(entry, branch, binding):
    """Restore explicit original flags; never infer them from accuracy or fitting."""
    result = deepcopy(entry)
    coarse = deepcopy(branch['coarse_pose'])
    coarse.setdefault('available', True)  # Exact historical guarded/run packaging.
    if result['actual_pose'] != coarse:
        raise ValueError('Original E0 status source must bind the exact saved pose')
    accepted, fallback = branch['baseline_accepted'], branch['baseline_fallback']
    if type(accepted) is not bool or type(fallback) is not bool or accepted == fallback:
        raise ValueError('Explicit complementary original E0 acceptance/fallback required')
    if any(result.get(k) is not None and result[k] != v
           for k, v in [('accepted', accepted), ('fallback', fallback)]):
        raise ValueError('Original E0 status conflicts with historical common metadata')
    result['restored_original_status'] = dict(
        historical_common={k: deepcopy(result.get(k)) for k in ('accepted', 'fallback', 'status', 'reason')},
        original=dict(accepted=accepted, fallback=fallback), source=deepcopy(binding),
        exact_saved_pose_bridge=True, accuracy_and_pose_selection_unchanged=True)
    result.update(accepted=accepted, fallback=fallback)
    return result


def identity_rows(rows, *, full=True):
    ids = [r['id'] for r in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Nonempty unique ordered frame IDs required')
    for r in rows:
        if r['partition'] not in PARTITIONS or any(
                not isinstance(r.get(k), str) or not r[k] for k in ('session', 'cluster_id')):
            raise ValueError('Original partition/session/recording cluster IDs required')
    if full and (len(rows) != 238 or Counter(r['partition'] for r in rows) != Counter(PARTITIONS)):
        raise ValueError('Exactly CAL46 + TEST49 + REMAINDER143 required')
    return ids


def prediction_gate(folder, prediction_seal_sha256, candidate_seal_sha256, candidate_file):
    """Complete both predictor-only ledgers before returning any reference path."""
    folder = Path(folder)
    _digest(prediction_seal_sha256)
    _digest(candidate_seal_sha256)
    if sha(folder/'PREDICTION_SEAL.json') != prediction_seal_sha256:
        raise ValueError('New prediction seal changed')
    if sha(folder/'CANDIDATE_SEAL.json') != candidate_seal_sha256:
        raise ValueError('Candidate coverage must be sealed before analysis')
    p, c = read(folder/'PREDICTION_SEAL.json'), read(folder/'CANDIDATE_SEAL.json')
    for value in (p, c):
        if value['rows'] != 238 or value['GT_inputs'] is not False or value['before_new_scoring'] is not True:
            raise ValueError('Complete prelabel238 prediction/candidate seal required')
    if (p['candidate_seal_sha256'] != candidate_seal_sha256
            or p['predictions_sha256'] != sha(folder/'REPLAY_PREDICTIONS.jsonl.gz')
            or c['generated_candidates_sha256'] != sha(candidate_file)
            or p['ordered_ids'] != c['ordered_ids'] or p['arms'] != list(DIAGNOSTICS)):
        raise ValueError('Prediction/candidate bytes, ordering or arms changed')
    rows = journal(folder/'REPLAY_PREDICTIONS.jsonl.gz')
    if identity_rows(rows) != p['ordered_ids']:
        raise ValueError('Predictions do not match ordered sealed238 IDs')
    for r in rows:
        if set(r['predictions']) != set(DIAGNOSTICS):
            raise ValueError('All four fixed diagnostic paths must retain every frame')
        for arm in DIAGNOSTICS:
            normalize_entry(r['predictions'][arm])
    return rows, p, c


def _saved_bundle(folder):
    folder = Path(folder)
    seal = read(folder/'PREDICTION_SEAL.json')
    protocol = folder/('METHOD_PROTOCOL.json' if len(seal['arms']) == 12 else 'PROTOCOL.json')
    if (seal['predictions_sha256'] != sha(folder/'PREDICTIONS.jsonl.gz') or seal['GT_inputs'] is not False
            or seal['protocol_sha256'] != sha(protocol)
            or not (seal.get('before_new_scoring') is True or seal.get('before_new_CAL_scoring') is True)):
        raise ValueError('Existing baseline prediction seal changed')
    rows = journal(folder/'PREDICTIONS.jsonl.gz')
    if identity_rows(rows) != seal['ordered_ids']:
        raise ValueError('Existing baseline identities changed')
    for r in rows:
        if set(r['predictions']) != set(seal['arms']):
            raise ValueError('Existing baseline arm inventory changed')
    return rows


def collect_baselines(*, single_folder, guarded_folder, source_root, diagnostic_rows, lineage_folder=None):
    """Join saved Rt only. No existing score is used to choose a pose or arm."""
    single, guarded = _saved_bundle(single_folder), _saved_bundle(guarded_folder)
    wanted = identity_rows(diagnostic_rows)
    if [r['id'] for r in single] != wanted or [r['id'] for r in guarded] != wanted:
        raise ValueError('Every existing and new baseline must share ordered238')
    efolder = Path(guarded_folder)/'e_inputs'
    eseal = read(efolder/'INPUT_SEAL.json')
    edigest = sha(efolder/'INPUTS.json')
    if eseal['inputs_sha256'] != edigest or eseal['ordered_ids'] != wanted:
        raise ValueError('Frozen original E0 status input bytes or ordering changed')
    eframes = read(efolder/'INPUTS.json')['frames']
    if [f['id'] for f in eframes] != wanted:
        raise ValueError('Complete exact original E0 status frame inventory required')
    ebinding = dict(path='PRIVATE/guarded/e_inputs/INPUTS.json', sha256=edigest,
                    input_seal_sha256=sha(efolder/'INPUT_SEAL.json'),
                    fields=['arms.E_FROZEN.baseline_accepted', 'arms.E_FROZEN.baseline_fallback'])
    if lineage_folder is not None:
        extraction = read(Path(lineage_folder)/'EXTRACTION_SEAL.json')
        if extraction['guarded_inputs_sha256'] != edigest or extraction['ordered_ids'] != wanted:
            raise ValueError('Original E0 status source differs from frozen typed lineage')
        ebinding['lineage_extraction_seal_sha256'] = sha(Path(lineage_folder)/'EXTRACTION_SEAL.json')
    root = Path(source_root)
    pf = {}
    for relative, seal_name in ((PF95, 'PREDICTION_SEAL.json'), (PF143, 'BASELINE_SEAL.json')):
        folder = root/relative
        seal = read(folder/seal_name)
        if (seal['predictions_sha256'] != sha(folder/'PREDICTIONS.jsonl.gz')
                or seal['protocol_sha256'] != sha(folder/'PROTOCOL.json')):
            raise ValueError('Exact original saved PoseFix prediction/protocol required')
        protocol = read(folder/'PROTOCOL.json')
        checkpoint = protocol['PoseFix']['checkpoint_sha256'] if relative == PF95 else protocol['checkpoint_sha256']
        if checkpoint != PF_CHECKPOINT:
            raise ValueError('Frozen synthetic-only seed1 RAW checkpoint changed')
        for r in journal(folder/'PREDICTIONS.jsonl.gz'):
            if relative == PF95 and r['method'] != 'POSEFIX_RAW':
                continue
            if r['id'] in pf:
                raise ValueError('Duplicate PoseFix image across original cohorts')
            pf[r['id']] = r
    if set(pf) != set(wanted):
        raise ValueError('Complete exact95+143 PoseFix RAW identities required')
    result = []
    for new, s, g, eframe in zip(diagnostic_rows, single, guarded, eframes):
        for old in (s, g):
            if any(old[k] != new[k] for k in ('id', 'partition', 'session', 'cluster_id')):
                raise ValueError('Original metadata/recording pairing changed')
        for arm in ('R0', 'R1', 'N0', 'N1'):
            if s['predictions'][arm] != g['predictions'][arm]:
                raise ValueError('Previously exact preserved baseline differs between archives')
        pp = pf[new['id']]
        if pp.get('session', new['session']) != new['session'] or pp.get('partition', new['partition']) != new['partition']:
            raise ValueError('Original PoseFix metadata differs')
        entries = {arm: deepcopy(s['predictions'][arm]) for arm in ('R0', 'R1', 'N0', 'N1')}
        entries.update({arm: deepcopy(g['predictions'][arm]) for arm in ('E0', 'E1', 'RA', 'RB', 'NA', 'NB', 'EA', 'EB')})
        if any(eframe[k] != new[k] for k in ('id', 'partition', 'session', 'cluster_id')):
            raise ValueError('Original E0 status frame metadata differs')
        entries['E0'] = restore_e0_status(entries['E0'], eframe['arms']['E_FROZEN'], ebinding)
        entries.update(N3_ONLY=deepcopy(s['predictions']['N3_F']),
                       SUBPIX_ONLY=deepcopy(s['predictions']['SUBPIX_F']),
                       POSEFIX_RAW=dict(actual_pose=deepcopy(pp['actual_pose']), accepted=None,
                                        fallback=None, status='EXACT_CACHED_POSEFIX_RAW_PASS1',
                                        reason='Historical acceptance flags unavailable'))
        entries.update(deepcopy(new['predictions']))
        result.append({**{k: new[k] for k in ('id', 'partition', 'session', 'cluster_id')},
                       'predictions': {arm: normalize_entry(entries[arm]) for arm in ARMS}})
    return result


def _transform(value):
    T = np.asarray(value, float)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.array_equal(T[3], [0, 0, 0, 1]):
        raise ValueError('Original centroid-origin reference transform required')
    rotation(T[:3, :3])
    return T


def _signed_yaw(name):
    index = {'YAW_0': 0, 'YAW_90': 1, 'YAW_180': 2, 'YAW_270': 3}[name]
    return np.linalg.matrix_power(np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]), index)


def reference_document(document):
    """Independent fixed candidate-zero policy; no nearest-prediction fit."""
    if document.get('schema_version') != 'real_pallet_gt_v2' or len(document.get('objects', [])) != 1:
        raise ValueError('Original single-object derived reference required')
    obj = document['objects'][0]
    cf = obj['camera_facing_pnp']
    nominal = _transform(cf['pose_transform'])
    dims = [float(obj['physical_dimensions_m'][a]) for a in ('x', 'y', 'z')]
    if not np.isfinite(dims).all() or min(dims) <= 0:
        raise ValueError('Original registered positive dimensions required')
    records = obj.get('canonical_pose_candidates') or []
    R = None
    if len(records) == 2:
        try:
            ts = [_transform(r['pose_transform']) for r in records]
            valid = all(np.array_equal(t[:3, 3], nominal[:3, 3]) and np.allclose(
                t[:3, :3], nominal[:3, :3]@_signed_yaw(r['axis_assignment']),
                rtol=0, atol=1e-6) for t, r in zip(ts, records))
            first, second = [int(r['axis_assignment'].split('_')[1]) for r in records]
            valid = valid and (second-first) % 360 == 180 and np.array_equal(
                ts[1][:3, :3], ts[0][:3, :3]@_signed_yaw('YAW_180'))
            if valid:
                R = ts[0][:3, :3]
        except (KeyError, ValueError, TypeError):
            pass
    confirmed = obj.get('canonical_pose')
    if confirmed is not None:
        T = _transform(confirmed['pose_transform'])
        valid = (cf.get('axis_assignment_confirmed') is True
                 and np.array_equal(T[:3, 3], nominal[:3, 3])
                 and np.allclose(T[:3, :3], nominal[:3, :3]@_signed_yaw(confirmed['axis_assignment']),
                                 rtol=0, atol=1e-6))
        R = T[:3, :3] if valid else None
    return dict(t=nominal[:3, 3].tolist(), R=None if R is None else R.tolist(), xyz=dims, order=2)


def fixed_references(source_root, manifest_path, rows):
    root, manifest = Path(source_root), read(manifest_path)
    if (sha(root/REFERENCE_ANCHOR) != REFERENCE_ANCHOR_SHA256
            or manifest['reference_bindings'] != read(root/REFERENCE_ANCHOR)['reference_bindings']
            or set(manifest['reference_bindings']) != set(PARTITIONS)):
        raise ValueError('Fixed reference ancestry changed')
    references, annotations = {}, {}
    for part, relative in REFERENCE_PATHS.items():
        binding = manifest['reference_bindings'][part]
        path = _relative(root, binding['path'])
        if binding['path'] != str(relative) or sha(path) != binding['sha256']:
            raise ValueError('Original fixed reference bytes/path changed')
        data = read(path)
        wanted = [r['id'] for r in rows if r['partition'] == part]
        if part == 'REMAINDER':
            if data['ids'] != wanted or [r['id'] for r in data['annotations']] != wanted:
                raise ValueError('All143 original reference rows in fixed order required')
            references[part] = {}
            for a in data['annotations']:
                p = _relative(root, a['annotation'])
                if sha(p) != a['annotation_sha256']:
                    raise ValueError('Original annotation SHA changed')
                references[part][a['id']] = reference_document(read(p))
                annotations[a['annotation']] = a['annotation_sha256']
        else:
            references[part] = {fid: data['rows'][fid]['target']['truth'] for fid in wanted}
    return references, manifest['reference_bindings'], annotations


def metric(pose, reference):
    """Center norm and trace/acos physical C2 angle, directly from one Rt pair."""
    if type(pose.get('available')) is not bool or reference['order'] != 2:
        raise ValueError('Explicit pose availability and fixed physical/C2 contract required')
    ref_R = reference['R']
    if not pose['available']:
        return dict(pose_available=False, translation_cm=None, rotation_deg=None,
                    success_le5=False, success_strict_lt5=False,
                    reference_rotation_available=ref_R is not None)
    t = point(pose['centroid'])
    target = point(reference['t'])
    translation = float(np.linalg.norm(t-target)*100.)
    R = rotation(pose['R_physical'])
    if ref_R is None:
        angle = None
    else:
        G = rotation(ref_R)
        # Same floating representation of Y(pi) as the fixed original metric.
        # This is C2 only; no 90deg or prediction-dependent correspondence.
        c, s = np.cos(np.pi), np.sin(np.pi)
        C2 = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        angle = min(float(math.degrees(math.acos(float(np.clip(
            (np.trace((G@S).T@R)-1.)/2., -1., 1.))))) for S in (np.eye(3), C2))
    return dict(pose_available=True, translation_cm=translation, rotation_deg=angle,
                success_le5=None if angle is None else translation <= 5. and angle <= 5.,
                success_strict_lt5=None if angle is None else translation < 5. and angle < 5.,
                reference_rotation_available=angle is not None)


def score_rows(rows, references):
    return [dict(id=r['id'], partition=r['partition'], session=r['session'], cluster_id=r['cluster_id'],
                 metrics={arm: metric(r['predictions'][arm]['actual_pose'],
                                      references[r['partition']][r['id']]) for arm in ARMS},
                 prediction_status={arm: r['predictions'][arm]['prediction_status'] for arm in ARMS},
                 restored_original_status={arm: r['predictions'][arm].get('restored_original_status')
                                           for arm in ARMS if r['predictions'][arm].get('restored_original_status')})
            for r in rows]


def distribution(values):
    values = list(values)
    x = [float(v) for v in values if v is not None]
    if not all(math.isfinite(v) for v in x):
        raise ValueError('Nonfinite error is a failure, not an available-only omission')
    n = len(x)
    mean = math.fsum(x)/n if n else None
    variance = math.fsum((v-mean)**2 for v in x)/(n-1) if n > 1 else None
    return dict(n=n, full_denominator=len(values), missing=len(values)-n, mean=mean,
                sample_variance=variance, sample_std=math.sqrt(variance) if variance is not None else None,
                median=float(np.median(x)) if n else None,
                P90=float(np.quantile(x, .90)) if n else None,
                P95=float(np.quantile(x, .95)) if n else None,
                max=max(x) if n else None, ddof=1, available_only=True, tail_trimming=False)


class RecordingBootstrap:
    """Paired whole-recording bootstrap from this exact population's ID list."""
    def __init__(self, rows, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
        self.rows = list(rows)
        identity_rows(self.rows, full=False)
        self.units = sorted({r['cluster_id'] for r in self.rows})
        index = {u: i for i, u in enumerate(self.units)}
        self.inverse = np.asarray([index[r['cluster_id']] for r in self.rows])
        n = len(self.units)
        self.draws = np.random.default_rng(seed).multinomial(
            n, np.full(n, 1/n), size=draws).astype('<u2')
        self.seed, self.draw_count = seed, draws

    def metadata(self):
        return dict(seed=self.seed, resamples=self.draw_count, units=len(self.units),
                    clusters=self.units, cluster_sizes=dict(Counter(r['cluster_id'] for r in self.rows)),
                    session_sizes=dict(Counter(r['session'] for r in self.rows)),
                    draw_sha256=hashlib.sha256(self.draws.tobytes()).hexdigest(),
                    paired_whole_recording=True, original_frame_weighted_estimand=True,
                    historical_draw_array_reused=False, session_as_cluster_invented=False,
                    method='whole-recording multinomial percentile bootstrap; linear quantile',
                    limitation='Repeated development data and few recording clusters; not independent validation')

    def contrast(self, values):
        if len(values) != len(self.rows):
            raise ValueError('Every paired availability mask must retain population length')
        mask = np.asarray([v is not None for v in values], bool)
        x = np.asarray([0. if v is None else float(v) for v in values])
        if not np.isfinite(x).all():
            raise ValueError('Finite paired deltas required')
        count = np.bincount(self.inverse, weights=mask, minlength=len(self.units))
        total = np.bincount(self.inverse, weights=x, minlength=len(self.units))
        denominator = (self.draws*count[None]).sum(1)
        keep = denominator > 0
        means = (self.draws*total[None]).sum(1)[keep]/denominator[keep]
        n = int(mask.sum())
        eligible = int((count > 0).sum())
        return dict(mean_difference=math.fsum(x[mask])/n if n else None,
                    CI95=np.quantile(means, [.025, .975]).tolist() if eligible > 1 and len(means) else None,
                    CI_unavailable_reason=None if eligible > 1 else 'Fewer than two eligible recording clusters',
                    full_denominator=len(values), common_available_n=n, excluded_unavailable_n=len(values)-n,
                    eligible_clusters=eligible, valid_resamples=int(keep.sum()), empty_resamples=int((~keep).sum()),
                    improved=int((x[mask] < 0).sum()), worsened=int((x[mask] > 0).sum()), tied=int((x[mask] == 0).sum()))


def paired(rows, new, before, bootstrap):
    groups = {name: [] for name in ('both_success', 'failure_to_success', 'success_to_failure',
                                   'both_failure', 'reference_unresolved')}
    for r in rows:
        a, b = [r['metrics'][arm]['success_le5'] for arm in (new, before)]
        name = ('reference_unresolved' if a is None or b is None else 'failure_to_success' if a and not b
                else 'success_to_failure' if b and not a else 'both_success' if a else 'both_failure')
        groups[name].append(r)
    binary = [None if any(r['metrics'][a]['success_le5'] is None for a in (new, before)) else
              int(r['metrics'][new]['success_le5'])-int(r['metrics'][before]['success_le5']) for r in rows]
    success = bootstrap.contrast(binary)
    success['gains'] = success.pop('worsened')
    success['losses'] = success.pop('improved')
    success['unit'] = 'success-rate difference; multiply by100 for percentage points'
    if any(v is None for v in binary):
        success.update(conditional_known_reference_mean=success['mean_difference'], mean_difference=None, CI95=None,
                       CI_unavailable_reason='Reference success unresolved; full-denominator delta not imputed')
    continuous = {}
    for key in METRICS:
        delta = {r['id']: None if any(r['metrics'][a][key] is None for a in (new, before)) else
                 r['metrics'][new][key]-r['metrics'][before][key] for r in rows}
        available = [v for v in delta.values() if v is not None]
        n = len(available)
        value = bootstrap.contrast(list(delta.values()))
        contribution = {}
        for category, subset in groups.items():
            xs = [delta[r['id']] for r in subset if delta[r['id']] is not None]
            total = math.fsum(xs)
            contribution[category] = dict(rows=len(subset), common_available_n=len(xs), signed_delta_sum=total,
                improvement_sum=math.fsum(x for x in xs if x < 0), worsening_sum=math.fsum(x for x in xs if x > 0),
                contribution_to_common_available_mean=total/n if n else None,
                observed_total_over_full_N=total/len(rows),
                observed_total_over_full_N_is_not_imputed_full_mean=bool(n != len(rows)))
        value.update(improvement_sum=math.fsum(x for x in available if x < 0),
                     worsening_sum=math.fsum(x for x in available if x > 0),
                     group_contributions=contribution,
                     before_common_available=distribution(r['metrics'][before][key] for r in rows if delta[r['id']] is not None),
                     new_common_available=distribution(r['metrics'][new][key] for r in rows if delta[r['id']] is not None))
        if n and not math.isclose(math.fsum(c['signed_delta_sum'] for c in contribution.values())/n,
                                  value['mean_difference'], rel_tol=0, abs_tol=1e-10):
            raise ArithmeticError('Four-way delta contributions do not reproduce paired mean')
        continuous[key] = value
    counts = {k: len(v) for k, v in groups.items()}
    if not groups['reference_unresolved']:
        difference = sum(r['metrics'][new]['success_le5'] for r in rows)-sum(r['metrics'][before]['success_le5'] for r in rows)
        if counts['failure_to_success']-counts['success_to_failure'] != difference:
            raise ArithmeticError('Recovery minus damage does not reproduce success difference')
    return dict(new=new, reference=before, full_denominator=len(rows),
                primary_prespecified_contrast=(new, before) in PRIMARY_CONTRASTS,
                transition=dict(counts=counts, ids={k: [r['id'] for r in v] for k, v in groups.items()},
                                session_counts={k: dict(Counter(r['session'] for r in v)) for k, v in groups.items()}),
                success_rate=success, continuous=continuous, p_raw=None, p_Holm=None,
                p_value_reason='Descriptive predeclared paired CIs; no new confirmatory p-value family',
                not_equivalence_or_noninferiority=True)


def summarize(rows, *, arms=ARMS, contrasts=CONTRASTS):
    boot = RecordingBootstrap(rows)
    result = {}
    for arm in arms:
        metrics = [r['metrics'][arm] for r in rows]
        statuses = [r['prediction_status'][arm] for r in rows]
        status_counts = {}
        for key in ('accepted', 'fallback', 'technical_failure', 'solver_rejected', 'timeout', 'ambiguous_but_accepted'):
            status_counts[key] = dict(true_n=sum(s.get(key) is True for s in statuses),
                                     false_n=sum(s.get(key) is False for s in statuses),
                                     unknown_n=sum(s.get(key) is None for s in statuses))
        unresolved = sum(m['success_le5'] is None for m in metrics)
        result[arm] = dict(denominator=len(rows), successes_le5=sum(m['success_le5'] is True for m in metrics),
            success_rate_le5=sum(m['success_le5'] is True for m in metrics)/len(rows),
            success_rate_is_confirmed_lower_bound=bool(unresolved), reference_unresolved_n=unresolved,
            successes_strict_lt5=sum(m['success_strict_lt5'] is True for m in metrics),
            strict_vs_inclusive_changed_ids=[r['id'] for r in rows if r['metrics'][arm]['success_le5'] is True
                                            and r['metrics'][arm]['success_strict_lt5'] is False],
            pose_available_n=sum(m['pose_available'] for m in metrics), no_pose_n=sum(not m['pose_available'] for m in metrics),
            no_pose_ids=[r['id'] for r in rows if not r['metrics'][arm]['pose_available']],
            failure_ids=[r['id'] for r in rows if r['metrics'][arm]['success_le5'] is not True],
            errors={key: distribution(m[key] for m in metrics) for key in METRICS},
            translation_tails={str(limit): dict(count=sum(m['translation_cm'] is not None and m['translation_cm'] > limit for m in metrics),
                ids=[r['id'] for r in rows if r['metrics'][arm]['translation_cm'] is not None
                     and r['metrics'][arm]['translation_cm'] > limit], operator='>', threshold_cm=limit,
                full_denominator=len(rows)) for limit in (50, 100)},
            status_counts=status_counts, raw_status_counts=dict(Counter(str(s.get('status')) for s in statuses)),
            original_status_restored_n=sum(s.get('original_status_restored') is True for s in statuses),
            status_source_counts=dict(Counter(s.get('status_source', 'UNRECORDED') for s in statuses)),
            ambiguity_scope_counts=dict(Counter(s.get('ambiguity_assessment_scope', 'UNRECORDED') for s in statuses)))
    return dict(rows=len(rows), arms=result, comparisons={a+'-'+b: paired(rows, a, b, boot) for a, b in contrasts},
                bootstrap=boot.metadata(), all_rows_kept=True, all_failures_kept=True,
                default_changed=False, independent_validation=False)


def _csv(path, rows, columns):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _saved_baselines(path, rows):
    """Prediction-only display/independent-check view; all raw coordinates absent."""
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            for r in rows:
                value = {k: r[k] for k in ('id', 'partition', 'session', 'cluster_id')}
                value['predictions'] = {a: dict(actual_pose=r['predictions'][a]['actual_pose'],
                    **r['predictions'][a]['prediction_status']) for a in BASELINES}
                for arm in BASELINES:
                    if r['predictions'][arm].get('restored_original_status'):
                        value['predictions'][arm]['restored_original_status'] = r['predictions'][arm]['restored_original_status']
                stream.write((json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                                         allow_nan=False)+'\n').encode())


def posefix_subpix_common95(source_root, scored, references):
    """Both preexisting caps on their exact common95; never extend/select one."""
    folder = Path(source_root)/PF95
    seal, protocol = read(folder/'PREDICTION_SEAL.json'), read(folder/'PROTOCOL.json')
    combinations = protocol['new_combinations']
    if (seal['predictions_sha256'] != sha(folder/'PREDICTIONS.jsonl.gz')
            or seal['protocol_sha256'] != sha(folder/'PROTOCOL.json')
            or protocol['PoseFix']['checkpoint_sha256'] != PF_CHECKPOINT
            or combinations['both_controls_fixed_before_scoring'] is not True
            or combinations['no_best_anchor_selected'] is not True):
        raise ValueError('The two preexisting fixed cap controls and original checkpoint required')
    existing = {}
    for row in journal(folder/'PREDICTIONS.jsonl.gz'):
        if row['method'] in PF_SUBPIX_ARMS:
            key = (row['id'], row['method'])
            if key in existing:
                raise ValueError('Duplicate original PoseFix+SubPix candidate')
            existing[key] = row
    requested = [r for r in scored if r['partition'] in ('CAL', 'TEST')]
    if (Counter(r['partition'] for r in requested) != Counter(CAL=46, TEST=49)
            or set(existing) != {(r['id'], a) for r in requested for a in PF_SUBPIX_ARMS}):
        raise ValueError('Both fixed PoseFix+SubPix arms must retain all original95 images')
    arms = ('POSEFIX_RAW', *PF_SUBPIX_ARMS, 'N0', 'E0')
    contrasts = tuple((a, b) for a in PF_SUBPIX_ARMS for b in ('POSEFIX_RAW', 'N0', 'E0'))
    rows = []
    for original in requested:
        row = deepcopy(original)
        row['metrics'] = {a: row['metrics'][a] for a in ('POSEFIX_RAW', 'N0', 'E0')}
        row['prediction_status'] = {a: row['prediction_status'][a] for a in ('POSEFIX_RAW', 'N0', 'E0')}
        for a in PF_SUBPIX_ARMS:
            old = existing[(row['id'], a)]
            if any(old.get(k, row[k]) != row[k] for k in ('partition', 'session')):
                raise ValueError('Original PoseFix+SubPix metadata mismatch')
            row['metrics'][a] = metric(old['actual_pose'], references[row['partition']][row['id']])
            row['prediction_status'][a] = dict(accepted=None, fallback=None, status='PREEXISTING_FIXED_CAP_CONTROL',
                reason='No new SubPix or solver call; directly rescore saved Rt', technical_failure=None,
                solver_rejected=None, timeout=None, ambiguous_but_accepted=None, ambiguity_assessment_scope='UNRECORDED')
        rows.append(row)
    populations = {p: summarize([r for r in rows if r['partition'] == p], arms=arms, contrasts=contrasts)
                   for p in ('CAL', 'TEST')}
    populations['COMMON95'] = summarize(rows, arms=arms, contrasts=contrasts)
    result = dict(schema='PREEXISTING_POSEFIX_SUBPIX_TWO_FIXED_CAPS_COMMON95_DIRECT_RESCORE_V1',
        populations=populations, arms=list(arms), no_cap_or_method_selection=True,
        PRIMARY192_established=False, full_cohort_status='POSEFIX_SUBPIX_FULL_COHORT_NOT_ESTABLISHED',
        both_caps_reported=True, missing143_not_imputed_or_inferred=True, new_SubPix_calls=0, new_pose_solves=0,
        source_protocol_sha256=sha(folder/'PROTOCOL.json'), source_prediction_seal_sha256=sha(folder/'PREDICTION_SEAL.json'),
        fixed_cap_definitions={k: combinations[k] for k in ('cap_fraction_raw_image_diagonal', *PF_SUBPIX_ARMS)},
        no_independent_validation=True)
    flat = [dict(id=r['id'], partition=r['partition'], session=r['session'], cluster_id=r['cluster_id'],
                 arm=a, **r['metrics'][a]) for r in rows for a in arms]
    return result, flat


def score(*, folder, source_root, single_folder, guarded_folder, reference_manifest,
          expected_prediction_seal_sha256, expected_candidate_seal_sha256,
          expected_reference_manifest_sha256, candidate_file, output, root_scoring_GO=False):
    if root_scoring_GO is not True:
        raise PermissionError('Explicit root scoring GO required before any data IO')
    started, cpu = time.monotonic(), time.process_time()
    folder, output = Path(folder), Path(output)
    if output.exists():
        raise FileExistsError('Preserve every previous analysis attempt; use a new output')
    new, prediction_seal, candidate_seal = prediction_gate(folder, expected_prediction_seal_sha256,
                                                         expected_candidate_seal_sha256, candidate_file)
    _digest(expected_reference_manifest_sha256)
    if sha(reference_manifest) != expected_reference_manifest_sha256:
        raise ValueError('Fixed reference manifest changed before decoding labels')
    rows = collect_baselines(single_folder=single_folder, guarded_folder=guarded_folder,
                             source_root=source_root, diagnostic_rows=new, lineage_folder=folder/'e_trace_v2')
    references, bindings, annotations = fixed_references(source_root, reference_manifest, rows)
    scored = score_rows(rows, references)
    populations = {p: [r for r in scored if r['partition'] == p] for p in PARTITIONS}
    populations['PRIMARY'] = populations['TEST']+populations['REMAINDER']
    result = {p: summarize(values) for p, values in populations.items()}
    common95, common95_rows = posefix_subpix_common95(source_root, scored, references)
    output.mkdir(parents=True)
    _saved_baselines(output/'BASELINE_PREDICTIONS.jsonl.gz', rows)
    # Private postseal checker input only; never passed back to inference.
    write(output/'FIXED_REFERENCES.json', dict(references=references,
        reference_bindings=bindings, annotation_sha256=annotations,
        available_only_after_prediction_and_candidate_seals=True))
    write(output/'SCORED_ROWS.json', dict(rows=scored, arm_rows=len(scored)*len(ARMS)))
    write(output/'POSEFIX_SUBPIX_COMMON95.json', common95)
    _csv(output/'POSEFIX_SUBPIX_COMMON95.csv', common95_rows, list(common95_rows[0]))
    write(output/'RESULTS.json', dict(schema='STATEWISE_FIXED238_DIRECT_PHYSICAL_C2_ANALYSIS_V1',
        populations=result, arms=list(ARMS), primary_contrasts=[a+'-'+b for a, b in PRIMARY_CONTRASTS],
        all_rows_and_failures_kept=True, no_fit_or_method_selection=True, default_changed=False,
        success='center_translation_cm<=5 AND physical_C2_rotation_deg<=5', sample_SD_ddof=1,
        failure_continuous_policy='No imputation; available-only and paired common-available denominators explicit',
        POSEFIX_SUBPIX_FULL_COHORT_NOT_ESTABLISHED=True, independent_validation=False))
    flat = [dict(id=r['id'], partition=r['partition'], session=r['session'], cluster_id=r['cluster_id'],
                 arm=a, **r['metrics'][a], **r['prediction_status'][a]) for r in scored for a in ARMS]
    _csv(output/'ROW_METRICS.csv', flat, list(flat[0]))
    transitions = [dict(population=p, contrast=c, category=category, id=fid)
                   for p, summary in result.items() for c, value in summary['comparisons'].items()
                   for category, ids in value['transition']['ids'].items() for fid in ids]
    _csv(output/'PAIR_TRANSITIONS.csv', transitions, ['population', 'contrast', 'category', 'id'])
    summaries = [dict(population=p, arm=a, denominator=x['denominator'], successes=x['successes_le5'],
                      no_pose_n=x['no_pose_n'],
                      **{name+'_'+kind: x['status_counts'][name][kind] for name in x['status_counts']
                         for kind in ('true_n', 'false_n', 'unknown_n')},
                      **{metric+'_'+key: x['errors'][metric][key]
                      for metric in METRICS for key in ('n', 'mean', 'sample_variance', 'sample_std', 'max')})
                 for p, summary in result.items() for a, x in summary['arms'].items()]
    _csv(output/'COHORT_SUMMARY.csv', summaries, list(summaries[0]))
    inputs = {'PRIVATE/single/PREDICTION_SEAL.json': sha(Path(single_folder)/'PREDICTION_SEAL.json'),
              'PRIVATE/guarded/PREDICTION_SEAL.json': sha(Path(guarded_folder)/'PREDICTION_SEAL.json'),
              'PRIVATE/guarded/e_inputs/INPUTS.json': sha(Path(guarded_folder)/'e_inputs/INPUTS.json'),
              'PRIVATE/guarded/e_inputs/INPUT_SEAL.json': sha(Path(guarded_folder)/'e_inputs/INPUT_SEAL.json'),
              'PRIVATE/e_trace_v2/EXTRACTION_SEAL.json': sha(folder/'e_trace_v2/EXTRACTION_SEAL.json')}
    for relative in (PF95, PF143):
        for name in ('PREDICTIONS.jsonl.gz', 'PROTOCOL.json'):
            inputs[str(relative/name)] = sha(Path(source_root)/relative/name)
    # Prove postseal analysis did not alter either predictor-only ledger.
    if (sha(folder/'PREDICTION_SEAL.json') != expected_prediction_seal_sha256
            or sha(folder/'CANDIDATE_SEAL.json') != expected_candidate_seal_sha256
            or sha(folder/'REPLAY_PREDICTIONS.jsonl.gz') != prediction_seal['predictions_sha256']
            or sha(candidate_file) != candidate_seal['generated_candidates_sha256']):
        raise ValueError('Prediction/candidate bytes changed during analysis')
    write(output/'SCORE_SEAL.json', dict(schema='STATEWISE_ANALYSIS_SEAL_V1', rows=238,
        arm_rows=238*len(ARMS), prediction_seal_sha256=expected_prediction_seal_sha256,
        candidate_seal_sha256=expected_candidate_seal_sha256,
        reference_manifest_sha256=expected_reference_manifest_sha256,
        reference_bindings=bindings, annotation_sha256=annotations, input_sha256=inputs,
        source_sha256={'analysis.py': sha(__file__)}, new_model_forwards=0, new_pose_solves=0,
        new_training_updates=0, root_scoring_GO=True, all_rows_kept=True,
        elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
        artifacts={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('folder', 'source-root', 'single-folder', 'guarded-folder', 'reference-manifest',
                'candidate-file', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    for key in ('prediction-seal-sha256', 'candidate-seal-sha256', 'reference-manifest-sha256'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--root-scoring-go', action='store_true')
    a = p.parse_args()
    score(folder=a.folder, source_root=a.source_root, single_folder=a.single_folder,
          guarded_folder=a.guarded_folder, reference_manifest=a.reference_manifest,
          candidate_file=a.candidate_file, output=a.output,
          expected_prediction_seal_sha256=a.prediction_seal_sha256,
          expected_candidate_seal_sha256=a.candidate_seal_sha256,
          expected_reference_manifest_sha256=a.reference_manifest_sha256,
          root_scoring_GO=a.root_scoring_go)


if __name__ == '__main__':
    main()
