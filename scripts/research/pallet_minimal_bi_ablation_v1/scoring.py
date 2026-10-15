"""Separate all-predictions-first scorer for the fixed 238 x four paths.

No reference, prediction, model or image file is opened on import. Reference
metadata preparation hashes files without decoding reference coordinates.
The scorer accepts only the complete prediction seal before numeric labels.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import gzip
import hashlib
import json
import re

import numpy as np

from .statistics import ARMS, METRICS, summarize_population

PARTITIONS = {'CAL': 46, 'TEST': 49, 'REMAINDER': 143}
LEGACY_DOC = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1')
B = LEGACY_DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
REFERENCE_ANCHOR = B/'geometry_guided_edge_eval_v4_30/SCORE_SEAL.json'
# Keep the exact published anchor, including its historical byte digest.
REFERENCE_ANCHOR_SHA256 = 'c2c85714f81dd90b78d46fd9e82541c9dc2ec1578e5e93c8f2209815ccf91d87'
REFERENCE_PATHS = dict(CAL=LEGACY_DOC/'nonsevere_v1/OOF_pose_v1/ISOLATED_CAL_SCORING.json',
    TEST=LEGACY_DOC/'nonsevere_v1/test_reused_v1/ISOLATED_TEST_SCORING.json',
    REMAINDER=LEGACY_DOC/'nonsevere_v1/remainder_after_test49_v1/derived_reference_score_v4_14/PLAN.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve every previous scoring attempt: '+path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('An explicit full SHA256 is required before IO')


def _inside(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Repository-relative binding required')
    return Path(root)/p


def verify_current_bindings(protocol, input_seal, source_root):
    """Hash current inference ancestry without numeric reference decoding."""
    code_root = Path(__file__).resolve().parents[3]
    source_root = Path(source_root)
    mappings = ((protocol['source_sha256'], True), (protocol['legacy_source_sha256'], True),
                (input_seal['source_sha256'], True), (input_seal['input_sha256'], False))
    count = 0
    for mapping, is_source in mappings:
        for name, expected in mapping.items():
            _digest(expected)
            relative = Path(name)
            if relative.is_absolute():
                path = relative  # Explicit local input-builder source binding.
            else:
                path = _inside(source_root, name)
                if is_source and (code_root/relative).exists():
                    path = code_root/relative
            if sha(path) != expected:
                raise ValueError('Inference source/input changed: '+str(name))
            count += 1
    for binding in protocol['model_descriptors'].values():
        if sha(_inside(source_root,binding['path'])) != binding['sha256']:
            raise ValueError('Frozen model descriptor changed')
        count += 1
    return count


def prepare_reference_manifest(*, source_root, input_seal_path, expected_input_seal_sha256,
                               output, root_metadata_GO=False):
    """Metadata-only fixed ancestry, never coordinates or pose values."""
    if root_metadata_GO is not True:
        raise PermissionError('Metadata preparation GO before IO')
    _digest(expected_input_seal_sha256)
    source_root, input_seal_path = Path(source_root), Path(input_seal_path)
    if sha(input_seal_path) != expected_input_seal_sha256:
        raise ValueError('Exact new inference input seal required')
    anchor = source_root/REFERENCE_ANCHOR
    if sha(anchor) != REFERENCE_ANCHOR_SHA256:
        raise ValueError('Fixed published reference anchor changed')
    bindings = read(anchor)['reference_bindings']
    if set(bindings) != set(PARTITIONS):
        raise ValueError('The same three fixed reference roots are required')
    for part, relative in REFERENCE_PATHS.items():
        binding = bindings[part]
        if binding['path'] != str(relative) or sha(source_root/relative) != binding['sha256']:
            raise ValueError('Fixed reference bytes/path changed: '+part)
    result = dict(schema='MINIMAL_BI_SEPARATE_FIXED_REFERENCE_MANIFEST_V1',
                  input_seal_sha256=expected_input_seal_sha256,
                  reference_anchor=dict(path=str(REFERENCE_ANCHOR), sha256=REFERENCE_ANCHOR_SHA256),
                  reference_bindings=bindings, reference_numeric_decodes=0,
                  fixed_alignment_and_C2_definition_unchanged=True)
    write(output, result)
    return sha(output)


def _rotation(value):
    x = np.asarray(value, dtype=float)
    if (x.shape != (3, 3) or not np.isfinite(x).all()
            or not np.allclose(x.T@x, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(x), 1., atol=1e-7, rtol=0)):
        raise ValueError('Finite proper physical rotation required')
    return x


def normalize_entry(entry, arm):
    pose = entry['actual_pose']
    if not isinstance(pose, dict) or not isinstance(pose.get('available'), bool):
        raise ValueError('Explicit available/unavailable pose required')
    accepted, fallback = entry.get('accepted'), entry.get('fallback')
    if accepted is not None and not isinstance(accepted, bool):
        raise ValueError('Accepted is boolean or NA')
    if fallback is not None and not isinstance(fallback, bool):
        raise ValueError('Fallback is boolean or NA')
    if accepted and (fallback or not pose['available']):
        raise ValueError('Accepted cannot mean unavailable or fallback')
    if pose['available']:
        _rotation(pose['R_physical'])
        t = np.asarray(pose['centroid'], dtype=float)
        if t.shape != (3,) or not np.isfinite(t).all():
            raise ValueError('Finite geometric model center required')
        if 'R_cf' in pose and 'Q' in pose:
            if not np.allclose(_rotation(pose['R_cf'])@_rotation(pose['Q']),
                               pose['R_physical'], atol=1e-7, rtol=0):
                raise ValueError('Native/physical pose basis changed')
    return dict(actual_pose=pose, accepted=accepted, fallback=fallback,
                status=entry['status'], reason=entry.get('reason'))


def compact_predictions(rows, frames, ordered_ids):
    """Pure all-row identity/physics gate, completed before label IO."""
    rows, frames = list(rows), list(frames)
    if len(rows) != 238 or len(frames) != 238 or len(set(ordered_ids)) != 238:
        raise ValueError('The fixed ordered 238 unique original frames are required')
    compact = []
    expected_parts = [p for p, n in PARTITIONS.items() for _ in range(n)]
    for i, (r, f) in enumerate(zip(rows, frames)):
        if (r['id'] != f['id'] or r['id'] != ordered_ids[i]
                or r['partition'] != f['partition'] or r['partition'] != expected_parts[i]
                or r['session'] != f['session'] or r['cluster_id'] != f['cluster_id']
                or not isinstance(r['cluster_id'], str) or not r['cluster_id']
                or set(r['predictions']) != set(ARMS)):
            raise ValueError('Missing, reordered, unpaired or relabeled population row')
        for arm, branch in (('R0','RAW_YOLO'), ('N0','N3_SUBPIX')):
            initial = dict(f['arms'][branch]['coarse_pose'])
            initial.setdefault('available',True)
            if (r['predictions'][arm]['actual_pose'] != initial
                    or r['predictions'][arm].get('accepted') is not None
                    or r['predictions'][arm].get('fallback') is not None):
                raise ValueError('Unchanged own F baseline and unknown acceptance must remain exact')
        compact.append(dict(id=r['id'], partition=r['partition'], session=r['session'],
                            cluster_id=r['cluster_id'],
                            predictions={a: normalize_entry(r['predictions'][a], a) for a in ARMS}))
    return compact


def _rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if not line.strip():
                raise ValueError('Blank prediction row')
            yield json.loads(line)


def prediction_gate(*, folder, input_dir, expected_prediction_seal_sha256):
    _digest(expected_prediction_seal_sha256)
    folder, input_dir = Path(folder), Path(input_dir)
    if sha(folder/'PREDICTION_SEAL.json') != expected_prediction_seal_sha256:
        raise ValueError('Prediction seal changed')
    seal, protocol = read(folder/'PREDICTION_SEAL.json'), read(folder/'PROTOCOL.json')
    input_seal = read(input_dir/'INPUT_SEAL.json')
    links = {'protocol_sha256': folder/'PROTOCOL.json',
             'predictions_sha256': folder/'PREDICTIONS.jsonl.gz',
             'input_seal_sha256': input_dir/'INPUT_SEAL.json',
             'inputs_sha256': input_dir/'INPUTS.json'}
    if (seal['rows'] != 238 or seal['arm_rows'] != 952 or seal['arms'] != list(ARMS)
            or seal['ordered_ids'] != input_seal['ordered_ids']
            or protocol['ordered_ids'] != input_seal['ordered_ids']
            or seal['before_scoring'] is not True or seal['GT_inputs'] is not False
            or any(seal[k] != sha(path) for k, path in links.items())
            or input_seal['inputs_sha256'] != seal['inputs_sha256']):
        raise ValueError('The complete same-input 238 x four prelabel seal is required')
    observation_seal = read(folder/'OBSERVATION_SEAL.json')
    if (sha(folder/'OBSERVATION_SEAL.json') != seal['observation_seal_sha256']
            or observation_seal['rows'] != 238 or observation_seal['branches'] != 476
            or observation_seal['GT_inputs'] is not False
            or observation_seal['protocol_sha256'] != seal['protocol_sha256']
            or sha(folder/'OBSERVATIONS.jsonl.gz') != observation_seal['observations_sha256']
            or protocol['GT_inputs'] is not False
            or protocol['success'] != 'translation_cm<=5 AND rotation_deg<=5'
            or protocol['historical_success'] != 'translation_cm<5 AND rotation_deg<5'):
        raise ValueError('Same complete own-branch observations and fixed success definition required')
    inputs = read(input_dir/'INPUTS.json')
    compact = compact_predictions(_rows(folder/'PREDICTIONS.jsonl.gz'), inputs['frames'], seal['ordered_ids'])
    return compact, seal, protocol


def read_fixed_references(*, source_root, manifest, compact):
    """Called only after every numeric prediction was validated."""
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import score_remainder_derived_reference_v4_14 as REF
    source_root = Path(source_root)
    anchor = manifest['reference_anchor']
    if (anchor != dict(path=str(REFERENCE_ANCHOR), sha256=REFERENCE_ANCHOR_SHA256)
            or sha(source_root/REFERENCE_ANCHOR) != REFERENCE_ANCHOR_SHA256
            or manifest['reference_bindings'] != read(source_root/REFERENCE_ANCHOR)['reference_bindings']):
        raise ValueError('The same fixed reference ancestry must survive scoring')
    result, decoded = {}, {}
    for part, relative in REFERENCE_PATHS.items():
        binding = manifest['reference_bindings'][part]
        if binding['path'] != str(relative) or sha(source_root/relative) != binding['sha256']:
            raise ValueError('Fixed scoring root changed')
        requested = [r['id'] for r in compact if r['partition'] == part]
        labels = read(source_root/relative)
        if part == 'REMAINDER':
            annotations = labels['annotations']
            if labels['ids'] != requested or [r['id'] for r in annotations] != requested:
                raise ValueError('All fixed143 derived-reference rows required')
            refs = {}
            for row in annotations:
                path = _inside(source_root, row['annotation'])
                if sha(path) != row['annotation_sha256']:
                    raise ValueError('Original derived annotation bytes changed')
                refs[row['id']] = REF.reference_from_document(read(path))
                decoded[row['annotation']] = row['annotation_sha256']
            result[part] = refs
        else:
            result[part] = {fid: labels['rows'][fid]['target']['truth'] for fid in requested}
    return result, decoded


def metric_for_pose(pose, reference, *, remainder=False):
    if not pose['available']:
        return dict(pose_available=False, translation_cm=None, rotation_deg=None, ADDsym_cm=None,
                    success_le5=False, success_strict_lt5=False, reference_rotation_available=reference['R'] is not None)
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import scoring as OLD
    if remainder:
        from scripts.research.pallet_visibility_curriculum_20261013_v1 import score_remainder_derived_reference_v4_14 as REF
        value = REF.metric_for_reference(pose, reference)
    else:
        value = OLD.pose_metric(pose, reference)
    t, r, add = (value[k] for k in METRICS)
    if not np.isfinite(t) or any(x is not None and not np.isfinite(x) for x in (r, add)):
        raise ValueError('Finite available prediction yielded an invalid reference error')
    return dict(pose_available=True, translation_cm=float(t),
                rotation_deg=None if r is None else float(r), ADDsym_cm=None if add is None else float(add),
                success_le5=None if r is None else bool(t <= 5. and r <= 5.),
                success_strict_lt5=None if r is None else bool(t < 5. and r < 5.),
                reference_rotation_available=r is not None)


def score_compact(compact, references):
    return [dict(id=r['id'], partition=r['partition'], session=r['session'], cluster_id=r['cluster_id'],
                 metrics={a: metric_for_pose(r['predictions'][a]['actual_pose'], references[r['partition']][r['id']],
                                            remainder=r['partition']=='REMAINDER') for a in ARMS},
                 prediction_status={a: {k:r['predictions'][a][k] for k in ('accepted','fallback','status','reason')} for a in ARMS})
            for r in compact]


def _CSV(path, rows, fields):
    if Path(path).exists():
        raise FileExistsError('Preserve previous CSV')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def score(*, folder, input_dir, source_root, reference_manifest, expected_reference_manifest_sha256,
          expected_prediction_seal_sha256, output, root_scoring_GO=False):
    if root_scoring_GO is not True:
        raise PermissionError('Scoring GO before any data IO')
    _digest(expected_prediction_seal_sha256); _digest(expected_reference_manifest_sha256)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Every scoring attempt must use a new output directory')
    compact, seal, protocol = prediction_gate(folder=folder, input_dir=input_dir,
        expected_prediction_seal_sha256=expected_prediction_seal_sha256)
    binding_count = verify_current_bindings(protocol, read(Path(input_dir)/'INPUT_SEAL.json'), source_root)
    if sha(reference_manifest) != expected_reference_manifest_sha256:
        raise ValueError('Separate reference metadata changed')
    manifest = read(reference_manifest)
    if manifest['input_seal_sha256'] != sha(Path(input_dir)/'INPUT_SEAL.json'):
        raise ValueError('Reference manifest must bind the same inference inputs')
    references, annotations = read_fixed_references(source_root=source_root, manifest=manifest, compact=compact)
    scored = score_compact(compact, references)
    populations = {p:[r for r in scored if r['partition']==p] for p in PARTITIONS}
    populations['PRIMARY'] = populations['TEST']+populations['REMAINDER']
    results = {p:summarize_population(rows) for p,rows in populations.items()}
    output.mkdir(parents=True)
    flat = [dict(id=r['id'], partition=r['partition'], session=r['session'], cluster_id=r['cluster_id'], arm=a,
                 **r['metrics'][a], **r['prediction_status'][a]) for r in scored for a in ARMS]
    if len(flat) != 952:
        raise ValueError('Every original frame x four path must remain')
    _CSV(output/'ROW_METRICS.csv', flat, list(flat[0]))
    changed = [dict(population=p, contrast=c, category=category, id=fid)
               for p,res in results.items() for c,v in res['comparisons'].items()
               for category in ('failure_to_success','success_to_failure') for fid in v['transition']['ids'][category]]
    _CSV(output/'CHANGED_CASES.csv', changed, ['population','contrast','category','id'])
    write(output/'SCORED_ROWS.json', dict(rows=scored, arm_rows=952))
    write(output/'RESULTS.json', dict(schema='MINIMAL_BI_FULL_FOUR_PATH_SCORE_V1', populations=results,
        headline='translation_cm <=5 AND rotation_deg <=5', historical='strict <5 AND <5 separately',
        repeated_development_data=True, default_changed=False, model_or_parameter_update=False,
        numeric_references_opened_only_after_all952_prediction_gate=True))
    write(output/'PAIRED_RESULTS.json', {p:dict(comparisons=v['comparisons'], bootstrap=v['bootstrap'],
        Holm_family=v['Holm_family']) for p,v in results.items()})
    if sha(Path(folder)/'PREDICTION_SEAL.json') != expected_prediction_seal_sha256:
        raise ValueError('Prediction changed during scoring')
    if verify_current_bindings(protocol, read(Path(input_dir)/'INPUT_SEAL.json'), source_root) != binding_count:
        raise ValueError('Inference binding inventory changed during scoring')
    artifacts = {p.name:sha(p) for p in output.iterdir() if p.is_file()}
    write(output/'SCORE_SEAL.json', dict(schema='SEALED_MINIMAL_BI_FULL_FOUR_PATH_SCORE_V1', rows=238, arm_rows=952,
        prediction_seal_sha256=expected_prediction_seal_sha256,
        reference_manifest_sha256=expected_reference_manifest_sha256, reference_bindings=manifest['reference_bindings'],
        annotation_sha256=annotations, assets=artifacts, all_rows_kept=True,
        current_inference_ancestry_bindings_verified=binding_count,
        source_sha256={Path(__file__).name:sha(__file__), 'statistics.py':sha(Path(__file__).with_name('statistics.py'))}))
    return sha(output/'SCORE_SEAL.json')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', required=True, type=Path); p.add_argument('--input-dir', required=True, type=Path)
    p.add_argument('--source-root', required=True, type=Path); p.add_argument('--reference-manifest', required=True, type=Path)
    p.add_argument('--reference-manifest-sha256', required=True); p.add_argument('--prediction-seal-sha256', required=True)
    p.add_argument('--output', required=True, type=Path); p.add_argument('--root-scoring-GO', action='store_true')
    a = p.parse_args()
    print(score(folder=a.folder, input_dir=a.input_dir, source_root=a.source_root, reference_manifest=a.reference_manifest,
        expected_reference_manifest_sha256=a.reference_manifest_sha256,
        expected_prediction_seal_sha256=a.prediction_seal_sha256, output=a.output, root_scoring_GO=a.root_scoring_GO))
