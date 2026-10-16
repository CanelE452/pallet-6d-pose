"""Inspect published V13 bytes, lineage and all five saved operational cohorts.

This standalone standard-library supplement opens no private weights, source
RGB or reference geometry. It copies saved per-frame scalar errors to CSV;
the independently runnable verify_review.py owns the 810-moment/81-CI audit.
It never imports production, rerenders, fits, scores or generates draws.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_finite_edge_consensus_repair_20261011_v13'
FILES = ('PUBLIC_REVIEW_PROTOCOL.json', 'PUBLIC_REVIEW_STARTED.json', 'PUBLIC_REVIEW_CHECKS.json')
CSV_NAME = 'PUBLIC_FRAME_METRICS.csv'
METHODS = ('BASE', 'N3_SUBPIX', 'ROLE_BOUNDARY_H_ROBUST',
           'ROLE_BOUNDARY_LOCAL_POINT_LINE', 'FINITE_EDGE_CONSENSUS')
PRIMARY = METHODS[-1]
RECEIPTS = tuple('''PROTOCOL GEOMETRY_SEAL CONTROL_RECEIPT PARENT_POPULATION_CHECKS
VALIDATION_PROTOCOL VALIDATION_STARTED VALIDATION_CHECKS
REVIEW_VALIDATION_PROTOCOL REVIEW_VALIDATION_STARTED REVIEW_VALIDATION_CHECKS
CPU_PROTOCOL CPU_STARTED CPU_CHECKS CPU_REVIEW_PROTOCOL CPU_REVIEW_STARTED CPU_REVIEW_CHECKS
SCORING_AUTHORITY_SUPPLEMENT SCORING_STARTED SCORING_RECEIPT
STATISTICS_STARTED STATISTICS_FIRST_FAILURE REVIEW_STATISTICS_STARTED STATISTICS_AUTHORITY_SUPPLEMENT
METRICS REVIEW_STATISTICS_VERIFICATION_PROTOCOL REVIEW_STATISTICS_VERIFICATION_STARTED REVIEW_VERIFICATION
RUNTIME_PROTOCOL RUNTIME_PREFLIGHT RUNTIME_STARTED RUNTIME_RESOURCES RUNTIME_RECEIPT
RUNTIME_CHECK_PROTOCOL RUNTIME_CHECK_STARTED RUNTIME_CHECKS
FIGURE_PROTOCOL FIGURES_STARTED FIGURES_FAILURE FIGURE_COMPLETION_PROTOCOL FIGURE_COMPLETION_STARTED
SUPPLEMENTAL_FIGURE_CHECKS ROOT_VISUAL_REVIEW
POSTHOC_METADATA_PROTOCOL POSTHOC_METADATA_STARTED POSTHOC_METADATA_CHECKS
ARCHIVE_PROTOCOL ARCHIVE_STARTED ARCHIVE_CHECKS ARCHIVE_MANIFEST'''.split())


def require(value, message):
    if not value:
        raise RuntimeError(message)


def reject_symlink(path):
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink: ' + str(path))


def read(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def public_path(name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts and relative.parts
            and relative.parts[0] in ('scripts', '_docs'), 'unsafe public dependency: ' + name)
    path = REPO / relative
    reject_symlink(path)
    return path


def binding(path):
    path = Path(path).absolute()
    reject_symlink(path)
    require(path.is_file(), 'missing public input: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                bytes=path.stat().st_size, sha256=h.hexdigest())


def same(left, right):
    return all(left[k] == right[k] for k in ('bytes', 'sha256'))


def write_new(path, value):
    reject_symlink(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def inputs(folder):
    files = {name + '.json': folder / (name + '.json') for name in RECEIPTS}
    files.update({p.name: p for p in folder.glob('*.gz')})
    files['METRICS.csv'] = folder / 'METRICS.csv'
    for p in sorted((folder / 'figures').glob('*.png')):
        files['PNG:' + p.name] = p
    for name in ('public_review.py', 'archive_evidence.py', 'restore_archives.py', 'verify_review.py'):
        files['helper:' + name] = Path(__file__).resolve().with_name(name)
    expected, skipped = {}, []
    # Only direct declared inputs are traversed. No private protection inventory
    # or unlisted filesystem tree is decoded to discover dependencies.
    for name in RECEIPTS:
        data = read(folder / (name + '.json'))
        for alias, value in data.get('inputs', {}).items():
            require(isinstance(value, dict) and all(k in value for k in ('path', 'bytes', 'sha256')),
                    'declared byte binding required: ' + name + ':' + alias)
            path = value['path']; origin = value.get('origin')
            is_public = origin == 'public_repository' or (origin != 'external_readonly_dependency'
                and isinstance(path, str) and path.startswith(('_docs/', 'scripts/')))
            if name in ('ARCHIVE_PROTOCOL', 'ARCHIVE_CHECKS'):
                require(Path(path).name == path, 'archive helper uses exact basenames')
                target = (Path(__file__).resolve().with_name(path) if alias in ('archive_code', 'restore_code')
                          else folder / path)
                key = 'archive:' + alias; files[key] = target
                expected[name + ':' + alias] = dict(input_key=key, binding=value)
            elif is_public:
                key = 'public:' + path
                files[key] = public_path(path)
                expected[name + ':' + alias] = dict(input_key=key, binding=value)
            else:
                skipped.append(dict(receipt=name, alias=alias, binding=value,
                    reason='private/external witness retained; dependency not opened'))
    manifest = read(folder / 'ARCHIVE_MANIFEST.json')
    require(manifest['schema'] == 'finite_edge_consensus_original_gzip_archive_manifest_v13', 'V13 archive manifest required')
    for item in manifest['files']:
        for value in (item['original'], *item['parts']):
            relative = Path(value['path'])
            require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe archive witness path')
            key = 'archive-witness:' + value['path']; files[key] = folder / relative
            expected[key] = dict(input_key=key, binding=value)
    for index, value in enumerate(read(folder / 'ROOT_VISUAL_REVIEW.json')['images']):
        key = 'visual:' + str(index); files[key] = public_path(value['path'])
        expected[key] = dict(input_key=key, binding=value)
    return files, expected, skipped


def guard(args, stage):
    folder, output = args.input.absolute(), args.output.absolute()
    for path in (folder, output): reject_symlink(path)
    require(folder.is_dir(), 'public evidence directory missing')
    output.mkdir(parents=True, exist_ok=True)
    for name in (FILES if stage == 'freeze' else FILES[1:]) + (CSV_NAME,):
        require(not (output / name).exists(), 'preserve existing output: ' + name)
    return folder, output


def snapshot(folder):
    files, expected, skipped = inputs(folder)
    bound = {key: binding(path) for key, path in files.items()}
    for key, value in expected.items():
        require(same(bound[value['input_key']], value['binding']), 'public declared bytes differ: ' + key)
    return files, bound, expected, skipped


def freeze(args):
    folder, output = guard(args, 'freeze'); _, bound, expected, skipped = snapshot(folder)
    write_new(output / FILES[0], dict(schema='finite_edge_public_evidence_protocol_v13',
        inputs=bound, declared_public_bindings=expected, skipped_external_dependencies=skipped,
        input_folder=str(folder), methods=METHODS, configured_scored_rows=1225,
        frozen_before_own_inspection=True, independent_moments_CIs_delegate='verify_review.py',
        private_weights_RGB_reference_GT_opened=False, production_imports=0, automatic_retry=False))
    print(json.dumps(dict(frozen=True, own_protocol=binding(output / FILES[0]),
        public_inputs=len(bound), declared_public_bindings=len(expected))), flush=True)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def receipts(folder):
    data = {name: read(folder / (name + '.json')) for name in RECEIPTS}
    for check_name, protocol_name, key in (
        ('VALIDATION_CHECKS', 'VALIDATION_PROTOCOL', 'own_protocol'),
        ('REVIEW_VALIDATION_CHECKS', 'REVIEW_VALIDATION_PROTOCOL', 'own_protocol'),
        ('CPU_CHECKS', 'CPU_PROTOCOL', 'own_protocol'),
        ('CPU_REVIEW_CHECKS', 'CPU_REVIEW_PROTOCOL', 'own_protocol'),
        ('REVIEW_VERIFICATION', 'REVIEW_STATISTICS_VERIFICATION_PROTOCOL', 'protocol'),
        ('RUNTIME_CHECKS', 'RUNTIME_CHECK_PROTOCOL', 'own_protocol'),
        ('SUPPLEMENTAL_FIGURE_CHECKS', 'FIGURE_COMPLETION_PROTOCOL', 'own_protocol'),
        ('POSTHOC_METADATA_CHECKS', 'POSTHOC_METADATA_PROTOCOL', 'own_protocol'),
        ('ARCHIVE_CHECKS', 'ARCHIVE_PROTOCOL', 'protocol')):
        require(same(data[check_name][key], binding(folder / (protocol_name + '.json')))
            and data[check_name]['inputs'] == data[protocol_name]['inputs'],
            'actual own protocol/check input closure differs: ' + check_name)
    for name in ('REVIEW_VALIDATION_CHECKS', 'CPU_REVIEW_CHECKS', 'REVIEW_VERIFICATION',
                 'RUNTIME_CHECKS', 'SUPPLEMENTAL_FIGURE_CHECKS', 'POSTHOC_METADATA_CHECKS', 'ARCHIVE_CHECKS'):
        require(data[name]['complete'] is True and data[name]['passed'] is True, 'actual separate PASS required: ' + name)
    failed = data['VALIDATION_CHECKS']
    require(failed['complete'] is True and failed['passed'] is False and failed['checks'] == 713802
        and failed['failure_count'] == 74 and failed['exception'] is None, 'original scalar FAIL must remain separate')
    require(data['STATISTICS_FIRST_FAILURE']['complete'] is False and data['FIGURES_FAILURE']['complete'] is False,
            'preserve original statistics/render failures')
    require(data['CPU_CHECKS']['passed'] is False, 'preserve original CPU fixture failure')
    require(data['GEOMETRY_SEAL']['complete'] is True and data['GEOMETRY_SEAL']['frames'] == 245
        and data['CONTROL_RECEIPT']['cleanup_completed'] is True and data['SCORING_RECEIPT']['scored_rows'] == 245,
            'complete sealed and scored245 required')
    for source, key, name in (('GEOMETRY_SEAL', 'geometry', 'GEOMETRY_SEALED.jsonl.gz'),
        ('GEOMETRY_SEAL', 'ledgers', 'CONTROL_LEDGERS.jsonl.gz'),
        ('SCORING_RECEIPT', 'predictions', 'PREDICTIONS.jsonl.gz'),
        ('METRICS', 'posthoc_rows', 'POSTHOC_ROWS.jsonl.gz'),
        ('POSTHOC_METADATA_CHECKS', 'rows', 'POSTHOC_METADATA_ROWS.jsonl.gz'),
        ('RUNTIME_RECEIPT', 'raw_rows', 'RUNTIME_ROWS.jsonl.gz'),
        ('ARCHIVE_CHECKS', 'manifest', 'ARCHIVE_MANIFEST.json')):
        require(same(data[source][key], binding(folder / name)), 'actual output authority differs: ' + name)
    counts = data['REVIEW_VERIFICATION']['actual_counts']
    require(all(counts[k] == n for k, n in (('saved_scored_rows_streamed', 1225),
        ('moment_scalars_checked', 810), ('distributions_checked', 135),
        ('metric_CI_slots_checked', 81), ('pair_groups_checked', 27))), 'actual independent statistics scope differs')
    runtime = data['RUNTIME_RECEIPT']; check = data['RUNTIME_CHECKS']
    require(runtime['complete'] is True and runtime['statistics_official'] is True
        and runtime['serialized_rows'] == 150 and runtime['cleanup_errors'] == []
        and check['rows'] == 150 and check['measured'] == 130 and check['summary_slots'] == 56,
            'complete fresh path and independent timing scope required')
    visual = data['ROOT_VISUAL_REVIEW']; figure = data['SUPPLEMENTAL_FIGURE_CHECKS']
    require(visual['complete'] is True and visual['PNG_files'] == 8 and visual['case_panels'] == 24
        and visual['visual_readability_passed'] is True and all(v['actually_viewed_by_root'] is True for v in visual['images'])
        and figure['PNG_files'] == 8 and figure['case_panels'] == 24 and figure['saved_score_rows'] == 1225
        and figure['original_renderer_receipt_passed'] is False, 'separate figure and actual visual authority required')
    return data


def run(args):
    folder, output = guard(args, 'run'); own = read(output / FILES[0])
    require(own['schema'] == 'finite_edge_public_evidence_protocol_v13' and own['input_folder'] == str(folder), 'own roots/schema differ')
    files, bound, expected, skipped = snapshot(folder)
    require(bound == own['inputs'] and expected == own['declared_public_bindings']
        and skipped == own['skipped_external_dependencies'], 'own frozen public input drift')
    write_new(output / FILES[1], dict(own_protocol=binding(output / FILES[0]), inputs=bound,
        public_inspection_attempt=1, new_model_RGB_GT_score_fit_draw_calls=0))
    began = time.perf_counter()
    result = dict(schema='finite_edge_public_evidence_checks_v13', complete=False, passed=False,
        own_protocol=binding(output / FILES[0]), inputs=bound, checker=bound['helper:public_review.py'],
        external_dependencies_opened=0, moments_CIs_recomputed_here=0,
        independent_statistics_command='python -I -S scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/verify_review.py freeze/run --output NEW_DIRECTORY',
        new_model_RGB_GT_score_fit_draw_calls=0, production_modules_imported=False,
        saved_scalar_CSV_is_transcription_only=True, physical_reference_or_ownership_certified=False,
        report_snapshot_text_not_checked=True, failure_count=0, failures=[], automatic_retry=False)
    try:
        data = receipts(folder); metrics = data['METRICS']; score_inputs = metrics['inputs']
        cohort = read(public_path(score_inputs['cohort']['path'])); ids = cohort['ids']
        require(len(ids) == 245 and len(set(ids)) == 245 and cohort['counts']['clean'] == 153
            and cohort['counts']['moderate'] == 92 and cohort['counts']['severe_excluded'] == 74,
            'fixed easy/medium cohort differs')
        by_id = {f['id']: f for f in cohort['frames']}; groups = {method: {} for method in METHODS}
        for alias in ('fixed_scores', 'old_local_scores', 'point_scores', 'new_scores'):
            for row in rows(public_path(score_inputs[alias]['path'])):
                method, fid, pose = row['method'], row['id'], row['pose']
                require(method in groups and fid in by_id and fid not in groups[method], 'unique saved method/cohort ID required')
                require(row['session'] == by_id[fid]['session'] and type(pose['available']) is bool
                    and type(row['new_pose_estimated']) is bool and type(row['fallback_used']) is bool
                    and row['pose_available'] == pose['available']
                    and not (row['new_pose_estimated'] and row['fallback_used']), 'saved session/operational status differs')
                values = {k: pose[k] for k in ('translation_cm', 'rotation_deg', 'ADDsym_m')} if pose['available'] else {}
                require(all(type(v) in (int, float) and math.isfinite(v) for v in values.values()), 'nonfinite saved scalar')
                groups[method][fid] = dict(id=fid, session=row['session'], method=method,
                    available=pose['available'], new_pose_estimated=row['new_pose_estimated'], fallback_used=row['fallback_used'],
                    output_status=row['output_status'], **values)
        require(all(set(group) == set(ids) for group in groups.values()), 'all five methods retain every245 frame')
        status_groups = 0
        for stratum, section in metrics['strata'].items():
            selected = section['ids']; require(set(selected) <= set(ids), 'stratum outside cohort')
            expected_ids = ids if stratum == 'combined' else [fid for fid in ids if by_id[fid]['label'] == {'easy': 'clean', 'medium': 'moderate'}[stratum]]
            require(selected == expected_ids, 'ordered fixed stratum IDs differ')
            for method, group in groups.items():
                selected_rows = [group[fid] for fid in selected]; saved = section['methods'][method]
                actual = dict(denominator=len(selected_rows), operational=sum(r['available'] for r in selected_rows),
                    new_pose=sum(r['new_pose_estimated'] and r['available'] for r in selected_rows),
                    fallback=sum(r['fallback_used'] and r['available'] for r in selected_rows),
                    no_pose=sum(not r['available'] for r in selected_rows),
                    fixed_control=sum(r['output_status'] == 'FRESH_FIXED_CONTROL' for r in selected_rows),
                    statuses=dict(Counter(r['output_status'] for r in selected_rows)))
                require(all(saved[k] == v for k, v in actual.items()), 'saved operational counts differ: ' + stratum + ':' + method)
                status_groups += 1
        for name in ('GEOMETRY_SEALED', 'CONTROL_LEDGERS', 'POSTHOC_ROWS', 'POSTHOC_METADATA_ROWS'):
            observed = [r['id'] for r in rows(folder / (name + '.jsonl.gz'))]
            require(observed == ids, 'ordered245 preserved raw IDs differ: ' + name)
        require(sum(1 for _ in rows(folder / 'FIGURE_CASE_ROWS.jsonl.gz')) == 24, 'saved24 case panels required')
        require(sum(1 for _ in rows(folder / 'RUNTIME_ROWS.jsonl.gz')) == 150, 'saved150 fresh runtime rows required')
        with (output / CSV_NAME).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=('method', 'id', 'session', 'output_status', 'available',
                'new_pose_estimated', 'fallback_used', 'translation_cm', 'rotation_deg', 'ADDsym_m'))
            writer.writeheader()
            for method in METHODS:
                for fid in ids: writer.writerow(groups[method][fid])
            stream.flush(); os.fsync(stream.fileno())
        require(snapshot(folder)[1:] == (bound, expected, skipped), 'public inputs changed during inspection')
        result.update(complete=True, passed=True, scored_rows=1225, methods=5, frames=245,
            geometry_rows=245, ledger_rows=245, posthoc_rows=245, metadata_rows=245, case_panels=24,
            PNG_files=8, status_groups=status_groups, public_inputs=len(bound),
            declared_public_bindings=len(expected), skipped_external_bindings=len(skipped),
            scalar_CSV=binding(output / CSV_NAME), original_failures_preserved=True,
            independent_statistics_receipt=binding(folder / 'REVIEW_VERIFICATION.json'))
    except Exception as error:
        result.update(failure_count=1, failures=[dict(type=type(error).__name__, message=str(error))],
            preserved_partial_CSV=binding(output / CSV_NAME) if (output / CSV_NAME).is_file() else None)
    result['elapsed_seconds'] = time.perf_counter() - began
    write_new(output / FILES[2], result)
    print(json.dumps({k: result.get(k) for k in ('complete', 'passed', 'scored_rows', 'status_groups',
        'public_inputs', 'declared_public_bindings', 'failure_count')}), flush=True)
    if not result['passed']: raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    parser.add_argument('--input', type=Path, default=DOC)
    parser.add_argument('--output', type=Path, required=True, help='new own receipt/CSV directory')
    args = parser.parse_args()
    freeze(args) if args.stage == 'freeze' else run(args)


if __name__ == '__main__': main()
