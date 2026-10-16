"""Check final report transcriptions using bound saved evidence, independently.

This supplemental stdlib inspector checks 648 main moment scalars, 81 stored CI
slots, 196 runtime n/moment values, operational and diagnostic table entries and
local links. It does not recompute any source truth, geometry, moment statistic,
bootstrap interval, timing, conditional group moment, model or pose score.
Freeze final report/code/evidence before the one report-inspection pass.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_sparse_local_line_20261010_v9'
NAMES = ('REPORT_SCALAR_PROTOCOL.json', 'REPORT_SCALAR_STARTED.json', 'REPORT_SCALAR_CHECKS.json')
METHODS = {'BASE': 'BASE', 'N3': 'N3_SUBPIX', 'POINT': 'ROLE_BOUNDARY_H_ROBUST',
           'LOCAL': 'ROLE_BOUNDARY_LOCAL_POINT_LINE'}
STRATA = ('combined', 'easy', 'medium')
METRICS = ('translation_cm', 'rotation_deg', 'ADDsym_cm')
FIELDS = ('mean', 'sample_variance', 'sample_std', 'median', 'P90', 'max')
TIMING_FIELDS = ('mean', 'sample_variance', 'sample_std', 'median', 'P90', 'maximum')
SCOPES = ('operational', 'new_pose', 'fallback')
PAIR_SCOPES = ('common_operational', 'candidate_new_pose', 'both_new_pose')
EVIDENCE = ('RESULT_KO.md', 'README.md', 'BUILD_LEDGER.json', 'REPRODUCE.md',
    'EVALUATION_CONTRACT_KO.md', 'PROTOCOL.json', 'METRICS.json', 'VERIFICATION.json',
    'GEOMETRY_SEAL.json', 'CONTROL_RECEIPT.json', 'VALIDATION_CHECKS.json',
    'SCORING_RECEIPT.json', 'SCORING_PARITY.json', 'JOINED_LOCAL_CONTRACT_CHECKS.json',
    'LOCAL_LINE_CHECKS.json', 'LOCAL_LINE_REVIEW_CHECKS.json', 'MASK_POSE_AUDIT_CHECKS.json',
    'SOURCE_ROLE_CHECKS.json', 'RUNTIME_REVIEW.json', 'RUNTIME_VALIDATION_CHECKS.json',
    'RUNTIME_ANCHOR_REVIEW_CHECKS.json', 'JOINED_RUNTIME_CONTRACT_CHECKS.json',
    'FIGURE_BINDINGS.json', 'FIGURE_LAYOUT_CHECKS.json', 'ROOT_VISUAL_REVIEW.json',
    'VISUAL_REVIEW_6.json', 'VISUAL_LAYOUT_REVIEW.json', 'PUBLIC_REVIEW_CHECKS.json',
    'PUBLIC_FRESH_BEFORE.json', 'PUBLIC_FRESH_RESTORE_CHECKS.json',
    'PUBLIC_FRESH_PUBLIC_REVIEW_CHECKS.json', 'PUBLIC_FRESH_BUNDLE_CHECKS.json',
    'ARCHIVE_MANIFEST.json', 'ARCHIVE_CHECKS.json', 'PROTECTION_AFTER.json',
    'PROTECTION_CLI_ATTEMPT_A.json', 'PROTECTION_CLI_ATTEMPT_B.json',
    'VALIDATION_CLI_ATTEMPT_A.json', 'RENDER_CLI_ATTEMPT_A.json', 'RUNTIME_CLI_ATTEMPT_A.json')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_text())


def binding(path):
    path = path.absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'missing/symlink evidence: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    parser.add_argument('--input', type=Path, default=DOC)
    parser.add_argument('--output', type=Path, default=DOC)
    args = parser.parse_args()
    source, output = args.input.absolute(), args.output.absolute()
    require(source.is_dir() and not any(p.is_symlink() for p in (source, *source.parents)),
            'real input directory required')
    require(not any(p.is_symlink() for p in (output, *output.parents)), 'symlink output refused')
    output.mkdir(parents=True, exist_ok=True)
    for name in NAMES if args.stage == 'freeze' else NAMES[1:]:
        require(not (output / name).exists(), 'preserve completed/failed report attempt: ' + name)
    files = {name: source / name for name in EVIDENCE}
    files['checker'] = Path(__file__).resolve()
    bindings = {name: binding(path) for name, path in files.items()}
    if args.stage == 'freeze':
        write_new(output / NAMES[0], dict(schema='supplemental_V9_final_report_scalar_protocol',
            inputs=bindings, frozen_before_own_report_checks=True, relative_tolerance=1e-12,
            absolute_tolerance=1e-12, expected_main_moment_scalars=648, expected_stored_CI_slots=81,
            expected_runtime_n_and_moment_values=196, new_model_GT_pose_score_fit_draw_timing_calls=0,
            extra_conditional_moments_checked_by_transcription_only=True))
        print('V9_REPORT_SCALARS_FROZEN', flush=True)
        return
    own = read(output / NAMES[0])
    require(own['schema'] == 'supplemental_V9_final_report_scalar_protocol' and
            own['inputs'] == bindings, 'final report/code/evidence differs from own freeze')
    write_new(output / NAMES[1], dict(protocol=binding(output / NAMES[0]), actual_report_checks=1,
        new_model_GT_pose_score_fit_draw_timing_calls=0))
    counts = dict(main_moment_rows=0, main_moment_scalars=0, main_moment_denominators=0,
        paired_CI_rows=0, paired_CI_slots=0, paired_CI_nonempty=0, paired_CI_None=0,
        paired_population_cells=0, paired_mean_delta_scalars=0, paired_resample_cells=0,
        runtime_rows=0, runtime_n_and_moment_values=0, operational_rows=0,
        operational_population_cells=0, operational_mean_scalars=0,
        local_diagnostic_rows=0, local_diagnostic_population_and_outcome_cells=0,
        local_diagnostic_mean_transcriptions=0, corrected_mask_rows=0,
        corrected_mask_population_cells=0, corrected_mask_mean_transcriptions=0,
        local_links=0, report_images=0)
    result = dict(schema='supplemental_V9_final_report_scalar_checks', complete=False,
        passed=False, protocol=binding(output / NAMES[0]), inputs=bindings,
        arithmetic_scope='saved report/evidence scalar comparison only; no statistics recomputation',
        conditional_moments_independently_recomputed=False, new_model_GT_pose_score_fit_draw_timing_calls=0,
        original_failed_attempts_preserved=True, independent_physical_truth_certified=False,
        goal_complete=False, max_numeric_difference=0)
    try:
        m, ledger = read(files['METRICS.json']), read(files['BUILD_LEDGER.json'])
        mask, runtime = read(files['MASK_POSE_AUDIT_CHECKS.json']), read(files['RUNTIME_REVIEW.json'])
        require(m['complete'] and ledger['complete'] and ledger['goal_complete'] is False,
                'complete V9 experiment with unresolved accuracy goal required')
        for name in ('VERIFICATION.json', 'VALIDATION_CHECKS.json', 'SCORING_PARITY.json',
                'JOINED_LOCAL_CONTRACT_CHECKS.json', 'LOCAL_LINE_REVIEW_CHECKS.json',
                'MASK_POSE_AUDIT_CHECKS.json', 'SOURCE_ROLE_CHECKS.json',
                'RUNTIME_ANCHOR_REVIEW_CHECKS.json', 'JOINED_RUNTIME_CONTRACT_CHECKS.json',
                'FIGURE_LAYOUT_CHECKS.json', 'VISUAL_LAYOUT_REVIEW.json',
                'PUBLIC_REVIEW_CHECKS.json', 'PUBLIC_FRESH_RESTORE_CHECKS.json',
                'PUBLIC_FRESH_PUBLIC_REVIEW_CHECKS.json', 'PUBLIC_FRESH_BUNDLE_CHECKS.json',
                'ARCHIVE_CHECKS.json', 'PROTECTION_AFTER.json'):
            require(read(files[name])['passed'] is True, 'actual successful authority required: ' + name)
        require(read(files['LOCAL_LINE_CHECKS.json'])['passed'] is False and
                read(files['RUNTIME_VALIDATION_CHECKS.json'])['passed'] is False,
                'original failed CPU/runtime receipts must remain false')
        joined = read(files['JOINED_RUNTIME_CONTRACT_CHECKS.json'])
        require(joined['original_checks'] == 112104 and joined['original_failure_count'] == 300 and
                joined['original_accepted_unchanged_nonanchor_checks'] == 111804 and
                joined['corrected_source_anchor_predicates'] == 300 and
                joined['original196moment_slots_recomputed'] == 0, 'honest targeted runtime join')
        require(read(files['VISUAL_LAYOUT_REVIEW.json'])['actual_images_viewed'] == 6 and
                read(files['ROOT_VISUAL_REVIEW.json'])['actual_view_image_calls'] >= 2,
                'actual case and aggregate visual witnesses required')
        require(read(files['PUBLIC_FRESH_RESTORE_CHECKS.json'])['actual_created_files'] == 10 and
                read(files['PUBLIC_FRESH_RESTORE_CHECKS.json'])['existing_unchanged_files'] == 0,
                'fresh absence/actual creation required')
        require(ledger['completed_geometry']['actual_runs'] == 1 and
                ledger['pose_scoring']['actual_total_rows'] == 980 and
                ledger['statistics']['moment_scalars'] == 648 and
                ledger['statistics']['CI_slots'] == 81 and
                ledger['whole_path_runtime']['fresh_total_calls'] == 600 and
                ledger['runtime_independent_review']['original_passed'] is False and
                ledger['runtime_independent_review']['joined_passed'] is True,
                'ledger actual runs, denominators and preserved failures')
        text = files['RESULT_KO.md'].read_text()
        require(not re.search('[\u3040-\u30ff]', text) and '<!--' not in text,
                'no Japanese residue or draft placeholders')
        require('1204.8027435564068' in text and 'wood_night_01:031426' in text and
                '2164' in text and '9000' in text and 'historical_pose_identity_verified=False' in text,
                'large failure, history and source identity limitation disclosed')
        seen = {key: set() for key in ('moment', 'CI', 'operational', 'runtime', 'local', 'mask')}
        def num(actual, expected, label):
            if expected is None:
                require(actual == 'None', 'empty scalar: ' + label)
                return
            value = float(actual)
            require(math.isfinite(value) and math.isfinite(expected), 'finite scalar: ' + label)
            diff = abs(value - expected)
            result['max_numeric_difference'] = max(result['max_numeric_difference'], diff)
            require(diff <= 1e-12 + 1e-12 * abs(expected), 'scalar transcription: ' + label)
        for line in text.splitlines():
            if not line.startswith('|'):
                continue
            c = [x.strip() for x in line.strip('|').split('|')]
            if len(c) == 11 and c[0] in STRATA and c[1] in METHODS:
                stratum, method = c[0], METHODS[c[1]]
                block = m['strata'][stratum]['methods'][method]
                if c[2] in SCOPES and c[3] in METRICS:
                    key = (stratum, method, c[2], c[3])
                    require(key not in seen['moment'], 'duplicate main moment')
                    seen['moment'].add(key)
                    data = block['metrics'][c[2]][c[3]]
                    require(int(c[4]) == data['n'], 'main moment denominator')
                    for actual, field in zip(c[5:], FIELDS):
                        num(actual, data[field], str(key) + field)
                        counts['main_moment_scalars'] += 1
                    counts['main_moment_rows'] += 1
                    counts['main_moment_denominators'] += 1
                else:
                    key = (stratum, method)
                    require(key not in seen['operational'], 'duplicate operational row')
                    seen['operational'].add(key)
                    fields = ('denominator', 'operational', 'new_pose', 'fallback', 'no_pose', 'fixed_control')
                    require([int(x) for x in c[2:8]] == [block[x] for x in fields],
                            'operational denominator/status cells')
                    for actual, metric in zip(c[8:], METRICS):
                        num(actual, block['metrics']['operational'][metric]['mean'], 'operational mean')
                    counts['operational_rows'] += 1
                    counts['operational_population_cells'] += 6
                    counts['operational_mean_scalars'] += 3
            elif len(c) == 11 and c[0] in STRATA and c[1] in ('LOCAL−BASE', 'LOCAL−N3', 'LOCAL−POINT'):
                a, b = c[1].split('−')
                contrast = METHODS[a] + '_minus_' + METHODS[b]
                require(c[2] in PAIR_SCOPES and c[3] in METRICS, 'fixed paired scope/metric')
                key = (c[0], contrast, c[2], c[3])
                require(key not in seen['CI'], 'duplicate CI row')
                seen['CI'].add(key)
                block = m['strata'][c[0]]['contrasts'][contrast][c[2]]
                data = block['metrics'][c[3]]
                require([int(c[4]), int(c[5])] == [block['denominator'], data['n']], 'paired denominators')
                num(c[6], data['mean_delta'], 'paired mean delta')
                if data['CI95'] is None:
                    require(c[7:9] == ['None', 'None'], 'None CI endpoints')
                    counts['paired_CI_None'] += 1
                else:
                    for actual, expected in zip(c[7:9], data['CI95']):
                        num(actual, expected, 'stored CI endpoint')
                    counts['paired_CI_nonempty'] += 1
                require([int(x) for x in c[9:]] == [block['bootstrap_nonempty_resamples'],
                        block['bootstrap_empty_resamples']], 'stored bootstrap denominator/empty counts')
                counts['paired_CI_rows'] += 1
                counts['paired_CI_slots'] += 1
                counts['paired_population_cells'] += 2
                counts['paired_mean_delta_scalars'] += 1
                counts['paired_resample_cells'] += 2
            elif len(c) == 9 and c[0] in METHODS:
                method, stage = METHODS[c[0]], c[1]
                key = (method, stage)
                require(key not in seen['runtime'] and stage in runtime['summaries'][method],
                        'unique saved runtime stage')
                seen['runtime'].add(key)
                block = runtime['summaries'][method][stage]
                require(int(c[2]) == block['n'], 'runtime denominator')
                for actual, field in zip(c[3:], TIMING_FIELDS):
                    num(actual, block[field], 'runtime stage ' + str(key) + field)
                counts['runtime_rows'] += 1
                counts['runtime_n_and_moment_values'] += 7
            elif len(c) == 15 and c[0] in STRATA and c[1] in m['strata'][c[0]]['local_point_line_diagnostics']['groups']:
                key = (c[0], c[1])
                require(key not in seen['local'], 'duplicate local diagnostic row')
                seen['local'].add(key)
                block = m['strata'][c[0]]['local_point_line_diagnostics']['groups'][c[1]]
                saved = block['saved_actual_pose_and_status']
                expected = [block['frames']] + [saved[k] for k in ('new_pose', 'fallback', 'no_pose')]
                for target in ('N3', 'point'):
                    outcomes = block['actual_pose_outcomes_vs_' + target]
                    expected.extend(outcomes.get(k, 0) for k in ('BOTH_BETTER', 'BOTH_WORSE', 'MIXED_OR_UNCHANGED'))
                require([int(x) for x in c[2:12]] == expected, 'local counts and actual T/R outcomes')
                for actual, metric in zip(c[12:], METRICS):
                    num(actual, saved['metrics']['operational'][metric]['mean'], 'local producer mean transcription')
                counts['local_diagnostic_rows'] += 1
                counts['local_diagnostic_population_and_outcome_cells'] += 10
                counts['local_diagnostic_mean_transcriptions'] += 3
            elif len(c) == 9 and c[0] in STRATA and c[1] in mask['strata'][c[0]]['groups']:
                key = (c[0], c[1])
                require(key not in seen['mask'], 'duplicate corrected mask row')
                seen['mask'].add(key)
                block = mask['strata'][c[0]]['groups'][c[1]]
                require([int(x) for x in c[2:6]] == [block[k] for k in ('frames', 'NEW', 'fallback', 'no_pose')],
                        'actual corrected mask populations')
                for actual, metric in zip(c[6:], METRICS):
                    num(actual, block['actual_saved_pose'][metric]['mean'], 'mask producer mean transcription')
                counts['corrected_mask_rows'] += 1
                counts['corrected_mask_population_cells'] += 4
                counts['corrected_mask_mean_transcriptions'] += 3
        require(len(seen['moment']) == 108 and counts['main_moment_scalars'] == 648 and
                len(seen['CI']) == 81 and counts['paired_CI_nonempty'] == 63 and counts['paired_CI_None'] == 18 and
                len(seen['operational']) == 12 and len(seen['runtime']) == 28 and
                counts['runtime_n_and_moment_values'] == 196 and len(seen['local']) == 33 and
                len(seen['mask']) == 48, 'all fixed final report table cells required')
        for name in ('RESULT_KO.md', 'README.md', 'REPRODUCE.md', 'EVALUATION_CONTRACT_KO.md'):
            body = files[name].read_text()
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', body):
                if target.startswith(('https://', 'http://', '#')):
                    continue
                relative, _, fragment = target.partition('#')
                path = (source / relative).resolve()
                require(path.is_file(), 'missing local link: ' + target)
                if fragment.startswith('L') and fragment[1:].isdigit():
                    require(int(fragment[1:]) <= len(path.read_text().splitlines()), 'line fragment beyond EOF')
                counts['local_links'] += 1
            if name == 'RESULT_KO.md':
                images = re.findall(r'!\[[^\]]*\]\((reviewed_figures/[^)]+)\)', body)
                require(len(images) == 8 and len(set(images)) == 8, 'eight reviewed images in report')
                counts['report_images'] = len(images)
        require({name: binding(path) for name, path in files.items()} == bindings,
                'report/source/evidence changed during own inspection')
        result.update(complete=True, passed=True)
    except Exception as error:
        result['error'] = dict(type=type(error).__name__, message=str(error))
    result['actual_counts'] = counts
    write_new(output / NAMES[2], result)
    print(json.dumps(dict(passed=result['passed'], counts=counts)), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
