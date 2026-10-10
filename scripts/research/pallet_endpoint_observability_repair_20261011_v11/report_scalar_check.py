"""Own-frozen inspection of final V11 report scalar transcriptions.

Compare saved report tables with saved evidence, without recomputing statistics,
bootstrap CIs, timing, geometry, truth, fits, models or conditional moments.
Preserve original failures; numerical NEW and completed audits do not prove the
accuracy goal. Supplemental code only until separately frozen and run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_endpoint_observability_repair_20261011_v11'
PRIVATE = Path('/tmp/pallet-endpoint-observability-repair-private-20261011-v11')
OUTPUTS = ('REPORT_SCALAR_PROTOCOL.json', 'REPORT_SCALAR_STARTED.json', 'REPORT_SCALAR_CHECKS.json')
METHODS = dict(BASE='BASE', N3='N3_SUBPIX', POINT='ROLE_BOUNDARY_H_ROBUST',
    LOCAL='ROLE_BOUNDARY_LOCAL_POINT_LINE', FILTER='FILTERED_LOCAL', SAFE='FILTERED_LOCAL_WEAK_SAFE')
STRATA = ('combined', 'easy', 'medium')
SCOPES = ('operational', 'new_pose', 'fallback')
PAIRED = ('common_operational', 'candidate_new_pose', 'both_new_pose')
METRICS = ('translation_cm', 'rotation_deg', 'ADDsym_cm')
MOMENTS = ('mean', 'sample_variance', 'sample_std', 'median', 'P90', 'max')
TIMES = ('full', 'detector', 'N3_correction', 'initial_pose', 'boundary_observation',
    'endpoint_validation_and_filtered_local', 'final_pose_reprojection_and_metadata', 'return')
CONTRASTS = ('FILTER−LOCAL', 'SAFE−FILTER', 'SAFE−LOCAL', 'SAFE−POINT', 'SAFE−N3')
EVIDENCE = ('RESULT_KO.md', 'README.md', 'BUILD_LEDGER.json', 'METHOD_CONTRACT_KO.md',
    'RUNTIME_CONTRACT_KO.md', 'RUNTIME_REVIEW_CONTRACT_KO.md', 'RUNTIME_FORWARD_REVIEW_CONTRACT_KO.md',
    'PROTOCOL.json', 'METRICS.json', 'VERIFICATION.json', 'CONTROL_RECEIPT.json', 'GEOMETRY_SEAL.json',
    'VALIDATION_CHECKS.json', 'REVIEW_VALIDATION_CHECKS.json', 'REVIEW_VALIDATION_PROTOCOL.json',
    'SCORING_RECEIPT.json', 'SCORING_AUTHORITY_SUPPLEMENT.json', 'FIGURE_BINDINGS.json', 'VISUAL_REVIEW.json',
    'RUNTIME_RECEIPT.json', 'runtime_review/RUNTIME_RECEIPT.json',
    'runtime_forward_review/RUNTIME_RECEIPT.json', 'FORWARD_RUNTIME_CHECKS.json', 'FORWARD_RUNTIME_CHECK_PROTOCOL.json')
LIMITS = dict(new_model_GT_pose_score_fit_draw_timing_calls=0, statistics_recomputed=False,
    conditional_moments_recomputed=False, goal_solved=False, physical_truth_certified=False,
    publication_or_fresh_restore_certified=False, preserved_original_failures=True)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_text())


def binding(path):
    path = path.absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'missing/symlink evidence ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def write(path, value):
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
    for path in (source, output):
        require(path.is_dir() and not any(p.is_symlink() for p in (path, *path.parents)), 'existing nonsymlink directory required')
    require(output.resolve() == DOC or output.resolve().is_relative_to(PRIVATE), 'new V11 DOC or PRIVATE outputs only')
    for name in OUTPUTS if args.stage == 'freeze' else OUTPUTS[1:]:
        require(not (output / name).exists(), 'preserve first attempt ' + name)
    files = {name: source / name for name in EVIDENCE}
    files['checker'] = Path(__file__).resolve()
    bindings = {name: binding(path) for name, path in files.items()}
    if args.stage == 'freeze':
        write(output / OUTPUTS[0], dict(schema='supplemental_V11_report_scalar_protocol',
            inputs=bindings, limits=LIMITS, expected_moment_scalars=972, expected_CI_slots=135,
            expected_runtime_n_moments=56, frozen_before_own_checks=True))
        print('V11_REPORT_SCALARS_FROZEN', flush=True)
        return
    own = read(output / OUTPUTS[0])
    require(own['schema'] == 'supplemental_V11_report_scalar_protocol' and own['inputs'] == bindings and
        own['limits'] == LIMITS and own['frozen_before_own_checks'] is True, 'own freeze/code/evidence differs')
    write(output / OUTPUTS[1], dict(protocol=binding(output / OUTPUTS[0]), inputs=bindings, limits=LIMITS))
    result = dict(schema='supplemental_V11_report_scalar_checks', complete=False, passed=False,
        protocol=binding(output / OUTPUTS[0]), inputs=bindings, limits=LIMITS, max_numeric_difference=0.)
    counts = dict(moment_rows=0, moment_scalars=0, moment_denominators=0, operational_rows=0,
        operational_population_cells=0, operational_mean_scalars=0, paired_rows=0, CI_slots=0,
        nonempty_CI_slots=0, None_CI_slots=0, paired_mean_scalars=0, paired_population_cells=0,
        paired_resample_cells=0, runtime_rows=0, runtime_n_moments=0, mask_rows=0,
        mask_population_cells=0, corner_rows=0, corner_population_cells=0, corner_mean_median_P90_scalars=0,
        local_links=0, figure_links=0, figures_actually_viewed=0)
    try:
        m, ledger = read(files['METRICS.json']), read(files['BUILD_LEDGER.json'])
        runtime, checked = read(files['runtime_forward_review/RUNTIME_RECEIPT.json']), read(files['FORWARD_RUNTIME_CHECKS.json'])
        require(m['complete'] is True and m['primary_both_T_R_means_better_than_N3_on_full245'] is False and
            ledger['goal_solved'] is False and ledger['accuracy_and_whole_path_runtime_complete'] is True,
            'completed accuracy/runtime audits with unresolved accuracy goal required')
        for name in ('VERIFICATION.json', 'REVIEW_VALIDATION_CHECKS.json', 'FORWARD_RUNTIME_CHECKS.json'):
            require(read(files[name])['complete'] is True and read(files[name])['passed'] is True, 'actual PASS required ' + name)
        failed = read(files['VALIDATION_CHECKS.json'])
        require(failed['complete'] is False and failed['passed'] is False and failed['checks'] == 95 and
            failed['exception']['message'] == "'best_candidate'", 'preserve original optional-field KeyError95')
        for name, n in (('RUNTIME_RECEIPT.json', 0), ('runtime_review/RUNTIME_RECEIPT.json', 1)):
            r = read(files[name])
            require(r['complete'] is False and r['serialized_rows'] == n and r['status'] == 'FAILED_RUNTIME' and
                r['summaries'] is None and r['statistics_official'] is False, 'preserve original runtime failure ' + name)
        require(runtime['complete'] is True and runtime['statistics_official'] is True and runtime['serialized_rows'] == 150 and
            runtime['model_forwards_after_loading'] == dict(detector=0, N3=0) and
            runtime['model_forwards'] == dict(detector=151, N3=150, detector_internal_initialization_calls=1), 'actual 150 runtime and first-predict preparation')
        require(checked['checks'] == 33905 and checked['rows'] == 150 and checked['measured'] == 130 and
            checked['summary_blocks'] == 8 and checked['summary_slots'] == 56, 'actual runtime independent checks')
        require(read(files['SCORING_RECEIPT.json'])['complete'] is True and
            ledger['stages']['statistics']['actual_arithmetic']['scored_rows'] == 1470 and
            ledger['stages']['statistics']['actual_arithmetic']['moment_scalars'] == 972 and
            ledger['stages']['statistics']['actual_arithmetic']['metric_CIs'] == 135, 'ledger actual score/stat scope')
        visual = read(files['VISUAL_REVIEW.json'])
        require(visual['complete'] is True and visual['PNG_files_inspected'] == 8 and
            visual['case_panels_inspected'] == 36 and visual['plots_inspected'] == 2 and
            visual['titles_legends_readable'] is True and visual['panel_titles_do_not_overlap_images'] is True, 'actual eight-image visual review')
        counts['figures_actually_viewed'] = visual['PNG_files_inspected']
        text = files['RESULT_KO.md'].read_text()
        require(not re.search('[\u3040-\u30ff]', text) and '<!--' not in text, 'no Japanese residue or hidden placeholder')
        require(all(value in text for value in ('아직 아니다', '2164', '9000', '1204.8027435564068',
            '221.7114377170665', 'wood_night_01:031344', '첫 예측 warmup 내부', '모델 로딩 직후 forward',
            '149.6915392153846', '33905')), 'history, large failures and first-predict preparation disclosed')
        seen = {key: set() for key in ('moment', 'operational', 'CI', 'runtime', 'mask', 'corner')}
        def num(actual, expected, label):
            if expected is None:
                require(actual == 'None', 'None transcription ' + label)
                return
            value = float(actual)
            require(math.isfinite(value) and math.isfinite(expected), 'finite transcription ' + label)
            diff = abs(value - expected)
            result['max_numeric_difference'] = max(result['max_numeric_difference'], diff)
            require(diff <= 1e-12 + 1e-12 * abs(expected), 'scalar transcription ' + label)
        for line in text.splitlines():
            if not line.startswith('|'):
                continue
            c = [value.strip() for value in line.strip('|').split('|')]
            if len(c) == 11 and c[0] in STRATA and c[1] in METHODS:
                key = (c[0], METHODS[c[1]])
                require(key not in seen['operational'], 'duplicate operational row')
                seen['operational'].add(key)
                block = m['strata'][key[0]]['methods'][key[1]]
                fields = ('denominator', 'operational', 'new_pose', 'fallback', 'no_pose', 'fixed_control')
                require([int(v) for v in c[2:8]] == [block[k] for k in fields], 'operational counts')
                for actual, metric in zip(c[8:], METRICS):
                    num(actual, block['metrics']['operational'][metric]['mean'], str(key) + metric)
                counts['operational_rows'] += 1; counts['operational_population_cells'] += 6; counts['operational_mean_scalars'] += 3
            elif len(c) == 7 and c[0] in STRATA and c[1] in METHODS and c[2] in SCOPES:
                key = (c[0], METHODS[c[1]], c[2])
                require(key not in seen['moment'], 'duplicate moment row')
                seen['moment'].add(key)
                block = m['strata'][key[0]]['methods'][key[1]]['metrics'][key[2]]
                for cell, metric in zip(c[4:], METRICS):
                    values = cell.split('<br>'); require(len(values) == 6, 'all six moment fields required')
                    require(int(c[3]) == block[metric]['n'], 'moment denominator')
                    for actual, field in zip(values, MOMENTS):
                        num(actual, block[metric][field], str(key) + metric + field)
                        counts['moment_scalars'] += 1
                counts['moment_rows'] += 1; counts['moment_denominators'] += 1
            elif len(c) == 11 and c[0] in STRATA and c[1] in CONTRASTS and c[2] in PAIRED:
                left, right = c[1].split('−'); contrast = METHODS[left] + '_minus_' + METHODS[right]
                key = (c[0], contrast, c[2]); require(key not in seen['CI'], 'duplicate paired row'); seen['CI'].add(key)
                block = m['strata'][key[0]]['contrasts'][key[1]][key[2]]
                values = [block['metrics'][METRICS[0]]['n'], block['denominator'],
                    block['candidate_new_pose_in_pairs'], block['comparator_new_pose_in_pairs']]
                require([int(v) for v in c[3:7]] == values, 'paired counts')
                require([int(v) for v in c[7].split('/')] == [block['bootstrap_nonempty_resamples'],
                    block['bootstrap_empty_resamples']], 'paired bootstrap nonempty/empty denominators')
                for cell, metric in zip(c[8:], METRICS):
                    parts = cell.split('<br>'); require(len(parts) == 2, 'paired mean and CI required')
                    data = block['metrics'][metric]; num(parts[0], data['mean_delta'], str(key) + metric)
                    if data['CI95'] is None:
                        require(parts[1] == 'None', 'empty CI'); counts['None_CI_slots'] += 1
                    else:
                        endpoints = json.loads(parts[1]); require(len(endpoints) == 2, 'two CI endpoints')
                        for actual, expected in zip(endpoints, data['CI95']):
                            num(str(actual), expected, str(key) + metric + '.CI')
                        counts['nonempty_CI_slots'] += 1
                    counts['CI_slots'] += 1; counts['paired_mean_scalars'] += 1
                counts['paired_rows'] += 1; counts['paired_population_cells'] += 4; counts['paired_resample_cells'] += 2
            elif len(c) == 8 and c[0] in TIMES:
                key = c[0]; require(key not in seen['runtime'], 'duplicate runtime row'); seen['runtime'].add(key)
                block = runtime['summaries'][key]; require(int(c[1]) == block['n'], 'runtime n')
                for actual, field in zip(c[2:], (*MOMENTS[:-1], 'maximum')):
                    num(actual, block[field], 'runtime ' + key + field)
                counts['runtime_rows'] += 1; counts['runtime_n_moments'] += 7
            elif len(c) == 6 and c[0] in ('FILTER', 'SAFE'):
                key = (METHODS[c[0]], c[1] + '__' + c[2]); require(key not in seen['mask'], 'duplicate mask row'); seen['mask'].add(key)
                block = m['strata']['combined']['diagnostics'][key[0]]['mask_and_pose_groups_vs_fixed_N3'][key[1]]
                require([int(v) for v in c[3:]] == [block[k] for k in ('n', 'new_pose', 'fallback')], 'mask/pose counts')
                counts['mask_rows'] += 1; counts['mask_population_cells'] += 3
            elif len(c) == 9 and c[0] in ('FILTER', 'SAFE'):
                key = (METHODS[c[0]], c[1]); require(key not in seen['corner'], 'duplicate corner row'); seen['corner'].add(key)
                block = m['strata']['combined']['diagnostics'][key[0]][key[1]]
                require(int(c[2]) == block['corners'], 'corner count')
                for index, field in enumerate(('mean', 'median', 'P90')):
                    for offset, version in enumerate(('N3', 'output')):
                        num(c[3 + index * 2 + offset], block['errors'][version][field], str(key) + version + field)
                        counts['corner_mean_median_P90_scalars'] += 1
                counts['corner_rows'] += 1; counts['corner_population_cells'] += 1
        require(counts['moment_rows'] == 54 and counts['moment_scalars'] == 972 and counts['operational_rows'] == 18 and
            counts['paired_rows'] == 45 and counts['CI_slots'] == 135 and counts['nonempty_CI_slots'] == 126 and
            counts['None_CI_slots'] == 9 and counts['runtime_rows'] == 8 and counts['runtime_n_moments'] == 56 and
            counts['mask_rows'] == 12 and counts['corner_rows'] == 6, 'all fixed report table cells required')
        figure_targets = set()
        for name in ('RESULT_KO.md', 'README.md', 'METHOD_CONTRACT_KO.md', 'RUNTIME_CONTRACT_KO.md',
            'RUNTIME_REVIEW_CONTRACT_KO.md', 'RUNTIME_FORWARD_REVIEW_CONTRACT_KO.md'):
            body = files[name].read_text()
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', body):
                if target.startswith(('https://', 'http://', '#')):
                    continue
                relative, _, fragment = target.partition('#'); path = (source / relative).resolve()
                require(path.is_file(), 'missing local link ' + target)
                if fragment.startswith('L') and fragment[1:].isdigit():
                    require(int(fragment[1:]) <= len(path.read_text().splitlines()), 'line fragment beyond EOF')
                if name == 'RESULT_KO.md' and relative.startswith('figures/'):
                    figure_targets.add(relative); counts['figure_links'] += 1
                counts['local_links'] += 1
        require(figure_targets == {'figures/01_all_operational.png', 'figures/02_same_frame_pairs.png',
            *('figures/case_' + str(i).zfill(2) + '.png' for i in range(1, 7))}, 'all eight fixed figures linked')
        require({name: binding(path) for name, path in files.items()} == bindings, 'report/code/evidence changed during own inspection')
        result.update(complete=True, passed=True)
    except Exception as error:
        result['error'] = dict(type=type(error).__name__, message=str(error))
    result['actual_counts'] = counts
    write(output / OUTPUTS[2], result)
    print(json.dumps(dict(passed=result['passed'], counts=counts)), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
