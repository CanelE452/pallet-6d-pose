"""Independent stdlib check of the report's tables, links and frozen receipts.

This supplement compares rendered numbers to already independently checked
METRICS. It does not score poses, regenerate CIs, fit geometry or import any
experiment module. Own freeze must precede one report arithmetic pass.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_same_observation_controls_20261010_v8'
LABELS = {'Base': 'BASE', 'N3→cornerSubPix': 'N3_SUBPIX',
    'H+강건': 'ROLE_BOUNDARY_H_ROBUST', 'H+일반': 'ROLE_BOUNDARY_H_STANDARD',
    '마스크 없음+강건': 'ROLE_BOUNDARY_NO_MASK_ROBUST'}
METRICS = ('translation_cm', 'rotation_deg', 'ADDsym_cm')
FIELDS = ('mean', 'sample_variance', 'sample_std', 'median', 'P90', 'max')
NAMES = ('REPORT_VALIDATION_PROTOCOL.json', 'REPORT_VALIDATION_STARTED.json', 'REPORT_CHECKS.json')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_text())


def binding(path):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'missing/symlink input')
    return dict(path=str(path.relative_to(REPO)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def inputs():
    files = {name: DOC / name for name in ('RESULT_KO.md', 'README.md', 'BUILD_LEDGER.json',
        'REPRODUCE.md', 'EVALUATION_CONTRACT_KO.md', 'METRICS.json', 'VERIFICATION.json',
        'PUBLIC_REVIEW_CHECKS.json', 'PUBLIC_FRESH_BUNDLE_CHECKS.json', 'ARCHIVE_CHECKS.json',
        'VISUAL_REVIEW_ROOT2.json', 'VISUAL_REVIEW_6.json', 'PROTOCOL.json')}
    files['checker'] = Path(__file__).resolve()
    return files


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def guard(stage):
    for name in NAMES if stage == 'freeze' else NAMES[1:]:
        require(not (DOC / name).exists(), 'preserve existing report inspection ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    args = parser.parse_args()
    guard(args.stage)
    files = inputs()
    bindings = {name: binding(path) for name, path in files.items()}
    if args.stage == 'freeze':
        write_new(DOC / NAMES[0], dict(schema='independent_report_transcription_protocol_v8',
            inputs=bindings, frozen_before_own_report_arithmetic=True,
            decimal_rounding_absolute_tolerance=5.00001e-7,
            report_accuracy_policy_or_raw_rows_modified=0, new_geometry_GT_model_draw_score_calls=0))
        print('REPORT_TRANSCRIPTION_FROZEN', flush=True)
        return
    own = read(DOC / NAMES[0])
    require(own['inputs'] == bindings, 'frozen report/code/evidence differs')
    write_new(DOC / NAMES[1], dict(protocol=binding(DOC / NAMES[0]), actual_report_checks=1,
        new_geometry_GT_model_draw_score_calls=0))
    counts = dict(moment_tables=0, moment_denominators=0, moment_scalars=0,
        paired_groups=0, paired_population_cells=0, paired_mean_scalars=0,
        paired_CI_slots=0, operational_rows=0, operational_population_cells=0,
        operational_mean_scalars=0, local_links=0)
    result = dict(complete=False, passed=False, inputs=bindings, protocol=binding(DOC / NAMES[0]),
        new_geometry_GT_model_draw_score_calls=0, independent_physical_truth_certified=False)
    try:
        metrics, ledger = read(files['METRICS.json']), read(files['BUILD_LEDGER.json'])
        require(ledger['goal_complete'] is False, 'unresolved goal reported complete')
        require(read(files['VERIFICATION.json'])['passed'] is True and
            read(files['PUBLIC_REVIEW_CHECKS.json'])['passed'] is True and
            read(files['PUBLIC_FRESH_BUNDLE_CHECKS.json'])['passed'] is True and
            read(files['ARCHIVE_CHECKS.json'])['passed'] is True, 'required completed audits')
        require(read(files['VISUAL_REVIEW_ROOT2.json'])['actual_images_viewed'] == 2 and
            read(files['VISUAL_REVIEW_6.json'])['actual_local_images_viewed'] == 6, 'eight actual visual inspections')
        def number(actual, expected, label):
            if expected is None:
                require(actual == 'None', 'empty scalar ' + label)
            else:
                require(math.isfinite(float(actual)) and abs(float(actual)-expected) <= 5.00001e-7,
                        'rounded scalar ' + label)
        context = None
        seen_moments, seen_pairs, seen_operational = set(), set(), set()
        for line in files['RESULT_KO.md'].read_text().splitlines():
            heading = re.fullmatch(r'### (combined|easy|medium) / ([a-z_]+)', line)
            if heading:
                context = heading.groups()
            if not line.startswith('|'):
                continue
            cells = [x.strip() for x in line.strip('|').split('|')]
            if len(cells) == 10 and cells[0] in ('combined', 'easy', 'medium') and cells[1] in LABELS:
                stratum, method = cells[0], LABELS[cells[1]]
                require((stratum, method) not in seen_operational, 'duplicate operational report row')
                seen_operational.add((stratum, method))
                block = metrics['strata'][stratum]['methods'][method]
                expected = [block[k] for k in ('denominator', 'operational', 'new_pose', 'fallback', 'no_pose')]
                require([int(x) for x in cells[2:7]] == expected, 'operational population transcription')
                for metric, actual in zip(METRICS, cells[7:]):
                    number(actual, block['metrics']['operational'][metric]['mean'], 'operational mean')
                    counts['operational_mean_scalars'] += 1
                counts['operational_rows'] += 1
                counts['operational_population_cells'] += 5
            elif len(cells) == 9 and cells[0] in LABELS and cells[1] in METRICS:
                require(context and context[1] in ('operational', 'new_pose', 'fallback'), 'moment heading')
                stratum, scope = context
                method, metric = LABELS[cells[0]], cells[1]
                key = (stratum, scope, method, metric)
                require(key not in seen_moments, 'duplicate moment report row')
                seen_moments.add(key)
                block = metrics['strata'][stratum]['methods'][method]['metrics'][scope][metric]
                require(int(cells[2]) == block['n'], 'moment n transcription')
                for field, actual in zip(FIELDS, cells[3:]):
                    number(actual, block[field], 'moment ' + str(key) + field)
                    counts['moment_scalars'] += 1
                counts['moment_denominators'] += 1
            elif len(cells) == 10 and ' − ' in cells[0]:
                require(context and context[1] in ('common_operational', 'candidate_new_pose', 'both_new_pose'), 'paired heading')
                a, b = cells[0].split(' − ')
                require(a in LABELS and b in LABELS, 'known contrast labels')
                contrast = LABELS[a] + '_minus_' + LABELS[b]
                stratum, scope = context
                key = (stratum, scope, contrast)
                require(key not in seen_pairs, 'duplicate contrast report row')
                seen_pairs.add(key)
                block = metrics['strata'][stratum]['contrasts'][contrast][scope]
                require([int(x) for x in cells[1:4]] == [block[k] for k in
                    ('common_frames', 'candidate_new_pose_in_pairs', 'comparator_new_pose_in_pairs')], 'paired population')
                for metric, actual, interval in zip(METRICS, cells[4::2], cells[5::2]):
                    expected = block['metrics'][metric]
                    number(actual, expected['mean_delta'], 'paired mean ' + str(key))
                    counts['paired_mean_scalars'] += 1
                    if expected['CI95'] is None:
                        require(interval == 'None', 'empty CI transcription')
                    else:
                        limits = json.loads(interval)
                        require(len(limits) == 2, 'two CI endpoints')
                        for value, truth in zip(limits, expected['CI95']):
                            number(str(value), truth, 'paired CI')
                    counts['paired_CI_slots'] += 1
                counts['paired_groups'] += 1
                counts['paired_population_cells'] += 3
        require(len(seen_moments) == 135 and counts['moment_scalars'] == 810 and
            len(seen_pairs) == 72 and counts['paired_CI_slots'] == 216 and
            len(seen_operational) == 15, 'complete report tables required')
        counts['moment_tables'] = 9
        for name in ('RESULT_KO.md', 'README.md', 'REPRODUCE.md', 'EVALUATION_CONTRACT_KO.md'):
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', files[name].read_text()):
                if target.startswith(('https://', 'http://', '#')):
                    continue
                relative, _, fragment = target.partition('#')
                path = (DOC / relative).resolve()
                require(path.is_file(), 'missing local report link ' + target)
                if fragment.startswith('L') and fragment[1:].isdigit():
                    require(int(fragment[1:]) <= len(path.read_text().splitlines()), 'code line beyond EOF')
                counts['local_links'] += 1
        require({name: binding(path) for name, path in files.items()} == bindings,
                'report/evidence changed during inspection')
        result.update(complete=True, passed=True)
    except Exception as error:
        result['error'] = dict(type=type(error).__name__, message=str(error))
    result['actual_counts'] = counts
    write_new(DOC / NAMES[2], result)
    print(json.dumps(dict(passed=result['passed'], actual_counts=counts)), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
