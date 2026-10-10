"""Independent public check of V7 report transcription and local links.

This supplemental audit is frozen after completed accuracy and before its own
arithmetic. It reads saved METRICS/RUNTIME, not models, images, GT or PnP.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_three_head_observation_20261010_v7'
NAMES = ('RESULT_KO.md', 'REPRODUCE.md', 'BUILD_LEDGER.json', 'README.md',
         'METRICS.json', 'RUNTIME.json', 'FIGURE_BINDINGS.json')
ALIASES = {'BASE': 'BASE', 'N3': 'N3_SUBPIX', 'ROLE_B': 'IMAGE_ROLE_BOUNDARY_ONLY',
           'GEOM_B': 'GEOMETRY_ONLY_BOUNDARY_ONLY', 'NO_ROLE_B': 'IMAGE_NO_ROLE_BOUNDARY_ONLY',
           'GEOM_H': 'GEOMETRY_ONLY_CORNERWISE_HYBRID',
           'NO_ROLE_H': 'IMAGE_NO_ROLE_CORNERWISE_HYBRID',
           'ROLE_H': 'IMAGE_ROLE_CORNERWISE_HYBRID', 'NATIVE_H': 'N3_INDEPENDENT_ROBUST_H',
           'NATIVE_NO_MASK': 'N3_INDEPENDENT_ROBUST_NO_MASK'}
METRICS = {'T(cm)': 'translation_cm', 'R(°)': 'rotation_deg', 'ADDsym(cm)': 'ADDsym_cm'}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def binding(path):
    require(path.is_file() and not path.is_symlink(), str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest.hexdigest())


def inputs():
    return {name: binding(DOC / name) for name in NAMES} | {'checker': binding(Path(__file__).resolve())}


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def freeze():
    require(not (DOC / 'REPORT_VALIDATION_PROTOCOL.json').exists() and
            not (DOC / 'REPORT_VALIDATION_CHECKS.json').exists(), 'preserve previous report audit')
    write_new(DOC / 'REPORT_VALIDATION_PROTOCOL.json', dict(schema='v7_report_scalar_transcription_protocol',
        inputs=inputs(), accuracy_policy_changes=0, frozen_after_completed_accuracy_and_before_own_arithmetic=True,
        rounded_decimal_tolerance=5.1e-6, expected_counts=dict(operational_tables=30, moments=90, runtime=10),
        reference_limit='Checks saved report transcription and file links, not raw physical GT or hardware.'))
    print('REPORT_AUDIT_FROZEN')


def run():
    protocol = read(DOC / 'REPORT_VALIDATION_PROTOCOL.json')
    require(protocol['inputs'] == inputs(), 'own frozen report/checker/input drift')
    require(not (DOC / 'REPORT_VALIDATION_CHECKS.json').exists(), 'preserve previous report audit')
    metrics, runtime = read(DOC / 'METRICS.json'), read(DOC / 'RUNTIME.json')
    counts = dict(operational_tables=0, moments=0, runtime=0, scalar_checks=0, local_links=0)
    failures = []
    def near(text, value, label):
        counts['scalar_checks'] += 1
        require(value is not None and abs(float(text) - value) <= 5.1e-6, 'transcription ' + label)
    try:
        report = (DOC / 'RESULT_KO.md').read_text(encoding='utf-8')
        phase, stratum = None, 'combined'
        for line in report.splitlines():
            if line.startswith('### 쉬움153·중간92'):
                phase, stratum = 'operational', 'combined'
            elif line.startswith('### 전체·쉬움·중간'):
                phase, stratum = 'moments', 'combined'
            elif line.startswith('### NEW와 fallback'):
                phase = None
            elif line.startswith('## 실제 전체 경로 시간'):
                phase = 'runtime'
            elif line.startswith('### 실제 검토한'):
                phase = None
            if line == '**전체245**':
                stratum = 'combined'
            elif line == '**쉬움153**':
                stratum = 'easy'
            elif line == '**중간92**':
                stratum = 'medium'
            if not line.startswith('| '):
                continue
            parts = [part.strip() for part in line.strip().split('|')[1:-1]]
            if not parts or parts[0] not in ALIASES:
                continue
            method = ALIASES[parts[0]]
            # The first all-operational table precedes the named sub-strata.
            if phase is None and len(parts) == 8 and parts[1].isdigit() and counts['operational_tables'] < 10:
                active_phase = 'operational'
            else:
                active_phase = phase
            if active_phase == 'operational' and len(parts) == 8:
                saved = metrics['strata'][stratum]['methods'][method]
                require([int(v) for v in parts[1:5]] == [saved[k] for k in
                    ('operational', 'new_pose', 'fallback', 'no_pose')], 'operational status transcription')
                for text, key in zip(parts[5:], ('translation_cm', 'rotation_deg', 'ADDsym_cm')):
                    near(text, saved['metrics']['operational'][key]['mean'], stratum + '.' + method + '.' + key)
                counts['operational_tables'] += 1
            elif active_phase == 'moments' and len(parts) == 9 and parts[1] in METRICS:
                saved = metrics['strata'][stratum]['methods'][method]['metrics']['operational'][METRICS[parts[1]]]
                require(int(parts[2]) == saved['n'], 'moment n transcription')
                for text, key in zip(parts[3:], ('mean', 'sample_variance', 'sample_std', 'median', 'P90', 'max')):
                    near(text, saved[key], stratum + '.' + method + '.' + key)
                counts['moments'] += 1
            elif active_phase == 'runtime' and len(parts) == 8:
                saved = runtime['summaries'][method]['full']
                require(int(parts[1]) == saved['n'], 'runtime n transcription')
                for text, key in zip(parts[2:], ('mean', 'sample_variance', 'sample_std', 'median', 'P90', 'maximum')):
                    near(text, saved[key], 'runtime.' + method + '.' + key)
                counts['runtime'] += 1
        require({key: counts[key] for key in protocol['expected_counts']} == protocol['expected_counts'],
                'report table counts differ')
        for name in ('RESULT_KO.md', 'REPRODUCE.md', 'README.md'):
            text = (DOC / name).read_text(encoding='utf-8')
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if target.startswith(('http://', 'https://', '#')):
                    continue
                target = target.split('#')[0].strip('<>')
                require((DOC / target).resolve().is_file(), 'broken local link: ' + target)
                counts['local_links'] += 1
        ledger = read(DOC / 'BUILD_LEDGER.json')
        require(ledger['goal_complete'] is False, 'negative accuracy goal misreported')
        require(protocol['inputs'] == inputs(), 'report bytes changed during own audit')
    except Exception as error:
        failures.append(dict(type=type(error).__name__, message=str(error)))
    result = dict(passed=not failures, complete=not failures, inputs=protocol['inputs'], counts=counts,
        failures=failures, actual_saved_report_audit_runs=1, new_model_GT_PnP_training_RGB_calls=0,
        rounded_decimal_tolerance=5.1e-6, limitations=protocol['reference_limit'])
    write_new(DOC / 'REPORT_VALIDATION_CHECKS.json', result)
    print(json.dumps(result['counts']), 'PASS' if result['passed'] else 'FAIL')
    require(result['passed'], 'report audit failed; preserve first result')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    args = parser.parse_args()
    freeze() if args.stage == 'freeze' else run()
