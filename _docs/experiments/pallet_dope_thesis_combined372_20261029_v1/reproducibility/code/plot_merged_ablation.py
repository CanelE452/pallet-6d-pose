"""Plot audited primary372 ablation tables, without recomputing statistics.

Source-only until ROOT freezes this source and supplies both exact-path input
and plotting GO manifests. No NN, PnP, bootstrap, images, annotations, model
weights, private prediction caches, or historical DEV292 tables are opened.

CLI: --describe-contract; --tag TAG [--validate-only]. Rendering is authorized
only by preflight/MERGED_ABLATION_PLOT_GO.json and never replaces old figures.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re

OUT = Path(__file__).resolve().parents[1]
MERGED = OUT.parent / 'dope-wood640-eval80-20261027-v1/combined372/evaluation'
INPUTS = OUT / 'preflight/MERGED_ABLATION_PLOT_INPUTS.json'
GO = OUT / 'preflight/MERGED_ABLATION_PLOT_GO.json'
KINDS = ('PLASTIC', 'WOOD', 'GREEN')
EXPECTED = {'PLASTIC': (128, 985), 'WOOD': (125, 963), 'GREEN': (119, 943)}
REMOVED = ('no_confidence', 'no_ransac', 'no_loo', 'no_flip',
           'no_final_support', 'confidence_only')
CONDITIONS = ('control', 'full', *REMOVED)
QUALITY = ('PCK5', 'PCK10', 'PCK20', 'coverage', 'conditional_precision10')
DISPLAY_METRICS = ('PCK10', 'coverage', 'conditional_precision10')
ORDER = (('R0', 0), *((c, r) for c in CONDITIONS for r in (1, 2)))
LABELS = {'R0': 'Initial R0', 'control': 'Synthetic control', 'full': 'Full LOO + flip',
          'no_confidence': 'Without confidence', 'no_ransac': 'Without RANSAC',
          'no_loo': 'Without LOO', 'no_flip': 'Without flip',
          'no_final_support': 'Without final >=6', 'confidence_only': 'Confidence only'}
METRIC_LABELS = {'PCK5': 'PCK@5 px', 'PCK10': 'PCK@10 px', 'PCK20': 'PCK@20 px',
                 'coverage': 'Reference coverage',
                 'conditional_precision10': 'Conditional precision@10 px'}
COLORS = {0: '#666666', 1: '#C05B40', 2: '#2577AE'}
ROLES = {'table_ablation_2d', 'ci_ablation', 'merged_complete',
         'merged_bootstrap_receipt', 'numerical_audit'}
RAW_KEYS = {'points', 'xy', 'reference_xy', 'predictions', 'all_annotated_reference',
            'projected_cuboid', 'manual_kps', 'canonical_xyz', 'target_coordinates'}
STYLE_REFERENCE = dict(
    path=str(OUT.parent / 'dope-filter-ablation-per-type-20261027-v1/code/plot_results.py'),
    sha256='6ae0b31349aa62ef15cc929ecdb701f10a69410ce5f1950681c091987cef01b3',
    use='Palette, DejaVu Sans, vector PDF fonts and paired-CI layout; no old data reads')
POPULATION_NOTE = ('Primary evaluation: PLASTIC 128 / 985 reference points; WOOD 125 / 963; '
                   'GREEN 119 / 943. Total: 372 images / 2,891 reference points.')
PIXEL_NOTE = ('Errors use Euclidean distance in native pixels. WOOD combines 1280 x 720 '
              'and 640 x 480 images; a fixed pixel threshold has different relative tolerance.')
CI_NOTE = ('Full minus removed condition; paired-frame bootstrap, 10,000 draws, 95% percentile CI. '
           'One training seed; no multiplicity correction. WOOD has only four known recording '
           'clusters: frame CIs are primary and do not measure training-seed or recording-level uncertainty.')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_ok(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def no_coordinates(value):
    if isinstance(value, dict):
        require(not (RAW_KEYS & set(value)), 'Raw coordinate/prediction input is prohibited')
        for item in value.values():
            no_coordinates(item)
    elif isinstance(value, list):
        for item in value:
            no_coordinates(item)


def number(value):
    if value in ('', None, 'NA', 'None'):
        return None
    value = float(value)
    require(math.isfinite(value), 'Nonfinite table value')
    return value


def integer(value):
    value = number(value)
    require(value is not None and value == int(value), 'Expected integer count')
    return int(value)


def same(a, b, message):
    require((a is None and b is None) or (a is not None and b is not None
            and math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-12)), message)


def contract():
    """ROOT writes these metadata files only after source review/freeze."""
    return dict(
        input_manifest_path=str(INPUTS), go_path=str(GO),
        inputs=dict(schema='primary372_ablation_plot_inputs_v1', primary_frames=372,
                    primary_reference_points=2891, plot_source_sha256='CURRENT_SOURCE_SHA256',
                    source_files={'/EXACT/ABSOLUTE/PATH': {'role': 'ONE_OF_REQUIRED_ROLES',
                                                           'sha256': 'CURRENT_FILE_SHA256'}}),
        required_roles=sorted(ROLES),
        input_rules={'four_merged_files': 'Same combined372/evaluation/TAG/final directory; '
                     'MODEL_SUMMARY.csv, PAIRED_CI.csv, COMPLETE.json, BOOTSTRAP_RECEIPT.json',
                     'numerical_audit': 'Exact DELIVERY/preflight/*.json, passed=true, complete=true, '
                     'source_sha256 binds both CSVs and merged COMPLETE.json'},
        go=dict(schema='primary372_ablation_plot_go_v1', passed=True, plotting_authorized=True,
                input_manifest_sha256='CURRENT_INPUT_MANIFEST_SHA256',
                plot_source_sha256='CURRENT_SOURCE_SHA256'),
        outputs='DELIVERY/figures/merged_ablation/TAG/{three PNG, three PDF, PLOT_QA.json, PROVENANCE.json}',
        visual_QA='Generation QA does not certify page/image appearance; ROOT must inspect all six files',
        statistics='No recomputed bootstrap, no invented percentage SD, all 51 evaluation groups and 180 CIs retained')


class Inputs:
    def __init__(self):
        self.bindings = {}
        self.source_sha = sha(__file__)
        self.spec = self.fixed_json(INPUTS)
        self.go = self.fixed_json(GO)
        require(self.spec['schema'] == 'primary372_ablation_plot_inputs_v1'
                and self.spec['primary_frames'] == 372
                and self.spec['primary_reference_points'] == 2891, 'Not a primary372 plot contract')
        require(self.go['schema'] == 'primary372_ablation_plot_go_v1'
                and self.go['passed'] is True and self.go['plotting_authorized'] is True
                and self.go['input_manifest_sha256'] == self.bindings[str(INPUTS)]
                and self.go['plot_source_sha256'] == self.spec['plot_source_sha256'] == self.source_sha,
                'ROOT plotting GO/source/input binding is missing or stale')
        self.files = self.spec['source_files']
        require(isinstance(self.files, dict) and len(self.files) == len(ROLES), 'Exactly five inputs required')
        require({v['role'] for v in self.files.values()} == ROLES, 'Exact source roles differ')
        self.paths = {}
        for name, item in self.files.items():
            p = Path(name)
            require(p.is_absolute() and str(p.resolve()) == name and not p.is_symlink()
                    and digest_ok(item['sha256']), 'Exact regular path and SHA required')
            require(p.is_file() and p.suffix in ('.csv', '.json'), 'Only derived CSV/JSON inputs allowed')
            self.paths[item['role']] = p
        summary = self.paths['table_ablation_2d']
        parent = summary.parent
        require(parent.parent.parent == MERGED and parent.name == 'final'
                and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', parent.parent.name),
                'Historical DEV292 or non-merged table is prohibited')
        for role, basename in (('table_ablation_2d', 'MODEL_SUMMARY.csv'),
                               ('ci_ablation', 'PAIRED_CI.csv'), ('merged_complete', 'COMPLETE.json'),
                               ('merged_bootstrap_receipt', 'BOOTSTRAP_RECEIPT.json')):
            require(self.paths[role] == parent / basename, 'Merged input directory/basename differs')
        audit = self.paths['numerical_audit']
        require(audit.parent == OUT / 'preflight' and audit.suffix == '.json',
                'ROOT numerical audit must be an exact DELIVERY preflight receipt')

    def fixed_json(self, path):
        p = Path(path)
        require(p.resolve() == p and not p.is_symlink() and p.is_file(), 'Missing/unsafe ROOT metadata')
        digest = sha(p)
        obj = json.loads(p.read_text())
        no_coordinates(obj)
        require(sha(p) == digest, 'ROOT metadata changed during read')
        self.bindings[str(p)] = digest
        return obj

    def bind(self, role):
        p = self.paths[role]
        digest = sha(p)
        require(digest == self.files[str(p)]['sha256'], 'ROOT input SHA differs: ' + role)
        self.bindings[str(p)] = digest
        return p

    def read(self, role):
        p = self.bind(role)
        result = json.loads(p.read_text())
        no_coordinates(result)
        require(sha(p) == self.bindings[str(p)], 'JSON changed during read')
        return result

    def csv(self, role):
        p = self.bind(role)
        with p.open(newline='') as stream:
            reader = csv.DictReader(stream)
            require(not (RAW_KEYS & set(reader.fieldnames or ())), 'Coordinate CSV blocked')
            rows = list(reader)
        require(sha(p) == self.bindings[str(p)], 'CSV changed during read')
        return rows

    def unchanged(self):
        require(sha(__file__) == self.source_sha, 'Plot source changed')
        for path, digest in self.bindings.items():
            require(sha(path) == digest, 'Bound source changed: ' + path)


def validate(io):
    complete = io.read('merged_complete')
    for key, value in dict(complete=True, passed=True, primary_population='ALL372', primary_frames=372,
                          primary_reference_points=2891, main_groups=51, WOOD_frames=125, WOOD_points=963,
                          primary_CI_rows=180, frame_rows=6324).items():
        require(complete.get(key) == value, 'Merged completion differs: ' + key)
    for role in ('table_ablation_2d', 'ci_ablation', 'merged_bootstrap_receipt'):
        p = io.paths[role]
        require(complete['output_sha256'].get(str(p)) == io.files[str(p)]['sha256'],
                'Merged COMPLETE does not bind current output: ' + role)
    audit = io.read('numerical_audit')
    require(audit.get('passed') is True and audit.get('complete') is True, 'Numerical audit is incomplete')
    for role in ('table_ablation_2d', 'ci_ablation', 'merged_complete'):
        p = io.paths[role]
        require(audit['source_sha256'].get(str(p)) == io.files[str(p)]['sha256'],
                'Numerical audit does not bind current plot input: ' + role)
    bootstrap = io.read('merged_bootstrap_receipt')
    for key, value in dict(seed=20261026, replicates=10000, unit='paired_frame', primary_rows=180,
                          no_undefined_draws_discarded=True, undefined_denominator_makes_entire_CI_NA=True,
                          WOOD_known_recordings=4, frame_CI_primary_fewer_than5_recordings=True,
                          multiplicity_correction='none').items():
        require(bootstrap.get(key) == value, 'Bootstrap receipt differs: ' + key)
    rows = io.csv('table_ablation_2d')
    expected = {(kind, condition, r) for kind in KINDS for condition, r in ORDER}
    lookup = {}
    for row in rows:
        key = (row['kind'], row['condition'], integer(row['round']))
        require(key in expected and key not in lookup, 'Unexpected/duplicate summary group')
        require(row['evaluation_population'] == 'PRIMARY372' and row['panel'] == 'all_annotated',
                'Primary summary scope differs')
        n, nref = EXPECTED[key[0]]
        require(integer(row['n_DEV']) == n and integer(row['n_reference_points']) == nref,
                'Same-kind population/reference denominator differs')
        require(row['available'] in ('True', 'False'), 'Availability must be explicit')
        available = row['available'] == 'True'
        if available:
            seen = integer(row['n_observed_reference_predictions'])
            require(0 <= seen <= nref and integer(row['conditional_pixel_error_n']) == seen,
                    'Observed-error denominator differs')
            for metric in QUALITY:
                value = number(row[metric])
                require(value is not None or (metric == 'conditional_precision10' and seen == 0),
                        'Available-model metric is missing')
                require(value is None or 0 <= value <= 1, 'Metric is outside [0,1]')
            require(number(row['PCK5']) <= number(row['PCK10']) <= number(row['PCK20']), 'PCK is not monotone')
            same(number(row['coverage']), seen / nref, 'Coverage denominator differs')
            correct10 = number(row['PCK10']) * nref
            require(abs(correct10 - round(correct10)) <= 1e-8 and correct10 <= seen + 1e-8,
                    'PCK10 cannot correspond to an observed point count')
            same(number(row['conditional_precision10']), correct10 / seen if seen else None,
                 'Conditional precision denominator differs')
            mean, std, variance = (number(row['conditional_pixel_error_' + k]) for k in ('mean', 'std', 'variance'))
            if seen:
                require(all(v is not None and v >= 0 for v in (mean, std, variance)), 'Observed error mean/SD missing')
                same(std * std, variance, 'Observed SD/variance are inconsistent')
            else:
                require(mean is std is variance is None, 'Zero-observation error statistics must remain NA')
        else:
            require(all(number(row[k]) is None for k in (*QUALITY, 'n_observed_reference_predictions',
                        *('conditional_pixel_error_' + k for k in ('n', 'mean', 'std', 'variance')))),
                    'Unavailable model must remain NA, not zero')
        lookup[key] = row
    require(len(rows) == len({r['group'] for r in rows}) == 51 and set(lookup) == expected,
            'All 51 evaluation groups are required')
    cis = io.csv('ci_ablation')
    ci_lookup = {}
    expected_ci = {(k, c, r, m) for k in KINDS for c in REMOVED for r in (1, 2) for m in QUALITY}
    for row in cis:
        key = (row['kind'], row['comparator'], integer(row['round']), row['metric'])
        require(key in expected_ci and key not in ci_lookup and row['candidate'] == 'full'
                and row['panel'] == 'all_annotated'
                and row['difference_direction'] == 'full_minus_removed_condition', 'Unexpected/duplicate comparison')
        require(integer(row['replicates']) == 10000 and integer(row['seed']) == 20261026
                and integer(row['n_frame_units']) == EXPECTED[key[0]][0]
                and row['multiplicity_correction'] == 'none' and digest_ok(row['weights_sha256']),
                'Prespecified paired-CI settings differ')
        if key[0] == 'WOOD':
            require(row['resampling_unit'] == 'paired_frame' and integer(row['n_resampled_units']) == 125,
                    'Recording CI cannot silently replace primary frame CI')
        a, b = lookup[key[0], 'full', key[2]], lookup[key[:3]]
        av, bv = number(a[key[3]]), number(b[key[3]])
        point = number(row['point_difference_full_minus_ablation'])
        same(point, av - bv if av is not None and bv is not None else None,
             'CI point estimate differs from displayed summary')
        low, high, undefined = number(row['ci95_low']), number(row['ci95_high']), number(row['undefined_replicates'])
        require((low is None) == (high is None), 'Partial CI bounds')
        if low is not None:
            require(-1 <= low <= high <= 1 and point is not None and undefined == 0, 'Invalid CI interval')
            decision = 'full_higher' if low > 0 else 'full_lower' if high < 0 else 'includes_zero'
            require(row['decision'] == decision, 'CI decision differs from bounds')
        elif a['available'] == b['available'] == 'True':
            require(undefined is not None and undefined == int(undefined) and 0 < undefined <= 10000
                    and row['decision'] == 'NA_undefined_denominator_no_draws_dropped',
                    'Undefined CI must remain explicitly NA')
        else:
            require(point is None and undefined is None and row['decision'] == 'NA_model_unavailable',
                    'Unavailable-model comparison must remain NA')
        ci_lookup[key] = row
    require(len(cis) == 180 and set(ci_lookup) == expected_ci, 'All 180 prespecified comparisons are required')
    io.unchanged()
    return lookup, ci_lookup


def row_label(condition, round_idx):
    return LABELS[condition] + ('' if condition == 'R0' else '  R' + str(round_idx))


def setup_axis(axis):
    axis.grid(axis='x', alpha=0.2)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)


def footnotes(fig, lines):
    for index, text in enumerate(lines):
        fig.text(0.5, 0.012 + 0.018 * (len(lines) - index - 1), text, ha='center', fontsize=8)


def save_pair(fig, target, stem, plt):
    result = {}
    for extension, options in (('png', {'dpi': 220}), ('pdf', {})):
        path = target / (stem + '.' + extension)
        require(not path.exists(), 'Existing figure is immutable')
        fig.savefig(path, facecolor='white', **options)
        require(path.is_file() and path.stat().st_size > 0, 'Empty figure output')
        result[path.name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    plt.close(fig)
    return result


def summary_figure(rows, target, plt):
    fig, axes = plt.subplots(3, 3, figsize=(15.2, 18.0), sharex=True, sharey=True)
    labels = [row_label(*item) for item in ORDER]
    for ki, kind in enumerate(KINDS):
        for mi, metric in enumerate(DISPLAY_METRICS):
            axis = axes[ki, mi]
            for y, (condition, round_idx) in enumerate(ORDER):
                row = rows[kind, condition, round_idx]
                value = number(row[metric])
                if value is None:
                    axis.text(1, y, 'N/A', va='center', fontsize=7, color=COLORS[round_idx])
                else:
                    marker = '^' if condition == 'R0' else 's' if condition == 'control' else 'D' if condition == 'full' else 'o'
                    axis.plot(value * 100, y, marker, color=COLORS[round_idx], markersize=4.5)
            n, ref = EXPECTED[kind]
            axis.set_title(f'{kind}: {n} images / {ref} refs\n{METRIC_LABELS[metric]}')
            axis.set_yticks(range(len(ORDER)), labels)
            axis.set_xlim(-2, 102)
            axis.set_xlabel('Percent')
            axis.tick_params(axis='y', labelsize=7.5, labelleft=(mi == 0))
            setup_axis(axis)
    axes[0, 0].set_ylim(len(ORDER) - 0.5, -0.5)
    fig.suptitle('DOPE component ablations: primary 372 images (WOOD 125)', fontsize=15)
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.09, top=0.94, hspace=0.3, wspace=0.13)
    footnotes(fig, [POPULATION_NOTE,
        'PCK@10 = correct / all reference points (missing = failure); coverage = observed / all; precision = correct / observed.',
        'All 51 evaluation groups retained. Absolute percentages have no error bars: no percentage SD was supplied. R1 rust; R2 blue; R0 gray.',
        PIXEL_NOTE])
    return save_pair(fig, target, 'PRIMARY372_MODEL_METRICS', plt)


def difference_figure(cis, target, plt):
    fig, axes = plt.subplots(5, 3, figsize=(15.4, 19.5), sharey=True)
    for mi, metric in enumerate(QUALITY):
        extent = [0.0]
        for key, row in cis.items():
            if key[3] == metric:
                extent += [value * 100 for field in ('point_difference_full_minus_ablation', 'ci95_low', 'ci95_high')
                           if (value := number(row[field])) is not None]
        span = max(max(extent) - min(extent), 1)
        limits = (min(extent) - 0.06 * span - 0.5, max(extent) + 0.06 * span + 0.5)
        for ki, kind in enumerate(KINDS):
            axis = axes[mi, ki]
            axis.axvline(0, color='#777777', linewidth=1, zorder=0)
            for index, condition in enumerate(REMOVED):
                for r, offset in ((1, -0.13), (2, 0.13)):
                    row = cis[kind, condition, r, metric]
                    point = number(row['point_difference_full_minus_ablation'])
                    low, high = number(row['ci95_low']), number(row['ci95_high'])
                    y = index + offset
                    if point is None:
                        axis.text(0, y, f'R{r} N/A', color=COLORS[r], fontsize=6.5, va='center')
                        continue
                    axis.plot(point * 100, y, 'o', color=COLORS[r], markersize=3.8)
                    if low is not None:
                        axis.hlines(y, low * 100, high * 100, color=COLORS[r], linewidth=1.2)
                        axis.vlines([low * 100, high * 100], y - 0.04, y + 0.04, color=COLORS[r], linewidth=0.8)
                    else:
                        axis.annotate('CI N/A', (point * 100, y), xytext=(4, 0), textcoords='offset points',
                                      color=COLORS[r], fontsize=6, va='center')
            axis.set_title(f'{kind} ({EXPECTED[kind][0]} images) / {METRIC_LABELS[metric]}', fontsize=9)
            axis.set_xlim(*limits)
            axis.set_ylim(5.5, -0.5)
            axis.set_yticks(range(6), [LABELS[c] for c in REMOVED])
            axis.tick_params(axis='y', labelsize=7, labelleft=(ki == 0))
            axis.set_xlabel('Full minus removed condition (percentage points)', fontsize=8)
            setup_axis(axis)
    fig.suptitle('All 180 prespecified paired comparisons: primary 372 images (WOOD 125)', fontsize=15)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.08, top=0.95, hspace=0.42, wspace=0.12)
    footnotes(fig, [POPULATION_NOTE,
        'R1 rust; R2 blue. Positive values favor full. N/A models and undefined intervals remain N/A; no comparisons are selected by outcome.',
        'Paired-frame bootstrap: 10,000 draws; 95% percentile CI. One training seed; no multiplicity correction.',
        'WOOD has only four known recording clusters; frame CIs are primary, not seed or recording-level uncertainty. Zero-containing CI is inconclusive.',
        PIXEL_NOTE])
    return save_pair(fig, target, 'PRIMARY372_ALL_PAIRED_DIFFERENCES', plt)


def observed_error_figure(rows, target, plt):
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 10.5))
    for axis, kind in zip(axes, KINDS):
        extent = [0.0]
        labels = []
        for y, (condition, r) in enumerate(ORDER):
            row = rows[kind, condition, r]
            mean = number(row['conditional_pixel_error_mean'])
            std = number(row['conditional_pixel_error_std'])
            count = number(row['conditional_pixel_error_n'])
            labels.append(row_label(condition, r) + ('  n=NA' if count is None else f'  n={int(count)}'))
            if mean is None:
                axis.text(0, y, 'N/A', color=COLORS[r], fontsize=7, va='center')
                continue
            extent.extend((mean - std, mean + std))
            axis.errorbar(mean, y, xerr=std, fmt='o', color=COLORS[r], markersize=4, capsize=2, linewidth=1)
        span = max(max(extent) - min(extent), 1)
        axis.set_xlim(min(extent) - 0.04 * span, max(extent) + 0.08 * span)
        axis.set_ylim(len(ORDER) - 0.5, -0.5)
        axis.set_yticks(range(len(ORDER)), labels)
        axis.tick_params(axis='y', labelsize=7)
        axis.set_title(f'{kind}: {EXPECTED[kind][0]} images', fontsize=11)
        axis.set_xlabel('Observed point error: mean +/- SD (native px)')
        setup_axis(axis)
    fig.suptitle('Conditional observed-point errors: all 51 evaluation groups, primary 372 images (WOOD 125)', fontsize=14)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.13, top=0.92, wspace=0.8)
    footnotes(fig, [POPULATION_NOTE,
        'Mean +/- population SD (ddof=0) of observed reference-point errors; n is the number of observed points, conditional on detection.',
        'These bars describe point-error dispersion, not bootstrap uncertainty or training-seed SD. Negative mean-SD endpoints are descriptive.',
        'All models retained, with missing models or zero observations shown as N/A. Horizontal scales may differ by kind.', PIXEL_NOTE])
    return save_pair(fig, target, 'PRIMARY372_OBSERVED_ERROR_MEAN_SD', plt)


def write_json(path, obj):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag')
    parser.add_argument('--validate-only', action='store_true', help='Check GO, all input hashes and all table rows; write nothing')
    parser.add_argument('--describe-contract', action='store_true', help='Print metadata contract without any input reads')
    args = parser.parse_args()
    if args.describe_contract:
        print(json.dumps(contract(), indent=2), flush=True)
        return
    require(args.tag is not None and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.tag), 'A safe output tag is required')
    io = Inputs()
    rows, cis = validate(io)
    if args.validate_only:
        print(json.dumps(dict(passed=True,summary_rows=51,paired_CI_rows=180,primary_frames=372,
                              primary_reference_points=2891,source_sha256=io.bindings,
                              plot_source_sha256=io.source_sha,files_written=0)), flush=True)
        return
    target = OUT / 'figures/merged_ablation' / args.tag
    require(target.resolve() == target and not target.exists(), 'Preserve prior figures; unsafe or existing output tag')
    target.mkdir(parents=True, exist_ok=False)
    try:
        # All Matplotlib-generated cache files are confined to the new figure directory.
        os.environ['MPLCONFIGDIR'] = str(target / '_matplotlib_config')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                             'pdf.fonttype': 42, 'ps.fonttype': 42})
        artifacts = {}
        artifacts.update(summary_figure(rows, target, plt))
        artifacts.update(difference_figure(cis, target, plt))
        artifacts.update(observed_error_figure(rows, target, plt))
        io.unchanged()
        require(len(artifacts) == 6 and sum(n.endswith('.png') for n in artifacts) == 3
                and sum(n.endswith('.pdf') for n in artifacts) == 3, 'Expected three PNG/PDF pairs')
        qa = dict(passed=True,figure_generation_passed=True,visual_review_status='PENDING_ROOT_IMAGE_INSPECTION',
                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),primary_frames=372,
                  primary_reference_points=2891,frames_and_references_per_kind=EXPECTED,summary_groups=51,
                  plotted_absolute_metric_values=153,plotted_CI_rows=180,observed_error_groups=51,
                  missing_models_and_undefined_CIs_preserved=True,percentage_SD_invented=False,
                  observed_point_error_SD_ddof=0,bootstrap_draws_recomputed=0,NN_forwards=0,PnP_fits=0,
                  raw_RGB_reads=0,raw_annotation_reads=0,checkpoint_reads=0,
                  source_sha256=io.bindings,plot_source_sha256=io.source_sha,artifacts=artifacts)
        write_json(target / 'PLOT_QA.json', qa)
        write_json(target / 'PROVENANCE.json', dict(schema='primary372_ablation_figures_v1',
                   source_sha256=io.bindings,plot_source_sha256=io.source_sha,style_reference=STYLE_REFERENCE,
                   selection='All 51 prespecified evaluation groups and all 180 paired comparisons; no outcome-based selection',
                   metric_definitions={'PCK10':'correct native error <=10px / all reference points; missing is failure',
                       'coverage':'observed reference predictions / all reference points',
                       'conditional_precision10':'correct <=10px / observed reference predictions',
                       'observed_error_mean_SD':'Observed-point native error, population SD ddof=0; conditional on detection'},
                   population_note=POPULATION_NOTE,pixel_note=PIXEL_NOTE,CI_note=CI_NOTE,
                   visual_QA_required=True,artifacts=artifacts,PLOT_QA_sha256=sha(target / 'PLOT_QA.json')))
        io.unchanged()
    except Exception as exc:
        failure = target / 'FAILURE.json'
        if not failure.exists():
            write_json(failure, dict(passed=False,reason=str(exc),plot_source_sha256=io.source_sha,
                                    partial_outputs_preserved=True,no_automatic_retry=True))
        raise
    print('PRIMARY372_ABLATION_PLOTS_GENERATED', target, 'VISUAL_QA_PENDING', flush=True)


if __name__ == '__main__':
    main()
