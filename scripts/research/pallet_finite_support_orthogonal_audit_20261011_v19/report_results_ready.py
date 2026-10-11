"""Exclusive Korean report preserving V19 runtime cleanup failure.

ROOT runs freeze/run only after scientific, correspondence, saved-runtime-row
and visual receipts exist. The original runtime remains failed and unofficial.
No scientific row is decoded; no metric, moment, CI,
pose, image, model, training, projection, Jacobian or SVD is recomputed here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

PRIVATE = Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19')
FAMILY = 'pallet_finite_support_orthogonal_audit_20261011_v19'
DOC_LABEL = Path('_docs/experiments') / FAMILY
CODE_LABEL = Path('scripts/research') / FAMILY
NEW = ('HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS', 'HYBRID_OBSERVED_FRAME_POINT_ONLY')
METHODS = ('BASE', 'N3_SUBPIX', 'HYBRID_STANDARD_PNP', 'NATIVE_STANDARD_PNP', 'HYBRID_POINT_CONSENSUS_RESTORED', *NEW)
FIELDS = ('translation_cm', 'rotation_deg', 'ADDsym_cm')
MOMENTS = ('n', 'mean', 'population_variance', 'population_std', 'sample_variance', 'sample_std', 'median', 'P90', 'max', 'unit')
CONTRASTS = ((NEW[0], NEW[1]), (NEW[0], 'N3_SUBPIX'), (NEW[0], METHODS[3]), (NEW[0], METHODS[2]), (NEW[0], METHODS[4]), (NEW[1], METHODS[4]))
SCOPES = ('common_operational', 'candidate_new_pose', 'both_new_pose')
TIMES = ('full', 'detector', 'N3_correction', 'initial_pose', 'boundary_observation',
         'endpoint_validation_and_filtered_local', 'final_pose_reprojection_and_metadata', 'return')
FILES = ('REPORT_RESULTS_PROTOCOL.json', 'REPORT_RESULTS_STARTED.json', 'REPORT_RESULTS_CHECKS.json', 'RESULT_KO.md.ready')
LIMITS = dict(saved_JSON_transcription_only=True, new_metric_moment_CI_raw_decode_model_GT_fit_RGB_J_SVD_calls=0,
    physical_truth_or_ownership_certified=False, old_full_V16_564_FAIL_remains_FAIL=True,
    publication_push_not_claimed=True, passed_checks_not_accuracy_goal=True,
    original_runtime_cleanup_failed_preserved=True, original_runtime_statistics_official=False,
    runtime_timing_is_unofficial_saved_clock_posthoc=True, clean_deployment_runtime_certified=False,
    runtime_remeasurement_calls=0)


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_bytes())


def binding(path):
    path = Path(path).absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'missing/symlink ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''): digest.update(part)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def same(a, b):
    return isinstance(a, dict) and isinstance(b, dict) and all(a.get(k) == b.get(k) for k in ('bytes', 'sha256'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def paths(a):
    roots = dict(g=a.geometry_input, v=a.validation_input, e=a.evaluation_input, s=a.statistics_input,
                 d=a.correspondence_input, t=a.runtime_input, c=a.runtime_checks_input, f=a.render_input)
    names = dict(g=('GEOMETRY_PROTOCOL.json', 'GEOMETRY_STARTED.json', 'GEOMETRY_RECEIPT.json'),
        v=('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json'),
        e=('EVALUATION_PROTOCOL.json', 'SCORING_AUTHORITY.json', 'SCORING_STARTED.json', 'SCORING_RECEIPT.json', 'STATISTICS_STARTED.json', 'METRICS.json', 'POINT_REPRODUCTION.json'),
        s=('STATISTICS_CHECK_PROTOCOL.json', 'STATISTICS_CHECK_STARTED.json', 'STATISTICS_CHECKS.json'),
        d=('CORRESPONDENCE_PROTOCOL.json', 'CORRESPONDENCE_STARTED.json', 'CORRESPONDENCE_CHECKS.json', 'CORRESPONDENCE_SUMMARY.json'),
        t=('RUNTIME_PROTOCOL.json', 'RUNTIME_STARTED.json', 'RUNTIME_RECEIPT.json', 'RUNTIME_PREFLIGHT.json', 'RUNTIME_RESOURCES.json'),
        c=('RUNTIME_CHECK_PROTOCOL.json', 'RUNTIME_CHECK_STARTED.json', 'RUNTIME_CHECKS.json'),
        f=('FIGURE_PROTOCOL.json', 'FIGURES_STARTED.json', 'FIGURE_BINDINGS.json', 'FIGURE_DIAGNOSTIC_COUNTS.json', 'ROOT_VISUAL_REVIEW.json'))
    f = {prefix + ':' + n: root / n for prefix, root in roots.items() for n in names[prefix]}
    for prefix, n in (('g', 'GEOMETRY.jsonl.gz'), ('g', 'LEDGERS.jsonl.gz'), ('e', 'PREDICTIONS.jsonl.gz'),
                      ('e', 'POSTHOC_ROWS.jsonl.gz'), ('d', 'CORRESPONDENCE_ROWS.jsonl.gz'), ('t', 'INTERRUPTED_RUNTIME_ROWS.jsonl.gz'), ('f', 'FIGURE_CASE_ROWS.jsonl.gz')):
        f[prefix + ':' + n] = roots[prefix] / n
    for prefix, name in (('v', 'VALIDATION_CHECKS.json'), ('s', 'STATISTICS_CHECKS.json'), ('d', 'CORRESPONDENCE_CHECKS.json'), ('c', 'RUNTIME_CHECKS.json')):
        f[prefix + ':checker_source'] = Path(read(f[prefix + ':' + name])['checker']['path'])
    for prefix, alias, key in (('e', 'evaluator_source', 'adapter'), ('e', 'posthoc_source', 'unchanged_statistics')):
        f[alias] = Path(read(f['e:METRICS.json'])['inputs'][key]['path'])
    f.update(checker=Path(__file__), preserved_original_reporter=PRIVATE / 'report_results.py',
        original_mathematical_runtime_checker=Path(read(f['c:RUNTIME_CHECK_PROTOCOL.json'])['inputs']['original_mathematical_checker']['path']),
        first_geometry_receipt=PRIVATE / 'geometry/GEOMETRY_RECEIPT.json',
        serialization_ready_proof=PRIVATE / 'GEOMETRY_READY_PROOF.json',
        publication_checkpoint=a.repo / '_docs/experiments/pallet_observation_refiner_status_20261011/PUSH_V16_V18_RECEIPT.json')
    for name in ('PROTOCOL.json', 'STARTED.json', 'CHECKS.json', 'SAVED_LINE_DIAGNOSTIC.json'):
        f['toy:' + name] = PRIVATE / 'evidence' / name
    f['toy:PNG'] = PRIVATE / 'evidence/figures/03_finite_support_failure.png'
    for name in ('CPU_PROTOCOL.json', 'CPU_STARTED.json', 'CPU_CHECKS.json'):
        f['cpu:' + name] = Path(read(f['v:VALIDATION_PROTOCOL.json'])['inputs']['cpu:' + name]['path'])
    fig = read(f['f:FIGURE_BINDINGS.json'])
    f['renderer_source'] = Path(fig['renderer']['path'])
    for i, value in enumerate(fig['figures']): f['PNG:' + str(i)] = Path(value['path'])
    return f


def proof(f, b, prefix, stem, schema, error_key):
    check_name = dict(VALIDATION='VALIDATION_CHECKS', CORRESPONDENCE='CORRESPONDENCE_CHECKS').get(stem, stem + 'S')
    p, s, c = [read(f[prefix + ':' + name + '.json']) for name in (stem + '_PROTOCOL', stem + '_STARTED', check_name)]
    rows_only = prefix == 'c' and schema == 'independent_observed_frame_runtime_rows_checks_v19'
    require(p['schema'] == schema.replace('_checks_', '_protocol_') and c['schema'] == schema and c['complete'] is True
        and c['passed'] is (False if rows_only else True)
        and c[error_key] is None and p['inputs'] == s['inputs'] == c['inputs'], prefix + ' completed same-input evidence')
    if rows_only:
        require(c['scientific_rows_passed'] is True and c['operational_harness_cleanup_passed'] is False
            and c['expected_historical_cleanup_failure_count'] == c['overall_contract_failure_count'] == 1
            and c['failure_count'] == 0, 'saved runtime rows PASS; original cleanup FAIL retained')
        review = c['runtime_cleanup_review']; source_review = c['checker_source_review']
        require(review == p['runtime_cleanup_review'] == s['runtime_cleanup_review']
            and source_review == p['checker_source_review'] == s['checker_source_review']
            and review['original_runtime_complete'] is False and review['original_runtime_statistics_official'] is False
            and review['original_runtime_preserved'] is True and review['original_runtime_checker_modified'] is False
            and review['original_runtime_cleanup_passed'] is False and review['automatic_remeasurement'] is False
            and review['final_input_bytes_proof_executed_in_original_cleanup'] is False
            and review['final_input_bytes_independently_checked_by_this_rows_audit'] is True
            and review['current_process_environment_restoration_certified'] is False
            and review['source_namespace_restoration_in_original_process_certified'] is False
            and review['summaries_are_posthoc_saved_clocks_not_official_runtime_statistics'] is True
            and review['new_measurement_model_RGB_GT_fit_projection_J_SVD_calls'] == 0
            and source_review['inverse_whole_AST_equal'] is True
            and source_review['clock_describe_Audit_candidate_counter_parity_formulas_changed'] is False
            and source_review['original_runtime_or_checker_or_receipt_or_rows_changed'] is False
            and source_review['new_measurements_or_scientific_fits'] == 0
            and same(source_review['original_checker'], b['original_mathematical_runtime_checker'])
            and same(p['inputs']['original_mathematical_checker'], b['original_mathematical_runtime_checker'])
            and same(source_review['ready_checker'], b[prefix + ':checker_source']), 'explicit original cleanup failure and unchanged row-audit kernels')
    require(c['new_production_calls'] == s['new_production_calls'] == 0, prefix + ' saved-check scope')
    for item in (c['own_protocol'], s['own_protocol']): require(same(item, b[prefix + ':' + stem + '_PROTOCOL.json']), prefix + ' own protocol')
    for item in (c['checker'], p['checker'] if 'checker' in p else c['checker']): require(same(item, b[prefix + ':checker_source']), prefix + ' current checker source')
    return p, s, c


def gate(f, b):
    require(b['preserved_original_reporter']['sha256'] == 'f9383fb5eff1849a9d3acbe78e2bacaeeb02b11a7d41381beb389efd2e844a6b',
        'preserve original clean-runtime report source without rewriting its gate')
    core, receipt, geometry_start = [read(f['g:' + name]) for name in ('GEOMETRY_PROTOCOL.json', 'GEOMETRY_RECEIPT.json', 'GEOMETRY_STARTED.json')]
    require(core['schema'] == 'observed_frame_repair_geometry_protocol_v19' and core['methods'] == list(NEW)
        and core['frames'] == core['ledger_rows'] == 245 and core['rows'] == 490 and core['limits']['GT_allowed'] is False, 'paired geometry protocol')
    require(receipt['schema'] == 'observed_frame_repair_geometry_receipt_v19' and receipt['complete'] is True
        and receipt['error'] is None and receipt['cleanup_errors'] == [] and receipt['inputs_unchanged'] is True
        and receipt['environment_restored'] is True and receipt['actual_execution']['frames_complete'] == receipt['actual_execution']['ledger_rows'] == 245
        and receipt['actual_execution']['geometry_rows'] == 490 and receipt['elapsed_is_not_pipeline_latency'] is True
        and core['inputs'] == geometry_start['inputs'] == receipt['inputs'], 'complete restored245/490 geometry')
    require(same(receipt['protocol'], b['g:GEOMETRY_PROTOCOL.json']) and same(receipt['geometry'], b['g:GEOMETRY.jsonl.gz'])
        and same(receipt['ledgers'], b['g:LEDGERS.jsonl.gz']), 'raw geometry byte joins')
    vp, vs, vc = proof(f, b, 'v', 'VALIDATION', 'observed_frame_hybrid_scalar_checks_v19', 'exception')
    require(vc['failure_count'] == 0 and vc['actual_counts']['geometry_rows'] == 490
        and vc['actual_counts']['ledger_rows'] == vc['actual_counts']['frames'] == 245
        and vc['old_full_V16_validation_passed'] is False and vc['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True, 'new geometry PASS with old564 FAIL retained')
    for alias, name in (('geometry', 'GEOMETRY.jsonl.gz'), ('ledgers', 'LEDGERS.jsonl.gz'), ('geometry_protocol', 'GEOMETRY_PROTOCOL.json'),
                        ('geometry_started', 'GEOMETRY_STARTED.json'), ('geometry_receipt', 'GEOMETRY_RECEIPT.json')):
        require(same(vp['inputs'][alias], b['g:' + name]), 'validated geometry ' + alias)
    cpu = read(f['cpu:CPU_CHECKS.json'])
    require(cpu['schema'] == 'observed_frame_actual_CPU_checks_v19' and cpu['complete'] is cpu['passed'] is True
        and cpu['tests_run'] == 6 and cpu['failures'] == cpu['errors'] == [] and cpu['exception'] is None
        and cpu['source_unchanged'] is True and same(cpu['protocol'], b['cpu:CPU_PROTOCOL.json']), 'actual synthetic CPU6 contract evidence')
    for name in ('CPU_PROTOCOL.json', 'CPU_STARTED.json', 'CPU_CHECKS.json'):
        require(same(vp['inputs']['cpu:' + name], b['cpu:' + name]), 'actual CPU input bound to full scalar validation')
    m, ep, score, authority, score_start, stats_start = [read(f['e:' + name]) for name in ('METRICS.json', 'EVALUATION_PROTOCOL.json', 'SCORING_RECEIPT.json', 'SCORING_AUTHORITY.json', 'SCORING_STARTED.json', 'STATISTICS_STARTED.json')]
    require(m['schema'] == 'observed_frame_raw_row_metrics_v19' and m['complete'] is True and m['methods'] == list(METHODS)
        and m['contrasts'] == [list(c) for c in CONTRASTS] and m['all245_preserved_no_performance_filter'] is True
        and m['V16_full_scalar_validation_passed'] is False and m['population_variance_ddof'] == m['population_std_ddof'] == 0
        and m['unchanged_sample_variance_ddof'] == 1, 'seven checked methods and explicit moment definitions')
    require(score['complete'] is True and score['frames'] == 245 and score['scored_rows'] == 490 and score['methods'] == list(NEW)
        and score['GT_access_only_after_complete_geometry_and_cleanup'] is True, 'actual new490 postseal scoring')
    require(ep['schema'] == 'observed_frame_postseal_evaluation_protocol_v19' and authority['schema'] == 'isolated_observed_frame_proxy_score_authority_v19'
        and ep['inputs'] == authority['inputs'] and m['inputs'] == stats_start['inputs']
        and same(authority['own_evaluation_protocol'], b['e:EVALUATION_PROTOCOL.json'])
        and same(score_start['independent_validation'], b['v:VALIDATION_PROTOCOL.json'])
        and authority['reference_or_error_formula_changes'] == authority['numerical_geometry_calls'] == 0, 'actual isolated scoring and statistics closure')
    for actual, item in (('e:PREDICTIONS.jsonl.gz', score['predictions']), ('e:POSTHOC_ROWS.jsonl.gz', m['posthoc_rows']),
                        ('evaluator_source', m['inputs']['adapter']), ('posthoc_source', m['inputs']['unchanged_statistics'])):
        require(same(b[actual], item), 'saved evaluation ' + actual)
    require(ep['new_methods'] == list(NEW) and ep['reference'] == m['reference'], 'same known DEV proxy reference')
    sp, ss, sc = proof(f, b, 's', 'STATISTICS_CHECK', 'independent_observed_frame_statistics_checks_v19', 'error')
    for k, n in dict(saved_error_rows_streamed=1960, saved_error_rows_selected=1715, frames=245, methods=7,
                    moment_distributions_checked=675, moment_scalars_checked=5400, paired_groups_checked=54, metric_CI_slots_checked=162).items():
        require(sc['actual_counts'][k] == n, 'full statistics coverage ' + k)
    for alias, actual in (('METRICS.json', 'e:METRICS.json'), ('new_scores', 'e:PREDICTIONS.jsonl.gz'),
                         ('STATISTICS_STARTED.json', 'e:STATISTICS_STARTED.json'), ('EVALUATION_PROTOCOL.json', 'e:EVALUATION_PROTOCOL.json')):
        require(same(sp['inputs'][alias], b[actual]), 'independently checked statistics input ' + alias)
    dp, ds, dc = proof(f, b, 'd', 'CORRESPONDENCE', 'saved_observed_frame_correspondence_checks_v19', 'error')
    require(dc['actual_counts']['rows'] == 490 and dc['actual_counts']['frames'] == 245
        and dc['old_full_V16_validation_passed'] is False and dc['diagnostic_PASS_is_membership_count_join_not_independent_pose_quality_or_goal_PASS'] is True, 'completed saved490 correspondence joins')
    require(same(dc['summary'], b['d:CORRESPONDENCE_SUMMARY.json']) and same(dc['rows'], b['d:CORRESPONDENCE_ROWS.jsonl.gz']), 'actual diagnostic outputs')
    for alias, actual in (('evaluation:METRICS.json', 'e:METRICS.json'), ('evaluation:POSTHOC_ROWS.jsonl.gz', 'e:POSTHOC_ROWS.jsonl.gz'),
                         ('geometry:GEOMETRY.jsonl.gz', 'g:GEOMETRY.jsonl.gz'), ('statistics:STATISTICS_CHECKS.json', 's:STATISTICS_CHECKS.json')):
        require(same(dp['inputs'][alias], b[actual]), 'diagnosed current inputs ' + alias)
    d = read(f['d:CORRESPONDENCE_SUMMARY.json']); require(d['complete'] is True and d['methods'] == list(NEW), 'actual diagnostic summary')
    cp, cs, cc = proof(f, b, 'c', 'RUNTIME_CHECK', 'independent_observed_frame_runtime_rows_checks_v19', 'exception')
    require(cc['failure_count'] == 0 and cc['rows'] == 150 and cc['measured'] == 130 and cc['summary_blocks'] == 8
        and cc['summary_slots'] == 56 and cc['old_full_V16_validation_passed'] is False
        and cc['summaries_unofficial_due_to_historical_cleanup_failure'] is True
        and cc['originalruntime_statistics_official'] is False and set(cc['summaries']) == set(TIMES),
        'independent150/130/56 saved-row audit with unofficial time and overall cleanup FAIL')
    t, tp, journal = [read(f['t:' + n]) for n in ('RUNTIME_RECEIPT.json', 'RUNTIME_PROTOCOL.json', 'RUNTIME_STARTED.json')]
    require(t['schema'] == 'fresh_observed_frame_primary_runtime_receipt_v19' and t['complete'] is False
        and t['status'] == 'FAILED_RUNTIME_CLEANUP' and t['statistics_official'] is False and t['summaries'] is None
        and t.get('error') is None and t['configured_calls'] == t['serialized_rows'] == 150
        and t['execution'] == dict(image_decode_calls=26, model_initializations=1, calls_started=150, calls_complete=150)
        and t['cleanup'] == dict(instruments_entered=True, instrument_restoration_errors=[], instruments_restored=True,
            opencv_threads_restored=True, head_hooks_removed=True, pipeline_closed=True, environment_restored=False)
        and t['cleanup_errors'] == [dict(phase='restoration', type='RuntimeError', message='V19_RUNTIME: environment restoration')]
        and t['all_recorded_resource_snapshots_quiet'] is True and t['protected_before'] == t['protected_after'],
        'original full150 saved; exact environment cleanup failure remains FAILED and unofficial')
    require(tp['schema'] == 'fresh_observed_frame_primary_runtime_protocol_v19' and journal['inputs'] == tp['inputs']
        and journal['status'] == 'FAILED_RUNTIME_CLEANUP' and journal['execution'] == t['execution']
        and same(journal['protocol'], b['t:RUNTIME_PROTOCOL.json']) and same(journal['receipt'], b['t:RUNTIME_RECEIPT.json']), 'durable failed actual runtime context')
    for alias, actual in (('runtime_protocol', 't:RUNTIME_PROTOCOL.json'), ('receipt', 't:RUNTIME_RECEIPT.json'),
                         ('journal', 't:RUNTIME_STARTED.json'), ('rows', 't:INTERRUPTED_RUNTIME_ROWS.jsonl.gz'),
                         ('resources', 't:RUNTIME_RESOURCES.json'), ('preflight', 't:RUNTIME_PREFLIGHT.json')):
        require(same(cp['inputs'][alias], b[actual]), 'independently checked actual runtime ' + alias)
    fig, fp, visual = [read(f['f:' + n]) for n in ('FIGURE_BINDINGS.json', 'FIGURE_PROTOCOL.json', 'ROOT_VISUAL_REVIEW.json')]
    require(fig['schema'] == 'observed_frame_results_figure_bindings_v19' and fig['complete'] is True
        and fig['inputs'] == fp['inputs'] and same(fig['own_protocol'], b['f:FIGURE_PROTOCOL.json'])
        and same(fig['renderer'], b['renderer_source']) and same(fig['case_rows'], b['f:FIGURE_CASE_ROWS.jsonl.gz'])
        and same(fig['diagnostic_counts'], b['f:FIGURE_DIAGNOSTIC_COUNTS.json']), 'actual unchanged completed figures')
    require(fig['actual_counts']['PNG_files'] == 9 and fig['actual_counts']['case_panels'] == 18
        and fp['planned_PNGs'] == 9 and fp['planned_case_panels'] == 18 and len(fp['case_ids']) == 6, 'nine PNG/six fixed cases/eighteen panels')
    require(visual['complete'] is visual['passed'] is True and visual['actual_images_viewed'] == 9 and visual['case_panels'] == 18, 'ROOT actual nine-image review')
    viewed = {v['path']: v for v in visual['figures']}
    require(len(fig['figures']) == len(visual['figures']) == len(viewed) == 9, 'ROOT viewed nine unique actual PNG paths')
    for i, item in enumerate(fig['figures']):
        require(same(item, b['PNG:' + str(i)]) and item['path'] in viewed and same(item, viewed[item['path']]), 'actual rendered and viewed PNG bytes')
    for alias, actual in (('METRICS.json', 'e:METRICS.json'), ('new_scores', 'e:PREDICTIONS.jsonl.gz'), ('posthoc', 'e:POSTHOC_ROWS.jsonl.gz'),
                         ('statistics_checks', 's:STATISTICS_CHECKS.json')):
        require(same(fp['inputs'][alias], b[actual]), 'figures use actual checked saved source ' + alias)
    for name, n in (('combined', 245), ('easy', 153), ('medium', 92)):
        require(m['strata'][name]['frames'] == d['strata'][name]['frames'] == n, 'all retained ' + name)
        for method in METHODS: require(m['strata'][name]['methods'][method]['denominator'] == n, 'complete method population')
    reproduction = read(f['e:POINT_REPRODUCTION.json'])
    require(reproduction['complete'] is True and reproduction['passed'] is True and reproduction['frames'] == 245
        and same(m['point_reproduction'], b['e:POINT_REPRODUCTION.json']), 'same-q point route reproduction')
    first, ready = read(f['first_geometry_receipt']), read(f['serialization_ready_proof'])
    require(first['complete'] is False and first['error']['type'] == 'TypeError' and ready['failed_frame_solve_entries'] == 2
        and ready['failed_optimizer_entries'] == 46 and ready['failed_geometry_rows'] == 0 and ready['changed_calls'] == 2
        and ready['inverse_whole_AST_equal'] is True and ready['solver_point_line_observations_numeric_parameters_changed'] is False
        and same(ready['original_first_failed_receipt'], b['first_geometry_receipt']), 'original2solve46optimizer0row failure and serialization-only repair')
    toy, toy_protocol, toy_started, saved_line = [read(f['toy:' + name]) for name in ('CHECKS.json', 'PROTOCOL.json', 'STARTED.json', 'SAVED_LINE_DIAGNOSTIC.json')]
    require(toy['schema'] == 'finite_support_orthogonal_fixture_checks_v19' and toy['complete'] is toy['passed'] is True
        and toy['inputs'] == toy_protocol['inputs'] == toy_started['inputs'] and len(toy['fixture_cases']) == 4
        and toy['derivative_scalar_checks'] == 30 and toy['saved_contexts'] == 21 and toy['saved_line_blocks'] == 29
        and toy['old_finite_support_false_inlier_reproduced'] is toy['repaired_false_inlier_rejected'] is True
        and toy['old_V16_FAIL_status_unchanged'] is True and toy['real_performance_experiment_executed'] is False
        and same(toy['protocol'], b['toy:PROTOCOL.json']) and same(toy_started['protocol'], b['toy:PROTOCOL.json'])
        and same(toy['saved_diagnostic'], b['toy:SAVED_LINE_DIAGNOSTIC.json']), 'prior actual four-fixture/saved21 structural diagnosis')
    require(saved_line['schema'] == 'fixed_saved21_finite_support_comparison_v19' and saved_line['contexts'] == 21
        and saved_line['line_blocks'] == len(saved_line['rows']) == 29 and saved_line['new_fits_projections_GT_or_pose_performance_evaluation'] == 0,
        'literal saved29 line blocks with no new R,t evaluation')
    return m, d, t, receipt, vc, sc, dc, cc, fp, fig, cpu, toy, saved_line


def public_label(path, a):
    path = Path(path).resolve()
    if path.is_relative_to(PRIVATE):
        rel = path.relative_to(PRIVATE)
        label = CODE_LABEL / rel if len(rel.parts) == 1 and path.suffix == '.py' else DOC_LABEL / rel
    else:
        require(path.is_relative_to(a.repo.resolve()), 'public link must be declared repository/private evidence')
        label = path.relative_to(a.repo.resolve())
    return Path(os.path.relpath(label, DOC_LABEL)).as_posix()


def cell(value):
    if value is None: return 'None'
    if isinstance(value, (list, dict)): return json.dumps(value, ensure_ascii=False, separators=(',', ':')).replace('|', '\\|')
    return str(value).replace('|', '\\|').replace('\n', ' ')


def table(headers, values):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |',
                      *('| ' + ' | '.join(cell(v) for v in row) + ' |' for row in values)])


def report(a, f, evidence):
    m, d, t, g, vc, sc, dc, cc, fp, fig, cpu, toy, saved_line = evidence
    def link(key, label=None):
        path = Path(f[key]).resolve(); original = public_label(path, a)
        if path.is_relative_to(PRIVATE) and path.suffix == '.gz' and path.stat().st_size > 40 * 1024 * 1024:
            return '[' + (label or original) + ' (ordered parts)](ARCHIVE_MANIFEST.json)'
        return '[' + (label or path.name) + '](' + original + ')'
    joint = {}; results = []
    for comparator in ('N3_SUBPIX', METHODS[3], METHODS[2], NEW[1], METHODS[4]):
        pair = m['strata']['combined']['contrasts'][NEW[0] + '_minus_' + comparator]['common_operational']
        joint[comparator] = pair['common_frames'] == 245 and all(pair['metrics'][k]['mean_delta'] < 0 for k in FIELDS[:2])
        results.append([comparator, pair['common_frames'], joint[comparator], *[pair['metrics'][k]['mean_delta'] for k in FIELDS],
                        pair['metrics'][FIELDS[0]]['CI95'], pair['metrics'][FIELDS[1]]['CI95']])
    require(joint['N3_SUBPIX'] == m['primary_both_T_R_means_better_than_N3_full245']
        and joint[NEW[1]] == m['primary_both_T_R_means_better_than_point_full245'], 'saved first-answer verdict transcription')
    answer = ('현재 245장 평가 참조 기준의 운용 평균에서는 N3 및 native ordinary PnP 대조보다 위치·회전 오차가 모두 낮았다.'
              if joint['N3_SUBPIX'] and joint[METHODS[3]] else '현재 245장 평가 참조 기준의 운용 평균에서 N3와 native ordinary PnP 모두에 대한 위치·회전 동시 개선은 확인되지 않았다.')
    text = [
        '# V19 관측 좌표·유한 선분 보정의 실제 실행 결과',
        '> 가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?',
        '**' + answer + '** 아래 판단은 기존 주석·K·등록된 cuboid에서 재구성한 GEOMETRIC_PROXY에 대한 알려진 DEV 결과다. 독립 실측 자세 개선까지 검증한 결과는 아니다. 처음 보는 데이터의 성능이나 물리 corner ownership도 인증하지 않는다. 같은 q의 점 전용 경로와 비교한 값은 선 경로의 추가 효과를 구분하며, 가림 오판 여부는 새 자세 산출 성공의 필요조건으로 쓰지 않았다.',
        table(['비교(새 선 경로−대조)', '공통 산출', '평균 T·R 동시 개선', 'ΔT cm', 'ΔR degree', 'ΔADDsym cm', 'T 95% CI', 'R 95% CI'], results),
        'CI는 이전 13개 session의 고정 10,000 draw를 재사용한다. 음수는 새 방법에 유리하다. 각 구간과 실제 공통 집합을 함께 확인해야 하며, 평균 개선을 모든 영상·모든 난이도의 개선으로 해석하지 않는다.',
        '## 1. 실행 범위와 대응의 의미',
        '245장(Clean 153, Moderate 92)을 모두 유지했다. Severe 74는 사용자 변경 요청에 따라 이번 비교에서 제외했다. Base·N3→cornerSubPix·기존 ordinary 두 방법·V18 점 경로와 새 두 경로, 총 7방법 1,715개 평가 행을 비교한다. 입력 스트림은 1,960행이며, 이전 ORIGINAL 245행은 대조 선정에서만 제외했다. 새 자세 평가 490행은 기하·cleanup·독립 검산 완료 후 수행했다.',
        '각 query의 목표는 의미가 부여된 실제 물리 wire와 query 광선의 대응 또는 대응 없음이다. corner q는 한 query 위치를 직접 3D corner에 붙인 것이 아니라, 같은 corner에 incident한 두 서로 다른 경계선의 교점이다. 합성 CAL의 virtual cuboid corner는 실제 메쉬 꼭짓점일 필요가 없지만, 실사에서 선택된 두 선이 올바른 물리 경계를 뜻하는지는 별도로 확인되지 않았다.',
        '8px 이내 CORRECT_WITHIN8PX는 저장된 기하 참조와의 픽셀 오차 범주다. UNKNOWN/UNANNOTATED는 의미상 알려진 마스크 판정에서 제외한다. 알려진 가시성에는 SELF_OCCLUDED·외부 가림도 포함될 수 있어 DIRECT_VISIBLE와 같지 않다. 직접 가시성 표시와 manual_click 출처도 물리 edge/corner ID의 독립 승인은 아니다.',
        '동일 H·동일 q를 쓰는 후단 절제는 hybrid ordinary·V18 복원 점 경로·V19 선 경로·V19 점 경로의 네 hybrid 방법에 한정된다. native ordinary는 원 N3 q를 사용하며 fixed Base/N3의 H 진단 metadata는 unrecorded다. 이번 비교는 가림 분류기나 새 학습의 필요성 자체를 증명하지 않는다. 가림 없는 강건 PnP 대조는 기존 무학습 진단 자료에서 별도로 확인해야 한다. same-backend라고 부를 수 있는 ordinary hybrid/native 비교는 V17 내부 대조이며, V19의 부분집합·선 consensus와 ordinary backend가 같다는 뜻이 아니다.',
        '코너 선택은 초기 자기 가림 H와 검사 코너 k를 함께 제외하는 기존 H∪{k} LOO에서 native 좌표와 boundary q를 비교한다. 사용하지 않은 선의 endpoint 검사는 H∪{a,b}로 한다. 자기 가림 초기 XY는 승인 NEW의 fit·합의 점수에서 제외하고, 새 R,t가 구해지면 H 좌표를 최종 재투영으로 한 번 교체한다. 그 재투영을 독립 관측으로 재적합하지 않는다. 실패한 새 경로의 fallback은 기존 N3 전체 출력을 반환하며, 이는 H를 제외해 새로 구한 자세가 아니다.',
        '## 2. 이번에 실제로 고친 부분과 보존한 실패',
        '확정한 부분선 경로의 허점은 모델 투영 선분의 tangent를 사용한 기존 longitudinal hinge가 매우 짧고 관측선과 직교한 선분에서도 실제 긴 support를 inside로 볼 수 있다는 점이었다. 새 objective는 고정된 관측선 tangent에서 endpoint interval과 support interval을 비교한다. 이는 새로운 유한 관측 surrogate이며, 모든 support의 실제 nearest-segment 거리를 직접 최소화하는 식이나 물리 ownership 인증은 아니다. 같은 q의 점 전용 경로에는 이 선 objective가 없다.',
        table(['고정 2D fixture', '기존 block norm px', '새 block norm px', '실제 support→유한선분 거리 RMS px', '기존 inlier', '새 inlier'],
            [[v['name'], v['old_edge_RMS_px'], v['repaired_edge_RMS_px'], v['actual_support_RMS_distance_px'], v['old_inlier'], v['repaired_inlier']] for v in toy['fixture_cases']]),
        '짧은 직교 fixture의 기존 block norm은 약 0.01px였지만 새 block norm은 약 81.65px였다. 같은 저장 context의 model 투영 선분은 약 0.03609px, 실제 support 범위는 약 185.339px였고 기존 block norm 약 0.73590px가 새 식에서는 약 139.476px였다. 고정된 21 context·29 line block 중 23개의 기존 inlier가 새 식에서는 outlier였다. 5개의 accepted candidate context가 있었지만 approved NEW 출력은 0개였으므로, 이 허점이 V18 점 전용의 나쁜 자세 원인이라고 단정하지 않는다. 모든 원 값은 ' + link('toy:SAVED_LINE_DIAGNOSTIC.json') + '에 있다.',
        '![실제 저장 support와 짧은 직교 model segment의 과거 2D 진단](' + public_label(f['toy:PNG'], a) + ')\n\n이 그림은 별도 구조 진단이며 아래 실사 9PNG와 다른 증거다. block norm과 실제 closest-distance RMS를 같은 지표로 부르지 않는다.',
        '기존 V16 전체 scalar 198,134,007개 중 564 FAIL은 그대로 FAIL이다. line 경로의 비선택 후보 21 context에 대한 hinge/column-norm 실패를 새 결과의 PASS로 바꾸지 않았다. 새 V19의 모든 실제 proposal/refit·native endpoint와 새 식에 대한 독립 검산은 별도 권위다.',
        '첫 기하 실행은 첫 프레임의 실제 2 solve와 46 optimizer 후 ndarray JSON 직렬화에서 멈췄고 원행 0개를 기록했다. 원 실패 receipt와 source를 보존했다. 이후 수학·관측·설정을 바꾸지 않고 row/ledger JSON의 unchanged common.finite 두 호출만 별도 driver에 적용해 수동으로 실행했다. 성능에 따른 설정 재시도나 자동 retry가 아니다. 기하 replay elapsed는 추론 latency가 아니다.',
        '이전 별도 새 head의 training.py targets()에서 실제 대응이 있었던 2,164 query의 수치 실패를 no_match로 감독한 오류는 이미 실제 wire/depth 증거와 POSITIVE/NONE/IGNORE로 수정했다. main의 N3→cornerSubPix 오류가 아니었다. 최초 오류 인수인계 시점에는 재학습이 없었고, 이후 별도 승인으로 수정 감독의 세 조건×3,000=9,000 업데이트를 실제 완료했다. 수정 전 IMAGE_ROLE의 실사 0/319 결과를 수정 후 결과라고 쓰지 않는다. 이번 V19 추가 학습·새 RGB·seed·설정 sweep은 0이다.',
        '원 감독/수정/학습 이력: [수정 준비·검산](../pallet_kp_supervision_gate_20261010_v1/RESULT_KO.md), [실제 감독 수정](../pallet_kp_supervision_repair_20261010_v1/RESULT_KO.md), [수정 후 실제 학습](../pallet_kp_corrected_supervision_20261010_v1/TRAINING_COMPLETION.json), [훈련 코드](../../../scripts/research/pallet_observation_refiner_20261009_v1/training.py). 원 319장 실사 평가와 이후 245장 비교의 범위를 섞지 않는다.',
        '## 3. 모든 영상의 산출 상태와 오차 모멘트',
        'operational은 새 자세와 기본 출력 반환을 모두 포함한다. NEW/fallback/no_pose를 따로 세고, 큰 오차를 실패로 제외해 평균이 좋아 보이지 않도록 전체 운용 결과와 아래 공통 ID 비교를 함께 제공한다. None은 빈 집합 또는 정의되지 않는 표본 모멘트이며 0으로 바꾸지 않았다. 분산은 해당 단위의 제곱이다.',
    ]
    status_rows, marginal, paired, matched = [], [], [], []
    for stratum, block in m['strata'].items():
        for method, value in block['methods'].items():
            status_rows.append([stratum, method, value['denominator'], value['operational'], value['new_pose'], value['fallback'], value['no_pose'], value['statuses']])
            for scope, values in value['metrics'].items():
                for field, moment in values.items(): marginal.append([stratum, method, scope, field, *[moment[k] for k in MOMENTS]])
        for contrast, values in block['contrasts'].items():
            for scope, value in values.items():
                for field, metric in value['metrics'].items():
                    paired.append([stratum, contrast, scope, value['scope'], value['denominator'], value['common_frames'],
                        value['candidate_new_pose_in_pairs'], value['comparator_new_pose_in_pairs'], field, metric['n'], metric['mean_delta'],
                        metric['CI95'], metric['bootstrap_nonempty_resamples'], value['bootstrap_empty_resamples']])
                    for population, moment in value['matched_recorded_error_and_delta_moments'][field].items():
                        matched.append([stratum, contrast, scope, field, population, *[moment[k] for k in MOMENTS]])
    require(len(status_rows) == 21 and len(marginal) == 189 and len(paired) == 162 and len(matched) == 486, 'all report status/moment/CI rows')
    text.extend([table(['층', '방법', '전체', '산출', 'NEW', 'fallback', '완전 실패', '원 상태 수'], status_rows),
        table(['층', '방법', '산출 집합', '지표', *MOMENTS], marginal),
        '## 4. 공통 ID·NEW 조건부 비교와 고정 session CI',
        'common_operational, candidate_new_pose, both_new_pose를 따로 표시한다. N3와의 both_new_pose 이름은 실제 candidate_NEW_and_fixed_available이며 N3를 NEW로 재분류하지 않는다. 전체 pair_ids는 ' + link('e:METRICS.json') + '에 들어 있다.',
        table(['층', '대조', '키', '실제 집합', '전체 분모', '공통 n', 'candidate NEW', 'comparator NEW', '지표', 'n', '평균 차', '95% CI', '비어 있지 않은 draw', '빈 draw'], paired),
        '<details><summary>모든 공통 ID의 양쪽 원 오차 및 차이 모멘트 486개 분포</summary>\n',
        table(['층', '대조', '공통 집합', '지표', '양쪽/차이', *MOMENTS], matched), '\n</details>',
        '## 5. 마스크 오류·남은 대응·실제 자세 결과',
        '아래 수는 새 모델 실행이 아니라 저장된 posthoc와 sealed geometry의 membership/count join이다. 마스크 오류를 이유로 프레임을 버리지 않는다. 잘못 제거한 알려진 비자기 가림 점, 남겨 둔 알려진 자기 가림 점, 8px 초과 좌표를 구분한다. 전체 proxy 8px, 알려진 가시성의 proxy 8px, DIRECT_VISIBLE proxy 8px는 서로 다른 수다. UNKNOWN 좌표/가시성을 NONE이나 올바른 관측으로 바꾸지 않았다.',
        'eligible / solver_used / accepted_fit / accepted_final_inliers 각 점 ID·quality·직접 가시 ID·선 ID·기존 cuboid 부호 배치·저장 rank/특이값·초기 H와 실제 재투영은 ' + link('d:CORRESPONDENCE_ROWS.jsonl.gz') + '에 있다. fallback의 후보 inlier와 승인 NEW의 실제 fit을 구별한다. 두 선에서 만든 q의 소비 edge를 같은 fit의 선으로 중복 세지 않는다.',
        '전체 count×mask×pose×status 및 실제 배치별 frame ID는 ' + link('d:CORRESPONDENCE_SUMMARY.json') + '에 있다. rank6와 충분한 8px 점 수만으로 올바른 물리 대응·좋은 conditioning·유일해를 인증하지 않는다.',
    ])
    mask_rows, named_rows, visible_rows, error_rows, pool_rows, layout_rows = [], [], [], [], [], []
    for stratum, block in m['strata'].items():
        for method in NEW:
            diag = block['diagnostics'][method]
            for key, value in diag['mask_and_pose_groups_vs_fixed_N3'].items(): mask_rows.append([stratum, method, key, value['n'], value['new_pose'], value['fallback']])
            for group in ('DIRECT_VISIBLE', 'SELF_OCCLUDED', 'ACTUALLY_REPROJECTED'):
                value = diag[group]
                for stage, moment in value['errors'].items():
                    visible_rows.append([stratum, method, group, stage, value['corners'], value.get('damage_5px_to_over10px'),
                        *[moment[k] for k in ('n', 'mean', 'sample_variance', 'sample_std', 'median', 'P90')]])
            saved = d['strata'][stratum]['methods'][method]
            for key, value in saved['named_groups'].items(): named_rows.append([stratum, method, key, value['n'], value['ids']])
            for value in saved['mask_error_count_pose_status_groups']:
                error_rows.append([stratum, method, *[value[k] for k in ('mask_state', 'saved_pose_outcome_vs_N3', 'output_status',
                    'wrong_excluded_known_nonself_count', 'wrong_retained_known_self_count', 'proxy_correct_wrongly_excluded_count',
                    'proxy_incorrect_retained_used_count', 'proxy_incorrect_retained_final_inlier_count',
                    'wrong_selected_head_q_final_inlier_count', 'n', 'ids')]])
            for value in saved['pool_quality_mask_pose_status_groups']:
                pool_rows.append([stratum, method, *[value[k] for k in ('mask_state', 'saved_pose_outcome_vs_N3', 'output_status', 'pool',
                    'proxy_correct_count', 'proxy_incorrect_count', 'proxy_unknown_count', 'known_visibility_proxy_correct_count',
                    'unknown_visibility_count', 'directly_visible_proxy_correct_count', 'n', 'ids')]])
            for value in saved['layout_mask_pose_status_groups']:
                layout_rows.append([stratum, method, *[value[k] for k in ('mask_state', 'saved_pose_outcome_vs_N3', 'output_status',
                    'actual_fit_corner_ids', 'actual_fit_edge_ids', 'stored_finite_observed_rank', 'stored_finite_modeled_rank',
                    'stored_incidence_observed_rank', 'stored_incidence_modeled_rank', 'n', 'ids')]])
    text.extend([table(['층', '방법', '마스크×T/R 결과(대조 N3)', 'n', 'NEW', 'fallback'], mask_rows),
        table(['층', '방법', '문제/반례 집합', 'n', '전체 frame ID'], named_rows),
        '<details><summary>잘못 제외한 점과 남긴 잘못된 좌표: 개수·동시 오류·산출 상태·전체 frame ID</summary>\n',
        table(['층', '방법', '마스크', 'T/R 결과', '산출 상태', '잘못 제외한 known nonself', '남긴 known self',
            '잘못 제외한 proxy 8px', '남긴 used 8px초과', '남긴 final inlier 8px초과', 'wrong head q final inlier', 'n', '전체 frame ID'], error_rows), '\n</details>',
        '<details><summary>남은 대응의 4 pool별 개수: 전체 proxy·가시성 known·DIRECT_VISIBLE를 분리</summary>\n',
        table(['층', '방법', '마스크', 'T/R 결과', '산출 상태', 'pool', 'proxy 8px', 'proxy 8px초과', 'proxy UNKNOWN',
            'known 가시성 proxy 8px', 'UNKNOWN 가시성', 'DIRECT_VISIBLE proxy 8px', 'n', '전체 frame ID'], pool_rows), '\n</details>',
        '<details><summary>실제 fit 점·선 ID와 저장 rank별 결과: 배치·산출·전체 frame ID</summary>\n',
        table(['층', '방법', '마스크', 'T/R 결과', '산출 상태', 'fit corner IDs', 'fit edge IDs', 'finite observed rank',
            'finite modeled rank', 'incidence observed rank', 'incidence modeled rank', 'n', '전체 frame ID'], layout_rows), '\n</details>',
        '## 6. 직접 가시 코너 손상과 숨은 코너 재투영 오차',
        '직접 가시점의 N3≤5px→output>10px 증가를 손상 flag로 기록한다. H 재투영 오차가 줄었다는 이유만으로 R,t 개선을 선언하지 않는다. 픽셀 오차는 저장된 proxy에 대한 값이며 물리 팔레트 자체의 손상이 아니다. ACTUALLY_REPROJECTED는 실제 승인 NEW에서 교체한 H만 포함한다. SELF_OCCLUDED 사람 상태와 예측 H는 같은 집합일 필요가 없다.',
        table(['층', '방법', '코너 집합', '좌표 단계', '코너 수', '직접 손상 flag 수', '유효 n', '평균 px', '표본분산 px²', '표본 SD px', '중앙값 px', 'P90 px'], visible_rows),
        '## 7. 실제 전체 경로의 저장 시간: 환경 복원 실패로 비공식',
        '**시간 측정 harness의 최종 상태는 FAILED_RUNTIME_CLEANUP이며 공식 runtime PASS가 아니다.** 실제 150회 호출은 반환·직렬화됐지만 종료 시 환경 복원 검사가 실패했다. 원 receipt의 complete=False·statistics_official=False·summaries=None, 실패 journal과 원행을 그대로 보존했다. 아래 수치는 같은 저장 tick 150행 중 measured 130행을 별도 row audit이 원 describe 식으로 계산한 비공식·탐색적 통계다. 재측정하지 않았으며 환경 복원이나 배포 경로의 정상 종료를 인증하지 않는다.',
        '아래 값은 새 RGB를 입력해 검출기→N3/cornerSubPix→초기 자세와 Base feature 자세→고정 IMAGE_ROLE head/CAL→코너 LOO/강건 점 PnP→native endpoint LOO→새 선 consensus/refit→최종 H 재투영·metadata까지 실제 실행한 경로다. 저장 cache 재생 시간이나 단계 시간의 단순 합산이 아니다. 파일 decode·모델 로딩·기록·반환 후 저장 결과 parity는 clock 밖이다. 같은 q 점 전용 비교는 이번 primary runtime으로 재측정했다고 주장하지 않는다.',
        '26개 고정 원영상·13session, warmup20 + 5×26=130 measured, 총150 호출이다. 첫 예측 내부의 검출기 준비 forward를 포함해 detector151/N3150이며 로딩 forward는 0이다. resource quiet는 기록된 snapshot에 한정한다. 실패 상태는 ' + link('t:RUNTIME_RECEIPT.json') + ', tick/primitive 원행은 ' + link('t:INTERRUPTED_RUNTIME_ROWS.jsonl.gz') + ', 별도 저장 행·비공식 시간 검산은 ' + link('c:RUNTIME_CHECKS.json') + '을 따른다. 이 검산의 complete=True·scientific_rows_passed=True는 저장 행에 대한 것이며, 전체 passed=False·operational_harness_cleanup_passed=False와 구분한다.',
        table(['시간 단계', 'n', '평균 ms', '표본분산 ms²', '표본 SD ms', '중앙값 ms', 'P90 ms', '최대 ms', '단위', 'ddof'],
            [[k, *[cc['summaries'][k][field] for field in ('n', 'mean', 'sample_variance', 'sample_std', 'median', 'P90', 'maximum', 'unit', 'ddof')]] for k in TIMES]),
        '환경 복원 실패는 중첩 context 종료 순서와 관련된 harness 문제다. instrumentation·OpenCV thread·head hook·pipeline 종료는 원 receipt에 복원됐다고 기록됐지만 environment_restored=False가 남았다. 종료 시 원 runtime의 마지막 input-byte 검사까지 실행됐다고 주장하지 않는다. 새 row audit은 자신의 현재 입력 SHA를 별도로 검사한다. 원 실패를 수정된 완료 receipt로 바꾸거나 해당 process의 namespace 복원이 검증됐다고 말하지 않는다.',
        '## 8. 실제 실행량·검산 범위',
        table(['권위', '확인 범위', '실제 수/상태'], [
            ['실제 합성 CPU', '6개 통합 contract; 실사 정확도 성공 인증 아님', dict(tests_run=cpu['tests_run'], failures=cpu['failures'], errors=cpu['errors'], primitive_calls=cpu['actual_CPU_numerical_entry_calls'])],
            ['완료 기하', 'paired490/245ledger/245frames', g['actual_execution']],
            ['기하 replay elapsed', '초기 영상 추론을 다시 수행한 latency가 아님', dict(seconds=g['elapsed_geometry_seconds'], elapsed_is_not_pipeline_latency=g['elapsed_is_not_pipeline_latency'])],
            ['완료 기하 primitive', '실제 CV/SciPy/SVD entry·return·exception/callback', g['actual_numerical_calls']],
            ['새 전체 기하 독립 검산', '전 proposal/refit·native 경로', dict(checks=vc['checks'], failures=vc['failure_count'], counts=vc['actual_counts'], max_difference=vc['max_scalar_absolute_difference'])],
            ['저장 오차 통계 검산', '675분포/5,400필드/54공통집합/162CI', dict(exact_checks=sc['exact_checks'], numeric_checks=sc['numeric_checks'], counts=sc['actual_counts'], max_difference=sc['max_absolute_difference'])],
            ['관측 membership/count', '2방법490·4pool', dc['actual_counts']],
            ['실제 fresh runtime', '150/130/26 저장됨; 최종 환경 cleanup FAIL', dict(execution=t['execution'], status=t['status'], complete=t['complete'], statistics_official=t['statistics_official'])],
            ['독립 저장 행·primitive·H·비공식 시간 검산', '8블록56summary slot; row audit와 전체 FAIL을 분리', dict(checks=cc['checks'], scalar_failures=cc['failure_count'], scientific_rows_passed=cc['scientific_rows_passed'], operational_harness_cleanup_passed=cc['operational_harness_cleanup_passed'], overall_passed=cc['passed'], expected_historical_cleanup_failures=cc['expected_historical_cleanup_failure_count'], overall_contract_failures=cc['overall_contract_failure_count'], max_difference=cc['max_absolute_difference'])],
        ]),
        table(['fresh 측정 counter 그룹', '기록 값'], [[k, t[k]] for k in ('model_forwards', 'model_forwards_after_loading', 'actual_base_counts',
            'actual_fresh_counts', 'local_operation_counts', 'native_operation_counts', 'direct_point_operation_counts', 'point_bank_capture_counts',
            'native_bank_reuse_counts', 'live_cv_counts', 'live_optimizer_counts', 'live_SVD_counts', 'live_head_counts', 'cleanup', 'cleanup_errors')]),
        '675/5,400/162 검산은 pose 오차 모멘트·공통 집합·CI에 대한 독립 저장 산술이다. mask/corner 진단의 픽셀 오차 모멘트는 기존 posthoc 생산자가 계산한 저장값이며 같은 통계 검산이 독립 재계산했다고 확대하지 않는다. correspondence PASS는 membership/count join이고 이미지 review PASS는 가독성 확인이다. runtime row audit의 CLI exit0도 저장 행의 산술·계수 검사 통과를 뜻하며 원 cleanup FAIL이나 비공식 시간 상태를 지우지 않는다. 모두 물리 GT·ownership·성능 목표 성공과 다르다.',
        '## 9. 고정 시각 사례와 전체 집계 그림',
        '원래 지정한 Clean3/Moderate3, 같은 6개 ID를 사용했다. 결과를 보고 좋은 사례를 새로 고르지 않았다. 9PNG/18case panels를 ROOT가 실제 열어 확인했다. N3 패널에서 가림·fit/inlier가 unrecorded인 것은 immutable fixed-row에 진단 metadata가 기록되지 않았다는 뜻이며 실제로 가림이 없거나 빈 fit을 수행했다는 인증이 아니다. cyan=출력, green+=입력 q, orange=승인 NEW의 H 재투영, white=저장 proxy, yellow=실제 선 support, magenta=virtual model segment, red×=직접 가시 proxy 오차 증가다.',
        '시각 검토: ' + link('f:ROOT_VISUAL_REVIEW.json') + ' / 실제 그림 binding: ' + link('f:FIGURE_BINDINGS.json') + '.',
    ])
    for i, value in enumerate(fig['figures']): text.append('![' + Path(value['path']).name + '](' + public_label(f['PNG:' + str(i)], a) + ')')
    text.extend(['## 10. 무엇이 확인됐고 무엇이 남았는가',
        '감독의 잘못된 NONE과 decision의 추가 cost veto는 과거에 각각 수정했다. 이번에는 부분선 finite-support 식의 확정 허점을 새 observed-frame objective로 고쳤다. 가림을 완벽히 맞히는 조건을 추가하지 않았으며, 충분한 다른 대응이 남으면 강건 합의로 계산하고 H를 재투영하는 계약을 유지했다. 분류·선택·강건 PnP·선식·재투영의 효과는 위 대조와 실제 원행으로 나눠 판단한다. 재투영만의 숨은 점 개선을 최종 자세 개선으로 바꾸지 않는다.',
        '실사에서 두 물리 선의 의미가 맞는지, 선택된 q가 해당 native virtual corner를 뜻하는지, 남은 잘못된 대응이 좋은 proxy inlier처럼 들어갔는지는 이 실험의 물리적 독립 감독으로 확인하지 못했다. 기존 주석의 source tag에는 unknown/manual_click/pnp_projected가 혼재하며 fixed N3는 번호 정렬 기준이지 R,t 정답 생성자가 아니다. GT R,t는 기존 8점 주석에서 이상적 cuboid를 적합한 GEOMETRIC_PROXY이므로 전점 독립 클릭·물리 치수/자세 실측 보장이 없다.',
        '기존 P0 합성 학습에서는 외부 가림·가리지 않는 방해물·저대비 등을 포함한 네 통제 변형을 별도로 검증하지 않았다. 이번 실행은 그 누락을 새 합성 생성이나 학습으로 채운 것이 아니다. 고정 CAL128의 348 corner에서 virtual 교점/native cuboid 평균 차이 3.0674443457516407e−05px·최대 0.0002457793608984587px가 작았다는 source 계약도 실사 물리 ownership 보증으로 확대하지 않는다.',
        '13개 과거 문제 query의 원 annotation source tag와 현재 관측 pool의 연결은 correspondence 원행에 기록했다. unknown은 좌표가 반드시 틀렸다는 증거도, 대응 없음이라는 라벨도 아니다. 새로운 실사 촬영·수동 annotation·훈련·설정 변경은 수행하지 않았다.',
        '공개 경로는 게시용 상대 경로이며 이 보고서 생성 자체가 push 완료를 뜻하지 않는다. 이전 게시 checkpoint는 ' + link('publication_checkpoint') + '이고 V19의 commit/원격 SHA 일치와 공개 replay/restore 권위는 별도 후속 receipt가 확인해야 한다. 원 main 자동 병합·force push를 실행했다고 주장하지 않는다.',
        '40MiB를 넘는 V19 gzip은 원 gzip을 byte 순서대로 나눈 ordered parts로 게시한다. 해당 원행 링크는 원 상대 경로를 label에 남기고 [ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json)의 parts·원 SHA로 연결한다. GitHub에 ignored 원 gzip이 직접 존재한다는 뜻이 아니다. 아래는 게시 clone의 root에서 실행할 복원 절차이며, 출력은 clone 안의 아직 사용하지 않은 디렉터리여야 한다. 복원·portable reader 실행 완료를 이 보고서 생성만으로 주장하지 않는다.',
        '```bash\nset -e\npallet_read_repo="$(pwd -P)"\npallet_archive_manifest="$pallet_read_repo/_docs/experiments/' + FAMILY + '/ARCHIVE_MANIFEST.json"\npallet_restore_output="$pallet_read_repo/_docs/reader_v19_restore"\npython -I -S "$pallet_read_repo/scripts/research/' + FAMILY + '/publication_ready.py" restore-freeze --repo "$pallet_read_repo" --manifest "$pallet_archive_manifest" --output "$pallet_restore_output"\npython -I -S "$pallet_read_repo/scripts/research/' + FAMILY + '/publication_ready.py" restore-run --repo "$pallet_read_repo" --manifest "$pallet_archive_manifest" --output "$pallet_restore_output"\n```',
        '원행/기하 ' + link('g:GEOMETRY.jsonl.gz') + ' · ' + link('g:LEDGERS.jsonl.gz') + ' · 새 score ' + link('e:PREDICTIONS.jsonl.gz') + ' · posthoc ' + link('e:POSTHOC_ROWS.jsonl.gz') + ' · 전체 원 JSON ' + link('e:METRICS.json') + '.',
        '실행 증거: ' + link('first_geometry_receipt') + ' · ' + link('serialization_ready_proof') + ' · ' + link('cpu:CPU_CHECKS.json') + ' · ' + link('v:VALIDATION_CHECKS.json') + ' · ' + link('s:STATISTICS_CHECKS.json') + ' · ' + link('d:CORRESPONDENCE_CHECKS.json') + ' · ' + link('c:RUNTIME_CHECKS.json') + '.',
        '코드: [고정 관측 선분 식](../../../scripts/research/' + FAMILY + '/geometry.py), [분리 solver](../../../scripts/research/' + FAMILY + '/solver.py), [관측 pipeline](../../../scripts/research/' + FAMILY + '/pipeline.py), ' + link('evaluator_source') + ', ' + link('d:checker_source') + ', ' + link('c:checker_source') + '.',
    ])
    return '\n\n'.join(text) + '\n', dict(status_rows=len(status_rows), marginal_distributions=len(marginal),
        matched_distributions=len(matched), core_moment_fields=5400, paired_CI_rows=len(paired), runtime_summary_slots=56,
        mask_pose_rows=len(mask_rows), named_group_rows=len(named_rows), mask_error_rows=len(error_rows),
        remaining_pool_rows=len(pool_rows), actual_layout_rows=len(layout_rows), visible_hidden_error_rows=len(visible_rows),
        PNG_images=9, prior_structural_diagnostic_PNG=1, fixed_case_panels=18, joint_saved_verdicts=joint,
        original_runtime_cleanup_failed=True, runtime_statistics_official=False,
        saved_runtime_rows_scientific_passed=cc['scientific_rows_passed'], original_runtime_contract_passed=cc['passed'])


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=('freeze', 'run'))
    for name in ('repo', 'geometry-input', 'validation-input', 'evaluation-input', 'statistics-input', 'correspondence-input',
                 'runtime-input', 'runtime-checks-input', 'render-input', 'output'): p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); out = a.output.absolute()
    require(out.is_relative_to(PRIVATE) and out != PRIVATE and not any(v.is_symlink() for v in (out, *out.parents)), 'new nonsymlink private report directory')
    for key, value in vars(a).items():
        if key not in ('stage', 'output'):
            root = value.resolve(); require(not out.is_relative_to(root) and not root.is_relative_to(out), 'report output overlaps protected input')
    out.mkdir(parents=True, exist_ok=True)
    for n in FILES if a.stage == 'freeze' else FILES[1:]: require(not (out / n).exists(), 'preserve ' + n)
    f = paths(a); b = {k: binding(v) for k, v in f.items()}; evidence = gate(f, b)
    if a.stage == 'freeze':
        write(out / FILES[0], dict(schema='saved_observed_frame_report_protocol_v19', inputs=b, checker=b['checker'],
            frozen_before_report_transcription=True, preserved_existing_draft='PRIVATE19/RESULT_KO.md',
            new_output=FILES[3], limits=LIMITS)); return
    own = read(out / FILES[0]); require(own['schema'] == 'saved_observed_frame_report_protocol_v19' and own['inputs'] == b
        and own['limits'] == LIMITS and own['new_output'] == FILES[3], 'frozen report code/evidence changed')
    protocol = binding(out / FILES[0]); write(out / FILES[1], dict(inputs=b, own_protocol=protocol, checker=b['checker'], new_production_calls=0))
    error = None; counts = {}
    try:
        text, counts = report(a, f, evidence)
        require(b == {k: binding(v) for k, v in f.items()}, 'report inputs changed during transcription')
        with (out / FILES[3]).open('x', encoding='utf-8') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
    except BaseException as exc: error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    write(out / FILES[2], dict(schema='saved_observed_frame_report_checks_v19', complete=error is None, passed=error is None,
        inputs=b, own_protocol=protocol, checker=b['checker'], error=error, transcription_counts=counts,
        report=binding(out / FILES[3]) if (out / FILES[3]).exists() else None, limits=LIMITS, new_production_calls=0,
        independent_transcription_scalar_recheck_not_claimed=True, publication_complete=False))
    if error: raise SystemExit(1)


if __name__ == '__main__': main()
