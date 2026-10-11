"""Independent saved-error statistics for the one frozen whole-U closure.

ROOT alone executes freeze/run. The immutable V19 stdlib scalar, moment and
bootstrap bodies are isolated with six explicit topology-only AST regions.
No production module, GT, model, image, pose fit, projection or SVD is executed.
The historical full V16 564-failure audit remains failed.
"""
import argparse
import ast
from collections import Counter
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

PRIVATE = Path('/tmp/pallet-whole-u-closure-private-20261011-v21')
HISTORY = Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19')
ORIGINAL = HISTORY / 'stats_check.py'
ORIGINAL_SHA = '4cad27af50af1711ac1817e3c24cfddf73386b535f1cf8c30950950ef2ab1b0c'
EVALUATOR_SHA = '3938edcc3491e9c19eae99adc8bd1af04595a8c9e42be02dd968a6c5fbae52d1'
HISTORY_EVALUATOR_SHA = 'bad7bdf3271098b5728165fa8868fbb60cbcb13e04271115883094da1814dd2a'
SCALAR_SHA = 'da0a91d232914a15768f233a698ebe33e5c1604e2506fee143f4dffb6732afb7'
NEW = ('HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE',)
OLD = 'HYBRID_POINT_CONSENSUS_RESTORED'
HISTORY_NEW = ('HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS', 'HYBRID_OBSERVED_FRAME_POINT_ONLY')
METHODS = ('BASE', 'N3_SUBPIX', 'HYBRID_STANDARD_PNP', 'NATIVE_STANDARD_PNP', OLD, *HISTORY_NEW, *NEW)
CONTRASTS = tuple((NEW[0], method) for method in METHODS[:-1])
METRICS = ('translation_cm', 'rotation_deg', 'ADDsym_cm')
UNITS = dict(translation_cm='cm', rotation_deg='degree', ADDsym_cm='cm')
SCOPES = ('common_operational', 'candidate_new_pose', 'both_new_pose')
FILES = ('STATISTICS_CHECK_PROTOCOL.json', 'STATISTICS_CHECK_STARTED.json', 'STATISTICS_CHECKS.json')
ABS_TOL = 1e-9
REL_TOL = 1e-12
LIMITS = dict(saved_proxy_arithmetic_only=True, physical_truth_or_unseen_accuracy_certified=False,
    geometry_recomputed=False, old_full_V16_564_FAIL_remains_FAIL=True,
    new_GT_model_fit_projection_SVD_seed_calls=0, historical_ORIGINAL245_discarded=True,
    current_evaluation_protocol_inputs_strict=True, historical_metadata_not_recursively_opened=True)

def require(value, message):
    if not value: raise ValueError(message)

def read(path): return json.loads(Path(path).read_bytes())

def binding(path):
    path = Path(path).absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'missing/symlink: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())

def same(a, b): return all(a.get(k) == b.get(k) for k in ('bytes', 'sha256'))

def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())

def paths(args):
    folder = args.input.absolute(); m = read(folder / 'METRICS.json')
    aliases = ('cohort', 'bootstrap', 'fixed_scores', 'ordinary_scores', 'point_scores', 'history_scores',
        'new_scores', 'unchanged_statistics', 'unchanged_score', 'adapter', 'checker')
    f = {key: Path(m['inputs'][key]['path']) for key in aliases if key != 'checker'}
    f['geometry_checker'] = Path(m['inputs']['checker']['path'])
    f.update({n: folder / n for n in ('METRICS.json', 'STATISTICS_STARTED.json', 'EVALUATION_PROTOCOL.json',
        'SCORING_RECEIPT.json', 'SCORING_AUTHORITY.json', 'SCORING_STARTED.json')})
    for n in ('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json',
              'GEOMETRY_PROTOCOL.json', 'GEOMETRY_STARTED.json', 'GEOMETRY_RECEIPT.json'):
        f[n] = Path(m['inputs'][n]['path'])
    f.update(checker=Path(__file__).resolve(), original_statistics=ORIGINAL,
        independent_scalar=args.repo / 'scripts/research/pallet_sparse_local_line_20261010_v9/verify.py')
    for key, value in m['inputs'].items(): f['evaluation_input:' + key] = Path(value['path'])
    return f

def metadata(f, b):
    m, e, s, score, authority, start = [read(f[n]) for n in ('METRICS.json', 'EVALUATION_PROTOCOL.json',
        'STATISTICS_STARTED.json', 'SCORING_RECEIPT.json', 'SCORING_AUTHORITY.json', 'SCORING_STARTED.json')]
    require(m['schema'] == 'whole_U_closure_raw_row_metrics_v21' and m['complete'] is True
        and m['V16_full_scalar_validation_passed'] is False, 'completed V21 statistics; full old FAIL preserved')
    require(e['schema'] == 'whole_U_closure_postseal_evaluation_protocol_v21'
        and e['methods'] == m['methods'] == s['methods'] == list(METHODS)
        and e['contrasts'] == m['contrasts'] == s['contrasts'] == [list(pair) for pair in CONTRASTS]
        and e['new_methods'] == list(NEW), 'fixed eight methods/seven contrasts/one new method')
    require(e['old_full564_scalar_FAIL_remains_FAIL'] is True
        and e['frozen_before_new_GT_access_and_statistical_arithmetic'] is True
        and s['new_seed_draws'] == s['new_geometry_or_GT_score_calls'] == 0
        and e['arithmetic'] == s['arithmetic'] == m['actual_arithmetic'], 'frozen arithmetic and no new draws/fit')
    arithmetic = dict(saved_scored_rows=1960, source_score_rows_streamed=2205, historical_original_rows_ignored=245,
        newly_scored_rows=245, posthoc_rows=245, methods=8, strata=3, moment_distributions=216,
        inherited_moment_scalars=1296, additional_population_moment_scalars=432, paired_groups=63, metric_CIs=189,
        supplemental_paired_matched_distributions=567, supplemental_paired_inherited_moment_scalars=3402,
        supplemental_paired_population_moment_scalars=1134, total_distributions=783, total_moment_fields=6264,
        new_bootstrap_draws=0)
    require(e['arithmetic'] == arithmetic, 'exact predeclared783/6264/63/189 arithmetic')
    extra = dict(new_scores=b['new_scores'], scoring_receipt=b['SCORING_RECEIPT.json'],
        scoring_authority=b['SCORING_AUTHORITY.json'], scoring_started=b['SCORING_STARTED.json'],
        evaluation_protocol=b['EVALUATION_PROTOCOL.json'])
    require(m['inputs'] == s['inputs'] and set(m['inputs']) == set(e['inputs']) | set(extra)
        and all(m['inputs'][k] == v for k, v in e['inputs'].items())
        and all(same(m['inputs'][k], v) for k, v in extra.items()), 'exact complete evaluation/statistics closure')
    for key, expected in m['inputs'].items():
        require(same(b['evaluation_input:' + key], expected), 'current exact declared input: ' + key)
    require(b['adapter']['sha256'] == EVALUATOR_SHA and b['original_statistics']['sha256'] == ORIGINAL_SHA,
        'immutable V21 evaluator and V19 checker sources')
    require(score['schema'] == 'repair_postseal_proxy_scoring_v13' and score['complete'] is True
        and score['frames'] == score['scored_rows'] == start['configured_scored_rows'] == 245
        and score['methods'] == list(NEW) and same(b['new_scores'], score['predictions'])
        and score['GT_access_only_after_complete_geometry_and_cleanup'] is True
        and start['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True, 'actual own245 postseal score')
    require(authority['schema'] == 'isolated_whole_U_closure_proxy_score_authority_v21'
        and authority['inputs'] == e['inputs'] and authority['score_dispatch'] == e['score_dispatch']
        and authority['reference_or_error_formula_changes'] == authority['numerical_geometry_calls'] == 0
        and same(b['EVALUATION_PROTOCOL.json'], authority['own_evaluation_protocol']), 'unchanged isolated score formula')
    v, p, vs = [read(f[n]) for n in ('VALIDATION_CHECKS.json', 'VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json')]
    require(p['schema'] == 'whole_U_closure_point_scalar_protocol_v21'
        and v['schema'] == 'whole_U_closure_point_scalar_checks_v21' and v['complete'] is True
        and v['passed'] is True and v['failure_count'] == 0 and v['exception'] is None
        and v['new_production_calls'] == vs['new_production_calls'] == 0
        and v['old_full_V16_validation_passed'] is False and v['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True
        and v['inputs'] == p['inputs'] == vs['inputs'], 'actual independent point geometry PASS; old full FAIL separate')
    for item in (v['own_protocol'], vs['own_protocol'], start['independent_validation']):
        require(same(b['VALIDATION_PROTOCOL.json'], item), 'actual validation protocol')
    for item in (v['checker'], p['checker']): require(same(b['geometry_checker'], item), 'current full245 checker')
    require(all(v['actual_counts'][key] == 245 for key in ('geometry_rows', 'ledger_rows', 'frames')), 'own full245 geometry')
    core, gs, g = [read(f[n]) for n in ('GEOMETRY_PROTOCOL.json', 'GEOMETRY_STARTED.json', 'GEOMETRY_RECEIPT.json')]
    require(core['schema'] == 'whole_U_closure_geometry_protocol_v21' and core['methods'] == list(NEW)
        and core['frames'] == core['rows'] == core['ledger_rows'] == 245 and core['limits']['GT_allowed'] is False
        and g['schema'] == 'whole_U_closure_geometry_receipt_v21' and g['complete'] is True
        and g['error'] is None and not g['cleanup_errors'] and g['inputs_unchanged'] is True
        and g['environment_restored'] is True and g['counter_dispatch_closed'] is True
        and all(value is True for value in g['restoration'].values())
        and g['inputs'] == core['inputs'] == gs['inputs']
        and g['protection_before'] == g['protection_after'] == core['protection'], 'complete restored own geometry scope')
    for alias, name in (('geometry_protocol', 'GEOMETRY_PROTOCOL.json'), ('geometry_started', 'GEOMETRY_STARTED.json'),
                        ('geometry_receipt', 'GEOMETRY_RECEIPT.json')):
        require(same(b[name], p['inputs'][alias]), 'independently checked geometry metadata ' + alias)
    for item in (g['protocol'], gs['protocol'], score['protocol']): require(same(b['GEOMETRY_PROTOCOL.json'], item), 'own geometry protocol bytes')
    require(same(b['GEOMETRY_RECEIPT.json'], score['geometry_seal']), 'score actual completed geometry receipt')
    history = {n: read(f['evaluation_input:history_validation:' + n]) for n in
        ('STATISTICS_CHECK_PROTOCOL.json', 'STATISTICS_CHECK_STARTED.json', 'STATISTICS_CHECKS.json')}
    hc, hp, hs = [history[n] for n in ('STATISTICS_CHECKS.json', 'STATISTICS_CHECK_PROTOCOL.json', 'STATISTICS_CHECK_STARTED.json')]
    require(hp['schema'] == 'independent_observed_frame_statistics_protocol_v19'
        and hc['schema'] == 'independent_observed_frame_statistics_checks_v19' and hc['complete'] is True
        and hc['passed'] is True and hc['error'] is None and hc['new_production_calls'] == 0
        and hc['old_full_V16_validation_passed'] is False and hc['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True
        and hc['inputs'] == hp['inputs'] == hs['inputs'], 'historical complete675/5400/54/162 independent authority')
    for key, value in dict(saved_error_rows_streamed=1960, saved_error_rows_selected=1715, frames=245, methods=7,
        moment_distributions_checked=675, moment_scalars_checked=5400, paired_groups_checked=54, metric_CI_slots_checked=162).items():
        require(hc['actual_counts'][key] == value, 'exact historical arithmetic ' + key)
    for item in (hc['own_protocol'], hs['own_protocol']):
        require(same(b['evaluation_input:history_validation:STATISTICS_CHECK_PROTOCOL.json'], item), 'actual old checker protocol')
    require(same(b['original_statistics'], hc['checker'])
        and b['evaluation_input:history_adapter']['sha256'] == HISTORY_EVALUATOR_SHA,
        'unchanged historically executed source')
    require(same(b['history_scores'], hc['inputs']['new_scores'])
        and same(b['evaluation_input:history:METRICS.json'], hc['inputs']['METRICS.json'])
        and same(b['evaluation_input:history_statistics_checker'], hc['checker']), 'exact checked historical paired490 controls')
    require(m['population_variance_ddof'] == m['population_std_ddof'] == 0
        and m['unchanged_sample_variance_ddof'] == 1 and m['all245_preserved_no_performance_filter'] is True,
        'population/sample definitions; no filtered accuracy denominator')
    return m

# Exact source regions change population topology and serialized verdict keys,
# never the original scalar, moment, bootstrap, quantile or pair arithmetic.
REGIONS = (
    ("for key,allowed in (('fixed_scores',METHODS[:2]),('ordinary_scores',METHODS[2:4]),('point_scores',('HYBRID_POINT_CONSENSUS_ORIGINAL',OLD)),('new_scores',NEW)):",
     "for key,allowed in (('fixed_scores',METHODS[:2]),('ordinary_scores',METHODS[2:4]),('point_scores',('HYBRID_POINT_CONSENSUS_ORIGINAL',OLD)),('history_scores',HISTORY_NEW),('new_scores',NEW)):"),
    ("audit.eq(count,1960,'four complete paired streams');audit.eq({m:len(g) for m,g in groups.items()},{m:245 for m in METHODS},'selected seven full245 methods')",
     "audit.eq(count,2205,'four paired490 streams plus own245');audit.eq({m:len(g) for m,g in groups.items()},{m:245 for m in METHODS},'selected eight full245 methods')"),
    ("counts=Counter(saved_error_rows_streamed=1960,saved_error_rows_selected=1715,frames=245,methods=7);out={}",
     "counts=Counter(saved_error_rows_streamed=2205,saved_error_rows_selected=1960,frames=245,methods=8);out={}"),
    ("fixed=comparator=='N3_SUBPIX' and scope=='both_new_pose';effective='candidate_new_pose' if fixed else scope",
     "fixed=comparator in METHODS[:2] and scope=='both_new_pose';effective='candidate_new_pose' if fixed else scope"),
    ("audit.eq(counts['moment_distributions_checked'],675,'189 marginal +486 matched distributions');audit.eq(counts['moment_scalars_checked'],5400,'eight fields per distribution');audit.eq(counts['paired_groups_checked'],54,'paired groups');audit.eq(counts['metric_CI_slots_checked'],162,'paired CI slots')",
     "audit.eq(counts['moment_distributions_checked'],783,'216 marginal +567 matched distributions');audit.eq(counts['moment_scalars_checked'],6264,'eight fields per distribution');audit.eq(counts['paired_groups_checked'],63,'paired groups');audit.eq(counts['metric_CI_slots_checked'],189,'paired CI slots')"),
    ("for comparator,field in (('N3_SUBPIX','primary_both_T_R_means_better_than_N3_full245'),(NEW[1],'primary_both_T_R_means_better_than_point_full245')):\n        p=out['combined']['contrasts'][NEW[0]+'_minus_'+comparator]['common_operational'];audit.eq(m[field],p['common_frames']==245 and all(p['metrics'][k]['mean_delta']<0 for k in METRICS[:2]),'saved full245 T/R conclusion')",
     "for comparator in METHODS[:-1]:\n        p=out['combined']['contrasts'][NEW[0]+'_minus_'+comparator]['common_operational'];audit.eq(m['full245_T_R_mean_comparisons'][comparator],p['common_frames']==245 and all(p['metrics'][k]['mean_delta']<0 for k in METRICS[:2]),'saved full245 T/R conclusion')\n    audit.eq(m['primary_both_T_R_means_better_than_N3_full245'],m['full245_T_R_mean_comparisons']['N3_SUBPIX'],'N3 primary verdict alias')\n    audit.eq(m['primary_both_T_R_means_better_than_old_point_full245'],m['full245_T_R_mean_comparisons'][HISTORY_NEW[1]],'historical point verdict alias')"),
)

def extraction(f, instantiate=False):
    require(binding(ORIGINAL)['sha256'] == ORIGINAL_SHA, 'immutable original stdlib checker')
    require(binding(f['independent_scalar'])['sha256'] == SCALAR_SHA, 'immutable independent scalar kernels')
    source = ORIGINAL.read_text(); changed = source
    for before, after in REGIONS:
        require(changed.count(before) == 1, 'exact one topology region'); changed = changed.replace(before, after, 1)
    restored = changed
    for before, after in reversed(REGIONS):
        require(restored.count(after) == 1, 'exact one inverse region'); restored = restored.replace(after, before, 1)
    require(ast.dump(ast.parse(restored), include_attributes=False) == ast.dump(ast.parse(source), include_attributes=False),
        'whole canonical original restoration proof')
    whitelist = ('kernels', 'moments', 'compare', 'compact', 'calculate')
    parsed = ast.parse(changed); nodes = [copy.deepcopy(n) for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in whitelist]
    require(len(nodes) == len(whitelist) and {n.name for n in nodes} == set(whitelist), 'exact inherited five-function whitelist')
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    witness = dict(original=binding(ORIGINAL), exact_topology_regions=len(REGIONS),
        whole_canonical_original_restoration_proven=True, inherited_moment_and_bootstrap_formula_AST_changes=0,
        inherited_AST_whitelist=list(whitelist),
        adapted_AST_sha256=hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        scalar_source=binding(f['independent_scalar']), source_records_2205_selected_1960=True,
        historical_ORIGINAL245_discarded=True, fixed_BASE_N3_not_relabelled_NEW=True)
    if not instantiate: return witness
    environment = dict(math=math, require=require, read=read, binding=binding, ast=ast, Counter=Counter,
        gzip=gzip, json=json, Path=Path, METHODS=METHODS, NEW=NEW, OLD=OLD, HISTORY_NEW=HISTORY_NEW,
        CONTRASTS=CONTRASTS, METRICS=METRICS, UNITS=UNITS, SCOPES=SCOPES,
        ABS_TOL=ABS_TOL, REL_TOL=REL_TOL, SCALAR_SHA=SCALAR_SHA)
    exec(compile(tree, str(ORIGINAL) + ' [V21 topology only; independent stdlib]', 'exec'), environment)
    return witness, environment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    for name in ('repo', 'input', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); args.repo = args.repo.absolute(); args.input = args.input.absolute()
    out = args.output.absolute()
    require(out != args.input and out != PRIVATE and out.is_relative_to(PRIVATE)
        and not any(p.is_symlink() for p in (out, *out.parents)), 'distinct nonsymlink private V21 output')
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES if args.stage == 'freeze' else FILES[1:]: require(not (out / name).exists(), 'preserve ' + name)
    f = paths(args); b = {key: binding(path) for key, path in f.items()}; m = metadata(f, b); witness = extraction(f)
    require(not any(path.absolute().is_relative_to(out) for key, path in f.items() if key != 'checker'), 'output overlaps inputs')
    schema = 'independent_whole_U_closure_statistics_protocol_v21'
    if args.stage == 'freeze':
        write(out / FILES[0], dict(schema=schema, inputs=b, checker=b['checker'], methods=list(METHODS),
            contrasts=[list(pair) for pair in CONTRASTS], source_witness=witness,
            scalar_absolute_tolerance=ABS_TOL, scalar_relative_tolerance=REL_TOL,
            frozen_before_own_saved_error_arithmetic=True, limits=LIMITS)); return
    own = read(out / FILES[0])
    require(own['schema'] == schema and own['inputs'] == b and own['checker'] == b['checker']
        and own['methods'] == list(METHODS) and own['contrasts'] == [list(pair) for pair in CONTRASTS]
        and own['source_witness'] == witness and own['limits'] == LIMITS
        and own['scalar_absolute_tolerance'] == ABS_TOL and own['scalar_relative_tolerance'] == REL_TOL
        and own['frozen_before_own_saved_error_arithmetic'] is True, 'own frozen code/inputs/topology differs')
    protocol = binding(out / FILES[0])
    write(out / FILES[1], dict(inputs=b, own_protocol=protocol, checker=b['checker'], source_witness=witness, new_production_calls=0))
    error = None; counts = {}; reconstructed = {}; audit = None
    try:
        _, env = extraction(f, instantiate=True); h = env['kernels'](f); audit = h['Audit']()
        counts, reconstructed = env['calculate'](f, m, h, audit)
        require(b == {key: binding(path) for key, path in f.items()}, 'inputs changed during saved arithmetic')
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    write(out / FILES[2], dict(schema='independent_whole_U_closure_statistics_checks_v21', complete=error is None,
        passed=error is None, inputs=b, own_protocol=protocol, checker=b['checker'], error=error,
        exact_checks=audit.exact_checks if audit else 0, numeric_checks=audit.numeric_checks if audit else 0,
        max_absolute_difference=audit.max_difference if audit else None, actual_counts=counts,
        reconstructed=reconstructed, source_witness=witness, limits=LIMITS, new_production_calls=0,
        old_full_V16_validation_passed=False))
    if error: raise SystemExit(1)

if __name__ == '__main__': main()
