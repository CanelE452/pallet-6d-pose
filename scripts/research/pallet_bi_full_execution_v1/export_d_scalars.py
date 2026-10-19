"""Export authenticated D postseal scalars, without decoding poses or references.

Only the fixed, explicitly named scalar artifacts below are JSON-decoded.
SCORED_ROWS and prediction packets contain poses: their bytes/seals are hashed
only. This module has no scientific imports and performs no pose evaluation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

VARIANTS = {
    'D_SIGMA_GAMMA_CONFOUNDED': 'D_POSTSEAL_EVALUATION',
    'D_FIXED_COMMON_GAMMA': 'D_FIXED_GAMMA_POSTSEAL_EVALUATION',
}
ANCHORS = {
    'D_POSTSEAL_EVALUATION/EVALUATION_SEAL.json': '4d7537ca298e57f5729aedc29ce60e62cb29b20ff41605bfa503aebf17de402a',
    'D_FIXED_GAMMA_POSTSEAL_EVALUATION/EVALUATION_SEAL.json': 'f7fb55b63cb3743228940ca22c0c1690bab1e4a21a19e521f25847898f148625',
    'D_FIXED_GAMMA_POSTSEAL_EVALUATION/SUPPLEMENT_SEAL.json': '75f14e4ceb9d04e0c6161c41ea647475f652ca6268a294e136742404fa0f5090',
    'D_GAMMA_AUDIT.json': 'b3416f379d60e326f250baa4b02cbdb59539f647557dff7e6ac15221f7cc56c2',
    'D_FIXED_GAMMA_EXECUTION.json': '6963dccafbf119c72cea7c66ea4defea2b77e8ace2db8c598ecea850961f5051',
}
POPS = {'PRIMARY192': 192, 'TEST49': 49, 'REMAINDER143': 143, 'CAL46': 46}
PARTITIONS = {'PRIMARY192': {'TEST', 'REMAINDER'}, 'TEST49': {'TEST'},
              'REMAINDER143': {'REMAINDER'}, 'CAL46': {'CAL'}}
CELLS = ('Q0S0', 'Q1S0', 'Q0S1', 'Q1S1', 'ACTUAL_COMMON', 'ACTUAL_BI')
CONTRASTS = (('Q0S0', 'Q1S0'), ('Q0S1', 'Q1S1'), ('Q0S0', 'Q0S1'), ('Q1S0', 'Q1S1'))
AXES = ('translation_cm', 'rotation_deg', 'success')
PARTS = ('total', 'within_before_selected_family', 'family_readout_contribution')
FAMILIES = ('ORIGINAL_WD', 'SWAPPED_WD')
STATS = ('n', 'mean', 'sample_SD', 'median', 'P90', 'P95', 'max')
DIST_FIELDS = STATS + ('sample_variance_from_saved_SD',)
BASE = ('variant', 'population', 'population_kind', 'denominator')
SUMMARY_FIELDS = BASE + ('cell', 'successes', 'success_rate', 'T_gt50cm', 'T_gt100cm', 'R_gt45deg') + tuple(
    prefix + '_' + name for prefix in ('T_cm', 'R_deg') for name in DIST_FIELDS)
COUNT_FIELDS = BASE + ('contrast', 'family_switches', 'recoveries_n', 'damages_n', 'joint_success_delta',
    'within_before_family_success_delta', 'family_readout_success_delta')
DECOMP_FIELDS = BASE + ('contrast', 'axis', 'component') + DIST_FIELDS
FIXED_FIELDS = BASE + ('contrast', 'family', 'family_denominator', 'joint_success_delta', 'axis') + DIST_FIELDS
ROW_FIELDS = ('variant', 'id', 'partition', 'cluster_id', 'strict_source_compatible', 'contrast',
              'before_family', 'after_family', 'family_changed', 'axis') + PARTS
GAMMA_FIELDS = ('population', 'denominator', 'cell', 'success_delta', 'changed_endpoint_dict_n') + tuple(
    prefix + '_' + name for prefix in ('T_delta_cm', 'R_delta_deg') for name in DIST_FIELDS)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def verify_seal(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError('Authority seal changed: ' + path.name)
    seal = read(path)
    if seal.get('complete') is not True:
        raise ValueError('Completed authority seal required')
    for name, digest in seal['artifacts'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Relative sealed artifact required')
        if sha(path.parent / relative) != digest:
            raise ValueError('Sealed artifact changed: ' + name)
    return seal


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Finite saved scalar required')
    return value


def count(value, n):
    if type(value) is not int or not 0 <= value <= n:
        raise ValueError('Invalid saved count')
    return value


def distribution(value, n):
    if set(value) != set(STATS):
        raise ValueError('Unexpected saved distribution schema')
    result = dict(value)
    available = count(value['n'], n)
    for name in STATS[1:]:
        undefined = available < (2 if name == 'sample_SD' else 1)
        if undefined:
            if value[name] is not None:
                raise ValueError('Undefined statistic must remain null')
        else:
            finite(value[name])
    if value['sample_SD'] is not None and value['sample_SD'] < 0:
        raise ValueError('Negative saved sample SD')
    result['sample_variance_from_saved_SD'] = None if value['sample_SD'] is None else value['sample_SD'] ** 2
    return result


def near(actual, expected, checks, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError('Scalar null mismatch: ' + label)
        return
    error = abs(finite(actual) - finite(expected))
    checks['numeric_comparisons'] += 1
    checks['max_absolute_scalar_difference'] = max(checks['max_absolute_scalar_difference'], error)
    if error > 1e-8 + 1e-12 * max(abs(actual), abs(expected)):
        raise ValueError('Scalar consistency mismatch: ' + label)


def linear_quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (position - lo) * (values[hi] - values[lo])


def check_distribution(saved, values, checks):
    if saved['n'] != len(values):
        raise ValueError('Saved scalar distribution count differs')
    calculated = dict(n=len(values), mean=math.fsum(values) / len(values) if values else None,
        sample_SD=statistics.stdev(values) if len(values) > 1 else None,
        median=statistics.median(values) if values else None,
        P90=linear_quantile(values, .9) if values else None,
        P95=linear_quantile(values, .95) if values else None, max=max(values) if values else None)
    for key in STATS:
        near(saved[key], calculated[key], checks, 'paired ' + key)


def check_effect(effect, summary, checks):
    """Check scalar reassembly with fsum/stdev, independently of NumPy scorer."""
    n = effect['denominator']
    rows = effect['rows']
    if len(rows) != n or len({r['id'] for r in rows}) != n:
        raise ValueError('Duplicate/missing decomposition identities')
    recoveries, damages = [], []
    for row in rows:
        if (row['partition'] not in ('TEST', 'REMAINDER', 'CAL') or
            row['before_family'] not in FAMILIES or row['after_family'] not in FAMILIES or
            type(row['family_changed']) is not bool or
            row['family_changed'] != (row['before_family'] != row['after_family']) or
            set(row['metrics']) != set(AXES)):
            raise ValueError('Unexpected scalar decomposition identity/family schema')
        for axis in AXES:
            parts = row['metrics'][axis]
            if parts is None:
                continue
            if set(parts) != set(PARTS):
                raise ValueError('Unexpected scalar decomposition components')
            near(parts['total'], math.fsum(parts[p] for p in PARTS[1:]), checks, 'within plus family')
            checks['row_component_reassemblies'] += 1
            if axis == 'success':
                if any(v not in (-1., 0., 1.) for v in parts.values()):
                    raise ValueError('Binary success contribution must be -1/0/1')
                if parts['total'] == 1: recoveries.append(row['id'])
                if parts['total'] == -1: damages.append(row['id'])
    if (sorted(recoveries) != sorted(effect['recoveries']) or sorted(damages) != sorted(effect['damages']) or
        effect['family_switches'] != sum(r['family_changed'] for r in rows)):
        raise ValueError('Saved recoveries/damages/family switch count differs')
    for axis in AXES:
        for part in PARTS:
            values = [r['metrics'][axis][part] for r in rows if r['metrics'][axis] is not None]
            check_distribution(effect['decomposition'][axis][part], values, checks)
        d = effect['decomposition'][axis]
        near(d['total']['mean'], math.fsum(d[p]['mean'] for p in PARTS[1:]), checks, 'mean component sum')
    before, after = summary
    near(math.fsum(r['metrics']['success']['total'] for r in rows),
         after['successes'] - before['successes'], checks, 'summary success delta')
    for axis in AXES[:2]:
        near(effect['decomposition'][axis]['total']['mean'],
             after[axis]['mean'] - before[axis]['mean'], checks, 'summary mean delta')


def authenticate(run):
    run = Path(run)
    bindings = {'authority_anchors': ANCHORS, 'sources': {}, 'variants': {}}
    data = {}
    for variant, folder in VARIANTS.items():
        base = run / folder
        seal = verify_seal(base / 'EVALUATION_SEAL.json', ANCHORS[folder + '/EVALUATION_SEAL.json'])
        if seal['rows'] != 238 or seal['status'] != 'DIAGNOSTIC_ONLY':
            raise ValueError('All238 diagnostic score authority required')
        source_name = 'D_DIAGNOSTIC_SCORE.py' if variant == 'D_SIGMA_GAMMA_CONFOUNDED' else 'D_FIXED_GAMMA_DIAGNOSTIC_SCORE.py'
        if sha(run / source_name) != seal['source_sha256']:
            raise ValueError('Original scorer source differs')
        bindings['sources'][source_name] = seal['source_sha256']
        prediction_hashes = {}
        for stage, expected in seal['prediction_seals'].items():
            path = run / stage / 'PREDICTION_SEAL.json'
            if sha(path) != expected:
                raise ValueError('Score prediction seal ancestry differs')
            # Metadata seal only: frame/packet contents are not decoded.
            pred = read(path)
            if pred.get('complete') is not True or pred.get('frame_n', pred.get('full_denominator')) != 238:
                raise ValueError('Complete238 prediction ancestry required')
            prediction_hashes[stage] = expected
        bindings['variants'][variant] = dict(evaluation_seal_sha256=sha(base / 'EVALUATION_SEAL.json'),
            artifacts=seal['artifacts'], prediction_seal_hashes=prediction_hashes,
            reference_sha256=seal['reference_sha256'],
            mixed_pose_scored_rows='hash authenticated only; never decompressed or JSON decoded')
        data[variant] = {name: read(base / (name + '.json')) for name in (
            'SUMMARY', 'PAIRED_DECOMPOSITION', 'STRICT_SOURCE_SHARED_SUBSET', 'RECORDING_GROUPS')}
    folder = VARIANTS['D_FIXED_COMMON_GAMMA']
    supp = verify_seal(run / folder / 'SUPPLEMENT_SEAL.json', ANCHORS[folder + '/SUPPLEMENT_SEAL.json'])
    if (supp['new_score_seal_sha256'] != ANCHORS[folder + '/EVALUATION_SEAL.json'] or
        supp['original_score_seal_sha256'] != ANCHORS['D_POSTSEAL_EVALUATION/EVALUATION_SEAL.json'] or
        supp['numeric_reference_inputs'] is not False or supp['new_solves'] != 0 or
        supp['source_sha256'] != sha(run / 'D_FIXED_GAMMA_POSTSEAL_SUPPLEMENT.py')):
        raise ValueError('Supplement authority/ancestry differs')
    bindings['sources']['D_FIXED_GAMMA_POSTSEAL_SUPPLEMENT.py'] = supp['source_sha256']
    bindings['supplement_artifacts'] = supp['artifacts']
    if len({b['reference_sha256'] for b in bindings['variants'].values()}) != 1:
        raise ValueError('Diagnostic variants have different reference bindings')
    fixed = data['D_FIXED_COMMON_GAMMA']
    fixed['ALL238'] = read(run / folder / 'ALL238_AGGREGATE.json')
    fixed['GAMMA_COMPARISON'] = read(run / folder / 'GAMMA_CORRECTION_COMPARISON.json')
    for name in ('D_GAMMA_AUDIT.json', 'D_FIXED_GAMMA_EXECUTION.json'):
        if sha(run / name) != ANCHORS[name]:
            raise ValueError('Diagnostic provenance metadata differs')
    audit, execution = read(run / 'D_GAMMA_AUDIT.json'), read(run / 'D_FIXED_GAMMA_EXECUTION.json')
    fixed_pred = read(run / 'D_FIXED_COMMON_GRAPH_FIXED_GAMMA/PREDICTION_SEAL.json')
    summary_path = run / 'D_FIXED_COMMON_GRAPH_FIXED_GAMMA/SUMMARY.json'
    if (sha(summary_path) != fixed_pred['artifacts']['SUMMARY.json'] or
        execution['summary_sha256'] != sha(summary_path) or
        execution['prediction_seal_sha256'] != bindings['variants']['D_FIXED_COMMON_GAMMA']['prediction_seal_hashes']['D_FIXED_COMMON_GRAPH_FIXED_GAMMA']):
        raise ValueError('Fixed Gamma execution summary/score ancestry differs')
    production = read(summary_path)
    if (production['Gamma_fixed_COMMON'] is not True or production['full_denominator'] != 238 or
        production['original_D_retained_as_Sigma_plus_Gamma'] is not True or
        production['status'] != 'DIAGNOSTIC_ONLY' or production['numeric_reference_inputs'] is not False):
        raise ValueError('Corrected diagnostic production scope differs')
    bindings['metadata_inputs'] = {'D_FIXED_COMMON_GRAPH_FIXED_GAMMA/SUMMARY.json': sha(summary_path)}
    bindings['authentication_scope'] = ('all canonical score and supplement artifact bytes verified; '
        'prediction-seal metadata authenticated, endpoint packets not decoded or independently rescored')
    return data, bindings, audit, execution, production


def base(variant, population, kind, n):
    return dict(variant=variant, population=population, population_kind=kind, denominator=n)


def flatten_distribution(prefix, saved, n):
    return {prefix + '_' + key: value for key, value in distribution(saved, n).items()}


def tables(data):
    result = dict(POPULATION_METRICS=[], CONTRAST_COUNTS=[], DECOMPOSITION_SUMMARY=[],
                  FIXED_FAMILY_METRICS=[], DECOMPOSITION_ROWS=[], GAMMA_CORRECTION=[])
    checks = dict(PASS=True, numeric_comparisons=0, max_absolute_scalar_difference=0.,
        row_component_reassemblies=0, pathways_checked=0, identity_subset_checks=0,
        no_new_pose_metrics=True, independent_expression='math.fsum, statistics.stdev and explicit sorted linear quantile of saved deltas')
    common_ids, common_strict = None, None
    for variant, source in data.items():
        summary, effects, strict = (source[k] for k in ('SUMMARY', 'PAIRED_DECOMPOSITION', 'STRICT_SOURCE_SHARED_SUBSET'))
        if set(summary) != set(POPS) or set(effects) != set(POPS):
            raise ValueError('Exactly four original populations required')
        if (strict['full_inventory_denominator'] != 238 or strict['valid_counterfactual_n'] != 103 or
            strict['excluded_cross_n'] != 135 or strict['excluded_are_algorithm_failures'] is not False):
            raise ValueError('Strict/source-changing inventory contract differs')
        first = 'Q1S0_minus_Q0S0'
        canonical = {r['id']: r for pop in ('PRIMARY192', 'CAL46') for r in effects[pop][first]['rows']}
        strict_ids = {r['id'] for r in strict['valid_subset_effects'][first]['rows']}
        if len(canonical) != 238 or len(strict_ids) != 103 or not strict_ids <= canonical.keys():
            raise ValueError('All238/strict103 scalar identity inventory differs')
        identities = {key: (r['partition'], r['cluster_id']) for key, r in canonical.items()}
        if common_ids is not None and (common_ids != identities or common_strict != strict_ids):
            raise ValueError('Old/corrected identities or source strata differ')
        common_ids, common_strict = identities, strict_ids
        clusters = source['RECORDING_GROUPS']
        if len(clusters) != 6 or set(clusters) != {r['cluster_id'] for r in canonical.values() if r['partition'] != 'CAL'}:
            raise ValueError('Six PRIMARY recording groups required')
        scopes = [(pop, 'original_reporting_population', n, summary[pop], effects[pop]) for pop, n in POPS.items()]
        scopes.append(('STRICT103', 'conditional_source_compatible_subset', 103,
                       strict['valid_subset_summary'], strict['valid_subset_effects']))
        scopes.extend((cluster, 'PRIMARY_recording_group', item['summary']['Q0S0']['denominator'],
                       item['summary'], item['effects']) for cluster, item in sorted(clusters.items()))
        if 'ALL238' in source:
            all238 = source['ALL238']
            if all238['denominator'] != 238: raise ValueError('Explicit ALL238 supplement required')
            scopes.append(('ALL238', 'supplementary_CAL_plus_PRIMARY', 238, all238['summary'], all238['paired_decomposition']))
        for population, kind, n, sm, ef in scopes:
            b = base(variant, population, kind, n)
            if set(sm) != set(CELLS) or set(ef) != {a + '_minus_' + z for z, a in CONTRASTS}:
                raise ValueError('Complete six readouts/four contrasts required')
            expected_ids = {fid for fid, row in canonical.items() if (
                row['partition'] in PARTITIONS[population] if population in POPS else
                fid in strict_ids if population == 'STRICT103' else
                True if population == 'ALL238' else row['cluster_id'] == population and row['partition'] != 'CAL')}
            if len(expected_ids) != n:
                raise ValueError('Population identity denominator differs')
            for cell in CELLS:
                saved = sm[cell]
                if saved['denominator'] != n: raise ValueError('Metric full denominator differs')
                row = b | dict(cell=cell, successes=count(saved['successes'], n), success_rate=saved['successes'] / n)
                for key in ('T_gt50cm', 'T_gt100cm', 'R_gt45deg'): row[key] = count(saved[key], n)
                for axis, prefix in (('translation_cm', 'T_cm'), ('rotation_deg', 'R_deg')):
                    row.update(flatten_distribution(prefix, saved[axis], n))
                result['POPULATION_METRICS'].append(row)
            by_contrast = {}
            for before, after in CONTRASTS:
                contrast = after + '_minus_' + before
                e = ef[contrast]
                actual = {r['id']: r for r in e['rows']}
                if set(actual) != expected_ids or e['denominator'] != n:
                    raise ValueError('Effect population identity differs')
                for fid, row in actual.items():
                    if (row['partition'], row['cluster_id']) != identities[fid]:
                        raise ValueError('Effect row identity metadata differs')
                # Every repeated scalar row must be exactly the same saved row,
                # including strict subsets, groups and the ALL238 supplement.
                full = {r['id']: r for pop in ('PRIMARY192', 'CAL46') for r in effects[pop][contrast]['rows']}
                if any(row != full[fid] for fid, row in actual.items()):
                    raise ValueError('Repeated population scalar row differs')
                checks['identity_subset_checks'] += n
                by_contrast[contrast] = actual
                check_effect(e, (sm[before], sm[after]), checks)
                sums = {part: int(math.fsum(r['metrics']['success'][part] for r in e['rows'])) for part in PARTS}
                result['CONTRAST_COUNTS'].append(b | dict(contrast=contrast, family_switches=e['family_switches'],
                    recoveries_n=len(e['recoveries']), damages_n=len(e['damages']), joint_success_delta=sums['total'],
                    within_before_family_success_delta=sums[PARTS[1]], family_readout_success_delta=sums[PARTS[2]]))
                for axis in AXES:
                    for part in PARTS:
                        result['DECOMPOSITION_SUMMARY'].append(b | dict(contrast=contrast, axis=axis, component=part) |
                            distribution(e['decomposition'][axis][part], n))
                if set(e['within_fixed_family']) != set(FAMILIES): raise ValueError('Both actual WD families required')
                for family in FAMILIES:
                    fs = e['within_fixed_family'][family]
                    for axis, key in (('translation_cm', 'translation_delta_cm'), ('rotation_deg', 'rotation_delta_deg')):
                        result['FIXED_FAMILY_METRICS'].append(b | dict(contrast=contrast, family=family,
                            family_denominator=count(fs['denominator'], n), joint_success_delta=finite(fs['joint_success_delta']), axis=axis) |
                            distribution(fs[key], n))
                if population in ('PRIMARY192', 'CAL46'):
                    for row in sorted(e['rows'], key=lambda r: r['id']):
                        for axis in AXES:
                            result['DECOMPOSITION_ROWS'].append(dict(variant=variant,
                                **{key: row[key] for key in ('id', 'partition', 'cluster_id', 'before_family', 'after_family', 'family_changed')},
                                strict_source_compatible=row['id'] in strict_ids, contrast=contrast, axis=axis) |
                                (row['metrics'][axis] or {part: None for part in PARTS}))
            for fid in expected_ids:
                for axis in AXES:
                    terms = [by_contrast[key][fid]['metrics'][axis] for key in (
                        'Q1S0_minus_Q0S0', 'Q1S1_minus_Q1S0', 'Q0S1_minus_Q0S0', 'Q1S1_minus_Q0S1')]
                    if all(t is not None for t in terms):
                        near(math.fsum(t['total'] for t in terms[:2]), math.fsum(t['total'] for t in terms[2:]), checks, 'q/Sigma paths')
                        checks['pathways_checked'] += 1
    compare = data['D_FIXED_COMMON_GAMMA']['GAMMA_COMPARISON']
    for pop, n in (('ALL238', 238), ('PRIMARY192', 192), ('CAL46', 46)):
        for cell, item in compare[pop].items():
            if cell not in CELLS[:4] or item['denominator'] != n: raise ValueError('Gamma comparison cell/population differs')
            row = dict(population=pop, denominator=n, cell=cell, success_delta=finite(item['success_delta']),
                changed_endpoint_dict_n=count(item['actual_pose_dict_changed_n'], n))
            for axis, prefix in (('translation_delta_cm', 'T_delta_cm'), ('rotation_delta_deg', 'R_delta_deg')):
                row.update(flatten_distribution(prefix, item[axis], n))
            if pop in POPS:
                old = data['D_SIGMA_GAMMA_CONFOUNDED']['SUMMARY'][pop][cell]
                new = data['D_FIXED_COMMON_GAMMA']['SUMMARY'][pop][cell]
                near(item['success_delta'], new['successes'] - old['successes'], checks, 'Gamma corrected count')
                for axis, metric in (('translation_delta_cm', 'translation_cm'), ('rotation_delta_deg', 'rotation_deg')):
                    near(item[axis]['mean'], new[metric]['mean'] - old[metric]['mean'], checks, 'Gamma corrected mean')
            result['GAMMA_CORRECTION'].append(row)
    return result, checks


def export(run, output):
    output = Path(output)
    if output.exists(): raise FileExistsError('Fresh output folder required; sealed originals are never overwritten')
    data, bindings, audit, execution, production = authenticate(run)
    result, checks = tables(data)
    fields = dict(POPULATION_METRICS=SUMMARY_FIELDS, CONTRAST_COUNTS=COUNT_FIELDS,
        DECOMPOSITION_SUMMARY=DECOMP_FIELDS, FIXED_FAMILY_METRICS=FIXED_FIELDS,
        DECOMPOSITION_ROWS=ROW_FIELDS, GAMMA_CORRECTION=GAMMA_FIELDS)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in result.items():
        with (output / (name + '.csv')).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields[name]); writer.writeheader(); writer.writerows(rows)
    scope = dict(status='DIAGNOSTIC_ONLY', representative_observation_seed=20261018, independent_observation_seeds=1,
        variants=dict(D_SIGMA_GAMMA_CONFOUNDED='old COMMON logical-graph covariance donor also transported reliability Gamma',
            D_FIXED_COMMON_GAMMA='corrected COMMON logical graph and COMMON reliability Gamma fixed; q/Sigma donor sensitivity'),
        original_reporting_populations=POPS, PRIMARY_denominator=192,
        population_overlap='PRIMARY192 is TEST49 plus REMAINDER143; CAL46 is separate; ALL238 includes both and is supplementary',
        strict_source_compatible_n=103, source_changing_n=135, inventory_n=238,
        source_changing_frames_are_algorithm_failures=False,
        strict_interpretation='103 compatible-source rows permit strict counterfactual pairing; 135 source-changing rows do not identify separate physical q/Sigma causal effects',
        fixed_graph_interpretation='conditional algorithmic sensitivity on COMMON logical graph; fixed logical slots do not certify identical physical measurements or owner identity',
        family_decomposition='after-minus-before total = within BEFORE selected WD family + automatic family-readout contribution; family choices are algorithmic, not certified physical ownership',
        solver='limited two-WD-family observation solver, not original D1 full candidate producer or original120-success H1 full system',
        ACTUAL_diagonals='saved limited observation-solver COMMON/BI diagonals, not original H1 estimator',
        initialization_uncertainty='one representative observation seed; not a three-independent-observation-seed study',
        Gamma_audit=dict(changed_frames=audit['changed_frames'], changed_factors=audit['changed_factors']),
        corrected_execution=dict(optimizer_calls=production['actual_new_family_optimizer_calls'],
            family_dispatches=production['actual_new_family_dispatches'], exact_payload_reuses=production['exact_numerical_packet_reuses'],
            CPU_seconds=execution['CPU_seconds'], wall_seconds=execution['wall_seconds'],
            cost_scope='historical fixed-Gamma correction only; not export cost or total campaign wall'),
        uncertainty='descriptive saved diagnostics; no new fit, bootstrap, tuning, independent real test or population-generalization claim',
        continuous='available-only saved n and tails; full binary denominators retained; no trimming, failure deletion or zero imputation',
        saved_status_limitation='these D scalar summaries did not save fallback/accepted/rejected counts; not reconstructed from pose packets',
        privacy='no RGB, q coordinates, covariance matrices, rotation/translation poses, reference values, model weights or private absolute paths',
        raw_reference_decodes=0, mixed_pose_rows_decoded=0, prediction_packets_decoded=0,
        new_pose_metric_calls=0, new_solve_fit_forward_render_calls=0, whole_campaign_complete=False)
    schema = dict(schema='PUBLIC_D_SAVED_SCALAR_TABLES_V1', csv_columns={key + '.csv': list(value) for key, value in fields.items()},
        populations=POPS, supplementary_population='ALL238 only from corrected sealed supplement; no unsaved old ALL238 quantiles invented',
        units=dict(translation_cm='cm', rotation_deg='deg', success='binary delta or fraction as named', sample_variance='axis unit squared'),
        sample_SD_ddof=1, sample_variance_from_saved_SD='square of authenticated saved sample_SD, not a new pose-metric calculation',
        other_derived_scalar_columns='success_rate = saved successes / full denominator; count contributions are sums of saved binary deltas; recovery/damage counts are lengths of saved identity lists',
        saved_quantiles='linear quantiles copied unchanged; per-identity decomposition distributions independently checked with explicit sorted interpolation; fixed-family and Gamma-correction distributions copied from authenticated scalars',
        CSV_null='empty field; genuine numeric zero remains 0/0.0', CSV_boolean='True or False',
        signed_deltas='after minus before; negative translation/rotation is improvement; positive success is improvement',
        per_identity_rows='each variant/contrast/axis exported once across PRIMARY+CAL, with strict source membership; no population duplicate rows',
        changed_endpoint_dict_n='saved actual_pose_dict_changed_n count only; no pose dictionaries exported or decoded',
        metric_threshold='saved joint T<=5cm and C2-aware R<=5deg scorer outcome; native C2 permits registered 180-degree symmetry, not arbitrary 90 degrees',
        no_pose_recomputation=True, original_scoring_independence='D has canonical scorer plus independent scalar arithmetic here; no independent raw-reference pose rescore is claimed')
    write(output / 'METRICS_SCHEMA.json', schema)
    write(output / 'SCOPE_AND_LIMITATIONS.json', scope)
    write(output / 'SCALAR_CHECKS.json', checks)
    write(output / 'INPUT_BINDINGS.json', bindings)
    names = [key + '.csv' for key in fields] + ['METRICS_SCHEMA.json', 'SCOPE_AND_LIMITATIONS.json', 'SCALAR_CHECKS.json', 'INPUT_BINDINGS.json']
    seal = dict(schema='PUBLIC_D_SAVED_SCALAR_EXPORT_SEAL_V1', complete=True, source_sha256=sha(__file__),
        original_reporting_populations=POPS, representative_observation_seed=20261018,
        variants=list(VARIANTS), CSV_rows={key: len(rows) for key, rows in result.items()},
        scalar_checks_PASS=checks['PASS'], raw_reference_decodes=0, mixed_pose_rows_decoded=0,
        prediction_packets_decoded=0, RGB_decodes=0, new_pose_metric_calls=0, new_solve_fit_forward_render_calls=0,
        artifacts={name: sha(output / name) for name in names})
    write(output / 'EXPORT_SEAL.json', seal)
    return seal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.output), sort_keys=True))
