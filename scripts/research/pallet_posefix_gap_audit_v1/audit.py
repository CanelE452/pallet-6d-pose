"""Recount saved PoseFix/current-method metrics, preserving all fixed frames.

Only saved predictions, score journals and seals are read. This module imports
no project inference/scoring code, opens no image or original annotation, and
performs no optimization. Public outputs contain metrics and IDs, not points.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


VIS = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1')
PF95 = VIS / 'posefix_sub_comparison_v1'
PF143 = VIS / 'remainder_after_test49_v1/posefix_baseline_supplement_v4_14'
PFAUDIT = VIS / ('statewise_joint_ot_20261014_v2/nonsevere_v4_fix/independent_review/'
                 'process_visualization_review_v1/POSEFIX_METRIC_POPULATION_AUDIT_V1.json')
ARMS = ('POSEFIX_RAW', 'R0', 'N0', 'N1', 'E0')
PAIRS = (('E0', 'POSEFIX_RAW'), ('N0', 'POSEFIX_RAW'), ('N1', 'POSEFIX_RAW'),
         ('N1', 'N0'), ('E0', 'N0'))
POPULATIONS = ('PRIMARY', 'CAL', 'TEST', 'REMAINDER')
BRIDGE_TOLERANCE = 1e-9
SIGN_TOLERANCE = 1e-9


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def journal(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def distribution(values):
    values = sorted(float(x) for x in values if x is not None)
    if any(not math.isfinite(x) for x in values):
        raise ValueError('Nonfinite numeric metric must not silently disappear')
    n = len(values)
    def quantile(q):
        if not n:
            return None
        position = (n - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (position - lo) * (values[hi] - values[lo])
    return dict(n=n, mean=statistics.fmean(values) if n else None,
        variance_sample=statistics.variance(values) if n > 1 else None,
        variance_population=statistics.pvariance(values) if n else None,
        SD_sample=statistics.stdev(values) if n > 1 else None,
        median=quantile(.5), P90=quantile(.9), P95=quantile(.95),
        max=values[-1] if n else None)


def normalize(metric, *, pose_available=None, old=False):
    available = metric.get('pose_available') if pose_available is None else pose_available
    if not isinstance(available, bool):
        raise ValueError('Explicit actual-pose availability required')
    out = {'pose_available': available}
    for key in ('translation_cm', 'rotation_deg', 'ADDsym_cm'):
        value = metric.get(key)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError('Invalid saved metric: ' + key)
        out[key] = value
    t, r = out['translation_cm'], out['rotation_deg']
    if not available:
        inclusive = strict = False
    elif t is None or r is None:
        inclusive = strict = None
    else:
        inclusive, strict = t <= 5 and r <= 5, t < 5 and r < 5
    out.update(success_le5=inclusive, success_strict_lt5=strict)
    expected = {'success_5cm_5deg': strict} if old else {
        'success_le5': inclusive, 'success_strict_lt5': strict}
    for key, value in expected.items():
        if key not in metric or metric[key] is not value:
            raise ValueError('Saved success flag contradicts metrics: ' + key)
    return out


def group_cases(rows):
    by_session = {}
    for row in rows:
        by_session.setdefault(row['session'], []).append(row['id'])
    return dict(count=len(rows), ids=[r['id'] for r in rows], by_session=by_session)


def paired(rows, arm, baseline):
    cells = {name: [] for name in ('recovery', 'damage', 'both_success', 'both_failure', 'unresolved')}
    differences = {key: [] for key in ('translation_cm', 'rotation_deg')}
    signs = {}
    case_rows = []
    for row in rows:
        a, b = row['metrics'][arm], row['metrics'][baseline]
        sa, sb = a['success_le5'], b['success_le5']
        category = ('unresolved' if sa is None or sb is None else
                    'recovery' if sa and not sb else 'damage' if sb and not sa else
                    'both_success' if sa else 'both_failure')
        cells[category].append(row)
        delta = {}
        sign = []
        for key in differences:
            d = None if a[key] is None or b[key] is None else a[key] - b[key]
            delta[key] = d
            if d is not None:
                differences[key].append(d)
            sign.append('?' if d is None else '-' if d < -SIGN_TOLERANCE else '+' if d > SIGN_TOLERANCE else '=')
        signs.setdefault('T' + sign[0] + '/R' + sign[1], []).append(row)
        case_rows.append(dict(id=row['id'], session=row['session'], transition=category,
                              arm_success=sa, baseline_success=sb, difference=delta))
    return dict(arm=arm, baseline=baseline, difference_definition='arm minus baseline; negative error difference improves',
        full_n=len(rows), known_joint_pairs=len(rows)-len(cells['unresolved']),
        transition={k: group_cases(v) for k, v in cells.items()},
        error_differences={k: distribution(v) for k, v in differences.items()},
        sign_tolerance=SIGN_TOLERANCE, error_sign_patterns={k: group_cases(v) for k, v in sorted(signs.items())},
        cases=case_rows)


def success_patterns(rows, third):
    arms = ['POSEFIX_RAW', 'N0', third]
    cells = {format(i, '03b'): [] for i in range(8)}
    for row in rows:
        key = ''.join('?' if row['metrics'][a]['success_le5'] is None else
                      str(int(row['metrics'][a]['success_le5'])) for a in arms)
        cells.setdefault(key, []).append(row)
    return dict(arms_in_bit_order=arms, bit1='joint success T<=5cm AND R<=5deg', full_n=len(rows),
                cells={k: group_cases(v) for k, v in cells.items()})


def translation_tail(rows):
    records = []
    for row in rows:
        a, b = row['metrics']['N1']['translation_cm'], row['metrics']['N0']['translation_cm']
        if a is not None and b is not None:
            records.append(dict(id=row['id'], session=row['session'], N0_cm=b, N1_cm=a, delta_cm=a-b))
    worsened = sorted((r for r in records if r['delta_cm'] > SIGN_TOLERANCE),
                      key=lambda r: (-r['delta_cm'], r['id']))
    improved = sorted((r for r in records if r['delta_cm'] < -SIGN_TOLERANCE),
                      key=lambda r: (r['delta_cm'], r['id']))
    total = math.fsum(r['delta_cm'] for r in records)
    def portion(records):
        selected = records[:10]
        summed = math.fsum(r['delta_cm'] for r in selected)
        return dict(count=len(selected), cases=selected, sum_delta_cm=summed,
            contribution_to_full_denominator_mean_cm=summed/len(rows) if rows else None,
            share_of_net_total_delta=summed/total if total else None)
    return dict(difference='N1 translation minus N0 translation', full_n=len(rows), finite_pairs=len(records),
        sum_delta_cm=total, mean_delta_finite_pairs_cm=total/len(records) if records else None,
        full_denominator_mean_contribution_cm=total/len(rows) if rows else None,
        positive_delta_sum_cm=math.fsum(r['delta_cm'] for r in worsened),
        negative_delta_sum_cm=math.fsum(r['delta_cm'] for r in improved),
        worsened_count=len(worsened), improved_count=len(improved),
        top10_worsened=portion(worsened), top10_improved=portion(improved),
        warning='Error-improvement ranking, not necessarily a joint-success recovery. Net-total shares may exceed1 when gains and losses cancel.')


def summarize(rows):
    arms = {}
    for arm in ARMS:
        ms = [r['metrics'][arm] for r in rows]
        arms[arm] = dict(full_n=len(rows), no_pose_n=sum(not m['pose_available'] for m in ms),
            finite_translation_n=sum(m['translation_cm'] is not None for m in ms),
            finite_rotation_n=sum(m['rotation_deg'] is not None for m in ms),
            joint_known_n=sum(m['success_le5'] is not None for m in ms),
            joint_success_le5=sum(m['success_le5'] is True for m in ms),
            joint_success_strict_lt5=sum(m['success_strict_lt5'] is True for m in ms),
            joint_failure_le5=sum(m['success_le5'] is False for m in ms),
            joint_unresolved_n=sum(m['success_le5'] is None for m in ms),
            errors={k: distribution([m[k] for m in ms]) for k in ('translation_cm', 'rotation_deg', 'ADDsym_cm')})
    return dict(full_n=len(rows), arms=arms,
        paired={a+'-'+b: paired(rows, a, b) for a, b in PAIRS},
        three_way_success_patterns={c: success_patterns(rows, c) for c in ('N1', 'E0')},
        N1_minus_N0_translation_tail=translation_tail(rows))


def run(*, source_root, guarded_score, single_score, output):
    started = time.monotonic()
    source_root, guarded_score, single_score, output = map(Path, (source_root, guarded_score, single_score, output))
    if output.exists():
        raise FileExistsError('Preserve previous saved-result audit attempts')
    a, b = source_root/PF95, source_root/PF143
    inputs = []
    def binding(path, alias=None):
        if alias is None:
            alias = str(path.relative_to(source_root))
        return dict(path=alias, sha256=sha(path), bytes=path.stat().st_size)
    def load(path, alias=None):
        inputs.append(binding(path, alias))
        return read(path)
    def sealed_asset(folder, seal_name, asset_name, key):
        seal = load(folder/seal_name)
        inputs.append(binding(folder/asset_name))
        if sha(folder/asset_name) != seal[key]:
            raise ValueError('Saved PoseFix seal differs: ' + asset_name)
        return seal
    seal95 = sealed_asset(a, 'PREDICTION_SEAL.json', 'PREDICTIONS.jsonl.gz', 'predictions_sha256')
    seal143 = sealed_asset(b, 'BASELINE_SEAL.json', 'PREDICTIONS.jsonl.gz', 'predictions_sha256')
    sealed_asset(b, 'SCORE_SEAL.json', 'POSTSEAL_SCORE.json', 'score_sha256')
    scored95 = journal(a/'SCORED_ROWS.jsonl.gz')
    inputs.append(binding(a/'SCORED_ROWS.jsonl.gz'))
    audit95 = load(source_root/PFAUDIT)
    if (not audit95['passed'] or audit95['failures'] or
        audit95['bindings'][str(PF95/'SCORED_ROWS.jsonl.gz')] != sha(a/'SCORED_ROWS.jsonl.gz')):
        raise ValueError('Historical independent95 scored-journal seal differs')
    protocol95, protocol143 = load(a/'PROTOCOL.json'), load(b/'PROTOCOL.json')
    if sha(a/'PROTOCOL.json') != seal95['protocol_sha256']:
        raise ValueError('Old95 protocol changed')
    if sha(b/'PROTOCOL.json') != seal143['protocol_sha256']:
        raise ValueError('Old143 protocol changed')
    if protocol95['PoseFix']['checkpoint_sha256'] != protocol143['checkpoint_sha256']:
        raise ValueError('The two saved PoseFix populations use different checkpoints')
    comp95 = load(a/'COMPARISON.json')
    if sha(a/'PREDICTION_SEAL.json') != comp95['prediction_seal_sha256']:
        raise ValueError('Old95 comparison prediction-seal binding changed')
    source143 = read(b/'POSTSEAL_SCORE.json')['rows']
    pred143 = journal(b/'PREDICTIONS.jsonl.gz')
    avail143 = {r['id']: r['actual_pose']['available'] for r in pred143}
    current = {}
    seals = {}
    for name, path in [('single_score', single_score), ('guarded_score', guarded_score)]:
        packet = load(path, 'PRIVATE:'+name+'/'+path.name)
        seal_path = path.with_name('SCORE_SEAL.json')
        seal = load(seal_path, 'PRIVATE:'+name+'/SCORE_SEAL.json')
        if seal.get('artifacts', {}).get(path.name) != sha(path):
            raise ValueError('Current scored journal does not match its saved score seal')
        current[name], seals[name] = packet['rows'], seal
    single, guarded = current['single_score'], current['guarded_score']
    if len(single) != 238 or len({r['id'] for r in single}) != 238:
        raise ValueError('Exactly238 fixed current rows required')
    if [r['id'] for r in single] != [r['id'] for r in guarded]:
        raise ValueError('Current E and N/RAW scored row order differs')
    if Counter(r['partition'] for r in single) != Counter(CAL=46, TEST=49, REMAINDER=143):
        raise ValueError('Fixed CAL46/TEST49/REMAINDER143 required')
    cur = {r['id']: r for r in single}
    for s, g in zip(single, guarded):
        for key in ('partition', 'session', 'cluster_id'):
            if s[key] != g[key]:
                raise ValueError('Fixed current identity/cluster metadata differs')
    old95 = {(r['id'], r['method']): r for r in scored95}
    old143 = {r['id']: r for r in source143}
    pf = {r['id']: normalize(r['metric'], pose_available=r['actual_pose']['available'], old=True)
          for r in scored95 if r['method'] == 'POSEFIX_RAW'}
    pf.update({r['id']: normalize(r['metrics']['POSEFIX_RAW'], pose_available=avail143[r['id']], old=True)
               for r in source143})
    if len(pf) != 238 or set(pf) != set(cur) or len(old143) != 143 or len(avail143) != 143:
        raise ValueError('Saved PoseFix238 must match fixed current IDs exactly')
    for r in scored95:
        if r['partition'] != cur[r['id']]['partition'] or r['session'] != cur[r['id']]['session']:
            raise ValueError('Old95 partition/session differs')
    for r in source143:
        if cur[r['id']]['partition'] != 'REMAINDER' or r['session'] != cur[r['id']]['session']:
            raise ValueError('Old143 identity/session differs')
    shared = seals['single_score']['reference_bindings']
    if shared != seals['guarded_score']['reference_bindings']:
        raise ValueError('Current saved E/N fixed reference bindings differ')
    reference_checks = {}
    for part in ('CAL', 'TEST'):
        binding_ref = shared[part]
        matches = [v for k, v in comp95['isolated_scoring_sha256'].items()
                   if k.replace('\\', '/').endswith('/'+binding_ref['path'])]
        if matches != [binding_ref['sha256']]:
            raise ValueError('PoseFix/current CAL/TEST reference SHA differs')
        reference_checks[part] = binding_ref
    if protocol143['reference_plan']['sha256'] != shared['REMAINDER']['sha256']:
        raise ValueError('PoseFix/current REMAINDER fixed reference plan differs')
    reference_checks['REMAINDER'] = shared['REMAINDER']
    bridge_max = {a: {k: 0. for k in ('translation_cm', 'rotation_deg', 'ADDsym_cm')} for a in ('N0', 'R0')}
    for s in single:
        for arm, oldname in [('N0', 'INITIAL_N3_SUBPIX'), ('R0', 'NATIVE_BASE')]:
            if arm == 'R0' and s['partition'] == 'REMAINDER':
                continue
            old_metric = (old143[s['id']]['metrics'][oldname] if s['partition'] == 'REMAINDER' else
                          old95[s['id'], oldname]['metric'])
            for key in bridge_max[arm]:
                x, y = s['metrics'][arm][key], old_metric[key]
                if x is None or y is None:
                    if x is not y:
                        raise ValueError('Shared baseline availability/metric bridge differs')
                else:
                    bridge_max[arm][key] = max(bridge_max[arm][key], abs(x-y))
                    if abs(x-y) > BRIDGE_TOLERANCE:
                        raise ValueError('Shared old/current baseline metric bridge exceeds tolerance')
    output_rows = []
    boundaries = []
    for s, g in zip(single, guarded):
        metrics = {'POSEFIX_RAW': pf[s['id']]}
        metrics.update({arm: normalize(s['metrics'][arm]) for arm in ('R0', 'N0', 'N1')})
        metrics['E0'] = normalize(g['metrics']['E0'])
        out = {k: s[k] for k in ('id', 'partition', 'session', 'cluster_id')}
        out['metrics'] = metrics
        output_rows.append(out)
        for arm, m in metrics.items():
            if m['success_le5'] is not m['success_strict_lt5']:
                boundaries.append(dict(id=s['id'], partition=s['partition'], method=arm,
                    translation_cm=m['translation_cm'], rotation_deg=m['rotation_deg']))
    groups = {p: [r for r in output_rows if r['partition'] != 'CAL'] if p == 'PRIMARY' else
              [r for r in output_rows if r['partition'] == p] for p in POPULATIONS}
    result = dict(schema='POSEFIX_SAVED_FULL_PIPELINE_GAP_AUDIT_V1', arms=list(ARMS), rows=238,
        populations={p: summarize(group) for p, group in groups.items()},
        boundary_check=dict(checked_arm_rows=238*len(ARMS), strict_and_inclusive_differ_count=len(boundaries), cases=boundaries),
        bridge=dict(tolerance=BRIDGE_TOLERANCE, max_abs_difference=bridge_max, N0_rows=238, R0_rows=95,
            R0_remainder143_bridge='Not available in old143 journal; INITIAL_N3_SUBPIX is a different method.'),
        limitations=['Saved scores only; no new inference, annotation or image access.',
            'PoseFix is the frozen synthetic-trained pallet9 RGB ResNet152-derived first RAW pass, not official human-benchmark reproduction.',
            'PoseFix/SubPix two-anchor combinations exist only for CAL46/TEST49; no PRIMARY192 result is inferred for them.',
            'Head coordinates, chosen W/D family, observation extraction and optimization differ: this is a paired full-pipeline comparison.',
            'Previously observed development data and annotation-derived pose references; not a new independent evaluation.'],
        default_or_model_or_method_changed=False)
    review = output.parent/'CONTRACT_REVIEW.json'
    if review.exists():
        inputs.append(binding(review, 'PRIVATE:posefix_gap_audit/CONTRACT_REVIEW.json'))
    receipt = dict(schema='POSEFIX_GAP_SAVED_RESULT_RECOUNT_RECEIPT_V1', status='PASS',
        source_sha256={'scripts/research/pallet_posefix_gap_audit_v1/audit.py': sha(__file__)},
        input_bindings=inputs, shared_reference_bindings=reference_checks,
        expected_population_counts=dict(PRIMARY=192, CAL=46, TEST=49, REMAINDER=143),
        actual_population_counts={k: len(v) for k, v in groups.items()},
        boundary_check=result['boundary_check'], shared_baseline_bridge=result['bridge'],
        PF_checkpoint_sha256=protocol95['PoseFix']['checkpoint_sha256'],
        PF_variant=protocol95['PoseFix'], prior_independent_TEST49_metric_audit_checks=audit95['checks'],
        prior_audit_scope='TEST49 numeric parity audit; its SHA binding authenticates the complete95-frame scored journal.',
        GT_raw_coordinate_or_pose_decodes=0, images_opened=0, model_or_mask_forwards=0,
        PnP_or_SubPix_calls=0, new_training_or_seed=0, old_files_modified=0,
        wall_seconds=time.monotonic()-started)
    output.mkdir(parents=True)
    write(output/'RESULTS.json', result)
    fields = ['id', 'partition', 'session', 'cluster_id']
    keys = ['pose_available', 'translation_cm', 'rotation_deg', 'ADDsym_cm', 'success_le5', 'success_strict_lt5']
    fields += [a+'__'+k for a in ARMS for k in keys]
    with (output/'ALL_ROWS.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in output_rows:
            flat = {k: row[k] for k in fields[:4]}
            flat.update({a+'__'+k: row['metrics'][a][k] for a in ARMS for k in keys})
            writer.writerow(flat)
    write(output/'RECEIPT.json', receipt)
    write(output/'SEAL.json', dict(source_sha256=receipt['source_sha256'],
        artifacts={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        all238_ids_preserved=True, no_new_predictions_or_GT_edits=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'guarded-score', 'single-score', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    result = run(source_root=args.source_root, guarded_score=args.guarded_score,
                 single_score=args.single_score, output=args.output)
    print(json.dumps(dict(status='PASS', rows=result['rows'],
        PRIMARY={a: s['joint_success_le5'] for a, s in result['populations']['PRIMARY']['arms'].items()},
        strict_inclusive_boundary_differences=result['boundary_check']['strict_and_inclusive_differ_count'])))
