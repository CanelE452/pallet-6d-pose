"""Independently sum PRIMARY372 filter counters; source-only until ROOT GO.

Reads exactly seven ROOT-hash-bound derived metadata/counter tables and the
frozen producer source (hash only). No raw RGB/annotation, XY/prediction/mask
cache, checkpoint, NN, PnP, bootstrap, or new selection is opened/executed.
Pooled n/sum/sum-squared determine mean and population SD, but not a pooled
median; this verifier explicitly does not claim to recheck aggregate medians.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import datetime
import hashlib
import json
import math
import re

OUT = Path(__file__).resolve().parents[1]
WOOD = OUT.parent / 'dope-wood640-eval80-20261027-v1'
COMBINED = WOOD / 'combined372'
PRODUCER = WOOD / 'code/merged_filter_diagnostics.py'
PRODUCER_SHA = 'e83f23bfe3f1e1429654ed7e2036993abd68849498ff0cb0b30a1f9693f6832a'
INPUTS = OUT / 'preflight/MERGED_FILTER_STATISTICS_INPUTS.json'
GO = OUT / 'preflight/MERGED_FILTER_STATISTICS_GO.json'
RESULT = OUT / 'preflight/MERGED_FILTER_STATISTICS_QA.json'
EXPECTED = {'PLASTIC': (128, 985), 'WOOD': (125, 963), 'GREEN': (119, 943)}
CONDITIONS = ('full', 'no_confidence', 'no_ransac', 'no_loo', 'no_flip',
              'no_final_support', 'confidence_only')
STAGES = (('main', 's0'), ('main', 's1'), ('main', 's2'), ('main', 's3'), ('main', 'final'),
          ('noflip', 's0'), ('noflip', 's1'), ('noflip', 's2'), ('noflip', 'final'),
          ('confidence_only', 'original_only'), ('confidence_only', 'both_view'))
ROLES = {'component_summary', 'stage_summary', 'component_frames', 'stage_frames',
         'selection_equivalence', 'diagnostic_complete', 'merged_manifest'}
COMPONENT_KEY = ('kind', 'condition', 'panel')
STAGE_KEY = ('model', 'teacher_role', 'kind', 'filter', 'stage', 'panel')
COMPONENT_COUNTERS = ('accepted_frames', 'accepted_points', 'n_reference', 'scorable_accepted_points',
                      'observed_teacher_points', 'missing_teacher_points',
                      *[p + str(t) + 'px' for t in (5, 10, 20) for p in ('teacher_correct_le', 'selected_correct_le')])
STAGE_COUNTERS = ('total_mask_corners', 'frames_nonempty', 'frames_ge6', 'eligible_ref_count',
                  'annotated_eligible_selected_ref_count', 'rejected_ref_count',
                  'selected_ref_observed_count', 'rejected_ref_observed_count',
                  *[p + '_correct_le' + str(t) + 'px' for t in (5, 10, 20) for p in ('selected', 'rejected')])
RAW_KEYS = {'xy', 'points', 'reference_xy', 'predictions', 'all_annotated_reference',
            'projected_cuboid', 'manual_kps', 'trusted_mask', 'canonical_xyz'}


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
        require(not (RAW_KEYS & set(value)), 'Coordinate/mask cache is prohibited')
        for item in value.values():
            no_coordinates(item)
    elif isinstance(value, list):
        for item in value:
            no_coordinates(item)


def number(value):
    if value in ('', None, 'NA', 'None'):
        return None
    result = float(value)
    require(math.isfinite(result), 'Nonfinite numerical input')
    return result


def count(value):
    result = number(value)
    require(result is not None and result >= 0 and result == int(result), 'Invalid nonnegative count')
    return int(result)


def same(actual, saved, context, abs_tol=1e-10):
    saved = number(saved)
    require((actual is None and saved is None) or (actual is not None and saved is not None
            and math.isclose(actual, saved, rel_tol=1e-12, abs_tol=abs_tol)), 'Mismatch: ' + context)


def ratio(a, b):
    return a / b if b else None


def panels(kind):
    return ('all_annotated_reference', 'verified_manual_only') if kind in ('WOOD', 'GREEN') else ('all_annotated_reference',)


def nref(kind, panel):
    return (330 if kind == 'WOOD' else 600) if panel == 'verified_manual_only' else EXPECTED[kind][1]


def teachers(kind):
    return (('R0_before_selftraining', 'R0'), (kind + '_LOO_flip_R1', 'flip_R1'),
            (kind + '_LOO_only_R1', 'loo_R1'))


def key(row, fields):
    return tuple(row[k] for k in fields)


def contract():
    return dict(input_manifest_path=str(INPUTS), go_path=str(GO), output=str(RESULT),
        manifest=dict(schema='primary372_filter_statistics_inputs_v1', verifier_source_sha256='CURRENT_VERIFIER_SHA256',
            producer_source_sha256=PRODUCER_SHA,
            input_files={'/EXACT/ABSOLUTE/PATH': {'role': 'ONE_REQUIRED_ROLE', 'sha256': 'CURRENT_SHA256'}},
            upstream_sha256={'/EXACT/UPSTREAM/PATH': 'SHA_FROM_EQUIVALENCE_OR_COMPLETE_METADATA'}),
        required_roles=sorted(ROLES),
        upstream_rule='Exact conflict-free union of EQUIVALENCE.source_input_sha256 and COMPLETE.input_sha256; '
                      'compared as metadata only, never opened or rehashed',
        GO=dict(schema='primary372_filter_statistics_go_v1', passed=True, statistics_verification_authorized=True,
                input_manifest_sha256='CURRENT_MANIFEST_SHA256', verifier_source_sha256='CURRENT_VERIFIER_SHA256'),
        median_limitation='Pooled median cannot be recovered from moments; not independently verified',
        source_only_until_separate_ROOT_GO=True)


class Inputs:
    def __init__(self):
        self.bindings = {}
        self.code_sha = sha(__file__)
        self.spec = self.fixed_json(INPUTS)
        go = self.fixed_json(GO)
        require(self.spec['schema'] == 'primary372_filter_statistics_inputs_v1'
                and self.spec['producer_source_sha256'] == PRODUCER_SHA
                and self.spec['verifier_source_sha256'] == self.code_sha, 'Frozen source contract differs')
        require(go['schema'] == 'primary372_filter_statistics_go_v1' and go['passed'] is True
                and go['statistics_verification_authorized'] is True
                and go['input_manifest_sha256'] == self.bindings[str(INPUTS)]
                and go['verifier_source_sha256'] == self.code_sha, 'Explicit ROOT statistics GO is absent/stale')
        self.files = self.spec['input_files']
        require(isinstance(self.files, dict) and len(self.files) == 7
                and {v['role'] for v in self.files.values()} == ROLES, 'Exactly seven derived input roles required')
        self.paths = {}
        for path, item in self.files.items():
            p = Path(path)
            require(p.is_absolute() and str(p.resolve()) == path and not p.is_symlink()
                    and p.is_file() and digest_ok(item['sha256']), 'Unsafe path/SHA')
            self.paths[item['role']] = p
        score = self.paths['diagnostic_complete'].parent
        require(score.name == 'score' and score.parent.parent == COMBINED / 'filter_diagnostics'
                and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', score.parent.name), 'Not the merged diagnostic directory')
        for role, suffix in dict(component_summary='public/COMPONENT_QUALITY.csv',
                stage_summary='public/T3_FILTER_STAGES.csv', component_frames='private/COMPONENT_FRAME_COUNTERS.csv',
                stage_frames='private/T3_FRAME_COUNTERS.csv', diagnostic_complete='COMPLETE.json').items():
            require(self.paths[role] == score / suffix, 'Exact statistic output path differs: ' + role)
        require(self.paths['selection_equivalence'] == score.parent / 'select/EQUIVALENCE.json'
                and self.paths['merged_manifest'] == COMBINED / 'MERGED_MANIFEST.json', 'Exact metadata path differs')
        require(PRODUCER.resolve() == PRODUCER and not PRODUCER.is_symlink()
                and sha(PRODUCER) == PRODUCER_SHA, 'Frozen diagnostic producer differs')
        self.bindings[str(PRODUCER)] = PRODUCER_SHA

    def fixed_json(self, path):
        require(path.resolve() == path and path.is_file() and not path.is_symlink(), 'Missing/unsafe ROOT metadata')
        digest = sha(path)
        obj = json.loads(path.read_text())
        no_coordinates(obj)
        require(sha(path) == digest, 'ROOT metadata changed during read')
        self.bindings[str(path)] = digest
        return obj

    def bind(self, role):
        p = self.paths[role]
        digest = sha(p)
        require(digest == self.files[str(p)]['sha256'], 'Input SHA changed: ' + role)
        self.bindings[str(p)] = digest
        return p

    def read(self, role):
        p = self.bind(role)
        obj = json.loads(p.read_text())
        no_coordinates(obj)
        require(sha(p) == self.bindings[str(p)], 'JSON changed during read')
        return obj

    def csv(self, role):
        p = self.bind(role)
        with p.open(newline='') as stream:
            reader = csv.DictReader(stream)
            require(not (RAW_KEYS & set(reader.fieldnames or ())), 'Coordinate/mask CSV blocked')
            rows = list(reader)
        require(sha(p) == self.bindings[str(p)], 'CSV changed during read')
        return rows

    def unchanged(self):
        require(sha(__file__) == self.code_sha, 'Verifier source changed')
        for path, digest in self.bindings.items():
            require(sha(path) == digest, 'Bound safe input/source changed: ' + path)


def gates(io):
    proof = io.read('selection_equivalence')
    for k, v in dict(complete=True, passed=True, old_teacher_frame_pairs=876, new_teacher_frame_pairs=240,
            combined_teacher_frame_pairs=1116, new_chosen_comparisons=480, old_MAIN_comparisons=1752,
            mask9_pass_inliers_exact=True, original_final_decomposition_exact=True, raw_targets_unchanged=True,
            centroid_mask_false=True, reference_coordinates_read=False, NN_forwards=0, raw_RGB_reads=0,
            raw_annotation_reads=0).items():
        require(proof.get(k) == v, 'Original selection/equivalence contract failed: ' + k)
    done = io.read('diagnostic_complete')
    for k, v in dict(complete=True, passed=True, population='PRIMARY372', frames=372, WOOD_frames=125,
            component_rows=35, T3_stage_rows=165, component_frame_rows=4312, T3_frame_rows=20328,
            original292_component28_exact=True, original292_T3_stage132_exact=True,
            selection_frozen_before_reference_read=True, NN_forwards=0, PnP_fits=0, training_updates=0,
            raw_RGB_reads=0, raw_annotation_reads=0, public_coordinate_arrays=False).items():
        require(done.get(k) == v, 'Diagnostic completion contract failed: ' + k)
    require(done['references'] == {k: v[1] for k, v in EXPECTED.items()}
            and done['manual_references'] == {'WOOD': 330, 'GREEN': 600}, 'Reference denominators differ')
    sources = {}
    for mapping in (proof['source_input_sha256'], done['input_sha256']):
        for path, digest in mapping.items():
            require(Path(path).is_absolute() and digest_ok(digest), 'Invalid upstream metadata SHA')
            require(path not in sources or sources[path] == digest, 'Conflicting upstream input identity')
            sources[path] = digest
    require(sources == io.spec['upstream_sha256'] and sources[str(PRODUCER)] == PRODUCER_SHA,
            'ROOT upstream metadata union differs')
    equiv_path = io.paths['selection_equivalence']
    require(done['input_sha256'].get(str(equiv_path)) == io.bindings[str(equiv_path)], 'Score did not bind exact equivalence receipt')
    selected = equiv_path.parent
    require(set(proof['files_sha256']) == {str(selected / name) for name in
            ('COMPONENT_SELECTIONS_PRIVATE.json', 'T3_STAGE_MASKS_PRIVATE.json')}, 'Selection output catalogue differs')
    for path, digest in proof['files_sha256'].items():
        require(digest_ok(digest) and done['input_sha256'].get(path) == digest, 'Frozen selection output binding differs')
    for role in ('component_summary', 'stage_summary', 'component_frames', 'stage_frames'):
        p = io.paths[role]
        require(done['output_sha256'].get(str(p)) == io.files[str(p)]['sha256'], 'Score output SHA differs: ' + role)
    manifest = io.read('merged_manifest')
    frames = manifest['records']
    known = {r['id']: r['kind'] for r in frames}
    require(len(frames) == len(known) == 372 and set(known.values()) == set(EXPECTED), 'Exact primary372 ID/kind catalogue required')
    ids = {kind: {i for i, k in known.items() if k == kind} for kind in EXPECTED}
    require(all(len(ids[k]) == n for k, (n, _) in EXPECTED.items()), 'Frame count per kind differs')
    return ids


def moments(row, prefix, expected_n=None):
    """Check one saved moment tuple; return additive n/sum/sum-squared only."""
    n = count(row[prefix + 'error_n'])
    if expected_n is not None:
        require(n == expected_n, 'Observed error n differs: ' + prefix)
    s, q = number(row[prefix + 'error_sum_px']), number(row[prefix + 'error_sum_squared_px'])
    mean, sd, median = (number(row[prefix + k]) for k in ('error_mean_px', 'error_SD_px', 'error_median_px'))
    require(s is not None and q is not None and s >= 0 and q >= 0, 'Invalid additive error moments')
    if not n:
        require(s == q == 0 and mean is sd is median is None, 'Empty error distribution must stay NA with zero moments')
    else:
        require(mean is not None and sd is not None and median is not None and min(mean, sd, median) >= 0,
                'Observed error statistics missing/negative')
        same(s / n, mean, 'Moment mean ' + prefix)
        v = q / n - (s / n) ** 2
        # Cancellation tolerance is fixed before data: compare variance units,
        # scaled by the second moment, rather than silently clipping an SD bar.
        tolerance = max(1e-10, 1e-12 * max(q / n, (s / n) ** 2, 1.0))
        require(v >= -tolerance and abs(sd * sd - max(v, 0.0)) <= tolerance,
                'Population SD disagrees with saved moments: ' + prefix)
    return n, s, q


def pooled(summary, frame_rows, prefix, expected_n):
    values = [moments(row, prefix) for row in frame_rows]
    n = sum(v[0] for v in values)
    s = math.fsum(v[1] for v in values)
    q = math.fsum(v[2] for v in values)
    require(n == expected_n, 'Pooled observation denominator differs')
    same(n, summary[prefix + 'error_n'], 'Pooled error n')
    same(s, summary[prefix + 'error_sum_px'], 'Pooled error sum', abs_tol=1e-8)
    same(q, summary[prefix + 'error_sum_squared_px'], 'Pooled error sum-squared', abs_tol=1e-8)
    moments(summary, prefix, n)
    same(s / n if n else None, summary[prefix + 'error_mean_px'], 'Pooled mean')
    if n:
        sd = number(summary[prefix + 'error_SD_px'])
        v = q / n - (s / n) ** 2
        tolerance = max(1e-10, 1e-12 * max(q / n, (s / n) ** 2, 1.0))
        require(v >= -tolerance and abs(sd * sd - max(v, 0.0)) <= tolerance, 'Pooled population SD differs')


def grouped(rows, fields, expected, ids):
    groups = defaultdict(list)
    frame_keys = set()
    for row in rows:
        k = key(row, fields)
        require(k in expected and (k, row['id']) not in frame_keys, 'Unexpected/duplicate per-frame counter key')
        frame_keys.add((k, row['id']))
        kind = row['kind']
        require(row['id'] in ids[kind], 'Counter frame outside fixed kind catalogue')
        groups[k].append(row)
    require(set(groups) == expected, 'Missing frame-counter group')
    for k, rr in groups.items():
        require({r['id'] for r in rr} == ids[rr[0]['kind']], 'Frame counter population is incomplete')
    return groups


def catalogue(rows, fields, expected):
    result = {key(r, fields): r for r in rows}
    require(len(result) == len(rows) and set(result) == expected, 'Summary catalogue is not exact')
    return result


def components(io, ids):
    expected = {(k, c, p) for k in EXPECTED for c in CONDITIONS for p in panels(k)}
    summaries = catalogue(io.csv('component_summary'), COMPONENT_KEY, expected)
    frames = io.csv('component_frames')
    groups = grouped(frames, COMPONENT_KEY, expected, ids)
    require(len(summaries) == 35 and len(frames) == 4312, 'Expected35/4312 component rows')
    teacher_invariants = {}
    for k, rr in groups.items():
        kind, condition, panel = k
        s = summaries[k]
        require(s['teacher'] == 'R0_before_selftraining' and s['status'] == 'COMPUTED'
                and count(s['n_frames']) == EXPECTED[kind][0]
                and count(s['n_reference_points']) == nref(kind, panel), 'Component summary identity/denominator differs')
        total = {field: 0 for field in COMPONENT_COUNTERS}
        for row in rr:
            fc = {field: count(row[field]) for field in COMPONENT_COUNTERS}
            require(fc['accepted_frames'] <= 1 and fc['accepted_points'] <= 8 and fc['n_reference'] <= 8
                    and fc['observed_teacher_points'] + fc['missing_teacher_points'] == fc['n_reference']
                    and fc['scorable_accepted_points'] <= min(fc['observed_teacher_points'], fc['accepted_points']),
                    'Component scorable/observed/missing counts differ')
            moments(row, 'teacher_', fc['observed_teacher_points'])
            moments(row, 'selected_', fc['scorable_accepted_points'])
            prev_teacher = prev_selected = 0
            for tau in (5, 10, 20):
                good, chosen = fc[f'teacher_correct_le{tau}px'], fc[f'selected_correct_le{tau}px']
                require(prev_teacher <= good <= fc['observed_teacher_points']
                        and prev_selected <= chosen <= min(good, fc['scorable_accepted_points']), 'Component correctness counts differ')
                prev_teacher, prev_selected = good, chosen
            if condition != 'no_confidence':
                signature = tuple(row[f] for f in ('n_reference', 'observed_teacher_points', 'missing_teacher_points',
                    'teacher_correct_le5px', 'teacher_correct_le10px', 'teacher_correct_le20px',
                    'teacher_error_n', 'teacher_error_sum_px', 'teacher_error_sum_squared_px'))
                invariant_key = (kind, panel, row['id'])
                require(invariant_key not in teacher_invariants or teacher_invariants[invariant_key] == signature,
                        'Canonical R0 teacher population changed between component conditions')
                teacher_invariants[invariant_key] = signature
            for field, value in fc.items():
                total[field] += value
        require(total['n_reference'] == nref(kind, panel), 'Component full reference count differs')
        for field in COMPONENT_COUNTERS:
            if field != 'n_reference':
                same(total[field], s[field], 'Component total ' + field)
        for tau in (5, 10, 20):
            good = total[f'teacher_correct_le{tau}px']
            chosen = total[f'selected_correct_le{tau}px']
            bad = total['observed_teacher_points'] - good
            chosen_bad = total['scorable_accepted_points'] - chosen
            for field, value in {f'precision_le{tau}px': ratio(chosen, total['scorable_accepted_points']),
                    f'correct_retention_le{tau}px': ratio(chosen, good), f'teacher_incorrect_gt{tau}px': bad,
                    f'selected_incorrect_gt{tau}px': chosen_bad, f'incorrect_acceptance_gt{tau}px': ratio(chosen_bad, bad)}.items():
                same(value, s[field], 'Component ratio/count ' + field)
        same(ratio(total['scorable_accepted_points'], nref(kind, panel)), s['selected_reference_coverage'], 'Selected reference coverage')
        pooled(s, rr, 'teacher_', total['observed_teacher_points'])
        pooled(s, rr, 'selected_', total['scorable_accepted_points'])
    return dict(summary_rows=35, frame_rows=4312, pooled_distributions_checked=70,
                no_confidence_retention_uses_its_own_decoder_teacher_denominator=True)


def stages(io, ids):
    expected = {(model, role, k, mode, stage, panel) for k in EXPECTED
                for model, role in teachers(k) for mode, stage in STAGES for panel in panels(k)}
    summaries = catalogue(io.csv('stage_summary'), STAGE_KEY, expected)
    frames = io.csv('stage_frames')
    groups = grouped(frames, STAGE_KEY, expected, ids)
    require(len(summaries) == 165 and len(frames) == 20328, 'Expected165/20328 stage rows')
    partition = {}
    sequence = {}
    missing_selected = missing_rejected = 0
    for k, rr in groups.items():
        model, role, kind, mode, stage, panel = k
        s = summaries[k]
        require(count(s['n_frames']) == EXPECTED[kind][0], 'T3 frame denominator differs')
        total = {field: 0 for field in STAGE_COUNTERS}
        for row in rr:
            fc = {field: count(row[field]) for field in STAGE_COUNTERS}
            mask_n = fc['total_mask_corners']
            require(mask_n <= 8 and fc['frames_nonempty'] == int(mask_n > 0)
                    and fc['frames_ge6'] == int(mask_n >= 6) and fc['eligible_ref_count'] <= 8
                    and fc['annotated_eligible_selected_ref_count'] + fc['rejected_ref_count'] == fc['eligible_ref_count']
                    and fc['annotated_eligible_selected_ref_count'] <= mask_n, 'Stage mask/reference counters differ')
            if stage == 'final' or mode == 'confidence_only':
                require(fc['frames_nonempty'] == fc['frames_ge6'], 'Final/confidence-only support must be >=6 or empty')
            for part, denominator in (('selected', 'annotated_eligible_selected_ref_count'), ('rejected', 'rejected_ref_count')):
                observed = fc[part + '_ref_observed_count']
                require(observed <= fc[denominator], 'Observed subset exceeds its reference count')
                moments(row, part + '_', observed)
                prev = 0
                for tau in (5, 10, 20):
                    good = fc[f'{part}_correct_le{tau}px']
                    require(prev <= good <= observed, 'Stage correctness count differs')
                    prev = good
            missing_selected += fc['annotated_eligible_selected_ref_count'] - fc['selected_ref_observed_count']
            missing_rejected += fc['rejected_ref_count'] - fc['rejected_ref_observed_count']
            signature = (fc['eligible_ref_count'], fc['selected_ref_observed_count'] + fc['rejected_ref_observed_count'],
                *[fc[f'selected_correct_le{t}px'] + fc[f'rejected_correct_le{t}px'] for t in (5, 10, 20)])
            pk = (model, role, kind, panel, row['id'])
            require(pk not in partition or partition[pk] == signature, 'Selected/rejected teacher partitions changed across stages')
            partition[pk] = signature
            sequence[pk + (mode, stage)] = (mask_n, fc['annotated_eligible_selected_ref_count'])
            for field, value in fc.items():
                total[field] += value
        require(total['eligible_ref_count'] == nref(kind, panel), 'Stage reference denominator differs')
        for field in STAGE_COUNTERS:
            same(total[field], s[field], 'T3 total ' + field)
        for tau in (5, 10, 20):
            sc, rc = total[f'selected_correct_le{tau}px'], total[f'rejected_correct_le{tau}px']
            for field, value in {f'selected_ref_precision_le{tau}px': ratio(sc, total['annotated_eligible_selected_ref_count']),
                    f'rejected_ref_PCK_le{tau}px_full_denominator': ratio(rc, total['rejected_ref_count']),
                    f'rejected_ref_precision_le{tau}px_observed_only': ratio(rc, total['rejected_ref_observed_count'])}.items():
                same(value, s[field], 'T3 conditional/full ratio ' + field)
        pooled(s, rr, 'selected_', total['selected_ref_observed_count'])
        pooled(s, rr, 'rejected_', total['rejected_ref_observed_count'])
    for pk in partition:
        for mode, order in (('main', ('s0', 's1', 's2', 's3', 'final')), ('noflip', ('s0', 's1', 's2', 'final'))):
            chain = [sequence[pk + (mode, stage)] for stage in order]
            require(all(b[0] <= a[0] and b[1] <= a[1] for a, b in zip(chain, chain[1:])), 'Cumulative stage counts increased')
    return dict(summary_rows=165, frame_rows=20328, stages_per_teacher=11, pooled_distributions_checked=330,
                missing_selected_references_across_all_stage_rows=missing_selected,
                missing_rejected_references_across_all_stage_rows=missing_rejected,
                repeated_stage_counts_are_not_unique_reference_totals=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe-contract', action='store_true')
    args = parser.parse_args()
    if args.describe_contract:
        print(json.dumps(contract(), indent=2), flush=True)
        return
    require(RESULT.resolve() == RESULT and not RESULT.exists(), 'Preserve prior QA; unsafe/existing result')
    io = Inputs()
    ids = gates(io)
    checks = {'components': components(io, ids), 'T3_stages': stages(io, ids)}
    io.unchanged()
    result = dict(passed=True, complete=True, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        population='PRIMARY372', frames=372, frames_and_references_per_kind=EXPECTED,
        manual_references={'WOOD': 330, 'GREEN': 600}, checks=checks, mismatch_count=0,
        upstream_equivalence_all_exact=True, source_sha256=io.bindings, verifier_source_sha256=io.code_sha,
        producer_source_sha256=PRODUCER_SHA, pooled_median_recomputed=False,
        median_limitation='Per-frame moments cannot recover the pooled median; aggregate median not independently verified',
        scalar_counter_and_pooled_moment_verification_only=True, error_SD_ddof=0,
        no_confidence_teacher_candidate_population_differs=True,
        retention_is_not_directly_comparable_across_decoder_populations=True,
        upstream_private_coordinate_files_opened_or_rehashed=0, raw_RGB_reads=0, raw_annotation_reads=0,
        neural_forwards=0, PnP_fits=0, bootstrap_draws=0, training_updates=0)
    with RESULT.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    io.unchanged()
    print('PRIMARY372_FILTER_STATISTICS_PASS', 35, 165, 4312, 20328, flush=True)


if __name__ == '__main__':
    main()
