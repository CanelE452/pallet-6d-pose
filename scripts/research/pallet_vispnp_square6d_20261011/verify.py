"""Independent numeric verification; never import experiment/model/F modules.

Public rows alone suffice for statistics, masks and paired intervals. Optional
private arguments additionally verify original-checkout preservation and every
bound existing RGB/cache/annotation/checkpoint byte hash, without decoding them.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOC = ROOT / '_docs/experiments/pallet_vispnp_square6d_20261011'
METHODS = ('BASE', 'N3_DIM_SYM', 'SUBPIX', 'N3_THEN_SUBPIX')
NUMERIC = dict(T_cm='translation_cm', R_deg='rotation_deg', ADDsym_m='ADDsym_m', IoU3D='IoU3D')
PRIMARY = 'N3_THEN_SUBPIX_VIS__minus__N3_THEN_SUBPIX'
FACES = dict(front=(0, 1, 2, 3), back=(4, 7, 6, 5), top=(0, 4, 5, 1),
             bottom=(3, 2, 6, 7), left=(0, 3, 7, 4), right=(1, 5, 6, 2))
_DRAW_WEIGHTS = {}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def quantile(values, fraction):
    x = sorted(values)
    position = (len(x) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return x[lo] + (x[hi] - x[lo]) * (position - lo)


def distribution(values):
    x = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not x:
        return dict(n=0, mean=None, variance=None, std=None, median=None, P90=None, max=None)
    mean = math.fsum(x) / len(x)
    variance = math.fsum((v - mean) ** 2 for v in x) / (len(x) - 1) if len(x) > 1 else None
    return dict(n=len(x), mean=mean, variance=variance,
                std=math.sqrt(variance) if variance is not None else None,
                median=quantile(x, .5), P90=quantile(x, .9), max=max(x))


def indicators(pose):
    available = bool(pose.get('available', False))
    return dict(available=available,
        confusion_rate=float(available and pose['rotation_deg'] > 45 and abs(pose['yaw_deg']) >= 60),
        success_rate=float(available and pose['translation_cm'] < 5 and pose['rotation_deg'] < 5))


def vector(records):
    result = {name: [float(r['pose'][key]) if r['pose'].get('available') and
                    r['pose'].get(key) is not None else None for r in records]
              for name, key in NUMERIC.items()}
    for name in ('available', 'confusion_rate', 'success_rate'):
        result[name] = [indicators(r['pose'])[name] for r in records]
    return result


def mean_vectors(per_seed):
    result = {}
    for name in per_seed[0]:
        triples = zip(*(v[name] for v in per_seed))
        result[name] = [all(x) if name == 'available' else
                        math.fsum(x) / len(x) if all(v is not None for v in x) else None
                        for x in triples]
    return result


def independent_visibility(points, support):
    usable = [bool(support[k]) and all(v is not None and math.isfinite(v) for v in points[k])
              and points[k] != [-1, -1] for k in range(8)]
    areas, facing = {}, {}
    for name, indices in FACES.items():
        area = (math.fsum(points[a][0] * points[b][1] - points[b][0] * points[a][1]
                         for a, b in zip(indices, indices[1:] + indices[:1])) / 2
                if all(usable[k] for k in indices) else None)
        areas[name] = area
        facing[name] = area > 0 if area is not None else None
    visible = [any(facing[n] is None or facing[n] for n, indices in FACES.items() if k in indices)
               for k in range(8)]
    remaining = [v and u for v, u in zip(visible, usable)]
    fallback = sum(remaining) < 4
    return dict(areas=areas, facing=facing, visible_mask=visible,
                hidden_count=sum(u and not v for u, v in zip(usable, visible)),
                fallback=fallback, effective_mask=usable if fallback else remaining)


class IndependentDraws:
    def __init__(self, universe_labels, selected_labels, level):
        labels = list(universe_labels) if level == 'frame' else sorted(set(universe_labels))
        assert len(set(labels)) == len(labels)
        group = {label: i for i, label in enumerate(labels)}
        self.group = [group[label] for label in selected_labels]
        self.units = len(labels)
        if self.units not in _DRAW_WEIGHTS:
            rng = np.random.default_rng(20260917)
            raw = np.vstack([
                rng.multinomial(self.units, np.ones(self.units) / self.units, size=100)
                for _ in range(100)])
            _DRAW_WEIGHTS[self.units] = raw.astype(np.float64)
        self.draws = _DRAW_WEIGHTS[self.units]
        self.sha256 = hashlib.sha256(self.draws.astype('<u2').tobytes()).hexdigest()
        self.level = level
        self.cache = {}

    def interval(self, values):
        cache_key = tuple(values)
        if cache_key in self.cache:
            return self.cache[cache_key]
        bins = [[] for _ in range(self.units)]
        for group, value in zip(self.group, values):
            if value is not None and math.isfinite(value):
                bins[group].append(float(value))
        if not any(bins):
            return None
        totals = np.asarray([math.fsum(x) for x in bins])
        counts = np.asarray([len(x) for x in bins])
        numerator, denominator = self.draws @ totals, self.draws @ counts
        samples = (numerator[denominator > 0] / denominator[denominator > 0]).tolist()
        result = [quantile(samples, .025), quantile(samples, .975)]
        self.cache[cache_key] = result
        return result


class Verifier:
    def __init__(self):
        self.comparisons = 0
        self.max_difference = Counter()

    def close(self, actual, expected, label):
        self.comparisons += 1
        if expected is None:
            assert actual is None, (label, actual, expected)
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected), label
            for i, (a, e) in enumerate(zip(actual, expected)):
                self.close(a, e, label + '/' + str(i))
        elif isinstance(expected, (bool, int)):
            assert actual == expected, (label, actual, expected)
        else:
            difference = abs(float(actual) - float(expected))
            field = label.split('/')[-1]
            self.max_difference[field] = max(self.max_difference[field], difference)
            assert difference <= 1e-7 + 1e-12 * abs(expected), (label, difference)

    def described(self, actual, values, label, draws=None):
        for key, value in distribution(values).items():
            self.close(actual[key], value, label + '/' + key)
        if 'CI95' in actual:
            self.close(actual['CI95'], draws.interval(values) if draws is not None else None, label + '/CI95')
        if 'variance_ddof' in actual:
            assert actual['variance_ddof'] == 1

    def metadata(self, actual, draws, label):
        assert actual['units'] == draws.units, (label, actual['units'], draws.units)
        assert actual['draws_sha256_uint16_le'] == draws.sha256, label
        assert actual['resamples'] == 10000 and actual['seed'] == 20260917
        assert actual['multiplicity_adjusted'] is False

    def corner(self, actual, records, label):
        corners = [r['corner'] for r in records if r.get('corner', {}).get('evaluable', False)]
        errors = [v for c in corners for v in c['errors']]
        observed = [v for c in corners for v in c['observed_errors']]
        self.close(actual['evaluable_frames'], len(corners), label + '/frames')
        self.close(actual['reference_corners'], len(errors), label + '/references')
        self.close(actual['observed_corners'], len(observed), label + '/observed')
        self.described(actual['errors'], errors, label + '/errors')
        self.described(actual['observed_errors'], observed, label + '/observed_errors')
        for t in (5, 10, 20):
            self.close(actual['PCK'][str(t)], math.fsum(v <= t for v in errors) / len(errors)
                       if errors else None, label + '/PCK' + str(t))
        self.close(actual['gross20'], math.fsum(v > 20 for v in errors) / len(errors)
                   if errors else None, label + '/gross20')
        if corners:
            self.close(actual['E_sym'], math.fsum(c['E_sym'] for c in corners) / len(corners), label + '/E_sym')

    def summary(self, packet, records, values, draws, label, seed_packets=None):
        assert packet['frames'] == len(records)
        available = sum(values['available'])
        assert packet['available'] == available and packet['unavailable'] == len(records) - available
        self.close(packet['coverage'], available / len(records), label + '/coverage')
        for name in NUMERIC:
            self.described(packet['metrics'][name], values[name], label + '/' + name, draws)
        for name in ('confusion_rate', 'success_rate'):
            r, x = packet['rates'][name], values[name]
            self.close(r['rate'], math.fsum(x) / len(x), label + '/' + name)
            self.close(r['positive_count_sum'], math.fsum(x), label + '/' + name + '/count')
            assert r['denominator'] == len(x) and r['binary_before_seed_mean'] is True
            self.close(r['CI95'], draws.interval(x), label + '/' + name + '/CI95')
        if seed_packets is None:
            self.corner(packet['corner'], records, label + '/corner')
        else:
            for t in (5, 10, 20):
                self.close(packet['corner']['arithmetic_mean_seed_PCK'][str(t)],
                    math.fsum(p['corner']['PCK'][str(t)] for p in seed_packets) / 3,
                    label + '/corner/PCK' + str(t))
            self.close(packet['corner']['arithmetic_mean_seed_gross20'],
                math.fsum(p['corner']['gross20'] for p in seed_packets) / 3, label + '/corner/gross20')

    def contrast(self, packet, left, right, draws, label, secondary=None):
        coverage = all(v for x in left + right for v in x['available'])
        assert packet['coverage_complete'] == coverage
        self.metadata(packet['bootstrap'], draws, label + '/bootstrap')
        for name in (*NUMERIC, 'confusion_rate', 'success_rate'):
            deltas = [[a - b if a is not None and b is not None else None
                       for a, b in zip(l[name], r[name])] for l, r in zip(left, right)]
            mean = [math.fsum(x) / len(x) if all(v is not None for v in x) else None
                    for x in zip(*deltas)]
            per_seed = [distribution(x)['mean'] for x in deltas]
            lower = name not in ('success_rate', 'IoU3D')
            actual = packet[name] if name.endswith('_rate') else packet['metrics'][name]
            self.close(actual['delta'], distribution(mean)['mean'], label + '/' + name + '/delta')
            self.close(actual['CI95'], draws.interval(mean), label + '/' + name + '/CI95')
            self.close(actual['per_seed_delta'], per_seed, label + '/' + name + '/per_seed')
            assert actual['paired_frames'] == sum(v is not None for v in mean)
            assert actual['improved_seeds'] == sum(v is not None and (v < 0 if lower else v > 0) for v in per_seed)
            assert actual['coverage_complete'] == coverage and actual['lower_is_better'] == lower
            if secondary is not None:
                second = actual['scenario_cluster_secondary']
                self.close(second['delta'], distribution(mean)['mean'], label + '/secondary/' + name)
                self.close(second['CI95'], secondary.interval(mean), label + '/secondary/' + name + '/CI95')
                self.metadata(second['bootstrap'], secondary, label + '/secondary/bootstrap')

    def aggregate(self, all_rows, vis_rows, metrics, paired, failures):
        base = {(int(r['seed']), r['method'].removesuffix('_VIS'), r['id']): r for r in all_rows}
        changed = {(int(r['seed']), r['method'].removesuffix('_VIS'), r['id']): r for r in vis_rows}
        assert len(base) == len(all_rows) and len(changed) == len(vis_rows) and set(base) == set(changed)
        ids = [r['id'] for r in all_rows if int(r['seed']) == 1 and r['method'] == 'BASE']
        assert len(set(ids)) == len(ids)
        assert set(base) == {(s, m, i) for s in (1, 2, 3) for m in METHODS for i in ids}
        scopes = metrics.get('scope_frame_ids', {'ALL': ids})
        if set(scopes) != set(metrics['seed_mean']):
            scopes = {'ALL': ids}
            for name in metrics['seed_mean']:
                if name in ('clean', 'moderate', 'severe'):
                    scopes[name] = [i for i in ids if base[(1, 'BASE', i)]['grade'] == name]
                elif name in ('plastic', 'wood'):
                    scopes[name] = [i for i in ids if (base[(1, 'BASE', i)]['fixed_metadata']['dimensions_pnp_WH_D_m'][0] == 1.1) == (name == 'plastic')]
                elif name != 'ALL':
                    raise AssertionError('Published ordered IDs required for scope ' + name)
        assert scopes['ALL'] == ids and metrics['unique_frames'] == len(ids)
        universe_sessions = [base[(1, 'BASE', i)]['session'] for i in ids]
        for key in base:
            assert base[key]['corner'] == changed[key]['corner'], ('corner invariance', key)
        for scope, selected in scopes.items():
            assert selected and len(selected) == len(set(selected)) and set(selected) <= set(ids)
            if scope in ('clean', 'moderate', 'severe'):
                assert selected == [i for i in ids if base[(1, 'BASE', i)]['grade'] == scope]
            elif scope in ('plastic', 'wood'):
                width = 1.1 if scope == 'plastic' else .8
                assert selected == [i for i in ids if abs(base[(1, 'BASE', i)]['fixed_metadata']['dimensions_pnp_WH_D_m'][0] - width) < 1e-9]
            elif scope in ('G38', 'P0', 'TEX'):
                assert selected == [i for i in ids if base[(1, 'BASE', i)]['source'] == scope]
            elif scope in ('C1', 'C2', 'C4'):
                assert selected == [i for i in ids if base[(1, 'BASE', i)]['canonical_symmetry_order'] == int(scope[1:])]
            cluster_labels = [base[(1, 'BASE', i)]['session'] for i in selected]
            frame_level = paired['bootstrap'][scope]['primary']['level'] == 'frame'
            draws = IndependentDraws(ids if frame_level else universe_sessions,
                                     selected if frame_level else cluster_labels,
                                     'frame' if frame_level else 'cluster')
            secondary = IndependentDraws(universe_sessions, cluster_labels, 'scenario_cluster') if frame_level else None
            self.metadata(paired['bootstrap'][scope]['primary'], draws, scope + '/draws')
            vectors, grouped = {}, {}
            for s in (1, 2, 3):
                for m in METHODS:
                    for index, suffix in ((base, ''), (changed, '_VIS')):
                        name = m + suffix
                        rr = [index[(s, m, i)] for i in selected]
                        vv = vector(rr)
                        vectors[(s, name)], grouped[(s, name)] = vv, rr
                        self.summary(metrics['by_seed'][str(s)][scope][name], rr, vv, draws,
                                     str(s) + '/' + scope + '/' + name)
                    self.contrast(paired['by_seed'][str(s)][scope][m + '_VIS__minus__' + m],
                        [vectors[(s, m + '_VIS')]], [vectors[(s, m)]], draws,
                        str(s) + '/' + scope + '/' + m, secondary)
            for m in METHODS:
                for suffix in ('', '_VIS'):
                    name = m + suffix
                    vv = mean_vectors([vectors[(s, name)] for s in (1, 2, 3)])
                    self.summary(metrics['seed_mean'][scope][name], grouped[(1, name)], vv, draws,
                        'seed_mean/' + scope + '/' + name,
                        [metrics['by_seed'][str(s)][scope][name] for s in (1, 2, 3)])
                self.contrast(paired['seed_mean'][scope][m + '_VIS__minus__' + m],
                    [vectors[(s, m + '_VIS')] for s in (1, 2, 3)],
                    [vectors[(s, m)] for s in (1, 2, 3)], draws, 'seed_mean/' + scope + '/' + m, secondary)
        for s in (1, 2, 3):
            for m in METHODS:
                packet = failures['per_seed'][str(s)][m]
                damage, recovery, switches, missing, fallback, hidden = [], [], [], [], [], Counter()
                for i in ids:
                    b, v = base[(s, m, i)], changed[(s, m, i)]
                    bi, vi = indicators(b['pose']), indicators(v['pose'])
                    if bi['success_rate'] and not vi['success_rate']: damage.append(i)
                    if not bi['success_rate'] and vi['success_rate']: recovery.append(i)
                    if b.get('final_hypothesis') != v.get('final_hypothesis'): switches.append(i)
                    if not vi['available']: missing.append(i)
                    if v.get('visibility', {}).get('fallback'): fallback.append(i)
                    hidden[str(v.get('visibility', {}).get('hidden_count', 0))] += 1
                assert packet['success_to_failure_ids'] == damage and packet['success_to_failure'] == len(damage)
                assert packet['failure_to_success_ids'] == recovery and packet['failure_to_success'] == len(recovery)
                assert packet['hypothesis_changes'] == len(switches)
                assert [r['id'] for r in packet['hypothesis_transition_records']] == switches
                for transition in packet['hypothesis_transition_records']:
                    b, v = base[(s, m, transition['id'])], changed[(s, m, transition['id'])]
                    assert transition['ALL'] == b.get('final_hypothesis') and transition['VIS'] == v.get('final_hypothesis')
                    for name, key in (('delta_T_cm', 'translation_cm'), ('delta_R_deg', 'rotation_deg')):
                        delta = v['pose'][key] - b['pose'][key] if b['pose'].get('available') and v['pose'].get('available') else None
                        self.close(transition[name], delta, 'transition/' + name)
                assert packet['unavailable_VIS_ids'] == missing
                assert packet['fallback_lt4_ids'] == fallback and packet['fallback_lt4'] == len(fallback)
                assert packet['hidden_count_distribution'] == dict(hidden)
                hidden_packets = metrics.get('diagnostics', {}).get('hidden_count_by_seed', {}).get(str(s), {}).get(m, {})
                for count, hidden_packet in hidden_packets.items():
                    local_ids = [i for i in ids if str(changed[(s, m, i)]['visibility']['hidden_count']) == count]
                    assert hidden_packet['frame_ids'] == local_ids and hidden_packet['frames'] == len(local_ids)
                    local_labels = [base[(1, 'BASE', i)]['session'] for i in local_ids]
                    local_draws = IndependentDraws(ids if frame_level else universe_sessions,
                        local_ids if frame_level else local_labels, 'frame' if frame_level else 'cluster')
                    self.metadata(hidden_packet['bootstrap'], local_draws, 'hidden/' + str(s) + '/' + m + '/' + count)
                    for index, name in ((base, 'ALL'), (changed, 'VIS')):
                        local_rows = [index[(s, m, i)] for i in local_ids]
                        self.summary(hidden_packet[name], local_rows, vector(local_rows), local_draws,
                            'hidden/' + str(s) + '/' + m + '/' + count + '/' + name)
                    d = hidden_packet['damage']
                    expected_damage = [i for i in damage if i in set(local_ids)]
                    expected_recovery = [i for i in recovery if i in set(local_ids)]
                    assert d['success_to_failure_ids'] == expected_damage and d['success_to_failure'] == len(expected_damage)
                    assert d['failure_to_success_ids'] == expected_recovery and d['failure_to_success'] == len(expected_recovery)
                    assert d['hypothesis_changes'] == sum(i in set(local_ids) for i in switches)
        assert paired['primary'] == paired['seed_mean']['ALL'][PRIMARY]
        return dict(status='PASS', frames=len(ids), ALL_rows=len(all_rows), VIS_rows=len(vis_rows),
                    scopes={k: len(v) for k, v in scopes.items()},
                    primary_draws_sha256=paired['bootstrap']['ALL']['primary']['draws_sha256_uint16_le'])


def source_contract(doc):
    directory = ROOT / 'scripts/research/pallet_vispnp_square6d_20261011'
    signatures = {'visibility.py': ('visibility', ['points9', 'support8']),
                  'adapter.py': ('correspondence_mask', ['points9', 'K', 'mask8'])}
    for filename, (name, arguments) in signatures.items():
        tree = ast.parse((directory / filename).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert [a.arg for a in function.args.args] == arguments
        assert not function.args.kwarg and not function.args.vararg
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert not (isinstance(node.func, ast.Name) and node.func.id in ('open', 'eval', 'exec', '__import__'))
                assert not (isinstance(node.func, ast.Attribute) and node.func.attr in
                            ('read_text', 'read_bytes', 'load', 'loads', 'imread', 'VideoCapture'))
    adapter = ast.parse((directory / 'adapter.py').read_text())
    patched = [n for n in ast.walk(adapter) if isinstance(n, ast.Call) and
               isinstance(n.func, ast.Name) and n.func.id == 'setattr']
    assert len(patched) == 2  # install and restore the same two functions
    text = (directory / 'adapter.py').read_text()
    assert "('solvePnP', 'solvePnPRefineLM')" in text
    assert 'return original[name](*args, **kwargs)' in text
    locks = []
    if (doc / 'SOURCE_LOCK.json').exists():
        packet = read(doc / 'SOURCE_LOCK.json')
        bindings = packet.get('new_core', packet.get('source_bindings', packet.get('bindings', [])))
        if isinstance(bindings, dict): bindings = [dict(path=k, sha256=v) for k, v in bindings.items()]
        for binding in bindings:
            path = ROOT / binding['path']
            assert sha(path) == binding['sha256'], ('new source binding', binding['path'])
            locks.append(binding['path'])
    return dict(status='PASS', GT_free_public_interfaces=True, file_IO_in_inference_interfaces=False,
                selector_score_projection_unchanged=True, bound_new_source_files=locks,
                limitation='AST and exact hashes verify code contracts, not complete OS access tracing')


def masks_and_rows(doc, base_rows):
    seal = read(doc / 'COORDINATES_SEAL.json')
    path = doc / 'REAL_COORDINATES_MASKS.jsonl.gz'
    assert sha(path) == seal['sha256']
    baseline = {(r['seed'], r['method'], r['id']): r for r in base_rows}
    sealed = rows(path)
    assert len(sealed) == 3828
    for r in sealed:
        b = baseline[(r['seed'], r['method'], r['id'])]
        assert r['qFinal'] == b['qFinal'] and r['prediction_support'] == b['prediction_support']
        assert r['fixed_metadata'] == b['fixed_metadata']
        expected = independent_visibility(r['qFinal'], r['prediction_support'][:8])
        for key, value in expected.items(): assert r['visibility'][key] == value, (r['id'], key)
    count = 0
    if (doc / 'PREDICTIONS.jsonl.gz').exists():
        for r in rows(doc / 'PREDICTIONS.jsonl.gz'):
            b = baseline[(r['seed'], r['method'].removesuffix('_VIS'), r['id'])]
            for field in ('q0', 'qN', 'qS', 'qFinal', 'prediction_support', 'canonical_observed', 'corner', 'fixed_metadata'):
                assert r[field] == b[field], (r['id'], field)
            v = independent_visibility(r['qFinal'], r['prediction_support'][:8])
            for key, value in v.items(): assert r['visibility'][key] == value
            audit = r['solver_mask_audit']
            indices = [k for k, keep in enumerate(v['effective_mask']) if keep]
            assert audit['canonical_indices'] == indices and audit['mask8'] == v['effective_mask']
            assert audit['original_points_preserved'] and audit['camera_preserved'] and audit['projectPoints_unpatched']
            for function in ('solvePnP', 'solvePnPRefineLM'):
                calls = [c for c in audit['calls'] if c['function'] == function]
                assert audit[function] == len(calls) == r['PnP_counts'][function]
                assert audit['filtered_' + function] == sum(c['filtered'] for c in calls)
                for call in calls:
                    assert call['canonical_indices'] == indices and call['input_correspondences'] == 8
                    assert call['passed_correspondences'] == len(indices)
                    assert call['original_image_points_exact'] and call['original_camera_exact']
            count += 1
        assert count == 3828
    return dict(status='PASS', sealed_masks=3828, VIS_rows_verified=count,
                original_coordinates_corner_metrics_unchanged=True, independent_VIS_RULE_V1=True)


def synthetic_masks_and_calls(doc):
    seal_path = doc / 'SYNTH_COORDINATE_SEAL.json'
    seal = read(seal_path)
    assert seal['status'] == 'SEALED_BEFORE_POSE_REFERENCES'
    assert seal['pose_reference_fields_accessed_before_seal'] is False and seal['human_visibility_used'] is False
    assert sha(doc / 'METHOD_LOCK.json') == seal['method_lock_sha256']
    assert sha(doc / 'A0.json') == seal['A0_sha256']
    assert sha(doc / 'SYNTH_COORDINATES_AND_MASKS.jsonl.gz') == seal['coordinates_sha256']
    for binding in seal['code']:
        assert sha(ROOT / binding['path']) == binding['sha256']
    seal_sha256 = sha(seal_path)
    sealed = rows(doc / 'SYNTH_COORDINATES_AND_MASKS.jsonl.gz')
    assert len(sealed) == 1985 * 4 * 3 == seal['rows']
    index = {(r['seed'], r['method'], r['id']): r for r in sealed}
    assert len(index) == len(sealed)
    totals = Counter()
    for filename, variant in (('SYNTH_ALL.jsonl.gz', 'ALL'), ('SYNTH_PREDICTIONS.jsonl.gz', 'VIS')):
        saved = rows(doc / filename)
        assert len(saved) == len(sealed)
        for r in saved:
            original = index[(r['seed'], r['method'].removesuffix('_VIS'), r['id'])]
            for key in ('q0', 'qN', 'qS', 'qFinal', 'prediction_support', 'fixed_metadata', 'visibility', 'correction'):
                assert r[key] == original[key], (r['id'], key)
            assert r['evaluation_reference_used_in_inference'] is False
            assert r['reference_seal_sha256'] == seal_sha256
            expected = independent_visibility(r['qFinal'], r['prediction_support'][:8])
            for key, value in expected.items(): assert r['visibility'][key] == value
            mask = [True] * 8 if variant == 'ALL' else expected['effective_mask']
            indices = [k for k, keep in enumerate(mask) if keep]
            audit = r['PnP_counts']
            assert audit['mask8'] == mask and audit['canonical_indices'] == indices
            assert audit['original_points_preserved'] and audit['camera_preserved'] and audit['projectPoints_unpatched']
            for function in ('solvePnP', 'solvePnPRefineLM'):
                calls = [c for c in audit['calls'] if c['function'] == function]
                assert audit[function] == len(calls)
                assert audit['filtered_' + function] == sum(c['filtered'] for c in calls)
                totals[function] += len(calls)
                for call in calls:
                    assert call['canonical_indices'] == indices and call['passed_correspondences'] == len(indices)
                    assert call['input_correspondences'] == 8
                    assert call['original_image_points_exact'] and call['original_camera_exact']
                    assert call['filtered'] == (len(indices) != 8)
            assert r['actual_pose']['available'] == r['pose']['available']
            assert r['final_hypothesis'] == r['actual_pose'].get('selected_hypothesis')
    execution = read(doc / 'SYNTH_EXECUTION.json')
    assert execution['status'] == 'COMPLETE' and execution['actual_F_calls'] == len(sealed) * 2
    assert execution['PnP_counts'] == dict(totals)
    assert sha(doc / 'SYNTH_ALL.jsonl.gz') == execution['ALL_sha256']
    assert sha(doc / 'SYNTH_PREDICTIONS.jsonl.gz') == execution['VIS_sha256']
    return dict(status='PASS', sealed_rows=len(sealed), actual_F_calls_recorded=len(sealed) * 2,
                solver_correspondences_only=True, solver_counts=dict(totals), independent_F_calls=0)


def preservation(doc, source, private):
    if source is None or private is None:
        return dict(status='NOT_RUN_PUBLIC_ONLY', reason='Optional existing private inputs not supplied')
    start = read(private / 'START.json')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args])
    checks = dict(head=git('rev-parse', 'HEAD').decode().strip() == start['head'],
        branch=git('branch', '--show-current').decode().strip() == start['branch'],
        status_bytes=git('status', '--porcelain', '-z') == (private / 'original_status.bin').read_bytes(),
        tracked_diff_bytes=git('diff', '--binary', 'HEAD') == (private / 'original_tracked.diff').read_bytes(),
        modified_tracked_SHA=all(sha(source / p) == value for p, value in start['changed_tracked_sha256'].items()))
    assert all(checks.values()), checks
    audit = read(ROOT / '_docs/experiments/pallet_n3_subpix_final_20261010/INPUT_AUDIT.json')
    count = 0
    for model in audit['models']:
        for key in ('checkpoint', 'prediction'):
            binding = model[key]
            assert sha(source / binding['path']) == binding['sha256']
            count += 1
    for frame in audit['inputs']:
        for key in ('image', 'detector_feature_cache', 'annotation'):
            binding = frame[key]
            assert sha(source / binding['path']) == binding['sha256']
            count += 1
    for binding in read(doc / 'METHOD_LOCK.json')['source_bindings']:
        assert sha(source / binding['path']) == binding['sha256']
        count += 1
    if (doc / 'SYNTH_COORDINATE_SEAL.json').exists():
        packet = read(doc / 'SYNTH_COORDINATE_SEAL.json')
        for binding in packet.get('input_bindings', []):
            assert sha(source / binding['path']) == binding['sha256']
            count += 1
        manifest = read(source / 'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json')
        heldout = [r for r in manifest['records'] if r['partition'] == 'heldout']
        assert len(heldout) == 1985
        for record in heldout:
            path = Path(record['image'])
            if not path.is_absolute(): path = source / path
            assert sha(path) == record['image_sha256']
            count += 1
        detection = read(source / '_docs/experiments/pallet_dim_conditioned_p_v1/SYNTH_DETECTION_AUDIT.json')
        for binding in detection['files']:
            assert sha(source / binding['path']) == binding['sha256']
            count += 1
    return dict(status='PASS', checks=checks, existing_input_file_hashes_checked=count,
                RGB_decodes=0, model_loads=0, F_calls=0)


def privacy(doc):
    code = ROOT / 'scripts/research/pallet_vispnp_square6d_20261011'
    files = [p for directory in (doc, code) for p in directory.rglob('*') if p.is_file()]
    for p in files:
        assert p.suffix.lower() not in ('.pt', '.npz', '.npy', '.jpg', '.jpeg', '.bmp', '.pyc'), p.name
        if p.suffix == '.png': continue
        if p.suffix == '.gz':
            with gzip.open(p, 'rt', encoding='utf-8') as stream: text = stream.read()
        else: text = p.read_text()
        assert not re.search(r'/' + 'home' + r'/[^\s/]+/', text), p.name
        assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}', text), p.name
    return dict(status='PASS', files_checked=len(files),
                no_private_image_or_checkpoint_file_extensions=True, no_personal_absolute_paths=True,
                raw_RGB_decodes=0, limitation='PNG visual inspection belongs to separate publication review')


def independent_verdict(packet, phase):
    p = packet['primary']
    confusion, success = p['confusion_rate'], p['success_rate']
    if not p['coverage_complete']:
        verdict = 'NOT_ESTIMABLE'
    elif confusion['CI95'][0] > 0 or success['CI95'][1] < 0:
        verdict = 'WORSENED'
    elif ((confusion['CI95'][1] < 0 and confusion['improved_seeds'] >= 2) or
          (success['CI95'][0] > 0 and success['improved_seeds'] >= 2)):
        verdict = 'SUPPORTED'
    else:
        verdict = 'UNRESOLVED'
    return dict(phase=phase, verdict=verdict,
                continue_to_A2=verdict not in ('WORSENED', 'NOT_ESTIMABLE') if phase == 'A1' else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doc', type=Path, default=DEFAULT_DOC)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--private-dir', type=Path)
    args = parser.parse_args()
    doc = args.doc
    output = args.output or doc / 'VERIFICATION.json'
    assert not output.exists(), 'Use a fresh verifier output filename'
    verifier = Verifier()
    base = [r for r in rows(ROOT / '_docs/experiments/pallet_feature_gradient_joint_20261010/PREDICTIONS.jsonl.gz')
            if r['method'] in METHODS]
    a0 = read(doc / 'A0.json')
    assert a0['status'] == 'PASS' and a0['actual_F_calls'] == 1276 and a0['max_pose_metric_absolute_delta'] <= 1e-7
    packet = dict(status='PASS', A0_recorded='PASS', masks=masks_and_rows(doc, base),
                  source_contract=source_contract(doc), populations={})
    assert (doc / 'SYNTH_METRICS.json').exists(), 'Synthetic prerequisite results are not ready'
    synth_paired = read(doc / 'SYNTH_PAIRED.json')
    packet['populations']['SYNTH_HELDOUT'] = verifier.aggregate(rows(doc / 'SYNTH_ALL.jsonl.gz'),
        rows(doc / 'SYNTH_PREDICTIONS.jsonl.gz'), read(doc / 'SYNTH_METRICS.json'),
        synth_paired, read(doc / 'SYNTH_FAILURES.json'))
    expected_gate = independent_verdict(synth_paired, 'A1')
    recorded_gate = read(doc / 'SYNTH_VERDICT.json')
    assert recorded_gate['verdict'] == expected_gate['verdict']
    assert recorded_gate['continue_to_A2'] == expected_gate['continue_to_A2']
    if 'continue_real' in recorded_gate:
        assert recorded_gate['continue_real'] == expected_gate['continue_to_A2']
    packet['synthetic_gate_independent'] = expected_gate
    packet['synthetic_masks_and_calls'] = synthetic_masks_and_calls(doc)
    if (doc / 'METRICS.json').exists():
        assert expected_gate['continue_to_A2'] is True
        packet['populations']['REAL_DEV'] = verifier.aggregate(base, rows(doc / 'PREDICTIONS.jsonl.gz'),
            read(doc / 'METRICS.json'), read(doc / 'PAIRED.json'), read(doc / 'FAILURES.json'))
        expected_real = independent_verdict(read(doc / 'PAIRED.json'), 'A2')
        assert read(doc / 'VERDICT.json')['verdict'] == expected_real['verdict']
        packet['real_verdict_independent'] = expected_real
    elif not expected_gate['continue_to_A2']:
        assert not (doc / 'REAL_STARTED.json').exists() and not (doc / 'PREDICTIONS.jsonl.gz').exists()
        packet['real_execution'] = 'SKIPPED_BY_PREREGISTERED_SYNTHETIC_GATE'
    else:
        raise AssertionError('The permitted real-data experiment and statistics must finish before final verification')
    packet['original_preservation_and_private_inputs'] = preservation(doc, args.source_root, args.private_dir)
    packet['privacy'] = privacy(doc)
    packet.update(independent_numeric_comparisons=verifier.comparisons,
        max_abs_difference=dict(verifier.max_difference), independent_F_calls=0,
        independent_model_calls=0, independent_training_updates=0,
        implementation='math.fsum sample moments and scalar sorted linear quantiles; independent fixed RNG draws',
        source_sha256=sha(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(packet, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('INDEPENDENT_VERIFICATION_PASS', verifier.comparisons, flush=True)


if __name__ == '__main__':
    main()
