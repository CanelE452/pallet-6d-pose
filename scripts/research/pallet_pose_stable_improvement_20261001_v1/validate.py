"""Independent artifact/publication audit; no training, prediction or Git writes.

Run only after complete RESULTS, report and figures exist. This audit proves
artifact consistency within its stated scope; a PASS never changes the
experiment's stability verdict or means the active research goal was achieved.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import gc
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

import cv2
import numpy as np
import torch

from . import common as C


PUBLIC_BASE = 'babef3568a02118f4e9fc5ab4fd07ad24aff2757'
CHECKOUT = Path('/tmp/pallet-pose-github-review-20260930')
KEYS = ('translation_cm', 'rotation_deg')
SUMMARY_FIGURES = ('01_training_composition.png', '02_pose_medians.png', '03_pose_P90.png',
                   '04_all_seed_paired_changes.png', '05_outcome_directions.png',
                   '06_recording_changes.png', '07_learning_curves.png')
SUPPORT = (
    'scripts/research/pallet_posefix_heatmap_diversity_v1/data.py',
    '_docs/experiments/pallet_posefix_limited_adaptation_pilot_v1/RESULTS_KO.md',
    '_docs/experiments/pallet_posefix_limited_adaptation_pilot_v1/TRAINABLE_CONTRACTS.json',
    '_docs/experiments/pallet_posefix_limited_adaptation_pilot_v1/FIT_FULL.json',
    '_docs/experiments/pallet_posefix_limited_adaptation_pilot_v1/DECISION.json',
    '_docs/experiments/pallet_posefix_limited_adaptation_pilot_v1/REAL_RESULTS.json',
    '_docs/experiments/pallet_posefix_limited_adaptation_pilot_v1/ERROR_BAND_RESULTS.json',
    '_docs/experiments/pallet_posefix_limited_adaptation_pilot_v1/PRESERVATION_TRANSITIONS.json',
    '_docs/experiments/pallet_posefix_limited_adaptation_pilot_v1/SOURCE_RESULTS.json',
    '_docs/experiments/pallet_posefix_limited_adaptation_pilot_v1/AUDIT.json',
    '_docs/experiments/pallet_posefix_structured_error_v1/RESULTS_KO.md',
)


def write_owned(name, value):
    assert name in ('PUBLICATION_VALIDATION.json', 'PUBLICATION_MANIFEST.json', 'INFERENCE_RECEIPTS.json')
    destination = C.DOC / name
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(C.D.M.clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination)


class Verifier:
    def __init__(self):
        self.checked = {}

    def verify(self, binding):
        path = C.ROOT / binding['path']
        actual = self.checked.get(binding['path'])
        if actual is None:
            actual = C.bind(path)
            self.checked[binding['path']] = actual
        assert actual['sha256'] == binding['sha256'], ('SHA mismatch', binding['path'])
        if 'bytes' in binding:
            assert actual['bytes'] == binding['bytes'], binding['path']
        return actual

    def walk(self, value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                self.verify(value)
            else:
                for child in value.values():
                    self.walk(child)
        elif isinstance(value, list):
            for child in value:
                self.walk(child)


def array_sha(value):
    array = np.ascontiguousarray(value)
    header = str(array.shape).encode() + str(array.dtype).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def audit_training(verify):
    protocol = C.protocol()
    original = C.read(C.DOC / 'GOAL_PROTOCOL.json')
    amendment = C.read(C.DOC / 'PROTOCOL_AMENDMENT_01.json')
    pre = C.read(C.DOC / 'PRETRAIN_TESTS.json')
    inputs = C.read(C.DOC / 'INPUTS.json')
    complete = C.read(C.DOC / 'TRAINING_COMPLETE.json')
    assert original['sample_size'] == 252 and protocol['sample_size'] == inputs['paired_count'] == 251
    assert pre['PASS'] and inputs['GT_input'] is False
    verify.walk(pre)
    verify.walk(inputs)
    verify.walk(complete)
    for filename in ('PROTOCOL_SHA.json', 'EFFECTIVE_PROTOCOL_SHA.json', 'TRAIN_CODE_LOCK.json', 'TRAIN_TRANSITIVE_CODE_LOCK.json'):
        verify.walk(C.read(C.DOC / filename))
    assert complete['fits'] == 6 and complete['optimizer_updates'] == 1800
    manifests = C.read(C.ROOT / 'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json')['items']
    eval_hashes = {C.bind(C.ROOT / r['image_path'])['sha256'] for r in manifests}
    assert len(manifests) == len(eval_hashes) == 319
    pairs = {}
    for arm in C.ARMS:
        records = inputs['arms'][arm]
        assert len(records) == len({r['id'] for r in records}) == 251
        assert not ({r['image']['sha256'] for r in records} & eval_hashes)
        assert not any(r['id'] == original['exclude_original_DAY_id'] for r in records)
        pairs[arm] = []
        for r in records:
            data = torch.load(C.ROOT / r['pair']['path'], map_location='cpu', weights_only=False)['pair']
            clean, occ = data['CLEAN'], data['OCC']
            np.testing.assert_array_equal(clean['target_valid'], occ['target_valid'])
            assert not occ['target_valid'][8]
            pairs[arm].append({k: occ[k] for k in ('id', 'points', 'target', 'target_valid')})
    source = np.load(C.ROOT / inputs['source_orders']['path'])
    assert source['source_rows'].shape == (300, 8) and len(source['held_rows']) == 256
    assert not np.intersect1d(source['source_rows'], source['held_rows']).size
    source_data = C.L.L.SourceData()
    assert np.isin(source['source_rows'], source_data.train_rows).all()
    source_partition_verified = int(source['source_rows'].size)
    del source_data
    fit_rows, trace_rows = [], {}
    for seed in C.SEEDS:
        expected_order = np.random.default_rng(6400 + seed).integers(0, 251, size=(300, 8))
        order = torch.load(C.ROOT / inputs['real_orders'][str(seed)]['path'], map_location='cpu', weights_only=True).numpy()
        np.testing.assert_array_equal(order, expected_order)
        for arm in C.ARMS:
            name = f'{arm}_s{seed}'
            fit = C.read(C.DOC / f'FIT_{name}.json')
            verify.walk(fit)
            assert fit['complete'] and fit['updates'] == 300
            assert fit['arm'] == arm and fit['seed'] == seed
            assert fit['real_exposures'] == fit['source_exposures'] == 2400
            assert fit['initial_state_sha'] == pre['initial_model_state_sha']
            assert all(fit[k] for k in ('frozen_parameters_buffers_unchanged', 'BN_all_unchanged',
                                         'trainable_changed', 'optimizer_trainables_only', 'final_only', 'no_real_evaluation_GT_input'))
            ck = torch.load(C.ROOT / fit['checkpoint']['path'], map_location='cpu', weights_only=False)
            assert ck['complete'] and ck['step'] == 300 and ck['arm'] == arm and ck['seed'] == seed
            assert ck['protocol_sha256'] == pre['protocol']['sha256']
            assert ck['final_state_sha'] == fit['final_state_sha'] == C.L.state_hash(ck['state'])
            assert not any(k.startswith('model.') for k in ck['state'])
            del ck
            gc.collect()
            trace = [json.loads(line) for line in (C.ROOT / fit['trace']['path']).read_text().splitlines()]
            assert [r['step'] for r in trace] == list(range(1, 301))
            for index, r in enumerate(trace):
                assert r['real_indices'] == expected_order[index].tolist()
                assert r['source_rows'] == source['source_rows'][index].tolist()
                selected = [pairs[arm][i] for i in r['real_indices']]
                assert r['real_ids'] == [item['id'] for item in selected]
                for key, field in [('points', 'real_points_sha'), ('target', 'real_target_sha'), ('target_valid', 'real_mask_sha')]:
                    assert r[field] == array_sha(np.stack([item[key] for item in selected]))
                assert r['BN_sha'] == pre['BN_sha']
                assert np.isfinite(r['update_norm']) and r['update_norm'] > 0
                assert all(np.isfinite(v) for values in r['loss'].values() for v in values.values())
            trace_rows[name] = trace
            fit_rows.append(dict(model=name, updates=300, canonical_state_verified=True,
                                 checkpoint=fit['checkpoint'], trace=fit['trace'], seconds=fit['seconds']))
        left, right = (trace_rows[f'{arm}_s{seed}'] for arm in C.ARMS)
        for a, b in zip(left, right):
            for key in ('step', 'real_indices', 'source_rows', 'source_ids', 'source_corrupted_points_sha', 'BN_sha'):
                assert a[key] == b[key], (seed, a['step'], key)
    return dict(PASS=True, fits=fit_rows, total_optimizer_updates=1800,
                source_train_partition_entries_per_fit=source_partition_verified,
                paired_source_and_real_index_steps=900, reconstructed_real_tensor_trace_steps=1800,
                all_319_eval_image_hashes_rechecked=True, train_eval_image_overlap=0,
                amendment_preserved=True, sample_size=251)


def truthy(value):
    assert value in ('True', 'False', 'true', 'false')
    return value.lower() == 'true'


def audit_results(verify):
    from . import evaluate as E
    lock, metadata, predictions, protocol = E.locked_inputs()
    result = C.read(C.DOC / 'RESULTS.json')
    verify.walk(result)
    verify.walk(C.read(C.DOC / 'REFERENCE_BINDINGS.json'))
    raw = C.read(C.RAW / 'POSE_METRICS.json')
    groups = C.read(C.RAW / 'EVAL_GROUPS.json')
    path = C.DOC / 'FRAME_RESULTS.csv'
    assert path.is_file(), 'Report must publish the complete frame CSV'
    rows = list(csv.DictReader(path.open()))
    names = E.model_names()
    assert len(rows) == 1557
    assert len({(r['model'], r['id']) for r in rows}) == 1557
    assert Counter(r['model'] for r in rows) == Counter({n: 173 for n in names})
    index = {(r['model'], r['id']): r for r in rows}
    scalar_checks = 0
    for name in names:
        assert set(raw[name]) == set(lock['IDs'])
        for fid in lock['IDs']:
            public, metric = index[name, fid], raw[name][fid]
            available = truthy(public['pose_available'])
            assert available == metric['available']
            for key in KEYS:
                if available:
                    np.testing.assert_allclose(float(public[key]), metric[key], rtol=1e-12, atol=1e-12)
                    scalar_checks += 1
                else:
                    assert public[key] == '' and public['full_population_error_status'] == 'POSITIVE_INFINITY'
    point_checks = []
    def median(name, ids, key):
        # Independent CSV calculation; never calls E.seed_average or bootstrap.
        values = [float(index[name, fid][key]) for fid in ids if truthy(index[name, fid]['pose_available'])]
        return float(np.median(values)) if values else float('nan')
    for pop in ('NATURAL99', 'CLEAN29', 'WOOD45'):
        ids = groups[pop]
        for before in (C.ARMS[0], 'R0', 'PRIOR1', 'FULL125'):
            name = f'{C.ARMS[1]}-minus-{before}'
            comparison = result['hierarchy'][pop][name]
            for key in KEYS:
                after = np.mean([median(f'{C.ARMS[1]}_s{s}', ids, key) for s in C.SEEDS])
                bnames = [f'{before}_s{s}' for s in C.SEEDS] if before in C.ARMS else [before] * len(C.SEEDS)
                reference = np.mean([median(n, ids, key) for n in bnames])
                expected = after - reference
                for stored in (comparison['mean_seed_median_difference'][key], comparison['hierarchical_bootstrap']['metrics'][key]['point_estimate']):
                    if np.isfinite(expected):
                        assert stored['status'] == 'FINITE'
                        np.testing.assert_allclose(stored['value'], expected, atol=1e-10, rtol=1e-10)
                    else:
                        assert stored['value'] is None
                point_checks.append(dict(population=pop, comparison=name, metric=key, CSV_point_estimate=E.scalar(expected)))
    receipts = {}
    for name in names:
        receipts[name] = C.read(C.RAW / 'predictions' / f'{name}_RECEIPT.json')
        verify.walk(receipts[name])
    write_owned('INFERENCE_RECEIPTS.json', dict(prediction_lock=C.bind(C.DOC / 'PREDICTIONS_LOCK.json'),
        model_receipts=receipts, image_forwards=sum(v['image_forwards'] for v in receipts.values()),
        new_fits=0, optimizer_updates=0, no_new_inference_by_validator=True))
    return dict(PASS=True, public_CSV_rows=len(rows), unique_model_frames=len(index),
                per_model_frames=173, scalar_T_R_checks=scalar_checks,
                independent_CSV_bootstrap_point_checks=point_checks,
                original_stability_verdict=result['stability']['verdict'],
                original_stability_PASS=result['stability']['PASS'],
                artifact_validation_does_not_override_stability_verdict=True)


def audit_figures():
    plan = C.read(C.DOC / 'FIGURE_CASE_SELECTION.json')
    assert plan['gallery_seed'] == 1 and plan['outcome_based_frame_selection'] is False
    pages = plan['gallery_pages']
    ids = [fid for page in pages for fid in page['IDs']]
    assert len(ids) == len(set(ids)) == 99 and len(pages) == 19
    original = C.D.metadata()
    assert ids == [r['id'] for r in sorted(original, key=lambda r: (r['recording'], r['id'])) if r['severity'] != 'CLEAN']
    required = set(SUMMARY_FIGURES) | {r['figure'] for r in pages}
    actual = {p.name for p in (C.DOC / 'figures').iterdir() if p.is_file()}
    assert actual == required, ('Figure inventory differs', sorted(required - actual), sorted(actual - required))
    decoded = []
    for name in sorted(actual):
        p = C.DOC / 'figures' / name
        image = cv2.imread(str(p))
        assert image is not None and min(image.shape[:2]) > 100, name
        decoded.append(dict(file=C.bind(p), width=int(image.shape[1]), height=int(image.shape[0])))
    gallery = (C.DOC / 'GALLERY_NATURAL99.md').read_text()
    for p in pages:
        assert gallery.count(f'(figures/{p["figure"]})') == 1
    return dict(PASS=True, gallery_seed=1, full_natural_frames=99, gallery_pages=19,
                images=len(decoded), decodable=decoded, best_case_or_best_seed_selection=False,
                semantic_visual_review='Independent rendering/legibility review is reported separately by the author; decoding alone does not prove visual correctness.')


def publication_files(checkout, verify):
    assert subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip() == PUBLIC_BASE
    files = {p for p in C.DOC.rglob('*') if p.is_file() and p.name != 'PUBLICATION_MANIFEST.json'}
    files.update(C.HERE.glob('*.py'))
    files.update(C.ROOT / p for p in ('.gitignore', 'readme.md'))
    assert sum((C.ROOT / p).stat().st_size for p in SUPPORT) < 5_000_000
    files.update(C.ROOT / p for p in SUPPORT)
    added_code = []
    for name in ('TRAIN_CODE_LOCK.json', 'TRAIN_TRANSITIVE_CODE_LOCK.json'):
        for b in C.read(C.DOC / name)['files']:
            verify.verify(b)
            source = C.ROOT / b['path']
            assert source.suffix == '.py'
            remote = checkout / b['path']
            if not remote.is_file() or hashlib.sha256(remote.read_bytes()).hexdigest() != b['sha256']:
                files.add(source)
                added_code.append(b['path'])
    # Planned validator receipt is included after it is written below.
    files.add(C.DOC / 'PUBLICATION_VALIDATION.json')
    forbidden_suffixes = {'.pt', '.pth', '.npz', '.npy', '.zip', '.mp4', '.pkl', '.bin'}
    for p in files:
        assert p.suffix not in forbidden_suffixes and not p.is_symlink(), p
        assert not p.is_relative_to(C.RAW), ('Raw outputs are private', p)
        if p.exists():
            assert p.stat().st_size < 50_000_000, ('File exceeds publication limit', p)
        if p.suffix in ('.png', '.jpg', '.jpeg'):
            assert p.is_relative_to(C.DOC / 'figures'), ('Only generated review figures are publishable', p)
    return files, sorted(set(added_code))


def audit_links(files, checkout):
    tracked = set(subprocess.check_output(['git', '-C', str(checkout), 'ls-files', '-z'], text=True).split('\0'))
    offered = {str(p.relative_to(C.ROOT)) for p in files}
    # The manifest cannot contain its own hash; root publishes it explicitly.
    offered.add(str((C.DOC / 'PUBLICATION_MANIFEST.json').relative_to(C.ROOT)))
    checked, errors = [], []
    for p in sorted(files):
        if p.suffix.lower() != '.md':
            continue
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', p.read_text()):
            target = target.strip().removeprefix('<').removesuffix('>')
            if target.startswith(('https://', 'http://', 'mailto:', '#')):
                continue
            href = unquote(target.split('#', 1)[0])
            if not href:
                continue
            resolved = (p.parent / href).resolve()
            try:
                relative = str(resolved.relative_to(C.ROOT))
            except ValueError:
                errors.append(dict(document=str(p.relative_to(C.ROOT)), href=target, reason='OUTSIDE_REPOSITORY'))
                continue
            exists = resolved.exists() or relative in offered
            published = relative in offered or relative in tracked or any(t.startswith(relative.rstrip('/') + '/') for t in tracked)
            row = dict(document=str(p.relative_to(C.ROOT)), href=target, target=relative,
                       exists=exists, included_or_previously_tracked=published)
            checked.append(row)
            if not exists or not published:
                errors.append(row)
    return dict(PASS=not errors, checked_links=len(checked), errors=errors, links=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, default=CHECKOUT)
    args = parser.parse_args()
    for name in ('RESULTS.json', 'REPORT_KO.md', 'FRAME_RESULTS.csv', 'GALLERY_NATURAL99.md', 'TRAINING_COMPLETE.json'):
        assert (C.DOC / name).is_file(), ('Not ready for final publication validation', name)
    assert all((C.DOC / 'figures' / f).is_file() for f in SUMMARY_FIGURES)
    verify = Verifier()
    training = audit_training(verify)
    results = audit_results(verify)
    figures = audit_figures()
    files, missing_code = publication_files(args.checkout, verify)
    links = audit_links(files, args.checkout)
    code = []
    for p in sorted(files):
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=str(p))
            code.append(str(p.relative_to(C.ROOT)))
    validation = dict(PASS=links['PASS'], audited_at=C.now(),
        scope='Training trace/checkpoint/input integrity, public scalar values, CSV-derived bootstrap point estimates, figure membership/decoding and publication link/file completeness.',
        excludes='Does not establish independent new-recording generalization, certify physical reference accuracy, or convert failed stability gates into success.',
        training=training, results=results, figures=figures, markdown_links=links,
        AST_parsed_files=code, unique_hashed_input_files=len(verify.checked),
        publication_base_commit=PUBLIC_BASE, added_missing_or_changed_dependency_code=missing_code,
        inference_forwards=0, optimizer_updates=0, goal_marked_complete=False)
    write_owned('PUBLICATION_VALIDATION.json', validation)
    if not links['PASS']:
        raise AssertionError(('Publication contains missing links', links['errors']))
    bindings = [C.bind(p) for p in sorted(files)]
    manifest = dict(created_at=C.now(), name=C.NAME, publication_base_commit=PUBLIC_BASE,
        files=bindings, file_count=len(bindings), total_bytes=sum(b['bytes'] for b in bindings),
        validation=C.bind(C.DOC / 'PUBLICATION_VALIDATION.json'),
        explicit_support_files=list(SUPPORT), support_recursion=False,
        dependency_policy='Only missing/different repository Python files named by frozen TRAIN_CODE_LOCK/TRAIN_TRANSITIVE_CODE_LOCK, plus explicitly authorized small linked support files.',
        excluded=['weights', 'crop tensors', 'raw standalone input images', 'raw predictions', 'files >=50MB'],
        contains_actual_review_images=True, git_actions_performed=False,
        root_action='Copy only these bound files plus this manifest to the isolated publication checkout, then verify hashes before commit/push.')
    write_owned('PUBLICATION_MANIFEST.json', manifest)
    print('PUBLICATION_VALIDATION_PASS', len(bindings), 'files', len(code), 'Python', figures['images'], 'figures', flush=True)


if __name__ == '__main__':
    main()
