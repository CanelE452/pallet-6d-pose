"""One saved-row verification; no NN, pose solves or bootstrap regeneration."""
import collections
import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np

from .common import (ROOT, DOC, OUTPUT, BANKS, COST, OLD_DOC, LOSS_SETUP, Data,
                     read, write, sha, code_bindings, verify_derived)
from .evaluation import read_jsonl, raw_identity, reuse_evaluation
from .reporting import NEW, ORDER, PAIRS, load_split
from scripts.research.pallet_quick_pose_loss_20261006_v1.verification import (
    independent_reaggregate, numeric_equal)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def preservation():
    initial = read(OUTPUT / 'INITIAL_USER_STATE.json')
    for entry in initial['entries']:
        path = ROOT / entry['path']
        if 'symlink' in entry:
            assert path.is_symlink() and str(path.readlink()) == entry['symlink'], entry['path']
        if 'bytes' in entry:
            stat = path.stat()
            assert stat.st_size == entry['bytes'] and stat.st_mtime_ns == entry['mtime_ns'], entry['path']
            if 'sha256' in entry:
                assert sha(path) == entry['sha256'], entry['path']
    assert hashlib.sha256(git('diff', '--binary')).hexdigest() == initial['tracked_diff_sha256']
    assert git('branch', '--show-current').decode().strip() == 'main'
    staged = git('diff', '--cached', '--name-only', '-z').decode().split('\0')
    owned = ('scripts/research/pallet_quick_joint_scorer_20261006_v1/',
             '_docs/experiments/pallet_quick_joint_scorer_20261006_v1/')
    assert all(not name or name.startswith(owned) for name in staged)
    assert initial['cached_diff_sha256'] == hashlib.sha256(b'').hexdigest()
    return dict(status='PASS', existing_user_entries_checked=len(initial['entries']),
                user_tracked_diff_unchanged=True, existing_staging_was_empty=True,
                staged_scope='new namespaces only', branch='main', initial_head=initial['head'])


def signs(rows, controls):
    bins = collections.Counter()
    for a, b in zip(rows, controls):
        if not (a['pose']['available'] and b['pose']['available']):
            bins['UNAVAILABLE'] += 1
            continue
        signed = []
        for key in ('translation_cm', 'rotation_deg'):
            delta = a['pose'][key] - b['pose'][key]
            signed.append(-1 if delta < -1e-9 else 1 if delta > 1e-9 else 0)
        bins[f'T{signed[0]}_R{signed[1]}'] += 1
    return dict(bins)


def verify():
    began = time.monotonic()
    protocol, run, result = (read(DOC / name) for name in
                             ('PROTOCOL.json', 'RUN_RECEIPTS.json', 'RESULTS.json'))
    setup = read(LOSS_SETUP)
    assert protocol['training_code_bindings'] == code_bindings()
    assert run['tests']['status'] == 'PASS' and not run.get('failures')
    verify_derived(setup)
    for binding in result['source_bindings'] + [result['code_binding']]:
        path = ROOT / binding['path']
        assert sha(path) == binding['sha256'] and path.stat().st_size == binding['bytes']
    registry = read(OLD_DOC / 'results/A_ID_MANIFEST.json')['IDs']
    train = set(registry['train'])
    assert len(train) == 55915 and not train.intersection(registry['synthetic_evaluation'])
    assert not train.intersection(registry['real_evaluation'])
    fits = run['methods']
    for key in ('initial_state_sha256', 'common_initial_state_sha256',
                'phi_initial_state_sha256', 'order_sha256'):
        assert fits['LOCAL_CAP'][key] == fits['JOINT8'][key]
    data = Data(ROOT)
    lookup = {data.source['records'][data.indices[r]]['id']: r for r in data.eval_rows}
    source_banks = np.load(BANKS / 'source_banks.npy', mmap_mode='r')
    real_banks = np.load(BANKS / 'sampling_masks/REAL_DEV_generated_banks.npz')
    real_lookup = {str(fid): i for i, fid in enumerate(real_banks['ids'])}
    cost = np.load(COST / 'cost_ADDsym_m.npy', mmap_mode='r')
    available = np.load(COST / 'F_available.npy', mmap_mode='r')
    numeric_checks = native_checks = final_F = batches = examples = probes = 0
    pnp = dict(solvePnP=0, solvePnPGeneric=0, solvePnPRefineLM=0)
    artifacts = []
    for method in NEW:
        fit = fits[method]
        assert fit['status'] == 'DONE' and fit['complete'] and fit['seed'] == 1
        assert fit['updates'] == 6000 and fit['exposures'] == 96000
        assert fit['params'] == 26169 and fit['head_params'] == 4680 and fit['phi_params'] == 1230
        assert fit['code_bindings'] == protocol['training_code_bindings']
        assert fit['new_backbone_forwards'] == fit['new_final_F_calls'] == 0
        assert fit['front_new_layer_gradients_seen_after_first_step']
        path = Path(fit['checkpoint_path'])
        assert sha(path) == fit['checkpoint_sha256'] and path.stat().st_size == fit['checkpoint_bytes']
        probe_path = DOC / f'TRAIN_PROBE_{method}_seed1.json'
        probe = read(probe_path)
        guard = read(OUTPUT / 'probes' / f'{method}_seed1_ATTEMPT.json')
        assert probe['complete'] and guard['status'] == 'DONE' and guard['result_sha256'] == sha(probe_path)
        assert probe['binding'] == guard['binding'] and probe['checkpoint_sha256'] == fit['checkpoint_sha256']
        assert [r['id'] for r in probe['rows']] == setup['calibration']['ids'][:256]
        assert [r['source_cache_row'] for r in probe['rows']] == setup['calibration']['rows'][:256]
        assert probe['execution']['new_F_calls'] == 0 and probe['execution']['new_refiner_batches'] == 16
        assert probe['execution']['new_refiner_examples'] == 256 and probe['GT_inference_access'] is False
        for row in probe['rows']:
            c = cost[row['source_cache_row']]
            index, oracle = row['selected_index'], int(c.argmin())
            assert row['oracle_index'] == oracle and row['oracle_exact'] == (index == oracle)
            assert row['selected_F_available'] == bool(available[row['source_cache_row'], index])
            numeric_equal(row['oracle_ADDsym_m'], c[oracle])
            if row['selected_F_available']:
                numeric_equal(row['selected_ADDsym_m'], c[index])
                numeric_equal(row['oracle_gap_m'], c[index] - c[oracle])
            else:
                assert row['selected_ADDsym_m'] is None
            assert np.isfinite([row[k] for k in ('CE', 'target_entropy', 'CE_minus_target_entropy',
                                               'NoOp_probability', 'entropy')]).all()
            assert np.isclose(row['CE_minus_target_entropy'], row['CE'] - row['target_entropy'], atol=1e-6, rtol=0)
            assert 0 <= row['NoOp_probability'] <= 1 and 0 <= row['entropy'] <= np.log(row['action_count']) + 1e-5
            numeric_checks += 7
        probes += len(probe['rows'])
    for split, expected in (('SYNTH_HELDOUT', registry['synthetic_evaluation']),
                            ('REAL_DEV', registry['real_evaluation'])):
        ids, methods, _ = load_split(split, fits)
        assert ids == expected and set(methods) == set(ORDER)
        saved = result['splits'][split]
        for method in ORDER:
            summary = saved['summaries'][method]
            rows = methods[method]
            # Baselines have no GEO candidate index, so NoOp is NA rather than 0.
            if summary['NoOp'] is None:
                assert all(r.get('selected_index') is None for r in rows)
                summary = dict(summary, NoOp=0)
            numeric_checks += independent_reaggregate(rows, summary)
            full = np.asarray([v for r in rows if r['corner']['evaluable']
                               for v in r['corner']['errors']], dtype=float)
            numeric_equal(np.median(full), summary['corner']['full_penalty_median_px'])
            numeric_equal(np.quantile(full, .9), summary['corner']['full_penalty_P90_px'])
            good_to_bad = bad_to_good = 0
            for original, changed in zip(methods['RAW'], rows):
                a, b = original['corner'], changed['corner']
                if not (a['evaluable'] and b['evaluable']):
                    continue
                assert a['canonical_valid'] == b['canonical_valid']
                for k, valid in enumerate(a['canonical_valid']):
                    if valid:
                        x, y = a['canonical_errors'][k], b['canonical_errors'][k]
                        good_to_bad += x < 5 and y > 10
                        bad_to_good += x > 20 and y < 10
            damage = saved['RAW_canonical_damage'][method]
            assert good_to_bad == damage['good5_to_bad10'] and bad_to_good == damage['bad20_to_good10']
            numeric_checks += 4
        for method in NEW:
            receipt = reuse_evaluation(split, method, OUTPUT, fits[method])
            assert receipt is not None
            rows = methods[method]
            for row, raw, oracle in zip(rows, methods['RAW'], methods['GEO_ORACLE']):
                fid, index = row['id'], row['selected_index']
                bank = (source_banks[lookup[fid]] if split == 'SYNTH_HELDOUT'
                        else real_banks['points'][real_lookup[fid]])
                recorded = np.asarray(row['native_points'], dtype=float)
                assert np.array_equal(recorded, bank[index], equal_nan=True), fid
                assert np.array_equal(recorded[8:], bank[0, 8:], equal_nan=True), fid
                identity = raw_identity(dict(row, native_points=recorded), {'q': bank[0]}, raw)
                for key in ('RAW_native_equal', 'RAW_pose_metric_equal', 'RAW_pose_equal'):
                    assert row[key] == identity[key], (fid, key)
                if row['pose']['available']:
                    assert row['pose']['ADDsym_m'] >= oracle['pose']['ADDsym_m'] - 1e-7
                native_checks += 1
            equality = saved['RAW_equality'][method]
            assert equality['native_equal_denominator'] == equality['pose_metric_equal_denominator'] == len(rows)
            assert equality['native_equal_count'] == sum(r['RAW_native_equal'] for r in rows)
            assert equality['pose_metric_and_final_hypothesis_equal_count'] == sum(r['RAW_pose_metric_equal'] for r in rows)
            assert equality['pose_equal_by_same_native_same_F_contract_count'] == sum(r['RAW_pose_equal'] is True for r in rows)
            assert equality['pose_same_input_contract_eligible'] == sum(r['RAW_pose_equal'] is not None for r in rows)
            execution = receipt['execution']
            counts = {k: sum(r['PnP_counts'][k] for r in rows) for k in pnp}
            assert counts == execution['PnP_counts']
            for key in pnp:
                pnp[key] += counts[key]
            assert execution['new_final_F_attempts'] == execution['new_final_F_completed'] == len(rows)
            assert execution['new_refiner_examples'] == len(rows)
            assert execution['new_refiner_batches'] == (125 if split == 'SYNTH_HELDOUT' else 319)
            final_F += len(rows); batches += execution['new_refiner_batches']; examples += len(rows)
            artifacts.append(dict(method=method, split=split, rows=len(rows), sha256=receipt['rowfile_sha256']))
        pairs = [*PAIRS, ('LOCAL_CAP', 'JOINT8')]
        for new, base in pairs:
            comparison = (saved['LOCAL_CAP_minus_JOINT8'] if (new, base) == ('LOCAL_CAP', 'JOINT8')
                          else saved['comparisons'][f'{new}_minus_{base}'])
            a, b = methods[new], methods[base]
            assert comparison['joint_frame_improvement']['signed_counts'] == signs(a, b)
            corner_delta = [x['corner']['E_sym'] - y['corner']['E_sym'] for x, y in zip(a, b)
                            if x['corner']['evaluable'] and y['corner']['evaluable']]
            for level, stats in comparison['paired']['statistics']['E_sym'].items():
                numeric_equal(np.mean(corner_delta), stats['mean_paired_difference'])
                numeric_equal(np.median(corner_delta), stats['median_paired_difference'])
                assert stats['common_eligible_frames'] == len(corner_delta)
                assert stats['draw_sha256'] == saved['bootstrap'][level]['draw_sha256']
                numeric_checks += 4
            for metric in ('translation_cm', 'rotation_deg', 'ADDsym_m'):
                delta = [x['pose'][metric] - y['pose'][metric] for x, y in zip(a, b)
                         if x['pose']['available'] and y['pose']['available']]
                for level, stats in comparison['paired']['statistics'][metric].items():
                    numeric_equal(np.mean(delta), stats['mean_paired_difference'])
                    numeric_equal(np.median(delta), stats['median_paired_difference'])
                    assert stats['common_eligible_frames'] == len(delta)
                    assert stats['seed'] == 20260917 and stats['resamples'] == 10000
                    assert stats['draw_sha256'] == saved['bootstrap'][level]['draw_sha256']
                    numeric_checks += 5
                for statistic, target in (('median', 'difference_of_medians'), ('P90', 'difference_of_P90')):
                    numeric_equal(saved['summaries'][new]['pose'][metric][statistic] -
                                  saved['summaries'][base]['pose'][metric][statistic],
                                  comparison['pose_preservation'][target][metric])
                    numeric_checks += 1
    assert final_F == examples == native_checks == 4608 and probes == 512 and batches == 888
    assert sum(f['updates'] for f in fits.values()) == 12000
    declared = run['execution']
    assert declared['formal_updates'] == 12000 and declared['formal_exposures'] == 192000
    assert declared['final_F_attempts'] == declared['final_F_completed'] == final_F
    assert declared['evaluation_refiner_batches'] == batches and declared['evaluation_examples'] == examples
    assert declared['TRAIN_probe_examples'] == probes and declared['TRAIN_probe_batches'] == 32
    assert declared['TRAIN_probe_F'] == 0 and declared['PnP_counts'] == pnp
    assert run['completed_stage_code_bindings'] == {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    owned = [p for directory in (DOC, Path(__file__).parent) for p in directory.rglob('*') if p.is_file()]
    assert not any(p.suffix.lower() in ('.tex', '.bib', '.pdf', '.png', '.pt', '.npy', '.npz') for p in owned)
    protected = preservation()
    verification = dict(status='PASS', seconds=time.monotonic() - began, preservation=protected,
        scope='saved raw-row metrics, native bank candidate/RAW center/identity, paired point estimates and draw metadata; no interval regeneration',
        exact_native_candidates_checked=native_checks, metric_checks=numeric_checks, row_artifacts=artifacts,
        input_states_checked=len(run['inputs']['states']), input_scope=run['inputs']['input_verification'],
        initialization_and_order_equal=True, TRAIN_eval_disjoint=True,
        execution=dict(updates=12000, exposures=192000, evaluation_examples=examples,
            evaluation_batches=batches, final_F=final_F, PnP_counts=pnp, TRAIN_probe_examples=probes,
            TRAIN_probe_batches=32, TRAIN_probe_F=0, cost_regeneration=0, backbone_forwards=0,
            additional_verifier_NN=0, additional_verifier_F=0, additional_verifier_updates=0, paper_edits=0),
        final_code_bindings={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')})
    result['verification'] = verification
    write(DOC / 'RESULTS.json', result)
    print('JOINT_VERIFICATION', verification['status'], native_checks, numeric_checks,
          round(verification['seconds'], 2), flush=True)
    return verification
