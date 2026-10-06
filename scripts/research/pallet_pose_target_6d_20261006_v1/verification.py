"""Independently validate all durable TRAIN cost records and published results.

This reader never calls a model, pose function, optimizer or legacy mutator.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from .preflight import ROOT, DOC, read, write, sha, inherited_files, original_state, audit
from .baseline import BASELINE_ROOT

RECORD = np.dtype(dict(names=['row_a','index_a','tag_a','row_c','index_c','tag_c','ADD','T','R','WD','PnP','Generic','LM'],
    formats=['<u4','<u2','u1','<u4','<u2','u1','<f8','<f4','<f4','u1','<u2','<u2','<u2'],
    offsets=[0,4,6,7,11,13,14,22,26,30,31,33,35],itemsize=37))
PNP = ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM')
ACTION_FIT_KEYS = ('observed_exposures', 'eligible_target_exposures', 'excluded_target_exposures',
    'matching_eligible_exposures', 'moving_target_exposures', 'matching_moving_target_exposures',
    'NoOp_target_exposures', 'matching_NoOp_target_exposures', 'eligible_CE_sum')


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def artifact(path, expected):
    path = Path(path)
    assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], path


def action_fit_checks(record, exposures):
    """Check saved existing-forward diagnostics without invoking a scorer."""
    for key in ACTION_FIT_KEYS[:-1]:
        assert isinstance(record[key], int) and record[key] >= 0, key
    assert record['observed_exposures'] == exposures
    assert record['eligible_target_exposures'] + record['excluded_target_exposures'] == exposures
    assert record['moving_target_exposures'] + record['NoOp_target_exposures'] == record['eligible_target_exposures']
    assert record['matching_moving_target_exposures'] <= record['moving_target_exposures']
    assert record['matching_NoOp_target_exposures'] <= record['NoOp_target_exposures']
    assert record['matching_moving_target_exposures'] + record['matching_NoOp_target_exposures'] == record['matching_eligible_exposures']
    assert np.isfinite(record['eligible_CE_sum']) and record['eligible_CE_sum'] >= 0
    for key, numerator, denominator in (
        ('eligible_action_accuracy', 'matching_eligible_exposures', 'eligible_target_exposures'),
        ('moving_target_action_accuracy', 'matching_moving_target_exposures', 'moving_target_exposures'),
        ('NoOp_target_action_accuracy', 'matching_NoOp_target_exposures', 'NoOp_target_exposures'),
        ('eligible_mean_CE', 'eligible_CE_sum', 'eligible_target_exposures')):
        expected = record[numerator] / record[denominator] if record[denominator] else None
        assert record[key] == expected, key
    assert record['additional_model_forwards'] == record['additional_optimizer_updates'] == 0
    return 20  # Saved-diagnostic invariant assertions, including eight count fields.


def fit_diagnostics(fit, cache, seed):
    total = fit['observed_preupdate_training_action_fit']
    checks = action_fit_checks(total, 96000)
    assert total['excluded_target_exposures'] == fit['excluded_target_exposures']
    windows = [r for r in fit['history'] if r['step'] % 100 == 0]
    assert [r['step'] for r in windows] == list(range(100, 6001, 100))
    checks += 2
    accumulated = {key: 0 for key in ACTION_FIT_KEYS}
    for record in windows:
        step = record['step']
        window = record['window_observed_preupdate_action_fit']
        cumulative = record['cumulative_observed_preupdate_action_fit']
        assert window['first_step'] == step - 99 and window['last_step'] == step
        checks += 1
        checks += action_fit_checks(window, 1600) + action_fit_checks(cumulative, step * 16)
        for key in ACTION_FIT_KEYS:
            accumulated[key] += window[key]
            if key == 'eligible_CE_sum':
                assert np.isclose(accumulated[key], cumulative[key], rtol=1e-12, atol=1e-9), key
            else:
                assert accumulated[key] == cumulative[key], key
        checks += len(ACTION_FIT_KEYS)
    assert windows[-1]['cumulative_observed_preupdate_action_fit'] == total
    final = fit['final_observed_preupdate_training_window']
    assert final == windows[-1]['window_observed_preupdate_action_fit']
    assert final['first_step'] == 5901 and final['last_step'] == 6000
    directory = cache / 'fits' / f'POSE_TARGET_GEO_seed{seed}'
    first = read(directory / 'FIRST_STEP.json')
    assert first == fit['first_step']
    checks += action_fit_checks(first['observed_preupdate_action_fit'], 16)
    attempt = read(directory / 'ATTEMPT_STATE.json')
    assert attempt['binding'] == fit['binding'] and attempt['status'] == 'CHECKPOINTED'
    assert attempt['attempted_step'] == attempt['completed_updates'] == 6000
    assert attempt['completed_exposures'] == 96000
    return checks + 7


def evaluation_artifacts(cache, split, seed, path, result):
    """Validate per-ID durable rows against the final published JSON."""
    directory = cache / 'evaluations' / split / f'seed{seed}'
    completion_path = directory / 'COMPLETE.json'
    completion = read(completion_path)
    manifest_path = directory / 'ROW_MANIFEST.json'
    manifest = read(manifest_path)
    packet = read(directory / 'ROW_BINDING.json')
    count = result['full_denominator']
    for value in (completion, manifest, packet):
        assert value['binding'] == result['binding'] and value['full_denominator'] == count
    assert packet['checkpoint_sha256'] == result['checkpoint_sha256']
    assert packet['code_sha256'] == result['code_sha256']
    assert completion['result_sha256'] == sha(path)
    assert completion['row_manifest_sha256'] == sha(manifest_path)
    assert completion['execution'] == result['execution']
    assert len(manifest['rows']) == count
    seen = set()
    for index, (saved, row) in enumerate(zip(manifest['rows'], result['rows'])):
        assert saved['id'] == row['id'] and saved['id'] not in seen
        seen.add(saved['id'])
        assert saved['file'] == f'row_{index:06d}.json'
        row_path = directory / saved['file']
        artifact(row_path, saved)
        payload = read(row_path)
        assert payload['binding'] == result['binding'] and payload['row'] == row
    journal = read(DOC / f'results/{split}_POSE_TARGET_GEO_J_seed{seed}_ATTEMPTS.json')
    assert journal['status'] == 'DONE' and journal['binding'] == result['binding']
    assert journal['full_denominator'] == count and journal['execution'] == result['execution']
    assert journal['result_sha256'] == completion['result_sha256']
    assert Path(journal['external_completion_path']).resolve() == completion_path.resolve()
    assert journal['external_completion_sha256'] == sha(completion_path)
    return count, 5 * count + 14


def evaluation_counts(result):
    rows = result['rows']
    expected = dict(new_final_F_attempts=sum(r['F_attempt'] for r in rows),
        new_final_F_completed=sum(r['F_complete'] for r in rows),
        final_F_available=sum(r['F_available'] for r in rows),
        new_backbone_calls=0, new_bank_generation=0, optimizer_updates=0,
        PnP_counts={name: sum(r['PnP_counts'][name] for r in rows) for name in PNP})
    if result['split'] == 'SYNTH_HELDOUT':
        expected.update(new_refiner_batches=(len(rows) + 15) // 16, new_refiner_examples=len(rows))
    else:
        # Every saved real action bank with >1 member has a captured object;
        # absent captured objects were frozen as one-member NoOp banks.
        active = sum(r['action_count'] > 1 for r in rows)
        expected.update(new_refiner_batches=active, new_refiner_examples=active)
    for key, value in expected.items():
        assert result['execution'][key] == value, (result['split'], key)
    assert result['summary']['pose']['total_frames'] == len(rows)
    assert result['summary']['pose']['available'] == expected['final_F_available']
    assert result['summary']['pose']['failures'] == len(rows) - expected['final_F_available']
    assert result['summary']['NoOp'] == sum(r['selected_index'] == 0 for r in rows)
    return expected


def verify(cache, full_input_hash=True):
    cache=Path(cache);began=time.monotonic();bindings=read(DOC/'INPUT_BINDINGS.json')
    assert cache.resolve()==Path(bindings['diagnostic_cache']).resolve()
    assert sha(DOC/'PROTOCOL.json')==bindings['protocol_sha256']
    assert inherited_files()==bindings['protected_baseline_files']
    assert original_state(Path(bindings['source_ROOT']))==bindings['original_user_checkout']
    for entry in bindings['external_inputs']:
        stat=Path(entry['path']).stat()
        assert stat.st_size==entry['bytes'] and stat.st_mtime_ns==entry['mtime_ns'],entry['path']
    manifest=read(DOC/'POSE_COST_CACHE_MANIFEST.json');header=read(cache/'cache_header.json')
    assert manifest['status']=='PASS' and manifest['completed_rows']==55915
    assert manifest['binding']==header['binding']
    assert manifest['header_sha256']==sha(cache/'cache_header.json')
    assert header['protocol_sha256']==sha(DOC/'PROTOCOL.json')
    assert header['input_bindings_sha256']==sha(DOC/'INPUT_BINDINGS.json')
    assert header['cost_cache_code_sha256']==sha(Path(__file__).with_name('cost_cache.py'))
    assert Path(header['cache_dir']).resolve()==cache.resolve()
    assert Path(header['source_root']).resolve()==Path(bindings['source_ROOT']).resolve()
    assert Path(header['bank_cache']).resolve()==Path(bindings['baseline_cache']).resolve()
    assert header['source_rows']==60000 and header['usable_TRAIN_rows']==55915
    assert manifest['candidate_F_attempted']==manifest['candidate_F_completed']==11238915
    assert manifest['incomplete_attempted_F']==0
    assert {e['file'] for e in manifest['files']}=={e['file'] for e in header['files'].values()}|{'train_rows.npy'}
    assert len(manifest['files'])==len(header['files'])+1
    for entry in manifest['files']:
        artifact(cache/entry['file'],entry)
    rows=np.load(cache/'train_rows.npy',mmap_mode='r')
    assert rows.dtype==np.dtype('int64') and len(rows)==55915 and len(np.unique(rows))==55915
    row_hash=hashlib.sha256(str(rows.dtype).encode()+str(rows.shape).encode())
    row_hash.update(np.ascontiguousarray(rows).tobytes())
    assert row_hash.hexdigest()==header['train_rows_sha256']
    arrays={name:np.load(cache/(name+'.npy'),mmap_mode='r') for name in header['files']}
    for name, array in arrays.items():
        assert list(array.shape)==header['files'][name]['shape']
        assert array.dtype==np.dtype(header['files'][name]['dtype']) and not array.flags.writeable
    assert arrays['done'][rows].all() and (arrays['row_state'][rows]==2).all()
    assert (arrays['action_counts'][rows]==201).all()
    assert np.array_equal(np.flatnonzero(arrays['usable_train']),rows)
    pnp={'solvePnP':0,'solvePnPGeneric':0,'solvePnPRefineLM':0};cells=0;checks=0
    completed_chunks=0;available_cells=0;failed_cells=0;worker_seconds=0.;invocation_F=0
    for start in range(0,len(rows),64):
        group=rows[start:start+64];chunk=start//64
        journal=cache/'journals'/f'chunk_{chunk:04d}.bin';receipt=read(cache/'receipts'/f'chunk_{chunk:04d}.json')
        assert read(journal.with_suffix('.binding.json'))==dict(binding=header['binding'],chunk=chunk)
        assert receipt['chunk']==chunk and receipt['rows']==group.tolist()
        raw=journal.read_bytes();assert len(raw)==len(group)*201*37
        assert sha(journal)==receipt['journal_sha256'] and len(raw)==receipt['journal_bytes']
        records=np.frombuffer(raw,dtype=RECORD)
        expected_rows=np.repeat(group,201);expected_indices=np.tile(np.arange(201),len(group))
        for key,expected in [('row_a',expected_rows),('row_c',expected_rows),('index_a',expected_indices),('index_c',expected_indices)]:
            assert np.array_equal(records[key],expected),(chunk,key);checks+=len(records)
        assert (records['tag_a']==1).all() and (records['tag_c']==2).all();checks+=2*len(records)
        source_rows=records['row_a'];indices=records['index_a']
        for field,name in [('ADD','cost_ADDsym_m'),('T','translation_cm'),('R','rotation_deg'),('WD','WD_hypothesis')]:
            assert np.array_equal(records[field],arrays[name][source_rows,indices],equal_nan=True),(chunk,name);checks+=len(records)
        available=np.isfinite(records['ADD'])
        assert np.array_equal(available,arrays['F_available'][source_rows,indices]);checks+=len(records)
        assert (arrays['candidate_state'][source_rows,indices]==2).all();checks+=len(records)
        assert np.isposinf(records['ADD'][~available]).all()
        assert np.isposinf(records['T'][~available]).all() and np.isposinf(records['R'][~available]).all()
        assert (records['WD'][~available]==0).all()
        assert np.isfinite(records['T'][available]).all() and np.isfinite(records['R'][available]).all()
        assert ((records['WD'][available]>=1)&(records['WD'][available]<=3)).all()
        costs=np.array(arrays['cost_ADDsym_m'][group]);exists=np.isfinite(costs).any(-1)
        target=np.where(exists,costs.argmin(-1),-1)
        assert np.array_equal(target,arrays['oracle_index'][group]);checks+=len(group)
        assert receipt['binding']==header['binding'] and receipt['status']=='PASS'
        assert receipt['attempted']==receipt['completed']==len(records)
        assert receipt['incomplete_rows']==0 and receipt['all_F_invalid_rows']==int((~exists).sum())
        assert 0<=receipt['new_F_calls']<=len(records)
        for field,name in [('PnP','solvePnP'),('Generic','solvePnPGeneric'),('LM','solvePnPRefineLM')]:
            actual=int(records[field].astype(np.uint64).sum());assert actual==receipt['PnP_counts'][name];pnp[name]+=actual
        cells+=len(records);completed_chunks+=1
        available_cells+=int(available.sum());failed_cells+=int((~available).sum())
        worker_seconds+=receipt['seconds'];invocation_F+=receipt['new_F_calls']
    assert cells==11238915 and completed_chunks==874 and pnp==manifest['PnP_counts']
    assert manifest['candidate_F_available']==available_cells
    assert manifest['candidate_F_failures']==failed_cells and available_cells+failed_cells==cells
    assert manifest['available_targets']==int((arrays['oracle_index'][rows]>=0).sum())
    assert manifest['excluded_all_F_invalid']==int((arrays['oracle_index'][rows]<0).sum())
    assert manifest['available_targets']+manifest['excluded_all_F_invalid']==55915
    assert manifest['oracle_NoOp']==int((arrays['oracle_index'][rows]==0).sum())
    assert manifest['pending_rows']==manifest['incomplete_attempt_rows']==0
    assert manifest['registered_F_calls_total']==cells
    assert manifest['chunk_seconds_sum']==worker_seconds and manifest['invocation_new_F_calls']==invocation_F
    parity=read(DOC/'POSE_TARGET_PARITY.json')
    assert parity['status']=='PASS' and parity['registered_cases']==len(parity['cases'])==6
    assert parity['actual_F_calls']==sum(c['actual_F_calls'] for c in parity['cases'])==1206
    assert parity['completed_F_calls']==sum(c['completed_F_calls'] for c in parity['cases'])==1206
    assert parity['PnP_counts']=={name:sum(c['PnP_counts'][name] for c in parity['cases']) for name in PNP}
    assert read(DOC/'TRAIN_INITIALIZATION_PARITY.json')['status']=='PASS'
    aux=read(DOC/'TRAIN_2D_6D_TARGET_COMPARISON.json');assert aux['status']=='PASS' and aux['full_usable_TRAIN']==55915
    assert not aux['unsupported_nonzero_pose_target_rows']
    assert aux['binding']==hash_value(dict(protocol=sha(DOC/'PROTOCOL.json'),code=sha(Path(__file__).with_name('training.py')),
        target_header=sha(cache/'cache_header.json'),bank=header['bank_sha256']))
    assert read(cache/'teacher_2d_binding.json')['binding']==aux['binding']
    for entry in aux['array_artifacts']:
        assert Path(entry['path']).resolve().parent==cache.resolve()
        artifact(Path(entry['path']),entry)
    assert np.load(cache/'teacher_2d_done.npy',mmap_mode='r')[rows].all()
    assert aux['new_refiner_forward_calls']==aux['new_backbone_calls']==aux['new_F_calls']==aux['optimizer_updates']==0
    counts=read(DOC/'EXECUTION_COUNTS.json');train=read(DOC/'TRAIN_RECEIPTS.json')
    assert counts['actual_fits']==train['actual_fits'] in (1,3)
    assert counts['optimizer_updates']==6000*counts['actual_fits']<=18000
    assert counts['training_forward_examples']==96000*counts['actual_fits']
    assert counts['bank_generation']==counts['backbone_detector_forwards']==counts['real_training']==0
    expected_ids=read(DOC/'ID_MANIFEST.json')['IDs']
    old_doc=BASELINE_ROOT/'_docs/experiments/pallet_joint_action_handoff_20261006_v1'
    assert expected_ids==read(old_doc/'results/A_ID_MANIFEST.json')['IDs']
    assert hash_value(expected_ids['train'])==header['train_ids_sha256']
    assert train['seeds'] in ([1],[1,2,3]) and len(train['seeds'])==train['actual_fits']
    screen=read(DOC/'SEED1_SCREENING.json')
    assert screen['criterion_population']=='SYNTH_HELDOUT_ONLY' and screen['REAL_used'] is False
    assert screen['continue_seeds']==(train['seeds']==[1,2,3])
    assert screen['decision'] in (('CONTINUE','MIXED_CONTINUE') if screen['continue_seeds'] else ('STOP','STOP_MIXED'))
    fits=[];evaluations=[];external_rows=0;diagnostic_checks=0
    for seed in train['seeds']:
        fit=read(DOC/f'fits/POSE_TARGET_GEO_seed{seed}.json')
        assert fit['complete'] and fit['updates']==6000 and fit['params']==20259 and fit['final_checkpoint_only']
        assert fit['seed']==seed and fit['exposures']==96000 and fit['real_training']==0
        assert fit['code_sha256']==sha(Path(__file__).with_name('training.py'))
        assert Path(fit['checkpoint_path']).resolve()==(cache/'fits'/f'POSE_TARGET_GEO_seed{seed}'/'last.pt').resolve()
        assert sha(Path(fit['checkpoint_path']))==fit['checkpoint_sha256']
        assert fit['matched_contract']['changed_components']==['training_target']
        old_fit=read(old_doc/f'A_fits/GEO_seed{seed}.json')
        assert fit['first_step']['initial_state_sha256']==old_fit['first_step']['initial_state_sha256']
        assert fit['order_sha256']==old_fit['order_sha256']
        cost_content=hash_value(dict(binding=header['binding'],files=manifest['files']))
        assert fit['cost_content_sha256']==cost_content
        assert fit['binding']==hash_value(dict(protocol=sha(DOC/'PROTOCOL.json'),code=fit['code_sha256'],
            old_training_code={name:sha(BASELINE_ROOT/'scripts/research/pallet_joint_action_handoff_20261006_v1'/name)
                for name in ('scorer.py','a_data.py','a_common.py')},bank=header['bank_sha256'],
            cost_content=cost_content,cost_header=sha(cache/'cache_header.json'),
            auxiliary=sha(DOC/'TRAIN_2D_6D_TARGET_COMPARISON.json')))
        diagnostic_checks+=fit_diagnostics(fit,cache,seed)
        fits.append(fit)
        for split,key in [('SYNTH_HELDOUT','synthetic_evaluation'),('REAL_DEV','real_evaluation')]:
            path=DOC/f'results/{split}_POSE_TARGET_GEO_J_seed{seed}.json';result=read(path)
            assert result['complete'] and result['status']=='DONE' and result['GT_inference_access'] is False
            assert result['seed']==seed and result['split']==split and result['readout']=='unchanged J'
            assert result['full_denominator']==len(expected_ids[key])==(1985 if split=='SYNTH_HELDOUT' else 319)
            assert result['code_sha256']==sha(Path(__file__).with_name('evaluation.py'))
            actual_ids=[r['id'] for r in result['rows']]
            assert len(actual_ids)==len(set(actual_ids))==len(expected_ids[key])
            assert set(actual_ids)==set(expected_ids[key])
            assert result['checkpoint_sha256']==fit['checkpoint_sha256']
            baseline=read(old_doc/f'results/A_{split}_BASELINES.json')
            raw={r['id']:r for r in baseline['rows']['RAW']}
            oracle={r['id']:r for r in read(old_doc/f'results/A_{split}_ORACLE.json')['rows'] if r['arm']=='GEO'}
            assert result['target_sha256']==baseline['target_sha256']
            assert result['bank_binding']==baseline['bank_binding']==header['bank_binding']
            if split=='SYNTH_HELDOUT':
                assert result['bank_artifact']['sha256']==header['bank_sha256']
            else:
                expected_bank=next(e for e in read(old_doc/'results/A_SAMPLING_MASK_RECEIPT.json')['external_mask_files']
                    if Path(e['path']).name=='REAL_DEV_generated_banks.npz')
                assert result['bank_artifact']['sha256']==expected_bank['sha256']
                artifact(Path(result['bank_artifact']['path']),expected_bank)
            assert result['binding']==hash_value(dict(protocol=sha(DOC/'PROTOCOL.json'),code=result['code_sha256'],
                checkpoint=fit['checkpoint_sha256'],bank=result['bank_artifact'],target=result['target_sha256']))
            for row in result['rows']:
                assert row['F_attempt'] and row['F_complete'] and 0<=row['selected_index']<row['action_count']
                assert row['action_count']==oracle[row['id']]['actions']
                assert row['oracle_selected_index']==oracle[row['id']]['index']
                assert row['oracle_ADDsym_m']==oracle[row['id']]['oracle_ADDsym_m']
                assert row['F_available']==row['pose']['available']
                expected_gap=row['pose']['ADDsym_m']-row['oracle_ADDsym_m'] if row['pose']['available'] and row['oracle_ADDsym_m'] is not None else None
                assert row['oracle_gap_m']==expected_gap
                assert row['oracle_gap_m'] is None or row['oracle_gap_m']>=-1e-7
                assert row['session']==raw[row['id']]['session']
                assert row['corner']['evaluable']==raw[row['id']]['corner']['evaluable']
                if row['corner']['evaluable']:
                    assert row['corner']['canonical_valid']==raw[row['id']]['corner']['canonical_valid']
                assert all(isinstance(row['PnP_counts'][name],int) and row['PnP_counts'][name]>=0 for name in PNP)
                checks+=10+int(row['corner']['evaluable'])
            evaluation_counts(result)
            verified,extra_checks=evaluation_artifacts(cache,split,seed,path,result)
            external_rows+=verified;checks+=extra_checks;evaluations.append(result)
    assert train['fits']==fits
    assert train['optimizer_updates']==sum(f['updates'] for f in fits)
    assert train['exposures']==sum(f['exposures'] for f in fits)
    assert train['excluded_target_exposures']==sum(f['excluded_target_exposures'] for f in fits)
    assert train['standalone_smoke_fits']==train['standalone_smoke_optimizer_updates']==train['hyperparameter_search']==0
    first_timing=read(DOC/'COST_TIMING.json')
    expected_counts=dict(
        pose_cost_F_attempted=manifest['candidate_F_attempted'],pose_cost_F_completed=manifest['candidate_F_completed'],
        pose_cost_F_failures=manifest['candidate_F_failures'],parity_F_attempted=parity['actual_F_calls'],
        parity_F_completed=parity['completed_F_calls'],
        evaluation_F_attempted=sum(e['execution']['new_final_F_attempts'] for e in evaluations),
        evaluation_F_completed=sum(e['execution']['new_final_F_completed'] for e in evaluations),
        actual_fits=len(fits),optimizer_updates=sum(f['updates'] for f in fits),
        training_forward_batches=sum(f['updates'] for f in fits),training_forward_examples=sum(f['exposures'] for f in fits),
        excluded_target_exposures=sum(f['excluded_target_exposures'] for f in fits),
        evaluation_forward_batches=sum(e['execution']['new_refiner_batches'] for e in evaluations),
        evaluation_forward_examples=sum(e['execution']['new_refiner_examples'] for e in evaluations),
        PnP_counts={name:pnp[name]+parity['PnP_counts'][name]+sum(e['execution']['PnP_counts'][name] for e in evaluations) for name in PNP},
        CPU_pose_cost_wall_seconds=first_timing['seconds_wall']+manifest['seconds_wall'],
        CPU_pose_cost_worker_seconds_sum=worker_seconds,CPU_parity_wall_seconds=parity['seconds_wall'],
        TRAIN_auxiliary_wall_seconds=aux['seconds'],fit_wall_seconds=sum(f['seconds'] for f in fits),
        evaluation_wall_seconds=sum(e['execution']['seconds_wall'] for e in evaluations),
        bank_generation=0,backbone_detector_forwards=0,real_training=0,PERM_fits=0,
        auxiliary_refiner_forwards=0,auxiliary_F=0,hyperparameter_search=0,manuscript_writes=0)
    expected_counts['all_final_F_attempted']=expected_counts['pose_cost_F_attempted']+expected_counts['parity_F_attempted']+expected_counts['evaluation_F_attempted']
    for key,value in expected_counts.items():
        assert counts[key]==value,('EXECUTION_COUNTS',key,counts[key],value)
    checks+=len(expected_counts)+7
    tests=read(DOC/'CONTRACT_TEST_RESULTS.json')
    assert tests['status']=='PASS' and tests['artifact_postconditions'] is True
    assert tests['tests_run']==tests['passed']==30 and tests['failed']==tests['skipped']==0
    assert tests['code_sha256']==sha(Path(__file__).with_name('contract_tests.py'))
    if full_input_hash:
        ending=audit(Path(bindings['source_ROOT']),Path(bindings['baseline_cache']),cache,ending=True)
        assert ending['status']=='PASS'
    result=dict(schema='pose_target_independent_saved_output_verification_v1',status='PASS',
        complete_native_cost_records=cells,complete_chunks=completed_chunks,independent_cell_checks=checks,
        check_count_scope='tracked WAL scalar comparisons plus saved-row/aggregate invariants; additional whole-array and metadata assertions are not included',
        verified_external_evaluation_rows=external_rows,existing_forward_training_diagnostic_checks=diagnostic_checks,
        execution_count_fields_independently_reconciled=list(expected_counts),reconciled_execution_counts=expected_counts,
        native_cost_available=available_cells,native_cost_failed=failed_cells,cost_PnP_counts=pnp,
        protected_baseline_files=len(bindings['protected_baseline_files']),original_user_checkout='PRESERVED BEFORE AUTHORIZED MAIN PUBLICATION',
        full_source_input_SHA_end='PASS' if full_input_hash else 'NOT_REPEATED',
        interpretation='pre-update training diagnostics use repeated ordered exposures, not final whole-TRAIN accuracy; REAL is repeated DEV, not independent physical metrology',
        new_model_calls=0,new_F_calls=0,new_PnP_calls=0,new_optimizer_updates=0,
        scientific_paper_path_changes=0,seconds=time.monotonic()-began,code_sha256=sha(Path(__file__)))
    write(DOC/'audit/INDEPENDENT_VERIFICATION.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cost-cache',type=Path,default=Path('/tmp/pallet-pose-target-6d-cache'))
    p.add_argument('--skip-full-input-hash',action='store_true')
    args=p.parse_args(argv)
    print(json.dumps(verify(args.cost_cache,not args.skip_full_input_hash),indent=2))


if __name__=='__main__':main()
