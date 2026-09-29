"""CPU-only final audit; no fit, model inference, historical write, or Git mutation.

Missing artifacts are NOT_READY. Existing locks are verified, never repaired.
Read-only Git queries record publication scope; all writes are new audit docs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import unquote

import numpy as np

from . import common as C

ARMS = ('CLEAN_RAW_CLEAR', 'CLEAN_REF_CLEAR', 'CLEAN_RAW_OCC', 'CLEAN_REF_OCC')


class NotReady(Exception):
    pass


def need(*paths):
    missing = [str(path.relative_to(C.ROOT) if path.is_relative_to(C.ROOT) else path)
               for path in map(Path, paths) if not path.exists()]
    if missing:
        raise NotReady('Missing required artifacts: '+', '.join(missing))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def bindings(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            yield value
        for child in value.values():
            yield from bindings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from bindings(child)


class Verifier:
    def __init__(self):
        self.checked = set()

    def verify(self, value):
        count = 0
        for row in bindings(value):
            key = row['path'], row['sha256']
            if key not in self.checked:
                C.verify(row)
                if 'bytes' in row:
                    require((C.ROOT/row['path']).stat().st_size == row['bytes'], 'Binding size mismatch: '+row['path'])
                self.checked.add(key)
            count += 1
        return count


def numeric_close(left, right, location='$'):
    if isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), 'Dictionary mismatch '+location)
        for key in left:
            numeric_close(left[key], right[key], location+'.'+key)
    elif isinstance(left, (list, tuple)):
        require(isinstance(right, (list, tuple)) and len(left) == len(right), 'List mismatch '+location)
        for i, (a, b) in enumerate(zip(left, right)):
            numeric_close(a, b, f'{location}[{i}]')
    elif isinstance(left, (float, int)) and not isinstance(left, bool):
        require(isinstance(right, (float, int)) and np.isclose(left, right, rtol=1e-7, atol=1e-8), 'Number mismatch '+location)
    else:
        require(left == right, 'Value mismatch '+location)


def final_status(checks):
    statuses = [value['status'] for value in checks.values()]
    return 'FAIL' if 'FAIL' in statuses else 'NOT_READY' if 'NOT_READY' in statuses else 'PASS'


def source_gate(verifier):
    names = ('CODE_LOCK.json', 'PRIMARY_PROTOCOL.json', 'PREFLIGHT.json',
             'DATASET_PAIR_PREFLIGHT.json', 'CLEAN_LOCK.json', 'START.json')
    need(*(C.DOC/name for name in names))
    values = {name: C.read(C.DOC/name) for name in names}
    for value in values.values():
        verifier.verify(value)
    p, pre, clean = values['PRIMARY_PROTOCOL.json'], values['PREFLIGHT.json'], values['CLEAN_LOCK.json']
    require(p['locked_before_fit'] and values['CODE_LOCK.json']['locked_before_fit'], 'Missing pre-fit lock')
    require(pre['passed'] and values['DATASET_PAIR_PREFLIGHT.json']['passed'], 'Preflight failed')
    require(p['new_manual_coordinates'] == p['new_training_RGB'] == 0, 'Unexpected new supervision/input')
    require(clean['locked_before_fit_and_new_scoring'] and p['clean_locked_count'] == 78, 'Clean membership lock changed')
    require(p['args']['lr0'] == 1e-5 and p['updates_per_fit'] == 320 and p['epochs'] == 5, 'Primary optimizer contract changed')
    require(set(p['arms']) == set(ARMS), 'Four-arm contract changed')
    require(pre['seed_checks']['seed43_repeat_exact_batches'] == 8, 'Seed43 reproducibility missing')
    require(all(pre['seed_checks'][key] == 8 for key in ('order_changed_batches', 'base_RGB_stream_changed_batches', 'plan_stream_changed_batches')), 'Effective seed variation absent')
    require(C.read(C.ROOT/pre['loss_gradient_contract']['path'])['status'] == 'PASS', 'True-ignore loss contract failed')
    return dict(all_bound_source_RGB_labels_teacher_and_code_unchanged=True,
        prior_user_annotation_progress_hash_unchanged=True, new_manual_coordinates=0,
        clean_locked=78, K8_actual_worker_preflight=True, effective_seed_stream_change=True,
        scope='Preservation of explicitly bound original assets; not a claim about every unbound annotation in the repository.')


def supplemental_dependency(verifier):
    dependency = C.ROOT/'scripts/research/pallet_clean19_structured_easyhard_v1/augmentation.py'
    old_lock = C.ROOT/'_docs/experiments/pallet_clean19_structured_easyhard_v1/PROTOCOL.json'
    rows = [b for b in bindings(C.read(old_lock)) if b['path'] == str(dependency.relative_to(C.ROOT))]
    require(rows, 'No historical binding for reused fill/cover/overlap')
    for row in rows:
        verifier.verify(row)
    path = C.DOC/'SUPPLEMENTAL_DEPENDENCY_AUDIT.json'
    if path.exists():
        verifier.verify(C.read(path))
    else:
        C.save(path, dict(recorded_at=C.now(), dependency=C.bind(dependency), historical_lock=C.bind(old_lock),
            historical_bindings=rows, matches_historical_locked_version=True,
            reused_functions=['fill', 'cover', 'overlap'],
            temporal_scope='Supplement recorded after study start; not retroactively claimed inside original pre-fit CODE_LOCK.',
            original_CODE_LOCK_modified=False), True)
    return dict(supplement=C.bind(path), unchanged_historical_helper=True, original_CODE_LOCK_modified=False)


def checkpoint_and_trace_audit(verifier, seed):
    import torch
    from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import pose_parameter
    from . import parity
    paths = [C.DOC/f'FIT_{arm}_S{seed}.json' for arm in ARMS]
    need(*paths, C.DOC/f'PAIR_INTEGRITY_S{seed}.json')
    protocol = C.read(C.DOC/'PRIMARY_PROTOCOL.json')
    base_model = torch.load(C.ROOT/protocol['initialization']['path'], map_location='cpu', weights_only=False)['model'].float()
    base = base_model.state_dict()
    allowed = {name for name, _ in base_model.named_parameters() if pose_parameter(name)}
    protected = set(base)-allowed
    require((len(base), len(allowed), len(protected)) == (879, 132, 747), 'Unexpected model inventory')
    fits = {}; traces = {}; states = {}
    for arm, path in zip(ARMS, paths):
        fit = C.read(path); verifier.verify(fit)
        require(fit['complete'] and fit['optimizer_steps'] == 320, 'Incomplete or wrong update count')
        require(fit['exact_R0_initialization'] and fit['protected_state_exact'], 'Runtime state assertion missing')
        require(fit['initialization'] == protocol['initialization'], 'Initialization is not frozen R0')
        require(fit['protected_tensors'] == 747 and fit['trainable_tensors'] == 132, 'Declared scope mismatch')
        final = torch.load(C.ROOT/fit['checkpoint']['path'], map_location='cpu', weights_only=False)['model'].float().state_dict()
        require(set(final) == set(base), 'Final state inventory changed')
        require(all(torch.equal(final[k], base[k]) for k in protected), 'A protected parameter/buffer changed')
        changed = {k for k in base if not torch.equal(base[k], final[k])}
        require(changed and changed <= allowed and changed == set(fit['changed_tensors']), 'Changed tensor declaration/scope mismatch')
        with (C.ROOT/fit['results_csv']['path']).open() as file:
            epochs = list(csv.DictReader(file))
        require(len(epochs) == 5, 'Epoch CSV does not have five epochs')
        require([r['steps'] for r in fit['history']] == [64, 128, 192, 256, 320], 'Optimizer history changed')
        require(all(r['protected_tensors'] == 747 for r in fit['history']), 'Epoch protection audit mismatch')
        traces[arm] = C.read(C.ROOT/fit['trace']['path']); fits[arm] = fit
        require(len(traces[arm]) == 320, 'Training trace is not320 batches')
        pre = C.read(C.RAW/f'PREFLIGHT_TRACE_S{seed}_{fit["target"]}_{fit["condition"]}_PRIVATE.json')
        for actual, expected in zip(traces[arm][:8], pre):
            for key in ('names', 'before_images', 'after_images', 'boxes', 'support', 'coordinates', 'batch_idx', 'roles', 'transfer'):
                require(actual[key] == expected[key], 'Actual training differs from preflight prefix: '+arm+'/'+key)
        states[arm] = dict(state_tensors=879, independently_compared_protected=747,
            changed_tensors=len(changed), optimizer_steps=320, first8_actual_batches_match_preflight=True,
            initial879_equality='Runtime torch.equal assertion plus bound R0 provenance; no persisted step0 tensor snapshot')
    declared = C.read(C.DOC/f'PAIR_INTEGRITY_S{seed}.json'); verifier.verify(declared)
    recomputed = parity.analyze(traces)
    numeric_close(recomputed, {key: declared[key] for key in recomputed})
    return dict(fits=4, optimizer_steps=1280, states=states, full320_four_arm_trace_recomputed=True,
                bindings=[C.bind(path) for path in paths])


def budget_audit(verifier):
    need(C.DOC/'RESOURCE_LEDGER.json')
    ledger = C.read(C.DOC/'RESOURCE_LEDGER.json')
    keys = ('student_fits', 'selector_fits', 'GPU_training_seconds', 'optimizer_updates')
    totals = {key: sum(event[key] for event in ledger['events']) for key in keys}
    numeric_close(totals, ledger['totals'])
    require(len({e['event'] for e in ledger['events']}) == len(ledger['events']), 'Duplicate resource events')
    for key, cap in [('student_fits', 10), ('selector_fits', 1), ('GPU_training_seconds', 21600)]:
        require(0 <= totals[key] <= cap, 'Goal budget exceeded: '+key)
    require(totals['optimizer_updates'] <= 3200, 'Student update cap exceeded')
    checked = 0
    for path in C.DOC.rglob('FIT_*.json'):
        fit = C.read(path)
        if not fit.get('complete'):
            continue
        prefix = '' if path.parent == C.DOC else path.parent.name+'::'
        event = next((r for r in ledger['events'] if r['event'] == prefix+'FIT_'+fit['arm']+'_S'+str(fit['seed'])), None)
        require(event is not None and event['student_fits'] == 1, 'Completed fit absent from ledger: '+path.name)
        require(event['optimizer_updates'] == fit['optimizer_steps'], 'Ledger update count mismatch')
        require(np.isclose(event['GPU_training_seconds'], fit['seconds']), 'Fit cost missing from ledger')
        verifier.verify(fit['checkpoint']); checked += 1
    return dict(recomputed_totals=totals, caps=ledger['caps'], completed_fit_costs_checked=checked,
                failures_and_partial_costs_retained=True, ledger=C.bind(C.DOC/'RESOURCE_LEDGER.json'))


def evaluation_audit(verifier, seed):
    from . import eval_student as E
    p = E.paths(seed)
    need(p['lock'], p['results'], p['raw']/'SCORING_START.json', p['candidate_lock'], p['oracle_result'])
    lock = E.prediction_lock(seed); verifier.verify(lock)
    result = C.read(p['results']); verifier.verify(result)
    score_start = C.read(p['raw']/'SCORING_START.json'); verifier.verify(score_start)
    require(datetime.fromisoformat(score_start['at']) >= datetime.fromisoformat(lock['created_at']), 'Reference scoring preceded prediction lock')
    require(score_start['prediction_lock'] == C.bind(p['lock']), 'Scoring used another prediction lock')
    rows = C.read(p['metadata']); ids = E.validate_membership(rows)
    data = [C.read(p['raw']/(name+'.json')) for name in ('FRAME_METRICS', 'FIXED_ID_METRICS', 'POSE_METRICS')]
    for collection in data:
        require(set(collection) == set(ARMS)|{'R0', 'OLD_REF'}, 'Evaluation arm denominator changed')
        require(all(set(v) == set(ids) for v in collection.values()), 'Evaluation frame denominator changed')
    groups, summaries = E.aggregate(rows, *data)
    numeric_close(summaries, result['groups'])
    numeric_close(E.classify(summaries), result['classification'])
    for group, members in groups.items():
        for before, after in E.PAIRS:
            actual = E.M.paired(data[2][before], data[2][after], members)
            stored = result['contrasts'][group][after+'-minus-'+before]
            numeric_close(actual, {k: stored[k] for k in actual})
    require(len(groups['NATURAL99']) == 99 and len(groups['FULL128']) == 128, 'Primary population changed')
    for arm, value in summaries['FULL128'].items():
        require(value['frames'] == 128 and value['twoD']['corners'] == value['fixed_ID']['corners'] == 985, '2D full denominator changed')
        require(value['twoD']['detected'] == 128 and value['twoD']['matched'] == 120, 'Detector matching parity changed')
    candidate_lock = C.read(p['candidate_lock']); verifier.verify(candidate_lock)
    oracle = C.read(p['oracle_result']); verifier.verify(oracle)
    require(candidate_lock['no_reference_coordinates_read'], 'Oracle candidate generation not GT-free')
    require(datetime.fromisoformat(oracle['created_at']) >= datetime.fromisoformat(candidate_lock['created_at']), 'Oracle scoring preceded candidate freeze')
    require(oracle['actual_selector_unchanged'] and oracle['oracle_not_used_for_training_or_deployment'], 'Oracle isolation absent')
    private = C.read(p['raw']/'ORACLE_METRICS_SELECTIONS_PRIVATE.json')
    choices = private['candidate_metrics']; oracle_checks = 0
    for group, members in groups.items():
        for arm in choices:
            actual, _ = E.M.oracle_for(members, choices[arm], data[2][arm])
            numeric_close(actual, oracle['groups'][group][arm]); oracle_checks += 1
        for baseline in ('R0', 'OLD_REF', 'CLEAN_REF_CLEAR', 'CLEAN_RAW_OCC'):
            actual = E.same_candidate_joint_count(members, choices['CLEAN_REF_OCC'], data[2][baseline])
            require(actual == oracle['groups'][group]['CLEAN_REF_OCC_joint_candidate_vs_baseline'][baseline], 'Joint oracle mixes different candidates')
    return dict(full_frames=128, natural_occlusion_frames=99, reference_corners=985, detected=128, matched=120,
        aggregate_and_paired_groups_recomputed=len(groups), full_candidate_vector_oracles_recomputed=oracle_checks,
        prediction_lock_before_reference_scoring=True, original_D9_and_frozen_detector_verified=True,
        no_new_inference=True, scope='Reaggregation of bound per-frame metrics; not new independent physical GT or new geometric solver execution')


def followup_audit(verifier):
    """The only approved repeat: two OCC arms at43, not another four-arm trial."""
    if not (C.DOC/'REPLICATION_DECISION.json').exists():
        raise NotReady('Replication branch decision not yet present')
    import torch
    from . import followup_pair as F, eval_student as E
    from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import pose_parameter
    from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M
    stage = 'REPEAT_PRIMARY_S43'; context = F.StageContext(stage)
    names = ('PRIMARY_PROTOCOL.json', 'PREFLIGHT.json', 'PAIR_INTEGRITY_S43.json', 'EVAL_RESULTS_S43.json',
             'RESOURCE_LEDGER_PREFIT_S43.json', 'PREFLIGHT_ENVIRONMENT_AND_LEDGER_NOTE.md')
    need(*(context.DOC/name for name in names), *(context.DOC/f'FIT_{arm}_S43.json' for arm in F.PAIR_ARMS))
    protocol = C.read(context.DOC/'PRIMARY_PROTOCOL.json')
    # This binding records historical accounting, not a promise the live ledger never changes.
    snapshot = context.DOC/'RESOURCE_LEDGER_PREFIT_S43.json'
    old_binding = protocol['global_resource_ledger']
    require(old_binding['path'] == str((C.DOC/'RESOURCE_LEDGER.json').relative_to(C.ROOT)), 'Unexpected historical ledger target')
    require(C.sha(snapshot) == old_binding['sha256'] and snapshot.stat().st_size == old_binding['bytes'], 'Historical prefit ledger snapshot mismatch')
    verifier.verify({k:v for k,v in protocol.items() if k != 'global_resource_ledger'})
    verifier.verify(C.bind(snapshot))
    require(protocol['seeds'] == [43] and set(protocol['arms']) == set(F.PAIR_ARMS), 'Repeat seed/arm expansion')
    original = C.read(C.DOC/'PRIMARY_PROTOCOL.json')
    for key in ('args', 'masking', 'initialization', 'updates_per_fit', 'epochs', 'trainable_policy'):
        if key in original:
            numeric_close(original[key], protocol[key])
    pre = C.read(context.DOC/'PREFLIGHT.json'); verifier.verify(pre)
    require(pre['passed'] and pre['seed43_actual_stream_matches_prior_preflight'], 'Repeat preflight missing')
    base_model = torch.load(C.ROOT/protocol['initialization']['path'], map_location='cpu', weights_only=False)['model'].float()
    base = base_model.state_dict(); allowed = {name for name,_ in base_model.named_parameters() if pose_parameter(name)}
    protected = set(base)-allowed
    require((len(base),len(protected),len(allowed)) == (879,747,132), 'Repeat state inventory changed')
    traces = {}; states = {}; stream_changes = {}
    for arm in F.PAIR_ARMS:
        fit = C.read(context.DOC/f'FIT_{arm}_S43.json'); verifier.verify(fit)
        require(fit['complete'] and fit['optimizer_steps'] == 320 and fit['exact_R0_initialization'] and fit['protected_state_exact'], 'Repeat fit incomplete')
        require(fit['initialization'] == protocol['initialization'], 'Repeat R0 initialization changed')
        final = torch.load(C.ROOT/fit['checkpoint']['path'], map_location='cpu', weights_only=False)['model'].float().state_dict()
        require(set(final) == set(base) and all(torch.equal(final[k], base[k]) for k in protected), 'Repeat protected tensor/buffer changed')
        changed = {k for k in base if not torch.equal(base[k], final[k])}
        require(changed and changed <= allowed and changed == set(fit['changed_tensors']), 'Repeat actual trainable scope mismatch')
        prior_fit = C.read(C.DOC/f'FIT_{arm}_S42.json')
        prior_state = torch.load(C.ROOT/prior_fit['checkpoint']['path'], map_location='cpu', weights_only=False)['model'].float().state_dict()
        different = sum(not torch.equal(final[k], prior_state[k]) for k in base)
        require(different > 0, 'Repeat final model is bit-identical to42')
        trace = C.read(C.ROOT/fit['trace']['path']); require(len(trace) == 320, 'Repeat trace is not320 updates')
        require([r['steps'] for r in fit['history']] == [64,128,192,256,320], 'Repeat epoch update history changed')
        prefix = C.read(context.RAW/f'PREFLIGHT_TRACE_S43_{fit["target"]}_OCC_PRIVATE.json')
        for actual, expected in zip(trace[:8], prefix):
            for key in ('names','before_images','after_images','boxes','support','coordinates','batch_idx','roles','transfer'):
                require(actual[key] == expected[key], 'Repeat actual preflight prefix mismatch '+arm+'/'+key)
        old_trace = C.read(C.RAW/f'TRACE_{arm}_S42.json')
        differences = {key:sum(a[key] != b[key] for a,b in zip(trace,old_trace)) for key in ('names','before_images','images')}
        differences['plan_batches'] = sum([v['plan'] for v in a['transfer']] != [v['plan'] for v in b['transfer']] for a,b in zip(trace,old_trace))
        require(all(v > 0 for v in differences.values()), 'Repeat effective streams did not change')
        traces[arm] = trace; stream_changes[arm] = differences
        states[arm] = dict(initial_state_tensors=879,initial_equality_evidence='runtime assertion plus bound R0, not persisted step0 snapshot',
            protected_exact=747,changed_final_tensors=len(changed),final_tensors_different_from42=different,updates=320,actual_first8_match_preflight=True)
    parity = C.read(context.DOC/'PAIR_INTEGRITY_S43.json'); verifier.verify(parity)
    actual = F.pair_trace_checks(*(traces[arm] for arm in F.PAIR_ARMS), full=True)
    numeric_close(actual, {key:parity[key] for key in actual})
    numeric_close(stream_changes, parity['actual_stream_different_from_seed42'])
    paths = F.evaluation_paths(context,43); need(paths['lock'],paths['result'],paths['candidate_lock'])
    lock = C.read(paths['lock']); verifier.verify(lock)
    require(lock['configured_new_arms'] == 2 and lock['no_evaluation_reference_coordinates_read'] and lock['original_D9'], 'Repeat prediction boundary changed')
    result = C.read(paths['result']); verifier.verify(result)
    rows = C.read(paths['metadata']); E.validate_membership(rows)
    collections = [C.read(paths['raw']/(name+'.json')) for name in ('FRAME_METRICS','FIXED_ID_METRICS','POSE_METRICS')]
    expected_arms = set(F.PAIR_ARMS)|{'R0','OLD_REF','BASE_RAW_OCC','BASE_REF_OCC'}
    require(all(set(data) == expected_arms for data in collections), 'Repeat evaluation arm set changed')
    groups, summaries = E.aggregate(rows,*collections); numeric_close(summaries,result['groups'])
    for arm, row in summaries['FULL128'].items():
        require(row['frames'] == 128 and row['twoD']['corners'] == 985 and row['twoD']['matched'] == 120, 'Repeat full denominator changed '+arm)
    for group, group_rows in result['contrasts'].items():
        for key, stored in group_rows.items():
            after,before = key.split('-minus-')
            actual = M.paired(collections[2][before],collections[2][after],groups[group])
            numeric_close(actual,{k:stored[k] for k in actual})
    for baseline, stored in result['classification'].items():
        numeric_close(M.classify_candidate(summaries['NATURAL99']['CLEAN_REF_OCC'],summaries['NATURAL99'][baseline]),stored)
    candidate_lock = C.read(paths['candidate_lock']); verifier.verify(candidate_lock)
    require(candidate_lock['no_reference_coordinates_read'] and candidate_lock['existing_D9_current_parity'] == 256, 'Repeat candidate boundary changed')
    need(paths['raw']/'CANDIDATE_METRICS_LOCK.json')
    verifier.verify(C.read(paths['raw']/'CANDIDATE_METRICS_LOCK.json'))
    return dict(states=states,full320_paired_trace_recomputed=True,effective_stream_changes=stream_changes,
        natural99=99,full128=128,corners=985,matched=120,all_aggregate_and_paired_groups_recomputed=True,
        historical_live_ledger_binding_resolved_to_byte_exact_prefit_snapshot=C.bind(snapshot),
        original_protocol_unchanged=True,environment_retry_note=C.bind(context.DOC/'PREFLIGHT_ENVIRONMENT_AND_LEDGER_NOTE.md'),
        GT_boundary='Runtime no-reference assertion plus separate infer/score functions and immutable prediction/candidate locks; no independent persisted file-access log for D9 scoring',
        independent_dataset_confirmation=False)


def selector_and_provenance_audit(verifier):
    from . import eval_student as E
    from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M
    need(*(C.DOC/f'SELECTOR_PAIR_RESULTS_S{seed}.json' for seed in (42,43)), C.DOC/'SELECTOR_SUPERVISION_PROVENANCE.json')
    summaries_checked = decisions_checked = 0; shared_scorer = None
    for seed in (42,43):
        result = C.read(C.DOC/f'SELECTOR_PAIR_RESULTS_S{seed}.json'); verifier.verify(result)
        require(result['matched_common_selector'] and result['unchanged_2D'] and result['reference42_preserved'], 'Common selector pair contract changed')
        require(result['new_selector_fits'] == result['new_student_fits'] == 0, 'Compatibility unexpectedly fitted')
        if shared_scorer is not None:
            require(result['scorer'] == shared_scorer, 'Different scorer across seeds')
        shared_scorer = result['scorer']
        pair_metrics = C.read(C.ROOT/result['private_artifacts'][0]['path'])
        metadata_binding = next(b for b in result['sources'] if Path(b['path']).name == 'METADATA.json')
        rows = C.read(C.ROOT/metadata_binding['path']); groups = E.group_ids(rows)
        require(len(groups['FULL128']) == 128 and len(groups['NATURAL99']) == 99, 'Selector populations changed')
        for target in ('RAW','REF'):
            child = result['children'][target]; lock = C.read(C.ROOT/child['lock']['path']); verifier.verify(lock)
            child_result = C.read(C.ROOT/child['result']['path']); verifier.verify(child_result)
            require(lock['read_guard_active'] and lock['reference_or_metrics_read_before_decisions'] is False, 'Selector reference-read boundary absent')
            require(lock['old_scorer'] == shared_scorer and lock['counts']['frames'] == 128, 'Selector scorer/denominator changed')
            require(datetime.fromisoformat(child_result['created_at']) >= datetime.fromisoformat(lock['created_at']), 'Selector scored before names were locked')
            private = {Path(b['path']).stem:C.read(C.ROOT/b['path']) for b in lock['files'] if Path(b['path']).stem in ('DECISIONS','POSES')}
            candidate_binding = next(b for b in lock['sources'] if Path(b['path']).name == 'CANDIDATES.json')
            candidates = C.read(C.ROOT/candidate_binding['path'])[lock['arm']]
            require(set(private['DECISIONS']) == set(groups['FULL128']), 'Missing selector decision')
            for fid, decision in private['DECISIONS'].items():
                candidate = candidates[fid]
                selected = next((v for v in candidate['hypotheses'] if v['name'] == decision['selected']),None)
                expected = selected['pose'] if selected is not None else candidate['current']
                numeric_close(expected,private['POSES'][fid])
                require(decision['changed'] == (decision['selected'] != candidate['selected_name']), 'Incorrect switch flag')
                if not decision['changed']:
                    numeric_close(pair_metrics[target+'_D9'][fid],pair_metrics[target+'_GEO'][fid])
                decisions_checked += 1
            require(sum(d['changed'] for d in private['DECISIONS'].values()) == lock['counts']['changed'], 'Selector changed count mismatch')
        for group,members in groups.items():
            require(all(set(data) == set(groups['FULL128']) for data in pair_metrics.values()), 'Selector full per-frame denominator changed')
            for arm, data in pair_metrics.items():
                numeric_close(M.summarize(data[fid] for fid in members), result['groups'][group][arm]); summaries_checked += 1
            for key,stored in result['paired'][group].items():
                after,before = key.split('-minus-')
                numeric_close(M.paired(pair_metrics[before],pair_metrics[after],members),stored)
        for key,stored in result['classification_NATURAL99'].items():
            after,before = key.split('-minus-')
            numeric_close(M.classify_candidate(result['groups']['NATURAL99'][after],result['groups']['NATURAL99'][before]),stored)
    provenance = C.read(C.DOC/'SELECTOR_SUPERVISION_PROVENANCE.json'); verifier.verify(provenance)
    old,current = provenance['old_GEO']['records'],provenance['current_students']['records']
    sets = lambda rows: ({r['id'] for r in rows},{r['image']['sha256'] for r in rows},{(r['id'],p) for r in rows for p in r['manual_corner_ids']})
    old_sets,current_sets = sets(old),sets(current)
    require(tuple(map(len,old_sets)) == (10,10,48) and tuple(map(len,current_sets)) == (9,9,38), 'Historical supervision count mismatch')
    require(all(not a&b for a,b in zip(old_sets,current_sets)), 'Old/current manual supervision overlap misreported')
    require(tuple(len(a|b) for a,b in zip(old_sets,current_sets)) == (19,19,86), 'Manual supervision union mismatch')
    require(provenance['old_GEO']['old_wood9_39_in_selector_feature_predictor_chain'] is False, 'Wrong material added to selector ancestry')
    require(provenance['direct_new_manual_corners'] == provenance['direct_new_manual_images'] == 0, 'New manual supervision unexpected')
    return dict(seeds=[42,43],same_frozen_scorer=shared_scorer,fixed_candidate_decisions_checked=decisions_checked,
        aggregate_summaries_recomputed=summaries_checked,all_paired_and_classification_values_recomputed=True,
        traceable_manual_union=dict(images=19,corners=86,new_images=0,new_corners=0),old_wood_not_dependency=True,
        no_GT_selector_input=True,no_new_fit=True,independent_TEST=False)


def augmented_train_audit(verifier):
    from . import train_augmented_following as A
    need(A.REPORT,A.LOCK)
    result,lock = C.read(A.REPORT),C.read(A.LOCK)
    verifier.verify(result); verifier.verify(lock)
    require(lock['teacher_or_evaluation_GT_read'] is False and lock['new_fits'] == lock['optimizer_updates'] == 0,
            'Augmented TRAIN diagnostic crossed no-GT/no-fit boundary')
    require(lock['prefit_trace_reconstruction_exact'] and lock['same_condition_detector_parity'], 'Augmented input/detector contract missing')
    require(lock['source_inference'] is False and result['coordinate_space'] == 'augmented640x640pixels, not nativepx', 'Augmented units/domain mislabeled')
    rows = C.read(A.OUT/'TARGETS_PRIVATE.json')
    residuals = C.read(C.ROOT/result['private_artifacts'][0]['path'])
    require(len(rows) == result['real_occurrences'] == 62 and len({row['name'] for row in rows}) == result['real_unique'] == 45,
            'Augmented TRAIN prefix population changed')
    support = sum(sum(row['support']) for row in rows)
    covered = sum(len(row['canonical_REF_planned_covered']) for row in rows)
    require((support,covered) == (476,21), 'Augmented common support counts changed')
    predictions = {}
    expected_rows = {row['id']:row for row in rows}
    for arm in A.ARMS:
        stored = C.read(A.OUT/(arm+'.json')); verifier.verify(stored['stamp'])
        require(stored['state_unchanged'] and stored['gradients_absent'] and stored['optimizer_constructed'] is False,
                'Augmented frozen model diagnostic changed state or built optimizer')
        predictions[arm] = stored['predictions']
        require(set(predictions[arm]) == set(expected_rows), 'Augmented occurrence denominator changed')
    for point in residuals:
        row = expected_rows[point['id']]; corner = point['corner']
        require(0 <= corner < 8 and row['support'][corner], 'Ignored/center point included in diagnostic')
        pred = predictions[point['arm']][point['id']]
        if pred['candidates']:
            require(pred['selected_index'] == int(np.argmax([v['score'] for v in pred['candidates']])), 'Target-nearest detector selection')
            q = np.asarray(pred['candidates'][pred['selected_index']]['keypoints_xy'])[corner]
            delta = q-np.asarray(row[point['target']][corner])
            require(point['missing'] == (not np.isfinite(delta).all()), 'Missing residual flag incorrect')
            if not point['missing']:
                numeric_close(float(np.abs(delta).mean()),point['l1_mean_xy_px'])
                numeric_close(float(np.linalg.norm(delta)),point['l2_px'])
        else:
            require(point['missing'] and pred['selected_index'] is None, 'Missing prediction excluded')
        expected_split = 'CANONICAL_REF_PLANNED_COVERED' if corner in row['canonical_REF_planned_covered'] else 'CANONICAL_REF_UNMASKED'
        require(point['split'] == expected_split, 'Canonical REF grouping changed')
    summaries = 0
    for group,arms in result['groups'].items():
        for arm,targets in arms.items():
            for target,splits in targets.items():
                pp = [p for p in residuals if p['arm'] == arm and p['target'] == target and (group == 'ALL' or p['recording'] == group)]
                for split,stored in splits.items():
                    actual = A.summary(pp if split == 'ALL_SUPERVISED' else [p for p in pp if p['split'] == split])
                    numeric_close(actual,stored); summaries += 1
    return dict(real_occurrences=62,unique_real_images=45,supervised_corner_occurrences=476,
        canonical_REF_planned_covered=21,canonical_REF_unmasked=455,scalar_residuals_recomputed=len(residuals),
        group_summaries_recomputed=summaries,no_new_fit=True,no_evaluation_GT=True,
        units='Augmented640 pixels, not native image pixels',
        scope='Own CLEAR/OCC input per corresponding frozen student; input difficulty and student differences are not separated.')


def forbidden_public_paths(paths):
    raw, out = [str(p.relative_to(C.ROOT))+'/' for p in (C.RAW, C.OUT)]
    scoped = (str(C.DOC.relative_to(C.ROOT))+'/',
              'scripts/research/'+C.NAME+'/', raw, out)
    return [p for p in paths if p.startswith(scoped) and
            (p.startswith((raw, out)) or 'PRIVATE' in Path(p).name.upper() or
             Path(p).suffix.lower() in ('.pt', '.pth', '.ckpt', '.npy', '.npz', '.pkl'))]


def privacy_fields(value, location='$'):
    private_keys = {'keypoints', 'keypoints_xy', 'pred_xy', 'target_xy', 'manual_xy',
                    'pixels', 'rgb', 'R', 'K', 't', 'bbox', 'box_xyxy', 'state_dict'}
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in private_keys and isinstance(child, (list, dict)):
                found.append(location+'.'+key)
            found += privacy_fields(child, location+'.'+key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found += privacy_fields(child, f'{location}[{index}]')
    return found


def publication_audit(verifier):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=C.ROOT, text=True).splitlines()
    scopes = [str(p.relative_to(C.ROOT)) for p in (C.DOC, C.RAW, C.OUT)]
    scopes.append('scripts/research/'+C.NAME)
    staged = git('diff', '--cached', '--name-only')
    tracked = git('ls-files', '--', *scopes)
    bad = forbidden_public_paths(staged+tracked)
    scope = dict(created_at=C.now(), staged_paths=staged, tracked_paths=tracked,
                 checked_with='git diff --cached --name-only; git ls-files -- new namespace scopes',
                 no_git_mutation=True, forbidden_paths=sorted(set(bad)))
    C.save(C.DOC/'PUBLICATION_SCOPE.json', scope)
    require(not bad, 'Private/checkpoint/raw-output path is staged or tracked: '+repr(bad))
    scanned = 0
    for path in C.DOC.rglob('*.json'):
        if path.name in ('AUDIT.json', 'PUBLICATION_SCOPE.json'):
            continue
        require(not privacy_fields(C.read(path)), 'Named private payload fields in public JSON: '+path.name)
        scanned += 1
    return dict(scope=C.bind(C.DOC/'PUBLICATION_SCOPE.json'), staged_files=len(staged),
        tracked_new_namespace_files=len(tracked), public_JSON_scanned=scanned, private_raw_checkpoint_files_published=0,
        caveat='Point-in-time index/tracked scope; rerun after final staging. Named-field scan is not arbitrary encoded-payload detection.')


def local_links(text, parent):
    missing = []
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
        target = target.strip().split(' "')[0].strip('<>')
        if re.match(r'^[a-zA-Z]+:', target) or target.startswith('#'):
            continue
        target = unquote(target.split('#')[0])
        if target and not (parent/target).exists():
            missing.append(target)
    return missing


def report_audit(verifier, seed):
    required = ['REPORT_KO.md', 'REPORT_BUILD.json', 'CONTRACT_DIFFERENCE_AUDIT.md',
        'CLEAN_POOL_AUDIT.md', 'OLD_CLEAN19_POSE_REFERENCE.md',
        f'PRIMARY_ANALYSIS_S{seed}.json', f'PRIMARY_ANALYSIS_S{seed}.md', f'TRAIN_TARGET_FOLLOWING_S{seed}.json']
    need(*(C.DOC/name for name in required))
    build = C.read(C.DOC/'REPORT_BUILD.json'); verifier.verify(build)
    require(build['passed'] and build.get('artifacts') and build.get('inputs'), 'Report build provenance incomplete')
    if build.get('status') != 'FINAL':
        raise NotReady('Report explicitly remains NOT_FINAL')
    analyses = C.read(C.DOC/f'PRIMARY_ANALYSIS_S{seed}.json'); verifier.verify(analyses)
    require(analyses['status'] == 'COMPLETE_WITH_ORACLE', 'Final analysis lacks candidate diagnosis')
    target = C.read(C.DOC/f'TRAIN_TARGET_FOLLOWING_S{seed}.json'); verifier.verify(target)
    require(target['images'] == 78, 'TRAIN target-following population changed')
    require('TARGET_FOLLOWING' in target['status'], 'TRAIN pseudo fit mislabeled as physical accuracy')
    markdown = sorted(C.DOC.rglob('*.md'))
    broken = {path.name: local_links(path.read_text(), path.parent) for path in markdown}
    require(not any(broken.values()), 'Broken report links: '+repr({k:v for k,v in broken.items() if v}))
    registered = {}
    for path in C.DOC.glob('*.json'):
        if path.name in ('AUDIT.json', 'PUBLICATION_SCOPE.json'):
            continue
        for row in bindings(C.read(path)):
            if Path(row['path']).suffix.lower() in ('.png', '.svg', '.jpg', '.jpeg'):
                registered[row['path']] = row
    required_figures = [C.DOC/'figures/old_clean19_pose_reference.png', C.DOC/f'figures/primary_pose_transfer_S{seed}.png']
    need(*required_figures)
    figures = [p for p in (C.DOC/'figures').rglob('*') if p.is_file() and p.suffix.lower() in ('.png', '.svg', '.jpg', '.jpeg')]
    for figure in figures:
        key = str(figure.relative_to(C.ROOT))
        require(key in registered, 'Unregistered public figure: '+key)
        verifier.verify(registered[key])
    return dict(markdown_files_with_links_checked=len(markdown), registered_figures_checked=len(figures),
        report_build=C.bind(C.DOC/'REPORT_BUILD.json'), full_analysis_and_native_TRAIN_target_following_present=True)


def run(seed=42):
    start = time.monotonic(); verifier = Verifier(); checks = {}
    operations = [('source_protocol_preflight', lambda: source_gate(verifier)),
        ('supplemental_dependency', lambda: supplemental_dependency(verifier)),
        ('four_fits_and_full_trace', lambda: checkpoint_and_trace_audit(verifier, seed)),
        ('goal_budget', lambda: budget_audit(verifier)),
        ('evaluation_and_oracles', lambda: evaluation_audit(verifier, seed)),
        ('effective_seed43_pair', lambda: followup_audit(verifier)),
        ('common_selector_and_manual_ancestry', lambda: selector_and_provenance_audit(verifier)),
        ('augmented_TRAIN_corner_diagnostic', lambda: augmented_train_audit(verifier)),
        ('publication_scope', lambda: publication_audit(verifier)),
        ('report_and_figures', lambda: report_audit(verifier, seed))]
    for name, operation in operations:
        try:
            checks[name] = dict(status='PASS', details=operation())
        except (NotReady, FileNotFoundError) as error:
            checks[name] = dict(status='NOT_READY', reason=str(error))
        except Exception as error:
            checks[name] = dict(status='FAIL', error=type(error).__name__+': '+str(error))
        print('AUDIT', name, checks[name]['status'], flush=True)
    result = dict(status=final_status(checks), created_at=C.now(), seed=seed, checks=checks,
        unique_bindings_verified=len(verifier.checked), CPU_seconds=time.monotonic()-start,
        GPU_seconds=0, new_fits=0, optimizer_updates=0, historical_writes=0, git_mutations=0,
        implementation=C.bind(Path(__file__)),
        interpretation='PASS means completed declared checks, not improvement, independent physical GT, or a publication guarantee after further changes.')
    C.save(C.DOC/'AUDIT.json', result)
    print('FINAL_AUDIT', result['status'], 'bindings', len(verifier.checked), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args().seed)
