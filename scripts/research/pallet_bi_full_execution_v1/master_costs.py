"""Read-only, typed campaign cost accounting after honest D4 recovery.

No numerical package, dataset, image, bank, model or reference is decoded.
Only explicit receipt paths and ROW/job metadata below two declared bases are
read. Completed older attempts are reused through their bound metadata audit.
The CLI requires a source/catalog-bound ROOT GO; importing this file does not
run an audit. Unknown costs remain unknown after successful recovery.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

UNKNOWN = 'UNKNOWN_NOT_ZERO'
SCHEMA = 'RECOVERY_AWARE_MASTER_COSTS_V1'
SEEDS = (20261018, 20261019, 20261020)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value):
    require(type(value) is int and value >= 0, 'Expected a nonnegative integer counter')
    return value


def duration(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'Expected a finite nonnegative timer')
    return value


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'Receipt-relative path escaped its declared root')
    return path


class Reader:
    """Hash every metadata input; optional expected hash authenticates ancestry."""
    def __init__(self):
        self.bindings = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, 'Changed receipt/source: ' + str(path))
        require(str(path) not in self.bindings or self.bindings[str(path)] == digest,
                'Input changed during audit: ' + str(path))
        self.bindings[str(path)] = digest
        return digest

    def read(self, path, expected=None):
        self.bind(path, expected)
        return json.loads(Path(path).read_text())

    def lines(self, path, expected=None):
        self.bind(path, expected)
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    def stable(self):
        for path, digest in self.bindings.items():
            require(sha(path) == digest, 'Input changed before output seal: ' + path)


def closed_attempts(reader, spec):
    d = reader.read(spec['path'], spec['sha256'])
    require(d['schema'] == 'CLOSED_PRIOR_CORE_ATTEMPTS_METADATA_CHECK_V1'
            and d['active_D4_totals_included'] is False and d['new_scientific_calls'] == 0,
            'Closed-attempt audit has wrong scope')
    r = d['report']
    require(r['cost_complete'] is True, 'Historical closed attempts are not closed')
    roots, attempts, total = set(), [], 0
    for a in r['attempts']:
        root = str(Path(a['root']).resolve())
        require(root not in roots and a['cost_complete'] is True, 'Repeated/nonclosed historical attempt')
        roots.add(root)
        calls = count(a['actual_optimizer_calls'])
        require(calls == a['known_optimizer_calls'], 'Closed current-call identity differs')
        for p, digest in a['evidence'].items():
            reader.bind(p, digest)  # hashes only; never decode copied FRAME/RESULT content
        total += calls
        attempts.append({k: a[k] for k in ('name', 'root', 'actual_optimizer_calls', 'cost_complete')})
    require(total == r['actual_optimizer_calls'] == d['expected_closed_call_sum'],
            'Closed-attempt sum differs')
    if 'expected_calls' in spec:
        require(total == spec['expected_calls'], 'Reviewed closed-attempt total differs')
    return dict(known_calls=total, counted_roots=sorted(roots), attempts=attempts,
                prior_S2=r['preserved_prior_S2'],
                timing_by_attempt=[dict(name=a['name'], timing=a.get('timing', {})) for a in r['attempts']],
                actual_family_dispatches=UNKNOWN,
                scope='Runtime minimize calls from the six closed attempts; each root once')


def recovery_attempt(reader, spec):
    base, invocation = Path(spec['base']).resolve(), Path(spec['invocation']).resolve()
    require(base != invocation and not invocation.is_relative_to(base), 'Recovery journal must be independent')
    lock = reader.read(base/'PRODUCER_LOCK.json')
    inv = reader.read(invocation/'INVOCATION_LOCK.json')
    before = reader.read(invocation/'HISTORICAL_STATE.json')
    require(inv['original_output'] == str(base) and inv['delegated_initializer_output'] == str(base)
            and inv['delegated_execute_output'] == str(invocation), 'Recovery ownership differs')
    require(inv['workers'] == 4 and lock['workers'] == 4, 'Wrong D4 scheduling metadata')
    reader.bind(base/'PRODUCER_LOCK.json', inv['original_producer_lock_sha256'])
    reader.bind(inv['approved_GO'], inv['approved_GO_sha256'])
    source = inv['source_binding']
    if spec.get('recovery_source'):
        reader.bind(spec['recovery_source'], source['recovery_controller_sha256'])
    require(source['delegated_D4_source_binding']['row_helper_sha256'] == lock['row_helper_sha256']
            and source['delegated_D4_source_binding']['numerical_kernel_source_binding']
            == lock['numerical_kernel_source_binding'], 'Recovery numerical ancestry differs')
    ordered = lock['static_ordered_row_ids']
    expected = {(domain, ordinal): fid for domain, ids in ordered.items() for ordinal, fid in enumerate(ids)}
    arms = lock['arms']
    require(arms == spec['arms'] and len(expected) == inv['total_jobs'] == spec.get('expected_jobs', 1241),
            'Recovery denominator/arms differs')
    # The original parent journal was interrupted. Authenticate its actual,
    # intentionally partial map; never require 1241 old envelopes.
    for rel, digest in inv['original_row_jobs_sha256'].items():
        reader.bind(inside(base/'row_jobs', rel), digest)
    original_jobs = {str(p.relative_to(base/'row_jobs')) for p in (base/'row_jobs').rglob('*') if p.is_file()}
    require(original_jobs == set(inv['original_row_jobs_sha256']), 'Original parent journal changed')
    for rel, digest in before['completed_artifacts_sha256'].items():
        reader.bind(inside(base, rel), digest)
    prior = count(before['known_stored_new_optimizer_calls'])
    require(before['interrupted_prior_optimizer_calls'] == UNKNOWN, 'Lost historical work became zero')
    historical_unknowns = before['incomplete_arm_directories']
    require(all(x['prior_optimizer_progress'] == UNKNOWN for x in historical_unknowns), 'Unknown arm cost erased')
    completion = invocation/'RECOVERY_COMPLETION.json'
    result = dict(name=spec['name'], base=str(base), invocation=str(invocation),
        old_parent_job_envelopes=len(original_jobs), expected_new_parent_job_envelopes=len(expected),
        prior_stored_known_calls=prior, interrupted_historical_optimizer_calls=UNKNOWN,
        historical_unknown_arm_directories=historical_unknowns,
        old_prefix_is_included_in_cumulative_base=True, copied_FRAME_intrinsic_calls_added=0,
        finalization_outside_controller_timer=UNKNOWN,
        startup_before_import_timestamp=UNKNOWN)
    if not completion.exists():
        result.update(complete=False, status='PENDING_RECOVERY_COMPLETION',
                      known_base_call_lower_bound=prior, current_invocation_calls=UNKNOWN,
                      base_calls_cumulative=UNKNOWN, actual_optimizer_calls=UNKNOWN)
        return result
    c = reader.read(completion)
    for field, leaf in [('invocation_lock_sha256','INVOCATION_LOCK.json'),
                        ('historical_state_sha256','HISTORICAL_STATE.json'),
                        ('unsealed_history_sha256','UNSEALED_HISTORY.json'),
                        ('dispatch_result_sha256','DISPATCH_RESULT.json')]:
        reader.bind(invocation/leaf, c[field])
    require(c['original_output'] == str(base) and c['invocation'] == str(invocation)
            and c['recovery_controller_sha256'] == source['recovery_controller_sha256']
            and c['original_producer_lock_sha256'] == inv['original_producer_lock_sha256'],
            'Recovery completion ancestry differs')
    require(c['interrupted_historical_optimizer_calls'] == UNKNOWN
            and c['total_campaign_actual_calls'] == UNKNOWN
            and c['copied_historical_intrinsic_calls_added'] == 0
            and c['prior_stored_known_calls'] == prior, 'Recovery erased/duplicated historical cost')
    for rel, digest in c['job_envelopes_sha256'].items():
        reader.bind(inside(invocation/'row_jobs', rel), digest)
    actual_job_paths = {str(p.relative_to(invocation/'row_jobs')) for p in (invocation/'row_jobs').rglob('*') if p.is_file()}
    require(actual_job_paths == set(c['job_envelopes_sha256']), 'Recovery job set changed')
    if c['complete'] is not True:
        result.update(complete=False, status='INCOMPLETE_RECOVERY', actual_optimizer_calls=UNKNOWN,
            current_invocation_calls=UNKNOWN, base_calls_cumulative=UNKNOWN,
            current_returned_job_calls_lower_bound=count(c['successful_returned_jobs_new_optimizer_calls_lower_bound']),
            known_base_call_lower_bound=prior + count(c['successful_returned_jobs_new_optimizer_calls_lower_bound']),
            current_unattributed_work=c['current_invocation_unattributed_optimizer_calls'])
        return result
    dispatch = c['dispatch']
    require(c['cost_identity_verified'] is True and c['technical_failures'] == []
            and c['original_completed_artifacts_preserved'] is True
            and c['original_parent_row_jobs_preserved'] is True
            and c['current_invocation_call_count_status'] == 'EXACT_FULL_ROW_IDENTITY'
            and c['current_invocation_unattributed_optimizer_calls'] == 0
            and dispatch == reader.read(invocation/'DISPATCH_RESULT.json')
            and dispatch['submitted_jobs'] == len(expected) and dispatch['unsubmitted_jobs'] == 0
            and dispatch['checkpoint_stop_observed'] is False and dispatch['active_rows_drained_before_exit'] is True,
            'Recovery not completely drained/identity verified')
    seen, row_paths, cumulative, new_calls = set(), set(), 0, 0
    row_CPU, row_wall, job_CPU, job_wall = 0., 0., 0., 0.
    by_arm = {arm: 0 for arm in arms}
    for rel, digest in c['job_envelopes_sha256'].items():
        job = reader.read(inside(invocation/'row_jobs', rel), digest)
        key = job['domain'], job['ordinal']
        require(key not in seen and expected.get(key) == job['id'], 'Repeated/wrong recovery job')
        require(rel == f'{key[0]}/{key[1]:04d}.json', 'Recovery envelope path differs')
        seen.add(key)
        require([r['arm'] for r in job['records']] == arms, 'Recovery job arm order differs')
        for record in job['records']:
            p = base/record['arm']/key[0]/f'{key[1]:04d}'/'ROW_EXECUTION.json'
            require(str(p) == record['path'] and str(p) not in row_paths, 'Repeated/wrong base ROW account')
            row_paths.add(str(p))
            row = reader.read(p, record['sha256'])
            require(row == record['receipt'] and row['complete'] is True
                    and (row['domain'], row['ordinal'], row['id'], row['arm'])
                    == (key[0], key[1], job['id'], record['arm']), 'ROW account identity differs')
            require(row['driver_sha256'] == lock['row_helper_sha256']
                    and row['numerical_kernel_source_binding'] == lock['numerical_kernel_source_binding'],
                    'ROW numerical ancestry differs')
            reader.bind(p.parent/'FRAME_SEAL.json', row['original_frame_seal_sha256'])
            calls = count(row['resource']['actual_new_pose_optimizer_calls'])
            cumulative += calls
            by_arm[record['arm']] += calls
            row_CPU += duration(row['resource']['current_execution_CPU_seconds'])
            row_wall += duration(row['resource']['current_execution_wall_seconds'])
        new_calls += count(job['actual_new_optimizer_calls_this_job'])
        job_CPU += duration(job['current_job_CPU_seconds'])
        job_wall += duration(job['current_job_wall_seconds'])
    require(seen == set(expected), 'Recovery must bind every original row')
    require(new_calls == c['actual_new_optimizer_calls_this_invocation']
            == c['successful_returned_jobs_new_optimizer_calls_lower_bound']
            and cumulative == c['base_stored_calls_cumulative'] and cumulative - prior == new_calls,
            'Recovery after-before/current/cumulative call identity differs')
    summary = reader.read(base/'SUMMARY.json', c['summary_sha256'])
    require(summary['status'] == 'COMPLETE_NEW' and summary['missing_rows'] == []
            and summary['technical_failures'] == [] and summary['new_pose_optimizer_calls'] == cumulative
            and summary['current_invocation_new_pose_optimizer_calls'] == new_calls
            and summary['producer_lock_sha256'] == inv['original_producer_lock_sha256'],
            'Base summary disagrees with ROW/cumulative accounts')
    end = reader.read(base/'DISPATCH_COMPLETION.json')
    require(end['recovery_completion_sha256'] == reader.bind(completion)
            and end['summary_sha256'] == reader.bind(base/'SUMMARY.json'), 'Base completion lacks recovery parent')
    for arm in arms:
        for domain, ids in ordered.items():
            folder = base/arm/domain
            seal = reader.read(folder/'FEATURE_SEAL.json')
            require(seal['complete'] is True and seal['rows'] == len(ids)
                    and seal['producer_lock_sha256'] == inv['original_producer_lock_sha256'], 'Feature aggregate differs')
            for i in range(len(ids)):
                reader.bind(folder/f'{i:04d}'/'ROW_EXECUTION.json', seal['row_execution_sha256'][f'{i:04d}'])
                reader.bind(folder/f'{i:04d}'/'FRAME_SEAL.json', seal['frame_seal_sha256'][f'{i:04d}'])
            # No FEATURES payload decoding or hashing is needed for current-call counting.
    result.update(complete=True, status='CUMULATIVE_BASE_COST_IDENTITY_VERIFIED',
        known_base_call_lower_bound=cumulative, base_calls_cumulative=cumulative,
        current_invocation_calls=new_calls, actual_optimizer_calls=UNKNOWN,
        by_arm_cumulative_calls=by_arm, new_parent_job_envelopes=len(seen),
        timing=dict(controller={k:c.get(k, UNKNOWN) for k in
            ('wall_seconds','setup_wall_seconds','dispatch_wall_seconds','aggregate_and_verification_wall_seconds',
             'module_import_wall_seconds','module_import_CPU_seconds','controller_CPU_seconds','worker_CPU_seconds','worker_peak_RSS_bytes')},
            current_job_CPU_sum=job_CPU,current_job_wall_sum=job_wall,
            base_ROW_CPU_sum=row_CPU,base_ROW_wall_sum=row_wall,
            relation='Base ROW includes old accounts; current job/worker/controller/import timers overlap or nest. Never sum wall into campaign elapsed.'))
    return result


def head_accounts(reader, run, prediction_specs, expected_fits=33):
    run = Path(run)
    rows = reader.lines(run/'GLOBAL_H1_FIT_ATTEMPTS.jsonl')
    events = reader.lines(run/'H1_EXECUTION_EVENTS.jsonl')
    require(len(rows) <= 36 and all(a['seed'] in SEEDS for a in rows), 'H1 reservation cap/seed protocol differs')
    finished_events = [e for e in events if e.get('event') == 'COMPLETE']
    complete_events = {(e['arm'], e['seed'], e['attempt']):e for e in finished_events}
    require(len(complete_events) == len(finished_events), 'Duplicate H1 COMPLETE event')
    seen, completed, pending, CPU, wall, steps, exposures = set(), [], [], 0., 0., 0, 0
    for a in rows:
        key = a['arm'], a['seed'], a['attempt']
        require(key not in seen and a['registered_before_optimizer'] is True, 'Duplicate/unregistered H1 reservation')
        seen.add(key)
        p = Path(a['output'])/f"{a['arm']}_s{a['seed']}"/'RUN_RECEIPT.json'
        if not p.exists():
            pending.append(dict(arm=a['arm'], seed=a['seed'], attempt=a['attempt'], receipt=str(p), unreceipted_cost=UNKNOWN))
            continue
        r = reader.read(p)
        require(r['complete'] is True and (r['arm'],r['seed'],r['attempt']) == key
                and r['bindings'] == a['bindings'] and r['actual_new_training_runs'] == 1,
                'H1 completed receipt differs from reservation')
        require(key in complete_events, 'Completed H1 receipt lacks global COMPLETE event')
        require(complete_events[key]['receipt_sha256'] == reader.bind(p), 'H1 COMPLETE receipt hash differs')
        require(r['optimizer_steps'] == complete_events[key]['optimizer_steps'], 'H1 steps differ')
        for leaf, field in [('best.pt','checkpoint_sha256'),('last.pt','last_checkpoint_sha256'),('HISTORY.json','history_sha256')]:
            # History is only hashed; no weight/tensor deserialization occurs.
            reader.bind(p.parent/leaf, r[field])
        c = r['cost']
        require(c['threads'] == 1 and c['device'] == 'CPU' and c['GPU_seconds'] == 0, 'H1 execution resource metadata differs')
        CPU += duration(c['CPU_seconds']); wall += duration(c['wall_seconds'])
        steps += count(r['optimizer_steps']); exposures += count(r['training_frame_exposures'])
        completed.append(dict(arm=a['arm'],seed=a['seed'],attempt=a['attempt'],receipt=str(p),
            receipt_sha256=reader.bind(p),resumed_same_optimizer=r['resumed_same_optimizer'],cost=c))
    require(set(complete_events) <= seen, 'H1 COMPLETE lacks registered reservation')
    predictions, forwards, backbone, pred_CPU, pred_wall = [], 0, 0, 0., 0.
    prediction_paths, prediction_arms = set(), set()
    for spec in prediction_specs:
        p = Path(spec['path']).resolve()
        require(str(p) not in prediction_paths and spec['arm'] not in prediction_arms,
                'Repeated H1 prediction arm/path would double count forwards')
        prediction_paths.add(str(p)); prediction_arms.add(spec['arm'])
        if not p.exists():
            predictions.append(dict(arm=spec['arm'],complete=False,path=str(p),cost=UNKNOWN)); continue
        d = reader.read(p)
        require(d['complete'] is True and d['rows'] == 238
                and len(d['checkpoints']) == len(SEEDS) and len(d['ordered_ids']) == 238
                and len(set(d['ordered_ids'])) == 238, 'H1 prediction denominator differs')
        manifest = reader.read(spec['training_manifest'], d['training_manifest_sha256'])
        require(manifest['complete'] is True and len(manifest['runs']) == len(SEEDS)
                and {r['seed'] for r in manifest['runs']} == set(SEEDS)
                and all(r['arm'] == spec['arm'] for r in manifest['runs']), 'Prediction training manifest incomplete')
        expected_names = {f"{spec['arm']}_s{s}" for s in SEEDS}
        require(set(d['checkpoints']) == expected_names, 'H1 prediction seeds differ')
        for r in manifest['runs']:
            rp = Path(spec['training_manifest']).parent/f"{r['arm']}_s{r['seed']}"/'RUN_RECEIPT.json'
            require(reader.read(rp) == r and (r['arm'],r['seed'],r['attempt']) in complete_events,
                    'Manifest copied H1 receipt differs or is unregistered')
            require(d['checkpoints'][f"{r['arm']}_s{r['seed']}"]['checkpoint_sha256'] == r['checkpoint_sha256']
                    and d['checkpoints'][f"{r['arm']}_s{r['seed']}"]['training_binding'] == r['bindings'],
                    'Prediction checkpoint ancestry differs')
        tiny = count(d['tiny_head_forward_calls']); bb = count(d['new_backbone_forwards'])
        require(tiny == 238*len(SEEDS), 'H1 forward unit differs')
        forwards += tiny; backbone += bb
        pred_CPU += duration(d['cost']['CPU_seconds']); pred_wall += duration(d['cost']['wall_seconds'])
        predictions.append(dict(arm=spec['arm'],complete=True,path=str(p),tiny_head_forward_calls=tiny,
                                new_backbone_forwards=bb,cost=d['cost']))
    return dict(reservations=len(rows), completed_new_fits=len(completed), expected_final_new_fits=expected_fits,
        all_expected_fits_complete=len(completed)==expected_fits and not pending,
        all_declared_prediction_seals_complete=bool(predictions) and all(p['complete']for p in predictions),
        completed=completed,pending_reservations=pending, optimizer_steps=steps, training_frame_exposures=exposures,
        internal_fit_CPU_seconds=CPU,internal_fit_wall_seconds_sum=wall,
        tiny_head_real_prediction_forwards=forwards,new_backbone_prediction_forwards=backbone,
        prediction_CPU_seconds=pred_CPU,prediction_wall_seconds_sum=pred_wall,predictions=predictions,
        cost_scope='One cumulative RUN receipt per registered optimizer; manifest/event copies add no cost. Same-optimizer resumes are not new fits. Internal timers are subsets of child/service timers.',
        global_technical_failure_events=[e for e in events if e.get('event')=='TECHNICAL_FAILURE'],
        unreceipted_attempt_cost=UNKNOWN if pending else 0)


def command_ledger(reader, spec):
    p = Path(spec['path'])
    if not p.exists():
        return dict(name=spec['name'],path=str(p),status='NOT_YET_PRESENT', untracked_cost=UNKNOWN)
    rows = reader.lines(p, spec.get('sha256'))
    pending, pairs, wall, CPU, failures = {}, [], 0., 0., 0
    for i, row in enumerate(rows):
        event = row.get('event'); key = row.get('phase')
        if event == 'START':
            require(key not in pending, 'Nested/repeated unfinished ledger phase')
            pending[key] = (i,row)
        elif event in ('COMPLETE','TECHNICAL_FAILURE','INTERRUPTED'):
            require(key in pending, 'Ledger terminal lacks START')
            first,start = pending.pop(key)
            if 'argv' in row and 'argv' in start:
                require(row['argv'] == start['argv'], 'Ledger argv changed')
            w = duration(row['wall_seconds'])
            c = duration(row.get('child_CPU_seconds',row.get('CPU_seconds')))
            wall += w; CPU += c; failures += event != 'COMPLETE'
            pairs.append(dict(phase=key,event=event,start_index=first,end_index=i,wall_seconds=w,CPU_seconds=c))
    return dict(name=spec['name'],path=str(p),status='CLOSED_PAIRS' if not pending else 'OPEN_PHASES',
        pairs=pairs,completed_or_failed_child_calls=len(pairs),failures=failures,
        additive_child_wall_seconds=wall,additive_child_CPU_seconds=CPU,
        unfinished=list(pending),untracked_cost=UNKNOWN if pending else 0,
        parent_scope=spec.get('parent_scope'),relation=spec.get('relation','independent child ledger'),
        elapsed_interpretation='Additive child wall, never campaign elapsed; do not add again to containing service/parent child timer')


def services(reader, spec):
    launches, roles = [], {}
    for p in spec['launches']:
        d = reader.read(p)
        launches.append(str(p))
        for role,s in d['services'].items():
            require(role not in roles, 'Service role launched twice')
            require(s['returncode'] == 0 and '--property=Restart=no' in s['actual_argv']
                    and '--property=CPUAccounting=yes' in s['actual_argv'], 'Service launch/accounting differs')
            roles[role] = s
    folder = Path(spec['lifecycle_root'])
    out = dict(launches=launches,complete=False,scientific_completion_claim=False,
               campaign_elapsed_wall_seconds=UNKNOWN,
               scope='Recovery/post-core service envelope only; does not measure the whole campaign. Launch command duration is not service lifetime.')
    if not (folder/'COMPLETION.json').exists():
        out.update(status='LIFECYCLE_CAPTURE_PENDING',service_CPU_seconds=UNKNOWN,service_elapsed_seconds=UNKNOWN)
        return out
    lock = reader.read(folder/'LOCK.json'); end = reader.read(folder/'COMPLETION.json')
    snapshots = reader.lines(folder/'SNAPSHOTS.jsonl',end['snapshot_sha256'])
    require(end['complete'] is True and end['scientific_completion_claim'] is False and snapshots
            and snapshots[-1] == end['last_snapshot'] and end['source_sha256'] == lock['source_sha256'],
            'Service lifecycle completion ancestry differs')
    unit_map = {s['unit']:s for s in end['last_snapshot']['units']}
    require(len(unit_map) == len(roles), 'Service lifecycle unit count differs')
    entries, cpu_sum, starts, exits = [], 0., [], []
    for role,s in roles.items():
        unit = s['unit'] if s['unit'].endswith('.service') else s['unit']+'.service'
        require(unit in unit_map, 'Service lifecycle is missing a declared unit')
        v = unit_map[unit]
        require(v['returncode'] == 0 and v['SubState'] in ('exited','failed','dead')
                and int(v['NRestarts']) == 0, 'Service not terminal or silently restarted')
        c = count(int(v['CPUUsageNSec']))/1e9
        start,exit_ = int(v['ExecMainStartTimestampMonotonic']),int(v['ExecMainExitTimestampMonotonic'])
        require(start > 0 and exit_ >= start, 'Service lifetime timestamps unavailable')
        elapsed = (exit_-start)/1e6
        cpu_sum += c; starts.append(start); exits.append(exit_)
        entries.append(dict(role=role,unit=unit,CPU_seconds=c,elapsed_wall_seconds=elapsed,
            peak_memory_bytes=v.get('MemoryPeak',UNKNOWN),Result=v['Result'],ExecMainStatus=v['ExecMainStatus'],
            scientific_completion_not_implied_by_exit=True,
            relation='Inclusive service cgroup CPU contains children. Child ledgers and internal fits are subsets, not extra CPU.'))
    out.update(complete=True,status='CAPTURED_SERVICE_ENVELOPE',entries=entries,
        service_CPU_seconds=cpu_sum,
        recovery_service_envelope_elapsed_seconds=(max(exits)-min(starts))/1e6,
        additive_service_elapsed_seconds=sum(x['elapsed_wall_seconds']for x in entries),
        finalization_after_own_internal_timer_is_in_service_CPU=True)
    return out


def optional_metadata(reader, specs):
    """Whitelist scalar receipt fields, retain absence rather than infer zero."""
    result = []
    for s in specs:
        p = Path(s['path'])
        if not p.exists():
            result.append(dict(name=s['name'],path=str(p),status='NOT_YET_PRESENT',cost=UNKNOWN));continue
        d = reader.read(p,s.get('sha256'))
        values = {}
        for dotted in s.get('fields',[]):
            v = d
            for k in dotted.split('.'):
                if not isinstance(v,dict) or k not in v:
                    v = UNKNOWN; break
                v = v[k]
            values[dotted] = v
        result.append(dict(name=s['name'],path=str(p),values=values,scope=s['scope']))
    return result


def campaign_elapsed(reader, catalog):
    """Documented campaign start to an explicit external finalization epoch.

    A measurement cutoff can precede reporting/publication and is labeled as
    such. The audit never invents its own timestamp as whole campaign end.
    """
    start_path = Path(catalog['run'])/'START.json'
    start = reader.read(start_path)
    epoch = datetime.fromisoformat(start['created_utc'].replace('Z','+00:00')).timestamp()
    result = dict(documented_start_utc=start['created_utc'],documented_start_epoch=epoch,
        original_dirty_HEAD=start['head'],executed_origin=UNKNOWN,
        elapsed_wall_seconds=UNKNOWN,external_finalization_epoch=UNKNOWN,
        elapsed_at_measurement_cutoff_seconds=UNKNOWN,measurement_cutoff_epoch=UNKNOWN,
        measurement_cutoff_scope=None,
        scope='Documented START.json to explicit ROOT external finalization epoch, not child wall sum/service span/goal API time')
    for key,default_field in [('elapsed_measurement','measurement_cutoff_epoch'),
                              ('external_finalization','external_finalization_epoch')]:
        spec=catalog.get(key)
        if spec is None or not Path(spec['path']).exists():continue
        end=reader.read(spec['path'],spec.get('sha256'))
        value=end[spec.get('epoch_field',default_field)]
        duration(value-epoch)
        require(type(end.get('campaign_complete'))is bool, 'Elapsed receipt must declare completion status')
        if end['campaign_complete'] is True:
            result.update(external_finalization_epoch=value,elapsed_wall_seconds=value-epoch,
                          finalization_receipt=str(spec['path']))
        else:
            require(isinstance(end.get('scope'),str) and bool(end['scope'].strip()),
                    'An unfinished elapsed cutoff needs an explicit scope')
            result.update(measurement_cutoff_epoch=value,elapsed_at_measurement_cutoff_seconds=value-epoch,
                          measurement_cutoff_scope=end['scope'],measurement_cutoff_receipt=str(spec['path']))
    if catalog.get('executed_origin'):
        result['executed_origin']=catalog['executed_origin']
    return result


def audit(catalog, reader=None):
    reader = reader or Reader()
    require(catalog['schema'] == SCHEMA, 'Wrong master cost catalog')
    closed = closed_attempts(reader,catalog['closed'])
    recovered = [recovery_attempt(reader,s) for s in catalog['recoveries']]
    roots = closed['counted_roots'] + [str(Path(s['base']).resolve())for s in catalog['recoveries']]
    require(len(set(roots)) == len(roots), 'Execution base counted twice')
    known = closed['known_calls'] + sum(x['known_base_call_lower_bound']for x in recovered)
    historical_prefix = sum(x['prior_stored_known_calls']for x in recovered)
    if 'expected_old_D4_prefix' in catalog:
        require(historical_prefix == catalog['expected_old_D4_prefix'], 'Historical prefix identity differs')
    run = Path(catalog['run'])
    d_old = reader.read(run/'D_FIXED_COMMON_GRAPH/SUMMARY.json')
    d_fixed = reader.read(run/'D_FIXED_COMMON_GRAPH_FIXED_GAMMA/SUMMARY.json')
    d_calls = [count(d_old['actual_new_family_optimizer_calls']),count(d_fixed['actual_new_family_optimizer_calls'])]
    material = reader.read(run/'s3_s4_postseal_analysis/ACTUAL_COST_INVENTORY.json')
    require(material['complete'] is True and material['old_original_D1_or_PF_production_cost_charged_again'] is False,
            'Material cost scope differs')
    for path,digest in material['source_bindings'].items():
        reader.bind(path,digest)
    if 's3_s4_inventory_sha256' in catalog:
        reader.bind(run/'s3_s4_postseal_analysis/ACTUAL_COST_INVENTORY.json',catalog['s3_s4_inventory_sha256'])
    heads = head_accounts(reader,run,catalog['predictions'],catalog.get('expected_new_fits',33))
    ledgers = [command_ledger(reader,s) for s in catalog.get('ledgers',[])]
    service = services(reader,catalog['services'])
    receipts = optional_metadata(reader,catalog.get('metadata',[]))
    elapsed = campaign_elapsed(reader,catalog)
    receipt_values={r['name']:r.get('values',{})for r in receipts}
    s1=receipt_values.get('S1',{})
    s1_decomp=receipt_values.get('S1_COST_DECOMPOSITION',{})
    s2=receipt_values.get('S2_CAL_BODY',{})
    if s1_decomp and s1:
        require(count(s1_decomp['totals.new_unique_PF_cost_evaluations'])
                + count(s1_decomp['totals.old_individual_cost_reconstruction_evaluations'])
                == count(s1['new_individual_cost_evaluations']), 'S1 scientific cost decomposition differs')
    result = dict(schema=SCHEMA,created_epoch=time.time(),complete_all_cost_scopes=False,
        status='AUDITED_KNOWN_COSTS_WITH_EXPLICIT_UNKNOWNS',
        numerical_reference_reads=0,dataset_or_pose_bank_decodes=0,new_pose_optimizer_calls=0,
        new_forwards=0,new_fits=0,typed_counters=dict(
            fullnormal_runtime_minimize_calls=dict(known_lower_bound=known,actual_total=UNKNOWN,
                closed_historical_calls=closed['known_calls'],cumulative_recovered_base_calls=[x['base_calls_cumulative']for x in recovered],
                old_D4_prefix_included_not_added_again=historical_prefix,
                original_interrupted_arm_directories=sum(len(x['historical_unknown_arm_directories'])for x in recovered),
                interrupted_historical_calls=UNKNOWN),
            D_diagnostic_runtime_minimize_calls=dict(old_Sigma_plus_Gamma=d_calls[0],fixed_Gamma=d_calls[1],sum_distinct_attempts=sum(d_calls),
                family_dispatch_counter_fixed_Gamma=d_fixed['actual_new_family_dispatches'],
                scope='Different diagnostic objectives; no equality asserted with fullnormal or S3 counters'),
            S3_S4=material['instrumented_completed_generation_all_version_call_sum'],
            S3_S4_render_resource_counters=material['instrumented_completed_generation_all_version_resource_sum'],
            S1_candidate_cost_evaluations=dict(
                scientific_actual=s1.get('new_individual_cost_evaluations',UNKNOWN),
                new_unique_PF=s1_decomp.get('totals.new_unique_PF_cost_evaluations',UNKNOWN),
                old_cost_fallback_reconstruction=s1_decomp.get('totals.old_individual_cost_reconstruction_evaluations',UNKNOWN),
                postseal_diagnostic=s1.get('postseal_gallery_diagnostic_cost_evaluations',UNKNOWN),
                scope='Scientific source summary includes actual fallback old-cost evaluations; do not drop them as constructed work.'),
            S2_frozen_observation_head=dict(
                frames_with_scored_full_candidates=s2.get('synthetic_packet_export.frames_with_scored_full_candidates',UNKNOWN),
                actual_forward_API_calls=UNKNOWN,
                unit='Saved synthetic frame-level scored-candidate count; distinct from backbone and tiny H1 real prediction forwards.')),
        no_cross_type_grand_call_total=True,closed_attempts=closed,recovery_attempts=recovered,H1=heads,
        material_cost_inventory=material,command_ledgers=ledgers,service_lifecycle=service,other_receipts=receipts,
        campaign_elapsed_wall_seconds=elapsed['elapsed_wall_seconds'],
        elapsed_at_measurement_cutoff_seconds=elapsed['elapsed_at_measurement_cutoff_seconds'],campaign_elapsed=elapsed,
        untracked_costs=dict(original_D4_interrupted_in_memory_work=UNKNOWN,
            S3_S4_partial_technical_work=UNKNOWN,pre_service_startup_and_metadata_wrappers=UNKNOWN,
            pre_service_uninstrumented_import_finalization=UNKNOWN,human_agent_work=UNKNOWN),
        timer_rules=[
            'No universal summed wall is campaign elapsed; overlapping producers and idle waiters have different scopes.',
            'Service cgroup CPU includes child commands; child ledgers include internal fit and prediction timers.',
            'Base ROW timers include original completed accounts; recovery job timers contain only this invocation overhead/work.',
            'S3 FULL/PILOT invocation timers include rows already represented by the distinct completed-row inventory; do not sum both.',
            'Closed-attempt audit and cumulative recovered base receipts count original ROW current-call accounts once; copied FRAME intrinsic costs add zero.',
            'Reservations, completed optimizer fits, tiny H1 predictions, backbone forwards and scientific solver calls are different counters.'],
        input_bindings=reader.bindings)
    reader.stable()
    return result


def default_catalog(run):
    """Explicit metadata inventory, including future terminal receipt locations."""
    r = Path(run).resolve(); source = Path(__file__).with_name('fullnormal_recovery.py')
    predictions=[]
    pairs=[('S1_UNION',r/'s1_training',r/'s1_predictions')]
    pairs += [(a,r/'s3_s4_training'/a,r/'s3_s4_predictions'/a)for a in ('S3_COMMON','S3_BI','S4_PL','S4_CONTOUR','S4_IMAGE')]
    pairs += [('S2_'+a,r/'S2_UNIT_WEIGHT_FOUR_CPU_H1'/a/'training',r/'S2_UNIT_WEIGHT_FOUR_CPU_H1'/a/'prediction')for a in ('C0','CP','CB')]
    pairs += [(a,r/'F_FINALIZE'/a/'training',r/'F_FINALIZE'/a/'prediction')for a in ('F_COMMON','F_BI')]
    for a,t,p in pairs:predictions.append(dict(arm=a,training_manifest=str(t/'TRAINING_MANIFEST.json'),path=str(p/'PREDICTION_SEAL.json')))
    ledgers=[
        ('S3_S4_POSTPROCESS','s3_s4_postprocess/POSTPROCESS_LEDGER.jsonl',None),
        ('S3_S4_POSTSEAL','ROOT_S3_S4_POSTSEAL_EXECUTION/POSTPROCESS_LEDGER.jsonl',None),
        ('FINAL_SEQUENCE','FINAL_SEQUENCE_RECOVERY_V1/ACTUAL_COMMAND_LEDGER.jsonl','FINAL_SEQUENCE_SERVICE'),
        ('F_FINISH','F_FINALIZE/POSTPROCESS_LEDGER.jsonl','FINAL_SEQUENCE/F_FINISH_CHILD')]
    md=[
        ('S1','S1_EXECUTION_SUMMARY.json',['NN_forward_calls_new','new_synthetic_F_calls','new_SubPix_corner_calls','new_individual_cost_evaluations','postseal_gallery_diagnostic_cost_evaluations','phase_wall_seconds','instrumented_phase_wall_sum_seconds','wall_scope'],'NN_forward_calls_new is the source backbone field only, not zero tiny-H1 forwards. S1 phase sum excludes outside timers; fits/real H1 forwards separately authenticated'),
        ('S1_COST_DECOMPOSITION','S1_REUSE_AND_ADMISSION_AUDIT.json',['totals.new_unique_PF_cost_evaluations','totals.old_individual_cost_reconstruction_evaluations'],'1348 new PF cost evaluations plus4 actual synthetic old-cost fallback reconstructions; the4 are not constructed tests'),
        ('D_FIXED_GAMMA_WRAPPER','D_FIXED_GAMMA_EXECUTION.json',['wall_seconds','CPU_seconds','summary_sha256'],'Outer child timer contains fixed-Gamma summary body timer'),
        ('F_FRONTEND_INITIAL','F_PRECOMPUTE_FRONTEND_EXECUTION.json',['complete','commands'],'Partial controller journal; successful observation child wall includes import/I/O'),
        ('F_FRONTEND_REPAIR','F_FRONTEND_BOOKKEEPING_REPAIR.json',['complete','original_CAL_body_wall_seconds','initial_CAL_child_import_setup_wall_seconds','initial_CAL_child_returncode','recovered_receipt_path','recovered_receipt_sha256'],'CAL body timer distinct; missing original outer timer remains unknown'),
        ('F_OBSERVATION_BODY','F_OBSERVATIONS/OBSERVATION_MANIFEST.json',['complete','rows','wall_seconds','CPU_seconds','new_state_mask_PF_forwards','new_observation_network_fits','pose_optimizer_calls','H1_fits','final_F_complete'],'Observation body is inside initial child wall timer. New state/mask/PF backbone forwards0; this does not mean all downstream H1 forwards0'),
        ('F_CAL_BODY','F_CALIBRATION/PACKET_MANIFEST.json',['complete','rows','wall_seconds','observations_sha256','models_sha256','numeric_reference_scope'],'CAL-only fitting/packet body; not a H1 fit and not proof of a fully measured outer process'),
        ('S2_CAL_BODY','S2_CALIBRATION/SUMMARY.json',['status','wall_seconds','new_H1_fits','new_pose_optimizer_calls','synthetic_packet_export.wall_seconds','synthetic_packet_export.frames_with_scored_full_candidates','synthetic_packet_export.rows','synthetic_packet_export.observation_model_fits'],'Own CAL and fixed-observation-head synthetic packet export; nested export/body timers, frame count is not an inferred neural forward API count'),
        ('DISPLAY','DERIVED_VISUAL_EXECUTION/FINAL_EXECUTION_SEAL.json',['all_visual_calls','initial_failed_controller_launcher','cost_scope','command_ledger_sha256'],'All display versions incl interruption; not scientific producers'),
        ('D_PUBLIC_EXPORT','D_SCALAR_EXPORT_EXECUTION/EXECUTION_SEAL.json',['all_recorded_child_invocations','final_output_calls'],'Metadata/scalar export incl failed checks; no new science'),
        ('S3_FINALIZATION','s3_s4_production/EXTERNAL_FINALIZATION_TIME_PROVENANCE.json',['producer_receipt_timer_wall_seconds','producer_receipt_timer_CPU_seconds','post_timer_CPU_seconds','log_creation_to_last_stdout_fileclock_seconds','fileclock_interval_is_external_proxy_not_monotonic_process_lifetime'],'External file clock proxy and sampled CPU lower bound; not exact process lifetime')]
    return dict(schema=SCHEMA,run=str(r),expected_new_fits=33,expected_old_D4_prefix=112387,
        elapsed_measurement=dict(path=str(r/'ROOT_COST_ELAPSED_CUTOFF.json'),epoch_field='measurement_cutoff_epoch'),
        external_finalization=dict(path=str(r/'ROOT_EXTERNAL_CAMPAIGN_FINALIZATION.json'),epoch_field='external_finalization_epoch'),
        closed=dict(path=str(r/'CORE_COSTS_CLOSED_ATTEMPTS_CHECK.json'),
            sha256='f0f4dad1eb335d77077a8b85a7ad78b4933338572b3a6c9a582348949fb031bf',expected_calls=479400),
        recoveries=[dict(name=name,base=str(r/base),invocation=str(r/inv),arms=arms,expected_jobs=1241,recovery_source=str(source))for name,base,inv,arms in (
            ('S2_D4_CUMULATIVE','S2_FULL_UNIT_WEIGHT_FOUR_CPU','S2_FOUR_CPU_RECOVERY_INVOCATION_V1',['C0','CP','CB']),
            ('F_D4_CUMULATIVE','F_CORE_FOUR_CPU','F_FOUR_CPU_RECOVERY_INVOCATION_V1',['F_COMMON','F_BI']))],
        s3_s4_inventory_sha256='a404915c820ed241c3eac5aa77b9994ebd52410602690ffb356db6cbed0e3865',
        predictions=predictions,
        ledgers=[dict(name=n,path=str(r/p),parent_scope=parent)for n,p,parent in ledgers],
        services=dict(launches=[str(r/p)for p in ('ROOT_FULLNORMAL_RECOVERY_SERVICE_LAUNCH.json','ROOT_POST_CORE_RECOVERY_SERVICES_LAUNCH.json')],lifecycle_root=str(r/'ROOT_SERVICE_LIFECYCLE_V1')),
        metadata=[dict(name=n,path=str(r/p),fields=f,scope=s,**(
            {'sha256':'e33234659d035e9a0671e7a0612dd492b528f2cd8834b76f197d4cbf5f53fef1'}
            if n=='S1_COST_DECOMPOSITION'else{}))for n,p,f,s in md])


def public_projection(result, private_receipt_sha256):
    """Bounded scalar whitelist; no paths, frame IDs, argv, tensors or poses."""
    counters=result['typed_counters']; core=counters['fullnormal_runtime_minimize_calls']; h=result['H1']
    bindings=result['input_bindings']
    binding_digest=hashlib.sha256(json.dumps(bindings,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    summary=dict(schema='PUBLIC_RECOVERY_AWARE_MASTER_COST_SUMMARY_V1',
        private_master_receipt_sha256=private_receipt_sha256,
        private_input_bindings_canonical_sha256=binding_digest,private_input_binding_count=len(bindings),
        complete_cost_measurement=False,unknown_cost_token=UNKNOWN,
        all_declared_H1_fits_complete=h['all_expected_fits_complete'],
        all_declared_H1_prediction_seals_complete=h['all_declared_prediction_seals_complete'],
        fullnormal_runtime_minimize_calls={k:core[k]for k in
            ('known_lower_bound','actual_total','closed_historical_calls','cumulative_recovered_base_calls',
             'old_D4_prefix_included_not_added_again','original_interrupted_arm_directories','interrupted_historical_calls')},
        D_diagnostic_runtime_minimize_calls={k:counters['D_diagnostic_runtime_minimize_calls'][k]for k in
            ('old_Sigma_plus_Gamma','fixed_Gamma','sum_distinct_attempts','family_dispatch_counter_fixed_Gamma')},
        S1_candidate_cost_evaluations=counters['S1_candidate_cost_evaluations'],
        S2_frozen_observation_head=counters['S2_frozen_observation_head'],
        S3_S4_completed_all_version_counters={k:v for k,v in counters['S3_S4'].items()if k in
            ('objective_calls','jacobian_calls','DIS_calls','PnP_calls')},
        S3_S4_completed_all_version_resources={k:v for k,v in counters['S3_S4_render_resource_counters'].items()if k in
            ('CPU_seconds','wall_seconds','actual_raster_passes','logical_render_calls','renderer_wall_seconds')},
        H1={k:h[k]for k in ('reservations','completed_new_fits','expected_final_new_fits','optimizer_steps',
            'training_frame_exposures','internal_fit_CPU_seconds','internal_fit_wall_seconds_sum',
            'tiny_head_real_prediction_forwards','new_backbone_prediction_forwards','prediction_CPU_seconds',
            'prediction_wall_seconds_sum','unreceipted_attempt_cost')},
        campaign_elapsed_wall_seconds=result['campaign_elapsed_wall_seconds'],
        elapsed_at_measurement_cutoff_seconds=result['elapsed_at_measurement_cutoff_seconds'],
        measurement_cutoff_scope=result['campaign_elapsed']['measurement_cutoff_scope'],
        documented_campaign_start_utc=result['campaign_elapsed']['documented_start_utc'],
        external_finalization_epoch=result['campaign_elapsed']['external_finalization_epoch'],
        service_lifecycle=dict(complete=result['service_lifecycle']['complete'],
            inclusive_cgroup_CPU_seconds=result['service_lifecycle'].get('service_CPU_seconds',UNKNOWN),
            recovery_service_envelope_elapsed_seconds=result['service_lifecycle'].get('recovery_service_envelope_elapsed_seconds',UNKNOWN),
            per_role=[{k:e[k]for k in ('role','CPU_seconds','elapsed_wall_seconds','Result','ExecMainStatus')}
                for e in result['service_lifecycle'].get('entries',[])]),
        child_command_ledgers=[{k:e[k]for k in ('name','status','completed_or_failed_child_calls','failures',
            'additive_child_wall_seconds','additive_child_CPU_seconds')if k in e}for e in result['command_ledgers']],
        untracked_costs=result['untracked_costs'],no_cross_type_grand_call_total=True,
        interpretation=[
            'Known solver calls are a lower bound; the original interrupted in-memory work remains unknown.',
            'The original D4 prefix is included in the recovered cumulative bases, never added twice.',
            'D, full-normal minimize, objective/Jacobian, DIS/PnP and rendering are different counter types.',
            'H1 cumulative RUN costs count each optimizer once, including same-optimizer resume.',
            'Inclusive service CPU, child command costs and internal fit/prediction timers nest; do not add these scopes.',
            'Summed child wall and overlapping service elapsed are not whole campaign elapsed.',
            'Documented final elapsed is START.json to an explicit external campaign finalization epoch.',
            'A cost/report cutoff with campaign_complete false is elapsed_at_measurement_cutoff, never final workflow elapsed.'],
        metadata_receipt_hashes={},other_measured_costs=[])
    public_fields={
        'S1':('new_synthetic_F_calls','new_SubPix_corner_calls','new_individual_cost_evaluations',
            'postseal_gallery_diagnostic_cost_evaluations','phase_wall_seconds','instrumented_phase_wall_sum_seconds'),
        'D_FIXED_GAMMA_WRAPPER':('wall_seconds','CPU_seconds'),
        'F_FRONTEND_REPAIR':('original_CAL_body_wall_seconds','initial_CAL_child_import_setup_wall_seconds'),
        'F_OBSERVATION_BODY':('wall_seconds','CPU_seconds','new_state_mask_PF_forwards','new_observation_network_fits'),
        'F_CAL_BODY':('wall_seconds',),
        'S2_CAL_BODY':('wall_seconds','synthetic_packet_export.wall_seconds','synthetic_packet_export.rows',
            'synthetic_packet_export.frames_with_scored_full_candidates'),
        'DISPLAY':('all_visual_calls',), 'D_PUBLIC_EXPORT':('all_recorded_child_invocations','final_output_calls'),
        'S3_FINALIZATION':('producer_receipt_timer_wall_seconds','producer_receipt_timer_CPU_seconds','post_timer_CPU_seconds',
            'log_creation_to_last_stdout_fileclock_seconds','fileclock_interval_is_external_proxy_not_monotonic_process_lifetime')}
    for receipt in result['other_receipts']:
        if receipt['name'] not in public_fields or 'values' not in receipt:continue
        summary['other_measured_costs'].append(dict(name=receipt['name'],
            values={k:receipt['values'][k]for k in public_fields[receipt['name']]if k in receipt['values']}))
    summary['S3_S4_postprocessing_cost']=result['material_cost_inventory'].get('postprocessing_command_cost',{})
    summary['S3_S4_unknown_partial_work']=UNKNOWN
    names=('START.json','GLOBAL_H1_FIT_ATTEMPTS.jsonl','H1_EXECUTION_EVENTS.jsonl',
        'CORE_COSTS_CLOSED_ATTEMPTS_CHECK.json','ACTUAL_COST_INVENTORY.json')
    for name in names:
        matches=[digest for path,digest in bindings.items()if Path(path).name==name]
        if len(matches)==1:summary['metadata_receipt_hashes'][name]=matches[0]
    for i,recovery_ in enumerate(result['recovery_attempts']):
        for leaf in ('INVOCATION_LOCK.json','RECOVERY_COMPLETION.json'):
            key=str(Path(recovery_['invocation'])/leaf)
            if key in bindings:summary['metadata_receipt_hashes'][f'recovery_{i}_{leaf}']=bindings[key]
    require(len(json.dumps(summary))<100000, 'Public cost projection unexpectedly large')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog-path',required=True)
    parser.add_argument('--root-go',required=True,help='JSON approving this metadata-only audit, binding source/catalog SHA')
    parser.add_argument('--output',required=True)
    a=parser.parse_args()
    reader=Reader(); catalog=reader.read(a.catalog_path); go=reader.read(a.root_go)
    require(go.get('approved') is True and go.get('metadata_only') is True
            and go.get('auditor_source_sha256') == sha(__file__)
            and go.get('catalog_sha256') == sha(a.catalog_path), 'Fresh source/catalog-bound metadata ROOT GO required')
    out=Path(a.output)
    require(not out.exists(), 'Never overwrite a previous cost audit')
    tick=time.monotonic(); cpu=time.process_time()
    result=audit(catalog,reader)
    result['auditor_source_sha256']=sha(__file__)
    result['auditor_own_timer_scope']='Read/verification only, stops before output encoding/write; external invocation ledger may capture finalization'
    result['auditor_CPU_seconds']=time.process_time()-cpu
    result['auditor_wall_seconds']=time.monotonic()-tick
    out.mkdir(parents=True,exist_ok=False)
    with (out/'MASTER_COSTS.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    public=public_projection(result,sha(out/'MASTER_COSTS.json'))
    with (out/'PUBLIC_COST_SUMMARY.json').open('x')as f:json.dump(public,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    with (out/'MASTER_COSTS.md').open('x')as f:
        core=result['typed_counters']['fullnormal_runtime_minimize_calls']
        f.write(f"# Campaign cost audit\n\nKnown full-normal minimize calls: **{core['known_lower_bound']}** (lower bound). Actual campaign optimizer total: **{UNKNOWN}**. Documented campaign elapsed: **{result['campaign_elapsed_wall_seconds']}** seconds.\n\n")
        f.write(f"Elapsed at the explicit measurement cutoff: **{result['elapsed_at_measurement_cutoff_seconds']}** seconds; scope: {result['campaign_elapsed']['measurement_cutoff_scope']}. This is not final workflow elapsed.\n\n")
        f.write(f"Closed attempts: {core['closed_historical_calls']}; cumulative S2/F D4 bases: {core['cumulative_recovered_base_calls']}. The old D4 prefix {core['old_D4_prefix_included_not_added_again']} is already included.\n\n")
        f.write(f"H1: {result['H1']['completed_new_fits']} completed / {result['H1']['reservations']} registered / {result['H1']['expected_final_new_fits']} planned fits; tiny real prediction forwards {result['H1']['tiny_head_real_prediction_forwards']}; backbone prediction forwards {result['H1']['new_backbone_prediction_forwards']}.\n\n")
        f.write('D, full-normal, S3 objective/Jacobian, DIS, PnP, renders, training and display counters remain separate. Child wall sums and service lifetimes do not define whole campaign elapsed. Inclusive service CPU is not added to its child timers. Interrupted work and earlier untracked import/finalization remain unknown. See MASTER_COSTS.json for every binding, timer scope and unfinished receipt.\n')
    print(json.dumps(dict(status=result['status'],output=str(out),new_scientific_calls=0)))


if __name__=='__main__':main()
