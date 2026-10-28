"""Count-proportional R2 continuation. All writes stay in this new experiment.

Reuse the already validated immutable R1, optimizer, pseudo-labels and subsets.
No evaluation labels enter training, sampling, or checkpoint selection.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'dope-r2-ratio-per-type-20261027-v1'
DELIVERY = ROOT.parent / 'dope-thesis-delivery-20261029-v1'
KINDS = ('WOOD', 'GREEN', 'PLASTIC')
LEVELS = ('A', 'B', 'C', 'D', 'E')
SEEDS = (42, 43, 44)
PYTHON = '/home/minjae/anaconda3/envs/pallet-pose/bin/python'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, replace=False):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT) and not path.is_symlink(), 'Output outside experiment')
    require(replace or not path.exists(), 'Preserve existing artifact: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old_context():
    inputs = read(ROOT / 'preflight/INPUTS.json')
    for path, digest in inputs['old_entrypoint_bindings'].items():
        require(sha(path) == digest, 'Original entrypoint changed: ' + path)
    driver = load(OLD / 'code/experiment_driver.py', 'proportional_immutable_original_driver')
    return driver, driver.Guard()


def prepare():
    require(not (ROOT / 'FIXED_PLAN.json').exists(), 'Already prepared; preserve frozen plan')
    base = read(OLD / 'FIXED_PLAN.json')
    allow = read(OLD / 'preflight/ALLOWLIST.json')
    eval_allow = read(OLD / 'preflight/EVAL_ALLOWLIST.json')
    for name in ('ALLOWLIST.json', 'EVAL_ALLOWLIST.json'):
        dest = ROOT / 'preflight' / name
        require(not dest.exists(), 'Preserve existing allowlist')
        dest.write_bytes((OLD / 'preflight' / name).read_bytes())
    (ROOT / 'SUBSETS.json').write_bytes((OLD / 'SUBSETS.json').read_bytes())
    source_files = {}
    for envelope in (allow, eval_allow):
        for path, item in envelope['input_files'].items():
            if path in source_files:
                require(source_files[path]['sha256'] == item['sha256'], 'Conflicting prior allowlist')
            else:
                source_files[path] = dict(path=path, sha256=item['sha256'], role=item['category'])
    bindings = {}
    for rel in ('FIXED_PLAN.json', 'SUBSETS.json', 'preflight/ALLOWLIST.json',
                'preflight/TRAINING_GO.json', 'code/experiment_driver.py', 'code/count_trainer.py'):
        path = OLD / rel
        bindings[str(path)] = sha(path)
        source_files[str(path)] = dict(path=str(path), sha256=bindings[str(path)], role='prior_protocol_or_code')
    for name in ('ratio_eval_common.py', 'evaluate_ratios.py'):
        path = OLD / 'code' / name
        source_files[str(path)] = dict(path=str(path), sha256=sha(path), role='source_code')
    inputs = dict(schema='proportional_input_authority_v1', created_utc=now(),
        original_ratio_root=str(OLD), old_entrypoint_bindings=bindings,
        source_files=list(source_files.values()),
        eval_allowlist_sha256=sha(ROOT / 'preflight/EVAL_ALLOWLIST.json'),
        policy='Prior exact allowlists retained. No raw reference annotations or other protected FINAL. '
               'Only explicit historical WOOD80 RGB/cache exception for evaluation. '
               'Declared input hashes are verified on use; training union is independently verified preflight.')
    write(ROOT / 'preflight/INPUTS.json', inputs)
    training = dict(base['training'], batch_size=16,
        loss_mode='count_weighted_original_branch_means_sum',
        real_batch_policy='joint_uniform_shuffle_cycle_fill')
    plan = dict(schema='R2_count_proportional_v1', status='FROZEN_BEFORE_TRAINING',
        created_utc=now(), kinds=list(KINDS), conditions=list(LEVELS), training_seeds=list(SEEDS),
        subset_seed=20261027, original_teacher_training_seed=42,
        real_unique_counts=base['real_unique_counts'], synthetic_unique_counts=base['synthetic_unique_counts'],
        synthetic_percent=base['levels'], training=training, original_training=base['training'],
        updates_per_model=828, effective_batch_size=16, total_models=45,
        sampling='One continuous jointly shuffled concatenated real+synthetic pool. '
                 'Cycle-fill across batch and epoch boundaries. Every image equally likely; '
                 'full-pool traversal before reshuffle. No 8+8 source quota.',
        loss='2*(n_syn/16*original_syn_branch_mean+n_real/16*original_real_branch_mean). '
             'Preserve original valid-channel/pixel normalization within each branch. '
             'Empty branches contribute zero. At 8+8 exactly the original sum, including variable masks. '
             'This is not a newly normalized per-image masked loss.',
        seed_interpretation='Three R2 sampling/augmentation seeds conditional on the same original R1 weights, '
                            'Adam state, scaler, and fixed pseudo-label/subset sets; not full-pipeline seed repetitions.',
        optimizer_accounting='828 R2 optimizer.step calls from the frozen R1 state. '
            'Standard Adam skips parameters with grad=None, so final affinity-head counters legitimately '
            'lag on real-only batches. Validate every named parameter against its actual gradient-present '
            'update count; never fabricate gradients, parameter steps, or count skipped AMP updates.',
        evaluation=dict(primary_total=372, PLASTIC=128, WOOD=125, GREEN=119,
            reference_points=dict(PLASTIC=985, WOOD=963, GREEN=943),
            original_view_only=True, manual_panels_separate=True,
            bootstrap_seed=20261026, bootstrap_replicates=10000,
            interval='percentile95 paired frame; conditional on fitted seeds', multiplicity_correction='none',
            training_seed_sd='sample standard deviation ddof=1 across 42/43/44',
            checkpoint_selection='Final update828 for every condition; no evaluation selection',
            comparator_C='New proportional C with the same R2 seed',
            comparator_control='Historical shared synthetic control; contextual R1 history differs'),
        storage='Preserve45 FP32 final state_dict checkpoints; only one active new optimizer resume. '
                'Delete only own active resume after committed final checkpoint+receipt validation.',
        fixed_endpoints=True, all_conditions_reported=True, desired_direction_not_required=True,
        constraints=['No original files modified', 'No new teacher inference or pseudo-label filtering',
            'No raw reference annotations', 'No unapproved FINAL', 'No Git operations in dirty primary repository',
            'Prior results retained; this follow-up is exploratory after inspecting prior results'])
    write(ROOT / 'FIXED_PLAN.json', plan)
    d, g = old_context()
    for kind in KINDS:
        ctx = d.context(g, kind, 'E')
        d.verify_training_bytes(g, ctx['synthetic'], ctx['real'])
    g.check_unchanged()
    write(ROOT / 'preflight/TRAINING_INPUT_HASHES_BEFORE.json', dict(passed=True, utc=now(), files=g.bindings))
    print('PREPARE_PASS', len(g.bindings), 'training/metadata bindings; no model imported', flush=True)


def verify_go():
    g = read(ROOT / 'preflight/TRAINING_GO.json')
    require(g.get('passed') is True and g.get('training_authorized') is True, 'Missing new training GO')
    for path, digest in g['frozen_files'].items():
        require(sha(path) == digest, 'Frozen new protocol/code changed: ' + path)
    return g


@contextlib.contextmanager
def gpu_lock():
    path = DELIVERY / 'GPU_QUEUE.lock'
    require(path.is_file(), 'Shared GPU queue lock missing')
    with path.open('r') as shared, (ROOT / 'logs/CAMPAIGN.lock').open('a') as own:
        fcntl.flock(own.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(shared.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(shared.fileno(), fcntl.LOCK_UN)
            fcntl.flock(own.fileno(), fcntl.LOCK_UN)


def trial(kind, condition, seed):
    go = verify_go()
    require(kind in KINDS and condition in LEVELS and seed in SEEDS, 'Outside fixed scope')
    plan = read(ROOT / 'FIXED_PLAN.json')
    d, g = old_context()
    ctx = d.context(g, kind, condition)
    out = ROOT / 'runs' / kind / condition / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    cp, receipt_path = out / 'R2.pth', out / 'TRAIN_COMPLETE.json'
    protocol = dict(kind=kind, condition=condition, seed=seed, round=2,
        fixed_plan_sha256=sha(ROOT / 'FIXED_PLAN.json'),
        training_go_sha256=sha(ROOT / 'preflight/TRAINING_GO.json'),
        subsets_sha256=sha(ROOT / 'SUBSETS.json'), training=plan['training'],
        real_ids=[str(x['id']) for x in ctx['real']], synthetic_ids=[str(x['id']) for x in ctx['synthetic']],
        teacher_checkpoint_sha256=ctx['first']['checkpoint_sha256'],
        R1_resume_sha256=ctx['refs']['r1_resume_sha256'],
        pseudo_sha256=g.sha(ctx['refs']['r2_pseudo']), evaluation_used_for_selection=False)
    if (out / 'PROTOCOL.json').exists():
        require(read(out / 'PROTOCOL.json') == protocol, 'Per-run frozen protocol changed')
    else:
        write(out / 'PROTOCOL.json', protocol)
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt.get('complete') and receipt['checkpoint_sha256'] == sha(cp)
            and receipt['protocol_sha256'] == sha(out / 'PROTOCOL.json'), 'Completed run is inconsistent')
        cleanup_committed_resume(receipt, cp, preserve_unrelated=True)
        return
    require(shutil.disk_usage(ROOT).free > 2_000_000_000, 'Less than2GB headroom; preserve data and stop')
    d.verify_training_bytes(g, ctx['synthetic'], ctx['real'])
    rt = d.runtime(g, ctx)
    sampler = load(ROOT / 'code/train_proportional.py', 'count_proportional_new_sampler')
    active = ROOT / 'state/ACTIVE_RESUME.pth'
    result = sampler.train_r2(rt['helper'], rt['network'], rt['optimizer'], rt['scaler'],
        ctx['synthetic'], ctx['real'], cp, plan['training'], seed=seed, active_resume_path=active)
    stats = result['stats']
    require(stats['complete'] and stats['updates_completed'] == 828
            and stats['global_optimizer_steps'] == 1656
            and stats['parameter_step_validation_passed'] is True,
            'R2 update or Adam budget differs')
    require(len(stats['history']) == 828, 'Incomplete update history')
    require(stats['committed_synthetic_draws'] + stats['committed_real_draws'] == 13248, 'Draw count differs')
    g.check_unchanged()
    receipt = dict(complete=True, utc=now(), kind=kind, condition=condition, seed=seed, training_seed=seed,
        checkpoint_path=str(cp), checkpoint_sha256=sha(cp), round=2,
        protocol_sha256=sha(out / 'PROTOCOL.json'), fixed_plan_sha256=sha(ROOT / 'FIXED_PLAN.json'),
        real_unique_count=len(ctx['real']), synthetic_unique_count=len(ctx['synthetic']),
        teacher_checkpoint_sha256=ctx['first']['checkpoint_sha256'],
        R1_resume_sha256=ctx['refs']['r1_resume_sha256'],
        frozen_R2_pseudo_sha256=protocol['pseudo_sha256'],
        optimizer_steps=stats['optimizer_parameter_step_values'],
        exposure_log_sha256=sha(out / 'R2.pth.exposures.jsonl'), stats=stats,
        active_resume_sha256_at_commit=sha(active),
        inputs_unchanged=True, source_and_input_sha256=g.bindings,
        evaluation_used_for_selection=False, intermediate_resume_policy=plan['storage'])
    write(receipt_path, receipt)
    require(sha(cp) == receipt['checkpoint_sha256'] and read(receipt_path)['complete'], 'Commit validation failed')
    cleanup_committed_resume(receipt, cp)
    write(out / 'RESUME_CLEANUP.json', dict(complete=True, removed_only_own_active_resume=True,
        checkpoint_sha256=receipt['checkpoint_sha256'], final_receipt_sha256=sha(receipt_path)))
    print('TRIAL_COMPLETE', kind, condition, seed, stats['elapsed_seconds'], flush=True)


def cleanup_committed_resume(receipt, checkpoint, preserve_unrelated=False):
    """Recover the narrow receipt-before-cleanup crash window without new reads.

    A committed receipt binds the exact active-state bytes and run identity.
    Never remove an active state for another run or a changed final checkpoint.
    """
    active = ROOT / 'state/ACTIVE_RESUME.pth'
    if not active.exists():
        return
    require(not active.is_symlink() and active.resolve().is_relative_to(ROOT / 'state'), 'Unsafe resume cleanup')
    active_sha = sha(active)
    if preserve_unrelated and active_sha != receipt['active_resume_sha256_at_commit']:
        # A restarted campaign walks earlier complete runs before reaching the
        # current partial run. Its active state must survive those earlier runs.
        return
    require(receipt.get('complete') is True
        and receipt['stats']['run_identity']['output_checkpoint'] == str(checkpoint.resolve())
        and receipt['stats']['run_identity']['fingerprint'] == receipt['stats']['fingerprint']
        and receipt['stats']['updates_completed'] == 828
        and receipt['checkpoint_sha256'] == sha(checkpoint)
        and receipt['active_resume_sha256_at_commit'] == active_sha,
        'Active state is not bound to this committed final checkpoint; preserve and stop')
    active.unlink()


def logged_command(argv, label):
    log = ROOT / 'logs' / (label + '.stdout.log')
    attempt = 1
    while log.exists():
        # A campaign is only restarted explicitly; preserve each failed attempt.
        # This function never retries a failed child by itself.
        attempt += 1
        log = ROOT / 'logs' / (label + f'.attempt{attempt}.stdout.log')
    start = time.monotonic()
    with log.open('x') as f:
        p = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    row = dict(utc=now(), command=argv, label=label, attempt=attempt, exit_code=p.returncode,
        elapsed_seconds=time.monotonic()-start, stdout_path=str(log), stdout_sha256=sha(log))
    with (ROOT / 'logs/COMMANDS.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    require(p.returncode == 0, 'Child command failed; stopped without changing conditions: ' + label)


def campaign():
    verify_go()
    start = time.monotonic()
    complete = []
    # First15 runs yield a complete three-type seed42 panel; subsequent seeds
    # repeat exactly the same condition order. No interim efficacy stopping.
    for seed in SEEDS:
        for kind in KINDS:
            for condition in LEVELS:
                run = f'{kind}_{condition}_seed{seed}'
                receipt = ROOT / 'runs' / kind / condition / f'seed_{seed}' / 'TRAIN_COMPLETE.json'
                write(ROOT / 'CAMPAIGN_STATUS.json', dict(stage='training', current=run, seed=seed,
                    kind=kind, condition=condition, completed=len(complete), total=45,
                    started_or_updated_utc=now(), elapsed_seconds=time.monotonic()-start), replace=True)
                if not receipt.exists():
                    logged_command([PYTHON, '-B', '-u', str(ROOT / 'code/campaign.py'), 'trial',
                        '--kind', kind, '--condition', condition, '--seed', str(seed)], run)
                else:
                    trial(kind, condition, seed)
                complete.append(dict(kind=kind, condition=condition, seed=seed,
                    receipt_sha256=sha(receipt), checkpoint_sha256=read(receipt)['checkpoint_sha256']))
    write(ROOT / 'TRAINING_COMPLETE.json', dict(complete=True, utc=now(), models=complete,
        model_count=45, updates=45*828, elapsed_seconds=time.monotonic()-start))
    write(ROOT / 'CAMPAIGN_STATUS.json', dict(stage='evaluation', completed=45, total=45, utc=now()), replace=True)
    logged_command([PYTHON, '-B', '-u', str(ROOT / 'code/evaluate_proportional.py'),
        '--root', str(ROOT), '--phase', 'all'], 'evaluate_all45')
    # Verify the same training inputs actually opened before launch remain unchanged.
    baseline = read(ROOT / 'preflight/TRAINING_INPUT_HASHES_BEFORE.json')['files']
    mismatches = [path for path, digest in baseline.items() if sha(path) != digest]
    write(ROOT / 'preflight/TRAINING_INPUT_HASHES_AFTER.json', dict(passed=not mismatches,
        utc=now(), checked_count=len(baseline), mismatches=mismatches, files=baseline))
    require(not mismatches, 'Original input bytes changed')
    write(ROOT / 'CAMPAIGN_COMPLETION.json', dict(complete=True, utc=now(), model_count=45,
        all_input_hashes_unchanged=True, elapsed_seconds=time.monotonic()-start,
        training_complete_sha256=sha(ROOT / 'TRAINING_COMPLETE.json')))
    write(ROOT / 'CAMPAIGN_STATUS.json', dict(stage='complete', completed=45, total=45, utc=now()), replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'trial', 'campaign'))
    parser.add_argument('--kind', choices=KINDS)
    parser.add_argument('--condition', choices=LEVELS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            prepare()
        elif args.action == 'trial':
            with gpu_lock():
                trial(args.kind, args.condition, args.seed)
        else:
            with (ROOT / 'logs/ORCHESTRATOR.lock').open('a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                campaign()
    except Exception as error:
        failure = dict(failed=True, utc=now(), action=args.action,
            error_type=type(error).__name__, message=str(error))
        with (ROOT / 'logs/FAILURES.jsonl').open('a') as handle:
            handle.write(json.dumps(failure) + '\n')
        write(ROOT / 'CAMPAIGN_FAILURE.json', failure, replace=True)
        raise


if __name__ == '__main__':
    main()
