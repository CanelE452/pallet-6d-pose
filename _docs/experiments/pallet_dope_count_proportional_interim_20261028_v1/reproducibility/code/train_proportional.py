"""R2 count-proportional sampling, with the immutable original branch losses.

No model or tensor library is imported at module load. The campaign driver owns
all input allowlists, hashes, R1 optimizer/scaler restoration and the GPU lock.
Only this new campaign directory may receive outputs. No teacher/eval forward
is performed here. Uniform *image* sampling uses the selected joint pool.
One shuffled joint pool is traversed before the next shuffle, filling
batches across traversal and epoch boundaries. Sources are never rebalanced.

For an actual batch containing ns synthetic and nr real draws, the objective is
2*ns/16*original_syn_mean + 2*nr/16*original_real_mean. Each original mean retains
its complete-branch valid-channel/pixel denominator, six stages and supervision.
This is not a mean of separately normalized per-image masked losses. For ns=nr=8
it is exactly the previous Lsyn+Lreal objective, including varying point masks.

Adam retains its normal per-parameter update semantics: a real-only batch does
not use the final affinity head m6_1, so its gradients remain None and its Adam
counter does not advance. A named gradient-presence ledger verifies those steps;
no artificial zero gradient or optimizer counter increment is introduced.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

OUT = Path(__file__).resolve().parents[1]
LOSS_MODE = 'count_weighted_original_branch_means_sum'
SAMPLING_MODE = 'joint_uniform_shuffle_cycle_fill'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def branch_weights(n_synthetic, n_real, batch_size=16):
    """Actual source-count coefficients; no branch-frequency rebalancing."""
    require(n_synthetic >= 0 and n_real >= 0 and n_synthetic + n_real == batch_size
            and batch_size > 0, 'Actual source counts must sum to the complete batch')
    return 2.0 * n_synthetic / batch_size, 2.0 * n_real / batch_size


def joint_schedule(n_synthetic, n_real, batch_size, steps, seed):
    """Traverse shuffled joint-image pools using a private Python RNG.

    Joint index [0, ns) is synthetic; [ns, ns+nr) is real. Draws can
    leave one source empty in a batch. Batch filling across a shuffle boundary
    may duplicate an image; no image repeats before its complete pool traversal.
    This one schedule spans every update, including nominal epoch boundaries.
    No sampling RNG is affected by augmentation or by another condition's run.
    """
    require(min(n_synthetic, n_real, batch_size, steps) > 0, 'Empty selected pool or invalid schedule')
    rng = random.Random(int(seed))
    total = n_synthetic + n_real
    batches, order, cursor = [], [], 0
    for _ in range(steps):
        batch = []
        while len(batch) < batch_size:
            if cursor == len(order):
                order = list(range(total))
                rng.shuffle(order)
                cursor = 0
            take = min(batch_size - len(batch), len(order) - cursor)
            batch.extend(order[cursor:cursor + take])
            cursor += take
        batches.append(batch)
    counts = Counter(index for batch in batches for index in batch)
    synthetic_draws = sum(value for index, value in counts.items() if index < n_synthetic)
    metadata = dict(policy=SAMPLING_MODE, seed=int(seed), joint_unique_pool=total,
        synthetic_unique_pool=n_synthetic, real_unique_pool=n_real,
        total_draws=batch_size * steps, synthetic_draws=synthetic_draws,
        real_draws=batch_size * steps - synthetic_draws,
        unique_exposed=len(counts), repeated_draws=batch_size * steps - len(counts),
        within_batch_duplicate_draws=sum(len(batch) - len(set(batch)) for batch in batches),
        synthetic_empty_batches=sum(all(i >= n_synthetic for i in batch) for batch in batches),
        real_empty_batches=sum(all(i < n_synthetic for i in batch) for batch in batches),
        synthetic_probability=n_synthetic / total, real_probability=n_real / total,
        per_image_probability_at_cycle_start=1.0 / total,
        complete_joint_traversals=batch_size * steps // total,
        final_partial_traversal_draws=batch_size * steps % total,
        minimum_per_image_exposure=min(counts.get(i, 0) for i in range(total)),
        maximum_per_image_exposure=max(counts.get(i, 0) for i in range(total)),
        joint_index_schedule_sha256=digest_json(batches))
    require(metadata['maximum_per_image_exposure'] - metadata['minimum_per_image_exposure'] <= 1,
            'Joint cycle exposure balance differs')
    return batches, metadata


def split_batch(batch, synthetic_records, real_entries):
    """Preserve within-source draw order and record the full joint draw order."""
    ns, nr = len(synthetic_records), len(real_entries)
    synthetic, real, joint = [], [], []
    for index in batch:
        require(isinstance(index, int) and 0 <= index < ns + nr, 'Joint index outside selected pool')
        if index < ns:
            synthetic.append(index)
            joint.append(dict(source='synthetic', id=str(synthetic_records[index]['id'])))
        else:
            real_index = index - ns
            real.append(real_index)
            joint.append(dict(source='real', id=str(real_entries[real_index]['id'])))
    return synthetic, real, joint


def backward_count_weighted(H, network, synthetic_rows, real_rows, synthetic_indices,
                            real_indices, device, microbatch, augmentation, amp, scaler):
    """Use the original masked loss unchanged, with actual draw-count weights."""
    ws, wr = branch_weights(len(synthetic_indices), len(real_indices))
    # Empty branches intentionally have no forward, target access or backward.
    ls = H._backward_branch(network, synthetic_rows, synthetic_indices, device, microbatch,
        True, ws, amp=amp, scaler=scaler) if synthetic_indices else 0.0
    lr = H._backward_branch(network, real_rows, real_indices, device, microbatch,
        False, wr, augmentation, amp=amp, scaler=scaler) if real_indices else 0.0
    return dict(loss_syn=ls, loss_real=lr, loss_total=ls + lr,
        loss_syn_branch_mean=ls / ws if ws else None,
        loss_real_branch_mean=lr / wr if wr else None,
        synthetic_loss_coefficient=ws, real_loss_coefficient=wr)


def validate_history(history, decoded, synthetic_records, real_entries):
    """Check the full committed history against the frozen continuous schedule."""
    require(len(history) <= len(decoded), 'Committed history exceeds the frozen schedule')
    require([row['update'] for row in history] == list(range(1, len(history) + 1)),
            'Committed history update sequence differs')
    for row, (si, ri, joint) in zip(history, decoded):
        epoch, local = divmod(row['update'] - 1, 276)
        ws, wr = branch_weights(len(si), len(ri))
        require(row['epoch'] == epoch + 1 and row['epoch_step'] == local + 1
                and row['joint_draws'] == joint
                and row['synthetic_ids'] == [str(synthetic_records[i]['id']) for i in si]
                and row['real_ids'] == [str(real_entries[i]['id']) for i in ri]
                and row['n_synthetic'] == len(si) and row['n_real'] == len(ri)
                and row['synthetic_loss_coefficient'] == ws and row['real_loss_coefficient'] == wr,
                'Committed history exposure/weight differs from the frozen schedule')
        require(all(math.isfinite(row[key]) for key in ('loss_syn', 'loss_real', 'loss_total',
                                                       'elapsed_seconds', 'scaler_scale'))
                and row['elapsed_seconds'] >= 0 and row['scaler_scale'] > 0
                and row['loss_total'] == row['loss_syn'] + row['loss_real'],
                'Committed history has invalid scalar accounting')
        for source, weight in [('syn', ws), ('real', wr)]:
            value, mean = row['loss_' + source], row['loss_' + source + '_branch_mean']
            require((weight == 0 and value == 0 and mean is None)
                    or (weight > 0 and mean is not None and math.isfinite(mean)
                        and math.isclose(value, weight * mean, rel_tol=1e-12, abs_tol=1e-12)),
                    'Committed history branch mean / weighted contribution differs')


def expected_missing_gradient_names(parameter_names, n_synthetic):
    """Frozen DOPE graph: only m6_1 is unused for a real-only belief loss."""
    names = sorted(parameter_names)
    final_affinity = [name for name in names if name.startswith('m6_1.')]
    require(final_affinity, 'Frozen original model must contain its final affinity head')
    require(n_synthetic >= 0, 'Invalid synthetic draw count')
    return final_affinity if n_synthetic == 0 else []


def gradient_ledger_from_history(parameter_names, history):
    """Reconstruct every parameter's R2 update count from committed gradients."""
    names = sorted(parameter_names)
    require(names and len(names) == len(set(names)), 'Parameter names must be unique')
    ledger = {name: 0 for name in names}
    for row in history:
        missing = row['no_gradient_parameter_names']
        require(missing == expected_missing_gradient_names(names, row['n_synthetic']),
                'Committed missing-gradient names differ from the original DOPE graph')
        absent = set(missing)
        for name in names:
            ledger[name] += int(name not in absent)
    return ledger


def inspect_gradient_presence(named_parameters, n_synthetic):
    names = [name for name, _ in named_parameters]
    missing = sorted(name for name, parameter in named_parameters if parameter.grad is None)
    require(missing == expected_missing_gradient_names(names, n_synthetic),
            'Actual missing gradients differ from the original DOPE graph')
    return missing


def optimizer_parameter_steps(named_parameters, optimizer):
    require(len(optimizer.state) == len(named_parameters), 'Restored Adam parameter state count differs')
    return {name: int(optimizer.state[parameter]['step']) for name, parameter in named_parameters}


def validate_parameter_steps(named_parameters, optimizer, ledger, updates_completed):
    names = [name for name, _ in named_parameters]
    require(set(ledger) == set(names) and all(type(v) is int and 0 <= v <= updates_completed
                                             for v in ledger.values()),
            'Gradient-presence ledger keyset or counts differ')
    actual = optimizer_parameter_steps(named_parameters, optimizer)
    require(all(actual[name] == 828 + ledger[name] for name in names),
            'Actual per-parameter Adam steps differ from the gradient-presence ledger')
    return actual


def verify_final_recovery(torch, final_state, restored_state, updates_completed,
                          optimizer_steps, gradient_ledger):
    """Refuse partial or different states before preserving an existing final."""
    require(updates_completed == 828 and optimizer_steps and set(optimizer_steps) == set(gradient_ledger)
            and all(type(count) is int and 0 <= count <= 828
                    and optimizer_steps[name] == 828 + count for name, count in gradient_ledger.items()),
            'An existing final checkpoint requires a fully committed R2 resume')
    require(isinstance(final_state, dict) and set(final_state) == set(restored_state),
            'Existing final checkpoint state keyset differs from the complete resume')
    for key, value in restored_state.items():
        existing = final_state[key]
        require(isinstance(existing, torch.Tensor) and existing.shape == value.shape
                and existing.dtype == value.dtype and torch.equal(existing.cpu(), value.detach().cpu()),
                'Existing final checkpoint differs from the complete resume at ' + key)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def train_r2(H, network, optimizer, scaler, synthetic_records, real_entries,
             output_checkpoint, config, seed=42, active_resume_path=None):
    """828 R2 updates from restored original R1 state, final checkpoint only.

    Exactly one resumable optimizer state is retained for the active campaign
    arm. The driver may delete that new active state only after its final
    checkpoint and complete receipt have both been hash-verified. A resume for
    any other arm/fingerprint is refused. Original R1 states remain read-only.
    """
    torch, np = H.torch, H.np
    config = dict(config)
    require(config['loss_mode'] == LOSS_MODE and config['real_batch_policy'] == SAMPLING_MODE,
            'Count-proportional objective/sampling contract differs')
    require(config['epochs'] == 3 and config['steps_per_epoch'] == 276
            and config['batch_size'] == 16 and config['microbatch'] == 8
            and config['amp'] is True and config['lambda_real'] == 1.0,
            'Controlled R2 recipe differs')
    require(seed in (42, 43, 44), 'Training seed outside the frozen three-seed scope')
    require(synthetic_records and real_entries, 'Both selected source pools must exist')
    for records in (synthetic_records, real_entries):
        require(len({str(r['id']) for r in records}) == len(records), 'Selected source IDs must be unique')
    original_output_path = Path(output_checkpoint)
    require(not original_output_path.is_symlink(), 'Final checkpoint symlinks are forbidden')
    output_checkpoint = original_output_path.resolve()
    require(output_checkpoint.is_relative_to(OUT), 'Write only in this new campaign')
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    resume_path = Path(active_resume_path or OUT/'state/ACTIVE_RESUME.pth').resolve()
    require(resume_path.is_relative_to(OUT) and not resume_path.is_symlink(),
            'Only the new active campaign resume may be written')
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    final_already_exists = output_checkpoint.exists()
    require(not final_already_exists or (output_checkpoint.is_file() and resume_path.is_file()),
            'An existing final checkpoint requires the active resume for validated recovery')
    log_path = output_checkpoint.with_suffix('.pth.steps.jsonl')
    status_path = output_checkpoint.with_suffix('.pth.status.json')
    exposure_path = output_checkpoint.with_suffix('.pth.exposures.jsonl')
    exposure_summary_path = output_checkpoint.with_suffix('.pth.exposure_summary.json')
    fingerprint = H._fingerprint(synthetic_records, real_entries, config, seed, 2)
    run_identity = dict(output_checkpoint=str(output_checkpoint), fingerprint=fingerprint,
                        training_seed=int(seed), round_idx=2)
    random.seed(seed + 200000)
    np.random.seed((seed + 200000) % 2**32)
    torch.manual_seed(seed + 200000)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    network.float().to('cuda').requires_grad_(True).train()
    named_parameters = list(network.named_parameters())
    parameter_names = [name for name, _ in named_parameters]
    expected_missing_gradient_names(parameter_names, 0)
    syn = H.NativeRows(synthetic_records, cache_size=config['cache_size'], sigma=config['sigma'])
    real = H.NativeRows(real_entries, pseudo=True, cache_size=config['cache_size'], sigma=config['sigma'])
    require(len(set(syn.shapes)) == len(set(real.shapes)) == 1,
            'Each source must retain one original prepared shape')
    augmentation = H.AUGMENTATION.StrongAugmentation(config['augmentation'])
    done, history, elapsed_before = 0, [], 0.0
    ledger = {name: 0 for name in parameter_names}
    if resume_path.exists():
        saved = torch.load(resume_path, map_location='cuda', weights_only=False)
        require(saved['run_identity'] == run_identity and saved['fingerprint'] == fingerprint,
                'Active resume belongs to a different arm or changed inputs')
        network.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        done, history = int(saved['updates_completed']), saved['history']
        ledger = saved['per_parameter_gradient_updates']
        require([row['update'] for row in history] == list(range(1, done + 1)), 'Resume history differs')
        random.setstate(saved['random_state'])
        np.random.set_state(saved['numpy_state'])
        torch.set_rng_state(saved['torch_state'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_state']])
        elapsed_before = saved['elapsed_seconds']
        del saved
    require(0 <= done <= 828, 'Invalid local update count')
    schedule, metadata = joint_schedule(len(syn), len(real), 16, 828, seed + 2000000)
    decoded = [split_batch(batch, synthetic_records, real_entries) for batch in schedule]
    metadata['joint_id_schedule_sha256'] = digest_json([joint for _, _, joint in decoded])
    schedules = [dict(spans_epochs=[1, 2, 3], continuous_epoch_boundaries=True, **metadata)]
    batches = [decoded[start:start + 276] for start in range(0, 828, 276)]
    validate_history(history, decoded, synthetic_records, real_entries)
    require(ledger == gradient_ledger_from_history(parameter_names, history),
            'Active resume gradient ledger differs from its committed history')
    parameter_steps = validate_parameter_steps(named_parameters, optimizer, ledger, done)
    final_recovery_sha256 = None
    if final_already_exists:
        require(done == 828 and len(history) == 828,
                'An existing final checkpoint cannot recover from a partial active resume')
        final_recovery_sha256 = sha_file(output_checkpoint)
        final_state = torch.load(output_checkpoint, map_location='cpu', weights_only=True)
        verify_final_recovery(torch, final_state, network.state_dict(), done,
                              parameter_steps, ledger)
        require(sha_file(output_checkpoint) == final_recovery_sha256,
                'Existing final checkpoint changed during recovery verification')
        del final_state
    begin = time.monotonic()

    def elapsed():
        return elapsed_before + time.monotonic() - begin

    exposure_keys = ('update', 'epoch', 'epoch_step', 'synthetic_ids', 'real_ids', 'joint_draws',
                     'n_synthetic', 'n_real', 'synthetic_loss_coefficient', 'real_loss_coefficient',
                     'no_gradient_parameter_names')

    def exposure_summary():
        source_counts = {source: Counter(draw['id'] for row in history for draw in row['joint_draws']
                                        if draw['source'] == source) for source in ('synthetic', 'real')}
        result = dict(schema='joint_pool_per_image_exposure_v1', run_identity=run_identity,
            committed_updates=done, total_committed_draws=16 * done,
            sampling_policy=SAMPLING_MODE, sources={})
        for source, records in [('synthetic', synthetic_records), ('real', real_entries)]:
            counts = source_counts[source]
            result['sources'][source] = dict(unique_pool=len(records), committed_draws=sum(counts.values()),
                unique_exposed=len(counts), unique_unexposed=len(records) - len(counts),
                per_image=[dict(id=str(record['id']), draws=counts[str(record['id'])]) for record in records])
        require(sum(v['committed_draws'] for v in result['sources'].values()) == 16 * done,
                'Exposure accounting differs')
        return result

    def status(complete=False):
        exposures = exposure_summary()
        H._atomic_json(exposures, exposure_summary_path)
        actual_parameter_steps = optimizer_parameter_steps(named_parameters, optimizer)
        step_validation = all(actual_parameter_steps[name] == 828 + ledger[name]
                              for name in parameter_names)
        value = dict(complete=complete, fingerprint=fingerprint, run_identity=run_identity,
            round_idx=2, training_seed=int(seed), updates_completed=done, updates_required=828,
            global_optimizer_steps=828 + done,
            optimizer_parameter_steps=actual_parameter_steps,
            optimizer_parameter_initial_steps={name: 828 for name in parameter_names},
            optimizer_parameter_step_values=sorted(set(actual_parameter_steps.values())),
            per_parameter_gradient_updates=dict(ledger), parameter_step_validation_passed=step_validation,
            epochs=3, effective_batch_size=16, microbatch=8,
            amp=True, loss_dtype='float32', parameter_dtype='float32',
            loss_mode=LOSS_MODE, sampling_mode=SAMPLING_MODE, lambda_real=1.0,
            original_branch_denominator='valid_channels_times_pixels_over_present_source_draws',
            loss_formula='2*ns/16*Lsyn_original_mean + 2*nr/16*Lreal_original_mean',
            pseudo_samples=len(real), synthetic_unique_pool=len(syn), real_unique_pool=len(real),
            committed_synthetic_draws=exposures['sources']['synthetic']['committed_draws'],
            committed_real_draws=exposures['sources']['real']['committed_draws'],
            committed_total_draws=16 * done, elapsed_seconds=elapsed(), schedules=schedules,
            history=history, scaler_state=scaler.state_dict(), active_resume_path=str(resume_path),
            final_checkpoint_recovery=final_already_exists,
            preserved_final_checkpoint_sha256=final_recovery_sha256,
            exposure_summary_path=str(exposure_summary_path),
            exposure_summary_sha256=hashlib.sha256(exposure_summary_path.read_bytes()).hexdigest())
        H._atomic_json(value, status_path)
        return value

    def save():
        H._atomic_torch_save(dict(fingerprint=fingerprint, run_identity=run_identity,
            updates_completed=done, state_dict=network.state_dict(), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), history=history, elapsed_seconds=elapsed(),
            per_parameter_gradient_updates=dict(ledger),
            random_state=random.getstate(), numpy_state=np.random.get_state(),
            torch_state=torch.get_rng_state(), cuda_state=torch.cuda.get_rng_state_all()), resume_path)

    # On a resume the committed checkpoint history is authoritative; uncommitted
    # log tail is replaced, so no duplicated/skipped exposure enters the report.
    log_path.write_text(''.join(json.dumps(row) + '\n' for row in history))
    exposure_path.write_text(''.join(json.dumps({key: row[key] for key in exposure_keys}) + '\n'
                                     for row in history))
    status()
    for epoch, decoded in enumerate(batches):
        epoch_start_done = done
        for local, (si, ri, joint) in enumerate(decoded):
            if epoch * 276 + local < done:
                continue
            optimizer.zero_grad(set_to_none=True)
            try:
                validate_parameter_steps(named_parameters, optimizer, ledger, done)
                losses = backward_count_weighted(H, network, syn, real, si, ri, 'cuda', 8,
                                                 augmentation, True, scaler)
                missing_gradients = inspect_gradient_presence(named_parameters, len(si))
                scaler.unscale_(optimizer)
                finite = [torch.isfinite(parameter.grad).all() for parameter in network.parameters()
                          if parameter.grad is not None]
                require(finite and torch.stack(finite).all().item(),
                        'Nonfinite gradients; no skipped update is counted')
                next_ledger = {name: count + int(name not in missing_gradients)
                               for name, count in ledger.items()}
                scaler.step(optimizer)
                validate_parameter_steps(named_parameters, optimizer, next_ledger, done + 1)
                scaler.update()
            except Exception as error:
                H._atomic_json(dict(failed=True, updates_completed=done, error_type=type(error).__name__,
                    error=str(error), fingerprint=fingerprint, run_identity=run_identity),
                    output_checkpoint.with_suffix('.pth.failure.json'))
                status()
                raise
            done += 1
            ledger = next_ledger
            row = dict(update=done, epoch=epoch + 1, epoch_step=local + 1, **losses,
                n_synthetic=len(si), n_real=len(ri),
                synthetic_ids=[str(synthetic_records[i]['id']) for i in si],
                real_ids=[str(real_entries[i]['id']) for i in ri], joint_draws=joint,
                no_gradient_parameter_names=missing_gradients,
                scaler_scale=float(scaler.get_scale()), elapsed_seconds=elapsed())
            history.append(row)
            with log_path.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            with exposure_path.open('a') as handle:
                handle.write(json.dumps({key: row[key] for key in exposure_keys}) + '\n')
            if done % config['log_interval'] == 0 or done == 828:
                print(f'PROPORTIONAL_R2 seed={seed} update={done}/828 ns={len(si)} nr={len(ri)} '
                      f'loss={losses["loss_total"]:.6f} elapsed={elapsed():.1f}s', flush=True)
                status()
            if done % config['resume_interval'] == 0:
                save()
        if done > epoch_start_done:
            save()
    require(done == 828 and ledger == gradient_ledger_from_history(parameter_names, history),
            'Final global update budget / committed gradient ledger differs')
    validate_parameter_steps(named_parameters, optimizer, ledger, done)
    require(sum(len(row['joint_draws']) for row in history) == 13248,
            'Complete 16-image update exposure budget differs')
    if final_already_exists:
        require(sha_file(output_checkpoint) == final_recovery_sha256,
                'Existing final checkpoint changed before completed recovery')
    else:
        H._atomic_torch_save({key: value.detach().cpu() for key, value in network.state_dict().items()},
                            output_checkpoint)
    stats = status(True)
    return dict(optimizer=optimizer, scaler=scaler, stats=stats)
