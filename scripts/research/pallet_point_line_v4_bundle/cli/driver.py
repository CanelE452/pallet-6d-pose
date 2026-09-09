"""Stage A+B foreground runner: train 12 cells -> calibrate -> synth_val -> verdict -> notify.

No background detach, no broad pgrep waiting, no automatic retry, no overwrite.
Completion is decided by artifacts, never by exit code alone.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from cli.common import ROOT, RESULTS, read, write, sha256  # noqa: E402

EXPORT = RESULTS / 'export'
ARMS = ('P', 'S', 'H', 'HA')
SEEDS = (1, 2, 3)
CELL_TIMEOUT = 3600
PY_ENV = dict(os.environ, PYTHONPATH=f"{KIT}{os.pathsep}{os.environ.get('PYTHONPATH','')}")


def run(args, *, timeout=CELL_TIMEOUT, log=None):
    """Stream child output to a live log so progress stays visible, no pipe buffering."""
    started = time.monotonic()
    command = [sys.executable, '-u', '-m', *args]
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with Path(log).open('w', encoding='utf8') as stream:
            proc = subprocess.run(command, cwd=str(RESULTS), env=PY_ENV, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=timeout)
        tail = Path(log).read_text(encoding='utf8')[-3000:]
    else:
        proc = subprocess.run(command, cwd=str(RESULTS), env=PY_ENV, capture_output=True,
                              text=True, timeout=timeout)
        tail = proc.stderr[-3000:]
    if proc.returncode != 0:
        raise RuntimeError(f'{args[0]} failed rc={proc.returncode}\n{tail}')
    return time.monotonic() - started


def core_sha():
    return {p.name: sha256(p) for p in sorted((KIT / 'pointline_v4').glob('*.py'))}


def train_cells(protocol, steps):
    cells = {}
    for arm in ARMS:
        for seed in SEEDS:
            out = RESULTS / 'heads' / f'{arm}_seed{seed}'
            completion = out / 'COMPLETION.json'
            if completion.exists():
                done = read(completion)
                if not done['training_completed'] or done['optimizer_steps'] != steps:
                    raise SystemExit(f'INCOMPLETE_CELL_REQUIRES_RECONCILIATION: {out}')
                if done['checkpoint_sha256'] != sha256(out / 'checkpoint_final.pt'):
                    raise SystemExit(f'INCOMPLETE_CELL_REQUIRES_RECONCILIATION: checkpoint changed {out}')
                print(f'reuse verified cell {arm}_seed{seed}', flush=True)
            elif out.exists() and any(out.iterdir()):
                raise SystemExit(f'INCOMPLETE_CELL_REQUIRES_RECONCILIATION: {out}')
            else:
                elapsed = run(['pointline_v4.train_cache', '--manifest', str(EXPORT / 'train.json'),
                               '--protocol', str(RESULTS / 'PROTOCOL_LOCK.json'), '--output', str(out),
                               '--arm', arm, '--seed', str(seed), '--device', 'cuda:0'],
                              log=RESULTS / 'logs' / f'train_{arm}_seed{seed}.log')
                done = read(completion)
                if not done['training_completed'] or done['optimizer_steps'] != steps:
                    raise SystemExit(f'Cell did not reach the registered budget: {out}')
                print(f'trained {arm}_seed{seed} in {elapsed:.0f}s', flush=True)
            start = read(out / 'START.json')
            cells[(arm, seed)] = dict(output=str(out), completion=done, start=start)
    return cells


def check_bindings(cells):
    report = dict(distinct_checkpoints=len({c['completion']['checkpoint_sha256'] for c in cells.values()}))
    for seed in SEEDS:
        for key in ('initial_state_sha256', 'plan_sha256'):
            values = {cells[(a, seed)]['start'][key] for a in ARMS}
            report[f'{key}_seed{seed}_identical_across_arms'] = (len(values) == 1)
    distinct_states = len({c['completion']['final_state_sha256'] for c in cells.values()})
    report['distinct_final_states'] = distinct_states
    report['PASS'] = bool(report['distinct_checkpoints'] == 12 and distinct_states == 12
                          and all(v for k, v in report.items() if k.endswith('identical_across_arms')))
    return report


def stage_b(cells):
    timings = {}
    for arm in ARMS:
        for seed in SEEDS:
            tag = f'{arm}_seed{seed}'
            ckpt = RESULTS / 'heads' / tag / 'checkpoint_final.pt'
            cal = RESULTS / 'scores' / f'{tag}_cal.json'
            policy = RESULTS / 'policies' / f'{tag}.json'
            val = RESULTS / 'scores' / f'{tag}_synth.json'
            evaluation = RESULTS / 'evaluations' / f'{tag}_synth.json'
            if not cal.exists():
                cal.parent.mkdir(parents=True, exist_ok=True)
                run(['pointline_v4.score_cache', '--manifest', str(EXPORT / 'calibration.json'),
                     '--checkpoint', str(ckpt), '--output', str(cal), '--device', 'cuda:0'])
            if not policy.exists():
                policy.parent.mkdir(parents=True, exist_ok=True)
                run(['pointline_v4.evaluate_scores', 'calibrate', '--manifest', str(EXPORT / 'calibration.json'),
                     '--scores', str(cal), '--output', str(policy)])
            if not val.exists():
                run(['pointline_v4.score_cache', '--manifest', str(EXPORT / 'synth_val.json'),
                     '--checkpoint', str(ckpt), '--output', str(val), '--device', 'cuda:0'])
            if not evaluation.exists():
                evaluation.parent.mkdir(parents=True, exist_ok=True)
                run(['pointline_v4.evaluate_scores', 'evaluate', '--manifest', str(EXPORT / 'synth_val.json'),
                     '--scores', str(val), '--policy', str(policy), '--output', str(evaluation)])
            timings[tag] = dict(policy_margin=read(policy)['margin'],
                                changed_frames=read(evaluation)['result_8']['changed_frames'])
            print(f'{tag}: margin={timings[tag]["policy_margin"]} changed={timings[tag]["changed_frames"]}', flush=True)
    comparison = RESULTS / 'COMPARISON_synth_val.json'
    if not comparison.exists():
        args = ['pointline_v4.compare_evaluations']
        for arm in ARMS:
            args += ['--' + arm] + [str(RESULTS / 'evaluations' / f'{arm}_seed{s}_synth.json') for s in SEEDS]
        args += ['--output', str(comparison)]
        run(args)
    return timings, read(comparison)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--notify', action='store_true', help='User explicitly requested a completion message')
    args = ap.parse_args()
    lock = RESULTS / 'DRIVER_LOCK.json'
    if lock.exists():
        raise SystemExit(f'A driver lock exists: {lock}. Reconcile it deliberately.')
    write(lock, dict(pid=os.getpid(), started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     command=' '.join(sys.argv), host=os.uname().nodename))
    started = time.monotonic()
    try:
        protocol = read(RESULTS / 'PROTOCOL_LOCK.json')
        if protocol.get('locked') is not True:
            raise SystemExit('PROTOCOL_LOCK is not locked')
        if protocol['core_source_sha256'] != core_sha():
            raise SystemExit('Bundle core changed after locking')
        if not read(RESULTS / 'PREFLIGHT.json')['all_gates_PASS']:
            raise SystemExit('Preflight gates did not pass')
        steps = protocol['training']['steps']
        cells = train_cells(protocol, steps)
        bindings = check_bindings(cells)
        write(RESULTS / 'BINDING_CHECK.json', bindings)
        if not bindings['PASS']:
            raise SystemExit('Cross-arm binding check failed: ' + json.dumps(bindings))
        policies, comparison = stage_b(cells)

        identity_only_everywhere = all(v['policy_margin'] is None for v in policies.values())
        h_changed = sum(policies[f'H_seed{s}']['changed_frames'] for s in SEEDS)
        synthetic_signal = bool(comparison['synthetic_progression_signal']) and not identity_only_everywhere and h_changed > 0
        verdict = dict(
            schema='pointline_v4_verdict_1',
            execution_completed=True,
            integrity_valid=bool(bindings['PASS'] and read(RESULTS / 'PREFLIGHT.json')['all_gates_PASS']
                                 and read(RESULTS / 'EXPORT_COMPLETION.json')['strict_p4_parity_PASS']),
            synthetic_signal=synthetic_signal,
            real_cached_head_signal=None,
            full_network_executed=False,
            accuracy_claim_scope=('Synthetic source validation of a cached head on a frozen joint backbone. '
                                  'No real DEV inference, no full-network joint training, no FINAL.'),
            stop_reason=None if synthetic_signal else 'NO_SYNTHETIC_ADVANCEMENT_SIGNAL',
            registered_gate=dict(
                relative_primary_gain_over_baseline_and_P=0.01, median_p90_nonworse=True,
                maximum_good_point_damage_rate=0.01, required_head_seeds=3,
                identity_only_everywhere_fails=True),
            comparison_summary=dict(
                baseline_frame_normalized=comparison['baseline_frame_normalized'],
                per_seed_frame_normalized=comparison['per_seed_frame_normalized'],
                checks=comparison['checks'],
                contrasts={k: {kk: vv for kk, vv in v.items() if kk in ('mean', 'reason')}
                           for k, v in comparison['contrasts'].items()}),
            policies=policies,
            identity_only_everywhere=identity_only_everywhere,
            elapsed_seconds=time.monotonic() - started)
        write(RESULTS / 'VERDICT.json', verdict)
        print(json.dumps({k: verdict[k] for k in ('execution_completed', 'integrity_valid', 'synthetic_signal',
                                                  'full_network_executed', 'stop_reason')}, ensure_ascii=False))
        if args.notify:
            sys.path.insert(0, str(ROOT))
            from scripts.research.discord_notify import notify
            per_seed = verdict['comparison_summary']['per_seed_frame_normalized']
            base = verdict['comparison_summary']['baseline_frame_normalized']
            message = (f"pallet_point_line_v4 Stage A+B 완료 — 판정: "
                       f"{'합성 진입 신호 있음' if synthetic_signal else 'NO_SYNTHETIC_ADVANCEMENT_SIGNAL (실사 미진입)'}. "
                       f"baseline {base:.6f} / H {['%.6f' % x for x in per_seed['H']]} / "
                       f"P {['%.6f' % x for x in per_seed['P']]} (프레임평균 8코너 오차÷대각선, synth_val512). "
                       f"12셀 × 2,000 updates 실제 완료, 전체 네트워크 미실행.")
            notify(message, RESULTS, event='pointline_v4_stage_ab', evidence=RESULTS / 'VERDICT.json')
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
