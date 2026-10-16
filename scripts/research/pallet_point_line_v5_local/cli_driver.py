"""Freeze the contract, train nine cells, score, evaluate, compare, then judge.

Foreground and synchronous. No automatic retry, no silent restart, no overwrite.
The verdict rule is written before the numbers exist.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
RUN = ROOT / 'data/pallet/results/pallet_point_line_v5_local'
EXPORT = RUN / 'export'
sys.path.insert(0, str(KIT))
from cli_train import CFG, sha                                  # noqa: E402

ARMS, SEEDS, STEPS, BATCH, LR = ('point', 'direct', 'hough'), (1, 2, 3), 2000, 4, 1e-4
ENV = dict(os.environ, PYTHONPATH=f"{KIT}{os.pathsep}{os.environ.get('PYTHONPATH','')}")


def run(args, log=None, timeout=3600):
    cmd = [sys.executable, '-u', *args]
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, 'w') as f:
            p = subprocess.run(cmd, cwd=str(ROOT), env=ENV, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        tail = Path(log).read_text()[-2500:]
    else:
        p = subprocess.run(cmd, cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=timeout)
        tail = (p.stdout + p.stderr)[-2500:]
    if p.returncode != 0:
        raise RuntimeError(f'{args[0]} failed rc={p.returncode}\n{tail}')
    return tail


def main():
    started = time.monotonic()
    lock = RUN / 'PROTOCOL_LOCK.json'
    if not lock.exists():
        lock.write_text(json.dumps(dict(
            schema='plpose_local_protocol_1', locked=True, arms=list(ARMS), seeds=list(SEEDS),
            steps=STEPS, batch=BATCH, lr=LR, config=CFG,
            manifest_sha256={s: sha(EXPORT / f'{s}.json') for s in ('train', 'calibration', 'synth_val')},
            source_sha256={p.name: sha(p) for p in sorted((KIT / 'plpose_local').glob('*.py'))},
            cli_sha256={p.name: sha(p) for p in sorted(KIT.glob('cli_*.py'))},
            budget_cap_optimizer_steps=len(ARMS) * len(SEEDS) * STEPS,
            gate=dict(relative_gain_over_reference=.01, median_p90_nonworse=True,
                      coverage_nonworse=True, new_catastrophic_frames=0, required_seeds=3,
                      basis='external v5 instruction section 8; unvalidated exploratory budget, not a '
                            'safety certification'),
            symmetry_note='C4 is unreachable in this population (no asset has a confirmed 90 degree '
                          'equivalence); only C1 and C2 are exercised.'),
            ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print('protocol locked', flush=True)

    cells = {}
    for arm in ARMS:
        for seed in SEEDS:
            out = RUN / 'cache_heads' / f'{arm}_seed{seed}'
            done = out / 'COMPLETION.json'
            if done.exists():
                d = json.loads(done.read_text())
                if d['optimizer_steps'] != STEPS:
                    raise SystemExit(f'INCOMPLETE_CELL_REQUIRES_RECONCILIATION: {out}')
                print(f'reuse {arm}_seed{seed}', flush=True)
            elif out.exists() and any(out.iterdir()):
                raise SystemExit(f'INCOMPLETE_CELL_REQUIRES_RECONCILIATION: {out}')
            else:
                t0 = time.monotonic()
                run([str(KIT / 'cli_train.py'), '--manifest', str(EXPORT / 'train.json'),
                     '--output', str(out), '--arm', arm, '--seed', str(seed),
                     '--steps', str(STEPS), '--batch', str(BATCH), '--lr', str(LR)],
                    log=RUN / 'logs' / f'train_{arm}_seed{seed}.log')
                d = json.loads(done.read_text())
                if d['optimizer_steps'] != STEPS:
                    raise SystemExit(f'cell did not reach the budget: {out}')
                print(f'trained {arm}_seed{seed} in {time.monotonic()-t0:.0f}s', flush=True)
            cells[(arm, seed)] = dict(start=json.loads((out / 'START.json').read_text()), completion=d)

    binding = dict(distinct_checkpoints=len({c['completion']['checkpoint_sha256'] for c in cells.values()}),
                   distinct_final_states=len({c['completion']['final_state_sha256'] for c in cells.values()}))
    for seed in SEEDS:
        binding[f'shared_init_identical_seed{seed}'] = (
            len({cells[(a, seed)]['start']['shared_state_sha256'] for a in ARMS}) == 1)
        binding[f'plan_identical_seed{seed}'] = (
            len({cells[(a, seed)]['start']['plan_sha256'] for a in ARMS}) == 1)
    binding['trainable'] = {a: cells[(a, 1)]['start']['trainable'] for a in ARMS}
    binding['trainable_line_branch'] = {a: cells[(a, 1)]['start']['trainable_line_branch'] for a in ARMS}
    binding['PASS'] = bool(binding['distinct_checkpoints'] == 9 and binding['distinct_final_states'] == 9
                           and all(v for k, v in binding.items() if k.startswith(('shared_init', 'plan'))))
    (RUN / 'BINDING_CHECK.json').write_text(json.dumps(binding, ensure_ascii=False, indent=2) + '\n',
                                            encoding='utf8')
    if not binding['PASS']:
        raise SystemExit('binding check failed: ' + json.dumps(binding))

    for arm in ARMS:
        for seed in SEEDS:
            ck = RUN / 'cache_heads' / f'{arm}_seed{seed}' / 'checkpoint_final.pt'
            for split in ('calibration', 'synth_val'):
                pred = RUN / 'predictions' / f'{arm}_seed{seed}_{split}.json'
                ev = RUN / 'evaluations' / f'{arm}_seed{seed}_{split}.json'
                if not pred.exists():
                    run([str(KIT / 'cli_assess.py'), 'score', '--manifest', str(EXPORT / f'{split}.json'),
                         '--checkpoint', str(ck), '--output', str(pred)])
                if not ev.exists():
                    print(run([str(KIT / 'cli_assess.py'), 'evaluate', '--manifest',
                               str(EXPORT / f'{split}.json'), '--predictions', str(pred),
                               '--output', str(ev)]).strip(), flush=True)

    comparisons = {}
    for name, ref, met in (('H_VS_P', 'point', 'hough'), ('H_VS_D', 'direct', 'hough'),
                           ('D_VS_P', 'point', 'direct')):
        p = RUN / f'{name}.json'
        if not p.exists():
            args = [str(KIT / 'cli_assess.py'), 'compare', '--reference']
            args += [str(RUN / 'evaluations' / f'{ref}_seed{s}_synth_val.json') for s in SEEDS]
            args += ['--method'] + [str(RUN / 'evaluations' / f'{met}_seed{s}_synth_val.json') for s in SEEDS]
            args += ['--output', str(p)]
            run(args)
        comparisons[name] = json.loads(p.read_text())

    primary = comparisons['H_VS_P']
    verdict = dict(
        schema='plpose_local_verdict_1', execution_completed=True,
        integrity_valid=bool(binding['PASS']),
        primary_comparison='hough versus point (fixed before the run)',
        synthetic_signal=bool(primary['numeric_gate_passed']),
        real_evaluation_executed=False, online_full_network_executed=False,
        stop_reason=None if primary['numeric_gate_passed'] else 'NO_SYNTHETIC_ADVANCEMENT_SIGNAL',
        comparisons={k: dict(numeric_gate_passed=v['numeric_gate_passed'],
                             mean_difference=v['seed_averaged_difference_mean'],
                             checks=v['checks'], bootstrap=v['session_bootstrap'])
                     for k, v in comparisons.items()},
        per_arm_primary={a: [json.loads((RUN / 'evaluations' / f'{a}_seed{s}_synth_val.json').read_text())
                             ['primary_normalized_mean'] for s in SEEDS] for a in ARMS},
        per_arm_coverage={a: [json.loads((RUN / 'evaluations' / f'{a}_seed{s}_synth_val.json').read_text())
                              ['coverage'] for s in SEEDS] for a in ARMS},
        trainable=binding['trainable'], trainable_line_branch=binding['trainable_line_branch'],
        accuracy_claim_scope='Synthetic, detection-conditioned, fixed-backbone cached-feature heads in an '
                             'independent re-implementation. Not the author v5 core, not a real-image '
                             'result, not an online full-network result.',
        elapsed_seconds=time.monotonic() - started)
    (RUN / 'VERDICT.json').write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: verdict[k] for k in ('execution_completed', 'integrity_valid', 'synthetic_signal',
                                              'stop_reason', 'per_arm_primary')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
