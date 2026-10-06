"""Account completed scientific rows and measured calls without rerunning them."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import json
from .run import DOC, read, sha, write


def generate():
    summary_path = DOC / 'results/A_summary.json'
    summary = read(summary_path)
    fits = summary['fits']
    assert len(fits) == 6 and sum(f['updates'] for f in fits) == 36000
    paths = [summary_path, Path(__file__)]
    oracles = {}
    for split in ('SOURCE', 'SYNTH_HELDOUT', 'REAL_DEV'):
        path = DOC / f'results/A_{split}_ORACLE.json'; paths.append(path)
        x = read(path); rows = x['rows']
        oracles[split] = {'population_frames': len(rows) // 2, 'bank_frames_GEO_and_PERM': len(rows),
                          'actual_candidate_F_calls': sum(r['actions'] for r in rows),
                          'candidate_F_failures': sum(r['failures'] for r in rows),
                          'wall_seconds': x['seconds'], 'deployment': False}
    final_rows = {}; head_frames = 0; head_batches = 0; evaluation_seconds = 0.
    for split in ('SYNTH_HELDOUT', 'REAL_DEV'):
        baseline_path = DOC / f'results/A_{split}_BASELINES.json'; paths.append(baseline_path)
        base = read(baseline_path)
        final_rows[split] = {'baseline_final_F_calls': sum(len(rr) for rr in base['rows'].values()),
                            'frozen_final_F_calls': 0, 'fit_final_F_calls': 0}
        for phase in ('FROZEN', 'FIT'):
            for path in sorted((DOC / 'results').glob(f'A_{split}_{phase}_*seed*.json')):
                x = read(path); paths.append(path)
                final_rows[split][phase.lower() + '_final_F_calls'] += sum(len(rr) for rr in x['rows'].values())
                evaluation_seconds += x['seconds']
                arms = ('GEO', 'PERM') if phase == 'FROZEN' else (path.stem.split('_')[-2],)
                # Source evaluates all frames in batch16, including NoOp rows.
                # Real skips heads only for bank1/no-detection; oracle records
                # preserve the actual action count of each frame/arm.
                oracle = read(DOC / f'results/A_{split}_ORACLE.json')['rows']
                for arm in arms:
                    rr = [r for r in oracle if r['arm'] == arm]
                    count = len(rr) if split == 'SYNTH_HELDOUT' else sum(r['actions'] > 1 for r in rr)
                    head_frames += count
                    head_batches += math.ceil(count / 16) if split == 'SYNTH_HELDOUT' else count
    runtime = {}
    for key, rel in (('matched_valid', 'RUNTIME_MATCHED.json'), ('matched_discarded', 'results/discarded_runtime_tf32_off/RUNTIME_MATCHED.json'), ('A_valid', 'RUNTIME_A.json'), ('A_discarded_concurrent_analysis', 'results/discarded_runtime_a_concurrent_analysis/RUNTIME_A.json')):
        path = DOC / rel
        if not path.exists():
            raise FileNotFoundError('Completed runtime dependency required: ' + str(path))
        x = read(path); paths.append(path)
        runtime[key] = {'warmup_calls': x['warmup_calls'], 'measured_calls': x['measured_calls'],
                        'calls': x['warmup_calls'] + x['measured_calls'],
                        'wall_seconds': x['elapsed_seconds'], 'optimizer_updates': 0,
                        'included_in_reported_cost_table': key in ('matched_valid', 'A_valid')}
    mask_path = DOC / 'results/A_SAMPLING_MASK_RECEIPT.json'
    mask = read(mask_path); paths.append(mask_path)
    result = {'schema': 'joint_action_execution_counts_v1', 'status': 'DONE',
              'formal': {'fits': len(fits), 'updates': sum(f['updates'] for f in fits),
                         'training_forward_bank_calls': sum(f['updates'] for f in fits),
                         'exposures': sum(f['exposures'] for f in fits), 'wall_seconds_sum': sum(f['seconds'] for f in fits),
                         'failed_formal_fits': 0, 'failure_count_basis': 'Uninterrupted formal process and progress were monitored; pre-fit stopped CPU attempts had zero updates. Completed receipts alone are not proof of absence of other attempts.',
                         'parameter_count_each': sorted({f['params'] for f in fits}),
                         'real_training_exposures': sum(f['real_training'] for f in fits)},
              'discarded_smoke': {'fits': 2, 'updates': summary['smoke_updates'], 'exposures': summary['smoke_updates'] * 16,
                                  'training_forward_bank_calls': summary['smoke_updates'],
                                  'weights_reused_for_formal_initialization': False},
              'oracle': oracles,
              'offline_candidate_F_calls': sum(v['actual_candidate_F_calls'] for v in oracles.values()),
              'offline_oracle_wall_seconds_sum': sum(v['wall_seconds'] for v in oracles.values()),
              'baseline_frozen_fit_final_F_rows': final_rows,
              'baseline_frozen_fit_final_F_calls': sum(sum(v.values()) for v in final_rows.values()),
              'frozen_and_fit_frame_head_evaluations': head_frames,
              'frozen_and_fit_batched_forward_bank_calls': head_batches,
              'frozen_and_fit_evaluation_wall_seconds_sum': evaluation_seconds,
              'runtime': runtime, 'runtime_calls_including_discarded': sum(v['calls'] for v in runtime.values()),
              'sampling_mask_reconstruction': {'frames': len(mask['frame_rows']),
                                               'real_bank_initialization': mask['real_initialization'],
                                               'source_bank_initialization_PnP_calls': mask['source_initialization_PnP_calls'],
                                               'CNN_forward_calls': mask['new_CNN_forward_calls'],
                                               'final_F_scoring_calls': mask['new_final_F_scoring_calls'],
                                               'optimizer_updates': mask['new_optimizer_updates'],
                                               'wall_seconds': mask['seconds_wall'],
                                               'executed_GPU_mask_tensors_preserved': False},
              'derivation_scope': 'Completed receipt rows + executed evaluate batching/skip contract. Frozen/fit head counts are evaluation only; training forwards are listed separately. Separate oracle candidate solves, cached-feature heads, final F outputs and live full-path timings. Counts exclude CPU mathematical/fixture forwards and initialization selector/solve calls during bank generation; those are contract checks or generation work, not extra fits. Resuming/aggregating completed rows adds no inference or update.',
              'stopped_precompute_attempts': {'formal_updates': 0, 'smoke_updates': 0,
                                             'exact_attempt_wall_seconds': None, 'reason': 'Earlier interrupted CPU attempts lacked complete elapsed receipts; no fabricated total'},
              'bindings': [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths]}
    write(DOC / 'results/EXECUTION_COUNTS.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(generate(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
