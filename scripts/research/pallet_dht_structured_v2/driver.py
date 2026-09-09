"""Run the registered learned-verifier pilot; a pilot is not goal completion."""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timezone
import torch
from .cache import read,write,sha
from .train import train_arm
from .evaluation import calibrate_and_evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();run=args.run_dir.resolve()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    protocol=read(run/'PROTOCOL.json')
    frozen=read(run/'SOURCE_FREEZE.json')
    def check_sources():
        for path,expected in frozen['source_sha256'].items():
            if sha(path)!=expected:raise ValueError(f'Frozen pilot source changed: {path}')
    check_sources()
    cells=[]
    for seed in protocol['training']['seeds']:
        for arm in protocol['architecture']['arms']:
            write(run/'DRIVER_PROGRESS.json',dict(stage='training',arm=arm,seed=seed,active_goal_complete=False))
            cells.append(train_arm(run,arm,seed,args.device))
    for seed in protocol['training']['seeds']:
        grouped=[c for c in cells if c['seed']==seed]
        for key in ('initial_state_tensor_sha256','trace_sha256'):
            if len({c[key] for c in grouped})!=1:raise ValueError(f'Arms differ in matched {key}')
    check_sources()
    write(run/'TRAINING_COMPLETION.json',dict(complete=True,PASS=True,runs=cells,
        identical_initial_state_and_full_trace_within_each_seed=True,
        protocol_sha256=sha(run/'PROTOCOL.json'),active_goal_complete=False))
    evaluations=[]
    for seed in protocol['training']['seeds']:
        for arm in protocol['architecture']['arms']:
            write(run/'DRIVER_PROGRESS.json',dict(stage='synthetic_calibration_and_validation',arm=arm,seed=seed,active_goal_complete=False))
            evaluations.append(calibrate_and_evaluate(run,arm,seed,args.device,batch_size=8))
    check_sources()
    write(run/'PILOT_RESULTS.json',dict(complete=True,generated_at_utc=datetime.now(timezone.utc).isoformat(),
        evaluations=evaluations,protocol_sha256=sha(run/'PROTOCOL.json'),
        active_goal_complete=False,real_evaluation_done=False,
        meaning='Synthetic pilot only. Read advancement checks before real evaluation; no claim of real accuracy improvement.'))
    write(run/'DRIVER_PROGRESS.json',dict(stage='synthetic_pilot_finished',active_goal_complete=False))
    print('Synthetic pilot finished; active user goal remains open.',flush=True)


if __name__=='__main__':main()
