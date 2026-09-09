"""Frozen local high-resolution two-arm training and synthetic evaluation."""
import argparse
from pathlib import Path
import torch
from scripts.research.pallet_dht_structured_v2.cache import read,write
from .train import train_arm,verify
from .evaluation import evaluate_synthetic


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--device',default='cuda:0');a=parser.parse_args()
    run=a.run_dir.resolve();p,_=verify(run);torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cells=[];results=[]
    for seed in p['training']['seeds']:
        grouped=[]
        for arm in p['model']['arms']:
            write(run/'DRIVER_PROGRESS.json',dict(stage='training',arm=arm,seed=seed,active_goal_complete=False))
            cell=train_arm(run,arm,seed,a.device);cells.append(cell);grouped.append(cell)
        for key in ('initial_state_tensor_sha256','trace_sha256','parameters'):
            if len({c[key] for c in grouped})!=1:raise ValueError(f'Matched arms differ: {key}')
    write(run/'TRAINING_COMPLETION.json',dict(complete=True,PASS=True,runs=cells,matched_initial_and_trace=True))
    for seed in p['training']['seeds']:
        for arm in p['model']['arms']:
            write(run/'DRIVER_PROGRESS.json',dict(stage='synthetic_evaluation',arm=arm,seed=seed,active_goal_complete=False))
            results.append(evaluate_synthetic(run,arm,seed,a.device))
    verify(run);write(run/'PILOT_RESULTS.json',dict(complete=True,PASS=True,results=results,active_goal_complete=False,real_evaluation_done=False))
    write(run/'DRIVER_PROGRESS.json',dict(stage='synthetic_pilot_finished',active_goal_complete=False))


if __name__=='__main__':main()
