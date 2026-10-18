"""Execute one declared batch and preserve actual exit, wall, CPU and peak RSS.

This wrapper performs no experiment selection and has no reference imports.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--batch-id',required=True);p.add_argument('--scope',required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:]if a.command[:1]==['--']else a.command
    if not command:raise ValueError('A concrete command is required')
    if a.receipt.exists():raise FileExistsError('Preserve previous execution receipts')
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);epoch=time.time()
    launch_error=None
    try:
        result=subprocess.run(command,check=False);exit_code=result.returncode
    except OSError as error:
        launch_error=dict(type=type(error).__name__,message=str(error));exit_code=127
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    value=dict(schema='ACTUAL_BATCH_EXECUTION_COST_V1',batch_id=a.batch_id,scope=a.scope,
        command=command,start_epoch=epoch,finish_epoch=time.time(),wall_seconds=time.monotonic()-started,
        child_user_CPU_seconds=after.ru_utime-before.ru_utime,child_system_CPU_seconds=after.ru_stime-before.ru_stime,
        child_peak_RSS_KiB=after.ru_maxrss,exit_code=exit_code,launch_error=launch_error,
        cost_scope='whole child process tree; no per-worker wall multiplication; peak RSS is wait4 child high-water mark, not sum of simultaneous RSS')
    a.receipt.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    raise SystemExit(exit_code)


if __name__=='__main__':main()
