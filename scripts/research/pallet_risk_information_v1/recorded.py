"""Record one actual child command, raw logs, exit and nonduplicated batch time."""
from __future__ import annotations
import argparse
from pathlib import Path
import resource
import subprocess
import time
from .common import append_log, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("An actual command is required")
    folder = args.run_dir / "execution" / args.name
    folder.mkdir(parents=True, exist_ok=False)
    start, epoch = time.monotonic(), time.time()
    append_log(args.run_dir / "RUN_LOG.jsonl", dict(event="BATCH_START", name=args.name,
        command=command, original_argv_preserved=True))
    with open(folder / "stdout.log", "xb") as stdout, open(folder / "stderr.log", "xb") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = dict(status="COMPLETED" if completed.returncode == 0 else "FAILED",
        name=args.name, command=command, actual_exit_code=completed.returncode,
        start_epoch=epoch, end_epoch=time.time(), wall_seconds=time.monotonic() - start,
        child_CPU_user_seconds=usage.ru_utime, child_CPU_system_seconds=usage.ru_stime,
        child_peak_rss_kib=usage.ru_maxrss, standalone_batch_wall=True,
        CUDA_activity_seconds=None, CUDA_note="not inferred from CPU or whole process wall",
        log_hashes={p.name: sha(p) for p in folder.iterdir() if p.is_file()})
    write(folder / "RECEIPT.json", receipt)
    append_log(args.run_dir / "RUN_LOG.jsonl", dict(event="BATCH_EXIT", **receipt))
    print(args.name, completed.returncode, f"{receipt['wall_seconds']:.3f}s", flush=True)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
