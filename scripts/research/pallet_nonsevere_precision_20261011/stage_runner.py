"""Record a sequential stage's conservative device-reserved wall time.

This is process wall time, including imports and CPU work, not CUDA kernel
occupancy. Full stdout stays private. It never changes numerical arguments.
"""
import argparse, hashlib, json, resource, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--private-dir', required=True)
    parser.add_argument('--public-dir', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command and '/' not in args.stage and '\\' not in args.stage
    private, public = Path(args.private_dir), Path(args.public_dir)
    receipt = public / ('STAGE_' + args.stage + '.json')
    log = private / ('STAGE_' + args.stage + '.log')
    assert not receipt.exists() and not log.exists(), 'Existing stage retained'
    start_utc = datetime.now(timezone.utc).isoformat()
    begin = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with log.open('x') as output:
        child = subprocess.Popen(command, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
        for line in child.stdout:
            output.write(line)
            output.flush()
            print(line, end='', flush=True)
        returncode = child.wait()
    elapsed = time.monotonic() - begin
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    data = dict(status='DONE' if returncode == 0 else 'FAILED',
                stage=args.stage, start_utc=start_utc,
                end_utc=datetime.now(timezone.utc).isoformat(),
                wall_seconds=elapsed, device=args.device,
                device_reserved_upper_seconds=elapsed if args.device == 'cuda' else 0,
                timing_scope='Whole child process including imports, load, CPU work; not kernel occupancy',
                child_CPU_user_seconds=usage_after.ru_utime - usage_before.ru_utime,
                child_CPU_system_seconds=usage_after.ru_stime - usage_before.ru_stime,
                returncode=returncode,
                private_log_name=log.name,
                private_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                command_basename=Path(command[0]).name,
                numerical_configuration_changed=False)
    with receipt.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print('STAGE_RECEIPT', args.stage, returncode, round(elapsed, 3), flush=True)
    raise SystemExit(returncode)


if __name__ == '__main__':
    main()
