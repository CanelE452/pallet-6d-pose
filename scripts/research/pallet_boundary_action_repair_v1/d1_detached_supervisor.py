"""Wait for only our detached recovery process and preserve its actual exit."""
from __future__ import annotations
import argparse,hashlib,json,os,resource,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,value):
    with Path(path).open('x')as f:f.write(json.dumps(value,indent=2)+'\n')


def run(args):
    root=Path(args.private);start=time.monotonic();started=utc()
    argv=[sys.executable,'-B','-m','scripts.research.pallet_boundary_action_repair_v1.d1_recovery_batch','run',
        '--private',args.private,'--old-run',args.old_run,'--original-run',args.original_run,
        '--typed-run',args.typed_run,'--workers',str(args.workers),'--root-GO']
    with(root/'D1_RECOVERY_LIVE.log').open('xb')as logfile:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=logfile,stderr=subprocess.STDOUT,close_fds=True)
        write_new(root/'D1_RECOVERY_SUPERVISOR_START.json',dict(schema='D1_DETACHED_SUPERVISOR_START_V1',
            supervisor_pid=os.getpid(),child_pid=child.pid,child_argv=argv,started_utc=started,
            supervisor_source_sha256=sha(__file__),reference_or_accuracy_reads=0))
        code=child.wait()
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    write_new(root/'D1_RECOVERY_SUPERVISOR_RECEIPT.json',dict(schema='D1_DETACHED_SUPERVISOR_EXIT_V1',
        status='COMPLETE'if code==0 else'FAILED_PROCESS',supervisor_pid=os.getpid(),child_pid=child.pid,
        child_actual_exit_code=code,started_utc=started,ended_utc=utc(),wall_seconds=time.monotonic()-start,
        exited_child_CPU_seconds=usage.ru_utime+usage.ru_stime,exited_child_max_RSS_KiB=usage.ru_maxrss,
        source_sha256=sha(__file__),reference_or_accuracy_reads=0,
        scope='Actual waited recovery child process including Python imports and final receipt serialization; worker sums and internal batch timers are separate.'))
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--old-run',required=True)
    p.add_argument('--original-run',required=True);p.add_argument('--typed-run',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();raise SystemExit(run(a))
