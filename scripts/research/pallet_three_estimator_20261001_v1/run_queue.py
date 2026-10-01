"""Serial experiment queue: actual processes, persistent logs, no outcome retries."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
NAME = Path(__file__).parent.name
RAW = ROOT / 'data/pallet/results' / NAME
DOC = ROOT / '_docs/experiments' / NAME
PYTHON = '/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
DOPE = 'pallet_dope_refiner_20261001_v1'
RESNET = 'pallet_resnet18_refiner_20261001_v1'
STOP = False
CHILD = None


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    pending.replace(path)


def stages():
    rows=[]
    def add(name, namespace, script, args, receipt, raw=False):
        rows.append(dict(name=name,namespace=namespace,script=f'scripts/research/{namespace}/{script}',
            args=args,receipt=f'{"data/pallet/results" if raw else "_docs/experiments"}/{namespace}/{receipt}'))
    add('resnet18_compute_smoke',RESNET,'baseline_train.py',['smoke'],'BASELINE_GPU_SMOKE.json')
    for namespace,label in ((DOPE,'dope'),(RESNET,'resnet18')):
        if namespace==RESNET:
            add(label+'_baseline_train',namespace,'baseline_train.py',['train'],'BASELINE_TRAINING_COMPLETE.json')
            add(label+'_head_smoke',namespace,'run.py',['smoke'],'GPU_SMOKE.json')
            add(label+'_source_cache',namespace,'run.py',['cache'],'SOURCE_CACHE_COMPLETE.json')
        for suffix,script,args,receipt in (
            ('head_train','run.py',['train'],'TRAINING_COMPLETE.json'),
            ('train_audit','audit_artifacts.py',['train'],'TRAINING_ARTIFACT_AUDIT.json'),
            ('source_validation','run.py',['validation'],'VALIDATION_OUTPUTS.json'),
            ('source_selection','run.py',['select'],'SYNTHETIC_HELDOUT.json'),
            ('dev_inference','inference.py',[],'DEV_INFERENCE_COMPLETE.json'),
            ('dev_score','evaluation.py',['score'],'DEV_RESULTS.json'),
            ('runtime','runtime.py',['measure'],'RUNTIME.json'),
            ('report','report.py',[],'REPORT_COMPLETE.json')):
            add(label+'_'+suffix,namespace,script,args,receipt)
    return rows


def stop(signum,frame):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:
        # Send only to our child, never unrelated GPU jobs.
        CHILD.send_signal(signal.SIGTERM)


def verify_completed(path):
    value=read(path)
    assert value.get('complete') is True or value.get('PASS') is True,path
    for key in ('protocol','checkpoint','final_checkpoint','predictions','selection','validation','code'):
        entries=value.get(key)
        if isinstance(entries,dict):entries=[entries]
        if not isinstance(entries,list):continue
        for entry in entries:
            if isinstance(entry,dict) and 'path' in entry and 'sha256' in entry:
                assert sha(ROOT/entry['path'])==entry['sha256'],entry
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))


def run(wait_pid):
    global CHILD
    RAW.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    with (RAW/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=stages();started=now();history=[]
        write(DOC/'QUEUE_PLAN.json',dict(schema='three_estimator_queue_v1',created=started,
            pipeline=plan,model_count=3,estimators=['historical YOLO','frozen DOPE/VGG','new source-trained ResNet18'],
            serial_GPU=True,no_outcome_based_retries=True,existing_cache_pid=wait_pid,
            publication='Reports are generated only from actual completed outputs; GitHub publication and manuscript review remain explicit final steps.',
            original_stable_TR_goal_achieved=False,code_sha256=sha(__file__)))
        def status(state,stage,**extra):
            value=dict(state=state,stage=stage,time=now(),queue_pid=os.getpid(),started=started,
                completed_stages=history,**extra)
            write(RAW/'QUEUE_STATUS.json',value)
            (DOC/'STATUS_KO.md').write_text('# 세 기반 추정기 실험 진행 상황\n\n'
                f'갱신(UTC): {value["time"]}\n\n상태: **{state}** · 단계: `{stage}`\n\n'
                'YOLO 기존 결과와 DOPE, SimpleBaseline-derived ResNet-18을 각각 동일 P/D 보정 전후로 비교합니다. '
                '실행 순서와 실제 로그를 보존합니다. 실행 중인 단계는 완료 결과가 아닙니다.\n\n'
                + '\n'.join(f'- 완료: `{r["name"]}`' for r in history)
                + '\n\n세 모델 결과·원고 최종 검토·GitHub 게시가 모두 끝나기 전에는 전체 완료로 표시하지 않습니다. '
                '실사 T/R 안정적 공동 개선은 아직 입증되지 않았습니다.\n')
        if wait_pid:
            status('WAITING_EXISTING_CACHE','dope_source_cache',child_pid=wait_pid)
            while True:
                try:os.kill(wait_pid,0)
                except ProcessLookupError:break
                if STOP:raise SystemExit('Queue stopped while waiting')
                time.sleep(5)
        verify_completed(ROOT/f'_docs/experiments/{DOPE}/SOURCE_CACHE_COMPLETE.json')
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/pallet-mpl',MPLBACKEND='Agg')
        for row in plan:
            if STOP:raise SystemExit('Queue stopped')
            receipt=ROOT/row['receipt']
            if receipt.exists():
                history.append(dict(name=row['name'],receipt=verify_completed(receipt),already_complete=True))
                continue
            command=[PYTHON,'-B',str(ROOT/row['script']),*row['args']]
            script_sha=sha(ROOT/row['script'])
            log=RAW/(row['name']+'.log')
            with log.open('a',buffering=1) as output:
                output.write(json.dumps(dict(event='START',time=now(),command=command,script_sha256=script_sha))+'\n')
                CHILD=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
                status('RUNNING',row['name'],child_pid=CHILD.pid,log=str(log.relative_to(ROOT)))
                code=CHILD.wait();CHILD=None
                output.write(json.dumps(dict(event='EXIT',time=now(),returncode=code))+'\n')
            if code:
                status('STOPPED_ON_ERROR',row['name'],returncode=code,log=str(log.relative_to(ROOT)))
                raise SystemExit(code)
            assert sha(ROOT/row['script'])==script_sha,'Running script changed'
            history.append(dict(name=row['name'],receipt=verify_completed(receipt),finished=now(),log=str(log.relative_to(ROOT))))
            status('STAGE_COMPLETE',row['name'])
        status('EXPERIMENT_REPORTS_COMPLETE_PUBLICATION_PENDING','all_scheduled_experiments')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int,default=0)
    parser.add_argument('--show-plan',action='store_true');args=parser.parse_args()
    if args.show_plan:print(json.dumps(stages(),indent=2));sys.exit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:run(args.wait_pid)
    except BaseException as exc:
        path=RAW/'QUEUE_STATUS.json'
        value=read(path) if path.exists() else {}
        value.update(state='STOPPED_ON_ERROR',time=now(),error_type=type(exc).__name__,error=str(exc))
        write(path,value)
        raise
