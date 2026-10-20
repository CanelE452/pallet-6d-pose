"""Resumable recorded execution of the approved frozen-D1 comparison stages."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import uuid

PACKAGE = 'scripts.research.pallet_d1_three_backbones_20261020_v1.'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def status(run, **value):
    path = run/('PIPELINE_STATUS.'+uuid.uuid4().hex+'.tmp')
    write(path, dict(updated_utc=utc(), **value))
    path.replace(run/'PIPELINE_STATUS.json')


def guard(repo, run, go):
    if not go.get('approved') or sha(run/'EXECUTION_PLAN.json') != go['execution_plan_sha256']:
        raise ValueError('Explicit approved immutable execution plan is required')
    for name, digest in go['new_module_source_sha256'].items():
        if sha(repo/'scripts/research/pallet_d1_three_backbones_20261020_v1'/name) != digest:
            raise ValueError('Approved scientific source changed: '+name)


def checked_prediction(run, backbone, original):
    folder = run/backbone/'predictions'
    if not folder.exists():
        return False
    seal = read(folder/'PREDICTION_SEAL.json')
    features = run/backbone/'candidates'
    reference = read(run/'yolo/predictions/PREDICTION_SEAL.json')
    if (not seal['complete'] or seal['rows'] != 238
            or sha(folder/'PREDICTIONS.jsonl.gz') != seal['predictions_sha256']
            or sha(features/'HEAD_FEATURES.jsonl.gz') != seal['features_sha256']
            or sha(features/'FEATURE_SEAL.json') != seal['feature_seal_sha256']
            or sha(original/'action_training_v1/TRAINING_MANIFEST.json') != seal['training_manifest_sha256']
            or seal['checkpoints'] != reference['checkpoints']
            or seal['source_sha256'] != sha(Path(__file__).with_name('prediction.py'))):
        raise ValueError('Completed frozen-head predictions failed ancestry verification')
    return True


def run(repo, private, original_d1, original_score, root_go):
    repo, private, original_d1, original_score, root_go = map(Path, (repo, private, original_d1, original_score, root_go))
    go = read(root_go)
    guard(repo, private, go)
    for backbone in ('dope','resnet18'):
        pilot = private/'technical/cal_pilot_v1'/backbone
        receipt = read(pilot/'PILOT_RECEIPT.json')
        if not receipt['complete'] or receipt['plan_sha256'] != go['execution_plan_sha256']:
            raise ValueError('Both approved common-CAL technical pilots must finish first')
        for name, digest in receipt['artifacts'].items():
            if sha(pilot/name) != digest:
                raise ValueError('Technical pilot artifact changed: '+name)
    attempt = private/'pipeline_attempts'/('attempt_'+uuid.uuid4().hex)
    attempt.mkdir(parents=True, exist_ok=False)
    with (private/'PIPELINE_OWNER.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX|fcntl.LOCK_NB)
        start, cpu = time.monotonic(), time.process_time()
        write(attempt/'START.json', dict(started_utc=utc(), argv=sys.argv, launcher_sha256=sha(__file__),
            ROOT_GO_sha256=sha(root_go), same_fixed_H1_transfer_not_own_backbone_fit=True))
        stages = [('full_D1', [sys.executable,'-B','-m',PACKAGE+'execution','run','--plan',str(private/'EXECUTION_PLAN.json'),'--root-go'])]
        for backbone in ('dope','resnet18'):
            folder = private/backbone/'candidates'
            stages.append(('predict_'+backbone, [sys.executable,'-B','-m',PACKAGE+'prediction',
                '--features',str(folder/'HEAD_FEATURES.jsonl.gz'),'--feature-seal',str(folder/'FEATURE_SEAL.json'),
                '--training-dir',str(original_d1/'action_training_v1'),'--output',str(private/backbone/'predictions')]))
        stages.append(('score', [sys.executable,'-B','-m',PACKAGE+'evaluation','--private',str(private),
            '--original-d1',str(original_d1),'--original-score',str(original_score),'--output',str(private/'scoring_v1')]))
        results = []
        try:
            for name, argv in stages:
                guard(repo, private, go)
                if name.startswith('predict_') and checked_prediction(private, name[len('predict_'):], original_d1):
                    results.append(dict(stage=name, exact_completed_prediction_reuse=True, new_calls=0))
                    continue
                if name == 'score' and (private/'scoring_v1/SCORE_SEAL.json').exists():
                    seal = read(private/'scoring_v1/SCORE_SEAL.json')
                    if not seal['complete'] or seal['source_sha256'] != sha(Path(__file__).with_name('evaluation.py')):
                        raise ValueError('Scoring cache source mismatch')
                    for filename, digest in seal['artifacts'].items():
                        if sha(private/'scoring_v1'/filename) != digest:
                            raise ValueError('Scoring cache hash mismatch')
                    results.append(dict(stage=name, exact_completed_score_reuse=True, new_calls=0))
                    continue
                tick = time.monotonic()
                before = resource.getrusage(resource.RUSAGE_CHILDREN)
                status(private, state='RUNNING', stage=name, actual_argv=argv, attempt=attempt.name)
                with (attempt/(name+'.log')).open('x') as log:
                    child = subprocess.Popen(argv, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
                    write(attempt/(name+'_START.json'), dict(stage=name, started_utc=utc(), PID=child.pid, actual_argv=argv))
                    code = child.wait()
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                result = dict(stage=name, returncode=code, finished_utc=utc(), wall_seconds=time.monotonic()-tick,
                    exited_child_CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                    CPU_scope='This subprocess and its waited descendants; nested row timers are not added again')
                write(attempt/(name+'_FINISH.json'), result)
                results.append(result)
                if code:
                    raise RuntimeError('Stage failed; all attempts preserved: '+name)
            guard(repo, private, go)
            receipt = dict(schema='FROZEN_D1_THREE_BACKBONE_CORE_COMPLETION_V1', complete=True,
                ended_utc=utc(), wall_seconds=time.monotonic()-start, parent_CPU_seconds=time.process_time()-cpu,
                stages=results, ROOT_GO_sha256=sha(root_go), launcher_sha256=sha(__file__),
                score_seal_sha256=sha(private/'scoring_v1/SCORE_SEAL.json'),
                new_fit_or_seed_or_data_or_reference_or_bootstrap=0, report_and_independent_audit_pending=True)
            write(attempt/'FINISH.json', receipt)
            write(private/'CORE_PIPELINE_COMPLETION.json', receipt)
            status(private, state='CORE_COMPLETE', stage='report_and_independent_audit_pending', attempt=attempt.name)
        except BaseException as exc:
            write(attempt/'FAILURE.json', dict(ended_utc=utc(), error=repr(exc), stages=results,
                all_prior_attempts_preserved=True, unsealed_interrupted_cost='UNKNOWN_NOT_ZERO'))
            status(private, state='TECHNICAL_FAILURE', stage=results[-1]['stage'] if results else 'startup', error=repr(exc), attempt=attempt.name)
            raise
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo','private','original-d1','original-score','root-go'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo,args.private,args.original_d1,args.original_score,args.root_go)), flush=True)
