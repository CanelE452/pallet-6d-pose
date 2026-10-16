"""Hash-checked stage driver. Completed numeric stages are reused, never outcome-tuned."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from . import common as C

PACKAGE='scripts.research.pallet_observation_refiner_20261009_v1'
STAGES={
 'audit':('audit',[],'PROTOCOL.json'),
 'solver_tests':('test_solver',['--output',str(C.DOC/'SOLVER_CHECKS.json')],'SOLVER_CHECKS.json'),
 'contract_audit':('contract_audit',[],'CONTRACT_AUDIT.json'),
 'cpu_pilot':('pilot',[],'CPU_PILOT.json'),
 'pose_diagnostics':('evaluate',[],'POSE_EXECUTION.json'),
 'mask_stress':('geometry_stress',['--scenes','128','--output',str(C.DOC)],'GEOMETRY_STRESS_SUMMARY.json'),
 'source_audit':('source_audit',[],'SYNTH_SUPERVISION_AUDIT.json'),
 'prepare_supervision':('training',['prepare'],'SUPERVISION_PREPARATION.json'),
 'train':('training',['train'],'TRAINING_COMPLETION.json'),
 'infer_observations':('learned_infer',['--output','LEARNED_OBSERVATIONS.jsonl.gz'],'OBSERVATION_SEAL.json'),
 'solve_evaluate':('learned_evaluate',[],'LEARNED_POSE_EXECUTION.json'),
 'statistics':('statistics',[],'METRICS.json'),
 'benchmark':('benchmark',['measure','--image-role-adapter',PACKAGE+'.benchmark:benchmark_adapter',
             '--learned-reference',str(C.DOC/'LEARNED_PREDICTIONS.jsonl.gz')],'RUNTIME.json'),
 'verify':('verify',[],'VERIFICATION.json'),
 'report':('report',[],'RESULT_KO.md')}
DEPENDENCIES={
 'cpu_pilot':['INPUTS.json','SOLVER_CHECKS.json'],
 'pose_diagnostics':['INPUTS.json','SOLVER_CHECKS.json'],
 'prepare_supervision':['SOURCE_FAMILY_SPLIT.json','SOURCE_SUPERVISION_STATUS.json'],
 'train':['LEARNING_PROTOCOL.json','SUPERVISION_PREPARATION.json'],
 'infer_observations':['INPUTS.json','TRAINING_COMPLETION.json'],
 'solve_evaluate':['OBSERVATION_SEAL.json','LEARNED_OBSERVATIONS.jsonl.gz'],
 'statistics':['PREDICTIONS.jsonl.gz','FIXED_CONTROLS.jsonl.gz','LEARNED_PREDICTIONS.jsonl.gz'],
 'benchmark':['POSE_DIAGNOSTICS.jsonl.gz','LEARNED_PREDICTIONS.jsonl.gz','CONTRACT_AUDIT.json'],
 'verify':['METRICS.json','METRICS.csv','PAIRED_COMPARISONS.json','RUNTIME.json'],
 'report':['METRICS.json','VERIFICATION.json','EXECUTION_LEDGER.json','RUNTIME.json']}

def verify_source_bindings():
    if not (C.DOC/'SOURCE_BINDINGS.json').exists():return
    packet=C.read(C.DOC/'SOURCE_BINDINGS.json')
    for binding in packet['inputs']:
        if binding['origin']=='source':
            assert C.sha(C.ROOT/binding['path'])==binding['sha256'],binding['path']
    if (C.DOC/'PROTOCOL.json').exists():
        b=C.read(C.DOC/'PROTOCOL.json')['inputs'];assert C.sha(C.DOC/'INPUTS.json')==b['sha256']

def run(stage,refresh=False):
    verify_source_bindings();module,args,artifact=STAGES[stage];output=C.DOC/artifact
    if stage=='source_audit':
        archive=os.environ.get('PALLET_TEX_ARCHIVE')
        if not archive:
            raise RuntimeError('Set PALLET_TEX_ARCHIVE to the existing read-only TEX archive before source_audit')
        args=['--tex-archive',archive]
    receipt=C.SCRATCH/'DRIVER_RECEIPTS.json'
    history=C.read(receipt) if receipt.exists() else dict(records=[])
    identities=[C.binding(Path(__file__).with_name(module+'.py'))]
    identities.extend(C.binding(C.DOC/name) for name in DEPENDENCIES.get(stage,[]) if (C.DOC/name).exists())
    input_identity=C.digest(identities)
    previous=next((r for r in reversed(history['records']) if r['stage']==stage),None)
    if previous and previous.get('input_identity') and previous['input_identity']!=input_identity and not refresh:
        raise RuntimeError('Stage inputs/code changed; preserve outputs and diagnose before resuming '+stage)
    if output.exists() and not refresh:
        if output.suffix=='.json':
            result=C.read(output)
            assert result.get('complete',result.get('passed',result.get('status')=='PASS')) is not False, 'Incomplete stage requires diagnosis, not automatic rerun'
        record=dict(stage=stage,status='REUSED',artifact=C.binding(output))
    else:
        if refresh:
            assert stage in ('statistics','verify','report'),'Refresh is limited to raw-row aggregation/verification/reporting'
            if output.exists():
                prior=C.SCRATCH/('PRE_REFRESH_'+stage+'_'+str(time.time_ns())+output.suffix)
                prior.write_bytes(output.read_bytes())
        begin=time.monotonic()
        completed=subprocess.run([sys.executable,'-m',PACKAGE+'.'+module,*args],cwd=C.WORKTREE)
        record=dict(stage=stage,status='COMPLETE' if completed.returncode==0 else 'FAILED',
            returncode=completed.returncode,seconds=time.monotonic()-begin,
            artifact=C.binding(output) if output.exists() else None)
        if completed.returncode:
            history['records'].append(record);C.write(receipt,history);raise RuntimeError(stage+' failed; outputs preserved')
    record.update(input_identity=input_identity,source_inputs=identities)
    history['records'].append(record);C.write(receipt,history);print(C.finite(record),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=list(STAGES));parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args();run(args.stage,args.refresh)
