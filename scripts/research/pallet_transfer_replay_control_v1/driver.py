"""Run the immutable 12-fit order; never select by intermediate performance."""
import json
import os
from pathlib import Path
import subprocess
import sys

from contracts import ARM_WEIGHTS
from runtime import ROOT,RAW,DOC,atomic_json,sha


def gpu_snapshot():
    command=['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits']
    gpu=subprocess.check_output(command,text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    return dict(gpu=gpu,compute_processes=processes)


def main():
    assert json.loads((DOC/'ACTUAL_MODEL_GATE.json').read_text())['status']=='PASS'
    start=gpu_snapshot(); atomic_json(RAW/'GPU_START.json',start)
    order=[(arm,seed) for seed in (1,2,3) for arm in ARM_WEIGHTS]
    for arm,seed in order:
        out=RAW/'runs'/f'{arm}_seed{seed}'
        if (out/'TRAINING_AUDIT.json').exists(): continue
        out.mkdir(parents=True,exist_ok=True); log=out/'TRAIN.log'
        mode='a' if log.exists() else 'x'
        with log.open(mode) as stream:
            stream.write(f'INVOCATION_START arm={arm} seed={seed}\n');stream.flush()
            process=subprocess.run([sys.executable,str(Path(__file__).with_name('runtime.py')),'--arm',arm,'--seed',str(seed)],stdout=stream,stderr=subprocess.STDOUT)
        if process.returncode:
            atomic_json(DOC/'TRAINING_STOP.json',dict(status='TECHNICAL_STOP',arm=arm,seed=seed,
                exit_code=process.returncode,completed=[f'{a}_seed{s}' for a,s in order if (RAW/'runs'/f'{a}_seed{s}'/'TRAINING_AUDIT.json').exists()],
                resource=gpu_snapshot()))
            raise SystemExit(process.returncode)
        print(f'{arm}_seed{seed} training exit0',flush=True)
    audits={f'{a}_seed{s}':json.loads((RAW/'runs'/f'{a}_seed{s}'/'TRAINING_AUDIT.json').read_text()) for a,s in order}
    assert len(audits)==12 and all(v['optimizer_updates']==300 for v in audits.values())
    for seed in (1,2,3):
        traces={arm:json.loads((RAW/'runs'/f'{arm}_seed{seed}'/'EXPOSURE.json').read_text()) for arm in ARM_WEIGHTS}
        for step in range(300):
            hashes={arm:traces[arm][step]['streams']['target_base']['tensor_sha256'] for arm in ARM_WEIGHTS}
            assert len(set(hashes.values()))==1,(seed,step,hashes)
    atomic_json(DOC/'TRAINING_COMPLETE.json',dict(status='PASS',fits=12,actual_main_optimizer_updates=3600,
        actual_smoke_optimizer_updates=json.loads((DOC/'ACTUAL_MODEL_GATE.json').read_text())['smoke_optimizer_updates'],
        all_arms_target_base_tensor_parity_all300=True,audits=audits,gpu_start=start,gpu_end=gpu_snapshot()))
    print('ALL12_FITS_COMPLETE; 3600 MAIN UPDATES; EVALUATION MAY BEGIN',flush=True)


if __name__=='__main__':main()
