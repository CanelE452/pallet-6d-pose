"""Resumable scheduling wrapper; all production/fit/scoring stays separated.

Completed scientific artifacts are authenticated and reused, never rerun just
to repair a controller receipt. Each invocation writes a fresh immutable log.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
from . import s1_union as U, f_batch as B, f_front_audit as A


def run(d1,campaign,reference,*,GO=False):
    if GO is not True:raise PermissionError('Authorized unconditional F core precomputation GO required')
    d1,campaign,reference=map(Path,(d1,campaign,reference))
    go=U.read(campaign/'F_PRECOMPUTE_FRONTEND_GO.json')
    if B.source_binding()!=go['source_binding']:raise ValueError('Frozen F observation producer changed')
    calibration_source=Path(__file__).with_name('f_calibration.py')
    if U.sha(calibration_source)!=go['calibration_adapter_sha256']:raise ValueError('Frozen own-CAL recipe changed')
    obs=campaign/'F_OBSERVATIONS';cal=campaign/'F_CALIBRATION'
    invocation=campaign/'F_FRONTEND_INVOCATIONS'/str(time.time_ns());invocation.mkdir(parents=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    root=Path(__file__).resolve().parents[3];start=time.monotonic();commands=[]
    stages=[(obs/'OBSERVATION_MANIFEST.json',[sys.executable,'-B','-m',B.__name__,'--d1',str(d1),'--campaign',str(campaign),
             '--output',str(obs),'--GO','--precompute-core']),
            (cal/'PACKET_MANIFEST.json',[sys.executable,'-B','-m','scripts.research.pallet_bi_full_execution_v1.f_calibration',
             '--d1',str(d1),'--campaign',str(campaign),'--observations',str(obs),'--reference',str(reference),'--output',str(cal)])]
    for index,(seal,argv)in enumerate(stages):
        if seal.exists():
            existing=U.read(seal)
            if not existing.get('complete')or existing.get('rows')!=1241:raise ValueError('Existing stage manifest is incomplete')
            record=dict(stage=index,exact_completed_component_reused=True,manifest=str(seal),manifest_sha256=U.sha(seal),
                        actual_new_child_commands=0,downstream_fullpopulation_audit_required=True)
        else:
            tick=time.monotonic();result=subprocess.run(argv,cwd=root,env=env)
            record=dict(stage=index,argv=argv,returncode=result.returncode,wall_seconds=time.monotonic()-tick,
                        exact_completed_component_reused=False,actual_new_child_commands=1)
            U.write(invocation/f'COMMAND_{index:02d}.json',record)
            if result.returncode:raise RuntimeError('Preserved F substep failed; inspect its immutable command receipt')
        commands.append(record)
        if not (invocation/f'COMMAND_{index:02d}.json').exists():U.write(invocation/f'COMMAND_{index:02d}.json',record)
    if U.sha(reference)!=U.read(cal/'CAL_REFERENCE_ACCESS.json')['reference_sha256']:
        raise ValueError('Requested canonical reference bytes differ from own-CAL fit')
    result=A.run(d1,obs,cal,campaign/'F_FRONT_AUDIT')
    receipt=dict(complete=True,component='F fixed observations plus own CAL-only fit',final_F_complete=False,
        invocation_directory=str(invocation),source_sha256=U.sha(__file__),commands=commands,wall_seconds=time.monotonic()-start,
        fullpopulation_audit_sha256=U.sha(campaign/'F_FRONT_AUDIT/SUMMARY.json'),numeric_reference_decodes_in_wrapper=0,
        newly_executed_pose_calls=0,new_H1_fits=0,unchanged_completed_stages_not_reexecuted=True)
    U.write(invocation/'COMPLETE.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('d1','campaign','reference'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--GO',action='store_true');a=p.parse_args();print(run(a.d1,a.campaign,a.reference,GO=a.GO))
