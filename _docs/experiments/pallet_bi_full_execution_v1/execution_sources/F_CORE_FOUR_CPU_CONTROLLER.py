"""Approved fresh four-worker F core only; no F choice, H1 or scoring."""
from pathlib import Path
import os
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from scripts.research.pallet_bi_full_execution_v1 import fullnormal_dispatch_four as D4

ROOT=Path('/home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1')
GO_SHA='7270aa4deb0a785036715925329d7e69d36df7918cf636a4decc49f216336238'


def main():
    q=D4.Q;go=ROOT/'FOUR_CPU_SCHEDULING_ROOT_GO.json';assert q.sha(go)==GO_SHA
    approval=q.read(go);assert approval['approved']and approval['F_H1_fits_authorized']==0
    pre=q.read(ROOT/'F_FOUR_CPU_CONTROLLER_PLAN.json');assert pre['source_sha256']==q.sha(__file__)
    assert D4.source_bindings()==pre['source_binding']
    assert q.sha(ROOT/'F_CALIBRATION/PACKET_MANIFEST.json')==pre['packet_manifest_sha256']
    assert q.sha(ROOT/'ENVIRONMENT_INITIAL.json')==pre['environment_sha256']
    output=ROOT/'F_CORE_FOUR_CPU'
    q.write(ROOT/'F_FOUR_CPU_CONTROLLER_LOCK.json',dict(schema='F_FOUR_CPU_CORE_CONTROLLER_LOCK_V1',
        source_sha256=q.sha(__file__),plan_sha256=q.sha(ROOT/'F_FOUR_CPU_CONTROLLER_PLAN.json'),root_GO_sha256=GO_SHA,
        actual_argv=[v.decode()for v in Path('/proc/self/cmdline').read_bytes().split(b'\0')if v],
        delegated_function='fullnormal_dispatch_four.run',delegated_call=pre['delegated_call'],
        delegated_argv=pre['delegated_argv'],delegated_argv_is_equivalent_configuration_not_subprocess=True,
        output=str(output),workers=4,source_binding=pre['source_binding'],
        packet_manifest_sha256=pre['packet_manifest_sha256'],environment_sha256=pre['environment_sha256'],
        exact_reuse_manifest_sha256=q.sha(ROOT/'F_CORE_FOUR_CPU_EXACT_REUSE/REUSE_MANIFEST.json'),
        quiesce_receipt_sha256=q.sha(ROOT/'F_FOUR_CPU_QUIESCE_COMPLETE.json'),
        head_fit_calls=0,final_F_complete=False,epoch=time.time()))
    result=D4.run(ROOT.parent/'pallet-bi-conditioned-observation-20261019-v1',ROOT/'F_CALIBRATION',
        ROOT/'F_CALIBRATION/PACKET_MANIFEST.json',ROOT/'ENVIRONMENT_INITIAL.json',output,
        arms=('F_COMMON','F_BI'),workers=4,reuse_root=ROOT/'F_CORE_FOUR_CPU_EXACT_REUSE',root_GO=True,
        stop_file=ROOT/'F_FOUR_CPU_CHECKPOINT_STOP')
    feature_bindings={}
    if result['status']=='COMPLETE_NEW':
        for arm in ('F_COMMON','F_BI'):
            for domain,count in (('real',238),('synthetic',1003)):
                folder=output/arm/domain;seal=q.read(folder/'FEATURE_SEAL.json')
                assert seal['complete']and seal['rows']==count and q.sha(folder/'FEATURES.jsonl.gz')==seal['head_features_sha256']
                assert seal['producer_lock_sha256']==q.sha(output/'PRODUCER_LOCK.json')
                feature_bindings[arm+'/'+domain]=dict(feature_seal_sha256=q.sha(folder/'FEATURE_SEAL.json'),features_sha256=q.sha(folder/'FEATURES.jsonl.gz'))
    q.write(ROOT/'F_FOUR_CPU_CONTROLLER_COMPLETION.json',dict(complete=result['status']=='COMPLETE_NEW',
        component='unconditional F core banks only',final_F_complete=False,unconditional_core_only=True,
        source_sha256=q.sha(__file__),root_GO_sha256=GO_SHA,workers=4,
        controller_lock_sha256=q.sha(ROOT/'F_FOUR_CPU_CONTROLLER_LOCK.json'),
        producer_summary_sha256=q.sha(output/'SUMMARY.json'),producer_lock_sha256=q.sha(output/'PRODUCER_LOCK.json'),
        producer_dispatch_sha256=q.sha(output/'DISPATCH_COMPLETION.json'),feature_bindings=feature_bindings,
        H1_fits=0,real_reference_reads=0,F_choice_calls=0,epoch=time.time()))
    print(dict(status=result['status'],unconditional_core_only=True,H1_fits=0,F_choice_calls=0),flush=True)


if __name__=='__main__':main()
