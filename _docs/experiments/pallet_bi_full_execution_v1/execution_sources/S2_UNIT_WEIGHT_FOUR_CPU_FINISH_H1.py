"""Frozen nine-fit waiter for the fresh D4 S2 banks; metadata until ready."""
from pathlib import Path
import hashlib
import json
import os
import time
import traceback
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'

ROOT=Path('/home/minjae/Documents/pallet-research/pallet-bi-full-execution-20261019-v1')
PRODUCER=ROOT/'S2_FULL_UNIT_WEIGHT_FOUR_CPU'
OUTPUT=ROOT/'S2_UNIT_WEIGHT_FOUR_CPU_H1'
SYNTHETIC=ROOT.parent/'pallet-statewise-end-to-end-recovery-v1'/'synthetic_v1'
GO_SHA='7270aa4deb0a785036715925329d7e69d36df7918cf636a4decc49f216336238'
PLAN=ROOT/'S2_UNIT_WEIGHT_FOUR_CPU_H1_AUTOMATION_PLAN.json'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def main():
    go=ROOT/'FOUR_CPU_SCHEDULING_ROOT_GO.json';assert sha(go)==GO_SHA
    approval=read(go);assert approval['approved']and approval['S2_H1_fits_authorized']==9
    plan=read(PLAN);assert plan['source_sha256']==sha(__file__)
    execution=read(ROOT/'S2_FOUR_CPU_EXECUTION_LOCK.json')
    assert execution['H1_controller_plan_sha256']==sha(PLAN)and execution['root_GO_sha256']==GO_SHA
    assert execution['new_output']==str(PRODUCER)and execution['H1_output']==str(OUTPUT)
    assert plan['campaign_plan_sha256']==sha(ROOT/'CAMPAIGN_PLAN.json')
    assert plan['corrected_packet_seal_sha256']==sha(ROOT/'S2_UNIT_WEIGHT/PACKET_SEAL.json')
    assert plan['environment_sha256']==sha(ROOT/'ENVIRONMENT_INITIAL.json')
    while True:
        sp,dp=PRODUCER/'SUMMARY.json',PRODUCER/'DISPATCH_COMPLETION.json'
        if sp.exists()and dp.exists()and read(sp)['status']=='COMPLETE_NEW':break
        time.sleep(20)
    # Pool context has exited before D4 aggregation/dispatch receipts. No
    # neural/scientific module is imported while waiting for those receipts.
    summary,dispatch,lock=read(sp),read(dp),read(PRODUCER/'PRODUCER_LOCK.json')
    assert summary['technical_failures']==[]and summary['missing_rows']==[]and summary['worker_count']==4
    assert dispatch['active_rows_drained_before_exit']and not dispatch['checkpoint_stop_observed']
    assert dispatch['submitted_jobs']==1241 and dispatch['unsubmitted_jobs']==0 and dispatch['summary_sha256']==sha(sp)
    assert lock['schema']=='FULL_NORMAL_FOUR_CPU_DISPATCH_V1'and lock['workers']==4 and lock['arms']==['C0','CP','CB']
    assert lock['orchestration_sha256']==plan['D4_source_sha256']and lock['row_helper_sha256']==plan['row_helper_sha256']
    assert lock['numerical_kernel_source_binding']==plan['numerical_source_binding']
    assert sha(Path(plan['worktree'])/'scripts/research/pallet_bi_full_execution_v1/fullnormal_dispatch_four.py')==plan['D4_source_sha256']
    assert sha(Path(plan['worktree'])/'scripts/research/pallet_bi_full_execution_v1/fullnormal_parallel.py')==plan['row_helper_sha256']
    for path,digest in plan['numerical_source_binding'].items():assert sha(Path(plan['worktree'])/path)==digest
    assert lock['packet_seal_sha256']==plan['corrected_packet_seal_sha256']and lock['environment_sha256']==plan['environment_sha256']
    assert summary['producer_lock_sha256']==sha(PRODUCER/'PRODUCER_LOCK.json')
    for path,digest in plan['head_source_sha256'].items():assert sha(path)==digest
    for name,digest in plan['synthetic_source_sha256'].items():assert sha(SYNTHETIC/name)==digest
    features={}
    for arm in ('C0','CP','CB'):
        for domain,count in (('real',238),('synthetic',1003)):
            folder=PRODUCER/arm/domain;seal=read(folder/'FEATURE_SEAL.json')
            assert seal['complete']and seal['rows']==count and seal['domain']==domain and seal['arm']==arm
            assert sha(folder/'FEATURES.jsonl.gz')==seal['head_features_sha256']
            assert seal['producer_lock_sha256']==sha(PRODUCER/'PRODUCER_LOCK.json')
            assert seal['source_binding']==plan['numerical_source_binding']and seal['numeric_reference_inputs']is False
            assert len(seal['frame_seal_sha256'])==len(seal['row_execution_sha256'])==count
            for i in range(count):
                key=f'{i:04d}';row=folder/key
                assert sha(row/'FRAME_SEAL.json')==seal['frame_seal_sha256'][key]
                assert sha(row/'ROW_EXECUTION.json')==seal['row_execution_sha256'][key]
                frame,receipt=read(row/'FRAME_SEAL.json'),read(row/'ROW_EXECUTION.json')
                assert frame['complete']and receipt['complete']and receipt['original_frame_seal_sha256']==sha(row/'FRAME_SEAL.json')
                assert frame['cache_payload_sha256']==receipt['cache_payload_sha256']
                for name,digest in frame['artifacts'].items():assert sha(row/name)==digest
            features[arm+'/'+domain]=dict(feature_seal_sha256=sha(folder/'FEATURE_SEAL.json'),features_sha256=sha(folder/'FEATURES.jsonl.gz'))
    from scripts.research.pallet_bi_full_execution_v1 import heads as H
    assert H.source_hashes()==plan['head_source_sha256']
    write_new(OUTPUT/'PREFIT_SEAL.json',dict(epoch=time.time(),source_sha256=sha(__file__),automation_plan_sha256=sha(PLAN),
        root_GO_sha256=GO_SHA,execution_lock_sha256=sha(ROOT/'S2_FOUR_CPU_EXECUTION_LOCK.json'),head_source_sha256=H.source_hashes(),
        producer_summary_sha256=sha(sp),producer_dispatch_sha256=sha(dp),producer_lock_sha256=sha(PRODUCER/'PRODUCER_LOCK.json'),
        feature_bindings=features,all_six_full_feature_seals_and_rows_verified=True,fitting_data='original synthetic train768/validation235 only',
        PRIMARY_numeric_reference_inputs=False))
    for arm in ('C0','CP','CB'):
        folder=OUTPUT/arm;synthetic=PRODUCER/arm/'synthetic';real=PRODUCER/arm/'real'
        print(json.dumps(dict(stage='S2_H1_LABELS',arm=arm,epoch=time.time())),flush=True)
        H.prepare(synthetic/'FEATURES.jsonl.gz',synthetic/'FEATURE_SEAL.json',SYNTHETIC/'SYNTH_GT_LABELS.jsonl.gz',
            SYNTHETIC/'PROVISION_SEAL.json',SYNTHETIC/'DATA_SPLIT_LOCK.json',folder/'labels',ROOT)
        H.train(synthetic/'FEATURES.jsonl.gz',synthetic/'FEATURE_SEAL.json',folder/'labels',folder/'training','S2_'+arm,ROOT)
        H.predict(real/'FEATURES.jsonl.gz',real/'FEATURE_SEAL.json',folder/'training',folder/'prediction',ROOT,'S2_'+arm)
        print(json.dumps(dict(stage='S2_H1_PREDICTIONS_SEALED',arm=arm,epoch=time.time())),flush=True)
    write_new(OUTPUT/'ALL_REAL_PREDICTIONS_SEAL.json',dict(complete=True,arms=['S2_C0','S2_CP','S2_CB'],
        seeds=[20261018,20261019,20261020],rows_per_arm=238,source_sha256=sha(__file__),
        prediction_seal_sha256={arm:sha(OUTPUT/arm/'prediction/PREDICTION_SEAL.json')for arm in ('C0','CP','CB')},
        current_real_reference_reads=0,before_new_scoring=True))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write_new(OUTPUT/'AUTOMATION_TECHNICAL_FAILURE.json',dict(epoch=time.time(),error=repr(error),traceback=traceback.format_exc(),
            source_sha256=sha(__file__),no_failed_or_incomplete_row_zero_filled=True))
        raise
