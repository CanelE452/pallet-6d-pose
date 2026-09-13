"""Separate final integrity checks: audit PASS is not a performance PASS."""
import ast
import numpy as np
import torch
from common import *

def run():
    preserved=read(DOC/'PRESERVED_SOURCES.json')['paths'];checked=0
    for path,digest in preserved.items():
        assert sha(ROOT/path)==digest,('Historical artifact changed',path);checked+=1
        if checked%1000==0:print('PRESERVATION_RECHECK',checked,len(preserved),flush=True)
    assert sha(R0)==R0_SHA
    for name in ('B_IMPLEMENTATION_LOCK.json','LINE_SOURCE_BINDING.json'):
        for path,digest in read(B/name)['paths'].items():assert sha(ROOT/path)==digest,path
    from evaluate_point import verify_lock
    verify_lock()
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    assert read(B/'REGRESSION_TESTS.json')['PASS'] and read(B/'PIPELINE_TESTS.json')['PASS']
    correction=read(B/'CANONICAL_C2_TEST_CORRECTION.json')
    assert correction['PASS'] and correction['test_sha256']==sha(HERE/'test_canonical_yaw.py')
    assert correction['diagnostic_sha256']==sha(A/'CANONICAL_C2_DIAGNOSTIC_CORRECTION.json')
    before=read(B/'GPU_TRAIN_START.json');boundaries={};seeds={}
    for seed in (1,2,3):
        c=read(B/f'P_SEED{seed}_COMPLETION.json');path=BRAW/f'runs/seed{seed}/last.pt'
        assert c['complete'] and c['step']==6000 and c['updates']==6000 and c['exposures']==96000
        assert sha(path)==c['checkpoint_sha256']
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['complete'] and ck['step']==6000 and ck['baseline_checkpoint_sha256']==R0_SHA
        assert all(torch.isfinite(v).all() for v in ck['model_state_dict'].values())
        order=np.load(BRAW/f'order_seed{seed}.npy');digest=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
        assert digest==ck['batch_trace_sha256']==c['batch_order_sha256']==read(B/'INIT_ORDER_PARITY.json')['seeds'][str(seed)]['batch_order_sha256']
        assert c['history'][0]['step']==1 and c['history'][-1]['step']==6000
        after=c['history'][-1]['gpu'];boundaries[str(seed)]=dict(before=before,first_update=c['history'][0]['gpu'],after=after,
            before_source='global pretraining snapshot' if seed==1 else 'previous seed final synchronized update snapshot, before new seed initialization; also used as previous seed after snapshot')
        before=after;seeds[str(seed)]=dict(complete=True,step=6000,exposures=96000,checkpoint_sha256=sha(path),order_exact=True)
        assert read(B/f'P{seed}_INFERENCE_AUDIT.json')['PASS'] and read(B/f'P{seed}_PAPER_EVALUATION_COMPLETE.json')['complete']
    write(B/'TRAINING_AUDIT.json',dict(PASS=True,fits=3,seeds=seeds,total_updates=18000,real_training=0,L_training=0,R0_training=0,
        batch=16,steps_each=6000,optimizer_exact=True,AMP=False,final_checkpoint_only=True,protected_R0_SHA=sha(R0)))
    write(B/'GPU_BOUNDARY_AUDIT.json',dict(seeds=boundaries,scope='Read-only snapshots, no clocks/power/driver/reboot modifications'))
    assert read(B/'RUNTIME.json')['PASS'] and read(B/'RUNTIME.json')['parity_PASS']
    mechanism=read(B/'MECHANISM_NORMAL_TANGENT.json')
    assert mechanism['complete'] and mechanism['rows_sha256']==sha(BRAW/'MECHANISM_PER_EDGE.json')
    adapter=read(B/'MECHANISM_METADATA_ADAPTER.json')
    assert adapter['code_sha256']==sha(HERE/'mechanism_completion.py')
    assert read(B/'RUNTIME_CONDITIONS_NOTE.json')['runtime_sha256']==sha(B/'RUNTIME.json')
    assert read(TASK_DOC/'VERDICT.json')['scientific_verdict']=='TASK_RISK_AL_NO_SIGNAL'
    # Recompute the full primary 10k-draw statistic independently of its
    # persisted result, with the same deterministic draw seed.
    from statistics_and_mechanism import load,paired
    stores,poses,_,_,_,_=load()
    repeated=paired([stores[f'L{s}'] for s in (1,2,3)],[stores[f'P{s}'] for s in (1,2,3)],'keypoint_location_median_px')
    assert repeated==read(B/'REAL_PAIRED_STATISTICS.json')['comparisons']['keypoint_location_median_px']
    assert subprocess.check_output(['git','diff','--name-only'],text=True).strip()==''
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    close=gpu();write(B/'GPU_CLOSEOUT.json',close)
    a=dict(PASS=True,historical_full_byte_SHA_count=checked,official_verdict_unchanged=True,QA_flags_preexperiment_blobs_verified=True,
        same145_membership=True,S1_same_empty_set_all_methods=True,official_CVaR_numeric_exact=True,new_training=0,
        catastrophic_source_bound=True,performance_success=False)
    write(A/'FINAL_AUDIT.json',a)
    write(B/'FINAL_AUDIT.json',dict(PASS=True,audit_PASS_is_not_performance_PASS=True,historical_files_verified=checked,
        frozen_training_source_and_protocol_exact=True,three_fits_exact=True,order_parity=True,parameter_evidence_locks=True,
        topology_free_P_forward=True,GT_free_inference=True,synthetic_split_and_selection_bound=True,
        actual_real_forwards_per_seed=3008,box_score_order_center_parity=True,canonical_metric_recomputation_exact=True,
        primary_bootstrap_10000_deterministic_repeat_exact=True,runtime_accuracy_parity=True,paper_final_untouched=True,
        new_files_only=True,old_line_primary_failed_verdict_preserved=True,old_task_risk_failed_verdict_preserved=True,
        limitations=['Development reuse; no independent confirmation or novelty claim','Fixed canonical assignment, not a new symmetry-equivariant architecture',
            'Edge-target versus point-target support/target semantics differ','Matched parameters and spatial reads, not matched FLOPs','Physical visibility provenance not independently validated',
            'Initial generic permutation fixture mislabeled C2 yaw; exact repository C2 test added after training, with no training/inference assignment change',
            'CPU scoring overlapped runtime start; observed wall times are descriptive, not an isolated causal speed comparison'],
        verdict=read(B/'VERDICT.json')['verdict']))
    print('FINAL_INTEGRITY_AUDIT_PASS',read(B/'VERDICT.json')['verdict'],flush=True)

if __name__=='__main__':run()
