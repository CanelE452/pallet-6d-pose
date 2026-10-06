"""Minimal fixed protocol, reused inputs and sequential experiment stages."""
import argparse
import time
from pathlib import Path
import subprocess
import traceback
import numpy as np
from .common import *


def prepare():
    started=time.monotonic(); receipts=fit_receipts()
    if receipts.get('inputs',{}).get('status')=='PASS':
        verify_derived(load_setup());return receipts
    setup=load_setup(); prior=read(QUICK_DOC/'VERIFICATION.json');assert prior['status']=='PASS'
    assert setup['eligible_TRAIN']==55915 and setup['excluded_all_F_invalid']==0
    base='2848501039859d618b6fae2c111e1d9a0ce5a1e3'
    scientific=['scripts/research/pallet_quick_pose_loss_20261006_v1/'+f for f in
        ('common.py','losses.py','training.py','evaluation.py','reporting.py')]
    scientific+=['scripts/research/pallet_pose_target_6d_20261006_v1/baseline.py']
    scientific+=['_docs/experiments/pallet_quick_pose_loss_20261006_v1/'+f for f in
        ('LOSS_SETUP.json','PROTOCOL.json','TRAIN_RECEIPTS.json','SUMMARY.json','VERIFICATION.json')]
    scientific+=['_docs/experiments/pallet_pose_target_6d_20261006_v1/POSE_COST_CACHE_MANIFEST.json']
    hashes={}
    for relative in scientific:
        expected=subprocess.check_output(['git','-C',str(ROOT),'show',base+':'+relative])
        assert (ROOT/relative).read_bytes()==expected, 'Scientific dependency differs: '+relative
        hashes[relative]=sha(ROOT/relative)
    # This prior receipt links the bank/cost/large features to completed content
    # checks. Unchanged size/mtime is explicitly a reused relationship.
    previous_states=read(Path('/tmp/pallet-quick-pose-loss-20261006-cache/FROZEN_INPUT_STATES.json'))['files']
    original={str(Path(e['path']).resolve()):e for e in read(HARD_DOC/'INPUT_BINDINGS.json')['external_inputs']}
    for entry in setup['validation']['cost_artifacts']+setup['validation']['auxiliary_artifacts']:
        original[str(Path(entry['path']).resolve())]=entry
    bank_manifest=read(OLD_DOC/'A_manifest.json')
    for e in bank_manifest['cache_files']:
        original[str((BANKS/e['name']).resolve())]=e
    states=[]; rehashed_changed=[]
    for e in previous_states:
        path=Path(e['path']);st=path.stat()
        if st.st_size!=e['bytes'] or st.st_mtime_ns!=e['mtime_ns']:
            known=original.get(str(path.resolve()));assert known and 'sha256' in known, 'BLOCKED_INTEGRITY: changed input lacks prior content hash: '+str(path)
            assert sha(path)==known['sha256'], 'BLOCKED_INTEGRITY: changed immutable input content: '+str(path)
            rehashed_changed.append(str(path))
        states.append(dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
    artifacts=[]
    for e in setup['artifacts']:
        path=Path(e['path']);assert path.is_file() and path.stat().st_size==e['bytes'] and sha(path)==e['sha256']
        artifacts.append(e)
        st=path.stat();states.append(dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
    data=Data(ROOT);ids=read(OLD_DOC/'results/A_ID_MANIFEST.json')['IDs']
    train_ids=[data.source['records'][data.indices[r]]['id'] for r in data.train_rows]
    assert train_ids==ids['train'] and len(train_ids)==55915
    assert not set(train_ids).intersection(ids['synthetic_evaluation']) and not set(train_ids).intersection(ids['real_evaluation'])
    order_path=ROOT/'data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/order_seed1.npy'
    order=np.load(order_path,mmap_mode='r');old=read(OLD_DOC/'A_fits/GEO_seed1.json')
    assert order.shape==(6000,16) and np.isin(order,data.train_rows).all() and sha(order_path)==old['order_sha256']
    eligible=np.load(setup['paths']['eligible'],mmap_mode='r');assert eligible[data.train_rows].all()
    receipts['inputs']=dict(status='PASS',seconds=time.monotonic()-started,scientific_code_and_evidence_sha256=hashes,
        loss_setup_sha256=sha(LOSS_SETUP),target_artifacts=artifacts,states=states,changed_files_rehashed=rehashed_changed,
        input_verification='fresh small code/manifest/target hashes once; prior completed cost/bank/feature content verification linked to current unchanged size/mtime; not a new full content audit',
        order_sha256=old['order_sha256'],TRAIN_rows=55915,eligible_TRAIN=55915,TRAIN_eval_disjoint=True,
        scale_m_reused=setup['scale_m'],tau_reused=setup['tau'],target_regeneration=0,cost_regeneration=0,backbone_forwards=0,bank_generation=0)
    write(DOC/'RUN_RECEIPTS.json',receipts)
    instruction=Path('/home/minjae/Downloads/pallet_cli_quick_joint_scorer_20261006.txt')
    protocol=dict(schema='quick_capacity_matched_joint_scorer_v1',status='LOCKED_BEFORE_MODEL_CREATION',
        instruction=dict(path=str(instruction),sha256=sha(instruction),lines=len(instruction.read_text().splitlines()),read_scope='entire'),
        source_root=str(ROOT),candidate_bank_cache=str(BANKS),pose_cost_cache=str(COST),output_cache=str(OUTPUT),
        baseline_root=str(BASELINE_ROOT),publication_branch='main',base_results_commit=base,
        methods=['LOCAL_CAP','JOINT8'],execution_order=['LOCAL_CAP','JOINT8'],seed=1,config=read(OLD_DOC/'A_protocol.json')['config'],
        shared_descriptor=dict(pooled='existing motion pooled; NoOp existing null_pool',raw_context='existing location/box/role',
            displacement='existing normalized; NoOp zero',coverage='existing inside fraction; NoOp valid-motion average',embedding='existing metadata_encoder',NoOp_flag=True,
            channel_count=81,channel_calculation='48 pooled +13 raw_context +2 displacement +1 coverage +16 metadata +1 NoOp',shape='B,M,8,D'),
        phi=dict(input=81,output=15,bias=True,activation='ReLU',seed=101),token='concat(mask*phi(desc),maskbit); B,M,8,16; support0 all-zero via torch.where',
        LOCAL_CAP=dict(hidden=260,input=16,activation='ReLU',output=1,output_bias=False,shared_corners=True,aggregation='supported corner mean',head_params=4680),
        JOINT8=dict(hidden=36,input=128,activation='ReLU',output=1,output_bias=False,ordered_roles=list(range(8)),head_params=4680),
        initialization=dict(inherited_seed=1,phi_seed=101,head_first_layer_seed=102,head_last_weight=0,warmstart=False),
        parameter_count=dict(inherited=20259,phi=1230,head=4680,total=26169,dummy=0),delta_coefficient=1,
        loss='unchanged existing losses.pose_loss(SOFT6D), exact saved FP32 targets; no new loss/temperature',
        loss_setup_sha256=sha(LOSS_SETUP),target_paths=setup['paths'],
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,betas=[.9,.999],warmup=100,cosine_steps=5900,final_lr=.0001,clip_norm=5),
        numeric=dict(dtype='FP32',TF32_matmul=False,TF32_cudnn=True,cudnn_benchmark=False),
        budgets=dict(fits=2,updates=12000,exposures=192000,batch=16,updates_per_fit=6000,initial_parity_batches=3,final_eval_examples=4608,final_F_max=4608,probe_examples=512,probe_F=0),
        inference='unchanged hard joint firstargmax; native float64 candidate and RAWcenter; unchanged prediction-only WD/SQPnP/refinementF',
        decisions=dict(N3_GAIN_SIGNAL='REAL T/Rmed both strictly lower and REAL ADDmed/P90,T/RP90,failure,all corner preservation nonworse; no extra SYNTH gate',
            CONTROL_ONLY_GAIN='JOINT8 same directional+preservation criteria versus LOCAL_CAP while N3 criteria fail',
            TRADEOFF='any pose mean/median decrease with another pose/tail/corner/failure increase; retain individual flags',
            NOOP_COLLAPSE='all2304 evaluated native indices0 means RAW retention, no correction gain',
            CI_zero='directional unconfirmed; single training seed, reusedDEV; no multiple comparison correction',
            labels=['N3_GAIN_SIGNAL','TRADEOFF','CONTROL_ONLY_GAIN','NO_N3_GAIN','NOOP_COLLAPSE','BLOCKED_INTEGRITY','RUN_FAILED']),
        tolerances=read(HARD_DOC/'PROTOCOL.json')['screening']['tolerances'],bootstrap=dict(resamples=10000,seed=20260917,REAL_primary='session',SYNTH_primary='frame',SYNTH_secondary='scenario',shared=True),
        forbidden=['new cost/targets/banks/features','model/width/loss/temperature/epoch/seed search','extra fits','GT/oracle/filename/index/session head input','paper/LaTeX/PDF/bibliography changes'],
        external_dependency_scope='private feature/checkpoint/cache and unchanged a22 worktree required; public checkout alone incomplete')
    write(DOC/'PROTOCOL.json',protocol)
    return receipts


def finalize_evaluation_receipts():
    """Bind actual completed stages before the saved-row report freezes RUN."""
    receipts=fit_receipts()
    evaluations={}; probes={}
    for method in ('LOCAL_CAP','JOINT8'):
        probes[method]=read(DOC/f'TRAIN_PROBE_{method}_seed1.json')
        for split in ('SYNTH_HELDOUT','REAL_DEV'):
            evaluations[f'{split}_{method}']=read(DOC/'results'/f'{split}_{method}_seed1_EXECUTION.json')
    assert all(e['complete'] and e['status']=='DONE' for e in (*evaluations.values(),*probes.values()))
    execution=dict(formal_fits=2,formal_updates=sum(f['updates'] for f in receipts['methods'].values()),
        formal_exposures=sum(f['exposures'] for f in receipts['methods'].values()),
        initial_parity_model_batches=receipts['tests']['actual_model_batches'],
        initial_parity_model_examples=receipts['tests']['actual_model_examples'],
        toy_backward_calls=receipts['tests']['toy_backward_calls'],toy_optimizer_updates=0,
        evaluation_examples=sum(e['execution']['new_refiner_examples'] for e in evaluations.values()),
        evaluation_refiner_batches=sum(e['execution']['new_refiner_batches'] for e in evaluations.values()),
        final_F_attempts=sum(e['execution']['new_final_F_attempts'] for e in evaluations.values()),
        final_F_completed=sum(e['execution']['new_final_F_completed'] for e in evaluations.values()),
        TRAIN_probe_examples=sum(p['execution']['new_refiner_examples'] for p in probes.values()),
        TRAIN_probe_batches=sum(p['execution']['new_refiner_batches'] for p in probes.values()),
        TRAIN_probe_F=sum(p['execution']['new_F_calls'] for p in probes.values()),
        PnP_counts={k:sum(e['execution']['PnP_counts'][k] for e in evaluations.values())
                    for k in ('solvePnP','solvePnPGeneric','solvePnPRefineLM')},
        preparation_seconds=receipts['inputs']['seconds'],minimum_tests_seconds=receipts['tests']['seconds'],
        training_seconds={m:f['seconds'] for m,f in receipts['methods'].items()},
        evaluation_seconds={name:e['execution']['seconds_wall'] for name,e in evaluations.items()},
        probe_seconds={m:p['execution']['seconds_wall'] for m,p in probes.items()},
        cost_regeneration=0,target_regeneration=0,bank_regeneration=0,backbone_forwards=0,
        extra_fits=0,extra_seeds=0,paper_edits=0,paper_builds=0,quiet_retries=0)
    assert execution['formal_updates']==12000 and execution['formal_exposures']==192000
    assert execution['evaluation_examples']==execution['final_F_completed']==execution['final_F_attempts']==4608
    assert execution['TRAIN_probe_examples']==512 and execution['TRAIN_probe_F']==0
    receipts['execution']=execution
    receipts['evaluations']={k:dict(path=str(DOC/'results'/f'{k}_seed1_EXECUTION.json'),
        sha256=sha(DOC/'results'/f'{k}_seed1_EXECUTION.json'),execution=v['execution']) for k,v in evaluations.items()}
    receipts['probes']={m:dict(path=str(DOC/f'TRAIN_PROBE_{m}_seed1.json'),
        sha256=sha(DOC/f'TRAIN_PROBE_{m}_seed1.json'),execution=p['execution']) for m,p in probes.items()}
    receipts['completed_stage_code_bindings']={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    write(DOC/'RUN_RECEIPTS.json',receipts)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('prepare','tests','train','evaluate','report','verify','all'),required=True);args=parser.parse_args()
    if args.stage in ('prepare','all'):prepare()
    if args.stage in ('tests','all'):
        from .tests import run_tests
        result=run_tests();receipts=fit_receipts();receipts['tests']=result;write(DOC/'RUN_RECEIPTS.json',receipts)
    if args.stage in ('train','all'):
        from .training import fit
        protocol=read(DOC/'PROTOCOL.json')
        if 'training_code_bindings' not in protocol:
            assert fit_receipts()['tests']['status']=='PASS'
            protocol['training_code_bindings']=code_bindings()
            write(DOC/'PROTOCOL.json',protocol)
        else:assert protocol['training_code_bindings']==code_bindings()
        failures={}; prior=fit_receipts()
        for method in ('LOCAL_CAP','JOINT8'):
            state=OUTPUT/'fits'/f'{method}_seed1'/'ATTEMPT_STATE.json'
            assert not state.exists() or prior['methods'].get(method,{}).get('complete'), 'Incomplete prior fit cannot be silently replayed'
        for method in ('LOCAL_CAP','JOINT8'):
            try:fit(method)
            except Exception as error:
                failures[method]=dict(status='RUN_FAILED',error=repr(error),traceback=traceback.format_exc());r=fit_receipts();r['failures']=failures;write(DOC/'RUN_RECEIPTS.json',r)
                print('JOINT_RUN_FAILED',method,repr(error),flush=True)
        if failures:raise RuntimeError('Failed fixed training attempts: '+','.join(failures))
    if args.stage in ('evaluate','all'):
        from .evaluation import probe,evaluate
        for method in ('LOCAL_CAP','JOINT8'):probe(method);evaluate(method)
        finalize_evaluation_receipts()
    if args.stage in ('report','all'):
        from .reporting import report
        report()
    if args.stage in ('verify','all'):
        from .verification import verify
        verify()


if __name__=='__main__':main()
