"""Implemented stage CLI for the fixed two-loss experiment."""
import argparse
from pathlib import Path
from .common import ROOT,DOC,OUTPUT,BANKS,COST,HARD_DOC,OLD_DOC,read,write,sha,code_bindings


def seal():
    setup=read(DOC/'LOSS_SETUP.json')
    assert setup['status']=='PASS'
    instruction=Path('/home/minjae/Downloads/pallet_cli_quick_soft_expected_6d_20261006.txt')
    old=read(OLD_DOC/'A_protocol.json'); hard=read(HARD_DOC/'PROTOCOL.json')
    protocol=dict(schema='quick_fixed_soft_expected_6d_v1',status='LOCKED_BEFORE_FIRST_FORMAL_UPDATE',
        instruction=dict(path=str(instruction),sha256=sha(instruction),lines=len(instruction.read_text().splitlines()),read_scope='entire'),
        baseline_commit='a22fb14beb5e8df08076385000e0d53503c1ae29',
        base_results_commit='e8e219ab111bc121235c49078459c679d6dc656e',publication_branch='main',
        source_root=str(ROOT),candidate_bank_cache=str(BANKS),pose_cost_cache=str(COST),output_cache=str(OUTPUT),
        methods=['SOFT6D','EXPECT6D'],execution_order=['SOFT6D','EXPECT6D'],seed=1,
        training_initialization='same original random seed1 initialization; no warmstart',
        config=old['config'],parameter_count=20259,action_count=201,NoOp_index=0,
        changed_component='only fixed training loss; native GEO bank/scorer/hardJ/prediction-only finalF unchanged',
        losses=dict(SOFT6D='mean eligible -sum y log softmax(original scores), model temperature1',
                    EXPECT6D='mean eligible sum softmax(original scores)*r_effective, model temperature1'),
        scale_m=setup['scale_m'],tau=setup['tau'],loss_setup_sha256=sha(DOC/'LOSS_SETUP.json'),
        cost='proper-rotation symmetry minimum corresponding canonical8corner ADDsym finalF against SYNTH GT, meters',
        failure='original +inf cache preserved; SOFT target0 but model denominator retains candidate; EXPECT max valid scaled regret+1 approximation',
        code_bindings=code_bindings(),code_binding_scope='all formal-training dependencies sealed; evaluation/reporting separately hashed before execution',
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,betas=[.9,.999],warmup=100,cosine_steps=5900,final_lr=.0001,clip_norm=5),
        numeric=dict(dtype='FP32',TF32_matmul=False,TF32_cudnn=True,cudnn_benchmark=False),
        budgets=dict(max_fits=2,updates_per_fit=6000,batch=16,max_updates=12000,max_train_exposures=192000,
            max_probe_examples=512,max_final_F_calls=4608,cost_regeneration=0,backbone_forwards=0,new_seeds=0),
        checkpoint=dict(every_updates=500,atomic=True,log_every_updates=100,selection='final6000 only'),
        identities=hard['identity_registry'],
        evaluation=dict(SYNTH_HELDOUT=1985,REAL_DEV=319,REAL_sessions=13,independent_real_6d_pairs=0,
            uncertainty='single training seed and repeatedly reused DEV; intervals are frame/session uncertainty, not training-seed variability or independent confirmation'),
        bootstrap=dict(resamples=10000,seed=20260917,REAL_primary='session',SYNTH_primary='frame',SYNTH_secondary='scenario',shared_draws=True),
        decision=dict(ADD_direction_tolerance_m=1e-7,comparison='NEW-minus-baseline paired sameID mean',
            N3_6d_conditions='REAL session ADD mean CI upper<0; REAL T/R medians nonworse; both splits pose failures nonincrease; SYNTH ADD mean nonworse',
            preservation_C='each split failures, RAW canonical good5bad10, fullreference gross20, PCK10, conditional2D pooled median/P90 each nonworse',
            pose_P90='separate B-tail flags, not additional C decision requirements',
            labels=['N3_BEAT_SIGNAL','N3_BEAT_WITH_TRADEOFF','POSE_TRADEOFF_SIGNAL','OLD_ONLY_GAIN','NO_CLEAR_GAIN','BLOCKED_DATA','BLOCKED_INTEGRITY','RUN_FAILED'],
            strict_T_and_R_medians_both_lower='separate explicit boolean'),
        secondary_analyses=['oracle gap and RAW recovery/headroom ratio >1e-7m, unclamped','oracle final W/D hypothesis SWITCH recovery'],
        forbidden=['paper/LaTeX/PDF/bibliography edits or builds','cost regeneration','backbone/detector forwards',
            'architecture/candidates/F changes','temperature/LR/epoch search','extra fits or seeds','GT input/mask at inference','old output overwrite'],
        current_user_state_receipt=str(OUTPUT/'INITIAL_USER_STATE.json'))
    path=DOC/'PROTOCOL.json'
    if path.exists():assert read(path)==protocol, 'Sealed design must not change'
    else:write(path,protocol)
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('setup','seal','train','evaluate','report','all'),required=True)
    args=parser.parse_args()
    if args.stage in ('setup','all'):
        from .setup import prepare
        prepare(ROOT,BANKS,COST,OUTPUT)
    if args.stage in ('seal','train','all'):seal()
    if args.stage in ('train','all'):
        from .training import fit
        import traceback
        # A 500-update checkpoint does not prove that its following interval
        # never began. Reuse completed results, but fail closed on every
        # incomplete prior attempt instead of replaying unknown updates.
        prior=read(DOC/'TRAIN_RECEIPTS.json') if (DOC/'TRAIN_RECEIPTS.json').exists() else {'methods':{}}
        for method in ('SOFT6D','EXPECT6D'):
            state=OUTPUT/'fits'/f'{method}_seed1'/'ATTEMPT_STATE.json'
            if state.exists() and not prior['methods'].get(method,{}).get('complete',False):
                raise RuntimeError('RUN_FAILED: incomplete prior attempt cannot be silently resumed: '+method)
        failures={}
        for method in ('SOFT6D','EXPECT6D'):
            try:fit(method)
            except Exception as error:
                state=OUTPUT/'fits'/f'{method}_seed1'/'ATTEMPT_STATE.json'
                failures[method]=dict(status='RUN_FAILED',complete=False,error=repr(error),
                    traceback=traceback.format_exc(),last_recorded_state=read(state) if state.exists() else None,
                    no_silent_restart=True)
                write(DOC/'FAILED_RUNS.json',failures)
                print('QUICK_RUN_FAILED',method,repr(error),flush=True)
                # The second fixed loss is attempted even if the first fails.
                # Its own immutable-input checks reject any shared corruption.
        if failures:raise RuntimeError('Fixed training attempts failed: '+','.join(failures))
    if args.stage in ('evaluate','all'):
        from .evaluation import evaluate,probe
        for method in ('SOFT6D','EXPECT6D'):
            probe(method)
            evaluate(method)
    if args.stage in ('report','all'):
        from .reporting import report
        report()


if __name__=='__main__':main()
